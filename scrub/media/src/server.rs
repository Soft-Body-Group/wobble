//! The frame server: decode workers that keep the frames around the
//! playhead in the cache.
//!
//! The app says where the playhead is (`set`) every redraw. The server
//! turns that into a WANTED list in priority order: the playhead frame,
//! then the frames it will show next in the play direction (stepping at
//! the shuttle speed), or a few either side when paused, for frame steps.
//! Whatever is wanted, not cached and not in flight is QUEUED; whatever
//! was queued and is no longer wanted is simply dropped from the queue,
//! which is the whole of cancellation: a scrub never waits behind decodes
//! for frames the hand has left. Work already running finishes and lands
//! in the cache (it is a frame someone may come back to).
//!
//! Two ways to decode, picked by the codec:
//! - **Intra** (ProRes, DNxHR, image files): any frame from any worker,
//!   several at once.
//! - **Long-GOP** (H.264): a worker that starts on a frame STREAMS on,
//!   taking the next wanted frame after it on the same warm decoder, and
//!   reserves the frames just ahead of it so no other worker seeks into
//!   the GOP it is already decoding. Frames decoded on the way to a target
//!   are kept when wanted, which is what makes reverse play affordable.

use crate::cache::{Cache, CacheStats};
use crate::frame::{Frame, FrameRef};
use crate::movie::MAX_CATCHUP;
use crate::source::{Sink, SourceRef};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

/// Most frames prefetched ahead of the playhead while playing.
const AHEAD_MAX: u64 = 48;
/// Frames either side of a paused playhead, for stepping.
const PAUSED_RADIUS: u64 = 3;
/// Frames past the hand, in the drag's direction, while scrubbing.
const SCRUB_AHEAD: u64 = 2;
/// How far `pick` walks from the target toward the picture.
const PICK_SPAN: u64 = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlayState {
    /// The playhead, as an index into the source.
    pub frame: u64,
    /// +1 forward, -1 reverse.
    pub dir: i8,
    /// Frames between successive shown frames (2 at 2x shuttle).
    pub step: u64,
    pub playing: bool,
    /// The hand is dragging the playhead: only the frame under it (and a
    /// step or two ahead in the drag's direction) is worth decoding.
    pub scrubbing: bool,
    /// Prefetch wraps past the ends (looped playback).
    pub looping: bool,
}

impl PlayState {
    pub fn paused(frame: u64) -> Self {
        Self { frame, dir: 1, step: 1, playing: false, scrubbing: false, looping: false }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub cache: CacheStats,
    pub decoded: u64,
    pub errors: u64,
    pub last_error: Option<String>,
    /// Mean decode time of the last frames, ms.
    pub decode_ms: f32,
    pub queued: usize,
    pub in_flight: usize,
}

struct Sched {
    wanted: HashSet<u64>,
    /// Long-GOP scrub: frames worth keeping if a decode passes through
    /// them on its way to the hand (never queued on their own). The first
    /// is the keyframe a seek lands on: a picture near the hand in one
    /// frame's decode, while the exact one is still being reached.
    keep: HashSet<u64>,
    queue: VecDeque<u64>,
    in_flight: HashSet<u64>,
    /// Long-GOP streams: worker -> the frame it is decoding toward.
    streams: HashMap<usize, u64>,
    /// A paused playhead whose frame is not in yet: while it decodes no
    /// other work starts, so a seek has every core (measured on 4K
    /// ProRes: 48 ms sharing with its neighbours, a seventh of it alone).
    gate: Option<u64>,
    quit: bool,
    errors: u64,
    last_error: Option<String>,
}

struct Shared {
    id: u64,
    source: SourceRef,
    cache: Arc<Cache>,
    sched: Mutex<Sched>,
    work: Condvar,
    /// Notified whenever a frame lands (`wait_for`).
    landed: Condvar,
    decoded: AtomicU64,
    /// Decode ms of recent frames, as an exponential mean (f32 bits).
    decode_ms: AtomicU64,
}

pub struct FrameServer {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
}

/// A cache key for a file: FNV-1a of its path (with the frame count, so a
/// sequence that grew is a new key).
fn file_id(path: &std::path::Path) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for b in path.to_string_lossy().bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

impl FrameServer {
    pub fn new(source: SourceRef, cache: Arc<Cache>) -> Self {
        let cores = crate::sys::cores();
        let n = crate::sys::var("WORKERS").ok().and_then(|s| s.parse().ok()).unwrap_or(
            // Intra decodes are spread over workers; a long-GOP clip is
            // mostly one stream (FFmpeg threads inside it), plus one
            // spare for a jump elsewhere.
            if source.info().intra { (cores / 2).clamp(2, 8) } else { 2 },
        );
        Self::with_workers(source, cache, n)
    }

    pub fn with_workers(source: SourceRef, cache: Arc<Cache>, workers: usize) -> Self {
        // Cache keys are by FILE, so two servers on one file (the source
        // viewer and a timeline clip) share every decoded frame.
        let id = file_id(&source.info().path);
        let shared = Arc::new(Shared {
            id,
            source,
            cache,
            sched: Mutex::new(Sched {
                wanted: HashSet::new(),
                keep: HashSet::new(),
                queue: VecDeque::new(),
                in_flight: HashSet::new(),
                streams: HashMap::new(),
                gate: None,
                quit: false,
                errors: 0,
                last_error: None,
            }),
            work: Condvar::new(),
            landed: Condvar::new(),
            decoded: AtomicU64::new(0),
            decode_ms: AtomicU64::new(0),
        });
        let workers = (0..workers.max(1))
            .map(|i| {
                let s = shared.clone();
                std::thread::Builder::new()
                    .name(format!("decode-{i}"))
                    .spawn(move || worker(s, i))
                    .expect("spawn decode worker")
            })
            .collect();
        Self { shared, workers }
    }

    pub fn source(&self) -> &SourceRef {
        &self.shared.source
    }

    pub fn id(&self) -> u64 {
        self.shared.id
    }

    pub fn workers(&self) -> usize {
        self.workers.len()
    }

    /// Tell the server where the playhead is. Cheap when nothing changed.
    pub fn set(&self, st: PlayState) {
        let frames = self.shared.source.info().frames;
        if frames == 0 {
            return;
        }
        let wanted = wanted_list(st, frames, self.ahead());
        let keep = if st.scrubbing && !self.shared.source.info().intra {
            st.frame.saturating_sub(MAX_CATCHUP)..st.frame
        } else {
            0..0
        };
        let gate = (!st.playing).then_some(st.frame);
        self.set_list(&wanted, gate, keep);
    }

    /// Say exactly which frames to have, most urgent first (the record
    /// viewer's prefetch runs across cuts, so each clip's server gets the
    /// frames of its own stretch). `gate`: a paused frame to decode alone
    /// before anything else starts. `keep`: frames to keep if a long-GOP
    /// decode passes through them. An empty list cancels all queued work.
    pub fn set_list(&self, wanted: &[u64], gate: Option<u64>, keep: std::ops::Range<u64>) {
        let sh = &self.shared;
        sh.source.prefetch(wanted, gate.is_none());
        sh.cache.touch(wanted.iter().map(|&f| (sh.id, f)));
        let mut g = sh.sched.lock().unwrap();
        g.wanted = wanted.iter().copied().collect();
        g.keep = keep.collect();
        g.gate = gate.filter(|&f| !sh.cache.contains((sh.id, f)));
        g.queue.clear();
        for &f in wanted {
            if f < sh.source.info().frames && !g.in_flight.contains(&f) && !sh.cache.contains((sh.id, f)) && !g.queue.contains(&f) {
                g.queue.push_back(f);
            }
        }
        if !g.queue.is_empty() {
            sh.work.notify_all();
        }
    }

    /// How many frames to prefetch: AHEAD_MAX, or a quarter of the cache.
    pub fn ahead(&self) -> u64 {
        let i = self.shared.source.info();
        let est = (i.width as u64 * i.height as u64 * 2).max(1);
        (self.shared.cache.stats().budget as u64 / 4 / est).clamp(4, AHEAD_MAX)
    }

    pub fn get(&self, frame: u64) -> Option<FrameRef> {
        self.shared.cache.get((self.shared.id, frame))
    }

    /// The frame to put on screen for a playhead at `target` when the
    /// picture is `shown`: the target if cached; else the cached frame
    /// nearest the target between the picture and the target (the
    /// picture may lag the hand, never run past it or backwards); else,
    /// for a long-GOP clip, the cached frame nearest below the target in
    /// its GOP (a keyframe a scrub's decode passed), when nearer than the
    /// picture. `None`: keep what is shown.
    pub fn pick(&self, target: u64, shown: Option<u64>) -> Option<(u64, FrameRef)> {
        if let Some(f) = self.get(target) {
            return Some((target, f));
        }
        let shown = shown?;
        let mut f = target;
        for _ in 0..PICK_SPAN {
            if f == shown {
                break;
            }
            f = if shown < f { f - 1 } else { f + 1 };
            if let Some(fr) = self.get(f) {
                return Some((f, fr));
            }
        }
        if self.shared.source.info().intra {
            return None;
        }
        let far = shown.abs_diff(target);
        (1..MAX_CATCHUP.min(far))
            .filter_map(|d| target.checked_sub(d))
            .find_map(|f| self.get(f).map(|fr| (f, fr)))
    }

    pub fn is_cached(&self, frame: u64) -> bool {
        self.shared.cache.contains((self.shared.id, frame))
    }

    pub fn cached_frames(&self) -> Vec<u64> {
        self.shared.cache.frames_of(self.shared.id)
    }

    /// Block until `frame` is cached or `timeout` passes.
    pub fn wait_for(&self, frame: u64, timeout: Duration) -> Option<FrameRef> {
        let deadline = std::time::Instant::now() + timeout;
        let mut g = self.shared.sched.lock().unwrap();
        loop {
            if let Some(f) = self.get(frame) {
                return Some(f);
            }
            let now = std::time::Instant::now();
            if now >= deadline {
                return None;
            }
            g = self.shared.landed.wait_timeout(g, deadline - now).unwrap().0;
        }
    }

    pub fn stats(&self) -> Stats {
        let sh = &self.shared;
        let g = sh.sched.lock().unwrap();
        Stats {
            cache: sh.cache.stats(),
            decoded: sh.decoded.load(Ordering::Relaxed),
            errors: g.errors,
            last_error: g.last_error.clone(),
            decode_ms: f32::from_bits(sh.decode_ms.load(Ordering::Relaxed) as u32),
            queued: g.queue.len(),
            in_flight: g.in_flight.len(),
        }
    }
}

impl Drop for FrameServer {
    fn drop(&mut self) {
        self.shared.sched.lock().unwrap().quit = true;
        self.shared.work.notify_all();
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
        // The frames stay in the cache: another server on the same file
        // may want them, and the LRU retires them otherwise.
    }
}

/// The frames to have, most urgent first.
pub fn wanted_list(st: PlayState, frames: u64, ahead: u64) -> Vec<u64> {
    let at = |off: i64| -> Option<u64> {
        let f = st.frame as i64 + off;
        if st.looping {
            Some(f.rem_euclid(frames as i64) as u64)
        } else {
            (0..frames as i64).contains(&f).then_some(f as u64)
        }
    };
    let mut out = Vec::new();
    let mut push = |f: Option<u64>| {
        if let Some(f) = f {
            if !out.contains(&f) {
                out.push(f);
            }
        }
    };
    push(at(0));
    if st.playing {
        let d = st.dir.signum() as i64 * st.step.max(1) as i64;
        for k in 1..=ahead as i64 {
            push(at(k * d));
        }
    } else if st.scrubbing {
        let d = st.dir.signum() as i64 * st.step.max(1) as i64;
        for k in 1..=SCRUB_AHEAD as i64 {
            push(at(k * d));
        }
    } else {
        for k in 1..=PAUSED_RADIUS as i64 {
            push(at(k * st.dir.signum() as i64));
            push(at(-k * st.dir.signum() as i64));
        }
    }
    out
}

fn worker(sh: Arc<Shared>, me: usize) {
    let intra = sh.source.info().intra;
    loop {
        let first = {
            let mut g = sh.sched.lock().unwrap();
            loop {
                if g.quit {
                    return;
                }
                if let Some(f) = claim(&mut g, intra, None) {
                    g.streams.insert(me, f);
                    break f;
                }
                g = sh.work.wait(g).unwrap();
            }
        };
        let mut sink = WorkerSink { sh: &sh, me, intra };
        sh.source.stream(first, &mut sink);
        sh.sched.lock().unwrap().streams.remove(&me);
    }
}

/// Take a queued frame for a worker: the most urgent one, or for a stream
/// continuing from `after`, the nearest one past it. Long-GOP: never a
/// frame just ahead of another worker's stream.
fn claim(g: &mut Sched, intra: bool, after: Option<u64>) -> Option<u64> {
    if let Some(gate) = g.gate {
        if g.in_flight.contains(&gate) {
            return None;
        }
        // The gated frame first, whoever asks.
        if let Some(i) = g.queue.iter().position(|&f| f == gate) {
            g.queue.remove(i);
            g.in_flight.insert(gate);
            return Some(gate);
        }
    }
    let i = if intra {
        if g.queue.is_empty() {
            return None;
        }
        0
    } else if let Some(a) = after {
        g.queue.iter().enumerate().filter(|(_, &f)| f > a && f - a <= MAX_CATCHUP).min_by_key(|(_, &f)| f)?.0
    } else {
        let reserved = |f: u64| g.streams.values().any(|&s| f >= s && f - s <= MAX_CATCHUP);
        g.queue.iter().position(|&f| !reserved(f))?
    };
    let f = g.queue.remove(i)?;
    g.in_flight.insert(f);
    Some(f)
}

fn ungate(g: &mut Sched, frame: u64, sh: &Shared) {
    if g.gate == Some(frame) {
        g.gate = None;
        sh.work.notify_all();
    }
}

struct WorkerSink<'a> {
    sh: &'a Shared,
    me: usize,
    intra: bool,
}

impl WorkerSink<'_> {
    fn land(&self, frame: u64, picture: Frame) {
        let ms = picture.decode_ms;
        self.sh.cache.insert((self.sh.id, frame), Arc::new(picture));
        self.sh.decoded.fetch_add(1, Ordering::Relaxed);
        let old = f32::from_bits(self.sh.decode_ms.load(Ordering::Relaxed) as u32);
        let mean = if old == 0.0 { ms } else { old * 0.9 + ms * 0.1 };
        self.sh.decode_ms.store(mean.to_bits() as u64, Ordering::Relaxed);
    }
}

impl Sink for WorkerSink<'_> {
    fn wants(&mut self, frame: u64) -> bool {
        let g = self.sh.sched.lock().unwrap();
        (g.wanted.contains(&frame) || g.keep.contains(&frame)) && !g.in_flight.contains(&frame) && !self.sh.cache.contains((self.sh.id, frame))
    }

    fn passing(&mut self, frame: u64, picture: Frame) {
        self.land(frame, picture);
        let mut g = self.sh.sched.lock().unwrap();
        g.queue.retain(|&f| f != frame);
        ungate(&mut g, frame, self.sh);
        self.sh.landed.notify_all();
    }

    fn done(&mut self, frame: u64, picture: anyhow::Result<Frame>) -> Option<u64> {
        let ok = match picture {
            Ok(p) => {
                self.land(frame, p);
                None
            }
            Err(e) => Some(format!("frame {frame}: {e:#}")),
        };
        let mut g = self.sh.sched.lock().unwrap();
        g.in_flight.remove(&frame);
        ungate(&mut g, frame, self.sh);
        if let Some(e) = ok {
            log::debug!("{e}");
            g.errors += 1;
            g.last_error = Some(e);
        }
        self.sh.landed.notify_all();
        if g.quit {
            return None;
        }
        let next = claim(&mut g, self.intra, Some(frame));
        if let Some(n) = next {
            g.streams.insert(self.me, n);
        }
        next
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn playing_wants_the_playhead_then_frames_ahead_at_the_shuttle_step() {
        let st = PlayState { frame: 10, dir: 1, step: 2, playing: true, scrubbing: false, looping: false };
        assert_eq!(wanted_list(st, 100, 3), vec![10, 12, 14, 16]);
        let back = PlayState { dir: -1, ..st };
        assert_eq!(wanted_list(back, 100, 3), vec![10, 8, 6, 4]);
    }

    #[test]
    fn paused_wants_neighbours_for_stepping_and_stops_at_the_ends() {
        assert_eq!(wanted_list(PlayState::paused(1), 100, 8), vec![1, 2, 0, 3, 4]);
    }

    #[test]
    fn looping_prefetch_wraps_to_the_start() {
        let st = PlayState { frame: 98, dir: 1, step: 1, playing: true, scrubbing: false, looping: true };
        assert_eq!(wanted_list(st, 100, 3), vec![98, 99, 0, 1]);
    }
}
