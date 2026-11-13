//! Sequence playback: rayon decode pool feeding the staging ring, frames
//! decoded N ahead of display, in-order delivery to the render thread.
//!
//! Deadlock freedom: decode jobs are only spawned once a staging slot is
//! already acquired (render thread, `try_acquire`), and jobs are issued in
//! strict (tick, reader) order — so the slots in flight are always the ones
//! the display needs next.

use crate::decode::{ExrDecoder, ImageInfo, Pack};
use crate::diskcache::DiskCache;
use crate::graph::{ReadEdge, ReadEdges, TimeMap};
use crate::staging::StagingRing;
use crate::timing::RollingAvg;
use anyhow::Result;
use crossbeam_channel::{Receiver, Sender};
use half::f16;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use web_time::Instant;

/// Decoded-frame LRU, bounded by a byte budget. Frames are stored with the
/// staging row stride so a cache hit is one straight memcpy into the mapped
/// slot. (The brief said "no caching beyond the frame ring" — this exists
/// because real media at real compression rates decodes at 100–400 ms/frame,
/// and looped playback/scrubbing is unusable without it. Set budget 0 to
/// get the honest decode-every-frame behavior; the bench does.)
pub struct FrameCache {
    map: HashMap<(usize, u64), CacheEntry>,
    total: usize,
    budget: usize,
    clock: u64,
    /// Cache lookups per timeline loop (readers × loop length). An entry hit
    /// within the last ~1.5 loops is "in rotation" and never evicted, no
    /// matter how small the resident set is relative to the loop.
    cycle_lookups: u64,
    pub hits: u64,
    pub misses: u64,
}

struct CacheEntry {
    data: Arc<Vec<f16>>,
    bytes: usize,
    last_used: u64,
    /// The file's size and mtime when this was decoded (`file_stamp`).
    stamp: Stamp,
}

/// A file's size and modified time (ns): what tells a frame decoded from
/// it apart from the file a renderer has since written over it. `None`
/// where there is nothing to ask (a remote path, a page).
pub type Stamp = Option<(u64, u64)>;

pub fn file_stamp(path: &Path) -> Stamp {
    let m = std::fs::metadata(path).ok()?;
    let t = m.modified().ok()?.duration_since(std::time::UNIX_EPOCH).ok()?;
    Some((m.len(), t.as_nanos() as u64))
}

/// Both known and different: the file changed under the frame.
fn stamps_differ(a: Stamp, b: Stamp) -> bool {
    matches!((a, b), (Some(x), Some(y)) if x != y)
}

/// What a decoded frame IS, whoever asked for it: file, frame in file,
/// layer, proxy divisor.
type ContentKey = (PathBuf, u64, Option<String>, u32, Stamp);

/// The last few FULL decodes, shared by every reader that wants a part of
/// them. One plate cut into four sections is four readers, and each used to
/// decode the whole file for its own box: four 150 ms decodes of the same
/// 8K frame per tick. Here the first reader to ask decodes, anyone asking
/// while it runs waits for that one decode, and the frame stays for the
/// readers a few frames behind it (a slipped layer).
struct SharedDecodes {
    map: HashMap<ContentKey, (Arc<Mutex<Option<Arc<Vec<f16>>>>>, u64, usize)>,
    clock: u64,
    /// Buffers of frames that left the store, for the next decode. A fresh
    /// 265 MB allocation is page-faulted in from nothing every time, which
    /// costs more than the decode it is for (8K ZIP: 374 ms a frame fresh,
    /// 185 ms into a buffer that is already there).
    free: Vec<Vec<f16>>,
}

/// A full-frame buffer of `len` halfs, zeroed: a recycled one when the
/// store has it.
fn scratch_frame(len: usize) -> Vec<f16> {
    use rayon::prelude::*;
    let got = {
        let mut s = shared_decodes().lock().unwrap_or_else(|e| e.into_inner());
        s.free.iter().position(|b| b.len() == len).map(|i| s.free.swap_remove(i))
    };
    match got {
        Some(mut b) => {
            // Zeroed like a fresh one: a decoder need not write every pixel.
            b.par_chunks_mut(1 << 20).for_each(|c| c.fill(f16::ZERO));
            b
        }
        None => vec![f16::ZERO; len],
    }
}

/// Frames this big queue for a decode rather than all starting at once.
const BIG_FRAME_BYTES: usize = crate::decode::PARALLEL_FRAME_BYTES as usize;
/// How many big decodes run at once. Measured on 8K ZIP, 32 threads: 4 in
/// flight is the most frames a second (10.8); 8 is fewer (9.2) at more
/// than twice the wait for each, 16 fewer again (7.3).
const BIG_DECODES: usize = 4;

/// Held while a big frame decodes on a job thread (see `BIG_DECODES`).
struct BigDecode;

fn big_gate() -> &'static (Mutex<usize>, std::sync::Condvar) {
    static G: std::sync::OnceLock<(Mutex<usize>, std::sync::Condvar)> = std::sync::OnceLock::new();
    G.get_or_init(|| (Mutex::new(0), std::sync::Condvar::new()))
}

impl BigDecode {
    /// Waits only on a job thread (a rayon worker must never block here).
    fn enter(bytes: usize) -> Option<BigDecode> {
        if bytes < BIG_FRAME_BYTES || !ON_JOB_THREAD.with(|j| j.get()) {
            return None;
        }
        let (m, cv) = big_gate();
        let mut n = m.lock().unwrap_or_else(|e| e.into_inner());
        while *n >= BIG_DECODES {
            n = cv.wait(n).unwrap_or_else(|e| e.into_inner());
        }
        *n += 1;
        Some(BigDecode)
    }
}

impl Drop for BigDecode {
    fn drop(&mut self) {
        let (m, cv) = big_gate();
        *m.lock().unwrap_or_else(|e| e.into_inner()) -= 1;
        cv.notify_one();
    }
}

/// Real decodes run (not cache hits, not shared frames) and the
/// milliseconds they took: what a stress run reads to tell a decode-bound
/// pass from a cache-bound one.
pub static FULL_DECODES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static FULL_DECODE_MS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn note_decode(t0: Instant) {
    use std::sync::atomic::Ordering::Relaxed;
    FULL_DECODES.fetch_add(1, Relaxed);
    FULL_DECODE_MS.fetch_add(t0.elapsed().as_millis() as u64, Relaxed);
}

thread_local! {
    static ON_JOB_THREAD: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Run a decode job. Natively on a small pool of PLAIN threads, not on
/// rayon: a job forks into rayon (the decoder, the cache unpack), and a
/// rayon worker waiting on its own fork steals other queued jobs and runs
/// them nested — one 150 ms decode of an 8K frame measured a full second
/// that way, stuck under the three it had picked up. A plain thread just
/// waits. (A page has no threads: rayon's one-thread pool, as before.)
fn spawn_job(f: impl FnOnce() + Send + 'static) {
    #[cfg(target_arch = "wasm32")]
    rayon::spawn(f);
    #[cfg(not(target_arch = "wasm32"))]
    {
        type Job = Box<dyn FnOnce() + Send>;
        static TX: std::sync::OnceLock<Sender<Job>> = std::sync::OnceLock::new();
        let tx = TX.get_or_init(|| {
            let (tx, rx) = crossbeam_channel::unbounded::<Job>();
            let n = std::thread::available_parallelism().map_or(8, |n| n.get()).clamp(4, 16);
            for i in 0..n {
                let rx = rx.clone();
                let _ = std::thread::Builder::new().name(format!("decode-job-{i}")).spawn(move || {
                    ON_JOB_THREAD.with(|j| j.set(true));
                    for job in rx {
                        job();
                    }
                });
            }
            tx
        });
        let _ = tx.send(Box::new(f));
    }
}

/// Queue an NVMe cache write. ONE thread writes, after the frame has been
/// handed over: a compress + write a decoded frame, run wherever the decode
/// ran, took the cores the next decodes wanted (a first pass over an 8K
/// plate in four sections measured twice as long with the tier on). A
/// backlog past `WRITE_BACKLOG` bytes drops the write — the frame is cached
/// the next time it is decoded, and a slow disk never holds more than that
/// in memory on the queue's account.
fn queue_cache_write(d: Arc<DiskCache>, file: PathBuf, info: ImageInfo, data: Arc<Vec<f16>>) {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    const WRITE_BACKLOG: u64 = 4 << 30;
    static QUEUED: AtomicU64 = AtomicU64::new(0);
    let bytes = data.len() as u64 * 2;
    if QUEUED.load(Relaxed) + bytes > WRITE_BACKLOG {
        return;
    }
    QUEUED.fetch_add(bytes, Relaxed);
    let write = move || {
        if let Err(e) = d.write_frame(&file, info, &data) {
            log::warn!("disk cache write {}: {e:#}", file.display());
        }
        QUEUED.fetch_sub(bytes, Relaxed);
    };
    #[cfg(target_arch = "wasm32")]
    write();
    #[cfg(not(target_arch = "wasm32"))]
    {
        type Job = Box<dyn FnOnce() + Send>;
        static TX: std::sync::OnceLock<Sender<Job>> = std::sync::OnceLock::new();
        let tx = TX.get_or_init(|| {
            let (tx, rx) = crossbeam_channel::unbounded::<Job>();
            let _ = std::thread::Builder::new().name("cache-write".into()).spawn(move || {
                for job in rx {
                    job();
                }
            });
            tx
        });
        let _ = tx.send(Box::new(write));
    }
}

/// Bytes of full frames kept (never fewer than two frames).
const SHARED_BUDGET: usize = 1 << 30;

fn shared_decodes() -> &'static Mutex<SharedDecodes> {
    static S: std::sync::OnceLock<Mutex<SharedDecodes>> = std::sync::OnceLock::new();
    S.get_or_init(|| Mutex::new(SharedDecodes { map: HashMap::new(), clock: 0, free: Vec::new() }))
}

/// The full frame for `key`: from the shared store, from the decode
/// another thread is running right now, or from `decode`.
fn shared_full(
    key: &ContentKey,
    decode: impl FnOnce() -> Result<Vec<f16>>,
) -> Result<Arc<Vec<f16>>> {
    let cell = {
        let mut s = shared_decodes().lock().unwrap_or_else(|e| e.into_inner());
        s.clock += 1;
        let clock = s.clock;
        let e = s.map.entry(key.clone()).or_insert_with(|| (Arc::new(Mutex::new(None)), clock, 0));
        e.1 = clock;
        e.0.clone()
    };
    // Waiting for another thread's decode is only safe on a JOB thread. A
    // rayon worker that blocks here may be the very thread that holds the
    // lock: the decoder forks into rayon, and a worker waiting on that fork
    // steals queued tasks — another decode of the same frame among them.
    // Anywhere else, a decode already running means decode it again.
    let mut slot = match cell.try_lock() {
        Ok(g) => g,
        Err(std::sync::TryLockError::Poisoned(e)) => e.into_inner(),
        Err(std::sync::TryLockError::WouldBlock) if ON_JOB_THREAD.with(|j| j.get()) => {
            cell.lock().unwrap_or_else(|e| e.into_inner())
        }
        Err(std::sync::TryLockError::WouldBlock) => return decode().map(Arc::new),
    };
    if let Some(d) = &*slot {
        return Ok(d.clone());
    }
    match decode() {
        Ok(v) => {
            let bytes = v.len() * 2;
            let a = Arc::new(v);
            *slot = Some(a.clone());
            drop(slot);
            let mut s = shared_decodes().lock().unwrap_or_else(|e| e.into_inner());
            if let Some(e) = s.map.get_mut(key) {
                e.2 = bytes;
            }
            while s.map.len() > 2 && s.map.values().map(|e| e.2).sum::<usize>() > SHARED_BUDGET {
                let Some(old) = s.map.iter().min_by_key(|(_, e)| e.1).map(|(k, _)| k.clone()) else { break };
                // Nobody else holds it: its buffer is the next decode's.
                let buf = s
                    .map
                    .remove(&old)
                    .and_then(|e| e.0.try_lock().ok().and_then(|mut g| g.take()))
                    .and_then(|a| Arc::try_unwrap(a).ok());
                if let Some(b) = buf {
                    if s.free.len() < 2 {
                        s.free.push(b);
                    }
                }
            }
            Ok(a)
        }
        Err(e) => {
            drop(slot);
            // Not kept: the file may be one the renderer is still writing.
            shared_decodes().lock().unwrap_or_else(|e| e.into_inner()).map.remove(key);
            Err(e)
        }
    }
}

impl FrameCache {
    fn has(&self, key: (usize, u64)) -> bool {
        self.map.contains_key(&key)
    }

    fn new(budget: usize) -> Self {
        Self {
            map: HashMap::new(),
            total: 0,
            budget,
            clock: 0,
            cycle_lookups: 0,
            hits: 0,
            misses: 0,
        }
    }

    /// Drop one frame, because the file it was decoded from changed.
    fn remove(&mut self, key: (usize, u64)) {
        if let Some(e) = self.map.remove(&key) {
            self.total -= e.bytes;
        }
    }

    /// `get`, unless the file has changed since the entry was decoded: a
    /// stale entry is dropped and the answer is a miss, so a re-rendered
    /// frame is decoded again the next time anything asks for it.
    fn get_fresh(&mut self, key: (usize, u64), now: Stamp) -> Option<Arc<Vec<f16>>> {
        if self.map.get(&key).is_some_and(|e| stamps_differ(e.stamp, now)) {
            self.remove(key);
        }
        self.get(key)
    }

    fn insert_stamped(&mut self, key: (usize, u64), data: Arc<Vec<f16>>, stamp: Stamp) {
        self.insert(key, data);
        if let Some(e) = self.map.get_mut(&key) {
            e.stamp = stamp;
        }
    }

    fn get(&mut self, key: (usize, u64)) -> Option<Arc<Vec<f16>>> {
        self.clock += 1;
        let clock = self.clock;
        match self.map.get_mut(&key) {
            Some(e) => {
                e.last_used = clock;
                self.hits += 1;
                Some(e.data.clone())
            }
            None => {
                self.misses += 1;
                None
            }
        }
    }

    fn insert(&mut self, key: (usize, u64), data: Arc<Vec<f16>>) {
        let bytes = data.len() * 2;
        if bytes > self.budget {
            return;
        }
        // Make room first — but never evict entries that are still in active
        // rotation. Plain LRU collapses to 0% hits on looped playback when
        // the loop is bigger than the budget (each pass evicts exactly what
        // the next pass needs); refusing to evict fresh entries pins a
        // resident prefix instead (hit rate = budget/loop). "In rotation"
        // means hit within the last ~1.5 loops of lookups (the player knows
        // the loop period); entries idle longer than that — e.g. after
        // seeking to a different part of a long timeline — are fair game.
        let stale_after = if self.cycle_lookups > 0 {
            self.cycle_lookups + self.cycle_lookups / 2
        } else {
            (3 * self.map.len() as u64).max(16)
        };
        while self.total + bytes > self.budget {
            let Some((&k, e)) = self.map.iter().min_by_key(|(_, e)| e.last_used) else {
                return;
            };
            if self.clock.saturating_sub(e.last_used) < stale_after {
                return; // all residents are hot: skip caching this frame
            }
            let e = self.map.remove(&k).unwrap();
            self.total -= e.bytes;
        }
        self.clock += 1;
        if let Some(old) = self.map.insert(
            key,
            CacheEntry { data, bytes, last_used: self.clock, stamp: None },
        ) {
            self.total -= old.bytes;
        }
        self.total += bytes;
    }

    pub fn stats(&self) -> (usize, usize, u64, u64) {
        (self.total, self.map.len(), self.hits, self.misses)
    }

    /// Which of a reader's frames are resident, indexed by sequence frame.
    /// The UI draws this as an occupancy strip per Read node.
    pub fn residency(&self, reader: usize, frames: usize) -> Vec<bool> {
        let mut out = vec![false; frames];
        for (r, f) in self.map.keys() {
            if *r == reader && (*f as usize) < frames {
                out[*f as usize] = true;
            }
        }
        out
    }

    /// Forget every frame of one reader.
    pub fn drop_reader(&mut self, reader: usize) {
        let mut freed = 0;
        self.map.retain(|(r, _), e| {
            let keep = *r != reader;
            if !keep {
                freed += e.bytes;
            }
            keep
        });
        self.total -= freed;
    }

    /// Bytes this reader holds.
    pub fn reader_bytes(&self, reader: usize) -> u64 {
        self.map
            .iter()
            .filter(|((r, _), _)| *r == reader)
            .map(|(_, e)| e.bytes as u64)
            .sum()
    }
}

/// Which cache tiers one reader may use (mirrors the Read node's toggles).
#[derive(Clone, Copy, Debug)]
pub struct ReaderCacheOpts {
    pub ram: bool,
    pub disk: bool,
    /// Playback may WRITE what it decodes into the tiers (the Load's "cache
    /// on play"). Off, the tiers are still read; only a precache fills them.
    pub write: bool,
}

impl Default for ReaderCacheOpts {
    fn default() -> Self {
        // RAM opt-in per reader; NVMe is the standing tier.
        Self { ram: false, disk: true, write: true }
    }
}

/// The path a numbered frame WOULD have, `by` frames later.
///
/// Advances the last run of digits in the file name, which is how every
/// sequence this reads is named (`plate.0023.exr` -> `plate.0024.exr`),
/// keeping the zero padding. `None` when there is no number to advance.
pub fn bump_frame_number(path: &Path, by: u64) -> Option<PathBuf> {
    renumber_frame(path, by as i64)
}

/// `bump_frame_number` in either direction: the path a numbered frame WOULD
/// have `by` frames later (or earlier, negative). `None` when there is no
/// number, or the number would go below zero.
pub fn renumber_frame(path: &Path, by: i64) -> Option<PathBuf> {
    let name = path.file_name()?.to_str()?;
    let end = name.rfind(|c: char| c.is_ascii_digit())? + 1;
    let start = name[..end]
        .rfind(|c: char| !c.is_ascii_digit())
        .map_or(0, |i| i + 1);
    let digits = &name[start..end];
    let width = digits.len();
    let next = u64::try_from(digits.parse::<i64>().ok()?.checked_add(by)?).ok()?;
    Some(path.with_file_name(format!(
        "{}{:0width$}{}",
        &name[..start],
        next,
        &name[end..],
    )))
}

pub struct Sequence {
    /// One path per timeline frame. For a MOVIE every entry is the same
    /// path — the container holds the frames, and the index into this list
    /// is the frame number handed to `decode_frame_into`. That is the whole
    /// trick: time maps, residency, the staging ring and the caches all
    /// keep working on a movie without knowing one exists.
    pub frames: Vec<PathBuf>,
    pub info: ImageInfo,
    /// Proxy divisor `info` was scaled by (1 = full res). Folded into the
    /// NVMe frame key so half-res entries never fight the full-res ones.
    pub proxy: u32,
    /// True when `frames` is one container repeated, rather than a file
    /// per frame.
    pub movie: bool,
    /// Frame index, INSIDE the container, of `frames[0]`. Non-zero only
    /// for a movie that has been trimmed: slicing the path list cannot
    /// carry the frame numbers with it the way slicing a list of DISTINCT
    /// paths does, so the offset has to ride alongside — otherwise a
    /// trimmed clip replays from its start while claiming to begin at `a`.
    pub offset: u64,
    /// How the frames are stored from the decode on (single channel,
    /// 8-bit). Set by the app from the Load; the default everywhere else.
    pub pack: Pack,
}

impl Sequence {
    pub fn from_dir(dir: &Path, decoder: &dyn ExrDecoder) -> Result<Self> {
        let mut frames: Vec<PathBuf> = crate::vfs::list(dir)?
            .into_iter()
            // A movie in the directory is its OWN clip, not a frame of a
            // still sequence; loading one is a different call.
            .filter(|p| crate::decode::is_supported(p) && !crate::decode::is_movie(p))
            .collect();
        anyhow::ensure!(
            !frames.is_empty(),
            "no readable images in {} (looking for {})",
            dir.display(),
            crate::decode::SUPPORTED_EXTS.join(", ")
        );
        frames.sort();
        let info = decoder.read_header(&frames[0])?;
        Ok(Self { frames, info, movie: false, offset: 0, proxy: 1, pack: Pack::default() })
    }

    /// Build from a MOVIE: one container, its own declared length. The
    /// path is repeated once per frame so everything downstream indexes it
    /// exactly as it indexes a still sequence.
    pub fn from_movie(path: &Path, decoder: &dyn ExrDecoder) -> Result<Self> {
        let info = decoder.read_header(path)?;
        let count = decoder.frame_count(path)?.max(1);
        anyhow::ensure!(
            count <= 1_000_000,
            "{} declares {count} frames, which is not a length we will \
             build a timeline for",
            path.display()
        );
        Ok(Self {
            frames: vec![path.to_path_buf(); count as usize],
            info,
            movie: true,
            offset: 0,
            proxy: 1,
            pack: Pack::default(),
        })
    }

    /// Build from an explicit frame list (e.g. one sequence picked out of a
    /// directory by a sequence-aware browser). Order is taken as given.
    pub fn from_files(frames: Vec<PathBuf>, decoder: &dyn ExrDecoder) -> Result<Self> {
        anyhow::ensure!(!frames.is_empty(), "empty frame list");
        let info = decoder.read_header(&frames[0])?;
        Ok(Self { frames, info, movie: false, offset: 0, proxy: 1, pack: Pack::default() })
    }

    /// The inclusive sub-range `a..=b` as a new sequence. Use this rather
    /// than slicing `frames` by hand: a movie's frame numbers live in the
    /// INDEX, so a trim has to move `offset` too.
    ///
    /// `b` MAY run past the frames that exist. A Read can be given a range
    /// longer than its source so a render can be watched while it is still
    /// being written: the frames past the end get the path they WILL have,
    /// and the decoder treats a file that is not there yet as a black
    /// frame rather than an error. Nothing about a missing frame is
    /// cached, so it lights up on the next pass once the renderer writes
    /// it — that is the whole feature.
    pub fn trimmed(&self, a: usize, b: usize) -> Self {
        let last = self.frames.len().saturating_sub(1);
        let a = a.min(last);
        let b = b.max(a);
        let mut frames = Vec::with_capacity(b - a + 1);
        for i in a..=b {
            match self.frames.get(i) {
                Some(p) => frames.push(p.clone()),
                // Past the end: extrapolate from the last real frame's
                // name. If the name has no number to advance (a single
                // still, an odd naming scheme) the last path is repeated —
                // it exists, so it simply holds, which is a better answer
                // than a black hole in the timeline.
                None => {
                    let past = (i - last) as u64;
                    let guess = self
                        .frames
                        .last()
                        .and_then(|p| bump_frame_number(p, past))
                        .or_else(|| self.frames.last().cloned());
                    if let Some(g) = guess {
                        frames.push(g);
                    }
                }
            }
        }
        Self {
            frames,
            info: self.info,
            movie: self.movie,
            offset: self.offset + a as u64,
            proxy: self.proxy,
            pack: self.pack,
        }
    }

    /// The frames `off .. off + count` of this sequence, by FILE index —
    /// which may run before its first file or past its last. A Read's
    /// first/last frame range is built this way.
    ///
    /// Frames outside the files get the path they WOULD have (the nearest
    /// real file's number moved on), so a render that lands there later is
    /// picked up; with no number to move they get a path that cannot exist.
    /// Either way a frame that is not on disk decodes as BLACK and is never
    /// cached (`spawn_decode`'s not-there-yet path) — never a repeat of a
    /// neighbouring frame pretending to be it.
    pub fn window(&self, off: i64, count: u64) -> Self {
        let n = self.frames.len() as i64;
        let missing = |p: &Path| -> PathBuf {
            let mut s = p.as_os_str().to_owned();
            s.push(".wobble-missing");
            PathBuf::from(s)
        };
        let mut frames = Vec::with_capacity(count as usize);
        for k in 0..count as i64 {
            let j = off + k;
            let p = if (0..n).contains(&j) {
                self.frames[j as usize].clone()
            } else if self.movie || n == 0 {
                // A container has no frame outside itself to wait for.
                missing(&self.frames.first().cloned().unwrap_or_default())
            } else if j >= n {
                let last = &self.frames[(n - 1) as usize];
                renumber_frame(last, j - (n - 1)).unwrap_or_else(|| missing(last))
            } else {
                let first = &self.frames[0];
                renumber_frame(first, j).unwrap_or_else(|| missing(first))
            };
            frames.push(p);
        }
        Self {
            frames,
            info: self.info,
            movie: self.movie,
            offset: self.offset.wrapping_add(off as u64),
            proxy: self.proxy,
            pack: self.pack,
        }
    }

    /// The half-resolution view of this sequence: same frames, halved
    /// `info`, proxy divisor doubled — what a proxy playback rig feeds the
    /// player, alongside a `ProxyDecoder` wrapping the real backend.
    pub fn proxied(&self) -> Self {
        Self {
            frames: self.frames.clone(),
            info: crate::decode::ImageInfo {
                width: self.info.width.div_ceil(2),
                height: self.info.height.div_ceil(2),
            },
            movie: self.movie,
            offset: self.offset,
            proxy: self.proxy * 2,
            pack: Pack { window: self.pack.window.map(|w| w.half()), ..self.pack },
        }
    }

    /// The same frames, stored as `pack`.
    pub fn packed(&self, pack: Pack) -> Self {
        Self { frames: self.frames.clone(), pack, ..*self }
    }

    /// Bytes one frame takes in a staging slot (and in either cache tier).
    pub fn slot_bytes(&self) -> u64 {
        self.pack.frame_bytes(self.info)
    }

    /// The NVMe entry for frame `i`, keyed by file, layer, proxy and pack.
    fn disk_file(&self, d: &DiskCache, i: u64, layer: Option<&str>) -> Option<PathBuf> {
        let path = self.frames.get(i as usize)?;
        d.frame_file_windowed(path, layer, self.file_frame(i), self.proxy, self.pack.tag(), self.pack.window_key())
    }

    /// Decode frame `i` into `dst`, in this sequence's pack (`dst` is
    /// `pack.storage_info(info)`'s staging layout). The default decodes
    /// straight in. A packed or windowed reader packs from the FULL frame,
    /// which comes through the shared store (`shared_full`) — one decode
    /// however many readers want a part of it — and is handed back so the
    /// caller can fill the other readers' caches from it.
    fn decode_packed(
        &self,
        decoder: &dyn ExrDecoder,
        i: u64,
        layer: Option<&str>,
        dst: &mut [f16],
    ) -> Result<Option<Arc<Vec<f16>>>> {
        let info = self.info;
        let path = &self.frames[i as usize];
        let ff = self.file_frame(i);
        if self.pack.is_default() {
            let _gate = BigDecode::enter(info.row_stride_f16() * info.height as usize * 2);
            let t0 = Instant::now();
            decoder.decode_frame_into(path, info, info.row_stride_f16(), layer, ff, dst)?;
            note_decode(t0);
            return Ok(None);
        }
        let key: ContentKey = (path.clone(), ff, layer.map(str::to_owned), self.proxy, file_stamp(path));
        let rgba = shared_full(&key, || {
            let len = info.row_stride_f16() * info.height as usize;
            let _gate = BigDecode::enter(len * 2);
            let mut rgba = scratch_frame(len);
            let t0 = Instant::now();
            decoder.decode_frame_into(path, info, info.row_stride_f16(), layer, ff, &mut rgba)?;
            note_decode(t0);
            Ok(rgba)
        })?;
        self.pack.pack(&rgba, info, bytemuck::cast_slice_mut(dst));
        Ok(Some(rgba))
    }

    /// Which frame of `frames[i]` to decode. A file-per-frame sequence
    /// names a different path each time and always wants frame 0; a movie
    /// names one path and wants its position in the clip.
    pub fn file_frame(&self, i: u64) -> u64 {
        if self.movie {
            // Wrapping: a window that opens BEFORE the movie's first frame
            // stores a negative offset in two's complement; every frame it
            // actually decodes lands back in range.
            self.offset.wrapping_add(i)
        } else {
            0
        }
    }

    pub fn path_for_tick(&self, tick: u64) -> &Path {
        &self.frames[(tick % self.frames.len() as u64) as usize]
    }
}

pub struct DecodedFrame {
    pub reader: usize,
    pub tick: u64,
    pub slot: usize,
    pub decode_ms: f32,
    /// Seek generation this job was issued under; stale arrivals are dropped.
    generation: u64,
    /// A black stand-in — the file is absent or would not decode — which is
    /// shown once and never kept: cache a missing frame and it stays missing
    /// forever, and watching a render arrive is the whole feature.
    transient: bool,
    /// The file as it was when this was decoded (`stale_at`).
    stamp: Stamp,
}

/// Everything the render thread needs to upload one display tick.
/// What a scrub gets this redraw: see `Player::show_nearest_set`.
pub enum Scrub {
    /// The hand's tick, landed.
    Exact(FrameSet),
    /// A landed tick between the picture and the hand: progress, not the
    /// answer — keep asking.
    Between(FrameSet),
    /// The hand is on the picture already; nothing to upload.
    OnScreen,
    /// Nothing landed that would be an improvement; keep the picture.
    NotYet,
}

pub struct FrameSet {
    pub tick: u64,
    /// (reader index, staging slot, image info)
    pub uploads: Vec<(usize, usize, ImageInfo)>,
}

pub struct DecodeStats {
    pub per_frame_ms: RollingAvg,
    pub frames_decoded: u64,
    pub stalls: u64,
    pub workers: usize,
}

/// Apply a reader's time map at a tick, HELD inside its sequence.
///
/// A tick is a comp frame number, and a reader occupies the frames its own
/// media covers (a 1001-numbered render sits at 1001..; see the leaf
/// placement in `Graph::walk_time`). Outside that stretch the reader holds
/// its first or last frame, exactly as a compositor's Read does — it must NOT wrap,
/// or a comp whose plates start at 1001 would show frame 1 of the plate at
/// comp frame 1 instead of nothing, and a sped-up retime would restart the
/// clip rather than freeze on its end.
pub fn source_frame_of(map: &TimeMap, len: u64, tick: u64) -> u64 {
    map.apply(tick).min(len.saturating_sub(1))
}

/// `source_frame_of` under a source's EDGE behaviour: what a clip answers
/// for a frame outside its own range — held, looped, bounced, or `None`
/// for black. Inside the clip it is the same frame every way.
pub fn source_index_of(map: &TimeMap, len: u64, tick: u64, edges: ReadEdges) -> Option<u64> {
    ReadEdge::resolve(edges, map.apply_i(tick), len)
}

/// Steps from `from` to `to` in playback order under `span`, or `None` if
/// playing on never lands there. Free function so callers can hold it while
/// the player is borrowed mutably.
fn dist_in(span: Option<(u64, u64)>, from: u64, to: u64) -> Option<u64> {
    let Some((a, b)) = span else { return to.checked_sub(from) };
    let len = b - a + 1;
    if from > b {
        // Past the out point: the next step joins the loop at its start.
        return match to {
            t if t == from => Some(0),
            t if t >= a && t <= b => Some(1 + (t - a)),
            _ => None,
        };
    }
    if from < a {
        // Before the in point: play forward into the loop, then loop.
        return (to >= from && to <= b).then(|| to - from);
    }
    (to >= a && to <= b).then(|| (to + len - from) % len)
}

pub struct Player {
    sequences: Vec<Arc<Sequence>>,
    ring: Arc<StagingRing>,
    decoder: Arc<dyn ExrDecoder>,
    ready_tx: Sender<DecodedFrame>,
    ready_rx: Receiver<DecodedFrame>,
    /// Decoded frames resident in the ring: those waiting for their tick
    /// AND those already shown. A set is SHOWN, never taken — the frame
    /// stays in its slot until an eviction wants the slot — so a tick the
    /// hand comes back to is one GPU copy, no decode, no memcpy. The ring
    /// is sized by a byte budget; this map is the scrub cache.
    pending: HashMap<(usize, u64), DecodedFrame>,
    /// How many `pending` entries name each slot. A reader whose tick maps
    /// to the SAME source frame as a tick already decoded (a held frame, a
    /// Time Remap or Freeze, an edge holding past the clip) shares that
    /// tick's slot instead of decoding and uploading 275 MB again; the slot
    /// goes back to the ring when the last entry naming it goes.
    slot_refs: HashMap<usize, u32>,
    /// Slots the app is copying from THIS redraw: named by a set handed out
    /// by `show_*` and not yet submitted. Nothing may remap them — not an
    /// eviction, not `invalidate`, not `forget` — until `frame_submitted`.
    held: Vec<usize>,
    /// Slots whose frames were retired while held (a time-map invalidate
    /// between the show and the submit): released at `frame_submitted`.
    release_after_submit: Vec<usize>,
    /// Jobs spawned but not yet returned, so a re-targeted playhead does
    /// not order the same frame twice.
    inflight: std::collections::HashSet<(usize, u64)>,
    /// Next (tick, reader) pair to issue, in playback order.
    issue_tick: u64,
    issue_reader: usize,
    pub display_tick: u64,
    /// The inclusive frame span playback loops over — the app's in/out (or
    /// the shot). `None` runs forward forever, which is what a rig nobody
    /// has configured does.
    ///
    /// A tick IS a frame number, so the loop lives here rather than in the
    /// modulo it used to be: the sequence length cannot say where a comp
    /// starts or ends once readers are placed on the timeline.
    loop_span: Option<(u64, u64)>,
    /// How many ticks past the display tick to keep decoding.
    pub ahead: u64,
    /// Bumped by seek(); in-flight decodes from older generations are
    /// discarded on arrival (their staging slots go straight back).
    generation: u64,
    cache: Arc<Mutex<FrameCache>>,
    /// Per-reader tier toggles, index-aligned with `sequences` (missing =
    /// both tiers on). Set after `new()`; read per spawned job.
    pub reader_opts: Vec<ReaderCacheOpts>,
    /// Per-reader EXR layer selection, index-aligned with `sequences`
    /// (missing/None = the file's own RGBA).
    pub reader_layers: Vec<Option<String>>,
    /// Per-reader time remap from the graph's time nodes, index-aligned with
    /// `sequences` (missing = identity). Only the SOURCE FRAME moves: the
    /// timeline, the ring and the display tick are untouched.
    time_maps: Vec<TimeMap>,
    /// Per-reader edge behaviour (hold / loop / bounce / black outside the
    /// clip's own frames), index-aligned with `sequences`; missing = hold.
    edges: Vec<ReadEdges>,
    /// NVMe tier under the RAM cache; None = disabled.
    disk: Option<Arc<DiskCache>>,
    /// Which readers are WANTED right now, index-aligned with `sequences`
    /// (missing = wanted). An inactive reader is neither issued nor waited
    /// on: a tick is complete when every ACTIVE reader has it. The rig can
    /// hold every reader the graph could show while decoding only what the
    /// viewers and a running bake actually sample — so a nine-string harp
    /// costs one decode while one string is on screen, and a view flip
    /// changes this mask instead of rebuilding the rig (slots, time maps
    /// and the RAM cache's reader indices never move).
    active: Vec<bool>,
    /// The tick the app has on screen, if any. The scrub rules — which
    /// landed frames to keep, which one to show — are relative to it.
    pub shown: Option<u64>,
    /// The hand is heading for smaller ticks: the issue cursor and the
    /// `ahead` speculation walk backwards from the playhead. Forward-only
    /// prefetch made a backward drag pay a full decode per step while a
    /// forward one found its frames already waiting.
    pub backward: bool,
    /// Ring slots speculation may not take: a reversal must be able to
    /// issue its target AT ONCE while the decodes fired the other way land
    /// and are dropped. In-flight work cannot be evicted, so without a
    /// reserve a direction change queued its first steps behind ten
    /// wrong-way decodes (measured: 3.1 redraws per reversal against 1.0
    /// for a straight step). The app sets it to one tick's worth of slots
    /// while scrubbing and 0 while playing, where there are no reversals.
    pub reserve: usize,
    /// The pixels on screen are not this player's current answer for
    /// `shown`: a fresh rig, a viewer-cache swap, a time-map change, a file
    /// rewritten under the shown tick. While set, "the hand is on the
    /// picture" is not an answer — the tick is shown again as soon as it has
    /// landed, and playback does not step past it. Any successful show
    /// clears it. The app sets it on the swaps; `invalidate` and `forget`
    /// set it themselves.
    pub picture_stale: bool,
    pub stats: DecodeStats,
}

impl Player {
    pub fn new(
        sequences: Vec<Arc<Sequence>>,
        ring: Arc<StagingRing>,
        decoder: Arc<dyn ExrDecoder>,
        ahead: u64,
        cache_budget_bytes: usize,
    ) -> Self {
        assert!(!sequences.is_empty());
        // A display tick needs every reader's slot at once, and the tick on
        // screen stays resident and pinned: a ring holding fewer than two
        // ticks' worth could never make room for the next frame.
        let two_ticks: u64 = 2 * sequences.iter().map(|s| s.slot_bytes()).sum::<u64>();
        assert!(
            ring.budget_bytes() >= two_ticks,
            "staging ring needs two ticks' worth of bytes, {} readers want {} > {}",
            sequences.len(),
            two_ticks,
            ring.budget_bytes()
        );
        let (ready_tx, ready_rx) = crossbeam_channel::unbounded();
        let cache = {
            let mut c = FrameCache::new(cache_budget_bytes);
            let loop_len = sequences.iter().map(|s| s.frames.len() as u64).max().unwrap_or(1);
            c.cycle_lookups = sequences.len() as u64 * loop_len;
            Arc::new(Mutex::new(c))
        };
        Self {
            sequences,
            ring,
            decoder,
            ready_tx,
            ready_rx,
            pending: HashMap::new(),
            slot_refs: HashMap::new(),
            held: Vec::new(),
            release_after_submit: Vec::new(),
            inflight: Default::default(),
            issue_tick: 0,
            issue_reader: 0,
            display_tick: 0,
            loop_span: None,
            ahead,
            generation: 0,
            cache,
            reader_opts: Vec::new(),
            reader_layers: Vec::new(),
            time_maps: Vec::new(),
            edges: Vec::new(),
            disk: None,
            active: Vec::new(),
            shown: None,
            backward: false,
            reserve: 0,
            picture_stale: false,
            stats: DecodeStats {
                per_frame_ms: RollingAvg::default(),
                frames_decoded: 0,
                stalls: 0,
                workers: rayon::current_num_threads(),
            },
        }
    }

    pub fn sequences(&self) -> &[Arc<Sequence>] {
        &self.sequences
    }

    pub fn set_disk_cache(&mut self, disk: Option<Arc<DiskCache>>) {
        self.disk = disk;
    }

    /// Per-reader source-frame remap (the graph's time nodes). Returns
    /// whether anything moved; when it did, frames already decoded under the
    /// old map are dropped and decode restarts at the playhead — otherwise
    /// the next few displayed ticks would still show the old timing.
    pub fn set_time_maps(&mut self, maps: Vec<TimeMap>) -> bool {
        // A missing entry IS the identity, so handing a fresh player its
        // first (all-identity) list is not a change and must not re-seek.
        let unchanged = (0..self.time_maps.len().max(maps.len())).all(|i| {
            let was = self.time_maps.get(i).cloned().unwrap_or_default();
            let now = maps.get(i).cloned().unwrap_or_default();
            was == now
        });
        if unchanged {
            self.time_maps = maps;
            return false;
        }
        // The CALLER re-seeks: display_tick sits one past the shown frame
        // when paused, so seeking it here would nudge the playhead — and the
        // playhead must NEVER move for a time edit.
        self.time_maps = maps;
        true
    }

    pub fn time_maps(&self) -> &[TimeMap] {
        &self.time_maps
    }

    /// Set which readers are wanted (see `active`). Returns whether it
    /// changed. Nothing is thrown away: frames an inactive reader already
    /// decoded stay pending until the usual whole-tick eviction takes them,
    /// and a reader that comes back active simply leaves a hole at the
    /// display tick, which `pump` rewinds the issue cursor onto.
    pub fn set_active(&mut self, active: Vec<bool>) -> bool {
        let n = self.sequences.len();
        let norm = |v: &[bool]| -> Vec<bool> {
            (0..n).map(|r| v.get(r).copied().unwrap_or(true)).collect()
        };
        let (was, now) = (norm(&self.active), norm(&active));
        if was == now {
            return false;
        }
        self.active = now;
        true
    }

    /// Is reader `r` wanted? Missing entries are.
    pub fn is_active(&self, r: usize) -> bool {
        self.active.get(r).copied().unwrap_or(true)
    }

    /// Set each reader's edge behaviour. Returns whether anything changed:
    /// a changed edge makes decoded frames WRONG (a frame past the end that
    /// was held is now black), so the caller invalidates.
    pub fn set_edges(&mut self, edges: Vec<ReadEdges>) -> bool {
        let n = self.sequences.len();
        let norm = |v: &[ReadEdges]| -> Vec<ReadEdges> {
            (0..n).map(|r| v.get(r).copied().unwrap_or_default()).collect()
        };
        let changed = norm(&self.edges) != norm(&edges);
        self.edges = edges;
        changed
    }

    /// The source frame a reader plays at a timeline tick — `None` when
    /// its edge behaviour says black.
    pub fn source_index(&self, reader: usize, tick: u64) -> Option<u64> {
        let len = self.sequences.get(reader).map(|s| s.frames.len() as u64).unwrap_or(1);
        source_index_of(
            &self.time_maps.get(reader).cloned().unwrap_or_default(),
            len,
            tick,
            self.edges.get(reader).copied().unwrap_or_default(),
        )
    }

    /// The source frame a reader plays at a timeline tick (a black edge
    /// reports the nearest real frame — for meters and the like).
    pub fn source_frame(&self, reader: usize, tick: u64) -> u64 {
        let len = self.sequences.get(reader).map(|s| s.frames.len() as u64).unwrap_or(1);
        self.source_index(reader, tick).unwrap_or_else(|| {
            source_frame_of(&self.time_maps.get(reader).cloned().unwrap_or_default(), len, tick)
        })
    }

    /// Change the RAM budget live. 0 drains resident frames immediately —
    /// the viewer render-cache borrows the RAM this way ("read caches fall
    /// back to NVMe"); restoring a budget just lets the cache refill.
    pub fn set_cache_budget(&mut self, bytes: usize) {
        let mut c = self.cache.lock().unwrap();
        c.budget = bytes;
        if bytes == 0 {
            c.map.clear();
            c.total = 0;
        }
    }

    /// Set the inclusive frame span playback loops over (`None` = forward
    /// forever). Cheap and idempotent: the app pushes the current in/out
    /// every frame.
    pub fn set_loop(&mut self, span: Option<(u64, u64)>) {
        self.loop_span = span.map(|(a, b)| (a, b.max(a)));
    }

    pub fn loop_span(&self) -> Option<(u64, u64)> {
        self.loop_span
    }

    /// The frame after `t` in playback order.
    pub fn next_tick(&self, t: u64) -> u64 {
        match self.loop_span {
            Some((a, b)) if t >= b => a,
            _ => t + 1,
        }
    }

    /// How many `next_tick` steps from the playhead to `t`, or `None` when
    /// playing on never reaches it — the loop's replacement for "`t` is
    /// behind the playhead". A tick outside the loop is only ever reached
    /// when the playhead is already outside it too (you scrubbed into the
    /// handles and pressed play).
    fn dist(&self, t: u64) -> Option<u64> {
        if self.backward {
            // A backward scrub: the loop is irrelevant, the hand is heading
            // for smaller ticks and so is the prefetch.
            return (t <= self.display_tick).then(|| self.display_tick - t);
        }
        dist_in(self.loop_span, self.display_tick, t)
    }

    /// Is `t` between the picture on screen and the playhead (inclusive)?
    /// With nothing on screen yet, everything is. The between rule of a
    /// scrub: a landed frame outside this span can never be shown until the
    /// hand comes back for it, and the RAM tier is the cache for that.
    fn in_scrub_span(&self, t: u64) -> bool {
        match self.shown {
            Some(s) => (s.min(self.display_tick)..=s.max(self.display_tick)).contains(&t),
            None => true,
        }
    }

    /// Is `t` where the hand is heading — past the playhead in the drag's
    /// direction? Speculation lands there, and a landing that beats the
    /// hand to its next tick is the whole point of it. The between rule
    /// says what may be SHOWN; what is KEPT must include these, or every
    /// prefetched tick that lands during the redraw or two the player is
    /// chasing is dropped and decoded again (measured: 572 decodes for 120
    /// backward seeks, against 283 with the RAM tier hiding it).
    fn ahead_of_hand(&self, t: u64) -> bool {
        if self.backward {
            t < self.display_tick
        } else {
            t > self.display_tick
        }
    }

    /// What a scrub keeps a landed frame for: to show it on the way to the
    /// hand, or because the hand is heading for it. Behind the picture is
    /// dead weight.
    fn scrub_wants(&self, t: u64) -> bool {
        self.in_scrub_span(t) || self.ahead_of_hand(t)
    }

    /// Longest loaded sequence, in frames — how much media the rig holds.
    pub fn timeline_len(&self) -> u64 {
        self.sequences.iter().map(|s| s.frames.len() as u64).max().unwrap_or(1)
    }

    /// Move the playhead. Deliberately does NOT throw work away.
    ///
    /// A scrub re-targets the playhead every UI frame — every ~16ms — and a
    /// decode of a 2K plate takes ~27ms. So anything discarded here is work
    /// that can never be replaced before the next re-target: the old
    /// discard-everything seek measured **969 decodes issued and ZERO
    /// frames displayed** across a three-second drag, while playback (which
    /// never seeks) ran at 120 sets/s with nothing skipped. That gap was
    /// the whole difference between "playback is great" and "scrubbing is
    /// not".
    ///
    /// So: already-decoded frames stay in `pending` (a drag comes back over
    /// them constantly), in-flight decodes are allowed to land and be used,
    /// and ring pressure is relieved by evicting whatever sits FURTHEST
    /// from the playhead rather than by dropping everything near it.
    ///
    /// Use [`Player::invalidate`] when the frames are actually wrong.
    pub fn seek(&mut self, tick: u64) {
        self.display_tick = tick;
        self.issue_tick = tick;
        self.issue_reader = 0;
    }

    /// Throw away every decoded and in-flight frame, then move the playhead.
    ///
    /// Only for when the pixels themselves became WRONG — a time map
    /// changed, so a tick now names a different source frame. A plain seek
    /// must never do this; see [`Player::seek`].
    pub fn invalidate(&mut self, tick: u64) {
        self.generation += 1;
        self.picture_stale = true;
        let gone: Vec<usize> = self.pending.drain().map(|(_, f)| f.slot).collect();
        for slot in gone {
            // A slot the app is copying from this redraw goes back when
            // the copy is submitted, not now (`drop_slot`).
            self.drop_slot(slot);
        }
        self.inflight.clear();
        self.seek(tick);
    }

    /// Forget ONE reader's frame for `tick`, because the file underneath it
    /// changed — a live Blender render landed where a black "not there
    /// yet" frame, or an older render, had been decoded. The RAM entry and
    /// the pending copy go; a decode still in flight for it lands stale
    /// and is dropped (the generation moves, so everything in flight is
    /// re-issued — cheap next to the render that just took seconds). The
    /// NVMe tier keys on the file's size and mtime and forgets by itself.
    ///
    /// The one hole: a decode that was already reading the OLD file when
    /// it was overwritten still inserts what it read into the RAM cache.
    /// That needs a render to land in the same few milliseconds as a
    /// decode of the previous render of that very frame, and the next
    /// render of it clears the entry again.
    /// Has a file under the frames resident for `tick` changed since they
    /// were decoded? Those frames are forgotten (so the tick decodes again)
    /// and the answer is true. One `stat` per active reader: the app asks
    /// for the tick on screen, which is where a re-render is looked at.
    pub fn stale_at(&mut self, tick: u64) -> bool {
        let stale: Vec<usize> = (0..self.sequences.len())
            .filter(|&r| self.is_active(r))
            .filter(|&r| {
                let Some(f) = self.pending.get(&(r, tick)) else { return false };
                let Some(i) = self.source_index(r, tick) else { return false };
                let Some(path) = self.sequences[r].frames.get(i as usize) else { return false };
                !f.transient && stamps_differ(f.stamp, file_stamp(path))
            })
            .collect();
        for &r in &stale {
            self.forget(r, tick);
        }
        !stale.is_empty()
    }

    pub fn forget(&mut self, reader: usize, tick: u64) {
        if self.shown == Some(tick) {
            self.picture_stale = true;
        }
        let src = self.source_frame(reader, tick);
        self.cache.lock().unwrap().remove((reader, src));
        if let Some(f) = self.pending.remove(&(reader, tick)) {
            // Every tick sharing that decode shows the same stale file.
            let aliases: Vec<(usize, u64)> = self
                .pending
                .iter()
                .filter(|(_, g)| g.slot == f.slot)
                .map(|(k, _)| *k)
                .collect();
            for k in aliases {
                self.pending.remove(&k);
                self.drop_slot(f.slot);
            }
            self.drop_slot(f.slot);
        }
        if self.inflight.contains(&(reader, tick)) {
            self.generation += 1;
            self.inflight.clear();
        }
        // The display tick may now be a hole; `pump` rewinds the issue
        // cursor onto it, but only if it is not sitting past it already
        // by more than one reader's step — make that unconditional.
        if self.issue_tick != tick {
            self.issue_tick = tick;
            self.issue_reader = 0;
        }
    }

    /// Give a ring slot back by dropping the decoded frame furthest from the
    /// playhead — never the one the display is waiting on, and never one
    /// nearer than the frame we are trying to issue. The slot returns via
    /// the ring's remap, so it lands on a later pump, not this one.
    fn evict_farthest(&mut self) -> bool {
        let display = self.display_tick;
        let chasing = self.ahead == 0;
        // Ticks with a slot the app is copying from this redraw stay put,
        // and so does the picture on screen: evicting it costs a decode the
        // moment the hand touches it.
        let pinned: std::collections::HashSet<u64> = self
            .pending
            .iter()
            .filter(|(_, f)| self.held.contains(&f.slot))
            .map(|((_, t), _)| *t)
            .collect();
        // Uselessness, highest first. Playing, anything the loop will not
        // reach again can never be shown and goes before anything ahead of
        // it, however far ahead. Chasing a scrub, a frame the picture has
        // already passed (outside the shown..playhead span) goes first, then
        // whatever is farthest from the playhead — either direction may be
        // asked for next.
        let score = |t: u64| -> (u8, u64) {
            if chasing {
                return (u8::from(!self.scrub_wants(t)), t.abs_diff(display));
            }
            match self.dist(t) {
                Some(k) => (0, k),
                None => (1, t.abs_diff(display)),
            }
        };
        // Chasing a scrub, a tick the between rule can show THIS redraw —
        // whole, between the picture and the hand — is the best picture
        // there is, and one still landing is about to be: neither goes to
        // make room for a newer target. On a ring a few ticks deep (four
        // 8K layers) the hand re-targets faster than a tick lands, every
        // landed tick was evicted for the next target in the very pump it
        // completed in, and a three-second drag showed nothing at all. The
        // target waits instead; the tick is shown, the picture it replaces
        // becomes evictable, and the picture follows the hand at the rate
        // the ring can carry.
        let readers = self.sequences.len();
        let showable = |t: u64| {
            chasing
                && self.in_scrub_span(t)
                && (self.have_all(t) || (0..readers).any(|r| self.inflight.contains(&(r, t))))
        };
        let victim = self
            .pending
            .keys()
            .map(|(_, t)| *t)
            .filter(|t| *t != display && Some(*t) != self.shown && !pinned.contains(t))
            .filter(|t| !showable(*t))
            .max_by_key(|t| score(*t));
        // Never evict something nearer the playhead than the frame we are
        // trying to make room for — that would just churn.
        let want = score(self.issue_tick);
        let Some(t) = victim.filter(|t| score(*t) > want) else { return false };
        // A whole TICK, never half a pair. A display set needs every
        // reader's frame for one tick, so evicting one reader's half turns
        // the other half into dead weight AND leaves a hole the issue
        // cursor has already moved past.
        for r in 0..self.sequences.len() {
            if let Some(f) = self.pending.remove(&(r, t)) {
                self.drop_slot(f.slot);
            }
        }
        true
    }

    /// One `pending` entry naming `slot` is gone: the slot goes back to the
    /// ring once none is left — after this redraw's copies are submitted
    /// when the app is copying from it now.
    fn drop_slot(&mut self, slot: usize) {
        let left = match self.slot_refs.get_mut(&slot) {
            Some(n) if *n > 1 => {
                *n -= 1;
                return;
            }
            _ => 0,
        };
        debug_assert_eq!(left, 0);
        self.slot_refs.remove(&slot);
        if self.held.contains(&slot) {
            self.release_after_submit.push(slot);
        } else {
            self.ring.release_after_copy(slot);
        }
    }

    /// A slot already holding what `(reader, tick)` would decode: a pending
    /// frame of the same reader whose tick maps to the same source frame.
    /// `Err(())` while such a frame is still in flight (wait for it rather
    /// than decode it twice); `Ok(None)` when there is nothing to share.
    fn shared_slot(&self, reader: usize, tick: u64) -> Result<Option<usize>, ()> {
        let Some(src) = self.source_index(reader, tick) else { return Ok(None) };
        let same = |t: u64| t != tick && self.source_index(reader, t) == Some(src);
        if let Some(f) = self.pending.iter().find_map(|((r, t), f)| {
            (*r == reader && !f.transient && f.generation == self.generation && same(*t)).then_some(f)
        }) {
            return Ok(Some(f.slot));
        }
        if self.inflight.iter().any(|(r, t)| *r == reader && same(*t)) {
            return Err(());
        }
        Ok(None)
    }

    /// Drain finished decodes, top up the job queue. Call once per frame.
    pub fn pump(&mut self) {
        let ahead = self.ahead;
        for f in self.ready_rx.try_iter() {
            self.stats.per_frame_ms.push(f.decode_ms);
            self.stats.frames_decoded += 1;
            self.inflight.remove(&(f.reader, f.tick));
            // `ahead == 0` is the app telling us it is chasing a seek.
            let chasing = self.ahead == 0;
            let keep = if f.generation != self.generation {
                false
            } else if f.transient && f.tick != self.display_tick {
                // A black stand-in is never cached (see `transient`); one
                // that is not about to be shown is not even kept.
                false
            } else if chasing {
                // Chasing a scrub: a frame BEHIND the picture — on the far
                // side of what is on screen from the playhead — can never
                // be shown by the between rule (`show_nearest_set`),
                // and holding its slot is what starved the ring: landings
                // arrive out of order, the ones the picture had passed
                // piled up, and every new target waited on an eviction's
                // remap first (measured: 46 pictures for 117 seeks, and
                // 15ms added to every landing). A frame the hand is heading
                // for is kept: see `scrub_wants`.
                self.scrub_wants(f.tick)
            } else {
                // A frame the loop will not reach soon is dead weight
                // during PLAYBACK (the display only moves forward) and
                // keeping it starves the ring — every slot ends up holding
                // a frame that will never be shown.
                self.dist(f.tick).is_some_and(|k| k <= ahead)
            };
            if !keep {
                self.ring.release_after_copy(f.slot);
                continue;
            }
            *self.slot_refs.entry(f.slot).or_insert(0) += 1;
            self.pending.insert((f.reader, f.tick), f);
        }
        // INVARIANT: the display tick is always decoded, or on its way.
        // `issue_tick` only ever moves forward, so any hole left behind it —
        // by an eviction, a drop, or a decode that errored — would be a
        // frame nobody is ever going to produce, and the display would wait
        // for it forever. Rewinding the cursor onto the hole is what makes
        // that unrepresentable rather than merely unlikely.
        let d = self.display_tick;
        let hole = (0..self.sequences.len()).any(|r| {
            self.is_active(r)
                && !self.pending.contains_key(&(r, d))
                && !self.inflight.contains(&(r, d))
        });
        if hole && self.issue_tick != d {
            self.issue_tick = d;
            self.issue_reader = 0;
        }
        // The cursor restarts from the playhead whenever it is outside the
        // prefetch window — the loop span moved under it (the app pushes
        // in/out every frame), the drag changed direction, or it simply
        // finished the window last pump. The walk skips what is pending or
        // in flight, so this costs nothing and issues nothing twice; a
        // cursor left parked behind the playhead by a backward prefetch
        // used to never issue again once the direction flipped.
        if self.dist(self.issue_tick).is_none_or(|k| k > self.ahead) {
            self.issue_tick = d;
            self.issue_reader = 0;
        }

        // Chasing a seek (ahead == 0): cap how many TARGETS may be in
        // flight at once. A drag re-targets every ~16ms; on a plate that
        // takes ~150ms to decode that is nine targets started before the
        // first one lands, all competing for twelve ring slots, and every
        // one of them obsolete on arrival. Two in flight keeps the pipe
        // full enough to hide latency without letting the hand outrun the
        // decoder by an order of magnitude.
        // Leave a third of the ring free for frames that have LANDED.
        // Filling every slot with in-flight work is what starved the
        // display: an arriving frame had nowhere to sit and was evicted to
        // make room for the next speculative decode. Measured on a 12-slot
        // ring with two readers, 4K plates: 8 in-flight jobs (four targets)
        // peaks at ~17fps, 10 collapses back to 7.
        // Decoupled from the ring, which is a cache budget now: four targets'
        // worth, the value that measured best when it WAS two thirds of a
        // twelve-slot ring.
        let readers = self.sequences.len();
        // The ring is a BYTE budget and every reader's frame has its own
        // size: one tick is the active readers' bytes, and everything the
        // old slot counts said is said in ticks of that.
        let active: Vec<usize> = (0..readers).filter(|r| self.is_active(*r)).collect();
        let tick_bytes: u64 =
            active.iter().map(|r| self.sequences[*r].slot_bytes()).sum::<u64>().max(1);
        // `reserve` is in slots, a tick's worth being one a reader.
        let reserve_bytes = tick_bytes * self.reserve as u64 / readers.max(1) as u64;
        let ring_ticks = (self.ring.budget_bytes() / tick_bytes) as usize;
        let per_tick = active.len().max(1);
        let chase_cap = (readers * 4).min(ring_ticks * per_tick * 2 / 3).max(readers);
        if self.ahead == 0 && self.inflight.len() >= chase_cap {
            return;
        }
        // The cursor walks in LOOP order now, so it can come back round to
        // ticks it has already issued — bound the work per pump rather than
        // relying on the old "issue_tick only grows" termination.
        let mut budget = (self.ahead + 1) * self.sequences.len() as u64 + 1;
        while self.dist(self.issue_tick).is_some_and(|k| k <= self.ahead) {
            if budget == 0 || (self.ahead == 0 && self.inflight.len() >= chase_cap) {
                break;
            }
            budget -= 1;
            let key = (self.issue_reader, self.issue_tick);
            // Already decoded, or already on its way: a re-targeted
            // playhead must not order the same frame a second time. A
            // reader nothing is looking at is not ordered at all.
            if !self.is_active(key.0)
                || self.pending.contains_key(&key)
                || self.inflight.contains(&key)
            {
                self.advance_issue();
                continue;
            }
            // The same source frame already decoded for another tick: share
            // its slot (no decode, no second 275 MB in the ring), or wait
            // for the one in flight.
            match self.shared_slot(key.0, key.1) {
                Ok(Some(slot)) => {
                    // The same file's stamp: the shared pixels are its.
                    let stamp = self.pending.values().find(|f| f.slot == slot).and_then(|f| f.stamp);
                    *self.slot_refs.entry(slot).or_insert(0) += 1;
                    self.pending.insert(
                        key,
                        DecodedFrame {
                            reader: key.0,
                            tick: key.1,
                            slot,
                            decode_ms: 0.0,
                            generation: self.generation,
                            transient: false,
                            stamp,
                        },
                    );
                    self.advance_issue();
                    continue;
                }
                Err(()) => {
                    self.advance_issue();
                    continue;
                }
                Ok(None) => {}
            }
            // The display tick may take any slot; speculation leaves the
            // reserve alone (see `reserve`). What it could not issue is
            // counted below and freed for the next pump.
            if key.1 != self.display_tick && self.ring.free_bytes() <= reserve_bytes {
                break;
            }
            // Chasing a scrub, a tick is issued WHOLE or not at all. The
            // hand re-targets every redraw and the cursor follows it, so a
            // tick with room for only some of its readers was left with
            // the rest never issued: every slot ended up behind half a
            // tick nobody could show (four 8K layers on a ring of fifteen
            // slots: 240 frames delivered in three seconds, none shown).
            if self.ahead == 0 {
                let need: u64 = active
                    .iter()
                    .filter(|r| {
                        !self.pending.contains_key(&(**r, key.1)) && !self.inflight.contains(&(**r, key.1))
                    })
                    .map(|r| self.sequences[*r].slot_bytes())
                    .sum();
                if need > self.ring.free_bytes() {
                    self.evict_farthest();
                    break;
                }
            }
            let Some(slot) = self.ring.try_acquire_sized(self.sequences[key.0].slot_bytes()) else {
                // Out of slots: free the least useful frame and come back
                // for it next pump (the ring remaps asynchronously).
                self.evict_farthest();
                break;
            };
            self.inflight.insert(key);
            self.spawn_decode(key.0, key.1, slot);
            self.advance_issue();
        }
        // Free, ahead of need, what the window still lacks plus the
        // reserve plus one tick for the next speculative issue. An evicted
        // slot returns one poll after its remap, so evicting only when an
        // acquire fails costs every target a redraw; and once the ring has
        // grown to its budget EVERY new decode needs an eviction — a clip
        // longer than the ring used to stop prefetching altogether and
        // play at one late decode per tick. `evict_farthest` refuses
        // anything nearer the playhead than the cursor, so this only ever
        // spends the resident cache behind the hand; slots already on
        // their way back count as free.
        let missing = self.window_holes();
        let want_free = reserve_bytes + tick_bytes * missing.max(1) as u64;
        let mut evictions = missing + 2;
        while evictions > 0 && self.ring.free_bytes() + self.ring.remapping_bytes() < want_free {
            if !self.evict_farthest() {
                break;
            }
            evictions -= 1;
        }
    }

    /// Ticks in the prefetch window (the playhead and `ahead` past it, in
    /// the drag's direction) that some active reader has neither landed nor
    /// in flight — what the next pump would like to issue.
    fn window_holes(&self) -> usize {
        let mut t = self.display_tick;
        let mut holes = 0;
        for _ in 0..=self.ahead {
            let hole = (0..self.sequences.len()).any(|r| {
                self.is_active(r)
                    && !self.pending.contains_key(&(r, t))
                    && !self.inflight.contains(&(r, t))
            });
            holes += usize::from(hole);
            t = if self.backward {
                match t.checked_sub(1) {
                    Some(t) => t,
                    None => break,
                }
            } else {
                self.next_tick(t)
            };
        }
        holes
    }

    /// (pending, inflight, free slots, complete ticks) — scrub diagnostics.
    pub fn debug_pressure(&self) -> (usize, usize, usize, usize) {
        let ticks: std::collections::HashSet<u64> =
            self.pending.keys().map(|(_, t)| *t).collect();
        let complete = ticks.iter().filter(|t| self.have_all(**t)).count();
        (self.pending.len(), self.inflight.len(), self.ring.free_count(), complete)
    }

    fn advance_issue(&mut self) {
        self.issue_reader += 1;
        if self.issue_reader >= self.sequences.len() {
            self.issue_reader = 0;
            self.issue_tick = if self.backward {
                // Below frame 0 there is nothing to issue: park the cursor
                // where `dist` is None, so the issue loop ends and the next
                // pump puts it back on the playhead.
                self.issue_tick.checked_sub(1).unwrap_or(u64::MAX)
            } else {
                self.next_tick(self.issue_tick)
            };
        }
    }

    fn spawn_decode(&self, reader: usize, tick: u64, slot: usize) {
        let seq = self.sequences[reader].clone();
        let ring = self.ring.clone();
        let decoder = self.decoder.clone();
        let tx = self.ready_tx.clone();
        let generation = self.generation;
        let cache = self.cache.clone();
        let opts = self.reader_opts.get(reader).copied().unwrap_or_default();
        let disk = if opts.disk { self.disk.clone() } else { None };
        let layer = self.reader_layers.get(reader).cloned().flatten();
        // The time nodes move the SOURCE frame only; the job still belongs
        // to `tick`, so delivery, the ring and the display are unchanged.
        let src_index = self.source_index(reader, tick);
        // The other readers of the same frames (a plate cut into sections
        // is one reader a section): a decode made for this one fills their
        // caches too, so each file is decoded once however far apart the
        // sections are slipped.
        let siblings: Vec<(usize, Arc<Sequence>, ReaderCacheOpts)> = match src_index {
            Some(i) if !seq.pack.is_default() => (0..self.sequences.len())
                .filter(|r| *r != reader)
                .filter_map(|r| {
                    let s = &self.sequences[r];
                    let same = s.info == seq.info
                        && s.proxy == seq.proxy
                        && s.frames.get(i as usize) == seq.frames.get(i as usize)
                        && s.file_frame(i) == seq.file_frame(i)
                        && self.reader_layers.get(r).cloned().flatten() == layer;
                    same.then(|| (r, s.clone(), self.reader_opts.get(r).copied().unwrap_or_default()))
                })
                .collect(),
            _ => Vec::new(),
        };
        let all_disk = self.disk.clone();
        spawn_job(move || {
            // Outside the clip with a BLACK edge: there is no frame to read.
            // Same answer as "not there yet" below — black, never cached.
            let black = src_index.is_none();
            let seq_frame = src_index.unwrap_or(0);
            let path = seq.frames[seq_frame as usize].clone();
            // What the caches and the slot hold: the plate's staging layout,
            // or the packed one's (`Pack::storage_info`).
            let info = seq.pack.storage_info(seq.info);
            let frame_f16 = info.row_stride_f16() * info.height as usize;
            let t0 = Instant::now();
            let buffer = ring.buffer(slot);

            // NOT THERE YET. A Read may be given a range longer than its
            // source so a render can be watched as it is written; the
            // frames past the end simply do not exist on disk. That is an
            // expected state, not an error: hand back a black frame, log
            // nothing, and — the part that matters — cache nothing, so the
            // next pass over this tick picks the file up the moment the
            // renderer finishes writing it.
            // A frame the ring's slots cannot hold is never written: black
            // for this pass (the app is already building a ring that fits).
            let fits = seq.slot_bytes() <= ring.slot_size(slot);
            if black || !fits || !crate::vfs::exists(&path) {
                let mut view = buffer.slice(..).get_mapped_range_mut().expect("a mapped slot");
                view.slice(..).fill(0);
                drop(view);
                buffer.unmap();
                ring.note_unmapped(slot);
                let decode_ms = t0.elapsed().as_secs_f32() * 1e3;
                let _ = tx.send(DecodedFrame {
                    reader,
                    tick,
                    slot,
                    decode_ms,
                    generation,
                    transient: true,
                    stamp: None,
                });
                return;
            }

            // Set wherever black stands in for pixels that could not be read.
            let mut transient = false;
            // The full frame, when this job decoded (or shared) one.
            let mut full: Option<Arc<Vec<f16>>> = None;
            // An NVMe write owed once the frame has been handed over: the
            // picture must not wait on a compress and a write.
            let mut owed: Option<(Arc<DiskCache>, PathBuf, Arc<Vec<f16>>)> = None;
            // The file as it is NOW: a RAM entry decoded from an older
            // version of it is a miss (the NVMe key moves by itself).
            let stamp = file_stamp(&path);
            let use_ram = opts.ram && cache.lock().unwrap().budget > 0;
            let cached = if use_ram {
                cache.lock().unwrap().get_fresh((reader, seq_frame), stamp)
            } else {
                None
            };
            // Tier order: RAM -> NVMe -> decode. Either enabled tier needs
            // the owned-buffer path (the data must outlive the slot copy).
            // Keyed by layer too: the same file decoded as `diffuse` and as
            // `rgba` is two different frames.
            let disk_file = disk.as_ref().and_then(|d| {
                seq.disk_file(d, seq_frame, layer.as_deref()).map(|f| (d.clone(), f))
            });

            {
                let mut view = buffer.slice(..).get_mapped_range_mut().expect("a mapped slot");
                let len = view.len();
                let mut wo = view.slice(..);
                // NVMe hit with no RAM tier to feed: unpack straight into
                // the mapped slot. Through an owned frame it cost a fresh
                // full-frame allocation and a single-threaded copy on top
                // of the read (7680² PNGs: ~450 MB a frame, both bigger
                // than the read itself).
                let mut direct = false;
                if cached.is_none() && !use_ram {
                    if let Some((d, f)) = &disk_file {
                        // SAFETY: as in the no-cache arm below — the
                        // write-only read never reads its destination.
                        let dst: &mut [f16] = unsafe {
                            std::slice::from_raw_parts_mut(
                                wo.as_raw_element_ptr().as_ptr().cast::<f16>(),
                                len / 2,
                            )
                        };
                        match DiskCache::read_frame_into_write_only(f, info, dst) {
                            Ok(true) => {
                                direct = true;
                                d.hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            }
                            Ok(false) => {}
                            Err(e) => log::warn!("disk cache read {}: {e:#}", f.display()),
                        }
                    }
                }
                if direct {
                    // Landed above.
                } else if let Some(data) = cached {
                    // RAM hit: one memcpy into the mapped slot.
                    let bytes: &[u8] = bytemuck::cast_slice(&data[..frame_f16]);
                    wo.slice(0..bytes.len()).copy_from_slice(bytes);
                    // In RAM but not on NVMe (its write was dropped by a
                    // full queue, or the tier was switched on since): owe
                    // it now, so the tier converges on what is played.
                    if opts.write {
                        if let Some((d, f)) = disk_file.as_ref().filter(|(_, f)| !f.exists()) {
                            owed = Some((d.clone(), f.clone(), data.clone()));
                        }
                    }
                } else if use_ram || disk_file.is_some() {
                    let mut owned = vec![f16::ZERO; frame_f16];
                    let mut have = false;
                    let mut from_disk = false;
                    if let Some((d, f)) = &disk_file {
                        match DiskCache::read_frame_into(f, info, &mut owned) {
                            Ok(true) => {
                                have = true;
                                from_disk = true;
                                d.hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            }
                            Ok(false) => {}
                            Err(e) => log::warn!("disk cache read {}: {e:#}", f.display()),
                        }
                    }
                    if !have {
                        if let Some((d, _)) = &disk_file {
                            d.misses.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                        match seq.decode_packed(&*decoder, seq_frame, layer.as_deref(), &mut owned) {
                            Ok(f) => {
                                have = true;
                                full = f;
                            }
                            Err(e) => log_decode_failure(&path, &e),
                        }
                    }
                    if !have {
                        // A file that exists but will not decode: usually
                        // one the renderer is still writing. Black, and
                        // NOT cached, so the next pass gets the finished
                        // file instead of a cached failure.
                        wo.slice(..).fill(0);
                        transient = true;
                    }
                    if have {
                        let bytes: &[u8] = bytemuck::cast_slice(&owned[..]);
                        wo.slice(0..bytes.len()).copy_from_slice(bytes);
                        let owned = Arc::new(owned);
                        if !from_disk && opts.write {
                            if let Some((d, f)) = &disk_file {
                                owed = Some((d.clone(), f.clone(), owned.clone()));
                            }
                        }
                        // A frame read off the NVMe tier joins the RAM tier even
                        // with "cache on play" off: nothing new is written to
                        // disk, and a reader left out of RAM re-read 275 MB
                        // off the SSD every tick of every loop (a uv plate
                        // that held a 6K comp at 9 fps).
                        if use_ram && (opts.write || from_disk) {
                            cache.lock().unwrap().insert_stamped((reader, seq_frame), owned, stamp);
                        }
                    }
                } else {
                    // No cache: decode straight into the mapped slot.
                    // SAFETY: wgpu 29 exposes mapped-write memory as WriteOnly
                    // because staging may be write-combined; plain non-atomic
                    // stores are fine, reads are the hazard. The decode path
                    // only ever writes (each pixel stored exactly once, no
                    // read-modify-write), and mapped ranges are 8-byte
                    // aligned so f16 alignment holds.
                    let dst: &mut [f16] = unsafe {
                        std::slice::from_raw_parts_mut(
                            wo.as_raw_element_ptr().as_ptr().cast::<f16>(),
                            len / 2,
                        )
                    };
                    let got = seq.decode_packed(&*decoder, seq_frame, layer.as_deref(), dst);
                    if let Ok(f) = &got {
                        full = f.clone();
                    }
                    if let Err(e) = got {
                        log::debug!("decode {} failed: {e:#}", path.display());
                        // Black, not whatever the slot happened to hold: a
                        // half-written frame must not show as a stale
                        // picture from somewhere else in the timeline.
                        // The frame still flows, so playback never wedges.
                        dst.fill(f16::ZERO);
                        transient = true;
                    }
                }
            }
            buffer.unmap();
            ring.note_unmapped(slot);
            let decode_ms = t0.elapsed().as_secs_f32() * 1e3;
            let _ = tx.send(DecodedFrame { reader, tick, slot, decode_ms, generation, transient, stamp });
            if let Some((d, f, data)) = owed {
                queue_cache_write(d, f, info, data);
            }
            // Fill the siblings' tiers from the frame this job decoded.
            let Some(full) = full else { return };
            for (r, s, o) in siblings {
                let (cache, full, layer, all_disk) = (cache.clone(), full.clone(), layer.clone(), all_disk.clone());
                rayon::spawn(move || {
                    if !o.write {
                        return;
                    }
                    let want_ram = o.ram && {
                        let c = cache.lock().unwrap();
                        c.budget > 0 && !c.has((r, seq_frame))
                    };
                    let file = if o.disk {
                        all_disk
                            .as_ref()
                            .and_then(|d| s.disk_file(d, seq_frame, layer.as_deref()).map(|f| (d.clone(), f)))
                            .filter(|(_, f)| !f.exists())
                    } else {
                        None
                    };
                    if !want_ram && file.is_none() {
                        return;
                    }
                    let sinfo = s.pack.storage_info(s.info);
                    let mut packed = vec![f16::ZERO; sinfo.row_stride_f16() * sinfo.height as usize];
                    s.pack.pack(&full, s.info, bytemuck::cast_slice_mut(&mut packed));
                    let packed = Arc::new(packed);
                    if want_ram {
                        cache.lock().unwrap().insert_stamped((r, seq_frame), packed.clone(), stamp);
                    }
                    if let Some((d, f)) = file {
                        queue_cache_write(d, f, sinfo, packed);
                    }
                });
            }
        });
    }

    /// (bytes used, frames cached, hits, misses)
    /// RAM residency for one reader: a flag per sequence frame, plus the
    /// bytes it occupies.
    pub fn ram_residency(&self, reader: usize) -> (Vec<bool>, u64) {
        let frames = self
            .sequences
            .get(reader)
            .map(|s| s.frames.len())
            .unwrap_or(0);
        let c = self.cache.lock().unwrap();
        (c.residency(reader, frames), c.reader_bytes(reader))
    }

    /// Take over the RAM tier of the rig this one replaces. Its entries are
    /// keyed by (reader, frame in the reader's WINDOW), so a trim, a slip or
    /// a load elsewhere in the comp used to start every Read cold again;
    /// here each frame is re-keyed by what it IS — file, frame in file,
    /// layer and proxy — and kept wherever this rig reads the same thing.
    /// Call before the first decode (straight after `reader_layers`).
    pub fn adopt_ram(&mut self, old: &Player) {
        type Key = (PathBuf, u64, Option<String>, u32, Pack);
        let key = |p: &Player, r: usize, i: u64| -> Option<Key> {
            let seq = p.sequences.get(r)?;
            let path = seq.frames.get(i as usize)?.clone();
            let layer = p.reader_layers.get(r).cloned().flatten();
            Some((path, seq.file_frame(i), layer, seq.proxy, seq.pack))
        };
        let mut here: HashMap<Key, Vec<(usize, u64)>> = HashMap::new();
        for r in 0..self.sequences.len() {
            if !self.reader_opts.get(r).copied().unwrap_or_default().ram {
                continue;
            }
            for i in 0..self.sequences[r].frames.len() as u64 {
                if let Some(k) = key(self, r, i) {
                    here.entry(k).or_default().push((r, i));
                }
            }
        }
        let taken = {
            let mut c = old.cache.lock().unwrap();
            c.total = 0;
            std::mem::take(&mut c.map)
        };
        let mut entries: Vec<_> = taken.into_iter().collect();
        // Oldest first, so the budget keeps the most recently used.
        entries.sort_by_key(|(_, e)| e.last_used);
        let mut c = self.cache.lock().unwrap();
        for ((r, i), e) in entries {
            let Some(to) = key(old, r, i).and_then(|k| here.get(&k)) else { continue };
            for &k in to {
                if e.data.len() * 2 == self.sequences[k.0].slot_bytes() as usize {
                    c.insert_stamped(k, e.data.clone(), e.stamp);
                }
            }
        }
    }

    /// Bytes per row of reader `r`'s frames in a staging slot: what an
    /// upload of its set copies with (`Pack::padded_row_bytes`).
    pub fn upload_row_bytes(&self, r: usize) -> u32 {
        self.sequences.get(r).map_or(0, |s| s.pack.padded_row_bytes(s.info))
    }

    /// Drop one reader's frames from the RAM tier (a Load's "clear cache").
    pub fn drop_ram(&self, reader: usize) {
        self.cache.lock().unwrap().drop_reader(reader);
    }

    pub fn cache_stats(&self) -> (usize, usize, u64, u64) {
        self.cache.lock().unwrap().stats()
    }

    pub fn cache_budget(&self) -> usize {
        self.cache.lock().unwrap().budget
    }

    /// Bytes needed to hold every frame of every reader's loop in the cache.
    pub fn loop_working_set_bytes(&self) -> u64 {
        self.sequences
            .iter()
            .map(|s| s.slot_bytes() * s.frames.len() as u64)
            .sum()
    }

    /// The frames for `tick`, if every ACTIVE reader has landed it: a set
    /// naming their slots. The frames STAY in `pending` — a set is shown,
    /// never taken — so the tick can be shown again for one GPU copy, and
    /// the slots are held against eviction until `frame_submitted`. An
    /// inactive reader's frame, if one landed while it was wanted, simply
    /// stays resident too.
    fn show(&mut self, tick: u64) -> Option<FrameSet> {
        if !self.have_all(tick) {
            self.stats.stalls += 1;
            return None;
        }
        let mut uploads = Vec::new();
        let mut once = Vec::new();
        for r in 0..self.sequences.len() {
            if !self.is_active(r) {
                continue;
            }
            let f = &self.pending[&(r, tick)];
            // The upload's extent: the reader's window, else its frame.
            uploads.push((f.reader, f.slot, self.sequences[r].pack.dims(self.sequences[r].info)));
            self.held.push(f.slot);
            if f.transient {
                once.push((r, tick));
            }
        }
        // A black stand-in is shown this once and then goes: the next pass
        // over its tick re-checks the disk (the render may have landed).
        for key in once {
            if let Some(f) = self.pending.remove(&key) {
                // Held this redraw: `drop_slot` defers it to the submit.
                self.drop_slot(f.slot);
            }
        }
        self.picture_stale = false;
        Some(FrameSet { tick, uploads })
    }

    /// Show `tick` exactly, moving nothing: a paused re-show after a reader
    /// woke (a view flip, a landed render, an adopted bake), or a bake's
    /// next frame. The caller has seeked to it, so `pump` is decoding it.
    pub fn show_set(&mut self, tick: u64) -> Option<FrameSet> {
        self.show(tick)
    }

    /// Playback: the frame after the one on screen, and the playhead moves
    /// past it. Pressing play on a paused frame starts with the NEXT one —
    /// the paused frame is already up, and repeating it would be a
    /// duplicate first frame.
    pub fn show_display_set(&mut self) -> Option<FrameSet> {
        // Step the PLAYHEAD past the frame on screen, not just the request:
        // the display tick is what `pump` keeps decoded, and a frame it
        // already has does not get a cursor. (Left on the shown frame with
        // the cursor parked behind it by a backward scrub, playback never
        // asked for the next tick and never advanced — `--test-seek`.)
        if self.shown == Some(self.display_tick) && !self.picture_stale {
            self.display_tick = self.next_tick(self.display_tick);
        }
        let tick = self.display_tick;
        let set = self.show(tick)?;
        self.display_tick = self.next_tick(tick);
        Some(set)
    }

    /// Scrubbing: the hand's tick if it has landed, else a landed tick
    /// BETWEEN the picture on screen (`self.shown`) and the hand — the
    /// nearest such to the hand. Never moves the playhead: the frame on
    /// screen stays resident, so there is nothing to step past.
    ///
    /// This is for scrubbing, and it is the difference between a picture
    /// that tracks your hand and one that does not move at all. A drag
    /// re-targets every ~16ms; a 4K plate takes far longer than that to
    /// decode, so insisting on the exact tick means the display waits for a
    /// frame that is already obsolete — measured at 2fps on 4K plates while
    /// the decoders were managing ~46 ticks/s of real work. Showing a
    /// decoded neighbour spends that work instead of discarding it.
    ///
    /// BETWEEN is the whole rule. The first version handed over whatever
    /// pending tick was nearest the playhead, and consumed it. `pending`
    /// during a drag is speculation plus decodes for ticks the hand already
    /// left, so every UI frame the exact tick was still in flight the
    /// next-nearest of those was shown, and the picture walked away from
    /// the hand one frame at a time; a backward step was shown the
    /// prefetched FORWARD neighbours first. `--test-scrubfid` measured a
    /// 60Hz backward drag at 49 regressions and 47 flips per 120 seeks (170
    /// pictures for 120 targets, up to 19 frames off), buffered or not —
    /// the junk was already decoded. Restricting candidates to the span the
    /// hand just crossed makes the picture monotonic by construction: it
    /// may LAG the hand, it never leads it, never jumps past it and never
    /// moves against it. With nothing in that span the caller keeps the
    /// picture it has, which is always the better frame.
    ///
    /// Playback must not use this — there, a neighbouring frame is simply
    /// the wrong frame.
    pub fn show_nearest_set(&mut self) -> Scrub {
        let tick = self.display_tick;
        if self.shown == Some(tick) && !self.picture_stale {
            return Scrub::OnScreen;
        }
        if let Some(set) = self.show(tick) {
            return Scrub::Exact(set);
        }
        let shown = self.shown;
        let Some(near) = self
            .pending
            .keys()
            .map(|(_, t)| *t)
            .filter(|t| self.in_scrub_span(*t) && Some(*t) != shown && self.have_all(*t))
            .min_by_key(|t| t.abs_diff(tick))
        else {
            return Scrub::NotYet;
        };
        match self.show(near) {
            Some(set) => Scrub::Between(set),
            None => Scrub::NotYet,
        }
    }

    /// The copies for every set shown this redraw are submitted: the slots
    /// may move again, and frames retired while held go back to the ring.
    pub fn frame_submitted(&mut self) {
        self.held.clear();
        for slot in self.release_after_submit.drain(..) {
            self.ring.release_after_copy(slot);
        }
    }

    fn have_all(&self, tick: u64) -> bool {
        (0..self.sequences.len())
            .all(|r| !self.is_active(r) || self.pending.contains_key(&(r, tick)))
    }

    /// (display_tick, issue_tick, pending entries, generation) — suite
    /// diagnostics for playback stalls.
    pub fn debug_state(&self) -> (u64, u64, usize, u64) {
        (self.display_tick, self.issue_tick, self.pending.len(), self.generation)
    }

    /// Decoded ticks at or ahead of the playhead (the HUD's frames-ahead).
    /// Resident frames behind it are cache, not runway.
    pub fn frames_ahead(&self) -> usize {
        let ticks: std::collections::HashSet<u64> = self
            .pending
            .keys()
            .map(|(_, t)| *t)
            .filter(|t| self.dist(*t).is_some_and(|k| k <= self.ahead))
            .collect();
        ticks.len()
    }

    /// Which of a reader's SEQUENCE frames sit decoded in the ring right
    /// now — the scrub cache proper. Indexed like `ram_residency`.
    pub fn ring_residency(&self, reader: usize) -> Vec<bool> {
        let frames = self.sequences.get(reader).map(|s| s.frames.len()).unwrap_or(0);
        let mut out = vec![false; frames];
        for (r, t) in self.pending.keys() {
            if *r == reader {
                if let Some(i) = self.source_index(reader, *t) {
                    if (i as usize) < frames {
                        out[i as usize] = true;
                    }
                }
            }
        }
        out
    }
}

/// A frame that will not decode is retried on every pass (it may be one the
/// renderer is still writing), so its error is logged ONCE per path and
/// message; the repeats go to debug. A whole sequence of undecodable files
/// used to put one error line per frame per pass in the log.
fn log_decode_failure(path: &Path, e: &anyhow::Error) {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static SEEN: OnceLock<Mutex<HashMap<std::path::PathBuf, String>>> = OnceLock::new();
    let msg = format!("{e:#}");
    let first = {
        let mut seen = SEEN.get_or_init(Default::default).lock().unwrap();
        if seen.len() > 4096 {
            seen.clear();
        }
        seen.insert(path.to_path_buf(), msg.clone()).as_ref() != Some(&msg)
    };
    if first {
        log::error!("decode {} failed: {msg}", path.display());
    } else {
        log::debug!("decode {} failed: {msg}", path.display());
    }
}

/// A background fill of the cache tiers: every frame of some readers'
/// sequences decoded once, so playback starts warm. The frames land exactly
/// where `spawn_decode` would look for them — the RAM tier when the reader
/// uses it, the NVMe tier when it uses that — so nothing downstream knows
/// a precache happened. It never touches the ring, the playhead or
/// `pending`, and it runs on its own threads, not rayon's: the pool is the
/// playback decoder, and a flood of precache jobs there would queue every
/// frame the display asks for behind them.
///
/// The RAM tier's budget is respected through the cache's own insert (a
/// frame that will not fit is not kept); `ram_full` says that happened.
/// Absent frames (a range running past its source) are skipped and never
/// cached, the same rule as playback.
pub struct Precache {
    /// Per group (one per Read, as the app asked): (frames to do, frames done).
    pub groups: Vec<(usize, Arc<std::sync::atomic::AtomicUsize>)>,
    pub cancel: Arc<std::sync::atomic::AtomicBool>,
    /// A frame the RAM tier would not take: the budget is full.
    pub ram_full: Arc<std::sync::atomic::AtomicBool>,
    /// Frames that exist but would not decode.
    pub failed: Arc<std::sync::atomic::AtomicUsize>,
    /// Worker threads still running; 0 = finished (or cancelled).
    pub running: Arc<std::sync::atomic::AtomicUsize>,
}

impl Precache {
    pub fn total(&self) -> usize {
        self.groups.iter().map(|g| g.0).sum()
    }
    pub fn done(&self) -> usize {
        use std::sync::atomic::Ordering::Relaxed;
        self.groups.iter().map(|g| g.1.load(Relaxed).min(g.0)).sum()
    }
    pub fn finished(&self) -> bool {
        self.running.load(std::sync::atomic::Ordering::Acquire) == 0
    }
    pub fn stop(&self) {
        self.cancel.store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

/// One distinct thing to decode: readers that read the same frames of the
/// same files with the same layer share it (two time paths into one Read),
/// so each frame is decoded once and handed to every reader's RAM key.
struct PrecacheUnit {
    group: usize,
    readers: Vec<usize>,
    seq: Arc<Sequence>,
    layer: Option<String>,
    ram: bool,
    disk: Option<Arc<DiskCache>>,
}

impl Player {
    /// What a precache of these readers would put in the RAM tier, in
    /// bytes, counting each distinct sequence once — the number to hold
    /// against `cache_budget()` before starting.
    pub fn precache_ram_bytes(&self, readers: &[usize]) -> u64 {
        self.precache_units(&[readers.to_vec()])
            .iter()
            .filter(|u| u.ram)
            .map(|u| {
                u.seq.slot_bytes() * u.seq.frames.len() as u64
            })
            .sum()
    }

    fn precache_units(&self, groups: &[Vec<usize>]) -> Vec<PrecacheUnit> {
        let budget = self.cache.lock().unwrap().budget;
        let mut units: Vec<PrecacheUnit> = Vec::new();
        for (g, readers) in groups.iter().enumerate() {
            for &r in readers {
                let Some(seq) = self.sequences.get(r) else { continue };
                let opts = self.reader_opts.get(r).copied().unwrap_or_default();
                let layer = self.reader_layers.get(r).cloned().flatten();
                let ram = opts.ram && budget > 0;
                let disk = if opts.disk { self.disk.clone() } else { None };
                if !ram && disk.is_none() {
                    continue; // no tier to fill
                }
                if let Some(u) = units.iter_mut().find(|u| {
                    u.group == g
                        && u.layer == layer
                        && u.ram == ram
                        && u.disk.is_some() == disk.is_some()
                        && u.seq.frames == seq.frames
                        && u.seq.offset == seq.offset
                        && u.seq.proxy == seq.proxy
                }) {
                    u.readers.push(r);
                } else {
                    units.push(PrecacheUnit {
                        group: g,
                        readers: vec![r],
                        seq: seq.clone(),
                        layer,
                        ram,
                        disk,
                    });
                }
            }
        }
        units
    }

    /// Start filling the tiers for `groups` (each a list of reader indices,
    /// reported separately). `threads` decode in parallel; the caller keeps
    /// the handle and polls it. The job holds THIS player's RAM tier, so a
    /// rig rebuild (a new player, a new tier) orphans it — stop it then.
    pub fn precache(&self, groups: Vec<Vec<usize>>, threads: usize) -> Precache {
        self.precache_files(groups, threads, Vec::new())
    }

    /// [`Self::precache`], plus, per group, the WHOLE file list of its
    /// source (`(frames, offset)`, the Load's every file rather than its
    /// in..out): those frames go to the NVMe tier only — the RAM tier is
    /// keyed by the reader's own frames, which are the in..out.
    pub fn precache_files(
        &self,
        groups: Vec<Vec<usize>>,
        threads: usize,
        full: Vec<Option<(Vec<PathBuf>, u64)>>,
    ) -> Precache {
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        let mut units = self.precache_units(&groups);
        for (g, f) in full.into_iter().enumerate() {
            let (Some((frames, offset)), Some(disk)) = (f, self.disk.clone()) else { continue };
            let Some(&r) = groups.get(g).and_then(|rs| rs.first()) else { continue };
            let Some(seq) = self.sequences.get(r) else { continue };
            if !self.reader_opts.get(r).copied().unwrap_or_default().disk {
                continue;
            }
            let full = Sequence { frames, offset, info: seq.info, proxy: seq.proxy, movie: seq.movie, pack: seq.pack };
            units.push(PrecacheUnit {
                group: g,
                readers: vec![r],
                seq: Arc::new(full),
                layer: self.reader_layers.get(r).cloned().flatten(),
                ram: false,
                disk: Some(disk),
            });
        }
        let units = Arc::new(units);
        let mut per_group = vec![0usize; groups.len()];
        for u in units.iter() {
            per_group[u.group] += u.seq.frames.len();
        }
        let job = Precache {
            groups: per_group
                .into_iter()
                .map(|n| (n, Arc::new(AtomicUsize::new(0))))
                .collect(),
            cancel: Arc::new(AtomicBool::new(false)),
            ram_full: Arc::new(AtomicBool::new(false)),
            failed: Arc::new(AtomicUsize::new(0)),
            running: Arc::new(AtomicUsize::new(0)),
        };
        // Work items in frame order ACROSS units, so every Read warms from
        // its start together rather than one after the other.
        let mut items: Vec<(usize, usize)> = Vec::new();
        let longest = units.iter().map(|u| u.seq.frames.len()).max().unwrap_or(0);
        for f in 0..longest {
            for (ui, u) in units.iter().enumerate() {
                if f < u.seq.frames.len() {
                    items.push((ui, f));
                }
            }
        }
        let items = Arc::new(items);
        let next = Arc::new(AtomicUsize::new(0));
        let threads = threads.max(1).min(items.len().max(1));
        job.running.store(threads, Ordering::Release);
        for _ in 0..threads {
            let (units, items, next) = (units.clone(), items.clone(), next.clone());
            let cache = self.cache.clone();
            let decoder = self.decoder.clone();
            let done: Vec<_> = job.groups.iter().map(|g| g.1.clone()).collect();
            let (cancel, ram_full, failed, running) = (
                job.cancel.clone(),
                job.ram_full.clone(),
                job.failed.clone(),
                job.running.clone(),
            );
            let spawned = std::thread::Builder::new()
                .name("precache".into())
                .spawn(move || {
                    loop {
                        if cancel.load(Ordering::Relaxed) {
                            break;
                        }
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        let Some(&(ui, f)) = items.get(i) else { break };
                        let u = &units[ui];
                        precache_one(u, f as u64, &cache, &*decoder, &ram_full, &failed);
                        done[u.group].fetch_add(1, Ordering::Relaxed);
                    }
                    running.fetch_sub(1, Ordering::AcqRel);
                });
            if spawned.is_err() {
                job.running.fetch_sub(1, Ordering::AcqRel);
            }
        }
        job
    }
}

fn precache_one(
    u: &PrecacheUnit,
    f: u64,
    cache: &Mutex<FrameCache>,
    decoder: &dyn ExrDecoder,
    ram_full: &std::sync::atomic::AtomicBool,
    failed: &std::sync::atomic::AtomicUsize,
) {
    use std::sync::atomic::Ordering::Relaxed;
    let seq = &u.seq;
    let path = &seq.frames[f as usize];
    // Not there (yet): never cached, exactly as playback treats it.
    if !crate::vfs::exists(path) {
        return;
    }
    let info = seq.pack.storage_info(seq.info);
    let disk_file = u.disk.as_ref().and_then(|d| {
        seq.disk_file(d, f, u.layer.as_deref()).map(|file| (d.clone(), file))
    });
    // Once the RAM tier is full, filling it is over: keep going only for the
    // NVMe tier (and skip a frame it already has).
    let want_ram = u.ram && !ram_full.load(Relaxed) && {
        let c = cache.lock().unwrap();
        u.readers.iter().any(|r| !c.map.contains_key(&(*r, f)))
    };
    let on_disk = disk_file.as_ref().is_some_and(|(_, file)| file.exists());
    if !want_ram && (disk_file.is_none() || on_disk) {
        return;
    }
    let frame_f16 = info.row_stride_f16() * info.height as usize;
    let mut owned = vec![f16::ZERO; frame_f16];
    let mut from_disk = false;
    if let (true, Some((d, file))) = (on_disk, &disk_file) {
        if let Ok(true) = DiskCache::read_frame_into(file, info, &mut owned) {
            from_disk = true;
            d.hits.fetch_add(1, Relaxed);
        }
    }
    if !from_disk {
        if let Err(e) = seq.decode_packed(decoder, f, u.layer.as_deref(), &mut owned).map(|_| ()) {
            // Likely a frame still being written: not cached, counted.
            log::debug!("precache decode {} failed: {e:#}", path.display());
            failed.fetch_add(1, Relaxed);
            return;
        }
        if let Some((d, file)) = &disk_file {
            d.misses.fetch_add(1, Relaxed);
            if let Err(e) = d.write_frame(file, info, &owned) {
                log::warn!("precache disk write {}: {e:#}", file.display());
            }
        }
    }
    if want_ram {
        let data = Arc::new(owned);
        let mut c = cache.lock().unwrap();
        for &r in &u.readers {
            c.insert((r, f), data.clone());
            if !c.map.contains_key(&(r, f)) {
                ram_full.store(true, Relaxed);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A loop much larger than the budget must settle into a pinned resident
    /// set with steady hits — plain LRU degrades to zero here.
    #[test]
    fn cache_pins_resident_set_under_cyclic_pressure() {
        let entry_len = 1000usize; // 2 KB per entry
        let budget = 20 * 2000; // room for 20 of 200 entries
        let mut c = FrameCache::new(budget);
        let loop_frames = 100u64;
        let readers = 2usize;
        c.cycle_lookups = readers as u64 * loop_frames;
        let data = Arc::new(vec![f16::ZERO; entry_len]);
        let mut last_loop_hits = 0;
        for _ in 0..6 {
            let before = c.hits;
            for t in 0..loop_frames {
                for r in 0..readers {
                    if c.get((r, t)).is_none() {
                        c.insert((r, t), data.clone());
                    }
                }
            }
            last_loop_hits = c.hits - before;
        }
        assert!(last_loop_hits >= 15, "cyclic hit rate collapsed: {last_loop_hits} per loop");
        assert!(c.total <= budget);
    }

    /// The time nodes move which frame a reader decodes, nothing else.
    #[test]
    fn time_maps_remap_the_source_frame_only() {
        let id = TimeMap::default();
        assert_eq!(source_frame_of(&id, 10, 3), 3);
        // Four frames ahead, HELD at the reader's last frame past its end
        // (a reader that has run out does not restart).
        let ahead = TimeMap::affine(1.0, 4.0);
        assert_eq!(source_frame_of(&ahead, 10, 3), 7);
        assert_eq!(source_frame_of(&ahead, 10, 8), 9);
        // Half speed holds each source frame for two ticks.
        let slow = TimeMap::affine(0.5, 0.0);
        let played: Vec<u64> = (0..6).map(|t| source_frame_of(&slow, 10, t)).collect();
        assert_eq!(played, [0, 1, 1, 2, 2, 3]);
        // Running backwards from a start frame, and never off the front.
        let rev = TimeMap::affine(-1.0, 4.0);
        assert_eq!(
            (0..6).map(|t| source_frame_of(&rev, 10, t)).collect::<Vec<_>>(),
            [4, 3, 2, 1, 0, 0]
        );
        // A zero-length sequence must not divide by zero.
        assert_eq!(source_frame_of(&ahead, 0, 3), 0);
        // Two instances of the SAME sequence, one shifted: both frames are
        // live at the same tick — that is the whole point of instancing.
        assert_eq!(
            (source_frame_of(&id, 10, 7), source_frame_of(&ahead, 10, 5)),
            (7, 9)
        );
    }

    /// A Read's window can open before its first file and close past its
    /// last: those frames get the names they WOULD have (so a render that
    /// lands later is picked up) and nothing pretends to be them.
    #[test]
    fn a_window_names_the_frames_it_does_not_have() {
        let seq = Sequence {
            frames: (1001..=1003).map(|f| PathBuf::from(format!("/p/plate.{f:04}.exr"))).collect(),
            info: ImageInfo { width: 1, height: 1 },
            proxy: 1,
            movie: false,
            offset: 0,
            pack: Pack::default(),
        };
        let w = seq.window(-2, 7);
        let names: Vec<String> =
            w.frames.iter().map(|p| p.file_name().unwrap().to_string_lossy().into_owned()).collect();
        assert_eq!(
            names,
            [
                "plate.0999.exr",
                "plate.1000.exr",
                "plate.1001.exr",
                "plate.1002.exr",
                "plate.1003.exr",
                "plate.1004.exr",
                "plate.1005.exr"
            ]
        );
        // Below frame zero there is no name to give: a path that cannot exist.
        let early = Sequence {
            frames: vec![PathBuf::from("/p/plate.0001.exr")],
            ..seq.window(0, 1)
        }
        .window(-3, 1);
        assert!(early.frames[0].to_string_lossy().ends_with(".wobble-missing"));
    }

    /// Outside its own frames a clip answers by its edge behaviour, the
    /// same four compositors usually offer. A 1001-placed 4-frame clip, asked for 999..1007.
    #[test]
    fn a_clip_answers_outside_its_range_by_its_edges() {
        let placed = TimeMap::affine(1.0, -1001.0);
        let run = |before: ReadEdge, after: ReadEdge| -> Vec<Option<u64>> {
            let e = ReadEdges { before, after };
            (999..=1007).map(|t| source_index_of(&placed, 4, t, e)).collect()
        };
        let s = Some;
        assert_eq!(
            run(ReadEdge::Hold, ReadEdge::Hold),
            [s(0), s(0), s(0), s(1), s(2), s(3), s(3), s(3), s(3)]
        );
        assert_eq!(
            run(ReadEdge::Loop, ReadEdge::Loop),
            [s(2), s(3), s(0), s(1), s(2), s(3), s(0), s(1), s(2)]
        );
        assert_eq!(
            run(ReadEdge::Bounce, ReadEdge::Bounce),
            [s(2), s(1), s(0), s(1), s(2), s(3), s(2), s(1), s(0)]
        );
        assert_eq!(
            run(ReadEdge::Black, ReadEdge::Black),
            [None, None, s(0), s(1), s(2), s(3), None, None, None]
        );
        // The two ends are independent.
        assert_eq!(run(ReadEdge::Black, ReadEdge::Hold)[0], None);
        assert_eq!(run(ReadEdge::Black, ReadEdge::Hold)[8], s(3));
        // A one-frame clip bounces onto itself, not into a divide by zero.
        let e = ReadEdges { before: ReadEdge::Bounce, after: ReadEdge::Bounce };
        assert_eq!(source_index_of(&placed, 1, 1005, e), s(0));
    }

    /// A tick is a frame number and the loop is explicit: the playhead runs
    /// to the out point and comes back to the in point, and "behind the
    /// playhead" means "the loop will not reach it again".
    #[test]
    fn loop_span_wraps_the_playhead_not_the_media() {
        assert_eq!(dist_in(None, 5, 9), Some(4));
        assert_eq!(dist_in(None, 5, 4), None);
        let span = Some((10, 14));
        // Inside the loop: forward distance, wrapping at the out point.
        assert_eq!(dist_in(span, 10, 14), Some(4));
        assert_eq!(dist_in(span, 14, 10), Some(1));
        assert_eq!(dist_in(span, 12, 11), Some(4));
        // Outside it: reachable only from where the playhead actually is.
        assert_eq!(dist_in(span, 12, 20), None);
        assert_eq!(dist_in(span, 7, 9), Some(2));
        assert_eq!(dist_in(span, 7, 12), Some(5));
        assert_eq!(dist_in(span, 20, 20), Some(0));
        assert_eq!(dist_in(span, 20, 10), Some(1));
    }
}
