//! Hover-scrub for the bin's tiles: tiny frames decoded off the UI thread,
//! on decoders of their own (never the viewers' frame servers, so a tile
//! scrub cannot disturb playback), boxed down to `MAX_W` x `MAX_H` RGBA
//! (`media::thumb`) and kept as textures under their own byte budget.
//!
//! - A tile on screen asks for its POSTER (a third of the way in).
//! - The first hover over a tile starts its STRIP: ONE forward decode pass
//!   over the clip keeping every `step`-th frame (at most `STRIP` of them;
//!   every frame of a short clip). A long-GOP movie decodes forward fast
//!   and seeks slowly, so a pass beats random seeks many times over; while
//!   it runs a scrub shows the nearest frame already in. One strip at a
//!   time, so a worker is always free for the frame under the pointer.
//! - On an intra clip (ProRes, images) the frame under the pointer is
//!   asked for EXACTLY, in one slot: a new hover position replaces the old
//!   request (stale work is never done). A long-GOP clip scrubs from its
//!   strip alone (a seek there is ~0.5 s at 4K): exact up to `STRIP`
//!   frames, within `step` frames past that.
//!
//! Priority: the hover frame, then strips, then posters. Two workers.

use egui::{ColorImage, TextureHandle, TextureOptions};
use media::SourceRef;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

pub const MAX_W: u32 = 192;
pub const MAX_H: u32 = 108;
/// Frames kept from a clip's strip pass (a clip this short keeps all).
pub const STRIP: u64 = 240;
/// Bytes of tile textures kept (RGBA8): ~1500 frames at 192x108.
pub const BUDGET: usize = 96 << 20;
const WORKERS: usize = 3;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Pos {
    /// A fraction of the clip (the poster, before the length is known).
    Frac(f32),
    Index(u64),
    /// The strip pass.
    Strip,
}

#[derive(Clone, Debug)]
struct Job {
    path: PathBuf,
    pos: Pos,
    asked: Instant,
    hover: bool,
}

#[derive(Default)]
struct Queue {
    hover: Option<Job>,
    strip: VecDeque<Job>,
    posters: VecDeque<Job>,
    /// A strip pass is running (one at a time).
    striping: bool,
    quit: bool,
}

impl Queue {
    fn next(&mut self) -> Option<Job> {
        if let Some(j) = self.hover.take().or_else(|| self.posters.pop_front()) {
            return Some(j);
        }
        if !self.striping {
            let j = self.strip.pop_front();
            self.striping = j.is_some();
            return j;
        }
        None
    }
}

/// What a clip is, as far as a tile needs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Info {
    pub frames: u64,
    pub fps: f64,
    /// Every frame decodes on its own (ProRes, images): an exact frame is
    /// a quick decode. A long-GOP movie's is a seek (~0.5 s at 4K), so it
    /// scrubs from its strip instead.
    pub intra: bool,
}

impl Info {
    /// The strip keeps every `step`-th frame.
    pub fn step(&self) -> u64 {
        self.frames.div_ceil(STRIP).max(1)
    }
}

struct Done {
    path: PathBuf,
    /// `None`: the clip would not open.
    info: Option<Info>,
    frame: u64,
    poster: bool,
    img: Option<ColorImage>,
    asked: Instant,
    hover: bool,
    ms: f32,
}

type Shared = Arc<(Mutex<Queue>, Condvar)>;

pub struct Scrub {
    q: Shared,
    rx: Receiver<Done>,
    /// Per clip: its info, its poster frame, its frames by index.
    clips: HashMap<PathBuf, Clip>,
    /// (path, frame) in the order they were last used, for the budget.
    lru: VecDeque<(PathBuf, u64)>,
    bytes: usize,
    /// Asked for and not yet back.
    pending: HashSet<(PathBuf, u64)>,
    posters_asked: HashSet<PathBuf>,
    strips_asked: HashSet<PathBuf>,
    /// Pointer move -> exact frame on the tile, in ms (the last few).
    pub latency: VecDeque<f32>,
    /// Decode + box-down time per frame, in ms (the last few).
    pub decode_ms: VecDeque<f32>,
}

#[derive(Default)]
struct Clip {
    info: Option<Option<Info>>,
    poster: Option<u64>,
    frames: BTreeMap<u64, TextureHandle>,
}

impl Default for Scrub {
    fn default() -> Self {
        Self::new()
    }
}

impl Scrub {
    pub fn new() -> Self {
        let q: Shared = Arc::new((Mutex::new(Queue::default()), Condvar::new()));
        let (tx, rx) = channel();
        let sources: Arc<Mutex<HashMap<PathBuf, Option<SourceRef>>>> = Arc::default();
        for i in 0..WORKERS {
            let (q, tx, sources) = (q.clone(), tx.clone(), sources.clone());
            let _ = std::thread::Builder::new().name(format!("scrub-tiles-{i}")).spawn(move || worker(q, tx, sources));
        }
        Scrub { q, rx, clips: HashMap::new(), lru: VecDeque::new(), bytes: 0, pending: HashSet::new(), posters_asked: HashSet::new(), strips_asked: HashSet::new(), latency: VecDeque::new(), decode_ms: VecDeque::new() }
    }

    fn push(&self, f: impl FnOnce(&mut Queue)) {
        let (m, cv) = &*self.q;
        if let Ok(mut g) = m.lock() {
            f(&mut g);
        }
        cv.notify_all();
    }

    /// Take what the workers finished into textures. Returns true when
    /// something landed (repaint).
    pub fn poll(&mut self, ctx: &egui::Context) -> bool {
        let mut any = false;
        while let Ok(d) = self.rx.try_recv() {
            any = true;
            self.pending.remove(&(d.path.clone(), d.frame));
            let c = self.clips.entry(d.path.clone()).or_default();
            c.info = Some(d.info);
            let Some(img) = d.img else { continue };
            if d.poster {
                c.poster = Some(d.frame);
            }
            let bytes = img.pixels.len() * 4;
            let tex = ctx.load_texture(format!("tile:{}:{}", d.path.display(), d.frame), img, TextureOptions::LINEAR);
            if c.frames.insert(d.frame, tex).is_none() {
                self.bytes += bytes;
            }
            self.lru.push_back((d.path.clone(), d.frame));
            push_capped(&mut self.decode_ms, d.ms);
            if d.hover {
                push_capped(&mut self.latency, d.asked.elapsed().as_secs_f32() * 1000.0);
            }
            while self.bytes > BUDGET {
                let Some((p, f)) = self.lru.pop_front() else { break };
                if let Some(c) = self.clips.get_mut(&p) {
                    if c.poster != Some(f) {
                        if let Some(t) = c.frames.remove(&f) {
                            let [w, h] = t.size();
                            self.bytes = self.bytes.saturating_sub(w * h * 4);
                        }
                    }
                }
            }
        }
        any
    }

    pub fn info(&self, path: &Path) -> Option<Info> {
        self.clips.get(path).and_then(|c| c.info).flatten()
    }

    /// The clip would not open.
    pub fn broken(&self, path: &Path) -> bool {
        self.clips.get(path).is_some_and(|c| c.info == Some(None))
    }

    /// A visible tile: its poster (asked for once).
    pub fn poster(&mut self, path: &Path) -> Option<TextureHandle> {
        if !self.posters_asked.contains(path) {
            self.posters_asked.insert(path.to_path_buf());
            let j = Job { path: path.to_path_buf(), pos: Pos::Frac(1.0 / 3.0), asked: Instant::now(), hover: false };
            self.push(|q| q.posters.push_back(j));
        }
        let c = self.clips.get(path)?;
        c.poster.and_then(|f| c.frames.get(&f)).or_else(|| c.frames.values().next()).cloned()
    }

    /// The pointer is at `frac` (0..1) across a tile: the frame index it
    /// means, and the best texture to show now — the exact frame when it is
    /// here, else the nearest one we have (and the exact one is asked for,
    /// replacing any older hover request).
    pub fn hover(&mut self, path: &Path, frac: f32) -> (Option<u64>, Option<TextureHandle>, bool) {
        let Some(info) = self.info(path) else {
            return (None, self.poster(path), false);
        };
        let n = info.frames.max(1);
        let f = ((frac.clamp(0.0, 1.0) * (n - 1) as f32).round() as u64).min(n - 1);
        if !self.strips_asked.contains(path) {
            self.strips_asked.insert(path.to_path_buf());
            let j = Job { path: path.to_path_buf(), pos: Pos::Strip, asked: Instant::now(), hover: false };
            self.push(|q| q.strip.push_back(j));
        }
        let c = self.clips.get(path);
        if let Some(t) = c.and_then(|c| c.frames.get(&f)) {
            let t = t.clone();
            self.touch(path, f);
            return (Some(f), Some(t), true);
        }
        let key = (path.to_path_buf(), f);
        if info.intra && !self.pending.contains(&key) {
            // Replace the hover slot: the old request is stale.
            if let Ok(g) = self.q.0.lock() {
                if let Some(old) = &g.hover {
                    if let Pos::Index(i) = old.pos {
                        self.pending.remove(&(old.path.clone(), i));
                    }
                }
            }
            self.pending.insert(key);
            let j = Job { path: path.to_path_buf(), pos: Pos::Index(f), asked: Instant::now(), hover: true };
            self.push(|q| q.hover = Some(j));
        }
        let near = c.and_then(|c| {
            let below = c.frames.range(..=f).next_back();
            let above = c.frames.range(f..).next();
            match (below, above) {
                (Some(b), Some(a)) => Some(if f - b.0 <= a.0 - f { b.1 } else { a.1 }),
                (b, a) => b.or(a).map(|x| x.1),
            }
            .cloned()
        });
        (Some(f), near.or_else(|| self.poster(path)), false)
    }

    fn touch(&mut self, path: &Path, f: u64) {
        // Cheap LRU refresh: re-append (duplicates are harmless; eviction
        // skips frames already gone).
        if self.lru.back().is_none_or(|(p, g)| p != path || *g != f) {
            self.lru.push_back((path.to_path_buf(), f));
            if self.lru.len() > 8192 {
                self.lru.pop_front();
            }
        }
    }

    /// Frames we hold for `path` (tests).
    pub fn held(&self, path: &Path) -> usize {
        self.clips.get(path).map_or(0, |c| c.frames.len())
    }

    /// The exact frame `f` of `path` is here (tests).
    pub fn has(&self, path: &Path, f: u64) -> bool {
        self.clips.get(path).is_some_and(|c| c.frames.contains_key(&f))
    }
}

impl Drop for Scrub {
    fn drop(&mut self) {
        self.push(|q| q.quit = true);
    }
}

fn push_capped(v: &mut VecDeque<f32>, x: f32) {
    v.push_back(x);
    if v.len() > 64 {
        v.pop_front();
    }
}

/// Every `step`-th frame of the clip in one forward pass.
fn strip(src: &dyn media::Source, job: &Job, info: Info, tx: &Sender<Done>, q: &Shared) {
    struct S<'a> {
        step: u64,
        n: u64,
        job: &'a Job,
        info: Info,
        tx: &'a Sender<Done>,
        q: &'a Shared,
        t: Instant,
    }
    impl S<'_> {
        fn send(&mut self, frame: u64, fr: &media::Frame) {
            let (w, h, px) = media::thumb::thumbnail(fr, MAX_W, MAX_H);
            let img = ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &px);
            let ms = self.t.elapsed().as_secs_f32() * 1000.0;
            self.t = Instant::now();
            let _ = self.tx.send(Done { path: self.job.path.clone(), info: Some(self.info), frame, poster: false, img: Some(img), asked: self.job.asked, hover: false, ms });
        }
        fn stop(&self) -> bool {
            self.q.0.lock().map_or(true, |g| g.quit)
        }
    }
    impl media::source::Sink for S<'_> {
        fn wants(&mut self, frame: u64) -> bool {
            frame % self.step == 0
        }
        fn passing(&mut self, frame: u64, picture: media::Frame) {
            self.send(frame, &picture);
        }
        fn done(&mut self, frame: u64, picture: anyhow::Result<media::Frame>) -> Option<u64> {
            if let Ok(p) = picture {
                self.send(frame, &p);
            }
            let next = frame + self.step;
            (next < self.n && !self.stop()).then_some(next)
        }
    }
    let n = info.frames;
    let step = info.step();
    let mut s = S { step, n, job, info, tx, q, t: Instant::now() };
    src.stream(0, &mut s);
}

fn worker(q: Shared, tx: Sender<Done>, sources: Arc<Mutex<HashMap<PathBuf, Option<SourceRef>>>>) {
    loop {
        let job = {
            let (m, cv) = &*q;
            let Ok(mut g) = m.lock() else { return };
            loop {
                if g.quit {
                    return;
                }
                if let Some(j) = g.next() {
                    break j;
                }
                let Ok(ng) = cv.wait(g) else { return };
                g = ng;
            }
        };
        let src = {
            let cached = sources.lock().ok().and_then(|s| s.get(&job.path).cloned());
            match cached {
                Some(s) => s,
                None => {
                    let s = media::open(&job.path).ok();
                    if let Ok(mut m) = sources.lock() {
                        m.insert(job.path.clone(), s.clone());
                    }
                    s
                }
            }
        };
        let Some(src) = src else {
            let _ = tx.send(Done { path: job.path, info: None, frame: 0, poster: false, img: None, asked: job.asked, hover: job.hover, ms: 0.0 });
            continue;
        };
        let si = src.info();
        let info = Info { frames: si.frames.max(1), fps: si.fps(), intra: si.intra };
        let n = info.frames;
        let (frame, poster) = match job.pos {
            Pos::Frac(f) => (((n - 1) as f32 * f) as u64, true),
            Pos::Index(i) => (i.min(n - 1), false),
            Pos::Strip => {
                strip(&*src, &job, info, &tx, &q);
                if let Ok(mut g) = q.0.lock() {
                    g.striping = false;
                }
                q.1.notify_all();
                continue;
            }
        };
        let t0 = Instant::now();
        let img = src.decode(frame).ok().map(|fr| {
            let (w, h, px) = media::thumb::thumbnail(&fr, MAX_W, MAX_H);
            ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &px)
        });
        if tx.send(Done { path: job.path, info: Some(info), frame, poster, img, asked: job.asked, hover: job.hover, ms: t0.elapsed().as_secs_f32() * 1000.0 }).is_err() {
            return;
        }
    }
}
