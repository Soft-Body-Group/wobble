//! The raw bytes of a sequence's frame files, read ahead of decoding.
//!
//! Decode workers are few (they want cores); file reads want MANY at once
//! when the files are on a share, because each read waits a round trip
//! before its bytes start and a single stream rarely fills the link. So a
//! sequence keeps its own pool of reader threads and a cache of raw file
//! bytes (separate from decoded frames, budgeted in bytes), and reads
//! ahead in the play direction as deep as that budget allows — far deeper
//! than the decoded-frame prefetch, which is held to a few dozen frames
//! of RAM-hungry pictures. A decode then finds its bytes already here.
//!
//! - `want(list, playing)`: the frames the decode prefetch wants, most
//!   urgent first; playing, the read-ahead runs on past them. Anything
//!   queued and no longer wanted is dropped (a scrub cancels it); reads
//!   already running land and are kept.
//! - Frames on a network share also go to the local disk cache
//!   (`netcache`), and are read from it next time.

use crate::netcache;
use crate::netio;
use anyhow::Result;
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// What the read-ahead is doing, for the HUD.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct NetStats {
    /// The files are on a network share (or the simulator is on).
    pub network: bool,
    /// Bytes per second read over the last two seconds.
    pub bytes_per_sec: f64,
    /// Frames read ahead and waiting.
    pub ahead: usize,
    /// A frame file's typical size.
    pub frame_bytes: usize,
    /// Reads served by the local disk cache / by the share.
    pub local_hits: u64,
    pub remote_reads: u64,
    pub queued: usize,
}

impl NetStats {
    /// Frames per second the reads have been managing (0 when idle).
    pub fn fps(&self) -> f64 {
        if self.frame_bytes == 0 { 0.0 } else { self.bytes_per_sec / self.frame_bytes as f64 }
    }
}

struct State {
    raw: HashMap<u64, Arc<Vec<u8>>>,
    lru: VecDeque<u64>,
    bytes: usize,
    budget: usize,
    queue: VecDeque<u64>,
    inflight: HashSet<u64>,
    /// The current read-ahead: never evicted for another frame.
    protect: HashSet<u64>,
    est: usize,
    log: VecDeque<(Instant, usize)>,
    local_hits: u64,
    remote_reads: u64,
    quit: bool,
}

struct Shared {
    st: Mutex<State>,
    work: Condvar,
    landed: Condvar,
    path_of: Box<dyn Fn(u64) -> PathBuf + Send + Sync>,
    frames: u64,
    network: bool,
}

pub struct Fetcher {
    sh: Arc<Shared>,
    threads: Vec<JoinHandle<()>>,
}

impl Fetcher {
    /// `path_of(i)`: frame i's file; `frames`: how many; `sample`: one of
    /// them (to tell a share from a local disk).
    pub fn new(frames: u64, sample: PathBuf, path_of: impl Fn(u64) -> PathBuf + Send + Sync + 'static) -> Self {
        let network = netio::is_network(&sample);
        let mb: usize = crate::sys::var("RAW_MB").ok().and_then(|s| s.parse().ok()).unwrap_or(if network { 2048 } else { 768 });
        let n: usize = crate::sys::var("READERS").ok().and_then(|s| s.parse().ok()).unwrap_or(if network { 12 } else { 8 });
        let est = std::fs::metadata(&sample).map(|m| m.len() as usize).unwrap_or(8 << 20).max(1);
        let sh = Arc::new(Shared {
            st: Mutex::new(State {
                raw: HashMap::new(),
                lru: VecDeque::new(),
                bytes: 0,
                budget: mb << 20,
                queue: VecDeque::new(),
                inflight: HashSet::new(),
                protect: HashSet::new(),
                est,
                log: VecDeque::new(),
                local_hits: 0,
                remote_reads: 0,
                quit: false,
            }),
            work: Condvar::new(),
            landed: Condvar::new(),
            path_of: Box::new(path_of),
            frames,
            network,
        });
        let threads = (0..n.max(1))
            .map(|i| {
                let s = sh.clone();
                std::thread::Builder::new().name(format!("read-{i}")).spawn(move || reader(s)).expect("spawn reader")
            })
            .collect();
        Fetcher { sh, threads }
    }

    /// Frame `i`'s bytes: from the read-ahead, waiting for a read already
    /// running, or read now.
    pub fn get(&self, i: u64) -> Result<Arc<Vec<u8>>> {
        let sh = &self.sh;
        {
            let mut g = sh.st.lock().unwrap();
            loop {
                if let Some(b) = g.raw.get(&i).cloned() {
                    touch(&mut g, i);
                    return Ok(b);
                }
                if !g.inflight.contains(&i) {
                    break;
                }
                g = sh.landed.wait_timeout(g, Duration::from_millis(500)).unwrap().0;
            }
            g.queue.retain(|&q| q != i);
            g.inflight.insert(i);
        }
        let r = read_one(sh, i);
        let mut g = sh.st.lock().unwrap();
        g.inflight.remove(&i);
        let b = Arc::new(r?);
        insert(&mut g, i, b.clone());
        sh.landed.notify_all();
        Ok(b)
    }

    /// What the decode prefetch wants (most urgent first); `playing`: read
    /// on ahead in its direction as far as the byte budget allows.
    pub fn want(&self, list: &[u64], playing: bool) {
        let sh = &self.sh;
        let mut g = sh.st.lock().unwrap();
        let mut order: Vec<u64> = list.iter().copied().filter(|&f| f < sh.frames).collect();
        if playing && order.len() >= 2 {
            let (a, b) = (order[0] as i64, order[1] as i64);
            let step = (b - a).signum() * (b - a).abs().max(1);
            let depth = (g.budget * 4 / 5 / g.est.max(1)).clamp(order.len(), 400);
            let mut f = *order.last().unwrap() as i64;
            let seen: HashSet<u64> = order.iter().copied().collect();
            while order.len() < depth {
                f += step;
                if f < 0 || f >= sh.frames as i64 {
                    break;
                }
                if !seen.contains(&(f as u64)) {
                    order.push(f as u64);
                }
            }
        }
        g.protect = order.iter().copied().collect();
        let q: VecDeque<u64> = order.iter().copied().filter(|f| !g.raw.contains_key(f) && !g.inflight.contains(f)).collect();
        g.queue = q;
        if !g.queue.is_empty() {
            sh.work.notify_all();
        }
    }

    pub fn stats(&self) -> NetStats {
        let mut g = self.sh.st.lock().unwrap();
        let now = Instant::now();
        while g.log.front().is_some_and(|(t, _)| now.duration_since(*t) > Duration::from_secs(2)) {
            g.log.pop_front();
        }
        let bytes: usize = g.log.iter().map(|(_, n)| n).sum();
        let ahead = g.protect.iter().filter(|f| g.raw.contains_key(f)).count();
        NetStats { network: self.sh.network, bytes_per_sec: bytes as f64 / 2.0, ahead, frame_bytes: g.est, local_hits: g.local_hits, remote_reads: g.remote_reads, queued: g.queue.len() }
    }
}

impl Drop for Fetcher {
    fn drop(&mut self) {
        self.sh.st.lock().unwrap().quit = true;
        self.sh.work.notify_all();
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}

fn touch(g: &mut State, i: u64) {
    if let Some(p) = g.lru.iter().position(|&x| x == i) {
        g.lru.remove(p);
    }
    g.lru.push_back(i);
}

fn insert(g: &mut State, i: u64, b: Arc<Vec<u8>>) {
    if let Some(old) = g.raw.insert(i, b.clone()) {
        g.bytes -= old.len();
    }
    g.bytes += b.len();
    g.est = (g.est * 7 + b.len()) / 8;
    touch(g, i);
    // Over budget: the least recently used that is not in the read-ahead.
    let mut k = 0;
    while g.bytes > g.budget && k < g.lru.len() {
        let f = g.lru[k];
        if g.protect.contains(&f) || f == i {
            k += 1;
            continue;
        }
        g.lru.remove(k);
        if let Some(b) = g.raw.remove(&f) {
            g.bytes -= b.len();
        }
    }
}

/// One frame file: the local copy when the share's file has one, else the
/// file (and a local copy kept, for a file on a share).
fn read_one(sh: &Shared, i: u64) -> Result<Vec<u8>> {
    let p = (sh.path_of)(i);
    let cache = if sh.network { netcache::get() } else { None };
    let key = cache.and_then(|c| c.key(&p));
    if let (Some(c), Some(k)) = (cache, &key) {
        if c.has(k) {
            if let Some(b) = c.read(k) {
                let mut g = sh.st.lock().unwrap();
                g.local_hits += 1;
                return Ok(b);
            }
        }
    }
    let b = netio::read_file(&p).map_err(|e| anyhow::anyhow!("reading {}: {e}", p.display()))?;
    {
        let mut g = sh.st.lock().unwrap();
        g.log.push_back((Instant::now(), b.len()));
        if sh.network {
            g.remote_reads += 1;
        }
    }
    if let (Some(c), Some(k)) = (cache, key) {
        let b2 = Arc::new(b.clone());
        c.put(k, b2);
    }
    Ok(b)
}

fn reader(sh: Arc<Shared>) {
    loop {
        let i = {
            let mut g = sh.st.lock().unwrap();
            loop {
                if g.quit {
                    return;
                }
                if let Some(i) = g.queue.pop_front() {
                    g.inflight.insert(i);
                    break i;
                }
                g = sh.work.wait(g).unwrap();
            }
        };
        let r = read_one(&sh, i);
        let mut g = sh.st.lock().unwrap();
        g.inflight.remove(&i);
        if let Ok(b) = r {
            insert(&mut g, i, Arc::new(b));
        }
        sh.landed.notify_all();
    }
}

/// Copy every frame of a sequence on a share to the local disk cache
/// ("cache shot locally"): `done` counts frames, `stop` ends it early.
pub fn cache_locally(frames: u64, path_of: impl Fn(u64) -> PathBuf + Send + Sync + 'static, done: Arc<std::sync::atomic::AtomicU64>, stop: Arc<std::sync::atomic::AtomicBool>) -> JoinHandle<()> {
    use std::sync::atomic::Ordering;
    std::thread::spawn(move || {
        let Some(c) = netcache::get() else { return };
        let next = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let path_of = Arc::new(path_of);
        let hs: Vec<_> = (0..8)
            .map(|_| {
                let (next, done, stop, path_of) = (next.clone(), done.clone(), stop.clone(), path_of.clone());
                std::thread::spawn(move || loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= frames || stop.load(Ordering::Relaxed) {
                        break;
                    }
                    let p = path_of(i);
                    if let Some(k) = c.key(&p) {
                        if !c.has(&k) {
                            if let Ok(b) = netio::read_file(&p) {
                                c.put(k, Arc::new(b));
                            }
                        }
                    }
                    done.fetch_add(1, Ordering::Relaxed);
                })
            })
            .collect();
        for h in hs {
            let _ = h.join();
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_read_ahead_runs_past_the_wanted_frames_and_a_scrub_cancels_it() {
        let dir = std::env::temp_dir().join(format!("scrub-fetch-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for i in 0..100 {
            std::fs::write(dir.join(format!("f.{i:04}.bin")), vec![i as u8; 1000]).unwrap();
        }
        let d2 = dir.clone();
        let f = Fetcher::new(100, dir.join("f.0000.bin"), move |i| d2.join(format!("f.{i:04}.bin")));
        f.want(&[10, 11, 12], true);
        let t = Instant::now();
        while f.stats().ahead < 50 && t.elapsed().as_secs() < 5 {
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(f.stats().ahead >= 50, "{:?}", f.stats());
        assert_eq!(f.get(40).unwrap()[0], 40);
        // Backwards: the read-ahead turns round.
        f.want(&[90, 89, 88], true);
        std::thread::sleep(Duration::from_millis(200));
        assert_eq!(f.get(60).unwrap()[0], 60);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
