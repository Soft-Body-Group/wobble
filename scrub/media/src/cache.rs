//! The RAM frame cache: decoded frames by (source, index), least recently
//! used out first, bounded by BYTES (a 4K ProRes frame is 33 MB, a 1080p
//! JPEG 8 MB; a frame count would mean nothing). Shared by every frame
//! server, so two viewers on one clip hold it once.

use crate::frame::FrameRef;
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};

pub type Key = (u64, u64);

struct Entry {
    frame: FrameRef,
    tick: u64,
}

struct Inner {
    map: HashMap<Key, Entry>,
    /// tick -> key, oldest first.
    lru: BTreeMap<u64, Key>,
    tick: u64,
    bytes: usize,
    budget: usize,
}

impl Inner {
    fn touch(&mut self, key: Key) {
        self.tick += 1;
        let tick = self.tick;
        if let Some(e) = self.map.get_mut(&key) {
            self.lru.remove(&e.tick);
            e.tick = tick;
            self.lru.insert(tick, key);
        }
    }
}

pub struct Cache {
    inner: Mutex<Inner>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CacheStats {
    pub frames: usize,
    pub bytes: usize,
    pub budget: usize,
}

impl Cache {
    pub fn new(budget: usize) -> Arc<Self> {
        Arc::new(Self {
            inner: Mutex::new(Inner { map: HashMap::new(), lru: BTreeMap::new(), tick: 0, bytes: 0, budget }),
        })
    }

    /// A budget from `EDIFY_CACHE_GB`, else a quarter of the machine's RAM
    /// capped at 16 GB (and never under 1 GB).
    pub fn default_budget() -> usize {
        if let Some(gb) = crate::sys::var("CACHE_GB").ok().and_then(|s| s.parse::<f64>().ok()) {
            return (gb * (1u64 << 30) as f64) as usize;
        }
        let ram = crate::sys::ram_bytes().unwrap_or(8 << 30);
        (ram / 4).clamp(1 << 30, 16 << 30) as usize
    }

    /// The frame, marked as just used.
    pub fn get(&self, key: Key) -> Option<FrameRef> {
        let mut g = self.inner.lock().unwrap();
        let f = g.map.get(&key).map(|e| e.frame.clone())?;
        g.touch(key);
        Some(f)
    }

    pub fn contains(&self, key: Key) -> bool {
        self.inner.lock().unwrap().map.contains_key(&key)
    }

    /// Mark frames as just used (the frames around the playhead), so the
    /// LRU never evicts what is about to be shown.
    pub fn touch(&self, keys: impl IntoIterator<Item = Key>) {
        let mut g = self.inner.lock().unwrap();
        for k in keys {
            g.touch(k);
        }
    }

    pub fn insert(&self, key: Key, frame: FrameRef) {
        let mut g = self.inner.lock().unwrap();
        let bytes = frame.bytes();
        g.tick += 1;
        let tick = g.tick;
        if let Some(old) = g.map.insert(key, Entry { frame, tick }) {
            g.lru.remove(&old.tick);
            g.bytes -= old.frame.bytes();
        }
        g.lru.insert(tick, key);
        g.bytes += bytes;
        while g.bytes > g.budget && g.lru.len() > 1 {
            let (_, k) = g.lru.pop_first().expect("non-empty");
            if let Some(e) = g.map.remove(&k) {
                g.bytes -= e.frame.bytes();
            }
        }
    }

    pub fn stats(&self) -> CacheStats {
        let g = self.inner.lock().unwrap();
        CacheStats { frames: g.map.len(), bytes: g.bytes, budget: g.budget }
    }

    /// Indices cached for one source, unordered.
    pub fn frames_of(&self, source: u64) -> Vec<u64> {
        let g = self.inner.lock().unwrap();
        g.map.keys().filter(|k| k.0 == source).map(|k| k.1).collect()
    }

    pub fn clear_source(&self, source: u64) {
        let mut g = self.inner.lock().unwrap();
        let keys: Vec<Key> = g.map.keys().filter(|k| k.0 == source).copied().collect();
        for k in keys {
            if let Some(e) = g.map.remove(&k) {
                g.lru.remove(&e.tick);
                g.bytes -= e.frame.bytes();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::{Frame, Layout};

    fn frame(bytes: usize) -> FrameRef {
        Arc::new(Frame::owned((bytes / 4) as u32, 1, Layout::Rgba8, vec![vec![0; bytes]]))
    }

    #[test]
    fn the_least_recently_used_frame_leaves_first_when_over_budget() {
        let c = Cache::new(300);
        c.insert((0, 1), frame(100));
        c.insert((0, 2), frame(100));
        c.insert((0, 3), frame(100));
        c.get((0, 1));
        c.insert((0, 4), frame(100));
        assert!(c.contains((0, 1)), "touched, so kept");
        assert!(!c.contains((0, 2)), "oldest, so evicted");
        assert_eq!(c.stats().bytes, 300);
    }

    #[test]
    fn reinserting_a_key_does_not_double_count() {
        let c = Cache::new(1000);
        c.insert((0, 1), frame(100));
        c.insert((0, 1), frame(200));
        assert_eq!(c.stats().bytes, 200);
        assert_eq!(c.stats().frames, 1);
    }
}
