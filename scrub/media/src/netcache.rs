//! A local copy of frames read off network storage, on the machine's own
//! disk (NVMe): byte-limited, least recently used goes first, keyed by the
//! remote file's path, size and modification time (a frame re-rendered on
//! the share is a new key). Filled as frames are read during play, and by
//! "cache shot locally". Written on one background thread, never where a
//! frame is wanted.
//!
//! `EDIFY_NETCACHE_GB` sets the budget (default 50; 0 turns it off),
//! `EDIFY_NETCACHE_DIR` the place (default the user cache dir).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::SystemTime;

pub struct NetCache {
    dir: PathBuf,
    budget: u64,
    tx: Mutex<Sender<(PathBuf, Arc<Vec<u8>>)>>,
    /// Entry -> (bytes, last used), for the budget.
    index: Arc<Mutex<Index>>,
}

#[derive(Default)]
struct Index {
    files: HashMap<PathBuf, (u64, SystemTime)>,
    total: u64,
}

static CACHE: OnceLock<Option<NetCache>> = OnceLock::new();

/// The cache, if it is on.
pub fn get() -> Option<&'static NetCache> {
    CACHE.get_or_init(|| NetCache::open(default_dir(), default_budget())).as_ref()
}

fn default_dir() -> PathBuf {
    if let Some(d) = crate::sys::var_os("NETCACHE_DIR") {
        return PathBuf::from(d);
    }
    #[cfg(target_os = "macos")]
    return crate::sys::cache_dir().join("network-frames");
    #[allow(unreachable_code)]
    crate::sys::cache_dir().join("network-frames")
}

fn default_budget() -> u64 {
    let gb: f64 = crate::sys::var("NETCACHE_GB").ok().and_then(|s| s.parse().ok()).unwrap_or(50.0);
    (gb * 1e9) as u64
}

impl NetCache {
    pub fn open(dir: PathBuf, budget: u64) -> Option<Self> {
        if budget == 0 || std::fs::create_dir_all(&dir).is_err() {
            return None;
        }
        let mut idx = Index::default();
        if let Ok(rd) = std::fs::read_dir(&dir) {
            for e in rd.flatten() {
                if let Ok(m) = e.metadata() {
                    let t = m.accessed().or_else(|_| m.modified()).unwrap_or(SystemTime::UNIX_EPOCH);
                    idx.files.insert(e.path(), (m.len(), t));
                    idx.total += m.len();
                }
            }
        }
        let index = Arc::new(Mutex::new(idx));
        let (tx, rx) = channel::<(PathBuf, Arc<Vec<u8>>)>();
        let (ix, d2) = (index.clone(), dir.clone());
        let _ = std::thread::Builder::new().name("netcache-write".into()).spawn(move || {
            while let Ok((p, bytes)) = rx.recv() {
                let tmp = d2.join(format!(".{}.tmp", std::process::id()));
                if std::fs::write(&tmp, &bytes[..]).is_ok() && std::fs::rename(&tmp, &p).is_ok() {
                    let mut g = ix.lock().unwrap();
                    let n = bytes.len() as u64;
                    if let Some((old, _)) = g.files.insert(p, (n, SystemTime::now())) {
                        g.total -= old;
                    }
                    g.total += n;
                    prune(&mut g, budget);
                }
            }
        });
        Some(NetCache { dir, budget, tx: Mutex::new(tx), index })
    }

    /// Where `remote`'s copy lives (its size and mtime in the key).
    pub fn key(&self, remote: &Path) -> Option<PathBuf> {
        let m = std::fs::metadata(remote).ok()?;
        let t = m.modified().ok()?.duration_since(SystemTime::UNIX_EPOCH).ok()?.as_nanos();
        let mut h = 0xcbf29ce484222325u64;
        for b in remote.as_os_str().as_encoded_bytes().iter().chain(&m.len().to_le_bytes()).chain(&t.to_le_bytes()) {
            h ^= *b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        let ext = remote.extension().map(|e| e.to_string_lossy().into_owned()).unwrap_or_default();
        Some(self.dir.join(format!("{h:016x}.{ext}")))
    }

    /// The local copy's bytes, when there is one.
    pub fn read(&self, key: &Path) -> Option<Vec<u8>> {
        let b = std::fs::read(key).ok()?;
        if let Some(e) = self.index.lock().unwrap().files.get_mut(key) {
            e.1 = SystemTime::now();
        }
        Some(b)
    }

    pub fn has(&self, key: &Path) -> bool {
        self.index.lock().unwrap().files.contains_key(key)
    }

    /// Keep a copy (written off-thread).
    pub fn put(&self, key: PathBuf, bytes: Arc<Vec<u8>>) {
        let _ = self.tx.lock().unwrap().send((key, bytes));
    }

    /// (bytes held, budget, files).
    pub fn usage(&self) -> (u64, u64, usize) {
        let g = self.index.lock().unwrap();
        (g.total, self.budget, g.files.len())
    }

    /// Empty it.
    pub fn clear(&self) {
        let mut g = self.index.lock().unwrap();
        for p in g.files.keys() {
            let _ = std::fs::remove_file(p);
        }
        g.files.clear();
        g.total = 0;
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }
}

fn prune(g: &mut Index, budget: u64) {
    if g.total <= budget {
        return;
    }
    let mut v: Vec<(PathBuf, u64, SystemTime)> = g.files.iter().map(|(p, (n, t))| (p.clone(), *n, *t)).collect();
    v.sort_by_key(|x| x.2);
    for (p, n, _) in v {
        if g.total <= budget * 9 / 10 {
            break;
        }
        let _ = std::fs::remove_file(&p);
        g.files.remove(&p);
        g.total -= n;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_frame_is_kept_by_path_size_and_mtime_and_the_budget_holds() {
        let root = std::env::temp_dir().join(format!("scrub-netcache-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("remote")).unwrap();
        let c = NetCache::open(root.join("cache"), 2500).unwrap();
        let remote: Vec<PathBuf> = (0..3).map(|i| root.join(format!("remote/f.{i}.exr"))).collect();
        for r in &remote {
            std::fs::write(r, vec![7u8; 1000]).unwrap();
        }
        for r in &remote {
            let k = c.key(r).unwrap();
            c.put(k, Arc::new(std::fs::read(r).unwrap()));
            std::thread::sleep(std::time::Duration::from_millis(30));
        }
        std::thread::sleep(std::time::Duration::from_millis(200));
        let (bytes, _, n) = c.usage();
        assert!(bytes <= 2500 && n == 2, "{bytes} {n}");
        // The newest is there; rewriting the remote makes a new key.
        let k2 = c.key(&remote[2]).unwrap();
        assert_eq!(c.read(&k2).unwrap().len(), 1000);
        std::thread::sleep(std::time::Duration::from_millis(1100));
        std::fs::write(&remote[2], vec![8u8; 900]).unwrap();
        assert_ne!(c.key(&remote[2]).unwrap(), k2);
        let _ = std::fs::remove_dir_all(&root);
    }
}
