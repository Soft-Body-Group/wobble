//! Where the decoders get file bytes.
//!
//! Native: the filesystem, through the same calls as ever — [`open`] on a
//! local path is exactly the `BufReader<File>` that exr's own `from_file`
//! makes. An `s3://bucket/key` path goes to [`crate::objstore`] instead:
//! a `Read + Seek` that fetches byte ranges on demand through an NVMe
//! cache, so a decoder that seeks to one part's chunks pulls only those.
//!
//! The browser build (`web/`) has no filesystem, so files live in an
//! in-memory table the page fills — dropped, picked or fetched — under
//! paths the app then loads like any other ([`insert`]). Every decoder
//! reads through here, so a Read, its thumbnails and its layers work the
//! same on both.
//!
//! `Path::is_dir` / `is_file` are false for an `s3://` path; anything that
//! may hold one asks [`is_dir`] / [`is_file`] here instead.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

/// A buffered reader over one file.
#[cfg(not(target_arch = "wasm32"))]
pub enum Reader {
    File(std::io::BufReader<std::fs::File>),
    Remote(crate::objstore::RemoteReader),
}
#[cfg(target_arch = "wasm32")]
pub type Reader = std::io::Cursor<Bytes>;

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use super::Reader;
    use std::io::{self, BufRead, Read, Seek, SeekFrom};

    impl Read for Reader {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            match self {
                Reader::File(r) => r.read(buf),
                Reader::Remote(r) => r.read(buf),
            }
        }
    }

    impl BufRead for Reader {
        fn fill_buf(&mut self) -> io::Result<&[u8]> {
            match self {
                Reader::File(r) => r.fill_buf(),
                Reader::Remote(r) => r.fill_buf(),
            }
        }
        fn consume(&mut self, amt: usize) {
            match self {
                Reader::File(r) => r.consume(amt),
                Reader::Remote(r) => r.consume(amt),
            }
        }
    }

    impl Seek for Reader {
        fn seek(&mut self, to: SeekFrom) -> io::Result<u64> {
            match self {
                Reader::File(r) => r.seek(to),
                Reader::Remote(r) => r.seek(to),
            }
        }
    }
}

/// A whole file's bytes (derefs to `[u8]`).
#[cfg(not(target_arch = "wasm32"))]
pub type Bytes = Vec<u8>;
#[cfg(target_arch = "wasm32")]
pub type Bytes = std::sync::Arc<[u8]>;

#[cfg(not(target_arch = "wasm32"))]
fn remote(path: &Path) -> Option<Result<crate::objstore::Loc>> {
    crate::objstore::is_remote(path).then(|| crate::objstore::Loc::parse(path))
}

pub fn open(path: &Path) -> Result<Reader> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        if let Some(loc) = remote(path) {
            return crate::objstore::RemoteReader::open(&loc?)
                .map(Reader::Remote)
                .with_context(|| format!("opening {}", path.display()));
        }
        std::fs::File::open(path)
            .map(|f| Reader::File(std::io::BufReader::new(f)))
            .with_context(|| format!("opening {}", path.display()))
    }
    #[cfg(target_arch = "wasm32")]
    web::get(path).map(std::io::Cursor::new).with_context(|| format!("opening {}", path.display()))
}

pub fn read(path: &Path) -> Result<Bytes> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        if let Some(loc) = remote(path) {
            return crate::objstore::read_all(&loc?)
                .with_context(|| format!("reading {}", path.display()));
        }
        std::fs::read(path).with_context(|| format!("reading {}", path.display()))
    }
    #[cfg(target_arch = "wasm32")]
    web::get(path).with_context(|| format!("reading {}", path.display()))
}

pub fn exists(path: &Path) -> bool {
    #[cfg(not(target_arch = "wasm32"))]
    {
        if let Some(loc) = remote(path) {
            return loc.map(|l| crate::objstore::exists(&l)).unwrap_or(false);
        }
        path.exists()
    }
    #[cfg(target_arch = "wasm32")]
    web::get(path).is_ok()
}

/// `Path::is_file`, including for an `s3://` object.
pub fn is_file(path: &Path) -> bool {
    #[cfg(not(target_arch = "wasm32"))]
    {
        if let Some(loc) = remote(path) {
            return loc.map(|l| crate::objstore::exists(&l)).unwrap_or(false);
        }
        path.is_file()
    }
    #[cfg(target_arch = "wasm32")]
    web::get(path).is_ok()
}

/// `Path::is_dir`, including for an `s3://` prefix with something under it.
pub fn is_dir(path: &Path) -> bool {
    #[cfg(not(target_arch = "wasm32"))]
    {
        if let Some(loc) = remote(path) {
            return loc.map(|l| crate::objstore::is_dir(&l)).unwrap_or(false);
        }
        path.is_dir()
    }
    #[cfg(target_arch = "wasm32")]
    !web::list(path).is_empty()
}

/// The paths directly inside `dir` (unsorted, like `read_dir`).
pub fn list(dir: &Path) -> Result<Vec<PathBuf>> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        if let Some(loc) = remote(dir) {
            return crate::objstore::list(&loc?).with_context(|| format!("listing {}", dir.display()));
        }
        std::fs::read_dir(dir)
            .map(|rd| rd.filter_map(|e| e.ok()).map(|e| e.path()).collect())
            .with_context(|| format!("reading {}", dir.display()))
    }
    #[cfg(target_arch = "wasm32")]
    Ok(web::list(dir))
}

#[cfg(target_arch = "wasm32")]
pub use web::{insert, list_all};

#[cfg(target_arch = "wasm32")]
mod web {
    use std::collections::BTreeMap;
    use std::io;
    use std::path::{Path, PathBuf};
    use std::sync::{Arc, Mutex};

    static FILES: Mutex<BTreeMap<PathBuf, Arc<[u8]>>> = Mutex::new(BTreeMap::new());

    pub(super) fn get(path: &Path) -> io::Result<Arc<[u8]>> {
        FILES
            .lock()
            .ok()
            .and_then(|f| f.get(path).cloned())
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "not in the page's files"))
    }

    pub(super) fn list(dir: &Path) -> Vec<PathBuf> {
        FILES
            .lock()
            .map(|f| f.keys().filter(|p| p.parent() == Some(dir)).cloned().collect())
            .unwrap_or_default()
    }

    /// Put a file in the table (replacing one at the same path).
    pub fn insert(path: impl Into<PathBuf>, bytes: Vec<u8>) {
        if let Ok(mut f) = FILES.lock() {
            f.insert(path.into(), bytes.into());
        }
    }

    /// Every path in the table, sorted.
    pub fn list_all() -> Vec<PathBuf> {
        FILES.lock().map(|f| f.keys().cloned().collect()).unwrap_or_default()
    }
}
