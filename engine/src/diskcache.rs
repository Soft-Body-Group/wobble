//! NVMe frame-cache tier: decoded frames stored raw on disk so a cache hit
//! is one sequential file read (~10–20 ms for a 4K half frame) instead of a
//! full EXR decode (100–400 ms).
//!
//! Files are the staging layout verbatim — RGBA f16 rows padded to wgpu's
//! 256-byte copy alignment — behind a 16-byte header, so a hit streams
//! straight into a mapped staging slot with zero conversion. The same format
//! backs the viewer render cache (`render_dir`), which is why `RawFrameDecoder`
//! implements `ExrDecoder`: a directory of cached renders plays through the
//! ordinary `Player` with all its pacing/seek machinery.
//!
//! Keys hash (absolute path, file size, mtime), so an overwritten source
//! frame misses instead of serving stale pixels. Eviction is oldest-written
//! first (std can't touch mtimes portably, so this is FIFO, not true LRU),
//! checked every 64 writes.

use crate::decode::{ExrDecoder, ImageInfo};
use anyhow::{Context as _, Result};
use half::f16;
use std::io::{Read as _, Write as _};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use rayon::prelude::*;
use std::sync::Mutex;
use web_time::{Duration, Instant};

const MAGIC: u32 = 0x3246_4347; // "GCF2" little-endian — raw payload
/// "GCF3": same header block, payload is lz4 blocks. Written only when
/// compression actually paid, so GCF2 files stay valid and nothing already
/// cached is invalidated; a pre-change binary rejects GCF3 on the magic and
/// treats it as a miss, which costs a re-decode and nothing worse.
const MAGIC_Z: u32 = 0x3346_4347;

/// Payload chunk, uncompressed: long enough for lz4 to find real matches,
/// short enough that one frame decompresses across cores.
const CHUNK: usize = 8 << 20;
/// Per-chunk compressed lengths live in the header from word 7 on.
const MAX_CHUNKS: usize = (HEADER - 28) / 4;
/// Keep the compressed form only if it saves at least this much. Measured on
/// real 6K frames: sparse AOV layers 10-20x, a dense full-frame plate 1.05x.
/// Below the threshold the read path would pay a decompress that buys
/// nothing, so those frames stay raw.
const KEEP_RATIO: f64 = 0.87;
/// Bail out after probing ONE chunk if even that much does not compress.
/// Packing a dense 6K frame in full costs ~75 ms and is then thrown away,
/// and this runs on the first-pass decode path. Deliberately lenient: the
/// probe only has to rule out the hopeless case, `KEEP_RATIO` still decides.
const PROBE_RATIO: f64 = 0.95;
/// Header block size. Page-aligned so the payload starts on a 4096 boundary:
/// F_NOCACHE (direct) reads collapse to a slow bounce path on unaligned file
/// offsets, and readahead-free streaming wants big aligned requests anyway.
const HEADER: usize = 4096;

#[derive(PartialEq)]
enum IoRole {
    Read,
    Write,
}

/// WOBBLE_NOCACHE bypasses the OS page cache on .gcf IO (macOS F_NOCACHE)
/// — measurement switches for TRUE NVMe playback numbers, since any working
/// set that fits in RAM is otherwise silently served from the page cache
/// (writes populate it too). Values: "1"/"rw" = both sides, "w" = writes
/// only (bake a cold tier, then measure buffered reads WITH readahead — the
/// realistic beyond-RAM cold-read case), "r" = reads only (direct-IO floor:
/// no readahead, queue-depth-1 per stream). No-op off macOS.
/// `GPU_COMP_NOCACHE` is still honoured as the pre-rename spelling.
fn nocache(f: &std::fs::File, role: IoRole) {
    #[cfg(target_os = "macos")]
    {
        static MODE: std::sync::OnceLock<(bool, bool)> = std::sync::OnceLock::new();
        let (r, w) = *MODE.get_or_init(|| {
            let raw = std::env::var("WOBBLE_NOCACHE")
                .or_else(|_| std::env::var("GPU_COMP_NOCACHE"));
            match raw.as_deref() {
                Ok("1") | Ok("rw") => (true, true),
                Ok("r") => (true, false),
                Ok("w") => (false, true),
                _ => (false, false),
            }
        });
        if match role {
            IoRole::Read => r,
            IoRole::Write => w,
        } {
            use std::os::fd::AsRawFd;
            unsafe { libc::fcntl(f.as_raw_fd(), libc::F_NOCACHE, 1) };
        }
    }
    #[cfg(not(target_os = "macos"))]
    let _ = (f, role);
}

/// Open a cache file for reading. On Windows the open says the read will
/// be one sequential pass (`FILE_FLAG_SEQUENTIAL_SCAN`), so the cache
/// manager reads ahead aggressively and drops pages behind — a 6K frame is
/// ~290 MB read front to back exactly once.
fn open_for_read(file: &Path) -> std::io::Result<std::fs::File> {
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        const FILE_FLAG_SEQUENTIAL_SCAN: u32 = 0x0800_0000;
        std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(FILE_FLAG_SEQUENTIAL_SCAN)
            .open(file)
    }
    #[cfg(not(windows))]
    std::fs::File::open(file)
}

/// One cold read of `p` into `buf` (sector-aligned, whole sectors long)
/// bypassing the Windows file cache (`FILE_FLAG_NO_BUFFERING`): the
/// benchmark's read number is the DRIVE, as `F_NOCACHE` makes it on macOS.
/// Without it a file just written reads back from RAM at memcpy speed.
#[cfg(windows)]
fn read_unbuffered(p: &Path, buf: &mut [u8]) -> std::io::Result<()> {
    use std::io::Read as _;
    use std::os::windows::fs::OpenOptionsExt;
    const FILE_FLAG_NO_BUFFERING: u32 = 0x2000_0000;
    const FILE_FLAG_SEQUENTIAL_SCAN: u32 = 0x0800_0000;
    let mut f = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN)
        .open(p)?;
    // Unbuffered reads come back in whole sectors, so every offset handed
    // to `read` stays aligned; the last one stops short at end of file.
    let mut got = 0;
    while got < buf.len() {
        match f.read(&mut buf[got..])? {
            0 => break,
            n => got += n,
        }
    }
    Ok(())
}

pub struct DiskCache {
    root: PathBuf,
    budget: AtomicU64,
    writes: AtomicU64,
    pub hits: AtomicU64,
    /// Frames that had to be decoded (disk miss on a disk-enabled read).
    pub misses: AtomicU64,
    /// (when computed, bytes) — walking the cache dir is too slow per frame.
    usage: Mutex<Option<(Instant, u64)>>,
}

impl DiskCache {
    pub fn new(root: impl Into<PathBuf>, budget_bytes: u64) -> Result<Self> {
        let root = root.into();
        std::fs::create_dir_all(root.join("frames"))
            .with_context(|| format!("creating cache dir {}", root.display()))?;
        std::fs::create_dir_all(root.join("render"))?;
        Ok(Self {
            root,
            budget: AtomicU64::new(budget_bytes),
            writes: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            usage: Mutex::new(None),
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn set_budget(&self, bytes: u64) {
        self.budget.store(bytes, Ordering::Relaxed);
    }

    /// Directory for one baked viewer-cache sequence (keyed by comp hash).
    pub fn render_dir(&self, key: u64) -> PathBuf {
        self.root.join("render").join(format!("{key:016x}"))
    }

    /// Cache file for one source frame; None if the source can't be stat'd.
    /// Cache file for one source frame, including which LAYER was decoded
    /// from it — two layers of one EXR are different pixels and must never
    /// share a cache entry.
    ///
    /// `frame` distinguishes frames INSIDE one file — a movie, where every
    /// tick names the same path and only the index differs. Frame 0 keeps
    /// the bare name so a still sequence's existing cache entries stay
    /// valid; a file-per-frame format never passes anything else.
    pub fn frame_file_layer(
        &self,
        source: &Path,
        layer: Option<&str>,
        frame: u64,
    ) -> Option<PathBuf> {
        self.frame_file_layer_scaled(source, layer, frame, 1)
    }

    /// The frame key with a PROXY divisor folded in: a half-res entry and
    /// the full-res one are different files, so toggling proxy never
    /// overwrites the tier it came from (the size check would make them
    /// perpetual misses of each other otherwise). Proxy entries populate
    /// lazily like everything else and age out through the same eviction.
    pub fn frame_file_layer_scaled(
        &self,
        source: &Path,
        layer: Option<&str>,
        frame: u64,
        proxy: u32,
    ) -> Option<PathBuf> {
        self.frame_file_packed(source, layer, frame, proxy, 0)
    }

    /// The frame key with the storage PACK folded in (`Pack::tag`, 0 for
    /// RGBA half): a Load stored as one 8-bit channel and the same file
    /// stored whole are different entries of different sizes.
    pub fn frame_file_packed(
        &self,
        source: &Path,
        layer: Option<&str>,
        frame: u64,
        proxy: u32,
        pack: u8,
    ) -> Option<PathBuf> {
        self.frame_file_windowed(source, layer, frame, proxy, pack, None)
    }

    /// The frame key with a reader WINDOW folded in too (`Pack::window_key`):
    /// a section of a plate is its own, smaller entry.
    pub fn frame_file_windowed(
        &self,
        source: &Path,
        layer: Option<&str>,
        frame: u64,
        proxy: u32,
        pack: u8,
        window: Option<[u32; 4]>,
    ) -> Option<PathBuf> {
        let base = self.frame_file(source)?;
        let layer = layer.filter(|l| !l.is_empty());
        if layer.is_none() && frame == 0 && proxy <= 1 && pack == 0 && window.is_none() {
            return Some(base);
        }
        let mut h = fnv64(source.to_string_lossy().as_bytes());
        if let Some(l) = layer {
            h = fnv64_more(h, l.as_bytes());
        }
        if frame != 0 {
            h = fnv64_more(h, &frame.to_le_bytes());
        }
        if proxy > 1 {
            h = fnv64_more(h, &[b'p', proxy as u8]);
        }
        if pack != 0 {
            h = fnv64_more(h, &[b'k', pack]);
        }
        if let Some(w) = window {
            h = fnv64_more(h, b"w");
            for v in w {
                h = fnv64_more(h, &v.to_le_bytes());
            }
        }
        let stem = base.file_stem()?.to_string_lossy().into_owned();
        Some(base.with_file_name(format!("{stem}-{h:016x}.gcf")))
    }

    pub fn frame_file(&self, source: &Path) -> Option<PathBuf> {
        let meta = std::fs::metadata(source).ok()?;
        let mtime = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        let mut h = fnv64(source.to_string_lossy().as_bytes());
        h = fnv64_more(h, &meta.len().to_le_bytes());
        h = fnv64_more(h, &mtime.to_le_bytes());
        // DPX decode changed meaning (linear -> code values, see dpx.rs):
        // entries written before that hold linear pixels and must miss.
        if source.extension().is_some_and(|e| e.eq_ignore_ascii_case("dpx")) {
            h = fnv64_more(h, b"dpx-codes");
        }
        Some(self.root.join("frames").join(format!("{h:016x}.gcf")))
    }

    /// Is this source frame in the NVMe tier? (Existence only — the read
    /// path still validates the header.)
    ///
    /// Keyed EXACTLY as `write_frame`'s path is keyed. Layer, frame-in-file
    /// and proxy all move the key, so a residency check that leaves them out
    /// reports 0% for every layered or proxied read while the entries sit on
    /// disk — which reads as "the cache is broken" rather than "the meter
    /// is asking the wrong question".
    pub fn has_frame(
        &self,
        source: &Path,
        layer: Option<&str>,
        frame: u64,
        proxy: u32,
    ) -> bool {
        self.has_frame_packed(source, layer, frame, proxy, 0)
    }

    /// `has_frame` for a packed reader (`Pack::tag`).
    pub fn has_frame_packed(
        &self,
        source: &Path,
        layer: Option<&str>,
        frame: u64,
        proxy: u32,
        pack: u8,
    ) -> bool {
        self.frame_file_packed(source, layer, frame, proxy, pack)
            .is_some_and(|f| f.exists())
    }

    /// Read a cached frame into `dst` (staging layout). Ok(false) = file
    /// missing or layout mismatch — treat as a miss, not an error.
    pub fn read_frame_into(file: &Path, info: ImageInfo, dst: &mut [f16]) -> Result<bool> {
        Self::read_impl(file, info, dst, false)
    }

    /// Same, for a destination that may only ever be WRITTEN — a GPU-mapped
    /// staging slot. lz4 resolves a match by reading back what it already
    /// wrote, and on write-combined memory every one of those reads is an
    /// uncached fetch, so each compressed chunk lands in a chunk-sized
    /// scratch and is copied across from the same thread — never a whole
    /// frame's scratch (450 MB at 7680², page-faulted in on every read) and
    /// never one thread copying the lot. A raw frame still streams straight
    /// in.
    pub fn read_frame_into_write_only(
        file: &Path,
        info: ImageInfo,
        dst: &mut [f16],
    ) -> Result<bool> {
        Self::read_impl(file, info, dst, true)
    }

    fn read_impl(
        file: &Path,
        info: ImageInfo,
        dst: &mut [f16],
        write_only: bool,
    ) -> Result<bool> {
        let Ok(mut f) = open_for_read(file) else { return Ok(false) };
        nocache(&f, IoRole::Read);
        let mut head = [0u8; HEADER];
        if f.read_exact(&mut head).is_err() {
            return Ok(false);
        }
        let word = |i: usize| u32::from_le_bytes(head[i * 4..i * 4 + 4].try_into().unwrap());
        let stride = info.row_stride_f16();
        let magic = word(0);
        if (magic != MAGIC && magic != MAGIC_Z)
            || word(1) != info.width
            || word(2) != info.height
            || word(3) as usize != stride
        {
            return Ok(false);
        }
        let n = stride * info.height as usize;
        anyhow::ensure!(dst.len() >= n, "cache read dst too small");
        if magic == MAGIC {
            let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut dst[..n]);
            f.read_exact(bytes).with_context(|| format!("reading {}", file.display()))?;
            return Ok(true);
        }
        // Compressed. An unknown codec is a MISS, not an error: a newer
        // build's entry re-decodes here instead of failing the read.
        if word(4) != 1 {
            return Ok(false);
        }
        let chunk = word(5) as usize;
        let count = word(6) as usize;
        if chunk == 0 || count == 0 || count > MAX_CHUNKS || count != (n * 2).div_ceil(chunk) {
            return Ok(false);
        }
        let lens: Vec<usize> = (0..count).map(|i| word(7 + i) as usize).collect();
        let mut payload = vec![0u8; lens.iter().sum()];
        f.read_exact(&mut payload)
            .with_context(|| format!("reading {}", file.display()))?;
        let mut ins: Vec<&[u8]> = Vec::with_capacity(count);
        let mut at = 0;
        for l in &lens {
            ins.push(&payload[at..at + l]);
            at += l;
        }
        let out: &mut [u8] = bytemuck::cast_slice_mut(&mut dst[..n]);
        let outs: Vec<&mut [u8]> = out.chunks_mut(chunk).collect();
        anyhow::ensure!(outs.len() == count, "gcf chunk count mismatch");
        let unpack = |o: &mut [u8], i: &[u8]| match lz4_flex::block::decompress_into(i, o) {
            Ok(got) if got == o.len() => Ok(()),
            Ok(got) => Err(format!("chunk decompressed to {got}, want {}", o.len())),
            Err(e) => Err(format!("{e}")),
        };
        let done = if write_only {
            outs.into_par_iter().zip(ins.into_par_iter()).try_for_each_init(
                Vec::new,
                |scratch: &mut Vec<u8>, (o, i)| {
                    scratch.resize(o.len(), 0);
                    unpack(scratch, i)?;
                    o.copy_from_slice(scratch);
                    Ok(())
                },
            )
        } else {
            outs.into_par_iter().zip(ins.into_par_iter()).try_for_each(|(o, i)| unpack(o, i))
        };
        done.map_err(|e: String| anyhow::anyhow!("decompressing {}: {e}", file.display()))?;
        Ok(true)
    }

    /// Atomically write one frame (tmp + rename); occasionally evicts.
    /// The tmp name is unique per attempt: two readers decoding the same
    /// source frame concurrently (same cache key) must not share a tmp or
    /// one rename steals the other's file out from under it.
    pub fn write_frame(&self, file: &Path, info: ImageInfo, data: &[f16]) -> Result<()> {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let n = info.row_stride_f16() * info.height as usize;
        anyhow::ensure!(data.len() >= n, "cache write src too small");
        let tmp = file.with_extension(format!(
            "tmp{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        {
            let raw = std::fs::File::create(&tmp)
                .with_context(|| format!("creating {}", tmp.display()))?;
            nocache(&raw, IoRole::Write);
            let mut f = std::io::BufWriter::new(raw);
            let src: &[u8] = bytemuck::cast_slice(&data[..n]);
            match pack(src) {
                Some(chunks) => {
                    let mut head = [0u8; HEADER];
                    let mut w = |i: usize, v: u32| {
                        head[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes())
                    };
                    w(0, MAGIC_Z);
                    w(1, info.width);
                    w(2, info.height);
                    w(3, info.row_stride_f16() as u32);
                    w(4, 1); // codec: lz4 block
                    w(5, CHUNK as u32);
                    w(6, chunks.len() as u32);
                    for (i, c) in chunks.iter().enumerate() {
                        w(7 + i, c.len() as u32);
                    }
                    f.write_all(&head)?;
                    for c in &chunks {
                        f.write_all(c)?;
                    }
                }
                None => {
                    f.write_all(&MAGIC.to_le_bytes())?;
                    f.write_all(&info.width.to_le_bytes())?;
                    f.write_all(&info.height.to_le_bytes())?;
                    f.write_all(&(info.row_stride_f16() as u32).to_le_bytes())?;
                    f.write_all(&[0u8; HEADER - 16])?; // pad payload to the block boundary
                    f.write_all(src)?;
                }
            }
        }
        std::fs::rename(&tmp, file)
            .with_context(|| format!("renaming into {}", file.display()))?;
        if self.writes.fetch_add(1, Ordering::Relaxed) % 64 == 0 {
            self.evict_to_budget();
        }
        Ok(())
    }

    /// Total cached bytes; recomputed at most every 5 s (dir walks are slow).
    pub fn usage_bytes(&self) -> u64 {
        let mut guard = self.usage.lock().unwrap();
        if let Some((at, bytes)) = *guard {
            if at.elapsed() < Duration::from_secs(5) {
                return bytes;
            }
        }
        let bytes = walk_files(&self.root).into_iter().map(|(_, len, _)| len).sum();
        *guard = Some((Instant::now(), bytes));
        bytes
    }

    /// Delete oldest-written files until the store fits the budget.
    fn evict_to_budget(&self) {
        let budget = self.budget.load(Ordering::Relaxed);
        let mut files = walk_files(&self.root);
        let mut total: u64 = files.iter().map(|(_, len, _)| len).sum();
        if total <= budget {
            return;
        }
        files.sort_by_key(|(_, _, mtime)| *mtime);
        for (path, len, _) in files {
            if total <= budget {
                break;
            }
            if std::fs::remove_file(&path).is_ok() {
                total -= len;
            }
        }
        *self.usage.lock().unwrap() = Some((Instant::now(), total));
    }

    /// Wipe both tiers (the UI's "clear cache").
    pub fn clear(&self) {
        for sub in ["frames", "render"] {
            let dir = self.root.join(sub);
            let _ = std::fs::remove_dir_all(&dir);
            let _ = std::fs::create_dir_all(&dir);
        }
        *self.usage.lock().unwrap() = Some((Instant::now(), 0));
    }
}

/// lz4 every chunk of `raw`, or None when compression does not pay. Chunks
/// compress in parallel: this runs once per frame ever written, against a
/// decode that costs hundreds of ms, so the attempt is cheap even when it
/// is thrown away.
fn pack(raw: &[u8]) -> Option<Vec<Vec<u8>>> {
    let count = raw.len().div_ceil(CHUNK);
    if count == 0 || count > MAX_CHUNKS {
        return None;
    }
    // Probe the MIDDLE chunk: a centred subject lives there, so it is the
    // least flattering part of the frame to judge the whole by.
    if count > 2 {
        let mid = raw.chunks(CHUNK).nth(count / 2)?;
        if lz4_flex::block::compress(mid).len() as f64 > mid.len() as f64 * PROBE_RATIO {
            return None;
        }
    }
    let chunks: Vec<Vec<u8>> =
        raw.par_chunks(CHUNK).map(lz4_flex::block::compress).collect();
    let total: usize = chunks.iter().map(|c| c.len()).sum();
    (total as f64 <= raw.len() as f64 * KEEP_RATIO).then_some(chunks)
}

/// Every .gcf file under root, recursively: (path, len, mtime-nanos).
fn walk_files(root: &Path) -> Vec<(PathBuf, u64, u64)> {
    fn walk(dir: &Path, out: &mut Vec<(PathBuf, u64, u64)>) {
        let Ok(rd) = std::fs::read_dir(dir) else { return };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, out);
            } else if p.extension().is_some_and(|x| x == "gcf") {
                if let Ok(m) = e.metadata() {
                    let mtime = m
                        .modified()
                        .ok()
                        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                        .map(|d| d.as_nanos() as u64)
                        .unwrap_or(0);
                    out.push((p, m.len(), mtime));
                }
            }
        }
    }
    let mut out = Vec::new();
    walk(root, &mut out);
    out
}

fn fnv64(bytes: &[u8]) -> u64 {
    fnv64_more(0xcbf2_9ce4_8422_2325, bytes)
}

fn fnv64_more(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Stable content hash for arbitrary bytes (used by the app to key baked
/// viewer caches on graph topology + params).
pub fn content_hash(parts: &[&[u8]]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for p in parts {
        h = fnv64_more(h, p);
        h = fnv64_more(h, &[0xff]); // separator so ("ab","c") != ("a","bc")
    }
    h
}

/// `ExrDecoder` over .gcf raw frames: a baked render-cache directory becomes
/// an ordinary playable Sequence (pacing, seeks and the RAM tier for free).
pub struct RawFrameDecoder;

impl ExrDecoder for RawFrameDecoder {
    fn read_header(&self, path: &Path) -> Result<ImageInfo> {
        let mut f = std::fs::File::open(path)
            .with_context(|| format!("opening {}", path.display()))?;
        let mut head = [0u8; HEADER];
        f.read_exact(&mut head)?;
        let word = |i: usize| u32::from_le_bytes(head[i * 4..i * 4 + 4].try_into().unwrap());
        anyhow::ensure!(
            word(0) == MAGIC || word(0) == MAGIC_Z,
            "not a gcf frame: {}",
            path.display()
        );
        Ok(ImageInfo { width: word(1), height: word(2) })
    }

    fn decode_into(
        &self,
        path: &Path,
        info: ImageInfo,
        row_stride: usize,
        dst: &mut [f16],
    ) -> Result<()> {
        anyhow::ensure!(
            row_stride == info.row_stride_f16(),
            "gcf stride mismatch (want {}, file layout {})",
            row_stride,
            info.row_stride_f16()
        );
        // `dst` here can be a MAPPED staging slot (the uncached player path
        // decodes straight into one), so take the write-only route.
        let ok = DiskCache::read_frame_into_write_only(path, info, dst)?;
        anyhow::ensure!(ok, "gcf header mismatch: {}", path.display());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(info: ImageInfo, seed: u16) -> Vec<f16> {
        let n = info.row_stride_f16() * info.height as usize;
        (0..n).map(|i| f16::from_f32(((i as u16).wrapping_add(seed)) as f32)).collect()
    }

    #[test]
    fn roundtrip_and_layout_guard() {
        let dir = std::env::temp_dir().join(format!("gcf-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let dc = DiskCache::new(&dir, u64::MAX).unwrap();
        let info = ImageInfo { width: 37, height: 5 }; // width forces row padding
        let data = frame(info, 7);
        let file = dir.join("frames/x.gcf");
        dc.write_frame(&file, info, &data).unwrap();

        let mut back = vec![f16::ZERO; data.len()];
        assert!(DiskCache::read_frame_into(&file, info, &mut back).unwrap());
        assert_eq!(data, back);

        // A different geometry must read as a miss, not garbage.
        let other = ImageInfo { width: 36, height: 5 };
        let mut buf = vec![f16::ZERO; other.row_stride_f16() * 5];
        assert!(!DiskCache::read_frame_into(&file, other, &mut buf).unwrap());

        // RawFrameDecoder agrees with the header.
        let d = RawFrameDecoder;
        let h = d.read_header(&file).unwrap();
        assert_eq!((h.width, h.height), (37, 5));
        let mut dec = vec![f16::ZERO; data.len()];
        d.decode_into(&file, info, info.row_stride_f16(), &mut dec).unwrap();
        assert_eq!(data, dec);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A sparse AOV layer — what the harp writes by the hundred. These
    /// measured 10-20x on real 6K frames, so the entry must shrink AND come
    /// back bit-identical through both read routes.
    #[test]
    fn sparse_frame_compresses_and_roundtrips() {
        let dir = std::env::temp_dir().join(format!("gcf-zip-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let dc = DiskCache::new(&dir, u64::MAX).unwrap();
        let info = ImageInfo { width: 640, height: 64 };
        let n = info.row_stride_f16() * info.height as usize;
        let mut data = vec![f16::ZERO; n];
        for i in (0..n).step_by(977) {
            data[i] = f16::from_f32(0.5);
        }
        let file = dir.join("frames/z.gcf");
        dc.write_frame(&file, info, &data).unwrap();

        let raw = (n * 2 + HEADER) as u64;
        let got = std::fs::metadata(&file).unwrap().len();
        assert!(got * 4 < raw, "stored {got} B, raw would be {raw} B");
        assert_eq!(magic_of(&file), MAGIC_Z);

        let mut back = vec![f16::from_f32(9.0); n];
        assert!(DiskCache::read_frame_into(&file, info, &mut back).unwrap());
        assert_eq!(data, back);
        // The mapped-slot route must agree byte for byte.
        let mut wo = vec![f16::from_f32(9.0); n];
        assert!(DiskCache::read_frame_into_write_only(&file, info, &mut wo).unwrap());
        assert_eq!(data, wo);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A dense plate measures 1.05x on lz4. Compressing it would buy nothing
    /// and cost a decompress on every read, so it must stay raw — and raw
    /// means GCF2, which a pre-change binary can still read.
    #[test]
    fn dense_frame_stays_raw() {
        let dir = std::env::temp_dir().join(format!("gcf-raw-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let dc = DiskCache::new(&dir, u64::MAX).unwrap();
        let info = ImageInfo { width: 256, height: 32 };
        let n = info.row_stride_f16() * info.height as usize;
        let mut st = 0x2545_f491_4f6c_dd1du64;
        let data: Vec<f16> = (0..n)
            .map(|_| {
                st ^= st << 13;
                st ^= st >> 7;
                st ^= st << 17;
                f16::from_bits(st as u16)
            })
            .collect();
        let file = dir.join("frames/d.gcf");
        dc.write_frame(&file, info, &data).unwrap();

        assert_eq!(std::fs::metadata(&file).unwrap().len(), (n * 2 + HEADER) as u64);
        assert_eq!(magic_of(&file), MAGIC);

        // Random f16s include NaNs, which never compare equal — check bits.
        let mut back = vec![f16::ZERO; n];
        assert!(DiskCache::read_frame_into(&file, info, &mut back).unwrap());
        assert_eq!(
            bytemuck::cast_slice::<f16, u8>(&data),
            bytemuck::cast_slice::<f16, u8>(&back)
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The residency meter and the write path must agree on the key. They
    /// did not: writes key on (layer, frame-in-file, proxy) while the meter
    /// asked for the bare key, so a layered or proxied Read showed "NVMe 0%"
    /// with every frame sitting on disk.
    #[test]
    fn residency_matches_the_written_key() {
        // NOT "gcf-key-<pid>": that is `frame_key_tracks_source_identity`'s
        // directory, and these two run in parallel in one process.
        let dir = std::env::temp_dir().join(format!("gcf-reskey-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let dc = DiskCache::new(&dir, u64::MAX).unwrap();
        let info = ImageInfo { width: 64, height: 8 };
        let data = frame(info, 3);
        // `frame_file` hashes the source's size and mtime, so it has to exist.
        let src = dir.join("plate.0001.exr");
        std::fs::write(&src, b"not really an exr").unwrap();

        let keyed = dc.frame_file_layer_scaled(&src, Some("beauty_aux"), 0, 2).unwrap();
        dc.write_frame(&keyed, info, &data).unwrap();

        assert!(dc.has_frame(&src, Some("beauty_aux"), 0, 2), "keyed lookup missed it");
        // Every other key names a different file and must NOT claim it.
        assert!(!dc.has_frame(&src, None, 0, 1), "bare key claimed a layered entry");
        assert!(!dc.has_frame(&src, Some("beauty_aux"), 0, 1), "proxy ignored");
        assert!(!dc.has_frame(&src, Some("other"), 0, 2), "layer ignored");
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn magic_of(file: &Path) -> u32 {
        let mut f = std::fs::File::open(file).unwrap();
        let mut b = [0u8; 4];
        f.read_exact(&mut b).unwrap();
        u32::from_le_bytes(b)
    }

    #[test]
    fn eviction_drops_oldest_first() {
        let dir = std::env::temp_dir().join(format!("gcf-evict-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let info = ImageInfo { width: 32, height: 4 };
        let one = (info.row_stride_f16() * 4 * 2 + HEADER) as u64;
        let dc = DiskCache::new(&dir, one * 2).unwrap(); // room for two frames
        let data = frame(info, 0);
        for i in 0..3 {
            let f = dir.join(format!("frames/{i}.gcf"));
            dc.write_frame(&f, info, &data).unwrap();
            // mtime granularity: space the writes out.
            std::thread::sleep(web_time::Duration::from_millis(20));
        }
        dc.evict_to_budget();
        let left: Vec<_> = walk_files(&dir).into_iter().map(|(p, _, _)| p).collect();
        assert!(left.len() <= 2, "evicted to {left:?}");
        assert!(
            !left.iter().any(|p| p.ends_with("0.gcf")),
            "oldest survived: {left:?}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn frame_key_tracks_source_identity() {
        let dir = std::env::temp_dir().join(format!("gcf-key-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let dc = DiskCache::new(dir.join("cache"), u64::MAX).unwrap();
        let src = dir.join("plate.exr");
        std::fs::write(&src, b"aaaa").unwrap();
        let k1 = dc.frame_file(&src).unwrap();
        let k1b = dc.frame_file(&src).unwrap();
        assert_eq!(k1, k1b, "key must be stable");
        std::thread::sleep(web_time::Duration::from_millis(20));
        std::fs::write(&src, b"bbbbbb").unwrap(); // new size + mtime
        let k2 = dc.frame_file(&src).unwrap();
        assert_ne!(k1, k2, "modified source must miss");
        let _ = std::fs::remove_dir_all(&dir);
    }
}

/// One cache-volume measurement, made with the cache's own frame format.
#[derive(Clone, Copy, Debug)]
pub struct Bench {
    pub write_bps: f64,
    pub read_bps: f64,
    pub bytes: u64,
    pub frames: u32,
    /// Reads bypassed the page cache (macOS F_NOCACHE, Windows
    /// FILE_FLAG_NO_BUFFERING) — on Linux the read number can be
    /// flattered by RAM.
    pub cold_read: bool,
}

impl DiskCache {
    /// Time the cache volume: write `frames` frames of noise through the
    /// real frame writer, fsync them, then read them back raw — with
    /// F_NOCACHE on macOS so the read hits the DISK, not the page cache.
    /// Files live under `bench/` in the root (an extension eviction never
    /// touches) and are removed afterwards. This is what answers "is this
    /// drive fast enough for playback" — the number the USB-cache mystery
    /// needed.
    pub fn benchmark(&self, info: ImageInfo, frames: u32) -> Result<Bench> {
        anyhow::ensure!(frames > 0, "benchmark needs at least one frame");
        let dir = self.root.join("bench");
        std::fs::create_dir_all(&dir)?;
        let n = info.row_stride_f16() * info.height as usize;
        // Incompressible payload: a filesystem that compresses (or a drive
        // that dedupes) would flatter zeros into fiction.
        let mut data = vec![f16::ZERO; n];
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        for v in data.iter_mut() {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *v = f16::from_bits((x >> 48) as u16);
        }
        let paths: Vec<PathBuf> =
            (0..frames).map(|i| dir.join(format!("bench_{i}.gcfb"))).collect();
        let cleanup = |paths: &[PathBuf]| {
            for p in paths {
                let _ = std::fs::remove_file(p);
            }
            let _ = std::fs::remove_dir(&dir);
        };
        let run = || -> Result<Bench> {
            let t0 = Instant::now();
            for p in &paths {
                self.write_frame(p, info, &data)?;
            }
            // fsync so the write number is the drive, not the write-back
            // cache lying about it. WRITABLE handle: on Windows `sync_all`
            // is FlushFileBuffers, which refuses a read-only one with
            // "Access is denied" (macOS fsync takes either) — that refusal
            // was the whole "benchmark shows nothing on Windows".
            for p in &paths {
                std::fs::OpenOptions::new()
                    .write(true)
                    .open(p)
                    .and_then(|f| f.sync_all())
                    .with_context(|| format!("flushing {}", p.display()))?;
            }
            let wrote = t0.elapsed();
            let file_bytes = std::fs::metadata(&paths[0])?.len() as usize;
            // Sector-aligned and a whole number of sectors long, which an
            // unbuffered read on Windows requires of both buffer and length.
            let span = file_bytes.div_ceil(HEADER) * HEADER;
            let mut backing = vec![0u8; span + HEADER];
            let at = backing.as_ptr().align_offset(HEADER);
            let buf = &mut backing[at..at + span];
            #[allow(unused_mut)]
            let mut cold = false;
            let t1 = Instant::now();
            for p in &paths {
                #[cfg(windows)]
                {
                    read_unbuffered(p, buf)
                        .with_context(|| format!("reading {}", p.display()))?;
                    cold = true;
                }
                #[cfg(not(windows))]
                {
                    let mut f = std::fs::File::open(p)?;
                    #[cfg(target_os = "macos")]
                    {
                        use std::os::fd::AsRawFd;
                        unsafe { libc::fcntl(f.as_raw_fd(), libc::F_NOCACHE, 1) };
                        cold = true;
                    }
                    use std::io::Read as _;
                    f.read_exact(&mut buf[..file_bytes])?;
                }
                std::hint::black_box(&buf[0]);
            }
            let read = t1.elapsed();
            let total = file_bytes as u64 * frames as u64;
            Ok(Bench {
                write_bps: total as f64 / wrote.as_secs_f64().max(1e-9),
                read_bps: total as f64 / read.as_secs_f64().max(1e-9),
                bytes: total,
                frames,
                cold_read: cold,
            })
        };
        let out = run();
        cleanup(&paths);
        out
    }
}

#[cfg(test)]
mod bench_tests {
    use super::*;

    /// The benchmark measures, cleans up after itself, and reports sane
    /// (positive, finite) numbers.
    #[test]
    fn benchmark_runs_and_cleans_up() {
        let dir = std::env::temp_dir().join(format!("wobble_test_bench_{}", std::process::id()));
        let c = DiskCache::new(&dir, u64::MAX / 4).unwrap();
        let info = ImageInfo { width: 256, height: 128 };
        let b = c.benchmark(info, 8).unwrap();
        assert!(b.write_bps > 0.0 && b.write_bps.is_finite());
        assert!(b.read_bps > 0.0 && b.read_bps.is_finite());
        assert_eq!(b.frames, 8);
        assert!(b.bytes > 8 * 256 * 128 * 8, "payload plus headers");
        assert!(!c.root().join("bench").exists(), "bench files removed");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
