//! Optimised media: a long-GOP clip (H.264, HEVC) transcoded to ProRes, an
//! all-intra format any frame of which decodes on its own, so scrubbing it
//! costs one frame's decode wherever the hand lands.
//!
//! **The original is never written, moved or deleted.** The optimised file
//! is a new file, written under a temporary name and renamed into place
//! only when complete, in one of two places (picked per transcode):
//!
//! - **Beside the original**: `name.edify.mov` in the same folder. The
//!   source's size and mtime are stamped into the file's metadata, so an
//!   edited source no longer matches it.
//! - **The managed cache** (`~/Library/Caches/Edify/optimised` unless set):
//!   files named by a hash of the source's path, size and mtime, listed in
//!   `index.json` with their sizes and last use. A size cap evicts the least
//!   recently used; `purge` and `purge_source` delete what the cache holds
//!   (only ever its own files).
//!
//! Opening a source uses whichever optimised copy exists (`find`), beside
//! the original first.
//!
//! The encoder is VideoToolbox's ProRes (the media engine) where present,
//! else FFmpeg's `prores_ks`. Audio streams are copied as they are, the
//! frame rate, size, colour tags and timecode carry over.

use crate::movie;
use anyhow::{bail, Context as _, Result};
use ffmpeg_next as ff;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// The metadata key holding the source's stamp in a beside-the-original file.
const STAMP_KEY: &str = "com.scrub.source";
const SUFFIX: &str = ".scrub.mov";
/// Copies made before the rename: still found and used.
const OLD_STAMP_KEY: &str = "com.edify.source";
const OLD_SUFFIX: &str = ".edify.mov";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Dest {
    /// `name.edify.mov` next to the original.
    #[default]
    Beside,
    /// Scrub's managed cache folder.
    Cache,
}

impl Dest {
    pub fn parse(s: &str) -> Option<Dest> {
        match s {
            "beside" => Some(Dest::Beside),
            "cache" => Some(Dest::Cache),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Profile {
    Lt,
    #[default]
    Standard,
    Hq,
}

impl Profile {
    pub fn name(self) -> &'static str {
        match self {
            Profile::Lt => "lt",
            Profile::Standard => "standard",
            Profile::Hq => "hq",
        }
    }

    pub fn parse(s: &str) -> Option<Profile> {
        match s {
            "lt" => Some(Profile::Lt),
            "standard" | "422" => Some(Profile::Standard),
            "hq" => Some(Profile::Hq),
            _ => None,
        }
    }
}

/// What the user chose once: where optimised files go by default, the
/// cache folder and its cap, the ProRes flavour.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    pub dest: Dest,
    pub cache_dir: PathBuf,
    /// Evict least recently used past this many GB; `None`: no cap.
    pub cache_cap_gb: Option<f64>,
    pub profile: Profile,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            dest: Dest::Beside,
            cache_dir: crate::sys::cache_dir().join("optimised"),
            cache_cap_gb: None,
            profile: Profile::Standard,
        }
    }
}

impl Settings {
    fn path() -> PathBuf {
        crate::sys::support_dir().join("optimise.json")
    }

    pub fn load() -> Settings {
        std::fs::read(Self::path()).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default()
    }

    pub fn save(&self) -> Result<()> {
        let p = Self::path();
        if let Some(d) = p.parent() {
            std::fs::create_dir_all(d)?;
        }
        std::fs::write(&p, serde_json::to_vec_pretty(self)?)?;
        Ok(())
    }
}

/// Size and modification time of a source: what makes an optimised copy
/// stale when either moves.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Stamp {
    pub size: u64,
    pub mtime_ns: u128,
}

impl Stamp {
    pub fn of(path: &Path) -> Result<Stamp> {
        let m = std::fs::metadata(path).with_context(|| format!("reading {}", path.display()))?;
        let mtime_ns = m.modified()?.duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
        Ok(Stamp { size: m.len(), mtime_ns })
    }

    fn text(&self) -> String {
        format!("{}:{}", self.size, self.mtime_ns)
    }
}

/// `name.edify.mov` beside the source.
pub fn beside_path(source: &Path) -> PathBuf {
    let stem = source.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    source.with_file_name(format!("{stem}{SUFFIX}"))
}

/// Is this file an optimised copy (so never optimised again)?
pub fn is_optimised(path: &Path) -> bool {
    path.file_name().is_some_and(|n| n.to_string_lossy().ends_with(SUFFIX) || n.to_string_lossy().ends_with(OLD_SUFFIX))
}

/// The absolute, link-free path a cache entry is keyed by.
pub fn canonical(p: &Path) -> PathBuf {
    std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf())
}

// --- the managed cache -----------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Entry {
    pub source: PathBuf,
    pub stamp: Stamp,
    /// File name inside the cache folder.
    pub file: String,
    pub bytes: u64,
    /// Seconds since the epoch.
    pub last_used: u64,
    pub profile: Profile,
}

#[derive(Default, Serialize, Deserialize)]
struct Index {
    entries: Vec<Entry>,
}

/// The managed cache folder and its `index.json`.
pub struct CacheDir {
    pub dir: PathBuf,
}

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// FNV-1a: stable across builds (std's hasher is not).
fn fnv(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

impl CacheDir {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    fn index_path(&self) -> PathBuf {
        self.dir.join("index.json")
    }

    fn load(&self) -> Index {
        std::fs::read(self.index_path()).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default()
    }

    fn save(&self, idx: &Index) -> Result<()> {
        std::fs::create_dir_all(&self.dir)?;
        let tmp = self.dir.join("index.json.tmp");
        std::fs::write(&tmp, serde_json::to_vec_pretty(idx)?)?;
        std::fs::rename(&tmp, self.index_path())?;
        Ok(())
    }

    /// The cache file a source at this stamp would have.
    pub fn file_for(&self, source: &Path, stamp: Stamp) -> PathBuf {
        let key = format!("{}\n{}", source.display(), stamp.text());
        let stem = source.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        let stem: String = stem.chars().filter(|c| c.is_alphanumeric() || "-_".contains(*c)).take(40).collect();
        self.dir.join(format!("{stem}-{:016x}.mov", fnv(key.as_bytes())))
    }

    /// The cached copy of a source as it is now, marked used.
    pub fn lookup(&self, source: &Path) -> Option<PathBuf> {
        let stamp = Stamp::of(source).ok()?;
        let mut idx = self.load();
        let e = idx.entries.iter_mut().find(|e| e.source == source && e.stamp == stamp)?;
        let file = self.dir.join(&e.file);
        if !file.is_file() {
            return None;
        }
        e.last_used = now_secs();
        let _ = self.save(&idx);
        Some(file)
    }

    /// Record a finished file, replacing any older copy of the same source.
    fn add(&self, entry: Entry, cap: Option<u64>) -> Result<()> {
        let mut idx = self.load();
        let mut gone = Vec::new();
        idx.entries.retain(|e| {
            let stale = e.source == entry.source && e.file != entry.file;
            if stale {
                gone.push(e.file.clone());
            }
            !stale
        });
        idx.entries.retain(|e| e.file != entry.file);
        idx.entries.push(entry);
        for f in gone {
            let _ = std::fs::remove_file(self.dir.join(f));
        }
        if let Some(cap) = cap {
            self.evict(&mut idx, cap);
        }
        self.save(&idx)
    }

    /// Least recently used out until the cache fits `cap` bytes (the newest
    /// entry stays even when it alone is over).
    fn evict(&self, idx: &mut Index, cap: u64) {
        idx.entries.sort_by_key(|e| e.last_used);
        while idx.entries.len() > 1 && idx.entries.iter().map(|e| e.bytes).sum::<u64>() > cap {
            let e = idx.entries.remove(0);
            let _ = std::fs::remove_file(self.dir.join(&e.file));
        }
    }

    pub fn enforce_cap(&self, cap_gb: Option<f64>) -> Result<()> {
        if let Some(gb) = cap_gb {
            let mut idx = self.load();
            self.evict(&mut idx, (gb * (1u64 << 30) as f64) as u64);
            self.save(&idx)?;
        }
        Ok(())
    }

    pub fn entries(&self) -> Vec<Entry> {
        self.load().entries
    }

    /// Bytes the cache's files take.
    pub fn size(&self) -> u64 {
        self.load().entries.iter().map(|e| e.bytes).sum()
    }

    /// Delete every cached file (and only those).
    pub fn purge(&self) -> Result<u64> {
        let idx = self.load();
        let mut freed = 0;
        for e in &idx.entries {
            if std::fs::remove_file(self.dir.join(&e.file)).is_ok() {
                freed += e.bytes;
            }
        }
        self.save(&Index::default())?;
        Ok(freed)
    }

    /// Delete the cached copies of one source.
    pub fn purge_source(&self, source: &Path) -> Result<u64> {
        let source = &canonical(source);
        let mut idx = self.load();
        let mut freed = 0;
        idx.entries.retain(|e| {
            if e.source == *source {
                if std::fs::remove_file(self.dir.join(&e.file)).is_ok() {
                    freed += e.bytes;
                }
                false
            } else {
                true
            }
        });
        self.save(&idx)?;
        Ok(freed)
    }
}

/// The optimised copy of `source` that matches it now: beside the original
/// first, then the managed cache.
pub fn find(source: &Path, settings: &Settings) -> Option<PathBuf> {
    let source = &canonical(source);
    if is_optimised(source) {
        return None;
    }
    let stem = source.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    for beside in [beside_path(source), source.with_file_name(format!("{stem}{OLD_SUFFIX}"))] {
        if beside.is_file() {
            let stamp = Stamp::of(source).ok()?;
            movie::init();
            if let Ok(ictx) = ff::format::input(&beside) {
                let m = ictx.metadata();
                if m.get(STAMP_KEY).or_else(|| m.get(OLD_STAMP_KEY)) == Some(stamp.text().as_str()) {
                    return Some(beside);
                }
            }
        }
    }
    CacheDir::new(&settings.cache_dir).lookup(source)
}

// --- the transcode ---------------------------------------------------------

/// A transcode's progress, shared with whoever started it.
#[derive(Default)]
pub struct Progress {
    pub done: AtomicU64,
    pub total: AtomicU64,
    pub cancel: AtomicBool,
}

impl Progress {
    pub fn fraction(&self) -> f32 {
        let t = self.total.load(Ordering::Relaxed).max(1);
        self.done.load(Ordering::Relaxed) as f32 / t as f32
    }
}

pub struct Job {
    pub source: PathBuf,
    pub dest: Dest,
    pub profile: Profile,
    pub cache_dir: PathBuf,
    pub cache_cap_gb: Option<f64>,
}

/// Transcode `job.source` and return the optimised file's path.
pub fn run(job: &Job, progress: &Arc<Progress>) -> Result<PathBuf> {
    let source = canonical(&job.source);
    let job = &Job { source, dest: job.dest, profile: job.profile, cache_dir: job.cache_dir.clone(), cache_cap_gb: job.cache_cap_gb };
    anyhow::ensure!(!is_optimised(&job.source), "{} is already an optimised file", job.source.display());
    let stamp = Stamp::of(&job.source)?;
    let out = match job.dest {
        Dest::Beside => beside_path(&job.source),
        Dest::Cache => CacheDir::new(&job.cache_dir).file_for(&job.source, stamp),
    };
    anyhow::ensure!(out != job.source, "refusing to write over the source");
    if let Some(d) = out.parent() {
        std::fs::create_dir_all(d)?;
    }
    // Written under a temporary name; renamed only when complete, so a
    // half-written file is never mistaken for an optimised copy.
    let tmp = out.with_extension("partial.mov");
    let r = transcode(&job.source, &tmp, job.profile, stamp, progress);
    if let Err(e) = r {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    std::fs::rename(&tmp, &out)?;
    if job.dest == Dest::Cache {
        let cache = CacheDir::new(&job.cache_dir);
        let entry = Entry {
            source: job.source.clone(),
            stamp,
            file: out.file_name().expect("a file").to_string_lossy().into_owned(),
            bytes: std::fs::metadata(&out)?.len(),
            last_used: now_secs(),
            profile: job.profile,
        };
        cache.add(entry, job.cache_cap_gb.map(|gb| (gb * (1u64 << 30) as f64) as u64))?;
    }
    Ok(out)
}

/// Which ProRes encoder, and the pixel format to feed it.
fn pick_encoder(ten_bit: bool) -> Result<(ff::Codec, ff::format::Pixel)> {
    if cfg!(target_os = "macos") && crate::sys::var("PRORES").as_deref() != Ok("ks") {
        if let Some(c) = ff::encoder::find_by_name("prores_videotoolbox") {
            let fmt = if ten_bit { ff::format::Pixel::P010LE } else { ff::format::Pixel::NV12 };
            return Ok((c, fmt));
        }
    }
    let c = ff::encoder::find_by_name("prores_ks").context("this FFmpeg has no ProRes encoder")?;
    Ok((c, ff::format::Pixel::YUV422P10LE))
}

fn transcode(src: &Path, dst: &Path, profile: Profile, stamp: Stamp, progress: &Arc<Progress>) -> Result<()> {
    movie::init();
    let mut ictx = ff::format::input(src).with_context(|| format!("opening {}", src.display()))?;
    let vidx = ictx.streams().best(ff::media::Type::Video).map(|s| s.index()).context("no video stream")?;
    let info = movie::Movie::open(src)?;
    let info = crate::Source::info(&info).clone();
    progress.total.store(info.frames, Ordering::Relaxed);

    let vstream = ictx.stream(vidx).expect("index from this context");
    let in_tb = vstream.time_base();
    let mut dctx = ff::codec::context::Context::from_parameters(vstream.parameters())?;
    dctx.set_threading(ff::codec::threading::Config { kind: ff::codec::threading::Type::Frame, count: 0 });
    let mut decoder = dctx.decoder().video()?;
    // Bit depth of the luma component, off the pixel format's descriptor.
    let ten_bit = unsafe {
        let d = ff::ffi::av_pix_fmt_desc_get(decoder.format().into());
        !d.is_null() && (*d).comp[0].depth > 8
    };

    let mut octx = ff::format::output(dst).with_context(|| format!("creating {}", dst.display()))?;
    let global = octx.format().flags().contains(ff::format::Flags::GLOBAL_HEADER);

    let (codec, enc_fmt) = pick_encoder(ten_bit)?;
    let enc_tb = ff::Rational::new(info.fps_den as i32, info.fps_num as i32);
    let mut ectx = ff::codec::context::Context::new_with_codec(codec);
    ectx.set_time_base(enc_tb);
    if global {
        ectx.set_flags(ff::codec::Flags::GLOBAL_HEADER);
    }
    // prores_ks threads by slice; unthreaded it encodes 4K at 3 fps.
    ectx.set_threading(ff::codec::threading::Config { kind: ff::codec::threading::Type::Slice, count: 0 });
    let mut enc = ectx.encoder().video()?;
    enc.set_width(decoder.width());
    enc.set_height(decoder.height());
    enc.set_format(enc_fmt);
    enc.set_colorspace(decoder.color_space());
    enc.set_color_range(decoder.color_range());
    unsafe {
        let e = enc.as_mut_ptr();
        let d = decoder.as_ptr();
        (*e).framerate = ff::ffi::AVRational { num: info.fps_num as i32, den: info.fps_den as i32 };
        (*e).color_primaries = (*d).color_primaries;
        (*e).color_trc = (*d).color_trc;
        (*e).sample_aspect_ratio = (*d).sample_aspect_ratio;
    }
    let mut opts = ff::Dictionary::new();
    opts.set("profile", profile.name());
    let mut encoder = enc.open_with(opts).context("opening the ProRes encoder")?;

    // Output streams: video, then every audio stream copied as it is.
    let mut map: Vec<Option<usize>> = vec![None; ictx.nb_streams() as usize];
    {
        let mut vs = octx.add_stream(codec)?;
        vs.set_parameters(&encoder);
        vs.set_time_base(enc_tb);
        let mut md = ff::Dictionary::new();
        let tc = vstream.metadata().get("timecode").map(str::to_string);
        let tc = tc.or_else(|| ictx.metadata().get("timecode").map(str::to_string));
        if let Some(tc) = tc {
            md.set("timecode", &tc);
        }
        vs.set_metadata(md);
        // The muxer writes the timecode track only for a stream with a rate.
        unsafe {
            let r = ff::ffi::AVRational { num: info.fps_num as i32, den: info.fps_den as i32 };
            (*vs.as_mut_ptr()).avg_frame_rate = r;
            (*vs.as_mut_ptr()).r_frame_rate = r;
        }
        map[vidx] = Some(0);
    }
    for s in ictx.streams() {
        if s.parameters().medium() == ff::media::Type::Audio {
            let mut os = octx.add_stream(ff::encoder::find(ff::codec::Id::None))?;
            os.set_parameters(s.parameters());
            unsafe {
                (*os.parameters().as_mut_ptr()).codec_tag = 0;
            }
            os.set_time_base(s.time_base());
            map[s.index()] = Some(os.index());
        }
    }
    let mut md = ff::Dictionary::new();
    md.set(STAMP_KEY, &stamp.text());
    md.set("comment", &format!("optimised by Scrub from {}", src.display()));
    octx.set_metadata(md);
    let mut hopts = ff::Dictionary::new();
    hopts.set("movflags", "+use_metadata_tags");
    octx.write_header_with(hopts).context("writing the header")?;
    let out_tbs: Vec<ff::Rational> = octx.streams().map(|s| s.time_base()).collect();
    let in_tbs: Vec<ff::Rational> = ictx.streams().map(|s| s.time_base()).collect();
    let _ = in_tb;

    let mut scaler: Option<ff::software::scaling::Context> = None;
    let mut index: i64 = 0;
    let drain = |encoder: &mut ff::encoder::video::Encoder, octx: &mut ff::format::context::Output| -> Result<()> {
        let mut pkt = ff::Packet::empty();
        while encoder.receive_packet(&mut pkt).is_ok() {
            pkt.set_stream(0);
            pkt.rescale_ts(enc_tb, out_tbs[0]);
            pkt.write_interleaved(octx)?;
        }
        Ok(())
    };
    let mut encode = |frame: ff::frame::Video,
                      encoder: &mut ff::encoder::video::Encoder,
                      octx: &mut ff::format::context::Output,
                      index: &mut i64|
     -> Result<()> {
        let frame = if frame.format() == ff::format::Pixel::VIDEOTOOLBOX {
            let mut sw = ff::frame::Video::empty();
            let r = unsafe { ff::ffi::av_hwframe_transfer_data(sw.as_mut_ptr(), frame.as_ptr(), 0) };
            anyhow::ensure!(r >= 0, "reading back a VideoToolbox frame failed ({r})");
            sw
        } else {
            frame
        };
        let mut f = if frame.format() == enc_fmt {
            frame
        } else {
            if scaler.is_none() {
                scaler = Some(ff::software::scaling::Context::get(
                    frame.format(),
                    frame.width(),
                    frame.height(),
                    enc_fmt,
                    frame.width(),
                    frame.height(),
                    ff::software::scaling::Flags::BICUBIC,
                )?);
            }
            let mut out = ff::frame::Video::empty();
            scaler.as_mut().expect("built").run(&frame, &mut out)?;
            out
        };
        f.set_pts(Some(*index));
        unsafe {
            (*f.as_mut_ptr()).pict_type = ff::ffi::AVPictureType::AV_PICTURE_TYPE_NONE;
        }
        *index += 1;
        encoder.send_frame(&f)?;
        drain(encoder, octx)
    };

    for (stream, mut packet) in ictx.packets() {
        if progress.cancel.load(Ordering::Relaxed) {
            bail!("cancelled");
        }
        let si = stream.index();
        if si == vidx {
            decoder.send_packet(&packet)?;
            let mut frame = ff::frame::Video::empty();
            while decoder.receive_frame(&mut frame).is_ok() {
                encode(std::mem::replace(&mut frame, ff::frame::Video::empty()), &mut encoder, &mut octx, &mut index)?;
                progress.done.store(index as u64, Ordering::Relaxed);
            }
        } else if let Some(o) = map[si] {
            packet.rescale_ts(in_tbs[si], out_tbs[o]);
            packet.set_position(-1);
            packet.set_stream(o);
            packet.write_interleaved(&mut octx)?;
        }
    }
    decoder.send_eof()?;
    let mut frame = ff::frame::Video::empty();
    while decoder.receive_frame(&mut frame).is_ok() {
        encode(std::mem::replace(&mut frame, ff::frame::Video::empty()), &mut encoder, &mut octx, &mut index)?;
    }
    encoder.send_eof()?;
    drain(&mut encoder, &mut octx)?;
    octx.write_trailer()?;
    progress.done.store(index as u64, Ordering::Relaxed);
    anyhow::ensure!(index > 0, "no frames decoded from {}", src.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_beside_path_never_is_the_source_and_is_recognised() {
        let p = beside_path(Path::new("/x/clip.mov"));
        assert_eq!(p, Path::new("/x/clip.scrub.mov"));
        assert!(is_optimised(&p));
        // Edify's copies are still recognised.
        assert!(is_optimised(Path::new("/x/clip.edify.mov")));
        assert!(!is_optimised(Path::new("/x/clip.mov")));
    }

    #[test]
    fn a_changed_stamp_is_a_different_cache_file() {
        let c = CacheDir::new("/c");
        let a = c.file_for(Path::new("/x/clip.mp4"), Stamp { size: 1, mtime_ns: 2 });
        let b = c.file_for(Path::new("/x/clip.mp4"), Stamp { size: 1, mtime_ns: 3 });
        assert_ne!(a, b);
        assert!(a.starts_with("/c"));
    }

    #[test]
    fn the_cache_evicts_least_recently_used_past_its_cap_and_purges_only_its_files() {
        let dir = std::env::temp_dir().join(format!("scrub-cache-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let outside = dir.join("not-ours.txt");
        std::fs::write(&outside, b"keep").unwrap();
        let c = CacheDir::new(dir.join("cache"));
        std::fs::create_dir_all(&c.dir).unwrap();
        for (i, used) in [(0u64, 30u64), (1, 10), (2, 20)] {
            let file = format!("f{i}.mov");
            std::fs::write(c.dir.join(&file), vec![0u8; 100]).unwrap();
            let e = Entry {
                source: PathBuf::from(format!("/src/{i}")),
                stamp: Stamp { size: i, mtime_ns: 0 },
                file,
                bytes: 100,
                last_used: used,
                profile: Profile::Standard,
            };
            c.add(e, Some(250)).unwrap();
        }
        let left: Vec<String> = c.entries().iter().map(|e| e.file.clone()).collect();
        assert!(!left.contains(&"f1.mov".to_string()), "least recently used went first: {left:?}");
        assert!(!c.dir.join("f1.mov").exists());
        assert_eq!(c.size(), 200);
        c.purge_source(Path::new("/src/0")).unwrap();
        assert_eq!(c.size(), 100);
        c.purge().unwrap();
        assert_eq!(c.size(), 0);
        assert!(outside.exists(), "purge touches only the cache's own files");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
