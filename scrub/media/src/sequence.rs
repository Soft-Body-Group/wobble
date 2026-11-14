//! Image sequences: shot.1001.exr .. shot.1100.exr as one clip.
//!
//! The sequence is found from ONE sample file: the last run of digits in
//! its stem is the frame number, and every sibling with the same prefix,
//! suffix and digit count belongs to it. Index i is frame `first + i`; a
//! number with no file (a gap, a render still arriving) fails to decode
//! and is never cached, so it is picked up when it appears.

use crate::frame::{Frame, Layout};
use crate::source::{Source, SourceInfo};
use anyhow::{bail, Context as _, Result};
use std::path::{Path, PathBuf};
use std::time::Instant;

pub const IMAGE_EXTS: &[&str] = &["exr", "dpx", "png", "jpg", "jpeg"];

fn ext_of(path: &Path) -> String {
    path.extension().map(|e| e.to_string_lossy().to_ascii_lowercase()).unwrap_or_default()
}

pub fn is_image(path: &Path) -> bool {
    IMAGE_EXTS.contains(&ext_of(path).as_str())
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Exr,
    Dpx,
    Png,
    Jpeg,
}

impl Kind {
    fn of(path: &Path) -> Result<Kind> {
        Ok(match ext_of(path).as_str() {
            "exr" => Kind::Exr,
            "dpx" => Kind::Dpx,
            "png" => Kind::Png,
            "jpg" | "jpeg" => Kind::Jpeg,
            _ => bail!("{} is not an image format Scrub reads", path.display()),
        })
    }
}

/// A file name split around its frame number.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pattern {
    pub dir: PathBuf,
    pub prefix: String,
    pub suffix: String,
    /// Digit count of the sample; numbers are zero-padded to it.
    pub pad: usize,
}

impl Pattern {
    /// Split a file name at the last digit run of its stem.
    pub fn of(path: &Path) -> Option<(Pattern, i64)> {
        let name = path.file_name()?.to_str()?;
        let stem_end = name.rfind('.').unwrap_or(name.len());
        let stem = &name[..stem_end];
        let end = stem.rfind(|c: char| c.is_ascii_digit())? + 1;
        let start = stem[..end].rfind(|c: char| !c.is_ascii_digit()).map_or(0, |i| i + 1);
        let n: i64 = stem[start..end].parse().ok()?;
        let dir = path.parent().map(Path::to_path_buf).unwrap_or_default();
        Some((
            Pattern { dir, prefix: name[..start].to_string(), suffix: name[end..].to_string(), pad: end - start },
            n,
        ))
    }

    pub fn path(&self, n: i64) -> PathBuf {
        self.dir.join(format!("{}{:0pad$}{}", self.prefix, n, self.suffix, pad = self.pad))
    }

    /// The frame number of a sibling's name, if it belongs.
    fn number(&self, name: &str) -> Option<i64> {
        let mid = name.strip_prefix(&self.prefix)?.strip_suffix(&self.suffix)?;
        if mid.is_empty() || !mid.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        // Padded: exactly the pad. Unpadded (pad 1): any length, no
        // leading zero.
        let fits = mid.len() == self.pad || (self.pad < mid.len() && !mid.starts_with('0'));
        fits.then(|| mid.parse().ok()).flatten()
    }

    /// First and last number on disk.
    pub fn scan(&self) -> Result<(i64, i64)> {
        let mut lo = i64::MAX;
        let mut hi = i64::MIN;
        for e in std::fs::read_dir(&self.dir).with_context(|| format!("listing {}", self.dir.display()))? {
            let e = e?;
            if let Some(n) = e.file_name().to_str().and_then(|s| self.number(s)) {
                lo = lo.min(n);
                hi = hi.max(n);
            }
        }
        anyhow::ensure!(lo <= hi, "no frames match {}", self.path(0).display());
        Ok((lo, hi))
    }
}

pub struct Sequence {
    info: SourceInfo,
    pattern: Option<Pattern>,
    kind: Kind,
    /// The frame files' bytes, read ahead (`fetch`).
    fetch: crate::fetch::Fetcher,
}

impl Sequence {
    pub fn open(sample: &Path) -> Result<Self> {
        let kind = Kind::of(sample)?;
        let (pattern, first, last) = match Pattern::of(sample) {
            Some((p, _)) => {
                let (a, b) = p.scan()?;
                (Some(p), a, b)
            }
            None => (None, 0, 0),
        };
        let probe = pattern.as_ref().map_or(sample.to_path_buf(), |p| p.path(first));
        let (width, height, has_alpha) = header(kind, &probe)?;
        let what = match kind {
            Kind::Exr => "exr",
            Kind::Dpx => "dpx",
            Kind::Png => "png",
            Kind::Jpeg => "jpeg",
        };
        let info = SourceInfo {
            path: sample.to_path_buf(),
            label: format!("{what} sequence"),
            width,
            height,
            frames: (last - first + 1) as u64,
            first,
            fps_num: 24,
            fps_den: 1,
            has_alpha,
            has_audio: false,
            intra: true,
        };
        let (pat, single, first) = (pattern.clone(), sample.to_path_buf(), info.first);
        let fetch = crate::fetch::Fetcher::new(info.frames, probe.clone(), move |i| match &pat {
            Some(p) => p.path(first + i as i64),
            None => single.clone(),
        });
        Ok(Self { info, pattern, kind, fetch })
    }

    /// Decode a frame file's bytes (benches: no read-ahead).
    pub fn decode_bytes(&self, bytes: &[u8], path: &Path) -> Result<Frame> {
        Ok(match self.kind {
            Kind::Exr => decode_exr(bytes, path).or_else(|e| crate::movie::decode_image(path).map_err(|_| e))?,
            Kind::Dpx => crate::dpx::decode(bytes).or_else(|_| crate::movie::decode_image(path))?,
            Kind::Png => decode_png(bytes, path)?,
            Kind::Jpeg => decode_jpeg(bytes, path)?,
        })
    }

    pub fn path_of(&self, index: u64) -> PathBuf {
        match &self.pattern {
            Some(p) => p.path(self.info.first + index as i64),
            None => self.info.path.clone(),
        }
    }
}

impl Source for Sequence {
    fn info(&self) -> &SourceInfo {
        &self.info
    }

    fn prefetch(&self, frames: &[u64], playing: bool) {
        self.fetch.want(frames, playing);
    }

    fn net(&self) -> Option<crate::fetch::NetStats> {
        Some(self.fetch.stats())
    }

    fn decode(&self, index: u64) -> Result<Frame> {
        let t0 = Instant::now();
        let path = self.path_of(index);
        let bytes = self.fetch.get(index)?;
        // (DWAA / DWAB, which the exr crate does not read: FFmpeg's EXR
        // decoder does.)
        let mut f = self.decode_bytes(&bytes, &path)?;
        f.decode_ms = t0.elapsed().as_secs_f32() * 1000.0;
        Ok(f)
    }
}

fn header(kind: Kind, path: &Path) -> Result<(u32, u32, bool)> {
    match kind {
        Kind::Exr => {
            let meta = exr::meta::MetaData::read_from_file(path, false)
                .with_context(|| format!("reading header of {}", path.display()))?;
            let h = &meta.headers[0];
            let size = h.shared_attributes.display_window.size;
            let alpha = h.channels.list.iter().any(|c| c.name.eq("A"));
            Ok((size.width() as u32, size.height() as u32, alpha))
        }
        Kind::Dpx => {
            let (w, h) = crate::movie::image_size(path)?;
            Ok((w, h, false))
        }
        Kind::Png => {
            let dec = png::Decoder::new(std::io::BufReader::new(std::fs::File::open(path)?));
            let r = dec.read_info().with_context(|| format!("reading {}", path.display()))?;
            let i = r.info();
            let alpha = matches!(i.color_type, png::ColorType::Rgba | png::ColorType::GrayscaleAlpha) || i.trns.is_some();
            Ok((i.width, i.height, alpha))
        }
        Kind::Jpeg => {
            let f = decode_jpeg(&std::fs::read(path)?, path)?;
            Ok((f.width, f.height, false))
        }
    }
}

/// RGBA half at the DISPLAY window: a tight data window is placed in it,
/// overscan is cropped (Wobble's `Framing`).
fn decode_exr(bytes: &[u8], path: &Path) -> Result<Frame> {
    use exr::prelude::*;
    use std::cell::Cell;

    let meta = exr::meta::MetaData::read_from_buffered(std::io::Cursor::new(bytes), false)
        .with_context(|| format!("reading header of {}", path.display()))?;
    let h = &meta.headers[0];
    let disp = h.shared_attributes.display_window;
    let dpos = h.own_attributes.layer_position;
    let (dx, dy) = (dpos.x() as i64 - disp.position.x() as i64, dpos.y() as i64 - disp.position.y() as i64);
    let (w, hgt) = (disp.size.width(), disp.size.height());
    let stride = w * 4;
    // Transparent black where the data window does not reach.
    let mut px = vec![f16::ZERO; stride * hgt];
    let slot = Cell::new(Some(&mut px[..]));
    read()
        .no_deep_data()
        .largest_resolution_level()
        .rgba_channels(
            |_res, _chans| slot.take().expect("create called twice"),
            move |buf: &mut &mut [f16], pos, (r, g, b, a): (f16, f16, f16, f16)| {
                let (x, y) = (pos.x() as i64 + dx, pos.y() as i64 + dy);
                if x < 0 || y < 0 || x >= w as i64 || y >= hgt as i64 {
                    return;
                }
                let i = y as usize * stride + x as usize * 4;
                buf[i..i + 4].copy_from_slice(&[r, g, b, a]);
            },
        )
        .first_valid_layer()
        .all_attributes()
        .from_buffered(std::io::Cursor::new(bytes))
        .with_context(|| format!("decoding {}", path.display()))?;
    // (One memcpy: halves are little-endian u16s on every target we build.)
    let bytes: Vec<u8> = bytemuck::cast_slice::<f16, u8>(&px).to_vec();
    Ok(Frame::owned(w as u32, hgt as u32, Layout::RgbaF16, vec![bytes]))
}

fn decode_png(bytes: &[u8], path: &Path) -> Result<Frame> {
    use png::{BitDepth, ColorType, Transformations};
    let mut dec = png::Decoder::new(std::io::Cursor::new(bytes));
    dec.set_transformations(Transformations::EXPAND);
    let mut reader = dec.read_info().with_context(|| format!("reading {}", path.display()))?;
    let mut buf = vec![0u8; reader.output_buffer_size().unwrap_or(0)];
    let out = reader.next_frame(&mut buf).with_context(|| format!("decoding {}", path.display()))?;
    let (w, h) = (out.width as usize, out.height as usize);
    let (ct, depth) = reader.output_color_type();
    let comps = match ct {
        ColorType::Grayscale => 1,
        ColorType::GrayscaleAlpha => 2,
        ColorType::Rgb => 3,
        ColorType::Rgba => 4,
        ColorType::Indexed => bail!("palette PNG did not expand: {}", path.display()),
    };
    let wide = depth == BitDepth::Sixteen;
    let sb = if wide { 2 } else { 1 };
    let max = if wide { [0xff, 0xff] } else { [0xff, 0] };
    let mut px = vec![0u8; w * h * 4 * sb];
    for i in 0..w * h {
        let src = &buf[i * comps * sb..(i + 1) * comps * sb];
        let dst = &mut px[i * 4 * sb..(i + 1) * 4 * sb];
        // PNG's 16-bit samples are big-endian; ours are little.
        let get = |c: usize| -> [u8; 2] { if wide { [src[c * 2 + 1], src[c * 2]] } else { [src[c], 0] } };
        let rgba = match comps {
            1 => [get(0), get(0), get(0), max],
            2 => [get(0), get(0), get(0), get(1)],
            3 => [get(0), get(1), get(2), max],
            _ => [get(0), get(1), get(2), get(3)],
        };
        for (c, s) in rgba.iter().enumerate() {
            dst[c * sb..(c + 1) * sb].copy_from_slice(&s[..sb]);
        }
    }
    let layout = if wide { Layout::Rgba16 } else { Layout::Rgba8 };
    Ok(Frame::owned(w as u32, h as u32, layout, vec![px]))
}

fn decode_jpeg(bytes: &[u8], path: &Path) -> Result<Frame> {
    use zune_jpeg::zune_core::bytestream::ZCursor;
    use zune_jpeg::zune_core::colorspace::ColorSpace;
    use zune_jpeg::JpegDecoder;
    let mut dec = JpegDecoder::new(ZCursor::new(bytes));
    let opts = dec.options().jpeg_set_out_colorspace(ColorSpace::RGBA);
    dec.set_options(opts);
    let out = dec.decode().map_err(|e| anyhow::anyhow!("decoding {}: {e}", path.display()))?;
    let info = dec.info().with_context(|| format!("{} has no JPEG frame header", path.display()))?;
    let (w, h) = (info.width as u32, info.height as u32);
    anyhow::ensure!(out.len() >= (w * h * 4) as usize, "{}: short JPEG decode", path.display());
    Ok(Frame::owned(w, h, Layout::Rgba8, vec![out]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_padded_name_splits_at_its_frame_number() {
        let (p, n) = Pattern::of(Path::new("/x/shot_v2.1001.exr")).unwrap();
        assert_eq!((p.prefix.as_str(), p.suffix.as_str(), p.pad, n), ("shot_v2.", ".exr", 4, 1001));
        assert_eq!(p.path(7), Path::new("/x/shot_v2.0007.exr"));
        assert_eq!(p.number("shot_v2.0042.exr"), Some(42));
        assert_eq!(p.number("shot_v2.42.exr"), None);
        assert_eq!(p.number("shot_v2.10000.exr"), Some(10000));
        assert_eq!(p.number("other.0042.exr"), None);
    }

    #[test]
    fn a_sequence_reads_ahead_over_a_slow_link() {
        let t = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata/exr/shot.1001.exr");
        if !t.exists() {
            return;
        }
        std::env::set_var("SCRUB_NETCACHE_GB", "0");
        crate::netio::set_sim(Some(crate::netio::NetSim { bytes_per_sec: 200e6, latency: std::time::Duration::from_millis(5) }));
        let s = Sequence::open(&t).unwrap();
        assert!(s.net().unwrap().network);
        s.prefetch(&[0, 1, 2], true);
        let t0 = Instant::now();
        while s.net().unwrap().ahead < 20 && t0.elapsed().as_secs() < 10 {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        let ahead = s.net().unwrap().ahead;
        // A frame already read ahead decodes with no wait on the link.
        let t1 = Instant::now();
        let f = s.decode(15).unwrap();
        let ms = t1.elapsed().as_millis();
        crate::netio::set_sim(None);
        assert!(ahead >= 20, "read ahead {ahead}");
        assert_eq!((f.width, f.height), (1920, 1080));
        assert!(ms < 200, "{ms} ms");
    }

    #[test]
    fn an_unpadded_name_takes_any_length() {
        let (p, n) = Pattern::of(Path::new("frame7.png")).unwrap();
        assert_eq!((p.pad, n), (1, 7));
        assert_eq!(p.number("frame123.png"), Some(123));
        assert_eq!(p.number("frame0123.png"), None);
    }
}
