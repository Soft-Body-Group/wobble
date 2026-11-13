//! EXR decode behind a small trait so the backend is swappable (pure-Rust
//! `exr` today; `openexr` C-core bindings as a future fast path).

use anyhow::{Context as _, Result};
use half::f16;
use std::path::Path;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageInfo {
    pub width: u32,
    pub height: u32,
}

impl ImageInfo {
    /// Row stride in BYTES, padded to wgpu's 256-byte copy alignment.
    pub fn padded_row_bytes(&self) -> u32 {
        (self.width * 8).div_ceil(256) * 256
    }

    /// Row stride in f16 elements.
    pub fn row_stride_f16(&self) -> usize {
        self.padded_row_bytes() as usize / 2
    }

    /// Total staging bytes for one frame (padded rows).
    pub fn staging_bytes(&self) -> u64 {
        self.padded_row_bytes() as u64 * self.height as u64
    }
}

/// Which ONE channel a Load keeps when it stores a single channel.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SingleChannel {
    /// All four channels (the default).
    #[default]
    Off,
    Red,
    Green,
    Blue,
    Alpha,
    /// Rec.709 weights over the decoded rgb.
    Luma,
}

impl SingleChannel {
    pub const ALL: [SingleChannel; 6] = [
        SingleChannel::Off,
        SingleChannel::Red,
        SingleChannel::Green,
        SingleChannel::Blue,
        SingleChannel::Alpha,
        SingleChannel::Luma,
    ];

    pub fn label(self) -> &'static str {
        match self {
            SingleChannel::Off => "off (rgba)",
            SingleChannel::Red => "red",
            SingleChannel::Green => "green",
            SingleChannel::Blue => "blue",
            SingleChannel::Alpha => "alpha",
            SingleChannel::Luma => "luma",
        }
    }

    pub fn is_on(self) -> bool {
        self != SingleChannel::Off
    }
}

/// How a reader's frames are STORED from the decode on: in the RAM and NVMe
/// tiers, the staging ring and the GPU texture. The default is the classic
/// RGBA half float. A matte-like plate at 7680² is 450 MB a frame that way;
/// one 8-bit channel is 56 MB, which is the difference between an upload
/// that dominates the frame and one that does not.
///
/// The decoders still hand back RGBA f16; [`Pack::pack`] converts once, on
/// the way into the caches, so a cache hit is the compact bytes straight
/// into the slot. The shader broadcasts a single channel to all four
/// (`codegen`), and an 8-bit texture samples as 0..1 like any other.
/// The part of a plate a reader keeps: a pixel box of the frame the
/// decoder hands back (`ImageInfo`, proxy included). Everything outside it
/// is dropped on the way into the caches, the staging ring and the GPU —
/// a quarter-frame section of an 8K plate is 66 MB a frame instead of 265.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Window {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

impl Window {
    /// The window clamped inside `info`, never empty; `None` when it is
    /// the whole frame (which is not a window at all).
    pub fn fit(self, info: ImageInfo) -> Option<Window> {
        let x = self.x.min(info.width.saturating_sub(1));
        let y = self.y.min(info.height.saturating_sub(1));
        let w = self.w.clamp(1, info.width - x);
        let h = self.h.clamp(1, info.height - y);
        (w < info.width || h < info.height).then_some(Window { x, y, w, h })
    }

    /// The same box of the half-resolution frame (rounded outward).
    pub fn half(self) -> Window {
        let (x, y) = (self.x / 2, self.y / 2);
        Window {
            x,
            y,
            w: ((self.x + self.w).div_ceil(2) - x).max(1),
            h: ((self.y + self.h).div_ceil(2) - y).max(1),
        }
    }

    pub fn contains(&self, o: &Window) -> bool {
        o.x >= self.x && o.y >= self.y && o.x + o.w <= self.x + self.w && o.y + o.h <= self.y + self.h
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Pack {
    /// Keep only this box of the frame (`None`: all of it).
    pub window: Option<Window>,
    pub single: SingleChannel,
    /// Store 8-bit unorm: values clamp to 0..1 and quantise to 1/255.
    /// Lossless for an 8-bit source (PNG, JPEG code values); a float plate
    /// loses everything outside 0..1.
    pub eight: bool,
    /// The layer has two channels (a uv, a motion vector): store only those
    /// two, half the bytes of RGBA. Set by the app from the file, never
    /// authored; an RG texture samples as (r, g, 0, 1), which is exactly
    /// what the decoder fills a two-channel layer's rgba with, so nothing
    /// downstream changes. `single` wins when both are set.
    pub rg: bool,
}

impl Pack {
    pub fn is_default(&self) -> bool {
        *self == Pack::default()
    }

    /// Channels stored: 1 (single), 2 (rg) or 4.
    pub fn lanes(&self) -> u32 {
        if self.single.is_on() {
            1
        } else if self.rg {
            2
        } else {
            4
        }
    }

    pub fn bytes_per_pixel(&self) -> u32 {
        self.lanes() * if self.eight { 1 } else { 2 }
    }

    /// The size of what is stored: the window's, else the frame's. This is
    /// the reader texture's size and an upload's extent.
    pub fn dims(&self, info: ImageInfo) -> ImageInfo {
        match self.window.and_then(|w| w.fit(info)) {
            Some(w) => ImageInfo { width: w.w, height: w.h },
            None => info,
        }
    }

    /// Row stride in bytes, padded to wgpu's 256-byte copy alignment.
    pub fn padded_row_bytes(&self, info: ImageInfo) -> u32 {
        (self.dims(info).width * self.bytes_per_pixel()).div_ceil(256) * 256
    }

    pub fn frame_bytes(&self, info: ImageInfo) -> u64 {
        self.padded_row_bytes(info) as u64 * self.dims(info).height as u64
    }

    /// The `ImageInfo` whose RGBA f16 staging layout is exactly this pack's
    /// bytes: what the caches (which count in f16 rows) size and check a
    /// stored frame by. The identity for the default pack.
    pub fn storage_info(&self, info: ImageInfo) -> ImageInfo {
        if self.is_default() {
            return info;
        }
        ImageInfo { width: self.padded_row_bytes(info) / 8, height: self.dims(info).height }
    }

    /// The window as key material for the NVMe tier (`None` when whole).
    pub fn window_key(&self) -> Option<[u32; 4]> {
        self.window.map(|w| [w.x, w.y, w.w, w.h])
    }

    /// Folded into the NVMe key: two packs of one file are two entries.
    /// 0 for the default, so every existing entry keeps its key.
    pub fn tag(&self) -> u8 {
        if self.single == SingleChannel::Off && !self.eight && !self.rg {
            0
        } else if self.single.is_on() || !self.rg {
            1 + self.single as u8 * 2 + self.eight as u8
        } else {
            // RG: past every single-channel tag (1 + 5 * 2 + 1 = 12).
            13 + self.eight as u8
        }
    }

    /// Convert a decoded RGBA f16 frame (`info`'s staging layout) into this
    /// pack's layout. `dst` is only ever written, once per byte, so it may
    /// be a mapped staging slot.
    pub fn pack(&self, src: &[f16], info: ImageInfo, dst: &mut [u8]) {
        use rayon::prelude::*;
        let win = self.window.and_then(|w| w.fit(info));
        let (wx, wy) = win.map_or((0, 0), |w| (w.x as usize, w.y as usize));
        let d = self.dims(info);
        let (w, h) = (d.width as usize, d.height as usize);
        let src_row = info.row_stride_f16();
        let dst_row = self.padded_row_bytes(info) as usize;
        let single = self.single;
        let eight = self.eight;
        let q = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        let pick = move |p: &[f16]| -> f32 {
            match single {
                SingleChannel::Off | SingleChannel::Red => p[0].to_f32(),
                SingleChannel::Green => p[1].to_f32(),
                SingleChannel::Blue => p[2].to_f32(),
                SingleChannel::Alpha => p[3].to_f32(),
                SingleChannel::Luma => {
                    0.2126 * p[0].to_f32() + 0.7152 * p[1].to_f32() + 0.0722 * p[2].to_f32()
                }
            }
        };
        dst[..dst_row * h].par_chunks_mut(dst_row).enumerate().for_each(|(y, out)| {
            let at = (y + wy) * src_row + wx * 4;
            let row = &src[at..at + w * 4];
            let rg = !single.is_on() && self.rg;
            if rg {
                if eight {
                    for (o, p) in out[..w * 2].chunks_exact_mut(2).zip(row.chunks_exact(4)) {
                        o[0] = q(p[0].to_f32());
                        o[1] = q(p[1].to_f32());
                    }
                } else {
                    for (o, p) in out[..w * 4].chunks_exact_mut(4).zip(row.chunks_exact(4)) {
                        o[..2].copy_from_slice(&p[0].to_le_bytes());
                        o[2..].copy_from_slice(&p[1].to_le_bytes());
                    }
                }
                return;
            }
            match (single.is_on(), eight) {
                (true, true) => {
                    for (o, p) in out[..w].iter_mut().zip(row.chunks_exact(4)) {
                        *o = q(pick(p));
                    }
                }
                (true, false) => {
                    for (o, p) in out[..w * 2].chunks_exact_mut(2).zip(row.chunks_exact(4)) {
                        o.copy_from_slice(&f16::from_f32(pick(p)).to_le_bytes());
                    }
                }
                (false, true) => {
                    for (o, v) in out[..w * 4].iter_mut().zip(row) {
                        *o = q(v.to_f32());
                    }
                }
                (false, false) => {
                    out[..w * 8].copy_from_slice(bytemuck::cast_slice(row));
                }
            }
        });
    }
}

/// One selectable layer in a file: a group of channels sharing a name
/// prefix, which is how renderers write passes (`diffuse.R`, `N.X`,
/// `depth.Z`). The un-prefixed channels are the base layer, named "".
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerDesc {
    /// Prefix without the trailing dot; empty for the base RGBA layer.
    pub name: String,
    /// Channel names as they appear in the file, sorted.
    pub channels: Vec<String>,
    /// Which part of a multi-part file the channels live in. Only part 0 is
    /// readable today; the rest are listed (never hidden) and refused with a
    /// clear error rather than quietly decoding black.
    pub part: usize,
}

impl LayerDesc {
    /// What to show in a menu: the base layer has no name of its own.
    pub fn label(&self) -> &str {
        if self.name.is_empty() { "rgba" } else { &self.name }
    }

    /// The suffixes after the prefix (`R`, `G`, `Z`, …), in file order.
    pub fn suffixes(&self) -> Vec<&str> {
        self.channels
            .iter()
            .map(|c| match self.name.is_empty() {
                true => c.as_str(),
                false => c.get(self.name.len() + 1..).unwrap_or(c),
            })
            .collect()
    }
}

/// 2x2 box-downsample `src` (full res) into `dst` (half res, ceil), both
/// RGBA-interleaved f16 with their own row strides in f16 elements.
pub fn downsample_half(
    src: &[f16],
    sw: usize,
    sh: usize,
    s_stride: usize,
    dst: &mut [f16],
    d_stride: usize,
) {
    let (dw, dh) = (sw.div_ceil(2), sh.div_ceil(2));
    for y in 0..dh {
        for x in 0..dw {
            let (x0, y0) = (x * 2, y * 2);
            let (x1, y1) = ((x0 + 1).min(sw - 1), (y0 + 1).min(sh - 1));
            let di = y * d_stride + x * 4;
            for c in 0..4 {
                let sum = src[y0 * s_stride + x0 * 4 + c].to_f32()
                    + src[y0 * s_stride + x1 * 4 + c].to_f32()
                    + src[y1 * s_stride + x0 * 4 + c].to_f32()
                    + src[y1 * s_stride + x1 * 4 + c].to_f32();
                dst[di + c] = f16::from_f32(sum * 0.25);
            }
        }
    }
}

/// Half-resolution view over any decoder: headers report the halved frame,
/// decodes run the wrapped backend at FULL resolution and box-filter down.
/// The first touch of a frame costs the full decode; the point is what the
/// caches hold afterwards — quarter-size entries, quarter the IO and upload
/// from then on. Wraps every backend (EXR, DPX, stills, movies) the same
/// way.
pub struct ProxyDecoder {
    pub inner: Arc<dyn ExrDecoder>,
}

impl ProxyDecoder {
    fn full_into_half(
        &self,
        path: &Path,
        info: ImageInfo,
        row_stride: usize,
        dst: &mut [f16],
        decode: impl FnOnce(&mut [f16], ImageInfo, usize) -> Result<()>,
    ) -> Result<()> {
        let full = self.inner.read_header(path)?;
        // A file already at the asked size is already a proxy frame (a
        // Bake made under proxy): decode it as it is.
        if full.width == info.width && full.height == info.height {
            return decode(dst, full, row_stride);
        }
        let fs = full.row_stride_f16();
        let mut buf = vec![f16::ZERO; fs * full.height as usize];
        decode(&mut buf, full, fs)?;
        anyhow::ensure!(
            info.width as usize == (full.width as usize).div_ceil(2)
                && info.height as usize == (full.height as usize).div_ceil(2),
            "proxy dst is not the half frame of {}",
            path.display()
        );
        downsample_half(
            &buf,
            full.width as usize,
            full.height as usize,
            fs,
            dst,
            row_stride,
        );
        Ok(())
    }
}

impl ExrDecoder for ProxyDecoder {
    fn read_header(&self, path: &Path) -> Result<ImageInfo> {
        let f = self.inner.read_header(path)?;
        Ok(ImageInfo { width: f.width.div_ceil(2), height: f.height.div_ceil(2) })
    }

    fn decode_into(
        &self,
        path: &Path,
        info: ImageInfo,
        row_stride: usize,
        dst: &mut [f16],
    ) -> Result<()> {
        self.full_into_half(path, info, row_stride, dst, |buf, full, fs| {
            self.inner.decode_into(path, full, fs, buf)
        })
    }

    fn layers(&self, path: &Path) -> Result<Vec<LayerDesc>> {
        self.inner.layers(path)
    }

    fn decode_layer_into(
        &self,
        path: &Path,
        info: ImageInfo,
        row_stride: usize,
        layer: Option<&str>,
        dst: &mut [f16],
    ) -> Result<()> {
        self.full_into_half(path, info, row_stride, dst, |buf, full, fs| {
            self.inner.decode_layer_into(path, full, fs, layer, buf)
        })
    }

    fn frame_count(&self, path: &Path) -> Result<u64> {
        self.inner.frame_count(path)
    }

    fn decode_frame_into(
        &self,
        path: &Path,
        info: ImageInfo,
        row_stride: usize,
        layer: Option<&str>,
        frame: u64,
        dst: &mut [f16],
    ) -> Result<()> {
        self.full_into_half(path, info, row_stride, dst, |buf, full, fs| {
            self.inner.decode_frame_into(path, full, fs, layer, frame, buf)
        })
    }
}

pub trait ExrDecoder: Send + Sync {
    fn read_header(&self, path: &Path) -> Result<ImageInfo>;
    /// Decode into `dst`: row-major, RGBA-interleaved f16, with `row_stride`
    /// f16 elements between row starts (>= width*4; padding for wgpu's
    /// 256-byte bytes_per_row alignment).
    fn decode_into(
        &self,
        path: &Path,
        info: ImageInfo,
        row_stride: usize,
        dst: &mut [f16],
    ) -> Result<()>;

    /// The layers this file offers. Formats with a single fixed channel set
    /// (DPX) report none, which the UI reads as "nothing to choose".
    fn layers(&self, _path: &Path) -> Result<Vec<LayerDesc>> {
        Ok(Vec::new())
    }

    /// Decode one named layer (None = the file's own RGBA). Backends
    /// without layers ignore the name.
    fn decode_layer_into(
        &self,
        path: &Path,
        info: ImageInfo,
        row_stride: usize,
        _layer: Option<&str>,
        dst: &mut [f16],
    ) -> Result<()> {
        self.decode_into(path, info, row_stride, dst)
    }

    /// How many frames this file holds. One image per file — EXR, DPX, PNG,
    /// JPEG — is 1, which is why this defaults rather than being required.
    /// A movie reports its real length, and a `Sequence` built from it
    /// repeats the same path that many times.
    fn frame_count(&self, _path: &Path) -> Result<u64> {
        Ok(1)
    }

    /// Decode frame `frame` of a multi-frame CONTAINER. The frame index is
    /// the ONLY thing a movie needs that a still sequence does not — where
    /// one file is one image the index is meaningless and the default
    /// forwards to the single-image path, so no existing backend changes.
    fn decode_frame_into(
        &self,
        path: &Path,
        info: ImageInfo,
        row_stride: usize,
        layer: Option<&str>,
        _frame: u64,
        dst: &mut [f16],
    ) -> Result<()> {
        self.decode_layer_into(path, info, row_stride, layer, dst)
    }
}

/// Movie containers: ONE file holding many frames. Split out from the
/// still list because they behave differently everywhere it matters — the
/// browser lists one row instead of a sequence, and the loader builds a
/// Sequence from a frame COUNT rather than from a directory listing.
///
/// Only present when the `movie` feature is on: without an FFmpeg to link
/// against there is no backend, and offering the extension anyway would
/// mean a file that lists but never decodes.
#[cfg(feature = "movie")]
pub const MOVIE_EXTS: &[&str] = &["mov", "mp4", "m4v", "mxf", "avi", "mkv"];
/// In a page the browser decodes (WebCodecs, behind `set_web_movies`):
/// the MP4 family only, which is what its demuxer reads.
#[cfg(all(not(feature = "movie"), target_arch = "wasm32"))]
pub const MOVIE_EXTS: &[&str] = &["mov", "mp4", "m4v"];
#[cfg(all(not(feature = "movie"), not(target_arch = "wasm32")))]
pub const MOVIE_EXTS: &[&str] = &[];

/// Formats the app can read, stills then movies. One list, so the
/// browser's filter, the sequence scanner and the decoder can never
/// disagree about what "media" means.
#[cfg(feature = "movie")]
pub const SUPPORTED_EXTS: &[&str] = &[
    "exr", "dpx", "png", "jpg", "jpeg", "mov", "mp4", "m4v", "mxf", "avi", "mkv",
];
#[cfg(all(not(feature = "movie"), target_arch = "wasm32"))]
pub const SUPPORTED_EXTS: &[&str] = &["exr", "dpx", "png", "jpg", "jpeg", "mov", "mp4", "m4v"];
#[cfg(all(not(feature = "movie"), not(target_arch = "wasm32")))]
pub const SUPPORTED_EXTS: &[&str] = &["exr", "dpx", "png", "jpg", "jpeg"];

/// The page's movie decoder (the app's `webmain`, over WebCodecs): the
/// engine knows nothing about the browser, so the app hands it in.
#[cfg(target_arch = "wasm32")]
static WEB_MOVIES: std::sync::OnceLock<Box<dyn ExrDecoder>> = std::sync::OnceLock::new();

#[cfg(target_arch = "wasm32")]
pub fn set_web_movies(backend: Box<dyn ExrDecoder>) {
    let _ = WEB_MOVIES.set(backend);
}

fn ext_is(path: &Path, list: &[&str]) -> bool {
    path.extension().is_some_and(|e| {
        let e = e.to_string_lossy();
        list.iter().any(|s| e.eq_ignore_ascii_case(s))
    })
}

pub fn is_supported(path: &Path) -> bool {
    ext_is(path, SUPPORTED_EXTS)
}

/// True for a movie container — one path that plays many frames. Callers
/// that build timelines have to branch on this; callers that just decode
/// do not.
pub fn is_movie(path: &Path) -> bool {
    ext_is(path, MOVIE_EXTS)
}

/// Picks a backend per file by extension, so one sequence — or one comp —
/// can mix formats.
pub struct AnyDecoder {
    exr: ExrsDecoder,
    dpx: crate::dpx::DpxDecoder,
    still: crate::still::StillDecoder,
    #[cfg(feature = "movie")]
    movie: crate::movie::MovieDecoder,
}

impl AnyDecoder {
    pub fn new(single_threaded: bool) -> Self {
        Self {
            exr: ExrsDecoder { single_threaded },
            dpx: crate::dpx::DpxDecoder,
            still: crate::still::StillDecoder,
            #[cfg(feature = "movie")]
            movie: crate::movie::MovieDecoder::default(),
        }
    }

    fn backend(&self, path: &Path) -> &dyn ExrDecoder {
        #[cfg(feature = "movie")]
        if is_movie(path) {
            return &self.movie;
        }
        #[cfg(target_arch = "wasm32")]
        if is_movie(path) {
            if let Some(b) = WEB_MOVIES.get() {
                return b.as_ref();
            }
        }
        if ext_is(path, &["dpx"]) {
            &self.dpx
        } else if ext_is(path, &["png", "jpg", "jpeg"]) {
            &self.still
        } else {
            &self.exr
        }
    }
}

impl Default for AnyDecoder {
    fn default() -> Self {
        Self::new(true)
    }
}

impl ExrDecoder for AnyDecoder {
    fn read_header(&self, path: &Path) -> Result<ImageInfo> {
        self.backend(path).read_header(path)
    }

    fn decode_into(
        &self,
        path: &Path,
        info: ImageInfo,
        row_stride: usize,
        dst: &mut [f16],
    ) -> Result<()> {
        self.backend(path).decode_into(path, info, row_stride, dst)
    }

    fn layers(&self, path: &Path) -> Result<Vec<LayerDesc>> {
        self.backend(path).layers(path)
    }

    fn decode_layer_into(
        &self,
        path: &Path,
        info: ImageInfo,
        row_stride: usize,
        layer: Option<&str>,
        dst: &mut [f16],
    ) -> Result<()> {
        self.backend(path).decode_layer_into(path, info, row_stride, layer, dst)
    }

    fn frame_count(&self, path: &Path) -> Result<u64> {
        self.backend(path).frame_count(path)
    }

    fn decode_frame_into(
        &self,
        path: &Path,
        info: ImageInfo,
        row_stride: usize,
        layer: Option<&str>,
        frame: u64,
        dst: &mut [f16],
    ) -> Result<()> {
        self.backend(path)
            .decode_frame_into(path, info, row_stride, layer, frame, dst)
    }
}

/// Group channel names into layers by their prefix, the way renderers write
/// passes. `R`/`G`/`B`/`A` (and anything else un-prefixed) form the base
/// layer; `diffuse.R` joins the `diffuse` layer. Deterministic order: base
/// first, then alphabetical.
pub fn group_channels(names: impl IntoIterator<Item = (String, usize)>) -> Vec<LayerDesc> {
    let mut by_prefix: std::collections::BTreeMap<String, (Vec<String>, usize)> =
        Default::default();
    for (full, part) in names {
        // The prefix is everything before the LAST dot, so nested names
        // (`view.diffuse.R`) group by their full path.
        let prefix = match full.rfind('.') {
            Some(i) => full[..i].to_string(),
            None => String::new(),
        };
        let e = by_prefix.entry(prefix).or_insert_with(|| (Vec::new(), part));
        e.0.push(full);
        e.1 = e.1.min(part);
    }
    let mut out: Vec<LayerDesc> = by_prefix
        .into_iter()
        .map(|(name, (mut channels, part))| {
            channels.sort();
            LayerDesc { name, channels, part }
        })
        .collect();
    // Base layer first; it is what a Read shows by default.
    out.sort_by_key(|l| (!l.name.is_empty(), l.name.clone()));
    out
}

/// Which file channels feed R, G, B and A for a layer, and what to use when
/// one is missing. Colour triples map straight through; `X/Y/Z` vectors land
/// in RGB; a lone channel (a depth or a mask) is spread across RGB so it is
/// visible rather than red-only.
pub fn rgba_channels_of(layer: &LayerDesc) -> ([Option<String>; 4], [f16; 4]) {
    let full = |suffix: &str| -> Option<String> {
        let want = if layer.name.is_empty() {
            suffix.to_string()
        } else {
            format!("{}.{suffix}", layer.name)
        };
        layer.channels.iter().find(|c| *c == &want).cloned()
    };
    let first = |cands: &[&str]| cands.iter().find_map(|s| full(s));
    // A single channel (depth, a mask, an AO pass) reads as GREY whatever it
    // is called — `depth.Z` would otherwise land in blue alone.
    if layer.channels.len() == 1 {
        let c = layer.channels[0].clone();
        return (
            [Some(c.clone()), Some(c.clone()), Some(c), None],
            [f16::ZERO, f16::ZERO, f16::ZERO, f16::ONE],
        );
    }
    let r = first(&["R", "r", "X", "x", "U", "u"]);
    let g = first(&["G", "g", "Y", "y", "V", "v"]);
    let b = first(&["B", "b", "Z", "z", "W", "w"]);
    let a = first(&["A", "a"]);
    // Missing colour reads as black; a missing matte reads as OPAQUE, so a
    // pass written without alpha is not invisible.
    ([r, g, b, a], [f16::ZERO, f16::ZERO, f16::ZERO, f16::ONE])
}

/// Pure-Rust `exr` crate backend. Block decompression is internally
/// parallel (rayon) unless `single_threaded` is set; we run it
/// single-threaded per file because the player already decodes several
/// frames concurrently — per-file parallelism just adds contention there.
/// The bench can flip this to compare.
pub struct ExrsDecoder {
    pub single_threaded: bool,
}

impl Default for ExrsDecoder {
    fn default() -> Self {
        Self { single_threaded: true }
    }
}

/// A frame this many bytes or more (RGBA half) decodes with the file's
/// blocks in PARALLEL whatever `single_threaded` says. One thread a file
/// is right for 2K plates, where the player has a dozen files in flight;
/// an 8K frame is 580 ms on one thread against 185 ms in parallel, and the
/// player lets only a few of those run at once (`player::BIG_DECODES`).
pub const PARALLEL_FRAME_BYTES: u64 = 100 << 20;

impl ExrsDecoder {
    fn single_for(&self, info: ImageInfo) -> bool {
        self.single_threaded && info.staging_bytes() < PARALLEL_FRAME_BYTES
    }
}

impl ExrsDecoder {
    /// A cryptomatte's finished matte, computed HERE rather than on the GPU:
    /// all six ranks' f32 (id, coverage) pairs in one read of the file, the
    /// picked ids' coverage summed per pixel, written to all four channels.
    /// This is the slow half of the node done once — the player's cache
    /// tiers key on the layer string, so the result is kept per selection
    /// and the next request for this frame is a plain cache read.
    fn decode_matte_into(
        &self,
        path: &Path,
        info: ImageInfo,
        row_stride: usize,
        spec: &crate::cryptomatte::MatteSpec,
        dst: &mut [f16],
    ) -> Result<()> {
        use exr::prelude::*;
        use std::cell::Cell;

        let w = info.width as usize;
        anyhow::ensure!(row_stride >= w * 4, "row stride smaller than row");
        let needed = row_stride * info.height as usize;
        anyhow::ensure!(dst.len() >= needed, "decode dst too small");
        let ch = |pair: usize, c: &str| format!("{}{:02}.{c}", spec.layer, pair);
        let names: Vec<String> = (0..crate::cryptomatte::RANKS / 2)
            .flat_map(|p| ["R", "G", "B", "A"].map(|c| ch(p, c)))
            .collect();
        let spec = spec.clone();
        let slot = Cell::new(Some(&mut dst[..needed]));
        #[allow(clippy::type_complexity)]
        let reader = read()
            .no_deep_data()
            .largest_resolution_level()
            .specific_channels()
            .optional(names[0].as_str(), 0.0f32)
            .optional(names[1].as_str(), 0.0f32)
            .optional(names[2].as_str(), 0.0f32)
            .optional(names[3].as_str(), 0.0f32)
            .optional(names[4].as_str(), 0.0f32)
            .optional(names[5].as_str(), 0.0f32)
            .optional(names[6].as_str(), 0.0f32)
            .optional(names[7].as_str(), 0.0f32)
            .optional(names[8].as_str(), 0.0f32)
            .optional(names[9].as_str(), 0.0f32)
            .optional(names[10].as_str(), 0.0f32)
            .optional(names[11].as_str(), 0.0f32)
            .collect_pixels(
                |_size, _channels| slot.take().expect("create called twice"),
                move |buf: &mut &mut [f16],
                      pos,
                      (i0, c0, i1, c1, i2, c2, i3, c3, i4, c4, i5, c5): (
                    f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32, f32,
                )| {
                    let m = f16::from_f32(spec.matte(&[
                        (i0.to_bits(), c0),
                        (i1.to_bits(), c1),
                        (i2.to_bits(), c2),
                        (i3.to_bits(), c3),
                        (i4.to_bits(), c4),
                        (i5.to_bits(), c5),
                    ]));
                    let i = pos.y() * row_stride + pos.x() * 4;
                    buf[i] = m;
                    buf[i + 1] = m;
                    buf[i + 2] = m;
                    buf[i + 3] = m;
                },
            )
            .first_valid_layer()
            .all_attributes();
        let ctx = || format!("computing the matte of {}", path.display());
        if self.single_threaded {
            reader.non_parallel().from_buffered(crate::vfs::open(path)?).with_context(ctx)?;
        } else {
            reader.from_buffered(crate::vfs::open(path)?).with_context(ctx)?;
        }
        Ok(())
    }

    /// One rank of a cryptomatte layer: pair `half` (0 = R/G, 1 = B/A) of
    /// EXR layer `base`, written as `[chunk0, chunk1, chunk2, coverage]`.
    /// The channels are read as f32 — a cryptomatte id IS its bit pattern,
    /// and the crate would otherwise round it to half on the way in.
    fn decode_rank_into(
        &self,
        path: &Path,
        info: ImageInfo,
        row_stride: usize,
        base: &str,
        half: usize,
        dst: &mut [f16],
    ) -> Result<()> {
        use exr::prelude::*;
        use std::cell::Cell;

        let w = info.width as usize;
        anyhow::ensure!(row_stride >= w * 4, "row stride smaller than row");
        let needed = row_stride * info.height as usize;
        anyhow::ensure!(dst.len() >= needed, "decode dst too small");
        let (id_ch, cov_ch) = match half {
            0 => (format!("{base}.R"), format!("{base}.G")),
            _ => (format!("{base}.B"), format!("{base}.A")),
        };
        let fr = Framing::read(path)?;
        if fr.partial {
            dst[..needed].fill(f16::ZERO);
        }
        let slot = Cell::new(Some(&mut dst[..needed]));
        let reader = read()
            .no_deep_data()
            .largest_resolution_level()
            .specific_channels()
            .optional(id_ch.as_str(), 0.0f32)
            .optional(cov_ch.as_str(), 0.0f32)
            .collect_pixels(
                |_size, _channels| slot.take().expect("create called twice"),
                move |buf: &mut &mut [f16], pos, (id, cov): (f32, f32)| {
                    let Some(i) = fr.index(pos.x(), pos.y(), row_stride) else { return };
                    let c = crate::cryptomatte::encode_id(id.to_bits());
                    buf[i] = c[0];
                    buf[i + 1] = c[1];
                    buf[i + 2] = c[2];
                    buf[i + 3] = f16::from_f32(cov);
                },
            )
            .first_valid_layer()
            .all_attributes();
        let ctx = || format!("decoding rank {half} of '{base}' in {}", path.display());
        if self.single_threaded {
            reader.non_parallel().from_buffered(crate::vfs::open(path)?).with_context(ctx)?;
        } else {
            reader.from_buffered(crate::vfs::open(path)?).with_context(ctx)?;
        }
        Ok(())
    }
}

/// Where an EXR's DATA window (the pixel bounding box) lands inside its
/// DISPLAY window (the framing — what the image's resolution MEANS). A
/// tight-bboxed render must come in at the framing's resolution with its
/// pixels placed, not at the bbox's; overscan must crop.
struct Framing {
    /// Data-window pixel -> frame pixel offset.
    dx: i64,
    dy: i64,
    /// Frame size (the display window).
    w: i64,
    h: i64,
    /// The bbox does not exactly cover the frame: the buffer needs a clear
    /// first (padding), and writes need a bounds check (overscan).
    partial: bool,
}

impl Framing {
    /// Each PART of a multi-part file has its own data window; the display
    /// window is shared.
    fn of(header: &exr::meta::header::Header) -> Framing {
        let disp = header.shared_attributes.display_window;
        let dpos = header.own_attributes.layer_position;
        let dx = dpos.x() as i64 - disp.position.x() as i64;
        let dy = dpos.y() as i64 - disp.position.y() as i64;
        let (w, h) = (disp.size.width() as i64, disp.size.height() as i64);
        let partial = dx != 0
            || dy != 0
            || header.layer_size.width() as i64 != w
            || header.layer_size.height() as i64 != h;
        Framing { dx, dy, w, h, partial }
    }

    fn read(path: &Path) -> Result<Framing> {
        let meta = exr::meta::MetaData::read_from_buffered(crate::vfs::open(path)?, false)
            .with_context(|| format!("reading header of {}", path.display()))?;
        Ok(Framing::of(&meta.headers[0]))
    }

    /// The write index for a data-window pixel, or None outside the frame.
    #[inline]
    fn index(&self, x: usize, y: usize, row_stride: usize) -> Option<usize> {
        let (fx, fy) = (x as i64 + self.dx, y as i64 + self.dy);
        if fx < 0 || fy < 0 || fx >= self.w || fy >= self.h {
            return None;
        }
        Some(fy as usize * row_stride + fx as usize * 4)
    }
}

impl ExrsDecoder {
    /// Decode one PART of a multi-part EXR into the frame. A filtered block
    /// read: only that part's chunks are decompressed — a 20-AOV multi-part
    /// costs one part, not twenty. `chans` are the part's REAL channel
    /// names feeding r/g/b/a; a missing channel reads its default (inside
    /// the data window; outside is padding like everywhere else).
    fn decode_part_into(
        &self,
        path: &Path,
        info: ImageInfo,
        row_stride: usize,
        part: usize,
        chans: &[Option<String>; 4],
        defaults: [f16; 4],
        dst: &mut [f16],
    ) -> Result<()> {
        use exr::block::reader::ChunksReader as _;
        use exr::meta::attribute::SampleType;

        let w = info.width as usize;
        anyhow::ensure!(row_stride >= w * 4, "row stride smaller than row");
        let needed = row_stride * info.height as usize;
        anyhow::ensure!(dst.len() >= needed, "decode dst too small");

        let file = crate::vfs::open(path)?;
        let reader = exr::block::read(file, false)
            .with_context(|| format!("reading {}", path.display()))?;
        anyhow::ensure!(
            part < reader.meta_data().headers.len(),
            "{} has no part {part}",
            path.display()
        );
        // Cloned: `filter_chunks` consumes the reader below.
        let header = reader.meta_data().headers[part].clone();
        let fr = Framing::of(&header);
        // Which rgba slots each of the part's channels feeds (a lone grey
        // channel feeds three).
        let list = &header.channels.list;
        let mut feeds: Vec<Vec<usize>> = vec![Vec::new(); list.len()];
        let mut missing: [bool; 4] = [false; 4];
        for (slot, want) in chans.iter().enumerate() {
            match want.as_ref().and_then(|n| list.iter().position(|c| c.name.eq(n.as_str()))) {
                Some(ci) => feeds[ci].push(slot),
                None => missing[slot] = true,
            }
        }
        let types: Vec<SampleType> = list.iter().map(|c| c.sample_type).collect();
        // Padding first; the data window then fills its own rect (including
        // per-pixel defaults for channels the part does not have).
        dst[..needed].fill(f16::ZERO);
        let buf = &mut dst[..needed];
        let single = self.single_for(info);
        let decode = |_meta: &exr::meta::MetaData, block: exr::block::UncompressedBlock| {
            // Defaults over this block's rect for absent channels.
            if missing.iter().any(|m| *m) {
                for by in 0..block.index.pixel_size.height() {
                    for bx in 0..block.index.pixel_size.width() {
                        let (x, y) = (
                            block.index.pixel_position.x() + bx,
                            block.index.pixel_position.y() + by,
                        );
                        if let Some(i) = fr.index(x, y, row_stride) {
                            for (slot, m) in missing.iter().enumerate() {
                                if *m {
                                    buf[i + slot] = defaults[slot];
                                }
                            }
                        }
                    }
                }
            }
            for line in block.lines(&header.channels) {
                let slots = &feeds[line.location.channel];
                if slots.is_empty() {
                    continue;
                }
                let (lx, ly) = (line.location.position.x(), line.location.position.y());
                let mut put = |k: usize, v: f16| {
                    if let Some(i) = fr.index(lx + k, ly, row_stride) {
                        for slot in slots {
                            buf[i + slot] = v;
                        }
                    }
                };
                match types[line.location.channel] {
                    SampleType::F16 => {
                        for (k, v) in line.read_samples::<f16>().enumerate() {
                            put(k, v?);
                        }
                    }
                    SampleType::F32 => {
                        for (k, v) in line.read_samples::<f32>().enumerate() {
                            put(k, f16::from_f32(v?));
                        }
                    }
                    SampleType::U32 => {
                        for (k, v) in line.read_samples::<u32>().enumerate() {
                            put(k, f16::from_f32(v? as f32));
                        }
                    }
                }
            }
            Ok(())
        };
        let filtered = reader
            .filter_chunks(false, |_meta, _tile, block| block.layer == part)
            .with_context(|| format!("indexing {}", path.display()))?;
        let res = if single {
            filtered.decompress_sequential(false, decode)
        } else {
            filtered.decompress_parallel(false, decode)
        };
        res.with_context(|| format!("decoding part {part} of {}", path.display()))?;
        Ok(())
    }
}

impl ExrDecoder for ExrsDecoder {
    fn read_header(&self, path: &Path) -> Result<ImageInfo> {
        let meta = exr::meta::MetaData::read_from_buffered(crate::vfs::open(path)?, false)
            .with_context(|| format!("reading header of {}", path.display()))?;
        // The FRAMING is the display window; the layer size is only the
        // pixel bounding box (a tight render, or overscan).
        let size = meta.headers[0].shared_attributes.display_window.size;
        Ok(ImageInfo { width: size.width() as u32, height: size.height() as u32 })
    }

    fn decode_into(
        &self,
        path: &Path,
        info: ImageInfo,
        row_stride: usize,
        dst: &mut [f16],
    ) -> Result<()> {
        use exr::prelude::*;
        use std::cell::Cell;

        let w = info.width as usize;
        anyhow::ensure!(row_stride >= w * 4, "row stride smaller than row");
        let needed = row_stride * info.height as usize;
        anyhow::ensure!(dst.len() >= needed, "decode dst too small");

        let meta = exr::meta::MetaData::read_from_buffered(crate::vfs::open(path)?, false)
            .with_context(|| format!("reading header of {}", path.display()))?;
        // The RGBA read takes the first part with R, G AND B. A file with
        // none (a cryptomatte-only render, an AOV-only part set) used to
        // fail every frame with "no layer matched"; it shows its FIRST
        // layer instead, the way picking that layer by hand would.
        let has_rgb = meta.headers.iter().any(|h| {
            ["R", "G", "B"].iter().all(|c| h.channels.list.iter().any(|ch| ch.name.eq(*c)))
        });
        if !has_rgb {
            let desc = self
                .layers(path)?
                .into_iter()
                .next()
                .with_context(|| format!("{} has no channels", path.display()))?;
            return self.decode_desc_into(path, info, row_stride, &desc, dst);
        }
        let fr = Framing::of(&meta.headers[0]);
        if fr.partial {
            dst[..needed].fill(f16::ZERO);
        }
        let slot = Cell::new(Some(&mut dst[..needed]));
        let reader = read()
            .no_deep_data()
            .largest_resolution_level()
            .rgba_channels(
                |_resolution, _channels| slot.take().expect("create called twice"),
                move |buf: &mut &mut [f16], pos, (r, g, b, a): (f16, f16, f16, f16)| {
                    let Some(i) = fr.index(pos.x(), pos.y(), row_stride) else { return };
                    buf[i] = r;
                    buf[i + 1] = g;
                    buf[i + 2] = b;
                    buf[i + 3] = a;
                },
            )
            .first_valid_layer()
            .all_attributes();

        if self.single_for(info) {
            reader
                .non_parallel()
                .from_buffered(crate::vfs::open(path)?)
                .with_context(|| format!("decoding {}", path.display()))?;
        } else {
            reader.from_buffered(crate::vfs::open(path)?).with_context(|| format!("decoding {}", path.display()))?;
        }
        Ok(())
    }

    fn layers(&self, path: &Path) -> Result<Vec<LayerDesc>> {
        let meta = exr::meta::MetaData::read_from_buffered(crate::vfs::open(path)?, false)
            .with_context(|| format!("reading header of {}", path.display()))?;
        // Multi-part files: every part's channels are offered, prefixed by
        // the part name when it has one, so nothing is silently hidden.
        let mut names = Vec::new();
        // A single-part file's part NAME is just a label ("rgba"), not a
        // channel prefix — only a multi-part file needs it to tell two
        // same-named channels apart.
        let multi = meta.headers.len() > 1;
        for (i, header) in meta.headers.iter().enumerate() {
            let part = header.own_attributes.layer_name.as_ref().map(|t| t.to_string());
            for ch in header.channels.list.iter() {
                let base = ch.name.to_string();
                let full = match &part {
                    Some(p) if multi && !p.is_empty() => format!("{p}.{base}"),
                    _ if multi => format!("part{i}.{base}"),
                    _ => base,
                };
                names.push((full, i));
            }
        }
        Ok(group_channels(names))
    }

    fn decode_layer_into(
        &self,
        path: &Path,
        info: ImageInfo,
        row_stride: usize,
        layer: Option<&str>,
        dst: &mut [f16],
    ) -> Result<()> {

        // No selection (or the base layer) is the plain RGBA path, which is
        // the fast one every existing Read already takes.
        let Some(name) = layer.filter(|l| !l.is_empty()) else {
            return self.decode_into(path, info, row_stride, dst);
        };
        // A cryptomatte RANK (`CryptoObject00#1`): f32 ids that must survive
        // the f16 texture bit-exact, so they go in as integer chunks. See
        // `cryptomatte.rs` for the encoding both sides share.
        // A finished cryptomatte MATTE (`CryptoObject@matte;inv=0;ids=…`):
        // the node's decode-side bake. Checked first — its name carries a
        // `#`-free tag, but order makes the precedence explicit.
        if let Some(spec) = crate::cryptomatte::parse_matte_layer(name) {
            return self.decode_matte_into(path, info, row_stride, &spec, dst);
        }
        if let Some((base, half)) = crate::cryptomatte::parse_rank_layer(name) {
            return self.decode_rank_into(path, info, row_stride, base, half, dst);
        }

        let desc = self
            .layers(path)?
            .into_iter()
            .find(|l| l.name == name)
            .with_context(|| format!("{} has no layer '{name}'", path.display()))?;
        self.decode_desc_into(path, info, row_stride, &desc, dst)
    }
}

impl ExrsDecoder {
    /// Decode one layer as `layers()` described it: its channels mapped
    /// onto rgba by `rgba_channels_of`, from whichever part holds it.
    fn decode_desc_into(
        &self,
        path: &Path,
        info: ImageInfo,
        row_stride: usize,
        desc: &LayerDesc,
        dst: &mut [f16],
    ) -> Result<()> {
        use exr::prelude::*;
        use std::cell::Cell;

        let name = desc.label();
        let w = info.width as usize;
        anyhow::ensure!(row_stride >= w * 4, "row stride smaller than row");
        let needed = row_stride * info.height as usize;
        anyhow::ensure!(dst.len() >= needed, "decode dst too small");
        // In a MULTI-part file `layers()` prefixed every part's channels
        // with the part label for display — part 0's included — so the
        // label comes back off before the reader is asked for a channel.
        // (Part 0 named "beauty" once listed as `beauty.R` and was then
        // read as `beauty.R`, which no part has: a black layer.)
        let meta = exr::meta::MetaData::read_from_buffered(crate::vfs::open(path)?, false)
            .with_context(|| format!("reading header of {}", path.display()))?;
        let (chans, defaults) = rgba_channels_of(desc);
        let chans = if meta.headers.len() > 1 {
            let h = &meta.headers[desc.part];
            let label = h
                .own_attributes
                .layer_name
                .as_ref()
                .map(|t| t.to_string())
                .filter(|l| !l.is_empty())
                .unwrap_or_else(|| format!("part{}", desc.part));
            let strip = |c: &Option<String>| {
                c.as_ref().map(|c| {
                    c.strip_prefix(&format!("{label}."))
                        .map(str::to_string)
                        .unwrap_or_else(|| c.clone())
                })
            };
            [strip(&chans[0]), strip(&chans[1]), strip(&chans[2]), strip(&chans[3])]
        } else {
            chans
        };
        // A layer in another PART takes the filtered block path: only that
        // part's chunks decompress.
        if desc.part > 0 {
            return self.decode_part_into(
                path, info, row_stride, desc.part, &chans, defaults, dst,
            );
        }
        // A lone channel (depth, a mask) is read ONCE and spread over rgb in
        // the write below; asking the reader for the same channel three
        // times is not something the format contemplates.
        let grey = chans[0].is_some() && chans[0] == chans[1] && chans[1] == chans[2];
        // Every output needs SOME channel name for the reader; where the
        // layer has none, read a channel that is certainly absent and let
        // the optional default stand in.
        // The placeholder differs per slot: the reader refuses to be asked
        // for the same channel name twice, and several slots can be absent.
        let missing = |i: usize| format!("__wobble_absent_{i}__");
        let pick = |i: usize| match (grey, i) {
            (true, 1 | 2) => missing(i),
            _ => chans[i].clone().unwrap_or_else(|| missing(i)),
        };

        let fr = Framing::read(path)?;
        if fr.partial {
            dst[..needed].fill(f16::ZERO);
        }
        let slot = Cell::new(Some(&mut dst[..needed]));
        let reader = read()
            .no_deep_data()
            .largest_resolution_level()
            .specific_channels()
            .optional(pick(0).as_str(), defaults[0])
            .optional(pick(1).as_str(), defaults[1])
            .optional(pick(2).as_str(), defaults[2])
            .optional(pick(3).as_str(), defaults[3])
            .collect_pixels(
                |_size, _channels| slot.take().expect("create called twice"),
                move |buf: &mut &mut [f16], pos, (r, g, b, a): (f16, f16, f16, f16)| {
                    let Some(i) = fr.index(pos.x(), pos.y(), row_stride) else { return };
                    buf[i] = r;
                    buf[i + 1] = if grey { r } else { g };
                    buf[i + 2] = if grey { r } else { b };
                    buf[i + 3] = a;
                },
            )
            .first_valid_layer()
            .all_attributes();

        if self.single_for(info) {
            reader
                .non_parallel()
                .from_buffered(crate::vfs::open(path)?)
                .with_context(|| format!("decoding layer '{name}' of {}", path.display()))?;
        } else {
            reader
                .from_buffered(crate::vfs::open(path)?)
                .with_context(|| format!("decoding layer '{name}' of {}", path.display()))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod pack_tests {
    use super::*;

    fn frame(info: ImageInfo, px: impl Fn(usize, usize) -> [f32; 4]) -> Vec<f16> {
        let stride = info.row_stride_f16();
        let mut v = vec![f16::ZERO; stride * info.height as usize];
        for y in 0..info.height as usize {
            for x in 0..info.width as usize {
                for (c, val) in px(x, y).iter().enumerate() {
                    v[y * stride + x * 4 + c] = f16::from_f32(*val);
                }
            }
        }
        v
    }

    #[test]
    fn sizes_shrink_and_storage_info_matches_the_bytes() {
        let info = ImageInfo { width: 7680, height: 7680 };
        let whole = Pack::default().frame_bytes(info);
        assert_eq!(whole, info.staging_bytes());
        for (pack, div) in [
            (Pack { single: SingleChannel::Red, eight: false, rg: false, window: None }, 4),
            (Pack { single: SingleChannel::Off, eight: true, rg: false, window: None }, 2),
            (Pack { single: SingleChannel::Alpha, eight: true, rg: false, window: None }, 8),
        ] {
            assert_eq!(pack.frame_bytes(info) * div, whole, "{pack:?}");
            // The caches size and check a stored frame by this.
            assert_eq!(pack.storage_info(info).staging_bytes(), pack.frame_bytes(info), "{pack:?}");
        }
        // An odd width still pads every row to 256 bytes.
        let odd = ImageInfo { width: 301, height: 3 };
        let p = Pack { single: SingleChannel::Red, eight: true, rg: false, window: None };
        assert_eq!(p.padded_row_bytes(odd), 512);
        assert_eq!(p.storage_info(odd).staging_bytes(), p.frame_bytes(odd));
    }

    /// A windowed pack stores the box alone, and the bytes are the box's.
    #[test]
    fn a_window_keeps_its_box() {
        let info = ImageInfo { width: 8, height: 6 };
        let src = frame(info, |x, y| [x as f32, y as f32, 0.0, 1.0]);
        let p = Pack { window: Some(Window { x: 3, y: 2, w: 4, h: 3 }), ..Default::default() };
        assert_eq!(p.dims(info), ImageInfo { width: 4, height: 3 });
        assert_eq!(p.storage_info(info).staging_bytes(), p.frame_bytes(info));
        let mut out = vec![0u8; p.frame_bytes(info) as usize];
        p.pack(&src, info, &mut out);
        let row = p.padded_row_bytes(info) as usize;
        let px = |x: usize, y: usize| {
            let o = y * row + x * 8;
            [f16::from_le_bytes([out[o], out[o + 1]]).to_f32(), f16::from_le_bytes([out[o + 2], out[o + 3]]).to_f32()]
        };
        assert_eq!(px(0, 0), [3.0, 2.0]);
        assert_eq!(px(3, 2), [6.0, 4.0]);
        // Half res rounds outward; the whole frame is no window.
        assert_eq!(Window { x: 3, y: 2, w: 4, h: 3 }.half(), Window { x: 1, y: 1, w: 3, h: 2 });
        assert_eq!(Window { x: 0, y: 0, w: 8, h: 6 }.fit(info), None);
        assert_eq!(Window { x: 6, y: 0, w: 9, h: 6 }.fit(info), Some(Window { x: 6, y: 0, w: 2, h: 6 }));
    }

    #[test]
    fn a_two_channel_layer_stores_rg_at_half_the_bytes() {
        let info = ImageInfo { width: 5, height: 2 };
        let rg = Pack { rg: true, ..Default::default() };
        let wide = ImageInfo { width: 6000, height: 2 };
        assert_eq!(rg.frame_bytes(wide) * 2, Pack::default().frame_bytes(wide), "half of RGBA");
        assert_eq!(rg.bytes_per_pixel(), 4);
        assert_eq!(rg.storage_info(info).staging_bytes(), rg.frame_bytes(info));
        let src = frame(info, |x, _| [x as f32, -0.5, 9.0, 1.0]);
        let mut out = vec![0u8; rg.frame_bytes(info) as usize];
        rg.pack(&src, info, &mut out);
        let at = |i: usize| f16::from_le_bytes([out[i], out[i + 1]]).to_f32();
        assert_eq!((at(3 * 4), at(3 * 4 + 2)), (3.0, -0.5), "r and g, nothing else");
        // Its own NVMe key, apart from every other pack.
        assert!(![Pack::default(), Pack { eight: true, ..Default::default() }].iter().any(|p| p.tag() == rg.tag()));
        assert_ne!(rg.tag(), Pack { rg: true, eight: true, ..Default::default() }.tag());
    }

    #[test]
    fn tags_are_distinct_and_the_default_keeps_old_keys() {
        assert_eq!(Pack::default().tag(), 0);
        let mut seen = std::collections::HashSet::new();
        for single in SingleChannel::ALL {
            for eight in [false, true] {
                assert!(seen.insert(Pack { single, eight, rg: false, window: None }.tag()), "{single:?} {eight}");
            }
        }
    }

    #[test]
    fn packing_keeps_the_picked_channel() {
        let info = ImageInfo { width: 5, height: 2 };
        let src = frame(info, |x, y| [x as f32 / 4.0, 0.5, 2.0, y as f32]);
        let mut r8 = vec![0u8; Pack { single: SingleChannel::Red, eight: true, rg: false, window: None }.frame_bytes(info) as usize];
        Pack { single: SingleChannel::Red, eight: true, rg: false, window: None }.pack(&src, info, &mut r8);
        assert_eq!(&r8[..5], &[0, 64, 128, 191, 255]);
        // Clamped to 0..1 in 8 bits; alpha is row 1's 1.0.
        let mut a8 = vec![0u8; Pack { single: SingleChannel::Alpha, eight: true, rg: false, window: None }.frame_bytes(info) as usize];
        Pack { single: SingleChannel::Alpha, eight: true, rg: false, window: None }.pack(&src, info, &mut a8);
        let row = Pack { single: SingleChannel::Alpha, eight: true, rg: false, window: None }.padded_row_bytes(info) as usize;
        assert_eq!((a8[0], a8[row]), (0, 255));
        // Half float keeps what 8 bits clamps.
        let p = Pack { single: SingleChannel::Blue, eight: false, rg: false, window: None };
        let mut b16 = vec![0u8; p.frame_bytes(info) as usize];
        p.pack(&src, info, &mut b16);
        assert_eq!(f16::from_le_bytes([b16[0], b16[1]]).to_f32(), 2.0);
        // RGBA 8-bit, four bytes a pixel.
        let p = Pack { single: SingleChannel::Off, eight: true, rg: false, window: None };
        let mut c8 = vec![0u8; p.frame_bytes(info) as usize];
        p.pack(&src, info, &mut c8);
        assert_eq!(&c8[4..8], &[64, 128, 255, 0]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The proxy wrapper: headers halve (ceil), pixels box-average, and the
    /// downsample places rows through the padded stride correctly.
    #[test]
    fn proxy_decoder_halves_and_averages() {
        use exr::prelude::*;
        let dir = std::env::temp_dir()
            .join(format!("wobble_test_proxy_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("checker.exr");
        // 4x4 checker: 1.0 / 0.0 alternating -> every 2x2 box averages 0.5.
        let layer = Layer::new(
            Vec2(5usize, 3usize), // odd sizes: ceil-halving + edge clamp
            LayerAttributes::named("rgba"),
            Encoding::FAST_LOSSLESS,
            SpecificChannels::rgba(|p: Vec2<usize>| {
                let v = ((p.x() + p.y()) % 2) as f32;
                (v, v, v, 1.0f32)
            }),
        );
        let img = Image::from_layer(layer);
        img.write().to_file(&path).unwrap();

        let dec = ProxyDecoder {
            inner: Arc::new(ExrsDecoder { single_threaded: true }),
        };
        let info = dec.read_header(&path).unwrap();
        assert_eq!((info.width, info.height), (3, 2), "ceil halves of 5x3");
        let stride = info.row_stride_f16();
        let mut dst = vec![f16::ZERO; stride * 2];
        dec.decode_into(&path, info, stride, &mut dst).unwrap();
        // Interior boxes of an alternating checker average 0.5 exactly.
        assert_eq!(dst[0].to_f32(), 0.5, "box (0,0)");
        assert_eq!(dst[4].to_f32(), 0.5, "box (1,0)");
        assert_eq!(dst[3].to_f32(), 1.0, "alpha stays 1");
        // Edge boxes clamp: column 4 duplicated, row 2 duplicated.
        assert!(dst[2 * 4 + 1].to_f32() <= 1.0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A MULTI-PART file: part 0 the beauty, part 1 an AOV part named
    /// "env" with its own TIGHT data window. Selecting the env layer must
    /// decode part 1 (not part 0's pixels), framed by the shared display
    /// window with the part's bbox placed and padded.
    #[test]
    fn multipart_layers_decode_their_own_part_framed() {
        use exr::prelude::*;
        let dir = std::env::temp_dir()
            .join(format!("wobble_test_multipart_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("multi.exr");
        let beauty = Layer::new(
            Vec2(16usize, 8usize),
            LayerAttributes::named("rgba"),
            Encoding::FAST_LOSSLESS,
            SpecificChannels::rgba(
                (|_| (0.1f32, 0.1f32, 0.1f32, 1.0f32))
                    as fn(Vec2<usize>) -> (f32, f32, f32, f32),
            ),
        );
        let env = Layer::new(
            Vec2(6usize, 4usize),
            LayerAttributes {
                layer_position: Vec2(4i32, 2i32),
                ..LayerAttributes::named("env")
            },
            Encoding::FAST_LOSSLESS,
            SpecificChannels::rgba(
                (|_| (0.75f32, 0.5f32, 0.25f32, 1.0f32))
                    as fn(Vec2<usize>) -> (f32, f32, f32, f32),
            ),
        );
        let layers: Layers<_> = [beauty, env].into_iter().collect();
        let mut img = Image::from_layers(
            ImageAttributes::new(IntegerBounds::new(Vec2(0i32, 0i32), Vec2(16usize, 8usize))),
            layers,
        );
        img.attributes.display_window =
            IntegerBounds::new(Vec2(0i32, 0i32), Vec2(16usize, 8usize));
        img.write().to_file(&path).unwrap();

        let dec = ExrsDecoder { single_threaded: true };
        let layers = dec.layers(&path).unwrap();
        let env_layer = layers.iter().find(|l| l.name.contains("env")).expect("env listed");
        assert!(env_layer.part > 0, "env lives in its own part");

        let info = dec.read_header(&path).unwrap();
        assert_eq!((info.width, info.height), (16, 8));
        let stride = info.row_stride_f16();
        let mut dst = vec![f16::from_f32(9.0); stride * 8];
        dec.decode_layer_into(&path, info, stride, Some(&env_layer.name), &mut dst)
            .unwrap();
        let px = |x: usize, y: usize| {
            let i = y * stride + x * 4;
            [dst[i].to_f32(), dst[i + 1].to_f32(), dst[i + 2].to_f32(), dst[i + 3].to_f32()]
        };
        assert_eq!(px(4, 2), [0.75, 0.5, 0.25, 1.0], "part 1's pixels, placed");
        assert_eq!(px(9, 5), [0.75, 0.5, 0.25, 1.0], "bbox far corner");
        assert_eq!(px(0, 0), [0.0; 4], "outside the part's bbox is padding");
        assert_eq!(px(10, 2), [0.0; 4], "just past the bbox");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Write an EXR whose DATA window (bbox) differs from its DISPLAY
    /// window, decode it, and assert the framing came from the display
    /// window with the bbox PLACED inside it — the tight-render case every
    /// 3D renderer produces with autocrop on.
    #[test]
    fn framing_comes_from_the_display_window_not_the_bbox() {
        use exr::prelude::*;
        let dir = std::env::temp_dir()
            .join(format!("wobble_test_dispwin_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tight.exr");
        // 8x4 of solid (1, 0.25, 0, 1) placed at (10, 6) in a 64x32 frame.
        let layer = Layer::new(
            Vec2(8usize, 4usize),
            LayerAttributes {
                layer_position: Vec2(10i32, 6i32),
                ..Default::default()
            },
            Encoding::FAST_LOSSLESS,
            SpecificChannels::rgba(|_: Vec2<usize>| (1.0f32, 0.25f32, 0.0f32, 1.0f32)),
        );
        let mut img = Image::from_layer(layer);
        img.attributes.display_window =
            IntegerBounds::new(Vec2(0i32, 0i32), Vec2(64usize, 32usize));
        img.write().to_file(&path).unwrap();

        let dec = ExrsDecoder { single_threaded: true };
        let info = dec.read_header(&path).unwrap();
        assert_eq!((info.width, info.height), (64, 32), "framing = display window");
        let stride = info.row_stride_f16();
        // Sentinel fill proves the padding is CLEARED, not stale.
        let mut dst = vec![f16::from_f32(9.0); stride * 32];
        dec.decode_into(&path, info, stride, &mut dst).unwrap();
        let px = |x: usize, y: usize| {
            let i = y * stride + x * 4;
            [dst[i].to_f32(), dst[i + 1].to_f32(), dst[i + 2].to_f32(), dst[i + 3].to_f32()]
        };
        assert_eq!(px(10, 6), [1.0, 0.25, 0.0, 1.0], "bbox pixel placed");
        assert_eq!(px(17, 9), [1.0, 0.25, 0.0, 1.0], "bbox far corner placed");
        assert_eq!(px(0, 0), [0.0; 4], "outside the bbox is cleared padding");
        assert_eq!(px(18, 6), [0.0; 4], "just past the bbox is padding");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Overscan: a bbox EXTENDING PAST the display window crops instead of
    /// crashing, and the frame stays display-sized.
    #[test]
    fn overscan_bboxes_crop_to_the_display_window() {
        use exr::prelude::*;
        let dir = std::env::temp_dir()
            .join(format!("wobble_test_overscan_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("overscan.exr");
        // 24x16 of data at (-4, -4) over a 16x8 frame: margins on all sides.
        let layer = Layer::new(
            Vec2(24usize, 16usize),
            LayerAttributes {
                layer_position: Vec2(-4i32, -4i32),
                ..Default::default()
            },
            Encoding::FAST_LOSSLESS,
            SpecificChannels::rgba(|_: Vec2<usize>| (0.5f32, 0.5f32, 0.5f32, 1.0f32)),
        );
        let mut img = Image::from_layer(layer);
        img.attributes.display_window =
            IntegerBounds::new(Vec2(0i32, 0i32), Vec2(16usize, 8usize));
        img.write().to_file(&path).unwrap();

        let dec = ExrsDecoder { single_threaded: true };
        let info = dec.read_header(&path).unwrap();
        assert_eq!((info.width, info.height), (16, 8));
        let stride = info.row_stride_f16();
        let mut dst = vec![f16::ZERO; stride * 8];
        dec.decode_into(&path, info, stride, &mut dst).unwrap();
        for (x, y) in [(0usize, 0usize), (15, 7), (8, 4)] {
            let i = y * stride + x * 4;
            assert_eq!(dst[i].to_f32(), 0.5, "({x},{y}) covered by the overscan data");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn channels_group_into_layers_base_first() {
        let layers = group_channels(
            [
                "diffuse.G", "R", "A", "depth.Z", "diffuse.R", "G", "B",
                "diffuse.B", "N.X", "N.Y", "N.Z",
            ]
            .map(|s| (s.to_string(), 0)),
        );
        let names: Vec<&str> = layers.iter().map(|l| l.label()).collect();
        assert_eq!(names, ["rgba", "N", "depth", "diffuse"], "base first, then sorted");
        // The base layer is the UN-prefixed channels only.
        assert_eq!(layers[0].channels, ["A", "B", "G", "R"]);
        assert_eq!(layers[2].suffixes(), ["Z"]);
        // A nested pass name groups by its whole path, not its last word.
        let nested = group_channels(
            ["left.diffuse.R", "left.diffuse.G"].map(|s| (s.to_string(), 0)),
        );
        assert_eq!(nested[0].label(), "left.diffuse");
    }

    /// A file with no R/G/B anywhere (a Redshift cryptomatte-only render)
    /// used to fail every frame with "no layer matched"; with no layer
    /// picked it shows its first layer instead.
    #[test]
    fn a_file_without_rgb_shows_its_first_layer() {
        use exr::prelude::*;

        let dir = std::env::temp_dir().join(format!("wobble-norgb-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("crypto.exr");
        let (w, h) = (3usize, 2usize);
        let chan = |name: &str, v: f32| {
            AnyChannel::new(name, FlatSamples::F32(vec![v; w * h]))
        };
        let layer = Layer::new(
            (w, h),
            LayerAttributes::named("rgba"),
            Encoding::FAST_LOSSLESS,
            AnyChannels::sort(
                vec![
                    chan("CryptoObject00.R", 0.1),
                    chan("CryptoObject00.G", 0.2),
                    chan("CryptoObject00.B", 0.3),
                    chan("CryptoObject00.A", 0.4),
                    chan("CryptoObject01.R", 0.5),
                ]
                .into_iter()
                .collect(),
            ),
        );
        Image::from_layer(layer).write().to_file(&path).unwrap();

        let dec = ExrsDecoder::default();
        let info = dec.read_header(&path).unwrap();
        let stride = info.row_stride_f16();
        let mut buf = vec![f16::ZERO; stride * h];
        dec.decode_layer_into(&path, info, stride, None, &mut buf).unwrap();
        let px: Vec<f32> = buf[stride + 4..stride + 8].iter().map(|v| v.to_f32()).collect();
        let want = [0.1f32, 0.2, 0.3, 0.4].map(|v| f16::from_f32(v).to_f32());
        assert_eq!(px, want, "CryptoObject00 in rgba");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn layers_map_onto_rgba_sensibly() {
        let of = |name: &str, chans: &[&str]| LayerDesc {
            name: name.into(),
            channels: chans.iter().map(|c| c.to_string()).collect(),
            part: 0,
        };
        // A colour pass maps straight through.
        let (c, d) = rgba_channels_of(&of("diffuse", &[
            "diffuse.R", "diffuse.G", "diffuse.B",
        ]));
        assert_eq!(c[0].as_deref(), Some("diffuse.R"));
        assert_eq!(c[2].as_deref(), Some("diffuse.B"));
        // No alpha in the file: OPAQUE, so the pass is not invisible.
        assert_eq!(c[3], None);
        assert_eq!(d[3], f16::ONE);

        // A vector pass lands XYZ in RGB.
        let (c, _) = rgba_channels_of(&of("N", &["N.X", "N.Y", "N.Z"]));
        assert_eq!(c[0].as_deref(), Some("N.X"));
        assert_eq!(c[1].as_deref(), Some("N.Y"));
        assert_eq!(c[2].as_deref(), Some("N.Z"));

        // A lone channel reads as GREY, not as red-only.
        let (c, d) = rgba_channels_of(&of("depth", &["depth.Z"]));
        assert_eq!(c[0].as_deref(), Some("depth.Z"));
        assert_eq!(c[1], c[0]);
        assert_eq!(c[2], c[0]);
        assert_eq!(d[3], f16::ONE);

        // The base layer names its channels without a prefix.
        let (c, _) = rgba_channels_of(&of("", &["A", "B", "G", "R"]));
        assert_eq!(c[0].as_deref(), Some("R"));
        assert_eq!(c[3].as_deref(), Some("A"));
    }

    /// Write a real multi-layer EXR, then read each layer back through the
    /// decoder the player uses.
    #[test]
    fn multi_layer_exr_round_trips_per_layer() {
        use exr::prelude::*;

        let dir = std::env::temp_dir().join("wobble-layer-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("passes.exr");
        let (w, h) = (4usize, 2usize);
        // Each channel is a flat constant, so a mis-wired read is obvious.
        // f16 samples and a vector pass, matching what `gen_exr passes`
        // writes — the base layer has to survive the company.
        let chan = |name: &str, v: f32| {
            AnyChannel::new(name, FlatSamples::F16(vec![f16::from_f32(v); w * h]))
        };
        let layer = Layer::new(
            (w, h),
            LayerAttributes::named("rgba"),
            Encoding::FAST_LOSSLESS,
            AnyChannels::sort(
                vec![
                    chan("R", 0.1),
                    chan("G", 0.2),
                    chan("B", 0.3),
                    chan("A", 0.4),
                    chan("diffuse.R", 0.5),
                    chan("diffuse.G", 0.6),
                    chan("diffuse.B", 0.7),
                    chan("depth.Z", 0.9),
                    chan("N.X", -0.5),
                    chan("N.Y", 0.25),
                    chan("N.Z", 1.0),
                ]
                .into_iter()
                .collect(),
            ),
        );
        Image::from_layer(layer).write().to_file(&path).unwrap();

        let dec = ExrsDecoder::default();
        let info = dec.read_header(&path).unwrap();
        assert_eq!((info.width, info.height), (4, 2));

        let names: Vec<String> =
            dec.layers(&path).unwrap().iter().map(|l| l.label().to_string()).collect();
        assert_eq!(names, ["rgba", "N", "depth", "diffuse"]);

        let stride = info.row_stride_f16();
        let mut buf = vec![f16::ZERO; stride * h];
        let px = |buf: &[f16], x: usize, y: usize| {
            let i = y * stride + x * 4;
            [
                buf[i].to_f32(),
                buf[i + 1].to_f32(),
                buf[i + 2].to_f32(),
                buf[i + 3].to_f32(),
            ]
        };
        let close = |a: [f32; 4], b: [f32; 4]| {
            a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-3)
        };

        // No selection = the base RGBA, exactly as before layers existed.
        dec.decode_layer_into(&path, info, stride, None, &mut buf).unwrap();
        assert!(close(px(&buf, 1, 1), [0.1, 0.2, 0.3, 0.4]), "{:?}", px(&buf, 1, 1));

        // A named colour pass, with no alpha of its own -> opaque.
        dec.decode_layer_into(&path, info, stride, Some("diffuse"), &mut buf).unwrap();
        assert!(close(px(&buf, 2, 0), [0.5, 0.6, 0.7, 1.0]), "{:?}", px(&buf, 2, 0));

        // A single-channel pass spreads over rgb rather than tinting red.
        dec.decode_layer_into(&path, info, stride, Some("depth"), &mut buf).unwrap();
        assert!(close(px(&buf, 0, 1), [0.9, 0.9, 0.9, 1.0]), "{:?}", px(&buf, 0, 1));

        // An unknown layer is an ERROR, never a silently black frame.
        let err = dec
            .decode_layer_into(&path, info, stride, Some("nope"), &mut buf)
            .unwrap_err()
            .to_string();
        assert!(err.contains("no layer 'nope'"), "{err}");

        let _ = std::fs::remove_file(&path);
    }
}
