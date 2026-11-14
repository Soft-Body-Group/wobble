//! A decoded picture, kept in the layout its decoder produced.
//!
//! **No CPU colour conversion.** A 4K ProRes 422 frame converted to RGBA
//! half floats is 66 MB and ~10 ms of CPU; left as 10-bit planar YCbCr it
//! is 33 MB and nothing. The GPU does YCbCr -> RGB in one pass (the app's
//! viewer), so frames stay in their native planes from the decoder to the
//! texture upload. FFmpeg frames are not even copied: the plane memory is
//! the decoder's own refcounted buffer.

use std::sync::Arc;

/// Which YCbCr -> RGB matrix a planar frame needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Matrix {
    Bt601,
    Bt709,
    Bt2020,
}

impl Matrix {
    /// (Kr, Kb): the luma weights of red and blue.
    pub fn kr_kb(self) -> (f32, f32) {
        match self {
            Matrix::Bt601 => (0.299, 0.114),
            Matrix::Bt709 => (0.2126, 0.0722),
            Matrix::Bt2020 => (0.2627, 0.0593),
        }
    }
}

/// How the samples of a frame are laid out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layout {
    /// Planar Y, Cb, Cr (and A when `alpha`). Samples are `bits` deep, in
    /// u8 when `bits == 8`, else in little-endian u16 holding the low
    /// `bits` bits. Chroma planes are subsampled by `1 << cx` across and
    /// `1 << cy` down. `semi`: Cb and Cr interleaved in ONE plane (NV12,
    /// P010: what VideoToolbox hands back). `shift`: samples sit in the
    /// HIGH bits of their u16 (P010's 10 bits are `v >> 6`).
    Yuv { bits: u8, cx: u8, cy: u8, alpha: bool, matrix: Matrix, full_range: bool, semi: bool, shift: u8 },
    /// Interleaved RGBA, u8 code values (display-referred).
    Rgba8,
    /// Interleaved RGBA, u16 code values (display-referred).
    Rgba16,
    /// Interleaved RGBA half floats, scene-linear (EXR).
    RgbaF16,
}

impl Layout {
    /// Bytes per sample of each plane.
    pub fn sample_bytes(self) -> usize {
        match self {
            Layout::Yuv { bits, .. } => if bits > 8 { 2 } else { 1 },
            Layout::Rgba8 => 4,
            Layout::Rgba16 | Layout::RgbaF16 => 8,
        }
    }

    pub fn plane_count(self) -> usize {
        match self {
            Layout::Yuv { alpha, semi, .. } => 3 - semi as usize + alpha as usize,
            _ => 1,
        }
    }

    /// Bytes per texel of plane `i` (a semi-planar chroma texel is a pair).
    pub fn texel_bytes(self, i: usize) -> usize {
        match self {
            Layout::Yuv { semi: true, .. } if i == 1 => 2 * self.sample_bytes(),
            _ => self.sample_bytes(),
        }
    }

    /// Plane `i` is chroma.
    fn is_chroma(self, i: usize) -> bool {
        match self {
            Layout::Yuv { semi: true, .. } => i == 1,
            Layout::Yuv { .. } => i == 1 || i == 2,
            _ => false,
        }
    }

    /// Size in samples of plane `i` for a `w`x`h` frame.
    pub fn plane_size(self, i: usize, w: u32, h: u32) -> (u32, u32) {
        match self {
            Layout::Yuv { cx, cy, .. } if self.is_chroma(i) => {
                (w.div_ceil(1 << cx), h.div_ceil(1 << cy))
            }
            _ => (w, h),
        }
    }
}

/// Where a frame's bytes live.
pub enum Backing {
    /// Our own allocation, one Vec per plane.
    Owned(Vec<Vec<u8>>),
    /// The decoder's refcounted AVFrame: zero copy.
    Av(ffmpeg_next::frame::Video),
}

/// One plane: its bytes, the stride between rows, its size in samples.
#[derive(Clone, Copy)]
pub struct Plane<'a> {
    pub data: &'a [u8],
    pub stride: usize,
    pub width: u32,
    pub height: u32,
}

pub struct Frame {
    pub width: u32,
    pub height: u32,
    pub layout: Layout,
    backing: Backing,
    strides: [usize; 4],
    /// Wall time the decode took (ms), for the HUD and the bench.
    pub decode_ms: f32,
}

pub type FrameRef = Arc<Frame>;

impl Frame {
    /// Owned planes, tightly packed (stride = width * sample bytes).
    pub fn owned(width: u32, height: u32, layout: Layout, planes: Vec<Vec<u8>>) -> Self {
        let mut strides = [0; 4];
        for (i, s) in strides.iter_mut().enumerate().take(layout.plane_count()) {
            *s = layout.plane_size(i, width, height).0 as usize * layout.texel_bytes(i);
        }
        Self { width, height, layout, backing: Backing::Owned(planes), strides, decode_ms: 0.0 }
    }

    /// Wrap an AVFrame whose pixel format the caller has matched to `layout`.
    pub fn av(layout: Layout, frame: ffmpeg_next::frame::Video) -> Self {
        let mut strides = [0; 4];
        for (i, s) in strides.iter_mut().enumerate().take(layout.plane_count()) {
            *s = frame.stride(i);
        }
        Self {
            width: frame.width(),
            height: frame.height(),
            layout,
            backing: Backing::Av(frame),
            strides,
            decode_ms: 0.0,
        }
    }

    /// The decoder's AVFrame, when the frame is one (zero copy).
    pub fn av_frame(&self) -> Option<&ffmpeg_next::frame::Video> {
        match &self.backing {
            Backing::Av(f) => Some(f),
            Backing::Owned(_) => None,
        }
    }

    pub fn plane(&self, i: usize) -> Plane<'_> {
        let (width, height) = self.layout.plane_size(i, self.width, self.height);
        let stride = self.strides[i];
        let data = match &self.backing {
            Backing::Owned(p) => &p[i][..],
            Backing::Av(f) => f.data(i),
        };
        // An AVFrame's last row may stop short of the full stride.
        let need = stride * (height as usize - 1) + width as usize * self.layout.texel_bytes(i);
        Plane { data: &data[..need.min(data.len())], stride, width, height }
    }

    /// Bytes this frame holds: what the cache budgets.
    pub fn bytes(&self) -> usize {
        (0..self.layout.plane_count())
            .map(|i| self.strides[i] * self.layout.plane_size(i, self.width, self.height).1 as usize)
            .sum()
    }
}
