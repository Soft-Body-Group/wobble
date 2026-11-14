//! Tiny pictures for the bin's tiles: a frame boxed down to a few hundred
//! pixels as display RGBA8, on the CPU. A tile frame is ~200x110, so
//! converting only those samples (four per pixel, from the native planes)
//! costs a fraction of a millisecond; the decode is the cost.

use crate::frame::{Frame, Layout};

/// `frame` fitted into `max_w` x `max_h` (aspect kept), as display-referred
/// RGBA8 (EXR's scene-linear goes through the sRGB curve). Returns
/// (width, height, pixels).
pub fn thumbnail(frame: &Frame, max_w: u32, max_h: u32) -> (u32, u32, Vec<u8>) {
    let (fw, fh) = (frame.width.max(1), frame.height.max(1));
    let s = (max_w as f32 / fw as f32).min(max_h as f32 / fh as f32).min(1.0);
    let (w, h) = (((fw as f32 * s).round() as u32).max(1), ((fh as f32 * s).round() as u32).max(1));
    let mut out = vec![0u8; (w * h * 4) as usize];
    let read = Reader::new(frame);
    for y in 0..h {
        for x in 0..w {
            let mut acc = [0f32; 3];
            // Four samples a pixel, spread over the box it covers.
            for (ox, oy) in [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)] {
                let sx = (((x as f32 + ox) / w as f32) * fw as f32) as u32;
                let sy = (((y as f32 + oy) / h as f32) * fh as f32) as u32;
                let c = read.rgb(sx.min(fw - 1), sy.min(fh - 1));
                for k in 0..3 {
                    acc[k] += c[k];
                }
            }
            let o = ((y * w + x) * 4) as usize;
            for k in 0..3 {
                out[o + k] = (acc[k] * 0.25 * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
            }
            out[o + 3] = 255;
        }
    }
    (w, h, out)
}

struct Reader<'a> {
    f: &'a Frame,
}

impl<'a> Reader<'a> {
    fn new(f: &'a Frame) -> Self {
        Reader { f }
    }

    /// Sample `i` of plane `p` at (x, y) in that plane's samples, 0..1.
    fn sample(&self, p: usize, x: u32, y: u32, i: usize, bits: u8, shift: u8) -> f32 {
        let pl = self.f.plane(p);
        let tb = self.f.layout.texel_bytes(p);
        let sb = self.f.layout.sample_bytes();
        let x = x.min(pl.width.saturating_sub(1));
        let y = y.min(pl.height.saturating_sub(1));
        let o = y as usize * pl.stride + x as usize * tb + i * sb;
        let max = ((1u32 << bits) - 1) as f32;
        if sb == 1 {
            pl.data.get(o).copied().unwrap_or(0) as f32 / max
        } else {
            let v = u16::from_le_bytes([pl.data.get(o).copied().unwrap_or(0), pl.data.get(o + 1).copied().unwrap_or(0)]);
            (v >> shift) as f32 / max
        }
    }

    fn rgb(&self, x: u32, y: u32) -> [f32; 3] {
        match self.f.layout {
            Layout::Yuv { bits, cx, cy, matrix, full_range, semi, shift, .. } => {
                let yv = self.sample(0, x, y, 0, bits, shift);
                let (cxp, cyp) = (x >> cx, y >> cy);
                let (cb, cr) = if semi { (self.sample(1, cxp, cyp, 0, bits, shift), self.sample(1, cxp, cyp, 1, bits, shift)) } else { (self.sample(1, cxp, cyp, 0, bits, shift), self.sample(2, cxp, cyp, 0, bits, shift)) };
                let (yy, pb, pr) = if full_range { (yv, cb - 0.5, cr - 0.5) } else { ((yv - 16.0 / 255.0) * 255.0 / 219.0, (cb - 128.0 / 255.0) * 255.0 / 224.0, (cr - 128.0 / 255.0) * 255.0 / 224.0) };
                let (kr, kb) = matrix.kr_kb();
                let r = yy + 2.0 * (1.0 - kr) * pr;
                let b = yy + 2.0 * (1.0 - kb) * pb;
                let g = (yy - kr * r - kb * b) / (1.0 - kr - kb);
                [r, g, b]
            }
            Layout::Rgba8 => {
                let pl = self.f.plane(0);
                let o = y as usize * pl.stride + x as usize * 4;
                [0, 1, 2].map(|k| pl.data.get(o + k).copied().unwrap_or(0) as f32 / 255.0)
            }
            Layout::Rgba16 => {
                let pl = self.f.plane(0);
                let o = y as usize * pl.stride + x as usize * 8;
                [0, 1, 2].map(|k| u16::from_le_bytes([pl.data.get(o + 2 * k).copied().unwrap_or(0), pl.data.get(o + 2 * k + 1).copied().unwrap_or(0)]) as f32 / 65535.0)
            }
            Layout::RgbaF16 => {
                let pl = self.f.plane(0);
                let o = y as usize * pl.stride + x as usize * 8;
                [0, 1, 2].map(|k| {
                    let v = half::f16::from_le_bytes([pl.data.get(o + 2 * k).copied().unwrap_or(0), pl.data.get(o + 2 * k + 1).copied().unwrap_or(0)]).to_f32();
                    srgb(v)
                })
            }
        }
    }
}

fn srgb(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.003_130_8 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::Matrix;

    #[test]
    fn a_limited_range_yuv_frame_boxes_down_to_its_colours() {
        // 8x4 4:2:0 limited 709: left half white (Y 235), right half black
        // (Y 16), neutral chroma.
        let (w, h) = (8u32, 4u32);
        let ypl: Vec<u8> = (0..h).flat_map(|_| (0..w).map(|x| if x < 4 { 235 } else { 16 })).collect();
        let c = vec![128u8; 4 * 2];
        let f = Frame::owned(w, h, Layout::Yuv { bits: 8, cx: 1, cy: 1, alpha: false, matrix: Matrix::Bt709, full_range: false, semi: false, shift: 0 }, vec![ypl, c.clone(), c]);
        let (tw, th, px) = thumbnail(&f, 4, 4);
        assert_eq!((tw, th), (4, 2));
        assert_eq!(&px[0..4], &[255, 255, 255, 255]);
        assert_eq!(&px[(3 * 4)..(3 * 4 + 3)], &[0, 0, 0]);
    }

    #[test]
    fn linear_exr_goes_through_the_srgb_curve() {
        let v = half::f16::from_f32(0.18).to_le_bytes();
        let px: Vec<u8> = (0..4).flat_map(|_| [v[0], v[1], v[0], v[1], v[0], v[1], 0, 0]).collect();
        let f = Frame::owned(2, 2, Layout::RgbaF16, vec![px]);
        let (_, _, out) = thumbnail(&f, 2, 2);
        assert!((out[0] as i32 - 118).abs() <= 1, "{}", out[0]);
    }
}
