//! DPX from memory: 8, 10 (filled, method A or B) and 16-bit RGB / RGBA,
//! either byte order, to RGBA16 code values (display-referred; what the
//! transfer curve means is the colour pipeline's business). Rows convert
//! in parallel: a 4K 10-bit frame is ~35 MB of words to shift.

use crate::frame::{Frame, Layout};
use anyhow::{bail, ensure, Result};
use rayon::prelude::*;

pub fn decode(b: &[u8]) -> Result<Frame> {
    ensure!(b.len() >= 2048, "not a DPX file (too short)");
    let be = match &b[0..4] {
        b"SDPX" => true,
        b"XPDS" => false,
        _ => bail!("not a DPX file"),
    };
    let u32_at = |o: usize| -> u32 {
        let w = [b[o], b[o + 1], b[o + 2], b[o + 3]];
        if be { u32::from_be_bytes(w) } else { u32::from_le_bytes(w) }
    };
    let u16_at = |o: usize| -> u16 {
        let w = [b[o], b[o + 1]];
        if be { u16::from_be_bytes(w) } else { u16::from_le_bytes(w) }
    };
    let (w, h) = (u32_at(772) as usize, u32_at(776) as usize);
    let desc = b[800];
    let bits = b[803];
    let packing = u16_at(804);
    let encoding = u16_at(806);
    let mut off = u32_at(808) as usize;
    if off == 0 || off == u32::MAX as usize {
        off = u32_at(4) as usize;
    }
    ensure!(encoding == 0, "RLE-encoded DPX is not read");
    let comps = match desc {
        50 => 3,
        51 => 4,
        6 => 1,
        _ => bail!("DPX descriptor {desc} is not read (RGB, RGBA or luma only)"),
    };
    ensure!(w > 0 && h > 0, "DPX with no size");
    let mut px = vec![0u8; w * h * 8];
    let row_out = w * 8;
    match bits {
        10 => {
            ensure!(packing == 1 || packing == 2 || comps == 3, "packed 10-bit DPX of {comps} channels is not read");
            // One 32-bit word per pixel (3 channels) or per 3 samples.
            let words_per_row = (w * comps).div_ceil(3);
            let row_in = words_per_row * 4;
            ensure!(b.len() >= off + row_in * h, "DPX shorter than its picture");
            let data = &b[off..off + row_in * h];
            px.par_chunks_mut(row_out).zip(data.par_chunks(row_in)).for_each(|(dst, src)| {
                let mut samples = Vec::with_capacity(words_per_row * 3);
                for k in 0..words_per_row {
                    let wd = [src[k * 4], src[k * 4 + 1], src[k * 4 + 2], src[k * 4 + 3]];
                    let v = if be { u32::from_be_bytes(wd) } else { u32::from_le_bytes(wd) };
                    let v = if packing == 2 { v << 2 } else { v };
                    samples.extend_from_slice(&[(v >> 22) & 0x3ff, (v >> 12) & 0x3ff, (v >> 2) & 0x3ff]);
                }
                put_row(dst, &samples, comps, |s| (s << 6 | s >> 4) as u16);
            });
        }
        16 => {
            let row_in = w * comps * 2;
            ensure!(b.len() >= off + row_in * h, "DPX shorter than its picture");
            let data = &b[off..off + row_in * h];
            px.par_chunks_mut(row_out).zip(data.par_chunks(row_in)).for_each(|(dst, src)| {
                let samples: Vec<u32> = src.chunks_exact(2).map(|c| if be { u16::from_be_bytes([c[0], c[1]]) } else { u16::from_le_bytes([c[0], c[1]]) } as u32).collect();
                put_row(dst, &samples, comps, |s| s as u16);
            });
        }
        8 => {
            let row_in = (w * comps).div_ceil(4) * 4;
            ensure!(b.len() >= off + row_in * h, "DPX shorter than its picture");
            let data = &b[off..off + row_in * h];
            px.par_chunks_mut(row_out).zip(data.par_chunks(row_in)).for_each(|(dst, src)| {
                let samples: Vec<u32> = src.iter().map(|&v| v as u32).collect();
                put_row(dst, &samples, comps, |s| (s << 8 | s) as u16);
            });
        }
        _ => bail!("{bits}-bit DPX is not read"),
    }
    Ok(Frame::owned(w as u32, h as u32, Layout::Rgba16, vec![px]))
}

fn put_row(dst: &mut [u8], s: &[u32], comps: usize, to16: impl Fn(u32) -> u16) {
    let w = dst.len() / 8;
    for x in 0..w {
        let rgba = match comps {
            1 => {
                let v = to16(s[x]);
                [v, v, v, 0xffff]
            }
            3 => [to16(s[x * 3]), to16(s[x * 3 + 1]), to16(s[x * 3 + 2]), 0xffff],
            _ => [to16(s[x * 4]), to16(s[x * 4 + 1]), to16(s[x * 4 + 2]), to16(s[x * 4 + 3])],
        };
        for (c, v) in rgba.iter().enumerate() {
            dst[x * 8 + c * 2..x * 8 + c * 2 + 2].copy_from_slice(&v.to_le_bytes());
        }
    }
}

/// A 10-bit RGB DPX (method A, big-endian), for tests and benches.
pub fn encode_10bit(w: usize, h: usize, rgb: impl Fn(usize, usize) -> [u16; 3]) -> Vec<u8> {
    let off = 2048;
    let mut b = vec![0u8; off + w * h * 4];
    b[0..4].copy_from_slice(b"SDPX");
    b[4..8].copy_from_slice(&(off as u32).to_be_bytes());
    let total = b.len() as u32;
    b[16..20].copy_from_slice(&total.to_be_bytes());
    b[768..770].copy_from_slice(&0u16.to_be_bytes());
    b[770..772].copy_from_slice(&1u16.to_be_bytes());
    b[772..776].copy_from_slice(&(w as u32).to_be_bytes());
    b[776..780].copy_from_slice(&(h as u32).to_be_bytes());
    b[800] = 50;
    b[803] = 10;
    b[804..806].copy_from_slice(&1u16.to_be_bytes());
    b[808..812].copy_from_slice(&(off as u32).to_be_bytes());
    for y in 0..h {
        for x in 0..w {
            let [r, g, bl] = rgb(x, y);
            let v = ((r as u32 & 0x3ff) << 22) | ((g as u32 & 0x3ff) << 12) | ((bl as u32 & 0x3ff) << 2);
            let o = off + (y * w + x) * 4;
            b[o..o + 4].copy_from_slice(&v.to_be_bytes());
        }
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ten_bit_dpx_round_trips_and_ffmpegs_file_reads() {
        let b = encode_10bit(5, 3, |x, y| [(x * 200) as u16, (y * 300) as u16, 1023]);
        let f = decode(&b).unwrap();
        assert_eq!((f.width, f.height, f.layout), (5, 3, Layout::Rgba16));
        let p = f.plane(0);
        let at = |x: usize, y: usize, c: usize| u16::from_le_bytes([p.data[y * p.stride + x * 8 + c * 2], p.data[y * p.stride + x * 8 + c * 2 + 1]]) >> 6;
        assert_eq!((at(2, 1, 0), at(2, 1, 1), at(2, 1, 2)), (400, 300, 1023));
        // The test media's DPX (ffmpeg's writer) reads too.
        let t = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata/dpx/shot.1001.dpx");
        if let Ok(bytes) = std::fs::read(&t) {
            let f = decode(&bytes).unwrap();
            assert_eq!((f.width, f.height), (1920, 1080));
        }
    }
}
