//! DPX reader (SMPTE 268M), pure Rust.
//!
//! Covers what actually comes out of film scanners and DI pipelines:
//! 10-bit RGB packed into 32-bit words (the "filled, method A" layout), plus
//! 8/12/16-bit RGB and RGBA, in either byte order. Anything else is reported
//! rather than guessed at.
//!
//! The decoder hands back CODE VALUES normalised to 0..1 and nothing else —
//! no curve. What those codes mean (Cineon log, Rec.709, ...) is the
//! Read's input transform, the one place the user can see and change it
//! (`color::InputSpace` / `color::FormatDefaults`), the same arrangement as
//! `still.rs` and `movie.rs`. The header's transfer byte is parsed but only
//! reported: linearising here as well is how a plate gets decoded twice.

use crate::decode::{ExrDecoder, ImageInfo};
use anyhow::{bail, ensure, Context, Result};
use half::f16;
use std::io::Read;
use std::path::Path;

const MAGIC_BE: u32 = 0x5344_5058; // "SDPX"
const MAGIC_LE: u32 = 0x5850_4453; // "XPDS"
/// Generic file header (768) + generic image header (640). Image data can
/// never start before the element descriptors we read.
const HEADER_MIN: usize = 852;
/// What a conformant writer emits: file + image + orientation + film + tv.
const HEADERS_TOTAL: u32 = 2048;

/// Undefined-value markers in DPX headers.
const UNDEF_U32: u32 = 0xFFFF_FFFF;
const UNDEF_U16: u16 = 0xFFFF;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Transfer {
    /// Printing density or logarithmic: Cineon curve.
    Log,
    /// Linear code values.
    Linear,
}

#[derive(Clone, Copy, Debug)]
pub struct DpxInfo {
    pub width: u32,
    pub height: u32,
    pub data_offset: u32,
    pub bits: u8,
    /// 50 = RGB, 51 = RGBA, 6 = luma.
    pub descriptor: u8,
    /// 1 = filled to 32-bit words (method A), 0 = tightly packed.
    pub packing: u16,
    pub big_endian: bool,
    pub transfer: Transfer,
    /// Channels per pixel implied by the descriptor.
    pub channels: usize,
}

fn rd_u32(b: &[u8], at: usize, be: bool) -> u32 {
    let v = [b[at], b[at + 1], b[at + 2], b[at + 3]];
    if be {
        u32::from_be_bytes(v)
    } else {
        u32::from_le_bytes(v)
    }
}

fn rd_u16(b: &[u8], at: usize, be: bool) -> u16 {
    let v = [b[at], b[at + 1]];
    if be {
        u16::from_be_bytes(v)
    } else {
        u16::from_le_bytes(v)
    }
}

/// Parse the generic + image headers. `head` must hold at least 768 bytes.
pub fn parse_header(head: &[u8]) -> Result<DpxInfo> {
    ensure!(head.len() >= HEADER_MIN, "short DPX header");
    let magic = u32::from_be_bytes([head[0], head[1], head[2], head[3]]);
    let big_endian = match magic {
        MAGIC_BE => true,
        MAGIC_LE => false,
        _ => bail!("not a DPX file (magic {magic:#010x})"),
    };
    let be = big_endian;
    let data_offset = rd_u32(head, 4, be);
    ensure!(
        data_offset as usize >= HEADER_MIN && data_offset != UNDEF_U32,
        "bad image data offset {data_offset}"
    );

    // Image element 0 lives at 768 + 0x14 within the image header at 768.
    // Layout: orientation u16 @768, elements u16 @770, pixels/line u32 @772,
    // lines u32 @776, then 72-byte elements starting at 780.
    let width = rd_u32(head, 772, be);
    let height = rd_u32(head, 776, be);
    ensure!(
        width > 0 && height > 0 && width != UNDEF_U32 && height != UNDEF_U32,
        "bad DPX dimensions {width}x{height}"
    );

    // Image element 0 (72 bytes) starts at 780. Fields, per SMPTE 268M:
    // +20 descriptor, +21 transfer, +22 colorimetry, +23 bit size,
    // +24 packing (u16), +26 encoding (u16), +28 offset to this element.
    let el = 780usize;
    let descriptor = head[el + 20];
    let transfer_byte = head[el + 21];
    let bits = head[el + 23];
    let packing = rd_u16(head, el + 24, be);
    let encoding = rd_u16(head, el + 26, be);
    let el_offset = rd_u32(head, el + 28, be);
    ensure!(
        encoding == 0 || encoding == UNDEF_U16,
        "run-length encoded DPX not supported"
    );

    let channels = match descriptor {
        6 | 50 | 51 => match descriptor {
            6 => 1,
            50 => 3,
            _ => 4,
        },
        d => bail!("unsupported DPX descriptor {d} (only luma/RGB/RGBA)"),
    };
    ensure!(
        matches!(bits, 8 | 10 | 12 | 16),
        "unsupported DPX bit depth {bits}"
    );

    // Printing density (1) and logarithmic (3) are the film-log cases.
    let transfer = match transfer_byte {
        1 | 3 => Transfer::Log,
        _ => Transfer::Linear,
    };

    // An element may carry its own data offset; prefer it when set.
    let data_offset = if el_offset as usize >= HEADER_MIN && el_offset != UNDEF_U32 {
        el_offset
    } else {
        data_offset
    };

    Ok(DpxInfo {
        width,
        height,
        data_offset,
        bits,
        descriptor,
        packing: if packing == UNDEF_U16 { 1 } else { packing },
        big_endian,
        transfer,
        channels,
    })
}

/// Decode one DPX into RGBA f16 with `row_stride` f16 elements per row.
pub fn decode(info: &DpxInfo, body: &[u8], row_stride: usize, dst: &mut [f16]) -> Result<()> {
    let (w, h) = (info.width as usize, info.height as usize);
    ensure!(row_stride >= w * 4, "row stride smaller than row");
    ensure!(dst.len() >= row_stride * h, "decode dst too small");

    let max = ((1u32 << info.bits) - 1) as f32;
    // Code value -> 0..1. No curve: see the module docs.
    let to_linear = |code: u32| -> f32 { code as f32 / max };

    let be = info.big_endian;
    let ch = info.channels;
    match info.bits {
        10 => {
            // Method A: three 10-bit samples left-aligned in each 32-bit
            // word, so a row is ceil(w*ch/3) words.
            ensure!(info.packing != 0, "tightly packed 10-bit DPX not supported");
            let words_per_row = (w * ch).div_ceil(3);
            ensure!(
                body.len() >= words_per_row * 4 * h,
                "DPX body too short: {} bytes for {}x{}",
                body.len(),
                w,
                h
            );
            for y in 0..h {
                let row = &body[y * words_per_row * 4..];
                let out = &mut dst[y * row_stride..];
                for x in 0..w {
                    let mut px = [0.0f32; 4];
                    for c in 0..ch {
                        let s = x * ch + c;
                        let word = rd_u32(row, (s / 3) * 4, be);
                        // Samples sit at bits 22, 12, 2 (2 padding bits low).
                        let shift = 22 - (s % 3) * 10;
                        px[c] = to_linear((word >> shift) & 0x3FF);
                    }
                    write_px(&mut out[x * 4..x * 4 + 4], px, ch);
                }
            }
        }
        8 => {
            let stride = w * ch;
            ensure!(body.len() >= stride * h, "DPX body too short");
            for y in 0..h {
                let row = &body[y * stride..];
                let out = &mut dst[y * row_stride..];
                for x in 0..w {
                    let mut px = [0.0f32; 4];
                    for c in 0..ch {
                        px[c] = to_linear(row[x * ch + c] as u32);
                    }
                    write_px(&mut out[x * 4..x * 4 + 4], px, ch);
                }
            }
        }
        12 | 16 => {
            // 12-bit "filled" and 16-bit both occupy one 16-bit word per
            // sample; 12-bit method A left-aligns within the word.
            let stride = w * ch * 2;
            ensure!(body.len() >= stride * h, "DPX body too short");
            let shift = if info.bits == 12 && info.packing != 0 { 4 } else { 0 };
            for y in 0..h {
                let row = &body[y * stride..];
                let out = &mut dst[y * row_stride..];
                for x in 0..w {
                    let mut px = [0.0f32; 4];
                    for c in 0..ch {
                        let v = rd_u16(row, (x * ch + c) * 2, be) >> shift;
                        px[c] = to_linear(v as u32);
                    }
                    write_px(&mut out[x * 4..x * 4 + 4], px, ch);
                }
            }
        }
        b => bail!("unsupported DPX bit depth {b}"),
    }
    Ok(())
}

/// Splat a decoded pixel into RGBA (luma replicates; RGB gets alpha 1).
fn write_px(out: &mut [f16], px: [f32; 4], channels: usize) {
    let (r, g, b, a) = match channels {
        1 => (px[0], px[0], px[0], 1.0),
        3 => (px[0], px[1], px[2], 1.0),
        _ => (px[0], px[1], px[2], px[3]),
    };
    out[0] = f16::from_f32(r);
    out[1] = f16::from_f32(g);
    out[2] = f16::from_f32(b);
    out[3] = f16::from_f32(a);
}

/// Write a minimal 10-bit RGB DPX. Used by the self-tests to make fixtures
/// without shipping binary sample media.
pub fn write_test_dpx(
    path: &Path,
    w: u32,
    h: u32,
    log: bool,
    px: impl Fn(u32, u32) -> [u16; 3],
) -> Result<()> {
    let mut b = vec![0u8; HEADERS_TOTAL as usize];
    b[0..4].copy_from_slice(&MAGIC_BE.to_be_bytes());
    b[4..8].copy_from_slice(&HEADERS_TOTAL.to_be_bytes());
    b[772..776].copy_from_slice(&w.to_be_bytes());
    b[776..780].copy_from_slice(&h.to_be_bytes());
    let el = 780;
    b[el + 28..el + 32].copy_from_slice(&HEADERS_TOTAL.to_be_bytes());
    b[el + 20] = 50; // RGB
    b[el + 21] = if log { 1 } else { 2 };
    b[el + 23] = 10;
    b[el + 24..el + 26].copy_from_slice(&1u16.to_be_bytes());
    for y in 0..h {
        let samples: Vec<u32> = (0..w)
            .flat_map(|x| px(x, y).into_iter().map(u32::from).collect::<Vec<_>>())
            .collect();
        for chunk in samples.chunks(3) {
            let mut word = 0u32;
            for (k, s) in chunk.iter().enumerate() {
                word |= (s & 0x3FF) << (22 - k * 10);
            }
            b.extend_from_slice(&word.to_be_bytes());
        }
    }
    std::fs::write(path, b).with_context(|| format!("writing {}", path.display()))
}

pub struct DpxDecoder;

impl DpxDecoder {
    fn read_info(path: &Path) -> Result<DpxInfo> {
        let mut f = crate::vfs::open(path)?;
        let mut head = vec![0u8; HEADERS_TOTAL as usize];
        let n = f.read(&mut head)?;
        head.truncate(n);
        parse_header(&head).with_context(|| format!("parsing {}", path.display()))
    }
}

impl ExrDecoder for DpxDecoder {
    fn read_header(&self, path: &Path) -> Result<ImageInfo> {
        let i = Self::read_info(path)?;
        Ok(ImageInfo { width: i.width, height: i.height })
    }

    fn decode_into(
        &self,
        path: &Path,
        _info: ImageInfo,
        row_stride: usize,
        dst: &mut [f16],
    ) -> Result<()> {
        let bytes = crate::vfs::read(path)?;
        let i = parse_header(&bytes)
            .with_context(|| format!("parsing {}", path.display()))?;
        let start = i.data_offset as usize;
        ensure!(bytes.len() > start, "DPX has no image data");
        decode(&i, &bytes[start..], row_stride, dst)
            .with_context(|| format!("decoding {}", path.display()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal 10-bit RGB DPX in memory.
    fn synth(w: u32, h: u32, be: bool, transfer: u8, codes: &[[u16; 3]]) -> Vec<u8> {
        let mut b = vec![0u8; HEADERS_TOTAL as usize];
        let magic: u32 = if be { MAGIC_BE } else { MAGIC_LE };
        b[0..4].copy_from_slice(&magic.to_be_bytes());
        let put32 = |b: &mut Vec<u8>, at: usize, v: u32| {
            let bytes = if be { v.to_be_bytes() } else { v.to_le_bytes() };
            b[at..at + 4].copy_from_slice(&bytes);
        };
        let put16 = |b: &mut Vec<u8>, at: usize, v: u16| {
            let bytes = if be { v.to_be_bytes() } else { v.to_le_bytes() };
            b[at..at + 2].copy_from_slice(&bytes);
        };
        put32(&mut b, 4, HEADERS_TOTAL);
        put32(&mut b, 772, w);
        put32(&mut b, 776, h);
        let el = 780;
        put32(&mut b, el + 28, HEADERS_TOTAL);
        b[el + 20] = 50; // RGB
        b[el + 21] = transfer;
        b[el + 23] = 10; // bits
        put16(&mut b, el + 24, 1); // filled, method A
        // Body: three 10-bit samples per 32-bit word.
        let samples: Vec<u32> =
            codes.iter().flat_map(|p| p.iter().map(|v| *v as u32)).collect();
        let words = samples.len().div_ceil(3);
        for wi in 0..words {
            let mut word = 0u32;
            for k in 0..3 {
                let s = samples.get(wi * 3 + k).copied().unwrap_or(0);
                word |= (s & 0x3FF) << (22 - k * 10);
            }
            let bytes = if be { word.to_be_bytes() } else { word.to_le_bytes() };
            b.extend_from_slice(&bytes);
        }
        b
    }

    fn decode_one(file: &[u8], w: usize, h: usize) -> Vec<f16> {
        let info = parse_header(file).unwrap();
        let stride = w * 4;
        let mut dst = vec![f16::ZERO; stride * h];
        decode(&info, &file[info.data_offset as usize..], stride, &mut dst).unwrap();
        dst
    }

    #[test]
    fn parses_both_byte_orders() {
        for be in [true, false] {
            let f = synth(2, 1, be, 2, &[[1023, 0, 512], [0, 1023, 0]]);
            let i = parse_header(&f).unwrap();
            assert_eq!((i.width, i.height, i.bits, i.channels), (2, 1, 10, 3));
            assert_eq!(i.big_endian, be);
            assert_eq!(i.transfer, Transfer::Linear);
        }
    }

    #[test]
    fn linear_codes_map_to_unit_range() {
        let f = synth(2, 1, true, 2, &[[1023, 0, 512], [0, 1023, 0]]);
        let px = decode_one(&f, 2, 1);
        assert!((px[0].to_f32() - 1.0).abs() < 1e-3, "{}", px[0]);
        assert_eq!(px[1].to_f32(), 0.0);
        assert!((px[2].to_f32() - 512.0 / 1023.0).abs() < 1e-3);
        assert_eq!(px[3].to_f32(), 1.0); // alpha filled for RGB
        assert!((px[5].to_f32() - 1.0).abs() < 1e-3);
    }

    /// A log-flagged file comes back as CODES, not linear: the Read's
    /// Cineon input transform linearises it, on exactly the curve this
    /// decoder used to apply itself.
    #[test]
    fn log_files_decode_to_codes_not_linear() {
        let f = synth(3, 1, true, 1, &[[95, 95, 95], [685, 685, 685], [1023, 1023, 1023]]);
        assert_eq!(parse_header(&f).unwrap().transfer, Transfer::Log);
        let px = decode_one(&f, 3, 1);
        for (i, code) in [(0, 95.0f32), (4, 685.0), (8, 1023.0)] {
            let v = px[i].to_f32();
            assert!((v - code / 1023.0).abs() < 1e-3, "code {code}: {v}");
            // ...and the transform the Read applies lands where the old
            // in-decoder curve did: black at 95, the white point at 685.
            let lin = crate::color::cineon_to_linear(v * 1023.0);
            let want = if code == 95.0 { 0.0 } else if code == 685.0 { 1.0 } else { lin };
            assert!((lin - want).abs() < 0.02, "code {code} -> {lin}");
        }
    }

    #[test]
    fn row_padding_is_respected() {
        // Two rows with a padded stride: row 1 must land at the stride, not
        // packed after row 0.
        let f = synth(1, 2, true, 2, &[[1023, 1023, 1023], [0, 0, 0]]);
        let info = parse_header(&f).unwrap();
        let stride = 16; // 4 pixels' worth for a 1-pixel row
        let mut dst = vec![f16::from_f32(-1.0); stride * 2];
        decode(&info, &f[info.data_offset as usize..], stride, &mut dst).unwrap();
        assert!((dst[0].to_f32() - 1.0).abs() < 1e-3);
        assert_eq!(dst[stride].to_f32(), 0.0);
    }

    #[test]
    fn rejects_non_dpx() {
        assert!(parse_header(&vec![0u8; 2048]).is_err());
    }

    #[test]
    fn rejects_unsupported_descriptor() {
        let mut f = synth(1, 1, true, 2, &[[0, 0, 0]]);
        f[780 + 20] = 100; // undefined descriptor
        assert!(parse_header(&f).is_err());
    }
}
