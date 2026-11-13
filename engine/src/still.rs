//! PNG and JPEG readers, pure Rust.
//!
//! These are display-referred delivery formats — an 8- or 16-bit integer
//! picture in some encoding the file does not reliably state. So the
//! decoder's whole job is to hand back the CODE VALUES, normalised to
//! 0..1, and nothing else: no curve, no matrix, no guess. The reader
//! leaf's `InputSpace` applies the transform, which is the one place the
//! user can see and change it (`color::InputSpace::guess_for_extension`
//! prefills sRGB for both of these). Silently linearising here would make
//! a wrong colour space invisible, which is exactly the failure the
//! working-space rules exist to prevent.
//!
//! (`dpx.rs` predates that arrangement and linearises internally; these do
//! not, deliberately.)
//!
//! Both formats are one image per file, so the layer and frame machinery
//! is a no-op: `layers()` reports none and `decode_frame_into` ignores the
//! frame index, both via the trait's defaults.

use crate::decode::{ExrDecoder, ImageInfo};
use anyhow::{bail, Context as _, Result};
use half::f16;
use std::path::Path;

/// Alpha handling. PNG carries straight (unassociated) alpha by
/// specification, and JPEG has none at all, so nothing here is ever
/// premultiplied — matching what a Read hands the rest of the comp.
pub struct StillDecoder;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    Png,
    Jpeg,
}

fn kind_of(path: &Path) -> Result<Kind> {
    let ext = path
        .extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    match ext.as_str() {
        "png" => Ok(Kind::Png),
        "jpg" | "jpeg" => Ok(Kind::Jpeg),
        _ => bail!("{} is not a PNG or JPEG", path.display()),
    }
}

/// One decoded picture as unit-range code values, RGBA-interleaved.
struct Rgba {
    width: usize,
    height: usize,
    px: Vec<f32>,
}

fn decode_png(path: &Path) -> Result<Rgba> {
    use png::{BitDepth, ColorType, Transformations};

    let mut dec = png::Decoder::new(crate::vfs::open(path)?);
    // EXPAND turns palettes into RGB, sub-byte greys into 8-bit and a tRNS
    // chunk into a real alpha channel, so the match below only has to cover
    // the four full-byte layouts at two depths.
    dec.set_transformations(Transformations::EXPAND);
    let mut reader = dec
        .read_info()
        .with_context(|| format!("reading PNG header of {}", path.display()))?;

    let mut buf = vec![0u8; reader.output_buffer_size().unwrap_or(0)];
    let out = reader
        .next_frame(&mut buf)
        .with_context(|| format!("decoding {}", path.display()))?;
    let (w, h) = (out.width as usize, out.height as usize);
    let (ct, depth) = reader.output_color_type();

    let comps = match ct {
        ColorType::Grayscale => 1,
        ColorType::GrayscaleAlpha => 2,
        ColorType::Rgb => 3,
        ColorType::Rgba => 4,
        // EXPAND has already resolved these two.
        ColorType::Indexed => bail!("palette PNG did not expand: {}", path.display()),
    };
    // PNG stores 16-bit samples BIG-endian regardless of host order.
    let sample = |i: usize| -> f32 {
        match depth {
            BitDepth::Sixteen => {
                u16::from_be_bytes([buf[i * 2], buf[i * 2 + 1]]) as f32 / 65535.0
            }
            _ => buf[i] as f32 / 255.0,
        }
    };

    let mut px = vec![0f32; w * h * 4];
    for i in 0..w * h {
        let s = i * comps;
        let (r, g, b, a) = match comps {
            1 => (sample(s), sample(s), sample(s), 1.0),
            2 => (sample(s), sample(s), sample(s), sample(s + 1)),
            3 => (sample(s), sample(s + 1), sample(s + 2), 1.0),
            _ => (sample(s), sample(s + 1), sample(s + 2), sample(s + 3)),
        };
        px[i * 4] = r;
        px[i * 4 + 1] = g;
        px[i * 4 + 2] = b;
        px[i * 4 + 3] = a;
    }
    Ok(Rgba { width: w, height: h, px })
}

fn decode_jpeg(path: &Path) -> Result<Rgba> {
    use zune_jpeg::zune_core::bytestream::ZCursor;
    use zune_jpeg::zune_core::colorspace::ColorSpace;
    use zune_jpeg::JpegDecoder;

    let bytes = crate::vfs::read(path)?;
    let mut dec = JpegDecoder::new(ZCursor::new(&bytes[..]));
    // Ask for RGBA up front: the library's own YCbCr->RGB and greyscale
    // expansion are faster than doing it a pixel at a time out here, and a
    // CMYK/YCCK scan converts on the same path instead of erroring.
    let opts = dec.options().jpeg_set_out_colorspace(ColorSpace::RGBA);
    dec.set_options(opts);
    let out = dec
        .decode()
        .map_err(|e| anyhow::anyhow!("decoding {}: {e}", path.display()))?;
    let info = dec
        .info()
        .with_context(|| format!("{} has no JPEG frame header", path.display()))?;
    let (w, h) = (info.width as usize, info.height as usize);
    anyhow::ensure!(
        out.len() >= w * h * 4,
        "{}: JPEG decoded {} bytes for a {w}x{h} RGBA image",
        path.display(),
        out.len()
    );
    let px = out[..w * h * 4].iter().map(|&v| v as f32 / 255.0).collect();
    Ok(Rgba { width: w, height: h, px })
}

fn header(path: &Path) -> Result<ImageInfo> {
    match kind_of(path)? {
        Kind::Png => {
            let mut dec = png::Decoder::new(crate::vfs::open(path)?);
            let info = dec
                .read_header_info()
                .with_context(|| format!("reading PNG header of {}", path.display()))?;
            Ok(ImageInfo { width: info.width, height: info.height })
        }
        Kind::Jpeg => {
            use zune_jpeg::zune_core::bytestream::ZCursor;
            // The frame header is in the first few KB; reading the whole
            // file to find it would make a directory scan quadratic.
            let bytes = read_prefix(path, 64 * 1024)?;
            let mut dec = zune_jpeg::JpegDecoder::new(ZCursor::new(&bytes));
            dec.decode_headers()
                .map_err(|e| anyhow::anyhow!("reading JPEG header of {}: {e}", path.display()))?;
            let info = dec
                .info()
                .with_context(|| format!("{} has no JPEG frame header", path.display()))?;
            Ok(ImageInfo { width: info.width as u32, height: info.height as u32 })
        }
    }
}

/// A still's pixel size from its header, or None for anything that is not
/// a PNG/JPEG (or cannot be read). What the Read panel uses to count the
/// frames `decode_into` will fit.
pub fn size_of(path: &Path) -> Option<(u32, u32)> {
    let i = header(path).ok()?;
    Some((i.width, i.height))
}

/// The placement of a `iw` x `ih` picture fitted whole into `w` x `h`:
/// (x, y, width, height), centred, at least one pixel each way.
fn fit_rect(iw: usize, ih: usize, w: usize, h: usize) -> (usize, usize, usize, usize) {
    let s = (w as f64 / iw as f64).min(h as f64 / ih as f64);
    let fw = ((iw as f64 * s).round() as usize).clamp(1, w);
    let fh = ((ih as f64 * s).round() as usize).clamp(1, h);
    ((w - fw) / 2, (h - fh) / 2, fw, fh)
}

/// Tent-filter weights taking `n` source samples to `m`: per output sample,
/// the first source index and its normalised weights. The tent widens with
/// the reduction, so a downscale averages what it drops instead of
/// aliasing it, and an upscale is plain linear interpolation.
fn weights(n: usize, m: usize) -> Vec<(usize, Vec<f32>)> {
    let scale = m as f64 / n as f64;
    let r = (1.0 / scale).max(1.0);
    (0..m)
        .map(|j| {
            let c = (j as f64 + 0.5) / scale - 0.5;
            let lo = ((c - r).ceil().max(0.0)) as usize;
            let hi = ((c + r).floor() as isize).min(n as isize - 1).max(lo as isize) as usize;
            let mut ws: Vec<f32> =
                (lo..=hi).map(|i| (1.0 - ((i as f64 - c).abs() / r)).max(0.0) as f32).collect();
            let sum: f32 = ws.iter().sum();
            if sum > 0.0 {
                ws.iter_mut().for_each(|v| *v /= sum);
            } else {
                ws = vec![1.0];
            }
            (lo, ws)
        })
        .collect()
}

/// `img` fitted into a `w` x `h` frame (see `fit_rect`), separable tent
/// resample, rows in parallel. Outside the picture every lane is 0.
fn fit(img: &Rgba, w: usize, h: usize) -> Rgba {
    use rayon::prelude::*;
    let (ox, oy, fw, fh) = fit_rect(img.width, img.height, w, h);
    let (wx, wy) = (weights(img.width, fw), weights(img.height, fh));
    let iw = img.width;
    // Horizontal pass: every source row to `fw` wide.
    let mut tmp = vec![0f32; img.height * fw * 4];
    tmp.par_chunks_mut(fw * 4).enumerate().for_each(|(y, row)| {
        let src = &img.px[y * iw * 4..(y + 1) * iw * 4];
        for (x, (lo, ws)) in wx.iter().enumerate() {
            let mut acc = [0f32; 4];
            for (k, wt) in ws.iter().enumerate() {
                let s = &src[(lo + k) * 4..(lo + k) * 4 + 4];
                for c in 0..4 {
                    acc[c] += s[c] * wt;
                }
            }
            row[x * 4..x * 4 + 4].copy_from_slice(&acc);
        }
    });
    // Vertical pass, straight into the frame.
    let mut px = vec![0f32; w * h * 4];
    px.par_chunks_mut(w * 4).enumerate().for_each(|(y, row)| {
        if y < oy || y >= oy + fh {
            return;
        }
        let (lo, ws) = &wy[y - oy];
        let dst = &mut row[ox * 4..(ox + fw) * 4];
        for (k, wt) in ws.iter().enumerate() {
            let src = &tmp[(lo + k) * fw * 4..(lo + k + 1) * fw * 4];
            for (d, s) in dst.iter_mut().zip(src) {
                *d += s * wt;
            }
        }
    });
    Rgba { width: w, height: h, px }
}

/// At most `max` bytes from the front of a file (short files read whole).
fn read_prefix(path: &Path, max: usize) -> Result<Vec<u8>> {
    use std::io::Read as _;
    let mut f = crate::vfs::open(path)?;
    let mut buf = Vec::new();
    f.by_ref()
        .take(max as u64)
        .read_to_end(&mut buf)
        .with_context(|| format!("reading {}", path.display()))?;
    Ok(buf)
}

impl ExrDecoder for StillDecoder {
    fn read_header(&self, path: &Path) -> Result<ImageInfo> {
        header(path)
    }

    fn decode_into(
        &self,
        path: &Path,
        info: ImageInfo,
        row_stride: usize,
        dst: &mut [f16],
    ) -> Result<()> {
        let w = info.width as usize;
        let h = info.height as usize;
        anyhow::ensure!(row_stride >= w * 4, "row stride smaller than row");
        anyhow::ensure!(dst.len() >= row_stride * h, "decode dst too small");

        let img = match kind_of(path)? {
            Kind::Png => decode_png(path)?,
            Kind::Jpeg => decode_jpeg(path)?,
        };
        // A sequence's size comes from its first frame. A folder of photos
        // is rarely one size (portrait and landscape, a second camera), so
        // a frame that disagrees is FITTED into it: scaled to fit whole,
        // aspect kept, centred, the rest transparent black. Never silent:
        // the Read panel counts these frames (`size_of` at load).
        let img = if img.width == w && img.height == h { img } else { fit(&img, w, h) };

        for y in 0..h {
            let src = &img.px[y * w * 4..(y + 1) * w * 4];
            let row = &mut dst[y * row_stride..y * row_stride + w * 4];
            for (d, s) in row.iter_mut().zip(src) {
                *d = f16::from_f32(*s);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join("wobble-still-test");
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    fn read_px(path: &Path, w: usize, h: usize) -> Vec<[f32; 4]> {
        let dec = StillDecoder;
        let info = dec.read_header(path).unwrap();
        assert_eq!((info.width as usize, info.height as usize), (w, h));
        let stride = info.row_stride_f16();
        let mut buf = vec![f16::ZERO; stride * h];
        dec.decode_into(path, info, stride, &mut buf).unwrap();
        (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                let o = y * stride + x * 4;
                [
                    buf[o].to_f32(),
                    buf[o + 1].to_f32(),
                    buf[o + 2].to_f32(),
                    buf[o + 3].to_f32(),
                ]
            })
            .collect()
    }

    fn close(a: [f32; 4], b: [f32; 4]) -> bool {
        a.iter().zip(b).all(|(x, y)| (x - y).abs() < 2e-3)
    }

    /// 8-bit RGBA PNG: code values arrive UNCHANGED — no sRGB decode here,
    /// because that is `InputSpace`'s job and doing it twice is invisible.
    #[test]
    fn png_rgba8_round_trips_as_code_values() {
        let path = tmp("rgba8.png");
        let (w, h) = (2usize, 2usize);
        let data: Vec<u8> = vec![
            255, 0, 0, 255, //
            0, 255, 0, 128, //
            0, 0, 255, 255, //
            128, 128, 128, 255,
        ];
        let file = std::fs::File::create(&path).unwrap();
        let mut enc = png::Encoder::new(file, w as u32, h as u32);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        enc.write_header().unwrap().write_image_data(&data).unwrap();

        let px = read_px(&path, w, h);
        assert!(close(px[0], [1.0, 0.0, 0.0, 1.0]), "{:?}", px[0]);
        assert!(close(px[1], [0.0, 1.0, 0.0, 128.0 / 255.0]), "{:?}", px[1]);
        // 128/255 is 0.502, NOT the 0.216 an sRGB decode would give.
        assert!(close(px[3], [0.502, 0.502, 0.502, 1.0]), "{:?}", px[3]);
        let _ = std::fs::remove_file(&path);
    }

    /// 16-bit greyscale: big-endian samples, spread across rgb, opaque.
    #[test]
    fn png_grey16_spreads_over_rgb() {
        let path = tmp("grey16.png");
        let (w, h) = (2usize, 1usize);
        let data: Vec<u8> = vec![0x00, 0x00, 0xFF, 0xFF];
        let file = std::fs::File::create(&path).unwrap();
        let mut enc = png::Encoder::new(file, w as u32, h as u32);
        enc.set_color(png::ColorType::Grayscale);
        enc.set_depth(png::BitDepth::Sixteen);
        enc.write_header().unwrap().write_image_data(&data).unwrap();

        let px = read_px(&path, w, h);
        assert!(close(px[0], [0.0, 0.0, 0.0, 1.0]), "{:?}", px[0]);
        assert!(close(px[1], [1.0, 1.0, 1.0, 1.0]), "{:?}", px[1]);
        let _ = std::fs::remove_file(&path);
    }

    /// A real JPEG, encoded here (we ship a decoder only) and read back.
    /// A flat mid-grey block: chroma subsampling and the DCT both leave it
    /// alone, so anything but flat neutral grey is a wiring bug.
    #[test]
    fn jpeg_decodes_to_rgba() {
        let path = tmp("grey.jpg");
        let (w, h) = (16usize, 16usize);
        let rgb = vec![128u8; w * h * 3];
        jpeg_encoder::Encoder::new_file(&path, 95)
            .unwrap()
            .encode(&rgb, w as u16, h as u16, jpeg_encoder::ColorType::Rgb)
            .unwrap();

        let dec = StillDecoder;
        let info = dec.read_header(&path).unwrap();
        assert_eq!((info.width, info.height), (16, 16));
        // One image per file: no layers to choose, one frame to seek to.
        assert!(dec.layers(&path).unwrap().is_empty());
        assert_eq!(dec.frame_count(&path).unwrap(), 1);

        let px = read_px(&path, w, h);
        // JPEG has no alpha: every pixel opaque, never a transparent plate.
        assert!(px.iter().all(|p| p[3] == 1.0));
        // Code values, not linear: 128/255 stays 0.502, not sRGB's 0.216.
        assert!(px.iter().all(|p| close(*p, [0.502, 0.502, 0.502, 1.0])), "{:?}", px[0]);
        let _ = std::fs::remove_file(&path);
    }

    /// A frame of another size is fitted whole into the sequence's:
    /// a 4x8 portrait in an 8x4 frame becomes a 2x4 column in the middle,
    /// its colour kept, the sides transparent black.
    #[test]
    fn a_size_mismatch_is_fitted_into_the_frame() {
        let path = tmp("portrait.png");
        let file = std::fs::File::create(&path).unwrap();
        let mut enc = png::Encoder::new(file, 4, 8);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        let red: Vec<u8> = [255u8, 0, 0, 255].repeat(32);
        enc.write_header().unwrap().write_image_data(&red).unwrap();

        let dec = StillDecoder;
        let info = ImageInfo { width: 8, height: 4 };
        let stride = info.row_stride_f16();
        let mut buf = vec![f16::ONE; stride * 4];
        dec.decode_into(&path, info, stride, &mut buf).unwrap();
        let px = |x: usize, y: usize| -> [f32; 4] {
            let o = y * stride + x * 4;
            [buf[o].to_f32(), buf[o + 1].to_f32(), buf[o + 2].to_f32(), buf[o + 3].to_f32()]
        };
        for y in 0..4 {
            for x in [0, 2, 5, 7] {
                assert_eq!(px(x, y), [0.0; 4], "bar at {x},{y}");
            }
            for x in [3, 4] {
                assert!(close(px(x, y), [1.0, 0.0, 0.0, 1.0]), "{x},{y}: {:?}", px(x, y));
            }
        }
        assert_eq!(size_of(&path), Some((4, 8)));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn fit_keeps_aspect_and_centres() {
        assert_eq!(fit_rect(4160, 6240, 6240, 4160), (1733, 0, 2773, 4160));
        assert_eq!(fit_rect(6720, 4480, 6240, 4160), (0, 0, 6240, 4160));
        assert_eq!(fit_rect(10, 10, 10, 10), (0, 0, 10, 10));
    }

    /// The tent averages a downscale: a 1/0 stripe pattern halved is grey,
    /// not one of the stripes.
    #[test]
    fn a_downscale_averages() {
        let w = weights(8, 4);
        let src = [1.0f32, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        for (lo, ws) in &w {
            let v: f32 = ws.iter().enumerate().map(|(k, t)| src[lo + k] * t).sum();
            assert!((v - 0.5).abs() < 0.2, "{v}");
        }
    }
}
