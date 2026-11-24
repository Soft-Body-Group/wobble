//! Pictures leaving and entering the comp.
//!
//! Every model we call works on DISPLAY-REFERRED sRGB: it was trained on
//! photographs as people look at them, not on scene-linear light. So a
//! reference leaves the comp as linear Rec.709 through the sRGB curve,
//! clipped to [0, 1], and a result comes back through the inverse (the
//! node's reader decodes with the sRGB input transform). The round trip is
//! exact below 1.0 and clips above it — which the panel says, because a
//! plate with highlights will come back without them.
//!
//! The app hands us Rec.709 linear; converting from the comp's working
//! space is its job (it owns `engine::color`), and this crate stays free of
//! the engine.

use anyhow::Result;

/// The sRGB transfer curve, linear -> encoded, clipped.
pub fn srgb_encode(x: f32) -> f32 {
    let x = if x.is_finite() { x.clamp(0.0, 1.0) } else { 0.0 };
    if x <= 0.003_130_8 {
        x * 12.92
    } else if x >= 1.0 {
        // Exactly white: 1.055 - 0.055 is not 1.0 in f32.
        1.0
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    }
}

pub fn srgb_decode(v: f32) -> f32 {
    if v <= 0.040_45 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

fn to_u8(v: f32) -> u8 {
    (v * 255.0 + 0.5).clamp(0.0, 255.0) as u8
}

/// Encode linear Rec.709 RGBA (straight, not premultiplied; rows top to
/// bottom, `w * 4` floats a row) as an sRGB RGB PNG. Alpha is dropped: no
/// model we call reads one, and a transparent reference is a request for
/// trouble.
pub fn png_srgb_rgb(rgba: &[f32], w: u32, h: u32) -> Result<Vec<u8>> {
    let n = (w * h) as usize;
    anyhow::ensure!(rgba.len() >= n * 4, "reference buffer too short");
    let mut px = Vec::with_capacity(n * 3);
    for p in rgba.chunks_exact(4).take(n) {
        px.push(to_u8(srgb_encode(p[0])));
        px.push(to_u8(srgb_encode(p[1])));
        px.push(to_u8(srgb_encode(p[2])));
    }
    encode(&px, w, h, png::ColorType::Rgb)
}

/// Encode a coverage matte (one float per pixel, 0..1) as an 8-bit grey PNG:
/// white where the model should work. Coverage is not light, so no curve.
pub fn png_matte(cov: &[f32], w: u32, h: u32) -> Result<Vec<u8>> {
    let n = (w * h) as usize;
    anyhow::ensure!(cov.len() >= n, "matte buffer too short");
    let px: Vec<u8> = cov[..n].iter().map(|&c| to_u8(if c.is_finite() { c } else { 0.0 })).collect();
    encode(&px, w, h, png::ColorType::Grayscale)
}

fn encode(px: &[u8], w: u32, h: u32, ct: png::ColorType) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut out, w, h);
        enc.set_color(ct);
        enc.set_depth(png::BitDepth::Eight);
        // Tag it, so a viewer (or a model's own preprocessing) that looks
        // does not have to guess.
        enc.set_source_srgb(png::SrgbRenderingIntent::Perceptual);
        let mut wr = enc.write_header()?;
        wr.write_image_data(px)?;
    }
    Ok(out)
}

/// Shrink RGBA (`w * 4` floats a row) by a whole factor until it fits in
/// `max_mp` megapixels: a box average, so a 4K plate becomes a clean 1.4K
/// reference rather than an aliased one. A reference travels as a data URI;
/// 4K of PNG in base64 is tens of megabytes on every call, and no model we
/// call works above two megapixels anyway.
pub fn shrink_rgba(rgba: &[f32], w: u32, h: u32, max_mp: f32) -> (Vec<f32>, u32, u32) {
    shrink(rgba, w, h, 4, max_mp)
}

/// `shrink_rgba` for one channel.
pub fn shrink_mono(v: &[f32], w: u32, h: u32, max_mp: f32) -> (Vec<f32>, u32, u32) {
    shrink(v, w, h, 1, max_mp)
}

fn shrink(src: &[f32], w: u32, h: u32, ch: usize, max_mp: f32) -> (Vec<f32>, u32, u32) {
    let mp = w as f64 * h as f64 / 1e6;
    let k = (mp / max_mp.max(0.05) as f64).sqrt().ceil().max(1.0) as u32;
    if k <= 1 {
        return (src.to_vec(), w, h);
    }
    let (nw, nh) = ((w / k).max(1), (h / k).max(1));
    let mut out = vec![0.0f32; (nw * nh) as usize * ch];
    let inv = 1.0 / (k * k) as f32;
    for y in 0..nh {
        for x in 0..nw {
            let o = ((y * nw + x) as usize) * ch;
            for dy in 0..k {
                let row = ((y * k + dy) * w) as usize;
                for dx in 0..k {
                    let i = (row + (x * k + dx) as usize) * ch;
                    for c in 0..ch {
                        out[o + c] += src[i + c];
                    }
                }
            }
            for c in 0..ch {
                out[o + c] *= inv;
            }
        }
    }
    (out, nw, nh)
}

/// Fit an aspect into a pixel budget, both sides multiples of 16 (every
/// diffusion model we call works in 8- or 16-pixel latents).
pub fn fit_size(aspect_w: u32, aspect_h: u32, megapixels: f32) -> (u32, u32) {
    let a = aspect_w.max(1) as f64 / aspect_h.max(1) as f64;
    let px = (megapixels.clamp(0.1, 4.0) as f64) * 1.0e6;
    let h = (px / a).sqrt();
    let w = h * a;
    let r16 = |v: f64| ((v / 16.0).round() as u32).clamp(16, 256) * 16;
    (r16(w), r16(h))
}

/// A file extension for a content type, else from the URL.
pub fn ext_for(content_type: Option<&str>, url: &str) -> &'static str {
    match content_type {
        Some("image/png") => "png",
        Some("image/jpeg") | Some("image/jpg") => "jpg",
        Some("image/webp") => "webp",
        Some("video/mp4") => "mp4",
        Some("video/quicktime") => "mov",
        Some("video/webm") => "webm",
        _ => {
            let path = url.split('?').next().unwrap_or(url).to_ascii_lowercase();
            if path.ends_with(".jpg") || path.ends_with(".jpeg") {
                "jpg"
            } else if path.ends_with(".webp") {
                "webp"
            } else if path.ends_with(".mp4") {
                "mp4"
            } else if path.ends_with(".mov") {
                "mov"
            } else if path.ends_with(".webm") {
                "webm"
            } else {
                "png"
            }
        }
    }
}

const PNG_SIG: &[u8] = b"\x89PNG\r\n\x1a\n";

fn crc32(parts: &[&[u8]]) -> u32 {
    let mut c = 0xffff_ffffu32;
    for p in parts {
        for &b in *p {
            c ^= b as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 { 0xedb8_8320 ^ (c >> 1) } else { c >> 1 };
            }
        }
    }
    !c
}

/// The same PNG with UTF-8 text chunks (`iTXt`, uncompressed) added right
/// after its header, so what made a picture travels INSIDE the file: copy
/// it out of the results folder and the prompt goes with it (exiftool,
/// Preview's inspector and most image tools show the chunks). `None` when
/// `bytes` is not a PNG; the caller keeps the file as it came.
pub fn png_with_text(bytes: &[u8], entries: &[(&str, &str)]) -> Option<Vec<u8>> {
    // Signature, then IHDR: length (4) + type (4) + 13 bytes + crc (4).
    let ihdr_end = PNG_SIG.len() + 4 + 4 + 13 + 4;
    if bytes.len() < ihdr_end || &bytes[..8] != PNG_SIG || &bytes[12..16] != b"IHDR" {
        return None;
    }
    let mut out = Vec::with_capacity(bytes.len() + 256);
    out.extend_from_slice(&bytes[..ihdr_end]);
    for (key, text) in entries {
        // keyword \0, compression flag 0, method 0, language \0, translated \0, text.
        let mut data = Vec::with_capacity(key.len() + text.len() + 5);
        data.extend_from_slice(key.as_bytes());
        data.extend_from_slice(&[0, 0, 0, 0, 0]);
        data.extend_from_slice(text.as_bytes());
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        out.extend_from_slice(b"iTXt");
        out.extend_from_slice(&data);
        out.extend_from_slice(&crc32(&[b"iTXt", &data]).to_be_bytes());
    }
    out.extend_from_slice(&bytes[ihdr_end..]);
    Some(out)
}

/// The uncompressed `iTXt` / `tEXt` chunks of a PNG, as (keyword, text).
pub fn png_text(bytes: &[u8]) -> Vec<(String, String)> {
    let mut v = vec![];
    if bytes.len() < 8 || &bytes[..8] != PNG_SIG {
        return v;
    }
    let mut at = 8;
    while at + 12 <= bytes.len() {
        let len = u32::from_be_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
        let ty = &bytes[at + 4..at + 8];
        let Some(data) = bytes.get(at + 8..at + 8 + len) else { break };
        let mut parts = data.splitn(2, |&b| b == 0);
        let key = String::from_utf8_lossy(parts.next().unwrap_or(&[])).into_owned();
        let rest = parts.next().unwrap_or(&[]);
        match ty {
            b"tEXt" => v.push((key, String::from_utf8_lossy(rest).into_owned())),
            b"iTXt" if rest.len() >= 2 && rest[0] == 0 => {
                // Skip the method byte, then language and translated keyword.
                let mut r = rest[2..].splitn(3, |&b| b == 0);
                let _lang = r.next();
                let _tr = r.next();
                v.push((key, String::from_utf8_lossy(r.next().unwrap_or(&[])).into_owned()));
            }
            b"IEND" => break,
            _ => {}
        }
        at += 12 + len;
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_srgb_curve_round_trips_below_one_and_clips_above() {
        for &x in &[0.0f32, 0.001, 0.18, 0.5, 0.99] {
            let back = srgb_decode(srgb_encode(x));
            assert!((back - x).abs() < 1e-5, "{x} -> {back}");
        }
        assert_eq!(srgb_encode(4.0), 1.0);
        assert_eq!(srgb_encode(f32::NAN), 0.0);
    }

    #[test]
    fn mid_grey_encodes_to_its_familiar_code_value() {
        // 18% grey is ~118 in 8-bit sRGB.
        assert_eq!(to_u8(srgb_encode(0.18)), 118);
    }

    #[test]
    fn a_png_is_written_and_reads_back_at_its_size() {
        let (w, h) = (5u32, 3u32);
        let buf = vec![0.18f32; (w * h * 4) as usize];
        let bytes = png_srgb_rgb(&buf, w, h).unwrap();
        let dec = png::Decoder::new(std::io::Cursor::new(bytes));
        let mut r = dec.read_info().unwrap();
        let mut out = vec![0; r.output_buffer_size().unwrap()];
        let info = r.next_frame(&mut out).unwrap();
        assert_eq!((info.width, info.height), (w, h));
        assert_eq!(out[0], 118);
    }

    #[test]
    fn a_4k_plate_shrinks_by_a_whole_factor_under_the_budget() {
        let (w, h) = (4096u32, 2160u32);
        let src = vec![0.5f32; (w * h * 4) as usize];
        let (out, nw, nh) = shrink_rgba(&src, w, h, 2.0);
        assert_eq!((nw, nh), (1365, 720));
        assert!((nw * nh) as f32 / 1e6 <= 2.0);
        assert!(out.iter().all(|v| (v - 0.5).abs() < 1e-6));
        // Already small: untouched.
        let (_, sw, sh) = shrink_mono(&vec![0.0; 100], 10, 10, 2.0);
        assert_eq!((sw, sh), (10, 10));
    }

    #[test]
    fn a_2k_comp_fits_into_a_megapixel_at_its_aspect() {
        let (w, h) = fit_size(2048, 1080, 1.0);
        assert_eq!((w % 16, h % 16), (0, 0));
        let a = w as f32 / h as f32;
        assert!((a - 2048.0 / 1080.0).abs() < 0.03, "{w}x{h}");
        assert!(((w * h) as f32 - 1.0e6).abs() < 0.05e6, "{w}x{h}");
    }

    #[test]
    fn text_chunks_ride_inside_a_png_and_it_still_decodes() {
        let png = super::png_srgb_rgb(&[0.5; 16], 2, 2).unwrap();
        let tagged = super::png_with_text(&png, &[("Description", "a moth, at dusk — 蛾"), ("wobble-genai", "{}")]).unwrap();
        assert_eq!(
            super::png_text(&tagged),
            vec![("Description".into(), "a moth, at dusk — 蛾".into()), ("wobble-genai".into(), "{}".into())]
        );
        let dec = png::Decoder::new(std::io::Cursor::new(&tagged));
        let r = dec.read_info().unwrap();
        assert_eq!(r.info().width, 2);
        assert!(super::png_with_text(b"not a png at all, sorry", &[("a", "b")]).is_none());
    }
}
