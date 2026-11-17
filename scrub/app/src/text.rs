//! Text rasterising: a text clip's words in its style, laid out and drawn
//! ONCE into a premultiplied RGBA bitmap placed in the raster. The record
//! viewer uploads it as a texture and composites it over the picture; the
//! renderer blends the very same bitmap (made at the output's size) into
//! its frames — one layout, so what the viewer shows is what renders.
//!
//! Fonts: any TTF / OTF / TTC (`TextStyle::font`, with its face index), or
//! the built-in face (egui's Ubuntu, so nothing needs installing). `fonts()`
//! lists the Mac's own font folders. Faces are loaded once and kept.
//!
//! The look, in drawing order: a box behind the block, a soft shadow down
//! and right, an outline (the glyphs grown by `outline × size`), the fill.

use ab_glyph::{point, Font, FontArc, PxScale, ScaleFont};
use model::{Align, TextStyle, VAlign};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

/// A rasterised text block: premultiplied RGBA, `w`×`h`, whose top left
/// sits at (`x`, `y`) in raster pixels.
#[derive(Clone, Debug)]
pub struct Bitmap {
    pub w: u32,
    pub h: u32,
    pub x: i32,
    pub y: i32,
    pub rgba: Vec<u8>,
}

fn faces() -> &'static Mutex<HashMap<(PathBuf, u32), Option<FontArc>>> {
    static F: OnceLock<Mutex<HashMap<(PathBuf, u32), Option<FontArc>>>> = OnceLock::new();
    F.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The built-in face.
fn builtin() -> FontArc {
    static B: OnceLock<FontArc> = OnceLock::new();
    B.get_or_init(|| {
        let defs = egui::FontDefinitions::default();
        let data = defs.font_data.get("Ubuntu-Light").or_else(|| defs.font_data.values().next()).expect("egui ships fonts");
        let bytes: Vec<u8> = data.font.to_vec();
        FontArc::try_from_vec(bytes).expect("egui's font parses")
    })
    .clone()
}

/// A style's face (the built-in one when it names none, or one that will
/// not load).
pub fn face(style: &TextStyle) -> FontArc {
    let Some(p) = &style.font else { return builtin() };
    let key = (p.clone(), style.font_index);
    let mut m = faces().lock().unwrap_or_else(|e| e.into_inner());
    m.entry(key)
        .or_insert_with(|| {
            let bytes = std::fs::read(p).ok()?;
            ab_glyph::FontVec::try_from_vec_and_index(bytes, style.font_index).ok().map(FontArc::new)
        })
        .clone()
        .unwrap_or_else(builtin)
}

/// Whether a font file loads (for the picker's warning).
pub fn loads(p: &Path, index: u32) -> bool {
    std::fs::read(p).ok().is_some_and(|b| ab_glyph::FontVec::try_from_vec_and_index(b, index).is_ok())
}

/// The Mac's fonts: (name, file), from the system, library and user font
/// folders, sorted by name. Names are the files' (cheap: nothing is read).
pub fn fonts() -> Arc<Vec<(String, PathBuf)>> {
    static L: OnceLock<Arc<Vec<(String, PathBuf)>>> = OnceLock::new();
    L.get_or_init(|| {
        let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_default();
        let dirs = [
            PathBuf::from("/System/Library/Fonts"),
            PathBuf::from("/System/Library/Fonts/Supplemental"),
            PathBuf::from("/Library/Fonts"),
            home.join("Library/Fonts"),
            PathBuf::from(std::env::var_os("WINDIR").unwrap_or_else(|| "C:\\Windows".into())).join("Fonts"),
            PathBuf::from(std::env::var_os("LOCALAPPDATA").unwrap_or_default()).join("Microsoft/Windows/Fonts"),
        ];
        let mut v = Vec::new();
        for d in dirs {
            for e in std::fs::read_dir(d).into_iter().flatten().flatten() {
                let p = e.path();
                let ok = p.extension().and_then(|x| x.to_str()).is_some_and(|x| matches!(x.to_ascii_lowercase().as_str(), "ttf" | "otf" | "ttc"));
                if ok {
                    let name = p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
                    v.push((name, p));
                }
            }
        }
        v.sort_by(|a, b| a.0.to_lowercase().cmp(&b.0.to_lowercase()));
        v.dedup_by(|a, b| a.0 == b.0);
        Arc::new(v)
    })
    .clone()
}

/// A cache key for `rasterize(text, style, raster)`.
pub fn key(text: &str, style: &TextStyle, raster: (u32, u32)) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    text.hash(&mut h);
    format!("{style:?}").hash(&mut h);
    raster.hash(&mut h);
    h.finish()
}

/// Lay `text` out in `style` for a `raster`-sized frame and draw it.
/// `None` for empty text or a frame too small to draw in.
pub fn rasterize(text: &str, style: &TextStyle, raster: (u32, u32)) -> Option<Bitmap> {
    if text.trim().is_empty() || raster.0 < 8 || raster.1 < 8 {
        return None;
    }
    let font = face(style);
    let px = (style.size.max(0.001) * raster.1 as f32).max(4.0);
    let sf = font.as_scaled(PxScale::from(px));
    let line_h = (sf.ascent() - sf.descent() + sf.line_gap()) * style.line_spacing.max(0.1);
    // Lines: glyph ids with their pen x, and each line's width.
    let lines: Vec<(Vec<(ab_glyph::GlyphId, f32)>, f32)> = text
        .split('\n')
        .map(|l| {
            let mut x = 0.0;
            let mut prev = None;
            let mut gs = Vec::new();
            for ch in l.chars() {
                let id = font.glyph_id(ch);
                if let Some(p) = prev {
                    x += sf.kern(p, id);
                }
                gs.push((id, x));
                x += sf.h_advance(id);
                prev = Some(id);
            }
            (gs, x)
        })
        .collect();
    let block_w = lines.iter().map(|l| l.1).fold(0.0f32, f32::max);
    let block_h = sf.ascent() - sf.descent() + line_h * (lines.len() as f32 - 1.0);
    let outline = style.outline.max(0.0) * px;
    let shadow_off = if style.shadow { px * 0.06 } else { 0.0 };
    let shadow_blur = if style.shadow { (px * 0.05).max(1.0) } else { 0.0 };
    let box_pad = if style.boxed { style.box_pad.max(0.0) * px } else { 0.0 };
    let pad = (outline + shadow_off + shadow_blur * 2.0 + box_pad + 2.0).ceil();
    let (w, h) = ((block_w + 2.0 * pad).ceil() as usize, (block_h + 2.0 * pad).ceil() as usize);
    if w == 0 || h == 0 || w * h > 64_000_000 {
        return None;
    }
    // Fill coverage.
    let mut fill = vec![0f32; w * h];
    for (k, (gs, lw)) in lines.iter().enumerate() {
        let lx = pad
            + match style.align {
                Align::Left => 0.0,
                Align::Center => (block_w - lw) / 2.0,
                Align::Right => block_w - lw,
            };
        let base = pad + sf.ascent() + line_h * k as f32;
        for &(id, gx) in gs {
            let g = id.with_scale_and_position(PxScale::from(px), point(lx + gx, base));
            if let Some(o) = font.outline_glyph(g) {
                let b = o.px_bounds();
                o.draw(|x, y, c| {
                    let (xx, yy) = (b.min.x as i32 + x as i32, b.min.y as i32 + y as i32);
                    if xx >= 0 && yy >= 0 && (xx as usize) < w && (yy as usize) < h {
                        let i = yy as usize * w + xx as usize;
                        fill[i] = (fill[i] + c).min(1.0);
                    }
                });
            }
        }
    }
    let mut out = vec![0f32; w * h * 4];
    let over = |out: &mut [f32], i: usize, col: [u8; 4], a: f32| {
        let a = a * col[3] as f32 / 255.0;
        if a <= 0.0 {
            return;
        }
        for c in 0..3 {
            out[i * 4 + c] = col[c] as f32 / 255.0 * a + out[i * 4 + c] * (1.0 - a);
        }
        out[i * 4 + 3] = a + out[i * 4 + 3] * (1.0 - a);
    };
    // The box.
    if style.boxed {
        let (x0, y0) = ((pad - box_pad).max(0.0) as usize, (pad - box_pad).max(0.0) as usize);
        let (x1, y1) = (((pad + block_w + box_pad).ceil() as usize).min(w), ((pad + block_h + box_pad).ceil() as usize).min(h));
        for y in y0..y1 {
            for x in x0..x1 {
                over(&mut out, y * w + x, style.box_color, 1.0);
            }
        }
    }
    // The outline: the fill grown by `outline` (a disc, sampled).
    let body = if outline > 0.25 { dilate(&fill, w, h, outline) } else { fill.clone() };
    // The shadow: the body moved down-right and blurred.
    if style.shadow {
        let mut sh = vec![0f32; w * h];
        let o = shadow_off.round() as usize;
        for y in o..h {
            for x in o..w {
                sh[y * w + x] = body[(y - o) * w + (x - o)];
            }
        }
        let sh = blur(&sh, w, h, shadow_blur.round().max(1.0) as usize);
        for (i, a) in sh.iter().enumerate() {
            over(&mut out, i, style.shadow_color, *a);
        }
    }
    if outline > 0.25 {
        for (i, a) in body.iter().enumerate() {
            over(&mut out, i, style.outline_color, *a);
        }
    }
    for (i, a) in fill.iter().enumerate() {
        over(&mut out, i, style.color, *a);
    }
    // Where the block goes in the raster.
    let (ax, ay) = (style.x * raster.0 as f32, style.y * raster.1 as f32);
    let left = match style.align {
        Align::Left => ax,
        Align::Center => ax - block_w / 2.0,
        Align::Right => ax - block_w,
    };
    let top = match style.valign {
        VAlign::Top => ay,
        VAlign::Middle => ay - block_h / 2.0,
        VAlign::Bottom => ay - block_h,
    };
    Some(Bitmap {
        w: w as u32,
        h: h as u32,
        x: (left - pad).round() as i32,
        y: (top - pad).round() as i32,
        rgba: out.iter().map(|v| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8).collect(),
    })
}

/// Coverage grown by radius `r`: the max over a sampled disc.
fn dilate(src: &[f32], w: usize, h: usize, r: f32) -> Vec<f32> {
    let mut offs = Vec::new();
    let ri = r.ceil() as i32;
    for dy in -ri..=ri {
        for dx in -ri..=ri {
            let d = ((dx * dx + dy * dy) as f32).sqrt();
            if d <= r + 0.5 && (dx, dy) != (0, 0) {
                offs.push((dx, dy, (r + 0.5 - d).clamp(0.0, 1.0)));
            }
        }
    }
    let mut out = src.to_vec();
    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let mut m = out[(y as usize) * w + x as usize];
            for &(dx, dy, wgt) in &offs {
                let (sx, sy) = (x + dx, y + dy);
                if sx >= 0 && sy >= 0 && (sx as usize) < w && (sy as usize) < h {
                    m = m.max(src[sy as usize * w + sx as usize] * wgt);
                }
            }
            out[y as usize * w + x as usize] = m;
        }
    }
    out
}

/// Two passes of a separable box blur of radius `r`.
fn blur(src: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    let pass = |s: &[f32], horiz: bool| -> Vec<f32> {
        let mut o = vec![0f32; w * h];
        let (n, m) = if horiz { (h, w) } else { (w, h) };
        for a in 0..n {
            let at = |b: usize| if horiz { a * w + b } else { b * w + a };
            let mut acc = 0.0;
            for b in 0..m.min(r + 1) {
                acc += s[at(b)];
            }
            for b in 0..m {
                o[at(b)] = acc / (2 * r + 1) as f32;
                if b + r + 1 < m {
                    acc += s[at(b + r + 1)];
                }
                if b >= r {
                    acc -= s[at(b - r)];
                }
            }
        }
        o
    };
    let a = pass(src, true);
    let a = pass(&a, false);
    let a = pass(&a, true);
    pass(&a, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ink(b: &Bitmap) -> usize {
        b.rgba.chunks(4).filter(|p| p[3] > 128).count()
    }

    #[test]
    fn text_draws_where_its_style_puts_it() {
        let s = TextStyle { shadow: false, ..Default::default() };
        let b = rasterize("Hello", &s, (1920, 1080)).unwrap();
        assert!(ink(&b) > 1000, "{}", ink(&b));
        // Centred in a 1920x1080 frame.
        let (cx, cy) = (b.x + b.w as i32 / 2, b.y + b.h as i32 / 2);
        assert!((cx - 960).abs() < 4 && (cy - 540).abs() < 4, "{cx} {cy}");
        // Twice the frame, twice the text.
        let b2 = rasterize("Hello", &s, (3840, 2160)).unwrap();
        assert!((b2.w as f32 / b.w as f32 - 2.0).abs() < 0.1);
        // Left aligned at 10%: the block starts there.
        let l = TextStyle { align: Align::Left, x: 0.1, ..s.clone() };
        let bl = rasterize("Hello", &l, (1920, 1080)).unwrap();
        assert!((bl.x - 192).abs() < 8, "{}", bl.x);
        assert!(rasterize("  ", &s, (1920, 1080)).is_none());
    }

    #[test]
    fn outline_box_and_lines_grow_the_ink() {
        let s = TextStyle { shadow: false, ..Default::default() };
        let plain = rasterize("Hi", &s, (1280, 720)).unwrap();
        let outlined = rasterize("Hi", &TextStyle { outline: 0.06, ..s.clone() }, (1280, 720)).unwrap();
        assert!(ink(&outlined) > ink(&plain) * 3 / 2);
        let boxed = rasterize("Hi", &TextStyle { boxed: true, ..s.clone() }, (1280, 720)).unwrap();
        assert!(ink(&boxed) > ink(&outlined));
        let two = rasterize("Hi\nHi", &s, (1280, 720)).unwrap();
        assert!(two.h > plain.h * 3 / 2);
        // The same key for the same inputs only.
        assert_eq!(key("a", &s, (1, 2)), key("a", &s, (1, 2)));
        assert_ne!(key("a", &s, (1, 2)), key("b", &s, (1, 2)));
    }

    #[test]
    fn a_system_font_loads_when_there_is_one() {
        let list = fonts();
        if let Some((_, p)) = list.iter().find(|(n, _)| n.eq_ignore_ascii_case("Helvetica") || n.eq_ignore_ascii_case("Arial")) {
            let s = TextStyle { font: Some(p.clone()), shadow: false, ..Default::default() };
            assert!(loads(p, 0));
            assert!(rasterize("Type", &s, (1920, 1080)).is_some());
        }
        // A missing file falls back to the built-in face rather than nothing.
        let s = TextStyle { font: Some("/nope.ttf".into()), ..Default::default() };
        assert!(rasterize("x", &s, (1920, 1080)).is_some());
    }
}
