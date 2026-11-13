//! Text as a texture: the glyph atlases the Text and Burnin nodes sample.
//!
//! A string is laid out by [`crate::geotext`] (the 2dini Text node's own
//! layout: egui's faces or a font file, kerning, lines, alignment) at a
//! fixed em of [`EM`] atlas pixels, and its outlines are turned into a
//! SIGNED DISTANCE field: each texel holds how far it is from the nearest
//! outline, positive inside. Sampled filtered and scaled, a distance field
//! stays sharp at any size, so a node's size, position, colour and the
//! proxy are UNIFORMS — keying the size of a title never re-rasterises.
//! Only the text itself (and its font, alignment, leading) does, and the
//! result is cached by that content's hash ([`text_block`], [`burnin`]),
//! so the compositor and the packer both ask freely.
//!
//! Texels are one half float, `0.5 + d / (2 * SPREAD)` clamped to 0..1:
//! the dummy texture (zero) reads as "far outside", so an atlas that has not
//! landed yet draws nothing rather than a grey block.
//!
//! The Burnin atlas holds its prefix and suffix as blocks and the glyphs
//! `0`–`9` and `-` in cells of their own: the frame number is a uniform and
//! the shader picks a cell per digit — no re-raster and no upload when the
//! frame changes.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, OnceLock};

use rayon::prelude::*;

use crate::geo::{Geometry, PrimKind, Style};
use crate::geotext::{self, TextAlign, TextFont, TextSpec};

/// Atlas pixels per em (scaled down for a block that would not fit the
/// largest texture; [`Raster::em`] is what was used).
pub const EM: f32 = 96.0;
/// How far, in atlas pixels, the field reaches either side of an outline.
pub const SPREAD: f32 = 10.0;
/// Margin round every block and cell, so filtering never reads a
/// neighbour's ink.
pub const PAD: f32 = SPREAD + 4.0;
/// Texture side the atlas is held under (wgpu's portable minimum).
const MAX_SIDE: f32 = 8000.0;
/// The glyphs a Burnin needs for its number, in cell order.
pub const CELLS: &str = "0123456789-";

/// One rasterised atlas and where things sit in it, in ATLAS pixels.
#[derive(Debug)]
pub struct Raster {
    /// The content hash it was made from (the compositor's upload key).
    pub hash: u64,
    pub width: u32,
    pub height: u32,
    /// `width * height` encoded distances, row-major, top row first.
    pub texels: Vec<half::f16>,
    /// Atlas pixels per em this one was made at.
    pub em: f32,
    /// Face ascent and descent (positive), atlas pixels.
    pub ascent: f32,
    pub descent: f32,
    /// Text: the block's width (widest line) and height (ascent of the
    /// first line to descent of the last), and its top-left in the atlas.
    pub block: [f32; 2],
    pub origin: [f32; 2],
    /// Burnin: where the prefix / suffix blocks start and their pen width.
    pub prefix: [f32; 2],
    pub suffix: [f32; 2],
    /// Burnin: the first cell's left edge, a cell's width (the widest
    /// digit's advance) and the distance between cells.
    pub cells_x: f32,
    pub cell_w: f32,
    pub pitch: f32,
}

/// Encode a signed distance (atlas px) as a texel value.
fn encode(d: f32) -> f32 {
    (0.5 + d / (2.0 * SPREAD)).clamp(0.0, 1.0)
}

/// Rasterise the closed prims of `geo` (atlas pixels, y down) into a
/// `w` x `h` distance field. Each prim's rings fill even-odd (a glyph's
/// counters are its holes); prims unite by taking the larger distance.
pub fn sdf(geo: &Geometry, w: u32, h: u32) -> Vec<f32> {
    sdf_spread(geo, w, h, SPREAD)
}

/// [`sdf`] with the field reaching `spread` pixels either side.
pub fn sdf_spread(geo: &Geometry, w: u32, h: u32, spread: f32) -> Vec<f32> {
    let mut field = vec![-spread; (w * h) as usize];
    let parts: Vec<(usize, usize, usize, usize, Vec<f32>)> = geo
        .prims
        .par_iter()
        .filter(|p| matches!(p.kind, PrimKind::Poly { closed: true }) && p.verts.len() >= 3)
        .filter_map(|p| {
            // The prim's rings as point lists.
            let mut starts: Vec<usize> = vec![0];
            starts.extend(p.rings.iter().map(|o| *o as usize));
            starts.push(p.verts.len());
            let rings: Vec<Vec<[f32; 2]>> = starts
                .windows(2)
                .map(|s| p.verts[s[0]..s[1]].iter().map(|v| geo.points[*v as usize].p).collect())
                .filter(|r: &Vec<[f32; 2]>| r.len() >= 3)
                .collect();
            let (mut lo, mut hi) = ([f32::MAX; 2], [f32::MIN; 2]);
            for q in rings.iter().flatten() {
                lo = [lo[0].min(q[0]), lo[1].min(q[1])];
                hi = [hi[0].max(q[0]), hi[1].max(q[1])];
            }
            let x0 = (lo[0] - spread).floor().max(0.0) as usize;
            let y0 = (lo[1] - spread).floor().max(0.0) as usize;
            let x1 = ((hi[0] + spread).ceil() as usize).min(w as usize);
            let y1 = ((hi[1] + spread).ceil() as usize).min(h as usize);
            if x1 <= x0 || y1 <= y0 {
                return None;
            }
            let mut out = vec![-spread; (x1 - x0) * (y1 - y0)];
            for y in y0..y1 {
                let qy = y as f32 + 0.5;
                for x in x0..x1 {
                    let qx = x as f32 + 0.5;
                    let mut inside = false;
                    let mut best = f32::MAX;
                    for r in &rings {
                        let n = r.len();
                        let mut j = n - 1;
                        for i in 0..n {
                            let (a, b) = (r[j], r[i]);
                            if (a[1] > qy) != (b[1] > qy)
                                && qx < (b[0] - a[0]) * (qy - a[1]) / (b[1] - a[1]) + a[0]
                            {
                                inside = !inside;
                            }
                            // Squared distance to the segment a..b.
                            let (ex, ey) = (b[0] - a[0], b[1] - a[1]);
                            let (px, py) = (qx - a[0], qy - a[1]);
                            let l2 = ex * ex + ey * ey;
                            let t = if l2 > 0.0 { ((px * ex + py * ey) / l2).clamp(0.0, 1.0) } else { 0.0 };
                            let (dx, dy) = (px - t * ex, py - t * ey);
                            best = best.min(dx * dx + dy * dy);
                            j = i;
                        }
                    }
                    let d = best.sqrt().min(spread);
                    out[(y - y0) * (x1 - x0) + (x - x0)] = if inside { d } else { -d };
                }
            }
            Some((x0, y0, x1, y1, out))
        })
        .collect();
    for (x0, y0, x1, _y1, out) in parts {
        let pw = x1 - x0;
        for (k, d) in out.iter().enumerate() {
            let i = (y0 + k / pw) * w as usize + x0 + k % pw;
            field[i] = field[i].max(*d);
        }
    }
    field
}

/// The coverage of output texel `(x, y)` at scale 1 — the CPU mirror of the
/// shader's edge, for tests.
pub fn coverage(r: &Raster, x: u32, y: u32) -> f32 {
    let v = r.texels[(y * r.width + x) as usize].to_f32();
    ((v - 0.5) * 2.0 * SPREAD + 0.5).clamp(0.0, 1.0)
}

fn spec<'a>(text: &'a str, font: &'a TextFont, em: f32, at: [f32; 2], align: TextAlign, leading: f32) -> TextSpec<'a> {
    TextSpec {
        text,
        font,
        size: em,
        position: at,
        align,
        tracking: 0.0,
        leading,
        rotation: 0.0,
        style: Style::default(),
        px_per_unit: 1.0,
    }
}

fn cache() -> &'static Mutex<HashMap<u64, Arc<Raster>>> {
    static C: OnceLock<Mutex<HashMap<u64, Arc<Raster>>>> = OnceLock::new();
    C.get_or_init(Default::default)
}

/// Cached by `hash`; `make` runs only on a miss. The cache is small and is
/// simply emptied when full — atlases are cheap to remake and a comp holds
/// a handful.
fn cached(hash: u64, make: impl FnOnce() -> Option<Raster>) -> Option<Arc<Raster>> {
    if let Some(r) = cache().lock().ok()?.get(&hash) {
        return Some(r.clone());
    }
    let r = Arc::new(make()?);
    let mut c = cache().lock().ok()?;
    if c.len() >= 64 {
        c.clear();
    }
    c.insert(hash, r.clone());
    Some(r)
}

fn key(parts: &impl Hash) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    parts.hash(&mut h);
    h.finish()
}

fn font_key(f: &TextFont) -> String {
    match f {
        TextFont::Sans => "sans".into(),
        TextFont::Mono => "mono".into(),
        TextFont::File(p) => format!("file:{p}"),
    }
}

fn finish(geo: &Geometry, w: f32, h: f32) -> (u32, u32, Vec<half::f16>) {
    let (w, h) = (w.ceil().max(1.0) as u32, h.ceil().max(1.0) as u32);
    let texels = sdf(geo, w, h).into_iter().map(|d| half::f16::from_f32(encode(d))).collect();
    (w, h, texels)
}

/// The atlas of a block of text: lines `leading` ems apart, aligned within
/// the block's width. `None` when the font will not load.
pub fn text_block(text: &str, font: &TextFont, align: TextAlign, leading: f32) -> Option<Arc<Raster>> {
    let leading = if leading.is_finite() { leading.clamp(0.1, 10.0) } else { 1.2 };
    let hash = key(&("text", text, font_key(font), align as u8, leading.to_bits()));
    cached(hash, || {
        let m = geotext::measure(text, font, 0.0)?;
        let wide = m.lines.iter().cloned().fold(0.0f32, f32::max);
        let tall = m.ascent + m.descent + (m.lines.len().max(1) - 1) as f32 * leading;
        // Scale down rather than exceed a texture (manifesto 3).
        let em = EM.min((MAX_SIDE - 2.0 * PAD) / wide.max(1e-3)).min((MAX_SIDE - 2.0 * PAD) / tall.max(1e-3));
        let x = PAD + wide * em * match align {
            TextAlign::Left => 0.0,
            TextAlign::Centre => 0.5,
            TextAlign::Right => 1.0,
        };
        let geo = geotext::layout(&spec(text, font, em, [x, PAD + m.ascent * em], align, leading));
        let (w, h, texels) = finish(&geo, wide * em + 2.0 * PAD, tall * em + 2.0 * PAD);
        Some(Raster {
            hash,
            width: w,
            height: h,
            texels,
            em,
            ascent: m.ascent * em,
            descent: m.descent * em,
            block: [wide * em, tall * em],
            origin: [PAD, PAD],
            prefix: [0.0; 2],
            suffix: [0.0; 2],
            cells_x: 0.0,
            cell_w: 0.0,
            pitch: 0.0,
        })
    })
}

/// A Burnin's atlas: `prefix`, `suffix` and the [`CELLS`] glyphs, one line.
pub fn burnin(prefix: &str, suffix: &str, font: &TextFont) -> Option<Arc<Raster>> {
    let prefix = prefix.lines().next().unwrap_or("");
    let suffix = suffix.lines().next().unwrap_or("");
    let hash = key(&("burnin", prefix, suffix, font_key(font)));
    cached(hash, || {
        let em = EM;
        let m = geotext::measure(prefix, font, 0.0)?;
        let pw = m.lines.first().copied().unwrap_or(0.0) * em;
        let sw = geotext::measure(suffix, font, 0.0)?.lines.first().copied().unwrap_or(0.0) * em;
        let advs: Vec<f32> = CELLS
            .chars()
            .map(|c| geotext::measure(&c.to_string(), font, 0.0).map_or(0.0, |m| m.lines[0] * em))
            .collect();
        let cell_w = advs[..10].iter().cloned().fold(0.0f32, f32::max).max(1.0);
        let pitch = cell_w + 2.0 * PAD;
        let base = PAD + m.ascent * em;
        let px = PAD;
        let sx = px + pw + 2.0 * PAD;
        let cx = sx + sw + 2.0 * PAD;
        let mut geo = geotext::layout(&spec(prefix, font, em, [px, base], TextAlign::Left, 1.2));
        let mut add = |g: Geometry| {
            let off = geo.points.len() as u32;
            geo.points.extend(g.points);
            for mut p in g.prims {
                for v in &mut p.verts {
                    *v += off;
                }
                geo.prims.push(p);
            }
        };
        add(geotext::layout(&spec(suffix, font, em, [sx, base], TextAlign::Left, 1.2)));
        for (i, c) in CELLS.chars().enumerate() {
            // Each glyph centred in its cell: digits are tabular in most
            // faces, and a narrow '1' stays put if not.
            let x = cx + i as f32 * pitch + 0.5 * (cell_w - advs[i]);
            add(geotext::layout(&spec(&c.to_string(), font, em, [x, base], TextAlign::Left, 1.2)));
        }
        let w = cx + CELLS.len() as f32 * pitch;
        let h = (m.ascent + m.descent) * em + 2.0 * PAD;
        let (w, h, texels) = finish(&geo, w, h);
        Some(Raster {
            hash,
            width: w,
            height: h,
            texels,
            em,
            ascent: m.ascent * em,
            descent: m.descent * em,
            block: [0.0; 2],
            origin: [PAD, PAD],
            prefix: [px, pw],
            suffix: [sx, sw],
            cells_x: cx,
            cell_w,
            pitch,
        })
    })
}

/// The shared WGSL: coverage of atlas point `a` (atlas pixels) when one
/// atlas pixel is `s` output pixels. Outside the atlas is outside the ink.
pub const TEXT_WGSL: &str = "// A text atlas's coverage at atlas point a (pixels) drawn at s output
// pixels per atlas pixel: the distance field's edge, one output pixel wide.
fn text_cov(t: texture_2d<f32>, a: vec2f, s: f32) -> f32 {
    let dims = vec2f(textureDimensions(t));
    if (any(a < vec2f(0.0)) || any(a > dims)) { return 0.0; }
    let v = textureSampleLevel(t, samp, a / dims, 0.0).r;
    return clamp((v - 0.5) * SPREADX2 * s + 0.5, 0.0, 1.0);
}
";

/// [`TEXT_WGSL`] with its constant filled in.
pub fn text_wgsl() -> String {
    TEXT_WGSL.replace("SPREADX2", &format!("{:.1}", 2.0 * SPREAD))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_o_has_a_hole_and_ink_round_it() {
        let r = text_block("o", &TextFont::Sans, TextAlign::Left, 1.2).unwrap();
        // The middle of the o (half way across, half way down the x-height).
        let cx = (r.origin[0] + r.block[0] * 0.5) as u32;
        let base = r.origin[1] + r.ascent;
        let xh = r.em * 0.5;
        let mid = (base - xh * 0.5) as u32;
        assert_eq!(coverage(&r, cx, mid), 0.0, "the counter is empty");
        // Walking left from the middle crosses ink before the edge.
        let ink = (0..cx).rev().any(|x| coverage(&r, x, mid) > 0.99);
        assert!(ink, "the o's stroke is inked");
        assert_eq!(coverage(&r, 0, 0), 0.0, "the margin is empty");
    }

    #[test]
    fn the_same_text_is_the_same_atlas_and_new_text_is_not() {
        let a = text_block("cache me", &TextFont::Sans, TextAlign::Left, 1.2).unwrap();
        let b = text_block("cache me", &TextFont::Sans, TextAlign::Left, 1.2).unwrap();
        assert!(Arc::ptr_eq(&a, &b));
        let c = text_block("cache you", &TextFont::Sans, TextAlign::Left, 1.2).unwrap();
        assert_ne!(a.hash, c.hash);
    }

    #[test]
    fn burnin_cells_hold_each_digit() {
        let r = burnin("f ", "", &TextFont::Mono).unwrap();
        assert!(r.cell_w > 0.0 && r.pitch > r.cell_w);
        assert!(r.width as f32 >= r.cells_x + CELLS.len() as f32 * r.pitch - 1.0);
        // Every cell has ink.
        for i in 0..CELLS.len() {
            let x0 = (r.cells_x + i as f32 * r.pitch) as u32;
            let x1 = (r.cells_x + i as f32 * r.pitch + r.cell_w) as u32;
            let any = (x0..x1).any(|x| (0..r.height).any(|y| coverage(&r, x, y) > 0.5));
            assert!(any, "cell {i} is empty");
        }
    }
}
