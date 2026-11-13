//! 2dini Text: a string laid out in a font and turned into geometry — one
//! closed shape per glyph outline, its counters (the hole in an "o") as
//! rings of that shape, so it fills, strokes, resamples, displaces, scatters
//! and simulates like anything else drawn.
//!
//! Faces: the two egui ships with (Ubuntu Light, Hack), already in the
//! build, or any .ttf/.otf file by path. Curves are flattened to about a
//! third of an output pixel. Kerning comes from the font's `kern` table
//! where it has one; the advance otherwise.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::geo::{Geometry, Point, Prim, PrimKind, Style};

#[derive(Clone, Debug, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TextFont {
    #[default]
    Sans,
    Mono,
    /// A font file on disk (.ttf / .otf; the first face of a collection).
    File(String),
}

impl TextFont {
    pub fn name(&self) -> String {
        match self {
            TextFont::Sans => "sans".into(),
            TextFont::Mono => "mono".into(),
            TextFont::File(p) => std::path::Path::new(p)
                .file_name()
                .map(|f| f.to_string_lossy().into_owned())
                .unwrap_or_else(|| p.clone()),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TextAlign {
    #[default]
    Left,
    Centre,
    Right,
}

impl TextAlign {
    pub const ALL: [TextAlign; 3] = [TextAlign::Left, TextAlign::Centre, TextAlign::Right];
    pub fn name(self) -> &'static str {
        match self {
            TextAlign::Left => "left",
            TextAlign::Centre => "centre",
            TextAlign::Right => "right",
        }
    }
}

/// A font's bytes: the built-ins borrowed, files read once and kept.
fn font_bytes(font: &TextFont) -> Option<Arc<Vec<u8>>> {
    static FILES: OnceLock<Mutex<HashMap<String, Option<Arc<Vec<u8>>>>>> = OnceLock::new();
    static SANS: OnceLock<Arc<Vec<u8>>> = OnceLock::new();
    static MONO: OnceLock<Arc<Vec<u8>>> = OnceLock::new();
    match font {
        TextFont::Sans => Some(SANS.get_or_init(|| Arc::new(epaint_default_fonts::UBUNTU_LIGHT.to_vec())).clone()),
        TextFont::Mono => Some(MONO.get_or_init(|| Arc::new(epaint_default_fonts::HACK_REGULAR.to_vec())).clone()),
        TextFont::File(path) => {
            let mut files = FILES.get_or_init(Default::default).lock().ok()?;
            files
                .entry(path.clone())
                .or_insert_with(|| std::fs::read(crate::paths::expand_home(path)).ok().map(Arc::new))
                .clone()
        }
    }
}

/// Collects a glyph's contours, flattening curves, in font units.
struct Outline {
    contours: Vec<Vec<[f32; 2]>>,
    cur: Vec<[f32; 2]>,
    /// Font units per flattening step's tolerance.
    tol: f32,
}

impl Outline {
    fn last(&self) -> [f32; 2] {
        self.cur.last().copied().unwrap_or([0.0, 0.0])
    }
    fn steps(&self, span: f32) -> usize {
        ((span / self.tol).sqrt().ceil() as usize).clamp(1, 32)
    }
}

impl ttf_parser::OutlineBuilder for Outline {
    fn move_to(&mut self, x: f32, y: f32) {
        if self.cur.len() > 2 {
            self.contours.push(std::mem::take(&mut self.cur));
        }
        self.cur.clear();
        self.cur.push([x, y]);
    }
    fn line_to(&mut self, x: f32, y: f32) {
        self.cur.push([x, y]);
    }
    fn quad_to(&mut self, x1: f32, y1: f32, x: f32, y: f32) {
        let p0 = self.last();
        // How far the curve bows from its chord bounds the steps it needs.
        let bow = ((p0[0] - 2.0 * x1 + x).powi(2) + (p0[1] - 2.0 * y1 + y).powi(2)).sqrt();
        let n = self.steps(bow);
        for k in 1..=n {
            let t = k as f32 / n as f32;
            let u = 1.0 - t;
            self.cur.push([
                u * u * p0[0] + 2.0 * u * t * x1 + t * t * x,
                u * u * p0[1] + 2.0 * u * t * y1 + t * t * y,
            ]);
        }
    }
    fn curve_to(&mut self, x1: f32, y1: f32, x2: f32, y2: f32, x: f32, y: f32) {
        let p0 = self.last();
        let bow = ((p0[0] - 2.0 * x1 + x2).abs() + (x1 - 2.0 * x2 + x).abs())
            .max((p0[1] - 2.0 * y1 + y2).abs() + (y1 - 2.0 * y2 + y).abs());
        let n = self.steps(bow * 1.5);
        for k in 1..=n {
            let t = k as f32 / n as f32;
            let u = 1.0 - t;
            let (a, b, c, d) = (u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t);
            self.cur.push([a * p0[0] + b * x1 + c * x2 + d * x, a * p0[1] + b * y1 + c * y2 + d * y]);
        }
    }
    fn close(&mut self) {
        if self.cur.len() > 2 {
            self.contours.push(std::mem::take(&mut self.cur));
        }
        self.cur.clear();
    }
}

fn area(p: &[[f32; 2]]) -> f32 {
    let n = p.len();
    (0..n).map(|i| p[i][0] * p[(i + 1) % n][1] - p[(i + 1) % n][0] * p[i][1]).sum::<f32>() * 0.5
}

fn inside(ring: &[[f32; 2]], q: [f32; 2]) -> bool {
    let n = ring.len();
    let mut c = false;
    let mut j = n - 1;
    for i in 0..n {
        let (a, b) = (ring[i], ring[j]);
        if (a[1] > q[1]) != (b[1] > q[1]) && q[0] < (b[0] - a[0]) * (q[1] - a[1]) / (b[1] - a[1]) + a[0] {
            c = !c;
        }
        j = i;
    }
    c
}

/// Everything the Text node lays out with.
pub struct TextSpec<'a> {
    pub text: &'a str,
    pub font: &'a TextFont,
    /// Em height, geo units.
    pub size: f32,
    /// The first line's baseline start (or centre / end, by `align`).
    pub position: [f32; 2],
    pub align: TextAlign,
    /// Extra space after every glyph, in ems.
    pub tracking: f32,
    /// Baseline to baseline, in ems.
    pub leading: f32,
    /// Radians, about `position`.
    pub rotation: f32,
    pub style: Style,
    /// Output pixels per geo unit, for the flattening tolerance.
    pub px_per_unit: f32,
}

/// A face's vertical metrics and each line's advance width, all in ems —
/// what a block of text needs to be sized and aligned before it is laid out
/// (the Text and Burnin nodes' atlases). `None` when the font will not load.
pub struct Measure {
    /// Baseline to the top of the tallest glyph the face declares.
    pub ascent: f32,
    /// Baseline to the bottom, positive.
    pub descent: f32,
    /// Pen travel of each line, kerning included, last tracking dropped —
    /// the width `layout` aligns by.
    pub lines: Vec<f32>,
}

pub fn measure(text: &str, font: &TextFont, tracking: f32) -> Option<Measure> {
    let bytes = font_bytes(font)?;
    let face = ttf_parser::Face::parse(&bytes, 0).ok()?;
    let upem = face.units_per_em().max(1) as f32;
    let kern = face.tables().kern;
    let mut lines = Vec::new();
    for line in text.split('\n') {
        let mut pen = 0.0f32;
        let mut prev: Option<ttf_parser::GlyphId> = None;
        for ch in line.chars() {
            let Some(gid) = face.glyph_index(ch).or_else(|| face.glyph_index('?')) else { continue };
            if let (Some(p), Some(kt)) = (prev, kern) {
                pen += kt
                    .subtables
                    .into_iter()
                    .filter(|s| s.horizontal && !s.variable)
                    .find_map(|s| s.glyphs_kerning(p, gid))
                    .map_or(0.0, |v| v as f32);
            }
            pen += face.glyph_hor_advance(gid).unwrap_or(0) as f32 + tracking * upem;
            prev = Some(gid);
        }
        lines.push(if prev.is_some() { (pen - tracking * upem).max(0.0) / upem } else { 0.0 });
    }
    Some(Measure {
        ascent: face.ascender() as f32 / upem,
        descent: -(face.descender() as f32) / upem,
        lines,
    })
}

/// A face's vertical metrics in ems: what a per-glyph layout (the Text
/// node's, `crate::textfx`) needs to stack and align lines.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FaceMetrics {
    pub ascent: f32,
    /// Positive.
    pub descent: f32,
    /// Cap height and x-height, from the OS/2 table, else estimated.
    pub cap: f32,
    pub x_height: f32,
}

pub fn face_metrics(font: &TextFont) -> Option<FaceMetrics> {
    let bytes = font_bytes(font)?;
    let face = ttf_parser::Face::parse(&bytes, 0).ok()?;
    let upem = face.units_per_em().max(1) as f32;
    let ascent = face.ascender() as f32 / upem;
    Some(FaceMetrics {
        ascent,
        descent: -(face.descender() as f32) / upem,
        cap: face.capital_height().map_or(ascent * 0.7, |v| v as f32 / upem),
        x_height: face.x_height().map_or(ascent * 0.5, |v| v as f32 / upem),
    })
}

/// Each char's advance and its kerning against the char BEFORE it (0 for
/// the first, or with `kerning` off), in ems. A char the face lacks takes
/// '?''s glyph, as [`layout`] draws it; one with neither advances 0.
pub fn advances(font: &TextFont, chars: &[char], kerning: bool) -> Option<Vec<(f32, f32)>> {
    let bytes = font_bytes(font)?;
    let face = ttf_parser::Face::parse(&bytes, 0).ok()?;
    let upem = face.units_per_em().max(1) as f32;
    let kern = if kerning { face.tables().kern } else { None };
    let mut prev: Option<ttf_parser::GlyphId> = None;
    Some(
        chars
            .iter()
            .map(|&ch| {
                let Some(gid) = face.glyph_index(ch).or_else(|| face.glyph_index('?')) else {
                    prev = None;
                    return (0.0, 0.0);
                };
                let k = match (prev, kern) {
                    (Some(p), Some(kt)) => kt
                        .subtables
                        .into_iter()
                        .filter(|s| s.horizontal && !s.variable)
                        .find_map(|s| s.glyphs_kerning(p, gid))
                        .map_or(0.0, |v| v as f32),
                    _ => 0.0,
                };
                prev = Some(gid);
                (face.glyph_hor_advance(gid).unwrap_or(0) as f32 / upem, k / upem)
            })
            .collect(),
    )
}

/// The text as geometry: one closed shape per glyph outline, counters as
/// its rings, in reading order. Empty when the font will not load.
pub fn layout(t: &TextSpec) -> Geometry {
    let mut geo = Geometry::default();
    let Some(bytes) = font_bytes(t.font) else { return geo };
    let Ok(face) = ttf_parser::Face::parse(&bytes, 0) else { return geo };
    let upem = face.units_per_em().max(1) as f32;
    // Font units -> geo units.
    let k = t.size / upem;
    if !(k.is_finite() && k > 0.0) {
        return geo;
    }
    let tol = (0.3 / t.px_per_unit.max(1.0)) / k;
    let kern = face.tables().kern;
    let kerning = |l: ttf_parser::GlyphId, r: ttf_parser::GlyphId| -> f32 {
        kern.map_or(0.0, |kt| {
            kt.subtables
                .into_iter()
                .filter(|s| s.horizontal && !s.variable)
                .find_map(|s| s.glyphs_kerning(l, r))
                .map_or(0.0, |v| v as f32)
        })
    };
    let (sn, cs) = t.rotation.sin_cos();
    let place = |x: f32, y: f32| -> [f32; 2] {
        // x right, font y UP -> geo y DOWN, then rotate about the anchor.
        let (dx, dy) = (x * k, -y * k);
        [t.position[0] + cs * dx - sn * dy, t.position[1] + sn * dx + cs * dy]
    };
    for (line_no, line) in t.text.split('\n').enumerate() {
        // Pen positions first, so the line can be aligned as a whole.
        let mut pen = 0.0f32;
        let mut glyphs = Vec::new();
        let mut prev: Option<ttf_parser::GlyphId> = None;
        for ch in line.chars() {
            let Some(gid) = face.glyph_index(ch).or_else(|| face.glyph_index('?')) else { continue };
            if let Some(p) = prev {
                pen += kerning(p, gid);
            }
            glyphs.push((gid, pen));
            pen += face.glyph_hor_advance(gid).unwrap_or(0) as f32 + t.tracking * upem;
            prev = Some(gid);
        }
        let width = (pen - t.tracking * upem).max(0.0);
        let shift = match t.align {
            TextAlign::Left => 0.0,
            TextAlign::Centre => -width * 0.5,
            TextAlign::Right => -width,
        };
        let base_y = -(line_no as f32) * t.leading * upem;
        for (gid, x0) in glyphs {
            let mut o = Outline { contours: Vec::new(), cur: Vec::new(), tol: tol.max(1e-3) };
            if face.outline_glyph(gid, &mut o).is_none() {
                continue;
            }
            if o.cur.len() > 2 {
                let c = std::mem::take(&mut o.cur);
                o.contours.push(c);
            }
            // In geo space now.
            let rings: Vec<Vec<[f32; 2]>> = o
                .contours
                .iter()
                .map(|c| c.iter().map(|p| place(p[0] + x0 + shift, p[1] + base_y)).collect())
                .filter(|r: &Vec<[f32; 2]>| r.len() >= 3 && area(r).abs() > 1e-12)
                .collect();
            if rings.is_empty() {
                continue;
            }
            // Outlines wind one way, counters the other: the biggest ring
            // says which is which. Each counter joins the smallest outline
            // around it.
            let big = rings.iter().map(|r| area(r)).fold(0.0f32, |a, v| if v.abs() > a.abs() { v } else { a });
            let (outers, holes): (Vec<usize>, Vec<usize>) =
                (0..rings.len()).partition(|&i| area(&rings[i]).signum() == big.signum());
            let mut owned: Vec<Vec<usize>> = vec![Vec::new(); outers.len()];
            for h in holes {
                let q = rings[h][0];
                let home = outers
                    .iter()
                    .enumerate()
                    .filter(|(_, o)| inside(&rings[**o], q))
                    .min_by(|a, b| area(&rings[*a.1]).abs().total_cmp(&area(&rings[*b.1]).abs()))
                    .map(|(k, _)| k);
                if let Some(k) = home {
                    owned[k].push(h);
                }
            }
            for (k, o) in outers.iter().enumerate() {
                let first = geo.points.len() as u32;
                let mut offs = Vec::new();
                for (n, r) in std::iter::once(*o).chain(owned[k].iter().copied()).enumerate() {
                    if n > 0 {
                        offs.push(geo.points.len() as u32 - first);
                    }
                    for p in &rings[r] {
                        geo.points.push(Point { p: *p, orient: t.rotation, ..Default::default() });
                    }
                }
                let verts = (first..geo.points.len() as u32).collect();
                geo.prims.push(Prim {
                    layer: Default::default(),
                    kind: PrimKind::Poly { closed: true },
                    verts,
                    style: t.style,
                    phys: None,
                    src: 0,
                    rings: offs,
                });
            }
        }
    }
    geo
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec<'a>(text: &'a str, font: &'a TextFont) -> TextSpec<'a> {
        TextSpec {
            text,
            font,
            size: 0.1,
            position: [0.1, 0.3],
            align: TextAlign::Left,
            tracking: 0.0,
            leading: 1.2,
            rotation: 0.0,
            style: Style::default(),
            px_per_unit: 1920.0,
        }
    }

    /// An "o" is one shape with a hole: a point in the counter is not inside,
    /// a point in the ink is. An "i" is two shapes (stem and dot).
    #[test]
    fn counters_are_holes_and_separate_outlines_are_shapes() {
        for font in [TextFont::Sans, TextFont::Mono] {
            let g = layout(&spec("o", &font));
            assert_eq!(g.prims.len(), 1, "{font:?}");
            assert_eq!(g.rings(0).len(), 2, "{font:?}: the o has a counter");
            let (lo, hi) = g.bbox().unwrap();
            let mid = [(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5];
            assert!(!g.contains(mid), "{font:?}: the middle of an o is its hole");
            // The ink: just inside the left edge, halfway down.
            assert!(g.contains([lo[0] + (hi[0] - lo[0]) * 0.04, mid[1]]), "{font:?}");
            let i = layout(&spec("i", &font));
            assert_eq!(i.prims.len(), 2, "{font:?}: stem and dot");
        }
    }

    /// The baseline is where it was put, y grows down, the size is the em,
    /// alignment moves the line, lines stack by the leading.
    #[test]
    fn placement_size_alignment_and_lines() {
        let f = TextFont::Sans;
        let g = layout(&spec("H", &f));
        let (lo, hi) = g.bbox().unwrap();
        // The H sits ON the baseline (y 0.3) and rises above it (smaller y).
        assert!((hi[1] - 0.3).abs() < 0.002, "baseline {}", hi[1]);
        let cap = hi[1] - lo[1];
        assert!(cap > 0.05 && cap < 0.1, "cap height {cap} of a 0.1 em");
        assert!(lo[0] >= 0.1 - 0.001);
        let right = layout(&TextSpec { align: TextAlign::Right, ..spec("H", &f) });
        assert!(right.bbox().unwrap().1[0] <= 0.1 + 0.002, "right-aligned ends at the anchor");
        let two = layout(&spec("H\nH", &f));
        let ys: Vec<f32> = (0..two.prims.len()).map(|i| two.centroid(i)[1]).collect();
        assert!((ys[1] - ys[0] - 0.12).abs() < 0.003, "leading {:?}", ys);
        // A missing file draws nothing rather than failing.
        assert!(layout(&spec("x", &TextFont::File("/no/such/font.ttf".into()))).prims.is_empty());
    }
}
