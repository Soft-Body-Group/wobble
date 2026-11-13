//! The Text node's typesetting: glyphs laid out one by one, animated one by
//! one, optionally run along a path — and turned into GPU instances the
//! text layer (`crate::textgpu`) draws as quads over a distance-field atlas.
//!
//! This is TextMeshPro's arrangement: every glyph is an SDF cell in a small
//! per-font atlas (rasterised once per char and cached), and what a glyph
//! does — where it sits, how big it is, its turn, colour and opacity — is a
//! few floats on its instance. Nothing here ever recompiles: the fused pass
//! samples one finished layer, and the layer is re-recorded only when the
//! instances' bytes change (`TextLayer::hash`).
//!
//! Coordinates are canvas pixels, y down. The layout works in EMS first
//! (block space: x from the alignment point, y from the block's top) and
//! scales by `size` at the end, so the animators' ranges and the path see
//! one coherent line of glyphs.
//!
//! **Animators** are After Effects range selectors over a unit (chars,
//! chars without spaces, words, lines): a start / end / offset window in
//! percent, a shape (square with softness, ramps, triangle, round,
//! smooth), an amount, random order, and a wiggle that jitters each unit's
//! share over time. What a unit's selection weight drives is a set of
//! deltas — offset, scale, rotation, skew, tracking, opacity, a fill tint,
//! blur and weight — all keyable channels.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, OnceLock};

use crate::geo::Style;
use crate::geotext::{self, TextAlign, TextFont, TextSpec};

/// Atlas pixels per em.
pub const EM: f32 = 96.0;
/// How far the field reaches either side of an outline, atlas px: the
/// budget outline + weight + glow + shadow softness share (a quarter em).
pub const SPREAD: f32 = 24.0;
/// Margin round each cell.
pub const PAD: f32 = SPREAD + 3.0;
/// Atlas row width before a new shelf starts.
const ROW: f32 = 2048.0;
/// Glyphs drawn at most (a long subtitle is a few hundred).
pub const MAX_GLYPHS: usize = 8192;

// ---------------------------------------------------------------- params --

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum VAlign {
    /// `position` is the block's top (the old Text node's meaning).
    #[default]
    Top,
    Middle,
    Bottom,
    /// `position` is the first line's baseline.
    Baseline,
}

impl VAlign {
    pub const ALL: [VAlign; 4] = [VAlign::Top, VAlign::Middle, VAlign::Bottom, VAlign::Baseline];
    pub fn name(self) -> &'static str {
        match self {
            VAlign::Top => "top",
            VAlign::Middle => "middle",
            VAlign::Bottom => "bottom",
            VAlign::Baseline => "baseline",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TextCase {
    #[default]
    AsTyped,
    Upper,
    Lower,
}

impl TextCase {
    pub const ALL: [TextCase; 3] = [TextCase::AsTyped, TextCase::Upper, TextCase::Lower];
    pub fn name(self) -> &'static str {
        match self {
            TextCase::AsTyped => "Aa",
            TextCase::Upper => "AA",
            TextCase::Lower => "aa",
        }
    }
}

/// What an animator's range counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TextUnit {
    #[default]
    Chars,
    /// Spaces take no share and are not counted.
    Letters,
    Words,
    Lines,
}

impl TextUnit {
    pub const ALL: [TextUnit; 4] = [TextUnit::Chars, TextUnit::Letters, TextUnit::Words, TextUnit::Lines];
    pub fn name(self) -> &'static str {
        match self {
            TextUnit::Chars => "chars",
            TextUnit::Letters => "letters",
            TextUnit::Words => "words",
            TextUnit::Lines => "lines",
        }
    }
}

/// How the selection falls across the range.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum RangeShape {
    /// Full inside, edges eased over `softness`.
    #[default]
    Square,
    /// 0 before the start, rising to 1 at the end, 1 after.
    RampUp,
    RampDown,
    Triangle,
    Round,
    Smooth,
}

impl RangeShape {
    pub const ALL: [RangeShape; 6] = [
        RangeShape::Square,
        RangeShape::RampUp,
        RangeShape::RampDown,
        RangeShape::Triangle,
        RangeShape::Round,
        RangeShape::Smooth,
    ];
    pub fn name(self) -> &'static str {
        match self {
            RangeShape::Square => "square",
            RangeShape::RampUp => "ramp up",
            RangeShape::RampDown => "ramp down",
            RangeShape::Triangle => "triangle",
            RangeShape::Round => "round",
            RangeShape::Smooth => "smooth",
        }
    }
}

/// A range selector and the deltas its selection drives. Every number is
/// a keyable channel (`text.a<i>.<name>`, see [`ANIM_PARAMS`]).
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize), serde(default))]
pub struct TextAnimator {
    pub name: String,
    pub on: bool,
    pub unit: TextUnit,
    pub shape: RangeShape,
    /// The window, percent of the units; `offset` slides both ends.
    pub start: f32,
    pub end: f32,
    pub offset: f32,
    /// Square only: how far the edges ease, percent.
    pub softness: f32,
    /// Percent: 100 drives the full deltas, negative inverts them.
    pub amount: f32,
    /// Units take their place in the range in a shuffled order.
    pub random: bool,
    pub seed: u32,
    /// 0..1: how much of each unit's share is a random walk over time.
    pub wiggle: f32,
    /// Wiggles a second.
    pub wiggle_rate: f32,
    // The deltas, at full selection.
    /// Canvas px.
    pub position: [f32; 2],
    /// Factor (1 = unchanged), per axis.
    pub scale: [f32; 2],
    /// Degrees.
    pub rotation: f32,
    /// Degrees (a lean, about the baseline).
    pub skew: f32,
    /// Thousandths of an em after the glyph, as in a design app.
    pub tracking: f32,
    /// Factor (1 = unchanged).
    pub opacity: f32,
    /// A fill tint; alpha is how much of it.
    pub tint: [f32; 4],
    /// Canvas px of extra edge softness.
    pub blur: f32,
    /// Ems of extra stroke weight.
    pub weight: f32,
}

impl Default for TextAnimator {
    fn default() -> Self {
        Self {
            name: "animator".into(),
            on: true,
            unit: TextUnit::Chars,
            shape: RangeShape::Square,
            start: 0.0,
            end: 100.0,
            offset: 0.0,
            softness: 0.0,
            amount: 100.0,
            random: false,
            seed: 1,
            wiggle: 0.0,
            wiggle_rate: 2.0,
            position: [0.0, 0.0],
            scale: [1.0, 1.0],
            rotation: 0.0,
            skew: 0.0,
            tracking: 0.0,
            opacity: 1.0,
            tint: [1.0, 1.0, 1.0, 0.0],
            blur: 0.0,
            weight: 0.0,
        }
    }
}

/// Every keyable animator number, by its channel name after `a<i>.`, in
/// panel order.
pub const ANIM_PARAMS: &[&str] = &[
    "start", "end", "offset", "softness", "amount", "wiggle", "wiggle_rate", "position.x", "position.y",
    "scale.x", "scale.y", "rotation", "skew", "tracking", "opacity", "tint.r", "tint.g", "tint.b", "tint.a",
    "blur", "weight",
];

impl TextAnimator {
    pub fn get(&self, name: &str) -> Option<f32> {
        Some(match name {
            "start" => self.start,
            "end" => self.end,
            "offset" => self.offset,
            "softness" => self.softness,
            "amount" => self.amount,
            "wiggle" => self.wiggle,
            "wiggle_rate" => self.wiggle_rate,
            "position.x" => self.position[0],
            "position.y" => self.position[1],
            "scale.x" => self.scale[0],
            "scale.y" => self.scale[1],
            "rotation" => self.rotation,
            "skew" => self.skew,
            "tracking" => self.tracking,
            "opacity" => self.opacity,
            "tint.r" => self.tint[0],
            "tint.g" => self.tint[1],
            "tint.b" => self.tint[2],
            "tint.a" => self.tint[3],
            "blur" => self.blur,
            "weight" => self.weight,
            _ => return None,
        })
    }

    pub fn set(&mut self, name: &str, v: f32) -> bool {
        let f = match name {
            "start" => &mut self.start,
            "end" => &mut self.end,
            "offset" => &mut self.offset,
            "softness" => &mut self.softness,
            "amount" => &mut self.amount,
            "wiggle" => &mut self.wiggle,
            "wiggle_rate" => &mut self.wiggle_rate,
            "position.x" => &mut self.position[0],
            "position.y" => &mut self.position[1],
            "scale.x" => &mut self.scale[0],
            "scale.y" => &mut self.scale[1],
            "rotation" => &mut self.rotation,
            "skew" => &mut self.skew,
            "tracking" => &mut self.tracking,
            "opacity" => &mut self.opacity,
            "tint.r" => &mut self.tint[0],
            "tint.g" => &mut self.tint[1],
            "tint.b" => &mut self.tint[2],
            "tint.a" => &mut self.tint[3],
            "blur" => &mut self.blur,
            "weight" => &mut self.weight,
            _ => return false,
        };
        *f = v;
        true
    }

    /// The selection weight of unit `k` of `n` (shuffled when `random`),
    /// amount included — the range selector.
    pub fn weight_of(&self, k: usize, n: usize) -> f32 {
        if n == 0 {
            return 0.0;
        }
        let k = if self.random { shuffle(n, self.seed)[k.min(n - 1)] } else { k.min(n - 1) };
        let nf = n as f32;
        let c = (k as f32 + 0.5) / nf;
        let (mut s, mut e) = ((self.start + self.offset) / 100.0, (self.end + self.offset) / 100.0);
        if s > e {
            std::mem::swap(&mut s, &mut e);
        }
        let w = match self.shape {
            RangeShape::Square => {
                // The edge ramps over the softness, never less than one
                // unit, so a sweeping range moves a unit at a time without
                // popping and the default reads as the overlap.
                let soft = (self.softness / 100.0).max(1.0 / nf);
                ((c - s) / soft + 0.5).min((e - c) / soft + 0.5).clamp(0.0, 1.0)
            }
            RangeShape::RampUp => ramp(c, s, e),
            RangeShape::RampDown => 1.0 - ramp(c, s, e),
            _ => {
                if c < s || c > e || e <= s {
                    0.0
                } else {
                    // -1..1 across the range.
                    let x = (c - s) / (e - s) * 2.0 - 1.0;
                    match self.shape {
                        RangeShape::Triangle => 1.0 - x.abs(),
                        RangeShape::Round => (1.0 - x * x).max(0.0).sqrt(),
                        _ => 0.5 + 0.5 * (x * std::f32::consts::PI).cos(),
                    }
                }
            }
        };
        w * self.amount / 100.0
    }
}

fn ramp(c: f32, s: f32, e: f32) -> f32 {
    if e <= s {
        return if c >= e { 1.0 } else { 0.0 };
    }
    ((c - s) / (e - s)).clamp(0.0, 1.0)
}

/// A seeded permutation of 0..n (cached per (n, seed): a frame asks once
/// per glyph).
fn shuffle(n: usize, seed: u32) -> Arc<Vec<usize>> {
    static C: OnceLock<Mutex<HashMap<(usize, u32), Arc<Vec<usize>>>>> = OnceLock::new();
    let mut c = C.get_or_init(Default::default).lock().unwrap_or_else(|e| e.into_inner());
    if c.len() > 256 {
        c.clear();
    }
    c.entry((n, seed))
        .or_insert_with(|| {
            let mut v: Vec<usize> = (0..n).collect();
            let mut s = seed as u64 ^ 0x9E37_79B9_7F4A_7C15;
            for i in (1..n).rev() {
                s = splitmix(s);
                v.swap(i, (s % (i as u64 + 1)) as usize);
            }
            Arc::new(v)
        })
        .clone()
}

fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// -1..1 from integers.
fn hash11(a: u64) -> f32 {
    (splitmix(a) >> 40) as f32 / (1u64 << 23) as f32 - 1.0
}

/// Smooth 1D value noise, -1..1, one channel per `key`.
fn wiggle_noise(key: u64, t: f32) -> f32 {
    let i = t.floor();
    let f = t - i;
    let f = f * f * (3.0 - 2.0 * f);
    let i = i as i64 as u64;
    let a = hash11(key.wrapping_mul(0x1000_0001).wrapping_add(i));
    let b = hash11(key.wrapping_mul(0x1000_0001).wrapping_add(i.wrapping_add(1)));
    a + (b - a) * f
}

/// One point of a text path: a position and its two bezier handles,
/// relative to it (zero handles make a corner).
#[derive(Clone, Copy, Debug, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize), serde(default))]
pub struct PathPoint {
    pub p: [f32; 2],
    pub h_in: [f32; 2],
    pub h_out: [f32; 2],
}

/// Text on a path: the baseline follows a bezier spline in canvas px.
#[derive(Clone, Debug, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize), serde(default))]
pub struct TextPath {
    pub on: bool,
    pub points: Vec<PathPoint>,
    pub closed: bool,
    /// Canvas px along the path the text starts from (the alignment point:
    /// the start, middle or end of the path by the text's alignment).
    pub offset: f32,
    /// Glyphs turn with the path.
    pub perpendicular: bool,
    /// Run the other way.
    pub reverse: bool,
}

impl TextPath {
    /// Whether the path draws text (on, and at least two points).
    pub fn active(&self) -> bool {
        self.on && self.points.len() >= 2
    }

    /// The path as a polyline (canvas px) with its cumulative lengths.
    pub fn flatten(&self) -> (Vec<[f32; 2]>, Vec<f32>) {
        let mut pts: Vec<PathPoint> = self.points.clone();
        if self.reverse {
            pts.reverse();
            for p in &mut pts {
                std::mem::swap(&mut p.h_in, &mut p.h_out);
            }
        }
        let n = pts.len();
        let segs = if self.closed { n } else { n.saturating_sub(1) };
        let mut out = Vec::new();
        for i in 0..segs {
            let (a, b) = (pts[i], pts[(i + 1) % n]);
            let p0 = a.p;
            let p1 = [a.p[0] + a.h_out[0], a.p[1] + a.h_out[1]];
            let p2 = [b.p[0] + b.h_in[0], b.p[1] + b.h_in[1]];
            let p3 = b.p;
            const STEPS: usize = 48;
            for k in if i == 0 { 0 } else { 1 }..=STEPS {
                let t = k as f32 / STEPS as f32;
                let u = 1.0 - t;
                let (c0, c1, c2, c3) = (u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t);
                out.push([
                    c0 * p0[0] + c1 * p1[0] + c2 * p2[0] + c3 * p3[0],
                    c0 * p0[1] + c1 * p1[1] + c2 * p2[1] + c3 * p3[1],
                ]);
            }
        }
        let mut len = vec![0.0];
        for w in out.windows(2) {
            let l = len.last().unwrap() + ((w[1][0] - w[0][0]).hypot(w[1][1] - w[0][1]));
            len.push(l);
        }
        (out, len)
    }
}

/// A flattened path, sampled by arc length.
pub struct PathSampler {
    pts: Vec<[f32; 2]>,
    len: Vec<f32>,
    closed: bool,
}

impl PathSampler {
    pub fn new(path: &TextPath) -> Option<Self> {
        let (pts, len) = path.flatten();
        (pts.len() >= 2 && *len.last()? > 1e-3).then_some(Self { pts, len, closed: path.closed })
    }

    pub fn length(&self) -> f32 {
        *self.len.last().unwrap()
    }

    /// The point `s` px along, and the direction there (radians, y down).
    /// A closed path wraps; an open one runs on along its end tangents.
    pub fn at(&self, s: f32) -> ([f32; 2], f32) {
        let l = self.length();
        let s = if self.closed { s.rem_euclid(l) } else { s };
        let n = self.pts.len();
        let i = match self.len.binary_search_by(|v| v.total_cmp(&s)) {
            Ok(i) => i.min(n - 2),
            Err(i) => i.saturating_sub(1).min(n - 2),
        };
        let (a, b) = (self.pts[i], self.pts[i + 1]);
        let seg = (self.len[i + 1] - self.len[i]).max(1e-6);
        let t = (s - self.len[i]) / seg;
        let ang = (b[1] - a[1]).atan2(b[0] - a[0]);
        ([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t], ang)
    }
}

/// What a 2dini Text adds to its plain layout: the animators, the words'
/// case and box, how it rides a path wired into its `path` socket, and the
/// standalone Text node's typography and styling — anchor, kerning, italic,
/// scale, weight, an outline ring and a drop shadow — built as GEOMETRY
/// (copies of each glyph's outline under it), so 2dini's fills, gradients,
/// light and effects still apply. Lengths are GEO units (the canvas is 1
/// wide) except where marked; an animator's offsets and blur are canvas
/// pixels, as on the Text node. All default = the old 2dini Text, drawn by
/// the old code path.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize), serde(default))]
pub struct GeoTextFx {
    pub animators: Vec<TextAnimator>,
    pub case: TextCase,
    /// Paragraph box width, geo units; 0 = no wrapping.
    pub wrap: f32,
    pub justify: bool,
    /// Glyphs turn with the path.
    pub perpendicular: bool,
    pub reverse: bool,
    /// Geo units along the path from the alignment point.
    pub path_offset: f32,
    /// What `position` names: the first baseline (2dini's own meaning,
    /// the default), or the block's top, middle or bottom.
    pub valign: VAlign,
    pub kerning: bool,
    /// Degrees of lean (a faux italic).
    pub skew: f32,
    /// The block about its position.
    pub scale: [f32; 2],
    /// Ems of faux bold (0..): a stroke in the fill colour under each glyph.
    pub weight: f32,
    /// Ems of outline ring outside the glyph (0 = none), and its colour.
    pub outline: f32,
    pub outline_color: [f32; 4],
    /// A hard drop shadow: a copy of the glyphs (with their weight and
    /// outline) offset by `shadow_offset` CANVAS PIXELS, in its colour.
    pub shadow: bool,
    pub shadow_offset: [f32; 2],
    pub shadow_color: [f32; 4],
}

impl Default for GeoTextFx {
    fn default() -> Self {
        Self {
            animators: Vec::new(),
            case: TextCase::AsTyped,
            wrap: 0.0,
            justify: false,
            perpendicular: true,
            reverse: false,
            path_offset: 0.0,
            valign: VAlign::Baseline,
            kerning: true,
            skew: 0.0,
            scale: [1.0, 1.0],
            weight: 0.0,
            outline: 0.0,
            outline_color: [0.0, 0.0, 0.0, 1.0],
            shadow: false,
            shadow_offset: [6.0, 6.0],
            shadow_color: [0.0, 0.0, 0.0, 0.6],
        }
    }
}

/// A 2dini Text's keyable style numbers, by name, in panel order: its
/// literals sit between the path offset and the animators
/// (`geo::text_anim_index`).
pub const GEO_STYLE_PARAMS: &[&str] = &[
    "italic",
    "scale.x",
    "scale.y",
    "weight",
    "outline",
    "outline.r",
    "outline.g",
    "outline.b",
    "outline.a",
    "shadow.x",
    "shadow.y",
    "shadow.r",
    "shadow.g",
    "shadow.b",
    "shadow.a",
];

impl GeoTextFx {
    /// Whether anything here changes what the plain layout draws (the path
    /// is the socket's business).
    pub fn active(&self) -> bool {
        let d = GeoTextFx::default();
        !self.animators.is_empty()
            || self.case != TextCase::AsTyped
            || self.wrap > 0.0
            || self.valign != d.valign
            || self.kerning != d.kerning
            || self.skew != 0.0
            || self.scale != d.scale
            || self.weight > 0.0
            || self.outline > 0.0
            || self.shadow
    }

    fn style_slot(&mut self, name: &str) -> Option<&mut f32> {
        Some(match name {
            "italic" => &mut self.skew,
            "scale.x" => &mut self.scale[0],
            "scale.y" => &mut self.scale[1],
            "weight" => &mut self.weight,
            "outline" => &mut self.outline,
            "outline.r" => &mut self.outline_color[0],
            "outline.g" => &mut self.outline_color[1],
            "outline.b" => &mut self.outline_color[2],
            "outline.a" => &mut self.outline_color[3],
            "shadow.x" => &mut self.shadow_offset[0],
            "shadow.y" => &mut self.shadow_offset[1],
            "shadow.r" => &mut self.shadow_color[0],
            "shadow.g" => &mut self.shadow_color[1],
            "shadow.b" => &mut self.shadow_color[2],
            "shadow.a" => &mut self.shadow_color[3],
            _ => return None,
        })
    }

    /// A style number by its name in [`GEO_STYLE_PARAMS`].
    pub fn style_get(&self, name: &str) -> Option<f32> {
        self.clone().style_slot(name).map(|v| *v)
    }

    pub fn style_set(&mut self, name: &str, v: f32) -> bool {
        self.style_slot(name).map(|s| *s = v).is_some()
    }
}

// ----------------------------------------------------------------- atlas --

/// One glyph's distance field, cached per (font, char).
struct CellSdf {
    w: u32,
    h: u32,
    /// Encoded texels (0.5 + d / 2·SPREAD).
    texels: Vec<f32>,
    /// The pen's baseline origin inside the cell, cell px.
    origin: [f32; 2],
}

/// Where a char sits in the atlas, atlas px.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cell {
    /// x, y, w, h. An empty glyph (a space) has w = h = 0.
    pub rect: [f32; 4],
    /// The pen's baseline origin inside the rect.
    pub origin: [f32; 2],
}

/// A font's atlas for a set of chars.
pub struct Atlas {
    pub hash: u64,
    pub width: u32,
    pub height: u32,
    pub texels: Vec<half::f16>,
    pub cells: HashMap<char, Cell>,
}

fn font_key(f: &TextFont) -> String {
    match f {
        TextFont::Sans => "sans".into(),
        TextFont::Mono => "mono".into(),
        TextFont::File(p) => format!("file:{p}"),
    }
}

fn cell_sdf(font: &TextFont, ch: char) -> Option<Arc<CellSdf>> {
    static C: OnceLock<Mutex<HashMap<(String, char), Option<Arc<CellSdf>>>>> = OnceLock::new();
    let key = (font_key(font), ch);
    if let Some(c) = C.get_or_init(Default::default).lock().ok()?.get(&key) {
        return c.clone();
    }
    let s = ch.to_string();
    let mut geo = geotext::layout(&TextSpec {
        text: &s,
        font,
        size: EM,
        position: [0.0, 0.0],
        align: TextAlign::Left,
        tracking: 0.0,
        leading: 1.0,
        rotation: 0.0,
        style: Style::default(),
        px_per_unit: 1.0,
    });
    let made = if let Some((lo, hi)) = geo.bbox() {
        let (ox, oy) = (PAD - lo[0].floor(), PAD - lo[1].floor());
        for p in &mut geo.points {
            p.p = [p.p[0] + ox, p.p[1] + oy];
        }
        let w = (hi[0].ceil() - lo[0].floor() + 2.0 * PAD).max(1.0) as u32;
        let h = (hi[1].ceil() - lo[1].floor() + 2.0 * PAD).max(1.0) as u32;
        let texels = crate::textraster::sdf_spread(&geo, w, h, SPREAD)
            .into_iter()
            .map(|d| (0.5 + d / (2.0 * SPREAD)).clamp(0.0, 1.0))
            .collect();
        Some(Arc::new(CellSdf { w, h, texels, origin: [ox, oy] }))
    } else {
        None
    };
    let mut c = C.get_or_init(Default::default).lock().ok()?;
    if c.len() > 4096 {
        c.clear();
    }
    c.insert(key, made.clone());
    made
}

fn hash_of(h: &impl Hash) -> u64 {
    let mut s = std::collections::hash_map::DefaultHasher::new();
    h.hash(&mut s);
    s.finish()
}

/// The atlas of `chars` in `font` (cached by the set; each glyph's field is
/// cached on its own, so a new char costs one glyph, not the set). `None`
/// when the font will not load.
pub fn atlas(font: &TextFont, chars: &str) -> Option<Arc<Atlas>> {
    let mut set: Vec<char> = chars.chars().filter(|c| *c != '\n').collect();
    set.sort_unstable();
    set.dedup();
    let hash = hash_of(&("textfx", font_key(font), &set));
    static C: OnceLock<Mutex<HashMap<u64, Arc<Atlas>>>> = OnceLock::new();
    if let Some(a) = C.get_or_init(Default::default).lock().ok()?.get(&hash) {
        return Some(a.clone());
    }
    geotext::face_metrics(font)?;
    let cells: Vec<(char, Option<Arc<CellSdf>>)> = {
        use rayon::prelude::*;
        set.par_iter().map(|&c| (c, cell_sdf(font, c))).collect()
    };
    // Shelf packing, tallest first.
    let mut order: Vec<usize> = (0..cells.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(cells[i].1.as_ref().map_or(0, |c| c.h)));
    let (mut x, mut y, mut row_h, mut width) = (0u32, 0u32, 0u32, 1u32);
    let mut placed: HashMap<char, Cell> = HashMap::new();
    let mut at: Vec<(usize, u32, u32)> = Vec::new();
    for &i in &order {
        let (ch, c) = &cells[i];
        let Some(c) = c else {
            placed.insert(*ch, Cell { rect: [0.0; 4], origin: [0.0; 2] });
            continue;
        };
        if x > 0 && x + c.w > ROW as u32 {
            x = 0;
            y += row_h;
            row_h = 0;
        }
        placed.insert(*ch, Cell { rect: [x as f32, y as f32, c.w as f32, c.h as f32], origin: c.origin });
        at.push((i, x, y));
        x += c.w;
        width = width.max(x);
        row_h = row_h.max(c.h);
    }
    let height = (y + row_h).max(1);
    let mut texels = vec![half::f16::ZERO; (width * height) as usize];
    for (i, x0, y0) in at {
        let c = cells[i].1.as_ref().unwrap();
        for r in 0..c.h {
            let dst = ((y0 + r) * width + x0) as usize;
            for (k, v) in c.texels[(r * c.w) as usize..((r + 1) * c.w) as usize].iter().enumerate() {
                texels[dst + k] = half::f16::from_f32(*v);
            }
        }
    }
    let a = Arc::new(Atlas { hash, width, height, texels, cells: placed });
    let mut c = C.get_or_init(Default::default).lock().ok()?;
    if c.len() >= 64 {
        c.clear();
    }
    c.insert(hash, a.clone());
    Some(a)
}

// ---------------------------------------------------------------- layout --

/// A glyph placed in BLOCK space (ems: x from the alignment point, y from
/// the block's top, the pen's baseline origin), before any animator.
#[derive(Clone, Debug, PartialEq)]
pub struct Placed {
    pub ch: char,
    /// Pen x within its line from the line's start, ems (kerning and base
    /// tracking in).
    pub x: f32,
    /// Advance, ems.
    pub adv: f32,
    pub line: usize,
    /// Index among all glyphs, among letters (None for whitespace), and
    /// the word it belongs to (None for whitespace).
    pub index: usize,
    pub letter: Option<usize>,
    pub word: Option<usize>,
}

/// A laid-out block.
#[derive(Clone, Debug, PartialEq)]
pub struct Laid {
    pub glyphs: Vec<Placed>,
    /// Per line: width in ems (trailing spaces out), extra em per space a
    /// justified line spreads, and whether it ends a paragraph.
    pub lines: Vec<(f32, f32)>,
    pub letters: usize,
    pub words: usize,
    pub ascent: f32,
    pub descent: f32,
    pub cap: f32,
    /// Baseline to baseline, ems.
    pub leading: f32,
    /// The block's top, in ems relative to the position (valign).
    pub top: f32,
}

/// The text as cased.
pub fn cased(text: &str, case: TextCase) -> String {
    match case {
        TextCase::AsTyped => text.to_string(),
        TextCase::Upper => text.to_uppercase(),
        TextCase::Lower => text.to_lowercase(),
    }
}

/// Everything the layout reads.
pub struct LayoutSpec<'a> {
    pub text: &'a str,
    pub font: &'a TextFont,
    /// Thousandths of an em.
    pub tracking: f32,
    pub kerning: bool,
    pub leading: f32,
    /// Box width in ems (0 = no wrapping).
    pub wrap: f32,
    pub justify: bool,
    pub valign: VAlign,
}

pub fn layout(s: &LayoutSpec) -> Option<Laid> {
    let m = geotext::face_metrics(s.font)?;
    let track = s.tracking / 1000.0;
    let leading = if s.leading.is_finite() { s.leading.clamp(0.05, 10.0) } else { 1.2 };
    let mut glyphs = Vec::new();
    let mut lines = Vec::new();
    let (mut index, mut letters, mut words) = (0usize, 0usize, 0usize);
    let mut in_word = false;
    for para in s.text.split('\n') {
        let chars: Vec<char> = para.chars().collect();
        let adv = geotext::advances(s.font, &chars, s.kerning)?;
        // Line breaks within the paragraph: [start, end) of chars.
        let mut breaks: Vec<(usize, usize)> = Vec::new();
        let mut start = 0usize;
        let mut x = 0.0f32;
        let mut last_space: Option<usize> = None;
        let mut i = 0usize;
        while i < chars.len() {
            let k = if i == start { 0.0 } else { adv[i].1 };
            let right = x + k + adv[i].0;
            if s.wrap > 0.0 && right > s.wrap && !chars[i].is_whitespace() {
                if let Some(sp) = last_space.filter(|sp| *sp > start) {
                    breaks.push((start, sp));
                    start = sp + 1;
                    i = start;
                    x = 0.0;
                    last_space = None;
                    continue;
                }
            }
            if chars[i].is_whitespace() {
                last_space = Some(i);
            }
            x = right + track;
            i += 1;
        }
        breaks.push((start, chars.len()));
        let nb = breaks.len();
        for (bi, (a, b)) in breaks.into_iter().enumerate() {
            let line = lines.len();
            let mut x = 0.0f32;
            let mut width = 0.0f32;
            let mut spaces = 0usize;
            for (j, c) in chars.iter().enumerate().take(b).skip(a) {
                let c = *c;
                if j > a {
                    x += adv[j].1;
                }
                let ws = c.is_whitespace();
                if !ws && !in_word {
                    words += 1;
                }
                in_word = !ws;
                glyphs.push(Placed {
                    ch: c,
                    x,
                    adv: adv[j].0,
                    line,
                    index,
                    letter: (!ws).then_some(letters),
                    word: (!ws).then(|| words - 1),
                });
                index += 1;
                if !ws {
                    letters += 1;
                    width = x + adv[j].0;
                } else {
                    spaces += 1;
                }
                x += adv[j].0 + track;
            }
            // Spaces after the last ink do not count for justification.
            let trailing = chars[a..b].iter().rev().take_while(|c| c.is_whitespace()).count();
            let spread = if s.justify && s.wrap > 0.0 && bi + 1 < nb && spaces > trailing {
                ((s.wrap - width) / (spaces - trailing) as f32).max(0.0)
            } else {
                0.0
            };
            lines.push((width, spread));
        }
        in_word = false;
    }
    let n = lines.len().max(1) as f32;
    let h = m.ascent + m.descent + (n - 1.0) * leading;
    let top = match s.valign {
        VAlign::Top => 0.0,
        VAlign::Middle => -h * 0.5,
        VAlign::Bottom => -h,
        VAlign::Baseline => -m.ascent,
    };
    Some(Laid { glyphs, lines, letters, words, ascent: m.ascent, descent: m.descent, cap: m.cap, leading, top })
}

// ------------------------------------------------------------- instances --

/// One glyph on the GPU: six vec4s (`textgpu` mirrors the layout).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Instance {
    /// Cell px -> layer px: the 2x2 (columns) ...
    pub m: [f32; 4],
    /// ... the translation of the cell's corner, and the cell's corner in
    /// the atlas.
    pub t: [f32; 4],
    /// Cell size; outline width and weight, atlas px.
    pub c: [f32; 4],
    /// Straight fill and outline colours.
    pub fill: [f32; 4],
    pub line: [f32; 4],
    /// Opacity, edge softness (atlas px), -, -.
    pub x: [f32; 4],
}

/// The layer pass's uniform: six vec4s.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Uniform {
    /// Layer w, h; atlas w, h.
    pub dims: [f32; 4],
    /// Shadow offset (layer px), softness (atlas px), on.
    pub shadow: [f32; 4],
    pub shadow_col: [f32; 4],
    /// Glow width (atlas px), on, instance count, -.
    pub glow: [f32; 4],
    pub glow_col: [f32; 4],
    pub _pad: [f32; 4],
}

/// A 2D affine map: (x, y) -> (a x + c y + e, b x + d y + f).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Aff([f32; 6]);

impl Aff {
    fn tr(x: f32, y: f32) -> Aff {
        Aff([1.0, 0.0, 0.0, 1.0, x, y])
    }
    fn scale(x: f32, y: f32) -> Aff {
        Aff([x, 0.0, 0.0, y, 0.0, 0.0])
    }
    fn rot(r: f32) -> Aff {
        let (s, c) = r.sin_cos();
        Aff([c, s, -s, c, 0.0, 0.0])
    }
    /// A lean: x moves against height above the baseline (y down).
    fn skew(deg: f32) -> Aff {
        Aff([1.0, 0.0, -deg.to_radians().tan(), 1.0, 0.0, 0.0])
    }
    /// self ∘ o: apply `o` first.
    fn then(self, o: Aff) -> Aff {
        let [a, b, c, d, e, f] = self.0;
        let [p, q, r, s, t, u] = o.0;
        Aff([a * p + c * q, b * p + d * q, a * r + c * s, b * r + d * s, a * t + c * u + e, b * t + d * u + f])
    }
    fn apply(&self, p: [f32; 2]) -> [f32; 2] {
        let [a, b, c, d, e, f] = self.0;
        [a * p[0] + c * p[1] + e, b * p[0] + d * p[1] + f]
    }
}

/// What a glyph's animators add up to.
#[derive(Clone, Copy, Debug)]
struct Delta {
    pos: [f32; 2],
    scale: [f32; 2],
    rot: f32,
    skew: f32,
    track: f32,
    opacity: f32,
    tint: [f32; 4],
    blur: f32,
    weight: f32,
}

impl Default for Delta {
    fn default() -> Self {
        Self {
            pos: [0.0; 2],
            scale: [1.0; 2],
            rot: 0.0,
            skew: 0.0,
            track: 0.0,
            opacity: 1.0,
            tint: [0.0; 4],
            blur: 0.0,
            weight: 0.0,
        }
    }
}

/// The deltas every animator gives glyph `g` at `seconds`.
fn deltas(anims: &[TextAnimator], laid: &Laid, g: &Placed, seconds: f32) -> Delta {
    let mut d = Delta::default();
    for (ai, a) in anims.iter().enumerate() {
        if !a.on {
            continue;
        }
        let (k, n) = match a.unit {
            TextUnit::Chars => (Some(g.index), laid.glyphs.len()),
            TextUnit::Letters => (g.letter, laid.letters),
            TextUnit::Words => (g.word, laid.words),
            TextUnit::Lines => (Some(g.line), laid.lines.len()),
        };
        let Some(k) = k else { continue };
        let w = a.weight_of(k, n);
        if w == 0.0 && a.wiggle == 0.0 {
            continue;
        }
        // Each property's share: the selection, partly a random walk.
        let share = |prop: u64| -> f32 {
            if a.wiggle <= 0.0 {
                return w;
            }
            let key = (ai as u64) << 40 ^ (k as u64) << 8 ^ prop ^ (a.seed as u64) << 20;
            let j = wiggle_noise(key, seconds * a.wiggle_rate + (k as f32) * 0.37);
            w * ((1.0 - a.wiggle) + a.wiggle * j)
        };
        d.pos[0] += a.position[0] * share(1);
        d.pos[1] += a.position[1] * share(2);
        let ss = share(3);
        d.scale[0] *= 1.0 + (a.scale[0] - 1.0) * ss;
        d.scale[1] *= 1.0 + (a.scale[1] - 1.0) * ss;
        d.rot += a.rotation * share(4);
        d.skew += a.skew * share(5);
        d.track += a.tracking / 1000.0 * w;
        d.opacity *= (1.0 + (a.opacity - 1.0) * share(6)).max(0.0);
        let t = (a.tint[3] * w).clamp(0.0, 1.0);
        if t > 0.0 {
            // Later animators tint over earlier ones.
            for c in 0..3 {
                d.tint[c] = d.tint[c] * (1.0 - t) + a.tint[c] * t;
            }
            d.tint[3] = d.tint[3] + t * (1.0 - d.tint[3]);
        }
        d.blur += a.blur * share(7);
        d.weight += a.weight * w;
    }
    d
}

/// The text's look, as the instances read it.
pub struct Look<'a> {
    pub size: f32,
    pub position: [f32; 2],
    pub align: TextAlign,
    /// Degrees, factors, degrees (a lean) — the block about its position.
    pub rotation: f32,
    pub scale: [f32; 2],
    pub skew: f32,
    pub color: [f32; 4],
    /// Ems.
    pub weight: f32,
    /// Canvas px.
    pub softness: f32,
    /// Ems; 0 = none.
    pub outline: f32,
    pub outline_color: [f32; 4],
    pub anims: &'a [TextAnimator],
    pub path: Option<&'a TextPath>,
    pub seconds: f32,
}

/// One glyph placed: where its pen-relative pixels land, and what its
/// animators made of its colour and edges. The shared half of the Text
/// node's layer and 2dini's Text outlines.
#[derive(Clone, Copy, Debug)]
pub struct Place {
    /// Index into `Laid::glyphs`.
    pub glyph: usize,
    pub ch: char,
    /// Pen-relative px (x right, y down from the baseline origin) ->
    /// canvas px, as a 2x3 affine `[a, b, c, d, e, f]`:
    /// `(x, y) -> (a x + c y + e, b x + d y + f)`.
    pub map: [f32; 6],
    pub opacity: f32,
    /// Fill after the animators' tint.
    pub fill: [f32; 4],
    /// Canvas px of edge softness; ems of weight.
    pub blur: f32,
    pub weight: f32,
}

impl Place {
    pub fn apply(&self, p: [f32; 2]) -> [f32; 2] {
        Aff(self.map).apply(p)
    }
}

/// Every visible glyph of `laid` placed under `look` (whitespace and fully
/// faded glyphs left out).
pub fn places(laid: &Laid, look: &Look) -> Vec<Place> {
    let size = look.size.max(0.0);
    // Animator deltas first: tracking moves everything after it on a line,
    // so the lines are re-measured and re-aligned with them in.
    let ds: Vec<Delta> = laid.glyphs.iter().map(|g| deltas(look.anims, laid, g, look.seconds)).collect();
    let nl = laid.lines.len();
    let mut shift_x = vec![0.0f32; laid.glyphs.len()];
    let mut line_w = vec![0.0f32; nl];
    {
        let mut acc = vec![0.0f32; nl];
        for (i, g) in laid.glyphs.iter().enumerate() {
            let (_, spread) = laid.lines[g.line];
            shift_x[i] = acc[g.line];
            acc[g.line] += ds[i].track + if g.ch.is_whitespace() { spread } else { 0.0 };
            if !g.ch.is_whitespace() {
                line_w[g.line] = g.x + g.adv + shift_x[i];
            }
        }
    }
    let align = match look.align {
        TextAlign::Left => 0.0,
        TextAlign::Centre => 0.5,
        TextAlign::Right => 1.0,
    };
    let sampler = look.path.filter(|p| p.active()).and_then(PathSampler::new);
    let block = Aff::tr(look.position[0], look.position[1])
        .then(Aff::rot(look.rotation.to_radians()))
        .then(Aff::scale(look.scale[0], look.scale[1]));
    let mut out = Vec::new();
    for (i, g) in laid.glyphs.iter().enumerate() {
        if out.len() >= MAX_GLYPHS {
            break;
        }
        if g.ch.is_whitespace() {
            continue;
        }
        let d = &ds[i];
        if d.opacity <= 0.0 {
            continue;
        }
        let adv = g.adv * size;
        // Block-space pen (px): x from the alignment point, y the baseline.
        let bx = (g.x + shift_x[i] - align * line_w[g.line]) * size;
        let by = (laid.top + laid.ascent + g.line as f32 * laid.leading) * size;
        // The animators' transform about the glyph's middle.
        let (ax, ay) = (adv * 0.5, -laid.cap * size * 0.5);
        let anim = Aff::tr(ax + d.pos[0], ay + d.pos[1])
            .then(Aff::rot(d.rot.to_radians()))
            .then(Aff::scale(d.scale[0], d.scale[1]))
            .then(Aff::skew(d.skew))
            .then(Aff::tr(-ax, -ay));
        let local = Aff::skew(look.skew).then(anim);
        let place = match &sampler {
            Some(ps) => {
                let base = match look.align {
                    TextAlign::Left => 0.0,
                    TextAlign::Centre => ps.length() * 0.5,
                    TextAlign::Right => ps.length(),
                };
                let path = look.path.unwrap();
                let s = path.offset + base + bx + adv * 0.5;
                let (p, ang) = ps.at(s);
                let ang = if path.perpendicular { ang } else { 0.0 };
                // Lines after the first sit below the path by the leading.
                let dy = by - (laid.top + laid.ascent) * size;
                Aff::tr(p[0], p[1]).then(Aff::rot(ang)).then(Aff::tr(-adv * 0.5, dy))
            }
            None => block.then(Aff::tr(bx, by)),
        };
        let mut fill = look.color;
        for c in 0..3 {
            fill[c] = fill[c] * (1.0 - d.tint[3]) + d.tint[c] * d.tint[3];
        }
        out.push(Place {
            glyph: i,
            ch: g.ch,
            map: place.then(local).0,
            opacity: d.opacity,
            fill,
            blur: look.softness + d.blur,
            weight: look.weight + d.weight,
        });
    }
    out
}

/// The glyph instances for `laid` in `atlas` (`proxy` scales canvas px to
/// layer px), and each drawn glyph's quad in canvas px (for the viewer).
pub fn instances(laid: &Laid, atlas: &Atlas, look: &Look, proxy: f32) -> (Vec<Instance>, Vec<[[f32; 2]; 4]>) {
    let k = look.size.max(0.0) / EM;
    let layer = Aff::scale(proxy, proxy);
    let outline_atlas = (look.outline * EM).clamp(0.0, SPREAD - 2.0);
    let mut out = Vec::new();
    let mut quads = Vec::new();
    for pl in places(laid, look) {
        let Some(cell) = atlas.cells.get(&pl.ch) else { continue };
        if cell.rect[2] <= 0.0 {
            continue;
        }
        // Cell px -> pen px -> canvas px.
        let full = Aff(pl.map).then(Aff::scale(k, k)).then(Aff::tr(-cell.origin[0], -cell.origin[1]));
        let m = layer.then(full);
        let px_per_atlas = (m.0[0] * m.0[3] - m.0[1] * m.0[2]).abs().sqrt().max(1e-6);
        let soft = pl.blur.max(0.0) * proxy / px_per_atlas;
        let weight = (pl.weight * EM).clamp(-SPREAD + 2.0, SPREAD - 2.0 - outline_atlas);
        let [a, b, c, dd, e, f] = m.0;
        out.push(Instance {
            m: [a, b, c, dd],
            t: [e, f, cell.rect[0], cell.rect[1]],
            c: [cell.rect[2], cell.rect[3], outline_atlas, weight],
            fill: pl.fill,
            line: look.outline_color,
            x: [pl.opacity, soft, 0.0, 0.0],
        });
        let (w, h) = (cell.rect[2], cell.rect[3]);
        quads.push([[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]].map(|q| full.apply(q)));
    }
    (out, quads)
}

/// The bounding box of every quad, canvas px: (top-left, size).
pub fn bounds(quads: &[[[f32; 2]; 4]]) -> Option<([f32; 2], [f32; 2])> {
    let mut lo = [f32::MAX; 2];
    let mut hi = [f32::MIN; 2];
    for p in quads.iter().flatten() {
        lo = [lo[0].min(p[0]), lo[1].min(p[1])];
        hi = [hi[0].max(p[0]), hi[1].max(p[1])];
    }
    (lo[0] <= hi[0]).then(|| (lo, [hi[0] - lo[0], hi[1] - lo[1]]))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec(text: &str) -> LayoutSpec<'_> {
        LayoutSpec {
            text,
            font: &TextFont::Sans,
            tracking: 0.0,
            kerning: true,
            leading: 1.2,
            wrap: 0.0,
            justify: false,
            valign: VAlign::Top,
        }
    }

    #[test]
    fn units_are_counted_the_way_a_range_selector_counts_them() {
        let l = layout(&spec("ab cd\nef")).unwrap();
        assert_eq!(l.glyphs.len(), 7);
        assert_eq!(l.letters, 6);
        assert_eq!(l.words, 3);
        assert_eq!(l.lines.len(), 2);
        let words: Vec<_> = l.glyphs.iter().map(|g| g.word).collect();
        assert_eq!(words, vec![Some(0), Some(0), None, Some(1), Some(1), Some(2), Some(2)]);
        assert_eq!(l.glyphs[5].line, 1);
        assert_eq!(l.glyphs[5].x, 0.0, "a new line starts at its pen origin");
    }

    #[test]
    fn wrapping_breaks_at_spaces_and_justify_fills_the_box() {
        let one = layout(&spec("aaa bbb ccc")).unwrap();
        let w = one.lines[0].0;
        let l = layout(&LayoutSpec { wrap: w * 0.7, ..spec("aaa bbb ccc") }).unwrap();
        assert_eq!(l.lines.len(), 2);
        assert!(l.lines.iter().all(|(lw, _)| *lw <= w * 0.7 + 1e-4));
        let j = layout(&LayoutSpec { wrap: w * 0.7, justify: true, ..spec("aaa bbb ccc") }).unwrap();
        assert!(j.lines[0].1 > 0.0, "the first line spreads its space");
        assert_eq!(j.lines[1].1, 0.0, "the last line of a paragraph does not");
    }

    #[test]
    fn the_range_selector_shapes() {
        let a = TextAnimator { end: 50.0, ..Default::default() };
        let w: Vec<f32> = (0..4).map(|k| a.weight_of(k, 4)).collect();
        assert_eq!(w, vec![1.0, 1.0, 0.0, 0.0], "square: the first half");
        // Sweeping the offset moves the window a unit at a time, smoothly.
        let s = TextAnimator { start: 0.0, end: 0.0, offset: 0.0, ..Default::default() };
        let mid = TextAnimator { end: 12.5, ..s.clone() };
        assert!((mid.weight_of(0, 4) - 0.5).abs() < 1e-5, "half a unit in: half weight");
        let up = TextAnimator { shape: RangeShape::RampUp, ..Default::default() };
        assert!(up.weight_of(0, 4) < up.weight_of(3, 4));
        let tri = TextAnimator { shape: RangeShape::Triangle, ..Default::default() };
        assert!(tri.weight_of(1, 4) > tri.weight_of(0, 4) && tri.weight_of(2, 4) > tri.weight_of(3, 4));
        let neg = TextAnimator { amount: -50.0, ..Default::default() };
        assert_eq!(neg.weight_of(0, 4), -0.5);
        // Random order is a permutation: same weights, other places.
        let r = TextAnimator { end: 50.0, random: true, seed: 7, ..Default::default() };
        let mut ws: Vec<f32> = (0..8).map(|k| r.weight_of(k, 8)).collect();
        ws.sort_by(f32::total_cmp);
        assert_eq!(ws, vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn every_animator_param_reads_back_what_it_set() {
        let mut a = TextAnimator::default();
        for (i, n) in ANIM_PARAMS.iter().enumerate() {
            assert!(a.set(n, i as f32 + 0.5), "{n}");
            assert_eq!(a.get(n), Some(i as f32 + 0.5), "{n}");
        }
    }

    fn look(anims: &[TextAnimator]) -> Look<'_> {
        Look {
            size: 100.0,
            position: [0.0, 0.0],
            align: TextAlign::Left,
            rotation: 0.0,
            scale: [1.0; 2],
            skew: 0.0,
            color: [1.0; 4],
            weight: 0.0,
            softness: 0.0,
            outline: 0.0,
            outline_color: [0.0, 0.0, 0.0, 1.0],
            anims,
            path: None,
            seconds: 0.0,
        }
    }

    #[test]
    fn an_animator_moves_only_the_glyphs_it_selects() {
        let l = layout(&spec("HHHH")).unwrap();
        let a = atlas(&TextFont::Sans, "HHHH").unwrap();
        let (_, base) = instances(&l, &a, &look(&[]), 1.0);
        let up = TextAnimator { end: 50.0, position: [0.0, -30.0], ..Default::default() };
        let (_, moved) = instances(&l, &a, &look(std::slice::from_ref(&up)), 1.0);
        for i in 0..4 {
            let dy = moved[i][0][1] - base[i][0][1];
            let want = if i < 2 { -30.0 } else { 0.0 };
            assert!((dy - want).abs() < 1e-3, "glyph {i}: {dy}");
        }
        // Scale about the glyph's middle keeps it in place.
        let big = TextAnimator { scale: [2.0, 2.0], ..Default::default() };
        let (_, s) = instances(&l, &a, &look(std::slice::from_ref(&big)), 1.0);
        let cx = |q: &[[f32; 2]; 4]| (q[0][0] + q[2][0]) * 0.5;
        let wide = |q: &[[f32; 2]; 4]| q[2][0] - q[0][0];
        assert!((wide(&s[0]) / wide(&base[0]) - 2.0).abs() < 1e-3);
        assert!((cx(&s[1]) - cx(&base[1])).abs() < 60.0);
        // Opacity 0 drops the glyph.
        let gone = TextAnimator { end: 25.0, opacity: 0.0, ..Default::default() };
        let (inst, _) = instances(&l, &a, &look(std::slice::from_ref(&gone)), 1.0);
        assert_eq!(inst.len(), 3);
    }

    #[test]
    fn animated_tracking_spreads_a_centred_line_about_its_middle() {
        let l = layout(&spec("HHHH")).unwrap();
        let a = atlas(&TextFont::Sans, "H").unwrap();
        let wide = TextAnimator { tracking: 500.0, ..Default::default() };
        let mut lk = look(std::slice::from_ref(&wide));
        lk.align = TextAlign::Centre;
        let (_, q) = instances(&l, &a, &lk, 1.0);
        let (tl, wh) = bounds(&q).unwrap();
        assert!((tl[0] + wh[0] * 0.5).abs() < 2.0, "still centred: {tl:?} {wh:?}");
        let mut plain = look(&[]);
        plain.align = TextAlign::Centre;
        let (_, q0) = instances(&l, &a, &plain, 1.0);
        assert!(bounds(&q).unwrap().1[0] > bounds(&q0).unwrap().1[0] + 100.0);
    }

    #[test]
    fn text_on_a_path_follows_it_and_turns_with_it() {
        let l = layout(&spec("HHHHHH")).unwrap();
        let a = atlas(&TextFont::Sans, "H").unwrap();
        // A vertical line going down: glyphs march down it, turned a
        // quarter round.
        let path = TextPath {
            on: true,
            points: vec![
                PathPoint { p: [500.0, 100.0], ..Default::default() },
                PathPoint { p: [500.0, 900.0], ..Default::default() },
            ],
            perpendicular: true,
            ..Default::default()
        };
        let mut lk = look(&[]);
        lk.path = Some(&path);
        let (inst, q) = instances(&l, &a, &lk, 1.0);
        assert_eq!(inst.len(), 6);
        let c: Vec<[f32; 2]> = q.iter().map(|q| [(q[0][0] + q[2][0]) * 0.5, (q[0][1] + q[2][1]) * 0.5]).collect();
        for w in c.windows(2) {
            assert!(w[1][1] > w[0][1] + 20.0, "down the path: {c:?}");
        }
        assert!(c.iter().all(|p| (p[0] - 500.0).abs() < 80.0), "on the line: {c:?}");
        // Turned: the cell's x axis now points down.
        assert!(inst[0].m[1] > 0.9 * inst[0].m[0].abs().max(1e-3), "{:?}", inst[0].m);
        // An arc: a closed circle keeps every glyph on its radius.
        let r = 300.0;
        let kappa = 0.5523 * r;
        let circle = TextPath {
            on: true,
            closed: true,
            perpendicular: true,
            points: vec![
                PathPoint { p: [0.0, -r], h_in: [-kappa, 0.0], h_out: [kappa, 0.0] },
                PathPoint { p: [r, 0.0], h_in: [0.0, -kappa], h_out: [0.0, kappa] },
                PathPoint { p: [0.0, r], h_in: [kappa, 0.0], h_out: [-kappa, 0.0] },
                PathPoint { p: [-r, 0.0], h_in: [0.0, kappa], h_out: [0.0, -kappa] },
            ],
            ..Default::default()
        };
        let ps = PathSampler::new(&circle).unwrap();
        assert!((ps.length() - 2.0 * std::f32::consts::PI * r).abs() < 3.0, "{}", ps.length());
        for s in [0.0, 300.0, 1000.0, 2500.0] {
            let (p, _) = ps.at(s);
            assert!((p[0].hypot(p[1]) - r).abs() < 1.5, "{p:?}");
        }
    }

    #[test]
    fn wiggle_moves_over_time_and_holds_without_it() {
        let l = layout(&spec("HH")).unwrap();
        let a = atlas(&TextFont::Sans, "H").unwrap();
        let w = TextAnimator { position: [0.0, 40.0], wiggle: 1.0, ..Default::default() };
        let mut lk = look(std::slice::from_ref(&w));
        let (_, q0) = instances(&l, &a, &lk, 1.0);
        lk.seconds = 0.73;
        let (_, q1) = instances(&l, &a, &lk, 1.0);
        assert_ne!(q0, q1);
        let still = TextAnimator { position: [0.0, 40.0], ..Default::default() };
        let mut lk = look(std::slice::from_ref(&still));
        let (_, s0) = instances(&l, &a, &lk, 1.0);
        lk.seconds = 0.73;
        assert_eq!(s0, instances(&l, &a, &lk, 1.0).1);
    }

    #[test]
    fn a_new_char_reuses_every_cached_glyph() {
        let a = atlas(&TextFont::Sans, "abc").unwrap();
        let b = atlas(&TextFont::Sans, "cba").unwrap();
        assert!(Arc::ptr_eq(&a, &b), "the set, not the order");
        let c = atlas(&TextFont::Sans, "abcd").unwrap();
        assert_ne!(a.hash, c.hash);
        assert_eq!(c.cells.len(), 4);
        assert_eq!(c.cells[&'a'].rect[2], a.cells[&'a'].rect[2]);
    }
}
