//! Text animators: per-letter (word, line) motion for text clips, ported
//! from Wobble's Text node animators (`engine/src/textfx.rs`).
//!
//! An animator is After Effects' range selector plus the deltas its
//! selection drives: a unit (chars / letters / words / lines), a weight per
//! unit, and at full weight the offset, scale, rotation, tracking, opacity
//! and tint it applies to that unit's glyphs.
//!
//! Scrub has no keyframes, so time is the animator's own: an IN sweep over
//! the clip's first `in_frames` (every unit starts fully driven and is
//! released in turn, so the deltas are where the text comes FROM) and an
//! OUT sweep over its last `out_frames` (units are taken in turn). Between
//! the two an animator is settled and does nothing, so the text is exactly
//! its static self and the rasteriser reuses one bitmap. An animator with
//! neither is a HOLD: its range window (start / end / offset / shape)
//! selects constantly, Wobble's selector verbatim — a tint on one word, a
//! wave in the baseline.
//!
//! Positions are in ems (the style's size), so an animator means the same
//! at any resolution.

use serde::{Deserialize, Serialize};

/// What an animator's range counts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum TextUnit {
    Chars,
    /// Spaces take no share and are not counted.
    #[default]
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

/// How a hold animator's selection falls across its window.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
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
    pub const ALL: [RangeShape; 6] = [RangeShape::Square, RangeShape::RampUp, RangeShape::RampDown, RangeShape::Triangle, RangeShape::Round, RangeShape::Smooth];
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

/// How each unit moves through its own part of a sweep.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum Ease {
    Linear,
    #[default]
    Out,
    In,
    InOut,
    /// Overshoots and settles back (a pop).
    Back,
    /// All or nothing (a typewriter).
    Step,
}

impl Ease {
    pub const ALL: [Ease; 6] = [Ease::Linear, Ease::Out, Ease::In, Ease::InOut, Ease::Back, Ease::Step];
    pub fn name(self) -> &'static str {
        match self {
            Ease::Linear => "linear",
            Ease::Out => "ease out",
            Ease::In => "ease in",
            Ease::InOut => "ease in out",
            Ease::Back => "overshoot",
            Ease::Step => "step",
        }
    }
    /// 0..1 -> 0..1 (Back runs past 1 on the way).
    pub fn apply(self, u: f32) -> f32 {
        let u = u.clamp(0.0, 1.0);
        match self {
            Ease::Linear => u,
            Ease::Out => 1.0 - (1.0 - u) * (1.0 - u) * (1.0 - u),
            Ease::In => u * u * u,
            Ease::InOut => u * u * (3.0 - 2.0 * u),
            Ease::Back => {
                let (c1, v) = (1.70158f32, u - 1.0);
                1.0 + (c1 + 1.0) * v * v * v + c1 * v * v
            }
            Ease::Step => {
                if u >= 0.5 {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }
}

/// A range selector, its timing, and the deltas its selection drives.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TextAnim {
    pub name: String,
    pub on: bool,
    pub unit: TextUnit,
    /// Hold only: the window, percent of the units; `offset` slides both
    /// ends; `shape` is how the selection falls across it.
    pub shape: RangeShape,
    pub start: f32,
    pub end: f32,
    pub offset: f32,
    /// Percent of the units a sweep's (or a square window's) edge spans:
    /// 0 is one unit at a time, 100 the whole text at once.
    pub softness: f32,
    /// Percent: 100 drives the full deltas, negative inverts them.
    pub amount: f32,
    /// Units take their turn in a shuffled order.
    pub random: bool,
    pub seed: u32,
    /// Frames from the clip's start the in sweep takes (0: none).
    pub in_frames: u32,
    /// Frames before the clip's end the out sweep takes (0: none).
    pub out_frames: u32,
    pub ease: Ease,
    // The deltas, at full selection.
    /// Ems, x right, y down.
    pub position: [f32; 2],
    /// Factor (1 = unchanged), about the glyph's middle.
    pub scale: f32,
    /// Degrees, clockwise.
    pub rotation: f32,
    /// Thousandths of an em after the glyph, as in a design app.
    pub tracking: f32,
    /// Factor (1 = unchanged).
    pub opacity: f32,
    /// A fill tint, straight RGBA; alpha is how much of it.
    pub tint: [u8; 4],
}

impl Default for TextAnim {
    fn default() -> Self {
        TextAnim {
            name: "animation".into(),
            on: true,
            unit: TextUnit::Letters,
            shape: RangeShape::Square,
            start: 0.0,
            end: 100.0,
            offset: 0.0,
            softness: 40.0,
            amount: 100.0,
            random: false,
            seed: 1,
            in_frames: 15,
            out_frames: 0,
            ease: Ease::Out,
            position: [0.0, 0.0],
            scale: 1.0,
            rotation: 0.0,
            tracking: 0.0,
            opacity: 1.0,
            tint: [255, 255, 255, 0],
        }
    }
}

/// The presets "+ add effect > text animation" offers.
pub const PRESETS: [&str; 7] = ["fade in by letter", "slide up by word", "pop in by letter", "typewriter", "drop in by line", "fade out by letter", "tint a word"];

impl TextAnim {
    /// A preset by `PRESETS` name (the default animator for any other).
    pub fn preset(name: &str) -> TextAnim {
        let base = TextAnim { name: name.into(), ..Default::default() };
        match name {
            "fade in by letter" => TextAnim { opacity: 0.0, in_frames: 18, softness: 50.0, ease: Ease::InOut, ..base },
            "slide up by word" => TextAnim { unit: TextUnit::Words, position: [0.0, 0.6], opacity: 0.0, in_frames: 20, softness: 60.0, ..base },
            "pop in by letter" => TextAnim { scale: 0.0, opacity: 0.0, in_frames: 14, softness: 35.0, ease: Ease::Back, ..base },
            "typewriter" => TextAnim { unit: TextUnit::Chars, opacity: 0.0, in_frames: 30, softness: 0.0, ease: Ease::Step, ..base },
            "drop in by line" => TextAnim { unit: TextUnit::Lines, position: [0.0, -0.8], opacity: 0.0, in_frames: 20, softness: 70.0, ease: Ease::Back, ..base },
            "fade out by letter" => TextAnim { opacity: 0.0, in_frames: 0, out_frames: 18, softness: 50.0, ease: Ease::InOut, ..base },
            "tint a word" => TextAnim { unit: TextUnit::Words, in_frames: 0, start: 0.0, end: 100.0 / 3.0, softness: 0.0, tint: [255, 210, 80, 255], ..base },
            _ => base,
        }
    }

    /// No sweeps: the window selects all the time.
    pub fn hold(&self) -> bool {
        self.in_frames == 0 && self.out_frames == 0
    }

    /// Whether its weights change at clip frame `frame` of `len` (false
    /// when off, settled, or a hold).
    pub fn moving(&self, frame: i64, len: i64) -> bool {
        if !self.on || self.hold() || self.amount == 0.0 {
            return false;
        }
        let f = frame.clamp(0, (len - 1).max(0));
        (self.in_frames > 0 && f < self.in_frames as i64) || (self.out_frames > 0 && f >= len - self.out_frames as i64)
    }

    /// The window's selection of unit `k` of `n` (shuffled when `random`),
    /// amount included — Wobble's range selector, verbatim.
    pub fn weight_of(&self, k: usize, n: usize) -> f32 {
        if n == 0 {
            return 0.0;
        }
        let k = self.order(k, n);
        let nf = n as f32;
        let c = (k as f32 + 0.5) / nf;
        let (mut s, mut e) = ((self.start + self.offset) / 100.0, (self.end + self.offset) / 100.0);
        if s > e {
            std::mem::swap(&mut s, &mut e);
        }
        let w = match self.shape {
            RangeShape::Square => {
                let soft = (self.softness / 100.0).max(1.0 / nf);
                ((c - s) / soft + 0.5).min((e - c) / soft + 0.5).clamp(0.0, 1.0)
            }
            RangeShape::RampUp => ramp(c, s, e),
            RangeShape::RampDown => 1.0 - ramp(c, s, e),
            _ => {
                if c < s || c > e || e <= s {
                    0.0
                } else {
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

    /// Unit `k` of `n`'s weight at clip frame `frame` of `len`: the hold
    /// window, or the sweeps (each unit eased through its own turn; 1 is
    /// fully driven, 0 at rest), amount included.
    pub fn weight_at(&self, k: usize, n: usize, frame: i64, len: i64) -> f32 {
        if !self.on || n == 0 {
            return 0.0;
        }
        if self.hold() {
            return self.weight_of(k, n);
        }
        let c = (self.order(k, n) as f32 + 0.5) / n as f32;
        let soft = (self.softness / 100.0).max(1.0 / n as f32);
        let f = frame.clamp(0, (len - 1).max(0)) as f32;
        // A front crossing the units, linear in time; each unit's part of
        // the crossing (0..1) is eased.
        let front = |p: f32| -soft / 2.0 + p.clamp(0.0, 1.0) * (1.0 + soft);
        let mut w = 0.0f32;
        let mut driven = false;
        if self.in_frames > 0 && f < self.in_frames as f32 {
            let u = ((front(f / self.in_frames as f32) - c) / soft + 0.5).clamp(0.0, 1.0);
            w = 1.0 - self.ease.apply(u);
            driven = true;
        }
        if self.out_frames > 0 {
            let o = self.out_frames as f32;
            let at = f - (len as f32 - o);
            if at >= 0.0 {
                let u = ((front((at + 1.0) / o) - c) / soft + 0.5).clamp(0.0, 1.0);
                let wo = self.ease.apply(u);
                w = if driven { w.max(wo) } else { wo };
            }
        }
        w * self.amount / 100.0
    }

    fn order(&self, k: usize, n: usize) -> usize {
        let k = k.min(n - 1);
        if self.random {
            shuffle(n, self.seed)[k]
        } else {
            k
        }
    }
}

fn ramp(c: f32, s: f32, e: f32) -> f32 {
    if e <= s {
        return if c >= e { 1.0 } else { 0.0 };
    }
    ((c - s) / (e - s)).clamp(0.0, 1.0)
}

/// A seeded permutation of 0..n (Wobble's, so a seed shuffles alike).
fn shuffle(n: usize, seed: u32) -> Vec<usize> {
    let mut v: Vec<usize> = (0..n).collect();
    let mut s = seed as u64 ^ 0x9E37_79B9_7F4A_7C15;
    for i in (1..n).rev() {
        s = splitmix(s);
        v.swap(i, (s % (i as u64 + 1)) as usize);
    }
    v
}

fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Which unit of each kind a glyph is (`None`: a space, for letters and
/// words), and how many of each the text has.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Units {
    pub char: usize,
    pub letter: Option<usize>,
    pub word: Option<usize>,
    pub line: usize,
}

/// What every animator does to one glyph.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GlyphDelta {
    /// Ems.
    pub position: [f32; 2],
    pub scale: f32,
    /// Degrees.
    pub rotation: f32,
    /// Ems after the glyph.
    pub track: f32,
    pub opacity: f32,
    /// RGB 0..1 and how much of it.
    pub tint: [f32; 4],
}

impl Default for GlyphDelta {
    fn default() -> Self {
        GlyphDelta { position: [0.0; 2], scale: 1.0, rotation: 0.0, track: 0.0, opacity: 1.0, tint: [0.0; 4] }
    }
}

impl GlyphDelta {
    pub fn is_identity(&self) -> bool {
        *self == GlyphDelta::default()
    }
}

/// The deltas `anims` give the glyph `u` (of `counts` units) at clip frame
/// `frame` of `len`.
pub fn glyph_delta(anims: &[TextAnim], u: Units, counts: Units, frame: i64, len: i64) -> GlyphDelta {
    let mut d = GlyphDelta::default();
    for a in anims {
        let (k, n) = match a.unit {
            TextUnit::Chars => (Some(u.char), counts.char),
            TextUnit::Letters => (u.letter, counts.letter.unwrap_or(0)),
            TextUnit::Words => (u.word, counts.word.unwrap_or(0)),
            TextUnit::Lines => (Some(u.line), counts.line),
        };
        let Some(k) = k else { continue };
        let w = a.weight_at(k, n, frame, len);
        if w == 0.0 {
            continue;
        }
        d.position[0] += a.position[0] * w;
        d.position[1] += a.position[1] * w;
        d.scale *= (1.0 + (a.scale - 1.0) * w).max(0.0);
        d.rotation += a.rotation * w;
        d.track += a.tracking / 1000.0 * w;
        d.opacity *= (1.0 + (a.opacity - 1.0) * w).clamp(0.0, 1.0);
        let t = (a.tint[3] as f32 / 255.0 * w).clamp(0.0, 1.0);
        if t > 0.0 {
            // Later animators tint over earlier ones.
            for c in 0..3 {
                d.tint[c] = d.tint[c] * (1.0 - t) + a.tint[c] as f32 / 255.0 * t;
            }
            d.tint[3] += t * (1.0 - d.tint[3]);
        }
    }
    d
}

/// Whether any of `anims` moves at clip frame `frame` of `len`.
pub fn moving(anims: &[TextAnim], frame: i64, len: i64) -> bool {
    anims.iter().any(|a| a.moving(frame, len))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_range_selector_matches_wobbles() {
        // Wobble's own cases (engine/src/textfx.rs the_range_selector_shapes).
        let d = TextAnim { softness: 0.0, ..Default::default() };
        let a = TextAnim { end: 50.0, ..d.clone() };
        let w: Vec<f32> = (0..4).map(|k| a.weight_of(k, 4)).collect();
        assert_eq!(w, vec![1.0, 1.0, 0.0, 0.0], "square: the first half");
        let mid = TextAnim { start: 0.0, end: 12.5, ..d.clone() };
        assert!((mid.weight_of(0, 4) - 0.5).abs() < 1e-5, "half a unit in: half weight");
        let up = TextAnim { shape: RangeShape::RampUp, ..d.clone() };
        assert!(up.weight_of(0, 4) < up.weight_of(3, 4));
        let tri = TextAnim { shape: RangeShape::Triangle, ..d.clone() };
        assert!(tri.weight_of(1, 4) > tri.weight_of(0, 4) && tri.weight_of(2, 4) > tri.weight_of(3, 4));
        let neg = TextAnim { amount: -50.0, ..d.clone() };
        assert_eq!(neg.weight_of(0, 4), -0.5);
        let r = TextAnim { end: 50.0, random: true, seed: 7, ..d.clone() };
        let mut ws: Vec<f32> = (0..8).map(|k| r.weight_of(k, 8)).collect();
        ws.sort_by(f32::total_cmp);
        assert_eq!(ws, vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]);
        // The formula by hand: ramp up over 0..100, unit 1 of 4 sits at 37.5%.
        assert!((up.weight_of(1, 4) - 0.375).abs() < 1e-6);
        // Round: sqrt(1 - x^2) with x = -0.25 for unit 1 of 4.
        let round = TextAnim { shape: RangeShape::Round, ..d.clone() };
        assert!((round.weight_of(1, 4) - (1.0f32 - 0.0625).sqrt()).abs() < 1e-6);
        // Offset slides the window: 0..50 moved by 50 selects the back half.
        let slid = TextAnim { end: 50.0, offset: 50.0, ..d };
        assert_eq!((0..4).map(|k| slid.weight_of(k, 4)).collect::<Vec<_>>(), vec![0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn an_in_sweep_starts_driven_and_settles() {
        let a = TextAnim::preset("fade in by letter");
        let n = 10;
        let len = 100;
        // Frame 0: every unit fully driven (hidden).
        assert!((0..n).all(|k| (a.weight_at(k, n, 0, len) - 1.0).abs() < 1e-6));
        // Settled from in_frames on: nothing driven, and not moving.
        assert!((0..n).all(|k| a.weight_at(k, n, a.in_frames as i64, len) == 0.0));
        assert!(a.moving(0, len) && !a.moving(a.in_frames as i64, len) && !a.moving(len - 1, len));
        // Half way, the first letters are released before the last.
        let h = a.in_frames as i64 / 2;
        assert!(a.weight_at(0, n, h, len) < a.weight_at(n - 1, n, h, len));
        // An out sweep: at rest before it, all driven on the last frame.
        let o = TextAnim::preset("fade out by letter");
        assert!((0..n).all(|k| o.weight_at(k, n, len - o.out_frames as i64 - 1, len) == 0.0));
        assert!((0..n).all(|k| (o.weight_at(k, n, len - 1, len) - 1.0).abs() < 1e-6));
        assert!(o.moving(len - 1, len) && !o.moving(0, len));
        // A hold never moves and selects by its window.
        let t = TextAnim::preset("tint a word");
        assert!(t.hold() && !t.moving(0, len));
        assert!((t.weight_at(0, 3, 50, len) - 1.0).abs() < 1e-5 && t.weight_at(2, 3, 50, len) == 0.0);
    }

    #[test]
    fn deltas_combine() {
        let anims = vec![TextAnim::preset("pop in by letter")];
        let counts = Units { char: 3, letter: Some(3), word: Some(1), line: 1 };
        let u = Units { char: 0, letter: Some(0), word: Some(0), line: 0 };
        let d0 = glyph_delta(&anims, u, counts, 0, 50);
        assert_eq!(d0.scale, 0.0);
        assert_eq!(d0.opacity, 0.0);
        assert!(glyph_delta(&anims, u, counts, 30, 50).is_identity());
        // A space is no letter: letter animators leave it alone.
        let sp = Units { letter: None, ..u };
        assert!(glyph_delta(&anims, sp, counts, 0, 50).is_identity());
        // An off animator does nothing.
        let off = vec![TextAnim { on: false, ..TextAnim::preset("pop in by letter") }];
        assert!(glyph_delta(&off, u, counts, 0, 50).is_identity());
    }

    #[test]
    fn old_projects_load_and_plain_text_saves_as_before() {
        let old: crate::ClipText = serde_json::from_str(r#"{"text":"hi","style":1}"#).unwrap();
        assert!(old.anims.is_empty());
        assert_eq!(serde_json::to_string(&old).unwrap(), r#"{"text":"hi","style":1}"#);
        let with = crate::ClipText { anims: vec![TextAnim::preset("typewriter")], ..old };
        let back: crate::ClipText = serde_json::from_str(&serde_json::to_string(&with).unwrap()).unwrap();
        assert_eq!(back, with);
        // A partial animator fills the rest from the default.
        let a: TextAnim = serde_json::from_str(r#"{"opacity":0.0}"#).unwrap();
        assert_eq!(a, TextAnim { opacity: 0.0, ..Default::default() });
    }

    #[test]
    fn eases_end_where_they_start_and_stop() {
        for e in Ease::ALL {
            assert_eq!(e.apply(0.0), 0.0, "{e:?}");
            assert!((e.apply(1.0) - 1.0).abs() < 1e-5, "{e:?}");
        }
        assert!(Ease::Back.apply(0.8) > 1.0, "overshoots");
    }
}
