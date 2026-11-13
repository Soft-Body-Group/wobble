//! Tone Map: highlight rolloff and tone mapping INSIDE the comp, on
//! scene-linear values — not the viewer transform, which only changes what
//! the monitor shows. This changes the pixels a Write gets.
//!
//! One scalar curve `F`, built in three steps:
//!
//! 1. **exposure** — `x * 2^stops`.
//! 2. **toe** (optional) — `x - k·x·(1 - x/p)²` below the pivot `p`, the
//!    identity above it. Value and slope match at the pivot, so it bends the
//!    shadows down without a crease; `k` 0 is off, 1 the deepest.
//! 3. **the operator**:
//!    - **Knee** (the default, the "Rolloff" entry): the identity below
//!      `threshold`, then a shoulder that leaves it at slope 1 and runs up
//!      to `white` without reaching it:
//!      `t + (w - t)·s(u)`, `u = (x - t)/(w - t)`, `s(u) = u / (1 + u^q)^(1/q)`.
//!      `softness` sets `q = 1/softness`: 1 is Reinhard's shoulder, near 0 is
//!      a hard clip at `white`.
//!    - **Exponential**: the same frame with `s(u) = 1 - e^-u`.
//!    - **Reinhard** (extended): `x(1 + x/W²)/(1 + x)`, input `white` → 1.
//!    - **Hable** (Uncharted 2 filmic), **ACES** (Narkowicz's fit) and
//!      **AgX-like** (the AgX log2 range and its contrast polynomial, back
//!      to linear by 2.2): whole-range curves, normalised so input `white`
//!      lands on 1. "AgX-like" because there is no inset/outset gamut
//!      matrix — on one scalar it would mean nothing.
//!
//! **How the curve meets rgb** (`mode`):
//! - **Hue preserving** (default): `rgb · F(m)/m`, `m = max(r,g,b)` — the
//!   ratios hold, so a hot saturated light keeps its hue and no channel
//!   passes the ceiling.
//! - **Luma**: the same with `m` = the working space's luminance. Colours
//!   hold, a saturated channel can run past white.
//! - **Per channel**: `F` on r, g and b separately — highlights skew and
//!   bleach towards white the way film does.
//!
//! Only positive values are curved; negatives (and a non-positive `m`)
//! pass through, alpha is never touched. `mix` and the mask dissolve back
//! to the input. Operator and mode are STRUCTURAL (the code differs);
//! every number is a uniform, and the normaliser `1/F(white)` is solved
//! on the CPU in `pack` (the ColorCompress precedent).
//! [`ToneMapParams::apply`] is the CPU mirror the tests hold the GPU to.

use crate::codegen::Emit;
use crate::nodes::NodeDef;

const TOPO_BASE: u64 = 0x70_4E_3A;

/// The curve itself.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ToneOp {
    /// Identity to the threshold, then a soft shoulder up to white.
    #[default]
    Knee,
    /// Identity to the threshold, then `1 - e^-u` up to white.
    Exponential,
    Reinhard,
    Hable,
    Aces,
    AgX,
}

impl ToneOp {
    pub const ALL: [ToneOp; 6] =
        [ToneOp::Knee, ToneOp::Exponential, ToneOp::Reinhard, ToneOp::Hable, ToneOp::Aces, ToneOp::AgX];

    pub fn label(self) -> &'static str {
        match self {
            ToneOp::Knee => "Knee",
            ToneOp::Exponential => "Exponential",
            ToneOp::Reinhard => "Reinhard",
            ToneOp::Hable => "Hable filmic",
            ToneOp::Aces => "ACES fit",
            ToneOp::AgX => "AgX-like",
        }
    }

    /// The shoulders keep everything under the threshold; the rest bend
    /// the whole range.
    pub fn has_threshold(self) -> bool {
        matches!(self, ToneOp::Knee | ToneOp::Exponential)
    }

    /// A sensible `white` to switch to: a ceiling for the shoulders, the
    /// input level that lands on 1 for the rest.
    pub fn default_white(self) -> f32 {
        match self {
            ToneOp::Knee | ToneOp::Exponential => 1.0,
            ToneOp::Reinhard => 4.0,
            ToneOp::Hable => 11.2,
            ToneOp::Aces => 16.0,
            ToneOp::AgX => 16.29,
        }
    }

    fn slug(self) -> &'static str {
        match self {
            ToneOp::Knee => "knee",
            ToneOp::Exponential => "exp",
            ToneOp::Reinhard => "reinhard",
            ToneOp::Hable => "hable",
            ToneOp::Aces => "aces",
            ToneOp::AgX => "agx",
        }
    }
}

/// What the curve is applied to.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ToneMode {
    /// By the largest channel: ratios (hue and saturation) hold.
    #[default]
    HuePreserving,
    /// By the working space's luminance.
    Luma,
    /// On r, g and b separately.
    PerChannel,
}

impl ToneMode {
    pub const ALL: [ToneMode; 3] = [ToneMode::HuePreserving, ToneMode::Luma, ToneMode::PerChannel];

    pub fn label(self) -> &'static str {
        match self {
            ToneMode::HuePreserving => "hue preserving",
            ToneMode::Luma => "luma",
            ToneMode::PerChannel => "per channel",
        }
    }

    fn slug(self) -> &'static str {
        match self {
            ToneMode::HuePreserving => "max",
            ToneMode::Luma => "luma",
            ToneMode::PerChannel => "rgb",
        }
    }
}

/// A [`crate::graph::NodeKind::ToneMap`]'s controls.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct ToneMapParams {
    pub op: ToneOp,
    pub mode: ToneMode,
    /// Stops in, before anything else.
    pub exposure: f32,
    /// Knee / Exponential: where the shoulder starts.
    pub threshold: f32,
    /// Knee / Exponential: the ceiling. The others: the input that lands on 1.
    pub white: f32,
    /// Knee: 1 is Reinhard's shoulder, towards 0 a hard clip.
    pub softness: f32,
    /// 0 is no toe, 1 the deepest.
    pub toe: f32,
    /// Where the toe hands back to the identity.
    pub toe_pivot: f32,
    /// 0 is the input, 1 the full curve.
    pub mix: f32,
}

impl Default for ToneMapParams {
    fn default() -> Self {
        Self {
            op: ToneOp::Knee,
            mode: ToneMode::HuePreserving,
            exposure: 0.0,
            threshold: 0.8,
            white: 1.0,
            softness: 0.5,
            toe: 0.0,
            toe_pivot: 0.18,
            mix: 1.0,
        }
    }
}

fn fin(v: f32, or: f32) -> f32 {
    if v.is_finite() { v } else { or }
}

/// Hable's curve, unnormalised.
fn hable(x: f32) -> f32 {
    let (a, b, c, d, e, f) = (0.15, 0.50, 0.10, 0.20, 0.02, 0.30);
    (x * (a * x + c * b) + d * e) / (x * (a * x + b) + d * f) - e / f
}

/// Narkowicz's ACES fit, unnormalised (his 0.6 pre-scale included).
fn aces(x: f32) -> f32 {
    let x = x * 0.6;
    (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)
}

const AGX_MIN_EV: f32 = -12.47393;
const AGX_MAX_EV: f32 = 4.026069;

/// AgX's log encoding and contrast polynomial, back to linear by 2.2.
fn agx(x: f32) -> f32 {
    let l = ((x.max(1e-10).log2() - AGX_MIN_EV) / (AGX_MAX_EV - AGX_MIN_EV)).clamp(0.0, 1.0);
    let (x2, x4) = (l * l, l * l * l * l);
    let y = 15.5 * x4 * x2 - 40.14 * x4 * l + 31.96 * x4 - 6.868 * x2 * l + 0.4298 * x2 + 0.1191 * l
        - 0.00232;
    y.max(0.0).powf(2.2)
}

impl ToneMapParams {
    /// The uniforms as the shader reads them, already guarded:
    /// `(exposure gain, threshold, white, q)`, `(toe, pivot, mix, norm)`.
    pub fn uniforms(&self) -> [f32; 8] {
        let gain = fin(self.exposure, 0.0).clamp(-30.0, 30.0).exp2();
        let white = fin(self.white, 1.0).max(1e-3);
        let thr = fin(self.threshold, 0.8).clamp(0.0, white * 0.999);
        let soft = fin(self.softness, 0.5).clamp(0.02, 1.0);
        let toe = fin(self.toe, 0.0).clamp(0.0, 1.0);
        let pivot = fin(self.toe_pivot, 0.18).max(1e-4);
        let mix = fin(self.mix, 1.0);
        let norm = match self.op {
            ToneOp::Hable => 1.0 / hable(white).max(1e-6),
            ToneOp::Aces => 1.0 / aces(white).max(1e-6),
            ToneOp::AgX => 1.0 / agx(white).max(1e-6),
            _ => 1.0,
        };
        [gain, thr, white, 1.0 / soft, toe, pivot, mix, norm]
    }

    /// The scalar curve `F` (exposure, toe, operator) for one positive value.
    pub fn curve(&self, x: f32) -> f32 {
        let [gain, t, w, q, k, p, _, norm] = self.uniforms();
        let mut x = x * gain;
        if x < p {
            let d = 1.0 - x / p;
            x -= k * x * d * d;
        }
        match self.op {
            ToneOp::Knee | ToneOp::Exponential => {
                if x <= t {
                    return x;
                }
                let u = (x - t) / (w - t);
                let s = if self.op == ToneOp::Knee {
                    if u < 1.0 {
                        u / (1.0 + u.powf(q)).powf(1.0 / q)
                    } else {
                        1.0 / (1.0 + u.powf(-q)).powf(1.0 / q)
                    }
                } else {
                    1.0 - (-u).exp()
                };
                t + (w - t) * s
            }
            ToneOp::Reinhard => x * (1.0 + x / (w * w)) / (1.0 + x),
            ToneOp::Hable => hable(x) * norm,
            ToneOp::Aces => aces(x) * norm,
            ToneOp::AgX => agx(x) * norm,
        }
    }

    /// The CPU mirror of the generated code (unmasked), for a working space
    /// whose luminance weights are `luma`.
    pub fn apply(&self, c: [f32; 4], luma: [f32; 3]) -> [f32; 4] {
        let mix = self.uniforms()[6];
        let toned: [f32; 3] = match self.mode {
            ToneMode::PerChannel => {
                std::array::from_fn(|i| if c[i] > 0.0 { self.curve(c[i]) } else { c[i] })
            }
            ToneMode::HuePreserving | ToneMode::Luma => {
                let m = if self.mode == ToneMode::Luma {
                    c[0] * luma[0] + c[1] * luma[1] + c[2] * luma[2]
                } else {
                    c[0].max(c[1]).max(c[2])
                };
                if m > 0.0 {
                    let r = self.curve(m) / m;
                    [c[0] * r, c[1] * r, c[2] * r]
                } else {
                    [c[0], c[1], c[2]]
                }
            }
        };
        let mut out = c;
        for i in 0..3 {
            out[i] = c[i] + (toned[i] - c[i]) * mix;
        }
        out
    }

    /// The WGSL body of the scalar curve: reads `x`, `k0`, `k1`, returns.
    fn curve_wgsl(&self) -> String {
        let op = match self.op {
            ToneOp::Knee => "    if (x <= k0.y) { return x; }
    let u = (x - k0.y) / (k0.z - k0.y);
    let q = k0.w;
    // Two forms of u/(1+u^q)^(1/q), each finite on its own side of 1.
    let s = select(1.0 / pow(1.0 + pow(u, -q), 1.0 / q), u / pow(1.0 + pow(u, q), 1.0 / q), u < 1.0);
    return k0.y + (k0.z - k0.y) * s;"
                .to_string(),
            ToneOp::Exponential => "    if (x <= k0.y) { return x; }
    let u = (x - k0.y) / (k0.z - k0.y);
    return k0.y + (k0.z - k0.y) * (1.0 - exp(-u));"
                .to_string(),
            ToneOp::Reinhard => "    return x * (1.0 + x / (k0.z * k0.z)) / (1.0 + x);".to_string(),
            ToneOp::Hable => "    let n = (x * (0.15 * x + 0.05) + 0.004) / (x * (0.15 * x + 0.5) + 0.06) - 0.0666666667;
    return n * k1.w;"
                .to_string(),
            ToneOp::Aces => "    let a = x * 0.6;
    return (a * (2.51 * a + 0.03)) / (a * (2.43 * a + 0.59) + 0.14) * k1.w;"
                .to_string(),
            ToneOp::AgX => format!(
                "    let l = clamp((log2(max(x, 1e-10)) - ({AGX_MIN_EV:.6})) / {:.6}, 0.0, 1.0);
    let x2 = l * l;
    let x4 = x2 * x2;
    let y = 15.5 * x4 * x2 - 40.14 * x4 * l + 31.96 * x4 - 6.868 * x2 * l + 0.4298 * x2 + 0.1191 * l - 0.00232;
    return pow(max(y, 0.0), 2.2) * k1.w;",
                AGX_MAX_EV - AGX_MIN_EV
            ),
        };
        format!(
            "    var x = x0 * k0.x;
    // The toe: x - k x (1 - x/p)^2 under the pivot, C1 into the identity.
    if (x < k1.y) {{
        let d = 1.0 - x / k1.y;
        x = x - k1.x * x * d * d;
    }}
{op}"
        )
    }
}

impl NodeDef for ToneMapParams {
    const NAME: &'static str = "Tone Map";
    /// Image, then the mask.
    const INPUTS: usize = 2;
    const MASK_INPUT: Option<usize> = Some(1);
    const VEC4_SLOTS: usize = 2;

    /// The operator and the mode pick the code; every number is a uniform.
    fn topo_tag(&self) -> u64 {
        TOPO_BASE ^ ((self.op as u64) << 8) ^ ((self.mode as u64) << 16)
    }

    fn pack(&self, v: &mut [f32]) {
        v[..8].copy_from_slice(&self.uniforms());
    }

    fn emit(&self, e: &mut Emit) -> String {
        let input = e.input(0);
        let slot = e.slot();
        let w = e.working().primaries.luma_weights();
        let curve = format!("tonemap_curve_{}", self.op.slug());
        let name = format!("tonemap_{}_{}", self.op.slug(), self.mode.slug());
        let (curves, fns) = e.color_helpers();
        if !curves.contains_key(&curve) {
            curves.insert(
                curve.clone(),
                format!(
                    "\n// Tone Map's scalar curve ({}): exposure, toe, operator.\nfn {curve}(x0: f32, k0: vec4f, k1: vec4f) -> f32 {{\n{}\n}}\n",
                    self.op.label(),
                    self.curve_wgsl()
                ),
            );
        }
        if !fns.contains_key(&name) {
            let body = match self.mode {
                ToneMode::PerChannel => format!(
                    "    let t = vec3f({curve}(c.r, k0, k1), {curve}(c.g, k0, k1), {curve}(c.b, k0, k1));
    let o = select(c.rgb, t, c.rgb > vec3f(0.0));"
                ),
                ToneMode::HuePreserving | ToneMode::Luma => {
                    let m = if self.mode == ToneMode::Luma {
                        format!("dot(c.rgb, vec3f({:.7}, {:.7}, {:.7}))", w[0], w[1], w[2])
                    } else {
                        "max(c.r, max(c.g, c.b))".to_string()
                    };
                    format!(
                        "    let m = {m};
    let o = select(c.rgb, c.rgb * ({curve}(m, k0, k1) / max(m, 1e-30)), m > 0.0);"
                    )
                }
            };
            fns.insert(
                name.clone(),
                format!(
                    "\n// Tone Map ({}, {}). slot+0 = (gain, threshold, white, q), slot+1 = (toe, pivot, mix, norm).\nfn {name}(c: vec4f, slot: u32) -> vec4f {{\n    let k0 = P.v[slot];\n    let k1 = P.v[slot + 1u];\n{body}\n    return vec4f(mix(c.rgb, o, k1.z), c.a);\n}}\n",
                    self.op.label(),
                    self.mode.label()
                ),
            );
        }
        let v = e.fresh();
        e.line(format_args!("let {v} = {name}({input}, {slot}u);"));
        e.masked(Self::MASK_INPUT, &input, v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LUMA: [f32; 3] = [0.2126, 0.7152, 0.0722];

    fn all() -> impl Iterator<Item = ToneMapParams> {
        ToneOp::ALL.into_iter().flat_map(|op| {
            ToneMode::ALL.into_iter().map(move |mode| ToneMapParams {
                op,
                mode,
                white: op.default_white(),
                ..Default::default()
            })
        })
    }

    #[test]
    fn the_shoulders_leave_everything_under_the_threshold_alone() {
        for op in [ToneOp::Knee, ToneOp::Exponential] {
            let p = ToneMapParams { op, ..Default::default() };
            for x in [0.0, 0.01, 0.18, 0.5, 0.79] {
                assert_eq!(p.curve(x), x, "{op:?} {x}");
            }
        }
    }

    #[test]
    fn the_shoulders_are_smooth_monotone_and_stay_under_white() {
        for op in [ToneOp::Knee, ToneOp::Exponential] {
            for soft in [0.05, 0.5, 1.0] {
                let p = ToneMapParams { op, softness: soft, threshold: 0.5, white: 2.0, ..Default::default() };
                let mut last = 0.0;
                for i in 1..4000 {
                    let x = i as f32 * 0.05;
                    let y = p.curve(x);
                    assert!(y >= last && y <= 2.0, "{op:?} s{soft} x{x} y{y}");
                    last = y;
                }
                // Slope 1 leaving the threshold: no crease.
                let d = (p.curve(0.5 + 1e-3) - p.curve(0.5)) / 1e-3;
                assert!((d - 1.0).abs() < 0.02, "{op:?} s{soft} slope {d}");
            }
        }
    }

    #[test]
    fn a_tiny_softness_is_nearly_a_clip() {
        let p = ToneMapParams { softness: 0.02, threshold: 0.0, white: 1.0, ..Default::default() };
        assert!((p.curve(0.5) - 0.5).abs() < 0.01);
        assert!((p.curve(100.0) - 1.0).abs() < 1e-3);
        assert!(p.curve(1e30).is_finite());
    }

    #[test]
    fn the_tone_maps_land_white_on_one() {
        for op in [ToneOp::Reinhard, ToneOp::Hable, ToneOp::Aces, ToneOp::AgX] {
            let p = ToneMapParams { op, white: op.default_white(), ..Default::default() };
            assert!((p.curve(op.default_white()) - 1.0).abs() < 1e-3, "{op:?} {}", p.curve(op.default_white()));
            assert!(p.curve(0.18) > 0.0 && p.curve(0.18) < 0.5, "{op:?} mid {}", p.curve(0.18));
        }
    }

    #[test]
    fn hue_preserving_keeps_the_ratios_and_the_ceiling() {
        let p = ToneMapParams::default();
        let out = p.apply([8.0, 2.0, 0.5, 1.0], LUMA);
        assert!((out[0] / out[1] - 4.0).abs() < 1e-4, "{out:?}");
        assert!((out[1] / out[2] - 4.0).abs() < 1e-4, "{out:?}");
        assert!(out[0] < 1.0);
        assert_eq!(out[3], 1.0);
    }

    #[test]
    fn per_channel_bleaches_towards_white() {
        let p = ToneMapParams { mode: ToneMode::PerChannel, ..Default::default() };
        let out = p.apply([8.0, 2.0, 0.5, 1.0], LUMA);
        assert!(out[0] / out[1] < 1.2, "{out:?}");
        assert_eq!(out[2], 0.5);
    }

    #[test]
    fn negatives_alpha_and_mix() {
        for p in all() {
            let c = [-0.5, -2.0, 0.0, 0.3];
            assert_eq!(p.apply(c, LUMA), c, "{p:?}");
            let off = ToneMapParams { mix: 0.0, ..p };
            assert_eq!(off.apply([4.0, 2.0, 1.0, 0.5], LUMA), [4.0, 2.0, 1.0, 0.5]);
        }
    }

    #[test]
    fn the_toe_bends_shadows_and_holds_the_pivot() {
        let p = ToneMapParams { toe: 0.5, threshold: 1.0, white: 2.0, ..Default::default() };
        assert!(p.curve(0.05) < 0.05);
        assert!((p.curve(0.18) - 0.18).abs() < 1e-6);
        assert_eq!(p.curve(0.5), 0.5);
    }

    #[test]
    fn exposure_is_stops() {
        let p = ToneMapParams { exposure: 1.0, threshold: 10.0, white: 20.0, ..Default::default() };
        assert!((p.curve(0.25) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn degenerate_numbers_stay_finite() {
        for bad in [f32::NAN, f32::INFINITY, -1.0, 0.0] {
            let p = ToneMapParams { white: bad, threshold: bad, softness: bad, toe_pivot: bad, exposure: bad, ..Default::default() };
            for x in [0.1, 1.0, 1000.0] {
                assert!(p.curve(x).is_finite(), "{bad} {x}");
            }
        }
    }

    #[test]
    fn op_and_mode_are_the_topology() {
        let mut seen = std::collections::HashSet::new();
        for p in all() {
            assert!(seen.insert(p.topo_tag()), "{p:?}");
        }
        let a = ToneMapParams::default();
        let b = ToneMapParams { white: 9.0, exposure: 2.0, ..a };
        assert_eq!(a.topo_tag(), b.topo_tag());
    }
}
