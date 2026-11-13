//! Saturation, done the way a colour corrector does it rather than the way
//! an HSV converter does.
//!
//! The core op everywhere — Resolve's saturation slider, the classic Saturation
//! node, every "saturation" in every grading tool — is one line: lerp the
//! pixel away from (or past) its own luminance.
//!
//! ```text
//! out = y + (rgb - y) * s        y = dot(rgb, luma_weights)
//! ```
//!
//! With matching weights that is exactly luminance-preserving, because
//! `dot(w, y + (c - y)s) = y + (dot(w,c) - y)s = y`. Everything interesting
//! about a saturation control is what surrounds that line:
//!
//! **The domain.** Resolve does not run it on scene-linear values — in a
//! colour-managed timeline it runs in the timeline space, which is normally
//! a log encoding (DaVinci Intermediate). That single fact is most of why
//! Resolve's saturation feels the way it does, and the reason is sharper
//! than "log looks nicer". Write each channel as its ratio to the neutral,
//! `r = c / y`. Then:
//!
//! ```text
//! linear:  r' = 1 + (r - 1) * s      affine
//! log:     r' = r^s                  power law
//! ```
//!
//! Both are exposure-invariant, so neither one favours highlights over
//! midtones — that part is a myth. What differs is the floor. The affine
//! form sends any channel below the neutral NEGATIVE the moment
//! `s > r / (r - 1)`: at s = 2, a channel at a fifth of the neutral lands
//! at -0.6, outside the gamut, and the far side of that is a hue shift when
//! something eventually clips it. The power law cannot produce a negative
//! at all — it just raises the ratio to a power, so a fifth becomes a
//! twenty-fifth and stays a colour. That is what a colourist is buying by
//! grading in log. [`SatDomain`] makes the choice explicit.
//!
//! **Which luminance.** Resolve uses the timeline space's weights; other compositors
//! offer Rec.709 / Average / Maximum. The Y row of the working space's
//! RGB->XYZ matrix is the correct default and the one the rest of this
//! codebase already uses ([`crate::color::Primaries::luma_weights`]) — but
//! matching another tool means matching its (often wrong) weights, so all
//! four are here. See [`SatLuma`].
//!
//! **Weighting by how saturated the pixel already is.** Resolve's Color
//! Boost adds saturation preferentially to pixels that have little, so a
//! sky and a coat gain without the already-loud reds clipping. Vibrance,
//! under a different name. [`SatParams::boost`].
//!
//! **Weighting by how bright the pixel is.** Resolve's Sat-vs-Lum curve,
//! and the per-zone saturation in its HDR palette: hold the shadows down,
//! push the highlights. A three-zone partition of unity over log luminance
//! is the cheap version of the curve. [`SatParams::zones`].
//!
//! **Where the colour lands afterwards.** Push saturation far enough in any
//! wide working space and channels go negative — legal scene-linear values
//! that are outside the gamut and will clip at display or on export.
//! [`SatParams::compress`] pulls them back to the gamut edge instead.
//!
//! The GPU path is generated WGSL, fused into the comp shader like every
//! other pointwise node; [`apply`] is the CPU mirror the tests check it
//! against.

use crate::codegen::Emit;
use crate::color::{Colorspace, Curve, Mat3, Primaries};
use crate::nodes::NodeDef;
use std::collections::BTreeMap;

/// Where a [`crate::graph::NodeKind::Saturation`] does its arithmetic. Structural: the
/// domain IS the shader, so changing it recompiles.
///
/// This is the single most consequential control on the node, and the one
/// no tool exposes. Resolve does not saturate scene-linear values — in a
/// managed timeline it works in the timeline space, which is a log
/// encoding, and that is most of why its saturation slider behaves the way
/// it does. In terms of each channel's ratio to the neutral, `r = c / y`,
/// linear is `r' = 1 + (r - 1) * s` and log is `r' = r^s`: affine versus a
/// power law. Both are exposure-invariant, so neither favours highlights;
/// the difference is that the affine one crosses zero and the power law
/// cannot. See the module doc above.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SatDomain {
    /// Straight on the scene-linear working-space values, which is what
    /// every naive implementation does. Affine on the ratio to neutral, so
    /// a channel well below the neutral goes NEGATIVE — out of gamut, and
    /// a hue shift once something clips it.
    Linear,
    /// ACEScct-encoded, then back: a power law on the ratio to neutral, so
    /// it saturates hard without ever producing a negative. A Resolve
    /// timeline space, in other words, and the default here.
    #[default]
    Log,
    /// Oklab: scale the chroma and leave the lightness alone. Holds
    /// perceived lightness and hue far better than an RGB lerp can, at the
    /// price of not matching any other tool's numbers.
    Perceptual,
}

impl SatDomain {
    pub const ALL: [SatDomain; 3] = [SatDomain::Linear, SatDomain::Log, SatDomain::Perceptual];

    pub fn label(self) -> &'static str {
        match self {
            SatDomain::Linear => "linear",
            SatDomain::Log => "log (ACEScct)",
            SatDomain::Perceptual => "perceptual (Oklab)",
        }
    }

    /// Short tag for generated WGSL function names.
    pub fn slug(self) -> &'static str {
        match self {
            SatDomain::Linear => "lin",
            SatDomain::Log => "log",
            SatDomain::Perceptual => "ok",
        }
    }
}

/// Which luminance a [`crate::graph::NodeKind::Saturation`] holds still. The usual
/// "luminance math" dropdown, plus the one it is missing. Structural.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SatLuma {
    /// The Y row of the WORKING space's RGB->XYZ matrix — the only
    /// definition that is actually correct for the pixels in hand, and what
    /// Resolve uses for its timeline space. AP1's green weight is 0.674,
    /// not Rec.709's 0.715.
    #[default]
    Working,
    /// Rec.709 weights whatever the working space is. Wrong, on purpose:
    /// it is what almost every other implementation hardcodes, so it is how
    /// you match one.
    Rec709,
    /// Equal thirds. The usual "average".
    Average,
    /// The largest channel, which makes this exactly an HSV saturation
    /// scale. The usual "maximum".
    Max,
}

impl SatLuma {
    pub const ALL: [SatLuma; 4] = [
        SatLuma::Working,
        SatLuma::Rec709,
        SatLuma::Average,
        SatLuma::Max,
    ];

    pub fn label(self) -> &'static str {
        match self {
            SatLuma::Working => "working space",
            SatLuma::Rec709 => "Rec.709",
            SatLuma::Average => "average",
            SatLuma::Max => "maximum (HSV)",
        }
    }

    pub fn slug(self) -> &'static str {
        match self {
            SatLuma::Working => "work",
            SatLuma::Rec709 => "r709",
            SatLuma::Average => "avg",
            SatLuma::Max => "max",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
// `serde(default)` on the CONTAINER, not per field: a missing field then
// comes back as `SatParams::default()`'s value rather than the field type's
// zero, and a setup that predates a new control loads as a NEUTRAL node
// instead of a silent monochrome one.
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct SatParams {
    /// Master saturation. 1 = untouched, 0 = monochrome, >1 richer.
    /// Resolve's 0..100 slider with 50 at neutral is this times 50.
    pub sat: f32,
    /// Resolve's Color Boost: extra saturation weighted toward the pixels
    /// that have least, so a sky and a coat come up without the reds
    /// clipping. Vibrance, under the name Resolve gives it. Negative pulls
    /// the pale colours out instead.
    pub boost: f32,
    /// Sat vs Lum: shadow / midtone / highlight multipliers on top of the
    /// master. Resolve's curve, and its HDR palette's per-zone saturation,
    /// reduced to three numbers.
    pub zones: [f32; 3],
    /// Scene-linear luminance the midtone zone is centred on. 0.18 is an
    /// 18% grey card.
    pub pivot: f32,
    /// How many stops each zone reaches either side of the pivot.
    pub width: f32,
    /// Where the arithmetic happens. Structural.
    pub domain: SatDomain,
    /// Which luminance the lerp holds still. Structural.
    pub luma: SatLuma,
    /// Put the original luminance back afterwards. A lerp in the working
    /// space preserves it already; log and Oklab do not, and a saturation
    /// that also changes exposure cannot be used on its own.
    pub preserve: f32,
    /// Pull colours that left the gamut back to its edge rather than
    /// letting the display or the file clip them into a hue shift.
    pub compress: f32,
}

impl Default for SatParams {
    fn default() -> Self {
        Self {
            sat: 1.0,
            boost: 0.0,
            zones: [1.0; 3],
            pivot: 0.18,
            width: 2.5,
            domain: SatDomain::default(),
            luma: SatLuma::default(),
            preserve: 0.0,
            compress: 0.0,
        }
    }
}

/// The luminance weights the *lerp* holds still, or `None` for
/// [`SatLuma::Max`], which holds the largest channel still instead (which
/// is what an HSV saturation does, and is not a dot product).
///
/// Note this is only the axis the saturation is measured against. The zone
/// weighting, the luminance restore and the gamut compression always use
/// the WORKING space's own weights — those are questions about the picture,
/// not about which tool we are imitating.
pub fn lerp_weights(luma: SatLuma, working: Colorspace) -> Option<[f32; 3]> {
    match luma {
        SatLuma::Working => Some(working.primaries.luma_weights()),
        SatLuma::Rec709 => Some(Primaries::Rec709.luma_weights()),
        SatLuma::Average => Some([1.0 / 3.0; 3]),
        SatLuma::Max => None,
    }
}

/// The log encoding [`SatDomain::Log`] grades in.
///
/// ACEScct rather than DaVinci Intermediate: it is the ACES grading log, it
/// is already in [`crate::color`], and its linear toe means negatives round
/// trip instead of being swallowed. The curve is applied to the working
/// space's own channels — no gamut hop — so this is "the working space,
/// log encoded", exactly the shape of a Resolve timeline space.
pub const LOG_CURVE: Curve = Curve::AcesCct;

/// Björn Ottosson's Oklab -> LMS' matrix (the inverse of the front half of
/// the forward transform in [`crate::keyer`], which only ever needed the
/// forward direction).
const OKLAB_TO_LMS: Mat3 = [
    [1.0, 0.396_337_78, 0.215_803_76],
    [1.0, -0.105_561_346, -0.063_854_17],
    [1.0, -0.089_484_18, -1.291_485_5],
];

/// Signed cube (the inverse of the keyer's signed cube root).
#[inline]
fn cube_s(x: f32) -> f32 {
    x * x * x
}

/// Working-space linear -> Oklab, through the working space's own RGB->LMS
/// matrix (Oklab is defined from Rec.709 primaries, so comp pixels have to
/// reach it through their gamut first — same rule as the keyer).
fn oklab_from_linear(c: [f32; 3], working: Colorspace) -> [f32; 3] {
    crate::keyer::oklab_from_linear_in(c, crate::keyer::oklab_matrix(working))
}

/// Oklab -> working-space linear. Public because [`crate::colorwarp`] needs
/// the same inverse on the CPU that it needs in the shader.
pub fn linear_from_oklab(lab: [f32; 3], working: Colorspace) -> [f32; 3] {
    let lms_ = crate::color::mat3_apply(OKLAB_TO_LMS, lab);
    let lms = [cube_s(lms_[0]), cube_s(lms_[1]), cube_s(lms_[2])];
    let inv = crate::color::mat3_inverse(crate::keyer::oklab_matrix(working));
    crate::color::mat3_apply(inv, lms)
}

/// Emit `ok_to_linear` (Oklab -> working-space linear) into the shader's
/// helper set once, and return its name.
///
/// `keyer.rs` already emits the FORWARD direction, `oklab_from_linear`, into
/// every fused shader — it never needed the way back. Two nodes now do (this
/// one and [`crate::colorwarp`]), so the inverse lives here, emitted once and
/// shared, rather than copied into each and left to drift apart.
pub fn oklab_inverse_wgsl(
    working: Colorspace,
    helpers: &mut BTreeMap<String, String>,
) -> &'static str {
    const NAME: &str = "ok_to_linear";
    if !helpers.contains_key(NAME) {
        let inv = crate::color::mat3_inverse(crate::keyer::oklab_matrix(working));
        helpers.insert(
            NAME.to_string(),
            format!(
                "// Oklab -> working-space linear, the inverse of the keyer's
// `oklab_from_linear`. Cubing rather than a cube root, so negatives (a
// scene-linear plate outside the gamut) stay finite and signed.
fn ok_to_linear(lab: vec3f) -> vec3f {{
    let q = vec3f(
        lab.x + 0.3963377774 * lab.y + 0.2158037573 * lab.z,
        lab.x - 0.1055613458 * lab.y - 0.0638541728 * lab.z,
        lab.x - 0.0894841775 * lab.y - 1.2914855480 * lab.z);
    let lms = q * q * q;
    return vec3f(
        {:.10} * lms.x + {:.10} * lms.y + {:.10} * lms.z,
        {:.10} * lms.x + {:.10} * lms.y + {:.10} * lms.z,
        {:.10} * lms.x + {:.10} * lms.y + {:.10} * lms.z);
}}
",
                inv[0][0],
                inv[0][1],
                inv[0][2],
                inv[1][0],
                inv[1][1],
                inv[1][2],
                inv[2][0],
                inv[2][1],
                inv[2][2],
            ),
        );
    }
    NAME
}

/// How much extra saturation [`SatParams::boost`] gives a pixel that is
/// already `cur` saturated (0 = neutral, 1 = fully). Squared falloff: the
/// already-loud colours are left alone, which is the whole point of the
/// control.
#[inline]
fn boost_weight(cur: f32) -> f32 {
    let d = 1.0 - cur.clamp(0.0, 1.0);
    d * d
}

/// The three zone weights at scene-linear luminance `y`: a smooth partition
/// of unity over `width` stops either side of `pivot`.
#[inline]
fn zone_weights(y: f32, pivot: f32, width: f32) -> [f32; 3] {
    let t = ((y.max(1e-6) / pivot.max(1e-6)).log2() / width.max(0.25)).clamp(-1.0, 1.0);
    let smooth = |e0: f32, e1: f32, x: f32| {
        let u = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
        u * u * (3.0 - 2.0 * u)
    };
    let lo = 1.0 - smooth(-1.0, 0.0, t);
    let hi = smooth(0.0, 1.0, t);
    [lo, 1.0 - lo - hi, hi]
}

/// The CPU mirror of the generated shader, value for value. Tests render a
/// plate through the fused pass and check it against this; the app never
/// calls it on pixels.
pub fn apply(p: &SatParams, working: Colorspace, rgb: [f32; 3]) -> [f32; 3] {
    let w = working.primaries.luma_weights();
    let dot = |c: [f32; 3], w: [f32; 3]| c[0] * w[0] + c[1] * w[1] + c[2] * w[2];
    let y0 = dot(rgb, w);

    let z = zone_weights(y0, p.pivot, p.width);
    let zone = z[0] * p.zones[0] + z[1] * p.zones[1] + z[2] * p.zones[2];

    let mut out = match p.domain {
        SatDomain::Perceptual => {
            let lab = oklab_from_linear(rgb, working);
            let chroma = (lab[1] * lab[1] + lab[2] * lab[2]).sqrt();
            let cur = (chroma / lab[0].abs().max(1e-4)).clamp(0.0, 1.0);
            let s = (p.sat * zone * (1.0 + p.boost * boost_weight(cur))).max(0.0);
            linear_from_oklab([lab[0], lab[1] * s, lab[2] * s], working)
        }
        _ => {
            let enc = |c: [f32; 3]| match p.domain {
                SatDomain::Log => [
                    LOG_CURVE.from_linear(c[0]),
                    LOG_CURVE.from_linear(c[1]),
                    LOG_CURVE.from_linear(c[2]),
                ],
                _ => c,
            };
            let dec = |c: [f32; 3]| match p.domain {
                SatDomain::Log => [
                    LOG_CURVE.to_linear(c[0]),
                    LOG_CURVE.to_linear(c[1]),
                    LOG_CURVE.to_linear(c[2]),
                ],
                _ => c,
            };
            let v = enc(rgb);
            let mx = v[0].max(v[1]).max(v[2]);
            let mn = v[0].min(v[1]).min(v[2]);
            let cur = if mx > 1e-6 { (mx - mn) / mx } else { 0.0 };
            let s = (p.sat * zone * (1.0 + p.boost * boost_weight(cur))).max(0.0);
            let y = match lerp_weights(p.luma, working) {
                Some(lw) => dot(v, lw),
                None => mx,
            };
            dec([y + (v[0] - y) * s, y + (v[1] - y) * s, y + (v[2] - y) * s])
        }
    };

    // Luminance restore. A lerp in the working space already preserves it
    // exactly; log and Oklab do not, and a saturation that also changes
    // exposure is a saturation you cannot use on its own.
    let keep = p.preserve.clamp(0.0, 1.0);
    if keep > 0.0 {
        let y1 = dot(out, w);
        if y1 > 1e-6 && y0 > 0.0 {
            let g = 1.0 + (y0 / y1 - 1.0) * keep;
            out = [out[0] * g, out[1] * g, out[2] * g];
        }
    }

    // Gamut. A negative channel is a colour outside the working gamut; pull
    // it back to the edge along the line to its own luminance rather than
    // letting the display or the file format clip it into a hue shift.
    let comp = p.compress.clamp(0.0, 1.0);
    if comp > 0.0 {
        let m = out[0].min(out[1]).min(out[2]);
        if m < 0.0 {
            let y = dot(out, w);
            let a = (y / (y - m).max(1e-6)).clamp(0.0, 1.0);
            let k = 1.0 - comp * (1.0 - a);
            out = [
                y + (out[0] - y) * k,
                y + (out[1] - y) * k,
                y + (out[2] - y) * k,
            ];
        }
    }
    out
}

/// Name of the generated WGSL entry point for `p`, with its definition (and
/// any helpers it needs) added to the shader's helper sets.
///
/// `helpers` is codegen's `color_curves` map, emitted STRICTLY BEFORE
/// `fns` (= `color_fns`) — WGSL wants a function declared before it is
/// used, and one alphabetically-sorted map would not guarantee that. The
/// domain conversions go in the first, the entry point in the second.
pub fn wgsl(
    p: &SatParams,
    working: Colorspace,
    helpers: &mut BTreeMap<String, String>,
    fns: &mut BTreeMap<String, String>,
) -> String {
    // Only the STRUCTURAL params pick the code — everything else is a
    // uniform, so moving a slider never recompiles.
    let name = format!("sat_op_{}_{}", p.domain.slug(), p.luma.slug());
    if fns.contains_key(&name) {
        return name;
    }
    let w = working.primaries.luma_weights();
    let wv = format!("vec3f({:.7}, {:.7}, {:.7})", w[0], w[1], w[2]);

    // Domain conversions. `Transform` already knows how to write a curve
    // pair; with the same primaries on both sides its matrix is identity,
    // so this is exactly "encode the working space to log and back".
    if p.domain == SatDomain::Log {
        let lin = Colorspace::new(working.primaries, Curve::Linear);
        let log = Colorspace::new(working.primaries, LOG_CURVE);
        for (n, t) in [
            ("sat_lin_to_log", crate::color::Transform::new(lin, log)),
            ("sat_log_to_lin", crate::color::Transform::new(log, lin)),
        ] {
            if !helpers.contains_key(n) {
                let def = t.wgsl(n, helpers);
                helpers.insert(n.to_string(), def);
            }
        }
    }
    if p.domain == SatDomain::Perceptual {
        oklab_inverse_wgsl(working, helpers);
    }

    // The saturating step, which is the only part the domain changes.
    let core = match p.domain {
        SatDomain::Perceptual => format!(
            "    // Oklab: scale the chroma and leave L alone. Lightness and
    // hue hold by construction, which is what the RGB lerp only
    // approximates.
    let lab = oklab_from_linear(lin);
    let chroma = length(lab.yz);
    let cur = clamp(chroma / max(abs(lab.x), 1e-4), 0.0, 1.0);
    let d = 1.0 - cur;
    let s = max(k.x * zone * (1.0 + k.y * d * d), 0.0);
    var res = ok_to_linear(vec3f(lab.x, lab.yz * s));"
        ),
        _ => {
            let (to, from) = match p.domain {
                SatDomain::Log => ("sat_lin_to_log(lin)", "sat_log_to_lin(v2)"),
                _ => ("lin", "v2"),
            };
            // The usual "maximum" luminance math holds the largest channel
            // still, so the result is exactly an HSV saturation scale.
            let y = match lerp_weights(p.luma, working) {
                Some(lw) => format!("dot(v, vec3f({:.7}, {:.7}, {:.7}))", lw[0], lw[1], lw[2]),
                None => "mx".to_string(),
            };
            format!(
                "    let v = {to};
    let mx = max(v.r, max(v.g, v.b));
    let mn = min(v.r, min(v.g, v.b));
    // How saturated it already is, so Color Boost can spare the loud ones.
    let cur = select(0.0, (mx - mn) / max(mx, 1e-6), (mx > 1e-6));
    let d = 1.0 - clamp(cur, 0.0, 1.0);
    let s = max(k.x * zone * (1.0 + k.y * d * d), 0.0);
    let y = {y};
    let v2 = vec3f(y) + (v - vec3f(y)) * s;
    var res = {from};"
            )
        }
    };

    let def = format!(
        r#"
// Saturation: a lerp between the pixel and its luminance, weighted by how
// saturated and how bright the pixel already is, then put back where it
// belongs. slot+0 = (sat, boost, preserve, compress), slot+1 = (shadow,
// mid, high, pivot), slot+2.x = zone width in stops.
fn {name}(c: vec4f, slot: u32) -> vec4f {{
    let k = P.v[slot];
    let z = P.v[slot + 1u];
    let wd = max(P.v[slot + 2u].x, 0.25);
    let lin = c.rgb;
    // Zone weighting rides the SCENE-LINEAR luminance whatever domain the
    // saturation itself happens in: a highlight is a highlight.
    let y0 = dot(lin, {wv});
    let t = clamp(log2(max(y0, 1e-6) / max(z.w, 1e-6)) / wd, -1.0, 1.0);
    let lo = 1.0 - smoothstep(-1.0, 0.0, t);
    let hi = smoothstep(0.0, 1.0, t);
    let zone = lo * z.x + (1.0 - lo - hi) * z.y + hi * z.z;
{core}
    // Put the luminance back. The working-space lerp preserves it already;
    // log and Oklab do not.
    let y1 = dot(res, {wv});
    if (k.z > 0.0 && y1 > 1e-6 && y0 > 0.0) {{
        res = res * (1.0 + (y0 / y1 - 1.0) * clamp(k.z, 0.0, 1.0));
    }}
    // A negative channel is a colour outside the working gamut. Pull it to
    // the gamut edge instead of letting the display clip it into a hue
    // shift.
    let m = min(res.r, min(res.g, res.b));
    if (k.w > 0.0 && m < 0.0) {{
        let yc = dot(res, {wv});
        let a = clamp(yc / max(yc - m, 1e-6), 0.0, 1.0);
        let g = 1.0 - clamp(k.w, 0.0, 1.0) * (1.0 - a);
        res = vec3f(yc) + (res - vec3f(yc)) * g;
    }}
    return vec4f(res, c.a);
}}
"#
    );
    fns.insert(name.clone(), def);
    name
}

// ---------------------------------------------------------------------------
// What the graph, codegen and the uniform packer need to know. `NodeKind`
// dispatches here from one-line arms; this is the whole of the node's
// engine-side contract.
// ---------------------------------------------------------------------------

/// Topology tags are one flat u64 space shared by every kind, so each
/// node owns a base that no other node's span can reach. 0xC00 is clear:
/// LumaKey owns 0xB00 and nothing sits above it.
const TOPO_BASE: u64 = 0xC00;

impl NodeDef for SatParams {
    const NAME: &'static str = "Saturation";
    /// Image, then the mask.
    const INPUTS: usize = 2;
    const MASK_INPUT: Option<usize> = Some(1);
    /// (sat, boost, preserve, compress), (shadow, mid, high, pivot), then
    /// the zone width. Domain and luma basis are in the code, not the
    /// uniform.
    const VEC4_SLOTS: usize = 3;

    /// Domain and luma basis pick the generated code; everything else on
    /// the node is a uniform, so only those two recompile.
    fn topo_tag(&self) -> u64 {
        TOPO_BASE + (self.domain as u64) * 16 + self.luma as u64
    }

    fn pack(&self, v: &mut [f32]) {
        v[0] = self.sat;
        v[1] = self.boost;
        v[2] = self.preserve;
        v[3] = self.compress;
        v[4..7].copy_from_slice(&self.zones);
        v[7] = self.pivot;
        v[8] = self.width;
    }

    fn emit(&self, e: &mut Emit) -> String {
        let input = e.input(0);
        let slot = e.slot();
        let working = e.working();
        // The domain conversions land in the curve helpers and the entry
        // point among the functions that call them, which is what
        // guarantees the helpers are declared before the function.
        let (curves, fns) = e.color_helpers();
        let f = wgsl(self, working, curves, fns);
        let v = e.fresh();
        e.line(format_args!("let {v} = {f}({input}, {slot}u);"));
        // Masked: crossfade against the untouched input, the same way a
        // masked Grade does.
        e.masked(Self::MASK_INPUT, &input, v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p() -> SatParams {
        SatParams::default()
    }

    #[test]
    fn the_default_is_an_identity() {
        for domain in SatDomain::ALL {
            let mut s = p();
            s.domain = domain;
            for c in [[0.18, 0.18, 0.18], [0.6, 0.2, 0.05], [4.0, 1.0, 0.2]] {
                let out = apply(&s, Colorspace::ACESCG, c);
                for i in 0..3 {
                    assert!(
                        (out[i] - c[i]).abs() < 1e-4,
                        "{domain:?} moved {c:?} to {out:?}"
                    );
                }
            }
        }
    }

    /// In the LINEAR domain the grey a fully desaturated pixel lands on is
    /// its luminance, exactly. (Not in log — there it lands on the
    /// luminance of the LOG values, which is a different, darker grey. That
    /// is not a bug, it is what grading in another space means, and the
    /// second half of this test pins it so nobody "fixes" it.)
    #[test]
    fn zero_saturation_is_the_luminance_it_was_asked_to_hold() {
        let working = Colorspace::ACESCG;
        let w = working.primaries.luma_weights();
        let c = [0.6, 0.2, 0.05];
        let y = c[0] * w[0] + c[1] * w[1] + c[2] * w[2];

        let mut s = p();
        s.sat = 0.0;
        s.domain = SatDomain::Linear;
        let out = apply(&s, working, c);
        assert!((out[0] - out[1]).abs() < 1e-5 && (out[1] - out[2]).abs() < 1e-5);
        assert!((out[0] - y).abs() < 1e-5, "{out:?} vs luma {y}");

        s.domain = SatDomain::Log;
        let log = apply(&s, working, c);
        assert!(
            (log[0] - log[1]).abs() < 1e-5 && (log[1] - log[2]).abs() < 1e-5,
            "not neutral: {log:?}"
        );
        assert!(
            log[0] < y * 0.95,
            "log grey {} should sit below {y}",
            log[0]
        );
        // ...and `preserve` is exactly the control that reconciles them.
        s.preserve = 1.0;
        let held = apply(&s, working, c);
        assert!((held[0] - y).abs() < 1e-4, "{held:?} vs luma {y}");
    }

    /// The reason the luma basis is a control at all: on a saturated colour
    /// the four choices land on genuinely different greys.
    #[test]
    fn the_luma_basis_changes_where_a_desaturated_pixel_lands() {
        let mut greys = Vec::new();
        for luma in SatLuma::ALL {
            let mut s = p();
            s.sat = 0.0;
            s.luma = luma;
            greys.push(apply(&s, Colorspace::ACESCG, [0.6, 0.2, 0.05])[0]);
        }
        for i in 0..greys.len() {
            for j in i + 1..greys.len() {
                assert!(
                    (greys[i] - greys[j]).abs() > 1e-3,
                    "{:?} and {:?} agree: {greys:?}",
                    SatLuma::ALL[i],
                    SatLuma::ALL[j]
                );
            }
        }
        // Maximum holds the largest channel, so it lands on the red.
        assert!((greys[3] - 0.6).abs() < 1e-5, "{greys:?}");
    }

    /// The actual point of grading in log, and the thing that is easy to
    /// get backwards: it is NOT that log moves the pixel less. On a ratio
    /// basis log moves it a great deal more — it raises the ratio to a
    /// power. What it cannot do is produce a negative, because a power law
    /// has no zero crossing, while the affine linear form walks straight
    /// through one.
    ///
    /// (An earlier version of this test asserted that log produced the
    /// SMALLER channel spread. It does not: on [4, 1, 0.2] at s = 2 linear
    /// spreads to 7.6 and log to 11.9. Both are exposure-invariant; the
    /// floor is the whole difference.)
    #[test]
    fn linear_leaves_the_gamut_where_log_cannot() {
        let hot = [4.0, 1.0, 0.2];
        let out = |domain| {
            let mut s = p();
            s.sat = 2.0;
            s.domain = domain;
            apply(&s, Colorspace::ACESCG, hot)
        };
        let lin = out(SatDomain::Linear);
        let log = out(SatDomain::Log);
        assert!(
            lin.iter().any(|v| *v < 0.0),
            "linear should have gone negative: {lin:?}"
        );
        assert!(
            log.iter().all(|v| *v > 0.0),
            "log must not go negative: {log:?}"
        );
        // Both really did saturate it, so this is not passing on a no-op.
        // Measured as channel SPREAD, not as a ratio: once linear has put a
        // channel below zero a ratio says nothing at all, which is the same
        // fact this test exists to record.
        let spread = |o: [f32; 3]| o[0].max(o[1]).max(o[2]) - o[0].min(o[1]).min(o[2]);
        let was = spread(hot);
        for (name, o) in [("linear", lin), ("log", log)] {
            assert!(
                spread(o) > was,
                "{name} did nothing: {} vs {was}",
                spread(o)
            );
        }
        // And log really is the power law: the ratio between two channels
        // comes out raised to s, which is the claim in one line.
        let want = (hot[0] / hot[2]).powf(2.0);
        assert!(
            ((log[0] / log[2]) / want - 1.0).abs() < 0.01,
            "ratio {} is not {want}",
            log[0] / log[2]
        );
    }

    /// Color Boost is the whole reason it is not just a second saturation
    /// slider: the near-neutral pixel gains, the loud one barely moves.
    #[test]
    fn boost_favours_the_less_saturated_pixel() {
        let working = Colorspace::ACESCG;
        let mut s = p();
        s.boost = 1.0;
        // LINEAR, so the metric is clean: the lerp scales the channel
        // spread by exactly the effective saturation, and the ratio below
        // IS that number. In log the spread is measured after a power law
        // and the comparison stops meaning anything.
        s.domain = SatDomain::Linear;
        let chroma = |c: [f32; 3], s: &SatParams| {
            let o = apply(s, working, c);
            (o[0].max(o[1]).max(o[2]) - o[0].min(o[1]).min(o[2]))
                / (c[0].max(c[1]).max(c[2]) - c[0].min(c[1]).min(c[2]))
        };
        let pale = chroma([0.30, 0.27, 0.25], &s);
        let loud = chroma([0.60, 0.05, 0.02], &s);
        assert!(pale > loud * 1.5, "pale x{pale}, loud x{loud}");
        assert!(pale > 1.5, "boost did nothing: x{pale}");
    }

    #[test]
    fn zones_reach_only_their_own_end_of_the_range() {
        let working = Colorspace::ACESCG;
        let hue = |y: f32| [y * 1.6, y * 0.9, y * 0.5];
        let mut s = p();
        s.zones = [0.0, 1.0, 1.0]; // kill shadow saturation only
        let dark = apply(&s, working, hue(0.01));
        let mid = apply(&s, working, hue(0.18));
        assert!(
            (dark[0] - dark[2]).abs() < 1e-4,
            "shadow kept its colour: {dark:?}"
        );
        let m = hue(0.18);
        assert!(
            (mid[0] - m[0]).abs() < 1e-4,
            "midtone was touched: {mid:?} vs {m:?}"
        );
    }

    #[test]
    fn preserve_restores_the_luminance_log_lost() {
        let working = Colorspace::ACESCG;
        let w = working.primaries.luma_weights();
        let c = [0.6, 0.2, 0.05];
        let y0 = c[0] * w[0] + c[1] * w[1] + c[2] * w[2];
        let mut s = p();
        s.domain = SatDomain::Log;
        s.sat = 2.0;
        let loose = apply(&s, working, c);
        let yl = loose[0] * w[0] + loose[1] * w[1] + loose[2] * w[2];
        assert!((yl - y0).abs() > 1e-3, "log was already luma-preserving?");
        s.preserve = 1.0;
        let held = apply(&s, working, c);
        let yh = held[0] * w[0] + held[1] * w[1] + held[2] * w[2];
        assert!((yh - y0).abs() < 1e-4, "{yh} vs {y0}");
    }

    #[test]
    fn compress_pulls_a_negative_channel_back_to_zero() {
        let working = Colorspace::ACESCG;
        let mut s = p();
        s.sat = 3.0;
        // Linear: the domain that actually produces out-of-gamut values,
        // and by a wide margin rather than the hair log manages through
        // ACEScct's linear toe.
        s.domain = SatDomain::Linear;
        let loose = apply(&s, working, [0.6, 0.2, 0.05]);
        assert!(
            loose.iter().any(|v| *v < -1e-3),
            "nothing went negative: {loose:?}"
        );
        s.compress = 1.0;
        let held = apply(&s, working, [0.6, 0.2, 0.05]);
        assert!(
            held.iter().all(|v| *v > -1e-4),
            "still out of gamut: {held:?}"
        );
        // Compression is a chroma pull, not a clamp: the hue order survives.
        assert!(held[0] > held[1] && held[1] > held[2], "{held:?}");
    }

    #[test]
    fn oklab_round_trips() {
        for working in [Colorspace::ACESCG, Colorspace::LINEAR_REC709] {
            for c in [[0.18; 3], [0.6, 0.2, 0.05], [3.0, 1.0, 0.02]] {
                let back = linear_from_oklab(oklab_from_linear(c, working), working);
                for i in 0..3 {
                    assert!(
                        (back[i] - c[i]).abs() < 1e-4,
                        "{c:?} -> {back:?} in {working:?}"
                    );
                }
            }
        }
    }

    /// The signed cube root has to survive the negatives a wide working
    /// space really produces, or a saturated plate turns into NaN.
    #[test]
    fn negatives_stay_finite() {
        for domain in SatDomain::ALL {
            let mut s = p();
            s.domain = domain;
            s.sat = 2.5;
            let out = apply(&s, Colorspace::ACESCG, [0.4, -0.02, -0.05]);
            assert!(out.iter().all(|v| v.is_finite()), "{domain:?} -> {out:?}");
        }
    }
}
