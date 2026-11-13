//! Colour keyer: soft isolation of arbitrary colour ranges, for grading.
//!
//! This is NOT a green-screen keyer. It answers "give me *that* colour — the
//! skin, the sky, the foliage — with an edge I can direct", which is the job
//! Flame's 3D Keyer and Colour Warper and Resolve's 3D qualifier do. The
//! selection is a set of soft **ellipsoids in Oklab**, unioned; the output is
//! a mask that feeds any node's mask input (Grade above all).
//!
//! Why Oklab rather than the HSL/YUV those tools qualify in:
//!
//! - Distances are perceptual, so ONE softness knob means the same thing on
//!   a face and on a sky. In HSL it does not — hue degrees are worth wildly
//!   different amounts at different saturations.
//! - Hue is measured as a distance in the a/b plane, not as an angle, so it
//!   automatically stops mattering as a pixel desaturates. That is the single
//!   biggest source of speckle in an HSL qualifier: near-neutral pixels have
//!   a hue, it is noise, and the qualifier believes it.
//! - It is defined by a cube root, not a transfer function with a domain, so
//!   scene-linear values above 1.0 (and small negatives out of a camera
//!   gamut) stay meaningful instead of clipping. The comp is scene-linear;
//!   the keyer has to be too.
//!
//! Each blob is an ellipsoid whose axes are the three things an artist
//! actually names — **lightness**, **saturation** and **hue** — measured on a
//! frame aligned to the blob's own hue: `L` up, `u` radially out from the
//! neutral axis (saturation), `v` tangential (hue). A near-neutral blob
//! degenerates gracefully: `u`/`v` are then just two orthogonal axes and the
//! selection is a ball around the neutral point, which is what "select the
//! greys at this brightness" should mean.
//!
//! Blobs union as `1 - Π(1 - m_i)`: order independent, associative, and
//! creaseless — a `max()` union leaves a visible ridge where two selections
//! meet, and that ridge shows up in the grade. Negative-weight blobs subtract
//! by the same formula, so "this, but not that" is one gesture.
//!
//! Everything here is pointwise: the node fuses into the same shader as its
//! neighbours, costs a handful of ALU, and never splits the graph.

/// Most blobs one node can hold. The uniform span is `2 + 2 * blobs`, and the
/// whole comp shares a 4096-byte block — this bounds one node at 18 slots.
pub const MAX_BLOBS: usize = 8;

/// Smallest tolerance a fit will produce, in Oklab units, per axis. A click
/// on a flat area has no variance to measure; these are what it gets instead.
pub const TOL_FLOOR: [f32; 3] = [0.06, 0.02, 0.02];

/// One soft ellipsoid in Oklab.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct KeyBlob {
    /// Centre, Oklab (L, a, b).
    pub center: [f32; 3],
    /// Core radii on the hue-aligned frame: lightness, saturation (radial),
    /// hue (tangential). Inside this ellipsoid the matte is solid 1.
    pub tol: [f32; 3],
    /// Falloff width as a MULTIPLE of `tol`, so softening keeps the shape:
    /// the matte reaches 0 on the ellipsoid scaled by `1 + soft`. 0 is a hard
    /// edge; 1 (the default) is Flame's "softness range equal to tolerance".
    pub soft: f32,
    /// Signed strength. Positive blobs union into the matte, negative ones
    /// carve out of it; magnitude is how much.
    pub weight: f32,
}

impl Default for KeyBlob {
    fn default() -> Self {
        // Mid-grey, generous: a fresh keyer selects SOMETHING (a black matte
        // out of the box reads as a broken node), and the first sample
        // replaces this anyway.
        Self {
            center: [0.565, 0.0, 0.0],
            tol: [0.25, 0.06, 0.06],
            soft: 1.0,
            weight: 1.0,
        }
    }
}

impl KeyBlob {
    /// Chroma and hue (radians) of the centre — the polar reading the UI
    /// shows and the frame the tolerances are measured on.
    pub fn chroma_hue(&self) -> (f32, f32) {
        let (a, b) = (self.center[1], self.center[2]);
        (a.hypot(b), b.atan2(a))
    }

    /// The hue-aligned frame: `u` points away from neutral at the blob's hue
    /// (saturation axis), `v` is perpendicular (hue axis). Neutral centres
    /// have no hue to align to and fall back to the a/b axes, which makes the
    /// ellipsoid a ball when `tol[1] == tol[2]`.
    pub fn frame(&self) -> ([f32; 2], [f32; 2]) {
        let (c, _) = self.chroma_hue();
        if c < 1e-5 {
            ([1.0, 0.0], [0.0, 1.0])
        } else {
            let u = [self.center[1] / c, self.center[2] / c];
            (u, [-u[1], u[0]])
        }
    }

    /// The hue tolerance read as an angle at this blob's own chroma — the
    /// number is only meaningful next to a chroma, which is why it is derived
    /// for display instead of stored.
    pub fn hue_degrees(&self) -> f32 {
        let (c, _) = self.chroma_hue();
        if c < 1e-4 {
            180.0
        } else {
            (self.tol[2] / c).atan().to_degrees().min(180.0)
        }
    }
}

/// How the node presents itself WHEN IT IS THE VIEW TARGET. Downstream
/// consumers are unaffected — the app only applies this to the engine graph
/// for the node being viewed (the same transient trick `GraphBake.baked`
/// uses), so a preview can never leak into a render.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum KeyerView {
    /// Show the source image with the matte in alpha (press A for the matte).
    /// Sampling gestures need this: the pixels on screen are the pixels the
    /// keyer is judging.
    #[default]
    Source,
    /// Show the matte itself, exactly as downstream sees it.
    Matte,
}

impl KeyerView {
    pub const ALL: [KeyerView; 2] = [KeyerView::Source, KeyerView::Matte];
    pub fn name(self) -> &'static str {
        match self {
            KeyerView::Source => "source",
            KeyerView::Matte => "matte",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct KeyerParams {
    pub blobs: Vec<KeyBlob>,
    /// Shear of the saturation axis by lightness: `dC -= tilt * dL`. Real
    /// subjects are not upright ellipsoids — skin loses chroma as it goes
    /// into highlight, foliage gains it — and one knob leans the selection
    /// along that line instead of forcing a fatter, sloppier one.
    pub tilt: f32,
    /// Matte levels, applied after the union: `(m - lo) / (hi - lo)`.
    pub lo: f32,
    pub hi: f32,
    /// Edge shaping on the finished matte, `m^gamma`. <1 fattens the soft
    /// edge, >1 tightens it, without touching what is selected.
    pub gamma: f32,
    pub invert: bool,
    /// Preview mode, honoured only while this node is the view target.
    #[cfg_attr(feature = "serde", serde(default))]
    pub view: KeyerView,
    /// Transient: set by the app when this node IS the view target. Never
    /// authored, never serialized — see `KeyerView`.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub previewing: bool,
}

impl Default for KeyerParams {
    fn default() -> Self {
        Self {
            blobs: vec![KeyBlob::default()],
            tilt: 0.0,
            lo: 0.0,
            hi: 1.0,
            gamma: 1.0,
            invert: false,
            view: KeyerView::default(),
            previewing: false,
        }
    }
}

/// Signed cube root: scene-linear plates carry small negatives out of camera
/// gamuts, and `pow()` on those is NaN. Mirrors `cbrt_s` in the WGSL.
#[inline]
fn cbrt_s(x: f32) -> f32 {
    if x < 0.0 {
        -(-x).powf(1.0 / 3.0)
    } else {
        x.powf(1.0 / 3.0)
    }
}

/// Björn Ottosson's linear-Rec.709 -> LMS matrix, the front half of Oklab.
pub const REC709_TO_LMS: crate::color::Mat3 = [
    [0.412_221_47, 0.536_332_54, 0.051_445_995],
    [0.211_903_5, 0.680_699_5, 0.107_396_96],
    [0.088_302_46, 0.281_718_84, 0.629_978_7],
];

/// The RGB->LMS matrix for pixels in `working`.
///
/// Oklab is DEFINED from Rec.709/D65 primaries, so comp pixels have to reach
/// it through their own gamut first. Without this the keyer's "hue" and
/// "saturation" axes are measured on the wrong frame, and a selection made
/// on an ACEScg plate lands somewhere other than where the artist clicked.
pub fn oklab_matrix(working: crate::color::Colorspace) -> crate::color::Mat3 {
    crate::color::mat3_mul(
        REC709_TO_LMS,
        crate::color::primaries_matrix(working.primaries, crate::color::Primaries::Rec709),
    )
}

/// Linear Rec.709/sRGB primaries -> Oklab. (Björn Ottosson's matrices.)
///
/// For UI colours, which ARE Rec.709. Comp pixels go through
/// [`oklab_from_linear_in`] with the working space's matrix.
pub fn oklab_from_linear(c: [f32; 3]) -> [f32; 3] {
    oklab_from_linear_in(c, REC709_TO_LMS)
}

/// Linear RGB -> Oklab through an explicit RGB->LMS matrix (see
/// [`oklab_matrix`]).
pub fn oklab_from_linear_in(c: [f32; 3], m: crate::color::Mat3) -> [f32; 3] {
    let l = m[0][0] * c[0] + m[0][1] * c[1] + m[0][2] * c[2];
    let m_lms = m[1][0] * c[0] + m[1][1] * c[1] + m[1][2] * c[2];
    let s = m[2][0] * c[0] + m[2][1] * c[1] + m[2][2] * c[2];
    let (l_, m_, s_) = (cbrt_s(l), cbrt_s(m_lms), cbrt_s(s));
    [
        0.210_454_26 * l_ + 0.793_617_8 * m_ - 0.004_072_047 * s_,
        1.977_998_5 * l_ - 2.428_592_2 * m_ + 0.450_593_7 * s_,
        0.025_904_037 * l_ + 0.782_771_77 * m_ - 0.808_675_77 * s_,
    ]
}

/// Oklab -> linear Rec.709/sRGB primaries. Used by the UI to paint the
/// colour-space plots; the shader never needs it.
pub fn linear_from_oklab(lab: [f32; 3]) -> [f32; 3] {
    let l_ = lab[0] + 0.396_337_78 * lab[1] + 0.215_803_76 * lab[2];
    let m_ = lab[0] - 0.105_561_346 * lab[1] - 0.063_854_17 * lab[2];
    let s_ = lab[0] - 0.089_484_18 * lab[1] - 1.291_485_5 * lab[2];
    let (l, m, s) = (l_ * l_ * l_, m_ * m_ * m_, s_ * s_ * s_);
    [
        4.076_741_7 * l - 3.307_711_6 * m + 0.230_969_94 * s,
        -1.268_438 * l + 2.609_757_4 * m - 0.341_319_38 * s,
        -0.004_196_086 * l - 0.703_418_6 * m + 1.707_614_7 * s,
    ]
}

/// One blob's coverage of an Oklab point, before the union. The CPU mirror of
/// `keyer_blob` in the WGSL — the plots shade the selection with this, and a
/// headless test holds the shader to it.
pub fn blob_coverage(b: &KeyBlob, lab: [f32; 3]) -> f32 {
    blob_coverage_tilted(b, lab, 0.0)
}

/// `blob_coverage` with the node-wide lightness/saturation shear applied.
/// The shear belongs in the METRIC, not the centre: leaning the measurement
/// tips the ellipsoid, moving the centre would just relocate it.
pub fn blob_coverage_tilted(b: &KeyBlob, lab: [f32; 3], tilt: f32) -> f32 {
    let (u, v) = b.frame();
    let dl = lab[0] - b.center[0];
    let d = [lab[1] - b.center[1], lab[2] - b.center[2]];
    let dc = d[0] * u[0] + d[1] * u[1] - tilt * dl;
    let dh = d[0] * v[0] + d[1] * v[1];
    let q = ((dl / b.tol[0].max(1e-4)).powi(2)
        + (dc / b.tol[1].max(1e-4)).powi(2)
        + (dh / b.tol[2].max(1e-4)).powi(2))
    .sqrt();
    let t = ((q - 1.0) / b.soft.max(1e-4)).clamp(0.0, 1.0);
    1.0 - t * t * (3.0 - 2.0 * t)
}

/// The finished matte for one linear-RGB pixel. CPU mirror of `keyer_matte`.
pub fn matte(p: &KeyerParams, rgb: [f32; 3]) -> f32 {
    matte_in(p, rgb, REC709_TO_LMS)
}

/// `matte` for pixels in a working space other than Rec.709.
pub fn matte_in(p: &KeyerParams, rgb: [f32; 3], m: crate::color::Mat3) -> f32 {
    let lab = oklab_from_linear_in(rgb, m);
    let mut add = 1.0f32;
    let mut sub = 1.0f32;
    for b in p.blobs.iter().take(MAX_BLOBS) {
        let m = blob_coverage_tilted(b, lab, p.tilt) * b.weight.abs();
        if b.weight < 0.0 {
            sub *= 1.0 - m;
        } else {
            add *= 1.0 - m;
        }
    }
    let mut m = (1.0 - add) * sub;
    m = ((m - p.lo) / (p.hi - p.lo).max(1e-4)).clamp(0.0, 1.0);
    m = m.powf(p.gamma.max(1e-3));
    if p.invert {
        m = 1.0 - m;
    }
    m.clamp(0.0, 1.0)
}

/// Fit a blob to sampled linear-RGB pixels: mean, then per-axis spread on the
/// mean's own hue frame, with one outlier-rejection pass so a drag that
/// clips an edge does not inflate the whole selection.
///
/// `k` scales the measured spread into the CORE radius (2.0 is a sane
/// default: the core covers most of what was dragged over, and the softness
/// multiplier carries the rest).
pub fn fit(samples: &[[f32; 3]], k: f32) -> KeyBlob {
    fit_in(samples, k, REC709_TO_LMS)
}

/// `fit` for samples in a working space other than Rec.709. The blob centre
/// MUST be fitted on the same frame the shader keys on, or the selection
/// lands somewhere other than where the artist sampled.
pub fn fit_in(samples: &[[f32; 3]], k: f32, m: crate::color::Mat3) -> KeyBlob {
    let mut blob = KeyBlob::default();
    if samples.is_empty() {
        return blob;
    }
    let labs: Vec<[f32; 3]> = samples.iter().map(|c| oklab_from_linear_in(*c, m)).collect();
    let mut keep: Vec<bool> = vec![true; labs.len()];
    // Two passes: measure, reject beyond 2.5 sigma, measure again.
    for pass in 0..2 {
        let n = keep.iter().filter(|k| **k).count().max(1) as f32;
        let mut mean = [0.0f32; 3];
        for (lab, ok) in labs.iter().zip(&keep) {
            if *ok {
                for i in 0..3 {
                    mean[i] += lab[i] / n;
                }
            }
        }
        blob.center = mean;
        let (u, v) = blob.frame();
        let mut var = [0.0f32; 3];
        for (lab, ok) in labs.iter().zip(&keep) {
            if !*ok {
                continue;
            }
            let dl = lab[0] - mean[0];
            let d = [lab[1] - mean[1], lab[2] - mean[2]];
            let e = [dl, d[0] * u[0] + d[1] * u[1], d[0] * v[0] + d[1] * v[1]];
            for i in 0..3 {
                var[i] += e[i] * e[i] / n;
            }
        }
        let sd = [var[0].sqrt(), var[1].sqrt(), var[2].sqrt()];
        if pass == 0 && labs.len() > 16 {
            for (lab, ok) in labs.iter().zip(keep.iter_mut()) {
                let dl = lab[0] - mean[0];
                let d = [lab[1] - mean[1], lab[2] - mean[2]];
                let e = [dl, d[0] * u[0] + d[1] * u[1], d[0] * v[0] + d[1] * v[1]];
                let q: f32 = (0..3)
                    .map(|i| (e[i] / sd[i].max(1e-5)).powi(2))
                    .sum::<f32>()
                    .sqrt();
                *ok = q <= 2.5;
            }
            continue;
        }
        for i in 0..3 {
            blob.tol[i] = (sd[i] * k).max(TOL_FLOOR[i]);
        }
    }
    blob
}

/// WGSL spliced into every fused shader's prelude (unused functions are
/// stripped by the shader compiler). `keyer_matte` walks the blob list from
/// the shared uniform, so adding a blob is the only thing that recompiles.
///
/// Uniform layout from the node's first slot:
///   +0  [tilt, lo, hi, gamma]
///   +1  [blob count, invert, preview, unused]
///   +2  [L, a, b, weight]      \ per blob
///   +3  [tolL, tolC, tolH, soft] /
/// The keyer's WGSL, with the Oklab entry matrix built for `working`.
///
/// Was a `const &str`; it has to be generated now because the RGB->LMS step
/// depends on the comp's primaries and must agree with `fit_in`/`matte_in` on
/// the CPU to the digit.
pub fn keyer_wgsl(working: crate::color::Colorspace) -> String {
    let m = oklab_matrix(working);
    format!(
        r#"
fn cbrt_s(x: f32) -> f32 {{
    return sign(x) * pow(abs(x), 0.3333333333);
}}
// Working-space linear -> Oklab. Signed cube root so scene-linear negatives
// (out of a camera gamut) stay finite instead of turning the matte into NaN.
fn oklab_from_linear(c: vec3f) -> vec3f {{
    let l = {:.10} * c.r + {:.10} * c.g + {:.10} * c.b;
    let m = {:.10} * c.r + {:.10} * c.g + {:.10} * c.b;
    let s = {:.10} * c.r + {:.10} * c.g + {:.10} * c.b;
    let q = vec3f(cbrt_s(l), cbrt_s(m), cbrt_s(s));
    return vec3f(
        0.2104542553 * q.x + 0.7936177850 * q.y - 0.0040720468 * q.z,
        1.9779984951 * q.x - 2.4285922050 * q.y + 0.4505937099 * q.z,
        0.0259040371 * q.x + 0.7827717662 * q.y - 0.8086757660 * q.z);
}}
// One blob's coverage: distance to the centre on a frame aligned to the
// blob's own hue (lightness / saturation / hue), normalized by the per-axis
// tolerance, then a smooth falloff over `soft` times that ellipsoid.
fn keyer_blob(lab: vec3f, c: vec4f, t: vec4f, tilt: f32) -> f32 {{
    let ch = length(c.yz);
    var u = vec2f(1.0, 0.0);
    if (ch > 1e-5) {{ u = c.yz / ch; }}
    let v = vec2f(-u.y, u.x);
    let dl = lab.x - c.x;
    let d = lab.yz - c.yz;
    let dc = dot(d, u) - tilt * dl;
    let dh = dot(d, v);
    let e = vec3f(dl, dc, dh) / max(t.xyz, vec3f(1e-4));
    let q = length(e);
    let k = clamp((q - 1.0) / max(t.w, 1e-4), 0.0, 1.0);
    return 1.0 - k * k * (3.0 - 2.0 * k);
}}
// Union is 1 - product(1 - m): order independent and creaseless, so two
// overlapping selections blend instead of leaving a max() ridge in the grade.
fn keyer_matte(rgb: vec3f, slot: u32) -> f32 {{
    let h0 = P.v[slot];
    let h1 = P.v[slot + 1u];
    let lab = oklab_from_linear(rgb);
    var add = 1.0;
    var sub = 1.0;
    let n = u32(h1.x);
    for (var i = 0u; i < n; i = i + 1u) {{
        let c = P.v[slot + 2u + i * 2u];
        let t = P.v[slot + 3u + i * 2u];
        let m = keyer_blob(lab, c, t, h0.x) * abs(c.w);
        if (c.w < 0.0) {{ sub = sub * (1.0 - m); }} else {{ add = add * (1.0 - m); }}
    }}
    var m = (1.0 - add) * sub;
    m = clamp((m - h0.y) / max(h0.z - h0.y, 1e-4), 0.0, 1.0);
    m = pow(m, max(h0.w, 1e-3));
    if (h1.y > 0.5) {{ m = 1.0 - m; }}
    return clamp(m, 0.0, 1.0);
}}
"#,
        m[0][0], m[0][1], m[0][2],
        m[1][0], m[1][1], m[1][2],
        m[2][0], m[2][1], m[2][2],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: [f32; 3], b: [f32; 3], eps: f32) {
        for i in 0..3 {
            assert!((a[i] - b[i]).abs() < eps, "{a:?} vs {b:?}");
        }
    }

    #[test]
    fn oklab_round_trips_including_hdr_and_negatives() {
        for c in [
            [0.0, 0.0, 0.0],
            [0.18, 0.18, 0.18],
            [1.0, 1.0, 1.0],
            [0.4, 0.22, 0.15],
            [12.0, 8.0, 3.0],
            [0.3, -0.02, 0.5],
        ] {
            close(linear_from_oklab(oklab_from_linear(c)), c, 2e-4);
        }
    }

    #[test]
    fn neutrals_are_achromatic_and_lightness_is_monotone() {
        let mut last = f32::MIN;
        for v in [0.0, 0.05, 0.18, 0.5, 1.0, 4.0, 40.0] {
            let lab = oklab_from_linear([v, v, v]);
            assert!(lab[1].abs() < 1e-3 && lab[2].abs() < 1e-3, "{lab:?}");
            assert!(lab[0] > last);
            last = lab[0];
        }
    }

    #[test]
    fn hue_distance_shrinks_as_a_pixel_desaturates() {
        // The HSL-qualifier failure this design exists to avoid: a nearly
        // neutral pixel whose hue is noise must not be judged on that hue.
        let b = KeyBlob {
            center: oklab_from_linear([0.4, 0.22, 0.15]),
            tol: [1.0, 1.0, 0.02],
            soft: 0.0,
            weight: 1.0,
        };
        // Same (wrong) hue angle, two chromas: saturated one is rejected on
        // hue, near-neutral one is not.
        let sat = oklab_from_linear([0.12, 0.30, 0.12]);
        let flat = oklab_from_linear([0.20, 0.212, 0.20]);
        assert_eq!(blob_coverage(&b, sat), 0.0);
        assert!(blob_coverage(&b, flat) > 0.5);
    }

    #[test]
    fn core_is_solid_and_falloff_reaches_zero_at_one_plus_soft() {
        let b = KeyBlob {
            center: [0.5, 0.05, 0.02],
            tol: [0.1, 0.1, 0.1],
            soft: 1.0,
            weight: 1.0,
        };
        assert_eq!(blob_coverage(&b, [0.5, 0.05, 0.02]), 1.0);
        assert_eq!(blob_coverage(&b, [0.59, 0.05, 0.02]), 1.0); // inside core
        let mid = blob_coverage(&b, [0.65, 0.05, 0.02]); // q = 1.5
        assert!((mid - 0.5).abs() < 1e-5, "{mid}");
        assert_eq!(blob_coverage(&b, [0.71, 0.05, 0.02]), 0.0); // q = 2.1
    }

    #[test]
    fn union_is_creaseless_and_subtraction_carves() {
        // Two blobs far enough apart that the midpoint sits in BOTH falloffs.
        let a = KeyBlob {
            center: [0.5, 0.0, 0.0],
            tol: [0.05; 3],
            soft: 1.0,
            weight: 1.0,
        };
        let mut b = a;
        b.center = [0.5, 0.15, 0.0];
        let p = KeyerParams { blobs: vec![a, b], ..Default::default() };
        // A max() union would leave a ridge here exactly equal to the larger
        // of the two; the product union rises above both.
        let mid = [0.5, 0.075, 0.0];
        let each = blob_coverage(&a, mid).max(blob_coverage(&b, mid));
        assert!(each > 0.4 && each < 0.6, "{each}"); // genuinely in the falloff
        assert!(matte(&p, linear_from_oklab(mid)) > each + 0.2);

        // Subtraction: a wide blob with a tight negative one carved out.
        let wide = KeyBlob { tol: [0.3; 3], ..a };
        let cut = KeyBlob { center: [0.5, 0.15, 0.0], weight: -1.0, ..a };
        let p2 = KeyerParams { blobs: vec![wide, cut], ..Default::default() };
        assert!(matte(&p2, linear_from_oklab([0.5, 0.15, 0.0])) < 0.02);
        assert!(matte(&p2, linear_from_oklab([0.5, 0.0, 0.0])) > 0.9);
    }

    #[test]
    fn fit_recovers_a_spread_and_floors_a_flat_pick() {
        let base = [0.4, 0.22, 0.15];
        let spread: Vec<[f32; 3]> = (0..64)
            .map(|i| {
                let t = i as f32 / 63.0 - 0.5;
                [base[0] * (1.0 + t), base[1] * (1.0 + t), base[2] * (1.0 + t)]
            })
            .collect();
        let b = fit(&spread, 2.0);
        // Lightness varies a lot, hue barely at all: the fit has to be
        // anisotropic, not a ball.
        assert!(b.tol[0] > b.tol[2] * 3.0, "{:?}", b.tol);
        assert!(matte(&KeyerParams { blobs: vec![b], ..Default::default() }, base) > 0.99);

        let flat = fit(&[base; 32], 2.0);
        assert_eq!(flat.tol, TOL_FLOOR);
    }

    #[test]
    fn fit_rejects_outliers() {
        let mut s = vec![[0.4, 0.22, 0.15]; 200];
        s.extend([[0.9, 0.05, 0.9]; 12]); // a stray magenta highlight
        let b = fit(&s, 2.0);
        let lab = oklab_from_linear([0.4, 0.22, 0.15]);
        for i in 0..3 {
            assert!((b.center[i] - lab[i]).abs() < 0.02, "{:?} vs {lab:?}", b.center);
        }
    }

    #[test]
    fn levels_and_gamma_are_identity_by_default() {
        let p = KeyerParams::default();
        let m = matte(&p, linear_from_oklab([0.565, 0.03, 0.0]));
        let raw = blob_coverage(&p.blobs[0], [0.565, 0.03, 0.0]);
        assert!((m - raw).abs() < 1e-6, "{m} vs {raw}");
    }
}
