//! ACES 1.3 Output Transform (RRT + SDR ODT), ported from the reference CTL.
//!
//! Source: `ampas/aces-dev` at tag **v1.3.1** — `RRT.ctl`,
//! `ODT.Academy.Rec709_100nits_dim.ctl`, `ACESlib.RRT_Common.ctl`,
//! `ACESlib.ODT_Common.ctl`, `ACESlib.Tonescales.ctl`. Licence is the
//! Academy's permissive terms, so unlike OpenDRT this needs no feature gate
//! and ships in the stock build.
//!
//! Two deliberate departures from a literal transcription:
//!
//! 1. **The AP1->AP0->AP1 round trip is dropped.** The RRT ends by converting
//!    its tonescaled AP1 values to AP0, and the ODT begins by converting them
//!    straight back. They are exact inverses, so the pair is skipped and the
//!    RRT tonescale feeds the ODT tonescale directly.
//!
//! 2. **Matrices are derived, not pasted.** `AP0_2_AP1_MAT`, `AP1_2_XYZ_MAT`,
//!    `D60_2_D65_CAT` and `XYZ_2_DISPLAY_PRI_MAT` come out of
//!    [`crate::color::primaries_matrix`], which already composes the Bradford
//!    adaptation the CTL applies separately and is unit-tested against the
//!    published ACES matrices. The CTL's row-vector matrix convention is
//!    therefore never in play here — one less thing to get backwards.
//!
//! The transform returns **display-linear**; the blit's tail applies the
//! chosen EOTF, exactly as the Filmic path does. To match the reference ODT
//! bit for bit, pair this with the Rec.1886 (2.4) EOTF: the CTL's final step
//! is `bt1886_r(v, 2.4, 1.0, 0.0)`, which for those parameters is `v^(1/2.4)`.
//!
//! Only the SDR (48-nit tonescale, dim surround) rendering is implemented.
//! For P3-D65 and Rec.2020 targets the same rendering is sent through a
//! different output matrix; that is a faithful generalisation but it is NOT
//! the same thing as ACES's own P3/Rec.2020 ODTs, which differ in surround
//! and peak luminance. Rec.709 is the one that matches a reference ODT.

// Constants keep the digit counts their specifications publish, so they can
// be diffed against the source by eye rather than trimmed to f32's shortest
// round-tripping form. Same rationale as `viewtf.rs`.
#![allow(clippy::excessive_precision)]

use super::Gamut;
use crate::color::{primaries_matrix, Primaries};

// --- RRT sweetener constants (ACESlib.RRT_Common.ctl) ----------------------
/// The CTL clamps to `HALF_MIN` before taking a log, so the same value is
/// used here rather than the smallest f32 — with a zero low slope the result
/// is identical, but matching the reference costs nothing.
const HALF_MIN: f32 = 6.103_515_6e-5;

const RRT_GLOW_GAIN: f32 = 0.05;
const RRT_GLOW_MID: f32 = 0.08;
const RRT_RED_SCALE: f32 = 0.82;
const RRT_RED_PIVOT: f32 = 0.03;
const RRT_RED_HUE: f32 = 0.0;
const RRT_RED_WIDTH: f32 = 135.0;
const RRT_SAT_FACTOR: f32 = 0.96;

// --- ODT constants (ACESlib.ODT_Common.ctl) --------------------------------
const CINEMA_WHITE: f32 = 48.0;
const CINEMA_BLACK: f32 = 0.02;
const DIM_SURROUND_GAMMA: f32 = 0.9811;
const ODT_SAT_FACTOR: f32 = 0.93;

// --- Tonescales (ACESlib.Tonescales.ctl) -----------------------------------
// RRT_PARAMS, a 6-coefficient / 4-knot segmented spline in log-log space.
const RRT_COEFS_LOW: [f32; 6] =
    [-4.0, -4.0, -3.1573765773, -0.4852499958, 1.8477324706, 1.8477324706];
const RRT_COEFS_HIGH: [f32; 6] =
    [-0.7185482425, 2.0810307172, 3.6681241237, 4.0, 4.0, 4.0];
/// minPoint, midPoint, maxPoint as (x, y) luminance pairs.
const RRT_MIN: (f32, f32) = (0.18 * 3.0517578125e-5, 0.0001); // 0.18 * 2^-15
const RRT_MID: (f32, f32) = (0.18, 4.8);
const RRT_MAX: (f32, f32) = (0.18 * 262144.0, 10000.0); // 0.18 * 2^18
const RRT_SLOPE_LOW: f32 = 0.0;
const RRT_SLOPE_HIGH: f32 = 0.0;

// ODT_48nits, a 10-coefficient / 8-knot spline. Its min/mid/max x values are
// the RRT spline evaluated at 0.18 * 2^-6.5, 0.18 and 0.18 * 2^6.5, which is
// why they are computed rather than written down.
const ODT_COEFS_LOW: [f32; 10] = [
    -1.6989700043,
    -1.6989700043,
    -1.4779,
    -1.2291,
    -0.8648,
    -0.4480,
    0.00518,
    0.4511080334,
    0.9113744414,
    0.9113744414,
];
const ODT_COEFS_HIGH: [f32; 10] = [
    0.5154386965,
    0.8470437783,
    1.1358,
    1.3802,
    1.5197,
    1.5985,
    1.6467,
    1.6746091357,
    1.6878733390,
    1.6878733390,
];
const ODT_SLOPE_LOW: f32 = 0.0;
const ODT_SLOPE_HIGH: f32 = 0.04;

/// The ODT spline's x anchors, from the RRT spline. `0.18 * 2^-6.5` and
/// `0.18 * 2^6.5`.
fn odt_anchors() -> ((f32, f32), (f32, f32), (f32, f32)) {
    (
        (rrt_spline(0.18 * 2f32.powf(-6.5)), 0.02),
        (rrt_spline(0.18), 4.8),
        (rrt_spline(0.18 * 2f32.powf(6.5)), 48.0),
    )
}

/// A saturation matrix in the textbook column-vector form: `sat * c +
/// (1 - sat) * luma(c)`.
///
/// Written out rather than transcribed from `calc_sat_adjust_matrix` so the
/// CTL's transposed storage cannot leak in. The tests pin the two properties
/// that make it right — neutrals unchanged, `sat = 1` the identity.
fn sat_matrix(sat: f32, y: [f32; 3]) -> crate::color::Mat3 {
    let k = 1.0 - sat;
    [
        [k * y[0] + sat, k * y[1], k * y[2]],
        [k * y[0], k * y[1] + sat, k * y[2]],
        [k * y[0], k * y[1], k * y[2] + sat],
    ]
}

/// Evaluate one segment of a log-log B-spline. `coefs` is the knot vector,
/// `n_knots` the count actually used.
fn spline_segment(logx: f32, lo: (f32, f32), hi: (f32, f32), coefs: &[f32], n_knots: usize) -> f32 {
    let span = hi.0.log10() - lo.0.log10();
    let knot = (n_knots - 1) as f32 * (logx - lo.0.log10()) / span;
    // j indexes a 3-coefficient window: knot_coord lands in (0, n_knots-1),
    // so j runs 0..=n_knots-2 and the window's last index stays in range.
    let j = (knot.floor().max(0.0) as usize).min(n_knots - 2);
    let t = knot - j as f32;
    let cf = [coefs[j], coefs[j + 1], coefs[j + 2]];
    // Monomial -> basis conversion, the `M` in ACESlib.Tonescales.ctl.
    let b0 = 0.5 * cf[0] - cf[1] + 0.5 * cf[2];
    let b1 = -cf[0] + cf[1];
    let b2 = 0.5 * cf[0] + 0.5 * cf[1];
    t * t * b0 + t * b1 + b2
}

/// `segmented_spline_c5_fwd` with RRT_PARAMS: scene-linear -> OCES luminance.
pub fn rrt_spline(x: f32) -> f32 {
    let logx = x.max(HALF_MIN).log10();
    let logy = if logx <= RRT_MIN.0.log10() {
        logx * RRT_SLOPE_LOW + (RRT_MIN.1.log10() - RRT_SLOPE_LOW * RRT_MIN.0.log10())
    } else if logx < RRT_MID.0.log10() {
        spline_segment(logx, RRT_MIN, RRT_MID, &RRT_COEFS_LOW, 4)
    } else if logx < RRT_MAX.0.log10() {
        spline_segment(logx, RRT_MID, RRT_MAX, &RRT_COEFS_HIGH, 4)
    } else {
        logx * RRT_SLOPE_HIGH + (RRT_MAX.1.log10() - RRT_SLOPE_HIGH * RRT_MAX.0.log10())
    };
    10f32.powf(logy)
}

/// `segmented_spline_c9_fwd` with ODT_48nits: OCES luminance -> display cd/m2.
pub fn odt_spline(x: f32) -> f32 {
    let (lo, mid, hi) = odt_anchors();
    let logx = x.max(HALF_MIN).log10();
    let logy = if logx <= lo.0.log10() {
        logx * ODT_SLOPE_LOW + (lo.1.log10() - ODT_SLOPE_LOW * lo.0.log10())
    } else if logx < mid.0.log10() {
        spline_segment(logx, lo, mid, &ODT_COEFS_LOW, 8)
    } else if logx < hi.0.log10() {
        spline_segment(logx, mid, hi, &ODT_COEFS_HIGH, 8)
    } else {
        logx * ODT_SLOPE_HIGH + (hi.1.log10() - ODT_SLOPE_HIGH * hi.0.log10())
    };
    10f32.powf(logy)
}

// ---------------------------------------------------------------------------
// RRT sweeteners (ACESlib.RRT_Common.ctl)
// ---------------------------------------------------------------------------

fn rgb_2_saturation(c: [f32; 3]) -> f32 {
    const TINY: f32 = 1e-10;
    let hi = c[0].max(c[1]).max(c[2]);
    let lo = c[0].min(c[1]).min(c[2]);
    (hi.max(TINY) - lo.max(TINY)) / hi.max(1e-2)
}

fn rgb_2_yc(c: [f32; 3]) -> f32 {
    let (r, g, b) = (c[0], c[1], c[2]);
    let chroma = (b * (b - g) + g * (g - r) + r * (r - b)).max(0.0).sqrt();
    (b + g + r + 1.75 * chroma) / 3.0
}

fn sigmoid_shaper(x: f32) -> f32 {
    let t = (1.0 - (x / 2.0).abs()).max(0.0);
    let y = 1.0 + x.signum() * (1.0 - t * t);
    y / 2.0
}

fn glow_fwd(yc: f32, gain: f32, mid: f32) -> f32 {
    if yc <= 2.0 / 3.0 * mid {
        gain
    } else if yc >= 2.0 * mid {
        0.0
    } else {
        gain * (mid / yc - 0.5)
    }
}

/// Hue in degrees. The CTL returns NaN for a neutral (its hue is genuinely
/// undefined); here it returns 0, because the only consumer multiplies the
/// result by saturation, which is 0 for a neutral — and `NaN * 0` is NaN,
/// which would put a hole in the picture.
fn rgb_2_hue(c: [f32; 3]) -> f32 {
    if c[0] == c[1] && c[1] == c[2] {
        return 0.0;
    }
    let h = (3f32.sqrt() * (c[1] - c[2])).atan2(2.0 * c[0] - c[1] - c[2]).to_degrees();
    if h < 0.0 {
        h + 360.0
    } else {
        h
    }
}

fn center_hue(hue: f32, center: f32) -> f32 {
    let h = hue - center;
    if h < -180.0 {
        h + 360.0
    } else if h > 180.0 {
        h - 360.0
    } else {
        h
    }
}

fn cubic_basis_shaper(x: f32, w: f32) -> f32 {
    if x <= -w / 2.0 || x >= w / 2.0 {
        return 0.0;
    }
    let knot_coord = (x + w / 2.0) * 4.0 / w;
    let j = knot_coord.floor().clamp(0.0, 3.0) as i32;
    let t = knot_coord - j as f32;
    let (m0, m1, m2, m3) = (t * t * t, t * t, t, 1.0);
    const S: f32 = 1.0 / 6.0;
    let y = match j {
        0 => m0 * S,
        1 => m0 * (-3.0 * S) + m1 * (3.0 * S) + m2 * (3.0 * S) + m3 * S,
        2 => m0 * (3.0 * S) + m1 * (-6.0 * S) + m3 * (4.0 * S),
        _ => m0 * (-S) + m1 * (3.0 * S) + m2 * (-3.0 * S) + m3 * S,
    };
    y * 1.5
}

/// The RRT's three sweeteners: glow, red modifier, and the clamp that keeps
/// saturated negatives from turning positive in the AP0->AP1 matrix.
fn rrt_sweeteners(mut aces: [f32; 3]) -> [f32; 3] {
    let saturation = rgb_2_saturation(aces);
    let yc = rgb_2_yc(aces);
    let s = sigmoid_shaper((saturation - 0.4) / 0.2);
    let glow = 1.0 + glow_fwd(yc, RRT_GLOW_GAIN * s, RRT_GLOW_MID);
    for c in aces.iter_mut() {
        *c *= glow;
    }
    let centered = center_hue(rgb_2_hue(aces), RRT_RED_HUE);
    let weight = cubic_basis_shaper(centered, RRT_RED_WIDTH);
    aces[0] += weight * saturation * (RRT_RED_PIVOT - aces[0]) * (1.0 - RRT_RED_SCALE);
    [aces[0].max(0.0), aces[1].max(0.0), aces[2].max(0.0)]
}

/// The whole transform on the CPU: working-space scene-linear in,
/// display-linear out (the caller's EOTF still has to encode it).
///
/// This is the mirror the tests hold the shader to, and the reference for
/// anything that needs to reason about ACES brightness off-GPU.
pub fn apply(rgb: [f32; 3], working: Primaries, display: Gamut) -> [f32; 3] {
    apply_look(rgb, working, display, false)
}

/// The ACES 1.3 Reference Gamut Compression LMT's per-channel thresholds,
/// limits and power (`ACES-LMT - ACES 1.3 Reference Gamut Compression`,
/// what Blender's "ACES 1.3 - Reference Gamut Compression" look runs).
pub const RGC_THRESHOLD: [f32; 3] = [0.815, 0.803, 0.880];
pub const RGC_LIMIT: [f32; 3] = [1.147, 1.264, 1.312];
pub const RGC_POWER: f32 = 1.2;

/// The compression curve's scale, solved so `RGC_LIMIT` lands on 1.
fn rgc_scale() -> [f32; 3] {
    [0, 1, 2].map(|i| {
        let (t, l) = (RGC_THRESHOLD[i] as f64, RGC_LIMIT[i] as f64);
        let p = RGC_POWER as f64;
        ((l - t) / (((1.0 - t) / (l - t)).powf(-p) - 1.0).powf(1.0 / p)) as f32
    })
}

/// Reference Gamut Compression, AP0 in and out: distances from the
/// achromatic axis in AP1 past the threshold are compressed so the limit
/// lands on the gamut boundary. Checked against OCIO's built-in.
pub fn rgc(ap0: [f32; 3]) -> [f32; 3] {
    use crate::color::mat3_apply;
    let a = mat3_apply(primaries_matrix(Primaries::Ap0, Primaries::Ap1), ap0);
    let ach = a[0].max(a[1]).max(a[2]);
    if ach == 0.0 {
        return ap0;
    }
    let s = rgc_scale();
    let c = [0, 1, 2].map(|i| {
        let d = (ach - a[i]) / ach.abs();
        let t = RGC_THRESHOLD[i];
        let cd = if d < t {
            d
        } else {
            let nd = (d - t) / s[i];
            t + s[i] * nd / (1.0 + nd.powf(RGC_POWER)).powf(1.0 / RGC_POWER)
        };
        ach - cd * ach.abs()
    });
    mat3_apply(primaries_matrix(Primaries::Ap1, Primaries::Ap0), c)
}

/// [`apply`] with the gamut-compression look on or off.
pub fn apply_look(rgb: [f32; 3], working: Primaries, display: Gamut, compress: bool) -> [f32; 3] {
    use crate::color::mat3_apply;
    let to_ap0 = primaries_matrix(working, Primaries::Ap0);
    let mut ap0 = mat3_apply(to_ap0, rgb);
    if compress {
        ap0 = rgc(ap0);
    }
    let aces = rrt_sweeteners(ap0);

    // AP0 -> AP1, then the RRT saturation adjust and tonescale.
    let ap1 = mat3_apply(primaries_matrix(Primaries::Ap0, Primaries::Ap1), aces);
    let ap1 = [ap1[0].clamp(0.0, 65504.0), ap1[1].clamp(0.0, 65504.0), ap1[2].clamp(0.0, 65504.0)];
    let y = Primaries::Ap1.luma_weights();
    let ap1 = mat3_apply(sat_matrix(RRT_SAT_FACTOR, y), ap1);
    let oces = [rrt_spline(ap1[0]), rrt_spline(ap1[1]), rrt_spline(ap1[2])];

    // ODT. The CTL's AP1->AP0->AP1 round trip between the two tonescales is
    // an exact identity and is skipped.
    let disp = [odt_spline(oces[0]), odt_spline(oces[1]), odt_spline(oces[2])];
    let mut cv = [
        (disp[0] - CINEMA_BLACK) / (CINEMA_WHITE - CINEMA_BLACK),
        (disp[1] - CINEMA_BLACK) / (CINEMA_WHITE - CINEMA_BLACK),
        (disp[2] - CINEMA_BLACK) / (CINEMA_WHITE - CINEMA_BLACK),
    ];
    cv = dark_to_dim(cv);
    cv = mat3_apply(sat_matrix(ODT_SAT_FACTOR, y), cv);

    let out_p = match display {
        Gamut::Rec2020 => Primaries::Rec2020,
        Gamut::P3D65 => Primaries::P3D65,
        _ => Primaries::Rec709,
    };
    let out = mat3_apply(primaries_matrix(Primaries::Ap1, out_p), cv);
    [out[0].clamp(0.0, 1.0), out[1].clamp(0.0, 1.0), out[2].clamp(0.0, 1.0)]
}

/// `darkSurround_to_dimSurround`: raise Y to 0.9811 with chromaticity held,
/// which is what a cinema-referred rendering needs on a dim-surround monitor.
fn dark_to_dim(cv: [f32; 3]) -> [f32; 3] {
    use crate::color::mat3_apply;
    let xyz = mat3_apply(Primaries::Ap1.to_xyz(), cv);
    let sum = xyz[0] + xyz[1] + xyz[2];
    if sum <= 0.0 {
        return [0.0, 0.0, 0.0];
    }
    let (x, y) = (xyz[0] / sum, xyz[1] / sum);
    let big_y = xyz[1].max(0.0).powf(DIM_SURROUND_GAMMA);
    if y <= 0.0 {
        return [0.0, 0.0, 0.0];
    }
    let xyz2 = [x * big_y / y, big_y, (1.0 - x - y) * big_y / y];
    mat3_apply(Primaries::Ap1.from_xyz(), xyz2)
}

#[cfg(test)]
mod tests {

    /// OCIO 2.6's `ACES-LMT - ACES 1.3 Reference Gamut Compression` on the
    /// same AP0 values (what Blender's look of that name runs).
    #[test]
    fn rgc_matches_ocio() {
        let cases: [([f32; 3], [f32; 3]); 6] = [
            ([0.18, 0.18, 0.18], [0.18, 0.18, 0.18]),
            ([1.0, 0.0, 0.0], [1.030505, 0.130857, 0.062008]),
            ([0.02, 0.3, -0.05], [0.055513, 0.307117, 0.006708]),
            ([5.0, 0.1, 0.2], [5.122938, 0.682958, 0.38872]),
            ([-0.1, 0.5, 0.5], [0.129046, 0.514753, 0.49818]),
            ([0.3, 0.25, 0.9], [0.3, 0.25, 0.9]),
        ];
        for (x, want) in cases {
            let got = super::rgc(x);
            for k in 0..3 {
                let tol = 1e-4 * want[k].abs().max(1.0);
                assert!((got[k] - want[k]).abs() < tol, "{x:?}: {got:?} vs {want:?}");
            }
        }
        assert_eq!(super::rgc([0.0; 3]), [0.0; 3]);
    }
    use super::*;

    /// ACES renders 18% scene grey to roughly 10% of display peak. That
    /// number is the one everybody quotes and the one a broken spline moves
    /// first, so it is the anchor.
    #[test]
    fn mid_grey_lands_near_ten_percent_of_peak() {
        let g = apply([0.18, 0.18, 0.18], Primaries::Ap1, Gamut::Rec709);
        assert!(
            (0.085..=0.115).contains(&g[1]),
            "18% grey -> {:.4} display-linear, expected ~0.10",
            g[1]
        );
        // ...and it stays grey.
        assert!((g[0] - g[1]).abs() < 2e-3 && (g[2] - g[1]).abs() < 2e-3, "{g:?}");
    }

    #[test]
    fn black_is_black_and_the_curve_saturates_below_peak() {
        let k = apply([0.0, 0.0, 0.0], Primaries::Ap1, Gamut::Rec709);
        assert!(k[1] < 1e-3, "black -> {k:?}");
        // A huge value must approach, not exceed, display peak.
        let hot = apply([200.0, 200.0, 200.0], Primaries::Ap1, Gamut::Rec709);
        assert!(hot[1] <= 1.0 && hot[1] > 0.9, "200.0 -> {hot:?}");
    }

    #[test]
    fn the_tone_curve_is_monotonic() {
        let mut prev = -1.0f32;
        let mut x = 1e-4f32;
        while x < 100.0 {
            let y = apply([x, x, x], Primaries::Ap1, Gamut::Rec709)[1];
            assert!(y >= prev - 1e-6, "not monotonic at {x}: {y} < {prev}");
            prev = y;
            x *= 1.1;
        }
    }

    /// Every output must be finite for anything a comp can produce, including
    /// the negatives a wide-gamut conversion leaves behind.
    #[test]
    fn survives_negatives_and_neutral_hue() {
        for c in [
            [-0.1, 0.2, 0.3],
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [-1.0, -1.0, -1.0],
            [50.0, 0.0, -0.2],
        ] {
            let o = apply(c, Primaries::Ap1, Gamut::Rec709);
            assert!(o.iter().all(|v| v.is_finite()), "{c:?} -> {o:?}");
            assert!(o.iter().all(|v| (0.0..=1.0).contains(v)), "{c:?} -> {o:?}");
        }
    }

    /// The saturation matrix is the one piece written from the formula
    /// rather than transcribed, so its two defining properties are pinned.
    #[test]
    fn sat_matrix_holds_neutrals_and_is_identity_at_one() {
        use crate::color::{mat3_apply, mat3_is_identity};
        let y = Primaries::Ap1.luma_weights();
        assert!(mat3_is_identity(sat_matrix(1.0, y)));
        for s in [0.0f32, 0.5, 0.93, 0.96] {
            let n = mat3_apply(sat_matrix(s, y), [0.42, 0.42, 0.42]);
            for v in n {
                assert!((v - 0.42).abs() < 1e-5, "sat {s} moved a neutral: {n:?}");
            }
        }
        // Fully desaturated really is luminance.
        let g = mat3_apply(sat_matrix(0.0, y), [0.7, 0.3, 0.1]);
        let luma = 0.7 * y[0] + 0.3 * y[1] + 0.1 * y[2];
        assert!(g.iter().all(|v| (v - luma).abs() < 1e-5), "{g:?} vs {luma}");
    }

    /// The RRT spline's own anchor: its mid point is (0.18, 4.8) by
    /// definition, so the spline has to actually pass through it.
    #[test]
    fn the_rrt_spline_passes_through_its_own_knots() {
        assert!((rrt_spline(0.18) - 4.8).abs() < 0.05, "{}", rrt_spline(0.18));
        // And the ODT spline maps that to its own mid point, 4.8 cd/m2.
        assert!((odt_spline(rrt_spline(0.18)) - 4.8).abs() < 0.05);
    }

    /// A wider display gamut must not change the neutral axis.
    #[test]
    fn every_display_gamut_keeps_grey_grey() {
        for d in [Gamut::Rec709, Gamut::P3D65, Gamut::Rec2020] {
            let g = apply([0.18, 0.18, 0.18], Primaries::Ap1, d);
            assert!(
                (g[0] - g[1]).abs() < 3e-3 && (g[2] - g[1]).abs() < 3e-3,
                "{d:?}: {g:?}"
            );
        }
    }

    /// Feeding the same picture in a different working space must give the
    /// same answer: the input matrix is part of the transform, not an
    /// assumption about what the caller already did.
    #[test]
    fn working_space_is_converted_not_assumed() {
        let in_ap1 = [0.3, 0.18, 0.09];
        let to709 = primaries_matrix(Primaries::Ap1, Primaries::Rec709);
        let in_709 = crate::color::mat3_apply(to709, in_ap1);
        let a = apply(in_ap1, Primaries::Ap1, Gamut::Rec709);
        let b = apply(in_709, Primaries::Rec709, Gamut::Rec709);
        for i in 0..3 {
            assert!((a[i] - b[i]).abs() < 2e-3, "ch{i}: {a:?} vs {b:?}");
        }
    }
}

// ---------------------------------------------------------------------------
// Shader
// ---------------------------------------------------------------------------

/// Format a row-major 3x3 as a WGSL `mat3x3<f32>` literal (column-major
/// constructor: each `vec3f` is a COLUMN).
fn mat_lit(m: crate::color::Mat3) -> String {
    format!(
        "mat3x3<f32>(vec3f({:.9}, {:.9}, {:.9}), vec3f({:.9}, {:.9}, {:.9}), vec3f({:.9}, {:.9}, {:.9}))",
        m[0][0], m[1][0], m[2][0],
        m[0][1], m[1][1], m[2][1],
        m[0][2], m[1][2], m[2][2],
    )
}

/// The `view_transform` entry point plus its helpers.
///
/// Uses two of the blit's existing uniform matrices rather than adding any:
/// `in_to_rs` carries working -> AP0, `out_cwp` carries AP1 -> display. Both
/// are filled by `ViewSettings::matrices`.
pub fn wgsl(_display: Gamut) -> String {
    let y = Primaries::Ap1.luma_weights();
    let (lo, mid, hi) = odt_anchors();
    let consts = format!(
        r#"
// --- ACES 1.3 constants, derived in engine/src/viewtf/aces.rs -------------
const ACES_AP0_2_AP1: mat3x3<f32> = {ap0_2_ap1};
const ACES_AP1_2_AP0: mat3x3<f32> = {ap1_2_ap0};
const ACES_RGC_THR: vec3f = vec3f({thr0}, {thr1}, {thr2});
const ACES_RGC_SCALE: vec3f = vec3f({sc0:.9}, {sc1:.9}, {sc2:.9});
const ACES_RGC_POW: f32 = {pw};
const ACES_RRT_SAT: mat3x3<f32> = {rrt_sat};
const ACES_ODT_SAT: mat3x3<f32> = {odt_sat};
const ACES_AP1_2_XYZ: mat3x3<f32> = {ap1_2_xyz};
const ACES_XYZ_2_AP1: mat3x3<f32> = {xyz_2_ap1};
const ACES_RRT_LOG_MIN: f32 = {rrt_log_min:.9};
const ACES_RRT_LOG_MID: f32 = {rrt_log_mid:.9};
const ACES_RRT_LOG_MAX: f32 = {rrt_log_max:.9};
const ACES_ODT_LOG_MIN: f32 = {odt_log_min:.9};
const ACES_ODT_LOG_MID: f32 = {odt_log_mid:.9};
const ACES_ODT_LOG_MAX: f32 = {odt_log_max:.9};
"#,
        ap0_2_ap1 = mat_lit(primaries_matrix(Primaries::Ap0, Primaries::Ap1)),
        ap1_2_ap0 = mat_lit(primaries_matrix(Primaries::Ap1, Primaries::Ap0)),
        thr0 = RGC_THRESHOLD[0],
        thr1 = RGC_THRESHOLD[1],
        thr2 = RGC_THRESHOLD[2],
        sc0 = rgc_scale()[0],
        sc1 = rgc_scale()[1],
        sc2 = rgc_scale()[2],
        pw = RGC_POWER,
        rrt_sat = mat_lit(sat_matrix(RRT_SAT_FACTOR, y)),
        odt_sat = mat_lit(sat_matrix(ODT_SAT_FACTOR, y)),
        ap1_2_xyz = mat_lit(Primaries::Ap1.to_xyz()),
        xyz_2_ap1 = mat_lit(Primaries::Ap1.from_xyz()),
        rrt_log_min = RRT_MIN.0.log10(),
        rrt_log_mid = RRT_MID.0.log10(),
        rrt_log_max = RRT_MAX.0.log10(),
        odt_log_min = lo.0.log10(),
        odt_log_mid = mid.0.log10(),
        odt_log_max = hi.0.log10(),
    );
    format!("{consts}{ACES_BODY}")
}

/// Everything with no interpolated constants. Mirrors the CPU path in this
/// module function for function; `engine/tests/aces.rs` holds them together.
const ACES_BODY: &str = r#"
// ACES 1.3 Reference Gamut Compression, AP0 in and out. Mirrors `rgc`.
fn aces_rgc(ap0: vec3f) -> vec3f {
    let a = ACES_AP0_2_AP1 * ap0;
    let ach = max(a.r, max(a.g, a.b));
    if (ach == 0.0) {
        return ap0;
    }
    let d = (vec3f(ach) - a) / abs(ach);
    let nd = max(d - ACES_RGC_THR, vec3f(0.0)) / ACES_RGC_SCALE;
    let cd = select(
        ACES_RGC_THR + ACES_RGC_SCALE * nd / pow(vec3f(1.0) + pow(nd, vec3f(ACES_RGC_POW)), vec3f(1.0 / ACES_RGC_POW)),
        d,
        d < ACES_RGC_THR,
    );
    return ACES_AP1_2_AP0 * (vec3f(ach) - cd * abs(ach));
}

// One segment of a log-log B-spline. `cf` is the three-coefficient window
// already selected by the caller, `t` the position within it.
fn aces_seg(cf: vec3f, t: f32) -> f32 {
    // Monomial -> basis conversion, the `M` in ACESlib.Tonescales.ctl.
    let b0 = 0.5 * cf.x - cf.y + 0.5 * cf.z;
    let b1 = -cf.x + cf.y;
    let b2 = 0.5 * cf.x + 0.5 * cf.y;
    return t * t * b0 + t * b1 + b2;
}

// segmented_spline_c5_fwd with RRT_PARAMS. Both slopes are zero, so the
// extensions are constants rather than lines.
fn aces_rrt_spline(x: f32) -> f32 {
    let logx = log(max(x, 6.1035156e-5)) / log(10.0);
    var logy: f32;
    if (logx <= ACES_RRT_LOG_MIN) {
        logy = -4.0; // log10(0.0001)
    } else if (logx < ACES_RRT_LOG_MID) {
        var c = array<f32, 6>(-4.0, -4.0, -3.1573765773, -0.4852499958, 1.8477324706, 1.8477324706);
        let k = 3.0 * (logx - ACES_RRT_LOG_MIN) / (ACES_RRT_LOG_MID - ACES_RRT_LOG_MIN);
        let j = i32(clamp(floor(k), 0.0, 2.0));
        logy = aces_seg(vec3f(c[j], c[j + 1], c[j + 2]), k - f32(j));
    } else if (logx < ACES_RRT_LOG_MAX) {
        var c = array<f32, 6>(-0.7185482425, 2.0810307172, 3.6681241237, 4.0, 4.0, 4.0);
        let k = 3.0 * (logx - ACES_RRT_LOG_MID) / (ACES_RRT_LOG_MAX - ACES_RRT_LOG_MID);
        let j = i32(clamp(floor(k), 0.0, 2.0));
        logy = aces_seg(vec3f(c[j], c[j + 1], c[j + 2]), k - f32(j));
    } else {
        logy = 4.0; // log10(10000)
    }
    return pow(10.0, logy);
}

// segmented_spline_c9_fwd with ODT_48nits.
fn aces_odt_spline(x: f32) -> f32 {
    let logx = log(max(x, 6.1035156e-5)) / log(10.0);
    var logy: f32;
    if (logx <= ACES_ODT_LOG_MIN) {
        logy = -1.6989700043; // log10(0.02)
    } else if (logx < ACES_ODT_LOG_MID) {
        var c = array<f32, 10>(-1.6989700043, -1.6989700043, -1.4779, -1.2291, -0.8648,
                               -0.4480, 0.00518, 0.4511080334, 0.9113744414, 0.9113744414);
        let k = 7.0 * (logx - ACES_ODT_LOG_MIN) / (ACES_ODT_LOG_MID - ACES_ODT_LOG_MIN);
        let j = i32(clamp(floor(k), 0.0, 6.0));
        logy = aces_seg(vec3f(c[j], c[j + 1], c[j + 2]), k - f32(j));
    } else if (logx < ACES_ODT_LOG_MAX) {
        var c = array<f32, 10>(0.5154386965, 0.8470437783, 1.1358, 1.3802, 1.5197,
                               1.5985, 1.6467, 1.6746091357, 1.6878733390, 1.6878733390);
        let k = 7.0 * (logx - ACES_ODT_LOG_MID) / (ACES_ODT_LOG_MAX - ACES_ODT_LOG_MID);
        let j = i32(clamp(floor(k), 0.0, 6.0));
        logy = aces_seg(vec3f(c[j], c[j + 1], c[j + 2]), k - f32(j));
    } else {
        // slopeHigh is 0.04, so this extension really is a line.
        logy = logx * 0.04 + (log(48.0) / log(10.0) - 0.04 * ACES_ODT_LOG_MAX);
    }
    return pow(10.0, logy);
}

fn aces_saturation(c: vec3f) -> f32 {
    let hi = max(max(c.r, c.g), c.b);
    let lo = min(min(c.r, c.g), c.b);
    return (max(hi, 1e-10) - max(lo, 1e-10)) / max(hi, 1e-2);
}

fn aces_yc(c: vec3f) -> f32 {
    let chroma = sqrt(max(c.b * (c.b - c.g) + c.g * (c.g - c.r) + c.r * (c.r - c.b), 0.0));
    return (c.b + c.g + c.r + 1.75 * chroma) / 3.0;
}

fn aces_sigmoid(x: f32) -> f32 {
    let t = max(1.0 - abs(x / 2.0), 0.0);
    return (1.0 + sign(x) * (1.0 - t * t)) / 2.0;
}

fn aces_glow(yc: f32, gain: f32, mid: f32) -> f32 {
    if (yc <= 2.0 / 3.0 * mid) { return gain; }
    if (yc >= 2.0 * mid) { return 0.0; }
    return gain * (mid / yc - 0.5);
}

// Hue in degrees. Neutrals return 0 rather than the CTL's NaN: the only
// consumer multiplies by saturation, which is 0 there, and NaN * 0 is NaN.
fn aces_hue(c: vec3f) -> f32 {
    if (c.r == c.g && c.g == c.b) { return 0.0; }
    var h = degrees(atan2(sqrt(3.0) * (c.g - c.b), 2.0 * c.r - c.g - c.b));
    if (h < 0.0) { h = h + 360.0; }
    return h;
}

// cubic_basis_shaper, w = 135 degrees, centred on 0.
fn aces_red_weight(hue: f32) -> f32 {
    var x = hue;
    if (x < -180.0) { x = x + 360.0; }
    if (x > 180.0) { x = x - 360.0; }
    let w = 135.0;
    if (x <= -w / 2.0 || x >= w / 2.0) { return 0.0; }
    let k = (x + w / 2.0) * 4.0 / w;
    let j = i32(clamp(floor(k), 0.0, 3.0));
    let t = k - f32(j);
    let m = vec4f(t * t * t, t * t, t, 1.0);
    let s = 1.0 / 6.0;
    var y = 0.0;
    if (j == 0) {
        y = m.x * s;
    } else if (j == 1) {
        y = m.x * (-3.0 * s) + m.y * (3.0 * s) + m.z * (3.0 * s) + m.w * s;
    } else if (j == 2) {
        y = m.x * (3.0 * s) + m.y * (-6.0 * s) + m.w * (4.0 * s);
    } else {
        y = m.x * (-s) + m.y * (3.0 * s) + m.z * (-3.0 * s) + m.w * s;
    }
    return y * 1.5;
}

// darkSurround_to_dimSurround: raise Y, hold chromaticity.
fn aces_dim(cv: vec3f) -> vec3f {
    let xyz = ACES_AP1_2_XYZ * cv;
    let sum = xyz.x + xyz.y + xyz.z;
    if (sum <= 0.0) { return vec3f(0.0); }
    let x = xyz.x / sum;
    let y = xyz.y / sum;
    if (y <= 0.0) { return vec3f(0.0); }
    let big = pow(max(xyz.y, 0.0), 0.9811);
    return ACES_XYZ_2_AP1 * vec3f(x * big / y, big, (1.0 - x - y) * big / y);
}

// ACES 1.3 Output Transform. `in_to_rs` is working -> AP0; `out_cwp` is
// AP1 -> display. Display-LINEAR out: the blit's tail applies the EOTF, so
// pair this with Rec.1886 to match the reference ODT exactly.
fn view_transform(rgb: vec3f) -> vec3f {
    var aces = u.in_to_rs * rgb;
    // The look: ACES 1.3 Reference Gamut Compression (drt[0].x = 1).
    if (u.drt[0].x > 0.5) {
        aces = aces_rgc(aces);
    }

    // RRT sweeteners: glow, then the red modifier.
    let sat = aces_saturation(aces);
    let s = aces_sigmoid((sat - 0.4) / 0.2);
    aces = aces * (1.0 + aces_glow(aces_yc(aces), 0.05 * s, 0.08));
    let weight = aces_red_weight(aces_hue(aces));
    aces.r = aces.r + weight * sat * (0.03 - aces.r) * (1.0 - 0.82);
    aces = max(aces, vec3f(0.0));

    // AP0 -> AP1, RRT saturation adjust, RRT tonescale.
    var ap1 = clamp(ACES_AP0_2_AP1 * aces, vec3f(0.0), vec3f(65504.0));
    ap1 = ACES_RRT_SAT * ap1;
    let oces = vec3f(aces_rrt_spline(ap1.r), aces_rrt_spline(ap1.g), aces_rrt_spline(ap1.b));

    // ODT. The reference's AP1 -> AP0 -> AP1 hop between the two tonescales
    // is an exact identity and is skipped.
    let d = vec3f(aces_odt_spline(oces.r), aces_odt_spline(oces.g), aces_odt_spline(oces.b));
    var cv = (d - vec3f(0.02)) / (48.0 - 0.02);
    cv = ACES_ODT_SAT * aces_dim(cv);
    return clamp(u.out_cwp * cv, vec3f(0.0), vec3f(1.0));
}
"#;
