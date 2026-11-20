//! Scene-referred colour management: what a plate *is*, what the comp works
//! in, and what a Write puts on disk.
//!
//! The comp has ONE working space (ACEScg by default). Everything entering it
//! is converted at the reader leaf; everything leaving it is converted at the
//! Write. In between, every node can assume its pixels are scene-linear in
//! the working primaries — which is the assumption the grade maths, the
//! keyer's Oklab and the luma weights all quietly needed and never had.
//!
//! Every transform here has the same three-part shape:
//!
//! ```text
//! src encoding --decode--> scene-linear src --matrix--> scene-linear dst --encode--> dst encoding
//! ```
//!
//! That is not a simplification for our convenience: every ACES input
//! transform for every camera ever shipped is a log curve and a 3x3, because
//! that is what a camera vendor can specify exactly. LUT-shaped transforms
//! (a show LUT, an OCIO config's baked ops) do not fit and are deliberately
//! not modelled yet — see the module docs in `color/README` when they land.
//!
//! **Matrices are derived, not pasted.** Primaries are stored as the xy
//! chromaticities the standards actually publish, and the RGB->XYZ matrix is
//! solved from them in f64. Pasted matrices are how you end up with two
//! subtly different copies of AP1 in one codebase (see `viewtf.rs`, whose set
//! is D65-adapted for OpenDRT and therefore *not* the ACES matrices). The
//! tests pin the derived matrices against the published ones.

// Constants keep the digit counts their specifications publish, so they can
// be diffed against the source by eye rather than trimmed to f32's shortest
// round-tripping form. Same rationale as `viewtf.rs`.
#![allow(clippy::excessive_precision)]

use std::fmt::Write as _;

/// Row-major 3x3, `m[row][col]`. f32 at the boundary; the derivation inside
/// runs in f64 because inverting a matrix of chromaticities in f32 loses
/// digits that then show up as a white-point drift.
pub type Mat3 = [[f32; 3]; 3];
type M3 = [[f64; 3]; 3];

pub const IDENTITY: Mat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

// ---------------------------------------------------------------------------
// White points
// ---------------------------------------------------------------------------

/// A white point as CIE xy. Named rather than free-form: these are the only
/// ones the built-in spaces use, and a typo'd white is invisible until
/// someone compares against another compositor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum White {
    /// CIE D65, the display standard.
    D65,
    /// The ACES white, ~D60 but defined by its own xy rather than by a
    /// Planckian fit — hence "ACES white" and not "D60" in the spec.
    Aces,
    /// DCI projector white.
    Dci,
}

impl White {
    pub const fn xy(self) -> (f64, f64) {
        match self {
            White::D65 => (0.3127, 0.3290),
            White::Aces => (0.32168, 0.33767),
            White::Dci => (0.314, 0.351),
        }
    }

    /// XYZ with Y normalised to 1.
    fn xyz(self) -> [f64; 3] {
        let (x, y) = self.xy();
        [x / y, 1.0, (1.0 - x - y) / y]
    }
}

// ---------------------------------------------------------------------------
// Primaries
// ---------------------------------------------------------------------------

/// A set of RGB primaries plus its native white point.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Primaries {
    /// The default, and deliberately NOT ACEScg: this is what the comp
    /// worked in before there was a working space to choose, so a graph
    /// built without one behaves exactly as it always did. New comps are
    /// created in ACEScg by the app, which records the choice in the setup.
    #[default]
    Rec709,

    /// ACES AP0, the ACES2065-1 primaries. Encloses the entire visible
    /// locus, so its green and blue are imaginary.
    Ap0,
    /// ACES AP1, the ACEScg/ACEScc/ACEScct primaries.
    Ap1,
    P3D65,
    P3Dci,
    Rec2020,
    ArriWideGamut3,
    ArriWideGamut4,
    SGamut3Cine,
    RedWideGamut,
    CanonCinema,
    VGamut,
    BmdWideGamutGen5,
    /// DaVinci Wide Gamut: Resolve's colour-managed timeline gamut, what a
    /// LUT exported from a DWG/Intermediate project expects.
    DaVinciWideGamut,
    /// CIE XYZ itself, for completeness and for tests.
    Xyz,
}

impl Primaries {
    pub const ALL: [Self; 15] = [
        Self::Rec709,
        Self::Ap0,
        Self::Ap1,
        Self::P3D65,
        Self::P3Dci,
        Self::Rec2020,
        Self::ArriWideGamut3,
        Self::ArriWideGamut4,
        Self::SGamut3Cine,
        Self::RedWideGamut,
        Self::CanonCinema,
        Self::VGamut,
        Self::BmdWideGamutGen5,
        Self::DaVinciWideGamut,
        Self::Xyz,
    ];

    /// Red, green, blue chromaticities as published by the standard.
    ///
    /// Several of these have negative or >1 coordinates. That is correct and
    /// intentional: a camera gamut wide enough to hold real sensor response
    /// has imaginary primaries.
    pub const fn rgb_xy(self) -> [(f64, f64); 3] {
        match self {
            Self::Rec709 => [(0.640, 0.330), (0.300, 0.600), (0.150, 0.060)],
            Self::Ap0 => [(0.7347, 0.2653), (0.0, 1.0), (0.0001, -0.0770)],
            Self::Ap1 => [(0.713, 0.293), (0.165, 0.830), (0.128, 0.044)],
            Self::P3D65 | Self::P3Dci => [(0.680, 0.320), (0.265, 0.690), (0.150, 0.060)],
            Self::Rec2020 => [(0.708, 0.292), (0.170, 0.797), (0.131, 0.046)],
            Self::ArriWideGamut3 => [(0.6840, 0.3130), (0.2210, 0.8480), (0.0861, -0.1020)],
            Self::ArriWideGamut4 => [(0.7347, 0.2653), (0.1424, 0.8576), (0.0991, -0.0308)],
            Self::SGamut3Cine => [(0.766, 0.275), (0.225, -0.800), (0.089, -0.087)],
            Self::RedWideGamut => {
                [(0.780308, 0.304253), (0.121595, 1.493994), (0.095612, -0.084589)]
            }
            Self::CanonCinema => [(0.740, 0.270), (0.170, 1.140), (0.080, -0.100)],
            Self::VGamut => [(0.730, 0.280), (0.165, 0.840), (0.100, -0.030)],
            Self::BmdWideGamutGen5 => [(0.7422, 0.2859), (0.2228, 0.8615), (0.0925, -0.0605)],
            // Blackmagic's "DaVinci Resolve 17 Wide Gamut Intermediate" note.
            Self::DaVinciWideGamut => [(0.8000, 0.3130), (0.1682, 0.9877), (0.0790, -0.1155)],
            // Unused for XYZ: to_xyz short-circuits to the identity.
            Self::Xyz => [(1.0, 0.0), (0.0, 1.0), (0.0, 0.0)],
        }
    }

    pub const fn white(self) -> White {
        match self {
            Self::Ap0 | Self::Ap1 => White::Aces,
            Self::P3Dci => White::Dci,
            _ => White::D65,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Rec709 => "Rec.709 / sRGB",
            Self::Ap0 => "ACES AP0",
            Self::Ap1 => "ACES AP1",
            Self::P3D65 => "P3-D65",
            Self::P3Dci => "P3-DCI",
            Self::Rec2020 => "Rec.2020",
            Self::ArriWideGamut3 => "ARRI Wide Gamut 3",
            Self::ArriWideGamut4 => "ARRI Wide Gamut 4",
            Self::SGamut3Cine => "Sony S-Gamut3.Cine",
            Self::RedWideGamut => "REDWideGamutRGB",
            Self::CanonCinema => "Canon Cinema Gamut",
            Self::VGamut => "Panasonic V-Gamut",
            Self::BmdWideGamutGen5 => "Blackmagic Wide Gamut Gen5",
            Self::DaVinciWideGamut => "DaVinci Wide Gamut",
            Self::Xyz => "CIE XYZ",
        }
    }

    /// RGB -> XYZ at this space's OWN white point, solved from the published
    /// chromaticities (Wyszecki & Stiles' construction, as every colour
    /// library implements it).
    fn to_xyz_f64(self) -> M3 {
        if matches!(self, Self::Xyz) {
            return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        }
        let p = self.rgb_xy();
        // Columns are each primary's xyz with z = 1 - x - y.
        let mut m = [[0.0f64; 3]; 3];
        for (c, (x, y)) in p.iter().enumerate() {
            m[0][c] = *x;
            m[1][c] = *y;
            m[2][c] = 1.0 - x - y;
        }
        // Scale each column so the primaries sum to the white point.
        let s = mat3_apply64(mat3_inverse64(m), self.white().xyz());
        for (c, sc) in s.iter().enumerate() {
            for row in m.iter_mut() {
                row[c] *= sc;
            }
        }
        m
    }

    /// RGB -> XYZ at this space's own white point.
    pub fn to_xyz(self) -> Mat3 {
        to_f32(self.to_xyz_f64())
    }

    /// XYZ (at this space's own white point) -> RGB.
    pub fn from_xyz(self) -> Mat3 {
        to_f32(mat3_inverse64(self.to_xyz_f64()))
    }

    /// Red, green, blue and white xy, in the order an OpenEXR
    /// `chromaticities` attribute wants them.
    ///
    /// Writing this is not optional in practice: the EXR spec says a file
    /// WITHOUT the attribute is to be read as Rec.709, so an untagged ACEScg
    /// render is not merely undocumented, it is actively mislabelled.
    pub fn chromaticities(self) -> [(f32, f32); 4] {
        let p = self.rgb_xy();
        let w = self.white().xy();
        [
            (p[0].0 as f32, p[0].1 as f32),
            (p[1].0 as f32, p[1].1 as f32),
            (p[2].0 as f32, p[2].1 as f32),
            (w.0 as f32, w.1 as f32),
        ]
    }

    /// Relative luminance weights for these primaries — the Y row of the
    /// RGB->XYZ matrix, by definition. This is what a "luma" key should use;
    /// the Rec.709 triple is only correct when the working space IS Rec.709.
    pub fn luma_weights(self) -> [f32; 3] {
        let m = self.to_xyz();
        m[1]
    }
}

/// Bradford chromatic adaptation `src` white -> `dst` white, in XYZ.
///
/// Without this, converting ACEScg (ACES white) to Rec.709 (D65) leaves a
/// warm cast on neutrals: the grey card stops being grey.
pub fn bradford(src: White, dst: White) -> Mat3 {
    to_f32(bradford64(src, dst))
}

fn bradford64(src: White, dst: White) -> M3 {
    if src == dst {
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }
    const MA: M3 = [
        [0.8951, 0.2664, -0.1614],
        [-0.7502, 1.7135, 0.0367],
        [0.0389, -0.0685, 1.0296],
    ];
    let s = mat3_apply64(MA, src.xyz());
    let d = mat3_apply64(MA, dst.xyz());
    let diag = [
        [d[0] / s[0], 0.0, 0.0],
        [0.0, d[1] / s[1], 0.0],
        [0.0, 0.0, d[2] / s[2]],
    ];
    mat3_mul64(mat3_inverse64(MA), mat3_mul64(diag, MA))
}

/// The full `src` RGB -> `dst` RGB matrix, adapting white points on the way.
pub fn primaries_matrix(src: Primaries, dst: Primaries) -> Mat3 {
    if src == dst {
        return IDENTITY;
    }
    let to = src.to_xyz_f64();
    let cat = bradford64(src.white(), dst.white());
    let from = mat3_inverse64(dst.to_xyz_f64());
    to_f32(mat3_mul64(from, mat3_mul64(cat, to)))
}

// ---------------------------------------------------------------------------
// Transfer curves
// ---------------------------------------------------------------------------

/// A transfer function: encoded code value <-> scene-linear.
///
/// Constants are the published ones, and every curve has a test pinning its
/// published 18% mid-grey code value. That anchor is the cheapest possible
/// guard against a transposed digit, because it is the one number every
/// vendor states on the first page of the spec.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Curve {
    /// Already scene-linear.
    #[default]
    Linear,
    /// The sRGB piecewise curve (NOT a pure 2.2 power — the toe matters in
    /// the shadows, which is exactly where a mismatch shows).
    Srgb,
    Gamma22,
    Gamma24,
    Gamma26,
    /// Rec.709 camera OETF.
    Rec709,
    AcesCc,
    AcesCct,
    /// ARRI LogC3, EI 800. The other EIs differ, and are not modelled: an EI
    /// picker that silently used EI 800 would be worse than not offering one.
    LogC3,
    /// ARRI LogC4. Exposure-independent by construction.
    LogC4,
    SLog3,
    /// RED Log3G10 v3 — the current revision, matching the RED SDK.
    Log3G10,
    VLog,
    BmdFilmGen5,
    /// DaVinci Intermediate: Resolve's log for a DWG timeline.
    DaVinciIntermediate,
    /// Kodak Cineon printing density, 95 black / 685 white / 0.6 gamma.
    Cineon,
    /// ACES 1.2 `Output - Rec.709`: RRT + ODT between ACES2065-1 and BT.1886
    /// code values. NOT per channel, the one kind of "curve" that is not:
    /// it works on whole pixels ([`Self::decode_rgb`]), which is why it is
    /// outside [`Self::ALL`]. The scalar methods give its neutral axis. Pair
    /// it with [`Primaries::Ap0`].
    AcesOutRec709,
    /// ACES 1.2 `Output - sRGB`, the same with the sRGB curve.
    AcesOutSrgb,
}

impl Curve {
    /// The per-channel curves.
    pub const ALL: [Self; 16] = [
        Self::Linear,
        Self::Srgb,
        Self::Gamma22,
        Self::Gamma24,
        Self::Gamma26,
        Self::Rec709,
        Self::AcesCc,
        Self::AcesCct,
        Self::LogC3,
        Self::LogC4,
        Self::SLog3,
        Self::Log3G10,
        Self::VLog,
        Self::BmdFilmGen5,
        Self::DaVinciIntermediate,
        Self::Cineon,
    ];

    /// The ACES rendering this stands for, if it is one of those.
    fn aces_output(self) -> Option<crate::viewtf::aces::Output> {
        use crate::viewtf::aces::Output;
        match self {
            Self::AcesOutRec709 => Some(Output::Rec709),
            Self::AcesOutSrgb => Some(Output::Srgb),
            _ => None,
        }
    }

    /// Encoded -> scene-linear, a whole pixel.
    pub fn decode_rgb(self, c: [f32; 3]) -> [f32; 3] {
        match self.aces_output() {
            Some(o) => crate::viewtf::aces::output_to_ap0(c, o),
            None => c.map(|v| self.to_linear(v)),
        }
    }

    /// Scene-linear -> encoded, a whole pixel.
    pub fn encode_rgb(self, c: [f32; 3]) -> [f32; 3] {
        match self.aces_output() {
            Some(o) => crate::viewtf::aces::output_from_ap0(c, o),
            None => c.map(|v| self.from_linear(v)),
        }
    }

    /// Encoded -> scene-linear.
    pub fn to_linear(self, x: f32) -> f32 {
        match self {
            Self::Linear => x,
            Self::Srgb => signed(x, |v| {
                if v <= 0.04045 {
                    v / 12.92
                } else {
                    ((v + 0.055) / 1.055).powf(2.4)
                }
            }),
            Self::Gamma22 => signed(x, |v| v.powf(2.2)),
            Self::Gamma24 => signed(x, |v| v.powf(2.4)),
            Self::Gamma26 => signed(x, |v| v.powf(2.6)),
            Self::Rec709 => signed(x, |v| {
                if v < 0.081 {
                    v / 4.5
                } else {
                    ((v + 0.099) / 1.099).powf(1.0 / 0.45)
                }
            }),
            Self::AcesCc => {
                if x < -0.301369863 {
                    (2f32.powf(x * 17.52 - 9.72) - 2f32.powi(-16)) * 2.0
                } else {
                    2f32.powf(x * 17.52 - 9.72)
                }
            }
            Self::AcesCct => {
                if x <= 0.155251141552511 {
                    (x - 0.0729055341958355) / 10.5402377416545
                } else {
                    2f32.powf(x * 17.52 - 9.72)
                }
            }
            Self::LogC3 => {
                const CUT: f32 = 0.010591;
                const A: f32 = 5.555556;
                const B: f32 = 0.052272;
                const C: f32 = 0.247190;
                const D: f32 = 0.385537;
                const E: f32 = 5.367655;
                const F: f32 = 0.092809;
                // The cut is stated in LINEAR; its encoded position is where
                // the two segments meet.
                if x > E * CUT + F {
                    (10f32.powf((x - D) / C) - B) / A
                } else {
                    (x - F) / E
                }
            }
            Self::LogC4 => {
                let (a, b, c, s, t) = logc4_consts();
                if x < 0.0 {
                    x * s + t
                } else {
                    (2f32.powf(14.0 * (x - c) / b + 6.0) - 64.0) / a
                }
            }
            Self::SLog3 => {
                if x >= 171.2102946929 / 1023.0 {
                    10f32.powf((x * 1023.0 - 420.0) / 261.5) * 0.19 - 0.01
                } else {
                    (x * 1023.0 - 95.0) * 0.01125 / (171.2102946929 - 95.0)
                }
            }
            Self::Log3G10 => {
                const A: f32 = 0.224282;
                const B: f32 = 155.975327;
                const C: f32 = 0.01;
                const G: f32 = 15.1927;
                let v = if x < 0.0 { x / G } else { (10f32.powf(x / A) - 1.0) / B };
                v - C
            }
            Self::VLog => {
                const B: f32 = 0.00873;
                const C: f32 = 0.241514;
                const D: f32 = 0.598206;
                if x < 0.181 {
                    (x - 0.125) / 5.6
                } else {
                    10f32.powf((x - D) / C) - B
                }
            }
            Self::BmdFilmGen5 => {
                let (a, b, c, d, e, cut) = bmd_consts();
                if x < d * cut + e {
                    (x - e) / d
                } else {
                    ((x - c) / a).exp() - b
                }
            }
            Self::DaVinciIntermediate => {
                let (a, b, c, m, _, log_cut) = di_consts();
                if x <= log_cut {
                    x / m
                } else {
                    2f32.powf(x / c - b) - a
                }
            }
            Self::Cineon => cineon_to_linear(x * 1023.0),
            Self::AcesOutRec709 | Self::AcesOutSrgb => self.decode_rgb([x; 3])[1],
        }
    }

    /// Scene-linear -> encoded.
    pub fn from_linear(self, x: f32) -> f32 {
        match self {
            Self::Linear => x,
            Self::Srgb => signed(x, |v| {
                if v <= 0.0031308 {
                    v * 12.92
                } else {
                    1.055 * v.powf(1.0 / 2.4) - 0.055
                }
            }),
            Self::Gamma22 => signed(x, |v| v.powf(1.0 / 2.2)),
            Self::Gamma24 => signed(x, |v| v.powf(1.0 / 2.4)),
            Self::Gamma26 => signed(x, |v| v.powf(1.0 / 2.6)),
            Self::Rec709 => signed(x, |v| {
                if v < 0.018 {
                    v * 4.5
                } else {
                    1.099 * v.powf(0.45) - 0.099
                }
            }),
            Self::AcesCc => {
                let v = if x <= 0.0 {
                    2f32.powi(-16)
                } else if x < 2f32.powi(-15) {
                    2f32.powi(-16) + x * 0.5
                } else {
                    x
                };
                (v.log2() + 9.72) / 17.52
            }
            Self::AcesCct => {
                if x <= 0.0078125 {
                    10.5402377416545 * x + 0.0729055341958355
                } else {
                    (x.log2() + 9.72) / 17.52
                }
            }
            Self::LogC3 => {
                const CUT: f32 = 0.010591;
                const A: f32 = 5.555556;
                const B: f32 = 0.052272;
                const C: f32 = 0.247190;
                const D: f32 = 0.385537;
                const E: f32 = 5.367655;
                const F: f32 = 0.092809;
                if x > CUT {
                    C * (A * x + B).log10() + D
                } else {
                    E * x + F
                }
            }
            Self::LogC4 => {
                let (a, b, c, s, t) = logc4_consts();
                if x < t {
                    (x - t) / s
                } else {
                    ((a * x + 64.0).log2() - 6.0) / 14.0 * b + c
                }
            }
            Self::SLog3 => {
                if x >= 0.01125000 {
                    (420.0 + ((x + 0.01) / 0.19).log10() * 261.5) / 1023.0
                } else {
                    (x * (171.2102946929 - 95.0) / 0.01125 + 95.0) / 1023.0
                }
            }
            Self::Log3G10 => {
                const A: f32 = 0.224282;
                const B: f32 = 155.975327;
                const C: f32 = 0.01;
                const G: f32 = 15.1927;
                let v = x + C;
                if v < 0.0 {
                    v * G
                } else {
                    A * (v * B + 1.0).log10()
                }
            }
            Self::VLog => {
                const B: f32 = 0.00873;
                const C: f32 = 0.241514;
                const D: f32 = 0.598206;
                if x < 0.01 {
                    5.6 * x + 0.125
                } else {
                    C * (x + B).log10() + D
                }
            }
            Self::BmdFilmGen5 => {
                let (a, b, c, d, e, cut) = bmd_consts();
                if x < cut {
                    d * x + e
                } else {
                    a * (x + b).ln() + c
                }
            }
            Self::DaVinciIntermediate => {
                let (a, b, c, m, lin_cut, _) = di_consts();
                if x <= lin_cut {
                    x * m
                } else {
                    ((x + a).log2() + b) * c
                }
            }
            Self::Cineon => linear_to_cineon(x) / 1023.0,
            Self::AcesOutRec709 | Self::AcesOutSrgb => self.encode_rgb([x; 3])[1],
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Linear => "Linear",
            Self::Srgb => "sRGB",
            Self::Gamma22 => "Gamma 2.2",
            Self::Gamma24 => "Gamma 2.4",
            Self::Gamma26 => "Gamma 2.6",
            Self::Rec709 => "Rec.709 OETF",
            Self::AcesCc => "ACEScc",
            Self::AcesCct => "ACEScct",
            Self::LogC3 => "ARRI LogC3 (EI 800)",
            Self::LogC4 => "ARRI LogC4",
            Self::SLog3 => "Sony S-Log3",
            Self::Log3G10 => "RED Log3G10",
            Self::VLog => "Panasonic V-Log",
            Self::BmdFilmGen5 => "Blackmagic Film Gen5",
            Self::DaVinciIntermediate => "DaVinci Intermediate",
            Self::Cineon => "Cineon (printing density)",
            Self::AcesOutRec709 => "ACES 1.2 Output - Rec.709",
            Self::AcesOutSrgb => "ACES 1.2 Output - sRGB",
        }
    }

    /// Short identifier used to name the generated WGSL helpers.
    fn slug(self) -> &'static str {
        match self {
            Self::Linear => "lin",
            Self::Srgb => "srgb",
            Self::Gamma22 => "g22",
            Self::Gamma24 => "g24",
            Self::Gamma26 => "g26",
            Self::Rec709 => "r709",
            Self::AcesCc => "acescc",
            Self::AcesCct => "acescct",
            Self::LogC3 => "logc3",
            Self::LogC4 => "logc4",
            Self::SLog3 => "slog3",
            Self::Log3G10 => "log3g10",
            Self::VLog => "vlog",
            Self::BmdFilmGen5 => "bmd5",
            Self::DaVinciIntermediate => "dvi",
            Self::Cineon => "cineon",
            Self::AcesOutRec709 => "aces_out709",
            Self::AcesOutSrgb => "aces_outsrgb",
        }
    }
}

/// Apply `f` symmetrically about zero, so a power law stays defined on the
/// small negatives that come out of a camera gamut conversion instead of
/// turning them into NaN.
fn signed(x: f32, f: impl Fn(f32) -> f32) -> f32 {
    if x < 0.0 {
        -f(-x)
    } else {
        f(x)
    }
}

/// LogC4's constants are defined in terms of each other, so they are solved
/// once here rather than pasted as decimals.
fn logc4_consts() -> (f32, f32, f32, f32, f32) {
    let a = (2f32.powi(18) - 16.0) / 117.45;
    let b = (1023.0 - 95.0) / 1023.0;
    let c = 95.0 / 1023.0;
    let s = (7.0 * std::f32::consts::LN_2 * 2f32.powf(7.0 - 14.0 * c / b)) / (a * b);
    let t = (2f32.powf(14.0 * (-c / b) + 6.0) - 64.0) / a;
    (a, b, c, s, t)
}

/// Blackmagic Film Gen5: A ln(x + B) + C above the cut, D x + E below.
fn bmd_consts() -> (f32, f32, f32, f32, f32, f32) {
    (
        0.086_928_76,
        0.005_494_072_4,
        0.530_013_34,
        8.283_606,
        0.092_465_75,
        0.005,
    )
}

/// DaVinci Intermediate, from Blackmagic's published note:
/// `(A, B, C, M, LIN_CUT, LOG_CUT)`, log2 above the cut, linear below.
fn di_consts() -> (f32, f32, f32, f32, f32, f32) {
    (0.0075, 7.0, 0.07329248, 10.44426855, 0.00262409, 0.02740668)
}

/// Cineon code value (0..1023) -> relative linear. Matches `dpx.rs`, which is
/// where this curve first appeared; a DPX and a Cineon-tagged EXR must
/// linearise identically or the same plate reads two ways.
pub fn cineon_to_linear(code: f32) -> f32 {
    const REF_BLACK: f32 = 95.0;
    const REF_WHITE: f32 = 685.0;
    const GAMMA: f32 = 0.6;
    const DENSITY_PER_CODE: f32 = 0.002;
    let gain = 1.0 / (1.0 - 10f32.powf((REF_BLACK - REF_WHITE) * DENSITY_PER_CODE / GAMMA));
    let offset = gain - 1.0;
    let v = 10f32.powf((code - REF_WHITE) * DENSITY_PER_CODE / GAMMA) * gain - offset;
    v.max(0.0)
}

/// Relative linear -> Cineon code value (0..1023).
pub fn linear_to_cineon(x: f32) -> f32 {
    const REF_WHITE: f32 = 685.0;
    const GAMMA: f32 = 0.6;
    const DENSITY_PER_CODE: f32 = 0.002;
    let gain = 1.0 / (1.0 - 10f32.powf((95.0 - REF_WHITE) * DENSITY_PER_CODE / GAMMA));
    let offset = gain - 1.0;
    let v = ((x + offset) / gain).max(1e-10);
    REF_WHITE + v.log10() * GAMMA / DENSITY_PER_CODE
}

// ---------------------------------------------------------------------------
// Colour spaces and transforms
// ---------------------------------------------------------------------------

/// A colour encoding: primaries plus transfer curve. Storing the pair rather
/// than a flat list of named spaces means an odd combination ("Rec.709
/// primaries, Cineon curve") is expressible instead of being a missing enum
/// variant somebody has to add under deadline.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Colorspace {
    pub primaries: Primaries,
    pub curve: Curve,
}

impl Default for Colorspace {
    /// Linear Rec.709 — the legacy working space, for the reason spelled out
    /// on [`Primaries::Rec709`]. `Colorspace::ACESCG` is what a new comp
    /// should be given, explicitly.
    fn default() -> Self {
        Self::LINEAR_REC709
    }
}

impl Colorspace {
    pub const fn new(primaries: Primaries, curve: Curve) -> Self {
        Self { primaries, curve }
    }

    pub const ACESCG: Self = Self::new(Primaries::Ap1, Curve::Linear);
    pub const ACES2065: Self = Self::new(Primaries::Ap0, Curve::Linear);
    pub const ACESCC: Self = Self::new(Primaries::Ap1, Curve::AcesCc);
    pub const ACESCCT: Self = Self::new(Primaries::Ap1, Curve::AcesCct);
    pub const LINEAR_REC709: Self = Self::new(Primaries::Rec709, Curve::Linear);
    pub const SRGB: Self = Self::new(Primaries::Rec709, Curve::Srgb);
    pub const REC709: Self = Self::new(Primaries::Rec709, Curve::Rec709);
    /// Display-referred Rec.709: BT.1886 with a zero black level, which is a
    /// pure 2.4 power. What a graded Rec.709 deliverable holds, as opposed to
    /// [`Self::REC709`]'s camera OETF.
    pub const REC709_BT1886: Self = Self::new(Primaries::Rec709, Curve::Gamma24);
    pub const LINEAR_P3D65: Self = Self::new(Primaries::P3D65, Curve::Linear);
    pub const LINEAR_REC2020: Self = Self::new(Primaries::Rec2020, Curve::Linear);
    pub const LOGC3: Self = Self::new(Primaries::ArriWideGamut3, Curve::LogC3);
    pub const LOGC4: Self = Self::new(Primaries::ArriWideGamut4, Curve::LogC4);
    pub const SLOG3: Self = Self::new(Primaries::SGamut3Cine, Curve::SLog3);
    pub const LOG3G10: Self = Self::new(Primaries::RedWideGamut, Curve::Log3G10);
    pub const VLOG: Self = Self::new(Primaries::VGamut, Curve::VLog);
    pub const BMD_GEN5: Self = Self::new(Primaries::BmdWideGamutGen5, Curve::BmdFilmGen5);
    pub const CINEON: Self = Self::new(Primaries::Rec709, Curve::Cineon);
    /// Resolve's colour-managed timeline: DaVinci Wide Gamut, Intermediate
    /// log. What a LUT exported from a DWG project expects and returns.
    pub const DAVINCI_INTERMEDIATE: Self = Self::new(Primaries::DaVinciWideGamut, Curve::DaVinciIntermediate);
    /// What an ACES 1.2 `Rec.709` view put on screen: as an input, the
    /// picture goes back through the inverse ODT and RRT to scene-linear.
    pub const ACES_OUTPUT_REC709: Self = Self::new(Primaries::Ap0, Curve::AcesOutRec709);
    /// The same for the `sRGB` view.
    pub const ACES_OUTPUT_SRGB: Self = Self::new(Primaries::Ap0, Curve::AcesOutSrgb);

    /// The presets the UI offers, in menu order. Camera spaces come as the
    /// vendor ships them (their own primaries AND their own curve) because
    /// that pairing is what an IDT means; mixing them is possible but is
    /// never what someone picking "S-Log3" wants.
    pub const PRESETS: [(&'static str, Self); 20] = [
        ("ACEScg", Self::ACESCG),
        ("ACES2065-1", Self::ACES2065),
        ("ACEScc", Self::ACESCC),
        ("ACEScct", Self::ACESCCT),
        ("Linear Rec.709", Self::LINEAR_REC709),
        ("sRGB", Self::SRGB),
        ("Rec.709 (BT.1886 / 2.4)", Self::REC709_BT1886),
        ("Rec.709 (camera OETF)", Self::REC709),
        ("Linear P3-D65", Self::LINEAR_P3D65),
        ("Linear Rec.2020", Self::LINEAR_REC2020),
        ("ARRI LogC3 / AWG3", Self::LOGC3),
        ("ARRI LogC4 / AWG4", Self::LOGC4),
        ("Sony S-Log3 / S-Gamut3.Cine", Self::SLOG3),
        ("RED Log3G10 / REDWideGamut", Self::LOG3G10),
        ("Panasonic V-Log / V-Gamut", Self::VLOG),
        ("Blackmagic Film Gen5", Self::BMD_GEN5),
        ("DaVinci Intermediate / DWG", Self::DAVINCI_INTERMEDIATE),
        ("Cineon", Self::CINEON),
        // Appended: the colour menus address presets by position.
        ("Output - Rec.709 (ACES 1.2)", Self::ACES_OUTPUT_REC709),
        ("Output - sRGB (ACES 1.2)", Self::ACES_OUTPUT_SRGB),
    ];

    pub fn label(&self) -> String {
        for (name, cs) in Self::PRESETS {
            if cs == *self {
                return name.to_string();
            }
        }
        format!("{} / {}", self.primaries.label(), self.curve.label())
    }

    /// True when this is scene-linear — the only kind of space a comp can
    /// safely work in.
    pub fn is_scene_linear(&self) -> bool {
        self.curve == Curve::Linear
    }
}

/// What a source node's pixels are, as the user declares them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum InputSpace {
    /// Already in the working space: no transform is emitted at all.
    Working,
    Explicit(Colorspace),
    /// Follow the per-format default ([`FormatDefaults`]) for this file's
    /// extension. The APP resolves it to `Working` / `Explicit` before the
    /// engine sees a graph (the table is a user setting the engine has no
    /// business owning); anything that reaches the engine unresolved is
    /// treated as `Working`. New Reads start here, so the table the user
    /// set up decides — setups saved before it existed carry an explicit
    /// `Working` and keep rendering as they did.
    #[default]
    Default,
}

impl InputSpace {
    pub fn resolve(self, working: Colorspace) -> Colorspace {
        match self {
            Self::Working | Self::Default => working,
            Self::Explicit(cs) => cs,
        }
    }

    pub fn label(&self, working: Colorspace) -> String {
        match self {
            Self::Working => format!("working ({})", working.label()),
            Self::Explicit(cs) => cs.label(),
            Self::Default => "default".into(),
        }
    }

    /// The space a file of this kind is *most likely* in, used to prefill the
    /// picker when a plate is loaded. A guess that is shown in the UI, never
    /// one that is applied silently.
    pub fn guess_for_extension(ext: &str) -> Option<Colorspace> {
        match ext.to_ascii_lowercase().as_str() {
            // Scene-linear containers: assume the comp's own space rather
            // than inventing one.
            "exr" | "hdr" => None,
            // Log film scans.
            "dpx" | "cin" => Some(Colorspace::CINEON),
            // Delivery movies. Rec.709 is what a camera or an edit system
            // writes by default; a log clip in a .mov is common enough that
            // this stays a GUESS shown in the UI, never an applied default.
            "mov" | "mp4" | "m4v" | "mxf" | "avi" | "mkv" => Some(Colorspace::REC709),
            // Everything display-referred that a browser would show.
            "png" | "jpg" | "jpeg" | "tif" | "tiff" | "tga" | "bmp" | "webp" => {
                Some(Colorspace::SRGB)
            }
            _ => None,
        }
    }
}

/// The per-format colour defaults: what a Read of each file format is
/// assumed to hold, and what a Write of each format encodes into. A Read set
/// to [`InputSpace::Default`] (and a Write with `output_default`) follows
/// this table; anything set explicitly on the node wins.
///
/// Keyed by FORMAT, not extension: `jpg`/`jpeg` are one row, and every movie
/// container is one `movie` row (see [`format_key`]).
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FormatDefaults {
    #[cfg_attr(feature = "serde", serde(default))]
    pub input: std::collections::BTreeMap<String, InputSpace>,
    /// `None` = the working space, unconverted.
    #[cfg_attr(feature = "serde", serde(default))]
    pub output: std::collections::BTreeMap<String, Option<Colorspace>>,
}

/// The formats a Read can decode, as rows of the defaults table.
pub const INPUT_FORMATS: [&str; 5] = ["exr", "dpx", "png", "jpg", "movie"];
/// The formats a Write can encode.
pub const OUTPUT_FORMATS: [&str; 2] = ["exr", "movie"];

/// A movie Write's built-in output space: display-referred Rec.709 (BT.1886,
/// a 2.4 power), what a ProRes or H.264 review file is expected to hold. A
/// movie in the scene-linear working space would play back far too dark.
pub const MOVIE_OUTPUT: Colorspace = Colorspace::REC709_BT1886;

/// The defaults-table row a file extension belongs to.
pub fn format_key(ext: &str) -> Option<&'static str> {
    match ext.to_ascii_lowercase().as_str() {
        "exr" => Some("exr"),
        "dpx" => Some("dpx"),
        "png" => Some("png"),
        "jpg" | "jpeg" => Some("jpg"),
        "mov" | "mp4" | "m4v" | "mxf" | "avi" | "mkv" => Some("movie"),
        _ => None,
    }
}

/// The table row for a path (a file, or a `####` pattern — only the
/// extension is read).
fn path_format(path: &str) -> Option<&'static str> {
    std::path::Path::new(path).extension().and_then(|e| e.to_str()).and_then(format_key)
}

impl Default for FormatDefaults {
    /// The built-in table: the per-extension guesses, applied.
    fn default() -> Self {
        let input = INPUT_FORMATS
            .iter()
            .map(|f| {
                let ext = if *f == "movie" { "mov" } else { f };
                let v = match InputSpace::guess_for_extension(ext) {
                    Some(cs) => InputSpace::Explicit(cs),
                    None => InputSpace::Working,
                };
                (f.to_string(), v)
            })
            .collect();
        let output = OUTPUT_FORMATS
            .iter()
            .map(|f| (f.to_string(), (*f == "movie").then_some(MOVIE_OUTPUT)))
            .collect();
        Self { input, output }
    }
}

impl FormatDefaults {
    /// The default for one format row. Never `Default` (that would loop):
    /// a missing or self-referential row means `Working`.
    pub fn input_for_format(&self, format: &str) -> InputSpace {
        match self.input.get(format) {
            Some(InputSpace::Default) | None => InputSpace::Working,
            Some(v) => *v,
        }
    }

    /// What a Read of `path` should assume when set to `Default`.
    pub fn input_for(&self, path: &str) -> InputSpace {
        path_format(path).map_or(InputSpace::Working, |f| self.input_for_format(f))
    }

    pub fn output_for_format(&self, format: &str) -> Option<Colorspace> {
        match self.output.get(format) {
            Some(v) => *v,
            // Prefs saved before movies were written have no row: the
            // built-in default, not the working space.
            None if format == "movie" => Some(MOVIE_OUTPUT),
            None => None,
        }
    }

    /// What a Write to `path` encodes into when following the default.
    /// A path with no known extension is an EXR — the only format a Write
    /// produces today.
    pub fn output_for(&self, path: &str) -> Option<Colorspace> {
        self.output_for_format(path_format(path).unwrap_or("exr"))
    }
}

/// A compiled conversion: decode the source curve, change primaries, encode
/// the destination curve. Every built-in transform has this shape.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transform {
    pub decode: Curve,
    pub matrix: Mat3,
    pub encode: Curve,
}

impl Transform {
    /// The transform taking `src` pixels to `dst` pixels.
    pub fn new(src: Colorspace, dst: Colorspace) -> Self {
        Self {
            decode: src.curve,
            matrix: primaries_matrix(src.primaries, dst.primaries),
            encode: dst.curve,
        }
    }

    /// Nothing to do — emit no code and bind no textures.
    pub fn is_identity(&self) -> bool {
        self.decode == Curve::Linear
            && self.encode == Curve::Linear
            && mat3_is_identity(self.matrix)
    }

    /// CPU evaluation. Used by the tests, by the colour picker, and by
    /// anything that reads pixels back and has to reason about them.
    pub fn apply(&self, rgb: [f32; 3]) -> [f32; 3] {
        self.encode.encode_rgb(mat3_apply(self.matrix, self.decode.decode_rgb(rgb)))
    }

    /// Emit `fn {name}(c: vec3f) -> vec3f`, collecting the curve helpers it
    /// needs into `helpers` (keyed by function name, so two readers sharing a
    /// curve emit it once).
    pub fn wgsl(&self, name: &str, helpers: &mut std::collections::BTreeMap<String, String>) -> String {
        let mut body = String::from("    var v = c;\n");
        if self.decode.aces_output().is_some() || self.encode.aces_output().is_some() {
            // Named to sort ahead of the `cs_aces_out*` functions that call it.
            helpers.entry("cs_aces_0lib".into()).or_insert_with(crate::viewtf::aces::lib_wgsl);
        }
        if let Some((f, def)) = curve_wgsl(self.decode, true) {
            helpers.entry(f.clone()).or_insert(def);
            let _ = writeln!(body, "    v = {f}(v);");
        }
        if !mat3_is_identity(self.matrix) {
            let m = self.matrix;
            let _ = writeln!(
                body,
                "    v = mat3x3<f32>(\n        vec3f({:.9}, {:.9}, {:.9}),\n        vec3f({:.9}, {:.9}, {:.9}),\n        vec3f({:.9}, {:.9}, {:.9})) * v;",
                // WGSL mat3x3 is column-major: each vec3f is a COLUMN.
                m[0][0], m[1][0], m[2][0],
                m[0][1], m[1][1], m[2][1],
                m[0][2], m[1][2], m[2][2],
            );
        }
        if let Some((f, def)) = curve_wgsl(self.encode, false) {
            helpers.entry(f.clone()).or_insert(def);
            let _ = writeln!(body, "    v = {f}(v);");
        }
        format!("fn {name}(c: vec3f) -> vec3f {{\n{body}    return v;\n}}\n")
    }
}

/// WGSL for one curve in one direction: `(function name, definition)`, or
/// `None` for Linear, which emits nothing.
///
/// Every curve but the `AcesOut*` pair is componentwise on a `vec3f`, and
/// every comparison is
/// parenthesised — naga parses a bare `a < b` in some positions as the start
/// of a template argument list, which fails with a message that does not
/// mention the comparison at all.
fn curve_wgsl(c: Curve, to_linear: bool) -> Option<(String, String)> {
    if c == Curve::Linear {
        return None;
    }
    let name = format!("cs_{}_{}", c.slug(), if to_linear { "to_lin" } else { "from_lin" });
    let body = match (c, to_linear) {
        (Curve::Linear, _) => unreachable!(),

        (Curve::Srgb, true) => "    let a = abs(c);
    let lo = a / 12.92;
    let hi = pow((a + vec3f(0.055)) / 1.055, vec3f(2.4));
    return sign(c) * select(hi, lo, (a <= vec3f(0.04045)));"
            .to_string(),
        (Curve::Srgb, false) => "    let a = abs(c);
    let lo = a * 12.92;
    let hi = 1.055 * pow(a, vec3f(1.0 / 2.4)) - vec3f(0.055);
    return sign(c) * select(hi, lo, (a <= vec3f(0.0031308)));"
            .to_string(),

        (Curve::Gamma22 | Curve::Gamma24 | Curve::Gamma26, _) => {
            let g: f32 = match c {
                Curve::Gamma22 => 2.2,
                Curve::Gamma24 => 2.4,
                _ => 2.6,
            };
            let e = if to_linear { g } else { 1.0 / g };
            format!("    return sign(c) * pow(abs(c), vec3f({e:.9}));")
        }

        (Curve::Rec709, true) => "    let a = abs(c);
    let lo = a / 4.5;
    let hi = pow((a + vec3f(0.099)) / 1.099, vec3f(1.0 / 0.45));
    return sign(c) * select(hi, lo, (a < vec3f(0.081)));"
            .to_string(),
        (Curve::Rec709, false) => "    let a = abs(c);
    let lo = a * 4.5;
    let hi = 1.099 * pow(a, vec3f(0.45)) - vec3f(0.099);
    return sign(c) * select(hi, lo, (a < vec3f(0.018)));"
            .to_string(),

        (Curve::AcesCc, true) => "    let hi = exp2(c * 17.52 - vec3f(9.72));
    let lo = (exp2(c * 17.52 - vec3f(9.72)) - vec3f(0.0000152587890625)) * 2.0;
    return select(hi, lo, (c < vec3f(-0.301369863)));"
            .to_string(),
        (Curve::AcesCc, false) => "    let g = max(c, vec3f(0.0));
    let v = select(g, vec3f(0.0000152587890625) + g * 0.5, (g < vec3f(0.000030517578125)));
    let w = select(v, vec3f(0.0000152587890625), (c <= vec3f(0.0)));
    return (log2(w) + vec3f(9.72)) / 17.52;"
            .to_string(),

        (Curve::AcesCct, true) => "    let lo = (c - vec3f(0.0729055341958355)) / 10.5402377416545;
    let hi = exp2(c * 17.52 - vec3f(9.72));
    return select(hi, lo, (c <= vec3f(0.155251141552511)));"
            .to_string(),
        (Curve::AcesCct, false) => "    let lo = 10.5402377416545 * c + vec3f(0.0729055341958355);
    let hi = (log2(max(c, vec3f(1e-10))) + vec3f(9.72)) / 17.52;
    return select(hi, lo, (c <= vec3f(0.0078125)));"
            .to_string(),

        (Curve::LogC3, true) => "    let hi = (pow(vec3f(10.0), (c - vec3f(0.385537)) / 0.247190) - vec3f(0.052272)) / 5.555556;
    let lo = (c - vec3f(0.092809)) / 5.367655;
    return select(hi, lo, (c <= vec3f(5.367655 * 0.010591 + 0.092809)));"
            .to_string(),
        (Curve::LogC3, false) => "    let hi = 0.247190 * log(5.555556 * c + vec3f(0.052272)) / log(10.0) + vec3f(0.385537);
    let lo = 5.367655 * c + vec3f(0.092809);
    return select(hi, lo, (c <= vec3f(0.010591)));"
            .to_string(),

        (Curve::LogC4, dir) => {
            let (a, b, cc, s, t) = logc4_consts();
            if dir {
                format!(
                    "    let hi = (exp2(14.0 * (c - vec3f({cc:.9})) / {b:.9} + vec3f(6.0)) - vec3f(64.0)) / {a:.6};
    let lo = c * {s:.9} + vec3f({t:.9});
    return select(hi, lo, (c < vec3f(0.0)));"
                )
            } else {
                format!(
                    "    let hi = (log2(max({a:.6} * c + vec3f(64.0), vec3f(1e-6))) - vec3f(6.0)) / 14.0 * {b:.9} + vec3f({cc:.9});
    let lo = (c - vec3f({t:.9})) / {s:.9};
    return select(hi, lo, (c < vec3f({t:.9})));"
                )
            }
        }

        (Curve::SLog3, true) => "    let hi = pow(vec3f(10.0), (c * 1023.0 - vec3f(420.0)) / 261.5) * 0.19 - vec3f(0.01);
    let lo = (c * 1023.0 - vec3f(95.0)) * 0.01125 / (171.2102946929 - 95.0);
    return select(hi, lo, (c < vec3f(171.2102946929 / 1023.0)));"
            .to_string(),
        (Curve::SLog3, false) => "    let hi = (vec3f(420.0) + log(max((c + vec3f(0.01)) / 0.19, vec3f(1e-10))) / log(10.0) * 261.5) / 1023.0;
    let lo = (c * ((171.2102946929 - 95.0) / 0.01125) + vec3f(95.0)) / 1023.0;
    return select(hi, lo, (c < vec3f(0.01125)));"
            .to_string(),

        (Curve::Log3G10, true) => "    let hi = (pow(vec3f(10.0), c / 0.224282) - vec3f(1.0)) / 155.975327;
    let lo = c / 15.1927;
    return select(hi, lo, (c < vec3f(0.0))) - vec3f(0.01);"
            .to_string(),
        (Curve::Log3G10, false) => "    let v = c + vec3f(0.01);
    let hi = 0.224282 * log(max(v * 155.975327 + vec3f(1.0), vec3f(1e-10))) / log(10.0);
    let lo = v * 15.1927;
    return select(hi, lo, (v < vec3f(0.0)));"
            .to_string(),

        (Curve::VLog, true) => "    let hi = pow(vec3f(10.0), (c - vec3f(0.598206)) / 0.241514) - vec3f(0.00873);
    let lo = (c - vec3f(0.125)) / 5.6;
    return select(hi, lo, (c < vec3f(0.181)));"
            .to_string(),
        (Curve::VLog, false) => "    let hi = 0.241514 * log(max(c + vec3f(0.00873), vec3f(1e-10))) / log(10.0) + vec3f(0.598206);
    let lo = 5.6 * c + vec3f(0.125);
    return select(hi, lo, (c < vec3f(0.01)));"
            .to_string(),

        (Curve::BmdFilmGen5, dir) => {
            let (a, b, cc, d, e, cut) = bmd_consts();
            if dir {
                format!(
                    "    let hi = exp((c - vec3f({cc:.9})) / {a:.9}) - vec3f({b:.9});
    let lo = (c - vec3f({e:.9})) / {d:.9};
    return select(hi, lo, (c < vec3f({:.9})));",
                    d * cut + e
                )
            } else {
                format!(
                    "    let hi = {a:.9} * log(max(c + vec3f({b:.9}), vec3f(1e-10))) + vec3f({cc:.9});
    let lo = {d:.9} * c + vec3f({e:.9});
    return select(hi, lo, (c < vec3f({cut:.9})));"
                )
            }
        }

        (Curve::DaVinciIntermediate, dir) => {
            let (a, b, cc, m, lin_cut, log_cut) = di_consts();
            if dir {
                format!(
                    "    let hi = exp2(c / {cc:.9} - vec3f({b:.9})) - vec3f({a:.9});
    let lo = c / {m:.9};
    return select(hi, lo, (c <= vec3f({log_cut:.9})));"
                )
            } else {
                format!(
                    "    let hi = (log2(max(c + vec3f({a:.9}), vec3f(1e-10))) + vec3f({b:.9})) * {cc:.9};
    let lo = c * {m:.9};
    return select(hi, lo, (c <= vec3f({lin_cut:.9})));"
                )
            }
        }

        (Curve::AcesOutRec709, true) => "    return aces_out_to_ap0(pow(max(c, vec3f(0.0)), vec3f(2.4)));".to_string(),
        (Curve::AcesOutRec709, false) => "    return pow(aces_ap0_to_out(c), vec3f(1.0 / 2.4));".to_string(),
        (Curve::AcesOutSrgb, true) => "    let a = max(c, vec3f(0.0));
    let lo = a / 12.92;
    let hi = pow((a + vec3f(0.055)) / 1.055, vec3f(2.4));
    return aces_out_to_ap0(select(hi, lo, (a <= vec3f(0.04045))));"
            .to_string(),
        (Curve::AcesOutSrgb, false) => "    let a = aces_ap0_to_out(c);
    let lo = a * 12.92;
    let hi = 1.055 * pow(a, vec3f(1.0 / 2.4)) - vec3f(0.055);
    return select(hi, lo, (a <= vec3f(0.0031308)));"
            .to_string(),

        (Curve::Cineon, dir) => {
            // Gain/offset are solved from the reference black/white so the
            // shader cannot drift from `cineon_to_linear`.
            let gain = 1.0f32 / (1.0 - 10f32.powf((95.0 - 685.0) * 0.002 / 0.6));
            let offset = gain - 1.0;
            if dir {
                format!(
                    "    let code = c * 1023.0;
    return max(pow(vec3f(10.0), (code - vec3f(685.0)) * (0.002 / 0.6)) * {gain:.9} - vec3f({offset:.9}), vec3f(0.0));"
                )
            } else {
                format!(
                    "    let v = max((c + vec3f({offset:.9})) / {gain:.9}, vec3f(1e-10));
    return (vec3f(685.0) + log(v) / log(10.0) * (0.6 / 0.002)) / 1023.0;"
                )
            }
        }
    };
    Some((name.clone(), format!("fn {name}(c: vec3f) -> vec3f {{\n{body}\n}}\n")))
}

// ---------------------------------------------------------------------------
// Small matrix maths
// ---------------------------------------------------------------------------

fn to_f32(m: M3) -> Mat3 {
    let mut o = [[0.0f32; 3]; 3];
    for (r, row) in o.iter_mut().enumerate() {
        for (c, cell) in row.iter_mut().enumerate() {
            *cell = m[r][c] as f32;
        }
    }
    o
}

fn mat3_mul64(a: M3, b: M3) -> M3 {
    let mut o = [[0.0f64; 3]; 3];
    for (i, row) in o.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    o
}

fn mat3_apply64(m: M3, v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

fn mat3_inverse64(m: M3) -> M3 {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    // A singular set of primaries is a typo, not a runtime condition; falling
    // back to the identity keeps the app alive and makes the wrongness
    // visible rather than crashing on a config edit.
    if det.abs() < 1e-12 {
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }
    let inv = 1.0 / det;
    [
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv,
        ],
    ]
}

/// Row-major 3x3 multiply, `a * b`.
pub fn mat3_mul(a: Mat3, b: Mat3) -> Mat3 {
    let mut o = [[0.0f32; 3]; 3];
    for (i, row) in o.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    o
}

pub fn mat3_apply(m: Mat3, v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

pub fn mat3_inverse(m: Mat3) -> Mat3 {
    let mut d = [[0.0f64; 3]; 3];
    for (r, row) in d.iter_mut().enumerate() {
        for (c, cell) in row.iter_mut().enumerate() {
            *cell = m[r][c] as f64;
        }
    }
    to_f32(mat3_inverse64(d))
}

pub fn mat3_is_identity(m: Mat3) -> bool {
    for (r, row) in m.iter().enumerate() {
        for (c, v) in row.iter().enumerate() {
            let want = if r == c { 1.0 } else { 0.0 };
            if (v - want).abs() > 1e-6 {
                return false;
            }
        }
    }
    true
}

/// Row-major 3x3 -> WGSL `mat3x3<f32>` storage for a uniform: column-major,
/// each column padded to 16 bytes.
pub fn mat3_columns(m: Mat3) -> [[f32; 4]; 3] {
    [
        [m[0][0], m[1][0], m[2][0], 0.0],
        [m[0][1], m[1][1], m[2][1], 0.0],
        [m[0][2], m[1][2], m[2][2], 0.0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32, tol: f32, what: &str) {
        assert!((a - b).abs() <= tol, "{what}: {a} != {b} (tol {tol})");
    }

    fn mat_close(got: Mat3, want: Mat3, tol: f32, what: &str) {
        for r in 0..3 {
            for c in 0..3 {
                assert!(
                    (got[r][c] - want[r][c]).abs() <= tol,
                    "{what} [{r}][{c}]: {} != {} (tol {tol})\ngot  {got:?}\nwant {want:?}",
                    got[r][c],
                    want[r][c]
                );
            }
        }
    }

    /// The derivation from chromaticities has to reproduce the matrices the
    /// standards publish. If this drifts, everything downstream is wrong in a
    /// way no picture will show you.
    #[test]
    fn derived_matrices_match_the_published_ones() {
        // SMPTE RP 177 / IEC 61966-2-1.
        mat_close(
            Primaries::Rec709.to_xyz(),
            [
                [0.4123908, 0.3575843, 0.1804808],
                [0.2126390, 0.7151687, 0.0721923],
                [0.0193308, 0.1191948, 0.9505322],
            ],
            1e-5,
            "Rec.709 -> XYZ",
        );
        // SMPTE ST 2065-1 (ACES AP0, ACES white).
        mat_close(
            Primaries::Ap0.to_xyz(),
            [
                [0.9525524, 0.0000000, 0.0000937],
                [0.3439664, 0.7281661, -0.0721325],
                [0.0000000, 0.0000000, 1.0088252],
            ],
            1e-5,
            "AP0 -> XYZ",
        );
        // ACEScg (AP1), as published in S-2014-004.
        mat_close(
            Primaries::Ap1.to_xyz(),
            [
                [0.6624542, 0.1340042, 0.1561877],
                [0.2722287, 0.6740818, 0.0536895],
                [-0.0055746, 0.0040607, 1.0103391],
            ],
            1e-5,
            "AP1 -> XYZ",
        );
        // SMPTE ST 2113 (P3-D65) and ITU-R BT.2020.
        mat_close(
            Primaries::P3D65.to_xyz(),
            [
                [0.4865709, 0.2656677, 0.1982173],
                [0.2289746, 0.6917385, 0.0792869],
                [0.0000000, 0.0451134, 1.0439444],
            ],
            1e-5,
            "P3-D65 -> XYZ",
        );
        mat_close(
            Primaries::Rec2020.to_xyz(),
            [
                [0.6369580, 0.1446169, 0.1688810],
                [0.2627002, 0.6779981, 0.0593017],
                [0.0000000, 0.0280727, 1.0609851],
            ],
            1e-5,
            "Rec.2020 -> XYZ",
        );
    }

    /// The single most consequential property in the whole module: after a
    /// gamut conversion a neutral is still a neutral. Without the Bradford
    /// step, ACEScg -> Rec.709 tints every grey card warm.
    #[test]
    fn neutrals_survive_every_gamut_conversion() {
        for src in Primaries::ALL {
            for dst in Primaries::ALL {
                if src == Primaries::Xyz || dst == Primaries::Xyz {
                    continue;
                }
                let w = mat3_apply(primaries_matrix(src, dst), [1.0, 1.0, 1.0]);
                for (i, v) in w.iter().enumerate() {
                    close(*v, 1.0, 2e-4, &format!("{src:?} -> {dst:?} white ch{i}"));
                }
            }
        }
    }

    /// Specifically the conversion this project now does by default, spelled
    /// out so a regression names itself.
    #[test]
    fn acescg_to_rec709_keeps_white_and_is_not_the_identity() {
        let m = primaries_matrix(Primaries::Ap1, Primaries::Rec709);
        let w = mat3_apply(m, [1.0, 1.0, 1.0]);
        close(w[0], 1.0, 2e-4, "white r");
        close(w[1], 1.0, 2e-4, "white g");
        close(w[2], 1.0, 2e-4, "white b");
        assert!(!mat3_is_identity(m), "AP1 and Rec.709 are different gamuts");
        // AP1 is wider, so its pure primaries land outside Rec.709 — a
        // saturated AP1 red must go NEGATIVE in 709 rather than clipping.
        let red = mat3_apply(m, [1.0, 0.0, 0.0]);
        assert!(red[1] < 0.0 && red[2] < 0.0, "AP1 red out of 709 gamut: {red:?}");
    }

    #[test]
    fn gamut_conversions_round_trip() {
        for src in Primaries::ALL {
            for dst in Primaries::ALL {
                let m = mat3_mul(primaries_matrix(dst, src), primaries_matrix(src, dst));
                mat_close(m, IDENTITY, 1e-4, &format!("{src:?} <-> {dst:?}"));
            }
        }
    }

    #[test]
    fn bradford_maps_source_white_onto_destination_white() {
        for src in [White::D65, White::Aces, White::Dci] {
            for dst in [White::D65, White::Aces, White::Dci] {
                let m = bradford(src, dst);
                let s = src.xyz();
                let got = mat3_apply(m, [s[0] as f32, s[1] as f32, s[2] as f32]);
                let d = dst.xyz();
                for i in 0..3 {
                    close(got[i], d[i] as f32, 1e-4, &format!("{src:?}->{dst:?} ch{i}"));
                }
            }
        }
    }

    /// Rec.709's luma weights are the ones everybody hardcodes; the point of
    /// deriving them is that ACEScg's are DIFFERENT, and a luma key that
    /// used 709 weights on AP1 pixels is quietly wrong.
    #[test]
    fn luma_weights_come_from_the_primaries() {
        let w = Primaries::Rec709.luma_weights();
        close(w[0], 0.2126, 1e-4, "709 r");
        close(w[1], 0.7152, 1e-4, "709 g");
        close(w[2], 0.0722, 1e-4, "709 b");
        let a = Primaries::Ap1.luma_weights();
        close(a[0], 0.2722, 1e-3, "AP1 r");
        close(a[1], 0.6741, 1e-3, "AP1 g");
        close(a[2], 0.0537, 1e-3, "AP1 b");
        assert!(
            (a[0] - w[0]).abs() > 0.05,
            "AP1 and 709 luma weights must differ, else this fix is pointless"
        );
    }

    /// Every vendor states where 18% scene grey lands in their encoding, on
    /// the first page of the spec. Pinning that one number per curve is the
    /// cheapest guard there is against a transposed digit.
    #[test]
    fn mid_grey_lands_where_the_specs_say() {
        let cases: &[(Curve, f32, f32)] = &[
            (Curve::Srgb, 0.4613, 1e-3),
            (Curve::Rec709, 0.4090, 1e-3),
            (Curve::AcesCc, 0.4135884, 1e-5),
            (Curve::AcesCct, 0.4135884, 1e-5),
            (Curve::LogC3, 0.391007, 1e-4),
            (Curve::LogC4, 0.2783, 1e-3),
            (Curve::SLog3, 0.4105572, 1e-5),
            (Curve::Log3G10, 0.3333333, 1e-3),
            (Curve::VLog, 0.4233, 1e-3),
            (Curve::BmdFilmGen5, 0.3835, 1e-3),
            (Curve::DaVinciIntermediate, 0.336043, 1e-4),
        ];
        for (c, want, tol) in cases {
            close(c.from_linear(0.18), *want, *tol, &format!("{:?} mid grey", c));
        }
    }

    /// Cineon is anchored by its reference codes rather than by mid grey:
    /// 95 is black, 685 is the 90% white patch.
    #[test]
    fn cineon_hits_its_reference_codes() {
        close(cineon_to_linear(685.0), 1.0, 1e-5, "685 -> white");
        close(cineon_to_linear(95.0), 0.0, 1e-4, "95 -> black");
        close(linear_to_cineon(1.0), 685.0, 1e-2, "white -> 685");
    }

    #[test]
    fn every_curve_round_trips() {
        for c in Curve::ALL {
            for &x in &[0.0f32, 0.001, 0.005, 0.018, 0.18, 0.5, 1.0, 4.0, 16.0] {
                let back = c.to_linear(c.from_linear(x));
                let tol = 2e-3 * x.max(0.01);
                assert!(
                    (back - x).abs() <= tol,
                    "{c:?}: linear {x} -> {} -> {back}",
                    c.from_linear(x)
                );
            }
        }
    }

    /// Camera curves are defined below their cut by a linear segment, so a
    /// small negative (which a gamut conversion WILL produce) has to survive
    /// the trip instead of becoming NaN.
    #[test]
    fn curves_stay_finite_on_negatives() {
        for c in Curve::ALL {
            for &x in &[-0.001f32, -0.02, -0.2] {
                let e = c.from_linear(x);
                assert!(e.is_finite(), "{c:?} encode({x}) = {e}");
                assert!(c.to_linear(e).is_finite(), "{c:?} decode round trip of {x}");
            }
        }
    }

    #[test]
    fn identity_transforms_emit_nothing() {
        let t = Transform::new(Colorspace::ACESCG, Colorspace::ACESCG);
        assert!(t.is_identity());
        let mut h = std::collections::BTreeMap::new();
        let src = t.wgsl("idt", &mut h);
        assert!(h.is_empty(), "identity needs no helpers");
        assert!(!src.contains("mat3x3"), "identity emits no matrix:\n{src}");
    }

    #[test]
    fn a_real_idt_is_not_identity_and_moves_pixels() {
        let t = Transform::new(Colorspace::SLOG3, Colorspace::ACESCG);
        assert!(!t.is_identity());
        // S-Log3 mid grey must decode back to 18% scene linear, and stay
        // neutral through the gamut change.
        let g = t.apply([0.4105572, 0.4105572, 0.4105572]);
        for (i, v) in g.iter().enumerate() {
            close(*v, 0.18, 2e-3, &format!("S-Log3 grey -> ACEScg ch{i}"));
        }
    }

    /// Adding a working space must not change what an existing graph does.
    /// The engine default is the space the comp already worked in.
    #[test]
    fn the_engine_default_is_the_legacy_working_space() {
        assert_eq!(Colorspace::default(), Colorspace::LINEAR_REC709);
        assert!(Transform::new(Colorspace::default(), Colorspace::LINEAR_REC709).is_identity());
    }

    #[test]
    fn working_space_input_is_a_no_op() {
        let cs = Colorspace::ACESCG;
        assert_eq!(InputSpace::Working.resolve(cs), cs);
        assert!(Transform::new(InputSpace::Working.resolve(cs), cs).is_identity());
    }

    /// A scene-linear container gets no guess (we do not know, and guessing
    /// is how plates get double-transformed); a stills format does.
    #[test]
    fn extension_guesses_are_conservative() {
        assert_eq!(InputSpace::guess_for_extension("exr"), None);
        assert_eq!(InputSpace::guess_for_extension("png"), Some(Colorspace::SRGB));
        assert_eq!(InputSpace::guess_for_extension("DPX"), Some(Colorspace::CINEON));
        // A movie is display-referred by overwhelming default, so it gets
        // Rec.709 — still only OFFERED, never applied, because a log clip
        // in a .mov is common enough to make a silent guess dangerous.
        assert_eq!(InputSpace::guess_for_extension("mov"), Some(Colorspace::REC709));
        // A container we cannot decode at all still gets no guess.
        assert_eq!(InputSpace::guess_for_extension("r3d"), None);
    }

    /// The built-in table is the guesses, applied; a user row wins; formats
    /// group their extensions; an unknown one is the working space.
    #[test]
    fn format_defaults_resolve_by_format() {
        let mut d = FormatDefaults::default();
        assert_eq!(d.input_for("a/plate.####.dpx"), InputSpace::Explicit(Colorspace::CINEON));
        assert_eq!(d.input_for("x.JPEG"), InputSpace::Explicit(Colorspace::SRGB));
        assert_eq!(d.input_for("clip.mxf"), InputSpace::Explicit(Colorspace::REC709));
        assert_eq!(d.input_for("r.exr"), InputSpace::Working);
        assert_eq!(d.input_for("clip.r3d"), InputSpace::Working);
        d.input.insert("dpx".into(), InputSpace::Explicit(Colorspace::REC709));
        assert_eq!(d.input_for("p.dpx"), InputSpace::Explicit(Colorspace::REC709));
        // A row can never point back at "default".
        d.input.insert("png".into(), InputSpace::Default);
        assert_eq!(d.input_for("p.png"), InputSpace::Working);
        // Output: working unless set; a directory Write is an EXR.
        assert_eq!(d.output_for("renders/v003"), None);
        d.output.insert("exr".into(), Some(Colorspace::ACES2065));
        assert_eq!(d.output_for("renders/v003"), Some(Colorspace::ACES2065));
        assert_eq!(d.output_for("out/c.####.exr"), Some(Colorspace::ACES2065));
    }

    /// Both Rec.709s are offered and they are different curves: the display
    /// one is a pure 2.4 power, the camera one the OETF with its linear toe.
    #[test]
    fn rec709_display_and_camera_are_both_offered() {
        let has = |cs| Colorspace::PRESETS.iter().any(|(_, p)| *p == cs);
        assert!(has(Colorspace::REC709_BT1886) && has(Colorspace::REC709));
        let d = Colorspace::REC709_BT1886.curve.to_linear(0.5);
        assert!((d - 0.5f32.powf(2.4)).abs() < 1e-6);
        let c = Colorspace::REC709.curve.to_linear(0.5);
        assert!((c - 0.2597).abs() < 1e-3, "{c}");
        assert_eq!(Colorspace::REC709_BT1886.label(), "Rec.709 (BT.1886 / 2.4)");
    }

    #[cfg(feature = "serde")]
    #[test]
    fn format_defaults_round_trip() {
        let mut d = FormatDefaults::default();
        d.input.insert("dpx".into(), InputSpace::Explicit(Colorspace::REC709));
        let back: FormatDefaults =
            serde_json::from_str(&serde_json::to_string(&d).unwrap()).unwrap();
        assert_eq!(back, d);
    }

    /// Every generated transform must be valid WGSL. naga's default error
    /// path is a panic in the shader module, which in the app shows up as a
    /// dead pipeline with no message — so it is checked here instead.
    #[test]
    fn every_transform_generates_valid_wgsl() {
        let mut checked = 0;
        for (_, src) in Colorspace::PRESETS {
            for (_, dst) in Colorspace::PRESETS {
                let t = Transform::new(src, dst);
                let mut helpers = std::collections::BTreeMap::new();
                let f = t.wgsl("cs_test", &mut helpers);
                let mut wgsl = String::new();
                for h in helpers.values() {
                    wgsl.push_str(h);
                }
                wgsl.push_str(&f);
                wgsl.push_str(
                    "@compute @workgroup_size(1) fn main() { _ = cs_test(vec3f(0.18)); }\n",
                );
                let module = naga::front::wgsl::parse_str(&wgsl)
                    .unwrap_or_else(|e| panic!("{src:?} -> {dst:?} parse: {e:?}\n{wgsl}"));
                naga::valid::Validator::new(
                    naga::valid::ValidationFlags::all(),
                    naga::valid::Capabilities::all(),
                )
                .validate(&module)
                .unwrap_or_else(|e| panic!("{src:?} -> {dst:?} validate: {e:?}\n{wgsl}"));
                checked += 1;
            }
        }
        assert_eq!(checked, Colorspace::PRESETS.len() * Colorspace::PRESETS.len());
    }
}
