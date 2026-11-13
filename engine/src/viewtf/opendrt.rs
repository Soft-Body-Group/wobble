//! OpenDRT v1.1.0 — port of Jed Smith's display rendering transform.
//!
//! ## Licensing — READ BEFORE VENDORING
//!
//! Upstream (<https://github.com/jedypod/open-display-transform>) is **GPLv3**.
//! This file is a derivative work of `display-transforms/opendrt/OpenDRT.dctl`
//! and is therefore GPLv3 too, which is why the whole module sits behind the
//! non-default `opendrt` cargo feature: a default build of Wobble contains
//! none of this code and keeps its own (permissive) licence. Enabling the
//! feature produces a GPLv3 binary. Same arrangement ffmpeg uses for its
//! `--enable-gpl` components.
//!
//! ## What is ported
//!
//! The SDR display path, faithfully: input gamut -> P3-D65, rendering-space
//! desaturation, tonescale norm + RGB ratios, brilliance, low/high contrast,
//! hyperbolic compression, hue contrast, RGB/CMY hue shift, purity limits,
//! mid-range purity, inverse rendering space, display gamut + creative white,
//! post brilliance, purity softclip, and the inverse display EOTF.
//!
//! Not ported (yet): PQ / HLG encodings and the DCI-P3 / X'Y'Z' containers
//! (they need an HDR or DCI surface, which the viewer has no path to), the
//! log input transfer functions (Wobble reads scene-linear EXR), and the
//! tonescale-overlay debug draw.
//!
//! ## Where the maths runs
//!
//! DCTL cannot precompute, so upstream recalculates the tonescale constraint
//! constants on every pixel. Here they are solved once on the CPU in
//! [`Derived`] and uploaded as uniforms; the shader only does per-pixel work.
//! The numbered comments in [`WGSL`] map onto upstream's section headers so
//! the two can be diffed by eye when a new OpenDRT lands.

// Constants are upstream's, at upstream's precision, for the same reason.
#![allow(clippy::excessive_precision)]

use super::Gamut;

/// Creative white point. Upstream's `cwp` combo, D65 being the no-op.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CreativeWhite {
    D93,
    D75,
    D65,
    D60,
    D55,
    D50,
}

impl CreativeWhite {
    pub const ALL: [Self; 6] =
        [Self::D93, Self::D75, Self::D65, Self::D60, Self::D55, Self::D50];

    pub fn label(&self) -> &'static str {
        match self {
            Self::D93 => "D93",
            Self::D75 => "D75",
            Self::D65 => "D65",
            Self::D60 => "D60",
            Self::D55 => "D55",
            Self::D50 => "D50",
        }
    }

    /// Chromatic adaptation D65 -> this white, in XYZ. Identity for D65.
    fn cat_from_d65(&self) -> [[f32; 3]; 3] {
        match self {
            Self::D65 => super::IDENTITY3,
            Self::D93 => super::CAT_D65_TO_D93,
            Self::D75 => super::CAT_D65_TO_D75,
            Self::D60 => super::CAT_D65_TO_D60,
            Self::D55 => super::CAT_D65_TO_D55,
            Self::D50 => super::CAT_D65_TO_D50,
        }
    }

    /// Upstream's precalculated peak-luminance renormalisation, per display
    /// gamut. Keeps a creative white from pushing peak white over display max.
    fn norm(&self, display: Gamut) -> f32 {
        // Rec.709 column and the P3-D65 column (also used for the P3-limited
        // Rec.2020 container), lifted from OpenDRT.dctl.
        let rec709 = match self {
            Self::D93 => 0.744_192_7,
            Self::D75 => 0.873_470_84,
            Self::D65 => 1.0,
            Self::D60 => 0.955_937,
            Self::D55 => 0.905_671_3,
            Self::D50 => 0.850_004_4,
        };
        let p3d65 = match self {
            Self::D93 => 0.762_687_04,
            Self::D75 => 0.884_054_1,
            Self::D65 => 1.0,
            Self::D60 => 0.964_320_2,
            Self::D55 => 0.923_076_5,
            Self::D50 => 0.876_572_84,
        };
        match display {
            Gamut::Rec709 => rec709,
            _ => p3d65,
        }
    }
}

/// One of upstream's named look presets. These set the whole parameter block;
/// individual knobs can then be nudged from the UI.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum LookPreset {
    Standard,
    Arriba,
    Sylvan,
    Colorful,
    Aery,
    Dystopic,
    Umbra,
    Base,
}

impl LookPreset {
    pub const ALL: [Self; 8] = [
        Self::Standard,
        Self::Arriba,
        Self::Sylvan,
        Self::Colorful,
        Self::Aery,
        Self::Dystopic,
        Self::Umbra,
        Self::Base,
    ];

    pub fn label(&self) -> &'static str {
        match self {
            Self::Standard => "Standard",
            Self::Arriba => "Arriba",
            Self::Sylvan => "Sylvan",
            Self::Colorful => "Colorful",
            Self::Aery => "Aery",
            Self::Dystopic => "Dystopic",
            Self::Umbra => "Umbra",
            Self::Base => "Base",
        }
    }
}

/// The full OpenDRT parameter space (upstream's "stickshift" set), plus the
/// display-encoding choices. Field names match the DCTL so upstream's docs and
/// desmos links carry over verbatim.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
#[allow(non_snake_case)]
pub struct OpenDrtParams {
    // --- tonescale ---
    /// Display peak luminance, nits.
    pub tn_Lp: f32,
    /// Display grey luminance, nits.
    pub tn_Lg: f32,
    /// HDR grey boost.
    pub tn_gb: f32,
    /// HDR purity.
    pub pt_hdr: f32,
    pub tn_con: f32,
    pub tn_sh: f32,
    pub tn_toe: f32,
    pub tn_off: f32,
    /// Surround compensation index (0 = none, 1 = Rec.1886, 2 = sRGB).
    pub tn_su: u32,

    pub tn_hcon_enable: bool,
    pub tn_hcon: f32,
    pub tn_hcon_pv: f32,
    pub tn_hcon_st: f32,
    pub tn_lcon_enable: bool,
    pub tn_lcon: f32,
    pub tn_lcon_w: f32,

    // --- rendering space ---
    pub rs_sa: f32,
    pub rs_rw: f32,
    pub rs_bw: f32,

    // --- purity ---
    pub pt_enable: bool,
    pub pt_lml: f32,
    pub pt_lml_r: f32,
    pub pt_lml_g: f32,
    pub pt_lml_b: f32,
    pub pt_lmh: f32,
    pub pt_lmh_r: f32,
    pub pt_lmh_b: f32,
    pub ptl_enable: bool,
    pub ptl_c: f32,
    pub ptl_m: f32,
    pub ptl_y: f32,
    pub ptm_enable: bool,
    pub ptm_low: f32,
    pub ptm_low_rng: f32,
    pub ptm_low_st: f32,
    pub ptm_high: f32,
    pub ptm_high_rng: f32,
    pub ptm_high_st: f32,

    // --- brilliance ---
    pub brl_enable: bool,
    pub brl: f32,
    pub brl_r: f32,
    pub brl_g: f32,
    pub brl_b: f32,
    pub brl_rng: f32,
    pub brl_st: f32,
    pub brlp_enable: bool,
    pub brlp: f32,
    pub brlp_r: f32,
    pub brlp_g: f32,
    pub brlp_b: f32,

    // --- hue ---
    pub hc_enable: bool,
    pub hc_r: f32,
    pub hc_r_rng: f32,
    pub hs_rgb_enable: bool,
    pub hs_r: f32,
    pub hs_r_rng: f32,
    pub hs_g: f32,
    pub hs_g_rng: f32,
    pub hs_b: f32,
    pub hs_b_rng: f32,
    pub hs_cmy_enable: bool,
    pub hs_c: f32,
    pub hs_c_rng: f32,
    pub hs_m: f32,
    pub hs_m_rng: f32,
    pub hs_y: f32,
    pub hs_y_rng: f32,

    // --- display ---
    pub cwp: CreativeWhite,
    pub cwp_lm: f32,
}

impl Default for OpenDrtParams {
    fn default() -> Self {
        Self::preset(LookPreset::Standard)
    }
}

impl OpenDrtParams {
    /// Upstream's look presets, verbatim from the DCTL's preset table.
    #[allow(non_snake_case)]
    pub fn preset(p: LookPreset) -> Self {
        // Everything the presets don't vary.
        let base = Self {
            tn_Lp: 100.0,
            tn_Lg: 10.0,
            tn_gb: 0.13,
            pt_hdr: 0.5,
            tn_su: 2, // sRGB display; the encoding preset overrides
            tn_con: 1.66,
            tn_sh: 0.5,
            tn_toe: 0.003,
            tn_off: 0.005,
            tn_hcon_enable: false,
            tn_hcon: 0.0,
            tn_hcon_pv: 1.0,
            tn_hcon_st: 4.0,
            tn_lcon_enable: false,
            tn_lcon: 0.0,
            tn_lcon_w: 0.5,
            rs_sa: 0.35,
            rs_rw: 0.25,
            rs_bw: 0.55,
            pt_enable: true,
            pt_lml: 0.25,
            pt_lml_r: 0.5,
            pt_lml_g: 0.0,
            pt_lml_b: 0.1,
            pt_lmh: 0.25,
            pt_lmh_r: 0.5,
            pt_lmh_b: 0.0,
            ptl_enable: true,
            ptl_c: 0.06,
            ptl_m: 0.08,
            ptl_y: 0.06,
            ptm_enable: true,
            ptm_low: 0.4,
            ptm_low_rng: 0.25,
            ptm_low_st: 0.5,
            ptm_high: -0.8,
            ptm_high_rng: 0.35,
            ptm_high_st: 0.4,
            brl_enable: true,
            brl: 0.0,
            brl_r: -2.5,
            brl_g: -1.5,
            brl_b: -1.5,
            brl_rng: 0.5,
            brl_st: 0.35,
            brlp_enable: true,
            brlp: -0.5,
            brlp_r: -1.25,
            brlp_g: -1.25,
            brlp_b: -0.25,
            hc_enable: true,
            hc_r: 1.0,
            hc_r_rng: 0.3,
            hs_rgb_enable: true,
            hs_r: 0.6,
            hs_r_rng: 0.6,
            hs_g: 0.35,
            hs_g_rng: 1.0,
            hs_b: 0.66,
            hs_b_rng: 1.0,
            hs_cmy_enable: true,
            hs_c: 0.25,
            hs_c_rng: 1.0,
            hs_m: 0.0,
            hs_m_rng: 1.0,
            hs_y: 0.0,
            hs_y_rng: 1.0,
            cwp: CreativeWhite::D65,
            cwp_lm: 0.25,
        };
        match p {
            LookPreset::Standard => base,
            LookPreset::Arriba => Self {
                tn_con: 1.05,
                tn_toe: 0.1,
                tn_off: 0.01,
                tn_lcon_enable: true,
                tn_lcon: 1.5,
                tn_lcon_w: 0.2,
                pt_lml_r: 0.45,
                pt_lmh_r: 0.25,
                ptm_low: 1.0,
                ptm_low_rng: 0.4,
                ptm_high_rng: 0.66,
                ptm_high_st: 0.6,
                brlp: 0.0,
                brlp_r: -1.7,
                brlp_g: -2.0,
                brlp_b: -0.5,
                hs_r_rng: 0.8,
                hs_c: 0.15,
                ..base
            },
            LookPreset::Sylvan => Self {
                tn_con: 1.6,
                tn_toe: 0.01,
                tn_off: 0.01,
                tn_lcon_enable: true,
                tn_lcon: 0.25,
                tn_lcon_w: 0.75,
                rs_sa: 0.25,
                pt_lml: 0.15,
                pt_lml_g: 0.15,
                pt_lmh_r: 0.15,
                pt_lmh_b: 0.15,
                ptl_c: 0.05,
                ptl_y: 0.05,
                ptm_low: 0.5,
                ptm_low_rng: 0.5,
                ptm_high_rng: 0.5,
                ptm_high_st: 0.5,
                brl: -1.0,
                brl_r: -2.0,
                brl_g: -2.0,
                brl_b: 0.0,
                brl_rng: 0.25,
                brl_st: 0.25,
                brlp: -1.0,
                brlp_r: -0.5,
                brlp_g: -0.25,
                brlp_b: -0.25,
                hc_r_rng: 0.4,
                hs_r_rng: 1.15,
                hs_g: 0.8,
                hs_g_rng: 1.25,
                hs_b: 0.6,
                hs_c_rng: 0.25,
                hs_m: 0.25,
                hs_m_rng: 0.5,
                hs_y: 0.35,
                hs_y_rng: 0.5,
                ..base
            },
            LookPreset::Colorful => Self {
                tn_con: 1.5,
                tn_off: 0.003,
                tn_lcon_enable: true,
                tn_lcon: 0.4,
                pt_lml: 0.5,
                pt_lml_r: 1.0,
                pt_lml_b: 0.5,
                pt_lmh: 0.15,
                pt_lmh_r: 0.15,
                pt_lmh_b: 0.15,
                ptl_c: 0.05,
                ptl_m: 0.06,
                ptl_y: 0.05,
                ptm_low: 0.8,
                ptm_low_rng: 0.5,
                ptm_low_st: 0.4,
                ptm_high_rng: 0.4,
                brl_r: -1.25,
                brl_g: -1.25,
                brl_b: -0.25,
                brl_rng: 0.3,
                brl_st: 0.5,
                brlp_b: -0.5,
                hc_r_rng: 0.4,
                hs_r: 0.5,
                hs_r_rng: 0.8,
                hs_b: 0.5,
                hs_y: 0.25,
                ..base
            },
            LookPreset::Aery => Self {
                tn_con: 1.15,
                tn_toe: 0.04,
                tn_off: 0.006,
                tn_hcon_pv: 0.0,
                tn_hcon_st: 0.5,
                tn_lcon_enable: true,
                tn_lcon: 0.5,
                tn_lcon_w: 2.0,
                rs_sa: 0.25,
                rs_rw: 0.2,
                rs_bw: 0.5,
                pt_lml: 0.0,
                pt_lml_g: 0.15,
                pt_lmh: 0.0,
                pt_lmh_r: 0.1,
                ptl_c: 0.05,
                ptl_y: 0.05,
                ptm_low: 0.8,
                ptm_low_rng: 0.35,
                ptm_high: -0.9,
                ptm_high_rng: 0.5,
                ptm_high_st: 0.3,
                brl: -3.0,
                brl_r: 0.0,
                brl_g: 0.0,
                brl_b: 1.0,
                brl_rng: 0.8,
                brl_st: 0.15,
                brlp: -1.0,
                brlp_r: -1.0,
                brlp_g: -1.0,
                brlp_b: 0.0,
                hc_r: 0.5,
                hc_r_rng: 0.25,
                hs_r_rng: 1.0,
                hs_g_rng: 2.0,
                hs_b: 0.5,
                hs_b_rng: 1.5,
                hs_c: 0.35,
                hs_m: 0.25,
                hs_y: 0.35,
                hs_y_rng: 0.5,
                cwp: CreativeWhite::D75,
                ..base
            },
            LookPreset::Dystopic => Self {
                tn_con: 1.6,
                tn_toe: 0.01,
                tn_off: 0.008,
                tn_hcon_enable: true,
                tn_hcon: 0.25,
                tn_hcon_pv: 0.0,
                tn_hcon_st: 1.0,
                tn_lcon_enable: true,
                tn_lcon: 1.0,
                tn_lcon_w: 0.75,
                rs_sa: 0.2,
                pt_lml: 0.15,
                pt_lml_r: 0.0,
                pt_lml_b: 0.0,
                pt_lmh: 0.0,
                pt_lmh_r: 0.0,
                ptl_c: 0.05,
                ptl_y: 0.05,
                ptm_low: 0.25,
                ptm_low_st: 0.8,
                ptm_high_rng: 0.6,
                ptm_high_st: 0.25,
                brl: -2.0,
                brl_r: -2.0,
                brl_g: -2.0,
                brl_b: 0.0,
                brl_rng: 0.35,
                brlp: 0.0,
                brlp_r: -1.0,
                brlp_g: -1.0,
                brlp_b: -1.0,
                hc_r_rng: 0.25,
                hs_r: 0.7,
                hs_r_rng: 1.33,
                hs_g: 1.0,
                hs_g_rng: 2.0,
                hs_b: 0.75,
                hs_b_rng: 2.0,
                hs_c: 1.0,
                hs_c_rng: 0.5,
                hs_m: 1.0,
                hs_y: 1.0,
                hs_y_rng: 0.765,
                cwp: CreativeWhite::D60,
                ..base
            },
            LookPreset::Umbra => Self {
                tn_con: 1.8,
                tn_toe: 0.001,
                tn_off: 0.015,
                tn_lcon_enable: true,
                tn_lcon: 1.0,
                tn_lcon_w: 1.0,
                pt_lml: 0.0,
                pt_lml_b: 0.15,
                pt_lmh_r: 0.25,
                ptl_c: 0.05,
                ptl_m: 0.06,
                ptl_y: 0.05,
                ptm_low_rng: 0.35,
                ptm_low_st: 0.66,
                ptm_high: -0.6,
                ptm_high_rng: 0.45,
                ptm_high_st: 0.45,
                brl: -2.0,
                brl_r: -4.5,
                brl_g: -3.0,
                brl_b: -4.0,
                brl_rng: 0.35,
                brl_st: 0.3,
                brlp: 0.0,
                brlp_r: -2.0,
                brlp_g: -1.0,
                brlp_b: -0.5,
                hc_r_rng: 0.35,
                hs_r: 0.66,
                hs_r_rng: 1.0,
                hs_g: 0.5,
                hs_g_rng: 2.0,
                hs_b: 0.85,
                hs_b_rng: 2.0,
                hs_c: 0.0,
                hs_m: 0.25,
                hs_y: 0.66,
                hs_y_rng: 0.66,
                cwp: CreativeWhite::D50,
                ..base
            },
            LookPreset::Base => Self {
                pt_lml: 0.5,
                pt_lml_g: 0.15,
                pt_lml_b: 0.15,
                pt_lmh: 0.8,
                pt_lmh_r: 0.5,
                ptl_c: 0.05,
                ptl_m: 0.06,
                ptl_y: 0.05,
                ptm_enable: false,
                ptm_low: 0.0,
                ptm_low_rng: 0.5,
                ptm_high: 0.0,
                ptm_high_rng: 0.5,
                ptm_high_st: 0.5,
                brl_enable: false,
                brl_r: 0.0,
                brl_g: 0.0,
                brl_b: 0.0,
                brlp_r: -1.6,
                brlp_g: -1.6,
                brlp_b: -0.8,
                hc_enable: false,
                hc_r: 0.0,
                hc_r_rng: 0.25,
                hs_rgb_enable: false,
                hs_r: 0.0,
                hs_r_rng: 1.0,
                hs_g: 0.0,
                hs_b: 0.0,
                hs_cmy_enable: false,
                hs_c: 0.0,
                ..base
            },
        }
    }
}

/// Slot map for the `drt` half of the blit uniform. The shader unpacks these
/// into named `let`s in the same order — change one, change the other.
///
/// | slot | x | y | z | w |
/// |---|---|---|---|---|
/// | 0 | ts_x0 | ts_s | ts_p | ts_m2 |
/// | 1 | ts_s1 | s_Lp100 | ts_dsc | tn_toe |
/// | 2 | tn_off | rs_sa | rs_rw | rs_bw |
/// | 3 | lcon_m | lcon_w | lcon_sc | hcon_p |
/// | 4 | tn_hcon_pv | tn_hcon_st | hc_r | hc_r_rng |
/// | 5 | brl | brl_r | brl_g | brl_b |
/// | 6 | brl_rng | brl_st | brlp | brlp_r |
/// | 7 | brlp_g | brlp_b | hs_r | hs_r_rng |
/// | 8 | hs_g | hs_g_rng | hs_b | hs_b_rng |
/// | 9 | hs_c | hs_c_rng | hs_m | hs_m_rng |
/// | 10 | hs_y | hs_y_rng | pt_lml | pt_lml_r |
/// | 11 | pt_lml_g | pt_lml_b | pt_lmh | pt_lmh_r |
/// | 12 | pt_lmh_b | ptl_c | ptl_m | ptl_y |
/// | 13 | ptm_low | ptm_low_rng | ptm_low_st | ptm_high |
/// | 14 | ptm_high_rng | ptm_high_st | cwp_lm | cwp_norm |
/// | 15 | en_brl | en_lcon | en_hcon | en_hc |
/// | 16 | en_hs_rgb | en_hs_cmy | en_ptm | en_brlp |
/// | 17 | en_ptl | — | — | — |
pub const SLOTS: usize = 18;

/// Sign-preserving power, upstream's `spowf`.
fn spow(a: f32, b: f32) -> f32 {
    if a <= 0.0 {
        a
    } else {
        a.powf(b)
    }
}

/// Upstream's `compress_toe_quadratic` with `inv = 1`.
fn toe_quad_inv(x: f32, toe: f32) -> f32 {
    if toe == 0.0 {
        return x;
    }
    (x + (x * (4.0 * toe + x)).max(0.0).sqrt()) / 2.0
}

/// Upstream's `compress_toe_cubic` with `inv = 1` — the real cubic root, used
/// once on the CPU to solve the low-contrast normalisation constant.
fn toe_cubic_inv(x: f32, m: f32, w: f32) -> f32 {
    if m == 1.0 {
        return x;
    }
    let x2 = x * x;
    let p0 = x2 - 3.0 * m * w;
    let p1 = 2.0 * x2 + 27.0 * w - 9.0 * m * w;
    let inner = (x2 * p1 * p1 - 4.0 * p0 * p0 * p0).max(0.0).sqrt() / 2.0 + x * p1 / 2.0;
    let p2 = spow(inner, 1.0 / 3.0);
    if p2 == 0.0 {
        return x;
    }
    p0 / (3.0 * p2) + p2 / 3.0 + x / 3.0
}

/// Everything upstream recomputes per pixel because DCTL has no constant
/// folding. Solved once here; the shader reads the results as uniforms.
#[derive(Clone, Copy, Debug)]
pub struct Derived {
    pub ts_x0: f32,
    pub ts_s: f32,
    pub ts_p: f32,
    pub ts_m2: f32,
    pub ts_s1: f32,
    pub s_lp100: f32,
    pub ts_dsc: f32,
    pub lcon_m: f32,
    pub lcon_w: f32,
    pub lcon_sc: f32,
    pub hcon_p: f32,
}

impl Derived {
    #[allow(non_snake_case)]
    pub fn new(p: &OpenDrtParams) -> Self {
        let tn_con = p.tn_con.max(1e-3);
        let ts_x1 = (6.0 * p.tn_sh + 4.0).exp2();
        let ts_y1 = p.tn_Lp / 100.0;
        let ts_x0 = 0.18 + p.tn_off;
        let ts_y0 = p.tn_Lg / 100.0 * (1.0 + p.tn_gb * ts_y1.log2());
        let ts_s0 = toe_quad_inv(ts_y0, p.tn_toe);
        // Surround compensation is unconstrained, exactly as upstream.
        let ts_p = tn_con / (1.0 + p.tn_su as f32 * 0.05);
        let ts_s10 = ts_x0 * (spow(ts_s0, -1.0 / tn_con) - 1.0);
        let ts_m1 = ts_y1 / spow(ts_x1 / (ts_x1 + ts_s10), tn_con);
        let ts_m2 = toe_quad_inv(ts_m1, p.tn_toe);
        let ts_s = ts_x0 * (spow(ts_s0 / ts_m2, -1.0 / tn_con) - 1.0);
        // PQ / HLG would use 0.01 / 0.1 here; SDR only for now.
        let ts_dsc = 100.0 / p.tn_Lp;

        // Purity compression lerps from SDR at 100 nits toward pt_hdr at 1000.
        let pt_cmp_lf = p.pt_hdr * (1.0f32).min((p.tn_Lp - 100.0) / 900.0).max(0.0);
        let s_lp100 = ts_x0 * (spow(p.tn_Lg / 100.0, -1.0 / tn_con) - 1.0);
        let ts_s1 = ts_s * pt_cmp_lf + s_lp100 * (1.0 - pt_cmp_lf);

        let lcon_m = (-p.tn_lcon).exp2();
        let lcon_w = {
            let w = p.tn_lcon_w / 4.0;
            w * w
        };
        // Normalise so the curve still passes through ts_x0.
        let lcon_sc = if p.tn_lcon_enable && lcon_w > 0.0 {
            toe_cubic_inv(ts_x0, lcon_m, lcon_w) / ts_x0
        } else {
            1.0
        };
        let hcon_p = p.tn_hcon.exp2();

        Self {
            ts_x0,
            ts_s,
            ts_p,
            ts_m2,
            ts_s1,
            s_lp100,
            ts_dsc,
            lcon_m,
            lcon_w,
            lcon_sc,
            hcon_p,
        }
    }
}

/// Pack the parameter block for the blit uniform's `drt` array.
pub fn pack(p: &OpenDrtParams, display: Gamut) -> [[f32; 4]; SLOTS] {
    let d = Derived::new(p);
    let en = |b: bool| if b { 1.0 } else { 0.0 };
    // Guards for divisions upstream leaves open at slider extremes.
    let nz = |v: f32| if v.abs() < 1e-4 { 1e-4 } else { v };
    [
        [d.ts_x0, d.ts_s, d.ts_p, d.ts_m2],
        [d.ts_s1, d.s_lp100, d.ts_dsc, p.tn_toe],
        [p.tn_off, p.rs_sa.clamp(0.0, 0.99), p.rs_rw, p.rs_bw],
        [d.lcon_m, d.lcon_w, d.lcon_sc, d.hcon_p],
        [p.tn_hcon_pv, p.tn_hcon_st, p.hc_r, nz(p.hc_r_rng)],
        [p.brl, p.brl_r, p.brl_g, p.brl_b],
        [p.brl_rng, nz(p.brl_st), p.brlp, p.brlp_r],
        [p.brlp_g, p.brlp_b, p.hs_r, nz(p.hs_r_rng)],
        [p.hs_g, nz(p.hs_g_rng), p.hs_b, nz(p.hs_b_rng)],
        [p.hs_c, nz(p.hs_c_rng), p.hs_m, nz(p.hs_m_rng)],
        [p.hs_y, nz(p.hs_y_rng), p.pt_lml, p.pt_lml_r],
        [p.pt_lml_g, p.pt_lml_b, p.pt_lmh, p.pt_lmh_r],
        [p.pt_lmh_b, p.ptl_c, p.ptl_m, p.ptl_y],
        [p.ptm_low, p.ptm_low_rng, p.ptm_low_st, p.ptm_high],
        [
            p.ptm_high_rng,
            p.ptm_high_st,
            p.cwp_lm,
            p.cwp.norm(display),
        ],
        [
            en(p.brl_enable),
            en(p.tn_lcon_enable),
            en(p.tn_hcon_enable),
            en(p.hc_enable),
        ],
        [
            en(p.hs_rgb_enable),
            en(p.hs_cmy_enable),
            en(p.ptm_enable),
            en(p.brlp_enable),
        ],
        [en(p.ptl_enable), 0.0, 0.0, 0.0],
    ]
}

/// P3-D65 -> display, with the creative white adaptation folded in, and the
/// same matrix without it. Upstream lerps between the two per pixel by
/// `cwp_f`; both paths are linear, so the lerp can happen after the matrices.
pub fn display_matrices(p: &OpenDrtParams, display: Gamut) -> ([[f32; 3]; 3], [[f32; 3]; 3]) {
    // Rec.2020 is delivered as a P3-limited container, so its matrix stays
    // P3-D65 here and the P3 -> Rec.2020 step happens after the tonescale.
    let xyz_to_display = match display {
        Gamut::Rec709 => super::XYZ_TO_REC709,
        _ => super::XYZ_TO_P3D65,
    };
    let neutral = super::mat3_mul(xyz_to_display, super::P3D65_TO_XYZ);
    let adapted = super::mat3_mul(
        xyz_to_display,
        super::mat3_mul(p.cwp.cat_from_d65(), super::P3D65_TO_XYZ),
    );
    (adapted, neutral)
}

/// WGSL body. Emitted into the blit shader when the OpenDRT view transform is
/// selected. Section headings mirror OpenDRT.dctl's so the two can be diffed.
pub fn wgsl(display: Gamut) -> String {
    // Rec.2020 output is P3-limited then matrixed, per upstream.
    let rec2020 = if matches!(display, Gamut::Rec2020) {
        let m = super::P3_TO_REC2020;
        format!(
            "    rgb = max(rgb, vec3f(0.0));\n    \
             rgb = mat3x3<f32>(vec3f({:?}, {:?}, {:?}), vec3f({:?}, {:?}, {:?}), \
             vec3f({:?}, {:?}, {:?})) * rgb;\n",
            m[0][0], m[1][0], m[2][0], m[0][1], m[1][1], m[2][1], m[0][2], m[1][2], m[2][2]
        )
    } else {
        String::new()
    };
    format!(
        r#"
// ---------------------------------------------------------------------------
// OpenDRT v1.1.0 — ported from Jed Smith's OpenDRT.dctl (GPLv3).
// https://github.com/jedypod/open-display-transform
// Returns DISPLAY-LINEAR values; the blit tail applies the inverse EOTF.
// ---------------------------------------------------------------------------
const OD_PI: f32 = 3.14159265358979323846;
const OD_SQRT3: f32 = 1.73205080756887729353;

fn od_spow(a: f32, b: f32) -> f32 {{ if (a <= 0.0) {{ return a; }} return pow(a, b); }}
fn od_sdiv(a: f32, b: f32) -> f32 {{ if (b == 0.0) {{ return 0.0; }} return a / b; }}
fn od_sdiv3(a: vec3f, b: f32) -> vec3f {{
    return vec3f(od_sdiv(a.x, b), od_sdiv(a.y, b), od_sdiv(a.z, b));
}}
fn od_hypot2(v: vec2f) -> f32 {{ return sqrt(max(0.0, v.x * v.x + v.y * v.y)); }}
fn od_hypot3(v: vec3f) -> f32 {{ return sqrt(max(0.0, dot(v, v))); }}

// Quadratic toe, forward direction (https://www.desmos.com/calculator/skk8ahmnws)
fn od_toe_quad(x: f32, toe: f32) -> f32 {{
    if (toe == 0.0) {{ return x; }}
    return od_sdiv(od_spow(x, 2.0), x + toe);
}}
// Cubic toe, forward direction (https://www.desmos.com/calculator/ubgteikoke)
fn od_toe_cubic(x: f32, m: f32, w: f32) -> f32 {{
    if (m == 1.0) {{ return x; }}
    let x2 = x * x;
    return od_sdiv(x * (x2 + m * w), x2 + w);
}}
// High exposure adjustment with linear extension
fn od_contrast_high(x: f32, p: f32, pv: f32, pv_lx: f32) -> f32 {{
    let x0 = 0.18 * pow(2.0, pv);
    if (x < x0 || p == 1.0) {{ return x; }}
    let o = x0 - x0 / p;
    let s0 = pow(x0, 1.0 - p) / p;
    let x1 = x0 * pow(2.0, pv_lx);
    let k1 = p * s0 * pow(x1, p) / x1;
    let y1 = s0 * pow(x1, p) + o;
    if (x > x1) {{ return k1 * (x - x1) + y1; }}
    return s0 * od_spow(x, p) + o;
}}
fn od_softplus(x: f32, s: f32) -> f32 {{
    if (x > 10.0 * s || s < 1e-4) {{ return x; }}
    return s * log(max(1e-8, 1.0 + exp(x / s)));
}}
fn od_gauss(x: f32, w: f32) -> f32 {{ return exp(-x * x / w); }}
// Cyan-Yellow / Green-Magenta opponent space
fn od_opponent(rgb: vec3f) -> vec2f {{
    return vec2f(rgb.x - rgb.z, rgb.y - (rgb.x + rgb.z) * 0.5);
}}
fn od_hue_offset(h: f32, o: f32) -> f32 {{
    return ((h - o + OD_PI) % (2.0 * OD_PI)) - OD_PI;
}}

fn view_transform(rgb_in: vec3f) -> vec3f {{
    // --- unpack (slot map lives in viewtf/opendrt.rs) ---
    let ts_x0 = u.drt[0].x; let ts_s = u.drt[0].y;
    let ts_p = u.drt[0].z; let ts_m2 = u.drt[0].w;
    let ts_s1 = u.drt[1].x; let s_lp100 = u.drt[1].y;
    let ts_dsc = u.drt[1].z; let tn_toe = u.drt[1].w;
    let tn_off = u.drt[2].x; let rs_sa = u.drt[2].y;
    let rs_rw = u.drt[2].z; let rs_bw = u.drt[2].w;
    let lcon_m = u.drt[3].x; let lcon_w = u.drt[3].y;
    let lcon_sc = u.drt[3].z; let hcon_p = u.drt[3].w;
    let tn_hcon_pv = u.drt[4].x; let tn_hcon_st = u.drt[4].y;
    let hc_r = u.drt[4].z; let hc_r_rng = u.drt[4].w;
    let brl = u.drt[5].x; let brl_r = u.drt[5].y;
    let brl_g = u.drt[5].z; let brl_b = u.drt[5].w;
    let brl_rng = u.drt[6].x; let brl_st = u.drt[6].y;
    let brlp = u.drt[6].z; let brlp_r = u.drt[6].w;
    let brlp_g = u.drt[7].x; let brlp_b = u.drt[7].y;
    let hs_r = u.drt[7].z; let hs_r_rng = u.drt[7].w;
    let hs_g = u.drt[8].x; let hs_g_rng = u.drt[8].y;
    let hs_b = u.drt[8].z; let hs_b_rng = u.drt[8].w;
    let hs_c = u.drt[9].x; let hs_c_rng = u.drt[9].y;
    let hs_m = u.drt[9].z; let hs_m_rng = u.drt[9].w;
    let hs_y = u.drt[10].x; let hs_y_rng = u.drt[10].y;
    let pt_lml = u.drt[10].z; let pt_lml_r = u.drt[10].w;
    let pt_lml_g = u.drt[11].x; let pt_lml_b = u.drt[11].y;
    let pt_lmh = u.drt[11].z; let pt_lmh_r = u.drt[11].w;
    let pt_lmh_b = u.drt[12].x; let ptl_c = u.drt[12].y;
    let ptl_m = u.drt[12].z; let ptl_y = u.drt[12].w;
    let ptm_low = u.drt[13].x; let ptm_low_rng = u.drt[13].y;
    let ptm_low_st = u.drt[13].z; let ptm_high = u.drt[13].w;
    let ptm_high_rng = u.drt[14].x; let ptm_high_st = u.drt[14].y;
    let cwp_lm = u.drt[14].z; let cwp_norm = u.drt[14].w;
    let en_brl = u.drt[15].x; let en_lcon = u.drt[15].y;
    let en_hcon = u.drt[15].z; let en_hc = u.drt[15].w;
    let en_hs_rgb = u.drt[16].x; let en_hs_cmy = u.drt[16].y;
    let en_ptm = u.drt[16].z; let en_brlp = u.drt[16].w;
    let en_ptl = u.drt[17].x;

    // Input gamut -> P3-D65
    var rgb = u.in_to_rs * rgb_in;

    // Rendering space: "desaturate" to control the scale of the colour volume
    // in the rgb ratios.
    let rs_w = vec3f(rs_rw, 1.0 - rs_rw - rs_bw, rs_bw);
    var sat_l = dot(rgb, rs_w);
    rgb = vec3f(sat_l * rs_sa) + rgb * (1.0 - rs_sa);

    rgb += vec3f(tn_off);

    // Tonescale norm + RGB ratios
    var tsn = od_hypot3(rgb) / OD_SQRT3;
    rgb = od_sdiv3(rgb, tsn);

    let opp = od_opponent(rgb);
    var ach_d = od_hypot2(opp) * 0.5;
    ach_d = 1.25 * od_toe_quad(ach_d, 0.25);

    // Hue angle, rotated so red = 0
    let hue = (atan2(opp.x, opp.y) + OD_PI + 1.10714931) % (2.0 * OD_PI);
    let ha_rgb = vec3f(
        od_gauss(od_hue_offset(hue, 0.1), 0.66),
        od_gauss(od_hue_offset(hue, 4.3), 0.66),
        od_gauss(od_hue_offset(hue, 2.3), 0.66));
    let ha_rgb_hs = vec3f(
        od_gauss(od_hue_offset(hue, -0.4), 0.66),
        ha_rgb.y,
        od_gauss(od_hue_offset(hue, 2.5), 0.66));
    let ha_cmy = vec3f(
        od_gauss(od_hue_offset(hue, 3.3), 0.5),
        od_gauss(od_hue_offset(hue, 1.3), 0.5),
        od_gauss(od_hue_offset(hue, -1.15), 0.5));

    // Brilliance
    if (en_brl > 0.5) {{
        let brl_tsf = od_spow(od_sdiv(tsn, tsn + 1.0), 1.0 - brl_rng);
        let brl_exf = (brl + brl_r * ha_rgb.x + brl_g * ha_rgb.y + brl_b * ha_rgb.z)
            * od_spow(ach_d, 1.0 / brl_st);
        var w = 1.0 - brl_tsf;
        if (brl_exf < 0.0) {{ w = brl_tsf; }}
        tsn *= pow(2.0, brl_exf * w);
    }}

    // Contrast low
    if (en_lcon > 0.5) {{
        tsn *= lcon_sc;
        tsn = od_toe_cubic(tsn, lcon_m, lcon_w);
    }}
    // Contrast high
    if (en_hcon > 0.5) {{
        tsn = od_contrast_high(tsn, hcon_p, tn_hcon_pv, tn_hcon_st);
    }}

    // Hyperbolic compression
    let tsn_pt = od_spow(od_sdiv(tsn, tsn + ts_s1), ts_p);
    let tsn_const = od_spow(od_sdiv(tsn, tsn + s_lp100), ts_p);
    tsn = od_spow(od_sdiv(tsn, tsn + ts_s), ts_p);

    // Hue contrast R
    if (en_hc > 0.5) {{
        var hc_ts = 1.0 - tsn_const;
        var hc_c = hc_ts * (1.0 - ach_d) + ach_d * (1.0 - hc_ts);
        hc_c *= ach_d * ha_rgb.x;
        hc_ts = od_spow(hc_ts, 1.0 / hc_r_rng);
        let hc_f = hc_r * (hc_c - 2.0 * hc_c * hc_ts) + 1.0;
        rgb = vec3f(rgb.x, rgb.y * hc_f, rgb.z * hc_f);
    }}

    // Hue shift RGB — shifts more as intensity increases
    if (en_hs_rgb > 0.5) {{
        let hs = vec3f(
            ha_rgb_hs.x * ach_d * od_spow(tsn_pt, 1.0 / hs_r_rng),
            ha_rgb_hs.y * ach_d * od_spow(tsn_pt, 1.0 / hs_g_rng),
            ha_rgb_hs.z * ach_d * od_spow(tsn_pt, 1.0 / hs_b_rng));
        let f = vec3f(hs.x * hs_r, hs.y * -hs_g, hs.z * -hs_b);
        rgb += vec3f(f.z - f.y, f.x - f.z, f.y - f.x);
    }}
    // Hue shift CMY — shifts less as intensity increases
    if (en_hs_cmy > 0.5) {{
        let compl = 1.0 - tsn_pt;
        let hs = vec3f(
            ha_cmy.x * ach_d * od_spow(compl, 1.0 / hs_c_rng),
            ha_cmy.y * ach_d * od_spow(compl, 1.0 / hs_m_rng),
            ha_cmy.z * ach_d * od_spow(compl, 1.0 / hs_y_rng));
        let f = vec3f(hs.x * -hs_c, hs.y * hs_m, hs.z * hs_y);
        rgb += vec3f(f.z - f.y, f.x - f.z, f.y - f.x);
    }}

    // Purity compression (https://www.desmos.com/calculator/adtzkjofgn)
    let pt_lml_p = 1.0 + 4.0 * (1.0 - tsn_pt)
        * (pt_lml + pt_lml_r * ha_rgb_hs.x + pt_lml_g * ha_rgb_hs.y + pt_lml_b * ha_rgb_hs.z);
    var ptf = 1.0 - od_spow(tsn_pt, pt_lml_p);
    let pt_lmh_p = (1.0 - ach_d * (pt_lmh_r * ha_rgb_hs.x + pt_lmh_b * ha_rgb_hs.z))
        * (1.0 - pt_lmh * ach_d);
    ptf = od_spow(ptf, pt_lmh_p);

    // Mid-range purity: boost on the low end, reduce on the high end
    if (en_ptm > 0.5) {{
        var lo = 1.0;
        if (ptm_low_st != 0.0 && ptm_low_rng != 0.0) {{
            lo = 1.0 + ptm_low * exp(-2.0 * ach_d * ach_d / ptm_low_st)
                * od_spow(1.0 - tsn_const, 1.0 / ptm_low_rng);
        }}
        var hi = 1.0;
        if (ptm_high_st != 0.0 && ptm_high_rng != 0.0) {{
            hi = 1.0 + ptm_high * exp(-2.0 * ach_d * ach_d / ptm_high_st)
                * od_spow(tsn_pt, 1.0 / (4.0 * ptm_high_rng));
        }}
        ptf *= lo * hi;
    }}

    // Lerp to peak achromatic by ptf, in rgb ratios
    rgb = rgb * ptf + vec3f(1.0 - ptf);

    // Inverse rendering space
    sat_l = dot(rgb, rs_w);
    rgb = (vec3f(sat_l * rs_sa) - rgb) / (rs_sa - 1.0);

    // Display gamut + creative whitepoint. Both matrices are baked CPU-side;
    // upstream lerps in XYZ, which is equivalent since both paths are linear.
    let cwp_f = od_spow(tsn_const, 2.0 * cwp_lm);
    rgb = (u.out_cwp * rgb) * cwp_f + (u.out_neu * rgb) * (1.0 - cwp_f);
    rgb *= cwp_norm * cwp_f + 1.0 - cwp_f;

    // Post brilliance
    if (en_brlp > 0.5) {{
        let bopp = od_opponent(rgb);
        var bd = od_hypot2(bopp) * 0.25;
        bd = 1.1 * od_sdiv(bd * bd, bd + 0.1);
        let bh = ach_d * ha_rgb;
        let m = brlp + brlp_r * bh.x + brlp_g * bh.y + brlp_b * bh.z;
        rgb *= pow(2.0, m * bd * tsn);
    }}

    // Purity compress low (softclip the display primaries)
    if (en_ptl > 0.5) {{
        rgb = vec3f(od_softplus(rgb.x, ptl_c), od_softplus(rgb.y, ptl_m),
                    od_softplus(rgb.z, ptl_y));
    }}

    // Final tonescale adjustments
    tsn *= ts_m2;
    tsn = od_toe_quad(tsn, tn_toe);
    tsn *= ts_dsc;

    // Return from RGB ratios
    rgb *= tsn;
{rec2020}
    return clamp(rgb, vec3f(0.0), vec3f(1.0));
}}
"#,
        rec2020 = rec2020
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derived_constants_are_finite_for_every_preset() {
        for p in LookPreset::ALL {
            let params = OpenDrtParams::preset(p);
            let d = Derived::new(&params);
            for (name, v) in [
                ("ts_x0", d.ts_x0),
                ("ts_s", d.ts_s),
                ("ts_p", d.ts_p),
                ("ts_m2", d.ts_m2),
                ("ts_s1", d.ts_s1),
                ("s_lp100", d.s_lp100),
                ("ts_dsc", d.ts_dsc),
                ("lcon_sc", d.lcon_sc),
                ("hcon_p", d.hcon_p),
            ] {
                assert!(v.is_finite(), "{} {name} = {v}", p.label());
            }
        }
    }

    #[test]
    fn mid_grey_hits_the_tonescale_constraint() {
        // The whole tonescale is solved so that 0.18 scene-linear lands on
        // tn_Lg/tn_Lp of display peak. That only holds exactly when surround
        // compensation is off, because upstream leaves `tn_su` deliberately
        // unconstrained — so this pins the constraint solve itself, which is
        // where a port error in `Derived` would show up first.
        let p = OpenDrtParams {
            tn_su: 0,
            ..OpenDrtParams::default()
        };
        let d = Derived::new(&p);
        assert!(
            (grey_through_tonescale(&p, &d) - p.tn_Lg / p.tn_Lp).abs() < 1e-4,
            "constrained mid grey drifted: {}",
            grey_through_tonescale(&p, &d)
        );
    }

    #[test]
    fn surround_compensation_lifts_mid_grey() {
        // tn_su lowers the exponent without re-solving the constraint, so an
        // sRGB (dim surround) view sits slightly above the nominal grey.
        let dark = OpenDrtParams {
            tn_su: 0,
            ..OpenDrtParams::default()
        };
        let srgb = OpenDrtParams {
            tn_su: 2,
            ..OpenDrtParams::default()
        };
        let a = grey_through_tonescale(&dark, &Derived::new(&dark));
        let b = grey_through_tonescale(&srgb, &Derived::new(&srgb));
        assert!(b > a, "surround compensation should lift grey: {a} -> {b}");
        assert!(b < a * 1.5, "and not by a lot: {a} -> {b}");
    }

    /// Achromatic 0.18 through the tonescale half of the transform: offset,
    /// hyperbolic compression, inverse-toe scale, toe, display scale.
    fn grey_through_tonescale(p: &OpenDrtParams, d: &Derived) -> f32 {
        let x = 0.18 + p.tn_off;
        let mut tsn = spow(x / (x + d.ts_s), d.ts_p);
        tsn *= d.ts_m2;
        if p.tn_toe != 0.0 {
            tsn = tsn * tsn / (tsn + p.tn_toe);
        }
        tsn * d.ts_dsc
    }

    #[test]
    fn packing_is_finite() {
        let p = OpenDrtParams::default();
        for slot in pack(&p, Gamut::Rec709) {
            for v in slot {
                assert!(v.is_finite());
            }
        }
    }
}
