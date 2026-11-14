//! View transforms: scene-linear comp -> display.
//!
//! The fused compute pass leaves the output texture **scene-linear**. Turning
//! that into pixels for a monitor happens here, at the blit, which means the
//! transform is display state rather than comp state: changing it never
//! recompiles a graph, never invalidates the render cache, and never bakes a
//! display curve into cached or exported frames.
//!
//! Anything that reads `Compositor::output_texture()` and wants to reason
//! about brightness has to apply a transform first — [`display_luma`] is the
//! cheap fixed-curve helper the viewer overlays use for that.

// Colour matrices below keep their published digit counts so they can be
// diffed against the source they came from, rather than being trimmed to
// f32's shortest round-tripping form.
#![allow(clippy::excessive_precision)]

pub mod aces;
pub mod agx;

// (wobble's OpenDRT is GPLv3, behind a feature there: only its stub here.)
#[path = "viewtf/opendrt_stub.rs"]
pub mod opendrt;

/// Which display rendering transform the viewer applies.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ViewTransform {
    /// Scene-linear straight through, clamped. Useful for checking values.
    None,
    /// The Narkowicz curve fit. Named honestly: it APPROXIMATES the shape of
    /// an ACES rendering in a handful of ALU and is not a colour-managed
    /// transform. Kept because it is cheap and because it is what the fused
    /// tail used to do — do not deliver from it.
    Filmic,
    /// ACES 1.3 Output Transform (RRT + SDR ODT), ported from the reference
    /// CTL. The default: the Academy's licence is permissive, so unlike
    /// OpenDRT this ships in the stock build, and a real standard transform
    /// is a better thing to open on than a curve fit.
    #[default]
    Aces13,
    /// OpenDRT v1.1.0. Requires the `opendrt` cargo feature (GPLv3 — see
    /// `viewtf/opendrt.rs`); without it this falls back to [`Self::Filmic`].
    OpenDrt,
    /// Blender's AgX, from the cubes in the user's Blender install (see
    /// [`agx`]). Without a Blender this falls back to [`Self::Filmic`].
    AgX,
}

impl ViewTransform {
    pub const ALL: [Self; 5] = [Self::None, Self::Filmic, Self::Aces13, Self::AgX, Self::OpenDrt];

    pub fn label(&self) -> &'static str {
        match self {
            Self::None => "None (clamp)",
            Self::Filmic => "Narkowicz fit (approx)",
            Self::Aces13 => "ACES 1.3 Output Transform",
            Self::AgX => {
                if agx::available() {
                    "AgX (Blender)"
                } else {
                    "AgX (Blender not found)"
                }
            }
            Self::OpenDrt => {
                if false {
                    "OpenDRT v1.1.0"
                } else {
                    "OpenDRT (build without feature)"
                }
            }
        }
    }

    /// True when selecting this actually gets you this. False for OpenDRT in a
    /// build without the feature, so the UI can say so instead of lying.
    pub fn available(&self) -> bool {
        match self {
            Self::OpenDrt => false,
            Self::AgX => agx::available(),
            _ => true,
        }
    }

    /// Short name for menus and preset labels.
    pub fn short(&self) -> &'static str {
        match self {
            Self::None => "Standard",
            Self::Filmic => "Narkowicz",
            Self::Aces13 => "ACES 1.3",
            Self::OpenDrt => "OpenDRT",
            Self::AgX => "AgX",
        }
    }

    /// The looks this transform offers, `Look::None` first. Blender's set:
    /// AgX's nine, ACES 1.3's gamut compression, nothing on the others.
    pub fn looks(&self) -> &'static [Look] {
        match self {
            Self::AgX => &Look::AGX,
            Self::Aces13 => &[Look::None, Look::GamutCompress],
            _ => &[Look::None],
        }
    }
}

/// A creative look applied inside the view transform, Blender's looks by
/// name and by value (`agx::look_params`, `aces::rgc`). Display state like
/// the transform itself: a uniform, never a recompile, never a cache key.
/// Each look belongs to one transform ([`ViewTransform::looks`]); under any
/// other it is off ([`ViewSettings::effective_look`]), so switching the
/// transform and back keeps the pick.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Look {
    #[default]
    None,
    /// AgX - Punchy: GradingTone shadows + a CDL power, darkening.
    Punchy,
    /// AgX - Greyscale: luminance in the AgX Log primaries.
    Greyscale,
    VeryHighContrast,
    HighContrast,
    MediumHighContrast,
    /// Contrast 1, but still the look path: the luminance-compensated AgX
    /// Log round trip, so saturated colours differ from `None` (as in
    /// Blender).
    BaseContrast,
    MediumLowContrast,
    LowContrast,
    VeryLowContrast,
    /// ACES 1.3 Reference Gamut Compression (the Academy's LMT).
    GamutCompress,
}

impl Look {
    /// AgX's looks, in Blender's menu order.
    pub const AGX: [Self; 10] = [
        Self::None,
        Self::Punchy,
        Self::Greyscale,
        Self::VeryHighContrast,
        Self::HighContrast,
        Self::MediumHighContrast,
        Self::BaseContrast,
        Self::MediumLowContrast,
        Self::LowContrast,
        Self::VeryLowContrast,
    ];

    /// Blender's name, without the "AgX - " family prefix.
    pub fn label(&self) -> &'static str {
        match self {
            Self::None => "None",
            Self::Punchy => "Punchy",
            Self::Greyscale => "Greyscale",
            Self::VeryHighContrast => "Very High Contrast",
            Self::HighContrast => "High Contrast",
            Self::MediumHighContrast => "Medium High Contrast",
            Self::BaseContrast => "Base Contrast",
            Self::MediumLowContrast => "Medium Low Contrast",
            Self::LowContrast => "Low Contrast",
            Self::VeryLowContrast => "Very Low Contrast",
            Self::GamutCompress => "Reference Gamut Compression",
        }
    }
}

/// Colour primaries.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Gamut {
    #[default]
    Rec709,
    AcesCg,
    Aces2065,
    P3D65,
    Rec2020,
    Xyz,
}

impl Gamut {
    pub const ALL: [Self; 6] = [
        Self::Rec709,
        Self::AcesCg,
        Self::Aces2065,
        Self::P3D65,
        Self::Rec2020,
        Self::Xyz,
    ];

    pub fn label(&self) -> &'static str {
        match self {
            Self::Rec709 => "Rec.709 / sRGB",
            Self::AcesCg => "ACEScg (AP1)",
            Self::Aces2065 => "ACES 2065-1 (AP0)",
            Self::P3D65 => "P3-D65",
            Self::Rec2020 => "Rec.2020",
            Self::Xyz => "CIE XYZ",
        }
    }

    pub fn to_xyz(&self) -> [[f32; 3]; 3] {
        match self {
            Self::Rec709 => REC709_TO_XYZ,
            Self::AcesCg => AP1_TO_XYZ,
            Self::Aces2065 => AP0_TO_XYZ,
            Self::P3D65 => P3D65_TO_XYZ,
            Self::Rec2020 => REC2020_TO_XYZ,
            Self::Xyz => IDENTITY3,
        }
    }

    /// Display gamuts the viewer can target. Rec.2020 is delivered
    /// P3-limited, matching OpenDRT's container behaviour.
    pub const DISPLAYS: [Self; 3] = [Self::Rec709, Self::P3D65, Self::Rec2020];

    /// The same primaries in `crate::color` terms.
    ///
    /// The matrices in THIS module are D65-normalised to match OpenDRT, so
    /// AP0/AP1 here are not the published ACES matrices. Anything that needs
    /// the real ones — the ACES transform, every input and output transform
    /// — goes through `color` instead, which derives them from published
    /// chromaticities and adapts white points explicitly.
    pub fn primaries(self) -> crate::color::Primaries {
        use crate::color::Primaries as P;
        match self {
            Self::Rec709 => P::Rec709,
            Self::AcesCg => P::Ap1,
            Self::Aces2065 => P::Ap0,
            Self::P3D65 => P::P3D65,
            Self::Rec2020 => P::Rec2020,
            Self::Xyz => P::Xyz,
        }
    }
}

/// Inverse display EOTF applied at the very end of the blit.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Eotf {
    /// Display-linear out. Only sensible for debugging.
    Linear,
    /// The sRGB piecewise curve.
    #[default]
    Srgb,
    /// Pure 2.2 power.
    Gamma22,
    /// Rec.1886, pure 2.4 power.
    Rec1886,
    /// DCI, pure 2.6 power.
    Dci,
}

impl Eotf {
    pub const ALL: [Self; 5] = [
        Self::Linear,
        Self::Srgb,
        Self::Gamma22,
        Self::Rec1886,
        Self::Dci,
    ];

    pub fn label(&self) -> &'static str {
        match self {
            Self::Linear => "Linear",
            Self::Srgb => "sRGB",
            Self::Gamma22 => "Gamma 2.2",
            Self::Rec1886 => "Rec.1886 (2.4)",
            Self::Dci => "DCI (2.6)",
        }
    }

    /// WGSL expression encoding display-linear `c` (a `vec3f`).
    fn wgsl_encode(&self) -> &'static str {
        match self {
            Self::Linear => "c",
            Self::Srgb => "vt_srgb_encode(c)",
            Self::Gamma22 => "pow(max(c, vec3f(0.0)), vec3f(1.0 / 2.2))",
            Self::Rec1886 => "pow(max(c, vec3f(0.0)), vec3f(1.0 / 2.4))",
            Self::Dci => "pow(max(c, vec3f(0.0)), vec3f(1.0 / 2.6))",
        }
    }

    /// The same encode on the CPU. Mirrors `wgsl_encode` — the pixel
    /// readout has to report the value that is really on screen, not an
    /// approximation of it.
    pub fn encode(self, c: f32) -> f32 {
        let c = c.max(0.0);
        match self {
            Self::Linear => c,
            Self::Srgb => {
                if c <= 0.0031308 {
                    c * 12.92
                } else {
                    1.055 * c.powf(1.0 / 2.4) - 0.055
                }
            }
            Self::Gamma22 => c.powf(1.0 / 2.2),
            Self::Rec1886 => c.powf(1.0 / 2.4),
            Self::Dci => c.powf(1.0 / 2.6),
        }
    }

    /// True when a hardware sRGB swapchain would encode this correctly on our
    /// behalf. Only sRGB itself qualifies.
    fn is_hardware_srgb(&self) -> bool {
        matches!(self, Self::Srgb)
    }
}

/// One entry of the viewer's transform menu: a view transform on a display
/// (primaries + encoding curve) — "ACES 1.3 on a Rec.709 TV", "AgX on an
/// sRGB monitor". The OCIO display/view pair, minus OCIO.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ViewPreset {
    pub transform: ViewTransform,
    pub display: Gamut,
    pub eotf: Eotf,
}

impl ViewPreset {
    pub const fn new(transform: ViewTransform, display: Gamut, eotf: Eotf) -> Self {
        Self { transform, display, eotf }
    }

    /// The catalogue offered in colour…, named the way Blender and others
    /// name them. The first few are the default menu ([`Self::DEFAULT_MENU`]).
    pub const CATALOGUE: [(&'static str, Self); 11] = [
        ("ACES 1.3 · sRGB", Self::new(ViewTransform::Aces13, Gamut::Rec709, Eotf::Srgb)),
        ("ACES 1.3 · Rec.709", Self::new(ViewTransform::Aces13, Gamut::Rec709, Eotf::Rec1886)),
        ("AgX · sRGB", Self::new(ViewTransform::AgX, Gamut::Rec709, Eotf::Srgb)),
        ("AgX · Rec.1886", Self::new(ViewTransform::AgX, Gamut::Rec709, Eotf::Rec1886)),
        ("Standard · sRGB", Self::new(ViewTransform::None, Gamut::Rec709, Eotf::Srgb)),
        ("ACES 1.3 · Display P3", Self::new(ViewTransform::Aces13, Gamut::P3D65, Eotf::Srgb)),
        ("AgX · Display P3", Self::new(ViewTransform::AgX, Gamut::P3D65, Eotf::Srgb)),
        ("AgX · Rec.2020", Self::new(ViewTransform::AgX, Gamut::Rec2020, Eotf::Rec1886)),
        ("Standard · Rec.709", Self::new(ViewTransform::None, Gamut::Rec709, Eotf::Rec1886)),
        ("Narkowicz · sRGB", Self::new(ViewTransform::Filmic, Gamut::Rec709, Eotf::Srgb)),
        ("OpenDRT · sRGB", Self::new(ViewTransform::OpenDrt, Gamut::Rec709, Eotf::Srgb)),
    ];

    /// How many of [`Self::CATALOGUE`] a fresh install offers.
    pub const DEFAULT_MENU: usize = 5;

    pub fn default_menu() -> Vec<Self> {
        Self::CATALOGUE[..Self::DEFAULT_MENU].iter().map(|(_, p)| *p).collect()
    }

    /// Its catalogue name, else "transform · display / curve".
    pub fn label(&self) -> String {
        for (name, p) in Self::CATALOGUE {
            if p == *self {
                return name.to_string();
            }
        }
        format!("{} · {} / {}", self.transform.short(), self.display.label(), self.eotf.label())
    }
}

/// Everything the display side needs. Lives in app prefs / setups, is passed
/// to the blit each frame, and is deliberately NOT part of the render-cache
/// key — cached frames are scene-linear.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct ViewSettings {
    pub transform: ViewTransform,
    /// Primaries the comp's scene-linear values are in.
    ///
    /// MIRRORED from the comp's working space, never authored here: the
    /// working space decides what the pixels ARE (and is in the render-cache
    /// key), while everything else on this struct decides how they are SHOWN
    /// (and is deliberately not). A second, editable copy in the view panel
    /// could disagree with the comp and silently mis-saturate the picture.
    /// Set it with [`ViewSettings::mirror_working`].
    pub working: Gamut,
    /// Primaries of the monitor.
    pub display: Gamut,
    pub eotf: Eotf,
    /// Viewer exposure in stops, applied before the transform.
    pub exposure: f32,
    /// Viewer gamma, applied to the DISPLAY-referred value after the
    /// transform and before the encode: `d^(1/gamma)`, 1 = off.
    pub gamma: f32,
    /// Blender's looks, for the transforms that have them (AgX, ACES 1.3).
    pub look: Look,
    pub opendrt: opendrt::OpenDrtParams,
}

impl Default for ViewSettings {
    fn default() -> Self {
        Self {
            transform: Default::default(),
            working: Default::default(),
            display: Default::default(),
            eotf: Default::default(),
            exposure: 0.0,
            gamma: 1.0,
            look: Look::None,
            opendrt: Default::default(),
        }
    }
}

impl ViewSettings {
    /// Track the comp's working space. Called every frame from the app; the
    /// mapping is total because the display side only cares about primaries,
    /// and a working space with an exotic transfer curve is still viewed
    /// through the primaries it uses.
    pub fn mirror_working(&mut self, working: crate::color::Colorspace) {
        self.working = match working.primaries {
            crate::color::Primaries::Ap0 => Gamut::Aces2065,
            crate::color::Primaries::Ap1 => Gamut::AcesCg,
            crate::color::Primaries::P3D65 | crate::color::Primaries::P3Dci => Gamut::P3D65,
            crate::color::Primaries::Rec2020 => Gamut::Rec2020,
            crate::color::Primaries::Xyz => Gamut::Xyz,
            // Camera gamuts are not working spaces; if someone forces one,
            // Rec.709 is the least surprising thing to show it through.
            _ => Gamut::Rec709,
        };
    }

    /// Which transform will actually run, accounting for a missing feature.
    pub fn effective(&self) -> ViewTransform {
        if self.transform.available() {
            self.transform
        } else {
            ViewTransform::Filmic
        }
    }

    /// The look that actually runs: the pick, if the transform that runs
    /// offers it.
    pub fn effective_look(&self) -> Look {
        if self.effective().looks().contains(&self.look) {
            self.look
        } else {
            Look::None
        }
    }

    /// Identity of the generated shader. The blit rebuilds its pipeline when
    /// this changes; everything else rides the uniform.
    pub fn shader_key(&self) -> u64 {
        let mut k = self.effective() as u64;
        k = k * 8 + self.display as u64;
        k = k * 8 + self.eotf as u64;
        k
    }

    /// The `view_transform` function plus whatever helpers it needs.
    pub fn wgsl(&self) -> String {
        match self.effective() {
            ViewTransform::None => r#"
fn view_transform(rgb: vec3f) -> vec3f {
    return clamp(u.in_to_rs * rgb, vec3f(0.0), vec3f(1.0));
}
"#
            .to_string(),
            ViewTransform::Filmic => format!(
                r#"
// ACES-ish filmic fit (Narkowicz). Display-linear out; the tail encodes.
// `in_to_rs` carries working -> display primaries for this path.
fn view_transform(rgb: vec3f) -> vec3f {{
    let x = max(u.in_to_rs * rgb, vec3f(0.0));
    let tone = (x * ({A} * x + {B})) / (x * ({C} * x + {D}) + {E});
    return clamp(tone, vec3f(0.0), vec3f(1.0));
}}
"#,
                A = 2.51, B = 0.03, C = 2.43, D = 0.59, E = 0.14
            ),
            ViewTransform::Aces13 => aces::wgsl(self.display),
            ViewTransform::OpenDrt => opendrt::wgsl(self.display),
            ViewTransform::AgX => agx::wgsl(),
        }
    }

    /// The 3D LUT the selected transform needs bound, if any: the display
    /// gamut whose AgX cube the blit must upload.
    pub fn lut_display(&self) -> Option<Gamut> {
        (self.effective() == ViewTransform::AgX).then_some(self.display)
    }

    /// This settings with `p`'s transform, display and curve.
    pub fn with_preset(mut self, p: ViewPreset) -> Self {
        self.transform = p.transform;
        self.display = p.display;
        self.eotf = p.eotf;
        self
    }

    /// The preset this is showing (transform, display, curve).
    pub fn preset(&self) -> ViewPreset {
        ViewPreset { transform: self.transform, display: self.display, eotf: self.eotf }
    }

    /// Matrices for the uniform, as WGSL column-major `mat3x3<f32>` rows.
    ///
    /// OpenDRT wants working -> P3-D65 (its rendering space) and does its own
    /// display conversion; the simpler transforms want working -> display
    /// directly, so `in_to_rs` means slightly different things per transform.
    /// Both are documented at their use site in the shader.
    /// Working primaries -> display primaries, row-major.
    ///
    /// Uses THIS module's D65-normalised set (see `Gamut::primaries`), which
    /// is what the None and Filmic shader paths get. Shared with
    /// `display_rgb` so the pixel readout reports the same numbers the
    /// picture was made from.
    fn working_to_display(&self) -> [[f32; 3]; 3] {
        let xyz_to_display = match self.display {
            Gamut::Rec709 => XYZ_TO_REC709,
            Gamut::Rec2020 => mat3_mul(P3_TO_REC2020, XYZ_TO_P3D65),
            _ => XYZ_TO_P3D65,
        };
        mat3_mul(xyz_to_display, self.working.to_xyz())
    }

    /// The whole display side on the CPU: exposure, view transform, inverse
    /// EOTF. Feeds the viewer's pixel readout.
    ///
    /// `None` for OpenDRT, which exists only as generated WGSL — a readout
    /// that quietly substituted a different curve would be worse than one
    /// that says it cannot tell you.
    pub fn display_rgb(&self, scene_linear: [f32; 3]) -> Option<[f32; 3]> {
        let g = 2f32.powf(self.exposure);
        let rgb = [scene_linear[0] * g, scene_linear[1] * g, scene_linear[2] * g];
        let d = match self.effective() {
            ViewTransform::OpenDrt => return None,
            ViewTransform::Aces13 => aces::apply_look(
                rgb,
                self.working.primaries(),
                self.display,
                self.effective_look() == Look::GamutCompress,
            ),
            ViewTransform::AgX => {
                agx::apply_look(rgb, self.working.primaries(), self.display, self.effective_look())?
            }
            ViewTransform::None => {
                let v = apply_row_major(self.working_to_display(), rgb);
                [v[0].clamp(0.0, 1.0), v[1].clamp(0.0, 1.0), v[2].clamp(0.0, 1.0)]
            }
            ViewTransform::Filmic => {
                let v = apply_row_major(self.working_to_display(), rgb);
                let f = |x: f32| {
                    let x = x.max(0.0);
                    ((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)).clamp(0.0, 1.0)
                };
                [f(v[0]), f(v[1]), f(v[2])]
            }
        };
        let d = d.map(|x| x.max(0.0).powf(self.inv_gamma()));
        Some([self.eotf.encode(d[0]), self.eotf.encode(d[1]), self.eotf.encode(d[2])])
    }

    /// The exponent the blit raises the display value to. A zero or negative
    /// gamma (an old pref, a typo'd setup) reads as off, never as black.
    pub fn inv_gamma(&self) -> f32 {
        if self.gamma > 1e-3 { 1.0 / self.gamma } else { 1.0 }
    }

    fn matrices(&self) -> (ShaderMat3, ShaderMat3, ShaderMat3) {
        match self.effective() {
            // ACES needs working -> AP0 in, AP1 -> display out, and it
            // needs them from `color`'s PUBLISHED matrices with a real
            // Bradford adaptation — not from this module's D65-normalised
            // OpenDRT set, which is a different (display-oriented) choice.
            ViewTransform::Aces13 => {
                use crate::color::{primaries_matrix, Primaries};
                let w = self.working.primaries();
                (
                    mat3_columns(primaries_matrix(w, Primaries::Ap0)),
                    mat3_columns(primaries_matrix(Primaries::Ap1, self.display.primaries())),
                    mat3_columns(IDENTITY3),
                )
            }
            ViewTransform::AgX => {
                let ident = mat3_columns(IDENTITY3);
                (mat3_columns(agx::in_matrix(self.working.primaries())), ident, ident)
            }
            ViewTransform::OpenDrt => {
                let in_to_rs = mat3_mul(XYZ_TO_P3D65, self.working.to_xyz());
                let (adapted, neutral) = opendrt::display_matrices(&self.opendrt, self.display);
                (
                    mat3_columns(in_to_rs),
                    mat3_columns(adapted),
                    mat3_columns(neutral),
                )
            }
            _ => {
                let m = self.working_to_display();
                let ident = mat3_columns(IDENTITY3);
                (mat3_columns(m), ident, ident)
            }
        }
    }

    /// True when the swapchain's own sRGB encode should be used instead of
    /// ours. Only valid when the chosen EOTF *is* sRGB.
    pub fn hardware_encode_ok(&self) -> bool {
        self.eotf.is_hardware_srgb()
    }

    pub fn encode_expr(&self, target_is_srgb: bool) -> &'static str {
        if target_is_srgb && self.eotf.is_hardware_srgb() {
            "c" // hardware encodes
        } else {
            self.eotf.wgsl_encode()
        }
    }

    /// Fill the transform half of the blit uniform.
    pub fn fill_uniform(&self, u: &mut BlitUniform) {
        let (a, b, c) = self.matrices();
        u.in_to_rs = a;
        u.out_cwp = b;
        u.out_neu = c;
        u.drt = match self.effective() {
            ViewTransform::OpenDrt => opendrt::pack(&self.opendrt, self.display),
            ViewTransform::AgX => {
                let mut d = [[0.0; 4]; opendrt::SLOTS];
                d[..agx::SLOTS].copy_from_slice(&agx::pack(self.effective_look(), self.display));
                d
            }
            ViewTransform::Aces13 => {
                let mut d = [[0.0; 4]; opendrt::SLOTS];
                d[0][0] = (self.effective_look() == Look::GamutCompress) as u8 as f32;
                d
            }
            _ => [[0.0; 4]; opendrt::SLOTS],
        };
    }
}

/// A 3x3 matrix in WGSL `mat3x3<f32>` storage: column-major, each column
/// padded to 16 bytes.
pub type ShaderMat3 = [[f32; 4]; 3];

/// Blit uniform. Layout mirrors the WGSL `U` struct exactly — 480 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BlitUniform {
    /// Destination rect in target pixels: xy = min, zw = size.
    pub dest: [f32; 4],
    /// x = show alpha as grey, y = exposure multiplier, z = 1/gamma,
    /// w = single channel (0 = rgb, 1/2/3 = r/g/b as grey).
    pub mode: [f32; 4],
    /// Written as-is outside `dest`: already encoded for the target.
    pub backdrop: [f32; 4],
    pub in_to_rs: ShaderMat3,
    pub out_cwp: ShaderMat3,
    pub out_neu: ShaderMat3,
    pub drt: [[f32; 4]; opendrt::SLOTS],
}

impl Default for BlitUniform {
    fn default() -> Self {
        Self {
            dest: [0.0; 4],
            mode: [0.0, 1.0, 1.0, 0.0],
            backdrop: [0.0, 0.0, 0.0, 1.0],
            in_to_rs: mat3_columns(IDENTITY3),
            out_cwp: mat3_columns(IDENTITY3),
            out_neu: mat3_columns(IDENTITY3),
            drt: [[0.0; 4]; opendrt::SLOTS],
        }
    }
}

/// The WGSL struct declaration + shared helpers the blit shader prepends.
pub fn uniform_wgsl() -> String {
    format!(
        r#"
struct U {{
    dest: vec4f,
    mode: vec4f,
    backdrop: vec4f,
    in_to_rs: mat3x3<f32>,
    out_cwp: mat3x3<f32>,
    out_neu: mat3x3<f32>,
    drt: array<vec4f, {slots}>,
}};

fn vt_srgb_encode(c: vec3f) -> vec3f {{
    let x = max(c, vec3f(0.0));
    let lo = x * 12.92;
    let hi = 1.055 * pow(x, vec3f(1.0 / 2.4)) - 0.055;
    return select(hi, lo, x <= vec3f(0.0031308));
}}
"#,
        slots = opendrt::SLOTS
    )
}

/// Approximate display luminance of a scene-linear colour, for overlays that
/// only need to decide "is this pixel light or dark" — the viewer's adaptive
/// ink and the HUD luma probes.
///
/// Deliberately the fixed Narkowicz curve rather than the live view transform:
/// it keeps the decision stable while the user auditions transforms, it needs
/// no uniforms, and a black/white pick does not warrant a full DRT per tap.
pub fn display_luma(rgb: [f32; 3]) -> f32 {
    let tone = |x: f32| {
        let x = x.max(0.0);
        ((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)).clamp(0.0, 1.0)
    };
    0.2126 * tone(rgb[0]) + 0.7152 * tone(rgb[1]) + 0.0722 * tone(rgb[2])
}

/// WGSL for the same estimate: `vt_display_tone` for shaders that need to
/// *show* scene-linear pixels (the HUD frost), `vt_display_luma` for shaders
/// that only need to decide light-or-dark (the adaptive ink).
pub const DISPLAY_ESTIMATE_WGSL: &str = r#"
// Fixed filmic estimate of what a scene-linear colour will look like on
// screen. Not the live view transform on purpose: overlay decisions stay
// stable while the user auditions transforms, and it needs no uniforms.
fn vt_display_tone(rgb: vec3f) -> vec3f {
    let x = max(rgb, vec3f(0.0));
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14),
                 vec3f(0.0), vec3f(1.0));
}
fn vt_display_luma(rgb: vec3f) -> f32 {
    return dot(vt_display_tone(rgb), vec3f(0.2126, 0.7152, 0.0722));
}
"#;

// ---------------------------------------------------------------------------
// Colour matrices. Row-major, m[row][col]. Values match the published
// primaries (and OpenDRT.dctl, which uses the same set).
//
// Digits are kept as published rather than trimmed to f32's shortest
// round-trip form, so these can be diffed against upstream by eye — hence
// the excessive_precision allow at the top of the module.
// ---------------------------------------------------------------------------

pub const IDENTITY3: [[f32; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

pub const REC709_TO_XYZ: [[f32; 3]; 3] = [
    [0.412_390_8, 0.357_584_33, 0.180_480_79],
    [0.212_639, 0.715_168_7, 0.072_192_32],
    [0.019_330_82, 0.119_194_78, 0.950_532_15],
];
pub const XYZ_TO_REC709: [[f32; 3]; 3] = [
    [3.240_97, -1.537_383_2, -0.498_610_76],
    [-0.969_243_65, 1.875_967_5, 0.041_555_06],
    [0.055_630_08, -0.203_976_96, 1.056_971_5],
];
pub const P3D65_TO_XYZ: [[f32; 3]; 3] = [
    [0.486_570_95, 0.265_667_7, 0.198_217_29],
    [0.228_974_56, 0.691_738_5, 0.079_286_91],
    [0.0, 0.045_113_38, 1.043_944_4],
];
pub const XYZ_TO_P3D65: [[f32; 3]; 3] = [
    [2.493_496_9, -0.931_383_6, -0.402_710_79],
    [-0.829_488_97, 1.762_664_1, 0.023_624_686],
    [0.035_845_83, -0.076_172_39, 0.956_884_5],
];
pub const AP0_TO_XYZ: [[f32; 3]; 3] = [
    [0.938_630_94, -0.005_741_92, 0.017_566_899],
    [0.338_093_6, 0.727_213_9, -0.065_307_5],
    [0.000_723_121_5, 0.000_818_441_9, 1.087_516_2],
];
pub const AP1_TO_XYZ: [[f32; 3]; 3] = [
    [0.652_418_7, 0.127_179_93, 0.170_857_28],
    [0.268_064_06, 0.672_464_5, 0.059_471_46],
    [-0.005_469_928_5, 0.005_182_8, 1.089_344_9],
];
pub const REC2020_TO_XYZ: [[f32; 3]; 3] = [
    [0.636_958_05, 0.144_616_9, 0.168_880_98],
    [0.262_700_2, 0.677_998_1, 0.059_301_715],
    [0.0, 0.028_072_693, 1.060_985_1],
];
pub const P3_TO_REC2020: [[f32; 3]; 3] = [
    [0.753_833_04, 0.198_597_37, 0.047_569_597],
    [0.045_743_85, 0.941_777_2, 0.012_478_931],
    [-0.001_210_340_4, 0.017_601_717, 0.983_608_6],
];

// Bradford chromatic adaptation from D65 to the creative whites OpenDRT
// offers. Same constants as OpenDRT.dctl.
pub const CAT_D65_TO_D93: [[f32; 3]; 3] = [
    [0.957_034_2, -0.024_717_15, 0.062_402_86],
    [-0.017_929_696, 0.990_019_9, 0.024_811_953],
    [0.001_275_891_4, 0.004_279_190_7, 1.293_457_2],
];
pub const CAT_D65_TO_D75: [[f32; 3]; 3] = [
    [0.981_001_1, -0.011_661_925, 0.026_561_409],
    [-0.008_434_881, 0.996_506_1, 0.010_569_654],
    [0.000_552_809_57, 0.001_798_408_1, 1.123_747_2],
];
pub const CAT_D65_TO_D60: [[f32; 3]; 3] = [
    [1.011_822_5, 0.007_788_793, -0.015_778_303],
    [0.005_616_828_4, 1.001_506_4, -0.006_285_176],
    [-0.000_335_735_73, -0.001_050_95, 0.927_366_7],
];
pub const CAT_D65_TO_D55: [[f32; 3]; 3] = [
    [1.025_850_9, 0.017_943_982, -0.033_213_78],
    [0.012_913_385, 1.002_147_8, -0.013_242_103],
    [-0.000_719_940_3, -0.002_181_068, 0.848_680_14],
];
pub const CAT_D65_TO_D50: [[f32; 3]; 3] = [
    [1.042_574, 0.030_891_176, -0.052_812_62],
    [0.022_193_536, 1.001_856_7, -0.021_073_762],
    [-0.001_164_883_1, -0.003_420_527, 0.761_789_1],
];

fn apply_row_major(m: [[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// Row-major 3x3 multiply: `a * b`.
pub fn mat3_mul(a: [[f32; 3]; 3], b: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut out = [[0.0f32; 3]; 3];
    for (i, row) in out.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = (0..3).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    out
}

/// Row-major matrix -> WGSL `mat3x3<f32>` storage (column-major, each column
/// padded to 16 bytes).
pub fn mat3_columns(m: [[f32; 3]; 3]) -> ShaderMat3 {
    [
        [m[0][0], m[1][0], m[2][0], 0.0],
        [m[0][1], m[1][1], m[2][1], 0.0],
        [m[0][2], m[1][2], m[2][2], 0.0],
    ]
}

#[cfg(test)]
mod tests {

    /// A look is display state: it rides the uniform, so picking one never
    /// rebuilds the blit, and one the transform does not offer is off.
    #[test]
    fn a_look_is_a_uniform_and_belongs_to_its_transform() {
        let mut v = ViewSettings { transform: ViewTransform::Aces13, ..Default::default() };
        let key = v.shader_key();
        v.look = Look::GamutCompress;
        assert_eq!(v.shader_key(), key);
        assert_eq!(v.effective_look(), Look::GamutCompress);
        let mut u = BlitUniform::default();
        v.fill_uniform(&mut u);
        assert_eq!(u.drt[0][0], 1.0);
        v.look = Look::Punchy;
        assert_eq!(v.effective_look(), Look::None, "Punchy is AgX's");
        v.fill_uniform(&mut u);
        assert_eq!(u.drt[0][0], 0.0);
        for t in ViewTransform::ALL {
            assert_eq!(t.looks()[0], Look::None, "{t:?}");
        }
        assert_eq!(ViewTransform::AgX.looks().len(), 10, "Blender's nine AgX looks + None");
    }
    use super::*;

    fn apply(m: [[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
        [
            m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
            m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
            m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
        ]
    }

    #[test]
    fn rec709_roundtrips_through_xyz() {
        let m = mat3_mul(XYZ_TO_REC709, REC709_TO_XYZ);
        for i in 0..3 {
            for j in 0..3 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((m[i][j] - want).abs() < 1e-4, "{i},{j} = {}", m[i][j]);
            }
        }
    }

    #[test]
    fn p3d65_roundtrips_through_xyz() {
        let m = mat3_mul(XYZ_TO_P3D65, P3D65_TO_XYZ);
        for i in 0..3 {
            for j in 0..3 {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((m[i][j] - want).abs() < 1e-4, "{i},{j} = {}", m[i][j]);
            }
        }
    }

    #[test]
    fn white_maps_to_white_between_gamuts() {
        // Rec.709 and P3-D65 share D65, so 1,1,1 must survive the trip.
        let m = mat3_mul(XYZ_TO_P3D65, REC709_TO_XYZ);
        let w = apply(m, [1.0, 1.0, 1.0]);
        for c in w {
            assert!((c - 1.0).abs() < 1e-3, "white drifted: {w:?}");
        }
    }

    #[test]
    fn uniform_matches_wgsl_layout() {
        // 3 vec4 + 3 mat3x3 (48 each) + SLOTS vec4.
        assert_eq!(
            std::mem::size_of::<BlitUniform>(),
            48 + 144 + opendrt::SLOTS * 16
        );
    }

    #[test]
    fn display_luma_is_monotonic_and_bounded() {
        // Never decreasing across the useful range, always a valid 0..1 pick,
        // and genuinely discriminating around mid grey — the ink and the HUD
        // probes both threshold on this.
        let mut prev = -1.0;
        for stop in -8..8 {
            let v = 0.18 * 2.0f32.powi(stop);
            let l = display_luma([v, v, v]);
            assert!((0.0..=1.0).contains(&l), "out of range at {stop}: {l}");
            assert!(l >= prev, "not monotonic at {stop}");
            prev = l;
        }
        assert!(display_luma([0.02, 0.02, 0.02]) < 0.2, "shadows read dark");
        assert!(display_luma([1.0, 1.0, 1.0]) > 0.6, "highlights read bright");
    }

    #[test]
    fn opendrt_falls_back_without_its_feature() {
        let s = ViewSettings {
            transform: ViewTransform::OpenDrt,
            ..Default::default()
        };
        if false {
            assert_eq!(s.effective(), ViewTransform::OpenDrt);
        } else {
            assert_eq!(s.effective(), ViewTransform::Filmic);
        }
        // Either way the shader has to declare the entry point.
        assert!(s.wgsl().contains("fn view_transform"));
    }
}
