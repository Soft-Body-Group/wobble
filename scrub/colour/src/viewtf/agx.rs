//! AgX, as Blender ships it ("AgX" view, Blender 4.0+ / EaryChow's AgX).
//!
//! Blender's AgX is not a formula, it is a LUT: its OCIO config
//! (`datafiles/colormanagement/config.ocio`, "AgX Base Rec.1886") does
//!
//! ```text
//! scene -> Linear FilmLight E-Gamut -> log2, -12.47393..+12.5260688 stops -> 0..1
//!       -> AgX_Base_sRGB.cube (57^3, tetrahedral) -> Rec.709 encoded with 2.4
//! ```
//!
//! and the display then decodes the 2.4 and encodes for itself (piecewise
//! sRGB on Blender's default "sRGB" display, 2.4 again on "Rec.1886"). This
//! module does exactly that, with the cube read FROM THE USER'S BLENDER
//! INSTALL rather than copied into wobble: the pictures match the Blender
//! they have by construction, and nothing of Blender's is redistributed.
//! No Blender, no AgX — [`available`] says so and the viewer falls back.
//!
//! Display P3 and Rec.2020 use Blender's own P3 / Rec.2020 cubes, which
//! bake their gamut in the same way.
//!
//! ## Looks
//!
//! Blender's nine AgX looks (Punchy, Greyscale, and seven contrasts) are
//! OCIO grading ops applied in the config's "AgX Log" space and then taken
//! back to scene-linear BEFORE the base view runs:
//!
//! ```text
//! scene -> E-Gamut -> log2 -> luminance_compensation_bt2020.cube -> linear
//!       -> Rec.2020 -> the AgX inset matrix -> log2      ("AgX Log")
//!       -> the look -> back down the same road to E-Gamut -> the base view
//! ```
//!
//! The compensation cube is read from the user's Blender like the base
//! cube; the looks' numbers are the config's, and the grading maths is
//! OCIO's (`GradingPrimary` log style, `GradingTone` shadows, a CDL power),
//! ported here. Even "Base Contrast" is not "None": the compensation round
//! trip moves saturated colours, as it does in Blender.

use super::{Gamut, Look};
use crate::color::{primaries_matrix, Primaries};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// The config's `AllocationTransform {allocation: lg2, vars: [...]}`.
pub const LOG2_MIN: f32 = -12.47393;
pub const LOG2_MAX: f32 = 12.526_068_811_7;

/// CIE XYZ (D65) -> Linear FilmLight E-Gamut, row-major, verbatim from
/// Blender's config ("Linear FilmLight E-Gamut").
pub const XYZ_TO_EGAMUT: [[f32; 3]; 3] = [
    [1.5250528, -0.3159135, -0.1226583],
    [-0.5091526, 1.3333274, 0.1382844],
    [0.0957153, 0.0508974, 0.7879558],
];

/// Rec.2020 primaries from CIE XYZ (D65), verbatim from Blender's config
/// ("Linear Rec.2020"); the config's XYZ is the same space E-Gamut's is.
pub const XYZ_TO_REC2020: [[f32; 3]; 3] = [
    [1.7166512, -0.3556708, -0.2533663],
    [-0.6666844, 1.6164812, 0.0157685],
    [0.0176399, -0.0427706, 0.9421031],
];

/// "AgX Log"'s chroma inset and rotation, applied to linear Rec.2020,
/// verbatim from the config.
pub const INSET: [[f32; 3]; 3] = [
    [0.856627153315983, 0.0951212405381588, 0.0482516061458583],
    [0.137318972929847, 0.761241990602591, 0.101439036467562],
    [0.11189821299995, 0.0767994186031903, 0.811302368396859],
];

/// The Greyscale look's luminance weights in the inset space (the config's
/// matrix, every row the same).
pub const GREY_WEIGHTS: [f32; 3] = [0.2589235355689848, 0.6104985346066525, 0.13057792982436284];

/// The cube Blender's "Luminance Compensation Rec.2020" space runs on.
pub const LUMCOMP_FILE: &str = "luminance_compensation_bt2020.cube";

/// Uniform slots the AgX path takes from the blit's `drt` block.
pub const SLOTS: usize = 2;

/// How a look grades in AgX Log, with its numbers from Blender's config.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LookOp {
    /// No look: the base view alone, no compensation round trip.
    Off,
    /// `GradingPrimaryTransform`, log style, pivot -0.2 (0.4 encoded):
    /// contrast about the pivot, then saturation with Rec.709 weights.
    Primary { contrast: f32, saturation: f32 },
    Greyscale,
    /// `GradingToneTransform` shadows (rgb 0.2, master 0.35, start 0.4,
    /// pivot 0.1), then a CDL power of 1.0912.
    Punchy,
}

/// What `look` does under AgX (`Off` for a look that is not AgX's).
pub fn look_op(look: Look) -> LookOp {
    let p = |contrast, saturation| LookOp::Primary { contrast, saturation };
    match look {
        Look::Punchy => LookOp::Punchy,
        Look::Greyscale => LookOp::Greyscale,
        Look::VeryHighContrast => p(1.57, 0.9),
        Look::HighContrast => p(1.4, 0.95),
        Look::MediumHighContrast => p(1.2, 1.0),
        Look::BaseContrast => p(1.0, 1.0),
        Look::MediumLowContrast => p(0.9, 1.05),
        Look::LowContrast => p(0.8, 1.1),
        Look::VeryLowContrast => p(0.7, 1.15),
        Look::None | Look::GamutCompress => LookOp::Off,
    }
}

/// The log-style GradingPrimary pivot: 0.5 + pivot * 0.5 with the config's
/// -0.2. It is also 0.18's log2 allocation, so mid grey holds still.
const PIVOT: f32 = 0.4;
const PUNCHY_POWER: f32 = 1.0912;
/// GradingTone shadows: (start, pivot) and the rgb then master values.
const SHADOW_START: f32 = 0.4;
const SHADOW_PIVOT: f32 = 0.1;
const SHADOW_RGB: f32 = 0.2;
const SHADOW_MASTER: f32 = 0.35;

/// OCIO's forward shadow curve (`ComputeHSFwd`, val < 1): a quadratic
/// spline from slope `val` at the pivot to slope 1 at the start, linear
/// past both ends.
fn shadow(t: f32, val: f32) -> f32 {
    let (x0, x2) = (SHADOW_PIVOT, SHADOW_START);
    let x1 = x0 + (x2 - x0) * 0.5;
    let (y0, y2, m0, m2) = (x0, x2, val.max(0.01), 1.0);
    let y1 = (0.5 / (x2 - x0))
        * ((2.0 * y0 + m0 * (x1 - x0)) * (x2 - x1) + (2.0 * y2 - m2 * (x2 - x1)) * (x1 - x0));
    if t < x0 {
        (t - x0) * m0 + y0
    } else if t >= x2 {
        (t - x2) * m2 + y2
    } else if t < x1 {
        let tl = (t - x0) / (x1 - x0);
        y0 * (1.0 - tl * tl) + y1 * tl * tl + m0 * (1.0 - tl) * tl * (x1 - x0)
    } else {
        let tr = (t - x1) / (x2 - x1);
        y1 * (1.0 - tr) * (1.0 - tr) + y2 * (2.0 - tr) * tr + m2 * (tr - 1.0) * tr * (x2 - x1)
    }
}

/// The look itself, AgX Log in and out.
fn grade(op: LookOp, c: [f32; 3]) -> [f32; 3] {
    match op {
        LookOp::Off | LookOp::Greyscale => c,
        LookOp::Primary { contrast, saturation } => {
            let x = c.map(|v| (v - PIVOT) * contrast + PIVOT);
            let l = 0.2126 * x[0] + 0.7152 * x[1] + 0.0722 * x[2];
            x.map(|v| l + saturation * (v - l))
        }
        LookOp::Punchy => c
            .map(|v| shadow(shadow(v, SHADOW_RGB), SHADOW_MASTER))
            // CDL, no-clamp style: a negative passes through.
            .map(|v| if v > 0.0 { v.powf(PUNCHY_POWER) } else { v }),
    }
}

/// Linear E-Gamut -> the inset Rec.2020 "AgX Log" is the log2 of.
pub fn to_look_space() -> [[f32; 3]; 3] {
    super::mat3_mul(INSET, super::mat3_mul(XYZ_TO_REC2020, crate::color::mat3_inverse(XYZ_TO_EGAMUT)))
}

pub use crate::lut::Lut3d;

/// Blender's cube for a display gamut.
pub fn lut_file(display: Gamut) -> &'static str {
    match display {
        Gamut::P3D65 => "AgX_Base_P3.cube",
        Gamut::Rec2020 => "AgX_Base_Rec2020.cube",
        _ => "AgX_Base_sRGB.cube",
    }
}

/// Where Blender keeps its OCIO LUTs: `WOBBLE_BLENDER_LUTS` if set, else
/// the newest Blender found in the usual install places.
pub fn lut_dir() -> Option<PathBuf> {
    if let Some(d) = std::env::var_os("WOBBLE_BLENDER_LUTS") {
        return Some(PathBuf::from(d));
    }
    let tail = ["datafiles", "colormanagement", "luts"];
    let mut roots: Vec<PathBuf> = vec![
        "/Applications/Blender.app/Contents/Resources".into(),
        "/usr/share/blender".into(),
        "/opt/blender".into(),
    ];
    if let Some(h) = std::env::var_os("HOME") {
        roots.push(PathBuf::from(h).join("Applications/Blender.app/Contents/Resources"));
    }
    // Windows: "Blender Foundation/Blender 4.2/4.2/datafiles/...".
    if let Ok(rd) = std::fs::read_dir("C:\\Program Files\\Blender Foundation") {
        roots.extend(rd.flatten().map(|e| e.path()));
    }
    let mut found: Vec<PathBuf> = roots
        .iter()
        .filter_map(|r| std::fs::read_dir(r).ok())
        .flat_map(|rd| rd.flatten().map(|e| e.path()))
        .map(|v| tail.iter().fold(v, |p, t| p.join(t)))
        .filter(|p| p.join(lut_file(Gamut::Rec709)).is_file())
        .collect();
    // Newest version directory wins ("4.2" < "5.2"; numeric per part).
    let ver = |p: &PathBuf| -> Vec<u32> {
        p.ancestors()
            .nth(3)
            .and_then(|v| v.file_name())
            .map(|n| n.to_string_lossy().split('.').filter_map(|s| s.parse().ok()).collect())
            .unwrap_or_default()
    };
    found.sort_by_key(ver);
    found.pop()
}

type Cache = Mutex<HashMap<&'static str, Option<Arc<Lut3d>>>>;
static CACHE: std::sync::OnceLock<Cache> = std::sync::OnceLock::new();

/// Blender's AgX cube for `display`, loaded once (a 57^3 cube is ~2MB of
/// text). `None` when there is no Blender to read it from.
pub fn lut(display: Gamut) -> Option<Arc<Lut3d>> {
    load(lut_file(display))
}

/// The luminance-compensation cube the looks run through. `None` without
/// a Blender, or with one too old to ship it (the looks are then off).
pub fn lumcomp() -> Option<Arc<Lut3d>> {
    load(LUMCOMP_FILE)
}

fn load(file: &'static str) -> Option<Arc<Lut3d>> {
    let cache = CACHE.get_or_init(Default::default);
    if let Some(hit) = cache.lock().ok()?.get(file) {
        return hit.clone();
    }
    let loaded = lut_dir()
        .map(|d| d.join(file))
        .and_then(|p| match std::fs::read_to_string(&p).map_err(anyhow::Error::from).and_then(|t| Lut3d::parse_cube(&t)) {
            Ok(l) => {
                log::info!("AgX: {}", p.display());
                Some(Arc::new(l))
            }
            Err(e) => {
                log::warn!("AgX: reading {}: {e:#}", p.display());
                None
            }
        });
    cache.lock().ok()?.insert(file, loaded.clone());
    loaded
}

/// True when this machine has Blender's AgX to show.
pub fn available() -> bool {
    lut(Gamut::Rec709).is_some()
}

/// Working primaries -> Linear E-Gamut, row-major. Working -> Rec.709 with
/// a Bradford adaptation (Blender's ACES->XYZ path is Bradford too), then
/// Rec.709 -> XYZ D65 -> E-Gamut through the config's own matrix.
pub fn in_matrix(working: Primaries) -> [[f32; 3]; 3] {
    let to709 = primaries_matrix(working, Primaries::Rec709);
    super::mat3_mul(XYZ_TO_EGAMUT, super::mat3_mul(super::REC709_TO_XYZ, to709))
}

/// The log2 allocation, 0..1.
pub fn log_encode(x: f32) -> f32 {
    ((x.max(f32::MIN_POSITIVE).log2() - LOG2_MIN) / (LOG2_MAX - LOG2_MIN)).clamp(0.0, 1.0)
}

/// The log2 allocation without the clamp: what OCIO's `AllocationTransform`
/// does mid-chain, where the next step is not a cube.
fn log_open(x: f32) -> f32 {
    (x.max(f32::MIN_POSITIVE).log2() - LOG2_MIN) / (LOG2_MAX - LOG2_MIN)
}

fn log_decode(y: f32) -> f32 {
    (y * (LOG2_MAX - LOG2_MIN) + LOG2_MIN).exp2()
}

fn mat_apply(m: [[f32; 3]; 3], v: [f32; 3]) -> [f32; 3] {
    [0, 1, 2].map(|i| m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2])
}

/// A look's whole round trip on linear E-Gamut: into AgX Log, the grade,
/// back out. `None` when the compensation cube is missing.
pub fn look_egamut(e: [f32; 3], op: LookOp) -> Option<[f32; 3]> {
    if op == LookOp::Off {
        return Some(e);
    }
    let lc = lumcomp()?;
    let to = to_look_space();
    let c = mat_apply(to, lc.sample(e.map(log_encode)).map(log_decode));
    let c = match op {
        LookOp::Greyscale => {
            let y = GREY_WEIGHTS[0] * c[0] + GREY_WEIGHTS[1] * c[1] + GREY_WEIGHTS[2] * c[2];
            [y; 3]
        }
        _ => grade(op, c.map(log_open)).map(log_decode),
    };
    Some(mat_apply(crate::color::mat3_inverse(to), c))
}

/// The whole transform on the CPU: scene-linear in `working` -> display-
/// LINEAR in `display` (the blit tail encodes). `None` without Blender.
pub fn apply(rgb: [f32; 3], working: Primaries, display: Gamut) -> Option<[f32; 3]> {
    apply_look(rgb, working, display, Look::None)
}

/// [`apply`] with one of Blender's looks.
pub fn apply_look(rgb: [f32; 3], working: Primaries, display: Gamut, look: Look) -> Option<[f32; 3]> {
    let l = lut(display)?;
    let e = mat_apply(in_matrix(working), rgb);
    // A look whose cube is missing is off, as the shader has it.
    let e = look_egamut(e, look_op(look)).unwrap_or(e);
    let out = l.sample(e.map(log_encode));
    Some(out.map(|v| v.max(0.0).powf(2.4)))
}

/// The cubes the blit binds at `lut3d`, stacked in depth: the base view's
/// at the origin, the compensation cube (when there is one) from z = the
/// base's size. One texture, so the blit's four bindings stay four.
pub struct LutStack {
    pub extent: [u32; 3],
    /// rgb, `extent` order, red fastest; unused texels are zero.
    pub data: Vec<[f32; 3]>,
}

pub fn lut_stack(display: Gamut) -> Option<LutStack> {
    let base = lut(display)?;
    let lc = lumcomp();
    let n = base.size;
    let m = lc.as_ref().map_or(0, |l| l.size);
    let (w, d) = (n.max(m), n + m);
    let mut data = vec![[0.0; 3]; w * w * d];
    for (cube, z0) in std::iter::once((&base, 0)).chain(lc.as_ref().map(|l| (l, n))) {
        let s = cube.size;
        for b in 0..s {
            for g in 0..s {
                for r in 0..s {
                    data[r + w * (g + w * (b + z0))] = cube.at(r, g, b);
                }
            }
        }
    }
    Some(LutStack { extent: [w as u32, w as u32, d as u32], data })
}

/// The AgX uniform slots: `[kind, contrast, saturation, 0]` (kind 0 off,
/// 1 primary, 2 greyscale, 3 punchy) and `[base n, compensation n, 0, 0]`.
pub fn pack(look: Look, display: Gamut) -> [[f32; 4]; SLOTS] {
    let n = lut(display).map_or(0, |l| l.size) as f32;
    let m = lumcomp().map_or(0, |l| l.size) as f32;
    let op = if m > 0.0 { look_op(look) } else { LookOp::Off };
    let row = match op {
        LookOp::Off => [0.0, 1.0, 1.0, 0.0],
        LookOp::Primary { contrast, saturation } => [1.0, contrast, saturation, 0.0],
        LookOp::Greyscale => [2.0, 1.0, 1.0, 0.0],
        LookOp::Punchy => [3.0, 1.0, 1.0, 0.0],
    };
    [row, [n, m, 0.0, 0.0]]
}

/// `view_transform` for the blit. The cube is bound at `lut3d` (binding 3)
/// as an rgba32float 3D texture, read with `textureLoad`, interpolated
/// tetrahedrally exactly as [`Lut3d::sample`].
pub fn wgsl() -> String {
    format!(
        r#"
fn agx_at(i: vec3i) -> vec3f {{
    return textureLoad(lut3d, i, 0).rgb;
}}
// Tetrahedral lookup in the n^3 cube stacked from depth z0 (`LutStack`).
fn agx_cube(c: vec3f, n: i32, z0: i32) -> vec3f {{
    let p = clamp(c, vec3f(0.0), vec3f(1.0)) * f32(n - 1);
    let i0 = min(vec3i(floor(p)), vec3i(n - 2));
    let f = p - vec3f(i0);
    let i = i0 + vec3i(0, 0, z0);
    let c000 = agx_at(i);
    let c111 = agx_at(i + vec3i(1, 1, 1));
    if (f.x > f.y) {{
        if (f.y > f.z) {{
            return c000 * (1.0 - f.x) + agx_at(i + vec3i(1, 0, 0)) * (f.x - f.y) + agx_at(i + vec3i(1, 1, 0)) * (f.y - f.z) + c111 * f.z;
        }} else if (f.x > f.z) {{
            return c000 * (1.0 - f.x) + agx_at(i + vec3i(1, 0, 0)) * (f.x - f.z) + agx_at(i + vec3i(1, 0, 1)) * (f.z - f.y) + c111 * f.y;
        }} else {{
            return c000 * (1.0 - f.z) + agx_at(i + vec3i(0, 0, 1)) * (f.z - f.x) + agx_at(i + vec3i(1, 0, 1)) * (f.x - f.y) + c111 * f.y;
        }}
    }} else {{
        if (f.z > f.y) {{
            return c000 * (1.0 - f.z) + agx_at(i + vec3i(0, 0, 1)) * (f.z - f.y) + agx_at(i + vec3i(0, 1, 1)) * (f.y - f.x) + c111 * f.x;
        }} else if (f.z > f.x) {{
            return c000 * (1.0 - f.y) + agx_at(i + vec3i(0, 1, 0)) * (f.y - f.z) + agx_at(i + vec3i(0, 1, 1)) * (f.z - f.x) + c111 * f.x;
        }} else {{
            return c000 * (1.0 - f.y) + agx_at(i + vec3i(0, 1, 0)) * (f.y - f.x) + agx_at(i + vec3i(1, 1, 0)) * (f.x - f.z) + c111 * f.z;
        }}
    }}
}}
fn agx_log(x: vec3f) -> vec3f {{
    return (log2(max(x, vec3f(1.17549435e-38))) - ({lo})) / ({span});
}}
fn agx_unlog(y: vec3f) -> vec3f {{
    return exp2(y * ({span}) + ({lo}));
}}
// OCIO's forward GradingTone shadow curve (`agx::shadow`).
fn agx_shadow(t: vec3f, val: f32) -> vec3f {{
    let x0 = {sp};
    let x2 = {ss};
    let x1 = x0 + (x2 - x0) * 0.5;
    let m0 = max(val, 0.01);
    let y1 = (0.5 / (x2 - x0)) * ((2.0 * x0 + m0 * (x1 - x0)) * (x2 - x1) + (2.0 * x2 - (x2 - x1)) * (x1 - x0));
    let tl = (t - x0) / (x1 - x0);
    let tr = (t - x1) / (x2 - x1);
    let fl = x0 * (1.0 - tl * tl) + y1 * tl * tl + m0 * (1.0 - tl) * tl * (x1 - x0);
    let fr = y1 * (1.0 - tr) * (1.0 - tr) + x2 * (2.0 - tr) * tr + (tr - 1.0) * tr * (x2 - x1);
    var r = select(fr, fl, t < vec3f(x1));
    r = select(r, (t - x0) * m0 + x0, t < vec3f(x0));
    return select(t, r, t < vec3f(x2));
}}
const AGX_TO_LOOK: mat3x3<f32> = {to_look};
const AGX_FROM_LOOK: mat3x3<f32> = {from_look};
// A look (drt[0].x: 1 primary, 2 greyscale, 3 punchy) on linear E-Gamut,
// through Blender's "AgX Log" and back (`agx::look_egamut`).
fn agx_look(e: vec3f) -> vec3f {{
    let kind = i32(u.drt[0].x + 0.5);
    let lc = agx_unlog(agx_cube(agx_log(e), i32(u.drt[1].y), i32(u.drt[1].x)));
    var c = AGX_TO_LOOK * lc;
    if (kind == 2) {{
        c = vec3f(dot(c, vec3f({gw0}, {gw1}, {gw2})));
    }} else {{
        var x = agx_log(c);
        if (kind == 1) {{
            x = (x - vec3f({pivot})) * u.drt[0].y + vec3f({pivot});
            let l = dot(x, vec3f(0.2126, 0.7152, 0.0722));
            x = vec3f(l) + u.drt[0].z * (x - vec3f(l));
        }} else {{
            x = agx_shadow(agx_shadow(x, {srgb}), {smaster});
            x = select(x, pow(max(x, vec3f(0.0)), vec3f({power})), x > vec3f(0.0));
        }}
        c = agx_unlog(x);
    }}
    return AGX_FROM_LOOK * c;
}}
// Blender AgX: working -> E-Gamut (in_to_rs), the look if any, log2
// allocation, the cube, then decode the cube's 2.4 to display-linear; the
// tail encodes.
fn view_transform(rgb: vec3f) -> vec3f {{
    var e = u.in_to_rs * rgb;
    if (u.drt[0].x > 0.5) {{
        e = agx_look(e);
    }}
    let lg = clamp(agx_log(e), vec3f(0.0), vec3f(1.0));
    return pow(max(agx_cube(lg, i32(u.drt[1].x), 0), vec3f(0.0)), vec3f(2.4));
}}
"#,
        lo = LOG2_MIN,
        span = LOG2_MAX - LOG2_MIN,
        sp = SHADOW_PIVOT,
        ss = SHADOW_START,
        srgb = SHADOW_RGB,
        smaster = SHADOW_MASTER,
        power = PUNCHY_POWER,
        pivot = PIVOT,
        gw0 = GREY_WEIGHTS[0],
        gw1 = GREY_WEIGHTS[1],
        gw2 = GREY_WEIGHTS[2],
        to_look = mat_wgsl(to_look_space()),
        from_look = mat_wgsl(crate::color::mat3_inverse(to_look_space())),
    )
}

/// A row-major matrix as a WGSL `mat3x3<f32>` literal (column-major).
fn mat_wgsl(m: [[f32; 3]; 3]) -> String {
    let c = |j: usize| format!("vec3f({:?}, {:?}, {:?})", m[0][j], m[1][j], m[2][j]);
    format!("mat3x3<f32>({}, {}, {})", c(0), c(1), c(2))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp_cube(n: usize) -> Lut3d {
        let mut data = Vec::new();
        for b in 0..n {
            for g in 0..n {
                for r in 0..n {
                    let s = (n - 1) as f32;
                    // Something non-linear so interpolation is exercised.
                    data.push([(r as f32 / s).powi(2), g as f32 / s, (b as f32 / s).sqrt()]);
                }
            }
        }
        Lut3d::new(n, data)
    }

    #[test]
    fn parse_cube_reads_size_and_order() {
        let text = "TITLE \"t\"\n# c\nLUT_3D_SIZE 2\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 1 1 1\n\
                    0 0 0\n1 0 0\n0 1 0\n1 1 0\n0 0 1\n1 0 1\n0 1 1\n1 1 1\n";
        let l = Lut3d::parse_cube(text).unwrap();
        assert_eq!(l.size, 2);
        // Red fastest: the identity cube returns its input.
        for c in [[0.25, 0.5, 0.75], [0.9, 0.1, 0.4], [0.0, 1.0, 0.3]] {
            let o = l.sample(c);
            for k in 0..3 {
                assert!((o[k] - c[k]).abs() < 1e-6, "{c:?} -> {o:?}");
            }
        }
        assert!(Lut3d::parse_cube("LUT_3D_SIZE 3\n0 0 0\n").is_err());
    }

    /// On the lattice the interpolator returns the stored entry; between
    /// lattice points it stays inside the enclosing values.
    #[test]
    fn tetrahedral_hits_nodes_and_stays_bounded() {
        let l = ramp_cube(5);
        for (r, g, b) in [(0, 0, 0), (4, 4, 4), (1, 3, 2), (4, 0, 2)] {
            let c = [r as f32 / 4.0, g as f32 / 4.0, b as f32 / 4.0];
            let o = l.sample(c);
            let want = l.at(r, g, b);
            for k in 0..3 {
                assert!((o[k] - want[k]).abs() < 1e-6);
            }
        }
        let o = l.sample([0.33, 0.61, 0.07]);
        assert!(o.iter().all(|v| (0.0..=1.0).contains(v)));
    }

    #[test]
    fn log_allocation_matches_the_config() {
        // 0.18 sits log2(0.18) - LOG2_MIN stops up a 25-stop range.
        let want = (0.18f32.log2() + 12.47393) / 25.0000;
        assert!((log_encode(0.18) - want).abs() < 1e-5);
        assert_eq!(log_encode(0.0), 0.0);
        assert_eq!(log_encode(1e9), 1.0);
    }

    /// E-Gamut holds a D65 grey neutral (so does its derivation from ACEScg).
    #[test]
    fn in_matrix_keeps_grey_grey() {
        for w in [Primaries::Rec709, Primaries::Ap1] {
            let m = in_matrix(w);
            let g = [0, 1, 2].map(|i| m[i][0] + m[i][1] + m[i][2]);
            assert!((g[0] - g[1]).abs() < 2e-3 && (g[1] - g[2]).abs() < 2e-3, "{w:?}: {g:?}");
        }
    }
}
