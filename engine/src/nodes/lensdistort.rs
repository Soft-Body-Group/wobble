//! Lens Distortion: the Brown-Conrady radial + tangential model (k1 k2 k3,
//! p1 p2 — what OpenCV, 3DEqualizer's "classic" and most solvers export),
//! applied either way round.
//!
//! - **Undistort** takes a plate shot through the lens and gives back the
//!   rectilinear picture: straight lines in the scene come out straight.
//!   Each output pixel reads the plate at `D(u)`, the forward model, in
//!   closed form.
//! - **Distort** takes a rectilinear picture (CG, a patch, a matte painting)
//!   and bends it the way the lens bent the plate, so it sits back on it.
//!   Each output pixel needs `D⁻¹`, which has no closed form: the shader
//!   runs Newton's method on the model's exact Jacobian, seeded at the pixel
//!   itself. A pixel the inverse cannot reach (a model that folds over
//!   inside the frame) renders TRANSPARENT, never a plausible wrong sample —
//!   the GridWarp rule. So does an undistort past the fold radius, where the
//!   forward model starts running backwards.
//!
//! The same params both ways round are exact inverses, so undistort →
//! (work) → distort puts the plate back pixel for pixel
//! (`engine/tests/lensdistort.rs`).
//!
//! Coordinates. The model runs in normalised coordinates: pixels from the
//! principal point (`center`, normalised over the PLATE's format, y down),
//! x multiplied by the anamorphic `squeeze` first so the model sees the
//! desqueezed frame, all divided by half the desqueezed plate's DIAGONAL —
//! so r = 1 at the corners and a k means the same thing at 2K and 4K, 16:9
//! and 2.39. `zoom` scales the undistorted picture (> 1 crops in).
//!
//! **Overscan.** An undistorted barrel plate is larger than the plate: its
//! corners stretch past the frame. `overscan` grows the node's FORMAT by
//! that factor in undistort mode (the plate stays 1:1 in the middle, the
//! margin holds the stretched edges) and shrinks it by the same factor in
//! distort mode, so one value on both nodes makes the round trip land on
//! the plate's own format. The plate — whose size the model is normalised
//! by — is the input in undistort mode and the output in distort mode.
//!
//! **Chromatic aberration** (`chromatic`, structural: it samples the input
//! three times): red and blue are magnified by `1 + ca` relative to green.
//! Undistort removes it, distort adds it, and the round trip still closes.
//!
//! **STMap output** writes the lookup itself instead of the picture: red =
//! U, green = V over the SOURCE's format, V up, pixel centres at
//! `(x + 0.5) / w` — exactly what UV Warp reads, so a Lens Distortion's map
//! into a UV Warp reproduces the node, and the map can leave the building.
//! It is the green channel's map: CA is not in it. Invalid pixels are 0.
//!
//! It fuses like UV Warp and GridWarp: the input is re-evaluated at the
//! looked-up position (`Emit::input_at`), no pass, no texture, no stage.
//! Every number is a uniform; mode, output and `chromatic` are the only
//! structure, `overscan` only moves the format.
//!
//! Not built (yet): solving the coefficients from a grid shot or from lines
//! that should be straight. The CPU mirror here (`distort`, `undistort`,
//! `jacobian`) is what such a solver would minimise through.

use crate::codegen::Emit;
use crate::nodes::NodeDef;

const TOPO_BASE: u64 = 0x1e25_0000;

/// Newton steps for the inverse. Real lenses converge in 3-5 from the
/// pixel itself; the spare ones cover strong fisheye-ish k1.
pub const ITERS: u32 = 12;

/// Residual (normalised units, ~1e-4 of half a diagonal ≈ 0.1 px at 2K)
/// below which an inverse counts as found.
const TOL: f32 = 1e-4;

/// Which way round the model is applied.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum LensMode {
    /// Plate in, rectilinear picture out.
    #[default]
    Undistort,
    /// Rectilinear picture in, the lens's bend out.
    Distort,
}

impl LensMode {
    pub const ALL: [LensMode; 2] = [LensMode::Undistort, LensMode::Distort];
    pub fn name(self) -> &'static str {
        match self {
            LensMode::Undistort => "undistort",
            LensMode::Distort => "distort",
        }
    }
}

/// What the node outputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum LensOutput {
    /// The input, warped.
    #[default]
    Image,
    /// The warp as a UV map (UV Warp's convention).
    StMap,
}

impl LensOutput {
    pub const ALL: [LensOutput; 2] = [LensOutput::Image, LensOutput::StMap];
    pub fn name(self) -> &'static str {
        match self {
            LensOutput::Image => "image",
            LensOutput::StMap => "stmap",
        }
    }
}

/// A [`crate::graph::NodeKind::LensDistort`]'s controls.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct LensDistortParams {
    pub mode: LensMode,
    pub output: LensOutput,
    /// Radial terms: r², r⁴, r⁶. Negative k1 is barrel, positive pincushion.
    pub k1: f32,
    pub k2: f32,
    pub k3: f32,
    /// Tangential (decentering) terms.
    pub p1: f32,
    pub p2: f32,
    /// The principal point, normalised over the plate's format (y down).
    pub center: [f32; 2],
    /// Anamorphic squeeze: the plate's pixel aspect. 2 for a 2x anamorphic.
    pub squeeze: f32,
    /// Scales the undistorted picture; > 1 crops in.
    pub zoom: f32,
    /// Format growth in undistort mode (shrink in distort). 1 = none.
    /// Not a channel: an animated format is not a format.
    pub overscan: f32,
    /// Sample red and blue at their own magnification.
    pub chromatic: bool,
    /// Red and blue magnification relative to green, minus one.
    pub ca: [f32; 2],
}

impl Default for LensDistortParams {
    fn default() -> Self {
        Self {
            mode: LensMode::Undistort,
            output: LensOutput::Image,
            k1: 0.0,
            k2: 0.0,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
            center: [0.5, 0.5],
            squeeze: 1.0,
            zoom: 1.0,
            overscan: 1.0,
            chromatic: false,
            ca: [0.0, 0.0],
        }
    }
}

/// The three formats the mapping relates, in pixels: the node's INPUT, its
/// OUTPUT, and the PLATE the model is normalised by (the input when
/// undistorting, the output when distorting).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LensGeom {
    pub src: [f32; 2],
    pub out: [f32; 2],
    pub plate: [f32; 2],
}

/// `D(u)`: undistorted → distorted, normalised. The CPU mirror of `lens_d`.
pub fn distort(k: [f32; 3], p: [f32; 2], u: [f32; 2]) -> [f32; 2] {
    let (x, y) = (u[0], u[1]);
    let r2 = x * x + y * y;
    let l = 1.0 + r2 * (k[0] + r2 * (k[1] + r2 * k[2]));
    [
        x * l + 2.0 * p[0] * x * y + p[1] * (r2 + 2.0 * x * x),
        y * l + p[0] * (r2 + 2.0 * y * y) + 2.0 * p[1] * x * y,
    ]
}

/// The Jacobian of `D` at `u`: `[[dDx/dx, dDx/dy], [dDy/dx, dDy/dy]]`.
pub fn jacobian(k: [f32; 3], p: [f32; 2], u: [f32; 2]) -> [[f32; 2]; 2] {
    let (x, y) = (u[0], u[1]);
    let r2 = x * x + y * y;
    let l = 1.0 + r2 * (k[0] + r2 * (k[1] + r2 * k[2]));
    let lp = k[0] + r2 * (2.0 * k[1] + 3.0 * k[2] * r2);
    let a = l + 2.0 * x * x * lp + 2.0 * p[0] * y + 6.0 * p[1] * x;
    let b = 2.0 * x * y * lp + 2.0 * p[0] * x + 2.0 * p[1] * y;
    let d = l + 2.0 * y * y * lp + 6.0 * p[0] * y + 2.0 * p[1] * x;
    [[a, b], [b, d]]
}

fn det(j: [[f32; 2]; 2]) -> f32 {
    j[0][0] * j[1][1] - j[0][1] * j[1][0]
}

/// `D⁻¹(d)` by Newton from `d` — the CPU mirror of `lens_inv`. `None` where
/// it does not converge or lands past the fold (Jacobian not positive).
pub fn undistort(k: [f32; 3], p: [f32; 2], d: [f32; 2]) -> Option<[f32; 2]> {
    let mut u = d;
    for _ in 0..ITERS {
        let f = distort(k, p, u);
        let f = [f[0] - d[0], f[1] - d[1]];
        let j = jacobian(k, p, u);
        let dt = det(j);
        if dt.abs() < 1e-8 {
            break;
        }
        u[0] -= (j[1][1] * f[0] - j[0][1] * f[1]) / dt;
        u[1] -= (-j[1][0] * f[0] + j[0][0] * f[1]) / dt;
    }
    let f = distort(k, p, u);
    let res = (f[0] - d[0]).hypot(f[1] - d[1]);
    (res < TOL && det(jacobian(k, p, u)) > 0.0).then_some(u)
}

impl LensDistortParams {
    fn ks(&self) -> [f32; 3] {
        [self.k1, self.k2, self.k3]
    }
    fn ps(&self) -> [f32; 2] {
        [self.p1, self.p2]
    }
    fn sq(&self) -> f32 {
        if self.squeeze.is_finite() { self.squeeze.max(0.01) } else { 1.0 }
    }
    fn z(&self) -> f32 {
        if self.zoom.is_finite() { self.zoom.max(0.01) } else { 1.0 }
    }

    /// The overscan factor actually applied (clamped to a sane range).
    pub fn overscan_factor(&self) -> f32 {
        if self.overscan.is_finite() { self.overscan.clamp(1.0, 4.0) } else { 1.0 }
    }

    /// The node's format given its input's, in whole pixels: grown by the
    /// overscan when undistorting, shrunk by it when distorting.
    pub fn output_format(&self, input: (u32, u32)) -> (u32, u32) {
        let o = self.overscan_factor();
        let f = match self.mode {
            LensMode::Undistort => o,
            LensMode::Distort => 1.0 / o,
        };
        let px = |v: u32| ((v as f32 * f).round() as u32).max(1);
        (px(input.0), px(input.1))
    }

    /// The geometry from the input and output formats.
    pub fn geom(&self, src: [f32; 2], out: [f32; 2]) -> LensGeom {
        let plate = match self.mode {
            LensMode::Undistort => src,
            LensMode::Distort => out,
        };
        LensGeom { src, out, plate }
    }

    /// Half the desqueezed plate's diagonal: the model's unit.
    pub fn radius(&self, g: &LensGeom) -> f32 {
        (0.5 * (g.plate[0] * self.sq()).hypot(g.plate[1])).max(1e-3)
    }

    /// The principal point in output pixels and in source pixels. Any
    /// overscan margin is symmetric, so both are the plate's point offset
    /// by half the difference in size.
    pub fn centres(&self, g: &LensGeom) -> ([f32; 2], [f32; 2]) {
        let at = |size: [f32; 2]| {
            [
                size[0] * 0.5 + (self.center[0] - 0.5) * g.plate[0],
                size[1] * 0.5 + (self.center[1] - 0.5) * g.plate[1],
            ]
        };
        (at(g.out), at(g.src))
    }

    /// Where output pixel `q` reads the source, in source pixels, for a
    /// channel magnified by `s` (1 for green) — the CPU mirror of
    /// `lens_src`. `None` where the shader renders transparent.
    pub fn src_of(&self, g: &LensGeom, q: [f32; 2], s: f32) -> Option<[f32; 2]> {
        let (co, cs) = self.centres(g);
        let (sq, r) = (self.sq(), self.radius(g));
        let n = [(q[0] - co[0]) * sq / r, (q[1] - co[1]) / r];
        let src = match self.mode {
            LensMode::Distort => {
                let u = undistort(self.ks(), self.ps(), [n[0] / s, n[1] / s])?;
                [u[0] * self.z(), u[1] * self.z()]
            }
            LensMode::Undistort => {
                let u = [n[0] / self.z(), n[1] / self.z()];
                if det(jacobian(self.ks(), self.ps(), u)) <= 0.0 {
                    return None;
                }
                let d = distort(self.ks(), self.ps(), u);
                [d[0] * s, d[1] * s]
            }
        };
        Some([cs[0] + src[0] * r / sq, cs[1] + src[1] * r])
    }

    /// The inverse: where source pixel `q` lands in the output (green).
    /// What the viewer overlay draws through.
    pub fn out_of(&self, g: &LensGeom, q: [f32; 2]) -> Option<[f32; 2]> {
        let (co, cs) = self.centres(g);
        let (sq, r) = (self.sq(), self.radius(g));
        let m = [(q[0] - cs[0]) * sq / r, (q[1] - cs[1]) / r];
        let n = match self.mode {
            LensMode::Undistort => {
                let u = undistort(self.ks(), self.ps(), m)?;
                [u[0] * self.z(), u[1] * self.z()]
            }
            LensMode::Distort => {
                let u = [m[0] / self.z(), m[1] / self.z()];
                if det(jacobian(self.ks(), self.ps(), u)) <= 0.0 {
                    return None;
                }
                distort(self.ks(), self.ps(), u)
            }
        };
        Some([co[0] + n[0] * r / sq, co[1] + n[1] * r])
    }

    /// The span with the formats in it: the codegen's central packer knows
    /// the per-node formats, `pack` does not.
    pub fn pack_geom(&self, g: &LensGeom, v: &mut [f32]) {
        self.pack(v);
        let (co, cs) = self.centres(g);
        v[0..4].copy_from_slice(&[co[0], co[1], cs[0], cs[1]]);
        v[11] = self.radius(g);
        v[14] = g.src[0].max(1.0);
        v[15] = g.src[1].max(1.0);
    }
}

impl NodeDef for LensDistortParams {
    const NAME: &'static str = "Lens Distortion";
    const INPUTS: usize = 1;
    const MASK_INPUT: Option<usize> = None;
    /// (centre out xy, centre src xy), (k1 k2 k3 zoom), (p1 p2 squeeze
    /// radius), (ca red, ca blue, src w, src h).
    const VEC4_SLOTS: usize = 4;

    fn topo_tag(&self) -> u64 {
        TOPO_BASE
            | match self.mode {
                LensMode::Undistort => 0,
                LensMode::Distort => 1,
            }
            | match self.output {
                LensOutput::Image => 0,
                LensOutput::StMap => 2,
            }
            | if self.chromatic { 4 } else { 0 }
    }

    /// The numbers; [`Self::pack_geom`] writes the format-derived lanes.
    fn pack(&self, v: &mut [f32]) {
        let f = |x: f32| if x.is_finite() { x } else { 0.0 };
        v[0..4].copy_from_slice(&[0.0, 0.0, 0.0, 0.0]);
        v[4..8].copy_from_slice(&[f(self.k1), f(self.k2), f(self.k3), self.z()]);
        v[8..12].copy_from_slice(&[f(self.p1), f(self.p2), self.sq(), 1.0]);
        let ca = |x: f32| f(x).clamp(-0.5, 0.5);
        v[12..16].copy_from_slice(&[ca(self.ca[0]), ca(self.ca[1]), 1.0, 1.0]);
    }

    fn emit(&self, e: &mut Emit) -> String {
        let name = wgsl(e);
        let slot = e.slot();
        let pos = e.pos().to_string();
        let dist = self.mode == LensMode::Distort;
        let r = e.fresh();
        e.line(format_args!("let {r} = {name}({pos}, {slot}u, 1.0, {dist});"));
        if self.output == LensOutput::StMap {
            let m = e.fresh();
            e.line(format_args!(
                "let {m} = select(vec4f(0.0), vec4f({r}.x / P.v[{slot}u + 3u].z, \
                 1.0 - {r}.y / P.v[{slot}u + 3u].w, 0.0, 1.0), {r}.z > 0.5);"
            ));
            return m;
        }
        if !e.has_input(0) {
            return "vec4f(0.0)".into();
        }
        let q = e.fresh();
        e.line(format_args!("let {q} = {r}.xy;"));
        let v = e.input_at(0, &q);
        let g = e.fresh();
        e.line(format_args!("let {g} = select(vec4f(0.0), {v}, {r}.z > 0.5);"));
        if !self.chromatic {
            return g;
        }
        // Red and blue at their own magnification: two more lookups.
        let mut lanes = Vec::new();
        for lane in ["x", "y"] {
            let rc = e.fresh();
            e.line(format_args!(
                "let {rc} = {name}({pos}, {slot}u, 1.0 + P.v[{slot}u + 3u].{lane}, {dist});"
            ));
            let qc = e.fresh();
            e.line(format_args!("let {qc} = {rc}.xy;"));
            let vc = e.input_at(0, &qc);
            let c = e.fresh();
            e.line(format_args!("let {c} = select(vec4f(0.0), {vc}, {rc}.z > 0.5);"));
            lanes.push(c);
        }
        let out = e.fresh();
        e.line(format_args!("let {out} = vec4f({}.r, {g}.g, {}.b, {g}.a);", lanes[0], lanes[1]));
        out
    }
}

/// Emit the model's functions once per shader; returns the entry's name.
fn wgsl(e: &mut Emit) -> String {
    let name = "lens_src".to_string();
    let (_, fns) = e.color_helpers();
    if fns.contains_key(&name) {
        return name;
    }
    fns.insert(
        name.clone(),
        format!(
            "// Lens Distortion (Brown-Conrady). k = (k1, k2, k3, zoom),
// p = (p1, p2, squeeze, radius), normalised coordinates.
fn lens_d(u: vec2f, k: vec4f, p: vec4f) -> vec2f {{
    let r2 = dot(u, u);
    let l = 1.0 + r2 * (k.x + r2 * (k.y + r2 * k.z));
    return vec2f(
        u.x * l + 2.0 * p.x * u.x * u.y + p.y * (r2 + 2.0 * u.x * u.x),
        u.y * l + p.x * (r2 + 2.0 * u.y * u.y) + 2.0 * p.y * u.x * u.y);
}}
// The Jacobian as (dDx/dx, dDx/dy = dDy/dx, dDy/dy).
fn lens_j(u: vec2f, k: vec4f, p: vec4f) -> vec3f {{
    let r2 = dot(u, u);
    let l = 1.0 + r2 * (k.x + r2 * (k.y + r2 * k.z));
    let lp = k.x + r2 * (2.0 * k.y + 3.0 * k.z * r2);
    return vec3f(
        l + 2.0 * u.x * u.x * lp + 2.0 * p.x * u.y + 6.0 * p.y * u.x,
        2.0 * u.x * u.y * lp + 2.0 * p.x * u.x + 2.0 * p.y * u.y,
        l + 2.0 * u.y * u.y * lp + 6.0 * p.x * u.y + 2.0 * p.y * u.x);
}}
// D^-1 by Newton from d; z = 1 where found and before the fold.
fn lens_inv(d: vec2f, k: vec4f, p: vec4f) -> vec3f {{
    var u = d;
    for (var i = 0u; i < {ITERS}u; i = i + 1u) {{
        let f = lens_d(u, k, p) - d;
        let j = lens_j(u, k, p);
        let dt = j.x * j.z - j.y * j.y;
        if (abs(dt) < 1e-8) {{ break; }}
        u = u - vec2f(j.z * f.x - j.y * f.y, -j.y * f.x + j.x * f.y) / dt;
    }}
    let j = lens_j(u, k, p);
    let ok = length(lens_d(u, k, p) - d) < {TOL:e} && j.x * j.z - j.y * j.y > 0.0;
    return vec3f(u, select(0.0, 1.0, ok));
}}
// Output pixel -> source pixel for a channel magnified by s; z = valid.
fn lens_src(pos: vec2f, slot: u32, s: f32, dist: bool) -> vec3f {{
    let c = P.v[slot];
    let k = P.v[slot + 1u];
    let p = P.v[slot + 2u];
    let sq = vec2f(p.z, 1.0);
    let n = (pos - c.xy) * sq / p.w;
    var src = vec2f(0.0);
    var ok = 1.0;
    if (dist) {{
        let r = lens_inv(n / s, k, p);
        src = r.xy * k.w;
        ok = r.z;
    }} else {{
        let u = n / k.w;
        let j = lens_j(u, k, p);
        ok = select(0.0, 1.0, j.x * j.z - j.y * j.y > 0.0);
        src = lens_d(u, k, p) * s;
    }}
    return vec3f(c.zw + src * p.w / sq, ok);
}}
"
        ),
    );
    name
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lens() -> LensDistortParams {
        LensDistortParams {
            k1: -0.12,
            k2: 0.03,
            k3: -0.004,
            p1: 0.002,
            p2: -0.0015,
            center: [0.52, 0.47],
            squeeze: 1.3,
            ..Default::default()
        }
    }

    #[test]
    fn the_inverse_inverts() {
        let l = lens();
        for &d in &[[0.0, 0.0], [0.3, -0.2], [0.7, 0.55], [-0.85, 0.4]] {
            let u = undistort(l.ks(), l.ps(), d).expect("converges");
            let back = distort(l.ks(), l.ps(), u);
            assert!((back[0] - d[0]).abs() < 1e-5 && (back[1] - d[1]).abs() < 1e-5, "{d:?} {back:?}");
        }
    }

    #[test]
    fn the_jacobian_matches_finite_differences() {
        let l = lens();
        let u = [0.4, -0.3];
        let j = jacobian(l.ks(), l.ps(), u);
        let h = 1e-3;
        for c in 0..2 {
            let mut a = u;
            let mut b = u;
            a[c] += h;
            b[c] -= h;
            let (fa, fb) = (distort(l.ks(), l.ps(), a), distort(l.ks(), l.ps(), b));
            for r in 0..2 {
                let fd = (fa[r] - fb[r]) / (2.0 * h);
                assert!((fd - j[r][c]).abs() < 1e-3, "J[{r}][{c}] {} vs {fd}", j[r][c]);
            }
        }
    }

    /// Undistort then distort with the same params and overscan lands every
    /// pixel back on itself — including through the overscanned format.
    #[test]
    fn undistort_then_distort_is_the_identity() {
        let plate = [400.0, 240.0];
        let mut un = lens();
        un.overscan = 1.25;
        let mid = un.output_format((400, 240));
        let mid = [mid.0 as f32, mid.1 as f32];
        let gu = un.geom(plate, mid);
        let di = LensDistortParams { mode: LensMode::Distort, ..un };
        let out = di.output_format((mid[0] as u32, mid[1] as u32));
        assert_eq!(out, (400, 240), "the overscan comes back off");
        let gd = di.geom(mid, plate);
        for &q in &[[200.5, 120.5], [10.5, 8.5], [390.5, 230.5], [123.0, 200.0]] {
            for s in [1.0, 1.004] {
                let m = di.src_of(&gd, q, s).expect("distort finds it");
                let p = un.src_of(&gu, m, s).expect("undistort reads it");
                assert!((p[0] - q[0]).abs() < 0.02 && (p[1] - q[1]).abs() < 0.02, "{q:?} -> {p:?}");
            }
        }
    }

    #[test]
    fn out_of_inverts_src_of() {
        for mode in LensMode::ALL {
            let l = LensDistortParams { mode, zoom: 1.1, ..lens() };
            let g = l.geom([320.0, 200.0], [320.0, 200.0]);
            for &q in &[[160.0, 100.0], [30.0, 40.0], [300.0, 180.0]] {
                let s = l.src_of(&g, q, 1.0).unwrap();
                let o = l.out_of(&g, s).unwrap();
                assert!((o[0] - q[0]).abs() < 0.02 && (o[1] - q[1]).abs() < 0.02, "{mode:?} {q:?} {o:?}");
            }
        }
    }

    #[test]
    fn default_is_the_identity_and_only_structure_moves_the_tag() {
        let l = LensDistortParams::default();
        let g = l.geom([100.0, 50.0], [100.0, 50.0]);
        let s = l.src_of(&g, [12.5, 40.5], 1.0).unwrap();
        assert!((s[0] - 12.5).abs() < 1e-4 && (s[1] - 40.5).abs() < 1e-4);
        let moved = LensDistortParams { k1: 0.3, center: [0.1, 0.2], zoom: 2.0, ca: [0.01, 0.0], ..l };
        assert_eq!(l.topo_tag(), moved.topo_tag());
        assert_ne!(l.topo_tag(), LensDistortParams { mode: LensMode::Distort, ..l }.topo_tag());
        assert_ne!(l.topo_tag(), LensDistortParams { chromatic: true, ..l }.topo_tag());
        assert_ne!(l.topo_tag(), LensDistortParams { output: LensOutput::StMap, ..l }.topo_tag());
    }
}
