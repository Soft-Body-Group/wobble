//! Starfield: a generator that fills its format with a sky of stars at
//! infinity, seen through a camera.
//!
//! **The stars live on the sphere, not on the picture.** Every pixel is a
//! ray (the camera's rotation and focal length, the Camera3D convention:
//! Y up, looking down −Z, `R = Rz · Ry · Rx`), and the ray is looked up on
//! a hashed cube-sphere: each face is cut into `cells` x `cells` cells
//! (equal-angle, so cells near a face's corner are not squashed), a cell
//! holds at most one star jittered inside it, and a pixel tests its own
//! cell and the eight round it. Stars are at infinity, so a camera's
//! TRANSLATION does nothing and its rotation and lens are everything: pan
//! a tracked camera and the stars slide exactly as a real sky would.
//!
//! **Layers make the magnitudes.** Layer `l` has `cells · 2^l` cells a
//! side and `falloff^l` the brightness, so each layer has four times the
//! stars at a fraction of the light — the many-faint, few-bright look of a
//! real sky, without a per-star power law to tune.
//!
//! **Distribution** picks which cells hold a star (all uniforms; a switch
//! is a branch on a uniform, never a recompile):
//! - `Random` — each cell holds one with probability `density` (Poisson-ish);
//! - `Even` — every cell holds one (stratified: no clumps, no gaps);
//! - `Clustered` — `density` scaled by a 3D noise on the sphere;
//! - `Band` — `density` scaled by distance from a great circle (a galactic
//!   plane), never below `floor`.
//!
//! Clustered and Band also lay a soft `glow` along the same field: the
//! unresolved stars of a Milky Way.
//!
//! **A camera on input 0 drives the view.** The input is the scene pipe
//! (a Camera3D, including one sourced from a CameraTrack, a file or
//! Blender); [`lower`] — run after animation and sources, next to
//! `scene3d::expand` — copies its rotation and lens into the transient
//! [`StarfieldParams::cam`] and drops the wire, so the engine sees a plain
//! generator. Unwired, the node's own `rotate` and `focal` are the view.
//!
//! Everything is a uniform: the topology is one constant whatever the node
//! holds. Star size is PIXELS (scaled by the proxy at pack time); the field
//! of view comes from the node's own format (proxy included), so a proxy
//! render is the same sky, smaller.

use crate::codegen::Emit;
use crate::graph::{Graph, NodeId, NodeKind};
use crate::nodes::NodeDef;
use std::collections::BTreeMap;

/// Topology tag, clear of every other node's span.
const TOPO: u64 = 0x57A4_0000;

/// Most magnitude layers (each four times the stars of the last).
pub const MAX_LAYERS: u32 = 5;

/// Channel prefix for the node's keyable numbers (`star.<name>` on
/// `Channel::Custom`, the Text node's precedent).
pub const PREFIX: &str = "star.";

/// Which cells hold a star.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum StarDist {
    #[default]
    Random,
    Even,
    Clustered,
    Band,
}

impl StarDist {
    pub const ALL: [StarDist; 4] = [StarDist::Random, StarDist::Even, StarDist::Clustered, StarDist::Band];

    pub fn label(self) -> &'static str {
        match self {
            StarDist::Random => "random",
            StarDist::Even => "even",
            StarDist::Clustered => "clustered",
            StarDist::Band => "band",
        }
    }

    pub fn hint(self) -> &'static str {
        match self {
            StarDist::Random => "each cell holds a star with probability `density`: natural, a little clumpy",
            StarDist::Even => "every cell holds one star: evenly spread, no clumps and no gaps",
            StarDist::Clustered => "density follows a noise on the sky: clusters and voids",
            StarDist::Band => "density gathers along a great circle, like the Milky Way",
        }
    }
}

/// A wired camera's view, resolved for the frame by [`lower`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StarCam {
    /// Euler degrees, the Camera3D convention.
    pub rotate: [f32; 3],
    /// Focal length and filmback width, mm.
    pub focal: f32,
    pub aperture: f32,
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct StarfieldParams {
    /// The node's format (see `inherit`).
    pub size: (u32, u32),
    /// Take the comp's canvas instead of `size`.
    pub inherit: bool,
    /// Which sky: a different seed is a different set of stars.
    pub seed: f32,
    /// Cells a cube face's side on the brightest layer.
    pub cells: f32,
    /// Magnitude layers, 1..=[`MAX_LAYERS`].
    pub layers: f32,
    /// Chance a cell holds a star (Random, Clustered, Band).
    pub density: f32,
    /// Peak brightness of the brightest layer's stars.
    pub brightness: f32,
    /// Brightness kept per layer down (0..1); each layer's stars are also
    /// 0.7 the size.
    pub falloff: f32,
    /// Star radius in pixels (a Gaussian's sigma).
    pub star_size: f32,
    /// How much stars differ in brightness and size (0 = all alike).
    pub variation: f32,
    /// How far star colours spread from white to blue and warm (0..1).
    pub color_var: f32,
    /// Multiplies every star, working space.
    pub tint: [f32; 3],
    pub dist: StarDist,
    /// Clustered: the noise's frequency on the sphere.
    pub cluster_scale: f32,
    /// Clustered: how hard the clusters are (0 = no effect).
    pub cluster_contrast: f32,
    /// Band: the band's normal, as tilt about X and turn about Y (degrees).
    pub band_tilt: f32,
    pub band_turn: f32,
    /// Band: its half-width, degrees.
    pub band_width: f32,
    /// Band: density kept far from it (0..1).
    pub band_floor: f32,
    /// Clustered / Band: the soft glow along the density field.
    pub glow: f32,
    pub glow_color: [f32; 3],
    /// Alpha 1 everywhere (a background plate); off, alpha is the light.
    pub opaque: bool,
    /// The view when no camera is wired: Euler degrees (Camera3D's).
    pub rotate: [f32; 3],
    /// Focal length, mm, on `aperture` mm of filmback width.
    pub focal: f32,
    pub aperture: f32,
    /// Set by [`lower`] when a camera is wired; never authored.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub cam: Option<StarCam>,
}

impl Default for StarfieldParams {
    fn default() -> Self {
        Self {
            size: (1920, 1080),
            inherit: false,
            seed: 0.0,
            cells: 16.0,
            layers: 4.0,
            density: 0.5,
            brightness: 3.0,
            falloff: 0.3,
            star_size: 0.8,
            variation: 0.6,
            color_var: 0.5,
            tint: [1.0; 3],
            dist: StarDist::Random,
            cluster_scale: 3.0,
            cluster_contrast: 0.6,
            band_tilt: 60.0,
            band_turn: 0.0,
            band_width: 12.0,
            band_floor: 0.15,
            glow: 0.05,
            glow_color: [0.8, 0.85, 1.0],
            opaque: true,
            rotate: [0.0; 3],
            focal: 35.0,
            aperture: 36.0,
            cam: None,
        }
    }
}

/// The keyable numbers, by channel name after [`PREFIX`].
const NAMES: &[&str] = &[
    "seed", "cells", "layers", "density", "brightness", "falloff", "size", "variation", "colorvar",
    "tint.r", "tint.g", "tint.b", "cluster.scale", "cluster.contrast", "band.tilt", "band.turn",
    "band.width", "band.floor", "glow", "glow.r", "glow.g", "glow.b", "rotate.x", "rotate.y",
    "rotate.z", "focal",
];

impl StarfieldParams {
    /// Every keyable number's name (after [`PREFIX`]).
    pub fn names() -> &'static [&'static str] {
        NAMES
    }

    fn field(&mut self, name: &str) -> Option<&mut f32> {
        Some(match name {
            "seed" => &mut self.seed,
            "cells" => &mut self.cells,
            "layers" => &mut self.layers,
            "density" => &mut self.density,
            "brightness" => &mut self.brightness,
            "falloff" => &mut self.falloff,
            "size" => &mut self.star_size,
            "variation" => &mut self.variation,
            "colorvar" => &mut self.color_var,
            "tint.r" => &mut self.tint[0],
            "tint.g" => &mut self.tint[1],
            "tint.b" => &mut self.tint[2],
            "cluster.scale" => &mut self.cluster_scale,
            "cluster.contrast" => &mut self.cluster_contrast,
            "band.tilt" => &mut self.band_tilt,
            "band.turn" => &mut self.band_turn,
            "band.width" => &mut self.band_width,
            "band.floor" => &mut self.band_floor,
            "glow" => &mut self.glow,
            "glow.r" => &mut self.glow_color[0],
            "glow.g" => &mut self.glow_color[1],
            "glow.b" => &mut self.glow_color[2],
            "rotate.x" => &mut self.rotate[0],
            "rotate.y" => &mut self.rotate[1],
            "rotate.z" => &mut self.rotate[2],
            "focal" => &mut self.focal,
            _ => return None,
        })
    }

    pub fn get(&self, name: &str) -> Option<f32> {
        self.clone().field(name).map(|v| *v)
    }

    pub fn set(&mut self, name: &str, v: f32) -> bool {
        match self.field(name) {
            Some(f) => {
                *f = v;
                true
            }
            None => false,
        }
    }

    /// The view this frame: the wired camera's, else the node's own.
    pub fn view(&self) -> StarCam {
        self.cam.unwrap_or(StarCam { rotate: self.rotate, focal: self.focal, aperture: self.aperture })
    }

    /// The band's unit normal.
    pub fn band_normal(&self) -> [f32; 3] {
        let r = crate::scene3d::rotation([self.band_tilt, self.band_turn, 0.0]);
        [r[0][1], r[1][1], r[2][1]]
    }

    /// The world direction through pixel `p` (pixel coords, y down) of a
    /// `w` x `h` format — the CPU mirror of the shader's ray.
    pub fn ray(&self, p: [f32; 2], w: f32, h: f32) -> [f32; 3] {
        let v = self.view();
        let f = v.focal.max(1e-3) / v.aperture.max(1e-3) * w;
        let d = [(p[0] - 0.5 * w) / f, -(p[1] - 0.5 * h) / f, -1.0];
        let r = crate::scene3d::rotation(v.rotate);
        let o = [0, 1, 2].map(|i| r[i][0] * d[0] + r[i][1] * d[1] + r[i][2] * d[2]);
        let l = (o[0] * o[0] + o[1] * o[1] + o[2] * o[2]).sqrt();
        o.map(|x| x / l)
    }

    /// The pixel a world direction lands on, if it is in front of the
    /// camera — the inverse of [`ray`](Self::ray).
    pub fn project(&self, d: [f32; 3], w: f32, h: f32) -> Option<[f32; 2]> {
        let v = self.view();
        let f = v.focal.max(1e-3) / v.aperture.max(1e-3) * w;
        let r = crate::scene3d::rotation(v.rotate);
        // camera = R^T world
        let c = [0, 1, 2].map(|j| r[0][j] * d[0] + r[1][j] * d[1] + r[2][j] * d[2]);
        if c[2] >= -1e-6 {
            return None;
        }
        Some([0.5 * w + f * c[0] / -c[2], 0.5 * h - f * c[1] / -c[2]])
    }

    /// [`NodeDef::pack`] with the node's format and the proxy scale.
    pub fn pack_fmt(&self, v: &mut [f32], fmt: [f32; 2], proxy: f32) {
        let view = self.view();
        let r = crate::scene3d::rotation(view.rotate);
        let fpx = view.focal.max(1e-3) / view.aperture.max(1e-3) * fmt[0].max(1.0);
        for i in 0..3 {
            v[i * 4..i * 4 + 3].copy_from_slice(&r[i]);
        }
        v[3] = fpx;
        v[7] = fmt[0];
        v[11] = fmt[1];
        v[12] = self.seed.floor();
        v[13] = self.cells.clamp(1.0, 4096.0).floor();
        v[14] = self.layers.clamp(1.0, MAX_LAYERS as f32).round();
        v[15] = self.density.clamp(0.0, 1.0);
        v[16] = self.brightness.max(0.0);
        v[17] = self.falloff.clamp(0.0, 4.0);
        v[18] = (self.star_size * proxy).max(1e-3);
        v[19] = self.variation.clamp(0.0, 1.0);
        v[20..23].copy_from_slice(&self.tint);
        v[23] = self.color_var.clamp(0.0, 1.0);
        v[24] = self.dist as u32 as f32;
        v[25] = self.cluster_scale.max(0.0);
        v[26] = self.cluster_contrast.clamp(0.0, 1.0);
        v[27] = self.band_floor.clamp(0.0, 1.0);
        v[28..31].copy_from_slice(&self.band_normal());
        v[31] = self.band_width.to_radians().max(1e-3).sin();
        v[32..35].copy_from_slice(&self.glow_color.map(|c| c * self.glow.max(0.0)));
        v[35] = if self.opaque { 1.0 } else { 0.0 };
    }
}

/// Copy each Starfield's wired camera into its transient `cam` and drop
/// the scene wire, so codegen sees a generator (see the module docs). Run
/// after animation and camera sources are applied.
pub fn lower(g: &mut Graph) {
    let ids: Vec<NodeId> =
        g.nodes.iter().filter(|(_, k)| matches!(k, NodeKind::Starfield(_))).map(|(id, _)| *id).collect();
    for id in ids {
        let cam = match g.input_of(id, 0).and_then(|c| g.nodes.get(&c)) {
            Some(NodeKind::Camera3D(c)) => Some(StarCam {
                rotate: c.xform.rotate,
                focal: c.focal,
                aperture: c.aperture[0],
            }),
            _ => None,
        };
        g.edges.remove(&(id, 0));
        g.edge_out.remove(&(id, 0));
        if let Some(NodeKind::Starfield(p)) = g.nodes.get_mut(&id) {
            p.cam = cam;
        }
    }
}

/// The node's WGSL, once per shader.
fn wgsl(fns: &mut BTreeMap<String, String>) -> &'static str {
    const NAME: &str = "starfield_px";
    if fns.contains_key(NAME) {
        return NAME;
    }
    fns.insert(
        NAME.into(),
        format!(
            "// Starfield: the pixel's ray through the camera, looked up on a hashed
// equal-angle cube-sphere, one star at most a cell, layer by layer.
fn sf_hash(v0: vec4u) -> vec4u {{
    var v = v0 * 1664525u + 1013904223u;
    v.x += v.y * v.w; v.y += v.z * v.x; v.z += v.x * v.y; v.w += v.y * v.z;
    v = v ^ (v >> vec4u(16u));
    v.x += v.y * v.w; v.y += v.z * v.x; v.z += v.x * v.y; v.w += v.y * v.z;
    return v;
}}
fn sf_rand(v: vec4u) -> vec4f {{
    return vec4f(sf_hash(v) >> vec4u(8u)) * (1.0 / 16777216.0);
}}
// Gradient noise on the sphere, about 0..1.
fn sf_noise1(p: vec3f, seed: u32) -> f32 {{
    let i = floor(p);
    let f = p - i;
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let b = vec3i(i);
    var acc = 0.0;
    for (var k = 0u; k < 8u; k = k + 1u) {{
        let o = vec3i(i32(k & 1u), i32((k >> 1u) & 1u), i32((k >> 2u) & 1u));
        let c = bitcast<vec3u>(b + o);
        let g = normalize(sf_rand(vec4u(c, seed ^ 0x5bd1e995u)).xyz * 2.0 - 1.0 + 1e-4);
        let w = mix(1.0 - u, u, vec3f(o));
        acc += dot(g, f - vec3f(o)) * w.x * w.y * w.z;
    }}
    return clamp(acc * 0.9 + 0.5, 0.0, 1.0);
}}
fn sf_noise(p: vec3f, seed: u32) -> f32 {{
    return 0.6 * sf_noise1(p, seed) + 0.28 * sf_noise1(p * 2.3 + 11.3, seed + 7u)
        + 0.12 * sf_noise1(p * 5.1 + 3.9, seed + 13u);
}}
// The density field: the chance a cell holds a star, and the glow weight.
fn sf_field(d: vec3f, slot: u32, seed: u32) -> vec2f {{
    let m = P.v[slot + 6u];
    let mode = u32(m.x);
    if (mode == 2u) {{
        let n = sf_noise(d * max(m.y, 0.01) + 3.7, seed);
        let w = clamp(pow(n * 1.3, 1.0 + 5.0 * m.z), 0.0, 1.0);
        return vec2f(mix(1.0, w * 3.0, m.z), w);
    }}
    if (mode == 3u) {{
        let bn = P.v[slot + 7u];
        let s = dot(d, bn.xyz) / bn.w;
        let g = exp(-s * s);
        let n = sf_noise(d * 5.0 + 1.3, seed);
        let fine = sf_noise1(d * 23.0 + 4.1, seed + 3u);
        let glow = g * clamp(n * 1.6 - 0.25, 0.0, 1.5) * mix(0.55, 1.2, fine);
        return vec2f(mix(m.w, 1.0, g) * mix(0.5, 1.5, n), glow);
    }}
    return vec2f(1.0, 0.0);
}}
// A face's equal-angle coordinate (-1..1) back to a unit direction.
fn sf_dir(face: u32, uv: vec2f) -> vec3f {{
    let t = tan(uv * 0.78539816);
    var v: vec3f;
    switch face {{
        case 0u: {{ v = vec3f(1.0, t.y, -t.x); }}
        case 1u: {{ v = vec3f(-1.0, t.y, t.x); }}
        case 2u: {{ v = vec3f(t.x, 1.0, -t.y); }}
        case 3u: {{ v = vec3f(t.x, -1.0, t.y); }}
        case 4u: {{ v = vec3f(t.x, t.y, 1.0); }}
        default: {{ v = vec3f(-t.x, t.y, -1.0); }}
    }}
    return normalize(v);
}}
fn {NAME}(p: vec2f, slot: u32) -> vec4f {{
    let r0 = P.v[slot];
    let r1 = P.v[slot + 1u];
    let r2 = P.v[slot + 2u];
    let fmt = vec2f(r1.w, r2.w);
    let fpx = r0.w;
    let dc = vec3f((p.x - 0.5 * fmt.x) / fpx, -(p.y - 0.5 * fmt.y) / fpx, -1.0);
    let d = normalize(vec3f(dot(r0.xyz, dc), dot(r1.xyz, dc), dot(r2.xyz, dc)));
    let a = P.v[slot + 3u];
    let b = P.v[slot + 4u];
    let tint = P.v[slot + 5u];
    let seed = bitcast<u32>(i32(a.x));
    let mode = u32(P.v[slot + 6u].x);
    let field = sf_field(d, slot, seed);
    // The face and its equal-angle coordinate.
    let ad = abs(d);
    var face: u32;
    var uv: vec2f;
    if (ad.x >= ad.y && ad.x >= ad.z) {{
        face = select(1u, 0u, d.x > 0.0);
        uv = vec2f(select(d.z, -d.z, d.x > 0.0), d.y) / ad.x;
    }} else if (ad.y >= ad.z) {{
        face = select(3u, 2u, d.y > 0.0);
        uv = vec2f(d.x, select(d.z, -d.z, d.y > 0.0)) / ad.y;
    }} else {{
        face = select(5u, 4u, d.z > 0.0);
        uv = vec2f(select(-d.x, d.x, d.z > 0.0), d.y) / ad.z;
    }}
    let w = atan(uv) * 1.2732395;
    // Pixel size on the sphere, for the Gaussian; under half a pixel the
    // star keeps its light and spreads to stay sampled.
    let px = 1.0 / fpx;
    var col = vec3f(0.0);
    let layers = u32(a.z);
    var cells = a.y;
    var bright = b.x;
    var size = b.z;
    for (var l = 0u; l < layers; l = l + 1u) {{
        let c = floor((w * 0.5 + 0.5) * cells);
        for (var j = -1; j <= 1; j = j + 1) {{
            for (var i = -1; i <= 1; i = i + 1) {{
                let cc = c + vec2f(f32(i), f32(j));
                if (any(cc < vec2f(0.0)) || any(cc >= vec2f(cells))) {{ continue; }}
                let ci = vec2u(cc);
                let h = sf_rand(vec4u(ci, face + 8u * l, seed));
                let h2 = sf_rand(vec4u(ci.yx ^ vec2u(0x9e3779b9u), face + 8u * l + 101u, seed));
                // Even: every cell; else the density, shaped by the field.
                if (mode != 1u && h.z >= a.w * field.x) {{ continue; }}
                let suv = (cc + mix(vec2f(0.15), vec2f(0.85), h.xy)) / cells * 2.0 - 1.0;
                let s = sf_dir(face, suv);
                let r = length(d - s) / px;
                let sig_req = size * mix(1.0, 0.5 + h2.x, b.w);
                let sig = max(sig_req, 0.5);
                if (r > 4.0 * sig) {{ continue; }}
                let mag = bright * mix(1.0, h.w * h.w * h.w * 3.0, b.w) * min(1.0, (sig_req * sig_req) / (sig * sig));
                // Colour: blue-white to white to warm, by the star's hash.
                let t = h2.y;
                let hot = vec3f(0.62, 0.74, 1.0);
                let warm = vec3f(1.0, 0.72, 0.45);
                let tone = select(mix(vec3f(1.0), warm, (t - 0.5) * 2.0), mix(hot, vec3f(1.0), t * 2.0), t < 0.5);
                col += mix(vec3f(1.0), tone, tint.w) * mag * exp(-0.5 * r * r / (sig * sig));
            }}
        }}
        cells = cells * 2.0;
        bright = bright * b.y;
        size = size * 0.7;
    }}
    col = col * tint.rgb + P.v[slot + 8u].rgb * field.y;
    let alpha = select(clamp(max(col.r, max(col.g, col.b)), 0.0, 1.0), 1.0, P.v[slot + 8u].w > 0.5);
    return vec4f(col, alpha);
}}
"
        ),
    );
    NAME
}

impl NodeDef for StarfieldParams {
    const NAME: &'static str = "Starfield";
    /// The camera, on the scene pipe (lowered away before codegen).
    const INPUTS: usize = 1;
    const MASK_INPUT: Option<usize> = None;
    /// Rotation rows + focal/format, counts, star look, tint, distribution,
    /// band normal, glow + opaque.
    const VEC4_SLOTS: usize = 9;

    /// One topology whatever the node holds: the distribution is a uniform.
    fn topo_tag(&self) -> u64 {
        TOPO
    }

    fn pack(&self, v: &mut [f32]) {
        self.pack_fmt(v, [self.size.0 as f32, self.size.1 as f32], 1.0);
    }

    fn emit(&self, e: &mut Emit) -> String {
        let slot = e.slot();
        let pos = e.pos().to_string();
        let (_, fns) = e.color_helpers();
        let f = wgsl(fns);
        let v = e.fresh();
        e.line(format_args!("let {v} = {f}({pos}, {slot}u);"));
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_edit_is_a_uniform() {
        let a = StarfieldParams::default();
        let mut b = a.clone();
        b.dist = StarDist::Band;
        b.layers = 2.0;
        b.rotate = [10.0, 20.0, 30.0];
        b.cam = Some(StarCam { rotate: [1.0; 3], focal: 80.0, aperture: 24.0 });
        assert_eq!(a.topo_tag(), b.topo_tag());
    }

    #[test]
    fn every_name_reads_and_writes() {
        let mut p = StarfieldParams::default();
        for (i, n) in StarfieldParams::names().iter().enumerate() {
            assert!(p.set(n, i as f32 + 0.5), "{n}");
            assert_eq!(p.get(n), Some(i as f32 + 0.5), "{n}");
        }
        assert!(!p.set("nope", 1.0));
    }

    #[test]
    fn ray_and_project_are_inverses() {
        let p = StarfieldParams { rotate: [12.0, -40.0, 7.0], focal: 50.0, ..Default::default() };
        for px in [[0.0, 0.0], [960.0, 540.0], [1900.0, 100.0]] {
            let d = p.ray(px, 1920.0, 1080.0);
            let q = p.project(d, 1920.0, 1080.0).unwrap();
            assert!((q[0] - px[0]).abs() < 1e-2 && (q[1] - px[1]).abs() < 1e-2, "{px:?} -> {q:?}");
        }
    }

    #[test]
    fn a_wired_camera_overrides_the_manual_view() {
        let p = StarfieldParams {
            cam: Some(StarCam { rotate: [0.0, 90.0, 0.0], focal: 50.0, aperture: 36.0 }),
            ..Default::default()
        };
        // Turned 90 degrees about Y, the camera looks down -X.
        let d = p.ray([960.0, 540.0], 1920.0, 1080.0);
        assert!((d[0] + 1.0).abs() < 1e-5, "{d:?}");
    }

    #[test]
    fn lowering_takes_the_camera_and_drops_the_wire() {
        let mut g = Graph::default();
        let cam = crate::scene3d::CameraParams {
            xform: crate::scene3d::Xform { rotate: [5.0, 6.0, 7.0], ..Default::default() },
            focal: 85.0,
            ..Default::default()
        };
        g.nodes.insert(1, NodeKind::Camera3D(cam));
        g.nodes.insert(2, NodeKind::Starfield(StarfieldParams::default()));
        g.connect_out(2, 0, 1, 0);
        lower(&mut g);
        assert!(g.input_of(2, 0).is_none());
        let Some(NodeKind::Starfield(p)) = g.nodes.get(&2) else { panic!() };
        let c = p.cam.unwrap();
        assert_eq!(c.rotate, [5.0, 6.0, 7.0]);
        assert_eq!(c.focal, 85.0);
        assert_eq!(c.aperture, 36.0);
    }
}
