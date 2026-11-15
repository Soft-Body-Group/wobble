//! Refract: a picture seen through a sheet of bumpy glass.
//!
//! Input 1 says what the glass's surface does — a HEIGHT map (one channel,
//! red; the normal is its slope) or a tangent-space NORMAL map — and input
//! 0 is what is behind it, one of two things ([`RefractMode`]):
//!
//! - **Background**: a plate straight behind the sheet. A ray down the
//!   lens axis bends at the surface (Snell, `ior`), crosses `thickness`
//!   pixels of glass and reads the plate where it lands. Flat glass reads
//!   the plate where it stands.
//! - **Latlong**: an environment at infinity (the Env node's, Blender's
//!   equirectangular — [`crate::nodes::env::latlong_px`]). Every pixel is
//!   a camera ray ([`crate::skyview`]), bent going IN through the bumpy
//!   face and again coming OUT through a flat back face, and the latlong
//!   is read down the ray that leaves. Flat glass is therefore exactly an
//!   Env. Thickness means nothing at infinity (a parallel shift of a ray
//!   reads the same sky), so it is not used here.
//!
//! `strength` scales the bend either way (0 = no glass). `dispersion`
//! spreads the index across the taps, red bending least and blue most;
//! `roughness` scatters the normal over a disc. Both are the same loop —
//! `samples` taps, each with its own index and its own nudge — so the two
//! together cost what either costs alone.
//!
//! **It fuses.** The background's subtree is re-evaluated at each tap
//! (`Emit::input_at` inside ONE WGSL loop, so the code is one copy
//! whatever the count): no pass, no texture, cyan. The price is the
//! background's own cost times the taps, which the panel says. Every
//! number is a uniform; only the mode and the map's kind are structure.
//!
//! **A camera on input 2 drives the latlong view**, lowered away like an
//! Env's ([`lower`]). The picture's format is the MAP's in latlong mode
//! (the glass is the frame) and the background's otherwise.

use crate::codegen::Emit;
use crate::graph::{Graph, NodeId, NodeKind};
use crate::nodes::NodeDef;
use crate::skyview::SkyCam;

const TOPO: u64 = 0x4EF4_0000;

/// What is behind the glass: a plate, or a latlong.
pub const IMAGE_INPUT: usize = 0;
/// The glass's surface: a height or a normal map.
pub const MAP_INPUT: usize = 1;
/// The camera, on the scene pipe (lowered away before codegen).
pub const CAMERA_INPUT: usize = 2;

/// Channel prefix for the keyable numbers (`refract.<name>` on
/// `Channel::Custom`, the Env precedent).
pub const PREFIX: &str = "refract.";

/// The most taps the loop runs.
pub const MAX_SAMPLES: u32 = 64;

/// What input 0 is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum RefractMode {
    /// A plate straight behind the glass.
    #[default]
    Background,
    /// An equirectangular environment at infinity, through a camera.
    Latlong,
}

/// What input 1 is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum RefractMap {
    /// One channel (red) of height; the normal is its slope.
    #[default]
    Height,
    /// A tangent-space normal map.
    Normal,
}

/// Which way a normal map's green points.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum NormalY {
    /// Green is up (OpenGL: Blender, Substance's default, Maya).
    #[default]
    Up,
    /// Green is down (DirectX: Unreal, 3ds Max).
    Down,
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct RefractParams {
    pub mode: RefractMode,
    pub map: RefractMap,
    pub normal_y: NormalY,
    /// The normal map already holds −1..1 (a float EXR) instead of the
    /// usual 0..1 encoding.
    pub signed: bool,
    /// Height maps: the relief, in pixels, a height of 1 stands for.
    pub height: f32,
    /// Normal maps: scales the tilt (x and y) before renormalising.
    pub normal: f32,
    pub ior: f32,
    /// Glass depth in pixels (Background mode).
    pub thickness: f32,
    /// Scales the bend: 0 is no glass, 1 is Snell.
    pub strength: f32,
    /// The index's spread across the spectrum: red reads `ior −`, blue
    /// `ior +` this much.
    pub dispersion: f32,
    /// The radius of the disc the normal is scattered over (0 = polished).
    pub roughness: f32,
    /// Taps, when dispersion or roughness asks for more than one.
    pub samples: u32,
    /// The latlong view when no camera is wired (Env's).
    pub rotate: [f32; 3],
    pub focal: f32,
    pub aperture: f32,
    /// Turns the latlong about the vertical, degrees.
    pub spin: f32,
    /// Stops on the result (2^x).
    pub exposure: f32,
    /// Set by [`lower`] when a camera is wired; never authored.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub cam: Option<SkyCam>,
}

impl Default for RefractParams {
    fn default() -> Self {
        Self {
            mode: RefractMode::Background,
            map: RefractMap::Height,
            normal_y: NormalY::Up,
            signed: false,
            height: 10.0,
            normal: 1.0,
            ior: 1.5,
            thickness: 50.0,
            strength: 1.0,
            dispersion: 0.0,
            roughness: 0.0,
            samples: 16,
            rotate: [0.0, 90.0, 0.0],
            focal: 35.0,
            aperture: 36.0,
            spin: 0.0,
            exposure: 0.0,
            cam: None,
        }
    }
}

const NAMES: &[&str] = &[
    "height",
    "normal",
    "ior",
    "thickness",
    "strength",
    "dispersion",
    "roughness",
    "rotate.x",
    "rotate.y",
    "rotate.z",
    "focal",
    "spin",
    "exposure",
];

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-20);
    v.map(|x| x / l)
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// GLSL's `refract`: zero on total internal reflection.
fn refract(i: [f32; 3], n: [f32; 3], eta: f32) -> [f32; 3] {
    let ni = dot(n, i);
    let k = 1.0 - eta * eta * (1.0 - ni * ni);
    if k < 0.0 {
        return [0.0; 3];
    }
    let s = eta * ni + k.sqrt();
    [0, 1, 2].map(|c| eta * i[c] - s * n[c])
}

/// `refract_enter` on the CPU: into the glass through a face with normal
/// `n` (turned to meet the ray); the ray unbent where Snell has no answer.
fn enter(i: [f32; 3], n: [f32; 3], eta: f32) -> [f32; 3] {
    let n = if dot(n, i) > 0.0 { n.map(|x| -x) } else { n };
    let t = refract(i, n, eta);
    if dot(t, t) == 0.0 { i } else { t }
}

/// `refract_slab` on the CPU: in through `n`, out through the flat `nf`.
fn slab(d: [f32; 3], n: [f32; 3], nf: [f32; 3], ior: f32) -> [f32; 3] {
    let t1 = enter(d, n, 1.0 / ior);
    let t2 = refract(t1, nf, ior);
    if dot(t2, t2) == 0.0 { t1 } else { t2 }
}

impl RefractParams {
    /// Every keyable number's name (after [`PREFIX`]).
    pub fn names() -> &'static [&'static str] {
        NAMES
    }

    fn field(&mut self, name: &str) -> Option<&mut f32> {
        Some(match name {
            "height" => &mut self.height,
            "normal" => &mut self.normal,
            "ior" => &mut self.ior,
            "thickness" => &mut self.thickness,
            "strength" => &mut self.strength,
            "dispersion" => &mut self.dispersion,
            "roughness" => &mut self.roughness,
            "rotate.x" => &mut self.rotate[0],
            "rotate.y" => &mut self.rotate[1],
            "rotate.z" => &mut self.rotate[2],
            "focal" => &mut self.focal,
            "spin" => &mut self.spin,
            "exposure" => &mut self.exposure,
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

    /// The latlong view this frame: the wired camera's, else the node's own.
    pub fn view(&self) -> SkyCam {
        self.cam.unwrap_or(SkyCam { rotate: self.rotate, focal: self.focal, aperture: self.aperture })
    }

    fn spin_rot(&self) -> [[f32; 3]; 3] {
        crate::scene3d::rotation([0.0, -self.spin, 0.0])
    }

    /// The taps the loop runs: one for polished, undispersed glass; else
    /// `samples`, and never fewer than the three a spectrum needs.
    pub fn taps(&self) -> u32 {
        let n = self.samples.clamp(1, MAX_SAMPLES);
        match (self.dispersion > 0.0, self.roughness > 0.0) {
            (false, false) => 1,
            (true, _) => n.max(3),
            (false, true) => n,
        }
    }

    /// The (unnormalised) tangent-space normal a normal-map pixel means.
    pub fn normal_of_map(&self, rgb: [f32; 3]) -> [f32; 3] {
        if !self.signed && rgb == [0.0; 3] {
            return [0.0, 0.0, 1.0];
        }
        let c = if self.signed { rgb } else { rgb.map(|x| x * 2.0 - 1.0) };
        let fy = if self.normal_y == NormalY::Down { -1.0 } else { 1.0 };
        [c[0] * self.normal, c[1] * fy * self.normal, c[2].max(1e-4)]
    }

    /// The normal of a height map whose central differences (per pixel, y
    /// DOWN) are `d`, under a proxy of `scale`.
    pub fn normal_of_slope(&self, d: [f32; 2], scale: f32) -> [f32; 3] {
        let k = self.height * scale;
        [-d[0] * k, d[1] * k, 1.0]
    }

    /// Where pixel `p` reads input 0 through glass whose tangent normal
    /// there is `n` (any length): the one-tap CPU mirror of the shader.
    /// `fmt` is the node's format, `src` the latlong's, `scale` the proxy.
    pub fn lookup(&self, p: [f32; 2], n: [f32; 3], fmt: [f32; 2], src: [f32; 2], scale: f32) -> [f32; 2] {
        let n = normalize(n);
        let ior = self.ior.max(0.05);
        match self.mode {
            RefractMode::Background => {
                let t = enter([0.0, 0.0, -1.0], n, 1.0 / ior);
                let k = self.thickness * scale * self.strength / (-t[2]).max(1e-3);
                [p[0] + t[0] * k, p[1] - t[1] * k]
            }
            RefractMode::Latlong => {
                let mut rows = [0.0f32; 12];
                crate::skyview::pack_rows_turned(&mut rows, &self.view(), fmt, &self.spin_rot());
                let m = [0, 1, 2].map(|i| [rows[i * 4], rows[i * 4 + 1], rows[i * 4 + 2]]);
                let dc = [(p[0] - 0.5 * fmt[0]) / rows[3], -(p[1] - 0.5 * fmt[1]) / rows[3], -1.0];
                let d = normalize(m.map(|r| dot(r, dc)));
                let nw = m.map(|r| dot(r, n));
                let nf = m.map(|r| r[2]);
                let t = slab(d, nw, nf, ior);
                let dir = normalize([0, 1, 2].map(|c| d[c] + (t[c] - d[c]) * self.strength));
                crate::nodes::env::latlong_px(dir, src)
            }
        }
    }

    /// [`NodeDef::pack`] with the node's format, input 0's, and the proxy.
    pub fn pack_fmt(&self, v: &mut [f32], fmt: [f32; 2], src: [f32; 2], scale: f32) {
        crate::skyview::pack_rows_turned(v, &self.view(), fmt, &self.spin_rot());
        v[12] = src[0];
        v[13] = src[1];
        v[14] = self.exposure.exp2();
        v[15] = self.taps() as f32;
        v[16] = self.ior.max(0.05);
        v[17] = self.thickness * scale;
        v[18] = self.strength;
        v[19] = self.dispersion.max(0.0);
        v[20] = self.roughness.max(0.0);
        v[21] = match self.map {
            RefractMap::Height => self.height * scale,
            RefractMap::Normal => self.normal,
        };
        v[22] = if self.normal_y == NormalY::Down { -1.0 } else { 1.0 };
        v[23] = f32::from(u8::from(self.signed));
    }
}

/// Copy each Refract's wired camera into its transient `cam` and drop the
/// scene wire, so codegen sees picture inputs only (Env's `lower`).
pub fn lower(g: &mut Graph) {
    let ids: Vec<NodeId> =
        g.nodes.iter().filter(|(_, k)| matches!(k, NodeKind::Refract(_))).map(|(id, _)| *id).collect();
    for id in ids {
        let cam = crate::skyview::camera_on(g, id, CAMERA_INPUT);
        g.edges.remove(&(id, CAMERA_INPUT));
        g.edge_out.remove(&(id, CAMERA_INPUT));
        if let Some(NodeKind::Refract(p)) = g.nodes.get_mut(&id) {
            p.cam = cam;
        }
    }
}

/// Snell at the two faces. `refract` answers zero where there is no
/// refracted ray; the ray then goes on as it was.
const WGSL: &str = "// Into glass through a face (turned to meet the ray).
fn refract_enter(i: vec3f, n: vec3f, eta: f32) -> vec3f {
    let nn = select(n, -n, dot(n, i) > 0.0);
    let t = refract(i, nn, eta);
    return select(t, i, dot(t, t) == 0.0);
}
// In through a bumpy face, out through the flat one behind it.
fn refract_slab(d: vec3f, n: vec3f, nf: vec3f, ior: f32) -> vec3f {
    let t1 = refract_enter(d, n, 1.0 / ior);
    let t2 = refract(t1, nf, ior);
    return select(t2, t1, dot(t2, t2) == 0.0);
}
";

impl NodeDef for RefractParams {
    const NAME: &'static str = "Refract";
    /// Behind the glass, the surface map, the camera.
    const INPUTS: usize = 3;
    const MASK_INPUT: Option<usize> = None;
    /// Rotation rows + focal/format (`skyview::pack_rows`), then (source
    /// w, h, gain, taps), (ior, thickness, strength, dispersion),
    /// (roughness, bump, green sign, signed).
    const VEC4_SLOTS: usize = 6;

    /// The mode and the map's kind change the code; nothing else does.
    fn topo_tag(&self) -> u64 {
        TOPO | (self.mode as u64) << 1 | self.map as u64
    }

    fn pack(&self, v: &mut [f32]) {
        self.pack_fmt(v, [1920.0, 1080.0], [1920.0, 1080.0], 1.0);
    }

    fn emit(&self, e: &mut Emit) -> String {
        if !e.has_input(IMAGE_INPUT) {
            return "vec4f(0.0)".into();
        }
        let s = e.slot();
        let pos = e.pos().to_string();
        let latlong = self.mode == RefractMode::Latlong;
        // The surface's tangent normal (x right, y up, z at the camera),
        // unnormalised. No map wired: flat glass.
        let n0 = e.fresh();
        if !e.has_input(MAP_INPUT) {
            e.line(format_args!("let {n0} = vec3f(0.0, 0.0, 1.0);"));
        } else if self.map == RefractMap::Height {
            let mut h = |dx: f32, dy: f32| e.input_at(MAP_INPUT, &format!("({pos} + vec2f({dx:.1}, {dy:.1}))"));
            let (xp, xm, yp, ym) = (h(1.0, 0.0), h(-1.0, 0.0), h(0.0, 1.0), h(0.0, -1.0));
            e.line(format_args!(
                "let {n0} = vec3f(({xm}.x - {xp}.x) * 0.5 * P.v[{s}u + 5u].y, \
                 ({yp}.x - {ym}.x) * 0.5 * P.v[{s}u + 5u].y, 1.0);"
            ));
        } else {
            let m = e.input(MAP_INPUT);
            let c = e.fresh();
            e.line(format_args!(
                "let {c} = select({m}.xyz * 2.0 - vec3f(1.0), {m}.xyz, P.v[{s}u + 5u].w > 0.5);"
            ));
            // A pixel off the map (all zero, unsigned) is flat, not (-1,-1,-1).
            e.line(format_args!(
                "let {n0} = select(vec3f({c}.x * P.v[{s}u + 5u].y, {c}.y * P.v[{s}u + 5u].z * P.v[{s}u + 5u].y, \
                 max({c}.z, 1e-4)), vec3f(0.0, 0.0, 1.0), \
                 P.v[{s}u + 5u].w < 0.5 && dot({m}.xyz, {m}.xyz) == 0.0);"
            ));
        }
        let (_, fns) = e.color_helpers();
        fns.entry("refract_enter".into()).or_insert_with(|| WGSL.into());
        let ray = crate::skyview::wgsl(fns);
        let (acc, ws, n, d) = (e.fresh(), e.fresh(), e.fresh(), e.fresh());
        e.line(format_args!("var {acc} = vec4f(0.0);"));
        e.line(format_args!("var {ws} = vec3f(0.0);"));
        e.line(format_args!("let {n} = i32(clamp(P.v[{s}u + 3u].w, 1.0, {MAX_SAMPLES}.0));"));
        if latlong {
            e.line(format_args!("let {d} = {ray}({pos}, {s}u);"));
        }
        let (k, fk, t, a, nk, ior, tr, q, w) = (
            e.fresh(),
            e.fresh(),
            e.fresh(),
            e.fresh(),
            e.fresh(),
            e.fresh(),
            e.fresh(),
            e.fresh(),
            e.fresh(),
        );
        e.line(format_args!("for (var {k} = 0; {k} < {n}; {k}++) {{"));
        e.line(format_args!("let {fk} = f32({k});"));
        // Where this tap sits in the spectrum, −1 (red) .. 1 (blue).
        e.line(format_args!(
            "let {t} = select(0.0, 2.0 * {fk} / f32(max({n} - 1, 1)) - 1.0, {n} > 1);"
        ));
        // Its nudge on the roughness disc: the R2 sequence, so radius does
        // not march with the spectrum.
        e.line(format_args!("let {a} = 6.2831853 * fract({fk} * 0.5698403);"));
        e.line(format_args!(
            "let {nk} = normalize({n0} + vec3f(vec2f(cos({a}), sin({a})) * \
             (sqrt(fract({fk} * 0.7548777 + 0.5)) * P.v[{s}u + 5u].x), 0.0));"
        ));
        e.line(format_args!("let {ior} = max(P.v[{s}u + 4u].x + P.v[{s}u + 4u].w * {t}, 0.05);"));
        if latlong {
            let (nw, nf) = (e.fresh(), e.fresh());
            e.line(format_args!(
                "let {nw} = vec3f(dot(P.v[{s}u].xyz, {nk}), dot(P.v[{s}u + 1u].xyz, {nk}), \
                 dot(P.v[{s}u + 2u].xyz, {nk}));"
            ));
            e.line(format_args!("let {nf} = vec3f(P.v[{s}u].z, P.v[{s}u + 1u].z, P.v[{s}u + 2u].z);"));
            e.line(format_args!(
                "let {tr} = normalize(mix({d}, refract_slab({d}, {nw}, {nf}, {ior}), P.v[{s}u + 4u].z));"
            ));
            // Blender's equirectangular, as Env reads it.
            e.line(format_args!(
                "let {q} = vec2f(fract((3.14159265 - atan2(-{tr}.z, -{tr}.x)) * 0.15915494), \
                 0.5 - asin(clamp({tr}.y, -1.0, 1.0)) * 0.31830989) * P.v[{s}u + 3u].xy;"
            ));
        } else {
            e.line(format_args!(
                "let {tr} = refract_enter(vec3f(0.0, 0.0, -1.0), {nk}, 1.0 / {ior});"
            ));
            e.line(format_args!(
                "let {q} = {pos} + vec2f({tr}.x, -{tr}.y) * \
                 (P.v[{s}u + 4u].y * P.v[{s}u + 4u].z / max(-{tr}.z, 1e-3));"
            ));
        }
        let sv = e.input_at(IMAGE_INPUT, &q);
        e.line(format_args!(
            "let {w} = select(vec3f(1.0), vec3f(max(-{t}, 0.0), 1.0 - abs({t}), max({t}, 0.0)), \
             P.v[{s}u + 4u].w > 0.0 && {n} > 2);"
        ));
        e.line(format_args!("{acc} += vec4f({sv}.rgb * {w}, {sv}.a);"));
        e.line(format_args!("{ws} += {w};"));
        e.line(format_args!("}}"));
        let out = e.fresh();
        e.line(format_args!(
            "let {out} = vec4f({acc}.rgb / max({ws}, vec3f(1e-6)) * P.v[{s}u + 3u].z, {acc}.a / f32({n}));"
        ));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_mode_and_the_map_kind_are_structure() {
        let a = RefractParams::default();
        let mut b = a.clone();
        b.ior = 2.4;
        b.dispersion = 0.1;
        b.roughness = 0.3;
        b.samples = 40;
        b.signed = true;
        b.normal_y = NormalY::Down;
        b.cam = Some(SkyCam { rotate: [1.0; 3], focal: 80.0, aperture: 24.0 });
        assert_eq!(a.topo_tag(), b.topo_tag());
        let tags: std::collections::HashSet<u64> = [RefractMode::Background, RefractMode::Latlong]
            .into_iter()
            .flat_map(|mode| {
                [RefractMap::Height, RefractMap::Normal]
                    .map(|map| RefractParams { mode, map, ..Default::default() }.topo_tag())
            })
            .collect();
        assert_eq!(tags.len(), 4);
    }

    #[test]
    fn every_name_reads_and_writes() {
        let mut p = RefractParams::default();
        for (i, n) in RefractParams::names().iter().enumerate() {
            assert!(p.set(n, i as f32 + 0.5), "{n}");
            assert_eq!(p.get(n), Some(i as f32 + 0.5), "{n}");
        }
        assert!(!p.set("nope", 1.0));
    }

    #[test]
    fn taps_follow_what_is_asked() {
        let mut p = RefractParams::default();
        assert_eq!(p.taps(), 1);
        p.roughness = 0.1;
        assert_eq!(p.taps(), 16);
        p.samples = 2;
        p.dispersion = 0.02;
        assert_eq!(p.taps(), 3);
        p.samples = 1000;
        assert_eq!(p.taps(), MAX_SAMPLES);
    }

    /// Flat glass: the plate where it stands, and the sky an Env shows.
    #[test]
    fn flat_glass_bends_nothing() {
        let flat = [0.0, 0.0, 1.0];
        let (fmt, src) = ([1920.0, 1080.0], [4096.0, 2048.0]);
        let p = RefractParams::default();
        assert_eq!(p.lookup([300.5, 200.5], flat, fmt, fmt, 1.0), [300.5, 200.5]);
        let l = RefractParams { mode: RefractMode::Latlong, rotate: [20.0, 50.0, 0.0], spin: 30.0, ..p };
        let env = crate::graph::EnvParams { rotate: l.rotate, spin: l.spin, ..Default::default() };
        for px in [[960.0, 540.0], [10.5, 900.5], [1800.5, 77.5]] {
            let (a, b) = (l.lookup(px, flat, fmt, src, 1.0), env.lookup(px, fmt, src));
            assert!((a[0] - b[0]).abs() < 0.05 && (a[1] - b[1]).abs() < 0.05, "{a:?} {b:?}");
        }
    }

    /// A surface tilted to face +x throws the ray towards −x; a denser
    /// glass throws it further; strength 0 is no glass.
    #[test]
    fn a_tilt_shifts_the_plate_by_snell() {
        let n = [0.2, 0.0, 1.0];
        let fmt = [100.0, 100.0];
        let at = |ior: f32, strength: f32| {
            RefractParams { ior, strength, ..Default::default() }.lookup([50.0, 50.0], n, fmt, fmt, 1.0)
        };
        let q = at(1.5, 1.0);
        // sin(i) = 1.5 sin(t); the ray leaves the normal's far side.
        let i = 0.2f32.atan();
        let tr = (i.sin() / 1.5).asin();
        let want = 50.0 - 50.0 * (i - tr).tan();
        assert!((q[0] - want).abs() < 1e-3 && q[1] == 50.0, "{q:?} want {want}");
        assert!(at(2.4, 1.0)[0] < q[0]);
        assert_eq!(at(1.5, 0.0), [50.0, 50.0]);
    }

    /// Green down flips y; a signed map is read as it is; a pixel off an
    /// unsigned map is flat.
    #[test]
    fn normal_map_conventions() {
        let p = RefractParams::default();
        let up = p.normal_of_map([0.5, 0.75, 1.0]);
        let down = RefractParams { normal_y: NormalY::Down, ..p.clone() }.normal_of_map([0.5, 0.75, 1.0]);
        assert_eq!(up, [0.0, 0.5, 1.0]);
        assert_eq!(down, [0.0, -0.5, 1.0]);
        let s = RefractParams { signed: true, ..p.clone() }.normal_of_map([0.0, 0.5, 1.0]);
        assert_eq!(s, up);
        assert_eq!(p.normal_of_map([0.0; 3]), [0.0, 0.0, 1.0]);
    }

    /// A slope rising to the right faces left; rising DOWN the picture
    /// (y down) faces up. Under a proxy the same relief is the same normal.
    #[test]
    fn a_height_slope_is_a_normal() {
        let p = RefractParams { height: 4.0, ..Default::default() };
        assert_eq!(p.normal_of_slope([0.1, 0.0], 1.0), [-0.4, 0.0, 1.0]);
        assert_eq!(p.normal_of_slope([0.0, 0.1], 1.0), [0.0, 0.4, 1.0]);
        // Half res: the same ramp is twice as steep per pixel.
        assert_eq!(p.normal_of_slope([0.2, 0.0], 0.5), [-0.4, 0.0, 1.0]);
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
        g.nodes.insert(2, NodeKind::Refract(RefractParams::default()));
        g.connect_out(2, CAMERA_INPUT, 1, 0);
        lower(&mut g);
        assert!(g.input_of(2, CAMERA_INPUT).is_none());
        let Some(NodeKind::Refract(p)) = g.nodes.get(&2) else { panic!() };
        assert_eq!(p.view(), SkyCam { rotate: [5.0, 6.0, 7.0], focal: 85.0, aperture: 36.0 });
    }
}
