//! Relight: add an environment's light to a plate, through a normal map.
//!
//! Inputs: the plate (0), its normals (1: a Normals node's output, or any
//! camera-space normal map), an optional depth (2: a Depth node's, or a
//! render's Z), a latlong environment (3: an HDRI, an Env-style sky, a
//! DiffusionLight estimate), and the mask (4). For every pixel the normal
//! picks up the environment's DIFFUSE light (irradiance, times an albedo)
//! and its SPECULAR light (the environment reflected about the normal,
//! blurred by roughness, weighted by Fresnel), and the sum is added onto
//! the plate — or output alone, for merging by hand.
//!
//! ```text
//! n    = normal (camera space: x right, y up, z towards the camera)
//! v    = the ray through the pixel (focal / aperture), r = reflect(v, n)
//! E    = irradiance(n)                 (the env, cosine-convolved)
//! S    = prefiltered(r, roughness)     (the env, GGX-blurred)
//! F    = f0 + (max(1 - rough, f0) - f0) (1 - n.v)^5
//! a    = mix(1, plate, from plate) * albedo
//! out  = plate + intensity (diffuse (1 - F) a E + specular F S) plate.a
//! ```
//!
//! The camera's `rotate` (Euler degrees, Camera3D's convention, default
//! looking at the latlong's middle like the Env node) turns camera space
//! into the latlong's; `spin` turns the latlong about the vertical, and
//! the mapping is Blender's equirectangular (`nodes::env`), so an HDRI
//! lined up for an Env node lines up here.
//!
//! **Depth, when wired, makes the environment finite**: with a `radius`
//! above 0 the environment is a sphere of that radius round the camera
//! (in the depth's units), and each pixel looks it up from where it IS —
//! its position along the ray at its depth — not from the camera. Near
//! things then catch light from the right part of a room; with radius 0
//! (or no depth) the environment is at infinity.
//!
//! **A stage**, Slow (manifesto rule 8): the environment is reduced every
//! frame — a 128x64 average (exposure and spin applied), a 32x16
//! irradiance map, four 64x32 GGX-prefiltered levels at roughness 1/4,
//! 2/4, 3/4, 1 (a mirror reads the full-resolution environment, blended
//! into the first level) — then one pass lights the plate. The reduction
//! is a few tens of millions of taps whatever the plate's size; the light
//! pass is a handful of samples a pixel.
//!
//! **The environment arrives whole.** An aux port is rendered at the
//! canvas, in pixels, which would crop a latlong of any other size;
//! [`lower`] slips a minted Reformat (size 0 = "this node's format",
//! distort, top-left) between the environment and the port, so the aux
//! texture holds the whole latlong, squashed to the plate's format. The
//! passes read it through that format.
//!
//! Unwired normals or environment: no passes, the plate passes through.

use crate::codegen::Emit;
use crate::cost::Speed;
use crate::graph::{Graph, NodeId, NodeKind, ReformatParams, ResizeMode};
use crate::nodes::stage::{Bind, Dispatch, Pass, StageCx, StageDef, StageProgram, TexFormat, TexSpec, Wired};
use crate::nodes::NodeDef;

pub const PLATE_INPUT: usize = 0;
pub const NORMAL_INPUT: usize = 1;
pub const DEPTH_INPUT: usize = 2;
pub const ENV_INPUT: usize = 3;
pub const MASK: usize = 4;

/// Channel prefix for the keyable numbers (`relight.<name>` on
/// `Channel::Custom`, the Env / Text precedent).
pub const PREFIX: &str = "relight.";

/// The reduced environment, the irradiance map and one specular level.
pub const ENV_W: u32 = 128;
pub const ENV_H: u32 = 64;
pub const IRR_W: u32 = 32;
pub const IRR_H: u32 = 16;
pub const SPEC_W: u32 = 64;
pub const SPEC_H: u32 = 32;
pub const SPEC_LEVELS: u32 = 4;

/// What the node outputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum RelightMode {
    /// The plate with the light added.
    #[default]
    Add,
    /// The added light alone (alpha the plate's), to merge by hand.
    LightOnly,
}

impl RelightMode {
    pub const ALL: [RelightMode; 2] = [RelightMode::Add, RelightMode::LightOnly];
    pub fn name(self) -> &'static str {
        match self {
            RelightMode::Add => "add to plate",
            RelightMode::LightOnly => "light only",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct RelightParams {
    /// Multiplies all the added light.
    pub intensity: f32,
    /// Diffuse and specular gains.
    pub diffuse: f32,
    pub specular: f32,
    /// 0 a mirror, 1 fully rough.
    pub roughness: f32,
    /// Reflectance head-on (0.04 for most dielectrics).
    pub f0: f32,
    /// The albedo approximation: `mix(1, plate, from_plate) * albedo`.
    pub albedo: f32,
    pub from_plate: f32,
    /// The environment's exposure, stops.
    pub exposure: f32,
    /// Turns the latlong about the vertical, degrees.
    pub spin: f32,
    /// The camera: Euler degrees and lens, Camera3D's convention.
    pub rotate: [f32; 3],
    pub focal: f32,
    pub aperture: f32,
    /// The environment's radius in the depth's units; 0 = at infinity.
    pub radius: f32,
    /// How the normal input is stored.
    pub normals_01: bool,
    pub flip_x: bool,
    pub flip_y: bool,
    pub mode: RelightMode,
}

impl Default for RelightParams {
    fn default() -> Self {
        Self {
            intensity: 1.0,
            diffuse: 1.0,
            specular: 1.0,
            roughness: 0.4,
            f0: 0.04,
            albedo: 1.0,
            from_plate: 1.0,
            exposure: 0.0,
            spin: 0.0,
            rotate: [0.0, 90.0, 0.0],
            focal: 35.0,
            aperture: 36.0,
            radius: 0.0,
            normals_01: false,
            flip_x: false,
            flip_y: false,
            mode: RelightMode::Add,
        }
    }
}

const NAMES: &[&str] = &[
    "intensity", "diffuse", "specular", "roughness", "f0", "albedo", "from_plate", "exposure", "spin", "rotate.x",
    "rotate.y", "rotate.z", "focal", "radius",
];

impl RelightParams {
    /// Every keyable number's name (after [`PREFIX`]).
    pub fn names() -> &'static [&'static str] {
        NAMES
    }

    fn field(&mut self, name: &str) -> Option<&mut f32> {
        Some(match name {
            "intensity" => &mut self.intensity,
            "diffuse" => &mut self.diffuse,
            "specular" => &mut self.specular,
            "roughness" => &mut self.roughness,
            "f0" => &mut self.f0,
            "albedo" => &mut self.albedo,
            "from_plate" => &mut self.from_plate,
            "exposure" => &mut self.exposure,
            "spin" => &mut self.spin,
            "rotate.x" => &mut self.rotate[0],
            "rotate.y" => &mut self.rotate[1],
            "rotate.z" => &mut self.rotate[2],
            "focal" => &mut self.focal,
            "radius" => &mut self.radius,
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

    /// The latlong u offset `spin` applies (a turn about the vertical).
    fn spin_u(&self) -> f32 {
        -self.spin / 360.0
    }

    /// Fresnel-Schlick with the roughness term (CPU mirror of the shader).
    pub fn fresnel(&self, n_dot_v: f32) -> f32 {
        let f0 = self.f0.clamp(0.0, 1.0);
        let r = self.roughness.clamp(0.0, 1.0);
        f0 + ((1.0 - r).max(f0) - f0) * (1.0 - n_dot_v.clamp(0.0, 1.0)).powi(5)
    }

    /// The CPU mirror of the light pass for one channel, given the
    /// environment's irradiance `e` and prefiltered radiance `s` there.
    pub fn shade(&self, plate: f32, alpha: f32, e: f32, s: f32, n_dot_v: f32) -> f32 {
        let f = self.fresnel(n_dot_v);
        let a = (1.0 + (plate.clamp(0.0, 1.0) - 1.0) * self.from_plate.clamp(0.0, 1.0)) * self.albedo.max(0.0);
        let light = self.intensity * (self.diffuse * (1.0 - f) * a * e + self.specular * f * s) * alpha;
        match self.mode {
            RelightMode::Add => plate + light,
            RelightMode::LightOnly => light,
        }
    }
}

/// Blender's equirectangular mapping, both ways (`nodes::env`): u from
/// the picture's left, v from its top; the middle is -X.
macro_rules! latlong_wgsl {
    () => {
        r#"
const PI: f32 = 3.14159265358979;
fn ll_uv(d: vec3f) -> vec2f {
    let u = (PI - atan2(-d.z, -d.x)) / (2.0 * PI);
    let v = 0.5 - asin(clamp(d.y, -1.0, 1.0)) / PI;
    return vec2f(fract(u), v);
}
fn ll_dir(uv: vec2f) -> vec3f {
    let phi = PI - uv.x * 2.0 * PI;
    let lat = (0.5 - uv.y) * PI;
    let c = cos(lat);
    return vec3f(-c * cos(phi), sin(lat), -c * sin(phi));
}
"#
    };
}

/// The environment, averaged into 128x64 (exposure and spin applied):
/// 12x12 bilinear taps a texel over the latlong's region of the aux
/// texture.
pub const DOWN_WGSL: &str = concat!(
    latlong_wgsl!(),
    r#"
struct D { region: vec2f, canvas: vec2f, gain: f32, spin_u: f32, _p0: f32, _p1: f32 }
@group(0) @binding(0) var<uniform> U: D;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var env: texture_2d<f32>;
@group(0) @binding(3) var dst: texture_storage_2d<rgba32float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(dst);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let n = 12;
    var acc = vec3f(0.0);
    for (var j = 0; j < n; j = j + 1) {
        for (var i = 0; i < n; i = i + 1) {
            let uv = (vec2f(gid.xy) + (vec2f(f32(i), f32(j)) + 0.5) / f32(n)) / vec2f(dims);
            let t = vec2f(fract(uv.x + U.spin_u), uv.y) * U.region / U.canvas;
            acc += max(textureSampleLevel(env, samp, t, 0.0).rgb, vec3f(0.0));
        }
    }
    textureStore(dst, gid.xy, vec4f(acc / f32(n * n) * U.gain, 1.0));
}
"#
);

/// Irradiance / pi: the light a white Lambertian surface facing each
/// texel's direction sends back.
pub const IRR_WGSL: &str = concat!(
    latlong_wgsl!(),
    r#"
@group(0) @binding(0) var env: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(dst);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let n = ll_dir((vec2f(gid.xy) + 0.5) / vec2f(dims));
    let ed = vec2i(textureDimensions(env));
    let cell = (2.0 * PI / f32(ed.x)) * (PI / f32(ed.y));
    var acc = vec3f(0.0);
    for (var y = 0; y < ed.y; y = y + 1) {
        let v = (f32(y) + 0.5) / f32(ed.y);
        let sa = cos((0.5 - v) * PI) * cell;
        for (var x = 0; x < ed.x; x = x + 1) {
            let d = ll_dir(vec2f((f32(x) + 0.5) / f32(ed.x), v));
            let c = dot(n, d);
            if (c > 0.0) {
                acc += textureLoad(env, vec2i(x, y), 0).rgb * (c * sa);
            }
        }
    }
    textureStore(dst, gid.xy, vec4f(acc / PI, 1.0));
}
"#
);

/// Four GGX-prefiltered levels stacked top to bottom, roughness (k+1)/4,
/// n = v = r (the split-sum approximation), weighted by n.l.
pub const SPEC_WGSL: &str = concat!(
    latlong_wgsl!(),
    r#"
@group(0) @binding(0) var env: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba16float, write>;

const LEVELS: u32 = 4u;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(dst);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let lh = dims.y / LEVELS;
    let level = gid.y / lh;
    let rough = f32(level + 1u) / f32(LEVELS);
    let a = rough * rough;
    let a2 = a * a;
    let r = ll_dir((vec2f(f32(gid.x), f32(gid.y - level * lh)) + 0.5) / vec2f(f32(dims.x), f32(lh)));
    let ed = vec2i(textureDimensions(env));
    let cell = (2.0 * PI / f32(ed.x)) * (PI / f32(ed.y));
    var acc = vec3f(0.0);
    var wsum = 0.0;
    for (var y = 0; y < ed.y; y = y + 1) {
        let v = (f32(y) + 0.5) / f32(ed.y);
        let sa = cos((0.5 - v) * PI) * cell;
        for (var x = 0; x < ed.x; x = x + 1) {
            let d = ll_dir(vec2f((f32(x) + 0.5) / f32(ed.x), v));
            let nl = dot(r, d);
            if (nl > 0.0) {
                let h = normalize(r + d);
                let c = max(dot(r, h), 0.0);
                let q = c * c * (a2 - 1.0) + 1.0;
                let w = a2 / (PI * q * q) * nl * sa;
                acc += textureLoad(env, vec2i(x, y), 0).rgb * w;
                wsum += w;
            }
        }
    }
    textureStore(dst, gid.xy, vec4f(acc / max(wsum, 1e-8), 1.0));
}
"#
);

/// The light pass: one normal, one ray, a few samples a pixel.
pub const COMP_WGSL: &str = concat!(
    latlong_wgsl!(),
    r#"
struct C { v: array<vec4f, 8> }
@group(0) @binding(0) var<uniform> U: C;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var plate: texture_2d<f32>;
@group(0) @binding(3) var nrm: texture_2d<f32>;
@group(0) @binding(4) var env: texture_2d<f32>;
@group(0) @binding(5) var irr: texture_2d<f32>;
@group(0) @binding(6) var spec: texture_2d<f32>;
@group(0) @binding(7) var dep: texture_2d<f32>;
@group(0) @binding(8) var dst: texture_storage_2d<rgba16float, write>;

fn rot(d: vec3f) -> vec3f {
    return vec3f(dot(U.v[4].xyz, d), dot(U.v[5].xyz, d), dot(U.v[6].xyz, d));
}

// The full-resolution environment, for a mirror.
fn mirror(d: vec3f) -> vec3f {
    let uv = ll_uv(d);
    let t = vec2f(fract(uv.x + U.v[7].z), uv.y) * U.v[7].xy / U.v[0].xy;
    return max(textureSampleLevel(env, samp, t, 0.0).rgb, vec3f(0.0)) * U.v[7].w;
}

fn level(d: vec3f, k: f32) -> vec3f {
    let uv = ll_uv(d);
    let lh = 1.0 / 4.0;
    let ph = 0.5 / f32(textureDimensions(spec).y);
    let v = (k + clamp(uv.y, ph * 4.0, 1.0 - ph * 4.0)) * lh;
    return textureSampleLevel(spec, samp, vec2f(uv.x, v), 0.0).rgb;
}

fn glossy(d: vec3f, rough: f32) -> vec3f {
    let lv = clamp(rough, 0.0, 1.0) * 4.0;
    if (lv <= 1.0) {
        return mix(mirror(d), level(d, 0.0), lv);
    }
    let k = min(floor(lv - 1.0), 2.0);
    return mix(level(d, k), level(d, k + 1.0), clamp(lv - 1.0 - k, 0.0, 1.0));
}

// A finite environment: where the ray from `p` along `d` meets the sphere.
fn local(p: vec3f, d: vec3f) -> vec3f {
    let r = U.v[3].z;
    if (r <= 0.0 || U.v[3].w < 0.5) { return d; }
    let b = dot(p, d);
    let c = dot(p, p) - r * r;
    if (c >= 0.0) { return d; }
    let t = -b + sqrt(b * b - c);
    return normalize(p + t * d);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(dst);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let px = vec2i(gid.xy);
    let c = textureLoad(plate, px, 0);
    var nc = textureLoad(nrm, px, 0).xyz;
    if (U.v[2].z > 0.5) { nc = nc * 2.0 - 1.0; }
    nc = nc * vec3f(U.v[2].w, U.v[3].x, 1.0);
    let len = length(nc);
    var light = vec3f(0.0);
    if (len > 0.01) {
        let n = rot(nc / len);
        let f = U.v[0].z;
        let rc = normalize(vec3f((f32(px.x) + 0.5 - 0.5 * U.v[0].x) / f, -(f32(px.y) + 0.5 - 0.5 * U.v[0].y) / f, -1.0));
        let ray = rot(rc);
        var p = vec3f(0.0);
        if (U.v[3].w > 0.5) {
            let z = textureLoad(dep, px, 0).r;
            p = rot(rc * (z / max(-rc.z, 1e-4)));
        }
        let ndv = clamp(dot(n, -ray), 0.0, 1.0);
        let r = reflect(ray, n);
        let e = textureSampleLevel(irr, samp, ll_uv(local(p, n)), 0.0).rgb;
        let s = glossy(local(p, r), U.v[1].z);
        let f0 = clamp(U.v[1].w, 0.0, 1.0);
        let fr = f0 + (max(1.0 - clamp(U.v[1].z, 0.0, 1.0), f0) - f0) * pow(1.0 - ndv, 5.0);
        let alb = mix(vec3f(1.0), clamp(c.rgb, vec3f(0.0), vec3f(1.0)), clamp(U.v[2].y, 0.0, 1.0)) * max(U.v[2].x, 0.0);
        light = U.v[0].w * (U.v[1].x * (1.0 - fr) * alb * e + U.v[1].y * fr * s) * c.a;
    }
    var o = vec4f(c.rgb + light, c.a);
    if (U.v[3].y > 0.5) { o = vec4f(light, c.a); }
    textureStore(dst, gid.xy, o);
}
"#
);

impl NodeDef for RelightParams {
    const NAME: &'static str = "Relight";
    /// Plate, normals, depth, environment, then the mask.
    const INPUTS: usize = 5;
    const MASK_INPUT: Option<usize> = Some(MASK);
    /// Every param lives in the chain's own uniforms.
    const VEC4_SLOTS: usize = 0;

    /// Constant: every param is a uniform, and which ports are wired is
    /// already topology.
    fn topo_tag(&self) -> u64 {
        0x2E_1167
    }

    fn pack(&self, _: &mut [f32]) {}

    fn emit(&self, e: &mut Emit) -> String {
        let v = e.stage_result();
        e.masked_stage(Self::MASK_INPUT, v)
    }
}

impl StageDef for RelightParams {
    /// Normals and environment (both, or nothing to do), then depth when
    /// wired: `aux0` normals, `aux1` environment, `aux2` depth.
    fn aux_ports(&self, wired: Wired) -> Vec<usize> {
        if !(wired.has(NORMAL_INPUT) && wired.has(ENV_INPUT)) {
            return Vec::new();
        }
        let mut v = vec![NORMAL_INPUT, ENV_INPUT];
        if wired.has(DEPTH_INPUT) {
            v.push(DEPTH_INPUT);
        }
        v
    }

    fn program(&self, _canvas: (u32, u32), wired: Wired) -> StageProgram {
        let aux = self.aux_ports(wired);
        if aux.is_empty() {
            return StageProgram { textures: Vec::new(), passes: Vec::new(), result: "a".into() };
        }
        let depth = if aux.len() > 2 { "aux2" } else { "a" };
        StageProgram {
            textures: vec![
                TexSpec::sized("envs", ENV_W, ENV_H, TexFormat::Rgba32Float),
                TexSpec::sized("irr", IRR_W, IRR_H, TexFormat::Rgba16Float),
                TexSpec::sized("spec", SPEC_W, SPEC_H * SPEC_LEVELS, TexFormat::Rgba16Float),
                TexSpec::full("out"),
            ],
            passes: vec![
                Pass {
                    label: "relight-env",
                    wgsl: DOWN_WGSL,
                    binds: vec![Bind::Uniform, Bind::Sampler, Bind::Tex("aux1".into()), Bind::Storage("envs".into())],
                    dispatch: Dispatch::Pixels("envs".into()),
                    uniform_floats: 8,
                },
                Pass {
                    label: "relight-irradiance",
                    wgsl: IRR_WGSL,
                    binds: vec![Bind::Tex("envs".into()), Bind::Storage("irr".into())],
                    dispatch: Dispatch::Pixels("irr".into()),
                    uniform_floats: 0,
                },
                Pass {
                    label: "relight-specular",
                    wgsl: SPEC_WGSL,
                    binds: vec![Bind::Tex("envs".into()), Bind::Storage("spec".into())],
                    dispatch: Dispatch::Pixels("spec".into()),
                    uniform_floats: 0,
                },
                Pass {
                    label: "relight-light",
                    wgsl: COMP_WGSL,
                    binds: vec![
                        Bind::Uniform,
                        Bind::Sampler,
                        Bind::Tex("a".into()),
                        Bind::Tex("aux0".into()),
                        Bind::Tex("aux1".into()),
                        Bind::Tex("irr".into()),
                        Bind::Tex("spec".into()),
                        Bind::Tex(depth.into()),
                        Bind::Storage("out".into()),
                    ],
                    dispatch: Dispatch::Pixels("out".into()),
                    uniform_floats: 32,
                },
            ],
            result: "out".into(),
        }
    }

    fn uniforms(&self, pass: usize, cx: &StageCx) -> Vec<f32> {
        self.uniforms_aux(pass, cx, &[])
    }

    fn uniforms_aux(&self, pass: usize, cx: &StageCx, aux_fmt: &[(u32, u32)]) -> Vec<f32> {
        let canvas = [cx.canvas.0.max(1) as f32, cx.canvas.1.max(1) as f32];
        // The latlong's region of the aux texture: its (minted Reformat's)
        // format, never past the canvas.
        let env = aux_fmt.get(1).map(|f| [f.0 as f32, f.1 as f32]).unwrap_or(canvas);
        let region = [env[0].min(canvas[0]).max(1.0), env[1].min(canvas[1]).max(1.0)];
        let gain = self.exposure.exp2();
        match pass {
            0 => vec![region[0], region[1], canvas[0], canvas[1], gain, self.spin_u(), 0.0, 0.0],
            1 | 2 => Vec::new(),
            _ => {
                let f = self.focal.max(0.1) / self.aperture.max(0.1) * canvas[0];
                let r = crate::scene3d::rotation(self.rotate);
                let sign = |b: bool| if b { -1.0 } else { 1.0 };
                // One entry per aux port, then the node's own format.
                let has_depth = aux_fmt.len() > 3;
                vec![
                    canvas[0], canvas[1], f, self.intensity,
                    self.diffuse, self.specular, self.roughness, self.f0,
                    self.albedo, self.from_plate, if self.normals_01 { 1.0 } else { 0.0 }, sign(self.flip_x),
                    sign(self.flip_y), if self.mode == RelightMode::LightOnly { 1.0 } else { 0.0 }, self.radius.max(0.0), if has_depth { 1.0 } else { 0.0 },
                    r[0][0], r[0][1], r[0][2], 0.0,
                    r[1][0], r[1][1], r[1][2], 0.0,
                    r[2][0], r[2][1], r[2][2], 0.0,
                    region[0], region[1], self.spin_u(), gain,
                ]
            }
        }
    }

    fn speed(&self) -> Speed {
        Speed::Slow
    }
}

/// Every Relight's environment through a minted Reformat to the node's
/// own format (size 0 = "the format it is stamped with", distort,
/// top-left), so the aux texture the stage reads holds the WHOLE latlong
/// however large it is (an aux port is rendered at the canvas, in
/// pixels). The minted node takes the Relight's format by alias. Run
/// with the other lowerings, before codegen.
pub fn lower(g: &mut Graph) {
    let mut ids: Vec<NodeId> =
        g.nodes.iter().filter(|(_, k)| matches!(k, NodeKind::Relight(_))).map(|(id, _)| *id).collect();
    if ids.is_empty() {
        return;
    }
    ids.sort_unstable();
    let mut next = g.nodes.keys().copied().max().unwrap_or(0) + 1;
    for id in ids {
        let Some((src, out)) = g.input_port(id, ENV_INPUT) else { continue };
        let fit = next;
        next += 1;
        g.nodes.insert(
            fit,
            NodeKind::Reformat(ReformatParams { size: (0, 0), mode: ResizeMode::Distort, scale: 1.0, center: false, ..Default::default() }),
        );
        g.connect_out(fit, 0, src, out);
        g.connect_out(id, ENV_INPUT, fit, 0);
        g.format_alias.insert(fit, id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_pass_validates() {
        for (name, src) in [("down", DOWN_WGSL), ("irr", IRR_WGSL), ("spec", SPEC_WGSL), ("comp", COMP_WGSL)] {
            let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(src)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                .validate(&module)
                .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        }
    }

    #[test]
    fn every_number_reads_and_writes_by_name() {
        let mut p = RelightParams::default();
        for (i, n) in RelightParams::names().iter().enumerate() {
            assert!(p.set(n, i as f32 + 0.5), "{n}");
            assert_eq!(p.get(n), Some(i as f32 + 0.5), "{n}");
        }
        assert!(!p.set("nope", 1.0));
    }

    #[test]
    fn nothing_to_do_without_normals_and_an_environment() {
        let p = RelightParams::default();
        for w in [[true, false, false, true, false], [true, true, false, false, false], [true, false, true, false, false]] {
            let prog = p.program((64, 64), Wired(&w));
            assert!(prog.passes.is_empty() && prog.textures.is_empty(), "{w:?}");
            assert_eq!(prog.result, "a");
        }
        assert_eq!(p.aux_ports(Wired(&[true, true, false, true, false])), vec![1, 3]);
        assert_eq!(p.aux_ports(Wired(&[true, true, true, true, false])), vec![1, 3, 2]);
    }

    #[test]
    fn depth_binds_only_when_wired() {
        let p = RelightParams::default();
        let a = p.program((64, 64), Wired(&[true, true, false, true, false]));
        let b = p.program((64, 64), Wired(&[true, true, true, true, false]));
        assert_eq!(a.passes.last().unwrap().binds[7], Bind::Tex("a".into()));
        assert_eq!(b.passes.last().unwrap().binds[7], Bind::Tex("aux2".into()));
        for prog in [a, b] {
            for (i, pass) in prog.passes.iter().enumerate() {
                let cx = StageCx { canvas: (64, 64), proxy_scale: 1.0, frame: 0.0, fps: 24.0, passes: 4 };
                let u = p.uniforms_aux(i, &cx, &[(64, 64), (128, 64), (64, 64)]);
                assert_eq!(u.len(), pass.uniform_floats, "pass {i}");
            }
        }
    }

    #[test]
    fn the_depth_flag_follows_the_aux_ports() {
        let p = RelightParams::default();
        let cx = StageCx { canvas: (64, 64), proxy_scale: 1.0, frame: 0.0, fps: 24.0, passes: 4 };
        // aux_fmt is one entry per aux port, then the node's own format.
        assert_eq!(p.uniforms_aux(3, &cx, &[(64, 64), (64, 64), (64, 64)])[15], 0.0);
        assert_eq!(p.uniforms_aux(3, &cx, &[(64, 64), (64, 64), (64, 64), (64, 64)])[15], 1.0);
    }

    #[test]
    fn the_light_adds_and_the_terms_behave() {
        let p = RelightParams::default();
        // No light in the environment: the plate as it was.
        assert_eq!(p.shade(0.3, 1.0, 0.0, 0.0, 1.0), 0.3);
        // No coverage, no light.
        assert_eq!(p.shade(0.3, 0.0, 5.0, 5.0, 1.0), 0.3);
        // Head-on, a dielectric reflects f0; grazing, everything.
        assert!((p.fresnel(1.0) - 0.04).abs() < 1e-6);
        let mirror = RelightParams { roughness: 0.0, ..Default::default() };
        assert!((mirror.fresnel(0.0) - 1.0).abs() < 1e-6);
        let only = RelightParams { mode: RelightMode::LightOnly, from_plate: 0.0, specular: 0.0, ..Default::default() };
        assert!((only.shade(0.3, 1.0, 2.0, 0.0, 1.0) - 0.96 * 2.0).abs() < 1e-5);
        // The albedo approximation: from the plate, or flat.
        let flat = RelightParams { from_plate: 0.0, albedo: 0.5, specular: 0.0, mode: RelightMode::LightOnly, ..Default::default() };
        let plate = RelightParams { from_plate: 1.0, albedo: 1.0, specular: 0.0, mode: RelightMode::LightOnly, ..Default::default() };
        assert!((flat.shade(0.9, 1.0, 1.0, 0.0, 1.0) - 0.48).abs() < 1e-5);
        assert!((plate.shade(0.25, 1.0, 1.0, 0.0, 1.0) - 0.24).abs() < 1e-5);
    }

    #[test]
    fn the_environment_is_fitted_to_the_node_by_a_minted_reformat() {
        let mut g = Graph::default();
        g.nodes.insert(1, NodeKind::Grid(Default::default()));
        g.nodes.insert(2, NodeKind::Grid(Default::default()));
        g.nodes.insert(3, NodeKind::Grid(Default::default()));
        g.nodes.insert(9, NodeKind::Relight(Default::default()));
        g.connect_out(9, PLATE_INPUT, 1, 0);
        g.connect_out(9, NORMAL_INPUT, 2, 0);
        g.connect_out(9, ENV_INPUT, 3, 0);
        lower(&mut g);
        let fit = g.input_of(9, ENV_INPUT).unwrap();
        assert_eq!(fit, 10);
        assert!(matches!(g.nodes[&fit], NodeKind::Reformat(ReformatParams { size: (0, 0), mode: ResizeMode::Distort, center: false, .. })));
        assert_eq!(g.input_of(fit, 0), Some(3));
        assert_eq!(g.format_alias.get(&fit), Some(&9));
        assert_eq!(g.input_of(9, NORMAL_INPUT), Some(2), "only the environment is fitted");
    }
}
