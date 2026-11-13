//! Planet: noise on a SPHERE, for stars, planets and moons.
//!
//! The usual way to texture a sphere is a flat noise picture wrapped round
//! it as a latlong, and a latlong pinches: every column meets at the poles,
//! so the noise there is smeared into a starburst, and the left and right
//! edges only meet if the noise happens to tile. This node never makes a
//! flat picture. Each pixel finds the point on the unit sphere it sees and
//! samples 3D noise (the prelude's `fbm_core`, the Noise node's fBm) AT that
//! point, so the texture has the same density everywhere on the surface:
//! no poles, no seam, by construction.
//!
//! **Two ways out.**
//! - `Disc`: the sphere seen straight down the view axis (orthographic),
//!   a disc at `center` with `radius` in canvas pixels — the Plasma node's
//!   convention, so a corona and its star line up when both say the same
//!   numbers. Lit or self-lit, limb-darkened, antialiased edge, premultiplied.
//! - `Latlong`: the SAME surface unwrapped as an equirect, in Blender's
//!   convention (see `nodes::env`). Being a projection of a 3D field rather
//!   than a picture wrapped onto one, it is seamless at the edges and
//!   undistorted on the sphere — feed it to an Env, or a card.
//!
//! **Rotation turns the planet, not the camera**: `rotate` (Euler degrees,
//! Camera3D's convention) is the pose, and `spin` adds degrees a frame
//! about the planet's own axis (its local Y), so the axis stays where
//! `rotate` tilted it. `stretch` squeezes the noise along that same axis:
//! gas-giant bands. `warp` pushes the sample point through a second,
//! drifting fBm, and `evolve` drifts both, so the surface churns while it
//! turns.
//!
//! **It fuses**, a generator with no inputs: every number, the mode and
//! the noise type included, is a uniform, so nothing recompiles.

use crate::codegen::Emit;
use crate::nodes::NodeDef;

const TOPO: u64 = 0x91A7_0000;

/// Channel prefix for the keyable numbers (`planet.<name>` on
/// `Channel::Custom`, the Starfield precedent).
pub const PREFIX: &str = "planet.";

/// What the node draws.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PlanetMode {
    /// The sphere as a lit disc.
    #[default]
    Disc,
    /// The sphere's surface as an equirect latlong.
    Latlong,
}

impl PlanetMode {
    pub const ALL: [PlanetMode; 2] = [PlanetMode::Disc, PlanetMode::Latlong];
    pub fn name(self) -> &'static str {
        match self {
            PlanetMode::Disc => "disc",
            PlanetMode::Latlong => "latlong",
        }
    }
}

/// How the octaves fold (the Noise node's fBm modes).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PlanetNoise {
    #[default]
    Fbm,
    Turbulence,
    Ridged,
}

impl PlanetNoise {
    pub const ALL: [PlanetNoise; 3] = [PlanetNoise::Fbm, PlanetNoise::Turbulence, PlanetNoise::Ridged];
    pub fn name(self) -> &'static str {
        match self {
            PlanetNoise::Fbm => "fbm",
            PlanetNoise::Turbulence => "turbulence",
            PlanetNoise::Ridged => "ridged",
        }
    }
    fn code(self) -> f32 {
        match self {
            PlanetNoise::Fbm => 0.0,
            PlanetNoise::Turbulence => 1.0,
            PlanetNoise::Ridged => 2.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize), serde(default))]
pub struct PlanetParams {
    /// The node's format (see `inherit`).
    pub size: (u32, u32),
    /// Take the comp's canvas instead of `size`.
    pub inherit: bool,
    pub mode: PlanetMode,
    /// The tick being rendered, stamped per frame like a Noise's. TRANSIENT.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub time: f32,
    /// The disc, canvas pixels (Plasma's convention).
    pub center: [f32; 2],
    pub radius: f32,
    /// Edge softness, pixels.
    pub edge: f32,
    /// The planet's pose, Euler degrees (Camera3D's).
    pub rotate: [f32; 3],
    /// Turn about the planet's own axis, degrees a frame.
    pub spin: f32,
    /// Noise features across the sphere's radius.
    pub scale: f32,
    pub noise: PlanetNoise,
    pub octaves: f32,
    pub lacunarity: f32,
    pub gain: f32,
    pub seed: f32,
    /// Noise frequency along the axis, times `scale`: above 1 squeezes the
    /// features into latitude bands.
    pub stretch: f32,
    /// How far the second fBm pushes the sample point (0 = off, and free).
    pub warp: f32,
    /// How fast the noise drifts through itself, units a frame.
    pub evolve: f32,
    /// The value ramp: contrast about the middle, then an offset.
    pub contrast: f32,
    pub offset: f32,
    /// The ramp's two ends, working space.
    pub low: [f32; 3],
    pub high: [f32; 3],
    /// Stops on the result (2^x).
    pub exposure: f32,
    /// Limb darkening: 0 = flat, 1 = black at the limb (a star's look).
    pub limb: f32,
    /// How much a light shades the disc (0 = self-lit, a star; 1 = a
    /// planet lit only from `light_az`/`light_el`).
    pub light: f32,
    /// The night side's floor under that light.
    pub ambient: f32,
    /// The light's direction, degrees: azimuth round from the viewer
    /// (positive = from the right), elevation up.
    pub light_az: f32,
    pub light_el: f32,
}

impl Default for PlanetParams {
    fn default() -> Self {
        Self {
            size: (1920, 1080),
            inherit: false,
            mode: PlanetMode::Disc,
            time: 0.0,
            center: [960.0, 540.0],
            radius: 300.0,
            edge: 1.0,
            rotate: [20.0, 0.0, 0.0],
            spin: 0.2,
            scale: 3.0,
            noise: PlanetNoise::Fbm,
            octaves: 6.0,
            lacunarity: 2.0,
            gain: 0.5,
            seed: 0.0,
            stretch: 1.0,
            warp: 0.4,
            evolve: 0.0,
            contrast: 1.4,
            offset: 0.0,
            // A star by default: hot granulation, limb-darkened, self-lit.
            low: [1.0, 0.28, 0.04],
            high: [1.0, 0.85, 0.55],
            exposure: 0.0,
            limb: 0.6,
            light: 0.0,
            ambient: 0.05,
            light_az: -50.0,
            light_el: 25.0,
        }
    }
}

const NAMES: &[&str] = &[
    "center.x", "center.y", "radius", "edge", "rotate.x", "rotate.y", "rotate.z", "spin", "scale",
    "octaves", "lacunarity", "gain", "seed", "stretch", "warp", "evolve", "contrast", "offset",
    "low.r", "low.g", "low.b", "high.r", "high.g", "high.b", "exposure", "limb", "light", "ambient",
    "light.az", "light.el",
];

impl PlanetParams {
    /// Every keyable number's name (after [`PREFIX`]).
    pub fn names() -> &'static [&'static str] {
        NAMES
    }

    fn field(&mut self, name: &str) -> Option<&mut f32> {
        Some(match name {
            "center.x" => &mut self.center[0],
            "center.y" => &mut self.center[1],
            "radius" => &mut self.radius,
            "edge" => &mut self.edge,
            "rotate.x" => &mut self.rotate[0],
            "rotate.y" => &mut self.rotate[1],
            "rotate.z" => &mut self.rotate[2],
            "spin" => &mut self.spin,
            "scale" => &mut self.scale,
            "octaves" => &mut self.octaves,
            "lacunarity" => &mut self.lacunarity,
            "gain" => &mut self.gain,
            "seed" => &mut self.seed,
            "stretch" => &mut self.stretch,
            "warp" => &mut self.warp,
            "evolve" => &mut self.evolve,
            "contrast" => &mut self.contrast,
            "offset" => &mut self.offset,
            "low.r" => &mut self.low[0],
            "low.g" => &mut self.low[1],
            "low.b" => &mut self.low[2],
            "high.r" => &mut self.high[0],
            "high.g" => &mut self.high[1],
            "high.b" => &mut self.high[2],
            "exposure" => &mut self.exposure,
            "limb" => &mut self.limb,
            "light" => &mut self.light,
            "ambient" => &mut self.ambient,
            "light.az" => &mut self.light_az,
            "light.el" => &mut self.light_el,
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

    /// The planet's pose this frame: `rotate`, then `spin` turned about
    /// the planet's own Y, as a matrix taking planet space to view space
    /// (x right, y up, z towards the viewer).
    pub fn pose(&self) -> [[f32; 3]; 3] {
        let tilt = crate::scene3d::rotation(self.rotate);
        let turn = crate::scene3d::rotation([0.0, self.spin * self.time, 0.0]);
        let mut m = [[0.0f32; 3]; 3];
        for (r, row) in m.iter_mut().enumerate() {
            for (c, v) in row.iter_mut().enumerate() {
                *v = (0..3).map(|k| tilt[r][k] * turn[k][c]).sum();
            }
        }
        m
    }

    /// The direction a light at (`light_az`, `light_el`) shines FROM, view
    /// space.
    pub fn light_dir(&self) -> [f32; 3] {
        let (sa, ca) = self.light_az.to_radians().sin_cos();
        let (se, ce) = self.light_el.to_radians().sin_cos();
        [ce * sa, se, ce * ca]
    }

    /// The unit-sphere point, in PLANET space, that pixel `p` of a `fmt`
    /// picture sees — `None` off the disc. The CPU mirror of the shader
    /// (proxy 1).
    pub fn surface(&self, p: [f32; 2], fmt: [f32; 2]) -> Option<[f32; 3]> {
        let n = match self.mode {
            PlanetMode::Disc => {
                let r = self.radius.max(1e-3);
                let dx = (p[0] - self.center[0]) / r;
                let dy = -(p[1] - self.center[1]) / r;
                let r2 = dx * dx + dy * dy;
                if r2 > 1.0 {
                    return None;
                }
                [dx, dy, (1.0 - r2).sqrt()]
            }
            PlanetMode::Latlong => return Some(latlong_dir(self.pose_t(), p, fmt)),
        };
        // View to planet: the pose's transpose.
        let m = self.pose();
        Some([0, 1, 2].map(|c| m[0][c] * n[0] + m[1][c] * n[1] + m[2][c] * n[2]))
    }

    fn pose_t(&self) -> [[f32; 3]; 3] {
        let m = self.pose();
        [0, 1, 2].map(|r| [m[0][r], m[1][r], m[2][r]])
    }

    /// [`NodeDef::pack`] with the node's format and the proxy scale (the
    /// disc is canvas pixels, so it scales with a proxy).
    pub fn pack_fmt(&self, v: &mut [f32], fmt: [f32; 2], proxy: f32) {
        let m = self.pose();
        v[0] = self.center[0] * proxy;
        v[1] = self.center[1] * proxy;
        v[2] = (self.radius * proxy).max(1e-3);
        v[3] = match self.mode {
            PlanetMode::Disc => 0.0,
            PlanetMode::Latlong => 1.0,
        };
        // Rows 1..3: view -> planet (the pose's TRANSPOSE: column c of m).
        for c in 0..3 {
            v[4 + c * 4] = m[0][c];
            v[5 + c * 4] = m[1][c];
            v[6 + c * 4] = m[2][c];
        }
        v[7] = (self.edge * proxy).max(1e-3);
        v[11] = fmt[0].max(1.0);
        v[15] = fmt[1].max(1.0);
        v[16] = self.scale.max(0.0);
        v[17] = self.octaves.clamp(1.0, crate::field::FBM_MAX_OCTAVES as f32);
        v[18] = self.lacunarity;
        v[19] = self.gain;
        v[20] = self.seed.max(0.0).floor();
        v[21] = self.noise.code();
        v[22] = self.stretch;
        v[23] = self.warp.max(0.0);
        v[24] = self.evolve * self.time;
        v[25] = self.contrast;
        v[26] = self.offset;
        v[27] = self.exposure.exp2();
        v[28..31].copy_from_slice(&self.low);
        v[31] = self.limb.clamp(0.0, 1.0);
        v[32..35].copy_from_slice(&self.high);
        v[35] = self.light.clamp(0.0, 1.0);
        v[36..39].copy_from_slice(&self.light_dir());
        v[39] = self.ambient.clamp(0.0, 1.0);
        // A latlong is the surface's colour: no viewer, so no limb and no
        // light.
        if self.mode == PlanetMode::Latlong {
            v[31] = 0.0;
            v[35] = 0.0;
        }
    }
}

/// The planet-space direction pixel `p` of a `fmt` latlong shows, given
/// the view-to-planet matrix — the inverse of `env::latlong_px`
/// (Blender's equirect: centre = -X, v down).
fn latlong_dir(to_planet: [[f32; 3]; 3], p: [f32; 2], fmt: [f32; 2]) -> [f32; 3] {
    use std::f32::consts::PI;
    let u = p[0] / fmt[0];
    let v = p[1] / fmt[1];
    let th = PI - 2.0 * PI * u;
    let lat = (0.5 - v) * PI;
    let cl = lat.cos();
    let d = [-th.cos() * cl, lat.sin(), -th.sin() * cl];
    [0, 1, 2].map(|r| to_planet[r][0] * d[0] + to_planet[r][1] * d[1] + to_planet[r][2] * d[2])
}

/// The shading, one function. `n` is the view-space normal (the latlong
/// passes +Z so lighting and limb drop out), `q` the planet-space point.
const PLANET_WGSL: &str = r#"
fn planet_shade(n: vec3f, q: vec3f, slot: u32) -> vec3f {
    let nz = P.v[slot + 4u];
    let nm = P.v[slot + 5u];
    let oct = u32(nz.y);
    let sd = u32(nm.x);
    let drift = P.v[slot + 6u].x;
    var s = q * vec3f(1.0, nm.z, 1.0) * nz.x;
    if (nm.w > 0.0) {
        // Domain warp: a second, lower fBm per axis, drifting the other
        // way, pushes the sample point — swirls rather than blobs.
        let wq = s * 0.5 - vec3f(0.23, 0.61, 0.17) * drift;
        let w = vec3f(
            fbm_core(wq, oct, nz.z, nz.w, sd + 101u, 0.0),
            fbm_core(wq + vec3f(5.2, 1.3, 2.8), oct, nz.z, nz.w, sd + 202u, 0.0),
            fbm_core(wq + vec3f(1.7, 9.2, 4.1), oct, nz.z, nz.w, sd + 303u, 0.0),
        ) - vec3f(0.5);
        s = s + w * (2.0 * nm.w);
    }
    s = s + vec3f(0.37, 0.11, 0.73) * drift;
    let f = fbm_core(s, oct, nz.z, nz.w, sd, nm.y);
    let ramp = P.v[slot + 6u];
    let t = clamp((f - 0.5) * ramp.y + 0.5 + ramp.z, 0.0, 1.0);
    let lo = P.v[slot + 7u];
    let hi = P.v[slot + 8u];
    var c = mix(lo.xyz, hi.xyz, t);
    // Limb darkening on the cosine to the viewer.
    c = c * (1.0 - lo.w * (1.0 - clamp(n.z, 0.0, 1.0)));
    // A light: lambert over an ambient floor, mixed in by `light`.
    let L = P.v[slot + 9u];
    let lit = L.w + (1.0 - L.w) * max(dot(n, L.xyz), 0.0);
    c = c * mix(1.0, lit, hi.w);
    return c * ramp.w;
}

fn planet_px(pos: vec2f, slot: u32) -> vec4f {
    let h = P.v[slot];
    let r0 = P.v[slot + 1u];
    let r1 = P.v[slot + 2u];
    let r2 = P.v[slot + 3u];
    if (h.w > 0.5) {
        // Latlong: Blender's equirect, inverted (centre = -X, v down).
        let u = pos.x / r1.w;
        let v = pos.y / r2.w;
        let th = 3.14159265 - 6.28318531 * u;
        let lat = (0.5 - v) * 3.14159265;
        let cl = cos(lat);
        let d = vec3f(-cos(th) * cl, sin(lat), -sin(th) * cl);
        let q = vec3f(dot(r0.xyz, d), dot(r1.xyz, d), dot(r2.xyz, d));
        return vec4f(planet_shade(vec3f(0.0, 0.0, 1.0), q, slot), 1.0);
    }
    let dpx = pos - h.xy;
    let dist = length(dpx);
    let a = clamp((h.z - dist) / r0.w + 0.5, 0.0, 1.0);
    if (a <= 0.0) {
        return vec4f(0.0);
    }
    var d2 = vec2f(dpx.x, -dpx.y) / h.z;
    let l2 = dot(d2, d2);
    if (l2 > 1.0) {
        d2 = d2 / sqrt(l2);
    }
    let n = vec3f(d2, sqrt(max(1.0 - dot(d2, d2), 0.0)));
    let q = vec3f(dot(r0.xyz, n), dot(r1.xyz, n), dot(r2.xyz, n));
    return vec4f(planet_shade(n, q, slot) * a, a);
}
"#;

impl NodeDef for PlanetParams {
    const NAME: &'static str = "Planet";
    const INPUTS: usize = 0;
    const MASK_INPUT: Option<usize> = None;
    /// Disc + mode, three pose rows (+ edge, fmt w, fmt h), noise, noise
    /// modifiers, ramp, low + limb, high + light, light dir + ambient.
    const VEC4_SLOTS: usize = 10;

    /// One topology whatever the node holds.
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
        fns.entry("planet".into()).or_insert_with(|| PLANET_WGSL.to_string());
        let v = e.fresh();
        e.line(format_args!("let {v} = planet_px({pos}, {slot}u);"));
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_edit_is_a_uniform() {
        let a = PlanetParams::default();
        let mut b = a.clone();
        b.mode = PlanetMode::Latlong;
        b.noise = PlanetNoise::Ridged;
        b.rotate = [10.0, 20.0, 30.0];
        b.warp = 0.0;
        b.time = 40.0;
        assert_eq!(a.topo_tag(), b.topo_tag());
    }

    #[test]
    fn every_name_reads_and_writes() {
        let mut p = PlanetParams::default();
        for (i, n) in PlanetParams::names().iter().enumerate() {
            assert!(p.set(n, i as f32 + 0.5), "{n}");
            assert_eq!(p.get(n), Some(i as f32 + 0.5), "{n}");
        }
        assert!(!p.set("nope", 1.0));
    }

    fn near(a: [f32; 3], b: [f32; 3]) -> bool {
        (0..3).all(|i| (a[i] - b[i]).abs() < 1e-4)
    }

    /// Unposed, the disc's centre sees the planet's +Z and its top edge
    /// the pole (+Y); off the disc is nothing.
    #[test]
    fn the_disc_sees_the_front_of_the_sphere() {
        let p = PlanetParams { rotate: [0.0; 3], spin: 0.0, ..Default::default() };
        let fmt = [1920.0, 1080.0];
        assert!(near(p.surface([960.0, 540.0], fmt).unwrap(), [0.0, 0.0, 1.0]));
        let top = p.surface([960.0, 540.0 - 299.999], fmt).unwrap();
        assert!(top[1] > 0.999, "{top:?}");
        assert!(p.surface([960.0 + 301.0, 540.0], fmt).is_none());
    }

    /// Spin turns about the planet's OWN axis: a tilted planet's pole
    /// stays put however far it has turned.
    #[test]
    fn spin_keeps_the_pole_where_the_tilt_put_it() {
        let mut p = PlanetParams { rotate: [30.0, 0.0, 15.0], spin: 7.0, ..Default::default() };
        let pole = |p: &PlanetParams| {
            let m = p.pose();
            [m[0][1], m[1][1], m[2][1]]
        };
        p.time = 0.0;
        let a = pole(&p);
        p.time = 33.0;
        assert!(near(a, pole(&p)));
        assert!(!near(p.pose()[0], PlanetParams { time: 0.0, ..p.clone() }.pose()[0]));
    }

    /// The latlong is `env::latlong_px` inverted: a direction found from a
    /// pixel goes back to that pixel.
    #[test]
    fn the_latlong_round_trips_through_envs_mapping() {
        let id = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let fmt = [400.0, 200.0];
        for p in [[200.0, 100.0], [37.0, 20.0], [390.0, 150.0], [123.0, 177.0]] {
            let d = latlong_dir(id, p, fmt);
            let q = crate::nodes::env::latlong_px(d, fmt);
            assert!((q[0] - p[0]).abs() < 1e-2 && (q[1] - p[1]).abs() < 1e-2, "{p:?} -> {q:?}");
        }
    }
}
