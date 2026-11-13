//! Bokeh Kernel: a generator that draws a lens aperture — the shape an
//! out-of-focus highlight takes. Wire it into a Defocus's kernel port (B)
//! in Accurate mode, or into a Convolve's kernel port (K) for a fixed
//! blur of that shape.
//!
//! The disc fills the format's short side (`size` 1 = edge to edge), centred
//! on the middle pixel — which is the unit disc a Defocus maps a CoC onto,
//! so the authored shape is exactly the bokeh. Everything is a uniform:
//!
//! - `blades` (0 or 1 = round, 3..=16 a polygon) and `roundness`, which bows
//!   a polygon's sides out towards the circle, like a real iris at f/2.8;
//! - `catseye`: the disc clipped by a second, offset one — optical
//!   vignetting, the lemon shapes at a frame's edge — along `catseye_angle`;
//! - `ring`: brightness towards the rim (> 0, the soap-bubble edge of a
//!   spherically over-corrected lens) or the centre (< 0);
//! - `fringe`: the channels' radii spread, a chromatic rim;
//! - `squeeze`: anamorphic, the disc narrowed sideways;
//! - `softness`: the edge's width in pixels.

use crate::codegen::Emit;
use crate::nodes::NodeDef;

const TOPO: u64 = 0x6D00;

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct BokehKernelParams {
    pub size: (u32, u32),
    pub inherit: bool,
    /// Aperture blades; under 3 is a circle. Rounded.
    pub blades: f32,
    /// 0 = straight blade edges, 1 = a circle.
    pub roundness: f32,
    /// Degrees.
    pub rotation: f32,
    /// Radius as a fraction of half the short side.
    pub radius: f32,
    /// Edge width, pixels (scaled by the proxy).
    pub softness: f32,
    /// 0 = round; 1 = clipped to a sliver.
    pub catseye: f32,
    /// Which way the clipping disc sits, degrees.
    pub catseye_angle: f32,
    /// Rim (> 0) or centre (< 0) weighting, -1..1.
    pub ring: f32,
    /// Spread of the channels' radii.
    pub fringe: f32,
    /// Horizontal scale, 1 = round.
    pub squeeze: f32,
    pub intensity: f32,
}

impl Default for BokehKernelParams {
    fn default() -> Self {
        Self {
            size: (256, 256),
            inherit: false,
            blades: 6.0,
            roundness: 0.3,
            rotation: 0.0,
            radius: 1.0,
            softness: 1.0,
            catseye: 0.0,
            catseye_angle: 0.0,
            ring: 0.0,
            fringe: 0.0,
            squeeze: 1.0,
            intensity: 1.0,
        }
    }
}

impl BokehKernelParams {
    /// [`NodeDef::pack`] with the node's format and the proxy.
    pub fn pack_fmt(&self, v: &mut [f32], fmt: [f32; 2], proxy: f32) {
        let half = 0.5 * fmt[0].min(fmt[1]);
        let r = (self.radius.max(0.0) * half).max(0.5);
        v[0] = (fmt[0] * 0.5).floor() + 0.5;
        v[1] = (fmt[1] * 0.5).floor() + 0.5;
        v[2] = r;
        let n = self.blades.round();
        v[3] = if n < 3.0 { 0.0 } else { n.min(16.0) };
        v[4] = self.roundness.clamp(0.0, 1.0);
        v[5] = self.rotation.to_radians();
        // Edge half-width in units of the radius.
        v[6] = (self.softness * proxy).max(0.35) / r;
        v[7] = self.squeeze.clamp(0.05, 4.0);
        v[8] = self.catseye.clamp(0.0, 1.0);
        v[9] = self.catseye_angle.to_radians();
        v[10] = self.ring.clamp(-1.0, 1.0);
        v[11] = self.fringe;
        v[12] = self.intensity;
        v[13] = 0.0;
        v[14] = 0.0;
        v[15] = 0.0;
    }

    /// Coverage at `q` (in units of the radius, centred, y down) for the
    /// green channel — the CPU mirror of the shader, for tests.
    pub fn coverage(&self, q: [f32; 2]) -> f32 {
        let mut v = [0.0; 16];
        self.pack_fmt(&mut v, [256.0, 256.0], 1.0);
        let (n, round, rot, soft, sq, cat, cat_a) = (v[3], v[4], v[5], v[6], v[7], v[8], v[9]);
        let x = q[0] / sq;
        let y = q[1];
        let (s, c) = (-rot).sin_cos();
        let (x, y) = (x * c - y * s, x * s + y * c);
        let d = x.hypot(y);
        let mut rr = d;
        if n >= 3.0 {
            let seg = std::f32::consts::TAU / n;
            let a = (-y).atan2(x);
            let a = a - seg * (a / seg).floor() - seg * 0.5;
            let poly = d * a.cos() / (seg * 0.5).cos();
            rr = poly + (d - poly) * round;
        }
        let edge = |r: f32| 1.0 - smoothstep(1.0 - soft, 1.0 + soft, r);
        let mut cov = edge(rr);
        if cat > 0.0 {
            let o = [cat_a.cos() * cat * 1.2, -cat_a.sin() * cat * 1.2];
            cov *= edge((q[0] / sq - o[0]).hypot(q[1] - o[1]));
        }
        cov
    }
}

fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The aperture at pixel `p`. Slots: (centre, radius px, blades),
/// (roundness, rotation, softness / radius, squeeze), (catseye, its angle,
/// ring, fringe), (intensity, _).
pub const WGSL: &str = r#"
fn bokeh_edge(r: f32, soft: f32) -> f32 {
    return 1.0 - smoothstep(1.0 - soft, 1.0 + soft, r);
}

// Distance in radii to the aperture's rim along this direction: the circle,
// or the polygon bowed toward it by `round`.
fn bokeh_r(x: f32, y: f32, n: f32, round: f32) -> f32 {
    let d = length(vec2f(x, y));
    if (n < 3.0) { return d; }
    let seg = 6.2831853 / n;
    var a = atan2(-y, x);
    a = a - seg * floor(a / seg) - seg * 0.5;
    let poly = d * cos(a) / cos(seg * 0.5);
    return mix(poly, d, round);
}

fn bokeh_kernel(p: vec2f, slot: u32) -> vec4f {
    let a = P.v[slot];
    let b = P.v[slot + 1u];
    let c = P.v[slot + 2u];
    let gain = P.v[slot + 3u].x;
    let q = (p - a.xy) / a.z;
    let qx = q.x / b.w;
    let s = sin(-b.y);
    let co = cos(-b.y);
    let x = qx * co - q.y * s;
    let y = qx * s + q.y * co;
    let r = bokeh_r(x, y, a.w, b.x);
    // Each channel's own radius: a fringe grows red, shrinks blue.
    let k = vec3f(1.0) + c.w * vec3f(0.06, 0.0, -0.06);
    var cov = vec3f(bokeh_edge(r / k.r, b.z), bokeh_edge(r / k.g, b.z), bokeh_edge(r / k.b, b.z));
    if (c.x > 0.0) {
        let o = vec2f(cos(c.y), -sin(c.y)) * c.x * 1.2;
        cov *= bokeh_edge(length(vec2f(qx, q.y) - o), b.z);
    }
    // Rim or centre weighting, never negative.
    let w = max(1.0 + c.z * (2.0 * min(r, 1.0) * min(r, 1.0) - 1.0), 0.0);
    let rgb = cov * w * gain;
    return vec4f(rgb, clamp(cov.g, 0.0, 1.0));
}
"#;

impl NodeDef for BokehKernelParams {
    const NAME: &'static str = "Bokeh Kernel";
    const INPUTS: usize = 0;
    const MASK_INPUT: Option<usize> = None;
    const VEC4_SLOTS: usize = 4;

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
        fns.entry("bokeh_kernel".into()).or_insert_with(|| WGSL.to_string());
        let v = e.fresh();
        e.line(format_args!("let {v} = bokeh_kernel({pos}, {slot}u);"));
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_hexagon_cuts_its_corners_and_roundness_restores_them() {
        let hex = BokehKernelParams { blades: 6.0, roundness: 0.0, softness: 0.01, ..Default::default() };
        // A point inside the circle but past a straight blade edge: with
        // flat sides (rotation 0 puts a vertex on +x) the edge mid-way
        // between vertices, at 30 degrees, is at cos(30) = 0.866.
        let a = 30f32.to_radians();
        let q = [0.93 * a.cos(), -0.93 * a.sin()];
        assert!(hex.coverage(q) < 0.01, "past the blade: {}", hex.coverage(q));
        let round = BokehKernelParams { roundness: 1.0, ..hex };
        assert!(round.coverage(q) > 0.99, "a circle holds it: {}", round.coverage(q));
        assert!(hex.coverage([0.95, 0.0]) > 0.99, "a vertex reaches out");
    }

    #[test]
    fn catseye_clips_one_side() {
        let p = BokehKernelParams { blades: 0.0, catseye: 0.5, softness: 0.01, ..Default::default() };
        // The clipping disc sits towards +x: the far -x side is cut.
        assert!(p.coverage([-0.9, 0.0]) < 0.01);
        assert!(p.coverage([0.5, 0.0]) > 0.99);
    }

    #[test]
    fn every_control_is_a_uniform() {
        let a = BokehKernelParams::default();
        let b = BokehKernelParams { blades: 9.0, catseye: 0.4, ring: 0.5, ..a };
        assert_eq!(a.topo_tag(), b.topo_tag());
    }
}
