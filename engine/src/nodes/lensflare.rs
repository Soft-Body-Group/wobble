//! Lens Flare: the light a bright source scatters inside a lens, laid over
//! the picture — a glow, a starburst, an anamorphic streak, a halo ring and
//! a chain of iris ghosts along the axis through the frame's centre.
//!
//! **Fused, never a stage.** Every element is a closed-form function of the
//! pixel's position relative to the LIGHT (a point, normalised to the
//! canvas, dragged on the viewer) and the frame centre, so the whole flare
//! is ALU in the main pass: no texture, no readback, no pass. It renders
//! cyan by `cost::tests::slow_matches_codegen`.
//!
//! - **glow** — a Gaussian core with a long soft skirt;
//! - **starburst** — `rays` spikes, each its own length (hashed), sharpened
//!   by `sharp`, rotatable;
//! - **streak** — the anamorphic line through the light, tinted (blue by
//!   default), at any angle;
//! - **halo** — a ring round the light, its radius split per channel by
//!   `chroma` (the rainbow edge);
//! - **ghosts** — `count` (up to [`MAX_GHOSTS`]) aperture-shaped discs on
//!   the line from the light through the centre, at `t` hashed from `seed`
//!   (t = 2 is the light mirrored through the centre), each its own size,
//!   strength and hue; `blades` 0 is round, 3.. a polygon; split per
//!   channel by their own `chroma`.
//!
//! **Lengths are fractions of the frame HEIGHT**, packed into pixels of the
//! node's own format (proxy included), so a proxy is the same flare,
//! smaller, and ghosts stay round on any aspect.
//!
//! **Occlusion** (`occlude`, topology): the input is fused again AT the
//! light and the flare is scaled by a soft threshold on its brightness — a
//! sun that slips behind a building takes its flare with it. It costs one
//! more evaluation of the upstream per pixel (ALU, or one texel read if a
//! stage is above), so it is off until asked for. Off, the input is only
//! read at the pixel. `fade` dims the flare as the light leaves the frame.
//!
//! Combine is the Glow's: add, screen, or the flare alone. Alpha is the
//! input's (light is additive). Every number is a uniform; only `occlude`
//! moves the topology.

use crate::codegen::Emit;
use crate::nodes::glow::GlowMode;
use crate::nodes::NodeDef;
use std::collections::BTreeMap;

/// Topology tag, clear of every other node's span; `occlude` adds 1.
const TOPO: u64 = 0x1F1A_0000;

/// Most ghosts (the loop bound is a uniform under this cap).
pub const MAX_GHOSTS: u32 = 16;

/// Channel prefix for the node's keyable numbers (`flare.<name>` on
/// `Channel::Custom`, the Starfield's precedent).
pub const PREFIX: &str = "flare.";

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct LensFlareParams {
    /// The light, normalised to the canvas (0,0 top left). May sit off the
    /// frame: a sun just out of shot still flares (see `fade`).
    pub center: [f32; 2],
    /// Multiplies the whole flare.
    pub intensity: f32,
    /// Multiplies the whole flare, working space.
    pub tint: [f32; 3],
    pub mode: GlowMode,
    /// Scale the flare by the input's brightness AT the light (topology).
    pub occlude: bool,
    /// Occlusion: the brightness the light counts as fully visible at...
    pub threshold: f32,
    /// ...with this soft shoulder either side.
    pub knee: f32,
    /// How far (frame heights) past the edge the flare fades out over; 0 =
    /// never fades.
    pub fade: f32,

    pub glow: f32,
    /// Glow radius, frame heights.
    pub glow_size: f32,

    pub burst: f32,
    pub burst_rays: f32,
    /// Starburst reach, frame heights.
    pub burst_len: f32,
    /// Degrees.
    pub burst_rot: f32,
    /// 0..1: soft petals to hairlines.
    pub burst_sharp: f32,

    pub streak: f32,
    /// Streak reach and thickness, frame heights.
    pub streak_len: f32,
    pub streak_thick: f32,
    /// Degrees; 0 is horizontal (an anamorphic lens).
    pub streak_angle: f32,
    pub streak_tint: [f32; 3],

    pub halo: f32,
    /// Ring radius and width, frame heights.
    pub halo_radius: f32,
    pub halo_width: f32,
    /// Per-channel split of the radius (0 = white ring).
    pub halo_chroma: f32,

    pub ghosts: f32,
    pub ghost_count: f32,
    /// Scales every ghost's place on the axis.
    pub ghost_spread: f32,
    /// Mean ghost radius, frame heights.
    pub ghost_size: f32,
    /// A different seed is a different set of ghosts.
    pub ghost_seed: f32,
    /// 0 = white ghosts, 1 = fully coloured.
    pub ghost_color: f32,
    /// Aperture blades: 0 round, else a polygon of this many sides.
    pub ghost_blades: f32,
    pub ghost_chroma: f32,
    /// 0..1: hard-edged discs to soft blobs.
    pub ghost_soft: f32,
}

impl Default for LensFlareParams {
    fn default() -> Self {
        Self {
            center: [0.3, 0.3],
            intensity: 1.0,
            tint: [1.0; 3],
            mode: GlowMode::Add,
            occlude: false,
            threshold: 1.0,
            knee: 0.5,
            fade: 0.15,
            glow: 0.6,
            glow_size: 0.05,
            burst: 0.25,
            burst_rays: 12.0,
            burst_len: 0.12,
            burst_rot: 0.0,
            burst_sharp: 0.5,
            streak: 0.5,
            streak_len: 0.6,
            streak_thick: 0.004,
            streak_angle: 0.0,
            streak_tint: [0.45, 0.65, 1.0],
            halo: 0.05,
            halo_radius: 0.35,
            halo_width: 0.04,
            halo_chroma: 0.04,
            ghosts: 0.25,
            ghost_count: 7.0,
            ghost_spread: 1.0,
            ghost_size: 0.06,
            ghost_seed: 0.0,
            ghost_color: 0.6,
            ghost_blades: 6.0,
            ghost_chroma: 0.04,
            ghost_soft: 0.3,
        }
    }
}

/// The keyable numbers, by channel name after [`PREFIX`].
const NAMES: &[&str] = &[
    "x", "y", "intensity", "tint.r", "tint.g", "tint.b", "threshold", "knee", "fade", "glow",
    "glow.size", "burst", "burst.rays", "burst.len", "burst.rot", "burst.sharp", "streak",
    "streak.len", "streak.thick", "streak.angle", "streak.r", "streak.g", "streak.b", "halo",
    "halo.radius", "halo.width", "halo.chroma", "ghosts", "ghost.count", "ghost.spread",
    "ghost.size", "ghost.seed", "ghost.color", "ghost.blades", "ghost.chroma", "ghost.soft",
];

impl LensFlareParams {
    /// Every keyable number's name (after [`PREFIX`]).
    pub fn names() -> &'static [&'static str] {
        NAMES
    }

    fn field(&mut self, name: &str) -> Option<&mut f32> {
        Some(match name {
            "x" => &mut self.center[0],
            "y" => &mut self.center[1],
            "intensity" => &mut self.intensity,
            "tint.r" => &mut self.tint[0],
            "tint.g" => &mut self.tint[1],
            "tint.b" => &mut self.tint[2],
            "threshold" => &mut self.threshold,
            "knee" => &mut self.knee,
            "fade" => &mut self.fade,
            "glow" => &mut self.glow,
            "glow.size" => &mut self.glow_size,
            "burst" => &mut self.burst,
            "burst.rays" => &mut self.burst_rays,
            "burst.len" => &mut self.burst_len,
            "burst.rot" => &mut self.burst_rot,
            "burst.sharp" => &mut self.burst_sharp,
            "streak" => &mut self.streak,
            "streak.len" => &mut self.streak_len,
            "streak.thick" => &mut self.streak_thick,
            "streak.angle" => &mut self.streak_angle,
            "streak.r" => &mut self.streak_tint[0],
            "streak.g" => &mut self.streak_tint[1],
            "streak.b" => &mut self.streak_tint[2],
            "halo" => &mut self.halo,
            "halo.radius" => &mut self.halo_radius,
            "halo.width" => &mut self.halo_width,
            "halo.chroma" => &mut self.halo_chroma,
            "ghosts" => &mut self.ghosts,
            "ghost.count" => &mut self.ghost_count,
            "ghost.spread" => &mut self.ghost_spread,
            "ghost.size" => &mut self.ghost_size,
            "ghost.seed" => &mut self.ghost_seed,
            "ghost.color" => &mut self.ghost_color,
            "ghost.blades" => &mut self.ghost_blades,
            "ghost.chroma" => &mut self.ghost_chroma,
            "ghost.soft" => &mut self.ghost_soft,
            _ => return None,
        })
    }

    pub fn get(&self, name: &str) -> Option<f32> {
        let mut c = *self;
        c.field(name).map(|v| *v)
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

    /// Ghost `i`'s place on the light→centre axis (0 = the light, 1 = the
    /// centre, 2 = the light mirrored), its radius factor and strength —
    /// the CPU mirror of the shader's hash, for the viewer overlay and the
    /// tests.
    pub fn ghost(&self, i: u32) -> (f32, f32, f32) {
        let h = hash4([i, self.ghost_seed.floor() as i32 as u32, 0x51ed, 0x7f4a]);
        let t = (h[0] * 2.6 - 0.3) * self.ghost_spread;
        let s = 0.25 + 1.35 * h[1] * h[1];
        let a = 0.3 + 0.7 * h[2];
        (t, s, a)
    }

    /// Ghosts actually drawn.
    pub fn ghost_n(&self) -> u32 {
        self.ghost_count.clamp(0.0, MAX_GHOSTS as f32).round() as u32
    }

    /// [`NodeDef::pack`] with the node's format (pixels, proxy included).
    pub fn pack_fmt(&self, v: &mut [f32], fmt: [f32; 2]) {
        let u = fmt[1].max(1.0);
        v[0] = self.center[0] * fmt[0];
        v[1] = self.center[1] * fmt[1];
        v[2] = fmt[0];
        v[3] = fmt[1];
        v[4..7].copy_from_slice(&self.tint);
        v[7] = self.intensity.max(0.0);
        v[8] = match self.mode {
            GlowMode::Add => 0.0,
            GlowMode::Screen => 1.0,
            GlowMode::Only => 2.0,
        };
        v[9] = self.threshold;
        v[10] = self.knee.max(1e-4);
        v[11] = self.fade.max(0.0) * u;
        v[12] = self.glow.max(0.0);
        v[13] = (self.glow_size * u).max(1e-3);
        v[14] = self.burst.max(0.0);
        v[15] = self.burst_rays.clamp(1.0, 64.0).round();
        v[16] = (self.burst_len * u).max(1e-3);
        v[17] = self.burst_rot.to_radians();
        let s = self.burst_sharp.clamp(0.0, 1.0);
        v[18] = 2.0 + 400.0 * s * s;
        v[19] = self.streak.max(0.0);
        v[20] = (self.streak_len * u).max(1e-3);
        v[21] = (self.streak_thick * u).max(0.25);
        let a = self.streak_angle.to_radians();
        v[22] = a.cos();
        v[23] = a.sin();
        v[24..27].copy_from_slice(&self.streak_tint);
        v[27] = self.halo.max(0.0);
        v[28] = self.halo_radius.max(0.0) * u;
        v[29] = (self.halo_width * u).max(0.5);
        v[30] = self.halo_chroma;
        v[31] = self.ghosts.max(0.0);
        v[32] = self.ghost_n() as f32;
        v[33] = self.ghost_spread;
        v[34] = (self.ghost_size * u).max(0.5);
        v[35] = self.ghost_seed.floor();
        v[36] = self.ghost_color.clamp(0.0, 1.0);
        let b = self.ghost_blades.round();
        v[37] = if b >= 3.0 { b.min(16.0) } else { 0.0 };
        v[38] = self.ghost_chroma;
        v[39] = self.ghost_soft.clamp(0.01, 1.0);
    }
}

/// The shader's hash, mirrored: four floats in 0..1 from four words.
pub fn hash4(v0: [u32; 4]) -> [f32; 4] {
    let mut v = v0.map(|x| x.wrapping_mul(1664525).wrapping_add(1013904223));
    let mix = |v: &mut [u32; 4]| {
        v[0] = v[0].wrapping_add(v[1].wrapping_mul(v[3]));
        v[1] = v[1].wrapping_add(v[2].wrapping_mul(v[0]));
        v[2] = v[2].wrapping_add(v[0].wrapping_mul(v[1]));
        v[3] = v[3].wrapping_add(v[1].wrapping_mul(v[2]));
    };
    mix(&mut v);
    v = v.map(|x| x ^ (x >> 16));
    mix(&mut v);
    v.map(|x| (x >> 8) as f32 * (1.0 / 16777216.0))
}

/// The flare's WGSL, once per shader: the light at pixel `p`, before the
/// tint, the intensity and the occlusion.
fn wgsl(fns: &mut BTreeMap<String, String>) -> &'static str {
    const NAME: &str = "lensflare_px";
    if fns.contains_key(NAME) {
        return NAME;
    }
    fns.insert(
        NAME.into(),
        format!(
            "// Lens Flare: glow, starburst, streak, halo and ghosts, all closed
// form in the pixel's offset from the light (see nodes::lensflare).
fn lf_hash(v0: vec4u) -> vec4f {{
    var v = v0 * 1664525u + 1013904223u;
    v.x += v.y * v.w; v.y += v.z * v.x; v.z += v.x * v.y; v.w += v.y * v.z;
    v = v ^ (v >> vec4u(16u));
    v.x += v.y * v.w; v.y += v.z * v.x; v.z += v.x * v.y; v.w += v.y * v.z;
    return vec4f(v >> vec4u(8u)) * (1.0 / 16777216.0);
}}
// A disc (blades 0) or an aperture polygon of radius `r`, soft edge,
// brighter at the rim: 0 outside.
fn lf_iris(q: vec2f, r: f32, blades: f32, soft: f32) -> f32 {{
    var k = 1.0;
    if (blades >= 3.0) {{
        let seg = 6.2831853 / blades;
        let a = atan2(q.y, q.x) + 1.5707963;
        let m = a - seg * floor(a / seg) - 0.5 * seg;
        k = cos(0.5 * seg) / cos(m);
    }}
    let d = length(q) / max(r * k, 1e-3);
    let disc = 1.0 - smoothstep(1.0 - soft, 1.0, d);
    return disc * (0.55 + 0.45 * smoothstep(0.3, 1.0, d));
}}
fn {NAME}(p: vec2f, slot: u32) -> vec3f {{
    let s0 = P.v[slot];
    let s3 = P.v[slot + 3u];
    let s4 = P.v[slot + 4u];
    let s5 = P.v[slot + 5u];
    let s6 = P.v[slot + 6u];
    let s7 = P.v[slot + 7u];
    let s8 = P.v[slot + 8u];
    let s9 = P.v[slot + 9u];
    let L = s0.xy;
    let d = p - L;
    let r = length(d);
    var col = vec3f(0.0);
    // Glow: a Gaussian core and a long soft skirt.
    let g = r / s3.y;
    col += vec3f(s3.x * (exp(-g * g) + 0.08 / (1.0 + g * g)));
    // Starburst: spikes at 2πi/rays, each its own reach.
    if (s3.z > 0.0) {{
        let n = s3.w;
        let th = atan2(d.y, d.x) - s4.y;
        // The ray's index wrapped into 0..n, so the spike on the ±π seam
        // is one ray with one length, not two halves.
        let ni = max(i32(n), 1);
        let ray = ((i32(round(th * n / 6.2831853)) % ni) + ni) % ni;
        let len = s4.x * (0.2 + 1.3 * pow(lf_hash(vec4u(u32(ray), 77u, 3u, 9u)).x, 2.0));
        let spike = pow(abs(cos(0.5 * n * th)), s4.z);
        col += vec3f(s3.z * spike * exp(-r / len) * min(1.0, r / 2.0));
    }}
    // Streak: a thin line through the light, and a fainter wide one.
    if (s4.w > 0.0) {{
        let x = d.x * s5.z + d.y * s5.w;
        let y = -d.x * s5.w + d.y * s5.z;
        let ty = y / s5.y;
        let tw = y / (s5.y * 5.0);
        let line = exp(-abs(x) / s5.x) * exp(-ty * ty)
            + 0.25 * exp(-abs(x) / (s5.x * 0.4)) * exp(-tw * tw);
        col += s6.rgb * (s4.w * line);
    }}
    // Halo: a ring round the light, split per channel.
    if (s6.w > 0.0) {{
        let rr = s7.x * (1.0 + s7.z * vec3f(-1.0, 0.0, 1.0));
        let e = (vec3f(r) - rr) / s7.y;
        col += s6.w * exp(-e * e);
    }}
    // Ghosts: on the axis through the centre, aperture-shaped.
    let ng = u32(s8.x);
    if (s7.w > 0.0 && ng > 0u) {{
        let axis = 0.5 * s0.zw - L;
        let seed = bitcast<u32>(i32(s8.w));
        for (var i = 0u; i < ng; i = i + 1u) {{
            let h = lf_hash(vec4u(i, seed, 0x51edu, 0x7f4au));
            let t = (h.x * 2.6 - 0.3) * s8.y;
            let q = p - (L + axis * t);
            let rad = s8.z * (0.25 + 1.35 * h.y * h.y);
            let hue = 0.5 + 0.5 * cos(6.2831853 * (h.w + vec3f(0.0, 0.33, 0.67)));
            let c = mix(vec3f(1.0), hue, s9.x);
            let k = 1.0 + s9.z * vec3f(-1.0, 0.0, 1.0);
            let shape = vec3f(
                lf_iris(q, rad * k.x, s9.y, s9.w),
                lf_iris(q, rad * k.y, s9.y, s9.w),
                lf_iris(q, rad * k.z, s9.y, s9.w),
            );
            col += c * shape * (s7.w * (0.3 + 0.7 * h.z));
        }}
    }}
    // Fade as the light leaves the frame.
    let s2 = P.v[slot + 2u];
    if (s2.w > 0.0) {{
        let beyond = length(max(max(-L, L - s0.zw), vec2f(0.0)));
        col *= exp(-beyond / s2.w);
    }}
    return col * P.v[slot + 1u].rgb * P.v[slot + 1u].w;
}}
"
        ),
    );
    NAME
}

impl NodeDef for LensFlareParams {
    const NAME: &'static str = "Lens Flare";
    /// The picture, the mask.
    const INPUTS: usize = 2;
    const MASK_INPUT: Option<usize> = Some(1);
    /// Light + format, tint, combine/occlusion/fade, glow + burst, burst +
    /// streak, streak, streak tint + halo, halo + ghosts, ghosts, ghosts.
    const VEC4_SLOTS: usize = 10;

    /// Every number is a uniform; only occlusion re-reads the input.
    fn topo_tag(&self) -> u64 {
        TOPO + self.occlude as u64
    }

    fn pack(&self, v: &mut [f32]) {
        self.pack_fmt(v, [1920.0, 1080.0]);
    }

    fn emit(&self, e: &mut Emit) -> String {
        let slot = e.slot();
        let pos = e.pos().to_string();
        let input = e.input(0);
        let (_, fns) = e.color_helpers();
        let f = wgsl(fns);
        let fl = e.fresh();
        e.line(format_args!("var {fl} = {f}({pos}, {slot}u);"));
        if self.occlude && e.has_input(0) {
            // The input fused again AT the light.
            let at = e.input_at(0, &format!("P.v[{slot}u].xy"));
            e.line(format_args!(
                "{fl} *= smoothstep(P.v[{slot}u + 2u].y - P.v[{slot}u + 2u].z, P.v[{slot}u + 2u].y + P.v[{slot}u + 2u].z, max({at}.r, max({at}.g, {at}.b)));"
            ));
        }
        let v = e.fresh();
        e.line(format_args!("var {v} = vec4f({input}.rgb + {fl}, {input}.a);"));
        e.line(format_args!(
            "if (P.v[{slot}u + 2u].x == 1.0) {{ let b = max({input}.rgb, vec3f(0.0)); {v} = vec4f({input}.rgb - b + (1.0 - (1.0 - b) * (1.0 - min({fl}, vec3f(1.0)))), {input}.a); }} else if (P.v[{slot}u + 2u].x == 2.0) {{ {v} = vec4f({fl}, {input}.a); }}"
        ));
        e.masked(Self::MASK_INPUT, &input, v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_name_reads_and_writes() {
        let mut p = LensFlareParams::default();
        for (i, n) in LensFlareParams::names().iter().enumerate() {
            assert!(p.set(n, i as f32 + 0.5), "{n}");
            assert_eq!(p.get(n), Some(i as f32 + 0.5), "{n}");
        }
        assert!(!p.set("nope", 1.0));
    }

    #[test]
    fn only_occlusion_moves_the_topology() {
        let a = LensFlareParams::default();
        let mut b = a;
        b.center = [0.9, 0.1];
        b.ghost_count = 3.0;
        b.ghost_blades = 0.0;
        b.mode = GlowMode::Screen;
        assert_eq!(a.topo_tag(), b.topo_tag());
        b.occlude = true;
        assert_ne!(a.topo_tag(), b.topo_tag());
    }

    #[test]
    fn the_hash_mirror_is_deterministic_and_in_range() {
        let p = LensFlareParams::default();
        for i in 0..MAX_GHOSTS {
            let (t, s, a) = p.ghost(i);
            assert!((-0.3..=2.3).contains(&t) && (0.25..=1.6).contains(&s) && (0.3..=1.0).contains(&a));
            assert_eq!(p.ghost(i), (t, s, a));
        }
    }
}
