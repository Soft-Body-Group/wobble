//! Glint: star streaks off the highlights. Key what is over the threshold,
//! then smear it along each ray's direction with an exponential tail, and
//! add the rays back over the picture.
//!
//! The rays run at half resolution, and each one is a CHAIN of four 4-tap
//! passes at strides s, 4s, 16s, 64s: composed, they visit every position
//! 0..255 s exactly once, and because each tap is weighted `exp(-k d)` the
//! product of the four weights is the exact exponential tail at that
//! position. Sixteen taps a ray whatever its length — a 1000 px streak costs
//! what a 50 px one does. Where s grows past a texel the seed pass fills the
//! gaps with up to four sub-taps, so a long ray is a line, not a dotted one.
//!
//! The ray COUNT is a uniform, not topology: the program always holds
//! [`MAX_RAYS`] ray chains, and a ray past the count returns at once (its
//! last pass only carries the running sum through). So rays can be keyed,
//! and moving the slider never rebuilds anything.

use crate::codegen::Emit;
use crate::cost::Speed;
use crate::nodes::glow::{GlowMode, COMP_WGSL};
use crate::nodes::stage::{Bind, Dispatch, Pass, StageCx, StageDef, StageProgram, TexSpec, Wired};
use crate::nodes::NodeDef;

/// Ray chains the program holds. The ceiling, not the cost: unused rays
/// are dispatches that return on their first line.
pub const MAX_RAYS: usize = 8;

/// The rays run at the canvas shifted right this many times.
pub const SHIFT: u32 = 1;

/// Passes per ray chain; positions reached are `4^STEPS - 1` strides.
const STEPS: usize = 4;

/// The chain's reach in base strides: 3 + 12 + 48 + 192.
pub const REACH: f32 = 255.0;

/// Sub-taps the seed pass may add to cover a stride past one texel.
pub const MAX_SUB: u32 = 4;

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct GlintParams {
    /// Scene-linear level where the rays start. A glint comes off specular
    /// hits, not off the whole frame.
    pub threshold: f32,
    /// Soft shoulder on the threshold. 0 = hard.
    pub knee: f32,
    /// Number of rays, evenly spaced round the circle. Rounded, 0..=MAX_RAYS.
    /// A float so it keys and patches like any other value.
    pub rays: f32,
    /// Ray length in output pixels. Costs nothing to raise.
    pub length: f32,
    /// Rotation of the whole star, degrees, counter-clockwise.
    pub rotation: f32,
    /// How quickly a ray fades along its length: 0 holds full strength to
    /// the tip, 1 is gone a few pixels out.
    pub falloff: f32,
    /// Multiplier on the rays before they are combined.
    pub intensity: f32,
    /// Per-channel tint on the rays, linear.
    pub tint: [f32; 3],
    /// How the rays are combined. Same choices, and meanings, as a Glow.
    pub mode: GlowMode,
    /// Isolate small pings: take away every highlight wider than `erode`
    /// before the rays are cast (the key minus its erode-then-dilate, a
    /// white top-hat), so a glint comes off the specks and not off a bright
    /// window. STRUCTURAL (four half-res passes when on, none when off);
    /// the size is a uniform.
    pub isolate: bool,
    /// Isolate's size in output pixels: highlights wider than this lose
    /// their rays.
    pub erode: f32,
    /// Dispersion: red's rays run longer and blue's shorter by this much
    /// (0.25 per unit), so the tips fringe like a real diffraction spike.
    /// Each channel's tail is stretched, not just tinted.
    pub spread: f32,
    /// Hue drift along a ray: > 0 fades blue first (warm tails), < 0 red
    /// first (cool tails). Up to 4 extra nepers over the ray, so it acts
    /// even at falloff 0.
    pub shift: f32,
}

impl Default for GlintParams {
    fn default() -> Self {
        Self {
            threshold: 1.0,
            knee: 0.5,
            rays: 4.0,
            length: 150.0,
            rotation: 0.0,
            falloff: 0.5,
            intensity: 1.0,
            tint: [1.0, 1.0, 1.0],
            mode: GlowMode::Add,
            isolate: false,
            erode: 6.0,
            spread: 0.0,
            shift: 0.0,
        }
    }
}

impl GlintParams {
    /// Red's and blue's ray length relative to green's.
    pub fn chroma_scales(&self) -> (f32, f32) {
        let s = self.spread.clamp(-2.0, 2.0) * 0.25;
        ((1.0 + s).max(0.1), (1.0 - s).max(0.1))
    }

    /// The ray count the shader runs.
    pub fn ray_count(&self) -> usize {
        (self.rays.round().max(0.0) as usize).min(MAX_RAYS)
    }
}

/// Half-res highlight key: each texel is the mean of its 2x2 block, each
/// full-res pixel thresholded BEFORE the mean, so a lone hot pixel is not
/// averaged under the threshold by its dark neighbours.
pub const EXTRACT_WGSL: &str = r#"
struct GE { threshold: f32, knee: f32, _p0: f32, _p1: f32 }
@group(0) @binding(0) var<uniform> P: GE;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<rgba16float, write>;

fn highlights(c: vec3f) -> vec3f {
    // The Glow's soft knee: hue is kept, one scalar scales rgb.
    let br = max(c.r, max(c.g, c.b));
    let kn = max(P.knee, 1e-4);
    let soft = clamp(br - P.threshold + kn, 0.0, 2.0 * kn);
    let quad = soft * soft / (4.0 * kn);
    return c * max(max(quad, br - P.threshold), 0.0) / max(br, 1e-4);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(dst);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let sd = vec2i(textureDimensions(src)) - 1;
    let o = vec2i(gid.xy) * 2;
    var acc = vec3f(0.0);
    for (var j = 0; j < 2; j++) {
        for (var i = 0; i < 2; i++) {
            let c = textureLoad(src, min(o + vec2i(i, j), sd), 0).rgb;
            acc += highlights(max(c, vec3f(0.0)));
        }
    }
    textureStore(dst, gid.xy, vec4f(acc * 0.25, 1.0));
}
"#;

/// Isolate: one separable min or max pass over the half-res key, along
/// `dir`, `radius` texels each way. `op` 0 is the erode (min), 1 the dilate
/// (max), 2 the dilate that finishes the opening and takes it away from the
/// key (`orig`): what is left is only what the opening removed — highlights
/// narrower than the erode.
pub const MORPH_WGSL: &str = r#"
struct GM { dir: vec2f, radius: f32, op: f32 }
@group(0) @binding(0) var<uniform> P: GM;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var orig: texture_2d<f32>;
@group(0) @binding(3) var dst: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = vec2i(textureDimensions(dst));
    if (gid.x >= u32(dims.x) || gid.y >= u32(dims.y)) { return; }
    let p = vec2i(gid.xy);
    let r = i32(P.radius);
    let d = vec2i(P.dir);
    var v = textureLoad(src, p, 0).rgb;
    for (var i = 1; i <= r; i++) {
        let a = textureLoad(src, clamp(p + d * i, vec2i(0), dims - 1), 0).rgb;
        let b = textureLoad(src, clamp(p - d * i, vec2i(0), dims - 1), 0).rgb;
        if (P.op == 0.0) { v = min(v, min(a, b)); } else { v = max(v, max(a, b)); }
    }
    if (P.op == 2.0) {
        v = max(textureLoad(orig, p, 0).rgb - v, vec3f(0.0));
    }
    textureStore(dst, gid.xy, vec4f(v, 1.0));
}
"#;

/// The widest isolate, in half-res texels each way.
pub const MAX_ERODE: f32 = 64.0;

/// One link of a ray chain. Four taps (times `sub` in the seed) back along
/// the ray, each weighted by the exponential tail at its distance. The last
/// link of a ray adds the ray to the running sum of the rays before it.
pub const STREAK_WGSL: &str = r#"
struct GS {
    dir: vec2f, stride: f32, decay: f32,
    sub: f32, scale: f32, enabled: f32, last: f32,
    has_prev: f32, kr: f32, kb: f32, decay_r: f32,
    decay_b: f32, chroma: f32, keep_r: f32, keep_b: f32,
}
@group(0) @binding(0) var<uniform> P: GS;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var src: texture_2d<f32>;
@group(0) @binding(3) var prev: texture_2d<f32>;
@group(0) @binding(4) var dst: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(dst);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    // A ray past the count: its links do nothing, and its last link only
    // carries the sum on (the next ray reads it).
    if (P.enabled == 0.0 && P.last == 0.0) { return; }
    var acc = vec3f(0.0);
    if (P.enabled != 0.0) {
        let uv = (vec2f(gid.xy) + 0.5) / vec2f(dims);
        let step = P.dir / vec2f(dims);
        let sub = max(i32(P.sub), 1);
        for (var i = 0; i < 4; i++) {
            for (var j = 0; j < sub; j++) {
                let d = (f32(i) + f32(j) / f32(sub)) * P.stride;
                if (P.chroma == 0.0) {
                    let w = exp(-P.decay * d);
                    acc += textureSampleLevel(src, samp, uv - step * d, 0.0).rgb * w;
                } else {
                    // Each channel along its own ray: red reaches kr times
                    // as far, blue kb, each with its own fade. Composed
                    // over the chain that stretches the whole tail.
                    let r = textureSampleLevel(src, samp, uv - step * d * P.kr, 0.0).r;
                    let g = textureSampleLevel(src, samp, uv - step * d, 0.0).g;
                    let b = textureSampleLevel(src, samp, uv - step * d * P.kb, 0.0).b;
                    acc += vec3f(r * exp(-P.decay_r * d), g * exp(-P.decay * d), b * exp(-P.decay_b * d));
                }
            }
        }
        acc *= P.scale * vec3f(P.keep_r, 1.0, P.keep_b);
    }
    if (P.last != 0.0 && P.has_prev != 0.0) {
        acc += textureLoad(prev, gid.xy, 0).rgb;
    }
    textureStore(dst, gid.xy, vec4f(acc, 1.0));
}
"#;

impl NodeDef for GlintParams {
    const NAME: &'static str = "Glint";
    /// Image, then the mask.
    const INPUTS: usize = 2;
    const MASK_INPUT: Option<usize> = Some(1);
    /// Every param lives in the chain's own uniforms.
    const VEC4_SLOTS: usize = 0;

    /// Every param is a uniform, the ray count included; only isolate
    /// adds passes.
    fn topo_tag(&self) -> u64 {
        27 + ((self.isolate as u64) << 8)
    }

    fn pack(&self, _: &mut [f32]) {}

    fn emit(&self, e: &mut Emit) -> String {
        let v = e.stage_result();
        e.masked_stage(Self::MASK_INPUT, v)
    }
}

fn acc(i: usize) -> String {
    format!("acc{}", i % 2)
}

/// Where ray `r`'s last link writes: the sum of rays 0..=r.
fn ray_out(r: usize) -> String {
    acc(r)
}

impl StageDef for GlintParams {
    fn program(&self, _canvas: (u32, u32), _wired: Wired) -> StageProgram {
        let mut textures = vec![
            TexSpec::full("b"),
            TexSpec::level("hi", SHIFT),
            TexSpec::level("p0", SHIFT),
            TexSpec::level("p1", SHIFT),
            TexSpec::level("acc0", SHIFT),
            TexSpec::level("acc1", SHIFT),
        ];
        if self.isolate {
            textures.push(TexSpec::level("key", SHIFT));
        }
        let mut passes = vec![Pass {
            label: "glint-extract",
            wgsl: EXTRACT_WGSL,
            binds: vec![Bind::Uniform, Bind::Tex("a".into()), Bind::Storage("hi".into())],
            dispatch: Dispatch::Pixels("hi".into()),
            uniform_floats: 4,
        }];
        // Isolate: hi -min-> p0 -min-> p1 -max-> p0 -max,sub-> key. The
        // rays' own ping-pong reuses p0/p1 after.
        let key = if self.isolate {
            for (src, dst) in [("hi", "p0"), ("p0", "p1"), ("p1", "p0"), ("p0", "key")] {
                passes.push(Pass {
                    label: "glint-isolate",
                    wgsl: MORPH_WGSL,
                    binds: vec![
                        Bind::Uniform,
                        Bind::Tex(src.into()),
                        Bind::Tex("hi".into()),
                        Bind::Storage(dst.into()),
                    ],
                    dispatch: Dispatch::Pixels(dst.into()),
                    uniform_floats: 4,
                });
            }
            "key"
        } else {
            "hi"
        };
        for r in 0..MAX_RAYS {
            // key -> p0 -> p1 -> p0 -> sum. `prev` is only read by the last
            // link; the others bind `hi`, which they never touch.
            let links: [(&str, String); STEPS] = [
                (key, "p0".into()),
                ("p0", "p1".into()),
                ("p1", "p0".into()),
                ("p0", ray_out(r)),
            ];
            for (k, (src, dst)) in links.into_iter().enumerate() {
                let prev = if k + 1 == STEPS && r > 0 { ray_out(r - 1) } else { "hi".into() };
                passes.push(Pass {
                    label: "glint-ray",
                    wgsl: STREAK_WGSL,
                    binds: vec![
                        Bind::Uniform,
                        Bind::Sampler,
                        Bind::Tex(src.into()),
                        Bind::Tex(prev),
                        Bind::Storage(dst.clone()),
                    ],
                    dispatch: Dispatch::Pixels(dst),
                    uniform_floats: 16,
                });
            }
        }
        passes.push(Pass {
            label: "glint-comp",
            wgsl: COMP_WGSL,
            binds: vec![
                Bind::Uniform,
                Bind::Sampler,
                Bind::Tex("a".into()),
                Bind::Tex(ray_out(MAX_RAYS - 1)),
                Bind::Storage("b".into()),
            ],
            dispatch: Dispatch::Pixels("b".into()),
            uniform_floats: 8,
        });
        StageProgram { textures, passes, result: "b".into() }
    }

    fn uniforms(&self, pass: usize, cx: &StageCx) -> Vec<f32> {
        if pass == 0 {
            return vec![self.threshold, self.knee.max(0.0), 0.0, 0.0];
        }
        let iso = if self.isolate { 4 } else { 0 };
        if pass <= iso {
            // The erode is output pixels; the key is half res.
            let r = (self.erode * cx.proxy_scale / (1u32 << SHIFT) as f32)
                .round()
                .clamp(0.0, MAX_ERODE);
            let k = pass - 1;
            let dir = if k % 2 == 0 { [1.0, 0.0] } else { [0.0, 1.0] };
            let op = [0.0, 0.0, 1.0, 2.0][k];
            return vec![dir[0], dir[1], r, op];
        }
        let pass = pass - iso;
        let rays_end = 1 + MAX_RAYS * STEPS;
        if pass >= rays_end {
            let mode = match self.mode {
                GlowMode::Add => 0.0f32,
                GlowMode::Screen => 1.0,
                GlowMode::Only => 2.0,
            };
            return vec![self.intensity, mode, 0.0, 0.0, self.tint[0], self.tint[1], self.tint[2], 1.0];
        }
        let (r, k) = ((pass - 1) / STEPS, (pass - 1) % STEPS);
        let n = self.ray_count();
        let enabled = r < n;
        // The ray in half-res texels, proxy-scaled like every pixel length.
        let len = (self.length * cx.proxy_scale).max(0.0) / (1u32 << SHIFT) as f32;
        let base = len / REACH;
        let sub = (base.ceil() as u32).clamp(1, MAX_SUB);
        let stride = base * 4f32.powi(k as i32);
        // k = falloff * 8 nepers over the whole ray: 0 is flat to the tip.
        let decay = self.falloff.clamp(0.0, 1.0) * 8.0 / len.max(1e-3);
        let a = (self.rotation + r as f32 * 360.0 / n.max(1) as f32).to_radians();
        // y is down in the picture: a positive angle turns counter-clockwise.
        let dir = [a.cos(), -a.sin()];
        let seed = k == 0;
        // Only the seed is scaled, by the spacing of its positions: then a
        // one-texel highlight reads as its own value along the ray, whatever
        // the stride.
        let scale = if seed { base / sub as f32 } else { 1.0 };
        // Dispersion and hue drift. A stretched channel is divided by its
        // stretch at the seed, so it keeps its light, only spread longer.
        let (kr, kb) = self.chroma_scales();
        let extra = 4.0 / len.max(1e-3);
        let decay_r = decay + extra * (-self.shift).max(0.0);
        let decay_b = decay + extra * self.shift.max(0.0);
        let chroma = (self.spread != 0.0 || self.shift != 0.0) as u32 as f32;
        // Seed: the stretched channels' light, kept; later links: the
        // channel offsets only.
        let (kr_s, kb_s) = if seed { (kr, kb) } else { (1.0, 1.0) };
        vec![
            dir[0],
            dir[1],
            stride,
            decay,
            if seed { sub as f32 } else { 1.0 },
            scale,
            enabled as u32 as f32,
            (k + 1 == STEPS) as u32 as f32,
            (r > 0) as u32 as f32,
            kr,
            kb,
            decay_r,
            decay_b,
            chroma,
            1.0 / kr_s,
            1.0 / kb_s,
        ]
    }

    fn speed(&self) -> Speed {
        Speed::Slow
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cx() -> StageCx {
        StageCx { canvas: (512, 256), proxy_scale: 1.0, frame: 0.0, fps: 24.0, passes: 0 }
    }

    #[test]
    fn every_pass_gets_the_uniforms_it_declares() {
        let p = GlintParams::default();
        let prog = p.program((512, 256), Wired(&[true, false]));
        for (i, pass) in prog.passes.iter().enumerate() {
            assert_eq!(p.uniforms(i, &cx()).len(), pass.uniform_floats, "pass {i}");
        }
    }

    #[test]
    fn the_ray_count_never_changes_the_program() {
        let a = GlintParams { rays: 2.0, ..Default::default() };
        let b = GlintParams { rays: 7.0, ..Default::default() };
        let w = Wired(&[true, false]);
        assert_eq!(a.program((512, 256), w), b.program((512, 256), w));
        assert_eq!(a.topo_tag(), b.topo_tag());
    }

    #[test]
    fn the_chain_visits_each_position_once() {
        // Strides 1, 4, 16, 64 with taps 0..3: every n in 0..=255, once.
        let mut seen = vec![0u32; 256];
        for a in 0..4 {
            for b in 0..4 {
                for c in 0..4 {
                    for d in 0..4 {
                        seen[a + 4 * b + 16 * c + 64 * d] += 1;
                    }
                }
            }
        }
        assert!(seen.iter().all(|&n| n == 1));
        assert_eq!(REACH as usize, 255);
    }

    #[test]
    fn isolate_adds_its_passes_and_their_uniforms_fit() {
        let p = GlintParams { isolate: true, ..Default::default() };
        let prog = p.program((512, 256), Wired(&[true, false]));
        assert_eq!(prog.passes.len(), GlintParams::default().program((512, 256), Wired(&[true, false])).passes.len() + 4);
        for (i, pass) in prog.passes.iter().enumerate() {
            assert_eq!(p.uniforms(i, &cx()).len(), pass.uniform_floats, "pass {i}");
        }
        assert_ne!(p.topo_tag(), GlintParams::default().topo_tag());
        let q = GlintParams { erode: 20.0, ..p };
        assert_eq!(p.program((512, 256), Wired(&[true, false])), q.program((512, 256), Wired(&[true, false])));
    }

    #[test]
    fn a_long_ray_gets_sub_taps_and_a_short_one_none() {
        let long = GlintParams { length: 1500.0, ..Default::default() };
        let short = GlintParams { length: 60.0, ..Default::default() };
        assert!(long.uniforms(1, &cx())[4] > 1.0);
        assert_eq!(short.uniforms(1, &cx())[4], 1.0);
    }
}
