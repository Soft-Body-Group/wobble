//! Godrays: volumetric light shafts streaming from a source point — the
//! sun behind a treeline, a lamp through fog.
//!
//! The classic screen-space scattering, as a stage:
//!
//! 1. **extract** — at half resolution, the picture's highlights (the
//!    Glow's soft-knee threshold) times the SOURCE matte on port 1 when one
//!    is wired: the matte says what casts rays (a sky key, a window), so
//!    whatever covers it — branches, a silhouette — cuts dark shafts.
//! 2. **three radial passes** — each pixel averages `samples` taps (n) on
//!    the line towards the source. The passes cover the ray at spans of
//!    `length / n²`, `length / n` and `length`, so together they gather
//!    `n³` evenly spaced points for the cost of `3n` taps, with none of
//!    the banding one long pass leaves. `dither` slides each pixel's taps
//!    by a per-pixel fraction of a step (interleaved gradient noise, a
//!    different offset per pass), which trades the stair-steps a long ray
//!    leaves at low `samples` for fine grain the passes average away. Each tap is weighted
//!    `exp(-falloff * t)` (t = 0 at the pixel, 1 at the ray's far end);
//!    an exponential is multiplicative, so the three passes' weights
//!    compose into (very nearly — a later pass starts from a point already
//!    nearer the source) that weight on the whole ray.
//! 3. **composite** — the Glow's own comp pass: add, screen, or rays only,
//!    with intensity and tint.
//!
//! The source point is NORMALISED to the canvas, so proxy and format
//! changes need nothing; nothing here is in pixels. Every param is a
//! uniform; only whether the source matte is wired changes the chain.

use crate::codegen::Emit;
use crate::cost::Speed;
use crate::nodes::glow::{GlowMode, COMP_WGSL};
use crate::nodes::stage::{Bind, Dispatch, Pass, StageCx, StageDef, StageProgram, TexSpec, Wired};
use crate::nodes::NodeDef;

/// Taps per radial pass by default. Three passes gather `n³` points.
pub const TAPS: u32 = 16;
/// The range `samples` is clamped to: the loop bound is a uniform, so any
/// count costs no recompile, only time (linear in it).
pub const MIN_TAPS: u32 = 4;
pub const MAX_TAPS: u32 = 64;
/// Radial passes in the chain.
pub const RADIAL_PASSES: usize = 3;

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct GodraysParams {
    /// The light's position, normalised to the canvas (0,0 top left). May
    /// sit off the frame: a sun just out of shot still throws shafts.
    pub center: [f32; 2],
    /// How far each ray reaches back towards the source, as a fraction of
    /// the pixel's distance to it: 1 runs all the way to the source.
    pub length: f32,
    /// Exponential falloff along the ray: 0 weighs every point equally,
    /// higher keeps the shafts close to what casts them.
    pub falloff: f32,
    /// Scene-linear level where a pixel starts to emit.
    pub threshold: f32,
    /// Soft shoulder on the threshold. 0 = hard.
    pub knee: f32,
    /// Multiplier on the rays before they are combined.
    pub intensity: f32,
    /// Per-channel tint on the rays, linear.
    pub tint: [f32; 3],
    pub mode: GlowMode,
    /// Taps per radial pass (the quality): the ray is `samples³` points.
    pub samples: u32,
    /// 0..1: how far each pixel's taps are slid by a per-pixel random
    /// fraction of a step. 0 = the old fixed lattice (stair-steps at low
    /// samples), 1 = a full step (grain instead).
    pub dither: f32,
}

impl Default for GodraysParams {
    fn default() -> Self {
        Self {
            center: [0.5, 0.3],
            length: 0.6,
            falloff: 1.0,
            threshold: 0.8,
            knee: 0.5,
            intensity: 1.5,
            tint: [1.0, 1.0, 1.0],
            mode: GlowMode::Add,
            samples: TAPS,
            dither: 1.0,
        }
    }
}

impl GodraysParams {
    /// The span (as a fraction of the distance to the source) radial pass
    /// `k` covers: the shortest first.
    pub fn span(&self, k: usize) -> f32 {
        let l = self.length.clamp(0.0, 1.0);
        l / (self.taps() as f32).powi((RADIAL_PASSES - 1 - k) as i32)
    }

    /// Taps per radial pass, clamped to [`MIN_TAPS`]..=[`MAX_TAPS`].
    pub fn taps(&self) -> u32 {
        self.samples.clamp(MIN_TAPS, MAX_TAPS)
    }

    /// The CPU mirror of the whole gather: the
    /// positions `t` (0..1 of the ray) the `n³` gathered points sit at,
    /// to first order in the spans (see the module docs).
    pub fn ray_positions(&self) -> Vec<f32> {
        let n = self.taps() as usize;
        let l = self.length.clamp(0.0, 1.0).max(1e-6);
        let mut out = Vec::with_capacity(n * n * n);
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let f = self.span(0) * k as f32 / n as f32
                        + self.span(1) * j as f32 / n as f32
                        + self.span(2) * i as f32 / n as f32;
                    out.push(f / l);
                }
            }
        }
        out
    }
}

macro_rules! extract_body {
    () => {
        r#"
fn highlights(c: vec3f) -> vec3f {
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
    let uv = (vec2f(gid.xy) + 0.5) / vec2f(dims);
    let ts = 0.5 / vec2f(textureDimensions(src));
    // A 2x2 box at the half-res texel: the four full-res pixels under it.
    var c = textureSampleLevel(src, samp, uv + vec2f(-ts.x, -ts.y), 0.0).rgb;
    c += textureSampleLevel(src, samp, uv + vec2f(ts.x, -ts.y), 0.0).rgb;
    c += textureSampleLevel(src, samp, uv + vec2f(-ts.x, ts.y), 0.0).rgb;
    c += textureSampleLevel(src, samp, uv + vec2f(ts.x, ts.y), 0.0).rgb;
    let e = highlights(max(c * 0.25, vec3f(0.0))) * source(uv);
    textureStore(dst, gid.xy, vec4f(e, 1.0));
}
"#
    };
}

/// Highlights at half resolution, every pixel a possible source.
pub const EXTRACT_WGSL: &str = concat!(
    r#"
struct GE { threshold: f32, knee: f32, _p0: f32, _p1: f32 }
@group(0) @binding(0) var<uniform> P: GE;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var src: texture_2d<f32>;
@group(0) @binding(3) var dst: texture_storage_2d<rgba16float, write>;
fn source(uv: vec2f) -> f32 { return 1.0; }
"#,
    extract_body!()
);

/// [`EXTRACT_WGSL`] with the source matte: only what it covers emits.
pub const EXTRACT_MATTE_WGSL: &str = concat!(
    r#"
struct GE { threshold: f32, knee: f32, _p0: f32, _p1: f32 }
@group(0) @binding(0) var<uniform> P: GE;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var src: texture_2d<f32>;
@group(0) @binding(3) var dst: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var matte: texture_2d<f32>;
fn source(uv: vec2f) -> f32 {
    return clamp(textureSampleLevel(matte, samp, uv, 0.0).x, 0.0, 1.0);
}
"#,
    extract_body!()
);

/// One radial pass: `taps` samples from the pixel towards the source over
/// `span` of the distance, weighted by `exp(-k * f)` at fraction `f` of the
/// span, normalised. `jitter` slides the taps by a per-pixel fraction of a
/// step; `seed` gives each pass its own noise.
/// Taps that leave the frame read black (the clamp sampler would smear the
/// edge into a streak).
pub const RADIAL_WGSL: &str = r#"
struct GR { center: vec2f, span: f32, k: f32, taps: f32, jitter: f32, seed: f32, _p: f32 }
@group(0) @binding(0) var<uniform> P: GR;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var src: texture_2d<f32>;
@group(0) @binding(3) var dst: texture_storage_2d<rgba16float, write>;

// Interleaved gradient noise (Jimenez 2014): 0..1, blue-ish, no texture.
fn ign(p: vec2f) -> f32 {
    return fract(52.9829189 * fract(dot(p, vec2f(0.06711056, 0.00583715))));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(dst);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let uv = (vec2f(gid.xy) + 0.5) / vec2f(dims);
    let d = (P.center - uv) * P.span;
    let n = u32(P.taps);
    let o = P.jitter * ign(vec2f(gid.xy) + P.seed * vec2f(5.588238, 3.1179));
    var acc = vec3f(0.0);
    var wsum = 0.0;
    for (var i = 0u; i < n; i = i + 1u) {
        let f = (f32(i) + o) / f32(n);
        let w = exp(-P.k * f);
        let q = uv + d * f;
        if (all(q >= vec2f(0.0)) && all(q <= vec2f(1.0))) {
            acc += textureSampleLevel(src, samp, q, 0.0).rgb * w;
        }
        wsum += w;
    }
    textureStore(dst, gid.xy, vec4f(acc / max(wsum, 1e-6), 1.0));
}
"#;

impl NodeDef for GodraysParams {
    const NAME: &'static str = "Godrays";
    /// The picture, the source matte (what casts rays), the mask.
    const INPUTS: usize = 3;
    const MASK_INPUT: Option<usize> = Some(2);
    const VEC4_SLOTS: usize = 0;

    /// Constant: every param is a uniform. Whether the source matte is
    /// wired is topology already (the wiring is in the hash).
    fn topo_tag(&self) -> u64 {
        0x60D5
    }

    fn pack(&self, _: &mut [f32]) {}

    fn emit(&self, e: &mut Emit) -> String {
        let v = e.stage_result();
        e.masked_stage(Self::MASK_INPUT, v)
    }
}

/// The source matte's port.
pub const SOURCE_INPUT: usize = 1;

impl StageDef for GodraysParams {
    /// The source matte, only when wired: unwired it costs nothing.
    fn aux_ports(&self, wired: Wired) -> Vec<usize> {
        if wired.has(SOURCE_INPUT) {
            vec![SOURCE_INPUT]
        } else {
            Vec::new()
        }
    }

    fn program(&self, _canvas: (u32, u32), wired: Wired) -> StageProgram {
        let matte = wired.has(SOURCE_INPUT);
        let mut ebinds =
            vec![Bind::Uniform, Bind::Sampler, Bind::Tex("a".into()), Bind::Storage("e".into())];
        if matte {
            ebinds.push(Bind::Tex("aux0".into()));
        }
        let mut passes = vec![Pass {
            label: "godrays-extract",
            wgsl: if matte { EXTRACT_MATTE_WGSL } else { EXTRACT_WGSL },
            binds: ebinds,
            dispatch: Dispatch::Pixels("e".into()),
            uniform_floats: 4,
        }];
        // e -> r -> e -> r: the rays end in r.
        for k in 0..RADIAL_PASSES {
            let (src, dst) = if k % 2 == 0 { ("e", "r") } else { ("r", "e") };
            passes.push(Pass {
                label: "godrays-radial",
                wgsl: RADIAL_WGSL,
                binds: vec![
                    Bind::Uniform,
                    Bind::Sampler,
                    Bind::Tex(src.into()),
                    Bind::Storage(dst.into()),
                ],
                dispatch: Dispatch::Pixels(dst.into()),
                uniform_floats: 8,
            });
        }
        let rays = if RADIAL_PASSES % 2 == 1 { "r" } else { "e" };
        passes.push(Pass {
            label: "godrays-comp",
            wgsl: COMP_WGSL,
            binds: vec![
                Bind::Uniform,
                Bind::Sampler,
                Bind::Tex("a".into()),
                Bind::Tex(rays.into()),
                Bind::Storage("b".into()),
            ],
            dispatch: Dispatch::Pixels("b".into()),
            uniform_floats: 8,
        });
        StageProgram {
            textures: vec![TexSpec::level("e", 1), TexSpec::level("r", 1), TexSpec::full("b")],
            passes,
            result: "b".into(),
        }
    }

    fn uniforms(&self, pass: usize, _cx: &StageCx) -> Vec<f32> {
        if pass == 0 {
            vec![self.threshold, self.knee.max(0.0), 0.0, 0.0]
        } else if pass <= RADIAL_PASSES {
            let k = pass - 1;
            let l = self.length.clamp(0.0, 1.0).max(1e-6);
            let span = self.span(k);
            // exp(-falloff * t) over the whole ray, with t this pass's
            // fraction of it: span / length per unit of the pass.
            vec![
                self.center[0],
                self.center[1],
                span,
                self.falloff.max(0.0) * span / l,
                self.taps() as f32,
                self.dither.clamp(0.0, 1.0),
                k as f32 + 1.0,
                0.0,
            ]
        } else {
            let mode = match self.mode {
                GlowMode::Add => 0.0f32,
                GlowMode::Screen => 1.0,
                GlowMode::Only => 2.0,
            };
            vec![self.intensity, mode, 0.0, 0.0, self.tint[0], self.tint[1], self.tint[2], 1.0]
        }
    }

    fn speed(&self) -> Speed {
        Speed::Slow
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn three_passes_gather_the_whole_ray_evenly() {
        let p = GodraysParams { length: 0.8, ..Default::default() };
        let mut t = p.ray_positions();
        t.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = t.len();
        assert_eq!(n, (TAPS * TAPS * TAPS) as usize);
        // Any quality gathers its own n³ just as evenly.
        let q = GodraysParams { samples: 6, ..p };
        let mut t6 = q.ray_positions();
        t6.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(t6.len(), 216);
        for (i, x) in t6.iter().enumerate() {
            assert!((x - i as f32 / 216.0).abs() < 1e-4, "tap {i} at {x}");
        }
        // Evenly spaced from 0 up to (not including) the far end.
        for (i, x) in t.iter().enumerate() {
            assert!((x - i as f32 / n as f32).abs() < 1e-4, "tap {i} at {x}");
        }
    }

    #[test]
    fn the_falloff_composes_across_passes() {
        let p = GodraysParams { length: 0.5, falloff: 3.0, ..Default::default() };
        let cx = StageCx { canvas: (64, 64), proxy_scale: 1.0, frame: 0.0, fps: 24.0, passes: 5 };
        // Per-pass k at its unit step must equal falloff * (span / length).
        let ks: Vec<f32> = (1..=RADIAL_PASSES).map(|i| p.uniforms(i, &cx)[3]).collect();
        let total: f32 = ks.iter().sum();
        // The three spans sum to 1 + 1/16 + 1/256 of the length.
        assert!((total - 3.0 * (1.0 + 1.0 / 16.0 + 1.0 / 256.0)).abs() < 1e-4);
    }

    #[test]
    fn the_source_matte_costs_a_binding_only_when_wired() {
        let p = GodraysParams::default();
        assert!(p.aux_ports(Wired(&[true, false, false])).is_empty());
        assert_eq!(p.aux_ports(Wired(&[true, true, false])), vec![1]);
        let plain = p.program((64, 64), Wired(&[true, false, false]));
        assert_eq!(plain.passes[0].binds.len(), 4);
        let m = p.program((64, 64), Wired(&[true, true, false]));
        assert!(m.passes[0].binds.contains(&Bind::Tex("aux0".into())));
        assert_eq!(m.passes.len(), 2 + RADIAL_PASSES);
    }
}
