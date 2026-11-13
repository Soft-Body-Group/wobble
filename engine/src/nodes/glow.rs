//! Glow: a mip-pyramid bloom. Threshold the highlights, downsample through
//! a chain of halved levels, add each octave's weighted contribution back
//! on the way up, composite. `size` moves weight up a pyramid that is built
//! anyway, so a 500 px halo costs what a 30 px one does — see README.
//!
//! The stage with the most passes, and the one to copy for a chain whose
//! length depends on the canvas.

use crate::codegen::Emit;
use crate::cost::Speed;
use crate::nodes::stage::{Bind, Dispatch, Pass, StageCx, StageDef, StageProgram, TexSpec, Wired};
use crate::nodes::NodeDef;

/// How the glow is combined with the image it was made from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum GlowMode {
    /// Linear add — the physical one, and what a compounded-blur setup does.
    #[default]
    Add,
    /// Screen: 1-(1-a)(1-b). Rolls off instead of clipping.
    Screen,
    /// The glow alone, for hand-wiring a merge downstream.
    Only,
}

impl GlowMode {
    pub const ALL: [GlowMode; 3] = [GlowMode::Add, GlowMode::Screen, GlowMode::Only];
    pub fn name(self) -> &'static str {
        match self {
            GlowMode::Add => "add",
            GlowMode::Screen => "screen",
            GlowMode::Only => "glow only",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GlowParams {
    /// Halo radius in output pixels. Selects how far up the pyramid the
    /// weights reach; it is NOT a tap count, so it is very nearly free.
    pub size: f32,
    /// Weight ratio between one octave and the next. 0.5 = each octave half
    /// the last (tight core, quick falloff); 1.0 = every octave equal
    /// (a very wide, soft veil). This is the shape control.
    pub falloff: f32,
    /// Scene-linear level where the glow starts. The comp is scene-linear,
    /// so 1.0 means "highlights only".
    pub threshold: f32,
    /// Soft shoulder on the threshold, in stops-ish linear width. 0 = hard.
    pub knee: f32,
    /// Multiplier on the glow before it is combined.
    pub intensity: f32,
    /// Per-channel tint on the glow, linear.
    pub tint: [f32; 3],
    pub mode: GlowMode,
}

impl Default for GlowParams {
    fn default() -> Self {
        Self {
            size: 60.0,
            falloff: 0.7,
            threshold: 1.0,
            knee: 0.5,
            intensity: 1.0,
            tint: [1.0, 1.0, 1.0],
            mode: GlowMode::default(),
        }
    }
}

/// Run the chain all the way down to a single texel. The last octave is the
/// frame's DC — its average — and a veil that cannot reach it is not
/// frame-wide. A glow simply never weights the top octaves.
const MAX_LEVELS: usize = 12;

/// Octaves the pyramid runs for a canvas. Stops before a level gets too
/// small to filter meaningfully, and caps the pass count regardless of
/// resolution. Level i is (w, h) >> (i + 1).
pub fn levels(w: u32, h: u32) -> usize {
    let mut n = 0;
    while n < MAX_LEVELS && (w >> (n + 1)) >= 1 && (h >> (n + 1)) >= 1 {
        n += 1;
    }
    n.max(1)
}

/// Per-octave contribution weights, normalized to sum 1.
///
/// This is the whole "compounded blurs" feel, as numbers: octave i behaves
/// like a gaussian of radius ~2^(i+1) px, `falloff` sets how much weight
/// each successive (wider) octave keeps, and `size` fades octaves in as the
/// radius grows past them so the control is continuous rather than steppy.
pub fn weights(size: f32, falloff: f32, levels: usize) -> Vec<f32> {
    let reach = size.max(1.0).log2();
    let f = falloff.clamp(0.0, 1.0);
    let mut w: Vec<f32> = (0..levels)
        .map(|i| (reach - i as f32).clamp(0.0, 1.0) * f.powi(i as i32))
        .collect();
    let sum: f32 = w.iter().sum();
    if sum > 1e-6 {
        for x in &mut w {
            *x /= sum;
        }
    } else {
        w[0] = 1.0;
    }
    w
}

/// 13-tap downsample; the first level also keys the highlights.
pub const DOWN_WGSL: &str = r#"
struct GD { threshold: f32, knee: f32, prefilter: f32, _pad: f32 }
@group(0) @binding(0) var<uniform> P: GD;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var src: texture_2d<f32>;
@group(0) @binding(3) var dst: texture_storage_2d<rgba16float, write>;

fn highlights(c: vec3f) -> vec3f {
    // Soft-knee threshold on max-channel brightness: below the knee nothing
    // glows, across it the response eases in quadratically, above it the
    // excess passes linearly. Hue is preserved (one scalar scales rgb).
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
    let ts = 1.0 / vec2f(textureDimensions(src));
    let a = textureSampleLevel(src, samp, uv + ts * vec2f(-2.0,  2.0), 0.0).rgb;
    let b = textureSampleLevel(src, samp, uv + ts * vec2f( 0.0,  2.0), 0.0).rgb;
    let c = textureSampleLevel(src, samp, uv + ts * vec2f( 2.0,  2.0), 0.0).rgb;
    let d = textureSampleLevel(src, samp, uv + ts * vec2f(-2.0,  0.0), 0.0).rgb;
    let e = textureSampleLevel(src, samp, uv,                          0.0).rgb;
    let f = textureSampleLevel(src, samp, uv + ts * vec2f( 2.0,  0.0), 0.0).rgb;
    let g = textureSampleLevel(src, samp, uv + ts * vec2f(-2.0, -2.0), 0.0).rgb;
    let h = textureSampleLevel(src, samp, uv + ts * vec2f( 0.0, -2.0), 0.0).rgb;
    let i = textureSampleLevel(src, samp, uv + ts * vec2f( 2.0, -2.0), 0.0).rgb;
    let j = textureSampleLevel(src, samp, uv + ts * vec2f(-1.0,  1.0), 0.0).rgb;
    let k = textureSampleLevel(src, samp, uv + ts * vec2f( 1.0,  1.0), 0.0).rgb;
    let l = textureSampleLevel(src, samp, uv + ts * vec2f(-1.0, -1.0), 0.0).rgb;
    let m = textureSampleLevel(src, samp, uv + ts * vec2f( 1.0, -1.0), 0.0).rgb;
    var o = e * 0.125;
    o += (a + c + g + i) * 0.03125;
    o += (b + d + f + h) * 0.0625;
    o += (j + k + l + m) * 0.125;
    if (P.prefilter != 0.0) { o = highlights(o); }
    textureStore(dst, gid.xy, vec4f(max(o, vec3f(0.0)), 1.0));
}
"#;

/// Tent upsample of the level above, plus this level's own weighted octave.
pub const UP_WGSL: &str = r#"
struct GU { weight: f32, has_upper: f32, _p0: f32, _p1: f32 }
@group(0) @binding(0) var<uniform> P: GU;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var mip: texture_2d<f32>;
@group(0) @binding(3) var upper: texture_2d<f32>;
@group(0) @binding(4) var dst: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(dst);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let uv = (vec2f(gid.xy) + 0.5) / vec2f(dims);
    var acc = textureLoad(mip, gid.xy, 0).rgb * P.weight;
    if (P.has_upper != 0.0) {
        let ts = 1.0 / vec2f(textureDimensions(upper));
        var t = textureSampleLevel(upper, samp, uv, 0.0).rgb * 4.0;
        t += textureSampleLevel(upper, samp, uv + ts * vec2f(-1.0,  0.0), 0.0).rgb * 2.0;
        t += textureSampleLevel(upper, samp, uv + ts * vec2f( 1.0,  0.0), 0.0).rgb * 2.0;
        t += textureSampleLevel(upper, samp, uv + ts * vec2f( 0.0, -1.0), 0.0).rgb * 2.0;
        t += textureSampleLevel(upper, samp, uv + ts * vec2f( 0.0,  1.0), 0.0).rgb * 2.0;
        t += textureSampleLevel(upper, samp, uv + ts * vec2f(-1.0, -1.0), 0.0).rgb;
        t += textureSampleLevel(upper, samp, uv + ts * vec2f( 1.0, -1.0), 0.0).rgb;
        t += textureSampleLevel(upper, samp, uv + ts * vec2f(-1.0,  1.0), 0.0).rgb;
        t += textureSampleLevel(upper, samp, uv + ts * vec2f( 1.0,  1.0), 0.0).rgb;
        acc += t / 16.0;
    }
    textureStore(dst, gid.xy, vec4f(acc, 1.0));
}
"#;

/// The glow over the original: add, screen, or the glow alone.
pub const COMP_WGSL: &str = r#"
struct GC { intensity: f32, mode: f32, _p0: f32, _p1: f32, tint: vec4f }
@group(0) @binding(0) var<uniform> P: GC;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var orig: texture_2d<f32>;
@group(0) @binding(3) var glow: texture_2d<f32>;
@group(0) @binding(4) var dst: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(dst);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let base = textureLoad(orig, gid.xy, 0);
    let uv = (vec2f(gid.xy) + 0.5) / vec2f(dims);
    let g = max(textureSampleLevel(glow, samp, uv, 0.0).rgb, vec3f(0.0))
        * P.intensity * P.tint.rgb;
    var rgb = base.rgb + g;
    if (P.mode == 1.0) {
        // Screen, on the positive part only: negatives in a scene-linear
        // plate would invert the roll-off.
        let b = max(base.rgb, vec3f(0.0));
        rgb = base.rgb - b + (1.0 - (1.0 - b) * (1.0 - min(g, vec3f(1.0))));
    } else if (P.mode == 2.0) {
        rgb = g;
    }
    textureStore(dst, gid.xy, vec4f(rgb, base.a));
}
"#;

impl NodeDef for GlowParams {
    const NAME: &'static str = "Glow";
    /// Image, then the mask.
    const INPUTS: usize = 2;
    const MASK_INPUT: Option<usize> = Some(1);
    /// Every param lives in the pyramid's own uniforms.
    const VEC4_SLOTS: usize = 0;

    /// Constant: every glow param is a uniform; the chain's shape depends
    /// on the canvas alone.
    fn topo_tag(&self) -> u64 {
        22
    }

    fn pack(&self, _: &mut [f32]) {}

    fn emit(&self, e: &mut Emit) -> String {
        let v = e.stage_result();
        e.masked_stage(Self::MASK_INPUT, v)
    }
}

impl StageDef for GlowParams {
    fn program(&self, canvas: (u32, u32), _wired: Wired) -> StageProgram {
        let n = levels(canvas.0, canvas.1);
        let mip = |i: usize| format!("mip{i}");
        let up = |i: usize| format!("up{i}");
        let mut textures = vec![TexSpec::full("b")];
        for i in 0..n {
            textures.push(TexSpec::level(&mip(i), i as u32 + 1));
            textures.push(TexSpec::level(&up(i), i as u32 + 1));
        }
        let mut passes = Vec::with_capacity(2 * n + 1);
        for i in 0..n {
            let src = if i == 0 { "a".to_string() } else { mip(i - 1) };
            passes.push(Pass {
                label: "glow-down",
                wgsl: DOWN_WGSL,
                binds: vec![Bind::Uniform, Bind::Sampler, Bind::Tex(src), Bind::Storage(mip(i))],
                dispatch: Dispatch::Pixels(mip(i)),
                uniform_floats: 4,
            });
        }
        // Top octave down to the finest: each adds its own weighted level
        // to the tent-upsampled sum of everything above it. The top has
        // nothing above it; its own mip stands in for the unused binding
        // (the uniform gates it off).
        for i in (0..n).rev() {
            let upper = if i + 1 < n { up(i + 1) } else { mip(i) };
            passes.push(Pass {
                label: "glow-up",
                wgsl: UP_WGSL,
                binds: vec![
                    Bind::Uniform,
                    Bind::Sampler,
                    Bind::Tex(mip(i)),
                    Bind::Tex(upper),
                    Bind::Storage(up(i)),
                ],
                dispatch: Dispatch::Pixels(up(i)),
                uniform_floats: 4,
            });
        }
        passes.push(Pass {
            label: "glow-comp",
            wgsl: COMP_WGSL,
            binds: vec![
                Bind::Uniform,
                Bind::Sampler,
                Bind::Tex("a".into()),
                Bind::Tex(up(0)),
                Bind::Storage("b".into()),
            ],
            dispatch: Dispatch::Pixels("b".into()),
            uniform_floats: 8,
        });
        StageProgram { textures, passes, result: "b".into() }
    }

    fn uniforms(&self, pass: usize, cx: &StageCx) -> Vec<f32> {
        let n = levels(cx.canvas.0, cx.canvas.1);
        if pass < n {
            // Only the first downsample thresholds; after that the chain is
            // filtering an already-keyed image.
            let pre = (pass == 0) as u32 as f32;
            vec![self.threshold, self.knee.max(0.0), pre, 0.0]
        } else if pass < 2 * n {
            let i = 2 * n - 1 - pass;
            let w = weights(self.size, self.falloff, n);
            let has_upper = (i + 1 < n) as u32 as f32;
            vec![w[i], has_upper, 0.0, 0.0]
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
