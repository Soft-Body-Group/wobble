//! Blur: a separable gaussian, sigma in pixels, as two passes (H then V)
//! over one scratch texture. The first node that took its own pass, and
//! the simplest shape a stage can have — copy this one for a filter.

use crate::codegen::Emit;
use crate::cost::Speed;
use crate::nodes::stage::{Bind, Dispatch, Pass, StageCx, StageDef, StageProgram, TexSpec, Wired};
use crate::nodes::NodeDef;

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BlurParams {
    /// Gaussian sigma in output pixels.
    pub size: f32,
    /// Which pipe it sits in: wired a MATTE it filters the coverage and
    /// gives a matte (`NodeKind::follows_pipe`). Kept true to its input by
    /// the app (`EditorGraph::connect_out`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub matte: bool,
}

impl Default for BlurParams {
    fn default() -> Self {
        Self { size: 10.0, matte: false }
    }
}

/// The pass body, shared by the plain and the modulated module: each
/// defines `mod_at(p)` (1.0, or the modulate matte at `p`) and the bindings.
macro_rules! blur_body {
    () => {
        r#"
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = vec2i(textureDimensions(dst));
    if (gid.x >= u32(dims.x) || gid.y >= u32(dims.y)) { return; }
    let p = vec2i(gid.xy);
    let sigma = B.sigma * mod_at(p);
    // Under a twentieth of a pixel the kernel is the centre tap: skip the
    // walk, which is most of a modulated frame where the matte is black.
    if (sigma < 0.05) {
        textureStore(dst, gid.xy, px_out(px(textureLoad(src, p, 0))));
        return;
    }
    let r = min(i32(ceil(sigma * 3.0)), 128);
    var acc = PX(0.0);
    var wsum = 0.0;
    for (var i = -r; i <= r; i = i + 1) {
        let w = exp(-f32(i * i) / (2.0 * sigma * sigma));
        let q = clamp(p + vec2i(B.dir) * i, vec2i(0), dims - 1);
        acc += px(textureLoad(src, q, 0)) * w;
        wsum += w;
    }
    textureStore(dst, gid.xy, px_out(acc / wsum));
}
"#
    };
}

/// One pass, run twice: `dir` picks the axis.
pub const WGSL: &str = concat!(
    r#"
struct BlurParams { dir: vec2f, sigma: f32, _pad: f32 }
@group(0) @binding(0) var<uniform> B: BlurParams;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<rgba16float, write>;
fn mod_at(p: vec2i) -> f32 { return 1.0; }
"#,
    crate::nodes::stage::px_rgba!(),
    blur_body!()
);

/// The same pass with the MODULATE matte bound: sigma scales by its
/// coverage at the pixel. Separable, so each pass uses the sigma at its
/// OWN pixel — exact where the matte is flat, and where it has a hard edge
/// the vertical pass gathers rows the horizontal one blurred by their own
/// amounts (faint vertical streaks of the sharper side). A pyramid pick
/// would not streak, but costs a pyramid whether the matte varies or not.
pub const MOD_WGSL: &str = concat!(
    r#"
struct BlurParams { dir: vec2f, sigma: f32, _pad: f32 }
@group(0) @binding(0) var<uniform> B: BlurParams;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var modt: texture_2d<f32>;
fn mod_at(p: vec2i) -> f32 { return clamp(textureLoad(modt, p, 0).x, 0.0, 1.0); }
"#,
    crate::nodes::stage::px_rgba!(),
    blur_body!()
);

/// [`WGSL`] on ONE channel: a matte's blur.
pub const ONE_WGSL: &str = concat!(
    r#"
struct BlurParams { dir: vec2f, sigma: f32, _pad: f32 }
@group(0) @binding(0) var<uniform> B: BlurParams;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<r16float, write>;
fn mod_at(p: vec2i) -> f32 { return 1.0; }
"#,
    crate::nodes::stage::px_one!(),
    blur_body!()
);

/// [`MOD_WGSL`] on ONE channel.
pub const ONE_MOD_WGSL: &str = concat!(
    r#"
struct BlurParams { dir: vec2f, sigma: f32, _pad: f32 }
@group(0) @binding(0) var<uniform> B: BlurParams;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<r16float, write>;
@group(0) @binding(3) var modt: texture_2d<f32>;
fn mod_at(p: vec2i) -> f32 { return clamp(textureLoad(modt, p, 0).x, 0.0, 1.0); }
"#,
    crate::nodes::stage::px_one!(),
    blur_body!()
);

impl NodeDef for BlurParams {
    const NAME: &'static str = "Blur";
    /// Image, then the matte — and on a blur the matte MODULATES: half
    /// coverage is half the blur. A mix-back mask (half sharp, half fully
    /// blurred, a double exposure) is not offered; it is never what a blur
    /// wants, and the port that was it now does this.
    const INPUTS: usize = 2;
    const MASK_INPUT: Option<usize> = None;
    const MOD_INPUT: Option<usize> = Some(1);
    /// Sigma lives in the pass's own uniform, not the shared block.
    const VEC4_SLOTS: usize = 0;

    /// Constant: a Blur's shader never changes shape.
    fn topo_tag(&self) -> u64 {
        5
    }

    fn pack(&self, _: &mut [f32]) {}

    fn emit(&self, e: &mut Emit) -> String {
        e.stage_result()
    }
}

fn blur_pass(label: &'static str, wgsl: &'static str, src: &str, dst: &str, modulated: bool) -> Pass {
    let mut binds = vec![Bind::Uniform, Bind::Tex(src.into()), Bind::Storage(dst.into())];
    if modulated {
        binds.push(Bind::Tex("aux0".into()));
    }
    Pass { label, wgsl, binds, dispatch: Dispatch::Pixels("a".into()), uniform_floats: 4 }
}

impl StageDef for BlurParams {
    /// The modulate matte, only when wired: unwired, the Blur is exactly
    /// the two passes it always was, with no extra subtree to render.
    fn aux_ports(&self, wired: Wired) -> Vec<usize> {
        Self::MOD_INPUT.filter(|m| wired.has(*m)).into_iter().collect()
    }

    fn program(&self, _canvas: (u32, u32), wired: Wired) -> StageProgram {
        let modulated = Self::MOD_INPUT.is_some_and(|m| wired.has(m));
        let wgsl = if modulated { MOD_WGSL } else { WGSL };
        StageProgram {
            textures: vec![TexSpec::full("b")],
            // H into the scratch, V back into `a`, which is what downstream
            // samples.
            passes: vec![
                blur_pass("blur-h", wgsl, "a", "b", modulated),
                blur_pass("blur-v", wgsl, "b", "a", modulated),
            ],
            result: "a".into(),
        }
    }

    /// A matte: H into one single-channel scratch, V into a second (`a`
    /// is the four-channel subtree render, which a one-channel pass cannot
    /// write back into).
    fn program_one_channel(&self, _canvas: (u32, u32), wired: Wired) -> Option<StageProgram> {
        let modulated = Self::MOD_INPUT.is_some_and(|m| wired.has(m));
        let wgsl = if modulated { ONE_MOD_WGSL } else { ONE_WGSL };
        Some(
            StageProgram {
                textures: vec![TexSpec::full("b"), TexSpec::full("c")],
                passes: vec![
                    blur_pass("blur-h", wgsl, "a", "b", modulated),
                    blur_pass("blur-v", wgsl, "b", "c", modulated),
                ],
                result: "c".into(),
            }
            .one_channel(),
        )
    }

    fn uniforms(&self, pass: usize, cx: &StageCx) -> Vec<f32> {
        // Sigma is pixels: scale with the proxy.
        let sigma = self.size.max(0.0) * cx.proxy_scale;
        match pass {
            0 => vec![1.0, 0.0, sigma, 0.0],
            _ => vec![0.0, 1.0, sigma, 0.0],
        }
    }

    fn speed(&self) -> Speed {
        Speed::Slow
    }
}
