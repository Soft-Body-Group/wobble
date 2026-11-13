//! Dilate: sub-pixel matte grow / shrink, one gather pass. A matte
//! operator end to end — its one input IS a matte, so it has no mask port.

use crate::codegen::Emit;
use crate::cost::Speed;
use crate::nodes::stage::{Bind, Dispatch, Pass, StageCx, StageDef, StageProgram, TexSpec, Wired};
use crate::nodes::NodeDef;

/// Sub-pixel matte dilate / erode.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DilateParams {
    /// Pixels to grow by; negative shrinks. The pass clamps the magnitude
    /// to `DILATE_MAX` — the gather is O(r^2) inside the whole part.
    pub amount: f32,
}

/// How far a Dilate can reach. Past this a disc gather is no longer a
/// sensible way to spend a frame.
pub const DILATE_MAX: f32 = 16.0;

/// The pass body, shared by the plain and the modulated module: each
/// defines `mod_at(p)` (1.0, or the modulate matte at `p`) and the bindings.
macro_rules! dilate_body {
    () => {
        r#"
// Grayscale morphology with a FLAT disc: dilation is the max over the disc,
// erosion the min. On an anti-aliased matte (coverage ramping ~1 per pixel
// across the edge) a flat disc of radius r shifts the whole ramp by exactly
// r — including fractions of a pixel, which is the whole point. The whole
// part of r is covered by integer taps; the fraction rides on a RING of
// bilinear taps at exactly radius r, because max is monotone in radius and
// the tap that decides the answer is the one along the gradient.
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = vec2i(textureDimensions(dst));
    if (gid.x >= u32(dims.x) || gid.y >= u32(dims.y)) { return; }
    let p = vec2i(gid.xy);
    var acc = px(textureLoad(src, p, 0));
    let r = min(abs(M.amount) * mod_at(p), 16.0);
    if (r < 1e-4) {
        textureStore(dst, gid.xy, px_out(acc));
        return;
    }
    let grow = M.amount > 0.0;
    let whole = i32(floor(r));
    // Integer disc for the whole part.
    for (var dy = -whole; dy <= whole; dy++) {
        for (var dx = -whole; dx <= whole; dx++) {
            if (dx * dx + dy * dy > whole * whole) { continue; }
            let q = clamp(p + vec2i(dx, dy), vec2i(0), dims - 1);
            let s = px(textureLoad(src, q, 0));
            acc = select(min(acc, s), max(acc, s), grow);
        }
    }
    // Ring at exactly r: enough samples that the arc between them is under
    // a pixel, which is what keeps a 0.3 px move from turning into a 0 or
    // a 1. Bilinear, so the tap really does sit at a fractional offset.
    let n = clamp(i32(ceil(6.2831853 * r)), 8, 128);
    let base = (vec2f(p) + vec2f(0.5)) / vec2f(dims);
    let step = vec2f(1.0) / vec2f(dims);
    for (var i = 0; i < n; i++) {
        let a = 6.2831853 * f32(i) / f32(n);
        let uv = base + vec2f(cos(a), sin(a)) * r * step;
        let s = px(textureSampleLevel(src, samp, clamp(uv, vec2f(0.0), vec2f(1.0)), 0.0));
        acc = select(min(acc, s), max(acc, s), grow);
    }
    textureStore(dst, gid.xy, px_out(acc));
}
"#
    };
}

pub const WGSL: &str = concat!(
    r#"
struct MP { amount: f32, _p0: f32, _p1: f32, _p2: f32 }
@group(0) @binding(0) var<uniform> M: MP;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var src: texture_2d<f32>;
@group(0) @binding(3) var dst: texture_storage_2d<rgba16float, write>;
fn mod_at(p: vec2i) -> f32 { return 1.0; }
"#,
    crate::nodes::stage::px_rgba!(),
    dilate_body!()
);

/// The same pass with the MODULATE matte bound: the radius scales by its
/// coverage at the pixel. A gather, so this is exact — each pixel's own
/// disc — and it only ever gets CHEAPER than the plain pass.
pub const MOD_WGSL: &str = concat!(
    r#"
struct MP { amount: f32, _p0: f32, _p1: f32, _p2: f32 }
@group(0) @binding(0) var<uniform> M: MP;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var src: texture_2d<f32>;
@group(0) @binding(3) var dst: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var modt: texture_2d<f32>;
fn mod_at(p: vec2i) -> f32 { return clamp(textureLoad(modt, p, 0).x, 0.0, 1.0); }
"#,
    crate::nodes::stage::px_rgba!(),
    dilate_body!()
);

/// [`WGSL`] on ONE channel: a Dilate's matte is one number per pixel.
pub const ONE_WGSL: &str = concat!(
    r#"
struct MP { amount: f32, _p0: f32, _p1: f32, _p2: f32 }
@group(0) @binding(0) var<uniform> M: MP;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var src: texture_2d<f32>;
@group(0) @binding(3) var dst: texture_storage_2d<r16float, write>;
fn mod_at(p: vec2i) -> f32 { return 1.0; }
"#,
    crate::nodes::stage::px_one!(),
    dilate_body!()
);

/// [`MOD_WGSL`] on ONE channel.
pub const ONE_MOD_WGSL: &str = concat!(
    r#"
struct MP { amount: f32, _p0: f32, _p1: f32, _p2: f32 }
@group(0) @binding(0) var<uniform> M: MP;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var src: texture_2d<f32>;
@group(0) @binding(3) var dst: texture_storage_2d<r16float, write>;
@group(0) @binding(4) var modt: texture_2d<f32>;
fn mod_at(p: vec2i) -> f32 { return clamp(textureLoad(modt, p, 0).x, 0.0, 1.0); }
"#,
    crate::nodes::stage::px_one!(),
    dilate_body!()
);

impl NodeDef for DilateParams {
    const NAME: &'static str = "Dilate";
    /// The matte, then the modulate matte.
    const INPUTS: usize = 2;
    const MASK_INPUT: Option<usize> = None;
    const MOD_INPUT: Option<usize> = Some(1);
    /// The amount rides in the pass's own uniform.
    const VEC4_SLOTS: usize = 0;

    /// Constant: dragging the amount never recompiles, exactly like a
    /// Blur's sigma.
    fn topo_tag(&self) -> u64 {
        30
    }

    fn pack(&self, _: &mut [f32]) {}

    fn emit(&self, e: &mut Emit) -> String {
        // A matte in all four channels, whichever chain ran: the one-channel
        // scratch reads back as (c, 0, 0, 1).
        let v = e.stage_result();
        let m = e.fresh();
        e.line(format_args!("let {m} = vec4f({v}.x);"));
        m
    }
}

impl DilateParams {
    fn chain(&self, modulated: bool, wgsl: &'static str) -> StageProgram {
        let mut binds =
            vec![Bind::Uniform, Bind::Sampler, Bind::Tex("a".into()), Bind::Storage("b".into())];
        if modulated {
            binds.push(Bind::Tex("aux0".into()));
        }
        StageProgram {
            textures: vec![TexSpec::full("b")],
            passes: vec![Pass {
                label: "morph",
                wgsl,
                binds,
                dispatch: Dispatch::Pixels("a".into()),
                uniform_floats: 4,
            }],
            result: "b".into(),
        }
    }
}

impl StageDef for DilateParams {
    /// The modulate matte, only when wired.
    fn aux_ports(&self, wired: Wired) -> Vec<usize> {
        Self::MOD_INPUT.filter(|m| wired.has(*m)).into_iter().collect()
    }

    fn program(&self, _canvas: (u32, u32), wired: Wired) -> StageProgram {
        let modulated = Self::MOD_INPUT.is_some_and(|m| wired.has(m));
        self.chain(modulated, if modulated { MOD_WGSL } else { WGSL })
    }

    fn program_one_channel(&self, _canvas: (u32, u32), wired: Wired) -> Option<StageProgram> {
        let modulated = Self::MOD_INPUT.is_some_and(|m| wired.has(m));
        Some(self.chain(modulated, if modulated { ONE_MOD_WGSL } else { ONE_WGSL }).one_channel())
    }

    fn uniforms(&self, _pass: usize, cx: &StageCx) -> Vec<f32> {
        // Signed, clamped to the reach the gather can afford.
        vec![(self.amount * cx.proxy_scale).clamp(-DILATE_MAX, DILATE_MAX), 0.0, 0.0, 0.0]
    }

    fn speed(&self) -> Speed {
        Speed::Slow
    }
}
