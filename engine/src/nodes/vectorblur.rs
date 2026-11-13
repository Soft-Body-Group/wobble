//! VectorBlur: motion blur along per-pixel motion vectors — a Flow node's,
//! or any picture that carries them the same way (`rg` forward, `ba`
//! backward, canvas pixels, y down; a render's motion pass shuffled into
//! that shape works too).
//!
//! A gather, in one pass: every output pixel averages the picture at
//! `samples` points along ITS OWN motion across the shutter — the forward
//! vector for the half after the frame, the backward one for the half
//! before it. That is the cheap, standard answer; it does not invent the
//! background a moving edge uncovered (nothing can, from one picture), so
//! an object's blur bleeds over a still background a little less than a
//! real shutter's would.
//!
//! Unlike MotionBlur — which re-renders the upstream at several moments
//! and so only smears what is ANIMATED in the comp — this smears what
//! moved IN THE PLATE: a car in a shot, a handheld camera.
//!
//! A stage (it samples off its own pixel): one pass, cheap next to the
//! MotionBlur that re-renders everything, red in the graph like any pass.
//! The matte on port 2 MODULATES the shutter, as a Blur's does its size.

use crate::codegen::Emit;
use crate::cost::Speed;
use crate::nodes::stage::{Bind, Dispatch, Pass, StageCx, StageDef, StageProgram, TexSpec, Wired};
use crate::nodes::NodeDef;

/// Ceiling on samples along a vector.
pub const MAX_SAMPLES: u32 = 64;

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct VectorBlurParams {
    /// Shutter open time as a fraction of a frame: 0.5 is a 180 degree
    /// shutter, 1.0 smears a whole frame's motion. Past 1 it exaggerates.
    pub amount: f32,
    /// Where the shutter sits about the frame: 0 centred, +0.5 opening on
    /// the frame (all forward motion), -0.5 closing on it.
    pub offset: f32,
    /// Points sampled along each vector. A uniform: dragging it never
    /// recompiles.
    pub samples: u32,
    /// Which pipe it sits in: wired a MATTE it filters the coverage and
    /// gives a matte (`NodeKind::follows_pipe`). Kept true to its input by
    /// the app (`EditorGraph::connect_out`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub matte: bool,
}

impl Default for VectorBlurParams {
    fn default() -> Self {
        Self { amount: 0.5, offset: 0.0, samples: 16, matte: false }
    }
}

impl VectorBlurParams {
    /// The shutter times sampled, in frames about the frame — the CPU
    /// mirror of the shader's loop.
    pub fn times(&self, modulate: f32) -> Vec<f32> {
        let n = self.samples.clamp(1, MAX_SAMPLES);
        let a = self.amount.max(0.0) * modulate.clamp(0.0, 1.0);
        (0..n)
            .map(|i| self.offset * modulate.clamp(0.0, 1.0) + a * ((i as f32 + 0.5) / n as f32 - 0.5))
            .collect()
    }

    /// Where the gather looks at shutter time `s`, from pixel `p` with
    /// forward vector `f` and backward vector `b`.
    pub fn at(p: [f32; 2], f: [f32; 2], b: [f32; 2], s: f32) -> [f32; 2] {
        if s >= 0.0 {
            [p[0] + f[0] * s, p[1] + f[1] * s]
        } else {
            [p[0] - b[0] * s, p[1] - b[1] * s]
        }
    }
}

macro_rules! vblur_body {
    () => {
        r#"
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = vec2i(textureDimensions(dst));
    if (gid.x >= u32(dims.x) || gid.y >= u32(dims.y)) { return; }
    let p = vec2i(gid.xy);
    let m = mod_at(p);
    let v = textureLoad(vecs, p, 0);
    let a = U.amount * m;
    let n = max(u32(U.samples), 1u);
    // No motion, or no shutter: the centre tap.
    if (a <= 0.0 || dot(v, v) < 1e-8) {
        textureStore(dst, gid.xy, px_out(px(textureLoad(src, p, 0))));
        return;
    }
    let fdims = vec2f(dims);
    let c = vec2f(p) + vec2f(0.5);
    var acc = PX(0.0);
    for (var i = 0u; i < n; i = i + 1u) {
        let s = U.offset * m + a * ((f32(i) + 0.5) / f32(n) - 0.5);
        let d = select(-v.zw * s, v.xy * s, s >= 0.0);
        acc += px(textureSampleLevel(src, samp, (c + d) / fdims, 0.0));
    }
    textureStore(dst, gid.xy, px_out(acc / f32(n)));
}
"#
    };
}

pub const WGSL: &str = concat!(
    r#"
struct VbParams { amount: f32, offset: f32, samples: f32, _pad: f32 }
@group(0) @binding(0) var<uniform> U: VbParams;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var src: texture_2d<f32>;
@group(0) @binding(3) var vecs: texture_2d<f32>;
@group(0) @binding(4) var dst: texture_storage_2d<rgba16float, write>;
fn mod_at(p: vec2i) -> f32 { return 1.0; }
"#,
    crate::nodes::stage::px_rgba!(),
    vblur_body!()
);

pub const MOD_WGSL: &str = concat!(
    r#"
struct VbParams { amount: f32, offset: f32, samples: f32, _pad: f32 }
@group(0) @binding(0) var<uniform> U: VbParams;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var src: texture_2d<f32>;
@group(0) @binding(3) var vecs: texture_2d<f32>;
@group(0) @binding(4) var dst: texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var modt: texture_2d<f32>;
fn mod_at(p: vec2i) -> f32 { return clamp(textureLoad(modt, p, 0).x, 0.0, 1.0); }
"#,
    crate::nodes::stage::px_rgba!(),
    vblur_body!()
);

/// [`WGSL`] on ONE channel: a matte smeared along the vectors.
pub const ONE_WGSL: &str = concat!(
    r#"
struct VbParams { amount: f32, offset: f32, samples: f32, _pad: f32 }
@group(0) @binding(0) var<uniform> U: VbParams;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var src: texture_2d<f32>;
@group(0) @binding(3) var vecs: texture_2d<f32>;
@group(0) @binding(4) var dst: texture_storage_2d<r16float, write>;
fn mod_at(p: vec2i) -> f32 { return 1.0; }
"#,
    crate::nodes::stage::px_one!(),
    vblur_body!()
);

/// [`MOD_WGSL`] on ONE channel.
pub const ONE_MOD_WGSL: &str = concat!(
    r#"
struct VbParams { amount: f32, offset: f32, samples: f32, _pad: f32 }
@group(0) @binding(0) var<uniform> U: VbParams;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var src: texture_2d<f32>;
@group(0) @binding(3) var vecs: texture_2d<f32>;
@group(0) @binding(4) var dst: texture_storage_2d<r16float, write>;
@group(0) @binding(5) var modt: texture_2d<f32>;
fn mod_at(p: vec2i) -> f32 { return clamp(textureLoad(modt, p, 0).x, 0.0, 1.0); }
"#,
    crate::nodes::stage::px_one!(),
    vblur_body!()
);

impl NodeDef for VectorBlurParams {
    const NAME: &'static str = "VectorBlur";
    /// The picture, the vectors, the modulate matte.
    const INPUTS: usize = 3;
    const MASK_INPUT: Option<usize> = None;
    const MOD_INPUT: Option<usize> = Some(2);
    const VEC4_SLOTS: usize = 0;

    fn topo_tag(&self) -> u64 {
        0x7B10
    }

    fn pack(&self, _: &mut [f32]) {}

    fn emit(&self, e: &mut Emit) -> String {
        e.stage_result()
    }
}

impl VectorBlurParams {
    fn chain(&self, wired: Wired, one: bool) -> StageProgram {
        // No vectors: nothing says how anything moved, the picture passes.
        if !wired.has(1) {
            return StageProgram { textures: Vec::new(), passes: Vec::new(), result: "a".into() };
        }
        let modulated = wired.has(2);
        let mut binds = vec![
            Bind::Uniform,
            Bind::Sampler,
            Bind::Tex("a".into()),
            Bind::Tex("aux0".into()),
            Bind::Storage("b".into()),
        ];
        if modulated {
            binds.push(Bind::Tex("aux1".into()));
        }
        StageProgram {
            textures: vec![TexSpec::full("b")],
            passes: vec![Pass {
                label: "vector-blur",
                wgsl: match (one, modulated) {
                    (false, false) => WGSL,
                    (false, true) => MOD_WGSL,
                    (true, false) => ONE_WGSL,
                    (true, true) => ONE_MOD_WGSL,
                },
                binds,
                dispatch: Dispatch::Pixels("a".into()),
                uniform_floats: 4,
            }],
            result: "b".into(),
        }
    }
}

impl StageDef for VectorBlurParams {
    /// The vectors, and the modulate matte, each only when wired.
    fn aux_ports(&self, wired: Wired) -> Vec<usize> {
        if !wired.has(1) {
            return Vec::new();
        }
        let mut v = vec![1];
        if wired.has(2) {
            v.push(2);
        }
        v
    }

    fn program(&self, _canvas: (u32, u32), wired: Wired) -> StageProgram {
        self.chain(wired, false)
    }

    fn program_one_channel(&self, _canvas: (u32, u32), wired: Wired) -> Option<StageProgram> {
        Some(self.chain(wired, true).one_channel())
    }

    /// The vectors are already canvas pixels (the Flow node scales them
    /// under proxy), so nothing here is pixel-sized.
    fn uniforms(&self, _pass: usize, _cx: &StageCx) -> Vec<f32> {
        vec![
            self.amount.max(0.0),
            self.offset.clamp(-0.5, 0.5),
            self.samples.clamp(1, MAX_SAMPLES) as f32,
            0.0,
        ]
    }

    fn speed(&self) -> Speed {
        Speed::Slow
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_shutter_is_centred_and_spans_the_amount() {
        let p = VectorBlurParams { amount: 1.0, offset: 0.0, samples: 4, matte: false };
        assert_eq!(p.times(1.0), vec![-0.375, -0.125, 0.125, 0.375]);
        let open = VectorBlurParams { offset: 0.5, ..p };
        assert!(open.times(1.0).iter().all(|s| *s >= 0.0), "opening on the frame looks forward only");
        assert!(p.times(0.0).iter().all(|s| *s == 0.0), "a black modulate matte is no blur");
    }

    #[test]
    fn each_half_of_the_shutter_reads_its_own_vector() {
        let f = [4.0, 0.0];
        let b = [0.0, -2.0];
        assert_eq!(VectorBlurParams::at([10.0, 10.0], f, b, 0.5), [12.0, 10.0]);
        assert_eq!(VectorBlurParams::at([10.0, 10.0], f, b, -0.5), [10.0, 9.0]);
    }

    #[test]
    fn the_vectors_cost_a_pass_only_when_wired() {
        let p = VectorBlurParams::default();
        let none = p.program((64, 64), Wired(&[true, false, false]));
        assert!(none.passes.is_empty() && none.result == "a");
        assert!(p.aux_ports(Wired(&[true, false, true])).is_empty(), "a matte alone blurs nothing");
        assert_eq!(p.aux_ports(Wired(&[true, true, true])), vec![1, 2]);
        let m = p.program((64, 64), Wired(&[true, true, true]));
        assert_eq!(m.passes[0].binds.len(), 6);
        assert!(m.passes[0].binds.contains(&Bind::Tex("aux1".into())));
    }
}
