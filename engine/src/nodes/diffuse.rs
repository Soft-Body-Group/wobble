//! Diffuse: atmospheric diffusion — light scattered by mist, haze or a
//! diffusion filter. Each pixel gives up `amount` of its light to a wide,
//! soft halo of its neighbourhood (the Glow pyramid, run over the whole
//! picture rather than its highlights), so contrast softens and highlights
//! bloom without the picture getting brighter: in scene-linear light a
//! bright source dominates the halo, which is the natural look a plain
//! blur mix does not have. `haze` lays a veil of the mist's own colour on
//! top.
//!
//! An optional DEPTH input (data pipe, port 2) scales both by distance: a
//! ramp from `near` to `far` (`far_low` for a pass where small is far). It
//! also weights the SOURCE of the scattered light by that ramp, and the
//! halo is normalised by the ramp blurred alongside it (carried in alpha
//! through the pyramid), so a sharp foreground never haloes out over the
//! mist behind it. An optional MODULATE matte (port 3) scales the same
//! weight by hand. Unwired, neither costs anything: no prep pass, no aux
//! subtree.

use crate::codegen::Emit;
use crate::cost::Speed;
use crate::nodes::glow::{levels, weights};
use crate::nodes::stage::{Bind, Dispatch, Pass, StageCx, StageDef, StageProgram, TexSpec, Wired};
use crate::nodes::NodeDef;

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct DiffuseParams {
    /// Halo radius in output pixels: how far scattered light travels.
    pub size: f32,
    /// Octave-to-octave weight ratio (as on Glow): low is a tight core,
    /// 1 an even veil.
    pub falloff: f32,
    /// Fraction of each pixel's light that scatters. Energy-conserving:
    /// 1 is the halo alone.
    pub amount: f32,
    /// Strength of the veil of `mist` colour laid over the result.
    pub haze: f32,
    /// The mist's colour, scene-linear in the working space.
    pub mist: [f32; 3],
    /// Depth where the effect starts (0 below it).
    pub near: f32,
    /// Depth where it is full.
    pub far: f32,
    /// The depth pass reads SMALL for far (a normalised Z with near = 1).
    pub far_low: bool,
}

impl Default for DiffuseParams {
    fn default() -> Self {
        Self {
            size: 120.0,
            falloff: 0.8,
            amount: 0.5,
            haze: 0.1,
            mist: [0.5, 0.55, 0.6],
            // A Blender mist pass: 0 at the camera, 1 at the far end.
            near: 0.0,
            far: 1.0,
            far_low: false,
        }
    }
}

/// The depth ramp: `near` to `far`, flipped for a pass where small is far.
macro_rules! ramp_fn {
    () => {
        r#"
fn ramp(z: f32) -> f32 {
    let span = P.far - P.near;
    var t = select(step(P.near, z), (z - P.near) / span, abs(span) > 1e-6);
    t = clamp(t, 0.0, 1.0);
    if (P.flip != 0.0) { t = 1.0 - t; }
    return t;
}
"#
    };
}

/// The prep pass's body: `weight(p)` is this pixel's share of the effect.
macro_rules! prep_body {
    () => {
        r#"
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(dst);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let c = textureLoad(src, gid.xy, 0);
    let t = weight(vec2i(gid.xy));
    textureStore(dst, gid.xy, vec4f(c.rgb * t, t));
}
"#
    };
}

/// The per-pixel weight — the depth ramp, the modulate matte, or both —
/// and the light weighted by it: `(rgb * w, w)`.
pub const PREP_WGSL: &str = concat!(
    r#"
struct DP { near: f32, far: f32, flip: f32, _p: f32 }
@group(0) @binding(0) var<uniform> P: DP;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var depth: texture_2d<f32>;
@group(0) @binding(3) var dst: texture_storage_2d<rgba16float, write>;
fn weight(p: vec2i) -> f32 { return ramp(textureLoad(depth, p, 0).x); }
"#,
    ramp_fn!(),
    prep_body!()
);

/// The modulate matte alone: no ramp, so no uniform.
pub const PREP_MOD_WGSL: &str = concat!(
    r#"
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var modt: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<rgba16float, write>;
fn weight(p: vec2i) -> f32 { return clamp(textureLoad(modt, p, 0).x, 0.0, 1.0); }
"#,
    prep_body!()
);

/// Depth and modulate: the ramp scaled by the matte.
pub const PREP_BOTH_WGSL: &str = concat!(
    r#"
struct DP { near: f32, far: f32, flip: f32, _p: f32 }
@group(0) @binding(0) var<uniform> P: DP;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var depth: texture_2d<f32>;
@group(0) @binding(3) var modt: texture_2d<f32>;
@group(0) @binding(4) var dst: texture_storage_2d<rgba16float, write>;
fn weight(p: vec2i) -> f32 {
    return ramp(textureLoad(depth, p, 0).x) * clamp(textureLoad(modt, p, 0).x, 0.0, 1.0);
}
"#,
    ramp_fn!(),
    prep_body!()
);

/// 13-tap downsample of all four channels. `opaque` makes the first level
/// read alpha as 1, so without depth the pyramid's alpha is plain coverage
/// of the weights.
pub const DOWN_WGSL: &str = r#"
struct DD { opaque: f32, _p0: f32, _p1: f32, _p2: f32 }
@group(0) @binding(0) var<uniform> P: DD;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var src: texture_2d<f32>;
@group(0) @binding(3) var dst: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(dst);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let uv = (vec2f(gid.xy) + 0.5) / vec2f(dims);
    let ts = 1.0 / vec2f(textureDimensions(src));
    let a = textureSampleLevel(src, samp, uv + ts * vec2f(-2.0,  2.0), 0.0);
    let b = textureSampleLevel(src, samp, uv + ts * vec2f( 0.0,  2.0), 0.0);
    let c = textureSampleLevel(src, samp, uv + ts * vec2f( 2.0,  2.0), 0.0);
    let d = textureSampleLevel(src, samp, uv + ts * vec2f(-2.0,  0.0), 0.0);
    let e = textureSampleLevel(src, samp, uv,                          0.0);
    let f = textureSampleLevel(src, samp, uv + ts * vec2f( 2.0,  0.0), 0.0);
    let g = textureSampleLevel(src, samp, uv + ts * vec2f(-2.0, -2.0), 0.0);
    let h = textureSampleLevel(src, samp, uv + ts * vec2f( 0.0, -2.0), 0.0);
    let i = textureSampleLevel(src, samp, uv + ts * vec2f( 2.0, -2.0), 0.0);
    let j = textureSampleLevel(src, samp, uv + ts * vec2f(-1.0,  1.0), 0.0);
    let k = textureSampleLevel(src, samp, uv + ts * vec2f( 1.0,  1.0), 0.0);
    let l = textureSampleLevel(src, samp, uv + ts * vec2f(-1.0, -1.0), 0.0);
    let m = textureSampleLevel(src, samp, uv + ts * vec2f( 1.0, -1.0), 0.0);
    var o = e * 0.125;
    o += (a + c + g + i) * 0.03125;
    o += (b + d + f + h) * 0.0625;
    o += (j + k + l + m) * 0.125;
    if (P.opaque != 0.0) { o.a = 1.0; }
    textureStore(dst, gid.xy, o);
}
"#;

/// Tent upsample of the level above plus this level's weighted octave,
/// all four channels.
pub const UP_WGSL: &str = r#"
struct DU { weight: f32, has_upper: f32, _p0: f32, _p1: f32 }
@group(0) @binding(0) var<uniform> P: DU;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var mip: texture_2d<f32>;
@group(0) @binding(3) var upper: texture_2d<f32>;
@group(0) @binding(4) var dst: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(dst);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let uv = (vec2f(gid.xy) + 0.5) / vec2f(dims);
    var acc = textureLoad(mip, gid.xy, 0) * P.weight;
    if (P.has_upper != 0.0) {
        let ts = 1.0 / vec2f(textureDimensions(upper));
        var t = textureSampleLevel(upper, samp, uv, 0.0) * 4.0;
        t += textureSampleLevel(upper, samp, uv + ts * vec2f(-1.0,  0.0), 0.0) * 2.0;
        t += textureSampleLevel(upper, samp, uv + ts * vec2f( 1.0,  0.0), 0.0) * 2.0;
        t += textureSampleLevel(upper, samp, uv + ts * vec2f( 0.0, -1.0), 0.0) * 2.0;
        t += textureSampleLevel(upper, samp, uv + ts * vec2f( 0.0,  1.0), 0.0) * 2.0;
        t += textureSampleLevel(upper, samp, uv + ts * vec2f(-1.0, -1.0), 0.0);
        t += textureSampleLevel(upper, samp, uv + ts * vec2f( 1.0, -1.0), 0.0);
        t += textureSampleLevel(upper, samp, uv + ts * vec2f(-1.0,  1.0), 0.0);
        t += textureSampleLevel(upper, samp, uv + ts * vec2f( 1.0,  1.0), 0.0);
        acc += t / 16.0;
    }
    textureStore(dst, gid.xy, acc);
}
"#;

/// Scatter and veil. `wsrc` is the prep texture (its alpha is this
/// pixel's weight) or the input itself when nothing weights the effect.
pub const COMP_WGSL: &str = r#"
struct DC { amount: f32, haze: f32, weighted: f32, _p: f32, mist: vec4f }
@group(0) @binding(0) var<uniform> P: DC;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var orig: texture_2d<f32>;
@group(0) @binding(3) var wsrc: texture_2d<f32>;
@group(0) @binding(4) var halo: texture_2d<f32>;
@group(0) @binding(5) var dst: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(dst);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let base = textureLoad(orig, gid.xy, 0);
    var f = 1.0;
    if (P.weighted != 0.0) { f = textureLoad(wsrc, gid.xy, 0).a; }
    let uv = (vec2f(gid.xy) + 0.5) / vec2f(dims);
    let d = textureSampleLevel(halo, samp, uv, 0.0);
    // Normalised by the ramp that came through the pyramid with it: the
    // light that reached here from the far part of the neighbourhood only.
    let scattered = select(base.rgb, d.rgb / d.a, d.a > 1e-4);
    var rgb = mix(base.rgb, scattered, clamp(P.amount * f, 0.0, 1.0));
    rgb = mix(rgb, P.mist.rgb, clamp(P.haze * f, 0.0, 1.0));
    textureStore(dst, gid.xy, vec4f(rgb, base.a));
}
"#;

/// The depth port.
pub const DEPTH_INPUT: usize = 2;

impl NodeDef for DiffuseParams {
    const NAME: &'static str = "Diffuse";
    /// Image, mask, depth, then the modulate matte: coverage that scales
    /// the effect per pixel (on top of the depth ramp, when both are wired)
    /// — for hand-drawn mist or anything else the depth pass cannot say.
    const INPUTS: usize = 4;
    const MASK_INPUT: Option<usize> = Some(1);
    const MOD_INPUT: Option<usize> = Some(3);
    /// Every param lives in the chain's own uniforms.
    const VEC4_SLOTS: usize = 0;

    /// Constant: every param is a uniform. Wiring the depth port changes
    /// the chain, but wiring is topology already.
    fn topo_tag(&self) -> u64 {
        31
    }

    fn pack(&self, _: &mut [f32]) {}

    fn emit(&self, e: &mut Emit) -> String {
        let v = e.stage_result();
        e.masked_stage(Self::MASK_INPUT, v)
    }
}

impl StageDef for DiffuseParams {
    /// Depth then modulate, each only when wired: `aux0` is the first of
    /// them that is.
    fn aux_ports(&self, wired: Wired) -> Vec<usize> {
        [Some(DEPTH_INPUT), Self::MOD_INPUT].into_iter().flatten().filter(|i| wired.has(*i)).collect()
    }

    fn program(&self, canvas: (u32, u32), wired: Wired) -> StageProgram {
        let depth = wired.has(DEPTH_INPUT);
        let modulated = Self::MOD_INPUT.is_some_and(|m| wired.has(m));
        // Either one takes the prep pass; the rest of the chain cannot tell
        // which made the weight.
        let prep = depth || modulated;
        let n = levels(canvas.0, canvas.1);
        let mip = |i: usize| format!("mip{i}");
        let up = |i: usize| format!("up{i}");
        let mut textures = vec![TexSpec::full("b")];
        if prep {
            textures.push(TexSpec::full("s"));
        }
        for i in 0..n {
            textures.push(TexSpec::level(&mip(i), i as u32 + 1));
            textures.push(TexSpec::level(&up(i), i as u32 + 1));
        }
        let src = if prep { "s" } else { "a" };
        let mut passes = Vec::with_capacity(2 * n + 2);
        if prep {
            let mut binds = vec![Bind::Tex("a".into()), Bind::Tex("aux0".into())];
            if depth {
                binds.insert(0, Bind::Uniform);
            }
            if depth && modulated {
                binds.push(Bind::Tex("aux1".into()));
            }
            binds.push(Bind::Storage("s".into()));
            passes.push(Pass {
                label: "diffuse-prep",
                wgsl: match (depth, modulated) {
                    (true, true) => PREP_BOTH_WGSL,
                    (true, false) => PREP_WGSL,
                    _ => PREP_MOD_WGSL,
                },
                binds,
                dispatch: Dispatch::Pixels("s".into()),
                uniform_floats: if depth { 4 } else { 0 },
            });
        }
        for i in 0..n {
            let from = if i == 0 { src.to_string() } else { mip(i - 1) };
            passes.push(Pass {
                label: "diffuse-down",
                wgsl: DOWN_WGSL,
                binds: vec![Bind::Uniform, Bind::Sampler, Bind::Tex(from), Bind::Storage(mip(i))],
                dispatch: Dispatch::Pixels(mip(i)),
                uniform_floats: 4,
            });
        }
        for i in (0..n).rev() {
            let upper = if i + 1 < n { up(i + 1) } else { mip(i) };
            passes.push(Pass {
                label: "diffuse-up",
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
            label: "diffuse-comp",
            wgsl: COMP_WGSL,
            binds: vec![
                Bind::Uniform,
                Bind::Sampler,
                Bind::Tex("a".into()),
                Bind::Tex(src.into()),
                Bind::Tex(up(0)),
                Bind::Storage("b".into()),
            ],
            dispatch: Dispatch::Pixels("b".into()),
            uniform_floats: 8,
        });
        StageProgram { textures, passes, result: "b".into() }
    }

    fn uniforms(&self, pass: usize, cx: &StageCx) -> Vec<f32> {
        // The uniforms cannot see the wiring; the pass count can. With a
        // weight (depth, modulate or both) the chain is one pass longer than
        // 2n + 1, and nothing past the prep pass cares which it was.
        let n = levels(cx.canvas.0, cx.canvas.1);
        let prep = cx.passes > 2 * n + 1;
        let flag = |b: bool| b as u32 as f32;
        let mut i = pass;
        if prep {
            if i == 0 {
                return vec![self.near, self.far, flag(self.far_low), 0.0];
            }
            i -= 1;
        }
        if i < n {
            // Without a weight the first level reads the input itself,
            // whose alpha is not one: read it as 1.
            vec![flag(i == 0 && !prep), 0.0, 0.0, 0.0]
        } else if i < 2 * n {
            let l = 2 * n - 1 - i;
            let w = weights(self.size * cx.proxy_scale, self.falloff, n);
            vec![w[l], flag(l + 1 < n), 0.0, 0.0]
        } else {
            let m = self.mist;
            vec![self.amount, self.haze, flag(prep), 0.0, m[0], m[1], m[2], 1.0]
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
    fn depth_costs_nothing_unwired() {
        let p = DiffuseParams::default();
        let n = levels(256, 256);
        let plain = p.program((256, 256), Wired(&[true, false, false]));
        assert_eq!(plain.passes.len(), 2 * n + 1);
        assert!(p.aux_ports(Wired(&[true, false, false])).is_empty());
        assert!(!plain.textures.iter().any(|t| t.name == "s"));
        let deep = p.program((256, 256), Wired(&[true, false, true, false]));
        assert_eq!(deep.passes.len(), 2 * n + 2);
        assert_eq!(p.aux_ports(Wired(&[true, false, true, false])), vec![DEPTH_INPUT]);
        assert_eq!(p.aux_ports(Wired(&[true, false, false, true])), vec![3]);
        assert_eq!(p.aux_ports(Wired(&[true, false, true, true])), vec![DEPTH_INPUT, 3]);
        let both = p.program((256, 256), Wired(&[true, false, true, true]));
        assert_eq!(both.passes[0].wgsl, PREP_BOTH_WGSL);
        assert!(both.passes[0].binds.contains(&Bind::Tex("aux1".into())));
    }

    #[test]
    fn uniform_spans_match_the_passes() {
        let p = DiffuseParams::default();
        for wired in [[true, false, false, false], [true, false, true, false], [true, false, false, true], [true, false, true, true]] {
            let prog = p.program((300, 200), Wired(&wired));
            let cx = StageCx {
                canvas: (300, 200),
                proxy_scale: 1.0,
                frame: 0.0,
                fps: 24.0,
                passes: prog.passes.len(),
            };
            for (i, pass) in prog.passes.iter().enumerate() {
                if pass.uniform_floats == 0 {
                    // No buffer: the compositor never asks.
                    assert!(!pass.binds.contains(&Bind::Uniform));
                    continue;
                }
                assert_eq!(p.uniforms(i, &cx).len(), pass.uniform_floats, "pass {i} {}", pass.label);
            }
        }
    }
}
