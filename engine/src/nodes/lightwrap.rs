//! Lightwrap: let the background's light bleed around the foreground's
//! edges, the way a real lens and a real atmosphere wrap a bright
//! background round whatever stands in front of it — the step that stops a
//! keyed element looking cut out and pasted on.
//!
//! Input 0 is the foreground (premultiplied, with its alpha), input 1 the
//! background it will sit over, input 2 the mask. The wrap is the
//! background, blurred, laid into a band just INSIDE the foreground's edge:
//!
//! ```text
//! band = fg.a · (1 - blur(fg.a, width))
//! wrap = blur(bg, bg_blur) · band · intensity
//! out  = fg + wrap            (or screen, or the wrap alone)
//! ```
//!
//! `width` is how far in the light reaches; `bg_blur` how soft the light
//! is. The output keeps the foreground's alpha, so it merges over the same
//! background as the plain foreground did.
//!
//! A stage: both blurs sample a neighbourhood. The foreground alpha and the
//! background ride ONE texture through one separable blur (rgb the
//! background, alpha the foreground's alpha, each with its own sigma), so
//! the chain is three passes whatever the widths. With no background wired
//! there is nothing to wrap: the program has no passes and the node passes
//! the foreground through, at no cost beyond the subtree pass every stage
//! has.

use crate::codegen::Emit;
use crate::cost::Speed;
use crate::nodes::stage::{Bind, Dispatch, Pass, StageCx, StageDef, StageProgram, TexSpec, Wired};
use crate::nodes::NodeDef;

/// How the wrap goes onto the foreground.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum WrapMode {
    /// Linear add — what light does.
    #[default]
    Add,
    /// Screen: rolls off instead of clipping.
    Screen,
    /// The wrap alone (alpha = the band), for merging by hand downstream.
    Only,
}

impl WrapMode {
    pub const ALL: [WrapMode; 3] = [WrapMode::Add, WrapMode::Screen, WrapMode::Only];
    pub fn name(self) -> &'static str {
        match self {
            WrapMode::Add => "add",
            WrapMode::Screen => "screen",
            WrapMode::Only => "wrap only",
        }
    }
    fn code(self) -> f32 {
        match self {
            WrapMode::Add => 0.0,
            WrapMode::Screen => 1.0,
            WrapMode::Only => 2.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct LightwrapParams {
    /// How far into the foreground the light reaches: the sigma, in output
    /// pixels, of the blur on the foreground's alpha.
    pub width: f32,
    /// How soft the wrapped light is: the sigma, in output pixels, of the
    /// blur on the background.
    pub bg_blur: f32,
    /// Multiplier on the wrap.
    pub intensity: f32,
    pub mode: WrapMode,
}

impl Default for LightwrapParams {
    fn default() -> Self {
        Self { width: 12.0, bg_blur: 20.0, intensity: 1.0, mode: WrapMode::Add }
    }
}

impl LightwrapParams {
    /// The CPU mirror of the comp pass, given the two blurred values at a
    /// pixel: `bg_blurred` (rgb) and `alpha_blurred` (the foreground alpha
    /// after its blur). Returns the output rgba.
    pub fn combine(&self, fg: [f32; 4], bg_blurred: [f32; 3], alpha_blurred: f32) -> [f32; 4] {
        let band = (fg[3] * (1.0 - alpha_blurred)).clamp(0.0, 1.0);
        let wrap = bg_blurred.map(|c| c.max(0.0) * band * self.intensity);
        match self.mode {
            WrapMode::Add => [fg[0] + wrap[0], fg[1] + wrap[1], fg[2] + wrap[2], fg[3]],
            WrapMode::Screen => {
                let s = |b: f32, w: f32| {
                    let p = b.max(0.0);
                    b - p + (1.0 - (1.0 - p) * (1.0 - w.min(1.0)))
                };
                [s(fg[0], wrap[0]), s(fg[1], wrap[1]), s(fg[2], wrap[2]), fg[3]]
            }
            WrapMode::Only => [wrap[0], wrap[1], wrap[2], band * self.intensity.clamp(0.0, 1.0)],
        }
    }
}

/// The separable blur over the packed (bg rgb, fg alpha) texture: rgb at
/// one sigma, alpha at another, one walk out to the larger. Each module
/// defines `fetch(q)` and the bindings; the body is shared.
macro_rules! wrap_blur_body {
    () => {
        r#"
fn gauss(i: i32, sigma: f32) -> f32 {
    if (sigma < 0.05) { return select(0.0, 1.0, i == 0); }
    return exp(-f32(i * i) / (2.0 * sigma * sigma));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = vec2i(textureDimensions(dst));
    if (gid.x >= u32(dims.x) || gid.y >= u32(dims.y)) { return; }
    let p = vec2i(gid.xy);
    let r = min(i32(ceil(max(B.sigma_rgb, B.sigma_a) * 3.0)), 128);
    var rgb = vec3f(0.0);
    var a = 0.0;
    var wr = 0.0;
    var wa = 0.0;
    for (var i = -r; i <= r; i = i + 1) {
        let q = clamp(p + vec2i(B.dir) * i, vec2i(0), dims - 1);
        let s = fetch(q);
        let gr = gauss(i, B.sigma_rgb);
        let ga = gauss(i, B.sigma_a);
        rgb += s.rgb * gr;
        a += s.a * ga;
        wr += gr;
        wa += ga;
    }
    textureStore(dst, gid.xy, vec4f(rgb / wr, a / wa));
}
"#
    };
}

/// Horizontal: packs the background's rgb and the foreground's alpha as it
/// reads them.
pub const PACK_H_WGSL: &str = concat!(
    r#"
struct WB { dir: vec2f, sigma_rgb: f32, sigma_a: f32 }
@group(0) @binding(0) var<uniform> B: WB;
@group(0) @binding(1) var fg: texture_2d<f32>;
@group(0) @binding(2) var bg: texture_2d<f32>;
@group(0) @binding(3) var dst: texture_storage_2d<rgba16float, write>;
fn fetch(q: vec2i) -> vec4f {
    return vec4f(textureLoad(bg, q, 0).rgb, textureLoad(fg, q, 0).a);
}
"#,
    wrap_blur_body!()
);

/// Vertical: the packed texture again, along the other axis.
pub const V_WGSL: &str = concat!(
    r#"
struct WB { dir: vec2f, sigma_rgb: f32, sigma_a: f32 }
@group(0) @binding(0) var<uniform> B: WB;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<rgba16float, write>;
fn fetch(q: vec2i) -> vec4f { return textureLoad(src, q, 0); }
"#,
    wrap_blur_body!()
);

/// The band, the wrap, and the wrap onto the foreground.
pub const COMP_WGSL: &str = r#"
struct WC { intensity: f32, mode: f32, _p0: f32, _p1: f32 }
@group(0) @binding(0) var<uniform> P: WC;
@group(0) @binding(1) var fg: texture_2d<f32>;
@group(0) @binding(2) var blurred: texture_2d<f32>;
@group(0) @binding(3) var dst: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(dst);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let f = textureLoad(fg, gid.xy, 0);
    let b = textureLoad(blurred, gid.xy, 0);
    let band = clamp(f.a * (1.0 - b.a), 0.0, 1.0);
    let wrap = max(b.rgb, vec3f(0.0)) * band * P.intensity;
    var out = vec4f(f.rgb + wrap, f.a);
    if (P.mode == 1.0) {
        // Screen, on the positive part only, like the Glow's.
        let p = max(f.rgb, vec3f(0.0));
        out = vec4f(f.rgb - p + (1.0 - (1.0 - p) * (1.0 - min(wrap, vec3f(1.0)))), f.a);
    } else if (P.mode == 2.0) {
        out = vec4f(wrap, band * clamp(P.intensity, 0.0, 1.0));
    }
    textureStore(dst, gid.xy, out);
}
"#;

impl NodeDef for LightwrapParams {
    const NAME: &'static str = "Lightwrap";
    /// Foreground, background, then the mask.
    const INPUTS: usize = 3;
    const MASK_INPUT: Option<usize> = Some(2);
    /// Every param lives in the chain's own uniforms.
    const VEC4_SLOTS: usize = 0;

    /// Constant: every param is a uniform. Whether the background is wired
    /// changes the chain, and wiring is already topology.
    fn topo_tag(&self) -> u64 {
        0x70
    }

    fn pack(&self, _: &mut [f32]) {}

    fn emit(&self, e: &mut Emit) -> String {
        let v = e.stage_result();
        e.masked_stage(Self::MASK_INPUT, v)
    }
}

impl StageDef for LightwrapParams {
    /// The background, only when wired: unwired there is nothing to wrap.
    fn aux_ports(&self, wired: Wired) -> Vec<usize> {
        if wired.has(1) {
            vec![1]
        } else {
            Vec::new()
        }
    }

    fn program(&self, _canvas: (u32, u32), wired: Wired) -> StageProgram {
        if !wired.has(1) {
            return StageProgram { textures: Vec::new(), passes: Vec::new(), result: "a".into() };
        }
        StageProgram {
            textures: vec![TexSpec::full("b"), TexSpec::full("c")],
            passes: vec![
                Pass {
                    label: "lightwrap-h",
                    wgsl: PACK_H_WGSL,
                    binds: vec![
                        Bind::Uniform,
                        Bind::Tex("a".into()),
                        Bind::Tex("aux0".into()),
                        Bind::Storage("b".into()),
                    ],
                    dispatch: Dispatch::Pixels("a".into()),
                    uniform_floats: 4,
                },
                Pass {
                    label: "lightwrap-v",
                    wgsl: V_WGSL,
                    binds: vec![Bind::Uniform, Bind::Tex("b".into()), Bind::Storage("c".into())],
                    dispatch: Dispatch::Pixels("a".into()),
                    uniform_floats: 4,
                },
                Pass {
                    label: "lightwrap-comp",
                    wgsl: COMP_WGSL,
                    binds: vec![
                        Bind::Uniform,
                        Bind::Tex("a".into()),
                        Bind::Tex("c".into()),
                        Bind::Storage("b".into()),
                    ],
                    dispatch: Dispatch::Pixels("a".into()),
                    uniform_floats: 4,
                },
            ],
            result: "b".into(),
        }
    }

    fn uniforms(&self, pass: usize, cx: &StageCx) -> Vec<f32> {
        // Both widths are pixels: scale with the proxy.
        let rgb = self.bg_blur.max(0.0) * cx.proxy_scale;
        let a = self.width.max(0.0) * cx.proxy_scale;
        match pass {
            0 => vec![1.0, 0.0, rgb, a],
            1 => vec![0.0, 1.0, rgb, a],
            _ => vec![self.intensity, self.mode.code(), 0.0, 0.0],
        }
    }

    fn speed(&self) -> Speed {
        Speed::Slow
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deep inside the foreground the blurred alpha is 1 and there is no
    /// band; outside it there is no foreground. Only the edge takes light.
    #[test]
    fn only_the_edge_takes_light() {
        let p = LightwrapParams::default();
        let fg = [0.2, 0.3, 0.4, 1.0];
        assert_eq!(p.combine(fg, [5.0; 3], 1.0), fg, "the core must not move");
        assert_eq!(p.combine([0.0; 4], [5.0; 3], 0.0), [0.0; 4], "nothing outside");
        let edge = p.combine(fg, [1.0, 0.5, 0.0], 0.5);
        assert!((edge[0] - 0.7).abs() < 1e-6 && (edge[1] - 0.55).abs() < 1e-6, "{edge:?}");
        assert_eq!(edge[2], 0.4);
        assert_eq!(edge[3], 1.0, "alpha is the foreground's");
    }

    #[test]
    fn intensity_and_modes() {
        let fg = [0.5, 0.5, 0.5, 1.0];
        let off = LightwrapParams { intensity: 0.0, ..Default::default() };
        assert_eq!(off.combine(fg, [3.0; 3], 0.2), fg);
        let only = LightwrapParams { mode: WrapMode::Only, ..Default::default() };
        let w = only.combine(fg, [1.0; 3], 0.25);
        assert_eq!(w, [0.75, 0.75, 0.75, 0.75]);
        let screen = LightwrapParams { mode: WrapMode::Screen, ..Default::default() };
        let s = screen.combine(fg, [10.0; 3], 0.0);
        assert!(s[0] <= 1.0 && s[0] > 0.5, "screen rolls off: {s:?}");
    }

    #[test]
    fn unwired_background_is_free() {
        let p = LightwrapParams::default();
        let prog = p.program((64, 64), Wired(&[true, false, false]));
        assert!(prog.passes.is_empty() && prog.textures.is_empty());
        assert_eq!(prog.result, "a");
        assert!(p.aux_ports(Wired(&[true, false, false])).is_empty());
        let wired = p.program((64, 64), Wired(&[true, true, false]));
        assert_eq!(wired.passes.len(), 3);
        assert_eq!(p.aux_ports(Wired(&[true, true, false])), vec![1]);
    }
}
