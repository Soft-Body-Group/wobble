//! Edge Detect: a 3x3 gradient (or Laplacian) per channel, one gather pass.
//! Out comes either the edge strength as a picture or a thresholded matte.

use crate::codegen::Emit;
use crate::cost::Speed;
use crate::nodes::stage::{Bind, Dispatch, Pass, StageCx, StageDef, StageProgram, TexSpec, Wired};
use crate::nodes::NodeDef;

/// Which kernel measures the edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum EdgeOp {
    /// 1-2-1 smoothing across the derivative: the usual choice.
    #[default]
    Sobel,
    /// 1-1-1: flatter, a touch noisier.
    Prewitt,
    /// 3-10-3: the most rotation-even of the three.
    Scharr,
    /// The second derivative, |∇²|: thin double lines, no direction.
    Laplacian,
}

impl EdgeOp {
    pub const ALL: [EdgeOp; 4] = [EdgeOp::Sobel, EdgeOp::Prewitt, EdgeOp::Scharr, EdgeOp::Laplacian];
    pub fn name(self) -> &'static str {
        match self {
            EdgeOp::Sobel => "sobel",
            EdgeOp::Prewitt => "prewitt",
            EdgeOp::Scharr => "scharr",
            EdgeOp::Laplacian => "laplacian",
        }
    }
    fn index(self) -> f32 {
        match self {
            EdgeOp::Sobel => 0.0,
            EdgeOp::Prewitt => 1.0,
            EdgeOp::Scharr => 2.0,
            EdgeOp::Laplacian => 3.0,
        }
    }
}

/// What comes out of the node.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum EdgeOutput {
    /// rgb = each channel's edge strength × gain; alpha passes through.
    #[default]
    Magnitude,
    /// The strongest channel's edge, through the threshold: a MATTE, so
    /// the output socket is on the matte pipe.
    Matte,
}

impl EdgeOutput {
    pub const ALL: [EdgeOutput; 2] = [EdgeOutput::Magnitude, EdgeOutput::Matte];
    pub fn name(self) -> &'static str {
        match self {
            EdgeOutput::Magnitude => "magnitude",
            EdgeOutput::Matte => "matte",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EdgeDetectParams {
    pub op: EdgeOp,
    pub output: EdgeOutput,
    /// Tap spacing in pixels: 1 is the plain 3x3; wider finds softer edges.
    pub size: f32,
    /// Scales the strength (both outputs, before the threshold).
    pub gain: f32,
    /// Matte only: the strength that counts as an edge.
    pub threshold: f32,
    /// Matte only: the width of the ramp centred on the threshold.
    pub softness: f32,
    /// Which pipe it sits in: wired a MATTE it filters the coverage and
    /// gives a matte (`NodeKind::follows_pipe`). Kept true to its input by
    /// the app (`EditorGraph::connect_out`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub matte: bool,
}

impl Default for EdgeDetectParams {
    fn default() -> Self {
        Self {
            op: EdgeOp::Sobel,
            output: EdgeOutput::Magnitude,
            size: 1.0,
            gain: 1.0,
            threshold: 0.1,
            softness: 0.05,
            matte: false,
        }
    }
}

/// How far apart the taps may sit.
pub const EDGE_SIZE_MAX: f32 = 8.0;

pub const WGSL: &str = r#"
struct EP { op: f32, size: f32, gain: f32, threshold: f32, softness: f32, output: f32, _p0: f32, _p1: f32 }
@group(0) @binding(0) var<uniform> E: EP;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var src: texture_2d<f32>;
@group(0) @binding(3) var dst: texture_storage_2d<rgba16float, write>;

// Normalised so a hard step of height 1 reads 1 whatever the kernel: the
// threshold means the same thing under every operator.
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = vec2i(textureDimensions(dst));
    if (gid.x >= u32(dims.x) || gid.y >= u32(dims.y)) { return; }
    let p = vec2i(gid.xy);
    let c = textureLoad(src, p, 0);
    // Bilinear taps, so a fractional size really sits between pixels.
    let base = (vec2f(p) + vec2f(0.5)) / vec2f(dims);
    let st = vec2f(E.size) / vec2f(dims);
    let nw = textureSampleLevel(src, samp, base + vec2f(-1.0, -1.0) * st, 0.0);
    let n  = textureSampleLevel(src, samp, base + vec2f( 0.0, -1.0) * st, 0.0);
    let ne = textureSampleLevel(src, samp, base + vec2f( 1.0, -1.0) * st, 0.0);
    let w  = textureSampleLevel(src, samp, base + vec2f(-1.0,  0.0) * st, 0.0);
    let e  = textureSampleLevel(src, samp, base + vec2f( 1.0,  0.0) * st, 0.0);
    let sw = textureSampleLevel(src, samp, base + vec2f(-1.0,  1.0) * st, 0.0);
    let s  = textureSampleLevel(src, samp, base + vec2f( 0.0,  1.0) * st, 0.0);
    let se = textureSampleLevel(src, samp, base + vec2f( 1.0,  1.0) * st, 0.0);
    let op = i32(E.op + 0.5);
    var m: vec4f;
    if (op == 3) {
        m = abs(n + s + e + w - 4.0 * c);
    } else {
        // Centre weight k, and the sum of one column, which is what a unit
        // step puts through it.
        var k = 2.0;
        var norm = 4.0;
        if (op == 1) { k = 1.0; norm = 3.0; }
        if (op == 2) { k = 10.0 / 3.0; norm = 16.0 / 3.0; }
        let gx = (ne + k * e + se) - (nw + k * w + sw);
        let gy = (sw + k * s + se) - (nw + k * n + ne);
        m = sqrt(gx * gx + gy * gy) / norm;
    }
    m = m * E.gain;
    var out = vec4f(m.rgb, c.a);
    if (E.output > 0.5) {
        let v = max(max(m.r, m.g), max(m.b, m.a));
        let h = 0.5 * E.softness;
        var k = select(0.0, 1.0, v >= E.threshold);
        if (h > 1e-5) { k = smoothstep(E.threshold - h, E.threshold + h, v); }
        out = vec4f(k);
    }
    textureStore(dst, gid.xy, out);
}
"#;

impl NodeDef for EdgeDetectParams {
    const NAME: &'static str = "EdgeDetect";
    /// Image, then the mask.
    const INPUTS: usize = 2;
    const MASK_INPUT: Option<usize> = Some(1);
    /// Every param rides in the pass's own uniform.
    const VEC4_SLOTS: usize = 0;

    /// The output alone: it picks the output's pipe and how the mask
    /// applies. The operator, size and threshold are uniforms.
    fn topo_tag(&self) -> u64 {
        0x4ED0 + self.output as u64
    }

    fn pack(&self, _: &mut [f32]) {}

    fn emit(&self, e: &mut Emit) -> String {
        let v = e.stage_result();
        match self.output {
            EdgeOutput::Magnitude => e.masked_stage(Self::MASK_INPUT, v),
            // A matte crossfaded against a picture means nothing; the mask
            // limits WHERE edges are found instead.
            EdgeOutput::Matte => match Self::MASK_INPUT.and_then(|m| e.mask(m)) {
                Some(mv) => {
                    let v2 = e.fresh();
                    e.line(format_args!("let {v2} = {v} * {mv}.x;"));
                    v2
                }
                None => v,
            },
        }
    }
}

impl StageDef for EdgeDetectParams {
    fn program(&self, _canvas: (u32, u32), _wired: Wired) -> StageProgram {
        StageProgram {
            textures: vec![TexSpec::full("b")],
            passes: vec![Pass {
                label: "edge",
                wgsl: WGSL,
                binds: vec![
                    Bind::Uniform,
                    Bind::Sampler,
                    Bind::Tex("a".into()),
                    Bind::Storage("b".into()),
                ],
                dispatch: Dispatch::Pixels("a".into()),
                uniform_floats: 8,
            }],
            result: "b".into(),
        }
    }

    fn uniforms(&self, _pass: usize, cx: &StageCx) -> Vec<f32> {
        let size = (self.size * cx.proxy_scale).clamp(0.05, EDGE_SIZE_MAX);
        let matte = if self.output == EdgeOutput::Matte { 1.0 } else { 0.0 };
        vec![
            self.op.index(),
            size,
            self.gain,
            self.threshold,
            self.softness.max(0.0),
            matte,
            0.0,
            0.0,
        ]
    }

    fn speed(&self) -> Speed {
        Speed::Slow
    }
}
