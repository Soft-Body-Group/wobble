//! Median: per-channel median over a square window. Exact for small radii,
//! separable (median of row medians) past them, two passes either way.

use crate::codegen::Emit;
use crate::cost::Speed;
use crate::nodes::stage::{Bind, Dispatch, Pass, StageCx, StageDef, StageProgram, TexSpec, Wired};
use crate::nodes::NodeDef;

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MedianParams {
    /// Window radius in pixels, rounded to a whole one: the window is
    /// (2r + 1) wide. 0 passes the input through.
    pub radius: f32,
    /// Which pipe it sits in: wired a MATTE it filters the coverage and
    /// gives a matte (`NodeKind::follows_pipe`). Kept true to its input by
    /// the app (`EditorGraph::connect_out`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub matte: bool,
}

impl Default for MedianParams {
    fn default() -> Self {
        Self { radius: 2.0, matte: false }
    }
}

/// The widest window: 2 * 24 + 1 = 49 taps per pass, the private array the
/// passes sort into.
pub const MEDIAN_MAX: f32 = 24.0;

/// Up to this radius the first pass takes the TRUE 2D median of the whole
/// (2r + 1)² window (7x7 = 49 values, the same array) and the second pass
/// copies. Past it, the window is too big to sort per pixel, so pass one
/// takes each row's median and pass two the median of those — the standard
/// separable approximation, which keeps the cost linear in the radius.
pub const MEDIAN_EXACT: f32 = 3.0;

/// The shared part: bindings, the sorted window and how a value joins it.
/// Inserting with componentwise min/max keeps EVERY channel sorted on its
/// own, so one array is four independent medians.
///
/// `$fmt` is the scratch's storage format and `$px` the matching pixel
/// header (`px_rgba!` / `px_one!`): one array of `PX`, so a matte's window
/// sorts single floats rather than four lanes of the same number.
macro_rules! median_head {
    ($fmt:literal, $px:expr) => {
        concat!(
            r#"
struct MP { r: f32, _p0: f32, _p1: f32, _p2: f32 }
@group(0) @binding(0) var<uniform> M: MP;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<"#,
            $fmt,
            r#", write>;
"#,
            $px,
            r#"

const EXACT: i32 = 3;
var<private> win: array<PX, 49>;
var<private> cnt: i32;

fn put(v: PX) {
    var x = v;
    for (var j = 0; j < cnt; j++) {
        let a = win[j];
        win[j] = min(a, x);
        x = max(a, x);
    }
    win[cnt] = x;
    cnt++;
}

fn tap(p: vec2i, dims: vec2i) -> PX {
    return px(textureLoad(src, clamp(p, vec2i(0), dims - 1), 0));
}
"#
        )
    };
}

/// Pass one: the exact 2D median, or each row's.
macro_rules! median_h {
    () => {
        r#"
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = vec2i(textureDimensions(dst));
    if (gid.x >= u32(dims.x) || gid.y >= u32(dims.y)) { return; }
    let p = vec2i(gid.xy);
    let r = i32(M.r);
    if (r <= 0) {
        textureStore(dst, gid.xy, px_out(tap(p, dims)));
        return;
    }
    cnt = 0;
    if (r <= EXACT) {
        for (var dy = -r; dy <= r; dy++) {
            for (var dx = -r; dx <= r; dx++) {
                put(tap(p + vec2i(dx, dy), dims));
            }
        }
    } else {
        for (var dx = -r; dx <= r; dx++) {
            put(tap(p + vec2i(dx, 0), dims));
        }
    }
    textureStore(dst, gid.xy, px_out(win[cnt / 2]));
}
"#
    };
}

/// Pass two: the median of the row medians, or a copy when pass one was
/// already exact.
macro_rules! median_v {
    () => {
        r#"
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = vec2i(textureDimensions(dst));
    if (gid.x >= u32(dims.x) || gid.y >= u32(dims.y)) { return; }
    let p = vec2i(gid.xy);
    let r = i32(M.r);
    if (r <= EXACT) {
        textureStore(dst, gid.xy, px_out(tap(p, dims)));
        return;
    }
    cnt = 0;
    for (var dy = -r; dy <= r; dy++) {
        put(tap(p + vec2i(0, dy), dims));
    }
    textureStore(dst, gid.xy, px_out(win[cnt / 2]));
}
"#
    };
}

pub const H_WGSL: &str =
    concat!(median_head!("rgba16float", crate::nodes::stage::px_rgba!()), median_h!());
pub const V_WGSL: &str =
    concat!(median_head!("rgba16float", crate::nodes::stage::px_rgba!()), median_v!());
/// The two passes on ONE channel: a matte's median.
pub const ONE_H_WGSL: &str =
    concat!(median_head!("r16float", crate::nodes::stage::px_one!()), median_h!());
pub const ONE_V_WGSL: &str =
    concat!(median_head!("r16float", crate::nodes::stage::px_one!()), median_v!());

impl NodeDef for MedianParams {
    const NAME: &'static str = "Median";
    /// Image, then the mask.
    const INPUTS: usize = 2;
    const MASK_INPUT: Option<usize> = Some(1);
    /// The radius rides in the passes' own uniforms.
    const VEC4_SLOTS: usize = 0;

    /// Constant: both passes always run, and which kind of median they take
    /// is a branch on the radius uniform, so a drag never recompiles.
    fn topo_tag(&self) -> u64 {
        0x3ED1
    }

    fn pack(&self, _: &mut [f32]) {}

    fn emit(&self, e: &mut Emit) -> String {
        let v = e.stage_result();
        e.masked_stage(Self::MASK_INPUT, v)
    }
}

impl StageDef for MedianParams {
    fn program(&self, _canvas: (u32, u32), _wired: Wired) -> StageProgram {
        let pass = |label, wgsl, from: &str, to: &str| Pass {
            label,
            wgsl,
            binds: vec![Bind::Uniform, Bind::Tex(from.into()), Bind::Storage(to.into())],
            dispatch: Dispatch::Pixels("a".into()),
            uniform_floats: 4,
        };
        StageProgram {
            textures: vec![TexSpec::full("b"), TexSpec::full("c")],
            passes: vec![pass("median-h", H_WGSL, "a", "b"), pass("median-v", V_WGSL, "b", "c")],
            result: "c".into(),
        }
    }

    fn program_one_channel(&self, _canvas: (u32, u32), _wired: Wired) -> Option<StageProgram> {
        let pass = |label, wgsl, from: &str, to: &str| Pass {
            label,
            wgsl,
            binds: vec![Bind::Uniform, Bind::Tex(from.into()), Bind::Storage(to.into())],
            dispatch: Dispatch::Pixels("a".into()),
            uniform_floats: 4,
        };
        Some(
            StageProgram {
                textures: vec![TexSpec::full("b"), TexSpec::full("c")],
                passes: vec![
                    pass("median-h", ONE_H_WGSL, "a", "b"),
                    pass("median-v", ONE_V_WGSL, "b", "c"),
                ],
                result: "c".into(),
            }
            .one_channel(),
        )
    }

    fn uniforms(&self, _pass: usize, cx: &StageCx) -> Vec<f32> {
        vec![(self.radius * cx.proxy_scale).round().clamp(0.0, MEDIAN_MAX), 0.0, 0.0, 0.0]
    }

    fn speed(&self) -> Speed {
        Speed::Slow
    }
}
