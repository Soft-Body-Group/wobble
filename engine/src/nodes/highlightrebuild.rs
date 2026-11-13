//! Highlight Rebuild: clipped highlights where only SOME channels hit the
//! clip, rebuilt from the channels that did not, at the colour of the
//! unclipped surface around them — the raw developers' "reconstruct
//! colour" / "inpaint opposed". No model, real data only: a pixel whose
//! three channels all clipped has nothing left to rebuild from and is left
//! as it is, and a pixel with no channel near the clip is stored untouched
//! (bit-exact).
//!
//! Four passes over the input `a`:
//!
//! 1. `ref`: every pixel that is a good colour reference — all channels
//!    under the knee, not dark — writes its colour normalised by its own
//!    max, weighted by how bright it is (the light nearest the clip is the
//!    best guess at the clipped light's colour), and its weight in alpha.
//!    Everything else writes zero.
//! 2. / 3. a separable tent average of that over the search radius, so a
//!    clipped pixel sees the mean reference colour around it. Averages, not
//!    sums, so half floats hold them.
//! 4. `rebuild`: per channel, how clipped it is (0 under the knee, 1 at the
//!    clip level); the trusted channels say how bright the pixel is
//!    relative to the reference colour, and the clipped ones become that
//!    brightness times the reference — never darker than they were.

use crate::codegen::Emit;
use crate::cost::Speed;
use crate::nodes::stage::{Bind, Dispatch, Pass, StageCx, StageDef, StageProgram, TexSpec, Wired};
use crate::nodes::NodeDef;

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HighlightRebuildParams {
    /// The value a channel clipped at, scene-linear. A plate's clip is
    /// whatever its input transform made of the sensor's white — 1.0 for
    /// most display-referred sources, often less (or more) for camera log.
    pub clip: f32,
    /// How far below the clip the rebuild fades in, as a fraction of it:
    /// a channel at `clip * (1 - softness)` is trusted fully, at `clip`
    /// not at all. Also the line a reference pixel must stay under.
    pub softness: f32,
    /// How far, in pixels, to look for unclipped colour around a clip. It
    /// must reach across the clipped region from its edge: a pixel with no
    /// reference in reach is left as it is.
    pub radius: f32,
}

impl Default for HighlightRebuildParams {
    fn default() -> Self {
        Self { clip: 1.0, softness: 0.1, radius: 16.0 }
    }
}

/// The widest search, in pixels at full resolution.
pub const HLR_RADIUS_MAX: f32 = 128.0;

/// Taps per side of the separable average stop growing past this: the
/// stride grows instead, which a tent average of a smooth colour field
/// does not notice, and the cost stays flat in the radius.
// (24, in `hlr_sum!`)

pub const REF_WGSL: &str = r#"
struct HP { clip: f32, knee: f32, radius: f32, _p: f32 }
@group(0) @binding(0) var<uniform> H: HP;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(dst);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let v = textureLoad(src, vec2i(gid.xy), 0).rgb;
    let m = max(v.r, max(v.g, v.b));
    let lo = min(v.r, min(v.g, v.b));
    // A reference: every channel under the knee, nothing negative, and not
    // so dark that its ratios are noise.
    if (m < H.knee && lo >= 0.0 && m > H.clip * 0.02) {
        let w = m / max(H.knee, 1e-6);
        textureStore(dst, gid.xy, vec4f(v / m * w, w));
    } else {
        textureStore(dst, gid.xy, vec4f(0.0));
    }
}
"#;

/// The tent average along one axis; `$axis` is the step direction.
macro_rules! hlr_sum {
    ($axis:literal) => {
        concat!(
            r#"
struct HP { clip: f32, knee: f32, radius: f32, _p: f32 }
@group(0) @binding(0) var<uniform> H: HP;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = vec2i(textureDimensions(dst));
    if (gid.x >= u32(dims.x) || gid.y >= u32(dims.y)) { return; }
    let p = vec2i(gid.xy);
    let r = i32(H.radius);
    let step = max(1, r / "#,
            "24",
            r#");
    let dir = "#,
            $axis,
            r#";
    var acc = vec4f(0.0);
    var tot = 0.0;
    for (var d = -r; d <= r; d += step) {
        let t = 1.0 - f32(abs(d)) / f32(r + 1);
        tot += t;
        let q = p + dir * d;
        if (all(q >= vec2i(0)) && all(q < dims)) {
            acc += t * textureLoad(src, q, 0);
        }
    }
    textureStore(dst, gid.xy, acc / max(tot, 1e-6));
}
"#
        )
    };
}

pub const H_WGSL: &str = hlr_sum!("vec2i(1, 0)");
pub const V_WGSL: &str = hlr_sum!("vec2i(0, 1)");

pub const REBUILD_WGSL: &str = r#"
struct HP { clip: f32, knee: f32, radius: f32, _p: f32 }
@group(0) @binding(0) var<uniform> H: HP;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var refs: texture_2d<f32>;
@group(0) @binding(3) var dst: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(dst);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let p = vec2i(gid.xy);
    let o = textureLoad(src, p, 0);
    let v = o.rgb;
    // How clipped each channel is: 0 under the knee, 1 at the clip.
    let span = max(H.clip - H.knee, 1e-6);
    let w = clamp((v - vec3f(H.knee)) / span, vec3f(0.0), vec3f(1.0));
    // Nothing near the clip: the pixel exactly as it came. All three
    // clipped: nothing left to rebuild from, so the same.
    if (max(w.r, max(w.g, w.b)) <= 0.0 || min(w.r, min(w.g, w.b)) >= 1.0) {
        textureStore(dst, gid.xy, o);
        return;
    }
    let r = textureLoad(refs, p, 0);
    if (r.a <= 1e-5) {
        textureStore(dst, gid.xy, o);
        return;
    }
    let c = r.rgb / r.a;
    // Trust per channel; the trusted ones say how bright this pixel is
    // against the reference colour.
    let u = vec3f(1.0) - w;
    let den = dot(u, c);
    if (den <= 1e-5) {
        textureStore(dst, gid.xy, o);
        return;
    }
    let rebuilt = max(v, c * (dot(u, v) / den));
    textureStore(dst, gid.xy, vec4f(mix(v, rebuilt, w), o.a));
}
"#;

impl NodeDef for HighlightRebuildParams {
    const NAME: &'static str = "HighlightRebuild";
    /// Image, then the mask.
    const INPUTS: usize = 2;
    const MASK_INPUT: Option<usize> = Some(1);
    /// Everything rides in the passes' own uniforms.
    const VEC4_SLOTS: usize = 0;

    /// Constant: the four passes always run; every knob is a uniform.
    fn topo_tag(&self) -> u64 {
        0x41E8
    }

    fn pack(&self, _: &mut [f32]) {}

    fn emit(&self, e: &mut Emit) -> String {
        let v = e.stage_result();
        e.masked_stage(Self::MASK_INPUT, v)
    }
}

impl StageDef for HighlightRebuildParams {
    fn program(&self, _canvas: (u32, u32), _wired: Wired) -> StageProgram {
        let pass = |label, wgsl, binds: Vec<Bind>| Pass {
            label,
            wgsl,
            binds,
            dispatch: Dispatch::Pixels("a".into()),
            uniform_floats: 4,
        };
        let one = |from: &str, to: &str| vec![Bind::Uniform, Bind::Tex(from.into()), Bind::Storage(to.into())];
        StageProgram {
            textures: vec![TexSpec::full("b"), TexSpec::full("c")],
            passes: vec![
                pass("hlr-ref", REF_WGSL, one("a", "b")),
                pass("hlr-h", H_WGSL, one("b", "c")),
                pass("hlr-v", V_WGSL, one("c", "b")),
                pass(
                    "hlr-rebuild",
                    REBUILD_WGSL,
                    vec![Bind::Uniform, Bind::Tex("a".into()), Bind::Tex("b".into()), Bind::Storage("c".into())],
                ),
            ],
            result: "c".into(),
        }
    }

    fn uniforms(&self, _pass: usize, cx: &StageCx) -> Vec<f32> {
        let clip = self.clip.max(1e-4);
        let knee = clip * (1.0 - self.softness.clamp(0.0, 0.95));
        let radius = (self.radius * cx.proxy_scale).round().clamp(0.0, HLR_RADIUS_MAX);
        vec![clip, knee, radius, 0.0]
    }

    fn speed(&self) -> Speed {
        Speed::Slow
    }
}
