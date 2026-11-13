//! Edge Extend: pushes a matte's edge colours outward (pixel spread) or
//! pulls them in from further inside (edge expander), so an unpremultiplied
//! edge never picks up black or the old background.
//!
//! The two jobs are ONE operation with different reaches, so they are one
//! node with two modes:
//!
//! - **spread** fills everything outside the matte with the nearest edge
//!   colours, as far as the frame goes;
//! - **expand** fills only `distance` px past the edge, with `softness`.
//!
//! Both take colour from the matte's CORE — the matte eroded by `inset` px
//! — and replace everything outside it, so an inset of 2 throws away a dark
//! or light key edge and puts the colour from 2 px in there instead.
//!
//! The chain, all on the GPU:
//!
//! 1. **seed** — every pixel on the matte's 0.5 line stores the sub-pixel
//!    offset to that line (from the matte's gradient).
//! 2. **jump flood** — [`JFA_PASSES`] passes at steps 128, 64 … 1 carry the
//!    nearest edge to every pixel within [`EXTEND_MAX`], as an offset in
//!    half floats (exact near the edge, where it matters). A pass whose step
//!    is past what the params can reach is a copy, so a small inset costs
//!    copies, not searches — and the pass count never depends on a slider.
//! 3. **prep** — the core's straight colour weighted by its coverage,
//!    `(c·w, w)`: level 0 of a push-pull pyramid.
//! 4. **push-pull** — box sums down to one pixel (weights clamped at 1, so a
//!    sparse level still counts as filled), then each level fills its holes
//!    from the bilinear level above. That is a smooth, streak-free fill of
//!    the whole frame for ~3 full-res passes of work.
//! 5. **comp** — the signed distance from the jump flood decides, per pixel,
//!    how much of the fill replaces the original, and what alpha comes out.
//!
//! A matte on port 1 replaces the input's alpha as the edge (the colour is
//! then read straight). Every param is a uniform; only wiring the matte
//! changes the chain.

use crate::codegen::Emit;
use crate::cost::Speed;
use crate::nodes::stage::{Bind, Dispatch, Pass, StageCx, StageDef, StageProgram, TexSpec, Wired};
use crate::nodes::NodeDef;

/// Furthest reach in px of `inset + distance + softness`: the jump flood's
/// first step is half of it.
pub const EXTEND_MAX: f32 = 250.0;
/// Jump-flood passes: steps 128 down to 1.
pub const JFA_PASSES: usize = 8;
/// The matte's port.
pub const MATTE_INPUT: usize = 1;
/// Push-pull levels are capped here (a 16k frame).
const MAX_LEVELS: usize = 14;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ExtendMode {
    /// Fill the whole frame outside the matte.
    #[default]
    Spread,
    /// Fill `distance` px past the edge.
    Expand,
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct EdgeExtendParams {
    pub mode: ExtendMode,
    /// Px the colour reaches past the edge (expand mode). 0 still covers
    /// the matte's anti-aliased rim.
    pub distance: f32,
    /// Px inside the edge the colour is taken from: the edge band this
    /// wide is replaced too. 0 keeps everything inside the matte.
    pub inset: f32,
    /// Px of ramp on both boundaries (the core's and the reach's).
    pub softness: f32,
    /// The input's colour is premultiplied by its alpha. Ignored with a
    /// matte wired: the colour is then straight.
    pub premultiplied: bool,
    /// Alpha out is the input's; off, it is the extended coverage.
    pub keep_alpha: bool,
    /// Multiply the result by the alpha that comes out.
    pub premult_out: bool,
}

impl Default for EdgeExtendParams {
    fn default() -> Self {
        Self {
            mode: ExtendMode::Spread,
            distance: 8.0,
            inset: 0.0,
            softness: 1.0,
            premultiplied: true,
            keep_alpha: true,
            premult_out: false,
        }
    }
}

impl EdgeExtendParams {
    /// How far the jump flood must carry an edge for these params, px.
    pub fn reach(&self, scale: f32) -> f32 {
        let out = match self.mode {
            ExtendMode::Spread => 0.0,
            ExtendMode::Expand => self.distance.max(0.0) + 1.0,
        };
        ((self.inset.max(0.0) + out + 2.0 * self.softness.max(0.0)) * scale + 2.0).min(EXTEND_MAX + 2.0)
    }
}

/// The matte at a pixel, read off the input's alpha or the wired matte
/// (bound as `mt`; unwired, `mt` is the input itself).
macro_rules! mat_alpha {
    () => {
        "const MATTED: bool = false;\nfn mat(p: vec2i) -> f32 { return clamp(textureLoad(mt, p, 0).a, 0.0, 1.0); }\n"
    };
}
macro_rules! mat_wired {
    () => {
        "const MATTED: bool = true;\nfn mat(p: vec2i) -> f32 { return clamp(textureLoad(mt, p, 0).x, 0.0, 1.0); }\n"
    };
}

macro_rules! seed_head {
    () => {
        r#"
@group(0) @binding(0) var mt: texture_2d<f32>;
@group(0) @binding(1) var dst: texture_storage_2d<rgba16float, write>;
"#
    };
}

macro_rules! seed_body {
    () => {
        r#"
fn m_at(p: vec2i, dims: vec2i) -> f32 { return mat(clamp(p, vec2i(0), dims - 1)); }

// A pixel on the 0.5 line (it and a 4-neighbour fall on different sides)
// stores the offset from its centre to the line, along the gradient.
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = vec2i(textureDimensions(dst));
    if (gid.x >= u32(dims.x) || gid.y >= u32(dims.y)) { return; }
    let p = vec2i(gid.xy);
    let m = m_at(p, dims);
    let l = m_at(p - vec2i(1, 0), dims);
    let r = m_at(p + vec2i(1, 0), dims);
    let u = m_at(p - vec2i(0, 1), dims);
    let d = m_at(p + vec2i(0, 1), dims);
    let inside = m >= 0.5;
    let edge = (l >= 0.5) != inside || (r >= 0.5) != inside || (u >= 0.5) != inside
        || (d >= 0.5) != inside;
    if (!edge) {
        textureStore(dst, gid.xy, vec4f(0.0));
        return;
    }
    let g = vec2f(r - l, d - u) * 0.5;
    let gl = length(g);
    var off = vec2f(0.0);
    if (gl > 1e-4) {
        off = g / gl * clamp((0.5 - m) / gl, -1.0, 1.0);
    }
    textureStore(dst, gid.xy, vec4f(off, 1.0, 0.0));
}
"#
    };
}

pub const SEED_WGSL: &str = concat!(seed_head!(), mat_alpha!(), seed_body!());
pub const SEED_MATTE_WGSL: &str = concat!(seed_head!(), mat_wired!(), seed_body!());

/// One jump-flood step. `active` 0 copies.
pub const JFA_WGSL: &str = r#"
struct JU { step: f32, live: f32, _p0: f32, _p1: f32 }
@group(0) @binding(0) var<uniform> U: JU;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = vec2i(textureDimensions(dst));
    if (gid.x >= u32(dims.x) || gid.y >= u32(dims.y)) { return; }
    let p = vec2i(gid.xy);
    let own = textureLoad(src, p, 0);
    if (U.live == 0.0) {
        textureStore(dst, gid.xy, own);
        return;
    }
    var best = own.xy;
    var found = own.z > 0.5;
    var bd = select(1e12, dot(own.xy, own.xy), found);
    let s = i32(U.step);
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) { continue; }
            let q = p + vec2i(dx, dy) * s;
            if (any(q < vec2i(0)) || any(q >= dims)) { continue; }
            let t = textureLoad(src, q, 0);
            if (t.z < 0.5) { continue; }
            let o = vec2f(q - p) + t.xy;
            let dd = dot(o, o);
            if (dd < bd) {
                bd = dd;
                best = o;
                found = true;
            }
        }
    }
    textureStore(dst, gid.xy, select(vec4f(0.0), vec4f(best, 1.0, 0.0), found));
}
"#;

/// Signed distance to the 0.5 line, + inside; far (1e4) off the flood.
macro_rules! depth_fn {
    () => {
        r#"
fn depth(p: vec2i) -> f32 {
    let j = textureLoad(jfa, p, 0);
    let d = select(1e4, length(j.xy), j.z > 0.5);
    return select(-d, d, mat(p) >= 0.5);
}
"#
    };
}

macro_rules! prep_head {
    () => {
        r#"
struct PU { inset: f32, premult: f32, _p0: f32, _p1: f32 }
@group(0) @binding(0) var<uniform> U: PU;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var mt: texture_2d<f32>;
@group(0) @binding(3) var jfa: texture_2d<f32>;
@group(0) @binding(4) var dst: texture_storage_2d<rgba16float, write>;
"#
    };
}

macro_rules! prep_body {
    () => {
        r#"
// The core's colour weighted by its coverage: (straight · w, w). The core
// is the matte from `inset` px in, with a one-pixel ramp.
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(dst);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let p = vec2i(gid.xy);
    let c = textureLoad(src, p, 0);
    let m = mat(p);
    let k = clamp(depth(p) - U.inset + 0.5, 0.0, 1.0);
    let pm = U.premult != 0.0 && !MATTED;
    // Premultiplied colour is already straight × m: no division.
    let cw = select(c.rgb * m, c.rgb, pm) * k;
    textureStore(dst, gid.xy, vec4f(cw, m * k));
}
"#
    };
}

pub const PREP_WGSL: &str = concat!(prep_head!(), mat_alpha!(), depth_fn!(), prep_body!());
pub const PREP_MATTE_WGSL: &str = concat!(prep_head!(), mat_wired!(), depth_fn!(), prep_body!());

/// Push-pull, down: the 2x2 box SUM, its weight clamped at 1.
pub const DOWN_WGSL: &str = r#"
@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(dst);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let uv = (vec2f(gid.xy) + 0.5) / vec2f(dims);
    var s = textureSampleLevel(src, samp, uv, 0.0) * 4.0;
    if (s.a > 1.0) { s = s / s.a; }
    textureStore(dst, gid.xy, s);
}
"#;

/// Push-pull, up: this level, its holes filled from the level above.
pub const UP_WGSL: &str = r#"
@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var fine: texture_2d<f32>;
@group(0) @binding(2) var coarse: texture_2d<f32>;
@group(0) @binding(3) var dst: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(dst);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let uv = (vec2f(gid.xy) + 0.5) / vec2f(dims);
    let f = textureLoad(fine, gid.xy, 0);
    let c = textureSampleLevel(coarse, samp, uv, 0.0);
    textureStore(dst, gid.xy, f + (1.0 - clamp(f.a, 0.0, 1.0)) * c);
}
"#;

macro_rules! comp_head {
    () => {
        r#"
struct CU {
    inset: f32, soft: f32, distance: f32, spread: f32,
    keep_alpha: f32, premult: f32, premult_out: f32, _p: f32,
}
@group(0) @binding(0) var<uniform> U: CU;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var src: texture_2d<f32>;
@group(0) @binding(3) var mt: texture_2d<f32>;
@group(0) @binding(4) var jfa: texture_2d<f32>;
@group(0) @binding(5) var lvl0: texture_2d<f32>;
@group(0) @binding(6) var coarse: texture_2d<f32>;
@group(0) @binding(7) var dst: texture_storage_2d<rgba16float, write>;
"#
    };
}

macro_rules! comp_body {
    () => {
        r#"
// 0 below `a`, 1 from `a + w`, smooth between; a step at w = 0.
fn rise(x: f32, a: f32, w: f32) -> f32 {
    if (w < 1e-3) { return select(0.0, 1.0, x >= a); }
    return smoothstep(a, a + w, x);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(dst);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let p = vec2i(gid.xy);
    let uv = (vec2f(gid.xy) + 0.5) / vec2f(dims);
    let c = textureLoad(src, p, 0);
    let m = mat(p);
    let pm = U.premult != 0.0 && !MATTED;
    let straight = select(c.rgb, select(c.rgb, c.rgb / m, m > 1e-4), pm);
    // Level 0 of the push-pull, filled from the level above.
    let f0 = textureLoad(lvl0, p, 0);
    let pu = f0 + (1.0 - clamp(f0.a, 0.0, 1.0)) * textureSampleLevel(coarse, samp, uv, 0.0);
    let filled = select(straight, pu.rgb / pu.a, pu.a > 1e-5);
    let e = depth(p);
    // Inside: kept from `inset + soft` in, replaced up to `inset`.
    let inner = 1.0 - rise(e, U.inset, U.soft);
    // Outside: the whole frame, or `distance` px past the anti-aliased rim.
    var reach = 1.0;
    if (U.spread == 0.0) { reach = 1.0 - rise(-e, U.distance + 1.0, U.soft); }
    let w = inner * reach;
    let rgb = mix(straight, filled, w);
    let a = select(max(m, reach), c.a, U.keep_alpha != 0.0);
    let res = select(rgb, rgb * a, U.premult_out != 0.0);
    textureStore(dst, gid.xy, vec4f(res, a));
}
"#
    };
}

pub const COMP_WGSL: &str = concat!(comp_head!(), mat_alpha!(), depth_fn!(), comp_body!());
pub const COMP_MATTE_WGSL: &str = concat!(comp_head!(), mat_wired!(), depth_fn!(), comp_body!());

/// Every module, for the startup compile.
pub const ALL_WGSL: [&str; 9] = [
    SEED_WGSL,
    SEED_MATTE_WGSL,
    JFA_WGSL,
    PREP_WGSL,
    PREP_MATTE_WGSL,
    DOWN_WGSL,
    UP_WGSL,
    COMP_WGSL,
    COMP_MATTE_WGSL,
];

/// Push-pull levels below full size: down to one pixel on the long side.
pub fn levels(w: u32, h: u32) -> usize {
    let mut n = 0;
    while n < MAX_LEVELS && ((w.max(h)) >> n) > 1 {
        n += 1;
    }
    n.max(1)
}

/// The jump flood's step for pass `i`.
fn jfa_step(i: usize) -> f32 {
    (1u32 << (JFA_PASSES - 1 - i)) as f32
}

impl NodeDef for EdgeExtendParams {
    const NAME: &'static str = "EdgeExtend";
    /// The picture, the matte, then the mask.
    const INPUTS: usize = 3;
    const MASK_INPUT: Option<usize> = Some(2);
    const VEC4_SLOTS: usize = 0;

    /// Constant: every param is a uniform, mode included.
    fn topo_tag(&self) -> u64 {
        0xEE7D
    }

    fn pack(&self, _: &mut [f32]) {}

    fn emit(&self, e: &mut Emit) -> String {
        let v = e.stage_result();
        e.masked_stage(Self::MASK_INPUT, v)
    }
}

impl StageDef for EdgeExtendParams {
    /// The matte, only when wired.
    fn aux_ports(&self, wired: Wired) -> Vec<usize> {
        if wired.has(MATTE_INPUT) {
            vec![MATTE_INPUT]
        } else {
            Vec::new()
        }
    }

    fn program(&self, canvas: (u32, u32), wired: Wired) -> StageProgram {
        let matte = wired.has(MATTE_INPUT);
        let mt = if matte { "aux0" } else { "a" };
        let t = |s: &str| Bind::Tex(s.into());
        let st = |s: &str| Bind::Storage(s.into());
        let n = levels(canvas.0, canvas.1);
        let lvl = |i: usize| format!("d{i}");
        let up = |i: usize| format!("u{i}");
        let mut textures = vec![TexSpec::full("j0"), TexSpec::full("j1"), TexSpec::full("b")];
        textures.push(TexSpec::full(&lvl(0)));
        for i in 1..=n {
            textures.push(TexSpec::level(&lvl(i), i as u32));
        }
        for i in 1..n {
            textures.push(TexSpec::level(&up(i), i as u32));
        }
        let mut passes = vec![Pass {
            label: "extend-seed",
            wgsl: if matte { SEED_MATTE_WGSL } else { SEED_WGSL },
            binds: vec![t(mt), st("j0")],
            dispatch: Dispatch::Pixels("j0".into()),
            uniform_floats: 0,
        }];
        let jn = |i: usize| if i % 2 == 0 { "j0" } else { "j1" };
        for i in 0..JFA_PASSES {
            passes.push(Pass {
                label: "extend-jfa",
                wgsl: JFA_WGSL,
                binds: vec![Bind::Uniform, t(jn(i)), st(jn(i + 1))],
                dispatch: Dispatch::Pixels("j0".into()),
                uniform_floats: 4,
            });
        }
        let jfa = jn(JFA_PASSES);
        passes.push(Pass {
            label: "extend-prep",
            wgsl: if matte { PREP_MATTE_WGSL } else { PREP_WGSL },
            binds: vec![Bind::Uniform, t("a"), t(mt), t(jfa), st(&lvl(0))],
            dispatch: Dispatch::Pixels(lvl(0)),
            uniform_floats: 4,
        });
        for i in 1..=n {
            passes.push(Pass {
                label: "extend-down",
                wgsl: DOWN_WGSL,
                binds: vec![Bind::Sampler, t(&lvl(i - 1)), st(&lvl(i))],
                dispatch: Dispatch::Pixels(lvl(i)),
                uniform_floats: 0,
            });
        }
        // u_n is d_n: the top level has nothing above it.
        let filled = |i: usize| if i == n { lvl(n) } else { up(i) };
        for i in (1..n).rev() {
            passes.push(Pass {
                label: "extend-up",
                wgsl: UP_WGSL,
                binds: vec![Bind::Sampler, t(&lvl(i)), t(&filled(i + 1)), st(&up(i))],
                dispatch: Dispatch::Pixels(up(i)),
                uniform_floats: 0,
            });
        }
        passes.push(Pass {
            label: "extend-comp",
            wgsl: if matte { COMP_MATTE_WGSL } else { COMP_WGSL },
            binds: vec![
                Bind::Uniform,
                Bind::Sampler,
                t("a"),
                t(mt),
                t(jfa),
                t(&lvl(0)),
                t(&filled(1)),
                st("b"),
            ],
            dispatch: Dispatch::Pixels("b".into()),
            uniform_floats: 8,
        });
        StageProgram { textures, passes, result: "b".into() }
    }

    fn uniforms(&self, pass: usize, cx: &StageCx) -> Vec<f32> {
        let s = cx.proxy_scale;
        let flag = |b: bool| b as u32 as f32;
        let inset = (self.inset.max(0.0) * s).min(EXTEND_MAX);
        if (1..=JFA_PASSES).contains(&pass) {
            let step = jfa_step(pass - 1);
            // A step at or past twice the reach finds nothing a smaller
            // one will not: copy instead of searching.
            return vec![step, flag(step < 2.0 * self.reach(s)), 0.0, 0.0];
        }
        if pass == JFA_PASSES + 1 {
            return vec![inset, flag(self.premultiplied), 0.0, 0.0];
        }
        if pass + 1 == cx.passes {
            let soft = (self.softness.max(0.0) * s).min(EXTEND_MAX);
            let dist = (self.distance.max(0.0) * s).min(EXTEND_MAX);
            return vec![
                inset,
                soft,
                dist,
                flag(self.mode == ExtendMode::Spread),
                flag(self.keep_alpha),
                flag(self.premultiplied),
                flag(self.premult_out),
                0.0,
            ];
        }
        Vec::new()
    }

    fn speed(&self) -> Speed {
        Speed::Slow
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_matte_costs_nothing_unwired_and_the_chain_is_whole() {
        let p = EdgeExtendParams::default();
        assert!(p.aux_ports(Wired(&[true, false, false])).is_empty());
        assert_eq!(p.aux_ports(Wired(&[true, true, false])), vec![MATTE_INPUT]);
        let plain = p.program((256, 128), Wired(&[true, false, false]));
        let n = levels(256, 128);
        assert_eq!(n, 8);
        // seed, flood, prep, down, up, comp.
        assert_eq!(plain.passes.len(), 1 + JFA_PASSES + 1 + n + (n - 1) + 1);
        assert!(!plain.passes.iter().any(|p| p.binds.contains(&Bind::Tex("aux0".into()))));
        let m = p.program((256, 128), Wired(&[true, true, false]));
        assert!(m.passes[0].binds.contains(&Bind::Tex("aux0".into())));
        // Every texture a pass names is declared, or is the input.
        for pr in [&plain, &m] {
            for pass in &pr.passes {
                for b in &pass.binds {
                    if let Bind::Tex(s) | Bind::Storage(s) = b {
                        assert!(
                            s == "a" || s == "aux0" || pr.textures.iter().any(|t| &t.name == s),
                            "{} names {s}",
                            pass.label
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn uniforms_are_sized_to_their_passes() {
        let p = EdgeExtendParams { mode: ExtendMode::Expand, inset: 3.0, ..Default::default() };
        let pr = p.program((64, 64), Wired(&[true, false, false]));
        let cx = StageCx { canvas: (64, 64), proxy_scale: 1.0, frame: 0.0, fps: 24.0, passes: pr.passes.len() };
        for (i, pass) in pr.passes.iter().enumerate() {
            assert_eq!(p.uniforms(i, &cx).len(), pass.uniform_floats, "{}", pass.label);
        }
        // Reach 3 + 9 + 2 + 2 = 16: steps 128 and 64 copy, 16 searches.
        assert_eq!(p.uniforms(1, &cx)[1], 0.0);
        assert_eq!(p.uniforms(4, &cx), vec![16.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn every_module_validates() {
        for (i, w) in ALL_WGSL.iter().enumerate() {
            let m = naga::front::wgsl::parse_str(w).unwrap_or_else(|e| panic!("module {i}: {e}"));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                .validate(&m)
                .unwrap_or_else(|e| panic!("module {i}: {e:?}"));
        }
    }
}
