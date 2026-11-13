//! Contrast: the tone control a colourist actually reaches for, on its own
//! node — a slope about a pivot, shaped, with an optional local-contrast
//! (clarity) term. The Grade's contrast is the one-knob version; this is the
//! one with the handles.
//!
//! **The core.** Every channel (or the luminance, see below) is measured as
//! a signed distance `t` from the pivot, the curve is applied to `t`, and
//! the result is put back:
//!
//! ```text
//! log:     t = log2(v / pivot)       v' = pivot · 2^curve(t)     (stops)
//! linear:  t = (v - pivot) / pivot   v' = pivot · (1 + curve(t))
//! ```
//!
//! Log is the default and the one that behaves: a slope in stops is
//! exposure-invariant, which is why a scene-linear comp grades contrast in
//! log. In log, zero and negatives are passed through untouched (they have
//! no stop to be measured in). Linear is the flat-timeline contrast, a
//! straight line through the pivot; it moves negatives like anything else.
//!
//! **The curve** is a slope, per side, eased by a soft toe and shoulder:
//!
//! ```text
//! c      = contrast · (shadows if t < 0 else highlights)
//! f(t)   = mix(t, R·tanh(t/R), soft)       soft = toe or shoulder
//! curve  = t + (c - 1) · f(t)
//! ```
//!
//! At softness 0 it is a pure slope (a power law about the pivot in log).
//! At softness 1 the slope is `c` at the pivot and eases back to 1 — the
//! picture's own gradation — over `range` (R, in stops): the extremes are
//! moved but not crushed, which is what a soft toe and a soft shoulder are
//! for. The slope is `1 + (c - 1)·f'` with `f'` in (0, 1], so for any
//! `c > 0` the curve is monotonic: contrast never inverts a tone.
//!
//! **Colour.** Per channel (`preserve` 0) runs the curve on r, g and b
//! independently: the film-like contrast, which saturates as it steepens.
//! Luma (`preserve` 1) runs it on the working luminance and scales rgb by
//! the ratio, so hue and saturation hold exactly. `preserve` is a uniform
//! and blends between the two.
//!
//! **Local contrast** (`local`, structural) adds a clarity term BEFORE the
//! curve: the luminance's distance in stops from a blurred base, scaled by
//! `clarity` and soft-limited to `limit` stops so a hard edge does not
//! halo. The base is a log-domain blur — each channel's log2 averaged down
//! a pyramid and the two octaves bracketing `radius` blended back up — so a
//! bright window does not drag the base of the wall beside it up the way a
//! linear blur would. That pyramid is a STAGE (its own passes, a red ring,
//! manifesto rule 8); with `local` off the node fuses like any colour node
//! and costs a handful of ALU ops.
//!
//! `mix` dissolves back to the input and the mask crossfades the same way
//! every colour node's does. Alpha is never touched. [`ContrastParams::
//! apply`] and [`ContrastParams::apply_local`] are the CPU mirrors the tests
//! check the GPU against.

use crate::codegen::Emit;
use crate::cost::Speed;
use crate::nodes::stage::{Bind, Dispatch, Pass, StageCx, StageDef, StageProgram, TexSpec, Wired};
use crate::nodes::NodeDef;

/// Channel-name prefix for every keyable number (`contrast.<name>`).
pub const PREFIX: &str = "contrast.";

const TOPO_BASE: u64 = 0xC047_0000;

/// Where the slope is measured.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ContrastDomain {
    /// Stops about the pivot. Exposure-invariant; the default.
    #[default]
    Log,
    /// A straight line through the pivot, in scene-linear.
    Linear,
}

impl ContrastDomain {
    pub const ALL: [ContrastDomain; 2] = [ContrastDomain::Log, ContrastDomain::Linear];
    pub fn label(self) -> &'static str {
        match self {
            ContrastDomain::Log => "log (stops)",
            ContrastDomain::Linear => "linear",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct ContrastParams {
    /// The slope at the pivot. 1 changes nothing; above steepens.
    pub contrast: f32,
    /// The scene-linear value that stays put. 0.18 is mid grey.
    pub pivot: f32,
    /// Multiplies the slope below the pivot.
    pub shadows: f32,
    /// Multiplies the slope above the pivot.
    pub highlights: f32,
    /// 0 a straight slope below the pivot; 1 eases back to the picture's
    /// own gradation over `range`.
    pub toe: f32,
    /// The same, above the pivot.
    pub shoulder: f32,
    /// How far from the pivot the toe and shoulder ease, in stops (log) or
    /// pivots (linear).
    pub range: f32,
    /// 0 per channel (saturates), 1 on luminance (holds hue and saturation).
    pub preserve: f32,
    pub domain: ContrastDomain,
    /// Local contrast on: a stage, so this one recompiles.
    pub local: bool,
    /// Local contrast amount; negative softens.
    pub clarity: f32,
    /// The local base's blur radius, in pixels.
    pub radius: f32,
    /// The most local detail can move a pixel, in stops (soft limit).
    pub limit: f32,
    /// 0 is the input, 1 the full effect.
    pub mix: f32,
}

impl Default for ContrastParams {
    fn default() -> Self {
        Self {
            contrast: 1.0,
            pivot: 0.18,
            shadows: 1.0,
            highlights: 1.0,
            toe: 0.0,
            shoulder: 0.0,
            range: 3.0,
            preserve: 1.0,
            domain: ContrastDomain::Log,
            local: false,
            clarity: 0.3,
            radius: 40.0,
            limit: 1.0,
            mix: 1.0,
        }
    }
}

/// The keyable numbers, by channel name after [`PREFIX`].
const NAMES: &[&str] = &[
    "amount", "pivot", "shadows", "highlights", "toe", "shoulder", "range", "preserve", "clarity",
    "radius", "limit", "mix",
];

/// Floor for a log taken of a pixel: a black or negative channel reads as
/// this, a little over 13 stops under 1.
const LOG_FLOOR: f32 = 1e-4;

impl ContrastParams {
    /// Every keyable number's name (after [`PREFIX`]).
    pub fn names() -> &'static [&'static str] {
        NAMES
    }

    fn field(&mut self, name: &str) -> Option<&mut f32> {
        Some(match name {
            "amount" => &mut self.contrast,
            "pivot" => &mut self.pivot,
            "shadows" => &mut self.shadows,
            "highlights" => &mut self.highlights,
            "toe" => &mut self.toe,
            "shoulder" => &mut self.shoulder,
            "range" => &mut self.range,
            "preserve" => &mut self.preserve,
            "clarity" => &mut self.clarity,
            "radius" => &mut self.radius,
            "limit" => &mut self.limit,
            "mix" => &mut self.mix,
            _ => return None,
        })
    }

    pub fn get(&self, name: &str) -> Option<f32> {
        let mut c = *self;
        c.field(name).map(|v| *v)
    }

    pub fn set(&mut self, name: &str, v: f32) -> bool {
        match self.field(name) {
            Some(f) => {
                *f = v;
                true
            }
            None => false,
        }
    }

    fn fin(v: f32, or: f32) -> f32 {
        if v.is_finite() { v } else { or }
    }

    /// The curve on a signed distance from the pivot.
    pub fn curve(&self, t: f32) -> f32 {
        let below = t < 0.0;
        let side = if below { self.shadows } else { self.highlights };
        let c = (Self::fin(self.contrast, 1.0) * Self::fin(side, 1.0)).max(0.0);
        let soft = Self::fin(if below { self.toe } else { self.shoulder }, 0.0).clamp(0.0, 1.0);
        let r = Self::fin(self.range, 3.0).max(1e-3);
        let f = t + (r * (t / r).clamp(-10.0, 10.0).tanh() - t) * soft;
        t + (c - 1.0) * f
    }

    fn pivot(&self) -> f32 {
        Self::fin(self.pivot, 0.18).max(1e-6)
    }

    /// One value through the domain and the curve.
    pub fn tone(&self, v: f32) -> f32 {
        let p = self.pivot();
        match self.domain {
            ContrastDomain::Linear => p + p * self.curve((v - p) / p),
            ContrastDomain::Log => {
                if v <= 0.0 {
                    v
                } else {
                    p * self.curve((v / p).log2()).exp2()
                }
            }
        }
    }

    /// The tone curve on one pixel, before the mix: per channel and on
    /// luminance, blended by `preserve`. `w` is the working luma weights.
    fn toned(&self, c: [f32; 3], w: [f32; 3]) -> [f32; 3] {
        let pc = c.map(|v| self.tone(v));
        let y = c[0] * w[0] + c[1] * w[1] + c[2] * w[2];
        let lr = if self.domain == ContrastDomain::Log && y <= 0.0 {
            c
        } else if y.abs() > 1e-6 {
            let k = self.tone(y) / y;
            c.map(|v| v * k)
        } else {
            let d = self.tone(y) - y;
            c.map(|v| v + d)
        };
        let s = Self::fin(self.preserve, 1.0).clamp(0.0, 1.0);
        std::array::from_fn(|i| pc[i] + (lr[i] - pc[i]) * s)
    }

    fn mix_amount(&self) -> f32 {
        Self::fin(self.mix, 1.0)
    }

    /// The CPU mirror of the fused code with local contrast OFF (unmasked).
    pub fn apply(&self, c: [f32; 4], w: [f32; 3]) -> [f32; 4] {
        self.finish(c, [c[0], c[1], c[2]], w)
    }

    /// The clarity term on one pixel, given its base: the log2 of each
    /// channel averaged over the neighbourhood (what the stage leaves).
    pub fn clarify(&self, c: [f32; 3], base_log: [f32; 3], w: [f32; 3]) -> [f32; 3] {
        let k = Self::fin(self.clarity, 0.0);
        let y = c[0] * w[0] + c[1] * w[1] + c[2] * w[2];
        if y <= 1e-6 || k == 0.0 {
            return c;
        }
        let b = base_log.map(|l| l.exp2());
        let by = (b[0] * w[0] + b[1] * w[1] + b[2] * w[2]).max(1e-6);
        let lim = Self::fin(self.limit, 1.0).max(1e-3);
        let d = y.max(LOG_FLOOR).log2() - by.log2();
        let d = d / (1.0 + d.abs() / lim);
        let g = (k * d).exp2();
        c.map(|v| v * g)
    }

    /// The CPU mirror with local contrast ON, given the pixel's base.
    pub fn apply_local(&self, c: [f32; 4], base_log: [f32; 3], w: [f32; 3]) -> [f32; 4] {
        let l = self.clarify([c[0], c[1], c[2]], base_log, w);
        self.finish(c, l, w)
    }

    fn finish(&self, c: [f32; 4], pre: [f32; 3], w: [f32; 3]) -> [f32; 4] {
        let t = self.toned(pre, w);
        let m = self.mix_amount();
        [c[0] + (t[0] - c[0]) * m, c[1] + (t[1] - c[1]) * m, c[2] + (t[2] - c[2]) * m, c[3]]
    }

    /// The two octaves bracketing the radius, blended, for a pyramid of
    /// `n` levels. Level i is the canvas at 1/2^(i+1), whose texels are a
    /// blur of about 2^(i+1) pixels.
    pub fn level_weights(radius: f32, n: usize) -> Vec<f32> {
        let mut w = vec![0.0; n.max(1)];
        let reach = (Self::fin(radius, 40.0).max(1.0).log2() - 1.0).clamp(0.0, (w.len() - 1) as f32);
        let i = reach.floor() as usize;
        let f = reach - i as f32;
        w[i] = 1.0 - f;
        if i + 1 < w.len() {
            w[i + 1] = f;
        } else {
            w[i] = 1.0;
        }
        w
    }
}

/// Levels the base pyramid runs: enough to reach a radius of ~1000 px.
const MAX_LEVELS: usize = 9;

pub fn levels(w: u32, h: u32) -> usize {
    let mut n = 0;
    while n < MAX_LEVELS && (w >> (n + 1)) >= 1 && (h >> (n + 1)) >= 1 {
        n += 1;
    }
    n.max(1)
}

/// 13-tap downsample. The first level takes each channel's log2 first, so
/// the whole pyramid averages in stops.
pub const DOWN_WGSL: &str = r#"
struct CD { first: f32, _p0: f32, _p1: f32, _p2: f32 }
@group(0) @binding(0) var<uniform> P: CD;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var src: texture_2d<f32>;
@group(0) @binding(3) var dst: texture_storage_2d<rgba16float, write>;

fn tap(uv: vec2f) -> vec3f {
    let c = textureSampleLevel(src, samp, uv, 0.0).rgb;
    if (P.first != 0.0) {
        return log2(clamp(c, vec3f(1e-4), vec3f(65504.0)));
    }
    return c;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(dst);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let uv = (vec2f(gid.xy) + 0.5) / vec2f(dims);
    let ts = 1.0 / vec2f(textureDimensions(src));
    var o = tap(uv) * 0.125;
    o += (tap(uv + ts * vec2f(-2.0,  2.0)) + tap(uv + ts * vec2f( 2.0,  2.0))
        + tap(uv + ts * vec2f(-2.0, -2.0)) + tap(uv + ts * vec2f( 2.0, -2.0))) * 0.03125;
    o += (tap(uv + ts * vec2f( 0.0,  2.0)) + tap(uv + ts * vec2f(-2.0,  0.0))
        + tap(uv + ts * vec2f( 2.0,  0.0)) + tap(uv + ts * vec2f( 0.0, -2.0))) * 0.0625;
    o += (tap(uv + ts * vec2f(-1.0,  1.0)) + tap(uv + ts * vec2f( 1.0,  1.0))
        + tap(uv + ts * vec2f(-1.0, -1.0)) + tap(uv + ts * vec2f( 1.0, -1.0))) * 0.125;
    textureStore(dst, gid.xy, vec4f(o, 1.0));
}
"#;

/// The blended base up to the full canvas, bilinear: it is a blur already.
pub const FINAL_WGSL: &str = r#"
@group(0) @binding(0) var samp: sampler;
@group(0) @binding(1) var src: texture_2d<f32>;
@group(0) @binding(2) var dst: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let dims = textureDimensions(dst);
    if (gid.x >= dims.x || gid.y >= dims.y) { return; }
    let uv = (vec2f(gid.xy) + 0.5) / vec2f(dims);
    textureStore(dst, gid.xy, textureSampleLevel(src, samp, uv, 0.0));
}
"#;

impl NodeDef for ContrastParams {
    const NAME: &'static str = "Contrast";
    /// Image, then the mask.
    const INPUTS: usize = 2;
    const MASK_INPUT: Option<usize> = Some(1);
    /// (contrast, pivot, shadows, highlights), (toe, shoulder, range,
    /// preserve), (domain, mix, clarity, limit).
    const VEC4_SLOTS: usize = 3;

    /// Only `local` changes the code (it adds the stage); the domain is a
    /// uniform `select`, so even that never recompiles.
    fn topo_tag(&self) -> u64 {
        TOPO_BASE + self.local as u64
    }

    fn pack(&self, v: &mut [f32]) {
        v[0] = Self::fin(self.contrast, 1.0);
        v[1] = self.pivot();
        v[2] = Self::fin(self.shadows, 1.0);
        v[3] = Self::fin(self.highlights, 1.0);
        v[4] = Self::fin(self.toe, 0.0).clamp(0.0, 1.0);
        v[5] = Self::fin(self.shoulder, 0.0).clamp(0.0, 1.0);
        v[6] = Self::fin(self.range, 3.0).max(1e-3);
        v[7] = Self::fin(self.preserve, 1.0).clamp(0.0, 1.0);
        v[8] = (self.domain == ContrastDomain::Linear) as u32 as f32;
        v[9] = self.mix_amount();
        v[10] = Self::fin(self.clarity, 0.0);
        v[11] = Self::fin(self.limit, 1.0).max(1e-3);
    }

    fn emit(&self, e: &mut Emit) -> String {
        let slot = e.slot();
        let w = e.working().primaries.luma_weights();
        let wv = format!("vec3f({:.7}, {:.7}, {:.7})", w[0], w[1], w[2]);
        let (_, fns) = e.color_helpers();
        // The luma weights are baked: one working space per comp, and a
        // working-space change already moves the topology hash.
        if !fns.contains_key("contrast_tone") {
            fns.insert("contrast_tone".into(), tone_wgsl(&wv));
        }
        if self.local && !fns.contains_key("contrast_clarify") {
            fns.insert("contrast_clarify".into(), clarify_wgsl(&wv));
        }
        let input = e.input(0);
        let pre = if self.local {
            let base = e.stage_result();
            let l = e.fresh();
            e.line(format_args!("let {l} = contrast_clarify({input}, {base}, {slot}u);"));
            l
        } else {
            input.clone()
        };
        let v = e.fresh();
        e.line(format_args!(
            "let {v} = vec4f(mix({input}.rgb, contrast_tone({pre}.rgb, {slot}u), P.v[{}].y), {input}.a);",
            slot + 2
        ));
        e.masked(Self::MASK_INPUT, &input, v)
    }
}

fn tone_wgsl(wv: &str) -> String {
    format!(
        r#"
// Contrast: a slope about a pivot, per side, eased by a soft toe and
// shoulder, in stops (log) or in pivots (linear); per channel and on
// luminance, blended by `preserve`. slot+0 = (contrast, pivot, shadows,
// highlights), slot+1 = (toe, shoulder, range, preserve), slot+2.x = linear.
fn contrast_curve(t: f32, k: vec4f, s: vec4f) -> f32 {{
    let below = t < 0.0;
    let c = max(k.x * select(k.w, k.z, below), 0.0);
    let soft = select(s.y, s.x, below);
    let r = s.z;
    let f = mix(t, r * tanh(clamp(t / r, -10.0, 10.0)), soft);
    return t + (c - 1.0) * f;
}}
fn contrast_v(v: f32, k: vec4f, s: vec4f, lin: bool) -> f32 {{
    let p = k.y;
    if (lin) {{ return p + p * contrast_curve((v - p) / p, k, s); }}
    if (v <= 0.0) {{ return v; }}
    return p * exp2(contrast_curve(log2(v / p), k, s));
}}
fn contrast_tone(c: vec3f, slot: u32) -> vec3f {{
    let k = P.v[slot];
    let s = P.v[slot + 1u];
    let lin = P.v[slot + 2u].x > 0.5;
    let pc = vec3f(contrast_v(c.r, k, s, lin), contrast_v(c.g, k, s, lin), contrast_v(c.b, k, s, lin));
    let y = dot(c, {wv});
    var lr = c;
    if (lin || y > 0.0) {{
        let y2 = contrast_v(y, k, s, lin);
        if (abs(y) > 1e-6) {{ lr = c * (y2 / y); }} else {{ lr = c + vec3f(y2 - y); }}
    }}
    return mix(pc, lr, s.w);
}}
"#
    )
}

fn clarify_wgsl(wv: &str) -> String {
    format!(
        r#"
// Local contrast: the luminance's distance in stops from its blurred base
// (each channel's log2, averaged), soft-limited, scaled. slot+2 = (_, _,
// clarity, limit).
fn contrast_clarify(c: vec4f, base: vec4f, slot: u32) -> vec4f {{
    let q = P.v[slot + 2u];
    let y = dot(c.rgb, {wv});
    if (y <= 1e-6 || q.z == 0.0) {{ return c; }}
    let by = max(dot(exp2(base.rgb), {wv}), 1e-6);
    var d = log2(max(y, 1e-4)) - log2(by);
    d = d / (1.0 + abs(d) / q.w);
    return vec4f(c.rgb * exp2(q.z * d), c.a);
}}
"#
    )
}

impl StageDef for ContrastParams {
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
                label: "contrast-down",
                wgsl: DOWN_WGSL,
                binds: vec![Bind::Uniform, Bind::Sampler, Bind::Tex(src), Bind::Storage(mip(i))],
                dispatch: Dispatch::Pixels(mip(i)),
                uniform_floats: 4,
            });
        }
        // Glow's tent upsample: each level adds its weighted octave to the
        // upsampled sum above it. Only two octaves weigh anything.
        for i in (0..n).rev() {
            let upper = if i + 1 < n { up(i + 1) } else { mip(i) };
            passes.push(Pass {
                label: "contrast-up",
                wgsl: crate::nodes::glow::UP_WGSL,
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
            label: "contrast-base",
            wgsl: FINAL_WGSL,
            binds: vec![Bind::Sampler, Bind::Tex(up(0)), Bind::Storage("b".into())],
            dispatch: Dispatch::Pixels("b".into()),
            uniform_floats: 0,
        });
        StageProgram { textures, passes, result: "b".into() }
    }

    fn uniforms(&self, pass: usize, cx: &StageCx) -> Vec<f32> {
        let n = levels(cx.canvas.0, cx.canvas.1);
        if pass < n {
            vec![(pass == 0) as u32 as f32, 0.0, 0.0, 0.0]
        } else if pass < 2 * n {
            let i = 2 * n - 1 - pass;
            let w = Self::level_weights(self.radius * cx.proxy_scale, n);
            vec![w[i], (i + 1 < n) as u32 as f32, 0.0, 0.0]
        } else {
            Vec::new()
        }
    }

    fn speed(&self) -> Speed {
        Speed::Slow
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: [f32; 3] = [0.2722287, 0.6740818, 0.0536895];

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5 + 1e-5 * b.abs()
    }

    #[test]
    fn the_defaults_change_nothing() {
        for domain in ContrastDomain::ALL {
            let p = ContrastParams { domain, ..Default::default() };
            for c in [[0.18, 0.5, 2.0, 1.0], [-0.3, 0.0, 7.5, 0.25], [0.01, 0.02, 0.0, 0.5]] {
                let out = p.apply(c, W);
                for i in 0..4 {
                    assert!(close(out[i], c[i]), "{domain:?} moved {c:?} to {out:?}");
                }
            }
        }
    }

    #[test]
    fn the_pivot_stays_put() {
        let p = ContrastParams { contrast: 1.8, preserve: 0.0, toe: 0.6, shoulder: 0.3, ..Default::default() };
        let out = p.apply([0.18, 0.18, 0.18, 1.0], W);
        assert!(out[..3].iter().all(|v| close(*v, 0.18)), "{out:?}");
    }

    /// In log a slope of 2 doubles every stop from the pivot.
    #[test]
    fn log_contrast_is_a_slope_in_stops() {
        let p = ContrastParams { contrast: 2.0, preserve: 0.0, ..Default::default() };
        assert!(close(p.tone(0.36), 0.72));
        assert!(close(p.tone(0.09), 0.045));
        // ... and passes black and negatives through.
        assert_eq!(p.tone(0.0), 0.0);
        assert_eq!(p.tone(-0.2), -0.2);
    }

    #[test]
    fn linear_contrast_is_a_line_through_the_pivot() {
        let p = ContrastParams { contrast: 1.5, domain: ContrastDomain::Linear, ..Default::default() };
        for v in [-0.1f32, 0.0, 0.18, 0.5, 3.0] {
            assert!(close(p.tone(v), 0.18 + (v - 0.18) * 1.5), "{v}");
        }
    }

    #[test]
    fn shadows_and_highlights_set_each_side() {
        let p = ContrastParams { shadows: 2.0, highlights: 0.5, ..Default::default() };
        // Two stops under lands four under; two over lands one over.
        assert!(close(p.tone(0.045), 0.18 / 16.0));
        assert!(close(p.tone(0.72), 0.36));
    }

    /// A soft toe and shoulder: slope c at the pivot, back to 1 far away.
    #[test]
    fn softness_eases_back_to_the_pictures_own_slope() {
        let p = ContrastParams { contrast: 2.0, toe: 1.0, shoulder: 1.0, range: 2.0, ..Default::default() };
        let slope = |t: f32| (p.curve(t + 1e-3) - p.curve(t - 1e-3)) / 2e-3;
        assert!((slope(0.0) - 2.0).abs() < 1e-3);
        assert!((slope(10.0) - 1.0).abs() < 1e-3);
        assert!((slope(-10.0) - 1.0).abs() < 1e-3);
        // Monotonic, steep or flat.
        for c in [0.2f32, 1.0, 3.0] {
            let q = ContrastParams { contrast: c, ..p };
            let mut last = f32::NEG_INFINITY;
            for i in -200..200 {
                let v = q.curve(i as f32 * 0.1);
                assert!(v > last, "c {c} not monotonic at {i}");
                last = v;
            }
        }
    }

    /// On luminance the ratios between channels — hue and saturation — hold.
    #[test]
    fn preserve_holds_the_channel_ratios() {
        let p = ContrastParams { contrast: 1.7, preserve: 1.0, ..Default::default() };
        let c = [0.6, 0.3, 0.1, 1.0];
        let o = p.apply(c, W);
        assert!(close(o[0] / o[1], 2.0) && close(o[1] / o[2], 3.0), "{o:?}");
        let y = |v: [f32; 4]| v[0] * W[0] + v[1] * W[1] + v[2] * W[2];
        assert!(close(y(o), p.tone(y(c))));
        // Per channel saturates instead.
        let q = ContrastParams { preserve: 0.0, ..p };
        let o = q.apply(c, W);
        assert!(o[0] / o[2] > 6.5, "{o:?}");
    }

    #[test]
    fn mix_dissolves_back_and_alpha_is_untouched() {
        let c = [0.5, 0.25, 0.05, 0.3];
        let p = ContrastParams { contrast: 2.0, mix: 0.0, ..Default::default() };
        assert_eq!(p.apply(c, W), c);
        let full = ContrastParams { mix: 1.0, ..p }.apply(c, W);
        let half = ContrastParams { mix: 0.5, ..p }.apply(c, W);
        for i in 0..3 {
            assert!(close(half[i], (c[i] + full[i]) * 0.5));
        }
        assert_eq!(full[3], 0.3);
    }

    #[test]
    fn clarity_pushes_away_from_the_base_and_is_limited() {
        let p = ContrastParams { local: true, clarity: 1.0, limit: 1.0, ..Default::default() };
        let base = [0.18f32.log2(); 3];
        // A pixel on its own base is left alone.
        let on = p.clarify([0.18; 3], base, W);
        assert!(on.iter().all(|v| close(*v, 0.18)));
        // Brighter than its surround gets brighter, darker darker.
        assert!(p.clarify([0.36; 3], base, W)[0] > 0.36);
        assert!(p.clarify([0.09; 3], base, W)[0] < 0.09);
        // A huge edge moves at most `limit` stops.
        let hot = p.clarify([0.18 * 1024.0; 3], base, W)[0];
        assert!(hot / (0.18 * 1024.0) < 2.0 + 1e-4, "{hot}");
        // Negative clarity softens.
        let soft = ContrastParams { clarity: -0.5, ..p };
        assert!(soft.clarify([0.36; 3], base, W)[0] < 0.36);
    }

    #[test]
    fn level_weights_bracket_the_radius_and_sum_to_one() {
        for r in [0.5f32, 2.0, 3.0, 40.0, 100.0, 5000.0] {
            let w = ContrastParams::level_weights(r, 8);
            assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-6, "{r}: {w:?}");
            assert!(w.iter().filter(|x| **x > 0.0).count() <= 2);
        }
        // 64 px sits on octave 5 (2^(5+1)).
        assert_eq!(ContrastParams::level_weights(64.0, 8)[5], 1.0);
    }

    #[test]
    fn every_name_round_trips() {
        let mut p = ContrastParams::default();
        for (i, n) in NAMES.iter().enumerate() {
            let v = 0.25 + i as f32;
            assert!(p.set(n, v), "{n}");
            assert_eq!(p.get(n), Some(v), "{n}");
        }
    }

    #[test]
    fn only_local_moves_the_topology() {
        let a = ContrastParams::default();
        let b = ContrastParams { contrast: 3.0, domain: ContrastDomain::Linear, ..a };
        assert_eq!(a.topo_tag(), b.topo_tag());
        assert_ne!(a.topo_tag(), ContrastParams { local: true, ..a }.topo_tag());
    }
}
