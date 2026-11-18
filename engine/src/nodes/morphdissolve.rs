//! Morph Dissolve: a transition that warps and dissolves at once.
//!
//! Input 0 is A (outgoing), input 1 is B (incoming). Over the transition A
//! is pushed along the A→B motion while B is pulled back along the B→A
//! motion, so the two pictures meet in the middle, and the two warped
//! pictures cross-dissolve. `mix` is the one knob you key.
//!
//! The motion is solved by the Flow node's networks (`crate::flowsolve`):
//! on a click the app renders A and B over the transition and solves both
//! directions at every tick, A against B at that SAME tick. The vectors
//! land on disk and play back through this node's reader slot, `rg` = A→B
//! and `ba` = B→A, in the Flow node's own format. So the analysis is the
//! explicit slow step, and the node itself FUSES: three flow taps a side,
//! two `input_at`s, no pass and no texture. Everything you drag is a
//! uniform; moving `mix` never recompiles and never solves.
//!
//! Unanalysed (or with no weights) the node is a plain cross-dissolve.
//!
//! # The warp
//!
//! A feature at `x` in A is at `x + F_ab(x)` in B, so at time `t` it sits at
//! `q = x + t F_ab(x)`. The output pixel `q` needs the `x` that lands on it:
//! a fixed point, `x = q - t F_ab(x)`, seeded by the usual linear estimate
//! from both flows. B is the same from the other end, with `1 - t` and
//! `F_ba`. At `t = 0` A's lookup is exactly `q` and B has no weight, and the
//! reverse at `t = 1`, so both ends are the untouched inputs whatever the
//! vectors say: nothing pops when the transition starts or ends.
//!
//! # Occlusion
//!
//! Each side earns a confidence: its fixed point must actually land on `q`
//! and on the picture, and a vector that does not round-trip through the
//! other flow (forward-backward consistency) counts for less. The dissolve
//! weights are scaled by it, so a pixel only one picture really has comes
//! from that picture rather than ghosting. Where neither side is confident
//! the weights fall back to the plain dissolve.

use crate::codegen::Emit;
pub use crate::nodes::flow::{FlowDetail, FlowEngine};

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct MorphDissolveParams {
    /// Reader slot the vectors play through.
    pub source: usize,
    pub engine: FlowEngine,
    pub detail: FlowDetail,
    /// Progress of the transition: 0 is A, 1 is B.
    pub mix: f32,
    /// How much of the motion is followed: 0 is a plain dissolve, 1 meets
    /// exactly in the middle.
    pub warp: f32,
    /// Shifts the dissolve against the warp: positive brings B in early,
    /// negative holds A longer.
    pub lead: f32,
    /// How much of the transition the dissolve takes: 1 all of it, small
    /// values a quick cut in the middle of the morph.
    pub width: f32,
    /// How far the occlusion weights are trusted: 0 ignores them.
    pub occlusion: f32,
    /// TRANSIENT: vectors are loaded in the reader slot.
    #[cfg_attr(feature = "serde", serde(skip_serializing))]
    pub loaded: bool,
    /// TRANSIENT: the comp frame the first vector file holds.
    #[cfg_attr(feature = "serde", serde(skip_serializing))]
    pub start: Option<i64>,
}

impl Default for MorphDissolveParams {
    fn default() -> Self {
        Self {
            source: 0,
            engine: FlowEngine::SeaRaft,
            detail: FlowDetail::Half,
            mix: 0.5,
            warp: 1.0,
            lead: 0.0,
            width: 1.0,
            occlusion: 1.0,
            loaded: false,
            start: None,
        }
    }
}

/// Fixed-point steps per side. Three settles any motion that is smooth over
/// the distance moved; what does not settle is caught by the residual.
const STEPS: usize = 3;

impl MorphDissolveParams {
    pub const NAME: &'static str = "Morph Dissolve";
    /// A, B, the mask.
    pub const INPUTS: usize = 3;
    pub const MASK_INPUT: Option<usize> = Some(2);
    /// (mix, warp, dissolve, occlusion), (proxy scale, -, -, -).
    pub const VEC4_SLOTS: usize = 2;

    pub fn new(source: usize) -> Self {
        Self { source, ..Default::default() }
    }

    /// Only whether vectors are loaded changes the shader.
    pub fn topo_tag(&self) -> u64 {
        self.loaded as u64
    }

    /// The settings that shape the vectors, for the app's result key. The
    /// transition's look is not among them: re-timing it solves nothing.
    pub fn recipe(&self) -> (&'static str, u8) {
        (self.engine.model(), self.detail.divisor() as u8)
    }

    /// The weights the picked engine needs.
    pub fn models(&self) -> &'static [&'static str] {
        match self.engine {
            FlowEngine::SeaRaft => &["searaft-m"],
            FlowEngine::Rife => &["rife-4.6"],
        }
    }

    /// B's share of the dissolve at this `mix`. Exactly 0 at `mix` 0 and
    /// exactly 1 at `mix` 1, whatever `lead` and `width` are.
    pub fn dissolve(&self) -> f32 {
        let t = self.mix.clamp(0.0, 1.0);
        let s = t.powf(2f32.powf(-2.0 * self.lead.clamp(-1.0, 1.0)));
        ((s - 0.5) / self.width.clamp(0.01, 1.0) + 0.5).clamp(0.0, 1.0)
    }

    pub fn pack(&self, proxy_scale: f32, out: &mut [f32]) {
        out[0] = self.mix.clamp(0.0, 1.0);
        out[1] = self.warp;
        out[2] = self.dissolve();
        out[3] = self.occlusion.clamp(0.0, 1.0);
        out[4] = proxy_scale;
        out[5] = 0.0;
        out[6] = 0.0;
        out[7] = 0.0;
    }

    /// One side's lookup: returns its colour and its confidence.
    ///
    /// `k` is how far along its own flow this side has travelled; `own` and
    /// `other` are the swizzles of its flow and the opposite one.
    #[allow(clippy::too_many_arguments)]
    fn side(e: &mut Emit, input: usize, src: &str, scale: &str, q: &str, seed: &str, k: &str, own: &str, other: &str) -> (String, String) {
        let mut x = seed.to_string();
        for _ in 0..STEPS {
            let n = e.fresh();
            e.line(format_args!("let {n} = {q} - {k} * (sample_px({src}, {x}) * {scale}).{own};"));
            x = n;
        }
        let (f, g, r, d, thr, c) = (e.fresh(), e.fresh(), e.fresh(), e.fresh(), e.fresh(), e.fresh());
        e.line(format_args!("let {f} = (sample_px({src}, {x}) * {scale}).{own};"));
        e.line(format_args!("let {g} = (sample_px({src}, {x} + {f}) * {scale}).{other};"));
        // Did the fixed point land on this pixel?
        e.line(format_args!("let {r} = {x} + {k} * {f} - {q};"));
        // Does the vector come back through the other flow?
        e.line(format_args!("let {d} = {f} + {g};"));
        e.line(format_args!("let {thr} = 0.01 * (dot({f}, {f}) + dot({g}, {g})) + 0.25;"));
        let size = format!("vec2f(textureDimensions({src}))");
        let on = |p: &str| format!("(all({p} >= vec2f(0.0)) && all({p} <= {size}))");
        let (y, rt) = (e.fresh(), e.fresh());
        e.line(format_args!("let {y} = {x} + {f};"));
        // Landing off the picture says nothing about the match.
        e.line(format_args!(
            "let {rt} = select(1.0, 1.0 - smoothstep({thr}, 4.0 * {thr}, dot({d}, {d})), {});",
            on(&y)
        ));
        // A lookup that missed, or left the picture, is wrong. One that
        // fails only the round trip is a pixel the OTHER picture does not
        // have: still real, so it keeps a share and beats a miss.
        e.line(format_args!(
            "let {c} = (1.0 - smoothstep(0.25, 4.0, dot({r}, {r}))) * mix(0.02, 1.0, {rt}) * select(0.0, 1.0, {});",
            on(&x)
        ));
        let colour = e.input_at(input, &x);
        (colour, c)
    }

    pub fn emit(&self, e: &mut Emit) -> String {
        match (e.has_input(0), e.has_input(1)) {
            (false, false) => return "vec4f(0.0)".into(),
            (true, false) => return e.input(0),
            (false, true) => return e.input(1),
            (true, true) => {}
        }
        let slot = e.slot();
        let s = format!("P.v[{slot}].z");
        let out = e.fresh();
        if self.loaded {
            let src = e.reader(self.source);
            let q = e.pos().to_string();
            let (t, w, occ, scale, f0) = (e.fresh(), e.fresh(), e.fresh(), e.fresh(), e.fresh());
            e.line(format_args!("let {t} = P.v[{slot}].x;"));
            e.line(format_args!("let {w} = P.v[{slot}].y;"));
            // With no warp there is nothing to be occluded by.
            e.line(format_args!("let {occ} = P.v[{slot}].w * saturate({w});"));
            e.line(format_args!("let {scale} = P.v[{slot} + 1].x;"));
            e.line(format_args!("let {f0} = sample_px({src}, {q}) * {scale};"));
            let (ka, kb, sa, sb) = (e.fresh(), e.fresh(), e.fresh(), e.fresh());
            e.line(format_args!("let {ka} = {t} * {w};"));
            e.line(format_args!("let {kb} = (1.0 - {t}) * {w};"));
            // The linear estimate of where each side comes from, using both
            // flows at this pixel: the fixed point starts close.
            e.line(format_args!("let {sa} = {q} + {w} * ({t} * {t} * {f0}.zw - {t} * (1.0 - {t}) * {f0}.xy);"));
            e.line(format_args!("let {sb} = {q} + {w} * ((1.0 - {t}) * (1.0 - {t}) * {f0}.xy - {t} * (1.0 - {t}) * {f0}.zw);"));
            let (a, ca) = Self::side(e, 0, &src, &scale, &q, &sa, &ka, "xy", "zw");
            let (b, cb) = Self::side(e, 1, &src, &scale, &q, &sb, &kb, "zw", "xy");
            let (wa, wb) = (e.fresh(), e.fresh());
            // The floor keeps the sum positive, so where neither side is
            // sure this comes out as the plain dissolve.
            e.line(format_args!("let {wa} = (1.0 - {s}) * max(mix(1.0, {ca}, {occ}), 0.001);"));
            e.line(format_args!("let {wb} = {s} * max(mix(1.0, {cb}, {occ}), 0.001);"));
            e.line(format_args!("let {out} = ({a} * {wa} + {b} * {wb}) / ({wa} + {wb});"));
        } else {
            let (a, b) = (e.input(0), e.input(1));
            e.line(format_args!("let {out} = mix({a}, {b}, {s});"));
        }
        if e.has_input(2) {
            let orig = e.input(0);
            e.masked(Self::MASK_INPUT, &orig, out)
        } else {
            out
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_dissolve_always_reaches_both_ends() {
        for lead in [-1.0, -0.3, 0.0, 0.5, 1.0] {
            for width in [0.01, 0.2, 1.0] {
                let p = |mix| MorphDissolveParams { mix, lead, width, ..Default::default() };
                assert_eq!(p(0.0).dissolve(), 0.0, "lead {lead} width {width}");
                assert_eq!(p(1.0).dissolve(), 1.0, "lead {lead} width {width}");
            }
        }
        let plain = MorphDissolveParams { mix: 0.25, ..Default::default() };
        assert!((plain.dissolve() - 0.25).abs() < 1e-6, "defaults are a linear dissolve");
        let early = MorphDissolveParams { mix: 0.25, lead: 0.5, ..Default::default() };
        assert!(early.dissolve() > 0.25, "a positive lead brings B in sooner");
        let late = MorphDissolveParams { mix: 0.25, lead: -0.5, ..Default::default() };
        assert!(late.dissolve() < 0.25, "a negative lead holds A");
    }

    #[test]
    fn only_loading_changes_the_shader_and_only_the_solve_settings_the_vectors() {
        let a = MorphDissolveParams::new(3);
        let look = MorphDissolveParams { mix: 0.9, warp: 0.2, lead: 0.4, width: 0.3, occlusion: 0.0, ..a.clone() };
        assert_eq!(a.topo_tag(), look.topo_tag(), "the knobs are uniforms");
        assert_eq!(a.recipe(), look.recipe(), "re-timing the transition solves nothing");
        assert_ne!(a.topo_tag(), MorphDissolveParams { loaded: true, ..a.clone() }.topo_tag());
        assert_ne!(a.recipe(), MorphDissolveParams { detail: FlowDetail::Full, ..a.clone() }.recipe());
        let rife = MorphDissolveParams { engine: FlowEngine::Rife, ..a.clone() };
        assert_ne!(a.recipe(), rife.recipe());
        assert_eq!(rife.models(), &["rife-4.6"]);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn transients_are_never_saved() {
        let p = MorphDissolveParams { loaded: true, start: Some(7), ..MorphDissolveParams::new(2) };
        let j = serde_json::to_string(&p).unwrap();
        assert!(!j.contains("loaded") && !j.contains("start"), "{j}");
    }
}
