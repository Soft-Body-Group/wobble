//! Gamma: a power curve per channel, `x^(1/γ)`, on its own — the middle
//! third of a Grade without the lift and the gain around it.
//!
//! One master γ scales every colour channel, and each of r, g, b and a has
//! its own on top: a channel's exponent is `1 / (master × channel)` for the
//! colours and `1 / a` for alpha, so the defaults (all 1) change nothing
//! and a master of 2 lifts the mids of every colour while alpha holds.
//! γ above 1 brightens, below 1 darkens; 0 and 1 are fixed points.
//!
//! Only POSITIVE values are curved. A scene-linear comp has negatives (a
//! gamut excursion, a keyed-out edge) and a power of one is either NaN or a
//! sign flip, so they pass through untouched — unlike a Grade, which clamps
//! them to zero first. A γ at or below zero, or a non-finite one, curves
//! nothing.
//!
//! `mix` dissolves the result back to the input (a uniform, never a
//! recompile) and the mask crossfades the same way every colour node's
//! does. The exponents are packed already inverted, so the shader is one
//! `pow` and one `select`. [`GammaParams::apply`] is the CPU mirror the tests
//! check the GPU against.

use crate::codegen::Emit;
use crate::nodes::NodeDef;

const TOPO_BASE: u64 = 0x9A44;

/// A [`crate::graph::NodeKind::Gamma`]'s controls.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct GammaParams {
    /// Scales the r, g and b gammas. 1 is no change.
    pub master: f32,
    /// Per channel r, g, b, times the master.
    pub rgb: [f32; 3],
    /// Alpha's own gamma; the master does not reach it.
    pub alpha: f32,
    /// 0 is the input, 1 the full curve.
    pub mix: f32,
    /// Which pipe it sits in: wired a MATTE it takes one and gives one
    /// (`NodeKind::follows_pipe`), the SoftCrop precedent. The app keeps it
    /// true to its input (`EditorGraph::connect_out`).
    pub matte: bool,
}

impl Default for GammaParams {
    fn default() -> Self {
        Self { master: 1.0, rgb: [1.0; 3], alpha: 1.0, mix: 1.0, matte: false }
    }
}

impl GammaParams {
    /// The gamma each channel actually gets.
    pub fn effective(&self) -> [f32; 4] {
        let [r, g, b] = self.rgb;
        [self.master * r, self.master * g, self.master * b, self.alpha]
    }

    /// The exponent per channel, `1/γ`; 1 (no change) where γ is not a
    /// usable positive number.
    pub fn exponents(&self) -> [f32; 4] {
        self.effective().map(|g| {
            if g > 0.0 && g.is_finite() { 1.0 / g } else { 1.0 }
        })
    }

    fn mix_amount(&self) -> f32 {
        if self.mix.is_finite() { self.mix } else { 1.0 }
    }

    /// The CPU mirror of the generated code (unmasked).
    pub fn apply(&self, c: [f32; 4]) -> [f32; 4] {
        let e = self.exponents();
        let m = self.mix_amount();
        std::array::from_fn(|i| {
            let curved = if c[i] > 0.0 { c[i].powf(e[i]) } else { c[i] };
            c[i] + (curved - c[i]) * m
        })
    }
}

impl NodeDef for GammaParams {
    const NAME: &'static str = "Gamma";
    /// Image, then the mask.
    const INPUTS: usize = 2;
    const MASK_INPUT: Option<usize> = Some(1);
    /// The four exponents, then the mix.
    const VEC4_SLOTS: usize = 2;

    /// Everything is a uniform: one shape of code, never a recompile.
    fn topo_tag(&self) -> u64 {
        TOPO_BASE
    }

    fn pack(&self, v: &mut [f32]) {
        v[..4].copy_from_slice(&self.exponents());
        v[4] = self.mix_amount();
    }

    fn emit(&self, e: &mut Emit) -> String {
        let input = e.input(0);
        let slot = e.slot();
        let (c, v) = (e.fresh(), e.fresh());
        // pow of a positive base only; zero and negatives keep their value.
        e.line(format_args!(
            "let {c} = select({input}, pow(max({input}, vec4f(1e-30)), P.v[{slot}]), {input} > vec4f(0.0));"
        ));
        e.line(format_args!("let {v} = mix({input}, {c}, P.v[{}].x);", slot + 1));
        e.masked(Self::MASK_INPUT, &input, v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_defaults_change_nothing() {
        let p = GammaParams::default();
        for c in [[0.18, 0.5, 2.0, 1.0], [-0.3, 0.0, 7.5, 0.25]] {
            assert_eq!(p.apply(c), c);
        }
    }

    /// γ 2 is a square root on the colours, alpha untouched by the master.
    #[test]
    fn master_is_a_power_on_colour_only() {
        let p = GammaParams { master: 2.0, ..Default::default() };
        let out = p.apply([0.25, 0.04, 4.0, 0.5]);
        let want = [0.5, 0.2, 2.0, 0.5];
        for i in 0..4 {
            assert!((out[i] - want[i]).abs() < 1e-6, "{out:?}");
        }
    }

    #[test]
    fn channels_multiply_the_master_and_alpha_takes_its_own() {
        let p = GammaParams { master: 2.0, rgb: [1.0, 0.5, 1.5], alpha: 2.0, mix: 1.0, matte: false };
        assert_eq!(p.effective(), [2.0, 1.0, 3.0, 2.0]);
        let out = p.apply([0.25, 0.25, 0.125, 0.25]);
        assert!((out[0] - 0.5).abs() < 1e-6);
        assert!((out[1] - 0.25).abs() < 1e-6);
        assert!((out[2] - 0.5).abs() < 1e-6);
        assert!((out[3] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn negatives_and_zero_pass_through() {
        let p = GammaParams { master: 2.2, ..Default::default() };
        let out = p.apply([-0.5, 0.0, -4.0, 0.0]);
        assert_eq!(out, [-0.5, 0.0, -4.0, 0.0]);
    }

    #[test]
    fn a_degenerate_gamma_curves_nothing() {
        for g in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let p = GammaParams { master: g, ..Default::default() };
            assert_eq!(p.exponents()[..3], [1.0; 3], "γ {g}");
        }
    }

    #[test]
    fn mix_dissolves_back_to_the_input() {
        let c = [0.25, 0.25, 0.25, 1.0];
        let p = GammaParams { master: 2.0, mix: 0.0, ..Default::default() };
        assert_eq!(p.apply(c), c);
        let half = GammaParams { mix: 0.5, ..p }.apply(c);
        assert!((half[0] - 0.375).abs() < 1e-6, "{half:?}");
    }
}
