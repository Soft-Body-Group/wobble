//! Despill: take the screen's colour back out of a keyed foreground.
//!
//! A green screen bounces green onto everything in front of it — hair,
//! skin, the edges a key leaves soft. The classic fix limits the screen
//! channel to what the OTHER two channels say it should be:
//!
//! ```text
//! limit = average(r, b)        (or max, or min)
//! spill = max(g - limit · bias, 0) · strength
//! g'    = g - spill
//! ```
//!
//! and the same with blue and (r, g) for a blue screen. **average** is the
//! everyday choice; **max** (limit to the larger) is gentle — it keeps
//! yellows and cyans — and **min** is aggressive. `bias` scales the limit:
//! above 1 lets more of the screen colour through, below 1 removes more.
//!
//! Taking spill out of one channel darkens the pixel by exactly
//! `spill · w` where `w` is that channel's luma weight in the WORKING
//! space; `restore` adds that much back as grey, so 1 holds luminance
//! (the spill turns neutral instead of going dark).
//!
//! Everything is a uniform — screen, method and all — so nothing on the
//! node ever recompiles; the selections are packed as one-hot vectors
//! the shader dots against. [`DespillParams::apply`] is the CPU mirror the
//! tests check the GPU against. Alpha is left alone: despill is a colour
//! fix, and a premultiplied pixel scaled by its alpha despills to the same
//! picture.

use crate::codegen::Emit;
use crate::color::Colorspace;
use crate::nodes::NodeDef;

const TOPO_BASE: u64 = 0x6F;

/// Which colour the screen was.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DespillScreen {
    #[default]
    Green,
    Blue,
}

impl DespillScreen {
    pub const ALL: [DespillScreen; 2] = [DespillScreen::Green, DespillScreen::Blue];
    pub fn name(self) -> &'static str {
        match self {
            DespillScreen::Green => "green",
            DespillScreen::Blue => "blue",
        }
    }
    /// The screen channel's index.
    fn channel(self) -> usize {
        match self {
            DespillScreen::Green => 1,
            DespillScreen::Blue => 2,
        }
    }
    /// The other two channels' indices.
    fn others(self) -> (usize, usize) {
        match self {
            DespillScreen::Green => (0, 2),
            DespillScreen::Blue => (0, 1),
        }
    }
}

/// What the screen channel is limited to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DespillMethod {
    /// The mean of the other two — the everyday despill.
    #[default]
    Average,
    /// The larger of the other two: gentle, keeps yellows and cyans.
    Max,
    /// The smaller of the other two: aggressive.
    Min,
}

impl DespillMethod {
    pub const ALL: [DespillMethod; 3] = [DespillMethod::Average, DespillMethod::Max, DespillMethod::Min];
    pub fn name(self) -> &'static str {
        match self {
            DespillMethod::Average => "average",
            DespillMethod::Max => "max",
            DespillMethod::Min => "min",
        }
    }
    fn one_hot(self) -> [f32; 3] {
        match self {
            DespillMethod::Average => [1.0, 0.0, 0.0],
            DespillMethod::Max => [0.0, 1.0, 0.0],
            DespillMethod::Min => [0.0, 0.0, 1.0],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct DespillParams {
    pub screen: DespillScreen,
    pub method: DespillMethod,
    /// How much of the spill comes out: 0 none, 1 all of it.
    pub strength: f32,
    /// Scales the limit. 1 is the textbook despill; above lets more of the
    /// screen colour through, below takes more out.
    pub bias: f32,
    /// How much of the luminance the spill took is put back as grey: 0
    /// leaves the pixel darker, 1 holds its luminance exactly.
    pub restore: f32,
}

impl Default for DespillParams {
    fn default() -> Self {
        Self {
            screen: DespillScreen::Green,
            method: DespillMethod::Average,
            strength: 1.0,
            bias: 1.0,
            restore: 0.0,
        }
    }
}

impl DespillParams {
    /// The spill this pixel carries, before `strength`.
    fn spill(&self, c: [f32; 3]) -> f32 {
        let (o1, o2) = self.screen.others();
        let (a, b) = (c[o1], c[o2]);
        let limit = match self.method {
            DespillMethod::Average => 0.5 * (a + b),
            DespillMethod::Max => a.max(b),
            DespillMethod::Min => a.min(b),
        };
        (c[self.screen.channel()] - limit * self.bias).max(0.0)
    }

    /// The CPU mirror of the generated code.
    pub fn apply(&self, c: [f32; 3], working: Colorspace) -> [f32; 3] {
        let s = self.spill(c) * self.strength;
        let ch = self.screen.channel();
        let grey = s * working.primaries.luma_weights()[ch] * self.restore;
        std::array::from_fn(|i| c[i] - if i == ch { s } else { 0.0 } + grey)
    }

    /// The spill, alone, scaled by strength — what a spill matte shows.
    pub fn spill_of(&self, c: [f32; 3]) -> f32 {
        self.spill(c) * self.strength
    }
}

impl NodeDef for DespillParams {
    const NAME: &'static str = "Despill";
    /// Image, then the mask.
    const INPUTS: usize = 2;
    const MASK_INPUT: Option<usize> = Some(1);
    /// (strength, bias, restore), the screen one-hot, the
    /// second "other" channel's one-hot, the method one-hot.
    const VEC4_SLOTS: usize = 4;

    /// Every control is a uniform: one shape of code, never a recompile.
    fn topo_tag(&self) -> u64 {
        TOPO_BASE
    }

    fn pack(&self, v: &mut [f32]) {
        // The screen channel's luma weight depends on the working space,
        // which is structural: emit bakes it in as a constant.
        v[0] = self.strength;
        v[1] = self.bias;
        v[2] = self.restore;
        let ch = self.screen.channel();
        let (_, o2) = self.screen.others();
        for i in 0..3 {
            v[4 + i] = (i == ch) as u32 as f32;
            v[8 + i] = (i == o2) as u32 as f32;
        }
        v[12..15].copy_from_slice(&self.method.one_hot());
    }

    fn emit(&self, e: &mut Emit) -> String {
        let input = e.input(0);
        let slot = e.slot();
        let w = e.working().primaries.luma_weights();
        let (sel, o2, lim, s, v) = (e.fresh(), e.fresh(), e.fresh(), e.fresh(), e.fresh());
        // The first "other" channel is red for both screens.
        e.line(format_args!("let {sel} = P.v[{}].xyz;", slot + 1));
        e.line(format_args!("let {o2} = dot({input}.rgb, P.v[{}].xyz);", slot + 2));
        e.line(format_args!(
            "let {lim} = dot(vec3f(0.5 * ({input}.r + {o2}), max({input}.r, {o2}), \
             min({input}.r, {o2})), P.v[{}].xyz) * P.v[{slot}].y;",
            slot + 3
        ));
        e.line(format_args!(
            "let {s} = max(dot({input}.rgb, {sel}) - {lim}, 0.0) * P.v[{slot}].x;"
        ));
        // Grey back in by the screen channel's luma weight, scaled by restore.
        e.line(format_args!(
            "let {v} = vec4f({input}.rgb - {sel} * {s} \
             + vec3f({s} * dot({sel}, vec3f({:.7}, {:.7}, {:.7})) * P.v[{slot}].z), {input}.a);",
            w[0], w[1], w[2]
        ));
        e.masked(Self::MASK_INPUT, &input, v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const WS: Colorspace = Colorspace::LINEAR_REC709;

    fn luma(c: [f32; 3], ws: Colorspace) -> f32 {
        let w = ws.primaries.luma_weights();
        c[0] * w[0] + c[1] * w[1] + c[2] * w[2]
    }

    /// Pure screen green goes to its limit; neutrals and colours with no
    /// excess screen channel do not move.
    #[test]
    fn spill_comes_out_and_nothing_else_moves() {
        let p = DespillParams::default();
        let out = p.apply([0.2, 0.8, 0.1], WS);
        assert!((out[1] - 0.15).abs() < 1e-6, "green should land at avg(r,b), got {out:?}");
        assert_eq!(out[0], 0.2);
        assert_eq!(out[2], 0.1);
        for c in [[0.18, 0.18, 0.18], [0.9, 0.3, 0.2], [0.1, 0.2, 0.6], [3.0, 2.0, 2.5]] {
            assert_eq!(p.apply(c, WS), c, "{c:?} has no spill and must not move");
        }
    }

    #[test]
    fn methods_order_from_gentle_to_aggressive() {
        let c = [0.6, 0.7, 0.1];
        let g = |m| DespillParams { method: m, ..Default::default() }.apply(c, WS)[1];
        let (avg, max, min) = (g(DespillMethod::Average), g(DespillMethod::Max), g(DespillMethod::Min));
        assert!(max > avg && avg > min, "max {max} avg {avg} min {min}");
        assert!((max - 0.6).abs() < 1e-6 && (min - 0.1).abs() < 1e-6);
    }

    #[test]
    fn blue_screen_limits_blue_by_red_and_green() {
        let p = DespillParams { screen: DespillScreen::Blue, ..Default::default() };
        let out = p.apply([0.2, 0.4, 0.9], WS);
        assert_eq!([out[0], out[1]], [0.2, 0.4]);
        assert!((out[2] - 0.3).abs() < 1e-6, "{out:?}");
        // Green spill is not a blue screen's business.
        assert_eq!(p.apply([0.1, 0.9, 0.1], WS), [0.1, 0.9, 0.1]);
    }

    #[test]
    fn strength_and_bias_scale_it() {
        let c = [0.2, 0.8, 0.2];
        let half = DespillParams { strength: 0.5, ..Default::default() }.apply(c, WS);
        assert!((half[1] - 0.5).abs() < 1e-6, "{half:?}");
        let none = DespillParams { strength: 0.0, ..Default::default() }.apply(c, WS);
        assert_eq!(none, c);
        let loose = DespillParams { bias: 2.0, ..Default::default() }.apply(c, WS);
        assert!((loose[1] - 0.4).abs() < 1e-6, "{loose:?}");
    }

    /// Restore 1 holds luminance exactly, in either working space.
    #[test]
    fn restore_holds_luminance() {
        for ws in [Colorspace::LINEAR_REC709, Colorspace::ACESCG] {
            for screen in DespillScreen::ALL {
                let p = DespillParams { screen, restore: 1.0, ..Default::default() };
                for c in [[0.2, 0.8, 0.1], [0.3, 0.2, 0.9], [1.5, 4.0, 3.0]] {
                    let (a, b) = (luma(c, ws), luma(p.apply(c, ws), ws));
                    assert!((a - b).abs() < 1e-5, "{screen:?} {c:?}: {a} -> {b}");
                }
            }
        }
    }
}
