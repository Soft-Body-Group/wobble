//! VectorWarp: sample a picture where motion vectors point.
//!
//! Every output pixel `p` is the source at `p + amount × v(p)`, `v` the
//! FORWARD vector (`rg`) for a positive amount and the BACKWARD one (`ba`)
//! for a negative amount — so with a Flow's vectors, amount 1 over the
//! NEXT frame (a TimeOffset of +1 upstream) pulls that frame back onto
//! this one: the neighbour aligned to the current frame, which is what a
//! temporal denoise, a frame average or a clean-plate fill wants. Amount
//! 0.5 over the same frame is a half-step drift, the building block of an
//! interpolated in-between.
//!
//! It fuses, exactly like UV Warp: the source subtree is evaluated at the
//! looked-up position (`Emit::input_at`), no pass and no texture. The
//! vectors are canvas pixels (a Flow node scales them under proxy).

use crate::codegen::Emit;
use crate::nodes::NodeDef;

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct VectorWarpParams {
    /// Frames of motion to follow: 1 is one whole frame forward, -1 one
    /// back, 0 no warp.
    pub amount: f32,
}

impl Default for VectorWarpParams {
    fn default() -> Self {
        Self { amount: 1.0 }
    }
}

impl VectorWarpParams {
    /// Where pixel `p` samples the source, given its vectors `rgba` — the
    /// CPU mirror of the generated code.
    pub fn lookup(&self, p: [f32; 2], v: [f32; 4]) -> [f32; 2] {
        let a = self.amount;
        if a >= 0.0 {
            [p[0] + v[0] * a, p[1] + v[1] * a]
        } else {
            [p[0] - v[2] * a, p[1] - v[3] * a]
        }
    }
}

impl NodeDef for VectorWarpParams {
    const NAME: &'static str = "VectorWarp";
    /// The source, the vectors, the mask.
    const INPUTS: usize = 3;
    const MASK_INPUT: Option<usize> = Some(2);
    /// (amount, -, -, -).
    const VEC4_SLOTS: usize = 1;

    fn topo_tag(&self) -> u64 {
        0x7E40
    }

    fn pack(&self, v: &mut [f32]) {
        v[0] = self.amount;
        v[1] = 0.0;
        v[2] = 0.0;
        v[3] = 0.0;
    }

    fn emit(&self, e: &mut Emit) -> String {
        if !e.has_input(0) {
            return "vec4f(0.0)".into();
        }
        // No vectors wired: nothing moved, the source passes.
        if !e.has_input(1) {
            return e.input(0);
        }
        let slot = e.slot();
        let pos = e.pos().to_string();
        let v = e.input(1);
        let (a, d, q) = (e.fresh(), e.fresh(), e.fresh());
        e.line(format_args!("let {a} = P.v[{slot}].x;"));
        e.line(format_args!("let {d} = select(-{v}.zw * {a}, {v}.xy * {a}, {a} >= 0.0);"));
        e.line(format_args!("let {q} = {pos} + {d};"));
        let out = e.input_at(0, &q);
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
    fn the_sign_of_the_amount_picks_the_vector() {
        let v = [4.0, 2.0, -1.0, 3.0];
        assert_eq!(VectorWarpParams { amount: 1.0 }.lookup([10.0, 10.0], v), [14.0, 12.0]);
        assert_eq!(VectorWarpParams { amount: 0.5 }.lookup([10.0, 10.0], v), [12.0, 11.0]);
        assert_eq!(VectorWarpParams { amount: -1.0 }.lookup([10.0, 10.0], v), [9.0, 13.0]);
        assert_eq!(VectorWarpParams { amount: 0.0 }.lookup([10.0, 10.0], v), [10.0, 10.0]);
    }
}
