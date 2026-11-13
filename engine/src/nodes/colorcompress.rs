//! ColorCompress: what a black point and a white point are actually FOR.
//!
//! Four knobs, one straight line per channel. **black** and **white** say
//! which input levels you care about; **lift** and **mult** say where those
//! two levels should land. Everything between them is carried along, and
//! everything outside them keeps going on the same line — nothing is
//! clipped, because in a scene-linear comp a clip is a decision you make at
//! a Write or a view transform, not one a colour node makes for you.
//!
//! ```text
//! a = (mult - lift) / (white - black)      out = a * x + b
//! b = lift - a * black
//! ```
//!
//! So `black -> lift` and `white -> mult`, exactly, by construction. That is
//! the same mapping a classic Grade calls blackpoint/whitepoint/lift/gain, and
//! the reason the four read as a pair of pairs rather than four unrelated
//! sliders: the first pair is a measurement of the plate, the second is the
//! picture you want out of it.
//!
//! **Compress or expand.** Narrow the output pair inside the input pair and
//! the range compresses — a full-swing plate tucked into a milky window to
//! sit under a soft comp. Widen it and it expands: a flat log-ish plate
//! pulled back out to contrast. Neither direction is privileged, and the
//! node has no opinion about which you meant.
//!
//! **What reaches the shader is the LINE, not the knobs.** `pack` solves
//! for the slope and the intercept on the CPU — the same precedent as the
//! Transform's inverse affine and the CornerPin's inverse homography — so
//! the fused pass costs exactly one multiply-add per pixel, and the
//! division (and its one guard) happens once per frame instead of once per
//! pixel.

use crate::codegen::Emit;
use crate::nodes::NodeDef;

const TOPO_BASE: u64 = 0x6C;

/// The narrowest input range that still has a slope. A range of zero asks
/// for an infinite one, which is not a picture; this is the width it is
/// treated as instead — steep, but finite, and the sign is kept so an
/// inverted range still inverts.
const MIN_RANGE: f32 = 1e-4;

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct ColorCompressParams {
    /// The input level that lands on `lift`.
    pub black: [f32; 3],
    /// The input level that lands on `mult`.
    pub white: [f32; 3],
    /// Where the black point goes.
    pub lift: [f32; 3],
    /// Where the white point goes.
    pub mult: [f32; 3],
}

impl Default for ColorCompressParams {
    /// 0 and 1 in, 0 and 1 out: the identity, so a fresh node changes
    /// nothing until a level is moved.
    fn default() -> Self {
        Self {
            black: [0.0; 3],
            white: [1.0; 3],
            lift: [0.0; 3],
            mult: [1.0; 3],
        }
    }
}

impl ColorCompressParams {
    /// The line this node IS, per channel: `(slope, intercept)` with
    /// `out = slope * x + intercept`.
    pub fn line(&self) -> ([f32; 3], [f32; 3]) {
        let mut slope = [0.0f32; 3];
        let mut intercept = [0.0f32; 3];
        for i in 0..3 {
            let d = self.white[i] - self.black[i];
            // Keep the SIGN: white below black is an inversion, which is a
            // thing people do on purpose, not an error to be clamped away.
            let d = if d < 0.0 { d.min(-MIN_RANGE) } else { d.max(MIN_RANGE) };
            slope[i] = (self.mult[i] - self.lift[i]) / d;
            intercept[i] = self.lift[i] - slope[i] * self.black[i];
        }
        (slope, intercept)
    }

    /// The CPU mirror of the generated line, for tests and for anything
    /// that needs to know where a level lands without rendering a frame.
    pub fn apply(&self, c: [f32; 3]) -> [f32; 3] {
        let (a, b) = self.line();
        [a[0] * c[0] + b[0], a[1] * c[1] + b[1], a[2] * c[2] + b[2]]
    }
}

impl NodeDef for ColorCompressParams {
    const NAME: &'static str = "ColorCompress";
    /// Image, then the mask.
    const INPUTS: usize = 2;
    const MASK_INPUT: Option<usize> = Some(1);
    /// The solved line — slope, then intercept — never the four knobs.
    const VEC4_SLOTS: usize = 2;

    /// Every control is a uniform: there is one shape of code here and no
    /// value can change it, so nothing this node does ever recompiles.
    fn topo_tag(&self) -> u64 {
        TOPO_BASE
    }

    fn pack(&self, v: &mut [f32]) {
        let (a, b) = self.line();
        v[..3].copy_from_slice(&a);
        v[4..7].copy_from_slice(&b);
    }

    fn emit(&self, e: &mut Emit) -> String {
        let input = e.input(0);
        let slot = e.slot();
        let v = e.fresh();
        // One multiply-add. Alpha is not a colour and is left alone.
        e.line(format_args!(
            "let {v} = vec4f({input}.rgb * P.v[{slot}].rgb + P.v[{}].rgb, {input}.a);",
            slot + 1
        ));
        // Masked: crossfade against the untouched input, the same way a
        // masked Grade does.
        e.masked(Self::MASK_INPUT, &input, v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    /// A fresh node is an identity at every level, negatives and highlights
    /// included — there is no range it quietly clamps.
    #[test]
    fn the_default_is_an_identity() {
        let p = ColorCompressParams::default();
        for c in [[-0.2, 0.0, 0.18], [0.5, 0.5, 0.5], [4.0, 12.0, 0.9]] {
            let out = p.apply(c);
            for i in 0..3 {
                assert!(close(out[i], c[i]), "{c:?} moved to {out:?}");
            }
        }
    }

    /// The whole contract: the two input levels land on the two output
    /// levels, exactly, and everything between rides the line between them.
    #[test]
    fn black_lands_on_lift_and_white_on_mult() {
        let p = ColorCompressParams {
            black: [0.05, 0.02, 0.0],
            white: [0.9, 1.2, 2.0],
            lift: [0.0, 0.1, -0.05],
            mult: [1.0, 0.8, 0.5],
        };
        let at_black = p.apply(p.black);
        let at_white = p.apply(p.white);
        for i in 0..3 {
            assert!(close(at_black[i], p.lift[i]), "black: {at_black:?}");
            assert!(close(at_white[i], p.mult[i]), "white: {at_white:?}");
        }
        // Halfway in is halfway out: it is a straight line, not a curve.
        let mid = [
            0.5 * (p.black[0] + p.white[0]),
            0.5 * (p.black[1] + p.white[1]),
            0.5 * (p.black[2] + p.white[2]),
        ];
        let out = p.apply(mid);
        for i in 0..3 {
            assert!(close(out[i], 0.5 * (p.lift[i] + p.mult[i])), "mid: {out:?}");
        }
    }

    /// Outside the two points the line KEEPS GOING. A node that clipped
    /// here would be throwing away highlights nobody asked it to judge.
    #[test]
    fn nothing_is_clipped_outside_the_points() {
        let p = ColorCompressParams {
            black: [0.0; 3],
            white: [1.0; 3],
            lift: [0.1; 3],
            mult: [0.6; 3],
        };
        assert!(p.apply([-1.0; 3])[0] < 0.1, "below black was clamped");
        assert!(p.apply([8.0; 3])[0] > 0.6, "above white was clamped");
        // ...and the slope is the same on both sides of each point.
        let step = |x: f32| p.apply([x + 0.1, 0.0, 0.0])[0] - p.apply([x, 0.0, 0.0])[0];
        assert!(close(step(-2.0), step(0.5)) && close(step(0.5), step(5.0)));
    }

    /// Compressing and expanding are the same control used two ways, and
    /// the node has no opinion about which was meant.
    #[test]
    fn it_compresses_and_expands_by_the_same_arithmetic() {
        let span = |p: &ColorCompressParams| p.apply([1.0; 3])[0] - p.apply([0.0; 3])[0];
        let squeeze = ColorCompressParams {
            lift: [0.2; 3],
            mult: [0.7; 3],
            ..Default::default()
        };
        assert!(close(span(&squeeze), 0.5), "{}", span(&squeeze));
        let stretch = ColorCompressParams {
            black: [0.2; 3],
            white: [0.7; 3],
            ..Default::default()
        };
        assert!(close(span(&stretch), 2.0), "{}", span(&stretch));
    }

    /// White below black inverts, on purpose. The guard is there to keep a
    /// degenerate range FINITE, not to decide which way round it may go.
    #[test]
    fn an_inverted_range_inverts_and_a_flat_one_stays_finite() {
        let flip = ColorCompressParams {
            black: [1.0; 3],
            white: [0.0; 3],
            ..Default::default()
        };
        assert!(close(flip.apply([1.0; 3])[0], 0.0));
        assert!(close(flip.apply([0.0; 3])[0], 1.0));

        let flat = ColorCompressParams {
            black: [0.5; 3],
            white: [0.5; 3],
            ..Default::default()
        };
        for x in [-1.0f32, 0.5, 9.0] {
            let out = flat.apply([x; 3])[0];
            assert!(out.is_finite(), "a zero-width range produced {out}");
        }
        // Steep, and still the right way up.
        assert!(flat.apply([0.6; 3])[0] > flat.apply([0.4; 3])[0]);
    }

    /// Nothing here is topology: every knob is a uniform, so no value of
    /// any of them can make a drag recompile the comp.
    #[test]
    fn no_control_is_topology() {
        let a = ColorCompressParams::default();
        let b = ColorCompressParams {
            black: [0.1, 0.2, 0.3],
            white: [3.0; 3],
            lift: [-0.1; 3],
            mult: [2.0; 3],
        };
        assert_eq!(a.topo_tag(), b.topo_tag());
    }

    /// The packed span is the LINE, in the order the generated code reads
    /// it: slope in the first slot's rgb, intercept in the second's.
    #[test]
    fn the_span_carries_the_solved_line() {
        let p = ColorCompressParams {
            black: [0.0; 3],
            white: [2.0; 3],
            lift: [0.0; 3],
            mult: [1.0; 3],
        };
        let mut v = [0.0f32; ColorCompressParams::VEC4_SLOTS * 4];
        p.pack(&mut v);
        assert!(close(v[0], 0.5) && close(v[1], 0.5) && close(v[2], 0.5));
        assert!(close(v[4], 0.0) && close(v[5], 0.0) && close(v[6], 0.0));
        // And it agrees with the CPU mirror at an arbitrary level.
        let x = 1.3f32;
        assert!(close(v[0] * x + v[4], p.apply([x; 3])[0]));
    }
}
