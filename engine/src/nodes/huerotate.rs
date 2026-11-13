//! HueRotate: turn every colour around the neutral axis by an angle,
//! holding each pixel's LUMINANCE exactly.
//!
//! A pixel is its luma `y = dot(rgb, w)` — the working space's own weights —
//! plus its chroma `c = rgb - y`, which lies in the plane `dot(w, c) = 0`
//! because the weights sum to one. Turning `c` about that plane's normal
//! `n = w / |w|` (Rodrigues) keeps it in the plane, so luma holds exactly,
//! and keeps its length:
//!
//! ```text
//! out = y + c cos θ + (n × c) sin θ
//! ```
//!
//! Greys have no chroma and do not move; 360° is the identity; a positive
//! angle turns red toward yellow, the way the hue wheel runs everywhere else
//! in the app. It is the move CSS's hue-rotate makes, with the weights this
//! comp actually uses rather than a hardcoded Rec.709 trio — the rgb
//! slider's colour axes hold luma the same way. A rotation can ask for a
//! colour the gamut does not have (a yellow turned blue at yellow's
//! luminance), and then a channel goes negative, as it would anywhere in a
//! scene-linear comp: that is left to the view transform or a later node,
//! not clamped here.
//!
//! The angle is a uniform — its cosine and sine are packed, not worked out
//! per pixel — so nothing on the node recompiles. [`HueRotateParams::apply`]
//! is the CPU mirror the tests check the GPU against.

use crate::codegen::Emit;
use crate::color::Colorspace;
use crate::nodes::NodeDef;

const TOPO_BASE: u64 = 0x6E;

/// A [`crate::graph::NodeKind::HueRotate`]'s one control.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct HueRotateParams {
    /// Degrees. Positive turns red toward yellow.
    pub angle: f32,
}

impl HueRotateParams {
    /// The unit axis the chroma turns about: the working space's luma
    /// weights, normalised.
    fn axis(working: Colorspace) -> [f32; 3] {
        let w = working.primaries.luma_weights();
        let len = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt();
        w.map(|x| x / len)
    }

    /// (cos, sin) of the angle; a non-finite angle turns nothing.
    fn turn(&self) -> (f32, f32) {
        if self.angle.is_finite() {
            let (s, c) = self.angle.to_radians().sin_cos();
            (c, s)
        } else {
            (1.0, 0.0)
        }
    }

    /// The CPU mirror of the generated code.
    pub fn apply(&self, c: [f32; 3], working: Colorspace) -> [f32; 3] {
        let w = working.primaries.luma_weights();
        let y = c[0] * w[0] + c[1] * w[1] + c[2] * w[2];
        let ch = [c[0] - y, c[1] - y, c[2] - y];
        let n = Self::axis(working);
        let x = [
            n[1] * ch[2] - n[2] * ch[1],
            n[2] * ch[0] - n[0] * ch[2],
            n[0] * ch[1] - n[1] * ch[0],
        ];
        let (co, s) = self.turn();
        std::array::from_fn(|i| y + ch[i] * co + x[i] * s)
    }
}

impl NodeDef for HueRotateParams {
    const NAME: &'static str = "HueRotate";
    /// Image, then the mask.
    const INPUTS: usize = 2;
    const MASK_INPUT: Option<usize> = Some(1);
    /// cos θ, sin θ.
    const VEC4_SLOTS: usize = 1;

    /// The angle is a uniform: one shape of code, never a recompile.
    fn topo_tag(&self) -> u64 {
        TOPO_BASE
    }

    fn pack(&self, v: &mut [f32]) {
        let (c, s) = self.turn();
        v[0] = c;
        v[1] = s;
    }

    fn emit(&self, e: &mut Emit) -> String {
        let input = e.input(0);
        let slot = e.slot();
        let working = e.working();
        let w = working.primaries.luma_weights();
        let n = Self::axis(working);
        let (y, c, v) = (e.fresh(), e.fresh(), e.fresh());
        e.line(format_args!(
            "let {y} = dot({input}.rgb, vec3f({:.7}, {:.7}, {:.7}));",
            w[0], w[1], w[2]
        ));
        e.line(format_args!("let {c} = {input}.rgb - vec3f({y});"));
        // Alpha is not a colour and is left alone.
        e.line(format_args!(
            "let {v} = vec4f(vec3f({y}) + {c} * P.v[{slot}].x \
             + cross(vec3f({:.7}, {:.7}, {:.7}), {c}) * P.v[{slot}].y, {input}.a);",
            n[0], n[1], n[2]
        ));
        e.masked(Self::MASK_INPUT, &input, v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPACES: [Colorspace; 2] = [Colorspace::LINEAR_REC709, Colorspace::ACESCG];

    fn luma(c: [f32; 3], ws: Colorspace) -> f32 {
        let w = ws.primaries.luma_weights();
        c[0] * w[0] + c[1] * w[1] + c[2] * w[2]
    }

    fn close(a: [f32; 3], b: [f32; 3], tol: f32) -> bool {
        (0..3).all(|i| (a[i] - b[i]).abs() <= tol)
    }

    const SAMPLES: [[f32; 3]; 6] = [
        [1.0, 0.0, 0.0],
        [0.2, 0.7, 0.1],
        [0.05, 0.1, 0.9],
        [4.0, 2.5, 0.3],
        [0.3, 0.3, 0.3],
        [-0.1, 0.4, 0.2],
    ];

    /// Luminance is exactly what it was, at every angle, in every space.
    #[test]
    fn luma_holds_at_any_angle() {
        for ws in SPACES {
            for angle in [-170.0, -45.0, 12.5, 90.0, 180.0, 333.0] {
                let p = HueRotateParams { angle };
                for c in SAMPLES {
                    let (a, b) = (luma(c, ws), luma(p.apply(c, ws), ws));
                    assert!((a - b).abs() < 1e-5 * a.abs().max(1.0), "{c:?} at {angle}°: {a} -> {b}");
                }
            }
        }
    }

    /// Greys have no hue to turn; 0° and 360° change nothing; a turn and its
    /// opposite cancel; three thirds make a whole.
    #[test]
    fn the_circle_closes() {
        for ws in SPACES {
            let grey = [0.18, 0.18, 0.18];
            assert!(close(HueRotateParams { angle: 77.0 }.apply(grey, ws), grey, 1e-6));
            for c in SAMPLES {
                for identity in [0.0, 360.0, -720.0] {
                    let out = HueRotateParams { angle: identity }.apply(c, ws);
                    assert!(close(out, c, 1e-4), "{identity}° moved {c:?} to {out:?}");
                }
                let there = HueRotateParams { angle: 50.0 }.apply(c, ws);
                let back = HueRotateParams { angle: -50.0 }.apply(there, ws);
                assert!(close(back, c, 1e-4), "{c:?} did not come back: {back:?}");
                let third = HueRotateParams { angle: 120.0 };
                let round = third.apply(third.apply(third.apply(c, ws), ws), ws);
                assert!(close(round, c, 1e-4), "three thirds of {c:?} gave {round:?}");
            }
        }
    }

    /// Positive turns red toward yellow — the direction every other hue
    /// control here runs — and the colour keeps its distance from grey.
    #[test]
    fn positive_turns_red_toward_yellow() {
        let ws = Colorspace::LINEAR_REC709;
        let red = [1.0, 0.0, 0.0];
        let out = HueRotateParams { angle: 30.0 }.apply(red, ws);
        assert!(out[1] > out[2], "+30° from red should lean to yellow, got {out:?}");
        let back = HueRotateParams { angle: -30.0 }.apply(red, ws);
        assert!(back[2] > back[1], "-30° from red should lean to magenta, got {back:?}");
        let y = luma(red, ws);
        let len = |c: [f32; 3]| ((c[0] - y).powi(2) + (c[1] - y).powi(2) + (c[2] - y).powi(2)).sqrt();
        assert!((len(out) - len(red)).abs() < 1e-5, "the chroma's length moved");
    }
}
