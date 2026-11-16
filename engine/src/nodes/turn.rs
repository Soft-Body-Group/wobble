//! Turn: rotate the picture in quarter turns, format and all. A 90° turn
//! swaps the format — 1920x1080 in is 1080x1920 out — and 180° keeps it.
//! Unlike a Transform's rotate nothing is resampled: every output pixel is
//! exactly one input pixel. It fuses: the input is evaluated at the turned
//! position (`Emit::input_at`), no pass, no texture. The turn is a uniform;
//! the format the app gives the node (`TurnParams::output_format`) is what
//! a change of turn actually changes downstream.

use crate::codegen::Emit;
use crate::nodes::NodeDef;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TurnMode {
    /// A quarter turn clockwise.
    #[default]
    Cw,
    /// A quarter turn counter-clockwise.
    Ccw,
    /// A half turn.
    Half,
}

impl TurnMode {
    pub const ALL: [TurnMode; 3] = [TurnMode::Cw, TurnMode::Ccw, TurnMode::Half];

    pub fn name(self) -> &'static str {
        match self {
            TurnMode::Cw => "90° clockwise",
            TurnMode::Ccw => "90° counter-clockwise",
            TurnMode::Half => "180°",
        }
    }

    fn code(self) -> f32 {
        match self {
            TurnMode::Cw => 0.0,
            TurnMode::Ccw => 1.0,
            TurnMode::Half => 2.0,
        }
    }

    /// Does this turn swap width and height?
    pub fn swaps(self) -> bool {
        self != TurnMode::Half
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Default)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct TurnParams {
    pub mode: TurnMode,
    /// Which pipe it sits in: wired a MATTE it turns the coverage and gives
    /// a matte (`NodeKind::follows_pipe`).
    pub matte: bool,
}

impl TurnParams {
    /// The node's format given its input's.
    pub fn output_format(&self, (w, h): (u32, u32)) -> (u32, u32) {
        if self.mode.swaps() { (h, w) } else { (w, h) }
    }

    /// Where output pixel `p` samples the input, `fmt` the node's OWN
    /// (output) format — the CPU mirror of the generated code.
    pub fn lookup(&self, p: [f32; 2], fmt: [f32; 2]) -> [f32; 2] {
        match self.mode {
            // Input (u, v) lands at (h - v, u); h is the output's width.
            TurnMode::Cw => [p[1], fmt[0] - p[0]],
            // Input (u, v) lands at (v, w - u); w is the output's height.
            TurnMode::Ccw => [fmt[1] - p[1], p[0]],
            TurnMode::Half => [fmt[0] - p[0], fmt[1] - p[1]],
        }
    }

    /// (out width, out height, mode, 0).
    pub fn pack_fmt(&self, v: &mut [f32], fmt: [f32; 2]) {
        v[0] = fmt[0];
        v[1] = fmt[1];
        v[2] = self.mode.code();
        v[3] = 0.0;
    }
}

impl NodeDef for TurnParams {
    const NAME: &'static str = "Turn";
    const INPUTS: usize = 1;
    const MASK_INPUT: Option<usize> = None;
    const VEC4_SLOTS: usize = 1;

    fn topo_tag(&self) -> u64 {
        0x7042
    }

    fn pack(&self, v: &mut [f32]) {
        self.pack_fmt(v, [1.0, 1.0]);
    }

    fn emit(&self, e: &mut Emit) -> String {
        if !e.has_input(0) {
            return "vec4f(0.0)".into();
        }
        let slot = e.slot();
        let pos = e.pos().to_string();
        let q = e.fresh();
        e.line(format_args!(
            "let {q} = select(select(vec2f({pos}.y, P.v[{slot}].x - {pos}.x), \
             vec2f(P.v[{slot}].y - {pos}.y, {pos}.x), P.v[{slot}].z > 0.5), \
             P.v[{slot}].xy - {pos}, P.v[{slot}].z > 1.5);"
        ));
        e.input_at(0, &q)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_quarter_turn_swaps_the_format_and_a_half_turn_keeps_it() {
        let t = |mode| TurnParams { mode, matte: false }.output_format((1920, 1080));
        assert_eq!(t(TurnMode::Cw), (1080, 1920));
        assert_eq!(t(TurnMode::Ccw), (1080, 1920));
        assert_eq!(t(TurnMode::Half), (1920, 1080));
    }

    #[test]
    fn each_turn_sends_the_corners_where_they_belong() {
        // A 4x2 input; pixel centres. Its top-left pixel is (0.5, 0.5).
        let at = |mode, p| {
            let tp = TurnParams { mode, matte: false };
            let (w, h) = tp.output_format((4, 2));
            tp.lookup(p, [w as f32, h as f32])
        };
        // Clockwise: the input's top-left lands top-RIGHT of the 2x4 output.
        assert_eq!(at(TurnMode::Cw, [1.5, 0.5]), [0.5, 0.5]);
        // ... and its top-right lands bottom-right.
        assert_eq!(at(TurnMode::Cw, [1.5, 3.5]), [3.5, 0.5]);
        // Counter-clockwise: the input's top-left lands bottom-LEFT.
        assert_eq!(at(TurnMode::Ccw, [0.5, 3.5]), [0.5, 0.5]);
        assert_eq!(at(TurnMode::Ccw, [0.5, 0.5]), [3.5, 0.5]);
        // Half: top-left to bottom-right.
        assert_eq!(at(TurnMode::Half, [3.5, 1.5]), [0.5, 0.5]);
    }
}
