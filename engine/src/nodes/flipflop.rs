//! Flip/Flop: turn the picture over. FLIP is top to bottom (a vertical
//! flip), FLOP left to right, BOTH is the two (a half turn). It fuses: the
//! input is evaluated at the mirrored position over the node's own format.
//! The mode is a uniform, so switching it never recompiles.

use crate::codegen::Emit;
use crate::nodes::NodeDef;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FlipMode {
    /// Top to bottom.
    #[default]
    Flip,
    /// Left to right.
    Flop,
    /// Both: a half turn.
    Both,
}

impl FlipMode {
    pub const ALL: [FlipMode; 3] = [FlipMode::Flip, FlipMode::Flop, FlipMode::Both];

    pub fn name(self) -> &'static str {
        match self {
            FlipMode::Flip => "flip (vertical)",
            FlipMode::Flop => "flop (horizontal)",
            FlipMode::Both => "both",
        }
    }

    /// (mirror x, mirror y).
    pub fn axes(self) -> (bool, bool) {
        match self {
            FlipMode::Flip => (false, true),
            FlipMode::Flop => (true, false),
            FlipMode::Both => (true, true),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Default)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct FlipFlopParams {
    pub mode: FlipMode,
    /// Which pipe it sits in: wired a MATTE it turns the coverage over and
    /// gives a matte (`NodeKind::follows_pipe`).
    pub matte: bool,
}

impl FlipFlopParams {
    /// Where output pixel `p` samples the input in a `fmt`-pixel format —
    /// the CPU mirror of the generated code.
    pub fn lookup(&self, p: [f32; 2], fmt: [f32; 2]) -> [f32; 2] {
        let (x, y) = self.mode.axes();
        [if x { fmt[0] - p[0] } else { p[0] }, if y { fmt[1] - p[1] } else { p[1] }]
    }

    /// (width, height, mirror x, mirror y).
    pub fn pack_fmt(&self, v: &mut [f32], fmt: [f32; 2]) {
        let (x, y) = self.mode.axes();
        v[0] = fmt[0];
        v[1] = fmt[1];
        v[2] = x as u32 as f32;
        v[3] = y as u32 as f32;
    }
}

impl NodeDef for FlipFlopParams {
    const NAME: &'static str = "FlipFlop";
    /// The picture, then the mask.
    const INPUTS: usize = 2;
    const MASK_INPUT: Option<usize> = Some(1);
    const VEC4_SLOTS: usize = 1;

    fn topo_tag(&self) -> u64 {
        0xF11F
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
            "let {q} = select({pos}, P.v[{slot}].xy - {pos}, P.v[{slot}].zw > vec2f(0.5));"
        ));
        let out = e.input_at(0, &q);
        if e.has_input(1) {
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
    fn each_mode_turns_its_axes() {
        let f = [100.0, 50.0];
        let at = |mode| FlipFlopParams { mode, matte: false }.lookup([10.0, 5.0], f);
        assert_eq!(at(FlipMode::Flip), [10.0, 45.0]);
        assert_eq!(at(FlipMode::Flop), [90.0, 5.0]);
        assert_eq!(at(FlipMode::Both), [90.0, 45.0]);
    }
}
