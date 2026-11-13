//! Blank: a generator that fills its format with one colour — a
//! Constant. The plainest source there is: a backing plate, a solid to
//! merge over, a known value to grade against.
//!
//! The colour is authored STRAIGHT (unpremultiplied) in the working space
//! with its alpha, and output premultiplied, like the Gradient's stops. It
//! is one uniform, so every edit is a uniform write and the node has one
//! topology whatever it holds.

use crate::codegen::Emit;
use crate::nodes::NodeDef;

/// Topology tag, clear of every other node's span.
const TOPO: u64 = 0x6B00;

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct BlankParams {
    /// The node's format, when it declares one (see `inherit`).
    pub size: (u32, u32),
    /// Take the comp's canvas instead of `size` — the format row's
    /// "inherit", resolved by the app like a roto shape's.
    pub inherit: bool,
    /// Straight colour in the working space, and alpha.
    pub color: [f32; 4],
}

impl Default for BlankParams {
    fn default() -> Self {
        Self { size: (1920, 1080), inherit: false, color: [0.0, 0.0, 0.0, 1.0] }
    }
}

impl NodeDef for BlankParams {
    const NAME: &'static str = "Blank";
    const INPUTS: usize = 0;
    const MASK_INPUT: Option<usize> = None;
    /// The colour.
    const VEC4_SLOTS: usize = 1;

    fn topo_tag(&self) -> u64 {
        TOPO
    }

    fn pack(&self, v: &mut [f32]) {
        v[..4].copy_from_slice(&self.color);
    }

    fn emit(&self, e: &mut Emit) -> String {
        let slot = e.slot();
        let v = e.fresh();
        e.line(format_args!("let {v} = vec4f(P.v[{slot}u].rgb * P.v[{slot}u].a, P.v[{slot}u].a);"));
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packs_the_straight_colour_and_never_moves_the_topology() {
        let a = BlankParams::default();
        let b = BlankParams { color: [0.2, 0.4, 0.6, 0.5], size: (64, 64), ..a };
        assert_eq!(a.topo_tag(), b.topo_tag(), "a colour or size edit is a uniform, not a recompile");
        let mut v = [0.0; 4];
        b.pack(&mut v);
        assert_eq!(v, [0.2, 0.4, 0.6, 0.5]);
    }
}
