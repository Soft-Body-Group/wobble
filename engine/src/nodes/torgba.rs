//! ToRgba: a matte put back on the IMAGE pipe — ToMatte's reverse.
//!
//! The coverage is broadcast to all four channels, (m, m, m, m): what you
//! want to look at a matte as a picture, grade it, blur it with an image
//! filter, or write it out. Alpha carries it too, so the result is already
//! the premultiplied white-on-transparent a Blend expects of a holdout.
//! Unwired, the matte is coverage 1 — the same convention as every mask
//! port — so a bare ToRgba is solid white.

use crate::codegen::Emit;
use crate::nodes::NodeDef;

/// Nothing structural to vary: one constant tag.
const TOPO_BASE: u64 = 0x7E;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct ToRgbaParams {}

impl NodeDef for ToRgbaParams {
    const NAME: &'static str = "ToRgba";
    /// One matte in. It is the node's subject, not a limit on an effect,
    /// so it is not the MASK port: `NodeKind::input_kind` types it.
    const INPUTS: usize = 1;
    const MASK_INPUT: Option<usize> = None;
    const VEC4_SLOTS: usize = 0;

    fn topo_tag(&self) -> u64 {
        TOPO_BASE
    }

    fn pack(&self, _v: &mut [f32]) {}

    fn emit(&self, e: &mut Emit) -> String {
        let v = e.fresh();
        match e.mask(0) {
            Some(m) => e.line(format_args!("let {v} = vec4f({m}.x);")),
            None => e.line(format_args!("let {v} = vec4f(1.0);")),
        }
        v
    }
}
