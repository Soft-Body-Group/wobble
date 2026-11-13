//! ToData: one channel of an image, put on the DATA pipe.
//!
//! The data pipe ([`crate::graph::PortKind::Data`]) carries ONE channel at
//! full depth — a depth pass, a disparity, a thickness, an ID. It is the
//! matte pipe's twin and its opposite at the same time:
//!
//! | | matte | data |
//! |---|---|---|
//! | channels | 1 | 1 |
//! | range | 0..1, guaranteed | whatever the number is |
//! | precision | as little as coverage needs | full |
//!
//! That difference is the whole reason for a second single-channel pipe.
//! A matte is coverage: it is defined on 0..1, and a compositor is free to
//! keep it cheaply, because a value outside that range would be a bug. A
//! depth is a measurement: clamping it at 1 destroys the shot, and so does
//! quantising it — a defocus reads the SPACING between depths, and 8 bits
//! of spacing is eight visible steps of blur. So a node that wants a
//! measurement asks for `Data`, and the wiring rules make it impossible to
//! hand it a matte's worth of precision by accident
//! ([`crate::graph::PortKind::can_feed`]).
//!
//! This node is the way IN to that pipe: an image on the input, a channel
//! pick, and the value comes out untouched. No clamp, no re-range — the
//! two things [`crate::graph::NodeKind::ToMatte`] does on purpose and this
//! must never do. The picked value is BROADCAST to rgb, so a consumer that
//! still names a channel of its own (a Defocus's Z channel) reads the same
//! number whichever it names, and no one can wire a data pipe up wrongly
//! enough to get a different answer.

use crate::codegen::Emit;
use crate::graph::MatteChannel;
use crate::nodes::NodeDef;

/// Topology tag base. The channel pick is a UNIFORM (a one-hot vector), so
/// changing it never recompiles — the generated code is the same dot
/// product whichever channel it names.
const TOPO_BASE: u64 = 0x7D;

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct ToDataParams {
    /// Which channel of the input carries the value. `Luma` is offered
    /// because a single-channel EXR layer usually decodes to (z, z, z, 1),
    /// where every answer is the same one.
    pub channel: MatteChannel,
}

impl Default for ToDataParams {
    fn default() -> Self {
        Self { channel: MatteChannel::default() }
    }
}

impl NodeDef for ToDataParams {
    const NAME: &'static str = "ToData";
    /// One image in. No mask: a mask would limit an EFFECT, and this node
    /// has none — it reads a number and passes it on.
    const INPUTS: usize = 1;
    const MASK_INPUT: Option<usize> = None;
    /// The channel weights, one-hot or luma.
    const VEC4_SLOTS: usize = 1;

    fn topo_tag(&self) -> u64 {
        TOPO_BASE
    }

    fn pack(&self, v: &mut [f32]) {
        v[..4].copy_from_slice(&self.channel.weights());
    }

    fn emit(&self, e: &mut Emit) -> String {
        let input = e.input(0);
        let slot = e.slot();
        let v = e.fresh();
        // A dot with the one-hot row, and NOTHING else: no clamp and no
        // re-range, which is exactly what separates this from ToMatte.
        // Broadcast, so a downstream channel pick cannot disagree with it.
        e.line(format_args!(
            "let {v} = vec4f(vec3f(dot({input}, P.v[{slot}])), 1.0);"
        ));
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The one thing this node must never grow: a clamp. A depth of 240
    /// has to arrive as 240.
    #[test]
    fn the_emitted_code_neither_clamps_nor_re_ranges() {
        let src = include_str!("todata.rs");
        // The emit line itself, not this test's own text.
        // Code only: the comments are allowed to SAY "no clamp".
        let emit: String = src
            .split("fn emit(&self")
            .nth(1)
            .and_then(|s| s.split("\n    }").next())
            .unwrap()
            .lines()
            .map(|l| l.split("//").next().unwrap_or(""))
            .collect::<Vec<_>>()
            .join("\n");
        for banned in ["clamp", "saturate", "min(", "max("] {
            assert!(!emit.contains(banned), "ToData::emit must not {banned}: {emit}");
        }
    }

    /// The channel is a uniform, so switching it is a uniform write — the
    /// same rule every other param obeys.
    #[test]
    fn the_channel_pick_is_not_topology() {
        let r = ToDataParams { channel: MatteChannel::R };
        let a = ToDataParams { channel: MatteChannel::A };
        assert_eq!(r.topo_tag(), a.topo_tag());
        let (mut vr, mut va) = ([0.0; 4], [0.0; 4]);
        r.pack(&mut vr);
        a.pack(&mut va);
        assert_ne!(vr, va, "...but it must still reach the uniform");
    }
}
