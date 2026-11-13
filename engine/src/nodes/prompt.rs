//! String (was Prompt): a node that holds TEXT, on the text pipe
//! (`PortKind::Text`).
//!
//! **The UI says String; the code and setups still say Prompt** — the
//! Tinter precedent: rename strings, never the variant or what a setup
//! stores, so every saved `Prompt` loads as a String with nothing to
//! migrate. 2dini has its own (`geo::GeoNode::String`) for its Text.
//!
//! The first comp node that is not a picture. It exists so words can
//! live in the graph — one prompt feeding several GenAI nodes, the words a
//! Text node writes — seen and swapped without diving anywhere. Its
//! consumers read the text on the CPU (a GenAI node's `prompt in` step, a
//! Text's words port); the renderer never does:
//!
//! - it packs nothing and its topology never moves, so typing is free;
//! - it is never sampled — a GenAI's text port is outside its pixel
//!   subtree — and viewed on its own it emits transparent black;
//! - it has no picture, so the resolution and frame-range walks
//!   (`PortKind::is_picture`) never follow a text wire.

use crate::codegen::Emit;
use crate::nodes::NodeDef;

/// Topology tag, clear of every other node's span.
const TOPO: u64 = 0x7E57;

#[derive(Clone, Debug, PartialEq, Default)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct PromptParams {
    pub text: String,
}

impl NodeDef for PromptParams {
    const NAME: &'static str = "String";
    const INPUTS: usize = 0;
    const MASK_INPUT: Option<usize> = None;
    const VEC4_SLOTS: usize = 0;

    fn topo_tag(&self) -> u64 {
        TOPO
    }

    fn pack(&self, _span: &mut [f32]) {}

    fn emit(&self, _e: &mut Emit) -> String {
        "vec4f(0.0)".into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_is_never_topology() {
        let a = PromptParams::default();
        let b = PromptParams { text: "a moth on a lamp".into() };
        assert_eq!(a.topo_tag(), b.topo_tag(), "typing is not a recompile");
    }

    #[test]
    fn the_text_pipe_feeds_a_genai_prompt_port_and_nothing_else() {
        use crate::graph::{GenAiParams, NodeKind, PortKind};
        let prompt = NodeKind::Prompt(PromptParams::default());
        let genai = NodeKind::GenAi(GenAiParams::default());
        assert_eq!(prompt.output_kind(), PortKind::Text);
        assert_eq!(genai.input_kind(2), PortKind::Text);
        assert!(PortKind::Text.can_feed(genai.input_kind(2)));
        assert!(!PortKind::Text.can_feed(genai.input_kind(0)), "text is not a picture");
        assert!(!PortKind::Image.can_feed(PortKind::Text));
        assert!(!PortKind::Text.is_picture(), "no resolution or frame range rides a prompt");
        assert_eq!(NodeKind::dot(PortKind::Text).output_kind(), PortKind::Text);
        let text = NodeKind::Text(Default::default());
        assert!(PortKind::Text.can_feed(text.input_kind(crate::nodes::text::WORDS_INPUT)), "a Text's words");
        assert!(!PortKind::Text.can_feed(text.input_kind(0)));
    }
}
