//! String's app half (the node was Prompt, and the code still says so):
//! the panel and the glyph. The params are `engine::nodes::prompt`; the
//! text travels on the text pipe into a GenAI node's prompt port (a
//! `prompt in` step reads it at run time) or a Text node's words port.

use crate::anim::Channel;
use egui::{Painter, Pos2, Stroke};
use engine::graph::PromptParams;

/// Nothing keyable: text is not a curve.
pub const CHANNELS: &[Channel] = &[];
/// Two tiles wide: room for a few lines of prompt.
pub const TILES: usize = 2;

pub fn label(_ch: Channel) -> Option<&'static str> {
    None
}

/// The properties panel: the text, as big as the tile allows.
pub fn panel(ui: &mut egui::Ui, node: u64, p: &mut PromptParams) {
    ui.add(
        egui::TextEdit::multiline(&mut p.text)
            .id(egui::Id::new(("prompt-text", node)))
            .hint_text("words: a prompt, a title")
            .desired_rows(6)
            .desired_width(f32::INFINITY),
    );
    ui.weak(egui::RichText::new("wire into a Text node's words, or a GenAI node's prompt port").small());
}

/// The graph glyph: a page with three lines of text on it.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: egui::Color32, stroke: Stroke) {
    let page = egui::Rect::from_center_size(c, egui::vec2(r * 1.3, r * 1.5));
    painter.rect(page, r * 0.15, fill, stroke, egui::StrokeKind::Middle);
    let ink = Stroke::new(stroke.width.max(1.0), stroke.color);
    for (k, w) in [(0.0f32, 0.8f32), (1.0, 0.8), (2.0, 0.5)] {
        let y = page.top() + r * (0.4 + 0.35 * k);
        let x0 = page.left() + r * 0.22;
        painter.line_segment([egui::pos2(x0, y), egui::pos2(x0 + r * w, y)], ink);
    }
}
