//! Blank's app half: the panel and the glyph. The params and the WGSL are
//! `engine::nodes::blank`.

use crate::anim::Channel;
use crate::colorpicker::{self, PickSpec};
use crate::widgets;
use egui::{Painter, Pos2, Stroke};
use engine::graph::BlankParams;

/// Nothing keyable: the colour is a swatch, like a Grid's line and back.
pub const CHANNELS: &[Channel] = &[];

pub fn label(_ch: Channel) -> Option<&'static str> {
    None
}

/// The properties panel: the format row every generator has, and the
/// colour.
pub fn panel(ui: &mut egui::Ui, node: u64, p: &mut BlankParams, spec: PickSpec) {
    widgets::gen_format_row(ui, ("blank-format", node), &mut p.size, &mut p.inherit);
    ui.horizontal(|ui| {
        ui.add_sized([50.0, 16.0], egui::Label::new(egui::RichText::new("colour").small()));
        colorpicker::button(ui, (node, "blank-color"), &mut p.color, spec)
            .on_hover_text("colour and alpha (output is premultiplied)");
    });
    ui.weak("one colour across its own format — a source, like a Load");
}

/// The graph glyph: a flat disc in the node's colour.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: egui::Color32, stroke: Stroke) {
    painter.circle(c, r * 0.9, fill, stroke);
}
