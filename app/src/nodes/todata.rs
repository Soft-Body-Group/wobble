//! ToData's app half: a channel pick, and a glyph that says "one channel,
//! full depth". The maths and the params are `engine::nodes::todata`.

use crate::widgets::{HudSlider, RowCtx};
use egui::{Color32, Painter, Pos2, Stroke};
use engine::graph::{MatteChannel, ToDataParams};

/// No channels, so no `CHANNELS`/`get`/`set`/`label`: the only control is
/// WHICH channel carries the value, and a channel pick is a choice, not a
/// number you drag or key.
pub const TILES: usize = 1;

pub fn panel(ui: &mut egui::Ui, _rcx: &mut RowCtx, p: &mut ToDataParams) -> Option<HudSlider> {
    ui.horizontal_wrapped(|ui| {
        ui.add_sized([50.0, 16.0], egui::Label::new(egui::RichText::new("channel").small()));
        for c in MatteChannel::ALL {
            ui.selectable_value(&mut p.channel, c, c.name());
        }
    });
    ui.weak("one channel on the DATA pipe: no clamp, no re-range");
    ui.weak("for a measurement — depth, disparity — not for coverage");
    None
}

/// A single lane running full height: the matte glyph's shape with its
/// range let out, which is exactly the difference between the two nodes.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let w = r * 0.42;
    painter.rect(
        egui::Rect::from_center_size(c, egui::vec2(w * 2.0, r * 1.5)),
        1.0,
        fill,
        stroke,
        egui::StrokeKind::Middle,
    );
    // The lane's value keeps going past the box, both ways.
    for s in [-1.0f32, 1.0] {
        painter.line_segment(
            [c + egui::vec2(0.0, s * r * 0.75), c + egui::vec2(0.0, s * r * 1.15)],
            stroke,
        );
    }
}
