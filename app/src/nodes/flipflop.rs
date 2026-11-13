//! Flip/Flop's app half: the mode menu and the glyph. The model and the
//! WGSL are `engine::nodes::flipflop`. No channels: a mode is a choice,
//! not a value to key.

use crate::widgets::{HudSlider, RowCtx};
use egui::{Color32, Painter, Pos2, Stroke, Vec2};
use engine::graph::{FlipFlopParams, FlipMode};

pub const TILES: usize = 1;

pub fn panel(ui: &mut egui::Ui, rcx: &mut RowCtx, p: &mut FlipFlopParams) -> Option<HudSlider> {
    ui.horizontal(|ui| {
        ui.label("turn");
        egui::ComboBox::from_id_salt(("flipflop-mode", rcx.node))
            .selected_text(p.mode.name())
            .show_ui(ui, |ui| {
                for m in FlipMode::ALL {
                    ui.selectable_value(&mut p.mode, m, m.name());
                }
            });
    });
    ui.weak(match p.mode {
        FlipMode::Flip => "top to bottom",
        FlipMode::Flop => "left to right",
        FlipMode::Both => "both: a half turn",
    });
    None
}

/// The graph glyph: a two-headed arrow each way.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.circle(c, r * 0.9, fill, stroke);
    let ink = Stroke::new((r * 0.12).max(1.0), crate::theme::node_ink());
    let l = r * 0.55;
    let k = r * 0.2;
    for d in [Vec2::new(1.0, 0.0), Vec2::new(0.0, 1.0)] {
        let n = Vec2::new(-d.y, d.x);
        let (a, b) = (c - d * l, c + d * l);
        painter.line_segment([a, b], ink);
        for (tip, s) in [(a, 1.0), (b, -1.0)] {
            painter.line_segment([tip, tip + (d * s + n) * k], ink);
            painter.line_segment([tip, tip + (d * s - n) * k], ink);
        }
    }
}
