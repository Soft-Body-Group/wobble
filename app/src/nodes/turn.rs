//! Turn's app half: the turn menu and the glyph. The model and the WGSL are
//! `engine::nodes::turn`; the format swap is `TurnParams::output_format`,
//! read by `node_resolution`. No channels: a quarter turn is a choice, not
//! a value to key.

use crate::widgets::{HudSlider, RowCtx};
use egui::{Color32, Painter, Pos2, Stroke, Vec2};
use engine::graph::{TurnMode, TurnParams};

pub const TILES: usize = 1;

pub fn panel(ui: &mut egui::Ui, rcx: &mut RowCtx, p: &mut TurnParams) -> Option<HudSlider> {
    ui.horizontal(|ui| {
        ui.label("turn");
        egui::ComboBox::from_id_salt(("turn-mode", rcx.node))
            .selected_text(p.mode.name())
            .show_ui(ui, |ui| {
                for m in TurnMode::ALL {
                    ui.selectable_value(&mut p.mode, m, m.name());
                }
            });
    });
    ui.weak(if p.mode.swaps() {
        "width and height swap"
    } else {
        "the format is kept"
    });
    None
}

/// The graph glyph: three quarters of a circle with an arrowhead.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.circle(c, r * 0.9, fill, stroke);
    let ink = Stroke::new((r * 0.12).max(1.0), crate::theme::node_ink());
    let rad = r * 0.5;
    let at = |a: f32| c + Vec2::new(a.cos(), a.sin()) * rad;
    let (a0, a1) = (-std::f32::consts::FRAC_PI_2, std::f32::consts::PI);
    let n = 12;
    let pts: Vec<Pos2> = (0..=n).map(|i| at(a0 + (a1 - a0) * i as f32 / n as f32)).collect();
    painter.add(egui::Shape::line(pts, ink));
    // The head at the end, pointing along the turn (clockwise on screen).
    let tip = at(a1);
    let d = Vec2::new(-a1.sin(), a1.cos());
    let side = Vec2::new(-d.y, d.x);
    let k = r * 0.22;
    painter.line_segment([tip, tip - d * k + side * k], ink);
    painter.line_segment([tip, tip - d * k - side * k], ink);
}
