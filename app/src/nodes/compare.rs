//! Compare's app half: the panel, the glyph and the channel binding. The
//! model, the fit and the WGSL are `engine::nodes::compare`; the viewer
//! handle (drag the wipe line, turn it by its arrow) is in
//! `viewer/transform.rs` beside the Mirror's, and the format (the larger
//! input's) in `lib.rs`'s `node_resolution`.

use crate::anim::Channel;
use crate::widgets::{self, HudSlider, RowCtx};
use egui::{Color32, Painter, Pos2, Stroke};
use engine::graph::{CompareFit, CompareMode, CompareParams};

pub const CHANNELS: &[Channel] = &[Channel::CompareX, Channel::CompareY, Channel::CompareAngle];

pub const TILES: usize = 2;

pub fn get(p: &CompareParams, ch: Channel) -> Option<f32> {
    match ch {
        Channel::CompareX => Some(p.center[0]),
        Channel::CompareY => Some(p.center[1]),
        Channel::CompareAngle => Some(p.angle),
        _ => None,
    }
}

pub fn set(p: &mut CompareParams, ch: Channel, v: f32) {
    match ch {
        Channel::CompareX => p.center[0] = v,
        Channel::CompareY => p.center[1] = v,
        Channel::CompareAngle => p.angle = v,
        _ => {}
    }
}

pub fn label(ch: Channel) -> Option<&'static str> {
    match ch {
        Channel::CompareX => Some("wipe x"),
        Channel::CompareY => Some("wipe y"),
        Channel::CompareAngle => Some("wipe angle"),
        _ => None,
    }
}

pub fn panel(ui: &mut egui::Ui, rcx: &mut RowCtx, p: &mut CompareParams) -> Option<HudSlider> {
    ui.horizontal(|ui| {
        ui.label("show");
        for m in CompareMode::ALL {
            ui.selectable_value(&mut p.mode, m, m.name());
        }
        // One click flips between the two pictures, from wipe too.
        if ui.button("flip").on_hover_text("flip between A and B").clicked() {
            p.mode = if p.mode == CompareMode::A { CompareMode::B } else { CompareMode::A };
        }
    });
    ui.horizontal(|ui| {
        ui.label("aspects differ");
        egui::ComboBox::from_id_salt(("compare-fit", rcx.node))
            .selected_text(p.fit.name())
            .show_ui(ui, |ui| {
                for f in CompareFit::ALL {
                    ui.selectable_value(&mut p.fit, f, f.name());
                }
            });
    });
    let mut h = widgets::drags(
        ui,
        rcx,
        "wipe at",
        {
            let [x, y] = &mut p.center;
            &mut [x, y]
        },
        0.002,
        (0.0, 1.0),
        &[Channel::CompareX, Channel::CompareY],
    );
    h = h.or(widgets::slider(ui, rcx, "angle", &mut p.angle, -180.0, 180.0, &[Channel::CompareAngle]));
    ui.weak("A shows where the arrow points · drag the line in the viewer, turn it by its arrow");
    ui.weak("the smaller input is scaled up into the larger one's format");
    h
}

/// The graph glyph: a frame split down the middle, the A half inked.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let rect = egui::Rect::from_center_size(c, egui::vec2(r * 1.8, r * 1.4));
    painter.rect(rect, r * 0.25, fill, stroke, egui::StrokeKind::Middle);
    let ink = Stroke::new((r * 0.12).max(1.0), crate::theme::node_ink());
    let inner = rect.shrink(r * 0.3);
    let half = egui::Rect::from_min_max(inner.min, egui::pos2(c.x, inner.max.y));
    painter.rect_filled(half, 0.0, ink.color);
    painter.rect_stroke(inner, 0.0, ink, egui::StrokeKind::Middle);
    painter.line_segment([egui::pos2(c.x, rect.min.y), egui::pos2(c.x, rect.max.y)], ink);
}
