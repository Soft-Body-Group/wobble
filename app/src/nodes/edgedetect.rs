//! Edge Detect's app half: operator and output picks, four sliders, an
//! outlined-square glyph.

use crate::anim::Channel;
use crate::widgets::{self, HudSlider, RowCtx};
use egui::{Color32, Painter, Pos2, Rect, Stroke, Vec2};
use engine::graph::{EdgeDetectParams, EdgeOp, EdgeOutput, EDGE_SIZE_MAX};

pub const CHANNELS: &[Channel] =
    &[Channel::EdgeSize, Channel::EdgeGain, Channel::EdgeThreshold, Channel::EdgeSoftness];

pub const TILES: usize = 2;

pub fn get(p: &EdgeDetectParams, ch: Channel) -> Option<f32> {
    match ch {
        Channel::EdgeSize => Some(p.size),
        Channel::EdgeGain => Some(p.gain),
        Channel::EdgeThreshold => Some(p.threshold),
        Channel::EdgeSoftness => Some(p.softness),
        _ => None,
    }
}

pub fn set(p: &mut EdgeDetectParams, ch: Channel, v: f32) {
    match ch {
        Channel::EdgeSize => p.size = v,
        Channel::EdgeGain => p.gain = v,
        Channel::EdgeThreshold => p.threshold = v,
        Channel::EdgeSoftness => p.softness = v,
        _ => {}
    }
}

pub fn label(ch: Channel) -> Option<&'static str> {
    match ch {
        Channel::EdgeSize => Some("edge size"),
        Channel::EdgeGain => Some("edge gain"),
        Channel::EdgeThreshold => Some("edge threshold"),
        Channel::EdgeSoftness => Some("edge softness"),
        _ => None,
    }
}

pub fn panel(ui: &mut egui::Ui, rcx: &mut RowCtx, p: &mut EdgeDetectParams) -> Option<HudSlider> {
    ui.horizontal(|ui| {
        ui.label("operator");
        egui::ComboBox::from_id_salt(("edge-op", rcx.node))
            .selected_text(p.op.name())
            .show_ui(ui, |ui| {
                for o in EdgeOp::ALL {
                    ui.selectable_value(&mut p.op, o, o.name());
                }
            });
        ui.add_space(8.0);
        ui.label("output");
        egui::ComboBox::from_id_salt(("edge-out", rcx.node))
            .selected_text(p.output.name())
            .show_ui(ui, |ui| {
                for o in EdgeOutput::ALL {
                    ui.selectable_value(&mut p.output, o, o.name());
                }
            });
    });
    let mut h = None;
    for got in [
        widgets::slider(ui, rcx, "size", &mut p.size, 0.5, EDGE_SIZE_MAX, &[Channel::EdgeSize]),
        widgets::slider(ui, rcx, "gain", &mut p.gain, 0.0, 10.0, &[Channel::EdgeGain]),
    ] {
        h = h.or(got);
    }
    if p.output == EdgeOutput::Matte {
        for got in [
            widgets::slider(ui, rcx, "threshold", &mut p.threshold, 0.0, 1.0, &[Channel::EdgeThreshold]),
            widgets::slider(ui, rcx, "softness", &mut p.softness, 0.0, 1.0, &[Channel::EdgeSoftness]),
        ] {
            h = h.or(got);
        }
        ui.weak("strongest channel's edge, thresholded: a matte");
    } else {
        ui.weak("rgb = each channel's edge strength; alpha passes");
    }
    ui.weak("a hard step of 1 reads 1 under every operator");
    h
}

/// A square drawn as its outline only: what is left of a shape once the
/// flat parts are gone.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let outer = Rect::from_center_size(c, Vec2::splat(r * 1.6));
    painter.rect_filled(outer, 0.0, fill);
    painter.rect_stroke(outer, 0.0, stroke, egui::StrokeKind::Middle);
    painter.rect_stroke(outer.shrink(r * 0.32), 0.0, stroke, egui::StrokeKind::Middle);
}
