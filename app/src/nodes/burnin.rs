//! Burnin's app half: the panel, the glyph and the channel binding. The
//! params and the WGSL are `engine::nodes::burnin`; the glyphs share the
//! Text node's atlas machinery (`engine::textraster`).

use crate::anim::Channel;
use crate::colorpicker::{self, PickSpec};
use crate::widgets::{self, HudSlider, RowCtx};
use egui::{Color32, Painter, Pos2, Stroke, Vec2};
use engine::graph::{BurninCorner, BurninParams};

pub const CHANNELS: &[Channel] =
    &[Channel::BurninSize, Channel::BurninMargin, Channel::BurninX, Channel::BurninY];

pub const TILES: usize = 2;

pub fn get(p: &BurninParams, ch: Channel) -> Option<f32> {
    match ch {
        Channel::BurninSize => Some(p.size),
        Channel::BurninMargin => Some(p.margin),
        Channel::BurninX => Some(p.position[0]),
        Channel::BurninY => Some(p.position[1]),
        _ => None,
    }
}

pub fn set(p: &mut BurninParams, ch: Channel, v: f32) {
    match ch {
        Channel::BurninSize => p.size = v.max(0.0),
        Channel::BurninMargin => p.margin = v,
        Channel::BurninX => p.position[0] = v,
        Channel::BurninY => p.position[1] = v,
        _ => {}
    }
}

pub fn label(ch: Channel) -> Option<&'static str> {
    match ch {
        Channel::BurninSize => Some("burnin size"),
        Channel::BurninMargin => Some("burnin margin"),
        Channel::BurninX => Some("burnin x"),
        Channel::BurninY => Some("burnin y"),
        _ => None,
    }
}

fn small(ui: &mut egui::Ui, s: &str) {
    ui.add_sized([widgets::LABEL_W, 16.0], egui::Label::new(egui::RichText::new(s).small()));
}

pub fn panel(ui: &mut egui::Ui, rcx: &mut RowCtx, p: &mut BurninParams, spec: PickSpec) -> Option<HudSlider> {
    use Channel::*;
    let node = rcx.node;
    ui.horizontal(|ui| {
        small(ui, "text");
        ui.add(egui::TextEdit::singleline(&mut p.format).desired_width(f32::INFINITY))
            .on_hover_text(format!("{} is replaced by the frame number", engine::nodes::burnin::TOKEN));
    });
    crate::nodes::text::font_row(ui, &mut p.font);
    ui.horizontal(|ui| {
        small(ui, "corner");
        egui::ComboBox::from_id_salt(("burnin-corner", node))
            .width(96.0)
            .selected_text(p.corner.name())
            .show_ui(ui, |ui| {
                for c in BurninCorner::ALL {
                    ui.selectable_value(&mut p.corner, c, c.name());
                }
            });
        ui.label(egui::RichText::new("pad").small());
        widgets::num_field(ui, egui::DragValue::new(&mut p.pad).range(0..=12))
            .on_hover_text("at least this many digits, zero-filled");
        ui.label(egui::RichText::new("offset").small());
        widgets::num_field(ui, egui::DragValue::new(&mut p.offset)).on_hover_text("added to the frame");
    });
    ui.horizontal(|ui| {
        small(ui, "colour");
        colorpicker::button(ui, (node, "burnin-color"), &mut p.color, spec).on_hover_text("text colour and alpha");
        ui.add_space(8.0);
        ui.checkbox(&mut p.boxed, "box");
        colorpicker::button(ui, (node, "burnin-box"), &mut p.box_color, spec)
            .on_hover_text("box colour and alpha");
    });
    let mut h = widgets::slider(ui, rcx, "size", &mut p.size, 4.0, 200.0, &[BurninSize]);
    if p.corner == BurninCorner::Custom {
        let [x, y] = &mut p.position;
        h = widgets::drags(ui, rcx, "position", &mut [x, y], 1.0, (0.0, 1920.0), &[BurninX, BurninY]).or(h);
    } else {
        h = widgets::slider(ui, rcx, "margin", &mut p.margin, 0.0, 200.0, &[BurninMargin]).or(h);
    }
    ui.weak(format!("reads \u{201c}{}\u{201d} at frame 1001", p.text_at(1001 + p.offset as i64)));
    h
}

/// The graph glyph: a frame counter — a box with digits' worth of ticks.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.circle(c, r * 0.9, fill, stroke);
    let ink = Stroke::new((r * 0.11).max(1.0), crate::theme::node_ink());
    let (w, h) = (r * 0.5, r * 0.3);
    let rect = egui::Rect::from_center_size(c, Vec2::new(2.0 * w, 2.0 * h));
    painter.rect_stroke(rect, 1.0, ink, egui::StrokeKind::Middle);
    for k in 0..4 {
        let x = c.x - w + (k as f32 + 0.5) * (2.0 * w / 4.0);
        painter.line_segment([Pos2::new(x, c.y - h * 0.5), Pos2::new(x, c.y + h * 0.5)], ink);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::graph::NodeKind;

    #[test]
    fn the_panel_fits_the_tiles_it_asks_for() {
        for corner in [BurninCorner::BottomLeft, BurninCorner::Custom] {
            let mut p = BurninParams { corner, ..Default::default() };
            let mut anim = Default::default();
            let ekind = NodeKind::Burnin(p.clone());
            let over = std::cell::Cell::new(0.0f32);
            let wide = std::cell::Cell::new(0.0f32);
            let span = |n: usize| crate::widgets::TILE * n as f32 + crate::widgets::TILE_GAP * (n as f32 - 1.0);
            let (w, hgt) = (span(TILES) - 14.0, span(1) - 14.0 - 22.0);
            let shot = crate::snapshot::shoot(w, hgt, |ui| {
                let out = egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                    let mut rcx =
                        RowCtx { node: 1, anim: &mut anim, ekind: &ekind, t: 0, hud_live: false, driven: Vec::new() };
                    let _ = panel(ui, &mut rcx, &mut p, PickSpec::rgba());
                });
                over.set(out.content_size.y - out.inner_rect.height());
                wide.set(out.content_size.x - out.inner_rect.width());
            });
            shot.save("burnin-panel");
            assert!(over.get() <= 0.5, "{corner:?}: the panel overflows its tiles by {}pt", over.get());
            assert!(wide.get() <= 0.5, "{corner:?}: the panel overflows SIDEWAYS by {}pt", wide.get());
        }
    }
}
