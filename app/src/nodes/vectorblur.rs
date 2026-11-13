//! VectorBlur's app half: the shutter and the sample count, a glyph of a
//! disc smeared along an arrow. The gather is `engine::nodes::vectorblur`.

use crate::anim::Channel;
use crate::widgets::{self, HudSlider, RowCtx};
use egui::{Color32, Painter, Pos2, Stroke, Vec2};
use engine::graph::VectorBlurParams;
use engine::nodes::vectorblur::MAX_SAMPLES;

pub const CHANNELS: &[Channel] = &[Channel::VectorBlurAmount, Channel::VectorBlurOffset];

pub const TILES: usize = 1;

pub fn get(p: &VectorBlurParams, ch: Channel) -> Option<f32> {
    match ch {
        Channel::VectorBlurAmount => Some(p.amount),
        Channel::VectorBlurOffset => Some(p.offset),
        _ => None,
    }
}

pub fn set(p: &mut VectorBlurParams, ch: Channel, v: f32) {
    match ch {
        Channel::VectorBlurAmount => p.amount = v,
        Channel::VectorBlurOffset => p.offset = v,
        _ => {}
    }
}

pub fn label(ch: Channel) -> Option<&'static str> {
    match ch {
        Channel::VectorBlurAmount => Some("vector blur shutter"),
        Channel::VectorBlurOffset => Some("vector blur offset"),
        _ => None,
    }
}

pub fn panel(ui: &mut egui::Ui, rcx: &mut RowCtx, p: &mut VectorBlurParams) -> Option<HudSlider> {
    let h = widgets::slider(ui, rcx, "shutter", &mut p.amount, 0.0, 2.0, &[Channel::VectorBlurAmount]);
    let h2 = widgets::slider(ui, rcx, "offset", &mut p.offset, -0.5, 0.5, &[Channel::VectorBlurOffset]);
    let mut n = p.samples as f32;
    widgets::slider(ui, rcx, "samples", &mut n, 2.0, MAX_SAMPLES as f32, &[]);
    p.samples = n.round().clamp(1.0, MAX_SAMPLES as f32) as u32;
    ui.weak("smears along the vectors on port v (a Flow). 0.5 = 180°");
    h.or(h2)
}

/// A disc dragged along an arrow: motion read off a picture.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let head = c + Vec2::new(r * 0.38, -r * 0.1);
    for k in 1..=3 {
        let t = k as f32 * 0.22;
        painter.circle_filled(head - Vec2::new(r * t * 1.6, -r * t * 0.4), r * 0.5, fill.gamma_multiply(0.22 * (4 - k) as f32));
    }
    painter.circle(head, r * 0.5, fill, stroke);
    let ink = Stroke::new((r * 0.12).max(1.0), crate::theme::node_ink());
    let (a, b) = (head - Vec2::new(r * 0.3, -r * 0.08), head + Vec2::new(r * 0.3, -r * 0.08));
    painter.line_segment([a, b], ink);
    painter.line_segment([b, b - Vec2::new(r * 0.16, r * 0.1)], ink);
    painter.line_segment([b, b - Vec2::new(r * 0.12, -r * 0.16)], ink);
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::graph::NodeKind;

    #[test]
    fn the_panel_fits_its_tile() {
        let mut p = VectorBlurParams::default();
        let mut anim = Default::default();
        let ekind = NodeKind::VectorBlur(p);
        let (over, wide) = (std::cell::Cell::new(0.0f32), std::cell::Cell::new(0.0f32));
        let t = crate::widgets::TILE;
        let shot = crate::snapshot::shoot(t - 14.0, t - 14.0 - 22.0, |ui| {
            let out = egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                let mut rcx = RowCtx { node: 1, anim: &mut anim, ekind: &ekind, t: 0, hud_live: false, driven: Vec::new() };
                let _ = panel(ui, &mut rcx, &mut p);
            });
            over.set(out.content_size.y - out.inner_rect.height());
            wide.set(out.content_size.x - out.inner_rect.width());
        });
        shot.save("vectorblur-panel");
        assert!(over.get() <= 0.5, "overflows its tile DOWN by {}pt", over.get());
        assert!(wide.get() <= 0.5, "overflows its tile SIDEWAYS by {}pt", wide.get());
    }
}
