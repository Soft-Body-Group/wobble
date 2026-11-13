//! UV Warp's app half: the filter pick, the V orientation and a glyph. The
//! lookup, the WGSL and the params are `engine::nodes::uvwarp`.

use crate::widgets::{HudSlider, RowCtx};
use egui::{Color32, Painter, Pos2, Stroke, Vec2};
use engine::graph::{UvWarpFilter, UvWarpParams};

/// No channels, like ToData: both controls are choices, not numbers to
/// drag or key.
pub const TILES: usize = 1;

pub fn panel(ui: &mut egui::Ui, _rcx: &mut RowCtx, p: &mut UvWarpParams) -> Option<HudSlider> {
    ui.horizontal(|ui| {
        ui.add_sized([50.0, 16.0], egui::Label::new(egui::RichText::new("filter").small()));
        ui.selectable_value(&mut p.filter, UvWarpFilter::Bilinear, "bilinear");
        ui.selectable_value(&mut p.filter, UvWarpFilter::Nearest, "nearest");
    });
    ui.checkbox(&mut p.v_up, "v up");
    ui.weak("source sampled at the map's (red, green)");
    ui.weak("0..1 over the source's format");
    None
}

/// A frame with its lines bent: a warp you were handed as a picture.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let h = r * 0.8;
    painter.rect(
        egui::Rect::from_center_size(c, egui::vec2(h * 2.0, h * 2.0)),
        2.0,
        fill,
        stroke,
        egui::StrokeKind::Middle,
    );
    let ink = Stroke::new((r * 0.11).max(1.0), crate::theme::node_ink());
    let n = 12;
    for k in [-1.0f32, 0.0, 1.0] {
        let off = k * h * 0.5;
        // One bowed line each way, bending further from the middle.
        let across: Vec<Pos2> = (0..=n)
            .map(|i| {
                let t = i as f32 / n as f32 * 2.0 - 1.0;
                c + Vec2::new(t * h * 0.8, off + (1.0 - t * t) * h * 0.22 * (1.0 + k.abs()))
            })
            .collect();
        let down: Vec<Pos2> = (0..=n)
            .map(|i| {
                let t = i as f32 / n as f32 * 2.0 - 1.0;
                c + Vec2::new(off + (1.0 - t * t) * h * 0.22 * (1.0 + k.abs()), t * h * 0.8)
            })
            .collect();
        painter.add(egui::Shape::line(across, ink));
        painter.add(egui::Shape::line(down, ink));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::graph::NodeKind;

    /// The panel fits its one tile, both ways.
    #[test]
    fn the_panel_fits_its_tile() {
        let mut p = UvWarpParams::default();
        let mut anim = Default::default();
        let ekind = NodeKind::UvWarp(p);
        let (over, wide) = (std::cell::Cell::new(0.0f32), std::cell::Cell::new(0.0f32));
        let t = crate::widgets::TILE;
        let shot = crate::snapshot::shoot(t - 14.0, t - 14.0 - 22.0, |ui| {
            let out = egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                let mut rcx = RowCtx {
                    node: 1,
                    anim: &mut anim,
                    ekind: &ekind,
                    t: 0,
                    hud_live: false,
                    driven: Vec::new(),
                };
                let _ = panel(ui, &mut rcx, &mut p);
            });
            over.set(out.content_size.y - out.inner_rect.height());
            wide.set(out.content_size.x - out.inner_rect.width());
        });
        shot.save("uvwarp-panel");
        assert!(over.get() <= 0.5, "overflows its tile DOWN by {}pt", over.get());
        assert!(wide.get() <= 0.5, "overflows its tile SIDEWAYS by {}pt", wide.get());
    }
}
