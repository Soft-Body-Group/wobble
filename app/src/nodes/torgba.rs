//! ToRgba's app half: no controls, and a glyph that is ToMatte's turned
//! round — matte in on the left, picture out on the right. The node itself
//! is `engine::nodes::torgba`.

use crate::widgets::{HudSlider, RowCtx};
use egui::{Color32, Painter, Pos2, Rect, Stroke, Vec2};
use engine::graph::ToRgbaParams;

/// No channels: nothing to drag, key or patch.
pub const TILES: usize = 1;

pub fn panel(ui: &mut egui::Ui, _rcx: &mut RowCtx, _p: &mut ToRgbaParams) -> Option<HudSlider> {
    ui.weak("the matte on the image pipe: coverage in r, g, b AND alpha");
    ui.weak("unwired it is coverage 1 — solid white");
    None
}

/// A square whose LEFT half is the mask colour it takes in.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let h = r * 0.8;
    painter.rect(
        Rect::from_center_size(c, Vec2::splat(h * 2.0)),
        1.0,
        fill,
        stroke,
        egui::StrokeKind::Outside,
    );
    painter.add(egui::Shape::convex_polygon(
        vec![
            Pos2::new(c.x - h, c.y - h),
            Pos2::new(c.x, c.y - h),
            Pos2::new(c.x, c.y + h),
            Pos2::new(c.x - h, c.y + h),
        ],
        crate::theme::wire_mask().gamma_multiply(0.9),
        Stroke::NONE,
    ));
}
