//! VectorWarp's app half: one amount, a glyph of a grid pushed along an
//! arrow. The lookup is `engine::nodes::vectorwarp`.

use crate::anim::Channel;
use crate::widgets::{self, HudSlider, RowCtx};
use egui::{Color32, Painter, Pos2, Stroke, Vec2};
use engine::graph::VectorWarpParams;

pub const CHANNELS: &[Channel] = &[Channel::VectorWarpAmount];

pub const TILES: usize = 1;

pub fn get(p: &VectorWarpParams, ch: Channel) -> Option<f32> {
    (ch == Channel::VectorWarpAmount).then_some(p.amount)
}

pub fn set(p: &mut VectorWarpParams, ch: Channel, v: f32) {
    if ch == Channel::VectorWarpAmount {
        p.amount = v;
    }
}

pub fn label(ch: Channel) -> Option<&'static str> {
    (ch == Channel::VectorWarpAmount).then_some("vector warp amount")
}

pub fn panel(ui: &mut egui::Ui, rcx: &mut RowCtx, p: &mut VectorWarpParams) -> Option<HudSlider> {
    let h = widgets::slider(ui, rcx, "amount", &mut p.amount, -2.0, 2.0, &[Channel::VectorWarpAmount]);
    ui.weak("frames of motion to follow: + forward vectors, − backward");
    ui.weak("1 over the next frame pulls it onto this one");
    h
}

/// A square with its right edge pushed along an arrow.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let h = r * 0.7;
    let pts = vec![
        c + Vec2::new(-h, -h),
        c + Vec2::new(h * 0.6, -h),
        c + Vec2::new(h * 1.15, 0.0),
        c + Vec2::new(h * 0.6, h),
        c + Vec2::new(-h, h),
    ];
    painter.add(egui::Shape::convex_polygon(pts, fill, stroke));
    let ink = Stroke::new((r * 0.12).max(1.0), crate::theme::node_ink());
    let (a, b) = (c + Vec2::new(-h * 0.6, 0.0), c + Vec2::new(h * 0.55, 0.0));
    painter.line_segment([a, b], ink);
    painter.line_segment([b, b + Vec2::new(-r * 0.2, -r * 0.18)], ink);
    painter.line_segment([b, b + Vec2::new(-r * 0.2, r * 0.18)], ink);
}
