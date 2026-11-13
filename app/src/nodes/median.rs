//! Median's app half: one radius slider, a three-bar glyph (the middle one
//! is the answer).

use crate::anim::Channel;
use crate::widgets::{self, HudSlider, RowCtx};
use egui::{Color32, Painter, Pos2, Rect, Stroke, Vec2};
use engine::graph::{MedianParams, MEDIAN_MAX};

pub const CHANNELS: &[Channel] = &[Channel::MedianRadius];

pub const TILES: usize = 1;

pub fn get(p: &MedianParams, ch: Channel) -> Option<f32> {
    match ch {
        Channel::MedianRadius => Some(p.radius),
        _ => None,
    }
}

pub fn set(p: &mut MedianParams, ch: Channel, v: f32) {
    if ch == Channel::MedianRadius {
        p.radius = v;
    }
}

pub fn label(ch: Channel) -> Option<&'static str> {
    match ch {
        Channel::MedianRadius => Some("median radius"),
        _ => None,
    }
}

pub fn panel(ui: &mut egui::Ui, rcx: &mut RowCtx, p: &mut MedianParams) -> Option<HudSlider> {
    let h = widgets::slider(ui, rcx, "radius", &mut p.radius, 0.0, MEDIAN_MAX, &[Channel::MedianRadius]);
    let r = p.radius.round().max(0.0);
    ui.weak(if r <= engine::graph::MEDIAN_EXACT {
        "exact median of the window, per channel"
    } else {
        "past 3 px: median of row medians (fast, near-exact)"
    });
    ui.weak("whole pixels; the window is 2r + 1 wide");
    h
}

/// A box with three bars, the middle one tallest: sort, take the middle.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let outer = Rect::from_center_size(c, Vec2::splat(r * 1.6));
    painter.rect(outer, 0.0, fill, stroke, egui::StrokeKind::Middle);
    let s = r * 0.42;
    for (k, h) in [(-1.0f32, 0.45f32), (0.0, 0.8), (1.0, 0.45)] {
        let x = c.x + k * s;
        painter.line_segment([Pos2::new(x, c.y + r * 0.5), Pos2::new(x, c.y + r * 0.5 - h * r)], stroke);
    }
}
