//! Blur's app half: one slider, the diamond glyph.

use crate::anim::Channel;
use crate::widgets::{self, HudSlider, RowCtx};
use egui::{Color32, Painter, Pos2, Stroke};
use engine::graph::BlurParams;

pub const CHANNELS: &[Channel] = &[Channel::BlurSize];

pub const TILES: usize = 1;

pub fn get(p: &BlurParams, ch: Channel) -> Option<f32> {
    match ch {
        Channel::BlurSize => Some(p.size),
        _ => None,
    }
}

pub fn set(p: &mut BlurParams, ch: Channel, v: f32) {
    if ch == Channel::BlurSize {
        p.size = v;
    }
}

pub fn label(ch: Channel) -> Option<&'static str> {
    match ch {
        Channel::BlurSize => Some("blur size"),
        _ => None,
    }
}

pub fn panel(ui: &mut egui::Ui, rcx: &mut RowCtx, p: &mut BlurParams) -> Option<HudSlider> {
    let h = widgets::slider(ui, rcx, "size", &mut p.size, 0.0, 120.0, &[Channel::BlurSize]);
    ui.weak("gaussian sigma, separable H+V");
    h
}

/// A diamond: the four-way spread.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let pts = vec![
        Pos2::new(c.x, c.y - r),
        Pos2::new(c.x + r, c.y),
        Pos2::new(c.x, c.y + r),
        Pos2::new(c.x - r, c.y),
    ];
    painter.add(egui::Shape::convex_polygon(pts, fill, stroke));
}
