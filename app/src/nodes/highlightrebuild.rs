//! Highlight Rebuild's app half: clip level, softness, search radius, and a
//! glyph of a sun whose clipped top is filled back in.

use crate::anim::Channel;
use crate::widgets::{self, HudSlider, RowCtx};
use egui::{Color32, Painter, Pos2, Stroke, Vec2};
use engine::graph::{HighlightRebuildParams, HLR_RADIUS_MAX};

pub const CHANNELS: &[Channel] = &[Channel::HlrClip, Channel::HlrSoftness, Channel::HlrRadius];

pub const TILES: usize = 1;

pub fn get(p: &HighlightRebuildParams, ch: Channel) -> Option<f32> {
    match ch {
        Channel::HlrClip => Some(p.clip),
        Channel::HlrSoftness => Some(p.softness),
        Channel::HlrRadius => Some(p.radius),
        _ => None,
    }
}

pub fn set(p: &mut HighlightRebuildParams, ch: Channel, v: f32) {
    match ch {
        Channel::HlrClip => p.clip = v,
        Channel::HlrSoftness => p.softness = v,
        Channel::HlrRadius => p.radius = v,
        _ => {}
    }
}

pub fn label(ch: Channel) -> Option<&'static str> {
    match ch {
        Channel::HlrClip => Some("highlight clip level"),
        Channel::HlrSoftness => Some("highlight softness"),
        Channel::HlrRadius => Some("highlight search radius"),
        _ => None,
    }
}

pub fn panel(ui: &mut egui::Ui, rcx: &mut RowCtx, p: &mut HighlightRebuildParams) -> Option<HudSlider> {
    let mut h = widgets::slider(ui, rcx, "clip", &mut p.clip, 0.05, 4.0, &[Channel::HlrClip]);
    h = h.or(widgets::slider(ui, rcx, "softness", &mut p.softness, 0.0, 0.5, &[Channel::HlrSoftness]));
    h = h.or(widgets::slider(ui, rcx, "radius", &mut p.radius, 0.0, HLR_RADIUS_MAX, &[Channel::HlrRadius]));
    ui.weak("rebuilds a clipped channel from the unclipped ones, at the colour around it");
    ui.weak("all three clipped: left as is (no data to rebuild from)");
    h
}

/// A sun whose top is clipped flat by a line, with its rays still drawn
/// above it: the clip, and what is rebuilt past it.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let rr = r * 0.62;
    painter.circle(c + Vec2::new(0.0, r * 0.18), rr, fill, stroke);
    let ink = crate::theme::node_ink();
    let cut = c.y - r * 0.12;
    let half = r * 0.75;
    painter.line_segment([Pos2::new(c.x - half, cut), Pos2::new(c.x + half, cut)], Stroke::new((r * 0.12).max(1.0), ink));
    for k in [-1.0f32, 0.0, 1.0] {
        let x = c.x + k * r * 0.42;
        painter.line_segment([Pos2::new(x, cut - r * 0.22), Pos2::new(x + k * r * 0.12, cut - r * 0.58)], stroke);
    }
}
