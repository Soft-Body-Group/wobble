//! MotionBlur's app half: the shutter's three knobs, a streaked-disc glyph.

use crate::anim::Channel;
use crate::widgets::{self, HudSlider, RowCtx};
use egui::{Color32, Painter, Pos2, Stroke};
use engine::graph::MotionBlurParams;
use engine::nodes::motionblur::MAX_SAMPLES;

/// Samples are a pass COUNT, so they are not a channel: a keyed sample
/// count would change the cost of the comp across the shot for no reason
/// anybody wants. Angle and offset are times, and animate.
pub const CHANNELS: &[Channel] = &[Channel::MotionBlurAngle, Channel::MotionBlurOffset];

pub const TILES: usize = 1;

pub fn get(p: &MotionBlurParams, ch: Channel) -> Option<f32> {
    match ch {
        Channel::MotionBlurAngle => Some(p.angle),
        Channel::MotionBlurOffset => Some(p.offset),
        _ => None,
    }
}

pub fn set(p: &mut MotionBlurParams, ch: Channel, v: f32) {
    match ch {
        Channel::MotionBlurAngle => p.angle = v,
        Channel::MotionBlurOffset => p.offset = v,
        _ => {}
    }
}

pub fn label(ch: Channel) -> Option<&'static str> {
    match ch {
        Channel::MotionBlurAngle => Some("shutter angle"),
        Channel::MotionBlurOffset => Some("shutter offset"),
        _ => None,
    }
}

pub fn panel(ui: &mut egui::Ui, rcx: &mut RowCtx, p: &mut MotionBlurParams) -> Option<HudSlider> {
    let mut n = p.samples as f32;
    widgets::slider(ui, rcx, "samples", &mut n, 2.0, MAX_SAMPLES as f32, &[]);
    p.samples = n.round().clamp(1.0, MAX_SAMPLES as f32) as u32;
    let h = widgets::slider(ui, rcx, "angle", &mut p.angle, 0.0, 2.0, &[Channel::MotionBlurAngle]);
    let h2 =
        widgets::slider(ui, rcx, "offset", &mut p.offset, -0.5, 0.5, &[Channel::MotionBlurOffset]);
    ui.weak(format!(
        "everything upstream, rendered {}x across the shutter (slow: \
         each sample re-runs the branch). 0.5 = 180°. A moving plate does \
         not smear — only animation does.",
        p.samples
    ));
    h.or(h2)
}

/// A disc trailing three streaks: the moving thing and its smear.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let head = Pos2::new(c.x + r * 0.35, c.y);
    for (dy, len) in [(-0.45, 0.9), (0.0, 1.2), (0.45, 0.9)] {
        let y = c.y + dy * r;
        painter.line_segment(
            [Pos2::new(head.x - r * len, y), Pos2::new(head.x - r * 0.3, y)],
            Stroke::new(stroke.width.max(1.0), fill.gamma_multiply(0.7)),
        );
    }
    painter.circle(head, r * 0.55, fill, stroke);
}
