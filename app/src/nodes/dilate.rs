//! Dilate's app half: one signed slider, a matte-and-its-reach glyph.

use crate::anim::Channel;
use crate::widgets::{self, HudSlider, RowCtx};
use egui::{Color32, Painter, Pos2, Stroke};
use engine::graph::{DilateParams, DILATE_MAX};

pub const CHANNELS: &[Channel] = &[Channel::DilateAmount];

pub const TILES: usize = 1;

pub fn get(p: &DilateParams, ch: Channel) -> Option<f32> {
    match ch {
        Channel::DilateAmount => Some(p.amount),
        _ => None,
    }
}

pub fn set(p: &mut DilateParams, ch: Channel, v: f32) {
    if ch == Channel::DilateAmount {
        p.amount = v;
    }
}

pub fn label(ch: Channel) -> Option<&'static str> {
    match ch {
        Channel::DilateAmount => Some("dilate"),
        _ => None,
    }
}

pub fn panel(ui: &mut egui::Ui, rcx: &mut RowCtx, p: &mut DilateParams) -> Option<HudSlider> {
    // One signed number: positive grows the matte, negative shrinks it,
    // fractions are real.
    let h = widgets::slider(
        ui,
        rcx,
        "amount",
        &mut p.amount,
        -DILATE_MAX,
        DILATE_MAX,
        &[Channel::DilateAmount],
    );
    ui.weak(if p.amount < 0.0 {
        "erodes the matte, in pixels (sub-pixel amounts count)"
    } else {
        "dilates the matte, in pixels (sub-pixel amounts count)"
    });
    h
}

/// A matte and where it grows to: the inner disc filled so there is a
/// silhouette, the outer ring its reach.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.circle(c, r * 0.5, fill, stroke);
    painter.circle_stroke(c, r * 0.95, stroke);
}
