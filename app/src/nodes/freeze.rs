//! Freeze's app half: the panel, the glyph and the channel. The params and
//! the time map are `engine::nodes::freeze`.

use crate::anim::Channel;
use crate::widgets::{self, HudSlider, RowCtx};
use egui::{Color32, Painter, Pos2, Stroke, Ui, Vec2};
use engine::graph::FreezeParams;

pub const CHANNELS: &[Channel] = &[Channel::FreezeFrame];
pub const TILES: usize = 1;

pub fn get(p: &FreezeParams, ch: Channel) -> Option<f32> {
    (ch == Channel::FreezeFrame).then_some(p.frame)
}

pub fn set(p: &mut FreezeParams, ch: Channel, v: f32) {
    if ch == Channel::FreezeFrame {
        p.frame = v;
    }
}

pub fn label(ch: Channel) -> Option<&'static str> {
    (ch == Channel::FreezeFrame).then_some("freeze frame")
}

pub fn panel(ui: &mut Ui, rcx: &mut RowCtx, p: &mut FreezeParams) -> Option<HudSlider> {
    let h = widgets::drags(ui, rcx, "frame", &mut [&mut p.frame], 0.25, (0.0, 100.0), &[Channel::FreezeFrame]);
    let keyed = rcx.anim.get(&Channel::FreezeFrame).is_some_and(|c| c.keys.len() > 1);
    let here = rcx.t as f32;
    ui.horizontal(|ui| {
        if ui
            .add_enabled(!keyed && p.frame.round() != here, egui::Button::new("hold this frame"))
            .on_hover_text("freeze on the frame under the playhead")
            .clicked()
        {
            p.frame = here;
        }
    });
    ui.weak(if keyed {
        "keyed: each key holds a frame (stepped keys jump, smooth ones slide)"
    } else {
        "every frame shows this comp frame; it costs nothing to render"
    });
    h
}

/// A clock face with a pause sign: time, held.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.circle(c, r * 0.9, fill, stroke);
    let ink = crate::theme::node_ink();
    for dx in [-0.22, 0.22] {
        let bar = egui::Rect::from_center_size(c + Vec2::new(r * dx, 0.0), Vec2::new(r * 0.2, r * 0.8));
        painter.rect_filled(bar, r * 0.04, ink);
    }
}
