//! Mirror's app half: the panel, the glyph and the channel binding. The
//! model and the WGSL are `engine::nodes::mirror`; the viewer handle (drag
//! the line, turn it by its arrow) is in `lib.rs` beside the Gradient's.

use crate::anim::Channel;
use crate::widgets::{self, HudSlider, RowCtx};
use egui::{Color32, Painter, Pos2, Stroke, Vec2};
use engine::graph::{MirrorParams, MirrorSide};

pub const CHANNELS: &[Channel] = &[Channel::MirrorX, Channel::MirrorY, Channel::MirrorAngle];

pub const TILES: usize = 2;

pub fn get(p: &MirrorParams, ch: Channel) -> Option<f32> {
    match ch {
        Channel::MirrorX => Some(p.center[0]),
        Channel::MirrorY => Some(p.center[1]),
        Channel::MirrorAngle => Some(p.angle),
        _ => None,
    }
}

pub fn set(p: &mut MirrorParams, ch: Channel, v: f32) {
    match ch {
        Channel::MirrorX => p.center[0] = v,
        Channel::MirrorY => p.center[1] = v,
        Channel::MirrorAngle => p.angle = v,
        _ => {}
    }
}

pub fn label(ch: Channel) -> Option<&'static str> {
    match ch {
        Channel::MirrorX => Some("mirror x"),
        Channel::MirrorY => Some("mirror y"),
        Channel::MirrorAngle => Some("mirror angle"),
        _ => None,
    }
}

pub fn panel(ui: &mut egui::Ui, rcx: &mut RowCtx, p: &mut MirrorParams) -> Option<HudSlider> {
    ui.horizontal(|ui| {
        ui.label("keep");
        egui::ComboBox::from_id_salt(("mirror-side", rcx.node))
            .selected_text(p.side.name())
            .show_ui(ui, |ui| {
                for s in MirrorSide::ALL {
                    ui.selectable_value(&mut p.side, s, s.name());
                }
            });
    });
    let mut h = widgets::drags(
        ui,
        rcx,
        "line at",
        {
            let [x, y] = &mut p.center;
            &mut [x, y]
        },
        0.002,
        (0.0, 1.0),
        &[Channel::MirrorX, Channel::MirrorY],
    );
    h = h.or(widgets::slider(ui, rcx, "angle", &mut p.angle, -180.0, 180.0, &[Channel::MirrorAngle]));
    ui.weak("90° is a vertical line · drag the line in the viewer, turn it by its arrow");
    h
}

/// The graph glyph: a dashed fold line with a wedge and its reflection.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.circle(c, r * 0.9, fill, stroke);
    let ink = Stroke::new((r * 0.12).max(1.0), crate::theme::node_ink());
    let h = r * 0.62;
    let w = r * 0.45;
    for (a, b) in [(-1.0, -0.6), (-0.3, 0.1), (0.4, 0.8)] {
        painter.line_segment([c + Vec2::new(0.0, h * a), c + Vec2::new(0.0, h * b)], ink);
    }
    let wedge = |sx: f32, filled: bool| {
        let pts = vec![
            c + Vec2::new(sx * r * 0.14, -h * 0.55),
            c + Vec2::new(sx * (r * 0.14 + w), 0.0),
            c + Vec2::new(sx * r * 0.14, h * 0.55),
        ];
        if filled {
            painter.add(egui::Shape::convex_polygon(pts, ink.color, Stroke::NONE));
        } else {
            painter.add(egui::Shape::closed_line(pts, ink));
        }
    };
    wedge(-1.0, true);
    wedge(1.0, false);
}
