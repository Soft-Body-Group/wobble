//! Tone Map's app half: the panel, the glyph and the channel bindings. The
//! model, the WGSL and the params are `engine::nodes::tonemap`.

use crate::anim::Channel;
use crate::widgets::{self, HudSlider, RowCtx};
use egui::{Color32, Painter, Pos2, Stroke, Vec2};
use engine::graph::{ToneMapParams, ToneMode, ToneOp};

pub const CHANNELS: &[Channel] = &[
    Channel::ToneExposure,
    Channel::ToneThreshold,
    Channel::ToneWhite,
    Channel::ToneSoftness,
    Channel::ToneToe,
    Channel::ToneToePivot,
    Channel::ToneMix,
];

pub const TILES: usize = 1;

pub fn get(p: &ToneMapParams, ch: Channel) -> Option<f32> {
    use Channel::*;
    Some(match ch {
        ToneExposure => p.exposure,
        ToneThreshold => p.threshold,
        ToneWhite => p.white,
        ToneSoftness => p.softness,
        ToneToe => p.toe,
        ToneToePivot => p.toe_pivot,
        ToneMix => p.mix,
        _ => return None,
    })
}

pub fn set(p: &mut ToneMapParams, ch: Channel, v: f32) {
    use Channel::*;
    match ch {
        ToneExposure => p.exposure = v,
        ToneThreshold => p.threshold = v,
        ToneWhite => p.white = v,
        ToneSoftness => p.softness = v,
        ToneToe => p.toe = v,
        ToneToePivot => p.toe_pivot = v,
        ToneMix => p.mix = v,
        _ => {}
    }
}

/// The curve editor's name for one of this node's channels.
pub fn label(ch: Channel) -> Option<&'static str> {
    use Channel::*;
    Some(match ch {
        ToneExposure => "tone exposure",
        ToneThreshold => "tone threshold",
        ToneWhite => "tone white",
        ToneSoftness => "tone softness",
        ToneToe => "tone toe",
        ToneToePivot => "tone toe pivot",
        ToneMix => "tone mix",
        _ => return None,
    })
}

/// The panel: operator and mode (shader shape, so combos), then the
/// numbers the operator reads.
pub fn panel(ui: &mut egui::Ui, rcx: &mut RowCtx, p: &mut ToneMapParams) -> Option<HudSlider> {
    use Channel::*;
    let mut h = None;
    ui.horizontal(|ui| {
        ui.label("operator");
        let before = p.op;
        egui::ComboBox::from_id_salt(("tone-op", rcx.node))
            .width(130.0)
            .selected_text(p.op.label())
            .show_ui(ui, |ui| {
                for o in ToneOp::ALL {
                    ui.selectable_value(&mut p.op, o, o.label());
                }
            });
        // `white` means a ceiling to the shoulders and an input level to
        // the rest: move it to where the new operator wants it.
        if p.op != before {
            p.white = p.op.default_white();
        }
    });
    ui.horizontal(|ui| {
        ui.label("apply");
        egui::ComboBox::from_id_salt(("tone-mode", rcx.node))
            .width(130.0)
            .selected_text(p.mode.label())
            .show_ui(ui, |ui| {
                for m in ToneMode::ALL {
                    ui.selectable_value(&mut p.mode, m, m.label());
                }
            });
    });
    h = h.or(widgets::slider(ui, rcx, "exposure", &mut p.exposure, -6.0, 6.0, &[ToneExposure]));
    let knee = p.op.has_threshold();
    if knee {
        h = h.or(widgets::slider(ui, rcx, "threshold", &mut p.threshold, 0.0, 4.0, &[ToneThreshold]));
        h = h.or(widgets::slider(ui, rcx, "white", &mut p.white, 0.1, 16.0, &[ToneWhite]));
        if p.op == ToneOp::Knee {
            h = h.or(widgets::slider(ui, rcx, "softness", &mut p.softness, 0.02, 1.0, &[ToneSoftness]));
        }
    } else {
        h = h.or(widgets::slider(ui, rcx, "white in", &mut p.white, 0.5, 64.0, &[ToneWhite]));
    }
    h = h.or(widgets::slider(ui, rcx, "toe", &mut p.toe, 0.0, 1.0, &[ToneToe]));
    h = h.or(widgets::slider(ui, rcx, "toe pivot", &mut p.toe_pivot, 0.01, 1.0, &[ToneToePivot]));
    h = h.or(widgets::slider(ui, rcx, "mix", &mut p.mix, 0.0, 1.0, &[ToneMix]));
    ui.add_space(4.0);
    ui.weak(if knee {
        "untouched below threshold, rolls off towards white"
    } else {
        "whole-range curve: input white lands on 1"
    });
    h
}

/// The graph glyph: a rounded square with a curve that rises straight and
/// bends over under a ceiling — a knee.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.rect(
        egui::Rect::from_center_size(c, egui::vec2(r * 1.7, r * 1.7)),
        r * 0.3,
        fill,
        stroke,
        egui::StrokeKind::Middle,
    );
    let ink = Stroke::new((r * 0.14).max(1.0), crate::theme::node_ink());
    let h = r * 0.6;
    // y is down. The ceiling, then the identity into a shoulder under it.
    let top = c.y - h * 0.8;
    painter.line_segment([Pos2::new(c.x - h, top), Pos2::new(c.x + h, top)], Stroke::new(ink.width * 0.5, ink.color));
    let n = 16;
    let pts: Vec<Pos2> = (0..=n)
        .map(|k| {
            let x = k as f32 / n as f32 * 2.0;
            let y = if x < 0.6 { x } else { 0.6 + 0.4 * (1.0 - (-(x - 0.6) / 0.4).exp()) };
            c + Vec2::new(-h + h * x, h - 1.8 * h * y)
        })
        .collect();
    painter.add(egui::Shape::line(pts, ink));
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::graph::NodeKind;

    #[test]
    fn every_channel_round_trips() {
        let mut p = ToneMapParams::default();
        for (i, ch) in CHANNELS.iter().enumerate() {
            let v = 0.5 + i as f32 * 0.25;
            set(&mut p, *ch, v);
            assert_eq!(get(&p, *ch), Some(v), "{ch:?}");
            assert!(label(*ch).is_some(), "{ch:?}");
        }
    }

    /// The panel fits its tile, for the operator with the most rows.
    #[test]
    fn the_panel_fits_its_tiles() {
        let mut p = ToneMapParams::default();
        let mut anim = Default::default();
        let ekind = NodeKind::ToneMap(p);
        let (over, wide) = (std::cell::Cell::new(0.0f32), std::cell::Cell::new(0.0f32));
        let t = crate::widgets::TILE;
        let w = t * TILES as f32 + crate::widgets::TILE_GAP * (TILES as f32 - 1.0);
        let shot = crate::snapshot::shoot(w - 14.0, t - 14.0 - 22.0, |ui| {
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
        shot.save("tonemap-panel");
        assert!(over.get() <= 0.5, "overflows its tile DOWN by {}pt", over.get());
        assert!(wide.get() <= 0.5, "overflows its tile SIDEWAYS by {}pt", wide.get());
    }
}
