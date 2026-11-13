//! EdgeExtend's app half: spread or expand, three px sliders, the alpha
//! switches, and a glyph of a matte bleeding its colour outward.

use crate::anim::Channel;
use crate::widgets::{self, HudSlider, RowCtx};
use egui::{Color32, Painter, Pos2, Stroke};
use engine::graph::{EdgeExtendParams, ExtendMode, EXTEND_MAX};

pub const CHANNELS: &[Channel] =
    &[Channel::EdgeExtendDistance, Channel::EdgeExtendInset, Channel::EdgeExtendSoftness];

pub const TILES: usize = 2;

pub fn get(p: &EdgeExtendParams, ch: Channel) -> Option<f32> {
    Some(match ch {
        Channel::EdgeExtendDistance => p.distance,
        Channel::EdgeExtendInset => p.inset,
        Channel::EdgeExtendSoftness => p.softness,
        _ => return None,
    })
}

pub fn set(p: &mut EdgeExtendParams, ch: Channel, v: f32) {
    match ch {
        Channel::EdgeExtendDistance => p.distance = v,
        Channel::EdgeExtendInset => p.inset = v,
        Channel::EdgeExtendSoftness => p.softness = v,
        _ => {}
    }
}

pub fn label(ch: Channel) -> Option<&'static str> {
    Some(match ch {
        Channel::EdgeExtendDistance => "extend distance",
        Channel::EdgeExtendInset => "extend inset",
        Channel::EdgeExtendSoftness => "extend softness",
        _ => return None,
    })
}

pub fn panel(
    ui: &mut egui::Ui,
    rcx: &mut RowCtx,
    p: &mut EdgeExtendParams,
    matte_wired: bool,
) -> Option<HudSlider> {
    use Channel::*;
    ui.horizontal(|ui| {
        ui.label("mode");
        ui.selectable_value(&mut p.mode, ExtendMode::Spread, "spread")
            .on_hover_text("fill the whole frame outside the matte with its edge colours");
        ui.selectable_value(&mut p.mode, ExtendMode::Expand, "expand")
            .on_hover_text("extend the edge colours this many pixels out");
    });
    let mut h = None;
    if p.mode == ExtendMode::Expand {
        h = widgets::slider(ui, rcx, "distance", &mut p.distance, 0.0, 64.0, &[EdgeExtendDistance]);
    }
    for got in [
        widgets::slider(ui, rcx, "inset", &mut p.inset, 0.0, 16.0, &[EdgeExtendInset]),
        widgets::slider(ui, rcx, "softness", &mut p.softness, 0.0, 16.0, &[EdgeExtendSoftness]),
    ]
    .into_iter()
    .flatten()
    {
        h = Some(got);
    }
    ui.horizontal(|ui| {
        ui.add_enabled_ui(!matte_wired, |ui| {
            ui.checkbox(&mut p.premultiplied, "premult in").on_hover_text(
                "the input's colour is premultiplied by its alpha (ignored with a matte wired: \
                 the colour is read straight)",
            );
        });
        ui.checkbox(&mut p.keep_alpha, "keep alpha")
            .on_hover_text("alpha out is the input's; off, it covers the extended area");
        ui.checkbox(&mut p.premult_out, "premult out")
            .on_hover_text("multiply the result by the alpha that comes out");
    });
    ui.weak(if p.inset > 0.0 {
        "colour from inside the edge replaces the edge band"
    } else {
        "inset pulls colour in from past a dark or light edge"
    });
    if !matte_wired {
        ui.weak("the edge is the input's alpha; wire a matte to choose it");
    }
    if p.distance + p.inset + 2.0 * p.softness > EXTEND_MAX {
        ui.weak(format!("reach is capped at {EXTEND_MAX} px"));
    }
    h
}

/// A matte with its colour bleeding out: a filled disc, then two fading
/// rings.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.circle(c, r * 0.45, fill, stroke);
    painter.circle_stroke(c, r * 0.7, Stroke::new(stroke.width, fill.gamma_multiply(0.7)));
    painter.circle_stroke(c, r * 0.95, Stroke::new(stroke.width, fill.gamma_multiply(0.35)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::graph::NodeKind;

    /// Expand mode with every row and hint showing, at the declared tiles:
    /// no overflow either way (an overflowed row cannot be clicked).
    #[test]
    fn the_panel_fits_the_tiles_it_asks_for() {
        let mut p = EdgeExtendParams {
            mode: ExtendMode::Expand,
            distance: 240.0,
            inset: 20.0,
            ..Default::default()
        };
        let mut anim = Default::default();
        let ekind = NodeKind::EdgeExtend(p);
        let over = std::cell::Cell::new(0.0f32);
        let wide = std::cell::Cell::new(0.0f32);
        let span = |n: usize| {
            crate::widgets::TILE * n as f32 + crate::widgets::TILE_GAP * (n as f32 - 1.0)
        };
        let (w, hgt) = (span(TILES) - 14.0, span(1) - 14.0 - 22.0);
        let shot = crate::snapshot::shoot(w, hgt, |ui| {
            let out = egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                let mut rcx = RowCtx {
                    node: 1,
                    anim: &mut anim,
                    ekind: &ekind,
                    t: 0,
                    hud_live: false,
                    driven: Vec::new(),
                };
                let _ = panel(ui, &mut rcx, &mut p, false);
            });
            over.set(out.content_size.y - out.inner_rect.height());
            wide.set(out.content_size.x - out.inner_rect.width());
        });
        shot.save("edgeextend-panel");
        assert!(over.get() <= 0.5, "overflows its tiles by {}pt", over.get());
        assert!(wide.get() <= 0.5, "overflows its tiles SIDEWAYS by {}pt", wide.get());
    }
}
