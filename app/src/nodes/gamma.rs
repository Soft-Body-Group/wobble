//! Gamma's app half: the panel, the glyph and the channel bindings. The
//! model, the WGSL and the params are `engine::nodes::gamma`.

use crate::anim::Channel;
use crate::rgbslider::RgbSpec;
use crate::widgets::{self, HudSlider, RowCtx};
use egui::{Color32, Painter, Pos2, Stroke, Vec2};
use engine::color::Colorspace;
use engine::graph::GammaParams;
use std::collections::BTreeSet;

pub const CHANNELS: &[Channel] = &[
    Channel::GammaNodeMaster,
    Channel::GammaNodeR,
    Channel::GammaNodeG,
    Channel::GammaNodeB,
    Channel::GammaNodeA,
    Channel::GammaNodeMix,
];

const RGB: [Channel; 3] = [Channel::GammaNodeR, Channel::GammaNodeG, Channel::GammaNodeB];

/// Two tiles wide: an rgb row wants about 420pt (see `colorcompress::TILES`).
pub const TILES: usize = 2;

pub fn get(p: &GammaParams, ch: Channel) -> Option<f32> {
    use Channel::*;
    Some(match ch {
        GammaNodeMaster => p.master,
        GammaNodeR => p.rgb[0],
        GammaNodeG => p.rgb[1],
        GammaNodeB => p.rgb[2],
        GammaNodeA => p.alpha,
        GammaNodeMix => p.mix,
        _ => return None,
    })
}

pub fn set(p: &mut GammaParams, ch: Channel, v: f32) {
    use Channel::*;
    match ch {
        GammaNodeMaster => p.master = v,
        GammaNodeR => p.rgb[0] = v,
        GammaNodeG => p.rgb[1] = v,
        GammaNodeB => p.rgb[2] = v,
        GammaNodeA => p.alpha = v,
        GammaNodeMix => p.mix = v,
        _ => {}
    }
}

/// The curve editor's name for one of this node's channels.
pub fn label(ch: Channel) -> Option<&'static str> {
    use Channel::*;
    Some(match ch {
        GammaNodeMaster => "gamma",
        GammaNodeR => "gamma r",
        GammaNodeG => "gamma g",
        GammaNodeB => "gamma b",
        GammaNodeA => "gamma alpha",
        GammaNodeMix => "gamma mix",
        _ => return None,
    })
}

/// What an rgb row hangs past the width it is given (its value chip); the
/// row is drawn into a child this much narrower so it stays on its tile.
const ROW_OVERHANG: f32 = crate::rgbslider::SWATCH_W + 4.0;

/// The panel: the master, the per-channel trim on top of it, alpha's own,
/// and the mix.
pub fn panel(
    ui: &mut egui::Ui,
    rcx: &mut RowCtx,
    p: &mut GammaParams,
    working: Colorspace,
    ungang: &mut BTreeSet<String>,
) -> Option<HudSlider> {
    let luma = working.primaries.luma_weights();
    let disp =
        engine::color::primaries_matrix(working.primaries, engine::color::Primaries::Rec709);
    let mut h: Option<HudSlider> = None;
    let mut take = |s: Option<HudSlider>| {
        if s.is_some() {
            h = s;
        }
    };
    take(widgets::slider(ui, rcx, "gamma", &mut p.master, 0.2, 5.0, &[Channel::GammaNodeMaster]));
    // The rgb row in its own narrower child, so its chip stays on the tile.
    let w = (ui.available_width() - ROW_OVERHANG).max(240.0);
    let mut c = ui.new_child(
        egui::UiBuilder::new()
            .max_rect(egui::Rect::from_min_size(ui.cursor().min, egui::vec2(w, 0.0)))
            .layout(egui::Layout::top_down(egui::Align::Min)),
    );
    let mut gang = !ungang.contains("rgb");
    let spec = RgbSpec::scale(0.2, 5.0, luma).with_display(disp);
    take(widgets::rgb_row(&mut c, rcx, "rgb", &mut p.rgb, &mut gang, spec, &RGB));
    ui.advance_cursor_after_rect(c.min_rect());
    if gang {
        ungang.remove("rgb");
    } else {
        ungang.insert("rgb".to_string());
    }
    take(widgets::slider(ui, rcx, "alpha", &mut p.alpha, 0.2, 5.0, &[Channel::GammaNodeA]));
    take(widgets::slider(ui, rcx, "mix", &mut p.mix, 0.0, 1.0, &[Channel::GammaNodeMix]));
    ui.add_space(4.0);
    ui.weak("x^(1/γ): above 1 lifts the mids; negatives pass through");
    h
}

/// The graph glyph: a bowed power curve inside a square.
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
    // y is down: x^(1/2.2) rising to the right, bowed up.
    let n = 16;
    let pts: Vec<Pos2> = (0..=n)
        .map(|k| {
            let t = k as f32 / n as f32;
            c + Vec2::new(-h + 2.0 * h * t, h - 2.0 * h * t.powf(1.0 / 2.2))
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
        let mut p = GammaParams::default();
        for (i, ch) in CHANNELS.iter().enumerate() {
            let v = 0.5 + i as f32 * 0.25;
            set(&mut p, *ch, v);
            assert_eq!(get(&p, *ch), Some(v), "{ch:?}");
            assert!(label(*ch).is_some(), "{ch:?}");
        }
    }

    /// The panel fits its tiles, both ways.
    #[test]
    fn the_panel_fits_its_tiles() {
        let mut p = GammaParams::default();
        let mut ungang = BTreeSet::new();
        let mut anim = Default::default();
        let ekind = NodeKind::Gamma(p);
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
                let _ = panel(ui, &mut rcx, &mut p, Colorspace::ACESCG, &mut ungang);
            });
            over.set(out.content_size.y - out.inner_rect.height());
            wide.set(out.content_size.x - out.inner_rect.width());
        });
        shot.save("gamma-panel");
        assert!(over.get() <= 0.5, "overflows its tile DOWN by {}pt", over.get());
        assert!(wide.get() <= 0.5, "overflows its tile SIDEWAYS by {}pt", wide.get());
    }
}
