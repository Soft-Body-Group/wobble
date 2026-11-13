//! Contrast's app half: two pages (tone, local), the channel binding and
//! the glyph. The maths, the WGSL and the local-contrast stage are
//! `engine::nodes::contrast`.
//!
//! Every number is `contrast.<name>` on `Channel::Custom` (the Starfield's
//! precedent), so the flat channel enum does not grow.

use crate::anim::{self, Channel};
use crate::widgets::{self, HudSlider, PanelTab, RowCtx};
use egui::{Color32, Painter, Pos2, Stroke, Vec2};
use engine::graph::{ContrastDomain, ContrastParams};
use engine::nodes::contrast::PREFIX;

/// Two tiles wide: the rows have names worth reading.
pub const TILES: usize = 2;

pub fn tabs() -> Vec<PanelTab> {
    vec![PanelTab::new("tone", "tone"), PanelTab::new("local", "local")]
}

/// The channel of a Contrast number by its name after `contrast.`.
pub fn chan(name: &str) -> Channel {
    Channel::Custom(anim::intern(&format!("{PREFIX}{name}")))
}

pub fn dyn_channels() -> Vec<Channel> {
    ContrastParams::names().iter().map(|n| chan(n)).collect()
}

pub fn get(p: &ContrastParams, ch: Channel) -> Option<f32> {
    match ch {
        Channel::Custom(i) => p.get(anim::interned(i).strip_prefix(PREFIX)?),
        _ => None,
    }
}

pub fn set(p: &mut ContrastParams, ch: Channel, v: f32) {
    if let Channel::Custom(i) = ch {
        if let Some(n) = anim::interned(i).strip_prefix(PREFIX) {
            p.set(n, v);
        }
    }
}

pub fn panel(ui: &mut egui::Ui, rcx: &mut RowCtx, p: &mut ContrastParams, tab: &str) -> Option<HudSlider> {
    let node = rcx.node;
    let mut h: Option<HudSlider> = None;
    let mut take = |got: Option<HudSlider>| {
        if got.is_some() {
            h = got;
        }
    };
    match tab {
        "local" => {
            // A stage: turning it on adds passes (and a red ring), so it is
            // a switch, not a slider at zero.
            ui.checkbox(&mut p.local, "local contrast").on_hover_text(
                "clarity: push each pixel away from its blurred surround, in stops. \
                 Its own passes — the node turns red while it is on",
            );
            ui.add_enabled_ui(p.local, |ui| {
                take(widgets::slider(ui, rcx, "clarity", &mut p.clarity, -1.0, 2.0, &[chan("clarity")]));
                take(widgets::slider(ui, rcx, "radius", &mut p.radius, 2.0, 400.0, &[chan("radius")]));
                take(widgets::slider(ui, rcx, "limit", &mut p.limit, 0.1, 4.0, &[chan("limit")]));
            });
            ui.add_space(4.0);
            ui.weak("limit: the most a pixel moves, in stops — keeps edges from haloing");
        }
        _ => {
            take(widgets::slider(ui, rcx, "contrast", &mut p.contrast, 0.0, 3.0, &[chan("amount")]));
            take(widgets::slider(ui, rcx, "pivot", &mut p.pivot, 0.01, 1.0, &[chan("pivot")]));
            take(widgets::slider(ui, rcx, "shadows", &mut p.shadows, 0.0, 3.0, &[chan("shadows")]));
            take(widgets::slider(ui, rcx, "highlights", &mut p.highlights, 0.0, 3.0, &[chan("highlights")]));
            take(widgets::slider(ui, rcx, "toe", &mut p.toe, 0.0, 1.0, &[chan("toe")]));
            take(widgets::slider(ui, rcx, "shoulder", &mut p.shoulder, 0.0, 1.0, &[chan("shoulder")]));
            take(widgets::slider(ui, rcx, "range", &mut p.range, 0.25, 8.0, &[chan("range")]));
            take(widgets::slider(ui, rcx, "keep colour", &mut p.preserve, 0.0, 1.0, &[chan("preserve")]));
            take(widgets::slider(ui, rcx, "mix", &mut p.mix, 0.0, 1.0, &[chan("mix")]));
            ui.horizontal(|ui| {
                ui.label("domain");
                egui::ComboBox::from_id_salt(("contrast-domain", node))
                    .width(120.0)
                    .selected_text(p.domain.label())
                    .show_ui(ui, |ui| {
                        for d in ContrastDomain::ALL {
                            ui.selectable_value(&mut p.domain, d, d.label());
                        }
                    })
                    .response
                    .on_hover_text("log: a slope in stops about the pivot; linear: a straight line through it");
            });
        }
    }
    h
}

/// The graph glyph: a disc split dark and light, an S-curve across it.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.circle(c, r * 0.9, fill, stroke);
    let ink = Stroke::new((r * 0.14).max(1.0), crate::theme::node_ink());
    let h = r * 0.55;
    let n = 16;
    let pts: Vec<Pos2> = (0..=n)
        .map(|k| {
            let t = k as f32 / n as f32 * 2.0 - 1.0;
            c + Vec2::new(h * t, -h * (t * 2.2).tanh() / 2.2f32.tanh())
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
        let mut p = ContrastParams::default();
        for (i, ch) in dyn_channels().into_iter().enumerate() {
            let v = 0.5 + i as f32 * 0.25;
            set(&mut p, ch, v);
            assert_eq!(get(&p, ch), Some(v), "{ch:?}");
        }
    }

    /// Both pages fit their tiles, both ways.
    #[test]
    fn the_panel_fits_its_tiles() {
        for tab in ["tone", "local"] {
            let mut p = ContrastParams { local: true, ..Default::default() };
            let mut anim = Default::default();
            let ekind = NodeKind::Contrast(p);
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
                    let _ = panel(ui, &mut rcx, &mut p, tab);
                });
                over.set(out.content_size.y - out.inner_rect.height());
                wide.set(out.content_size.x - out.inner_rect.width());
            });
            shot.save(&format!("contrast-panel-{tab}"));
            assert!(over.get() <= 0.5, "{tab} overflows its tile DOWN by {}pt", over.get());
            assert!(wide.get() <= 0.5, "{tab} overflows its tile SIDEWAYS by {}pt", wide.get());
        }
    }
}
