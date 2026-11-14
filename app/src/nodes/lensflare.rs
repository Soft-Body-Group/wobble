//! Lens Flare's app half: three pages (light, glow, ghosts), the channel
//! binding and the glyph. The params and the WGSL are
//! `engine::nodes::lensflare`; the light's viewer handle (a sun, with the
//! ghost axis and each ghost's place drawn) lives with the other gizmos in
//! `lib.rs`.
//!
//! Every number is `flare.<name>` on `Channel::Custom` (the Starfield's
//! precedent), so the flat channel enum does not grow.

use crate::anim::{self, Channel};
use crate::rgbslider::RgbSpec;
use crate::widgets::{self, HudSlider, PanelTab, RowCtx};
use egui::{Color32, Painter, Pos2, Stroke};
use engine::color::{primaries_matrix, Colorspace, Primaries};
use engine::graph::{GlowMode, LensFlareParams};
use engine::nodes::lensflare::{MAX_GHOSTS, PREFIX};
use std::collections::BTreeSet;

/// Two tiles: an rgb row needs track to aim along.
pub const TILES: usize = 2;
/// Two tall: the ghost page has the most rows.
pub const TILE_ROWS: usize = 2;

pub fn tabs() -> Vec<PanelTab> {
    vec![PanelTab::new("light", "light"), PanelTab::new("glow", "glow"), PanelTab::new("ghosts", "ghosts")]
}

/// The channel of a Lens Flare number by its name after `flare.`.
pub fn chan(name: &str) -> Channel {
    Channel::Custom(anim::intern(&format!("{PREFIX}{name}")))
}

pub fn dyn_channels() -> Vec<Channel> {
    LensFlareParams::names().iter().map(|n| chan(n)).collect()
}

pub fn get(p: &LensFlareParams, ch: Channel) -> Option<f32> {
    match ch {
        Channel::Custom(i) => p.get(anim::interned(i).strip_prefix(PREFIX)?),
        _ => None,
    }
}

pub fn set(p: &mut LensFlareParams, ch: Channel, v: f32) {
    if let Channel::Custom(i) = ch {
        if let Some(n) = anim::interned(i).strip_prefix(PREFIX) {
            p.set(n, v);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn rgb(
    ui: &mut egui::Ui,
    rcx: &mut RowCtx,
    label: &str,
    v: &mut [f32; 3],
    key: &str,
    ungang: &mut BTreeSet<String>,
    working: Colorspace,
    chans: [&str; 3],
) -> Option<HudSlider> {
    let mut gang = !ungang.contains(key);
    let h = widgets::rgb_row(
        ui,
        rcx,
        label,
        v,
        &mut gang,
        RgbSpec::scale(0.0, 4.0, working.primaries.luma_weights())
            .with_display(primaries_matrix(working.primaries, Primaries::Rec709)),
        &chans.map(chan),
    );
    if gang {
        ungang.remove(key);
    } else {
        ungang.insert(key.to_string());
    }
    h
}

/// `wired` is whether a picture feeds port 0 (occlusion reads it).
pub fn panel(
    ui: &mut egui::Ui,
    rcx: &mut RowCtx,
    p: &mut LensFlareParams,
    tab: &str,
    working: Colorspace,
    ungang: &mut BTreeSet<String>,
    wired: bool,
) -> Option<HudSlider> {
    let node = rcx.node;
    let mut h: Option<HudSlider> = None;
    let mut take = |got: Option<HudSlider>| {
        if got.is_some() {
            h = got;
        }
    };
    match tab {
        "glow" => {
            take(widgets::slider(ui, rcx, "glow", &mut p.glow, 0.0, 4.0, &[chan("glow")]));
            take(widgets::slider(ui, rcx, "glow size", &mut p.glow_size, 0.0, 0.5, &[chan("glow.size")]));
            take(widgets::slider(ui, rcx, "burst", &mut p.burst, 0.0, 4.0, &[chan("burst")]));
            take(widgets::slider(ui, rcx, "rays", &mut p.burst_rays, 2.0, 32.0, &[chan("burst.rays")]));
            take(widgets::slider(ui, rcx, "burst len", &mut p.burst_len, 0.0, 1.5, &[chan("burst.len")]));
            take(widgets::slider(ui, rcx, "burst rot", &mut p.burst_rot, -180.0, 180.0, &[chan("burst.rot")]));
            take(widgets::slider(ui, rcx, "sharp", &mut p.burst_sharp, 0.0, 1.0, &[chan("burst.sharp")]));
            take(widgets::slider(ui, rcx, "streak", &mut p.streak, 0.0, 4.0, &[chan("streak")]));
            take(widgets::slider(ui, rcx, "streak len", &mut p.streak_len, 0.0, 2.0, &[chan("streak.len")]));
            take(widgets::slider(ui, rcx, "thickness", &mut p.streak_thick, 0.0, 0.05, &[chan("streak.thick")]));
            take(widgets::slider(ui, rcx, "angle", &mut p.streak_angle, -90.0, 90.0, &[chan("streak.angle")]));
            take(rgb(ui, rcx, "streak tint", &mut p.streak_tint, "streak", ungang, working, ["streak.r", "streak.g", "streak.b"]));
        }
        "ghosts" => {
            take(widgets::slider(ui, rcx, "ghosts", &mut p.ghosts, 0.0, 2.0, &[chan("ghosts")]));
            take(widgets::slider(ui, rcx, "count", &mut p.ghost_count, 0.0, MAX_GHOSTS as f32, &[chan("ghost.count")]));
            take(widgets::slider(ui, rcx, "spread", &mut p.ghost_spread, 0.0, 2.0, &[chan("ghost.spread")]));
            take(widgets::slider(ui, rcx, "size", &mut p.ghost_size, 0.0, 0.3, &[chan("ghost.size")]));
            take(widgets::slider(ui, rcx, "seed", &mut p.ghost_seed, 0.0, 100.0, &[chan("ghost.seed")]));
            take(widgets::slider(ui, rcx, "colour", &mut p.ghost_color, 0.0, 1.0, &[chan("ghost.color")]));
            take(widgets::slider(ui, rcx, "blades", &mut p.ghost_blades, 0.0, 12.0, &[chan("ghost.blades")]));
            take(widgets::slider(ui, rcx, "chroma", &mut p.ghost_chroma, 0.0, 0.3, &[chan("ghost.chroma")]));
            take(widgets::slider(ui, rcx, "soft", &mut p.ghost_soft, 0.0, 1.0, &[chan("ghost.soft")]));
            take(widgets::slider(ui, rcx, "halo", &mut p.halo, 0.0, 2.0, &[chan("halo")]));
            take(widgets::slider(ui, rcx, "halo radius", &mut p.halo_radius, 0.0, 1.0, &[chan("halo.radius")]));
            take(widgets::slider(ui, rcx, "halo width", &mut p.halo_width, 0.0, 0.2, &[chan("halo.width")]));
            take(widgets::slider(ui, rcx, "halo chroma", &mut p.halo_chroma, 0.0, 0.3, &[chan("halo.chroma")]));
            ui.weak("blades 0 = round ghosts; 3+ = the iris polygon");
        }
        _ => {
            let [x, y] = &mut p.center;
            take(widgets::drags(ui, rcx, "light", &mut [x, y], 0.002, (-0.5, 1.5), &[chan("x"), chan("y")]));
            take(widgets::slider(ui, rcx, "intensity", &mut p.intensity, 0.0, 8.0, &[chan("intensity")]));
            take(rgb(ui, rcx, "tint", &mut p.tint, "tint", ungang, working, ["tint.r", "tint.g", "tint.b"]));
            take(widgets::slider(ui, rcx, "edge fade", &mut p.fade, 0.0, 1.0, &[chan("fade")]));
            ui.horizontal(|ui| {
                ui.label("combine");
                egui::ComboBox::from_id_salt(("lensflare-mode", node))
                    .selected_text(match p.mode {
                        GlowMode::Only => "flare only",
                        m => m.name(),
                    })
                    .show_ui(ui, |ui| {
                        for m in GlowMode::ALL {
                            let name = if m == GlowMode::Only { "flare only" } else { m.name() };
                            ui.selectable_value(&mut p.mode, m, name);
                        }
                    });
            });
            ui.checkbox(&mut p.occlude, egui::RichText::new("occlude by the picture at the light").small())
                .on_hover_text(
                    "the flare follows how bright the input is AT the light: a sun behind a \
                     building takes its flare with it. Costs one more read of the upstream \
                     per pixel",
                );
            if p.occlude {
                take(widgets::slider(ui, rcx, "threshold", &mut p.threshold, 0.0, 8.0, &[chan("threshold")]));
                take(widgets::slider(ui, rcx, "knee", &mut p.knee, 0.0, 4.0, &[chan("knee")]));
                if !wired {
                    ui.weak("nothing wired: occlusion reads black");
                }
            }
            ui.weak("drag the sun on the viewer to move the light;");
            ui.weak("lengths are fractions of the frame height");
        }
    }
    h
}

/// The graph glyph: a light with a streak, and two ghosts down the axis.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let sun = Pos2::new(c.x - r * 0.4, c.y - r * 0.4);
    painter.circle(sun, r * 0.26, fill, stroke);
    painter.line_segment([Pos2::new(sun.x - r * 0.75, sun.y), Pos2::new(sun.x + r * 0.95, sun.y)], stroke);
    painter.circle_stroke(Pos2::new(c.x + r * 0.2, c.y + r * 0.2), r * 0.16, stroke);
    painter.circle_stroke(Pos2::new(c.x + r * 0.6, c.y + r * 0.6), r * 0.28, stroke);
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::graph::NodeKind;

    /// Every page must FIT the tiles it asks for: a row pushed into the
    /// tile's scroll area cannot be clicked.
    #[test]
    fn every_page_fits_the_tiles_it_asks_for() {
        for tab in ["light", "glow", "ghosts"] {
            let mut p = LensFlareParams { occlude: true, ..Default::default() };
            let mut ungang = BTreeSet::new();
            let mut anim = Default::default();
            let ekind = NodeKind::LensFlare(p);
            let rows = crate::widgets::tile_rows(&ekind);
            let over = std::cell::Cell::new(0.0f32);
            let wide = std::cell::Cell::new(0.0f32);
            let span = |n: usize| {
                crate::widgets::TILE * n as f32 + crate::widgets::TILE_GAP * (n as f32 - 1.0)
            };
            let (w, hgt) = (span(TILES) - 14.0, span(rows) - 14.0 - 22.0);
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
                    let _ = panel(ui, &mut rcx, &mut p, tab, Colorspace::ACESCG, &mut ungang, false);
                });
                over.set(out.content_size.y - out.inner_rect.height());
                wide.set(out.content_size.x - out.inner_rect.width());
            });
            shot.save(&format!("lensflare-panel-{tab}"));
            assert!(over.get() <= 0.5, "{tab}: the panel overflows its tiles by {}pt", over.get());
            assert!(wide.get() <= 0.5, "{tab}: the panel overflows SIDEWAYS by {}pt", wide.get());
        }
    }
}
