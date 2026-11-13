//! Planet's app half: three pages (sphere, noise, look), the channel
//! binding and the glyph. The params and the WGSL are
//! `engine::nodes::planet`.
//!
//! Every number is `planet.<name>` on `Channel::Custom` (the Starfield's
//! precedent), so the flat channel enum does not grow.

use crate::anim::{self, Channel};
use crate::rgbslider::RgbSpec;
use crate::widgets::{self, HudSlider, PanelTab, RowCtx};
use egui::{Color32, Painter, Pos2, Stroke};
use engine::color::{primaries_matrix, Colorspace, Primaries};
use engine::graph::{PlanetMode, PlanetNoise, PlanetParams};
use engine::nodes::planet::PREFIX;
use std::collections::BTreeSet;

/// Two tiles: the rgb rows need track to aim along.
pub const TILES: usize = 2;
/// Two tall: the look page has the most rows.
pub const TILE_ROWS: usize = 2;

pub fn tabs() -> Vec<PanelTab> {
    vec![PanelTab::new("sphere", "sphere"), PanelTab::new("noise", "noise"), PanelTab::new("look", "look")]
}

/// The channel of a Planet number by its name after `planet.`.
pub fn chan(name: &str) -> Channel {
    Channel::Custom(anim::intern(&format!("{PREFIX}{name}")))
}

pub fn dyn_channels() -> Vec<Channel> {
    PlanetParams::names().iter().map(|n| chan(n)).collect()
}

pub fn get(p: &PlanetParams, ch: Channel) -> Option<f32> {
    match ch {
        Channel::Custom(i) => p.get(anim::interned(i).strip_prefix(PREFIX)?),
        _ => None,
    }
}

pub fn set(p: &mut PlanetParams, ch: Channel, v: f32) {
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

/// A row of small toggle buttons, one per choice.
fn choice<T: Copy + PartialEq>(ui: &mut egui::Ui, label: &str, v: &mut T, all: &[T], name: fn(T) -> &'static str) {
    ui.horizontal(|ui| {
        ui.add_sized([widgets::LABEL_W, 16.0], egui::Label::new(egui::RichText::new(label).small()));
        for &c in all {
            if ui.selectable_label(*v == c, egui::RichText::new(name(c)).small()).clicked() {
                *v = c;
            }
        }
    });
}

pub fn panel(
    ui: &mut egui::Ui,
    rcx: &mut RowCtx,
    p: &mut PlanetParams,
    tab: &str,
    working: Colorspace,
    ungang: &mut BTreeSet<String>,
) -> Option<HudSlider> {
    let node = rcx.node;
    let mut h: Option<HudSlider> = None;
    let mut take = |got: Option<HudSlider>| {
        if got.is_some() {
            h = got;
        }
    };
    match tab {
        "noise" => {
            choice(ui, "type", &mut p.noise, &PlanetNoise::ALL, PlanetNoise::name);
            take(widgets::slider(ui, rcx, "scale", &mut p.scale, 0.1, 20.0, &[chan("scale")]));
            take(widgets::slider(ui, rcx, "octaves", &mut p.octaves, 1.0, 8.0, &[chan("octaves")]));
            take(widgets::slider(ui, rcx, "lacunarity", &mut p.lacunarity, 1.0, 4.0, &[chan("lacunarity")]));
            take(widgets::slider(ui, rcx, "gain", &mut p.gain, 0.0, 1.0, &[chan("gain")]));
            take(widgets::slider(ui, rcx, "seed", &mut p.seed, 0.0, 100.0, &[chan("seed")]));
            take(widgets::slider(ui, rcx, "bands", &mut p.stretch, 0.1, 20.0, &[chan("stretch")]));
            take(widgets::slider(ui, rcx, "warp", &mut p.warp, 0.0, 2.0, &[chan("warp")]));
            take(widgets::slider(ui, rcx, "evolve", &mut p.evolve, 0.0, 0.2, &[chan("evolve")]));
            ui.weak("3D noise sampled ON the sphere: no poles, no seam;");
            ui.weak("bands > 1 squeezes it along the axis (gas giants)");
        }
        "look" => {
            take(widgets::slider(ui, rcx, "contrast", &mut p.contrast, 0.0, 6.0, &[chan("contrast")]));
            take(widgets::slider(ui, rcx, "offset", &mut p.offset, -1.0, 1.0, &[chan("offset")]));
            take(rgb(ui, rcx, "low", &mut p.low, "low", ungang, working, ["low.r", "low.g", "low.b"]));
            take(rgb(ui, rcx, "high", &mut p.high, "high", ungang, working, ["high.r", "high.g", "high.b"]));
            take(widgets::slider(ui, rcx, "exposure", &mut p.exposure, -6.0, 6.0, &[chan("exposure")]));
            take(widgets::slider(ui, rcx, "limb dark", &mut p.limb, 0.0, 1.0, &[chan("limb")]));
            take(widgets::slider(ui, rcx, "light", &mut p.light, 0.0, 1.0, &[chan("light")]));
            take(widgets::slider(ui, rcx, "ambient", &mut p.ambient, 0.0, 1.0, &[chan("ambient")]));
            take(widgets::slider(ui, rcx, "light az", &mut p.light_az, -180.0, 180.0, &[chan("light.az")]));
            take(widgets::slider(ui, rcx, "light el", &mut p.light_el, -90.0, 90.0, &[chan("light.el")]));
            ui.weak("light 0 = self-lit (a star); 1 = lit from az/el (a planet)");
        }
        _ => {
            widgets::gen_format_row(ui, ("planet-format", node), &mut p.size, &mut p.inherit);
            choice(ui, "draw", &mut p.mode, &PlanetMode::ALL, PlanetMode::name);
            if p.mode == PlanetMode::Disc {
                let [x, y] = &mut p.center;
                take(widgets::drags(ui, rcx, "centre px", &mut [x, y], 1.0, (-4096.0, 8192.0), &[chan("center.x"), chan("center.y")]));
                take(widgets::slider(ui, rcx, "radius px", &mut p.radius, 1.0, 2000.0, &[chan("radius")]));
                take(widgets::slider(ui, rcx, "edge px", &mut p.edge, 0.1, 20.0, &[chan("edge")]));
            }
            let [rx, ry, rz] = &mut p.rotate;
            take(widgets::drags(
                ui,
                rcx,
                "rotate",
                &mut [rx, ry, rz],
                0.5,
                (-180.0, 180.0),
                &[chan("rotate.x"), chan("rotate.y"), chan("rotate.z")],
            ));
            take(widgets::slider(ui, rcx, "spin °/f", &mut p.spin, -10.0, 10.0, &[chan("spin")]));
            if p.mode == PlanetMode::Latlong {
                ui.weak("the surface as an equirect (Blender's): seamless,");
                ui.weak("for an Env or a card; use a 2:1 format");
            } else {
                ui.weak("centre and radius match a Plasma corona's");
            }
        }
    }
    h
}

/// The graph glyph: a banded sphere with a terminator.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let rr = r * 0.8;
    painter.circle(c, rr, fill, stroke);
    for k in [-0.4f32, 0.05, 0.45] {
        let y = c.y + k * rr;
        let w = (1.0 - k * k).max(0.0).sqrt() * rr * 0.92;
        painter.line_segment([Pos2::new(c.x - w, y), Pos2::new(c.x + w, y)], stroke);
    }
    painter.circle_stroke(Pos2::new(c.x + rr * 0.35, c.y - rr * 0.35), rr * 0.18, stroke);
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::graph::NodeKind;

    /// Every page must FIT the tiles it asks for: a row pushed into the
    /// tile's scroll area cannot be clicked.
    #[test]
    fn every_page_fits_the_tiles_it_asks_for() {
        for tab in ["sphere", "noise", "look"] {
            let mut p = PlanetParams::default();
            let mut ungang = BTreeSet::new();
            let mut anim = Default::default();
            let ekind = NodeKind::Planet(p.clone());
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
                    let _ = panel(ui, &mut rcx, &mut p, tab, Colorspace::ACESCG, &mut ungang);
                });
                over.set(out.content_size.y - out.inner_rect.height());
                wide.set(out.content_size.x - out.inner_rect.width());
            });
            shot.save(&format!("planet-panel-{tab}"));
            assert!(over.get() <= 0.5, "{tab}: the panel overflows its tiles by {}pt", over.get());
            assert!(wide.get() <= 0.5, "{tab}: the panel overflows SIDEWAYS by {}pt", wide.get());
        }
    }
}
