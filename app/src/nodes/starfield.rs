//! Starfield's app half: the three pages (stars, sky, view), the channel
//! binding and the glyph. The params, the WGSL and the camera lowering are
//! `engine::nodes::starfield`.
//!
//! Every number is `star.<name>` on `Channel::Custom` (the Text node's
//! precedent), so the flat channel enum does not grow.

use crate::anim::{self, Channel};
use crate::rgbslider::RgbSpec;
use crate::widgets::{self, HudSlider, PanelTab, RowCtx};
use egui::{Color32, Painter, Pos2, Stroke};
use engine::color::{primaries_matrix, Colorspace, Primaries};
use engine::graph::{StarDist, StarfieldParams};
use engine::nodes::starfield::{MAX_LAYERS, PREFIX};
use std::collections::BTreeSet;

/// Two tiles: a pair of rgb rows needs track to aim along.
pub const TILES: usize = 2;

pub fn tabs() -> Vec<PanelTab> {
    vec![PanelTab::new("stars", "stars"), PanelTab::new("sky", "sky"), PanelTab::new("view", "view")]
}

/// The channel of a Starfield number by its name after `star.`.
pub fn chan(name: &str) -> Channel {
    Channel::Custom(anim::intern(&format!("{PREFIX}{name}")))
}

pub fn dyn_channels() -> Vec<Channel> {
    StarfieldParams::names().iter().map(|n| chan(n)).collect()
}

pub fn get(p: &StarfieldParams, ch: Channel) -> Option<f32> {
    match ch {
        Channel::Custom(i) => p.get(anim::interned(i).strip_prefix(PREFIX)?),
        _ => None,
    }
}

pub fn set(p: &mut StarfieldParams, ch: Channel, v: f32) {
    if let Channel::Custom(i) = ch {
        if let Some(n) = anim::interned(i).strip_prefix(PREFIX) {
            p.set(n, v);
        }
    }
}

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

/// `camera` is whether a Camera is wired into port 0: then the view page
/// says so instead of offering the manual view.
#[allow(clippy::too_many_arguments)]
pub fn panel(
    ui: &mut egui::Ui,
    rcx: &mut RowCtx,
    p: &mut StarfieldParams,
    tab: &str,
    working: Colorspace,
    ungang: &mut BTreeSet<String>,
    camera: bool,
) -> Option<HudSlider> {
    let node = rcx.node;
    let mut h: Option<HudSlider> = None;
    let mut take = |got: Option<HudSlider>| {
        if got.is_some() {
            h = got;
        }
    };
    match tab {
        "sky" => {
            ui.horizontal(|ui| {
                ui.add_sized([widgets::LABEL_W, 16.0], egui::Label::new(egui::RichText::new("spread").small()));
                for d in StarDist::ALL {
                    if ui
                        .selectable_label(p.dist == d, egui::RichText::new(d.label()).small())
                        .on_hover_text(d.hint())
                        .clicked()
                    {
                        p.dist = d;
                    }
                }
            });
            match p.dist {
                StarDist::Random | StarDist::Even => {
                    take(widgets::slider(ui, rcx, "density", &mut p.density, 0.0, 1.0, &[chan("density")]));
                    if p.dist == StarDist::Even {
                        ui.weak("even puts a star in every cell: density is unused");
                    }
                }
                StarDist::Clustered => {
                    take(widgets::slider(ui, rcx, "density", &mut p.density, 0.0, 1.0, &[chan("density")]));
                    take(widgets::slider(ui, rcx, "scale", &mut p.cluster_scale, 0.2, 12.0, &[chan("cluster.scale")]));
                    take(widgets::slider(
                        ui,
                        rcx,
                        "contrast",
                        &mut p.cluster_contrast,
                        0.0,
                        1.0,
                        &[chan("cluster.contrast")],
                    ));
                }
                StarDist::Band => {
                    take(widgets::slider(ui, rcx, "density", &mut p.density, 0.0, 1.0, &[chan("density")]));
                    take(widgets::slider(ui, rcx, "tilt", &mut p.band_tilt, -90.0, 90.0, &[chan("band.tilt")]));
                    take(widgets::slider(ui, rcx, "turn", &mut p.band_turn, -180.0, 180.0, &[chan("band.turn")]));
                    take(widgets::slider(ui, rcx, "width", &mut p.band_width, 1.0, 60.0, &[chan("band.width")]));
                    take(widgets::slider(ui, rcx, "floor", &mut p.band_floor, 0.0, 1.0, &[chan("band.floor")]));
                }
            }
            if matches!(p.dist, StarDist::Clustered | StarDist::Band) {
                take(widgets::slider(ui, rcx, "glow", &mut p.glow, 0.0, 1.0, &[chan("glow")]));
                take(rgb(ui, rcx, "glow colour", &mut p.glow_color, "glowcol", ungang, working, ["glow.r", "glow.g", "glow.b"]));
            }
        }
        "view" => {
            if camera {
                ui.weak("the wired camera's rotation and lens drive the sky;");
                ui.weak("its position does nothing (the stars are at infinity)");
            } else {
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
                take(widgets::slider(ui, rcx, "focal mm", &mut p.focal, 4.0, 300.0, &[chan("focal")]));
                let mut ap = p.aperture;
                widgets::plain_slider(ui, "filmback mm", &mut ap, 1.0, 70.0, false);
                p.aperture = ap.max(1.0);
                ui.weak("Y up, looking down -Z, like a Camera; wire a Camera");
                ui.weak("(a CameraTrack's too) into the input to follow it");
            }
        }
        _ => {
            widgets::gen_format_row(ui, ("starfield-format", node), &mut p.size, &mut p.inherit);
            take(widgets::slider(ui, rcx, "seed", &mut p.seed, 0.0, 100.0, &[chan("seed")]));
            take(widgets::slider(ui, rcx, "cells", &mut p.cells, 2.0, 128.0, &[chan("cells")]));
            take(widgets::slider(ui, rcx, "layers", &mut p.layers, 1.0, MAX_LAYERS as f32, &[chan("layers")]));
            take(widgets::slider(ui, rcx, "brightness", &mut p.brightness, 0.0, 20.0, &[chan("brightness")]));
            take(widgets::slider(ui, rcx, "falloff", &mut p.falloff, 0.0, 1.0, &[chan("falloff")]));
            take(widgets::slider(ui, rcx, "size px", &mut p.star_size, 0.1, 8.0, &[chan("size")]));
            take(widgets::slider(ui, rcx, "variation", &mut p.variation, 0.0, 1.0, &[chan("variation")]));
            take(widgets::slider(ui, rcx, "colour var", &mut p.color_var, 0.0, 1.0, &[chan("colorvar")]));
            take(rgb(ui, rcx, "tint", &mut p.tint, "tint", ungang, working, ["tint.r", "tint.g", "tint.b"]));
            ui.checkbox(&mut p.opaque, egui::RichText::new("opaque (alpha 1)").small())
                .on_hover_text("on: a background plate, alpha 1 everywhere. off: alpha is the light, to lay the stars over something");
        }
    }
    h
}

/// The graph glyph: a dark disc with a four-point star and two specks.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.circle(c, r * 0.9, Color32::from_gray(16), stroke);
    // Four spikes, each a triangle (a convex shape egui fills correctly).
    let star = |p: Pos2, s: f32| {
        let k = s * 0.22;
        for (dx, dy) in [(0.0, -1.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)] {
            let tip = Pos2::new(p.x + dx * s, p.y + dy * s);
            let a = Pos2::new(p.x - dy * k, p.y + dx * k);
            let b = Pos2::new(p.x + dy * k, p.y - dx * k);
            painter.add(egui::Shape::convex_polygon(vec![tip, a, b], fill, Stroke::NONE));
        }
    };
    star(Pos2::new(c.x - r * 0.1, c.y + r * 0.05), r * 0.55);
    painter.circle_filled(Pos2::new(c.x + r * 0.45, c.y - r * 0.4), r * 0.09, fill);
    painter.circle_filled(Pos2::new(c.x + r * 0.35, c.y + r * 0.5), r * 0.07, fill);
    painter.circle_filled(Pos2::new(c.x - r * 0.5, c.y - r * 0.45), r * 0.06, fill);
}
