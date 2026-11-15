//! Relight's app half: the light controls, the albedo approximation, the
//! camera that ties the normals to the environment, the normal-map
//! convention, and the glyph. The params, the passes and the lowering are
//! `engine::nodes::relight`.
//!
//! Every number is `relight.<name>` on `Channel::Custom` (the Env node's
//! precedent), so keys and patch cables reach them with no new variants.

use crate::anim::{self, Channel};
use crate::widgets::{self, HudSlider, RowCtx};
use egui::{Color32, Painter, Pos2, Stroke};
use engine::graph::{RelightMode, RelightParams};
use engine::nodes::relight::PREFIX;

pub const TILES: usize = 2;
pub const TILE_ROWS: usize = 2;

/// The channel of a Relight number by its name after `relight.`.
pub fn chan(name: &str) -> Channel {
    Channel::Custom(anim::intern(&format!("{PREFIX}{name}")))
}

/// Every Relight channel.
pub fn dyn_channels() -> Vec<Channel> {
    RelightParams::names().iter().map(|n| chan(n)).collect()
}

pub fn get(p: &RelightParams, ch: Channel) -> Option<f32> {
    match ch {
        Channel::Custom(i) => p.get(anim::interned(i).strip_prefix(PREFIX)?),
        _ => None,
    }
}

pub fn set(p: &mut RelightParams, ch: Channel, v: f32) {
    if let Channel::Custom(i) = ch {
        if let Some(name) = anim::interned(i).strip_prefix(PREFIX) {
            p.set(name, v);
        }
    }
}

/// `wired` = which of plate, normals, depth, environment are connected.
pub fn panel(ui: &mut egui::Ui, rcx: &mut RowCtx, node: u64, p: &mut RelightParams, wired: [bool; 4]) -> Option<HudSlider> {
    let mut hud = None;
    macro_rules! s {
        ($label:expr, $v:expr, $lo:expr, $hi:expr, $name:expr) => {
            if let Some(h) = widgets::slider(ui, rcx, $label, $v, $lo, $hi, &[chan($name)]) {
                hud = Some(h);
            }
        };
    }
    if !wired[1] || !wired[3] {
        ui.colored_label(crate::theme::error(), "wire normals and an environment: until then the plate passes through");
    }
    ui.horizontal(|ui| {
        ui.label("output");
        egui::ComboBox::from_id_salt(("relight-mode", node)).width(110.0).selected_text(p.mode.name()).show_ui(ui, |ui| {
            for m in RelightMode::ALL {
                ui.selectable_value(&mut p.mode, m, m.name());
            }
        });
    });
    // --- the light ------------------------------------------------------
    s!("intensity", &mut p.intensity, 0.0, 4.0, "intensity");
    s!("diffuse", &mut p.diffuse, 0.0, 2.0, "diffuse");
    s!("specular", &mut p.specular, 0.0, 2.0, "specular");
    s!("roughness", &mut p.roughness, 0.0, 1.0, "roughness");
    s!("f0", &mut p.f0, 0.0, 1.0, "f0");
    // --- albedo ---------------------------------------------------------
    s!("albedo", &mut p.albedo, 0.0, 2.0, "albedo");
    s!("from plate", &mut p.from_plate, 0.0, 1.0, "from_plate");
    // --- the environment and the camera -----------------------------------
    s!("exposure", &mut p.exposure, -8.0, 8.0, "exposure");
    s!("spin", &mut p.spin, -180.0, 180.0, "spin");
    {
        let [rx, ry, rz] = &mut p.rotate;
        let chans = [chan("rotate.x"), chan("rotate.y"), chan("rotate.z")];
        if let Some(h) = widgets::drags(ui, rcx, "camera", &mut [rx, ry, rz], 0.5, (-180.0, 180.0), &chans) {
            hud = Some(h);
        }
    }
    s!("focal mm", &mut p.focal, 8.0, 300.0, "focal");
    if wired[2] {
        s!("radius", &mut p.radius, 0.0, 100.0, "radius");
    }
    // --- the normal map's convention --------------------------------------
    ui.horizontal(|ui| {
        ui.checkbox(&mut p.normals_01, "0..1").on_hover_text("the normals are stored n * 0.5 + 0.5 (normal-map colours)");
        ui.checkbox(&mut p.flip_x, "flip x");
        ui.checkbox(&mut p.flip_y, "flip y").on_hover_text("the map's y points down (DirectX-style)");
    });
    ui.weak("normals in camera space (x right, y up, z to camera); the latlong is Blender's, as the Env node reads it");
    hud
}

/// A ball lit from the upper left: a crescent of light and a highlight.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let rr = r * 0.8;
    painter.circle(c, rr, fill, stroke);
    let ink = crate::theme::node_ink();
    let n = 16;
    // The terminator: a half-ellipse from top to bottom, lit side left.
    let pts: Vec<Pos2> = (0..=n)
        .map(|i| {
            let t = std::f32::consts::PI * (i as f32 / n as f32 - 0.5);
            Pos2::new(c.x + 0.35 * rr * t.cos(), c.y + rr * t.sin())
        })
        .collect();
    painter.add(egui::Shape::line(pts, Stroke::new((r * 0.1).max(1.0), ink)));
    painter.circle_filled(Pos2::new(c.x - rr * 0.42, c.y - rr * 0.42), (r * 0.16).max(1.5), ink);
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::graph::NodeKind;

    #[test]
    fn every_channel_reads_and_writes() {
        let mut p = RelightParams::default();
        for (i, ch) in dyn_channels().into_iter().enumerate() {
            set(&mut p, ch, i as f32 + 0.25);
            assert_eq!(get(&p, ch), Some(i as f32 + 0.25));
        }
    }

    /// The panel fits its tiles, both ways, depth wired or not.
    #[test]
    fn the_panel_fits_its_tiles() {
        for wired in [[true, true, false, true], [true, true, true, true], [true, false, false, false]] {
            let mut p = RelightParams::default();
            let mut anim = Default::default();
            let ekind = NodeKind::Relight(p.clone());
            let (over, wide) = (std::cell::Cell::new(0.0f32), std::cell::Cell::new(0.0f32));
            let span = |n: usize| crate::widgets::TILE * n as f32 + crate::widgets::TILE_GAP * (n as f32 - 1.0);
            let shot = crate::snapshot::shoot(span(TILES) - 14.0, span(TILE_ROWS) - 14.0 - 22.0, |ui| {
                let out = egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                    let mut rcx = RowCtx { node: 1, anim: &mut anim, ekind: &ekind, t: 0, hud_live: false, driven: Vec::new() };
                    let _ = panel(ui, &mut rcx, 1, &mut p, wired);
                });
                over.set(out.content_size.y - out.inner_rect.height());
                wide.set(out.content_size.x - out.inner_rect.width());
            });
            shot.save(&format!("relight-panel-{}", wired.iter().filter(|w| **w).count()));
            assert!(over.get() <= 0.5, "{wired:?} overflows its tiles DOWN by {}pt", over.get());
            assert!(wide.get() <= 0.5, "{wired:?} overflows its tiles SIDEWAYS by {}pt", wide.get());
        }
    }
}
