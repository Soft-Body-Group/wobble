//! Refract's app half: three pages (glass, surface, view), the channel
//! binding and the glyph. The optics, the lookup and the WGSL are
//! `engine::nodes::refract`.
//!
//! Every number is `refract.<name>` on `Channel::Custom` (the Env
//! precedent), so keys and patch cables reach them with no new variants.

use crate::anim::{self, Channel};
use crate::widgets::{self, HudSlider, PanelTab, RowCtx};
use egui::{Color32, Painter, Pos2, Stroke, Vec2};
use engine::graph::{NormalY, RefractMap, RefractMode, RefractParams};
use engine::nodes::refract::{MAX_SAMPLES, PREFIX};

pub const TILES: usize = 2;

pub fn tabs() -> Vec<PanelTab> {
    vec![PanelTab::new("glass", "glass"), PanelTab::new("surface", "surface"), PanelTab::new("view", "view")]
}

/// The channel of a Refract number by its name after `refract.`.
pub fn chan(name: &str) -> Channel {
    Channel::Custom(anim::intern(&format!("{PREFIX}{name}")))
}

pub fn dyn_channels() -> Vec<Channel> {
    RefractParams::names().iter().map(|n| chan(n)).collect()
}

pub fn get(p: &RefractParams, ch: Channel) -> Option<f32> {
    match ch {
        Channel::Custom(i) => p.get(anim::interned(i).strip_prefix(PREFIX)?),
        _ => None,
    }
}

pub fn set(p: &mut RefractParams, ch: Channel, v: f32) {
    if let Channel::Custom(i) = ch {
        if let Some(n) = anim::interned(i).strip_prefix(PREFIX) {
            p.set(n, v);
        }
    }
}

fn pick_label(ui: &mut egui::Ui, text: &str) {
    ui.add_sized([56.0, 16.0], egui::Label::new(egui::RichText::new(text).small()));
}

/// `camera` = a Camera is wired into port 2 (it then owns the latlong view).
pub fn panel(
    ui: &mut egui::Ui,
    rcx: &mut RowCtx,
    p: &mut RefractParams,
    tab: &str,
    camera: bool,
) -> Option<HudSlider> {
    let mut h: Option<HudSlider> = None;
    let mut take = |got: Option<HudSlider>| {
        if got.is_some() {
            h = got;
        }
    };
    let latlong = p.mode == RefractMode::Latlong;
    match tab {
        "surface" => {
            ui.horizontal(|ui| {
                pick_label(ui, "map");
                ui.selectable_value(&mut p.map, RefractMap::Height, "height")
                    .on_hover_text("one channel (red) of height; the normal is its slope");
                ui.selectable_value(&mut p.map, RefractMap::Normal, "normal")
                    .on_hover_text("a tangent-space normal map");
            });
            match p.map {
                RefractMap::Height => {
                    take(widgets::slider(ui, rcx, "relief px", &mut p.height, 0.0, 200.0, &[chan("height")]));
                    ui.weak("how many pixels deep a height of 1 is");
                }
                RefractMap::Normal => {
                    ui.horizontal(|ui| {
                        pick_label(ui, "green");
                        ui.selectable_value(&mut p.normal_y, NormalY::Up, "up (OpenGL)")
                            .on_hover_text("Blender, Substance, Maya");
                        ui.selectable_value(&mut p.normal_y, NormalY::Down, "down (DirectX)")
                            .on_hover_text("Unreal, 3ds Max");
                    });
                    ui.checkbox(&mut p.signed, "already −1..1")
                        .on_hover_text("a float map that stores the normal itself, not 0..1 around 0.5");
                    take(widgets::slider(ui, rcx, "tilt", &mut p.normal, 0.0, 5.0, &[chan("normal")]));
                    ui.weak("load the map raw: a colour transform bends the normals");
                }
            }
        }
        "view" => {
            if !latlong {
                ui.weak("the view is for a latlong: a plate sits straight behind the glass");
            } else if camera {
                ui.weak("view: the wired camera's rotation and lens");
            } else {
                let [rx, ry, rz] = &mut p.rotate;
                let chans = [chan("rotate.x"), chan("rotate.y"), chan("rotate.z")];
                take(widgets::drags(ui, rcx, "rotate", &mut [rx, ry, rz], 0.5, (-180.0, 180.0), &chans));
                take(widgets::slider(ui, rcx, "focal mm", &mut p.focal, 8.0, 300.0, &[chan("focal")]));
            }
            ui.add_enabled_ui(latlong, |ui| {
                take(widgets::slider(ui, rcx, "spin", &mut p.spin, -180.0, 180.0, &[chan("spin")]));
            });
            take(widgets::slider(ui, rcx, "exposure", &mut p.exposure, -6.0, 6.0, &[chan("exposure")]));
        }
        _ => {
            ui.horizontal(|ui| {
                pick_label(ui, "behind");
                ui.selectable_value(&mut p.mode, RefractMode::Background, "plate")
                    .on_hover_text("input 0 is a picture straight behind the glass");
                ui.selectable_value(&mut p.mode, RefractMode::Latlong, "latlong")
                    .on_hover_text("input 0 is an equirectangular environment, seen through the camera (Env's)");
            });
            take(widgets::slider(ui, rcx, "ior", &mut p.ior, 1.0, 3.0, &[chan("ior")]));
            ui.add_enabled_ui(!latlong, |ui| {
                take(widgets::slider(ui, rcx, "thickness px", &mut p.thickness, 0.0, 500.0, &[chan("thickness")]));
            })
            .response
            .on_disabled_hover_text("a latlong is at infinity: glass depth cannot move it");
            take(widgets::slider(ui, rcx, "strength", &mut p.strength, 0.0, 4.0, &[chan("strength")]));
            take(widgets::slider(ui, rcx, "dispersion", &mut p.dispersion, 0.0, 0.3, &[chan("dispersion")]));
            take(widgets::slider(ui, rcx, "roughness", &mut p.roughness, 0.0, 1.0, &[chan("roughness")]));
            let taps = p.taps();
            ui.horizontal(|ui| {
                pick_label(ui, "samples");
                ui.add(egui::DragValue::new(&mut p.samples).range(1..=MAX_SAMPLES).speed(0.2));
                // Fused, but not free: say what it reads.
                if taps > 1 {
                    ui.weak(format!("reads input 0 {taps}x a pixel"));
                }
            });
        }
    }
    h
}

/// A pane of glass with a ray kinking through it.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let h = r * 0.8;
    painter.rect(
        egui::Rect::from_center_size(c, egui::vec2(h * 2.0, h * 1.1)),
        2.0,
        fill,
        stroke,
        egui::StrokeKind::Middle,
    );
    let ink = Stroke::new((r * 0.13).max(1.0), crate::theme::node_ink());
    // In from above, bent towards the normal inside, out parallel below.
    let pts = [
        c + Vec2::new(-h * 0.85, -h * 1.05),
        c + Vec2::new(-h * 0.3, -h * 0.55),
        c + Vec2::new(h * 0.05, h * 0.55),
        c + Vec2::new(h * 0.6, h * 1.05),
    ];
    painter.add(egui::Shape::line(pts.to_vec(), ink));
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::graph::NodeKind;

    #[test]
    fn every_channel_reads_and_writes() {
        let mut p = RefractParams::default();
        for (i, ch) in dyn_channels().into_iter().enumerate() {
            set(&mut p, ch, i as f32 + 0.25);
            assert_eq!(get(&p, ch), Some(i as f32 + 0.25));
        }
    }

    /// Every page fits its tiles, both ways, in every state that adds rows.
    #[test]
    fn the_panel_fits_its_tiles() {
        for tab in ["glass", "surface", "view"] {
            for (mode, map) in [
                (RefractMode::Background, RefractMap::Height),
                (RefractMode::Latlong, RefractMap::Normal),
            ] {
                let mut p = RefractParams { mode, map, roughness: 0.2, ..Default::default() };
                let mut anim = Default::default();
                let ekind = NodeKind::Refract(p.clone());
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
                        let _ = panel(ui, &mut rcx, &mut p, tab, false);
                    });
                    over.set(out.content_size.y - out.inner_rect.height());
                    wide.set(out.content_size.x - out.inner_rect.width());
                });
                shot.save(&format!("refract-panel-{tab}-{mode:?}"));
                assert!(over.get() <= 0.5, "{tab} {mode:?} overflows DOWN by {}pt", over.get());
                assert!(wide.get() <= 0.5, "{tab} {mode:?} overflows SIDEWAYS by {}pt", wide.get());
            }
        }
    }
}
