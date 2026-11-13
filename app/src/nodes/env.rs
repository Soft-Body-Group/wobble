//! Env's app half: the view (a wired camera, or the node's own rotation
//! and lens), spin, exposure, the format and the glyph. The params, the
//! lookup and the WGSL are `engine::nodes::env`; the camera maths is
//! `engine::skyview`.
//!
//! Every number is `env.<name>` on `Channel::Custom` (the Text node's
//! precedent), so keys and patch cables reach them with no new variants.

use crate::anim::{self, Channel};
use crate::widgets::{self, HudSlider, RowCtx};
use egui::{Color32, Painter, Pos2, Stroke};
use engine::graph::EnvParams;
use engine::nodes::env::PREFIX;

pub const TILES: usize = 2;

/// The channel of an Env number by its name after `env.`.
pub fn chan(name: &str) -> Channel {
    Channel::Custom(anim::intern(&format!("{PREFIX}{name}")))
}

/// Every Env channel.
pub fn dyn_channels() -> Vec<Channel> {
    EnvParams::names().iter().map(|n| chan(n)).collect()
}

pub fn get(p: &EnvParams, ch: Channel) -> Option<f32> {
    match ch {
        Channel::Custom(i) => p.get(anim::interned(i).strip_prefix(PREFIX)?),
        _ => None,
    }
}

pub fn set(p: &mut EnvParams, ch: Channel, v: f32) {
    if let Channel::Custom(i) = ch {
        if let Some(name) = anim::interned(i).strip_prefix(PREFIX) {
            p.set(name, v);
        }
    }
}

/// `camera` = a Camera is wired into port 1 (it then owns the view).
pub fn panel(
    ui: &mut egui::Ui,
    rcx: &mut RowCtx,
    node: u64,
    p: &mut EnvParams,
    camera: bool,
) -> Option<HudSlider> {
    let mut hud = None;
    widgets::gen_format_row(ui, ("env-format", node), &mut p.size, &mut p.inherit);
    if camera {
        ui.weak("view: the wired camera's rotation and lens");
    } else {
        let [rx, ry, rz] = &mut p.rotate;
        let chans = [chan("rotate.x"), chan("rotate.y"), chan("rotate.z")];
        if let Some(h) = widgets::drags(ui, rcx, "rotate", &mut [rx, ry, rz], 0.5, (-180.0, 180.0), &chans) {
            hud = Some(h);
        }
        if let Some(h) = widgets::slider(ui, rcx, "focal mm", &mut p.focal, 8.0, 300.0, &[chan("focal")]) {
            hud = Some(h);
        }
    }
    if let Some(h) = widgets::slider(ui, rcx, "spin", &mut p.spin, -180.0, 180.0, &[chan("spin")]) {
        hud = Some(h);
    }
    if let Some(h) = widgets::slider(ui, rcx, "exposure", &mut p.exposure, -6.0, 6.0, &[chan("exposure")]) {
        hud = Some(h);
    }
    ui.weak("latlong on input 0, at infinity: a camera's translation does nothing");
    hud
}

/// A globe with its meridians, under a frame: the sky through a camera.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let rr = r * 0.82;
    painter.circle(c, rr, fill, stroke);
    let ink = Stroke::new((r * 0.1).max(1.0), crate::theme::node_ink());
    let n = 16;
    // The equator and two parallels.
    for k in [-0.5f32, 0.0, 0.5] {
        let y = c.y + k * rr;
        let hw = rr * (1.0 - k * k).sqrt();
        painter.line_segment([Pos2::new(c.x - hw, y), Pos2::new(c.x + hw, y)], ink);
    }
    // Two meridians, as ellipses' halves.
    for w in [0.45f32, 0.0] {
        let pts: Vec<Pos2> = (0..=n)
            .map(|i| {
                let t = std::f32::consts::PI * (i as f32 / n as f32 - 0.5);
                Pos2::new(c.x + w * rr * t.cos(), c.y + rr * t.sin())
            })
            .collect();
        painter.add(egui::Shape::line(pts, ink));
        if w > 0.0 {
            let pts: Vec<Pos2> = (0..=n)
                .map(|i| {
                    let t = std::f32::consts::PI * (i as f32 / n as f32 - 0.5);
                    Pos2::new(c.x - w * rr * t.cos(), c.y + rr * t.sin())
                })
                .collect();
            painter.add(egui::Shape::line(pts, ink));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::graph::NodeKind;

    #[test]
    fn every_channel_reads_and_writes() {
        let mut p = EnvParams::default();
        for (i, ch) in dyn_channels().into_iter().enumerate() {
            set(&mut p, ch, i as f32 + 0.25);
            assert_eq!(get(&p, ch), Some(i as f32 + 0.25));
        }
    }

    /// The panel fits its tiles, both ways, in either view.
    #[test]
    fn the_panel_fits_its_tiles() {
        for camera in [false, true] {
            let mut p = EnvParams::default();
            let mut anim = Default::default();
            let ekind = NodeKind::Env(p.clone());
            let (over, wide) = (std::cell::Cell::new(0.0f32), std::cell::Cell::new(0.0f32));
            let t = crate::widgets::TILE;
            let gap = crate::widgets::TILE_GAP;
            let shot = crate::snapshot::shoot(t * TILES as f32 + gap - 14.0, t - 14.0 - 22.0, |ui| {
                let out = egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                    let mut rcx = RowCtx {
                        node: 1,
                        anim: &mut anim,
                        ekind: &ekind,
                        t: 0,
                        hud_live: false,
                        driven: Vec::new(),
                    };
                    let _ = panel(ui, &mut rcx, 1, &mut p, camera);
                });
                over.set(out.content_size.y - out.inner_rect.height());
                wide.set(out.content_size.x - out.inner_rect.width());
            });
            shot.save(if camera { "env-panel-camera" } else { "env-panel" });
            assert!(over.get() <= 0.5, "overflows its tiles DOWN by {}pt", over.get());
            assert!(wide.get() <= 0.5, "overflows its tiles SIDEWAYS by {}pt", wide.get());
        }
    }
}
