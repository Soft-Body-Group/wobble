//! Splat's app half: the panel, the channel binding and the glyph. The
//! params, the loader, the GPU renderer and the camera lowering are
//! `engine::nodes::splat`, `engine::splat` and `engine::splatgpu`.
//!
//! Every number is `splat.<name>` on `Channel::Custom` (the Starfield's
//! precedent), so the flat channel enum does not grow.

use crate::anim::{self, Channel};
use crate::widgets::{self, HudSlider, RowCtx};
use egui::{Color32, Painter, Pos2, Stroke};
use engine::graph::SplatParams;
use engine::nodes::splat::PREFIX;
use engine::splat::State;

/// The path field wants the width of two tiles.
pub const TILES: usize = 2;

/// The channel of a Splat number by its name after `splat.`.
pub fn chan(name: &str) -> Channel {
    Channel::Custom(anim::intern(&format!("{PREFIX}{name}")))
}

pub fn dyn_channels() -> Vec<Channel> {
    SplatParams::names().iter().map(|n| chan(n)).collect()
}

pub fn get(p: &SplatParams, ch: Channel) -> Option<f32> {
    match ch {
        Channel::Custom(i) => p.get(anim::interned(i).strip_prefix(PREFIX)?),
        _ => None,
    }
}

pub fn set(p: &mut SplatParams, ch: Channel, v: f32) {
    if let Channel::Custom(i) = ch {
        if let Some(n) = anim::interned(i).strip_prefix(PREFIX) {
            p.set(n, v);
        }
    }
}

/// The path, typed or picked (the LUT node's row). Returns true when it
/// changed.
fn path_row(ui: &mut egui::Ui, node: u64, path: &mut String) -> bool {
    let id = egui::Id::new(("splat-path", node));
    let buf_id = id.with("buf");
    // Read before any data closure: egui's lock is not re-entrant.
    let editing = ui.memory(|m| m.has_focus(id));
    let held: Option<String> = if editing { ui.data(|d| d.get_temp(buf_id)) } else { None };
    let mut buf = held.unwrap_or_else(|| path.clone());
    let before = path.clone();
    ui.horizontal(|ui| {
        ui.add_sized([widgets::LABEL_W, 18.0], egui::Label::new("file"));
        let w = (ui.available_width() - 30.0).max(40.0);
        let r = ui.add(egui::TextEdit::singleline(&mut buf).id(id).desired_width(w).hint_text(".ply .spz"));
        if r.changed() || r.gained_focus() {
            ui.data_mut(|d| d.insert_temp(buf_id, buf.clone()));
        }
        if r.lost_focus() {
            let walked_away = ui.input(|i| i.key_pressed(egui::Key::Escape));
            if !walked_away && buf != *path {
                *path = buf.trim().to_string();
            }
            ui.data_mut(|d| d.remove::<String>(buf_id));
        }
        if ui.small_button("…").on_hover_text("pick a Gaussian splat capture").clicked() {
            if let Some(f) = crate::filedialog::FileDialog::new().add_filter("Gaussian splats", &["ply", "spz"]).pick_file()
            {
                *path = f.to_string_lossy().into_owned();
            }
        }
    });
    *path != before
}

/// `camera` is whether a Camera is wired into port 0.
pub fn panel(ui: &mut egui::Ui, rcx: &mut RowCtx, p: &mut SplatParams, camera: bool) -> Option<HudSlider> {
    let node = rcx.node;
    let mut h: Option<HudSlider> = None;
    let mut take = |got: Option<HudSlider>| {
        if got.is_some() {
            h = got;
        }
    };
    // A capture picked HERE is fitted to the default camera once it has
    // loaded; one that came with the setup keeps its placement.
    let fit_id = egui::Id::new(("splat-fit", node));
    if path_row(ui, node, &mut p.path) {
        ui.data_mut(|d| d.insert_temp(fit_id, true));
    }
    let state = if p.path.is_empty() { None } else { Some(engine::splat::request(&p.path).0) };
    ui.weak(p.status());
    let pending: bool = ui.data(|d| d.get_temp(fit_id)).unwrap_or(false);
    match &state {
        Some(State::Ready(c)) => {
            if pending {
                p.fit(c);
                ui.data_mut(|d| d.remove::<bool>(fit_id));
            }
            if ui
                .small_button("fit to frame")
                .on_hover_text("place and scale the capture so the default Camera sees all of it")
                .clicked()
            {
                p.fit(c);
            }
        }
        Some(State::Loading) => ui.ctx().request_repaint_after(std::time::Duration::from_millis(100)),
        _ => {}
    }
    widgets::gen_format_row(ui, ("splat-format", node), &mut p.size, &mut p.inherit);
    {
        let [x, y, z] = &mut p.translate;
        take(widgets::drags(
            ui,
            rcx,
            "translate",
            &mut [x, y, z],
            0.01,
            (-10.0, 10.0),
            &[chan("translate.x"), chan("translate.y"), chan("translate.z")],
        ));
    }
    {
        let [x, y, z] = &mut p.rotate;
        take(widgets::drags(
            ui,
            rcx,
            "rotate",
            &mut [x, y, z],
            0.5,
            (-180.0, 180.0),
            &[chan("rotate.x"), chan("rotate.y"), chan("rotate.z")],
        ));
    }
    take(widgets::drags(ui, rcx, "scale", &mut [&mut p.scale], 0.005, (0.0, 10.0), &[chan("scale")]));
    take(widgets::slider(ui, rcx, "opacity", &mut p.opacity, 0.0, 2.0, &[chan("opacity")]));
    take(widgets::slider(ui, rcx, "splat size", &mut p.splat_size, 0.0, 2.0, &[chan("size")]));
    ui.checkbox(&mut p.sh, egui::RichText::new("view-dependent colour").small())
        .on_hover_text("the capture's spherical harmonics past the first band: highlights that move with the camera");
    ui.checkbox(&mut p.convert, egui::RichText::new("turn a .ply Y up").small())
        .on_hover_text("a .ply from training is Y down, Z forward; on turns it to wobble's Y up. An .spz already is");
    ui.checkbox(&mut p.linear, egui::RichText::new("colours are linear").small())
        .on_hover_text("off: the capture was trained on sRGB pictures (nearly all are) and is decoded to scene-linear");
    if camera {
        ui.weak("drawn through the wired Camera");
    } else {
        ui.weak("wire a Camera (a CameraTrack's too) into the input;");
        ui.weak("unwired, a default 50 mm looking down -Z from z = 1.39");
    }
    ui.weak("output 2 is depth: distance from the camera");
    h
}

/// The graph glyph: a few soft blobs overlapping, one of them a tilted
/// ellipse — Gaussians.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let ellipse = |ctr: Pos2, rx: f32, ry: f32, tilt: f32, col: Color32| {
        let (s, co) = tilt.sin_cos();
        let pts: Vec<Pos2> = (0..20)
            .map(|i| {
                let a = i as f32 / 20.0 * std::f32::consts::TAU;
                let (x, y) = (a.cos() * rx, a.sin() * ry);
                Pos2::new(ctr.x + x * co - y * s, ctr.y + x * s + y * co)
            })
            .collect();
        painter.add(egui::Shape::convex_polygon(pts, col, stroke));
    };
    ellipse(Pos2::new(c.x - r * 0.3, c.y + r * 0.25), r * 0.5, r * 0.32, 0.5, fill.gamma_multiply(0.55));
    ellipse(Pos2::new(c.x + r * 0.3, c.y + r * 0.1), r * 0.42, r * 0.42, 0.0, fill.gamma_multiply(0.75));
    ellipse(Pos2::new(c.x, c.y - r * 0.3), r * 0.55, r * 0.26, -0.4, fill);
}
