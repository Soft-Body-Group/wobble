//! The Normals node's panel and glyph. The params live in
//! `engine::nodes::normals`; the runner (install, render, process, adopt)
//! in `crate::normals`. NormalCrafter fetches its own weights on the first
//! run (the CorridorKey precedent), so the panel says what that costs and
//! under which licence before the click.

use crate::normals::{Action, View, ViewState, LICENCE_URL, MODEL_PAGE, REPO, WEIGHTS_LICENCE_URL, WHERE};
use crate::theme;
use egui::{Color32, Painter, Pos2, Stroke, Ui, Vec2};
use engine::graph::NormalsParams;

pub const TILES: usize = 2;
pub const TILE_ROWS: usize = 2;

/// The properties panel. The click it hands back is acted on after the
/// node borrow ends.
pub fn panel(ui: &mut Ui, node: u64, p: &mut NormalsParams, v: &View) -> Option<Action> {
    let mut act = None;
    ui.horizontal_wrapped(|ui| {
        ui.weak("NormalCrafter by Bin et al. · code");
        ui.hyperlink_to("MIT", LICENCE_URL);
        ui.weak("· weights");
        ui.hyperlink_to(egui::RichText::new("Stability AI Community").color(theme::error()), WEIGHTS_LICENCE_URL).on_hover_text(
            "the weights are a fine-tune of Stable Video Diffusion: commercial use only under a revenue cap, with registration and an acceptable-use policy. The node is tagged, a Write below it warns, and a bake does not clear that",
        );
    });
    // --- install --------------------------------------------------------
    if v.installed {
        ui.weak(format!("installed at {}", v.dir)).on_hover_text(WHERE);
    } else if let Some(line) = &v.installing {
        ui.horizontal(|ui| {
            ui.add(egui::Spinner::new().size(12.0));
            ui.add(egui::Label::new(egui::RichText::new(format!("installing… {line}")).weak()).truncate());
        });
    } else {
        ui.horizontal_wrapped(|ui| {
            ui.colored_label(theme::error(), "not installed");
            let b = ui.add_enabled(v.uv, egui::Button::new("install…")).on_hover_text(format!(
                "fetch {REPO} (pinned) into {} and uv sync there (torch and diffusers: several GB). The first run then downloads the weights (~5 GB, {MODEL_PAGE}) into your Hugging Face cache. Nothing is fetched until you press this. {WHERE}",
                v.dir
            ));
            if b.clicked() {
                act = Some(Action::Install);
            }
            if !v.uv {
                ui.weak("needs uv: brew install uv");
            }
        });
    }
    if let Some(Err(e)) = &v.install_result {
        ui.colored_label(theme::error(), format!("install failed: {e}"));
    }
    // --- settings -------------------------------------------------------
    ui.horizontal(|ui| {
        ui.label("res");
        ui.add(egui::DragValue::new(&mut p.max_res).range(256..=2048).speed(8.0))
            .on_hover_text("the network's long side (a multiple of 64): finer and hungrier as it grows, ~20 GB of GPU memory at 1024; it steps down on its own when the card runs out. The result comes back at the plate's size");
        ui.label("window");
        ui.add(egui::DragValue::new(&mut p.window).range(2..=25))
            .on_hover_text("frames the network sees at once");
        ui.label("step");
        ui.add(egui::DragValue::new(&mut p.stride).range(1..=25))
            .on_hover_text("how far each window moves on: the overlap (window - step) is blended, which is what keeps the normals steady");
    });
    ui.horizontal(|ui| {
        ui.checkbox(&mut p.flip_x, "flip x");
        ui.checkbox(&mut p.flip_y, "flip y").on_hover_text("y down (DirectX-style) instead of y up");
        ui.checkbox(&mut p.encode01, "0..1").on_hover_text("n * 0.5 + 0.5: normal-map colours. Relight reads either");
    });
    ui.weak("camera space: x right, y up, z towards the camera; flips and encoding never re-run");
    // --- the run --------------------------------------------------------
    let running = matches!(v.state, ViewState::Rendering(..) | ViewState::Running(..));
    match &v.state {
        ViewState::Idle => {}
        ViewState::Rendering(done, total) => {
            let frac = if *total > 0 { *done as f32 / *total as f32 } else { 0.0 };
            ui.add(egui::ProgressBar::new(frac).text(format!("rendering the input {done}/{total}")));
        }
        ViewState::Running(done, total) => {
            let frac = if *total > 0 { *done as f32 / *total as f32 } else { 0.0 };
            let words = if *total > 0 { format!("normals {done}/{total}") } else { "loading the model…".into() };
            ui.add(egui::ProgressBar::new(frac).text(words));
        }
        ViewState::Failed(e) => {
            ui.colored_label(theme::error(), format!("failed: {e}"));
        }
        ViewState::Done(s) => {
            ui.weak(s);
        }
    }
    match v.loaded {
        Some((n, true)) => {
            ui.monospace(format!("normals — up to date  ({n} frames)"));
        }
        Some((_, false)) => {
            ui.colored_label(theme::error(), "STALE — the input or a setting changed (still playing the old normals)");
        }
        None if !running => {
            ui.weak(if v.on_disk { "a result for this input is on disk — load it" } else { "not run: no normals (outputs 0)" });
        }
        None => {}
    }
    ui.horizontal(|ui| {
        if running {
            if ui.button("cancel").clicked() {
                act = Some(Action::Cancel);
            }
        } else {
            let label = match v.loaded {
                Some((_, true)) => "re-run",
                _ if v.on_disk => "load result",
                _ => "run",
            };
            let can = v.has_input && (v.installed || v.on_disk) && !v.busy_elsewhere;
            let b = ui.add_enabled(can, egui::Button::new(label)).on_hover_text(
                "renders the input over its range, runs NormalCrafter over the clip (the first run downloads its weights), then plays the normals here. Keyed on everything upstream: an edit shows as STALE, nothing re-runs behind your back",
            );
            if b.clicked() {
                act = Some(Action::Run);
            }
            if v.busy_elsewhere {
                ui.weak("another Normals is running");
            }
            if matches!(v.state, ViewState::Failed(_) | ViewState::Done(_)) && ui.small_button("dismiss").clicked() {
                act = Some(Action::Dismiss);
            }
        }
    });
    if !v.log.is_empty() {
        egui::ScrollArea::vertical().max_height(60.0).id_salt(("normals-log", node)).show(ui, |ui| {
            for l in &v.log {
                ui.add(egui::Label::new(egui::RichText::new(l).weak().small()).truncate());
            }
        });
    }
    act
}

/// A hemisphere with three normals fanning out of it.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let sq = egui::Rect::from_center_size(c, Vec2::splat(r * 1.7));
    painter.rect(sq, r * 0.35, fill, stroke, egui::StrokeKind::Middle);
    let ink = Stroke::new((r * 0.12).max(1.0), theme::node_ink());
    let base = Pos2::new(c.x, c.y + r * 0.45);
    let rr = r * 0.55;
    let n = 12;
    let arc: Vec<Pos2> = (0..=n)
        .map(|i| {
            let t = std::f32::consts::PI * i as f32 / n as f32;
            Pos2::new(base.x - rr * t.cos(), base.y - rr * t.sin())
        })
        .collect();
    painter.add(egui::Shape::line(arc, ink));
    for a in [-0.55f32, 0.0, 0.55] {
        let t = std::f32::consts::FRAC_PI_2 + a;
        let (dx, dy) = (-t.cos(), -t.sin());
        let p0 = Pos2::new(base.x + rr * dx, base.y + rr * dy);
        let p1 = Pos2::new(base.x + (rr + r * 0.38) * dx, base.y + (rr + r * 0.38) * dy);
        painter.line_segment([p0, p1], ink);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_panel_fits_the_tiles_it_asks_for() {
        let span = |n: usize| crate::widgets::TILE * n as f32 + crate::widgets::TILE_GAP * (n as f32 - 1.0);
        let (w, hgt) = (span(TILES) - 14.0, span(TILE_ROWS) - 14.0 - 22.0);
        for (name, state, installed) in [
            ("normals-panel-install", ViewState::Failed("NormalCrafter is not installed: press install…".into()), false),
            ("normals-panel-running", ViewState::Running(22, 48), true),
        ] {
            let v = View {
                dir: "/Users/someone/.wobble/NormalCrafter".into(),
                installed,
                uv: true,
                installing: None,
                install_result: (!installed).then(|| Err("uv sync: exited with status 1".into())),
                state,
                log: vec!["running on NVIDIA GeForce RTX 5080, long side 1024, window 14 / stride 10".into(); 3],
                has_input: true,
                loaded: Some((48, false)),
                on_disk: false,
                busy_elsewhere: false,
            };
            let mut p = NormalsParams::new(0);
            let over = std::cell::Cell::new(0.0f32);
            let wide = std::cell::Cell::new(0.0f32);
            let shot = crate::snapshot::shoot(w, hgt, |ui| {
                let out = egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                    let _ = panel(ui, 1, &mut p, &v);
                });
                over.set(out.content_size.y - out.inner_rect.height());
                wide.set(out.content_size.x - out.inner_rect.width());
            });
            shot.save(name);
            assert!(over.get() <= 0.5, "{name} overflows its tiles by {}pt", over.get());
            assert!(wide.get() <= 0.5, "{name} overflows sideways by {}pt", wide.get());
        }
    }
}
