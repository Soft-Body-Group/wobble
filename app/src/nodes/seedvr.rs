//! SeedVR2's panel and glyph. The params live in `engine::nodes::seedvr`;
//! the runner (install, render, process, adopt) in `crate::seedvr`.

use crate::seedvr::{weights_words, Action, View, ViewState, LICENCE_URL, REPO};
use crate::theme;
use egui::{Color32, Painter, Pos2, Stroke, Ui, Vec2};
use engine::graph::{SeedColour, SeedModel, SeedVrParams};

pub const TILES: usize = 2;
pub const TILE_ROWS: usize = 2;

/// Where it runs, measured, for the panel.
pub const WHERE: &str = "runs on an NVIDIA GPU (CUDA) or Apple Silicon. On an M4 Pro (48 GB) the 3B takes ~32 s a frame at 1080p out and 28 GB of memory at its peak; the 7B wants a big card — a 16 GB NVIDIA card runs its fp8 builds, 24 GB+ the fp16";

/// The properties panel. The click it hands back is acted on after the
/// node borrow ends.
pub fn panel(ui: &mut Ui, node: u64, p: &mut SeedVrParams, v: &View, input: Option<(u32, u32)>) -> Option<Action> {
    let mut act = None;
    ui.horizontal_wrapped(|ui| {
        ui.weak("SeedVR2 by ByteDance-Seed ·");
        ui.hyperlink_to("Apache-2.0", LICENCE_URL).on_hover_text("code and weights: free to use commercially");
        ui.weak("· via the numz CLI").on_hover_text(WHERE);
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
                "fetch {REPO} (pinned) into {} and uv sync there (torch: several GB). The first run then downloads {} from Hugging Face. Nothing is fetched until you press this. {WHERE}",
                v.dir,
                weights_words(p)
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
        ui.label("model");
        egui::ComboBox::from_id_salt(("seedvr-model", node)).width(80.0).selected_text(p.model.label()).show_ui(ui, |ui| {
            for m in SeedModel::ALL {
                ui.selectable_value(&mut p.model, m, m.label());
            }
        });
        ui.label("scale");
        ui.add(egui::DragValue::new(&mut p.scale).range(0.25..=8.0).speed(0.05).fixed_decimals(2).suffix("x"));
        ui.label("seed");
        ui.add(egui::DragValue::new(&mut p.seed).speed(1.0));
    });
    ui.horizontal(|ui| {
        ui.label("batch");
        ui.add(egui::DragValue::new(&mut p.batch).range(1..=129).speed(0.2))
            .on_hover_text("frames the model sees at once (it takes 4n + 1): more is steadier across time, and more memory");
        ui.label("colour");
        egui::ComboBox::from_id_salt(("seedvr-colour", node)).width(110.0).selected_text(p.colour.arg()).show_ui(ui, |ui| {
            for c in SeedColour::ALL {
                ui.selectable_value(&mut p.colour, c, c.arg());
            }
        }).response.on_hover_text("how the tool pulls the result's colour back to the input's");
    });
    if let Some(size) = input {
        let r = p.resolution_for(size);
        let k = r as f32 / size.0.min(size.1).max(1) as f32;
        ui.weak(format!("{}×{} ➡ {}×{}", size.0, size.1, (size.0 as f32 * k).round(), (size.1 as f32 * k).round()));
    }
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
            let words = if *total > 0 { format!("restoring {done}/{total}") } else { "loading the model…".into() };
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
            ui.monospace(format!("restored — up to date  ({n} frames)"));
        }
        Some((_, false)) => {
            ui.colored_label(theme::error(), "STALE — the input or a setting changed (still playing the old result)");
        }
        None if !running => {
            ui.weak(if v.on_disk { "a result for this input is on disk — load it" } else { "not run: passing the input through" });
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
            let b = ui.add_enabled(can, egui::Button::new(label)).on_hover_text(format!(
                "renders the input over its range, runs SeedVR2 over every frame (the first run downloads {}), then plays the result here. Keyed on everything upstream: an edit shows as STALE, nothing re-runs behind your back",
                weights_words(p)
            ));
            if b.clicked() {
                act = Some(Action::Run);
            }
            if v.busy_elsewhere {
                ui.weak("another SeedVR2 is running");
            }
            if matches!(v.state, ViewState::Failed(_) | ViewState::Done(_)) && ui.small_button("dismiss").clicked() {
                act = Some(Action::Dismiss);
            }
        }
    });
    if !v.log.is_empty() {
        egui::ScrollArea::vertical().max_height(60.0).id_salt(("seedvr-log", node)).show(ui, |ui| {
            for l in &v.log {
                ui.add(egui::Label::new(egui::RichText::new(l).weak().small()).truncate());
            }
        });
    }
    act
}

/// A seed sprouting out of a soft square: restoration grown, not drawn.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let sq = egui::Rect::from_center_size(c, Vec2::splat(r * 1.7));
    painter.rect(sq, r * 0.35, fill, stroke, egui::StrokeKind::Middle);
    let ink = theme::node_ink();
    let seed = c + Vec2::new(0.0, r * 0.32);
    painter.add(egui::Shape::convex_polygon(
        vec![
            seed + Vec2::new(0.0, -r * 0.28),
            seed + Vec2::new(r * 0.2, 0.0),
            seed + Vec2::new(0.0, r * 0.28),
            seed + Vec2::new(-r * 0.2, 0.0),
        ],
        ink,
        Stroke::NONE,
    ));
    let s = Stroke::new((r * 0.12).max(1.0), ink);
    let top = seed + Vec2::new(0.0, -r * 0.62);
    painter.line_segment([seed + Vec2::new(0.0, -r * 0.28), top], s);
    painter.line_segment([top, top + Vec2::new(-r * 0.3, r * 0.12)], s);
    painter.line_segment([top, top + Vec2::new(r * 0.3, r * 0.12)], s);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_panel_fits_the_tiles_it_asks_for() {
        let span = |n: usize| crate::widgets::TILE * n as f32 + crate::widgets::TILE_GAP * (n as f32 - 1.0);
        let (w, hgt) = (span(TILES) - 14.0, span(TILE_ROWS) - 14.0 - 22.0);
        for (name, state, installed) in [
            ("seedvr-panel-install", ViewState::Failed("SeedVR2 is not installed: press install…".into()), false),
            ("seedvr-panel-running", ViewState::Running(8, 24), true),
        ] {
            let v = View {
                dir: "/Users/someone/.wobble/SeedVR2".into(),
                installed,
                uv: true,
                installing: None,
                install_result: (!installed).then(|| Err("uv sync: exited with status 1".into())),
                state,
                log: vec!["[16:39:02] Phase 2: DiT upscaling".into(); 3],
                has_input: true,
                loaded: Some((24, false)),
                on_disk: false,
                busy_elsewhere: false,
            };
            let mut p = SeedVrParams::new(0);
            let over = std::cell::Cell::new(0.0f32);
            let wide = std::cell::Cell::new(0.0f32);
            let shot = crate::snapshot::shoot(w, hgt, |ui| {
                let out = egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                    let _ = panel(ui, 1, &mut p, &v, Some((1920, 1080)));
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
