//! Sammie's panel and glyph. The params live in `engine::nodes::sammie`;
//! the runner (install, clicks, render, track, adopt) in `crate::sammie`.

use crate::sammie::{Action, PreviewState, View, ViewState, LICENCE_URL, MATANYONE_LICENCE_URL, REPO};
use crate::theme;
use egui::{Color32, Painter, Pos2, Stroke, Ui, Vec2};
use engine::graph::{SamModel, SammieMatting, SammieOut, SammieParams};

/// Licence, install, settings, the clicks, the run and its status.
pub const TILES: usize = 2;
pub const TILE_ROWS: usize = 2;

/// The properties panel. The click it hands back is acted on after the
/// node borrow ends (the GenAI pattern).
pub fn panel(ui: &mut Ui, node: u64, p: &mut SammieParams, v: &View) -> Option<Action> {
    let mut act = None;
    ui.horizontal_wrapped(|ui| {
        ui.weak("Sammie-Roto 2 by Zarxrax ·");
        ui.hyperlink_to("GPL-3.0", LICENCE_URL).on_hover_text(
            "the tool's licence. You install it yourself and wobble runs it as a separate process: nothing of it ships with wobble. SAM 2 and EfficientTAM are Apache-2.0, so a plain segment is free to deliver",
        );
        ui.weak("· segments free to deliver");
    });
    // --- install --------------------------------------------------------
    if v.installed {
        ui.weak(format!("installed at {}", v.dir));
    } else if let Some(line) = &v.installing {
        ui.horizontal(|ui| {
            ui.add(egui::Spinner::new().size(12.0));
            ui.add(egui::Label::new(egui::RichText::new(format!("installing… {line}")).weak()).truncate());
        });
    } else {
        ui.horizontal_wrapped(|ui| {
            ui.colored_label(theme::error(), "not installed");
            let b = ui.add_enabled(v.uv, egui::Button::new("install…")).on_hover_text(format!(
                "git clone {REPO} into {} and uv sync there (torch: several GB). The picked model's weights (SAM 2: 150-900 MB, MatAnyone 2: ~140 MB) download on the first track. Nothing is fetched until you press this.",
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
    ui.separator();
    // --- settings -------------------------------------------------------
    let combo = |ui: &mut Ui, label: &str, salt: &str, text: &str, add: &mut dyn FnMut(&mut Ui)| {
        ui.horizontal(|ui| {
            ui.label(label);
            egui::ComboBox::from_id_salt((salt, node)).width(190.0).selected_text(text).show_ui(ui, |ui| add(ui));
        });
    };
    combo(ui, "model", "sammie-model", p.model.label(), &mut |ui| {
        for m in SamModel::ALL {
            ui.selectable_value(&mut p.model, m, m.label());
        }
    });
    combo(ui, "matting", "sammie-matting", p.matting.label(), &mut |ui| {
        for m in SammieMatting::ALL {
            ui.selectable_value(&mut p.matting, m, m.label());
        }
    });
    if !p.matting.commercial() {
        ui.horizontal_wrapped(|ui| {
            ui.colored_label(theme::error(), "MatAnyone 2 is");
            ui.hyperlink_to("NTU S-Lab 1.0", MATANYONE_LICENCE_URL);
            ui.colored_label(theme::error(), "— non-commercial use only. This node and everything below it are tagged.");
        });
    } else if p.nc_take {
        ui.colored_label(theme::error(), "the matte loaded was made with MatAnyone (non-commercial): still tagged until re-tracked");
    }
    combo(ui, "output", "sammie-out", p.output.label(), &mut |ui| {
        for m in SammieOut::ALL {
            ui.selectable_value(&mut p.output, m, m.label());
        }
    });
    ui.separator();
    // --- the clicks -----------------------------------------------------
    ui.checkbox(&mut p.auto_preview, "auto preview").on_hover_text(
        "a click in the viewer shows its segment straight away. The model stays loaded between clicks; the first click loads it, and downloads its weights if they are not on disk yet",
    );
    let here = p.points.iter().filter(|q| q.frame == v.tick).count();
    let keys = p.keyframes();
    if p.points.is_empty() {
        ui.weak("S-click the subject in the viewer to include it, shift+S-click to exclude; D-click a point removes it");
    } else {
        ui.horizontal_wrapped(|ui| {
            ui.label(format!("{} clicks on {} frame{}:", p.points.len(), keys.len(), if keys.len() == 1 { "" } else { "s" }));
            for f in keys.iter().take(8) {
                if ui.selectable_label(*f == v.tick, format!("{f}")).on_hover_text("go to this frame").clicked() {
                    act = Some(Action::Seek(*f));
                }
            }
            if keys.len() > 8 {
                ui.weak("…");
            }
        });
        ui.horizontal(|ui| {
            if ui.add_enabled(here > 0, egui::Button::new("clear this frame").small()).clicked() {
                act = Some(Action::ClearFrame);
            }
            if ui.small_button("clear all").clicked() {
                act = Some(Action::ClearAll);
            }
        });
    }
    ui.separator();
    // --- the run --------------------------------------------------------
    let running = matches!(v.state, ViewState::Rendering(..) | ViewState::Tracking(..));
    match &v.state {
        ViewState::Idle => {}
        ViewState::Rendering(done, total) if v.previewing => {
            let _ = (done, total);
            ui.add(egui::ProgressBar::new(0.0).text("previewing: rendering the frame"));
        }
        ViewState::Rendering(done, total) => {
            let frac = if *total > 0 { *done as f32 / *total as f32 } else { 0.0 };
            ui.add(egui::ProgressBar::new(frac).text(format!("rendering the plate {done}/{total}")));
        }
        ViewState::Tracking(..) if v.previewing => {
            ui.add(egui::ProgressBar::new(0.5).text("previewing: segmenting this frame…"));
        }
        ViewState::Tracking(done, total) => {
            let frac = if *total > 0 { *done as f32 / *total as f32 } else { 0.0 };
            let words = if *total > 0 { format!("tracking {done}/{total}") } else { "loading the model…".into() };
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
            ui.monospace(format!("tracked — up to date  ({n} frames)"));
        }
        Some((_, false)) => {
            ui.colored_label(theme::error(), "STALE — a click, a setting or the plate changed since this matte (still playing the old one)");
        }
        None if !running => {
            ui.weak(if v.on_disk { "a matte for these clicks is on disk — load it" } else { "nothing tracked yet" });
        }
        None => {}
    }
    match v.preview {
        PreviewState::Current(f, cov) if f == v.tick => {
            ui.weak(format!("preview: this frame, {:.1}% of the picture (green in the viewer)", cov * 100.0));
        }
        PreviewState::Current(f, _) => {
            ui.weak(format!("preview on frame {f}"));
        }
        PreviewState::Stale(f) => {
            ui.weak(format!("preview on frame {f} is out of date (grey): the clicks changed"));
        }
        PreviewState::None => {}
    }
    ui.horizontal(|ui| {
        if running {
            if ui.button("cancel").clicked() {
                act = Some(Action::Cancel);
            }
        } else {
            let label = match v.loaded {
                Some((_, true)) => "re-track",
                _ if v.on_disk => "load matte",
                _ => "track",
            };
            let can = v.has_input && v.installed && !v.busy_elsewhere && p.points.iter().any(|q| q.positive);
            let b = ui.add_enabled(can, egui::Button::new(label)).on_hover_text(
                "renders the plate to disk, segments the subject from your clicks and tracks it through the plate's range, then plays the matte here. Keyed on the plate, the clicks and the settings: an edit shows as STALE, nothing re-runs behind your back",
            );
            if b.clicked() {
                act = Some(Action::Track);
            }
            let here = p.points.iter().any(|q| q.frame == v.tick && q.positive);
            let pv = ui
                .add_enabled(here && v.has_input && v.installed && !v.busy_elsewhere, egui::Button::new("preview"))
                .on_hover_text(
                    "segments the frame on screen from its clicks and shows it over the viewer, green, before you track",
                );
            if pv.clicked() {
                act = Some(Action::Preview);
            }
            if v.busy_elsewhere {
                ui.weak("another Sammie is running");
            }
            if matches!(v.state, ViewState::Failed(_) | ViewState::Done(_)) && ui.small_button("dismiss").clicked() {
                act = Some(Action::Dismiss);
            }
        }
    });
    if !v.log.is_empty() {
        egui::ScrollArea::vertical().max_height(60.0).id_salt(("sammie-log", node)).show(ui, |ui| {
            for l in &v.log {
                ui.add(egui::Label::new(egui::RichText::new(l).weak().small()).truncate());
            }
        });
    }
    act
}

/// The graph glyph: a silhouette with an include dot on it and an
/// exclude cross beside it — what the node is driven by.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.circle(c, r * 0.9, fill, stroke);
    let ink = crate::theme::node_ink();
    // Head and shoulders.
    painter.circle_filled(c + Vec2::new(-r * 0.12, -r * 0.22), r * 0.24, ink);
    let base = Pos2::new(c.x - r * 0.12, c.y + r * 0.5);
    let pts: Vec<Pos2> = (0..=10)
        .map(|k| {
            let a = std::f32::consts::PI * (1.0 + k as f32 / 10.0);
            base + Vec2::new(a.cos() * 0.46, a.sin() * 0.42) * r
        })
        .collect();
    painter.add(egui::Shape::convex_polygon(pts, ink, Stroke::NONE));
    // The include click, on the figure.
    painter.circle_filled(c + Vec2::new(-r * 0.12, r * 0.22), r * 0.09, fill);
    // The exclude click, off it.
    let x = c + Vec2::new(r * 0.46, -r * 0.3);
    let k = r * 0.12;
    let s = Stroke::new(stroke.width, ink);
    painter.line_segment([x - Vec2::new(k, k), x + Vec2::new(k, k)], s);
    painter.line_segment([x - Vec2::new(k, -k), x + Vec2::new(k, -k)], s);
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::graph::SamPoint;

    fn view(state: ViewState, installed: bool) -> View {
        View {
            dir: "/Users/someone/.wobble/Sammie-Roto-2".into(),
            installed,
            uv: true,
            installing: None,
            install_result: if installed { None } else { Some(Err("uv sync: exited with status 1".into())) },
            state,
            log: (0..8).map(|i| format!("propagate in video: {i}0% | {i}/9 [00:00<00:02, 3.08it/s]")).collect(),
            has_input: true,
            loaded: Some((24, false)),
            on_disk: true,
            busy_elsewhere: false,
            tick: 1004,
            preview: PreviewState::Stale(1004),
            previewing: false,
        }
    }

    /// The panel at its declared tiles, in its tallest states, must not
    /// push the track / cancel row off the tile.
    #[test]
    fn the_panel_fits_the_tiles_it_asks_for() {
        let span = |n: usize| crate::widgets::TILE * n as f32 + crate::widgets::TILE_GAP * (n as f32 - 1.0);
        let (w, hgt) = (span(TILES) - 14.0, span(TILE_ROWS) - 14.0 - 22.0);
        for (name, v) in [
            ("sammie-panel-uninstalled", view(ViewState::Failed("the tool stopped without finishing (exit status: 1)".into()), false)),
            ("sammie-panel-tracking", view(ViewState::Tracking(7, 24), true)),
        ] {
            let mut p = SammieParams::new(0);
            p.matting = SammieMatting::MatAnyone2;
            p.points = (0..12)
                .map(|i| SamPoint { frame: 1000 + i, x: 0.5, y: 0.5, positive: i % 3 != 0 })
                .collect();
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
