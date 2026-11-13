//! CorridorKey's panel and glyph. The params live in
//! `engine::nodes::corridorkey`; the runner (install, render, process,
//! adopt) in `crate::corridorkey`.

use crate::corridorkey::{Action, View, ViewState, LICENCE_URL, REPO};
use crate::theme;
use crate::widgets;
use egui::{Color32, Painter, Pos2, Rect, Stroke, Ui, Vec2};
use engine::graph::{CorridorKeyParams, CorridorOut, CorridorScreen};

/// Licence, install, settings, the run and its status: more rows than a
/// tile holds.
pub const TILES: usize = 2;
pub const TILE_ROWS: usize = 2;

/// The properties panel. The click it hands back is acted on after the
/// node borrow ends (the GenAI pattern).
pub fn panel(ui: &mut Ui, node: u64, p: &mut CorridorKeyParams, v: &View) -> Option<Action> {
    let mut act = None;
    ui.horizontal_wrapped(|ui| {
        ui.weak("CorridorKey by Corridor Digital ·");
        ui.hyperlink_to("CC BY-NC-SA 4.0", LICENCE_URL)
            .on_hover_text("the tool's licence: keyed frames are free to deliver, commercially too; the tool itself may not be resold or served as a paid API");
        ui.weak("· pixels free to deliver");
    });
    // --- install --------------------------------------------------------
    if v.installed {
        ui.weak(format!("installed at {}", v.dir));
    } else if let Some(line) = &v.installing {
        ui.add(egui::Spinner::new().size(12.0));
        ui.weak(format!("installing… {line}"));
    } else {
        ui.horizontal_wrapped(|ui| {
            ui.colored_label(theme::error(), "not installed");
            let b = ui
                .add_enabled(v.uv, egui::Button::new("install…"))
                .on_hover_text(format!(
                    "git clone {REPO} into {} and uv sync there (torch: several GB). The tool then downloads its ~300 MB weights from Hugging Face on the first key. Nothing is fetched until you press this.",
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
    if let Some(r) = &v.install_result {
        match r {
            Ok(()) => ui.weak("installed"),
            Err(e) => ui.colored_label(theme::error(), format!("install failed: {e}")),
        };
    }
    ui.separator();
    // --- settings -------------------------------------------------------
    ui.horizontal(|ui| {
        ui.label("screen");
        egui::ComboBox::from_id_salt(("corridor-screen", node))
            .width(190.0)
            .selected_text(p.screen.label())
            .show_ui(ui, |ui| {
                for s in CorridorScreen::ALL {
                    ui.selectable_value(&mut p.screen, s, s.label());
                }
            });
    });
    widgets::plain_slider(ui, "despill", &mut p.despill, 0.0, 1.0, false)
        .on_hover_text("the tool's traditional despill on the reconstructed foreground");
    widgets::plain_slider(ui, "refiner", &mut p.refiner, 0.0, 2.0, false)
        .on_hover_text("how hard the detail pass pushes: 1 is the model's own");
    ui.horizontal(|ui| {
        ui.checkbox(&mut p.despeckle, "despeckle").on_hover_text("drop matte islands smaller than this many pixels: markers, noise");
        let mut px = p.despeckle_px as f32;
        ui.add_enabled_ui(p.despeckle, |ui| {
            ui.add(egui::DragValue::new(&mut px).update_while_editing(false).range(1.0..=100000.0).speed(widgets::rated(ui, 4.0)).suffix(" px"));
        });
        p.despeckle_px = px.max(1.0) as u32;
    });
    ui.horizontal(|ui| {
        ui.label("output");
        egui::ComboBox::from_id_salt(("corridor-out", node))
            .width(190.0)
            .selected_text(p.output.label())
            .show_ui(ui, |ui| {
                for m in CorridorOut::ALL {
                    ui.selectable_value(&mut p.output, m, m.label());
                }
            });
    });
    ui.weak(if v.has_hint {
        "hint: the wired matte (white = the subject, roughly)"
    } else {
        "hint: none wired — a coarse colour-difference key of the plate stands in; a roto or ScreenKey matte does better"
    });
    ui.separator();
    // --- the run --------------------------------------------------------
    let running = matches!(v.state, ViewState::Rendering(..) | ViewState::Keying(..));
    match &v.state {
        ViewState::Idle => {}
        ViewState::Rendering(which, done, total) => {
            let frac = if *total > 0 { *done as f32 / *total as f32 } else { 0.0 };
            ui.add(egui::ProgressBar::new(frac).text(format!("rendering the {which} {done}/{total}")));
        }
        ViewState::Keying(done, total) => {
            let frac = if *total > 0 { *done as f32 / *total as f32 } else { 0.0 };
            let words = if *total > 0 { format!("keying {done}/{total}") } else { "loading the model…".into() };
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
            ui.monospace(format!("keyed — up to date  ({n} frames)"));
        }
        Some((_, false)) => {
            ui.colored_label(theme::error(), "STALE — an input or a setting changed since this key (still playing the old one)");
        }
        None => {
            if !running {
                ui.weak(if v.on_disk { "a key for this recipe is on disk — load it" } else { "live: passing the plate through (nothing keyed)" });
            }
        }
    }
    ui.horizontal(|ui| {
        if running {
            if ui.button("cancel").clicked() {
                act = Some(Action::Cancel);
            }
        } else {
            let label = match v.loaded {
                Some((_, true)) => "re-key",
                _ if v.on_disk => "load key",
                _ => "key",
            };
            let can = v.has_input && v.installed && !v.busy_elsewhere;
            let b = ui.add_enabled(can, egui::Button::new(label)).on_hover_text(
                "renders the plate and the hint to disk, runs the tool over every frame of the plate's range, then plays the result here. Keyed on everything upstream and every setting: an edit shows as STALE, nothing re-runs behind your back",
            );
            if b.clicked() {
                act = Some(Action::Key);
            }
            if v.busy_elsewhere {
                ui.weak("another CorridorKey is running");
            }
            if matches!(v.state, ViewState::Failed(_) | ViewState::Done(_)) && ui.small_button("dismiss").clicked() {
                act = Some(Action::Dismiss);
            }
        }
    });
    if !v.log.is_empty() {
        egui::ScrollArea::vertical().max_height(80.0).id_salt(("corridor-log", node)).show(ui, |ui| {
            for l in &v.log {
                ui.add(egui::Label::new(egui::RichText::new(l).weak().small()).truncate());
            }
        });
    }
    act
}

/// The graph glyph: a screen with a figure cut out of it, and a spark for
/// the network — the ScreenKey glyph with a star.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.circle(c, r * 0.9, fill, stroke);
    let ink = crate::theme::node_ink();
    let screen = Rect::from_center_size(c, Vec2::new(r * 1.1, r * 0.9));
    painter.rect_filled(screen, r * 0.12, ink);
    painter.circle_filled(c + Vec2::new(0.0, -r * 0.12), r * 0.17, fill);
    let base = Pos2::new(c.x, screen.bottom());
    let pts: Vec<Pos2> = (0..=10)
        .map(|k| {
            let a = std::f32::consts::PI * (1.0 + k as f32 / 10.0);
            base + Vec2::new(a.cos(), a.sin()) * r * 0.36
        })
        .collect();
    painter.add(egui::Shape::convex_polygon(pts, fill, Stroke::NONE));
    // The spark, top right of the screen.
    let s = c + Vec2::new(r * 0.42, -r * 0.34);
    let k = r * 0.16;
    painter.line_segment([s - Vec2::new(k, 0.0), s + Vec2::new(k, 0.0)], Stroke::new(stroke.width, fill));
    painter.line_segment([s - Vec2::new(0.0, k), s + Vec2::new(0.0, k)], Stroke::new(stroke.width, fill));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn view(state: ViewState, installed: bool) -> View {
        View {
            dir: "/Users/someone/.wobble/CorridorKey".into(),
            installed,
            uv: true,
            installing: None,
            install_result: if installed { None } else { Some(Err("uv sync: exited with status 1".into())) },
            state,
            log: (0..8).map(|i| format!("Loading CorridorKey from checkpoints/CorridorKey_v1.0.safetensors, line {i}")).collect(),
            has_input: true,
            has_hint: false,
            loaded: Some((24, false)),
            on_disk: true,
            busy_elsewhere: false,
        }
    }

    /// The panel at its declared tiles, in its tallest states — not
    /// installed with a failed install, and keying with a stale key and a
    /// log tail — must not push the key / cancel row off the tile, where a
    /// suite could press it and a hand could not.
    #[test]
    fn the_panel_fits_the_tiles_it_asks_for() {
        let span = |n: usize| crate::widgets::TILE * n as f32 + crate::widgets::TILE_GAP * (n as f32 - 1.0);
        let (w, hgt) = (span(TILES) - 14.0, span(TILE_ROWS) - 14.0 - 22.0);
        for (name, v) in [
            ("corridor-panel-uninstalled", view(ViewState::Failed("the tool stopped without finishing (exit status: 1)".into()), false)),
            ("corridor-panel-keying", view(ViewState::Keying(7, 24), true)),
        ] {
            let mut p = CorridorKeyParams::new(0);
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
