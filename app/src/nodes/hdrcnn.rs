//! HDRCNN's app half: the model picker, the clip level, the render button
//! and its progress, and the glyph. The download gate above the panel comes
//! from `NodeKind::models()`; the runner is `crate::hdrcnnui`.

use crate::flowui::{Action, View, ViewState};
use crate::theme;
use egui::{Color32, Painter, Pos2, Stroke, Ui, Vec2};
use engine::graph::{HdrCnnModel, HdrCnnParams};

pub const TILES: usize = 2;
pub const TILE_ROWS: usize = 1;

pub const LICENCE_URL: &str = "https://github.com/gabrieleilertsen/hdrcnn/blob/master/LICENSE";

/// The properties panel. The click it hands back is acted on after the
/// node borrow ends.
pub fn panel(ui: &mut Ui, node: u64, p: &mut HdrCnnParams, v: &View) -> Option<Action> {
    let mut act = None;
    ui.horizontal(|ui| {
        ui.label("model");
        egui::ComboBox::from_id_salt(("hdrcnn-model", node)).width(170.0).selected_text(p.model.label()).show_ui(ui, |ui| {
            for m in HdrCnnModel::ALL {
                ui.selectable_value(&mut p.model, m, m.label());
            }
        });
        ui.label("clip");
        ui.add(egui::DragValue::new(&mut p.clip).range(0.01..=100.0).speed(0.01)).on_hover_text(
            "the level the plate clipped at, linear: the net reconstructs what is within 5% of it and leaves the rest exactly as it was. A new clip is a new render",
        );
    });
    ui.horizontal_wrapped(|ui| {
        ui.weak("Eilertsen et al. 2017 ·");
        ui.hyperlink_to("BSD-3", LICENCE_URL);
        ui.weak("· a CNN's guess at what clipped, not recovered data");
    });
    let running = matches!(v.state, ViewState::Rendering(..) | ViewState::Solving(..));
    match &v.state {
        ViewState::Idle => {}
        ViewState::Rendering(done, total) => {
            let frac = if *total > 0 { *done as f32 / *total as f32 } else { 0.0 };
            ui.add(egui::ProgressBar::new(frac).text(format!("rendering the input {done}/{total}")));
        }
        ViewState::Solving(done, total) => {
            let frac = if *total > 0 { *done as f32 / *total as f32 } else { 0.0 };
            let words = if *total > 0 { format!("reconstructing {done}/{total}") } else { "loading the network…".into() };
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
            ui.monospace(format!("reconstructed — up to date  ({n} frames)"));
        }
        Some((_, false)) => {
            ui.colored_label(theme::error(), "STALE — the input or a setting changed (still playing the old result)");
        }
        None if !running => {
            ui.weak(if v.on_disk { "a result for this input is on disk — load it" } else { "not rendered: passing the input through" });
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
                Some((_, true)) => "re-render",
                _ if v.on_disk => "load result",
                _ => "render",
            };
            let can = v.has_input && (v.weights || v.on_disk) && !v.busy_elsewhere;
            let b = ui.add_enabled(can, egui::Button::new(label)).on_hover_text(
                "renders the input over its range, reconstructs every frame on the GPU, and plays the result here. Keyed on everything upstream: an edit shows as STALE, nothing re-runs behind your back",
            );
            if b.clicked() {
                act = Some(Action::Analyse);
            }
            if v.busy_elsewhere {
                ui.weak("another HDRCNN is running");
            }
            if matches!(v.state, ViewState::Failed(_) | ViewState::Done(_)) && ui.small_button("dismiss").clicked() {
                act = Some(Action::Dismiss);
            }
        }
    });
    act
}

/// A disc clipped flat across its top, the cap drawn back above the line
/// by a small grid of cells: the network's reconstruction.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let sq = egui::Rect::from_center_size(c, Vec2::splat(r * 1.7));
    painter.rect(sq, r * 0.15, fill, stroke, egui::StrokeKind::Middle);
    let ink = theme::node_ink();
    let cut = c.y - r * 0.05;
    painter.line_segment([Pos2::new(c.x - r * 0.62, cut), Pos2::new(c.x + r * 0.62, cut)], Stroke::new((r * 0.12).max(1.0), ink));
    for (i, j) in [(-1, 0), (0, 0), (1, 0), (0, 1)] {
        let cell = egui::Rect::from_center_size(
            Pos2::new(c.x + i as f32 * r * 0.3, cut - r * 0.2 - j as f32 * r * 0.3),
            Vec2::splat(r * 0.22),
        );
        painter.rect_filled(cell, 0.0, ink);
    }
    painter.circle_stroke(c + Vec2::new(0.0, r * 0.32), r * 0.3, Stroke::new((r * 0.1).max(1.0), ink));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_panel_fits_the_tiles_it_asks_for() {
        let span = |n: usize| crate::widgets::TILE * n as f32 + crate::widgets::TILE_GAP * (n as f32 - 1.0);
        let (w, hgt) = (span(TILES) - 14.0, span(TILE_ROWS) - 14.0 - 22.0);
        let v = View {
            state: ViewState::Solving(12, 48),
            has_input: true,
            weights: true,
            loaded: Some((48, false)),
            on_disk: false,
            busy_elsewhere: false,
        };
        let mut p = HdrCnnParams::new(0);
        let over = std::cell::Cell::new(0.0f32);
        let wide = std::cell::Cell::new(0.0f32);
        let shot = crate::snapshot::shoot(w, hgt, |ui| {
            let out = egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                let _ = panel(ui, 1, &mut p, &v);
            });
            over.set(out.content_size.y - out.inner_rect.height());
            wide.set(out.content_size.x - out.inner_rect.width());
        });
        shot.save("hdrcnn-panel");
        assert!(over.get() <= 0.5, "overflows its tiles by {}pt", over.get());
        assert!(wide.get() <= 0.5, "overflows sideways by {}pt", wide.get());
    }
}
