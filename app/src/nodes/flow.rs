//! Flow's panel and glyph. The params live in `engine::nodes::flow`; the
//! runner (render, solve, adopt) in `crate::flowui`. The weights gate sits
//! above this panel, drawn for every node that declares models.

use crate::flowui::{Action, View, ViewState};
use crate::theme;
use egui::{Color32, Painter, Pos2, Stroke, Ui, Vec2};
use engine::graph::{FlowDetail, FlowEngine, FlowParams};

pub const TILES: usize = 2;
pub const TILE_ROWS: usize = 1;

/// The properties panel. The click it hands back is acted on after the
/// node borrow ends.
pub fn panel(ui: &mut Ui, node: u64, p: &mut FlowParams, v: &View) -> Option<Action> {
    ui.horizontal(|ui| {
        ui.label("engine");
        egui::ComboBox::from_id_salt(("flow-engine", node))
            .width(90.0)
            .selected_text(p.engine.label())
            .show_ui(ui, |ui| {
                for e in FlowEngine::ALL {
                    ui.selectable_value(&mut p.engine, e, e.label());
                }
            })
            .response
            .on_hover_text("SEA-RAFT: real optical flow, sharp at edges (79 MB of weights). RIFE: an interpolator's flow, lighter and rougher (21 MB)");
        ui.label("detail");
        egui::ComboBox::from_id_salt(("flow-detail", node))
            .width(90.0)
            .selected_text(p.detail.label())
            .show_ui(ui, |ui| {
                for d in FlowDetail::ALL {
                    ui.selectable_value(&mut p.detail, d, d.label());
                }
            })
            .response
            .on_hover_text("analyse at this fraction of the size: half suits 4K, and big moves");
        ui.checkbox(&mut p.backward, "backward")
            .on_hover_text("also solve where each pixel came FROM (a second pass a frame); off, it is the forward vector reversed");
    });
    ui.weak("out: rg = forward, ba = backward motion, in pixels");
    run_rows(
        ui,
        v,
        "not analysed: no motion yet",
        "renders the input over its range, solves motion between every pair of frames on the GPU, and plays the vectors here. Keyed on everything upstream: an edit shows as STALE, nothing re-runs behind your back",
    )
}

/// The run's progress, what is loaded and the analyse button: every node
/// that plays vectors from this runner shows the same rows. `idle` is what
/// an unanalysed node says, `tip` what the button does.
pub fn run_rows(ui: &mut Ui, v: &View, idle: &str, tip: &str) -> Option<Action> {
    let mut act = None;
    let running = matches!(v.state, ViewState::Rendering(..) | ViewState::Solving(..));
    match &v.state {
        ViewState::Idle => {}
        ViewState::Rendering(done, total) => {
            let frac = if *total > 0 { *done as f32 / *total as f32 } else { 0.0 };
            ui.add(egui::ProgressBar::new(frac).text(format!("rendering the input {done}/{total}")));
        }
        ViewState::Solving(done, total) => {
            let frac = if *total > 0 { *done as f32 / *total as f32 } else { 0.0 };
            let words = if *total > 0 { format!("solving {done}/{total}") } else { "loading the network…".into() };
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
            ui.monospace(format!("analysed — up to date  ({n} frames)"));
        }
        Some((_, false)) => {
            ui.colored_label(theme::error(), "STALE — the input or a setting changed (still playing the old vectors)");
        }
        None if !running => {
            ui.weak(if v.on_disk { "vectors for this input are on disk — load them" } else { idle });
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
                Some((_, true)) => "re-analyse",
                _ if v.on_disk => "load vectors",
                _ => "analyse",
            };
            let can = v.has_input && (v.weights || v.on_disk) && !v.busy_elsewhere;
            let b = ui.add_enabled(can, egui::Button::new(label)).on_hover_text(tip);
            if b.clicked() {
                act = Some(Action::Analyse);
            }
            if v.busy_elsewhere {
                ui.weak("another Flow is running");
            }
            if matches!(v.state, ViewState::Failed(_) | ViewState::Done(_)) && ui.small_button("dismiss").clicked() {
                act = Some(Action::Dismiss);
            }
        }
    });
    act
}

/// Three arrows of different lengths: a field of motion.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.circle(c, r * 0.9, fill, stroke);
    let ink = Stroke::new((r * 0.12).max(1.0), crate::theme::node_ink());
    for (dy, len, tilt) in [(-0.4f32, 0.5f32, -0.15f32), (0.0, 0.75, 0.0), (0.4, 0.55, 0.15)] {
        let a = c + Vec2::new(-r * 0.45, dy * r);
        let b = a + Vec2::new(len * r, tilt * r);
        painter.line_segment([a, b], ink);
        let back = (a - b).normalized() * r * 0.18;
        let side = Vec2::new(-back.y, back.x) * 0.7;
        painter.line_segment([b, b + back + side], ink);
        painter.line_segment([b, b + back - side], ink);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_panel_fits_the_tiles_it_asks_for() {
        let span = |n: usize| crate::widgets::TILE * n as f32 + crate::widgets::TILE_GAP * (n as f32 - 1.0);
        let (w, hgt) = (span(TILES) - 14.0, span(TILE_ROWS) - 14.0 - 22.0);
        for (name, state) in [
            ("flow-panel-failed", ViewState::Failed("SEA-RAFT M is not downloaded (78.8 MB) — the node's panel has the button".into())),
            ("flow-panel-solving", ViewState::Solving(7, 24)),
        ] {
            let v = View { state, has_input: true, weights: true, loaded: Some((24, false)), on_disk: false, busy_elsewhere: false };
            let mut p = FlowParams::new(0);
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
