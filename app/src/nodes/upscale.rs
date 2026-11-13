//! Upscale's app half: the model and scale pickers, the render button and
//! its progress, and the growing-square glyph. The download gate above the
//! panel comes from `NodeKind::models()`, as for every node that declares
//! weights; the runner is `crate::upscaleui`.

use crate::flowui::{Action, View, ViewState};
use crate::theme;
use egui::{Color32, Painter, Pos2, Stroke, Ui, Vec2};
use engine::graph::{UpscaleModel, UpscaleParams, UpscaleScale};

pub const TILES: usize = 2;
pub const TILE_ROWS: usize = 1;

/// What a frame costs, for the model's hover.
fn cost_words(m: UpscaleModel) -> &'static str {
    match m {
        UpscaleModel::Span => "fast: ~1.3 s a 1080p frame to 4K on an M4 Pro",
        UpscaleModel::Compact => "fast: ~3.5 s a 1080p frame to 4K on an M4 Pro",
        UpscaleModel::X4Plus => "the quality one, and slow: ~60 s a 1080p frame to 4K on an M4 Pro",
    }
}

/// The properties panel. The click it hands back is acted on after the
/// node borrow ends.
pub fn panel(ui: &mut Ui, node: u64, p: &mut UpscaleParams, v: &View, input: Option<(u32, u32)>) -> Option<Action> {
    let mut act = None;
    ui.horizontal(|ui| {
        ui.label("model");
        egui::ComboBox::from_id_salt(("upscale-model", node))
            .width(170.0)
            .selected_text(p.model.label())
            .show_ui(ui, |ui| {
                for m in UpscaleModel::ALL {
                    ui.selectable_value(&mut p.model, m, m.label()).on_hover_text(cost_words(m));
                }
            });
        ui.label("scale");
        ui.selectable_value(&mut p.scale, UpscaleScale::X4, "4x");
        ui.selectable_value(&mut p.scale, UpscaleScale::X2, "2x")
            .on_hover_text("the model's 4x, halved: more detail than asking for 2x");
    });
    if let Some((w, h)) = input {
        let s = p.scale.factor();
        ui.weak(format!("{w}×{h} ➡ {}×{}", w * s, h * s)).on_hover_text(cost_words(p.model));
    }
    let running = matches!(v.state, ViewState::Rendering(..) | ViewState::Solving(..));
    match &v.state {
        ViewState::Idle => {}
        ViewState::Rendering(done, total) => {
            let frac = if *total > 0 { *done as f32 / *total as f32 } else { 0.0 };
            ui.add(egui::ProgressBar::new(frac).text(format!("rendering the input {done}/{total}")));
        }
        ViewState::Solving(done, total) => {
            let frac = if *total > 0 { *done as f32 / *total as f32 } else { 0.0 };
            let words = if *total > 0 { format!("upscaling {done}/{total}") } else { "loading the network…".into() };
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
            ui.monospace(format!("upscaled — up to date  ({n} frames)"));
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
                "renders the input over its range, upscales every frame on the GPU, and plays the result here. Keyed on everything upstream: an edit shows as STALE, nothing re-runs behind your back",
            );
            if b.clicked() {
                act = Some(Action::Analyse);
            }
            if v.busy_elsewhere {
                ui.weak("another Upscale is running");
            }
            if matches!(v.state, ViewState::Failed(_) | ViewState::Done(_)) && ui.small_button("dismiss").clicked() {
                act = Some(Action::Dismiss);
            }
        }
    });
    if p.model == UpscaleModel::Span {
        ui.weak("SPAN weights: Philip Hofmann, CC-BY-4.0 — credit him");
    }
    act
}

/// A small square growing into a big one along its diagonal.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let big = egui::Rect::from_center_size(c, Vec2::splat(r * 1.7));
    painter.rect(big, r * 0.15, fill, stroke, egui::StrokeKind::Middle);
    let s = r * 0.62;
    let small = egui::Rect::from_min_size(Pos2::new(big.min.x + r * 0.22, big.max.y - r * 0.22 - s), Vec2::splat(s));
    painter.rect_filled(small, r * 0.08, theme::node_ink());
    let ink = Stroke::new((r * 0.14).max(1.0), theme::node_ink());
    let a = Pos2::new(small.max.x + r * 0.1, small.min.y - r * 0.1);
    let b = Pos2::new(big.max.x - r * 0.25, big.min.y + r * 0.25);
    painter.line_segment([a, b], ink);
    painter.line_segment([b, b + Vec2::new(-r * 0.32, 0.0)], ink);
    painter.line_segment([b, b + Vec2::new(0.0, r * 0.32)], ink);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_panel_fits_the_tiles_it_asks_for() {
        let span = |n: usize| crate::widgets::TILE * n as f32 + crate::widgets::TILE_GAP * (n as f32 - 1.0);
        let (w, hgt) = (span(TILES) - 14.0, span(TILE_ROWS) - 14.0 - 22.0);
        for (name, state, model) in [
            ("upscale-panel-failed", ViewState::Failed("Real-ESRGAN x4plus is not downloaded (67.0 MB)".into()), UpscaleModel::X4Plus),
            ("upscale-panel-running", ViewState::Solving(7, 24), UpscaleModel::Span),
        ] {
            let v = View { state, has_input: true, weights: true, loaded: Some((24, false)), on_disk: false, busy_elsewhere: false };
            let mut p = UpscaleParams { model, ..UpscaleParams::new(0) };
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
