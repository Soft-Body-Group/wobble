//! Frame Fix's app half: the panel and the glyph. The params, the time
//! branches and the flow plan are `engine::nodes::framefix`; flow mode's
//! runner (render, make, adopt) is the Rifetime one (`crate::flowui`). The
//! weights gate sits above this panel in flow mode.

use crate::anim::Channel;
use crate::flowui::{Action, View, ViewState};
use crate::theme;
use egui::{Color32, Painter, Pos2, Stroke, Ui, Vec2};
use engine::graph::{FlowDetail, FrameFixMode, FrameFixParams};
use engine::nodes::framefix::{frames_text, parse_frames};

pub const CHANNELS: &[Channel] = &[];
pub const TILES: usize = 2;
pub const TILE_ROWS: usize = 1;

pub fn panel(ui: &mut Ui, node: u64, t: u64, p: &mut FrameFixParams, v: &View) -> Option<Action> {
    let mut act = None;
    // The list as text, kept while it is being typed into (a half-typed
    // range must not be folded under the cursor).
    let id = egui::Id::new(("framefix-frames", node));
    let mut text = ui.data_mut(|d| d.get_temp::<String>(id)).unwrap_or_else(|| frames_text(&p.frames));
    let mut bad_text = None;
    ui.horizontal(|ui| {
        ui.label("frames");
        let r = ui.add(egui::TextEdit::singleline(&mut text).desired_width(150.0).hint_text("1012, 1040-1042")).on_hover_text(
            "the bad comp frames: numbers and ranges, comma separated",
        );
        if r.changed() {
            match parse_frames(&text) {
                Ok(f) => p.frames = f,
                Err(e) => bad_text = Some(e),
            }
        }
        if r.has_focus() {
            ui.data_mut(|d| d.insert_temp(id, text.clone()));
        } else {
            ui.data_mut(|d| d.remove::<String>(id));
        }
        let here = t as i64;
        let on = p.frames.contains(&here);
        if ui
            .small_button(if on { "− this frame" } else { "+ this frame" })
            .on_hover_text(format!("{} frame {here}, the one under the playhead", if on { "un-mark" } else { "mark" }))
            .clicked()
        {
            if on {
                p.frames.retain(|f| *f != here);
            } else {
                p.frames.push(here);
                p.frames.sort_unstable();
            }
            ui.data_mut(|d| d.remove::<String>(id));
        }
    });
    if let Some(e) = bad_text {
        ui.colored_label(theme::error(), e);
    }
    ui.horizontal(|ui| {
        ui.label("fix");
        for m in FrameFixMode::ALL {
            ui.selectable_value(&mut p.mode, m, m.label());
        }
        if p.mode == FrameFixMode::Flow {
            egui::ComboBox::from_id_salt(("framefix-detail", node))
                .width(70.0)
                .selected_text(p.detail.label())
                .show_ui(ui, |ui| {
                    for d in FlowDetail::ALL {
                        ui.selectable_value(&mut p.detail, d, d.label());
                    }
                })
                .response
                .on_hover_text("judge the motion at this fraction of the size (frames are always made at full size)");
        }
    });
    if p.mode == FrameFixMode::Blend {
        ui.weak("live cross-fade of the good frames either side; decodes the input twice a frame");
        return None;
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
            let words = if *total > 0 { format!("fixing {done}/{total}") } else { "loading the network…".into() };
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
        Some((_, true)) => {
            ui.monospace("fixed — up to date");
        }
        Some((_, false)) => {
            ui.colored_label(theme::error(), "STALE — the input or the frames changed (still playing the old fix)");
        }
        None if !running => {
            ui.weak(if v.on_disk { "a fix for this input is on disk — load it" } else { "not rendered: showing the blend" });
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
                _ if v.on_disk => "load fix",
                _ => "render",
            };
            let can = v.has_input && !p.frames.is_empty() && (v.weights || v.on_disk) && !v.busy_elsewhere;
            let b = ui.add_enabled(can, egui::Button::new(label)).on_hover_text(
                "renders the input over its range, makes each bad frame with RIFE from its good neighbours, and plays the result here. Keyed on everything upstream and the frame list: an edit shows as STALE, nothing re-runs behind your back",
            );
            if b.clicked() {
                act = Some(Action::Analyse);
            }
            if v.busy_elsewhere {
                ui.weak("a Flow or Rifetime is running");
            }
            if matches!(v.state, ViewState::Failed(_) | ViewState::Done(_)) && ui.small_button("dismiss").clicked() {
                act = Some(Action::Dismiss);
            }
        }
    });
    act
}

/// Three frames, the middle one stitched: a frame mended from its
/// neighbours.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.circle(c, r * 0.9, fill, stroke);
    let ink = Stroke::new((r * 0.12).max(1.0), crate::theme::node_ink());
    for k in [-1.0f32, 1.0] {
        let rect = egui::Rect::from_center_size(c + Vec2::new(k * r * 0.42, 0.0), Vec2::new(r * 0.3, r * 0.5));
        painter.rect_stroke(rect, 1.0, ink, egui::StrokeKind::Middle);
    }
    let mid = egui::Rect::from_center_size(c, Vec2::new(r * 0.3, r * 0.5));
    painter.rect_filled(mid, 1.0, crate::theme::node_ink());
    painter.line_segment([c + Vec2::new(-r * 0.27, 0.0), c + Vec2::new(r * 0.27, 0.0)], ink);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_panel_fits_the_tiles_it_asks_for() {
        let span = |n: usize| crate::widgets::TILE * n as f32 + crate::widgets::TILE_GAP * (n as f32 - 1.0);
        let (w, hgt) = (span(TILES) - 14.0, span(TILE_ROWS) - 14.0 - 22.0);
        for (name, state, mode) in [
            ("framefix-panel-blend", ViewState::Idle, FrameFixMode::Blend),
            ("framefix-panel-making", ViewState::Solving(3, 95), FrameFixMode::Flow),
            ("framefix-panel-failed", ViewState::Failed("RIFE 4.6 (IFNet) is not downloaded (21.3 MB) — the node's panel has the button".into()), FrameFixMode::Flow),
        ] {
            let v = View { state, has_input: true, weights: true, loaded: Some((95, false)), on_disk: false, busy_elsewhere: false };
            let mut p = FrameFixParams { mode, frames: vec![1012, 1040, 1041, 1042], ..Default::default() };
            let over = std::cell::Cell::new(0.0f32);
            let wide = std::cell::Cell::new(0.0f32);
            let shot = crate::snapshot::shoot(w, hgt, |ui| {
                let out = egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                    let _ = panel(ui, 1, 1012, &mut p, &v);
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
