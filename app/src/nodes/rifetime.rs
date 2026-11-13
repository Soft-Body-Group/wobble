//! Rifetime's panel and glyph. The params live in `engine::nodes::rifetime`,
//! the in-between maker in `engine::retime`; the runner (render, make,
//! adopt) is shared with the Flow node (`crate::flowui`). The weights gate
//! sits above this panel.

use crate::anim::Channel;
use crate::flowui::{Action, View, ViewState};
use crate::theme;
use crate::widgets::{self, RowCtx};
use egui::{Color32, Painter, Pos2, Stroke, Ui, Vec2};
use engine::graph::{FlowDetail, RifetimeMode, RifetimeParams};
use engine::nodes::rifetime::{MAX_SPEED, MIN_SPEED};

pub const TILES: usize = 2;
pub const TILE_ROWS: usize = 1;

pub const CHANNELS: &[Channel] = &[Channel::RifetimeFrame];

pub fn get(p: &RifetimeParams, ch: Channel) -> Option<f32> {
    (ch == Channel::RifetimeFrame).then_some(p.frame)
}

pub fn set(p: &mut RifetimeParams, ch: Channel, v: f32) {
    if ch == Channel::RifetimeFrame {
        p.frame = v;
    }
}

pub fn label(ch: Channel) -> Option<&'static str> {
    (ch == Channel::RifetimeFrame).then_some("rifetime source frame")
}

pub fn panel(ui: &mut Ui, rcx: &mut RowCtx, node: u64, p: &mut RifetimeParams, v: &View) -> Option<Action> {
    let mut act = None;
    ui.horizontal(|ui| {
        ui.label("time");
        for m in RifetimeMode::ALL {
            ui.selectable_value(&mut p.mode, m, m.label());
        }
    });
    match p.mode {
        RifetimeMode::Speed => {
            widgets::plain_slider(ui, "speed", &mut p.speed, MIN_SPEED, MAX_SPEED, false)
                .on_hover_text("source frames per output frame: 0.5 is half speed, twice as long");
            p.speed = p.speed();
        }
        RifetimeMode::Curve => {
            let _ = widgets::drags(ui, rcx, "source frame", &mut [&mut p.frame], 0.25, (0.0, 100.0), &[Channel::RifetimeFrame]);
            let keyed = rcx.anim.get(&Channel::RifetimeFrame).is_some_and(|c| c.keys.len() > 1);
            ui.weak(if keyed {
                "the curve is the retime: shape it in C, then render"
            } else {
                "key it twice: the curve between the keys is the retime"
            });
        }
    }
    ui.horizontal(|ui| {
        ui.label("detail");
        egui::ComboBox::from_id_salt(("retime-detail", node))
            .width(90.0)
            .selected_text(p.detail.label())
            .show_ui(ui, |ui| {
                for d in FlowDetail::ALL {
                    ui.selectable_value(&mut p.detail, d, d.label());
                }
            })
            .response
            .on_hover_text("judge the motion at this fraction of the size (the frames are always made at full size): half suits 4K");
        if p.mode == RifetimeMode::Speed {
            ui.weak(format!("{:.2}x as long", 1.0 / p.speed()));
        }
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
            let words = if *total > 0 { format!("making frames {done}/{total}") } else { "loading the network…".into() };
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
            ui.monospace(format!("retimed — up to date  ({n} frames)"));
        }
        Some((_, false)) => {
            ui.colored_label(theme::error(), "STALE — the input or a setting changed (still playing the old frames)");
        }
        None if !running => {
            ui.weak(if v.on_disk { "frames for this input are on disk — load them" } else { "not rendered: passing the input through" });
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
                _ if v.on_disk => "load frames",
                _ => "render",
            };
            let can = v.has_input && (v.weights || v.on_disk) && !v.busy_elsewhere;
            let b = ui.add_enabled(can, egui::Button::new(label)).on_hover_text(
                "renders the input over its range, makes every in-between frame with RIFE on the GPU, and plays the retimed shot here. Keyed on everything upstream: an edit shows as STALE, nothing re-runs behind your back",
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

/// A clock face with three frames fanned out of it: time, stretched.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.circle(c, r * 0.9, fill, stroke);
    let ink = Stroke::new((r * 0.12).max(1.0), crate::theme::node_ink());
    for k in 0..3 {
        let x = c.x - r * 0.45 + k as f32 * r * 0.3;
        let rect = egui::Rect::from_center_size(Pos2::new(x, c.y + r * 0.1), Vec2::new(r * 0.36, r * 0.5));
        painter.rect_stroke(rect, 1.0, ink, egui::StrokeKind::Middle);
    }
    painter.line_segment([c + Vec2::new(-r * 0.5, -r * 0.42), c + Vec2::new(r * 0.5, -r * 0.42)], ink);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_panel_fits_the_tiles_it_asks_for() {
        let span = |n: usize| crate::widgets::TILE * n as f32 + crate::widgets::TILE_GAP * (n as f32 - 1.0);
        let (w, hgt) = (span(TILES) - 14.0, span(TILE_ROWS) - 14.0 - 22.0);
        for (name, state) in [
            ("retime-panel-failed", ViewState::Failed("RIFE 4.6 (IFNet) is not downloaded (21.3 MB) — the node's panel has the button".into())),
            ("retime-panel-making", ViewState::Solving(7, 95)),
        ] {
            let v = View { state, has_input: true, weights: true, loaded: Some((95, false)), on_disk: false, busy_elsewhere: false };
            let mut p = RifetimeParams { mode: RifetimeMode::Curve, ..RifetimeParams::new(0) };
            let mut anim = Default::default();
            let ekind = engine::graph::NodeKind::Rifetime(p.clone());
            let over = std::cell::Cell::new(0.0f32);
            let wide = std::cell::Cell::new(0.0f32);
            let shot = crate::snapshot::shoot(w, hgt, |ui| {
                let out = egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                    let mut rcx = RowCtx { node: 1, anim: &mut anim, ekind: &ekind, t: 0, hud_live: false, driven: Vec::new() };
                    let _ = panel(ui, &mut rcx, 1, &mut p, &v);
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
