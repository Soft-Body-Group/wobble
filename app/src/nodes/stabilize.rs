//! Stabilize's panel and glyph. The params and the path maths live in
//! `engine::nodes::stabilize`, the analysis in `engine::stabilize`, the
//! runner (render, solve, adopt) in `crate::stabilizeui` on the Flow
//! runner's types. The weights gate sits above this panel.

use crate::anim::Channel;
use crate::flowui::{Action, View, ViewState};
use crate::theme;
use crate::widgets::{self, HudSlider, RowCtx};
use egui::{Color32, Painter, Pos2, Stroke, Ui, Vec2};
use engine::graph::{StabMode, StabModel, StabilizeParams};

pub const CHANNELS: &[Channel] = &[Channel::StabSmooth, Channel::StabZoom];

pub const TILES: usize = 2;
pub const TILE_ROWS: usize = 2;

pub fn get(p: &StabilizeParams, ch: Channel) -> Option<f32> {
    match ch {
        Channel::StabSmooth => Some(p.smooth),
        Channel::StabZoom => Some(p.zoom),
        _ => None,
    }
}

pub fn set(p: &mut StabilizeParams, ch: Channel, v: f32) {
    match ch {
        Channel::StabSmooth => p.smooth = v.max(0.0),
        Channel::StabZoom => p.zoom = v.clamp(0.1, 10.0),
        _ => {}
    }
}

pub fn label(ch: Channel) -> Option<&'static str> {
    match ch {
        Channel::StabSmooth => Some("stabilize smoothing"),
        Channel::StabZoom => Some("stabilize zoom"),
        _ => None,
    }
}

/// What the panel hands back besides a runner action.
pub enum Click {
    Run(Action),
}

/// The properties panel. The click it hands back is acted on after the node
/// borrow ends.
pub fn panel(
    ui: &mut Ui,
    rcx: &mut RowCtx,
    p: &mut StabilizeParams,
    v: &View,
    hud: &mut Option<HudSlider>,
) -> Option<Click> {
    let node = rcx.node;
    let mut act = None;
    ui.horizontal(|ui| {
        for m in StabMode::ALL {
            ui.selectable_value(&mut p.mode, m, m.label()).on_hover_text(match m {
                StabMode::LockOff => "every frame warped onto the reference frame: a locked-off camera",
                StabMode::Smooth => "the camera path smoothed: the move stays, the shake goes",
            });
        }
        ui.checkbox(&mut p.reverse, "reverse")
            .on_hover_text("put the shake BACK: exactly undoes this node with the same settings (match CG to the plate, or un-stabilise after the work)");
    });
    ui.horizontal(|ui| {
        ui.label("remove");
        for (k, name) in ["x", "y", "rotate", "scale"].iter().enumerate() {
            ui.checkbox(&mut p.axes[k], *name);
        }
    });
    if p.mode == StabMode::Smooth {
        if let Some(h) = widgets::slider(ui, rcx, "smoothing", &mut p.smooth, 0.0, 100.0, &[Channel::StabSmooth]) {
            *hud = Some(h);
        }
    } else {
        ui.horizontal(|ui| {
            let r = p.reference.or(p.path.span().map(|s| s.0));
            ui.label(match r {
                Some(f) => format!("reference frame {f}"),
                None => "reference: the first analysed frame".into(),
            });
            if ui.small_button("set to playhead").clicked() {
                p.reference = Some(rcx.t as i64);
            }
            if p.reference.is_some() && ui.small_button("first").clicked() {
                p.reference = None;
            }
        });
    }
    if let Some(h) = widgets::slider(ui, rcx, "zoom", &mut p.zoom, 1.0, 1.5, &[Channel::StabZoom]) {
        *hud = Some(h);
    }
    ui.horizontal(|ui| {
        ui.label("fit");
        egui::ComboBox::from_id_salt(("stab-model", node))
            .width(90.0)
            .selected_text(p.model.label())
            .show_ui(ui, |ui| {
                for m in StabModel::ALL {
                    ui.selectable_value(&mut p.model, m, m.label());
                }
            })
            .response
            .on_hover_text("the motion fitted between neighbouring frames; changing it needs a re-analysis");
        ui.label("detail");
        egui::ComboBox::from_id_salt(("stab-detail", node))
            .width(70.0)
            .selected_text(match p.divisor {
                1 => "full",
                4 => "quarter",
                _ => "half",
            })
            .show_ui(ui, |ui| {
                for (d, name) in [(1u8, "full"), (2, "half"), (4, "quarter")] {
                    ui.selectable_value(&mut p.divisor, d, name);
                }
            })
            .response
            .on_hover_text("analyse at this fraction of the size: a camera's motion needs little detail");
    });
    let running = matches!(v.state, ViewState::Rendering(..) | ViewState::Solving(..));
    match &v.state {
        ViewState::Idle => {}
        ViewState::Rendering(done, total) => {
            let frac = if *total > 0 { *done as f32 / *total as f32 } else { 0.0 };
            ui.add(egui::ProgressBar::new(frac).text(format!("rendering {done}/{total}")));
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
            ui.colored_label(theme::error(), "STALE — the input or the fit changed (still using the old path)");
        }
        None if !running => {
            ui.weak("not analysed: the picture passes through");
        }
        None => {}
    }
    ui.horizontal(|ui| {
        if running {
            if ui.button("cancel").clicked() {
                act = Some(Click::Run(Action::Cancel));
            }
        } else {
            let label = match v.loaded {
                Some(_) => "re-analyse",
                None if v.on_disk => "load analysis",
                None => "analyse",
            };
            let can = v.has_input && (v.weights || v.on_disk) && !v.busy_elsewhere;
            let b = ui.add_enabled(can, egui::Button::new(label)).on_hover_text(
                "renders the input (and the matte) over its range, solves the motion between every pair of frames on the GPU and fits the camera to it. Keyed on everything upstream: an edit shows as STALE, nothing re-runs behind your back",
            );
            if b.clicked() {
                act = Some(Click::Run(Action::Analyse));
            }
            if v.busy_elsewhere {
                ui.weak("another analysis is running");
            }
            if matches!(v.state, ViewState::Failed(_) | ViewState::Done(_)) && ui.small_button("dismiss").clicked() {
                act = Some(Click::Run(Action::Dismiss));
            }
        }
    });
    act
}

/// A frame held level inside a tilted, jittered one.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let (s, co) = (0.25f32).sin_cos();
    let rot = |v: Vec2| Vec2::new(v.x * co - v.y * s, v.x * s + v.y * co);
    let outer: Vec<Pos2> = [Vec2::new(-1.0, -0.7), Vec2::new(1.0, -0.7), Vec2::new(1.0, 0.7), Vec2::new(-1.0, 0.7)]
        .iter()
        .map(|v| c + rot(*v * r * 0.85))
        .collect();
    painter.add(egui::Shape::convex_polygon(outer, fill, stroke));
    let inner = egui::Rect::from_center_size(c, Vec2::new(r * 1.0, r * 0.62));
    painter.rect_stroke(inner, 0.0, Stroke::new((r * 0.12).max(1.0), crate::theme::node_ink()), egui::StrokeKind::Middle);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_panel_fits_the_tiles_it_asks_for() {
        let span = |n: usize| crate::widgets::TILE * n as f32 + crate::widgets::TILE_GAP * (n as f32 - 1.0);
        let (w, hgt) = (span(TILES) - 14.0, span(TILE_ROWS) - 14.0 - 22.0);
        for (name, mode, state) in [
            ("stab-panel-failed", StabMode::Smooth, ViewState::Failed("SEA-RAFT M is not downloaded (78.8 MB)".into())),
            ("stab-panel-solving", StabMode::LockOff, ViewState::Solving(7, 24)),
        ] {
            let v = View { state, has_input: true, weights: true, loaded: Some((24, false)), on_disk: false, busy_elsewhere: false };
            let mut p = StabilizeParams { mode, ..Default::default() };
            let mut anim = Default::default();
            let ekind = engine::graph::NodeKind::Stabilize(p.clone());
            let over = std::cell::Cell::new(0.0f32);
            let wide = std::cell::Cell::new(0.0f32);
            let shot = crate::snapshot::shoot(w, hgt, |ui| {
                let out = egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                    let mut rcx = RowCtx { node: 1, anim: &mut anim, ekind: &ekind, t: 0, hud_live: false, driven: Vec::new() };
                    let mut hud = None;
                    let _ = panel(ui, &mut rcx, &mut p, &v, &mut hud);
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
