//! CameraTrack's panel and glyph. The params live in
//! `engine::nodes::camtrack`, the tracker and solver in `engine::camtrack`,
//! the runner (render, track, solve, adopt) and the viewer overlay in
//! `crate::camtrackui` on the Flow runner's types. The weights gate sits
//! above this panel.

use crate::anim::Channel;
use crate::flowui::{Action, View, ViewState};
use crate::theme;
use crate::widgets::{self, HudSlider, RowCtx};
use egui::{Color32, Painter, Pos2, Stroke, Ui, Vec2};
use engine::graph::CamTrackParams;
use engine::nodes::camtrack::LENS_PRESETS;

pub const CHANNELS: &[Channel] = &[];

pub const TILES: usize = 2;
pub const TILE_ROWS: usize = 3;

/// What the panel hands back besides a runner action.
pub enum Click {
    Run(Action),
    /// Make a Camera node reading this solve.
    MakeCamera,
    /// Make a Card on the picked points.
    MakeCard,
}

/// What the runner tells the panel beyond the Flow view: which half is
/// running, and what the solve came to.
pub struct Extra {
    /// The solving phase is the TRACKING (flow) half, not the bundle.
    pub tracking: bool,
    /// Frames, points, mean error (px), frames the solve could not place,
    /// the solved focal (mm).
    pub summary: Option<(usize, usize, f64, usize, f32)>,
}

/// The properties panel. The click it hands back is acted on after the node
/// borrow ends.
pub fn panel(
    ui: &mut Ui,
    rcx: &mut RowCtx,
    p: &mut CamTrackParams,
    v: &View,
    x: &Extra,
    hud: &mut Option<HudSlider>,
) -> Option<Click> {
    let node = rcx.node;
    let mut act = None;
    let _ = hud;
    ui.horizontal(|ui| {
        ui.label("lens");
        let shown = if p.guess {
            "guess".to_string()
        } else {
            LENS_PRESETS
                .iter()
                .find(|(_, f, a)| (*f - p.focal).abs() < 0.05 && (*a - p.aperture).abs() < 0.05)
                .map(|(n, _, _)| n.to_string())
                .unwrap_or_else(|| format!("{:.1} mm on {:.0}", p.focal, p.aperture))
        };
        egui::ComboBox::from_id_salt(("camtrack-lens", node)).width(150.0).selected_text(shown).show_ui(ui, |ui| {
            if ui.selectable_label(p.guess, "guess").on_hover_text("solve from several focal lengths, wide to long, and keep the best fit").clicked() {
                p.guess = true;
            }
            for (name, f, a) in LENS_PRESETS {
                let on = !p.guess && (*f - p.focal).abs() < 0.05 && (*a - p.aperture).abs() < 0.05;
                if ui.selectable_label(on, *name).clicked() {
                    p.guess = false;
                    p.focal = *f;
                    p.aperture = *a;
                }
            }
        });
        if !p.guess {
            ui.checkbox(&mut p.refine_focal, "refine").on_hover_text("let the solve adjust the focal length (off: held exactly)");
        }
    });
    ui.add_enabled_ui(!p.guess, |ui| {
        ui.horizontal(|ui| {
            ui.label("focal");
            let mut f = p.focal;
            widgets::num_field(ui, egui::DragValue::new(&mut f).speed(0.1).range(1.0..=500.0)).on_hover_text("focal length, mm: where the solve starts from");
            p.focal = f.clamp(1.0, 500.0);
            ui.label("mm on");
            let mut a = p.aperture;
            widgets::num_field(ui, egui::DragValue::new(&mut a).speed(0.1).range(1.0..=100.0)).on_hover_text("filmback width, mm (36 = full frame; a phone's lens in 35 mm-equivalent terms goes on 36)");
            p.aperture = a.clamp(1.0, 100.0);
            ui.label("mm");
        });
    });
    ui.horizontal(|ui| {
        ui.label("detail");
        egui::ComboBox::from_id_salt(("camtrack-detail", node))
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
            .on_hover_text("track at this fraction of the size; changing it re-tracks");
        ui.label("scale");
        let mut s = p.scale;
        widgets::num_field(ui, egui::DragValue::new(&mut s).speed(0.01).range(0.001..=1000.0)).on_hover_text("scene units per solve unit (the median point depth from the first frame is 1)");
        p.scale = s.max(1e-3);
        ui.checkbox(&mut p.show, "points").on_hover_text("draw the solved points over the viewer");
    });
    // The picks and what they can set.
    if p.solved() {
        let n = p.picks().len();
        ui.horizontal(|ui| {
            ui.label(format!("{n} point{} picked", if n == 1 { "" } else { "s" }))
                .on_hover_text("S-click a solved point over the viewer to pick it");
            if ui.add_enabled(n >= 3, egui::Button::new("ground")).on_hover_text("fit a plane to the picks and make it the ground (Y up), origin on their centre").clicked() {
                p.set_ground();
            }
            if ui.add_enabled(n >= 1, egui::Button::new("origin")).on_hover_text("put the origin on the picks' centre").clicked() {
                p.set_origin();
            }
            if ui.add_enabled(n >= 1, egui::Button::new("card…")).on_hover_text("a Card on the picks: on their plane (three or more) or facing the camera, sized to cover them").clicked() {
                act = Some(Click::MakeCard);
            }
            if n > 0 && ui.small_button("clear").clicked() {
                p.picked.clear();
            }
        });
        ui.horizontal(|ui| {
            ui.label("two picks are");
            let mut d = p.ruler;
            widgets::num_field(ui, egui::DragValue::new(&mut d).speed(0.01).range(0.001..=10000.0));
            p.ruler = d.max(1e-3);
            ui.label("units apart");
            if ui.add_enabled(n == 2, egui::Button::new("set scale")).on_hover_text("scale the solve so the two picked points are this far apart").clicked() {
                p.set_distance();
            }
            let aligned = p.align_rot != [0.0; 3] || p.align_off != [0.0; 3];
            if aligned && ui.small_button("reset align").on_hover_text("back to the solve's own frame: the first camera at the origin").clicked() {
                p.align_rot = [0.0; 3];
                p.align_off = [0.0; 3];
            }
        });
    }
    let running = matches!(v.state, ViewState::Rendering(..) | ViewState::Solving(..));
    match &v.state {
        ViewState::Idle => {}
        ViewState::Rendering(done, total) => {
            let frac = if *total > 0 { *done as f32 / *total as f32 } else { 0.0 };
            ui.add(egui::ProgressBar::new(frac).text(format!("rendering {done}/{total}")));
        }
        ViewState::Solving(done, total) => {
            let frac = if *total > 0 { *done as f32 / *total as f32 } else { 0.0 };
            let what = if x.tracking { "tracking" } else { "solving" };
            let words = if *total > 0 { format!("{what} {done}/{total}") } else { "loading the network…".into() };
            ui.add(egui::ProgressBar::new(frac).text(words));
        }
        ViewState::Failed(e) => {
            ui.colored_label(theme::error(), format!("failed: {e}"));
        }
        ViewState::Done(s) => {
            ui.weak(s);
        }
    }
    match (v.loaded, &x.summary) {
        (Some((_, fresh)), Some((n, pts, err, lost, focal))) => {
            let line = format!("{n} frames, {pts} points, {err:.2} px, {focal:.1} mm");
            if fresh {
                ui.monospace(format!("solved — up to date  ({line})"));
            } else {
                ui.colored_label(theme::error(), format!("STALE — the input, range, detail or lens changed ({line})"));
            }
            if *lost > 0 {
                ui.colored_label(theme::error(), format!("{lost} frame{} could not be placed (held from a neighbour)", if *lost == 1 { "" } else { "s" }));
            }
        }
        (None, _) if !running => {
            ui.weak("not solved: the picture passes through");
        }
        _ => {}
    }
    ui.horizontal(|ui| {
        if running {
            if ui.button("cancel").clicked() {
                act = Some(Click::Run(Action::Cancel));
            }
        } else {
            let label = match v.loaded {
                Some(_) => "re-solve",
                None if v.on_disk => "load solve",
                None => "solve",
            };
            let can = v.has_input && (v.weights || v.on_disk) && !v.busy_elsewhere;
            let b = ui.add_enabled(can, egui::Button::new(label)).on_hover_text(
                "renders the input (and the matte) over its range, tracks a grid of features along the flow on the GPU, and fits a 3D camera and a point cloud to them. Keyed on everything upstream: an edit shows as STALE, nothing re-runs behind your back. A lens change re-solves without re-tracking",
            );
            if b.clicked() {
                act = Some(Click::Run(Action::Analyse));
            }
            if v.loaded.is_some()
                && ui
                    .button("camera…")
                    .on_hover_text("a Camera node that reads this solve, per frame (bake it to keys from its panel)")
                    .clicked()
            {
                act = Some(Click::MakeCamera);
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

/// A camera body with a scatter of solved points in front of it.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let body = egui::Rect::from_center_size(c + Vec2::new(-r * 0.35, r * 0.25), Vec2::new(r * 0.8, r * 0.6));
    painter.rect(body, r * 0.1, fill, stroke, egui::StrokeKind::Middle);
    let lens = vec![
        Pos2::new(body.right(), body.center().y - r * 0.14),
        Pos2::new(body.right() + r * 0.35, body.center().y - r * 0.32),
        Pos2::new(body.right() + r * 0.35, body.center().y + r * 0.32),
        Pos2::new(body.right(), body.center().y + r * 0.14),
    ];
    painter.add(egui::Shape::convex_polygon(lens, fill, stroke));
    let ink = crate::theme::node_ink();
    for (dx, dy) in [(0.15, -0.75), (0.55, -0.55), (0.8, -0.2), (0.35, -0.35), (0.75, -0.8), (0.1, -0.45)] {
        painter.circle_filled(c + Vec2::new(r * dx, r * dy), (r * 0.08).max(1.0), ink);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_panel_fits_the_tiles_it_asks_for() {
        let span = |n: usize| crate::widgets::TILE * n as f32 + crate::widgets::TILE_GAP * (n as f32 - 1.0);
        let (w, hgt) = (span(TILES) - 14.0, span(TILE_ROWS) - 14.0 - 22.0);
        for (name, state, summary) in [
            ("camtrack-panel-failed", ViewState::Failed("SEA-RAFT M is not downloaded (78.8 MB)".into()), None),
            ("camtrack-panel-stale", ViewState::Done("done".into()), Some((48, 612, 0.31, 2, 31.2))),
        ] {
            let v = View { state, has_input: true, weights: true, loaded: Some((48, false)), on_disk: false, busy_elsewhere: false };
            let x = Extra { tracking: false, summary };
            // Solved, so the pick rows are on the panel too.
            let solve = engine::camtrack::Solve {
                size: [960, 540],
                focal_px: 825.0,
                poses: vec![engine::camtrack::Pose::IDENTITY; 2],
                unplaced: vec![],
                points: vec![[0.0, 0.0, 1.0]; 3],
                point_track: vec![0, 1, 2],
                errors: vec![0.1; 2],
            };
            let mut p = CamTrackParams {
                solve: engine::graph::CamSolveRef::new(engine::graph::CamSolve { first: 0, solve, recipe: 1 }),
                picked: vec![0, 1],
                picked_for: 1,
                ..Default::default()
            };
            let mut anim = Default::default();
            let ekind = engine::graph::NodeKind::CamTrack(p.clone());
            let over = std::cell::Cell::new(0.0f32);
            let wide = std::cell::Cell::new(0.0f32);
            let shot = crate::snapshot::shoot(w, hgt, |ui| {
                let out = egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                    let mut rcx = RowCtx { node: 1, anim: &mut anim, ekind: &ekind, t: 0, hud_live: false, driven: Vec::new() };
                    let mut hud = None;
                    let _ = panel(ui, &mut rcx, &mut p, &v, &x, &mut hud);
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
