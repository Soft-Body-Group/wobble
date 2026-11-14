//! Tracker: the points, the solve and the track buttons.
//! Moved out of `properties_ui` (lib.rs) unchanged.

use crate::*;

/// Tracker: the points, the solve and the track buttons.
pub(crate) fn panel(
    ui: &mut egui::Ui,
    rcx: &mut widgets::RowCtx,
    tk: &mut engine::graph::TrackerParams,
    cur_t: u64,
    node: u64,
    track_state: Option<(u64, u64, u64, u64)>,
    track_click: &mut Option<TrackAction>,
) {
    let running = track_state.is_some();
    let count = tk.count();
    let sel = innards::selected(ui.ctx(), node, count);
    // The points: the same list the node shows when you
    // zoom into it. Click to edit one, + to add, alt-click
    // to remove.
    ui.horizontal(|ui| {
        ui.label("points");
        for i in 0..count {
            let r = ui
                .add(egui::Button::selectable(i == sel, format!("{}", i + 1)))
                .on_hover_text("edit this point · alt-click removes it");
            if r.clicked() {
                if ui.input(|inp| inp.modifiers.alt) && count > 1 {
                    *track_click = Some(TrackAction::Remove(i));
                } else {
                    innards::select(ui.ctx(), node, i);
                }
            }
        }
        if count < engine::graph::TRACK_POINTS_MAX
            && ui.button("+").on_hover_text("add a point beside the last").clicked()
        {
            *track_click = Some(TrackAction::Add);
        }
        ui.weak(engine::pointsolve::model_label(count));
    });
    if let Some(mut pt) = tk.point(sel) {
        ui.horizontal(|ui| {
            ui.label(format!("{} region", sel + 1));
            for k in 0..2 {
                crate::widgets::num_field(ui, egui::DragValue::new(&mut pt.size[k]).update_while_editing(false)
                        .speed(widgets::rated(ui, 0.002))
                        .range(0.002..=0.5),
                );
            }
        });
        widgets::plain_slider(ui, "search", &mut pt.search, 0.005, 0.4, false)
            .on_hover_text("how far the point may move between frames");
        if Some(pt) != tk.point(sel) {
            tk.set_point(sel, pt);
        }
    }
    ui.horizontal(|ui| {
        use engine::graph::TrackMode;
        use engine::planar::Model;
        ui.selectable_value(&mut tk.mode, TrackMode::Pattern, "pattern")
            .on_hover_text(
                "align the box around the point into every frame: \
                 subpixel, any texture, coasts through occlusion",
            );
        ui.selectable_value(&mut tk.mode, TrackMode::Blob, "blob")
            .on_hover_text(
                "centroid of what clears a threshold: a bright or \
                 dark blob on a plain ground, whole pixels",
            );
        if tk.mode == TrackMode::Pattern {
            ui.add_space(8.0);
            for (m, label, tip) in [
                (Model::Translation, "move", "position only: stiffest, for a pattern that never turns"),
                (Model::Similarity, "+turn", "also rotation and scale: holds a pattern that turns or grows"),
            ] {
                ui.selectable_value(&mut tk.model, m, label).on_hover_text(tip);
            }
        }
    });
    if tk.mode == engine::graph::TrackMode::Blob {
        widgets::plain_slider(ui, "threshold", &mut tk.threshold, 0.0, 2.0, false);
        ui.checkbox(&mut tk.invert, "dark blob")
            .on_hover_text("track dark features on a light background");
    }
    let mut clear_track = false;
    ui.horizontal(|ui| {
        if running {
            if ui.button("cancel").clicked() {
                *track_click = Some(TrackAction::Cancel);
            }
        } else if ui
            .button("track")
            .on_hover_text(format!(
                "an explicit bake: decodes the input sequence once \
                 from the current frame ({cur_t}) to the end and \
                 keys every point per frame",
            ))
            .clicked()
        {
            *track_click = Some(TrackAction::Start(tk.clone()));
        }
        clear_track = ui
            .button("clear")
            .on_hover_text("drop every point's tracked keyframes")
            .clicked();
    });
    if clear_track {
        for i in 0..count {
            for c in tracker::chans(i) {
                rcx.anim.remove(&c);
            }
        }
        tk.anchors.clear();
    }
    if let Some((done, total, lost, start)) = track_state {
        let frac =
            if total > 0 { done as f32 / total as f32 } else { 0.0 };
        ui.add(egui::ProgressBar::new(frac).text(format!(
            "tracking {done}/{total} from frame {start}"
        )));
        if lost > 0 {
            ui.colored_label(
                theme::warn(),
                if tk.mode == engine::graph::TrackMode::Pattern {
                    format!("{lost} frames with a point lost (coasted, then held)")
                } else {
                    format!("{lost} frames with a point lost (held the last position)")
                },
            );
        }
    }
    let keys = rcx
        .anim
        .get(&tracker::chans(sel)[0])
        .map(|c| c.keys.len())
        .unwrap_or(0);
    if keys > 0 {
        let t = tk.point(sel).map(|p| p.track).unwrap_or_default();
        ui.monospace(format!(
            "point {} tracked {keys} frames  ({:.4}, {:.4}) now",
            sel + 1,
            t[0],
            t[1]
        ));
    } else {
        ui.weak(format!("point {} not tracked", sel + 1));
    }
    // The solve, cabled into a pin: 1 point moves it,
    // 2-3 (or 5+) move, turn and scale it, 4 corner-pin it.
    ui.horizontal(|ui| {
        ui.label("drive");
        if ui
            .button("Transform")
            .on_hover_text("cable the solved pin into the selected Transform (or a new one)")
            .clicked()
        {
            *track_click = Some(TrackAction::Drive(planarui::DriveTarget::Transform));
        }
        if ui
            .button("CornerPin")
            .on_hover_text("cable the solved pin into the selected CornerPin (or a new one)")
            .clicked()
        {
            *track_click = Some(TrackAction::Drive(planarui::DriveTarget::CornerPin));
        }
        if ui
            .button("stabilise")
            .on_hover_text(
                "cable the solved pin into the FROM corners of the selected CornerPin \
                 (or a new one): wire the plate into it and the points hold still",
            )
            .clicked()
        {
            *track_click = Some(TrackAction::Drive(planarui::DriveTarget::Stabilize));
        }
    });
    ui.weak(
        "passes the image through; zoom into the node to see its points",
    );
}
