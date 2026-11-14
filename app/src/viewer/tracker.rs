//! Tracker: the pattern box, the search box and the tracked position.
//! Moved out of `viewer_ui` (lib.rs) unchanged.

use crate::*;

impl Gfx {
    /// Tracker: the pattern box, the search box and the tracked position.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gizmo_tracker(
        &mut self,
        ui: &mut egui::Ui,
        cur_t: u64,
        owf: f32,
        ohf: f32,
        pointer_pos: Option<egui::Pos2>,
        out_to_pts: &impl Fn(egui::Vec2) -> egui::Pos2,
        pts_to_out: &impl Fn(egui::Pos2) -> egui::Vec2,
        selected: &Vec<u64>,
        ink: &mut ink::InkShapes,
    ) {
        // Tracker: the pattern box (drag to place, corner to resize), the
        // search box it may wander inside, and the tracked position at
        // this frame.
        for node in selected.iter().copied() {
            ink.set_tint(self.hud_ink(node));
            let hd = downstream_proj(
                &self.egraph,
                node,
                &|id| self.node_res_f32(id).unwrap_or((owf, ohf)),
                cur_t,
            );
            let hd_inv = engine::codegen::mat3_inv(&hd);
            let Some(ekind) = self.live_kind(node, cur_t) else {
                continue;
            };
            let NodeKind::Tracker(tkw) = ekind else { continue };
            let to_scr =
                |q: [f32; 2]| out_to_pts(apply_h(&hd, egui::vec2(q[0] * owf, q[1] * ohf)));
            let from_screen = |p: egui::Pos2| apply_h(&hd_inv, pts_to_out(p));
            let boxed = |ink: &mut ink::InkShapes,
                         c: [f32; 2],
                         h: [f32; 2],
                         w: f32,
                         alpha: f32| {
                let cs = [
                    to_scr([c[0] - h[0], c[1] - h[1]]),
                    to_scr([c[0] + h[0], c[1] - h[1]]),
                    to_scr([c[0] + h[0], c[1] + h[1]]),
                    to_scr([c[0] - h[0], c[1] + h[1]]),
                ];
                for k in 0..4 {
                    ink.seg(cs[k], cs[(k + 1) % 4], w, alpha, false);
                }
            };
            let count = tkw.count();
            // Each point's boxes ride WITH its track: once the bake has
            // keyed it they sit on the tracked point at this frame
            // (the usual tracker convention), else on the seed.
            let tracked: Vec<bool> = (0..count)
                .map(|i| {
                    self.egraph
                        .node(node)
                        .and_then(|n| n.anim.get(&tracker::chans(i)[0]))
                        .is_some_and(|c| !c.keys.is_empty())
                })
                .collect();
            let anchor = |i: usize| -> [f32; 2] {
                let p = tkw.point(i).unwrap_or(engine::graph::TrackPoint { center: [0.5; 2], size: [0.05; 2], search: 0.08, track: [0.5; 2] });
                if tracked[i] { p.track } else { p.center }
            };
            let sel = innards::selected(ui.ctx(), node, count);
            // The solved motion, once there is one: the picture's frame
            // carried by it, faint — the corner pin a cable would drive.
            if count > 1 && !tkw.anchors.is_empty() {
                let q = tkw.pins();
                for k in 0..4 {
                    ink.seg(to_scr(q[k]), to_scr(q[(k + 1) % 4]), 0.8, 0.25, false);
                }
            }
            for i in 0..count {
                let Some(p) = tkw.point(i) else { continue };
                let c = anchor(i);
                let on = i == sel;
                boxed(ink, c, p.size, if on { 1.2 } else { 0.9 }, if on { 0.85 } else { 0.4 });
                if on {
                    boxed(ink, c, [p.size[0] + p.search, p.size[1] + p.search], 0.8, 0.3);
                }
                let tp = to_scr(c);
                if tracked[i] {
                    ink.circle_stroke(tp, 6.0, 1.2, if on { 0.9 } else { 0.5 }, false);
                    ink.seg(tp - egui::vec2(9.0, 0.0), tp + egui::vec2(9.0, 0.0), 1.0, 0.7, false);
                    ink.seg(tp - egui::vec2(0.0, 9.0), tp + egui::vec2(0.0, 9.0), 1.0, 0.7, false);
                }
                // Another point's centre: a click makes it the one edited.
                if !on {
                    let pick = ui.interact(
                        egui::Rect::from_center_size(tp, egui::vec2(20.0, 20.0)),
                        ui.id().with(("tk-pick", node, i)),
                        egui::Sense::click(),
                    );
                    ink.circle_fill(tp, 2.5, 0.6, pick.hovered());
                    if pick.clicked() {
                        innards::select(ui.ctx(), node, i);
                    }
                }
            }
            // The selected point's handles.
            let Some(p0) = tkw.point(sel) else { continue };
            let mut p = p0;
            let c0 = anchor(sel);
            let cp = to_scr(c0);
            let mv = ui.interact(
                egui::Rect::from_center_size(cp, egui::vec2(24.0, 24.0)),
                ui.id().with(("tk-move", node)),
                egui::Sense::drag(),
            );
            ink.circle_fill(cp, 3.5, 0.9, mv.hovered());
            let mut moved: Option<[f32; 2]> = None;
            if mv.dragged() && !mv.drag_started() {
                if let Some(ptr) = pointer_pos {
                    let d = from_screen(ptr) - from_screen(ptr - widgets::rated_delta(ui, &mv));
                    // Moving a tracked box corrects the track at this
                    // frame (a key, like dragging a tracker) and
                    // re-seeds there, so "track" continues from where
                    // the box now sits.
                    let m = [c0[0] + d.x / owf, c0[1] + d.y / ohf];
                    p.center = m;
                    moved = Some(m);
                }
            }
            let corner = to_scr([c0[0] + p.size[0], c0[1] + p.size[1]]);
            let sz = ui.interact(
                egui::Rect::from_center_size(corner, egui::vec2(16.0, 16.0)),
                ui.id().with(("tk-size", node)),
                egui::Sense::drag(),
            );
            ink.circle_stroke(corner, 4.0, 1.2, 0.9, sz.hovered());
            if sz.dragged() && !sz.drag_started() {
                if let Some(ptr) = pointer_pos {
                    let d = from_screen(ptr) - from_screen(ptr - widgets::rated_delta(ui, &sz));
                    p.size[0] = (p.size[0] + d.x / owf).clamp(0.002, 0.5);
                    p.size[1] = (p.size[1] + d.y / ohf).clamp(0.002, 0.5);
                }
            }
            // The region is authored setup, not animation: it says where
            // to START tracking, so it never keys. The TRACK does, and a
            // drag of a tracked box writes it through the keyed gate
            // (a locked channel stays put). Only a MOVE keys.
            if p != p0 {
                let patch_driven: Vec<(u64, anim::Channel)> = self
                    .egraph
                    .patch
                    .links
                    .iter()
                    .filter_map(|l| match l.to {
                        editor::PDst::Chan(n, ch) => Some((n, ch)),
                        _ => None,
                    })
                    .collect();
                if let Some(editor::ENode { kind, anim, .. }) = self.egraph.node_mut(node) {
                    let before = kind.clone();
                    if let NodeKind::Tracker(auth) = kind {
                        if let Some(mut a) = auth.point(sel) {
                            a.center = p.center;
                            a.size = p.size;
                            auth.set_point(sel, a);
                        }
                    }
                    if let (Some(m), true) = (moved, tracked[sel]) {
                        let mut after = before.clone();
                        if let NodeKind::Tracker(t) = &mut after {
                            if let Some(mut a) = t.point(sel) {
                                a.track = m;
                                t.set_point(sel, a);
                            }
                        }
                        keyed_write_back(
                            anim,
                            kind,
                            &before,
                            &after,
                            cur_t,
                            &tracker::chans(sel),
                            true,
                            &patch_driven,
                            node,
                        );
                    }
                }
            }
        }
    }
}
