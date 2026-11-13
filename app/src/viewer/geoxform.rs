//! 2dini Transform: the node's own manipulator inside the dive.
//! Moved out of `viewer_ui` (lib.rs) unchanged.

use crate::*;

impl Gfx {
    /// 2dini Transform: the node's own manipulator inside the dive.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gizmo_geo_transform(
        &mut self,
        ui: &mut egui::Ui,
        cur_t: u64,
        hud_hide: bool,
        owf: f32,
        ohf: f32,
        pointer_pos: Option<egui::Pos2>,
        out_to_pts: &impl Fn(egui::Vec2) -> egui::Pos2,
        pts_to_out: &impl Fn(egui::Pos2) -> egui::Vec2,
        selected: &Vec<u64>,
        geo_handle_hot: &mut bool,
        ink: &mut ink::InkShapes,
    ) {
        if self.dive.last().is_some_and(|d| d.kind == DiveKind::Geo) && !hud_hide && self.geo_tool.is_none() {
            for node in selected.iter().copied() {
                let Some(NodeKind::GeoNode(gk)) = self.live_kind(node, cur_t) else { continue };
                // An inheriting Transform turns about what it is fed.
                let base = match &gk {
                    engine::geo::GeoNode::Transform { inherit: true, .. } => {
                        match self.egraph.input_of(node, 0) {
                            Some(src) => {
                                self.geo_hud_refresh(&[src], owf, ohf, cur_t);
                                self.geo_hud_cache
                                    .as_ref()
                                    .and_then(|c| c.geos.iter().find(|(n, _)| *n == src))
                                    .and_then(|(_, g)| g.centre())
                                    .unwrap_or([0.0, 0.0])
                            }
                            None => [0.0, 0.0],
                        }
                    }
                    _ => [0.0, 0.0],
                };
                let Some(fr) = geo_frame(&gk, base) else { continue };
                let g2s = |p: [f32; 2]| out_to_pts(egui::vec2(p[0] * owf, p[1] * owf));
                let s2g = |p: egui::Pos2| {
                    let q = pts_to_out(p);
                    [q.x / owf, q.y / owf]
                };
                let c = g2s(fr.centre);
                let ax = egui::vec2(fr.rot.cos(), fr.rot.sin());
                let ay = egui::vec2(-fr.rot.sin(), fr.rot.cos());
                const H: f32 = 45.0;
                const RING: f32 = 70.0;
                let hx = c + ax * H;
                let hy = c + ay * H;
                let hu = c + (ax + ay) * H;
                let hr = c + ax * RING;
                let mut handles = vec![(0u8, c), (2, hx), (3, hy), (4, hu)];
                if fr.turns {
                    handles.insert(1, (1, hr));
                }
                let near = pointer_pos
                    .and_then(|p| handles.iter().find(|(_, h)| (*h - p).length() < 9.0).map(|(k, _)| *k));
                let dragging = self.geo_xf_drag.as_ref().is_some_and(|d| d.node == node);
                *geo_handle_hot |= near.is_some() || dragging;
                // Draw: ring, axes, handles (hovered ones highlight).
                if fr.turns {
                    ink.circle_stroke(c, RING, 1.0, 0.5, false);
                }
                ink.seg(c, hx, 1.0, 0.8, false);
                ink.seg(c, hy, 1.0, 0.8, false);
                for (k, h) in &handles {
                    let hi = near == Some(*k)
                        || self.geo_xf_drag.as_ref().is_some_and(|d| d.node == node && d.handle == *k);
                    if *k <= 1 {
                        ink.circle_fill(*h, 5.0, 0.9, hi);
                    } else {
                        ink.rect(egui::Rect::from_center_size(*h, egui::vec2(8.0, 8.0)), 1.0, true, 0.9, 1.0, hi);
                    }
                }
                let (pressed, down, alt) =
                    ui.input(|i| (i.pointer.primary_pressed(), i.pointer.primary_down(), i.modifiers.alt));
                if pressed {
                    if let (Some(k), Some(p)) = (near, pointer_pos) {
                        let start = match self.egraph.node(node).map(|n| &n.kind) {
                            Some(NodeKind::GeoNode(k)) => k.clone(),
                            _ => continue,
                        };
                        self.geo_xf_drag = Some(GeoXfDrag { node, handle: k, origin: p, start, frame: fr, base });
                    }
                }
                let Some(d) = self.geo_xf_drag.clone().filter(|d| d.node == node) else { continue };
                if !down {
                    self.geo_xf_drag = None;
                    continue;
                }
                let Some(p) = pointer_pos else { continue };
                let c0 = g2s(d.frame.centre);
                let dg = {
                    let a = s2g(p);
                    let b = s2g(d.origin);
                    [a[0] - b[0], a[1] - b[1]]
                };
                // Snapping moves the CENTRE onto the target: the
                // travel is whatever gets it there.
                let free = ui.input(|i| i.modifiers.ctrl);
                let dg = if d.handle == 0 {
                    let want = [d.frame.centre[0] + dg[0], d.frame.centre[1] + dg[1]];
                    let scr = (out_to_pts(egui::vec2(owf, 0.0)) - out_to_pts(egui::vec2(0.0, 0.0))).x;
                    let (to, _) = self.geo_snap(want, owf, scr, Some(d.node), free);
                    [to[0] - d.frame.centre[0], to[1] - d.frame.centre[1]]
                } else {
                    dg
                };
                let edit = match d.handle {
                    0 if alt => GeoEdit::Pivot(dg),
                    0 => GeoEdit::Move(dg),
                    1 => GeoEdit::Rotate((p - c0).angle() - (d.origin - c0).angle()),
                    k => {
                        let r0 = d.frame.rot;
                        let (ax0, ay0) = (egui::vec2(r0.cos(), r0.sin()), egui::vec2(-r0.sin(), r0.cos()));
                        let along = |a: egui::Vec2, q: egui::Pos2| (q - c0).dot(a);
                        let f = |a: egui::Vec2| along(a, p) / along(a, d.origin).max(1.0);
                        GeoEdit::Scale(match k {
                            2 => [f(ax0), 1.0],
                            3 => [1.0, f(ay0)],
                            _ => {
                                let u = f((ax0 + ay0).normalized());
                                [u, u]
                            }
                        })
                    }
                };
                let out = geo_frame_apply(&d.start, d.base, edit);
                if let Some(n) = self.egraph.node_mut(d.node) {
                    n.kind = NodeKind::GeoNode(out);
                }
                self.playback_dirty = true;
            }
        }
    }
}
