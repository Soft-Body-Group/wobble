//! 2dini Edit: every vertex flowing through it, live.
//! Moved out of `viewer_ui` (lib.rs) unchanged.

use crate::*;

impl Gfx {
    /// 2dini Edit: every vertex flowing through it, live.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gizmo_geo_edit(
        &mut self,
        ui: &mut egui::Ui,
        cur_t: u64,
        hud_hide: bool,
        in_geo: bool,
        owf: f32,
        ohf: f32,
        pointer_pos: Option<egui::Pos2>,
        out_to_pts: &impl Fn(egui::Vec2) -> egui::Pos2,
        pts_to_out: &impl Fn(egui::Pos2) -> egui::Vec2,
        rect: egui::Rect,
        geo_handle_hot: &mut bool,
        ink: &mut ink::InkShapes,
    ) {
        // EDIT: with one Edit node selected, its points are live — every
        // vertex of what flows through it, as it draws now. A press on
        // one grabs it (and the rest of the pick), shift-click adds or
        // drops one, alt-click puts it back where its input has it, a
        // drag on nothing boxes points. Moves are offsets from the
        // press, so a drag is never summed.
        if in_geo && !hud_hide && self.geo.tool.is_none() {
            let edit_node = (self.selected.len() == 1)
                .then(|| self.selected[0])
                .filter(|id| matches!(self.egraph.node(*id).map(|n| &n.kind), Some(NodeKind::GeoNode(engine::geo::GeoNode::Edit { .. }))));
            if self.geo.edit_owner != edit_node {
                self.geo.edit_pick.clear();
                self.geo.edit_owner = edit_node;
            }
            match edit_node {
                None => {
                    self.geo.edit_drag = None;
                }
                Some(en) => {
                    self.geo_hud_refresh(&[en], owf, ohf, cur_t);
                    let g2s = |p: [f32; 2]| out_to_pts(egui::vec2(p[0] * owf, p[1] * owf));
                    let s2g = |p: egui::Pos2| {
                        let q = pts_to_out(p);
                        [q.x / owf, q.y / owf]
                    };
                    // (key, screen spot), every vertex, capped.
                    let spots: Vec<((u32, u32), egui::Pos2)> = self
                        .geo.hud_cache
                        .as_ref()
                        .and_then(|c| c.geos.iter().find(|(n, _)| *n == en))
                        .map(|(_, g)| {
                            g.prims
                                .iter()
                                .enumerate()
                                .flat_map(|(i, pr)| {
                                    pr.verts.iter().enumerate().filter_map(move |(j, v)| {
                                        g.points.get(*v as usize).map(|q| ((i as u32, j as u32), g2s(q.p)))
                                    })
                                })
                                .take(20_000)
                                .collect()
                        })
                        .unwrap_or_default();
                    let edits: Vec<engine::geo::PointEdit> = match self.egraph.node(en).map(|n| &n.kind) {
                        Some(NodeKind::GeoNode(engine::geo::GeoNode::Edit { edits })) => edits.clone(),
                        _ => Vec::new(),
                    };
                    let near = pointer_pos.and_then(|p| {
                        spots
                            .iter()
                            .map(|(k, s)| (*k, (*s - p).length()))
                            .filter(|(_, d)| *d < 7.0)
                            .min_by(|a, b| a.1.total_cmp(&b.1))
                            .map(|(k, _)| k)
                    });
                    let over_bar = pointer_pos.is_some_and(|p| {
                        self.geo.tool_bar.is_some_and(|b| b.contains(p)) || self.geo.viz_bar.is_some_and(|b| b.contains(p))
                    });
                    let in_view = pointer_pos.is_some_and(|p| rect.contains(p)) && !over_bar;
                    if in_view {
                        *geo_handle_hot = true; // the Edit owns the viewer's clicks
                    }
                    let (pressed, down, mods) =
                        ui.input(|i| (i.pointer.primary_pressed(), i.pointer.primary_down(), i.modifiers));
                    if pressed && in_view && self.geo.edit_drag.is_none() {
                        let p = pointer_pos.unwrap_or_default();
                        match near {
                            Some(k) if mods.alt => {
                                let left: Vec<_> = edits.iter().copied().filter(|e| (e.prim, e.vert) != k).collect();
                                if let Some(NodeKind::GeoNode(engine::geo::GeoNode::Edit { edits })) = self.egraph.node_mut(en).map(|n| &mut n.kind) {
                                    *edits = left;
                                }
                                self.playback_dirty = true;
                            }
                            Some(k) if mods.shift => {
                                if let Some(i) = self.geo.edit_pick.iter().position(|q| *q == k) {
                                    self.geo.edit_pick.remove(i);
                                } else {
                                    self.geo.edit_pick.push(k);
                                }
                            }
                            Some(k) => {
                                if !self.geo.edit_pick.contains(&k) {
                                    self.geo.edit_pick = vec![k];
                                }
                                self.geo.edit_drag = Some(GeoEditDrag::Move { node: en, origin: p, start: edits.clone() });
                            }
                            None => {
                                if !mods.shift {
                                    self.geo.edit_pick.clear();
                                }
                                self.geo.edit_drag = Some(GeoEditDrag::Box { node: en, origin: p });
                            }
                        }
                    }
                    let mut boxed: Option<egui::Rect> = None;
                    match self.geo.edit_drag.clone() {
                        Some(GeoEditDrag::Move { node, origin, start }) if node == en => {
                            if let Some(p) = pointer_pos {
                                let (a, b) = (s2g(p), s2g(origin));
                                let moved = edits_moved(&start, &self.geo.edit_pick, [a[0] - b[0], a[1] - b[1]]);
                                if let Some(NodeKind::GeoNode(engine::geo::GeoNode::Edit { edits })) =
                                    self.egraph.node_mut(en).map(|n| &mut n.kind)
                                {
                                    if *edits != moved {
                                        *edits = moved;
                                        self.playback_dirty = true;
                                    }
                                }
                            }
                            if !down {
                                self.geo.edit_drag = None;
                            }
                        }
                        Some(GeoEditDrag::Box { node, origin }) if node == en => {
                            if let Some(p) = pointer_pos {
                                let r = egui::Rect::from_two_pos(origin, p);
                                boxed = Some(r);
                                if !down {
                                    for (k, s) in &spots {
                                        if r.contains(*s) && !self.geo.edit_pick.contains(k) {
                                            self.geo.edit_pick.push(*k);
                                        }
                                    }
                                }
                            }
                            if !down {
                                self.geo.edit_drag = None;
                            }
                        }
                        Some(_) => self.geo.edit_drag = None,
                        None => {}
                    }
                    // Draw: every vertex a small square, moved ones
                    // filled, picked ones lit, the hovered one grown.
                    for (k, s) in &spots {
                        let moved = edits.iter().any(|e| (e.prim, e.vert) == *k);
                        let picked = self.geo.edit_pick.contains(k);
                        let r = if near == Some(*k) { 4.5 } else { 3.0 };
                        ink.rect(egui::Rect::from_center_size(*s, egui::vec2(2.0 * r, 2.0 * r)), 1.0, moved || picked, 0.9, 1.0, picked);
                    }
                    if let Some(r) = boxed {
                        ink.rect(r, 1.0, false, 0.6, 1.0, false);
                    }
                }
            }
        }
    }
}
