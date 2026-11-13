//! 2dini Skeleton: the bones, posed, and their drags.
//! Moved out of `viewer_ui` (lib.rs) unchanged.

use crate::*;

impl Gfx {
    /// 2dini Skeleton: the bones, posed, and their drags.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gizmo_geo_bones(
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
        // BONES: with one Skeleton selected its bones draw as the usual
        // tapered shapes, posed. Drag a bone to turn it about its head
        // (keys the pose), alt-drag a tail to move that joint at rest,
        // alt-drag the root's head to move the whole rig.
        if in_geo && !hud_hide && self.geo_tool.is_none() {
            let skel_node = (self.selected.len() == 1).then(|| self.selected[0]).filter(|id| {
                matches!(self.egraph.node(*id).map(|n| &n.kind), Some(NodeKind::GeoNode(engine::geo::GeoNode::Skeleton { .. })))
            });
            match skel_node {
                None => self.geo_bone_drag = None,
                Some(sn) => {
                    self.geo_hud_refresh(&[sn], owf, ohf, cur_t);
                    let g2s = |p: [f32; 2]| out_to_pts(egui::vec2(p[0] * owf, p[1] * owf));
                    let bones: Vec<engine::georig::SkelBone> = self
                        .geo_hud_cache
                        .as_ref()
                        .and_then(|c| c.geos.iter().find(|(n, _)| *n == sn))
                        .map(|(_, g)| g.bones.clone())
                        .unwrap_or_default();
                    // Screen head/tail of every bone, posed.
                    let ends: Vec<(egui::Pos2, egui::Pos2)> = bones
                        .iter()
                        .map(|b| (g2s(engine::georig::apply(&b.xf, b.head)), g2s(engine::georig::apply(&b.xf, b.tail))))
                        .collect();
                    let seg_d = |p: egui::Pos2, a: egui::Pos2, b: egui::Pos2| {
                        let ab = b - a;
                        let t = ((p - a).dot(ab) / ab.length_sq().max(1e-6)).clamp(0.0, 1.0);
                        (a + ab * t - p).length()
                    };
                    let mods = ui.input(|i| i.modifiers);
                    // What a press here would take: alt aims at joints
                    // (tails, the root's head), plain at bone bodies.
                    let hot: Option<(usize, bool)> = pointer_pos.and_then(|p| {
                        if mods.alt {
                            let mut best: Option<(usize, bool, f32)> = None;
                            for (i, (h, t)) in ends.iter().enumerate() {
                                let dt = (*t - p).length();
                                if dt < 9.0 && best.is_none_or(|b| dt < b.2) {
                                    best = Some((i, false, dt));
                                }
                                if bones[i].parent.is_none() {
                                    let dh = (*h - p).length();
                                    if dh < 9.0 && best.is_none_or(|b| dh < b.2) {
                                        best = Some((i, true, dh));
                                    }
                                }
                            }
                            best.map(|(i, r, _)| (i, r))
                        } else {
                            ends.iter()
                                .enumerate()
                                .map(|(i, (h, t))| (i, seg_d(p, *h, *t)))
                                .filter(|(_, d)| *d < 8.0)
                                .min_by(|a, b| a.1.total_cmp(&b.1))
                                .map(|(i, _)| (i, false))
                        }
                    });
                    let over_bar = pointer_pos.is_some_and(|p| {
                        self.geo_tool_bar.is_some_and(|b| b.contains(p)) || self.geo_viz_bar.is_some_and(|b| b.contains(p))
                    });
                    let in_view = pointer_pos.is_some_and(|p| rect.contains(p)) && !over_bar;
                    if in_view && (hot.is_some() || self.geo_bone_drag.is_some()) {
                        *geo_handle_hot = true;
                    }
                    let (pressed, down) = ui.input(|i| (i.pointer.primary_pressed(), i.pointer.primary_down()));
                    if pressed && in_view && self.geo_bone_drag.is_none() {
                        if let (Some((i, root)), Some(p)) = (hot, pointer_pos) {
                            let (rest_bones, pose0) = match self.egraph.node(sn).map(|n| n.kind_at(cur_t)) {
                                Some(NodeKind::GeoNode(engine::geo::GeoNode::Skeleton { bones, pose })) => {
                                    let v = pose.get(i).copied().unwrap_or(0.0);
                                    (bones, v)
                                }
                                _ => (Vec::new(), 0.0),
                            };
                            self.geo_bone_drag = Some(BoneDrag {
                                node: sn,
                                bone: i,
                                origin: p,
                                pivot: ends[i].0,
                                pose0,
                                rest: mods.alt.then_some((rest_bones, root)),
                            });
                        }
                    }
                    if let Some(bd) = self.geo_bone_drag.clone().filter(|b| b.node == sn) {
                        if let Some(p) = pointer_pos {
                            match &bd.rest {
                                None => {
                                    let a0 = (bd.origin - bd.pivot).angle();
                                    let a1 = (p - bd.pivot).angle();
                                    let mut turn = a1 - a0;
                                    turn = (turn + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI;
                                    let v = bd.pose0 + turn;
                                    let ch = anim::Channel::Field(bd.bone as u16);
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
                                    let autokey = self.autokey;
                                    if let Some(editor::ENode { kind, anim, .. }) = self.egraph.node_mut(sn) {
                                        let before = kind.clone();
                                        let mut after = before.clone();
                                        if let NodeKind::GeoNode(gn) = &mut after {
                                            gn.set_literal(bd.bone, v);
                                        }
                                        if after != before || anim.contains_key(&ch) {
                                            keyed_write_back(anim, kind, &before, &after, cur_t, &[ch], autokey, &patch_driven, sn);
                                            self.playback_dirty = true;
                                        }
                                    }
                                }
                                Some((start, root)) => {
                                    let s2g = |q: egui::Pos2| {
                                        let o = pts_to_out(q);
                                        [o.x / owf, o.y / owf]
                                    };
                                    let (a, b) = (s2g(p), s2g(bd.origin));
                                    let d = [a[0] - b[0], a[1] - b[1]];
                                    let mut nb = start.clone();
                                    if *root {
                                        for x in &mut nb {
                                            x.head = [x.head[0] + d[0], x.head[1] + d[1]];
                                            x.tail = [x.tail[0] + d[0], x.tail[1] + d[1]];
                                        }
                                    } else {
                                        // Screen moves are posed; the rest joint moves
                                        // by the inverse of the bone's turn.
                                        let r = bones.get(bd.bone).map(|b| b.xf[2].atan2(b.xf[0])).unwrap_or(0.0);
                                        let (s, k) = (-r).sin_cos();
                                        let dr = [k * d[0] - s * d[1], s * d[0] + k * d[1]];
                                        let t = &mut nb[bd.bone].tail;
                                        *t = [t[0] + dr[0], t[1] + dr[1]];
                                        for (j, x) in start.iter().enumerate() {
                                            if x.parent == Some(bd.bone as u32) {
                                                nb[j].head = [x.head[0] + dr[0], x.head[1] + dr[1]];
                                            }
                                        }
                                    }
                                    if let Some(NodeKind::GeoNode(engine::geo::GeoNode::Skeleton { bones, .. })) =
                                        self.egraph.node_mut(sn).map(|n| &mut n.kind)
                                    {
                                        if *bones != nb {
                                            *bones = nb;
                                            self.playback_dirty = true;
                                        }
                                    }
                                }
                            }
                        }
                        if !down {
                            self.geo_bone_drag = None;
                        }
                    }
                    // Draw: each bone a long diamond from head to tail,
                    // a ring at the head; the hot one lit.
                    let lit = self.geo_bone_drag.as_ref().map(|b| b.bone).or(hot.map(|h| h.0));
                    for (i, (h, t)) in ends.iter().enumerate() {
                        let ax = *t - *h;
                        let len = ax.length();
                        if len < 1.0 {
                            continue;
                        }
                        let n = egui::vec2(-ax.y, ax.x) / len * (len * 0.12).clamp(2.0, 9.0);
                        let m = *h + ax * 0.2;
                        let on = lit == Some(i);
                        ink.seg(*h, m + n, 1.5, 0.9, on);
                        ink.seg(*h, m - n, 1.5, 0.9, on);
                        ink.seg(m + n, *t, 1.5, 0.9, on);
                        ink.seg(m - n, *t, 1.5, 0.9, on);
                        ink.circle_stroke(*h, 3.5, 1.5, 0.9, on);
                    }
                    if let Some((i, root)) = hot.filter(|_| mods.alt) {
                        let at = if root { ends[i].0 } else { ends[i].1 };
                        ink.circle_fill(at, 4.5, 0.9, true);
                    }
                }
            }
        }
    }
}
