//! 2dini gradient handles: start and end of a linear or radial fill.
//! Moved out of `viewer_ui` (lib.rs) unchanged.

use crate::*;

impl Gfx {
    /// 2dini gradient handles: start and end of a linear or radial fill.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gizmo_geo_gradient(
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
        // GRADIENT handles: a selected shape or Style with a linear or
        // radial fill shows its start (a ring: the fill) and end (a dot:
        // the gradient colour) on a line; either drags, from the press.
        if self.dive.last().is_some_and(|d| d.kind == DiveKind::Geo) && !hud_hide && self.geo.tool.is_none() {
            let g2s = |p: [f32; 2]| out_to_pts(egui::vec2(p[0] * owf, p[1] * owf));
            let s2g = |p: egui::Pos2| {
                let q = pts_to_out(p);
                [q.x / owf, q.y / owf]
            };
            let (pressed, down) = ui.input(|i| (i.pointer.primary_pressed(), i.pointer.primary_down()));
            for node in selected.iter().copied() {
                let Some(NodeKind::GeoNode(mut gk)) = self.live_kind(node, cur_t) else { continue };
                let Some(st) = geo_style_mut(&mut gk) else { continue };
                let mut gr = st.grad;
                if !matches!(gr.kind, engine::geo::GradKind::Linear | engine::geo::GradKind::Radial) {
                    continue;
                }
                // Never placed: lay it across the shape it fills, left
                // to right (radial: from the middle out to the right).
                let dflt = engine::geo::Gradient::default();
                if gr.a == dflt.a && gr.b == dflt.b {
                    self.geo_hud_refresh(&[node], owf, ohf, cur_t);
                    let bb = self
                        .geo.hud_cache
                        .as_ref()
                        .and_then(|c| c.geos.iter().find(|(n, _)| *n == node))
                        .and_then(|(_, g)| g.bbox());
                    if let Some((lo, hi)) = bb {
                        let mid = (lo[1] + hi[1]) * 0.5;
                        let (a, b) = match gr.kind {
                            engine::geo::GradKind::Radial => ([(lo[0] + hi[0]) * 0.5, mid], [hi[0], mid]),
                            _ => ([lo[0], mid], [hi[0], mid]),
                        };
                        gr.a = a;
                        gr.b = b;
                        if let Some(st) = self.egraph.node_mut(node).and_then(|n| match &mut n.kind {
                            NodeKind::GeoNode(k) => geo_style_mut(k),
                            _ => None,
                        }) {
                            st.grad.a = a;
                            st.grad.b = b;
                        }
                        self.playback_dirty = true;
                    }
                }
                let (a, b) = (g2s(gr.a), g2s(gr.b));
                if gr.kind == engine::geo::GradKind::Radial {
                    ink.circle_stroke(a, (b - a).length(), 1.0, 0.45, false);
                }
                ink.seg(a, b, 1.0, 0.8, false);
                let near = pointer_pos.and_then(|p| {
                    if (p - b).length() < 9.0 {
                        Some(1u8)
                    } else if (p - a).length() < 9.0 {
                        Some(0)
                    } else {
                        None
                    }
                });
                let held = self.geo.grad_drag.filter(|d| d.0 == node);
                ink.circle_stroke(a, 6.0, 2.0, 0.95, near == Some(0) || held.is_some_and(|d| d.1 == 0));
                ink.circle_fill(b, 5.0, 0.95, near == Some(1) || held.is_some_and(|d| d.1 == 1));
                if near.is_some() || held.is_some() {
                    *geo_handle_hot = true;
                }
                if pressed && self.geo.xf_drag.is_none() {
                    if let (Some(k), Some(p)) = (near, pointer_pos) {
                        self.geo.grad_drag = Some((node, k, p, if k == 0 { gr.a } else { gr.b }));
                    }
                }
                let Some((_, k, origin, start)) = self.geo.grad_drag.filter(|d| d.0 == node) else { continue };
                if !down {
                    self.geo.grad_drag = None;
                    continue;
                }
                let Some(p) = pointer_pos else { continue };
                let (q, o) = (s2g(p), s2g(origin));
                let to = [start[0] + q[0] - o[0], start[1] + q[1] - o[1]];
                if let Some(st) = self.egraph.node_mut(node).and_then(|n| match &mut n.kind {
                    NodeKind::GeoNode(k) => geo_style_mut(k),
                    _ => None,
                }) {
                    if k == 0 {
                        st.grad.a = to;
                    } else {
                        st.grad.b = to;
                    }
                }
                self.playback_dirty = true;
            }
        }
    }
}
