//! PlanarTrack: the region, then the tracked quad and its mesh.
//! Moved out of `viewer_ui` (lib.rs) unchanged.

use crate::*;

impl Gfx {
    /// PlanarTrack: the region, then the tracked quad and its mesh.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gizmo_planar(
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
        // PlanarTrack: the region (a free quad, corner and centre
        // handles) until the frame has a result; then the tracked quad
        // and its mesh, red when the plane was lost. Drawn from the
        // authored node — the track is data, not animation.
        for node in selected.iter().copied() {
            let Some(NodeKind::PlanarTrack(pp)) = self.egraph.node(node).map(|n| &n.kind) else {
                continue;
            };
            let ov = planarui::overlay(pp, cur_t as i64);
            ink.set_tint(self.hud_ink(node));
            let (nw, nh) = self.node_res_f32(node).unwrap_or((owf, ohf));
            let hd = downstream_proj(
                &self.egraph,
                node,
                &|id| self.node_res_f32(id).unwrap_or((owf, ohf)),
                cur_t,
            );
            let hd_inv = engine::codegen::mat3_inv(&hd);
            let to_scr = |q: [f32; 2]| out_to_pts(apply_h(&hd, egui::vec2(q[0] * nw, q[1] * nh)));
            let from_screen = |p: egui::Pos2| {
                let v = apply_h(&hd_inv, pts_to_out(p));
                egui::pos2(v.x, v.y)
            };
            if let Some(cs) =
                planarui::gizmo(ui, ink, node, &ov, (nw, nh), &to_scr, &from_screen, pointer_pos)
            {
                if let Some(editor::ENode { kind: NodeKind::PlanarTrack(auth), .. }) =
                    self.egraph.node_mut(node)
                {
                    auth.corners = cs;
                }
            }
        }
    }
}
