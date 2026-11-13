//! CameraTrack: the solved points through the solved camera at this frame.
//! Moved out of `viewer_ui` (lib.rs) unchanged.

use crate::*;

impl Gfx {
    /// CameraTrack: the solved points through the solved camera at this frame.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gizmo_camtrack(
        &mut self,
        ui: &mut egui::Ui,
        cur_t: u64,
        owf: f32,
        ohf: f32,
        pointer_pos: Option<egui::Pos2>,
        out_to_pts: &impl Fn(egui::Vec2) -> egui::Pos2,
        selected: &Vec<u64>,
        nav: &egui::Response,
        ink: &mut ink::InkShapes,
        hud_key_list: &mut Vec<hudkeys::HudKey>,
    ) {
        // CameraTrack: the solved points through the solved camera at
        // this frame — one mesh, however many — and S-click picks.
        for node in selected.iter().copied() {
            if !matches!(self.egraph.node(node).map(|n| &n.kind), Some(NodeKind::CamTrack(p)) if p.solved() && p.show) {
                continue;
            }
            let (nw, nh) = self.node_res_f32(node).unwrap_or((owf, ohf));
            let hd = downstream_proj(
                &self.egraph,
                node,
                &|id| self.node_res_f32(id).unwrap_or((owf, ohf)),
                cur_t,
            );
            let to_scr = |q: [f32; 2]| out_to_pts(apply_h(&hd, egui::vec2(q[0] * nw, q[1] * nh)));
            if hud_key_list.is_empty() {
                hud_key_list.push(hudkeys::key("S", "click point: pick", egui::Key::S));
            }
            ink.set_tint(self.hud_ink(node));
            self.camtrack_gizmo(ui, ink, node, cur_t as i64, &nav, &to_scr, pointer_pos);
        }
    }
}
