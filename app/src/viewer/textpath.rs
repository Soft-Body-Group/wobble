//! Text: its path, editable while the node is selected.
//! Moved out of `viewer_ui` (lib.rs) unchanged.

use crate::*;

impl Gfx {
    /// Text: its path, editable while the node is selected.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gizmo_text_path(
        &mut self,
        ui: &mut egui::Ui,
        cur_t: u64,
        owf: f32,
        ohf: f32,
        pointer_pos: Option<egui::Pos2>,
        out_to_pts: &impl Fn(egui::Vec2) -> egui::Pos2,
        pts_to_out: &impl Fn(egui::Pos2) -> egui::Vec2,
        selected: &Vec<u64>,
        nav: &egui::Response,
        ink: &mut ink::InkShapes,
        hud_key_list: &mut Vec<hudkeys::HudKey>,
    ) {
        // Text: its path, editable while the node is selected and the
        // path is on or being started (S held).
        for node in selected.iter().copied() {
            let Some(NodeKind::Text(tp)) = self.egraph.node(node).map(|n| &n.kind) else {
                continue;
            };
            let s_held = ui.input(|i| i.key_down(egui::Key::S));
            if !(tp.path.on || s_held) {
                continue;
            }
            let tp = tp.clone();
            let pk = self.proxy.min(1.0);
            let hd = downstream_proj(
                &self.egraph,
                node,
                &|id| self.node_res_f32(id).unwrap_or((owf, ohf)),
                cur_t,
            );
            let hd_inv = engine::codegen::mat3_inv(&hd);
            let to_scr = |q: [f32; 2]| out_to_pts(apply_h(&hd, egui::vec2(q[0] * pk, q[1] * pk)));
            let from_screen = |p: egui::Pos2| {
                let v = apply_h(&hd_inv, pts_to_out(p));
                [v.x / pk, v.y / pk]
            };
            if hud_key_list.is_empty() {
                use hudkeys::{chord, key};
                hud_key_list.push(key("S", "click: add path point", egui::Key::S));
                hud_key_list.push(chord("alt", "drag point: pull handles"));
                hud_key_list.push(key("D", "click point: remove", egui::Key::D));
            }
            ink.set_tint(self.hud_ink(node));
            if let Some(e) =
                nodes::text::gizmo(ui, ink, node, &tp, &nav, &to_scr, &from_screen, pointer_pos)
            {
                if let Some(NodeKind::Text(auth)) = self.egraph.node_mut(node).map(|n| &mut n.kind) {
                    nodes::text::apply(&mut auth.path, e);
                }
            }
            ink.set_tint(None);
        }
    }
}
