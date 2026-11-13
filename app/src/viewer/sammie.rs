//! Sammie: the clicks on the frame on screen, editable.
//! Moved out of `viewer_ui` (lib.rs) unchanged.

use crate::*;

impl Gfx {
    /// Sammie: the clicks on the frame on screen, editable.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gizmo_sammie(
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
        // Sammie: the clicks on the frame on screen, editable (S-click
        // include, shift+S-click exclude, D-click remove, drag to move),
        // and a faint ring for each click on other frames.
        for node in selected.iter().copied() {
            let Some(NodeKind::Sammie(sp)) = self.egraph.node(node).map(|n| &n.kind) else {
                continue;
            };
            let sp = sp.clone();
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
                [v.x / nw, v.y / nh]
            };
            if hud_key_list.is_empty() {
                use hudkeys::{chord, key};
                hud_key_list.push(key("S", "click: include", egui::Key::S));
                hud_key_list.push(chord("shift+S", "click: exclude"));
                hud_key_list.push(key("D", "click point: remove", egui::Key::D));
            }
            let overlay = self.sammie_overlay(node, cur_t as i64);
            if let Some(e) = sammie::gizmo(
                ui, ink, node, &sp, cur_t as i64, &nav, &to_scr, &from_screen, pointer_pos, overlay,
            ) {
                if let Some(NodeKind::Sammie(auth)) = self.egraph.node_mut(node).map(|n| &mut n.kind) {
                    sammie::apply(&mut auth.points, e);
                }
                self.sammie_edited(node);
            }
        }
    }
}
