//! The Keyer's HUD: the keyed volume drawn on the viewer while you sample.
//! Moved out of `viewer_ui` (lib.rs) unchanged.

use crate::*;

impl Gfx {
    /// The Keyer's HUD: the keyed volume drawn on the viewer while you sample.
    pub(crate) fn gizmo_keyer_hud(
        &mut self,
        ui: &mut egui::Ui,
        avail: egui::Rect,
    ) {
        // Keyer HUD: the keyed VOLUME, on the viewer, while you sample.
        // Sampling is a gesture on the image — your eyes are on the
        // picture, not on the properties panel — so the thing that shows
        // what the gesture is doing has to be on the picture too. Same
        // orbit as the panel's view, so they are one object seen twice
        // rather than two unrelated pictures of it.
        if let Some(kid) = self.egraph.active_view().filter(|id| {
            matches!(self.egraph.node(*id).map(|n| &n.kind), Some(NodeKind::Keyer(_)))
        }) {
            if !self.hud.hidden.contains(&kid) && !self.prefs.hud_hide.unwrap_or(false) {
                let side = 156.0;
                let off = *self
                    .hud.pos
                    .entry(kid)
                    .or_insert(egui::vec2(avail.width() - side - 20.0, 16.0));
                let hud = egui::Rect::from_min_size(
                    avail.min + off,
                    egui::vec2(side, side),
                );
                let drag = ui.interact(
                    hud,
                    ui.id().with(("keyer-hud", kid)),
                    egui::Sense::click_and_drag(),
                );
                if drag.dragged() {
                    if let Some(o) = self.hud.pos.get_mut(&kid) {
                        *o += drag.drag_delta();
                        o.x = o.x.clamp(4.0, (avail.width() - side - 4.0).max(4.0));
                        o.y = o.y.clamp(4.0, (avail.height() - side - 4.0).max(4.0));
                    }
                }
                if let Some(NodeKind::Keyer(k)) =
                    self.egraph.node(kid).map(|n| &n.kind)
                {
                    let p = ui.painter().with_clip_rect(hud);
                    p.rect_filled(hud, 5.0, theme::hud_bg().gamma_multiply_u8(190));
                    keyerui::draw_space3d(
                        &p,
                        hud.shrink(4.0),
                        k,
                        self.keyer.cloud.get(&kid),
                        self.keyer.sel.get(&kid).copied().unwrap_or(0),
                        keyerui::orbit_of(ui.ctx(), ui.id().with(("keyer-space", kid))),
                        false,
                    );
                }
            }
        }
    }
}
