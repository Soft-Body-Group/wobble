//! Cryptomatte picking: ctrl-click reads the id under the pointer off the screen.
//! Moved out of `viewer_ui` (lib.rs) unchanged.

use crate::*;

impl Gfx {
    /// Cryptomatte picking: ctrl-click reads the id under the pointer off the screen.
    pub(crate) fn viewer_crypto_pick(
        &mut self,
        ui: &mut egui::Ui,
        nav: &egui::Response,
        owf: f32,
        ohf: f32,
        rect: egui::Rect,
        scrubbing: bool,
    ) {
        // Cryptomatte picking: with a Cryptomatte in its ids view as the VIEW
        // TARGET, ctrl-click (cmd on a Mac) reads the id under the pointer
        // off the screen and adds that object; ctrl+shift-click removes it;
        // a plain click is not a pick. Same rule as the
        // keyer: the pixels have to be the ones on screen, and a viewed
        // Cryptomatte's picture carries the ids.
        let crypto_id = self.egraph.active_view().filter(|id| {
            matches!(self.egraph.node(*id).map(|n| &n.kind), Some(NodeKind::Cryptomatte(_)))
        });
        if let Some(cid) = crypto_id {
            // A bare click AND the start of a drag: with a click-and-drag
            // sense egui only reports `drag_started` once the pointer has
            // moved past its threshold, so a still click arrives as
            // `clicked` on release — and a click is how you pick.
            let pressed = nav.clicked_by(egui::PointerButton::Primary)
                || nav.drag_started_by(egui::PointerButton::Primary);
            if !scrubbing && pressed {
                // Modifiers off the state OR the button event itself: egui
                // carries them on the event, and that is the copy that is
                // certain to belong to THIS press.
                let mods = ui.input(|i| {
                    let mut m = i.modifiers;
                    for e in &i.events {
                        if let egui::Event::PointerButton {
                            button: egui::PointerButton::Primary,
                            modifiers,
                            ..
                        } = e
                        {
                            m = m | *modifiers;
                        }
                    }
                    m
                });
                if let Some(p) = ui.ctx().pointer_latest_pos() {
                    let x = ((p.x - rect.min.x) / rect.width() * owf) as i32;
                    let y = ((p.y - rect.min.y) / rect.height() * ohf) as i32;
                    if let Some(mode) =
                        cryptoui::PickMode::from_mods(mods.command || mods.ctrl, mods.shift)
                    {
                        self.crypto.click = Some((cid, x, y, mode));
                    }
                }
            }
        }
    }
}
