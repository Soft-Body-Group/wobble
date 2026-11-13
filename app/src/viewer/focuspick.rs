//! Defocus focus pick, armed from the node's panel.
//! Moved out of `viewer_ui` (lib.rs) unchanged.

use crate::*;

impl Gfx {
    /// Defocus focus pick, armed from the node's panel.
    pub(crate) fn viewer_focus_pick(
        &mut self,
        ui: &mut egui::Ui,
        nav: &egui::Response,
        rect: egui::Rect,
        scrubbing: bool,
    ) {
        // Defocus focus pick, armed from the node's panel: a press on the
        // picture samples the node's depth pass there, a drag keeps
        // sampling (a live focus pull), the release disarms. Reads the
        // node's own Z texture, so the node need not be viewed.
        if let Some(fid) = self.focus_pick.armed {
            if !matches!(self.egraph.node(fid).map(|n| &n.kind), Some(NodeKind::Defocus(_))) {
                self.focus_pick.armed = None;
            } else {
                let down = nav.is_pointer_button_down_on()
                    && ui.input(|i| i.pointer.primary_down());
                if nav.hovered() {
                    ui.ctx().set_cursor_icon(egui::CursorIcon::Crosshair);
                }
                if !scrubbing && down {
                    self.focus_pick.held = true;
                    self.focus_pick.cursor = ui.ctx().pointer_latest_pos().map(|p| {
                        let q = (p - rect.min) / rect.size();
                        [q.x, q.y]
                    });
                } else if self.focus_pick.held {
                    // Released: one pick per arming. The last read in
                    // flight still lands.
                    self.focus_pick.held = false;
                    self.focus_pick.cursor = None;
                    self.focus_pick.armed = None;
                } else if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                    self.focus_pick.armed = None;
                }
            }
        }
    }
}
