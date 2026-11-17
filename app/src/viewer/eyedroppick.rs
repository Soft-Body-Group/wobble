//! The eyedropper, armed by E over a colour control.
//! Moved out of `viewer_ui` (lib.rs) unchanged.

use crate::*;

impl Gfx {
    /// The eyedropper, armed by E over a colour control.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn viewer_eyedrop(
        &mut self,
        ui: &mut egui::Ui,
        avail: egui::Rect,
        nav: &egui::Response,
        ow: u32,
        oh: u32,
        owf: f32,
        ohf: f32,
        rect: egui::Rect,
        scrubbing: bool,
    ) {
        // Eyedropper, armed by E over a colour control: a click samples the
        // readout's box under the pointer, a drag keeps sampling under the
        // pointer as it moves (release keeps the last), a shift-drag
        // averages the box it draws. Esc / right-click / a click off the
        // viewer stands it down.
        if eyedrop::armed(ui.ctx()) {
            let to_px = |p: egui::Pos2| {
                (
                    ((p.x - rect.min.x) / rect.width() * owf).floor() as i32,
                    ((p.y - rect.min.y) / rect.height() * ohf).floor() as i32,
                )
            };
            let ptr = ui.ctx().pointer_latest_pos();
            let (down, pressed, secondary, shift) = ui.input(|i| {
                (
                    i.pointer.primary_down(),
                    i.pointer.primary_pressed(),
                    i.pointer.secondary_pressed(),
                    i.modifiers.shift,
                )
            });
            let n = self.picker.box_size();
            if nav.hovered() {
                ui.ctx().set_cursor_icon(egui::CursorIcon::Crosshair);
            }
            if secondary
                || ui.input(|i| i.key_pressed(egui::Key::Escape))
                || (pressed && !nav.hovered() && self.eyedrop.start.is_none())
            {
                eyedrop::cancel(ui.ctx());
                self.eyedrop.start = None;
            } else if let Some(start) = self.eyedrop.start {
                let here = ptr.map(to_px).unwrap_or(start);
                if down && !self.eyedrop.boxing {
                    // Follow the pointer: the readout's box under it, live.
                    self.eyedrop.queue_live(eyedrop::read_box(here, here, (ow, oh), n));
                } else if down {
                    let a = rect.min
                        + egui::vec2(start.0 as f32 / owf, start.1 as f32 / ohf) * rect.size();
                    let b = ptr.unwrap_or(a);
                    if (b - a).length() > 3.0 {
                        ui.painter().rect_stroke(
                            egui::Rect::from_two_pos(a, b),
                            0.0,
                            egui::Stroke::new(1.0, theme::viewer_live()),
                            egui::StrokeKind::Middle,
                        );
                    }
                } else {
                    self.eyedrop.start = None;
                    let from = if self.eyedrop.boxing { start } else { here };
                    self.eyedrop.pending =
                        eyedrop::read_box(from, here, (ow, oh), n).map(|b| (b, true));
                }
            } else if pressed && nav.hovered() && !scrubbing {
                self.eyedrop.start = ptr.map(to_px);
                self.eyedrop.boxing = shift;
            }
            if let Some(p) = ptr.filter(|p| avail.contains(*p)) {
                ui.painter().text(
                    p + egui::vec2(14.0, 14.0),
                    egui::Align2::LEFT_TOP,
                    "sample: click or drag · shift-drag a box · esc",
                    egui::FontId::proportional(11.0),
                    theme::viewer_live(),
                );
            }
            ui.ctx().request_repaint();
        } else {
            self.eyedrop.start = None;
        }
    }
}
