//! Viewer zoom and pan, with wobble's hands (imfram's `nav`, Blender's
//! 2D rules): the wheel or a pinch zooms about the pointer, ctrl + two
//! fingers zooms, two fingers pan; MMB drag pans, ctrl+MMB drag-zooms,
//! alt+LMB pans, cmd/alt+RMB drag-zooms. A plain LMB drag stays the
//! viewer's own (the source picture drags onto the timeline).
//!
//! A `View` is a zoom about the picture's fitted centre plus a pan in
//! screen points, applied to the fitted rect; R over a viewer fits it
//! (zoom 1, no pan), shift+R gives the other viewer the same size and view.

use egui::{PointerButton, Pos2, Rect, Response, Vec2};
use imfram::nav;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct View {
    pub zoom: f32,
    pub pan: Vec2,
}

impl Default for View {
    fn default() -> Self {
        View { zoom: 1.0, pan: Vec2::ZERO }
    }
}

const MIN_ZOOM: f32 = 0.05;
const MAX_ZOOM: f32 = 64.0;

impl View {
    /// Where the fitted picture `img` (inside `frame`) sits under this view.
    pub fn apply(&self, frame: Rect, img: Rect) -> Rect {
        let c = frame.center();
        let map = |p: Pos2| c + (p - c) * self.zoom + self.pan;
        Rect::from_min_max(map(img.min), map(img.max))
    }

    /// Zoom by `f` keeping the screen point `p` still.
    pub fn zoom_about(&mut self, frame: Rect, p: Pos2, f: f32) {
        let z = (self.zoom * f).clamp(MIN_ZOOM, MAX_ZOOM);
        let f = z / self.zoom;
        let c = frame.center();
        // c + (q - c) * z' + pan' = p + (c + (q - c) * z + pan - p) * f
        self.pan = (self.pan + (c - p)) * f + (p - c);
        self.zoom = z;
    }

    pub fn fit(&mut self) {
        *self = View::default();
    }

    /// The frame's navigation over a viewer: `resp` is the viewer's
    /// click-and-drag response over `frame`. True when it was navigation
    /// (the caller then leaves the drag alone).
    pub fn nav(&mut self, ui: &egui::Ui, frame: Rect, resp: &Response) -> bool {
        let mut used = false;
        if resp.hovered() {
            nav::claim_wheel(ui, frame);
            let w = nav::wheel_nav(ui);
            let at = resp.hover_pos().unwrap_or(frame.center());
            if w.zoom != 1.0 {
                self.zoom_about(frame, at, w.zoom);
                used = true;
            }
            if w.pan != Vec2::ZERO {
                self.pan += w.pan;
                used = true;
            }
        }
        let mods = ui.input(|i| i.modifiers);
        for b in [PointerButton::Middle, PointerButton::Primary, PointerButton::Secondary] {
            if !resp.dragged_by(b) {
                continue;
            }
            match nav::drag_nav(b, mods) {
                Some(nav::DragNav::Pan) => {
                    self.pan += resp.drag_delta();
                    used = true;
                }
                Some(nav::DragNav::Zoom) => {
                    let at = ui.input(|i| i.pointer.press_origin()).unwrap_or(frame.center());
                    self.zoom_about(frame, at, nav::drag_zoom(resp.drag_delta()));
                    used = true;
                }
                None => {}
            }
        }
        used
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::{pos2, vec2};

    #[test]
    fn zooming_keeps_the_point_under_the_pointer() {
        let frame = Rect::from_min_size(pos2(0.0, 0.0), vec2(800.0, 450.0));
        let img = Rect::from_min_size(pos2(0.0, 0.0), vec2(800.0, 450.0));
        let mut v = View::default();
        let p = pos2(600.0, 100.0);
        // The picture point under p before the zoom...
        let before = v.apply(frame, img);
        let u = ((p.x - before.min.x) / before.width(), (p.y - before.min.y) / before.height());
        v.zoom_about(frame, p, 2.5);
        let after = v.apply(frame, img);
        let q = pos2(after.min.x + u.0 * after.width(), after.min.y + u.1 * after.height());
        assert!((q - p).length() < 1e-3, "{q:?}");
        assert!((after.width() - 2000.0).abs() < 1e-3);
        v.pan += vec2(10.0, -5.0);
        assert_eq!(v.apply(frame, img).min, after.min + vec2(10.0, -5.0));
        v.fit();
        assert_eq!(v.apply(frame, img), img);
    }
}
