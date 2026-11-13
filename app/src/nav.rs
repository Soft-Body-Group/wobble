//! Blender's navigation hands, one implementation for every 2D canvas
//! (manifesto 11: we are a Blender companion — muscle memory carries).
//!
//! Drags:
//!   MMB                  pan   (shift+MMB too — Blender's pan chord, so a
//!                               3D view can one day split them into
//!                               tumble vs pan without retraining anyone)
//!   ctrl+MMB             drag-zoom about the press point
//!   alt+LMB              pan   ("emulate 3 button mouse": the tablet path)
//!   shift+alt+LMB        pan
//!   ctrl+alt+LMB         drag-zoom
//!   alt/cmd+RMB          drag-zoom (the Houdini dolly this app always had;
//!                               macOS ALSO lands ctrl+alt+LMB here, because
//!                               the OS turns ctrl+LMB into a right-click)
//!
//! Wheel and trackpad (Blender's 2D-editor rules):
//!   wheel                zoom about the cursor
//!   ctrl+wheel           pan sideways
//!   shift+wheel          pan vertically
//!   trackpad two-finger  pan (the canvas follows the fingers)
//!   ctrl+trackpad        zoom
//!   pinch                zoom
//!
//! Every canvas — node graph, viewer, curve editors, patch bay — routes
//! through these two classifiers, so the hands are identical everywhere.
//! Canvases keep their own coordinate maths; this module only answers
//! "what does this input MEAN".

use egui::{Modifiers, PointerButton, Vec2};

/// What a drag with this button + modifier chord is, if it is navigation
/// at all. `None` = not nav; the canvas's own gestures may have it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DragNav {
    Pan,
    Zoom,
}

/// Classify a drag. Pure, so the table above is testable without a window.
pub fn drag_nav(button: PointerButton, m: Modifiers) -> Option<DragNav> {
    let ctrl = m.ctrl || m.command;
    match button {
        PointerButton::Middle => Some(if ctrl { DragNav::Zoom } else { DragNav::Pan }),
        PointerButton::Primary if m.alt => {
            Some(if ctrl { DragNav::Zoom } else { DragNav::Pan })
        }
        // The dolly we always had. Kept: it is also how macOS delivers
        // ctrl(+alt)+LMB, and Houdini hands are welcome where they don't
        // collide with Blender's.
        PointerButton::Secondary if m.alt || m.command => Some(DragNav::Zoom),
        _ => None,
    }
}

/// Multiplicative zoom for a drag-zoom step. Up or right zooms in: the
/// vertical axis is Blender's dolly, the horizontal is the Houdini habit —
/// they never disagree, because each gesture only moves one of them.
pub fn drag_zoom(delta: Vec2) -> f32 {
    ((delta.x - delta.y) * 0.005).exp()
}

/// Per-axis drag-zoom, for the curve editors: Blender's graph editor
/// stretches the view axis by axis under ctrl+MMB — drag right to stretch
/// time, up to stretch values.
pub fn drag_zoom_xy(delta: Vec2) -> (f32, f32) {
    ((delta.x * 0.005).exp(), (-delta.y * 0.005).exp())
}

/// What the wheel/trackpad meant this frame: screen-pixel pan plus a
/// multiplicative zoom (1.0 = none). `alt_wheel_zoom_y`: the curve editors
/// keep their old alt+wheel = value-axis-only zoom, which Blender does not
/// use for anything — those canvases split the returned zoom themselves,
/// so it is reported, not interpreted, here.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct WheelNav {
    pub pan: Vec2,
    pub zoom: f32,
    /// Alt was held on the zooming wheel events (for the curve editors'
    /// value-axis-only zoom). Meaningless when `zoom == 1`.
    pub alt: bool,
}

/// One wheel notch ("line") in pixels. ~16% zoom per notch lands where
/// Blender's per-notch step feels.
const LINE_PX: f32 = 50.0;
const WHEEL_ZOOM: f32 = 0.003;

/// Two fingers over a 3D eye, as Blender's 3D view reads them: a swipe
/// ORBITS, shift+swipe pans (and so do ctrl/shift on a wheel, by axis).
/// Zoom (pinch, ctrl+swipe, a wheel notch) is `classify_wheel`'s, unchanged.
#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct Eye3d {
    pub orbit: Vec2,
    pub pan: Vec2,
}

impl Eye3d {
    #[cfg(test)]
    fn is_none(&self) -> bool {
        self.orbit == Vec2::ZERO && self.pan == Vec2::ZERO
    }
}

/// `classify_wheel`'s 2D pan, re-read for a 3D eye. Pure, like it.
pub fn classify_eye3d(events: &[egui::Event]) -> Eye3d {
    let mut out = Eye3d::default();
    for e in events {
        if let egui::Event::MouseWheel { unit, delta, modifiers, .. } = e {
            let ctrl = modifiers.ctrl || modifiers.command;
            if matches!(unit, egui::MouseWheelUnit::Point) {
                if ctrl {
                    // ctrl + two fingers zooms (classify_wheel).
                } else if modifiers.shift {
                    out.pan += *delta;
                } else {
                    out.orbit += *delta;
                }
            } else if ctrl {
                out.pan.x += delta.y * LINE_PX;
            } else if modifiers.shift {
                out.pan.y += delta.y * LINE_PX;
            }
        }
    }
    out
}

/// The 3D eye's two-finger read for this frame's events.
pub fn eye3d_nav(ui: &egui::Ui) -> Eye3d {
    ui.input(|i| classify_eye3d(&i.events))
}

/// Classify raw wheel/pinch events. Pure for the same reason as `drag_nav`.
pub fn classify_wheel(events: &[egui::Event]) -> WheelNav {
    let mut out = WheelNav { pan: Vec2::ZERO, zoom: 1.0, alt: false };
    for e in events {
        match e {
            egui::Event::MouseWheel { unit, delta, modifiers, .. } => {
                let line = !matches!(unit, egui::MouseWheelUnit::Point);
                let px = if line { *delta * LINE_PX } else { *delta };
                let ctrl = modifiers.ctrl || modifiers.command;
                if line {
                    if ctrl {
                        // ctrl+wheel: sideways. A wheel only has a y axis,
                        // so its clicks become x travel.
                        out.pan.x += px.y;
                    } else if modifiers.shift {
                        out.pan.y += px.y;
                    } else {
                        out.zoom *= (px.y * WHEEL_ZOOM).exp();
                        out.alt |= modifiers.alt;
                    }
                } else if ctrl {
                    // ctrl + two fingers: zoom, per Blender (and macOS).
                    out.zoom *= (px.y * WHEEL_ZOOM).exp();
                    out.alt |= modifiers.alt;
                } else {
                    // Two fingers move the canvas. Both axes, unscaled.
                    out.pan += px;
                }
            }
            egui::Event::Zoom(f) => {
                out.zoom *= f;
            }
            _ => {}
        }
    }
    out
}

/// The frame's wheel/trackpad navigation for a hovered canvas. Reads RAW
/// events on purpose: egui's `smooth_scroll_delta`/`zoom_delta` bake in
/// their own ctrl-scroll-is-zoom opinion, which is not Blender's.
pub fn wheel_nav(ui: &egui::Ui) -> WheelNav {
    // A number field stepping under the wheel owns it (`numfield`).
    if crate::numfield::wheel_claimed(ui.ctx()) {
        return WheelNav { pan: Vec2::ZERO, zoom: 1.0, alt: false };
    }
    ui.input(|i| classify_wheel(&i.events))
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::{Event, MouseWheelUnit};

    fn wheel(unit: MouseWheelUnit, d: [f32; 2], m: Modifiers) -> Event {
        Event::MouseWheel {
            unit,
            delta: egui::vec2(d[0], d[1]),
            modifiers: m,
            phase: egui::TouchPhase::Move,
        }
    }

    #[test]
    fn the_drag_table_is_blenders() {
        use DragNav::*;
        use PointerButton::*;
        let m = |ctrl: bool, alt: bool, shift: bool, cmd: bool| Modifiers {
            ctrl,
            alt,
            shift,
            command: cmd,
            mac_cmd: false,
        };
        assert_eq!(drag_nav(Middle, m(false, false, false, false)), Some(Pan));
        assert_eq!(drag_nav(Middle, m(false, false, true, false)), Some(Pan));
        assert_eq!(drag_nav(Middle, m(true, false, false, false)), Some(Zoom));
        assert_eq!(drag_nav(Primary, m(false, true, false, false)), Some(Pan));
        assert_eq!(drag_nav(Primary, m(false, true, true, false)), Some(Pan));
        assert_eq!(drag_nav(Primary, m(true, true, false, false)), Some(Zoom));
        // Plain, shift and bare-ctrl left-drags belong to the canvas
        // (select, scrub, slice) — nav must not eat them.
        assert_eq!(drag_nav(Primary, m(false, false, false, false)), None);
        assert_eq!(drag_nav(Primary, m(false, false, true, false)), None);
        assert_eq!(drag_nav(Primary, m(true, false, false, false)), None);
        // The legacy dolly.
        assert_eq!(drag_nav(Secondary, m(false, true, false, false)), Some(Zoom));
        assert_eq!(drag_nav(Secondary, m(false, false, false, true)), Some(Zoom));
        assert_eq!(drag_nav(Secondary, m(false, false, false, false)), None);
    }

    #[test]
    fn over_a_3d_eye_two_fingers_orbit_like_blender() {
        let none = Modifiers::default();
        let shift = Modifiers { shift: true, ..Default::default() };
        let ctrl = Modifiers { ctrl: true, ..Default::default() };
        let swipe = |m| classify_eye3d(&[wheel(MouseWheelUnit::Point, [30.0, -12.0], m)]);
        assert_eq!(swipe(none), Eye3d { orbit: egui::vec2(30.0, -12.0), pan: Vec2::ZERO });
        assert_eq!(swipe(shift), Eye3d { orbit: Vec2::ZERO, pan: egui::vec2(30.0, -12.0) });
        // ctrl+swipe is a zoom, and classify_wheel says so.
        assert!(swipe(ctrl).is_none());
        let z = classify_wheel(&[wheel(MouseWheelUnit::Point, [0.0, 20.0], ctrl)]);
        assert!(z.zoom > 1.0);
        // A wheel notch still only zooms; ctrl/shift on it pan by axis.
        assert!(classify_eye3d(&[wheel(MouseWheelUnit::Line, [0.0, 1.0], none)]).is_none());
        assert_eq!(classify_eye3d(&[wheel(MouseWheelUnit::Line, [0.0, 1.0], shift)]).pan, egui::vec2(0.0, LINE_PX));
        // A pinch is zoom only.
        assert!(classify_eye3d(&[egui::Event::Zoom(1.1)]).is_none());
    }

    #[test]
    fn a_wheel_zooms_and_a_trackpad_pans() {
        let none = Modifiers::default();
        let w = classify_wheel(&[wheel(MouseWheelUnit::Line, [0.0, 1.0], none)]);
        assert_eq!(w.pan, Vec2::ZERO);
        assert!(w.zoom > 1.1, "a notch up zooms in, got {}", w.zoom);
        let t = classify_wheel(&[wheel(MouseWheelUnit::Point, [30.0, -12.0], none)]);
        assert_eq!(t.zoom, 1.0);
        assert_eq!(t.pan, egui::vec2(30.0, -12.0));
    }

    #[test]
    fn modifiers_reroute_the_wheel() {
        let ctrl = Modifiers { ctrl: true, ..Default::default() };
        let shift = Modifiers { shift: true, ..Default::default() };
        let w = classify_wheel(&[wheel(MouseWheelUnit::Line, [0.0, 2.0], ctrl)]);
        assert_eq!(w.zoom, 1.0);
        assert!(w.pan.x > 0.0 && w.pan.y == 0.0, "ctrl+wheel pans sideways");
        let w = classify_wheel(&[wheel(MouseWheelUnit::Line, [0.0, 2.0], shift)]);
        assert_eq!(w.zoom, 1.0);
        assert!(w.pan.y > 0.0 && w.pan.x == 0.0, "shift+wheel pans vertically");
        let w = classify_wheel(&[wheel(MouseWheelUnit::Point, [0.0, 40.0], ctrl)]);
        assert!(w.zoom > 1.0 && w.pan == Vec2::ZERO, "ctrl+trackpad zooms");
        let w = classify_wheel(&[Event::Zoom(1.3)]);
        assert!((w.zoom - 1.3).abs() < 1e-6, "pinch is zoom");
    }

    #[test]
    fn drag_zoom_curves_match_the_axes() {
        assert!(drag_zoom(egui::vec2(0.0, -80.0)) > 1.0, "up zooms in");
        assert!(drag_zoom(egui::vec2(80.0, 0.0)) > 1.0, "right zooms in");
        let (fx, fy) = drag_zoom_xy(egui::vec2(80.0, 0.0));
        assert!(fx > 1.0 && (fy - 1.0).abs() < 1e-6, "right stretches time only");
        let (fx, fy) = drag_zoom_xy(egui::vec2(0.0, -80.0));
        assert!(fy > 1.0 && (fx - 1.0).abs() < 1e-6, "up stretches value only");
    }
}

/// One frame of a 1-D bar's navigation (a time bar, a range bar): the
/// wheel and trackpad rules above, plus the drag chords (MMB / alt+LMB pan,
/// ctrl+MMB / ctrl+alt+LMB zoom about the press). `view` is the frames the
/// bar spans, left edge to right edge. Returns true while a nav drag holds
/// the pointer, so the bar's own gestures (scrub, span drag) stand down.
///
/// The drag is read off raw pointer state, so it works over any widget the
/// bar is made of; the wheel is claimed (`curveui::claim_wheel`) so the
/// panel around the bar neither scrolls nor zooms under it.
pub fn bar_nav(ui: &egui::Ui, id: egui::Id, rect: egui::Rect, view: &mut (f64, f64)) -> bool {
    #[derive(Clone, Copy)]
    struct Held {
        nav: DragNav,
        button: PointerButton,
        press_x: f32,
        last: egui::Pos2,
    }
    let n = (view.1 - view.0 + 1.0).max(1.0);
    let px_per = rect.width().max(1.0) as f64 / n;
    let zoom_about = |view: &mut (f64, f64), x: f32, z: f64| {
        let n = (view.1 - view.0 + 1.0).max(1.0);
        let at = view.0 + ((x - rect.min.x) as f64 / rect.width().max(1.0) as f64) * n;
        let n2 = (n / z).clamp(4.0, 1.0e8);
        let r = (at - view.0) / n;
        view.0 = at - r * n2;
        view.1 = view.0 + n2 - 1.0;
    };
    let (hover, mods, pressed, latest) = ui.input(|i| {
        let pressed = [PointerButton::Primary, PointerButton::Middle, PointerButton::Secondary]
            .into_iter()
            .find(|b| i.pointer.button_pressed(*b));
        (i.pointer.hover_pos(), i.modifiers, pressed, i.pointer.latest_pos())
    });
    let over = hover.is_some_and(|p| rect.contains(p));
    let mut held: Option<Held> = ui.data(|d| d.get_temp(id));
    if held.is_none() && over {
        if let (Some(b), Some(p)) = (pressed, hover) {
            if let Some(nav) = drag_nav(b, mods) {
                held = Some(Held { nav, button: b, press_x: p.x, last: p });
            }
        }
    }
    if let Some(mut h) = held {
        if ui.input(|i| i.pointer.button_down(h.button)) {
            if let Some(p) = latest {
                let d = p - h.last;
                match h.nav {
                    DragNav::Pan => {
                        view.0 -= d.x as f64 / px_per;
                        view.1 -= d.x as f64 / px_per;
                    }
                    DragNav::Zoom => zoom_about(view, h.press_x, drag_zoom(d) as f64),
                }
                h.last = p;
            }
            ui.ctx().set_cursor_icon(match h.nav {
                DragNav::Pan => egui::CursorIcon::Grabbing,
                DragNav::Zoom => egui::CursorIcon::ResizeHorizontal,
            });
            ui.data_mut(|d| d.insert_temp(id, h));
            return true;
        }
        ui.data_mut(|d| d.remove::<Held>(id));
        return false;
    }
    if over {
        crate::curveui::claim_wheel(ui, rect);
        let w = wheel_nav(ui);
        let pan = (w.pan.x + w.pan.y) as f64;
        if pan != 0.0 {
            view.0 -= pan / px_per;
            view.1 -= pan / px_per;
        }
        if w.zoom != 1.0 {
            if let Some(p) = hover {
                zoom_about(view, p.x, w.zoom as f64);
            }
        }
    }
    false
}
