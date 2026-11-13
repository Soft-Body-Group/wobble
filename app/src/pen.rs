//! Pen input: where the pointer went between frames, and how hard.
//!
//! egui hands the app ONE pointer position per frame and no pressure at
//! all, which is fine for a mouse and wrong for a brush: a quick pen stroke
//! moves tens of pixels a frame, and the tablet reports at 200Hz+. So this
//! keeps every cursor sample the window saw (in physical pixels, exactly
//! what winit reported) together with the pen pressure in force when it
//! arrived, and the brush drains them once a frame.
//!
//! Pressure reaches it from whichever source the platform has — and only
//! from here, so the brush never learns which one it was:
//!
//! * **macOS** — winit drops tablet pressure, so an `NSEvent` local monitor
//!   (`install`) reads it off the tablet-subtype mouse events and
//!   `NSEventTypeTabletPoint`. That is Wacom, and Apple Pencil on an iPad
//!   over Sidecar, which arrives as a tablet.
//! * **Windows** — winit's `WM_POINTER` handling reports a pen as `Touch`
//!   with a normalised force (`note_touch`).
//! * **A browser** (an iPad one day) — a `pointermove` listener on the
//!   canvas reads `PointerEvent.pressure` for `pointerType == "pen"`
//!   (`webmain`), because winit's web backend turns a pen into a mouse.
//! * **Linux** — winit 0.30 has no tablet support: no pressure, strokes
//!   paint at full.
//!
//! `None` pressure means "not a pen" (a mouse, or a pen out of proximity):
//! a stroke started that way records no samples at all and paints as it
//! always did.

use std::sync::Mutex;

/// One cursor sample: physical pixels in the window it landed in, and the
/// pen pressure (0..1) in force, `None` off a pen.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sample {
    pub phys: [f32; 2],
    pub pressure: Option<f32>,
}

impl Sample {
    /// In egui points, for the context the brush runs in.
    pub fn pos(&self, pixels_per_point: f32) -> egui::Pos2 {
        egui::pos2(self.phys[0] / pixels_per_point, self.phys[1] / pixels_per_point)
    }
}

struct State {
    samples: Vec<Sample>,
    pressure: Option<f32>,
}

static STATE: Mutex<State> = Mutex::new(State { samples: Vec::new(), pressure: None });

/// Samples kept between drains. A frame at 60Hz from a 1kHz mouse is ~17;
/// the cap only matters while nothing is painting and nobody drains.
const CAP: usize = 512;

fn with<R>(f: impl FnOnce(&mut State) -> R) -> R {
    f(&mut STATE.lock().unwrap_or_else(|e| e.into_inner()))
}

/// The pen pressure now (`None` off a pen) — for the panel's meter and a
/// stroke's first point.
pub fn pressure() -> Option<f32> {
    with(|s| s.pressure)
}

/// A platform hook saw the pen (or saw it leave).
pub fn set_pressure(p: Option<f32>) {
    with(|s| s.pressure = p.map(|p| p.clamp(0.0, 1.0)));
}

/// winit's `CursorMoved`, before egui sees it.
pub fn note_cursor(x: f64, y: f64) {
    with(|s| {
        if s.samples.len() >= CAP {
            s.samples.drain(..CAP / 2);
        }
        let pressure = s.pressure;
        s.samples.push(Sample { phys: [x as f32, y as f32], pressure });
    });
}

/// winit's `Touch`: a pen on Windows carries its pressure as the force.
/// A finger with no force is left to egui alone.
pub fn note_touch(t: &winit::event::Touch) {
    use winit::event::{Force, TouchPhase};
    let Some(force) = t.force else { return };
    let p = match force {
        Force::Normalized(f) => f as f32,
        Force::Calibrated { force, max_possible_force, .. } if max_possible_force > 0.0 => {
            (force / max_possible_force) as f32
        }
        Force::Calibrated { .. } => return,
    };
    match t.phase {
        TouchPhase::Ended | TouchPhase::Cancelled => set_pressure(None),
        _ => {
            set_pressure(Some(p));
            note_cursor(t.location.x, t.location.y);
        }
    }
}

/// A sample from somewhere other than this window's cursor: the remote
/// screen's Pencil (`apiserve::remote`), or the window's own samples put
/// back after a remote pass took the queue for itself.
pub fn push(sample: Sample) {
    with(|s| {
        if s.samples.len() >= CAP {
            s.samples.drain(..CAP / 2);
        }
        s.samples.push(sample);
    });
}

/// Every sample since the last drain, oldest first.
pub fn drain() -> Vec<Sample> {
    with(|s| std::mem::take(&mut s.samples))
}

/// macOS: read tablet pressure that winit drops. Installed once, for the
/// life of the app; the monitor hands every event back untouched.
#[cfg(target_os = "macos")]
pub fn install() {
    use objc2::runtime::{AnyClass, AnyObject};
    use objc2::msg_send;
    use std::sync::Once;

    // NSEventType values.
    const LEFT_DOWN: u64 = 1;
    const LEFT_UP: u64 = 2;
    const MOVED: u64 = 5;
    const LEFT_DRAGGED: u64 = 6;
    const TABLET_POINT: u64 = 23;
    const TABLET_PROXIMITY: u64 = 24;
    // NSEventSubtypeTabletPoint / TabletProximity on a mouse event.
    const SUB_TABLET_POINT: i16 = 1;
    const SUB_TABLET_PROXIMITY: i16 = 2;

    static ONCE: Once = Once::new();
    ONCE.call_once(|| unsafe {
        let mask: u64 = [LEFT_DOWN, LEFT_UP, MOVED, LEFT_DRAGGED, TABLET_POINT, TABLET_PROXIMITY]
            .iter()
            .fold(0, |m, t| m | (1u64 << t));
        let block = block2::RcBlock::new(|ev: *mut AnyObject| -> *mut AnyObject {
            if let Some(e) = ev.as_ref() {
                let ty: u64 = msg_send![e, type];
                if ty == TABLET_PROXIMITY {
                    let entering: bool = msg_send![e, isEnteringProximity];
                    if !entering {
                        set_pressure(None);
                    }
                } else if ty == TABLET_POINT {
                    let p: f32 = msg_send![e, pressure];
                    set_pressure(Some(p));
                } else {
                    let sub: i16 = msg_send![e, subtype];
                    match sub {
                        SUB_TABLET_POINT => {
                            let p: f32 = msg_send![e, pressure];
                            // A hover reads 0, and a lift (mouse up) too:
                            // the pen is still there, just not pressing.
                            set_pressure(Some(if ty == LEFT_UP { 0.0 } else { p }));
                        }
                        SUB_TABLET_PROXIMITY => {}
                        // A plain mouse event: whatever the pen said is over.
                        _ => set_pressure(None),
                    }
                }
            }
            ev
        });
        let Some(cls) = AnyClass::get(c"NSEvent") else { return };
        let monitor: *mut AnyObject =
            msg_send![cls, addLocalMonitorForEventsMatchingMask: mask, handler: &*block];
        // Kept for the life of the app: the monitor is never removed.
        let _ = monitor;
        std::mem::forget(block);
    });
}

#[cfg(not(target_os = "macos"))]
pub fn install() {}

#[cfg(test)]
mod tests {
    use super::*;

    /// One test touches the global, so the sequence is the claim: a cursor
    /// sample takes the pressure in force, a pen leaving clears it, and a
    /// drain empties the queue in order.
    #[test]
    fn samples_carry_the_pressure_in_force_and_drain_in_order() {
        drain();
        set_pressure(None);
        note_cursor(10.0, 20.0);
        set_pressure(Some(0.4));
        note_cursor(12.0, 22.0);
        set_pressure(Some(1.7)); // clamped
        note_cursor(14.0, 24.0);
        set_pressure(None);
        let got = drain();
        assert_eq!(got.len(), 3);
        assert_eq!(got[0].pressure, None);
        assert_eq!(got[1].pressure, Some(0.4));
        assert_eq!(got[2].pressure, Some(1.0));
        assert_eq!(got[2].pos(2.0), egui::pos2(7.0, 12.0));
        assert!(drain().is_empty());
        for i in 0..CAP + 10 {
            note_cursor(i as f64, 0.0);
        }
        let got = drain();
        assert!(got.len() <= CAP);
        assert_eq!(got.last().unwrap().phys[0], (CAP + 9) as f32, "the newest are kept");
    }
}
