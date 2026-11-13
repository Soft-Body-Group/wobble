//! The CornerPin gizmo's state: the pick, the grab and the marquee.
//! These were loose fields on `Gfx`; grouped, not changed.

#[allow(unused_imports)] // a group of plain types needs nothing from the crate
use crate::*;

/// The CornerPin gizmo's state: the pick, the grab and the marquee.
pub(crate) struct PinState {
    /// The CornerPin whose pins are picked, and which of its four are.
    /// One pin node owns the pick at a time, like `spline_sel_node`.
    pub(crate) sel: Option<(u64, [bool; 4])>,
    /// CornerPins whose viewer handles edit the `from` quad rather than the
    /// pin. Presentation only (a toggle in the panel), so not in the setup.
    pub(crate) from: std::collections::HashSet<u64>,
    /// A pin gesture in flight, snapshotted at its press (see `PinGrab`).
    pub(crate) grab: Option<PinGrab>,
    /// A background drag banding pins, in screen points.
    pub(crate) marquee: Option<egui::Rect>,
    /// Where the pin box's rotate knob was drawn last (for the suite).
    pub(crate) knob: Option<egui::Pos2>,
}
