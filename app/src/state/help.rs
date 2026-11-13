//! The hotkey help: open or not, what is hovered, where it is.
//! These were loose fields on `Gfx`; grouped, not changed.

#[allow(unused_imports)] // a group of plain types needs nothing from the crate
use crate::*;

/// The hotkey help: open or not, what is hovered, where it is.
pub(crate) struct HelpState {
    pub(crate) open: bool,
    /// The pointer is over the hotkeys window: a key pressed there finds
    /// its row instead of firing (last frame's answer, read by the hotkeys).
    pub(crate) hover: bool,
    /// Where the hotkeys window is (suites aim at it).
    pub(crate) rect: Option<egui::Rect>,
    /// The hotkeys row a key press jumped to: (key, which match, row, when).
    pub(crate) jump: Option<(egui::Key, usize, usize, Instant)>,
}
