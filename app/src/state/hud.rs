//! The viewer HUD's state: where it sits, its scale, what is hidden, and the slider being dragged out.
//! These were loose fields on `Gfx`; grouped, not changed.

#[allow(unused_imports)] // a group of plain types needs nothing from the crate
use crate::*;

/// The viewer HUD's state: where it sits, its scale, what is hidden, and the slider being dragged out.
pub(crate) struct HudState {
    /// Grade viewer-HUD positions (offset from image rect min), per-axis
    /// control scale, and hidden set.
    pub(crate) pos: HashMap<u64, egui::Vec2>,
    pub(crate) scale: HashMap<u64, egui::Vec2>,
    pub(crate) hidden: std::collections::HashSet<u64>,
    pub(crate) luma: HashMap<u64, f32>,
    /// HUDs that opted OUT of the frosted background (default = frosted).
    pub(crate) no_frost: std::collections::HashSet<u64>,
    pub(crate) ui_open: bool,
    /// The viewer hotkey strip's words, 0 = tucked into their caps.
    pub(crate) keys_reveal: f32,
    /// A HUD spawn in flight (following the cursor until release).
    pub(crate) drag: Option<widgets::HudSlider>,
    /// --test-click hud-off phase: translate.x before the inert drag.
    pub(crate) off_tx0: f32,
}
