//! The Layer stack's UI state: the selected layer per node, the pop-out, the bars and the drags on them.
//! These were loose fields on `Gfx`; grouped, not changed.

#[allow(unused_imports)] // a group of plain types needs nothing from the crate
use crate::*;

/// The Layer stack's UI state: the selected layer per node, the pop-out, the bars and the drags on them.
pub(crate) struct LayerUiState {
    /// Each Layer node's wired inputs last frame, for closing up gaps.
    pub(crate) wired_prev: std::collections::HashMap<u64, Vec<bool>>,
    /// Which layer of each Layer stack is selected: the one the viewer gives
    /// a manipulator and the one the panel's controls edit. Per node, so
    /// moving between two stacks doesn't lose your place. UI state, like
    /// `prop_order` — it is not authored data and never reaches a setup.
    pub(crate) sel: layerui::LayerSel,
    /// --test-layer: a screen point captured on the frame the gesture that
    /// uses it was planned, so a settling panel cannot move it mid-drag.
    pub(crate) grab: egui::Pos2,
    /// The layer stack popped out into its own resizable window, if any.
    /// UI state like `prop_order` — never authored, never saved.
    pub(crate) popout: Option<u64>,
    /// Where the layer timeline's bars landed last frame, one per layer.
    pub(crate) bars: Vec<egui::Rect>,
    /// --test-layer: the start the slid layer reached, and the wire it
    /// arrived on — both checked again after the restack.
    pub(crate) slid: i64,
    pub(crate) wire1: u64,
    pub(crate) speed: f32,
    /// `--test-layer`: reader slots one plate on two layers landed on.
    pub(crate) slots: usize,
}
