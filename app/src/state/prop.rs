//! The properties rack's state: the order of its panels, its zoom, stretch and scroll, and the drags on it.
//! These were loose fields on `Gfx`; grouped, not changed.

#[allow(unused_imports)] // a group of plain types needs nothing from the crate
use crate::*;

/// The properties rack's state: the order of its panels, its zoom, stretch and scroll, and the drags on it.
pub(crate) struct PropState {
    /// Named property-panel arrangements (which nodes are pinned, in what
    /// order). Saved with the setup, so a comp can carry "grade pass",
    /// "roto pass" and so on.
    pub(crate) layouts: Vec<PropLayout>,
    /// Name field for saving the current arrangement.
    pub(crate) layout_name: String,
    /// Properties zoom: scales the tile and the controls whose precision
    /// depends on size. Text and knobs stay put.
    pub(crate) zoom: f32,
    /// Per-axis stretch of the TILE BOX, on top of `prop_zoom`: widgets keep
    /// their scale, so x widens slider tracks and y shows rows a tile was
    /// cutting off. The tiles stop being square; nothing inside cares.
    pub(crate) stretch: egui::Vec2,
    /// The grid's scroll offset as of last frame, an override to apply on
    /// the next frame, the zoom the grid was last laid out at, and the
    /// screen point zoom should keep still (None = centre of the view).
    pub(crate) scroll: egui::Vec2,
    pub(crate) scroll_set: Option<egui::Vec2>,
    pub(crate) zoom_laid: f32,
    pub(crate) stretch_laid: egui::Vec2,
    /// The docked strip is STACKED: its pane is taller than it is wide (the
    /// dock put it down a side), so panels run top to bottom and the strip
    /// scrolls vertically. Decided per frame from the pane's shape, with a
    /// little hysteresis so a near-square pane does not flip back and forth.
    pub(crate) vertical: bool,
    /// A ctrl+MMB / ctrl+alt+LMB stretch drag over the strip: its press point.
    pub(crate) stretch_anchor: Option<egui::Pos2>,
    pub(crate) zoom_anchor: Option<egui::Pos2>,
    /// The zoom changed but has not been written to prefs yet.
    pub(crate) zoom_dirty: bool,
    /// Screen position of the grid's content origin last frame — the scroll
    /// area insets it from the panel rect, so anchoring measures from here.
    pub(crate) origin: egui::Pos2,
    /// Left-to-right arrangement of the properties panels (drag headers to
    /// reorder; panels snap to the tile grid).
    pub(crate) order: Vec<u64>,
    pub(crate) drag: Option<u64>,
    /// The panel whose title was clicked: the toolbar's `w` sizes it alone.
    pub(crate) selected: Option<u64>,
    /// Per-panel tile width stretch, replacing `prop_stretch.x` for that
    /// panel in the docked strip. Session state, keyed by node.
    pub(crate) width: std::collections::HashMap<u64, f32>,
    /// A panel's right edge being dragged: (node, press x, width at press).
    pub(crate) edge_drag: Option<(u64, f32, f32)>,
    /// The docked strip's panel rects as of last frame (suites aim at them).
    pub(crate) panel_rects: Vec<(u64, egui::Rect)>,
}
