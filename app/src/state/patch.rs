//! The patch bay's state: where each panel sits, the canvas pan and zoom, the cable in hand.
//! These were loose fields on `Gfx`; grouped, not changed.

#[allow(unused_imports)] // a group of plain types needs nothing from the crate
use crate::*;

/// The patch bay's state: where each panel sits, the canvas pan and zoom, the cable in hand.
pub(crate) struct PatchState {
    /// Patch bay (the detached properties window): grid cell + whole-tile
    /// span per panel. Geometry is remembered while a panel is closed.
    pub(crate) cell: std::collections::BTreeMap<u64, [i32; 2]>,
    /// The footprint each panel was placed for, so a node whose span
    /// changes (a resized tool) gets a fresh cell instead of overlapping.
    pub(crate) span: std::collections::BTreeMap<u64, [u8; 2]>,
    /// In-flight patch drag: (node, grab offset).
    pub(crate) drag: Option<(u64, egui::Vec2)>,
    /// Link mode: the patch bay hides widgets and exposes every float value
    /// as in/out jacks for expression wiring.
    pub(crate) link_mode: bool,
    /// Canvas-relative spot the add-value-node menu was opened at.
    pub(crate) menu_pos: egui::Vec2,
    /// Everything else the patch canvas remembers between frames —
    /// selection, slicer trail, marquee, search palette. See `patchbay`.
    pub(crate) ui: patchbay::PatchUi,
    /// The patch canvas's own view, exactly like `NodeEditor`'s: a free 2D
    /// pan in screen points with NO bounds, so you can push the arrangement
    /// off into empty space the way you can in the graph. (The docked
    /// properties strip is a strip and keeps its scroll area.)
    pub(crate) pan: egui::Vec2,
    /// Houdini-style dolly in flight: alt + right-drag zooms about this
    /// press point.
    pub(crate) zoom_anchor: Option<egui::Pos2>,
    /// A pan is in flight. Held as state so the gesture survives the pointer
    /// crossing a panel or leaving the canvas, as it does in the graph.
    pub(crate) panning: bool,
}
