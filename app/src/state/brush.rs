//! The Paint brush: its shape, colour, mode, pressure map and presets.
//! These were loose fields on `Gfx`; grouped, not changed.

use crate::*;

/// The Paint brush: its shape, colour, mode, pressure map and presets.
pub(crate) struct BrushState {
    /// Brush settings for the Paint node (global, not per-node).
    pub(crate) radius: f32,
    pub(crate) soft: f32,
    pub(crate) color: [f32; 4],
    /// What a stroke does — see `BrushMode`. Clone keeps its anchor below.
    pub(crate) mode: engine::graph::BrushMode,
    /// Ellipse: minor/major ratio (1 = round) and rotation in degrees.
    pub(crate) aspect: f32,
    pub(crate) angle: f32,
    /// Each mode's second knob, kept PER MODE so switching brushes never
    /// hands a dodge the blur's radius: blur px, smear drag px, dodge/burn
    /// stops.
    pub(crate) blur_px: f32,
    pub(crate) smear_px: f32,
    pub(crate) stops: f32,
    /// How a colour stroke lands (`PaintBlend`) and what the pen's pressure
    /// does to it. Both are copied onto each stroke as it starts.
    pub(crate) blend: engine::graph::PaintBlend,
    pub(crate) press: engine::graph::PressureMap,
    /// The preset row (`brushes.rs`), mirrored into prefs on change.
    pub(crate) presets: Vec<brushes::BrushPreset>,
    /// The alt-clicked SOURCE anchor, normalized. Setting it re-derives the
    /// offset on the next stroke; after that the offset is aligned (it
    /// stays put stroke to stroke, Photoshop-style) until the next alt-click.
    pub(crate) clone_src: Option<[f32; 2]>,
    /// The live offset, normalized — what actually goes on a stroke.
    pub(crate) clone_off: Option<[f32; 2]>,
}
