//! Stand-in for the OpenDRT module in builds without the `opendrt` feature.
//!
//! The real port is GPLv3 (see `viewtf/opendrt.rs`); this keeps the rest of
//! the view-transform plumbing — and setups saved by a GPL build — compiling
//! and loading in a default, permissively licensed build. Selecting OpenDRT
//! here resolves to Filmic via [`super::ViewTransform::effective`].

use super::Gamut;

/// Same slot count as the real module so the uniform layout never moves.
pub const SLOTS: usize = 18;

/// Placeholder parameter block. Accepts (and discards) a real build's saved
/// OpenDRT settings, so a setup round-trips between build flavours.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct OpenDrtParams {}

pub fn pack(_p: &OpenDrtParams, _display: Gamut) -> [[f32; 4]; SLOTS] {
    [[0.0; 4]; SLOTS]
}

pub fn display_matrices(_p: &OpenDrtParams, _display: Gamut) -> ([[f32; 3]; 3], [[f32; 3]; 3]) {
    (super::IDENTITY3, super::IDENTITY3)
}

/// Unreachable in practice — `effective()` never resolves to OpenDRT without
/// the feature — but it has to produce a valid entry point regardless.
pub fn wgsl(_display: Gamut) -> String {
    String::new()
}
