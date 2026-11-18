//! Colour management, copied from wobble's engine so Scrub matches it
//! pixel for pixel without depending on it: `color` (primaries, curves,
//! colourspaces, input spaces and their WGSL transforms), `viewtf` (view
//! transforms: ACES 1.3 and 1.2, AgX from Blender's cubes, the filmic fit,
//! none; OpenDRT is wobble's GPLv3 feature and only its stub is here) and
//! `lut` (the cube reader AgX uses). Scrub always has parity with wobble's
//! colour: a change there is ported here in the same work, and
//! `tests/parity.rs` fails until it is.

#![allow(clippy::all)]

pub mod color;
pub mod lut;
pub mod viewtf;
