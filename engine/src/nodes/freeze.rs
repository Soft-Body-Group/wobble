//! Freeze: hold the input on one comp frame, every tick.
//!
//! Not a shader effect. Like a Slip it remaps which frame the readers in
//! its input subtree decode — here with a speed of zero, so every output
//! tick reads the input at `frame`. The pixels flow through unchanged and
//! the node costs nothing to render (one decode a tick, the held frame's).
//!
//! `frame` is a COMP frame (the numbering of the playhead, keyframes and
//! placement), keyable like any parameter. Keyed, the app samples the curve
//! into `lut` the way it does a Time Curve's, because time has to be
//! answerable at every tick a reader prefetches, not just the one on
//! screen — so a keyed freeze is a hold that jumps (stepped keys) or slides
//! (smooth keys) between held frames. Moving the frame is a time-map
//! change, never a recompile: the node's `topo_tag` is a constant.

use crate::graph::TimeMap;

#[derive(Clone, Debug, PartialEq, Default)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct FreezeParams {
    /// The comp frame every tick shows.
    pub frame: f32,
    /// TRANSIENT: the animated `frame` sampled into (tick, frame) points by
    /// the app. Empty = unkeyed, the single authored frame.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub lut: Vec<[f32; 2]>,
}

impl FreezeParams {
    pub const NAME: &'static str = "Freeze";
    pub const INPUTS: usize = 1;

    pub fn at(frame: f32) -> Self {
        Self { frame, lut: Vec::new() }
    }

    /// Every tick to `frame` (or the sampled curve when keyed).
    pub fn map(&self) -> TimeMap {
        if self.lut.is_empty() {
            TimeMap::lut(vec![[0.0, self.frame]])
        } else {
            TimeMap::lut(self.lut.clone())
        }
    }

    /// Structural params only: none. The frame is a time map.
    pub fn topo_tag(&self) -> u64 {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_tick_reads_the_held_frame() {
        let m = FreezeParams::at(1012.0).map();
        for t in [0u64, 1, 1000, 1012, 1013, 5000] {
            assert_eq!(m.apply(t), 1012, "tick {t}");
        }
        // A fractional frame rounds like any reader time.
        assert_eq!(FreezeParams::at(7.6).map().apply(3), 8);
    }

    #[test]
    fn a_sampled_curve_wins_over_the_authored_frame() {
        let p = FreezeParams { frame: 3.0, lut: vec![[0.0, 10.0], [10.0, 10.0], [11.0, 20.0], [20.0, 20.0]] };
        assert_eq!(p.map().apply(5), 10);
        assert_eq!(p.map().apply(15), 20);
    }

    #[test]
    fn the_frame_never_reaches_the_topology() {
        assert_eq!(FreezeParams::at(1.0).topo_tag(), FreezeParams::at(99.0).topo_tag());
    }
}
