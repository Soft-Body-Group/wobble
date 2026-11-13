//! MotionBlur: everything UPSTREAM of the node, averaged across a shutter.
//!
//! Motion blur here is sub-frame SAMPLING, not a vector smear: the input's
//! whole subtree is evaluated at several moments while the shutter is open
//! and the results are averaged. Anything animated upstream — a Transform's
//! translate, a Grade's gain, a spline's points — smears; what is downstream
//! runs once, at the frame, on the averaged picture. That scoping is the
//! point of it being a node (it used to be a property of the whole comp):
//! blur the moving element, not the grade laid over the finished shot.
//!
//! As a stage it is the simplest there is — its input subtree renders into
//! `a`, no passes follow, `a` is the result. The sampling is done by the
//! app and the compositor (`Compositor::encode_motion_sample`): N submits,
//! each re-dispatching only this node's upstream stages under that moment's
//! uniforms and adding `a` into an accumulator, then a resolve back into `a`.
//! Params are uniforms, so a sample is a dispatch, never a recompile.
//!
//! A moving PLATE does not smear: which frame a reader decodes is a fact
//! about the whole tick, and there is no half frame on disk to land on.

use crate::codegen::Emit;
use crate::cost::Speed;
use crate::nodes::stage::{StageCx, StageDef, StageProgram, Wired};
use crate::nodes::NodeDef;

const TOPO_BASE: u64 = 0x3B10;

/// Ceiling on samples: past this the cost is real and the improvement is not.
pub const MAX_SAMPLES: u32 = 64;
/// From this many samples the node dominates the frame: its whole upstream
/// is rendered this many times over.
pub const HEAVY_SAMPLES: u32 = 8;

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct MotionBlurParams {
    /// Samples across the shutter. More is smoother and linearly dearer:
    /// each one is another evaluation of everything upstream.
    pub samples: u32,
    /// Open time as a fraction of a frame — 0.5 is a 180 degree shutter,
    /// the film default.
    pub angle: f32,
    /// Where the shutter sits about the frame: 0 centred, -0.5 closing on
    /// it, +0.5 opening on it.
    pub offset: f32,
    /// Which pipe it sits in: wired a MATTE it filters the coverage and
    /// gives a matte (`NodeKind::follows_pipe`). Kept true to its input by
    /// the app (`EditorGraph::connect_out`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub matte: bool,
}

impl Default for MotionBlurParams {
    fn default() -> Self {
        Self { samples: 8, angle: 0.5, offset: 0.0, matte: false }
    }
}

impl MotionBlurParams {
    /// The sub-frame times to evaluate the upstream at, about frame `t`, or
    /// None when the shutter cannot produce anything but the frame itself.
    pub fn times(&self, t: f32) -> Option<Vec<f32>> {
        let n = self.samples.min(MAX_SAMPLES);
        if n < 2 || self.angle.abs() < 1e-4 {
            return None;
        }
        Some(
            (0..n)
                .map(|i| {
                    // Sample at the CENTRE of each slice of the open shutter,
                    // not at its edges: the ends of the smear then get the
                    // same weight as the middle, and a 2-sample blur is not
                    // just the two extremes.
                    let f = (i as f32 + 0.5) / n as f32 - 0.5;
                    t + (f + self.offset) * self.angle
                })
                .collect(),
        )
    }
}

impl NodeDef for MotionBlurParams {
    const NAME: &'static str = "MotionBlur";
    const INPUTS: usize = 1;
    const MASK_INPUT: Option<usize> = None;
    /// The shutter never reaches a shader: it decides how often the
    /// upstream is dispatched, not what any pass computes.
    const VEC4_SLOTS: usize = 0;

    /// Constant: samples, angle and offset are all encode-time counts and
    /// times, so turning any of them never recompiles.
    fn topo_tag(&self) -> u64 {
        TOPO_BASE
    }

    fn pack(&self, _: &mut [f32]) {}

    fn emit(&self, e: &mut Emit) -> String {
        e.stage_result()
    }
}

impl StageDef for MotionBlurParams {
    /// No passes: the subtree lands in `a`, and `a` is either that one
    /// render (nothing to smear) or the resolved average of N of them.
    fn program(&self, _canvas: (u32, u32), _wired: Wired) -> StageProgram {
        StageProgram { textures: Vec::new(), passes: Vec::new(), result: "a".into() }
    }

    fn uniforms(&self, _pass: usize, _cx: &StageCx) -> Vec<f32> {
        Vec::new()
    }

    fn speed(&self) -> Speed {
        if self.samples.min(MAX_SAMPLES) >= HEAVY_SAMPLES {
            Speed::Heavy
        } else {
            Speed::Slow
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shutter decides WHEN the upstream is sampled, and the easy
    /// mistakes are all in the arithmetic: sampling the edges instead of
    /// the slices, or letting the offset stretch the span.
    #[test]
    fn the_shutter_samples_the_slices_it_is_open_for() {
        let sh = MotionBlurParams { samples: 4, angle: 1.0, offset: 0.0, matte: false };
        let t = sh.times(10.0).expect("open shutter samples");
        assert_eq!(t.len(), 4);
        assert!((t[0] - 9.625).abs() < 1e-5 && (t[3] - 10.375).abs() < 1e-5, "{t:?}");
        let mean = t.iter().sum::<f32>() / t.len() as f32;
        assert!((mean - 10.0).abs() < 1e-5, "the shutter is not centred: {mean}");

        // A half-open shutter spans half as much time.
        let half = MotionBlurParams { angle: 0.5, ..sh }.times(10.0).unwrap();
        assert!(((half[3] - half[0]) - 0.375).abs() < 1e-5, "{half:?}");

        // Offset slides the whole shutter without changing its length.
        let late = MotionBlurParams { offset: 0.5, ..sh }.times(10.0).unwrap();
        assert!(((late[3] - late[0]) - (t[3] - t[0])).abs() < 1e-5);
        assert!(late.iter().all(|x| *x > 10.0), "a +0.5 shutter opens on the frame");
    }

    /// One sample or a closed shutter mean "render it the ordinary way" —
    /// never N identical renders.
    #[test]
    fn a_shutter_that_cannot_blur_says_so() {
        let base = MotionBlurParams { samples: 8, angle: 0.5, offset: 0.0, matte: false };
        assert!(MotionBlurParams { samples: 1, ..base }.times(3.0).is_none(), "one sample");
        assert!(MotionBlurParams { angle: 0.0, ..base }.times(3.0).is_none(), "closed");
        assert_eq!(MotionBlurParams { samples: 999, ..base }.times(3.0).unwrap().len(), 64);
    }
}
