//! Rifetime: slow footage down (or speed it up) with REAL in-between frames,
//! made by RIFE.
//!
//! A Speed node at 0.5 can only repeat frames or cross-fade two of them,
//! which ghosts anything that moves. This node renders its input over its
//! range on a click, then for every output frame that falls between two
//! source frames runs RIFE at that fractional time and warps both
//! neighbours onto it along the motion — the in-between a camera would
//! have shot. The frames land on disk and play back through the node's
//! reader slot like a bake (`crate::retime` does the work, the app the
//! render and the adoption).
//!
//! Time: output frame T shows source time `a + (T - a) × speed`, `a` the
//! input's first frame — the Speed node's contract with the anchor on the
//! first frame, so the shot starts where it did and runs `1 / speed` as
//! long. The range resolver knows that (`ranges::time_shift`), so the
//! timeline, the viewer and a Write below all see the longer shot.
//!
//! The network sees the frames sRGB-encoded and clamped, as it was trained;
//! the PIXELS are warped from the linear, unclamped frames with the flow
//! and blend mask it returns, so highlights and HDR values survive the
//! retime. Frames on source frames are copied, never re-made.
//!
//! Two ways to say what time it is, `mode`:
//!
//! - **speed** — a constant rate, anchored on the input's first frame.
//! - **curve** — a keyable SOURCE FRAME, TimeRemap's contract: output frame
//!   T shows source frame `frame(T)`, any shape — ramps, freezes, reverses.
//!   The app samples the curve at every output frame when it renders
//!   (`plan_times`); the shot keeps its input's range.
//!
//! Not rendered yet, the input passes through untouched.

/// The weights it runs: RIFE is an interpolator, which is the job here.
pub const MODEL: &str = "rife-4.6";

/// The slowest and fastest rates offered.
pub const MIN_SPEED: f32 = 0.05;
pub const MAX_SPEED: f32 = 4.0;

/// How output time maps to source time.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum RifetimeMode {
    /// A constant rate.
    #[default]
    Speed,
    /// A keyable source frame per output frame.
    Curve,
}

impl RifetimeMode {
    pub const ALL: [RifetimeMode; 2] = [RifetimeMode::Speed, RifetimeMode::Curve];

    pub fn label(self) -> &'static str {
        match self {
            RifetimeMode::Speed => "speed",
            RifetimeMode::Curve => "curve",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct RifetimeParams {
    /// Reader slot the retimed frames play through.
    pub source: usize,
    pub mode: RifetimeMode,
    /// Source frames per output frame: 0.5 is half speed (twice as long).
    pub speed: f32,
    /// Curve mode: the source frame (a comp frame number) this output frame
    /// shows. Keyed, it IS the retime; unkeyed, a freeze on that frame.
    pub frame: f32,
    /// Analyse at this fraction of the size (the warp is always full size).
    pub detail: crate::nodes::flow::FlowDetail,
    /// TRANSIENT: frames are loaded in the reader slot.
    #[cfg_attr(feature = "serde", serde(skip_serializing))]
    pub loaded: bool,
    /// TRANSIENT: the comp frame the first output file holds.
    #[cfg_attr(feature = "serde", serde(skip_serializing))]
    pub start: Option<i64>,
}

impl Default for RifetimeParams {
    fn default() -> Self {
        Self {
            source: 0,
            mode: RifetimeMode::Speed,
            speed: 0.5,
            frame: 0.0,
            detail: Default::default(),
            loaded: false,
            start: None,
        }
    }
}

impl RifetimeParams {
    pub const NAME: &'static str = "Rifetime";
    pub const INPUTS: usize = 1;

    pub fn new(source: usize) -> Self {
        Self { source, ..Default::default() }
    }

    pub fn speed(&self) -> f32 {
        if self.speed.is_finite() {
            self.speed.clamp(MIN_SPEED, MAX_SPEED)
        } else {
            0.5
        }
    }

    pub fn topo_tag(&self) -> u64 {
        self.loaded as u64
    }

    /// What shapes the frames: change one and the render runs again into a
    /// new directory. In curve mode the curve's SAMPLES are the rest of the
    /// recipe, and only the app can take them (it owns the animation).
    pub fn recipe(&self) -> (&'static str, u8, u32, u8) {
        let speed = match self.mode {
            RifetimeMode::Speed => self.speed().to_bits(),
            RifetimeMode::Curve => 0,
        };
        (MODEL, self.mode as u8, speed, self.detail.divisor() as u8)
    }

    pub fn models(&self) -> &'static [&'static str] {
        &[MODEL]
    }

    /// Output frame `j` (counting from the first) as source frame `i` plus
    /// a fraction `t` towards `i + 1`, for an input of `inputs` frames. A
    /// fraction within a ten-thousandth of a frame snaps onto the frame:
    /// that one is copied, not made.
    pub fn plan(&self, inputs: usize) -> Vec<(usize, f32)> {
        if inputs == 0 {
            return Vec::new();
        }
        let speed = f64::from(self.speed());
        let last = (inputs - 1) as f64;
        let times: Vec<f64> = (0..).map(|j| j as f64 * speed).take_while(|tau| *tau <= last + 1e-6).collect();
        plan_times(&times, inputs)
    }
}

/// Source positions (in input frames from the first, fractional) as frame
/// `i` plus a fraction `t` towards `i + 1`. Positions outside the input
/// hold its first or last frame; a fraction within a ten-thousandth of a
/// frame snaps onto the frame, which is then copied, not made.
pub fn plan_times(times: &[f64], inputs: usize) -> Vec<(usize, f32)> {
    if inputs == 0 {
        return Vec::new();
    }
    let last = (inputs - 1) as f64;
    times
        .iter()
        .map(|&tau| {
            let tau = if tau.is_finite() { tau.clamp(0.0, last) } else { 0.0 };
            let (mut i, mut t) = (tau.floor() as usize, tau.fract());
            if t < 1e-4 {
                t = 0.0;
            } else if t > 1.0 - 1e-4 {
                i += 1;
                t = 0.0;
            }
            (i.min(inputs - 1), t as f32)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_speed_doubles_the_shot_and_lands_halfway() {
        let p = RifetimeParams::default();
        let plan = p.plan(3);
        assert_eq!(plan, vec![(0, 0.0), (0, 0.5), (1, 0.0), (1, 0.5), (2, 0.0)]);
        let q = RifetimeParams { speed: 0.25, ..p.clone() };
        assert_eq!(q.plan(2).len(), 5);
        let fast = RifetimeParams { speed: 2.0, ..p.clone() };
        assert_eq!(fast.plan(5), vec![(0, 0.0), (2, 0.0), (4, 0.0)]);
        assert!(p.plan(0).is_empty());
    }

    #[test]
    fn a_curve_plans_any_shape_and_holds_outside_the_shot() {
        // A ramp in, a freeze, a step back, then off the end.
        let times = [0.0, 0.25, 0.75, 1.5, 1.5, 1.0, 9.0, -3.0];
        assert_eq!(
            plan_times(&times, 3),
            vec![(0, 0.0), (0, 0.25), (0, 0.75), (1, 0.5), (1, 0.5), (1, 0.0), (2, 0.0), (0, 0.0)]
        );
    }

    #[test]
    fn the_mode_is_part_of_the_recipe_and_curve_mode_ignores_speed() {
        let a = RifetimeParams { mode: RifetimeMode::Curve, ..Default::default() };
        let b = RifetimeParams { speed: 0.1, ..a.clone() };
        assert_eq!(a.recipe(), b.recipe(), "speed does nothing in curve mode");
        assert_ne!(a.recipe(), RifetimeParams::default().recipe());
    }

    #[test]
    fn a_third_snaps_onto_frames_without_drifting() {
        let p = RifetimeParams { speed: 1.0 / 3.0, ..Default::default() };
        let plan = p.plan(3);
        assert_eq!(plan.len(), 7);
        assert_eq!(plan[3], (1, 0.0), "three thirds is a whole frame");
        assert_eq!(plan[6], (2, 0.0));
    }

    /// Setups saved before the rename stored the node as `Retime`.
    #[cfg(feature = "serde")]
    #[test]
    fn an_old_retime_loads_as_a_rifetime() {
        use crate::graph::NodeKind;
        let new = serde_json::to_string(&NodeKind::Rifetime(RifetimeParams::new(4))).unwrap();
        assert!(new.contains("Rifetime"), "{new}");
        let old: NodeKind = serde_json::from_str(&new.replace("Rifetime", "Retime")).unwrap();
        assert!(matches!(old, NodeKind::Rifetime(p) if p.source == 4));
    }

    #[test]
    fn only_loading_changes_the_topology_and_speed_is_the_recipe() {
        let a = RifetimeParams::default();
        let b = RifetimeParams { speed: 0.25, ..a.clone() };
        assert_eq!(a.topo_tag(), b.topo_tag());
        assert_ne!(a.recipe(), b.recipe());
        let c = RifetimeParams { loaded: true, start: Some(3), source: 9, ..a.clone() };
        assert_eq!(a.recipe(), c.recipe());
        assert_ne!(a.topo_tag(), c.topo_tag());
    }
}
