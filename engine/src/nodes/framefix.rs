//! Frame Fix: rebuild bad frames (a dropped frame, a flash, a render glitch)
//! from their good neighbours; every other frame passes through untouched.
//!
//! `frames` lists the bad COMP frames. A run of bad frames is rebuilt from
//! the good frame before it (`g0`) and the good frame after it (`g1`), each
//! bad frame `f` at the fraction `(f - g0) / (g1 - g0)` between them — so a
//! single bad frame is the halfway picture of its two neighbours.
//!
//! Two modes:
//!
//! - **blend** — live, a cross-fade of `g0` and `g1`. Built on the time
//!   machinery, not on a pass: the node puts its input on two time branches
//!   (`branch_maps`: one reads `g0` on a bad tick, the other `g1`, both the
//!   tick itself otherwise) and dissolves them by `blend_w`, which the app
//!   fills per tick (0 on a good tick, so the picture is the input exactly).
//!   It costs a second decode of the input a tick, no stage.
//! - **flow** — RIFE in-betweens, on a click (the Rifetime runner in
//!   `flowui`, `crate::rifetime` doing the work): the input is rendered over
//!   its range, every bad frame is MADE from `g0` and `g1` along the motion,
//!   every good frame is linked across, and the result plays back through
//!   the node's reader slot like a bake (`inputs_unsampled`, clears slow).
//!   Until it is rendered the node shows the blend.
//!
//! A bad frame with no good neighbour inside the input's range (the first or
//! last frame) holds the neighbour it has.

use crate::graph::{TimeMap, TimeOp, TimePhase};

/// The interpolator flow mode runs: Rifetime's.
pub const MODEL: &str = crate::nodes::rifetime::MODEL;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FrameFixMode {
    /// A live cross-fade of the good neighbours.
    #[default]
    Blend,
    /// RIFE in-betweens, rendered on a click.
    Flow,
}

impl FrameFixMode {
    pub const ALL: [FrameFixMode; 2] = [FrameFixMode::Blend, FrameFixMode::Flow];

    pub fn label(self) -> &'static str {
        match self {
            FrameFixMode::Blend => "blend",
            FrameFixMode::Flow => "optical flow",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Default)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct FrameFixParams {
    /// Reader slot the flow-mode frames play through.
    pub source: usize,
    /// The bad comp frames, any order (`bad()` sorts them).
    pub frames: Vec<i64>,
    pub mode: FrameFixMode,
    /// Flow mode: judge the motion at this fraction of the size.
    pub detail: crate::nodes::flow::FlowDetail,
    /// TRANSIENT: flow-mode frames are loaded in the reader slot (the app
    /// sets it only in flow mode).
    #[cfg_attr(feature = "serde", serde(skip_serializing))]
    pub loaded: bool,
    /// TRANSIENT: the comp frame the first loaded file holds.
    #[cfg_attr(feature = "serde", serde(skip_serializing))]
    pub start: Option<i64>,
    /// TRANSIENT: this tick's dissolve from the `g0` branch to the `g1` one.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub blend_w: f32,
}

impl FrameFixParams {
    pub const NAME: &'static str = "Frame Fix";
    pub const INPUTS: usize = 1;

    pub fn new(source: usize, frame: i64) -> Self {
        Self { source, frames: vec![frame], ..Default::default() }
    }

    /// The bad frames, sorted, each once.
    pub fn bad(&self) -> Vec<i64> {
        let mut b = self.frames.clone();
        b.sort_unstable();
        b.dedup();
        b
    }

    /// For a bad tick, the good frames either side of its run.
    pub fn neighbours(&self, t: i64) -> Option<(i64, i64)> {
        let bad = self.bad();
        bad.binary_search(&t).ok()?;
        Some((good_from(&bad, t, -1), good_from(&bad, t, 1)))
    }

    /// The dissolve at tick `t`: 0 on a good frame.
    pub fn weight_at(&self, t: i64) -> f32 {
        match self.neighbours(t) {
            Some((a, b)) => (t - a) as f32 / (b - a) as f32,
            None => 0.0,
        }
    }

    /// The two time branches the blend reads: the good frame before a bad
    /// run, and the one after it. Both are the tick itself on a good tick.
    pub fn branch_maps(&self) -> [TimeMap; 2] {
        let bad = self.bad();
        let side = |side: i8| TimeMap { ops: vec![TimeOp::Fix { bad: bad.clone(), side }], phase: TimePhase::Nearest };
        [side(-1), side(1)]
    }

    /// Flow mode's work for an input of `n` frames starting at comp frame
    /// `first`: per output frame, the two input frames (indices from the
    /// first) and how far from the one towards the other. A good frame is
    /// `(j, j, 0)`, a copy; a bad one at an edge holds its one neighbour.
    pub fn plan(&self, first: u64, n: usize) -> Vec<(usize, usize, f32)> {
        let first = first as i64;
        let inside = |f: i64| f >= first && f < first + n as i64;
        (0..n)
            .map(|j| {
                let t = first + j as i64;
                match self.neighbours(t) {
                    None => (j, j, 0.0),
                    Some((a, b)) => match (inside(a), inside(b)) {
                        (true, true) => ((a - first) as usize, (b - first) as usize, self.weight_at(t)),
                        (true, false) => ((a - first) as usize, (a - first) as usize, 0.0),
                        (false, true) => ((b - first) as usize, (b - first) as usize, 0.0),
                        (false, false) => (j, j, 0.0),
                    },
                }
            })
            .collect()
    }

    /// What shapes flow mode's frames: change one and the render runs again
    /// into a new directory.
    pub fn recipe(&self) -> (&'static str, Vec<i64>, u8) {
        (MODEL, self.bad(), self.detail.divisor() as u8)
    }

    /// Weights only in flow mode: blend needs none.
    pub fn models(&self) -> &'static [&'static str] {
        match self.mode {
            FrameFixMode::Flow => &[MODEL],
            FrameFixMode::Blend => &[],
        }
    }

    /// Loading flow frames swaps the time branches for a reader: topology.
    /// The list and the mode are not.
    pub fn topo_tag(&self) -> u64 {
        self.loaded as u64
    }
}

/// The nearest frame from `t` in direction `side` that is not in `bad`
/// (sorted).
pub fn good_from(bad: &[i64], t: i64, side: i8) -> i64 {
    let step = if side < 0 { -1 } else { 1 };
    let mut g = t;
    while bad.binary_search(&g).is_ok() {
        g += step;
    }
    g
}

/// "1012, 1040-1042" -> [1012, 1040, 1041, 1042]. Spaces, commas and
/// semicolons separate; a range is `a-b` (either order). A range over 10 000
/// frames is refused rather than allocated.
pub fn parse_frames(text: &str) -> Result<Vec<i64>, String> {
    let mut out = Vec::new();
    for tok in text.split([',', ';', ' ']).map(str::trim).filter(|s| !s.is_empty()) {
        // A leading '-' is a sign, not a range.
        let split = tok.char_indices().skip(1).find(|(_, c)| *c == '-').map(|(i, _)| i);
        let num = |s: &str| s.trim().parse::<i64>().map_err(|_| format!("'{s}' is not a frame"));
        match split {
            Some(i) => {
                let (a, b) = (num(&tok[..i])?, num(&tok[i + 1..])?);
                let (a, b) = (a.min(b), a.max(b));
                if b - a > 10_000 {
                    return Err(format!("{tok} is more than 10000 frames"));
                }
                out.extend(a..=b);
            }
            None => out.push(num(tok)?),
        }
    }
    out.sort_unstable();
    out.dedup();
    Ok(out)
}

/// The list as the panel shows it, runs folded: "1012, 1040-1042".
pub fn frames_text(frames: &[i64]) -> String {
    let mut b = frames.to_vec();
    b.sort_unstable();
    b.dedup();
    let mut parts = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let mut j = i;
        while j + 1 < b.len() && b[j + 1] == b[j] + 1 {
            j += 1;
        }
        parts.push(if j > i { format!("{}-{}", b[i], b[j]) } else { b[i].to_string() });
        i = j + 1;
    }
    parts.join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(frames: &[i64]) -> FrameFixParams {
        FrameFixParams { frames: frames.to_vec(), ..Default::default() }
    }

    #[test]
    fn one_bad_frame_is_the_halfway_of_its_neighbours() {
        let f = p(&[12]);
        assert_eq!(f.neighbours(12), Some((11, 13)));
        assert_eq!(f.weight_at(12), 0.5);
        assert_eq!(f.neighbours(11), None);
        assert_eq!(f.weight_at(13), 0.0);
        let [a, b] = f.branch_maps();
        assert_eq!((a.apply(12), b.apply(12)), (11, 13));
        for t in [0u64, 11, 13, 500] {
            assert_eq!((a.apply(t), b.apply(t)), (t, t), "a good tick reads itself on both branches");
        }
    }

    #[test]
    fn a_run_spreads_across_its_good_ends() {
        let f = p(&[22, 20, 21]);
        assert_eq!(f.bad(), vec![20, 21, 22]);
        assert_eq!(f.neighbours(21), Some((19, 23)));
        assert_eq!(f.weight_at(20), 0.25);
        assert_eq!(f.weight_at(22), 0.75);
        let [a, b] = f.branch_maps();
        assert_eq!((a.apply(22), b.apply(22)), (19, 23));
    }

    #[test]
    fn the_flow_plan_copies_good_frames_and_holds_at_an_edge() {
        let f = p(&[10, 12, 13, 14]);
        let plan = f.plan(10, 6); // frames 10..=15
        assert_eq!(plan[0], (1, 1, 0.0), "the first frame is bad: hold the next");
        assert_eq!(plan[1], (1, 1, 0.0));
        assert_eq!(plan[2], (1, 5, 0.25));
        assert_eq!(plan[4], (1, 5, 0.75));
        assert_eq!(plan[5], (5, 5, 0.0));
    }

    #[test]
    fn only_loading_is_topology_and_the_list_is_the_recipe() {
        let a = p(&[5]);
        let b = FrameFixParams { mode: FrameFixMode::Flow, frames: vec![5, 6], ..a.clone() };
        assert_eq!(a.topo_tag(), b.topo_tag());
        assert_ne!(a.recipe(), b.recipe());
        assert_eq!(a.recipe(), FrameFixParams { frames: vec![5, 5], ..a.clone() }.recipe());
        assert_ne!(a.topo_tag(), FrameFixParams { loaded: true, ..a.clone() }.topo_tag());
        assert!(a.models().is_empty());
        assert_eq!(b.models(), &[MODEL]);
    }

    #[test]
    fn frame_lists_parse_and_fold() {
        assert_eq!(parse_frames("1012, 1040-1042").unwrap(), vec![1012, 1040, 1041, 1042]);
        assert_eq!(parse_frames(" 7;3 5-4 ").unwrap(), vec![3, 4, 5, 7]);
        assert_eq!(parse_frames("-3").unwrap(), vec![-3]);
        assert!(parse_frames("x").is_err());
        assert!(parse_frames("0-20000").is_err());
        assert_eq!(frames_text(&[1042, 1012, 1040, 1041]), "1012, 1040-1042");
        assert_eq!(parse_frames(&frames_text(&[1, 2, 3, 9])).unwrap(), vec![1, 2, 3, 9]);
    }
}
