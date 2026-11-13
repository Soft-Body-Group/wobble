//! Depth: ByteDance's Video Depth Anything as a node whose result plays
//! back like a bake.
//!
//! A ViT depth network is not something the engine runs. The app renders
//! the node's input to disk, runs the tool as a process on an explicit
//! click (the CorridorKey / SeedVR2 precedent), and the node then samples
//! the result through its reader slot. A loaded result CLEARS the slow
//! taint above it like a GraphBake: the pixels come off disk.
//!
//! The network sees the whole clip in overlapping 32-frame windows and
//! aligns each window to the last (`infer_video_depth`), which is what
//! keeps the depth steady over time where a per-frame model flickers.
//!
//! What the result files hold: ONE channel, `Z` — the network's own output
//! (a relative model's disparity, a metric model's metres), half float —
//! plus the clip's `meta.json` (`top`, the disparity's robust clip maximum,
//! and `metric`). The three outputs are derived HERE from those two
//! uniforms, so the pick is a uniform and never a re-run:
//!
//! - **depth**: a relative model's disparity divided by `top` and inverted,
//!   so the nearest thing in the shot reads 1 and further things read more
//!   (capped at `FAR`). RELATIVE: right up to a scale and a shift the
//!   network cannot know. A metric model's output, metres, as it is.
//! - **disparity** (inverse depth): a relative model's output as it is, a
//!   metric model's 1/metres. For a Defocus with `inverse depth` on.
//! - **near**: 1 near, 0 far — disparity over `top`.
//!
//! (The first cut wrote all three, float32: 74 MB a 4K frame.)
//!
//! The output is the DATA pipe: one value, broadcast to rgb, alpha 1, so it
//! wires straight into a Defocus's Z port.
//!
//! Licence (manifesto rule 10): the code is Apache-2.0. Small and Metric
//! Small are Apache-2.0; Large and Metric Large are CC-BY-NC-4.0 and the
//! node is tagged while one is picked (the catalog row's `commercial`).

/// Which network. Each is one catalog row (`models()`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DepthModel {
    /// ViT-S, relative. Apache-2.0.
    #[default]
    Small,
    /// ViT-S, metric (metres). Apache-2.0.
    SmallMetric,
    /// ViT-L, relative. CC-BY-NC-4.0.
    Large,
    /// ViT-L, metric. CC-BY-NC-4.0.
    LargeMetric,
}

impl DepthModel {
    pub const ALL: [DepthModel; 4] =
        [DepthModel::Small, DepthModel::SmallMetric, DepthModel::Large, DepthModel::LargeMetric];

    pub fn label(self) -> &'static str {
        match self {
            DepthModel::Small => "small",
            DepthModel::SmallMetric => "small, metric",
            DepthModel::Large => "large (non-commercial)",
            DepthModel::LargeMetric => "large, metric (non-commercial)",
        }
    }

    /// The catalog id of its weights.
    pub fn model_id(self) -> &'static str {
        match self {
            DepthModel::Small => "vda-small",
            DepthModel::SmallMetric => "vda-small-metric",
            DepthModel::Large => "vda-large",
            DepthModel::LargeMetric => "vda-large-metric",
        }
    }

    /// The driver's `--encoder`.
    pub fn encoder(self) -> &'static str {
        match self {
            DepthModel::Small | DepthModel::SmallMetric => "vits",
            DepthModel::Large | DepthModel::LargeMetric => "vitl",
        }
    }

    /// Metres out, rather than relative disparity.
    pub fn metric(self) -> bool {
        matches!(self, DepthModel::SmallMetric | DepthModel::LargeMetric)
    }
}

/// Which of the result's three values the node outputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DepthOut {
    /// Depth-like: nearest = 1 (relative), or metres (metric).
    #[default]
    Depth,
    /// Inverse depth, as the model made it.
    Disparity,
    /// 1 near, 0 far.
    Near,
}

impl DepthOut {
    pub const ALL: [DepthOut; 3] = [DepthOut::Depth, DepthOut::Disparity, DepthOut::Near];

    pub fn label(self) -> &'static str {
        match self {
            DepthOut::Depth => "depth",
            DepthOut::Disparity => "disparity (1/depth)",
            DepthOut::Near => "near 1 .. far 0",
        }
    }

    /// The result file's channel it reads.
    fn lane(self) -> f32 {
        match self {
            DepthOut::Depth => 0.0,
            DepthOut::Disparity => 1.0,
            DepthOut::Near => 2.0,
        }
    }
}

/// The largest depth-like value a relative result is given: disparity 0
/// (the sky) would otherwise be infinite.
pub const FAR: f32 = 1000.0;

/// The node's authored state, plus two transient fields.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct DepthParams {
    /// Reader slot the result plays through.
    pub source: usize,
    pub model: DepthModel,
    /// The network's input size (short side, a multiple of 14). Larger is
    /// finer and slower; the result is scaled back to the plate's size.
    pub size: u32,
    pub output: DepthOut,
    /// TRANSIENT: a result is loaded in the reader slot.
    #[cfg_attr(feature = "serde", serde(skip_serializing))]
    pub loaded: bool,
    /// TRANSIENT: the comp frame the first result file holds.
    #[cfg_attr(feature = "serde", serde(skip_serializing))]
    pub start: Option<i64>,
    /// TRANSIENT: the loaded result's disparity clip maximum (`meta.json`).
    #[cfg_attr(feature = "serde", serde(skip_serializing))]
    pub top: f32,
    /// TRANSIENT: the loaded result holds metres, not disparity.
    #[cfg_attr(feature = "serde", serde(skip_serializing))]
    pub metric_take: bool,
}

impl Default for DepthParams {
    fn default() -> Self {
        Self {
            source: 0,
            model: DepthModel::Small,
            size: 518,
            output: DepthOut::Depth,
            loaded: false,
            start: None,
            top: 1.0,
            metric_take: false,
        }
    }
}

impl DepthParams {
    pub const NAME: &'static str = "Depth";
    pub const INPUTS: usize = 1;
    pub const VEC4_SLOTS: usize = 1;

    pub fn new(source: usize) -> Self {
        Self { source, ..Default::default() }
    }

    /// Only whether a result is loaded changes the shader: the output pick
    /// is a uniform.
    pub fn topo_tag(&self) -> u64 {
        self.loaded as u64
    }

    /// The weights the picked model needs (`NodeKind::models`).
    pub fn models(&self) -> &'static [&'static str] {
        match self.model {
            DepthModel::Small => &["vda-small"],
            DepthModel::SmallMetric => &["vda-small-metric"],
            DepthModel::Large => &["vda-large"],
            DepthModel::LargeMetric => &["vda-large-metric"],
        }
    }

    /// The input size the tool is given: a multiple of 14, 140..=1400.
    pub fn size_arg(&self) -> u32 {
        (self.size.clamp(140, 1400) + 7) / 14 * 14
    }

    /// Everything that shapes the pixels, for the app's result key. NOT
    /// the output pick, which is a uniform.
    pub fn recipe(&self) -> (u8, u32) {
        (self.model as u8, self.size_arg())
    }

    pub fn pack(&self, out: &mut [f32]) {
        out[0] = self.output.lane();
        out[1] = if self.top > 1e-8 { self.top } else { 1.0 };
        out[2] = if self.metric_take { 1.0 } else { 0.0 };
        out[3] = 0.0;
    }

    /// The WGSL for a loaded result: the file's one channel sampled at the
    /// output pixel, the picked output derived from it (`top`, `metric` in
    /// the slot), broadcast to rgb, alpha 1 (the Data pipe's shape,
    /// `nodes::todata`). No input transform: these are numbers.
    pub fn emit_loaded(src: &str, pos: &str, slot: usize, body: &mut String, fresh: &mut dyn FnMut() -> String) -> String {
        use std::fmt::Write;
        let (raw, disp, near, depth, o) = (fresh(), fresh(), fresh(), fresh(), fresh());
        let q = format!("P.v[{slot}]");
        writeln!(body, "    let {raw} = sample_px({src}, {pos}).r;").unwrap();
        writeln!(body, "    let {disp} = select({raw}, 1.0 / max({raw}, 1e-3), {q}.z > 0.5);").unwrap();
        writeln!(body, "    let {near} = clamp({disp} / {q}.y, 0.0, 1.0);").unwrap();
        writeln!(
            body,
            "    let {depth} = select(min(1.0 / max({disp} / {q}.y, {inv:?}), {far:?}), {raw}, {q}.z > 0.5);",
            inv = 1.0 / FAR,
            far = FAR
        )
        .unwrap();
        writeln!(
            body,
            "    let {o} = vec4f(vec3f(select(select({depth}, {disp}, {q}.x > 0.5), {near}, {q}.x > 1.5)), 1.0);"
        )
        .unwrap();
        o
    }

    /// The CPU mirror of `emit_loaded`: what the node outputs for a stored
    /// value. The suite and the tests judge the shader against it.
    pub fn derive(&self, raw: f32) -> f32 {
        let top = if self.top > 1e-8 { self.top } else { 1.0 };
        let disp = if self.metric_take { 1.0 / raw.max(1e-3) } else { raw };
        let near = (disp / top).clamp(0.0, 1.0);
        let depth = if self.metric_take { raw } else { (1.0 / (disp / top).max(1.0 / FAR)).min(FAR) };
        match self.output {
            DepthOut::Depth => depth,
            DepthOut::Disparity => disp,
            DepthOut::Near => near,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_loading_changes_the_topology() {
        let mut a = DepthParams::new(3);
        let b = a.clone();
        a.model = DepthModel::Large;
        a.output = DepthOut::Near;
        a.size = 700;
        assert_eq!(a.topo_tag(), b.topo_tag());
        a.loaded = true;
        assert_ne!(a.topo_tag(), b.topo_tag());
    }

    #[test]
    fn the_recipe_is_every_pixel_shaping_setting_and_not_the_pick() {
        let a = DepthParams::default();
        let mut b = a.clone();
        b.output = DepthOut::Disparity;
        b.source = 9;
        b.loaded = true;
        b.start = Some(1001);
        assert_eq!(a.recipe(), b.recipe(), "the output pick is a uniform, not a re-run");
        for m in DepthModel::ALL.into_iter().skip(1) {
            assert_ne!(a.recipe(), DepthParams { model: m, ..a.clone() }.recipe());
        }
        assert_ne!(a.recipe(), DepthParams { size: 700, ..a.clone() }.recipe());
        // 518 and 520 are the same input to the network.
        assert_eq!(a.recipe(), DepthParams { size: 520, ..a.clone() }.recipe());
    }

    #[test]
    fn the_size_is_a_multiple_of_fourteen() {
        for s in [0, 100, 518, 519, 700, 5000] {
            let a = DepthParams { size: s, ..Default::default() }.size_arg();
            assert_eq!(a % 14, 0, "{s} -> {a}");
            assert!((140..=1400).contains(&a));
        }
        assert_eq!(DepthParams::default().size_arg(), 518);
    }

    #[test]
    fn each_model_names_its_own_weights() {
        for m in DepthModel::ALL {
            assert_eq!(DepthParams { model: m, ..Default::default() }.models(), &[m.model_id()]);
        }
    }

    #[test]
    fn the_loaded_shader_picks_a_lane_and_broadcasts_it() {
        let mut body = String::new();
        let mut n = 0;
        let out = DepthParams::emit_loaded("src0", "pos", 4, &mut body, &mut || {
            n += 1;
            format!("t{n}")
        });
        assert!(body.contains("sample_px(src0, pos).r"));
        assert!(body.contains("P.v[4].x > 0.5") && body.contains("P.v[4].x > 1.5"), "{body}");
        assert!(body.contains("P.v[4].y") && body.contains("P.v[4].z > 0.5"), "{body}");
        assert!(body.contains("vec4f(vec3f("), "broadcast to rgb");
        assert_eq!(out, "t5");
        let mut v = [9.0; 4];
        for (o, lane) in [(DepthOut::Depth, 0.0), (DepthOut::Disparity, 1.0), (DepthOut::Near, 2.0)] {
            DepthParams { output: o, top: 2.0, metric_take: true, ..Default::default() }.pack(&mut v);
            assert_eq!(v, [lane, 2.0, 1.0, 0.0]);
        }
        // A missing statistic never divides by zero.
        DepthParams { top: 0.0, ..Default::default() }.pack(&mut v);
        assert_eq!(v[1], 1.0);
    }

    #[test]
    fn the_outputs_derive_from_one_stored_value() {
        let rel = |output| DepthParams { output, top: 2.0, ..Default::default() };
        assert_eq!(rel(DepthOut::Disparity).derive(0.5), 0.5);
        assert_eq!(rel(DepthOut::Near).derive(0.5), 0.25);
        assert_eq!(rel(DepthOut::Depth).derive(0.5), 4.0);
        assert_eq!(rel(DepthOut::Depth).derive(2.0), 1.0, "the clip's nearest reads 1");
        assert!((rel(DepthOut::Depth).derive(0.0) - FAR).abs() < 0.01, "the sky is capped");
        let met = |output| DepthParams { output, top: 0.5, metric_take: true, ..Default::default() };
        assert_eq!(met(DepthOut::Depth).derive(4.0), 4.0, "metres as they are");
        assert_eq!(met(DepthOut::Disparity).derive(4.0), 0.25);
        assert_eq!(met(DepthOut::Near).derive(4.0), 0.5);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn transients_are_never_saved() {
        let p = DepthParams { loaded: true, start: Some(7), top: 3.0, metric_take: true, ..DepthParams::new(2) };
        let j = serde_json::to_string(&p).unwrap();
        assert!(!j.contains("loaded") && !j.contains("start") && !j.contains("top") && !j.contains("metric_take"), "{j}");
        let old: DepthParams = serde_json::from_str(r#"{"source":1}"#).unwrap();
        assert_eq!(old.size, 518);
    }
}
