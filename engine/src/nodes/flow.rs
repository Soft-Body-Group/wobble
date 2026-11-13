//! Flow: optical flow as a node — per-pixel MOTION VECTORS, for anything
//! downstream to use.
//!
//! The vectors come from one of two networks, both ported to WGSL and run
//! on the GPU with no Python: SEA-RAFT (`crate::searaft`, BSD-3), real
//! two-frame optical flow and the default, or RIFE v4.6 (`crate::rife`,
//! MIT), a frame interpolator whose flow is lighter and rougher. The app
//! renders the node's input over its frame range, runs the network over
//! each pair of neighbouring frames on a click (`crate::flowsolve`), and
//! writes the vectors to disk; the node then plays them back through its
//! reader slot like a bake. So the analysis is the slow path, it is
//! explicit, and once it has run the node costs what a Read costs (a loaded
//! Flow clears the slow taint above it, like a GraphBake).
//!
//! What it hands downstream is DATA on the image pipe, never colour:
//!
//! - `rg` — FORWARD motion: where this pixel is on the NEXT frame, as an
//!   offset in canvas pixels (x right, y DOWN — the viewer's own pixel
//!   space). A pixel at `p` is at `p + rg` one frame later.
//! - `ba` — BACKWARD motion: where it was on the PREVIOUS frame, same
//!   units. With `backward` off it is `-rg`, the constant-velocity guess,
//!   so a consumer can always read both halves of a shutter.
//!
//! Nothing loaded, the node is zero motion everywhere — a VectorBlur below
//! it does nothing — and it samples no input at all: its input is only
//! ever read by the analysis render. So wiring a Flow costs nothing until
//! the button is pressed.
//!
//! Vectors are stored in FULL-resolution pixels; under proxy the node
//! scales them on the way out, so consumers always see canvas pixels and
//! never have to know.
//!
//! Consumers: VectorBlur (motion blur along the vectors) and VectorWarp
//! (sample a picture where the vectors point), and anything else that
//! reads a picture — an Expression or a Shader node can read the vectors
//! like any other input.

use crate::cost::Speed;

/// Which network solves the motion.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FlowEngine {
    /// SEA-RAFT (M, trained on TartanAir + Chairs + Things): flow is what
    /// it is trained and judged on. Sharper at motion edges.
    #[default]
    SeaRaft,
    /// RIFE 4.6: an interpolator's flow — lighter weights, rougher at
    /// edges and occlusions.
    Rife,
}

impl FlowEngine {
    pub const ALL: [FlowEngine; 2] = [FlowEngine::SeaRaft, FlowEngine::Rife];

    pub fn label(self) -> &'static str {
        match self {
            FlowEngine::SeaRaft => "SEA-RAFT",
            FlowEngine::Rife => "RIFE",
        }
    }

    /// The catalog id of its weights (`crate::models`).
    pub fn model(self) -> &'static str {
        match self {
            FlowEngine::SeaRaft => "searaft-m",
            FlowEngine::Rife => "rife-4.6",
        }
    }
}

/// How finely the network looks: the frames are analysed at this fraction
/// of their size and the vectors scaled back up. Half is the usual pick
/// at 4K — RIFE was trained on smaller pictures, and large motions are
/// easier for it at half size.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FlowDetail {
    #[default]
    Full,
    Half,
    Quarter,
}

impl FlowDetail {
    pub const ALL: [FlowDetail; 3] = [FlowDetail::Full, FlowDetail::Half, FlowDetail::Quarter];

    pub fn label(self) -> &'static str {
        match self {
            FlowDetail::Full => "full",
            FlowDetail::Half => "half",
            FlowDetail::Quarter => "quarter",
        }
    }

    /// The picture is shrunk by this before the network sees it.
    pub fn divisor(self) -> usize {
        match self {
            FlowDetail::Full => 1,
            FlowDetail::Half => 2,
            FlowDetail::Quarter => 4,
        }
    }
}

/// The node's authored state, plus two transient fields.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct FlowParams {
    /// Reader slot the vectors play through.
    pub source: usize,
    pub engine: FlowEngine,
    pub detail: FlowDetail,
    /// Also solve BACKWARD vectors (a second network run per frame). Off,
    /// `ba` is `-rg`.
    pub backward: bool,
    /// TRANSIENT: vectors are loaded in the reader slot. Set by the app on
    /// the engine-graph copy only, like a GraphBake's `baked`.
    #[cfg_attr(feature = "serde", serde(skip_serializing))]
    pub loaded: bool,
    /// TRANSIENT: the comp frame the first vector file holds.
    #[cfg_attr(feature = "serde", serde(skip_serializing))]
    pub start: Option<i64>,
}

impl Default for FlowParams {
    fn default() -> Self {
        Self { source: 0, engine: FlowEngine::SeaRaft, detail: FlowDetail::Full, backward: true, loaded: false, start: None }
    }
}

impl FlowParams {
    pub const NAME: &'static str = "Flow";
    /// The picture to analyse.
    pub const INPUTS: usize = 1;
    /// (proxy scale, -, -, -).
    pub const VEC4_SLOTS: usize = 1;

    pub fn new(source: usize) -> Self {
        Self { source, ..Default::default() }
    }

    /// Only whether vectors are loaded changes the shader; the settings
    /// shape what the ANALYSIS writes.
    pub fn topo_tag(&self) -> u64 {
        self.loaded as u64
    }

    /// The settings that shape the vectors, for the app's result key:
    /// change one and the analysis runs again, into a new directory.
    pub fn recipe(&self) -> (&'static str, u8, bool) {
        (self.engine.model(), self.detail.divisor() as u8, self.backward)
    }

    /// The weights the PICKED engine needs: only those are gated.
    pub fn models(&self) -> &'static [&'static str] {
        match self.engine {
            FlowEngine::SeaRaft => &["searaft-m"],
            FlowEngine::Rife => &["rife-4.6"],
        }
    }

    /// What it costs once loaded: frames off disk.
    pub fn speed(&self) -> Speed {
        Speed::Fast
    }

    pub fn pack(&self, proxy_scale: f32, out: &mut [f32]) {
        out[0] = proxy_scale;
        out[1] = 0.0;
        out[2] = 0.0;
        out[3] = 0.0;
    }

    /// The WGSL for loaded vectors: the file sampled at the output pixel,
    /// NO input transform (these are offsets, not colours), scaled into
    /// canvas pixels.
    pub fn emit_loaded(src: &str, pos: &str, slot: usize, body: &mut String, fresh: &mut dyn FnMut() -> String) -> String {
        use std::fmt::Write;
        let v = fresh();
        writeln!(body, "    let {v} = sample_px({src}, {pos}) * P.v[{slot}].x;").unwrap();
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Vectors ride their own pipe: a Flow puts them on it, the motion
    /// consumers take them off it, a picture reader may read them, and a
    /// picture may NOT go into a vectors port (it would be read as motion).
    #[test]
    fn vectors_ride_the_vector_pipe() {
        use crate::graph::{NodeKind, PortKind};
        let flow = NodeKind::Flow(FlowParams::default());
        assert_eq!(flow.output_kind(), PortKind::Vector);
        for k in [NodeKind::VectorBlur(Default::default()), NodeKind::VectorWarp(Default::default())] {
            assert_eq!(k.input_kind(0), PortKind::Image, "{}", k.name());
            assert_eq!(k.input_kind(1), PortKind::Vector, "{}", k.name());
        }
        assert!(PortKind::Vector.can_feed(PortKind::Vector));
        assert!(PortKind::Vector.can_feed(PortKind::Image), "an Expression reads the numbers");
        assert!(!PortKind::Image.can_feed(PortKind::Vector));
        assert!(!PortKind::Vector.can_feed(PortKind::Mask));
        assert!(PortKind::Vector.is_picture(), "vectors carry a resolution and a range");
        assert_eq!(NodeKind::dot(PortKind::Vector).output_kind(), PortKind::Vector);
        assert_eq!(NodeKind::dot(PortKind::Vector).input_kind(0), PortKind::Vector);
    }

    #[test]
    fn only_loading_changes_the_topology() {
        let mut a = FlowParams::new(3);
        let b = a.clone();
        a.detail = FlowDetail::Half;
        a.backward = false;
        assert_eq!(a.topo_tag(), b.topo_tag(), "settings are the analysis', not code");
        a.loaded = true;
        assert_ne!(a.topo_tag(), b.topo_tag());
    }

    #[test]
    fn the_recipe_is_what_shapes_the_vectors() {
        let a = FlowParams::default();
        let mut b = a.clone();
        b.source = 4;
        b.loaded = true;
        b.start = Some(1001);
        assert_eq!(a.recipe(), b.recipe());
        let c = FlowParams { detail: FlowDetail::Quarter, ..a.clone() };
        assert_ne!(a.recipe(), c.recipe());
        let d = FlowParams { backward: false, ..a.clone() };
        assert_ne!(a.recipe(), d.recipe());
        let e = FlowParams { engine: FlowEngine::Rife, ..a.clone() };
        assert_ne!(a.recipe(), e.recipe());
        assert_eq!(e.models(), &["rife-4.6"]);
        assert_eq!(a.models(), &[a.engine.model()]);
    }

    #[test]
    fn the_loaded_shader_reads_raw_and_scales_by_the_proxy() {
        let mut body = String::new();
        let out = FlowParams::emit_loaded("src2", "pos", 5, &mut body, &mut || "t1".into());
        assert_eq!(out, "t1");
        assert!(body.contains("sample_px(src2, pos) * P.v[5].x"), "{body}");
        assert!(!body.contains("cs_idt"), "vectors are not colour");
    }

    #[cfg(feature = "serde")]
    #[test]
    fn transients_are_never_saved() {
        let p = FlowParams { loaded: true, start: Some(7), ..FlowParams::new(2) };
        let j = serde_json::to_string(&p).unwrap();
        assert!(!j.contains("loaded") && !j.contains("start"), "{j}");
        let old: FlowParams = serde_json::from_str(r#"{"source":1}"#).unwrap();
        assert!(old.backward && old.detail == FlowDetail::Full);
    }
}
