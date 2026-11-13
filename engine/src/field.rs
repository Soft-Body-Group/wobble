//! Procedural noise FIELDS — a second kind of graph, and a genuinely
//! different data type from the images the rest of this engine moves.
//!
//! An image port carries pixels that have already been decided. A **field**
//! port carries a *function*: `f(p, seed) -> f32`, callable at any coordinate
//! the consumer likes. That distinction is the whole paradigm, and it is
//! forced by what the user asked for — a node's parameters driven by other
//! noise. If an fBm's scale is itself a field, and a Warp above it rewrites
//! the coordinates its input is sampled at, then a node cannot be evaluated
//! once and cached: the same node may be called twice in one pixel at two
//! different coordinates. So every field node compiles to its own WGSL
//! function and consumers CALL it. Data flows down the picture but the
//! coordinate and the seed flow UP the graph.
//!
//! Consequences worth knowing before editing this file:
//!
//!  - No recursion. WGSL has none, so the graph must be a DAG; [`compile`]
//!    refuses a cycle rather than emitting a shader that will not build.
//!  - Params stay uniforms, like everywhere else in this engine: a whole
//!    sub-graph's numbers live in one contiguous block reserved by the
//!    owning `Noise` node, so dragging a value never recompiles. Only
//!    rewiring does.
//!  - Nothing is rasterised on the way. A field is pointwise, so the entire
//!    sub-graph collapses into the fused pass with no intermediate texture
//!    and no bandwidth — which is what makes a deep noise graph affordable.

use crate::graph::NodeId;
use std::collections::BTreeMap;
use std::fmt::Write as _;

/// Stock graphs the Noise panel loads.
pub mod presets;

/// Octave cap for fBm, shared with the shader text (`__FBM_MAX__`).
pub const FBM_MAX_OCTAVES: u32 = 8;

/// A noise graph: field nodes, how they are wired, and which one is the
/// result.
///
/// Deliberately NOT the comp `Graph`. It carries none of that type's
/// baggage (readers, time bases, colour spaces, resolutions) because a field
/// has none of those things, and unlike the comp graph it is DATA — it
/// serializes into the setup as part of the node that owns it. The editor
/// still edits it through the ordinary node editor by holding the same nodes
/// as `NodeKind::Field`; this is what that editing session is saved as.
#[derive(Clone, Debug, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct FieldGraph {
    pub nodes: BTreeMap<NodeId, FieldNode>,
    /// (consumer, socket) -> producer.
    #[cfg_attr(feature = "serde", serde(with = "edge_map"))]
    pub edges: BTreeMap<(NodeId, usize), NodeId>,
    /// The node whose value the graph is. `None` falls back to whatever
    /// nothing else consumes.
    pub out: Option<NodeId>,
    /// Where each node sits on the canvas. The compiler ignores this
    /// entirely — it is the editor's business — but it is authored state,
    /// so it belongs with the graph rather than in a side table that a
    /// saved setup would lose.
    pub layout: BTreeMap<NodeId, [f32; 2]>,
    /// Highest id ever handed out, so a re-opened graph does not reuse one.
    pub next_id: NodeId,
}

impl FieldGraph {
    /// What the graph EXPOSES to the node that owns it: every Slider and
    /// Input, by name, in id order. Sliders become widgets on the owner's
    /// panel and channels; Inputs become channels only. Two nodes with one
    /// name share a channel — set writes both — which is the predictable
    /// reading of a duplicate rather than an error nobody sees.
    pub fn exposed(&self) -> Vec<(String, bool)> {
        let mut out: Vec<(String, bool)> = Vec::new();
        for n in self.nodes.values() {
            let (name, is_input) = match n {
                FieldNode::Const { name, .. } => (name, false),
                FieldNode::Input { name, .. } => (name, true),
                _ => continue,
            };
            let name = if name.is_empty() { "slider".to_string() } else { name.clone() };
            if !out.iter().any(|(n, _)| *n == name) {
                out.push((name, is_input));
            }
        }
        out
    }
    /// The value behind an exposed name (the first node carrying it).
    pub fn exposed_value(&self, name: &str) -> Option<f32> {
        self.nodes.values().find_map(|n| match n {
            FieldNode::Const { value, name: nm } | FieldNode::Input { value, name: nm }
                if exposed_name(nm) == name =>
            {
                Some(*value)
            }
            _ => None,
        })
    }
    /// Write an exposed value into EVERY node carrying the name.
    pub fn set_exposed(&mut self, name: &str, v: f32) -> bool {
        let mut hit = false;
        for n in self.nodes.values_mut() {
            if let FieldNode::Const { value, name: nm } | FieldNode::Input { value, name: nm } = n {
                if exposed_name(nm) == name {
                    *value = v;
                    hit = true;
                }
            }
        }
        hit
    }
    pub fn input_of(&self, node: NodeId, port: usize) -> Option<NodeId> {
        self.edges.get(&(node, port)).copied().filter(|s| self.nodes.contains_key(s))
    }
    /// The node the graph evaluates to.
    pub fn output(&self) -> Option<NodeId> {
        // An explicit Output node is the answer whenever there is one: it is
        // what the author SAID the graph evaluates to, where the viewer
        // marker only says what they are looking at right now.
        if let Some(id) =
            self.nodes.iter().find(|(_, n)| matches!(n, FieldNode::Output)).map(|(id, _)| *id)
        {
            return Some(id);
        }
        self.out.filter(|id| self.nodes.contains_key(id)).or_else(|| {
            let consumed: std::collections::HashSet<NodeId> =
                self.edges.values().copied().collect();
            self.nodes.keys().copied().filter(|id| !consumed.contains(id)).last()
        })
    }
}

/// A `(consumer, socket) -> producer` map on the wire. JSON keys are
/// strings, so a tuple-keyed map cannot be written as an object: it goes
/// as a list of pairs. A setup written before this (whose noise graphs had
/// no edges yet) holds an empty object, which reads back as no edges.
#[cfg(feature = "serde")]
pub mod edge_map {
    use super::NodeId;
    use std::collections::BTreeMap;

    pub fn serialize<S: serde::Serializer>(
        m: &BTreeMap<(NodeId, usize), NodeId>,
        s: S,
    ) -> Result<S::Ok, S::Error> {
        let v: Vec<((NodeId, usize), NodeId)> = m.iter().map(|(k, v)| (*k, *v)).collect();
        serde::Serialize::serialize(&v, s)
    }

    pub fn deserialize<'de, D: serde::Deserializer<'de>>(
        d: D,
    ) -> Result<BTreeMap<(NodeId, usize), NodeId>, D::Error> {
        #[derive(serde::Deserialize)]
        #[serde(untagged)]
        enum Wire {
            List(Vec<((NodeId, usize), NodeId)>),
            Legacy(BTreeMap<String, serde::de::IgnoredAny>),
            Other(serde::de::IgnoredAny),
        }
        // The legacy form is the EMPTY object a tuple-keyed map wrote (a
        // non-empty one could never be written). Anything else is a mistake
        // to report, not to read as "no wires": a script that guessed the
        // shape got silence and a graph with nothing connected.
        match serde::Deserialize::deserialize(d)? {
            Wire::List(v) => Ok(v.into_iter().collect()),
            Wire::Legacy(m) if m.is_empty() => Ok(BTreeMap::new()),
            _ => Err(serde::de::Error::custom(
                "edges must be a list of [[consumer, input], producer], \
                 e.g. [[[2, 0], 1]] wires node 1 into input 0 of node 2",
            )),
        }
    }
}

/// An unnamed slider still needs a handle to be reached by.
fn exposed_name(n: &str) -> &str {
    if n.is_empty() { "slider" } else { n }
}

/// Which coordinate a `Coord` node hands out.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CoordKind {
    #[default]
    X,
    Y,
    /// Distance from the centre of the format, in the same units as x/y.
    Radius,
    /// Angle about the centre, -1..1.
    Angle,
}

impl CoordKind {
    pub const ALL: [CoordKind; 4] =
        [CoordKind::X, CoordKind::Y, CoordKind::Radius, CoordKind::Angle];
    pub fn name(self) -> &'static str {
        match self {
            CoordKind::X => "x",
            CoordKind::Y => "y",
            CoordKind::Radius => "radius",
            CoordKind::Angle => "angle",
        }
    }
}

/// How a fractal sum folds its octaves. The same three the image-side fBm
/// has, and the same code path in the shader.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FieldFbmMode {
    #[default]
    Fbm,
    Turbulence,
    Ridged,
}

impl FieldFbmMode {
    pub const ALL: [FieldFbmMode; 3] =
        [FieldFbmMode::Fbm, FieldFbmMode::Turbulence, FieldFbmMode::Ridged];
    pub fn name(self) -> &'static str {
        match self {
            FieldFbmMode::Fbm => "fbm",
            FieldFbmMode::Turbulence => "turbulence",
            FieldFbmMode::Ridged => "ridged",
        }
    }
    fn code(self) -> f32 {
        match self {
            FieldFbmMode::Fbm => 0.0,
            FieldFbmMode::Turbulence => 1.0,
            FieldFbmMode::Ridged => 2.0,
        }
    }
}

/// What a Voronoi node reports about the cell a point lands in.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum VoronoiMode {
    /// Distance to the nearest feature point.
    #[default]
    F1,
    /// Distance to the second nearest.
    F2,
    /// F2 - F1: bright walls along the cell borders.
    Ridges,
    /// A flat random value per cell.
    Cell,
}

impl VoronoiMode {
    pub const ALL: [VoronoiMode; 4] =
        [VoronoiMode::F1, VoronoiMode::F2, VoronoiMode::Ridges, VoronoiMode::Cell];
    pub fn name(self) -> &'static str {
        match self {
            VoronoiMode::F1 => "f1",
            VoronoiMode::F2 => "f2",
            VoronoiMode::Ridges => "f2 - f1",
            VoronoiMode::Cell => "cell id",
        }
    }
    fn code(self) -> f32 {
        match self {
            VoronoiMode::F1 => 0.0,
            VoronoiMode::F2 => 1.0,
            VoronoiMode::Ridges => 2.0,
            VoronoiMode::Cell => 3.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum MathOp {
    #[default]
    Add,
    Sub,
    Mul,
    Div,
    Min,
    Max,
    Pow,
}

impl MathOp {
    pub const ALL: [MathOp; 7] = [
        MathOp::Add,
        MathOp::Sub,
        MathOp::Mul,
        MathOp::Div,
        MathOp::Min,
        MathOp::Max,
        MathOp::Pow,
    ];
    pub fn name(self) -> &'static str {
        match self {
            MathOp::Add => "add",
            MathOp::Sub => "subtract",
            MathOp::Mul => "multiply",
            MathOp::Div => "divide",
            MathOp::Min => "min",
            MathOp::Max => "max",
            MathOp::Pow => "power",
        }
    }
    fn code(self) -> f32 {
        match self {
            MathOp::Add => 0.0,
            MathOp::Sub => 1.0,
            MathOp::Mul => 2.0,
            MathOp::Div => 3.0,
            MathOp::Min => 4.0,
            MathOp::Max => 5.0,
            MathOp::Pow => 6.0,
        }
    }
}

/// A shape drawn in field space, as a mask.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ShapeKind {
    /// An ellipse: the two sizes are its radii.
    #[default]
    Ellipse,
    /// A rectangle: the two sizes are its half-extents.
    Box,
}

impl ShapeKind {
    pub const ALL: [ShapeKind; 2] = [ShapeKind::Ellipse, ShapeKind::Box];
    pub fn name(self) -> &'static str {
        match self {
            ShapeKind::Ellipse => "ellipse",
            ShapeKind::Box => "box",
        }
    }
    fn code(self) -> f32 {
        match self {
            ShapeKind::Ellipse => 0.0,
            ShapeKind::Box => 1.0,
        }
    }
}

/// Which axis a Gradient differentiates along.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Axis {
    #[default]
    X,
    Y,
}

impl Axis {
    pub const ALL: [Axis; 2] = [Axis::X, Axis::Y];
    pub fn name(self) -> &'static str {
        match self {
            Axis::X => "x",
            Axis::Y => "y",
        }
    }
}

/// How a Warp displaces the coordinates it hands upstream.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum WarpMode {
    /// Two driver fields, straight into x and y.
    #[default]
    Offset,
    /// One driver field read as a POTENTIAL: the offset is its curl,
    /// (dF/dy, -dF/dx), which is divergence-free — so the warp swirls the
    /// domain without pooling or tearing it.
    Curl,
}

impl WarpMode {
    pub const ALL: [WarpMode; 2] = [WarpMode::Offset, WarpMode::Curl];
    pub fn name(self) -> &'static str {
        match self {
            WarpMode::Offset => "offset",
            WarpMode::Curl => "curl",
        }
    }
}

/// One node of a noise graph.
///
/// Every numeric field here is a LITERAL that an input can override: wire a
/// field into `scale` and the literal stops being read. That is what "an
/// fBm whose parameters are themselves noise" means in practice, and it is
/// why the input list and the params below run in step ([`FieldNode::inputs`]).
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FieldNode {
    /// The coordinate itself — where a graph that depends on WHERE it is
    /// starts. Normalized so 0..1 spans the format's width.
    Coord { kind: CoordKind, scale: f32 },
    /// The slider. Named, because its name is how it appears OUTSIDE the
    /// graph: every Slider becomes a widget on the owning Noise node's panel
    /// and a channel the patch bay can drive.
    Const {
        value: f32,
        #[cfg_attr(feature = "serde", serde(default))]
        name: String,
    },
    /// A data input: a nodule on the owning Noise node with NO widget. The
    /// value arrives from outside — a cable in the comp's link mode — and
    /// the node hands it into the graph. A Slider without the slider.
    Input {
        name: String,
        #[cfg_attr(feature = "serde", serde(default))]
        value: f32,
    },
    Fbm {
        mode: FieldFbmMode,
        scale: [f32; 2],
        offset: [f32; 2],
        lacunarity: f32,
        gain: f32,
        seed: f32,
        /// Multiplier on the Noise node's global speed. 0 freezes THIS
        /// generator while the rest of the graph keeps moving; 2 runs it at
        /// double. Not drivable, like `octaves`: it is a rate, and a rate
        /// that varies per pixel is not a rate.
        speed: f32,
        /// NOT drivable: it is the loop bound. A per-pixel octave count
        /// costs the worst case anyway and quantises into visible bands.
        octaves: u32,
        /// Uniform scale: multiplies BOTH `scale` axes, so the noise can be
        /// sized in one move without losing its aspect. Socket 7 (appended,
        /// so every older wire and channel keeps its index); 1 for a setup
        /// from before it.
        #[cfg_attr(feature = "serde", serde(default = "one_f32"))]
        uniform: f32,
    },
    Voronoi {
        mode: VoronoiMode,
        scale: [f32; 2],
        /// 0 = a regular lattice, 1 = fully scattered feature points.
        jitter: f32,
        seed: f32,
        /// Multiplier on the Noise node's global speed; see `Fbm::speed`.
        speed: f32,
    },
    Math { op: MathOp, a: f32, b: f32 },
    Mix { a: f32, b: f32, t: f32 },
    Remap { in_lo: f32, in_hi: f32, out_lo: f32, out_hi: f32, clamp: bool },
    /// Rewrites the coordinate its source is evaluated at — the domain warp
    /// that makes a noise graph worth having.
    Warp { mode: WarpMode, amount: f32 },
    /// Shifts the seed handed to everything above it, so one sub-graph can
    /// be reused as several uncorrelated ones.
    Seed { shift: f32 },
    /// Finite-difference derivative of its input.
    Gradient { axis: Axis, step: f32 },
    /// A roto drawn IN the field graph: a closed spline, 1 inside and 0
    /// outside, feathered per point. The same outline machinery as the comp
    /// Spline (`spline_polyline`), but its points live in FIELD space — x is
    /// 0..1 across the width, y in the same units — so it is read wherever
    /// the graph asks: a Warp above it bends the roto's edge, and its offset
    /// and feather are sockets like any other. Points are edited in the
    /// viewer while inside the Noise node.
    Spline {
        points: Vec<crate::graph::SplinePoint>,
        mode: crate::graph::SplineMode,
        /// Multiplies every point's own feather.
        feather: f32,
        offset: [f32; 2],
    },
    /// A mask drawn IN the field graph: 1 inside the shape, 0 outside,
    /// feathered across the edge.
    ///
    /// The other Mask node reads a matte from the comp; this one needs
    /// nothing from outside, so a region can be defined where it is used.
    /// Every one of its numbers is drivable like any other socket, so the
    /// shape can be moved, resized or softened by another field — a circle
    /// whose radius is itself noise is a perfectly good mask.
    Shape { kind: ShapeKind, center: [f32; 2], size: [f32; 2], feather: f32 },
    /// The MASK wired into the owning Noise node, as a field.
    ///
    /// This is the one thing in a field graph that comes from outside it: a
    /// spline, a key, any matte in the comp, read as a number per pixel. Mix
    /// something through it and the field changes only where the mask is —
    /// finer noise inside a shape, say.
    ///
    /// Sampled at the PIXEL, not at the field coordinate: the mask is a
    /// region of the picture, so a Warp further up the graph moves the noise
    /// through it rather than dragging it along. Unwired it reads 1, the
    /// same convention every other matte input here follows.
    MaskIn,
    /// What the graph evaluates to. A marker, not a transform: it passes its
    /// input straight through.
    ///
    /// The viewer marker inside a dive is a PREVIEW — you move it around to
    /// look at things — whereas this is structural. Keeping the two apart
    /// means parking the viewer on an intermediate node to inspect it cannot
    /// silently change what the Noise node renders.
    Output,
}

#[cfg(feature = "serde")]
fn one_f32() -> f32 {
    1.0
}

impl Default for FieldNode {
    fn default() -> Self {
        FieldNode::Fbm {
            mode: FieldFbmMode::Fbm,
            scale: [0.25, 0.25],
            offset: [0.0, 0.0],
            lacunarity: 2.0,
            gain: 0.5,
            seed: 0.0,
            octaves: 5,
            speed: 1.0,
            uniform: 1.0,
        }
    }
}

/// An input socket: its label, and the uniform component holding the
/// literal it overrides.
pub struct FieldInput {
    pub name: &'static str,
    /// Offset of the literal within the node's own uniform block, in floats.
    pub slot: usize,
}

const fn fi(name: &'static str, slot: usize) -> FieldInput {
    FieldInput { name, slot }
}

impl FieldNode {
    pub fn name(&self) -> &'static str {
        match self {
            FieldNode::Coord { .. } => "Coord",
            // A Const IS the slider: a keyframeable number that wires into any
            // socket by an ordinary edge. That is the one way a float gets
            // driven in a field graph — no second cable system.
            FieldNode::Const { .. } => "Slider",
            FieldNode::Input { .. } => "Input",
            FieldNode::Fbm { .. } => "fBm",
            FieldNode::Voronoi { .. } => "Voronoi",
            FieldNode::Math { .. } => "Math",
            FieldNode::Mix { .. } => "Mix",
            FieldNode::Remap { .. } => "Remap",
            FieldNode::Warp { .. } => "Warp",
            FieldNode::Seed { .. } => "Seed",
            FieldNode::Gradient { .. } => "Gradient",
            FieldNode::Output => "Output",
            FieldNode::MaskIn => "Mask",
            FieldNode::Shape { .. } => "Shape",
            FieldNode::Spline { .. } => "Roto",
        }
    }

    /// Sockets, in order. Index 0 of a node that HAS a source (Warp, Seed,
    /// Gradient) is that source — the field being transformed — and carries
    /// no literal, so its `slot` is unused.
    pub fn inputs(&self) -> Vec<FieldInput> {
        match self {
            FieldNode::Coord { .. }
            | FieldNode::Const { .. }
            | FieldNode::Input { .. }
            | FieldNode::MaskIn => Vec::new(),
            FieldNode::Fbm { .. } => vec![
                fi("scale x", 0),
                fi("scale y", 1),
                fi("offset x", 2),
                fi("offset y", 3),
                fi("lacunarity", 4),
                fi("gain", 5),
                fi("seed", 6),
                fi("scale", 10),
            ],
            FieldNode::Voronoi { .. } => {
                vec![fi("scale x", 0), fi("scale y", 1), fi("jitter", 2), fi("seed", 3)]
            }
            // Header 0 is the outline's counts; the literals live in header
            // 1 (lanes 4..6), points and feathers after.
            FieldNode::Spline { .. } => {
                vec![fi("offset x", 4), fi("offset y", 5), fi("feather", 6)]
            }
            FieldNode::Shape { .. } => vec![
                fi("centre x", 0),
                fi("centre y", 1),
                fi("size x", 2),
                fi("size y", 3),
                fi("feather", 4),
            ],
            FieldNode::Math { .. } => vec![fi("a", 0), fi("b", 1)],
            FieldNode::Mix { .. } => vec![fi("a", 0), fi("b", 1), fi("t", 2)],
            FieldNode::Remap { .. } => vec![fi("field", 0)],
            FieldNode::Warp { mode, .. } => match mode {
                WarpMode::Offset => {
                    vec![fi("field", 0), fi("x", 1), fi("y", 2), fi("amount", 3)]
                }
                WarpMode::Curl => vec![fi("field", 0), fi("potential", 1), fi("amount", 3)],
            },
            FieldNode::Seed { .. } => vec![fi("field", 0), fi("shift", 1)],
            FieldNode::Gradient { .. } => vec![fi("field", 0)],
            FieldNode::Output => vec![fi("field", 0)],
        }
    }

    /// The literal behind socket `i` — the number an unwired socket reads.
    /// This is what animation and the patch bay drive: a keyed or cabled
    /// field param is this literal changing over time, exactly as a comp
    /// node's param would. Sockets that carry a FIELD rather than a number
    /// (a Warp's source) have none.
    pub fn literal(&self, i: usize) -> Option<f32> {
        Some(match (self, i) {
            // No socket, but a channel: the slider's own value.
            (FieldNode::Const { value, .. }, 0) => *value,
            (FieldNode::Input { value, .. }, 0) => *value,
            (FieldNode::Fbm { scale, .. }, 0) => scale[0],
            (FieldNode::Fbm { scale, .. }, 1) => scale[1],
            (FieldNode::Fbm { offset, .. }, 2) => offset[0],
            (FieldNode::Fbm { offset, .. }, 3) => offset[1],
            (FieldNode::Fbm { lacunarity, .. }, 4) => *lacunarity,
            (FieldNode::Fbm { gain, .. }, 5) => *gain,
            (FieldNode::Fbm { seed, .. }, 6) => *seed,
            (FieldNode::Fbm { uniform, .. }, 7) => *uniform,
            (FieldNode::Voronoi { scale, .. }, 0) => scale[0],
            (FieldNode::Voronoi { scale, .. }, 1) => scale[1],
            (FieldNode::Voronoi { jitter, .. }, 2) => *jitter,
            (FieldNode::Voronoi { seed, .. }, 3) => *seed,
            (FieldNode::Shape { center, .. }, 0) => center[0],
            (FieldNode::Shape { center, .. }, 1) => center[1],
            (FieldNode::Shape { size, .. }, 2) => size[0],
            (FieldNode::Shape { size, .. }, 3) => size[1],
            (FieldNode::Shape { feather, .. }, 4) => *feather,
            (FieldNode::Math { a, .. }, 0) => *a,
            (FieldNode::Math { b, .. }, 1) => *b,
            (FieldNode::Mix { a, .. }, 0) => *a,
            (FieldNode::Mix { b, .. }, 1) => *b,
            (FieldNode::Mix { t, .. }, 2) => *t,
            (FieldNode::Warp { mode: WarpMode::Offset, amount }, 3) => *amount,
            (FieldNode::Warp { mode: WarpMode::Curl, amount }, 2) => *amount,
            (FieldNode::Seed { shift }, 1) => *shift,
            _ => return None,
        })
    }

    /// Write the literal behind socket `i`; false if there is none.
    pub fn set_literal(&mut self, i: usize, v: f32) -> bool {
        let slot: &mut f32 = match (self, i) {
            (FieldNode::Const { value, .. }, 0) => value,
            (FieldNode::Input { value, .. }, 0) => value,
            (FieldNode::Fbm { scale, .. }, 0) => &mut scale[0],
            (FieldNode::Fbm { scale, .. }, 1) => &mut scale[1],
            (FieldNode::Fbm { offset, .. }, 2) => &mut offset[0],
            (FieldNode::Fbm { offset, .. }, 3) => &mut offset[1],
            (FieldNode::Fbm { lacunarity, .. }, 4) => lacunarity,
            (FieldNode::Fbm { gain, .. }, 5) => gain,
            (FieldNode::Fbm { seed, .. }, 6) => seed,
            (FieldNode::Fbm { uniform, .. }, 7) => uniform,
            (FieldNode::Voronoi { scale, .. }, 0) => &mut scale[0],
            (FieldNode::Voronoi { scale, .. }, 1) => &mut scale[1],
            (FieldNode::Voronoi { jitter, .. }, 2) => jitter,
            (FieldNode::Voronoi { seed, .. }, 3) => seed,
            (FieldNode::Shape { center, .. }, 0) => &mut center[0],
            (FieldNode::Shape { center, .. }, 1) => &mut center[1],
            (FieldNode::Shape { size, .. }, 2) => &mut size[0],
            (FieldNode::Shape { size, .. }, 3) => &mut size[1],
            (FieldNode::Shape { feather, .. }, 4) => feather,
            (FieldNode::Math { a, .. }, 0) => a,
            (FieldNode::Math { b, .. }, 1) => b,
            (FieldNode::Mix { a, .. }, 0) => a,
            (FieldNode::Mix { b, .. }, 1) => b,
            (FieldNode::Mix { t, .. }, 2) => t,
            (FieldNode::Warp { mode: WarpMode::Offset, amount }, 3) => amount,
            (FieldNode::Warp { mode: WarpMode::Curl, amount }, 2) => amount,
            (FieldNode::Seed { shift }, 1) => shift,
            _ => return false,
        };
        *slot = v;
        true
    }

    /// How many animatable channels the node has: one per socket, plus the
    /// slider's own value, which has no socket to sit behind.
    pub fn channel_count(&self) -> usize {
        match self {
            FieldNode::Const { .. } | FieldNode::Input { .. } => 1,
            _ => self.inputs().len(),
        }
    }

    /// vec4s this node needs in the shared uniform block.
    pub fn slots(&self) -> usize {
        match self {
            // fBm's seven drivable params plus its octave count fill two
            // vec4s exactly, so its mode needs a third lane of its own.
            FieldNode::Fbm { .. } => 3,
            FieldNode::Voronoi { .. } | FieldNode::Warp { .. } | FieldNode::Shape { .. } => 2,
            // Two headers, then the subdivided outline: points 2 per vec4,
            // feathers 4 per vec4. The COUNT is topology (it moves the slot
            // span); everything else in here is a uniform.
            FieldNode::Spline { points, .. } => {
                let n = crate::codegen::spline_samples(points.len());
                2 + n.div_ceil(2) + n.div_ceil(4)
            }
            _ => 1,
        }
    }

    /// Whether this kind reads a field on input 0 (rather than a literal).
    fn has_source(&self) -> bool {
        matches!(
            self,
            FieldNode::Warp { .. }
                | FieldNode::Seed { .. }
                | FieldNode::Gradient { .. }
                | FieldNode::Remap { .. }
                | FieldNode::Output
        )
    }

    /// The node's own numbers, written into `v` at `base` (float index).
    pub fn pack(&self, base: usize, v: &mut [f32]) {
        let put = |v: &mut [f32], i: usize, x: f32| {
            if base + i < v.len() {
                v[base + i] = x;
            }
        };
        match *self {
            FieldNode::Coord { kind, scale } => {
                put(v, 0, scale);
                put(
                    v,
                    1,
                    match kind {
                        CoordKind::X => 0.0,
                        CoordKind::Y => 1.0,
                        CoordKind::Radius => 2.0,
                        CoordKind::Angle => 3.0,
                    },
                );
            }
            FieldNode::Const { value, .. } | FieldNode::Input { value, .. } => put(v, 0, value),
            FieldNode::Fbm {
                mode,
                scale,
                offset,
                lacunarity,
                gain,
                seed,
                octaves,
                speed,
                uniform,
            } => {
                put(v, 0, scale[0]);
                put(v, 1, scale[1]);
                put(v, 2, offset[0]);
                put(v, 3, offset[1]);
                put(v, 4, lacunarity);
                put(v, 5, gain);
                put(v, 6, seed);
                put(v, 7, octaves.clamp(1, FBM_MAX_OCTAVES) as f32);
                // The mode is a uniform, not topology: switching fold is a
                // param edit, the same as it is on the image-side fBm.
                put(v, 8, mode.code());
                put(v, 9, speed);
                put(v, 10, uniform);
            }
            FieldNode::Voronoi { mode, scale, jitter, seed, speed } => {
                put(v, 0, scale[0]);
                put(v, 1, scale[1]);
                put(v, 2, jitter);
                put(v, 3, seed);
                put(v, 4, mode.code());
                put(v, 5, speed);
            }
            FieldNode::Math { op, a, b } => {
                put(v, 0, a);
                put(v, 1, b);
                put(v, 2, op.code());
            }
            FieldNode::Mix { a, b, t } => {
                put(v, 0, a);
                put(v, 1, b);
                put(v, 2, t);
            }
            FieldNode::Remap { in_lo, in_hi, out_lo, out_hi, clamp } => {
                put(v, 0, in_lo);
                put(v, 1, in_hi);
                put(v, 2, out_lo);
                put(v, 3, out_hi);
                let _ = clamp;
            }
            FieldNode::Warp { mode, amount } => {
                put(v, 0, 0.0);
                put(v, 1, 0.0);
                put(v, 2, 0.0);
                put(v, 3, amount);
                put(v, 4, if mode == WarpMode::Curl { 1.0 } else { 0.0 });
            }
            FieldNode::Seed { shift } => {
                put(v, 0, 0.0);
                put(v, 1, shift);
            }
            FieldNode::Gradient { step, .. } => put(v, 0, step.max(1e-4)),
            FieldNode::Spline { ref points, mode, feather, offset } => {
                let sp = crate::graph::SplineParams {
                    points: points.clone(),
                    mode,
                    ..Default::default()
                };
                // Field space already, no pixel scale: expand is 0 and the
                // transform identity, so this is pure subdivision.
                let poly = crate::codegen::spline_polyline(&sp, [1.0, 1.0]);
                let n = poly.len();
                let foff = 2 + n.div_ceil(2); // vec4s from base to the feathers
                put(v, 0, n as f32);
                put(v, 1, foff as f32);
                put(v, 4, offset[0]);
                put(v, 5, offset[1]);
                put(v, 6, feather);
                for (i, (pt, fe)) in poly.iter().enumerate() {
                    put(v, 8 + i * 2, pt[0]);
                    put(v, 8 + i * 2 + 1, pt[1]);
                    put(v, foff * 4 + i, *fe);
                }
            }
            FieldNode::Shape { kind, center, size, feather } => {
                put(v, 0, center[0]);
                put(v, 1, center[1]);
                put(v, 2, size[0]);
                put(v, 3, size[1]);
                put(v, 4, feather);
                put(v, 5, kind.code());
            }
            // Nothing to pack: markers and inputs carry no numbers.
            FieldNode::Output | FieldNode::MaskIn => {}
        }
    }

    /// Structure that must force a recompile: everything the emitted code
    /// branches on, and nothing the uniform carries.
    pub fn topo_tag(&self) -> u64 {
        match self {
            FieldNode::Coord { kind, .. } => 1 + (*kind as u64) * 16,
            FieldNode::Const { .. } | FieldNode::Input { .. } => 2,
            FieldNode::Fbm { .. } => 3,
            FieldNode::Voronoi { .. } => 4,
            FieldNode::Math { .. } => 5,
            FieldNode::Mix { .. } => 6,
            // The clamp is a branch in the emitted code.
            FieldNode::Remap { clamp, .. } => 7 + (*clamp as u64) * 16,
            // The two warp modes read a different number of inputs.
            FieldNode::Warp { mode, .. } => 8 + (*mode as u64) * 16,
            FieldNode::Seed { .. } => 9,
            FieldNode::Gradient { axis, .. } => 10 + (*axis as u64) * 16,
            FieldNode::Output => 11,
            FieldNode::MaskIn => 12,
            FieldNode::Shape { .. } => 13,
            // The subdivided count sets the slot span, like the comp Spline.
            FieldNode::Spline { points, .. } => {
                14 + (crate::codegen::spline_samples(points.len()) as u64) * 16
            }
        }
    }
}

/// Where each field node's uniform block starts, in vec4s from the owning
/// Noise node's base, and the total the Noise node must reserve.
///
/// Sorted by node id so the compiler and the packer agree without having to
/// pass a layout between them.
pub fn slot_map(g: &FieldGraph) -> (BTreeMap<NodeId, usize>, usize) {
    let mut map = BTreeMap::new();
    let mut at = 0usize;
    for (id, f) in &g.nodes {
        map.insert(*id, at);
        at += f.slots();
    }
    (map, at)
}

/// Pack a whole sub-graph's numbers into the block reserved at `base`.
pub fn pack(g: &FieldGraph, base: usize, v: &mut [f32]) {
    let (map, _) = slot_map(g);
    for (id, off) in &map {
        if let Some(f) = g.nodes.get(id) {
            f.pack(base + off * 4, v);
        }
    }
}

/// Everything about a sub-graph that changes the emitted code: kinds, the
/// wiring, and the node ids the layout is keyed on.
pub fn topo_tag(g: &FieldGraph) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    let mut mix = |x: u64| {
        h ^= x;
        h = h.wrapping_mul(0x100000001b3);
    };
    for (id, f) in &g.nodes {
        mix(*id);
        // The octave count is a loop BOUND, not a branch, so it stays out of
        // the tag: turning octaves up is a param edit.
        mix(f.topo_tag());
    }
    for ((dst, port), src) in &g.edges {
        mix(*dst ^ (*port as u64) << 32);
        mix(*src);
    }
    mix(g.out.unwrap_or(u64::MAX));
    h
}

/// The shared WGSL every noise graph calls into. Appended once, ahead of the
/// per-node functions.
///
/// The gradient noise itself is NOT redefined here: it reuses `fbm_noise` /
/// `fbm_core` from the main prelude, so a field fBm and the image-side fBm
/// are the same noise and match pixel for pixel.
pub const FIELD_WGSL: &str = r#"
// A feature point inside cell `c`, scattered by `jit`.
fn nzf_cell(c: vec2f, jit: f32, seed: u32, t: f32) -> vec2f {
    let h = fbm_hash(bitcast<u32>(i32(c.x)) ^ fbm_hash(bitcast<u32>(i32(c.y)) ^ seed));
    let rx = f32(h & 0xffffu) / 65535.0;
    let ry = f32((h >> 16u) & 0xffffu) / 65535.0;
    var o = (vec2f(rx, ry) - vec2f(0.5)) * clamp(jit, 0.0, 1.0);
    // Cellular noise is 2D, so it has no third axis to evolve along the way
    // fBm does. Instead each feature point ORBITS inside its own cell: the
    // pattern boils in place rather than sliding past.
    if (t != 0.0) {
        let sn = sin(t);
        let cs = cos(t);
        o = vec2f(cs * o.x - sn * o.y, sn * o.x + cs * o.y);
    }
    return c + vec2f(0.5) + o;
}
// A flat random value for the cell a point lands in.
fn nzf_cell_value(c: vec2f, seed: u32) -> f32 {
    let h = fbm_hash(bitcast<u32>(i32(c.x)) ^ fbm_hash(bitcast<u32>(i32(c.y)) ^ (seed + 7919u)));
    return f32(h & 0xffffffu) / 16777215.0;
}
// Worley / cellular: the 3x3 neighbourhood is enough while the jitter stays
// inside the cell, which is what clamping it to 0..1 guarantees.
fn nzf_voronoi(p: vec2f, jit: f32, seed: u32, mode: f32, t: f32) -> f32 {
    let base = floor(p);
    var f1 = 1e9;
    var f2 = 1e9;
    var best = vec2f(0.0);
    for (var j = -1; j <= 1; j++) {
        for (var i = -1; i <= 1; i++) {
            let c = base + vec2f(f32(i), f32(j));
            let d = distance(p, nzf_cell(c, jit, seed, t));
            if (d < f1) {
                f2 = f1;
                f1 = d;
                best = c;
            } else if (d < f2) {
                f2 = d;
            }
        }
    }
    if (mode == 1.0) { return clamp(f2, 0.0, 1.0); }
    if (mode == 2.0) { return clamp(f2 - f1, 0.0, 1.0); }
    if (mode == 3.0) { return nzf_cell_value(best, seed); }
    return clamp(f1, 0.0, 1.0);
}
// A shape as a mask: signed distance in field units, feathered to
// coverage. Feather 0 is a hard edge, which is why the half-width is
// floored rather than the smoothstep being skipped.
fn nzf_shape(p: vec2f, c: vec2f, sz: vec2f, feather: f32, kind: f32) -> f32 {
    let s = max(abs(sz), vec2f(1e-5));
    var d: f32;
    if (kind == 1.0) {
        let q = abs(p - c) - s;
        d = length(max(q, vec2f(0.0))) + min(max(q.x, q.y), 0.0);
    } else {
        // An ellipse has no closed-form distance; scaling the unit-circle
        // one by the smaller radius is the usual approximation and is
        // exact for a circle.
        let q = (p - c) / s;
        d = (length(q) - 1.0) * min(s.x, s.y);
    }
    let h = max(feather * 0.5, 1e-5);
    return 1.0 - smoothstep(-h, h, d);
}
// A closed outline as a mask, in FIELD units: the comp Spline's walk without
// its pixel scale. Header 0 = (count, vec4 offset of the feathers); header 1
// = (offset x, offset y, feather scale); points 2 per vec4 from base+2.
fn nzf_spt(base: u32, i: u32) -> vec2f {
    let v = P.v[base + 2u + (i >> 1u)];
    return select(v.xy, v.zw, (i & 1u) == 1u);
}
fn nzf_sfeather(base: u32, i: u32) -> f32 {
    let q = P.v[base + u32(P.v[base].y) + (i >> 2u)];
    let k = i & 3u;
    if (k == 0u) { return q.x; }
    if (k == 1u) { return q.y; }
    if (k == 2u) { return q.z; }
    return q.w;
}
fn nzf_spline(p0: vec2f, base: u32, off: vec2f, fscale: f32) -> f32 {
    let n = u32(P.v[base].x);
    if (n < 3u) { return 0.0; }
    let p = p0 - off;
    var d = 1e20;
    var feath = 0.0;
    var sgn = 1.0;
    var j = n - 1u;
    for (var i = 0u; i < n; i = i + 1u) {
        let a = nzf_spt(base, j);
        let b = nzf_spt(base, i);
        let e = a - b;
        let w = p - b;
        let t = clamp(dot(w, e) / max(dot(e, e), 1e-12), 0.0, 1.0);
        let bq = w - e * t;
        let dd = dot(bq, bq);
        if (dd < d) {
            d = dd;
            feath = mix(nzf_sfeather(base, i), nzf_sfeather(base, j), t);
        }
        // Winding, so the outline's inside is inside whichever way it was
        // drawn.
        // Each comparison PARENTHESISED: inside a `vec3<bool>(...)`
        // constructor WGSL reads a bare `p.y < a.y` as the start of a
        // template list and fails to parse. The comp's spline_mask has the
        // same parentheses for the same reason.
        let c = vec3<bool>((p.y >= b.y), (p.y < a.y), (e.x * w.y > e.y * w.x));
        if (all(c) || !any(c)) { sgn = -sgn; }
        j = i;
    }
    let dist = sqrt(d) * sgn;             // negative inside
    let h = max(feath * fscale * 0.5, 1e-6);
    return 1.0 - smoothstep(-h, h, dist);
}
fn nzf_math(a: f32, b: f32, op: f32) -> f32 {
    if (op == 1.0) { return a - b; }
    if (op == 2.0) { return a * b; }
    if (op == 3.0) { return a / select(b, 1e-5, abs(b) < 1e-5); }
    if (op == 4.0) { return min(a, b); }
    if (op == 5.0) { return max(a, b); }
    if (op == 6.0) { return pow(max(a, 0.0), b); }
    return a + b;
}
"#;

/// Compile a noise sub-graph into WGSL functions.
///
/// Returns the name of the entry point — the function for the sub-graph's
/// output node — with every function it depends on inserted into `fns`
/// (a BTreeMap keyed by name, emitted in dependency order because a
/// definition is written only after the ones it calls).
///
/// `None` means there is nothing to draw: no output node, or a cycle. WGSL
/// cannot express recursion, so a cycle has to be refused HERE rather than
/// handed to naga as a shader that will not compile.
pub fn compile(
    owner: NodeId,
    g: &FieldGraph,
    base: usize,
    // WGSL expression for the graph's animation phase: the owning node's
    // speed times the tick. Passed in rather than read from a known slot, so
    // this module needs to know nothing about the Noise node's own layout.
    phase: &str,
    fns: &mut Vec<(String, String)>,
) -> Option<String> {
    let out = g.output()?;
    let (map, _) = slot_map(g);
    let mut done: BTreeMap<NodeId, String> = BTreeMap::new();
    let mut stack = Vec::new();
    emit(owner, g, base, phase, out, &map, fns, &mut done, &mut stack)
}

#[allow(clippy::too_many_arguments)]
fn emit(
    owner: NodeId,
    g: &FieldGraph,
    base: usize,
    phase: &str,
    node: NodeId,
    map: &BTreeMap<NodeId, usize>,
    fns: &mut Vec<(String, String)>,
    done: &mut BTreeMap<NodeId, String>,
    stack: &mut Vec<NodeId>,
) -> Option<String> {
    if let Some(name) = done.get(&node) {
        return Some(name.clone());
    }
    if stack.contains(&node) {
        return None; // a cycle: no recursion in WGSL, so refuse it
    }
    let Some(f) = g.nodes.get(&node) else { return None };
    let off = *map.get(&node)?;
    let b = base + off;
    stack.push(node);

    // Resolve every input first: a driver's function must be DEFINED before
    // the function that calls it, which is what pushing them into `fns`
    // ahead of this one guarantees.
    let ins = f.inputs();
    let mut driver: Vec<Option<String>> = Vec::with_capacity(ins.len());
    for (i, _) in ins.iter().enumerate() {
        driver.push(match g.input_of(node, i) {
            Some(src) => Some(emit(owner, g, base, phase, src, map, fns, done, stack)?),
            None => None,
        });
    }
    stack.pop();

    // A literal reads the uniform lane its socket overrides.
    let lit = |slot: usize| format!("P.v[{}u].{}", b + slot / 4, "xyzw".as_bytes()[slot % 4] as char);
    // The value of input `i` at coordinate `q`: the driver's call, or the
    // literal the socket stands for. Input 0 of a transforming node is the
    // FIELD it transforms, not a number — unwired, it is silence.
    let src_port = f.has_source();
    let arg = |i: usize, q: &str| -> String {
        match (&driver[i], ins.get(i)) {
            (Some(name), _) => format!("{name}({q}, sd, m)"),
            (None, _) if src_port && i == 0 => "0.0".into(),
            (None, Some(inp)) => lit(inp.slot),
            (None, None) => "0.0".into(),
        }
    };

    let name = format!("nz{owner}_{node}");
    let mut body = String::new();
    writeln!(body, "fn {name}(p: vec2f, sd: f32, m: f32) -> f32 {{").unwrap();
    match f {
        FieldNode::Coord { .. } => {
            writeln!(body, "    let s = max({}, 1e-6);", lit(0)).unwrap();
            writeln!(body, "    let k = {};", lit(1)).unwrap();
            writeln!(body, "    let c = p - vec2f(0.5);").unwrap();
            writeln!(body, "    if (k == 1.0) {{ return p.y * s; }}").unwrap();
            writeln!(body, "    if (k == 2.0) {{ return length(c) * s; }}").unwrap();
            writeln!(
                body,
                "    if (k == 3.0) {{ return atan2(c.y, c.x) / 3.14159265 * s; }}"
            )
            .unwrap();
            writeln!(body, "    return p.x * s;").unwrap();
        }
        FieldNode::Const { .. } | FieldNode::Input { .. } => {
            writeln!(body, "    return {};", lit(0)).unwrap();
        }
        FieldNode::Fbm { .. } => {
            writeln!(body, "    let su = {};", arg(7, "p")).unwrap();
            writeln!(body, "    let sx = max(abs({} * su), 1e-5);", arg(0, "p")).unwrap();
            writeln!(body, "    let sy = max(abs({} * su), 1e-5);", arg(1, "p")).unwrap();
            writeln!(body, "    let ox = {};", arg(2, "p")).unwrap();
            writeln!(body, "    let oy = {};", arg(3, "p")).unwrap();
            writeln!(body, "    let lac = {};", arg(4, "p")).unwrap();
            writeln!(body, "    let gn = {};", arg(5, "p")).unwrap();
            writeln!(body, "    let sdv = {} + sd;", arg(6, "p")).unwrap();
            // The z axis IS the evolution: the same third dimension the
            // image-side fBm calls `evolve`.
            writeln!(
                body,
                "    let q = vec3f((p.x + ox) / sx, (p.y + oy) / sy, ({phase}) * P.v[{}u].y);",
                b + 2
            )
            .unwrap();
            writeln!(
                body,
                "    return fbm_core(q, u32(P.v[{}u].w), lac, gn, u32(max(sdv, 0.0)), P.v[{}u].x);",
                b + 1,
                b + 2
            )
            .unwrap();
        }
        FieldNode::Voronoi { .. } => {
            writeln!(body, "    let sx = max(abs({}), 1e-5);", arg(0, "p")).unwrap();
            writeln!(body, "    let sy = max(abs({}), 1e-5);", arg(1, "p")).unwrap();
            writeln!(body, "    let jit = {};", arg(2, "p")).unwrap();
            writeln!(body, "    let sdv = {} + sd;", arg(3, "p")).unwrap();
            writeln!(
                body,
                "    return nzf_voronoi(vec2f(p.x / sx, p.y / sy), jit, u32(max(sdv, 0.0)), P.v[{}u].x, ({phase}) * P.v[{}u].y);",
                b + 1,
                b + 1
            )
            .unwrap();
        }
        FieldNode::Math { .. } => {
            writeln!(
                body,
                "    return nzf_math({}, {}, {});",
                arg(0, "p"),
                arg(1, "p"),
                lit(2)
            )
            .unwrap();
        }
        FieldNode::Mix { .. } => {
            writeln!(
                body,
                "    return mix({}, {}, clamp({}, 0.0, 1.0));",
                arg(0, "p"),
                arg(1, "p"),
                arg(2, "p")
            )
            .unwrap();
        }
        FieldNode::Remap { clamp, .. } => {
            writeln!(body, "    let x = {};", arg(0, "p")).unwrap();
            writeln!(body, "    let a = {};", lit(0)).unwrap();
            writeln!(body, "    let bb = {};", lit(1)).unwrap();
            writeln!(body, "    let c = {};", lit(2)).unwrap();
            writeln!(body, "    let d = {};", lit(3)).unwrap();
            writeln!(body, "    let t = (x - a) / select(bb - a, 1e-5, abs(bb - a) < 1e-5);")
                .unwrap();
            if *clamp {
                writeln!(body, "    return mix(c, d, clamp(t, 0.0, 1.0));").unwrap();
            } else {
                writeln!(body, "    return mix(c, d, t);").unwrap();
            }
        }
        FieldNode::Warp { mode, .. } => {
            // The drivers are read at the ORIGINAL coordinate; only the
            // source is moved. Reading them at the warped coordinate would
            // be a feedback loop with no fixed point.
            // Curl reads one driver where Offset reads two, so `amount` is
            // the third socket there and the fourth here.
            let amt_i = if *mode == WarpMode::Curl { 2 } else { 3 };
            writeln!(body, "    let amt = {};", arg(amt_i, "p")).unwrap();
            match mode {
                WarpMode::Offset => {
                    writeln!(body, "    let dx = {};", arg(1, "p")).unwrap();
                    writeln!(body, "    let dy = {};", arg(2, "p")).unwrap();
                    writeln!(body, "    let q = p + vec2f(dx, dy) * amt;").unwrap();
                }
                WarpMode::Curl => {
                    // Curl of a scalar potential: (dF/dy, -dF/dx), which is
                    // divergence-free, so the domain swirls instead of
                    // pooling. Central differences, one texel-ish apart.
                    let pot = |q: &str| match &driver[1] {
                        Some(n) => format!("{n}({q}, sd, m)"),
                        None => lit(ins[1].slot),
                    };
                    writeln!(body, "    let e = 1.0 / 512.0;").unwrap();
                    writeln!(body, "    let px1 = {};", pot("p + vec2f(e, 0.0)")).unwrap();
                    writeln!(body, "    let px0 = {};", pot("p - vec2f(e, 0.0)")).unwrap();
                    writeln!(body, "    let py1 = {};", pot("p + vec2f(0.0, e)")).unwrap();
                    writeln!(body, "    let py0 = {};", pot("p - vec2f(0.0, e)")).unwrap();
                    writeln!(
                        body,
                        "    let q = p + vec2f((py1 - py0), -(px1 - px0)) / (2.0 * e) * amt;"
                    )
                    .unwrap();
                }
            }
            match &driver[0] {
                Some(src) => writeln!(body, "    return {src}(q, sd, m);").unwrap(),
                None => writeln!(body, "    return 0.0;").unwrap(),
            }
        }
        FieldNode::Seed { .. } => {
            writeln!(body, "    let sh = {};", arg(1, "p")).unwrap();
            match &driver[0] {
                Some(src) => writeln!(body, "    return {src}(p, sd + sh, m);").unwrap(),
                None => writeln!(body, "    return 0.0;").unwrap(),
            }
        }
        FieldNode::MaskIn => {
            writeln!(body, "    return m;").unwrap();
        }
        FieldNode::Spline { .. } => {
            writeln!(
                body,
                "    return nzf_spline(p, {b}u, vec2f({}, {}), {});",
                arg(0, "p"),
                arg(1, "p"),
                arg(2, "p")
            )
            .unwrap();
        }
        FieldNode::Shape { .. } => {
            writeln!(
                body,
                "    return nzf_shape(p, vec2f({}, {}), vec2f({}, {}), {}, {});",
                arg(0, "p"),
                arg(1, "p"),
                arg(2, "p"),
                arg(3, "p"),
                arg(4, "p"),
                lit(5)
            )
            .unwrap();
        }
        FieldNode::Output => match &driver[0] {
            Some(src) => writeln!(body, "    return {src}(p, sd, m);").unwrap(),
            None => writeln!(body, "    return 0.0;").unwrap(),
        },
        FieldNode::Gradient { axis, .. } => {
            writeln!(body, "    let e = max({}, 1e-4);", lit(0)).unwrap();
            let d = match axis {
                Axis::X => "vec2f(e, 0.0)",
                Axis::Y => "vec2f(0.0, e)",
            };
            match &driver[0] {
                Some(src) => {
                    writeln!(body, "    let a = {src}(p + {d}, sd, m);").unwrap();
                    writeln!(body, "    let b = {src}(p - {d}, sd, m);").unwrap();
                    writeln!(body, "    return (a - b) / (2.0 * e);").unwrap();
                }
                None => writeln!(body, "    return 0.0;").unwrap(),
            }
        }
    }
    writeln!(body, "}}").unwrap();
    fns.push((name.clone(), body));
    done.insert(node, name.clone());
    Some(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn g_with(
        nodes: Vec<(NodeId, FieldNode)>,
        edges: Vec<((NodeId, usize), NodeId)>,
    ) -> FieldGraph {
        let mut g = FieldGraph::default();
        for (id, f) in nodes {
            g.nodes.insert(id, f);
        }
        for (k, v) in edges {
            g.edges.insert(k, v);
        }
        g
    }

    #[test]
    fn a_driven_param_calls_its_driver_instead_of_reading_the_literal() {
        let g = g_with(
            vec![(1, FieldNode::Const { value: 3.0, name: String::new() }), (2, FieldNode::default())],
            // Const drives fBm's scale x (input 0).
            vec![((2, 0), 1)],
        );
        let mut fns = Vec::new();
        let entry = compile(9, &g, 0, "0.0", &mut fns).expect("compiles");
        assert_eq!(entry, "nz9_2");
        let src: String = fns.iter().map(|(_, b)| b.clone()).collect();
        assert!(src.contains("let sx = max(abs(nz9_1(p, sd, m) * su)"), "scale should CALL the driver:\n{src}");
        // The undriven ones still read their own uniform lanes.
        assert!(src.contains("let gn = P.v["), "an unwired socket reads its literal:\n{src}");
    }

    #[test]
    fn a_driver_is_defined_before_the_node_that_calls_it() {
        let g = g_with(
            vec![(5, FieldNode::Const { value: 1.0, name: String::new() }), (2, FieldNode::default())],
            vec![((2, 0), 5)],
        );
        let mut fns = Vec::new();
        compile(0, &g, 0, "0.0", &mut fns).unwrap();
        let names: Vec<&str> = fns.iter().map(|(n, _)| n.as_str()).collect();
        // WGSL wants the callee declared first, and the ids are deliberately
        // out of order so a sort would get this wrong.
        assert_eq!(names, vec!["nz0_5", "nz0_2"]);
    }

    #[test]
    fn a_node_used_twice_is_emitted_once() {
        // One noise driving both scale x and scale y.
        let g = g_with(
            vec![(1, FieldNode::default()), (2, FieldNode::default())],
            vec![((2, 0), 1), ((2, 1), 1)],
        );
        let mut fns = Vec::new();
        compile(0, &g, 0, "0.0", &mut fns).unwrap();
        assert_eq!(fns.len(), 2, "the shared driver should be one function");
        let src: String = fns.iter().map(|(_, b)| b.clone()).collect();
        assert_eq!(src.matches("nz0_1(p, sd, m)").count(), 2, "called at both sockets");
    }

    /// WGSL has no recursion, so a cycle cannot be compiled — it has to be
    /// refused here rather than handed to naga.
    #[test]
    fn a_cycle_is_refused() {
        let g = g_with(
            vec![(1, FieldNode::default()), (2, FieldNode::default())],
            vec![((2, 0), 1), ((1, 0), 2)],
        );
        let mut fns = Vec::new();
        assert!(compile(0, &g, 0, "0.0", &mut fns).is_none());
    }

    #[test]
    fn the_slot_map_gives_every_node_its_own_block() {
        let g = g_with(
            vec![
                (1, FieldNode::Const { value: 0.0, name: String::new() }),
                (2, FieldNode::default()),
                (3, FieldNode::Const { value: 0.0, name: String::new() }),
            ],
            vec![],
        );
        let (map, total) = slot_map(&g);
        assert_eq!(map[&1], 0);
        assert_eq!(map[&2], 1, "Const takes one vec4");
        assert_eq!(map[&3], 4, "fBm takes three: seven drivable params, octaves, mode");
        assert_eq!(total, 5);
    }

    #[test]
    fn rewiring_changes_the_tag_but_turning_a_knob_does_not() {
        let a = g_with(
            vec![(1, FieldNode::Const { value: 1.0, name: String::new() }), (2, FieldNode::default())],
            vec![((2, 0), 1)],
        );
        let mut b = a.clone();
        // A param edit: same shape, different number.
        if let Some(FieldNode::Const { value, .. }) = b.nodes.get_mut(&1) {
            *value = 99.0;
        }
        assert_eq!(topo_tag(&a), topo_tag(&b), "a knob must not recompile");
        let mut c = a.clone();
        c.edges.remove(&(2, 0));
        assert_ne!(topo_tag(&a), topo_tag(&c), "rewiring must recompile");
    }
}

// ---------------------------------------------------------------------------
// CPU evaluation
// ---------------------------------------------------------------------------
//
// The same field, evaluated on the CPU. This is not a convenience: the
// expression/patch system resolves a linked value ONCE per tick on the CPU,
// so a noise graph can only drive an ordinary parameter if it can be asked
// for a number here. Every function below mirrors its WGSL counterpart line
// for line — same hash, same fade, same per-mode normalisation — and
// `cpu_matches_the_gpu` in `engine/tests/field.rs` holds the two to each
// other so this cannot drift into a second, subtly different noise.

fn hash_u32(x: u32) -> u32 {
    let mut h = x;
    h ^= h >> 16;
    h = h.wrapping_mul(0x7feb_352d);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846c_a68b);
    h ^= h >> 16;
    h
}

fn grad3(c: [f32; 3], seed: u32) -> [f32; 3] {
    let ix = (c[0] as i32) as u32;
    let iy = (c[1] as i32) as u32;
    let iz = (c[2] as i32) as u32;
    let mut h = hash_u32(ix ^ hash_u32(iy ^ hash_u32(iz ^ seed)));
    let a = (h & 0xffff) as f32 / 65535.0 * std::f32::consts::TAU;
    h = hash_u32(h);
    let z = (h & 0xffff) as f32 / 65535.0 * 2.0 - 1.0;
    let r = (1.0f32 - z * z).max(0.0).sqrt();
    [a.cos() * r, a.sin() * r, z]
}

/// Gradient (Perlin) noise with the quintic fade, roughly -1..1.
fn noise3(p: [f32; 3], seed: u32) -> f32 {
    let i = [p[0].floor(), p[1].floor(), p[2].floor()];
    let f = [p[0] - i[0], p[1] - i[1], p[2] - i[2]];
    let fade = |t: f32| t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
    let u = [fade(f[0]), fade(f[1]), fade(f[2])];
    let mut acc = 0.0;
    for k in 0..8u32 {
        let c = [(k & 1) as f32, ((k >> 1) & 1) as f32, ((k >> 2) & 1) as f32];
        let g = grad3([i[0] + c[0], i[1] + c[1], i[2] + c[2]], seed);
        let d = [f[0] - c[0], f[1] - c[1], f[2] - c[2]];
        let w = [
            (1.0 - u[0]) * (1.0 - c[0]) + u[0] * c[0],
            (1.0 - u[1]) * (1.0 - c[1]) + u[1] * c[1],
            (1.0 - u[2]) * (1.0 - c[2]) + u[2] * c[2],
        ];
        acc += (g[0] * d[0] + g[1] * d[1] + g[2] * d[2]) * w[0] * w[1] * w[2];
    }
    acc * 1.6
}

/// The octave sum — `fbm_core` in the shader.
fn fbm_cpu(p: [f32; 3], oct: u32, lacunarity: f32, gain: f32, seed: u32, mode: f32) -> f32 {
    let oct = oct.clamp(1, FBM_MAX_OCTAVES);
    let (mut amp, mut freq) = (1.0f32, 1.0f32);
    let (mut sum, mut norm, mut norm2) = (0.0f32, 0.0f32, 0.0f32);
    for i in 0..oct {
        let n = noise3([p[0] * freq, p[1] * freq, p[2] * freq], seed.wrapping_add(i * 8191));
        if mode == 1.0 {
            sum += n.abs() * amp;
        } else if mode == 2.0 {
            let r = 1.0 - n.abs();
            sum += r * r * amp;
        } else {
            sum += n * amp;
        }
        norm += amp;
        norm2 += amp * amp;
        amp *= gain;
        freq *= lacunarity;
    }
    if mode == 0.0 {
        let x = sum / norm2.sqrt().max(1e-6);
        return (0.5 + 0.5 * x * 1.3).clamp(0.0, 1.0);
    }
    let t = sum / norm.max(1e-6);
    if mode == 1.0 {
        return (t * 2.0).clamp(0.0, 1.0);
    }
    ((t - 0.25) * 1.42).clamp(0.0, 1.0)
}

fn cell_point(c: [f32; 2], jit: f32, seed: u32, t: f32) -> [f32; 2] {
    let h = hash_u32((c[0] as i32) as u32 ^ hash_u32((c[1] as i32) as u32 ^ seed));
    let rx = (h & 0xffff) as f32 / 65535.0;
    let ry = ((h >> 16) & 0xffff) as f32 / 65535.0;
    let j = jit.clamp(0.0, 1.0);
    let (mut ox, mut oy) = ((rx - 0.5) * j, (ry - 0.5) * j);
    if t != 0.0 {
        let (sn, cs) = (t.sin(), t.cos());
        let r = (cs * ox - sn * oy, sn * ox + cs * oy);
        ox = r.0;
        oy = r.1;
    }
    [c[0] + 0.5 + ox, c[1] + 0.5 + oy]
}

fn voronoi_cpu(p: [f32; 2], jit: f32, seed: u32, mode: f32, t: f32) -> f32 {
    let base = [p[0].floor(), p[1].floor()];
    let (mut f1, mut f2) = (1e9f32, 1e9f32);
    let mut best = [0.0f32, 0.0];
    for j in -1..=1 {
        for i in -1..=1 {
            let c = [base[0] + i as f32, base[1] + j as f32];
            let q = cell_point(c, jit, seed, t);
            let d = ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2)).sqrt();
            if d < f1 {
                f2 = f1;
                f1 = d;
                best = c;
            } else if d < f2 {
                f2 = d;
            }
        }
    }
    if mode == 1.0 {
        return f2.clamp(0.0, 1.0);
    }
    if mode == 2.0 {
        return (f2 - f1).clamp(0.0, 1.0);
    }
    if mode == 3.0 {
        let h = hash_u32(
            (best[0] as i32) as u32
                ^ hash_u32((best[1] as i32) as u32 ^ seed.wrapping_add(7919)),
        );
        return (h & 0x00ff_ffff) as f32 / 16777215.0;
    }
    f1.clamp(0.0, 1.0)
}

pub(crate) fn math_cpu(a: f32, b: f32, op: MathOp) -> f32 {
    match op {
        MathOp::Add => a + b,
        MathOp::Sub => a - b,
        MathOp::Mul => a * b,
        MathOp::Div => a / if b.abs() < 1e-5 { 1e-5 } else { b },
        MathOp::Min => a.min(b),
        MathOp::Max => a.max(b),
        MathOp::Pow => a.max(0.0).powf(b),
    }
}

/// The value of the graph's output at `p`, with `seed` handed to the top.
///
/// `p` is in the same normalized field space the shader uses: 0..1 across
/// the width. Returns 0 for an empty graph or a cycle, exactly as the
/// compiled version renders black for one.
pub fn eval_cpu(g: &FieldGraph, p: [f32; 2], seed: f32, phase: f32, mask: f32) -> f32 {
    match g.output() {
        Some(out) => eval_node(g, out, p, seed, phase, mask, 0),
        None => 0.0,
    }
}

fn eval_node(
    g: &FieldGraph,
    node: NodeId,
    p: [f32; 2],
    sd: f32,
    phase: f32,
    mask: f32,
    depth: u32,
) -> f32 {
    // The shader refuses a cycle at compile time; here the guard is depth,
    // which also bounds the cost of a legitimately deep graph.
    if depth > 64 {
        return 0.0;
    }
    let Some(f) = g.nodes.get(&node) else { return 0.0 };
    eval_one(f, p, sd, phase, mask, &mut |i, q, s| {
        g.input_of(node, i).map(|src| eval_node(g, src, q, s, phase, mask, depth + 1))
    })
}

/// ONE field node at `p`, with its inputs supplied by the caller:
/// `input(socket, at, seed)` is the wired source evaluated at that point
/// and seed, or None when the socket is unwired (its literal is read).
///
/// This is the whole of a field node's CPU meaning, with the graph walk
/// left outside — so a noise graph and a 2dini graph, which hold the same
/// nodes in different graphs, evaluate them with the same arithmetic and
/// cannot drift apart. The coordinate and seed flow UPSTREAM through
/// `input` exactly as in the shader: a Warp asks its source at the warped
/// point, a Seed at the shifted seed.
pub fn eval_one(
    f: &FieldNode,
    p: [f32; 2],
    sd: f32,
    phase: f32,
    mask: f32,
    input: &mut dyn FnMut(usize, [f32; 2], f32) -> Option<f32>,
) -> f32 {
    let src_port = f.has_source();
    // Input `i` at coordinate `q`: its driver, or the literal it overrides.
    macro_rules! arg {
        ($i:expr, $q:expr, $lit:expr) => {
            match input($i, $q, sd) {
                Some(v) => v,
                None if src_port && $i == 0 => 0.0,
                None => $lit,
            }
        };
    }
    match *f {
        FieldNode::Coord { kind, scale } => {
            let c = [p[0] - 0.5, p[1] - 0.5];
            match kind {
                CoordKind::X => p[0] * scale,
                CoordKind::Y => p[1] * scale,
                CoordKind::Radius => (c[0] * c[0] + c[1] * c[1]).sqrt() * scale,
                CoordKind::Angle => c[1].atan2(c[0]) / std::f32::consts::PI * scale,
            }
        }
        FieldNode::Const { value, .. } | FieldNode::Input { value, .. } => value,
        FieldNode::Fbm { mode, scale, offset, lacunarity, gain, seed, octaves, speed, uniform } => {
            let su = arg!(7, p, uniform);
            let sx = (arg!(0, p, scale[0]) * su).abs().max(1e-5);
            let sy = (arg!(1, p, scale[1]) * su).abs().max(1e-5);
            let ox = arg!(2, p, offset[0]);
            let oy = arg!(3, p, offset[1]);
            let lac = arg!(4, p, lacunarity);
            let gn = arg!(5, p, gain);
            let sdv = arg!(6, p, seed) + sd;
            fbm_cpu(
                [(p[0] + ox) / sx, (p[1] + oy) / sy, phase * speed],
                octaves,
                lac,
                gn,
                sdv.max(0.0) as u32,
                mode.code(),
            )
        }
        FieldNode::Voronoi { mode, scale, jitter, seed, speed } => {
            let sx = arg!(0, p, scale[0]).abs().max(1e-5);
            let sy = arg!(1, p, scale[1]).abs().max(1e-5);
            let jit = arg!(2, p, jitter);
            let sdv = arg!(3, p, seed) + sd;
            voronoi_cpu(
                [p[0] / sx, p[1] / sy],
                jit,
                sdv.max(0.0) as u32,
                mode.code(),
                phase * speed,
            )
        }
        FieldNode::Math { op, a, b } => math_cpu(arg!(0, p, a), arg!(1, p, b), op),
        FieldNode::Mix { a, b, t } => {
            let (x, y, k) = (arg!(0, p, a), arg!(1, p, b), arg!(2, p, t).clamp(0.0, 1.0));
            x + (y - x) * k
        }
        FieldNode::Remap { in_lo, in_hi, out_lo, out_hi, clamp } => {
            let x = arg!(0, p, in_lo);
            let d = in_hi - in_lo;
            let mut t = (x - in_lo) / if d.abs() < 1e-5 { 1e-5 } else { d };
            if clamp {
                t = t.clamp(0.0, 1.0);
            }
            out_lo + (out_hi - out_lo) * t
        }
        FieldNode::Warp { mode, amount } => {
            let amt_i = if mode == WarpMode::Curl { 2 } else { 3 };
            let amt = arg!(amt_i, p, amount);
            let q = match mode {
                WarpMode::Offset => {
                    [p[0] + arg!(1, p, 0.0) * amt, p[1] + arg!(2, p, 0.0) * amt]
                }
                WarpMode::Curl => {
                    let e = 1.0 / 512.0;
                    let mut pot = |q: [f32; 2]| input(1, q, sd).unwrap_or(0.0);
                    let px1 = pot([p[0] + e, p[1]]);
                    let px0 = pot([p[0] - e, p[1]]);
                    let py1 = pot([p[0], p[1] + e]);
                    let py0 = pot([p[0], p[1] - e]);
                    [
                        p[0] + (py1 - py0) / (2.0 * e) * amt,
                        p[1] - (px1 - px0) / (2.0 * e) * amt,
                    ]
                }
            };
            input(0, q, sd).unwrap_or(0.0)
        }
        FieldNode::Seed { shift } => {
            let sh = arg!(1, p, shift);
            input(0, p, sd + sh).unwrap_or(0.0)
        }
        FieldNode::MaskIn => mask,
        FieldNode::Spline { ref points, mode, feather, offset } => {
            let sp = crate::graph::SplineParams {
                points: points.clone(),
                mode,
                ..Default::default()
            };
            let poly = crate::codegen::spline_polyline(&sp, [1.0, 1.0]);
            let n = poly.len();
            if n < 3 {
                return 0.0;
            }
            let q = [p[0] - arg!(0, p, offset[0]), p[1] - arg!(1, p, offset[1])];
            let fs = arg!(2, p, feather);
            let (mut d, mut fe, mut sgn) = (1e20f32, 0.0f32, 1.0f32);
            let mut j = n - 1;
            for i in 0..n {
                let (a, b) = (poly[j].0, poly[i].0);
                let e = [a[0] - b[0], a[1] - b[1]];
                let w = [q[0] - b[0], q[1] - b[1]];
                let ee = (e[0] * e[0] + e[1] * e[1]).max(1e-12);
                let t = ((w[0] * e[0] + w[1] * e[1]) / ee).clamp(0.0, 1.0);
                let bq = [w[0] - e[0] * t, w[1] - e[1] * t];
                let dd = bq[0] * bq[0] + bq[1] * bq[1];
                if dd < d {
                    d = dd;
                    fe = poly[i].1 + (poly[j].1 - poly[i].1) * t;
                }
                let c = [q[1] >= b[1], q[1] < a[1], e[0] * w[1] > e[1] * w[0]];
                if c.iter().all(|x| *x) || !c.iter().any(|x| *x) {
                    sgn = -sgn;
                }
                j = i;
            }
            let dist = d.sqrt() * sgn;
            let h = (fe * fs * 0.5).max(1e-6);
            let t = ((dist + h) / (2.0 * h)).clamp(0.0, 1.0);
            1.0 - t * t * (3.0 - 2.0 * t)
        }
        FieldNode::Shape { kind, center, size, feather } => {
            let c = [arg!(0, p, center[0]), arg!(1, p, center[1])];
            let sx = arg!(2, p, size[0]).abs().max(1e-5);
            let sy = arg!(3, p, size[1]).abs().max(1e-5);
            let f = arg!(4, p, feather);
            let d = match kind {
                ShapeKind::Box => {
                    let qx = (p[0] - c[0]).abs() - sx;
                    let qy = (p[1] - c[1]).abs() - sy;
                    (qx.max(0.0).powi(2) + qy.max(0.0).powi(2)).sqrt()
                        + qx.max(qy).min(0.0)
                }
                ShapeKind::Ellipse => {
                    let q = [(p[0] - c[0]) / sx, (p[1] - c[1]) / sy];
                    ((q[0] * q[0] + q[1] * q[1]).sqrt() - 1.0) * sx.min(sy)
                }
            };
            let h = (f * 0.5).max(1e-5);
            let t = ((d + h) / (2.0 * h)).clamp(0.0, 1.0);
            1.0 - t * t * (3.0 - 2.0 * t)
        }
        FieldNode::Output => input(0, p, sd).unwrap_or(0.0),
        FieldNode::Gradient { axis, step } => {
            let e = step.max(1e-4);
            let d = match axis {
                Axis::X => [e, 0.0],
                Axis::Y => [0.0, e],
            };
            match (input(0, [p[0] + d[0], p[1] + d[1]], sd), input(0, [p[0] - d[0], p[1] - d[1]], sd)) {
                (Some(a), Some(b)) => (a - b) / (2.0 * e),
                _ => 0.0,
            }
        }
    }
}

#[cfg(all(test, feature = "serde"))]
mod wire_tests {
    use super::*;

    /// A wired field graph has to survive a setup file: JSON cannot key an
    /// object by a tuple, so the edges go as pairs — and the empty object
    /// an older setup holds still reads.
    #[test]
    fn wired_graphs_round_trip_through_json() {
        let mut g = FieldGraph::default();
        g.nodes.insert(1, FieldNode::default());
        g.nodes.insert(2, FieldNode::Output);
        g.edges.insert((2, 0), 1);
        let j = serde_json::to_string(&g).unwrap();
        assert!(j.contains("[[2,0],1]"), "{j}");
        let back: FieldGraph = serde_json::from_str(&j).unwrap();
        assert_eq!(back, g);
        let legacy: FieldGraph =
            serde_json::from_str(r#"{"nodes":{},"edges":{},"out":null}"#).unwrap();
        assert!(legacy.edges.is_empty());

        let mut geo = crate::geo::GeoGraph::default();
        geo.nodes.insert(1, crate::geo::GeoNode::default());
        geo.nodes.insert(2, crate::geo::GeoNode::Output);
        geo.edges.insert((2, 0), 1);
        // A Noise node inside a drawing saves its own field graph.
        geo.nodes.insert(3, crate::geo::GeoNode::Noise { graph: g.clone(), scale: 1.0, seed: 0.0, speed: 0.0 });
        geo.edges.insert((1, 2), 3);
        let j = serde_json::to_string(&geo).unwrap();
        let back: crate::geo::GeoGraph = serde_json::from_str(&j).unwrap();
        assert_eq!(back, geo);
    }
}

#[cfg(test)]
mod uniform_scale_tests {
    use super::*;

    /// The fBm's uniform scale multiplies both axes: uniform 2 over scale
    /// (a, b) is the same field as scale (2a, 2b) — on the CPU, and driven
    /// through socket 7 like every other number.
    #[test]
    fn uniform_scale_multiplies_both_axes() {
        let one = |n: FieldNode| {
            let mut g = FieldGraph::default();
            g.nodes.insert(1, n);
            g.out = Some(1);
            g
        };
        let base = FieldNode::default();
        let FieldNode::Fbm { scale, .. } = base.clone() else { unreachable!() };
        let mut u = base.clone();
        assert!(u.set_literal(7, 2.0));
        assert_eq!(u.literal(7), Some(2.0));
        let mut s2 = base.clone();
        s2.set_literal(0, scale[0] * 2.0);
        s2.set_literal(1, scale[1] * 2.0);
        for p in [[0.13, 0.41], [0.9, 0.2], [0.5, 0.77]] {
            let a = eval_cpu(&one(u.clone()), p, 0.0, 0.0, 1.0);
            let b = eval_cpu(&one(s2.clone()), p, 0.0, 0.0, 1.0);
            assert!((a - b).abs() < 1e-5, "{a} vs {b} at {p:?}");
        }
        assert_eq!(base.inputs()[7].name, "scale");
        assert_eq!(base.inputs().len(), 8, "older sockets keep their index");
    }

    #[cfg(feature = "serde")]
    #[test]
    fn an_old_fbm_reads_uniform_one() {
        let mut v = serde_json::to_value(FieldNode::default()).unwrap();
        v["Fbm"].as_object_mut().unwrap().remove("uniform");
        let n: FieldNode = serde_json::from_value(v).unwrap();
        assert_eq!(n.literal(7), Some(1.0));
    }
}
