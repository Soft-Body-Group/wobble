//! SIGNED DISTANCE FIELDS — a sub-graph that draws with distance.
//!
//! An SDF node owns a graph of [`SdfNode`]s. Its shapes (a spline drawn in
//! the viewer, a circle, a rounded rect) are DISTANCES: at every pixel, how
//! far that pixel is from the outline, negative inside. Distance ops reshape
//! them (smooth unions, grow/shrink, shells, noise wobble), and effects turn
//! a distance into pixels — fill, stroke, glow, drop shadow, ripples, bevel,
//! refraction. That is the reason to have the type at all: one spline gives
//! a stroke, its glow and a ripple that travels out from it, because each
//! effect reads the same number and asks a different question of it.
//!
//! It is the noise field's paradigm with a second pipe. A distance is a
//! scalar function of position, so it rides the FIELD pipe and any socket
//! that takes a number takes one (a Slider, or another distance); effects
//! emit COLOUR (premultiplied rgba) on the image pipe, and Merge/Output
//! composite those. Consequences worth knowing before editing this file:
//!
//!  - Everything is fused. Distance nodes compile to WGSL functions
//!    (`sd{owner}_{node}(p) -> f32`) so an effect can read a shape at
//!    another point (a shadow's offset, a transformed domain, a gradient);
//!    the effect side compiles to ONE entry function in SSA form, where
//!    every node read at the pixel itself is evaluated exactly once — a
//!    spline feeding a fill, a stroke and a glow walks its outline once.
//!  - Params are uniforms, in one block reserved by the owning node, so a
//!    drag never recompiles. Topology is the wiring, the kinds, and each
//!    spline's point COUNT (it sizes the slot span, as on the comp Spline).
//!  - Units: positions and lengths are PIXELS of the node's format, packed
//!    raw and multiplied by the proxy scale in the shader, so a Slider wired
//!    into a width scales under proxy exactly like the literal it replaces.
//!    Spline points are the one exception: they live in FIELD space (x
//!    0..1 across the width, y in width units) so the viewer's roto gizmo
//!    edits them unchanged, and are converted to pixels at pack time.
//!  - Refraction samples the node's IMAGE input somewhere else. Codegen
//!    cannot call the input from inside a function, so every Refract adds
//!    its displacement into one function (`sdd{owner}`) that the comp emit
//!    evaluates first; the input is then fused a second time at the
//!    displaced position and handed to the entry as `bgr`.

use crate::graph::{NodeId, SplineMode, SplinePoint};
use std::collections::BTreeMap;
use std::fmt::Write as _;

/// An SDF graph: nodes, wiring, and what it evaluates to. DATA, like a
/// noise field's graph: it serializes into the setup inside its node.
#[derive(Clone, Debug, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct SdfGraph {
    pub nodes: BTreeMap<NodeId, SdfNode>,
    /// (consumer, socket) -> producer.
    #[cfg_attr(feature = "serde", serde(with = "crate::field::edge_map"))]
    pub edges: BTreeMap<(NodeId, usize), NodeId>,
    /// The node the graph evaluates to when there is no Output node.
    pub out: Option<NodeId>,
    /// Canvas positions — the editor's business, but authored state.
    pub layout: BTreeMap<NodeId, [f32; 2]>,
    /// Highest id ever handed out, so a re-opened graph does not reuse one.
    pub next_id: NodeId,
}

impl SdfGraph {
    /// Every Slider and Input, by name, in id order — what the owning node
    /// shows on its panel and offers as channels. Same contract as
    /// [`crate::field::FieldGraph::exposed`].
    pub fn exposed(&self) -> Vec<(String, bool)> {
        let mut out: Vec<(String, bool)> = Vec::new();
        for n in self.nodes.values() {
            let (name, is_input) = match n {
                SdfNode::Const { name, .. } => (name, false),
                SdfNode::Input { name, .. } => (name, true),
                _ => continue,
            };
            let name = exposed_name(name).to_string();
            if !out.iter().any(|(n, _)| *n == name) {
                out.push((name, is_input));
            }
        }
        out
    }
    pub fn exposed_value(&self, name: &str) -> Option<f32> {
        self.nodes.values().find_map(|n| match n {
            SdfNode::Const { value, name: nm } | SdfNode::Input { value, name: nm }
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
            if let SdfNode::Const { value, name: nm } | SdfNode::Input { value, name: nm } = n {
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
    /// What the graph evaluates to: the Output node, else `out`. Unlike a
    /// noise field there is no "last unconsumed node" fallback — an SDF node
    /// with nothing marked passes its input through, which is the honest
    /// reading of a graph that has not been finished.
    pub fn output(&self) -> Option<NodeId> {
        self.nodes
            .iter()
            .find(|(_, n)| matches!(n, SdfNode::Output))
            .map(|(id, _)| *id)
            .or(self.out.filter(|id| self.nodes.contains_key(id)))
    }
}

fn exposed_name(n: &str) -> &str {
    if n.is_empty() { "slider" } else { n }
}

/// What travels on a socket inside an SDF graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pipe {
    /// A number per pixel: a distance, or a Slider's value. The FIELD pipe.
    Dist,
    /// Premultiplied rgba. The IMAGE pipe.
    Color,
}

/// How a socket's number is read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unit {
    /// Pixels of the node's format: scaled by the proxy in the shader.
    Px,
    /// Degrees.
    Deg,
    /// A plain ratio.
    Plain,
}

/// One socket: its label, the uniform lane holding the literal it overrides
/// (None for a socket that carries a shape or a colour), its pipe and unit.
pub struct SdfInput {
    pub name: &'static str,
    pub lane: Option<usize>,
    pub pipe: Pipe,
    pub unit: Unit,
}

const fn lit(name: &'static str, lane: usize, unit: Unit) -> SdfInput {
    SdfInput { name, lane: Some(lane), pipe: Pipe::Dist, unit }
}
const fn shape(name: &'static str) -> SdfInput {
    SdfInput { name, lane: None, pipe: Pipe::Dist, unit: Unit::Px }
}
const fn color(name: &'static str) -> SdfInput {
    SdfInput { name, lane: None, pipe: Pipe::Color, unit: Unit::Plain }
}

/// How Combine joins two shapes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum CombineOp {
    #[default]
    Union,
    /// A minus B.
    Subtract,
    Intersect,
    /// Either but not both.
    Xor,
}

impl CombineOp {
    pub const ALL: [CombineOp; 4] =
        [CombineOp::Union, CombineOp::Subtract, CombineOp::Intersect, CombineOp::Xor];
    pub fn name(self) -> &'static str {
        match self {
            CombineOp::Union => "union",
            CombineOp::Subtract => "subtract",
            CombineOp::Intersect => "intersect",
            CombineOp::Xor => "xor",
        }
    }
    fn code(self) -> f32 {
        self as u32 as f32
    }
}

/// How Merge lays one colour on another.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum MergeOp {
    #[default]
    Over,
    Add,
    Multiply,
    Screen,
}

impl MergeOp {
    pub const ALL: [MergeOp; 4] = [MergeOp::Over, MergeOp::Add, MergeOp::Multiply, MergeOp::Screen];
    pub fn name(self) -> &'static str {
        match self {
            MergeOp::Over => "over",
            MergeOp::Add => "add",
            MergeOp::Multiply => "multiply",
            MergeOp::Screen => "screen",
        }
    }
    fn code(self) -> f32 {
        self as u32 as f32
    }
}

/// Which side of the outline Ripples run on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Side {
    #[default]
    Outside,
    Inside,
    Both,
}

impl Side {
    pub const ALL: [Side; 3] = [Side::Outside, Side::Inside, Side::Both];
    pub fn name(self) -> &'static str {
        match self {
            Side::Outside => "outside",
            Side::Inside => "inside",
            Side::Both => "both",
        }
    }
    fn code(self) -> f32 {
        match self {
            Side::Outside => 1.0,
            Side::Inside => -1.0,
            Side::Both => 0.0,
        }
    }
}

/// One node of an SDF graph. Lengths and positions are PIXELS of the
/// node's format (see the module docs); colours are straight rgba, the
/// alpha an opacity.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum SdfNode {
    // --- shapes ---------------------------------------------------------
    /// A spline drawn in the viewer. Closed (3+ points): a signed distance,
    /// negative inside. Open: the distance to the curve itself, which is
    /// what a stroke or a glow along a path wants.
    Spline {
        points: Vec<SplinePoint>,
        mode: SplineMode,
        closed: bool,
        offset: [f32; 2],
    },
    Circle { center: [f32; 2], radius: f32 },
    /// `size` is the full width and height; `round` rounds the corners.
    Rect { center: [f32; 2], size: [f32; 2], round: f32 },
    // --- distance ops ---------------------------------------------------
    /// A and B joined; `smooth` > 0 melts the seam over that many pixels.
    Combine { op: CombineOp, smooth: f32 },
    /// Grow (positive) or shrink (negative) the shape.
    Offset { amount: f32 },
    /// The shape's outline as a band `width` wide: a ring from a disc.
    Shell { width: f32 },
    /// A to B.
    Morph { t: f32 },
    /// Move, turn and scale the shape about `pivot`.
    Transform { translate: [f32; 2], rotate: f32, scale: f32, pivot: [f32; 2] },
    /// Wobble the outline with fBm: `amount` pixels, features `size` wide.
    /// `speed` evolves it per frame.
    Displace { amount: f32, size: f32, seed: f32, speed: f32, octaves: u32 },
    // --- numbers --------------------------------------------------------
    /// A named, keyable number that wires into any socket; shown on the
    /// owning SDF node's panel.
    Const {
        value: f32,
        #[cfg_attr(feature = "serde", serde(default))]
        name: String,
    },
    /// A data jack on the owning node, driven from the comp's link mode.
    Input {
        name: String,
        #[cfg_attr(feature = "serde", serde(default))]
        value: f32,
    },
    // --- effects: distance -> colour -------------------------------------
    Fill { color: [f32; 4], soft: f32 },
    Stroke { color: [f32; 4], width: f32, soft: f32 },
    /// Light, not paint: premultiplied colour with zero alpha, so it ADDS
    /// to whatever it lands on.
    Glow { color: [f32; 4], radius: f32, intensity: f32 },
    Shadow { color: [f32; 4], offset: [f32; 2], soft: f32 },
    /// Iso-lines of the distance, travelling at `speed` pixels a frame.
    Ripples { color: [f32; 4], spacing: f32, width: f32, speed: f32, falloff: f32, side: Side },
    /// The shape lit as a raised surface: a rim `width` wide, lit from
    /// `angle` at `elevation`.
    Bevel { color: [f32; 4], width: f32, angle: f32, elevation: f32, depth: f32 },
    /// The node's input seen through the shape, bent at its rim.
    Refract { amount: f32, width: f32 },
    /// The distance itself as colour: `inner` deep inside, `outer`
    /// `range` pixels out.
    Ramp { inner: [f32; 4], outer: [f32; 4], range: f32 },
    // --- colour ---------------------------------------------------------
    Merge { op: MergeOp, mix: f32 },
    /// The node's image input, at this pixel.
    ImageIn,
    /// What the node draws, laid OVER its input.
    Output,
}

impl Default for SdfNode {
    fn default() -> Self {
        SdfNode::Circle { center: [960.0, 540.0], radius: 200.0 }
    }
}

const WHITE: [f32; 4] = [1.0, 1.0, 1.0, 1.0];

impl SdfNode {
    /// Every kind, freshly made — what the add menu and the tests walk.
    pub fn catalog() -> Vec<SdfNode> {
        vec![
            SdfNode::Spline {
                points: Vec::new(),
                mode: SplineMode::CatmullRom,
                closed: true,
                offset: [0.0, 0.0],
            },
            SdfNode::Circle { center: [960.0, 540.0], radius: 200.0 },
            SdfNode::Rect { center: [960.0, 540.0], size: [600.0, 300.0], round: 0.0 },
            SdfNode::Combine { op: CombineOp::Union, smooth: 0.0 },
            SdfNode::Offset { amount: 10.0 },
            SdfNode::Shell { width: 8.0 },
            SdfNode::Morph { t: 0.5 },
            SdfNode::Transform {
                translate: [0.0, 0.0],
                rotate: 0.0,
                scale: 1.0,
                pivot: [960.0, 540.0],
            },
            SdfNode::Displace { amount: 12.0, size: 80.0, seed: 0.0, speed: 0.0, octaves: 3 },
            SdfNode::Const { value: 0.5, name: String::new() },
            SdfNode::Input { name: "in".into(), value: 0.0 },
            SdfNode::Fill { color: WHITE, soft: 0.0 },
            SdfNode::Stroke { color: WHITE, width: 4.0, soft: 0.0 },
            SdfNode::Glow { color: [0.3, 0.7, 1.0, 1.0], radius: 40.0, intensity: 1.0 },
            SdfNode::Shadow { color: [0.0, 0.0, 0.0, 0.6], offset: [12.0, 12.0], soft: 24.0 },
            SdfNode::Ripples {
                color: WHITE,
                spacing: 24.0,
                width: 2.0,
                speed: 1.0,
                falloff: 120.0,
                side: Side::Outside,
            },
            SdfNode::Bevel { color: WHITE, width: 24.0, angle: 135.0, elevation: 45.0, depth: 1.0 },
            SdfNode::Refract { amount: 20.0, width: 40.0 },
            SdfNode::Ramp { inner: WHITE, outer: [0.0, 0.0, 0.0, 1.0], range: 100.0 },
            SdfNode::Merge { op: MergeOp::Over, mix: 1.0 },
            SdfNode::ImageIn,
            SdfNode::Output,
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            SdfNode::Spline { .. } => "Spline",
            SdfNode::Circle { .. } => "Circle",
            SdfNode::Rect { .. } => "Rect",
            SdfNode::Combine { .. } => "Combine",
            SdfNode::Offset { .. } => "Offset",
            SdfNode::Shell { .. } => "Shell",
            SdfNode::Morph { .. } => "Morph",
            SdfNode::Transform { .. } => "Transform",
            SdfNode::Displace { .. } => "Displace",
            SdfNode::Const { .. } => "Slider",
            SdfNode::Input { .. } => "Input",
            SdfNode::Fill { .. } => "Fill",
            SdfNode::Stroke { .. } => "Stroke",
            SdfNode::Glow { .. } => "Glow",
            SdfNode::Shadow { .. } => "Shadow",
            SdfNode::Ripples { .. } => "Ripples",
            SdfNode::Bevel { .. } => "Bevel",
            SdfNode::Refract { .. } => "Refract",
            SdfNode::Ramp { .. } => "Ramp",
            SdfNode::Merge { .. } => "Merge",
            SdfNode::ImageIn => "Image",
            SdfNode::Output => "Output",
        }
    }

    /// The pipe the node's ONE output carries.
    pub fn out_pipe(&self) -> Pipe {
        match self {
            SdfNode::Fill { .. }
            | SdfNode::Stroke { .. }
            | SdfNode::Glow { .. }
            | SdfNode::Shadow { .. }
            | SdfNode::Ripples { .. }
            | SdfNode::Bevel { .. }
            | SdfNode::Refract { .. }
            | SdfNode::Ramp { .. }
            | SdfNode::Merge { .. }
            | SdfNode::ImageIn
            | SdfNode::Output => Pipe::Color,
            _ => Pipe::Dist,
        }
    }

    /// A Slider or an Input: a number in AUTHORED units, where a distance
    /// is already in render pixels. The one place that decides whether a
    /// driven px socket still needs the proxy scale.
    fn is_scalar(&self) -> bool {
        matches!(self, SdfNode::Const { .. } | SdfNode::Input { .. })
    }

    /// Sockets, in order. A shape or colour socket carries no literal.
    pub fn inputs(&self) -> Vec<SdfInput> {
        use Unit::*;
        match self {
            SdfNode::Spline { .. } => vec![lit("offset x", 2, Px), lit("offset y", 3, Px)],
            SdfNode::Circle { .. } => {
                vec![lit("centre x", 0, Px), lit("centre y", 1, Px), lit("radius", 2, Px)]
            }
            SdfNode::Rect { .. } => vec![
                lit("centre x", 0, Px),
                lit("centre y", 1, Px),
                lit("width", 2, Px),
                lit("height", 3, Px),
                lit("round", 4, Px),
            ],
            SdfNode::Combine { .. } => vec![shape("a"), shape("b"), lit("smooth", 0, Px)],
            SdfNode::Offset { .. } => vec![shape("shape"), lit("amount", 0, Px)],
            SdfNode::Shell { .. } => vec![shape("shape"), lit("width", 0, Px)],
            SdfNode::Morph { .. } => vec![shape("a"), shape("b"), lit("t", 0, Plain)],
            SdfNode::Transform { .. } => vec![
                shape("shape"),
                lit("move x", 0, Px),
                lit("move y", 1, Px),
                lit("rotate", 2, Deg),
                lit("scale", 3, Plain),
            ],
            SdfNode::Displace { .. } => {
                vec![shape("shape"), lit("amount", 0, Px), lit("size", 1, Px)]
            }
            SdfNode::Const { .. } | SdfNode::Input { .. } | SdfNode::ImageIn => Vec::new(),
            SdfNode::Fill { .. } => vec![shape("shape"), lit("softness", 0, Px)],
            SdfNode::Stroke { .. } => {
                vec![shape("shape"), lit("width", 0, Px), lit("softness", 1, Px)]
            }
            SdfNode::Glow { .. } => {
                vec![shape("shape"), lit("radius", 0, Px), lit("intensity", 1, Plain)]
            }
            SdfNode::Shadow { .. } => vec![
                shape("shape"),
                lit("offset x", 0, Px),
                lit("offset y", 1, Px),
                lit("softness", 2, Px),
            ],
            SdfNode::Ripples { .. } => vec![
                shape("shape"),
                lit("spacing", 0, Px),
                lit("width", 1, Px),
                lit("speed", 2, Px),
                lit("falloff", 3, Px),
            ],
            SdfNode::Bevel { .. } => vec![
                shape("shape"),
                lit("width", 0, Px),
                lit("angle", 1, Deg),
                lit("elevation", 2, Deg),
                lit("depth", 3, Plain),
            ],
            SdfNode::Refract { .. } => {
                vec![shape("shape"), lit("amount", 0, Px), lit("width", 1, Px)]
            }
            SdfNode::Ramp { .. } => vec![shape("shape"), lit("range", 0, Px)],
            SdfNode::Merge { .. } => vec![color("fg"), color("bg"), lit("mix", 0, Plain)],
            SdfNode::Output => vec![color("colour")],
        }
    }

    /// The literals, by SOCKET index — the ONE list the getter, the setter
    /// and the packer all read, so they cannot disagree about which field a
    /// socket stands for. A Slider's value is its socket-less channel 0.
    fn lits_mut(&mut self) -> Vec<Option<&mut f32>> {
        match self {
            SdfNode::Spline { offset: [x, y], .. } => vec![Some(x), Some(y)],
            SdfNode::Circle { center: [x, y], radius } => vec![Some(x), Some(y), Some(radius)],
            SdfNode::Rect { center: [x, y], size: [w, h], round } => {
                vec![Some(x), Some(y), Some(w), Some(h), Some(round)]
            }
            SdfNode::Combine { smooth, .. } => vec![None, None, Some(smooth)],
            SdfNode::Offset { amount } => vec![None, Some(amount)],
            SdfNode::Shell { width } => vec![None, Some(width)],
            SdfNode::Morph { t } => vec![None, None, Some(t)],
            SdfNode::Transform { translate: [x, y], rotate, scale, .. } => {
                vec![None, Some(x), Some(y), Some(rotate), Some(scale)]
            }
            SdfNode::Displace { amount, size, .. } => vec![None, Some(amount), Some(size)],
            SdfNode::Const { value, .. } | SdfNode::Input { value, .. } => vec![Some(value)],
            SdfNode::Fill { soft, .. } => vec![None, Some(soft)],
            SdfNode::Stroke { width, soft, .. } => vec![None, Some(width), Some(soft)],
            SdfNode::Glow { radius, intensity, .. } => vec![None, Some(radius), Some(intensity)],
            SdfNode::Shadow { offset: [x, y], soft, .. } => vec![None, Some(x), Some(y), Some(soft)],
            SdfNode::Ripples { spacing, width, speed, falloff, .. } => {
                vec![None, Some(spacing), Some(width), Some(speed), Some(falloff)]
            }
            SdfNode::Bevel { width, angle, elevation, depth, .. } => {
                vec![None, Some(width), Some(angle), Some(elevation), Some(depth)]
            }
            SdfNode::Refract { amount, width } => vec![None, Some(amount), Some(width)],
            SdfNode::Ramp { range, .. } => vec![None, Some(range)],
            SdfNode::Merge { mix, .. } => vec![None, None, Some(mix)],
            SdfNode::ImageIn | SdfNode::Output => Vec::new(),
        }
    }

    /// The literal behind socket `i` — what animation and the patch bay
    /// drive while the graph is open.
    pub fn literal(&self, i: usize) -> Option<f32> {
        let mut c = self.clone();
        c.lits_mut().into_iter().nth(i).flatten().map(|v| *v)
    }

    pub fn set_literal(&mut self, i: usize, v: f32) -> bool {
        match self.lits_mut().into_iter().nth(i).flatten() {
            Some(slot) => {
                *slot = v;
                true
            }
            None => false,
        }
    }

    /// Animatable channels: one per socket, or the Slider's own value.
    pub fn channel_count(&self) -> usize {
        match self {
            SdfNode::Const { .. } | SdfNode::Input { .. } => 1,
            _ => self.inputs().len(),
        }
    }

    /// vec4s this node needs in the owner's block.
    pub fn slots(&self) -> usize {
        match self {
            SdfNode::Spline { points, .. } => 1 + max_samples(points.len()).div_ceil(2),
            SdfNode::Rect { .. }
            | SdfNode::Transform { .. }
            | SdfNode::Displace { .. }
            | SdfNode::Fill { .. }
            | SdfNode::Stroke { .. }
            | SdfNode::Glow { .. }
            | SdfNode::Shadow { .. }
            | SdfNode::Bevel { .. } => 2,
            SdfNode::Ripples { .. } | SdfNode::Ramp { .. } => 3,
            SdfNode::ImageIn | SdfNode::Output => 0,
            _ => 1,
        }
    }

    /// Everything the emitted code branches on. Modes and ops are uniforms
    /// (a switch in the shader), so flipping one is a param edit; only the
    /// kind and a spline's point count are structure.
    pub fn topo_tag(&self) -> u64 {
        let k = match self {
            SdfNode::Spline { .. } => 1,
            SdfNode::Circle { .. } => 2,
            SdfNode::Rect { .. } => 3,
            SdfNode::Combine { .. } => 4,
            SdfNode::Offset { .. } => 5,
            SdfNode::Shell { .. } => 6,
            SdfNode::Morph { .. } => 7,
            SdfNode::Transform { .. } => 8,
            SdfNode::Displace { .. } => 9,
            SdfNode::Const { .. } => 10,
            SdfNode::Input { .. } => 11,
            SdfNode::Fill { .. } => 12,
            SdfNode::Stroke { .. } => 13,
            SdfNode::Glow { .. } => 14,
            SdfNode::Shadow { .. } => 15,
            SdfNode::Ripples { .. } => 16,
            SdfNode::Bevel { .. } => 17,
            SdfNode::Refract { .. } => 18,
            SdfNode::Ramp { .. } => 19,
            SdfNode::Merge { .. } => 20,
            SdfNode::ImageIn => 21,
            SdfNode::Output => 22,
        };
        match self {
            SdfNode::Spline { points, .. } => k + ((max_samples(points.len()) as u64) << 8),
            _ => k,
        }
    }

    /// The node's numbers into `v` at float index `base`. `fmt_w` is the
    /// node's render width (proxy included): spline points go from field
    /// space to render pixels here. Px literals go in RAW — the shader
    /// scales them, so a driven socket and its literal agree.
    fn pack(&self, base: usize, fmt_w: f32, v: &mut [f32]) {
        let mut put = |i: usize, x: f32| {
            if let Some(s) = v.get_mut(base + i) {
                *s = x;
            }
        };
        let ins = self.inputs();
        let mut me = self.clone();
        for (inp, val) in ins.iter().zip(me.lits_mut()) {
            if let (Some(lane), Some(val)) = (inp.lane, val) {
                put(lane, *val);
            }
        }
        match self {
            SdfNode::Spline { points, mode, closed, .. } => {
                let closed = *closed && points.len() >= 3;
                let pl = crate::geo::spline_outline(points, *mode, closed);
                let n = pl.len().min(max_samples(points.len()));
                put(0, n as f32);
                put(1, closed as u32 as f32);
                for (i, (p, _)) in pl.iter().take(n).enumerate() {
                    put(4 + i * 2, p[0] * fmt_w);
                    put(4 + i * 2 + 1, p[1] * fmt_w);
                }
            }
            SdfNode::Combine { op, .. } => put(1, op.code()),
            SdfNode::Transform { pivot, .. } => {
                put(4, pivot[0]);
                put(5, pivot[1]);
            }
            SdfNode::Displace { seed, speed, octaves, .. } => {
                put(2, *seed);
                put(3, *speed);
                put(4, (*octaves).clamp(1, crate::field::FBM_MAX_OCTAVES) as f32);
            }
            SdfNode::Const { value, .. } | SdfNode::Input { value, .. } => put(0, *value),
            SdfNode::Fill { color, .. }
            | SdfNode::Stroke { color, .. }
            | SdfNode::Glow { color, .. }
            | SdfNode::Shadow { color, .. }
            | SdfNode::Bevel { color, .. } => {
                for (k, c) in color.iter().enumerate() {
                    put(4 + k, *c);
                }
            }
            SdfNode::Ripples { color, side, .. } => {
                for (k, c) in color.iter().enumerate() {
                    put(4 + k, *c);
                }
                put(8, side.code());
            }
            SdfNode::Ramp { inner, outer, .. } => {
                for k in 0..4 {
                    put(4 + k, inner[k]);
                    put(8 + k, outer[k]);
                }
            }
            SdfNode::Merge { op, .. } => put(1, op.code()),
            _ => {}
        }
    }
}

/// Outline samples a spline of `n` points can emit, open or closed — the
/// span the layout reserves. Both counts are functions of `n` alone, so
/// opening or closing a spline is a param edit, not a recompile.
pub fn max_samples(n: usize) -> usize {
    if n < 2 {
        return 0;
    }
    let closed = crate::codegen::spline_samples(n);
    let open = (n - 1) * crate::codegen::spline_subd(n + 2) + 1;
    closed.max(open)
}

/// Where each node's block starts, in vec4s from the graph's base. Sorted
/// by id so compiler and packer agree without passing a layout.
pub fn slot_map(g: &SdfGraph) -> (BTreeMap<NodeId, usize>, usize) {
    let mut map = BTreeMap::new();
    let mut at = 0usize;
    for (id, n) in &g.nodes {
        map.insert(*id, at);
        at += n.slots();
    }
    (map, at)
}

/// Kinds, wiring, and the node ids the layout is keyed on.
pub fn topo_tag(g: &SdfGraph) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    let mut mix = |x: u64| {
        h ^= x;
        h = h.wrapping_mul(0x100000001b3);
    };
    for (id, n) in &g.nodes {
        mix(*id);
        mix(n.topo_tag());
    }
    for ((dst, port), src) in &g.edges {
        mix(*dst ^ ((*port as u64) << 32));
        mix(*src);
    }
    mix(g.out.unwrap_or(u64::MAX));
    h
}

/// An SDF node: the graph, the format it draws into when nothing feeds it,
/// and the transient state the app derives per frame.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct SdfParams {
    /// The format when the image input is unwired (a generator then).
    pub size: (u32, u32),
    #[cfg_attr(feature = "serde", serde(default))]
    pub inherit: bool,
    pub graph: SdfGraph,
    /// The tick being rendered. TRANSIENT, stamped per frame like a
    /// Noise's: what Ripples travel and Displace evolves with.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub time: f32,
    /// While the graph is open, the node being LOOKED at: the viewer shows
    /// that node rather than the Output — a colour over the input, a
    /// distance as a false-colour map. TRANSIENT, set only by the preview.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub preview: Option<NodeId>,
}

impl Default for SdfParams {
    fn default() -> Self {
        Self {
            size: (1920, 1080),
            inherit: false,
            graph: SdfGraph::default(),
            time: 0.0,
            preview: None,
        }
    }
}

impl SdfParams {
    pub const INPUTS: usize = 2;
    pub const MASK_INPUT: Option<usize> = Some(1);

    /// One header vec4 (render width, proxy, time), then the graph's blocks.
    pub fn slots(&self) -> usize {
        1 + slot_map(&self.graph).1
    }

    /// The node the viewer shows: the preview, else the graph's output.
    fn target(&self) -> Option<NodeId> {
        self.preview.filter(|p| self.graph.nodes.contains_key(p)).or_else(|| self.graph.output())
    }

    pub fn topo_tag(&self) -> u64 {
        let mut h = topo_tag(&self.graph) ^ 0x5DF0_0000;
        // What is being LOOKED at changes the entry point; it only ever
        // moves while the graph is open for editing.
        if let Some(p) = self.preview {
            h = h.wrapping_mul(0x100000001b3) ^ (p.wrapping_add(1) << 1);
        }
        h
    }

    /// The span at float index `base`: the header, then every node.
    pub fn pack(&self, base: usize, fmt_w: f32, proxy: f32, v: &mut [f32]) {
        let hdr = [fmt_w, proxy, self.time, 0.0];
        for (k, x) in hdr.iter().enumerate() {
            if let Some(s) = v.get_mut(base + k) {
                *s = *x;
            }
        }
        let (map, _) = slot_map(&self.graph);
        for (id, off) in &map {
            if let Some(n) = self.graph.nodes.get(id) {
                n.pack(base + 4 + off * 4, fmt_w, v);
            }
        }
    }

    /// Fused code for the node: the input, the graph over it, the mask.
    pub fn emit(&self, e: &mut crate::codegen::Emit) -> String {
        let bg = e.input(0);
        let slot = e.slot();
        let Some(prog) = compile(e.node_id(), &self.graph, self.target(), slot) else {
            // Nothing to draw (no Output, or a cycle): the input, untouched.
            return bg;
        };
        let (_, fns) = e.color_helpers();
        // Helpers once per shader however many SDF nodes it holds; the
        // per-graph functions keyed in emission order (callee first), and
        // skipped when an earlier visit of this node already emitted them.
        let have: std::collections::HashSet<String> =
            fns.keys().filter_map(|k| k.get(5..).map(str::to_string)).collect();
        if !have.contains("sdf_helpers") {
            let key = format!("{:04}_sdf_helpers", fns.len());
            fns.insert(key, SDF_WGSL.to_string());
        }
        for (name, def) in prog.fns {
            if !have.contains(&name) {
                let key = format!("{:04}_{name}", fns.len());
                fns.insert(key, def);
            }
        }
        let pos = e.pos().to_string();
        let bgr = match &prog.disp {
            Some(d) => {
                let dv = e.fresh();
                e.line(format_args!("let {dv} = {d}({pos});"));
                e.input_at(0, &format!("({pos} + {dv})"))
            }
            None => bg.clone(),
        };
        let v = e.fresh();
        e.line(format_args!("let {v} = {}({pos}, {bg}, {bgr});", prog.entry));
        e.masked(Self::MASK_INPUT, &bg, v)
    }
}

/// What [`compile`] hands the comp emit.
pub struct Program {
    /// Function definitions, callee first.
    pub fns: Vec<(String, String)>,
    /// `fn(p, bg, bgr) -> vec4f`: the node's finished pixel.
    pub entry: String,
    /// `fn(p) -> vec2f`: where to sample the input for refraction.
    pub disp: Option<String>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// Inside a `sd{o}_{n}(p)` function: every input is a call.
    Func,
    /// Inside the entry, at the pixel: an input read at `p` is its SSA value.
    Ssa,
}

struct Cx<'a> {
    owner: NodeId,
    g: &'a SdfGraph,
    /// vec4 index of the owner's header; the graph's blocks follow it.
    hdr: usize,
    map: BTreeMap<NodeId, usize>,
    fns: Vec<(String, String)>,
    fn_done: BTreeMap<NodeId, String>,
    fn_stack: Vec<NodeId>,
    ssa: String,
    ssa_done: BTreeMap<NodeId, String>,
    ssa_stack: Vec<NodeId>,
    /// A cycle was met: WGSL has no recursion, so the whole graph is refused.
    cycle: bool,
}

/// Compile an SDF graph for the node at uniform slot `slot`, entering at
/// `target`. None means nothing to draw: no target, or a cycle.
pub fn compile(owner: NodeId, g: &SdfGraph, target: Option<NodeId>, slot: usize) -> Option<Program> {
    let target = target?;
    let (map, _) = slot_map(g);
    let mut cx = Cx {
        owner,
        g,
        hdr: slot,
        map,
        fns: Vec::new(),
        fn_done: BTreeMap::new(),
        fn_stack: Vec::new(),
        ssa: String::new(),
        ssa_done: BTreeMap::new(),
        ssa_stack: Vec::new(),
        cycle: false,
    };
    let v = cx.ssa_val(target);
    if cx.cycle {
        return None;
    }
    let v = v?;
    let fin = match g.nodes.get(&target)?.out_pipe() {
        Pipe::Color => format!("over({v}, bg)"),
        // A distance on its own is not a picture: show it as one.
        Pipe::Dist => format!("sdf_viz({v})"),
    };
    // Refraction: every Refract that feeds the target adds its bend.
    let refracts: Vec<NodeId> = cx
        .ssa_done
        .keys()
        .copied()
        .filter(|id| matches!(g.nodes.get(id), Some(SdfNode::Refract { .. })))
        .collect();
    let disp = if refracts.is_empty() {
        None
    } else {
        let mut body = String::new();
        writeln!(body, "    var acc = vec2f(0.0);").unwrap();
        for r in refracts {
            let d = cx.inp(r, 0, "p", Mode::Func);
            let dx = cx.inp(r, 0, "p + vec2f(1.0, 0.0)", Mode::Func);
            let dy = cx.inp(r, 0, "p + vec2f(0.0, 1.0)", Mode::Func);
            let amt = cx.arg(r, 1, "p", Mode::Func);
            let wid = cx.arg(r, 2, "p", Mode::Func);
            writeln!(body, "    {{").unwrap();
            writeln!(body, "        let d0 = {d};").unwrap();
            writeln!(body, "        let g = vec2f({dx} - d0, {dy} - d0);").unwrap();
            writeln!(body, "        let t = clamp(-d0 / max({wid}, 1e-3), 0.0, 1.0);").unwrap();
            // Strongest at the rim, nothing past `width` in and nothing
            // outside: a lens edge. Sampling back along the normal pulls
            // the interior outward, which reads as magnification.
            writeln!(
                body,
                "        let k = select(0.0, (1.0 - t) * (1.0 - t), d0 < 0.0) * {amt};"
            )
            .unwrap();
            writeln!(body, "        acc -= g / max(length(g), 1e-6) * k;").unwrap();
            writeln!(body, "    }}").unwrap();
        }
        writeln!(body, "    return acc;").unwrap();
        let name = format!("sdd{owner}");
        cx.fns.push((name.clone(), format!("fn {name}(p: vec2f) -> vec2f {{\n{body}}}\n")));
        Some(name)
    };
    if cx.cycle {
        return None;
    }
    let entry = format!("sde{owner}");
    let def = format!(
        "fn {entry}(p: vec2f, bg: vec4f, bgr: vec4f) -> vec4f {{\n{}    return {fin};\n}}\n",
        cx.ssa
    );
    cx.fns.push((entry.clone(), def));
    Some(Program { fns: cx.fns, entry, disp })
}

impl Cx<'_> {
    fn block(&self, node: NodeId) -> usize {
        self.hdr + 1 + self.map.get(&node).copied().unwrap_or(0)
    }

    fn lane(&self, node: NodeId, lane: usize) -> String {
        format!("P.v[{}u].{}", self.block(node) + lane / 4, ['x', 'y', 'z', 'w'][lane % 4])
    }

    fn vec(&self, node: NodeId, k: usize) -> String {
        format!("P.v[{}u]", self.block(node) + k)
    }

    fn px(&self) -> String {
        format!("P.v[{}u].y", self.hdr)
    }

    fn time(&self) -> String {
        format!("P.v[{}u].z", self.hdr)
    }

    /// A shape or colour socket at `q`: the driver's value, or empty when
    /// unwired (a distance of 1e6 is "nowhere near", a colour of 0 is
    /// nothing).
    fn inp(&mut self, node: NodeId, i: usize, q: &str, mode: Mode) -> String {
        let pipe = self.g.nodes[&node].inputs().get(i).map(|s| s.pipe).unwrap_or(Pipe::Dist);
        let src = self
            .g
            .input_of(node, i)
            .filter(|s| self.g.nodes.get(s).is_some_and(|n| n.out_pipe() == pipe));
        match (src, pipe) {
            (None, Pipe::Dist) => "1e6".into(),
            (None, Pipe::Color) => "vec4f(0.0)".into(),
            (Some(s), Pipe::Color) => self.ssa_val(s).unwrap_or_else(|| "vec4f(0.0)".into()),
            (Some(s), Pipe::Dist) => {
                if mode == Mode::Ssa && q == "p" {
                    return self.ssa_val(s).unwrap_or_else(|| "1e6".into());
                }
                match self.fn_name(s) {
                    Some(f) => format!("{f}({q})"),
                    None => "1e6".into(),
                }
            }
        }
    }

    /// A number socket at `q`: the driver or the literal, in render pixels
    /// when the socket is a length.
    fn arg(&mut self, node: NodeId, i: usize, q: &str, mode: Mode) -> String {
        let n = &self.g.nodes[&node];
        let Some(spec) = n.inputs().into_iter().nth(i) else { return "0.0".into() };
        let src = self
            .g
            .input_of(node, i)
            .filter(|s| self.g.nodes.get(s).is_some_and(|n| n.out_pipe() == Pipe::Dist));
        let (expr, authored) = match src {
            Some(s) => {
                let scalar = self.g.nodes[&s].is_scalar();
                let e = if mode == Mode::Ssa && q == "p" {
                    self.ssa_val(s).unwrap_or_else(|| "0.0".into())
                } else {
                    match self.fn_name(s) {
                        Some(f) => format!("{f}({q})"),
                        None => "0.0".into(),
                    }
                };
                (e, scalar)
            }
            None => match spec.lane {
                Some(l) => (self.lane(node, l), true),
                None => ("0.0".into(), false),
            },
        };
        if spec.unit == Unit::Px && authored {
            format!("({expr} * {})", self.px())
        } else {
            expr
        }
    }

    /// The node as a function of position (distance nodes only).
    fn fn_name(&mut self, node: NodeId) -> Option<String> {
        if let Some(n) = self.fn_done.get(&node) {
            return Some(n.clone());
        }
        if self.fn_stack.contains(&node) {
            self.cycle = true;
            return None;
        }
        if self.g.nodes.get(&node)?.out_pipe() != Pipe::Dist {
            return None;
        }
        self.fn_stack.push(node);
        let body = self.gen(node, Mode::Func);
        self.fn_stack.pop();
        let (lines, r) = body?;
        let name = format!("sd{}_{node}", self.owner);
        self.fns.push((name.clone(), format!("fn {name}(p: vec2f) -> f32 {{\n{lines}    return {r};\n}}\n")));
        self.fn_done.insert(node, name.clone());
        Some(name)
    }

    /// The node's value AT THE PIXEL, as a local of the entry: evaluated
    /// once however many consumers read it there.
    fn ssa_val(&mut self, node: NodeId) -> Option<String> {
        if let Some(v) = self.ssa_done.get(&node) {
            return Some(v.clone());
        }
        if self.ssa_stack.contains(&node) {
            self.cycle = true;
            return None;
        }
        self.g.nodes.get(&node)?;
        self.ssa_stack.push(node);
        let body = self.gen(node, Mode::Ssa);
        self.ssa_stack.pop();
        let (lines, r) = body?;
        self.ssa.push_str(&lines);
        let v = format!("v{node}");
        writeln!(self.ssa, "    let {v} = {r};").unwrap();
        self.ssa_done.insert(node, v.clone());
        Some(v)
    }

    /// One node's code: statements, then the expression it evaluates to.
    /// Locals are prefixed with the node id, so the SSA form can inline
    /// every node into one function without collisions.
    fn gen(&mut self, node: NodeId, mode: Mode) -> Option<(String, String)> {
        let n = self.g.nodes.get(&node)?.clone();
        let s = format!("s{node}_");
        let mut l = String::new();
        let px = self.px();
        let time = self.time();
        // Resolved sockets. Closures cannot hold `&mut self`, so each is a
        // call site of its own.
        macro_rules! a {
            ($i:expr) => {
                self.arg(node, $i, "p", mode)
            };
        }
        macro_rules! d {
            ($i:expr) => {
                self.inp(node, $i, "p", mode)
            };
        }
        let r = match &n {
            SdfNode::Spline { .. } => {
                let (ox, oy) = (a!(0), a!(1));
                format!("sdf_spline(p, {}u, vec2f({ox}, {oy}))", self.block(node))
            }
            SdfNode::Circle { .. } => {
                let (x, y, r) = (a!(0), a!(1), a!(2));
                format!("length(p - vec2f({x}, {y})) - {r}")
            }
            SdfNode::Rect { .. } => {
                let (x, y, w, h, rd) = (a!(0), a!(1), a!(2), a!(3), a!(4));
                format!("sdf_rect(p, vec2f({x}, {y}), vec2f({w}, {h}), {rd})")
            }
            SdfNode::Combine { .. } => {
                let (da, db, k) = (d!(0), d!(1), a!(2));
                format!("sdf_combine({da}, {db}, {k}, {})", self.lane(node, 1))
            }
            SdfNode::Offset { .. } => {
                let (da, k) = (d!(0), a!(1));
                format!("{da} - {k}")
            }
            SdfNode::Shell { .. } => {
                let (da, k) = (d!(0), a!(1));
                format!("abs({da}) - {k} * 0.5")
            }
            SdfNode::Morph { .. } => {
                let (da, db, t) = (d!(0), d!(1), a!(2));
                format!("mix({da}, {db}, {t})")
            }
            SdfNode::Transform { .. } => {
                let (tx, ty, rot, sc) = (a!(1), a!(2), a!(3), a!(4));
                let pv = format!("vec2f({}, {}) * {px}", self.lane(node, 4), self.lane(node, 5));
                writeln!(l, "    let {s}r = radians({rot});").unwrap();
                writeln!(l, "    let {s}k = max(abs({sc}), 1e-4);").unwrap();
                writeln!(l, "    let {s}pv = {pv};").unwrap();
                writeln!(l, "    let {s}d = p - {s}pv - vec2f({tx}, {ty});").unwrap();
                writeln!(l, "    let {s}c = cos({s}r);").unwrap();
                writeln!(l, "    let {s}n = sin({s}r);").unwrap();
                writeln!(
                    l,
                    "    let {s}q = vec2f({s}c * {s}d.x + {s}n * {s}d.y, -{s}n * {s}d.x + {s}c * {s}d.y) / {s}k + {s}pv;"
                )
                .unwrap();
                // The source read at the moved point is a CALL, whatever
                // the mode: the SSA value is the source at the pixel.
                let src = self.inp(node, 0, &format!("{s}q"), Mode::Func);
                format!("{src} * {s}k")
            }
            SdfNode::Displace { .. } => {
                let (da, amt, size) = (d!(0), a!(1), a!(2));
                writeln!(
                    l,
                    "    let {s}n = fbm_core(vec3f(p / max({size}, 1e-3), {time} * {}), u32({}), 2.0, 0.5, u32(max({}, 0.0)), 0.0);",
                    self.lane(node, 3),
                    self.lane(node, 4),
                    self.lane(node, 2)
                )
                .unwrap();
                format!("{da} + ({s}n - 0.5) * 2.0 * {amt}")
            }
            SdfNode::Const { .. } | SdfNode::Input { .. } => self.lane(node, 0),
            SdfNode::Fill { .. } => {
                let (da, soft) = (d!(0), a!(1));
                writeln!(l, "    let {s}c = {};", self.vec(node, 1)).unwrap();
                writeln!(l, "    let {s}k = sdf_cov({da}, {soft}) * {s}c.a;").unwrap();
                format!("vec4f({s}c.rgb * {s}k, {s}k)")
            }
            SdfNode::Stroke { .. } => {
                let (da, w, soft) = (d!(0), a!(1), a!(2));
                writeln!(l, "    let {s}c = {};", self.vec(node, 1)).unwrap();
                writeln!(l, "    let {s}k = sdf_cov(abs({da}) - {w} * 0.5, {soft}) * {s}c.a;").unwrap();
                format!("vec4f({s}c.rgb * {s}k, {s}k)")
            }
            SdfNode::Glow { .. } => {
                let (da, rad, gain) = (d!(0), a!(1), a!(2));
                writeln!(l, "    let {s}c = {};", self.vec(node, 1)).unwrap();
                writeln!(
                    l,
                    "    let {s}k = {gain} * {s}c.a * exp(-max({da}, 0.0) / max({rad}, 1e-3));"
                )
                .unwrap();
                format!("vec4f({s}c.rgb * {s}k, 0.0)")
            }
            SdfNode::Shadow { .. } => {
                let (ox, oy, soft) = (a!(1), a!(2), a!(3));
                let d2 = self.inp(node, 0, &format!("(p - vec2f({ox}, {oy}))"), Mode::Func);
                writeln!(l, "    let {s}c = {};", self.vec(node, 1)).unwrap();
                writeln!(l, "    let {s}k = sdf_cov({d2}, {soft}) * {s}c.a;").unwrap();
                format!("vec4f({s}c.rgb * {s}k, {s}k)")
            }
            SdfNode::Ripples { .. } => {
                let (da, sp, w, speed, fo) = (d!(0), a!(1), a!(2), a!(3), a!(4));
                writeln!(l, "    let {s}d = {da};").unwrap();
                writeln!(l, "    let {s}sp = max({sp}, 1e-3);").unwrap();
                writeln!(l, "    let {s}t = {s}d - {time} * {speed};").unwrap();
                writeln!(l, "    let {s}m = {s}t - {s}sp * round({s}t / {s}sp);").unwrap();
                writeln!(l, "    var {s}k = sdf_cov(abs({s}m) - {w} * 0.5, 0.0);").unwrap();
                writeln!(l, "    let {s}fo = {fo};").unwrap();
                writeln!(l, "    if ({s}fo > 0.0) {{ {s}k *= exp(-abs({s}d) / {s}fo); }}").unwrap();
                writeln!(l, "    let {s}side = {};", self.lane(node, 8)).unwrap();
                writeln!(
                    l,
                    "    if (({s}side > 0.5 && {s}d < 0.0) || ({s}side < -0.5 && {s}d > 0.0)) {{ {s}k = 0.0; }}"
                )
                .unwrap();
                writeln!(l, "    let {s}c = {};", self.vec(node, 1)).unwrap();
                writeln!(l, "    {s}k *= {s}c.a;").unwrap();
                format!("vec4f({s}c.rgb * {s}k, {s}k)")
            }
            SdfNode::Bevel { .. } => {
                let (da, w, az, el, depth) = (d!(0), a!(1), a!(2), a!(3), a!(4));
                let dx = self.inp(node, 0, "p + vec2f(1.0, 0.0)", Mode::Func);
                let dy = self.inp(node, 0, "p + vec2f(0.0, 1.0)", Mode::Func);
                writeln!(l, "    let {s}d = {da};").unwrap();
                writeln!(l, "    let {s}g = vec2f({dx} - {s}d, {dy} - {s}d);").unwrap();
                writeln!(l, "    let {s}t = clamp(-{s}d / max({w}, 1e-3), 0.0, 1.0);").unwrap();
                // Height t(2-t) in units of the rim width, times depth: the
                // slope is steepest at the edge and flat once `width` in.
                writeln!(
                    l,
                    "    let {s}sl = select(0.0, (2.0 - 2.0 * {s}t) * {depth}, ({s}d < 0.0) && ({s}t < 1.0));"
                )
                .unwrap();
                writeln!(l, "    let {s}n = normalize(vec3f({s}g * {s}sl, 1.0));").unwrap();
                writeln!(l, "    let {s}az = radians({az});").unwrap();
                writeln!(l, "    let {s}el = radians({el});").unwrap();
                // y runs DOWN the picture, so a light at +angle (counter-
                // clockwise as seen) has a negative y.
                writeln!(
                    l,
                    "    let {s}l = vec3f(cos({s}el) * cos({s}az), -cos({s}el) * sin({s}az), sin({s}el));"
                )
                .unwrap();
                writeln!(
                    l,
                    "    let {s}sh = max(dot({s}n, {s}l), 0.0) / max({s}l.z, 0.05);"
                )
                .unwrap();
                writeln!(l, "    let {s}c = {};", self.vec(node, 1)).unwrap();
                writeln!(l, "    let {s}k = sdf_cov({s}d, 0.0) * {s}c.a;").unwrap();
                format!("vec4f({s}c.rgb * {s}sh * {s}k, {s}k)")
            }
            SdfNode::Refract { .. } => {
                let da = d!(0);
                format!("bgr * sdf_cov({da}, 0.0)")
            }
            SdfNode::Ramp { .. } => {
                let (da, range) = (d!(0), a!(1));
                writeln!(
                    l,
                    "    let {s}t = clamp({da} / max({range}, 1e-3) * 0.5 + 0.5, 0.0, 1.0);"
                )
                .unwrap();
                writeln!(l, "    let {s}c = mix({}, {}, {s}t);", self.vec(node, 1), self.vec(node, 2))
                    .unwrap();
                format!("vec4f({s}c.rgb * {s}c.a, {s}c.a)")
            }
            SdfNode::Merge { .. } => {
                let (fg, bg, mx) = (d!(0), d!(1), a!(2));
                format!("sdf_merge({fg}, {bg}, {}, {mx})", self.lane(node, 1))
            }
            SdfNode::ImageIn => "bg".into(),
            SdfNode::Output => d!(0),
        };
        // Colour nodes exist only at the pixel.
        if mode == Mode::Func && n.out_pipe() == Pipe::Color {
            return None;
        }
        Some((l, r))
    }
}

/// Shared helpers, emitted once per shader that holds an SDF node.
pub const SDF_WGSL: &str = r#"
// Coverage of a distance: a pixel-wide ramp at least, `soft` pixels at most.
fn sdf_cov(d: f32, soft: f32) -> f32 {
    let h = max(soft * 0.5, 0.5);
    return 1.0 - smoothstep(-h, h, d);
}
fn sdf_smin(a: f32, b: f32, k: f32) -> f32 {
    if (k <= 0.0) { return min(a, b); }
    let h = max(k - abs(a - b), 0.0) / k;
    return min(a, b) - h * h * k * 0.25;
}
fn sdf_combine(a: f32, b: f32, k: f32, op: f32) -> f32 {
    if (op == 1.0) { return -sdf_smin(-a, b, k); }
    if (op == 2.0) { return -sdf_smin(-a, -b, k); }
    if (op == 3.0) { return max(min(a, b), -max(a, b)); }
    return sdf_smin(a, b, k);
}
fn sdf_rect(p: vec2f, c: vec2f, size: vec2f, r0: f32) -> f32 {
    let hs = abs(size) * 0.5;
    let r = clamp(r0, 0.0, min(hs.x, hs.y));
    let q = abs(p - c) - hs + vec2f(r);
    return length(max(q, vec2f(0.0))) + min(max(q.x, q.y), 0.0) - r;
}
fn sdf_spt(base: u32, i: u32) -> vec2f {
    let v = P.v[base + 1u + (i >> 1u)];
    return select(v.xy, v.zw, (i & 1u) == 1u);
}
// Distance to the packed outline. Closed: signed by winding, negative
// inside whichever way it was drawn. Open: unsigned, to the curve.
fn sdf_spline(p0: vec2f, base: u32, off: vec2f) -> f32 {
    let n = u32(P.v[base].x);
    if (n < 2u) { return 1e6; }
    let closed = P.v[base].y > 0.5;
    let p = p0 - off;
    var d = 1e20;
    var sgn = 1.0;
    var j = select(0u, n - 1u, closed);
    for (var i = select(1u, 0u, closed); i < n; i = i + 1u) {
        let a = sdf_spt(base, j);
        let b = sdf_spt(base, i);
        let e = a - b;
        let w = p - b;
        let t = clamp(dot(w, e) / max(dot(e, e), 1e-12), 0.0, 1.0);
        let bq = w - e * t;
        d = min(d, dot(bq, bq));
        if (closed) {
            // Parenthesised: naga reads a bare `<` inside the constructor
            // as a template list.
            let c = vec3<bool>((p.y >= b.y), (p.y < a.y), (e.x * w.y > e.y * w.x));
            if (all(c) || !any(c)) { sgn = -sgn; }
        }
        j = i;
    }
    return sqrt(d) * sgn;
}
fn sdf_merge(fg: vec4f, bg: vec4f, op: f32, m: f32) -> vec4f {
    var c = fg + bg * (1.0 - fg.a);
    if (op == 1.0) { c = fg + bg; }
    if (op == 2.0) { c = vec4f(fg.rgb * bg.rgb + fg.rgb * (1.0 - bg.a) + bg.rgb * (1.0 - fg.a), fg.a + bg.a * (1.0 - fg.a)); }
    if (op == 3.0) { c = vec4f(fg.rgb + bg.rgb - fg.rgb * bg.rgb, fg.a + bg.a * (1.0 - fg.a)); }
    return mix(bg, c, clamp(m, 0.0, 1.0));
}
// A distance as a picture: teal outside, violet inside, a white zero line
// and a contour every 16 pixels so the slope can be read.
fn sdf_viz(d: f32) -> vec4f {
    var c = select(vec3f(0.55, 0.35, 0.85), vec3f(0.15, 0.6, 0.7), d > 0.0);
    c *= 1.0 - exp(-abs(d) / 80.0) * 0.6;
    c *= 0.85 + 0.15 * cos(6.2831853 * d / 16.0);
    c = mix(c, vec3f(1.0), 1.0 - smoothstep(0.0, 1.5, abs(d)));
    return vec4f(c, 1.0);
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    fn g_with(nodes: Vec<(NodeId, SdfNode)>, edges: Vec<((NodeId, usize), NodeId)>) -> SdfGraph {
        let mut g = SdfGraph::default();
        g.nodes.extend(nodes);
        g.edges.extend(edges);
        g
    }

    fn src(p: &Program) -> String {
        p.fns.iter().map(|(_, b)| b.as_str()).collect()
    }

    fn fill() -> SdfNode {
        SdfNode::Fill { color: WHITE, soft: 0.0 }
    }

    /// One spline read by three effects at the pixel walks its outline ONCE:
    /// the entry binds it to a local and every effect reads that.
    #[test]
    fn a_shape_read_by_three_effects_is_evaluated_once() {
        let g = g_with(
            vec![
                (1, SdfNode::default()),
                (2, fill()),
                (3, SdfNode::Stroke { color: WHITE, width: 2.0, soft: 0.0 }),
                (4, SdfNode::Glow { color: WHITE, radius: 10.0, intensity: 1.0 }),
                (5, SdfNode::Merge { op: MergeOp::Over, mix: 1.0 }),
                (6, SdfNode::Merge { op: MergeOp::Add, mix: 1.0 }),
                (7, SdfNode::Output),
            ],
            vec![
                ((2, 0), 1),
                ((3, 0), 1),
                ((4, 0), 1),
                ((5, 0), 3),
                ((5, 1), 2),
                ((6, 0), 4),
                ((6, 1), 5),
                ((7, 0), 6),
            ],
        );
        let p = compile(9, &g, g.output(), 0).expect("compiles");
        let s = src(&p);
        assert_eq!(s.matches("let v1 = ").count(), 1, "one evaluation:\n{s}");
        assert_eq!(s.matches("length(p - vec2f(").count(), 1, "no second circle:\n{s}");
        assert!(p.disp.is_none(), "no refraction, no displacement");
    }

    /// A shadow reads the shape at ANOTHER point, which needs it as a
    /// function; defined before the entry that calls it.
    #[test]
    fn an_offset_read_calls_the_shape_as_a_function() {
        let g = g_with(
            vec![
                (1, SdfNode::default()),
                (2, SdfNode::Shadow { color: WHITE, offset: [4.0, 4.0], soft: 2.0 }),
                (3, SdfNode::Output),
            ],
            vec![((2, 0), 1), ((3, 0), 2)],
        );
        let p = compile(0, &g, g.output(), 0).unwrap();
        let names: Vec<&str> = p.fns.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(names, vec!["sd0_1", "sde0"], "callee first");
        assert!(src(&p).contains("sd0_1((p - vec2f("), "{}", src(&p));
    }

    #[test]
    fn a_cycle_is_refused() {
        let g = g_with(
            vec![
                (1, SdfNode::Offset { amount: 1.0 }),
                (2, SdfNode::Offset { amount: 1.0 }),
                (3, fill()),
                (4, SdfNode::Output),
            ],
            vec![((1, 0), 2), ((2, 0), 1), ((3, 0), 1), ((4, 0), 3)],
        );
        assert!(compile(0, &g, g.output(), 0).is_none());
    }

    /// A Slider wired into a pixel socket is scaled by the proxy like the
    /// literal it replaces; a DISTANCE wired there is already in pixels.
    #[test]
    fn a_driven_px_socket_scales_only_authored_numbers() {
        let g = g_with(
            vec![
                (1, SdfNode::Const { value: 5.0, name: "w".into() }),
                (2, SdfNode::default()),
                (3, SdfNode::Stroke { color: WHITE, width: 2.0, soft: 0.0 }),
                (4, SdfNode::Output),
            ],
            vec![((3, 0), 2), ((3, 1), 1), ((4, 0), 3)],
        );
        let p = compile(0, &g, g.output(), 0).unwrap();
        assert!(src(&p).contains("abs(v2) - (v1 * P.v[0u].y) * 0.5"), "{}", src(&p));
        let mut g2 = g.clone();
        g2.nodes.insert(5, SdfNode::Circle { center: [0.0; 2], radius: 3.0 });
        g2.edges.insert((3, 1), 5);
        let p2 = compile(0, &g2, g2.output(), 0).unwrap();
        assert!(src(&p2).contains("abs(v2) - v5 * 0.5"), "{}", src(&p2));
    }

    #[test]
    fn a_refract_brings_a_displacement() {
        let g = g_with(
            vec![(1, SdfNode::default()), (2, SdfNode::Refract { amount: 9.0, width: 9.0 }), (3, SdfNode::Output)],
            vec![((2, 0), 1), ((3, 0), 2)],
        );
        let p = compile(4, &g, g.output(), 0).unwrap();
        assert_eq!(p.disp.as_deref(), Some("sdd4"));
        assert!(src(&p).contains("bgr * sdf_cov(v1"), "{}", src(&p));
    }

    #[test]
    fn a_dist_preview_is_drawn_as_a_map() {
        let g = g_with(vec![(1, SdfNode::default()), (2, SdfNode::Output)], vec![]);
        let p = compile(0, &g, Some(1), 0).unwrap();
        assert!(src(&p).contains("return sdf_viz(v1);"), "{}", src(&p));
    }

    #[test]
    fn nothing_marked_compiles_to_nothing() {
        let g = g_with(vec![(1, SdfNode::default())], vec![]);
        assert!(g.output().is_none());
        assert!(compile(0, &g, g.output(), 0).is_none());
    }

    #[test]
    fn dragging_a_number_is_not_topology_but_a_point_count_is() {
        let base = g_with(
            vec![
                (1, SdfNode::Spline {
                    points: vec![SplinePoint::new([0.1, 0.1], 0.0); 3],
                    mode: SplineMode::Linear,
                    closed: true,
                    offset: [0.0; 2],
                }),
                (2, fill()),
            ],
            vec![((2, 0), 1)],
        );
        let t0 = topo_tag(&base);
        let mut moved = base.clone();
        if let Some(SdfNode::Spline { points, closed, offset, .. }) = moved.nodes.get_mut(&1) {
            points[0].p = [0.4, 0.2];
            *closed = false;
            offset[0] = 30.0;
        }
        assert_eq!(topo_tag(&moved), t0, "points, closed and offset are uniforms");
        let mut grown = base.clone();
        if let Some(SdfNode::Spline { points, .. }) = grown.nodes.get_mut(&1) {
            points.push(SplinePoint::new([0.3, 0.3], 0.0));
        }
        assert_ne!(topo_tag(&grown), t0, "the point count sizes the span");
    }

    #[test]
    fn literals_round_trip_by_socket_and_pack_where_the_sockets_say() {
        for mut n in SdfNode::catalog() {
            let ins = n.inputs();
            for (i, inp) in ins.iter().enumerate() {
                let has = n.literal(i).is_some();
                assert_eq!(has, inp.lane.is_some(), "{} socket {i}", n.name());
                if has {
                    assert!(n.set_literal(i, 7.25));
                    assert_eq!(n.literal(i), Some(7.25));
                    let mut v = vec![0.0; n.slots() * 4 + 16];
                    n.pack(0, 1000.0, &mut v);
                    assert_eq!(v[inp.lane.unwrap()], 7.25, "{} {} lane", n.name(), inp.name);
                    assert!(inp.lane.unwrap() < n.slots() * 4, "{} lane in span", n.name());
                }
            }
        }
    }

    #[test]
    fn every_kind_emits_valid_wgsl_in_a_real_shader() {
        // A graph with every kind wired into something, compiled through a
        // real comp so naga judges the whole module.
        let mut g = SdfGraph::default();
        let mut id = 1;
        let circle = id;
        g.nodes.insert(circle, SdfNode::default());
        let mut colours = Vec::new();
        for n in SdfNode::catalog() {
            id += 1;
            let ins = n.inputs();
            let pipe = n.out_pipe();
            let is_out = matches!(n, SdfNode::Output);
            g.nodes.insert(id, n);
            for (i, s) in ins.iter().enumerate() {
                if s.pipe == Pipe::Dist && s.lane.is_none() {
                    g.edges.insert((id, i), circle);
                }
            }
            match pipe {
                Pipe::Color if !is_out => colours.push(id),
                Pipe::Color => {}
                // A distance (or a number) reaches the output through a
                // Shadow, so it is compiled BOTH as a pixel value and as a
                // function — every kind judged in both forms.
                Pipe::Dist => {
                    let me = id;
                    id += 1;
                    g.nodes.insert(id, SdfNode::Shadow { color: WHITE, offset: [1.0, 1.0], soft: 1.0 });
                    g.edges.insert((id, 0), me);
                    id += 1;
                    g.nodes.insert(id, fill());
                    g.edges.insert((id, 0), me);
                    colours.push(id - 1);
                    colours.push(id);
                }
            }
        }
        // Chain every colour through Merges into the Output.
        let out = g.nodes.iter().find(|(_, n)| matches!(n, SdfNode::Output)).map(|(i, _)| *i).unwrap();
        let mut acc = colours[0];
        for c in &colours[1..] {
            id += 1;
            g.nodes.insert(id, SdfNode::Merge { op: MergeOp::Screen, mix: 1.0 });
            g.edges.insert((id, 0), *c);
            g.edges.insert((id, 1), acc);
            acc = id;
        }
        g.edges.insert((out, 0), acc);
        let p = compile(3, &g, g.output(), 1).expect("compiles");
        assert!(p.disp.is_some());
        let mut module = String::from("struct Params { v: array<vec4f, 256> };\n@group(0) @binding(0) var<uniform> P: Params;\n");
        module.push_str("fn over(a: vec4f, b: vec4f) -> vec4f { return a + b * (1.0 - a.a); }\n");
        module.push_str("fn fbm_core(p0: vec3f, oct_in: u32, lacunarity: f32, gain: f32, seed: u32, mode: f32) -> f32 { return 0.5; }\n");
        module.push_str(SDF_WGSL);
        module.push_str(&src(&p));
        module.push_str("@compute @workgroup_size(1) fn main() { let x = sde3(vec2f(0.0), vec4f(0.0), vec4f(0.0)); let y = sdd3(vec2f(0.0)); }\n");
        let parsed = naga::front::wgsl::parse_str(&module);
        let m = match parsed {
            Ok(m) => m,
            Err(e) => panic!("{}\n{module}", e.emit_to_string(&module)),
        };
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&m)
            .unwrap_or_else(|e| panic!("{e:?}\n{module}"));
    }
}
