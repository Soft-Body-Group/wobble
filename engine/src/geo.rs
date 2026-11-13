//! 2dini — procedural VECTOR geometry, the third data type.
//!
//! A comp graph moves pictures; a field graph moves functions of position;
//! this one moves GEOMETRY: a set of points carrying attributes, and
//! primitives that index them (Houdini's point/prim model, cut down to what
//! two dimensions need). Nothing here touches a pixel: the graph is
//! evaluated on the CPU into a [`Geometry`], flattened to a [`DrawList`]
//! (segments + per-shape style in output pixels), and the rasteriser in
//! [`crate::georaster`] draws that into the owning node's LAYER — a
//! memoised texture the fused pass merely samples, on the Paint precedent.
//! So nothing in a geometry graph is topology: editing a shape re-records
//! the layer and never recompiles.
//!
//! Coordinates: GEO SPACE == FIELD SPACE. x runs 0..1 across the owning
//! node's format WIDTH and y is in the same units (0..h/w), which is what
//! lets the viewer's roto gizmo serve a geometry spline unchanged and lets
//! a `FieldIn` sample a Noise at `p * scale` exactly as the shader does.
//! Stroke widths and point radii are OUTPUT PIXELS, like a spline's feather.
//!
//! Scalars inside the graph follow the field graph's one rule: a value is a
//! NODE (Slider, Input, FieldIn, Math, Remap) wired into a socket by an
//! ordinary edge; a socket that is wired stops reading its literal. Sliders
//! and Inputs are what the owning node EXPOSES, by name, exactly as a Noise
//! node exposes its own ([`GeoGraph::exposed`]).

use std::collections::BTreeMap;

use crate::field::{FieldGraph, FieldNode, MathOp, WarpMode};
use crate::graph::{NodeId, PortKind, SplineMode, SplinePoint};

/// Caps, so a runaway Scatter × Instance cannot eat the machine. Truncation
/// is reported through [`stats`], never silent.
pub const MAX_POINTS: usize = 4096;
pub const MAX_PRIMS: usize = 4096;
pub const MAX_VERTS_PER_PRIM: usize = 512;
pub const MAX_TOTAL_VERTS: usize = 262_144;

/// A Merge's most inputs, and their socket names.
pub const MAX_MERGE: usize = 32;
const MERGE_INPUTS: [&str; MAX_MERGE] = [
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16",
    "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32",
];
#[cfg(feature = "serde")]
fn two() -> u32 {
    2
}

/// Grow every Merge whose LAST input is wired by one socket, so there is
/// always a free one to wire into (Houdini's Merge). Returns whether any
/// grew. The editor calls it after an edit; a saved graph is already grown.
pub fn grow_merges(g: &mut GeoGraph) -> bool {
    let mut grew = false;
    let ids: Vec<NodeId> = g.nodes.keys().copied().collect();
    for id in ids {
        let wired_last = match g.nodes.get(&id) {
            Some(GeoNode::Merge { count }) => {
                let c = (*count as usize).clamp(2, MAX_MERGE);
                c < MAX_MERGE && g.edges.contains_key(&(id, c - 1))
            }
            _ => false,
        };
        if wired_last {
            if let Some(GeoNode::Merge { count }) = g.nodes.get_mut(&id) {
                *count += 1;
                grew = true;
            }
        }
    }
    grew
}

pub type PointId = u32;

/// One point with its attributes.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Point {
    /// Position, geo space.
    pub p: [f32; 2],
    /// Instance size multiplier; for a bare point prim, its disc radius in
    /// output pixels.
    pub pscale: f32,
    /// Radians.
    pub orient: f32,
    /// Straight (not premultiplied) rgba.
    pub cd: [f32; 4],
    /// Stable identity: the scatter candidate index, the resample index,
    /// or a hash of the instancing point and template prim.
    pub id: u32,
}

impl Point {
    pub fn at(p: [f32; 2]) -> Self {
        Self { p, ..Default::default() }
    }
}

impl Default for Point {
    fn default() -> Self {
        Self { p: [0.0, 0.0], pscale: 1.0, orient: 0.0, cd: [1.0; 4], id: 0 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PrimKind {
    Poly { closed: bool },
    /// One vertex, drawn as a disc of radius `pscale` pixels.
    Point,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FillRule {
    #[default]
    NonZero,
    EvenOdd,
}

impl FillRule {
    pub const ALL: [FillRule; 2] = [FillRule::NonZero, FillRule::EvenOdd];
    pub fn name(self) -> &'static str {
        match self {
            FillRule::NonZero => "non-zero",
            FillRule::EvenOdd => "even-odd",
        }
    }
}

/// How a primitive is drawn. Colours are straight rgba; the flatten step
/// premultiplies. `width` is the stroke width in output pixels, 0 = none.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct Style {
    pub fill: [f32; 4],
    pub stroke: [f32; 4],
    pub width: f32,
    pub rule: FillRule,
    /// A fill that runs from `fill` to the gradient's own colour.
    #[cfg_attr(feature = "serde", serde(default))]
    pub grad: Gradient,
    /// The light it gives off, set by a Light node (`geolight`).
    #[cfg_attr(feature = "serde", serde(default))]
    pub emit: Emit,
}

impl Default for Style {
    fn default() -> Self {
        Self { fill: [1.0; 4], stroke: [0.0; 4], width: 0.0, rule: FillRule::NonZero, grad: Gradient::default(), emit: Emit::default() }
    }
}

/// What a shape does to light, read only when its 2dini node's look has
/// light on (`geolight`): it gives off its own colour (the fill, else the
/// stroke) times `tint` times `strength`, and stands in light's way unless
/// `clear`. Every shape stands in the way by default.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct Emit {
    pub strength: f32,
    pub tint: [f32; 3],
    pub clear: bool,
}

impl Default for Emit {
    fn default() -> Self {
        Self { strength: 0.0, tint: [1.0; 3], clear: false }
    }
}

/// How a gradient fill runs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum GradKind {
    /// Flat: the fill colour.
    #[default]
    None,
    /// From `a` (the fill) to `b` (the gradient colour), along the line.
    Linear,
    /// From the centre `a` (the fill) out to the radius |b - a|.
    Radial,
    /// From the shape's EDGE (the fill) inward to `depth` px (the gradient
    /// colour): the signed distance the rasteriser already has, so it
    /// follows any outline — a bevel, an inner glow, a soft core.
    Edge,
}

impl GradKind {
    pub const ALL: [GradKind; 4] = [GradKind::None, GradKind::Linear, GradKind::Radial, GradKind::Edge];

    pub fn name(self) -> &'static str {
        match self {
            GradKind::None => "flat",
            GradKind::Linear => "linear",
            GradKind::Radial => "radial",
            GradKind::Edge => "from edges",
        }
    }
}

/// Most stops a 2dini gradient holds (the Gradient node's cap).
pub const GEO_GRAD_STOPS: usize = 12;
/// Samples of the ramp handed to the rasteriser per shape: the stops are
/// mixed on the CPU (linear or OkLCh, as the Gradient node mixes them)
/// and the shader lerps between these.
pub const GEO_GRAD_SAMPLES: usize = 32;

/// A gradient's colour stops: (position 0..1, straight rgba), a fixed
/// array so a Style stays `Copy`. Empty = the legacy two-stop ramp, the
/// fill to `Gradient::to`. On the wire a list.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GradStops {
    pub n: u8,
    pub s: [[f32; 5]; GEO_GRAD_STOPS],
}

impl GradStops {
    pub fn as_slice(&self) -> &[[f32; 5]] {
        &self.s[..(self.n as usize).min(GEO_GRAD_STOPS)]
    }

    pub fn from_slice(v: &[[f32; 5]]) -> Self {
        let mut g = GradStops::default();
        for (i, x) in v.iter().take(GEO_GRAD_STOPS).enumerate() {
            g.s[i] = *x;
        }
        g.n = v.len().min(GEO_GRAD_STOPS) as u8;
        g
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for GradStops {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        self.as_slice().serialize(s)
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for GradStops {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let v: Vec<[f32; 5]> = serde::Deserialize::deserialize(d)?;
        Ok(GradStops::from_slice(&v))
    }
}

/// A gradient fill: a ramp of colour stops run along a line, out from a
/// centre, or in from the outline.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct Gradient {
    pub kind: GradKind,
    /// The stops. Empty: the fill colour at 0, `to` at 1 (the first
    /// version of this, which had no stops).
    pub stops: GradStops,
    /// Mix the stops in OkLCh (true) or scene-linear rgb.
    pub lch: bool,
    /// Straight rgba at the end, for a ramp with no stops of its own.
    pub to: [f32; 4],
    /// Geo space: the start (Linear) or centre (Radial).
    pub a: [f32; 2],
    /// Geo space: the end (Linear) or a point on the rim (Radial).
    pub b: [f32; 2],
    /// Edge: how far in, output px, the run from fill to `to` takes.
    pub depth: f32,
}

impl Default for Gradient {
    fn default() -> Self {
        Self {
            kind: GradKind::None,
            stops: GradStops::default(),
            lch: false,
            to: [0.0, 0.0, 0.0, 1.0],
            a: [0.3, 0.2],
            b: [0.7, 0.2],
            depth: 30.0,
        }
    }
}

impl Gradient {
    /// The stops as a Gradient node's ramp, the legacy pair filled in from
    /// `fill` — what the editor edits and what the samples are drawn from.
    pub fn ramp(&self, fill: [f32; 4]) -> crate::nodes::gradient::GradientParams {
        use crate::nodes::gradient::{GradInterp, GradStop, GradientParams};
        let stops: Vec<GradStop> = if self.stops.n == 0 {
            vec![GradStop { pos: 0.0, color: fill }, GradStop { pos: 1.0, color: self.to }]
        } else {
            self.stops.as_slice().iter().map(|s| GradStop { pos: s[0], color: [s[1], s[2], s[3], s[4]] }).collect()
        };
        GradientParams {
            stops,
            interp: if self.lch { GradInterp::Lch } else { GradInterp::Linear },
            ..Default::default()
        }
    }

    /// Take an edited ramp's stops (and its mix) back.
    pub fn set_ramp(&mut self, r: &crate::nodes::gradient::GradientParams) {
        let v: Vec<[f32; 5]> = r.stops.iter().map(|s| [s.pos, s.color[0], s.color[1], s.color[2], s.color[3]]).collect();
        self.stops = GradStops::from_slice(&v);
        self.lch = r.interp == crate::nodes::gradient::GradInterp::Lch;
    }
}

/// How a shape lands on what is drawn before it in the layer. Premultiplied
/// throughout; the rasteriser has one pipeline per mode.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BlendMode {
    #[default]
    Over,
    Add,
    /// Darkens the shapes below by this one's colour (onto nothing: nothing).
    Multiply,
    Screen,
    /// The brighter of the two, per channel.
    Lighten,
    /// The darker of the two, per channel (onto nothing: nothing).
    Darken,
    /// Takes this one's colour away from what is below.
    Subtract,
}

impl BlendMode {
    pub const ALL: [BlendMode; 7] = [
        BlendMode::Over,
        BlendMode::Add,
        BlendMode::Multiply,
        BlendMode::Screen,
        BlendMode::Lighten,
        BlendMode::Darken,
        BlendMode::Subtract,
    ];

    pub fn name(self) -> &'static str {
        match self {
            BlendMode::Over => "over",
            BlendMode::Add => "add",
            BlendMode::Multiply => "multiply",
            BlendMode::Screen => "screen",
            BlendMode::Lighten => "lighten",
            BlendMode::Darken => "darken",
            BlendMode::Subtract => "subtract",
        }
    }

    pub fn index(self) -> u32 {
        Self::ALL.iter().position(|m| *m == self).unwrap_or(0) as u32
    }
}

/// What a NOODLE says about the shapes it carries: where they stack (z,
/// not real depth — a Merge draws its shapes lowest z first, input order
/// breaking ties) and how they land on what is below (the blend). Set per
/// wire in the graph; it adds up along a chain, so a z on a merged group
/// moves the group and keeps its inside order.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct Layering {
    pub z: f32,
    pub blend: BlendMode,
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Prim {
    /// Stack order and blend, from the noodles it came down.
    #[cfg_attr(feature = "serde", serde(default))]
    pub layer: Layering,
    pub kind: PrimKind,
    pub verts: Vec<PointId>,
    pub style: Style,
    /// Physics settings for THIS shape, set by a Rigid Body or Soft Body
    /// node upstream of a Physics node (None = the Physics node's own).
    #[cfg_attr(feature = "serde", serde(default))]
    pub phys: Option<crate::geophys::PhysAttrs>,
    /// The node that made this shape (0 = not stamped yet). A node's output
    /// shapes are stamped with its id unless they already carry one, so a
    /// copy keeps its maker's: clicking a shape in the viewer selects this.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub src: NodeId,
    /// Where each ring after the first starts in `verts` (empty = one
    /// ring). A glyph's counters are rings of its prim: filled under the
    /// fill rule together with the outline, so an "o" has a hole. Offsets
    /// into `verts`, not point ids, so anything that copies or reindexes
    /// `verts` in order keeps them right.
    #[cfg_attr(feature = "serde", serde(default))]
    pub rings: Vec<u32>,
}

/// One point moved by an Edit node.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PointEdit {
    /// The shape's index in the Edit node's input.
    pub prim: u32,
    /// The vertex's index in that shape.
    pub vert: u32,
    /// How far it moved, geo space.
    pub offset: [f32; 2],
}

/// One stroke of a Brush: the hand's path, geo space, with a pressure per
/// point (0..1).
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BrushStroke {
    /// x, y, pressure.
    pub pts: Vec<[f32; 3]>,
}

/// What a geometry graph evaluates to.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Geometry {
    pub points: Vec<Point>,
    pub prims: Vec<Prim>,
    /// Where this geometry turns and scales about: a generator's centre,
    /// carried through whatever moves it (a Transform carries it along),
    /// and dropped where shapes are combined. A Transform downstream with
    /// `inherit` on uses it as its own pivot; `centre()` falls back to the
    /// bounds.
    pub pivot: Option<[f32; 2]>,
    /// A skeleton riding along (a Skeleton or Ragdoll node's output): each
    /// bone at rest and its rest -> pose. A Skin bends shapes by it.
    pub bones: Vec<crate::georig::SkelBone>,
}

impl Geometry {
    /// Append `other`, reindexing its prims past this geometry's points.
    pub fn merge(&mut self, other: Geometry) {
        let base = self.points.len() as PointId;
        self.points.extend(other.points);
        self.prims.extend(other.prims.into_iter().map(|mut p| {
            for v in &mut p.verts {
                *v += base;
            }
            p
        }));
    }

    /// The pivot, else the middle of the bounds.
    pub fn centre(&self) -> Option<[f32; 2]> {
        self.pivot.or_else(|| self.bbox().map(|(lo, hi)| [(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5]))
    }

    /// Bounds of every point, geo space.
    pub fn bbox(&self) -> Option<([f32; 2], [f32; 2])> {
        let mut it = self.points.iter().map(|q| q.p);
        let first = it.next()?;
        let (mut lo, mut hi) = (first, first);
        for p in it {
            lo = [lo[0].min(p[0]), lo[1].min(p[1])];
            hi = [hi[0].max(p[0]), hi[1].max(p[1])];
        }
        Some((lo, hi))
    }

    /// A prim's vertex positions in order.
    pub fn polyline(&self, prim: usize) -> Vec<[f32; 2]> {
        self.prims
            .get(prim)
            .map(|pr| {
                pr.verts
                    .iter()
                    .filter_map(|&v| self.points.get(v as usize).map(|q| q.p))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Per-vertex normals of a poly prim: the mean of its two edges'
    /// normals, unit length, pointing OUT of a closed shape (whichever way
    /// it winds). An open line's point one side, consistently. Empty for a
    /// bare point.
    pub fn normals(&self, prim: usize) -> Vec<[f32; 2]> {
        let Some(pr) = self.prims.get(prim) else { return Vec::new() };
        let closed = match pr.kind {
            PrimKind::Poly { closed } => closed,
            PrimKind::Point => return Vec::new(),
        };
        // One ring at a time, in `verts` order, so the list lines up with
        // `polyline`. Each ring's out is decided by its own winding, so a
        // counter's normals point into the hole — out of the ink.
        let mut out = Vec::new();
        for pl in self.rings(prim) {
            out.extend(ring_normals(&pl, closed));
        }
        out
    }

    /// Mean of a prim's vertices (the anchor a Style socket is evaluated at).
    pub fn centroid(&self, prim: usize) -> [f32; 2] {
        let pl = self.polyline(prim);
        if pl.is_empty() {
            return [0.0, 0.0];
        }
        let n = pl.len() as f32;
        let s = pl.iter().fold([0.0f32, 0.0], |a, p| [a[0] + p[0], a[1] + p[1]]);
        [s[0] / n, s[1] / n]
    }

    /// Is `p` inside any CLOSED poly, each judged alone under its own fill
    /// rule — the CPU mirror of the shader's fill test (one prim at a
    /// time), which is what makes a Scatter's region exactly the fill it
    /// sees: two overlapping rings of opposite winding still both fill,
    /// and an even-odd pentagram's centre is a hole here too.
    pub fn contains(&self, p: [f32; 2]) -> bool {
        self.prims.iter().enumerate().any(|(i, pr)| {
            if pr.kind != (PrimKind::Poly { closed: true }) {
                return false;
            }
            let rings = self.rings(i);
            let wn: i32 = rings.iter().filter(|r| r.len() >= 3).map(|r| winding(r, p)).sum();
            inside(wn, pr.style.rule)
        })
    }

    /// A prim's rings, each its vertex positions in order: the outline
    /// first, then any counters (see `Prim::rings`). One ring for most.
    pub fn rings(&self, prim: usize) -> Vec<Vec<[f32; 2]>> {
        let Some(pr) = self.prims.get(prim) else { return Vec::new() };
        let n = pr.verts.len();
        let mut cuts: Vec<usize> = pr.rings.iter().map(|&r| r as usize).filter(|&r| r > 0 && r < n).collect();
        cuts.sort_unstable();
        cuts.dedup();
        let mut out = Vec::with_capacity(cuts.len() + 1);
        let mut start = 0;
        for end in cuts.into_iter().chain(std::iter::once(n)) {
            out.push(
                pr.verts[start..end]
                    .iter()
                    .filter_map(|&v| self.points.get(v as usize).map(|q| q.p))
                    .collect(),
            );
            start = end;
        }
        out
    }
}

/// Per-vertex normals of one ring: the mean of its two edges' normals,
/// pointing OUT of it as drawn (y down), whichever way it winds.
fn ring_normals(pl: &[[f32; 2]], closed: bool) -> Vec<[f32; 2]> {
    let n = pl.len();
    if n < 2 {
        return vec![[0.0, 0.0]; n];
    }
    let area: f32 = (0..n)
        .map(|i| {
            let (a, b) = (pl[i], pl[(i + 1) % n]);
            a[0] * b[1] - b[0] * a[1]
        })
        .sum();
    let side = if closed && area < 0.0 { -1.0 } else { 1.0 };
    let edge_n = |a: [f32; 2], b: [f32; 2]| {
        let e = [b[0] - a[0], b[1] - a[1]];
        let l = (e[0] * e[0] + e[1] * e[1]).sqrt().max(1e-12);
        [e[1] / l * side, -e[0] / l * side]
    };
    (0..n)
        .map(|i| {
            let prev = if i > 0 { Some(edge_n(pl[i - 1], pl[i])) } else if closed { Some(edge_n(pl[n - 1], pl[0])) } else { None };
            let next = if i + 1 < n { Some(edge_n(pl[i], pl[i + 1])) } else if closed { Some(edge_n(pl[i], pl[0])) } else { None };
            let s = match (prev, next) {
                (Some(a), Some(b)) => [a[0] + b[0], a[1] + b[1]],
                (Some(a), None) | (None, Some(a)) => a,
                (None, None) => [0.0, 0.0],
            };
            let l = (s[0] * s[0] + s[1] * s[1]).sqrt();
            if l > 1e-9 { [s[0] / l, s[1] / l] } else { [0.0, 0.0] }
        })
        .collect()
}

/// Winding number of the closed ring `pl` about `p` (the shader's loop).
fn winding(pl: &[[f32; 2]], p: [f32; 2]) -> i32 {
    let mut wn: i32 = 0;
    let Some(mut j) = pl.len().checked_sub(1) else { return 0 };
    for k in 0..pl.len() {
        let (a, b) = (pl[j], pl[k]);
        let e = [a[0] - b[0], a[1] - b[1]];
        let w = [p[0] - b[0], p[1] - b[1]];
        let up = b[1] <= p[1] && a[1] > p[1];
        let dn = a[1] <= p[1] && b[1] > p[1];
        let side = e[0] * w[1] - e[1] * w[0];
        if up && side > 0.0 {
            wn += 1;
        }
        if dn && side < 0.0 {
            wn -= 1;
        }
        j = k;
    }
    wn
}

fn inside(wn: i32, rule: FillRule) -> bool {
    match rule {
        FillRule::NonZero => wn != 0,
        FillRule::EvenOdd => wn & 1 != 0,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ResampleMode {
    /// A vertex every `spacing` output pixels along the outline.
    #[default]
    Spacing,
    /// Exactly `count` vertices per prim.
    Count,
}

impl ResampleMode {
    pub const ALL: [ResampleMode; 2] = [ResampleMode::Spacing, ResampleMode::Count];
    pub fn name(self) -> &'static str {
        match self {
            ResampleMode::Spacing => "spacing",
            ResampleMode::Count => "count",
        }
    }
}

/// What an instanced template is placed BY.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Pivot {
    #[default]
    BboxCentre,
    Origin,
}

impl Pivot {
    pub const ALL: [Pivot; 2] = [Pivot::BboxCentre, Pivot::Origin];
    pub fn name(self) -> &'static str {
        match self {
            Pivot::BboxCentre => "bbox centre",
            Pivot::Origin => "origin",
        }
    }
}

/// One node inside a geometry graph.
///
/// Every numeric field is a LITERAL a socket can override — wire a Slider or
/// a FieldIn into `radius` and the number stops being read. The socket list
/// ([`GeoNode::inputs`]) and the literals ([`GeoNode::literal`]) run in step:
/// geometry sockets first, then the scalars.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum GeoNode {
    // --- generators (geometry out, no geometry in) -----------------------
    /// A drawn outline, authored in the viewer with the roto gizmo. Points
    /// live in geo space; the polyline is the comp Spline's
    /// (`codegen::spline_polyline`), open or closed.
    Spline {
        points: Vec<SplinePoint>,
        mode: SplineMode,
        closed: bool,
        style: Style,
    },
    /// Physics: input 0's shapes are BODIES, input 1's are static
    /// colliders (closed = solid, open = a wall line). Nothing simulates
    /// until the panel's bake runs; the node then plays the baked motion
    /// at the tick being drawn, and before a bake it passes the bodies
    /// through where they stand. See `geophys`.
    Physics {
        settings: crate::geophys::PhysicsSettings,
        #[cfg_attr(feature = "serde", serde(default))]
        bake: crate::geophys::BakeSlot,
    },
    /// Tags every shape of its input as a RIGID body with these settings,
    /// for a Physics node downstream (Houdini's per-primitive attributes).
    /// Each socket is read at the shape's centre, so a Noise can vary them.
    RigidBody {
        density: f32,
        friction: f32,
        bounce: f32,
        /// Never moves: a collider that is still drawn as a body.
        pinned: bool,
        /// Start velocity, m/s (y down).
        velocity: [f32; 2],
        /// Start spin, radians/s.
        spin: f32,
    },
    /// Tags every shape of its input as a SOFT body with these settings.
    SoftBody {
        blobbiness: f32,
        /// The area it wants, as a multiple of its drawn area.
        pressure: f32,
        friction: f32,
        velocity: [f32; 2],
        spin: f32,
    },
    /// Makes every shape of its input give off light (its own colour times
    /// `tint` times `strength`) for the 2dini node's lighting, and say
    /// whether it casts shadows. Does nothing until the node's look has
    /// light on.
    Light {
        strength: f32,
        tint: [f32; 3],
        /// Stands in light's way: casts shadows.
        occlude: bool,
    },
    /// A drawn LINE: the Spline's outline, always open, stroked in its own
    /// colour and width and never filled — for strokes, where the Spline
    /// is for shapes. Points are drawn in the viewer the same way.
    Stroke {
        points: Vec<SplinePoint>,
        mode: SplineMode,
        /// Straight rgba.
        color: [f32; 4],
        /// Output pixels; a socket.
        width: f32,
    },
    /// Text in a font: one closed shape per glyph outline, its counters as
    /// rings, so an "o" has a hole. See `geotext`.
    Text {
        text: String,
        font: crate::geotext::TextFont,
        /// Em height, geo units (the canvas is 1 wide).
        size: f32,
        /// The first line's baseline anchor.
        position: [f32; 2],
        align: crate::geotext::TextAlign,
        /// Extra space after each glyph, in ems.
        tracking: f32,
        /// Baseline to baseline, in ems.
        leading: f32,
        rotation: f32,
        style: Style,
        /// Animators, case, wrap and the path ride (`textfx`); default =
        /// the plain layout. Its numbers are channels past the sockets:
        /// 6 the path offset, then each animator's `ANIM_PARAMS`.
        #[cfg_attr(feature = "serde", serde(default))]
        fx: crate::textfx::GeoTextFx,
    },
    /// Words on the TEXT pipe (`PortKind::Text`), the comp's String node
    /// inside a 2dini: wired into a Text's words socket
    /// ([`TEXT_WORDS_INPUT`]), its text replaces the Text's own. Draws
    /// nothing.
    String { text: String },
    /// A regular polygon; `step > 1` walks the vertices `i * step mod n`,
    /// which is a star polygon ({5/2} is the pentagram).
    Polygon {
        center: [f32; 2],
        radius: f32,
        sides: u32,
        step: u32,
        rotation: f32,
        style: Style,
    },
    /// Freehand painting as vector shapes: each stroke a filled ribbon
    /// about the path the hand took, as wide as `width` (output px) times
    /// the stroke's pressure at each point, tapering at both ends. Strokes
    /// are painted in the viewer with the node selected.
    Brush {
        strokes: Vec<BrushStroke>,
        /// Straight rgba.
        color: [f32; 4],
        /// Output pixels; a socket.
        width: f32,
        /// How much of each end thins to a point, 0..1 of the stroke.
        taper: f32,
        /// How much the hand's wobble is smoothed out, 0..1.
        smooth: f32,
    },
    /// A circle about `center`, as many sides as its size on screen needs.
    Circle {
        center: [f32; 2],
        radius: f32,
        style: Style,
    },
    /// A rectangle about `center`, `size` wide and high, its corners
    /// rounded by `corner` (geo units, capped at half the short side).
    Rectangle {
        center: [f32; 2],
        size: [f32; 2],
        corner: f32,
        rotation: f32,
        style: Style,
    },
    /// Alternating outer/inner radii: `2 * points` vertices.
    Star {
        center: [f32; 2],
        outer: f32,
        inner: f32,
        points: u32,
        rotation: f32,
        style: Style,
    },
    // --- operators (geometry in 0) ---------------------------------------
    /// Re-vertex every poly evenly along its length.
    Resample { mode: ResampleMode, spacing: f32, count: u32 },
    /// Points inside the closed polys of its input (the whole canvas when
    /// unwired), `density` accepted per candidate — a FieldIn here is what
    /// makes the scatter follow a noise.
    Scatter { count: u32, seed: f32, density: f32 },
    /// Move every point by a scalar pair evaluated AT the point (Offset), or
    /// by the curl of one scalar (Curl) — the field graph's Warp, on points.
    Displace { mode: WarpMode, amount: f32, x: f32, y: f32 },
    /// Copy-to-points: input 0 is the template, input 1 the points it is
    /// stamped onto, each copy scaled by `pscale`, turned by `orient`.
    Instance { scale: f32, rotation: f32, inherit_cd: bool, pivot: Pivot },
    /// Scale and rotate about `pivot`, then move by `translate` — the
    /// order its on-screen handles work in.
    Transform {
        translate: [f32; 2],
        rotate: f32,
        scale: [f32; 2],
        /// Geo space. Existing graphs load with the origin, as before.
        /// With `inherit`, an OFFSET from the incoming geometry's pivot.
        #[cfg_attr(feature = "serde", serde(default))]
        pivot: [f32; 2],
        /// Turn about the incoming geometry's own pivot (a Star's centre),
        /// `pivot` then nudging it. New Transforms start with it on; old
        /// graphs load with it off and keep their absolute pivot.
        #[cfg_attr(feature = "serde", serde(default))]
        inherit: bool,
    },
    /// Input 0 then input 1, which is also the draw order.
    Merge {
        /// How many inputs: one more appears whenever the last is wired, as
        /// Houdini's Merge does. Draw order is input order.
        #[cfg_attr(feature = "serde", serde(default = "two"))]
        count: u32,
    },
    /// Union / intersect / subtract / exclude the filled areas of `a` and
    /// `b` (Houdini's Boolean). See `geobool`.
    Boolean { op: crate::geobool::BoolOp },
    /// Restyle every prim; the colour and width sockets are evaluated at
    /// the prim's centroid, so a FieldIn colours shapes by where they are.
    Style { style: Style },
    /// A bone rig: bones at rest (drawn in the viewer) and each one's
    /// turn about its head, relative to its parent — the POSE, one keyable
    /// channel per bone. Draws the posed bones as lines (no fill, no
    /// stroke: they show in the HUD, not the render) and carries the
    /// skeleton on its output for a Skin.
    Skeleton {
        bones: Vec<crate::georig::Bone>,
        #[cfg_attr(feature = "serde", serde(default))]
        pose: Vec<f32>,
    },
    /// Bends input 0's shapes by input 1's skeleton: each point weighted to
    /// the bones by its distance to them at REST (rigid: the nearest bone
    /// only; smooth: inverse distance to the power `falloff`), then carried
    /// by the blend of their rest -> pose (linear blend skinning).
    Skin {
        smooth: bool,
        falloff: f32,
    },
    /// The skeleton it is fed, played back from a baked simulation: each
    /// bone a capsule, each child hinged to its parent within `limit`,
    /// falling from the pose it was in at `start`. Bakes on a click; until
    /// then it passes the skeleton through.
    Ragdoll {
        gravity: f32,
        limit: f32,
        frames: u32,
        start: f32,
        walls: bool,
        /// Start velocity (widths/s) and spin (radians/s) of the root.
        kick: [f32; 3],
        #[cfg_attr(feature = "serde", serde(default))]
        bake: crate::georig::RagdollBake,
    },
    /// Particles, Houdini's POP network in one node: points born at `rate`
    /// a frame from the emitter (points, or inside its closed shapes; the
    /// canvas centre when unwired), living `life` frames, thrown at
    /// `speed` px a frame in `direction` ± `spread`, pulled by gravity and
    /// wind, slowed by drag. Each one's position is solved in CLOSED FORM
    /// from its age (no state carried frame to frame), so any frame can be
    /// drawn on its own and scrubbing is live. Out come points with
    /// pscale (size over life), orient (along the velocity), Cd alpha (the
    /// fade) and a stable id — an Instance stamps shapes onto them.
    Particles {
        /// Births a frame.
        rate: f32,
        /// Frames a particle lives (± `life_var` of that, per particle).
        life: f32,
        life_var: f32,
        /// Launch speed, output px a frame (± `speed_var` of it).
        speed: f32,
        speed_var: f32,
        /// Launch direction, radians (0 = right, y down), and the cone
        /// either side of it.
        direction: f32,
        spread: f32,
        /// Px a frame², down (+y) and right (+x).
        gravity: f32,
        wind: f32,
        /// Fraction of velocity lost a frame (continuous).
        drag: f32,
        /// Disc radius (px) at birth and at death.
        size: f32,
        size_end: f32,
        /// Alpha falls to 0 over the life.
        fade: bool,
        /// The comp frame emission starts on.
        start: f32,
        seed: f32,
    },
    /// Hand edits on whatever flows in, Houdini's Edit SOP: each moved
    /// point is an offset keyed by (shape, vertex) — the shape's place in
    /// its input and the vertex's place in that shape — so a Resample, a
    /// Star or a Text upstream is edited in the viewer like a drawn path.
    /// Keys past what the input now has are kept (and counted in the
    /// panel) but move nothing, so changing the upstream count never
    /// moves the wrong point silently into a wild place.
    Edit {
        #[cfg_attr(feature = "serde", serde(default))]
        edits: Vec<PointEdit>,
    },
    /// What the graph evaluates to: a marker, passing its input through.
    Output,
    // --- scalars (field out) ---------------------------------------------
    /// The slider: a named number, exposed as a widget on the owning node.
    Slider {
        value: f32,
        #[cfg_attr(feature = "serde", serde(default))]
        name: String,
    },
    /// A named number with no widget: a jack on the owning node that a
    /// comp-level cable drives.
    Input {
        name: String,
        #[cfg_attr(feature = "serde", serde(default))]
        value: f32,
    },
    /// The Noise wired into the owning node's field port, sampled at the
    /// point being evaluated. `shift` offsets the seed, so two FieldIns
    /// decorrelate. Unwired it reads 0.5.
    FieldIn { shift: f32 },
    Math { op: MathOp, a: f32, b: f32 },
    Remap { in_lo: f32, in_hi: f32, out_lo: f32, out_hi: f32, clamp: bool },
    /// A Noise NODE inside the drawing: it owns its own field graph (open
    /// it to build the noise — the field nodes live in there, not here, as
    /// in a Houdini VOP net) and reads as a scalar evaluated at the point
    /// being asked about. Scale, seed and speed mean what they mean on the
    /// comp's Noise node; the time is the 2dini's tick.
    Noise {
        graph: FieldGraph,
        scale: f32,
        seed: f32,
        speed: f32,
    },
}

impl Default for GeoNode {
    fn default() -> Self {
        GeoNode::Polygon {
            center: [0.5, 0.28],
            radius: 0.15,
            sides: 6,
            step: 1,
            rotation: 0.0,
            style: Style::default(),
        }
    }
}

/// An input socket: its label and what it carries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GeoInput {
    pub name: &'static str,
    /// `PortKind::Geo` for geometry, `PortKind::Field` for a scalar.
    pub kind: PortKind,
}

const fn gi(name: &'static str) -> GeoInput {
    GeoInput { name, kind: PortKind::Geo }
}
const fn si(name: &'static str) -> GeoInput {
    GeoInput { name, kind: PortKind::Field }
}

/// A 2dini Text's words socket (`PortKind::Text`): a String wired here
/// replaces the node's own text. Appended after the path socket.
pub const TEXT_WORDS_INPUT: usize = 6;

/// The words a String wired into `node`'s `port` holds, if one is.
pub fn words_on(g: &GeoGraph, node: NodeId, port: usize) -> Option<&str> {
    match g.nodes.get(&g.input_of(node, port)?)? {
        GeoNode::String { text } => Some(text.as_str()),
        _ => None,
    }
}

impl GeoNode {
    pub fn name(&self) -> &'static str {
        match self {
            GeoNode::Spline { .. } => "Spline",
            GeoNode::Stroke { .. } => "Stroke",
            GeoNode::Text { .. } => "Text",
            GeoNode::String { .. } => "String",
            GeoNode::Physics { .. } => "Physics",
            GeoNode::RigidBody { .. } => "Rigid Body",
            GeoNode::SoftBody { .. } => "Soft Body",
            GeoNode::Light { .. } => "Light",
            GeoNode::Polygon { .. } => "Polygon",
            GeoNode::Star { .. } => "Star",
            GeoNode::Circle { .. } => "Circle",
            GeoNode::Brush { .. } => "Brush",
            GeoNode::Rectangle { .. } => "Rectangle",
            GeoNode::Resample { .. } => "Resample",
            GeoNode::Scatter { .. } => "Scatter",
            GeoNode::Displace { .. } => "Displace",
            GeoNode::Instance { .. } => "Instance",
            GeoNode::Transform { .. } => "Transform",
            GeoNode::Merge { .. } => "Merge",
            GeoNode::Boolean { .. } => "Boolean",
            GeoNode::Style { .. } => "Style",
            GeoNode::Output => "Output",
            GeoNode::Edit { .. } => "Edit",
            GeoNode::Particles { .. } => "Particles",
            GeoNode::Skeleton { .. } => "Skeleton",
            GeoNode::Skin { .. } => "Skin",
            GeoNode::Ragdoll { .. } => "Ragdoll",
            GeoNode::Slider { .. } => "Slider",
            GeoNode::Input { .. } => "Input",
            GeoNode::FieldIn { .. } => "Field In",
            GeoNode::Math { .. } => "Math",
            GeoNode::Remap { .. } => "Remap",
            GeoNode::Noise { .. } => "Noise",
        }
    }

    /// Does the node produce a scalar (field) rather than geometry?
    pub fn is_scalar(&self) -> bool {
        matches!(
            self,
            GeoNode::Slider { .. }
                | GeoNode::Input { .. }
                | GeoNode::FieldIn { .. }
                | GeoNode::Math { .. }
                | GeoNode::Remap { .. }
                | GeoNode::Noise { .. }
        )
    }

    /// Does the node put out GEOMETRY (not a number, not words)?
    pub fn is_geometry(&self) -> bool {
        self.output_kind() == PortKind::Geo
    }

    pub fn output_kind(&self) -> PortKind {
        if matches!(self, GeoNode::String { .. }) {
            return PortKind::Text;
        }
        if self.is_scalar() { PortKind::Field } else { PortKind::Geo }
    }

    /// Sockets, in order: geometry first, then the scalars in
    /// [`GeoNode::literal`] order.
    pub fn inputs(&self) -> Vec<GeoInput> {
        match self {
            GeoNode::Spline { .. }
            | GeoNode::Slider { .. }
            | GeoNode::Input { .. }
            | GeoNode::FieldIn { .. }
            | GeoNode::String { .. } => Vec::new(),
            GeoNode::Stroke { .. } => vec![si("width")],
            // The path socket, then the words, come last so wires saved
            // before them keep their ports.
            GeoNode::Text { .. } => vec![
                si("x"),
                si("y"),
                si("size"),
                si("rotation"),
                si("tracking"),
                gi("path"),
                GeoInput { name: "words", kind: PortKind::Text },
            ],
            GeoNode::Physics { .. } => vec![gi("bodies"), gi("colliders")],
            GeoNode::RigidBody { .. } => vec![
                gi("geometry"),
                si("density"),
                si("friction"),
                si("bounce"),
                si("velocity x"),
                si("velocity y"),
                si("spin"),
            ],
            GeoNode::Light { .. } => vec![gi("geometry"), si("strength")],
            GeoNode::SoftBody { .. } => vec![
                gi("geometry"),
                si("blobbiness"),
                si("pressure"),
                si("friction"),
                si("velocity x"),
                si("velocity y"),
                si("spin"),
            ],
            GeoNode::Polygon { .. } => {
                vec![si("centre x"), si("centre y"), si("radius"), si("rotation")]
            }
            GeoNode::Star { .. } => vec![
                si("centre x"),
                si("centre y"),
                si("outer"),
                si("inner"),
                si("rotation"),
            ],
            GeoNode::Circle { .. } => vec![si("centre x"), si("centre y"), si("radius")],
            GeoNode::Brush { .. } => vec![si("width")],
            GeoNode::Rectangle { .. } => vec![
                si("centre x"),
                si("centre y"),
                si("width"),
                si("height"),
                si("corner"),
                si("rotation"),
            ],
            GeoNode::Resample { .. } => vec![gi("geometry"), si("spacing")],
            GeoNode::Scatter { .. } => vec![gi("region"), si("density")],
            GeoNode::Displace { mode, .. } => match mode {
                WarpMode::Offset => vec![gi("geometry"), si("x"), si("y"), si("amount")],
                WarpMode::Curl => vec![gi("geometry"), si("potential"), si("amount")],
            },
            GeoNode::Instance { .. } => {
                vec![gi("template"), gi("points"), si("scale"), si("rotation")]
            }
            GeoNode::Transform { .. } => vec![
                gi("geometry"),
                si("translate x"),
                si("translate y"),
                si("rotate"),
                si("scale x"),
                si("scale y"),
                si("pivot x"),
                si("pivot y"),
            ],
            GeoNode::Merge { count } => {
                MERGE_INPUTS[..(*count as usize).clamp(2, MAX_MERGE)].iter().map(|n| gi(n)).collect()
            }
            GeoNode::Boolean { .. } => vec![gi("a"), gi("b")],
            GeoNode::Style { .. } => vec![
                gi("geometry"),
                si("fill r"),
                si("fill g"),
                si("fill b"),
                si("fill a"),
                si("stroke r"),
                si("stroke g"),
                si("stroke b"),
                si("stroke a"),
                si("width"),
            ],
            GeoNode::Output => vec![gi("geometry")],
            GeoNode::Edit { .. } => vec![gi("geometry")],
            GeoNode::Particles { .. } => vec![gi("emitter"), si("rate"), si("speed"), si("direction"), si("gravity")],
            GeoNode::Skeleton { .. } => Vec::new(),
            GeoNode::Skin { .. } => vec![gi("shapes"), gi("skeleton")],
            GeoNode::Ragdoll { .. } => vec![gi("skeleton")],
            GeoNode::Math { .. } => vec![si("a"), si("b")],
            GeoNode::Remap { .. } => vec![si("value")],
            GeoNode::Noise { .. } => Vec::new(),
        }
    }

    /// The literal behind channel `i` — the number an unwired socket reads,
    /// and what animation and the patch bay drive. Geometry sockets have
    /// none. Channels past the socket list are plain numbers with no socket
    /// (a Slider's value, a Remap's range).
    pub fn literal(&self, i: usize) -> Option<f32> {
        Some(match (self, i) {
            (GeoNode::Slider { value, .. }, 0) => *value,
            (GeoNode::Input { value, .. }, 0) => *value,
            (GeoNode::FieldIn { shift }, 0) => *shift,
            (GeoNode::Stroke { width, .. }, 0) => *width,
            (GeoNode::Text { position, .. }, 0) => position[0],
            (GeoNode::Text { position, .. }, 1) => position[1],
            (GeoNode::Text { size, .. }, 2) => *size,
            (GeoNode::Text { rotation, .. }, 3) => *rotation,
            (GeoNode::Text { tracking, .. }, 4) => *tracking,
            (GeoNode::Text { fx, .. }, TEXT_PATH_OFFSET) => fx.path_offset,
            (GeoNode::Text { fx, .. }, i) if i > TEXT_PATH_OFFSET => match text_style_lit(i) {
                Some(name) => fx.style_get(name)?,
                None => {
                    let (a, k) = text_anim_lit(i);
                    fx.animators.get(a)?.get(crate::textfx::ANIM_PARAMS[k])?
                }
            },
            (GeoNode::Polygon { center, .. }, 0) => center[0],
            (GeoNode::Polygon { center, .. }, 1) => center[1],
            (GeoNode::Polygon { radius, .. }, 2) => *radius,
            (GeoNode::Polygon { rotation, .. }, 3) => *rotation,
            (GeoNode::Star { center, .. }, 0) => center[0],
            (GeoNode::Star { center, .. }, 1) => center[1],
            (GeoNode::Star { outer, .. }, 2) => *outer,
            (GeoNode::Star { inner, .. }, 3) => *inner,
            (GeoNode::Star { rotation, .. }, 4) => *rotation,
            (GeoNode::Brush { width, .. }, 0) => *width,
            (GeoNode::Skeleton { pose, bones }, i) if i < bones.len() => pose.get(i).copied().unwrap_or(0.0),
            (GeoNode::Particles { rate, .. }, 1) => *rate,
            (GeoNode::Particles { speed, .. }, 2) => *speed,
            (GeoNode::Particles { direction, .. }, 3) => *direction,
            (GeoNode::Particles { gravity, .. }, 4) => *gravity,
            (GeoNode::Particles { life, .. }, 5) => *life,
            (GeoNode::Particles { spread, .. }, 6) => *spread,
            (GeoNode::Particles { wind, .. }, 7) => *wind,
            (GeoNode::Particles { drag, .. }, 8) => *drag,
            (GeoNode::Particles { size, .. }, 9) => *size,
            (GeoNode::Particles { size_end, .. }, 10) => *size_end,
            (GeoNode::Circle { center, .. }, 0) => center[0],
            (GeoNode::Circle { center, .. }, 1) => center[1],
            (GeoNode::Circle { radius, .. }, 2) => *radius,
            (GeoNode::Rectangle { center, .. }, 0) => center[0],
            (GeoNode::Rectangle { center, .. }, 1) => center[1],
            (GeoNode::Rectangle { size, .. }, 2) => size[0],
            (GeoNode::Rectangle { size, .. }, 3) => size[1],
            (GeoNode::Rectangle { corner, .. }, 4) => *corner,
            (GeoNode::Rectangle { rotation, .. }, 5) => *rotation,
            (GeoNode::Resample { spacing, .. }, 1) => *spacing,
            (GeoNode::Scatter { density, .. }, 1) => *density,
            (GeoNode::Displace { mode: WarpMode::Offset, x, .. }, 1) => *x,
            (GeoNode::Displace { mode: WarpMode::Offset, y, .. }, 2) => *y,
            (GeoNode::Displace { mode: WarpMode::Offset, amount, .. }, 3) => *amount,
            (GeoNode::Displace { mode: WarpMode::Curl, x, .. }, 1) => *x,
            (GeoNode::Displace { mode: WarpMode::Curl, amount, .. }, 2) => *amount,
            (GeoNode::Instance { scale, .. }, 2) => *scale,
            (GeoNode::Instance { rotation, .. }, 3) => *rotation,
            (GeoNode::Transform { translate, .. }, 1) => translate[0],
            (GeoNode::Transform { translate, .. }, 2) => translate[1],
            (GeoNode::Transform { rotate, .. }, 3) => *rotate,
            (GeoNode::Transform { scale, .. }, 4) => scale[0],
            (GeoNode::Transform { scale, .. }, 5) => scale[1],
            (GeoNode::Transform { pivot, .. }, 6) => pivot[0],
            (GeoNode::Transform { pivot, .. }, 7) => pivot[1],
            (GeoNode::Style { style }, 1..=4) => style.fill[i - 1],
            (GeoNode::Style { style }, 5..=8) => style.stroke[i - 5],
            (GeoNode::Style { style }, 9) => style.width,
            (GeoNode::Math { a, .. }, 0) => *a,
            (GeoNode::Math { b, .. }, 1) => *b,
            (GeoNode::Remap { in_lo, .. }, 1) => *in_lo,
            (GeoNode::Remap { in_hi, .. }, 2) => *in_hi,
            (GeoNode::Remap { out_lo, .. }, 3) => *out_lo,
            (GeoNode::Remap { out_hi, .. }, 4) => *out_hi,
            (GeoNode::RigidBody { density, .. }, 1) => *density,
            (GeoNode::RigidBody { friction, .. }, 2) => *friction,
            (GeoNode::RigidBody { bounce, .. }, 3) => *bounce,
            (GeoNode::RigidBody { velocity, .. }, 4) => velocity[0],
            (GeoNode::RigidBody { velocity, .. }, 5) => velocity[1],
            (GeoNode::RigidBody { spin, .. }, 6) => *spin,
            (GeoNode::Light { strength, .. }, 1) => *strength,
            (GeoNode::SoftBody { blobbiness, .. }, 1) => *blobbiness,
            (GeoNode::SoftBody { pressure, .. }, 2) => *pressure,
            (GeoNode::SoftBody { friction, .. }, 3) => *friction,
            (GeoNode::SoftBody { velocity, .. }, 4) => velocity[0],
            (GeoNode::SoftBody { velocity, .. }, 5) => velocity[1],
            (GeoNode::SoftBody { spin, .. }, 6) => *spin,
            (GeoNode::Noise { scale, .. }, 0) => *scale,
            (GeoNode::Noise { seed, .. }, 1) => *seed,
            (GeoNode::Noise { speed, .. }, 2) => *speed,
            _ => return None,
        })
    }

    /// Write the literal behind channel `i`; false if there is none.
    pub fn set_literal(&mut self, i: usize, v: f32) -> bool {
        let slot: &mut f32 = match (self, i) {
            (GeoNode::Slider { value, .. }, 0) => value,
            (GeoNode::Input { value, .. }, 0) => value,
            (GeoNode::FieldIn { shift }, 0) => shift,
            (GeoNode::Stroke { width, .. }, 0) => width,
            (GeoNode::Text { position, .. }, 0) => &mut position[0],
            (GeoNode::Text { position, .. }, 1) => &mut position[1],
            (GeoNode::Text { size, .. }, 2) => size,
            (GeoNode::Text { rotation, .. }, 3) => rotation,
            (GeoNode::Text { tracking, .. }, 4) => tracking,
            (GeoNode::Text { fx, .. }, TEXT_PATH_OFFSET) => &mut fx.path_offset,
            (GeoNode::Text { fx, .. }, i) if i > TEXT_PATH_OFFSET => {
                if let Some(name) = text_style_lit(i) {
                    return fx.style_set(name, v);
                }
                let (a, k) = text_anim_lit(i);
                return fx.animators.get_mut(a).is_some_and(|an| an.set(crate::textfx::ANIM_PARAMS[k], v));
            }
            (GeoNode::Polygon { center, .. }, 0) => &mut center[0],
            (GeoNode::Polygon { center, .. }, 1) => &mut center[1],
            (GeoNode::Polygon { radius, .. }, 2) => radius,
            (GeoNode::Polygon { rotation, .. }, 3) => rotation,
            (GeoNode::Star { center, .. }, 0) => &mut center[0],
            (GeoNode::Star { center, .. }, 1) => &mut center[1],
            (GeoNode::Star { outer, .. }, 2) => outer,
            (GeoNode::Star { inner, .. }, 3) => inner,
            (GeoNode::Star { rotation, .. }, 4) => rotation,
            (GeoNode::Brush { width, .. }, 0) => width,
            (GeoNode::Skeleton { pose, bones }, i) if i < bones.len() => {
                if pose.len() < bones.len() {
                    pose.resize(bones.len(), 0.0);
                }
                &mut pose[i]
            }
            (GeoNode::Particles { rate, .. }, 1) => rate,
            (GeoNode::Particles { speed, .. }, 2) => speed,
            (GeoNode::Particles { direction, .. }, 3) => direction,
            (GeoNode::Particles { gravity, .. }, 4) => gravity,
            (GeoNode::Particles { life, .. }, 5) => life,
            (GeoNode::Particles { spread, .. }, 6) => spread,
            (GeoNode::Particles { wind, .. }, 7) => wind,
            (GeoNode::Particles { drag, .. }, 8) => drag,
            (GeoNode::Particles { size, .. }, 9) => size,
            (GeoNode::Particles { size_end, .. }, 10) => size_end,
            (GeoNode::Circle { center, .. }, 0) => &mut center[0],
            (GeoNode::Circle { center, .. }, 1) => &mut center[1],
            (GeoNode::Circle { radius, .. }, 2) => radius,
            (GeoNode::Rectangle { center, .. }, 0) => &mut center[0],
            (GeoNode::Rectangle { center, .. }, 1) => &mut center[1],
            (GeoNode::Rectangle { size, .. }, 2) => &mut size[0],
            (GeoNode::Rectangle { size, .. }, 3) => &mut size[1],
            (GeoNode::Rectangle { corner, .. }, 4) => corner,
            (GeoNode::Rectangle { rotation, .. }, 5) => rotation,
            (GeoNode::Resample { spacing, .. }, 1) => spacing,
            (GeoNode::Scatter { density, .. }, 1) => density,
            (GeoNode::Displace { mode: WarpMode::Offset, x, .. }, 1) => x,
            (GeoNode::Displace { mode: WarpMode::Offset, y, .. }, 2) => y,
            (GeoNode::Displace { mode: WarpMode::Offset, amount, .. }, 3) => amount,
            (GeoNode::Displace { mode: WarpMode::Curl, x, .. }, 1) => x,
            (GeoNode::Displace { mode: WarpMode::Curl, amount, .. }, 2) => amount,
            (GeoNode::Instance { scale, .. }, 2) => scale,
            (GeoNode::Instance { rotation, .. }, 3) => rotation,
            (GeoNode::Transform { translate, .. }, 1) => &mut translate[0],
            (GeoNode::Transform { translate, .. }, 2) => &mut translate[1],
            (GeoNode::Transform { rotate, .. }, 3) => rotate,
            (GeoNode::Transform { scale, .. }, 4) => &mut scale[0],
            (GeoNode::Transform { scale, .. }, 5) => &mut scale[1],
            (GeoNode::Transform { pivot, .. }, 6) => &mut pivot[0],
            (GeoNode::Transform { pivot, .. }, 7) => &mut pivot[1],
            (GeoNode::Style { style }, 1..=4) => &mut style.fill[i - 1],
            (GeoNode::Style { style }, 5..=8) => &mut style.stroke[i - 5],
            (GeoNode::Style { style }, 9) => &mut style.width,
            (GeoNode::Math { a, .. }, 0) => a,
            (GeoNode::Math { b, .. }, 1) => b,
            (GeoNode::Remap { in_lo, .. }, 1) => in_lo,
            (GeoNode::Remap { in_hi, .. }, 2) => in_hi,
            (GeoNode::Remap { out_lo, .. }, 3) => out_lo,
            (GeoNode::Remap { out_hi, .. }, 4) => out_hi,
            (GeoNode::RigidBody { density, .. }, 1) => density,
            (GeoNode::RigidBody { friction, .. }, 2) => friction,
            (GeoNode::RigidBody { bounce, .. }, 3) => bounce,
            (GeoNode::RigidBody { velocity, .. }, 4) => &mut velocity[0],
            (GeoNode::RigidBody { velocity, .. }, 5) => &mut velocity[1],
            (GeoNode::RigidBody { spin, .. }, 6) => spin,
            (GeoNode::Light { strength, .. }, 1) => strength,
            (GeoNode::SoftBody { blobbiness, .. }, 1) => blobbiness,
            (GeoNode::SoftBody { pressure, .. }, 2) => pressure,
            (GeoNode::SoftBody { friction, .. }, 3) => friction,
            (GeoNode::SoftBody { velocity, .. }, 4) => &mut velocity[0],
            (GeoNode::SoftBody { velocity, .. }, 5) => &mut velocity[1],
            (GeoNode::SoftBody { spin, .. }, 6) => spin,
            (GeoNode::Noise { scale, .. }, 0) => scale,
            (GeoNode::Noise { seed, .. }, 1) => seed,
            (GeoNode::Noise { speed, .. }, 2) => speed,
            _ => return false,
        };
        *slot = v;
        true
    }

    /// Animatable channels: one per socket, plus the socketless numbers
    /// (a Slider's value, a FieldIn's shift, a Remap's range).
    pub fn channel_count(&self) -> usize {
        match self {
            GeoNode::Slider { .. } | GeoNode::Input { .. } | GeoNode::FieldIn { .. } => 1,
            GeoNode::Remap { .. } => 5,
            GeoNode::Particles { .. } => 11,
            GeoNode::Skeleton { bones, .. } => bones.len(),
            GeoNode::Noise { .. } => 3,
            GeoNode::Text { fx, .. } => text_anim_index(fx.animators.len(), 0),
            _ => self.inputs().len(),
        }
    }

    /// The channel's label: the socket's name, or the plain number's.
    pub fn channel_name(&self, i: usize) -> &'static str {
        // A Text's words socket shares its index with the path offset's
        // channel; the socket carries words, never a number.
        if let Some(s) = self.inputs().get(i).filter(|s| s.kind != PortKind::Text) {
            return s.name;
        }
        match (self, i) {
            (GeoNode::Slider { .. }, 0) | (GeoNode::Input { .. }, 0) => "value",
            (GeoNode::Text { .. }, TEXT_PATH_OFFSET) => "path offset",
            (GeoNode::Text { .. }, i) if i > TEXT_PATH_OFFSET => {
                text_style_lit(i).unwrap_or_else(|| crate::textfx::ANIM_PARAMS[text_anim_lit(i).1])
            }
            (GeoNode::Noise { .. }, 0) => "scale",
            (GeoNode::Noise { .. }, 1) => "seed",
            (GeoNode::Noise { .. }, 2) => "speed",
            (GeoNode::FieldIn { .. }, 0) => "seed shift",
            (GeoNode::Skeleton { .. }, _) => "bone turn",
            (GeoNode::Particles { .. }, 5) => "life",
            (GeoNode::Particles { .. }, 6) => "spread",
            (GeoNode::Particles { .. }, 7) => "wind",
            (GeoNode::Particles { .. }, 8) => "drag",
            (GeoNode::Particles { .. }, 9) => "size",
            (GeoNode::Particles { .. }, 10) => "size at death",
            (GeoNode::Remap { .. }, 1) => "in lo",
            (GeoNode::Remap { .. }, 2) => "in hi",
            (GeoNode::Remap { .. }, 3) => "out lo",
            (GeoNode::Remap { .. }, 4) => "out hi",
            _ => "channel",
        }
    }
}

/// A geometry graph: its own serde type, like a field graph, because it is
/// DATA that lives inside the owning node and saves with the setup.
#[derive(Clone, Debug, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct GeoGraph {
    pub nodes: BTreeMap<NodeId, GeoNode>,
    /// (consumer, socket) -> producer. On the wire a list of pairs: see
    /// `field::edge_map`.
    #[cfg_attr(feature = "serde", serde(with = "crate::field::edge_map"))]
    pub edges: BTreeMap<(NodeId, usize), NodeId>,
    /// The node whose geometry the graph is, when no Output node says.
    pub out: Option<NodeId>,
    /// Canvas positions: authored state, so it belongs with the graph.
    pub layout: BTreeMap<NodeId, [f32; 2]>,
    /// Highest id ever handed out, so a re-opened graph does not reuse one.
    pub next_id: NodeId,
    /// A PREVIEW override: the node the viewer is looking at while you are
    /// inside the 2dini. Wins over the Output node so any node can be
    /// viewed; never saved, and never set on the graph the comp renders.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub view: Option<NodeId>,
    /// Per-noodle stacking and blend, keyed like `edges` by (consumer,
    /// socket). A wire with nothing set is absent.
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Vec::is_empty"))]
    pub wires: Vec<((NodeId, usize), Layering)>,
}

impl GeoGraph {
    /// The layering set on the noodle into (consumer, socket), if any.
    pub fn wire(&self, node: NodeId, socket: usize) -> Option<Layering> {
        self.wires.iter().find(|(k, _)| *k == (node, socket)).map(|(_, l)| *l)
    }
}

/// A curve on a node INSIDE the graph, carried by the 2dini node that owns
/// it as `Custom("inner.<node>.<socket>")` — a GeoGraph holds no curves of
/// its own, so keys made inside a dive (a Skeleton's bone turns, any socket)
/// live on the owner and are applied through `set_inner`.
pub const INNER_PREFIX: &str = "inner.";

/// The owner's channel name for `node`'s channel `i`.
pub fn inner_channel(node: NodeId, i: usize) -> String {
    format!("{INNER_PREFIX}{node}.{i}")
}

/// The (node, channel) an owner's channel name addresses, if it is one.
/// A 2dini Text with its `fx`, as [`text_fx_geometry`] reads it.
struct TextFxSpec<'a> {
    text: &'a str,
    font: &'a crate::geotext::TextFont,
    size: f32,
    position: [f32; 2],
    align: crate::geotext::TextAlign,
    tracking: f32,
    leading: f32,
    rotation: f32,
    style: Style,
    fx: &'a crate::textfx::GeoTextFx,
}

/// A 2dini Text through the Text node's typesetting (`textfx`): laid out,
/// animated per glyph and run along the first shape of `path`, then each
/// glyph's outline placed — so it fills, strokes and lights like any other
/// 2dini shape. Worked in canvas pixels and handed back in geo units.
fn text_fx_geometry(t: &TextFxSpec, path: &Geometry, cx: &EvalCx) -> Geometry {
    use crate::textfx;
    let ppu = cx.px_per_unit.max(1.0);
    let mut out = Geometry::default();
    let shown = textfx::cased(t.text, t.fx.case);
    let Some(laid) = textfx::layout(&textfx::LayoutSpec {
        text: &shown,
        font: t.font,
        tracking: t.tracking * 1000.0,
        kerning: t.fx.kerning,
        leading: t.leading,
        wrap: t.fx.wrap.max(0.0) / t.size,
        justify: t.fx.justify,
        valign: t.fx.valign,
    }) else {
        return out;
    };
    // The path: the first shape's points, in pixels.
    let tpath = path.prims.first().map(|p| textfx::TextPath {
        on: true,
        points: p
            .verts
            .iter()
            .map(|v| {
                let q = path.points[*v as usize].p;
                textfx::PathPoint { p: [q[0] * ppu, q[1] * ppu], ..Default::default() }
            })
            .collect(),
        closed: matches!(p.kind, PrimKind::Poly { closed: true }),
        offset: t.fx.path_offset * ppu,
        perpendicular: t.fx.perpendicular,
        reverse: t.fx.reverse,
    });
    let look = textfx::Look {
        size: t.size * ppu,
        position: [t.position[0] * ppu, t.position[1] * ppu],
        align: t.align,
        rotation: t.rotation.to_degrees(),
        scale: t.fx.scale,
        skew: t.fx.skew,
        color: t.style.fill,
        weight: 0.0,
        softness: 0.0,
        outline: 0.0,
        outline_color: [0.0; 4],
        anims: &t.fx.animators,
        path: tpath.as_ref(),
        // The engine has no frame rate here; wiggles count 24 a second.
        seconds: cx.tick / 24.0,
    };
    // Layers, bottom up, each over EVERY glyph so one glyph's shadow never
    // lands on its neighbour's face: shadow, outline ring, weight, face.
    let size_px = t.size * ppu;
    let weight_px = 2.0 * t.fx.weight.max(0.0) * size_px;
    let ring_px = weight_px + 2.0 * t.fx.outline.max(0.0) * size_px;
    let flat = |c: [f32; 4], width: f32, op: f32| Style {
        fill: [c[0], c[1], c[2], c[3] * op],
        stroke: if width > 0.0 { [c[0], c[1], c[2], c[3] * op] } else { [0.0; 4] },
        width,
        rule: t.style.rule,
        ..Default::default()
    };
    let (mut shadow, mut ring, mut bold) = (Geometry::default(), Geometry::default(), Geometry::default());
    let shift = [t.fx.shadow_offset[0] / ppu, t.fx.shadow_offset[1] / ppu];
    for pl in textfx::places(&laid, &look) {
        let s = pl.ch.to_string();
        let mut gl = crate::geotext::layout(&crate::geotext::TextSpec {
            text: &s,
            font: t.font,
            size: size_px,
            position: [0.0, 0.0],
            align: crate::geotext::TextAlign::Left,
            tracking: 0.0,
            leading: 1.0,
            rotation: 0.0,
            style: t.style,
            px_per_unit: 1.0,
        });
        for p in &mut gl.points {
            let q = pl.apply(p.p);
            p.p = [q[0] / ppu, q[1] / ppu];
        }
        let copy = |style: Style, by: [f32; 2]| {
            let mut c = gl.clone();
            for p in &mut c.points {
                p.p = [p.p[0] + by[0], p.p[1] + by[1]];
            }
            for pr in &mut c.prims {
                pr.style = style;
            }
            c
        };
        if t.fx.shadow {
            shadow.merge(copy(flat(t.fx.shadow_color, ring_px, pl.opacity), shift));
        }
        if t.fx.outline > 0.0 {
            ring.merge(copy(flat(t.fx.outline_color, ring_px, pl.opacity), [0.0; 2]));
        }
        if weight_px > 0.0 {
            bold.merge(copy(flat(pl.fill, weight_px, pl.opacity), [0.0; 2]));
        }
        for pr in &mut gl.prims {
            let st = &mut pr.style;
            st.fill = [pl.fill[0], pl.fill[1], pl.fill[2], pl.fill[3] * pl.opacity];
            st.stroke[3] *= pl.opacity;
        }
        out.merge(gl);
    }
    shadow.merge(ring);
    shadow.merge(bold);
    shadow.merge(out);
    shadow
}

/// A 2dini Text's path-offset channel; its animators' numbers follow.
pub const TEXT_PATH_OFFSET: usize = 6;

/// The literal index of a 2dini Text style number (`GEO_STYLE_PARAMS`):
/// the block between the path offset and the animators.
pub fn text_style_index(k: usize) -> usize {
    TEXT_PATH_OFFSET + 1 + k
}

/// The literal index of a 2dini Text animator's param `k`.
pub fn text_anim_index(a: usize, k: usize) -> usize {
    text_style_index(crate::textfx::GEO_STYLE_PARAMS.len()) + a * crate::textfx::ANIM_PARAMS.len() + k
}

/// (animator, param) of a 2dini Text literal past the path offset.
fn text_anim_lit(i: usize) -> (usize, usize) {
    let j = i - text_anim_index(0, 0);
    let n = crate::textfx::ANIM_PARAMS.len();
    (j / n, j % n)
}

/// The style number a 2dini Text literal names, if it is one.
fn text_style_lit(i: usize) -> Option<&'static str> {
    crate::textfx::GEO_STYLE_PARAMS.get(i.checked_sub(text_style_index(0))?).copied()
}

pub fn parse_inner(name: &str) -> Option<(NodeId, usize)> {
    let (n, i) = name.strip_prefix(INNER_PREFIX)?.split_once('.')?;
    Some((n.parse().ok()?, i.parse().ok()?))
}

impl GeoGraph {
    /// Set an inner node's channel by its owner-side name; false when the
    /// name is not an inner channel or the node is gone.
    pub fn set_inner(&mut self, name: &str, v: f32) -> bool {
        let Some((n, i)) = parse_inner(name) else { return false };
        self.nodes.get_mut(&n).is_some_and(|k| k.set_literal(i, v))
    }
    /// An inner node's channel value by its owner-side name.
    pub fn inner_value(&self, name: &str) -> Option<f32> {
        let (n, i) = parse_inner(name)?;
        self.nodes.get(&n)?.literal(i)
    }
}

/// An unnamed slider still needs a handle to be reached by.
fn exposed_name(n: &str) -> &str {
    if n.is_empty() { "slider" } else { n }
}

impl GeoGraph {
    /// What the graph EXPOSES to the node that owns it: every Slider and
    /// Input, by name, in id order — the same contract as
    /// `FieldGraph::exposed`, so the owning node's panel, channels and jacks
    /// are the Noise node's with the type swapped.
    pub fn exposed(&self) -> Vec<(String, bool)> {
        let mut out: Vec<(String, bool)> = Vec::new();
        for n in self.nodes.values() {
            let (name, is_input) = match n {
                GeoNode::Slider { name, .. } => (name, false),
                GeoNode::Input { name, .. } => (name, true),
                _ => continue,
            };
            let name = exposed_name(name).to_string();
            if !out.iter().any(|(n, _)| *n == name) {
                out.push((name, is_input));
            }
        }
        out
    }
    /// The value behind an exposed name (the first node carrying it).
    pub fn exposed_value(&self, name: &str) -> Option<f32> {
        self.nodes.values().find_map(|n| match n {
            GeoNode::Slider { value, name: nm } | GeoNode::Input { value, name: nm }
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
            if let GeoNode::Slider { value, name: nm } | GeoNode::Input { value, name: nm } = n {
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
    /// The node the graph evaluates to: an explicit Output wins, then the
    /// viewer marker, then whatever geometry nothing else consumes.
    pub fn output(&self) -> Option<NodeId> {
        // A preview of a geometry node wins (a scalar has no geometry to show).
        if let Some(v) = self.view.filter(|v| self.nodes.get(v).is_some_and(|n| n.is_geometry())) {
            return Some(v);
        }
        if let Some(id) =
            self.nodes.iter().find(|(_, n)| matches!(n, GeoNode::Output)).map(|(id, _)| *id)
        {
            return Some(id);
        }
        self.out.filter(|id| self.nodes.contains_key(id)).or_else(|| {
            let consumed: std::collections::HashSet<NodeId> =
                self.edges.values().copied().collect();
            self.nodes
                .iter()
                .filter(|(id, n)| !consumed.contains(id) && n.is_geometry())
                .map(|(id, _)| *id)
                .last()
        })
    }
    /// Does anything in here read the owning node's field port? When
    /// nothing does, the Noise wired there cannot change the geometry, and
    /// the memo key leaves it out.
    pub fn has_field_in(&self) -> bool {
        self.nodes.values().any(|n| matches!(n, GeoNode::FieldIn { .. }))
    }
}

/// The Noise a geometry graph reads, resolved by the compositor from the
/// owning node's field port: the field graph plus the numbers the shader
/// would evaluate it with, so a FieldIn reads exactly what the Noise draws.
#[derive(Clone, Debug, PartialEq)]
pub struct FieldCx {
    pub graph: FieldGraph,
    pub scale: f32,
    pub seed: f32,
    /// `time * speed`; forced to 0 when the field does not move, so a still
    /// noise scrubbed through time never re-evaluates the geometry.
    pub phase: f32,
}

impl FieldCx {
    /// The field at a geo-space point. Same coordinate rule as the shader
    /// (`pos / width * scale`); the mask reads 1, the probe caveat.
    pub fn sample(&self, p: [f32; 2], shift: f32) -> f32 {
        crate::field::eval_cpu(
            &self.graph,
            [p[0] * self.scale, p[1] * self.scale],
            self.seed + shift,
            self.phase,
            1.0,
        )
    }
}

/// Does the field graph consume its phase at all? Only generators with a
/// non-zero speed do.
pub fn field_moves(g: &FieldGraph) -> bool {
    g.nodes.values().any(|n| match n {
        FieldNode::Fbm { speed, .. } | FieldNode::Voronoi { speed, .. } => *speed != 0.0,
        _ => false,
    })
}

/// Does a Noise node inside the drawing move? Only then does the tick reach
/// its evaluation — the rule `field_moves` gives the comp's Noise.
pub fn noise_moves(g: &GeoGraph) -> bool {
    g.nodes
        .values()
        .any(|n| matches!(n, GeoNode::Noise { graph, speed, .. } if *speed != 0.0 && field_moves(graph)))
}

/// Does the graph play a baked simulation? Then the tick is part of what
/// it draws.
pub fn has_bake(g: &GeoGraph) -> bool {
    g.nodes.values().any(|n| {
        matches!(n, GeoNode::Physics { bake, .. } if bake.get().is_some())
            || matches!(n, GeoNode::Particles { .. })
            || matches!(n, GeoNode::Ragdoll { bake, .. } if !bake.is_empty())
    })
}

/// The tick as a memo key: the 2dini's time when something in it depends
/// on time (a moving Noise, a baked Physics), else 0 — so a still drawing
/// scrubbed through time is one evaluation.
pub fn geo_time_key(p: &crate::graph::GeoParams) -> f32 {
    if has_bake(&p.graph) || noise_moves(&p.graph) { p.time } else { 0.0 }
}

/// Everything an evaluation needs from outside the graph.
pub struct EvalCx<'a> {
    pub field: Option<&'a FieldCx>,
    /// The tick being drawn: what a Noise node inside animates with and a
    /// baked Physics node plays back.
    pub tick: f32,
    /// The owning node's seed; every Scatter adds its own.
    pub seed: f32,
    /// Height over width: the canvas in geo space is `[0,1] × [0,aspect]`.
    pub aspect: f32,
    /// Output pixels per geo unit (the format's width), so pixel-valued
    /// sockets — a resample spacing — can be converted.
    pub px_per_unit: f32,
}

/// Evaluate the graph to geometry. Deterministic: the same graph and
/// context give the same points in the same order, which is what makes the
/// layer memo and the scatter's prefix property hold.
///
/// Memoised post-order over the DAG with a depth guard; a cycle evaluates
/// to nothing rather than hanging.
pub fn eval(g: &GeoGraph, cx: &EvalCx) -> Geometry {
    eval_stats(g, cx).0
}

/// [`eval`], plus what the panel reports. `truncated` here is the truth —
/// a cap actually cut something on the way — which the finished geometry
/// cannot tell on its own: sitting exactly on a cap is legal.
pub fn eval_stats(g: &GeoGraph, cx: &EvalCx) -> (Geometry, GeoStats) {
    let Some(out) = g.output() else { return (Geometry::default(), GeoStats::default()) };
    let mut walk = Walk::default();
    let geo = eval_geo(g, out, cx, &mut walk);
    let mut st = stats(&geo);
    st.truncated = walk.truncated;
    (geo, st)
}

/// The geometry ONE node produces — what the viewer's vector HUD shows for
/// the node under selection inside a dive, Houdini's display flag. Same
/// walk as [`eval`], rooted elsewhere; a scalar node is empty.
pub fn eval_node(g: &GeoGraph, node: NodeId, cx: &EvalCx) -> Geometry {
    let mut walk = Walk::default();
    eval_geo(g, node, cx, &mut walk)
}

/// Several nodes' geometry from ONE walk, so a Scatter selected together
/// with the Instance below it is evaluated once, not twice.
pub fn eval_nodes(g: &GeoGraph, nodes: &[NodeId], cx: &EvalCx) -> Vec<Geometry> {
    let mut walk = Walk::default();
    nodes.iter().map(|&n| eval_geo(g, n, cx, &mut walk)).collect()
}

/// The Noise a geometry graph reads, as the CPU evaluator needs it: the
/// numbers the shader would draw the field with, and its phase held still
/// when the field does not move so a scrubbed still noise never changes the
/// geometry. ONE definition, used by the compositor's layer update and by
/// the viewer's HUD, so what is drawn and what is shown agree.
pub fn field_cx_for(n: &crate::graph::NoiseParams) -> FieldCx {
    FieldCx {
        graph: n.graph.clone(),
        scale: n.scale.max(1e-4),
        seed: n.seed,
        phase: if field_moves(&n.graph) { n.phase() } else { 0.0 },
    }
}

/// State of one evaluation: the memo, the nodes on the way down (cycle
/// guard), and whether any cap bit.
#[derive(Default)]
struct Walk {
    memo: BTreeMap<NodeId, Geometry>,
    stack: Vec<NodeId>,
    truncated: bool,
}

/// The geometry a node produces (empty for scalars and cycles).
fn eval_geo(g: &GeoGraph, node: NodeId, cx: &EvalCx, walk: &mut Walk) -> Geometry {
    if let Some(m) = walk.memo.get(&node) {
        return m.clone();
    }
    if walk.stack.contains(&node) || walk.stack.len() > 64 {
        return Geometry::default();
    }
    let Some(kind) = g.nodes.get(&node) else { return Geometry::default() };
    walk.stack.push(node);
    let geo_in = |i: usize, walk: &mut Walk| {
        let mut geo = g.input_of(node, i).map(|s| eval_geo(g, s, cx, walk)).unwrap_or_default();
        // The noodle's own z adds to the shapes'; its blend, when it has
        // one, is theirs.
        if let Some(l) = g.wire(node, i) {
            for pr in &mut geo.prims {
                pr.layer.z += l.z;
                if l.blend != BlendMode::Over {
                    pr.layer.blend = l.blend;
                }
            }
        }
        geo
    };
    // Scalar sockets read AT an anchor: generators at their centre, Style at
    // the prim centroid, Scatter at the candidate, the rest at the point.
    let sk = |i: usize, lit: f32, p: [f32; 2]| socket(g, node, i, lit, p, cx);
    let out = match kind {
        GeoNode::Output => geo_in(0, walk),
        GeoNode::Edit { edits } => {
            let mut geo = geo_in(0, walk);
            apply_edits(&mut geo, edits);
            geo
        }
        GeoNode::Skeleton { bones, pose } => {
            let angles: Vec<f32> = (0..bones.len()).map(|i| sk(i, pose.get(i).copied().unwrap_or(0.0), [0.0, 0.0])).collect();
            skeleton_geometry(crate::georig::pose(bones, &angles))
        }
        GeoNode::Skin { smooth, falloff } => {
            let mut geo = geo_in(0, walk);
            let skel = geo_in(1, walk).bones;
            if !skel.is_empty() {
                for q in &mut geo.points {
                    let w = crate::georig::weights(q.p, &skel, *smooth, *falloff);
                    let p2 = crate::georig::deform(q.p, &skel, &w);
                    // The turn of the heaviest bone turns the point's orient.
                    if let Some((i, _)) = w.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)) {
                        q.orient += skel[i].xf[2].atan2(skel[i].xf[0]);
                    }
                    q.p = p2;
                }
                if let Some(pv) = geo.pivot {
                    let w = crate::georig::weights(pv, &skel, *smooth, *falloff);
                    geo.pivot = Some(crate::georig::deform(pv, &skel, &w));
                }
            }
            geo
        }
        GeoNode::Ragdoll { bake, start, .. } => {
            let mut geo = geo_in(0, walk);
            if !bake.is_empty() && !geo.bones.is_empty() {
                let posed = bake.pose_at(&geo.bones, cx.tick - start);
                geo = skeleton_geometry(posed);
            }
            geo
        }
        GeoNode::Particles {
            rate, life, life_var, speed, speed_var, direction, spread, gravity, wind, drag, size, size_end, fade, start, seed,
        } => {
            let emitter = geo_in(0, walk);
            let at = emitter.centre().unwrap_or([0.5, cx.aspect * 0.5]);
            let p = ParticleSpec {
                rate: sk(1, *rate, at).max(0.0),
                life: sk(5, *life, at).max(0.5),
                life_var: life_var.clamp(0.0, 1.0),
                speed: sk(2, *speed, at),
                speed_var: speed_var.clamp(0.0, 1.0),
                direction: sk(3, *direction, at),
                spread: sk(6, *spread, at),
                gravity: sk(4, *gravity, at),
                wind: sk(7, *wind, at),
                drag: sk(8, *drag, at).max(0.0),
                size: sk(9, *size, at).max(0.0),
                size_end: sk(10, *size_end, at).max(0.0),
                fade: *fade,
                start: *start,
                seed: seed.to_bits() ^ cx.seed.to_bits() ^ node as u32,
            };
            let (g, capped) = particles(&emitter, &p, cx.tick, cx.px_per_unit, cx.aspect);
            walk.truncated |= capped;
            g
        }
        GeoNode::Merge { count } => {
            let mut a = geo_in(0, walk);
            for i in 1..(*count as usize).clamp(2, MAX_MERGE) {
                a.merge(geo_in(i, walk));
            }
            // Lowest z first; a stable sort, so input order breaks ties.
            a.prims.sort_by(|p, q| p.layer.z.total_cmp(&q.layer.z));
            // Many shapes: no one shape's centre speaks for them.
            a.pivot = None;
            a
        }
        GeoNode::Boolean { op } => {
            let (a, b) = (geo_in(0, walk), geo_in(1, walk));
            crate::geobool::boolean(&a, &b, *op, cx.px_per_unit)
        }
        GeoNode::RigidBody { density, friction, bounce, pinned, velocity, spin } => {
            let mut g = geo_in(0, walk);
            for i in 0..g.prims.len() {
                let c = g.centroid(i);
                g.prims[i].phys = Some(crate::geophys::PhysAttrs {
                    kind: crate::geophys::BodyKind::Rigid,
                    density: sk(1, *density, c).max(1e-3),
                    friction: sk(2, *friction, c),
                    bounce: sk(3, *bounce, c),
                    pinned: *pinned,
                    velocity: [sk(4, velocity[0], c), sk(5, velocity[1], c)],
                    spin: sk(6, *spin, c),
                    ..Default::default()
                });
            }
            g
        }
        GeoNode::Light { strength, tint, occlude } => {
            let mut g = geo_in(0, walk);
            for i in 0..g.prims.len() {
                let c = g.centroid(i);
                g.prims[i].style.emit = Emit { strength: sk(1, *strength, c).max(0.0), tint: *tint, clear: !*occlude };
            }
            g
        }
        GeoNode::SoftBody { blobbiness, pressure, friction, velocity, spin } => {
            let mut g = geo_in(0, walk);
            for i in 0..g.prims.len() {
                let c = g.centroid(i);
                g.prims[i].phys = Some(crate::geophys::PhysAttrs {
                    kind: crate::geophys::BodyKind::Soft,
                    blobbiness: sk(1, *blobbiness, c),
                    pressure: sk(2, *pressure, c).max(0.05),
                    friction: sk(3, *friction, c),
                    velocity: [sk(4, velocity[0], c), sk(5, velocity[1], c)],
                    spin: sk(6, *spin, c),
                    ..Default::default()
                });
            }
            g
        }
        GeoNode::Physics { settings, bake } => {
            let mut g = match bake.get() {
                Some(b) => crate::geophys::frame_geometry_look(b, cx.tick, &settings.fluid, cx.px_per_unit),
                None => geo_in(0, walk),
            };
            if settings.show_colliders {
                g.merge(geo_in(1, walk));
            }
            g
        }
        GeoNode::Spline { points, mode, closed, style } => {
            let mut g = curve_geometry(&spline_outline(points, *mode, *closed), *closed, *style);
            g.pivot = points_centre(points);
            g
        }
        // A stroke: the same outline, always open, drawn as a line of its
        // own colour and width and never filled. The width is a socket,
        // read where the stroke starts.
        GeoNode::Text { text, font, size, position, align, tracking, leading, rotation, style, fx } => {
            // A String on the words socket says what is written.
            let text = &words_on(g, node, TEXT_WORDS_INPUT).map(str::to_owned).unwrap_or_else(|| text.clone());
            // Sockets read at the anchor: one number for the whole string.
            let at = *position;
            let path = geo_in(5, walk);
            let size = sk(2, *size, at).max(1e-4);
            let pos = [sk(0, position[0], at), sk(1, position[1], at)];
            let tracking = sk(4, *tracking, at);
            let rotation = sk(3, *rotation, at);
            let mut g = if fx.active() || !path.prims.is_empty() {
                text_fx_geometry(
                    &TextFxSpec {
                        text,
                        font,
                        size,
                        position: pos,
                        align: *align,
                        tracking,
                        leading: *leading,
                        rotation,
                        style: *style,
                        fx,
                    },
                    &path,
                    cx,
                )
            } else {
                // The plain layout, exactly as before the animators.
                crate::geotext::layout(&crate::geotext::TextSpec {
                    text,
                    font,
                    size,
                    position: pos,
                    align: *align,
                    tracking,
                    leading: *leading,
                    rotation,
                    style: *style,
                    px_per_unit: cx.px_per_unit,
                })
            };
            g.pivot = g.bbox().map(|(lo, hi)| [(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5]);
            g
        }
        GeoNode::Stroke { points, mode, color, width } => {
            let at = points.first().map(|q| q.p).unwrap_or([0.0, 0.0]);
            let style = Style {
                fill: [0.0; 4],
                stroke: *color,
                width: sk(0, *width, at).max(0.0),
                rule: FillRule::NonZero,
                ..Default::default()
            };
            let mut g = curve_geometry(&spline_outline(points, *mode, false), false, style);
            g.pivot = points_centre(points);
            g
        }
        GeoNode::Polygon { center, radius, sides, step, rotation, style } => {
            let c = [sk(0, center[0], *center), sk(1, center[1], *center)];
            let r = sk(2, *radius, *center);
            let rot = sk(3, *rotation, *center);
            let n = (*sides).clamp(3, 256) as usize;
            let step = (*step).clamp(1, n as u32 - 1) as usize;
            let radii = |_: usize| r;
            let order = (0..n).map(|i| (i * step) % n);
            regular(c, n, rot, radii, order, *style)
        }
        GeoNode::Brush { strokes, color, width, taper, smooth } => {
            let at = strokes.first().and_then(|s| s.pts.first()).map(|p| [p[0], p[1]]).unwrap_or([0.0; 2]);
            let w = sk(0, *width, at).max(0.0) / cx.px_per_unit.max(1.0);
            let style = Style { fill: *color, stroke: [0.0; 4], width: 0.0, rule: FillRule::NonZero, ..Default::default() };
            let mut g = Geometry::default();
            for st in strokes {
                let ring = brush_ribbon(&st.pts, w, *taper, *smooth);
                if ring.len() < 3 {
                    continue;
                }
                let base = g.points.len() as PointId;
                for (k, p) in ring.iter().enumerate() {
                    g.points.push(Point { p: *p, id: base + k as u32, ..Default::default() });
                }
                let n = g.points.len() as PointId;
                g.prims.push(Prim {
                    layer: Default::default(),
                    kind: PrimKind::Poly { closed: true },
                    verts: (base..n).collect(),
                    style,
                    phys: None,
                    src: 0,
                    rings: Vec::new(),
                });
            }
            g.pivot = g.bbox().map(|(lo, hi)| [(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5]);
            g
        }
        GeoNode::Circle { center, radius, style } => {
            let c = [sk(0, center[0], *center), sk(1, center[1], *center)];
            let r = sk(2, *radius, *center).max(0.0);
            regular(c, circle_sides(r, cx.px_per_unit), 0.0, |_| r, 0..circle_sides(r, cx.px_per_unit), *style)
        }
        GeoNode::Rectangle { center, size, corner, rotation, style } => {
            let c = [sk(0, center[0], *center), sk(1, center[1], *center)];
            let sz = [sk(2, size[0], *center).abs(), sk(3, size[1], *center).abs()];
            let k = sk(4, *corner, *center);
            let rot = sk(5, *rotation, *center);
            rectangle(c, sz, k, rot, cx.px_per_unit, *style)
        }
        GeoNode::Star { center, outer, inner, points, rotation, style } => {
            let c = [sk(0, center[0], *center), sk(1, center[1], *center)];
            let ro = sk(2, *outer, *center);
            let ri = sk(3, *inner, *center);
            let rot = sk(4, *rotation, *center);
            let n = 2 * (*points).clamp(2, 128) as usize;
            let radii = move |k: usize| if k.is_multiple_of(2) { ro } else { ri };
            regular(c, n, rot, radii, 0..n, *style)
        }
        GeoNode::Resample { mode, spacing, count } => {
            let src = geo_in(0, walk);
            // Asking for more vertices than a prim may have IS the cap
            // biting, even though the clamp is what applies it.
            let bit = std::cell::Cell::new(false);
            let mut out = resample(&src, |i| {
                let want = match mode {
                    ResampleMode::Spacing => {
                        let s = sk(1, *spacing, src.centroid(i)).max(1e-6);
                        let l = arc_length(&src.polyline(i), src.prims[i].kind);
                        (l * cx.px_per_unit / s).round() as usize
                    }
                    ResampleMode::Count => *count as usize,
                };
                let closed = src.prims[i].kind == (PrimKind::Poly { closed: true });
                let lo = match mode {
                    ResampleMode::Spacing if closed => 3,
                    _ => 2,
                };
                if want > MAX_VERTS_PER_PRIM {
                    bit.set(true);
                }
                want.clamp(lo, MAX_VERTS_PER_PRIM)
            });
            walk.truncated |= bit.get();
            out.pivot = src.pivot;
            out
        }
        GeoNode::Scatter { count, seed, density } => {
            let region = geo_in(0, walk);
            if *count as usize > MAX_POINTS {
                walk.truncated = true;
            }
            scatter(&region, node, *count, seed + cx.seed, cx, |p| sk(1, *density, p))
        }
        GeoNode::Displace { mode, amount, x, y } => {
            use rayon::prelude::*;
            let mut geo = geo_in(0, walk);
            // Every point is independent and a Curl is five field samples
            // each, so this is the other half of a moving scatter's cost.
            geo.points.par_iter_mut().with_min_len(256).for_each(|q| {
                let p = q.p;
                let d = match mode {
                    WarpMode::Offset => {
                        let amt = sk(3, *amount, p);
                        [(sk(1, *x, p) - 0.5) * amt, (sk(2, *y, p) - 0.5) * amt]
                    }
                    WarpMode::Curl => {
                        // Curl of a scalar potential, (dF/dy, -dF/dx): the
                        // field graph's Warp with the same e, so a Displace
                        // swirls the points exactly as the Noise's Warp
                        // swirls its picture.
                        let e = 1.0 / 512.0;
                        let pot = |q: [f32; 2]| sk(1, *x, q);
                        let amt = sk(2, *amount, p);
                        let dx = pot([p[0] + e, p[1]]) - pot([p[0] - e, p[1]]);
                        let dy = pot([p[0], p[1] + e]) - pot([p[0], p[1] - e]);
                        [dy / (2.0 * e) * amt, -dx / (2.0 * e) * amt]
                    }
                };
                q.p = [p[0] + d[0], p[1] + d[1]];
            });
            geo
        }
        GeoNode::Instance { scale, rotation, inherit_cd, pivot } => {
            let template = geo_in(0, walk);
            if g.input_of(node, 1).is_none() {
                // Nothing to stamp onto yet: show the template where it is.
                template
            } else {
                let pts = geo_in(1, walk);
                let pv = match pivot {
                    Pivot::BboxCentre => template
                        .bbox()
                        .map(|(lo, hi)| [(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5])
                        .unwrap_or([0.0, 0.0]),
                    Pivot::Origin => [0.0, 0.0],
                };
                let mut geo = Geometry::default();
                'copies: for q in &pts.points {
                    let s = sk(2, *scale, q.p) * q.pscale;
                    let rot = sk(3, *rotation, q.p) + q.orient;
                    let (sn, cs) = rot.sin_cos();
                    for (pi, pr) in template.prims.iter().enumerate() {
                        // Whole copies only: a copy that would cross a cap
                        // is not started, so nothing downstream ever sees a
                        // hexagon with four corners.
                        if geo.points.len() + pr.verts.len() > MAX_POINTS
                            || geo.prims.len() >= MAX_PRIMS
                        {
                            walk.truncated = true;
                            break 'copies;
                        }
                        let base = geo.points.len() as PointId;
                        let id = fnv(q.id, pi as u32);
                        for &v in &pr.verts {
                            let Some(t) = template.points.get(v as usize) else { continue };
                            let d = [(t.p[0] - pv[0]) * s, (t.p[1] - pv[1]) * s];
                            geo.points.push(Point {
                                p: [q.p[0] + cs * d[0] - sn * d[1], q.p[1] + sn * d[0] + cs * d[1]],
                                pscale: t.pscale * s,
                                orient: t.orient + rot,
                                cd: t.cd,
                                id,
                            });
                        }
                        let n = geo.points.len() as PointId - base;
                        let mut style = pr.style;
                        if *inherit_cd {
                            for k in 0..4 {
                                style.fill[k] *= q.cd[k];
                            }
                        }
                        geo.prims.push(Prim { layer: Default::default(), kind: pr.kind, verts: (base..base + n).collect(), style, phys: pr.phys, src: pr.src, rings: pr.rings.clone() });
                    }
                }
                geo
            }
        }
        GeoNode::Transform { translate, rotate, scale, pivot, inherit } => {
            let mut geo = geo_in(0, walk);
            let base = if *inherit { geo.centre().unwrap_or([0.0, 0.0]) } else { [0.0, 0.0] };
            // The pivot it hands on: its own, carried by its translate.
            let at = geo.centre().unwrap_or(base);
            let out_pivot = {
                let pv = [base[0] + sk(6, pivot[0], at), base[1] + sk(7, pivot[1], at)];
                [pv[0] + sk(1, translate[0], at), pv[1] + sk(2, translate[1], at)]
            };
            for q in &mut geo.points {
                let p = q.p;
                let t = [sk(1, translate[0], p), sk(2, translate[1], p)];
                let rot = sk(3, *rotate, p);
                let s = [sk(4, scale[0], p), sk(5, scale[1], p)];
                let pv = [base[0] + sk(6, pivot[0], p), base[1] + sk(7, pivot[1], p)];
                let (sn, cs) = rot.sin_cos();
                let d = [(p[0] - pv[0]) * s[0], (p[1] - pv[1]) * s[1]];
                q.p = [pv[0] + t[0] + cs * d[0] - sn * d[1], pv[1] + t[1] + sn * d[0] + cs * d[1]];
                q.orient += rot;
                q.pscale *= (s[0].abs() + s[1].abs()) * 0.5;
            }
            geo.pivot = Some(out_pivot);
            geo
        }
        GeoNode::Style { style } => {
            let mut geo = geo_in(0, walk);
            for i in 0..geo.prims.len() {
                let c = geo.centroid(i);
                let mut st = Style { rule: style.rule, ..*style };
                for k in 0..4 {
                    st.fill[k] = sk(1 + k, style.fill[k], c);
                    st.stroke[k] = sk(5 + k, style.stroke[k], c);
                }
                st.width = sk(9, style.width, c);
                // (What it gives off is a Light node's, upstream or down.)
                st.emit = geo.prims[i].style.emit;
                geo.prims[i].style = st;
            }
            geo
        }
        GeoNode::Slider { .. }
        | GeoNode::Input { .. }
        | GeoNode::FieldIn { .. }
        | GeoNode::Math { .. }
        | GeoNode::Remap { .. }
        | GeoNode::Noise { .. }
        | GeoNode::String { .. } => Geometry::default(),
    };
    let (mut out, bit) = cap(out);
    walk.truncated |= bit;
    // Shapes this node made carry its id; shapes it passed on keep theirs.
    for pr in &mut out.prims {
        if pr.src == 0 {
            pr.src = node;
        }
    }
    walk.stack.pop();
    walk.memo.insert(node, out.clone());
    out
}

/// Which shape of `g` a click at `p` (geo space) lands on, topmost first:
/// inside a closed shape, or within `tol` of an open line or a bare point.
/// Returns the shape's index.
pub fn pick(g: &Geometry, p: [f32; 2], tol: f32) -> Option<usize> {
    let seg_dist = |a: [f32; 2], b: [f32; 2]| {
        let e = [b[0] - a[0], b[1] - a[1]];
        let w = [p[0] - a[0], p[1] - a[1]];
        let t = ((w[0] * e[0] + w[1] * e[1]) / (e[0] * e[0] + e[1] * e[1]).max(1e-12)).clamp(0.0, 1.0);
        let q = [a[0] + e[0] * t - p[0], a[1] + e[1] * t - p[1]];
        (q[0] * q[0] + q[1] * q[1]).sqrt()
    };
    for (i, pr) in g.prims.iter().enumerate().rev() {
        let pl = g.polyline(i);
        let hit = match pr.kind {
            PrimKind::Point => pl.first().is_some_and(|q| {
                let r = g.points.get(pr.verts[0] as usize).map_or(0.0, |q| q.pscale);
                ((q[0] - p[0]).powi(2) + (q[1] - p[1]).powi(2)).sqrt() < tol.max(r * tol)
            }),
            PrimKind::Poly { closed } => {
                let near = g.rings(i).iter().any(|pl| {
                    pl.windows(2).any(|w| seg_dist(w[0], w[1]) < tol)
                        || (closed && pl.len() > 2 && seg_dist(pl[pl.len() - 1], pl[0]) < tol)
                });
                near || (closed && {
                    let one = Geometry { points: g.points.clone(), prims: vec![pr.clone()], pivot: None, bones: Vec::new() };
                    one.contains(p)
                })
            }
        };
        if hit {
            return Some(i);
        }
    }
    None
}

/// A closed, styled ring of `n` vertices about `c`, the first straight up
/// (the polygon a Star or a Polygon is), walked in `order` so a step > 1
/// draws a star polygon.
/// The outline a drawn curve walks, in geo space, with each sample's
/// interpolated feather. Closed, it is the comp spline's ring. Open, the
/// two ends are padded — so the curve clamps there rather than bending
/// toward the far end — only the legs between real points are kept, and it
/// lands exactly on the last point. ONE definition for the Spline and Stroke
/// nodes and the viewer's overlay of them, so what is drawn over the
/// picture is what the picture is. Fewer than two points: nothing.
pub fn spline_outline(points: &[SplinePoint], mode: SplineMode, closed: bool) -> Vec<([f32; 2], f32)> {
    let n = points.len();
    if n < 2 {
        return Vec::new();
    }
    let pts: Vec<SplinePoint> = if closed {
        points.to_vec()
    } else {
        std::iter::once(&points[0])
            .chain(points.iter())
            .chain(std::iter::once(&points[n - 1]))
            .cloned()
            .collect()
    };
    let sp = crate::graph::SplineParams { points: pts, mode, ..Default::default() };
    let mut pl = crate::codegen::spline_polyline(&sp, [1.0, 1.0]);
    if !closed {
        // Padded ring segment i runs p[i-1] -> p[i]; the real legs are
        // 1..n-1, `subd` samples each.
        let subd = crate::codegen::spline_subd(n + 2);
        pl.truncate(n * subd);
        pl.drain(..subd.min(pl.len()));
        pl.push((points[n - 1].p, points[n - 1].feather));
    }
    pl
}

/// One poly prim along an outline.
fn curve_geometry(pl: &[([f32; 2], f32)], closed: bool, style: Style) -> Geometry {
    let mut geo = Geometry::default();
    if pl.is_empty() {
        return geo;
    }
    for (i, (p, _)) in pl.iter().enumerate() {
        geo.points.push(Point { p: *p, id: i as u32, ..Default::default() });
    }
    let verts = (0..geo.points.len() as PointId).collect();
    geo.prims.push(Prim { layer: Default::default(), kind: PrimKind::Poly { closed }, verts, style, phys: None, src: 0, rings: Vec::new() });
    geo
}

fn regular(
    c: [f32; 2],
    n: usize,
    rotation: f32,
    radius: impl Fn(usize) -> f32,
    order: impl Iterator<Item = usize>,
    style: Style,
) -> Geometry {
    let mut geo = Geometry::default();
    for k in 0..n {
        let a = rotation - std::f32::consts::FRAC_PI_2 + k as f32 * std::f32::consts::TAU / n as f32;
        let r = radius(k);
        geo.points.push(Point { p: [c[0] + r * a.cos(), c[1] + r * a.sin()], id: k as u32, ..Default::default() });
    }
    geo.prims.push(Prim {
        layer: Default::default(),
        kind: PrimKind::Poly { closed: true },
        verts: order.map(|i| i as PointId).collect(),
        style, phys: None, src: 0, rings: Vec::new() });
    geo.pivot = Some(c);
    geo
}

/// A skeleton as geometry: each posed bone a two-point open line with no
/// fill and no stroke (it shows in the HUD's outlines, never the render),
/// the bones themselves carried along for a Skin.
pub fn skeleton_geometry(bones: Vec<crate::georig::SkelBone>) -> Geometry {
    let mut g = Geometry::default();
    for b in &bones {
        let base = g.points.len() as PointId;
        for p in [b.head, b.tail] {
            g.points.push(Point { p: crate::georig::apply(&b.xf, p), id: base, ..Default::default() });
        }
        g.prims.push(Prim {
            layer: Default::default(),
            kind: PrimKind::Poly { closed: false },
            verts: vec![base, base + 1],
            style: Style { fill: [0.0; 4], stroke: [0.0; 4], width: 0.0, ..Default::default() },
            phys: None,
            src: 0,
            rings: Vec::new(),
        });
    }
    g.pivot = bones.first().map(|b| crate::georig::apply(&b.xf, b.head));
    g.bones = bones;
    g
}

/// A Particles node's numbers, after sockets.
struct ParticleSpec {
    rate: f32,
    life: f32,
    life_var: f32,
    speed: f32,
    speed_var: f32,
    direction: f32,
    spread: f32,
    gravity: f32,
    wind: f32,
    drag: f32,
    size: f32,
    size_end: f32,
    fade: bool,
    start: f32,
    seed: u32,
}

/// Where a particle launched with `v0` (px/frame) is `a` frames later,
/// and how fast it is going, under constant acceleration `g` (px/frame²)
/// and linear drag `k` (per frame) — solved, not stepped, so every frame
/// is drawn on its own.
fn ballistic(v0: [f32; 2], g: [f32; 2], k: f32, a: f32) -> ([f32; 2], [f32; 2]) {
    if k > 1e-5 {
        let e = (-k * a).exp();
        let f = (1.0 - e) / k;
        let h = (a - f) / k;
        (
            [v0[0] * f + g[0] * h, v0[1] * f + g[1] * h],
            [v0[0] * e + g[0] * f, v0[1] * e + g[1] * f],
        )
    } else {
        (
            [v0[0] * a + 0.5 * g[0] * a * a, v0[1] * a + 0.5 * g[1] * a * a],
            [v0[0] + g[0] * a, v0[1] + g[1] * a],
        )
    }
}

/// The live particles at `tick`: one point prim each. Particle k is born
/// at `start + k / rate`, so the set at any frame is a pure function of
/// the frame (and the emitter as it stands then).
fn particles(emitter: &Geometry, p: &ParticleSpec, tick: f32, ppu: f32, aspect: f32) -> (Geometry, bool) {
    let mut geo = Geometry::default();
    let t = tick - p.start;
    if t < 0.0 || p.rate <= 0.0 {
        return (geo, false);
    }
    let max_life = p.life * (1.0 + p.life_var);
    let last = (t * p.rate).floor() as i64;
    let first = (((t - max_life) * p.rate).floor() as i64).max(0);
    // Where to be born: inside closed shapes, else on points, else the
    // canvas centre.
    let closed = closed_bbox(emitter);
    let ppu = ppu.max(1.0);
    let mut capped = false;
    for k in first..=last {
        if geo.points.len() >= MAX_POINTS {
            capped = true;
            break;
        }
        let k32 = k as u32;
        let r = |i: u32| hash01(p.seed, k32.wrapping_mul(8).wrapping_add(i));
        let birth = k as f32 / p.rate;
        let life = p.life * (1.0 + p.life_var * (2.0 * r(0) - 1.0));
        let age = t - birth;
        if age < 0.0 || age >= life {
            continue;
        }
        let origin = match closed {
            Some((lo, hi)) => {
                let mut q = [(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5];
                for j in 0..16u32 {
                    let c = [
                        lo[0] + (hi[0] - lo[0]) * hash01(p.seed ^ 0x9e37, k32.wrapping_mul(37).wrapping_add(j)),
                        lo[1] + (hi[1] - lo[1]) * hash01(p.seed ^ 0x5bd1, k32.wrapping_mul(31).wrapping_add(j)),
                    ];
                    if emitter.contains(c) {
                        q = c;
                        break;
                    }
                }
                q
            }
            None if !emitter.points.is_empty() => {
                let i = (r(2) * emitter.points.len() as f32) as usize;
                emitter.points[i.min(emitter.points.len() - 1)].p
            }
            None => [0.5, aspect * 0.5],
        };
        let ang = p.direction + p.spread * (2.0 * r(3) - 1.0);
        let sp = p.speed * (1.0 + p.speed_var * (2.0 * r(4) - 1.0));
        let v0 = [ang.cos() * sp, ang.sin() * sp];
        let (d, v) = ballistic(v0, [p.wind, p.gravity], p.drag, age);
        let u = (age / life).clamp(0.0, 1.0);
        let id = fnv(k32, p.seed);
        let base = geo.points.len() as PointId;
        geo.points.push(Point {
            p: [origin[0] + d[0] / ppu, origin[1] + d[1] / ppu],
            pscale: p.size + (p.size_end - p.size) * u,
            orient: v[1].atan2(v[0]),
            cd: [1.0, 1.0, 1.0, if p.fade { 1.0 - u } else { 1.0 }],
            id,
        });
        geo.prims.push(Prim {
            layer: Default::default(),
            kind: PrimKind::Point,
            verts: vec![base],
            style: Style { fill: [1.0, 1.0, 1.0, if p.fade { 1.0 - u } else { 1.0 }], ..Default::default() },
            phys: None,
            src: 0,
            rings: Vec::new(),
        });
    }
    (geo, capped)
}

/// Move the points an Edit names. A point shared by two shapes moves
/// once per edit naming it; keys past the geometry are skipped.
pub fn apply_edits(geo: &mut Geometry, edits: &[PointEdit]) {
    for e in edits {
        let Some(&v) = geo.prims.get(e.prim as usize).and_then(|p| p.verts.get(e.vert as usize)) else {
            continue;
        };
        if let Some(q) = geo.points.get_mut(v as usize) {
            q.p = [q.p[0] + e.offset[0], q.p[1] + e.offset[1]];
        }
    }
}

/// How many of an Edit's keys name nothing in `geo` (the input changed).
pub fn orphan_edits(geo: &Geometry, edits: &[PointEdit]) -> usize {
    edits
        .iter()
        .filter(|e| geo.prims.get(e.prim as usize).and_then(|p| p.verts.get(e.vert as usize)).is_none())
        .count()
}

/// The middle of a drawn outline's control points: its pivot.
fn points_centre(points: &[SplinePoint]) -> Option<[f32; 2]> {
    let first = points.first()?.p;
    let (mut lo, mut hi) = (first, first);
    for q in points {
        lo = [lo[0].min(q.p[0]), lo[1].min(q.p[1])];
        hi = [hi[0].max(q.p[0]), hi[1].max(q.p[1])];
    }
    Some([(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5])
}

/// A brush stroke's outline: the path smoothed, then offset either side
/// by half the width at each point (pressure times the taper ramp), down
/// one side and back up the other, with a round cap at each end. Geo
/// space; `w` is the full width in geo units.
pub fn brush_ribbon(pts: &[[f32; 3]], w: f32, taper: f32, smooth: f32) -> Vec<[f32; 2]> {
    // Drop points that do not move the pen: a still hand draws nothing.
    let mut path: Vec<[f32; 3]> = Vec::with_capacity(pts.len());
    for p in pts {
        if path.last().is_none_or(|q: &[f32; 3]| ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2)).sqrt() > 1e-5) {
            path.push(*p);
        }
    }
    if path.is_empty() || w <= 0.0 {
        return Vec::new();
    }
    // Smooth: a moving average over a window the slider sets, ends held.
    let k = (smooth.clamp(0.0, 1.0) * 8.0).round() as usize;
    if k > 0 && path.len() > 2 {
        let src = path.clone();
        for i in 1..src.len() - 1 {
            let (a, b) = (i.saturating_sub(k), (i + k).min(src.len() - 1));
            let n = (b - a + 1) as f32;
            let mut m = [0.0f32; 3];
            for q in &src[a..=b] {
                for c in 0..3 {
                    m[c] += q[c] / n;
                }
            }
            path[i] = m;
        }
    }
    // Arc length, for the taper.
    let mut s = vec![0.0f32; path.len()];
    for i in 1..path.len() {
        s[i] = s[i - 1] + ((path[i][0] - path[i - 1][0]).powi(2) + (path[i][1] - path[i - 1][1]).powi(2)).sqrt();
    }
    let total = s[path.len() - 1].max(1e-9);
    let t = taper.clamp(0.0, 1.0) * 0.5 * total;
    let half = |i: usize| {
        let ramp = if t > 0.0 { (s[i].min(total - s[i]) / t).clamp(0.0, 1.0) } else { 1.0 };
        // An eased ramp, never quite to nothing, so a tapered end is a point
        // and not a hole.
        let e = 0.15 + 0.85 * (ramp * (2.0 - ramp));
        0.5 * w * path[i][2].clamp(0.0, 1.0) * e
    };
    let n = path.len();
    let cap = |c: [f32; 3], r: f32, dir: [f32; 2], out: &mut Vec<[f32; 2]>| {
        // Half a circle from the left side round to the right, facing `dir`.
        let a0 = dir[1].atan2(dir[0]) + std::f32::consts::FRAC_PI_2;
        for j in 1..8 {
            let a = a0 - std::f32::consts::PI * j as f32 / 8.0;
            out.push([c[0] + r * a.cos(), c[1] + r * a.sin()]);
        }
    };
    if n == 1 {
        // A dab: a disc.
        let r = half(0).max(0.5 * w * 0.15);
        return (0..16)
            .map(|j| {
                let a = std::f32::consts::TAU * j as f32 / 16.0;
                [path[0][0] + r * a.cos(), path[0][1] + r * a.sin()]
            })
            .collect();
    }
    let tangent = |i: usize| {
        let (a, b) = (path[i.saturating_sub(1)], path[(i + 1).min(n - 1)]);
        let d = [b[0] - a[0], b[1] - a[1]];
        let l = (d[0] * d[0] + d[1] * d[1]).sqrt().max(1e-9);
        [d[0] / l, d[1] / l]
    };
    let mut left = Vec::with_capacity(n);
    let mut right = Vec::with_capacity(n);
    for i in 0..n {
        let d = tangent(i);
        let nrm = [-d[1], d[0]];
        let h = half(i);
        left.push([path[i][0] + nrm[0] * h, path[i][1] + nrm[1] * h]);
        right.push([path[i][0] - nrm[0] * h, path[i][1] - nrm[1] * h]);
    }
    let mut out = left;
    let d_end = tangent(n - 1);
    cap(path[n - 1], half(n - 1), d_end, &mut out);
    out.extend(right.into_iter().rev());
    let d0 = tangent(0);
    cap(path[0], half(0), [-d0[0], -d0[1]], &mut out);
    out
}

/// Sides for a circle of radius `r` (geo units) to read round: about one
/// every 4 output pixels of its outline, never fewer than 24.
fn circle_sides(r: f32, px_per_unit: f32) -> usize {
    ((std::f32::consts::TAU * r * px_per_unit / 4.0).ceil() as usize).clamp(24, 256)
}

/// A (rounded) rectangle about `c`: one closed outline, clockwise from the
/// top-left corner as drawn, turned by `rot`.
fn rectangle(c: [f32; 2], size: [f32; 2], corner: f32, rot: f32, px_per_unit: f32, style: Style) -> Geometry {
    let (hw, hh) = (size[0] * 0.5, size[1] * 0.5);
    let k = corner.clamp(0.0, hw.min(hh));
    // A corner's arc: centre, starting angle. y is down, so the angles run
    // top-left, top-right, bottom-right, bottom-left.
    use std::f32::consts::{FRAC_PI_2, PI};
    let arcs = [
        ([-hw + k, -hh + k], PI),
        ([hw - k, -hh + k], -FRAC_PI_2),
        ([hw - k, hh - k], 0.0),
        ([-hw + k, hh - k], FRAC_PI_2),
    ];
    let steps = if k > 0.0 { (circle_sides(k, px_per_unit) / 4).max(2) } else { 0 };
    let (sn, cs) = rot.sin_cos();
    let mut geo = Geometry::default();
    for (o, a0) in arcs {
        for j in 0..=steps {
            let a = a0 + FRAC_PI_2 * j as f32 / steps.max(1) as f32;
            let l = [o[0] + k * a.cos(), o[1] + k * a.sin()];
            let id = geo.points.len() as u32;
            geo.points.push(Point {
                p: [c[0] + cs * l[0] - sn * l[1], c[1] + sn * l[0] + cs * l[1]],
                id,
                ..Default::default()
            });
            if k == 0.0 {
                break;
            }
        }
    }
    let n = geo.points.len() as PointId;
    geo.prims.push(Prim {
        layer: Default::default(),
        kind: PrimKind::Poly { closed: true },
        verts: (0..n).collect(),
        style,
        phys: None,
        src: 0,
        rings: Vec::new(),
    });
    geo.pivot = Some(c);
    geo
}

/// Outline length of a poly in geo units, including the wrap segment when
/// it is closed.
fn arc_length(pl: &[[f32; 2]], kind: PrimKind) -> f32 {
    let mut l = 0.0;
    for w in pl.windows(2) {
        l += dist(w[0], w[1]);
    }
    if kind == (PrimKind::Poly { closed: true }) && pl.len() > 2 {
        l += dist(pl[pl.len() - 1], pl[0]);
    }
    l
}

fn dist(a: [f32; 2], b: [f32; 2]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

/// Re-vertex every poly with `want(prim)` vertices evenly along its
/// outline, attributes interpolated. Point prims pass through.
fn resample(src: &Geometry, want: impl Fn(usize) -> usize) -> Geometry {
    let mut geo = Geometry::default();
    for (i, pr) in src.prims.iter().enumerate() {
        let base = geo.points.len() as PointId;
        let PrimKind::Poly { closed } = pr.kind else {
            // Copy the point through, reindexed.
            let verts: Vec<PointId> = pr
                .verts
                .iter()
                .filter_map(|&v| src.points.get(v as usize))
                .map(|q| {
                    geo.points.push(*q);
                    geo.points.len() as PointId - 1
                })
                .collect();
            geo.prims.push(Prim { layer: Default::default(), kind: pr.kind, verts, style: pr.style, phys: pr.phys, src: pr.src, rings: pr.rings.clone() });
            continue;
        };
        // Each ring on its own; the count is shared out by length, so a
        // glyph's counter gets its share of the vertices.
        let n = pr.verts.len();
        let mut cuts: Vec<usize> = pr.rings.iter().map(|&r| r as usize).filter(|&r| r > 0 && r < n).collect();
        cuts.sort_unstable();
        cuts.dedup();
        let mut bounds = Vec::new();
        let mut start = 0;
        for end in cuts.into_iter().chain(std::iter::once(n)) {
            bounds.push((start, end));
            start = end;
        }
        let rings: Vec<Vec<Point>> = bounds
            .iter()
            .map(|(a, b)| pr.verts[*a..*b].iter().filter_map(|&v| src.points.get(v as usize).copied()).collect())
            .collect();
        let lens: Vec<f32> = rings
            .iter()
            .map(|r| arc_length(&r.iter().map(|q| q.p).collect::<Vec<_>>(), pr.kind))
            .collect();
        let total: f32 = lens.iter().sum();
        if rings.iter().all(|r| r.len() < 2) || total <= 0.0 {
            for r in &rings {
                geo.points.extend(r.iter().copied());
            }
            let m = geo.points.len() as PointId - base;
            geo.prims.push(Prim { layer: Default::default(), kind: pr.kind, verts: (base..base + m).collect(), style: pr.style, phys: pr.phys, src: pr.src, rings: pr.rings.clone() });
            continue;
        }
        let want_n = want(i);
        let mut offsets = Vec::new();
        for (r, (ring, len)) in rings.iter().zip(&lens).enumerate() {
            if r > 0 {
                offsets.push(geo.points.len() as u32 - base);
            }
            let min = if closed { 3 } else { 2 };
            let share = if rings.len() == 1 {
                want_n
            } else {
                ((want_n as f32 * len / total).round() as usize).max(min)
            };
            if ring.len() < 2 || *len <= 0.0 {
                geo.points.extend(ring.iter().copied());
                continue;
            }
            resample_ring(ring, closed, *len, share, &mut geo.points);
        }
        let m = geo.points.len() as PointId - base;
        geo.prims.push(Prim { layer: Default::default(), kind: pr.kind, verts: (base..base + m).collect(), style: pr.style, phys: pr.phys, src: pr.src, rings: offsets });
    }
    geo
}

/// `n` vertices evenly along one ring of points, appended to `out`.
fn resample_ring(pts: &[Point], closed: bool, total: f32, n: usize, out: &mut Vec<Point>) {
    // Segments a->b with their start distance; closed adds the wrap.
    let segs = pts.len() - if closed { 0 } else { 1 };
    let seg = |k: usize| (pts[k], pts[(k + 1) % pts.len()]);
    let step = if closed { total / n as f32 } else { total / (n.max(2) - 1) as f32 };
    let (mut k, mut start) = (0usize, 0.0f32);
    for j in 0..n {
        let s = (j as f32 * step).min(total);
        // Advance to the segment holding s; the last one absorbs rounding
        // so the final open vertex lands on the endpoint.
        while k + 1 < segs && start + dist(seg(k).0.p, seg(k).1.p) < s {
            start += dist(seg(k).0.p, seg(k).1.p);
            k += 1;
        }
        let (a, b) = seg(k);
        let len = dist(a.p, b.p);
        let t = if len > 0.0 { ((s - start) / len).clamp(0.0, 1.0) } else { 0.0 };
        let lerp = |x: f32, y: f32| x + (y - x) * t;
        out.push(Point {
            p: [lerp(a.p[0], b.p[0]), lerp(a.p[1], b.p[1])],
            pscale: lerp(a.pscale, b.pscale),
            orient: (b.p[1] - a.p[1]).atan2(b.p[0] - a.p[0]),
            cd: [lerp(a.cd[0], b.cd[0]), lerp(a.cd[1], b.cd[1]), lerp(a.cd[2], b.cd[2]), lerp(a.cd[3], b.cd[3])],
            id: j as u32,
        });
    }
}

fn fract(x: f32) -> f32 {
    x - x.floor()
}

/// Integer hash to [0,1): the acceptance test is a pure function of
/// (seed, candidate index), which is what makes the prefix property hold.
fn hash01(seed: u32, k: u32) -> f32 {
    // lowbias32 over the pair, then the top 24 bits.
    let mut x = seed ^ k.wrapping_mul(0x9E37_79B9);
    x ^= x >> 16;
    x = x.wrapping_mul(0x7FEB_352D);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846C_A68B);
    x ^= x >> 16;
    (x >> 8) as f32 / (1u32 << 24) as f32
}

/// FNV-1a over two words: a stable id for a (point, prim) copy.
fn fnv(a: u32, b: u32) -> u32 {
    let mut h: u32 = 0x811C_9DC5;
    for byte in a.to_le_bytes().into_iter().chain(b.to_le_bytes()) {
        h ^= byte as u32;
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}

/// Bounds of the closed polys' vertices: the part of a region that has an
/// inside.
fn closed_bbox(g: &Geometry) -> Option<([f32; 2], [f32; 2])> {
    let mut acc: Option<([f32; 2], [f32; 2])> = None;
    for (i, pr) in g.prims.iter().enumerate() {
        if pr.kind != (PrimKind::Poly { closed: true }) || pr.verts.len() < 3 {
            continue;
        }
        for p in g.polyline(i) {
            acc = Some(match acc {
                None => (p, p),
                Some((lo, hi)) => ([lo[0].min(p[0]), lo[1].min(p[1])], [hi[0].max(p[0]), hi[1].max(p[1])]),
            });
        }
    }
    acc
}

/// Points inside `region` (the canvas when there is none), placed on the R2
/// low-discrepancy sequence and thinned by `density`. Candidate k is a pure
/// function of the seed, so lowering `count` drops a tail and nothing else
/// moves. Candidates are judged in parallel, a wave at a time, and kept in
/// index order — the wave is what stops a dense scatter from paying for
/// all 16·count candidates when the first few thousand were enough.
fn scatter(
    region: &Geometry,
    node: NodeId,
    count: u32,
    seed: f32,
    cx: &EvalCx,
    density: impl Fn([f32; 2]) -> f32 + Sync,
) -> Geometry {
    use rayon::prelude::*;
    let count = (count as usize).min(MAX_POINTS);
    let mut geo = Geometry::default();
    if count == 0 {
        return geo;
    }
    // The region is the closed polys' joint bbox with a containment test.
    // A wired input with nothing closed in it (an open outline, bare
    // points) scatters over its bbox instead: the user pointed at
    // something, so the canvas would be wrong and nothing would be a
    // silent failure.
    let (lo, size, test) = match (region.prims.is_empty(), closed_bbox(region), region.bbox()) {
        (false, Some((lo, hi)), _) => (lo, [hi[0] - lo[0], hi[1] - lo[1]], true),
        (false, None, Some((lo, hi))) => (lo, [hi[0] - lo[0], hi[1] - lo[1]], false),
        _ => ([0.0, 0.0], [1.0, cx.aspect.max(1e-6)], false),
    };
    let off = [fract(seed * 0.618034) as f64, fract(seed * 0.324718) as f64];
    let bits = seed.to_bits() ^ node as u32;
    let candidate = |k: usize| -> Option<[f32; 2]> {
        // R2 (the plastic constant's powers), in f64: at f32 the integer
        // part of k·α eats the mantissa past k ≈ 1000 and the candidates
        // collapse onto a lattice.
        let kf = k as f64;
        let u = [
            (0.5 + kf * 0.754_877_666_246_692_8 + off[0]).fract() as f32,
            (0.5 + kf * 0.569_840_290_998_053_3 + off[1]).fract() as f32,
        ];
        let p = [lo[0] + u[0] * size[0], lo[1] + u[1] * size[1]];
        if test && !region.contains(p) {
            return None;
        }
        // A NaN density (a Math chain can make one) accepts nobody, not
        // everybody: `>= NaN` is false, so the test has to be phrased as
        // acceptance.
        let d = density(p);
        let d = if d.is_nan() { 0.0 } else { d.clamp(0.0, 1.0) };
        (hash01(bits, k as u32) < d).then_some(p)
    };
    const CHUNK: usize = 512;
    let total = 16 * count;
    let wave = rayon::current_num_threads().max(1) * CHUNK;
    let mut accepted: Vec<(u32, [f32; 2])> = Vec::with_capacity(count);
    let mut start = 0;
    while start < total && accepted.len() < count {
        let end = (start + wave).min(total);
        if end - start <= CHUNK {
            for k in start..end {
                if let Some(p) = candidate(k) {
                    accepted.push((k as u32, p));
                    if accepted.len() >= count {
                        break;
                    }
                }
            }
        } else {
            let chunks: Vec<Vec<(u32, [f32; 2])>> = (start..end)
                .step_by(CHUNK)
                .collect::<Vec<_>>()
                .into_par_iter()
                .map(|c0| {
                    (c0..(c0 + CHUNK).min(end))
                        .filter_map(|k| candidate(k).map(|p| (k as u32, p)))
                        .collect()
                })
                .collect();
            for c in chunks {
                for x in c {
                    if accepted.len() >= count {
                        break;
                    }
                    accepted.push(x);
                }
            }
        }
        start = end;
    }
    for (k, p) in accepted {
        let id = geo.points.len() as PointId;
        geo.points.push(Point { p, id: k, ..Default::default() });
        geo.prims.push(Prim { layer: Default::default(), kind: PrimKind::Point, verts: vec![id], style: Style::default(), phys: None, src: 0, rings: Vec::new() });
    }
    geo
}

/// Enforce the MAX_* caps by truncation — points first, then the prims
/// that survive them — so a runaway node is bounded before the next one
/// multiplies it. A prim whose points fell to the cap is dropped whole
/// (a partial ring would draw as a shape nobody asked for); one over the
/// vertex caps is trimmed, since its points are all still there. Returns
/// whether anything was cut, which is what [`eval_stats`] reports.
fn cap(mut geo: Geometry) -> (Geometry, bool) {
    let had = geo.points.len() as PointId;
    let mut bit = geo.points.len() > MAX_POINTS || geo.prims.len() > MAX_PRIMS;
    geo.points.truncate(MAX_POINTS);
    geo.prims.truncate(MAX_PRIMS);
    let np = geo.points.len() as PointId;
    let mut total = 0usize;
    let mut prims = Vec::with_capacity(geo.prims.len());
    for mut pr in geo.prims {
        if pr.verts.iter().any(|&v| (np..had).contains(&v)) {
            bit = true;
            continue;
        }
        // A vertex with no point behind it was never a vertex.
        pr.verts.retain(|&v| v < np);
        let keep = MAX_VERTS_PER_PRIM.min(MAX_TOTAL_VERTS - total);
        if pr.verts.len() > keep {
            bit = true;
            pr.verts.truncate(keep);
        }
        if pr.verts.is_empty() {
            continue;
        }
        total += pr.verts.len();
        prims.push(pr);
    }
    geo.prims = prims;
    (geo, bit)
}

/// A scalar socket's value at point `p`: the wired scalar node evaluated
/// there, else the literal.
pub fn socket(g: &GeoGraph, node: NodeId, i: usize, lit: f32, p: [f32; 2], cx: &EvalCx) -> f32 {
    match g.input_of(node, i) {
        Some(src) => eval_scalar(g, src, p, cx, 0),
        None => lit,
    }
}

/// Where a scalar was asked: node, point and seed, bit for bit. A Warp
/// asks its source somewhere else and a Seed with another seed, so the
/// memo is per question, not per node.
type ScalarKey = (NodeId, [u32; 3]);

fn scalar_key(node: NodeId, p: [f32; 2], sd: f32) -> ScalarKey {
    (node, [p[0].to_bits(), p[1].to_bits(), sd.to_bits()])
}

/// Scratch for one scalar walk at one point: what each node came to, and
/// the nodes on the way down. Without the memo a node read by two sockets
/// is evaluated twice, and a chain of those is 2^depth per point; without
/// the stack a loaded graph with a loop in it hangs. Thread-local so the
/// per-candidate walk in a Scatter allocates nothing after the first, on
/// whichever rayon thread it lands.
#[derive(Default)]
struct ScalarWalk {
    memo: Vec<(ScalarKey, f32)>,
    stack: Vec<NodeId>,
}

thread_local! {
    static SCALAR_WALK: std::cell::RefCell<ScalarWalk> = std::cell::RefCell::new(ScalarWalk::default());
}

/// A scalar node at point `p`. Scalars recurse through their own sockets,
/// so `FieldIn -> Remap -> density` is the ordinary shaping chain.
pub fn eval_scalar(g: &GeoGraph, node: NodeId, p: [f32; 2], cx: &EvalCx, depth: u32) -> f32 {
    SCALAR_WALK.with(|cell| match cell.try_borrow_mut() {
        Ok(mut w) => {
            w.memo.clear();
            w.stack.clear();
            scalar_at(g, node, p, cx.seed, cx, depth, &mut w)
        }
        // Re-entered from inside a walk (nothing does today): a fresh one.
        Err(_) => scalar_at(g, node, p, cx.seed, cx, depth, &mut ScalarWalk::default()),
    })
}

fn scalar_at(
    g: &GeoGraph,
    node: NodeId,
    p: [f32; 2],
    sd: f32,
    cx: &EvalCx,
    depth: u32,
    w: &mut ScalarWalk,
) -> f32 {
    // A cycle is refused by the stack; a Warp asking its OWN source at
    // another point is not a cycle, but only depth bounds a deep chain.
    if depth > 64 || w.stack.contains(&node) {
        return 0.0;
    }
    let key = scalar_key(node, p, sd);
    if let Some(&(_, v)) = w.memo.iter().find(|(k, _)| *k == key) {
        return v;
    }
    let Some(kind) = g.nodes.get(&node) else { return 0.0 };
    w.stack.push(node);
    let mut sock = |i: usize, lit: f32| match g.input_of(node, i) {
        Some(src) => scalar_at(g, src, p, sd, cx, depth + 1, w),
        None => lit,
    };
    let v = match kind {
        GeoNode::Slider { value, .. } | GeoNode::Input { value, .. } => *value,
        GeoNode::FieldIn { shift } => cx.field.map(|f| f.sample(p, *shift)).unwrap_or(0.5),
        // The Noise node's own field graph, sampled here exactly as the
        // comp's Noise node would draw it (`pos / width * scale`).
        GeoNode::Noise { graph, scale, seed, speed } => {
            let sc = scale.max(1e-4);
            crate::field::eval_cpu(graph, [p[0] * sc, p[1] * sc], cx.seed + seed, cx.tick * speed, 1.0)
        }
        GeoNode::Math { op, a, b } => {
            let (x, y) = (sock(0, *a), sock(1, *b));
            crate::field::math_cpu(x, y, *op)
        }
        GeoNode::Remap { in_lo, in_hi, out_lo, out_hi, clamp } => {
            let v = sock(0, 0.0);
            let mut t = (v - in_lo) / (in_hi - in_lo).abs().max(1e-6) * (in_hi - in_lo).signum();
            if *clamp {
                t = t.clamp(0.0, 1.0);
            }
            out_lo + (out_hi - out_lo) * t
        }
        _ => 0.0,
    };
    w.stack.pop();
    if w.memo.len() < 4096 {
        w.memo.push((key, v));
    }
    v
}

// ---------------------------------------------------------------------------
// Flatten: geometry -> what the rasteriser draws.
// ---------------------------------------------------------------------------

/// One shape as the rasteriser sees it. 80 bytes, `repr(C)` for the storage
/// buffer; colours PREMULTIPLIED; bbox in output pixels, padded for the
/// stroke and the AA ramp.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PrimGpu {
    pub bbox: [f32; 4],
    pub fill: [f32; 4],
    pub stroke: [f32; 4],
    /// First vertex in the draw list's vertex array, and how many.
    pub first: u32,
    pub count: u32,
    /// 0 open poly, 1 closed poly, 2 disc, 3 a flat quad wholly inside a
    /// filled shape (no segment walk; see [`TILE`]).
    pub kind: u32,
    /// 0 non-zero, 1 even-odd.
    pub rule: u32,
    /// Stroke width, px.
    pub width: f32,
    /// Disc radius, px (kind 2).
    pub radius: f32,
    /// `BlendMode::index`: which pipeline draws it.
    pub blend: u32,
    pub _pad: f32,
    /// Where the gradient's samples start in the vertex array (two vec2
    /// each: r g, b a, premultiplied), and how many.
    pub grad_first: u32,
    pub grad_n: u32,
    pub _pad3: [f32; 2],
    /// Linear: start xy, end zw; Radial: centre xy, rim zw (px).
    pub grad_ab: [f32; 4],
    /// 0 flat, 1 linear, 2 radial, 3 from edges.
    pub grad_kind: u32,
    /// Edge: px from the outline to the end colour.
    pub grad_depth: f32,
    pub _pad2: [f32; 2],
}

/// The flattened drawing: shapes in draw order, vertices in output pixels.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DrawList {
    pub prims: Vec<PrimGpu>,
    pub verts: Vec<[f32; 2]>,
}

impl DrawList {
    /// What the rasteriser will do: every pixel of every quad walks its
    /// shape's segments, so the cost is the sum over quads of area times
    /// segment count (a flat interior tile costs its area once). The
    /// number the panel shows, so a drawing that is expensive to re-record
    /// is never expensive silently.
    pub fn edge_tests(&self) -> u64 {
        self.prims
            .iter()
            .map(|p| {
                let area = ((p.bbox[2] - p.bbox[0]).max(0.0) * (p.bbox[3] - p.bbox[1]).max(0.0))
                    as u64;
                area * (p.count.max(1) as u64)
            })
            .sum()
    }
}

fn premul(c: [f32; 4]) -> [f32; 4] {
    let a = c[3].clamp(0.0, 1.0);
    [c[0].max(0.0) * a, c[1].max(0.0) * a, c[2].max(0.0) * a, a]
}

/// Tile edge, px, for cutting up a big shape's quad. The rasteriser pays
/// bbox area × segments per shape, so a full-frame outline that paints 1%
/// of its bbox costs 100× what it draws, and a full-frame FILL walks every
/// segment for every pixel of its interior. Cut into tiles, only the ones
/// the outline touches keep the walk; a filled shape's interior tiles are
/// flat quads (kind 3), the rest are dropped. Tiles abut on integer pixel
/// boundaries, so no pixel is shaded twice and the picture is identical
/// tiled or not. (v2 is a tile-binned compute pass; this is the CPU half
/// of that behind the same draw list.)
pub const TILE: f32 = 64.0;
/// Between two rings of one shape in the draw list's vertices: the shader
/// ends a ring (closing it on its own start) when it meets one.
pub const RING_BREAK: [f32; 2] = [-1e30, -1e30];
/// Draw-list instances: a tiled shape is many, so this sits above
/// MAX_PRIMS; every prim can still have its one whole quad.
const MAX_INSTANCES: usize = 1 << 16;

/// Cut one shape's quad into tiles (see [`TILE`]). None when the shape
/// fits one tile or would take more than `budget` instances — the caller
/// draws the one big quad instead, which is always right, just slower.
fn tile(
    proto: &PrimGpu,
    rings: &[Vec<[f32; 2]>],
    has_fill: bool,
    rule: FillRule,
    pad: f32,
    budget: usize,
) -> Option<Vec<PrimGpu>> {
    let b = proto.bbox;
    let (tx0, ty0) = ((b[0] / TILE).floor() as i32, (b[1] / TILE).floor() as i32);
    let (tx1, ty1) = (((b[2] / TILE).ceil() as i32).max(tx0 + 1), ((b[3] / TILE).ceil() as i32).max(ty0 + 1));
    let (nx, ny) = ((tx1 - tx0) as usize, (ty1 - ty0) as usize);
    // A shape that spans a few tiles has nothing to skip: whole.
    if nx * ny < 6 || rings.iter().all(|r| r.len() < 2) {
        return None;
    }
    // Which tiles the outline touches: every segment's padded bbox over the
    // grid. A tile nothing touches is wholly in or wholly out.
    let mut touched = vec![false; nx * ny];
    let segments: Vec<([f32; 2], [f32; 2])> = rings
        .iter()
        .flat_map(|pts| {
            let n = pts.len();
            let segs = if proto.kind == 1 { n } else { n.saturating_sub(1) };
            (0..segs).map(move |s| (pts[s], pts[(s + 1) % n]))
        })
        .collect();
    for (a, c) in segments {
        let lo = [a[0].min(c[0]) - pad, a[1].min(c[1]) - pad];
        let hi = [a[0].max(c[0]) + pad, a[1].max(c[1]) + pad];
        let x0 = ((lo[0] / TILE).floor() as i32 - tx0).clamp(0, nx as i32) as usize;
        let x1 = ((hi[0] / TILE).ceil() as i32 - tx0).clamp(0, nx as i32) as usize;
        let y0 = ((lo[1] / TILE).floor() as i32 - ty0).clamp(0, ny as i32) as usize;
        let y1 = ((hi[1] / TILE).ceil() as i32 - ty0).clamp(0, ny as i32) as usize;
        for y in y0..y1 {
            touched[y * nx + x0..y * nx + x1].fill(true);
        }
    }
    let mut out = Vec::new();
    for ty in 0..ny {
        let mut tx = 0;
        while tx < nx {
            // A run of like tiles is one quad.
            let hit = touched[ty * nx + tx];
            let run0 = tx;
            while tx < nx && touched[ty * nx + tx] == hit {
                tx += 1;
            }
            let cell = [
                (tx0 + run0 as i32) as f32 * TILE,
                (ty0 + ty as i32) as f32 * TILE,
                (tx0 + tx as i32) as f32 * TILE,
                (ty0 + ty as i32 + 1) as f32 * TILE,
            ];
            if !hit {
                // Nothing crosses the run, so one winding test at its
                // centre speaks for all of it.
                if !has_fill {
                    continue;
                }
                let c = [(cell[0] + cell[2]) * 0.5, (cell[1] + cell[3]) * 0.5];
                let wn: i32 = rings.iter().map(|r| winding(r, c)).sum();
                if !inside(wn, rule) {
                    continue;
                }
            }
            let bbox = [cell[0].max(b[0]), cell[1].max(b[1]), cell[2].min(b[2]), cell[3].min(b[3])];
            if bbox[2] <= bbox[0] || bbox[3] <= bbox[1] {
                continue;
            }
            if out.len() >= budget {
                return None;
            }
            out.push(if hit {
                PrimGpu { bbox, ..*proto }
            } else {
                PrimGpu { bbox, kind: 3, stroke: [0.0; 4], width: 0.0, count: 0, ..*proto }
            });
        }
    }
    Some(out)
}

/// Geometry to pixels. y is in WIDTH units (geo space), so both axes scale
/// by `w`. Invisible, degenerate and wholly off-canvas shapes are dropped.
pub fn flatten(g: &Geometry, w: u32, h: u32) -> DrawList {
    let (wf, hf) = (w as f32, h as f32);
    let mut dl = DrawList::default();
    for (i, pr) in g.prims.iter().enumerate() {
        let st = pr.style;
        let fill = premul(st.fill);
        let stroke = premul(st.stroke);
        let width = if st.width.is_finite() { st.width.max(0.0) } else { 0.0 };
        let has_stroke = stroke[3] > 0.0 && width > 0.0;
        let (kind, radius) = match pr.kind {
            PrimKind::Poly { closed } => (if closed { 1u32 } else { 0 }, 0.0),
            PrimKind::Point => {
                let r = pr
                    .verts
                    .first()
                    .and_then(|&v| g.points.get(v as usize))
                    .map(|q| q.pscale)
                    .unwrap_or(0.0);
                (2, if r.is_finite() { r.max(0.0) } else { 0.0 })
            }
        };
        // An open poly has no interior; a fill on it would be a lie.
        let graded = st.grad.kind != GradKind::None;
        let has_fill = (fill[3] > 0.0 || graded) && kind != 0;
        if !has_fill && !has_stroke {
            continue;
        }
        let need = match kind {
            2 => 1,
            1 => 3,
            _ => 2,
        };
        // Each ring cleaned on its own; rings too small to draw are dropped.
        let mut rings: Vec<Vec<[f32; 2]>> = Vec::new();
        for ring in g.rings(i) {
            let mut pts: Vec<[f32; 2]> = ring
                .into_iter()
                .map(|p| [p[0] * wf, p[1] * wf])
                .filter(|p| p[0].is_finite() && p[1].is_finite())
                .collect();
            if kind == 2 {
                pts.truncate(1);
            } else {
                // Zero-length segments contribute nothing but a division.
                pts.dedup_by(|a, b| (a[0] - b[0]).abs() < 1e-4 && (a[1] - b[1]).abs() < 1e-4);
                if kind == 1 && pts.len() > 1 {
                    let (f, l) = (pts[0], pts[pts.len() - 1]);
                    if (f[0] - l[0]).abs() < 1e-4 && (f[1] - l[1]).abs() < 1e-4 {
                        pts.pop();
                    }
                }
                if pts.len() > MAX_VERTS_PER_PRIM {
                    pts.truncate(MAX_VERTS_PER_PRIM);
                }
            }
            if pts.len() >= need {
                rings.push(pts);
            }
            if kind == 2 {
                break;
            }
        }
        if rings.is_empty() {
            continue;
        }
        let total: usize = rings.iter().map(|r| r.len()).sum::<usize>() + rings.len() - 1;
        if dl.verts.len() + total > MAX_TOTAL_VERTS || dl.prims.len() >= MAX_INSTANCES {
            break;
        }
        let pad = width * 0.5 + 1.5 + radius;
        let (mut lo, mut hi) = (rings[0][0], rings[0][0]);
        for p in rings.iter().flatten() {
            lo = [lo[0].min(p[0]), lo[1].min(p[1])];
            hi = [hi[0].max(p[0]), hi[1].max(p[1])];
        }
        let bbox = [
            (lo[0] - pad).max(0.0),
            (lo[1] - pad).max(0.0),
            (hi[0] + pad).min(wf),
            (hi[1] + pad).min(hf),
        ];
        if bbox[2] <= bbox[0] || bbox[3] <= bbox[1] {
            continue;
        }
        // Rings go to the GPU one after another, a sentinel between: the
        // shader closes each ring on its own start when it meets one.
        let mut pts: Vec<[f32; 2]> = Vec::with_capacity(total);
        for (k, r) in rings.iter().enumerate() {
            if k > 0 {
                pts.push(RING_BREAK);
            }
            pts.extend_from_slice(r);
        }
        let first = dl.verts.len() as u32;
        let count = pts.len() as u32;
        // The ramp, sampled after the outline in the same buffer.
        let ramp: Vec<[f32; 2]> = if graded && has_fill {
            let r = st.grad.ramp(st.fill);
            let working = crate::color::Colorspace::default();
            (0..GEO_GRAD_SAMPLES)
                .flat_map(|k| {
                    let c = premul(r.sample(k as f32 / (GEO_GRAD_SAMPLES - 1) as f32, working));
                    [[c[0], c[1]], [c[2], c[3]]]
                })
                .collect()
        } else {
            Vec::new()
        };
        let emit = {
            let base = if has_fill { st.fill } else { st.stroke };
            let k = if st.emit.strength.is_finite() { st.emit.strength.max(0.0) } else { 0.0 };
            [0, 1, 2].map(|c| base[c] * st.emit.tint[c] * k)
        };
        let proto = PrimGpu {
            bbox,
            fill: if has_fill { fill } else { [0.0; 4] },
            stroke: if has_stroke { stroke } else { [0.0; 4] },
            first,
            count,
            kind,
            rule: match st.rule {
                FillRule::NonZero => 0,
                FillRule::EvenOdd => 1,
            },
            width: if has_stroke { width } else { 0.0 },
            radius,
            blend: pr.layer.blend.index(),
            _pad: 0.0,
            grad_first: 0,
            grad_n: 0,
            // What it gives off, rg here and b + "stands in light's way"
            // in `_pad2` (the lit raster's second target, `geolight`).
            _pad3: [emit[0], emit[1]],
            grad_ab: [st.grad.a[0] * wf, st.grad.a[1] * wf, st.grad.b[0] * wf, st.grad.b[1] * wf],
            grad_kind: if has_fill && !ramp.is_empty() { st.grad.kind as u32 } else { 0 },
            grad_depth: st.grad.depth.max(1e-3),
            _pad2: [emit[2], if st.emit.clear && st.emit.strength <= 0.0 { 0.0 } else { 1.0 }],
        };
        // Tiles come out of what is left after every later prim has kept
        // its one quad; a disc is O(1) per pixel already.
        let reserve = g.prims.len() - i - 1;
        let budget = (MAX_INSTANCES - dl.prims.len()).saturating_sub(reserve);
        // A gradient from the edges needs the distance at every pixel: the
        // flat interior tiles skip exactly that walk.
        let edge = proto.grad_kind == GradKind::Edge as u32;
        let start = dl.prims.len();
        match if kind == 2 || edge { None } else { tile(&proto, &rings, has_fill, st.rule, pad, budget) } {
            Some(tiles) => dl.prims.extend(tiles),
            None => dl.prims.push(proto),
        }
        dl.verts.extend(pts);
        if !ramp.is_empty() {
            let at = dl.verts.len() as u32;
            for p in &mut dl.prims[start..] {
                p.grad_first = at;
                p.grad_n = GEO_GRAD_SAMPLES as u32;
            }
            dl.verts.extend(ramp);
        }
    }
    dl
}

/// The layer's cache key: the bytes the GPU will draw, plus the canvas.
pub fn hash(dl: &DrawList, w: u32, h: u32) -> u64 {
    crate::diskcache::content_hash(&[
        bytemuck::cast_slice(&dl.prims),
        bytemuck::cast_slice(&dl.verts),
        &w.to_le_bytes(),
        &h.to_le_bytes(),
    ])
}

/// What the panel reports about an evaluation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GeoStats {
    pub prims: usize,
    pub verts: usize,
    pub points: usize,
    /// Did any cap bite?
    pub truncated: bool,
}

/// Counts of a geometry. `truncated` here means it is OVER a cap — what
/// `flatten` would trim. The caps `eval` applied on the way are reported
/// by [`eval_stats`]: a geometry sitting exactly on a cap is legal (a
/// Scatter asked for 4096 and got them) and looks the same either way.
pub fn stats(g: &Geometry) -> GeoStats {
    let verts: usize = g.prims.iter().map(|p| p.verts.len()).sum();
    GeoStats {
        prims: g.prims.len(),
        verts,
        points: g.points.len(),
        truncated: g.prims.len() > MAX_PRIMS
            || g.points.len() > MAX_POINTS
            || verts > MAX_TOTAL_VERTS
            || g.prims.iter().any(|p| p.verts.len() > MAX_VERTS_PER_PRIM),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(lo: [f32; 2], hi: [f32; 2], style: Style) -> Geometry {
        let mut g = Geometry::default();
        for p in [[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]] {
            g.points.push(Point::at(p));
        }
        g.prims.push(Prim { layer: Default::default(), kind: PrimKind::Poly { closed: true }, verts: vec![0, 1, 2, 3], style, phys: None, src: 0, rings: Vec::new() });
        g
    }

    fn text_node(fx: crate::textfx::GeoTextFx) -> GeoNode {
        GeoNode::Text {
            text: "IIII".into(),
            font: Default::default(),
            size: 0.1,
            position: [0.2, 0.3],
            align: crate::geotext::TextAlign::Left,
            tracking: 0.2,
            leading: 1.2,
            rotation: 0.0,
            style: Style { fill: [1.0; 4], ..Default::default() },
            fx,
        }
    }

    fn eval_text(node: GeoNode, path: Option<GeoNode>) -> Geometry {
        let mut g = GeoGraph::default();
        g.nodes.insert(1, node);
        g.nodes.insert(2, GeoNode::Output);
        g.edges.insert((2, 0), 1);
        if let Some(p) = path {
            g.nodes.insert(3, p);
            g.edges.insert((1, 5), 3);
        }
        g.next_id = 4;
        let cx = EvalCx { tick: 0.0, field: None, seed: 0.0, aspect: 0.5, px_per_unit: 1000.0 };
        eval(&g, &cx)
    }

    /// A String on the words socket writes the Text; the socket is never a
    /// number, so channel 6 is still the path offset; and a String is no
    /// shape, so a dangling one never becomes the drawing.
    #[test]
    fn a_string_on_the_words_socket_writes_the_text() {
        let mut g = GeoGraph::default();
        g.nodes.insert(1, text_node(Default::default()));
        g.nodes.insert(2, GeoNode::Output);
        g.nodes.insert(3, GeoNode::String { text: "II".into() });
        g.edges.insert((2, 0), 1);
        g.edges.insert((1, TEXT_WORDS_INPUT), 3);
        g.next_id = 4;
        let cx = EvalCx { tick: 0.0, field: None, seed: 0.0, aspect: 0.5, px_per_unit: 1000.0 };
        assert_eq!(eval(&g, &cx).prims.len(), 2, "two glyphs off the String, not the node's four");
        g.edges.remove(&(1, TEXT_WORDS_INPUT));
        assert_eq!(eval(&g, &cx).prims.len(), 4, "unwired, the node's own text");
        let t = &g.nodes[&1];
        assert_eq!(t.inputs()[TEXT_WORDS_INPUT].kind, PortKind::Text);
        assert_eq!(t.channel_name(TEXT_PATH_OFFSET), "path offset");
        assert_eq!(g.nodes[&3].output_kind(), PortKind::Text);
        assert!(PortKind::Text.can_feed(t.inputs()[TEXT_WORDS_INPUT].kind));
        // Out of the graph's own pick: nothing but the String and a Text
        // with no Output, and the Text is what it draws.
        g.nodes.remove(&2);
        g.edges.clear();
        g.out = None;
        assert_eq!(eval(&g, &cx).prims.len(), 4);
    }

    /// With nothing in `fx` and no path, the old layout — and through it, an
    /// animator lifts only the glyphs its range selects, in geo units.
    #[test]
    fn a_2dini_text_takes_animators_and_keeps_its_old_layout_without() {
        let plain = eval_text(text_node(Default::default()), None);
        assert_eq!(plain.prims.len(), 4);
        let ys = |g: &Geometry| (0..4).map(|i| g.centroid(i)[1]).collect::<Vec<_>>();
        // An animator that does nothing goes through the new path and lands
        // where the old one did.
        let idle = crate::textfx::GeoTextFx { animators: vec![crate::textfx::TextAnimator { amount: 0.0, ..Default::default() }], ..Default::default() };
        let same = eval_text(text_node(idle), None);
        for i in 0..4 {
            let (a, b) = (plain.centroid(i), same.centroid(i));
            assert!((a[0] - b[0]).abs() < 2e-3 && (a[1] - b[1]).abs() < 2e-3, "glyph {i}: {a:?} vs {b:?}");
        }
        // 50 px up on the first half: 0.05 geo units at 1000 px a unit.
        let lift = crate::textfx::GeoTextFx {
            animators: vec![crate::textfx::TextAnimator { end: 50.0, position: [0.0, -50.0], ..Default::default() }],
            ..Default::default()
        };
        let up = eval_text(text_node(lift), None);
        let (a, b) = (ys(&plain), ys(&up));
        for i in 0..4 {
            let want = if i < 2 { -0.05 } else { 0.0 };
            assert!((b[i] - a[i] - want).abs() < 1e-3, "glyph {i}: {} -> {}", a[i], b[i]);
        }
        // Its numbers are channels past the sockets.
        let mut n = text_node(crate::textfx::GeoTextFx { animators: vec![Default::default()], ..Default::default() });
        let i = text_anim_index(0, crate::textfx::ANIM_PARAMS.iter().position(|p| *p == "rotation").unwrap());
        assert!(n.set_literal(i, 30.0));
        assert_eq!(n.literal(i), Some(30.0));
        assert!(n.set_literal(TEXT_PATH_OFFSET, 0.25));
        assert_eq!(n.literal(TEXT_PATH_OFFSET), Some(0.25));
        assert_eq!(n.inputs()[5].name, "path", "the path socket follows the five scalars");
    }

    /// The standalone Text's styling, as geometry: a shadow, an outline
    /// ring and a weight are copies of every glyph UNDER the faces, in
    /// that order, and each style number is a channel.
    #[test]
    fn a_2dini_text_wears_the_text_nodes_styling() {
        use crate::textfx::GeoTextFx;
        let styled = GeoTextFx {
            shadow: true,
            shadow_offset: [20.0, 0.0],
            outline: 0.05,
            outline_color: [0.0, 1.0, 0.0, 1.0],
            weight: 0.02,
            ..Default::default()
        };
        let g = eval_text(text_node(styled), None);
        assert_eq!(g.prims.len(), 16, "shadow, ring, weight and face for four glyphs");
        let (shadow, ring, bold, face) = (&g.prims[0].style, &g.prims[4].style, &g.prims[8].style, &g.prims[12].style);
        assert_eq!(shadow.fill, [0.0, 0.0, 0.0, 0.6]);
        assert_eq!(ring.stroke, [0.0, 1.0, 0.0, 1.0]);
        assert!(ring.width > bold.width && bold.width > 0.0, "the ring reaches past the weight");
        assert_eq!(face.fill, [1.0; 4]);
        // 20 px right at 1000 px a unit.
        let dx = g.centroid(0)[0] - g.centroid(12)[0];
        assert!((dx - 0.02).abs() < 1e-4, "shadow offset {dx}");
        // Italic leans the glyphs, anchor moves the block.
        let lean = eval_text(text_node(GeoTextFx { skew: 15.0, ..Default::default() }), None);
        let plain = eval_text(text_node(Default::default()), None);
        assert_ne!(lean.bbox(), plain.bbox());
        let top = eval_text(text_node(GeoTextFx { valign: crate::textfx::VAlign::Top, ..Default::default() }), None);
        assert!(top.bbox().unwrap().0[1] > plain.bbox().unwrap().0[1], "top-anchored text hangs below its position");
        // Channels.
        let mut n = text_node(GeoTextFx::default());
        for (k, name) in crate::textfx::GEO_STYLE_PARAMS.iter().enumerate() {
            let i = text_style_index(k);
            assert!(n.set_literal(i, 0.25 + k as f32), "{name}");
            assert_eq!(n.literal(i), Some(0.25 + k as f32), "{name}");
            assert_eq!(n.channel_name(i), *name);
        }
    }

    /// A spline wired into the path socket: the glyphs march along it.
    #[test]
    fn a_2dini_text_follows_a_spline_wired_into_its_path() {
        let pt = |p: [f32; 2]| crate::graph::SplinePoint::new(p, 0.0);
        let line = GeoNode::Spline {
            points: vec![pt([0.5, 0.05]), pt([0.5, 0.45])],
            mode: crate::graph::SplineMode::Linear,
            closed: false,
            style: Style::default(),
        };
        let g = eval_text(text_node(Default::default()), Some(line));
        assert_eq!(g.prims.len(), 4);
        let c: Vec<[f32; 2]> = (0..4).map(|i| g.centroid(i)).collect();
        for w in c.windows(2) {
            assert!(w[1][1] > w[0][1] + 0.01, "down the spline: {c:?}");
        }
        assert!(c.iter().all(|p| (p[0] - 0.5).abs() < 0.05), "on the line: {c:?}");
    }

    #[test]
    fn contains_judges_each_closed_poly_alone_under_its_rule() {
        let g = square([0.2, 0.2], [0.6, 0.6], Style::default());
        assert!(g.contains([0.4, 0.4]));
        assert!(!g.contains([0.1, 0.4]));
        assert!(!g.contains([0.4, 0.7]));
        // Reversed winding is still inside.
        let mut r = g.clone();
        r.prims[0].verts.reverse();
        assert!(r.contains([0.4, 0.4]));
        // ... and merged over the original it stays inside: the shader
        // fills each prim on its own, so the windings never cancel.
        let mut both = g.clone();
        both.merge(r);
        assert!(both.contains([0.4, 0.4]));
        // An open poly has no inside.
        let mut o = g.clone();
        o.prims[0].kind = PrimKind::Poly { closed: false };
        assert!(!o.contains([0.4, 0.4]));
        // The pentagram's centre is winding 2: filled non-zero, a hole
        // even-odd, exactly as it draws.
        let mut star = eval(&chain(vec![polygon(5, 2)]), &cx());
        assert!(star.contains([0.5, 0.5]));
        star.prims[0].style.rule = FillRule::EvenOdd;
        assert!(!star.contains([0.5, 0.5]));
        assert!(star.contains([0.5, 0.32]), "the tip is winding 1 under both rules");
    }

    #[test]
    fn scalar_fan_in_is_linear_and_a_scalar_cycle_terminates() {
        // Slider -> 40 Max nodes each reading the previous on BOTH sockets:
        // 2^40 evaluations without the memo, 40 with it.
        let mut g = chain(vec![GeoNode::Scatter { count: 50, seed: 0.0, density: 1.0 }]);
        g.nodes.insert(2, GeoNode::Slider { value: 0.75, name: "d".into() });
        let mut prev = 2;
        for i in 0..40 {
            let id = 10 + i;
            g.nodes.insert(id, GeoNode::Math { op: MathOp::Max, a: 0.0, b: 0.0 });
            g.edges.insert((id, 0), prev);
            g.edges.insert((id, 1), prev);
            prev = id;
        }
        g.edges.insert((1, 1), prev);
        assert!((eval_scalar(&g, prev, [0.1, 0.1], &cx(), 0) - 0.75).abs() < 1e-6);
        let geo = eval(&g, &cx());
        assert_eq!(geo.points.len(), 50);
        // A Math reading itself on both sockets: 0, not a hang.
        let mut c = GeoGraph::default();
        c.nodes.insert(1, GeoNode::Math { op: MathOp::Add, a: 1.0, b: 1.0 });
        c.edges.insert((1, 0), 1);
        c.edges.insert((1, 1), 1);
        assert_eq!(eval_scalar(&c, 1, [0.0, 0.0], &cx(), 0), 0.0);
        // Memo is per POINT: a FieldIn read twice at two points differs.
        let mut f = GeoGraph::default();
        f.nodes.insert(1, GeoNode::FieldIn { shift: 0.0 });
        f.nodes.insert(2, GeoNode::Math { op: MathOp::Add, a: 0.0, b: 0.0 });
        f.edges.insert((2, 0), 1);
        f.edges.insert((2, 1), 1);
        let fx = coord_x_field();
        let cx = EvalCx { tick: 0.0, field: Some(&fx), seed: 0.0, aspect: 1.0, px_per_unit: 100.0 };
        assert!((eval_scalar(&f, 2, [0.25, 0.0], &cx, 0) - 0.5).abs() < 1e-5);
        assert!((eval_scalar(&f, 2, [0.4, 0.0], &cx, 0) - 0.8).abs() < 1e-5);
    }

    #[test]
    fn merge_reindexes_the_second_geometry() {
        let mut a = square([0.0, 0.0], [0.1, 0.1], Style::default());
        let b = square([0.5, 0.5], [0.6, 0.6], Style::default());
        a.merge(b);
        assert_eq!(a.points.len(), 8);
        assert_eq!(a.prims[1].verts, vec![4, 5, 6, 7]);
        assert_eq!(a.polyline(1)[0], [0.5, 0.5]);
    }

    #[test]
    fn flatten_scales_both_axes_by_width_and_premultiplies() {
        let st = Style { fill: [1.0, 0.0, 0.0, 0.5], ..Default::default() };
        let g = square([0.25, 0.25], [0.5, 0.5], st);
        let dl = flatten(&g, 200, 100);
        assert_eq!(dl.prims.len(), 1);
        assert_eq!(dl.verts[0], [50.0, 50.0]);
        assert_eq!(dl.verts[2], [100.0, 100.0]);
        assert_eq!(dl.prims[0].fill, [0.5, 0.0, 0.0, 0.5]);
        assert_eq!(dl.prims[0].kind, 1);
        // Padded by the AA ramp and clamped to the canvas (height 100).
        let b = dl.prims[0].bbox;
        assert!(b[0] < 50.0 && b[1] < 50.0 && b[2] > 100.0);
        assert_eq!(b[3], 100.0);
    }

    #[test]
    fn flatten_drops_what_cannot_be_seen() {
        // Invisible style.
        let mut g = square([0.1, 0.1], [0.3, 0.3], Style { fill: [1.0, 1.0, 1.0, 0.0], ..Default::default() });
        assert!(flatten(&g, 100, 100).prims.is_empty());
        // Off canvas.
        g.prims[0].style = Style::default();
        for q in &mut g.points {
            q.p[0] += 5.0;
        }
        assert!(flatten(&g, 100, 100).prims.is_empty());
        // A fill on an open poly is nothing; a stroke on it is something.
        let mut o = square([0.1, 0.1], [0.3, 0.3], Style::default());
        o.prims[0].kind = PrimKind::Poly { closed: false };
        assert!(flatten(&o, 100, 100).prims.is_empty());
        o.prims[0].style.stroke = [1.0; 4];
        o.prims[0].style.width = 2.0;
        let dl = flatten(&o, 100, 100);
        assert_eq!(dl.prims.len(), 1);
        assert_eq!(dl.prims[0].kind, 0);
        assert_eq!(dl.prims[0].fill, [0.0; 4]);
    }

    #[test]
    fn a_point_prim_becomes_a_disc_of_its_pscale() {
        let mut g = Geometry::default();
        g.points.push(Point { p: [0.5, 0.25], pscale: 7.0, ..Default::default() });
        g.prims.push(Prim { layer: Default::default(), kind: PrimKind::Point, verts: vec![0], style: Style::default(), phys: None, src: 0, rings: Vec::new() });
        let dl = flatten(&g, 100, 100);
        assert_eq!(dl.prims[0].kind, 2);
        assert_eq!(dl.prims[0].radius, 7.0);
        assert_eq!(dl.verts, vec![[50.0, 25.0]]);
    }

    #[test]
    fn eval_node_shows_one_nodes_geometry_and_a_still_noise_has_no_phase() {
        let mut g = GeoGraph::default();
        g.nodes.insert(1, GeoNode::default());
        g.nodes.insert(2, GeoNode::Scatter { count: 10, seed: 0.0, density: 1.0 });
        g.edges.insert((2, 0), 1);
        g.nodes.insert(3, GeoNode::Output);
        g.edges.insert((3, 0), 2);
        let cx = EvalCx { tick: 0.0, field: None, seed: 0.0, aspect: 1.0, px_per_unit: 100.0 };
        // The polygon on its own, the scatter on its own, and the output.
        assert_eq!(eval_node(&g, 1, &cx).prims.len(), 1);
        assert_eq!(eval_node(&g, 2, &cx).prims.len(), 10);
        assert_eq!(eval_node(&g, 3, &cx), eval(&g, &cx));
        let both = eval_nodes(&g, &[2, 3], &cx);
        assert_eq!(both[0], eval_node(&g, 2, &cx));
        assert_eq!(both[1], eval_node(&g, 3, &cx));
        // A scalar shows nothing; a missing node too.
        g.nodes.insert(4, GeoNode::Slider { value: 1.0, name: String::new() });
        assert!(eval_node(&g, 4, &cx).prims.is_empty());
        assert!(eval_node(&g, 99, &cx).prims.is_empty());

        let mut np = crate::graph::NoiseParams { time: 7.0, speed: 2.0, ..Default::default() };
        np.graph.nodes.insert(1, FieldNode::Coord { kind: crate::field::CoordKind::X, scale: 1.0 });
        assert_eq!(field_cx_for(&np).phase, 0.0, "a still field has no phase");
        np.graph.nodes.insert(2, FieldNode::default());
        assert!((field_cx_for(&np).phase - 1.4).abs() < 1e-5, "a moving field carries time * speed * the unit");
        np.speed_unit = 1.0;
        assert_eq!(field_cx_for(&np).phase, 14.0, "a legacy node keeps its old rate");
        np.scale = 0.0;
        assert!(field_cx_for(&np).scale > 0.0, "scale is clamped like the shader's");
    }

    /// A Stroke is a LINE: open, stroked in its colour at its width, never
    /// filled — and its outline is the Spline's open one exactly.
    #[test]
    fn a_stroke_is_an_open_unfilled_line_on_the_splines_outline() {
        let pts = vec![
            SplinePoint::new([0.1, 0.1], 0.0),
            SplinePoint::new([0.5, 0.3], 0.0),
            SplinePoint::new([0.9, 0.1], 0.0),
        ];
        let mut g = GeoGraph::default();
        g.nodes.insert(
            1,
            GeoNode::Stroke { points: pts.clone(), mode: SplineMode::CatmullRom, color: [1.0, 0.5, 0.0, 1.0], width: 4.0 },
        );
        g.nodes.insert(2, GeoNode::Slider { value: 9.0, name: String::new() });
        let cx = EvalCx { tick: 0.0, field: None, seed: 0.0, aspect: 1.0, px_per_unit: 100.0 };
        let geo = eval_node(&g, 1, &cx);
        assert_eq!(geo.prims.len(), 1);
        let pr = &geo.prims[0];
        assert_eq!(pr.kind, PrimKind::Poly { closed: false });
        assert_eq!(pr.style.fill[3], 0.0, "a stroke is never filled");
        assert_eq!(pr.style.stroke, [1.0, 0.5, 0.0, 1.0]);
        assert_eq!(pr.style.width, 4.0);
        let pl = geo.polyline(0);
        assert_eq!(pl.first().copied(), Some([0.1, 0.1]));
        assert_eq!(pl.last().copied(), Some([0.9, 0.1]));
        let want: Vec<[f32; 2]> =
            spline_outline(&pts, SplineMode::CatmullRom, false).into_iter().map(|(p, _)| p).collect();
        assert_eq!(pl, want);
        // The width is a socket.
        g.edges.insert((1, 0), 2);
        assert_eq!(eval_node(&g, 1, &cx).prims[0].style.width, 9.0);
        // It draws: flattened, a kind-0 (open) prim with its width.
        let dl = flatten(&eval(&g, &cx), 200, 200);
        assert!(dl.prims.iter().all(|p| p.kind == 0 && p.width == 9.0 && p.fill == [0.0; 4]));
        // One point is not a line.
        g.nodes.insert(1, GeoNode::Stroke { points: pts[..1].to_vec(), mode: SplineMode::Linear, color: [1.0; 4], width: 2.0 });
        assert!(eval_node(&g, 1, &cx).prims.is_empty());
    }

    /// Per-point tension on a Catmull-Rom curve: 0 is the curve it always
    /// was (and a hair above 0 is a hair away from it — the Hermite form is
    /// the same curve), 1 pulls the curve onto the straight legs, below 0
    /// bulges it further out. Measured on a closed square: how far the
    /// outline strays outside the square's own edges.
    #[test]
    fn tension_tightens_and_loosens_a_catmull_rom_curve() {
        let square = |t: f32| -> Vec<SplinePoint> {
            [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]
                .iter()
                .map(|p| SplinePoint { tension: t, ..SplinePoint::new(*p, 0.0) })
                .collect()
        };
        let outline = |t: f32| spline_outline(&square(t), SplineMode::CatmullRom, true);
        // How far outside [0.2, 0.8]^2 the curve reaches.
        let bulge = |pl: &[([f32; 2], f32)]| {
            pl.iter()
                .map(|(p, _)| {
                    let dx = (0.2 - p[0]).max(p[0] - 0.8).max(0.0);
                    let dy = (0.2 - p[1]).max(p[1] - 0.8).max(0.0);
                    dx.max(dy)
                })
                .fold(0.0f32, f32::max)
        };
        let (b0, b_eps, b1, b_neg) =
            (bulge(&outline(0.0)), bulge(&outline(1e-6)), bulge(&outline(1.0)), bulge(&outline(-0.5)));
        assert!(b0 > 0.01, "an untensioned Catmull-Rom square bulges ({b0})");
        assert!((b0 - b_eps).abs() < 1e-4, "tension is continuous at 0: {b0} vs {b_eps}");
        assert!(b1 < 1e-5, "tension 1 is a corner: the curve stays on the legs ({b1})");
        assert!(b_neg > b0, "negative tension is rounder ({b_neg} vs {b0})");
        // Every point is still passed through, whatever the tension.
        for t in [0.0, 0.5, 1.0, -0.5] {
            let pl = outline(t);
            for q in square(t) {
                assert!(
                    pl.iter().any(|(p, _)| (p[0] - q.p[0]).abs() < 1e-5 && (p[1] - q.p[1]).abs() < 1e-5),
                    "tension {t} lost the point {:?}",
                    q.p
                );
            }
        }
        // It is per point: tension on ONE corner leaves the opposite
        // corner's bulge alone.
        let mut one = square(0.0);
        one[0].tension = 1.0;
        let pl = spline_outline(&one, SplineMode::CatmullRom, true);
        let near = |pl: &[([f32; 2], f32)], c: [f32; 2]| {
            pl.iter()
                .filter(|(p, _)| (p[0] - c[0]).abs() < 0.15 && (p[1] - c[1]).abs() < 0.15)
                .map(|(p, _)| (0.2 - p[0]).max(p[0] - 0.8).max(0.2 - p[1]).max(p[1] - 0.8).max(0.0))
                .fold(0.0f32, f32::max)
        };
        let base = outline(0.0);
        assert!(near(&pl, [0.2, 0.2]) < near(&base, [0.2, 0.2]), "the tensioned corner did not tighten");
        assert!((near(&pl, [0.8, 0.8]) - near(&base, [0.8, 0.8])).abs() < 1e-6, "the far corner moved");
        // The other modes ignore it.
        for mode in [SplineMode::Linear, SplineMode::BSpline, SplineMode::Bezier] {
            assert_eq!(
                spline_outline(&square(1.0), mode, true),
                spline_outline(&square(0.0), mode, true),
                "{mode:?} read tension"
            );
        }
    }

    /// A setup saved before tension existed loads with none.
    #[cfg(feature = "serde")]
    #[test]
    fn a_point_saved_without_tension_loads_untensioned() {
        let q: SplinePoint =
            serde_json::from_str(r#"{"p":[0.5,0.5],"feather":2.0}"#).unwrap();
        assert_eq!(q.tension, 0.0);
        let back: SplinePoint = serde_json::from_str(&serde_json::to_string(&SplinePoint {
            tension: 0.7,
            ..q
        })
        .unwrap())
        .unwrap();
        assert_eq!(back.tension, 0.7);
    }

    /// A Noise node inside the drawing is the comp's Noise node: its own
    /// field graph, sampled at the point with the same scale/seed/phase rule
    /// — and it drives geometry, moving only when it has a speed.
    #[test]
    fn a_noise_subnode_reads_its_own_graph_and_drives_geometry() {
        let mut fg = FieldGraph::default();
        fg.nodes.insert(1, FieldNode::default());
        let noise = GeoNode::Noise { graph: fg.clone(), scale: 1.5, seed: 3.0, speed: 0.0 };
        assert!(noise.is_scalar());
        assert!(noise.inputs().is_empty());
        assert_eq!(noise.channel_count(), 3);
        assert_eq!(noise.channel_name(1), "seed");
        let mut g = GeoGraph::default();
        g.nodes.insert(1, noise);
        let cx = EvalCx { tick: 12.0, field: None, seed: 2.0, aspect: 1.0, px_per_unit: 100.0 };
        for p in [[0.1, 0.2], [0.6, 0.4]] {
            let want = crate::field::eval_cpu(&fg, [p[0] * 1.5, p[1] * 1.5], 5.0, 0.0, 1.0);
            assert!((eval_scalar(&g, 1, p, &cx, 0) - want).abs() < 1e-6);
        }
        // Wired into a Displace, it moves points by different amounts.
        g.nodes.insert(2, GeoNode::Scatter { count: 40, seed: 0.0, density: 1.0 });
        g.nodes.insert(3, GeoNode::Displace { mode: WarpMode::Offset, amount: 0.2, x: 0.5, y: 0.5 });
        g.edges.insert((3, 0), 2);
        g.edges.insert((3, 1), 1);
        g.nodes.insert(4, GeoNode::Output);
        g.edges.insert((4, 0), 3);
        let before = eval_node(&g, 2, &cx);
        let after = eval(&g, &cx);
        let dx: Vec<f32> = before.points.iter().zip(&after.points).map(|(a, b)| b.p[0] - a.p[0]).collect();
        assert!(dx.windows(2).any(|w| (w[0] - w[1]).abs() > 1e-4), "not a displace: {dx:?}");
        // Still without speed; the tick moves it once it has one.
        let mut p = crate::graph::GeoParams { graph: g, time: 7.0, ..Default::default() };
        assert!(!noise_moves(&p.graph));
        assert_eq!(geo_time_key(&p), 0.0);
        if let Some(GeoNode::Noise { speed, .. }) = p.graph.nodes.get_mut(&1) {
            *speed = 0.1;
        }
        assert!(noise_moves(&p.graph));
        assert_eq!(geo_time_key(&p), 7.0);
        let at = |t: f32| eval(&p.graph, &EvalCx { tick: t, ..cx });
        assert_ne!(at(0.0), at(7.0), "a moving noise did not move the drawing");
    }

    /// A skinned bar bends with its skeleton: a point by the second bone
    /// swings round its joint when that bone turns.
    #[test]
    fn a_skin_bends_shapes_with_the_skeleton() {
        use crate::georig::Bone;
        let cx = EvalCx { tick: 0.0, field: None, seed: 0.0, aspect: 1.0, px_per_unit: 100.0 };
        let mut g = GeoGraph::default();
        g.nodes.insert(1, GeoNode::Rectangle { center: [0.4, 0.2], size: [0.4, 0.04], corner: 0.0, rotation: 0.0, style: Style::default() });
        g.nodes.insert(2, GeoNode::Skeleton {
            bones: vec![
                Bone { name: String::new(), parent: None, head: [0.2, 0.2], tail: [0.4, 0.2] },
                Bone { name: String::new(), parent: Some(0), head: [0.4, 0.2], tail: [0.6, 0.2] },
            ],
            pose: vec![0.0, std::f32::consts::FRAC_PI_2],
        });
        g.nodes.insert(3, GeoNode::Skin { smooth: false, falloff: 2.0 });
        g.edges.insert((3, 0), 1);
        g.edges.insert((3, 1), 2);
        let bent = eval_node(&g, 3, &cx);
        // The right end of the bar (0.6, 0.2±0.02) swings down to x ≈ 0.4.
        let far = bent.points.iter().map(|q| q.p[1]).fold(f32::MIN, f32::max);
        assert!(far > 0.35, "the far end went down: {far}");
        let skel = eval_node(&g, 2, &cx);
        assert_eq!(skel.bones.len(), 2);
        assert!(skel.prims.iter().all(|p| p.style.fill[3] == 0.0 && p.style.stroke[3] == 0.0), "bones never render");
    }

    /// Particles: born at the rate, gone after their life, a falling one
    /// lower each frame, and the same frame drawn twice is the same.
    #[test]
    fn particles_are_born_live_and_fall_by_the_frame() {
        let mk = |t: f32| {
            let mut g = GeoGraph::default();
            g.nodes.insert(1, GeoNode::Particles {
                rate: 2.0, life: 10.0, life_var: 0.0, speed: 0.0, speed_var: 0.0, direction: 0.0, spread: 0.0,
                gravity: 1.0, wind: 0.0, drag: 0.0, size: 4.0, size_end: 0.0, fade: true, start: 0.0, seed: 0.0,
            });
            eval_node(&g, 1, &EvalCx { tick: t, field: None, seed: 0.0, aspect: 0.5, px_per_unit: 100.0 })
        };
        // At frame 5: births at 0, 0.5, ... 5 = 11 particles.
        assert_eq!(mk(5.0).points.len(), 11);
        // At frame 30, only the last 10 frames' worth: 20 (and frame 30's).
        let late = mk(30.0);
        assert!((20..=21).contains(&late.points.len()), "{}", late.points.len());
        // The oldest has fallen 0.5 g a² px: a = 9.5 frames -> 45 px.
        let a = mk(9.5);
        let first = a.points.iter().max_by(|x, y| x.p[1].total_cmp(&y.p[1])).unwrap();
        assert!((first.p[1] - (0.25 + 45.125 / 100.0)).abs() < 1e-4, "{:?}", first.p);
        assert!(first.pscale < 0.3 && first.cd[3] < 0.06, "shrinks and fades over its life");
        assert_eq!(mk(7.0), mk(7.0));
        assert!(mk(-1.0).points.is_empty());
        // Drag: the closed form against a fine step of v' = g - k v.
        let (d, _) = ballistic([3.0, -2.0], [0.5, 1.0], 0.2, 7.0);
        let (mut x, mut v) = ([0.0f32; 2], [3.0f32, -2.0]);
        let dt = 1e-3;
        for _ in 0..7000 {
            for c in 0..2 {
                let acc = [0.5, 1.0][c] - 0.2 * v[c];
                x[c] += v[c] * dt;
                v[c] += acc * dt;
            }
        }
        assert!((d[0] - x[0]).abs() < 0.02 && (d[1] - x[1]).abs() < 0.02, "{d:?} vs {x:?}");
    }

    /// An Edit moves exactly the vertices it names, by their offsets, and
    /// a key the input no longer has moves nothing.
    #[test]
    fn an_edit_moves_the_named_vertices_and_skips_orphans() {
        let cx = EvalCx { tick: 0.0, field: None, seed: 0.0, aspect: 1.0, px_per_unit: 100.0 };
        let mut g = GeoGraph::default();
        g.nodes.insert(1, GeoNode::Polygon { center: [0.5, 0.5], radius: 0.1, sides: 4, step: 1, rotation: 0.0, style: Style::default() });
        g.nodes.insert(2, GeoNode::Edit {
            edits: vec![
                PointEdit { prim: 0, vert: 1, offset: [0.05, 0.0] },
                PointEdit { prim: 0, vert: 9, offset: [1.0, 1.0] },
                PointEdit { prim: 3, vert: 0, offset: [1.0, 1.0] },
            ],
        });
        g.edges.insert((2, 0), 1);
        let a = eval_node(&g, 1, &cx);
        let b = eval_node(&g, 2, &cx);
        for (i, (p, q)) in a.points.iter().zip(&b.points).enumerate() {
            let d = [q.p[0] - p.p[0], q.p[1] - p.p[1]];
            let want = if i == a.prims[0].verts[1] as usize { [0.05, 0.0] } else { [0.0, 0.0] };
            assert!((d[0] - want[0]).abs() < 1e-6 && (d[1] - want[1]).abs() < 1e-6, "point {i} moved {d:?}");
        }
        let GeoNode::Edit { edits } = &g.nodes[&2] else { unreachable!() };
        assert_eq!(orphan_edits(&a, edits), 2);
    }

    /// A brush stroke is a ribbon as wide as it says along the path, and a
    /// tapered one is thinner at its ends than in its middle.
    #[test]
    fn a_brush_stroke_is_a_ribbon_along_the_path() {
        let cx = EvalCx { tick: 0.0, field: None, seed: 0.0, aspect: 1.0, px_per_unit: 1000.0 };
        let pts: Vec<[f32; 3]> = (0..=20).map(|i| [0.2 + 0.03 * i as f32, 0.5, 1.0]).collect();
        let mut g = GeoGraph::default();
        let brush = |taper: f32| GeoNode::Brush {
            strokes: vec![BrushStroke { pts: pts.clone() }],
            color: [1.0; 4],
            width: 20.0,
            taper,
            smooth: 0.0,
        };
        g.nodes.insert(1, brush(0.0));
        let geo = eval_node(&g, 1, &cx);
        assert_eq!(geo.prims.len(), 1);
        let (lo, hi) = geo.bbox().unwrap();
        assert!(((hi[1] - lo[1]) - 0.02).abs() < 1e-4, "20 px wide at 1000 px a unit: {lo:?} {hi:?}");
        assert!(geo.contains([0.5, 0.5]) && !geo.contains([0.5, 0.52]));
        g.nodes.insert(1, brush(1.0));
        let geo = eval_node(&g, 1, &cx);
        assert!(geo.contains([0.5, 0.509]), "the middle keeps its width");
        assert!(!geo.contains([0.21, 0.508]), "a tapered end is thin");
        assert!(geo.pivot.is_some());
    }

    /// A Circle is round and a Rectangle is its size, rounded corners and
    /// all; both hand on their centre as the pivot.
    #[test]
    fn circle_and_rectangle_are_their_shapes() {
        let cx = EvalCx { tick: 0.0, field: None, seed: 0.0, aspect: 1.0, px_per_unit: 1000.0 };
        let mut g = GeoGraph::default();
        g.nodes.insert(1, GeoNode::Circle { center: [0.4, 0.3], radius: 0.1, style: Style::default() });
        let c = eval_node(&g, 1, &cx);
        assert!(c.points.len() >= 24);
        for q in &c.points {
            let r = ((q.p[0] - 0.4).powi(2) + (q.p[1] - 0.3).powi(2)).sqrt();
            assert!((r - 0.1).abs() < 1e-5);
        }
        assert_eq!(c.pivot, Some([0.4, 0.3]));
        assert!(c.contains([0.4, 0.3]) && !c.contains([0.52, 0.3]));
        let rect = |corner: f32, rotation: f32| {
            let mut g = GeoGraph::default();
            g.nodes.insert(1, GeoNode::Rectangle {
                center: [0.5, 0.5], size: [0.4, 0.2], corner, rotation, style: Style::default(),
            });
            eval_node(&g, 1, &cx)
        };
        let sharp = rect(0.0, 0.0);
        assert_eq!(sharp.points.len(), 4);
        assert_eq!(sharp.bbox(), Some(([0.3, 0.4], [0.7, 0.6])));
        assert_eq!(sharp.pivot, Some([0.5, 0.5]));
        let round = rect(0.05, 0.0);
        assert!(round.points.len() > 8);
        let (lo, hi) = round.bbox().unwrap();
        assert!((lo[0] - 0.3).abs() < 1e-5 && (hi[1] - 0.6).abs() < 1e-5, "rounding keeps the size");
        assert!(!round.contains([0.301, 0.401]), "the corner is cut");
        let turned = rect(0.0, std::f32::consts::FRAC_PI_2);
        let (lo, hi) = turned.bbox().unwrap();
        assert!((hi[0] - lo[0] - 0.2).abs() < 1e-5 && (hi[1] - lo[1] - 0.4).abs() < 1e-5);
    }

    /// A Transform with `inherit` turns about the shape's own centre, and
    /// hands its moved pivot to the next one; without it, about `pivot`.
    #[test]
    fn a_transform_inherits_the_generators_pivot() {
        let cx = EvalCx { tick: 0.0, field: None, seed: 0.0, aspect: 1.0, px_per_unit: 100.0 };
        let mut g = GeoGraph::default();
        g.nodes.insert(1, GeoNode::Star { center: [0.6, 0.4], outer: 0.1, inner: 0.05, points: 5, rotation: 0.0, style: Style::default() });
        let xf = |inherit: bool, translate: [f32; 2]| GeoNode::Transform {
            translate, rotate: 1.0, scale: [2.0, 2.0], pivot: [0.0, 0.0], inherit,
        };
        g.nodes.insert(2, xf(true, [0.1, 0.0]));
        g.edges.insert((2, 0), 1);
        let a = eval_node(&g, 2, &cx);
        let pv = a.pivot.unwrap();
        assert!((pv[0] - 0.7).abs() < 1e-5 && (pv[1] - 0.4).abs() < 1e-5, "the pivot moves with the translate");
        // Turned and scaled about the centre: the centre only translates.
        let star = eval_node(&g, 1, &cx);
        let c = |geo: &Geometry| {
            let n = geo.points.len() as f32;
            let s = geo.points.iter().fold([0.0f32; 2], |s, q| [s[0] + q.p[0], s[1] + q.p[1]]);
            [s[0] / n, s[1] / n]
        };
        let (c0, c1) = (c(&star), c(&a));
        assert!((c1[0] - c0[0] - 0.1).abs() < 1e-5 && (c1[1] - c0[1]).abs() < 1e-5);
        // A second inheriting Transform turns about where the first left it.
        g.nodes.insert(3, xf(true, [0.0, 0.0]));
        g.edges.insert((3, 0), 2);
        let b = eval_node(&g, 3, &cx);
        let cb = c(&b);
        assert!((cb[0] - c1[0]).abs() < 1e-5 && (cb[1] - c1[1]).abs() < 1e-5);
        // Old behaviour: about the origin.
        g.nodes.insert(2, xf(false, [0.1, 0.0]));
        let o = eval_node(&g, 2, &cx);
        assert!((c(&o)[0] - c1[0]).abs() > 0.1, "an absolute pivot at the origin swings the star away");
        // A merge has no single pivot: the bounds stand in.
        g.nodes.insert(4, GeoNode::Merge { count: 2 });
        g.edges.insert((4, 0), 1);
        assert_eq!(eval_node(&g, 4, &cx).pivot, None);
    }

    /// Merge takes any number of inputs, in order, and grows a free socket
    /// whenever its last one is wired.
    #[test]
    fn merge_takes_n_inputs_and_grows() {
        let mut g = GeoGraph::default();
        for (i, x) in [0.1f32, 0.3, 0.5].iter().enumerate() {
            g.nodes.insert(i as NodeId + 1, GeoNode::Polygon {
                center: [*x, 0.2], radius: 0.05, sides: 4, step: 1, rotation: 0.0, style: Style::default(),
            });
        }
        g.nodes.insert(10, GeoNode::Merge { count: 2 });
        g.edges.insert((10, 0), 1);
        assert!(!grow_merges(&mut g), "a free last socket needs no growing");
        g.edges.insert((10, 1), 2);
        assert!(grow_merges(&mut g));
        assert_eq!(g.nodes[&10].inputs().len(), 3);
        g.edges.insert((10, 2), 3);
        assert!(grow_merges(&mut g));
        assert_eq!(g.nodes[&10].inputs().len(), 4);
        let cx = EvalCx { tick: 0.0, field: None, seed: 0.0, aspect: 1.0, px_per_unit: 100.0 };
        let geo = eval_node(&g, 10, &cx);
        assert_eq!(geo.prims.len(), 3, "every wired input is merged");
        // Draw order is input order.
        assert!(geo.centroid(0)[0] < geo.centroid(1)[0] && geo.centroid(1)[0] < geo.centroid(2)[0]);
    }

    /// The preview view wins over the Output node, but only for geometry.
    #[test]
    fn a_preview_view_shows_any_geometry_node() {
        let mut g = GeoGraph::default();
        g.nodes.insert(1, GeoNode::default());
        g.nodes.insert(2, GeoNode::Output);
        g.nodes.insert(3, GeoNode::Slider { value: 1.0, name: String::new() });
        assert_eq!(g.output(), Some(2));
        g.view = Some(1);
        assert_eq!(g.output(), Some(1));
        g.view = Some(3);
        assert_eq!(g.output(), Some(2), "a scalar cannot be viewed as geometry");
    }

    /// Every shape knows the node that made it — a copy keeps its maker's —
    /// and a click picks the topmost shape under it.
    #[test]
    fn shapes_know_their_maker_and_a_click_picks_one() {
        let mut g = GeoGraph::default();
        g.nodes.insert(1, GeoNode::Polygon { center: [0.3, 0.3], radius: 0.1, sides: 4, step: 1, rotation: 0.0, style: Style::default() });
        g.nodes.insert(2, GeoNode::Star { center: [0.35, 0.3], outer: 0.1, inner: 0.05, points: 5, rotation: 0.0, style: Style::default() });
        g.nodes.insert(3, GeoNode::Transform { translate: [0.1, 0.0], rotate: 0.0, scale: [1.0, 1.0], pivot: [0.0, 0.0], inherit: false });
        g.edges.insert((3, 0), 2);
        g.nodes.insert(4, GeoNode::Merge { count: 2 });
        g.edges.insert((4, 0), 1);
        g.edges.insert((4, 1), 3);
        let cx = EvalCx { tick: 0.0, field: None, seed: 0.0, aspect: 1.0, px_per_unit: 100.0 };
        let geo = eval_node(&g, 4, &cx);
        assert_eq!(geo.prims[0].src, 1, "the polygon is the Polygon's");
        assert_eq!(geo.prims[1].src, 2, "a moved star is still the Star's");
        // The star (on top) wins where both overlap; the diamond alone elsewhere.
        assert_eq!(pick(&geo, [0.45, 0.3], 0.005).map(|i| geo.prims[i].src), Some(2));
        assert_eq!(pick(&geo, [0.25, 0.3], 0.005).map(|i| geo.prims[i].src), Some(1));
        assert_eq!(pick(&geo, [0.9, 0.9], 0.005), None);
    }

    /// Normals point out of a closed shape whichever way it winds.
    #[test]
    fn normals_point_out_of_closed_shapes() {
        let mut g = square([0.2, 0.2], [0.6, 0.6], Style::default());
        let check = |g: &Geometry| {
            let pl = g.polyline(0);
            let c = g.centroid(0);
            for (p, n) in pl.iter().zip(g.normals(0)) {
                let out = [p[0] - c[0], p[1] - c[1]];
                assert!(out[0] * n[0] + out[1] * n[1] > 0.0, "normal {n:?} at {p:?} points in");
                assert!(((n[0] * n[0] + n[1] * n[1]).sqrt() - 1.0).abs() < 1e-5);
            }
        };
        check(&g);
        g.prims[0].verts.reverse();
        check(&g);
        g.points.push(Point::at([0.9, 0.9]));
        g.prims.push(Prim { layer: Default::default(), kind: PrimKind::Point, verts: vec![4], style: Style::default(), phys: None, src: 0, rings: Vec::new() });
        assert!(g.normals(1).is_empty());
    }

    #[test]
    fn edge_tests_is_area_times_segments() {
        let g = square([0.25, 0.25], [0.5, 0.5], Style::default());
        let dl = flatten(&g, 200, 100);
        // One quad, four segments: its padded area four times over.
        let b = dl.prims[0].bbox;
        let area = ((b[2] - b[0]) * (b[3] - b[1])) as u64;
        assert_eq!(dl.edge_tests(), area * 4);
        assert_eq!(DrawList::default().edge_tests(), 0);
    }

    #[test]
    fn hash_is_content_not_identity() {
        let g = square([0.1, 0.1], [0.3, 0.3], Style::default());
        let a = hash(&flatten(&g, 100, 100), 100, 100);
        let b = hash(&flatten(&g.clone(), 100, 100), 100, 100);
        assert_eq!(a, b);
        assert_ne!(a, hash(&flatten(&g, 200, 100), 200, 100));
    }

    #[test]
    fn exposed_mirrors_the_field_graph_contract() {
        let mut g = GeoGraph::default();
        g.nodes.insert(1, GeoNode::Slider { value: 0.2, name: "size".into() });
        g.nodes.insert(2, GeoNode::Input { name: "gain".into(), value: 0.0 });
        g.nodes.insert(3, GeoNode::Slider { value: 0.9, name: String::new() });
        g.nodes.insert(4, GeoNode::Slider { value: 0.7, name: "size".into() });
        assert_eq!(
            g.exposed(),
            vec![("size".to_string(), false), ("gain".to_string(), true), ("slider".to_string(), false)]
        );
        assert_eq!(g.exposed_value("size"), Some(0.2));
        assert!(g.set_exposed("size", 0.5));
        assert_eq!(g.nodes[&4], GeoNode::Slider { value: 0.5, name: "size".into() });
        assert!(!g.set_exposed("nope", 1.0));
    }

    #[test]
    fn every_channel_round_trips_through_its_literal() {
        let kinds = vec![
            GeoNode::default(),
            GeoNode::Star { center: [0.5, 0.5], outer: 0.2, inner: 0.1, points: 5, rotation: 0.0, style: Style::default() },
            GeoNode::Resample { mode: ResampleMode::Spacing, spacing: 8.0, count: 32 },
            GeoNode::Scatter { count: 100, seed: 0.0, density: 1.0 },
            GeoNode::Displace { mode: WarpMode::Offset, amount: 0.1, x: 0.5, y: 0.5 },
            GeoNode::Displace { mode: WarpMode::Curl, amount: 0.1, x: 0.5, y: 0.5 },
            GeoNode::Instance { scale: 1.0, rotation: 0.0, inherit_cd: true, pivot: Pivot::BboxCentre },
            GeoNode::Transform { translate: [0.0, 0.0], rotate: 0.0, scale: [1.0, 1.0], pivot: [0.0, 0.0], inherit: false },
            GeoNode::Merge { count: 2 },
            GeoNode::Stroke { points: Vec::new(), mode: SplineMode::Linear, color: [1.0; 4], width: 3.0 },
            GeoNode::Text {
                text: "hi".into(),
                font: Default::default(),
                size: 0.1,
                position: [0.1, 0.3],
                align: Default::default(),
                tracking: 0.0,
                leading: 1.2,
                rotation: 0.0,
                style: Style::default(),
                fx: Default::default(),
            },
            GeoNode::Style { style: Style::default() },
            GeoNode::Output,
            GeoNode::Slider { value: 0.0, name: String::new() },
            GeoNode::Input { name: "x".into(), value: 0.0 },
            GeoNode::FieldIn { shift: 0.0 },
            GeoNode::Math { op: MathOp::Add, a: 0.0, b: 0.0 },
            GeoNode::Remap { in_lo: 0.0, in_hi: 1.0, out_lo: 0.0, out_hi: 1.0, clamp: true },
        ];
        for mut k in kinds {
            let inputs = k.inputs();
            for i in 0..k.channel_count() {
                let geo = inputs.get(i).is_some_and(|s| s.kind == PortKind::Geo);
                let has = k.literal(i).is_some();
                // Geometry sockets never carry a number; a scalar socket
                // does unless it is a pure input (a Remap's value).
                assert!(!(geo && has), "{} channel {i}", k.name());
                let v = 0.125 + i as f32;
                let wrote = k.set_literal(i, v);
                assert_eq!(wrote, has, "{} channel {i}", k.name());
                if has {
                    assert_eq!(k.literal(i), Some(v), "{} channel {i}", k.name());
                }
            }
            // Past the end: nothing.
            let n = k.channel_count();
            assert_eq!(k.literal(n), None, "{}", k.name());
            assert!(!k.set_literal(n, 1.0));
        }
    }

    #[test]
    fn output_prefers_the_marker_and_skips_scalars() {
        let mut g = GeoGraph::default();
        g.nodes.insert(1, GeoNode::default());
        g.nodes.insert(2, GeoNode::Slider { value: 1.0, name: String::new() });
        // The last unconsumed GEOMETRY, not the slider.
        assert_eq!(g.output(), Some(1));
        g.nodes.insert(3, GeoNode::Output);
        g.edges.insert((3, 0), 1);
        assert_eq!(g.output(), Some(3));
        assert!(!g.has_field_in());
        g.nodes.insert(4, GeoNode::FieldIn { shift: 0.0 });
        assert!(g.has_field_in());
    }

    #[test]
    fn scalars_chain_and_a_field_in_reads_the_noise() {
        let mut g = GeoGraph::default();
        g.nodes.insert(1, GeoNode::FieldIn { shift: 0.0 });
        g.nodes.insert(
            2,
            GeoNode::Remap { in_lo: 0.0, in_hi: 1.0, out_lo: 10.0, out_hi: 20.0, clamp: true },
        );
        g.edges.insert((2, 0), 1);
        let no_field = EvalCx { tick: 0.0, field: None, seed: 0.0, aspect: 1.0, px_per_unit: 100.0 };
        // Unwired field reads 0.5 -> remapped to 15.
        assert!((eval_scalar(&g, 2, [0.3, 0.3], &no_field, 0) - 15.0).abs() < 1e-5);
        // A Coord X field reads the x coordinate scaled by the noise scale.
        let mut fg = FieldGraph::default();
        fg.nodes.insert(1, FieldNode::Coord { kind: crate::field::CoordKind::X, scale: 1.0 });
        let f = FieldCx { graph: fg, scale: 2.0, seed: 0.0, phase: 0.0 };
        let cx = EvalCx { tick: 0.0, field: Some(&f), seed: 0.0, aspect: 1.0, px_per_unit: 100.0 };
        let v = eval_scalar(&g, 2, [0.25, 0.0], &cx, 0);
        assert!((v - 15.0).abs() < 1e-4, "{v}");
        // Math over two sockets.
        g.nodes.insert(3, GeoNode::Math { op: MathOp::Mul, a: 0.0, b: 3.0 });
        g.edges.insert((3, 0), 2);
        assert!((eval_scalar(&g, 3, [0.25, 0.0], &cx, 0) - 45.0).abs() < 1e-3);
    }

    #[test]
    fn a_still_field_has_no_phase() {
        let mut fg = FieldGraph::default();
        fg.nodes.insert(1, FieldNode::default());
        assert!(field_moves(&fg));
        if let Some(FieldNode::Fbm { speed, .. }) = fg.nodes.get_mut(&1) {
            *speed = 0.0;
        }
        assert!(!field_moves(&fg));
    }

    // --- evaluator ---------------------------------------------------------

    fn cx() -> EvalCx<'static> {
        EvalCx { tick: 0.0, field: None, seed: 0.0, aspect: 1.0, px_per_unit: 100.0 }
    }

    /// A one-node graph, or a chain: each node takes the previous on socket 0.
    fn chain(nodes: Vec<GeoNode>) -> GeoGraph {
        let mut g = GeoGraph::default();
        for (i, n) in nodes.into_iter().enumerate() {
            let id = i as NodeId + 1;
            g.nodes.insert(id, n);
            if i > 0 {
                g.edges.insert((id, 0), id - 1);
            }
        }
        g
    }

    fn polygon(sides: u32, step: u32) -> GeoNode {
        GeoNode::Polygon { center: [0.5, 0.5], radius: 0.2, sides, step, rotation: 0.0, style: Style::default() }
    }

    fn coord_x_field() -> FieldCx {
        let mut fg = FieldGraph::default();
        fg.nodes.insert(1, FieldNode::Coord { kind: crate::field::CoordKind::X, scale: 1.0 });
        FieldCx { graph: fg, scale: 1.0, seed: 0.0, phase: 0.0 }
    }

    #[test]
    fn polygon_and_star_vertex_counts_and_step_order() {
        let geo = eval(&chain(vec![polygon(6, 1)]), &cx());
        assert_eq!(geo.points.len(), 6);
        assert_eq!(geo.prims[0].verts, vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(geo.prims[0].kind, PrimKind::Poly { closed: true });
        // First vertex straight up: y is down in geo space.
        let top = geo.points[0].p;
        assert!((top[0] - 0.5).abs() < 1e-5 && (top[1] - 0.3).abs() < 1e-5, "{top:?}");
        // {5/2} is the pentagram.
        let geo = eval(&chain(vec![polygon(5, 2)]), &cx());
        assert_eq!(geo.prims[0].verts, vec![0, 2, 4, 1, 3]);
        // Clamps: too few sides, too big a step.
        let geo = eval(&chain(vec![polygon(1, 9)]), &cx());
        assert_eq!(geo.points.len(), 3);
        assert_eq!(geo.prims[0].verts, vec![0, 2, 1]);
        let star = GeoNode::Star { center: [0.5, 0.5], outer: 0.2, inner: 0.1, points: 5, rotation: 0.0, style: Style::default() };
        let geo = eval(&chain(vec![star]), &cx());
        assert_eq!(geo.points.len(), 10);
        let r = |i: usize| dist(geo.points[i].p, [0.5, 0.5]);
        assert!((r(0) - 0.2).abs() < 1e-5 && (r(1) - 0.1).abs() < 1e-5 && (r(2) - 0.2).abs() < 1e-5);
    }

    #[test]
    fn spline_is_one_poly_and_needs_two_points() {
        let pt = |x: f32, y: f32| SplinePoint { p: [x, y], feather: 0.0, bias: 0.0, h_in: [0.0; 2], h_out: [0.0; 2], tension: 0.0 };
        let mk = |pts: Vec<SplinePoint>, closed: bool| GeoNode::Spline {
            points: pts,
            mode: SplineMode::Linear,
            closed,
            style: Style { fill: [0.0, 1.0, 0.0, 1.0], ..Default::default() },
        };
        assert!(eval(&chain(vec![mk(vec![pt(0.1, 0.1)], true)]), &cx()).prims.is_empty());
        let l = vec![pt(0.1, 0.1), pt(0.5, 0.1), pt(0.5, 0.5)];
        let geo = eval(&chain(vec![mk(l.clone(), false)]), &cx());
        assert_eq!(geo.prims.len(), 1);
        assert_eq!(geo.prims[0].kind, PrimKind::Poly { closed: false });
        assert!(geo.points.len() >= 3);
        assert_eq!(geo.prims[0].style.fill, [0.0, 1.0, 0.0, 1.0]);
        assert!(geo.points.iter().enumerate().all(|(i, q)| q.id == i as u32));
        // Open means an L, not a triangle: every sample is on one of the
        // two legs, it starts on the first point and ENDS on the last.
        let on_legs = |p: [f32; 2]| {
            ((p[1] - 0.1).abs() < 1e-5 && (0.1 - 1e-5..=0.5 + 1e-5).contains(&p[0]))
                || ((p[0] - 0.5).abs() < 1e-5 && (0.1 - 1e-5..=0.5 + 1e-5).contains(&p[1]))
        };
        assert!(geo.points.iter().all(|q| on_legs(q.p)), "{:?}", geo.points.iter().map(|q| q.p).collect::<Vec<_>>());
        assert!(dist(geo.points[0].p, [0.1, 0.1]) < 1e-6);
        assert!(dist(geo.points.last().unwrap().p, [0.5, 0.5]) < 1e-6);
        // Closed keeps the return leg.
        let ring = eval(&chain(vec![mk(l.clone(), true)]), &cx());
        assert!(ring.points.iter().any(|q| !on_legs(q.p)));
        // A smooth open curve clamps at its ends rather than bending toward
        // the far end: same endpoints, and nothing left of the first point.
        let mut curve = mk(l, false);
        if let GeoNode::Spline { mode, .. } = &mut curve {
            *mode = SplineMode::CatmullRom;
        }
        let geo = eval(&chain(vec![curve]), &cx());
        assert!(dist(geo.points[0].p, [0.1, 0.1]) < 1e-6);
        assert!(dist(geo.points.last().unwrap().p, [0.5, 0.5]) < 1e-6);
        assert!(geo.points.iter().all(|q| q.p[0] >= 0.1 - 1e-4));
        // Two points are a line.
        let line = eval(&chain(vec![mk(vec![pt(0.1, 0.1), pt(0.5, 0.1)], false)]), &cx());
        assert_eq!(line.prims.len(), 1);
        assert!(line.points.iter().all(|q| (q.p[1] - 0.1).abs() < 1e-6));
        assert!(dist(line.points.last().unwrap().p, [0.5, 0.1]) < 1e-6);
    }

    #[test]
    fn resample_spaces_evenly_keeps_closedness_and_interpolates_cd() {
        let mut g = chain(vec![
            polygon(4, 1),
            GeoNode::Resample { mode: ResampleMode::Count, spacing: 8.0, count: 40 },
        ]);
        let geo = eval(&g, &cx());
        assert_eq!(geo.points.len(), 40);
        assert_eq!(geo.prims[0].kind, PrimKind::Poly { closed: true });
        let pl = geo.polyline(0);
        let lens: Vec<f32> = (0..pl.len()).map(|i| dist(pl[i], pl[(i + 1) % pl.len()])).collect();
        let (lo, hi) = lens.iter().fold((f32::MAX, f32::MIN), |(l, h), &x| (l.min(x), h.max(x)));
        assert!(hi - lo < 1e-3, "{lo} {hi}");
        // No seam duplicate.
        assert!(dist(pl[0], pl[pl.len() - 1]) > 1e-3);
        // Spacing mode: perimeter is 4 * 0.2 * sqrt2 units = ~113 px; 10 px apart -> 11.
        g.nodes.insert(2, GeoNode::Resample { mode: ResampleMode::Spacing, spacing: 10.0, count: 0 });
        let geo = eval(&g, &cx());
        assert_eq!(geo.points.len(), 11);
        // An open line keeps its endpoints and interpolates cd along it.
        let mut src = Geometry::default();
        src.points.push(Point { p: [0.0, 0.0], cd: [0.0, 0.0, 0.0, 1.0], ..Default::default() });
        src.points.push(Point { p: [1.0, 0.0], cd: [1.0, 1.0, 1.0, 1.0], pscale: 3.0, ..Default::default() });
        src.prims.push(Prim { layer: Default::default(), kind: PrimKind::Poly { closed: false }, verts: vec![0, 1], style: Style::default(), phys: None, src: 0, rings: Vec::new() });
        src.points.push(Point { p: [0.7, 0.7], ..Default::default() });
        src.prims.push(Prim { layer: Default::default(), kind: PrimKind::Point, verts: vec![2], style: Style::default(), phys: None, src: 0, rings: Vec::new() });
        let out = resample(&src, |_| 5);
        assert_eq!(out.prims[0].kind, PrimKind::Poly { closed: false });
        let pl = out.polyline(0);
        assert_eq!(pl.len(), 5);
        assert_eq!(pl[0], [0.0, 0.0]);
        assert!((pl[4][0] - 1.0).abs() < 1e-6);
        assert!((out.points[2].cd[0] - 0.5).abs() < 1e-5);
        assert!((out.points[2].pscale - 2.0).abs() < 1e-5);
        assert_eq!(out.points[1].orient, 0.0);
        assert_eq!(out.points[3].id, 3);
        // The point prim rode through, reindexed.
        assert_eq!(out.prims[1].kind, PrimKind::Point);
        assert_eq!(out.polyline(1), vec![[0.7, 0.7]]);
    }

    #[test]
    fn scatter_is_deterministic_inside_prefix_stable_and_reseeds() {
        let mk = |count: u32, seed: f32| chain(vec![polygon(6, 1), GeoNode::Scatter { count, seed, density: 1.0 }]);
        let a = eval(&mk(100, 0.0), &cx());
        let b = eval(&mk(100, 0.0), &cx());
        assert_eq!(a, b);
        assert_eq!(a.points.len(), 100);
        assert!(a.prims.iter().all(|p| p.kind == PrimKind::Point));
        let region = eval(&chain(vec![polygon(6, 1)]), &cx());
        assert!(a.points.iter().all(|q| region.contains(q.p)));
        let half = eval(&mk(50, 0.0), &cx());
        assert_eq!(half.points, a.points[..50]);
        let other = eval(&mk(100, 3.0), &cx());
        assert_ne!(other.points, a.points);
        // Density 0 places nothing; unwired region fills the canvas.
        let mut none = mk(100, 0.0);
        none.nodes.insert(2, GeoNode::Scatter { count: 100, seed: 0.0, density: 0.0 });
        assert!(eval(&none, &cx()).points.is_empty());
        let canvas = eval(&chain(vec![GeoNode::Scatter { count: 200, seed: 0.0, density: 1.0 }]), &cx());
        assert_eq!(canvas.points.len(), 200);
        assert!(canvas.points.iter().any(|q| q.p[0] > 0.9) && canvas.points.iter().any(|q| q.p[0] < 0.1));
    }

    #[test]
    fn scatter_density_follows_the_field() {
        // Coord X through a steep Remap: density 0 left of 0.4, 1 right of 0.6.
        let mut g = chain(vec![GeoNode::Scatter { count: 400, seed: 0.0, density: 1.0 }]);
        g.nodes.insert(2, GeoNode::FieldIn { shift: 0.0 });
        g.nodes.insert(3, GeoNode::Remap { in_lo: 0.4, in_hi: 0.6, out_lo: 0.0, out_hi: 1.0, clamp: true });
        g.edges.insert((3, 0), 2);
        g.edges.insert((1, 1), 3);
        let f = coord_x_field();
        let cx = EvalCx { tick: 0.0, field: Some(&f), seed: 0.0, aspect: 1.0, px_per_unit: 100.0 };
        let geo = eval(&g, &cx);
        assert_eq!(geo.points.len(), 400);
        let right = geo.points.iter().filter(|q| q.p[0] > 0.5).count();
        assert!(right * 100 >= 85 * 400, "{right}");
    }

    #[test]
    fn displace_offsets_and_curls() {
        // Offset by sliders: (0.8 - 0.5, 0.5 - 0.5) * 0.1 = (0.03, 0).
        let mut g = chain(vec![polygon(4, 1), GeoNode::Displace { mode: WarpMode::Offset, amount: 0.1, x: 0.8, y: 0.5 }]);
        let before = eval(&chain(vec![polygon(4, 1)]), &cx());
        let after = eval(&g, &cx());
        for (a, b) in before.points.iter().zip(&after.points) {
            assert!((b.p[0] - a.p[0] - 0.03).abs() < 1e-6 && (b.p[1] - a.p[1]).abs() < 1e-6);
        }
        // Curl of the linear potential x is (0, -1): every point moves down by amount.
        g.nodes.insert(2, GeoNode::Displace { mode: WarpMode::Curl, amount: 0.05, x: 0.0, y: 0.0 });
        g.nodes.insert(3, GeoNode::FieldIn { shift: 0.0 });
        g.edges.insert((2, 1), 3);
        let f = coord_x_field();
        let cx = EvalCx { tick: 0.0, field: Some(&f), seed: 0.0, aspect: 1.0, px_per_unit: 100.0 };
        let after = eval(&g, &cx);
        for (a, b) in before.points.iter().zip(&after.points) {
            assert!((b.p[0] - a.p[0]).abs() < 1e-4 && (b.p[1] - a.p[1] + 0.05).abs() < 1e-4, "{:?} {:?}", a.p, b.p);
        }
    }

    #[test]
    fn field_ins_with_different_shifts_decorrelate() {
        let mut fg = FieldGraph::default();
        fg.nodes.insert(1, FieldNode::default());
        let f = FieldCx { graph: fg, scale: 4.0, seed: 0.0, phase: 0.0 };
        let cx = EvalCx { tick: 0.0, field: Some(&f), seed: 0.0, aspect: 1.0, px_per_unit: 100.0 };
        let mut g = GeoGraph::default();
        g.nodes.insert(1, GeoNode::FieldIn { shift: 0.0 });
        g.nodes.insert(2, GeoNode::FieldIn { shift: 1.7 });
        let p = [0.31, 0.62];
        assert_ne!(eval_scalar(&g, 1, p, &cx, 0), eval_scalar(&g, 2, p, &cx, 0));
        assert_eq!(eval_scalar(&g, 1, p, &cx, 0), eval_scalar(&g, 1, p, &cx, 0));
    }

    #[test]
    fn instance_copies_per_point_in_order_about_the_pivot() {
        let mut g = chain(vec![polygon(3, 1)]);
        // Two scattered points to stamp the triangle onto.
        g.nodes.insert(2, GeoNode::Scatter { count: 2, seed: 0.0, density: 1.0 });
        g.nodes.insert(3, GeoNode::Instance { scale: 1.0, rotation: 0.0, inherit_cd: true, pivot: Pivot::BboxCentre });
        g.edges.insert((3, 0), 1);
        g.edges.insert((3, 1), 2);
        g.nodes.insert(4, GeoNode::Output);
        g.edges.insert((4, 0), 3);
        let geo = eval(&g, &cx());
        assert_eq!(geo.prims.len(), 2);
        assert_eq!(geo.points.len(), 6);
        let pts = eval(&chain(vec![GeoNode::Scatter { count: 2, seed: 0.0, density: 1.0 }]), &cx());
        // Each copy sits centred (bbox pivot) on its point, in point order.
        for (i, q) in pts.points.iter().enumerate() {
            let c = geo.centroid(i);
            let (lo, hi) = {
                let pl = geo.polyline(i);
                pl.iter().fold((pl[0], pl[0]), |(l, h), p| ([l[0].min(p[0]), l[1].min(p[1])], [h[0].max(p[0]), h[1].max(p[1])]))
            };
            let bc = [(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5];
            assert!(dist(bc, q.p) < 1e-5, "{i}: {bc:?} vs {:?} (centroid {c:?})", q.p);
            assert_eq!(geo.points[3 * i].id, fnv(q.id, 0));
        }
        // Unwired points: the template as it is.
        g.edges.remove(&(3, 1));
        assert_eq!(eval(&g, &cx()), eval(&chain(vec![polygon(3, 1)]), &cx()));
        // Origin pivot + pscale/cd on the point, via a hand-built points geometry.
        let mut tmpl = Geometry::default();
        tmpl.points.push(Point::at([0.1, 0.0]));
        tmpl.prims.push(Prim { layer: Default::default(), kind: PrimKind::Point, verts: vec![0], style: Style::default(), phys: None, src: 0, rings: Vec::new() });
        let mut at = Geometry::default();
        at.points.push(Point { p: [0.5, 0.5], pscale: 2.0, orient: std::f32::consts::FRAC_PI_2, cd: [1.0, 0.0, 0.0, 1.0], id: 7 });
        let mut walk = Walk::default();
        let mut gg = GeoGraph::default();
        gg.nodes.insert(1, GeoNode::Instance { scale: 1.5, rotation: 0.0, inherit_cd: true, pivot: Pivot::Origin });
        gg.nodes.insert(2, GeoNode::Merge { count: 2 });
        gg.nodes.insert(3, GeoNode::Merge { count: 2 });
        gg.edges.insert((1, 0), 2);
        gg.edges.insert((1, 1), 3);
        walk.memo.insert(2, tmpl);
        walk.memo.insert(3, at);
        let memo = walk.memo.clone();
        let out = eval_geo(&gg, 1, &cx(), &mut walk);
        assert_eq!(out.points.len(), 1);
        // (0.1, 0) * 3 rotated a quarter turn = (0, 0.3), about the point.
        let p = out.points[0].p;
        assert!((p[0] - 0.5).abs() < 1e-5 && (p[1] - 0.8).abs() < 1e-5, "{p:?}");
        assert!((out.points[0].pscale - 3.0).abs() < 1e-5);
        assert_eq!(out.prims[0].style.fill, [1.0, 0.0, 0.0, 1.0]);
        gg.nodes.insert(1, GeoNode::Instance { scale: 1.5, rotation: 0.0, inherit_cd: false, pivot: Pivot::Origin });
        let mut walk = Walk { memo, ..Default::default() };
        let out = eval_geo(&gg, 1, &cx(), &mut walk);
        assert_eq!(out.prims[0].style.fill, [1.0; 4]);
    }

    #[test]
    fn transform_scales_rotates_then_translates() {
        let g = chain(vec![
            polygon(4, 1),
            GeoNode::Transform { translate: [0.1, 0.2], rotate: std::f32::consts::FRAC_PI_2, scale: [2.0, 0.5], pivot: [0.0, 0.0], inherit: false },
        ]);
        let before = eval(&chain(vec![polygon(4, 1)]), &cx());
        let after = eval(&g, &cx());
        for (a, b) in before.points.iter().zip(&after.points) {
            let s = [a.p[0] * 2.0, a.p[1] * 0.5];
            let want = [0.1 - s[1], 0.2 + s[0]];
            assert!(dist(b.p, want) < 1e-5, "{:?} {:?}", b.p, want);
            assert!((b.orient - std::f32::consts::FRAC_PI_2).abs() < 1e-6);
            assert!((b.pscale - 1.25).abs() < 1e-6);
        }
    }

    #[test]
    fn style_restyles_at_the_centroid_and_a_cycle_is_empty() {
        let mut g = chain(vec![
            polygon(4, 1),
            GeoNode::Style { style: Style { fill: [0.0, 0.0, 1.0, 1.0], stroke: [1.0; 4], width: 3.0, rule: FillRule::EvenOdd, ..Default::default() } },
        ]);
        // Fill r from a FieldIn: the centroid x, 0.5.
        g.nodes.insert(3, GeoNode::FieldIn { shift: 0.0 });
        g.edges.insert((2, 1), 3);
        let f = coord_x_field();
        let cx = EvalCx { tick: 0.0, field: Some(&f), seed: 0.0, aspect: 1.0, px_per_unit: 100.0 };
        let st = eval(&g, &cx).prims[0].style;
        assert!((st.fill[0] - 0.5).abs() < 1e-5);
        assert_eq!(st.fill[2], 1.0);
        assert_eq!(st.stroke, [1.0; 4]);
        assert_eq!(st.width, 3.0);
        assert_eq!(st.rule, FillRule::EvenOdd);
        // Merge feeding itself through the Style.
        g.nodes.insert(4, GeoNode::Merge { count: 2 });
        g.edges.insert((4, 0), 2);
        g.edges.insert((4, 1), 4);
        g.edges.insert((2, 0), 4);
        g.nodes.insert(5, GeoNode::Output);
        g.edges.insert((5, 0), 4);
        assert!(eval(&g, &cx).points.is_empty());
    }

    #[test]
    fn caps_truncate_and_stats_says_so() {
        // 4096 points x a 6-gon template = 24576 points wanted.
        let mut g = chain(vec![polygon(6, 1)]);
        g.nodes.insert(2, GeoNode::Scatter { count: 100_000, seed: 0.0, density: 1.0 });
        g.nodes.insert(3, GeoNode::Instance { scale: 0.1, rotation: 0.0, inherit_cd: false, pivot: Pivot::BboxCentre });
        g.edges.insert((3, 0), 1);
        g.edges.insert((3, 1), 2);
        g.nodes.insert(4, GeoNode::Output);
        g.edges.insert((4, 0), 3);
        let (geo, st) = eval_stats(&g, &cx());
        assert!(geo.points.len() <= MAX_POINTS);
        assert!(geo.prims.len() <= MAX_PRIMS);
        assert!(st.truncated);
        assert!(geo.prims.iter().all(|p| p.verts.iter().all(|&v| (v as usize) < geo.points.len())));
        // Whole copies only: no hexagon lost a corner to the cap.
        assert!(geo.prims.iter().all(|p| p.verts.len() == 6), "a partial copy survived");
        assert_eq!(geo.prims.len(), MAX_POINTS / 6);
        // Verts per prim.
        let (geo, st) = eval_stats(&chain(vec![polygon(4, 1), GeoNode::Resample { mode: ResampleMode::Count, spacing: 1.0, count: 100_000 }]), &cx());
        assert_eq!(geo.prims[0].verts.len(), MAX_VERTS_PER_PRIM);
        assert!(st.truncated);
        let small = eval_stats(&chain(vec![polygon(6, 1)]), &cx()).1;
        assert!(!small.truncated);
        // Landing exactly on a cap is not a bite: 4096 asked, 4096 given.
        let (geo, st) = eval_stats(&chain(vec![GeoNode::Scatter { count: MAX_POINTS as u32, seed: 0.0, density: 1.0 }]), &cx());
        assert_eq!(geo.points.len(), MAX_POINTS);
        assert!(!st.truncated, "an exact-cap scatter reported as capped");
        assert!(!stats(&geo).truncated);
        // One past it is.
        let st = eval_stats(&chain(vec![GeoNode::Scatter { count: MAX_POINTS as u32 + 1, seed: 0.0, density: 1.0 }]), &cx()).1;
        assert!(st.truncated);
        // A Merge that overflows the points cap drops the straddling prim
        // rather than keeping a fragment of it.
        let mut m = Geometry::default();
        for _ in 0..(MAX_POINTS / 6 + 1) {
            m.merge(eval(&chain(vec![polygon(6, 1)]), &cx()));
        }
        let (c, bit) = cap(m);
        assert!(bit);
        assert!(c.prims.iter().all(|p| p.verts.len() == 6));
    }

    #[test]
    fn scatter_rejects_nan_density_and_falls_back_to_an_open_regions_bbox() {
        // density <- (0^-1) - (0^-1) = inf - inf = NaN: nothing, not everything.
        let mut g = chain(vec![GeoNode::Scatter { count: 50, seed: 0.0, density: 1.0 }]);
        g.nodes.insert(2, GeoNode::Math { op: MathOp::Pow, a: 0.0, b: -1.0 });
        g.nodes.insert(3, GeoNode::Math { op: MathOp::Sub, a: 0.0, b: 0.0 });
        g.edges.insert((3, 0), 2);
        g.edges.insert((3, 1), 2);
        g.edges.insert((1, 1), 3);
        assert!(eval_scalar(&g, 3, [0.0, 0.0], &cx(), 0).is_nan());
        assert!(eval(&g, &cx()).points.is_empty());
        // An open outline as the region: its bbox, not nothing.
        let pt = |x: f32, y: f32| SplinePoint { p: [x, y], feather: 0.0, bias: 0.0, h_in: [0.0; 2], h_out: [0.0; 2], tension: 0.0 };
        let open = GeoNode::Spline {
            points: vec![pt(0.2, 0.2), pt(0.4, 0.2), pt(0.4, 0.6)],
            mode: SplineMode::Linear,
            closed: false,
            style: Style::default(),
        };
        let geo = eval(&chain(vec![open, GeoNode::Scatter { count: 100, seed: 0.0, density: 1.0 }]), &cx());
        assert_eq!(geo.points.len(), 100);
        assert!(geo.points.iter().all(|q| (0.2..=0.4).contains(&q.p[0]) && (0.2..=0.6).contains(&q.p[1])));
        // The R2 sequence stays distinct all the way to the cap (f32 would
        // fold candidates onto each other past k ≈ 1000).
        let geo = eval(&chain(vec![GeoNode::Scatter { count: MAX_POINTS as u32, seed: 0.0, density: 1.0 }]), &cx());
        let mut ps: Vec<[u32; 2]> = geo.points.iter().map(|q| [q.p[0].to_bits(), q.p[1].to_bits()]).collect();
        ps.sort();
        ps.dedup();
        assert_eq!(ps.len(), MAX_POINTS, "scatter candidates collapsed onto each other");
    }

    #[test]
    fn flatten_tiles_big_shapes_and_leaves_small_ones_whole() {
        // A 6-gon that spans one tile: one instance, as before.
        let small = eval(&chain(vec![polygon(6, 1)]), &cx());
        assert_eq!(flatten(&small, 100, 100).prims.len(), 1);
        // A filled square 100..900 px on a 1000 px canvas: many instances,
        // none overlapping, interior ones flat (kind 3) and strictly
        // inside, walking ones touching the outline.
        let g = square([0.1, 0.1], [0.9, 0.9], Style { stroke: [1.0; 4], width: 4.0, ..Default::default() });
        let dl = flatten(&g, 1000, 1000);
        assert!(dl.prims.len() > 20, "{}", dl.prims.len());
        assert_eq!(dl.verts.len(), 4, "the vertex list is shared by every tile");
        let flat = dl.prims.iter().filter(|p| p.kind == 3).count();
        let walk = dl.prims.iter().filter(|p| p.kind == 1).count();
        assert!(flat > 0 && walk > 0, "{flat} flat, {walk} walking");
        for (i, a) in dl.prims.iter().enumerate() {
            assert!(a.first == 0 && (a.kind == 3 || a.count == 4));
            if a.kind == 3 {
                assert!(a.bbox[0] > 100.0 && a.bbox[1] > 100.0 && a.bbox[2] < 900.0 && a.bbox[3] < 900.0, "{:?}", a.bbox);
                assert_eq!(a.fill, [1.0; 4]);
                assert_eq!(a.stroke, [0.0; 4]);
            } else {
                // Touches the outline: not wholly inside the square minus the pad.
                let pad = 2.0 + 1.5;
                let inside = a.bbox[0] > 100.0 + pad && a.bbox[1] > 100.0 + pad && a.bbox[2] < 900.0 - pad && a.bbox[3] < 900.0 - pad;
                assert!(!inside, "a walking tile with no outline in it: {:?}", a.bbox);
            }
            for b in &dl.prims[i + 1..] {
                let apart = a.bbox[2] <= b.bbox[0] || b.bbox[2] <= a.bbox[0] || a.bbox[3] <= b.bbox[1] || b.bbox[3] <= a.bbox[1];
                assert!(apart, "tiles overlap: {:?} {:?}", a.bbox, b.bbox);
            }
        }
        // Tiles cover the whole padded square: their area is its area.
        let area: f32 = dl.prims.iter().map(|p| (p.bbox[2] - p.bbox[0]) * (p.bbox[3] - p.bbox[1])).sum();
        let want = (800.0f32 + 7.0).powi(2);
        assert!((area - want).abs() < 1.0, "{area} vs {want}");
        // Stroke only: no flat tiles, and far fewer than the full grid.
        let mut o = g.clone();
        o.prims[0].style.fill = [0.0; 4];
        let dl = flatten(&o, 1000, 1000);
        assert!(dl.prims.iter().all(|p| p.kind == 1));
        assert!(dl.prims.len() < 16 * 16);
        // A shape bigger than the canvas is all interior: flat tiles only,
        // covering exactly the canvas.
        let big = square([-1.0, -1.0], [2.0, 2.0], Style::default());
        let dl = flatten(&big, 1000, 500);
        assert!(dl.prims.iter().all(|p| p.kind == 3));
        let area: f32 = dl.prims.iter().map(|p| (p.bbox[2] - p.bbox[0]) * (p.bbox[3] - p.bbox[1])).sum();
        assert!((area - 500_000.0).abs() < 1.0, "{area}");
        // An open stroked diagonal of many short segments: only the tiles
        // along it. (One long segment's bbox IS the canvas — tiling only
        // pays when the outline is finer than the shape.)
        let mut line = Geometry::default();
        for i in 0..=32 {
            line.points.push(Point::at([i as f32 / 32.0, i as f32 / 32.0]));
        }
        line.prims.push(Prim {
            layer: Default::default(),
            kind: PrimKind::Poly { closed: false },
            verts: (0..33).collect(),
            style: Style { stroke: [1.0; 4], width: 2.0, ..Default::default() }, phys: None, src: 0, rings: Vec::new() });
        let dl = flatten(&line, 1024, 1024);
        assert!(dl.prims.iter().all(|p| p.kind == 0));
        let area: f32 = dl.prims.iter().map(|p| (p.bbox[2] - p.bbox[0]) * (p.bbox[3] - p.bbox[1])).sum();
        assert!(area < 0.3 * 1024.0 * 1024.0, "diagonal covered {area} px");
    }
}
