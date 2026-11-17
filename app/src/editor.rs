//! Bespoke node editor: nodes are geometric shapes (type = silhouette),
//! flow runs top -> down, inputs/outputs sit on colored stalks emitted from
//! the shape. Replaces egui-snarl — the visual language there (framed boxes,
//! pin rows) couldn't be bent this far.

use crate::theme;
use crate::pathpaste::paste_loads;
use egui::{Color32, Pos2, Rect, Stroke, Vec2};
use engine::graph::{
    BlurParams, CornerPinParams, CurveParams, DefocusParams,     GlowParams,
    GradeParams, GridParams, LumaKeyParams, NodeKind, PortKind, ReformatParams,
    ShuffleParams, SpeedParams, SplineParams, SlipParams, TimeRemapParams, 
    TrackerParams,
    TransformParams,
};

/// Node bounding radius and stalk metrics, in graph units.
const R: f32 = 24.0;
/// The same radius, for layout code that has to reason about footprints
/// (`crate::nodeux`). Named, because `R` says nothing on its own.
pub const NODE_R: f32 = R;
/// How far a stalk reaches past the node body. Public with `NODE_R` so a
/// suite can aim at a socket it can only see drawn.
pub const STALK: f32 = 18.0;
/// How far past the graph panel a node may sit and still be drawn: its name
/// can run this far out from it.
const CULL_PAD: f32 = 320.0;
const SOCKET_R: f32 = 5.0;

/// BORING MODE (`Prefs::graph_boring`): every node but the bars wears one
/// plain pill (the Layer bar's height, narrower), a Load is a 4:3 card
/// showing its frame, sockets sit on the edge rather than on stalks, and
/// noodles are straight lines. One app-wide switch, set per frame from
/// prefs, because the socket and wire geometry it changes is computed in
/// free functions all over this file.
const BORING_HALF: Vec2 = Vec2::new(29.9, 13.0);
/// A boring Load's card: 4:3, as wide as the pill.
const BORING_READ_HALF: Vec2 = Vec2::new(29.9, 22.425);

/// Does `kind` wear the boring pill? Not the bars (merge bars, the harp
/// pair, a multi-input bar), not a dot (a bead on a wire), not a Load (its
/// card, `tv_on`).
fn boring_shape(kind: &NodeKind) -> bool {
    !crate::breakout::is_bar(kind)
        && !multi_input(kind)
        && !matches!(
            kind,
            NodeKind::Breakout(_) | NodeKind::Recombine(_) | NodeKind::Dot { .. } | NodeKind::Read { .. }
        )
}

/// Is this a Load drawn as a picture: TV mode, or any Load in boring mode.
pub fn tv_on(n: &ENode) -> bool {
    matches!(n.kind, NodeKind::Read { .. }) && (n.tv || boring())
}

/// The half extents of a picture Load: the CRT's cabinet, or the boring card.
fn tv_half() -> Vec2 {
    if boring() { BORING_READ_HALF } else { crate::tv::TV_HALF }
}

/// Where a ray from the centre along `d` leaves the boring pill.
fn boring_edge(d: Vec2) -> Vec2 {
    let h = BORING_HALF;
    let t = (h.x / d.x.abs().max(1e-6)).min(h.y / d.y.abs().max(1e-6));
    d * t
}

/// x of slot `slot` of `n` along a boring pill's edge, splayed to fit it.
fn boring_x(pos: Pos2, slot: usize, n: usize, size: f32) -> f32 {
    if n < 2 {
        return pos.x;
    }
    let pitch = (40.0f32).min(BORING_HALF.x * 1.6 / (n - 1) as f32);
    pos.x + (slot as f32 - (n as f32 - 1.0) / 2.0) * pitch * size
}
// Per THREAD: the UI draws on one, and unit tests running side by side
// must not see each other's switch.
thread_local! {
    static BORING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub fn set_boring(on: bool) {
    BORING.with(|b| b.set(on));
}

pub fn boring() -> bool {
    BORING.with(|b| b.get())
}

/// How far a stalk reaches past the body: none in boring mode.
fn stalk() -> f32 {
    if boring() { 0.0 } else { STALK }
}

/// Centre to a resting socket, graph units at size 1.
fn socket_reach() -> f32 {
    if boring() { BORING_HALF.y } else { R + STALK }
}

/// Below this drawn radius a node is a speck: too small to read as a shape,
/// so it twinkles instead (fading in as it shrinks past it).
const SPECK_R: f32 = 6.0;

/// Hit radius for a channel nodule, in screen px.
const CHAN_HIT: f32 = 11.0;

pub use crate::doc::graph::*;

/// What a node is FOR. The add menus group by this, so a new node lands
/// somewhere findable instead of at the bottom of a list of thirty.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum NodeCat {
    /// Noise-field nodes, grouped like the 2dini's. Only ever offered
    /// INSIDE a Noise node, because a field is not something a comp graph
    /// can hold. Fields from nothing:
    FieldPattern,
    /// Shapes and mattes: Shape, Roto, the matte port.
    FieldShape,
    /// Arithmetic on fields.
    FieldMath,
    /// Moving where a field is read.
    FieldWarp,
    /// Numbers and the Output marker.
    FieldValue,
    /// 2dini geometry nodes, in five groups by what they DO. Only ever
    /// offered INSIDE a 2dini node, for the same reason. Shapes from
    /// nothing:
    GeoCreate,
    /// One shape in, the same shapes changed.
    GeoModify,
    /// Several in, one out: merges, booleans, copies onto points.
    GeoCombine,
    /// Bodies and the simulation that moves them.
    GeoSimulate,
    /// Numbers: sliders, inputs, maths, noise — and the Output marker.
    GeoValue,
    /// SDF shapes, ops and effects. Only ever offered INSIDE an SDF node.
    Sdf,
    /// Emitters, forces and renders. Only ever offered INSIDE a Particles
    /// node.
    Particles,
    Source,
    Transform,
    Colour,
    Filter,
    Matte,
    Composite,
    /// Cards through a camera.
    ThreeD,
    Time,
    Utility,
    /// Nodes that run a learned model: generative pipelines, neural keyers.
    Ai,
    /// Steps inside a GenAI Pipeline's dive: prompts, its inputs, a model
    /// call each, Output.
    GenAiStep,
    /// Inside a Roto's dive: shapes, combines, axes.
    Roto,
}

impl NodeCat {
    /// Menu order: roughly the order a comp gets built in.
    pub const ALL: [NodeCat; 24] = [
        NodeCat::FieldPattern,
        NodeCat::FieldShape,
        NodeCat::FieldMath,
        NodeCat::FieldWarp,
        NodeCat::FieldValue,
        NodeCat::GeoCreate,
        NodeCat::GeoModify,
        NodeCat::GeoCombine,
        NodeCat::GeoSimulate,
        NodeCat::GeoValue,
        NodeCat::Sdf,
        NodeCat::Particles,
        NodeCat::Source,
        NodeCat::Transform,
        NodeCat::Colour,
        NodeCat::Filter,
        NodeCat::Matte,
        NodeCat::Composite,
        NodeCat::ThreeD,
        NodeCat::Time,
        NodeCat::Utility,
        NodeCat::Ai,
        NodeCat::GenAiStep,
        NodeCat::Roto,
    ];

    pub fn name(self) -> &'static str {
        match self {
            NodeCat::FieldPattern => "patterns",
            NodeCat::FieldShape => "shapes",
            NodeCat::FieldMath => "maths",
            NodeCat::FieldWarp => "warps",
            NodeCat::FieldValue => "values",
            NodeCat::GeoCreate => "creators",
            NodeCat::GeoModify => "modifiers",
            NodeCat::GeoCombine => "combiners",
            NodeCat::GeoSimulate => "simulators",
            NodeCat::GeoValue => "values",
            NodeCat::Sdf => "sdf",
            NodeCat::Particles => "particles",
            NodeCat::Source => "source",
            NodeCat::Transform => "transform",
            NodeCat::Colour => "colour",
            NodeCat::Filter => "filter",
            NodeCat::Matte => "matte",
            NodeCat::Composite => "composite",
            NodeCat::ThreeD => "3D",
            NodeCat::Time => "time",
            NodeCat::Utility => "utility",
            NodeCat::Ai => "AI / ML",
            NodeCat::GenAiStep => "steps",
            NodeCat::Roto => "roto",
        }
    }
}

impl NodeCat {
    /// One of the noise-field groups.
    pub fn is_field(self) -> bool {
        matches!(
            self,
            NodeCat::FieldPattern | NodeCat::FieldShape | NodeCat::FieldMath | NodeCat::FieldWarp | NodeCat::FieldValue
        )
    }

    /// One of the 2dini groups.
    pub fn is_geo(self) -> bool {
        matches!(
            self,
            NodeCat::GeoCreate
                | NodeCat::GeoModify
                | NodeCat::GeoCombine
                | NodeCat::GeoSimulate
                | NodeCat::GeoValue
        )
    }
}

/// Is a category on offer in the graph being edited? A field node means
/// nothing outside a Noise node, a geometry node nothing outside a 2dini
/// node, an SDF node nothing outside an SDF node, and a comp node nothing
/// inside any of them: four worlds, one menu each.
#[cfg(test)]
pub fn offered(cat: NodeCat, in_noise: bool, in_geo: bool, in_sdf: bool, in_particles: bool) -> bool {
    offered_in(cat, in_noise, in_geo, in_sdf, in_particles, false, false)
}

/// [`offered`], with the GenAI Pipeline's dive as a sixth world and a
/// Roto's as a seventh.
pub fn offered_in(
    cat: NodeCat,
    in_noise: bool,
    in_geo: bool,
    in_sdf: bool,
    in_particles: bool,
    in_genai: bool,
    in_roto: bool,
) -> bool {
    match cat {
        c if c.is_field() => in_noise,
        c if c.is_geo() => in_geo,
        NodeCat::Sdf => in_sdf,
        NodeCat::Particles => in_particles,
        NodeCat::GenAiStep => in_genai,
        NodeCat::Roto => in_roto,
        _ => !in_noise && !in_geo && !in_sdf && !in_particles && !in_genai && !in_roto,
    }
}

/// A fresh particle-graph node of the kind named, from the engine's
/// catalog.
fn particle_of(name: &str) -> NodeKind {
    NodeKind::ParticleNode(
        engine::nodes::particles::PNode::catalog()
            .into_iter()
            .find(|n| n.name() == name)
            .unwrap_or_default(),
    )
}

/// A fresh SDF graph node of the kind named, from the engine's catalog —
/// the add menu's constructor, so the defaults live in ONE place.
fn sdf_of(name: &str) -> NodeKind {
    NodeKind::SdfNode(
        engine::sdf::SdfNode::catalog()
            .into_iter()
            .find(|n| n.name() == name)
            .unwrap_or_default(),
    )
}

/// How far along a node is. Every node wears one: the add menus print it
/// beside the name in its colour, the graph marks it with a small dot, and
/// a project can hide everything below a threshold (`EditorGraph::
/// node_status_min`). Ordered least to most mature so the threshold is a
/// plain comparison.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Default,
    serde::Serialize, serde::Deserialize,
)]
pub enum NodeStatus {
    /// Works, may change shape, may have rough edges. Where a node starts.
    #[default]
    Alpha,
    /// Behaviour and parameters settled; being proven on real work.
    Beta,
    /// Done: the contract is stable and setups that use it are safe to keep.
    Production,
}

impl NodeStatus {
    pub const ALL: [NodeStatus; 3] =
        [NodeStatus::Alpha, NodeStatus::Beta, NodeStatus::Production];

    pub fn name(self) -> &'static str {
        match self {
            NodeStatus::Alpha => "alpha",
            NodeStatus::Beta => "beta",
            NodeStatus::Production => "production",
        }
    }

    /// The filter as a phrase: what a project set to this threshold offers.
    pub fn filter_label(self) -> &'static str {
        match self {
            NodeStatus::Alpha => "all (alpha)",
            NodeStatus::Beta => "beta and up",
            NodeStatus::Production => "production only",
        }
    }

    /// Red / lime / white, by rule: red for alpha, lime for beta, and
    /// production is plain text — the regular white, which is also why it
    /// draws no mark in the graph. Never orange: orange is banned from the
    /// UI outright (a user rule, see CLAUDE.md).
    pub fn color(self) -> Color32 {
        match self {
            NodeStatus::Production => theme::maturity_production(),
            NodeStatus::Beta => theme::maturity_beta(),
            NodeStatus::Alpha => theme::maturity_alpha(),
        }
    }

    /// The graph's mark: none for production, so a finished comp is plain
    /// and only what is still moving draws the eye.
    pub fn mark(self) -> Option<Color32> {
        (self != NodeStatus::Production).then(|| self.color())
    }

    /// Does a project set to this threshold offer a node of status `s`?
    pub fn admits(self, s: NodeStatus) -> bool {
        s >= self
    }
}

/// A node's maturity, read off its kind — ONE place, so the menu tag, the
/// graph mark and the project filter cannot disagree about a node.
///
/// THE POLICY, as of 2026-09-16: everything is alpha except Transform and
/// CornerPin, which are beta; nothing is production yet. Graduating a node
/// is a deliberate edit here and to `catalog_tests` (which pins the policy).
/// Plugins, tools and controls (`Custom`) are alpha.
pub fn status_of(kind: &NodeKind) -> NodeStatus {
    match kind {
        NodeKind::Transform(_) | NodeKind::CornerPin(_) => NodeStatus::Beta,
        _ => NodeStatus::Alpha,
    }
}

/// A menu entry's text: the name in its status colour, and no word — the
/// colour IS the label, the same one the graph's mark uses. Production is
/// plain text, exactly as it draws no mark in the graph.
pub fn status_text(name: &str, status: NodeStatus) -> egui::RichText {
    let text = egui::RichText::new(name);
    match status.mark() {
        Some(c) => text.color(c),
        None => text,
    }
}

/// How many matches the Tab search lists. Past this, three more letters
/// beat a scroll.
pub const SEARCH_MAX: usize = 8;

/// Score `name` against a FUZZY query: `q` (lowercase, no spaces) has to
/// appear in the name as a SUBSEQUENCE — its letters in order, not
/// necessarily together — so "chsl" finds ChannelSelect and "grd" finds
/// Grade. `None` when it does not appear at all.
///
/// Higher is tighter, and the three things worth a bonus are the three a
/// hand actually aims at: a letter that STARTS a word (the capital in a
/// CamelCase name, or a letter after a space), a run that stays TOGETHER,
/// and a match that starts EARLY. It is a full DP rather than a greedy
/// walk because greedy takes the first `a` in "Saturation" and then cannot
/// see that the second one led somewhere better.
fn fuzzy_score(name: &str, q: &str) -> Option<i32> {
    const MATCH: i32 = 16;
    const BOUNDARY: i32 = 14;
    const CONTIG: i32 = 10;
    const GAP: i32 = 2;
    const LEAD: i32 = 1;
    let n: Vec<char> = name.chars().collect();
    let q: Vec<char> = q.chars().map(|c| c.to_ascii_lowercase()).collect();
    if q.is_empty() || q.len() > n.len() {
        return None;
    }
    let low: Vec<char> = n.iter().map(|c| c.to_ascii_lowercase()).collect();
    let boundary: Vec<bool> = (0..n.len())
        .map(|i| {
            i == 0
                || !n[i - 1].is_alphanumeric()
                || (n[i].is_uppercase() && !n[i - 1].is_uppercase())
                || (n[i].is_numeric() && !n[i - 1].is_numeric())
        })
        .collect();
    // `best[i]` = the best score for the query SO FAR with its last letter
    // sitting at name index i; `None` where no such alignment exists.
    let mut best: Vec<Option<i32>> = vec![None; n.len()];
    for (j, qc) in q.iter().enumerate() {
        let mut next: Vec<Option<i32>> = vec![None; n.len()];
        for i in 0..n.len() {
            if low[i] != *qc {
                continue;
            }
            let here = MATCH + if boundary[i] { BOUNDARY } else { 0 };
            next[i] = if j == 0 {
                Some(here - LEAD * i as i32)
            } else {
                (0..i)
                    .filter_map(|k| {
                        let gap = (i - k - 1) as i32;
                        Some(best[k]? + here + if gap == 0 { CONTIG } else { -GAP * gap })
                    })
                    .max()
            };
        }
        best = next;
    }
    best.into_iter().flatten().max()
}

/// How much a node's ADD COUNT lifts it inside a search tier: log-shaped,
/// so the first few adds matter and the hundredth does not, and capped
/// below what two word-start letters are worth, so habit reorders close
/// matches but cannot drag a scattered one past a tight one.
fn use_bonus(uses: u32) -> i32 {
    ((8.0 * (1.0 + uses as f32).log2()) as i32).min(24)
}

/// Rank a node search. An empty query lists the MOST-USED nodes (by
/// `uses`, the hand-add counts the app keeps in prefs) — or nothing on a
/// fresh install, because eighty names in a column fill the screen and say
/// nothing you did not already know. A typed one lists the name it spells
/// EXACTLY first, then names that START with it, then those that merely
/// contain it, then the FUZZY matches ([`fuzzy_score`]), cut to
/// [`SEARCH_MAX`]. Exact first because a whole name typed out is the
/// clearest thing a query can say; fuzzy last because a scattered match is
/// the weakest claim on the letters, and a tier that can never demote a
/// literal one is a tier nobody has to think about. Inside a tier the
/// score plus [`use_bonus`] orders — so "gr" puts the Grade you add every
/// day above the Grain you never do — and the catalog's own order breaks a
/// tie (the sort is stable).
/// Returns the list and how many matched in all, so the popup can say how
/// much it is not showing.
pub fn rank_search<T>(
    all: Vec<(String, T)>,
    q: &str,
    uses: &std::collections::BTreeMap<String, u32>,
) -> (Vec<(String, T)>, usize) {
    let used = |n: &str| uses.get(n).copied().unwrap_or(0);
    let q = q.trim().to_lowercase();
    if q.is_empty() {
        let mut top: Vec<(u32, String, T)> = all
            .into_iter()
            .filter_map(|(n, t)| Some((used(&n), n, t)).filter(|h| h.0 > 0))
            .collect();
        top.sort_by_key(|h| std::cmp::Reverse(h.0));
        top.truncate(SEARCH_MAX);
        let n = top.len();
        return (top.into_iter().map(|(_, n, t)| (n, t)).collect(), n);
    }
    // Spaces are typing, not structure: "simple g" should still find
    // Simple Grade, and dropping them costs nothing — a literal substring
    // match stays a subsequence when its spaces go.
    let tight: String = q.chars().filter(|c| !c.is_whitespace()).collect();
    let mut hits: Vec<(i32, i32, String, T)> = all
        .into_iter()
        .filter_map(|(n, t)| {
            let low = n.to_lowercase();
            let tier = if low == q {
                0
            } else if low.starts_with(&q) {
                1
            } else if low.contains(&q) {
                2
            } else {
                3
            };
            let score = fuzzy_score(&n, &tight)? + use_bonus(used(&n));
            Some((tier, -score, n, t))
        })
        .collect();
    hits.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
    let total = hits.len();
    let mut out: Vec<(String, T)> =
        hits.into_iter().map(|(_, _, n, t)| (n, t)).collect();
    out.truncate(SEARCH_MAX);
    (out, total)
}

/// Walk a search selection by `delta`, wrapping. The list is a handful of
/// rows, so a hand that keeps pressing ↓ expects to come round again
/// rather than stick against the bottom.
pub fn step_sel(sel: usize, delta: i32, len: usize) -> usize {
    if len == 0 {
        return 0;
    }
    let len = len as i32;
    (((sel as i32 + delta) % len + len) % len) as usize
}

/// The Tab node-search popup's state.
struct NodeSearch {
    /// Graph-space spot the picked node lands on.
    at: Pos2,
    query: String,
    /// Which SHOWN hit the arrows have walked to. Clamped every frame and
    /// reset to the top whenever the query changes, because the list is
    /// rebuilt from scratch on each keystroke.
    sel: usize,
}

/// One addable node type.
pub struct NodeEntry {
    pub cat: NodeCat,
    pub label: &'static str,
    /// Takes a fresh reader slot when spawned (a Read, and everything that
    /// plays a bake through one).
    pub reader: bool,
    /// Built with the next free reader slot; ignored by kinds without one.
    /// A closure, not a fn pointer, so an entry generated from a table
    /// (one per GenAI model) can carry its row.
    pub make: std::sync::Arc<dyn Fn(usize) -> NodeKind + Send + Sync>,
    /// Maturity, read off the node `make` builds (see [`status_of`]).
    pub status: NodeStatus,
}

/// Every node the add menus offer, in ONE place: the right-click menu groups
/// it by category and the Tab search flattens it, so the two can no longer
/// drift apart the way two hand-written lists did.
/// A registered custom type that is a SUBWOB (a subnet saved as a node —
/// the UI's name for `tools.rs`'s tools) or one of the input proxies that
/// only exist inside one. The add menus group these apart from shader
/// plugins.
pub fn is_subwob_type(type_id: &str) -> bool {
    // Input nodes and controls are registered as tool furniture too; they
    // are parts OF a subwob, and have menus of their own.
    !crate::tools::is_proxy_id(type_id)
        && crate::tools::ControlKind::from_type(type_id).is_none()
        && engine::plugins::get(type_id).is_some_and(|d| d.tool)
}

/// Extra Tab-search names: what a node is called elsewhere, mapped onto
/// the node that does it, with the matching preset. Search only: the
/// right-click menu and the pie list each node once, by its own name.
pub fn search_aliases() -> Vec<NodeEntry> {
    use engine::graph::{EdgeExtendParams, ExtendMode};
    let e = |label, make: fn(usize) -> NodeKind| NodeEntry {
        cat: NodeCat::Matte,
        label,
        reader: false,
        make: std::sync::Arc::new(make),
        status: status_of(&make(0)),
    };
    vec![
        e("Pixel Spread", |_| {
            NodeKind::EdgeExtend(EdgeExtendParams { mode: ExtendMode::Spread, ..Default::default() })
        }),
        // Pulls clean colour out over a key's dark or light rim, in the
        // premultiplied comp it came from.
        e("Edge Expander", |_| {
            NodeKind::EdgeExtend(EdgeExtendParams {
                mode: ExtendMode::Expand,
                inset: 2.0,
                premult_out: true,
                ..Default::default()
            })
        }),
    ]
}

pub fn node_catalog() -> Vec<NodeEntry> {
    use engine::field as fld;
    use engine::geo;
    use NodeCat::*;
    let e = |cat, label, reader, make: fn(usize) -> NodeKind| NodeEntry {
        cat,
        label,
        reader,
        make: std::sync::Arc::new(make),
        status: status_of(&make(0)),
    };
    let mut v = vec![
        // --- source -----------------------------------------------------
        e(Source, "Load", true, |s| NodeKind::read(s, "")),
        e(Source, "Grid", false, |_| NodeKind::Grid(GridParams::default())),
        e(Source, "Gradient", false, |_| NodeKind::Gradient(Default::default())),
        e(Source, "Blank", false, |_| NodeKind::Blank(Default::default())),
        e(Source, "Starfield", false, |_| NodeKind::Starfield(Default::default())),
        e(Source, "Planet", false, |_| NodeKind::Planet(Default::default())),
        e(Source, "Plasma", false, |_| NodeKind::Plasma(Default::default())),
        e(Source, "Particles", false, |_| NodeKind::Particles(Default::default())),
        e(Source, "Text", false, |_| NodeKind::Text(Default::default())),
        // Words on the text pipe, for a Text's words or a GenAI's prompt.
        // Saved as `Prompt`, its name before the rename.
        e(Utility, "String", false, |_| NodeKind::Prompt(Default::default())),
        // Kernels, drawn: a star for a Convolve, an aperture for a Defocus.
        e(Source, "Glint Kernel", false, |_| NodeKind::GlintKernel(Default::default())),
        e(Source, "Bokeh Kernel", false, |_| NodeKind::BokehKernel(Default::default())),
        e(Source, "Noise", false, |_| NodeKind::Noise(engine::graph::NoiseParams::fresh())),
        e(Source, "2dini", false, |_| NodeKind::Geo(crate::demos::default_geo())),
        // Splines and shapes as DISTANCES, turned into strokes, glows,
        // ripples and bevels over its input.
        e(Source, "SDF", false, |_| NodeKind::Sdf(Default::default())),
        e(Source, "Blender", false, |_| NodeKind::Blender(Default::default())),
        // A free-form generative-AI pipeline (several steps, ComfyUI), run
        // on a click; its takes play back through the reader slot like a
        // bake. The one-model nodes are appended below, one per model.
        e(Ai, "GenAI Pipeline", true, |s| NodeKind::GenAi(engine::graph::GenAiParams::new(s))),
        // --- geometry (inside a 2dini node only) ------------------------
        e(GeoCreate, "Spline", false, |_| {
            NodeKind::GeoNode(geo::GeoNode::Spline {
                points: Vec::new(),
                mode: Default::default(),
                closed: true,
                style: Default::default(),
            })
        }),
        e(GeoCreate, "Text", false, |_| {
            NodeKind::GeoNode(geo::GeoNode::Text {
                text: "text".into(),
                font: Default::default(),
                size: 0.12,
                position: [0.3, 0.33],
                align: Default::default(),
                tracking: 0.0,
                leading: 1.2,
                rotation: 0.0,
                style: Default::default(),
                fx: Default::default(),
            })
        }),
        // A line, not a shape: always open, stroked, never filled.
        e(GeoCreate, "Stroke", false, |_| {
            NodeKind::GeoNode(geo::GeoNode::Stroke {
                points: Vec::new(),
                mode: Default::default(),
                color: [1.0; 4],
                width: 4.0,
            })
        }),
        e(GeoCreate, "Polygon", false, |_| NodeKind::GeoNode(geo::GeoNode::default())),
        // Freehand: drag in the viewer with it selected.
        e(GeoCreate, "Brush", false, |_| {
            NodeKind::GeoNode(geo::GeoNode::Brush {
                strokes: Vec::new(),
                color: [1.0; 4],
                width: 12.0,
                taper: 0.6,
                smooth: 0.4,
            })
        }),
        e(GeoCreate, "Circle", false, |_| {
            NodeKind::GeoNode(geo::GeoNode::Circle {
                center: [0.5, 0.28],
                radius: 0.12,
                style: Default::default(),
            })
        }),
        e(GeoCreate, "Rectangle", false, |_| {
            NodeKind::GeoNode(geo::GeoNode::Rectangle {
                center: [0.5, 0.28],
                size: [0.3, 0.18],
                corner: 0.0,
                rotation: 0.0,
                style: Default::default(),
            })
        }),
        e(GeoCreate, "Star", false, |_| {
            NodeKind::GeoNode(geo::GeoNode::Star {
                center: [0.5, 0.28],
                outer: 0.15,
                inner: 0.07,
                points: 5,
                rotation: 0.0,
                style: Default::default(),
            })
        }),
        e(GeoModify, "Resample", false, |_| {
            NodeKind::GeoNode(geo::GeoNode::Resample {
                mode: Default::default(),
                spacing: 12.0,
                count: 32,
            })
        }),
        e(GeoModify, "Scatter", false, |_| {
            NodeKind::GeoNode(geo::GeoNode::Scatter { count: 200, seed: 0.0, density: 1.0 })
        }),
        e(GeoModify, "Displace", false, |_| {
            NodeKind::GeoNode(geo::GeoNode::Displace {
                mode: fld::WarpMode::Offset,
                amount: 0.05,
                x: 0.5,
                y: 0.5,
            })
        }),
        e(GeoCombine, "Instance", false, |_| {
            NodeKind::GeoNode(geo::GeoNode::Instance {
                scale: 0.1,
                rotation: 0.0,
                inherit_cd: true,
                pivot: Default::default(),
            })
        }),
        e(GeoModify, "Transform", false, |_| {
            NodeKind::GeoNode(geo::GeoNode::Transform {
                translate: [0.0, 0.0],
                rotate: 0.0,
                scale: [1.0, 1.0],
                pivot: [0.0, 0.0],
                inherit: true,
            })
        }),
        e(GeoCombine, "Merge", false, |_| NodeKind::GeoNode(geo::GeoNode::Merge { count: 2 })),
        e(GeoCombine, "Boolean", false, |_| {
            NodeKind::GeoNode(geo::GeoNode::Boolean { op: engine::geobool::BoolOp::Subtract })
        }),
        // Hand edits on whatever flows in: drag its points in the viewer.
        e(GeoModify, "Edit", false, |_| NodeKind::GeoNode(geo::GeoNode::Edit { edits: Vec::new() })),
        e(GeoModify, "Light", false, |_| {
            NodeKind::GeoNode(geo::GeoNode::Light { strength: 2.0, tint: [1.0; 3], occlude: true })
        }),
        e(GeoModify, "Style", false, |_| {
            NodeKind::GeoNode(geo::GeoNode::Style { style: Default::default() })
        }),
        e(GeoValue, "Slider", false, |_| {
            NodeKind::GeoNode(geo::GeoNode::Slider { value: 0.5, name: String::new() })
        }),
        e(GeoValue, "Input", false, |_| {
            NodeKind::GeoNode(geo::GeoNode::Input { name: "in".into(), value: 0.0 })
        }),
        // Words for a Text's words socket, the comp's String node in here.
        e(GeoValue, "String", false, |_| NodeKind::GeoNode(geo::GeoNode::String { text: String::new() })),
        e(GeoValue, "Field In", false, |_| NodeKind::GeoNode(geo::GeoNode::FieldIn { shift: 0.0 })),
        e(GeoValue, "Math", false, |_| {
            NodeKind::GeoNode(geo::GeoNode::Math { op: fld::MathOp::Add, a: 0.0, b: 0.0 })
        }),
        e(GeoValue, "Remap", false, |_| {
            NodeKind::GeoNode(geo::GeoNode::Remap {
                in_lo: 0.0,
                in_hi: 1.0,
                out_lo: 0.0,
                out_hi: 1.0,
                clamp: true,
            })
        }),
        e(GeoValue, "Output", false, |_| NodeKind::GeoNode(geo::GeoNode::Output)),
        e(GeoSimulate, "Rigid Body", false, |_| {
            NodeKind::GeoNode(geo::GeoNode::RigidBody {
                density: 1.0,
                friction: 0.5,
                bounce: 0.2,
                pinned: false,
                velocity: [0.0, 0.0],
                spin: 0.0,
            })
        }),
        e(GeoSimulate, "Soft Body", false, |_| {
            NodeKind::GeoNode(geo::GeoNode::SoftBody {
                blobbiness: 0.6,
                pressure: 1.0,
                friction: 0.5,
                velocity: [0.0, 0.0],
                spin: 0.0,
            })
        }),
        // POP-style particles, solved per frame (no bake).
        e(GeoSimulate, "Particles", false, |_| {
            NodeKind::GeoNode(geo::GeoNode::Particles {
                    rate: 3.0,
                    life: 48.0,
                    life_var: 0.25,
                    speed: 14.0,
                    speed_var: 0.3,
                    direction: -std::f32::consts::FRAC_PI_2,
                    spread: 0.35,
                    gravity: 0.35,
                    wind: 0.0,
                    drag: 0.01,
                    size: 6.0,
                    size_end: 1.0,
                    fade: true,
                    start: 0.0,
                    seed: 0.0,
                })
        }),
        // A bone rig, posed by a turn per bone; a Skin bends shapes by it,
        // a Ragdoll lets it fall.
        e(GeoCreate, "Skeleton", false, |_| {
            use engine::georig::Bone;
            let b = |parent: Option<u32>, head: [f32; 2], tail: [f32; 2]| Bone { name: String::new(), parent, head, tail };
            NodeKind::GeoNode(geo::GeoNode::Skeleton {
                bones: vec![
                    b(None, [0.3, 0.35], [0.43, 0.35]),
                    b(Some(0), [0.43, 0.35], [0.56, 0.35]),
                    b(Some(1), [0.56, 0.35], [0.69, 0.35]),
                ],
                pose: vec![0.0; 3],
            })
        }),
        e(GeoModify, "Skin", false, |_| NodeKind::GeoNode(geo::GeoNode::Skin { smooth: true, falloff: 4.0 })),
        e(GeoSimulate, "Ragdoll", false, |_| {
            NodeKind::GeoNode(geo::GeoNode::Ragdoll {
                gravity: 9.8,
                limit: 1.2,
                frames: 120,
                start: 0.0,
                walls: true,
                kick: [0.0, 0.0, 0.0],
                bake: Default::default(),
            })
        }),
        e(GeoSimulate, "Physics", false, |_| {
            NodeKind::GeoNode(geo::GeoNode::Physics {
                settings: Default::default(),
                bake: Default::default(),
            })
        }),
        // A Noise node inside the drawing: it owns its own field graph —
        // open it to build the noise, where the field nodes live.
        e(GeoValue, "Noise", false, |_| {
            let mut graph = fld::FieldGraph::default();
            graph.nodes.insert(1, fld::FieldNode::default());
            graph.nodes.insert(2, fld::FieldNode::Output);
            graph.edges.insert((2, 0), 1);
            graph.layout.insert(1, [0.0, 0.0]);
            graph.layout.insert(2, [0.0, 220.0]);
            graph.next_id = 3;
            NodeKind::GeoNode(geo::GeoNode::Noise { graph, scale: 1.0, seed: 0.0, speed: 0.0 })
        }),
        // --- noise fields (inside a Noise node only) --------------------
        e(FieldPattern, "fBm", false, |_| NodeKind::Field(fld::FieldNode::default())),
        e(FieldPattern, "Voronoi", false, |_| {
            NodeKind::Field(fld::FieldNode::Voronoi {
                mode: fld::VoronoiMode::F1,
                scale: [0.15, 0.15],
                jitter: 1.0,
                seed: 0.0,
                speed: 1.0,
            })
        }),
        e(FieldPattern, "Coord", false, |_| {
            NodeKind::Field(fld::FieldNode::Coord {
                kind: fld::CoordKind::X,
                scale: 1.0,
            })
        }),
        e(FieldValue, "Input", false, |_| {
            NodeKind::Field(fld::FieldNode::Input { name: "in".into(), value: 0.0 })
        }),
        e(FieldValue, "Slider", false, |_| {
            NodeKind::Field(fld::FieldNode::Const { value: 0.5, name: String::new() })
        }),
        e(FieldMath, "Math", false, |_| {
            NodeKind::Field(fld::FieldNode::Math { op: fld::MathOp::Add, a: 0.0, b: 0.0 })
        }),
        e(FieldMath, "Mix", false, |_| {
            NodeKind::Field(fld::FieldNode::Mix { a: 0.0, b: 1.0, t: 0.5 })
        }),
        e(FieldMath, "Remap", false, |_| {
            NodeKind::Field(fld::FieldNode::Remap {
                in_lo: 0.0,
                in_hi: 1.0,
                out_lo: 0.0,
                out_hi: 1.0,
                clamp: true,
            })
        }),
        e(FieldWarp, "Warp", false, |_| {
            NodeKind::Field(fld::FieldNode::Warp {
                mode: fld::WarpMode::Offset,
                amount: 0.2,
            })
        }),
        e(FieldWarp, "Seed", false, |_| {
            NodeKind::Field(fld::FieldNode::Seed { shift: 17.0 })
        }),
        e(FieldValue, "Output", false, |_| NodeKind::Field(fld::FieldNode::Output)),
        e(FieldShape, "Mask", false, |_| NodeKind::Field(fld::FieldNode::MaskIn)),
        e(FieldShape, "Roto", false, |_| {
            NodeKind::Field(fld::FieldNode::Spline {
                points: Vec::new(),
                mode: Default::default(),
                feather: 1.0,
                offset: [0.0, 0.0],
            })
        }),
        e(FieldShape, "Shape", false, |_| {
            NodeKind::Field(fld::FieldNode::Shape {
                kind: fld::ShapeKind::Ellipse,
                center: [0.5, 0.5],
                size: [0.25, 0.25],
                feather: 0.05,
            })
        }),
        e(FieldMath, "Gradient", false, |_| {
            NodeKind::Field(fld::FieldNode::Gradient {
                axis: fld::Axis::X,
                step: 0.002,
            })
        }),
        // --- SDF shapes, ops, effects (inside an SDF node only) -----------
        e(Sdf, "Spline", false, |_| sdf_of("Spline")),
        e(Sdf, "Circle", false, |_| sdf_of("Circle")),
        e(Sdf, "Rect", false, |_| sdf_of("Rect")),
        e(Sdf, "Combine", false, |_| sdf_of("Combine")),
        e(Sdf, "Offset", false, |_| sdf_of("Offset")),
        e(Sdf, "Shell", false, |_| sdf_of("Shell")),
        e(Sdf, "Morph", false, |_| sdf_of("Morph")),
        e(Sdf, "Transform", false, |_| sdf_of("Transform")),
        e(Sdf, "Displace", false, |_| sdf_of("Displace")),
        e(Sdf, "Slider", false, |_| sdf_of("Slider")),
        e(Sdf, "Input", false, |_| sdf_of("Input")),
        e(Sdf, "Fill", false, |_| sdf_of("Fill")),
        e(Sdf, "Stroke", false, |_| sdf_of("Stroke")),
        e(Sdf, "Glow", false, |_| sdf_of("Glow")),
        e(Sdf, "Shadow", false, |_| sdf_of("Shadow")),
        e(Sdf, "Ripples", false, |_| sdf_of("Ripples")),
        e(Sdf, "Bevel", false, |_| sdf_of("Bevel")),
        e(Sdf, "Refract", false, |_| sdf_of("Refract")),
        e(Sdf, "Ramp", false, |_| sdf_of("Ramp")),
        e(Sdf, "Merge", false, |_| sdf_of("Merge")),
        e(Sdf, "Image", false, |_| sdf_of("Image")),
        e(Sdf, "Output", false, |_| sdf_of("Output")),
        e(Particles, "Emit", false, |_| particle_of("Emit")),
        e(Particles, "Gravity", false, |_| particle_of("Gravity")),
        e(Particles, "Radial", false, |_| particle_of("Radial")),
        e(Particles, "Vortex", false, |_| particle_of("Vortex")),
        e(Particles, "Drag", false, |_| particle_of("Drag")),
        e(Particles, "Turbulence", false, |_| particle_of("Turbulence")),
        e(Particles, "Field Lines", false, |_| particle_of("Field Lines")),
        e(Particles, "Kill", false, |_| particle_of("Kill")),
        e(Particles, "Merge", false, |_| particle_of("Merge")),
        e(Particles, "Render", false, |_| particle_of("Render")),
        // --- transform --------------------------------------------------
        e(Transform, "Transform", false, |_| {
            NodeKind::Transform(TransformParams::default())
        }),
        e(Transform, "Reformat", false, |_| {
            NodeKind::Reformat(ReformatParams::default())
        }),
        e(Transform, "CornerPin", false, |_| {
            NodeKind::CornerPin(CornerPinParams::default())
        }),
        // The camera's shake measured (SEA-RAFT, on a click) and taken out
        // as a fused pin.
        e(Transform, "Stabilize", false, |_| NodeKind::Stabilize(Default::default())),
        // The Crop node (a SoftCrop underneath): bounding box, crop or
        // reformat, picked on the node. A new one crops.
        e(Transform, "Crop", false, |_| {
            NodeKind::SoftCrop(engine::graph::SoftCropParams {
                mode: engine::graph::CropMode::Crop,
                ..Default::default()
            })
        }),
        // A Crop's region set back into the full format (a render of the
        // box, re-imported), linked to the Crop it undoes.
        e(Transform, "Uncrop", false, |_| NodeKind::Uncrop(Default::default())),
        // Two pictures A / B or split by a wipe, the smaller scaled up.
        e(Transform, "Compare", false, |_| NodeKind::Compare(Default::default())),
        e(Transform, "UV Warp", false, |_| NodeKind::UvWarp(Default::default())),
        // A plate or a latlong through bumpy glass (a height or normal map).
        e(Transform, "Refract", false, |_| NodeKind::Refract(Default::default())),
        e(Transform, "Mirror", false, |_| NodeKind::Mirror(Default::default())),
        e(Transform, "Lens Distortion", false, |_| NodeKind::LensDistort(Default::default())),
        e(Transform, "FlipFlop", false, |_| NodeKind::FlipFlop(Default::default())),
        // Quarter turns, the format turned with them (1920x1080 -> 1080x1920).
        e(Transform, "Turn", false, |_| NodeKind::Turn(Default::default())),
        e(Transform, "VectorWarp", false, |_| NodeKind::VectorWarp(Default::default())),
        e(Transform, "GridWarp", false, |_| NodeKind::GridWarp(Default::default())),
        e(Transform, "Warp", false, |_| NodeKind::Warp(Default::default())),
        // --- colour -----------------------------------------------------
        e(Colour, "Grade", false, |_| NodeKind::Grade(GradeParams::default())),
        e(Colour, "Mult", false, |_| NodeKind::Mult(Default::default())),
        e(Colour, "Curve", false, |_| NodeKind::Curve(CurveParams::default())),
        e(Colour, "Saturation", false, |_| NodeKind::Saturation(Default::default())),
        e(Colour, "Contrast", false, |_| NodeKind::Contrast(Default::default())),
        e(Colour, "ColorCompress", false, |_| NodeKind::ColorCompress(Default::default())),
        e(Colour, "HueRotate", false, |_| NodeKind::HueRotate(Default::default())),
        e(Colour, "LUT", false, |_| NodeKind::Lut(Default::default())),
        e(Colour, "Inverse View", false, |_| NodeKind::Lut(engine::graph::LutParams::inverse_view())),
        e(Colour, "Gamma", false, |_| NodeKind::Gamma(Default::default())),
        // A soft shoulder off the threshold, and the whole-range tone maps
        // (the same node, another operator).
        e(Colour, "Rolloff", false, |_| NodeKind::ToneMap(Default::default())),
        e(Colour, "Tone Map", false, |_| {
            let op = engine::graph::ToneOp::AgX;
            NodeKind::ToneMap(engine::graph::ToneMapParams { op, white: op.default_white(), ..Default::default() })
        }),
        e(Utility, "Expression", false, |_| NodeKind::Expression(Default::default())),
        e(Utility, "Burnin", false, |_| NodeKind::Burnin(Default::default())),
        e(Colour, "Tinter", false, |_| NodeKind::ColorSlice(Default::default())),
        e(Colour, "Reorder", false, |_| NodeKind::Shuffle(ShuffleParams::IDENTITY)),
        e(Colour, "Unpremult", false, |_| NodeKind::Premult { premult: false }),
        e(Colour, "Premult", false, |_| NodeKind::Premult { premult: true }),
        // --- filter -----------------------------------------------------
        e(Filter, "Blur", false, |_| NodeKind::Blur(BlurParams::default())),
        e(Filter, "Grain", false, |_| NodeKind::Grain(Default::default())),
        e(Filter, "MotionBlur", false, |_| NodeKind::MotionBlur(Default::default())),
        e(Filter, "VectorBlur", false, |_| NodeKind::VectorBlur(Default::default())),
        e(ThreeD, "Camera", false, |_| NodeKind::Camera3D(Default::default())),
        // A 3D camera solved off the picture (SEA-RAFT tracks + our own
        // bundle adjustment, on a click), read by a Camera's tracker source.
        e(ThreeD, "CameraTrack", false, |_| NodeKind::CamTrack(Default::default())),
        e(ThreeD, "Card", false, |_| NodeKind::Card3D(Default::default())),
        e(ThreeD, "Env", false, |_| NodeKind::Env(Default::default())),
        e(ThreeD, "Blend 3D", false, |_| NodeKind::Scene3D(Default::default())),
        e(ThreeD, "Render3D", false, |_| NodeKind::Render3D(Default::default())),
        e(ThreeD, "Splat", false, |_| NodeKind::Splat(Default::default())),
        e(Filter, "Defocus", false, |_| NodeKind::Defocus(DefocusParams::default())),
        e(Filter, "Glow", false, |_| NodeKind::Glow(GlowParams::default())),
        e(Filter, "Godrays", false, |_| NodeKind::Godrays(Default::default())),
        e(Filter, "Lens Flare", false, |_| NodeKind::LensFlare(Default::default())),
        e(Filter, "Glint", false, |_| NodeKind::Glint(Default::default())),
        e(Filter, "Convolve", false, |_| NodeKind::Convolve(Default::default())),
        e(Filter, "Median", false, |_| NodeKind::Median(Default::default())),
        // Clipped channels rebuilt from the unclipped ones, at the colour
        // around them: real data only, no model.
        e(Filter, "HighlightRebuild", false, |_| NodeKind::HighlightRebuild(Default::default())),
        e(Filter, "EdgeDetect", false, |_| NodeKind::EdgeDetect(Default::default())),
        e(Filter, "Diffuse", false, |_| NodeKind::Diffuse(Default::default())),
        e(Filter, "Lightwrap", false, |_| NodeKind::Lightwrap(Default::default())),
        // --- matte ------------------------------------------------------
        e(Matte, "ToMatte", false, |_| NodeKind::ToMatte(Default::default())),
        // Not a matte: the pipe next door. It sits here because the gesture
        // that reaches for it is the same one ("I have a channel I need on
        // its own wire"), and because there is nowhere else a channel pick
        // belongs.
        e(Matte, "ToData", false, |_| NodeKind::ToData(Default::default())),
        e(Matte, "ToRgba", false, |_| NodeKind::ToRgba(Default::default())),
        e(Matte, "MatteIn", false, |_| NodeKind::MatteIn),
        e(Matte, "Spline", false, |_| NodeKind::Spline(SplineParams::default())),
        e(Matte, "LumaKey", false, |_| NodeKind::LumaKey(LumaKeyParams::default())),
        e(Matte, "Keyer", false, |_| NodeKind::Keyer(Default::default())),
        e(Matte, "ScreenKey", false, |_| NodeKind::ScreenKey(Default::default())),
        e(Matte, "Despill", false, |_| NodeKind::Despill(Default::default())),
        // Corridor Digital's neural keyer, run as a process on a click; the
        // result plays through the reader slot like a bake.
        e(Ai, "CorridorKey", true, |s| NodeKind::CorridorKey(engine::graph::CorridorKeyParams::new(s))),
        // Click-to-roto: SAM 2 tracks what you click, optionally matted by
        // MatAnyone; a process on a click, played back like a bake.
        e(Ai, "Sammie", true, |s| NodeKind::Sammie(engine::graph::SammieParams::new(s))),
        // Neural super-resolution in our own WGSL (Real-ESRGAN, SPAN) on a
        // click, played back like a bake; the gate downloads the weights.
        e(Ai, "Upscale", true, |s| NodeKind::Upscale(engine::graph::UpscaleParams::new(s))),
        // HDRCNN: clipped highlights reconstructed by a CNN on a click, in
        // our own WGSL; pixels under the clip stay exactly the plate.
        e(Ai, "HDRCNN", true, |s| NodeKind::HdrCnn(engine::graph::HdrCnnParams::new(s))),
        // SeedVR2: diffusion restoration and upscaling, a process on a click
        // (CUDA or Apple Silicon), played back like a bake.
        e(Ai, "SeedVR2", true, |s| NodeKind::SeedVr(engine::graph::SeedVrParams::new(s))),
        // Video Depth Anything: steady depth for a clip on a click, played
        // back like a bake on the data pipe; the gate downloads the weights.
        e(Ai, "Depth", true, |s| NodeKind::Depth(engine::graph::DepthParams::new(s))),
        // LTX-2.5 SDR-to-HDR: clipped highlights made up by a video model on
        // a click, composited back where the plate clipped. Restricted
        // licence, tagged.
        e(Ai, "LTX HDR", true, |s| NodeKind::LtxHdr(engine::graph::LtxHdrParams::new(s))),
        // Motion vectors: RIFE on a click, played back like a bake, read by
        // VectorBlur / VectorWarp (or anything else that reads a picture).
        e(Ai, "Flow", true, |s| NodeKind::Flow(engine::graph::FlowParams::new(s))),
        // Slow-down with real in-between frames: RIFE on a click, played
        // back like a bake.
        e(Time, "Rifetime", true, |s| NodeKind::Rifetime(engine::graph::RifetimeParams::new(s))),
        // Hold one frame; mend bad frames from their neighbours (a live
        // blend, or RIFE in-betweens on a click — then a reader).
        e(Time, "Freeze", false, |_| NodeKind::Freeze(engine::graph::FreezeParams::at(playhead() as f32))),
        e(Time, "Frame Fix", true, |s| {
            NodeKind::FrameFix(engine::graph::FrameFixParams::new(s, playhead() as i64))
        }),
        e(Matte, "Cryptomatte", false, |_| NodeKind::Cryptomatte(Default::default())),
        e(Matte, "MatteCombine", false, |_| NodeKind::MatteCombine(Default::default())),
        e(Matte, "Mask", false, |_| NodeKind::Mask),
        e(Matte, "Extract", false, |_| NodeKind::Extract(Default::default())),
        e(Matte, "Dilate", false, |_| NodeKind::Dilate(Default::default())),
        e(Matte, "EdgeExtend", false, |_| NodeKind::EdgeExtend(Default::default())),
        // --- composite --------------------------------------------------
        e(Composite, "Blend", false, |_| NodeKind::blend()),
        e(Composite, "Mix", false, |_| NodeKind::Mix(Default::default())),
        e(Composite, "Layer", false, |_| NodeKind::layer()),
        // --- time -------------------------------------------------------
        e(Utility, "Switch", false, |_| {
            NodeKind::Switch(engine::graph::SwitchParams::default())
        }),
        e(Time, "Slip", false, |_| NodeKind::Slip(SlipParams::default())),
        e(Time, "Speed", false, |_| NodeKind::Speed(SpeedParams::default())),
        e(Time, "Time Curve", false, |_| {
            NodeKind::TimeRemap(TimeRemapParams::default())
        }),
        e(Time, "Time Remap", false, |_| NodeKind::RangeRemap(Default::default())),
        e(Time, "Time Range", false, |_| NodeKind::TimeRange(Default::default())),
        // --- utility ----------------------------------------------------
        e(Utility, "Breakout", false, |_| {
            NodeKind::Breakout(Default::default())
        }),
        e(Composite, "Recombine", false, |_| {
            NodeKind::Recombine(Default::default())
        }),
        e(Utility, "ChannelSelect", true, |s| {
            NodeKind::ChannelSelect(engine::graph::ChannelSelectParams {
                source: s,
                layer: String::new(),
            })
        }),
        e(Utility, "LoopIn", false, |_| NodeKind::LoopIn),
        e(Utility, "LoopOut", false, |_| NodeKind::LoopOut(Default::default())),
        e(Utility, "Shader", false, |_| NodeKind::Shader(Default::default())),
        e(Utility, "Render", false, |_| NodeKind::Write(Default::default())),
        e(Utility, "Paint", false, |_| NodeKind::Paint(Default::default())),
        e(Utility, "Tracker", false, |_| NodeKind::Tracker(TrackerParams::default())),
        e(Utility, "PlanarTrack", false, |_| NodeKind::PlanarTrack(Default::default())),
        e(Utility, "Bake", true, |s| {
            NodeKind::GraphBake { source: s, baked: false, start: None, mask: false, window: None, matte_half: false, single: false }
        }),
        // A routing anchor. Costs nothing: it is elided before the engine
        // sees a graph, so it is tidiness with no frame-rate attached.
        e(Utility, "Dot", false, |_| NodeKind::dot(PortKind::Image)),
    ];
    // One node per remote model, named for it: the model is the node, not
    // a dropdown inside one. Chain them in the comp, collapse a chain into
    // a subwob. Video models only where a video take can play.
    v.extend(crate::genaiui::offered_models().map(|m| {
        let id = m.id;
        let make = move |s| NodeKind::GenAi(engine::graph::GenAiParams::for_model(s, id));
        NodeEntry { cat: Ai, label: m.label, reader: true, status: status_of(&make(0)), make: std::sync::Arc::new(make) }
    }));
    // Ready-made GenAI pipelines (`genai::templates`): a GenAI Pipeline
    // node, already wired.
    v.extend(crate::genaiui::tmpl::offered_templates().map(|t| {
        let make = move |s| {
            let mut p = engine::graph::GenAiParams::new(s);
            p.pipeline = t.pipeline();
            NodeKind::GenAi(p)
        };
        NodeEntry { cat: Ai, label: t.name, reader: true, status: status_of(&make(0)), make: std::sync::Arc::new(make) }
    }));
    // Inside a Roto's dive: a shape is a Spline, a combine a MatteCombine,
    // and an Axis groups and moves them.
    v.push(e(Roto, "Shape", false, |_| NodeKind::Spline(SplineParams::default())));
    v.push(e(Roto, "Combine", false, |_| NodeKind::MatteCombine(Default::default())));
    v.push(e(Roto, "Axis", false, |_| crate::rotodive::new_axis()));
    // Inside a GenAI Pipeline's dive: its steps, one node each.
    v.extend(crate::genaiui::step_entries().into_iter().map(|(label, op)| {
        let make = move |_| NodeKind::GenAiStep(op.clone());
        NodeEntry { cat: GenAiStep, label, reader: false, status: status_of(&make(0)), make: std::sync::Arc::new(make) }
    }));
    // A matte drawn in there: a real Spline (`genai::Op::Shape`).
    v.push(e(GenAiStep, "Spline", false, |_| NodeKind::Spline(SplineParams::default())));
    v
}

/// Which of `selection` a node just added from the Tab search takes as its
/// inputs, and where: the FIRST selected node into input 0 — and when two
/// were selected and the new node has an fg/bg pair (a Blend's + and −, a
/// Mix's from and to), the second into the other half, so a Blend gets the
/// first pick as its foreground. Selection order is pick order (click,
/// then shift-clicks). A Layer stack takes EVERY pick, one a layer, the
/// first on top (`add_wired` grows it to fit). Nothing for a node with no
/// inputs.
pub(crate) fn feeds_for_new(
    graph: &EditorGraph,
    id: u64,
    selection: &[u64],
) -> Vec<(u64, usize)> {
    let Some(kind) = graph.node(id).map(|n| &n.kind) else { return Vec::new() };
    if kind.input_count() == 0 {
        return Vec::new();
    }
    let picked: Vec<u64> = selection
        .iter()
        .copied()
        .filter(|s| *s != id && graph.node(*s).is_some())
        .collect();
    if let (NodeKind::Layer(_), [_, _, ..]) = (kind, picked.as_slice()) {
        return picked.iter().take(kind.input_count()).enumerate().map(|(i, s)| (*s, i)).collect();
    }
    match (picked.as_slice(), swappable_inputs(kind)) {
        ([fg, bg, ..], Some((a, b))) => vec![(*fg, a), (*bg, b)],
        ([first, ..], _) => vec![(*first, 0)],
        _ => Vec::new(),
    }
}

/// Every creation path's add — Tab search, the right-click menu, the F1
/// node pie: `kind` wired from the SELECTION (`feeds_for_new`: a Blend
/// takes two, the first selected OVER), a single feed spliced into its
/// chain (the new node takes the fed node's place downstream), and PLACED
/// from what it is wired to (`placement`), falling back to `at` — the
/// cursor, the menu's click, the pie's centre — when nothing feeds it. The
/// new node becomes the selection. No wiring in patch mode, where the
/// ordinary wires are locked.
/// Which of a node's free ports courts an output first when several could
/// take it: the main input (0), then the mask (1), then the modulate (2).
fn port_rank(kind: &NodeKind, i: usize) -> u8 {
    if kind.mod_input() == Some(i) {
        2
    } else if kind.mask_input() == Some(i) {
        1
    } else {
        0
    }
}

/// How far around a new node's spot (in SCREEN pixels at the zoom it is
/// made at) a neighbour sets its size.
const NEIGHBOUR_PX: f32 = 220.0;

/// The size a node made at `at` (graph space) should take: the NEAREST
/// neighbour's, when one is within [`NEIGHBOUR_PX`] on screen, so a node
/// added beside a chain at any zoom matches the chain; with nobody near,
/// the zoom-relative size that reads the same on screen at every zoom.
/// Dots and harp strings say nothing about a node's size and are skipped.
pub(crate) fn new_node_size(graph: &EditorGraph, at: Pos2, zoom: f32, skip: &[u64]) -> f32 {
    let alone = (1.0 / zoom).clamp(0.02, 25.0);
    let reach = NEIGHBOUR_PX / zoom.max(1e-6);
    let kids = crate::breakout::kid_set(graph);
    graph
        .nodes
        .iter()
        .filter(|(id, n)| {
            !skip.contains(id) && !matches!(n.kind, NodeKind::Dot { .. }) && !kids.contains(id)
        })
        .map(|(_, n)| (n.pos.distance(at) - NODE_R * n.size, n.size))
        .filter(|(d, _)| *d <= reach)
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map_or(alone, |(_, s)| s.clamp(0.02, 25.0))
}

/// The tick on screen, noted each UI pass: a Freeze or Frame Fix added from
/// the menus starts on it. Only node SEEDING reads this.
static PLAYHEAD: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub(crate) fn note_playhead(t: u64) {
    PLAYHEAD.store(t, std::sync::atomic::Ordering::Relaxed);
}

fn playhead() -> u64 {
    PLAYHEAD.load(std::sync::atomic::Ordering::Relaxed)
}

pub(crate) fn add_wired(
    graph: &mut EditorGraph,
    kind: NodeKind,
    at: Pos2,
    selection: &mut Vec<u64>,
    zoom: f32,
    patch_mode: bool,
) -> u64 {
    let id = graph.add(kind, at);
    // The size of the nodes around it; alone, a consistent size on screen.
    let size = new_node_size(graph, at, zoom, &[id]);
    if let Some(n) = graph.nodes.get_mut(&id) {
        n.size = size;
        n.tv = new_read_tv(&n.kind);
    }
    // A Breakout arrives as a harp: receiver included.
    crate::breakout::after_add(graph, id);
    // A Layer made from several picks has a layer for each.
    if !patch_mode {
        let picks = selection.iter().filter(|s| **s != id && graph.node(**s).is_some()).count();
        if let Some(NodeKind::Layer(p)) = graph.nodes.get_mut(&id).map(|n| &mut n.kind) {
            if picks > p.layers.len() {
                p.layers.resize(picks.min(engine::graph::LAYER_MAX), Default::default());
            }
        }
    }
    let feeds = if patch_mode { Vec::new() } else { feeds_for_new(graph, id, selection) };
    for (src, input) in &feeds {
        connect_or_bridge(graph, *src, 0, id, *input, zoom);
    }
    // One pick only: with two (a Blend's fg and bg) there is no single
    // chain to insert into, and with a bridge between, the new node is not
    // what the selection feeds.
    if let [(src, 0)] = feeds[..] {
        if graph.input_of(id, 0) == Some(src) {
            splice_downstream(graph, src, 0, id);
        }
    }
    if let Some((p, size)) = placement(graph, id, &feeds) {
        let d = p - graph.node(id).map(|n| n.pos).unwrap_or(p);
        let mut moved = vec![id];
        moved.extend(crate::breakout::companions(graph, &[id]));
        for m in moved {
            if let Some(n) = graph.nodes.get_mut(&m) {
                n.pos += d;
                n.size = size;
            }
        }
    }
    *selection = vec![id];
    id
}

/// Where a new node fed by `feeds` belongs, and at what size (its
/// feeders'). Under them: centred between two — a Blend's fg and bg — one
/// tidy row below the lower; straight under one. A single feeder it took
/// the place of sits midway to what it now feeds when there is room for a
/// node between, and beside that midpoint when there is not, so the wires
/// still run downhill. Then it steps past anything already standing there.
/// `None` when nothing feeds it: the caller's point stands.
fn placement(graph: &EditorGraph, id: u64, feeds: &[(u64, usize)]) -> Option<(Pos2, f32)> {
    let srcs: Vec<&ENode> = feeds.iter().filter_map(|(s, _)| graph.node(*s)).collect();
    if srcs.is_empty() {
        return None;
    }
    let size = srcs.iter().map(|n| n.size).sum::<f32>() / srcs.len() as f32;
    let step = (2.0 * NODE_R + crate::nodeux::ROW_GAP) * size;
    let mut x = srcs.iter().map(|n| n.pos.x).sum::<f32>() / srcs.len() as f32;
    let low = srcs.iter().map(|n| n.pos.y).fold(f32::MIN, f32::max);
    let mut y = low + step;
    let mut sideways = false;
    if srcs.len() == 1 {
        // What the new node feeds after the splice, nearest first.
        let below = graph
            .edges
            .iter()
            .filter(|(d, _, s)| *s == id && *d != id)
            .filter_map(|(d, _, _)| graph.node(*d))
            .map(|n| n.pos.y)
            .filter(|cy| *cy > low)
            .fold(f32::MAX, f32::min);
        if below < f32::MAX {
            y = (low + below) * 0.5;
            if below - low < 1.6 * step {
                // No room in the column: beside it, at the same height.
                x += step;
                sideways = true;
            }
        }
    }
    let clear = |p: Pos2| {
        graph
            .nodes
            .iter()
            .all(|(nid, n)| *nid == id || n.pos.distance(p) > 2.2 * NODE_R * size.max(n.size))
    };
    let mut p = Pos2::new(x, y);
    for _ in 0..16 {
        if clear(p) {
            break;
        }
        if sideways {
            p.x += step * 0.5;
        } else {
            p.y += step * 0.5;
        }
    }
    Some((p, size))
}

/// A Tab-added node takes the selected node's PLACE: everything `src` fed
/// from output `out` feeds from `new` instead, so an add lands IN the chain
/// rather than hanging a branch off it. Never touches the new node's own
/// feed, nor a consumer whose port would not take what `new` gives (a key's
/// matte consumers when what was added makes an image) — that wire stays
/// where it is rather than being re-typed behind your back. A spliced wire
/// keeps the layer it was drawn on. Returns how many moved.
pub(crate) fn splice_downstream(graph: &mut EditorGraph, src: u64, out: u32, new: u64) -> usize {
    let Some(gives) = graph.node(new).map(|n| n.kind.output_kind_at(0)) else {
        return 0;
    };
    let moved: Vec<(u64, usize, bool)> = graph
        .edges
        .iter()
        .filter(|(d, i, s)| *s == src && *d != new && graph.out_of(*d, *i) == out)
        .filter(|(d, i, _)| {
            // Strictly the same pipe: a matte op added below a picture must
            // not pull a pipe-follower (a Grade) over onto the matte.
            graph.node(*d).is_some_and(|n| gives.can_feed(n.kind.input_kind(*i)))
        })
        .map(|(d, i, _)| (*d, *i, graph.is_hidden(*d, *i)))
        .collect();
    for (d, i, hidden) in &moved {
        graph.connect_out(*d, *i, new, 0);
        // An ordinary connect surfaces a wire; a spliced one stays as drawn.
        if *hidden {
            graph.hidden_wires.insert((*d, *i));
        }
    }
    moved.len()
}

/// The pair of inputs a kind lets you exchange. Only for kinds whose ports
/// mean DIFFERENT things and are the same port KIND — a Blend's over/under,
/// a Mix's from/to. Nothing here can swap an image port with a mask one:
/// that is not a flip, it is a broken graph.
pub fn swappable_inputs(kind: &NodeKind) -> Option<(usize, usize)> {
    match kind {
        NodeKind::Blend { .. } | NodeKind::Mix(_) => Some((0, 1)),
        // A layer stack has as many orderings as it has layers; two ports
        // is not the question being asked, so the stack panel owns it.
        _ => None,
    }
}

pub fn node_label(kind: &NodeKind) -> String {
    match kind {
        // A dot is a bend in a wire; a label beside it would be the loudest
        // thing on screen for the least reason.
        NodeKind::Dot { .. } => String::new(),
        NodeKind::Read { source, .. } => format!("Read {}", source + 1),
        NodeKind::GraphBake { source, .. } => format!("Bake {}", source + 1),
        NodeKind::Layer(p) => format!("Layer ×{}", p.layers.len()),
        // A one-model GenAI node is called by its model.
        NodeKind::GenAi(p) => match (&p.model, p.remote_model()) {
            (_, Some(m)) => m.label.to_string(),
            (Some(id), None) => format!("unknown model {id}"),
            (None, None) => "GenAI Pipeline".to_string(),
        },
        NodeKind::Custom { type_id, .. } => match engine::plugins::get(type_id) {
            Some(def) => def.label.clone(),
            None => format!("missing: {type_id}"),
        },
        other => other.name().to_string(),
    }
}

/// A node's body colour. Every kind reads a `node_*` slot in
/// `crate::theme`, so a new kind gets a slot there too.
/// What a node is CALLED on screen: its hand-given name when it has one,
/// its kind otherwise. Menus and catalogs keep using `node_label` — a
/// rename must never hide what a thing IS from the add paths.
pub fn display_label(n: &ENode) -> String {
    if n.ui_label.is_empty() { node_label(&n.kind) } else { n.ui_label.clone() }
}

/// A human name for a Read, guessed from its path: the file stem with
/// frame counters, version tags and filler words stripped — or the
/// deepest useful directory when the stem is all filler ("frame.0001").
/// `data/crypto/frame.0000.exr` guesses "crypto".
pub fn guess_read_label(path: &str) -> String {
    let clean = |s: &str| -> String {
        s.split(['.', '_', '-', ' '])
            .filter(|t| !t.is_empty())
            // Frame counters, and the # padding a typed pattern spells
            // them with.
            .filter(|t| !t.chars().all(|c| c.is_ascii_digit() || c == '#'))
            .filter(|t| {
                !(t.len() >= 2
                    && (t.starts_with('v') || t.starts_with('V'))
                    && t[1..].chars().all(|c| c.is_ascii_digit()))
            })
            .filter(|t| {
                !matches!(
                    t.to_ascii_lowercase().as_str(),
                    "frame" | "frames" | "render" | "renders" | "img" | "image"
                        | "plate" | "plates" | "out" | "output" | "exr" | "seq"
                        | "sequence" | "comp" | "data"
                )
            })
            .collect::<Vec<_>>()
            .join("_")
    };
    let p = std::path::Path::new(path);
    let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    let s = clean(stem);
    if !s.is_empty() {
        return s;
    }
    for anc in p.ancestors().skip(1) {
        if let Some(d) = anc.file_name().and_then(|x| x.to_str()) {
            let c = clean(d);
            if !c.is_empty() {
                return c;
            }
        }
    }
    String::new()
}

/// The colour a pipe wears on its own — the matte's lime, the data pipe's
/// violet — or `None` for the image pipe, which wears its COST instead. One
/// function, so a wire, the socket it lands in and the socket it leaves can
/// never disagree about which pipe they are.
pub fn pipe_color(kind: PortKind) -> Option<Color32> {
    match kind {
        PortKind::Mask => Some(theme::wire_mask()),
        PortKind::Data => Some(theme::wire_data()),
        PortKind::Text => Some(theme::wire_text()),
        PortKind::Vector => Some(theme::wire_vector()),
        PortKind::Scene => Some(theme::wire_scene()),
        _ => None,
    }
}

/// What a 2dini noodle carries, read off the node that feeds it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GeoWire {
    Shapes,
    Points,
    Bodies,
    Number,
    Field,
}

impl GeoWire {
    pub const ALL: [GeoWire; 5] =
        [GeoWire::Shapes, GeoWire::Points, GeoWire::Bodies, GeoWire::Number, GeoWire::Field];

    pub fn color(self) -> Color32 {
        match self {
            GeoWire::Shapes => theme::wire_geo(),
            GeoWire::Points => theme::wire_points(),
            GeoWire::Bodies => theme::wire_bodies(),
            GeoWire::Number => theme::wire_number(),
            GeoWire::Field => theme::wire_field(),
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            GeoWire::Shapes => "shapes",
            GeoWire::Points => "points — scattered, to instance onto",
            GeoWire::Bodies => "bodies — tagged for physics",
            GeoWire::Number => "a number",
            GeoWire::Field => "a field — a number that varies over the canvas",
        }
    }
}

/// What node `src`'s output carries inside a 2dini graph. Shapes unless
/// something upstream says otherwise: a Scatter makes points (until an
/// Instance stamps shapes onto them), a body tag makes bodies, and a
/// scalar is a field once a FieldIn or Noise is anywhere above it.
pub fn geo_wire_class(g: &EditorGraph, src: u64, memo: &mut std::collections::HashMap<u64, GeoWire>) -> GeoWire {
    use engine::geo::GeoNode as G;
    if let Some(c) = memo.get(&src) {
        return *c;
    }
    // A cycle cannot happen in a DAG; the placeholder only stops one.
    memo.insert(src, GeoWire::Shapes);
    let up = |i: usize, memo: &mut std::collections::HashMap<u64, GeoWire>| g.input_of(src, i).map(|s| geo_wire_class(g, s, memo));
    let c = match g.node(src).map(|n| &n.kind) {
        Some(NodeKind::GeoNode(k)) if k.is_scalar() => {
            let varies = matches!(k, G::FieldIn { .. } | G::Noise { .. })
                || (0..k.inputs().len()).any(|i| up(i, memo) == Some(GeoWire::Field));
            if varies { GeoWire::Field } else { GeoWire::Number }
        }
        Some(NodeKind::GeoNode(G::Scatter { .. } | G::Particles { .. })) => GeoWire::Points,
        Some(NodeKind::GeoNode(G::RigidBody { .. } | G::SoftBody { .. })) => GeoWire::Bodies,
        Some(NodeKind::GeoNode(G::Instance { .. })) => match up(0, memo) {
            Some(GeoWire::Bodies) => GeoWire::Bodies,
            _ => GeoWire::Shapes,
        },
        Some(NodeKind::GeoNode(G::Physics { .. })) => GeoWire::Bodies,
        Some(NodeKind::GeoNode(k)) => {
            // Anything else passes on what it is fed; a combiner is bodies
            // or points only if every input is.
            let ins: Vec<GeoWire> = k
                .inputs()
                .iter()
                .enumerate()
                .filter(|(_, s)| s.kind == PortKind::Geo)
                .filter_map(|(i, _)| up(i, memo))
                .collect();
            match ins.first() {
                Some(f) if ins.iter().all(|c| c == f) => *f,
                _ => GeoWire::Shapes,
            }
        }
        _ => GeoWire::Shapes,
    };
    memo.insert(src, c);
    c
}

/// What a noise-graph noodle carries, as the 2dini colours say it: a
/// PATTERN (anything that varies over the canvas), a SHAPE (a roto, a
/// shape, the matte port, and what is made only from them), or a plain
/// NUMBER (sliders, inputs, and arithmetic on nothing else).
pub fn field_wire_class(g: &EditorGraph, src: u64, memo: &mut std::collections::HashMap<u64, GeoWire>) -> GeoWire {
    use engine::field::FieldNode as F;
    if let Some(c) = memo.get(&src) {
        return *c;
    }
    memo.insert(src, GeoWire::Field);
    let c = match g.node(src).map(|n| &n.kind) {
        Some(NodeKind::Field(k)) => match k {
            F::Const { .. } | F::Input { .. } => GeoWire::Number,
            F::Shape { .. } | F::Spline { .. } | F::MaskIn => GeoWire::Shapes,
            F::Fbm { .. } | F::Voronoi { .. } | F::Coord { .. } | F::Gradient { .. } => GeoWire::Field,
            k => {
                let ins: Vec<GeoWire> =
                    (0..k.inputs().len()).filter_map(|i| g.input_of(src, i).map(|s| field_wire_class(g, s, memo))).collect();
                if ins.contains(&GeoWire::Field) {
                    GeoWire::Field
                } else if ins.contains(&GeoWire::Shapes) {
                    GeoWire::Shapes
                } else {
                    GeoWire::Number
                }
            }
        },
        _ => GeoWire::Field,
    };
    memo.insert(src, c);
    c
}

/// The noise graph's legend: its three noodle colours, in its words.
pub fn field_legend() -> Vec<(&'static str, Color32, Swatch)> {
    vec![
        ("a pattern — varies over the canvas", GeoWire::Field.color(), Swatch::Plain),
        ("a shape or matte", GeoWire::Shapes.color(), Swatch::Plain),
        ("a number", GeoWire::Number.color(), Swatch::Plain),
    ]
}

/// A nodule's hover tip: what flows through it.
fn pipe_words(kind: PortKind, input: bool) -> (&'static str, &'static str) {
    match kind {
        PortKind::Image => ("image", ""),
        PortKind::Mask if input => (
            "matte",
            "MIXES the finished effect over the original: white is the full \
             effect, black leaves it untouched",
        ),
        PortKind::Mask => ("matte", ""),
        PortKind::Data => ("data", "one channel at full depth (depth, an id…), never clamped"),
        PortKind::Channels => ("channels", "every layer of a multi-layer file, as one fat cable"),
        PortKind::Text => ("text", "a prompt"),
        PortKind::Vector => ("vectors", "motion: rg forward, ba backward, in pixels"),
        PortKind::Scene => ("scene", "cameras and cards, for a Render3D"),
        PortKind::Field => ("field", ""),
        PortKind::Geo => ("geometry", ""),
    }
}

fn port_tip(kind: PortKind, name: Option<&str>, input: bool) -> String {
    let (what, detail) = pipe_words(kind, input);
    let mut t = match name.filter(|n| !matches!(*n, "image" | "input" | "mask")) {
        Some(n) => format!("{n}: {what} {}", if input { "input" } else { "output" }),
        None => format!("{what} {}", if input { "input" } else { "output" }),
    };
    if !detail.is_empty() {
        t += &format!("\n{detail}");
    }
    if !input {
        t += "\ndrag a noodle from here to an input";
    }
    t
}

/// The hover tip on an input socket: its name, its pipe, and what the node
/// does with it (`NodeKind::input_about`), or the pipe's own words where
/// the port means the same on every node (a mask, a modulate).
pub fn input_tip(kind: &NodeKind, i: usize) -> String {
    if kind.mod_input() == Some(i) {
        return "modulate input (~)\na matte that SCALES the effect per pixel: \
                white is the full effect, black none"
            .to_string();
    }
    let name = kind.input_label(i);
    let pk = kind.input_kind(i);
    match kind.input_about(i) {
        Some(about) => {
            let (what, _) = pipe_words(pk, true);
            let head = if matches!(name.as_str(), "image" | "input") {
                format!("{what} input")
            } else {
                format!("{name}: {what} input")
            };
            format!("{head}\n{about}")
        }
        None => port_tip(pk, Some(&name), true),
    }
}

/// The big dot's "which input?" list: `src`'s output `out` is waiting to
/// land in one of `options` on `dst`.
#[derive(Clone, Debug, PartialEq)]
pub struct BigDotMenu {
    pub dst: u64,
    pub src: u64,
    pub out: u32,
    /// Screen position of the drop.
    pub at: Pos2,
    pub options: Vec<usize>,
    /// When it opened (egui time), for the splay.
    pub t0: f64,
}

/// The big dot's key in the stalk springs: below the output keys at the
/// top of `usize` (`out_stalk`), above any input index.
const BIG_DOT_STALK: usize = OUT_STALK - MAX_OUTPUTS - 1;

/// The big dot's radius against a socket's, graph units at size 1.
const BIG_DOT_R: f32 = 7.0;

/// What a socket's nodule looks like.
#[derive(Clone, Copy)]
enum Bead {
    /// An output (and anything that is not a plain port): a disc. It stays
    /// when a noodle leaves it, so there is always a nodule to branch
    /// another noodle off — a drag or a shift-join.
    Dot,
    /// An input, with the disc of whatever feeds it seated inside.
    Cup(Option<Color32>),
}

/// How big a cup, and the disc seated in it, are against a free disc.
const SEATED: f32 = 0.6;

/// A half-round cup at `s` with round lips, open along `out`, sized to hold
/// a disc of radius `r * SEATED`; its wall is as thick as a full cup's.
fn cup(painter: &egui::Painter, s: Pos2, out: Vec2, r: f32, color: Color32) {
    let (rc, w) = (r * 1.55 * SEATED, (r * 0.5).max(1.2));
    let back = (-out.y).atan2(-out.x);
    let half = std::f32::consts::FRAC_PI_2;
    let pts: Vec<Pos2> = (0..=20)
        .map(|k| {
            let a = back - half + 2.0 * half * k as f32 / 20.0;
            s + Vec2::new(a.cos(), a.sin()) * rc
        })
        .collect();
    let lips = [pts[0], pts[pts.len() - 1]];
    painter.add(egui::Shape::line(pts, Stroke::new(w, color)));
    for l in lips {
        painter.circle_filled(l, w * 0.5, color);
    }
}

/// The colour a socket wears, input or output alike: the pipe's, with a
/// matte socket white so it reads as a different kind of port at a glance.
/// Which END a socket is, the shape says (a cup is an input), so both ends
/// of one noodle wear one colour.
/// The colour input `i` of `kind` wears, socket and noodle alike: its pipe's,
/// except a MODULATE port, which is yellow — same pipe as a mask, opposite
/// contract (scales the effect rather than mixing it back).
pub fn input_pipe_color(kind: &NodeKind, i: usize) -> Option<Color32> {
    if kind.mod_input() == Some(i) {
        return Some(theme::wire_modulate());
    }
    pipe_color(kind.input_kind(i))
}

/// A picture carrying LESS than full float rgba — one channel, 8 bits, or
/// both — read off the node feeding a wire. It starts at a Load that
/// decodes less, and carries on through every node that keeps it: a node
/// that only moves, crops, picks or re-times pixels keeps both; one that
/// resamples (a Transform, a dissolving retime) still treats the lanes
/// alike, so it keeps the single channel but not the 8 bits. The wire then
/// says so: one thin strand for one channel (the fat cable's three
/// strands, inverted), dashes for 8 bits, and a small tag naming which.
/// A Bake fed one stores and plays it back the same way.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Reduced {
    pub single: engine::decode::SingleChannel,
    pub eight: bool,
    /// The 8-bit values reach the comp exactly as stored: the Load emits
    /// no input transform. Only then can a Bake keep them in 8 bits; after
    /// an sRGB decode they are linear values that 8 bits would band.
    pub coded: bool,
}

impl Reduced {
    pub fn of(graph: &EditorGraph, src: u64) -> Option<Reduced> {
        Self::walk(graph, src, 64)
    }

    fn walk(graph: &EditorGraph, src: u64, depth: u32) -> Option<Reduced> {
        use engine::graph::TimeInterp;
        if depth == 0 {
            return None;
        }
        let up = |i: usize| graph.input_of(src, i).and_then(|s| Self::walk(graph, s, depth - 1));
        // Keeps the single channel but not the 8 bits.
        let lanes = |r: Reduced| {
            r.single.is_on().then_some(Reduced { eight: false, coded: false, ..r })
        };
        match &graph.nodes.get(&src)?.kind {
            NodeKind::Read { single, eight_bit, input, path, .. } => (single.is_on() || *eight_bit)
                .then(|| Reduced {
                    single: *single,
                    eight: *eight_bit,
                    coded: *eight_bit
                        && crate::colorui::resolve_input(*input, path)
                            == engine::color::InputSpace::Working,
                }),
            // A Noise asked for less: its values reach the comp as drawn
            // (no input transform), so its 8 bits are codes a Bake can keep.
            NodeKind::Noise(n) => (n.single || n.eight_bit).then(|| Reduced {
                single: if n.single {
                    engine::decode::SingleChannel::Red
                } else {
                    engine::decode::SingleChannel::Off
                },
                eight: n.eight_bit,
                coded: n.eight_bit,
            }),
            // Moves, crops, picks or holds pixels: everything is kept.
            NodeKind::Dot { .. }
            | NodeKind::SoftCrop(_)
            | NodeKind::GraphBake { .. }
            | NodeKind::Slip(_)
            | NodeKind::Freeze(_)
            | NodeKind::TimeRange(_)
            | NodeKind::FlipFlop(_)
            | NodeKind::Mirror(_)
            | NodeKind::Turn(_)
            | NodeKind::Uncrop(_)
            | NodeKind::Write(_) => up(0),
            // A retime keeps the 8 bits only while it snaps to frames.
            NodeKind::Speed(engine::graph::SpeedParams { interp, .. })
            | NodeKind::RangeRemap(engine::graph::RangeRemapParams { interp, .. })
            | NodeKind::TimeRemap(engine::graph::TimeRemapParams { interp, .. }) => {
                let r = up(0)?;
                if *interp == TimeInterp::Nearest { Some(r) } else { lanes(r) }
            }
            // Resamples: the lanes stay alike, the values do not stay 8-bit.
            NodeKind::Transform(_) | NodeKind::CornerPin(_) | NodeKind::Reformat(_) => {
                lanes(up(0)?)
            }
            // A Switch shows one input at a time: it keeps what they all keep.
            NodeKind::Switch(_) => {
                let n = graph.nodes.get(&src)?.kind.input_count();
                let mut all = (0..n).filter(|i| graph.input_of(src, *i).is_some()).map(up);
                let mut acc = all.next()??;
                for r in all {
                    let r = r?;
                    if r.single != acc.single {
                        acc.single = engine::decode::SingleChannel::Off;
                    }
                    acc.eight &= r.eight;
                    acc.coded &= r.coded;
                }
                (acc.single.is_on() || acc.eight).then_some(acc)
            }
            _ => None,
        }
    }

    /// The wire's tag: the channel's letter and/or "8-bit".
    pub fn tag(self) -> String {
        use engine::decode::SingleChannel as S;
        let ch = match self.single {
            S::Off => "",
            S::Red => "R",
            S::Green => "G",
            S::Blue => "B",
            S::Alpha => "A",
            S::Luma => "Y",
        };
        match (ch.is_empty(), self.eight) {
            (false, true) => format!("{ch} · 8-bit"),
            (false, false) => ch.to_string(),
            _ => "8-bit".to_string(),
        }
    }
}

/// [`socket_color`] for input `i` of `kind`: a modulate socket is yellow.
pub fn input_socket_color(kind: &NodeKind, i: usize) -> Color32 {
    if kind.mod_input() == Some(i) {
        return theme::wire_modulate();
    }
    socket_color(kind.input_kind(i))
}

pub fn socket_color(kind: PortKind) -> Color32 {
    match kind {
        PortKind::Mask => theme::port_mask(),
        _ => pipe_color(kind).unwrap_or_else(theme::port_in),
    }
}

pub fn type_color(kind: &NodeKind) -> Color32 {
    use crate::theme as t;
    match kind {
        NodeKind::Read { .. } => t::node_read(),
        NodeKind::Transform(_) => t::node_transform(),
        NodeKind::Grade(_) => t::node_grade(),
        NodeKind::Blur(_) => t::node_blur(),
        NodeKind::Defocus(_) => t::node_defocus(),
        NodeKind::Glow(_) => t::node_glow(),
        NodeKind::Godrays(_) => t::node_godrays(),
        NodeKind::LensFlare(_) => t::node_lens_flare(),
        NodeKind::EdgeExtend(_) => t::node_edge_extend(),
        NodeKind::Glint(_) => t::node_glint(),
        NodeKind::EdgeDetect(_) => t::node_edge_detect(),
        NodeKind::Median(_) => t::node_median(),
        NodeKind::HighlightRebuild(_) => t::node_highlight_rebuild(),
        NodeKind::Convolve(_) => t::node_convolve(),
        NodeKind::Diffuse(_) => t::node_diffuse(),
        NodeKind::Blend { .. } => t::node_blend(),
        // A dot takes the colour of the WIRE it sits on, because that is
        // all it is — the eye should read it as a bend, not as an operator.
        NodeKind::Dot { mask, channels, data, text, vector, scene } => {
            if *scene {
                t::wire_scene()
            } else if *vector {
                t::wire_vector()
            } else if *text {
                t::wire_text()
            } else if *channels {
                Color32::from_rgb(0x3d, 0x8f, 0x8f)
            } else if *data {
                t::wire_data()
            } else if *mask {
                t::wire_mask()
            } else {
                t::wire()
            }
        }
        NodeKind::Mix(_) => t::node_mix(),
        NodeKind::Layer(_) => t::node_layer(),
        NodeKind::Dilate(_) => t::node_dilate(),
        NodeKind::MatteCombine(_) => t::node_matte_combine(),
        // The noise family shares one hue: the Noise node in the comp, and
        // every field node inside it.
        NodeKind::Noise(_)
        | NodeKind::Field(_)
        | NodeKind::GeoNode(engine::geo::GeoNode::Noise { .. }) => t::node_noise(),
        // The geometry family likewise: the 2dini node and everything in it.
        NodeKind::Geo(_) | NodeKind::GeoNode(_) => t::node_geo(),
        // And the SDF family.
        NodeKind::Sdf(_) | NodeKind::SdfNode(_) => t::node_sdf(),
        NodeKind::Blender(_) => t::node_blender(),
        NodeKind::GenAi(_) | NodeKind::GenAiStep(_) => t::node_genai(),
        NodeKind::RotoNode(engine::graph::RotoInner::Axis(_)) => t::node_scene3d(),
        NodeKind::RotoNode(_) => t::node_spline(),
        NodeKind::Prompt(_) => t::node_prompt(),
        NodeKind::Curve(_) => t::node_curve(),
        NodeKind::Mask => t::node_mask(),
        NodeKind::Extract(_) => t::node_extract(),
        NodeKind::LumaKey(_) => t::node_luma_key(),
        NodeKind::CornerPin(_) => t::node_corner_pin(),
        NodeKind::Stabilize(_) => t::node_stabilize(),
        NodeKind::CamTrack(_) => t::node_camera_track(),
        NodeKind::Paint(_) => t::node_paint(),
        NodeKind::Spline(_) => t::node_spline(),
        NodeKind::GraphBake { .. } => t::node_bake(),
        // Sources share the Read family of pinks; spatial nodes the blues;
        // colour nodes the purples; time nodes a cold steel.
        NodeKind::Grid(_) => t::node_grid(),
        NodeKind::Gradient(_) => t::node_gradient(),
        NodeKind::Blank(_) => t::node_blank(),
        NodeKind::Starfield(_) => t::node_starfield(),
        NodeKind::Planet(_) => t::node_planet(),
        NodeKind::Plasma(_) => t::node_plasma(),
        NodeKind::Particles(_) | NodeKind::ParticleNode(_) => t::node_particles(),
        NodeKind::GlintKernel(_) => t::node_glint_kernel(),
        NodeKind::BokehKernel(_) => t::node_bokeh_kernel(),
        NodeKind::Grain(_) => t::node_grain(),
        NodeKind::Reformat(_) => t::node_reformat(),
        NodeKind::ChannelSelect(_) => t::node_channel_select(),
        // The harp pair wears the fat cable's teal: same family, one
        // shade apart so bridge and receiver read as two ends of one thing.
        NodeKind::Breakout(_) => t::node_breakout(),
        NodeKind::Recombine(_) => t::node_recombine(),
        NodeKind::LoopIn | NodeKind::LoopOut(_) => t::node_loop(),
        NodeKind::Tracker(_) => t::node_tracker(),
        NodeKind::PlanarTrack(_) => t::node_planar(),
        NodeKind::Saturation(_) => t::node_saturation(),
        NodeKind::Contrast(_) => t::node_contrast(),
        NodeKind::Slip(_)
        | NodeKind::Speed(_)
        | NodeKind::TimeRemap(_)
        | NodeKind::RangeRemap(_)
        | NodeKind::TimeRange(_) => t::node_time(),
        NodeKind::Switch(_) => t::node_switch(),
        NodeKind::Premult { .. } => t::node_premult(),
        NodeKind::Shuffle(_) => t::node_shuffle(),
        NodeKind::Write(_) => t::node_write(),
        NodeKind::Mult(_) => t::node_mult(),
        NodeKind::ToMatte(_) => t::node_to_matte(),
        NodeKind::ToData(_) => t::node_to_data(),
        NodeKind::UvWarp(_) => t::node_uvwarp(),
        NodeKind::Env(_) => t::node_env(),
        NodeKind::Refract(_) => t::node_refract(),
        NodeKind::GridWarp(_) => t::node_grid_warp(),
        NodeKind::Warp(_) => t::node_warp(),
        NodeKind::ToRgba(_) => t::node_to_rgba(),
        NodeKind::SoftCrop(_) => t::node_soft_crop(),
        NodeKind::Uncrop(_) => t::node_uncrop(),
        NodeKind::Compare(_) => t::node_compare(),
        NodeKind::MotionBlur(_) => t::node_motion_blur(),
        NodeKind::Flow(_) => t::node_flow(),
        NodeKind::Rifetime(_) => t::node_rifetime(),
        NodeKind::Freeze(_) => t::node_freeze(),
        NodeKind::FrameFix(_) => t::node_framefix(),
        NodeKind::VectorBlur(_) => t::node_vector_blur(),
        NodeKind::Camera3D(_) => t::node_camera3d(),
        NodeKind::Card3D(_) => t::node_card3d(),
        NodeKind::Scene3D(_) => t::node_scene3d(),
        NodeKind::Render3D(_) => t::node_render3d(),
        NodeKind::Splat(_) => t::node_splat(),
        NodeKind::VectorWarp(_) => t::node_vector_warp(),
        NodeKind::ColorCompress(_) => t::node_color_compress(),
        NodeKind::HueRotate(_) => t::node_hue_rotate(),
        NodeKind::Lut(_) => t::node_lut(),
        NodeKind::Text(_) => t::node_text(),
        NodeKind::Burnin(_) => t::node_burnin(),
        NodeKind::Mirror(_) => t::node_mirror(),
        NodeKind::LensDistort(_) => t::node_lens_distort(),
        NodeKind::FlipFlop(_) => t::node_flip_flop(),
        NodeKind::Turn(_) => t::node_turn(),
        NodeKind::Gamma(_) => t::node_gamma(),
        NodeKind::ToneMap(_) => t::node_tonemap(),
        NodeKind::Expression(_) => t::node_expression(),
        NodeKind::Despill(_) => t::node_despill(),
        NodeKind::Lightwrap(_) => t::node_lightwrap(),
        NodeKind::Upscale(_) => t::node_upscale(),
        NodeKind::HdrCnn(_) => t::node_hdrcnn(),
        NodeKind::SeedVr(_) => t::node_seedvr(),
        NodeKind::Depth(_) => t::node_depth(),
        NodeKind::LtxHdr(_) => t::node_ltx_hdr(),
        NodeKind::ScreenKey(_) => t::node_screen_key(),
        NodeKind::CorridorKey(_) => t::node_corridor_key(),
        NodeKind::Sammie(_) => t::node_sammie(),
        // The Tinter takes the slot its predecessor had, so a colour
        // somebody set for "Tinter" before the prune is its colour again.
        NodeKind::ColorSlice(_) => t::node_tinter(),
        NodeKind::MatteIn => t::node_matte_in(),
        // STOPGAP for the colour-keyer lane so the tree compiles; replace
        // with that lane's own colour and glyph.
        NodeKind::Keyer(_) => t::node_keyer(),
        NodeKind::Cryptomatte(_) => t::node_cryptomatte(),
        NodeKind::Custom { .. } if crate::tools::control_of(kind).is_some() => {
            t::node_tool_control()
        }
        NodeKind::Custom { .. } if crate::tools::is_proxy(kind) => t::node_tool_input(),
        // A plugin's colour is its manifest's to declare — data, not theme.
        NodeKind::Custom { type_id, .. } => match engine::plugins::get(type_id) {
            Some(d) => Color32::from_rgb(d.color[0], d.color[1], d.color[2]),
            None => t::node_plugin_missing(),
        },
        // Muted while the shader does not compile: the node is passing
        // through, and it should look like it is.
        NodeKind::Shader(p) => {
            if p.def().ok() {
                t::node_shader()
            } else {
                t::node_shader_broken()
            }
        }
    }
}

/// A merge bar's slab (Extract, Layer), painted under its glyph. Returns the
/// radius the glyph shrinks to so it sits inside the slab; `None` for every
/// other kind. `scale`: screen points per graph unit.
pub(crate) fn paint_bar(
    painter: &egui::Painter,
    kind: &NodeKind,
    sp: Pos2,
    scale: f32,
    fill: Color32,
    stroke: Stroke,
) -> Option<f32> {
    let (h, dx) = crate::breakout::is_bar(kind).then(|| crate::breakout::wide_rect(kind)).flatten()?;
    let half = h * scale;
    let c = sp + Vec2::new(dx * scale, 0.0);
    painter.rect(Rect::from_center_size(c, half * 2.0), half.y, fill, stroke, egui::StrokeKind::Middle);
    Some(half.y * 0.8)
}

/// A node's body: the shape its kind wears, centred on `sp` at radius `r`
/// (`scale`: screen points per graph unit, for the wide kinds). Returns where
/// the "+" went on a kind that grows a row from the node itself (Breakout,
/// Layer, Extract), for the caller to hit-test. A Read's TV is not here: it
/// needs the frame.
///
/// The colour designer draws its node icons from a copy of this
/// (`colors/src/glyphs.rs`); `glyph_parity` fails when the two differ.
/// How far out (in glyph radii) a click still throws a Switch's lever.
const SWITCH_HIT: f32 = 0.8;

/// The lever angle for input `i` of `n`: inputs splay across the TOP, so
/// the lever sweeps the upper half, 3/4 pi (up-left, input 0) to 1/4 pi.
pub(crate) fn switch_angle(i: usize, n: usize) -> f32 {
    let t = if n > 1 { i as f32 / (n - 1) as f32 } else { 0.5 };
    std::f32::consts::PI * (0.75 - 0.5 * t)
}

/// Which input a click at `p` on a Switch glyph (centre `c`, radius `r`)
/// throws the lever to: in the upper half and off the hub it is the input
/// whose stud is nearest the click's direction; on the hub or the lower
/// half it is the next input after `live`, wrapping.
pub(crate) fn switch_pick(c: Pos2, r: f32, n: usize, live: usize, p: Pos2) -> usize {
    let n = n.max(1);
    let d = p - c;
    if d.length() < r * 0.3 || d.y > -r * 0.1 {
        return (live + 1) % n;
    }
    let a = (-d.y).atan2(d.x);
    (0..n)
        .min_by(|&i, &j| {
            let di = (switch_angle(i, n) - a).abs();
            let dj = (switch_angle(j, n) - a).abs();
            di.total_cmp(&dj)
        })
        .unwrap_or(0)
}

pub(crate) fn paint_body(
    painter: &egui::Painter,
    kind: &NodeKind,
    sp: Pos2,
    r: f32,
    scale: f32,
    fill: Color32,
    stroke: Stroke,
) -> Option<Rect> {
    let mut plus = None;
    // Boring: one disc for everything but the bars (merge bars, the harp
    // pair, a multi-input bar) and a dot, which is a bead on a wire.
    if boring() && (boring_shape(kind) || matches!(kind, NodeKind::Read { .. })) {
        let half = if matches!(kind, NodeKind::Read { .. }) { BORING_READ_HALF } else { BORING_HALF } * scale;
        let rad = if matches!(kind, NodeKind::Read { .. }) { half.y * 0.3 } else { half.y };
        painter.rect(Rect::from_center_size(sp, half * 2.0), rad, fill, stroke, egui::StrokeKind::Middle);
        // A Switch still throws its lever from a click on the body.
        if matches!(kind, NodeKind::Switch(_)) {
            plus = Some(Rect::from_center_size(sp, Vec2::splat(r * 2.0 * SWITCH_HIT)));
        }
        return plus;
    }
    match kind {
        // The harp pair: wide slabs, their width the strings' span —
        // the node IS its comb, so how much it covers reads at a
        // glance. The receiver is the same slab a shade darker,
        // hanging under it.
        NodeKind::Breakout(_) | NodeKind::Recombine(_) => {
            let (h, dx) = crate::breakout::wide_rect(kind)
                .unwrap_or((Vec2::splat(R), 0.0));
            let half = h * scale;
            let c = sp + Vec2::new(dx * scale, 0.0);
            let body = Rect::from_center_size(c, half * 2.0);
            painter.rect(body, half.y, fill, stroke, egui::StrokeKind::Middle);
            // The breakout's + : add an AOV straight off the bar —
            // it opens the upstream layer list without a trip to
            // the properties rack.
            if matches!(kind, NodeKind::Breakout(_)) && half.y > 5.0 {
                let pc = Pos2::new(body.max.x - half.y * 1.1, c.y);
                let pr = (half.y * 0.5).clamp(4.0, 9.0);
                let st = Stroke::new(
                    (pr * 0.3).max(1.0),
                    Color32::from_gray(0x14).gamma_multiply(0.9),
                );
                painter.line_segment(
                    [pc - Vec2::new(pr, 0.0), pc + Vec2::new(pr, 0.0)],
                    st,
                );
                painter.line_segment(
                    [pc - Vec2::new(0.0, pr), pc + Vec2::new(0.0, pr)],
                    st,
                );
                plus = Some(Rect::from_center_size(pc, Vec2::splat(pr * 3.0)));
            }
        }
        NodeKind::Switch(sw) => {
            // A points-lever: the body plus a stub aimed at the live
            // input, so which way it is thrown reads at a glance.
            painter.circle(sp, r * 0.8, fill, stroke);
            let n_in = sw.n().max(1) as f32;
            // Fan the inputs across the top, same order they splay.
            let t = if n_in > 1.0 {
                sw.live() as f32 / (n_in - 1.0)
            } else {
                0.5
            };
            // Inputs splay across the TOP, so the lever sweeps the
            // upper half: 3/4 pi (up-left, input 0) to 1/4 pi.
            let a = std::f32::consts::PI * (0.75 - 0.5 * t);
            painter.line_segment(
                [sp, sp + Vec2::new(a.cos(), -a.sin()) * r * 0.95],
                Stroke::new((r * 0.22).max(1.2), theme::node_ink_light()),
            );
            // A stud per input on the rim: the positions the lever can be
            // thrown to. A click on the glyph throws it (`switch_pick`).
            let n = sw.n();
            for i in 0..n {
                let a = switch_angle(i, n);
                let ink = theme::node_ink_light();
                let ink = if i == sw.live() { ink } else { ink.gamma_multiply(0.35) };
                painter.circle_filled(
                    sp + Vec2::new(a.cos(), -a.sin()) * r * 0.8,
                    (r * 0.11).max(1.0),
                    ink,
                );
            }
            plus = Some(Rect::from_center_size(sp, Vec2::splat(r * 2.0 * SWITCH_HIT)));
        }
        NodeKind::Read { .. } => {
            // Triangle pointing down: a source funneling into the comp.
            let pts = vec![
                Pos2::new(sp.x - r, sp.y - r * 0.7),
                Pos2::new(sp.x + r, sp.y - r * 0.7),
                Pos2::new(sp.x, sp.y + r),
            ];
            painter.add(egui::Shape::convex_polygon(pts, fill, stroke));
        }
        NodeKind::GraphBake { .. } => {
            // Disk platter: a cached point in the graph.
            let rr = r * 0.85;
            painter.rect(
                Rect::from_center_size(sp, Vec2::new(rr * 2.0, rr * 1.6)),
                rr * 0.35,
                fill,
                stroke,
                egui::StrokeKind::Outside,
            );
            let st = Stroke::new((r * 0.16).max(1.0), theme::node_ink());
            painter.line_segment(
                [
                    Pos2::new(sp.x - rr * 0.6, sp.y),
                    Pos2::new(sp.x + rr * 0.6, sp.y),
                ],
                st,
            );
        }
        NodeKind::Shader(_) => {
            // Angle brackets: source code, and distinct from the
            // plugin hexagon because the source lives IN the setup.
            let w = r * 0.75;
            for s in [-1.0f32, 1.0] {
                painter.add(egui::Shape::line(
                    vec![
                        Pos2::new(sp.x + s * w, sp.y - r * 0.8),
                        Pos2::new(sp.x + s * r * 0.25, sp.y),
                        Pos2::new(sp.x + s * w, sp.y + r * 0.8),
                    ],
                    stroke,
                ));
            }
        }
        NodeKind::Custom { .. } => {
            // Hexagon: the plugin shape.
            let pts = (0..6)
                .map(|k| {
                    let a = k as f32 / 6.0 * std::f32::consts::TAU
                        + std::f32::consts::FRAC_PI_6;
                    Pos2::new(sp.x + a.cos() * r, sp.y + a.sin() * r)
                })
                .collect();
            painter.add(egui::Shape::convex_polygon(pts, fill, stroke));
        }
        NodeKind::Dot { .. } => {
            // A bead on the wire, no bigger than a socket: it is
            // punctuation, and punctuation that draws the eye is a
            // typo. No stalks are drawn for it either (below).
            painter.circle_filled(sp, (r * 0.42).max(2.0), fill);
        }
        NodeKind::Blend { .. } => {
            // Triangle pointing up: two streams funnel to one.
            let pts = vec![
                Pos2::new(sp.x, sp.y - r),
                Pos2::new(sp.x + r, sp.y + r * 0.7),
                Pos2::new(sp.x - r, sp.y + r * 0.7),
            ];
            painter.add(egui::Shape::convex_polygon(pts, fill, stroke));
        }
        // First: a Noise node inside a drawing wears the noise glyph.
        NodeKind::Noise(_)
        | NodeKind::Field(_)
        | NodeKind::GeoNode(engine::geo::GeoNode::Noise { .. }) => {
            // A filled body FIRST: the waves below are drawn in the
            // outline stroke, which is near-black until the node is
            // selected, so on their own they are invisible.
            painter.rect(
                Rect::from_center_size(sp, Vec2::splat(r * 1.7)),
                r * 0.3,
                fill,
                stroke,
                egui::StrokeKind::Middle,
            );
            // Three stacked waves: a field, not a picture.
            for k in 0..3i32 {
                let y = sp.y + (k - 1) as f32 * r * 0.55;
                let mut pts = Vec::new();
                for i in 0..=8 {
                    let t = i as f32 / 8.0;
                    pts.push(Pos2::new(
                        sp.x - r + t * r * 2.0,
                        y + (t * std::f32::consts::TAU + k as f32).sin() * r * 0.18,
                    ));
                }
                painter.add(egui::Shape::line(pts, stroke));
            }
        }
        NodeKind::Sdf(_) | NodeKind::SdfNode(_) => {
            crate::sdfui::glyph(&painter, sp, r, fill, stroke)
        }
        NodeKind::Geo(_) | NodeKind::GeoNode(_) => {
            // A filled body, then a pentagon outline: geometry, not
            // a picture and not a field.
            painter.rect(
                Rect::from_center_size(sp, Vec2::splat(r * 1.7)),
                r * 0.3,
                fill,
                stroke,
                egui::StrokeKind::Middle,
            );
            let pts: Vec<Pos2> = (0..5)
                .map(|i| {
                    let a = -std::f32::consts::FRAC_PI_2
                        + i as f32 * std::f32::consts::TAU / 5.0;
                    Pos2::new(sp.x + a.cos() * r * 0.6, sp.y + a.sin() * r * 0.6)
                })
                .collect();
            painter.add(egui::Shape::closed_line(pts, stroke));
        }
        NodeKind::MatteCombine(_) => {
            // Two overlapping discs: the Venn diagram every one of
            // its ops is a region of.
            painter.circle(Pos2::new(sp.x - r * 0.36, sp.y), r * 0.72, fill, stroke);
            painter.circle_stroke(Pos2::new(sp.x + r * 0.36, sp.y), r * 0.72, stroke);
        }
        NodeKind::Dilate(_) => crate::nodes::dilate::glyph(&painter, sp, r, fill, stroke),
        NodeKind::Layer(_) => {
            // Three offset sheets: a stack seen slightly from above.
            for k in [1.0f32, 0.0, -1.0] {
                let c = Pos2::new(sp.x + k * r * 0.28, sp.y + k * r * 0.34);
                painter.rect(
                    Rect::from_center_size(c, Vec2::new(r * 1.35, r * 0.78)),
                    1.0,
                    fill,
                    stroke,
                    egui::StrokeKind::Outside,
                );
            }
            // No +: a noodle dropped on the bar is a new layer.
        }
        NodeKind::Mix(_) => {
            // Two triangles meeting nose to nose: the crossfade.
            for s in [-1.0f32, 1.0] {
                painter.add(egui::Shape::convex_polygon(
                    vec![
                        Pos2::new(sp.x, sp.y),
                        Pos2::new(sp.x + r * s, sp.y - r * 0.8),
                        Pos2::new(sp.x + r * s, sp.y + r * 0.8),
                    ],
                    fill,
                    stroke,
                ));
            }
        }
        NodeKind::Transform(_) => {
            let half = r * 0.82;
            painter.rect(
                Rect::from_center_size(sp, Vec2::splat(half * 2.0)),
                2.0,
                fill,
                stroke,
                egui::StrokeKind::Outside,
            );
        }
        NodeKind::Blur(_) => crate::nodes::blur::glyph(&painter, sp, r, fill, stroke),
        NodeKind::Grade(_) => {
            painter.circle(sp, r * 0.9, fill, stroke);
        }
        NodeKind::Defocus(_) => {
            // Hexagon: a bokeh iris.
            let pts: Vec<Pos2> = (0..6)
                .map(|k| {
                    let a = k as f32 * std::f32::consts::TAU / 6.0
                        - std::f32::consts::TAU / 12.0;
                    Pos2::new(sp.x + a.cos() * r, sp.y + a.sin() * r)
                })
                .collect();
            painter.add(egui::Shape::convex_polygon(pts, fill, stroke));
        }
        NodeKind::Glow(_) => crate::nodes::glow::glyph(&painter, sp, r, fill, stroke),
        NodeKind::Godrays(_) => crate::nodes::godrays::glyph(&painter, sp, r, fill, stroke),
        NodeKind::LensFlare(_) => crate::nodes::lensflare::glyph(&painter, sp, r, fill, stroke),
        NodeKind::EdgeExtend(_) => crate::nodes::edgeextend::glyph(&painter, sp, r, fill, stroke),
        NodeKind::Glint(_) => crate::nodes::glint::glyph(&painter, sp, r, fill, stroke),
        NodeKind::Convolve(_) => crate::nodes::convolve::glyph(&painter, sp, r, fill, stroke),
        NodeKind::Median(_) => crate::nodes::median::glyph(&painter, sp, r, fill, stroke),
        NodeKind::HighlightRebuild(_) => crate::nodes::highlightrebuild::glyph(&painter, sp, r, fill, stroke),
        NodeKind::Stabilize(_) => crate::nodes::stabilize::glyph(&painter, sp, r, fill, stroke),
        NodeKind::CamTrack(_) => crate::nodes::camtrack::glyph(&painter, sp, r, fill, stroke),
        NodeKind::EdgeDetect(_) => crate::nodes::edgedetect::glyph(&painter, sp, r, fill, stroke),
        NodeKind::Diffuse(_) => crate::nodes::diffuse::glyph(&painter, sp, r, fill, stroke),
        NodeKind::CornerPin(_) => {
            // Trapezoid: a plane in perspective.
            let pts = vec![
                Pos2::new(sp.x - r * 0.55, sp.y - r * 0.8),
                Pos2::new(sp.x + r * 0.55, sp.y - r * 0.8),
                Pos2::new(sp.x + r, sp.y + r * 0.8),
                Pos2::new(sp.x - r, sp.y + r * 0.8),
            ];
            painter.add(egui::Shape::convex_polygon(pts, fill, stroke));
        }
        NodeKind::Spline(_) => {
            // Ring: a matte shape.
            painter.circle(sp, r * 0.85, Color32::TRANSPARENT, stroke);
            painter.circle_stroke(
                sp,
                r * 0.55,
                Stroke::new((r * 0.3).max(1.5), fill),
            );
        }
        NodeKind::Paint(_) => {
            // Brush tip: kite pointing down.
            let pts = vec![
                Pos2::new(sp.x, sp.y + r),
                Pos2::new(sp.x - r * 0.7, sp.y + r * 0.1),
                Pos2::new(sp.x, sp.y - r),
                Pos2::new(sp.x + r * 0.7, sp.y + r * 0.1),
            ];
            painter.add(egui::Shape::convex_polygon(pts, fill, stroke));
        }
        NodeKind::Curve(_) => {
            // Rounded square with an S-curve through it.
            painter.rect(
                Rect::from_center_size(sp, Vec2::splat(r * 1.6)),
                r * 0.3,
                fill,
                stroke,
                egui::StrokeKind::Outside,
            );
            painter.add(egui::epaint::CubicBezierShape::from_points_stroke(
                [
                    Pos2::new(sp.x - r * 0.6, sp.y + r * 0.5),
                    Pos2::new(sp.x + r * 0.5, sp.y + r * 0.5),
                    Pos2::new(sp.x - r * 0.5, sp.y - r * 0.5),
                    Pos2::new(sp.x + r * 0.6, sp.y - r * 0.5),
                ],
                false,
                Color32::TRANSPARENT,
                Stroke::new((r * 0.18).max(1.0), theme::node_ink()),
            ));
        }
        NodeKind::Extract(p) => {
            // A Mask glyph sliced into its outputs: one wedge of
            // the lower half per pair, gaps between them.
            painter.circle_stroke(sp, r * 0.85, Stroke::new(2.0, fill));
            let k = p.pairs();
            let gap = 0.12 / k as f32;
            for j in 0..k {
                let a0 = std::f32::consts::PI * (j as f32 / k as f32 + gap);
                let a1 = std::f32::consts::PI * ((j + 1) as f32 / k as f32 - gap);
                let mut pts = vec![sp];
                pts.extend((0..=6).map(|s| {
                    let a = a0 + (a1 - a0) * s as f32 / 6.0;
                    Pos2::new(sp.x - a.cos() * r * 0.85, sp.y + a.sin() * r * 0.85)
                }));
                painter.add(egui::Shape::convex_polygon(pts, fill, Stroke::NONE));
            }
            // The +, off the right flank: one more pair, no trip
            // to the properties rack (the Breakout's gesture).
            if r > 5.0 && (p.pairs() as u32) < engine::graph::EXTRACT_MAX {
                // On the bar, at its right end.
                let pc = match crate::breakout::wide_rect(kind) {
                    Some((h, dx)) => sp + Vec2::new((dx + h.x) * scale - r * 1.1, 0.0),
                    None => sp + Vec2::new(r * 1.45, 0.0),
                };
                let pr = (r * 0.28).clamp(3.5, 8.0);
                let st = Stroke::new((pr * 0.3).max(1.0), theme::node_ink_light());
                painter.line_segment([pc - Vec2::new(pr, 0.0), pc + Vec2::new(pr, 0.0)], st);
                painter.line_segment([pc - Vec2::new(0.0, pr), pc + Vec2::new(0.0, pr)], st);
                plus = Some(Rect::from_center_size(pc, Vec2::splat(pr * 3.0)));
            }
        }
        NodeKind::Mask => {
            // Circle, lower half solid: image cut by coverage.
            painter.circle_stroke(sp, r * 0.85, Stroke::new(2.0, fill));
            let pts: Vec<Pos2> = (0..=12)
                .map(|k| {
                    let a = std::f32::consts::PI * k as f32 / 12.0;
                    Pos2::new(sp.x - a.cos() * r * 0.85, sp.y + a.sin() * r * 0.85)
                })
                .collect();
            painter.add(egui::Shape::convex_polygon(pts, fill, stroke));
        }
        NodeKind::LumaKey(_) => {
            // Crescent: bright circle bitten by the background.
            painter.circle_filled(sp, r * 0.85, fill);
            painter.circle_filled(
                Pos2::new(sp.x + r * 0.45, sp.y - r * 0.25),
                r * 0.6,
                theme::graph_bg(),
            );
        }
        NodeKind::Keyer(_) => {
            // The node itself, drawn: a selection blob sitting off
            // centre inside the colour circle.
            painter.circle_stroke(sp, r * 0.85, Stroke::new((r * 0.16).max(1.0), fill));
            let c = Pos2::new(sp.x + r * 0.2, sp.y - r * 0.16);
            let pts: Vec<Pos2> = (0..14)
                .map(|k| {
                    let a = std::f32::consts::TAU * k as f32 / 14.0 + 0.6;
                    Pos2::new(c.x + a.cos() * r * 0.46, c.y + a.sin() * r * 0.26)
                })
                .collect();
            painter.add(egui::Shape::convex_polygon(pts, fill, stroke));
        }
        NodeKind::Cryptomatte(_) => {
            // An id map: a row of tiles, one of them picked (filled).
            let s = r * 0.5;
            for (k, dx) in [-1.1f32, 0.0, 1.1].into_iter().enumerate() {
                let c = Pos2::new(sp.x + dx * s, sp.y);
                let rr = Rect::from_center_size(c, Vec2::splat(s * 0.95));
                if k == 1 {
                    painter.rect_filled(rr, 1.0, fill);
                } else {
                    painter.rect_stroke(rr, 1.0, stroke, egui::StrokeKind::Outside);
                }
            }
        }
        NodeKind::MatteIn => {
            // ToMatte's mirror: the mask colour enters from the left
            // and the square comes out an image.
            let h = r * 0.8;
            painter.rect(
                Rect::from_center_size(sp, Vec2::splat(h * 2.0)),
                1.0,
                fill,
                stroke,
                egui::StrokeKind::Outside,
            );
            let pts = vec![
                Pos2::new(sp.x - h, sp.y - h),
                Pos2::new(sp.x, sp.y - h),
                Pos2::new(sp.x, sp.y + h),
                Pos2::new(sp.x - h, sp.y + h),
            ];
            painter.add(egui::Shape::convex_polygon(
                pts,
                theme::wire_mask().gamma_multiply(0.9),
                Stroke::NONE,
            ));
        }
        NodeKind::SoftCrop(_) => {
            crate::nodes::softcrop::glyph(&painter, sp, r, fill, stroke)
        }
        NodeKind::Uncrop(_) => crate::nodes::uncrop::glyph(&painter, sp, r, fill, stroke),
        NodeKind::Compare(_) => crate::nodes::compare::glyph(&painter, sp, r, fill, stroke),
        NodeKind::MotionBlur(_) => {
            crate::nodes::motionblur::glyph(&painter, sp, r, fill, stroke)
        }
        NodeKind::Flow(_) => crate::nodes::flow::glyph(&painter, sp, r, fill, stroke),
        NodeKind::Rifetime(_) => crate::nodes::rifetime::glyph(&painter, sp, r, fill, stroke),
        NodeKind::Freeze(_) => crate::nodes::freeze::glyph(&painter, sp, r, fill, stroke),
        NodeKind::FrameFix(_) => crate::nodes::framefix::glyph(&painter, sp, r, fill, stroke),
        NodeKind::VectorBlur(_) => crate::nodes::vectorblur::glyph(&painter, sp, r, fill, stroke),
        NodeKind::Camera3D(_) => crate::nodes::scene3d::camera_glyph(&painter, sp, r, fill, stroke),
        NodeKind::Card3D(_) => crate::nodes::scene3d::card_glyph(&painter, sp, r, fill, stroke),
        NodeKind::Scene3D(_) => crate::nodes::scene3d::scene_glyph(&painter, sp, r, fill, stroke),
        NodeKind::Render3D(_) => crate::nodes::scene3d::render_glyph(&painter, sp, r, fill, stroke),
        NodeKind::VectorWarp(_) => crate::nodes::vectorwarp::glyph(&painter, sp, r, fill, stroke),
        NodeKind::ColorCompress(_) => {
            crate::nodes::colorcompress::glyph(&painter, sp, r, fill, stroke)
        }
        NodeKind::HueRotate(_) => {
            crate::nodes::huerotate::glyph(&painter, sp, r, fill, stroke)
        }
        NodeKind::Lut(_) => crate::nodes::lut::glyph(&painter, sp, r, fill, stroke),
        NodeKind::Text(_) => crate::nodes::text::glyph(&painter, sp, r, fill, stroke),
        NodeKind::Burnin(_) => crate::nodes::burnin::glyph(&painter, sp, r, fill, stroke),
        NodeKind::Mirror(_) => crate::nodes::mirror::glyph(&painter, sp, r, fill, stroke),
        NodeKind::LensDistort(_) => crate::nodes::lensdistort::glyph(&painter, sp, r, fill, stroke),
        NodeKind::FlipFlop(_) => crate::nodes::flipflop::glyph(&painter, sp, r, fill, stroke),
        NodeKind::Turn(_) => crate::nodes::turn::glyph(&painter, sp, r, fill, stroke),
        NodeKind::Gamma(_) => crate::nodes::gamma::glyph(&painter, sp, r, fill, stroke),
        NodeKind::ToneMap(_) => crate::nodes::tonemap::glyph(&painter, sp, r, fill, stroke),
        NodeKind::Expression(_) => {
            crate::nodes::expression::glyph(&painter, sp, r, fill, stroke)
        }
        NodeKind::Despill(_) => crate::nodes::despill::glyph(&painter, sp, r, fill, stroke),
        NodeKind::Lightwrap(_) => crate::nodes::lightwrap::glyph(&painter, sp, r, fill, stroke),
        NodeKind::Upscale(_) => crate::nodes::upscale::glyph(&painter, sp, r, fill, stroke),
        NodeKind::HdrCnn(_) => crate::nodes::hdrcnn::glyph(&painter, sp, r, fill, stroke),
        NodeKind::SeedVr(_) => crate::nodes::seedvr::glyph(&painter, sp, r, fill, stroke),
        NodeKind::Depth(_) => crate::nodes::depth::glyph(&painter, sp, r, fill, stroke),
        NodeKind::LtxHdr(_) => crate::nodes::ltxhdr::glyph(&painter, sp, r, fill, stroke),
        NodeKind::ScreenKey(_) => {
            crate::nodes::screenkey::glyph(&painter, sp, r, fill, stroke)
        }
        NodeKind::Sammie(_) => crate::nodes::sammie::glyph(&painter, sp, r, fill, stroke),
        NodeKind::CorridorKey(_) => {
            crate::nodes::corridorkey::glyph(&painter, sp, r, fill, stroke)
        }
        NodeKind::ColorSlice(_) => {
            crate::nodes::colorslice::glyph(&painter, sp, r, fill, stroke)
        }
        NodeKind::ToData(_) => {
            crate::nodes::todata::glyph(&painter, sp, r, fill, stroke)
        }
        NodeKind::UvWarp(_) => crate::nodes::uvwarp::glyph(&painter, sp, r, fill, stroke),
        NodeKind::Env(_) => crate::nodes::env::glyph(&painter, sp, r, fill, stroke),
        NodeKind::Refract(_) => crate::nodes::refract::glyph(&painter, sp, r, fill, stroke),
        NodeKind::GridWarp(_) => crate::nodes::gridwarp::glyph(&painter, sp, r, fill, stroke),
        NodeKind::Warp(_) => crate::nodes::warp::glyph(&painter, sp, r, fill, stroke),
        NodeKind::ToRgba(_) => {
            crate::nodes::torgba::glyph(&painter, sp, r, fill, stroke)
        }
        NodeKind::ToMatte(_) => {
            // Half image, half matte: a square whose right side is
            // cut away into the mask colour it outputs.
            let h = r * 0.8;
            painter.rect(
                Rect::from_center_size(sp, Vec2::splat(h * 2.0)),
                1.0,
                fill,
                stroke,
                egui::StrokeKind::Outside,
            );
            let pts = vec![
                Pos2::new(sp.x, sp.y - h),
                Pos2::new(sp.x + h, sp.y - h),
                Pos2::new(sp.x + h, sp.y + h),
                Pos2::new(sp.x, sp.y + h),
            ];
            painter.add(egui::Shape::convex_polygon(
                pts,
                theme::wire_mask().gamma_multiply(0.9),
                Stroke::NONE,
            ));
        }
        NodeKind::Mult(_) => {
            // A circle with a multiplication cross through it: the
            // Grade's disc, pared back to one operation.
            painter.circle(sp, r * 0.8, fill, stroke);
            let st = Stroke::new((r * 0.16).max(1.0), theme::node_ink());
            let d = r * 0.38;
            painter.line_segment(
                [Pos2::new(sp.x - d, sp.y - d), Pos2::new(sp.x + d, sp.y + d)],
                st,
            );
            painter.line_segment(
                [Pos2::new(sp.x - d, sp.y + d), Pos2::new(sp.x + d, sp.y - d)],
                st,
            );
        }
        NodeKind::Gradient(_) => crate::nodes::gradient::glyph(&painter, sp, r, fill, stroke),
        NodeKind::Blank(_) => crate::nodes::blank::glyph(&painter, sp, r, fill, stroke),
        NodeKind::Starfield(_) => crate::nodes::starfield::glyph(&painter, sp, r, fill, stroke),
        NodeKind::Splat(_) => crate::nodes::splat::glyph(&painter, sp, r, fill, stroke),
        NodeKind::Planet(_) => crate::nodes::planet::glyph(&painter, sp, r, fill, stroke),
        NodeKind::Plasma(_) => crate::nodes::plasma::glyph(&painter, sp, r, fill, stroke),
        NodeKind::Particles(_) | NodeKind::ParticleNode(_) => {
            crate::nodes::particles::glyph(&painter, sp, r, fill, stroke)
        }
        NodeKind::GlintKernel(_) => crate::nodes::glintkernel::glyph(&painter, sp, r, fill, stroke),
        NodeKind::BokehKernel(_) => crate::nodes::bokehkernel::glyph(&painter, sp, r, fill, stroke),
        NodeKind::Grain(_) => crate::nodes::grain::glyph(&painter, sp, r, fill, stroke),
        NodeKind::Blender(_) => crate::blenderui::glyph(&painter, sp, r, fill, stroke),
        NodeKind::GenAi(_) | NodeKind::GenAiStep(_) => crate::genaiui::glyph(&painter, sp, r, fill, stroke),
        NodeKind::RotoNode(engine::graph::RotoInner::Axis(_)) => {
            crate::nodes::scene3d::scene_glyph(&painter, sp, r, fill, stroke)
        }
        NodeKind::RotoNode(_) => crate::nodes::scene3d::card_glyph(&painter, sp, r, fill, stroke),
        NodeKind::Prompt(_) => crate::nodes::prompt::glyph(&painter, sp, r, fill, stroke),
        NodeKind::Grid(_) => {
            // Source triangle ruled into cells: a generated plate.
            let pts = vec![
                Pos2::new(sp.x - r, sp.y - r * 0.7),
                Pos2::new(sp.x + r, sp.y - r * 0.7),
                Pos2::new(sp.x, sp.y + r),
            ];
            painter.add(egui::Shape::convex_polygon(pts, fill, stroke));
            let st = Stroke::new((r * 0.12).max(0.8), theme::node_ink());
            painter.line_segment(
                [
                    Pos2::new(sp.x - r * 0.45, sp.y - r * 0.2),
                    Pos2::new(sp.x + r * 0.45, sp.y - r * 0.2),
                ],
                st,
            );
            painter.line_segment(
                [
                    Pos2::new(sp.x, sp.y - r * 0.6),
                    Pos2::new(sp.x, sp.y + r * 0.5),
                ],
                st,
            );
        }
        NodeKind::Reformat(_) => {
            // Small frame nested in a big one: a change of format.
            painter.rect(
                Rect::from_center_size(sp, egui::vec2(r * 1.9, r * 1.3)),
                1.0,
                fill,
                stroke,
                egui::StrokeKind::Outside,
            );
            painter.rect_stroke(
                Rect::from_center_size(sp, egui::vec2(r * 1.0, r * 0.7)),
                0.0,
                Stroke::new((r * 0.14).max(1.0), theme::node_ink()),
                egui::StrokeKind::Inside,
            );
        }
        NodeKind::LoopIn => {
            // A tap: the ring something loops back INTO.
            painter.circle_stroke(sp, r * 0.8, stroke);
            painter.circle(sp, r * 0.28, fill, stroke);
        }
        NodeKind::LoopOut(_) => {
            // A cycle: three arc dashes chasing each other.
            for k in 0..3 {
                let a0 = k as f32 * std::f32::consts::TAU / 3.0;
                let pts: Vec<Pos2> = (0..8)
                    .map(|j| {
                        let a = a0 + j as f32 * 0.09;
                        Pos2::new(
                            sp.x + a.cos() * r * 0.8,
                            sp.y + a.sin() * r * 0.8,
                        )
                    })
                    .collect();
                painter.add(egui::Shape::line(pts, stroke));
            }
            painter.circle(sp, r * 0.28, fill, stroke);
        }
        NodeKind::ChannelSelect(_) => {
            // Many strands in, one out: the fat cable narrowing.
            for k in [-1.0f32, 0.0, 1.0] {
                painter.line_segment(
                    [
                        Pos2::new(sp.x + k * r * 0.55, sp.y - r * 0.9),
                        Pos2::new(sp.x, sp.y + r * 0.1),
                    ],
                    stroke,
                );
            }
            painter.line_segment(
                [Pos2::new(sp.x, sp.y + r * 0.1), Pos2::new(sp.x, sp.y + r * 0.9)],
                stroke,
            );
            painter.circle(sp, r * 0.28, fill, stroke);
        }
        NodeKind::PlanarTrack(_) => {
            // A quad in perspective: the plane being followed.
            let st = Stroke::new((r * 0.18).max(1.4), fill);
            let q = [
                Pos2::new(sp.x - r * 0.9, sp.y - r * 0.5),
                Pos2::new(sp.x + r * 0.9, sp.y - r * 0.8),
                Pos2::new(sp.x + r * 0.6, sp.y + r * 0.8),
                Pos2::new(sp.x - r * 0.6, sp.y + r * 0.5),
            ];
            for k in 0..4 {
                painter.line_segment([q[k], q[(k + 1) % 4]], st);
            }
            painter.line_segment([q[0], q[2]], Stroke::new((r * 0.1).max(1.0), fill));
        }
        NodeKind::Tracker(_) => {
            // Crosshair in a box: the pattern being followed.
            painter.rect_stroke(
                Rect::from_center_size(sp, Vec2::splat(r * 1.6)),
                0.0,
                Stroke::new((r * 0.2).max(1.5), fill),
                egui::StrokeKind::Inside,
            );
            let st = Stroke::new((r * 0.14).max(1.0), fill);
            painter.line_segment(
                [Pos2::new(sp.x - r, sp.y), Pos2::new(sp.x + r, sp.y)],
                st,
            );
            painter.line_segment(
                [Pos2::new(sp.x, sp.y - r), Pos2::new(sp.x, sp.y + r)],
                st,
            );
        }
        NodeKind::Saturation(_) => crate::nodes::saturation::glyph(&painter, sp, r, fill, stroke),
        NodeKind::Contrast(_) => crate::nodes::contrast::glyph(&painter, sp, r, fill, stroke),
        NodeKind::Slip(_)
        | NodeKind::Speed(_)
        | NodeKind::TimeRemap(_)
        | NodeKind::RangeRemap(_)
        | NodeKind::TimeRange(_) => {
            // Clock face; a retime's hand is swept off the hour.
            painter.circle(sp, r * 0.85, fill, stroke);
            let st = Stroke::new((r * 0.16).max(1.0), theme::node_ink());
            painter.line_segment([sp, Pos2::new(sp.x, sp.y - r * 0.6)], st);
            let arm = if !matches!(kind, NodeKind::Slip(_)) {
                Pos2::new(sp.x + r * 0.55, sp.y + r * 0.3)
            } else {
                Pos2::new(sp.x + r * 0.5, sp.y)
            };
            painter.line_segment([sp, arm], st);
        }
        NodeKind::Write(_) => {
            // A down-triangle like a Read, inverted into a funnel
            // with a bar under it: the comp going to disk.
            let pts = vec![
                Pos2::new(sp.x - r, sp.y - r * 0.9),
                Pos2::new(sp.x + r, sp.y - r * 0.9),
                Pos2::new(sp.x, sp.y + r * 0.35),
            ];
            painter.add(egui::Shape::convex_polygon(pts, fill, stroke));
            painter.rect_filled(
                Rect::from_center_size(
                    Pos2::new(sp.x, sp.y + r * 0.75),
                    egui::vec2(r * 1.6, r * 0.36),
                ),
                1.0,
                fill,
            );
        }
        NodeKind::Shuffle(_) => {
            // Four stacked bars with two of them crossing: channels
            // going in one order and coming out another.
            painter.rect(
                Rect::from_center_size(sp, Vec2::splat(r * 1.6)),
                1.0,
                fill,
                stroke,
                egui::StrokeKind::Outside,
            );
            let st = Stroke::new((r * 0.13).max(0.8), theme::node_ink());
            let (x0, x1) = (sp.x - r * 0.55, sp.x + r * 0.55);
            for (k, dy) in [-0.45f32, 0.45].iter().enumerate() {
                let y = sp.y + dy * r;
                painter.line_segment(
                    [Pos2::new(x0, y), Pos2::new(x1, y)],
                    st,
                );
                let _ = k;
            }
            // The crossing pair.
            painter.line_segment(
                [Pos2::new(x0, sp.y - r * 0.15), Pos2::new(x1, sp.y + r * 0.15)],
                st,
            );
            painter.line_segment(
                [Pos2::new(x0, sp.y + r * 0.15), Pos2::new(x1, sp.y - r * 0.15)],
                st,
            );
        }
        NodeKind::Premult { premult } => {
            // Circle over a square: colour against its alpha. The
            // premult direction fills the disc, unpremult hollows it.
            painter.rect(
                Rect::from_center_size(sp, Vec2::splat(r * 1.5)),
                1.0,
                fill,
                stroke,
                egui::StrokeKind::Outside,
            );
            let ink = theme::node_ink();
            if *premult {
                painter.circle_filled(sp, r * 0.45, ink);
            } else {
                painter.circle_stroke(
                    sp,
                    r * 0.45,
                    Stroke::new((r * 0.16).max(1.0), ink),
                );
            }
        }
    }
    plus
}

/// Whose common sources a backtick pie is offering.
///
/// Two rosters, two keys, one instrument: the GLOBAL one answers anywhere
/// in the comp (the crypto read, the plate every region borrows), the
/// REGION one answers for the wash the view centre stands in. A pie with
/// no region to serve says so beside the cursor rather than not opening —
/// a key that silently does nothing teaches you it is dead.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PieScope {
    Global,
    Region(Option<u64>),
}

impl PieScope {
    /// The action whose key holds this pie open.
    fn act(self) -> crate::keymap::Act {
        match self {
            PieScope::Global => crate::keymap::Act::GraphSourcePie,
            PieScope::Region(_) => crate::keymap::Act::GraphRegionPie,
        }
    }
}

/// The plain-` pie: common node ops in one held orbit. A pick does not run
/// the op here — it lands in `ops_pick` and the op's own key site fires it
/// next frame, so the pie and the hotkey are the same code path.
const OPS_PIE: &[(crate::keymap::Act, &str)] = {
    use crate::keymap::Act;
    &[
        (Act::GraphGroup, "group"),
        (Act::GraphUnifySize, "unify size"),
        (Act::GraphTidy, "tidy"),
        (Act::GraphSwapInputs, "swap inputs"),
        (Act::GraphBypass, "bypass"),
        (Act::GraphFrame, "frame"),
        (Act::GraphRegionMenu, "region…"),
        (Act::GraphUngroup, "ungroup"),
        (Act::GraphSubwob, "subwob"),
        (Act::GraphBakeSelected, "bake"),
        (Act::GraphCacheSelected, "cache"),
    ]
};
/// The ops orbit sits farther out than the source pie: its slices are
/// read by label, not by a disc.
const OPS_PIE_R: f32 = 62.0;

/// The F1 pie's roster: the catalog ranked by how often each label has
/// been ADDED, catalog order breaking ties — before any history exists the
/// pie is simply the catalog's front rank, and every add anywhere (tab
/// search, the menus, the pie itself) tilts it toward this hand.
fn top_used(counts: &std::collections::BTreeMap<String, u32>) -> Vec<NodeEntry> {
    let mut v = node_catalog();
    v.sort_by_key(|e| std::cmp::Reverse(counts.get(e.label).copied().unwrap_or(0)));
    v.truncate(8);
    v
}

#[derive(Clone, Copy)]
struct Pie {
    at: Pos2,
    scope: PieScope,
    /// The one selected node this pie would register, if any.
    adding: Option<u64>,
}

/// One end of a wire, as a thing you can SELECT. Sockets were only ever
/// drag handles before; naming them lets a click land on one and `G` pick a
/// noodle up from it that outlives the gesture.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Port {
    /// (node, which output). Only a multi-band LumaKey has more than one.
    Out(u64, u32),
    In(u64, usize),
}

impl Port {
    fn node(self) -> u64 {
        match self {
            Port::Out(n, _) | Port::In(n, _) => n,
        }
    }
}

enum DragKind {
    /// Dragging a node (with the rest of the selection). `unjoin`: the drag
    /// began with SHIFT, which pulls the dragged set out of its backdrop
    /// and suppresses the join-on-drop at the end.
    Node {
        id: u64,
        unjoin: bool,
        /// Where every dragged node started — the M-swipe blend puts the
        /// dragged set back here when it hands the drag to the new Blend.
        from: Vec<(u64, Pos2)>,
        /// This drag already made its Blend: one merge per gesture, or the
        /// new Blend (born under the pointer, over the node it was swiped
        /// onto) would merge again on the very next frame.
        merged: bool,
        /// The pointer's whole travel since the press, in graph units. The
        /// set sits at `from + raw` plus whatever snap applies this frame,
        /// so a snap never accumulates into the drag.
        raw: Vec2,
    },
    Pan,
    /// A gesture inside a zoomed-in node's inner graph; the state is
    /// `innard_drag`.
    Innard,
    /// A drag on a zoomed-in node's CONTROLS (`crate::nodectl`); the state
    /// is `ctl_drag`.
    Ctl,
    WireFromOutput(u64, u32),
    WireFromInput(u64, usize),
    /// A Layer's input nodule picked up: slid along the bar it moves that
    /// layer (wire, keys and settings) to the gap it is dropped in; pulled
    /// off the bar it turns into the ordinary unplug.
    LayerReorder { node: u64, from: usize, press: Pos2 },
    /// ctrl-drag noodle slicer: cuts every wire the trail crosses (in link
    /// mode, every expression CABLE it crosses).
    Slice { last: Pos2, path: Vec<Pos2> },
    /// Link mode: dragging a cable out of a channel's output nodule.
    CableFrom(PSrc),
    /// Link mode: dragging a cable out of a channel's input nodule.
    CableTo(PDst),
    /// Marquee selection from a background drag (screen coords).
    BoxSelect { start: Pos2 },
    /// Houdini-style dolly: alt + right-drag zooms about the press point.
    Zoom { anchor: Pos2 },
    /// Hold `S` + drag (Blender's scale): the selection's SPACING scales
    /// about its centre by the pointer's distance from it over the press's.
    /// Positions only — nodes keep their size. Graph-space, applied fresh
    /// from the press-time snapshot each frame.
    SelScale {
        center: Pos2,
        start_dist: f32,
        orig: Vec<(u64, Pos2)>,
        /// Breakouts in the selection, with their press-time string pitch:
        /// scaling a harp scales its DISTANCES (string pitch here, bar gap
        /// through the positions), never the bars' own size.
        spacing: Vec<(u64, f32)>,
    },
    /// Hold `R` + drag (Blender's rotate): node POSITIONS orbit the
    /// selection's centre; the nodes themselves stay upright and their
    /// sockets stay top-down (flow direction is sacred).
    SelRotate {
        center: Pos2,
        start_angle: f32,
        orig: Vec<(u64, Pos2)>,
    },
    /// Houdini-style align swipe (`A` + drag): the stroke's direction picks
    /// the axis and the selection lands on the line it drew.
    AlignSwipe { start: Pos2, path: Vec<Pos2> },
    /// Dragging a backdrop's title bar. `members` is captured at press
    /// time — what was inside when you grabbed it is what comes along, so
    /// the group cannot shed nodes by outrunning them.
    GroupMove { members: Vec<u64> },
    /// Dragging a backdrop's grip: the members scale about the far corner
    /// of the blob, sizes included — resizing the group resizes what is in
    /// it, which is the point of grouping.
    GroupScale { anchor: Pos2, handle: Pos2, orig: Vec<(u64, Pos2, f32)> },
}

pub struct NodeEditor {
    /// alt+1 this frame: Some(target) locks viewer 1 there, Some(None)
    /// (nothing under the pointer) unlocks. The app drains it — the lock
    /// outlives this graph, it is held across dives.
    pub view_lock_req: Option<Option<(u64, u32)>>,
    /// Every noodle's first and last frame, keyed by the node the wire comes
    /// OUT of (`ranges.rs` — a Read's placed media, a generator's own range,
    /// anything else inherited). The app fills it each frame; a hovered
    /// wire shows it.
    pub wire_ranges: std::collections::BTreeMap<u64, (u64, u64)>,
    /// The wire range label drawn this frame, if any: (source node, range).
    /// Read by the suites — a label nobody can assert on is a label nobody
    /// notices breaking.
    pub shown_wire_range: Option<(u64, (u64, u64))>,
    /// Where each noodle's range came from (`ranges::Source`), keyed like
    /// `wire_ranges`: the hover says it, and the range view leans on it.
    pub wire_range_src: std::collections::BTreeMap<u64, crate::ranges::Source>,
    /// The script's own range, so a noodle can say how it sits against it.
    pub script_range: (u64, u64),
    /// The RANGE VIEW (shift+R, the bar's "ranges"): every noodle wears
    /// its first–last frame at its middle, brighter where the range is
    /// SET (a Read, an override, a retime, a merge that widens it) than
    /// where it is merely carried. The app keeps it in prefs.
    pub show_ranges: bool,
    /// The range tags drawn this frame: (source node, range). For suites.
    pub shown_range_tags: Vec<(u64, (u64, u64))>,
    /// Horseshoe-arc edits from this frame: (node, input, new value). The
    /// app drains them into the SAME keyed write-back the panel sliders
    /// use, so the arc and the properties row are one channel with two
    /// handles — keyframing, autokey and patch-driving included.
    pub gain_edits: Vec<(u64, usize, f32)>,
    /// In-graph shuffle edits from a harp bar's string dropdowns:
    /// (breakout, string index, new layer name). Applied by the app, which
    /// owns the panel mirror and the kid re-sync.
    pub shuffle_edits: Vec<(u64, usize, String)>,
    /// Per-string spacing drags from the bar grips: (breakout, string,
    /// delta in graph units; NaN = reset to the uniform pitch). Applied by
    /// the app to the bar AND its paired receiver.
    pub offset_edits: Vec<(u64, usize, f32)>,
    /// Layer lists for the harp dropdowns, keyed by BREAKOUT node id —
    /// (file spelling, display label). Filled by the app each frame from
    /// its loaded sequences; the editor cannot see those itself.
    pub harp_layers: std::collections::HashMap<u64, Vec<(String, String)>>,
    pub zoom: f32,
    pub pan: Vec2,
    /// Fullscreen-overlay mode: when Some, show() draws no background/grid,
    /// anchors the view to this remembered graph-panel rect (nodes hold
    /// their screen spots when the boundary goes away), and background
    /// drags pan instead of marquee-selecting.
    pub overlay_view_rect: Option<Rect>,
    /// Set when the view jumps programmatically (Q-frame): tells the
    /// FullNodes overlay host to skip mirroring that jump onto the image.
    pub suppress_nav_mirror: bool,
    /// Centre the view on this node next frame — how something OUTSIDE the
    /// graph (a ctrl-click on a viewer manipulator) says "show me where
    /// that lives". Keeps the current zoom: being shown a node should not
    /// also change how much of the graph you can see.
    pub focus: Option<u64>,
    /// A node asked to be opened in its external editor (a Blender node
    /// double-clicked). The app takes it: launching a process is not the
    /// graph's business, only noticing the gesture is.
    pub open_external: Option<u64>,
    /// Nodes were pasted this frame: the app adopts their footage (a pasted
    /// Read has a path and nothing loaded) the way it does after an undo.
    pub pasted: bool,
    /// Copies also go to the SYSTEM clipboard. Off under test, so a suite
    /// never clobbers what the user has on theirs.
    pub system_clipboard: bool,
    /// Input sockets to light, `(node, input)`: the app fills it each frame
    /// with the layer each Layer panel in the properties has selected, so
    /// the bar picked on the stack's timeline names its noodle in the graph.
    pub lit_inputs: Vec<(u64, usize)>,
    /// The last copy, for the ctrl+V paste: egui only hands clipboard TEXT
    /// over as the ⌘V event, so a chord it does not treat as paste reads
    /// what was copied here.
    clip: Option<String>,
    drag: Option<DragKind>,
    /// Spring-simulated stalk tips, graph space: (pos, velocity) per stalk.
    stalks: std::collections::HashMap<(u64, usize), (Pos2, Vec2)>,
    /// Stalks gliding to a new slot after a Layer reorder: where each set
    /// off from (graph space) and how far along it is, 0..1. Eased, so the
    /// switch reads as a move whatever the wobble setting.
    glides: std::collections::HashMap<(u64, usize), (Pos2, f32)>,
    /// A Layer nodule being slid: the slot it would land in now (last
    /// frame's answer), so the others open the gap for it while it moves.
    reorder_to: Option<usize>,
    /// Spring-simulated channel nodules in link mode, keyed
    /// (node, channel, is_output). Same underdamped spring as the stalks, so
    /// collapsing an rgb triple — which changes how many nodules a node has
    /// AND where every one of them sits — settles with a wobble instead of
    /// snapping. Nothing else needs to know a toggle happened.
    chan_springs: std::collections::HashMap<(u64, crate::anim::Channel, bool), (Pos2, Vec2)>,
    /// Wire a dragged node would splice into on drop: (dst, input, src).
    insert_target: Option<(u64, usize, u64)>,
    /// The lines a node drag snapped to this frame (graph space).
    snap_guides: Vec<(Pos2, Pos2)>,
    /// Primary press bookkeeping for click-selection: (origin, node hit,
    /// port hit). Selection applies on release if the pointer barely moved
    /// — independent of egui's click heuristics.
    press_state: Option<(Pos2, Option<u64>, Option<Port>)>,
    /// Right-click target while the context menu is open: Some(node) or
    /// None = background (with the graph-space position clicked).
    ctx_target: Option<(Option<u64>, Pos2)>,
    /// Drawn-position offsets from ⌘↑/⌘↓ swaps, decaying to zero (the swap
    /// animation). Node positions change instantly; the drawing eases.
    /// (drawn offset, start delay). The offset decays once the delay runs
    /// out; a delayed node holds its old drawn spot, then zips — which is
    /// what staggers the tidy cascade.
    anim: std::collections::HashMap<u64, (Vec2, f32)>,
    /// Canvas center last frame: panel show/hide shifts the layout; the pan
    /// compensates so graph content holds its screen position.
    last_center: Option<Pos2>,
    /// Tab node-search popup.
    search: Option<NodeSearch>,
    /// Each node name's (angle, angular velocity) on its way to the angle
    /// `place_label` picked: names swing and bounce into place rather than
    /// snapping.
    label_anim: std::collections::HashMap<u64, (f32, f32)>,
    /// Where each node's NAME was drawn last frame: (node, screen quad,
    /// the text's start point + angle). The rotated quad is the double-
    /// click target for renaming; same one-frame-stale scheme as the rest.
    label_hits: Vec<(u64, [Pos2; 4], Pos2, f32)>,
    /// A rename in progress: (node, buffer, just opened). Commits on Enter
    /// or on clicking away (Finder's rule); Esc cancels. An EMPTY name
    /// clears `ui_label`, so the node falls back to its kind's name.
    rename: Option<(u64, String, bool)>,
    /// The last single click on a node's NAME: (node, time, where). A
    /// second click on the same name inside the double-click window opens
    /// the rename — tracked by hand like every other click in this file,
    /// because egui's own click attribution is flaky under synthetic input.
    last_name_click: Option<(u64, f64, Pos2)>,
    /// Draw the expression patch's links over the graph, read-only. A view
    /// of the OTHER canvas: value cables are never edited from here.
    pub show_patch_links: bool,
    /// Mirrors the patch canvas's rgb toggle, so every expression view —
    /// the patch bay, this graph's link mode, and the read-only overlay —
    /// agrees on whether a triple is one port or three.
    pub expand_patch_rgb: bool,
    /// Expression-link mode: the image wiring steps aside and every node
    /// splays its animatable channels — inputs left, outputs right — so
    /// value cables can be patched in the graph itself, not just in the
    /// properties rack. Public so the app can put it in a menu.
    pub link_mode: bool,
    /// Cost + licence taint for the graph, computed once per frame by the
    /// app and pushed in. It lives there rather than here because it needs
    /// to know which Bakes actually have frames on disk — app state the
    /// graph cannot see — and because the properties panel and the Write
    /// warning read the same answer (manifesto rules 8 and 10).
    pub cost_flow: crate::costviz::Flow,
    /// The node a viewer key (1-4) just took, while the pointer is still on
    /// it: not a target again until the pointer leaves it or clicks.
    view_spent: Option<u64>,
    /// Channel nodule under the pointer last frame, for the label readout.
    /// A tool body is open, so an undriven channel input can become a KNOB
    /// on that tool. The editor does not know what a tool is; it only needs
    /// to know whether to offer the gesture.
    /// A tool body is open. The editor does not know what a tool is; it only
    /// needs to know whether to offer the CONTROL node types in its menus,
    /// which mean nothing in a comp's own graph.
    pub in_tool: bool,
    /// A noise field is being edited: the add menus offer field nodes and
    /// nothing else, since neither kind means anything in the other's graph.
    pub in_noise: bool,
    /// An SDF graph is being edited: SDF nodes only.
    pub in_sdf: bool,
    /// Editing a Particles node's graph.
    pub in_particles: bool,
    /// Editing a GenAI Pipeline's steps.
    pub in_genai: bool,
    /// Inside a Roto's dive: its shapes, combines and axes.
    pub in_roto: bool,
    /// A 2dini geometry graph is being edited: geometry nodes only.
    pub in_geo: bool,
    /// A 2dini preset network picked from the add menu: (preset index,
    /// graph position). The app inserts it — the editor does not know how.
    pub geo_preset_req: Option<(usize, Pos2)>,
    /// A noodle dropped on a big dot that fits more than one input: the
    /// short list asking which, until a pick or a click elsewhere.
    pub big_dot_menu: Option<BigDotMenu>,
    /// Last frame's big dots (screen) and the short list's rows, for the
    /// suites to aim at.
    big_dot_spots: Vec<(u64, Pos2)>,
    big_dot_rows: Vec<(usize, Rect)>,
    /// The channel nodule under the pointer in link mode, as
    /// (node, channel, is_output). Public because promoting a channel to a
    /// tool knob is the app's business, not the editor's — it only has to
    /// say which one the pointer is on.
    pub hover_chan: Option<(u64, crate::anim::Channel, bool)>,
    /// Set by the app when Tab was pressed this frame (the raw event is
    /// STOLEN before egui sees it — egui's focus traversal would otherwise
    /// grab Tab, focus a random button, and permanently kill the hotkeys).
    pub tab_open_request: bool,
    /// An S press over the graph that has not been spent on a scale drag:
    /// its RELEASE is the tap (swap inputs).
    s_armed: bool,
    /// How close two sockets must come before they court, in graph units at
    /// node size 1. The app sets it from prefs — one hand wants the wire to
    /// jump across the canvas, another wants to park two nodes side by side
    /// without them grabbing each other.
    pub magnet_reach: f32,
    /// Show the hotkey strip along the graph's top edge — the same
    /// instrument (and the same pref) as the viewer HUDs' `hudkeys` strip.
    pub show_keys: bool,
    /// Show the noodle-colour dots along the top-right edge. The graph's
    /// own top bar owns both toggles (not the viewer's "hud keys").
    pub show_legend: bool,
    /// Draw (and hit) every wire as straight runs with rounded corners —
    /// never a diagonal — instead of curves. The graph bar's "right angles".
    pub ortho_wires: bool,
    /// How much the graph's springs wobble (the graph bar's "wobble"):
    /// 1 as designed, 0 rigid, up to `WOBBLE_MAX` jelly. See `wobble_spring`.
    pub wobble: f32,
    /// How far the key strip's words are out, 0..1 — eased toward 1 while
    /// the hand hovers the caps.
    keys_reveal: f32,
    /// Last frame's caps row (screen), for the suite.
    keys_row: Option<Rect>,
    /// Hold the words out regardless of the pointer (headless renders).
    keys_hold: bool,
    /// How far the noodle legend's words are out, 0..1 — eased toward 1
    /// while the hand hovers its dots.
    legend_reveal: f32,
    /// Last frame's dot row (screen), for the suite.
    legend_row: Option<Rect>,
    /// Hold the legend's words out regardless of the pointer (renders).
    legend_hold: bool,
    /// Per TV-mode Read: its sequence and playhead frame. The app fills it
    /// each frame (Arc clones), because the loaded media and the player
    /// live there. See `crate::tv`.
    pub tv_feeds: std::collections::HashMap<u64, crate::tv::TvFeed>,
    /// The decoder the TVs' thumbnails go through (the app's own).
    pub tv_decoder: Option<std::sync::Arc<dyn engine::decode::ExrDecoder>>,
    tv: crate::tv::TvCache,
    /// Last frame's (frame index, big tier, exact picture, screen rect) per
    /// TV, for the suite.
    tv_state: std::collections::HashMap<u64, (usize, bool, bool, Rect)>,
    /// The socket the last click landed on. A selected port is what `G`
    /// grabs a noodle from.
    sel_port: Option<Port>,
    /// A noodle picked up with `G`: it trails the cursor ACROSS FRAMES, so
    /// you can pan and zoom to a node that was never on screen, and lands
    /// when you click a port (or is dropped by `G`/Esc).
    grab: Option<Port>,
    /// Hold `.` and press a wire or a dot: the pull setting its tension.
    dot_place: Option<DotPlace>,
    /// Selected backdrops. Kept apart from the node selection: Delete on a
    /// backdrop must remove the backdrop, never the nodes it sits behind.
    /// Public so the properties rack can lead with a panel for them.
    pub sel_groups: Vec<u64>,
    /// Group handles from LAST frame: (id, title rect, grip rect) in screen
    /// space — the same one-frame-stale scheme as `node_rects`.
    group_hits: Vec<(u64, Rect, Rect)>,
    /// Each backdrop's marching-squares shape, cached on `nodeux::blob_key`
    /// — rebuilt only when a member moves, resizes or rewires (or the zoom
    /// asks for a finer contour).
    blobs: std::collections::HashMap<u64, (u64, crate::nodeux::Blob)>,
    /// Right-click target while a backdrop's menu is open.
    ctx_group: Option<u64>,
    /// The wire the context menu was opened over, if any — what "cut a dot
    /// in here" acts on.
    ctx_wire: Option<((u64, usize, u64), Pos2)>,
    /// True once the current Z / X hold has been SPENT on a click, so
    /// releasing the key is not also a tap. Same discrimination space
    /// used to make between tap-to-play and hold-to-scrub.
    z_used: bool,
    /// The current M hold made a Blend (swipe), so its release is not a tap
    /// and must not open the region popup.
    m_used: bool,
    x_used: bool,
    /// Node-patch mode (shift+`): the ordinary noodles lock and grey out,
    /// the hidden layer draws in amber, and wires dragged here are created
    /// DRAWN HIDDEN — patching behind the PCB. Public for the app's menus.
    pub patch_mode: bool,
    /// A backtick pie, while its key is held: where it opened, WHOSE
    /// sources it offers, and — when exactly one node was selected at
    /// press — that node, which flips it into ADD mode (up registers it
    /// under its own name, down names it first).
    pie: Option<Pie>,
    /// The add-pie's naming popup: (node, the roster it joins, screen pos,
    /// buffer). Commits on Enter: the name lands on the node (`ui_label`)
    /// and the node lands in that roster.
    name_popup: Option<(u64, PieScope, Pos2, String)>,
    /// The orbit's landing spring, (value, velocity) toward 1 — the same
    /// underdamped instrument as the stalks and channel nodules.
    pie_anim: (f32, f32),
    /// The held node-ops pie (plain `): its centre, and its landing spring.
    ops_pie: Option<Pos2>,
    /// F1 held: the most-used-nodes pie (centre, spring). Its roster is
    /// `use_counts` ranked over the catalog.
    node_pie: Option<Pos2>,
    node_pie_anim: (f32, f32),
    /// How often each catalog label has been ADDED by hand — tab search,
    /// the menus, the pie itself. The app persists it in prefs; the F1 pie
    /// is its top eight.
    pub use_counts: std::collections::BTreeMap<String, u32>,
    /// Set when `use_counts` changed; the app saves prefs and clears it.
    pub use_counts_dirty: bool,
    ops_anim: (f32, f32),
    /// The op the ops pie picked last frame; its key site fires it once.
    ops_pick: Option<crate::keymap::Act>,
    /// The subwob op fired (alt+G or the ` pie): the app collapses the
    /// selection — it owns the registry, the error line and the undo record.
    pub collapse_req: bool,
    /// The selection's bakes queued / its Loads cached (alt+B / alt+C, or
    /// the ` pie): taken up by the app, which owns bakes and the cache.
    pub bake_req: bool,
    pub cache_req: bool,
    /// Reads the context menu asked to precache; the app takes it.
    pub precache_req: Option<Vec<u64>>,
    /// Each breakout bar's + button from LAST frame: (id, screen rect) —
    /// the usual one-frame-stale hit scheme.
    plus_hits: Vec<(u64, Rect)>,
    /// Inner graphs drawn this frame (`crate::innards`): per host node, the
    /// square it was drawn in, every port's, node's and `+`'s hit rect, and
    /// whether it is in far enough to edit. A press there is theirs.
    pub innard_hits: Vec<(u64, Rect, Vec<(crate::innards::Hit, Rect)>, bool)>,
    /// The inner-graph gesture in progress (`DragKind::Innard`).
    innard_drag: Option<crate::innards::Drag>,
    /// Controls drawn this frame (`crate::nodectl`): per host, where they
    /// sit, their row rects, and whether they are in far enough to edit.
    pub ctl_hits: Vec<(u64, crate::nodectl::Place, Vec<(crate::nodectl::Hit, Rect)>, bool)>,
    ctl_drag: Option<crate::nodectl::Drag>,
    /// In-node control edits from this frame: (node, channel, value). The
    /// app drains them into the keyed write-back the panel rows use.
    pub ctl_edits: Vec<(u64, crate::anim::Channel, f32)>,
    /// The frame animation evaluates at, set by the app each frame: the
    /// controls SHOW values there, and a drag starts from them.
    pub eval_tick: u64,
    /// A + was clicked: (breakout, screen pos). The APP consumes this and
    /// opens the upstream layer list there — the editor cannot know what
    /// is loaded, only that the gesture happened.
    pub plus_request: Option<(u64, Pos2)>,
    /// Nodes an add menu, the Tab search or the add pie made this frame.
    /// The app drains it (a Transform under a soft crop takes the crop's
    /// centre as its pivot).
    pub added: Vec<u64>,
    /// The M popup, while open: (screen pos, graph pos it was opened at).
    region_menu: Option<(Pos2, Pos2)>,
    /// The region whose wash holds the view centre — whose sources the pie
    /// offers and whose name the corner HUD shows. Recomputed each frame.
    pub active_region: Option<u64>,
    /// Region name chips from LAST frame: (id, screen rect) — the same
    /// one-frame-stale scheme as `group_hits`.
    region_hits: Vec<(u64, Rect)>,
    /// Right-click target while a region's menu is open.
    ctx_region: Option<u64>,
}

/// Default shift-to-join reach, in graph units at size 1. Deliberately
/// shorter than a node is wide: at the old 90 two nodes could not be parked
/// beside each other without lunging at one another.
pub const MAGNET_REACH: f32 = 34.0;

impl NodeEditor {
    /// Where a backdrop's scale grip was drawn last frame (screen coords);
    /// --test-nodeux aims its synthetic press here.
    pub fn test_group_grip(&self, id: u64) -> Option<Pos2> {
        self.group_hits.iter().find(|(g, ..)| *g == id).map(|(_, _, gr)| gr.center())
    }

    /// Whether the backtick pie is open, and where slice `i` of `n` sits
    /// (screen coords) — --test-region aims its synthetic click there.

    /// Whose sources the open pie is offering — what `--test-region` checks
    /// before it aims at a disc.
    pub fn test_pie_scope(&self) -> Option<PieScope> {
        self.pie.map(|p| p.scope)
    }

    pub fn test_ops_pie_open(&self) -> bool {
        self.ops_pie.is_some()
    }

    /// Last frame's (frame index, big tier, exact picture, screen rect)
    /// for TV `id`, if it was on screen.
    pub fn test_tv(&self, id: u64) -> Option<(usize, bool, bool, Rect)> {
        self.tv_state.get(&id).copied()
    }

    /// Land finished TV decodes now (a headless render waits on them).
    #[cfg(test)]
    pub fn test_tv_pump(&mut self, ctx: &egui::Context) -> bool {
        match self.tv_decoder.clone() {
            Some(d) => self.tv.pump(ctx, &d),
            None => false,
        }
    }

    /// How far the key strip's words are out (0 hidden .. 1 fully out).
    pub fn test_keys_reveal(&self) -> f32 {
        self.keys_reveal
    }
    pub fn test_keys_row(&self) -> Option<Rect> {
        self.keys_row
    }
    /// Keep the key strip's words out (a headless render has no hand).
    #[cfg(test)]
    pub fn test_hold_keys_open(&mut self) {
        self.keys_hold = true;
    }

    pub fn test_region_menu_open(&self) -> bool {
        self.region_menu.is_some()
    }

    /// Last frame's noodle-legend dot row (screen), if drawn.
    pub fn test_legend_row(&self) -> Option<Rect> {
        self.legend_row
    }
    pub fn test_legend_reveal(&self) -> f32 {
        self.legend_reveal
    }
    /// Keep the legend's words out (a headless render has no hand).
    #[cfg(test)]
    pub fn test_hold_legend_open(&mut self) {
        self.legend_hold = true;
    }

    pub fn test_renaming(&self) -> Option<u64> {
        self.rename.as_ref().map(|(id, ..)| *id)
    }
    /// Centre of node `id`'s drawn name last frame — the double-click
    /// target for renaming.
    pub fn test_label_centre(&self, id: u64) -> Option<Pos2> {
        self.label_hits.iter().find(|(l, ..)| *l == id).map(|(_, q, ..)| {
            Pos2::new(
                (q[0].x + q[1].x + q[2].x + q[3].x) * 0.25,
                (q[0].y + q[1].y + q[2].y + q[3].y) * 0.25,
            )
        })
    }

    /// A gesture is in flight in the graph (a wire, a node, a box…).
    pub fn is_dragging(&self) -> bool {
        self.drag.is_some()
    }

    /// Node `id`'s big dot, screen, last frame.
    pub fn test_big_dot(&self, id: u64) -> Option<Pos2> {
        self.big_dot_spots.iter().find(|(n, _)| *n == id).map(|(_, p)| *p)
    }
    /// The big dot's short list row for `input`, if the list is up.
    pub fn test_big_dot_row(&self, input: usize) -> Option<Pos2> {
        self.big_dot_rows.iter().find(|(i, _)| *i == input).map(|(_, r)| r.center())
    }
    pub fn test_big_dot_menu(&self) -> Option<&BigDotMenu> {
        self.big_dot_menu.as_ref()
    }

    pub fn test_node_pie_open(&self) -> bool {
        self.node_pie.is_some()
    }
    /// (slice centre, label) of the F1 pie's slice `i`, if open.
    pub fn test_node_slice(&self, i: usize) -> Option<(Pos2, String)> {
        let c = self.node_pie?;
        let top = top_used(&self.use_counts);
        let e = top.get(i)?;
        Some((
            crate::regions::pie_slice_pos(c, i, top.len(), OPS_PIE_R),
            e.label.to_string(),
        ))
    }
    /// Screen centre of the ops pie's slice for `act`, if open.
    pub fn test_ops_slice(&self, act: crate::keymap::Act) -> Option<Pos2> {
        let i = OPS_PIE.iter().position(|(a, _)| *a == act)?;
        self.ops_pie.map(|c| crate::regions::pie_slice_pos(c, i, OPS_PIE.len(), OPS_PIE_R))
    }
    pub fn test_pie_slice(&self, i: usize, n: usize) -> Option<Pos2> {
        self.pie.map(|p| {
            crate::regions::pie_slice_pos(p.at, i, n, crate::regions::PIE_R)
        })
    }

    /// Tidy with the CASCADE: every node that moved keeps its old drawn
    /// spot for a few staggered milliseconds (top rows first), then zips to
    /// its place — the whole shuffle lands in well under a second, but the
    /// eye gets to follow it.
    /// Pick a graph op as the ` pie does: it fires through the op's own key
    /// site on the next show, on the selection (or the whole comp when
    /// nothing is selected). The scripting API's `graph_op` lands here.
    pub(crate) fn pick_op(&mut self, act: crate::keymap::Act) {
        self.ops_pick = Some(act);
    }

    pub fn tidy_cascade(&mut self, graph: &mut EditorGraph, ids: &[u64]) {
        let before: std::collections::HashMap<u64, Pos2> =
            graph.nodes.iter().map(|(id, n)| (*id, n.pos)).collect();
        if !crate::nodeux::auto_layout_for(graph, ids, self.ortho_wires) {
            return;
        }
        let mut moved: Vec<(u64, Vec2, f32)> = graph
            .nodes
            .iter()
            .filter_map(|(id, n)| {
                let d = *before.get(id)? - n.pos;
                (d.length() > 0.5).then_some((*id, d, n.pos.y))
            })
            .collect();
        moved.sort_by(|a, b| a.2.total_cmp(&b.2).then(a.0.cmp(&b.0)));
        for (k, (id, d, _)) in moved.iter().enumerate() {
            // Existing motion (a mid-flight swap) is replaced, not stacked:
            // the cascade owns the picture for its fraction of a second.
            self.anim.insert(*id, (*d, (k as f32 * 0.012).min(0.25)));
        }
    }
}

impl Default for NodeEditor {
    fn default() -> Self {
        Self {
            wire_ranges: Default::default(),
            shown_wire_range: None,
            wire_range_src: Default::default(),
            script_range: (0, 99),
            show_ranges: false,
            shown_range_tags: Vec::new(),
            gain_edits: Vec::new(),
            shuffle_edits: Vec::new(),
            offset_edits: Vec::new(),
            harp_layers: Default::default(),
            zoom: 1.0,
            pan: Vec2::ZERO,
            overlay_view_rect: None,
            suppress_nav_mirror: false,
            focus: None,
            open_external: None,
            pasted: false,
            system_clipboard: true,
            lit_inputs: Vec::new(),
            clip: None,
            drag: None,
            stalks: std::collections::HashMap::new(),
            glides: std::collections::HashMap::new(),
            reorder_to: None,
            chan_springs: std::collections::HashMap::new(),
            insert_target: None,
            snap_guides: Vec::new(),
            press_state: None,
            ctx_target: None,
            anim: std::collections::HashMap::new(),
            last_center: None,
            search: None,
            label_anim: Default::default(),
            label_hits: Vec::new(),
            rename: None,
            last_name_click: None,
            show_patch_links: false,
            expand_patch_rgb: false,
            tab_open_request: false,
            link_mode: false,
            cost_flow: Default::default(),
            view_spent: None,
            in_tool: false,
            in_noise: false,
            in_geo: false,
            geo_preset_req: None,
            big_dot_menu: None,
            big_dot_spots: Vec::new(),
            big_dot_rows: Vec::new(),
            in_sdf: false,
            in_particles: false,
            in_genai: false,
            in_roto: false,
            hover_chan: None,
            s_armed: false,
            magnet_reach: MAGNET_REACH,
            show_keys: true,
            show_legend: true,
            ortho_wires: false,
            wobble: 1.0,
            keys_reveal: 0.0,
            keys_row: None,
            keys_hold: false,
            view_lock_req: None,
            legend_reveal: 0.0,
            legend_row: None,
            legend_hold: false,
            tv_feeds: std::collections::HashMap::new(),
            tv_decoder: None,
            tv: Default::default(),
            tv_state: std::collections::HashMap::new(),
            sel_port: None,
            grab: None,
            dot_place: None,
            sel_groups: Vec::new(),
            group_hits: Vec::new(),
            blobs: std::collections::HashMap::new(),
            ctx_group: None,
            ctx_wire: None,
            z_used: false,
            m_used: false,
            x_used: false,
            patch_mode: false,
            pie: None,
            name_popup: None,
            pie_anim: (0.0, 0.0),
            ops_pie: None,
            node_pie: None,
            node_pie_anim: (0.0, 0.0),
            use_counts: std::collections::BTreeMap::new(),
            use_counts_dirty: false,
            ops_anim: (0.0, 0.0),
            ops_pick: None,
            collapse_req: false,
            bake_req: false,
            cache_req: false,
            precache_req: None,
            plus_hits: Vec::new(),
            innard_hits: Vec::new(),
            innard_drag: None,
            ctl_hits: Vec::new(),
            ctl_drag: None,
            ctl_edits: Vec::new(),
            eval_tick: 0,
            plus_request: None,
            added: Vec::new(),
            region_menu: None,
            active_region: None,
            region_hits: Vec::new(),
            ctx_region: None,
        }
    }
}

struct View {
    origin: Pos2,
    zoom: f32,
}

impl View {
    fn to_screen(&self, p: Pos2) -> Pos2 {
        self.origin + p.to_vec2() * self.zoom
    }
    fn to_graph(&self, p: Pos2) -> Pos2 {
        ((p - self.origin) / self.zoom).to_pos2()
    }
}

/// Snap distance for a dragged node, in screen points.
const DRAG_SNAP_PX: f32 = 8.0;

/// Where a dragged set should snap: the offset to add to the nodes in
/// `moved` (already at their unsnapped positions) and the guides to draw
/// for it (graph space). Each axis snaps on its own, to the nearest of a
/// centre lined up with another node inside `area`, or a wire between the
/// set and the rest coming out exactly vertical or horizontal (socket to
/// socket, so it holds in boring mode and for side sockets too). A wire
/// wins a tie: it is the line the eye follows.
fn drag_snap(
    graph: &EditorGraph,
    mask_dirs: &std::collections::HashMap<u64, Vec2>,
    moved: &[u64],
    area: Rect,
    tol: f32,
) -> (Vec2, Vec<(Pos2, Pos2)>) {
    // Per axis: (|delta| with the wire bias, delta, from, to).
    let mut best: [Option<(f32, f32, Pos2, Pos2)>; 2] = [None, None];
    let mut offer = |axis: usize, from: Pos2, to: Pos2, wire: bool| {
        let d = if axis == 0 { to.x - from.x } else { to.y - from.y };
        if d.abs() > tol {
            return;
        }
        let score = d.abs() - if wire { tol * 0.25 } else { 0.0 };
        if best[axis].is_none_or(|(b, ..)| score < b) {
            best[axis] = Some((score, d, from, to));
        }
    };
    // A harp's hidden kids are machinery, never something to line up with.
    let kids = crate::breakout::kid_set(graph);
    for m in moved {
        let Some(mn) = graph.node(*m) else { continue };
        for (oid, on) in &graph.nodes {
            if moved.contains(oid) || kids.contains(oid) || !area.contains(on.pos) {
                continue;
            }
            offer(0, mn.pos, on.pos, false);
            offer(1, mn.pos, on.pos, false);
        }
    }
    for (dst, input, src) in &graph.edges {
        let (dm, sm) = (moved.contains(dst), moved.contains(src));
        if dm == sm || kids.contains(dst) || kids.contains(src) || graph.is_hidden(*dst, *input) {
            continue;
        }
        let (Some(dn), Some(sn)) = (graph.node(*dst), graph.node(*src)) else { continue };
        let inp = rest_of(graph, mask_dirs, *dst, *input, dn.pos, dn.size);
        let out = output_rest_of(&graph.nodes, *src, graph.out_of(*dst, *input) as usize, sn.size);
        let (from, to) = if dm { (inp, out) } else { (out, inp) };
        offer(0, from, to, true);
        offer(1, from, to, true);
    }
    let mut off = Vec2::ZERO;
    let mut guides = Vec::new();
    for (axis, b) in best.iter().enumerate() {
        if let Some((_, d, from, to)) = b {
            if axis == 0 {
                off.x = *d;
            } else {
                off.y = *d;
            }
            guides.push((*from, *to));
        }
    }
    // The guides end where the snapped set now is.
    for g in &mut guides {
        g.0 += off;
    }
    (off, guides)
}

/// Rest position a stalk tip springs toward (inputs splay across the top).
/// Everything scales with the node's own size.
/// Rest point for an input stalk. Mask inputs rest horizontally, leaning
/// toward the nearest mask output (see `mask_dirs` in show()).
fn rest_of(
    graph: &EditorGraph,
    mask_dirs: &std::collections::HashMap<u64, Vec2>,
    id: u64,
    input: usize,
    pos: Pos2,
    size: f32,
) -> Pos2 {
    socket_of(graph, mask_dirs, id, input, pos, size).0
}

/// `rest_of`, plus the way the socket faces when it rests on the node's
/// SIDE (`side_dir`) — `None` for one on top. Right-angle wires enter a
/// side socket sideways.
fn socket_of(
    graph: &EditorGraph,
    mask_dirs: &std::collections::HashMap<u64, Vec2>,
    id: u64,
    input: usize,
    pos: Pos2,
    size: f32,
) -> (Pos2, Option<Vec2>) {
    let top = |p: Pos2| (p, None);
    let Some(n) = graph.nodes.get(&id) else { return top(input_rest(pos, 0, 1, size)) };
    if multi_input(&n.kind) {
        return top(multi_rest(pos, input, n.kind.input_count(), size));
    }
    // A big-dot node: the wired inputs in a row, then the dot, which is
    // where every hidden one rests.
    if big_dot(&n.kind) {
        let shown = shown_inputs(graph, id);
        return match shown.iter().position(|i| *i == input) {
            Some(slot) => top(input_rest(pos, slot, shown.len() + 1, size)),
            None => top(big_dot_rest(graph, id, pos, size)),
        };
    }
    // A merge bar (Blend, Extract, Layer): strings across the top, one
    // bookend on the left porch — the harp's top edge, below.
    if let Some(x) = crate::breakout::bar_in_x(&n.kind, pos, input, size) {
        return top(Pos2::new(x, pos.y - (R * 0.55 + stalk()) * size));
    }
    // The harp's top edge: a Recombine's inputs sit at the string pitch,
    // and a Breakout's ONE input (the fat cable) lands on its left porch —
    // a bookend, not a string.
    match &n.kind {
        NodeKind::Breakout(_) => {
            if let Some(x) = crate::breakout::end_x(&n.kind, pos, size) {
                return top(Pos2::new(x, pos.y - (R * 0.55 + stalk()) * size));
            }
        }
        NodeKind::Recombine(p) => {
            // The base rgba lands on the left bookend (its alpha is what
            // the rebuild wears); every other input is a string.
            let x = if input == p.base_input() {
                crate::breakout::base_x(&n.kind, pos, size)
            } else {
                crate::breakout::string_x(&n.kind, pos, input, size)
            };
            if let Some(x) = x {
                return top(Pos2::new(x, pos.y - (R * 0.55 + stalk()) * size));
            }
        }
        _ => {}
    }
    if matches!(n.kind, NodeKind::Dot { .. }) {
        return top(pos);
    }
    let lean = mask_dirs.get(&id).copied().unwrap_or(Vec2::new(-1.0, 0.0));
    if let Some(d) = side_dir(&n.kind, input, lean) {
        (pos + if boring() { boring_edge(d) } else { d * socket_reach() } * size, Some(d))
    } else {
        top(input_rest(pos, n.kind.image_slot(input), n.kind.image_input_count(), size))
    }
}

/// How a right-angle wire travels into a side socket facing `d`: toward its
/// node, along x. `None` (enter from above) when it faces mostly up.
fn side_entry(d: Vec2) -> Option<f32> {
    (d.x.abs() > 0.2).then(|| -d.x.signum())
}

/// Is input `i` the node's CHAIN input — the picture it works on, which
/// chain gestures (delete-reconnect, swaps) follow? An image port, or a
/// pipe-follower's port whichever pipe it is in (an Invert of a matte).
pub(crate) fn is_chain_input(kind: &NodeKind, i: usize) -> bool {
    kind.input_kind(i) == PortKind::Image || kind.follows_pipe(i)
}

/// Does this input rest on the node's SIDE, leaning toward a matte? A mask
/// that LIMITS an effect does (a Grade's matte), which is what tells it
/// apart from the picture above. A roto shape's one input is the matte it
/// combines with — its subject, not a limit — so it stands straight up like
/// any chain input: sideways it read as a Grade's matte socket whenever
/// the two sat side by side.
fn side_mask(kind: &NodeKind, input: usize) -> bool {
    kind.is_side_input(input) && !matches!(kind, NodeKind::Spline(_))
}

/// Which way a side socket points, given the node's `lean` toward the
/// nearest matte: the MASK takes the lean, and a MODULATE socket takes the
/// mirror side when there is a mask to share with — both carry a matte and
/// wear its colour, so the side (and the `~` glyph) is what tells a scaled
/// effect from a mixed-back one at a glance. `None` for a socket on top.
fn side_dir(kind: &NodeKind, input: usize, lean: Vec2) -> Option<Vec2> {
    if !side_mask(kind, input) {
        return None;
    }
    if kind.mod_input() == Some(input) && kind.mask_input().is_some() {
        return Some(Vec2::new(-lean.x, lean.y));
    }
    Some(lean)
}

#[cfg(test)]
mod side_mask_tests {
    use super::*;

    /// A roto shape's input stands on TOP, like any chain input; a
    /// Grade's matte still rests on the side, so the two never look alike.
    #[test]
    fn a_roto_input_stands_up_and_a_matte_socket_does_not() {
        let mut g = EditorGraph::default();
        let sp = g.add(NodeKind::Spline(Default::default()), Pos2::ZERO);
        let gr = g.add(NodeKind::Grade(Default::default()), Pos2::new(300.0, 0.0));
        let dirs = std::collections::HashMap::new();
        let up = rest_of(&g, &dirs, sp, 0, Pos2::ZERO, 1.0);
        assert!(up.x.abs() < 1e-3 && up.y < 0.0, "roto input rests at {up:?}");
        let m = g.node(gr).and_then(|n| n.kind.mask_input()).unwrap();
        let side = rest_of(&g, &dirs, gr, m, Pos2::new(300.0, 0.0), 1.0);
        assert!((side.y - 0.0).abs() < 1e-3, "a Grade's matte rests level: {side:?}");
    }

    /// A modulate socket rests where a mask would — the side, leaning at the
    /// matte — and wears `~`, which a mask never does; a node with BOTH
    /// would put the modulate on the mirror side (`side_dir`).
    #[test]
    fn a_modulate_socket_is_a_side_socket_with_a_glyph() {
        let mut g = EditorGraph::default();
        let dirs = std::collections::HashMap::new();
        for kind in [
            NodeKind::Blur(Default::default()),
            NodeKind::Dilate(Default::default()),
            NodeKind::Transform(Default::default()),
            NodeKind::CornerPin(Default::default()),
        ] {
            let name = kind.name();
            let md = kind.mod_input().unwrap();
            assert_eq!(kind.input_kind(md), PortKind::Mask, "{name}: modulate takes a matte");
            assert_eq!(kind.input_tag(md), Some("~"), "{name}");
            assert_eq!(kind.image_input_count(), 1, "{name}");
            let id = g.add(kind, Pos2::ZERO);
            let p = rest_of(&g, &dirs, id, md, Pos2::ZERO, 1.0);
            assert!(p.x < -1.0 && p.y.abs() < 1e-3, "{name}: modulate rests at {p:?}");
        }
        let gr = NodeKind::Grade(Default::default());
        assert_eq!(gr.input_tag(gr.mask_input().unwrap()), None, "a mask wears no glyph");
        // The mirror rule, for a node that has both.
        let both = side_dir(&NodeKind::Blur(Default::default()), 1, Vec2::new(-1.0, 0.0));
        assert_eq!(both, Some(Vec2::new(-1.0, 0.0)), "a Blur has no mask to mirror");
    }

    /// A setup saved while a Blur had mask 1 + modulate 2 loads with the
    /// modulate matte on the one port, 1, and the old mask dropped.
    #[test]
    fn a_blur_modulate_on_the_old_port_moves_to_the_one_port() {
        let mut g = EditorGraph::default();
        let img = g.add(NodeKind::Grid(Default::default()), Pos2::ZERO);
        let old_mask = g.add(NodeKind::Spline(Default::default()), Pos2::ZERO);
        let modm = g.add(NodeKind::Spline(Default::default()), Pos2::ZERO);
        let b = g.add(NodeKind::Blur(Default::default()), Pos2::ZERO);
        g.edges.extend([(b, 0, img), (b, 1, old_mask), (b, 2, modm)]);
        g.bridge_mistyped();
        assert_eq!(g.input_of(b, 1), Some(modm));
        assert_eq!(g.input_of(b, 2), None);
        assert_eq!(g.input_of(b, 0), Some(img));
    }
}

/// Houdini's Merge: ONE wide socket every wire lands on, in input order,
/// rather than a nodule per input. The node's last input is always the
/// free one (the app grows and packs it), and it is where the whole bar
/// hits.
/// Does this node wear the BIG DOT: one nodule standing for every input
/// it has, with only the wired ones popped out beside it? A node with a
/// row of number sockets (a Transform has seven) is a comb of cups nobody
/// can aim at; with the dot, a noodle dropped on it goes where it can —
/// the one input of its kind, or a pick from a short list. A prototype in
/// the 2dini graph; the predicate is the only thing that knows that.
pub fn big_dot(kind: &NodeKind) -> bool {
    match kind {
        NodeKind::GeoNode(k) => !matches!(k, engine::geo::GeoNode::Merge { .. }) && k.inputs().len() >= 4,
        // The noise graph's many-socket nodes (an fBm has six).
        NodeKind::Field(k) => k.inputs().len() >= 4,
        _ => false,
    }
}

/// The inputs a node shows as nodules: all of them, or on a big-dot node
/// the wired ones in order (the dot stands for the rest).
pub fn shown_inputs(graph: &EditorGraph, id: u64) -> Vec<usize> {
    let Some(n) = graph.nodes.get(&id) else { return Vec::new() };
    let all = 0..n.kind.input_count();
    if big_dot(&n.kind) {
        all.filter(|i| graph.input_of(id, *i).is_some()).collect()
    } else {
        all.collect()
    }
}

/// Where a big-dot node's dot sits: the slot after its shown inputs.
fn big_dot_rest(graph: &EditorGraph, id: u64, pos: Pos2, size: f32) -> Pos2 {
    let shown = shown_inputs(graph, id).len();
    input_rest(pos, shown, shown + 1, size)
}

/// The inputs of `dst` a noodle off `src`'s output `out` could land in,
/// free ones first (only free ones, when there are any).
pub fn big_dot_options(graph: &EditorGraph, src: u64, out: u32, dst: u64) -> Vec<usize> {
    let (Some(sn), Some(dn)) = (graph.node(src), graph.node(dst)) else { return Vec::new() };
    let kind = sn.kind.output_kind_at(out as usize);
    let fits: Vec<usize> = (0..dn.kind.input_count()).filter(|i| dn.kind.accepts(*i, kind)).collect();
    let free: Vec<usize> = fits.iter().copied().filter(|i| graph.input_of(dst, *i).is_none()).collect();
    if free.is_empty() { fits } else { free }
}

/// One bar that takes any number of wires: 2dini's Merge, and a Blend 3D
/// (whose cards are depth-sorted, so their order means nothing).
pub fn multi_input(kind: &NodeKind) -> bool {
    matches!(
        kind,
        NodeKind::GeoNode(engine::geo::GeoNode::Merge { .. }) | NodeKind::Scene3D(_) | NodeKind::RotoNode(_)
    )
}

/// Pitch between wire ends on a multi-input bar, graph units at size 1:
/// tighter as wires pile on so the bar never outgrows its node much.
fn multi_pitch(n_inputs: usize) -> f32 {
    let wired = n_inputs.saturating_sub(1);
    if wired < 2 { 0.0 } else { 12.0f32.min(72.0 / (wired - 1) as f32) }
}

/// Half the bar's length, graph units at size 1.
fn multi_half(n_inputs: usize) -> f32 {
    let wired = n_inputs.saturating_sub(1).max(1);
    ((wired - 1) as f32 * multi_pitch(n_inputs) * 0.5 + 9.0).max(16.0)
}

/// Where input `input` of a multi-input node lands on its bar: wired ones
/// spread across it, the free one at its centre.
fn multi_rest(pos: Pos2, input: usize, n_inputs: usize, size: f32) -> Pos2 {
    let y = pos.y - socket_reach() * size;
    let wired = n_inputs.saturating_sub(1);
    if input >= wired {
        return Pos2::new(pos.x, y);
    }
    let k = input as f32 - (wired as f32 - 1.0) / 2.0;
    Pos2::new(pos.x + k * multi_pitch(n_inputs) * size, y)
}

/// `slot` is the port's ordinal among the node's IMAGE inputs, which is not
/// the raw input index when a mask sits in the middle of them.
fn input_rest(pos: Pos2, slot: usize, n_inputs: usize, size: f32) -> Pos2 {
    if boring() {
        return Pos2::new(boring_x(pos, slot, n_inputs, size), pos.y - BORING_HALF.y * size);
    }
    let spread = if n_inputs > 1 { 20.0 * size } else { 0.0 };
    let x = pos.x + (slot as f32 - (n_inputs as f32 - 1.0) / 2.0) * 2.0 * spread;
    Pos2::new(x, pos.y - (R + STALK) * size)
}

/// The channels a node exposes to expression links, in a stable order:
/// everything its kind actually has a value for.
pub fn node_channels(kind: &NodeKind) -> Vec<crate::anim::Channel> {
    crate::anim::channels_for(kind)
        .into_iter()
        .filter(|c| crate::anim::get(kind, *c).is_some())
        .collect()
}

/// Vertical spacing between channel nodules, in graph units. Wide enough
/// that neighbouring nodules never touch at any sane zoom.
const CHAN_PITCH: f32 = 15.0;

/// Where channel `i` of `n` sits in link mode: inputs to the LEFT, outputs
/// to the RIGHT. The tentacles get longer the more channels a node has, so a
/// Grade's twelve stay as readable as a Blur's one.
fn chan_rest(pos: Pos2, size: f32, i: usize, n: usize, out: bool) -> Pos2 {
    let pitch = CHAN_PITCH * size;
    let span = n.saturating_sub(1) as f32 * pitch;
    let reach = ((R + STALK) * 1.3 + n as f32 * 2.5) * size;
    Pos2::new(
        pos.x + if out { reach } else { -reach },
        pos.y - span * 0.5 + i as f32 * pitch,
    )
}

/// Where output `slot` of a node with `n_out` outputs rests. Splayed the
/// same way `input_rest` splays inputs, so a node with several mattes reads
/// as several mattes rather than one socket you have to pick a band from.
fn output_rest(pos: Pos2, slot: usize, n_out: usize, size: f32) -> Pos2 {
    if boring() {
        return Pos2::new(boring_x(pos, slot, n_out, size), pos.y + BORING_HALF.y * size);
    }
    let spread = if n_out > 1 { 20.0 * size } else { 0.0 };
    let x = pos.x + (slot as f32 - (n_out as f32 - 1.0) / 2.0) * 2.0 * spread;
    Pos2::new(x, pos.y + (R + STALK) * size)
}

/// `output_rest` with the KIND in hand: a Breakout's outputs sit at the
/// harp's pitch instead of the standard splay. Everything else falls
/// through unchanged.
fn output_rest_of(
    nodes: &std::collections::BTreeMap<u64, ENode>,
    id: u64,
    slot: usize,
    size: f32,
) -> Pos2 {
    let Some(n) = nodes.get(&id) else { return output_rest(Pos2::ZERO, slot, 1, size) };
    if matches!(n.kind, NodeKind::Recombine(_)) {
        // The receiver's output leaves from its right porch — the summed
        // beauty is a bookend, mirrored from the breakout's cable input.
        if let Some(x) = crate::breakout::end_x(&n.kind, n.pos, size) {
            return Pos2::new(x, n.pos.y + (R * 0.55 + stalk()) * size);
        }
    }
    if let Some(x) = crate::breakout::string_x(&n.kind, n.pos, slot, size)
        .or_else(|| crate::breakout::bar_out_x(&n.kind, n.pos, slot, size))
    {
        return Pos2::new(x, n.pos.y + (R * 0.55 + stalk()) * size);
    }
    let n_o = n.kind.output_count().clamp(1, MAX_OUTPUTS);
    if boring() && tv_on(n) {
        // Off the card's bottom edge.
        return Pos2::new(boring_x(n.pos, slot, n_o, size), n.pos.y + BORING_READ_HALF.y * size);
    }
    output_rest(n.pos, slot, n_o, size)
}

/// Stalk key for output `k`. `k == 0` is `OUT_STALK` itself, so every
/// single-output node keys exactly as it always did.
pub fn out_stalk(k: usize) -> usize {
    OUT_STALK - k
}

/// The output index a stalk key refers to, if it is an output at all.
pub fn stalk_out(k: usize) -> Option<usize> {
    (k >= OUT_STALK - MAX_OUTPUTS).then(|| OUT_STALK - k)
}

/// Most outputs a node may show. Bounds the key space that `stalk_out`
/// carves out of the top of `usize`.
pub const MAX_OUTPUTS: usize = 64;

/// Stalk key: (node, input index), with usize::MAX = the output stalk.
pub const OUT_STALK: usize = usize::MAX;

/// Distance from node center to a resting socket (test harness geometry).
pub const SOCKET_OFFSET: f32 = R + STALK;

/// Control points of a wire between two screen-space sockets.
fn wire_ctrl(from: Pos2, to: Pos2, zoom: f32) -> [Pos2; 4] {
    let d = ((to.y - from.y).abs() * 0.5).max(24.0) * zoom.min(1.5);
    [from, from + Vec2::new(0.0, d), to - Vec2::new(0.0, d), to]
}

/// `wire_ctrl` with an optional TANGENT per end. A node's end keeps the
/// top-to-down tangent; a DOT does not impose a direction — its tangent is
/// the wire's own travel through it, so a sideways route through dots
/// reads as one smooth line instead of a chain of vertical S-bends.
fn wire_ctrl_flow(
    from: Pos2,
    to: Pos2,
    zoom: f32,
    ts: Option<Vec2>,
    td: Option<Vec2>,
) -> [Pos2; 4] {
    let dn = ((to.y - from.y).abs() * 0.5).max(24.0) * zoom.min(1.5);
    let dd = dot_handle_len(from, to, zoom);
    let p1 = from + ts.map(|t| t * dd).unwrap_or(Vec2::new(0.0, dn));
    let p2 = to - td.map(|t| t * dd).unwrap_or(Vec2::new(0.0, dn));
    [from, p1, p2, to]
}

/// A wire's drawn path: the classic curve, or (the graph's "right angles"
/// mode) straight runs with rounded corners. Every draw and every hit test
/// on a wire goes through one of these, so what you click is what you see.
#[derive(Clone)]
pub(crate) enum WirePath {
    Curve([Pos2; 4]),
    Ortho(Vec<Pos2>),
}

impl WirePath {
    /// The path as points: the curve sampled `n` times, the right-angle
    /// route as it is (its corners are already arcs).
    pub(crate) fn points(&self, n: usize) -> Vec<Pos2> {
        match self {
            WirePath::Curve(c) => (0..=n).map(|k| cubic_point(c, k as f32 / n as f32)).collect(),
            WirePath::Ortho(p) => p.clone(),
        }
    }

    /// `n` consecutive pieces of the path, for dashes and colour ramps: the
    /// curve's sample-to-sample steps, or the right-angle route cut into
    /// equal lengths that keep their corners.
    pub(crate) fn pieces(&self, n: usize) -> Vec<Vec<Pos2>> {
        match self {
            WirePath::Curve(_) => self.points(n).windows(2).map(|w| w.to_vec()).collect(),
            WirePath::Ortho(p) => crate::ortho::split(p, n),
        }
    }

    pub(crate) fn min_dist(&self, q: Pos2) -> f32 {
        match self {
            WirePath::Curve(c) => bezier_min_dist(c, q),
            WirePath::Ortho(p) => crate::ortho::min_dist(p, q),
        }
    }

    /// Does the segment a–b cross the wire? (The slicer's question.)
    pub(crate) fn hits_segment(&self, a: Pos2, b: Pos2) -> bool {
        match self {
            WirePath::Curve(c) => bezier_hits_segment(c, a, b),
            WirePath::Ortho(p) => crate::ortho::hits_segment(p, a, b),
        }
    }

    /// The point on the wire nearest `q`, so a dot lands ON it.
    pub(crate) fn nearest(&self, q: Pos2) -> Pos2 {
        match self {
            WirePath::Curve(c) => {
                let (mut bt, mut bd) = (0.0f32, f32::MAX);
                for k in 0..=32 {
                    let t = k as f32 / 32.0;
                    let d = cubic_point(c, t).distance(q);
                    if d < bd {
                        bd = d;
                        bt = t;
                    }
                }
                cubic_point(c, bt)
            }
            WirePath::Ortho(p) => crate::ortho::nearest(p, q),
        }
    }

    /// Halfway along.
    pub(crate) fn mid(&self) -> Pos2 {
        match self {
            WirePath::Curve(c) => cubic_point(c, 0.5),
            WirePath::Ortho(p) => crate::ortho::point_at(p, 0.5),
        }
    }

    pub(crate) fn shape(&self, stroke: Stroke) -> egui::Shape {
        match self {
            WirePath::Curve(c) => egui::epaint::CubicBezierShape::from_points_stroke(
                *c,
                false,
                Color32::TRANSPARENT,
                stroke,
            )
            .into(),
            WirePath::Ortho(p) => egui::Shape::line(p.clone(), stroke),
        }
    }
}

/// An image wire between two sockets: `wire_ctrl_flow`'s curve, or the
/// right-angle route. A dot's end (`ts` / `td` set) has no direction of
/// its own in either style — in right angles, the dot is the corner.
/// `side`: the destination is a side socket, entered travelling that way
/// along x (`side_entry`) — right angles only; a curve keeps its drop.
#[allow(clippy::too_many_arguments)]
fn wire_path(
    from: Pos2,
    to: Pos2,
    zoom: f32,
    ts: Option<Vec2>,
    td: Option<Vec2>,
    side: Option<f32>,
    ortho: bool,
) -> WirePath {
    if boring() {
        return WirePath::Ortho(vec![from, to]);
    }
    if !ortho {
        return WirePath::Curve(wire_ctrl_flow(from, to, zoom, ts, td));
    }
    if let Some(r) = ORTHO_LANES.with(|l| l.borrow().get(&lane_key(from, to)).cloned()) {
        return WirePath::Ortho(r);
    }
    let (start, end) = ortho_ends(ts, td, side);
    WirePath::Ortho(crate::ortho::route(from, start, to, end, &crate::ortho::Style::at(zoom)))
}

/// How a right-angle wire meets its ends: a dot's end is free (the dot is
/// the corner), a side socket is entered sideways, the rest go down.
fn ortho_ends(
    ts: Option<Vec2>,
    td: Option<Vec2>,
    side: Option<f32>,
) -> (crate::ortho::End, crate::ortho::End) {
    use crate::ortho::End;
    let start = if ts.is_some() { End::Free } else { End::Down };
    let end = match (td, side) {
        (Some(_), _) => End::Free,
        (None, Some(s)) => End::Side(s),
        (None, None) => End::Down,
    };
    (start, end)
}

type LaneKey = (u32, u32, u32, u32);

fn lane_key(a: Pos2, b: Pos2) -> LaneKey {
    (a.x.to_bits(), a.y.to_bits(), b.x.to_bits(), b.y.to_bits())
}

thread_local! {
    /// This frame's laned right-angle routes (`crate::ortho::lanes`), keyed
    /// by their exact screen ends. Filled once per graph frame before the
    /// hit tests; a wire whose ends moved since then simply misses and gets
    /// its plain route.
    static ORTHO_LANES: std::cell::RefCell<std::collections::HashMap<LaneKey, Vec<Pos2>>> =
        Default::default();
}

#[cfg(not(test))]
fn lanes_enabled() -> bool {
    true
}

#[cfg(test)]
thread_local! {
    /// Lets a render test draw the same graph without lanes, for a before.
    static LANES_OFF: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
fn lanes_enabled() -> bool {
    !LANES_OFF.with(|l| l.get())
}

fn set_ortho_lanes(routes: std::collections::HashMap<LaneKey, Vec<Pos2>>) {
    ORTHO_LANES.with(|l| *l.borrow_mut() = routes);
}

/// A wire with no dot at either end: the one being dragged or held.
fn plain_wire_path(from: Pos2, to: Pos2, zoom: f32, ortho: bool) -> WirePath {
    if boring() {
        return WirePath::Ortho(vec![from, to]);
    }
    if ortho {
        wire_path(from, to, zoom, None, None, None, true)
    } else {
        WirePath::Curve(wire_ctrl(from, to, zoom))
    }
}

/// An expression cable: it runs SIDEWAYS, leaving along `sign` x and
/// arriving the same way (positive: left to right).
fn cable_path(from: Pos2, to: Pos2, zoom: f32, sign: f32, ortho: bool) -> WirePath {
    if boring() {
        return WirePath::Ortho(vec![from, to]);
    }
    if ortho {
        use crate::ortho::End;
        let st = crate::ortho::Style::at(zoom);
        return WirePath::Ortho(crate::ortho::route(from, End::Side(sign), to, End::Side(sign), &st));
    }
    let d = ((to.x - from.x).abs() * 0.5).max(30.0) * zoom.min(1.5) * sign.signum();
    WirePath::Curve([from, from + Vec2::new(d, 0.0), to - Vec2::new(d, 0.0), to])
}

/// How far a heavy node's halo reaches, as a multiple of the node's own
/// radius, and how wide the light beside a lit wire runs. TIGHT on
/// purpose: the glow has to say which node BURNS, and light that spills a
/// node's width in every direction says "this corner of the graph" — the
/// eye lands on the wash, not on the node under it. Brightness carries the
/// weight instead, so a smaller halo is a harder one — and the innermost
/// ring runs WHITE-hot, because heat reads as whiteness, not as more red.
const HEAVY_REACH: f32 = 2.0;
const LIT_WIRE_HALF: f32 = 16.0;

/// How a node's name searches for room: level first, then swinging UP
/// around the node's centre in coarse steps to vertical. Up, because the
/// wires leave a node DOWNWARD — below is where the graph is busy. The
/// first clear step is then bisected back towards level, so the name tilts
/// exactly as far as it has to and a dragged neighbour moves it smoothly
/// rather than in jumps.
const LABEL_STEP_DEG: f32 = 5.0;
const LABEL_BISECT: usize = 5;

/// The spring a name's angle rides to its target: the pie's and the
/// stalks' underdamped one (k, damping), so it overshoots, bounces and
/// settles like everything else in the graph that moves on its own.
const LABEL_SPRING: (f32, f32) = (260.0, 10.0);

/// One step of that spring at wobbliness `w`: semi-implicit Euler, stable
/// at the frame times the editor clamps to. Returns (angle, angular
/// velocity).
fn label_spring(theta: f32, vel: f32, target: f32, dt: f32, w: f32) -> (f32, f32) {
    let Some((k, c)) = wobble_spring(LABEL_SPRING.0, LABEL_SPRING.1, w) else {
        return (target, 0.0);
    };
    let v = vel + ((target - theta) * k - vel * c) * dt;
    (theta + v * dt, v)
}

/// The top of the graph bar's "wobble" slider.
pub const WOBBLE_MAX: f32 = 3.0;

/// A graph spring (stiffness `k`, damping `c`, as designed) at wobbliness
/// `w`. `None` at 0: rigid, the caller snaps to its target. Under 1 the
/// damping ratio slides toward critical, so the overshoot shrinks to none;
/// over 1 it falls as 1/w, so the swing grows and rings longer. Stiffness
/// never changes — the timing stays the graph's, and the Euler step stays
/// stable at the clamped frame time.
pub fn wobble_spring(k: f32, c: f32, w: f32) -> Option<(f32, f32)> {
    if w <= 0.005 {
        return None;
    }
    let crit = 2.0 * k.sqrt();
    let z0 = c / crit;
    let z = if w <= 1.0 { 1.0 + (z0 - 1.0) * w } else { z0 / w };
    Some((k, z * crit))
}

/// Where Layer input `k` sits while input `from` is being slid to `to`:
/// the slid one at `to`, the ones it passes one step back towards `from`.
fn reorder_slot(k: usize, from: usize, to: usize) -> usize {
    if k == from {
        to
    } else if from < to && k > from && k <= to {
        k - 1
    } else if from > to && k >= to && k < from {
        k + 1
    } else {
        k
    }
}

/// How long a Layer nodule takes to glide to its new slot.
const GLIDE_SECS: f32 = 0.28;

/// A pie's opening spring (value, velocity) one step toward 1.
fn pie_spring(s: f32, v: f32, dt: f32, w: f32) -> (f32, f32) {
    let Some((k, c)) = wobble_spring(260.0, 10.0, w) else {
        return (1.0, 0.0);
    };
    let v = v + ((1.0 - s) * k - v * c) * dt;
    (s + v * dt, v)
}

/// A label laid along a ray out of its node's centre: `start` px out, the
/// text's run along the ray and its height across it. The four corners of
/// that oriented box, in screen space.
fn label_box(c: Pos2, start: f32, size: Vec2, theta: f32) -> [Pos2; 4] {
    let u = Vec2::angled(theta);
    let v = Vec2::new(-u.y, u.x) * (size.y * 0.5);
    let (a, b) = (c + u * start, c + u * (start + size.x));
    [a - v, b - v, b + v, a + v]
}

/// Do two convex quads overlap? Separating axes on both quads' edge
/// normals — exact for boxes at any angle, and a handful of dot products.
fn quads_overlap(a: &[Pos2; 4], b: &[Pos2; 4]) -> bool {
    let axes = |q: &[Pos2; 4]| [q[1] - q[0], q[3] - q[0]];
    for axis in axes(a).into_iter().chain(axes(b)) {
        let n = Vec2::new(-axis.y, axis.x);
        let span = |q: &[Pos2; 4]| {
            q.iter().map(|p| p.to_vec2().dot(n)).fold((f32::MAX, f32::MIN), |(lo, hi), d| {
                (lo.min(d), hi.max(d))
            })
        };
        let ((a0, a1), (b0, b1)) = (span(a), span(b));
        if a1 <= b0 || b1 <= a0 {
            return false;
        }
    }
    true
}

/// A node's shape when it is wider than its circle, as (half extents,
/// x offset) in graph units at size 1: a TV-mode Read's cabinet, else a
/// merge bar or harp (`breakout::wide_rect`). Hit tests, name placement,
/// node rects and the selection box all ask HERE, so a wide shape is
/// grabbable along its whole body.
fn wide_of(n: &ENode) -> Option<(Vec2, f32)> {
    if tv_on(n) {
        return Some((tv_half(), 0.0));
    }
    if boring() && boring_shape(&n.kind) {
        return Some((BORING_HALF, 0.0));
    }
    crate::breakout::wide_rect(&n.kind)
}

/// Point in a convex quad (any winding): the name labels are ROTATED, so
/// an axis-aligned rect would claim clicks the text is nowhere near.
fn quad_contains(q: &[Pos2; 4], p: Pos2) -> bool {
    let mut sign = 0.0f32;
    for i in 0..4 {
        let (a, b) = (q[i], q[(i + 1) % 4]);
        let c = (b - a).x * (p - a).y - (b - a).y * (p - a).x;
        if c.abs() < 1e-6 {
            continue;
        }
        if sign == 0.0 {
            sign = c.signum();
        } else if c.signum() != sign {
            return false;
        }
    }
    true
}

fn rect_quad(r: Rect) -> [Pos2; 4] {
    [r.left_top(), r.right_top(), r.right_bottom(), r.left_bottom()]
}

/// Which angle a node's name is drawn at, swinging up around the node's
/// centre until it clears every OTHER node and every label already placed.
/// The first clear angle wins, so a name with room stays level; when none
/// is clear, the one that hits the fewest things. Returns the angle and
/// the box it lands in, for the next label to avoid.
fn place_label(
    own: u64,
    c: Pos2,
    start: f32,
    size: Vec2,
    nodes: &[(u64, Rect)],
    placed: &[[Pos2; 4]],
) -> (f32, [Pos2; 4]) {
    let hits_at = |theta: f32| -> (usize, [Pos2; 4]) {
        let q = label_box(c, start, size, theta);
        let aabb = Rect::from_points(&q);
        let hits = nodes
            .iter()
            .filter(|(id, r)| *id != own && r.intersects(aabb) && quads_overlap(&q, &rect_quad(*r)))
            .count()
            + placed
                .iter()
                .filter(|p| Rect::from_points(&p[..]).intersects(aabb) && quads_overlap(&q, p))
                .count();
        (hits, q)
    };
    let steps = (90.0 / LABEL_STEP_DEG).round() as usize;
    let mut best: Option<(usize, f32, [Pos2; 4])> = None;
    for i in 0..=steps {
        let theta = -(i as f32 * LABEL_STEP_DEG).to_radians();
        let (hits, q) = hits_at(theta);
        if hits == 0 {
            if i == 0 {
                return (theta, q);
            }
            // Clear here, blocked one step flatter: find the flattest clear
            // angle between the two, so the tilt is exactly what the room
            // asks for.
            let (mut lo, mut hi, mut hq) = (theta + LABEL_STEP_DEG.to_radians(), theta, q);
            for _ in 0..LABEL_BISECT {
                let mid = 0.5 * (lo + hi);
                match hits_at(mid) {
                    (0, mq) => (hi, hq) = (mid, mq),
                    _ => lo = mid,
                }
            }
            return (hi, hq);
        }
        if best.as_ref().is_none_or(|(h, _, _)| hits < *h) {
            best = Some((hits, theta, q));
        }
    }
    let (_, theta, q) = best.expect("at least one angle is tried");
    (theta, q)
}

/// Light, not paint: a premultiplied colour with ZERO alpha ADDS to what is
/// under it, so a glow brightens the wire and the grid instead of washing a
/// grey film over them. `k` scales it (0 = dark, 1 = the colour at full).
fn glow_light(c: Color32, k: f32) -> Color32 {
    let ch = |v: u8| (v as f32 * k.clamp(0.0, 1.0)) as u8;
    Color32::from_rgba_premultiplied(ch(c.r()), ch(c.g()), ch(c.b()), 0)
}

/// A soft halo: an additive ring mesh, `color` at `r0` fading to nothing at
/// `r1`. A gradient, not a stack of strokes — stacked strokes band, and
/// banding reads as rings rather than as light.
fn glow_ring(painter: &egui::Painter, at: Pos2, r0: f32, r1: f32, color: Color32) {
    if r1 <= r0 || r1 < 0.5 {
        return;
    }
    // Segments by size: a speck's halo is a few pixels across, and a
    // zoomed-out graph draws thousands of them.
    let n_seg = ((r1 * 1.5) as usize).clamp(12, 44);
    let mut mesh = egui::epaint::Mesh::default();
    let out = glow_light(color, 0.0);
    for i in 0..=n_seg {
        let a = i as f32 / n_seg as f32 * std::f32::consts::TAU;
        let (s, c) = a.sin_cos();
        let d = Vec2::new(c, s);
        mesh.colored_vertex(at + d * r0, color);
        mesh.colored_vertex(at + d * r1, out);
        if i > 0 {
            let k = (i as u32 - 1) * 2;
            mesh.add_triangle(k, k + 1, k + 2);
            mesh.add_triangle(k + 1, k + 3, k + 2);
        }
    }
    painter.add(egui::Shape::mesh(mesh));
}

/// The same falloff along a line: `color` on the path, fading to nothing
/// `half` away on either side. What makes a noodle look lit.
/// How a legend swatch draws its sample noodle.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Swatch {
    Plain,
    /// Lit: the light a heavy node's wires carry.
    Glow,
    /// The fat multi-channel cable.
    Fat,
}

/// Every colour a noodle can wear, in the order the legend lists them:
/// cost first (what the image wire says about the frame rate), then the
/// pipes that wear their own colour, then the hand's states. Read LIVE
/// from the theme, so a recoloured theme relabels itself.
pub fn legend_entries() -> Vec<(&'static str, Color32, Swatch)> {
    use crate::theme as t;
    vec![
        ("fused — fast", t::cost_fast(), Swatch::Plain),
        ("own pass — slow", t::cost_slow(), Swatch::Plain),
        ("below a heavy node", t::cost_slow(), Swatch::Glow),
        ("matte", t::wire_mask(), Swatch::Plain),
        ("data — depth, vectors", t::wire_data(), Swatch::Plain),
        ("multi-channel", t::wire_channels(), Swatch::Fat),
        ("hidden / expression", t::hidden_wire(), Swatch::Plain),
        ("held (G)", t::wire_held(), Swatch::Plain),
        ("locked (patch mode)", t::wire_locked(), Swatch::Plain),
    ]
}

fn glow_ribbon(painter: &egui::Painter, path: &[Pos2], half: f32, color: Color32) {
    if path.len() < 2 || half < 0.5 {
        return;
    }
    let mut mesh = egui::epaint::Mesh::default();
    let out = glow_light(color, 0.0);
    for (i, p) in path.iter().enumerate() {
        let d = if i == 0 {
            path[1] - path[0]
        } else if i == path.len() - 1 {
            path[i] - path[i - 1]
        } else {
            path[i + 1] - path[i - 1]
        };
        let n = if d.length() > 1e-4 {
            Vec2::new(-d.y, d.x) / d.length()
        } else {
            Vec2::new(0.0, 1.0)
        };
        mesh.colored_vertex(*p - n * half, out);
        mesh.colored_vertex(*p, color);
        mesh.colored_vertex(*p + n * half, out);
        if i > 0 {
            let k = (i as u32 - 1) * 3;
            for (a, b, c) in [(k, k + 1, k + 3), (k + 1, k + 4, k + 3), (k + 1, k + 2, k + 4), (k + 2, k + 5, k + 4)] {
                mesh.add_triangle(a, b, c);
            }
        }
    }
    painter.add(egui::Shape::mesh(mesh));
}

/// A star of light: a soft halo, four spikes and a hot core. What a node
/// scaled down to a speck wears instead of a shape nobody can see —
/// `lit` (0..1) fades the whole thing in as the node shrinks, `tw` is its
/// twinkle, and `tint` keeps a hint of the node's own type colour.
fn twinkle_star(painter: &egui::Painter, at: Pos2, lit: f32, tw: f32, tint: Color32) {
    let light = theme::node_twinkle();
    glow_ring(painter, at, 0.0, (6.0 + 15.0 * tw) * lit, glow_light(tint, 0.42 * lit));
    let spike = (5.0 + 15.0 * tw) * lit;
    let hot = glow_light(light, (0.40 + 0.60 * tw) * lit);
    let mut mesh = egui::epaint::Mesh::default();
    let fade = glow_light(light, 0.0);
    for (k, d) in [Vec2::X, Vec2::Y, -Vec2::X, -Vec2::Y].iter().enumerate() {
        let n = Vec2::new(-d.y, d.x) * (0.55 + 0.35 * lit);
        let b = k as u32 * 3;
        mesh.colored_vertex(at - n, hot);
        mesh.colored_vertex(at + n, hot);
        mesh.colored_vertex(at + *d * spike, fade);
        mesh.add_triangle(b, b + 1, b + 2);
    }
    painter.add(egui::Shape::mesh(mesh));
    // The core stays put whatever the twinkle does: a speck must never
    // blink out of existence, it only gets brighter.
    let core = (1.5 + 0.9 * tw) * lit;
    if core > 0.2 {
        painter.circle_filled(at, core, light.gamma_multiply(0.7 + 0.3 * tw));
    }
}

/// Where in its twinkle a node is: its own phase off the id, THREE beats at
/// odd ratios so the field never pulses in lockstep and no speck repeats on
/// a rhythm the eye can follow. Fast enough to read as scintillation rather
/// than breathing — the slow pair it started with looked like the node was
/// idling — and the swing is symmetrical about the middle, so the average
/// speck is as bright as it ever was and only the extremes moved. 0..1.
fn twinkle_phase(id: u64, t: f32) -> f32 {
    let p = (id % 977) as f32 * 0.618;
    let a = (t * 3.4 + p).sin();
    let b = (t * 5.9 + p * 1.7).sin();
    let c = (t * 9.1 + p * 2.9).sin();
    (0.5 + 0.30 * a + 0.14 * b + 0.06 * c).clamp(0.0, 1.0)
}

/// A dot end's automatic handle length on a hop, in screen units. A dot's
/// tangent is its flow direction times its tension, and this is what that
/// multiplies — shared with the tension pull so the gesture and the drawn
/// curve measure the same thing.
fn dot_handle_len(from: Pos2, to: Pos2, zoom: f32) -> f32 {
    (from.distance(to) * 0.35).clamp(10.0, 120.0) * zoom.min(1.5)
}

/// The most a dot's tension can pull (see `ENode::tension`).
pub const DOT_TENSION_MAX: f32 = 4.0;

/// The tension a pull sets: the pointer's distance from the bead over the
/// bead's automatic handle length, so the handle tips sit where the pointer
/// is. Snaps to 1 near it — the automatic curve is easy to find again.
pub fn dot_tension_for(distance: f32, reference: f32) -> f32 {
    let t = (distance / reference.max(1.0)).clamp(0.0, DOT_TENSION_MAX);
    if (t - 1.0).abs() < 0.06 {
        1.0
    } else {
        t
    }
}

/// What a `.` press landed on.
enum DotPress {
    /// An existing dot: pull its tension.
    Dot(u64),
    /// A wire: cut a dot in at this graph point, then pull.
    Wire((u64, usize, u64), Pos2),
}

/// A live `.` pull. `reference` is the bead's automatic handle length on
/// screen at the press; `pulling` flips once the pointer leaves the press,
/// so a click leaves the tension alone.
#[derive(Clone, Copy)]
struct DotPlace {
    id: u64,
    press: Pos2,
    reference: f32,
    pulling: bool,
}

pub(crate) fn cubic_point(p: &[Pos2; 4], t: f32) -> Pos2 {
    let u = 1.0 - t;
    let w = [u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t];
    Pos2::new(
        w[0] * p[0].x + w[1] * p[1].x + w[2] * p[2].x + w[3] * p[3].x,
        w[0] * p[0].y + w[1] * p[1].y + w[2] * p[2].y + w[3] * p[3].y,
    )
}

fn segs_intersect(a: Pos2, b: Pos2, c: Pos2, d: Pos2) -> bool {
    let cross = |o: Pos2, p: Pos2, q: Pos2| (p.x - o.x) * (q.y - o.y) - (p.y - o.y) * (q.x - o.x);
    let (d1, d2) = (cross(c, d, a), cross(c, d, b));
    let (d3, d4) = (cross(a, b, c), cross(a, b, d));
    ((d1 > 0.0) != (d2 > 0.0)) && ((d3 > 0.0) != (d4 > 0.0))
}

/// Does segment a-b cross the wire? (bezier approximated by 16 segments)
pub(crate) fn bezier_hits_segment(p: &[Pos2; 4], a: Pos2, b: Pos2) -> bool {
    let mut prev = p[0];
    for i in 1..=16 {
        let q = cubic_point(p, i as f32 / 16.0);
        if segs_intersect(a, b, prev, q) {
            return true;
        }
        prev = q;
    }
    false
}

fn point_seg_dist(p: Pos2, a: Pos2, b: Pos2) -> f32 {
    let ab = b - a;
    let len2 = ab.length_sq();
    let t = if len2 <= 1e-6 { 0.0 } else { ((p - a).dot(ab) / len2).clamp(0.0, 1.0) };
    (a + ab * t - p).length()
}

pub(crate) fn bezier_min_dist(p: &[Pos2; 4], q: Pos2) -> f32 {
    let mut best = f32::MAX;
    let mut prev = p[0];
    for i in 1..=16 {
        let s = cubic_point(p, i as f32 / 16.0);
        best = best.min(point_seg_dist(q, prev, s));
        prev = s;
    }
    best
}

impl NodeEditor {
    /// The dot a `.` pull is holding and its reference handle length, while
    /// one is live (self-tests read it).
    pub fn dot_pull(&self) -> Option<(u64, f32)> {
        self.dot_place.map(|d| (d.id, d.reference))
    }

    /// Draw + interact. Maintains `selection` (plain click selects, shift
    /// adds, background click / Esc clears) and fills `node_rects` with each
    /// node's screen rect (used by --test-click).
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    /// Is the range on the noodle out of `src` decided AT `src` — an
    /// origin (a Read, a generator, an override, a retime), or a node whose
    /// inputs disagree with what comes out (a merge that widens) — rather
    /// than carried through from its input unchanged?
    pub(crate) fn range_set_here(&self, graph: &EditorGraph, src: u64) -> bool {
        use crate::ranges::Source;
        let Some(&r) = self.wire_ranges.get(&src) else { return false };
        match self.wire_range_src.get(&src) {
            Some(Source::Override | Source::Media | Source::Retimed) => return true,
            Some(Source::Script) => return false,
            _ => {}
        }
        let Some(n) = graph.node(src) else { return false };
        let ins: Vec<(u64, u64)> = (0..n.kind.input_count())
            .filter_map(|i| graph.input_of(src, i))
            .filter_map(|s| self.wire_ranges.get(&s).copied())
            .collect();
        !ins.is_empty() && ins.iter().any(|x| *x != r)
    }

    /// The hover's words for a noodle's range: the span, its length, where
    /// it came from, how it sits against the script, and (out of a Read)
    /// what the plate does past its ends.
    pub(crate) fn range_detail(&self, graph: &EditorGraph, src: u64, (a, b): (u64, u64)) -> String {
        let mut s = format!("{a}–{b}  {} fr", b.saturating_sub(a) + 1);
        if let Some(from) = self.wire_range_src.get(&src) {
            s.push_str("  · ");
            s.push_str(from.label());
            if self.range_set_here(graph, src) && *from == crate::ranges::Source::Upstream {
                s.push_str(" (widened here)");
            }
        }
        let (sa, sb) = self.script_range;
        let mut against = Vec::new();
        if a > sa {
            against.push(format!("starts {} in", a - sa));
        } else if a < sa {
            against.push(format!("{} before", sa - a));
        }
        if b < sb {
            against.push(format!("ends {} early", sb - b));
        } else if b > sb {
            against.push(format!("{} past", b - sb));
        }
        if against.is_empty() {
            s.push_str(&format!("\nscript {sa}–{sb}: matches"));
        } else {
            s.push_str(&format!("\nscript {sa}–{sb}: {}", against.join(", ")));
        }
        if let Some(NodeKind::Read { edges, .. }) = graph.node(src).map(|n| &n.kind) {
            s.push_str(&format!(
                "\noutside: {} before, {} after",
                edges.before.label(),
                edges.after.label()
            ));
        }
        s
    }

    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        graph: &mut EditorGraph,
        selection: &mut Vec<u64>,
        activated: &mut Vec<u64>,
        gizmo_pinned: &mut Vec<u64>,
        next_read_slot: &mut usize,
        node_rects: &mut Vec<(u64, Rect)>,
    ) -> bool {
        node_rects.clear();
        // Node names, collected while the nodes draw and placed after.
        let mut labels: Vec<(u64, Pos2, f32, String, f32)> = Vec::new();
        if let Some(id) = self.focus.take() {
            if let Some(n) = graph.node(id) {
                self.pan = -n.pos.to_vec2() * self.zoom;
                self.suppress_nav_mirror = true;
            }
        }
        // A dot wears the kind of whatever reaches it. Re-derived rather
        // than maintained: rewiring anything upstream of a chain of dots
        // would otherwise leave them the wrong colour.
        graph.sync_dot_kinds();
        // v1 rectangular backdrops adopt their contents as members.
        graph.migrate_groups();
        // Harps keep their hidden machinery true to their face: kids, the
        // x-lock, the receiver's input count. Idempotent, like the two
        // passes above.
        crate::breakout::sync(graph);
        let harp_kids = crate::breakout::kid_set(graph);
        // Claim the whole panel (a resizable panel collapses to content size
        // if nothing allocates space).
        let (rect, resp) =
            ui.allocate_exact_size(ui.available_size(), egui::Sense::click_and_drag());
        if let Some(prev) = self.last_center {
            let d = prev - rect.center();
            if d != Vec2::ZERO {
                self.pan += d;
            }
        }
        self.last_center = Some(rect.center());
        let overlay = self.overlay_view_rect.is_some();
        // One flag for every expression view: collapse rgb triples to one
        // port, or show the per-lane links as the model really holds them.
        let expand_rgb = self.expand_patch_rgb;
        let vrect = self.overlay_view_rect.unwrap_or(rect);
        let painter = ui.painter_at(rect);
        if !overlay {
            painter.rect_filled(rect, 0.0, theme::graph_bg());
        }
        let view = View { origin: vrect.center() + self.pan, zoom: self.zoom };
        // Which region the view is IN, this frame: the wash under the view
        // centre. Computed before the hotkeys run so the backtick pie opens
        // on this frame's answer, not last frame's.
        self.active_region = crate::regions::active(graph, view.to_graph(vrect.center()));

        // Faint dot grid.
        let step = 60.0 * self.zoom;
        if !overlay && step > 8.0 {
            let gx0 = ((rect.min.x - view.origin.x) / step).floor();
            let gy0 = ((rect.min.y - view.origin.y) / step).floor();
            let mut x = view.origin.x + gx0 * step;
            while x < rect.max.x {
                let mut y = view.origin.y + gy0 * step;
                while y < rect.max.y {
                    painter.circle_filled(Pos2::new(x, y), 1.0, theme::graph_grid());
                    y += step;
                }
                x += step;
            }
        }

        let pointer = resp.hover_pos().or_else(|| ui.ctx().pointer_latest_pos());

        // --- wheel / trackpad: Blender's 2D-editor rules (`crate::nav`) —
        // wheel zooms about the cursor, two trackpad fingers PAN, pinch and
        // ctrl+anything zoom, ctrl/shift+wheel pan by axis.
        if resp.hovered() {
            let w = crate::nav::wheel_nav(ui);
            if (w.zoom - 1.0).abs() > 1e-4 {
                if let Some(c) = pointer {
                    let g = view.to_graph(c);
                    // "Unlimited within reason": deep zoom-in for tiny nodes,
                    // far zoom-out for huge graphs.
                    self.zoom = (self.zoom * w.zoom).clamp(0.002, 200.0);
                    // Keep the graph point under the cursor fixed.
                    self.pan = (c - vrect.center()) - g.to_vec2() * self.zoom;
                }
            }
            self.pan += w.pan;
        }
        let view = View { origin: vrect.center() + self.pan, zoom: self.zoom };

        // A `.` pull in progress: the dot's tension follows the pointer's
        // distance from the bead. Applied before the wires are laid out
        // below, so they bend this frame rather than the next.
        if let Some(mut pull) = self.dot_place {
            let bead = graph.node(pull.id).map(|n| view.to_screen(n.pos));
            match (ui.input(|i| i.pointer.primary_down()), bead, pointer) {
                (true, Some(bead), Some(p)) => {
                    pull.pulling |= p.distance(pull.press) > 4.0;
                    if pull.pulling {
                        if let Some(n) = graph.node_mut(pull.id) {
                            n.tension = dot_tension_for(p.distance(bead), pull.reference);
                        }
                    }
                    self.dot_place = Some(pull);
                }
                (true, Some(_), None) => {}
                _ => self.dot_place = None,
            }
        }

        // --- stalk physics: every stalk emits from the shape center and its
        // tip springs (underdamped, so it wobbles) toward a splayed rest
        // point; dragging a node makes the stalks flex and settle ---
        let dt = ui.input(|i| i.stable_dt).clamp(0.001, 0.05);
        // ⌘↑/⌘↓ swap animation: positions change instantly in the graph, the
        // DRAWING carries a decaying offset so the two nodes glide past each
        // other. Everything visual (stalks, hits, wires) uses drawn positions.
        self.anim
            .retain(|id, (v, _)| graph.nodes.contains_key(id) && v.length() > 0.5);
        for (v, delay) in self.anim.values_mut() {
            if *delay > 0.0 {
                *delay -= dt;
            } else {
                *v *= (-dt * 12.0).exp();
            }
        }
        if !self.anim.is_empty() {
            // The cascade must not stall waiting for input.
            ui.ctx().request_repaint();
        }
        let drawn: std::collections::HashMap<u64, (Pos2, f32, f32)> = graph
            .nodes
            .iter()
            .map(|(id, n)| {
                let dot = matches!(n.kind, NodeKind::Dot { .. });
                (
                    *id,
                    (
                        n.pos + self.anim.get(id).map(|(v, _)| *v).unwrap_or(Vec2::ZERO),
                        n.size,
                        if dot { 0.0 } else { n.size },
                    ),
                )
            })
            .collect();
        let dp = |id: u64| drawn.get(&id).map(|(p, ..)| *p).unwrap_or(Pos2::ZERO);
        let ds = |id: u64| drawn.get(&id).map(|(_, s, _)| *s).unwrap_or(1.0);
        // Stalk reach. A dot has none: its sockets sit ON the bead, so a
        // wire runs through it as one continuous line rather than jumping
        // two stalks stuck out of a full stop.
        let dz = |id: u64| drawn.get(&id).map(|(.., z)| *z).unwrap_or(1.0);

        // Each dot's flow direction: from its source toward its consumers,
        // in DRAWN positions, scaled by the dot's tension. Feeds
        // `wire_ctrl_flow`, so every wire hit test and every drawn wire agree
        // on the same curve.
        let dot_flow: std::collections::HashMap<u64, Vec2> = graph
            .nodes
            .iter()
            .filter(|(_, n)| matches!(n.kind, NodeKind::Dot { .. }))
            .map(|(id, dot)| {
                let prev = graph.input_of(*id, 0).map(&dp).unwrap_or(dp(*id));
                let (mut next, mut n) = (Vec2::ZERO, 0.0f32);
                for (d, _, src) in &graph.edges {
                    if src == id {
                        next += dp(*d).to_vec2();
                        n += 1.0;
                    }
                }
                let next = if n > 0.0 { (next / n).to_pos2() } else { dp(*id) };
                let v = next - prev;
                let dir = if v.length() > 1e-3 { v / v.length() } else { Vec2::new(0.0, 1.0) };
                (*id, dir * dot.tension)
            })
            .collect();
        let tan = |id: u64| dot_flow.get(&id).copied();

        // Mask stalks bias horizontal, leaning toward the nearest mask
        // output weighted by distance (close source = point right at it).
        // The mask outputs, gathered once rather than per masked node.
        let mask_outs: Vec<(u64, Pos2)> = graph
            .nodes
            .iter()
            .filter(|(_, sn)| sn.kind.output_kind() == PortKind::Mask)
            .map(|(sid, _)| (*sid, dp(*sid)))
            .collect();
        let mask_dirs: std::collections::HashMap<u64, Vec2> = graph
            .nodes
            .iter()
            .filter(|(_, n)| n.kind.mask_input().is_some() || n.kind.mod_input().is_some())
            .map(|(id, _)| {
                let p = dp(*id);
                let mut best: Option<(f32, Vec2)> = None;
                for (sid, sp) in &mask_outs {
                    if sid == id {
                        continue;
                    }
                    let d = *sp - p;
                    let len = d.length();
                    if best.is_none_or(|(bl, _)| len < bl) {
                        best = Some((len, d));
                    }
                }
                let dir = match best {
                    Some((len, d)) if len > 1.0 => {
                        let to = d / len;
                        let w = (1.0 - len / 700.0).clamp(0.0, 1.0);
                        let horiz = Vec2::new(if to.x < 0.0 { -1.0 } else { 1.0 }, 0.0);
                        let v = horiz * (1.0 - w) + to * w;
                        if v.length() > 1e-3 { v / v.length() } else { horiz }
                    }
                    _ => Vec2::new(-1.0, 0.0),
                };
                (*id, dir)
            })
            .collect();
        // How a right-angle wire enters input `input` of `id`: sideways into a
        // side socket, from above otherwise (the graph is passed in, so this
        // holds no borrow across the edits below).
        let side_of = |g: &EditorGraph, id: u64, input: usize| {
            socket_of(g, &mask_dirs, id, input, dp(id), ds(id)).1.and_then(side_entry)
        };

        // --- magnetism: a free input and a compatible output that come
        // close reach for each other (the stalks bend toward the midpoint,
        // strongest when nearest). Holding shift closes the deal. Pairs are
        // chosen mutually-nearest, so a socket only ever courts one partner.
        let prev_tip = |id: u64, k: usize, fallback: Pos2| -> Pos2 {
            self.stalks.get(&(id, k)).map(|(p, _)| *p).unwrap_or(fallback)
        };
        let reach_of = |a: f32, b: f32| self.magnet_reach * (a + b) * 0.5;
        // (dst, input, src, src OUTPUT, pull, input tip, output tip). The
        // output is part of the pair: a multi-band LumaKey or a harp courts
        // with every socket it has, and shift wires the one that courted.
        let mut pairs: Vec<(u64, usize, u64, usize, f32, Pos2, Pos2)> = Vec::new();
        // Every output that may court, with its tip, once per frame (a
        // hidden harp kid's output courts nobody).
        // Sorted by x, so each input looks only at the slice within reach
        // instead of at every output in the graph.
        let mut outs: Vec<(u64, usize, Pos2, PortKind, f32)> = if self.link_mode || self.patch_mode {
            Vec::new()
        } else {
            graph
                .nodes
                .iter()
                .filter(|(sid, _)| !harp_kids.contains(sid))
                .flat_map(|(sid, sn)| {
                    (0..sn.kind.output_count().clamp(1, MAX_OUTPUTS)).map(move |k| (*sid, sn, k))
                })
                .map(|(sid, sn, k)| {
                    let op = prev_tip(sid, out_stalk(k), output_rest_of(&graph.nodes, sid, k, dz(sid)));
                    (sid, k, op, sn.kind.output_kind_at(k), ds(sid))
                })
                .collect()
        };
        outs.sort_by(|a, b| a.2.x.total_cmp(&b.2.x));
        let max_s = outs.iter().map(|o| o.4).fold(0.0f32, f32::max);
        // The outputs whose x lies within the largest reach of `p`.
        let near = |p: Pos2, s: f32| {
            let r = reach_of(s, max_s);
            let lo = outs.partition_point(|o| o.2.x < p.x - r);
            let hi = outs.partition_point(|o| o.2.x <= p.x + r);
            &outs[lo..hi]
        };
        // Every node's feeds, indexed once: `input_of` and a filter over
        // every edge, per node, were quadratic in a big graph.
        let mut feeds_of: std::collections::HashMap<u64, Vec<(usize, u64)>> =
            std::collections::HashMap::new();
        for (d, i, s) in &graph.edges {
            feeds_of.entry(*d).or_default().push((*i, *s));
        }
        let fed = |d: u64, i: usize| feeds_of.get(&d).is_some_and(|f| f.iter().any(|(fi, _)| *fi == i));
        for (did, dn) in graph.nodes.iter().filter(|(id, _)| {
            !self.link_mode && !self.patch_mode && !harp_kids.contains(id)
        }) {
            // Sources already feeding this node. A Blend whose + is wired
            // must not court the SAME output with its −: shift is held
            // through a whole drag, so the frame after the first join the
            // second port would otherwise grab the identical wire and the
            // merge would composite a plate over itself.
            let feeding: Vec<(u64, usize)> = feeds_of
                .get(did)
                .map(|f| f.iter().map(|(i, s)| (*s, graph.out_of(*did, *i) as usize)).collect())
                .unwrap_or_default();
            // A big-dot node courts with its DOT: any output that fits one
            // of its free inputs. Shift lands it as a drop on the dot does.
            if big_dot(&dn.kind) {
                let ip = prev_tip(*did, BIG_DOT_STALK, big_dot_rest(graph, *did, dp(*did), ds(*did)));
                let mut best: Option<(f32, u64, usize, Pos2)> = None;
                // Distance first: it is the cheap test and almost always
                // the one that says no; the graph walk runs only for an
                // output already within reach.
                for &(sid, k, op, _, sds) in near(ip, ds(*did)) {
                    if sid == *did {
                        continue;
                    }
                    let d = (op - ip).length();
                    if !(d < reach_of(ds(*did), sds) && best.is_none_or(|(bd, ..)| d < bd)) {
                        continue;
                    }
                    if feeding.contains(&(sid, k)) || reaches(graph, sid, *did) {
                        continue;
                    }
                    let fits = big_dot_options(graph, sid, k as u32, *did);
                    if !fits.iter().any(|i| !fed(*did, *i)) {
                        continue;
                    }
                    best = Some((d, sid, k, op));
                }
                if let Some((d, sid, k, op)) = best {
                    let reach = reach_of(ds(*did), ds(sid));
                    pairs.push((*did, BIG_DOT_STALK, sid, k, (1.0 - d / reach).clamp(0.0, 1.0), ip, op));
                }
            }
            for i in 0..dn.kind.input_count() {
                if fed(*did, i) {
                    continue; // taken: no courting
                }
                if big_dot(&dn.kind) {
                    continue; // behind the big dot: drop on the dot instead
                }
                let wants = &dn.kind;
                let ip = prev_tip(
                    *did,
                    i,
                    rest_of(graph, &mask_dirs, *did, i, dp(*did), ds(*did)),
                );
                let mut best: Option<(f32, u64, usize, Pos2)> = None;
                for &(sid, k, op, kind_out, sds) in near(ip, ds(*did)) {
                    if sid == *did {
                        continue;
                    }
                    // Distance first (cheap), the graph walk last.
                    let d = (op - ip).length();
                    let reach = reach_of(ds(*did), sds);
                    if !(d < reach && best.is_none_or(|(bd, ..)| d < bd)) {
                        continue;
                    }
                    if !wants.accepts(i, kind_out) {
                        continue;
                    }
                    // Already wired into another of this node's ports:
                    // one output, one port, never both. Per OUTPUT: band
                    // 0 feeding one port does not stop band 1 courting
                    // another.
                    if feeding.contains(&(sid, k)) {
                        continue;
                    }
                    // Never court a downstream node: that edge is a cycle.
                    if reaches(graph, sid, *did) {
                        continue;
                    }
                    best = Some((d, sid, k, op));
                }
                if let Some((d, sid, k, op)) = best {
                    let reach = reach_of(ds(*did), ds(sid));
                    pairs.push((*did, i, sid, k, (1.0 - d / reach).clamp(0.0, 1.0), ip, op));
                }
            }
        }
        // One node's ports court an output in PORT ORDER, not by distance:
        // a free main input first (a Blur reaches for a matte with its
        // picture port, which takes the matte pipe), then a mask, then a
        // modulate. Only the winner reaches, so the stalk shows which port
        // shift will wire. Ports of one rank (a Blend's + and −) still go
        // by distance below.
        {
            let rank = |did: u64, i: usize| {
                graph.node(did).map_or(0, |n| port_rank(&n.kind, i))
            };
            let mut first: std::collections::HashMap<(u64, u64, usize), (u8, f32)> =
                std::collections::HashMap::new();
            for (did, i, sid, k, w, ..) in &pairs {
                let r = rank(*did, *i);
                let e = first.entry((*did, *sid, *k)).or_insert((r, *w));
                if (r, -*w) < (e.0, -e.1) {
                    *e = (r, *w);
                }
            }
            pairs.retain(|(did, i, sid, k, w, ..)| {
                first.get(&(*did, *sid, *k)).is_some_and(|(r, bw)| *r == rank(*did, *i) && *bw == *w)
            });
        }
        // One suitor per output: the closest input wins it. That is also
        // what stops a node with two free ports — a Blend's + and − — from
        // offering BOTH to one output; only the nearer one courts it.
        pairs.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap_or(std::cmp::Ordering::Equal));
        {
            let mut taken = std::collections::HashSet::new();
            pairs.retain(|(_, _, sid, k, ..)| taken.insert((*sid, *k)));
        }
        let mut pull: std::collections::HashMap<(u64, usize), (Pos2, f32)> =
            std::collections::HashMap::new();
        // INPUTS reach; outputs hold still. When both bent toward the
        // midpoint, two nodes side by side each had a socket leaning at the
        // other and nothing said which end was which. An output stays on
        // its node, so the moving tip IS the input.
        for (did, i, _sid, _k, w, _ip, op) in &pairs {
            pull.insert((*did, *i), (*op, *w));
        }

        const K: f32 = 320.0; // spring stiffness
        const C: f32 = 11.0; // damping
        let spring = wobble_spring(K, C, self.wobble);
        for (id, n) in &graph.nodes {
            let n_in = n.kind.input_count();
            let (p, sz) = (dp(*id), ds(*id));
            let mut targets: Vec<(usize, Pos2)> =
                (0..n_in).map(|i| (i, rest_of(graph, &mask_dirs, *id, i, p, sz))).collect();
            if big_dot(&n.kind) {
                targets.push((BIG_DOT_STALK, big_dot_rest(graph, *id, p, sz)));
            }
            // One stalk per output. `out_stalk(0)` IS `OUT_STALK`, so a
            // single-output node keys and rests exactly as it always did.
            let n_o = n.kind.output_count().clamp(1, MAX_OUTPUTS);
            for k in 0..n_o {
                targets.push((out_stalk(k), output_rest_of(&graph.nodes, *id, k, dz(*id))));
            }
            // A nodule being slid along this bar rides the hand; the others
            // step aside into the slots they would have after the drop.
            let live = match self.drag {
                Some(DragKind::LayerReorder { node, from, .. }) if node == *id => {
                    Some((from, self.reorder_to.unwrap_or(from)))
                }
                _ => None,
            };
            let mut targets = targets;
            if let Some((from, to)) = live {
                for (k, t) in targets.iter_mut() {
                    if *k >= n_in {
                        continue;
                    }
                    *t = if *k == from {
                        match pointer {
                            Some(q) => Pos2::new(view.to_graph(q).x, t.y),
                            None => *t,
                        }
                    } else {
                        rest_of(graph, &mask_dirs, *id, reorder_slot(*k, from, to), p, sz)
                    };
                }
            }
            for (k, mut target) in targets {
                if live.is_some() && k < n_in {
                    let e = self.stalks.entry((*id, k)).or_insert((target, Vec2::ZERO));
                    let a = 1.0 - (-dt * 22.0).exp();
                    *e = (e.0 + (target - e.0) * a, Vec2::ZERO);
                    continue;
                }
                if let Some((to, w)) = pull.get(&(*id, k)).filter(|_| !boring()) {
                    // Lean toward the partner's socket: at touching distance
                    // the input tip sits most of the way there.
                    target += (*to - target) * (0.85 * w);
                }
                let e = self.stalks.entry((*id, k)).or_insert((target, Vec2::ZERO));
                // Boring: a socket is part of the body and moves with it.
                if let Some((k, c)) = spring.filter(|_| !boring()) {
                    let acc = (target - e.0) * k - e.1 * c;
                    e.1 += acc * dt;
                    e.0 += e.1 * dt;
                } else {
                    *e = (target, Vec2::ZERO);
                }
                if let Some((from, t)) = self.glides.get_mut(&(*id, k)) {
                    *t = (*t + dt / GLIDE_SECS).min(1.0);
                    let u = *t * *t * (3.0 - 2.0 * *t);
                    *e = (*from + (target - *from) * u, Vec2::ZERO);
                }
            }
        }
        self.glides.retain(|_, (_, t)| *t < 1.0);
        if !self.glides.is_empty() {
            ui.ctx().request_repaint();
        }
        // Shift while a pair is courting = wire it up (idempotent, so
        // holding shift through a drag keeps the newest pairing).
        if !ui.ctx().egui_wants_keyboard_input()
            && ui.input(|i| i.modifiers.shift)
            && resp.hovered()
        {
            for (did, i, sid, k, w, ip, _) in &pairs {
                if *w <= 0.35 {
                    continue;
                }
                if *i != BIG_DOT_STALK {
                    graph.connect_out(*did, *i, *sid, *k as u32);
                    continue;
                }
                // The big dot: one fitting free input takes it, several ask
                // (once — shift stays held through the drag).
                let free: Vec<usize> = big_dot_options(graph, *sid, *k as u32, *did)
                    .into_iter()
                    .filter(|i| graph.input_of(*did, *i).is_none())
                    .collect();
                match free.as_slice() {
                    [] => {}
                    [one] => graph.connect_out(*did, *one, *sid, *k as u32),
                    _ if self.big_dot_menu.is_none() => {
                        self.big_dot_menu = Some(BigDotMenu {
                            dst: *did,
                            src: *sid,
                            out: *k as u32,
                            at: view.to_screen(*ip),
                            options: free,
                            t0: ui.input(|i| i.time),
                        });
                    }
                    _ => {}
                }
            }
        }
        // Channel nodules run on the same spring. Their rest points move
        // when a node moves, when it is resized, and — the reason this
        // exists — when an rgb triple collapses or expands, which changes
        // both the number of nodules and the position of every one of them.
        if self.link_mode {
            const CK: f32 = 260.0; // a touch softer than the stalks: these
            const CC: f32 = 10.0; // travel further, so they get more swing
            let spring = wobble_spring(CK, CC, self.wobble);
            for (id, n) in &graph.nodes {
                let chans = crate::patchbay::node_ports(&n.kind, expand_rgb);
                let (np, sz) = (dp(*id), ds(*id));
                for (i, ch) in chans.iter().enumerate() {
                    for out in [false, true] {
                        let target = chan_rest(np, sz, i, chans.len(), out);
                        let e = self
                            .chan_springs
                            .entry((*id, *ch, out))
                            .or_insert((target, Vec2::ZERO));
                        if let Some((k, c)) = spring {
                            let acc = (target - e.0) * k - e.1 * c;
                            e.1 += acc * dt;
                            e.0 += e.1 * dt;
                        } else {
                            *e = (target, Vec2::ZERO);
                        }
                    }
                }
            }
            // A nodule that stopped existing (its node went, or its lane
            // folded into a port) drops its spring, so re-expanding starts
            // from the port it came out of rather than from a stale spot.
            self.chan_springs.retain(|(id, ch, _), _| {
                graph.nodes.get(id).is_some_and(|n| {
                    crate::patchbay::node_ports(&n.kind, expand_rgb).contains(ch)
                })
            });
        } else if !self.chan_springs.is_empty() {
            self.chan_springs.clear();
        }
        self.stalks.retain(|(id, k), _| {
            graph
                .nodes
                .get(id)
                .is_some_and(|n| {
                    stalk_out(*k).is_some_and(|o| o < n.kind.output_count().max(1))
                        || *k < n.kind.input_count()
                        || (*k == BIG_DOT_STALK && big_dot(&n.kind))
                })
        });
        let tips: std::collections::HashMap<(u64, usize), Pos2> =
            self.stalks.iter().map(|(k, v)| (*k, v.0)).collect();
        // Where each nodule ACTUALLY is this frame — mid-wobble, that is not
        // its rest point. Hits read the same map as the drawing, so you can
        // always grab the bead you can see.
        let chan_tips: std::collections::HashMap<
            (u64, crate::anim::Channel, bool),
            Pos2,
        > = self.chan_springs.iter().map(|(k, v)| (*k, v.0)).collect();
        fn tip(
            tips: &std::collections::HashMap<(u64, usize), Pos2>,
            id: u64,
            k: usize,
            fallback: Pos2,
        ) -> Pos2 {
            tips.get(&(id, k)).copied().unwrap_or(fallback)
        }

        // --- hit tests (screen space) ---
        let sock_hit = CHAN_HIT;
        let zoom_now = self.zoom;
        let ortho = self.ortho_wires;
        // Right angles: every wire's route this frame, with runs that would
        // lie on top of another output's spread into lanes (a subway map).
        // `wire_path` picks them up by their ends, so every draw and every
        // hit test below sees the same laned route.
        set_ortho_lanes(if ortho && !self.link_mode && lanes_enabled() {
            let pm = self.patch_mode;
            let st = crate::ortho::Style::at(zoom_now);
            let mut keys = Vec::new();
            let mut wires = Vec::new();
            for (dst, input, src) in &graph.edges {
                if graph.node(*dst).is_none() || graph.node(*src).is_none() {
                    continue;
                }
                if pm != graph.is_hidden(*dst, *input) || harp_kids.contains(dst) {
                    continue;
                }
                let ok = graph.out_of(*dst, *input) as usize;
                let from = view.to_screen(tip(
                    &tips,
                    *src,
                    out_stalk(ok),
                    output_rest_of(&graph.nodes, *src, ok, dz(*src)),
                ));
                let to = view.to_screen(tip(
                    &tips,
                    *dst,
                    *input,
                    rest_of(graph, &mask_dirs, *dst, *input, dp(*dst), ds(*dst)),
                ));
                let (ea, eb) = ortho_ends(tan(*src), tan(*dst), side_of(graph, *dst, *input));
                keys.push(lane_key(from, to));
                wires.push(crate::ortho::LaneWire {
                    group: (*src << 8) ^ ok as u64,
                    a: from,
                    ea,
                    b: to,
                    eb,
                });
            }
            keys.into_iter().zip(crate::ortho::lane_routes(&wires, &st)).collect()
        } else {
            Default::default()
        });
        // CONTROL nodes belong to the tool's PANEL, not to the image flow:
        // they take no input, produce no pixels, and exist only to run values
        // down patch cables. Outside link mode they are pure clutter, so they
        // are neither drawn nor hit — a node you cannot see must not be a
        // node you can accidentally grab.
        let lm = self.link_mode;
        // Off-stage nodes: control nodes outside link mode, and a harp's
        // hidden ChannelSelect kids always — a node you cannot see must not
        // be a node you can accidentally grab.
        let hidden = |id: &u64, n: &ENode| {
            (!lm && crate::tools::control_of(&n.kind).is_some())
                || harp_kids.contains(id)
        };
        let hits_at =
            |p: Pos2| -> (Option<u64>, Option<(u64, u32)>, Option<(u64, usize)>) {
            let (mut node, mut out, mut inp) = (None, None, None);
            for (id, n) in graph.nodes.iter().filter(|(id, n)| !hidden(id, n)) {
                let (np, sz) = (dp(*id), ds(*id));
                let sp = view.to_screen(np);
                // Boring: a pill is thinner than the sockets' reach, so
                // inside the body the BODY wins; a socket is grabbed from
                // the half of it that stands outside the edge.
                let on_body = boring()
                    && (boring_shape(&n.kind) || tv_on(n))
                    && wide_of(n).is_some_and(|(h, dx)| {
                        let c = sp + Vec2::new(dx * sz * zoom_now, 0.0);
                        Rect::from_center_size(c, h * 2.0 * sz * zoom_now).contains(p)
                    });
                if !on_body {
                    let n_o = n.kind.output_count().clamp(1, MAX_OUTPUTS);
                    for k in 0..n_o {
                        let os =
                            tip(&tips, *id, out_stalk(k), output_rest_of(&graph.nodes, *id, k, dz(*id)));
                        if p.distance(view.to_screen(os)) < sock_hit {
                            out = Some((*id, k as u32));
                        }
                    }
                }
                let mut near_wire = false;
                let dotted = big_dot(&n.kind);
                for i in 0..n.kind.input_count() {
                    if on_body {
                        break;
                    }
                    if dotted && graph.input_of(*id, i).is_none() {
                        continue; // behind the big dot
                    }
                    let is =
                        tip(&tips, *id, i, rest_of(graph, &mask_dirs, *id, i, np, sz));
                    if p.distance(view.to_screen(is)) < sock_hit {
                        inp = Some((*id, i));
                        near_wire = true;
                    }
                }
                // Anywhere on a multi-input bar is its free input — unless
                // the pointer sits on a wire's end, which it picks up.
                if multi_input(&n.kind) && !near_wire {
                    let n_in = n.kind.input_count();
                    let free = n_in.saturating_sub(1);
                    if graph.input_of(*id, free).is_none() {
                        let c = view.to_screen(multi_rest(np, free, n_in, sz));
                        let half = Vec2::new(
                            multi_half(n_in) * sz * zoom_now + 4.0,
                            sock_hit.max(6.0),
                        );
                        if Rect::from_center_size(c, half * 2.0).contains(p) {
                            inp = Some((*id, free));
                        }
                    }
                }
                match wide_of(n) {
                    Some((h, dx)) => {
                        let c = sp + Vec2::new(dx * sz * zoom_now, 0.0);
                        let body =
                            Rect::from_center_size(c, h * 2.0 * sz * zoom_now);
                        if body.expand(4.0).contains(p) {
                            node = Some(*id);
                        }
                    }
                    None => {
                        if p.distance(sp) < (R * sz * zoom_now).max(6.0) {
                            node = Some(*id);
                        }
                    }
                }
            }
            (node, out, inp)
        };
        let (hit_node, hit_out, hit_in) = pointer.map(hits_at).unwrap_or((None, None, None));
        // A Layer's bar under a dragged noodle, anywhere along it: where a
        // new layer would go, by the strings either side (Houdini's Merge,
        // and 2dini's). (node, insertion index, screen x of the gap).
        let reorder = match self.drag {
            Some(DragKind::LayerReorder { node, from, press }) => Some((node, from, press)),
            _ => None,
        };
        let hit_layer_bar: Option<(u64, usize, f32, f32, f32)> = pointer
            .filter(|_| match self.drag {
                Some(DragKind::WireFromOutput(..)) => hit_in.is_none(),
                Some(DragKind::LayerReorder { .. }) => true,
                _ => false,
            })
            .and_then(|p| {
                graph.nodes.iter().filter(|(id, n)| !hidden(id, n)).find_map(|(id, n)| {
                    let genai = match &n.kind {
                        NodeKind::Layer(_) => None,
                        NodeKind::GenAi(p) if p.model.is_none() => Some(p),
                        _ => return None,
                    };
                    if reorder.is_some_and(|r| r.0 != *id) {
                        return None;
                    }
                    let (h, dx) = wide_of(n)?;
                    let (np, sz) = (dp(*id), ds(*id));
                    let c = view.to_screen(np + Vec2::new(dx * sz, 0.0));
                    // The slab and the strings above it.
                    let half = h * sz * zoom_now;
                    let reach = (R + STALK) * sz * zoom_now;
                    let r = Rect::from_min_max(
                        Pos2::new(c.x - half.x - 12.0, c.y - half.y - reach - 10.0),
                        Pos2::new(c.x + half.x + 12.0, c.y + half.y),
                    );
                    if !r.contains(p) {
                        return None;
                    }
                    let xs = crate::breakout::bar_strings_x(&n.kind, np, sz);
                    let sx = |i: usize| xs.get(i).map(|(_, x)| view.to_screen(Pos2::new(*x, np.y)).x);
                    let mut k = xs.iter().filter(|(_, x)| view.to_screen(Pos2::new(*x, np.y)).x < p.x).count();
                    // A GenAI bar grows on the side the noodle is over: a
                    // new picture after the last picture (left half), a
                    // new matte after the last matte (right half). k past
                    // the pictures means the matte side.
                    if let Some(g) = genai {
                        let images = (0..n.kind.input_count())
                            .filter(|&i| g.pipeline.in_kind(i) == genai::InKind::Image)
                            .count();
                        let split = match (images.checked_sub(1).and_then(sx), sx(images)) {
                            (Some(a), Some(b)) => (a + b) * 0.5,
                            (_, _) => c.x,
                        };
                        k = if p.x < split { images } else { xs.len() + 1 };
                        let gx = if k == images {
                            match (images.checked_sub(1).and_then(sx), sx(images)) {
                                (Some(a), Some(b)) => (a + b) * 0.5,
                                (Some(a), None) => a + 20.0 * sz * zoom_now,
                                (None, Some(b)) => b - 20.0 * sz * zoom_now,
                                (None, None) => c.x,
                            }
                        } else {
                            xs.len().checked_sub(1).and_then(sx).map_or(c.x, |a| a + 20.0 * sz * zoom_now)
                        };
                        return Some((*id, k, gx, r.top() + 6.0, c.y + half.y));
                    }
                    // The gap's x: between the strings either side.
                    let gx = match (k.checked_sub(1).and_then(sx), sx(k)) {
                        (Some(a), Some(b)) => (a + b) * 0.5,
                        (Some(a), None) => a + 20.0 * sz * zoom_now,
                        (None, Some(b)) => b - 20.0 * sz * zoom_now,
                        (None, None) => c.x,
                    };
                    Some((*id, k, gx, r.top() + 6.0, c.y + half.y))
                })
            });
        // A picked-up Layer nodule pulled off its bar is the ordinary
        // unplug: its wire comes away and follows the hand (an empty one
        // seeks a source), exactly as a press on any input does.
        let mut unplug_layer: Option<(u64, usize)> = None;
        let hit_layer_bar = match (reorder, pointer) {
            (Some((node, from, press)), Some(p)) if hit_layer_bar.is_none() && p.distance(press) > 6.0 => {
                unplug_layer = Some((node, from));
                None
            }
            // Over its own slot, or the gap either side of it: nowhere to
            // go, so no marker.
            (Some((_, from, _)), _) => hit_layer_bar.filter(|h| h.1 != from && h.1 != from + 1),
            _ => hit_layer_bar,
        };
        self.reorder_to = reorder.map(|(node, from, _)| {
            hit_layer_bar
                .filter(|h| h.0 == node)
                .map_or(from, |h| if h.1 > from { h.1 - 1 } else { h.1 })
        });
        // The big dot under the pointer, if no single nodule is.
        let hit_big: Option<u64> = pointer.filter(|_| hit_in.is_none() && hit_out.is_none()).and_then(|p| {
            graph.nodes.iter().filter(|(id, n)| !hidden(id, n) && big_dot(&n.kind)).find_map(|(id, _)| {
                let c = view.to_screen(big_dot_rest(graph, *id, dp(*id), ds(*id)));
                (p.distance(c) < (BIG_DOT_R * ds(*id) * zoom_now).max(sock_hit)).then_some(*id)
            })
        });
        let hit_port = hit_out
            .map(|(n, o)| Port::Out(n, o))
            .or(hit_in.map(|(n, i)| Port::In(n, i)));
        // What the nodule under the pointer is, in words: a hover tip.
        if resp.hovered() && self.drag.is_none() {
            let tip_text = match hit_port {
                Some(Port::Out(n, o)) => graph.node(n).map(|e| {
                    port_tip(e.kind.output_kind_at(o as usize), None, false)
                }),
                Some(Port::In(n, i)) => graph.node(n).map(|e| {
                    let fed = graph.input_of(n, i).is_some();
                    let what = input_tip(&e.kind, i);
                    let state = if fed { "fed" } else { "empty — drag a noodle in" };
                    format!("{what}\n{state}")
                }),
                None => hit_big.map(|_| {
                    "every input of this node, behind one dot\ndrop a noodle here: it lands \
                     in the one input it fits, or offers the few it could"
                        .to_string()
                }),
            };
            if let Some(t) = tip_text {
                resp.clone().on_hover_text_at_pointer(t);
            }
        }

        // The wire under the pointer and the point ON it nearest the cursor
        // — what a dot gets cut into. Resolved to a VALUE here rather than
        // left as a closure: everything below this line mutates the graph.
        let pm = self.patch_mode;
        self.shown_wire_range = None;
        let wire_under: Option<((u64, usize, u64), Pos2)> = pointer.and_then(|p| {
            let mut best: Option<((u64, usize, u64), Pos2, f32)> = None;
            for (dst, input, src) in &graph.edges {
                if graph.node(*dst).is_none() || graph.node(*src).is_none() {
                    continue;
                }
                // Each mode hits only the wires it draws: the hidden layer
                // in patch mode, the ordinary wiring otherwise — so the
                // slicer, dot cuts and the context menu act on what you see.
                if pm != graph.is_hidden(*dst, *input) {
                    continue;
                }
                if harp_kids.contains(dst) {
                    continue; // a kid's feed is machinery, never a target
                }
                let ok = graph.out_of(*dst, *input) as usize;
                let from = view.to_screen(tip(
                    &tips,
                    *src,
                    out_stalk(ok),
                    output_rest_of(&graph.nodes, *src, ok, dz(*src)),
                ));
                let to = view.to_screen(tip(
                    &tips,
                    *dst,
                    *input,
                    rest_of(graph, &mask_dirs, *dst, *input, dp(*dst), ds(*dst)),
                ));
                let side = side_of(graph, *dst, *input);
                let c = wire_path(from, to, zoom_now, tan(*src), tan(*dst), side, ortho);
                let d = c.min_dist(p);
                if d < 18.0 && best.is_none_or(|(_, _, bd)| d < bd) {
                    // Where along the wire, so a dot lands ON it rather than
                    // beside it.
                    best = Some(((*dst, *input, *src), view.to_graph(c.nearest(p)), d));
                }
            }
            best.map(|(e, at, _)| (e, at))
        });
        // Inside a 2dini, [ and ] move the noodle under the pointer down
        // and up the stack: its z, which the Merge it feeds sorts by.
        // Applied below, once the graph may be written.
        let mut wire_z_step: Option<(u64, usize, f32)> = None;
        if self.in_geo && resp.hovered() && !self.patch_mode {
            if let Some(((dst, input, _), _)) = wire_under {
                let step = ui.input(|i| {
                    if crate::keymap::pressed(i, crate::keymap::Act::GraphWireZUp) {
                        1.0
                    } else if crate::keymap::pressed(i, crate::keymap::Act::GraphWireZDown) {
                        -1.0
                    } else {
                        0.0
                    }
                });
                if step != 0.0 {
                    wire_z_step = Some((dst, input, step));
                }
            }
        }

        // Group handles, from last frame's draw (same stale scheme as the
        // scale handles). Read into values before anything takes `&mut`.
        let group_hits = self.group_hits.clone();
        let group_at = |p: Pos2| -> (Option<u64>, Option<u64>) {
            let mut title = None;
            let mut grip = None;
            for (id, tr, gr) in &group_hits {
                if gr.contains(p) {
                    grip = Some(*id);
                }
                if tr.contains(p) {
                    title = Some(*id);
                }
            }
            (title, grip)
        };

        // Channel nodules (link mode). Returns (node, channel, is_output) —
        // the CHANNEL of the port, which for a collapsed triple is its R lane.
        let link_mode = self.link_mode;
        let chan_hit_at = |p: Pos2| -> Option<(u64, crate::anim::Channel, bool)> {
            if !link_mode {
                return None;
            }
            let mut best: Option<(f32, (u64, crate::anim::Channel, bool))> = None;
            for (id, n) in &graph.nodes {
                let chans = crate::patchbay::node_ports(&n.kind, expand_rgb);
                let (np, sz) = (dp(*id), ds(*id));
                for (i, ch) in chans.iter().enumerate() {
                    for out in [false, true] {
                        let rest = chan_rest(np, sz, i, chans.len(), out);
                        let at = view.to_screen(
                            chan_tips.get(&(*id, *ch, out)).copied().unwrap_or(rest),
                        );
                        let d = p.distance(at);
                        if d < sock_hit && best.is_none_or(|(bd, _)| d < bd) {
                            best = Some((d, (*id, *ch, out)));
                        }
                    }
                }
            }
            best.map(|(_, v)| v)
        };
        let hit_chan = pointer.and_then(chan_hit_at);
        self.hover_chan = hit_chan;

        // Hold `.` and press: on a dot, take it; on a wire (off any node
        // body), cut one in there. The press then belongs to the dot — the
        // drag that follows pulls its tension, not a wire, a move or a
        // marquee. Resolved here, before the drag machine sees the press;
        // the cut itself waits until it has run (it mutates the graph).
        let dot_press = (ui.input(|i| i.pointer.primary_pressed())
            && resp.hovered()
            && !ui.ctx().egui_wants_keyboard_input()
            && !crate::keymap::capturing()
            && ui.input(|i| crate::keymap::down(i, crate::keymap::Act::GraphDot))
            && self.pie.is_none()
            && self.ops_pie.is_none()
            && self.node_pie.is_none()
            && self.region_menu.is_none()
            && self.grab.is_none()
            && !self.link_mode)
            .then(|| match hit_node {
                Some(id) if graph.is_dot(id) => Some(DotPress::Dot(id)),
                // A socket is on its wire: with `.` held the press means the
                // wire, so a port does not stop the cut; a node body does.
                None => wire_under.map(|(edge, at)| DotPress::Wire(edge, at)),
                _ => None,
            })
            .flatten();

        // --- drag state machine ---
        let ctrl = ui.input(|i| i.modifiers.ctrl);
        if resp.drag_started_by(egui::PointerButton::Secondary) && ctrl {
            // macOS reports ctrl-drag as a secondary-button drag.
            self.drag = pointer.map(|p| DragKind::Slice { last: p, path: vec![p] });
        }
        if resp.drag_started_by(egui::PointerButton::Secondary)
            && !ctrl
            && ui.input(|i| i.modifiers.alt)
        {
            // Houdini-style dolly (alt + right-drag): zoom about the press.
            self.drag = ui
                .input(|i| i.pointer.press_origin())
                .or(pointer)
                .map(|p| DragKind::Zoom { anchor: p });
        }
        let space_held = !ui.ctx().egui_wants_keyboard_input()
            && ui.input(|i| crate::keymap::down(i, crate::keymap::Act::PlayPause));
        // A held noodle owns the primary button: it lands on the port you
        // click and ignores everything else, so you can pan and zoom your
        // way to a node that was never on screen without losing the wire.
        if resp.drag_started_by(egui::PointerButton::Primary)
            && !space_held
            && self.grab.is_none()
            // The pies and the M popup own the pointer while open.
            && self.pie.is_none()
            && self.ops_pie.is_none()
            && self.node_pie.is_none()
            && self.region_menu.is_none()
            // A `.` press is pulling a dot.
            && dot_press.is_none()
            && self.dot_place.is_none()
        {
            let origin = ui.input(|i| i.pointer.press_origin()).or(pointer);
            let (o_node, o_out, o_in) = origin.map(hits_at).unwrap_or((None, None, None));
            let (o_title, o_grip) = origin.map(group_at).unwrap_or((None, None));
            // The grip scales members about the blob's far corner; capture
            // everything at press so the gesture stays exact however far it
            // wanders (same scheme as the selection-scale handles).
            let o_scale_group = o_grip.and_then(|gid| {
                let (_, b) = self.blobs.get(&gid)?;
                let (anchor, handle) = (b.bbox.min, b.grip);
                let orig: Vec<(u64, Pos2, f32)> = graph
                    .groups
                    .iter()
                    .find(|g| g.id == gid)?
                    .live_members(graph)
                    .iter()
                    .filter_map(|m| graph.node(*m).map(|n| (*m, n.pos, n.size)))
                    .collect();
                (!orig.is_empty()).then_some((anchor, handle, orig))
            });
            // `A` + drag: the Houdini align swipe. Held, not tapped, so the
            // stroke's own direction is what picks the axis.
            let a_swipe = !ui.ctx().egui_wants_keyboard_input()
                && ui.input(|i| crate::keymap::down(i, crate::keymap::Act::GraphAlign))
                && !selection.is_empty();
            // What S / R + drag operate on: the selected nodes plus
            // every member of a selected backdrop — selecting a group IS
            // selecting its nodes, as far as the transforms care.
            let handle_ids: Vec<u64> = {
                let mut v = selection.clone();
                for gid in &self.sel_groups {
                    if let Some(g) = graph.groups.iter().find(|g| g.id == *gid) {
                        for m in &g.members {
                            if !v.contains(m) {
                                v.push(*m);
                            }
                        }
                    }
                }
                // A harp is one instrument: either bar brings the other
                // along, so they scale together.
                v.extend(crate::breakout::companions(graph, &v));
                v
            };
            // `S` / `R` held + drag: scale / rotate the selection's layout
            // about its centre (Blender's keys). Held, like `A`; a TAP of
            // either is still its old action, fired on release.
            let held = |a: crate::keymap::Act| {
                !ui.ctx().egui_wants_keyboard_input()
                    && ui.input(|i| crate::keymap::down(i, a))
                    && handle_ids.len() >= 2
            };
            let s_drag = held(crate::keymap::Act::GraphSwapInputs);
            // A harp does not rotate: its bars are horizontal by definition.
            let r_drag = !s_drag
                && held(crate::keymap::Act::GraphRotate)
                && !handle_ids.iter().any(|id| {
                    graph
                        .node(*id)
                        .and_then(|n| crate::breakout::wide_half(&n.kind))
                        .is_some()
                });
            let xf_center = {
                let mut bb = Rect::NOTHING;
                for id in &handle_ids {
                    if let Some(n) = graph.node(*id) {
                        bb.extend_with(n.pos);
                    }
                }
                bb.center()
            };
            let o_chan = origin.and_then(chan_hit_at);
            // Blender's emulated-MMB chords (`nav::drag_nav`) outrank every
            // canvas gesture, exactly as real MMB does — the tablet hand
            // has no other way to nav from on top of a node or socket
            // (manifesto 11).
            let nav = crate::nav::drag_nav(
                egui::PointerButton::Primary,
                ui.input(|i| i.modifiers),
            );
            // A press inside a zoomed-in node's inner graph belongs to it
            // (after the nav chords, before everything else): the host is
            // not dragged, its node or wire is.
            let o_innard = origin.and_then(|p| {
                self.innard_hits
                    .iter()
                    .find(|(_, _, h, live)| *live && h.iter().any(|(_, r)| r.contains(p)))
                    .map(|(host, frame, h, _)| (*host, *frame, h.clone(), p))
            });
            let o_ctl = origin.and_then(|p| {
                let tick = self.eval_tick;
                self.ctl_hits.iter().filter(|(.., live)| *live).find_map(|(host, place, h, _)| {
                    let k = graph.node(*host)?.kind_at(tick);
                    crate::nodectl::begin(*host, &k, *place, h, p)
                })
            });
            self.drag = if nav == Some(crate::nav::DragNav::Zoom) {
                origin.map(|p| DragKind::Zoom { anchor: p })
            } else if nav == Some(crate::nav::DragNav::Pan) {
                Some(DragKind::Pan)
            } else if let Some(d) = o_ctl {
                self.ctl_drag = Some(d);
                Some(DragKind::Ctl)
            } else if let Some(d) = o_innard.filter(|_| !ctrl).and_then(|(host, frame, h, p)| {
                // ctrl is the slice, inside a node as outside it.
                crate::innards::begin(ui.ctx(), graph, host, frame, &h, p)
            }) {
                self.innard_drag = Some(d);
                Some(DragKind::Innard)
            } else if ctrl {
                origin.map(|p| DragKind::Slice { last: p, path: vec![p] })
            } else if let (true, Some(p)) = (s_drag, origin) {
                self.s_armed = false;
                let center = xf_center;
                let orig = handle_ids
                    .iter()
                    .filter_map(|id| graph.node(*id).map(|n| (*id, n.pos)))
                    .collect();
                let spacing = handle_ids
                    .iter()
                    .filter_map(|id| match graph.node(*id).map(|n| &n.kind) {
                        Some(NodeKind::Breakout(p)) => Some((*id, p.spacing)),
                        _ => None,
                    })
                    .collect();
                // A press right on the centre would make the first pixel of
                // travel a huge factor: measure from at least 30 px out.
                let start_dist = (view.to_graph(p) - center).length().max(30.0 / self.zoom);
                Some(DragKind::SelScale { center, start_dist, orig, spacing })
            } else if let (true, Some(p)) = (r_drag, origin) {
                let center = xf_center;
                let orig = handle_ids
                    .iter()
                    .filter_map(|id| graph.node(*id).map(|n| (*id, n.pos)))
                    .collect();
                Some(DragKind::SelRotate {
                    center,
                    start_angle: (view.to_graph(p) - center).angle(),
                    orig,
                })
            } else if a_swipe {
                origin.map(|p| DragKind::AlignSwipe { start: p, path: vec![p] })
            } else if let Some((anchor, handle, orig)) = o_scale_group {
                Some(DragKind::GroupScale { anchor, handle, orig })
            } else if let Some((id, ch, out)) = o_chan {
                // Grabbing a DRIVEN input takes its cable off and re-drags
                // from the far end, exactly as a wired image input does.
                if !out {
                    let dst = PDst::Chan(id, ch);
                    // Unplugging a collapsed port unplugs the whole triple.
                    let lanes = graph
                        .node(id)
                        .and_then(|n| crate::patchbay::lanes_for(&n.kind, ch, expand_rgb));
                    let all: Vec<PDst> = match lanes {
                        Some(l) => l.iter().map(|c| PDst::Chan(id, *c)).collect(),
                        None => vec![dst],
                    };
                    let held = all.iter().find_map(|d| graph.patch.link_to(*d).map(|l| l.from));
                    match held {
                        Some(PSrc::Chan(sn, sc)) => {
                            for d in &all {
                                graph.patch.unlink(*d);
                            }
                            Some(DragKind::CableFrom(PSrc::Chan(sn, sc)))
                        }
                        // The other end is a value node or a tool's own
                        // parameter, both of which live on the patch canvas:
                        // unplug rather than drag a cable this view cannot
                        // draw.
                        Some(PSrc::Mod(_) | PSrc::Param(_)) => {
                            for d in &all {
                                graph.patch.unlink(*d);
                            }
                            None
                        }
                        None => Some(DragKind::CableTo(dst)),
                    }
                } else {
                    Some(DragKind::CableFrom(PSrc::Chan(id, ch)))
                }
            } else if let Some((id, o)) =
                // A dot's sockets sit ON the bead: left to the sockets,
                // every press would start a wire and the dot could never be
                // MOVED. On a dot the body wins the drag; wires still drop
                // onto it, and a click still selects its port for G.
                o_out.filter(|(sid, _)| !(o_node == Some(*sid) && graph.is_dot(*sid)))
            {
                Some(DragKind::WireFromOutput(id, o))
            } else if let Some((id, i, press)) = o_in
                .filter(|(nid, _)| !self.patch_mode && matches!(graph.node(*nid).map(|n| &n.kind), Some(NodeKind::Layer(_))))
                .zip(origin)
                .map(|((id, i), p)| (id, i, p))
            {
                Some(DragKind::LayerReorder { node: id, from: i, press })
            } else if let Some((id, i)) =
                o_in.filter(|(nid, _)| !(o_node == Some(*nid) && graph.is_dot(*nid)))
            {
                // Grabbing a connected input detaches its wire and re-drags
                // from the old source; a free input starts a seeking wire.
                // Patch mode edits only the hidden layer: a VISIBLE feed
                // stays put (locked), and the drag seeks a hidden wire that
                // will steal the port on drop. In normal mode detaching a
                // HIDDEN feed surfaces it — the flag clears with the edge,
                // so what lands is an ordinary noodle you can see.
                if let Some(src) = graph
                    .input_of(id, i)
                    .filter(|_| !self.patch_mode || graph.is_hidden(id, i))
                {
                    let o = graph.out_of(id, i);
                    graph.disconnect_input(id, i);
                    Some(DragKind::WireFromOutput(src, o))
                } else {
                    Some(DragKind::WireFromInput(id, i))
                }
            } else if let Some(id) = o_title {
                // What is inside NOW is what travels: a group cannot shed
                // its nodes by being dragged out from under them.
                let members = graph
                    .groups
                    .iter()
                    .find(|g| g.id == id)
                    .map(|g| g.live_members(graph))
                    .unwrap_or_default();
                if !selection.contains(&id) {
                    if !ui.input(|i| i.modifiers.shift) {
                        self.sel_groups.clear();
                    }
                    if !self.sel_groups.contains(&id) {
                        self.sel_groups.push(id);
                    }
                }
                Some(DragKind::GroupMove { members })
            } else if overlay {
                // The overlay has no marquee, so a bare background drag
                // pans. (alt-drag pan moved to the TOP of this chain —
                // Blender's emulated MMB beats sockets and handles too.)
                Some(DragKind::Pan)
            } else if let Some(id) = o_node {
                let shift = ui.input(|i| i.modifiers.shift);
                if !selection.contains(&id) {
                    if shift {
                        selection.push(id);
                    } else {
                        *selection = vec![id];
                    }
                }
                // Shift-drag pulls the dragged set OUT of its backdrop, and
                // the drop at the end will not put it back in one.
                if shift {
                    let set: Vec<u64> =
                        if selection.contains(&id) { selection.clone() } else { vec![id] };
                    for g in &mut graph.groups {
                        g.members.retain(|m| !set.contains(m));
                    }
                    graph.prune_groups();
                }
                let mut set: Vec<u64> =
                    if selection.contains(&id) { selection.clone() } else { vec![id] };
                // A harp's top bar carries the nodes spliced into its
                // strings: they ride in `from` (moved with the set, not
                // selected), so the strings keep hanging straight.
                set.extend(crate::breakout::riders(graph, &set));
                let from = set
                    .iter()
                    .filter_map(|n| graph.node(*n).map(|nd| (*n, nd.pos)))
                    .collect();
                Some(DragKind::Node { id, unjoin: shift, from, merged: false, raw: Vec2::ZERO })
            } else {
                // Background drag = marquee selection (pan lives on middle
                // drag / alt-drag; double-click still resets the view).
                origin.map(|p| DragKind::BoxSelect { start: p })
            };
        }
        if resp.dragged_by(egui::PointerButton::Middle) {
            if ui.input(|i| i.modifiers.ctrl || i.modifiers.command) {
                // ctrl+MMB: Blender's drag-zoom, anchored where the drag
                // began (the same maths as the alt+right dolly).
                let factor = crate::nav::drag_zoom(resp.drag_delta());
                if let Some(a) = ui.input(|i| i.pointer.press_origin()) {
                    let g = view.to_graph(a);
                    self.zoom = (self.zoom * factor).clamp(0.002, 200.0);
                    self.pan = (a - vrect.center()) - g.to_vec2() * self.zoom;
                }
            } else {
                self.pan += resp.drag_delta();
            }
        }
        let zoom = self.zoom;
        // A drag the match below replaces (the M-swipe hands it on).
        let mut next_drag: Option<DragKind> = None;
        match &mut self.drag {
            Some(DragKind::Ctl) if resp.dragged() => {
                if let (Some(d), Some(p)) = (self.ctl_drag.as_mut(), pointer) {
                    if let Some(n) = graph.node(d.host) {
                        let fine = ui.input(|i| i.modifiers.shift);
                        let host = d.host;
                        for (ch, v) in crate::nodectl::update(d, &n.kind, p, fine) {
                            self.ctl_edits.push((host, ch, v));
                        }
                    }
                }
            }
            Some(DragKind::Innard) if resp.dragged() => {
                if let (Some(d), Some(p)) = (self.innard_drag, pointer) {
                    if let Some((_, frame, _, _)) = self.innard_hits.iter().find(|(h, ..)| *h == d.host()) {
                        let frame = *frame;
                        crate::innards::update(graph, &d, frame, p);
                        // The graph's M / I / O swipe, inside the node.
                        let armed = if ui.ctx().egui_wants_keyboard_input() {
                            None
                        } else {
                            use crate::keymap::{down, Act};
                            ui.input(|i| {
                                if down(i, Act::GraphRegionMenu) || down(i, Act::GraphSwipeBlend) {
                                    Some(SwipeOp::Blend)
                                } else if down(i, Act::GraphSwipeIn) {
                                    Some(SwipeOp::In)
                                } else if down(i, Act::GraphSwipeOut) {
                                    Some(SwipeOp::Out)
                                } else {
                                    None
                                }
                            })
                        };
                        if let Some(op) = armed {
                            if let Some(nd) = crate::innards::swipe(graph, &d, frame, p, op) {
                                self.innard_drag = Some(nd);
                                // M's release is not a region-popup tap.
                                if op == SwipeOp::Blend
                                    && ui.input(|i| {
                                        crate::keymap::down(i, crate::keymap::Act::GraphRegionMenu)
                                    })
                                {
                                    self.m_used = true;
                                }
                            }
                        }
                    }
                }
            }
            Some(DragKind::Node { id, from, merged, raw, .. }) if resp.dragged() => {
                let idv = *id;
                let d = resp.drag_delta() / zoom;
                *raw += d;
                // Drag every selected node together.
                let ids: Vec<u64> =
                    if selection.contains(&idv) { selection.clone() } else { vec![idv] };
                let solo = ids.len() == 1;
                // Plus the harp riders captured at the press.
                let mut ids = ids;
                for (f, _) in from.iter() {
                    if !ids.contains(f) {
                        ids.push(*f);
                    }
                }
                for nid in &ids {
                    let start = from.iter().find(|(f, _)| f == nid).map(|(_, p)| *p);
                    if let Some(n) = graph.node_mut(*nid) {
                        n.pos = match start {
                            Some(p0) => p0 + *raw,
                            None => n.pos + d,
                        };
                    }
                }
                // Snap: centres line up with nearby nodes, wires to the
                // rest of the graph come out straight. Ctrl drags free
                // (Blender's snap toggle).
                self.snap_guides.clear();
                if !ui.input(|i| i.modifiers.ctrl || i.modifiers.command) {
                    let area = Rect::from_two_pos(view.to_graph(vrect.min), view.to_graph(vrect.max));
                    let (off, guides) =
                        drag_snap(graph, &mask_dirs, &ids, area, DRAG_SNAP_PX / zoom);
                    if off != Vec2::ZERO {
                        for nid in &ids {
                            if let Some(n) = graph.node_mut(*nid) {
                                n.pos += off;
                            }
                        }
                    }
                    self.snap_guides = guides;
                }
                // Hold M and swipe the node over ANOTHER node: a Blend of
                // the two (dragged OVER, swiped-onto UNDER, the Blend taking
                // the under node's place downstream). The dragged set glides
                // back to where it started, and the drag carries on with the
                // Blend, so letting go is what places it.
                // M, I and O hold the same swipe; the first held wins. I
                // and O need a MATTE in hand, and an image to act on.
                let armed = if *merged || ui.ctx().egui_wants_keyboard_input() {
                    None
                } else {
                    use crate::keymap::{down, Act};
                    ui.input(|i| {
                        if down(i, Act::GraphRegionMenu) || down(i, Act::GraphSwipeBlend) {
                            Some(SwipeOp::Blend)
                        } else if down(i, Act::GraphSwipeIn) {
                            Some(SwipeOp::In)
                        } else if down(i, Act::GraphSwipeOut) {
                            Some(SwipeOp::Out)
                        } else {
                            None
                        }
                    })
                };
                let holds_matte = graph
                    .node(idv)
                    .is_some_and(|n| n.kind.output_kind() == PortKind::Mask);
                let armed = armed.filter(|op| !op.wants_matte() || holds_matte);
                let under = pointer.filter(|_| armed.is_some()).and_then(|p| {
                    graph
                        .nodes
                        .iter()
                        .filter(|(nid, n)| {
                            !ids.contains(nid)
                                && !matches!(n.kind, NodeKind::Dot { .. })
                                && (!armed.is_some_and(SwipeOp::wants_matte)
                                    || n.kind.output_kind() == PortKind::Image)
                        })
                        .find(|(_, n)| {
                            let c = view.to_screen(n.pos);
                            match wide_of(n) {
                                Some((h, dx)) => Rect::from_center_size(
                                    c + Vec2::new(dx * n.size * zoom, 0.0),
                                    h * 2.0 * n.size * zoom,
                                )
                                .contains(p),
                                None => p.distance(c) < (R * n.size * zoom).max(6.0),
                            }
                        })
                        .map(|(nid, _)| *nid)
                });
                if let (Some(under), Some(p), Some(op)) = (under, pointer, armed) {
                    let at = view.to_graph(p);
                    for (nid, p0) in from.iter() {
                        if let Some(n) = graph.node_mut(*nid) {
                            let was = n.pos;
                            n.pos = *p0;
                            // Glide back, don't teleport.
                            self.anim.insert(*nid, (was - *p0, 0.0));
                        }
                    }
                    let b = swipe_op(graph, op, idv, under, at, zoom);
                    *selection = vec![b];
                    if op == SwipeOp::Blend
                        && ui.input(|i| crate::keymap::down(i, crate::keymap::Act::GraphRegionMenu))
                    {
                        self.m_used = true;
                    }
                    self.insert_target = None;
                    next_drag = Some(DragKind::Node {
                        id: b,
                        unjoin: false,
                        from: vec![(b, at)],
                        merged: true,
                        raw: Vec2::ZERO,
                    });
                }
                // Insert-on-noodle: a lone dragged node with a free first
                // input and an unused output hovering a wire will splice in.
                self.insert_target = None;
                let eligible = solo
                    && !self.link_mode
                    && graph.node(idv).is_some_and(|n| n.kind.input_count() >= 1)
                    && graph.input_of(idv, 0).is_none()
                    && !graph.edges.iter().any(|(_, _, s)| *s == idv);
                if eligible {
                    // Live position (the node just moved this frame; the
                    // drawn-pos snapshot is one frame stale for it).
                    let live = graph.node(idv).map(|n| {
                        n.pos + self.anim.get(&idv).map(|(v, _)| *v).unwrap_or(Vec2::ZERO)
                    });
                    if let Some(center) = live.map(|p| view.to_screen(p)) {
                        let mut best: Option<((u64, usize, u64), f32)> = None;
                        for (dst, input, src) in &graph.edges {
                            if *dst == idv || *src == idv {
                                continue;
                            }
                            let (Some(_nd), Some(_ns)) = (graph.node(*dst), graph.node(*src))
                            else {
                                continue;
                            };
                            let ok = graph.out_of(*dst, *input) as usize;
                            let from = view.to_screen(tip(
                                &tips,
                                *src,
                                out_stalk(ok),
                                output_rest_of(&graph.nodes, *src, ok, dz(*src)),
                            ));
                            let to = view.to_screen(tip(
                                &tips,
                                *dst,
                                *input,
                                rest_of(graph, &mask_dirs, *dst, *input, dp(*dst), ds(*dst)),
                            ));
                            let side = side_of(graph, *dst, *input);
                            let d = wire_path(from, to, zoom, tan(*src), tan(*dst), side, ortho)
                                .min_dist(center);
                            if d < 16.0 && best.is_none_or(|(_, bd)| d < bd) {
                                best = Some(((*dst, *input, *src), d));
                            }
                        }
                        self.insert_target = best.map(|(e, _)| e);
                    }
                }
            }
            Some(DragKind::Pan) if resp.dragged() => {
                self.pan += resp.drag_delta();
            }
            Some(DragKind::AlignSwipe { path, .. }) if resp.dragged() => {
                if let Some(p) = pointer {
                    if path.last().is_none_or(|q| (p - *q).length() > 1.5) {
                        path.push(p);
                    }
                }
            }
            Some(DragKind::GroupMove { members, .. }) if resp.dragged() => {
                // The blob is DERIVED from the members, so moving them is
                // moving the group; there is no rectangle to keep in step.
                let d = resp.drag_delta() / zoom;
                for m in members.clone() {
                    if let Some(n) = graph.node_mut(m) {
                        n.pos += d;
                    }
                }
            }
            Some(DragKind::GroupScale { anchor, handle, orig }) if resp.dragged() => {
                if let Some(p) = pointer {
                    let g = view.to_graph(p);
                    let denom = (*handle - *anchor).length().max(1.0);
                    let f = ((g - *anchor).length() / denom).clamp(0.05, 20.0);
                    for (id, opos, osize) in orig.iter() {
                        if let Some(n) = graph.node_mut(*id) {
                            n.pos = *anchor + (*opos - *anchor) * f;
                            n.size = (osize * f).clamp(0.02, 25.0);
                        }
                    }
                }
            }
            Some(DragKind::Zoom { anchor }) if resp.dragged() => {
                let a = *anchor;
                let d = resp.drag_delta();
                let factor = ((d.x - d.y) * 0.005).exp();
                let g = view.to_graph(a);
                self.zoom = (self.zoom * factor).clamp(0.002, 200.0);
                self.pan = (a - vrect.center()) - g.to_vec2() * self.zoom;
            }
            Some(DragKind::SelScale { center, start_dist, orig, spacing })
                if resp.dragged() =>
            {
                if let Some(p) = pointer {
                    let f = ((view.to_graph(p) - *center).length() / *start_dist)
                        .clamp(0.05, 20.0);
                    for (id, opos) in orig.iter() {
                        if let Some(n) = graph.node_mut(*id) {
                            n.pos = *center + (*opos - *center) * f;
                        }
                    }
                    for (id, osp) in spacing.iter() {
                        if let Some(NodeKind::Breakout(bp)) =
                            graph.node_mut(*id).map(|n| &mut n.kind)
                        {
                            bp.spacing = (osp * f).clamp(20.0, 600.0);
                        }
                    }
                }
            }
            Some(DragKind::SelRotate { center, start_angle, orig }) if resp.dragged() => {
                if let Some(p) = pointer {
                    let g = view.to_graph(p);
                    let mut da = (g - *center).angle() - *start_angle;
                    // Shift snaps to 15° steps.
                    if ui.input(|i| i.modifiers.shift) {
                        let step = std::f32::consts::PI / 12.0;
                        da = (da / step).round() * step;
                    }
                    let (sn, cs) = da.sin_cos();
                    for (id, opos) in orig.iter() {
                        if let Some(n) = graph.node_mut(*id) {
                            let d = *opos - *center;
                            n.pos = *center
                                + Vec2::new(d.x * cs - d.y * sn, d.x * sn + d.y * cs);
                        }
                    }
                }
            }
            Some(DragKind::Slice { last, path }) if resp.dragged() => {
                if let Some(p) = pointer {
                    let a = *last;
                    if (p - a).length() > 1.0 && self.link_mode {
                        // Same gesture, the other kind of connection.
                        let cut: Vec<PDst> = graph
                            .patch
                            .links
                            .iter()
                            .filter_map(|l| {
                                let (PSrc::Chan(sn, sc), PDst::Chan(dn, dc)) =
                                    (l.from, l.to)
                                else {
                                    return None;
                                };
                                let ends = |id: u64, ch: crate::anim::Channel, out: bool| {
                                    let n = graph.nodes.get(&id)?;
                                    let chans =
                                        crate::patchbay::node_ports(&n.kind, expand_rgb);
                                    let ch =
                                        crate::patchbay::port_channel(&n.kind, ch, expand_rgb);
                                    let i = chans.iter().position(|c| *c == ch)?;
                                    let rest =
                                        chan_rest(dp(id), ds(id), i, chans.len(), out);
                                    Some(view.to_screen(
                                        chan_tips
                                            .get(&(id, ch, out))
                                            .copied()
                                            .unwrap_or(rest),
                                    ))
                                };
                                let (from, to) =
                                    (ends(sn, sc, true)?, ends(dn, dc, false)?);
                                cable_path(from, to, zoom, 1.0, ortho)
                                    .hits_segment(a, p)
                                    .then_some(l.to)
                            })
                            .collect();
                        for dst in cut {
                            graph.patch.unlink(dst);
                        }
                        *last = p;
                        path.push(p);
                    } else if (p - a).length() > 1.0 {
                        let pm = self.patch_mode;
                        let EditorGraph { nodes, edges, edge_out, hidden_wires, .. } =
                            graph;
                        edges.retain(|(dst, input, src)| {
                            // The slicer cuts the layer the mode shows:
                            // hidden wires in patch mode, ordinary ones
                            // otherwise — the other layer is locked. A harp
                            // kid's feed is machinery in every mode.
                            if hidden_wires.contains(&(*dst, *input)) != pm
                                || harp_kids.contains(dst)
                            {
                                return true;
                            }
                            let (Some(nd), Some(ns)) = (nodes.get(dst), nodes.get(src)) else {
                                return true;
                            };
                            let ok =
                                edge_out.get(&(*dst, *input)).copied().unwrap_or(0) as usize;
                            let from = view.to_screen(tip(
                                &tips,
                                *src,
                                out_stalk(ok),
                                output_rest_of(
                                    nodes,
                                    *src,
                                    ok,
                                    if matches!(ns.kind, NodeKind::Dot { .. }) {
                                        0.0
                                    } else {
                                        ns.size
                                    },
                                ),
                            ));
                            let mut side = None;
                            let to = view.to_screen(tip(
                                &tips,
                                *dst,
                                *input,
                                if matches!(nd.kind, NodeKind::Dot { .. }) {
                                    nd.pos
                                } else if let Some(d) = side_dir(
                                    &nd.kind,
                                    *input,
                                    mask_dirs.get(dst).copied().unwrap_or(Vec2::new(-1.0, 0.0)),
                                ) {
                                    side = side_entry(d);
                                    nd.pos + if boring() { boring_edge(d) } else { d * socket_reach() } * nd.size
                                } else {
                                    input_rest(
                                        nd.pos,
                                        nd.kind.image_slot(*input),
                                        nd.kind.image_input_count(),
                                        nd.size,
                                    )
                                },
                            ));
                            !wire_path(from, to, zoom, tan(*src), tan(*dst), side, ortho)
                                .hits_segment(a, p)
                        });
                        // Flags follow their edges: a sliced hidden wire
                        // takes its flag with it.
                        hidden_wires
                            .retain(|(d, i)| edges.iter().any(|(dd, ii, _)| dd == d && ii == i));
                        // The same stroke cuts the wires of any zoomed-in
                        // node's inner graph it crosses.
                        crate::innards::slice(graph, &self.innard_hits, a, p, ortho);
                        *last = p;
                        path.push(p);
                    }
                }
            }
            _ => {}
        }
        if let Some(d) = next_drag {
            self.drag = Some(d);
        }
        if let Some((node, from)) = unplug_layer {
            self.drag = Some(match graph.input_of(node, from) {
                Some(src) => {
                    let o = graph.out_of(node, from);
                    graph.disconnect_input(node, from);
                    DragKind::WireFromOutput(src, o)
                }
                None => DragKind::WireFromInput(node, from),
            });
        }
        if resp.drag_stopped() {
            match self.drag.take() {
                Some(DragKind::Ctl) => {
                    self.ctl_drag = None;
                }
                Some(DragKind::Innard) => {
                    if let (Some(d), Some(p)) = (self.innard_drag.take(), pointer) {
                        if let Some((_, _, h, _)) = self.innard_hits.iter().find(|(h, ..)| *h == d.host()) {
                            crate::innards::end(graph, &d, h, p);
                        }
                    }
                }
                // `link_lanes` fans a collapsed triple out into its lanes:
                // rgb to rgb wires lane to lane, a single value broadcasts.
                Some(DragKind::CableFrom(from)) => {
                    if let Some((id, ch, false)) = hit_chan {
                        let sl = match from {
                            PSrc::Chan(sn, sc) => graph
                                .node(sn)
                                .and_then(|n| {
                                    crate::patchbay::lanes_for(&n.kind, sc, expand_rgb)
                                })
                                .map(|l| (sn, l)),
                            _ => None,
                        };
                        let dl = graph
                            .node(id)
                            .and_then(|n| {
                                crate::patchbay::lanes_for(&n.kind, ch, expand_rgb)
                            })
                            .map(|l| (id, l));
                        crate::patchbay::link_lanes(
                            &mut graph.patch,
                            sl,
                            dl,
                            from,
                            PDst::Chan(id, ch),
                        );
                    }
                }
                Some(DragKind::CableTo(to)) => {
                    if let Some((id, ch, true)) = hit_chan {
                        let sl = graph
                            .node(id)
                            .and_then(|n| {
                                crate::patchbay::lanes_for(&n.kind, ch, expand_rgb)
                            })
                            .map(|l| (id, l));
                        let dl = match to {
                            PDst::Chan(dn, dc) => graph
                                .node(dn)
                                .and_then(|n| {
                                    crate::patchbay::lanes_for(&n.kind, dc, expand_rgb)
                                })
                                .map(|l| (dn, l)),
                            _ => None,
                        };
                        crate::patchbay::link_lanes(
                            &mut graph.patch,
                            sl,
                            dl,
                            PSrc::Chan(id, ch),
                            to,
                        );
                    }
                }
                Some(DragKind::WireFromOutput(src, out)) if hit_in.is_none() && hit_layer_bar.is_some() => {
                    // Dropped on a Layer's bar: an empty layer takes it (the
                    // one nearest the drop), else a new layer at the gap it
                    // was dropped in.
                    let (node, k, ..) = hit_layer_bar.unwrap_or_default();
                    let n = graph.node(node).map_or(0, |n| n.kind.input_count());
                    // A GenAI bar: an empty input on that side takes it,
                    // else a new one of that kind (`hit_layer_bar` says
                    // which side in k).
                    let genai_ins = match graph.node(node).map(|n| &n.kind) {
                        Some(NodeKind::GenAi(p)) => Some(p.pipeline.ins.clone()),
                        _ => None,
                    };
                    if let Some(ins) = genai_ins {
                        let images = ins.iter().filter(|i| **i == genai::InKind::Image).count();
                        let kind = if k <= images { genai::InKind::Image } else { genai::InKind::Matte };
                        let at = (0..n)
                            .find(|&i| ins.get(i) == Some(&kind) && graph.input_of(node, i).is_none())
                            .or_else(|| match graph.node_mut(node).map(|n| &mut n.kind) {
                                Some(NodeKind::GenAi(p)) => Some(p.pipeline.add_input(kind)),
                                _ => None,
                            });
                        if let Some(at) = at {
                            connect_or_bridge(graph, src, out, node, at, zoom);
                        }
                    } else {
                    let empty = (0..n)
                        .filter(|i| graph.input_of(node, *i).is_none())
                        .min_by_key(|i| (*i as i64 * 2 + 1 - k as i64 * 2).abs());
                    let at = match empty {
                        Some(i) => i,
                        None => {
                            crate::layerui::restack_at(graph, node, crate::layerui::Restack::Insert(k));
                            k
                        }
                    };
                    connect_or_bridge(graph, src, out, node, at, zoom);
                    }
                }
                Some(DragKind::WireFromOutput(src, out)) if hit_in.is_none() && hit_big.is_some() => {
                    // Dropped on a big dot: the one input it fits goes at
                    // once; several, and a short list asks which.
                    let dst = hit_big.unwrap_or_default();
                    let opts = big_dot_options(graph, src, out, dst);
                    match opts.as_slice() {
                        [] => {}
                        [one] => {
                            connect_or_bridge(graph, src, out, dst, *one, zoom);
                        }
                        _ => {
                            self.big_dot_menu = Some(BigDotMenu {
                                dst,
                                src,
                                out,
                                at: pointer.unwrap_or_default(),
                                options: opts,
                                t0: ui.input(|i| i.time),
                            })
                        }
                    }
                }
                Some(DragKind::WireFromOutput(src, out)) => {
                    if let Some((dst, input)) = hit_in {
                        // In patch mode the wire lands HIDDEN. Only when the
                        // feed actually changed: a refused connect must not
                        // flag whatever was already there.
                        let before = graph.input_of(dst, input);
                        connect_or_bridge(graph, src, out, dst, input, zoom);
                        if self.patch_mode && graph.input_of(dst, input) != before {
                            graph.hidden_wires.insert((dst, input));
                        }
                    }
                }
                Some(DragKind::LayerReorder { node, from, .. }) => {
                    // Dropped in a gap of its own bar: the layer moves
                    // there, one neighbour at a time, so its wire, keys and
                    // settings travel and everything between shifts by one.
                    if let Some((_, k, ..)) = hit_layer_bar.filter(|h| h.0 == node) {
                        let to = if k > from { k - 1 } else { k };
                        // Each nodule glides from where it was to its new
                        // slot; the moved one from where it was let go.
                        let old = |j: usize| self.stalks.get(&(node, j)).map(|(p, _)| *p);
                        let came_from = |m: usize| -> usize {
                            if m == to {
                                from
                            } else if from < to && m >= from && m < to {
                                m + 1
                            } else if from > to && m > to && m <= from {
                                m - 1
                            } else {
                                m
                            }
                        };
                        let starts: Vec<(usize, Pos2)> = (from.min(to)..=from.max(to))
                            .filter_map(|m| {
                                let p = if m == to { pointer.map(|p| view.to_graph(p)) } else { old(came_from(m)) };
                                p.map(|p| (m, p))
                            })
                            .collect();
                        for (m, p) in starts {
                            self.glides.insert((node, m), (p, 0.0));
                        }
                        let mut at = from;
                        while at < to {
                            crate::layerui::swap(graph, node, at, at + 1);
                            at += 1;
                        }
                        while at > to {
                            crate::layerui::swap(graph, node, at, at - 1);
                            at -= 1;
                        }
                    }
                }
                Some(DragKind::WireFromInput(dst, input)) => {
                    if let Some((src, out)) = hit_out {
                        let before = graph.input_of(dst, input);
                        connect_or_bridge(graph, src, out, dst, input, zoom);
                        if self.patch_mode && graph.input_of(dst, input) != before {
                            graph.hidden_wires.insert((dst, input));
                        }
                    }
                }
                Some(DragKind::Node { id, unjoin, .. }) => {
                    if let Some((dst, input, src)) = self.insert_target.take() {
                        // The spliced wire keeps its SOURCE OUTPUT: a node
                        // dropped on a harp string (or a band's matte)
                        // grades THAT string, not output 0.
                        let out = graph.out_of(dst, input);
                        graph.connect_out(id, 0, src, out);
                        graph.connect(dst, input, id);
                    } else if let (false, Some(np)) = (unjoin, graph.node(id).map(|n| n.pos)) {
                        // Dropping a node inside a backdrop adopts it —
                        // Blender's frame gesture. The dragged SELECTION
                        // travels together, and "inside" is the blob's own
                        // signed distance, so the concave parts of the
                        // shape don't grab what was dropped beside them.
                        let target = graph
                            .groups
                            .iter()
                            .find(|g| {
                                !g.members.contains(&id) && {
                                    let parts = crate::nodeux::blob_parts(
                                        graph,
                                        &g.live_members(graph),
                                        g.pad,
                                    );
                                    crate::nodeux::blob_sdf(&parts, np) < 0.0
                                }
                            })
                            .map(|g| g.id);
                        if let Some(gid) = target {
                            let set: Vec<u64> = if selection.contains(&id) {
                                selection.clone()
                            } else {
                                vec![id]
                            };
                            graph.adopt_into_group(gid, &set);
                        }
                    }
                }
                Some(DragKind::AlignSwipe { start, path }) => {
                    let end = path.last().copied().or(pointer).unwrap_or(start);
                    if (end - start).length() > 8.0 {
                        crate::nodeux::align_to_swipe(
                            graph,
                            &selection.clone(),
                            view.to_graph(start),
                            view.to_graph(end),
                        );
                    }
                }
                Some(DragKind::BoxSelect { start }) => {
                    if let Some(p) = pointer {
                        let r = Rect::from_two_pos(start, p);
                        // A sub-4px box is just a click; the click path below
                        // handles select/deselect for those.
                        if r.width() > 4.0 || r.height() > 4.0 {
                            // A marquee is a fresh pick; the port goes with
                            // the old one (see the click handler).
                            self.sel_port = None;
                            if !ui.input(|i| i.modifiers.shift) {
                                selection.clear();
                            }
                            for id in graph.nodes.keys() {
                                if harp_kids.contains(id) {
                                    continue; // hidden machinery, not a node
                                }
                                if r.contains(view.to_screen(dp(*id)))
                                    && !selection.contains(id)
                                {
                                    selection.push(*id);
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
            self.insert_target = None;
        }

        // --- `.` + press: cut (or take) the dot and start its pull ---
        if let (Some(press), Some(p)) = (dot_press, pointer) {
            let id = match press {
                DotPress::Dot(id) => id,
                DotPress::Wire(edge, at) => cut_dot(graph, edge, at),
            };
            if let Some(bead) = graph.node(id).map(|n| view.to_screen(n.pos)) {
                // The mean automatic handle over the bead's hops, so the
                // handle tips land where the pointer is.
                let mut ends: Vec<u64> = graph.input_of(id, 0).into_iter().collect();
                ends.extend(graph.edges.iter().filter(|e| e.2 == id).map(|e| e.0));
                let lens: Vec<f32> = ends
                    .iter()
                    .filter_map(|e| graph.node(*e))
                    .map(|n| dot_handle_len(bead, view.to_screen(n.pos), zoom_now))
                    .collect();
                let reference = if lens.is_empty() {
                    dot_handle_len(bead, bead, zoom_now)
                } else {
                    lens.iter().sum::<f32>() / lens.len() as f32
                };
                *selection = vec![id];
                self.sel_groups.clear();
                self.dot_place = Some(DotPlace { id, press: p, reference, pulling: false });
            }
        }

        // --- selection clicks (manual press->release tracking; egui's click
        // attribution proved timing-flaky under synthetic input) ---
        if ui.input(|i| i.pointer.primary_pressed())
            && resp.hovered()
            // A click while a pie or the M popup is open belongs to it.
            && self.pie.is_none()
            && self.ops_pie.is_none()
            && self.node_pie.is_none()
            && self.region_menu.is_none()
            // A `.` press belongs to its dot.
            && self.dot_place.is_none()
        {
            self.press_state = pointer.map(|p| (p, hit_node, hit_port));
        }
        if ui.input(|i| i.pointer.primary_released()) {
            if let (Some((op, p_node, p_port)), Some(p)) = (self.press_state.take(), pointer) {
                let innard = self
                    .innard_hits
                    .iter()
                    .find(|(_, _, h, live)| *live && h.iter().any(|(_, r)| r.contains(p)))
                    .map(|(id, _, h, _)| (*id, h.clone()));
                let ctl = self
                    .ctl_hits
                    .iter()
                    .filter(|(.., live)| *live)
                    .find_map(|(id, place, h, _)| Some((*id, crate::nodectl::hit_at(*place, h, p)?)));
                if (p - op).length() < 5.0 && ctl.is_some() {
                    // A click on a control selects its node; a double-click
                    // puts the control back to its default.
                    if let Some((cid, hit)) = ctl {
                        if ui.input(|i| i.pointer.button_double_clicked(egui::PointerButton::Primary)) {
                            if let Some(n) = graph.node(cid) {
                                for (ch, v) in crate::nodectl::reset(&n.kind, hit) {
                                    self.ctl_edits.push((cid, ch, v));
                                }
                            }
                        }
                        *selection = vec![cid];
                    }
                } else if (p - op).length() < 5.0 && innard.is_some() {
                    // A tiny node inside a zoomed-in node: select it, `+`
                    // adds, alt-click removes; the node itself becomes the
                    // selection so the viewer and panel edit that item.
                    if let Some((iid, hits)) = innard {
                        let alt = ui.input(|i| i.modifiers.alt);
                        crate::innards::click(ui.ctx(), graph, iid, &hits, p, alt);
                        *selection = vec![iid];
                    }
                } else if (p - op).length() < 5.0 {
                    if let Some((bid, rect_hit)) = self
                        .plus_hits
                        .iter()
                        .find(|(_, r)| r.contains(p))
                        .copied()
                    {
                        match graph.nodes.get_mut(&bid).map(|n| &mut n.kind) {
                            // A Switch's glyph: throw the lever. Through the
                            // keyed write-back, so autokey, locks and patch
                            // cables behave as a properties edit would.
                            Some(NodeKind::Switch(sw)) => {
                                let (c, r) = (rect_hit.center(), rect_hit.width() / (2.0 * SWITCH_HIT));
                                let to = switch_pick(c, r, sw.n(), sw.live(), p);
                                if to != sw.live() {
                                    self.ctl_edits.push((bid, crate::anim::Channel::SwitchWhich, to as f32));
                                }
                                *selection = vec![bid];
                            }
                            // An Extract's +: one more matte/output pair.
                            Some(NodeKind::Extract(x)) => {
                                x.n = (x.pairs() as u32 + 1).min(engine::graph::EXTRACT_MAX);
                            }
                            // The bar's +: the app opens the layer list there.
                            _ => self.plus_request = Some((bid, p)),
                        }
                    } else if let Some(port) = p_port {
                        // A grabbed noodle lands here; otherwise the port
                        // just becomes the selected one, which is what `G`
                        // picks a noodle up from.
                        match self.grab.take() {
                            Some(from) => {
                                land_grab(graph, from, port, zoom_now);
                                self.sel_port = Some(port);
                            }
                            None => self.sel_port = Some(port),
                        }
                        // A dot IS its port; clicking it also selects the
                        // node, so Delete and drag behave like on any node.
                        if graph.is_dot(port.node()) && !selection.contains(&port.node()) {
                            *selection = vec![port.node()];
                        }
                    } else if let Some(id) = p_node {
                        // z/x are also drag-rate keys on parameter rows;
                        // while a row claims them they are not a pin chord.
                        let typing = ui.ctx().egui_wants_keyboard_input()
                            || crate::keymap::capturing()
                            || crate::widgets::sens_grabbing(ui.ctx());
                        let z_held = !typing
                            && ui.input(|i| crate::keymap::down(i, crate::keymap::Act::GraphPinPanels));
                        let x_held = !typing
                            && ui.input(|i| crate::keymap::down(i, crate::keymap::Act::GraphPinGizmos));
                        let shift = ui.input(|i| i.modifiers.shift);
                        // Z/X-click ADD: the node's controls join whatever is
                        // already pinned. With shift they REPLACE: everything
                        // else is cleared and only this node stays. Unpinning
                        // is the panel's × and the viewer's HUD chips — a
                        // second click never toggles something away, so a
                        // pin is always safe to repeat. Both keys held is
                        // both gestures at once — shift+Z+X-click solos the
                        // node's panel AND its manipulators in one click —
                        // so the two are read independently, never else-if.
                        if x_held {
                            // X-click: this node's viewer manipulators
                            // (persists across selection).
                            self.x_used = true;
                            if shift {
                                gizmo_pinned.clear();
                            }
                            if !gizmo_pinned.contains(&id) {
                                gizmo_pinned.push(id);
                            }
                        }
                        if z_held {
                            // Z-click: the node's properties panel (persists
                            // independently of selection).
                            self.z_used = true;
                            if shift {
                                activated.clear();
                            }
                            if !activated.contains(&id) {
                                activated.push(id);
                            }
                        }
                        if x_held || z_held {
                        } else if shift {
                            // Picking a NODE retires the picked port: the
                            // view keys follow the last thing you chose,
                            // and a port from three clicks ago is not it.
                            self.sel_port = None;
                            if !selection.contains(&id) {
                                selection.push(id);
                            }
                        } else {
                            self.sel_port = None;
                            *selection = vec![id];
                        }
                    } else if let Some(nid) = self
                        .label_hits
                        .iter()
                        .find(|(_, q, ..)| quad_contains(q, op))
                        .map(|(id, ..)| *id)
                    {
                        // A click on a node's NAME: selects the node, and a
                        // second one on the same name inside the double-
                        // click window renames it (Houdini's gesture).
                        let now = ui.input(|i| i.time);
                        let double = self.last_name_click.is_some_and(|(lid, t, at)| {
                            lid == nid && now - t < 0.45 && at.distance(p) < 8.0
                        });
                        if double {
                            self.last_name_click = None;
                            let buf =
                                graph.node(nid).map(|n| n.ui_label.clone()).unwrap_or_default();
                            self.rename = Some((nid, buf, true));
                        } else {
                            self.last_name_click = Some((nid, now, p));
                            self.sel_port = None;
                            if ui.input(|i| i.modifiers.shift) {
                                if !selection.contains(&nid) {
                                    selection.push(nid);
                                }
                            } else {
                                *selection = vec![nid];
                            }
                        }
                    } else if ui.input(|i| i.modifiers.shift) {
                        // Shift-click ON A WIRE cuts a routing dot into it,
                        // right where you clicked.
                        if let Some((edge, at)) = wire_under {
                            let id = cut_dot(graph, edge, at);
                            *selection = vec![id];
                        }
                    } else {
                        selection.clear();
                        self.sel_groups.clear();
                        self.sel_port = None;
                    }
                }
            }
        }
        // A node's NAME is not background: double-clicking it renames the
        // node (handled on the manual click path below) instead of reading
        // as a background double-click that resets the view.
        let name_hit = pointer.and_then(|p| {
            self.label_hits
                .iter()
                .find(|(_, q, ..)| quad_contains(q, p))
                .map(|(id, ..)| *id)
        });
        if resp.double_clicked() && !(name_hit.is_some() && hit_node.is_none()) {
            match hit_node {
                // Double-click a node: pin its properties (same as Z-click,
                // just discoverable without knowing the modifier). A node
                // with an external editor opens it too — a Blender node's
                // scene is edited in Blender, and this is the gesture.
                Some(id) => {
                    if !activated.contains(&id) {
                        activated.push(id);
                    }
                    if graph.node(id).is_some_and(|n| matches!(n.kind, NodeKind::Blender(_))) {
                        self.open_external = Some(id);
                    }
                }
                None => {
                    self.zoom = 1.0;
                    self.pan = Vec2::ZERO;
                }
            }
        }
        if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
            // A popup swallows the Escape that closes it; the selection
            // only clears on a bare one. A colour picker open anywhere
            // counts: it takes the key itself, and clearing the selection
            // under it would drop a HUD chip's picker mid-pick.
            if self.pie.take().is_some()
                || self.rename.take().is_some()
                || self.ops_pie.take().is_some()
                || self.node_pie.take().is_some()
                || self.region_menu.take().is_some()
                || self.name_popup.take().is_some()
                || crate::colorpicker::any_open(ui.ctx())
            {
            } else {
                selection.clear();
                self.sel_groups.clear();
                self.sel_port = None;
                self.grab = None;
            }
        }
        let mut read_removed = false;
        // Tab: node-search popup at the cursor. The app strips
        // the Tab key from raw input and sets this flag instead.
        if std::mem::take(&mut self.tab_open_request)
            && resp.hovered()
            && !ui.ctx().egui_wants_keyboard_input()
        {
            let at = pointer.map(|p| view.to_graph(p)).unwrap_or(Pos2::ZERO);
            self.search = Some(NodeSearch { at, query: String::new(), sel: 0 });
        }
        if !ui.ctx().egui_wants_keyboard_input() && !crate::keymap::capturing() {
            use crate::keymap::{self, Act};
            // A key that edits the GRAPH only fires while the graph is under
            // the cursor. Otherwise a letter typed at the properties panel or
            // the viewer reaches back and bypasses or deletes nodes the hand
            // is nowhere near — and single letters are exactly what the
            // colour axes on the rgb sliders hold down.
            let over = resp.hovered();
            // An op picked off the ` pie last frame fires through its own
            // key site below, exactly as if its key had been pressed.
            let pick = self.ops_pick.take();
            // The pie was OPENED over the graph, but the slice it was
            // released on can sit past the pane's edge (it is 62px out), and
            // the op then never fired — every op here asks `over`. A pick
            // counts as hovered for the frame it fires on.
            let over = over || pick.is_some();
            let fire = |ui: &egui::Ui, a: Act| {
                pick == Some(a) || ui.input(|i| keymap::pressed(i, a))
            };
            // Delete/Backspace removes the selection; D toggles bypass.
            // Over a zoomed-in node's inner graph, Delete takes the node
            // picked in THERE, not the host.
            let innard_deleted = over
                && ui.input(|i| keymap::pressed(i, Act::GraphDelete))
                && ui.input(|i| i.pointer.hover_pos()).is_some_and(|p| {
                    crate::innards::delete_picked(ui.ctx(), graph, &self.innard_hits, p)
                });
            if over && !innard_deleted && ui.input(|i| keymap::pressed(i, Act::GraphDelete))
            {
                for id in selection.drain(..) {
                    if matches!(
                        graph.node(id).map(|n| &n.kind),
                        Some(NodeKind::Read { .. })
                    ) {
                        read_removed = true;
                    }
                    // Inside a GenAI Pipeline an Input IS one of the
                    // node's inputs: removed from the node, not in here.
                    if matches!(graph.node(id).map(|n| &n.kind), Some(NodeKind::GenAiStep(genai::Op::Input { .. }))) {
                        continue;
                    }
                    graph.remove(id);
                }
                // A backdrop is decoration: deleting one deletes the
                // rectangle and nothing that was standing on it.
                let doomed = std::mem::take(&mut self.sel_groups);
                graph.groups.retain(|g| !doomed.contains(&g.id));
            }
            // Select all: every node you can see (a harp's machinery and
            // hidden controls stay out of it).
            if over && ui.input(|i| keymap::pressed(i, Act::GraphSelectAll)) {
                let lm = self.link_mode;
                let kids = crate::breakout::kid_set(graph);
                *selection = graph
                    .nodes
                    .iter()
                    .filter(|(id, n)| {
                        !kids.contains(id)
                            && (lm || crate::tools::control_of(&n.kind).is_none())
                    })
                    .map(|(id, _)| *id)
                    .collect();
                self.sel_port = None;
            }
            // Copy / cut / paste. ⌘C ⌘X ⌘V reach egui as its own Copy / Cut /
            // Paste events (the paste carrying the system clipboard's text),
            // never as key presses; the ctrl chords are keymap bindings and
            // use the buffer the last copy filled.
            if over {
                let (copy_ev, cut_ev, paste_ev) = ui.input(|i| {
                    let mut out = (false, false, None);
                    for e in &i.events {
                        match e {
                            egui::Event::Copy => out.0 = true,
                            egui::Event::Cut => out.1 = true,
                            egui::Event::Paste(t) => out.2 = Some(t.clone()),
                            _ => {}
                        }
                    }
                    out
                });
                let copy = copy_ev || ui.input(|i| keymap::pressed(i, Act::GraphCopy));
                if (copy || cut_ev) && !selection.is_empty() {
                    if let Some(json) = copy_nodes(graph, selection) {
                        if self.system_clipboard {
                            ui.ctx().copy_text(json.clone());
                        }
                        self.clip = Some(json);
                    }
                    if cut_ev {
                        for id in selection.drain(..) {
                            if graph.node(id).is_some_and(|n| n.kind.reader_slot().is_some()) {
                                read_removed = true;
                            }
                            graph.remove(id);
                        }
                    }
                }
                let paste = paste_ev.or_else(|| {
                    ui.input(|i| keymap::pressed(i, Act::GraphPaste))
                        .then(|| self.clip.clone())
                        .flatten()
                });
                if let (Some(text), Some(p)) = (paste, pointer) {
                    let at = view.to_graph(p);
                    // A node clip first; else footage named by paths or a
                    // script's Read blocks becomes Loads at the cursor.
                    let ids = paste_nodes(graph, &text, at, next_read_slot)
                        .or_else(|| paste_loads(graph, &text, at, next_read_slot));
                    if let Some(ids) = ids {
                        *selection = ids;
                        self.sel_port = None;
                        self.pasted = true;
                    }
                }
            }
            // E: expression-link mode. The image wiring steps aside and
            // every node splays its channels for patching. (Was L, which now
            // opens the media browser.)
            // Not inside a noise graph: there, a socket takes an ordinary
            // edge and a value is a Slider node, so a second cable system
            // would be two ways to drive the same float.
            if over
                && !self.in_noise
                && !self.in_geo
                && !self.in_sdf
                && !self.in_particles && !self.in_genai && !self.in_roto
                && ui.input(|i| keymap::pressed(i, Act::GraphLinkMode))
            {
                self.link_mode = !self.link_mode;
                self.drag = None;
            }
            if self.in_noise || self.in_geo || self.in_sdf || self.in_particles || self.in_genai || self.in_roto {
                self.link_mode = false;
            }
            // Shift+`: node-patch mode. The ordinary noodles lock and grey,
            // the hidden layer draws in amber and is the one you edit —
            // wires dragged now are created drawn-hidden. Exclusive with
            // link mode: each is a different second wiring layer, and two
            // at once would be noise about noise.
            if over
                && !self.in_noise
                && !self.in_geo
                && !self.in_sdf
                && !self.in_particles && !self.in_genai && !self.in_roto
                && ui.input(|i| keymap::pressed(i, Act::GraphPatchMode))
            {
                self.patch_mode = !self.patch_mode;
                self.link_mode = false;
                self.drag = None;
            }
            if self.in_noise || self.in_geo || self.in_sdf || self.in_particles || self.in_genai || self.in_roto || self.link_mode {
                self.patch_mode = false;
            }
            // The source pies: a small orbit around the cursor, alive only
            // WHILE THE KEY IS HELD, releasing over a disc is the pick
            // (Blender's pie hands). ` serves the GLOBAL roster — the same
            // sources anywhere in the comp; cmd+` serves the region the
            // view centre stands in.
            // Plain `: the node-ops pie, held the same way.
            if over
                && self.pie.is_none()
                && self.ops_pie.is_none()
                && self.node_pie.is_none()
                && ui.input(|i| keymap::pressed(i, Act::GraphOpsPie))
            {
                if let Some(p) = pointer {
                    self.ops_pie = Some(p);
                    self.ops_anim = (0.0, 0.0);
                }
            }
            // F1: the most-used-nodes pie — the catalog ranked by how
            // often each kind has been ADDED, held and released over a
            // slice to drop that node under the cursor.
            if over
                && !self.in_noise
                && !self.in_geo
                && !self.in_sdf
                && !self.in_particles && !self.in_genai && !self.in_roto
                && self.pie.is_none()
                && self.ops_pie.is_none()
                && self.node_pie.is_none()
                && ui.input(|i| keymap::pressed(i, Act::GraphNodePie))
            {
                if let Some(p) = pointer {
                    self.node_pie = Some(p);
                    self.node_pie_anim = (0.0, 0.0);
                }
            }
            if over
                && !self.in_noise
                && !self.in_geo
                && !self.in_sdf
                && !self.in_particles && !self.in_genai && !self.in_roto
                && self.pie.is_none()
                && self.ops_pie.is_none()
                && self.node_pie.is_none()
            {
                let scope = if ui.input(|i| keymap::pressed(i, Act::GraphRegionPie)) {
                    Some(PieScope::Region(self.active_region))
                } else if ui.input(|i| keymap::pressed(i, Act::GraphSourcePie)) {
                    Some(PieScope::Global)
                } else {
                    None
                };
                if let (Some(scope), Some(p)) = (scope, pointer) {
                    let adding = (selection.len() == 1
                        && graph.nodes.contains_key(&selection[0])
                        && !harp_kids.contains(&selection[0]))
                    .then(|| selection[0]);
                    self.pie = Some(Pie { at: p, scope, adding });
                    self.pie_anim = (0.0, 0.0);
                }
            }
            // M: the region popup — add the selection to a region, or make
            // a new one over it (Blender's move-to-collection habit).
            // shift+H: the selected Reads become TVs (or stop). Mixed
            // selections all turn ON first, Blender's toggle rule.
            // shift+R: the range view, every noodle's first–last at once.
            if over && fire(ui, Act::GraphRanges) {
                self.show_ranges = !self.show_ranges;
            }
            if over && fire(ui, Act::GraphTv) {
                let reads: Vec<u64> = selection
                    .iter()
                    .copied()
                    .filter(|id| {
                        matches!(graph.node(*id).map(|n| &n.kind), Some(NodeKind::Read { .. }))
                    })
                    .collect();
                let on = reads.iter().any(|id| graph.node(*id).is_some_and(|n| !n.tv));
                for id in reads {
                    if let Some(n) = graph.node_mut(id) {
                        n.tv = on;
                    }
                }
            }
            // N: rename the one selected node (the usual key). The editor opens
            // over its name; see the label pass below.
            if over && fire(ui, Act::GraphRename) && selection.len() == 1 {
                let id = selection[0];
                if let Some(n) = graph.node(id) {
                    self.rename = Some((id, n.ui_label.clone(), true));
                }
            }
            // M: a TAP opens the region popup; a HOLD while dragging a node
            // over another makes a Blend (the drag handles that and spends
            // the hold). So the popup fires on RELEASE, unless spent.
            if ui.input(|i| keymap::pressed(i, Act::GraphRegionMenu)) {
                self.m_used = false;
            }
            let m_tap = ui.input(|i| keymap::released(i, Act::GraphRegionMenu))
                && !std::mem::take(&mut self.m_used);
            if over && (pick == Some(Act::GraphRegionMenu) || m_tap) {
                self.region_menu = match (self.region_menu.take(), pointer) {
                    (Some(_), _) | (_, None) => None,
                    (None, Some(p)) => Some((p, view.to_graph(p))),
                };
            }
            if over && fire(ui, Act::GraphBypass) {
                for id in selection.iter() {
                    if let Some(n) = graph.node_mut(*id) {
                        n.disabled = !n.disabled;
                    }
                }
            }
            // F: frame the selection (all nodes when nothing is selected),
            // like the viewer's F. Held R + drag rotates the layout instead.
            if over
                && (pick == Some(Act::GraphFrame)
                    || ui.input(|i| keymap::pressed(i, Act::GraphFrame)))
            {
                let ids: Vec<u64> = if selection.is_empty() {
                    graph.nodes.keys().copied().collect()
                } else {
                    selection.clone()
                };
                let mut bb = Rect::NOTHING;
                for id in &ids {
                    if let Some(n) = graph.node(*id) {
                        let e = (R + STALK + 30.0) * n.size;
                        bb = bb.union(Rect::from_center_size(n.pos, Vec2::splat(e * 2.0)));
                    }
                }
                if bb.is_positive() {
                    let z = ((rect.width() / bb.width()).min(rect.height() / bb.height())
                        * 0.85)
                        .clamp(0.002, 200.0);
                    self.zoom = z;
                    self.pan = -bb.center().to_vec2() * z;
                    self.suppress_nav_mirror = true;
                }
            }

            // shift+U: unify the selection's scale to its average.
            if over
                && fire(ui, Act::GraphUnifySize)
                && selection.len() >= 2
            {
                let sizes: Vec<f32> = selection
                    .iter()
                    .filter_map(|id| graph.node(*id).map(|n| n.size))
                    .collect();
                if !sizes.is_empty() {
                    let avg = sizes.iter().sum::<f32>() / sizes.len() as f32;
                    for id in selection.iter() {
                        if let Some(n) = graph.nodes.get_mut(id) {
                            n.size = avg;
                        }
                    }
                }
            }

            // Z / X as a TAP (a hold not spent on any click): the WHOLE
            // selection's controls go visible in one go — Z pins their
            // properties panels, X their viewer manipulators. Shift
            // REPLACES the pinned set instead of adding, so shift-tap with
            // nothing selected reads as "unpin everything". Mirrors the
            // click: click = one node, tap = the selection, shift = replace.
            {
                let (z_dn, x_dn, z_up, x_up, shift, cmd) = ui.input(|i| {
                    (
                        keymap::pressed(i, Act::GraphPinPanels),
                        keymap::pressed(i, Act::GraphPinGizmos),
                        keymap::released(i, Act::GraphPinPanels),
                        keymap::released(i, Act::GraphPinGizmos),
                        i.modifiers.shift,
                        i.modifiers.command,
                    )
                });
                if z_dn {
                    self.z_used = false;
                }
                if x_dn {
                    self.x_used = false;
                }
                // A hold spent adjusting a slider's drag rate is not a pin
                // tap: while a parameter row claims z/x, mark them spent.
                if crate::widgets::sens_grabbing(ui.ctx()) {
                    self.z_used = true;
                    self.x_used = true;
                }
                if over && !cmd {
                    if z_up && !std::mem::take(&mut self.z_used) {
                        if shift {
                            activated.clear();
                        }
                        for id in selection.iter() {
                            if !activated.contains(id) {
                                activated.push(*id);
                            }
                        }
                    }
                    if x_up && !std::mem::take(&mut self.x_used) {
                        if shift {
                            gizmo_pinned.clear();
                        }
                        for id in selection.iter() {
                            if !gizmo_pinned.contains(id) {
                                gizmo_pinned.push(*id);
                            }
                        }
                    }
                }
            }

            // G: pick a noodle UP off the selected port and carry it. It
            // trails the cursor across frames — pan, zoom, Tab in a new
            // node — until a click on a port lands it, or G/Esc drops it.
            // The point is reaching a node that is nowhere near the screen,
            // which a press-drag-release wire cannot do.
            if over && ui.input(|i| keymap::pressed(i, Act::GraphGrab)) {
                if self.grab.take().is_none() {
                    if let Some(port) = self.sel_port.or(hit_port) {
                        // Grabbing a WIRED input takes its wire off and
                        // carries the far end, exactly as dragging it does.
                        self.grab = match port {
                            Port::In(d, i) => match graph.input_of(d, i) {
                                Some(src) => {
                                    let o = graph.out_of(d, i);
                                    graph.disconnect_input(d, i);
                                    Some(Port::Out(src, o))
                                }
                                None => Some(port),
                            },
                            Port::Out(..) => Some(port),
                        };
                        self.drag = None;
                    }
                }
            }

            // S: exchange a node's two named inputs — a Blend's over/under,
            // a Mix's from/to. The wires swap; nothing else moves. A TAP, on
            // release: held + drag, S scales the selection's spacing.
            if over && ui.input(|i| keymap::pressed(i, Act::GraphSwapInputs)) {
                self.s_armed = true;
            }
            let s_tap = ui.input(|i| keymap::released(i, Act::GraphSwapInputs))
                && std::mem::take(&mut self.s_armed);
            if over && (pick == Some(Act::GraphSwapInputs) || s_tap) {
                let jobs: Vec<(u64, usize, usize)> = selection
                    .iter()
                    .filter_map(|id| {
                        let (a, b) = swappable_inputs(&graph.node(*id)?.kind)?;
                        Some((*id, a, b))
                    })
                    .collect();
                for (id, a, b) in jobs {
                    graph.swap_inputs(id, a, b);
                }
            }

            // T: tidy. Lay the selection (or the whole comp) out as a
            // layered top-to-bottom flow — see `crate::nodeux`.
            if over && fire(ui, Act::GraphTidy) {
                self.tidy_cascade(graph, &selection.clone());
            }

            // ⌘G groups the selection behind a backdrop; ⌘⇧G takes away
            // every backdrop the selection is standing on.
            {
                let (grp, ungrp) = ui.input(|i| {
                    (
                        keymap::pressed(i, Act::GraphGroup),
                        keymap::pressed(i, Act::GraphUngroup),
                    )
                });
                if over
                    && !selection.is_empty()
                    && (pick == Some(Act::GraphSubwob)
                        || ui.input(|i| keymap::pressed(i, Act::GraphSubwob)))
                {
                    self.collapse_req = true;
                }
                if over && !selection.is_empty() && fire(ui, Act::GraphBakeSelected) {
                    self.bake_req = true;
                }
                if over && !selection.is_empty() && fire(ui, Act::GraphCacheSelected) {
                    self.cache_req = true;
                }
                let grp = grp || pick == Some(Act::GraphGroup);
                let ungrp = ungrp || pick == Some(Act::GraphUngroup);
                if over && grp && !selection.is_empty() {
                    let id = graph.next_id;
                    graph.next_id += 1;
                    let g = crate::nodeux::group_over(selection, id);
                    let ids = g.members.clone();
                    graph.groups.push(g);
                    // One node, one backdrop: joining this group leaves any
                    // other, and a group emptied that way disappears.
                    graph.adopt_into_group(id, &ids);
                    self.sel_groups = vec![id];
                }
                if over && ungrp {
                    let sel = selection.clone();
                    graph
                        .groups
                        .retain(|g| !g.members.iter().any(|m| sel.contains(m)));
                    self.sel_groups.clear();
                }
            }

            // Two viewer slots: 1 assigns (and activates) slot A, 2 does
            // slot B; bare 1/2 with nothing under the cursor just flips the
            // active slot (quick A/B compare). (V is the properties toggle.)
            // Assignment needs the cursor over the GRAPH: over the viewer,
            // 1/2 only flip which slot shows, even with a node selected.
            // A hovered PORT outranks the node under it: an output socket
            // views THAT output, an input socket views whatever feeds it —
            // which is how you look at one string of a harp from either
            // end, and how a multi-band key's bands were always meant to
            // be picked.
            // With nothing under the cursor it is the SELECTION's turn —
            // the port first. Clicking a harp string's nodule and walking
            // the cursor away has to leave something to view, or the keys
            // only work while you hover the thing you already picked.
            let picked = |p: Port| -> Option<(u64, u32)> {
                match p {
                    Port::Out(n, o) => Some((n, o)),
                    Port::In(n, i) => {
                        graph.input_of(n, i).map(|src| (src, graph.out_of(n, i)))
                    }
                }
            };
            // A node a view key just took is SPENT while the pointer stays on
            // it: 1 then 2 without moving must flip to what slot 2 already
            // holds, not hand slot 2 the node too. Leaving it, or a click,
            // makes it a target again.
            let pressed_any = ui.input(|i| i.pointer.any_pressed());
            if pressed_any || self.view_spent.is_some_and(|n| hit_node != Some(n)) {
                self.view_spent = None;
            }
            let spent = self.view_spent;
            let live = |n: u64| Some(n) != spent;
            let target: Option<(u64, u32)> = hit_out
                .filter(|(n, _)| live(*n))
                .or_else(|| {
                    hit_in.filter(|(n, _)| live(*n)).and_then(|(n, i)| {
                        graph.input_of(n, i).map(|src| (src, graph.out_of(n, i)))
                    })
                })
                .or(hit_node.filter(|n| live(*n)).map(|n| (n, 0)))
                .or_else(|| self.sel_port.filter(|_| resp.hovered()).and_then(picked))
                .or_else(|| {
                    (resp.hovered() && selection.len() == 1)
                        .then(|| (selection[0], 0))
                });
            // A node handed to a viewer is done with: 1-4 deselect it (and
            // a picked port), so the next click starts fresh.
            let mut viewed = false;
            if ui.input(|i| keymap::pressed(i, Act::GraphView1)) {
                if let Some((t, o)) = target {
                    graph.view = Some(t);
                    viewed = true;
                    graph.view_out = o;
                }
                graph.view_active_b = false;
                graph.view_pane = 0;
            }
            if ui.input(|i| keymap::pressed(i, Act::GraphView2)) {
                if let Some((t, o)) = target {
                    graph.view_b = Some(t);
                    viewed = true;
                    graph.view_b_out = o;
                }
                graph.view_active_b = true;
                graph.view_pane = 0;
            }
            if ui.input(|i| keymap::pressed(i, Act::GraphViewLock)) {
                self.view_lock_req = Some(target);
            }
            // 3 and 4 are the same two slots for the SECOND viewer. In
            // single-view mode they still assign, and aim the viewer there,
            // so turning dual view on shows what you already picked.
            if ui.input(|i| keymap::pressed(i, Act::GraphView3)) {
                if let Some((t, o)) = target {
                    graph.view_c = Some(t);
                    viewed = true;
                    graph.view_c_out = o;
                }
                graph.view2_active_d = false;
                graph.view_pane = 1;
            }
            if ui.input(|i| keymap::pressed(i, Act::GraphView4)) {
                if let Some((t, o)) = target {
                    graph.view_d = Some(t);
                    viewed = true;
                    graph.view_d_out = o;
                }
                graph.view2_active_d = true;
                graph.view_pane = 1;
            }
            if viewed {
                selection.clear();
                self.sel_port = None;
                self.view_spent = hit_node;
            }
            // ⌘↑ / ⌘↓: swap the selected node with its parent / child in the
            // chain (1-in-1-out connections only, unambiguous rewire). The
            // two nodes exchange positions; the exchange is animated.
            if selection.len() == 1 {
                let sel = selection[0];
                let swapped = if ui.input(|i| keymap::pressed(i, Act::GraphSwapUp)) {
                    graph.swap_up(sel)
                } else if ui.input(|i| keymap::pressed(i, Act::GraphSwapDown)) {
                    graph.single_consumer(sel).and_then(|c| graph.swap_up(c))
                } else {
                    None
                };
                if let Some(pair) = swapped {
                    for (id, delta) in pair {
                        self.anim.entry(id).or_insert((Vec2::ZERO, 0.0)).0 += delta;
                    }
                }
            }
        }
        selection.retain(|id| graph.nodes.contains_key(id));
        // Port selection and a held noodle die with their node; a backdrop
        // selection dies with its backdrop.
        self.sel_port = self.sel_port.filter(|p| graph.nodes.contains_key(&p.node()));
        self.grab = self.grab.filter(|p| graph.nodes.contains_key(&p.node()));
        self.sel_groups.retain(|id| graph.groups.iter().any(|g| g.id == *id));

        // Every noodle carries a first and last frame: hover a wire and it
        // says which. Not over a node, and not mid-gesture — it is a
        // reading, not a control.
        if let (Some(((_, _, src), _)), Some(p)) = (wire_under, pointer) {
            let range = self.wire_ranges.get(&src).copied();
            if let (None, Some((a, b))) = (hit_node, range) {
                if !ui.input(|i| i.pointer.any_down()) {
                    let painter = ui.ctx().layer_painter(egui::LayerId::new(
                        egui::Order::Tooltip,
                        ui.id().with("wire-range"),
                    ));
                    let galley = painter.layout_no_wrap(
                        self.range_detail(graph, src, (a, b)),
                        egui::FontId::monospace(11.0),
                        crate::theme::text(),
                    );
                    let at = p + Vec2::new(14.0, -20.0);
                    let r = egui::Rect::from_min_size(at, galley.size()).expand2(Vec2::new(5.0, 2.0));
                    painter.rect_filled(r, 3.0, crate::theme::well());
                    painter.rect_stroke(
                        r,
                        3.0,
                        egui::Stroke::new(1.0, crate::theme::border()),
                        egui::StrokeKind::Inside,
                    );
                    painter.galley(at, galley, crate::theme::text());
                    self.shown_wire_range = Some((src, (a, b)));
                }
            }
        }

        // --- context menu ---
        if resp.secondary_clicked() {
            if let Some(p) = pointer {
                self.ctx_target = Some((hit_node, view.to_graph(p)));
                self.ctx_wire = wire_under;
                self.ctx_group = group_at(p).0;
                self.ctx_region = self
                    .region_hits
                    .iter()
                    .find(|(_, r)| r.contains(p))
                    .map(|(id, _)| *id);
            }
        }
        resp.context_menu(|ui| {
            // Same squeeze-guard as the category submenu below: opened near
            // the window's right edge, the popup gets the leftover width and
            // egui wraps every label — at its worst one word per line. A
            // minimum width makes it SHIFT to fit instead.
            ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
            ui.set_min_width(170.0);
            let Some((target, gpos)) = self.ctx_target else {
                ui.close();
                return;
            };
            // A region's own menu takes the click when it was opened on the
            // region's NAME chip and nothing nearer is under it.
            if let Some(rid) = self
                .ctx_region
                .filter(|_| target.is_none() && self.ctx_group.is_none())
            {
                if let Some(i) = graph.regions.iter().position(|r| r.id == rid) {
                    ui.label(egui::RichText::new("region").small().weak());
                    ui.add(
                        egui::TextEdit::singleline(&mut graph.regions[i].name)
                            .hint_text("name")
                            .desired_width(140.0),
                    );
                    ui.horizontal(|ui| {
                        for c in crate::regions::REGION_COLORS {
                            let col = Color32::from_rgb(c[0], c[1], c[2]);
                            let (r, resp) = ui.allocate_exact_size(
                                egui::Vec2::splat(16.0),
                                egui::Sense::click(),
                            );
                            ui.painter().rect_filled(r, 2.0, col);
                            if resp.clicked() {
                                graph.regions[i].color = c;
                            }
                        }
                    });
                    if !selection.is_empty()
                        && ui
                            .button("adopt the selection")
                            .on_hover_text("the selected nodes join this region (and leave any other)")
                            .clicked()
                    {
                        crate::regions::adopt(graph, rid, &selection.clone());
                        ui.close();
                    }
                    if graph.regions.get(i).is_some_and(|r| {
                        r.members.iter().any(|m| selection.contains(m))
                    }) && ui.button("release the selection").clicked()
                    {
                        let sel = selection.clone();
                        if let Some(r) = graph.regions.get_mut(i) {
                            r.members.retain(|m| !sel.contains(m));
                        }
                        crate::regions::prune(graph);
                        ui.close();
                    }
                    if ui.button("select members").clicked() {
                        *selection = graph
                            .regions
                            .get(i)
                            .map(|r| r.members.clone())
                            .unwrap_or_default();
                        ui.close();
                    }
                    if ui
                        .button("dissolve region")
                        .on_hover_text(
                            "the wash and its source registry go; the nodes \
                             and every wire stay",
                        )
                        .clicked()
                    {
                        graph.regions.retain(|r| r.id != rid);
                        ui.close();
                    }
                    return;
                }
            }
            // A backdrop's own menu takes the click when the pointer is on
            // its title bar and no node is under it.
            if let Some(gid) = self.ctx_group.filter(|_| target.is_none()) {
                if let Some(i) = graph.groups.iter().position(|g| g.id == gid) {
                    ui.label(egui::RichText::new("backdrop").small().weak());
                    ui.add(
                        egui::TextEdit::singleline(&mut graph.groups[i].title)
                            .hint_text("title")
                            .desired_width(140.0),
                    );
                    ui.horizontal(|ui| {
                        for c in crate::nodeux::GROUP_COLORS {
                            let col = Color32::from_rgb(c[0], c[1], c[2]);
                            let (r, resp) = ui.allocate_exact_size(
                                egui::Vec2::splat(16.0),
                                egui::Sense::click(),
                            );
                            ui.painter().rect_filled(r, 2.0, col);
                            if resp.clicked() {
                                graph.groups[i].color = c;
                            }
                        }
                    });
                    if !selection.is_empty()
                        && ui
                            .button("adopt the selection")
                            .on_hover_text(
                                "the selected nodes join this backdrop (and leave \
                                 any other); dropping a node inside the blob does \
                                 the same",
                            )
                            .clicked()
                    {
                        graph.adopt_into_group(gid, &selection.clone());
                        ui.close();
                    }
                    if graph.groups[i].members.iter().any(|m| selection.contains(m))
                        && ui.button("release the selection").clicked()
                    {
                        let sel = selection.clone();
                        graph.groups[i].members.retain(|m| !sel.contains(m));
                        graph.prune_groups();
                        ui.close();
                    }
                    if ui.button("select members").clicked() {
                        *selection = graph.groups[i].live_members(graph);
                        ui.close();
                    }
                    if ui.button("remove backdrop").clicked() {
                        graph.groups.remove(i);
                        self.sel_groups.retain(|g| *g != gid);
                        ui.close();
                    }
                    return;
                }
            }
            match target {
                Some(id) => {
                    if let Some(n) = graph
                        .nodes
                        .get_mut(&id)
                        .filter(|n| matches!(n.kind, NodeKind::Read { .. }))
                    {
                        ui.checkbox(&mut n.tv, "TV thumbnail  shift+H")
                            .on_hover_text("draw this Load as a little CRT showing its frame — zoom in to scrub it");
                    }
                    // Precache: the app does the work (it owns the player
                    // and the budgets); the menu only names the Reads.
                    if matches!(graph.node(id).map(|n| &n.kind), Some(NodeKind::Read { .. })) {
                        let is_read = |g: &EditorGraph, n: &u64| {
                            matches!(g.node(*n).map(|n| &n.kind), Some(NodeKind::Read { .. }))
                        };
                        let picked: Vec<u64> = if selection.contains(&id) {
                            selection.iter().copied().filter(|n| is_read(graph, n)).collect()
                        } else {
                            vec![id]
                        };
                        let label = if picked.len() > 1 {
                            format!("precache {} selected Loads", picked.len())
                        } else {
                            "precache this Load".into()
                        };
                        if ui
                            .button(label)
                            .on_hover_text(
                                "decode every frame of the range into the cache now, \
                                 in the background, so playback starts warm",
                            )
                            .clicked()
                        {
                            self.precache_req = Some(picked);
                            ui.close();
                        }
                        let all: Vec<u64> =
                            graph.nodes.keys().copied().filter(|n| is_read(graph, n)).collect();
                        if all.len() > 1
                            && ui.button(format!("precache all Loads ({})", all.len())).clicked()
                        {
                            self.precache_req = Some(all);
                            ui.close();
                        }
                    }
                    // Rename: presentation only, same contract as a
                    // control's caption — the node's KIND stays visible as
                    // the hint, so a renamed node never hides what it is.
                    if let Some(n) = graph.nodes.get_mut(&id) {
                        let hint = node_label(&n.kind);
                        ui.add(
                            egui::TextEdit::singleline(&mut n.ui_label)
                                .hint_text(hint)
                                .desired_width(140.0),
                        );
                    }
                    // Exchanging a Blend's over/under is the single most
                    // asked-for edit on a merge; it lives on the node's own
                    // menu as well as on S so it is findable.
                    if let Some((a, b)) =
                        graph.node(id).and_then(|n| swappable_inputs(&n.kind))
                    {
                        let names = graph
                            .node(id)
                            .map(|n| {
                                (
                                    n.kind.input_tag(a).unwrap_or("A"),
                                    n.kind.input_tag(b).unwrap_or("B"),
                                )
                            })
                            .unwrap_or(("A", "B"));
                        if ui
                            .button(format!("swap {} / {} inputs", names.0, names.1))
                            .on_hover_text("tap S — the wires exchange ports; nothing moves")
                            .clicked()
                        {
                            graph.swap_inputs(id, a, b);
                            ui.close();
                        }
                    }
                    if let Some(gid) = graph.group_of(id) {
                        if ui.button("release from backdrop").clicked() {
                            if let Some(g) =
                                graph.groups.iter_mut().find(|g| g.id == gid)
                            {
                                g.members.retain(|m| *m != id);
                            }
                            graph.prune_groups();
                            ui.close();
                        }
                    }
                    // The GLOBAL roster: offered anywhere, so it is offered
                    // on every node's menu, region or no region.
                    {
                        let is_global = graph.global_sources.contains(&id);
                        if !is_global
                            && ui
                                .button("make a global source")
                                .on_hover_text(
                                    "the ` pie offers it anywhere in the comp; \
                                     picking it spawns a chip hidden-wired here",
                                )
                                .clicked()
                        {
                            crate::regions::add_global_source(graph, id);
                            ui.close();
                        }
                        if is_global && ui.button("retire the global source").clicked() {
                            graph.global_sources.retain(|s| *s != id);
                            ui.close();
                        }
                    }
                    if let Some(rid) = crate::regions::region_of(graph, id) {
                        let is_src = graph
                            .regions
                            .iter()
                            .any(|r| r.id == rid && r.sources.contains(&id));
                        if !is_src
                            && ui
                                .button("make a common source")
                                .on_hover_text(
                                    "the region's cmd+` pie offers it; picking it \
                                     spawns a chip hidden-wired here",
                                )
                                .clicked()
                        {
                            if let Some(r) =
                                graph.regions.iter_mut().find(|r| r.id == rid)
                            {
                                r.sources.push(id);
                            }
                            ui.close();
                        }
                        if is_src && ui.button("retire the common source").clicked() {
                            if let Some(r) =
                                graph.regions.iter_mut().find(|r| r.id == rid)
                            {
                                r.sources.retain(|s| *s != id);
                            }
                            ui.close();
                        }
                        if ui.button("release from region").clicked() {
                            if let Some(r) =
                                graph.regions.iter_mut().find(|r| r.id == rid)
                            {
                                r.members.retain(|m| *m != id);
                            }
                            crate::regions::prune(graph);
                            ui.close();
                        }
                    } else if !graph.regions.is_empty() {
                        ui.menu_button("add to region", |ui| {
                            ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
                            ui.set_min_width(120.0);
                            let names: Vec<(u64, String)> = graph
                                .regions
                                .iter()
                                .map(|r| (r.id, r.name.clone()))
                                .collect();
                            for (rid, name) in names {
                                if ui.button(name).clicked() {
                                    crate::regions::adopt(graph, rid, &[id]);
                                    ui.close();
                                }
                            }
                        });
                    }
                    let dis = graph.node(id).is_some_and(|n| n.disabled);
                    if ui.button(if dis { "enable" } else { "disable" }).clicked() {
                        if let Some(n) = graph.node_mut(id) {
                            n.disabled = !n.disabled;
                        }
                        ui.close();
                    }
                    if ui.button("remove").clicked() {
                        if matches!(
                            graph.node(id).map(|n| &n.kind),
                            Some(NodeKind::Read { .. })
                                | Some(NodeKind::GraphBake { .. })
                                | Some(NodeKind::GenAi(_))
                                | Some(NodeKind::CorridorKey(_))
                                | Some(NodeKind::Sammie(_))
                                | Some(NodeKind::Flow(_))
                                | Some(NodeKind::Rifetime(_))
                                | Some(NodeKind::FrameFix(_))
                        ) {
                            read_removed = true;
                        }
                        graph.remove(id);
                        ui.close();
                    }
                }
                None => {
                    // Cutting a dot into the wire you right-clicked over.
                    if let Some(((dst, input, src), at)) = self.ctx_wire {
                        if ui
                            .button("cut a dot in here")
                            .on_hover_text("hold . and click a wire (drag for tension) — a routing anchor, free at render time")
                            .clicked()
                        {
                            cut_dot(graph, (dst, input, src), at);
                            ui.close();
                        }
                        ui.separator();
                    }
                    if !selection.is_empty() {
                        if ui
                            .button("group the selection")
                            .on_hover_text("⌘G — a titled backdrop behind them")
                            .clicked()
                        {
                            let gid = graph.next_id;
                            graph.next_id += 1;
                            let g = crate::nodeux::group_over(selection, gid);
                            let ids = g.members.clone();
                            graph.groups.push(g);
                            graph.adopt_into_group(gid, &ids);
                            self.sel_groups = vec![gid];
                            ui.close();
                        }
                        if ui
                            .button("region over the selection")
                            .on_hover_text(
                                "M — a named wash a whole render layer lives \
                                 in; its ` pie serves the sources you register",
                            )
                            .clicked()
                        {
                            crate::regions::create(graph, &selection.clone());
                            ui.close();
                        }
                    }
                    if ui
                        .button(if selection.is_empty() { "tidy the graph" } else { "tidy the selection" })
                        .on_hover_text("T — layered top-to-bottom flow, chains straightened")
                        .clicked()
                    {
                        self.tidy_cascade(graph, &selection.clone());
                        ui.close();
                    }
                    ui.separator();
                    let lm = self.link_mode;
                    let label =
                        if lm { "✔ expression links (L)" } else { "expression links (L)" };
                    if ui
                        .button(label)
                        .on_hover_text(
                            "swap the image wiring for value cables: every node \
                             splays its channels, inputs left and outputs right",
                        )
                        .clicked()
                    {
                        self.link_mode = !lm;
                        ui.close();
                    }
                    ui.separator();
                    let mut add: Option<NodeKind> = None;
                    // Grouped by what a node is FOR: thirty-odd buttons in
                    // one column is a list you read rather than a menu you
                    // use.
                    let (in_noise, in_geo, in_sdf, in_particles, in_genai, in_roto) =
                        (self.in_noise, self.in_geo, self.in_sdf, self.in_particles, self.in_genai, self.in_roto);
                    // The project's maturity threshold hides what is below
                    // it — from every submenu, plugins and controls included.
                    let min_status = graph.node_status_min;
                    let catalog: Vec<NodeEntry> = node_catalog()
                        .into_iter()
                        .filter(|e| offered_in(e.cat, in_noise, in_geo, in_sdf, in_particles, in_genai, in_roto))
                        .filter(|e| min_status.admits(e.status))
                        .collect();
                    for cat in NodeCat::ALL {
                        // A submenu that opens onto nothing is worse than no
                        // submenu: inside a Noise node every category but
                        // `field` is empty, and outside one `field` is.
                        if !catalog.iter().any(|e| e.cat == cat) {
                            continue;
                        }
                        ui.menu_button(cat.name(), |ui| {
                            // Opened near the right edge of the graph panel,
                            // a submenu gets squeezed and egui wraps the
                            // labels — at its worst, one CHARACTER per line.
                            // Menus live on the popup layer, so they may
                            // overhang the panel; they must not fold up.
                            ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
                            ui.set_min_width(150.0);
                            for entry in catalog.iter().filter(|e| e.cat == cat) {
                                let text = status_text(entry.label, entry.status);
                                if ui
                                    .add(egui::Button::new(text))
                                    .on_hover_text(entry.status.name())
                                    .clicked()
                                {
                                    let slot = *next_read_slot;
                                    if entry.reader {
                                        *next_read_slot += 1;
                                    }
                                    *self
                                        .use_counts
                                        .entry(entry.label.to_string())
                                        .or_insert(0) += 1;
                                    self.use_counts_dirty = true;
                                    add = Some((entry.make)(slot));
                                    ui.close();
                                }
                            }
                        });
                    }
                    // Tool input proxies are body-only furniture: they mean
                    // nothing in a comp's own graph, so they stay out of the
                    // menus even though they are registered node types.
                    // None of these belong in a noise graph: a plugin is an
                    // image expression and a control is a tool's panel.
                    let custom_status = status_of(&NodeKind::Custom {
                        type_id: String::new(),
                        params: Default::default(),
                    });
                    let registered: Vec<_> = if self.in_noise
                        || self.in_geo
                        || self.in_sdf
                        || self.in_particles || self.in_genai || self.in_roto
                        || !min_status.admits(custom_status)
                    {
                        Vec::new()
                    } else {
                        engine::plugins::all()
                            .into_iter()
                            // Input placeholders (image and matte) belong
                            // inside a tool body, where adding one is how a
                            // tool gains a port; nowhere else.
                            .filter(|d| self.in_tool || !crate::tools::is_proxy_id(&d.type_id))
                            .collect()
                    };
                    // CONTROLS get their own submenu. They used to be
                    // offered only inside a tool, on the grounds that a comp
                    // had no use for a panel — no longer true: a comp that
                    // exports to an edit publishes its controls as the shot's
                    // panel (see `To Scrub` and the shot contract), authored
                    // exactly the way a tool's panel is.
                    {
                        let controls: Vec<_> = registered
                            .iter()
                            .filter(|d| crate::tools::ControlKind::from_type(&d.type_id).is_some())
                            .collect();
                        if !controls.is_empty() {
                            ui.menu_button("control", |ui| {
                                for d in controls {
                                    let text = status_text(&d.label, custom_status);
                                    if ui
                                        .add(egui::Button::new(text))
                                        .on_hover_text(custom_status.name())
                                        .clicked()
                                    {
                                        add = Some(NodeKind::Custom {
                                            type_id: d.type_id.clone(),
                                            params: Default::default(),
                                        });
                                        ui.close();
                                    }
                                }
                            });
                        }
                    }
                    // INPUT nodes (only offered inside a subwob — the filter
                    // above), then SUBWOBS (subnets saved as nodes —
                    // `tools.rs`), then shader plugins: three menus, so a
                    // subwob's parts never sit among subwobs.
                    let rest: Vec<_> = registered
                        .iter()
                        .filter(|d| crate::tools::ControlKind::from_type(&d.type_id).is_none())
                        .collect();
                    let mut inputs: Vec<_> =
                        rest.iter().copied().filter(|d| crate::tools::is_proxy_id(&d.type_id)).collect();
                    // In pipe order (image, matte, data, fat), not by name.
                    inputs.sort_by_key(|d| {
                        crate::tools::INPUTS.iter().position(|(id, _, _)| *id == d.type_id)
                    });
                    let (subwobs, plugins): (Vec<_>, Vec<_>) = rest
                        .into_iter()
                        .filter(|d| !crate::tools::is_proxy_id(&d.type_id))
                        .partition(|d| is_subwob_type(&d.type_id));
                    for (menu, list) in [("input", inputs), ("subwob", subwobs), ("plugin", plugins)] {
                        if list.is_empty() {
                            continue;
                        }
                        ui.menu_button(menu, |ui| {
                            ui.style_mut().wrap_mode =
                                Some(egui::TextWrapMode::Extend);
                            ui.set_min_width(150.0);
                            for d in list {
                                let text = status_text(&d.label, custom_status);
                                if ui
                                    .add(egui::Button::new(text))
                                    .on_hover_text(custom_status.name())
                                    .clicked()
                                {
                                    add = Some(NodeKind::Custom {
                                        type_id: d.type_id.clone(),
                                        params: Default::default(),
                                    });
                                    ui.close();
                                }
                            }
                        });
                    }
                    // Inside a 2dini: ready-wired networks, drawing and physics.
                    if in_geo {
                        ui.separator();
                        ui.menu_button("presets", |ui| {
                            ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
                            ui.set_min_width(150.0);
                            for physics in [false, true] {
                                if physics {
                                    ui.separator();
                                }
                                for (i, p) in engine::geopresets::PRESETS.iter().enumerate() {
                                    if p.physics == physics && ui.button(p.name).clicked() {
                                        self.geo_preset_req = Some((i, gpos));
                                        ui.close();
                                    }
                                }
                            }
                        });
                    }
                    if let Some(kind) = add {
                        let id = add_wired(graph, kind, gpos, selection, zoom, self.patch_mode);
                        self.added.push(id);
                        ui.close();
                    }
                }
            }
        });

        if let Some((dst, input, step)) = wire_z_step {
            let mut l = graph.wire_layering(dst, input);
            l.z = (l.z + step).round();
            graph.set_wire_layering(dst, input, l);
        }

        // Where a noodle over a Layer's (or a GenAI's) bar would land: a
        // bright gap.
        if let Some((_, _, gx, top, bot)) = hit_layer_bar {
            let p = ui.painter();
            p.line_segment([Pos2::new(gx, top), Pos2::new(gx, bot)], Stroke::new(3.0, theme::selection()));
            p.circle_filled(Pos2::new(gx, top), 4.0, theme::selection());
        }

        // The big dot's short list: which input the dropped noodle takes.
        if let Some(m) = self.big_dot_menu.clone() {
            let mut close = ui.input(|i| i.key_pressed(egui::Key::Escape));
            let names: Vec<(usize, &'static str)> = graph
                .node(m.dst)
                .map(|n| m.options.iter().map(|i| (*i, n.kind.input_name(*i))).collect())
                .unwrap_or_default();
            if names.is_empty() || graph.node(m.src).is_none() {
                close = true;
            }
            // The options SPLAY out round the drop point: each name on a
            // spoke, flung out with a little overshoot one after another,
            // the hovered one lit and grown. A click on one wires it.
            let now = ui.input(|i| i.time);
            let n = names.len().max(1);
            let ring = (56.0 + 9.0 * n as f32).min(150.0);
            let spoke_col = graph
                .node(m.src)
                .map(|sn| match sn.kind.output_kind_at(m.out as usize) {
                    PortKind::Geo => theme::wire_geo(),
                    PortKind::Field => theme::wire_number(),
                    k => socket_color(k),
                })
                .unwrap_or_else(theme::port_in);
            let pad = ring + 90.0;
            let area = egui::Area::new(egui::Id::new("big-dot-menu"))
                .order(egui::Order::Foreground)
                .fixed_pos(m.at - Vec2::splat(pad))
                .show(ui.ctx(), |ui| {
                    let (frame, _) = ui.allocate_exact_size(Vec2::splat(pad * 2.0), egui::Sense::hover());
                    let c = frame.center();
                    let p = ui.painter();
                    let font = egui::FontId::proportional(12.0);
                    let mut rows = Vec::new();
                    let mut animating = false;
                    // A soft disc at the hub: where the noodle is waiting.
                    let k0 = ((now - m.t0) / 0.18).clamp(0.0, 1.0) as f32;
                    p.circle_filled(c, 18.0 * k0, theme::graph_bg().gamma_multiply(0.85));
                    p.circle_stroke(c, 18.0 * k0, Stroke::new(2.0, spoke_col));
                    p.circle_filled(c, 5.0, spoke_col);
                    for (j, (i, name)) in names.iter().enumerate() {
                        let k = ((now - m.t0 - 0.035 * j as f64) / 0.24).clamp(0.0, 1.0) as f32;
                        animating |= k < 1.0;
                        // Ease out with an overshoot: thrown, then settling.
                        let u = k - 1.0;
                        let e = 1.0 + 2.7 * u * u * u + 1.7 * u * u;
                        let at = crate::regions::pie_slice_pos(c, j, n, ring * e);
                        let g = p.layout_no_wrap(name.to_string(), font.clone(), theme::text_strong());
                        let r = Rect::from_center_size(at, g.size() + Vec2::new(18.0, 10.0));
                        let resp = ui.interact(r, egui::Id::new(("big-dot-opt", m.dst, *i)), egui::Sense::click());
                        let hot = resp.hovered();
                        let grow = if hot { 1.12 } else { 1.0 };
                        let r = Rect::from_center_size(at, r.size() * grow);
                        p.line_segment([c + (at - c).normalized() * 18.0 * k0, r.center()], Stroke::new(if hot { 2.5 } else { 1.5 }, spoke_col.gamma_multiply(0.4 + 0.6 * k)));
                        p.rect_filled(r, r.height() * 0.5, if hot { spoke_col } else { theme::graph_bg() });
                        p.rect_stroke(r, r.height() * 0.5, Stroke::new(1.5, spoke_col), egui::StrokeKind::Middle);
                        let ink = if hot { theme::graph_bg() } else { theme::text_strong() };
                        p.text(r.center(), egui::Align2::CENTER_CENTER, *name, font.clone(), ink.gamma_multiply(k.max(0.2)));
                        rows.push((*i, r));
                        if resp.clicked() {
                            connect_or_bridge(graph, m.src, m.out, m.dst, *i, zoom);
                            close = true;
                        }
                    }
                    if animating {
                        ui.ctx().request_repaint();
                    }
                    rows
                });
            self.big_dot_rows = area.inner;
            // A press anywhere but on a name lets the noodle go.
            let elsewhere = ui.input(|i| i.pointer.any_pressed())
                && ui.input(|i| i.pointer.interact_pos())
                    .is_some_and(|p| !self.big_dot_rows.iter().any(|(_, r)| r.contains(p)));
            if close || elsewhere {
                self.big_dot_menu = None;
                self.big_dot_rows.clear();
            }
        }

        // Gestures ran above: re-assert the harps' x-lock so a dragged
        // receiver never DRAWS off its column (its y keeps the hand's).
        crate::breakout::lock_x(graph);

        // --- regions: the floor UNDER the paper. A quiet rounded wash per
        // region, derived from its members like a backdrop's blob but
        // deliberately flatter — regions are for orientation at arm's
        // length, not for wrapping. The name chip (top-left) is the
        // right-click target; the active one (view centre inside) draws a
        // shade brighter and is named in the panel corner, because it is
        // what the backtick pie will answer for.
        self.region_hits.clear();
        self.plus_hits.clear();
        self.innard_hits.clear();
        self.ctl_hits.clear();
        if !self.link_mode {
            for r in &graph.regions {
                let Some(w) = crate::regions::wash(graph, r) else { continue };
                let pts: Vec<Pos2> = w.iter().map(|p| view.to_screen(*p)).collect();
                let sb = pts
                    .iter()
                    .fold(Rect::NOTHING, |bb, p| bb.union(Rect::from_min_max(*p, *p)));
                if !sb.intersects(rect) {
                    continue;
                }
                let active = self.active_region == Some(r.id);
                let col = Color32::from_rgb(r.color[0], r.color[1], r.color[2]);
                painter.add(egui::Shape::convex_polygon(
                    pts.clone(),
                    col.gamma_multiply(if active { 0.045 } else { 0.028 }),
                    Stroke::NONE,
                ));
                painter.add(egui::Shape::closed_line(
                    pts.clone(),
                    Stroke::new(1.0, col.gamma_multiply(if active { 0.4 } else { 0.22 })),
                ));
                // The name rides the wash's crown — the topmost point of
                // the hull, wherever the wrap puts it.
                let crown = pts
                    .iter()
                    .copied()
                    .min_by(|a, b| a.y.total_cmp(&b.y))
                    .unwrap_or(sb.center());
                let fs = (13.0 * self.zoom.clamp(0.5, 1.4)).max(11.0);
                let tr = painter.text(
                    crown + Vec2::new(6.0, 4.0),
                    egui::Align2::LEFT_TOP,
                    &r.name,
                    egui::FontId::proportional(fs),
                    col.gamma_multiply(if active { 0.95 } else { 0.65 }),
                );
                self.region_hits.push((r.id, tr.expand(4.0)));
            }
        }

        // --- backdrops: drawn first, because they are the paper, not the
        // ink. Each one is a BLOB shrink-wrapped to its members (padded
        // discs + sleeves along their wires, smooth-unioned and marched to
        // a contour in `nodeux`), so the backdrop is the shape of the
        // selection, not a rectangle over it. The title tab is the move
        // handle, the wedge on the lower-right of the shape scales the
        // members; the body is inert so a marquee still works over it.
        self.group_hits.clear();
        {
            // Finer contour when zoomed in; the bucket is in the cache key
            // so a zoom change re-marches, a plain pan never does.
            let cell = (12.0 / self.zoom.clamp(0.25, 4.0)).clamp(3.0, 48.0);
            let live: std::collections::HashSet<u64> =
                graph.groups.iter().map(|g| g.id).collect();
            self.blobs.retain(|id, _| live.contains(id));
            for gi in 0..graph.groups.len() {
                let (gid, members, base, sel, title, pad, label_h) = {
                    let g = &graph.groups[gi];
                    (
                        g.id,
                        g.live_members(graph),
                        Color32::from_rgb(g.color[0], g.color[1], g.color[2]),
                        self.sel_groups.contains(&g.id),
                        g.title.clone(),
                        g.pad,
                        g.label_h.clamp(10.0, 90.0),
                    )
                };
                if members.is_empty() {
                    continue;
                }
                let key = crate::nodeux::blob_key(graph, &members, cell.to_bits(), pad);
                if self.blobs.get(&gid).is_none_or(|(k, _)| *k != key) {
                    match crate::nodeux::build_blob(graph, &members, cell, pad) {
                        Some(b) => {
                            self.blobs.insert(gid, (key, b));
                        }
                        None => {
                            self.blobs.remove(&gid);
                            continue;
                        }
                    }
                }
                let Some((_, blob)) = self.blobs.get(&gid) else { continue };
                // Off-screen groups cost nothing but their cache.
                let sb = Rect::from_two_pos(
                    view.to_screen(blob.bbox.min),
                    view.to_screen(blob.bbox.max),
                );
                if !sb.intersects(rect) {
                    continue;
                }
                // Fill: one mesh of the marching-squares triangles — cells
                // never overlap, so the translucency stays uniform however
                // concave the shape gets.
                let fill = base.gamma_multiply(0.20);
                let mut mesh = egui::Mesh::default();
                for t in &blob.tris {
                    let i0 = mesh.vertices.len() as u32;
                    for p in t {
                        mesh.vertices.push(egui::epaint::Vertex {
                            pos: view.to_screen(*p),
                            uv: egui::epaint::WHITE_UV,
                            color: fill,
                        });
                    }
                    mesh.indices.extend([i0, i0 + 1, i0 + 2]);
                }
                painter.add(egui::Shape::mesh(mesh));
                let stroke = Stroke::new(
                    if sel { 2.0 } else { 1.2 },
                    if sel { theme::selection() } else { base.gamma_multiply(0.9) },
                );
                for lp in &blob.loops {
                    let mut pts: Vec<Pos2> = lp.iter().map(|p| view.to_screen(*p)).collect();
                    if let (Some(a), Some(b)) = (pts.first().copied(), pts.last().copied()) {
                        if a != b {
                            pts.push(a); // seal the loop
                        }
                    }
                    painter.add(egui::Shape::line(pts, stroke));
                }
                // Title tab, hanging off the topmost member. Width follows
                // the name so a long title is readable, clipped so an
                // absurd one cannot write across the neighbours.
                let ls = label_h / crate::nodeux::TITLE_H;
                let tw = ((96.0f32).max(12.0 + 7.5 * title.chars().count() as f32)) * ls;
                let tab_g = Rect::from_min_size(
                    blob.tab_anchor - Vec2::new(tw * 0.5, label_h),
                    Vec2::new(tw, label_h),
                );
                let tr = Rect::from_two_pos(
                    view.to_screen(tab_g.min),
                    view.to_screen(tab_g.max),
                );
                painter.rect_filled(tr, 4.0, base.gamma_multiply(0.55));
                painter.rect_stroke(
                    tr,
                    4.0,
                    Stroke::new(if sel { 1.6 } else { 1.0 }, stroke.color),
                    egui::StrokeKind::Inside,
                );
                if tr.height() > 7.0 {
                    let name: &str = if title.is_empty() { "group" } else { &title };
                    painter.with_clip_rect(tr.intersect(rect)).text(
                        tr.center(),
                        egui::Align2::CENTER_CENTER,
                        name,
                        egui::FontId::proportional((tr.height() * 0.55).clamp(7.0, 72.0)),
                        if title.is_empty() {
                            theme::node_label().gamma_multiply(0.6)
                        } else {
                            theme::text_strong()
                        },
                    );
                }
                // The scale grip: a wedge ON the shape's lower-right,
                // screen-constant so it stays grabbable zoomed out.
                let gp = view.to_screen(blob.grip);
                painter.add(egui::Shape::convex_polygon(
                    vec![
                        gp + Vec2::new(5.0, 5.0),
                        gp + Vec2::new(5.0, -9.0),
                        gp + Vec2::new(-9.0, 5.0),
                    ],
                    base.gamma_multiply(0.9),
                    Stroke::new(1.0, theme::node_outline()),
                ));
                let gr = Rect::from_center_size(gp, Vec2::splat(18.0));
                self.group_hits.push((gid, tr, gr));
            }
        }

        // --- the graph's hotkey strip: a row of small caps along the very
        // top edge — the pies and the held gestures, each cap showing its
        // LIVE chord (they are remappable) and lighting while held. The
        // words stay put away until the hand hovers the row; then they
        // slide out of their caps down and to the right, left to right.
        // The caps are a hint layer UNDER the wires and nodes; the words,
        // once asked for, are drawn on top (see the end of `show`). Same
        // pref as the viewer's strip; the FullNodes overlay stands down —
        // the viewer's own strip owns the top edge there.
        let mut key_strip: Option<crate::hudkeys::TopStrip> = None;
        self.keys_row = None;
        if self.show_keys && !overlay {
            use crate::keymap as km;
            // The DISPLAY form (⌘T), not the prefs string. egui's long-form
            // formatter (non-mac, or no ⌘ glyphs) spells the key out —
            // "Period", "Backtick" — so the cap swaps in the key's glyph.
            let item = |a: km::Act, label: &str| {
                let key = km::binding(a).key;
                let mut cap = km::label(ui.ctx(), a);
                if let Some(stem) = cap.strip_suffix(key.name()) {
                    cap = format!("{stem}{}", key.symbol_or_name());
                }
                (cap, label.to_string(), ui.input(|i| km::down(i, a)))
            };
            let items = vec![
                item(km::Act::GraphOpsPie, "hold: ops pie"),
                item(km::Act::GraphNodePie, "hold: node pie"),
                item(km::Act::GraphSourcePie, "hold: global patch"),
                item(km::Act::GraphRegionPie, "hold: region patch"),
                item(km::Act::GraphSearch, "search"),
                item(km::Act::GraphGrab, "grab a noodle"),
                item(km::Act::GraphTidy, "tidy"),
                item(km::Act::GraphRename, "rename"),
                item(km::Act::GraphRegionMenu, "tap: region · hold + swipe: blend"),
                item(km::Act::GraphSwipeBlend, "hold + swipe: blend"),
                (
                    "I / O".to_string(),
                    "hold + swipe a matte: mask / stencil".to_string(),
                    ui.input(|i| {
                        km::down(i, km::Act::GraphSwipeIn) || km::down(i, km::Act::GraphSwipeOut)
                    }),
                ),
                item(km::Act::GraphAlign, "hold + swipe: align"),
                item(km::Act::GraphSwapInputs, "tap: swap inputs · hold + drag: scale spacing"),
                item(km::Act::GraphFrame, "frame"),
                item(km::Act::GraphRotate, "hold + drag: rotate"),
                item(km::Act::GraphDot, "hold + click a wire: dot (drag for tension)"),
            ];
            let strip = crate::hudkeys::TopStrip::new(&painter, rect, &items);
            strip.draw_caps(&painter);
            let hot = self.drag.is_none()
                && resp.hovered()
                && pointer.is_some_and(|p| {
                    strip.row().expand(6.0).contains(p)
                        || (self.keys_reveal > 0.3 && strip.fan(&painter).contains(p))
                });
            self.keys_row = Some(strip.row());
            let target = if hot || self.keys_hold { 1.0 } else { 0.0 };
            self.keys_reveal += (target - self.keys_reveal) * (1.0 - (-dt * 12.0).exp());
            if (target - self.keys_reveal).abs() > 0.004 {
                ui.ctx().request_repaint();
            } else {
                self.keys_reveal = target;
            }
            key_strip = Some(strip);
        }

        // --- the noodle legend: a row of coloured dots on the TOP-RIGHT
        // edge, one per thing a wire's colour can mean. The words appear
        // only while the hand hovers the dots — the key strip's mirror
        // image, each label hanging down-left and reading up into its dot.
        // The dots are a hint layer under the wires and nodes; the words,
        // once asked for, are drawn on top (see the end of `show`). Shown
        // on the graph's own top bar ("legend").
        let mut legend_strip: Option<crate::hudkeys::DotStrip> = None;
        self.legend_row = None;
        if self.show_legend && !overlay {
            use crate::hudkeys::DotStyle;
            // Inside a 2dini the noodles say what they carry instead.
            let entries: Vec<(&str, Color32, Swatch)> = if self.in_geo {
                GeoWire::ALL.iter().map(|w| (w.label(), w.color(), Swatch::Plain)).collect()
            } else if self.in_noise {
                field_legend()
            } else {
                legend_entries()
            };
            let items: Vec<(Color32, DotStyle, String)> = entries
                .into_iter()
                .map(|(label, c, sw)| {
                    let st = match sw {
                        Swatch::Plain => DotStyle::Plain,
                        Swatch::Glow => DotStyle::Glow,
                        Swatch::Fat => DotStyle::Fat,
                    };
                    (c, st, label.to_string())
                })
                .collect();
            let strip = crate::hudkeys::DotStrip::new(
                rect,
                &items,
                key_strip.as_ref().map(|k| k.row()),
            );
            strip.draw_dots(&painter);
            let hot = self.drag.is_none()
                && resp.hovered()
                && pointer.is_some_and(|p| {
                    strip.row().expand(6.0).contains(p)
                        || (self.legend_reveal > 0.3 && strip.fan(&painter).contains(p))
                });
            self.legend_row = Some(strip.row());
            let target = if hot || self.legend_hold { 1.0 } else { 0.0 };
            self.legend_reveal += (target - self.legend_reveal) * (1.0 - (-dt * 12.0).exp());
            if (target - self.legend_reveal).abs() > 0.004 {
                ui.ctx().request_repaint();
            } else {
                self.legend_reveal = target;
            }
            legend_strip = Some(strip);
        }

        // --- wires (drawn under everything) ---
        let wire = |painter: &egui::Painter, from: Pos2, to: Pos2, stroke: Stroke| {
            painter.add(plain_wire_path(from, to, zoom, ortho).shape(stroke));
        };
        // The edge loop's variant: tangents flow THROUGH dots.
        let wire_edge = |painter: &egui::Painter,
                         from: Pos2,
                         to: Pos2,
                         ts: Option<Vec2>,
                         td: Option<Vec2>,
                         side: Option<f32>,
                         stroke: Stroke| {
            painter.add(wire_path(from, to, zoom, ts, td, side, ortho).shape(stroke));
        };
        // A horizontal cable: expression links run right-to-left across the
        // graph, so their control points lean sideways, not down.
        let cable = |painter: &egui::Painter, from: Pos2, to: Pos2, stroke: Stroke| {
            painter.add(cable_path(from, to, zoom, 1.0, ortho).shape(stroke));
        };
        // Where a channel nodule sits on screen, for cables and drawing.
        let chan_pos = |id: u64, ch: crate::anim::Channel, out: bool| -> Option<Pos2> {
            let n = graph.nodes.get(&id)?;
            let chans = crate::patchbay::node_ports(&n.kind, expand_rgb);
            // A lane hidden inside a collapsed triple rides its port.
            let ch = crate::patchbay::port_channel(&n.kind, ch, expand_rgb);
            let i = chans.iter().position(|c| *c == ch)?;
            let rest = chan_rest(dp(id), ds(id), i, chans.len(), out);
            Some(view.to_screen(chan_tips.get(&(id, ch, out)).copied().unwrap_or(rest)))
        };
        // Cost + licence taint, computed by the app this frame. Both the
        // wires below and the node rings further down read it.
        let flow = self.cost_flow.clone();
        // The range view's tags, gathered with the wires and drawn after
        // the nodes so a name or a body never sits on top of one.
        let mut range_tags: Vec<(Pos2, u64)> = Vec::new();
        self.shown_range_tags.clear();
        // The loop pairing is a real dataflow relation with no edge to show
        // it: draw a dotted arc from each LoopOut back up to its LoopIn, in
        // the loop colour, so the cycle reads at a glance.
        if !self.link_mode {
            for (id, n) in &graph.nodes {
                let NodeKind::LoopOut(p) = &n.kind else { continue };
                let Some(li) = graph.node(p.pair) else { continue };
                let a = view.to_screen(dp(*id));
                let b = view.to_screen(li.pos);
                let col = theme::node_loop();
                let bulge = 60.0 * view.zoom;
                let mid = Pos2::new(a.x.min(b.x) - bulge, (a.y + b.y) * 0.5);
                // Spaced dots: a quadratic bowing out to the left, or in
                // right angles a square loop round that side.
                let dots: Vec<Pos2> = if ortho {
                    let corners = [a, Pos2::new(mid.x, a.y), Pos2::new(mid.x, b.y), b];
                    let st = crate::ortho::Style::at(view.zoom);
                    crate::ortho::resample(&crate::ortho::round(&corners, st.radius), 23)
                } else {
                    (0..24)
                        .map(|k| {
                            let t = k as f32 / 23.0;
                            Pos2::new(
                                (1.0 - t) * (1.0 - t) * a.x
                                    + 2.0 * (1.0 - t) * t * mid.x
                                    + t * t * b.x,
                                (1.0 - t) * (1.0 - t) * a.y
                                    + 2.0 * (1.0 - t) * t * mid.y
                                    + t * t * b.y,
                            )
                        })
                        .collect()
                };
                for q in dots {
                    painter.circle_filled(q, 1.6 * view.zoom.clamp(0.5, 1.5), col);
                }
            }
        }
        let mut geo_memo = std::collections::HashMap::<u64, GeoWire>::new();
        for (dst, input, src) in &graph.edges {
            if self.link_mode {
                break; // the image wiring steps aside for the cables
            }
            // The hidden layer only shows itself in patch mode; there, the
            // ordinary wiring steps BACK — grey and locked — so the amber
            // wires you are editing are the loudest thing on the canvas.
            let hidden = graph.is_hidden(*dst, *input);
            if hidden && !self.patch_mode {
                continue;
            }
            if harp_kids.contains(dst) {
                continue; // a harp kid's feed is drawn nowhere
            }
            let (Some(nd), Some(_ns)) = (graph.node(*dst), graph.node(*src)) else { continue };
            let ok = graph.out_of(*dst, *input) as usize;
            let from = view.to_screen(tip(
                &tips,
                *src,
                out_stalk(ok),
                output_rest_of(&graph.nodes, *src, ok, dz(*src)),
            ));
            let to = view.to_screen(tip(
                &tips,
                *dst,
                *input,
                rest_of(graph, &mask_dirs, *dst, *input, dp(*dst), ds(*dst)),
            ));
            let side = side_of(graph, *dst, *input);
            // Cull a wire that cannot cross the panel: a curve lies inside
            // its control points, which sit within about half the span of
            // the ends (more round a dot's tangent, hence the slack).
            let slack = (to - from).length() * 0.6 + 120.0 * zoom + LIT_WIRE_HALF * 2.0;
            if !Rect::from_two_pos(from, to).expand(slack).intersects(rect) {
                continue;
            }
            // Cyan while everything upstream fuses, RED once anything in
            // this wire's history takes a pass of its own — so the cost of a
            // comp is legible from the wires, without opening a panel. A
            // loaded Bake turns it cyan again, which is the whole point.
            let stroke = if self.insert_target == Some((*dst, *input, *src)) {
                Stroke::new(4.0, theme::selection()) // splice preview
            } else if hidden {
                Stroke::new(2.0 * zoom.min(1.5), theme::hidden_wire())
            } else if self.patch_mode {
                Stroke::new(2.0 * zoom.min(1.5), theme::wire_locked())
            } else {
                // Inside a 2dini a noodle wears what it carries.
                let own = if self.in_geo {
                    Some(geo_wire_class(graph, *src, &mut geo_memo).color())
                } else if self.in_noise {
                    Some(field_wire_class(graph, *src, &mut geo_memo).color())
                } else {
                    input_pipe_color(&nd.kind, *input)
                };
                let c = flow.wire_color(
                    *src,
                    own.is_some(),
                    own.unwrap_or_default().gamma_multiply(0.8),
                );
                Stroke::new(2.0 * zoom.min(1.5), c)
            };
            // The FAT CABLE: a multi-channel wire draws as three strands, so
            // "this carries everything, lazily" reads at a glance.
            let channels = nd.kind.input_kind(*input) == PortKind::Channels
                && !hidden
                && !self.patch_mode;
            if channels {
                let c = theme::wire_channels();
                let w = 1.4 * zoom.min(1.5);
                let route =
                    ortho.then(|| wire_path(from, to, zoom, tan(*src), tan(*dst), side, true));
                for k in [-1.0f32, 0.0, 1.0] {
                    let gap = k * 2.4 * zoom.min(1.5);
                    match &route {
                        // In right angles each strand runs parallel through
                        // the corners, not merely shifted sideways.
                        Some(WirePath::Ortho(p)) => {
                            painter.add(egui::Shape::line(
                                crate::ortho::offset(p, gap),
                                Stroke::new(w, c),
                            ));
                        }
                        _ => {
                            let off = Vec2::new(gap, 0.0);
                            wire_edge(
                                &painter,
                                from + off,
                                to + off,
                                tan(*src),
                                tan(*dst),
                                side,
                                Stroke::new(w, c),
                            );
                        }
                    }
                }
            } else {
                // The layer picked in a Layer panel: a band in the timeline's
                // selected-bar colour under the noodle, so the two read as
                // one selection.
                if !hidden && !self.patch_mode && self.lit_inputs.contains(&(*dst, *input)) {
                    let z = zoom.min(1.5);
                    let path =
                        wire_path(from, to, zoom, tan(*src), tan(*dst), side, ortho).points(26);
                    painter.add(egui::Shape::line(
                        path,
                        Stroke::new(7.0 * z, theme::layer_bar_selected().gamma_multiply(0.55)),
                    ));
                }
                // A wire below a REALLY slow node is LIT: additive light
                // either side of the noodle, falling off smoothly and
                // breathing with the node's halo, so the chain that
                // dominates the frame glows the whole way down.
                let plain =
                    !hidden && !self.patch_mode && self.insert_target != Some((*dst, *input, *src));
                if let Some(g) = plain.then(|| flow.wire_glow(*src)).flatten() {
                    let t = ui.input(|i| i.time) as f32;
                    let breathe = 0.82 + 0.18 * (t * 1.5).sin();
                    let z = zoom.min(1.5);
                    let path =
                        wire_path(from, to, zoom, tan(*src), tan(*dst), side, ortho).points(26);
                    glow_ribbon(&painter, &path, LIT_WIRE_HALF * z, glow_light(g, 0.20 * breathe));
                    glow_ribbon(&painter, &path, 7.5 * z, glow_light(g, 0.42 * breathe));
                    glow_ribbon(&painter, &path, 2.8 * z, glow_light(g, 0.68 * breathe));
                }
                match plain.then(|| Reduced::of(graph, *src)).flatten() {
                    None => wire_edge(&painter, from, to, tan(*src), tan(*dst), side, stroke),
                    Some(r) => {
                        let z = zoom.min(1.5);
                        let path = wire_path(from, to, zoom, tan(*src), tan(*dst), side, ortho);
                        let width = if r.single.is_on() { stroke.width * 0.5 } else { stroke.width };
                        let s = Stroke::new(width, stroke.color);
                        if r.eight {
                            painter.extend(egui::Shape::dashed_line(
                                &path.points(48),
                                s,
                                5.0 * z,
                                3.5 * z,
                            ));
                        } else {
                            painter.add(path.shape(s));
                        }
                        // Tagged where the format is SET — the wire leaving
                        // the Load, or a node that keeps less than it was
                        // fed — 40% of the way along it (by length, so a
                        // straight boring wire places it like a curve does).
                        let fed = graph.input_of(*src, 0).and_then(|s| Reduced::of(graph, s));
                        let direct = fed.map(|f| (f.single, f.eight)) != Some((r.single, r.eight));
                        if zoom > 0.5 && direct {
                            let pts = path.points(24);
                            let lens: Vec<f32> =
                                pts.windows(2).map(|w| (w[1] - w[0]).length()).collect();
                            let mut left = lens.iter().sum::<f32>() * 0.4;
                            let mut at = pts[0];
                            for (w, l) in pts.windows(2).zip(&lens) {
                                if left <= *l {
                                    at = w[0] + (w[1] - w[0]) * (left / l.max(1e-3));
                                    break;
                                }
                                left -= l;
                            }
                            let font = egui::FontId::proportional((9.5 * z).max(8.0));
                            let g = painter.layout_no_wrap(r.tag(), font, stroke.color);
                            let rr = Rect::from_center_size(at, g.size() + Vec2::new(6.0, 1.0));
                            painter.rect_filled(rr, 3.0, theme::graph_bg());
                            painter.galley(rr.min + Vec2::new(3.0, 0.5), g, stroke.color);
                        }
                    }
                }
            }
            if self.show_ranges && !hidden && !self.patch_mode {
                let mid = wire_path(from, to, zoom, tan(*src), tan(*dst), side, ortho).mid();
                range_tags.push((mid, *src));
            }
            // A 2dini noodle that says something about its stacking wears
            // it: its z, and its blend when it is not plain over.
            if self.in_geo {
                let l = graph.wire_layering(*dst, *input);
                if l != engine::geo::Layering::default() {
                    let mid = wire_path(from, to, zoom, tan(*src), tan(*dst), side, ortho).mid();
                    let mut t = format!("z {}", l.z);
                    if l.blend != engine::geo::BlendMode::Over {
                        t.push_str(&format!(" · {}", l.blend.name()));
                    }
                    let font = egui::FontId::proportional((10.0 * zoom.min(1.5)).max(8.0));
                    let g = painter.layout_no_wrap(t, font, theme::text_strong());
                    let r = Rect::from_center_size(mid, g.size() + Vec2::new(8.0, 2.0));
                    painter.rect_filled(r, 3.0, theme::graph_bg());
                    painter.galley(r.min + Vec2::new(4.0, 1.0), g, theme::text_strong());
                }
            }
        }
        // The picked layer's socket wears a ring, wired or not — an empty
        // layer has no noodle to light.
        if !self.link_mode && !self.patch_mode {
            let z = zoom.min(1.5);
            for (id, i) in &self.lit_inputs {
                let Some(n) = graph.node(*id) else { continue };
                if *i >= n.kind.input_count() {
                    continue;
                }
                let at = view.to_screen(tip(
                    &tips,
                    *id,
                    *i,
                    rest_of(graph, &mask_dirs, *id, *i, dp(*id), ds(*id)),
                ));
                painter.circle_stroke(
                    at,
                    (SOCKET_R * zoom * ds(*id)).max(2.0) + 4.0 * z,
                    Stroke::new(2.0 * z.max(0.7), theme::layer_bar_selected()),
                );
            }
        }
        // The anti-stamp contract: a hidden feed is never a secret. Its
        // node wears an amber bead by the fed input, HOVERING the node
        // draws the ghost noodle back to the origin, and a source CHIP (a
        // Dot fed hidden) names its origin outright — "crypto" at a glance.
        if !self.link_mode {
            for (dst, input, src) in &graph.edges {
                if !graph.is_hidden(*dst, *input) {
                    continue;
                }
                let (Some(_), Some(ns)) = (graph.node(*dst), graph.node(*src)) else {
                    continue;
                };
                let to = view.to_screen(tip(
                    &tips,
                    *dst,
                    *input,
                    rest_of(graph, &mask_dirs, *dst, *input, dp(*dst), ds(*dst)),
                ));
                let z = zoom.min(1.5);
                // Selecting either end lights the OTHER: select a source and
                // every chip (portal) it feeds is ringed; select a chip and
                // its origin is. Without this a portal is the one wire in
                // the graph a selection says nothing about.
                let sel_src = selection.contains(src);
                let sel_dst = selection.contains(dst);
                if !self.patch_mode && (sel_src || sel_dst) {
                    let ring = |id: u64| {
                        let rr = if graph.is_dot(id) {
                            9.0 * z
                        } else {
                            R * zoom * ds(id) + 5.0 * z
                        };
                        painter.circle_stroke(
                            view.to_screen(dp(id)),
                            rr,
                            Stroke::new(2.2 * z.max(0.7), theme::hidden_wire()),
                        );
                    };
                    if sel_src && !sel_dst {
                        ring(*dst);
                    }
                    if sel_dst && !sel_src {
                        ring(*src);
                    }
                }
                if !self.patch_mode {
                    painter.circle_filled(to, 3.2 * z, theme::hidden_wire());
                    if hit_node == Some(*dst) || sel_src || sel_dst {
                        // The ghost: dashed, in the wire's own path, so it
                        // reads as "this noodle exists, just parked".
                        let ok = graph.out_of(*dst, *input) as usize;
                        let from = view.to_screen(tip(
                            &tips,
                            *src,
                            out_stalk(ok),
                            output_rest_of(&graph.nodes, *src, ok, dz(*src)),
                        ));
                        let side = side_of(graph, *dst, *input);
                        let pts = wire_path(from, to, zoom, tan(*src), tan(*dst), side, ortho)
                            .points(24);
                        painter.extend(egui::Shape::dashed_line(
                            &pts,
                            Stroke::new(1.6 * z, theme::hidden_wire().gamma_multiply(0.9)),
                            6.0 * z,
                            5.0 * z,
                        ));
                    }
                }
                if graph.is_dot(*dst) && self.zoom > 0.25 {
                    let name = node_label(&ns.kind);
                    painter.text(
                        view.to_screen(dp(*dst)) + Vec2::new(10.0 * z, 0.0),
                        egui::Align2::LEFT_CENTER,
                        name,
                        egui::FontId::proportional((11.0 * self.zoom.clamp(0.6, 1.2)).max(11.0)),
                        theme::hidden_wire(),
                    );
                }
            }
        }
        // Horseshoe gain arcs on a Blend's two image inputs: the recombine
        // weight, draggable right where the wire lands. Drawn and hit AFTER
        // the nodes (egui gives presses to the LAST claimant), with the
        // nodule's core left unclaimed so wire grabs still win there.
        self.gain_edits.clear();
        if !self.link_mode && zoom > 0.45 {
            for (dst, input, src) in &graph.edges {
                let Some(nd) = graph.node(*dst) else { continue };
                // Which inputs wear an arc, and their current weight — the
                // recombine surfaces: a Blend's two, a harp's n strings.
                let value = match &nd.kind {
                    NodeKind::Blend { gain, .. } if *input <= 1 => gain[*input],
                    NodeKind::Recombine(p)
                        if (*input as u32) < p.n.clamp(1, 16) =>
                    {
                        p.gain.get(*input).copied().unwrap_or(1.0)
                    }
                    _ => continue,
                };
                let _ = graph.node(*src); // wired end exists or skip drawing
                let to = view.to_screen(tip(
                    &tips,
                    *dst,
                    *input,
                    rest_of(graph, &mask_dirs, *dst, *input, dp(*dst), ds(*dst)),
                ));
                let r_in = 6.5 * zoom;
                let r_arc = 11.0 * zoom;
                let band = Rect::from_center_size(to, Vec2::splat(r_arc * 2.6));
                let id = ui.id().with(("gain-arc", *dst, *input));
                // Claim only the band, never the nodule core.
                let in_band = ui
                    .input(|i| i.pointer.interact_pos())
                    .map(|p| {
                        let d = (p - to).length();
                        d >= r_in && d <= r_arc * 1.55
                    })
                    .unwrap_or(false);
                let dragging = ui
                    .ctx()
                    .drag_started_id()
                    .map(|d| d == id)
                    .unwrap_or(false)
                    || ui.ctx().is_being_dragged(id);
                let resp = if in_band || dragging {
                    Some(ui.interact(band, id, egui::Sense::drag()))
                } else {
                    None
                };
                let mut v = value;
                let mut edited = false;
                if let Some(resp) = &resp {
                    if resp.dragged() {
                        let fine = ui.input(|i| i.modifiers.alt);
                        let rate = if fine { 0.0015 } else { 0.008 };
                        v = (v - resp.drag_delta().y * rate).clamp(0.0, 2.0);
                        edited = v != value;
                    }
                    if resp.double_clicked() {
                        v = 1.0;
                        edited = true;
                    }
                }
                if edited {
                    self.gain_edits.push((*dst, *input, v));
                }
                // Draw: a horseshoe opening toward the stalk (upward), the
                // filled sweep = value/2, a tick at 1.0.
                use std::f32::consts::PI;
                let (a0, a1) = (0.75 * PI, 2.25 * PI); // 270° sweep, gap up
                let arc = |from: f32, t: f32, w: f32, c: Color32| {
                    let steps = 20;
                    let pts: Vec<Pos2> = (0..=steps)
                        .map(|k| {
                            let a = from + (a1 - a0) * t * (k as f32 / steps as f32);
                            Pos2::new(to.x + a.cos() * r_arc, to.y + a.sin() * r_arc)
                        })
                        .collect();
                    painter.add(egui::Shape::line(pts, Stroke::new(w, c)));
                };
                let dim = Color32::from_gray(0x4a);
                let hot = if resp.is_some() {
                    Color32::from_rgb(0xe8, 0xb4, 0x4a)
                } else {
                    Color32::from_rgb(0xb0, 0x8a, 0x3a)
                };
                arc(a0, 1.0, 1.4 * zoom, dim);
                arc(a0, (v * 0.5).clamp(0.0, 1.0), 2.6 * zoom, hot);
                // Tick at the neutral 1.0.
                let am = a0 + (a1 - a0) * 0.5;
                painter.line_segment(
                    [
                        Pos2::new(to.x + am.cos() * (r_arc - 3.0 * zoom), to.y + am.sin() * (r_arc - 3.0 * zoom)),
                        Pos2::new(to.x + am.cos() * (r_arc + 3.0 * zoom), to.y + am.sin() * (r_arc + 3.0 * zoom)),
                    ],
                    Stroke::new(1.0 * zoom, dim),
                );
                if resp.as_ref().map(|r| r.hovered() || r.dragged()).unwrap_or(false) {
                    painter.text(
                        to + Vec2::new(0.0, r_arc + 9.0 * zoom),
                        egui::Align2::CENTER_CENTER,
                        format!("{v:.2}"),
                        egui::FontId::proportional(10.0 * zoom.clamp(0.8, 1.4)),
                        hot,
                    );
                }
            }
        }
        // Harp string chips, IN the bar at each shuffle point: click picks
        // the string's layer, a horizontal drag GRABS the string and moves
        // it — per-string spacing, mirrored onto the receiver so the
        // strings keep hanging straight. The drag interact registers first
        // (it takes drags), the combo after (later wins the clicks).
        self.shuffle_edits.clear();
        self.offset_edits.clear();
        let mut bar_lifts: Vec<(u64, f32)> = Vec::new();
        if !self.link_mode && zoom > 0.45 {
            // The slab paints AFTER this pass (nodes go over wires), so the
            // glyphs go on a foreground layer or the bar buries them.
            let chip_paint = ui
                .ctx()
                .layer_painter(egui::LayerId::new(
                    egui::Order::Foreground,
                    ui.id().with("harp-chips"),
                ))
                .with_clip_rect(ui.clip_rect());
            let bars: Vec<u64> = graph
                .nodes
                .iter()
                .filter_map(|(id, n)| match &n.kind {
                    NodeKind::Breakout(_) => Some(*id),
                    _ => None,
                })
                .collect();
            for bar in bars {
                let (bpos, bsize, bkind) = {
                    let n = graph.node(bar).unwrap();
                    (n.pos, n.size, n.kind.clone())
                };
                let NodeKind::Breakout(bp) = &bkind else { continue };
                let n_layers = bp.layers.len();
                for k in 0..n_layers {
                    let Some(x) = crate::breakout::string_x(&bkind, bpos, k, bsize)
                    else {
                        continue;
                    };
                    let pos = view.to_screen(Pos2::new(x, bpos.y));
                    let chip = Rect::from_center_size(
                        pos,
                        Vec2::new(20.0 * zoom, 16.0 * zoom),
                    );
                    // The grab point: horizontal drag nudges THIS string.
                    // Keyed by the string's own KID, not by its index: a
                    // drag that reorders the bar under itself would
                    // otherwise change the widget's id mid-gesture and
                    // egui would drop the drag the moment it passed a
                    // neighbour.
                    let ident = bp.kids.get(k).copied().unwrap_or(k as u64);
                    let grip_id = ui.id().with(("harp-grip", bar, ident));
                    let grip = ui.interact(chip, grip_id, egui::Sense::click_and_drag());
                    // The chip's drag picks ONE axis from its first few
                    // pixels: across nudges this string, up or down moves
                    // the whole bar and every node spliced into its
                    // strings (a body drag does the same). Locked so a
                    // sideways nudge never wobbles the bar's height.
                    let axis_id = grip_id.with("axis");
                    let d = grip.drag_delta();
                    let axis = if grip.dragged() {
                        let (mut acc, mut lock): (Vec2, Option<bool>) = ui
                            .ctx()
                            .data(|m| m.get_temp(axis_id))
                            .filter(|_| !grip.drag_started())
                            .unwrap_or((Vec2::ZERO, None));
                        acc += d;
                        if lock.is_none() && acc.length() > 4.0 {
                            lock = Some(acc.y.abs() > acc.x.abs());
                        }
                        ui.ctx().data_mut(|m| m.insert_temp(axis_id, (acc, lock)));
                        lock
                    } else {
                        None
                    };
                    let dx = if axis == Some(false) { d.x } else { 0.0 };
                    if axis == Some(true) && d.y != 0.0 {
                        bar_lifts.push((bar, d.y / view.zoom));
                    }
                    if grip.dragged() && dx != 0.0 {
                        // Applied by the app to the bar and its pair, so the
                        // string stays vertical.
                        let dg = dx / view.zoom / bsize.max(0.05);
                        self.offset_edits.push((bar, k, dg));
                    }
                    if grip.double_clicked() {
                        self.offset_edits.push((bar, k, f32::NAN));
                    }
                    // ALWAYS-VISIBLE affordance: grip bars (drag me) and a
                    // caret (pick me), brightening under the pointer.
                    let hot = grip.hovered() || grip.dragged();
                    let ink = if hot {
                        Color32::from_gray(0xe8)
                    } else {
                        Color32::from_gray(0x8a)
                    };
                    for gx in [-1.6f32, 1.6] {
                        chip_paint.line_segment(
                            [
                                pos + Vec2::new((gx - 4.0) * zoom, -4.0 * zoom),
                                pos + Vec2::new((gx - 4.0) * zoom, 4.0 * zoom),
                            ],
                            Stroke::new(1.2 * zoom, ink),
                        );
                    }
                    let c = pos + Vec2::new(5.0 * zoom, 0.0);
                    chip_paint.add(egui::Shape::convex_polygon(
                        vec![
                            c + Vec2::new(-3.0 * zoom, -2.0 * zoom),
                            c + Vec2::new(3.0 * zoom, -2.0 * zoom),
                            c + Vec2::new(0.0, 2.5 * zoom),
                        ],
                        ink,
                        Stroke::NONE,
                    ));
                    if hot {
                        chip_paint.rect_stroke(
                            chip.expand(1.0),
                            2.0,
                            Stroke::new(1.0, Color32::from_gray(0x9a)),
                            egui::StrokeKind::Outside,
                        );
                    }
                    // Click opens the layer menu (a real popup, so it can
                    // be SEEN — the old zero-width combo could not).
                    let popup_id = ui.id().with(("harp-menu", bar, k));
                    let Some(list) =
                        self.harp_layers.get(&bar).filter(|l| !l.is_empty()).cloned()
                    else {
                        continue;
                    };
                    let current = bp.layers.get(k).cloned().unwrap_or_default();
                    egui::Popup::menu(&grip)
                        .id(popup_id)
                        .close_behavior(egui::PopupCloseBehavior::CloseOnClick)
                        .show(|ui| {
                            ui.set_min_width(160.0);
                            for (name, label) in &list {
                                let sel = *name == current;
                                if ui.selectable_label(sel, label.as_str()).clicked()
                                    && !sel
                                {
                                    self.shuffle_edits.push((bar, k, name.clone()));
                                }
                            }
                        });
                }
            }
        }
        // A knob on the tool being edited has no node to run from, so its
        // cable arrives as a short stub at the channel it drives — enough to
        // see that the channel IS driven, and by what kind of thing.
        if self.link_mode {
            for l in &graph.patch.links {
                let (PSrc::Param(_), PDst::Chan(dn, dc)) = (l.from, l.to) else {
                    continue;
                };
                let Some(b) = chan_pos(dn, dc, false) else { continue };
                let a = b - Vec2::new(26.0 * zoom, 0.0);
                let w = 2.0 * zoom.min(1.5);
                painter.line_segment([a, b], Stroke::new(w, theme::cable_source()));
                painter.circle_filled(a, 3.0 * zoom.min(1.5), theme::cable_source());
            }
        }
        // Expression cables, drawn where the image wires would be. Lanes of
        // a collapsed triple share a pair of ports, so they share a cable —
        // one strand each, in the patch canvas's lane tints.
        if self.link_mode {
            // (ends, lanes a collapsed triple folded in, each end's own lane)
            type Bundle = (Pos2, Pos2, [bool; 3], Option<usize>, Option<usize>);
            let mut bundles: Vec<Bundle> = Vec::new();
            for l in &graph.patch.links {
                let (PSrc::Chan(sn, sc), PDst::Chan(dn, dc)) = (l.from, l.to) else {
                    continue; // one end is a value node: it lives on the rack
                };
                let (Some(a), Some(b)) =
                    (chan_pos(sn, sc, true), chan_pos(dn, dc, false))
                else {
                    continue;
                };
                let lane = graph.node(dn).and_then(|n| {
                    let l = crate::patchbay::lanes_for(&n.kind, dc, expand_rgb)?;
                    l.iter().position(|c| *c == dc)
                });
                // Each end's lane, whatever the collapse state.
                let end = |n: u64, c: crate::anim::Channel| {
                    graph.node(n).and_then(|nd| crate::patchbay::lane_index(&nd.kind, c))
                };
                match bundles.iter_mut().find(|(f, t, ..)| *f == a && *t == b) {
                    Some((_, _, lanes, ..)) => {
                        if let Some(i) = lane {
                            lanes[i] = true;
                        }
                    }
                    None => {
                        let mut lanes = [false; 3];
                        if let Some(i) = lane {
                            lanes[i] = true;
                        }
                        bundles.push((a, b, lanes, end(sn, sc), end(dn, dc)));
                    }
                }
            }
            let w = 2.0 * zoom.min(1.5);
            for (a, b, lanes, sl, dl) in bundles {
                // A collapsed triple ramps r -> g -> b, so the colour is the
                // arity. Anything else ramps from what it leaves to what it
                // lands on: an R output into a G input runs red to green.
                let cols = if lanes.iter().filter(|x| **x).count() > 1 {
                    crate::patchbay::lane_ramp(lanes, 1.0)
                } else {
                    crate::patchbay::flow_ramp(sl, dl, 1.0)
                };
                let pieces = cable_path(a, b, zoom, 1.0, ortho).pieces(28);
                for sh in crate::patchbay::gradient_pieces(&pieces, &cols, w, false) {
                    painter.add(sh);
                }
            }
        }
        // Courting pairs: a faint tether between sockets that are reaching
        // for each other, brightening as they close. Shift wires it.
        for (did, i, sid, k, w, ..) in &pairs {
            let (Some(dn), Some(_)) = (graph.node(*did), graph.node(*sid)) else {
                continue;
            };
            let a = view.to_screen(tip(
                &tips,
                *sid,
                out_stalk(*k),
                output_rest_of(&graph.nodes, *sid, *k, dz(*sid)),
            ));
            let b = view.to_screen(tip(
                &tips,
                *did,
                *i,
                rest_of(graph, &mask_dirs, *did, *i, dp(*did), ds(*did)),
            ));
            let col = input_pipe_color(&dn.kind, *i).unwrap_or_else(theme::port_in);
            let stroke = Stroke::new(1.0 + 1.2 * w, col.gamma_multiply((0.25 + 0.55 * w) * 0.9));
            let n_seg = 7;
            if ortho {
                // The right-angle wire it would become, dashed.
                let side = side_of(graph, *did, *i);
                let path = wire_path(a, b, zoom, tan(*sid), tan(*did), side, true);
                for piece in path.pieces(n_seg).into_iter().step_by(2) {
                    painter.add(egui::Shape::line(piece, stroke));
                }
            } else {
                for k in 0..n_seg {
                    if k % 2 == 1 {
                        continue; // dashed
                    }
                    let t0 = k as f32 / n_seg as f32;
                    let t1 = (k as f32 + 1.0) / n_seg as f32;
                    painter.line_segment([a + (b - a) * t0, a + (b - a) * t1], stroke);
                }
            }
        }
        // --- expression patch overlay (read-only) ---
        // A SUMMARY for the normal graph: one dashed line per node pair, so
        // you can see what is expression-linked without leaving the image
        // wiring. Value flow is LEFT→RIGHT on the patch canvas and stays
        // that way here — these leave and enter a node's SIDES with
        // horizontal tangents, against the image wires' top-to-bottom — and
        // they are dashed because nothing here edits them.
        //
        // Link mode draws the real cables, channel by channel. Summarising
        // them on top of themselves would just be noise, so this stands down.
        if self.show_patch_links && !self.link_mode {
            for e in crate::patchbay::node_to_node(graph, self.expand_patch_rgb) {
                if graph.node(e.src).is_none() || graph.node(e.dst).is_none() {
                    continue;
                }
                let (a, b) = (dp(e.src), dp(e.dst));
                // Leave by the side that faces the destination and arrive by
                // the side facing back, so a link never crosses its own nodes.
                let sgn = if b.x >= a.x { 1.0 } else { -1.0 };
                let from = view.to_screen(a + Vec2::new(sgn * R * ds(e.src), 0.0));
                let to = view.to_screen(b - Vec2::new(sgn * R * ds(e.dst), 0.0));
                let c = if ortho {
                    use crate::ortho::{route, End, Style};
                    WirePath::Ortho(route(from, End::Side(sgn), to, End::Side(sgn), &Style::at(zoom)))
                } else {
                    WirePath::Curve(crate::patchbay::cable_ctrl(from, to))
                };
                let w = (1.4 * zoom.min(1.5)).max(0.7);
                // Dash by sampling the curve: a straight dash would cut the
                // corners of a strongly bowed link. A collapsed rgb port
                // draws a strand per lane, the same vocabulary the patch
                // canvas uses, so the two views tell the same story.
                let tip = if e.is_rgb() {
                    // Same r -> g -> b ramp the live cables use, dashed
                    // because this view does not edit them.
                    let cols = crate::patchbay::lane_ramp(e.lanes, 0.85);
                    for sh in crate::patchbay::gradient_pieces(&c.pieces(28), &cols, w, true) {
                        painter.add(sh);
                    }
                    e.lanes
                        .iter()
                        .rposition(|on| *on)
                        .map(|k| crate::patchbay::LANE_COLOR[k])
                        .unwrap_or(theme::cable())
                } else {
                    let col = theme::cable().gamma_multiply(0.85);
                    for piece in c.pieces(24).into_iter().step_by(2) {
                        painter.add(egui::Shape::line(piece, Stroke::new(w, col)));
                    }
                    col
                };
                painter.circle_filled(to, 2.2 * zoom.min(1.5), tip);
                // The badge counts PORTS, so an rgb triple reads as one.
                if e.ports > 1 && zoom > 0.6 {
                    painter.text(
                        c.mid() - Vec2::new(0.0, 3.0),
                        egui::Align2::CENTER_BOTTOM,
                        format!("{}", e.ports),
                        egui::FontId::proportional(9.0 * zoom.min(1.5)),
                        theme::cable(),
                    );
                }
            }
        }

        // Align swipe: the stroke, plus the line the nodes will land on —
        // the axis is decided by the direction, so it has to be visible
        // BEFORE you let go.
        if let Some(DragKind::AlignSwipe { start, path }) = &self.drag {
            let end = path.last().copied().or(pointer).unwrap_or(*start);
            painter.add(egui::Shape::line(
                path.clone(),
                Stroke::new(1.5, theme::align_guide().gamma_multiply(0.7)),
            ));
            if (end - *start).length() > 8.0 {
                let horiz =
                    crate::nodeux::Axis::of_swipe(*start, end) == crate::nodeux::Axis::Horizontal;
                let (a, b) = if horiz {
                    let y = (start.y + end.y) * 0.5;
                    (Pos2::new(rect.min.x, y), Pos2::new(rect.max.x, y))
                } else {
                    let x = (start.x + end.x) * 0.5;
                    (Pos2::new(x, rect.min.y), Pos2::new(x, rect.max.y))
                };
                painter.line_segment(
                    [a, b],
                    Stroke::new(1.0, theme::align_guide().gamma_multiply(0.45)),
                );
            }
        }
        // Teach the align swipe where it would help: dragging several
        // nodes into line by hand, or holding the key with no stroke yet.
        // One line along the bottom of the graph, gone when the drag ends.
        let align_hint = match &self.drag {
            Some(DragKind::Node { .. }) if selection.len() >= 2 => Some(format!(
                "tip: lining them up? hold {} and swipe across the selection",
                crate::keymap::label(ui.ctx(), crate::keymap::Act::GraphAlign)
            )),
            Some(DragKind::AlignSwipe { .. }) => Some(
                "swipe across the selection: sideways makes a row, up or down a column".into(),
            ),
            _ => None,
        };
        if let Some(text) = align_hint {
            let font = egui::FontId::proportional(13.0);
            let at = Pos2::new(rect.center().x, rect.max.y - 18.0);
            let galley = painter.layout_no_wrap(text, font, theme::text_strong());
            let bg = Rect::from_center_size(at, galley.size() + Vec2::new(16.0, 8.0));
            painter.rect_filled(bg, 4.0, theme::graph_bg().gamma_multiply(0.85));
            painter.galley(bg.center() - galley.size() * 0.5, galley, theme::text_strong());
        }
        // Slicer trail.
        if let Some(DragKind::Slice { path, .. }) = &self.drag {
            painter.add(egui::Shape::line(
                path.clone(),
                Stroke::new(1.5, theme::cut()),
            ));
        }
        // Pending wire follows the pointer.
        if let (Some(drag), Some(p)) = (&self.drag, pointer) {
            let stroke = Stroke::new(2.0, theme::wire_dragged());
            match drag {
                DragKind::WireFromOutput(src, o) => {
                    if graph.node(*src).is_some() {
                        let ok = *o as usize;
                        let from = view.to_screen(tip(
                            &tips,
                            *src,
                            out_stalk(ok),
                            output_rest_of(&graph.nodes, *src, ok, dz(*src)),
                        ));
                        wire(&painter, from, p, stroke);
                        // The output's disc rides the noodle's tip.
                        let col = pipe_color(graph.nodes[src].kind.output_kind_at(ok))
                            .unwrap_or_else(theme::port_out);
                        painter.circle_filled(p, (SOCKET_R * self.zoom * dz(*src)).max(1.5), col);
                    }
                }
                DragKind::CableFrom(from) => {
                    if let PSrc::Chan(n, c) = from {
                        if let Some(a) = chan_pos(*n, *c, true) {
                            cable(&painter, a, p, Stroke::new(2.0, theme::cable()));
                        }
                    }
                }
                DragKind::CableTo(to) => {
                    if let PDst::Chan(n, c) = to {
                        if let Some(b) = chan_pos(*n, *c, false) {
                            cable(&painter, p, b, Stroke::new(2.0, theme::cable()));
                        }
                    }
                }
                DragKind::WireFromInput(dst, i) => {
                    if let Some(_n) = graph.node(*dst) {
                        let to = view.to_screen(tip(
                            &tips,
                            *dst,
                            *i,
                            rest_of(graph, &mask_dirs, *dst, *i, dp(*dst), ds(*dst)),
                        ));
                        wire(&painter, p, to, stroke);
                    }
                }
                _ => {}
            }
        }
        // A chip dragged up or down lifts the bar (applied here,
        // past the last read of the port-position closure) and its string riders.
        for (bar, dy) in bar_lifts {
            let mut movers = vec![bar];
            movers.extend(crate::breakout::riders(graph, &[bar]));
            for id in movers {
                if let Some(n) = graph.node_mut(id) {
                    n.pos.y += dy;
                }
            }
        }
        // Where a port sits on screen — for the held noodle and the ring
        // that says which port is selected.
        let port_pos = |port: Port| -> Option<Pos2> {
            let id = port.node();
            graph.node(id)?;
            let (np, sz) = (dp(id), dz(id));
            Some(view.to_screen(match port {
                Port::Out(_, o) => {
                    let k = o as usize;
                    tip(&tips, id, out_stalk(k), output_rest_of(&graph.nodes, id, k, sz))
                }
                Port::In(_, i) => tip(
                    &tips,
                    id,
                    i,
                    rest_of(graph, &mask_dirs, id, i, np, ds(id)),
                ),
            }))
        };
        // The selected port: a ring, so it reads as armed rather than lit.
        if let Some(port) = self.sel_port {
            if let Some(at) = port_pos(port) {
                // On a harp bar the ring stays socket-sized: the bar's own
                // scale would otherwise blow it up into a hoop over the rig.
                let wide = graph
                    .node(port.node())
                    .and_then(|n| crate::breakout::wide_half(&n.kind))
                    .is_some();
                let rr = (SOCKET_R * zoom * ds(port.node())).max(3.0);
                let rr = if wide { rr.min(6.0) } else { rr };
                painter.circle_stroke(at, rr + 3.0, Stroke::new(1.6, theme::selection()));
            }
        }
        // A noodle held with G. Unlike a drag, this lives BETWEEN frames:
        // pan, zoom and Tab in a new node with the wire still in hand.
        if let (Some(port), Some(p)) = (self.grab, pointer) {
            let hot = theme::wire_held();
            if let Some(at) = port_pos(port) {
                match port {
                    Port::Out(..) => wire(&painter, at, p, Stroke::new(2.5, hot)),
                    Port::In(..) => wire(&painter, p, at, Stroke::new(2.5, hot)),
                }
                painter.circle_filled(p, 4.5, hot);
                painter.circle_stroke(at, 6.0, Stroke::new(1.5, hot));
            } else {
                self.grab = None; // its node went away under it
            }
        }

        // M held mid-drag, before the swipe lands: say what crossing a node
        // will do, right at the hand.
        if let (Some(DragKind::Node { id: did, merged: false, .. }), Some(p)) =
            (&self.drag, pointer)
        {
            use crate::keymap::{down, Act};
            let matte = graph
                .node(*did)
                .is_some_and(|n| n.kind.output_kind() == PortKind::Mask);
            let say = ui.input(|i| {
                if down(i, Act::GraphRegionMenu) || down(i, Act::GraphSwipeBlend) {
                    Some(("blend: swipe over a node", true))
                } else if down(i, Act::GraphSwipeIn) {
                    Some(if matte {
                        ("mask: swipe over an image to isolate it", true)
                    } else {
                        ("mask needs a matte in hand", false)
                    })
                } else if down(i, Act::GraphSwipeOut) {
                    Some(if matte {
                        ("stencil: swipe over an image to cut it out", true)
                    } else {
                        ("stencil needs a matte in hand", false)
                    })
                } else {
                    None
                }
            });
            if let Some((text, ok)) = say {
                painter.text(
                    p + Vec2::new(14.0, -12.0),
                    egui::Align2::LEFT_BOTTOM,
                    text,
                    egui::FontId::proportional(11.0),
                    if ok { theme::cable() } else { theme::node_label().gamma_multiply(0.6) },
                );
            }
        }
        // The lines a node drag snapped to.
        if matches!(self.drag, Some(DragKind::Node { .. })) {
            for (a, b) in &self.snap_guides {
                let (a, b) = (view.to_screen(*a), view.to_screen(*b));
                let dir = (b - a).normalized();
                let ext = if dir.is_finite() { dir * 12.0 } else { Vec2::ZERO };
                painter.line_segment([a - ext, b + ext], Stroke::new(1.0, theme::cable().gamma_multiply(0.7)));
            }
        } else {
            self.snap_guides.clear();
        }
        // Marquee rectangle while a box-select drag is live.
        if let Some(DragKind::BoxSelect { start }) = &self.drag {
            if let Some(p) = pointer {
                let r = Rect::from_two_pos(*start, p);
                painter.rect_filled(r, 0.0, theme::marquee_fill());
                painter.rect_stroke(
                    r,
                    0.0,
                    Stroke::new(1.0, theme::marquee()),
                    egui::StrokeKind::Outside,
                );
            }
        }

        // A `.` pull: a faint ring where the tension is 1, a ring through the
        // pointer, and the number.
        if let (Some(pull), Some(p)) = (self.dot_place.filter(|d| d.pulling), pointer) {
            if let Some(n) = graph.node(pull.id) {
                let bead = view.to_screen(n.pos);
                painter.circle_stroke(
                    bead,
                    pull.reference,
                    Stroke::new(1.0, theme::marquee().gamma_multiply(0.5)),
                );
                painter.circle_stroke(bead, p.distance(bead), Stroke::new(1.5, theme::handle()));
                painter.text(
                    p + Vec2::new(14.0, -10.0),
                    egui::Align2::LEFT_BOTTOM,
                    format!("tension {:.2}", n.tension),
                    egui::FontId::proportional(12.0),
                    theme::node_label(),
                );
            }
        }

        // --- TVs: pick each TV-mode Read's picture BEFORE the draw loop
        // borrows the graph (the cache is &mut self). Off-screen TVs and
        // specks ask for nothing, so a big comp zoomed out decodes nothing.
        let mut tv_pics: std::collections::HashMap<
            u64,
            (Option<egui::TextureHandle>, Option<(f32, String)>),
        > = std::collections::HashMap::new();
        {
            if let Some(dec) = self.tv_decoder.clone() {
                if self.tv.pump(ui.ctx(), &dec) {
                    ui.ctx().request_repaint_after(web_time::Duration::from_millis(30));
                }
            }
            let now = ui.input(|i| i.time);
            let tv_ids: Vec<u64> = graph
                .nodes
                .iter()
                .filter(|(_, n)| tv_on(n))
                .map(|(id, _)| *id)
                .collect();
            self.tv.retain(|id| tv_ids.contains(&id));
            self.tv_state.clear();
            let mut snowing = false;
            for id in tv_ids {
                let cab = Rect::from_center_size(
                    view.to_screen(dp(id)),
                    tv_half() * 2.0 * self.zoom * ds(id),
                );
                if !cab.intersects(rect) {
                    continue;
                }
                let screen = if boring() { crate::tv::plain_screen(cab) } else { crate::tv::screen_of(cab) };
                let Some(feed) = self.tv_feeds.get(&id) else {
                    snowing = true; // unloaded: static, nothing to ask for
                    continue;
                };
                if screen.width() < 12.0 {
                    continue;
                }
                // Hover-scrub (the browser's hands): only on a screen big
                // enough to aim at, and never from under a pie or a drag.
                let scrub_x = pointer.filter(|p| {
                    resp.hovered()
                        && self.drag.is_none()
                        && screen.width() >= crate::tv::SCRUB_AT
                        && screen.contains(*p)
                });
                let scrub = scrub_x
                    .map(|p| crate::tv::scrub_index(p.x, screen.min.x, screen.max.x, feed.len()));
                let pic = self.tv.picture(id, feed, scrub, screen.width(), now);
                snowing |= pic.tex.is_none();
                self.tv_state.insert(id, (pic.idx, pic.big, pic.exact, screen));
                let overlay = scrub_x.map(|p| (p.x, feed.label(pic.idx)));
                tv_pics.insert(id, (pic.tex, overlay));
            }
            if snowing {
                ui.ctx().request_repaint_after(web_time::Duration::from_millis(40));
            }
        }

        // --- nodes: stalks + sockets, then the shape, then label ---
        self.big_dot_spots.clear();
        // (dst, input) -> src, once: the sockets below ask per input, and
        // `input_of` is a scan of every edge.
        let input_at: std::collections::HashMap<(u64, usize), u64> =
            graph.edges.iter().map(|(d, i, s)| ((*d, *i), *s)).collect();
        // Cull: a node whose body, stalks and name cannot reach the panel
        // is not drawn at all, only its rect recorded. In a big graph
        // zoomed in that is nearly all of them.
        let cull = rect.expand(CULL_PAD);
        for (id, n) in graph.nodes.iter().filter(|(id, n)| !hidden(id, n)) {
            let z = self.zoom;
            let (np, sz) = (dp(*id), ds(*id));
            let sp = view.to_screen(np);
            let node_rect = |r: f32| match wide_of(n) {
                Some((h, dx)) => Rect::from_center_size(
                    sp + Vec2::new(dx * z * sz, 0.0),
                    h * 2.0 * z * sz,
                ),
                None => Rect::from_center_size(sp, Vec2::splat(2.0 * r)),
            };
            if !node_rect(R * z * sz).expand((STALK + R) * 2.0 * z * sz).intersects(cull) {
                node_rects.push((*id, node_rect(R * z * sz)));
                continue;
            }
            let selected = selection.contains(id);
            let stalk_w = (2.0 * z.min(1.5) * sz).max(0.6);

            let stalk = |t: Pos2,
                         stem_dir: Vec2,
                         color: Color32,
                         label: Option<&str>,
                         name: Option<&str>,
                         bead: Bead| {
                let s = view.to_screen(t);
                // A harp's stalks drop straight off the slab at their own
                // string's x — a comb of verticals, not a fan out of the
                // centre. Everything else grows from the body as before.
                let (root, stem) = match crate::breakout::wide_half(&n.kind) {
                    Some(h) => {
                        let d = if stem_dir.y >= 0.0 { 1.0 } else { -1.0 };
                        let gx = view.to_graph(s).x;
                        (
                            view.to_screen(Pos2::new(gx, np.y + h.y * sz * d)),
                            view.to_screen(Pos2::new(
                                gx,
                                np.y + (h.y + STALK * 0.35) * sz * d,
                            )),
                        )
                    }
                    None => (sp, view.to_screen(np + stem_dir * (R * 0.6 * sz))),
                };
                // Boring: the socket sits on the body, no stalk to draw.
                if !boring() {
                    painter.add(egui::epaint::CubicBezierShape::from_points_stroke(
                        [root, stem, Pos2::new(s.x, (s.y + stem.y) * 0.5), s],
                        false,
                        Color32::TRANSPARENT,
                        Stroke::new(stalk_w, color.gamma_multiply(0.7)),
                    ));
                }
                let sr = (SOCKET_R * z * sz).max(1.5);
                match label {
                    // A labelled nodule (a merge's + / −): the sign IS the
                    // socket, ringed so the input silhouette survives.
                    Some(tag) if sr > 3.5 => {
                        let ring = sr * 1.55;
                        painter.circle(
                            s,
                            ring,
                            theme::graph_bg(),
                            Stroke::new((ring * 0.28).clamp(1.0, 2.5), color),
                        );
                        painter.text(
                            s,
                            egui::Align2::CENTER_CENTER,
                            tag,
                            egui::FontId::proportional((ring * 1.35).clamp(7.0, 16.0)),
                            color,
                        );
                    }
                    // An input is a CUP opening away from the node: a port
                    // waiting to receive something, and once fed the disc
                    // that came down the noodle sits in it. Too small to
                    // read as a cup, it stays a dot.
                    _ => match bead {
                        Bead::Cup(seated) if sr > 2.5 => {
                            cup(&painter, s, (s - stem).normalized(), sr, color);
                            if let Some(c) = seated {
                                painter.circle_filled(s, sr * SEATED, c);
                            }
                        }
                        _ => {
                            painter.circle_filled(s, sr, color);
                        }
                    },
                }
                // The socket's NAME, up and to the right at 45 degrees so it
                // clears both the nodule and its stalk. Only once there is
                // room to read it: zoomed out these would be a hedge of
                // unreadable text over the graph.
                if let Some(text) = name {
                    let fs = 11.0 * z * sz;
                    if fs >= 7.5 {
                        // Set ALONG the 45 degree line out of the nodule, so
                        // the label runs away from the node instead of
                        // across its neighbours — with seven sockets fanned
                        // out, horizontal labels overlap each other. On a
                        // harp bar the nodule sits UNDER the slab, so its
                        // name climbs from the slab's TOP edge instead —
                        // otherwise every layer name dies behind the bar.
                        let k = sr * 1.4 + 3.0;
                        let anchor = match crate::breakout::wide_half(&n.kind) {
                            Some(h) => {
                                let gx = view.to_graph(s).x;
                                view.to_screen(Pos2::new(
                                    gx,
                                    np.y - (h.y + 2.0) * sz,
                                )) + Vec2::new(2.0, -3.0)
                            }
                            None => s + Vec2::new(k, -k),
                        };
                        let galley = painter.layout_no_wrap(
                            text.to_owned(),
                            egui::FontId::proportional(fs.min(15.0)),
                            color.gamma_multiply(0.9),
                        );
                        painter.add(
                            egui::epaint::TextShape::new(
                                anchor,
                                galley,
                                color.gamma_multiply(0.9),
                            )
                            .with_angle(-std::f32::consts::FRAC_PI_4),
                        );
                    }
                }
            };
            // Link mode: the image sockets give way to a splay of channel
            // nodules, inputs left and outputs right, on tentacles long
            // enough to tell one from the next.
            if self.link_mode {
                let chans = crate::patchbay::node_ports(&n.kind, expand_rgb);
                let nch = chans.len();
                for (i, ch) in chans.iter().enumerate() {
                    for out in [false, true] {
                        let rest = chan_rest(np, sz, i, nch, out);
                        let at =
                            chan_tips.get(&(*id, *ch, out)).copied().unwrap_or(rest);
                        let s = view.to_screen(at);
                        let stem = view.to_screen(
                            np + Vec2::new(if out { 1.0 } else { -1.0 }, 0.0)
                                * (R * 0.6 * sz),
                        );
                        // Driven inputs and driving outputs read brighter, so
                        // what is actually patched stands out from what could
                        // be.
                        // A collapsed port is live when ANY of its lanes is.
                        let lanes = crate::patchbay::lanes_for(&n.kind, *ch, expand_rgb);
                        let of = |c: crate::anim::Channel| {
                            if out {
                                graph
                                    .patch
                                    .links
                                    .iter()
                                    .any(|l| l.from == PSrc::Chan(*id, c))
                            } else {
                                graph.patch.link_to(PDst::Chan(*id, c)).is_some()
                            }
                        };
                        let live = match lanes {
                            Some(l) => l.iter().any(|c| of(*c)),
                            None => of(*ch),
                        };
                        let hot = self.hover_chan == Some((*id, *ch, out));
                        // Full strength on the BEAD (a half-transparent dot
                        // over a near-black graph just reads as dark); the
                        // tentacle is what fades.
                        let col = if live {
                            theme::cable()
                        } else if out {
                            theme::port_out()
                        } else {
                            theme::port_in()
                        };
                        painter.add(egui::epaint::CubicBezierShape::from_points_stroke(
                            [sp, stem, Pos2::new((s.x + stem.x) * 0.5, s.y), s],
                            false,
                            Color32::TRANSPARENT,
                            Stroke::new(stalk_w * 0.7, col.gamma_multiply(0.4)),
                        ));
                        // The nodule has to read as a BEAD on the end of the
                        // tentacle, not as the line's last pixel: full
                        // strength, a dark rim, and a floor so it survives
                        // being zoomed out.
                        // Solid: the bead sits out on open background, so a
                        // rim only eats the colour that makes it legible.
                        let sr = (SOCKET_R * z * sz).max(2.5);
                        let br = if hot { sr * 1.45 } else { sr };
                        painter.circle_filled(s, br, col);
                        // A collapsed triple wears a ring: one nodule, three
                        // lanes, obvious before you touch it.
                        if lanes.is_some() {
                            painter.circle_stroke(
                                s,
                                br * 1.9,
                                Stroke::new(
                                    (0.9 * z * sz).max(0.5),
                                    col.gamma_multiply(0.75),
                                ),
                            );
                        }
                        // The name rides the nodule when there is room for
                        // it, and always on the one under the pointer.
                        if hot || z * sz > 0.75 {
                            let label =
                                crate::patchbay::port_label(&n.kind, *ch, expand_rgb);
                            painter.text(
                                s + Vec2::new(if out { 8.0 } else { -8.0 }, 0.0),
                                if out {
                                    egui::Align2::LEFT_CENTER
                                } else {
                                    egui::Align2::RIGHT_CENTER
                                },
                                label.as_ref(),
                                egui::FontId::proportional(
                                    (11.0 * (z * sz).clamp(0.7, 1.1)).max(8.0),
                                ),
                                if hot { theme::text_hot() } else { theme::node_label() },
                            );
                        }
                    }
                }
            }
            // A dot's sockets ARE the bead: no stalks, no sockets drawn —
            // the wires already meet at its centre.
            let bare = matches!(n.kind, NodeKind::Dot { .. });
            // A multi-input node wears ONE bar: a single stalk up to it,
            // the wires landing along it — no nodule per input.
            let multi = multi_input(&n.kind);
            if multi && !self.link_mode {
                let n_in = n.kind.input_count();
                let free = n_in.saturating_sub(1);
                let col = input_socket_color(&n.kind, free);
                let c = multi_rest(np, free, n_in, sz);
                stalk(tip(&tips, *id, free, c), Vec2::new(0.0, -1.0), col, None, None, Bead::Dot);
                let s = view.to_screen(c);
                let half = Vec2::new(
                    multi_half(n_in) * sz * z,
                    (SOCKET_R * z * sz).max(1.5) * 0.8,
                );
                painter.rect_filled(Rect::from_center_size(s, half * 2.0), half.y, col);
            }
            let dotted = big_dot(&n.kind) && !self.link_mode;
            let shown: Vec<usize> = if self.link_mode || bare || multi {
                Vec::new()
            } else if dotted {
                shown_inputs(graph, *id)
            } else {
                (0..n.kind.input_count()).collect()
            };
            if dotted {
                // The BIG DOT: a ring with a core, for every input not yet
                // popped out. Lit while a noodle hangs over it.
                let c = big_dot_rest(graph, *id, np, sz);
                let col = theme::port_in();
                let c = tip(&tips, *id, BIG_DOT_STALK, c);
                stalk(c, Vec2::new(0.0, -1.0), col, None, None, Bead::Dot);
                let s = view.to_screen(c);
                self.big_dot_spots.push((*id, s));
                let r = (BIG_DOT_R * z * sz).max(3.0);
                let hot = hit_big == Some(*id)
                    && matches!(self.drag, Some(DragKind::WireFromOutput(..)));
                painter.circle(
                    s,
                    r,
                    if hot { col.gamma_multiply(0.6) } else { theme::graph_bg() },
                    Stroke::new((r * 0.3).clamp(1.0, 3.0), col),
                );
                painter.circle_filled(s, r * 0.35, col);
            }
            for i in shown {
                // A pipe-follower in a matte pipe keeps its chain socket.
                let mask = n.kind.input_kind(i) == PortKind::Mask && !n.kind.follows_pipe(i);
                // A merge bar's inputs all stand up off its top edge, the
                // matte on the porch included — like the harp's cable.
                let bar = crate::breakout::is_bar(&n.kind);
                let lean = mask_dirs.get(id).copied().unwrap_or(Vec2::new(-1.0, 0.0));
                let sd = if bar {
                    Vec2::new(0.0, -1.0)
                } else if let Some(d) = side_dir(&n.kind, i, lean) {
                    d
                } else if mask {
                    lean
                } else {
                    Vec2::new(0.0, -1.0)
                };
                // Kinds whose ports are not interchangeable label the socket
                // itself — a merge's +/-, a defocus's Z and B. The kind owns
                // the glyph (`NodeKind::input_tag`) so a new node with named
                // ports gets this without touching the editor.
                let tag = match self.drag {
                    Some(DragKind::LayerReorder { node, from, .. }) if node == *id => {
                        n.kind.input_tag(reorder_slot(i, from, self.reorder_to.unwrap_or(from)))
                    }
                    _ => n.kind.input_tag(i),
                };
                // Only names that SAY something: "image" on every node in
                // the comp is a label that costs attention and gives none.
                let name = match n.kind.input_name(i) {
                    _ if !n.kind.labels_inputs() => None,
                    "image" | "input" => None,
                    other => Some(other),
                };
                stalk(
                    tip(&tips, *id, i, rest_of(graph, &mask_dirs, *id, i, np, sz)),
                    sd,
                    input_socket_color(&n.kind, i),
                    tag,
                    name,
                    Bead::Cup(input_at.get(&(*id, i)).copied().map(|src| {
                        let o = graph.out_of(*id, i) as usize;
                        graph
                            .node(src)
                            .map_or_else(theme::port_in, |sn| socket_color(sn.kind.output_kind_at(o)))
                    })),
                );
            }
            if !self.link_mode && !bare {
                // One nodule per output. A multi-band key shows its bands as
                // separate sockets, named, so wiring band 3 is pointing at
                // band 3 rather than picking an index off a menu.
                let n_o = n.kind.output_count().clamp(1, MAX_OUTPUTS);
                let labels: Vec<String> = match (&n.kind, n_o) {
                    (NodeKind::LumaKey(k), n_o) if n_o > 1 => {
                        (0..n_o).map(|i| k.band(i).label(i)).collect()
                    }
                    // The harp names its strings: each nodule wears its
                    // layer — and the last one "remainder" when the bar
                    // carries the undistributed-energy pass.
                    (NodeKind::Breakout(p), _)
                        if !p.layers.is_empty() || p.remainder =>
                    {
                        let mut v = p.layers.clone();
                        if p.remainder {
                            v.push("remainder".into());
                        }
                        v
                    }
                    // The comp, then each layer by its name or number.
                    (NodeKind::Layer(p), n_o) if n_o > 1 => std::iter::once("comp".to_string())
                        .chain(p.layers.iter().enumerate().map(|(i, l)| {
                            if l.name.is_empty() { (i + 1).to_string() } else { l.name.clone() }
                        }))
                        .collect(),
                    _ => Vec::new(),
                };
                for k in 0..n_o {
                    stalk(
                        tip(&tips, *id, out_stalk(k), output_rest_of(&graph.nodes, *id, k, dz(*id))),
                        Vec2::new(
                            if n_o > 1 {
                                (k as f32 - (n_o as f32 - 1.0) / 2.0) * 0.6
                            } else {
                                0.0
                            },
                            1.0,
                        ),
                        socket_color(n.kind.output_kind_at(k)),
                        None,
                        labels.get(k).map(|s| s.as_str()),
                        Bead::Dot,
                    );
                }
            }

            // A Load wears its cache state (its in..out all in a tier =
            // green, not = red) and a Bake its bake state, in place of the
            // type colour: the node says whether it is serving from disk.
            let base = match self.cost_flow.served.get(id) {
                Some(true) => theme::cache_ram(),
                Some(false) => theme::cost_slow(),
                None => type_color(&n.kind),
            };
            let fill = if n.disabled { base.gamma_multiply(0.35) } else { base };
            let stroke = if selected {
                Stroke::new(2.5, theme::selection())
            } else {
                Stroke::new(1.0, theme::node_outline())
            };
            let r_node = R * z * sz;
            // A merge bar is the harp's slab with its own glyph set in the
            // middle, shrunk to the slab's height.
            let r = paint_bar(&painter, &n.kind, sp, z * sz, fill, stroke).unwrap_or(r_node);
            // A node with INNARDS gives way to them as it grows on screen:
            // the glyph fades out while one tiny node per item fades in.
            let innards = crate::innards::provider(&n.kind)
                .map(|p| (p, crate::innards::reveal(r)))
                .filter(|(_, rev)| *rev > 0.0);
            // A node with CONTROLS does the same, giving way to a dial and
            // draggable values.
            let ctl_rev = crate::nodectl::controls(&n.kind)
                .map(|_| crate::nodectl::reveal(r))
                .filter(|rev| *rev > 0.0);
            {
            let faded;
            let painter = match innards.map(|(_, rev)| rev).or(ctl_rev) {
                Some(rev) => {
                    faded = {
                        let mut p = painter.clone();
                        p.multiply_opacity(1.0 - rev);
                        p
                    };
                    &faded
                }
                None => &painter,
            };
            if tv_on(n) {
                // A Read in TV mode: a little CRT showing its frame (in
                // boring mode, a plain card with the frame on it).
                let cab = Rect::from_center_size(sp, tv_half() * 2.0 * z * sz);
                let (tex, scrub) = match tv_pics.get(id) {
                    Some((t, o)) => (t.as_ref(), o.clone()),
                    None => (None, None),
                };
                if boring() {
                    crate::tv::plain(&painter, cab, fill, stroke, tex, scrub);
                } else {
                crate::tv::draw(
                    &painter,
                    cab,
                    fill,
                    stroke,
                    tex,
                    scrub,
                    *id,
                    ui.input(|i| i.time) as f32,
                );
                }
            } else if let Some(plus) = paint_body(painter, &n.kind, sp, r, z * sz, fill, stroke) {
                self.plus_hits.push((*id, plus));
                // A Switch under the pointer rings the stud a click would
                // throw the lever to.
                if let (NodeKind::Switch(sw), Some(p)) = (&n.kind, pointer) {
                    if plus.contains(p) && self.press_state.is_none_or(|(op, ..)| (p - op).length() < 5.0) {
                        let to = switch_pick(sp, r, sw.n(), sw.live(), p);
                        let a = switch_angle(to, sw.n());
                        painter.circle_stroke(
                            sp + Vec2::new(a.cos(), -a.sin()) * r * 0.8,
                            (r * 0.2).max(2.5),
                            Stroke::new((r * 0.06).max(1.0), theme::node_ink_light()),
                        );
                    }
                }
            }
            }
            if let Some((prov, rev)) = innards {
                let frame = crate::innards::frame_for(sp, r);
                let dragging = match self.innard_drag {
                    Some(crate::innards::Drag::Wire { host, from }) if host == *id => pointer.map(|p| (from, p)),
                    _ => None,
                };
                let hits = crate::innards::paint(
                    &painter,
                    frame,
                    &prov.mini(&n.kind),
                    prov.can_add(&n.kind),
                    crate::innards::picked(ui.ctx(), *id),
                    rev,
                    fill,
                    stroke,
                    dragging,
                    self.ortho_wires,
                );
                self.innard_hits.push((*id, frame, hits, crate::innards::live(r)));
            }
            if let Some(rev) = ctl_rev {
                let place = crate::nodectl::Place { c: sp, r };
                let live_kind = n.kind_at(self.eval_tick);
                let hot = match &self.ctl_drag {
                    Some(d) if d.host == *id => Some(d.hit),
                    Some(_) => None,
                    None if crate::nodectl::live(r) => {
                        pointer.and_then(|p| crate::nodectl::hover(place, &live_kind, p))
                    }
                    None => None,
                };
                let hits = crate::nodectl::paint(
                    &painter,
                    place,
                    &live_kind,
                    &|ch| n.anim.get(&ch).is_some_and(|c| !c.keys.is_empty()),
                    hot,
                    rev,
                    stroke,
                );
                self.ctl_hits.push((*id, place, hits, crate::nodectl::live(r)));
            }

            // Cost + licence marks, drawn over whatever shape the kind
            // chose so every node wears them the same way.
            //
            // The ring is on the node that IS slow, not on everything
            // downstream of it — the wires already carry the taint, and
            // ringing the whole tail would say "these are all expensive"
            // when the truth is "one of these is, and it is this one".
            let speed = if n.disabled {
                engine::cost::Speed::Fast
            } else {
                engine::cost::speed_of(&n.kind)
            };
            if speed == engine::cost::Speed::Heavy {
                // REALLY slow: the node BURNS. Additive light falling off
                // smoothly outward and breathing slowly, so the node that
                // dominates the frame is the one your eye lands on across a
                // whole comp. Outside the shape only: a wash over the fill
                // would recolour the node's own type.
                let t = ui.input(|i| i.time) as f32;
                let breathe = 0.82 + 0.18 * (t * 1.5).sin();
                let c = theme::cost_slow();
                let reach = r.max(9.0);
                glow_ring(&painter, sp, r, reach * HEAVY_REACH, glow_light(c, 0.40 * breathe));
                glow_ring(&painter, sp, r, reach * 1.5, glow_light(c, 0.85 * breathe));
                glow_ring(
                    &painter,
                    sp,
                    r,
                    reach * 1.16,
                    glow_light(c.lerp_to_gamma(egui::Color32::WHITE, 0.35), breathe),
                );
            }
            if speed != engine::cost::Speed::Fast {
                let w = (r * 0.16).max(1.2);
                painter.circle_stroke(sp, r * 1.25, Stroke::new(w, theme::cost_slow()));
                if speed == engine::cost::Speed::Heavy {
                    // The ring on a heavy node is not painted red, it is
                    // RED HOT: light added over the stroke, so it reads as
                    // the source the halo comes off.
                    let t = ui.input(|i| i.time) as f32;
                    let hot = 0.34 + 0.12 * (t * 1.5).sin();
                    painter.circle_stroke(
                        sp,
                        r * 1.25,
                        Stroke::new(w * 1.4, glow_light(theme::cost_slow(), hot)),
                    );
                }
            }
            // Research-only weights: a different hue, because it is a
            // restriction rather than a cost (manifesto rule 10). The node
            // still works — this tags it, it does not gate it.
            if engine::cost::licence_of(&n.kind).warns() {
                let br = (r * 0.34).clamp(2.6, 4.2);
                painter.circle(
                    sp + Vec2::new(-r * 0.95 - 4.0, -r * 0.95 - 4.0),
                    br,
                    theme::cost_nc(),
                    Stroke::new((br * 0.3).max(1.0), theme::node_outline()),
                );
            }
            // A pipeline that bills when you press run: a `$` in the same
            // corner (a GenAI node never declares weights).
            if let NodeKind::GenAi(p) = &n.kind {
                if crate::genaiui::spends(p) {
                    crate::genaiui::money_badge(&painter, sp, r);
                }
            }
            // Weights not downloaded: a down-arrow off the lower-left
            // corner, until the node's panel gate has fetched them. Only
            // nodes that declare models pay the (stat-only) cache check.
            if !n.kind.models().is_empty() && !engine::models::missing(n.kind.models().iter().copied()).is_empty() {
                let br = (r * 0.4).clamp(3.0, 5.0);
                let c = sp + Vec2::new(-r * 0.95 - 4.0, r * 0.95 + 4.0);
                painter.add(egui::Shape::convex_polygon(
                    vec![c + Vec2::new(-br, -br * 0.6), c + Vec2::new(br, -br * 0.6), c + Vec2::new(0.0, br)],
                    theme::warn(),
                    Stroke::new((br * 0.25).max(1.0), theme::node_outline()),
                ));
            }

            // A node scaled (or zoomed) down to a speck is a STAR: too
            // small to read as a shape, so it wears light instead — a hot
            // core that never blinks out, four spikes and a halo in a hint
            // of its own type colour, each on its own twinkle so a field of
            // them shimmers rather than pulsing in lockstep.
            let lit = (1.0 - r / SPECK_R).clamp(0.0, 1.0);
            if lit > 0.0 {
                let t = ui.input(|i| i.time) as f32;
                twinkle_star(&painter, sp, lit, twinkle_phase(*id, t), base);
            }

            // Pin markers: properties above right, viewer manipulators below
            // right. Ringed in near-black so they read against any node
            // colour and against the grid — these say what is pinned, which
            // is state you want to spot without hunting for it.
            // Corner ornaments (pins, the maturity dot, the bypass cross)
            // hang off the shape's corners: a TV's cabinet is wider than
            // the node circle, and a pip INSIDE the screen reads as dirt.
            let tv_half = wide_of(n)
                .filter(|_| tv_on(n) || (boring() && boring_shape(&n.kind)))
                .map(|(h, _)| h * z * sz);
            let (ox, oy) = match tv_half {
                Some(h) => (h.x, h.y),
                None => (r * 0.95, r * 0.95),
            };
            let pip = |at: Pos2, color: Color32| {
                let pr = (r * 0.35).clamp(2.8, 4.3);
                painter.circle(
                    at,
                    pr,
                    color,
                    Stroke::new((pr * 0.3).max(1.0), theme::node_outline()),
                );
            };
            if activated.contains(id) {
                pip(sp + Vec2::new(ox + 4.0, -oy - 4.0), theme::pinned());
            }
            if gizmo_pinned.contains(id) {
                pip(sp + Vec2::new(ox + 4.0, oy + 4.0), theme::port_out());
            }
            // Maturity: a small unringed dot below LEFT, away from the pins,
            // in the colour the menus use — and none at all for production,
            // so a finished comp is plain. Subtle on purpose: it is there
            // when you look for it, not something the graph shouts.
            if let Some(c) = status_of(&n.kind).mark() {
                // It fades out as a node turns into a speck: at that size
                // the dots are bigger than the node, and the twinkle is
                // what says "a node is here".
                painter.circle_filled(
                    sp + Vec2::new(-ox - 4.0, oy + 4.0),
                    (r * 0.22).clamp(2.0, 3.2),
                    c.gamma_multiply(0.8 * (1.0 - lit)),
                );
            }
            // Viewer halos: slot A gold, slot B cyan; the active slot is
            // bright, the parked one dim. On a multi-output node the halo
            // rides the VIEWED NODULE, not the body — which string of a
            // harp (or band of a key) is on screen must be readable off
            // the graph, and it is the first thing to check when a view
            // does not update.
            let n_o = n.kind.output_count().clamp(1, MAX_OUTPUTS);
            let halo_at = |out: u32| -> (Pos2, f32) {
                if n_o > 1 {
                    let k = (out as usize).min(n_o - 1);
                    let at = view.to_screen(tip(
                        &tips,
                        *id,
                        out_stalk(k),
                        output_rest_of(&graph.nodes, *id, k, dz(*id)),
                    ));
                    (at, (SOCKET_R * z * sz).clamp(3.0, 8.0) + 4.0)
                } else {
                    (sp, tv_half.map_or(r, |h| h.length()) + 7.0)
                }
            };
            if graph.view == Some(*id) {
                let a = if graph.view_active_b { 0.35 } else { 1.0 };
                let (at, rr) = halo_at(graph.view_out);
                painter.circle_stroke(
                    at,
                    rr,
                    Stroke::new(2.5, theme::port_out().gamma_multiply(a)),
                );
            }
            if graph.view_b == Some(*id) {
                let a = if graph.view_active_b { 1.0 } else { 0.35 };
                let (at, rr) = halo_at(graph.view_b_out);
                painter.circle_stroke(
                    at,
                    rr + 4.0,
                    Stroke::new(2.0, theme::port_in().gamma_multiply(a)),
                );
            }
            // Named by the expression being typed, or its highlighted
            // completion (`crate::chanexpr`).
            if crate::chanexpr::lit(*id) {
                painter.circle_stroke(
                    sp,
                    tv_half.map_or(r, |h| h.length()) + 9.0,
                    Stroke::new(2.0, theme::patch_expr()),
                );
            }
            if n.disabled {
                let (cx, cy) = tv_half.map_or((r, r), |h| (h.x, h.y));
                painter.line_segment(
                    [sp + Vec2::new(-cx, -cy), sp + Vec2::new(cx, cy)],
                    Stroke::new(2.0, theme::cut()),
                );
            }
            // Names are drawn AFTER every node, once every node's rect is
            // known: a long one that would run into its neighbour swings up
            // around its own node's centre until it clears (`place_label`).
            if z * sz > 0.4 {
                // A merge bar's name starts past its right end (and past
                // an Extract's +), not over the slab.
                let reach = match (crate::breakout::is_bar(&n.kind) || tv_on(n) || (boring() && boring_shape(&n.kind)))
                    .then(|| wide_of(n))
                    .flatten()
                {
                    Some((h, dx)) => (dx + h.x) * z * sz + 8.0,
                    None => r + 8.0,
                };
                // While an expression is typed, nodes go by the name a path
                // spells (`crate::chanexpr`): labels are not unique.
                let text = if crate::chanexpr::naming() {
                    crate::chanexpr::node_name(graph, *id).unwrap_or_else(|| display_label(n))
                } else {
                    display_label(n)
                };
                labels.push((*id, sp, reach, text, 12.0 * (z * sz).clamp(0.7, 1.2)));
            }

            node_rects.push((*id, node_rect(r)));
        }

        // --- node names, placed against every node at once. Left to right,
        // so a row of long names fans up in step rather than in draw order.
        self.label_hits.clear();
        labels.sort_by(|a, b| a.1.x.total_cmp(&b.1.x).then(a.1.y.total_cmp(&b.1.y)));
        let mut placed: Vec<[Pos2; 4]> = Vec::with_capacity(labels.len());
        let dt = ui.input(|i| i.stable_dt).clamp(0.001, 0.05);
        let mut moving = false;
        let seen: std::collections::HashSet<u64> = labels.iter().map(|l| l.0).collect();
        self.label_anim.retain(|id, _| seen.contains(id));
        // A name only ever collides with what is near the panel; the
        // culled nodes' rects are far outside it.
        let near_rects: Vec<(u64, Rect)> =
            node_rects.iter().filter(|(_, r)| r.intersects(cull)).copied().collect();
        for (id, c, start, text, size) in labels {
            let galley =
                painter.layout_no_wrap(text, egui::FontId::proportional(size), theme::node_label());
            // Placed against where the OTHER names will come to rest, not
            // where they are mid-bounce, so the targets never chase each
            // other's wobble.
            let (target, q) = place_label(id, c, start, galley.size(), &near_rects, &placed);
            placed.push(q);
            // A name that has just appeared starts where it belongs: a
            // setup opening with every label bouncing in would be noise.
            let (th, v) = self.label_anim.get(&id).copied().unwrap_or((target, 0.0));
            let (theta, v) = label_spring(th, v, target, dt, self.wobble);
            if (target - theta).abs() > 1e-3 || v.abs() > 1e-3 {
                moving = true;
                self.label_anim.insert(id, (theta, v));
            } else {
                self.label_anim.insert(id, (target, 0.0));
            }
            // The galley's top-left, so its middle runs along the ray.
            let u = Vec2::angled(theta);
            let v = Vec2::new(-u.y, u.x);
            let at = c + u * start - v * (galley.size().y * 0.5);
            let (gw, gh) = (galley.size().x, galley.size().y);
            self.label_hits.push((
                id,
                [at, at + u * gw, at + u * gw + v * gh, at + v * gh],
                at,
                theta,
            ));
            // The one being renamed is drawn by its editor instead.
            if self.rename.as_ref().is_none_or(|(rid, ..)| *rid != id) {
                painter.add(
                    egui::epaint::TextShape::new(at, galley, theme::node_label())
                        .with_angle(theta),
                );
            }
        }
        if moving {
            ui.ctx().request_repaint();
        }

        // --- the range view: each noodle's first–last at its middle.
        if self.show_ranges && self.zoom > 0.3 {
            let size = 10.5 * self.zoom.clamp(0.85, 1.25);
            let mut taken: Vec<Rect> = Vec::new();
            for (mid, src) in std::mem::take(&mut range_tags) {
                let Some(&(a, b)) = self.wire_ranges.get(&src) else { continue };
                let set = self.range_set_here(graph, src);
                let ink = if set { theme::text() } else { theme::text().gamma_multiply(0.55) };
                let galley = painter.layout_no_wrap(
                    format!("{a}–{b}"),
                    egui::FontId::monospace(size),
                    ink,
                );
                let r = Rect::from_center_size(mid, galley.size()).expand2(Vec2::new(4.0, 1.5));
                // Two tags on one spot (a node feeding two inputs, a
                // crowded merge) draw once: the first says it.
                if taken.iter().any(|t| t.intersects(r)) {
                    continue;
                }
                taken.push(r);
                painter.rect_filled(r, 3.0, theme::well());
                let edge = if set { theme::accent_text() } else { theme::border() };
                painter.rect_stroke(r, 3.0, Stroke::new(1.0, edge), egui::StrokeKind::Inside);
                painter.galley(r.min + Vec2::new(4.0, 1.5), galley, ink);
                self.shown_range_tags.push((src, (a, b)));
            }
        }

        // --- rename: a text field laid over the node's name. Level, even
        // when the name is tilted — a rotated text field is unreadable to
        // type into. Enter or clicking away commits (Finder's rule), Esc
        // cancels (the Escape handler above takes it), and an empty name,
        // or the kind's own name, clears `ui_label` so the node tracks its
        // kind again.
        if let Some((id, buf, fresh)) = &mut self.rename {
            let id = *id;
            match graph.node(id) {
                None => self.rename = None,
                Some(n) => {
                    let kind_name = node_label(&n.kind);
                    let at = self
                        .label_hits
                        .iter()
                        .find(|(lid, ..)| *lid == id)
                        .map(|(_, _, at, _)| *at)
                        .unwrap_or_else(|| {
                            let (c, r) = (view.to_screen(dp(id)), R * self.zoom * ds(id));
                            c + Vec2::new(r + 8.0, -9.0)
                        });
                    let size = (12.0 * (self.zoom * ds(id)).clamp(0.7, 1.2)).max(11.0);
                    let mut commit = false;
                    egui::Area::new(ui.id().with(("rename", id)))
                        .order(egui::Order::Foreground)
                        .fixed_pos(at - Vec2::new(4.0, 3.0))
                        .show(ui.ctx(), |ui| {
                            let te = ui.add(
                                egui::TextEdit::singleline(buf)
                                    .hint_text(kind_name.as_str())
                                    .font(egui::FontId::proportional(size))
                                    .desired_width(150.0),
                            );
                            if std::mem::take(fresh) {
                                te.request_focus();
                            } else if te.lost_focus() {
                                commit = true;
                            }
                        });
                    if commit {
                        let name = buf.trim().to_string();
                        if let Some(n) = graph.node_mut(id) {
                            n.ui_label =
                                if name == kind_name { String::new() } else { name };
                        }
                        self.rename = None;
                    }
                }
            }
        }

        // --- S / R + drag: a line from the selection's centre to the
        // pointer, the way Blender shows what a scale or rotate pivots on.
        if let (
            Some(DragKind::SelScale { center, .. } | DragKind::SelRotate { center, .. }),
            Some(p),
        ) = (&self.drag, pointer)
        {
            let c = view.to_screen(*center);
            painter.add(egui::Shape::dashed_line(
                &[c, p],
                Stroke::new(1.0, theme::selection_box()),
                6.0,
                4.0,
            ));
            painter.circle(c, 4.0, theme::handle(), Stroke::new(1.0, theme::handle_rim()));
        }

        // The key strip's and the legend's words, on top of everything
        // once asked for.
        if let Some(st) = &key_strip {
            st.draw_words(&painter, self.keys_reveal);
        }
        if let Some(st) = &legend_strip {
            st.draw_words(&painter, self.legend_reveal);
        }

        // --- tab search popup ---
        if let Some(NodeSearch { at: gpos, query, sel }) = &mut self.search {
            let gpos = *gpos;
            let spos = view.to_screen(gpos);
            let mut close = resp.clicked();
            let mut pick: Option<NodeKind> = None;
            let enter = ui.input(|i| i.key_pressed(egui::Key::Enter));
            // ↑/↓ walk the hits, and they are taken from the raw queue HERE,
            // before the field is built: a singleline TextEdit answers them
            // itself (they run its text cursor), so a read afterwards would
            // never see them. Both are consumed whatever the other did.
            let step = ui.input_mut(|i| {
                i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowDown) as i32
                    - i.consume_key(egui::Modifiers::NONE, egui::Key::ArrowUp) as i32
            });
            if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                close = true;
            }
            egui::Area::new(ui.id().with("node-search"))
                .order(egui::Order::Foreground)
                .fixed_pos(spos)
                .show(ui.ctx(), |ui| {
                    egui::Frame::window(ui.style()).show(ui, |ui| {
                        ui.set_width(160.0);
                        let te = ui.add(
                            egui::TextEdit::singleline(query).hint_text("add node…"),
                        );
                        te.request_focus();
                        // A new letter re-ranks everything, so the walk
                        // starts again at the top rather than pointing at
                        // whatever now happens to sit where it stopped.
                        if te.changed() {
                            *sel = 0;
                        }
                        let q = query.to_lowercase();
                        // Same catalog the right-click menu groups: one
                        // source of truth, flattened for typing at.
                        let slot = *next_read_slot;
                        // The project's maturity threshold applies here as
                        // it does in the right-click menu: one filter.
                        let min_status = graph.node_status_min;
                        let mut all: Vec<(String, NodeKind, Option<NodeCat>, NodeStatus)> =
                            node_catalog()
                                .into_iter()
                                .chain(search_aliases())
                                .filter(|e| {
                                    offered_in(e.cat, self.in_noise, self.in_geo, self.in_sdf, self.in_particles, self.in_genai, self.in_roto)
                                })
                                .filter(|e| min_status.admits(e.status))
                                .map(|e| {
                                    ((e.label).to_string(), (e.make)(slot), Some(e.cat), e.status)
                                })
                                .collect();
                        // Controls are offered here for the same reason the
                        // right-click menu offers them: a comp exporting to an
                        // edit publishes them as the shot's panel. Only the
                        // body-only input proxies stay hidden.
                        // Not inside a sub-graph, like the right-click
                        // menu: a plugin is an image node.
                        let sub = self.in_noise || self.in_geo || self.in_sdf || self.in_particles || self.in_genai || self.in_roto;
                        all.extend(engine::plugins::all().into_iter().filter(|d| {
                            !sub && (self.in_tool || !crate::tools::is_proxy_id(&d.type_id))
                        }).map(|d| {
                            let kind = NodeKind::Custom {
                                type_id: d.type_id.clone(),
                                params: Default::default(),
                            };
                            let status = status_of(&kind);
                            (d.label.clone(), kind, None, status)
                        }).filter(|(_, _, _, s)| min_status.admits(*s)));
                        // Empty, the most-used few; typed, the best few
                        // (usage lifting close matches) — see `rank_search`. A whole catalog under the
                        // field ran the full height of the screen.
                        let (shown, total) = rank_search(
                            all.into_iter().map(|(n, k, c, s)| (n, (k, c, s))).collect(),
                            &q,
                            &self.use_counts,
                        );
                        if shown.is_empty() {
                            ui.weak(if q.is_empty() { "type to search" } else { "no match" });
                        }
                        // Clamp before stepping: the list is rebuilt every
                        // keystroke and can be shorter than where the hand
                        // left the cursor. Enter takes the marked row.
                        *sel = (*sel).min(shown.len().saturating_sub(1));
                        *sel = step_sel(*sel, step, shown.len());
                        for (i, (name, (kind, cat, status))) in shown.into_iter().enumerate() {
                            let hit = ui
                                .add(
                                    egui::Button::new(status_text(&name, status))
                                        .right_text(
                                            egui::RichText::new(
                                                cat.map(|c| c.name()).unwrap_or(match &kind {
                                                    NodeKind::Custom { type_id, .. }
                                                        if crate::tools::is_proxy_id(type_id) =>
                                                    {
                                                        "input"
                                                    }
                                                    NodeKind::Custom { type_id, .. }
                                                        if is_subwob_type(type_id) =>
                                                    {
                                                        "subwob"
                                                    }
                                                    NodeKind::Custom { type_id, .. }
                                                        if crate::tools::ControlKind::from_type(type_id)
                                                            .is_some() =>
                                                    {
                                                        "control"
                                                    }
                                                    _ => "plugin",
                                                }),
                                            )
                                            .small()
                                            .weak(),
                                        )
                                        .selected(i == *sel),
                                )
                                .on_hover_text(status.name())
                                .clicked()
                                || (enter && i == *sel);
                            if hit {
                                *self.use_counts.entry(name.clone()).or_insert(0) += 1;
                                self.use_counts_dirty = true;
                                pick = Some(kind);
                            }
                        }
                        if total > SEARCH_MAX {
                            ui.weak(format!("{} more — keep typing", total - SEARCH_MAX));
                        }
                    });
                });
            if let Some(kind) = pick {
                // `reader_slot()` is the one place that knows which kinds own
                // a slot; a hand-written list here would be a third copy to
                // keep in step, and getting it wrong makes two nodes fight
                // over one texture.
                if kind.reader_slot().is_some() {
                    *next_read_slot += 1;
                }
                // Wired from the selection and placed from its wiring — the
                // same add every creation path makes (`add_wired`).
                let id = add_wired(graph, kind, gpos, selection, self.zoom, self.patch_mode);
                self.added.push(id);
                close = true;
            }
            if close {
                self.search = None;
            }
        }

        // --- the backtick pie: the active region's common sources in a
        // small orbit around the cursor, alive only while ` is held.
        // Releasing over a disc is the pick; releasing anywhere else just
        // lets go. Each source is a disc two nodules across, its label set
        // along the 45° line the socket names use — flipped downward for
        // the discs on the lower half, so nothing reads upside-out. The
        // orbit lands on the same underdamped spring as every stalk.
        if let Some(Pie { at: c, scope, adding }) = self.pie {
            let rid = match scope {
                PieScope::Region(r) => r,
                PieScope::Global => None,
            };
            let (s, v) = self.pie_anim;
            let (s, v) = pie_spring(s, v, dt, self.wobble);
            self.pie_anim = (s, v);
            let sc = s.max(0.0);
            let reg = rid.and_then(|rid| graph.regions.iter().find(|r| r.id == rid));
            // ADD mode (one node was selected at press): two discs — up
            // registers it as a source under its own name, down opens the
            // naming popup first. Otherwise: one disc per source to spawn.
            // A roster with nowhere to put a source (the region pie with no
            // region under the view) offers nothing and says why.
            let can_add = scope != PieScope::Region(None);
            let sources: Vec<(u64, String)> = match adding
                .filter(|_| can_add)
                .and_then(|a| graph.node(a).map(|n| (a, display_label(n))))
            {
                Some((_, label)) => {
                    vec![(0, format!("+ {label}")), (0, "name…".into())]
                }
                None if adding.is_some() => Vec::new(),
                None => match scope {
                    PieScope::Global => graph
                        .global_sources
                        .iter()
                        .filter_map(|s| graph.node(*s).map(|n| (*s, display_label(n))))
                        .collect(),
                    PieScope::Region(_) => reg
                        .map(|r| {
                            r.sources
                                .iter()
                                .filter_map(|s| {
                                    graph.node(*s).map(|n| (*s, display_label(n)))
                                })
                                .collect()
                        })
                        .unwrap_or_default(),
                },
            };
            let ring = crate::regions::PIE_R * sc;
            let disc_r = crate::regions::PIE_DISC_R * sc.min(1.1);
            let hot = pointer.and_then(|p| crate::regions::pie_pick(c, p, sources.len()));
            painter.circle_filled(c, 2.0, theme::node_label());
            if sources.is_empty() {
                painter.text(
                    c + Vec2::new(14.0, -14.0),
                    egui::Align2::LEFT_BOTTOM,
                    match (scope, reg) {
                        (PieScope::Region(_), None) => "no region",
                        (PieScope::Global, _) => "no global sources",
                        (PieScope::Region(_), Some(_)) => "no sources",
                    },
                    egui::FontId::proportional(11.0),
                    theme::text(),
                );
            }
            for (i, (_, name)) in sources.iter().enumerate() {
                let dp_ = crate::regions::pie_slice_pos(c, i, sources.len(), ring);
                let is_hot = hot == Some(i);
                let col = if is_hot { theme::text_hot() } else { theme::hidden_wire() };
                painter.circle(
                    dp_,
                    disc_r,
                    theme::graph_bg(),
                    Stroke::new(if is_hot { 2.0 } else { 1.3 }, col),
                );
                // The socket-label idiom, radiating: up-right above the
                // waist, down-right below it (inverted vertically so the
                // low labels still read left to right).
                let k = disc_r * 1.2 + 3.0;
                let below = dp_.y > c.y + 0.5;
                let galley = painter.layout_no_wrap(
                    name.clone(),
                    egui::FontId::proportional(11.0),
                    col.gamma_multiply(0.9),
                );
                let (off, ang) = if below {
                    (Vec2::new(k * 0.7, k), std::f32::consts::FRAC_PI_4)
                } else {
                    (Vec2::new(k * 0.7, -k - galley.size().y * 0.7), -std::f32::consts::FRAC_PI_4)
                };
                painter.add(
                    egui::epaint::TextShape::new(dp_ + off, galley, col.gamma_multiply(0.9))
                        .with_angle(ang),
                );
            }
            // Let go: over a disc it commits, anywhere else it just closes.
            let released = ui.input(|i| crate::keymap::released(i, scope.act()));
            let held = ui.input(|i| crate::keymap::down(i, scope.act()));
            if released || !held {
                match (adding, released.then_some(hot).flatten()) {
                    // ADD mode, up: register under the node's own name, in
                    // whichever roster this pie serves.
                    (Some(node), Some(0)) => match scope {
                        PieScope::Global => crate::regions::add_global_source(graph, node),
                        PieScope::Region(Some(rid)) => {
                            crate::regions::add_source(graph, rid, node)
                        }
                        PieScope::Region(None) => {}
                    },
                    // ADD mode, down: name it first — the popup commits
                    // the rename AND the registration together.
                    (Some(node), Some(1)) => {
                        let buf =
                            graph.node(node).map(display_label).unwrap_or_default();
                        self.name_popup = Some((node, scope, c, buf));
                    }
                    (Some(_), _) => {}
                    // Source mode: spawn the chip.
                    (None, Some(i)) => {
                        if let Some(src) = sources.get(i).map(|(id, _)| *id) {
                            let at = view.to_graph(c);
                            let id = graph.add(NodeKind::dot(PortKind::Image), at);
                            let size = new_node_size(graph, at, self.zoom, &[id]);
                            if let Some(n) = graph.nodes.get_mut(&id) {
                                n.size = size;
                            }
                            // The chip: a Dot hidden-wired to the source.
                            // It joins the region it was born in; magnetism
                            // takes it from here.
                            graph.connect_hidden(id, 0, src);
                            // The chip joins the region it LANDS in, which
                            // for the region pie is the one it served.
                            if let Some(rid) = rid.or(self.active_region) {
                                crate::regions::adopt(graph, rid, &[id]);
                            }
                            *selection = vec![id];
                        }
                    }
                    (None, None) => {}
                }
                self.pie = None;
            }
        }

        // --- the ops pie: plain ` held, common node ops on an orbit of
        // labels. Release over a slice queues that op for its key site.
        if let Some(c) = self.ops_pie {
            let (s, v) = self.ops_anim;
            let (s, v) = pie_spring(s, v, dt, self.wobble);
            self.ops_anim = (s, v);
            let ring = OPS_PIE_R * s.max(0.0);
            let n = OPS_PIE.len();
            let hot = pointer.and_then(|p| crate::regions::pie_pick(c, p, n));
            painter.circle_filled(c, 2.0, theme::node_label());
            for (i, (_, name)) in OPS_PIE.iter().enumerate() {
                let at = crate::regions::pie_slice_pos(c, i, n, ring);
                let is_hot = hot == Some(i);
                let col = if is_hot { theme::text_hot() } else { theme::text() };
                let galley = painter.layout_no_wrap(
                    (*name).to_string(),
                    egui::FontId::proportional(12.0),
                    col,
                );
                let r = Rect::from_center_size(at, galley.size() + Vec2::new(10.0, 4.0));
                painter.rect(
                    r,
                    4.0,
                    theme::graph_bg(),
                    Stroke::new(if is_hot { 2.0 } else { 1.0 }, col.gamma_multiply(0.7)),
                    egui::StrokeKind::Outside,
                );
                painter.galley(r.center() - galley.size() * 0.5, galley, col);
            }
            let act = crate::keymap::Act::GraphOpsPie;
            let released = ui.input(|i| crate::keymap::released(i, act));
            let held = ui.input(|i| crate::keymap::down(i, act));
            if released || !held {
                if released {
                    self.ops_pick = hot.map(|i| OPS_PIE[i].0);
                }
                self.ops_pie = None;
            }
        }

        // --- the node pie: F1 held, the eight most-added kinds on the ops
        // orbit. Release over a slice drops that node at the pie's centre.
        if let Some(c) = self.node_pie {
            let (s, v) = self.node_pie_anim;
            let (s, v) = pie_spring(s, v, dt, self.wobble);
            self.node_pie_anim = (s, v);
            let ring = OPS_PIE_R * s.max(0.0);
            let top = top_used(&self.use_counts);
            let n = top.len();
            let hot = pointer.and_then(|p| crate::regions::pie_pick(c, p, n));
            painter.circle_filled(c, 2.0, theme::node_label());
            for (i, e) in top.iter().enumerate() {
                let at = crate::regions::pie_slice_pos(c, i, n, ring);
                let is_hot = hot == Some(i);
                let col = if is_hot { theme::text_hot() } else { theme::text() };
                let galley = painter.layout_no_wrap(
                    e.label.to_string(),
                    egui::FontId::proportional(12.0),
                    col,
                );
                let r = Rect::from_center_size(at, galley.size() + Vec2::new(10.0, 4.0));
                painter.rect(
                    r,
                    4.0,
                    theme::graph_bg(),
                    Stroke::new(if is_hot { 2.0 } else { 1.0 }, col.gamma_multiply(0.7)),
                    egui::StrokeKind::Outside,
                );
                painter.galley(r.center() - galley.size() * 0.5, galley, col);
            }
            let act = crate::keymap::Act::GraphNodePie;
            let released = ui.input(|i| crate::keymap::released(i, act));
            let held = ui.input(|i| crate::keymap::down(i, act));
            if released || !held {
                if let Some(e) = released
                    .then_some(hot)
                    .flatten()
                    .and_then(|i| top.get(i))
                {
                    let slot = *next_read_slot;
                    if e.reader {
                        *next_read_slot += 1;
                    }
                    let id = add_wired(graph, (e.make)(slot), view.to_graph(c), selection, zoom, self.patch_mode);
                    self.added.push(id);
                    *self.use_counts.entry(e.label.to_string()).or_insert(0) += 1;
                    self.use_counts_dirty = true;
                }
                self.node_pie = None;
            }
        }

        // --- the add-pie's naming popup: a name for the source, committed
        // on Enter — it lands on the node itself (`ui_label`, presentation
        // only) and the node lands in the region's source roster.
        if let Some((node, pscope, at, buf)) = &mut self.name_popup {
            let (node, pscope, at) = (*node, *pscope, *at);
            let mut commit = false;
            let mut close = false;
            let ar = egui::Area::new(ui.id().with("source_name"))
                .fixed_pos(at)
                .order(egui::Order::Foreground)
                .show(ui.ctx(), |ui| {
                    egui::Frame::popup(ui.style()).show(ui, |ui| {
                        ui.set_min_width(150.0);
                        ui.label(egui::RichText::new("source name").small().weak());
                        let r = ui.add(
                            egui::TextEdit::singleline(buf).desired_width(140.0),
                        );
                        if ui.ctx().memory(|m| m.focused().is_none()) {
                            r.request_focus();
                        }
                        let enter = ui.input(|i| i.key_pressed(egui::Key::Enter));
                        if (r.lost_focus() && enter) || enter {
                            commit = true;
                        }
                        if ui.button("add").clicked() {
                            commit = true;
                        }
                    });
                });
            if commit {
                let name = buf.trim().to_string();
                if let Some(n) = graph.nodes.get_mut(&node) {
                    // Naming it back to its kind is clearing the rename.
                    n.ui_label =
                        if name == node_label(&n.kind) { String::new() } else { name };
                }
                match pscope {
                    PieScope::Global => crate::regions::add_global_source(graph, node),
                    PieScope::Region(Some(rid)) => {
                        crate::regions::add_source(graph, rid, node)
                    }
                    PieScope::Region(None) => {}
                }
                close = true;
            }
            if close || ar.response.clicked_elsewhere() {
                self.name_popup = None;
            }
        }

        // --- the M popup: the selection joins a region, or gets a new one
        // over it — Blender's move-to-collection habit, comp-sized.
        if let Some((sp, _)) = self.region_menu {
            let mut close = false;
            let ar = egui::Area::new(ui.id().with("region_menu"))
                .fixed_pos(sp)
                .order(egui::Order::Foreground)
                .show(ui.ctx(), |ui| {
                    egui::Frame::popup(ui.style()).show(ui, |ui| {
                        ui.style_mut().wrap_mode = Some(egui::TextWrapMode::Extend);
                        ui.set_min_width(140.0);
                        ui.label(
                            egui::RichText::new(if selection.is_empty() {
                                "regions"
                            } else {
                                "selection ➡ region"
                            })
                            .small()
                            .weak(),
                        );
                        let names: Vec<(u64, String)> = graph
                            .regions
                            .iter()
                            .map(|r| (r.id, r.name.clone()))
                            .collect();
                        for (rid, name) in names {
                            if ui.button(name).clicked() {
                                if !selection.is_empty() {
                                    crate::regions::adopt(graph, rid, &selection.clone());
                                }
                                close = true;
                            }
                        }
                        if !selection.is_empty() && ui.button("new region").clicked() {
                            crate::regions::create(graph, &selection.clone());
                            close = true;
                        }
                        if selection.is_empty() && graph.regions.is_empty() {
                            ui.weak("select nodes; M raises a region over them");
                        }
                    });
                });
            if close || ar.response.clicked_elsewhere() {
                self.region_menu = None;
            }
        }

        // Corner HUDs: which region the view is in (what ` will answer
        // for), and the patch-mode tell — a mode with no banner is a mode
        // you forget you are in.
        if !overlay {
            if let Some(name) = self
                .active_region
                .and_then(|rid| graph.regions.iter().find(|r| r.id == rid))
                .map(|r| r.name.clone())
            {
                painter.text(
                    rect.left_top() + Vec2::new(10.0, 8.0),
                    egui::Align2::LEFT_TOP,
                    name,
                    egui::FontId::proportional(12.0),
                    theme::text(),
                );
            }
            if self.patch_mode {
                painter.text(
                    rect.right_top() + Vec2::new(-10.0, 8.0),
                    egui::Align2::RIGHT_TOP,
                    "patch — hidden wires",
                    egui::FontId::proportional(12.0),
                    theme::hidden_wire(),
                );
            }
        }

        read_removed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A wire off output 1+ (a LumaKey band) lands in `edge_out`, a
    /// tuple-keyed map. Written as a JSON object, serde_json refused it and
    /// every save of such a comp failed. It must round-trip, and the old
    /// empty-object form must still load.
    #[test]
    fn a_setup_with_a_second_output_wire_saves_and_loads() {
        let mut g = EditorGraph::default();
        g.edge_out.insert((3, 0), 2);
        g.edge_out.insert((5, 1), 1);
        let json = serde_json::to_string_pretty(&g).expect("edge_out must serialize");
        let back: EditorGraph = serde_json::from_str(&json).unwrap();
        assert_eq!(back.edge_out, g.edge_out);

        let old = r#"{"nodes": {}, "edges": [], "edge_out": {}, "next_id": 0}"#;
        let g: EditorGraph = serde_json::from_str(old).expect("empty legacy edge_out loads");
        assert!(g.edge_out.is_empty());
    }

    /// R(0) -> A(1) -> B(2) -> M(3)
    /// A setup saved before colour management must open unchanged: working
    /// space Linear Rec.709 (what it was actually working in) and every Read
    /// applying no transform. Getting this wrong reinterprets every pixel of
    /// every existing comp, and it looks like a grade, not like a bug.
    #[test]
    fn setups_predating_colour_management_open_unchanged() {
        let json = r#"{
            "nodes": {"0": {"kind": {"Read": {"source": 0, "path": "plate"}},
                            "pos": [0.0, 0.0]}},
            "edges": [],
            "next_id": 1
        }"#;
        let g: EditorGraph = serde_json::from_str(json).expect("legacy setup must still load");
        assert_eq!(g.working, engine::color::Colorspace::LINEAR_REC709);
        let NodeKind::Read { input, .. } = &g.nodes[&0].kind else {
            panic!("expected a Read");
        };
        assert_eq!(*input, engine::color::InputSpace::Working);
        // ...and the engine graph it builds emits no transform at all.
        let eg = g.to_engine_graph();
        assert_eq!(eg.working, engine::color::Colorspace::LINEAR_REC709);
    }

    /// A new Read sits on "default" and lowers to its format's row of the
    /// defaults table; an explicit pick is left alone.
    #[test]
    fn a_new_read_follows_the_format_defaults() {
        use engine::color::{Colorspace, InputSpace};
        let mut g = EditorGraph::default_graph("a", "b", None);
        let id = g.add(NodeKind::read(7, "plates/sh010.####.dpx"), egui::pos2(0.0, 0.0));
        let lowered = |g: &EditorGraph| match g.to_engine_graph().nodes.get(&id) {
            Some(NodeKind::Read { input, .. }) => *input,
            _ => panic!("expected a Read"),
        };
        assert_eq!(lowered(&g), InputSpace::Explicit(Colorspace::CINEON));
        if let Some(ENode { kind: NodeKind::Read { input, .. }, .. }) = g.node_mut(id) {
            *input = InputSpace::Explicit(Colorspace::REC709);
        }
        assert_eq!(lowered(&g), InputSpace::Explicit(Colorspace::REC709));
    }

    /// A NEW comp, by contrast, works in ACEScg.
    #[test]
    fn a_new_comp_works_in_acescg() {
        let g = EditorGraph::default_graph("a", "b", None);
        assert_eq!(g.working, engine::color::Colorspace::ACESCG);
        assert_eq!(g.to_engine_graph().working, engine::color::Colorspace::ACESCG);
    }

    /// The working space has to survive a save/load round trip, or every
    /// reopened comp silently reverts to the legacy default.
    #[test]
    fn the_working_space_round_trips() {
        let mut g = EditorGraph::default_graph("a", "b", None);
        g.working = engine::color::Colorspace::LINEAR_P3D65;
        let back: EditorGraph =
            serde_json::from_str(&serde_json::to_string(&g).unwrap()).unwrap();
        assert_eq!(back.working, engine::color::Colorspace::LINEAR_P3D65);
    }

    fn chain() -> EditorGraph {
        let mut g = EditorGraph::default();
        let r = g.add(NodeKind::read(0, ""), Pos2::new(0.0, 0.0));
        let a = g.add(NodeKind::Transform(Default::default()), Pos2::new(0.0, 100.0));
        let b = g.add(NodeKind::Grade(Default::default()), Pos2::new(10.0, 200.0));
        let m = g.add(NodeKind::blend(), Pos2::new(0.0, 300.0));
        g.connect(a, 0, r);
        g.connect(b, 0, a);
        g.connect(m, 0, b);
        g
    }

    /// Every creation path's add: two selected nodes feed a new Blend
    /// (first selected OVER) and it lands centred under them, a tidy row
    /// below the lower; nothing selected leaves it where it was asked for.
    #[test]
    fn a_blend_made_from_two_selected_lands_wired_under_them() {
        let mut g = EditorGraph::default();
        let fg = g.add(NodeKind::read(0, ""), Pos2::new(-100.0, 0.0));
        let bg = g.add(NodeKind::read(1, ""), Pos2::new(100.0, 50.0));
        let mut sel = vec![fg, bg];
        let b = add_wired(&mut g, NodeKind::blend(), Pos2::new(900.0, 900.0), &mut sel, 1.0, false);
        assert_eq!(sel, vec![b], "the new node becomes the selection");
        assert_eq!(g.input_of(b, 0), Some(fg), "first selected goes OVER");
        assert_eq!(g.input_of(b, 1), Some(bg));
        let p = g.node(b).unwrap().pos;
        let step = 2.0 * NODE_R + crate::nodeux::ROW_GAP;
        assert!((p.x - 0.0).abs() < 0.5, "centred between its feeders: {p:?}");
        assert!((p.y - (50.0 + step)).abs() < 0.5, "a row under the lower feeder: {p:?}");
        // Nothing selected: the cursor's point stands.
        let mut none = Vec::new();
        let lone = add_wired(&mut g, NodeKind::blend(), Pos2::new(700.0, 700.0), &mut none, 1.0, false);
        assert_eq!(g.node(lone).unwrap().pos, Pos2::new(700.0, 700.0));
    }

    /// A Layer made from several picks stacks every one of them, in pick
    /// order, the first on top.
    #[test]
    fn a_layer_made_from_picks_stacks_them_in_pick_order() {
        let mut g = EditorGraph::default();
        let reads: Vec<u64> =
            (0..4).map(|i| g.add(NodeKind::read(i, ""), Pos2::new(i as f32 * 100.0, 0.0))).collect();
        let order = vec![reads[2], reads[0], reads[3], reads[1]];
        let mut sel = order.clone();
        let l = add_wired(&mut g, NodeKind::layer(), Pos2::ZERO, &mut sel, 1.0, false);
        let NodeKind::Layer(p) = &g.node(l).unwrap().kind else { panic!("not a stack") };
        assert_eq!(p.layers.len(), 4, "a layer per pick");
        for (i, r) in order.iter().enumerate() {
            assert_eq!(g.input_of(l, i), Some(*r), "layer {} is pick {i}", i + 1);
        }
        // One pick: the two-layer stack it always was, fed on top.
        let mut one = vec![reads[0]];
        let l1 = add_wired(&mut g, NodeKind::layer(), Pos2::ZERO, &mut one, 1.0, false);
        let NodeKind::Layer(p) = &g.node(l1).unwrap().kind else { panic!("not a stack") };
        assert_eq!(p.layers.len(), 2);
        assert_eq!(g.input_of(l1, 0), Some(reads[0]));
    }

    /// One selected mid-chain: the new node splices in and sits BETWEEN
    /// the two when there's room, beside the gap when there isn't — never
    /// below what it now feeds.
    #[test]
    fn a_spliced_add_sits_between_or_beside_never_below() {
        for (gap, beside) in [(400.0f32, false), (120.0, true)] {
            let mut g = EditorGraph::default();
            let a = g.add(NodeKind::read(0, ""), Pos2::new(0.0, 0.0));
            let c = g.add(NodeKind::Grade(Default::default()), Pos2::new(0.0, gap));
            g.connect(c, 0, a);
            let mut sel = vec![a];
            let n = add_wired(&mut g, NodeKind::Grade(Default::default()), Pos2::ZERO, &mut sel, 1.0, false);
            assert_eq!(g.input_of(c, 0), Some(n), "it took A's place above C");
            let p = g.node(n).unwrap().pos;
            assert!((p.y - gap * 0.5).abs() < 0.5, "midway down: {p:?} (gap {gap})");
            assert!(p.y < gap, "never below what it feeds");
            assert_eq!(p.x > 1.0, beside, "gap {gap}: beside = {beside}, got {p:?}");
        }
    }

    /// A spot already taken is stepped past, not stacked on.
    /// A Transform saved before shear and taper existed loads with neither.
    #[test]
    fn an_old_transform_loads_unbent() {
        let old = r#"{"translate":[3.0,4.0],"rotate":10.0,"scale":[1.0,1.0],"center":[0.5,0.5]}"#;
        let t: engine::graph::TransformParams = serde_json::from_str(old).unwrap();
        assert!(!t.bends());
        assert_eq!((t.shear, t.taper, t.taper_angle), ([0.0, 0.0], 0.0, 90.0));
    }

    /// A node added beside others takes THEIR size whatever the zoom; out
    /// on its own it keeps the size that reads the same on screen.
    #[test]
    fn a_new_node_matches_its_neighbours_and_alone_follows_the_zoom() {
        let mut g = EditorGraph::default();
        let small = g.add(NodeKind::Grade(Default::default()), Pos2::new(0.0, 0.0));
        let big = g.add(NodeKind::Grade(Default::default()), Pos2::new(2000.0, 0.0));
        g.node_mut(small).unwrap().size = 0.3;
        g.node_mut(big).unwrap().size = 3.0;
        let mut sel = Vec::new();
        for zoom in [0.5, 1.0, 4.0] {
            // A little way from each, in screen terms, at this zoom.
            let near = |x: f32| Pos2::new(x + 120.0 / zoom, 30.0 / zoom);
            let a = add_wired(&mut g, NodeKind::Gamma(Default::default()), near(0.0), &mut sel, zoom, false);
            assert_eq!(g.node(a).unwrap().size, 0.3, "beside the small chain at zoom {zoom}");
            sel.clear(); // nothing selected: an unfed add
            let b = add_wired(&mut g, NodeKind::Gamma(Default::default()), near(2000.0), &mut sel, zoom, false);
            assert_eq!(g.node(b).unwrap().size, 3.0, "beside the big chain at zoom {zoom}");
            g.remove(a);
            g.remove(b);
            sel.clear();
        }
        // Far from everything: the zoom decides.
        let lone = add_wired(&mut g, NodeKind::Gamma(Default::default()), Pos2::new(0.0, 9000.0), &mut sel, 2.0, false);
        assert_eq!(g.node(lone).unwrap().size, 0.5);
        // And the bridge a mistyped wire makes sits at its wire's size.
        let spline = g.add(NodeKind::Spline(Default::default()), Pos2::new(0.0, 200.0));
        g.node_mut(spline).unwrap().size = 0.3;
        let grade = g.add(NodeKind::Grade(Default::default()), Pos2::new(0.0, 300.0));
        g.node_mut(grade).unwrap().size = 0.3;
        let read = g.add(NodeKind::read(0, ""), Pos2::new(60.0, 200.0));
        g.node_mut(read).unwrap().size = 0.3;
        let m = g.node(grade).unwrap().kind.mask_input().unwrap();
        let bridge = connect_or_bridge(&mut g, read, 0, grade, m, 3.0).expect("a ToMatte");
        assert_eq!(g.node(bridge).unwrap().size, 0.3);
    }

    #[test]
    fn placement_steps_past_an_occupied_spot() {
        let mut g = EditorGraph::default();
        let fg = g.add(NodeKind::read(0, ""), Pos2::new(-100.0, 0.0));
        let bg = g.add(NodeKind::read(1, ""), Pos2::new(100.0, 0.0));
        let step = 2.0 * NODE_R + crate::nodeux::ROW_GAP;
        let squatter = g.add(NodeKind::Grade(Default::default()), Pos2::new(0.0, step));
        let mut sel = vec![fg, bg];
        let b = add_wired(&mut g, NodeKind::blend(), Pos2::ZERO, &mut sel, 1.0, false);
        let (pb, ps) = (g.node(b).unwrap().pos, g.node(squatter).unwrap().pos);
        assert!(pb.distance(ps) > 2.0 * NODE_R, "stacked on the squatter: {pb:?}");
        assert!(pb.y > ps.y, "stepped DOWN past it");
    }

    /// The M-swipe's wiring: dragged node OVER (+), swiped-onto node UNDER
    /// (−), and the Blend takes the under node's place downstream — except
    /// where that would close a loop, read a different output, or hand a
    /// picture to a matte input.
    #[test]
    fn a_merge_swipe_blends_over_and_takes_the_b_pipe() {
        let mut g = chain(); // R(0) -> A(1) -> B(2) -> M(3)
        let fg = g.add(NodeKind::read(1, ""), Pos2::new(300.0, 0.0));
        let b = swipe_op(&mut g, SwipeOp::Blend, fg, 1, Pos2::new(0.0, 150.0), 1.0);
        assert!(matches!(g.node(b).unwrap().kind, NodeKind::Blend { .. }));
        assert_eq!(g.input_of(b, 0), Some(fg), "the dragged node goes OVER");
        assert_eq!(g.input_of(b, 1), Some(1), "the swiped-onto node goes UNDER");
        assert_eq!(g.input_of(2, 0), Some(b), "the Blend takes A's place downstream");
        assert_eq!(g.node(b).unwrap().pos, Pos2::new(0.0, 150.0));

        // Swiping a node onto its OWN ancestor: rerouting the ancestor's
        // consumer (which leads to the dragged node) would be a loop.
        let mut g = chain();
        let b = swipe_op(&mut g, SwipeOp::Blend, 2, 1, Pos2::ZERO, 1.0);
        assert_eq!(g.input_of(b, 0), Some(2));
        assert_eq!(g.input_of(b, 1), Some(1));
        assert_eq!(g.input_of(2, 0), Some(1), "B still reads A: rerouting it would loop");
    }

    /// I and O: a Blend in REPLACE with the matte on its matte input —
    /// image on + with − empty is image × matte (mask), image on − with +
    /// empty is image × (1 − matte) (stencil) — labelled for a compositor's hand,
    /// and taking the image's place downstream like the M-swipe.
    #[test]
    fn in_and_out_swipes_wire_mask_and_stencil() {
        use engine::graph::BlendOp;
        for (op, img_port, label) in [(SwipeOp::In, 0usize, "mask"), (SwipeOp::Out, 1, "stencil")] {
            let mut g = chain(); // R(0) -> A(1) -> B(2) -> M(3)
            let matte = g.add(NodeKind::Spline(Default::default()), Pos2::new(200.0, 0.0));
            let b = swipe_op(&mut g, op, matte, 1, Pos2::new(0.0, 150.0), 1.0);
            let n = g.node(b).unwrap();
            assert!(matches!(n.kind, NodeKind::Blend { op: BlendOp::Replace, .. }), "{op:?}");
            assert_eq!(n.ui_label, label);
            assert_eq!(g.input_of(b, img_port), Some(1), "{op:?}: the image on its port");
            assert_eq!(g.input_of(b, 1 - img_port), None, "{op:?}: the other port stays empty");
            assert_eq!(g.input_of(b, 2), Some(matte), "{op:?}: the matte on the matte input");
            assert_eq!(g.input_of(2, 0), Some(b), "{op:?}: it takes A's place downstream");
        }
    }

    /// Inside a Noise node: a slider is a number, an fBm a pattern, a roto
    /// a shape, and maths takes the busiest of what it is fed. An fBm (six
    /// sockets) wears the big dot.
    #[test]
    fn noise_noodles_say_what_they_carry() {
        use engine::field::FieldNode as F;
        let mut g = EditorGraph::default();
        let num = g.add(NodeKind::Field(F::Const { value: 0.5, name: String::new() }), Pos2::ZERO);
        let fbm = g.add(NodeKind::Field(F::default()), Pos2::ZERO);
        let shape = g.add(NodeKind::Field(F::MaskIn), Pos2::ZERO);
        let math = |g: &mut EditorGraph| {
            g.add(NodeKind::Field(F::Math { op: engine::field::MathOp::Add, a: 0.0, b: 0.0 }), Pos2::ZERO)
        };
        let (m1, m2) = (math(&mut g), math(&mut g));
        g.connect(m1, 0, num);
        g.connect(m1, 1, shape);
        g.connect(m2, 0, m1);
        g.connect(m2, 1, fbm);
        let mut memo = std::collections::HashMap::new();
        let mut c = |id| field_wire_class(&g, id, &mut memo);
        assert_eq!(c(num), GeoWire::Number);
        assert_eq!(c(fbm), GeoWire::Field);
        assert_eq!(c(m1), GeoWire::Shapes);
        assert_eq!(c(m2), GeoWire::Field);
        assert!(big_dot(&NodeKind::Field(F::default())));
        assert!(!big_dot(&NodeKind::Field(F::Output)));
        // Every noise group has something in it, and nothing else shows.
        let cats: Vec<NodeCat> = node_catalog().into_iter().filter(|e| offered(e.cat, true, false, false, false)).map(|e| e.cat).collect();
        assert!(cats.iter().all(|c| c.is_field()));
        for c in NodeCat::ALL.into_iter().filter(|c| c.is_field()) {
            assert!(cats.contains(&c), "{} is empty", c.name());
        }
    }

    /// The legend names every colour a wire can wear — the cost pair, the
    /// lit wire, the pipes that carry their own colour, and the hand's
    /// states — read LIVE from the theme, so it can never show a colour the
    /// graph does not use or miss one it does.
    #[test]
    fn the_legend_names_every_wire_colour() {
        use crate::theme as t;
        let e = legend_entries();
        let has = |c: Color32, sw: Swatch| e.iter().any(|(_, ec, es)| *ec == c && *es == sw);
        assert!(has(t::cost_fast(), Swatch::Plain));
        assert!(has(t::cost_slow(), Swatch::Plain));
        assert!(has(t::cost_slow(), Swatch::Glow), "the heavy node's lit wire");
        assert!(has(t::wire_channels(), Swatch::Fat), "the fat multi-channel cable");
        for pipe in [PortKind::Mask, PortKind::Data] {
            let c = pipe_color(pipe).expect("a pipe with its own colour");
            assert!(has(c, Swatch::Plain), "{pipe:?}'s wire colour is in the legend");
        }
        for c in [t::hidden_wire(), t::wire_held(), t::wire_locked()] {
            assert!(has(c, Swatch::Plain));
        }
        let mut labels: Vec<&str> = e.iter().map(|(l, ..)| *l).collect();
        labels.sort_unstable();
        labels.dedup();
        assert_eq!(labels.len(), e.len(), "every row says something different");
    }

    /// A dot must be invisible to the engine: no node, no stage, and the
    /// SAME topology hash as the wire it was cut into. If this ever fails,
    /// every cached bake in every setup with a dot in it is thrown away.
    #[test]
    fn a_dot_is_wiring_not_a_node() {
        let plain = chain();
        let before = plain.to_engine_graph().topology_hash();
        let mut g = chain();
        // Cut a dot into R -> A (input 0 of node 1).
        let d = g.add(NodeKind::dot(PortKind::Image), Pos2::new(0.0, 50.0));
        g.connect(d, 0, 0);
        g.connect(1, 0, d);
        let e = g.to_engine_graph();
        assert!(!e.nodes.contains_key(&d), "a dot must never reach the engine");
        assert_eq!(e.edges.get(&(1, 0)), Some(&0), "the consumer sees the real source");
        assert_eq!(e.topology_hash(), before, "a dot must not recompile the comp");
    }

    /// A Crop that bakes itself is, to the engine, the live crop under a
    /// Bake that keeps the Crop's id: consumers and the viewer read the
    /// Bake, the crop reads the Crop's input. A Crop that never baked, or
    /// a bypassed one, reaches the engine as it always did.
    #[test]
    fn a_baking_crop_lowers_to_a_bake_over_the_live_crop() {
        let mut g = chain();
        let c = g.add(NodeKind::SoftCrop(Default::default()), Pos2::new(0.0, 50.0));
        g.connect(c, 0, 0);
        g.connect(1, 0, c);
        g.view = Some(c);
        let plain = g.to_engine_graph();
        assert!(matches!(plain.nodes.get(&c), Some(NodeKind::SoftCrop(_))));
        assert!(!plain.nodes.contains_key(&crop_live_id(c)));
        if let Some(ENode { kind: NodeKind::SoftCrop(p), .. }) = g.node_mut(c) {
            p.bake = Some(9);
            p.matte = true;
        }
        let e = g.to_engine_graph();
        let live = crop_live_id(c);
        assert!(matches!(e.nodes.get(&c), Some(NodeKind::GraphBake { source: 9, mask: true, .. })));
        assert!(matches!(e.nodes.get(&live), Some(NodeKind::SoftCrop(_))));
        assert_eq!(e.edges.get(&(c, 0)), Some(&live));
        assert_eq!(e.edges.get(&(live, 0)), Some(&0));
        assert_eq!(e.edges.get(&(1, 0)), Some(&c));
        assert_eq!(e.view, Some(c));
        g.node_mut(c).unwrap().disabled = true;
        assert!(!g.to_engine_graph().nodes.contains_key(&live), "a bypassed Crop is not lowered");
    }

    #[test]
    fn a_run_of_dots_resolves_and_an_unwired_one_reads_as_unplugged() {
        let mut g = chain();
        let d1 = g.add(NodeKind::dot(PortKind::Image), Pos2::new(0.0, 40.0));
        let d2 = g.add(NodeKind::dot(PortKind::Image), Pos2::new(0.0, 60.0));
        g.connect(d1, 0, 0);
        g.connect(d2, 0, d1);
        g.connect(1, 0, d2);
        assert_eq!(g.to_engine_graph().edges.get(&(1, 0)), Some(&0));
        // Pull the head off: the chain now leads nowhere, so the consumer's
        // input is unplugged rather than wired to a dot.
        g.disconnect_input(d1, 0);
        let e = g.to_engine_graph();
        assert_eq!(e.edges.get(&(1, 0)), None);
        assert!(!e.nodes.contains_key(&d1) && !e.nodes.contains_key(&d2));
    }

    /// A cycle of dots must not hang the conversion — the editor cannot
    /// author one, but a hand-edited setup can.
    #[test]
    fn a_ring_of_dots_terminates() {
        let mut g = EditorGraph::default();
        let a = g.add(NodeKind::dot(PortKind::Image), Pos2::ZERO);
        let b = g.add(NodeKind::dot(PortKind::Image), Pos2::ZERO);
        g.connect(a, 0, b);
        g.connect(b, 0, a);
        assert_eq!(g.resolve_dot(a), None);
        let _ = g.to_engine_graph();
    }

    #[test]
    fn the_viewer_parked_on_a_dot_shows_what_reaches_it() {
        let mut g = chain();
        let d = g.add(NodeKind::dot(PortKind::Image), Pos2::new(0.0, 250.0));
        g.connect(d, 0, 2);
        g.view = Some(d);
        assert_eq!(g.to_engine_graph().view, Some(2));
    }

    /// A dot cut into a MASK wire has to court mask outputs and draw in the
    /// mask colour; that follows the wire, and the wire can be rerouted.
    #[test]
    fn a_dot_wears_the_kind_of_the_wire_it_sits_on() {
        let mut g = EditorGraph::default();
        let sp = g.add(NodeKind::Spline(Default::default()), Pos2::ZERO);
        let img = g.add(NodeKind::read(0, ""), Pos2::ZERO);
        let d = g.add(NodeKind::dot(PortKind::Image), Pos2::ZERO);
        g.connect(d, 0, sp);
        g.sync_dot_kinds();
        assert_eq!(g.node(d).unwrap().kind.output_kind(), PortKind::Mask);
        g.connect(d, 0, img);
        g.sync_dot_kinds();
        assert_eq!(g.node(d).unwrap().kind.output_kind(), PortKind::Image);
    }

    /// Swapping a Blend's + and −: the wires exchange ports. A half-wired
    /// merge swaps too — that is "put it on the other input", which is what
    /// the gesture is mostly used for.
    #[test]
    fn a_merge_is_one_bar_with_no_holes() {
        use engine::geo::GeoNode;
        let mut g = EditorGraph::default();
        let a = g.add(NodeKind::GeoNode(GeoNode::default()), Pos2::ZERO);
        let b = g.add(NodeKind::GeoNode(GeoNode::default()), Pos2::ZERO);
        let m = g.add(NodeKind::GeoNode(GeoNode::Merge { count: 4 }), Pos2::new(0.0, 100.0));
        assert!(multi_input(&g.node(m).unwrap().kind));
        g.connect(m, 1, a);
        g.connect_out(m, 3, b, 2);
        g.connect_hidden(m, 3, b);
        g.edge_out.insert((m, 3), 2);
        // Wires pulled out of the middle close up, side tables and all.
        assert_eq!(g.pack_inputs(m), 2);
        assert_eq!(g.input_of(m, 0), Some(a));
        assert_eq!(g.input_of(m, 1), Some(b));
        assert_eq!(g.out_of(m, 1), 2);
        assert!(g.is_hidden(m, 1) && !g.is_hidden(m, 3));
        // Wired ends spread along the bar in order; the free one is its
        // centre, and the bar stays short however many wires pile on.
        let (p, n) = (Pos2::new(0.0, 100.0), 3);
        let (l, r, free) = (multi_rest(p, 0, n, 1.0), multi_rest(p, 1, n, 1.0), multi_rest(p, 2, n, 1.0));
        assert!(l.x < free.x && free.x < r.x && l.y == free.y && r.y == free.y);
        assert!(multi_half(engine::geo::MAX_MERGE) < 60.0);
        // A Blend 3D is one bar too: its cards are depth-sorted.
        assert!(multi_input(&NodeKind::Scene3D(Default::default())));
        // Other kinds keep their nodules.
        assert!(!multi_input(&NodeKind::blend()));
    }

    #[test]
    fn swapping_a_blends_inputs_exchanges_the_wires() {
        let mut g = EditorGraph::default();
        let fg = g.add(NodeKind::read(0, ""), Pos2::ZERO);
        let bg = g.add(NodeKind::read(1, ""), Pos2::ZERO);
        let m = g.add(NodeKind::blend(), Pos2::new(0.0, 100.0));
        g.connect(m, 0, fg);
        g.connect(m, 1, bg);
        assert_eq!(swappable_inputs(&g.node(m).unwrap().kind), Some((0, 1)));
        assert!(g.swap_inputs(m, 0, 1));
        assert_eq!(g.input_of(m, 0), Some(bg));
        assert_eq!(g.input_of(m, 1), Some(fg));
        // Half-wired: the wire moves and the vacated port is left free.
        g.disconnect_input(m, 1);
        assert!(g.swap_inputs(m, 0, 1));
        assert_eq!(g.input_of(m, 0), None);
        assert_eq!(g.input_of(m, 1), Some(bg));
        // Nothing wired at all is not an edit.
        g.disconnect_input(m, 1);
        assert!(!g.swap_inputs(m, 0, 1));
        // And a mask port is never offered: that is not a flip.
        let bl = g.add(NodeKind::Blur(Default::default()), Pos2::ZERO);
        assert_eq!(swappable_inputs(&g.node(bl).unwrap().kind), None);
    }

    #[test]
    fn swap_up_rewires_and_exchanges_positions() {
        let mut g = chain();
        let (pa, pb) = (g.node(1).unwrap().pos, g.node(2).unwrap().pos);
        let deltas = g.swap_up(2).expect("swap should apply");
        // R -> B -> A -> M
        assert_eq!(g.input_of(2, 0), Some(0));
        assert_eq!(g.input_of(1, 0), Some(2));
        assert_eq!(g.input_of(3, 0), Some(1));
        assert_eq!(g.node(2).unwrap().pos, pa);
        assert_eq!(g.node(1).unwrap().pos, pb);
        // Animation deltas point from the new position back at the old one.
        let d: std::collections::HashMap<u64, Vec2> = deltas.into_iter().collect();
        assert_eq!(d[&2], pb - pa);
        assert_eq!(d[&1], pa - pb);
        // ⌘↓ (= swap_up of the child) restores the original chain.
        g.swap_up(1).expect("swap back");
        assert_eq!(g.input_of(1, 0), Some(0));
        assert_eq!(g.input_of(2, 0), Some(1));
        assert_eq!(g.input_of(3, 0), Some(2));
    }

    #[test]
    fn transient_evaluation_applies_curves_and_leaves_authored_intact() {
        let mut g = EditorGraph::default();
        let id = g.add(NodeKind::Transform(Default::default()), Pos2::ZERO);
        let n = g.node_mut(id).unwrap();
        let c = n.anim.entry(crate::anim::Channel::TranslateX).or_default();
        c.set_key(0, 0.0);
        c.set_key(10, 100.0);
        // Evaluated at t=5: engine sees 50; authored value stays 0.
        let eg = g.to_engine_graph_at(Some(5));
        match eg.nodes.get(&id) {
            Some(NodeKind::Transform(t)) => assert_eq!(t.translate[0], 50.0),
            other => panic!("wrong kind: {other:?}"),
        }
        match &g.node(id).unwrap().kind {
            NodeKind::Transform(t) => assert_eq!(t.translate[0], 0.0),
            _ => unreachable!(),
        }
        // No-time build = authored values.
        let eg0 = g.to_engine_graph();
        match eg0.nodes.get(&id) {
            Some(NodeKind::Transform(t)) => assert_eq!(t.translate[0], 0.0),
            _ => unreachable!(),
        }
    }

    #[test]
    fn speed_node_pivots_around_its_anchor() {
        use engine::graph::SpeedParams;
        // Half speed anchored at 10: frame 10 stays put, everything else
        // stretches away from it.
        let p = SpeedParams { speed: 0.5, anchor: 10.0, interp: Default::default(), blend_w: 0.0 };
        let m = p.map();
        assert_eq!(m.apply_f(10.0), 10.0, "the anchor frame must not move");
        assert_eq!(m.apply_f(20.0), 15.0);
        assert_eq!(m.apply_f(0.0), 5.0);
        // Speed 1 is identity whatever the anchor.
        let id = SpeedParams { speed: 1.0, anchor: 7.0, interp: Default::default(), blend_w: 0.0 }.map();
        for t in [0.0f32, 3.0, 99.0] {
            assert_eq!(id.apply_f(t), t);
        }
        // Negative runs backwards through the anchor.
        let rev = SpeedParams { speed: -1.0, anchor: 10.0, interp: Default::default(), blend_w: 0.0 }.map();
        assert_eq!(rev.apply_f(10.0), 10.0);
        assert_eq!(rev.apply_f(12.0), 8.0);
    }

    /// A Grade's ranges preview is armed by the LOWERING, for the view
    /// target alone. That is the whole safety argument: a node that is the
    /// view root has nothing downstream, so a preview cannot reach a
    /// consumer, a render or a bake (which re-roots `view` at what it
    /// bakes) — and the authored graph never carries the arming at all.
    #[test]
    fn a_zone_preview_is_armed_only_for_the_view_target() {
        use engine::graph::{GradeParams, GradeZones, ZonePreview, ZoneView};
        let zoned = || {
            let mut z = GradeZones::default();
            z.preview = ZonePreview { view: ZoneView::Matte, zone: 1 };
            NodeKind::Grade(GradeParams { zones: Some(z), ..Default::default() })
        };
        let mut g = EditorGraph::default();
        let up = g.add(zoned(), Pos2::ZERO);
        let down = g.add(zoned(), Pos2::ZERO);
        g.edges.push((down, 0, up));
        g.view = Some(down);

        let eg = g.to_engine_graph();
        let armed = |id: u64| match &eg.nodes[&id] {
            NodeKind::Grade(gr) => gr.zones.as_ref().unwrap().previewing,
            other => panic!("not a grade: {other:?}"),
        };
        assert!(armed(down), "the view target previews");
        assert!(!armed(up), "a node upstream of the view does not");
        // The authored graph carries the MODE but never the arming, so
        // nothing that reads authored data can see a preview at all.
        let authored = match &g.nodes[&down].kind {
            NodeKind::Grade(gr) => gr.zones.as_ref().unwrap().clone(),
            _ => unreachable!(),
        };
        assert_eq!(authored.preview.view, ZoneView::Matte);
        assert!(!authored.previewing);
    }

    #[test]
    fn timeremap_curve_drives_time_and_remaps_upstream_animation() {
        use engine::graph::TimeRemapParams;
        // Transform (animated) -> TimeWarp (half speed) -> view.
        let mut g = EditorGraph::default();
        let tf = g.add(NodeKind::Transform(Default::default()), Pos2::ZERO);
        // The remap's own animated frame IS the curve: 0 -> 0, 100 -> 50,
        // i.e. half speed.
        let tw = g.add(NodeKind::TimeRemap(TimeRemapParams::default()), Pos2::ZERO);
        {
            let c = g
                .node_mut(tw)
                .unwrap()
                .anim
                .entry(crate::anim::Channel::TimeRemapFrame)
                .or_default();
            c.set_key(0, 0.0);
            c.set_key(100, 50.0);
        }
        g.connect(tw, 0, tf);
        g.view = Some(tw);
        let c = g.node_mut(tf).unwrap().anim.entry(crate::anim::Channel::TranslateX).or_default();
        c.set_key(0, 0.0);
        c.set_key(10, 100.0);

        let x_at = |g: &EditorGraph, t: u64| match g.to_engine_graph_at(Some(t)).nodes.get(&tf) {
            Some(NodeKind::Transform(t)) => t.translate[0],
            other => panic!("wrong kind: {other:?}"),
        };
        // Tick 10 reads source 5, so the curve gives 50 — not the 100 the
        // unwarped clock would.
        assert_eq!(x_at(&g, 10), 50.0);
        // Tick 5 lands on source 2.5: a whole-frame clock could not express
        // this at all, and it must not snap to 20 or 30.
        assert_eq!(x_at(&g, 5), 25.0);
        // Past the curve's last point it extrapolates along the end slope:
        // tick 40 -> source 20, where the animation holds its last key.
        assert_eq!(x_at(&g, 40), 100.0);

        // The warp itself is not upstream of anything, so an animated node
        // BELOW it keeps the global clock.
        let mut g2 = EditorGraph::default();
        let a = g2.add(NodeKind::Transform(Default::default()), Pos2::ZERO);
        let w = g2.add(NodeKind::TimeRemap(TimeRemapParams::default()), Pos2::ZERO);
        {
            let c = g2
                .node_mut(w)
                .unwrap()
                .anim
                .entry(crate::anim::Channel::TimeRemapFrame)
                .or_default();
            c.set_key(0, 0.0);
            c.set_key(100, 50.0);
        }
        let b = g2.add(NodeKind::Transform(Default::default()), Pos2::ZERO);
        g2.connect(w, 0, a);
        g2.connect(b, 0, w);
        g2.view = Some(b);
        for id in [a, b] {
            let c = g2.node_mut(id).unwrap().anim.entry(crate::anim::Channel::TranslateX).or_default();
            c.set_key(0, 0.0);
            c.set_key(10, 100.0);
        }
        let eg = g2.to_engine_graph_at(Some(10));
        let get = |id: u64| match eg.nodes.get(&id) {
            Some(NodeKind::Transform(t)) => t.translate[0],
            other => panic!("wrong kind: {other:?}"),
        };
        assert_eq!(get(b), 100.0, "below the warp stays on the global clock");
        assert_eq!(get(a), 50.0, "above the warp runs on warped time");
    }

    #[test]
    fn swap_requires_one_in_one_out_on_both_nodes() {
        let mut g = chain();
        // A Read (no inputs) can't move up.
        assert!(g.swap_up(0).is_none());
        // The tail Merge has no consumer: not 1-out.
        assert!(g.swap_up(3).is_none());
        // A Merge parent with two connected inputs blocks the swap.
        let r2 = g.add(NodeKind::read(1, ""), Pos2::ZERO);
        g.connect(3, 1, r2);
        let tail = g.add(NodeKind::Grade(Default::default()), Pos2::ZERO);
        g.connect(tail, 0, 3);
        assert!(g.swap_up(tail).is_none(), "two-input merge parent is ambiguous");
        // But a straight 1-1 pair inside the same graph still swaps.
        assert!(g.swap_up(2).is_some());
    }
}

#[cfg(test)]
mod extract_tests {
    use super::*;

    /// A plate on the left, three mattes above, three grades below.
    fn scene() -> (EditorGraph, u64) {
        let mut g = EditorGraph::default();
        g.nodes.clear();
        g.edges.clear();
        g.next_id = 0;
        let plate = g.add(NodeKind::read(0, ""), Pos2::new(-260.0, 0.0));
        let ex = g.add(NodeKind::Extract(engine::graph::ExtractParams { n: 3 }), Pos2::new(0.0, 0.0));
        g.connect(ex, 0, plate);
        for k in 0..3 {
            let m = g.add(NodeKind::Spline(Default::default()), Pos2::new(-120.0 + 120.0 * k as f32, -220.0));
            g.connect(ex, 1 + k, m);
            let gr = g.add(NodeKind::Grade(Default::default()), Pos2::new(-120.0 + 120.0 * k as f32, 220.0));
            g.connect_out(gr, 0, ex, k as u32);
        }
        (g, ex)
    }

    #[test]
    fn a_click_on_a_switch_throws_its_lever() {
        use super::{switch_angle, switch_pick};
        let c = Pos2::new(100.0, 100.0);
        let r = 20.0;
        let at = |i: usize, n: usize| {
            let a = switch_angle(i, n);
            c + Vec2::new(a.cos(), -a.sin()) * r * 0.8
        };
        // A click on a stud picks that input, whatever is live.
        for n in 2..=5 {
            for i in 0..n {
                assert_eq!(switch_pick(c, r, n, (i + 1) % n, at(i, n)), i, "stud {i} of {n}");
            }
        }
        // The hub and the lower half step to the next input, wrapping.
        assert_eq!(switch_pick(c, r, 3, 0, c), 1);
        assert_eq!(switch_pick(c, r, 3, 2, c + Vec2::new(0.0, r * 0.6)), 0);
        // Input 0 is up-left, the last up-right.
        assert!(at(0, 2).x < c.x && at(1, 2).x > c.x && at(0, 2).y < c.y);
    }

    #[test]
    fn it_lowers_to_one_mask_per_output_and_draws_its_plus() {
        let (mut g, ex) = scene();
        let eg = g.to_engine_graph();
        let masks = eg.nodes.values().filter(|k| matches!(k, NodeKind::Mask)).count();
        assert_eq!(masks, 3);
        assert!(!eg.nodes.values().any(|k| matches!(k, NodeKind::Extract(_))));

        let mut ed = NodeEditor::default();
        ed.zoom = 1.0;
        let (mut sel, mut act, mut pin, mut slot) = (Vec::new(), Vec::new(), Vec::new(), 2usize);
        let mut rects = Vec::new();
        let shot = crate::snapshot::shoot(760.0, 620.0, |ui| {
            ed.show(ui, &mut g, &mut sel, &mut act, &mut pin, &mut slot, &mut rects);
        });
        shot.save("extract-node");
        assert!(ed.plus_hits.iter().any(|(id, _)| *id == ex), "the + must be clickable");
    }
}

#[cfg(test)]
mod cost_look_tests {
    use super::*;

    /// Build a graph with one HEAVY branch (Convolve), one merely slow one
    /// (Blur) and a speck, and hand back the editor's node rects.
    fn cost_scene(zoom: f32) -> (crate::snapshot::Shot, Vec<(u64, Rect)>, [u64; 4]) {
        let mut g = EditorGraph::default();
        g.nodes.clear();
        g.edges.clear();
        g.next_id = 0;
        let ra = g.add(NodeKind::read(0, ""), Pos2::new(-220.0, -260.0));
        let heavy = DefocusParams { mode: engine::graph::DefocusMode::Accurate, ..Default::default() };
        let conv = g.add(NodeKind::Defocus(heavy), Pos2::new(-220.0, -60.0));
        let gr = g.add(NodeKind::Grade(Default::default()), Pos2::new(-220.0, 180.0));
        g.connect(conv, 0, ra);
        g.connect(gr, 0, conv);
        let rb = g.add(NodeKind::read(1, ""), Pos2::new(120.0, -260.0));
        let blur = g.add(NodeKind::Blur(Default::default()), Pos2::new(120.0, -60.0));
        let gr2 = g.add(NodeKind::Grade(Default::default()), Pos2::new(120.0, 180.0));
        g.connect(blur, 0, rb);
        g.connect(gr2, 0, blur);
        let speck = g.add(NodeKind::Grade(Default::default()), Pos2::new(330.0, 60.0));
        g.node_mut(speck).unwrap().size = 0.08;
        let mut ed = NodeEditor::default();
        ed.zoom = zoom;
        ed.cost_flow = crate::costviz::analyse(&g, &|_| false);
        let (mut sel, mut act, mut pin, mut slot) = (Vec::new(), Vec::new(), Vec::new(), 2usize);
        let mut rects = Vec::new();
        let shot = crate::snapshot::shoot(760.0, 620.0, |ui| {
            ed.show(ui, &mut g, &mut sel, &mut act, &mut pin, &mut slot, &mut rects);
        });
        (shot, rects, [conv, blur, gr, speck])
    }

    /// A REALLY slow node lights the graph around it, and a merely slow one
    /// does not: the halo has to be light on the background, not a faint
    /// wash nobody can see. Measured in pixels, because "it glows" is
    /// exactly the claim a stack of alpha tweaks can quietly lose.
    #[test]
    fn a_heavy_node_lights_the_graph_around_it() {
        let (shot, rects, [conv, blur, ..]) = cost_scene(1.0);
        let at = |id: u64| rects.iter().find(|(i, _)| *i == id).map(|(_, r)| *r).unwrap();
        let (cr, br) = (at(conv), at(blur));
        // Sampled in RADII, at fractions of the halo's own reach, so
        // tightening the glow cannot quietly turn "it falls off" into "it
        // was never there": the shape of the falloff is what is pinned.
        let side = |r: Rect, k: f32| shot.at(r.center().x + r.width() * 0.5 * k, r.center().y);
        let (hot, cool) = (side(cr, HEAVY_REACH * 0.55), side(br, HEAVY_REACH * 0.55));
        let bg = shot.at(20.0, 600.0);
        assert!(
            hot[0] as i32 > cool[0] as i32 + 20,
            "the heavy node's halo ({hot:?}) must outshine a merely slow one ({cool:?})"
        );
        assert!(
            hot[0] as i32 > bg[0] as i32 + 30,
            "the heavy node's halo ({hot:?}) must lift the background ({bg:?})"
        );
        // And it falls off smoothly, never as a hard band: brightest at the
        // node, dimmer further out, gone by the time it is off the halo.
        let (mid, far) = (side(cr, HEAVY_REACH * 0.8), side(cr, HEAVY_REACH * 1.25));
        assert!(
            hot[0] > mid[0] && mid[0] > far[0] && (far[0] as i32) < bg[0] as i32 + 4,
            "the halo must fade outward: {hot:?} -> {mid:?} -> {far:?} over {bg:?}"
        );
    }

    /// The wires below it are lit too — that is what makes the COST of a
    /// comp readable from the wiring rather than from one node.
    #[test]
    fn heavy_wires_are_lit_beside_the_noodle() {
        let (shot, rects, [conv, blur, gr, _]) = cost_scene(1.0);
        let at = |id: u64| rects.iter().find(|(i, _)| *i == id).map(|(_, r)| *r).unwrap();
        // Beside the wire, clear of the noodle's own 2px stroke.
        let beside = |a: Rect, b: Rect| {
            let mid = (a.center() + b.center().to_vec2()) * 0.5;
            shot.at(mid.x + 7.0, mid.y)
        };
        let lit = beside(at(conv), at(gr));
        let plain = beside(at(blur), at(blur).translate(egui::vec2(0.0, 240.0)));
        assert!(
            lit[0] as i32 > plain[0] as i32 + 15,
            "a wire under a heavy node ({lit:?}) must glow beside a plain slow one ({plain:?})"
        );
    }

    /// A node scaled down to a speck twinkles: bright light where the shape
    /// is too small to see.
    #[test]
    fn a_speck_twinkles() {
        let (shot, rects, [.., speck]) = cost_scene(1.0);
        let r = rects.iter().find(|(i, _)| *i == speck).map(|(_, r)| *r).unwrap();
        let c = r.center();
        let bg = shot.at(20.0, 600.0);
        let lit = shot.at(c.x + 2.0, c.y);
        assert!(
            lit.iter().all(|v| *v as i32 > bg[0] as i32 + 25),
            "a speck must read as light ({lit:?}) against the background ({bg:?})"
        );
    }

    /// The node graph as it really draws, for LOOKING at the cost visuals:
    /// a heavy chain (Read -> Convolve -> Grade -> Blend), a merely slow one
    /// (Blur), a fast one, and nodes scaled down until they are specks.
    /// `cargo test -p app graph_cost_render -- --ignored`
    #[test]
    #[ignore]
    fn graph_cost_render() {
        let read = |slot: usize| NodeKind::read(slot, "");
        let mut g = EditorGraph::default();
        g.nodes.clear();
        g.edges.clear();
        g.next_id = 0;
        // Heavy: an accurate Defocus, with a tail below it.
        let ra = g.add(read(0), Pos2::new(-260.0, -300.0));
        let heavy = DefocusParams { mode: engine::graph::DefocusMode::Accurate, ..Default::default() };
        let conv = g.add(NodeKind::Defocus(heavy), Pos2::new(-260.0, -120.0));
        let gr = g.add(NodeKind::Grade(Default::default()), Pos2::new(-260.0, 60.0));
        g.connect(conv, 0, ra);
        g.connect(gr, 0, conv);
        // Slow but not heavy: a Blur, for the contrast.
        let rb = g.add(read(1), Pos2::new(60.0, -300.0));
        let blur = g.add(NodeKind::Blur(Default::default()), Pos2::new(60.0, -120.0));
        let gr2 = g.add(NodeKind::Grade(Default::default()), Pos2::new(60.0, 60.0));
        g.connect(blur, 0, rb);
        g.connect(gr2, 0, blur);
        // The two meet, so the heavy taint runs on downstream.
        let blend = g.add(NodeKind::blend(), Pos2::new(-100.0, 240.0));
        g.connect(blend, 0, gr);
        g.connect(blend, 1, gr2);
        g.view = Some(blend);
        // Specks: scaled down until they are hard to see at all.
        for (n, size) in [(0usize, 0.30f32), (1, 0.18), (2, 0.10), (3, 0.06)] {
            let id = g.add(NodeKind::Grade(Default::default()), Pos2::new(300.0, -180.0 + n as f32 * 150.0));
            g.node_mut(id).unwrap().size = size;
            let t = g.add(NodeKind::Transform(Default::default()), Pos2::new(470.0, -180.0 + n as f32 * 150.0));
            g.node_mut(t).unwrap().size = size;
            g.connect(t, 0, id);
        }
        let mut ed = NodeEditor::default();
        ed.zoom = 0.9;
        ed.cost_flow = crate::costviz::analyse(&g, &|_| false);
        ed.test_hold_keys_open();
        ed.test_hold_legend_open();
        let (mut sel, mut act, mut pin, mut slot, mut rects) =
            (Vec::new(), Vec::new(), Vec::new(), 2usize, Vec::new());
        crate::uishot::render("graph-cost", 900.0, 620.0, 2.0, 6, |ui, _| {
            ed.show(ui, &mut g, &mut sel, &mut act, &mut pin, &mut slot, &mut rects);
        });
    }

    /// LOOK at right-angle wires: `cargo test -p app graph_ortho_render --
    /// --ignored` writes target/graph-ortho.png — a Z between nodes, a mask
    /// wire into a side socket, a route through two dots, a wire doubling
    /// back up, and a fat cable's three strands.
    /// LOOK at the range view: `cargo test -p app graph_ranges_render --
    /// --ignored` writes `target/graph-ranges.png` — two plates of
    /// different lengths, one slipped, merged (the merge widens: bright),
    /// a Grade carrying it (dim) and a generator on the script range.
    #[test]
    #[ignore]
    fn graph_ranges_render() {
        let mut g = EditorGraph::default();
        g.nodes.clear();
        g.edges.clear();
        g.next_id = 0;
        let place = |g: &mut EditorGraph, id: u64, a: i64, b: i64| {
            if let Some(NodeKind::Read { first, last, start, .. }) = g.node_mut(id).map(|n| &mut n.kind) {
                (*start, *first, *last) = (Some(a), Some(a), Some(b));
            }
        };
        let ra = g.add(NodeKind::read(0, ""), Pos2::new(-260.0, -260.0));
        place(&mut g, ra, 1001, 1100);
        let rb = g.add(NodeKind::read(1, ""), Pos2::new(60.0, -260.0));
        place(&mut g, rb, 1001, 1048);
        let sl = g.add(
            NodeKind::Slip(engine::graph::SlipParams { offset: -20 }),
            Pos2::new(60.0, -110.0),
        );
        g.connect(sl, 0, rb);
        let blend = g.add(NodeKind::blend(), Pos2::new(-100.0, 40.0));
        g.connect(blend, 0, ra);
        g.connect(blend, 1, sl);
        let gr = g.add(NodeKind::Grade(Default::default()), Pos2::new(-100.0, 180.0));
        g.connect(gr, 0, blend);
        let grid = g.add(NodeKind::Grid(Default::default()), Pos2::new(260.0, 40.0));
        let mu = g.add(NodeKind::Mult(Default::default()), Pos2::new(260.0, 180.0));
        g.connect(mu, 0, grid);
        g.view = Some(gr);
        let all = crate::ranges::all(&crate::ranges::Ctx {
            graph: &g,
            script: (1001, 1100),
            media: Box::new(|_| None),
        });
        let mut ed = NodeEditor::default();
        ed.zoom = 0.9;
        ed.show_keys = false;
        ed.show_legend = false;
        ed.show_ranges = true;
        ed.script_range = (1001, 1100);
        ed.wire_ranges = all.iter().map(|(id, (r, _))| (*id, *r)).collect();
        ed.wire_range_src = all.iter().map(|(id, (_, s))| (*id, *s)).collect();
        assert!(ed.range_set_here(&g, blend), "the merge widens its range");
        assert!(!ed.range_set_here(&g, gr), "the Grade only carries it");
        let (mut sel, mut act, mut pin, mut slot, mut rects) =
            (Vec::new(), Vec::new(), Vec::new(), 4usize, Vec::new());
        crate::uishot::render("graph-ranges", 900.0, 640.0, 2.0, 6, |ui, _| {
            ed.show(ui, &mut g, &mut sel, &mut act, &mut pin, &mut slot, &mut rects);
        });
    }

    #[test]
    #[ignore]
    fn graph_ortho_render() {
        let mut g = EditorGraph::default();
        g.nodes.clear();
        g.edges.clear();
        g.next_id = 0;
        let ra = g.add(NodeKind::read(0, ""), Pos2::new(-300.0, -260.0));
        let gr = g.add(NodeKind::Grade(Default::default()), Pos2::new(-180.0, -60.0));
        let sp = g.add(NodeKind::Spline(Default::default()), Pos2::new(-420.0, -110.0));
        g.connect(gr, 0, ra);
        if let Some(m) = g.node(gr).and_then(|n| n.kind.mask_input()) {
            g.connect(gr, m, sp);
        }
        let rb = g.add(NodeKind::read(1, ""), Pos2::new(40.0, -260.0));
        let blur = g.add(NodeKind::Blur(Default::default()), Pos2::new(40.0, -80.0));
        g.connect(blur, 0, rb);
        let blend = g.add(NodeKind::blend(), Pos2::new(-100.0, 120.0));
        g.connect(blend, 0, gr);
        g.connect(blend, 1, blur);
        let wr = g.add(NodeKind::Write(Default::default()), Pos2::new(-100.0, 290.0));
        g.connect(wr, 0, blend);
        // Through two dots: the dots are the corners.
        let rc = g.add(NodeKind::read(2, ""), Pos2::new(260.0, -260.0));
        let d1 = g.add(NodeKind::dot(PortKind::Image), Pos2::new(260.0, -150.0));
        let d2 = g.add(NodeKind::dot(PortKind::Image), Pos2::new(400.0, -150.0));
        let tr = g.add(NodeKind::Transform(Default::default()), Pos2::new(400.0, 20.0));
        g.connect(d1, 0, rc);
        g.connect(d2, 0, d1);
        g.connect(tr, 0, d2);
        // Doubling back: the Mult sits above what feeds it.
        let mu = g.add(NodeKind::Mult(Default::default()), Pos2::new(240.0, 60.0));
        g.connect(mu, 0, tr);
        // The fat cable, into a ChannelSelect off to the side.
        let rm = g.add(NodeKind::read(3, ""), Pos2::new(-480.0, 120.0));
        if let Some(NodeKind::Read { multi, .. }) = g.node_mut(rm).map(|n| &mut n.kind) {
            *multi = true;
        }
        let cs = g.add(
            NodeKind::ChannelSelect(engine::graph::ChannelSelectParams { source: 3, layer: String::new() }),
            Pos2::new(-330.0, 290.0),
        );
        g.connect(cs, 0, rm);
        g.view = Some(wr);
        // An expression link, Transform -> Grade: the value-links overlay
        // here, the live cable in link mode below.
        g.patch.link(PSrc::Chan(tr, crate::anim::Channel::ScaleX), PDst::Chan(gr, crate::anim::Channel::GainR));
        let mut ed = NodeEditor::default();
        ed.zoom = 0.85;
        ed.ortho_wires = true;
        ed.show_keys = false;
        ed.show_legend = false;
        ed.show_patch_links = true;
        let (mut sel, mut act, mut pin, mut slot, mut rects) =
            (Vec::new(), Vec::new(), Vec::new(), 4usize, Vec::new());
        crate::uishot::render("graph-ortho", 1000.0, 680.0, 2.0, 6, |ui, _| {
            ed.show(ui, &mut g, &mut sel, &mut act, &mut pin, &mut slot, &mut rects);
        });
        ed.show_patch_links = false;
        ed.link_mode = true;
        crate::uishot::render("graph-ortho-links", 1000.0, 680.0, 2.0, 6, |ui, _| {
            ed.show(ui, &mut g, &mut sel, &mut act, &mut pin, &mut slot, &mut rects);
        });
    }
}

#[cfg(test)]
mod lanes_render {
    use super::*;

    /// A Layer fed by four branches off one row, and a Read feeding two of
    /// them: the right-angle tidy and the lanes, drawn to
    /// `target/graph-lanes-*.png` (before tidy, tidy without lanes, tidy
    /// with lanes).
    #[test]
    #[ignore]
    fn graph_lanes_render() {
        let mut g = EditorGraph::default();
        g.nodes.clear();
        g.edges.clear();
        g.next_id = 0;
        let layer = g.add(NodeKind::layer(), Pos2::new(60.0, 160.0));
        if let Some(NodeKind::Layer(p)) = g.node_mut(layer).map(|n| &mut n.kind) {
            p.layers = vec![Default::default(); 4];
        }
        let mut prev = None;
        for i in 0..4 {
            let x = -330.0 + i as f32 * 150.0;
            let rd = g.add(NodeKind::read(i, ""), Pos2::new(x, -260.0));
            let gr = g.add(NodeKind::Grade(Default::default()), Pos2::new(x + 10.0, -120.0));
            g.connect(gr, 0, rd);
            g.connect(layer, i as usize, gr);
            prev = Some(rd);
        }
        let blur = g.add(NodeKind::Blur(Default::default()), Pos2::new(330.0, -60.0));
        g.connect(blur, 0, prev.unwrap());
        let mu = g.add(NodeKind::Mult(Default::default()), Pos2::new(330.0, 160.0));
        g.connect(mu, 0, blur);
        let blend = g.add(NodeKind::blend(), Pos2::new(160.0, 300.0));
        g.connect(blend, 0, layer);
        g.connect(blend, 1, mu);
        let wr = g.add(NodeKind::Write(Default::default()), Pos2::new(160.0, 420.0));
        g.connect(wr, 0, blend);
        g.view = Some(wr);
        let mut ed = NodeEditor::default();
        ed.zoom = 0.7;
        ed.ortho_wires = true;
        ed.show_keys = false;
        ed.show_legend = false;
        let (mut sel, mut act, mut pin, mut slot, mut rects) =
            (Vec::new(), Vec::new(), Vec::new(), 4usize, Vec::new());
        let mut shot = |name: &str, g: &mut EditorGraph, ed: &mut NodeEditor| {
            ed.stalks.clear();
            crate::uishot::render(name, 1000.0, 760.0, 2.0, 6, |ui, _| {
                ed.show(ui, g, &mut sel, &mut act, &mut pin, &mut slot, &mut rects);
            });
        };
        shot("graph-lanes-before", &mut g, &mut ed);
        assert!(crate::nodeux::auto_layout_for(&mut g, &[], true));
        // Centre the tidied graph.
        let bb = g.nodes.values().fold(Rect::NOTHING, |r, n| r.union(Rect::from_center_size(n.pos, Vec2::splat(1.0))));
        let c = bb.center().to_vec2();
        for n in g.nodes.values_mut() {
            n.pos -= c;
        }
        ed.anim.clear();
        LANES_OFF.with(|l| l.set(true));
        shot("graph-lanes-tidy-off", &mut g, &mut ed);
        LANES_OFF.with(|l| l.set(false));
        shot("graph-lanes-tidy", &mut g, &mut ed);
        set_boring(true);
        ed.ortho_wires = false;
        shot("graph-boring", &mut g, &mut ed);
        set_boring(false);
    }

    /// Four Loads — full, red only, 8-bit, luma at 8 bits (the last
    /// through a Dot) — each into a Grade: the reduced-feed wires, drawn to
    /// `target/graph-reduced.png` and, in boring mode, `-boring.png`.
    #[test]
    #[ignore]
    fn graph_reduced_render() {
        use engine::decode::SingleChannel as S;
        let mut g = EditorGraph::default();
        g.nodes.clear();
        g.edges.clear();
        g.next_id = 0;
        let cases = [(S::Off, false), (S::Red, false), (S::Off, true), (S::Luma, true)];
        let mut fed = Vec::new();
        for (i, (s, e)) in cases.into_iter().enumerate() {
            let x = -300.0 + i as f32 * 200.0;
            let rd = g.add(NodeKind::read(i, ""), Pos2::new(x, -200.0));
            if let Some(NodeKind::Read { single, eight_bit, .. }) = g.node_mut(rd).map(|n| &mut n.kind) {
                *single = s;
                *eight_bit = e;
            }
            let gr = g.add(NodeKind::Grade(Default::default()), Pos2::new(x, 120.0));
            let src = if i == 3 {
                let d = g.add(NodeKind::dot(PortKind::Image), Pos2::new(x + 60.0, -40.0));
                g.connect(d, 0, rd);
                d
            } else {
                rd
            };
            g.connect(gr, 0, src);
            fed.push((gr, src));
        }
        assert_eq!(Reduced::of(&g, fed[0].1), None);
        assert_eq!(Reduced::of(&g, fed[1].1).map(|r| r.tag()), Some("R".into()));
        assert_eq!(Reduced::of(&g, fed[2].1).map(|r| r.tag()), Some("8-bit".into()));
        assert_eq!(Reduced::of(&g, fed[3].1).map(|r| r.tag()), Some("Y · 8-bit".into()));
        g.view = Some(fed[0].0);
        let mut ed = NodeEditor::default();
        ed.zoom = 1.0;
        ed.show_keys = false;
        ed.show_legend = false;
        let (mut sel, mut act, mut pin, mut slot, mut rects) =
            (Vec::new(), Vec::new(), Vec::new(), 4usize, Vec::new());
        crate::uishot::render("graph-reduced", 900.0, 560.0, 2.0, 6, |ui, _| {
            ed.show(ui, &mut g, &mut sel, &mut act, &mut pin, &mut slot, &mut rects);
        });
        set_boring(true);
        ed.stalks.clear();
        crate::uishot::render("graph-reduced-boring", 900.0, 560.0, 2.0, 6, |ui, _| {
            ed.show(ui, &mut g, &mut sel, &mut act, &mut pin, &mut slot, &mut rects);
        });
        set_boring(false);
    }
}

#[cfg(test)]
mod boring_tests {
    use super::*;

    /// `set_boring` is process-wide; the tests here that flip it take this.
    static MODE: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// A dragged node snaps its wire straight and its centre in line with
    /// a neighbour, in either look; far off it stays where it was put.
    #[test]
    fn a_dragged_node_snaps_its_wire_straight_and_lines_up() {
        let _m = MODE.lock().unwrap_or_else(|e| e.into_inner());
        for on in [false, true] {
            set_boring(on);
            let mut g = EditorGraph::default();
            let a = g.add(NodeKind::Grade(Default::default()), Pos2::ZERO);
            let b = g.add(NodeKind::Grade(Default::default()), Pos2::new(5.0, 200.0));
            let c = g.add(NodeKind::Grade(Default::default()), Pos2::new(400.0, 203.0));
            g.connect(b, 0, a);
            let dirs = std::collections::HashMap::new();
            let area = Rect::from_min_max(Pos2::new(-1e4, -1e4), Pos2::new(1e4, 1e4));
            let (off, guides) = drag_snap(&g, &dirs, &[b], area, 8.0);
            assert_eq!(guides.len(), 2, "boring {on}");
            g.node_mut(b).unwrap().pos += off;
            let bp = g.node(b).unwrap().pos;
            let inp = rest_of(&g, &dirs, b, 0, bp, 1.0);
            let out = output_rest_of(&g.nodes, a, 0, 1.0);
            assert!((inp.x - out.x).abs() < 1e-3, "wire not vertical (boring {on}): {inp:?} {out:?}");
            assert!((bp.y - g.node(c).unwrap().pos.y).abs() < 1e-3, "not in line with c (boring {on})");
            // Out of reach: no snap.
            g.node_mut(b).unwrap().pos = Pos2::new(60.0, 260.0);
            let (off, guides) = drag_snap(&g, &dirs, &[b], area, 8.0);
            assert!(off == Vec2::ZERO && guides.is_empty(), "boring {on}: {off:?}");
        }
        set_boring(false);
    }

    /// Boring mode: sockets sit ON the disc (no stalk), a side matte too,
    /// and every noodle is one straight segment.
    #[test]
    fn boring_sockets_sit_on_the_body_and_wires_are_straight() {
        let _m = MODE.lock().unwrap_or_else(|e| e.into_inner());
        let mut g = EditorGraph::default();
        let gr = g.add(NodeKind::Grade(Default::default()), Pos2::ZERO);
        let dirs = std::collections::HashMap::new();
        let m = g.node(gr).and_then(|n| n.kind.mask_input()).unwrap();
        set_boring(true);
        let h = BORING_HALF;
        let on_edge = |p: Pos2| {
            ((p.x.abs() - h.x).abs() < 1e-3 && p.y.abs() <= h.y + 1e-3)
                || ((p.y.abs() - h.y).abs() < 1e-3 && p.x.abs() <= h.x + 1e-3)
        };
        for i in [0, m] {
            let p = rest_of(&g, &dirs, gr, i, Pos2::ZERO, 1.0);
            assert!(on_edge(p), "input {i} at {p:?}");
        }
        let o = output_rest_of(&g.nodes, gr, 0, 1.0);
        assert!((o.y - h.y).abs() < 1e-3 && o.x.abs() < 1e-3, "output at {o:?}");
        // A Load is a 4:3 card, its output off the bottom edge.
        let rd = g.add(NodeKind::read(0, ""), Pos2::ZERO);
        let n = g.node(rd).unwrap();
        assert!(tv_on(n) && wide_of(n) == Some((BORING_READ_HALF, 0.0)));
        assert!((BORING_READ_HALF.x / BORING_READ_HALF.y - 4.0 / 3.0).abs() < 1e-4);
        let o = output_rest_of(&g.nodes, rd, 0, 1.0);
        assert!((o.y - BORING_READ_HALF.y).abs() < 1e-3, "Load output at {o:?}");
        for w in [
            wire_path(Pos2::ZERO, Pos2::new(50.0, 90.0), 1.0, None, None, None, true),
            plain_wire_path(Pos2::ZERO, Pos2::new(50.0, 90.0), 1.0, false),
            cable_path(Pos2::ZERO, Pos2::new(50.0, 90.0), 1.0, 1.0, false),
        ] {
            assert!(matches!(&w, WirePath::Ortho(p) if p.len() == 2), "not straight");
        }
        set_boring(false);
        let p = rest_of(&g, &dirs, gr, 0, Pos2::ZERO, 1.0);
        assert!((p.y + R + STALK).abs() < 1e-3, "off again, the stalk is back: {p:?}");
    }
}

#[cfg(test)]
mod ortho_hit_tests {
    use super::*;

    /// A Read up top and a Grade below and to the right, so the wire is a
    /// Z in right angles and an S as a curve — driven through the real
    /// editor, headless.
    struct Rig {
        g: EditorGraph,
        ed: NodeEditor,
        ctx: egui::Context,
        rects: Vec<(u64, Rect)>,
        n: usize,
        read: u64,
        grade: u64,
    }

    impl Rig {
        fn new(ortho: bool) -> Self {
            let mut g = EditorGraph::default();
            g.nodes.clear();
            g.edges.clear();
            g.next_id = 0;
            let read = g.add(NodeKind::read(0, ""), Pos2::new(0.0, -200.0));
            let grade = g.add(NodeKind::Grade(Default::default()), Pos2::new(320.0, 160.0));
            g.connect(grade, 0, read);
            let mut ed = NodeEditor::default();
            ed.ortho_wires = ortho;
            let mut rig =
                Rig { g, ed, ctx: egui::Context::default(), rects: Vec::new(), n: 0, read, grade };
            rig.frames(120, Vec::new(), egui::Modifiers::default());
            rig
        }

        fn frames(&mut self, count: usize, first: Vec<egui::Event>, modifiers: egui::Modifiers) {
            let mut events = first;
            for _ in 0..count {
                let input = egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(900.0, 700.0))),
                    time: Some(self.n as f64 / 60.0),
                    events: {
                        let mut all = vec![egui::Event::ModifiersChanged(modifiers)];
                        all.extend(std::mem::take(&mut events));
                        all
                    },
                    ..Default::default()
                };
                self.n += 1;
                let (mut sel, mut act, mut pin, mut slot) =
                    (Vec::new(), Vec::new(), Vec::new(), 2usize);
                let (ed, g, rects) = (&mut self.ed, &mut self.g, &mut self.rects);
                let _ = self.ctx.run_ui(input, |ui| {
                    ed.show(ui, g, &mut sel, &mut act, &mut pin, &mut slot, rects);
                });
            }
        }

        fn to_screen(&self, p: Pos2) -> Pos2 {
            let r = self.rects.iter().find(|(i, _)| *i == self.read).unwrap().1;
            let origin = r.center() - self.g.node(self.read).unwrap().pos.to_vec2() * self.ed.zoom;
            origin + p.to_vec2() * self.ed.zoom
        }

        /// On the right-angle route's horizontal run, a quarter of the way
        /// across — where the curve is not: it is still dropping there.
        fn on_the_run(&self) -> Pos2 {
            let a = self.to_screen(self.ed.stalks[&(self.read, out_stalk(0))].0);
            let b = self.to_screen(self.ed.stalks[&(self.grade, 0)].0);
            Pos2::new(a.x + (b.x - a.x) * 0.25, (a.y + b.y) * 0.5)
        }
    }

    fn period(down: bool) -> egui::Event {
        egui::Event::Key {
            key: egui::Key::Period,
            physical_key: None,
            pressed: down,
            repeat: false,
            modifiers: Default::default(),
        }
    }

    fn button(p: Pos2, down: bool, modifiers: egui::Modifiers) -> egui::Event {
        egui::Event::PointerButton { pos: p, button: egui::PointerButton::Primary, pressed: down, modifiers }
    }

    fn key(k: egui::Key, down: bool) -> egui::Event {
        egui::Event::Key { key: k, physical_key: None, pressed: down, repeat: false, modifiers: Default::default() }
    }

    /// Boring: a press anywhere inside a pill grabs the NODE, even right
    /// beside a socket (whose circle reaches past the thin body); just
    /// outside the edge the socket still takes it.
    #[test]
    fn a_boring_pill_body_beats_its_sockets() {
        let none = egui::Modifiers::default();
        set_boring(true);
        for (dy, moves) in [(9.0f32, true), (-9.0, true), (19.0, false)] {
            let mut rig = Rig::new(false);
            let start = rig.g.node(rig.grade).unwrap().pos;
            let at = rig.to_screen(start + Vec2::new(4.0, dy));
            let to = at + Vec2::new(70.0, 40.0);
            rig.frames(2, vec![egui::Event::PointerMoved(at)], none);
            rig.frames(1, vec![button(at, true, none)], none);
            for k in 1..=10 {
                let q = at + (to - at) * (k as f32 / 10.0);
                rig.frames(1, vec![egui::Event::PointerMoved(q)], none);
            }
            rig.frames(1, vec![button(to, false, none)], none);
            rig.frames(2, Vec::new(), none);
            let moved = rig.g.node(rig.grade).unwrap().pos.distance(start) > 20.0;
            assert_eq!(moved, moves, "press at dy {dy}: node moved = {moved}");
            assert_eq!(rig.g.input_of(rig.grade, 0), Some(rig.read), "the wire stayed (dy {dy})");
        }
        set_boring(false);
    }

    /// 1 over a node views it in slot A; 2 straight after, the pointer
    /// still on it, flips to what slot B already holds rather than giving B
    /// the node too. Off the node and back, 2 takes it.
    #[test]
    fn a_node_one_takes_is_not_two_s_until_the_pointer_leaves() {
        let none = egui::Modifiers::default();
        let mut rig = Rig::new(false);
        rig.g.view_b = Some(rig.read);
        let on = rig.rects.iter().find(|(i, _)| *i == rig.grade).unwrap().1.center();
        rig.frames(2, vec![egui::Event::PointerMoved(on)], none);
        rig.frames(1, vec![key(egui::Key::Num1, true)], none);
        rig.frames(1, vec![key(egui::Key::Num1, false)], none);
        assert_eq!(rig.g.view, Some(rig.grade), "1 views the node under the pointer");
        rig.frames(1, vec![key(egui::Key::Num2, true)], none);
        rig.frames(1, vec![key(egui::Key::Num2, false)], none);
        assert_eq!(rig.g.view_b, Some(rig.read), "2 kept what slot B held");
        assert!(rig.g.view_active_b, "and flipped to it");
        rig.frames(2, vec![egui::Event::PointerMoved(Pos2::new(5.0, 690.0))], none);
        rig.frames(2, vec![egui::Event::PointerMoved(on)], none);
        rig.frames(1, vec![key(egui::Key::Num2, true)], none);
        rig.frames(1, vec![key(egui::Key::Num2, false)], none);
        assert_eq!(rig.g.view_b, Some(rig.grade), "back on the node, 2 takes it");
    }

    /// `.` + click on the drawn run cuts a dot in exactly there — and in
    /// curve mode the same spot is empty canvas.
    #[test]
    fn a_dot_cut_lands_on_the_run_that_is_drawn() {
        let none = egui::Modifiers::default();
        for ortho in [true, false] {
            let mut rig = Rig::new(ortho);
            let at = rig.on_the_run();
            rig.frames(2, vec![period(true)], none);
            rig.frames(1, vec![egui::Event::PointerMoved(at), button(at, true, none)], none);
            rig.frames(1, vec![button(at, false, none)], none);
            rig.frames(2, vec![period(false)], none);
            let dot = rig
                .g
                .nodes
                .iter()
                .find(|(_, n)| matches!(n.kind, NodeKind::Dot { .. }))
                .map(|(id, _)| *id);
            if ortho {
                let dot = dot.expect("right angles: a click on the run cuts a dot");
                assert_eq!(rig.g.input_of(rig.grade, 0), Some(dot));
                assert_eq!(rig.g.input_of(dot, 0), Some(rig.read));
                let bead = rig.to_screen(rig.g.node(dot).unwrap().pos);
                assert!(bead.distance(at) < 2.0, "the dot sits where it was cut: {bead:?} vs {at:?}");
            } else {
                assert!(dot.is_none(), "curves: that spot is off the wire");
            }
        }
    }

    /// A ctrl-drag across the drawn run cuts the wire; as a curve the wire
    /// is elsewhere and survives the same stroke.
    #[test]
    fn a_slice_cuts_the_run_that_is_drawn() {
        let ctrl = egui::Modifiers { ctrl: true, ..Default::default() };
        for ortho in [true, false] {
            let mut rig = Rig::new(ortho);
            let at = rig.on_the_run();
            let (top, bottom) = (at - Vec2::new(0.0, 12.0), at + Vec2::new(0.0, 12.0));
            rig.frames(1, vec![egui::Event::PointerMoved(top), button(top, true, ctrl)], ctrl);
            for k in 1..=8 {
                let p = top + (bottom - top) * (k as f32 / 8.0);
                rig.frames(1, vec![egui::Event::PointerMoved(p)], ctrl);
            }
            rig.frames(2, vec![button(bottom, false, ctrl)], ctrl);
            assert_eq!(
                rig.g.input_of(rig.grade, 0).is_none(),
                ortho,
                "ortho={ortho}: the stroke across the run cuts only the wire drawn there"
            );
        }
    }
    /// A Layer's input nodule slid along its bar moves that layer to the
    /// gap it is dropped in, wire and keys with it; pulled off the bar it
    /// unplugs like any input.
    #[test]
    fn a_layer_nodule_slid_along_the_bar_reorders_the_stack() {
        let none = egui::Modifiers::default();
        let mut rig = Rig::new(false);
        let reads: Vec<u64> = (0..3).map(|k| rig.g.add(NodeKind::read(k + 1, ""), Pos2::new(-300.0 + 150.0 * k as f32, -220.0))).collect();
        let three = engine::graph::LayerParams { layers: vec![Default::default(); 3], outputs: false };
        let layer = rig.g.add(NodeKind::Layer(three), Pos2::new(0.0, 120.0));
        for (k, r) in reads.iter().enumerate() {
            rig.g.connect(layer, k, *r);
        }
        let ch0 = crate::anim::layer_channels(0)[0];
        rig.g.node_mut(layer).unwrap().anim.entry(ch0).or_default().set_key(0, 0.25);
        rig.frames(60, Vec::new(), none);
        let nod = |rig: &Rig, i: usize| rig.to_screen(rig.ed.stalks[&(layer, i)].0);
        let drag = |rig: &mut Rig, a: Pos2, b: Pos2| {
            rig.frames(1, vec![egui::Event::PointerMoved(a), button(a, true, none)], none);
            for k in 1..=12 {
                rig.frames(1, vec![egui::Event::PointerMoved(a + (b - a) * (k as f32 / 12.0))], none);
            }
            rig.frames(2, vec![button(b, false, none)], none);
        };
        // Layer 0's nodule past layer 2's: it becomes the last layer.
        let (a, end) = (nod(&rig, 0), nod(&rig, 2));
        let one = nod(&rig, 1);
        // Mid-drag the slid nodule rides the hand and the others step aside.
        let b = end + Vec2::new(14.0, 0.0);
        rig.frames(1, vec![egui::Event::PointerMoved(a), button(a, true, none)], none);
        for k in 1..=12 {
            rig.frames(1, vec![egui::Event::PointerMoved(a + (b - a) * (k as f32 / 12.0))], none);
        }
        rig.frames(20, vec![egui::Event::PointerMoved(b)], none);
        assert!((nod(&rig, 0).x - b.x).abs() < 6.0, "the slid nodule follows the hand: {:?} vs {b:?}", nod(&rig, 0));
        assert!((nod(&rig, 1) - a).length() < 4.0, "layer 2 stepped into slot 1 while dragging: {:?} vs {a:?} (was {one:?})", nod(&rig, 1));
        rig.frames(2, vec![button(b, false, none)], none);
        let order: Vec<_> = (0..3).map(|i| rig.g.input_of(layer, i)).collect();
        assert_eq!(order, vec![Some(reads[1]), Some(reads[2]), Some(reads[0])], "the moved layer took its wire");
        assert_eq!(rig.ed.glides.len(), 3, "all three nodules glide to their new slots");
        rig.frames(40, Vec::new(), none);
        assert!(rig.ed.glides.is_empty(), "the glide ends");
        assert!((nod(&rig, 0) - a).length() < 2.0, "nothing jumps after the drop");
        let n = rig.g.node(layer).unwrap();
        assert!(n.anim.get(&crate::anim::layer_channels(2)[0]).is_some_and(|c| !c.keys.is_empty()), "its keys went with it");
        assert!(n.anim.get(&ch0).is_none_or(|c| c.keys.is_empty()), "nothing left behind");
        // Pulled straight up off the bar: an unplug, the stack closing up.
        rig.frames(30, Vec::new(), none);
        let a = nod(&rig, 1);
        drag(&mut rig, a, a + Vec2::new(0.0, -160.0));
        rig.frames(5, Vec::new(), none);
        assert_eq!(rig.g.input_of(layer, 1), None, "pulled off the bar, it unplugs");
    }

    /// A noodle dropped anywhere on a Layer's bar becomes a new layer at
    /// the gap it was dropped in, like a 2dini Merge: right of the only
    /// string is under it, left of it on top.
    #[test]
    fn a_noodle_dropped_on_a_layer_bar_adds_a_layer_there() {
        let none = egui::Modifiers::default();
        for (side, want) in [(1.0f32, 1usize), (-1.0, 0)] {
            let mut rig = Rig::new(false);
            let b = rig.g.add(NodeKind::read(1, ""), Pos2::new(-300.0, -200.0));
            let one = engine::graph::LayerParams { layers: vec![Default::default()], outputs: false };
            let layer = rig.g.add(NodeKind::Layer(one), Pos2::new(0.0, 120.0));
            rig.g.connect(layer, 0, rig.read);
            rig.frames(60, Vec::new(), none);
            let rect = |rig: &Rig, id: u64| rig.rects.iter().find(|(i, _)| *i == id).unwrap().1;
            let out = rig.to_screen(rig.ed.stalks[&(b, out_stalk(0))].0);
            let lr = rect(&rig, layer);
            let drop = Pos2::new(lr.center().x + side * (lr.width() * 0.5 - 6.0), lr.center().y);
            rig.frames(1, vec![egui::Event::PointerMoved(out), button(out, true, none)], none);
            for k in 1..=10 {
                let p = out + (drop - out) * (k as f32 / 10.0);
                rig.frames(1, vec![egui::Event::PointerMoved(p)], none);
            }
            rig.frames(2, vec![button(drop, false, none)], none);
            let NodeKind::Layer(p) = &rig.g.node(layer).unwrap().kind else { panic!() };
            assert_eq!(p.layers.len(), 2, "side {side}: a new layer");
            assert_eq!(rig.g.input_of(layer, want), Some(b), "side {side}: the noodle feeds the new layer");
            assert_eq!(rig.g.input_of(layer, 1 - want), Some(rig.read), "side {side}: the old layer keeps its wire");
            // A fresh stack's empty layer takes the next drop, wherever.
            let c = rig.g.add(NodeKind::read(2, ""), Pos2::new(300.0, -200.0));
            crate::layerui::restack_at(&mut rig.g, layer, crate::layerui::Restack::Insert(2));
            rig.frames(30, Vec::new(), none);
            let out = rig.to_screen(rig.ed.stalks[&(c, out_stalk(0))].0);
            let lr = rect(&rig, layer);
            let drop = Pos2::new(lr.left() + 8.0, lr.center().y);
            rig.frames(1, vec![egui::Event::PointerMoved(out), button(out, true, none)], none);
            for k in 1..=10 {
                rig.frames(1, vec![egui::Event::PointerMoved(out + (drop - out) * (k as f32 / 10.0))], none);
            }
            rig.frames(2, vec![button(drop, false, none)], none);
            let NodeKind::Layer(p) = &rig.g.node(layer).unwrap().kind else { panic!() };
            assert_eq!(p.layers.len(), 3, "side {side}: the empty layer was used, not a new one");
            assert_eq!(rig.g.input_of(layer, 2), Some(c));
        }
    }
}

#[cfg(test)]
mod band_matte_tests {
    use super::*;

    /// A Read, a three-band LumaKey off it, and a Grade off the Read with
    /// its matte free — driven through the real editor, headless.
    struct Rig {
        g: EditorGraph,
        ed: NodeEditor,
        ctx: egui::Context,
        rects: Vec<(u64, Rect)>,
        n: usize,
        lk: u64,
        gr: u64,
    }

    impl Rig {
        fn new() -> Self {
            let mut g = EditorGraph::default();
            g.nodes.clear();
            g.edges.clear();
            g.next_id = 0;
            let r = g.add(NodeKind::read(0, ""), Pos2::new(0.0, -200.0));
            let mut kp = engine::graph::LumaKeyParams::default();
            kp.adopt_ranges(engine::graph::LUMA_SPAN);
            *kp.ranges.as_mut().unwrap() =
                engine::bands::BandRanges::even(3, engine::graph::LUMA_SPAN, false, 0.65);
            kp.sync_from_ranges();
            let lk = g.add(NodeKind::LumaKey(kp), Pos2::new(220.0, -40.0));
            g.connect(lk, 0, r);
            let gr = g.add(NodeKind::Grade(Default::default()), Pos2::new(0.0, 160.0));
            g.connect(gr, 0, r);
            let mut rig = Rig {
                g,
                ed: NodeEditor::default(),
                ctx: egui::Context::default(),
                rects: Vec::new(),
                n: 0,
                lk,
                gr,
            };
            rig.frames(240, Vec::new(), false);
            rig
        }

        fn frames(&mut self, count: usize, first: Vec<egui::Event>, shift: bool) {
            let mut events = first;
            for _ in 0..count {
                let input = egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(900.0, 700.0))),
                    time: Some(self.n as f64 / 60.0),
                    events: {
                        let mut all = vec![egui::Event::ModifiersChanged(egui::Modifiers { shift, ..Default::default() })];
                        all.extend(std::mem::take(&mut events));
                        all
                    },
                    ..Default::default()
                };
                self.n += 1;
                let (mut sel, mut act, mut pin, mut slot) =
                    (Vec::new(), Vec::new(), Vec::new(), 2usize);
                let (ed, g, rects) = (&mut self.ed, &mut self.g, &mut self.rects);
                let _ = self.ctx.run_ui(input, |ui| {
                    ed.show(ui, g, &mut sel, &mut act, &mut pin, &mut slot, rects);
                });
            }
        }

        /// Graph space -> screen, read off the LumaKey's drawn rect.
        fn to_screen(&self, p: Pos2) -> Pos2 {
            let r = self.rects.iter().find(|(i, _)| *i == self.lk).unwrap().1;
            let origin = r.center() - self.g.node(self.lk).unwrap().pos.to_vec2() * self.ed.zoom;
            origin + p.to_vec2() * self.ed.zoom
        }

        fn tip(&self, id: u64, k: usize) -> Pos2 {
            self.ed.stalks[&(id, k)].0
        }
    }

    /// Dragging a wire from ANY band onto a Grade's matte lands that band.
    #[test]
    fn a_band_dragged_onto_a_grade_matte_lands_that_band() {
        for k in 0..3u32 {
            let mut rig = Rig::new();
            let from = rig.to_screen(rig.tip(rig.lk, out_stalk(k as usize)));
            let to = rig.to_screen(rig.tip(rig.gr, 1));
            let press = |p: Pos2, down: bool| {
                vec![
                    egui::Event::PointerMoved(p),
                    egui::Event::PointerButton {
                        pos: p,
                        button: egui::PointerButton::Primary,
                        pressed: down,
                        modifiers: Default::default(),
                    },
                ]
            };
            rig.frames(1, press(from, true), false);
            for step in 1..=10 {
                let p = from + (to - from) * (step as f32 / 10.0);
                rig.frames(1, vec![egui::Event::PointerMoved(p)], false);
            }
            rig.frames(2, press(to, false), false);
            assert_eq!(rig.g.input_of(rig.gr, 1), Some(rig.lk), "band {k} did not land");
            assert_eq!(rig.g.out_of(rig.gr, 1), k, "band {k} landed as another band");
        }
    }

    /// The magnet: a Grade whose free matte socket comes up against band 2
    /// courts BAND 2, and shift wires band 2 — it used to measure and join
    /// output 0 only, so no band past the first could ever be shift-joined.
    #[test]
    fn the_magnet_courts_and_joins_the_band_it_is_near() {
        for k in [1usize, 2] {
            let mut rig = Rig::new();
            // Walk the Grade until its matte tip sits on band k's tip. The
            // mask stalk leans toward the key, so re-measure after settling.
            for _ in 0..3 {
                let gap = rig.tip(rig.lk, out_stalk(k)) - rig.tip(rig.gr, 1);
                rig.g.node_mut(rig.gr).unwrap().pos += gap;
                rig.frames(90, Vec::new(), false);
            }
            // Pointer over the canvas (shift only joins while it is hovered).
            let hover = Pos2::new(40.0, 660.0);
            rig.frames(3, vec![egui::Event::PointerMoved(hover)], true);
            assert_eq!(rig.g.input_of(rig.gr, 1), Some(rig.lk), "band {k} was not joined");
            assert_eq!(rig.g.out_of(rig.gr, 1), k as u32, "shift joined the wrong band");
        }
    }

    /// Park `id` so band 0's tip sits between its free input `main` and its
    /// side input `side`, NEARER the side one, both in reach.
    fn park_between(rig: &mut Rig, id: u64, main: usize, side: usize) {
        for _ in 0..4 {
            let pos = rig.g.node(id).unwrap().pos;
            let (t0, t1) = (rig.tip(id, main) - pos, rig.tip(id, side) - pos);
            let spot = t1 + (t0 - t1) * 0.44;
            let gap = rig.tip(rig.lk, out_stalk(0)) - (pos + spot);
            rig.g.node_mut(id).unwrap().pos += gap;
            rig.frames(90, Vec::new(), false);
        }
    }

    /// Shift-connecting a matte to a filter: a free MAIN input is taken
    /// first, even with the side port nearer — the main port is the one that
    /// reaches, and the filter takes the matte pipe. With the main input
    /// wired, the matte goes to the mask (a Grade) or the modulate (a Blur).
    #[test]
    fn a_matte_shift_joins_the_main_input_first() {
        for (kind, side) in [
            (NodeKind::Blur(Default::default()), 1usize), // modulate
            (NodeKind::Grade(Default::default()), 1usize), // mask
        ] {
            let mut rig = Rig::new();
            let f = rig.g.add(kind.clone(), Pos2::new(-400.0, 300.0));
            rig.frames(90, Vec::new(), false);
            park_between(&mut rig, f, 0, side);
            let (m, s) = (rig.tip(f, 0), rig.tip(f, side));
            let band = rig.tip(rig.lk, out_stalk(0));
            assert!(
                band.distance(s) < band.distance(m) + 1.0 || band.distance(m) < 30.0,
                "{}: parked wrong (main {m:?} side {s:?} band {band:?})",
                kind.name()
            );
            let hover = Pos2::new(40.0, 660.0);
            rig.frames(3, vec![egui::Event::PointerMoved(hover)], true);
            assert_eq!(rig.g.input_of(f, 0), Some(rig.lk), "{}: main input not joined", kind.name());
            assert_eq!(rig.g.input_of(f, side), None, "{}: the side port took it", kind.name());
            assert!(rig.g.node(f).unwrap().kind.matte_mode(), "{} did not take the matte pipe", kind.name());
        }
        // Main input already fed: the same matte goes to the side port.
        for (kind, side) in [
            (NodeKind::Blur(Default::default()), 1usize),
            (NodeKind::Grade(Default::default()), 1usize),
        ] {
            let mut rig = Rig::new();
            let f = rig.g.add(kind.clone(), Pos2::new(-400.0, 300.0));
            let plate = rig.g.add(NodeKind::read(1, ""), Pos2::new(-400.0, 100.0));
            rig.g.connect(f, 0, plate);
            rig.frames(90, Vec::new(), false);
            for _ in 0..3 {
                let gap = rig.tip(rig.lk, out_stalk(0)) - rig.tip(f, side);
                rig.g.node_mut(f).unwrap().pos += gap;
                rig.frames(90, Vec::new(), false);
            }
            let hover = Pos2::new(40.0, 660.0);
            rig.frames(3, vec![egui::Event::PointerMoved(hover)], true);
            assert_eq!(rig.g.input_of(f, side), Some(rig.lk), "{}: side port not joined", kind.name());
            assert_eq!(rig.g.input_of(f, 0), Some(plate));
        }
    }
}

#[cfg(test)]
mod clipboard_tests {
    use super::*;

    fn scene() -> (EditorGraph, u64, u64, u64) {
        let mut g = EditorGraph::default();
        g.nodes.clear();
        g.edges.clear();
        g.next_id = 0;
        let r = g.add(NodeKind::read(0, "/plates/a.####.exr"), Pos2::new(0.0, 0.0));
        let gr = g.add(NodeKind::Grade(Default::default()), Pos2::new(0.0, 100.0));
        let bl = g.add(NodeKind::Blur(Default::default()), Pos2::new(0.0, 200.0));
        g.connect(gr, 0, r);
        g.connect(bl, 0, gr);
        (g, r, gr, bl)
    }

    /// A copy of a wired pair pastes as a fresh wired pair, centred where
    /// asked, with a reader slot of its own — and does not drag the wire in
    /// from outside the copy with it.
    #[test]
    fn copy_then_paste_rebuilds_the_branch() {
        let (mut g, r, gr, bl) = scene();
        let text = copy_nodes(&g, &[r, gr]).expect("copies");
        let mut slot = 7usize;
        let ids = paste_nodes(&mut g, &text, Pos2::new(500.0, 500.0), &mut slot).expect("pastes");
        assert_eq!(ids.len(), 2);
        let (r2, gr2) = (ids[0], ids[1]);
        assert!(!ids.contains(&r) && !ids.contains(&gr), "fresh ids");
        assert_eq!(g.input_of(gr2, 0), Some(r2), "the copied wire comes along");
        assert_eq!(g.input_of(r2, 0), None);
        assert_eq!(g.input_of(bl, 0), Some(gr), "the original is untouched");
        // Centred on the paste point, relative layout kept.
        let (p, q) = (g.node(r2).unwrap().pos, g.node(gr2).unwrap().pos);
        assert_eq!(((p.x + q.x) / 2.0, (p.y + q.y) / 2.0), (500.0, 500.0));
        assert_eq!(q.y - p.y, 100.0);
        // Its own reader slot: two nodes on one slot fight over a texture.
        assert_eq!(g.node(r2).unwrap().kind.reader_slot(), Some(7));
        assert_eq!(slot, 8);
        match &g.node(r2).unwrap().kind {
            NodeKind::Read { path, .. } => assert_eq!(path, "/plates/a.####.exr"),
            k => panic!("pasted {k:?}"),
        }
    }

    /// A wire from a band keeps its band through the clipboard.
    #[test]
    fn a_wire_keeps_its_output_through_the_clipboard() {
        let (mut g, r, gr, _) = scene();
        let mut kp = engine::graph::LumaKeyParams::default();
        kp.adopt_ranges(engine::graph::LUMA_SPAN);
        *kp.ranges.as_mut().unwrap() =
            engine::bands::BandRanges::even(3, engine::graph::LUMA_SPAN, false, 0.65);
        kp.sync_from_ranges();
        let lk = g.add(NodeKind::LumaKey(kp), Pos2::new(200.0, 0.0));
        g.connect(lk, 0, r);
        g.connect_out(gr, 1, lk, 2);
        let text = copy_nodes(&g, &[lk, gr]).unwrap();
        let ids = paste_nodes(&mut g, &text, Pos2::ZERO, &mut 3).unwrap();
        let gr2 = *ids.iter().find(|id| matches!(g.nodes[id].kind, NodeKind::Grade(_))).unwrap();
        assert_eq!(g.out_of(gr2, 1), 2, "band 2 stays band 2");
    }

    /// Minted readers go back to unminted, for the app to mint afresh.
    #[test]
    fn minted_readers_paste_unminted() {
        let mut k = NodeKind::ChannelSelect(engine::graph::ChannelSelectParams {
            source: 4,
            layer: String::new(),
        });
        reslot(&mut k, &mut 9);
        assert_eq!(k.reader_slot(), Some(usize::MAX));
    }

    #[test]
    fn anything_else_on_the_clipboard_is_not_pasted() {
        let (mut g, ..) = scene();
        let n = g.nodes.len();
        assert!(paste_nodes(&mut g, "hello", Pos2::ZERO, &mut 0).is_none());
        assert!(paste_nodes(&mut g, "{\"nodes\": []}", Pos2::ZERO, &mut 0).is_none());
        assert_eq!(g.nodes.len(), n);
        assert!(copy_nodes(&g, &[]).is_none(), "nothing selected, nothing copied");
    }

    /// Through the real editor: ctrl+A selects every node, ⌘C (egui's Copy
    /// event) copies, ⌘V (Paste, carrying the text) pastes at the pointer,
    /// and ctrl+V pastes the last copy again.
    #[test]
    fn pasted_paths_become_loads_through_the_editor() {
        let (mut g, ..) = scene();
        let ctx = egui::Context::default();
        let mut ed = NodeEditor::default();
        ed.system_clipboard = false;
        let (mut sel, mut act, mut pin, mut slot) = (Vec::new(), Vec::new(), Vec::new(), 10usize);
        let mut rects = Vec::new();
        let at = Pos2::new(300.0, 400.0);
        let before = g.nodes.len();
        for (i, ev) in [vec![], vec![egui::Event::Paste("/no/such/a.####.exr\nD:\\no\\b.%04d.exr".into())]]
            .into_iter()
            .enumerate()
        {
            let mut events = vec![egui::Event::PointerMoved(at)];
            events.extend(ev);
            let input = egui::RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(900.0, 700.0))),
                time: Some(i as f64 / 60.0),
                events,
                ..Default::default()
            };
            let _ = ctx.run_ui(input, |ui| {
                ed.show(ui, &mut g, &mut sel, &mut act, &mut pin, &mut slot, &mut rects);
            });
        }
        assert_eq!(g.nodes.len(), before + 2, "two lines, two Loads");
        assert!(ed.pasted, "the app is told, so the Loads get their footage");
        let mut got: Vec<(String, usize)> = sel
            .iter()
            .filter_map(|id| match &g.node(*id)?.kind {
                NodeKind::Read { path, source, .. } => Some((path.clone(), *source)),
                _ => None,
            })
            .collect();
        got.sort();
        assert_eq!(got, vec![("/no/such/a.####.exr".into(), 10), ("D:/no/b.####.exr".into(), 11)]);
        assert_eq!(slot, 12);
    }

    #[test]
    fn select_all_copy_and_paste_through_the_editor() {
        let (mut g, ..) = scene();
        let ctx = egui::Context::default();
        let mut ed = NodeEditor::default();
        ed.system_clipboard = false;
        let (mut sel, mut act, mut pin, mut slot) = (Vec::new(), Vec::new(), Vec::new(), 1usize);
        let mut rects = Vec::new();
        let at = Pos2::new(200.0, 500.0);
        let mut n = 0usize;
        let mut run = |ev: Vec<egui::Event>,
                       mods: egui::Modifiers,
                       g: &mut EditorGraph,
                       ed: &mut NodeEditor,
                       sel: &mut Vec<u64>| {
            let mut events = vec![egui::Event::ModifiersChanged(mods), egui::Event::PointerMoved(at)];
            events.extend(ev);
            let input = egui::RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(900.0, 700.0))),
                time: Some(n as f64 / 60.0),
                events,
                ..Default::default()
            };
            n += 1;
            let _ = ctx.run_ui(input, |ui| {
                ed.show(ui, g, sel, &mut act, &mut pin, &mut slot, &mut rects);
            });
        };
        // A TAP: down and up in one frame, or egui reads the next press of
        // the same key as a repeat (which no binding answers).
        let key = |k: Key, m: egui::Modifiers| -> Vec<egui::Event> {
            [true, false]
                .into_iter()
                .map(|pressed| egui::Event::Key {
                    key: k,
                    physical_key: None,
                    pressed,
                    repeat: false,
                    modifiers: m,
                })
                .collect()
        };
        use egui::Key;
        for _ in 0..3 {
            run(Vec::new(), Default::default(), &mut g, &mut ed, &mut sel);
        }
        run(key(Key::A, egui::Modifiers::CTRL), egui::Modifiers::CTRL, &mut g, &mut ed, &mut sel);
        let mut all: Vec<u64> = g.nodes.keys().copied().collect();
        all.sort_unstable();
        let mut got = sel.clone();
        got.sort_unstable();
        assert_eq!(got, all, "ctrl+A selects every node");
        // ⌘A too.
        sel.clear();
        run(key(Key::A, egui::Modifiers::COMMAND), egui::Modifiers::COMMAND, &mut g, &mut ed, &mut sel);
        assert_eq!(sel.len(), all.len(), "cmd+A selects every node");

        run(vec![egui::Event::Copy], Default::default(), &mut g, &mut ed, &mut sel);
        let text = ed.clip.clone().expect("the copy filled the buffer");
        run(vec![egui::Event::Paste(text)], Default::default(), &mut g, &mut ed, &mut sel);
        assert_eq!(g.nodes.len(), 6, "cmd+V pasted a second copy of all three");
        assert!(ed.pasted, "the app is told, so pasted Reads get their footage");
        assert_eq!(sel.len(), 3, "the paste is what is selected");
        ed.pasted = false;
        run(key(Key::V, egui::Modifiers::CTRL), egui::Modifiers::CTRL, &mut g, &mut ed, &mut sel);
        assert_eq!(g.nodes.len(), 9, "ctrl+V pastes the last copy again");
        // The pasted Reads each own a slot nobody else has.
        let mut slots: Vec<usize> =
            g.nodes.values().filter_map(|n| n.kind.reader_slot()).collect();
        slots.sort_unstable();
        let before = slots.len();
        slots.dedup();
        assert_eq!(slots.len(), before, "two Reads share a slot: {slots:?}");
    }
}

#[cfg(test)]
mod tab_feed_tests {
    use super::*;

    fn scene() -> (EditorGraph, u64, u64, u64) {
        let mut g = EditorGraph::default();
        g.nodes.clear();
        g.edges.clear();
        g.next_id = 0;
        let a = g.add(NodeKind::read(0, ""), Pos2::ZERO);
        let b = g.add(NodeKind::read(1, ""), Pos2::new(100.0, 0.0));
        let gr = g.add(NodeKind::Grade(Default::default()), Pos2::new(0.0, 100.0));
        (g, a, b, gr)
    }

    #[test]
    fn one_selected_feeds_input_zero() {
        let (mut g, a, _, _) = scene();
        let n = g.add(NodeKind::Blur(Default::default()), Pos2::new(0.0, 200.0));
        assert_eq!(feeds_for_new(&g, n, &[a]), vec![(a, 0)]);
        assert!(feeds_for_new(&g, n, &[]).is_empty(), "nothing selected, nothing wired");
    }

    /// Two into a Blend: pick order decides, first is the foreground (+).
    #[test]
    fn two_selected_feed_a_blend_first_as_foreground() {
        let (mut g, a, b, gr) = scene();
        let bl = g.add(NodeKind::blend(), Pos2::new(0.0, 200.0));
        assert_eq!(feeds_for_new(&g, bl, &[gr, b]), vec![(gr, 0), (b, 1)]);
        assert_eq!(feeds_for_new(&g, bl, &[b, gr]), vec![(b, 0), (gr, 1)], "order is pick order");
        assert_eq!(feeds_for_new(&g, bl, &[a]), vec![(a, 0)], "one pick is the foreground");
    }

    /// A node without an fg/bg pair takes the FIRST pick only — the second
    /// must never land in a matte or an aux port by accident.
    #[test]
    fn two_selected_into_a_one_pair_node_feed_only_the_first() {
        let (mut g, a, b, _) = scene();
        let gr = g.add(NodeKind::Grade(Default::default()), Pos2::new(0.0, 200.0));
        assert_eq!(feeds_for_new(&g, gr, &[a, b]), vec![(a, 0)]);
    }

    /// A Tab add INSERTS: the selected node feeds the new one, and what the
    /// selected node fed now feeds from the new one.
    #[test]
    fn the_new_node_takes_the_selected_node_s_place_downstream() {
        let (mut g, a, _, gr) = scene();
        let bl = g.add(NodeKind::Blur(Default::default()), Pos2::new(0.0, 300.0));
        g.connect(gr, 0, a);
        let second = g.add(NodeKind::Grade(Default::default()), Pos2::new(200.0, 300.0));
        g.connect(second, 0, a);
        // a -> gr and a -> second; a Blur added with `a` selected goes between.
        assert_eq!(splice_downstream(&mut g, a, 0, bl), 2);
        assert_eq!(g.input_of(gr, 0), Some(bl));
        assert_eq!(g.input_of(second, 0), Some(bl));
        assert_eq!(g.input_of(bl, 0), None, "its own feed is wired by the caller");
    }

    /// A consumer whose port would not take what the new node gives is left
    /// alone: adding a Blur below a key must not shove an image into the
    /// matte that key was feeding.
    #[test]
    fn a_port_that_would_not_take_the_output_keeps_its_wire() {
        let mut g = EditorGraph::default();
        g.nodes.clear();
        g.edges.clear();
        g.next_id = 0;
        let plate = g.add(NodeKind::read(0, ""), Pos2::ZERO);
        let key = g.add(NodeKind::LumaKey(Default::default()), Pos2::new(0.0, 100.0));
        g.connect(key, 0, plate);
        let gr = g.add(NodeKind::Grade(Default::default()), Pos2::new(0.0, 200.0));
        g.connect(gr, 0, plate);
        g.connect(gr, 1, key); // the key feeds a MATTE port
        let blur = g.add(NodeKind::Blur(Default::default()), Pos2::new(0.0, 300.0));
        assert_eq!(splice_downstream(&mut g, key, 0, blur), 0, "image out, matte in");
        assert_eq!(g.input_of(gr, 1), Some(key));
        // The plate's IMAGE consumers do move.
        assert_eq!(splice_downstream(&mut g, plate, 0, blur), 2);
        assert_eq!(g.input_of(gr, 0), Some(blur));
        assert_eq!(g.input_of(key, 0), Some(blur));
    }

    /// Only the output the new node was fed from is re-pointed: a band 2
    /// consumer is not handed something made from band 0.
    #[test]
    fn only_the_output_that_was_tapped_moves() {
        let mut g = EditorGraph::default();
        g.nodes.clear();
        g.edges.clear();
        g.next_id = 0;
        let plate = g.add(NodeKind::read(0, ""), Pos2::ZERO);
        let mut kp = engine::graph::LumaKeyParams::default();
        kp.adopt_ranges(engine::graph::LUMA_SPAN);
        *kp.ranges.as_mut().unwrap() =
            engine::bands::BandRanges::even(3, engine::graph::LUMA_SPAN, false, 0.65);
        kp.sync_from_ranges();
        let key = g.add(NodeKind::LumaKey(kp), Pos2::new(0.0, 100.0));
        g.connect(key, 0, plate);
        let (a, b) = (
            g.add(NodeKind::Dilate(Default::default()), Pos2::new(0.0, 200.0)),
            g.add(NodeKind::Dilate(Default::default()), Pos2::new(200.0, 200.0)),
        );
        g.connect_out(a, 0, key, 0);
        g.connect_out(b, 0, key, 2);
        let d = g.add(NodeKind::Dilate(Default::default()), Pos2::new(0.0, 300.0));
        assert_eq!(splice_downstream(&mut g, key, 0, d), 1, "band 0's consumer only");
        assert_eq!(g.input_of(a, 0), Some(d));
        assert_eq!((g.input_of(b, 0), g.out_of(b, 0)), (Some(key), 2), "band 2 untouched");
    }

    #[test]
    fn a_node_without_inputs_takes_nothing() {
        let (mut g, a, _, _) = scene();
        let r = g.add(NodeKind::read(2, ""), Pos2::new(0.0, 200.0));
        assert!(feeds_for_new(&g, r, &[a]).is_empty());
        // A stale id in the selection is skipped, not wired.
        let n = g.add(NodeKind::Blur(Default::default()), Pos2::new(0.0, 300.0));
        assert_eq!(feeds_for_new(&g, n, &[999, a]), vec![(a, 0)]);
    }
}

#[cfg(test)]
mod dot_tension_tests {
    use super::*;

    /// The handle tips sit where the pointer is: tension is distance over
    /// the automatic handle, linear, bounded, and 1 is easy to land on.
    #[test]
    fn tension_follows_the_pull() {
        let r = 40.0;
        assert_eq!(dot_tension_for(r, r), 1.0);
        assert_eq!(dot_tension_for(r * 1.04, r), 1.0, "snaps onto the automatic curve");
        assert!((dot_tension_for(r * 0.5, r) - 0.5).abs() < 1e-6);
        assert!((dot_tension_for(r * 2.5, r) - 2.5).abs() < 1e-6);
        assert_eq!(dot_tension_for(0.0, r), 0.0);
        assert_eq!(dot_tension_for(1e6, r), DOT_TENSION_MAX);
        assert!(dot_tension_for(5.0, 0.0).is_finite(), "a degenerate reference cannot blow up");
    }

    /// Tension scales exactly the dot end's handle, and nothing else: 0 is
    /// a straight run into the bead, 2 doubles the reach, the node end of
    /// the hop keeps its vertical tangent.
    #[test]
    fn tension_scales_the_dot_handle_only() {
        let (from, to, zoom) = (Pos2::new(0.0, 0.0), Pos2::new(120.0, 200.0), 1.0);
        let dir = Vec2::new(0.6, 0.8);
        let auto = wire_ctrl_flow(from, to, zoom, Some(dir), None);
        let taut = wire_ctrl_flow(from, to, zoom, Some(dir * 0.0), None);
        let slack = wire_ctrl_flow(from, to, zoom, Some(dir * 2.0), None);
        let reach = dot_handle_len(from, to, zoom);
        assert!(((auto[1] - from).length() - reach).abs() < 1e-3);
        assert_eq!(taut[1], from, "tension 0 has no handle at the dot");
        assert!(((slack[1] - from).length() - 2.0 * reach).abs() < 1e-3);
        assert_eq!(auto[2], slack[2], "the node end does not move");
    }

    /// A plain node never writes the field, and old setups read as 1.
    #[test]
    fn tension_is_not_serialized_at_one() {
        let n = ENode::new(NodeKind::dot(PortKind::Image), Pos2::ZERO);
        let json = serde_json::to_string(&n).unwrap();
        assert!(!json.contains("tension"), "{json}");
        let back: ENode = serde_json::from_str(&json).unwrap();
        assert_eq!(back.tension, 1.0);
        let mut pulled = n.clone();
        pulled.tension = 2.25;
        let back: ENode = serde_json::from_str(&serde_json::to_string(&pulled).unwrap()).unwrap();
        assert_eq!(back.tension, 2.25);
    }
}

#[cfg(test)]
mod patch_eval_tests {
    use super::*;
    use crate::anim::Channel;

    fn rig() -> (EditorGraph, u64, u64) {
        let mut g = EditorGraph::default();
        let tf = g.add(NodeKind::Transform(TransformParams::default()), Pos2::ZERO);
        let gr = g.add(NodeKind::Grade(GradeParams::default()), Pos2::ZERO);
        if let NodeKind::Transform(t) = &mut g.nodes.get_mut(&tf).unwrap().kind {
            t.scale = [3.0, 1.0];
        }
        (g, tf, gr)
    }

    fn gain_r(g: &EditorGraph, gr: u64, t: u64) -> f32 {
        let eng = g.to_engine_graph_at(Some(t));
        crate::anim::get(&eng.nodes[&gr], Channel::GainR).unwrap()
    }

    #[test]
    fn chan_drives_chan_through_mult() {
        let (mut g, tf, gr) = rig();
        let m = g.patch.add_mod(ModKind::Mult(2.0), [0.0, 0.0]);
        g.patch.link(PSrc::Chan(tf, Channel::ScaleX), PDst::Mod(m, 0));
        g.patch.link(PSrc::Mod(m), PDst::Chan(gr, Channel::GainR));
        assert_eq!(gain_r(&g, gr, 0), 6.0);
        // The authored grade is untouched — links are transient.
        if let NodeKind::Grade(p) = &g.nodes[&gr].kind {
            assert_eq!(p.gain[0], 1.0);
        }
    }

    #[test]
    fn animation_feeds_links() {
        let (mut g, tf, gr) = rig();
        let c = g.nodes.get_mut(&tf).unwrap().anim.entry(Channel::ScaleX).or_default();
        c.set_key(0, 1.0);
        c.set_key(10, 5.0);
        g.patch.link(PSrc::Chan(tf, Channel::ScaleX), PDst::Chan(gr, Channel::GainR));
        assert_eq!(gain_r(&g, gr, 0), 1.0);
        assert_eq!(gain_r(&g, gr, 5), 3.0);
        assert_eq!(gain_r(&g, gr, 10), 5.0);
    }

    #[test]
    fn expr_node_sees_t() {
        let (mut g, tf, gr) = rig();
        let m = g.patch.add_mod(ModKind::Expr("a + t*10".into()), [0.0, 0.0]);
        g.patch.link(PSrc::Chan(tf, Channel::ScaleX), PDst::Mod(m, 0));
        g.patch.link(PSrc::Mod(m), PDst::Chan(gr, Channel::GainR));
        assert_eq!(gain_r(&g, gr, 2), 23.0); // 3 + 2*10
    }

    #[test]
    fn merge_nodes_combine_two_sources() {
        let (mut g, tf, gr) = rig();
        let m = g.patch.add_mod(ModKind::Max(0.0), [0.0, 0.0]);
        g.patch.link(PSrc::Chan(tf, Channel::ScaleX), PDst::Mod(m, 0));
        g.patch.link(PSrc::Chan(tf, Channel::ScaleY), PDst::Mod(m, 1));
        g.patch.link(PSrc::Mod(m), PDst::Chan(gr, Channel::GainR));
        assert_eq!(gain_r(&g, gr, 0), 3.0); // max(3, 1)
    }

    #[test]
    fn chained_channels_read_driven_values() {
        let (mut g, tf, gr) = rig();
        // ScaleX -> GainR, then GainR -> GainB: B sees the driven R.
        g.patch.link(PSrc::Chan(tf, Channel::ScaleX), PDst::Chan(gr, Channel::GainR));
        g.patch.link(PSrc::Chan(gr, Channel::GainR), PDst::Chan(gr, Channel::GainB));
        let eng = g.to_engine_graph_at(Some(0));
        assert_eq!(crate::anim::get(&eng.nodes[&gr], Channel::GainB).unwrap(), 3.0);
    }

    /// The user's gesture: a wire from a downstream output back into an
    /// upstream input does not refuse — it builds the loop. The back-port's
    /// old feed becomes the seed, consumers of the source move to the
    /// LoopOut, and the engine graph it lowers to actually iterates.
    #[test]
    fn a_backwards_wire_becomes_a_loop() {
        let mut g = EditorGraph::default();
        let read = g.add(NodeKind::read(0, ""), Pos2::new(0.0, 0.0));
        let grade = g.add(NodeKind::Grade(Default::default()), Pos2::new(0.0, 100.0));
        let blend = g.add(NodeKind::blend(), Pos2::new(0.0, 200.0));
        g.connect(grade, 0, read);
        g.connect(blend, 0, grade);
        // Drag blend's output back into grade's input 0 — a cycle.
        let lo = connect_or_bridge(&mut g, blend, 0, grade, 0, 1.0).expect("lowered");
        let li = match &g.node(lo).unwrap().kind {
            NodeKind::LoopOut(p) => p.pair,
            k => panic!("expected LoopOut, got {k:?}"),
        };
        assert!(matches!(g.node(li).unwrap().kind, NodeKind::LoopIn));
        // grade's input 0 is now the tap; its old feed seeds the loop.
        assert_eq!(g.edges.iter().find(|(d, i, _)| (*d, *i) == (grade, 0)).unwrap().2, li);
        assert_eq!(g.edges.iter().find(|(d, i, _)| (*d, *i) == (lo, 1)).unwrap().2, read);
        // the body wires into the LoopOut.
        assert_eq!(g.edges.iter().find(|(d, i, _)| (*d, *i) == (lo, 0)).unwrap().2, blend);
        // and the lowered engine graph really iterates.
        let eng = g.to_engine_graph();
        assert!(engine::codegen::loop_ok(&eng, lo, li));
    }

    /// A self-wire (a node into its own input) is the minimal loop.
    #[test]
    fn a_self_wire_becomes_a_loop() {
        let mut g = EditorGraph::default();
        let read = g.add(NodeKind::read(0, ""), Pos2::new(0.0, 0.0));
        let grade = g.add(NodeKind::Grade(Default::default()), Pos2::new(0.0, 100.0));
        g.connect(grade, 0, read);
        let lo = connect_or_bridge(&mut g, grade, 0, grade, 0, 1.0).expect("lowered");
        let eng = g.to_engine_graph();
        let li = match &g.node(lo).unwrap().kind {
            NodeKind::LoopOut(p) => p.pair,
            _ => unreachable!(),
        };
        assert!(engine::codegen::loop_ok(&eng, lo, li));
    }

    /// Mask ports do not loop; the refusal stays a refusal.
    #[test]
    fn a_mask_port_still_refuses_a_cycle() {
        let mut g = EditorGraph::default();
        let read = g.add(NodeKind::read(0, ""), Pos2::new(0.0, 0.0));
        let grade = g.add(NodeKind::Grade(Default::default()), Pos2::new(0.0, 100.0));
        g.connect(grade, 0, read);
        let before = g.nodes.len();
        // Grade input 1 is its mask.
        assert!(connect_or_bridge(&mut g, grade, 0, grade, 1, 1.0).is_none());
        assert_eq!(g.nodes.len(), before, "nothing spawned");
    }

    /// Dropping the fat cable on an image input inserts a ChannelSelect,
    /// exactly as an image wire on a mask input inserts a ToMatte.
    #[test]
    fn a_channels_wire_on_an_image_input_bridges_a_select() {
        let mut g = EditorGraph::default();
        let read = g.add(
            {
                let mut k = NodeKind::read(0, "/x/multi.exr");
                if let NodeKind::Read { multi, .. } = &mut k {
                    *multi = true;
                }
                k
            },
            Pos2::new(0.0, 0.0),
        );
        let grade = g.add(NodeKind::Grade(Default::default()), Pos2::new(0.0, 140.0));
        let bridge = connect_or_bridge(&mut g, read, 0, grade, 0, 1.0).expect("bridged");
        assert!(matches!(
            g.node(bridge).unwrap().kind,
            NodeKind::ChannelSelect(_)
        ));
        assert_eq!(g.channel_source(bridge), Some(read));
        assert_eq!(
            g.edges.iter().find(|(d, i, _)| (*d, *i) == (grade, 0)).unwrap().2,
            bridge
        );
    }

    /// A Bake dropped into a matte pipe takes the pipe's type: mask in,
    /// mask out — and retypes back the moment an image is wired instead.
    #[test]
    fn a_bake_follows_a_matte_pipe() {
        use engine::graph::PortKind;
        let mut g = EditorGraph::default();
        let spline =
            g.add(NodeKind::Spline(Default::default()), Pos2::new(0.0, 0.0));
        let bake = g.add(
            NodeKind::GraphBake { source: 0, baked: false, start: None, mask: false, window: None, matte_half: false, single: false },
            Pos2::new(0.0, 100.0),
        );
        let comb =
            g.add(NodeKind::MatteCombine(Default::default()), Pos2::new(0.0, 200.0));
        // Matte output onto the bake: connects (no refusal, no bridge)...
        assert!(connect_or_bridge(&mut g, spline, 0, bake, 0, 1.0).is_none());
        assert_eq!(
            g.edges.iter().find(|(d, i, _)| (*d, *i) == (bake, 0)).unwrap().2,
            spline
        );
        // ...and the bake now reads Mask on both sides, so it slots into
        // the matte consumer like any matte node.
        let bk = &g.node(bake).unwrap().kind;
        assert_eq!(bk.input_kind(0), PortKind::Mask);
        assert_eq!(bk.output_kind(), PortKind::Mask);
        assert!(connect_or_bridge(&mut g, bake, 0, comb, 0, 1.0).is_none());
        // Rewire an image source in: the bake retypes back.
        let read = g.add(NodeKind::read(0, ""), Pos2::new(100.0, 0.0));
        g.connect(bake, 0, read);
        assert_eq!(g.node(bake).unwrap().kind.output_kind(), PortKind::Image);
    }

    /// Invert, Gamma and Mult take a matte the same way: a roto wired into
    /// them connects with no ToMatte/bridge in between, the node becomes a
    /// matte on both sides (so a Grade's mask port takes it), a run of them
    /// retypes together, and an image wired back in turns them all back.
    #[test]
    fn pointwise_ops_follow_a_matte_pipe() {
        use engine::graph::PortKind;
        let invert = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../plugins/invert/plugin.json");
        engine::plugins::register(engine::plugins::load_manifest(&invert).unwrap());
        let mut g = EditorGraph::default();
        let spline = g.add(NodeKind::Spline(Default::default()), Pos2::new(0.0, 0.0));
        let inv = g.add(
            NodeKind::Custom { type_id: "invert".into(), params: Default::default() },
            Pos2::new(0.0, 100.0),
        );
        let gamma = g.add(NodeKind::Gamma(Default::default()), Pos2::new(0.0, 200.0));
        let mult = g.add(NodeKind::Mult(Default::default()), Pos2::new(0.0, 300.0));
        let grade = g.add(NodeKind::Grade(Default::default()), Pos2::new(200.0, 400.0));
        // Build the chain on the image pipe first: inv -> gamma -> mult.
        assert!(connect_or_bridge(&mut g, inv, 0, gamma, 0, 1.0).is_none());
        assert!(connect_or_bridge(&mut g, gamma, 0, mult, 0, 1.0).is_none());
        // A matte into the head: no bridge, and the whole run retypes.
        assert!(connect_or_bridge(&mut g, spline, 0, inv, 0, 1.0).is_none());
        for id in [inv, gamma, mult] {
            let k = &g.node(id).unwrap().kind;
            assert!(k.matte_mode(), "{} should be in the matte pipe", k.name());
            assert_eq!(k.input_kind(0), PortKind::Mask);
            assert_eq!(k.output_kind(), PortKind::Mask);
        }
        // The Mult's matte goes straight into a Grade's mask port.
        let m = g.node(grade).unwrap().kind.mask_input().unwrap();
        assert!(connect_or_bridge(&mut g, mult, 0, grade, m, 1.0).is_none());
        assert_eq!(g.input_of(grade, m), Some(mult));
        // A mask port still refuses an image op: that one needs a ToMatte.
        let read = g.add(NodeKind::read(0, ""), Pos2::new(300.0, 0.0));
        let plain = g.add(NodeKind::Gamma(Default::default()), Pos2::new(300.0, 100.0));
        g.connect(plain, 0, read);
        let gr2 = g.add(NodeKind::Grade(Default::default()), Pos2::new(300.0, 200.0));
        let bridged = connect_or_bridge(&mut g, plain, 0, gr2, m, 1.0);
        assert!(matches!(g.node(bridged.unwrap()).unwrap().kind, NodeKind::ToMatte(_)));
        // An image into the head: back to pictures, end to end.
        g.connect(inv, 0, read);
        for id in [inv, gamma, mult] {
            assert!(!g.node(id).unwrap().kind.matte_mode());
        }
        // And the flag never becomes an animation channel, even on a comp
        // whose plugin is not installed (its stored keys are the channels).
        let mut params = std::collections::BTreeMap::new();
        params.insert("amount".to_string(), 0.5);
        params.insert(engine::graph::CUSTOM_MATTE_KEY.to_string(), 1.0);
        let gone = NodeKind::Custom { type_id: "not.installed".into(), params };
        let chans: Vec<_> = crate::anim::channels_for(&gone)
            .into_iter()
            .filter(|c| matches!(c, Channel::Custom(_)))
            .collect();
        assert_eq!(chans.len(), 1, "only the amount is a channel: {chans:?}");
    }

    #[test]
    fn cycles_fall_back_to_raw() {
        let (mut g, tf, gr) = rig();
        g.patch.link(PSrc::Chan(tf, Channel::ScaleX), PDst::Chan(gr, Channel::GainR));
        g.patch.link(PSrc::Chan(gr, Channel::GainR), PDst::Chan(tf, Channel::ScaleX));
        let eng = g.to_engine_graph_at(Some(0));
        // Each side resolves against the other's raw value, nothing hangs.
        assert_eq!(crate::anim::get(&eng.nodes[&gr], Channel::GainR).unwrap(), 3.0);
        assert_eq!(crate::anim::get(&eng.nodes[&tf], Channel::ScaleX).unwrap(), 1.0);
    }

    #[test]
    fn merge_constant_stands_in_for_unplugged_b() {
        let (mut g, tf, gr) = rig();
        let m = g.patch.add_mod(ModKind::Add(5.0), [0.0, 0.0]);
        g.patch.link(PSrc::Chan(tf, Channel::ScaleX), PDst::Mod(m, 0));
        g.patch.link(PSrc::Mod(m), PDst::Chan(gr, Channel::GainR));
        assert_eq!(gain_r(&g, gr, 0), 8.0); // 3 + k
        // Wiring the b jack overrides the constant...
        g.patch.link(PSrc::Chan(tf, Channel::ScaleY), PDst::Mod(m, 1));
        assert_eq!(gain_r(&g, gr, 0), 4.0); // 3 + ScaleY(1)
        // ...and unplugging it hands b back to the constant.
        g.patch.unlink(PDst::Mod(m, 1));
        assert_eq!(gain_r(&g, gr, 0), 8.0);
    }

    #[test]
    fn removing_a_mod_drops_its_cables() {
        let (mut g, tf, gr) = rig();
        let m = g.patch.add_mod(ModKind::Mult(2.0), [0.0, 0.0]);
        g.patch.link(PSrc::Chan(tf, Channel::ScaleX), PDst::Mod(m, 0));
        g.patch.link(PSrc::Mod(m), PDst::Chan(gr, Channel::GainR));
        g.patch.remove_mod(m);
        assert!(g.patch.links.is_empty());
        assert_eq!(gain_r(&g, gr, 0), 1.0);
    }
}

#[cfg(test)]
mod catalog_tests {
    use super::*;
    /// Every input of every node the menus offer says what it is FOR on
    /// hover: no socket is left with the bare "image input" / "input".
    #[test]
    fn every_input_socket_explains_itself() {
        for e in node_catalog() {
            let k = (e.make)(0);
            for i in 0..k.input_count() {
                let tip = input_tip(&k, i);
                assert!(tip.contains('\n'), "{} input {i}: no description ({tip:?})", e.label);
                let first = tip.lines().next().unwrap();
                if i > 0 {
                    assert!(
                        !first.starts_with("image input") && !first.starts_with("input:"),
                        "{} input {i} is unnamed: {tip:?}",
                        e.label
                    );
                }
            }
        }
        let conv = NodeKind::Convolve(Default::default());
        let tip = input_tip(&conv, engine::nodes::convolve::KERNEL_INPUT);
        assert!(tip.contains("convolution kernel"), "{tip}");
    }

    #[test]
    #[ignore]
    fn zz_dump_input_tips() {
        for e in node_catalog() {
            let k = (e.make)(0);
            for i in 0..k.input_count() {
                println!("{} #{i}: {}", e.label, input_tip(&k, i).replace('\n', " | "));
            }
        }
    }

    /// The names people search for these by find EdgeExtend in Tab search, with
    /// the mode they mean.
    #[test]
    fn search_aliases_find_edge_extend_in_the_right_mode() {
        use engine::graph::ExtendMode;
        let all: Vec<(String, std::sync::Arc<dyn Fn(usize) -> NodeKind + Send + Sync>)> = node_catalog()
            .into_iter()
            .chain(search_aliases())
            .map(|e| (e.label.to_string(), e.make))
            .collect();
        for (q, mode) in [
            ("pixel spread", ExtendMode::Spread),
            ("spread", ExtendMode::Spread),
            ("edge expand", ExtendMode::Expand),
            ("edge expander", ExtendMode::Expand),
            ("expand", ExtendMode::Expand),
        ] {
            let (shown, _) = rank_search(all.clone(), q, &Default::default());
            let (name, make) = shown.first().unwrap_or_else(|| panic!("{q}: no hit"));
            match make(0) {
                NodeKind::EdgeExtend(p) => assert_eq!(p.mode, mode, "{q} -> {name}"),
                other => panic!("{q} found {name} ({})", other.name()),
            }
        }
        for a in search_aliases() {
            assert_eq!(a.status, status_of(&(a.make)(0)));
            assert!(!node_catalog().iter().any(|e| e.label == a.label), "{} is an alias, not a node", a.label);
        }
    }

    /// Socket labels earn their place or stay off. A field node's sockets
    /// are its PARAMETERS — which is which is the whole question when seven
    /// of them fan out of one node — while "image" on every node in a comp
    /// is a label that costs attention and gives nothing back.
    #[test]
    fn only_sockets_that_say_something_are_labelled() {
        use engine::field::FieldNode;
        // The rule the editor draws by.
        let shown = |k: &NodeKind, i: usize| match k.input_name(i) {
            _ if !k.labels_inputs() => None,
            "image" | "input" => None,
            other => Some(other),
        };
        // A field fBm names all seven, in socket order: nothing about the
        // shape of a nodule says which one is lacunarity.
        let fbm = NodeKind::Field(FieldNode::default());
        assert_eq!(shown(&fbm, 0), Some("scale x"));
        assert_eq!(shown(&fbm, 4), Some("lacunarity"));
        assert_eq!(shown(&fbm, 6), Some("seed"));
        // A Blend says it with + and −, and a Defocus with Z and B: a word
        // beside those would be a second label for the same fact.
        assert_eq!(shown(&NodeKind::blend(), 1), None);
        assert_eq!(shown(&NodeKind::Defocus(Default::default()), 2), None);
        // And a plain image input never had anything to say.
        assert_eq!(shown(&NodeKind::Blur(Default::default()), 0), None);
        // The names themselves stay available for tooltips and panels even
        // where the graph does not draw them.
        assert_eq!(NodeKind::blend().input_name(1), "background (under)");
    }

    /// The add menus are built from one catalog, and the invariant that
    /// actually bites is the reader flag: a reader-backed node added without
    /// it hands two nodes the same slot, and they then fight over one
    /// texture — which looks like a decode bug, not a menu bug.
    #[test]
    fn every_entry_declares_whether_it_owns_a_reader_slot() {
        for e in node_catalog() {
            let kind = (e.make)(7);
            assert_eq!(
                e.reader,
                kind.reader_slot().is_some(),
                "{}: reader flag disagrees with the node it builds",
                e.label
            );
            if e.reader {
                assert_eq!(
                    kind.reader_slot(),
                    Some(7),
                    "{}: must take the slot it is given",
                    e.label
                );
            }
        }
    }

    #[test]
    fn labels_are_unique_and_every_category_is_populated() {
        let catalog = node_catalog();
        // Uniqueness is PER WORLD, not global. Field nodes are only ever
        // offered inside a Noise node and comp nodes only outside one — the
        // menus filter on exactly that — so a noise fBm and an image fBm can
        // share a name without either menu ever being ambiguous. Renaming
        // one of them to keep a global check happy would be inventing a
        // distinction the user never sees.
        // Three worlds now: the comp, a noise field, a 2dini drawing.
        for (in_noise, in_geo, in_sdf, in_particles, world) in [
            (false, false, false, false, "comp"),
            (true, false, false, false, "field"),
            (false, true, false, false, "geo"),
            (false, false, true, false, "sdf"),
            (false, false, false, true, "particles"),
        ] {
            let mut labels: Vec<&str> = catalog
                .iter()
                .filter(|e| offered(e.cat, in_noise, in_geo, in_sdf, in_particles))
                .map(|e| e.label)
                .collect();
            labels.sort_unstable();
            let before = labels.len();
            labels.dedup();
            assert_eq!(before, labels.len(), "two entries share a label on the {world} side");
        }
        // An empty submenu is a menu that opens onto nothing.
        for cat in NodeCat::ALL {
            assert!(
                catalog.iter().any(|e| e.cat == cat),
                "category {} has no nodes",
                cat.name()
            );
        }
    }

    /// Every node the menus offer carries a maturity, read off the kind it
    /// builds — one function, so the menu tag, the graph mark and the
    /// project filter cannot disagree. This pins the CURRENT policy:
    /// everything alpha except Transform and CornerPin (beta), nothing
    /// production yet. Graduating a node means changing `status_of` and
    /// this list together, on purpose.
    #[test]
    fn every_entry_has_the_status_its_kind_declares_and_the_policy_holds() {
        let mut beta: Vec<&str> = Vec::new();
        for e in node_catalog() {
            assert_eq!(e.status, status_of(&(e.make)(0)), "{}", e.label);
            match e.status {
                NodeStatus::Production => panic!("{}: nothing is production yet", e.label),
                NodeStatus::Beta => beta.push(e.label),
                NodeStatus::Alpha => {}
            }
        }
        beta.sort_unstable();
        assert_eq!(beta, ["CornerPin", "Transform"]);
        // Plugins, tools and controls are alpha too.
        let custom = NodeKind::Custom { type_id: "x".into(), params: Default::default() };
        assert_eq!(status_of(&custom), NodeStatus::Alpha);
        // The palette is a rule: red for alpha, lime for beta, white for
        // production — and never orange anywhere in the UI.
        let (a, b, p) =
            (NodeStatus::Alpha.color(), NodeStatus::Beta.color(), NodeStatus::Production.color());
        assert!(a.r() > a.g() + 0x60 && a.r() > a.b() + 0x60, "alpha is red: {a:?}");
        assert!(b.g() > b.r() && b.g() > b.b() + 0x60, "beta is lime: {b:?}");
        assert!(p.r() == p.g() && p.g() == p.b() && p.r() > 0xd0, "production is white: {p:?}");
        for s in NodeStatus::ALL {
            let c = s.color();
            let orange = c.r() > c.g() && c.g() > c.b() && c.g() * 4 > c.r() * 5 / 2 && c.g() < c.r() * 4 / 5;
            assert!(!orange, "{}: orange is banned from the UI: {c:?}", s.name());
        }
        assert!(NodeStatus::Production.mark().is_none(), "production draws no mark");
    }

    /// The project filter is a threshold: it admits its own status and
    /// everything more mature, and a setup written before it existed shows
    /// everything.
    #[test]
    fn the_status_filter_is_a_threshold() {
        use NodeStatus::*;
        assert!(Alpha.admits(Alpha) && Alpha.admits(Beta) && Alpha.admits(Production));
        assert!(!Beta.admits(Alpha) && Beta.admits(Beta) && Beta.admits(Production));
        assert!(!Production.admits(Alpha) && !Production.admits(Beta));
        assert!(Production.admits(Production));
        let g: EditorGraph =
            serde_json::from_str(r#"{"nodes":{},"edges":[],"next_id":0}"#).unwrap();
        assert_eq!(g.node_status_min, Alpha);
        // What the comp menus offer at each threshold, under the current
        // policy: everything, the two betas, nothing.
        let offered_at = |min: NodeStatus| {
            node_catalog()
                .into_iter()
                .filter(|e| offered(e.cat, false, false, false, false) && min.admits(e.status))
                .count()
        };
        let all = node_catalog().into_iter().filter(|e| offered(e.cat, false, false, false, false)).count();
        assert_eq!(offered_at(Alpha), all);
        assert_eq!(offered_at(Beta), 2);
        assert_eq!(offered_at(Production), 0);
    }

    /// The Tab search lists nothing until you type and never more than a
    /// few matches, with names that start with the query ahead of names
    /// that only contain it — so "lum" puts LumaKey first and "a" does not
    /// unroll the catalog down the screen.
    #[test]
    fn the_search_shows_a_few_best_matches_not_the_catalog() {
        let names = || -> Vec<(String, usize)> {
            node_catalog()
                .into_iter()
                .filter(|e| offered(e.cat, false, false, false, false))
                .enumerate()
                .map(|(i, e)| (e.label.to_string(), i))
                .collect()
        };
        assert!(rank_search(names(), "", &Default::default()).0.is_empty(), "an empty query lists nothing");
        let (shown, total) = rank_search(names(), "a", &Default::default());
        assert!(shown.len() <= SEARCH_MAX, "{} shown", shown.len());
        assert!(total > SEARCH_MAX, "a one-letter query should overflow the cap: {total}");
        // Prefix matches lead, and the catalog order holds inside a group.
        let (shown, _) = rank_search(names(), "gr", &Default::default());
        let starts: Vec<bool> = shown.iter().map(|(n, _)| n.to_lowercase().starts_with("gr")).collect();
        assert!(starts.first() == Some(&true), "{shown:?}");
        assert!(starts.windows(2).all(|w| w[0] || !w[1]), "an inner match jumped a prefix match: {shown:?}");
        assert_eq!(rank_search(names(), "lum", &Default::default()).0[0].0, "LumaKey");
        assert_eq!(rank_search(names(), "zzz", &Default::default()).0.len(), 0);
        // A whole name typed out wins over an earlier entry it prefixes.
        assert_eq!(rank_search(names(), "blend", &Default::default()).0[0].0, "Blend");
    }

    /// The query's letters only have to appear IN ORDER, so the initials
    /// of a CamelCase name reach it: this is what lets a hand type "chsl"
    /// at speed instead of spelling ChannelSelect out.
    #[test]
    fn the_search_is_fuzzy() {
        let names = || -> Vec<(String, usize)> {
            node_catalog()
                .into_iter()
                .filter(|e| offered(e.cat, false, false, false, false))
                .enumerate()
                .map(|(i, e)| (e.label.to_string(), i))
                .collect()
        };
        for (q, want) in [
            ("chsl", "ChannelSelect"),
            ("tfm", "Transform"),
            ("lk", "LumaKey"),
            ("blr", "Blur"),
            ("sat", "Saturation"),
        ] {
            let (shown, _) = rank_search(names(), q, &Default::default());
            assert_eq!(shown.first().map(|(n, _)| n.as_str()), Some(want), "{q:?} -> {shown:?}");
        }
        // Out of order is not a match: "rt" is not "Transform" backwards.
        assert!(rank_search(names(), "mrofsnart", &Default::default()).0.is_empty());
    }

    /// A scattered match never elbows past a literal one. The tiers exist
    /// so that adding fuzzy cannot change what a query that already worked
    /// does — type the name and you get the name.
    #[test]
    fn a_fuzzy_match_never_outranks_a_literal_one() {
        let all = || -> Vec<(String, usize)> {
            ["Blur", "Bevel Ultra", "Blue", "Unblur"]
                .into_iter()
                .enumerate()
                .map(|(i, n)| (n.to_string(), i))
                .collect()
        };
        let (shown, total) = rank_search(all(), "blu", &Default::default());
        let list: Vec<&str> = shown.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(total, 4, "{list:?}");
        // prefix, prefix, contains, then the one that only spells it out.
        assert_eq!(list, ["Blur", "Blue", "Unblur", "Bevel Ultra"]);
        // And within a tier the tighter match leads: a letter that starts
        // a word beats one buried mid-word.
        let cs = || -> Vec<(String, usize)> {
            ["Crossfade", "ColorSpace"].into_iter().enumerate().map(|(i, n)| (n.to_string(), i)).collect()
        };
        assert_eq!(rank_search(cs(), "cs", &Default::default()).0[0].0, "ColorSpace");
    }

    /// The nodes a hand actually adds come first: an empty query lists
    /// them by count (nothing on a fresh install), and a typed one lets
    /// habit reorder close matches — but never lift a scattered match
    /// over a literal one, nor past a name typed out whole.
    #[test]
    fn the_search_puts_the_most_used_first() {
        let all = || -> Vec<(String, usize)> {
            ["Grade", "Grain", "Grid", "Blur", "Bevel Ultra", "Merge"]
                .into_iter()
                .enumerate()
                .map(|(i, n)| (n.to_string(), i))
                .collect()
        };
        let uses: std::collections::BTreeMap<String, u32> =
            [("Grid", 9), ("Merge", 2), ("Bevel Ultra", 50)]
                .into_iter()
                .map(|(n, c)| (n.to_string(), c))
                .collect();
        let names = |v: Vec<(String, usize)>| v.into_iter().map(|(n, _)| n).collect::<Vec<_>>();
        // Empty: the used ones, most first; the unused are left out.
        let (shown, total) = rank_search(all(), "", &uses);
        assert_eq!(names(shown), ["Bevel Ultra", "Grid", "Merge"]);
        assert_eq!(total, 3);
        assert!(rank_search(all(), "", &Default::default()).0.is_empty());
        // Typed: three equal prefix matches, the used one leads.
        assert_eq!(rank_search(all(), "gr", &uses).0[0].0, "Grid");
        // Habit cannot cross a tier: "blu" is Blur's prefix and only a
        // subsequence of Bevel Ultra, however often that one is added.
        assert_eq!(names(rank_search(all(), "blu", &uses).0), ["Blur", "Bevel Ultra"]);
        // And the exact name still wins.
        assert_eq!(rank_search(all(), "grade", &uses).0[0].0, "Grade");
        // The bonus is bounded.
        assert_eq!(use_bonus(0), 0);
        assert!(use_bonus(1) > 0 && use_bonus(1_000_000) == 24);
    }

    /// ↑/↓ wrap, because the list is short enough that coming round again
    /// is what a hand holding the key expects.
    #[test]
    fn the_search_selection_wraps() {
        assert_eq!(step_sel(0, 1, 3), 1);
        assert_eq!(step_sel(2, 1, 3), 0);
        assert_eq!(step_sel(0, -1, 3), 2);
        assert_eq!(step_sel(1, 0, 3), 1);
        assert_eq!(step_sel(0, 1, 0), 0, "nothing to walk");
    }

    /// Anything a user can add has to survive the round trip through a
    /// setup file, and has to compile: a menu entry that produces a node the
    /// engine cannot render is worse than no entry.
    #[test]
    fn every_catalog_entry_builds_a_renderable_node() {
        for e in node_catalog() {
            let kind = (e.make)(0);
            let mut g = engine::graph::Graph::default();
            g.nodes.insert(1, kind.clone());
            g.view = Some(1);
            // Sources render on their own; everything else gets a Read.
            if kind.input_count() > 0 {
                g.nodes.insert(0, NodeKind::read(0, ""));
                g.edges.insert((1, 0), 0);
            }
            let plan = engine::codegen::generate(&g);
            assert!(
                plan.main.wgsl.contains("@compute"),
                "{}: produced no shader",
                e.label
            );
        }
    }
}

#[cfg(test)]
mod bridge_tests {
    use super::*;

    /// Dragging an IMAGE at a matte input used to do nothing at all. It now
    /// drops a ToMatte in the gap and wires both halves, so the gesture the
    /// user tried is the gesture that works.
    #[test]
    fn an_image_dropped_on_a_matte_input_gets_a_converter() {
        let mut g = EditorGraph::default();
        let read = g.add(NodeKind::read(0, ""), Pos2::new(0.0, 0.0));
        let noise = g.add(NodeKind::Grid(Default::default()), Pos2::new(200.0, 0.0));
        let mult = g.add(NodeKind::Mult(Default::default()), Pos2::new(0.0, 200.0));
        g.connect(mult, 0, read);

        // Input 1 of a Mult is its matte, and a Grid is an image.
        assert_eq!(g.node(mult).unwrap().kind.input_kind(1), PortKind::Mask);
        assert_eq!(g.node(noise).unwrap().kind.output_kind(), PortKind::Image);

        let made = connect_or_bridge(&mut g, noise, 0, mult, 1, 1.0)
            .expect("an image at a matte input should bridge");
        assert!(
            matches!(g.node(made).unwrap().kind, NodeKind::ToMatte(_)),
            "the bridge should be a ToMatte"
        );
        // Wired through, both halves.
        assert_eq!(g.input_of(made, 0), Some(noise));
        assert_eq!(g.input_of(mult, 1), Some(made));
        // And it lands between the two, where the wire was drawn.
        assert_eq!(g.node(made).unwrap().pos, Pos2::new(100.0, 100.0));

        // A compatible pair still connects DIRECTLY — no node is invented
        // where none is needed.
        let before = g.nodes.len();
        assert_eq!(connect_or_bridge(&mut g, read, 0, mult, 0, 1.0), None);
        assert_eq!(g.input_of(mult, 0), Some(read));
        assert_eq!(g.nodes.len(), before, "a matching pair must not add a node");

        // A mask at an IMAGE input has no obvious conversion, so it is still
        // refused rather than silently guessed at. (Not a Mult's: a Mult
        // follows its pipe and takes the matte — `pointwise_ops_follow_a_
        // matte_pipe`. A saturation of coverage means nothing.)
        let sat = g.add(NodeKind::Saturation(Default::default()), Pos2::new(0.0, 400.0));
        g.connect(sat, 0, read);
        let spline = g.add(NodeKind::Spline(Default::default()), Pos2::ZERO);
        let before = g.nodes.len();
        assert_eq!(connect_or_bridge(&mut g, spline, 0, sat, 0, 1.0), None);
        assert_eq!(g.input_of(sat, 0), Some(read), "the old wire should stand");
        assert_eq!(g.nodes.len(), before);

        // Self-connection stays refused.
        assert_eq!(connect_or_bridge(&mut g, mult, 0, mult, 1, 1.0), None);
    }

    #[test]
    fn wiring_the_third_nodule_grades_through_the_third_band() {
        use engine::graph::{LumaBand, LumaKeyParams};
        let mut g = EditorGraph::default();
        let band = |lo: f32, hi: f32| LumaBand { lo, hi, ..Default::default() };
        let mut kp = LumaKeyParams {
            bands: vec![band(0.0, 0.3), band(0.3, 0.6), band(0.6, 1.0)],
            ranges: None,
        };
        kp.adopt_ranges(engine::graph::LUMA_SPAN);
        let n_bands = kp.bands.len();
        assert!(n_bands >= 3);
        let key = g.add(NodeKind::LumaKey(kp), Pos2::new(0.0, 0.0));
        let grade = g.add(NodeKind::Grade(Default::default()), Pos2::new(0.0, 100.0));
        let read = g.add(NodeKind::read(0, ""), Pos2::new(-100.0, -100.0));
        g.connect(key, 0, read);
        g.connect(grade, 0, read);
        // The mask input takes the THIRD nodule.
        g.connect_out(grade, 1, key, 2);
        assert_eq!(g.out_of(grade, 1), 2);
        // Output 0 records nothing, so ordinary graphs are unchanged.
        g.connect(grade, 0, read);
        assert_eq!(g.out_of(grade, 0), 0);
        assert_eq!(g.edge_out.len(), 1);

        // Through lowering: the grade's mask must come from a ONE-band key
        // carrying band 2's response, not band 0's.
        let eg = g.to_engine_graph_at(None);
        let src = eg.edges.get(&(grade, 1)).copied().expect("mask edge lost");
        let Some(NodeKind::LumaKey(lowered)) = eg.nodes.get(&src) else {
            panic!("mask source is not a LumaKey: {:?}", eg.nodes.get(&src));
        };
        assert_eq!(lowered.bands.len(), 1, "lowering must leave one output");
        let want = {
            let NodeKind::LumaKey(orig) = &g.node(key).unwrap().kind else { unreachable!() };
            orig.band(2).curve()
        };
        assert_eq!(lowered.band(0).curve(), want, "wired the wrong band");
        // And band 0 still belongs to the original node id, which is what
        // animation, formats and the caches are all keyed by.
        assert_ne!(src, key, "band 2 should have been lowered onto a new node");
    }

    /// A Blender node's fat cable: a ChannelSelect off it lowers to a pass
    /// node of the host (no reader, no slot), a pass output likewise, and
    /// the combined picture stays on the host's own id.
    #[test]
    fn a_channel_select_off_a_blender_fat_cable_lowers_to_its_pass() {
        use engine::blender::{BlenderParams, Pass};
        let mut g = EditorGraph::default();
        let bl = g.add(
            NodeKind::Blender(BlenderParams { fat: true, passes: vec![Pass::Z, Pass::Normal], ..Default::default() }),
            Pos2::new(0.0, 0.0),
        );
        assert_eq!(g.node(bl).unwrap().kind.output_kind_at(0), PortKind::Channels);
        assert_eq!(g.node(bl).unwrap().kind.output_kind_at(1), PortKind::Data, "Z is data");
        assert_eq!(g.node(bl).unwrap().kind.output_kind_at(2), PortKind::Image);
        let cs = g.add(
            NodeKind::ChannelSelect(engine::graph::ChannelSelectParams { layer: "Z".into(), ..Default::default() }),
            Pos2::new(100.0, 0.0),
        );
        g.connect(cs, 0, bl);
        assert_eq!(g.channel_source(cs), Some(bl), "a fat Blender node is a channel source");
        let grade = g.add(NodeKind::Grade(Default::default()), Pos2::new(200.0, 0.0));
        g.connect(grade, 0, cs);
        let normal = g.add(NodeKind::Grade(Default::default()), Pos2::new(200.0, 100.0));
        g.connect_out(normal, 0, bl, 2);
        let eg = g.to_engine_graph_at(None);
        let src = eg.edges.get(&(grade, 0)).copied().expect("select edge lost");
        match eg.nodes.get(&src) {
            Some(NodeKind::Blender(p)) => {
                assert_eq!((p.pass, p.host), (Some(Pass::Z), Some(bl)));
                assert_eq!(p.layer_key(src), (bl, Pass::Z.id()));
                assert!(!p.fat && p.inputs == 0, "a lowered pass is a picture, not a session");
            }
            other => panic!("the select did not lower to a pass node: {other:?}"),
        }
        let nsrc = eg.edges.get(&(normal, 0)).copied().unwrap();
        assert!(matches!(eg.nodes.get(&nsrc), Some(NodeKind::Blender(p)) if p.pass == Some(Pass::Normal) && p.host == Some(bl)));
        assert!(matches!(eg.nodes.get(&bl), Some(NodeKind::Blender(p)) if p.pass.is_none() && p.host.is_none()));
        assert!(eg.edge_out.is_empty());
    }

    /// An old stack's per-layer transform becomes a real Transform in front
    /// of that layer: same values, its keys moved onto the Transform's own
    /// channels, the plate still reaching the stack through it. A layer
    /// whose transform did nothing gets nothing spliced in.
    #[test]
    fn a_legacy_layer_transform_is_lifted_into_an_upstream_transform() {
        use crate::anim::Channel;
        use engine::graph::TransformParams;
        let mut g = EditorGraph::default();
        g.nodes.clear();
        g.edges.clear();
        let (a, b) = (
            g.add(NodeKind::read(0, ""), Pos2::new(0.0, -200.0)),
            g.add(NodeKind::read(1, ""), Pos2::new(100.0, -200.0)),
        );
        let stack = g.add(NodeKind::layer(), Pos2::ZERO);
        g.connect(stack, 0, a);
        g.connect(stack, 1, b);
        let moved = TransformParams { translate: [40.0, -8.0], rotate: 15.0, ..Default::default() };
        if let Some(n) = g.node_mut(stack) {
            if let NodeKind::Layer(p) = &mut n.kind {
                p.layers[0].legacy_xform = Some(moved);
                p.layers[1].legacy_xform = Some(TransformParams::default());
            }
            let mut c = crate::anim::AnimCurve::default();
            c.set_key(0, 1.0);
            c.set_key(10, 2.0);
            n.anim.insert(Channel::LayerSx(0), c);
        }
        assert_eq!(g.lift_layer_xforms(), 1, "only the layer that moved gets a Transform");
        let xf = g.input_of(stack, 0).expect("layer 0 still wired");
        let xn = g.node(xf).unwrap();
        assert!(matches!(xn.kind, NodeKind::Transform(t) if t == moved), "values carried");
        assert!(xn.anim.contains_key(&Channel::ScaleX), "the key moved onto the Transform");
        assert_eq!(g.input_of(xf, 0), Some(a), "the plate feeds the Transform");
        assert_eq!(g.input_of(stack, 1), Some(b), "the still layer is untouched");
        let n = g.node(stack).unwrap();
        assert!(!n.anim.contains_key(&Channel::LayerSx(0)), "no legacy curve left behind");
        let NodeKind::Layer(p) = &n.kind else { unreachable!() };
        assert!(p.layers.iter().all(|l| l.legacy_xform.is_none()), "legacy field cleared");
        assert_eq!(g.lift_layer_xforms(), 0, "a second load lifts nothing");
    }

    #[test]
    fn bridge_for_only_answers_where_a_conversion_exists() {
        let img = NodeKind::Grade(Default::default());
        let mask = NodeKind::Spline(Default::default());
        let mult = NodeKind::Mult(Default::default());
        // image -> matte: bridge.
        assert!(matches!(
            bridge_for(&img, 0, &mult, 1),
            Some(NodeKind::ToMatte(_))
        ));
        // image -> image, mask -> matte: already fine, no bridge.
        assert!(bridge_for(&img, 0, &mult, 0).is_none());
        assert!(bridge_for(&mask, 0, &mult, 1).is_none());
        // mask -> image: no answer.
        assert!(bridge_for(&mask, 0, &mult, 0).is_none());
        // The data pipe: an image at a depth port gets a ToData (never a
        // ToMatte, which would clamp the depth); a matte widens into it
        // with no node; data at a matte port has no answer.
        let defocus = NodeKind::Defocus(Default::default());
        assert!(matches!(bridge_for(&img, 0, &defocus, 2), Some(NodeKind::ToData(_))));
        assert!(bridge_for(&mask, 0, &defocus, 2).is_none(), "a matte is valid data");
        let data = NodeKind::ToData(Default::default());
        assert!(bridge_for(&data, 0, &mult, 1).is_none(), "data into a matte: refused");
    }

    /// A setup written when the depth port was an IMAGE port loads with a
    /// ToData spliced into that wire, carrying the channel the Defocus used
    /// to pick — and the Defocus reset to R, which reads the ToData's
    /// broadcast the same. `a_todata_depth_renders_what_the_direct_wire_did`
    /// (engine/tests/defocus.rs) proves the pixels agree on the GPU.
    #[test]
    fn an_old_depth_wire_loads_through_a_todata() {
        use engine::graph::{DepthChannel, MatteChannel};
        let mut g = EditorGraph::default();
        g.nodes.clear();
        g.edges.clear();
        let plate = g.add(NodeKind::read(0, ""), Pos2::new(0.0, 0.0));
        let z = g.add(NodeKind::read(1, ""), Pos2::new(200.0, 0.0));
        let mut d = engine::graph::DefocusParams::default();
        d.depth_channel = DepthChannel::A;
        let defocus = g.add(NodeKind::Defocus(d), Pos2::new(0.0, 200.0));
        g.connect(defocus, 0, plate);
        g.connect(defocus, 2, z);
        g.hidden_wires.insert((defocus, 2));

        assert_eq!(g.bridge_mistyped(), 1);
        let bridge = g.input_of(defocus, 2).unwrap();
        assert_ne!(bridge, z, "the depth now arrives through a converter");
        match g.node(bridge).map(|n| &n.kind) {
            Some(NodeKind::ToData(p)) => assert_eq!(p.channel, MatteChannel::A),
            other => panic!("wanted a ToData carrying A, got {other:?}"),
        }
        assert_eq!(g.input_of(bridge, 0), Some(z));
        assert!(matches!(
            g.node(defocus).map(|n| &n.kind),
            Some(NodeKind::Defocus(d)) if d.depth_channel == DepthChannel::R
        ));
        assert_eq!(g.input_of(defocus, 0), Some(plate), "the picture wire is untouched");
        assert!(g.is_hidden(defocus, 2), "a hidden wire stays hidden");
        // Idempotent: a second load finds nothing left to bridge.
        assert_eq!(g.bridge_mistyped(), 0);
    }

    /// Wiring by hand obeys the same rule: a matte goes straight into a
    /// depth port, an image arrives through a ToData.
    #[test]
    fn a_drawn_wire_into_a_depth_port_takes_the_data_pipe() {
        let mut g = EditorGraph::default();
        g.nodes.clear();
        g.edges.clear();
        let img = g.add(NodeKind::read(0, ""), Pos2::new(0.0, 0.0));
        let key = g.add(NodeKind::Spline(Default::default()), Pos2::new(200.0, 0.0));
        let d = g.add(NodeKind::Defocus(Default::default()), Pos2::new(0.0, 200.0));
        assert!(connect_or_bridge(&mut g, key, 0, d, 2, 1.0).is_none(), "no node for a matte");
        assert_eq!(g.input_of(d, 2), Some(key));
        let made = connect_or_bridge(&mut g, img, 0, d, 2, 1.0).expect("an image needs a ToData");
        assert!(matches!(g.node(made).map(|n| &n.kind), Some(NodeKind::ToData(_))));
        assert_eq!(g.input_of(d, 2), Some(made));
    }
}

#[cfg(test)]
mod placement_tests {
    use super::*;

    /// Everything that reads a placed plate plays at the plate's frames.
    ///
    /// A ChannelSelect lowers to a Read of the same sequence and a
    /// Cryptomatte's ranks are readers of it too — both used to lower
    /// UNPLACED, so a harp string (or a crypto matte) showed frame one at
    /// the comp frame where the Read itself showed frame 1001.
    #[test]
    fn lowered_channel_readers_inherit_the_reads_placement() {
        let mut g = EditorGraph::default();
        let read = g.add(
            NodeKind::Read {
                source: 0,
                path: "plate".into(),
                range: None,
                ram_cache: false,
                nvme_cache: true,
                layer: None,
                input: Default::default(),
                multi: true,
                start: Some(1001),
                first: None,
                last: None,
                edges: Default::default(), cache_on_play: true,
                single: Default::default(), eight_bit: false,
            },
            Pos2::ZERO,
        );
        let cs = g.add(
            NodeKind::ChannelSelect(engine::graph::ChannelSelectParams {
                source: 7,
                layer: "depth".into(),
            }),
            Pos2::new(0.0, 100.0),
        );
        g.connect(cs, 0, read);
        let ck = g.add(NodeKind::Cryptomatte(Default::default()), Pos2::new(100.0, 100.0));
        g.connect(ck, 0, read);
        g.view = Some(cs);

        let eg = g.to_engine_graph();
        assert_eq!(
            eg.nodes.get(&cs).and_then(|k| k.placement()),
            Some(1001),
            "the selected channel must play where its plate plays"
        );
        assert_eq!(
            eg.nodes.get(&ck).and_then(|k| k.placement()),
            Some(1001),
            "crypto ranks must play where their plate plays"
        );
    }
}

#[cfg(test)]
mod link_view_tests {
    use super::*;
    use egui::epaint;

    /// Rasterize egui's tessellated output so the graph can be LOOKED at
    /// without a window. Text is skipped (it lives in a font atlas this has
    /// no access to); every other shape is exactly what the GPU would get.
    /// Same approach as the keyer panel's test.
    fn rasterize(prims: &[egui::ClippedPrimitive], w: usize, h: usize, ss: f32) -> Vec<u8> {
        let mut buf = vec![20u8; w * h * 3];
        for p in prims {
            let epaint::Primitive::Mesh(m) = &p.primitive else { continue };
            for tri in m.indices.chunks_exact(3) {
                let v: Vec<&epaint::Vertex> =
                    tri.iter().map(|i| &m.vertices[*i as usize]).collect();
                // Everything is drawn, including atlas-textured quads, with
                // the vertex colour standing in for the texture. That is why
                // a small filled circle shows up here as its QUAD: egui
                // draws little circles from a pre-rendered atlas sprite, not
                // as geometry. Filtering those out (by UV, as the obvious
                // "skip the glyphs" test does) hides the channel nodules
                // entirely — the render is set up below to avoid drawing
                // text instead.
                let pts: Vec<[f32; 2]> =
                    v.iter().map(|q| [q.pos.x * ss, q.pos.y * ss]).collect();
                let (mut x0, mut x1) = (f32::MAX, f32::MIN);
                let (mut y0, mut y1) = (f32::MAX, f32::MIN);
                for q in &pts {
                    x0 = x0.min(q[0]);
                    x1 = x1.max(q[0]);
                    y0 = y0.min(q[1]);
                    y1 = y1.max(q[1]);
                }
                let d = (pts[1][0] - pts[0][0]) * (pts[2][1] - pts[0][1])
                    - (pts[2][0] - pts[0][0]) * (pts[1][1] - pts[0][1]);
                if d.abs() < 1e-9 {
                    continue;
                }
                for y in (y0.floor().max(0.0) as usize)..(y1.ceil().min(h as f32) as usize) {
                    for x in (x0.floor().max(0.0) as usize)..(x1.ceil().min(w as f32) as usize)
                    {
                        let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
                        let b1 = ((px - pts[0][0]) * (pts[2][1] - pts[0][1])
                            - (pts[2][0] - pts[0][0]) * (py - pts[0][1]))
                            / d;
                        let b2 = ((pts[1][0] - pts[0][0]) * (py - pts[0][1])
                            - (px - pts[0][0]) * (pts[1][1] - pts[0][1]))
                            / d;
                        let b0 = 1.0 - b1 - b2;
                        // Accept EITHER winding: egui emits fills and strokes
                        // with opposite orientation, and a one-sided test
                        // silently drops every filled shape (this cost an
                        // hour of "why are the nodules invisible").
                        let inside = (b0 >= 0.0 && b1 >= 0.0 && b2 >= 0.0)
                            || (b0 <= 0.0 && b1 <= 0.0 && b2 <= 0.0);
                        if !inside {
                            continue;
                        }
                        let ws = [b0, b1, b2];
                        let mut c = [0.0f32; 4];
                        for (k, q) in v.iter().enumerate() {
                            let a = q.color.to_array();
                            for j in 0..4 {
                                c[j] += a[j] as f32 * ws[k];
                            }
                        }
                        let a = c[3] / 255.0;
                        let o = (y * w + x) * 3;
                        for j in 0..3 {
                            buf[o + j] =
                                (c[j] + buf[o + j] as f32 * (1.0 - a)).clamp(0.0, 255.0) as u8;
                        }
                    }
                }
            }
        }
        buf
    }

    fn render(link_mode: bool, name: &str) -> (Vec<u8>, usize, usize) {
        let (w_pt, h_pt, ss) = (620.0f32, 420.0f32, 2.0f32);
        let mut g = EditorGraph::default();
        let r = g.add(NodeKind::read(0, ""), Pos2::new(-150.0, -130.0));
        let tf = g.add(NodeKind::Transform(Default::default()), Pos2::new(-150.0, 10.0));
        let gr = g.add(NodeKind::Grade(Default::default()), Pos2::new(120.0, 10.0));
        let mg = g.add(NodeKind::blend(), Pos2::new(-20.0, 140.0));
        g.connect(tf, 0, r);
        g.connect(mg, 0, tf);
        g.connect(mg, 1, gr);
        // One cable, so the link view has something wired to draw.
        g.patch.link(
            PSrc::Chan(tf, crate::anim::Channel::Rotate),
            PDst::Chan(gr, crate::anim::Channel::GainR),
        );

        let ctx = egui::Context::default();
        let mut ed = NodeEditor { link_mode, ..Default::default() };
        // Under the label threshold (z * size > 0.75), so the picture is
        // tentacles and nodules with no text to smear.
        ed.zoom = 0.7;
        // Nothing selected: the selection bbox and its white handles would
        // sit on top of the very sockets this test samples.
        let (mut sel, mut act, mut pins, mut slot, mut rects) =
            (Vec::new(), vec![gr], vec![tf], 2usize, Vec::new());
        let mut input = egui::RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(w_pt, h_pt))),
            ..Default::default()
        };
        input.viewports.values_mut().for_each(|v| v.native_pixels_per_point = Some(1.0));
        let out = ctx.run_ui(input, |ui| {
            ed.show(ui, &mut g, &mut sel, &mut act, &mut pins, &mut slot, &mut rects);
        });
        let prims = ctx.tessellate(out.shapes, 1.0);
        let (w, h) = ((w_pt * ss) as usize, (h_pt * ss) as usize);
        let buf = rasterize(&prims, w, h, ss);
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../target")
            .join(name);
        if let Ok(file) = std::fs::File::create(&path) {
            let mut enc =
                png::Encoder::new(std::io::BufWriter::new(file), w as u32, h as u32);
            enc.set_color(png::ColorType::Rgb);
            enc.set_depth(png::BitDepth::Eight);
            if let Ok(mut wr) = enc.write_header() {
                let _ = wr.write_image_data(&buf);
            }
        }
        (buf, w, h)
    }

    /// A GenAI Pipeline wears a bar like a Layer's: its pictures stand on
    /// the left, its mattes on the right, whatever order they were added
    /// in. Dumps `target/graph-genai-bar.png` to judge by eye.
    #[test]
    fn a_genai_bar_has_pictures_left_and_mattes_right() {
        let (w_pt, h_pt, ss) = (620.0f32, 300.0f32, 2.0f32);
        let mut g = EditorGraph::default();
        let mut p = engine::graph::GenAiParams::new(0);
        // image, matte (the default), then a picture added after the matte.
        p.pipeline.add_input(genai::InKind::Image);
        p.pipeline.add_input(genai::InKind::Matte);
        let plate = g.add(NodeKind::read(0, ""), Pos2::new(-210.0, -110.0));
        let roto = g.add(NodeKind::read(1, ""), Pos2::new(130.0, -110.0));
        let gen = g.add(NodeKind::GenAi(p), Pos2::new(-40.0, 60.0));
        g.connect(gen, 0, plate);
        g.connect(gen, 1, roto);
        let kind = g.node(gen).unwrap().kind.clone();
        assert_eq!(kind.input_count(), 4);
        let xs: Vec<(usize, f32)> = crate::breakout::bar_strings_x(&kind, Pos2::ZERO, 1.0);
        let order: Vec<usize> = xs.iter().map(|x| x.0).collect();
        assert_eq!(order, vec![0, 2, 1, 3], "pictures (0, 2) left, mattes (1, 3) right");
        assert!(xs.windows(2).all(|w| w[0].1 < w[1].1));
        assert_eq!(kind.input_kind(1), PortKind::Mask);
        assert_eq!(kind.input_kind(2), PortKind::Image);

        let ctx = egui::Context::default();
        let mut ed = NodeEditor::default();
        ed.zoom = 1.3;
        let (mut sel, mut act, mut pins, mut slot, mut rects) =
            (Vec::new(), Vec::new(), Vec::new(), 3usize, Vec::new());
        let mut input = egui::RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(w_pt, h_pt))),
            ..Default::default()
        };
        input.viewports.values_mut().for_each(|v| v.native_pixels_per_point = Some(1.0));
        let out = ctx.run_ui(input, |ui| {
            ed.show(ui, &mut g, &mut sel, &mut act, &mut pins, &mut slot, &mut rects);
        });
        let prims = ctx.tessellate(out.shapes, 1.0);
        let (w, h) = ((w_pt * ss) as usize, (h_pt * ss) as usize);
        let buf = rasterize(&prims, w, h, ss);
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../target").join("graph-genai-bar.png");
        if let Ok(file) = std::fs::File::create(&path) {
            let mut enc = png::Encoder::new(std::io::BufWriter::new(file), w as u32, h as u32);
            enc.set_color(png::ColorType::Rgb);
            enc.set_depth(png::BitDepth::Eight);
            if let Ok(mut wr) = enc.write_header() {
                let _ = wr.write_image_data(&buf);
            }
        }
    }

    /// Ports that are not interchangeable have to be readable ON the node:
    /// a Defocus takes image / mask / depth / bokeh and nothing about the
    /// hexagon says which stalk is which. Dumps `target/graph-ports.png` so
    /// the glyphs can be judged by eye, and asserts the splay is right —
    /// the mask sits at index 1, in the MIDDLE of the image inputs, so a
    /// port laid out by raw index would hang off the end of the node.
    #[test]
    fn named_ports_are_labelled_and_evenly_splayed() {
        let (w_pt, h_pt, ss) = (620.0f32, 300.0f32, 2.0f32);
        let mut g = EditorGraph::default();
        let plate = g.add(NodeKind::read(0, ""), Pos2::new(-210.0, -110.0));
        let z = g.add(NodeKind::read(1, ""), Pos2::new(-40.0, -110.0));
        let bok = g.add(NodeKind::read(2, ""), Pos2::new(130.0, -110.0));
        let df = g.add(NodeKind::Defocus(DefocusParams::default()), Pos2::new(-40.0, 60.0));
        g.connect(df, 0, plate);
        g.connect(df, 2, z);
        g.connect(df, 3, bok);

        let ctx = egui::Context::default();
        let mut ed = NodeEditor::default();
        ed.zoom = 1.3; // above the tag threshold, so the glyphs actually draw
        let (mut sel, mut act, mut pins, mut slot, mut rects) =
            (Vec::new(), Vec::new(), Vec::new(), 3usize, Vec::new());
        let mut input = egui::RawInput {
            screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(w_pt, h_pt))),
            ..Default::default()
        };
        input.viewports.values_mut().for_each(|v| v.native_pixels_per_point = Some(1.0));
        let out = ctx.run_ui(input, |ui| {
            ed.show(ui, &mut g, &mut sel, &mut act, &mut pins, &mut slot, &mut rects);
        });
        let prims = ctx.tessellate(out.shapes, 1.0);
        let (w, h) = ((w_pt * ss) as usize, (h_pt * ss) as usize);
        let buf = rasterize(&prims, w, h, ss);
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../target")
            .join("graph-ports.png");
        if let Ok(file) = std::fs::File::create(&path) {
            let mut enc =
                png::Encoder::new(std::io::BufWriter::new(file), w as u32, h as u32);
            enc.set_color(png::ColorType::Rgb);
            enc.set_depth(png::BitDepth::Eight);
            if let Ok(mut wr) = enc.write_header() {
                let _ = wr.write_image_data(&buf);
            }
        }

        let k = NodeKind::Defocus(DefocusParams::default());
        assert_eq!(k.input_tag(0), None, "the plain image port stays plain");
        assert_eq!(k.input_tag(1), None, "the mask is a colour, not a glyph");
        assert_eq!(k.input_tag(2), Some("Z"));
        assert_eq!(k.input_tag(3), Some("B"));
        assert_eq!(k.input_name(2), "depth");
        assert_eq!(k.input_name(3), "bokeh kernel");

        // Image ordinals 0,1,2 for inputs 0,2,3 — the mask at 1 is skipped,
        // not counted, so the three image stalks splay symmetrically.
        let pos = Pos2::new(0.0, 0.0);
        let n = k.image_input_count();
        assert_eq!(n, 3);
        let xs: Vec<f32> = [0usize, 2, 3]
            .iter()
            .map(|i| input_rest(pos, k.image_slot(*i), n, 1.0).x)
            .collect();
        assert!(xs[0] < xs[1] && xs[1] < xs[2], "ports must be ordered: {xs:?}");
        assert!(
            (xs[1] - pos.x).abs() < 1e-3,
            "the middle image port should sit on the node's centre, got {}",
            xs[1]
        );
        assert!(
            ((xs[2] - xs[1]) - (xs[1] - xs[0])).abs() < 1e-3,
            "splay must be even: {xs:?}"
        );
    }

    /// Link mode has to actually LOOK different: amber channel nodules
    /// splayed left and right, and no image wiring. Writes both views to
    /// `target/` so the splay can be judged by eye as well as counted.
    #[test]
    fn link_mode_splays_channels_and_drops_the_image_wiring() {
        let (normal, w, h) = render(false, "graph-normal.png");
        let (link, _, _) = render(true, "graph-links.png");
        if std::env::var("EDITOR_DBG").is_ok() {
            // The Transform's first INPUT nodule, computed the same way the
            // drawing does, sampled straight out of the raster.
            let pos = Pos2::new(-150.0, 10.0);
            let at = chan_rest(pos, 1.0, 0, 7, false);
            let screen = Pos2::new(310.0 + at.x * 0.9, 210.0 + at.y * 0.9);
            let (px, py) = ((screen.x * 2.0) as usize, (screen.y * 2.0) as usize);
            for dy in -2i32..=2 {
                let row: Vec<String> = (-2i32..=2)
                    .map(|dx| {
                        let (x, y) = ((px as i32 + dx) as usize, (py as i32 + dy) as usize);
                        let o = (y * w + x) * 3;
                        format!("{:02x}{:02x}{:02x}", link[o], link[o + 1], link[o + 2])
                    })
                    .collect();
                eprintln!("DBG bead row {dy}: {}", row.join(" "));
            }
        }
        assert_eq!(normal.len(), link.len());

        // Count pixels close to each palette colour.
        let count = |buf: &[u8], want: [u8; 3], tol: i32| -> usize {
            buf.chunks_exact(3)
                .filter(|p| {
                    (0..3).all(|j| (p[j] as i32 - want[j] as i32).abs() < tol)
                })
                .count()
        };
        let cable = [0xd8, 0xa0, 0x30];

        // Amber (cables + driven nodules) is what link mode adds.
        assert!(
            count(&link, cable, 30) > count(&normal, cable, 30) + 200,
            "link mode should paint amber nodules and a cable"
        );

        // Every nodule must actually be PAINTED where the layout says, in
        // the colour that says what it is: cyan in, magenta out, amber for
        // the one the cable drives.
        let origin = Pos2::new(310.0, 210.0);
        let at_screen = |g: Pos2| -> (usize, usize) {
            (
                ((origin.x + g.x * 0.7) * 2.0) as usize,
                ((origin.y + g.y * 0.7) * 2.0) as usize,
            )
        };
        let near = |buf: &[u8], (x, y): (usize, usize), want: [u8; 3]| -> bool {
            // The bead is a few px across; look in a small window.
            (0..5).any(|dy| {
                (0..5).any(|dx| {
                    let (x, y) = (x + dx - 2, y + dy - 2);
                    let o = (y * w + x) * 3;
                    (0..3).all(|j| (buf[o + j] as i32 - want[j] as i32).abs() < 45)
                })
            })
        };
        // Transform sits at (-150, 10); its channels — the move/rotate/scale
        // seven and the corner pin's eight.
        let tf = Pos2::new(-150.0, 10.0);
        let n_tf = node_channels(&NodeKind::Transform(Default::default())).len();
        for i in 0..n_tf {
            assert!(
                near(&link, at_screen(chan_rest(tf, 1.0, i, n_tf, false)), [0x55, 0xb8, 0xd8]),
                "input nodule {i} missing"
            );
        }
        // Rotate is channel 2: it drives the cable, so it reads amber.
        assert!(
            near(&link, at_screen(chan_rest(tf, 1.0, 2, n_tf, true)), [0xd8, 0xa0, 0x30]),
            "the driving output nodule should be amber"
        );
        assert!(
            near(&link, at_screen(chan_rest(tf, 1.0, 0, n_tf, true)), [0xd8, 0x3a, 0xae]),
            "an unpatched output nodule should be magenta"
        );
        // ...and none of that is painted in the normal view.
        assert!(
            !near(&normal, at_screen(chan_rest(tf, 1.0, 0, 7, false)), [0x55, 0xb8, 0xd8]),
            "channel nodules must not appear outside link mode"
        );
        // The image socket the normal view draws at the top of the node is
        // gone in link mode — that is the wiring stepping aside.
        let img_socket = at_screen(input_rest(tf, 0, 1, 1.0));
        // A free input is a CUP: its colour is on the rim, not the centre.
        let near_cup = |buf: &[u8], (x, y): (usize, usize), want: [u8; 3]| -> bool {
            (0..17).any(|dy| (0..17).any(|dx| near(buf, (x + dx - 8, y + dy - 8), want)))
        };
        assert!(
            near_cup(&normal, img_socket, [0x55, 0xb8, 0xd8]),
            "the normal view should show the image input socket"
        );
        assert!(
            !near(&link, img_socket, [0x55, 0xb8, 0xd8]),
            "link mode should drop the image sockets"
        );

        // The nodules must be DISCERNIBLE: scan the column of a Transform's
        // output tentacles and demand separate blobs, not one smear.
        let row_has = |buf: &[u8], y: usize, want: [u8; 3]| -> Vec<usize> {
            (0..w)
                .filter(|x| {
                    let o = (y * w + x) * 3;
                    (0..3).all(|j| (buf[o + j] as i32 - want[j] as i32).abs() < 40)
                })
                .collect()
        };
        // Somewhere down the image there is a row crossing several nodules.
        let mut best_gaps = 0;
        for y in 0..h {
            let xs = row_has(&link, y, [0x55, 0xb8, 0xd8]);
            let gaps = xs.windows(2).filter(|p| p[1] - p[0] > 3).count();
            best_gaps = best_gaps.max(gaps);
        }
        assert!(
            best_gaps >= 1,
            "input nodules should read as separate blobs, not one smear"
        );
    }
}


#[cfg(test)]
mod label_tests {
    use super::*;

    fn row(n: usize, gap: f32) -> Vec<(u64, Rect)> {
        (0..n)
            .map(|i| (i as u64, Rect::from_center_size(Pos2::new(100.0 + gap * i as f32, 300.0), Vec2::splat(40.0))))
            .collect()
    }

    #[test]
    fn a_name_with_room_stays_level() {
        let nodes = row(1, 0.0);
        let (theta, _) = place_label(0, nodes[0].1.center(), 28.0, Vec2::new(300.0, 14.0), &nodes, &[]);
        assert_eq!(theta, 0.0);
    }

    /// The screenshot: a row of Reads whose long names ran through their
    /// neighbours. Each one swings up around its own centre, and once
    /// placed no name crosses another node or another name.
    #[test]
    fn a_row_of_long_names_swings_up_and_nothing_overlaps() {
        let nodes = row(6, 170.0);
        let size = Vec2::new(330.0, 14.0);
        let mut placed = Vec::new();
        for (id, r) in &nodes {
            let (theta, q) = place_label(*id, r.center(), 28.0, size, &nodes, &placed);
            if *id < 5 {
                assert!(theta < 0.0, "node {id}'s name runs into its neighbour: it must swing up");
            } else {
                assert_eq!(theta, 0.0, "the last node has room to its right");
            }
            assert!(theta >= -std::f32::consts::FRAC_PI_2 - 1e-4, "never past vertical");
            for (other, orc) in &nodes {
                if other != id {
                    assert!(!quads_overlap(&q, &rect_quad(*orc)), "{id}'s name crosses node {other}");
                }
            }
            for p in &placed {
                assert!(!quads_overlap(&q, p), "{id}'s name crosses another name");
            }
            placed.push(q);
        }
    }

    /// Tilts only as far as the room asks: finer than the coarse step,
    /// and just clear of the neighbour.
    #[test]
    fn the_tilt_is_the_least_that_clears() {
        let nodes = row(2, 170.0);
        let size = Vec2::new(330.0, 14.0);
        let (theta, _) = place_label(0, nodes[0].1.center(), 28.0, size, &nodes, &[]);
        let step = LABEL_STEP_DEG.to_radians();
        assert!(theta < 0.0);
        // One bisection resolution flatter is blocked.
        let flatter = theta + step / (1 << LABEL_BISECT) as f32 * 1.01;
        let q = label_box(nodes[0].1.center(), 28.0, size, flatter);
        assert!(quads_overlap(&q, &rect_quad(nodes[1].1)), "{theta} is steeper than it needs to be");
    }

    /// The wobble: a name swinging to a new angle overshoots it, comes
    /// back, and settles — the graph's own spring, not a tween.
    #[test]
    fn a_name_bounces_into_place() {
        let target = -0.5f32;
        let (mut th, mut v) = (0.0f32, 0.0f32);
        let mut overshoot = false;
        for _ in 0..240 {
            (th, v) = label_spring(th, v, target, 1.0 / 60.0, 1.0);
            overshoot |= th < target - 0.02;
        }
        assert!(overshoot, "an underdamped spring must swing past the target");
        assert!((th - target).abs() < 1e-3 && v.abs() < 1e-2, "and settle: {th} {v}");
    }
    /// The graph bar's "wobble": 0 is rigid, under 1 never overshoots more
    /// than the design, over 1 swings further.
    #[test]
    fn wobbliness_scales_the_swing() {
        let peak = |w: f32| {
            let (mut th, mut v, mut max) = (0.0f32, 0.0f32, 0.0f32);
            for _ in 0..240 {
                (th, v) = label_spring(th, v, 1.0, 1.0 / 60.0, w);
                max = max.max(th);
            }
            (max, th)
        };
        assert_eq!(peak(0.0).0, 1.0, "rigid lands on the target at once");
        let (calm, end) = peak(0.3);
        let (design, _) = peak(1.0);
        let (jelly, _) = peak(3.0);
        assert!(calm < design && design < jelly, "{calm} {design} {jelly}");
        assert!((end - 1.0).abs() < 1e-2, "a calm spring still settles");
    }


    #[test]
    fn separating_axes_see_a_tilted_miss_and_a_hit() {
        let r = rect_quad(Rect::from_min_size(Pos2::new(100.0, 0.0), Vec2::splat(20.0)));
        // Level, running straight through the box: a hit.
        assert!(quads_overlap(&label_box(Pos2::ZERO, 10.0, Vec2::new(200.0, 10.0), 0.0), &r));
        // Swung up 45 degrees: its AABB still covers the box, but it misses.
        let tilted = label_box(Pos2::new(0.0, 30.0), 10.0, Vec2::new(200.0, 10.0), (-45f32).to_radians());
        assert!(Rect::from_points(&tilted).intersects(Rect::from_points(&r)));
        assert!(!quads_overlap(&tilted, &r));
    }

    /// For LOOKING: the merge bars (Blend, Extract, Layer) wired up, drawn by
    /// the real editor. `cargo test -p app merge_bars_render -- --ignored`
    #[test]
    #[ignore]
    fn merge_bars_render() {
        let mut g = EditorGraph::default();
        g.nodes.clear();
        g.edges.clear();
        g.next_id = 0;
        let src = |g: &mut EditorGraph, x: f32| {
            g.add(NodeKind::Grid(Default::default()), Pos2::new(x, -160.0))
        };
        // Blend: A, B and a matte on the porch.
        let (a, b, m) = (src(&mut g, -620.0), src(&mut g, -540.0), src(&mut g, -700.0));
        let bl = g.add(
            NodeKind::Blend { op: engine::graph::BlendOp::Over, mix: 1.0, gain: [1.0, 1.0] },
            Pos2::new(-580.0, 0.0),
        );
        g.connect(bl, 0, a);
        g.connect(bl, 1, b);
        g.connect(bl, 2, m);
        // Extract: a render on the porch, three mattes over three outputs.
        let img = src(&mut g, -380.0);
        let ex = g.add(NodeKind::Extract(engine::graph::ExtractParams { n: 3 }), Pos2::new(-240.0, 0.0));
        g.connect(ex, 0, img);
        for k in 0..3 {
            let mt = src(&mut g, -290.0 + 48.0 * k as f32);
            g.connect(ex, 1 + k, mt);
        }
        // Layer: a fresh stack's two plates.
        let ly = g.add(NodeKind::layer(), Pos2::new(200.0, 0.0));
        for k in 0..2 {
            let pl = src(&mut g, 128.0 + 48.0 * k as f32);
            g.connect(ly, k, pl);
        }
        let mut ed = NodeEditor::default();
        let (mut sel, mut act, mut pin, mut slot) = (Vec::new(), Vec::new(), Vec::new(), 7usize);
        let mut rects = Vec::new();
        crate::uishot::render("merge-bars", 1400.0, 600.0, 2.0, 3, |ui, _| {
            ed.show(ui, &mut g, &mut sel, &mut act, &mut pin, &mut slot, &mut rects);
        });
    }

    /// For LOOKING: the screenshot's row of long-named Reads, drawn by the
    /// real editor. `cargo test -p app label_collision_render -- --ignored`
    #[test]
    #[ignore]
    fn label_collision_render() {
        let mut g = EditorGraph::default();
        g.nodes.clear();
        g.edges.clear();
        g.next_id = 0;
        for (i, name) in [
            "flame05_p02_sh030_render_shelf_05_UT02_BG",
            "shift_05_UT02_sh030_render_CHAR",
            "shift_05_UT02_sh030_render_ENV",
            "shift_05_UT02_sh030_render_FG",
            "shift_05_UT02_sh030_render_FX",
            "shift_08_UT02_sh030_render_STARS",
        ]
        .into_iter()
        .enumerate()
        {
            let id = g.add(NodeKind::read(i, ""), Pos2::new(-500.0 + 200.0 * i as f32, 0.0));
            g.node_mut(id).unwrap().ui_label = name.into();
        }
        let mut ed = NodeEditor::default();
        let (mut sel, mut act, mut pin, mut slot) = (Vec::new(), Vec::new(), Vec::new(), 7usize);
        let mut rects = Vec::new();
        crate::uishot::render("label-collision", 1400.0, 500.0, 2.0, 3, |ui, _| {
            ed.show(ui, &mut g, &mut sel, &mut act, &mut pin, &mut slot, &mut rects);
        });
    }
}

/// How long the editor takes to draw a big graph, headless: the `show`
/// pass AND egui's tessellation of what it painted (both are per-frame
/// CPU on the UI thread). `cargo test --release -p app big_graph_bench
/// -- --ignored --nocapture`.
#[cfg(test)]
mod big_graph_bench {
    use super::*;

    /// `cols` chains of `len` nodes, each from a Read, all merged by a
    /// ladder of Blends — the shape a big comp has.
    pub(crate) fn build(cols: usize, len: usize) -> EditorGraph {
        let mut g = EditorGraph::default();
        g.nodes.clear();
        g.edges.clear();
        g.next_id = 0;
        let mut acc: Option<u64> = None;
        for c in 0..cols {
            let x = c as f32 * 180.0;
            let mut prev = g.add(NodeKind::read(c, ""), Pos2::new(x, 0.0));
            for k in 1..len {
                let kind = match k % 4 {
                    0 => NodeKind::Grade(Default::default()),
                    1 => NodeKind::Blur(Default::default()),
                    2 => NodeKind::Saturation(Default::default()),
                    _ => NodeKind::Gamma(Default::default()),
                };
                let n = g.add(kind, Pos2::new(x, k as f32 * 110.0));
                g.connect(n, 0, prev);
                prev = n;
            }
            acc = Some(match acc {
                None => prev,
                Some(a) => {
                    let b = g.add(NodeKind::blend(), Pos2::new(x, len as f32 * 110.0 + 120.0));
                    g.connect(b, 0, prev);
                    g.connect(b, 1, a);
                    b
                }
            });
        }
        g.view = acc;
        g
    }

    fn run(g: &mut EditorGraph, zoom: f32, frames: usize) -> (f64, f64) {
        let ctx = egui::Context::default();
        let mut ed = NodeEditor::default();
        ed.zoom = zoom;
        // Centre the view on the graph.
        let (mut lo, mut hi) = (Pos2::new(f32::MAX, f32::MAX), Pos2::new(f32::MIN, f32::MIN));
        for n in g.nodes.values() {
            lo = lo.min(n.pos);
            hi = hi.max(n.pos);
        }
        ed.pan = -((lo.to_vec2() + hi.to_vec2()) * 0.5) * zoom;
        let mut rects = Vec::new();
        let (mut show_t, mut tess_t) = (0.0, 0.0);
        for i in 0..frames + 5 {
            let input = egui::RawInput {
                screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(1600.0, 1000.0))),
                time: Some(i as f64 / 60.0),
                events: vec![egui::Event::PointerMoved(Pos2::new(800.0 + (i % 7) as f32, 500.0))],
                ..Default::default()
            };
            let (mut sel, mut act, mut pin, mut slot) = (Vec::new(), Vec::new(), Vec::new(), 2usize);
            let t0 = web_time::Instant::now();
            let out = ctx.run_ui(input, |ui| {
                ed.show(ui, g, &mut sel, &mut act, &mut pin, &mut slot, &mut rects);
            });
            let t1 = web_time::Instant::now();
            let prims = ctx.tessellate(out.shapes, out.pixels_per_point);
            std::hint::black_box(prims.len());
            let t2 = web_time::Instant::now();
            if i >= 5 {
                show_t += (t1 - t0).as_secs_f64();
                tess_t += (t2 - t1).as_secs_f64();
            }
        }
        (show_t * 1e3 / frames as f64, tess_t * 1e3 / frames as f64)
    }

    /// Writes the bench graph as a setup to `$WOBBLE_BENCH_SETUP`, to open
    /// in the real app.
    #[test]
    #[ignore]
    fn big_graph_setup() {
        let Ok(path) = std::env::var("WOBBLE_BENCH_SETUP") else { return };
        let cols = std::env::var("WOBBLE_BENCH_COLS").ok().and_then(|c| c.parse().ok()).unwrap_or(50);
        let mut g = build(cols, 20);
        // Viewing it all would need more uniform space than one fused pass
        // has; the bench is about the graph's own drawing.
        g.view = None;
        let json = serde_json::json!({ "version": 4, "fps": 24.0, "graph": g });
        std::fs::write(path, serde_json::to_string(&json).unwrap()).unwrap();
    }

    #[test]
    #[ignore]
    fn big_graph_bench() {
        let sizes: &[(usize, usize)] = match std::env::var("WOBBLE_BENCH_SIZE").ok().as_deref() {
            Some("big") => &[(100, 20), (100, 20), (100, 20), (100, 20), (100, 20), (100, 20)],
            _ => &[(10, 10), (25, 20), (50, 20), (100, 20)],
        };
        for &(cols, len) in sizes {
            let mut g = build(cols, len);
            let n = g.nodes.len();
            let zooms: Vec<f32> = match std::env::var("WOBBLE_BENCH_ZOOM") {
                Ok(z) => vec![z.parse().unwrap()],
                _ => vec![1.0, 0.25, 0.05],
            };
            let t = web_time::Instant::now();
            for _ in 0..10 {
                std::hint::black_box(crate::costviz::analyse(&g, &|_| false));
            }
            let flow_ms = t.elapsed().as_secs_f64() * 1e2;
            let t = web_time::Instant::now();
            for _ in 0..10 {
                std::hint::black_box(g.to_engine_graph_at(Some(0)));
            }
            let eg_ms = t.elapsed().as_secs_f64() * 1e2;
            println!("BENCH nodes {n:5} cost flow {flow_ms:7.2} ms  engine graph {eg_ms:7.2} ms");
            for zoom in zooms {
                let frames = std::env::var("WOBBLE_BENCH_FRAMES")
                    .ok()
                    .and_then(|f| f.parse().ok())
                    .unwrap_or(if n > 1500 { 20 } else { 40 });
                let (s, t) = run(&mut g, zoom, frames);
                println!("BENCH nodes {n:5} zoom {zoom:4}: show {s:7.2} ms  tess {t:7.2} ms");
            }
        }
    }
}

#[cfg(test)]
mod reduced_tests {
    use super::*;
    use engine::decode::SingleChannel as S;

    fn read(g: &mut EditorGraph, single: S, eight: bool) -> u64 {
        let rd = g.add(NodeKind::read(0, "plate.exr"), Pos2::ZERO);
        if let Some(NodeKind::Read { single: s, eight_bit, input, .. }) =
            g.node_mut(rd).map(|n| &mut n.kind)
        {
            *s = single;
            *eight_bit = eight;
            *input = engine::color::InputSpace::Working;
        }
        rd
    }

    fn after(g: &mut EditorGraph, src: u64, kind: NodeKind) -> u64 {
        let n = g.add(kind, Pos2::ZERO);
        g.connect(n, 0, src);
        n
    }

    /// Crop, Bake and a snapping retime keep one 8-bit channel; a Transform
    /// keeps the channel but not the 8 bits; a Grade keeps neither.
    #[test]
    fn the_format_follows_the_nodes_that_keep_it() {
        let mut g = EditorGraph::default();
        let rd = read(&mut g, S::Red, true);
        let crop = after(&mut g, rd, NodeKind::SoftCrop(Default::default()));
        let bake = after(
            &mut g,
            crop,
            NodeKind::GraphBake {
                source: 9,
                baked: false,
                start: None,
                mask: false,
                window: None,
                matte_half: false,
                single: false,
            },
        );
        let slip = after(&mut g, bake, NodeKind::Slip(Default::default()));
        let full = Some(Reduced { single: S::Red, eight: true, coded: true });
        for n in [crop, bake, slip] {
            assert_eq!(Reduced::of(&g, n), full, "node {n}");
        }
        let xf = after(&mut g, slip, NodeKind::Transform(Default::default()));
        assert_eq!(Reduced::of(&g, xf), Some(Reduced { single: S::Red, eight: false, coded: false }));
        let gr = after(&mut g, slip, NodeKind::Grade(Default::default()));
        assert_eq!(Reduced::of(&g, gr), None);
        // 8-bit alone does not survive a resample.
        let rd8 = read(&mut g, S::Off, true);
        let xf8 = after(&mut g, rd8, NodeKind::Transform(Default::default()));
        assert_eq!(Reduced::of(&g, xf8), None);
    }
}

#[cfg(test)]
mod harp_drag_tests {
    use super::*;

    /// A harp (3 strings) under a Read, a Mult spliced into string 1 —
    /// driven through the real editor, headless.
    struct Rig {
        g: EditorGraph,
        ed: NodeEditor,
        ctx: egui::Context,
        rects: Vec<(u64, Rect)>,
        n: usize,
        read: u64,
        bar: u64,
        rec: u64,
        mult: u64,
    }

    impl Rig {
        fn new() -> Self {
            let mut g = EditorGraph::default();
            g.nodes.clear();
            g.edges.clear();
            g.next_id = 0;
            let read = g.add(NodeKind::read(0, "plate.exr"), Pos2::new(0.0, -160.0));
            if let Some(NodeKind::Read { multi, .. }) = g.node_mut(read).map(|n| &mut n.kind) {
                *multi = true;
            }
            let (bar, rec) = crate::breakout::spawn_pair(
                &mut g,
                Pos2::new(0.0, 0.0),
                vec!["diff".into(), "spec".into(), "gi".into()],
            );
            g.connect(bar, 0, read);
            let mult = g.add(NodeKind::Mult(Default::default()), Pos2::new(0.0, 150.0));
            g.connect_out(mult, 0, bar, 1);
            g.connect(rec, 1, mult);
            crate::breakout::sync(&mut g);
            crate::breakout::repair(&mut g);
            let mut rig = Rig {
                g,
                ed: NodeEditor::default(),
                ctx: egui::Context::default(),
                rects: Vec::new(),
                n: 0,
                read,
                bar,
                rec,
                mult,
            };
            rig.frames(120, Vec::new());
            rig
        }

        fn frames(&mut self, count: usize, first: Vec<egui::Event>) {
            let mut events = first;
            for _ in 0..count {
                let input = egui::RawInput {
                    screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(900.0, 700.0))),
                    time: Some(self.n as f64 / 60.0),
                    events: std::mem::take(&mut events),
                    ..Default::default()
                };
                self.n += 1;
                let (mut sel, mut act, mut pin, mut slot) =
                    (Vec::new(), Vec::new(), Vec::new(), 2usize);
                let (ed, g, rects) = (&mut self.ed, &mut self.g, &mut self.rects);
                let _ = self.ctx.run_ui(input, |ui| {
                    ed.show(ui, g, &mut sel, &mut act, &mut pin, &mut slot, rects);
                });
            }
        }

        fn to_screen(&self, p: Pos2) -> Pos2 {
            let r = self.rects.iter().find(|(i, _)| *i == self.read).unwrap().1;
            let origin = r.center() - self.g.node(self.read).unwrap().pos.to_vec2() * self.ed.zoom;
            origin + p.to_vec2() * self.ed.zoom
        }

        fn pos(&self, id: u64) -> Pos2 {
            self.g.node(id).unwrap().pos
        }

        /// Press at `at`, travel by `by` in ten steps, release.
        fn drag(&mut self, at: Pos2, by: Vec2) {
            let none = egui::Modifiers::default();
            let button = |p: Pos2, down: bool| egui::Event::PointerButton {
                pos: p,
                button: egui::PointerButton::Primary,
                pressed: down,
                modifiers: none,
            };
            self.frames(2, vec![egui::Event::PointerMoved(at)]);
            self.frames(1, vec![button(at, true)]);
            for k in 1..=10 {
                self.frames(1, vec![egui::Event::PointerMoved(at + by * (k as f32 / 10.0))]);
            }
            self.frames(1, vec![button(at + by, false)]);
            self.frames(2, Vec::new());
        }
    }

    /// Up or down on the top bar — on its body between strings, or on a
    /// string's chip — moves the bar AND the node spliced into a string by
    /// the same amount; the receiver holds its height.
    #[test]
    fn a_vertical_bar_drag_carries_the_string_nodes() {
        for on_chip in [false, true] {
            let mut rig = Rig::new();
            let (b0, m0, r0) = (rig.pos(rig.bar), rig.pos(rig.mult), rig.pos(rig.rec));
            let kind = rig.g.node(rig.bar).unwrap().kind.clone();
            let size = rig.g.node(rig.bar).unwrap().size;
            let s0 = crate::breakout::string_x(&kind, b0, 0, size).unwrap();
            let s1 = crate::breakout::string_x(&kind, b0, 1, size).unwrap();
            let x = if on_chip { s1 } else { (s0 + s1) * 0.5 };
            let at = rig.to_screen(Pos2::new(x, b0.y));
            let dy = -60.0;
            rig.drag(at, Vec2::new(0.0, dy));
            let z = rig.ed.zoom;
            let (b1, m1, r1) = (rig.pos(rig.bar), rig.pos(rig.mult), rig.pos(rig.rec));
            let moved = b1.y - b0.y;
            assert!(
                (moved - dy / z).abs() < 12.0 / z,
                "on_chip {on_chip}: the bar moved {moved}, wanted {}",
                dy / z
            );
            assert!(
                ((m1.y - m0.y) - moved).abs() < 1e-3 && (m1.x - b1.x - (m0.x - b0.x)).abs() < 1e-3,
                "on_chip {on_chip}: the string node rides with the bar ({m0:?} -> {m1:?}, bar {b0:?} -> {b1:?})"
            );
            assert_eq!(r1.y, r0.y, "on_chip {on_chip}: the receiver holds its height");
            assert_eq!(r1.x, b1.x, "on_chip {on_chip}: the receiver stays x-locked");
        }
    }
}
