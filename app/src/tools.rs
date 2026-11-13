//! **Tool nodes** — a node whose behaviour is a GRAPH.
//!
//! A tool is a subgraph plus a set of authored parameters, saved to a file
//! and registered as a node type. Drop `plugins/<name>/tool.json` in and a
//! new node appears in the menus with its own knobs; the knobs drive inner
//! nodes through ordinary patch cables. It is the HDA / gizmo idea,
//! built on the pieces this app already has.
//!
//! **There is no new `NodeKind` arm.** A tool instance IS a
//! [`NodeKind::Custom`] — the same variant Tier 0 shader plugins use — whose
//! registry entry happens to carry a graph instead of a WGSL expression.
//! That is deliberate (manifesto 5: do not entrench the enum), and it is
//! also why tools arrive fully wired into everything: params are keyable and
//! patchable through `Channel::Custom`, the properties panel builds itself
//! from the `ParamSpec` list, and a setup opened without the tool installed
//! keeps every value and renders as a labelled passthrough.
//!
//! **Instances never reach the engine.** [`expand`] unfolds each one into
//! its definition's nodes before `to_engine_graph_at` does anything else, so
//! codegen, the fused pass, the pipeline cache and the render cache are
//! untouched by this feature and a tool costs exactly what the same nodes
//! cost by hand — no wrapper, no indirection, nothing at render time.
//!
//! Inside a body, the tool's own parameters appear as [`PSrc::Param`]
//! sources on the patch canvas: cable "knee" into a Grade's gain and the
//! instance's slider drives it. Expansion resolves those cables and bakes
//! the values onto the inner nodes it emits.
//!
//! The instance node itself SURVIVES expansion, orphaned: it holds the param
//! values, so a host cable reading a tool's knob (driving something else
//! from it) still resolves. Nothing consumes it, so it is never emitted.

use crate::editor::{EditorGraph, PDst, PSrc};
use engine::graph::{NodeKind, PortKind};
use engine::plugins::{ParamSpec, ShaderNodeDef};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, OnceLock, RwLock};

/// The INPUT nodes, one type per pipe. Inside a subwob an input node is a
/// PORTAL: it has no input socket of its own — what reaches it comes from
/// the parent graph — and each one becomes one input socket on every
/// instance, typed as its pipe. Ports are ordered by the node's `port`
/// param (a collapse numbers the ones it makes), then by id, so an input
/// added by hand lands after the numbered ones.
pub const INPUT_IMAGE: &str = "input.image";
pub const INPUT_MATTE: &str = "input.matte";
pub const INPUT_DATA: &str = "input.data";
pub const INPUT_CHANNELS: &str = "input.channels";
pub const INPUTS: [(&str, PortKind, &str); 4] = [
    (INPUT_IMAGE, PortKind::Image, "Image Input"),
    (INPUT_MATTE, PortKind::Mask, "Matte Input"),
    (INPUT_DATA, PortKind::Data, "Data Input"),
    // The FAT cable — every channel of a multi-layer plate on one wire.
    (INPUT_CHANNELS, PortKind::Channels, "Fat Pipe Input"),
];

/// Before input nodes were one type per pipe a body carried one type per
/// port INDEX (`tool.in1`..`tool.in4`) and one matte. Migrated on load
/// ([`migrate_legacy_inputs`]).
const LEGACY_IN: [&str; 4] = ["tool.in1", "tool.in2", "tool.in3", "tool.in4"];
const LEGACY_MATTE: &str = "tool.matte";

/// Most input ports a subwob may have.
pub const MAX_INPUTS: usize = 16;

/// The pipe an input node is the portal for, if `type_id` is one.
pub fn input_pipe(type_id: &str) -> Option<PortKind> {
    INPUTS.iter().find(|(id, _, _)| *id == type_id).map(|(_, k, _)| *k)
}

/// The input node type for a pipe. A field or geometry pipe has none: those
/// live only inside their own node's graph.
pub fn input_type_for(k: PortKind) -> Option<&'static str> {
    INPUTS.iter().find(|(_, p, _)| *p == k).map(|(id, _, _)| *id)
}

/// The first node in `body` that reads footage itself (a ChannelSelect off
/// the fat cable, a Cryptomatte's ranks, a harp's strings, a Read). Reader
/// bookkeeping — slots, the Read behind a cable, the frames — is keyed by
/// COMP node ids, so a reader that only exists once a subwob is unfolded
/// would bind whatever sits in slot 0: a plausible, wrong picture. Until
/// readers are carried through expansion, a body holding one does not
/// unfold (the node passes through) and its panel says why.
pub fn body_reads_footage(body: &EditorGraph) -> Option<String> {
    body.nodes
        .values()
        .find(|n| !n.kind.reader_slots().is_empty() || matches!(n.kind, NodeKind::Breakout(_)))
        .map(|n| n.kind.name().to_string())
}

/// Is this a body's input node, of any pipe?
pub fn is_proxy(kind: &NodeKind) -> bool {
    matches!(kind, NodeKind::Custom { type_id, .. } if input_pipe(type_id).is_some())
}

/// A registered type id that is an input node (hidden from the add menus
/// outside a subwob's body).
pub fn is_proxy_id(type_id: &str) -> bool {
    input_pipe(type_id).is_some()
}

/// Unfolding budget. Generous — it bounds instances-plus-nesting across the
/// whole comp, not nesting depth — and exists only so a tool that manages to
/// contain itself terminates instead of unfolding forever.
const MAX_PASSES: usize = 512;

/// The CONTROL node types. A tool's panel is not authored separately from
/// its graph: you add one of these to the body like any other node, wire its
/// output into the param it should drive, and it IS a knob on every
/// instance. Registered as node types (like the input proxies) so they
/// inherit the add menus, per-instance values and channel jacks with no new
/// `NodeKind` arm and no second schema to keep in step.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ControlKind {
    Slider,
    Rgb,
    Wheel,
    Label,
}

impl ControlKind {
    pub const ALL: [ControlKind; 4] =
        [ControlKind::Slider, ControlKind::Rgb, ControlKind::Wheel, ControlKind::Label];

    pub fn type_id(self) -> &'static str {
        match self {
            ControlKind::Slider => "ui.slider",
            ControlKind::Rgb => "ui.rgb",
            ControlKind::Wheel => "ui.wheel",
            ControlKind::Label => "ui.label",
        }
    }

    pub fn from_type(id: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|k| k.type_id() == id)
    }

    pub fn menu_label(self) -> &'static str {
        match self {
            ControlKind::Slider => "Slider",
            ControlKind::Rgb => "RGB Slider",
            ControlKind::Wheel => "Colour Wheel",
            ControlKind::Label => "Label",
        }
    }

    /// The params it EXPOSES, in order. A `Label` drives nothing; `lo`/`hi`
    /// are configuration of the control itself, not knobs on the tool.
    pub fn lanes(self) -> &'static [&'static str] {
        match self {
            ControlKind::Slider => &["value"],
            ControlKind::Rgb | ControlKind::Wheel => &["r", "g", "b"],
            ControlKind::Label => &[],
        }
    }

    /// Cells on the panel grid: [columns, rows]. One whole tile wide by
    /// default — half-width is something you choose, not something you get
    /// by accident.
    pub fn span(self) -> [u8; 2] {
        let w = COLS_PER_TILE as u8;
        match self {
            ControlKind::Slider | ControlKind::Label => [w, 1],
            // Row counts are DERIVED from what the instruments actually
            // measure, not hand-tuned: a cell one row short clips the widget
            // inside it, and nobody re-does this arithmetic after changing a
            // lane height.
            ControlKind::Rgb => [w, rows_for(crate::rgbslider::NATURAL_H)],
            // A wheel is a fixed instrument, not a stretchy row: one column
            // NARROWER than a full tile, which is the closest the grid comes
            // to its natural width.
            ControlKind::Wheel => [w - 1, rows_for(crate::wheels::COL_H)],
        }
    }

    /// A wheel is drawn at a fixed aspect, so scaling it sideways only adds
    /// empty space — it has no scale handles.
    pub fn scalable(self) -> bool {
        self != ControlKind::Wheel
    }
}

/// Rows needed to hold `h` pixels of widget.
fn rows_for(h: f32) -> u8 {
    (h / CELL_H).ceil().max(1.0) as u8
}

/// The panel grid. A panel is a whole number of properties TILES, and each
/// tile is divided into [`COLS_PER_TILE`] columns and [`ROWS_PER_TILE`] rows.
/// Controls occupy whole cells, so a panel can never tear mid-widget.
///
/// Columns are HALF a tile: a full-tile slider is the useful default, but a
/// pair of half-width controls side by side is a layout worth being able to
/// make, and 250px steps were too coarse to aim a scale handle at.
/// Columns per tile. A THIRD of a tile: fine enough to place a control
/// against without the steps being unaimable, and it divides the tile
/// exactly, so the emphasised lines stay on the tile boundaries they were
/// already on.
pub const COLS_PER_TILE: i32 = 3;
pub const CELL_W: f32 = crate::widgets::TILE / COLS_PER_TILE as f32;
pub const CELL_H: f32 = 34.0;

/// Rows that fit in ONE tile of height.
pub const ROWS_PER_TILE: i32 = (crate::widgets::TILE / CELL_H) as i32;

/// Widest a panel may be, in tiles. One place, so the manifest check and
/// the composer's field can never disagree about it.
pub const MAX_TILES: usize = 18;
/// ...and the tallest.
pub const MAX_TILES_H: usize = 3;

/// Columns on a panel that is `tiles` tiles wide.
pub fn cols(tiles: usize) -> i32 {
    COLS_PER_TILE * tiles.max(1) as i32
}
/// Rows on a panel that is `tiles_h` tiles tall.
pub fn rows(tiles_h: usize) -> i32 {
    ROWS_PER_TILE * tiles_h.max(1) as i32
}

/// Is this node a control, and which kind?
pub fn control_of(kind: &NodeKind) -> Option<ControlKind> {
    match kind {
        NodeKind::Custom { type_id, .. } => ControlKind::from_type(type_id),
        _ => None,
    }
}

/// The param name a control's lane exposes on the tool. Derived from the
/// NODE ID, never the label, so relabelling a control cannot orphan the
/// values every instance has already stored against it.
pub fn lane_param(node: u64, kind: ControlKind, lane: usize) -> String {
    if kind.lanes().len() == 1 {
        format!("w{node}")
    } else {
        format!("w{node}_{}", kind.lanes()[lane])
    }
}

/// What an instance's knob READS: the value the instance stored, else the
/// value authored on the control inside the tool (its default), else the
/// kind's own resting value. The panel and [`bind_controls`] (which drives
/// the render) fall back the same way, so a fresh instance shows what it
/// renders — an offset authored at 0 is 0 on the panel, not the widget's 1.
pub fn lane_value(
    vals: &BTreeMap<String, f32>,
    control: &NodeKind,
    kind: ControlKind,
    lane: usize,
    name: &str,
) -> f32 {
    if let Some(v) = vals.get(name) {
        return *v;
    }
    let authored = match control {
        NodeKind::Custom { params, .. } => {
            kind.lanes().get(lane).and_then(|l| params.get(*l)).copied()
        }
        _ => None,
    };
    authored.unwrap_or(if kind == ControlKind::Slider { 0.0 } else { 1.0 })
}

/// The knobs a body exposes: one per lane of every control node in it, in
/// panel order, defaulted to the value authored on the control itself.
pub fn params_of(body: &EditorGraph) -> Vec<ParamSpec> {
    let mut out = Vec::new();
    for (id, n) in ordered_controls(body) {
        let Some(k) = control_of(&n.kind) else { continue };
        let NodeKind::Custom { params, .. } = &n.kind else { continue };
        let (lo, hi) = (
            params.get("lo").copied().unwrap_or(0.0),
            params.get("hi").copied().unwrap_or(1.0),
        );
        for (i, lane) in k.lanes().iter().enumerate() {
            let base = if n.ui_label.is_empty() { k.menu_label().to_string() } else { n.ui_label.clone() };
            out.push(ParamSpec {
                name: lane_param(id, k, i),
                label: if k.lanes().len() == 1 { base } else { format!("{base}.{lane}") },
                default: params.get(*lane).copied().unwrap_or(0.0),
                min: lo,
                max: hi,
            });
        }
    }
    out
}

/// Control nodes in the order they appear on the panel: by grid cell (row,
/// then column), with unplaced ones last in id order so a body always has a
/// deterministic layout even before anyone arranges it.
pub fn ordered_controls(body: &EditorGraph) -> Vec<(u64, &crate::editor::ENode)> {
    let mut v: Vec<(u64, &crate::editor::ENode)> = body
        .nodes
        .iter()
        .filter(|(_, n)| control_of(&n.kind).is_some())
        .map(|(id, n)| (*id, n))
        .collect();
    v.sort_by_key(|(id, n)| {
        let c = n.ui_cell.unwrap_or([i32::MAX, i32::MAX]);
        (c[1], c[0], *id)
    });
    v
}

/// A tool definition: the interface, and the graph that implements it./// A tool definition: the interface, and the graph that implements it.
#[derive(Clone, PartialEq)]
pub struct ToolDef {
    pub type_id: String,
    pub label: String,
    pub color: [u8; 3],
    /// Input port count: one per input node in the body.
    pub inputs: usize,
    /// The pipe each port carries, in port order.
    pub ports: Vec<PortKind>,
    /// Properties-panel width in whole tiles.
    pub tiles: usize,
    /// ...and its height. The properties grid is square tiles and a panel
    /// occupies whole ones, so a tool that needs more room grows by a tile
    /// rather than by an awkward number of pixels.
    pub tiles_h: usize,
    pub params: Vec<ParamSpec>,
    /// The implementation. Its `view` names the node that is the tool's
    /// output.
    pub body: EditorGraph,
    /// Identity of the body, folded into the registered `source_hash` so
    /// editing a tool recompiles its consumers and invalidates their cached
    /// renders — exactly as editing a plugin's WGSL does.
    pub body_hash: u64,
    /// Where it was loaded from, so editing the body writes back to the same
    /// file instead of forking a second copy under `~/.wobble` that silently
    /// shadows the original.
    pub path: Option<std::path::PathBuf>,
    /// Defined INSIDE a comp (`EditorGraph::subwobs`, made by [`collapse`])
    /// rather than by a file: editing it writes back into the document,
    /// never to disk.
    pub local: bool,
}

/// A subwob DEFINED IN A COMP — what [`collapse`] makes. The same shape a
/// `tool.json` stores, kept in `EditorGraph::subwobs` under its type id, so
/// it saves with the setup, rides undo and redo like any other edit, and
/// nests (its body is a graph that may hold subwobs of its own). A comp
/// that uses one never depends on a file being installed: the nodes a
/// collapse took away live here, not in `~/.wobble`.
#[derive(Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LocalDef {
    pub label: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color: Option<[u8; 3]>,
    #[serde(default = "one_usize")]
    pub tiles: usize,
    #[serde(default = "one_usize")]
    pub tiles_h: usize,
    pub body: EditorGraph,
}

/// A [`LocalDef`] behind an `Arc`, the `graph::TrackRef` pattern: the comp is
/// cloned for every frame's engine graph and deep-compared for every undo
/// check, and a subwob's whole body would ride along each time. Sharing
/// makes the clone a pointer bump and the compare a pointer test until
/// something actually changes.
#[derive(Clone)]
pub struct SubwobRef(Arc<LocalDef>);

impl SubwobRef {
    pub fn new(d: LocalDef) -> Self {
        Self(Arc::new(d))
    }
}

impl std::ops::Deref for SubwobRef {
    type Target = LocalDef;
    fn deref(&self) -> &LocalDef {
        &self.0
    }
}

impl PartialEq for SubwobRef {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0) || *self.0 == *other.0
    }
}

impl serde::Serialize for SubwobRef {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        (*self.0).serialize(s)
    }
}

impl<'de> serde::Deserialize<'de> for SubwobRef {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        LocalDef::deserialize(d).map(SubwobRef::new)
    }
}

fn fnv64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

/// The node a body hands out: its designated `view`, else the one node
/// nothing else consumes. Ambiguity is refused rather than guessed — a body
/// with two loose ends has no single answer, and picking one silently would
/// make the tool render the wrong branch.
pub fn body_output(body: &EditorGraph) -> Option<u64> {
    if let Some(v) = body.view.filter(|v| body.nodes.contains_key(v)) {
        return Some(v);
    }
    let consumed: BTreeSet<u64> = body.edges.iter().map(|(_, _, s)| *s).collect();
    let mut loose = body
        .nodes
        .iter()
        .filter(|(id, n)| !consumed.contains(id) && !is_proxy(&n.kind))
        .map(|(id, _)| *id);
    let first = loose.next()?;
    loose.next().is_none().then_some(first)
}

/// The body's input nodes in PORT order, with the pipe each carries:
/// numbered ones (`port` param) first, then the rest in id order.
pub fn body_ports(body: &EditorGraph) -> Vec<(u64, PortKind)> {
    let mut v: Vec<((u8, f32, u64), PortKind)> = body
        .nodes
        .iter()
        .filter_map(|(id, n)| match &n.kind {
            NodeKind::Custom { type_id, params } => input_pipe(type_id).map(|k| {
                let key = match params.get("port") {
                    Some(p) => (0, *p, *id),
                    None => (1, 0.0, *id),
                };
                (key, k)
            }),
            _ => None,
        })
        .collect();
    v.sort_by(|a, b| a.0 .0.cmp(&b.0 .0).then(a.0 .1.total_cmp(&b.0 .1)).then(a.0 .2.cmp(&b.0 .2)));
    v.into_iter().map(|((_, _, id), k)| (id, k)).collect()
}

/// Which instance port every input node in `body` stands for.
fn body_proxies(body: &EditorGraph) -> BTreeMap<u64, usize> {
    body_ports(body).into_iter().enumerate().map(|(i, (id, _))| (id, i)).collect()
}

/// Write an instance's knob values back onto the control nodes that expose
/// them. Cables run FROM a control's channel, so the value an instance is set
/// to has to sit on that node before the body's patch is evaluated — this is
/// the step that makes an instance's panel actually drive its copy of the
/// graph.
pub fn bind_controls(body: &mut EditorGraph) {
    let vals = body.params.clone();
    let ids: Vec<(u64, ControlKind)> = body
        .nodes
        .iter()
        .filter_map(|(id, n)| control_of(&n.kind).map(|k| (*id, k)))
        .collect();
    for (id, k) in ids {
        let Some(n) = body.nodes.get_mut(&id) else { continue };
        let NodeKind::Custom { params, .. } = &mut n.kind else { continue };
        for (i, lane) in k.lanes().iter().enumerate() {
            if let Some(v) = vals.get(&lane_param(id, k, i)) {
                params.insert((*lane).to_string(), *v);
            }
        }
    }
}

/// Resolve every control's cell: the stored one where it still fits, packed
/// into the first free space otherwise. PURE, so the renderer can lay out a
/// panel it has no right to mutate — a control added and never arranged
/// still lands somewhere sensible instead of stacking at the origin.
pub fn layout_cells(
    body: &EditorGraph,
    cols: i32,
    rows: i32,
) -> Vec<(u64, [i32; 2], [u8; 2])> {
    let mut taken: Vec<([i32; 2], [u8; 2])> = Vec::new();
    let mut out = Vec::new();
    for (id, n) in ordered_controls(body) {
        let Some(k) = control_of(&n.kind) else { continue };
        let mut span = n.ui_span.unwrap_or_else(|| k.span());
        // A control is never wider or taller than the panel holding it.
        // `first_free_cell` would otherwise quietly widen the grid to fit an
        // oversized span, and the control would hang off the edge where the
        // renderer simply clips it away.
        span[0] = span[0].clamp(1, cols.max(1) as u8);
        span[1] = span[1].clamp(1, rows.max(1) as u8);
        // A stored cell is kept only while it still FITS: shrinking the
        // panel has to pull its controls back in rather than leave them
        // parked off the edge where they would simply vanish.
        let fits = |c: [i32; 2]| {
            c[0] >= 0
                && c[1] >= 0
                && c[0] + span[0] as i32 <= cols.max(1)
                && c[1] + span[1] as i32 <= rows.max(1)
                && !crate::widgets::any_overlap(c, span, &taken)
        };
        let cell = match n.ui_cell {
            Some(c) if fits(c) => c,
            _ => crate::widgets::first_free_cell(span, &taken, cols.max(1)),
        };
        taken.push((cell, span));
        out.push((id, cell, span));
    }
    out
}

/// Commit [`layout_cells`] onto the nodes, so an arrangement that was only
/// implied becomes the one that gets saved.
pub fn place_controls(body: &mut EditorGraph, cols: i32, rows: i32) {
    for (id, cell, span) in layout_cells(body, cols, rows) {
        if let Some(n) = body.nodes.get_mut(&id) {
            n.ui_cell = Some(cell);
            n.ui_span = Some(span);
        }
    }
}

// ---- registry ------------------------------------------------------------

fn registry() -> &'static RwLock<BTreeMap<String, Arc<ToolDef>>> {
    static REG: OnceLock<RwLock<BTreeMap<String, Arc<ToolDef>>>> = OnceLock::new();
    REG.get_or_init(Default::default)
}

pub fn get(type_id: &str) -> Option<Arc<ToolDef>> {
    registry().read().unwrap().get(type_id).cloned()
}

/// Every registered tool, sorted by label (for menus).
pub fn all() -> Vec<Arc<ToolDef>> {
    let mut v: Vec<_> = registry().read().unwrap().values().cloned().collect();
    v.sort_by(|a, b| a.label.cmp(&b.label));
    v
}

pub fn clear() {
    registry().write().unwrap().clear();
}

/// Register a tool, and its INTERFACE with the plugin registry — that is
/// what gives instances their port count, their param schema (so the
/// properties panel and the channel list build themselves) and a topology
/// identity, none of which need the body.
pub fn register(def: ToolDef) {
    // At least one port: a subwob with no input nodes still wants somewhere
    // for a bypass to pass through.
    let mut kinds = def.ports.clone();
    if kinds.is_empty() {
        kinds.push(PortKind::Image);
    }
    // The first matte port is THE mask port — drawn at the side and
    // defaulted (unwired = coverage 1) like a built-in's; every port is
    // typed as its input node's pipe.
    let mask = kinds.iter().position(|k| *k == PortKind::Mask);
    engine::plugins::register(
        ShaderNodeDef::new_tool(
            def.type_id.clone(),
            def.label.clone(),
            def.color,
            kinds.len(),
            def.tiles,
            def.params.clone(),
            def.body_hash,
        )
        .with_mask_input(mask)
        .with_input_kinds(kinds),
    );
    registry().write().unwrap().insert(def.type_id.clone(), Arc::new(def));
}

/// The input proxies. Registered as one-input passthroughs so that a body
/// being previewed on its own — and any proxy that escapes into a comp —
/// draws and renders as a plain pass-through rather than an error.
pub fn register_ports() {
    // Controls: no image input, so their expression is a constant rather
    // than `$in` — they exist to produce VALUES down patch cables, and one
    // left in a comp by mistake renders as nothing rather than as garbage.
    for k in ControlKind::ALL {
        let mut params: Vec<ParamSpec> = k
            .lanes()
            .iter()
            .map(|l| ParamSpec {
                name: (*l).to_string(),
                label: (*l).to_string(),
                default: if k == ControlKind::Slider { 0.0 } else { 1.0 },
                min: 0.0,
                max: 1.0,
            })
            .collect();
        if !k.lanes().is_empty() {
            // The RANGE the knob gets on an instance. Configuration of the
            // control, not a knob itself.
            params.push(ParamSpec {
                name: "lo".into(),
                label: "lo".into(),
                default: 0.0,
                min: 0.0,
                max: 0.0,
            });
            params.push(ParamSpec {
                name: "hi".into(),
                label: "hi".into(),
                default: 1.0,
                min: 0.0,
                max: 0.0,
            });
        }
        if k == ControlKind::Rgb {
            // Is the triple a MULTIPLIER (a gain, a gamma) rather than an
            // offset? The rgb slider's colour axes work in log space for
            // one and additively for the other, and only an offset may
            // cross zero. A setting of the control, like `lo`/`hi`; 0/1
            // because there is no toggle control yet.
            params.push(ParamSpec {
                name: "scale".into(),
                label: "multiplier".into(),
                default: 0.0,
                min: 0.0,
                max: 1.0,
            });
        }
        engine::plugins::register(ShaderNodeDef::new_control(
            k.type_id().to_string(),
            k.menu_label().to_string(),
            [0xd8, 0xa0, 0x30],
            params,
            fnv64(k.type_id().as_bytes()),
        ));
    }
    // Input nodes: no input socket (a portal has nothing to read inside
    // the body), and an output of their pipe. Rendered on their own —
    // which only a body viewed outside a subwob would do — they are
    // transparent black, like a control.
    for (id, pipe, label) in INPUTS {
        engine::plugins::register(
            ShaderNodeDef::new_control(
                id.to_string(),
                label.to_string(),
                [0x6a, 0x94, 0xd8],
                Vec::new(),
                fnv64(id.as_bytes()),
            )
            .with_output_kind(pipe),
        );
    }
}

// ---- expansion -----------------------------------------------------------

/// One instance, resolved and ready to splice into the host graph.
struct Expansion {
    /// body id -> host id, for every node that is actually emitted
    /// (proxies are not — they become the host's own wires).
    map: BTreeMap<u64, u64>,
    /// body id -> input index, for the proxies.
    proxies: BTreeMap<u64, usize>,
    /// Host id of the node carrying the tool's output.
    out_id: u64,
    body: EditorGraph,
    /// Inner channels the body's patch drives, with the values they take for
    /// this instance's params.
    driven: Vec<(u64, crate::anim::Channel, f32)>,
}

/// Unfold every tool instance in `g`, repeatedly, until the graph is flat.
///
/// Returns `None` when there was nothing to unfold — the common case, and
/// the reason the caller's recursion is one level deep. The graph it returns
/// is guaranteed to contain no expandable instance, so re-entering is safe.
pub fn expand(g: &EditorGraph, t: Option<u64>) -> Option<EditorGraph> {
    let mut cur = expand_once(g, t)?;
    for _ in 0..MAX_PASSES {
        match expand_once(&cur, t) {
            Some(next) => cur = next,
            // Flat: nothing left that names a registered tool.
            None => return Some(cur),
        }
    }
    // A tool that contains itself. Bypass what is left rather than recursing
    // forever or silently dropping it: a disabled node passes input 0
    // through, and it is visibly disabled in the graph.
    log::warn!("subwob expansion hit {MAX_PASSES} passes — recursive definition?");
    for n in cur.nodes.values_mut() {
        if matches!(&n.kind, NodeKind::Custom { type_id, .. } if get(type_id).is_some()) {
            n.disabled = true;
        }
    }
    Some(cur)
}

/// Unfold every instance present right now, in one pass. Instances that feed
/// each other are handled here rather than over several passes: sources are
/// looked up through `final_src`, which maps an expanded instance to the
/// node that now carries its output.
fn expand_once(g: &EditorGraph, t: Option<u64>) -> Option<EditorGraph> {
    let insts: Vec<(u64, Arc<ToolDef>)> = g
        .nodes
        .iter()
        // A bypassed instance keeps its own passthrough behaviour; unfolding
        // it would throw that away.
        .filter(|(_, n)| !n.disabled)
        .filter_map(|(id, n)| match &n.kind {
            NodeKind::Custom { type_id, .. } => get(type_id).map(|d| (*id, d)),
            _ => None,
        })
        // A body with no single output cannot be spliced in; it stays a
        // passthrough, which is what a broken definition should look like.
        .filter(|(_, d)| body_output(&d.body).is_some())
        // Nor can one that reads footage, yet (`body_reads_footage`).
        .filter(|(_, d)| body_reads_footage(&d.body).is_none())
        .collect();
    if insts.is_empty() {
        return None;
    }

    // Instance params as they READ at `t` — the node's own animation and any
    // host cable applied. A tool driven by a keyframe or an expression has
    // to expand with the value it actually shows, so this repeats the two
    // steps `to_engine_graph_at` performs, early. One base build serves
    // every instance.
    let base: BTreeMap<u64, NodeKind> = g
        .nodes
        .iter()
        .map(|(id, n)| {
            (*id, t.map_or_else(|| n.kind.clone(), |t| n.kind_at(t)))
        })
        .collect();
    let tf = t.unwrap_or(0) as f32;

    // Ids for the emitted nodes come off one counter above everything that
    // exists, walked in sorted order, so the same comp always flattens to
    // the same graph — the topology hash and the render-cache key depend on
    // that being true.
    let mut next = g
        .nodes
        .keys()
        .copied()
        .max()
        .map_or(0, |m| m + 1)
        .max(g.next_id);

    let mut exps: BTreeMap<u64, Expansion> = BTreeMap::new();
    for (id, def) in insts {
        let mut body = def.body.clone();
        // What a PSrc::Param cable in this body resolves to.
        body.params = def
            .params
            .iter()
            .map(|spec| {
                let ch = crate::anim::Channel::Custom(crate::anim::intern(&spec.name));
                // Reads the authored value, the keyed one, or whatever a
                // host cable drives — and falls back to the manifest default
                // when the node has never stored this param.
                (
                    spec.name.clone(),
                    g.patch_value(PSrc::Chan(id, ch), &base, tf, &mut Vec::new()),
                )
            })
            .collect();

        // Put the instance's knob values onto the controls that expose them,
        // so every cable running off a control carries what this instance is
        // set to.
        bind_controls(&mut body);

        let proxies = body_proxies(&body);

        // The body's own patch, resolved against its params exactly as the
        // host resolves its cables against its nodes.
        let bbase: BTreeMap<u64, NodeKind> = body
            .nodes
            .iter()
            .map(|(bid, n)| {
                (*bid, t.map_or_else(|| n.kind.clone(), |t| n.kind_at(t)))
            })
            .collect();
        let driven: Vec<(u64, crate::anim::Channel, f32)> = body
            .patch
            .links
            .iter()
            .filter_map(|l| match l.to {
                PDst::Chan(n, ch) => Some((
                    n,
                    ch,
                    body.patch_value(l.from, &bbase, tf, &mut Vec::new()),
                )),
                PDst::Mod(..) => None,
            })
            .collect();

        let mut map = BTreeMap::new();
        for bid in body.nodes.keys() {
            if proxies.contains_key(bid) {
                continue;
            }
            map.insert(*bid, next);
            next += 1;
        }
        let Some(out_id) = body_output(&body).and_then(|o| map.get(&o).copied()) else {
            continue;
        };
        exps.insert(id, Expansion { map, proxies, out_id, body, driven });
    }
    if exps.is_empty() {
        return None;
    }

    let mut out = g.clone();
    // The instance stays as an orphaned param carrier (a host cable reading
    // one of its knobs still has to resolve), but it is marked bypassed so
    // the next pass does not treat it as an instance and unfold it a second
    // time. Without this the fixpoint loop never converges: every pass finds
    // the carrier it left behind and emits another copy of the body.
    for id in exps.keys() {
        if let Some(n) = out.nodes.get_mut(id) {
            n.disabled = true;
        }
    }
    for e in exps.values() {
        for (bid, nid) in &e.map {
            let Some(src) = e.body.nodes.get(bid) else { continue };
            let mut n = src.clone();
            // A cable beats a keyframe, the same way it does in the host:
            // bake the driven value and drop the curve that would otherwise
            // overwrite it when the host evaluates animation.
            for (dn, ch, v) in &e.driven {
                if dn == bid {
                    crate::anim::set(&mut n.kind, *ch, *v);
                    n.anim.remove(ch);
                }
            }
            out.nodes.insert(*nid, n);
        }
    }

    // Rebuild the edge list in one go. Everything that referred to an
    // instance now refers to the node carrying its output; everything that
    // fed an instance's port now feeds whatever that port's proxy fed.
    let final_src = |s: u64| exps.get(&s).map_or(s, |e| e.out_id);
    let mut edges: Vec<(u64, usize, u64)> = Vec::new();
    for (d, i, s) in &g.edges {
        // Edges INTO an instance are consumed below, as its inputs.
        if exps.contains_key(d) {
            continue;
        }
        edges.push((*d, *i, final_src(*s)));
    }
    for (id, e) in &exps {
        let host_src: BTreeMap<usize, u64> = g
            .edges
            .iter()
            .filter(|(d, _, _)| d == id)
            .map(|(_, i, s)| (*i, final_src(*s)))
            .collect();
        for (d, i, s) in &e.body.edges {
            // A proxy has no inputs, so an edge into one is meaningless.
            let Some(nd) = e.map.get(d) else { continue };
            let ns = match e.proxies.get(s) {
                // An unwired instance port leaves the inner edge off, which
                // reads as unwired inside too.
                Some(k) => match host_src.get(k) {
                    Some(v) => *v,
                    None => continue,
                },
                None => match e.map.get(s) {
                    Some(v) => *v,
                    None => continue,
                },
            };
            edges.push((*nd, *i, ns));
        }
    }
    out.edges = edges;
    // Viewing a tool node means viewing what it produces.
    out.view = out.view.map(final_src);
    out.view_b = out.view_b.map(final_src);
    out.next_id = next;
    Some(out)
}

/// The graph to RENDER while a tool body is being edited: `host` with the
/// LIVE `body` spliced in at instance `inst`, viewing `body`'s node `want`.
///
/// This is what lets a tool be built against its real inputs instead of
/// against black. It deliberately does not go through the registry — the
/// body on screen has not been saved yet, and re-registering it every frame
/// to preview it would re-hash the whole body per frame.
///
/// Returns the spliced graph and `want`'s id inside it, so nested dives can
/// fold outwards one level at a time.
pub fn preview(
    host: &EditorGraph,
    inst: u64,
    body: &EditorGraph,
    want: u64,
    t: Option<u64>,
) -> Option<(EditorGraph, u64)> {
    if !host.nodes.contains_key(&inst) || body_reads_footage(body).is_some() {
        return None;
    }
    let proxies = body_proxies(body);
    let mut out = host.clone();
    let mut next = out
        .nodes
        .keys()
        .copied()
        .max()
        .map_or(0, |m| m + 1)
        .max(out.next_id);
    let mut map = BTreeMap::new();
    for bid in body.nodes.keys() {
        if proxies.contains_key(bid) {
            continue;
        }
        map.insert(*bid, next);
        next += 1;
    }

    // The body's cables resolve against the param VALUES it is carrying,
    // which the dive seeded from the schema — so a knob reads a real number
    // while it is being built.
    let tf = t.unwrap_or(0) as f32;
    let bbase: BTreeMap<u64, NodeKind> = body
        .nodes
        .iter()
        .map(|(bid, n)| (*bid, t.map_or_else(|| n.kind.clone(), |t| n.kind_at(t))))
        .collect();
    let driven: Vec<(u64, crate::anim::Channel, f32)> = body
        .patch
        .links
        .iter()
        .filter_map(|l| match l.to {
            PDst::Chan(n, ch) => {
                Some((n, ch, body.patch_value(l.from, &bbase, tf, &mut Vec::new())))
            }
            PDst::Mod(..) => None,
        })
        .collect();
    for (bid, nid) in &map {
        let mut n = body.nodes.get(bid)?.clone();
        for (dn, ch, v) in &driven {
            if dn == bid {
                crate::anim::set(&mut n.kind, *ch, *v);
                n.anim.remove(ch);
            }
        }
        out.nodes.insert(*nid, n);
    }
    // Bypass the instance: the live copy stands in for it, and leaving it
    // enabled would have `expand` unfold the SAVED body alongside this one.
    if let Some(n) = out.nodes.get_mut(&inst) {
        n.disabled = true;
    }

    let host_src: BTreeMap<usize, u64> = host
        .edges
        .iter()
        .filter(|(d, _, _)| *d == inst)
        .map(|(_, i, s)| (*i, *s))
        .collect();
    let mut edges: Vec<(u64, usize, u64)> = host
        .edges
        .iter()
        .filter(|(d, _, s)| *d != inst && *s != inst)
        .cloned()
        .collect();
    for (d, i, s) in &body.edges {
        let Some(nd) = map.get(d) else { continue };
        let ns = match proxies.get(s) {
            Some(k) => match host_src.get(k) {
                Some(v) => *v,
                None => continue,
            },
            None => match map.get(s) {
                Some(v) => *v,
                None => continue,
            },
        };
        edges.push((*nd, *i, ns));
    }
    // What read the instance now reads the live body's output, so a viewer
    // held on a node DOWNSTREAM of the subwob (the lock) sees the edit in
    // context.
    if let Some(o) = body_output(body).and_then(|o| map.get(&o).copied()) {
        for (d, i, s) in &host.edges {
            if *s == inst && *d != inst {
                edges.push((*d, *i, o));
                out.edge_out.remove(&(*d, *i));
            }
        }
    }
    out.edges = edges;
    let view = *map.get(&want)?;
    out.view = Some(view);
    out.view_b = None;
    out.view_active_b = false;
    out.next_id = next;
    Some((out, view))
}

// ---- authoring: a selection becomes a tool -------------------------------

/// What a selection looks like from inside: its body, the instance ports
/// the wires crossing into it become, and the node whose picture leaves it.
struct Captured {
    body: EditorGraph,
    /// The instance's inputs in PORT order: (source, which of its outputs,
    /// the pipe). Input node `k` carries `port = k`.
    ports: Vec<(u64, u32, PortKind)>,
    /// The selected node the rest of the comp reads — the body's output.
    output: Option<u64>,
}

/// The shared half of [`from_selection`] and [`collapse`].
fn capture(g: &EditorGraph, sel: &BTreeSet<u64>) -> Result<Captured, String> {
    if sel.is_empty() {
        return Err("nothing selected".into());
    }
    let mut body = EditorGraph { next_id: 0, ..Default::default() };
    for id in sel {
        body.nodes.insert(*id, g.nodes[id].clone());
        body.next_id = body.next_id.max(id + 1);
    }

    // Outside sources become INPUT nodes — one per distinct source output
    // and pipe, so a plate feeding two selected nodes stays ONE input, two
    // bands off one LumaKey stay two, and each takes the input node of the
    // pipe it is (a wire into a matte port makes a matte input).
    // Keyed by the pipe's discriminant: `PortKind` has no ordering.
    let mut proxy_of: BTreeMap<(u64, u32, u8), u64> = BTreeMap::new();
    let mut ports = Vec::new();
    for (d, i, s) in &g.edges {
        if !sel.contains(d) {
            continue;
        }
        let out = g.edge_out.get(&(*d, *i)).copied().unwrap_or(0);
        if sel.contains(s) {
            body.edges.push((*d, *i, *s));
            // Which output of the source: a band past the first must not
            // quietly become band 0 inside.
            if out != 0 {
                body.edge_out.insert((*d, *i), out);
            }
            if g.hidden_wires.contains(&(*d, *i)) {
                body.hidden_wires.insert((*d, *i));
            }
            continue;
        }
        // The pipe the wire is: what the consuming port takes.
        let pipe = g.nodes[d].kind.input_kind(*i);
        let key = (*s, out, pipe as u8);
        if !proxy_of.contains_key(&key) {
            let Some(type_id) = input_type_for(pipe) else {
                return Err(format!("a {pipe:?} wire cannot cross into a subwob"));
            };
            if ports.len() >= MAX_INPUTS {
                return Err(format!("more than {MAX_INPUTS} inputs cross the selection"));
            }
            let pid = body.next_id;
            body.next_id += 1;
            let mut n = g.nodes[d].clone();
            let mut params = BTreeMap::new();
            params.insert("port".to_string(), ports.len() as f32);
            n.kind = NodeKind::Custom { type_id: type_id.to_string(), params };
            n.anim.clear();
            n.disabled = false;
            n.pos.y -= 120.0;
            body.nodes.insert(pid, n);
            proxy_of.insert(key, pid);
            ports.push((*s, out, pipe));
        }
        body.edges.push((*d, *i, proxy_of[&key]));
    }

    // The output: what the selection hands to the rest of the comp, else the
    // one loose end inside it.
    let leaving: BTreeSet<(u64, u32)> = g
        .edges
        .iter()
        .filter(|(d, _, s)| sel.contains(s) && !sel.contains(d))
        .map(|(d, i, s)| (*s, g.edge_out.get(&(*d, *i)).copied().unwrap_or(0)))
        .collect();
    let from: BTreeSet<u64> = leaving.iter().map(|(s, _)| *s).collect();
    body.view = match from.len() {
        0 => None,
        1 => from.into_iter().next(),
        _ => return Err("the selection has more than one output — a subwob has one".into()),
    };
    // A subwob has ONE output, so a band past the first (a LumaKey's second
    // range, say) cannot be what leaves it.
    if leaving.iter().any(|(_, o)| *o != 0) {
        return Err("what leaves the selection is a node's second output — take its readers in too".into());
    }
    if body.view.is_none() {
        body.view = body_output(&body);
    }
    if body.view.is_none() {
        return Err("no single output node in the selection".into());
    }

    body.patch.links = g
        .patch
        .links
        .iter()
        .filter(|l| {
            let inside = |e: Option<u64>| e.is_none_or(|n| sel.contains(&n));
            inside(match l.from {
                PSrc::Chan(n, _) => Some(n),
                _ => None,
            }) && inside(match l.to {
                PDst::Chan(n, _) => Some(n),
                _ => None,
            })
        })
        .cloned()
        .collect();
    body.patch.mods = g.patch.mods.clone();
    body.patch.next = g.patch.next;
    let output = body.view;
    Ok(Captured { body, ports, output })
}

/// Build a definition from `sel` inside `g`. Edges reaching in from outside
/// the selection become input proxies, in the order they are found; the
/// selected node whose output leaves the selection becomes the tool's
/// output. Patch cables wholly inside the selection come along.
///
/// Parameters start empty: this captures the STRUCTURE, and knobs are
/// promoted onto it afterwards.
pub fn from_selection(
    g: &EditorGraph,
    sel: &[u64],
    type_id: &str,
    label: &str,
) -> Result<ToolDef, String> {
    let sel: BTreeSet<u64> = sel.iter().copied().filter(|id| g.nodes.contains_key(id)).collect();
    let c = capture(g, &sel)?;
    Ok(finish(type_id.to_string(), label.to_string(), [0x6a, 0xa8, 0x8a], 1, 1, c.body))
}

/// The prefix of a comp-local subwob's type id. A `.` never comes out of a
/// library tool's file-safe id, so a local one cannot shadow one.
pub const LOCAL_PREFIX: &str = "subwob.";

/// A free local type id: one past the highest `subwob.N` registered or
/// defined in `g` (at any depth).
pub fn next_local_id(g: &EditorGraph) -> String {
    fn walk(g: &EditorGraph, max: &mut u64) {
        for (id, d) in &g.subwobs {
            if let Some(n) = id.strip_prefix(LOCAL_PREFIX).and_then(|n| n.parse::<u64>().ok()) {
                *max = (*max).max(n);
            }
            walk(&d.body, max);
        }
    }
    let mut max = 0;
    for id in registry().read().unwrap().keys() {
        if let Some(n) = id.strip_prefix(LOCAL_PREFIX).and_then(|n| n.parse::<u64>().ok()) {
            max = max.max(n);
        }
    }
    walk(g, &mut max);
    format!("{LOCAL_PREFIX}{}", max + 1)
}

/// COLLAPSE `sel` into one subwob node, in place.
///
/// The selected nodes leave the comp and become the body of a new
/// definition kept IN the comp (`g.subwobs`, see [`LocalDef`]); one instance
/// takes their place. The wires that crossed into the selection now feed
/// the instance's ports, whatever read the selection's output reads the
/// instance, and a viewer pointed inside it points at the instance. A value
/// cable from outside into a collapsed node becomes a KNOB: a Slider control
/// inside, driving that channel, with the cable re-landed on the instance's
/// new knob — so the value still arrives, and the subwob has an interface.
///
/// Refused with a reason, and `g` untouched: a Read (it stays in the comp;
/// the subwob takes its picture as an input), a harp, an input placeholder,
/// a value cable LEAVING the selection (its source would vanish), and
/// anything [`capture`] refuses (two outputs, too many inputs).
///
/// The caller registers the new definition ([`register_locals`]).
pub fn collapse(g: &mut EditorGraph, sel: &[u64]) -> Result<u64, String> {
    let set: BTreeSet<u64> = sel.iter().copied().filter(|id| g.nodes.contains_key(id)).collect();
    if set.is_empty() {
        return Err("select the nodes to collapse into a subwob".into());
    }
    for id in &set {
        let k = &g.nodes[id].kind;
        let name = k.name().to_string();
        if k.reader_slot().is_some() {
            return Err(format!(
                "{name} reads footage: it stays in the comp, and the subwob takes its picture as an input"
            ));
        }
        if is_proxy(k) {
            return Err("an input placeholder belongs to the subwob it is in".into());
        }
        if matches!(k, NodeKind::Breakout(_) | NodeKind::Recombine(_)) {
            return Err("a harp stays in the comp".into());
        }
    }
    for l in &g.patch.links {
        if let PSrc::Chan(s, _) = l.from {
            let outside = match l.to {
                PDst::Chan(d, _) => !set.contains(&d),
                PDst::Mod(..) => true,
            };
            if set.contains(&s) && outside {
                return Err(format!(
                    "{} drives a value outside the selection — collapse what it drives too, or cut the cable",
                    g.nodes[&s].kind.name()
                ));
            }
        }
    }
    let mut c = capture(g, &set)?;

    let type_id = next_local_id(g);
    let num = type_id.trim_start_matches(LOCAL_PREFIX).to_string();
    let centre = set.iter().fold(egui::Vec2::ZERO, |a, id| a + g.nodes[id].pos.to_vec2())
        / set.len() as f32;
    let inst = g.add(
        NodeKind::Custom { type_id: type_id.clone(), params: Default::default() },
        centre.to_pos2(),
    );

    // Value cables coming IN (from a node outside, or from the knob of the
    // subwob this one is being made inside) become knobs.
    let value = crate::anim::Channel::Custom(crate::anim::intern("value"));
    let mut knobs: BTreeMap<String, f32> = BTreeMap::new();
    for k in 0..g.patch.links.len() {
        let l = g.patch.links[k].clone();
        let PDst::Chan(d, ch) = l.to else { continue };
        let from_outside = match l.from {
            PSrc::Chan(s, _) => !set.contains(&s),
            PSrc::Param(_) => true,
            PSrc::Mod(_) => false,
        };
        if !set.contains(&d) || !from_outside {
            continue;
        }
        let v = crate::anim::get(&g.nodes[&d].kind, ch).unwrap_or(0.0);
        let mut params = BTreeMap::new();
        params.insert("value".to_string(), v);
        // lo == hi: an unbounded field rather than a guessed range.
        params.insert("lo".to_string(), 0.0);
        params.insert("hi".to_string(), 0.0);
        let pos = c.body.nodes[&d].pos + egui::vec2(-240.0, 40.0 * knobs.len() as f32);
        let ctrl = c.body.add(
            NodeKind::Custom { type_id: ControlKind::Slider.type_id().into(), params },
            pos,
        );
        if let Some(n) = c.body.nodes.get_mut(&ctrl) {
            n.ui_label = ch.key().trim_start_matches("Custom:").to_string();
        }
        // `link` replaces whatever fed the channel — a Param cable copied in.
        c.body.patch.link(PSrc::Chan(ctrl, value), PDst::Chan(d, ch));
        let knob = lane_param(ctrl, ControlKind::Slider, 0);
        g.patch.links[k].to =
            PDst::Chan(inst, crate::anim::Channel::Custom(crate::anim::intern(&knob)));
        knobs.insert(knob, v);
    }
    if let Some(NodeKind::Custom { params, .. }) = g.nodes.get_mut(&inst).map(|n| &mut n.kind) {
        *params = knobs;
    }

    // The instance takes the selection's place in the wiring: input node
    // `k` is port `k`.
    let mut into = Vec::new();
    for (port, (s, out, _)) in c.ports.iter().enumerate() {
        into.push((inst, port, *s));
        if *out != 0 {
            g.edge_out.insert((inst, port), *out);
        }
    }
    if let Some(o) = c.output {
        for e in g.edges.iter_mut() {
            if e.2 == o && !set.contains(&e.0) {
                e.2 = inst;
            }
        }
    }
    let at = |v: &mut Option<u64>, out: &mut u32| {
        if v.is_some_and(|v| set.contains(&v)) {
            *v = Some(inst);
            *out = 0;
        }
    };
    at(&mut g.view, &mut g.view_out);
    at(&mut g.view_b, &mut g.view_b_out);
    at(&mut g.view_c, &mut g.view_c_out);
    at(&mut g.view_d, &mut g.view_d_out);

    // The collapsed nodes leave — WITHOUT `EditorGraph::remove`, whose
    // chain healing would wire around a hole the instance already fills.
    g.edges.retain(|(d, _, s)| !set.contains(d) && !set.contains(s));
    g.edges.extend(into);
    g.edge_out.retain(|(d, _), _| !set.contains(d));
    g.hidden_wires.retain(|(d, _)| !set.contains(d));
    g.global_sources.retain(|s| !set.contains(s));
    g.patch.links.retain(|l| {
        !matches!(l.to, PDst::Chan(d, _) if set.contains(&d))
            && !matches!(l.from, PSrc::Chan(s, _) if set.contains(&s))
    });
    for id in &set {
        g.nodes.remove(id);
    }
    g.prune_groups();
    crate::regions::prune(g);

    place_controls(&mut c.body, cols(1), rows(1));
    g.subwobs.insert(
        type_id,
        SubwobRef::new(LocalDef {
            label: format!("subwob {num}"),
            color: None,
            tiles: 1,
            tiles_h: 1,
            body: c.body,
        }),
    );
    Ok(inst)
}

/// A local definition as the registry holds it.
pub fn local_tool_def(type_id: &str, d: &LocalDef) -> ToolDef {
    let mut body = d.body.clone();
    place_controls(&mut body, cols(d.tiles), rows(d.tiles_h));
    let mut def = finish(
        type_id.to_string(),
        d.label.clone(),
        d.color.unwrap_or([0x6a, 0xa8, 0x8a]),
        d.tiles,
        d.tiles_h,
        body,
    );
    def.local = true;
    def
}

/// Register every subwob defined in `g`, at any depth. Cheap to call again:
/// a definition already registered with the same body and interface is
/// left alone, so nothing recompiles that did not change. Called when the
/// document's definitions may have moved under the registry — a load, an
/// undo or redo, a collapse, a dive back out, a plugin reload.
pub fn register_locals(g: &EditorGraph) {
    for (id, d) in &g.subwobs {
        let def = local_tool_def(id, d);
        let same = get(id).is_some_and(|t| {
            t.local
                && t.body_hash == def.body_hash
                && t.label == def.label
                && t.color == def.color
                && (t.tiles, t.tiles_h) == (def.tiles, def.tiles_h)
        });
        if !same {
            register(def);
        }
        register_locals(&d.body);
    }
}

/// Complete a definition: derive the arity from the body and hash it.
pub fn finish(
    type_id: String,
    label: String,
    color: [u8; 3],
    tiles: usize,
    tiles_h: usize,
    body: EditorGraph,
) -> ToolDef {
    // The schema is not authored beside the graph — it IS the graph's control
    // nodes. One source of truth, so a knob cannot exist without a control to
    // drive it, or a control sit on a panel driving nothing.
    let params = params_of(&body);
    // PRESENTATION MUST NOT REACH THE HASH. Controls live in the body, so
    // their cells and captions are serialized with it — and moving a widget
    // on a panel, or nudging a node in the graph, would otherwise change the
    // tool's topology identity and recompile every comp that uses it. Hash a
    // copy with all of that flattened out.
    let body_hash = {
        let mut plain = body.clone();
        // The project's node filter is presentation too: it rides in the
        // graph so a setup keeps it, and must not change what a tool IS.
        plain.node_status_min = Default::default();
        for n in plain.nodes.values_mut() {
            n.ui_label.clear();
            n.ui_cell = None;
            n.ui_span = None;
            n.pos = egui::Pos2::ZERO;
            n.size = 1.0;
        }
        serde_json::to_string(&plain)
            .map(|s| fnv64(s.as_bytes()))
            .unwrap_or(0)
    };
    ToolDef {
        type_id,
        label,
        color,
        inputs: body_ports(&body).len(),
        ports: body_ports(&body).into_iter().map(|(_, k)| k).collect(),
        tiles,
        tiles_h: tiles_h.max(1),
        params,
        body,
        body_hash,
        path: None,
        local: false,
    }
}

// ---- manifests -----------------------------------------------------------

#[derive(serde::Serialize, serde::Deserialize)]
struct ParamJson {
    name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    label: Option<String>,
    #[serde(default)]
    default: f32,
    #[serde(default)]
    min: f32,
    #[serde(default = "one")]
    max: f32,
}

fn one() -> f32 {
    1.0
}
fn one_usize() -> usize {
    1
}

#[derive(serde::Serialize, serde::Deserialize)]
struct ToolJson {
    type_id: String,
    label: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    color: Option<[u8; 3]>,
    #[serde(default = "one_usize")]
    tiles: usize,
    #[serde(default = "one_usize")]
    tiles_h: usize,
    /// LEGACY ONLY. The schema is derived from the body's control nodes now;
    /// this is read to migrate a tool authored before that and is never
    /// written back.
    #[serde(default, skip_serializing)]
    params: Vec<ParamJson>,
    body: EditorGraph,
}

/// Upgrade a body authored before controls were nodes: every parameter a
/// `PSrc::Param` cable reads becomes a Slider control, wired the same way. A
/// tool saved by the old promote gesture therefore opens with its knobs
/// intact and — for the first time — visible in the graph as the things that
/// produce them.
fn migrate_params_to_controls(body: &mut EditorGraph, old: &[ParamJson]) {
    use crate::anim::Channel;
    let mut names: Vec<String> = Vec::new();
    for l in &body.patch.links {
        if let PSrc::Param(Channel::Custom(i)) = l.from {
            let n = crate::anim::interned(i);
            if !n.is_empty() && !names.contains(&n) {
                names.push(n);
            }
        }
    }
    if names.is_empty() {
        return;
    }
    // Declaration order where the manifest still has it, so the migrated
    // panel comes out in the order the tool's author wrote.
    names.sort_by_key(|n| old.iter().position(|p| &p.name == n).unwrap_or(usize::MAX));

    let mut made: BTreeMap<String, u64> = BTreeMap::new();
    for name in &names {
        let spec = old.iter().find(|p| &p.name == name);
        let mut params = BTreeMap::new();
        params.insert("value".to_string(), spec.map_or(0.0, |p| p.default));
        params.insert("lo".to_string(), spec.map_or(0.0, |p| p.min));
        params.insert("hi".to_string(), spec.map_or(1.0, |p| p.max));
        let id = body.add(
            NodeKind::Custom {
                type_id: ControlKind::Slider.type_id().to_string(),
                params,
            },
            egui::pos2(-260.0, 60.0 * made.len() as f32),
        );
        if let Some(n) = body.nodes.get_mut(&id) {
            n.ui_label = name.clone();
        }
        made.insert(name.clone(), id);
    }
    let vch = Channel::Custom(crate::anim::intern("value"));
    for l in body.patch.links.iter_mut() {
        if let PSrc::Param(Channel::Custom(i)) = l.from {
            if let Some(id) = made.get(&crate::anim::interned(i)) {
                l.from = PSrc::Chan(*id, vch);
            }
        }
    }
}

/// Bodies saved before input nodes were one type per pipe: `tool.inN`
/// becomes an image input numbered N-1, `tool.matte` a matte input after
/// them — the order those ports always had.
fn migrate_legacy_inputs(body: &mut EditorGraph) {
    for n in body.nodes.values_mut() {
        let NodeKind::Custom { type_id, params } = &mut n.kind else { continue };
        let port = if let Some(i) = LEGACY_IN.iter().position(|id| id == type_id) {
            *type_id = INPUT_IMAGE.to_string();
            i
        } else if type_id == LEGACY_MATTE {
            *type_id = INPUT_MATTE.to_string();
            LEGACY_IN.len()
        } else {
            continue;
        };
        params.insert("port".to_string(), port as f32);
    }
}

/// Serialize a definition to the on-disk form.
pub fn to_manifest(def: &ToolDef) -> Result<String, String> {
    let j = ToolJson {
        type_id: def.type_id.clone(),
        label: def.label.clone(),
        color: Some(def.color),
        tiles: def.tiles,
        tiles_h: def.tiles_h,
        params: Vec::new(),
        body: def.body.clone(),
    };
    serde_json::to_string_pretty(&j).map_err(|e| e.to_string())
}

pub fn from_manifest_str(text: &str) -> Result<ToolDef, String> {
    let m: ToolJson = serde_json::from_str(text).map_err(|e| e.to_string())?;
    if m.type_id.is_empty() {
        return Err("empty type_id".into());
    }
    if is_proxy_id(&m.type_id) || LEGACY_IN.contains(&m.type_id.as_str()) || m.type_id == LEGACY_MATTE {
        return Err(format!("`{}` is a reserved input-node id", m.type_id));
    }
    if !(1..=MAX_TILES).contains(&m.tiles) {
        return Err(format!("tiles must be 1..={MAX_TILES}, got {}", m.tiles));
    }
    if !(1..=MAX_TILES_H).contains(&m.tiles_h) {
        return Err(format!("tiles_h must be 1..={MAX_TILES_H}, got {}", m.tiles_h));
    }
    let mut body = m.body;
    migrate_legacy_inputs(&mut body);
    if body_output(&body).is_none() {
        return Err("body needs one output node (set its `view`)".into());
    }
    // A tool authored before controls were nodes carries its schema in the
    // manifest and drives it through `PSrc::Param` cables. Turn each of those
    // into a real Slider control so it opens as a first-class panel.
    migrate_params_to_controls(&mut body, &m.params);
    place_controls(&mut body, cols(m.tiles), rows(m.tiles_h));
    let def = finish(
        m.type_id,
        m.label,
        m.color.unwrap_or([0x6a, 0xa8, 0x8a]),
        m.tiles,
        m.tiles_h,
        body,
    );
    Ok(def)
}

/// Load every `*/tool.json` under `dir`. Returns (loaded type_ids, errors) —
/// one broken tool never blocks the rest, exactly like plugins.
pub fn load_dir(dir: &std::path::Path) -> (Vec<String>, Vec<String>) {
    let mut loaded = Vec::new();
    let mut errors = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return (loaded, errors); // no directory is not an error
    };
    for e in entries.flatten() {
        let manifest = e.path().join("tool.json");
        if !manifest.is_file() {
            continue;
        }
        let parsed = std::fs::read_to_string(&manifest)
            .map_err(|err| err.to_string())
            .and_then(|t| from_manifest_str(&t));
        match parsed {
            Ok(mut def) => {
                loaded.push(def.type_id.clone());
                def.path = Some(manifest.clone());
                register(def);
            }
            Err(err) => errors.push(format!("{}: {err}", manifest.display())),
        }
    }
    loaded.sort();
    (loaded, errors)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anim::Channel;
    use egui::Pos2;

    fn xf() -> NodeKind {
        NodeKind::Transform(Default::default())
    }

    /// A body: an image input -> Transform, Transform is the output.
    fn passthrough_body() -> (EditorGraph, u64) {
        let mut b = EditorGraph::default();
        let inp = b.add(
            NodeKind::Custom { type_id: INPUT_IMAGE.into(), params: Default::default() },
            Pos2::ZERO,
        );
        let xform = b.add(xf(), Pos2::ZERO);
        b.edges.push((xform, 0, inp));
        b.view = Some(xform);
        (b, xform)
    }

    /// A host: Transform (stands in for a plate) -> the tool instance, viewed.
    fn host(type_id: &str) -> (EditorGraph, u64, u64) {
        let mut g = EditorGraph::default();
        let plate = g.add(xf(), Pos2::ZERO);
        let inst = g.add(
            NodeKind::Custom { type_id: type_id.into(), params: Default::default() },
            Pos2::ZERO,
        );
        g.edges.push((inst, 0, plate));
        g.view = Some(inst);
        (g, plate, inst)
    }

    #[test]
    fn an_instance_unfolds_into_its_body() {
        let (body, _) = passthrough_body();
        register(finish("t.pass".into(), "Pass".into(), [0; 3], 1, 1, body));
        let (g, plate, inst) = host("t.pass");

        let flat = expand(&g, None).expect("expands");
        // The instance survives as an orphaned param carrier, but nothing
        // reaches it any more.
        assert!(!flat.edges.iter().any(|(d, _, s)| *d == inst || *s == inst));
        // The body's Transform now sits between the plate and the viewer.
        let view = flat.view.expect("view retargeted");
        assert_ne!(view, inst, "view still points at the instance");
        assert!(matches!(flat.nodes[&view].kind, NodeKind::Transform(_)));
        assert_eq!(
            flat.edges.iter().find(|(d, _, _)| *d == view).map(|(_, _, s)| *s),
            Some(plate),
            "the inner node did not inherit the instance's input"
        );
        // Flat means flat: expanding again finds nothing.
        assert!(expand(&flat, None).is_none());
    }

    /// A body with an image input AND a matte input, both feeding a Grade.
    fn graded_body() -> (EditorGraph, u64, u64) {
        let mut b = EditorGraph::default();
        let inp = b.add(
            NodeKind::Custom { type_id: INPUT_IMAGE.into(), params: Default::default() },
            Pos2::ZERO,
        );
        let matte = b.add(
            NodeKind::Custom { type_id: INPUT_MATTE.into(), params: Default::default() },
            Pos2::ZERO,
        );
        let gr = b.add(NodeKind::Grade(Default::default()), Pos2::ZERO);
        b.edges.push((gr, 0, inp));
        b.edges.push((gr, 1, matte));
        b.view = Some(gr);
        (b, gr, matte)
    }

    /// A tool with a matte input proxy has a MATTE PORT on every instance:
    /// after its image ports, typed so a matte wire lands and an image one
    /// is refused, and inside the body the proxy's output IS a matte.
    #[test]
    fn a_matte_proxy_gives_the_tool_a_matte_port() {
        register_ports();
        let (body, gr, matte) = graded_body();
        assert_eq!(
            body.node(matte).unwrap().kind.output_kind(),
            engine::graph::PortKind::Mask,
            "inside the body the matte proxy must wire into a mask port"
        );
        assert_eq!(body.node(gr).unwrap().kind.input_kind(1), engine::graph::PortKind::Mask);
        register(finish("t.graded".into(), "Graded".into(), [0; 3], 1, 1, body));
        let d = engine::plugins::get("t.graded").unwrap();
        assert_eq!((d.inputs, d.mask_input), (2, Some(1)), "image port, then the matte");
        let inst = NodeKind::Custom { type_id: "t.graded".into(), params: Default::default() };
        assert_eq!(inst.mask_input(), Some(1));
        assert_eq!(inst.input_kind(0), engine::graph::PortKind::Image);
        assert_eq!(inst.input_kind(1), engine::graph::PortKind::Mask);
    }

    /// Expanded, the host's matte wire lands on the inner Grade's matte;
    /// left unwired, the inner matte is unwired too (coverage 1).
    #[test]
    fn a_matte_wired_to_an_instance_reaches_the_inner_node() {
        register_ports();
        let (body, _, _) = graded_body();
        register(finish("t.graded2".into(), "Graded".into(), [0; 3], 1, 1, body));
        let (mut g, plate, inst) = host("t.graded2");
        let key = g.add(NodeKind::LumaKey(Default::default()), Pos2::ZERO);
        g.edges.push((key, 0, plate));
        g.edges.push((inst, 1, key));
        let flat = expand(&g, None).expect("expands");
        let view = flat.view.expect("view retargeted");
        assert!(matches!(flat.nodes[&view].kind, NodeKind::Grade(_)));
        assert_eq!(flat.input_of(view, 0), Some(plate), "image in");
        assert_eq!(flat.input_of(view, 1), Some(key), "the matte reaches the inner Grade");

        let (g2, _, _) = host("t.graded2");
        let flat2 = expand(&g2, None).expect("expands");
        let view2 = flat2.view.unwrap();
        assert_eq!(flat2.input_of(view2, 1), None, "no matte wired: none inside");
    }

    /// Making a tool from a selection turns a wire landing on a MATTE port
    /// into the matte input, not a fifth image input.
    #[test]
    fn make_tool_turns_a_matte_crossing_into_the_matte_input() {
        register_ports();
        let mut g = EditorGraph::default();
        let plate = g.add(xf(), Pos2::ZERO);
        let key = g.add(NodeKind::LumaKey(Default::default()), Pos2::ZERO);
        g.edges.push((key, 0, plate));
        let gr = g.add(NodeKind::Grade(Default::default()), Pos2::ZERO);
        g.edges.push((gr, 0, plate));
        g.edges.push((gr, 1, key));
        let def = from_selection(&g, &[gr], "t.made", "Made").expect("captures");
        assert_eq!(def.ports, [PortKind::Image, PortKind::Mask], "the plate, then the matte");
        let mattes = def
            .body
            .nodes
            .values()
            .filter(|n| matches!(&n.kind, NodeKind::Custom { type_id, .. } if type_id == INPUT_MATTE))
            .count();
        assert_eq!(mattes, 1);
    }

    /// Build a Slider control wired to `chan` on `dst`, and return the body.
    fn body_with_slider(value: f32) -> (EditorGraph, u64) {
        let (mut body, xform) = passthrough_body();
        let mut cp = BTreeMap::new();
        cp.insert("value".to_string(), value);
        cp.insert("lo".to_string(), 0.0);
        cp.insert("hi".to_string(), 20.0);
        let sl = body.add(
            NodeKind::Custom { type_id: "ui.slider".into(), params: cp },
            Pos2::ZERO,
        );
        let vch = Channel::Custom(crate::anim::intern("value"));
        body.patch.link(PSrc::Chan(sl, vch), PDst::Chan(xform, Channel::TranslateX));
        (body, sl)
    }

    /// A control in a COMP (not a tool body) is the shot's panel: it is
    /// published as a knob, it drives other nodes through the patch bay, and
    /// it never reaches the engine as a node of its own.
    #[test]
    fn a_control_in_a_comp_is_a_panel_not_a_picture() {
        let (comp, slider) = body_with_slider(2.5);
        // Published: the panel a timeline shows.
        let specs = params_of(&comp);
        assert_eq!(specs.len(), 1, "one control, one knob");
        assert_eq!(specs[0].default, 2.5, "the authored value is the default");
        assert_eq!(specs[0].name, lane_param(slider, ControlKind::Slider, 0));

        // Not a picture: it is absent from the engine graph entirely.
        // Evaluated AT a frame: patch cables resolve per-tick, so the
        // authored-values form carries no cable value.
        let eg = comp.to_engine_graph_at(Some(0));
        assert!(
            !eg.nodes.contains_key(&slider),
            "a control must not reach the engine as a node"
        );
        // ...but the value it carries still drives the node it is cabled to.
        let view = eg.view.expect("the comp still has an output");
        assert_eq!(
            crate::anim::get(&eg.nodes[&view], Channel::TranslateX),
            Some(2.5),
            "the knob drives the graph through the patch bay"
        );
    }

    /// A CONTROL node is the knob: its value reaches the inner node through
    /// an ordinary cable, and an instance's own setting overrides it.
    #[test]
    fn a_control_node_drives_an_inner_node() {
        let (body, _) = body_with_slider(3.0);
        register(finish("t.ctrl".into(), "Ctrl".into(), [0; 3], 1, 1, body));
        let def = get("t.ctrl").expect("registered");
        assert_eq!(def.params.len(), 1, "one control, one knob");
        let knob = def.params[0].name.clone();

        // Untouched: the value authored on the control reaches through.
        let (mut g, _, inst) = host("t.ctrl");
        let flat = expand(&g, None).unwrap();
        let view = flat.view.unwrap();
        assert_eq!(crate::anim::get(&flat.nodes[&view].kind, Channel::TranslateX), Some(3.0));

        // Set on the instance: its value wins.
        if let NodeKind::Custom { params, .. } = &mut g.nodes.get_mut(&inst).unwrap().kind {
            params.insert(knob, 12.0);
        }
        let flat = expand(&g, None).unwrap();
        let view = flat.view.unwrap();
        assert_eq!(crate::anim::get(&flat.nodes[&view].kind, Channel::TranslateX), Some(12.0));
    }

    /// The whole chain in one expansion: a cable in the HOST drives the
    /// tool's knob, which drives the control, which drives the inner node.
    #[test]
    fn a_host_cable_reaches_a_control() {
        let (body, _) = body_with_slider(0.0);
        register(finish("t.hostctrl".into(), "HostCtrl".into(), [0; 3], 1, 1, body));
        let knob = get("t.hostctrl").unwrap().params[0].name.clone();
        let ch = Channel::Custom(crate::anim::intern(&knob));

        let (mut g, plate, inst) = host("t.hostctrl");
        crate::anim::set(&mut g.nodes.get_mut(&plate).unwrap().kind, Channel::Rotate, 7.0);
        g.patch.link(PSrc::Chan(plate, Channel::Rotate), PDst::Chan(inst, ch));

        let flat = expand(&g, None).unwrap();
        let view = flat.view.unwrap();
        assert_eq!(crate::anim::get(&flat.nodes[&view].kind, Channel::TranslateX), Some(7.0));
    }

    /// A Shader node inside a tool body: a control drives its WGSL param, and
    /// the shader's source travels with the definition. This is the whole
    /// point of building the node on `Channel::Shader` rather than a private
    /// value bag — a user shader is wrappable in a tool for free.
    #[test]
    fn a_control_drives_a_user_shader_param() {
        const SRC: &str = "//! input: front\n\
//! param: float gain = 1.0 [0, 8]\n\
fn pass1(uv: vec2f, frag: vec2f) -> vec4f { return tex(front, uv) * P.gain; }\n";
        let mut body = EditorGraph::default();
        let inp = body.add(
            NodeKind::Custom { type_id: INPUT_IMAGE.into(), params: Default::default() },
            Pos2::ZERO,
        );
        let mut sp = engine::graph::ShaderParams::default();
        sp.set_source(SRC.to_string());
        let sh = body.add(NodeKind::Shader(sp), Pos2::ZERO);
        body.edges.push((sh, 0, inp));
        body.view = Some(sh);

        let mut cp = BTreeMap::new();
        cp.insert("value".to_string(), 3.0);
        cp.insert("lo".to_string(), 0.0);
        cp.insert("hi".to_string(), 8.0);
        let sl = body.add(
            NodeKind::Custom { type_id: "ui.slider".into(), params: cp },
            Pos2::ZERO,
        );
        let gain = Channel::Shader(crate::anim::intern("gain"));
        body.patch
            .link(PSrc::Chan(sl, Channel::Custom(crate::anim::intern("value"))),
                  PDst::Chan(sh, gain));

        register(finish("t.shader".into(), "ShaderTool".into(), [0; 3], 1, 1, body));
        let (g, _, _) = host("t.shader");
        let flat = expand(&g, None).unwrap();
        let view = flat.view.unwrap();
        assert!(
            matches!(&flat.nodes[&view].kind, NodeKind::Shader(p) if p.source == SRC),
            "the shader source did not travel with the tool"
        );
        assert_eq!(crate::anim::get(&flat.nodes[&view].kind, gain), Some(3.0));
    }

    /// A knob's NAME is derived from the control's node id, so relabelling a
    /// control cannot orphan the values instances have stored against it.
    #[test]
    fn a_knob_name_survives_relabelling() {
        let (mut body, sl) = body_with_slider(1.0);
        body.nodes.get_mut(&sl).unwrap().ui_label = "Amount".into();
        let before = params_of(&body)[0].name.clone();
        body.nodes.get_mut(&sl).unwrap().ui_label = "Something Else Entirely".into();
        let after = params_of(&body);
        assert_eq!(after[0].name, before, "the name moved when only the caption did");
        assert_eq!(after[0].label, "Something Else Entirely");
    }

    /// Shrinking a panel has to pull its controls back inside it. A stored
    /// cell that no longer fits is repacked, not kept — a control parked off
    /// the edge would simply never be drawn.
    #[test]
    fn shrinking_a_panel_pulls_its_controls_back_in() {
        let (mut body, _) = body_with_slider(1.0);
        // A second control, parked out in the right-hand column.
        let mut cp = BTreeMap::new();
        cp.insert("value".to_string(), 0.0);
        let far = body.add(
            NodeKind::Custom { type_id: "ui.slider".into(), params: cp },
            Pos2::ZERO,
        );
        body.nodes.get_mut(&far).unwrap().ui_cell = Some([1, 0]);
        body.nodes.get_mut(&far).unwrap().ui_span = Some([1, 1]);

        // Two columns: it keeps the cell it was given.
        let wide = layout_cells(&body, 2, ROWS_PER_TILE);
        assert_eq!(wide.iter().find(|(id, _, _)| *id == far).unwrap().1, [1, 0]);

        // One column: it has to come back inside, and land on nothing else.
        let narrow = layout_cells(&body, 1, ROWS_PER_TILE);
        for (_, cell, span) in &narrow {
            assert!(
                cell[0] + span[0] as i32 <= 1 && cell[1] + span[1] as i32 <= ROWS_PER_TILE,
                "control left outside a 1x1 panel at {cell:?}"
            );
        }
        let cells: Vec<[i32; 2]> = narrow.iter().map(|(_, c, _)| *c).collect();
        let mut uniq = cells.clone();
        uniq.sort();
        uniq.dedup();
        assert_eq!(uniq.len(), cells.len(), "two controls landed on the same cell");
    }

    /// A taller panel is more rows, and the rows are the grid INSIDE a tile.
    #[test]
    fn panel_height_is_counted_in_whole_tiles() {
        assert_eq!(rows(1), ROWS_PER_TILE);
        assert_eq!(rows(2), ROWS_PER_TILE * 2);
        assert_eq!(rows(0), ROWS_PER_TILE, "a panel is never shorter than one tile");
    }

    /// Arranging a panel has to SURVIVE: the cells the composer writes onto
    /// the control nodes must reach the definition, the file, and back.
    #[test]
    fn an_arrangement_survives_the_round_trip() {
        let (mut body, sl) = body_with_slider(1.0);
        let mut cp = BTreeMap::new();
        cp.insert("value".to_string(), 0.0);
        let second = body.add(
            NodeKind::Custom { type_id: "ui.slider".into(), params: cp },
            Pos2::ZERO,
        );
        // What the composer does: put them somewhere deliberate.
        body.nodes.get_mut(&sl).unwrap().ui_cell = Some([0, 3]);
        body.nodes.get_mut(&sl).unwrap().ui_span = Some([1, 1]);
        body.nodes.get_mut(&second).unwrap().ui_cell = Some([0, 0]);
        body.nodes.get_mut(&second).unwrap().ui_span = Some([1, 1]);
        body.nodes.get_mut(&second).unwrap().ui_label = "Top".into();

        let def = finish("t.arr".into(), "Arr".into(), [0; 3], 1, 1, body);
        let cells = layout_cells(&def.body, 1, ROWS_PER_TILE);
        assert_eq!(cells.iter().find(|(i, _, _)| *i == sl).unwrap().1, [0, 3]);
        assert_eq!(cells.iter().find(|(i, _, _)| *i == second).unwrap().1, [0, 0]);
        // Panel ORDER follows the grid, not the node ids: the one at the top
        // of the panel comes first.
        assert_eq!(ordered_controls(&def.body)[0].0, second);

        // Through the file and back.
        let back = from_manifest_str(&to_manifest(&def).unwrap()).unwrap();
        let cells = layout_cells(&back.body, 1, ROWS_PER_TILE);
        assert_eq!(cells.iter().find(|(i, _, _)| *i == sl).unwrap().1, [0, 3]);
        assert_eq!(cells.iter().find(|(i, _, _)| *i == second).unwrap().1, [0, 0]);
        assert_eq!(back.tiles, 1);
        assert_eq!(back.tiles_h, 1);
    }

    /// A control can be widened past its kind's natural span, that override
    /// survives the file, and it can never end up wider than the panel.
    #[test]
    fn a_control_can_span_extra_columns() {
        let (mut body, sl) = body_with_slider(1.0);
        body.nodes.get_mut(&sl).unwrap().ui_cell = Some([0, 0]);
        body.nodes.get_mut(&sl).unwrap().ui_span = Some([2, 1]);
        let def = finish("t.wide".into(), "Wide".into(), [0; 3], 2, 1, body);

        let wide = layout_cells(&def.body, 2, ROWS_PER_TILE);
        assert_eq!(wide.iter().find(|(i, _, _)| *i == sl).unwrap().2, [2, 1]);

        let back = from_manifest_str(&to_manifest(&def).unwrap()).unwrap();
        let wide = layout_cells(&back.body, 2, ROWS_PER_TILE);
        assert_eq!(
            wide.iter().find(|(i, _, _)| *i == sl).unwrap().2,
            [2, 1],
            "the width override did not survive the round trip"
        );

        // Narrow the panel: the control narrows with it rather than hanging
        // off the edge, where it would just be clipped away.
        let narrow = layout_cells(&back.body, 1, ROWS_PER_TILE);
        let (_, cell, span) = *narrow.iter().find(|(i, _, _)| *i == sl).unwrap();
        assert_eq!(span[0], 1, "control stayed wider than its panel");
        assert!(cell[0] + span[0] as i32 <= 1);
    }

    /// A panel's grid comes from the TOOL, never from whatever rect it is
    /// drawn into. This pins both halves of that: the declared grid preserves
    /// an arrangement, and a smaller one demonstrably does not — which is
    /// exactly how an instance whose properties area was shorter than the
    /// declared panel came to re-flow its controls.
    #[test]
    fn the_declared_grid_preserves_an_arrangement() {
        let (mut body, sl) = body_with_slider(1.0);
        let mut cp = BTreeMap::new();
        cp.insert("value".to_string(), 0.0);
        let low = body.add(
            NodeKind::Custom { type_id: "ui.slider".into(), params: cp },
            Pos2::ZERO,
        );
        body.nodes.get_mut(&sl).unwrap().ui_cell = Some([0, 0]);
        body.nodes.get_mut(&sl).unwrap().ui_span = Some([2, 1]);
        // Deliberately down in the SECOND tile of height.
        body.nodes.get_mut(&low).unwrap().ui_cell = Some([0, ROWS_PER_TILE + 2]);
        body.nodes.get_mut(&low).unwrap().ui_span = Some([2, 1]);
        let def = finish("t.grid".into(), "Grid".into(), [0; 3], 2, 2, body);

        let declared = layout_cells(&def.body, cols(def.tiles), rows(def.tiles_h));
        assert_eq!(
            declared.iter().find(|(i, _, _)| *i == low).unwrap().1,
            [0, ROWS_PER_TILE + 2],
            "the declared grid moved a control that fitted it"
        );

        // One tile shorter — the control no longer fits and is re-flowed.
        let squeezed = layout_cells(&def.body, cols(def.tiles), ROWS_PER_TILE);
        assert_ne!(
            squeezed.iter().find(|(i, _, _)| *i == low).unwrap().1,
            [0, ROWS_PER_TILE + 2],
            "a short grid was expected to repack — if it no longer does, the \
             renderer may derive its grid from the rect again without anyone noticing"
        );
    }

    /// The wheel's cell has to be big enough to draw a wheel in, and its
    /// span and the grid constants are set independently — so this is the
    /// arithmetic nobody will redo by hand after changing one of them.
    #[test]
    fn a_colour_wheel_fits_the_cell_it_asks_for() {
        let sp = ControlKind::Wheel.span();
        let (w, h) = (sp[0] as f32 * CELL_W, sp[1] as f32 * CELL_H);
        let fit = (w / crate::wheels::COL_W).min(h / crate::wheels::COL_H);
        assert!(fit >= 0.75, "a wheel would be drawn at {fit:.2}x — too small to drive");
        assert!(fit <= 1.5, "a wheel would be drawn at {fit:.2}x — rattling in its cell");
        // ...and it does not ask to be resized, because its aspect is fixed.
        assert!(!ControlKind::Wheel.scalable());
        assert!(ControlKind::Slider.scalable());
    }

    /// An rgb row needs its three lanes plus a tag; a slider needs one row.
    #[test]
    fn the_row_grid_fits_the_controls_built_on_it() {
        let rgb = ControlKind::Rgb.span();
        assert!(
            rgb[1] as f32 * CELL_H >= crate::rgbslider::NATURAL_H,
            "an rgb cell ({}px) is shorter than the widget in it ({}px)",
            rgb[1] as f32 * CELL_H,
            crate::rgbslider::NATURAL_H
        );
        assert!(
            rgb[0] as f32 * CELL_W >= crate::rgbslider::CHROME_W + crate::rgbslider::MIN_TRACK,
            "an rgb cell is too narrow for the widget's chrome plus a usable track"
        );
        assert!(CELL_H >= 20.0, "a one-row slider has no room for its value box");
        // The COLUMN grid divides the tile exactly, so the emphasised
        // vertical lines sit on tile boundaries.
        assert!(
            (COLS_PER_TILE as f32 * CELL_W - crate::widgets::TILE).abs() < 0.01,
            "column grid drifts off the tile"
        );
    }

    #[test]
    fn an_unregistered_type_is_left_alone() {
        let (g, _, _) = host("t.does.not.exist");
        assert!(expand(&g, None).is_none(), "a missing tool must stay a passthrough");
    }

    #[test]
    fn a_bypassed_instance_keeps_its_own_passthrough() {
        let (body, _) = passthrough_body();
        register(finish("t.bypass".into(), "Bypass".into(), [0; 3], 1, 1, body));
        let (mut g, _, inst) = host("t.bypass");
        g.nodes.get_mut(&inst).unwrap().disabled = true;
        assert!(expand(&g, None).is_none());
    }

    #[test]
    fn an_unwired_port_leaves_the_inner_edge_off() {
        let (body, _) = passthrough_body();
        register(finish("t.unwired".into(), "Unwired".into(), [0; 3], 1, 1, body));
        let mut g = EditorGraph::default();
        let inst = g.add(
            NodeKind::Custom { type_id: "t.unwired".into(), params: Default::default() },
            Pos2::ZERO,
        );
        g.view = Some(inst);
        let flat = expand(&g, None).unwrap();
        let view = flat.view.unwrap();
        assert!(
            !flat.edges.iter().any(|(d, _, _)| *d == view),
            "an unwired port must not invent an edge"
        );
    }

    #[test]
    fn tools_nest() {
        let (inner, _) = passthrough_body();
        register(finish("t.inner".into(), "Inner".into(), [0; 3], 1, 1, inner));
        // An outer body that is itself built from the inner tool.
        let mut outer = EditorGraph::default();
        let inp = outer.add(
            NodeKind::Custom { type_id: INPUT_IMAGE.into(), params: Default::default() },
            Pos2::ZERO,
        );
        let mid = outer.add(
            NodeKind::Custom { type_id: "t.inner".into(), params: Default::default() },
            Pos2::ZERO,
        );
        outer.edges.push((mid, 0, inp));
        outer.view = Some(mid);
        register(finish("t.outer".into(), "Outer".into(), [0; 3], 1, 1, outer));

        let (g, plate, _) = host("t.outer");
        let flat = expand(&g, None).unwrap();
        let view = flat.view.unwrap();
        // Both levels are gone; what is left is the innermost Transform, fed
        // by the plate.
        assert!(matches!(flat.nodes[&view].kind, NodeKind::Transform(_)));
        assert_eq!(
            flat.edges.iter().find(|(d, _, _)| *d == view).map(|(_, _, s)| *s),
            Some(plate)
        );
        assert!(expand(&flat, None).is_none());
    }

    #[test]
    fn a_recursive_tool_terminates() {
        // A body containing an instance of itself. It cannot be built in one
        // step, so register a placeholder and then swap its body.
        let mut body = EditorGraph::default();
        let inp = body.add(
            NodeKind::Custom { type_id: INPUT_IMAGE.into(), params: Default::default() },
            Pos2::ZERO,
        );
        let me = body.add(
            NodeKind::Custom { type_id: "t.loop".into(), params: Default::default() },
            Pos2::ZERO,
        );
        body.edges.push((me, 0, inp));
        body.view = Some(me);
        register(finish("t.loop".into(), "Loop".into(), [0; 3], 1, 1, body));

        let (g, _, _) = host("t.loop");
        let flat = expand(&g, None).expect("expands");
        // Terminated, and what could not be unfolded is visibly bypassed
        // rather than silently dropped.
        assert!(flat
            .nodes
            .values()
            .filter(|n| matches!(&n.kind, NodeKind::Custom { type_id, .. } if type_id == "t.loop"))
            .all(|n| n.disabled));
        assert!(expand(&flat, None).is_none(), "must reach a fixpoint");
    }

    /// The hook in `to_engine_graph_at` is where this feature meets the rest
    /// of the app: the engine must receive a flat graph, and the recursion
    /// must bottom out.
    #[test]
    fn the_engine_graph_never_sees_an_instance() {
        let (body, _) = passthrough_body();
        register(finish("t.engine".into(), "Engine".into(), [0; 3], 1, 1, body));
        let (g, _, inst) = host("t.engine");

        let eg = g.to_engine_graph_at(None);
        let view = eg.view.expect("view survives");
        assert_ne!(view, inst);
        assert!(matches!(eg.nodes.get(&view), Some(NodeKind::Transform(_))));
        // Nothing reachable is a tool instance.
        assert!(!eg
            .edges
            .iter()
            .any(|((d, _), s)| *d == inst || *s == inst));
    }

    /// A body that BRANCHES off its input: the plate goes into a Blur AND
    /// straight to a Blend, which takes the blur on B (the shape a glow
    /// has). Expanded, both branches read the host's plate.
    #[test]
    fn a_branching_body_unfolds_with_the_plate_on_both_branches() {
        register_ports();
        let mut b = EditorGraph::default();
        let inp = b.add(
            NodeKind::Custom { type_id: INPUT_IMAGE.into(), params: Default::default() },
            Pos2::ZERO,
        );
        let blur = b.add(NodeKind::Blur(Default::default()), Pos2::ZERO);
        let blend = b.add(
            NodeKind::Blend { op: engine::graph::BlendOp::Over, mix: 1.0, gain: [1.0, 1.0] },
            Pos2::ZERO,
        );
        b.edges.push((blur, 0, inp));
        b.edges.push((blend, 0, inp));
        b.edges.push((blend, 1, blur));
        b.view = Some(blend);
        register(finish("t.branch".into(), "Branch".into(), [0; 3], 1, 1, b));
        let (g, plate, _) = host("t.branch");
        let flat = expand(&g, None).expect("expands");
        let src = |d: u64, i: usize| {
            flat.edges.iter().find(|(dd, ii, _)| *dd == d && *ii == i).map(|(_, _, s)| *s)
        };
        let view = flat.view.expect("view");
        let blur_id = flat
            .nodes
            .iter()
            .find(|(_, n)| matches!(n.kind, NodeKind::Blur(_)))
            .map(|(id, _)| *id)
            .expect("a blur");
        assert_eq!(src(blur_id, 0), Some(plate));
        assert_eq!(src(view, 0), Some(plate), "blend A is the untouched plate");
        assert_eq!(src(view, 1), Some(blur_id), "blend B is the blur");
    }

    /// A fresh instance's panel shows what it renders: an untouched lane
    /// reads the value authored on the control (offset and lift at 0, gamma
    /// and gain at 1 here), a stored value wins, and only a control with no
    /// authored value falls back to the widget's own.
    #[test]
    fn an_untouched_lane_reads_its_authored_value() {
        let mut body = EditorGraph::default();
        for (label, v) in [("offset", 0.0), ("lift", 0.0), ("gamma", 1.0), ("gain", 1.0)] {
            let mut params = BTreeMap::new();
            for l in ["r", "g", "b"] {
                params.insert(l.to_string(), v);
            }
            let id = body.add(NodeKind::Custom { type_id: "ui.rgb".into(), params }, Pos2::ZERO);
            body.nodes.get_mut(&id).unwrap().ui_label = label.into();
        }
        let fresh: BTreeMap<String, f32> = BTreeMap::new();
        let mut seen = 0;
        for (id, n) in ordered_controls(&body) {
            let k = control_of(&n.kind).unwrap();
            let want = match n.ui_label.as_str() {
                "offset" | "lift" => 0.0,
                _ => 1.0,
            };
            for lane in 0..3 {
                let name = lane_param(id, k, lane);
                assert_eq!(lane_value(&fresh, &n.kind, k, lane, &name), want, "{} lane {lane}", n.ui_label);
                seen += 1;
            }
        }
        assert_eq!(seen, 12);
        let (id, n) = ordered_controls(&body)[0];
        let k = control_of(&n.kind).unwrap();
        let name = lane_param(id, k, 1);
        let stored: BTreeMap<String, f32> = [(name.clone(), 0.25)].into_iter().collect();
        assert_eq!(lane_value(&stored, &n.kind, k, 1, &name), 0.25);
        let bare = NodeKind::Custom { type_id: "ui.rgb".into(), params: Default::default() };
        assert_eq!(lane_value(&fresh, &bare, ControlKind::Rgb, 0, "w9_r"), 1.0);
        let bare = NodeKind::Custom { type_id: "ui.slider".into(), params: Default::default() };
        assert_eq!(lane_value(&fresh, &bare, ControlKind::Slider, 0, "w9"), 0.0);
    }

    /// An input node is a PORTAL: no input socket of its own, and an output
    /// of the pipe it stands for — one type per pipe.
    #[test]
    fn input_nodes_are_portals_one_type_per_pipe() {
        register_ports();
        for (id, pipe, _) in INPUTS {
            let k = NodeKind::Custom { type_id: id.into(), params: Default::default() };
            assert_eq!(k.input_count(), 0, "{id} has an input socket");
            assert_eq!(k.output_kind(), pipe, "{id}");
        }
    }

    /// Every input node becomes a port on the instance, typed as its pipe,
    /// in `port` order — so a subwob can take a data wire or the fat cable,
    /// not only pictures and one matte.
    #[test]
    fn each_input_node_is_a_port_of_its_pipe() {
        register_ports();
        let mut b = EditorGraph::default();
        let port = |p: f32| [("port".to_string(), p)].into_iter().collect::<BTreeMap<_, _>>();
        let data = b.add(NodeKind::Custom { type_id: INPUT_DATA.into(), params: port(1.0) }, Pos2::ZERO);
        let img = b.add(NodeKind::Custom { type_id: INPUT_IMAGE.into(), params: port(0.0) }, Pos2::ZERO);
        let chans = b.add(NodeKind::Custom { type_id: INPUT_CHANNELS.into(), params: port(2.0) }, Pos2::ZERO);
        let xform = b.add(xf(), Pos2::ZERO);
        b.edges.push((xform, 0, img));
        b.view = Some(xform);
        let _ = (data, chans);
        register(finish("t.pipes".into(), "Pipes".into(), [0; 3], 1, 1, b));
        let inst = NodeKind::Custom { type_id: "t.pipes".into(), params: Default::default() };
        assert_eq!(inst.input_count(), 3);
        assert_eq!(
            [inst.input_kind(0), inst.input_kind(1), inst.input_kind(2)],
            [PortKind::Image, PortKind::Data, PortKind::Channels]
        );
    }

    /// A body saved before input nodes were one type per pipe keeps its
    /// ports: `tool.in1` is the image input on port 0, `tool.matte` the
    /// matte after it.
    #[test]
    fn a_legacy_body_migrates_its_inputs() {
        register_ports();
        let (mut b, gr, _) = graded_body();
        for n in b.nodes.values_mut() {
            if let NodeKind::Custom { type_id, .. } = &mut n.kind {
                *type_id = if type_id == INPUT_IMAGE { "tool.in1".into() } else { "tool.matte".into() };
            }
        }
        let def = finish("t.legacy".into(), "Legacy".into(), [0; 3], 1, 1, b);
        let text = to_manifest(&def).unwrap();
        assert!(text.contains("tool.in1") && text.contains("tool.matte"));
        let back = from_manifest_str(&text).expect("a legacy body parses");
        assert_eq!(back.ports, [PortKind::Image, PortKind::Mask]);
        assert!(back.body.nodes.values().all(|n| !matches!(&n.kind,
            NodeKind::Custom { type_id, .. } if type_id.starts_with("tool."))));
        assert_eq!(body_output(&back.body), Some(gr));
    }

    /// plate -> a -> b -> tail, viewing b. COLLAPSE [a, b]: they leave, one
    /// subwob node takes their place — the plate feeds it, tail reads it,
    /// the viewer shows it — and its definition lives IN the graph.
    /// Expanded, the comp is exactly what it was.
    #[test]
    fn a_collapse_leaves_one_node_wired_in_the_selections_place() {
        register_ports();
        let mut g = EditorGraph::default();
        let plate = g.add(xf(), Pos2::ZERO);
        let a = g.add(xf(), Pos2::new(0.0, 100.0));
        let b = g.add(xf(), Pos2::new(0.0, 200.0));
        let tail = g.add(xf(), Pos2::new(0.0, 300.0));
        g.edges.push((a, 0, plate));
        g.edges.push((b, 0, a));
        g.edges.push((tail, 0, b));
        g.view = Some(b);
        crate::anim::set(&mut g.nodes.get_mut(&b).unwrap().kind, Channel::Rotate, 5.0);

        let inst = collapse(&mut g, &[a, b]).expect("collapses");
        assert!(!g.nodes.contains_key(&a) && !g.nodes.contains_key(&b), "the selection left");
        assert_eq!(g.nodes.len(), 3, "plate, tail and the one subwob node");
        assert_eq!(g.input_of(inst, 0), Some(plate));
        assert_eq!(g.input_of(tail, 0), Some(inst));
        assert_eq!(g.view, Some(inst));
        let NodeKind::Custom { type_id, .. } = &g.nodes[&inst].kind else { panic!("not a subwob") };
        assert!(type_id.starts_with(LOCAL_PREFIX), "{type_id}");
        let def = g.subwobs.get(type_id).expect("defined in the graph");
        assert_eq!(body_ports(&def.body).len(), 1, "one input node");

        register_locals(&g);
        let flat = expand(&g, None).expect("expands");
        let view = flat.view.unwrap();
        assert_eq!(crate::anim::get(&flat.nodes[&view].kind, Channel::Rotate), Some(5.0));
        let mid = flat.input_of(view, 0).unwrap();
        assert_eq!(flat.input_of(mid, 0), Some(plate));
        assert_eq!(flat.input_of(tail, 0), Some(view), "tail reads what the subwob makes");
    }

    /// The live preview of a body keeps the comp DOWNSTREAM of its instance
    /// connected: the tail reads the live body's output, so a viewer held
    /// on the tail (the lock) shows an inner edit in context.
    #[test]
    fn a_preview_rewires_the_instances_consumers_to_the_live_body() {
        register_ports();
        let mut g = EditorGraph::default();
        let plate = g.add(xf(), Pos2::ZERO);
        let a = g.add(xf(), Pos2::new(0.0, 100.0));
        let tail = g.add(xf(), Pos2::new(0.0, 200.0));
        g.edges.push((a, 0, plate));
        g.edges.push((tail, 0, a));
        let inst = collapse(&mut g, &[a]).expect("collapses");
        let NodeKind::Custom { type_id, .. } = &g.nodes[&inst].kind else { panic!() };
        let mut body = g.subwobs[type_id].body.clone();
        // The edit being made inside.
        let inner = body_output(&body).expect("one output");
        crate::anim::set(&mut body.nodes.get_mut(&inner).unwrap().kind, Channel::Rotate, 12.0);
        let (sp, w) = preview(&g, inst, &body, inner, Some(0)).expect("splices");
        assert_eq!(sp.input_of(tail, 0), Some(w), "tail reads the live body");
        assert_eq!(crate::anim::get(&sp.nodes[&w].kind, Channel::Rotate), Some(12.0));
        assert_eq!(sp.input_of(w, 0), Some(plate));
    }

    /// A value cable from OUTSIDE into a collapsed node becomes a knob: a
    /// Slider inside drives the channel, the cable lands on the instance's
    /// knob, and the value still arrives.
    #[test]
    fn a_value_cable_into_the_selection_becomes_a_knob() {
        register_ports();
        let mut g = EditorGraph::default();
        let plate = g.add(xf(), Pos2::ZERO);
        let drv = g.add(xf(), Pos2::ZERO);
        let a = g.add(xf(), Pos2::ZERO);
        g.edges.push((a, 0, plate));
        g.view = Some(a);
        crate::anim::set(&mut g.nodes.get_mut(&drv).unwrap().kind, Channel::Rotate, 30.0);
        g.patch.link(PSrc::Chan(drv, Channel::Rotate), PDst::Chan(a, Channel::Rotate));

        let inst = collapse(&mut g, &[a]).expect("collapses");
        let NodeKind::Custom { type_id, params } = &g.nodes[&inst].kind else { panic!() };
        let def = local_tool_def(type_id, &g.subwobs[type_id]);
        assert_eq!(def.params.len(), 1, "one knob");
        let knob = Channel::Custom(crate::anim::intern(&def.params[0].name));
        assert!(params.contains_key(&def.params[0].name));
        assert_eq!(
            g.patch.link_to(PDst::Chan(inst, knob)).map(|l| l.from),
            Some(PSrc::Chan(drv, Channel::Rotate)),
            "the cable re-landed on the knob"
        );
        register_locals(&g);
        let flat = expand(&g, None).unwrap();
        let view = flat.view.unwrap();
        assert_eq!(crate::anim::get(&flat.nodes[&view].kind, Channel::Rotate), Some(30.0));
    }

    /// Refused, with the graph untouched: a Read, and a value cable that
    /// LEAVES the selection (its source would vanish into the subwob).
    #[test]
    fn a_collapse_refuses_what_it_cannot_carry() {
        let mut g = EditorGraph::default();
        let read = g.add(NodeKind::read(0, ""), Pos2::ZERO);
        let a = g.add(xf(), Pos2::ZERO);
        let b = g.add(xf(), Pos2::ZERO);
        g.edges.push((a, 0, read));
        g.edges.push((b, 0, a));
        g.patch.link(PSrc::Chan(a, Channel::Rotate), PDst::Chan(b, Channel::Rotate));
        let before = g.clone();
        assert!(collapse(&mut g, &[read, a]).unwrap_err().contains("footage"));
        assert!(collapse(&mut g, &[a]).unwrap_err().contains("outside the selection"));
        assert!(g == before, "a refusal changes nothing");
    }

    /// Input nodes and controls are subwob FURNITURE: registered like
    /// subwobs, but the add menus must not list them among subwobs.
    #[test]
    fn input_nodes_and_controls_are_not_subwobs() {
        register_ports();
        let (body, _) = passthrough_body();
        register(finish("t.real".into(), "Real".into(), [0; 3], 1, 1, body));
        assert!(crate::editor::is_subwob_type("t.real"));
        for (id, _, _) in INPUTS {
            assert!(!crate::editor::is_subwob_type(id), "{id} listed as a subwob");
        }
        assert!(!crate::editor::is_subwob_type(ControlKind::Slider.type_id()));
        assert_eq!(INPUTS[3], (INPUT_CHANNELS, PortKind::Channels, "Fat Pipe Input"));
    }

    /// A fat pipe into a subwob whose body READS footage (a ChannelSelect off
    /// the fat input): until readers are carried through expansion it must
    /// not unfold — unfolded, the select would bind slot 0 and show another
    /// plate — and the reason is there for the panel to say.
    #[test]
    fn a_body_that_reads_footage_does_not_unfold() {
        register_ports();
        let mut b = EditorGraph::default();
        let fat = b.add(
            NodeKind::Custom { type_id: INPUT_CHANNELS.into(), params: Default::default() },
            Pos2::ZERO,
        );
        let sel = b.add(NodeKind::ChannelSelect(Default::default()), Pos2::ZERO);
        b.edges.push((sel, 0, fat));
        b.view = Some(sel);
        assert!(body_reads_footage(&b).is_some());
        register(finish("t.fat".into(), "Fat".into(), [0; 3], 1, 1, b));
        let inst = NodeKind::Custom { type_id: "t.fat".into(), params: Default::default() };
        assert_eq!(inst.input_kind(0), PortKind::Channels, "the port takes the fat cable");
        let (g, _, _) = host("t.fat");
        assert!(expand(&g, None).is_none(), "a footage-reading body unfolded");
    }

    /// A comp with a collapsed subwob survives save and load: the setup
    /// carries the definition, and the loaded graph renders the same.
    #[test]
    fn a_collapsed_subwob_travels_with_the_setup() {
        register_ports();
        let mut g = EditorGraph::default();
        let plate = g.add(xf(), Pos2::ZERO);
        let a = g.add(xf(), Pos2::ZERO);
        g.edges.push((a, 0, plate));
        g.view = Some(a);
        crate::anim::set(&mut g.nodes.get_mut(&a).unwrap().kind, Channel::Rotate, 7.0);
        collapse(&mut g, &[a]).unwrap();
        let text = serde_json::to_string(&g).unwrap();
        let back: EditorGraph = serde_json::from_str(&text).unwrap();
        assert!(back == g, "the graph round-trips, subwobs and all");
        register_locals(&back);
        let flat = expand(&back, None).unwrap();
        assert_eq!(crate::anim::get(&flat.nodes[&flat.view.unwrap()].kind, Channel::Rotate), Some(7.0));
    }

    #[test]
    fn a_selection_becomes_a_tool_and_round_trips() {
        // plate -> a -> b -> viewer; capture [a, b].
        let mut g = EditorGraph::default();
        let plate = g.add(xf(), Pos2::ZERO);
        let a = g.add(xf(), Pos2::ZERO);
        let b = g.add(xf(), Pos2::ZERO);
        let tail = g.add(xf(), Pos2::ZERO);
        g.edges.push((a, 0, plate));
        g.edges.push((b, 0, a));
        g.edges.push((tail, 0, b));
        crate::anim::set(&mut g.nodes.get_mut(&b).unwrap().kind, Channel::Rotate, 5.0);

        let def = from_selection(&g, &[a, b], "t.captured", "Captured").expect("captures");
        assert_eq!(def.inputs, 1, "one wire crossed the selection");
        // Through the file format and back out.
        let text = to_manifest(&def).expect("serializes");
        let back = from_manifest_str(&text).expect("parses");
        assert_eq!(back.body_hash, def.body_hash, "the body did not survive the round trip");
        register(back);

        let (mut g2, plate2, inst) = host("t.captured");
        let _ = (plate, tail);
        let flat = expand(&g2, None).unwrap();
        // Two Transforms came in, chained, with the outer plate feeding the
        // first — and the captured rotate is still on the second.
        let view = flat.view.unwrap();
        assert_eq!(crate::anim::get(&flat.nodes[&view].kind, Channel::Rotate), Some(5.0));
        let mid = flat.edges.iter().find(|(d, _, _)| *d == view).map(|(_, _, s)| *s).unwrap();
        assert_eq!(
            flat.edges.iter().find(|(d, _, _)| *d == mid).map(|(_, _, s)| *s),
            Some(plate2)
        );
        g2.nodes.remove(&inst);
    }
}
