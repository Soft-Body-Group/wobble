//! The expression patch canvas — the app's *second* node system.
//!
//! Everything here mirrors the image graph in [`crate::editor`]: the same
//! gestures (ctrl-drag slices cables, drag the empty grid to marquee-select,
//! Tab opens a search palette at the cursor, dropping a cable on bare grid
//! adds a node already wired, dropping a node on a cable splices it in),
//! the same magnetism, the same one-frame-stale hit geometry.
//!
//! The one deliberate difference is **flow direction**: images run TOP →
//! BOTTOM in the graph, values run LEFT → RIGHT here. Inputs sit on a value
//! node's left edge, outputs on its right, and every cable is a horizontal
//! bezier. That contrast is the point — a glance tells you which canvas you
//! are looking at.
//!
//! rgb triples (`GainR/G/B` and the nine others) collapse to ONE port
//! carrying a three-strand cable, with a header toggle that expands them
//! into the per-lane ports and links. That is a VIEW and nothing else: the
//! model stores one link per lane in both states, so expanding never
//! rewrites anything — it shows you what was already there.
//!
//! Geometry note: value-node positions are canvas units, and a value node
//! zooms WHOLE with the properties grid — box, type, sockets and all — the
//! way a node in the image graph zooms with its canvas. The properties
//! panels' rule ("geometry scales, type does not") is a rule about widget
//! precision; these are nodes on a canvas, and a canvas that scaled its
//! panels but not its nodes would just look broken.

use crate::anim::Channel;
use crate::editor::{self, EditorGraph, ModKind, PDst, PSrc, PatchBay, PatchLink};
use crate::theme;
use egui::{Color32, Pos2, Rect, Stroke, Vec2};

/// Value-node box at 100%, in points; it scales with the grid from there.
pub const MOD_W: f32 = 168.0;
pub const MOD_H: f32 = 62.0;

/// How near a dropped cable end has to be for a jack to claim it. Generous
/// on purpose — the magnet preview shows exactly which jack is winning.
const SNAP: f32 = 30.0;
/// Pointer distance at which a cable counts as hovered.
const CABLE_HIT: f32 = 7.0;
/// Strand tints for a collapsed rgb cable. Muted, so three strands read as
/// one bundle rather than as a rainbow.
/// Brighter versions for type — a label has to stay readable.
pub const LANE_LABEL: [Color32; 3] = [
    Color32::from_rgb(0xe8, 0x8a, 0x8a),
    Color32::from_rgb(0x8a, 0xd8, 0x9a),
    Color32::from_rgb(0x8a, 0xaa, 0xf0),
];
pub const LANE_COLOR: [Color32; 3] = [
    Color32::from_rgb(0xcf, 0x62, 0x62),
    Color32::from_rgb(0x63, 0xbe, 0x74),
    Color32::from_rgb(0x62, 0x8a, 0xdc),
];

/// One end of a patch jack, with identity — collected per frame so cables
/// know where to land.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SockEnd {
    In(PDst),
    Out(PSrc),
}

impl SockEnd {
    fn is_in(self) -> bool {
        matches!(self, SockEnd::In(_))
    }
}

/// A cable mid-drag: pulled off an output, or off an input looking for one.
#[derive(Clone, Copy)]
pub enum WireDrag {
    FromOut(PSrc),
    FromIn(PDst),
}

/// Every jack drawn this frame and where it landed.
pub type Sockets = Vec<(SockEnd, Pos2)>;

/// Value nodes being dragged: the press point plus each node's position at
/// press time, so the move is recomputed from scratch every frame (a delta
/// accumulator drifts when the pointer leaves the window).
struct ModDrag {
    grab: Pos2,
    orig: Vec<(u64, [f32; 2])>,
}

/// The add-a-value-node palette. `wire` is the cable that opened it by being
/// dropped on bare grid — the new node lands already plugged into it.
struct Search {
    at: Vec2,
    query: String,
    wire: Option<WireDrag>,
}

/// Everything the patch canvas remembers between frames. Lives on the app
/// as one field so the canvas can be lifted out whole.
#[derive(Default)]
pub struct PatchUi {
    /// Selected value nodes (click, shift-click, marquee).
    pub sel: Vec<u64>,
    drag: Option<ModDrag>,
    /// Slicer trail in screen points while ctrl-dragging bare grid.
    slice: Option<Vec<Pos2>>,
    /// Marquee: press origin (screen) and whether it adds to the selection.
    marquee: Option<(Pos2, bool)>,
    search: Option<Search>,
    /// The cable a dragged value node would splice into, resolved during the
    /// cable pass and applied on release.
    splice: Option<PatchLink>,
    /// Screen rects of this frame's value nodes (filled by `mods_ui`).
    rects: Vec<(u64, Rect)>,
    /// Nodes an off-canvas chip has asked to pin. Drained by the app.
    summon: Vec<u64>,
    /// This frame's collapsed ports: ((node, R lane), the three lanes).
    /// Filled by the panels as they draw, read by the cable pass to fold
    /// each lane's link onto the port that is actually on screen.
    rgb_ports: Vec<((u64, Channel), [Channel; 3])>,
    /// Set by the app when Tab was pressed — the raw event is stolen before
    /// egui's focus traversal can eat it, exactly as the graph does.
    pub tab_request: bool,
}

impl PatchUi {
    /// Nodes the canvas wants pinned — an off-canvas cable's chip was
    /// clicked. The app owns the pinned list, so it does the pinning.
    pub fn take_summons(&mut self) -> Vec<u64> {
        std::mem::take(&mut self.summon)
    }

    /// Is a canvas gesture mid-flight? The patch bay lives in its own
    /// window, so the main window's "is the mouse up?" test says nothing
    /// about it — history has to ask here instead, or a drag records a step
    /// per frame.
    pub fn gesture_live(&self) -> bool {
        self.drag.is_some() || self.slice.is_some() || self.marquee.is_some()
    }

    /// Start a frame's port collection. Ports are rebuilt every frame from
    /// what the panels actually drew.
    pub fn clear_ports(&mut self) {
        self.rgb_ports.clear();
    }

    /// A panel drew a collapsed rgb port standing for these three lanes.
    pub fn note_rgb_port(&mut self, node: u64, lanes: [Channel; 3]) {
        self.rgb_ports.push(((node, lanes[0]), lanes));
    }

    /// The lanes a channel belongs to, if it is currently collapsed.
    fn lanes_at(&self, node: u64, ch: Channel) -> Option<[Channel; 3]> {
        self.rgb_ports
            .iter()
            .find(|((n, _), l)| *n == node && l.contains(&ch))
            .map(|(_, l)| *l)
    }

    /// The channel whose jack is actually on screen for `ch`: itself, or its
    /// group's R lane while the triple is collapsed.
    fn port_of(&self, node: u64, ch: Channel) -> Channel {
        self.lanes_at(node, ch).map_or(ch, |l| l[0])
    }

    /// Which strand (0=R, 1=G, 2=B) a channel is, within its collapsed group.
    fn lane_of(&self, node: u64, ch: Channel) -> Option<usize> {
        self.lanes_at(node, ch)?.iter().position(|c| *c == ch)
    }

    /// Drop state that outlived its data (a value node deleted elsewhere).
    fn prune(&mut self, patch: &PatchBay) {
        self.sel.retain(|id| patch.mods.iter().any(|m| m.id == *id));
    }
}

/// Palette of value nodes, shared by the Tab search, the right-click menu
/// and the drop-a-cable-on-grid flow.
pub fn palette() -> Vec<(&'static str, ModKind)> {
    vec![
        ("mult", ModKind::Mult(2.0)),
        ("add", ModKind::Add(1.0)),
        ("max", ModKind::Max(0.0)),
        ("min", ModKind::Min(1.0)),
        ("expr", ModKind::Expr("a * 1".into())),
    ]
}

/// The three lanes of one rgb parameter on a node — `GainR/GainG/GainB` and
/// the nine others like it — found by NAME, so a triple added later groups
/// itself with no list to maintain here.
///
/// This is a VIEW: the patch model always stores one link per lane. Collapsed,
/// the three share one port and one three-strand cable; expanded, you see the
/// links exactly as they are. Nothing about the wiring changes either way.
#[derive(Clone, PartialEq)]
pub struct RgbGroup {
    /// The shared name with the lane letter dropped — "Gain", "GlowTint".
    pub stem: String,
    /// R, G, B in that order.
    pub lanes: [Channel; 3],
}

/// Find the rgb triples in a node's channel list. A stem only groups when
/// ALL THREE lanes are present, which is what keeps `KeyCenterL/A/B` (Oklab,
/// and only one of those letters is rgb) from collapsing into nonsense.
///
/// The lane letter matches in either case, because built-in channels spell
/// it `GainR` while a tool's params spell it `r` — `tools::ControlKind::Rgb`
/// (the RGB Slider control) exposes exactly `r`/`g`/`b`, so its three
/// outputs collapse into one port like any other triple.
pub fn rgb_groups(chans: &[Channel]) -> Vec<RgbGroup> {
    let mut found: Vec<(String, [Option<Channel>; 3])> = Vec::new();
    for ch in chans {
        let key = ch.key();
        // An INDEXED built-in spells its lane before the index
        // ("ZoneGainR:2"); everything else spells it last ("GainR", and a
        // tool control's bare "Custom:r"). The stem keeps the index, so a
        // zone's gain groups with its own zone's and no other's.
        let (name, lane_ch) = match key.split_once(':') {
            Some((name, idx)) if idx.parse::<u16>().is_ok() => {
                let Some(l) = name.as_bytes().last() else { continue };
                (format!("{}:{idx}", &name[..name.len() - 1]), *l)
            }
            _ => {
                let Some(l) = key.as_bytes().last() else { continue };
                (key[..key.len() - 1].to_string(), *l)
            }
        };
        let lane = match lane_ch {
            b'R' | b'r' => 0,
            b'G' | b'g' => 1,
            b'B' | b'b' => 2,
            _ => continue,
        };
        // A tool control's params are bare `r`/`g`/`b`, so the stem is just
        // the `Custom:` prefix — empty stems are fine, mismatched ones are
        // what the all-three-lanes rule already rejects.
        let stem = name;

        match found.iter_mut().find(|(s, _)| *s == stem) {
            Some((_, slots)) => slots[lane] = Some(*ch),
            None => {
                let mut slots = [None; 3];
                slots[lane] = Some(*ch);
                found.push((stem, slots));
            }
        }
    }
    found
        .into_iter()
        .filter_map(|(stem, slots)| {
            Some(RgbGroup { stem, lanes: [slots[0]?, slots[1]?, slots[2]?] })
        })
        .collect()
}

/// Type colour per value node, the patch-canvas answer to `editor::type_color`.
pub fn mod_color(k: &ModKind) -> Color32 {
    match k {
        ModKind::Mult(_) => theme::patch_mult(),
        ModKind::Add(_) => theme::patch_add(),
        ModKind::Max(_) => theme::patch_max(),
        ModKind::Min(_) => theme::patch_min(),
        ModKind::Expr(_) => theme::patch_expr(),
    }
}

/// Every patch link reduced to node-to-node, chasing back through value
/// nodes (which have no presence in the image graph). Returns one entry per
/// distinct source→destination pair with the number of channels it carries,
/// so the graph overlay draws one line where a rig wires six channels.
///
/// A channel that is itself driven still reports as the source here: its own
/// upstream link is a separate entry, so the whole chain stays visible.
///
/// rgb triples collapse the same way they do on the patch canvas, under the
/// same toggle: three lanes of one parameter are ONE port, and the line is
/// drawn with a strand per lane.
pub fn node_to_node(graph: &EditorGraph, expand_rgb: bool) -> Vec<OverlayLink> {
    let patch = &graph.patch;
    fn leaves(patch: &PatchBay, from: PSrc, seen: &mut Vec<u64>, out: &mut Vec<u64>) {
        match from {
            PSrc::Chan(n, _) => out.push(n),
            // A tool parameter is not a node, so it draws no graph line.
            PSrc::Param(_) => {}
            PSrc::Mod(id) => {
                if seen.contains(&id) {
                    return; // a cycle yields what it has, like `patch_value`
                }
                seen.push(id);
                let arity = patch
                    .mods
                    .iter()
                    .find(|m| m.id == id)
                    .map_or(0, |m| m.kind.inputs());
                for slot in 0..arity {
                    if let Some(l) = patch.link_to(PDst::Mod(id, slot)) {
                        leaves(patch, l.from, seen, out);
                    }
                }
            }
        }
    }
    // The destination node's triples, worked out once per node.
    let mut groups: std::collections::HashMap<u64, Vec<RgbGroup>> = Default::default();
    let mut pairs: Vec<OverlayLink> = Vec::new();
    // Ports already counted for a pair, so three lanes of one rgb parameter
    // count once — the badge says how many PORTS the pair carries.
    let mut counted: Vec<(u64, u64, Channel)> = Vec::new();
    for l in &patch.links {
        let PDst::Chan(dst, dc) = l.to else { continue };
        let lane = if expand_rgb {
            None
        } else {
            let gs = groups.entry(dst).or_insert_with(|| match graph.node(dst) {
                Some(n) => rgb_groups(&crate::anim::channels_for(&n.kind)),
                None => Vec::new(),
            });
            gs.iter()
                .find(|g| g.lanes.contains(&dc))
                .and_then(|g| g.lanes.iter().position(|c| *c == dc).map(|i| (g.lanes[0], i)))
        };
        let (port, lane_ix) = match lane {
            Some((p, i)) => (p, Some(i)),
            None => (dc, None),
        };
        let mut srcs = Vec::new();
        leaves(patch, l.from, &mut Vec::new(), &mut srcs);
        for src in srcs {
            if src == dst {
                continue; // a node feeding itself has nothing to draw
            }
            let entry = match pairs.iter_mut().find(|e| e.src == src && e.dst == dst) {
                Some(e) => e,
                None => {
                    pairs.push(OverlayLink {
                        src,
                        dst,
                        ports: 0,
                        lanes: [false; 3],
                    });
                    pairs.last_mut().expect("just pushed")
                }
            };
            if let Some(i) = lane_ix {
                entry.lanes[i] = true;
            }
            if !counted.contains(&(src, dst, port)) {
                counted.push((src, dst, port));
                entry.ports += 1;
            }
        }
    }
    pairs
}

/// One line in the image graph's read-only overlay: which nodes it runs
/// between, how many PORTS it carries (rgb triples counted once), and which
/// rgb lanes those ports cover, so the line draws with the same strands the
/// patch canvas uses.
#[derive(Debug, PartialEq)]
pub struct OverlayLink {
    pub src: u64,
    pub dst: u64,
    pub ports: usize,
    pub lanes: [bool; 3],
}

impl OverlayLink {
    /// Does this line carry a collapsed rgb port?
    pub fn is_rgb(&self) -> bool {
        self.lanes.iter().any(|b| *b)
    }
}

/// The PORTS a node shows in an expression view: `editor::node_channels`,
/// with each rgb triple folded to a single entry on its R lane. Collapsing
/// is a VIEW — the links underneath stay one per lane either way.
pub fn node_ports(kind: &engine::graph::NodeKind, expand_rgb: bool) -> Vec<Channel> {
    let chans = editor::node_channels(kind);
    if expand_rgb {
        return chans;
    }
    let groups = rgb_groups(&chans);
    chans
        .into_iter()
        .filter(|c| {
            groups
                .iter()
                .find(|g| g.lanes.contains(c))
                .is_none_or(|g| g.lanes[0] == *c)
        })
        .collect()
}

/// The triple `ch` belongs to on this node, while collapsed.
pub fn lanes_for(
    kind: &engine::graph::NodeKind,
    ch: Channel,
    expand_rgb: bool,
) -> Option<[Channel; 3]> {
    if expand_rgb {
        return None;
    }
    rgb_groups(&editor::node_channels(kind))
        .into_iter()
        .find(|g| g.lanes.contains(&ch))
        .map(|g| g.lanes)
}

/// The channel whose nodule actually exists for `ch` — itself, or its
/// group's R lane while the triple is collapsed.
pub fn port_channel(
    kind: &engine::graph::NodeKind,
    ch: Channel,
    expand_rgb: bool,
) -> Channel {
    lanes_for(kind, ch, expand_rgb).map_or(ch, |l| l[0])
}

/// The label an expression view puts on a port: the channel's own name, or
/// the triple's stem plus "rgb".
pub fn port_label(
    kind: &engine::graph::NodeKind,
    ch: Channel,
    expand_rgb: bool,
) -> std::borrow::Cow<'static, str> {
    match rgb_groups(&editor::node_channels(kind))
        .into_iter()
        .find(|g| !expand_rgb && g.lanes[0] == ch)
    {
        Some(g) => {
            // "GainR" -> "gain rgb", matching the curve editor's casing. A
            // control node's lane is just "r", which leaves nothing to
            // qualify — it is simply the rgb port.
            let _ = g;
            let pretty = crate::curves::ch_label_for(kind, ch);
            let base = pretty.trim_end_matches(|c: char| c == 'r' || c == ' ');
            std::borrow::Cow::Owned(if base.is_empty() {
                "rgb".to_string()
            } else {
                format!("{base} rgb")
            })
        }
        None => crate::curves::ch_label_for(kind, ch),
    }
}

/// Fold a socket end onto the port that is really drawn for it: while a
/// triple is collapsed, its G and B lanes have no jack of their own and
/// resolve to the group's R-lane port.
fn resolve(st: &PatchUi, end: SockEnd) -> SockEnd {
    match end {
        SockEnd::In(PDst::Chan(n, c)) => SockEnd::In(PDst::Chan(n, st.port_of(n, c))),
        SockEnd::Out(PSrc::Chan(n, c)) => SockEnd::Out(PSrc::Chan(n, st.port_of(n, c))),
        other => other,
    }
}

/// Wire one port to another, fanning out across a collapsed triple: rgb to
/// rgb links lane to lane, a single value BROADCASTS to all three, and an rgb
/// source into a scalar sink takes the R lane. Every one of these is an
/// ordinary per-lane link in the model — that is the whole point.
fn link_ports(patch: &mut PatchBay, st: &PatchUi, src: PSrc, dst: PDst) {
    let sl = match src {
        PSrc::Chan(n, c) => st.lanes_at(n, c).map(|l| (n, l)),
        _ => None,
    };
    let dl = match dst {
        PDst::Chan(n, c) => st.lanes_at(n, c).map(|l| (n, l)),
        _ => None,
    };
    link_lanes(patch, sl, dl, src, dst);
}

/// The same fan-out, for callers that work out the triples themselves — the
/// image graph's link mode has node kinds but no `PatchUi`.
pub fn link_lanes(
    patch: &mut PatchBay,
    sl: Option<(u64, [Channel; 3])>,
    dl: Option<(u64, [Channel; 3])>,
    src: PSrc,
    dst: PDst,
) {
    match (sl, dl) {
        (Some((sn, s)), Some((dn, d))) => {
            for i in 0..3 {
                patch.link(PSrc::Chan(sn, s[i]), PDst::Chan(dn, d[i]));
            }
        }
        (None, Some((dn, d))) => {
            for lane in d {
                patch.link(src, PDst::Chan(dn, lane));
            }
        }
        (Some((sn, s)), None) => patch.link(PSrc::Chan(sn, s[0]), dst),
        (None, None) => patch.link(src, dst),
    }
}

/// Control points of a cable. Horizontal tangents — this canvas flows left
/// to right (the image graph's `wire_ctrl` leans vertical for the same
/// reason).
pub fn cable_ctrl(from: Pos2, to: Pos2) -> [Pos2; 4] {
    let d = ((to.x - from.x).abs() * 0.5).clamp(24.0, 120.0);
    [from, from + Vec2::new(d, 0.0), to - Vec2::new(d, 0.0), to]
}

/// One drawn cable. Usually one link; for a collapsed rgb triple it is the
/// two or three lane links that share a pair of ports, drawn as parallel
/// strands so the port's arity is visible without expanding.
struct Bundle {
    from: SockEnd,
    to: SockEnd,
    links: Vec<PatchLink>,
    /// Which of R/G/B a COLLAPSED triple folded into this cable.
    lanes: [bool; 3],
    /// The lanes of each end, for a cable carrying a single value: red out
    /// into green in ramps red to green, whether or not anything collapsed.
    src_lane: Option<usize>,
    dst_lane: Option<usize>,
}

impl Bundle {
    fn new(from: SockEnd, to: SockEnd) -> Self {
        Self {
            from,
            to,
            links: Vec::new(),
            lanes: [false; 3],
            src_lane: None,
            dst_lane: None,
        }
    }
    fn push(&mut self, l: PatchLink, lane: Option<usize>) {
        if let Some(i) = lane {
            self.lanes[i] = true;
        }
        self.links.push(l);
    }
    /// The colours this cable runs through: the lanes a collapsed triple
    /// folded in, or the two ends it joins.
    fn ramp(&self, fade: f32) -> Vec<Color32> {
        if self.lanes.iter().filter(|b| **b).count() > 1 {
            lane_ramp(self.lanes, fade)
        } else {
            flow_ramp(self.src_lane, self.dst_lane, fade)
        }
    }
}

/// Cables leaving one jack for the same off-canvas node, gathered so a rig
/// that wires six channels shows one chip rather than six.
struct Stub {
    at: Pos2,
    /// Does the missing end lie downstream (to the right) or upstream?
    outward: bool,
    other: EndIdent,
    /// The far end's channels, for the chip's label.
    chans: Vec<String>,
    links: Vec<PatchLink>,
}

/// Who sits at a jack that isn't on this canvas.
#[derive(PartialEq)]
enum EndIdent {
    /// A node that exists but isn't pinned — one click brings it in.
    Node(u64, String),
    /// The node is gone; the cable is dangling and can only be cut.
    Gone,
}

/// Name the far end of a link, and the channel it lands on.
fn end_ident(graph: &EditorGraph, end: SockEnd) -> (EndIdent, String) {
    let (node, ch) = match end {
        SockEnd::In(PDst::Chan(n, c)) | SockEnd::Out(PSrc::Chan(n, c)) => (n, c.key()),
        // A value node is always drawn, so it is only ever missing if it was
        // deleted out from under the link.
        SockEnd::In(PDst::Mod(..)) | SockEnd::Out(PSrc::Mod(_)) => {
            return (EndIdent::Gone, String::new())
        }
        // A tool's own parameter: it names itself, and has no node to visit.
        SockEnd::Out(PSrc::Param(c)) => return (EndIdent::Gone, c.key()),
    };
    match graph.node(node) {
        Some(n) => (EndIdent::Node(node, editor::node_label(&n.kind)), ch),
        None => (EndIdent::Gone, ch),
    }
}

/// Add a link to the stub for its (jack, far node) pair, starting one if
/// this is the first cable between them.
fn push_stub(
    stubs: &mut Vec<Stub>,
    at: Pos2,
    outward: bool,
    (other, chan): (EndIdent, String),
    link: &PatchLink,
) {
    match stubs
        .iter_mut()
        .find(|s| s.at == at && s.outward == outward && s.other == other)
    {
        Some(s) => {
            s.chans.push(chan);
            s.links.push(link.clone());
        }
        None => stubs.push(Stub {
            at,
            outward,
            other,
            chans: vec![chan],
            links: vec![link.clone()],
        }),
    }
}

/// The pan that keeps the canvas point under `at` fixed while the zoom goes
/// from `before` to `after`. The canvas navigates by pan + zoom exactly like
/// the node graph — no scroll offsets, so no clamping and no one-frame lag.
pub fn zoom_about(pan: Vec2, view_min: Pos2, at: Pos2, before: f32, after: f32) -> Vec2 {
    if before <= 0.0 {
        return pan;
    }
    let c = (at - (view_min + pan)) / before; // canvas point under `at`
    (at - view_min) - c * after
}

/// Where a value node's box sits on screen. Anchor and box both scale, so
/// the node holds its place and its proportions as the grid zooms.
fn mod_rect(pos: [f32; 2], origin: Pos2, zoom: f32) -> Rect {
    Rect::from_min_size(
        origin + Vec2::new(pos[0], pos[1]) * zoom,
        Vec2::new(MOD_W, MOD_H) * zoom,
    )
}

/// Jack centres on a value node's box: inputs spread down the left edge
/// below the header, output on the right. Slot order is `a`, `b`, …, and the
/// count comes from the kind, so a value node with a different arity places
/// its sockets without any further help.
fn mod_jacks(r: Rect, n_in: u8) -> (Vec<Pos2>, Pos2) {
    let n = n_in.max(1) as f32;
    let ins = (0..n_in)
        .map(|i| {
            let t = (i as f32 + 0.5) / n;
            Pos2::new(r.min.x, r.min.y + r.height() * (0.36 + 0.44 * t))
        })
        .collect();
    (ins, Pos2::new(r.max.x, r.center().y))
}

// ---------------------------------------------------------------------------
// Jacks
// ---------------------------------------------------------------------------

/// One patch jack laid out in the flow (a channel row's left/right edge).
/// `lanes` is set when this jack is a COLLAPSED rgb port: unplugging or
/// pulling a cable off it acts on all three lanes, not just the one whose
/// channel happens to carry the port's identity.
pub fn jack(
    ui: &mut egui::Ui,
    end: SockEnd,
    patch: &mut PatchBay,
    wire: &mut Option<WireDrag>,
    sockets: &mut Sockets,
    lanes: Option<[Channel; 3]>,
) {
    let (rect, resp) =
        ui.allocate_exact_size(Vec2::splat(14.0), egui::Sense::click_and_drag());
    jack_body(ui, rect, &resp, end, patch, wire, sockets, lanes);
}

/// A jack pinned to a point (the value nodes' edge sockets).
#[allow(clippy::too_many_arguments)]
pub fn jack_at(
    ui: &mut egui::Ui,
    center: Pos2,
    size: f32,
    end: SockEnd,
    patch: &mut PatchBay,
    wire: &mut Option<WireDrag>,
    sockets: &mut Sockets,
) {
    let rect = Rect::from_center_size(center, Vec2::splat(size));
    let resp = ui.interact(
        rect,
        ui.id().with(("patch-jack", end)),
        egui::Sense::click_and_drag(),
    );
    jack_body(ui, rect, &resp, end, patch, wire, sockets, None);
}

/// Draw the socket, register cable gestures, record the position for the
/// cable pass. Dragging out of a CONNECTED input pulls the cable off (to
/// re-plug elsewhere); right-click an input to unplug it.
#[allow(clippy::too_many_arguments)]
fn jack_body(
    ui: &egui::Ui,
    rect: Rect,
    resp: &egui::Response,
    end: SockEnd,
    patch: &mut PatchBay,
    wire: &mut Option<WireDrag>,
    sockets: &mut Sockets,
    lanes: Option<[Channel; 3]>,
) {
    let center = rect.center();
    sockets.push((end, center));
    // A collapsed port counts as linked when ANY of its lanes is.
    let (is_in, linked) = match (end, lanes) {
        (SockEnd::In(PDst::Chan(n, _)), Some(l)) => (
            true,
            l.iter().any(|c| patch.link_to(PDst::Chan(n, *c)).is_some()),
        ),
        (SockEnd::Out(PSrc::Chan(n, _)), Some(l)) => (
            false,
            l.iter()
                .any(|c| patch.links.iter().any(|k| k.from == PSrc::Chan(n, *c))),
        ),
        (SockEnd::In(d), _) => (true, patch.link_to(d).is_some()),
        (SockEnd::Out(s), _) => (false, patch.links.iter().any(|l| l.from == s)),
    };
    let base = if is_in { theme::cable() } else { theme::cable_source() };
    // A cable looking for a home lights up every jack it could take, so the
    // legal targets are obvious before the drop. The socket scales with its
    // own hit rect, which is how a value node's jacks follow the grid zoom.
    let courted = matches!(
        (wire.as_ref(), is_in),
        (Some(WireDrag::FromOut(_)), true) | (Some(WireDrag::FromIn(_)), false)
    );
    let r = rect.width() * 0.32;
    let hot = linked || resp.hovered() || courted;
    let col = if hot { base } else { base.gamma_multiply(0.55) };
    let p = ui.painter();
    if courted {
        p.circle_filled(center, r * 1.7, base.gamma_multiply(0.12));
    }
    p.circle_stroke(center, r, Stroke::new(1.5, col));
    if linked {
        p.circle_filled(center, r * 0.49, col);
    }
    if resp.hovered() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
    }
    // alt-drag is the canvas pan, never a new cable.
    if resp.drag_started() && !ui.input(|i| i.modifiers.alt) {
        *wire = Some(match end {
            SockEnd::Out(s) => WireDrag::FromOut(s),
            SockEnd::In(d) => {
                // Pulling off a collapsed port pulls all of its lanes; the
                // cable then re-lands as a triple wherever you drop it.
                let all: Vec<PDst> = match (d, lanes) {
                    (PDst::Chan(n, _), Some(l)) => {
                        l.iter().map(|c| PDst::Chan(n, *c)).collect()
                    }
                    _ => vec![d],
                };
                let src = all.iter().find_map(|x| patch.link_to(*x).map(|l| l.from));
                match src {
                    Some(from) => {
                        for x in all {
                            patch.unlink(x);
                        }
                        WireDrag::FromOut(from)
                    }
                    None => WireDrag::FromIn(d),
                }
            }
        });
    }
    if resp.secondary_clicked() {
        if let SockEnd::In(d) = end {
            match (d, lanes) {
                // Unplugging a collapsed port unplugs the whole triple.
                (PDst::Chan(n, _), Some(l)) => {
                    for c in l {
                        patch.unlink(PDst::Chan(n, c));
                    }
                }
                _ => patch.unlink(d),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Canvas gestures (bare grid)
// ---------------------------------------------------------------------------

/// Gestures that belong to the empty grid: slice, marquee, Tab search, the
/// add menu, Delete. Call this right after the canvas rect is allocated —
/// the panels and value nodes drawn afterwards sit on top and win their own
/// hits, exactly as nodes do over the graph background.
pub fn canvas_gestures(
    st: &mut PatchUi,
    ui: &egui::Ui,
    resp: &egui::Response,
    graph: &mut EditorGraph,
    origin: Pos2,
    zoom: f32,
    menu_at: &mut Vec2,
) {
    st.prune(&graph.patch);
    // Plain ctrl, not command: ⌘-scroll already zooms this grid, and the
    // image graph binds its slicer to ctrl too.
    let ctrl = ui.input(|i| i.modifiers.ctrl);
    let shift = ui.input(|i| i.modifiers.shift);
    let typing = ui.ctx().egui_wants_keyboard_input();

    // ctrl-drag slices cables. macOS reports ctrl-drag as a secondary-button
    // drag, so accept both — the graph editor has the same pair.
    let sliced = (resp.drag_started_by(egui::PointerButton::Primary)
        || resp.drag_started_by(egui::PointerButton::Secondary))
        && ctrl;
    if sliced {
        if let Some(p) = ui.input(|i| i.pointer.press_origin()).or(resp.hover_pos()) {
            st.slice = Some(vec![p]);
        }
    } else if resp.drag_started_by(egui::PointerButton::Primary)
        && !ui.input(|i| i.modifiers.alt)
    {
        // alt-drag pans (the app handles it) — it must not also marquee.
        if let Some(p) = ui.input(|i| i.pointer.press_origin()).or(resp.hover_pos()) {
            st.marquee = Some((p, shift));
        }
    }
    // A plain click on bare grid drops the selection (Esc does too).
    if resp.clicked() && !shift {
        st.sel.clear();
    }
    if !typing && ui.input(|i| i.key_pressed(egui::Key::Escape)) {
        st.sel.clear();
        st.search = None;
    }

    // Tab: the palette at the cursor, same as the graph's node search.
    if std::mem::take(&mut st.tab_request) && resp.contains_pointer() && !typing {
        let at = resp
            .hover_pos()
            .map(|p| (p - origin) / zoom)
            .unwrap_or(Vec2::ZERO);
        st.search = Some(Search { at, query: String::new(), wire: None });
    }

    // Delete removes the selected value nodes and every cable touching them.
    if !typing
        && !st.sel.is_empty()
        && resp.contains_pointer()
        && ui.input(|i| {
            i.key_pressed(egui::Key::Delete) || i.key_pressed(egui::Key::Backspace)
        })
    {
        for id in std::mem::take(&mut st.sel) {
            graph.patch.remove_mod(id);
        }
    }

    if resp.secondary_clicked() && !ctrl {
        if let Some(p) = resp.interact_pointer_pos() {
            *menu_at = (p - origin) / zoom;
        }
    }
    let at = *menu_at;
    resp.context_menu(|ui| {
        ui.label(egui::RichText::new("value nodes").small().weak());
        for (name, kind) in palette() {
            if ui.button(name).clicked() {
                let id = graph.patch.add_mod(kind, [at.x, at.y]);
                st.sel = vec![id];
                ui.close();
            }
        }
        ui.separator();
        if ui
            .add_enabled(!graph.patch.links.is_empty(), egui::Button::new("unplug everything"))
            .clicked()
        {
            graph.patch.links.clear();
            ui.close();
        }
    });
}

// ---------------------------------------------------------------------------
// Value nodes
// ---------------------------------------------------------------------------

/// The value nodes floating on the patch canvas: mult / merges / expr. Drawn
/// like graph nodes — type colour, title, edge sockets — and driven the same
/// way: click selects, shift adds, dragging moves the whole selection.
pub fn mods_ui(
    st: &mut PatchUi,
    ui: &mut egui::Ui,
    graph: &mut EditorGraph,
    wire: &mut Option<WireDrag>,
    origin: Pos2,
    zoom: f32,
    sockets: &mut Sockets,
) {
    st.rects.clear();
    // A drag that ended while this canvas was hidden (link mode toggled off
    // mid-gesture) would otherwise resume from a stale grab point. The
    // release frame itself is left alone — `cables_ui` still needs it to
    // apply a pending splice.
    if st.drag.is_some()
        && !ui.input(|i| i.pointer.any_down() || i.pointer.any_released())
    {
        st.drag = None;
    }
    let mut kill: Option<u64> = None;
    let mut dup: Option<u64> = None;
    let (shift, alt) = ui.input(|i| (i.modifiers.shift, i.modifiers.alt));
    let snapshot: Vec<(u64, ModKind, [f32; 2])> = graph
        .patch
        .mods
        .iter()
        .map(|m| (m.id, m.kind.clone(), m.pos))
        .collect();

    for (id, kind, pos) in &snapshot {
        let (id, pos) = (*id, *pos);
        let rect = mod_rect(pos, origin, zoom);
        st.rects.push((id, rect));
        let selected = st.sel.contains(&id);

        // Body first, so the frame paints under its contents.
        let hovered = ui.rect_contains_pointer(rect);
        let stroke = if selected {
            Stroke::new(1.8 * zoom, theme::selection())
        } else if hovered {
            Stroke::new(1.2 * zoom, theme::border_hover())
        } else {
            Stroke::new(1.0 * zoom, theme::border())
        };
        let p = ui.painter();
        p.rect_filled(rect.expand(zoom), 6.0 * zoom, theme::shadow());
        p.rect_filled(rect, 5.0 * zoom, theme::window());
        p.rect_stroke(rect, 5.0 * zoom, stroke, egui::StrokeKind::Inside);
        // Type colour as a bar across the TOP — a graph node wears it as a
        // dot, but the left and right edges belong to the jacks here.
        p.rect_filled(
            Rect::from_min_size(
                rect.min + Vec2::new(4.0, 2.0) * zoom,
                Vec2::new(MOD_W - 8.0, 2.5) * zoom,
            ),
            1.0 * zoom,
            mod_color(kind),
        );

        // The whole box is the grab handle, registered BEFORE its contents so
        // the value widget and the jacks (drawn after) win their own hits.
        let dr = ui.interact(
            rect,
            ui.id().with(("patch-mod", id)),
            egui::Sense::click_and_drag(),
        );

        // Header + widget, inset to leave the edge jacks their room. The
        // contents zoom with the box — type included — because this is a
        // node on a canvas, not a properties panel. Clipped to the box so a
        // long expression can never bleed over its neighbours.
        let body = Rect::from_min_max(
            rect.min + Vec2::new(16.0, 7.0) * zoom,
            rect.max - Vec2::new(12.0, 4.0) * zoom,
        );
        ui.scope_builder(egui::UiBuilder::new().max_rect(body), |ui| {
            ui.set_clip_rect(body.intersect(ui.clip_rect()));
            ui.style_mut()
                .text_styles
                .values_mut()
                .for_each(|f| f.size *= zoom);
            let sp = ui.spacing_mut();
            sp.item_spacing = Vec2::new(4.0, 3.0) * zoom;
            sp.button_padding *= zoom;
            sp.interact_size *= zoom;
            ui.vertical(|ui| {
                ui.horizontal(|ui| {
                    ui.label(
                        egui::RichText::new(kind.label())
                            .strong()
                            .small()
                            .color(mod_color(kind)),
                    );
                    ui.with_layout(
                        egui::Layout::right_to_left(egui::Align::Center),
                        |ui| {
                            if ui
                                .add(egui::Button::new("×").frame(false).small())
                                .on_hover_text("delete value node")
                                .clicked()
                            {
                                kill = Some(id);
                            }
                        },
                    );
                });
                let b_wired = graph.patch.link_to(PDst::Mod(id, 1)).is_some();
                if let Some(m) = graph.patch.mods.iter_mut().find(|m| m.id == id) {
                    match &mut m.kind {
                        ModKind::Mult(k)
                        | ModKind::Add(k)
                        | ModKind::Max(k)
                        | ModKind::Min(k) => {
                            ui.add_enabled(
                                !b_wired,
                                egui::DragValue::new(k).update_while_editing(false).speed(crate::widgets::rated(ui, 0.01)).prefix("b "),
                            )
                            .on_hover_text("the b operand")
                            .on_disabled_hover_text("b comes from its jack while wired");
                        }
                        ModKind::Expr(x) => {
                            ui.add(
                                egui::TextEdit::singleline(x)
                                    .font(egui::TextStyle::Monospace)
                                    .hint_text("a * 2 + sin(t/10)")
                                    .desired_width(f32::INFINITY),
                            );
                        }
                    }
                }
            });
        });

        // Edge jacks, with their operand letters just inside the box.
        let jz = 16.0 * zoom;
        let (ins, out) = mod_jacks(rect, kind.inputs());
        for (slot, c) in ins.iter().enumerate() {
            jack_at(
                ui,
                *c,
                jz,
                SockEnd::In(PDst::Mod(id, slot as u8)),
                &mut graph.patch,
                wire,
                sockets,
            );
            ui.painter().text(
                *c + Vec2::new(7.0 * zoom, 0.0),
                egui::Align2::LEFT_CENTER,
                (b'a' + slot as u8) as char,
                egui::FontId::proportional(9.0 * zoom),
                theme::text(),
            );
        }
        jack_at(ui, out, jz, SockEnd::Out(PSrc::Mod(id)), &mut graph.patch, wire, sockets);

        if dr.hovered() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
        }
        // alt-drag is the canvas pan, never a value-node move.
        if dr.drag_started() && !alt {
            if !selected {
                if !shift {
                    st.sel.clear();
                }
                st.sel.push(id);
            }
            if let Some(g) = dr.interact_pointer_pos() {
                let moving = st.sel.clone();
                st.drag = Some(ModDrag {
                    grab: g,
                    orig: snapshot
                        .iter()
                        .filter(|(i, ..)| moving.contains(i))
                        .map(|(i, _, p)| (*i, *p))
                        .collect(),
                });
            }
        }
        if dr.clicked() {
            if shift {
                if !selected {
                    st.sel.push(id);
                }
            } else {
                st.sel = vec![id];
            }
        }
        dr.context_menu(|ui| {
            if ui.button("duplicate").clicked() {
                dup = Some(id);
                ui.close();
            }
            if ui.button("unplug").clicked() {
                graph.patch.links.retain(|l| {
                    !matches!(l.from, PSrc::Mod(m) if m == id)
                        && !matches!(l.to, PDst::Mod(m, _) if m == id)
                });
                ui.close();
            }
            if ui.button("delete").clicked() {
                kill = Some(id);
                ui.close();
            }
        });
    }

    // Live move: recomputed from the press-time snapshot, never accumulated.
    let held = ui.input(|i| i.pointer.any_down());
    if let (true, Some(d), Some(p)) = (held, &st.drag, ui.ctx().pointer_latest_pos()) {
        let off = (p - d.grab) / zoom;
        for (id, orig) in &d.orig {
            if let Some(m) = graph.patch.mods.iter_mut().find(|m| m.id == *id) {
                m.pos = [orig[0] + off.x, orig[1] + off.y];
            }
        }
    }

    if let Some(id) = dup {
        if let Some(m) = graph.patch.mods.iter().find(|m| m.id == id).cloned() {
            let new = graph
                .patch
                .add_mod(m.kind.clone(), [m.pos[0] + 24.0, m.pos[1] + 24.0]);
            st.sel = vec![new];
        }
    }
    if let Some(id) = kill {
        graph.patch.remove_mod(id);
        st.sel.retain(|s| *s != id);
    }
}

// ---------------------------------------------------------------------------
// Cables
// ---------------------------------------------------------------------------

/// Draw every cable, resolve the gestures that need cable geometry (slice,
/// marquee, splice, magnetised drop), and land the cable in flight.
///
/// `settled` is a shape slot the caller reserves after the panels and before
/// the value nodes: resting cables lie ACROSS the panels (their jacks are
/// inside those panels — routed underneath, the cables would be invisible)
/// but pass UNDER the value nodes, so none ever slashes through an
/// expression field. The cable in your hand, the slicer trail, the marquee
/// and the value pill all paint here, on top of everything.
#[allow(clippy::too_many_arguments)]
pub fn cables_ui(
    st: &mut PatchUi,
    ui: &egui::Ui,
    graph: &mut EditorGraph,
    wire: &mut Option<WireDrag>,
    sockets: &Sockets,
    vals: &[(PSrc, f32)],
    settled: egui::layers::ShapeIdx,
    origin: Pos2,
    zoom: f32,
) {
    let find = |e: SockEnd| sockets.iter().find(|(s, _)| *s == e).map(|(_, p)| *p);
    let pointer = ui.ctx().pointer_latest_pos();
    let released = ui.input(|i| i.pointer.any_released());
    let width = 1.6 * zoom.clamp(0.75, 1.6);

    // --- magnetism: the nearest legal jack claims the cable in flight, and
    // the cable visibly snaps to it well before the release ---
    let snap = match (wire.as_ref(), pointer) {
        (Some(w), Some(p)) => nearest(sockets, p, matches!(w, WireDrag::FromOut(_))),
        _ => None,
    };

    // --- sort the links: both ends on this canvas draw as cables, one end
    // missing draws as a stub you can summon the other end from, neither end
    // present can only be counted (nothing to anchor a stub to) ---
    let mut geom: Vec<(Bundle, [Pos2; 4])> = Vec::new();
    let mut stubs: Vec<Stub> = Vec::new();
    let mut adrift: Vec<PatchLink> = Vec::new();
    for l in &graph.patch.links {
        // Every end is resolved first: a lane hidden inside a collapsed
        // triple is carried by its group's port, not missing.
        let (rf, rt) = (resolve(st, SockEnd::Out(l.from)), resolve(st, SockEnd::In(l.to)));
        match (find(rf), find(rt)) {
            (Some(f), Some(t)) => {
                // Links that resolve to the same pair of ports are ONE cable.
                let lane = match l.to {
                    PDst::Chan(n, c) => st.lane_of(n, c),
                    _ => None,
                };
                // Each end's own lane, independent of the collapse state.
                let end = |n: u64, c: Channel| {
                    graph.node(n).and_then(|nd| lane_index(&nd.kind, c))
                };
                let sl = match l.from {
                    PSrc::Chan(n, c) => end(n, c),
                    _ => None,
                };
                let dl = match l.to {
                    PDst::Chan(n, c) => end(n, c),
                    _ => None,
                };
                match geom.iter_mut().find(|(b, _)| b.from == rf && b.to == rt) {
                    Some((b, _)) => b.push(l.clone(), lane),
                    None => {
                        let mut b = Bundle::new(rf, rt);
                        b.src_lane = sl;
                        b.dst_lane = dl;
                        b.push(l.clone(), lane);
                        geom.push((b, cable_ctrl(f, t)));
                    }
                }
            }
            // Source is here, the sink is not: the cable leaves to the right.
            (Some(f), None) => {
                push_stub(&mut stubs, f, true, end_ident(graph, rt), l)
            }
            // Sink is here, the source is not: it arrives from the left.
            (None, Some(t)) => {
                push_stub(&mut stubs, t, false, end_ident(graph, rf), l)
            }
            (None, None) => adrift.push(l.clone()),
        }
    }
    let hovered = (wire.is_none()
        && st.drag.is_none()
        && st.slice.is_none()
        && st.marquee.is_none())
        .then(|| {
            pointer.and_then(|p| {
                geom.iter()
                    .enumerate()
                    .map(|(i, (_, c))| (i, editor::bezier_min_dist(c, p)))
                    .min_by(|a, b| a.1.total_cmp(&b.1))
                    .filter(|(_, d)| *d < CABLE_HIT)
                    .map(|(i, _)| i)
            })
        })
        .flatten();

    // --- splice: a single value node dragged over a cable takes it over ---
    st.splice = None;
    if let Some(d) = &st.drag {
        if d.orig.len() == 1 {
            let id = d.orig[0].0;
            if let Some((_, r)) = st.rects.iter().find(|(i, _)| *i == id) {
                let c = r.center();
                // Only a single-link cable takes a splice: dropping one
                // value node onto an rgb bundle would need three of them,
                // and silently picking a lane would be a lie.
                // Only a single-link cable takes a splice: dropping one
                // value node onto an rgb bundle would need three of them,
                // and silently picking a lane would be a lie.
                st.splice = geom
                    .iter()
                    .filter_map(|(b, cc)| {
                        let [l] = b.links.as_slice() else { return None };
                        let touches_me = matches!(l.from, PSrc::Mod(m) if m == id)
                            || matches!(l.to, PDst::Mod(m, _) if m == id);
                        (!touches_me)
                            .then(|| (l.clone(), editor::bezier_min_dist(cc, c)))
                    })
                    .min_by(|a, b| a.1.total_cmp(&b.1))
                    .filter(|(_, d)| *d < r.height() * 0.7)
                    .map(|(l, _)| l);
            }
        }
    }

    // --- resting cables, into the reserved slot (under panels + nodes) ---
    let mut shapes: Vec<egui::Shape> = Vec::new();
    let strand = |shapes: &mut Vec<egui::Shape>, c: &[Pos2; 4], off: f32, w: f32, col| {
        let c = [
            c[0] + Vec2::new(0.0, off),
            c[1] + Vec2::new(0.0, off),
            c[2] + Vec2::new(0.0, off),
            c[3] + Vec2::new(0.0, off),
        ];
        shapes.push(egui::Shape::CubicBezier(
            egui::epaint::CubicBezierShape::from_points_stroke(
                c,
                false,
                Color32::TRANSPARENT,
                Stroke::new(w, col),
            ),
        ));
    };
    for (i, (b, c)) in geom.iter().enumerate() {
        let single = b.links.first();
        let splice = single.is_some() && st.splice.as_ref() == single;
        let (w, hot_col) = if splice {
            (width * 2.2, Some(theme::selection())) // splice preview, as the graph flags one
        } else if hovered == Some(i) {
            (width * 1.8, Some(theme::cable()))
        } else {
            (width, None)
        };
        match hot_col {
            // A splice preview goes solid white, as the graph flags one.
            Some(hot) if splice => {
                strand(&mut shapes, c, 0.0, w, hot);
                shapes.push(chevron(c, hot, w));
            }
            _ => {
                let fade = if hot_col.is_some() { 1.0 } else { 0.85 };
                // A collapsed triple ramps through its lanes; anything else
                // ramps from what it leaves to what it lands on.
                let cols = b.ramp(fade);
                shapes.extend(gradient_cable(c, &cols, w, false));
                let mid = cols.get(cols.len() / 2).copied().unwrap_or(theme::cable());
                shapes.push(chevron(c, mid, w));
            }
        }
    }
    ui.painter().set(settled, egui::Shape::Vec(shapes));

    // --- the cable in flight, plus the slicer trail, on top ---
    let painter = ui.painter();
    let live = |from: Pos2, to: Pos2, col: Color32, w: f32| {
        painter.add(egui::epaint::CubicBezierShape::from_points_stroke(
            cable_ctrl(from, to),
            false,
            Color32::TRANSPARENT,
            Stroke::new(w, col),
        ));
    };
    if let (Some(w), Some(p)) = (wire.as_ref(), pointer) {
        // Snapped end wins over the raw pointer — that IS the feedback.
        let end = snap.and_then(find).unwrap_or(p);
        let col = if snap.is_some() { theme::cable_source() } else { theme::wire_dragged() };
        match w {
            WireDrag::FromOut(src) => {
                if let Some(f) = find(SockEnd::Out(*src)) {
                    live(f, end, col, width * 1.4);
                }
            }
            WireDrag::FromIn(dst) => {
                if let Some(t) = find(SockEnd::In(*dst)) {
                    live(end, t, col, width * 1.4);
                }
            }
        }
        if let Some(sp) = snap.and_then(find) {
            painter.circle_stroke(sp, 8.0, Stroke::new(1.5, theme::cable_source().gamma_multiply(0.8)));
        }
    }

    // Live value on the hovered cable — the number actually travelling it.
    if let Some((b, c)) = hovered.and_then(|i| geom.get(i)) {
        {
            let mid = editor::cubic_point(c, 0.5);
            // An rgb bundle reads out all its lanes, in lane order.
            let mut nums: Vec<String> = b
                .links
                .iter()
                .map(|l| match vals.iter().find(|(s, _)| *s == l.from) {
                    Some((_, v)) => format!("{v:.3}"),
                    None => "—".into(),
                })
                .collect();
            nums.dedup();
            let txt = nums.join(" ");
            let g = painter.layout_no_wrap(
                txt,
                egui::FontId::monospace(10.0),
                theme::text_hot(),
            );
            let r = Rect::from_center_size(mid, g.size() + Vec2::new(8.0, 4.0));
            painter.rect_filled(r, 3.0, theme::hud_bg().gamma_multiply_u8(210));
            painter.rect_stroke(
                r,
                3.0,
                Stroke::new(1.0, theme::cable().gamma_multiply(0.7)),
                egui::StrokeKind::Inside,
            );
            painter.galley(r.center() - g.size() * 0.5, g, Color32::WHITE);
        }
    }

    // --- cables with one end off-canvas: a short dashed stub to a chip that
    // says who is on the other side, and pins them here when clicked. Losing
    // a patch collection must not lose the wiring silently. ---
    let mut cut: Vec<PatchLink> = Vec::new();
    for (i, stub) in stubs.iter().enumerate() {
        let dir = if stub.outward { 1.0 } else { -1.0 };
        // An output can feed several off-canvas nodes: fan those chips apart
        // by how many earlier stubs share this jack.
        let rank = stubs[..i].iter().filter(|s| s.at == stub.at).count();
        let tip = stub.at + Vec2::new(dir * 46.0 * zoom, rank as f32 * 17.0 * zoom);
        dashed(painter, stub.at, tip, theme::cable().gamma_multiply(0.7), width);
        let n = stub.links.len();
        let (text, live) = match &stub.other {
            EndIdent::Node(_, label) if n == 1 => {
                (format!("{label} · {}", stub.chans[0]), true)
            }
            EndIdent::Node(_, label) => (format!("{label} · {n} links"), true),
            EndIdent::Gone => (
                if n == 1 { "deleted node".into() } else { format!("{n} dead links") },
                false,
            ),
        };
        let g = painter.layout_no_wrap(
            text,
            egui::FontId::proportional(10.0),
            if live { theme::text_strong() } else { theme::cut() },
        );
        let pad = Vec2::new(7.0, 3.0);
        let size = g.size() + pad * 2.0;
        let chip = Rect::from_min_size(
            Pos2::new(if stub.outward { tip.x } else { tip.x - size.x }, tip.y - size.y * 0.5),
            size,
        );
        let r = ui.interact(
            chip,
            ui.id().with(("patch-offcanvas", i)),
            egui::Sense::click(),
        );
        let bg = if r.hovered() {
            theme::button_hover()
        } else {
            theme::hud_bg().gamma_multiply_u8(200)
        };
        painter.rect_filled(chip, 4.0, bg);
        painter.rect_stroke(
            chip,
            4.0,
            Stroke::new(1.0, if live { theme::cable().gamma_multiply(0.8) } else { theme::cut() }),
            egui::StrokeKind::Inside,
        );
        painter.galley(chip.min + pad, g, Color32::WHITE);
        if r.hovered() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
        }
        if r.secondary_clicked() {
            cut.extend(stub.links.iter().cloned());
        }
        match &stub.other {
            EndIdent::Node(id, label) => {
                if r
                    .on_hover_text(format!(
                        "{label} is wired here but is not on this canvas\n\
                         {}\n\
                         click to bring it in · right-click to cut",
                        stub.chans.join(", ")
                    ))
                    .clicked()
                {
                    st.summon.push(*id);
                }
            }
            EndIdent::Gone => {
                if r
                    .on_hover_text(
                        "the node on the other end is gone\nclick to cut the cable",
                    )
                    .clicked()
                {
                    cut.extend(stub.links.iter().cloned());
                }
            }
        }
    }

    // Links with BOTH ends off-canvas have nothing to hang a stub from, so
    // they get a badge pinned to the corner of the view instead.
    let hidden_nodes = |links: &[PatchLink]| -> Vec<u64> {
        let mut v: Vec<u64> = Vec::new();
        for l in links {
            for e in [SockEnd::Out(l.from), SockEnd::In(l.to)] {
                if let (EndIdent::Node(id, _), _) = end_ident(graph, e) {
                    if find(e).is_none() && !v.contains(&id) {
                        v.push(id);
                    }
                }
            }
        }
        v
    };
    if !adrift.is_empty() {
        let n = adrift.len();
        let g = painter.layout_no_wrap(
            format!("{n} cable{} off-canvas", if n == 1 { "" } else { "s" }),
            egui::FontId::proportional(11.0),
            theme::text_strong(),
        );
        let pad = Vec2::new(9.0, 5.0);
        let badge = Rect::from_min_size(
            ui.clip_rect().min + Vec2::splat(10.0),
            g.size() + pad * 2.0,
        );
        let r = ui.interact(
            badge,
            ui.id().with("patch-adrift"),
            egui::Sense::click(),
        );
        painter.rect_filled(
            badge,
            5.0,
            if r.hovered() { theme::button_hover() } else { theme::hud_bg().gamma_multiply_u8(215) },
        );
        painter.rect_stroke(
            badge,
            5.0,
            Stroke::new(1.0, theme::cable().gamma_multiply(0.8)),
            egui::StrokeKind::Inside,
        );
        painter.galley(badge.min + pad, g, Color32::WHITE);
        if r.hovered() {
            ui.ctx().set_cursor_icon(egui::CursorIcon::PointingHand);
        }
        if r
            .on_hover_text(
                "patch cables whose BOTH ends are off this canvas\nclick to bring every missing node in",
            )
            .clicked()
        {
            st.summon.extend(hidden_nodes(&adrift));
        }
    }
    if !cut.is_empty() {
        graph.patch.links.retain(|x| !cut.contains(x));
    }

    // --- slicer: cut every cable the trail crosses, red trail like the graph ---
    if let Some(path) = &mut st.slice {
        if let Some(p) = pointer {
            let a = *path.last().unwrap_or(&p);
            if (p - a).length() > 1.0 {
                // Slicing a collapsed rgb cable cuts all of its lanes: one
                // stroke through one cable is one cut, whatever it stands for.
                let severed: Vec<PatchLink> = geom
                    .iter()
                    .filter(|(_, c)| editor::bezier_hits_segment(c, a, p))
                    .flat_map(|(b, _)| b.links.iter().cloned())
                    .collect();
                graph.patch.links.retain(|l| !severed.contains(l));
                path.push(p);
            }
        }
        painter.add(egui::Shape::line(path.clone(), Stroke::new(1.5, theme::cut())));
        if released {
            st.slice = None;
        }
    }

    // --- marquee over the value nodes ---
    if let Some((start, additive)) = st.marquee {
        if let Some(p) = pointer {
            let r = Rect::from_two_pos(start, p);
            painter.rect_filled(r, 0.0, theme::marquee_fill());
            painter.rect_stroke(
                r,
                0.0,
                Stroke::new(1.0, theme::marquee()),
                egui::StrokeKind::Outside,
            );
            if released {
                if !additive {
                    st.sel.clear();
                }
                for (id, mr) in &st.rects {
                    if r.intersects(*mr) && !st.sel.contains(id) {
                        st.sel.push(*id);
                    }
                }
            }
        }
        if released {
            st.marquee = None;
        }
    }

    // --- releases: land the cable, apply the splice ---
    if released {
        if let Some(d) = st.drag.take() {
            if let (Some(l), Some((id, ..))) = (st.splice.take(), d.orig.first()) {
                // Same deal as the graph's node-on-wire splice: source into
                // the node's `a`, node's output onward to the old sink.
                graph.patch.link(l.from, PDst::Mod(*id, 0));
                graph.patch.link(PSrc::Mod(*id), l.to);
            }
        }
        if let (Some(w), Some(p)) = (wire.take(), pointer) {
            match (w, snap) {
                // `link_ports` fans a collapsed triple out into its lanes.
                (WireDrag::FromOut(src), Some(SockEnd::In(d))) => {
                    link_ports(&mut graph.patch, st, src, d)
                }
                (WireDrag::FromIn(dst), Some(SockEnd::Out(s))) => {
                    link_ports(&mut graph.patch, st, s, dst)
                }
                // Dropped on bare grid: offer the palette, pre-wired. The
                // graph's Tab search with the cable already in hand. A twitch
                // off the jack is a cancel, not a request for a palette.
                (w, None) => {
                    let src = match w {
                        WireDrag::FromOut(s) => find(SockEnd::Out(s)),
                        WireDrag::FromIn(d) => find(SockEnd::In(d)),
                    };
                    if src.is_none_or(|q| q.distance(p) > 24.0) {
                        st.search = Some(Search {
                            at: (p - origin) / zoom
                                - Vec2::new(MOD_W * 0.5, MOD_H * 0.5),
                            query: String::new(),
                            wire: Some(w),
                        });
                    }
                }
                _ => {}
            }
        }
    }
}

/// Blend two colours. Premultiplied throughout, which is what `Color32`
/// stores, so a faded lane mixes without going grey at the seam.
fn mix(a: Color32, b: Color32, t: f32) -> Color32 {
    let l = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t).round() as u8;
    Color32::from_rgba_premultiplied(
        l(a.r(), b.r()),
        l(a.g(), b.g()),
        l(a.b(), b.b()),
        l(a.a(), b.a()),
    )
}

/// A cable painted as a GRADIENT along its length, ramping through `cols`
/// from source end to sink end. An rgb bundle passes its wired lanes, so it
/// runs red to blue and the colour IS the arity; a plain cable passes the
/// two sockets it joins, so it runs from what it leaves to what it lands on
/// and the colour is the direction. One colour draws solid.
pub fn gradient_cable(c: &[Pos2; 4], cols: &[Color32], w: f32, dash: bool) -> Vec<egui::Shape> {
    // Enough segments that the ramp reads as continuous on a long cable and
    // the curve still looks like a curve.
    const N: usize = 28;
    let pieces: Vec<Vec<Pos2>> = (1..=N)
        .map(|k| {
            let t = |k: usize| k as f32 / N as f32;
            vec![editor::cubic_point(c, t(k - 1)), editor::cubic_point(c, t(k))]
        })
        .collect();
    gradient_pieces(&pieces, cols, w, dash)
}

/// `gradient_cable` along any path cut into pieces of equal length (the
/// graph's right-angle cables keep their corners in theirs), one ramp step
/// per piece.
pub fn gradient_pieces(pieces: &[Vec<Pos2>], cols: &[Color32], w: f32, dash: bool) -> Vec<egui::Shape> {
    if cols.is_empty() || pieces.is_empty() {
        return Vec::new();
    }
    let at = |u: f32| -> Color32 {
        if cols.len() == 1 {
            return cols[0];
        }
        let x = u.clamp(0.0, 1.0) * (cols.len() - 1) as f32;
        let i = (x.floor() as usize).min(cols.len() - 2);
        mix(cols[i], cols[i + 1], x - i as f32)
    };
    let n = pieces.len();
    let mut out = Vec::with_capacity(n);
    for (i, piece) in pieces.iter().enumerate() {
        let k = i + 1;
        if dash && k % 2 == 0 {
            continue;
        }
        let stroke = Stroke::new(w, at((k as f32 - 0.5) / n as f32));
        match piece.as_slice() {
            [a, b] => out.push(egui::Shape::line_segment([*a, *b], stroke)),
            p if p.len() > 2 => out.push(egui::Shape::line(p.to_vec(), stroke)),
            _ => {}
        }
    }
    out
}

/// The ramp for a collapsed triple: one stop per WIRED lane, so a two-lane
/// bundle ramps between those two and a single lane is solid. The cable
/// never claims more than it carries.
pub fn lane_ramp(lanes: [bool; 3], fade: f32) -> Vec<Color32> {
    lanes
        .iter()
        .enumerate()
        .filter(|(_, on)| **on)
        .map(|(k, _)| LANE_COLOR[k].gamma_multiply(fade))
        .collect()
}

/// The ramp for a cable carrying ONE value. Each end takes its own lane's
/// tint when the channel is an rgb lane, and the socket's colour when it is
/// not — so an R output wired into a G input runs red to green, an R into an
/// R is solid red, and two plain scalars run out-green to in-amber.
pub fn flow_ramp(src_lane: Option<usize>, dst_lane: Option<usize>, fade: f32) -> Vec<Color32> {
    vec![
        src_lane.map_or(theme::cable_source(), |k| LANE_COLOR[k]).gamma_multiply(fade),
        dst_lane.map_or(theme::cable(), |k| LANE_COLOR[k]).gamma_multiply(fade),
    ]
}

/// Which rgb lane a channel sits on, whatever the collapse state — a cable
/// end is red because the channel is the R lane, not because a toggle is on.
pub fn lane_index(kind: &engine::graph::NodeKind, ch: Channel) -> Option<usize> {
    rgb_groups(&editor::node_channels(kind))
        .into_iter()
        .find(|g| g.lanes.contains(&ch))
        .and_then(|g| g.lanes.iter().position(|c| *c == ch))
}

/// A dashed run between two points. Dashes say "not a live cable": a stub
/// heading off-canvas, or the read-only overlay in the image graph.
pub fn dashed(painter: &egui::Painter, a: Pos2, b: Pos2, col: Color32, w: f32) {
    let n = ((b - a).length() / 9.0).round().max(3.0) as usize;
    for k in (0..n).step_by(2) {
        let t0 = k as f32 / n as f32;
        let t1 = ((k + 1) as f32 / n as f32).min(1.0);
        painter.line_segment([a + (b - a) * t0, a + (b - a) * t1], Stroke::new(w, col));
    }
}

/// A small chevron at the cable's midpoint, pointing downstream.
fn chevron(c: &[Pos2; 4], col: Color32, w: f32) -> egui::Shape {
    let m = editor::cubic_point(c, 0.5);
    let d = editor::cubic_point(c, 0.53) - editor::cubic_point(c, 0.47);
    let len = d.length();
    if len < 1e-3 {
        return egui::Shape::Noop;
    }
    let f = d / len;
    let n = Vec2::new(-f.y, f.x);
    let s = 3.0 + w;
    egui::Shape::convex_polygon(
        vec![m + f * s, m - f * s * 0.6 + n * s * 0.6, m - f * s * 0.6 - n * s * 0.6],
        col,
        Stroke::NONE,
    )
}

/// Nearest jack of the wanted polarity, within the magnet radius.
fn nearest(sockets: &Sockets, p: Pos2, want_in: bool) -> Option<SockEnd> {
    sockets
        .iter()
        .filter(|(sk, _)| sk.is_in() == want_in)
        .map(|(sk, sp)| (*sk, sp.distance(p)))
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .filter(|(_, d)| *d < SNAP)
        .map(|(sk, _)| sk)
}

// ---------------------------------------------------------------------------
// Search palette
// ---------------------------------------------------------------------------

/// The Tab / drop palette. Type to filter, Enter takes the first hit — the
/// graph's node search, with the same muscle memory.
pub fn search_ui(
    st: &mut PatchUi,
    ui: &mut egui::Ui,
    graph: &mut EditorGraph,
    origin: Pos2,
    zoom: f32,
) {
    let Some(s) = &mut st.search else { return };
    let spos = origin + s.at * zoom;
    let mut pick: Option<ModKind> = None;
    let mut close = ui.input(|i| i.key_pressed(egui::Key::Escape));
    let enter = ui.input(|i| i.key_pressed(egui::Key::Enter));
    egui::Area::new(ui.id().with("patch-search"))
        .order(egui::Order::Foreground)
        .fixed_pos(spos)
        .show(ui.ctx(), |ui| {
            egui::Frame::window(ui.style()).show(ui, |ui| {
                ui.set_width(140.0);
                let te = ui.add(
                    egui::TextEdit::singleline(&mut s.query).hint_text("value node…"),
                );
                te.request_focus();
                let q = s.query.to_lowercase();
                let mut first = true;
                for (name, kind) in palette() {
                    if !q.is_empty() && !name.contains(&q) {
                        continue;
                    }
                    if ui.button(name).clicked() || (enter && first) {
                        pick = Some(kind);
                    }
                    first = false;
                }
                if first {
                    ui.weak("no match");
                }
            });
        });
    if let Some(kind) = pick {
        let id = graph.patch.add_mod(kind, [s.at.x, s.at.y]);
        // A palette opened by a dropped cable wires the new node up.
        match s.wire {
            Some(WireDrag::FromOut(src)) => graph.patch.link(src, PDst::Mod(id, 0)),
            Some(WireDrag::FromIn(dst)) => graph.patch.link(PSrc::Mod(id), dst),
            None => {}
        }
        st.sel = vec![id];
        close = true;
    }
    if close {
        st.search = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// (src, dst, ports) for the collapsed overlay — the shape the old tests
    /// were written against.
    fn flat(g: &EditorGraph) -> Vec<(u64, u64, usize)> {
        node_to_node(g, false)
            .into_iter()
            .map(|e| (e.src, e.dst, e.ports))
            .collect()
    }

    fn rig() -> (EditorGraph, u64) {
        let mut g = EditorGraph::default();
        let m = g.patch.add_mod(ModKind::Mult(2.0), [10.0, 10.0]);
        (g, m)
    }

    #[test]
    fn jacks_sit_on_the_left_and_right_edges() {
        // Flow is left to right here: both inputs on the near edge, the
        // output on the far one. (The image graph is top to bottom.)
        let r = mod_rect([0.0, 0.0], Pos2::ZERO, 1.0);
        let (ins, out) = mod_jacks(r, ModKind::Mult(1.0).inputs());
        assert_eq!(ins.len(), 2);
        assert!(ins.iter().all(|p| p.x == r.min.x));
        assert_eq!(out.x, r.max.x);
        assert!(ins[0].y < ins[1].y);
        // Every input clears the header bar and stays inside the box.
        assert!(ins.iter().all(|p| p.y > r.min.y + 12.0 && p.y < r.max.y));
        // Arity drives the spread — a one-input node centres its jack.
        let (one, _) = mod_jacks(r, 1);
        assert_eq!(one.len(), 1);
        assert!((one[0].y - r.center().y).abs() < MOD_H * 0.1);
    }

    #[test]
    fn cable_tangents_are_horizontal() {
        let c = cable_ctrl(Pos2::new(0.0, 0.0), Pos2::new(200.0, 80.0));
        assert_eq!(c[0].y, c[1].y);
        assert_eq!(c[3].y, c[2].y);
        assert!(c[1].x > c[0].x && c[2].x < c[3].x);
    }

    #[test]
    fn value_nodes_zoom_whole_like_the_panels_beside_them() {
        // Anchor AND box scale, so a node holds its spot and its proportions
        // as the grid zooms — a canvas whose panels grew but whose nodes did
        // not is exactly the thing that reads as broken.
        let a = mod_rect([100.0, 40.0], Pos2::ZERO, 1.0);
        let b = mod_rect([100.0, 40.0], Pos2::ZERO, 2.0);
        assert_eq!(b.min, Pos2::new(200.0, 80.0));
        assert_eq!(b.size(), a.size() * 2.0);
        // Sockets ride the box, so they stay on its edges at any zoom.
        let (ins, out) = mod_jacks(b, 2);
        assert!(ins.iter().all(|p| p.x == b.min.x && b.y_range().contains(p.y)));
        assert_eq!(out, Pos2::new(b.max.x, b.center().y));
    }

    #[test]
    fn magnet_takes_the_nearest_jack_of_the_right_polarity() {
        let far = SockEnd::In(PDst::Mod(1, 0));
        let near = SockEnd::In(PDst::Mod(2, 0));
        let out = SockEnd::Out(PSrc::Mod(3));
        let socks: Sockets = vec![
            (far, Pos2::new(20.0, 0.0)),
            (near, Pos2::new(4.0, 0.0)),
            (out, Pos2::new(1.0, 0.0)),
        ];
        assert!(nearest(&socks, Pos2::ZERO, true) == Some(near));
        assert!(nearest(&socks, Pos2::ZERO, false) == Some(out));
        // Outside the magnet radius, nothing claims it.
        assert!(nearest(&socks, Pos2::new(500.0, 0.0), true).is_none());
    }

    #[test]
    fn slicer_cuts_a_cable_it_crosses_and_spares_the_rest() {
        // A left-to-right cable and a vertical stroke through its middle.
        let c = cable_ctrl(Pos2::new(0.0, 0.0), Pos2::new(200.0, 0.0));
        assert!(editor::bezier_hits_segment(
            &c,
            Pos2::new(100.0, -20.0),
            Pos2::new(100.0, 20.0),
        ));
        // A stroke that runs alongside it never touches it — cables bulge
        // horizontally here, so a parallel drag must not cut.
        assert!(!editor::bezier_hits_segment(
            &c,
            Pos2::new(0.0, 40.0),
            Pos2::new(200.0, 40.0),
        ));
    }

    #[test]
    fn overlay_resolves_links_node_to_node_through_value_nodes() {
        use crate::anim::Channel as C;
        let mut g = EditorGraph::default();
        // A -> mult -> B, plus a second channel A -> B straight through.
        let m = g.patch.add_mod(ModKind::Mult(2.0), [0.0, 0.0]);
        g.patch.link(PSrc::Chan(1, C::ScaleX), PDst::Mod(m, 0));
        g.patch.link(PSrc::Mod(m), PDst::Chan(2, C::GainR));
        g.patch.link(PSrc::Chan(1, C::ScaleY), PDst::Chan(2, C::GainG));
        // The value node has no presence in the image graph, so both land as
        // one 1->2 pair carrying two ports. (No real nodes here, so nothing
        // groups — see the rgb test below for that.)
        assert_eq!(flat(&g), vec![(1, 2, 2)]);

        // A merge's second operand is a source too — one line per node.
        g.patch.link(PSrc::Chan(3, C::Rotate), PDst::Mod(m, 1));
        let mut got = flat(&g);
        got.sort();
        assert_eq!(got, vec![(1, 2, 2), (3, 2, 1)]);
    }

    /// The stroke colours of a gradient cable, source end first.
    fn ramp(shapes: &[egui::Shape]) -> Vec<Color32> {
        shapes
            .iter()
            .filter_map(|sh| match sh {
                egui::Shape::LineSegment { stroke, .. } => Some(stroke.color),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn an_rgb_cable_ramps_red_to_blue_along_its_length() {
        let c = cable_ctrl(Pos2::new(0.0, 0.0), Pos2::new(300.0, 0.0));
        let g = ramp(&gradient_cable(&c, &lane_ramp([true; 3], 1.0), 2.0, false));
        assert!(g.len() > 8);
        // Red where it leaves, blue where it lands, green through the middle.
        let close = |a: Color32, b: Color32| {
            (a.r() as i32 - b.r() as i32).abs()
                + (a.g() as i32 - b.g() as i32).abs()
                + (a.b() as i32 - b.b() as i32).abs()
                < 40
        };
        assert!(close(g[0], LANE_COLOR[0]), "{:?}", g[0]);
        assert!(close(*g.last().unwrap(), LANE_COLOR[2]), "{:?}", g.last());
        assert!(close(g[g.len() / 2], LANE_COLOR[1]), "{:?}", g[g.len() / 2]);
        // The ramp is monotonic in blue: it really travels, never bounces.
        assert!(g.windows(2).all(|w| w[1].b() >= w[0].b()));
    }

    #[test]
    fn a_partial_bundle_ramps_only_the_lanes_it_carries() {
        let c = cable_ctrl(Pos2::new(0.0, 0.0), Pos2::new(300.0, 0.0));
        // One lane is solid — nothing to ramp between.
        let one = ramp(&gradient_cable(&c, &lane_ramp([false, true, false], 1.0), 2.0, false));
        assert!(one.iter().all(|c| *c == LANE_COLOR[1]));
        // Two lanes ramp between those two, never through the missing one.
        let two = ramp(&gradient_cable(&c, &lane_ramp([true, false, true], 1.0), 2.0, false));
        assert!(two.iter().all(|c| c.g() <= LANE_COLOR[0].g().max(LANE_COLOR[2].g())));
        // Dashing draws about half the segments; no lanes draws nothing.
        let solid = gradient_cable(&c, &lane_ramp([true; 3], 1.0), 2.0, false).len();
        let dashed = gradient_cable(&c, &lane_ramp([true; 3], 1.0), 2.0, true).len();
        assert!(dashed < solid && dashed * 2 >= solid - 2);
        assert!(gradient_cable(&c, &lane_ramp([false; 3], 1.0), 2.0, false).is_empty());
    }

    #[test]
    fn a_single_cable_ramps_from_its_source_lane_to_its_sink_lane() {
        let c = cable_ctrl(Pos2::new(0.0, 0.0), Pos2::new(300.0, 0.0));
        let close = |a: Color32, b: Color32| {
            (a.r() as i32 - b.r() as i32).abs()
                + (a.g() as i32 - b.g() as i32).abs()
                + (a.b() as i32 - b.b() as i32).abs()
                < 40
        };
        let ends = |sl, dl| {
            let g = ramp(&gradient_cable(&c, &flow_ramp(sl, dl, 1.0), 2.0, false));
            (g[0], *g.last().unwrap())
        };
        // An R output into a G input runs red to green.
        let (a, b) = ends(Some(0), Some(1));
        assert!(close(a, LANE_COLOR[0]) && close(b, LANE_COLOR[1]), "{a:?} {b:?}");
        // Lane to its own lane is that colour end to end.
        let (a, b) = ends(Some(2), Some(2));
        assert!(close(a, LANE_COLOR[2]) && close(b, LANE_COLOR[2]));
        // Two plain scalars keep the socket colours: out-green to in-amber.
        let (a, b) = ends(None, None);
        assert!(close(a, theme::cable_source()) && close(b, theme::cable()), "{a:?} {b:?}");
        // A scalar into a lane takes the socket at one end and the lane at
        // the other — the cable never invents a colour for an end.
        let (a, b) = ends(None, Some(0));
        assert!(close(a, theme::cable_source()) && close(b, LANE_COLOR[0]));
    }

    #[test]
    fn a_tool_rgb_control_collapses_like_any_other_triple() {
        use crate::anim::{intern, Channel as C};
        // `tools::ControlKind::Rgb` — the RGB Slider you drop into a tool
        // body to build its panel — exposes params named exactly r/g/b, so
        // its three outputs are one port and one three-strand cable.
        let ctrl: Vec<C> = crate::tools::ControlKind::Rgb
            .lanes()
            .iter()
            .map(|n| C::Custom(intern(n)))
            .collect();
        let g = rgb_groups(&ctrl);
        assert_eq!(g.len(), 1, "the RGB Slider control must group");
        assert_eq!(g[0].lanes, [ctrl[0], ctrl[1], ctrl[2]]);

        // A Slider control has one param and nothing to group.
        let one: Vec<C> = crate::tools::ControlKind::Slider
            .lanes()
            .iter()
            .map(|n| C::Custom(intern(n)))
            .collect();
        assert!(rgb_groups(&one).is_empty());

        // A tool whose params merely END in r/g/b with different stems is
        // three scalars, not a triple.
        let mixed = vec![
            C::Custom(intern("blur")),
            C::Custom(intern("fog")),
            C::Custom(intern("comb")),
        ];
        assert!(rgb_groups(&mixed).is_empty());
    }

    #[test]
    fn a_graph_node_splays_ports_not_lanes_while_collapsed() {
        use crate::anim::Channel as C;
        use engine::graph::NodeKind;
        let grade = NodeKind::Grade(Default::default());
        let all = editor::node_channels(&grade);
        let ports = node_ports(&grade, false);
        // Grade's four rgb parameters fold to four nodules, so the splay
        // loses eight rows without losing a single link.
        assert_eq!(ports.len(), all.len() - 8);
        assert!(ports.contains(&C::GainR));
        assert!(!ports.contains(&C::GainG) && !ports.contains(&C::GainB));
        // Every scalar on the node survives untouched.
        assert!(all
            .iter()
            .filter(|c| rgb_groups(&all).iter().all(|g| !g.lanes.contains(c)))
            .all(|c| ports.contains(c)));
        // Expanded, every lane is its own port again.
        assert_eq!(node_ports(&grade, true), all);

        // Hidden lanes resolve onto the port that is really drawn.
        for lane in [C::GainR, C::GainG, C::GainB] {
            assert_eq!(port_channel(&grade, lane, false), C::GainR);
            assert_eq!(port_channel(&grade, lane, true), lane);
        }
        // ...and the port says it stands for three.
        assert!(port_label(&grade, C::GainR, false).ends_with("rgb"));
        assert!(!port_label(&grade, C::GainR, true).ends_with("rgb"));
    }

    #[test]
    fn overlay_counts_an_rgb_triple_as_one_port_and_marks_its_lanes() {
        use crate::anim::Channel as C;
        use engine::graph::NodeKind;
        let mut g = EditorGraph::default();
        let tf = g.add(NodeKind::Transform(Default::default()), Pos2::ZERO);
        let gr = g.add(NodeKind::Grade(Default::default()), Pos2::ZERO);
        for lane in [C::GainR, C::GainG, C::GainB] {
            g.patch.link(PSrc::Chan(tf, C::ScaleX), PDst::Chan(gr, lane));
        }
        // Collapsed: one line, ONE port, all three strands lit.
        let got = node_to_node(&g, false);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].ports, 1);
        assert_eq!(got[0].lanes, [true; 3]);
        assert!(got[0].is_rgb());

        // Expanded: the same wiring is three ports and no strands — the
        // overlay tells the same story the patch canvas does.
        let got = node_to_node(&g, true);
        assert_eq!(got[0].ports, 3);
        assert!(!got[0].is_rgb());

        // A half-wired triple lights only the lanes it actually carries.
        let mut h = EditorGraph::default();
        let tf = h.add(NodeKind::Transform(Default::default()), Pos2::ZERO);
        let gr = h.add(NodeKind::Grade(Default::default()), Pos2::ZERO);
        h.patch.link(PSrc::Chan(tf, C::ScaleX), PDst::Chan(gr, C::GainG));
        let got = node_to_node(&h, false);
        assert_eq!(got[0].lanes, [false, true, false]);
        assert_eq!(got[0].ports, 1);
    }

    #[test]
    fn overlay_skips_self_links_and_survives_cycles() {
        use crate::anim::Channel as C;
        let mut g = EditorGraph::default();
        // One node's channel driving another of its own: nothing to draw
        // between two nodes, so the overlay stays empty.
        g.patch.link(PSrc::Chan(1, C::ScaleX), PDst::Chan(1, C::GainR));
        assert!(flat(&g).is_empty());

        // A value node feeding itself must terminate, not recurse forever.
        let m = g.patch.add_mod(ModKind::Add(0.0), [0.0, 0.0]);
        g.patch.link(PSrc::Mod(m), PDst::Mod(m, 0));
        g.patch.link(PSrc::Chan(4, C::ScaleY), PDst::Mod(m, 1));
        g.patch.link(PSrc::Mod(m), PDst::Chan(5, C::GainB));
        assert_eq!(flat(&g), vec![(4, 5, 1)]);
    }

    #[test]
    fn zoom_holds_the_point_under_the_cursor_and_pans_without_bounds() {
        let view = Pos2::new(100.0, 50.0);
        let at = Pos2::new(340.0, 190.0);
        let pan = Vec2::new(-620.0, -410.0); // panned up and left, off-canvas
        let (before, after) = (1.0, 2.5);
        let under = |pan: Vec2, z: f32| (at - (view + pan)) / z;
        let moved = zoom_about(pan, view, at, before, after);
        // The canvas point under the cursor is the same before and after.
        let (a, b) = (under(pan, before), under(moved, after));
        assert!((a - b).length() < 1e-3, "{a:?} vs {b:?}");
        // Nothing clamps the pan back into positive space.
        assert!(moved.x < 0.0 && moved.y < 0.0, "{moved:?}");
        // A zoom that does not change leaves the pan exactly alone.
        assert_eq!(zoom_about(pan, view, at, 2.0, 2.0), pan);
    }

    /// A zone's triple carries its index after the lane letter; it still
    /// collapses, and only with its own zone's lanes.
    #[test]
    fn indexed_rgb_triples_group_per_index() {
        use crate::anim::Channel as C;
        let g = rgb_groups(&[
            C::ZoneGainR(0),
            C::ZoneGainG(0),
            C::ZoneGainB(0),
            C::ZoneGainR(1),
            C::ZoneGainG(1),
            C::ZoneSat(1),
            C::LayerTx(0),
            C::KeyCenterL(0),
            C::KeyCenterA(0),
            C::KeyCenterB(0),
        ]);
        let stems: Vec<&str> = g.iter().map(|x| x.stem.as_str()).collect();
        assert_eq!(stems, ["ZoneGain:0"]);
        assert_eq!(g[0].lanes, [C::ZoneGainR(0), C::ZoneGainG(0), C::ZoneGainB(0)]);
    }

    #[test]
    fn rgb_triples_group_by_name_and_partial_ones_do_not() {
        use crate::anim::Channel as C;
        let g = rgb_groups(&[C::GainR, C::GainG, C::GainB, C::BlurSize, C::LiftR]);
        // Gain has all three lanes; Lift has only R, so it stays a scalar.
        assert_eq!(g.len(), 1);
        assert_eq!(g[0].stem, "Gain");
        assert_eq!(g[0].lanes, [C::GainR, C::GainG, C::GainB]);

        // Oklab's L/A/B share a stem and end in one rgb letter between them.
        // Requiring all three lanes is what stops that collapsing.
        let oklab = rgb_groups(&[C::KeyCenterL(0), C::KeyCenterA(0), C::KeyCenterB(0)]);
        assert!(oklab.is_empty());
    }

    #[test]
    fn collapsed_ports_fan_out_and_expanded_ones_do_not() {
        use crate::anim::Channel as C;
        let lanes = [C::GainR, C::GainG, C::GainB];
        let src = [C::LiftR, C::LiftG, C::LiftB];
        let mut st = PatchUi::default();
        st.note_rgb_port(2, lanes);
        st.note_rgb_port(1, src);

        // rgb -> rgb wires lane to lane, one link each, not R to everything.
        let mut p = PatchBay::default();
        link_ports(&mut p, &st, PSrc::Chan(1, C::LiftR), PDst::Chan(2, C::GainR));
        assert_eq!(p.links.len(), 3);
        for i in 0..3 {
            let l = p.links.iter().find(|l| l.to == PDst::Chan(2, lanes[i])).unwrap();
            assert_eq!(l.from, PSrc::Chan(1, src[i]));
        }

        // A scalar into a collapsed port BROADCASTS to all three lanes.
        let mut p = PatchBay::default();
        link_ports(&mut p, &st, PSrc::Chan(9, C::BlurSize), PDst::Chan(2, C::GainR));
        assert_eq!(p.links.len(), 3);
        assert!(p.links.iter().all(|l| l.from == PSrc::Chan(9, C::BlurSize)));

        // An rgb source into a scalar sink takes R — one link, no fan.
        let mut p = PatchBay::default();
        link_ports(&mut p, &st, PSrc::Chan(1, C::LiftR), PDst::Chan(9, C::BlurSize));
        assert_eq!(p.links.len(), 1);
        assert_eq!(p.links[0].from, PSrc::Chan(1, C::LiftR));

        // Expanded (no ports noted), the same drag is one plain link.
        let bare = PatchUi::default();
        let mut p = PatchBay::default();
        link_ports(&mut p, &bare, PSrc::Chan(1, C::LiftR), PDst::Chan(2, C::GainR));
        assert_eq!(p.links.len(), 1);
    }

    #[test]
    fn hidden_lanes_resolve_onto_their_port() {
        use crate::anim::Channel as C;
        let lanes = [C::GainR, C::GainG, C::GainB];
        let mut st = PatchUi::default();
        st.note_rgb_port(2, lanes);
        // G and B have no jack while collapsed; they ride the R-lane port,
        // which is how three links draw as one cable instead of three stubs.
        for (i, ch) in lanes.iter().enumerate() {
            assert_eq!(
                resolve(&st, SockEnd::In(PDst::Chan(2, *ch))),
                SockEnd::In(PDst::Chan(2, C::GainR))
            );
            assert_eq!(st.lane_of(2, *ch), Some(i));
        }
        // An ungrouped channel, and any channel on another node, resolve to
        // themselves.
        let other = SockEnd::In(PDst::Chan(2, C::BlurSize));
        assert_eq!(resolve(&st, other), other);
        let elsewhere = SockEnd::In(PDst::Chan(7, C::GainG));
        assert_eq!(resolve(&st, elsewhere), elsewhere);
    }

    #[test]
    fn splice_rewires_source_through_the_node() {
        let (mut g, m) = rig();
        g.patch
            .link(PSrc::Chan(1, crate::anim::Channel::ScaleX), PDst::Chan(2, crate::anim::Channel::GainR));
        let l = g.patch.links[0].clone();
        // What cables_ui applies on a splice drop.
        g.patch.link(l.from, PDst::Mod(m, 0));
        g.patch.link(PSrc::Mod(m), l.to);
        assert_eq!(g.patch.links.len(), 2);
        assert!(g.patch.link_to(PDst::Mod(m, 0)).is_some());
        let sink = g.patch.link_to(PDst::Chan(2, crate::anim::Channel::GainR)).unwrap();
        assert!(matches!(sink.from, PSrc::Mod(x) if x == m));
    }
}
