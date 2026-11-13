//! A small node graph INSIDE a node, revealed by zoom.
//!
//! Zoom in on a node that holds a graph of its own — a Tracker's points and
//! its Solve, a roto's shapes and the ops that combine them — and its glyph
//! gives way to that graph, drawn inside the node's outline and editable in
//! place with a SUBSET of the graph's gestures: drag a node to move it, drag
//! from an output to an input to wire it, drag a wired input off to unplug
//! or re-plug it, click to select, `+` to add, alt-click to remove. No
//! Tab, no marquee, no dots: when a graph needs the full editor it is a dive
//! (Noise, 2dini, tools), not this.
//!
//! The inner graph is DATA on its node. It never reaches the engine as
//! nodes; what it means is the node kind's business.
//!
//! This module knows nothing about any node kind. A kind joins by
//! implementing `Provider` — describe the graph (`mini`), and apply a move,
//! a wire, an unplug, an add and a remove to its own data (a remove usually
//! has to slide the later items' curves and cables down) — and adding one
//! arm to `provider()`. The zoom, the drawing, the hits, the gestures and
//! the selection are this module's; the editor only routes to it.

use crate::editor::EditorGraph;
use crate::theme;
use egui::{Color32, Pos2, Rect, Stroke, Vec2};
use engine::graph::NodeKind;

/// One node of an inner graph.
#[derive(Clone, Debug, PartialEq)]
pub struct MiniNode {
    /// The kind's own id for it (stable for as long as the kind says).
    pub id: usize,
    /// Where it sits, in the host node's [-1, 1] square.
    pub at: [f32; 2],
    pub label: String,
    /// Input ports, left to right along its top.
    pub inputs: usize,
    /// One output along its bottom, or none.
    pub output: bool,
    /// An ITEM is what the host's viewer and panel edit when it is
    /// selected (a track point); a non-item (a Solve) is selectable to
    /// look at but does not take that over.
    pub item: bool,
    pub removable: bool,
}

/// `from`'s output feeds `to`'s input `port`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MiniEdge {
    pub from: usize,
    pub to: usize,
    pub port: usize,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Mini {
    pub nodes: Vec<MiniNode>,
    pub edges: Vec<MiniEdge>,
}

impl Mini {
    pub fn node(&self, id: usize) -> Option<&MiniNode> {
        self.nodes.iter().find(|n| n.id == id)
    }

    pub fn edge_into(&self, to: usize, port: usize) -> Option<MiniEdge> {
        self.edges.iter().copied().find(|e| e.to == to && e.port == port)
    }
}

/// What a node kind with an inner graph provides.
pub trait Provider: Sync {
    /// The graph as it stands.
    fn mini(&self, kind: &NodeKind) -> Mini;
    /// Whether `+` is offered (a cap reached hides it).
    fn can_add(&self, kind: &NodeKind) -> bool;
    /// Add an item to `node`; its id.
    fn add(&self, graph: &mut EditorGraph, node: u64) -> Option<usize>;
    /// Remove node `id` from `node`, renumbering whatever is indexed by it.
    fn remove(&self, graph: &mut EditorGraph, node: u64, id: usize) -> bool;
    fn move_to(&self, kind: &mut NodeKind, id: usize, at: [f32; 2]);
    /// Wire `from` into `to`'s `port` (replacing what fed it). False when
    /// the kind refuses the pairing.
    fn connect(&self, kind: &mut NodeKind, from: usize, to: usize, port: usize) -> bool;
    fn disconnect(&self, kind: &mut NodeKind, to: usize, port: usize);
    /// The graph's held-key swipe (M / I / O) of node `dragged` onto node
    /// `under`: make whatever that means here and return its id. Kinds
    /// with no such thing keep the default, which refuses.
    fn swipe(
        &self,
        _kind: &mut NodeKind,
        _dragged: usize,
        _under: usize,
        _op: crate::editor::SwipeOp,
    ) -> Option<usize> {
        None
    }
}

/// The one registry: which kinds have an inner graph.
pub fn provider(kind: &NodeKind) -> Option<&'static dyn Provider> {
    match kind {
        NodeKind::Tracker(_) => Some(&crate::tracker::POINTS),
        NodeKind::Spline(_) => Some(&crate::rotoshapes::SHAPES),
        NodeKind::GenAi(_) => Some(&crate::genaiui::STEPS),
        _ => None,
    }
}

/// What a pointer is over inside a host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Hit {
    Node(usize),
    Out(usize),
    In(usize, usize),
    Add,
}

/// A gesture in progress inside a host.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Drag {
    /// Moving node `id`; `grab` is the press's offset from its centre, in
    /// the host's [-1, 1] units, so the node does not jump to the pointer.
    /// `from` is where it started (a swipe puts it back there); `merged`
    /// once a swipe has fired, so one gesture makes one node.
    Move { host: u64, id: usize, grab: [f32; 2], from: [f32; 2], merged: bool },
    /// A wire out of `from`'s output, following the pointer.
    Wire { host: u64, from: usize },
}

impl Drag {
    pub fn host(&self) -> u64 {
        match *self {
            Drag::Move { host, .. } | Drag::Wire { host, .. } => host,
        }
    }
}

/// Node radius (screen px) at which the inner graph starts to show, and
/// where it is fully in. Below the first the node is its glyph. A node is
/// 24 px at zoom 1, so this is "zoom right in": twice to three times that.
pub const SHOW_AT: f32 = 48.0;
pub const FULL_AT: f32 = 72.0;

/// 0 (glyph only) → 1 (inner graph only) as the node grows on screen.
pub fn reveal(r: f32) -> f32 {
    ((r - SHOW_AT) / (FULL_AT - SHOW_AT)).clamp(0.0, 1.0)
}

/// The inner graph is editable once it is mostly in.
pub fn live(r: f32) -> bool {
    reveal(r) >= 0.5
}

/// The square the inner graph is drawn in, for a host drawn at `center`
/// with radius `r`.
pub fn frame_for(center: Pos2, r: f32) -> Rect {
    Rect::from_center_size(center, Vec2::splat(r * 1.7))
}

fn to_screen(frame: Rect, at: [f32; 2]) -> Pos2 {
    frame.center() + Vec2::new(at[0], at[1]) * frame.width() * 0.5
}

fn to_local(frame: Rect, p: Pos2) -> [f32; 2] {
    let v = (p - frame.center()) / (frame.width() * 0.5).max(1e-3);
    [v.x, v.y]
}

/// Screen geometry of one mini node: its radius, input ports, output port.
struct Geo {
    c: Pos2,
    nr: f32,
    ins: Vec<Pos2>,
    out: Option<Pos2>,
}

fn geo(frame: Rect, n: &MiniNode) -> Geo {
    let c = to_screen(frame, n.at);
    let nr = frame.width() * 0.05;
    let span = nr * 0.9;
    let ins = (0..n.inputs)
        .map(|k| c + Vec2::new((k as f32 - (n.inputs as f32 - 1.0) * 0.5) * span, -nr * 1.3))
        .collect();
    let out = n.output.then(|| c + Vec2::new(0.0, nr * 1.3));
    Geo { c, nr, ins, out }
}

/// Where `+` sits: the host's bottom-left corner.
fn plus_at(frame: Rect) -> Pos2 {
    to_screen(frame, [-0.82, 0.82])
}

/// The item the host's viewer and panel edit (UI state, not saved),
/// clamped to `n` items.
pub fn selected(ctx: &egui::Context, node: u64, n: usize) -> usize {
    ctx.data(|d| d.get_temp::<usize>(item_id(node))).unwrap_or(0).min(n.saturating_sub(1))
}

/// The node last clicked (item or not), for the highlight.
pub fn picked(ctx: &egui::Context, node: u64) -> Option<usize> {
    ctx.data(|d| d.get_temp::<usize>(pick_id(node)))
}

pub fn select(ctx: &egui::Context, node: u64, i: usize) {
    ctx.data_mut(|d| {
        d.insert_temp(item_id(node), i);
        d.insert_temp(pick_id(node), i);
    });
}

/// Highlight node `id` without handing it the viewer and panel (a
/// non-item: a Solve, a roto op).
pub fn pick_node(ctx: &egui::Context, node: u64, id: usize) {
    ctx.data_mut(|d| {
        d.insert_temp(pick_id(node), id);
    });
}

fn pick(ctx: &egui::Context, host: u64, m: &Mini, id: usize) {
    match m.node(id) {
        Some(n) if n.item => select(ctx, host, id),
        Some(_) => ctx.data_mut(|d| {
            d.insert_temp(pick_id(host), id);
        }),
        None => {}
    }
}

fn item_id(node: u64) -> egui::Id {
    egui::Id::new(("innards-sel", node))
}

fn pick_id(node: u64) -> egui::Id {
    egui::Id::new(("innards-pick", node))
}

/// A wire from `a` (an output, below its node) down or up to `b` (an
/// input, above its node): the graph's vertical flow. `ortho`: the graph's
/// right-angles mode, sized to the window it is drawn in.
fn wire(painter: &egui::Painter, a: Pos2, b: Pos2, st: Stroke, ortho: Option<Rect>) {
    match ortho {
        Some(frame) => {
            painter.add(egui::Shape::line(mini_route(frame, a, b), st));
        }
        None => {
            painter.add(egui::epaint::CubicBezierShape::from_points_stroke(
                wire_ctrl(a, b),
                false,
                Color32::TRANSPARENT,
                st,
            ));
        }
    }
}

/// The right-angle route of an inner wire, its corners scaled to `frame`.
fn mini_route(frame: Rect, a: Pos2, b: Pos2) -> Vec<Pos2> {
    use crate::ortho::{route, End, Style};
    let w = frame.width();
    route(a, End::Down, b, End::Down, &Style { radius: w * 0.02, stub: w * 0.02, clear: w * 0.06, lane: 0.0 })
}

/// The control points of that wire — what is drawn is what a slice cuts.
fn wire_ctrl(a: Pos2, b: Pos2) -> [Pos2; 4] {
    let pull = ((b.y - a.y).abs() * 0.5).max(12.0);
    [a, a + Vec2::new(0.0, pull), b - Vec2::new(0.0, pull), b]
}

/// Draw the inner graph into `frame` at `alpha` (from `reveal`), the
/// wire being dragged (if it is this host's) to `pointer`. Returns the hit
/// rects, ports before bodies so a port wins where they overlap.
#[allow(clippy::too_many_arguments)]
pub fn paint(
    painter: &egui::Painter,
    frame: Rect,
    m: &Mini,
    can_add: bool,
    picked: Option<usize>,
    alpha: f32,
    fill: Color32,
    stroke: Stroke,
    dragging: Option<(usize, Pos2)>,
    ortho: bool,
) -> Vec<(Hit, Rect)> {
    let mut hits = Vec::new();
    if alpha <= 0.0 {
        return hits;
    }
    let fade = |c: Color32| c.gamma_multiply(alpha);
    // A window into the node: its own floor, so the host's through-wire
    // and glyph do not run across the inner graph.
    painter.rect(
        frame,
        frame.width() * 0.04,
        fade(theme::graph_bg()),
        Stroke::new(stroke.width.max(1.0), fade(stroke.color)),
        egui::StrokeKind::Inside,
    );
    let geos: Vec<(usize, Geo)> = m.nodes.iter().map(|n| (n.id, geo(frame, n))).collect();
    let g_of = |id: usize| geos.iter().find(|(i, _)| *i == id).map(|(_, g)| g);
    let wire_st = Stroke::new((frame.width() * 0.006).max(1.0), fade(theme::text()));
    for e in &m.edges {
        if let (Some(a), Some(b)) = (g_of(e.from).and_then(|g| g.out), g_of(e.to).and_then(|g| g.ins.get(e.port).copied())) {
            wire(painter, a, b, wire_st, ortho.then_some(frame));
        }
    }
    if let Some((from, p)) = dragging {
        if let Some(a) = g_of(from).and_then(|g| g.out) {
            wire(painter, a, p, Stroke::new(wire_st.width, fade(theme::text_hot())), ortho.then_some(frame));
        }
    }
    for n in &m.nodes {
        let Some(g) = g_of(n.id) else { continue };
        let pr = (g.nr * 0.28).max(2.0);
        for (k, p) in g.ins.iter().enumerate() {
            painter.circle(*p, pr, fade(theme::port_in()), Stroke::NONE);
            hits.push((Hit::In(n.id, k), Rect::from_center_size(*p, Vec2::splat(pr * 3.2))));
        }
        if let Some(p) = g.out {
            painter.circle(p, pr, fade(theme::port_in()), Stroke::NONE);
            hits.push((Hit::Out(n.id), Rect::from_center_size(p, Vec2::splat(pr * 3.2))));
        }
    }
    for n in &m.nodes {
        let Some(g) = g_of(n.id) else { continue };
        let body = if n.item { fill } else { theme::node_ink_light() };
        painter.circle(g.c, g.nr, fade(body), Stroke::new(stroke.width.min(g.nr * 0.2), fade(stroke.color)));
        if picked == Some(n.id) {
            painter.circle_stroke(g.c, g.nr * 1.25, Stroke::new((g.nr * 0.15).max(1.0), fade(theme::selection())));
        }
        if g.nr > 6.0 {
            painter.text(
                g.c,
                egui::Align2::CENTER_CENTER,
                &n.label,
                egui::FontId::proportional((g.nr * 0.8).min(14.0)),
                fade(theme::node_ink()),
            );
        }
        hits.push((Hit::Node(n.id), Rect::from_center_size(g.c, Vec2::splat(g.nr * 2.2))));
    }
    if can_add {
        let c = plus_at(frame);
        let rr = frame.width() * 0.045;
        let st = Stroke::new((rr * 0.22).max(1.0), fade(theme::node_ink_light()));
        painter.line_segment([c - Vec2::new(rr, 0.0), c + Vec2::new(rr, 0.0)], st);
        painter.line_segment([c - Vec2::new(0.0, rr), c + Vec2::new(0.0, rr)], st);
        hits.push((Hit::Add, Rect::from_center_size(c, Vec2::splat(rr * 2.6))));
    }
    hits
}

fn hit_at(hits: &[(Hit, Rect)], p: Pos2) -> Option<Hit> {
    hits.iter().find(|(_, r)| r.contains(p)).map(|(h, _)| *h)
}

/// A CLICK (press and release in place) at `p`: select a node, `+` adds,
/// alt-click removes. True when it hit something (the click is consumed).
pub fn click(
    ctx: &egui::Context,
    graph: &mut EditorGraph,
    host: u64,
    hits: &[(Hit, Rect)],
    p: Pos2,
    alt: bool,
) -> bool {
    let Some(hit) = hit_at(hits, p) else { return false };
    let Some(prov) = graph.node(host).and_then(|n| provider(&n.kind)) else { return false };
    let m = graph.node(host).map(|n| prov.mini(&n.kind)).unwrap_or_default();
    match hit {
        Hit::Add => {
            if let Some(i) = prov.add(graph, host) {
                select(ctx, host, i);
            }
        }
        Hit::Node(id) | Hit::Out(id) | Hit::In(id, _) if alt => {
            if m.node(id).is_some_and(|n| n.removable) {
                prov.remove(graph, host, id);
            }
        }
        Hit::Node(id) | Hit::Out(id) | Hit::In(id, _) => pick(ctx, host, &m, id),
    }
    true
}

/// A drag starting at `p` inside `host`: which gesture it is. A wired
/// input is UNPLUGGED at the press and its wire follows the pointer from
/// its source, the graph's own re-plug gesture.
pub fn begin(
    ctx: &egui::Context,
    graph: &mut EditorGraph,
    host: u64,
    frame: Rect,
    hits: &[(Hit, Rect)],
    p: Pos2,
) -> Option<Drag> {
    let hit = hit_at(hits, p)?;
    let prov = graph.node(host).and_then(|n| provider(&n.kind))?;
    let m = prov.mini(&graph.node(host)?.kind);
    match hit {
        Hit::Out(id) => Some(Drag::Wire { host, from: id }),
        Hit::In(id, port) => {
            let e = m.edge_into(id, port)?;
            prov.disconnect(&mut graph.node_mut(host)?.kind, id, port);
            Some(Drag::Wire { host, from: e.from })
        }
        Hit::Node(id) => {
            let at = m.node(id)?.at;
            let l = to_local(frame, p);
            pick(ctx, host, &m, id);
            Some(Drag::Move { host, id, grab: [l[0] - at[0], l[1] - at[1]], from: at, merged: false })
        }
        Hit::Add => None,
    }
}

/// Follow the pointer: a moved node lands under it, kept inside the host.
pub fn update(graph: &mut EditorGraph, drag: &Drag, frame: Rect, p: Pos2) {
    if let Drag::Move { host, id, grab, .. } = *drag {
        let Some(prov) = graph.node(host).and_then(|n| provider(&n.kind)) else { return };
        let l = to_local(frame, p);
        let at = [(l[0] - grab[0]).clamp(-0.92, 0.92), (l[1] - grab[1]).clamp(-0.92, 0.92)];
        if let Some(n) = graph.node_mut(host) {
            prov.move_to(&mut n.kind, id, at);
        }
    }
}

/// A held-key SWIPE while moving a node: when the pointer is over another
/// node, the kind makes its op of the two (`Provider::swipe`), the dragged
/// node glides back to where it started, and the drag carries on with the
/// new node — the graph's swipe, in miniature. `None`: nothing happened.
pub fn swipe(
    graph: &mut EditorGraph,
    drag: &Drag,
    frame: Rect,
    p: Pos2,
    op: crate::editor::SwipeOp,
) -> Option<Drag> {
    let Drag::Move { host, id, from, merged, .. } = *drag else { return None };
    if merged {
        return None;
    }
    let prov = graph.node(host).and_then(|n| provider(&n.kind))?;
    let m = prov.mini(&graph.node(host)?.kind);
    let under = m
        .nodes
        .iter()
        .filter(|n| n.id != id)
        .find(|n| {
            let g = geo(frame, n);
            (g.c - p).length() <= g.nr * 1.1
        })?
        .id;
    let kind = &mut graph.node_mut(host)?.kind;
    let made = prov.swipe(kind, id, under, op)?;
    prov.move_to(kind, id, from);
    let at = prov.mini(kind).node(made).map(|n| n.at).unwrap_or(from);
    let l = to_local(frame, p);
    Some(Drag::Move { host, id: made, grab: [l[0] - at[0], l[1] - at[1]], from: at, merged: true })
}

/// The graph's ctrl-slice, inside every LIVE inner graph it crosses: each
/// wire the segment `a`→`b` (screen) cuts is unplugged. `hosts` is the
/// editor's last-frame (host, frame, hits, live) list.
pub fn slice(
    graph: &mut EditorGraph,
    hosts: &[(u64, Rect, Vec<(Hit, Rect)>, bool)],
    a: Pos2,
    b: Pos2,
    ortho: bool,
) {
    for (host, frame, _, live) in hosts {
        if !*live {
            continue;
        }
        let Some(prov) = graph.node(*host).and_then(|n| provider(&n.kind)) else { continue };
        let Some(m) = graph.node(*host).map(|n| prov.mini(&n.kind)) else { continue };
        let geo_of = |id: usize| m.node(id).map(|n| geo(*frame, n));
        let mut cut: Vec<(usize, usize)> = m
            .edges
            .iter()
            .filter(|e| {
                let (Some(from), Some(to)) = (geo_of(e.from), geo_of(e.to)) else { return false };
                let (Some(p0), Some(p3)) = (from.out, to.ins.get(e.port).copied()) else { return false };
                if ortho {
                    crate::ortho::hits_segment(&mini_route(*frame, p0, p3), a, b)
                } else {
                    crate::editor::bezier_hits_segment(&wire_ctrl(p0, p3), a, b)
                }
            })
            .map(|e| (e.to, e.port))
            .collect();
        // Highest port first: unplugging a port can renumber the ones after
        // it (Out's list closes up).
        cut.sort_by(|x, y| y.1.cmp(&x.1));
        if let Some(n) = graph.node_mut(*host) {
            for (to, port) in cut {
                prov.disconnect(&mut n.kind, to, port);
            }
        }
    }
}

/// Delete over a live inner graph: remove the node picked in it, if it
/// can go. True when it did — the host itself is then left alone.
pub fn delete_picked(
    ctx: &egui::Context,
    graph: &mut EditorGraph,
    hosts: &[(u64, Rect, Vec<(Hit, Rect)>, bool)],
    p: Pos2,
) -> bool {
    let Some(host) = hosts.iter().find(|(_, f, _, live)| *live && f.contains(p)).map(|(h, ..)| *h) else {
        return false;
    };
    let Some(prov) = graph.node(host).and_then(|n| provider(&n.kind)) else { return false };
    let Some(id) = picked(ctx, host) else { return false };
    let removable = graph.node(host).is_some_and(|n| prov.mini(&n.kind).node(id).is_some_and(|m| m.removable));
    if !removable || !prov.remove(graph, host, id) {
        return false;
    }
    // Nothing is picked now; the item selection falls to a neighbour.
    let n_items = graph.node(host).map(|n| prov.mini(&n.kind).nodes.iter().filter(|m| m.item).count()).unwrap_or(0);
    select(ctx, host, selected(ctx, host, n_items));
    true
}

/// The drag ends at `p`: a wire lands on the input under it, or is let go.
pub fn end(graph: &mut EditorGraph, drag: &Drag, hits: &[(Hit, Rect)], p: Pos2) {
    if let Drag::Wire { host, from } = *drag {
        if let Some(Hit::In(to, port)) = hit_at(hits, p) {
            if let Some(n) = graph.node_mut(host) {
                if let Some(prov) = provider(&n.kind) {
                    prov.connect(&mut n.kind, from, to, port);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn host() -> (EditorGraph, u64) {
        let mut g = EditorGraph::default();
        let id = g.add(NodeKind::Tracker(Default::default()), Pos2::ZERO);
        (g, id)
    }

    fn frame() -> Rect {
        Rect::from_center_size(Pos2::new(200.0, 200.0), Vec2::splat(300.0))
    }

    fn hits(g: &EditorGraph, id: u64) -> Vec<(Hit, Rect)> {
        let n = g.node(id).unwrap();
        let p = provider(&n.kind).unwrap();
        // Inside a frame: text needs the fonts a pass loads.
        let ctx = egui::Context::default();
        let mut out = Vec::new();
        let _ = ctx.run_ui(Default::default(), |ui| {
            let painter = ui.painter().clone();
            out = paint(&painter, frame(), &p.mini(&n.kind), p.can_add(&n.kind), None, 1.0, Color32::GRAY, Stroke::NONE, None, false);
        });
        out
    }

    fn centre(h: &[(Hit, Rect)], want: Hit) -> Pos2 {
        h.iter().find(|(k, _)| *k == want).unwrap_or_else(|| panic!("no {want:?} in {h:?}")).1.center()
    }

    fn solve_in(g: &EditorGraph, id: u64) -> Vec<usize> {
        match &g.node(id).unwrap().kind {
            NodeKind::Tracker(t) => t.solve_inputs(),
            _ => vec![],
        }
    }

    #[test]
    fn reveal_ramps_from_glyph_to_graph() {
        assert_eq!(reveal(24.0), 0.0, "zoom 1 is the glyph");
        assert_eq!(reveal(SHOW_AT), 0.0);
        assert_eq!(reveal(FULL_AT), 1.0);
        assert!(!live(SHOW_AT + 1.0) && live(FULL_AT));
    }

    #[test]
    fn a_tracker_is_registered_and_a_grade_is_not() {
        let t = NodeKind::Tracker(Default::default());
        let m = provider(&t).expect("Tracker has an inner graph").mini(&t);
        assert_eq!(m.nodes.len(), 2, "one point and the Solve");
        assert_eq!(m.edges.len(), 1, "the point feeds the solve");
        assert!(provider(&NodeKind::Grade(Default::default())).is_none());
    }

    #[test]
    fn clicks_select_add_and_remove() {
        let ctx = egui::Context::default();
        let (mut g, id) = host();
        for want in [2, 3] {
            let h = hits(&g, id);
            assert!(click(&ctx, &mut g, id, &h, centre(&h, Hit::Add), false));
            assert_eq!(solve_in(&g, id).len(), want, "a new point is wired in");
            assert_eq!(selected(&ctx, id, want), want - 1);
        }
        let h = hits(&g, id);
        assert!(click(&ctx, &mut g, id, &h, centre(&h, Hit::Node(0)), false));
        assert_eq!(selected(&ctx, id, 3), 0);
        let h = hits(&g, id);
        assert!(click(&ctx, &mut g, id, &h, centre(&h, Hit::Node(1)), true));
        assert_eq!(solve_in(&g, id), vec![0, 1], "removed, and the later point slid down");
        let h = hits(&g, id);
        assert!(!click(&ctx, &mut g, id, &h, Pos2::new(-500.0, -500.0), false));
    }

    #[test]
    fn a_node_moves_with_the_pointer_without_jumping() {
        let ctx = egui::Context::default();
        let (mut g, id) = host();
        let h = hits(&g, id);
        let press = centre(&h, Hit::Node(0)) + Vec2::new(3.0, 2.0);
        let d = begin(&ctx, &mut g, id, frame(), &h, press).expect("a press on a node moves it");
        let before = provider(&g.node(id).unwrap().kind).unwrap().mini(&g.node(id).unwrap().kind).nodes[0].at;
        update(&mut g, &d, frame(), press + Vec2::new(30.0, -15.0));
        let m = provider(&g.node(id).unwrap().kind).unwrap().mini(&g.node(id).unwrap().kind);
        let after = m.nodes[0].at;
        // 30 px on a 300 px frame is 0.2 of the [-1, 1] square.
        assert!((after[0] - before[0] - 0.2).abs() < 1e-4 && (after[1] - before[1] + 0.1).abs() < 1e-4, "{before:?} -> {after:?}");
    }

    #[test]
    fn wires_unplug_and_replug() {
        let ctx = egui::Context::default();
        let (mut g, id) = host();
        let h = hits(&g, id);
        click(&ctx, &mut g, id, &h, centre(&h, Hit::Add), false);
        assert_eq!(solve_in(&g, id), vec![0, 1]);
        let solve = crate::tracker::SOLVE;
        // Drag point 1's wire off Solve's port 1 and let go on nothing.
        let h = hits(&g, id);
        let d = begin(&ctx, &mut g, id, frame(), &h, centre(&h, Hit::In(solve, 1))).expect("a wired input unplugs");
        assert_eq!(d, Drag::Wire { host: id, from: 1 });
        end(&mut g, &d, &h, Pos2::new(-500.0, 0.0));
        assert_eq!(solve_in(&g, id), vec![0], "let go on nothing: unplugged");
        // Out of point 1 onto Solve's port 0: it takes that port.
        let h = hits(&g, id);
        let d = begin(&ctx, &mut g, id, frame(), &h, centre(&h, Hit::Out(1))).unwrap();
        end(&mut g, &d, &h, centre(&h, Hit::In(solve, 0)));
        assert_eq!(solve_in(&g, id), vec![1]);
        // An unwired input starts nothing.
        let h = hits(&g, id);
        let free = h.iter().filter_map(|(k, _)| match k { Hit::In(n, p) if *n == solve => Some(*p), _ => None }).max().unwrap();
        assert!(begin(&ctx, &mut g, id, frame(), &h, centre(&h, Hit::In(solve, free))).is_none());
    }
}

/// For LOOKING: a four-point Tracker at three zooms through the real
/// editor — glyph, half in, fully in. `cargo test -p app
/// innards_render -- --ignored` writes `target/innards-*.png`.
#[cfg(test)]
mod render {
    use super::*;
    use crate::editor::NodeEditor;

    #[test]
    #[ignore]
    fn innards_render() {
        for (name, z) in [("zoom1", 1.0f32), ("half", 2.5), ("in", 6.0)] {
            let mut g = EditorGraph::default();
            g.nodes.clear();
            g.edges.clear();
            let id = g.add(NodeKind::Tracker(Default::default()), Pos2::ZERO);
            for _ in 0..3 {
                crate::tracker::POINTS.add(&mut g, id);
            }
            // One helper point left out of the solve.
            if let Some(n) = g.node_mut(id) {
                crate::tracker::POINTS.disconnect(&mut n.kind, crate::tracker::SOLVE, 3);
            }
            let mut ed = NodeEditor::default();
            ed.zoom = z;
            ed.pan = Vec2::ZERO;
            let (mut sel, mut act, mut pin, mut slot, mut rects) =
                (vec![id], Vec::new(), Vec::new(), 2usize, Vec::new());
            crate::uishot::render(&format!("innards-{name}"), 700.0, 600.0, 2.0, 4, |ui, _| {
                ed.show(ui, &mut g, &mut sel, &mut act, &mut pin, &mut slot, &mut rects);
            });
        }
    }
}
