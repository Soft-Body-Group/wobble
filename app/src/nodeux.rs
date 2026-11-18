//! Node-graph UX: the parts of the graph editor that are ALGORITHM rather
//! than gesture — group backdrops, the align swipe, and the auto-layout.
//! Kept out of `editor.rs` because none of it needs an egui context, which
//! is what makes it testable: every function here is a pure edit of an
//! [`EditorGraph`], and the tests at the bottom drive them with no window.

use crate::editor::{EditorGraph, NODE_R};
use egui::{Pos2, Rect, Vec2};

/// A backdrop that SHRINK-WRAPS its members: a Blender-frame-style group
/// drawn under the nodes as one organic blob — the union of a padded disc
/// per member and a sleeve along the wires between them — so an L-shaped
/// selection gets an L-shaped backdrop, not a rectangle over the notch.
/// Membership is EXPLICIT (a node dragged away stretches the blob with it,
/// exactly as a Blender frame follows its children); it changes only by
/// ⌘G, dropping a node inside the blob, or the context menus. It is still
/// decoration: no pixels change, nothing enters the render-cache key.
#[derive(Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Group {
    pub id: u64,
    #[serde(default)]
    pub title: String,
    /// The nodes this backdrop wraps.
    #[serde(default)]
    pub members: Vec<u64>,
    /// The FIRST version of this feature was a rectangle that owned nothing
    /// and asked geometry who was inside. A setup saved then carries this;
    /// `EditorGraph::migrate_groups` adopts whoever stood on it into
    /// `members` once, and it is never written again.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rect: Option<Rect>,
    /// Metaball buffer: how far the blob stands off each member's disc, in
    /// graph units. The wire sleeves, tethers and melt radius all follow
    /// it, so one knob reshapes the whole wrap.
    #[serde(default = "default_pad")]
    pub pad: f32,
    /// Title-tab height in graph units — the label's size knob.
    #[serde(default = "default_label_h")]
    pub label_h: f32,
    /// Backdrop tint. Stored rather than derived so a comp can colour-code
    /// its own way.
    #[serde(default = "default_group_color")]
    pub color: [u8; 3],
}

fn default_group_color() -> [u8; 3] {
    [0x3a, 0x46, 0x58]
}

fn default_pad() -> f32 {
    BLOB_PAD
}

fn default_label_h() -> f32 {
    TITLE_H
}

/// Height of a group's title tab, in graph units. The tab is the handle:
/// dragging it moves every member; the blob's body is inert so a marquee
/// still works straight over a backdrop.
pub const TITLE_H: f32 = 26.0;

/// The palette new groups cycle through, so two backdrops made in a row do
/// not read as one.
pub const GROUP_COLORS: [[u8; 3]; 6] = [
    [0x3a, 0x46, 0x58],
    [0x4a, 0x3f, 0x58],
    [0x3e, 0x54, 0x48],
    [0x58, 0x48, 0x3a],
    [0x3a, 0x52, 0x58],
    [0x54, 0x3c, 0x42],
];

impl Group {
    /// Members that still exist, in a stable order.
    pub fn live_members(&self, graph: &EditorGraph) -> Vec<u64> {
        self.members.iter().copied().filter(|m| graph.nodes.contains_key(m)).collect()
    }
}

impl EditorGraph {
    /// Which backdrop a node belongs to, if any. A node is in at most ONE
    /// group — joining a new one leaves the old — so the answer is a group
    /// id, not a list.
    pub fn group_of(&self, node: u64) -> Option<u64> {
        self.groups.iter().find(|g| g.members.contains(&node)).map(|g| g.id)
    }

    /// Move `ids` into group `gid`, out of whichever groups held them.
    /// Emptied groups are dropped — a backdrop with nothing under it is
    /// litter, not state.
    pub fn adopt_into_group(&mut self, gid: u64, ids: &[u64]) {
        for g in &mut self.groups {
            if g.id != gid {
                g.members.retain(|m| !ids.contains(m));
            }
        }
        if let Some(g) = self.groups.iter_mut().find(|g| g.id == gid) {
            for id in ids {
                if self.nodes.contains_key(id) && !g.members.contains(id) {
                    g.members.push(*id);
                }
            }
        }
        self.prune_groups();
    }

    /// Drop dead member ids and empty groups. Called after node removals
    /// and membership edits.
    pub fn prune_groups(&mut self) {
        let alive: std::collections::BTreeSet<u64> = self.nodes.keys().copied().collect();
        for g in &mut self.groups {
            g.members.retain(|m| alive.contains(m));
        }
        self.groups.retain(|g| !g.members.is_empty() || g.rect.is_some());
    }

    /// Adopt v1 rectangular backdrops: whoever stood on the rectangle when
    /// this first runs becomes the membership, and the rectangle is gone.
    pub fn migrate_groups(&mut self) {
        let takes: Vec<(u64, Vec<u64>)> = self
            .groups
            .iter()
            .filter(|g| g.members.is_empty() && g.rect.is_some())
            .map(|g| {
                let r = g.rect.unwrap();
                (
                    g.id,
                    self.nodes
                        .iter()
                        .filter(|(_, n)| r.contains(n.pos))
                        .map(|(id, _)| *id)
                        .collect(),
                )
            })
            .collect();
        for g in &mut self.groups {
            if let Some((_, m)) = takes.iter().find(|(id, _)| *id == g.id) {
                g.members = m.clone();
            }
            g.rect = None;
        }
        self.prune_groups();
    }
}

/// A fresh group over `ids`.
pub fn group_over(ids: &[u64], id: u64) -> Group {
    let mut members: Vec<u64> = ids.to_vec();
    members.sort_unstable();
    members.dedup();
    Group {
        id,
        title: String::new(),
        members,
        rect: None,
        pad: BLOB_PAD,
        label_h: TITLE_H,
        color: GROUP_COLORS[(id as usize) % GROUP_COLORS.len()],
    }
}

// ---------------------------------------------------------------------------
// The blob: the shape a group draws as
// ---------------------------------------------------------------------------

/// Default metaball buffer around a member's disc, graph units at node
/// size 1. Per-group now (`Group::pad`); this is what a new group gets.
pub const BLOB_PAD: f32 = 30.0;

/// The derived radii: sleeve around a member-to-member wire, tether for
/// the spanning-tree links that keep an unwired selection ONE blob, and
/// the smooth-union melt. All follow the group's buffer, floored so a
/// tight wrap never disconnects into islands.
fn sleeve_r(pad: f32) -> f32 {
    (pad * 0.5).clamp(6.0, 44.0)
}
fn tether_r(pad: f32) -> f32 {
    (pad * 0.34).clamp(5.0, 32.0)
}
fn melt_k(pad: f32) -> f32 {
    (pad * 0.75).clamp(8.0, 44.0)
}

/// The pieces a group's field is built from: a padded disc per member and
/// a capsule per member-to-member wire plus a spanning tree, so the blob is
/// always one connected shape.
pub struct BlobParts {
    pub discs: Vec<(Pos2, f32)>,
    pub caps: Vec<(Pos2, Pos2, f32)>,
    /// Smooth-union melt radius, derived from the group's buffer.
    pub k: f32,
}

pub fn blob_parts(graph: &EditorGraph, members: &[u64], pad: f32) -> BlobParts {
    let live: Vec<(u64, Pos2, f32)> = members
        .iter()
        .filter_map(|m| graph.node(*m).map(|n| (*m, n.pos, n.size)))
        .collect();
    let pad = pad.clamp(0.0, 400.0);
    let discs: Vec<(Pos2, f32)> =
        live.iter().map(|(_, p, s)| (*p, NODE_R * s + pad)).collect();
    let mut caps: Vec<(Pos2, Pos2, f32)> = Vec::new();
    let pos_of = |id: u64| live.iter().find(|(m, ..)| *m == id).map(|(_, p, _)| *p);
    for (d, _, s) in &graph.edges {
        if let (Some(a), Some(b)) = (pos_of(*d), pos_of(*s)) {
            caps.push((a, b, sleeve_r(pad)));
        }
    }
    // Prim's spanning tree over the members: cheap at group sizes, and any
    // tether that runs where discs already overlap simply disappears into
    // the union.
    if live.len() >= 2 {
        let mut joined = vec![false; live.len()];
        joined[0] = true;
        for _ in 1..live.len() {
            let mut best: Option<(f32, usize, usize)> = None;
            for (i, a) in live.iter().enumerate().filter(|(i, _)| joined[*i]) {
                for (j, b) in live.iter().enumerate().filter(|(j, _)| !joined[*j]) {
                    let d = a.1.distance(b.1);
                    if best.is_none_or(|(bd, ..)| d < bd) {
                        best = Some((d, i, j));
                    }
                }
            }
            let Some((_, i, j)) = best else { break };
            joined[j] = true;
            caps.push((live[i].1, live[j].1, tether_r(pad)));
        }
    }
    BlobParts { discs, caps, k: melt_k(pad) }
}

fn smin(a: f32, b: f32, k: f32) -> f32 {
    let h = (0.5 + 0.5 * (b - a) / k).clamp(0.0, 1.0);
    b * (1.0 - h) + a * h - k * h * (1.0 - h)
}

fn seg_dist(p: Pos2, a: Pos2, b: Pos2) -> f32 {
    let ab = b - a;
    let len2 = ab.length_sq();
    let t = if len2 <= 1e-6 { 0.0 } else { ((p - a).dot(ab) / len2).clamp(0.0, 1.0) };
    (a + ab * t - p).length()
}

/// Signed distance to the group's shape: negative inside. This is the ONE
/// definition of "inside a backdrop" — the drawing, the drop-to-join test
/// and the unit tests all ask it.
pub fn blob_sdf(parts: &BlobParts, p: Pos2) -> f32 {
    let mut d = f32::MAX;
    for (c, r) in &parts.discs {
        d = smin(d, c.distance(p) - r, parts.k);
    }
    for (a, b, r) in &parts.caps {
        d = smin(d, seg_dist(p, *a, *b) - r, parts.k);
    }
    d
}

/// A group's drawable shape, all in GRAPH space: a triangle soup for the
/// fill (concave-safe — egui only fills convex polygons), closed outline
/// loops, and the handle anchors.
pub struct Blob {
    pub tris: Vec<[Pos2; 3]>,
    pub loops: Vec<Vec<Pos2>>,
    pub bbox: Rect,
    /// Bottom-centre of the title tab: above the topmost member, so the
    /// handle stays attached to the shape however it bends.
    pub tab_anchor: Pos2,
    /// Bottom-right-most outline point — the resize grip, ON the shape
    /// rather than floating at a bounding-box corner the blob never fills.
    pub grip: Pos2,
}

/// Contour + fill of `blob_sdf`'s zero level set by marching squares over
/// a grid of `cell` graph units (finer when zoomed in; the caller caches
/// the result keyed on `blob_key`). Cells are tiny, so each cell's inside
/// region is filled as its own convex polygon — no overlap, so the
/// translucent fill stays uniform.
pub fn build_blob(
    graph: &EditorGraph,
    members: &[u64],
    cell: f32,
    pad: f32,
) -> Option<Blob> {
    let parts = blob_parts(graph, members, pad);
    if parts.discs.is_empty() {
        return None;
    }
    let mut bbox = Rect::NOTHING;
    for (c, r) in &parts.discs {
        bbox = bbox.union(Rect::from_center_size(*c, Vec2::splat((r + parts.k) * 2.0)));
    }
    for (a, b, r) in &parts.caps {
        let e = Rect::from_two_pos(*a, *b).expand(r + parts.k);
        bbox = bbox.union(e);
    }
    // Cap the grid so a comp-wide group cannot ask for a million samples.
    let cell = cell.max(bbox.width() / 160.0).max(bbox.height() / 160.0).max(2.0);
    let (nx, ny) = (
        (bbox.width() / cell).ceil() as usize + 1,
        (bbox.height() / cell).ceil() as usize + 1,
    );
    let at = |i: usize, j: usize| {
        Pos2::new(bbox.min.x + i as f32 * cell, bbox.min.y + j as f32 * cell)
    };
    let mut field = vec![0.0f32; (nx + 1) * (ny + 1)];
    for j in 0..=ny {
        for i in 0..=nx {
            field[j * (nx + 1) + i] = blob_sdf(&parts, at(i, j));
        }
    }
    let f = |i: usize, j: usize| field[j * (nx + 1) + i];

    let mut tris: Vec<[Pos2; 3]> = Vec::new();
    let mut segs: Vec<(Pos2, Pos2)> = Vec::new();
    for j in 0..ny {
        for i in 0..nx {
            // Cell corners in perimeter order, with their field values.
            let corners = [
                (at(i, j), f(i, j)),
                (at(i + 1, j), f(i + 1, j)),
                (at(i + 1, j + 1), f(i + 1, j + 1)),
                (at(i, j + 1), f(i, j + 1)),
            ];
            // Walk the perimeter: inside corners join the polygon, and a
            // sign change on an edge contributes the interpolated crossing.
            // Consecutive crossing-crossing pairs in that walk ARE the
            // contour through this cell.
            let mut poly: Vec<(Pos2, bool)> = Vec::new(); // (point, is_crossing)
            for k in 0..4 {
                let (pa, va) = corners[k];
                let (pb, vb) = corners[(k + 1) % 4];
                if va < 0.0 {
                    poly.push((pa, false));
                }
                if (va < 0.0) != (vb < 0.0) {
                    let t = va / (va - vb);
                    poly.push((pa + (pb - pa) * t, true));
                }
            }
            if poly.len() >= 3 {
                for k in 1..poly.len() - 1 {
                    tris.push([poly[0].0, poly[k].0, poly[k + 1].0]);
                }
            }
            for k in 0..poly.len() {
                let (a, ac) = poly[k];
                let (b, bc) = poly[(k + 1) % poly.len()];
                if ac && bc {
                    segs.push((a, b));
                }
            }
        }
    }
    if tris.is_empty() {
        return None;
    }

    // Chain the segment soup into closed loops (endpoints land exactly on
    // shared cell edges, so quantized keys match without epsilon hunting).
    let key = |p: Pos2| ((p.x * 32.0).round() as i64, (p.y * 32.0).round() as i64);
    let mut adj: std::collections::HashMap<(i64, i64), Vec<usize>> = Default::default();
    for (idx, (a, b)) in segs.iter().enumerate() {
        adj.entry(key(*a)).or_default().push(idx);
        adj.entry(key(*b)).or_default().push(idx);
    }
    let mut used = vec![false; segs.len()];
    let mut loops: Vec<Vec<Pos2>> = Vec::new();
    for start in 0..segs.len() {
        if used[start] {
            continue;
        }
        used[start] = true;
        let mut path = vec![segs[start].0, segs[start].1];
        loop {
            let end = *path.last().unwrap();
            let Some(cands) = adj.get(&key(end)) else { break };
            let Some(&nxt) = cands.iter().find(|c| !used[**c]) else { break };
            used[nxt] = true;
            let (a, b) = segs[nxt];
            path.push(if key(a) == key(end) { b } else { a });
        }
        if path.len() >= 3 {
            loops.push(path);
        }
    }

    // Handles. Tab above the topmost member disc; grip at the outline
    // point furthest down-right.
    let top = parts
        .discs
        .iter()
        .min_by(|a, b| (a.0.y - a.1).total_cmp(&(b.0.y - b.1)))
        .unwrap();
    let tab_anchor = Pos2::new(top.0.x, top.0.y - top.1 + 4.0);
    let grip = loops
        .iter()
        .flatten()
        .copied()
        .max_by(|a, b| (a.x + a.y).total_cmp(&(b.x + b.y)))
        .unwrap_or(bbox.max);
    Some(Blob { tris, loops, bbox, tab_anchor, grip })
}

/// Cache key for a group's blob: members, their positions and sizes, the
/// wires between them, and the sampling density. Anything that moves the
/// shape moves the key.
pub fn blob_key(graph: &EditorGraph, members: &[u64], cell_bucket: u32, pad: f32) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    cell_bucket.hash(&mut h);
    pad.to_bits().hash(&mut h);
    for m in members {
        m.hash(&mut h);
        if let Some(n) = graph.node(*m) {
            n.pos.x.to_bits().hash(&mut h);
            n.pos.y.to_bits().hash(&mut h);
            n.size.to_bits().hash(&mut h);
        }
    }
    for (d, i, s) in &graph.edges {
        if members.contains(d) && members.contains(s) {
            (d, i, s).hash(&mut h);
        }
    }
    h.finish()
}

// ---------------------------------------------------------------------------
// Align swipe
// ---------------------------------------------------------------------------

/// Which way a swipe ran. The gesture's own direction picks the axis: a
/// sideways stroke makes a row, a downward one makes a column.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Axis {
    Horizontal,
    Vertical,
}

impl Axis {
    pub fn of_swipe(a: Pos2, b: Pos2) -> Axis {
        if (b.x - a.x).abs() >= (b.y - a.y).abs() {
            Axis::Horizontal
        } else {
            Axis::Vertical
        }
    }
}

/// Houdini-style align: the nodes land ON the line that was drawn, in the
/// order they already sit in along it, evenly spaced.
///
/// The line's OTHER coordinate comes from the swipe, not from the nodes'
/// average — the gesture is direct manipulation, so where you drew is where
/// they go. Spacing spans whichever is longer, the stroke or the nodes' own
/// extent, so a flick never crams a dozen nodes into fifty pixels.
pub fn align_to_swipe(graph: &mut EditorGraph, ids: &[u64], a: Pos2, b: Pos2) -> bool {
    let axis = Axis::of_swipe(a, b);
    let mut live: Vec<u64> = ids.iter().copied().filter(|i| graph.node(*i).is_some()).collect();
    if live.len() < 2 {
        return false;
    }
    let along = |p: Pos2| if axis == Axis::Horizontal { p.x } else { p.y };
    live.sort_by(|x, y| {
        let (px, py) = (graph.nodes[x].pos, graph.nodes[y].pos);
        along(px).partial_cmp(&along(py)).unwrap_or(std::cmp::Ordering::Equal).then(x.cmp(y))
    });
    // Footprint along the axis decides the minimum spacing: uniform gaps
    // between EDGES, not between centres, so a mixed-size row still reads.
    let extent: Vec<f32> =
        live.iter().map(|i| 2.0 * NODE_R * graph.nodes[i].size).collect();
    let gap = if axis == Axis::Horizontal { COL_GAP } else { ROW_GAP };
    let need: f32 =
        extent.iter().sum::<f32>() + gap * (live.len() - 1) as f32;
    let (n_lo, n_hi) = live.iter().fold((f32::MAX, f32::MIN), |(lo, hi), i| {
        let v = along(graph.nodes[i].pos);
        (lo.min(v), hi.max(v))
    });
    let (s_lo, s_hi) = (along(a).min(along(b)), along(a).max(along(b)));
    let span = (n_hi - n_lo).max(s_hi - s_lo).max(need);
    let start = ((n_lo + n_hi) * 0.5).min((s_lo + s_hi) * 0.5) - span * 0.5;
    // Even CENTRE spacing across the span reads as evenly spaced when the
    // nodes are the same size, and as balanced when they are not.
    let step = if live.len() > 1 { span / (live.len() - 1) as f32 } else { 0.0 };
    let cross = if axis == Axis::Horizontal { (a.y + b.y) * 0.5 } else { (a.x + b.x) * 0.5 };
    for (k, id) in live.iter().enumerate() {
        let v = start + step * k as f32;
        if let Some(n) = graph.node_mut(*id) {
            n.pos = match axis {
                Axis::Horizontal => Pos2::new(v, cross),
                Axis::Vertical => Pos2::new(cross, v),
            };
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Auto-layout
// ---------------------------------------------------------------------------

/// Horizontal clearance between node EDGES. Wide enough for the label that
/// rides each node's right shoulder.
pub const COL_GAP: f32 = 78.0;
/// Vertical clearance between node edges — has to clear two stalks
/// (`SOCKET_OFFSET` each) plus the wire's bow.
pub const ROW_GAP: f32 = 64.0;
/// The same clearance with right-angle noodles: every wire that jogs
/// sideways between two rows needs a straight across-run there, and runs
/// from different outputs sit in lanes of their own (`ortho::lanes`), so
/// the gap is roomier to start with and grows by `ORTHO_LANE_ROOM` per
/// jogging output past the first, up to `ORTHO_LANES_MAX` of them.
pub const ORTHO_ROW_GAP: f32 = 120.0;
const ORTHO_LANE_ROOM: f32 = 12.0;
const ORTHO_LANES_MAX: usize = 8;
/// Footprint of a routing channel (a virtual vertex that may become a
/// dot), and the clearance beside one. Narrow: channels pack tight.
const CHAN_W: f32 = 26.0;
const CHAN_GAP: f32 = 30.0;
/// A routed wire keeps a dot only where it actually TURNS: chain points
/// within this many units of the straight line through their neighbours
/// are dropped before anything is materialized.
const BEND_EPS: f32 = 4.0;

/// Tidy `ids` (or the whole graph, when empty) — the full Sugiyama
/// pipeline, not the naive pass it replaced:
///
///  1. existing dots in the set are DISSOLVED (routing belongs to this
///     pass; running tidy twice must not stack dot upon dot),
///  2. ranks by longest path, then tightened toward the median of each
///     node's neighbours (sources fall to sit just above their children
///     instead of jamming the top row),
///  3. every edge spanning more than one rank is split into VIRTUAL
///     vertices, one per crossed rank, which take part in everything below,
///  4. crossing reduction: barycentre sweeps plus the classic transpose
///     pass, keeping the best ordering seen,
///  5. x placement by the priority method — virtual vertices outrank real
///     ones, so wire channels come out straight and vertical,
///  6. the virtual chains are simplified to their BENDS and materialized
///     as real routing dots (free at render time, and their tangents flow,
///     so the routed wire reads as one smooth line).
/// `ortho`: spaced for right-angle noodles — taller row gaps, so the
/// across-runs have room to sit in lanes (`ortho::lanes`).
pub fn auto_layout_for(graph: &mut EditorGraph, ids: &[u64], ortho: bool) -> bool {
    // --- the set; dissolve its dots before anything is measured ---
    let scoped = !ids.is_empty();
    let doomed: Vec<u64> = graph
        .nodes
        .iter()
        .filter(|(id, n)| {
            matches!(n.kind, engine::graph::NodeKind::Dot { .. })
                && (!scoped || ids.contains(id))
        })
        .map(|(id, _)| *id)
        .collect();
    for d in doomed {
        graph.dissolve_dot(d);
    }
    let set: Vec<u64> = if scoped {
        let mut v: Vec<u64> =
            ids.iter().copied().filter(|i| graph.nodes.contains_key(i)).collect();
        v.sort_unstable();
        v.dedup();
        v
    } else {
        graph.nodes.keys().copied().collect()
    };
    if set.len() < 2 {
        return false;
    }
    let old_c = centre(&set.iter().map(|i| graph.nodes[i].pos).collect::<Vec<_>>());

    // One record per wire inside the set.
    let wires: Vec<WireRec> = graph
        .edges
        .iter()
        .filter(|(d, _, s)| set.contains(d) && set.contains(s))
        .map(|(d, i, s)| WireRec { dst_id: *d, input: *i, src_id: *s, out: graph.out_of(*d, *i) })
        .collect();

    // --- ZONES: weakly-connected components lay out independently and
    // pack side by side, so unrelated islands never interleave rows ---
    let mut comp_of: std::collections::HashMap<u64, usize> =
        set.iter().enumerate().map(|(i, id)| (*id, i)).collect();
    // Union-find, flattened the lazy way (n is small).
    let mut root: Vec<usize> = (0..set.len()).collect();
    fn find(root: &mut Vec<usize>, mut i: usize) -> usize {
        while root[i] != i {
            root[i] = root[root[i]];
            i = root[i];
        }
        i
    }
    for w in &wires {
        let (a, b) = (comp_of[&w.dst_id], comp_of[&w.src_id]);
        let (ra, rb) = (find(&mut root, a), find(&mut root, b));
        if ra != rb {
            root[ra] = rb;
        }
    }
    let mut comps: std::collections::BTreeMap<usize, Vec<u64>> = Default::default();
    for (i, id) in set.iter().enumerate() {
        comps.entry(find(&mut root, i)).or_default().push(*id);
    }
    for (id, c) in comp_of.iter_mut() {
        let i = set.iter().position(|s| s == id).unwrap();
        *c = find(&mut root, i);
    }
    let mut comps: Vec<Vec<u64>> = comps.into_values().collect();
    // Big zones first, then by their smallest id — deterministic packing.
    comps.sort_by_key(|c| (std::cmp::Reverse(c.len()), c.iter().min().copied()));

    let mut laid: Vec<Laid> = Vec::new();
    for comp in &comps {
        let cw: Vec<WireRec> =
            wires.iter().filter(|w| comp.contains(&w.dst_id)).cloned().collect();
        laid.push(layout_component(graph, comp, &cw, ortho));
    }

    // Pack the zones left to right, top-aligned, with a gutter wide enough
    // that "this is a separate island" is legible at a glance.
    let gutter = 3.0 * COL_GAP;
    let mut cursor = 0.0f32;
    let mut shifts: Vec<Vec2> = Vec::new();
    for l in &laid {
        shifts.push(Vec2::new(cursor - l.bbox.min.x, -l.bbox.min.y));
        cursor += l.bbox.width() + gutter;
    }
    // Land the whole arrangement where the graph stood.
    let mut new_pts: Vec<Pos2> = Vec::new();
    for (l, sh) in laid.iter().zip(&shifts) {
        for p in &l.pos {
            new_pts.push(*p + *sh);
        }
    }
    let d = old_c - centre(&new_pts);

    for (l, sh) in laid.iter().zip(&shifts) {
        for (i, id) in l.ids.iter().enumerate() {
            if let Some(node) = graph.node_mut(*id) {
                node.pos = l.pos[i] + *sh + d;
            }
        }
        // Materialize the routed channels: dots at the bends only.
        for (w, pts) in &l.chains {
            let keep: Vec<Pos2> = pts.iter().map(|p| *p + *sh + d).collect();
            if keep.is_empty() {
                continue;
            }
            let kind_of = graph
                .node(w.dst_id)
                .map(|n| n.kind.input_kind(w.input));
            let kind_of = kind_of.unwrap_or(engine::graph::PortKind::Image);
            let mut prev = w.src_id;
            for p in keep {
                let dot = graph.add(engine::graph::NodeKind::dot(kind_of), p);
                graph.connect_out(dot, 0, prev, w.out);
                prev = dot;
            }
            graph.connect_out(w.dst_id, w.input, prev, w.out);
        }
    }
    // Harps are instruments, not wire-length problems: restring them.
    crate::breakout::repair(graph);
    true
}

/// One wire inside the layout's scope.
#[derive(Clone)]
struct WireRec {
    dst_id: u64,
    input: usize,
    src_id: u64,
    out: u32,
}

/// One laid-out zone, in LOCAL coordinates: node positions, the routed
/// chains' BEND points (already simplified), and the bounding box the
/// packer slots side by side.
struct Laid {
    ids: Vec<u64>,
    pos: Vec<Pos2>,
    chains: Vec<(WireRec, Vec<Pos2>)>,
    bbox: Rect,
}

/// The Sugiyama pipeline for ONE connected component. See `auto_layout`
/// for the phase-by-phase story.
fn layout_component(graph: &EditorGraph, comp: &[u64], wires: &[WireRec], ortho: bool) -> Laid {
    let n_real = comp.len();
    let index: std::collections::HashMap<u64, usize> =
        comp.iter().enumerate().map(|(i, id)| (*id, i)).collect();
    let pos0: Vec<Pos2> = comp.iter().map(|i| graph.nodes[i].pos).collect();
    let size: Vec<f32> = comp.iter().map(|i| graph.nodes[i].size).collect();
    if n_real == 1 {
        let half = NODE_R * size[0] + 8.0;
        return Laid {
            ids: comp.to_vec(),
            pos: vec![Pos2::ZERO],
            chains: Vec::new(),
            bbox: Rect::from_center_size(Pos2::ZERO, Vec2::splat(half * 2.0)),
        };
    }
    struct Wire {
        dst: usize,
        src: usize,
        rec: WireRec,
    }
    let wires: Vec<Wire> = wires
        .iter()
        .map(|w| Wire { dst: index[&w.dst_id], src: index[&w.src_id], rec: w.clone() })
        .collect();
    let (mut parents, mut children) = (vec![Vec::new(); n_real], vec![Vec::new(); n_real]);
    for w in &wires {
        if w.dst != w.src && !parents[w.dst].contains(&w.src) {
            parents[w.dst].push(w.src);
            children[w.src].push(w.dst);
        }
    }

    // --- ranks: longest path down, then median tightening ---
    let mut indeg: Vec<usize> = parents.iter().map(|p| p.len()).collect();
    let mut rank = vec![0usize; n_real];
    let mut q: std::collections::VecDeque<usize> =
        (0..n_real).filter(|i| indeg[*i] == 0).collect();
    let mut settled = 0usize;
    while let Some(v) = q.pop_front() {
        settled += 1;
        for k in 0..children[v].len() {
            let c = children[v][k];
            rank[c] = rank[c].max(rank[v] + 1);
            indeg[c] -= 1;
            if indeg[c] == 0 {
                q.push_back(c);
            }
        }
    }
    if settled != n_real {
        // A cycle. The editor refuses to make one, but never lay out
        // garbage: park whatever did not resolve on a rank of its own.
        let max = rank.iter().copied().max().unwrap_or(0);
        for i in 0..n_real {
            if indeg[i] > 0 {
                rank[i] = max + 1;
            }
        }
    }
    // Tighten: each node walks toward the median of its neighbours within
    // its slack, which pulls a source down beside the branch it feeds and
    // shortens every long edge the ordering pass would otherwise fight.
    for _ in 0..4 {
        let mut moved = false;
        for v in 0..n_real {
            let lo = parents[v].iter().map(|p| rank[*p] + 1).max().unwrap_or(0);
            let hi = children[v]
                .iter()
                .map(|c| rank[*c].saturating_sub(1))
                .min()
                .unwrap_or(rank[v].max(lo));
            if hi < lo {
                continue; // no slack (or cycle remnant)
            }
            let mut wants: Vec<usize> = parents[v]
                .iter()
                .map(|p| rank[*p] + 1)
                .chain(children[v].iter().map(|c| rank[*c].saturating_sub(1)))
                .collect();
            if wants.is_empty() {
                continue;
            }
            wants.sort_unstable();
            let med = wants[wants.len() / 2].clamp(lo, hi);
            if med != rank[v] {
                rank[v] = med;
                moved = true;
            }
        }
        if !moved {
            break;
        }
    }
    // Compact empty ranks away (tightening can abandon a row).
    {
        let mut used: Vec<usize> = rank.iter().copied().collect();
        used.sort_unstable();
        used.dedup();
        let remap: std::collections::HashMap<usize, usize> =
            used.iter().enumerate().map(|(new, old)| (*old, new)).collect();
        for r in rank.iter_mut() {
            *r = remap[r];
        }
    }
    let nranks = rank.iter().copied().max().unwrap_or(0) + 1;

    // --- virtual vertices: one per rank a long edge crosses ---
    let mut v_rank: Vec<usize> = rank.clone();
    let mut v_real: Vec<bool> = vec![true; n_real];
    struct Chain {
        wire: usize,
        verts: Vec<usize>,
    }
    let mut chains: Vec<Chain> = Vec::new();
    let mut up: Vec<Vec<usize>> = vec![Vec::new(); n_real];
    let mut down: Vec<Vec<usize>> = vec![Vec::new(); n_real];
    for (wi, w) in wires.iter().enumerate() {
        if w.dst == w.src || rank[w.dst] <= rank[w.src] {
            continue; // self loops / cycle remnants: leave the wire direct
        }
        let mut verts = Vec::new();
        let mut prev = w.src;
        for r in (rank[w.src] + 1)..rank[w.dst] {
            let v = v_rank.len();
            v_rank.push(r);
            v_real.push(false);
            up.push(Vec::new());
            down.push(Vec::new());
            verts.push(v);
            down[prev].push(v);
            up[v].push(prev);
            prev = v;
        }
        down[prev].push(w.dst);
        up[w.dst].push(prev);
        chains.push(Chain { wire: wi, verts });
    }
    let n_all = v_rank.len();
    let width = |v: usize| -> f32 {
        if v_real[v] { 2.0 * NODE_R * size[v] } else { CHAN_W }
    };

    // --- rows, seeded from where things already stand ---
    let mut rows: Vec<Vec<usize>> = vec![Vec::new(); nranks];
    for v in 0..n_all {
        rows[v_rank[v]].push(v);
    }
    let seed_x = |v: usize, chains: &[Chain]| -> f32 {
        if v_real[v] {
            pos0[v].x
        } else {
            let ch = chains.iter().find(|c| c.verts.contains(&v)).unwrap();
            let w = &wires[ch.wire];
            let (a, b) = (pos0[w.src], pos0[w.dst]);
            let t = (v_rank[v] - rank[w.src]) as f32 / (rank[w.dst] - rank[w.src]) as f32;
            a.x + (b.x - a.x) * t
        }
    };
    for row in rows.iter_mut() {
        row.sort_by(|a, b| {
            seed_x(*a, &chains)
                .partial_cmp(&seed_x(*b, &chains))
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(b))
        });
    }

    // --- crossing reduction: barycentre sweeps + transpose, keep the best ---
    let order_of = |rows: &[Vec<usize>]| -> Vec<usize> {
        let mut o = vec![0usize; n_all];
        for row in rows {
            for (k, v) in row.iter().enumerate() {
                o[*v] = k;
            }
        }
        o
    };
    let crossings = |rows: &[Vec<usize>]| -> usize {
        let order = order_of(rows);
        let mut total = 0;
        for row in rows {
            let mut es: Vec<(usize, usize)> = Vec::new();
            for &v in row {
                for &c in &down[v] {
                    es.push((order[v], order[c]));
                }
            }
            for i in 0..es.len() {
                for j in (i + 1)..es.len() {
                    let (a, b) = (es[i], es[j]);
                    if (a.0 < b.0 && a.1 > b.1) || (a.0 > b.0 && a.1 < b.1) {
                        total += 1;
                    }
                }
            }
        }
        total
    };
    let pair_cross = |order: &[usize], a: usize, b: usize| -> usize {
        let mut c = 0;
        for na in up[a].iter().chain(down[a].iter()) {
            for nb in up[b].iter().chain(down[b].iter()) {
                if order[*na] > order[*nb] {
                    c += 1;
                }
            }
        }
        c
    };
    let mut best_rows = rows.clone();
    let mut best_x = crossings(&rows);
    for sweep in 0..12 {
        let downward = sweep % 2 == 0;
        let seq: Vec<usize> = if downward {
            (1..nranks).collect()
        } else {
            (0..nranks.saturating_sub(1)).rev().collect()
        };
        for r in seq {
            let order = order_of(&rows);
            let key = |v: usize| -> f32 {
                let nb = if downward { &up[v] } else { &down[v] };
                if nb.is_empty() {
                    order[v] as f32
                } else {
                    nb.iter().map(|k| order[*k] as f32).sum::<f32>() / nb.len() as f32
                }
            };
            rows[r].sort_by(|a, b| {
                key(*a)
                    .partial_cmp(&key(*b))
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.cmp(b))
            });
        }
        for _ in 0..4 {
            let mut improved = false;
            let mut order = order_of(&rows);
            for r in 0..nranks {
                for k in 0..rows[r].len().saturating_sub(1) {
                    let (a, b) = (rows[r][k], rows[r][k + 1]);
                    if pair_cross(&order, a, b) > pair_cross(&order, b, a) {
                        rows[r].swap(k, k + 1);
                        order[a] = k + 1;
                        order[b] = k;
                        improved = true;
                    }
                }
            }
            if !improved {
                break;
            }
        }
        let x = crossings(&rows);
        if x < best_x {
            best_x = x;
            best_rows = rows.clone();
        }
    }
    let rows = best_rows;

    // --- x: priority placement. Virtual vertices outrank real ones, so a
    // channel pulls straight and the nodes arrange around it. Adjacent
    // STRANGERS — real nodes that share no parent — get extra air, which
    // is what reads as "these are different zones" inside one component ---
    let gap_between = |a: usize, b: usize| -> f32 {
        let g = |v: usize| if v_real[v] { COL_GAP } else { CHAN_GAP };
        let zone = if v_real[a] && v_real[b] {
            let share = up[a].iter().any(|p| up[b].contains(p));
            if up[a].is_empty() && up[b].is_empty() {
                1.4 // two roots side by side: separate intakes
            } else if !share {
                1.6 // different branches rubbing shoulders
            } else {
                1.0
            }
        } else {
            1.0
        };
        (width(a) + width(b)) * 0.5 + (g(a) + g(b)) * 0.5 * zone
    };
    let prio = |v: usize| -> usize {
        let deg = up[v].len() + down[v].len();
        if v_real[v] { deg } else { 1_000_000 + deg }
    };
    let mut x = vec![0f32; n_all];
    for row in &rows {
        let mut cx = 0.0;
        for (k, &v) in row.iter().enumerate() {
            if k > 0 {
                cx += gap_between(row[k - 1], v);
            }
            x[v] = cx;
        }
    }
    let place = |row: &[usize], x: &mut [f32], k: usize, target: f32| {
        let v = row[k];
        if target > x[v] {
            let mut limit = f32::INFINITY;
            for j in (k + 1)..row.len() {
                if prio(row[j]) >= prio(v) {
                    let mut lim = x[row[j]];
                    for jj in (k..j).rev() {
                        lim -= gap_between(row[jj], row[jj + 1]);
                    }
                    limit = lim;
                    break;
                }
            }
            let t = target.min(limit);
            if t > x[v] {
                x[v] = t;
                for j in (k + 1)..row.len() {
                    let need = x[row[j - 1]] + gap_between(row[j - 1], row[j]);
                    if x[row[j]] < need {
                        x[row[j]] = need;
                    } else {
                        break;
                    }
                }
            }
        } else if target < x[v] {
            let mut limit = f32::NEG_INFINITY;
            for j in (0..k).rev() {
                if prio(row[j]) >= prio(v) {
                    let mut lim = x[row[j]];
                    for jj in j..k {
                        lim += gap_between(row[jj], row[jj + 1]);
                    }
                    limit = lim;
                    break;
                }
            }
            let t = target.max(limit);
            if t < x[v] {
                x[v] = t;
                for j in (0..k).rev() {
                    let need = x[row[j + 1]] - gap_between(row[j], row[j + 1]);
                    if x[row[j]] > need {
                        x[row[j]] = need;
                    } else {
                        break;
                    }
                }
            }
        }
    };
    for pass in 0..16 {
        let downward = pass % 2 == 0;
        let seq: Vec<usize> =
            if downward { (0..nranks).collect() } else { (0..nranks).rev().collect() };
        for r in seq {
            let mut by_prio: Vec<usize> = (0..rows[r].len()).collect();
            by_prio.sort_by_key(|k| std::cmp::Reverse(prio(rows[r][*k])));
            for k in by_prio {
                let v = rows[r][k];
                let nb: Vec<usize> =
                    if downward { up[v].clone() } else { down[v].clone() };
                let nb = if nb.is_empty() {
                    up[v].iter().chain(down[v].iter()).copied().collect()
                } else {
                    nb
                };
                if nb.is_empty() {
                    continue;
                }
                let mut xs: Vec<f32> = nb.iter().map(|n| x[*n]).collect();
                xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let target = xs[xs.len() / 2];
                place(&rows[r], &mut x, k, target);
            }
        }
    }

    // --- y: one row per rank, tall enough for its biggest REAL node ---
    // With right-angle noodles, a gap also has to hold a lane for every
    // output whose wire jogs sideways through it.
    let row_gap = |r: usize| -> f32 {
        if !ortho {
            return ROW_GAP;
        }
        let mut jogs: Vec<usize> = Vec::new();
        for ch in &chains {
            let w = &wires[ch.wire];
            let path: Vec<usize> =
                std::iter::once(w.src).chain(ch.verts.iter().copied()).chain([w.dst]).collect();
            let jog = path.windows(2).any(|s| {
                v_rank[s[0]] == r && (x[s[0]] - x[s[1]]).abs() > 2.0
            });
            if jog && !jogs.contains(&w.src) {
                jogs.push(w.src);
            }
        }
        let extra = jogs.len().saturating_sub(1).min(ORTHO_LANES_MAX);
        ORTHO_ROW_GAP + ORTHO_LANE_ROOM * extra as f32
    };
    let mut row_y = vec![0f32; nranks];
    let mut cy = 0.0f32;
    for (r, row) in rows.iter().enumerate() {
        let h = row
            .iter()
            .filter(|v| v_real[**v])
            .map(|v| width(*v))
            .fold(2.0 * NODE_R * 0.4, f32::max);
        cy += h * 0.5;
        row_y[r] = cy;
        cy += h * 0.5 + row_gap(r);
    }

    // --- collect: local positions, simplified chains, the zone's bbox ---
    let pos: Vec<Pos2> = (0..n_real).map(|i| Pos2::new(x[i], row_y[rank[i]])).collect();
    let mut bbox = Rect::NOTHING;
    for i in 0..n_real {
        let half = width(i) * 0.5 + 8.0;
        bbox = bbox.union(Rect::from_center_size(pos[i], Vec2::splat(half * 2.0)));
    }
    let mut out_chains: Vec<(WireRec, Vec<Pos2>)> = Vec::new();
    for ch in &chains {
        if ch.verts.is_empty() {
            continue;
        }
        let w = &wires[ch.wire];
        let mut pts: Vec<Pos2> = Vec::with_capacity(ch.verts.len() + 2);
        pts.push(pos[w.src]);
        for &v in &ch.verts {
            pts.push(Pos2::new(x[v], row_y[v_rank[v]]));
        }
        pts.push(pos[w.dst]);
        let keep = simplify_polyline(&pts, BEND_EPS);
        for p in &keep {
            bbox = bbox.union(Rect::from_center_size(*p, Vec2::splat(CHAN_W)));
        }
        if !keep.is_empty() {
            out_chains.push((w.rec.clone(), keep));
        }
    }
    Laid { ids: comp.to_vec(), pos, chains: out_chains, bbox }
}

/// Douglas-Peucker over an open polyline: returns the INTERIOR points that
/// are genuine bends (deviate more than `eps` from the straight line
/// through their kept neighbours). Endpoints are never returned.
fn simplify_polyline(pts: &[Pos2], eps: f32) -> Vec<Pos2> {
    fn dp(pts: &[Pos2], a: usize, b: usize, eps: f32, keep: &mut Vec<usize>) {
        if b <= a + 1 {
            return;
        }
        let (pa, pb) = (pts[a], pts[b]);
        let ab = pb - pa;
        let len = ab.length().max(1e-6);
        let (mut worst, mut worst_d) = (a, 0.0f32);
        for i in (a + 1)..b {
            let d = ((pts[i].x - pa.x) * ab.y - (pts[i].y - pa.y) * ab.x).abs() / len;
            if d > worst_d {
                worst_d = d;
                worst = i;
            }
        }
        if worst_d > eps {
            dp(pts, a, worst, eps, keep);
            keep.push(worst);
            dp(pts, worst, b, eps, keep);
        }
    }
    let mut keep = Vec::new();
    if pts.len() >= 3 {
        dp(pts, 0, pts.len() - 1, eps, &mut keep);
    }
    keep.sort_unstable();
    keep.into_iter().map(|i| pts[i]).collect()
}

fn centre(p: &[Pos2]) -> Pos2 {
    let mut bb = Rect::NOTHING;
    for q in p {
        bb = bb.union(Rect::from_center_size(*q, Vec2::splat(1.0)));
    }
    bb.center()
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::graph::{BlendOp, NodeKind};

    fn grade() -> NodeKind {
        NodeKind::Grade(Default::default())
    }

    #[test]
    fn a_swipe_puts_the_nodes_on_the_line_it_drew() {
        let mut g = EditorGraph::default();
        let ids: Vec<u64> = (0..4)
            .map(|k| g.add(grade(), Pos2::new(k as f32 * 13.0, 40.0 * k as f32)))
            .collect();
        // A sideways stroke well below them: they land on it, in a row.
        assert!(align_to_swipe(&mut g, &ids, Pos2::new(-200.0, 500.0), Pos2::new(200.0, 508.0)));
        let ys: Vec<f32> = ids.iter().map(|i| g.nodes[i].pos.y).collect();
        assert!(ys.iter().all(|y| (*y - 504.0).abs() < 0.01), "{ys:?}");
        let mut xs: Vec<f32> = ids.iter().map(|i| g.nodes[i].pos.x).collect();
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let steps: Vec<f32> = xs.windows(2).map(|w| w[1] - w[0]).collect();
        assert!(
            steps.windows(2).all(|s| (s[0] - s[1]).abs() < 0.01),
            "spacing must be even: {steps:?}"
        );
        // No two nodes may end up overlapping, however short the stroke.
        assert!(steps.iter().all(|s| *s >= 2.0 * NODE_R), "{steps:?}");
    }

    #[test]
    fn a_downward_swipe_makes_a_column() {
        let mut g = EditorGraph::default();
        let ids: Vec<u64> =
            (0..3).map(|k| g.add(grade(), Pos2::new(k as f32 * 90.0, 0.0))).collect();
        align_to_swipe(&mut g, &ids, Pos2::new(10.0, 0.0), Pos2::new(14.0, 400.0));
        let xs: Vec<f32> = ids.iter().map(|i| g.nodes[i].pos.x).collect();
        assert!(xs.iter().all(|x| (*x - 12.0).abs() < 0.01), "{xs:?}");
    }

    /// R -> A -> B -> C, scattered. Layout must put them one per rank, in
    /// flow order, on one straight vertical line — the thing a chain wants.
    #[test]
    fn a_chain_lays_out_straight_and_in_flow_order() {
        let mut g = EditorGraph::default();
        let r = g.add(NodeKind::Read { source: 0, path: String::new(), range: None, ram_cache: false, nvme_cache: true, layer: None, input: Default::default(), multi: false, start: None, first: None, last: None, edges: Default::default(), cache_on_play: true, single: Default::default(), eight_bit: false, scale: Default::default() }, Pos2::new(300.0, -80.0));
        let a = g.add(grade(), Pos2::new(-90.0, 400.0));
        let b = g.add(grade(), Pos2::new(700.0, 20.0));
        let c = g.add(grade(), Pos2::new(0.0, -300.0));
        g.connect(a, 0, r);
        g.connect(b, 0, a);
        g.connect(c, 0, b);
        assert!(auto_layout_for(&mut g, &[], false));
        let p: Vec<Pos2> = [r, a, b, c].iter().map(|i| g.nodes[i].pos).collect();
        for w in p.windows(2) {
            assert!(w[1].y > w[0].y + 2.0 * NODE_R, "flow must run downhill: {p:?}");
            assert!((w[1].x - w[0].x).abs() < 0.01, "a chain must come out straight: {p:?}");
        }
    }

    /// A Blend sits below BOTH its inputs, even when one branch is longer —
    /// that is what longest-path ranking buys over a plain depth-first
    /// depth, and getting it wrong draws a wire running back uphill.
    #[test]
    fn a_blend_lands_below_its_longer_branch() {
        let mut g = EditorGraph::default();
        let r0 = g.add(grade(), Pos2::new(0.0, 0.0));
        let r1 = g.add(grade(), Pos2::new(200.0, 0.0));
        let mid = g.add(grade(), Pos2::new(200.0, 100.0));
        let deep = g.add(grade(), Pos2::new(200.0, 200.0));
        let m = g.add(NodeKind::Blend { op: BlendOp::Over, mix: 1.0, gain: [1.0, 1.0] }, Pos2::new(100.0, 300.0));
        g.connect(mid, 0, r1);
        g.connect(deep, 0, mid);
        g.connect(m, 0, r0);
        g.connect(m, 1, deep);
        assert!(auto_layout_for(&mut g, &[], false));
        for up in [r0, r1, mid, deep] {
            assert!(
                g.nodes[&m].pos.y > g.nodes[&up].pos.y,
                "the blend must sit below {up}"
            );
        }
        // Two nodes must never be laid on top of each other.
        let mut ps: Vec<Pos2> = g.nodes.values().map(|n| n.pos).collect();
        for i in 0..ps.len() {
            for j in (i + 1)..ps.len() {
                assert!(
                    (ps[i].x - ps[j].x).abs() > 1.0 || (ps[i].y - ps[j].y).abs() > 1.0,
                    "overlap: {:?} {:?}",
                    ps[i],
                    ps[j]
                );
            }
        }
        ps.clear();
    }

    /// Laying out a SELECTION only moves the selection, and never consults
    /// wires leaving it — the rest of the comp is staying where it is.
    #[test]
    fn layout_of_a_selection_leaves_everything_else_alone() {
        let mut g = EditorGraph::default();
        let out = g.add(grade(), Pos2::new(999.0, 999.0));
        let a = g.add(grade(), Pos2::new(0.0, 0.0));
        let b = g.add(grade(), Pos2::new(40.0, 5.0));
        g.connect(b, 0, a);
        g.connect(out, 0, b);
        auto_layout_for(&mut g, &[a, b], false);
        assert_eq!(g.nodes[&out].pos, Pos2::new(999.0, 999.0));
        assert!(g.nodes[&b].pos.y > g.nodes[&a].pos.y);
    }

    /// The whole point of the blob over a rectangle: an L-shaped selection
    /// leaves its notch OPEN. The far corner of the bounding box is outside
    /// the shape, while every member and every wire between members is in.
    #[test]
    fn the_blob_conforms_to_the_selection_shape() {
        let mut g = EditorGraph::default();
        let a = g.add(grade(), Pos2::new(0.0, 0.0));
        let b = g.add(grade(), Pos2::new(0.0, 300.0));
        let c = g.add(grade(), Pos2::new(300.0, 300.0));
        g.connect(b, 0, a);
        g.connect(c, 0, b);
        let parts = blob_parts(&g, &[a, b, c], BLOB_PAD);
        for id in [a, b, c] {
            assert!(blob_sdf(&parts, g.nodes[&id].pos) < 0.0, "member {id} must be inside");
        }
        assert!(blob_sdf(&parts, Pos2::new(0.0, 150.0)) < 0.0, "the wire sleeve is inside");
        assert!(
            blob_sdf(&parts, Pos2::new(300.0, 0.0)) > 20.0,
            "the notch corner must stay OUTSIDE — that is what 'not just a square' means"
        );
    }

    /// The buffer is a per-group knob: a probe point outside a tight wrap
    /// is inside a generous one.
    #[test]
    fn the_buffer_is_per_group() {
        let mut g = EditorGraph::default();
        let a = g.add(grade(), Pos2::new(0.0, 0.0));
        let probe = Pos2::new(NODE_R + 40.0, 0.0);
        let tight = blob_parts(&g, &[a], 4.0);
        let roomy = blob_parts(&g, &[a], 80.0);
        assert!(blob_sdf(&tight, probe) > 0.0, "outside a tight wrap");
        assert!(blob_sdf(&roomy, probe) < 0.0, "inside a generous one");
    }

    /// An unwired pair still makes ONE blob: the spanning tether connects
    /// them, so a group never falls apart into islands.
    #[test]
    fn unwired_members_are_still_one_blob() {
        let mut g = EditorGraph::default();
        let a = g.add(grade(), Pos2::new(0.0, 0.0));
        let b = g.add(grade(), Pos2::new(400.0, 0.0));
        let parts = blob_parts(&g, &[a, b], BLOB_PAD);
        assert!(blob_sdf(&parts, Pos2::new(200.0, 0.0)) < 0.0, "the tether spans the gap");
    }

    #[test]
    fn build_blob_closes_its_outline_and_places_its_handles() {
        let mut g = EditorGraph::default();
        let a = g.add(grade(), Pos2::new(0.0, 0.0));
        let b = g.add(grade(), Pos2::new(180.0, 90.0));
        g.connect(b, 0, a);
        let blob = build_blob(&g, &[a, b], 10.0, BLOB_PAD).expect("two nodes make a blob");
        assert!(!blob.tris.is_empty() && !blob.loops.is_empty());
        for lp in &blob.loops {
            let (first, last) = (lp[0], *lp.last().unwrap());
            assert!(first.distance(last) <= 10.0 * 1.5, "outline must close: {first:?} vs {last:?}");
        }
        // The tab hangs off the topmost member; the grip is on the shape's
        // lower-right, inside the bbox.
        assert!(blob.tab_anchor.y < 0.0 && blob.tab_anchor.x.abs() < 1.0);
        assert!(blob.bbox.contains(blob.grip));
        assert!(blob.grip.x + blob.grip.y > 180.0);
    }

    #[test]
    fn membership_is_exclusive_and_survives_node_removal() {
        let mut g = EditorGraph::default();
        let a = g.add(grade(), Pos2::new(0.0, 0.0));
        let b = g.add(grade(), Pos2::new(100.0, 0.0));
        g.groups.push(group_over(&[a], 100));
        g.groups.push(group_over(&[b], 101));
        // b joins a's group and leaves its own; the emptied group dies.
        g.adopt_into_group(100, &[b]);
        assert_eq!(g.groups.len(), 1);
        assert!(g.groups[0].members.contains(&a) && g.groups[0].members.contains(&b));
        assert_eq!(g.group_of(b), Some(100));
        // Removing a node removes it from its group; the last one out
        // takes the backdrop with it.
        g.remove(a);
        assert_eq!(g.groups[0].members, vec![b]);
        g.remove(b);
        assert!(g.groups.is_empty());
    }

    /// A setup saved by the rectangular v1 backdrop adopts whoever stood on
    /// the rectangle, exactly once.
    #[test]
    fn legacy_rect_groups_adopt_their_contents() {
        let mut g = EditorGraph::default();
        let a = g.add(grade(), Pos2::new(0.0, 0.0));
        let b = g.add(grade(), Pos2::new(120.0, 40.0));
        let far = g.add(grade(), Pos2::new(2000.0, 0.0));
        g.groups.push(Group {
            id: 9,
            title: "old".into(),
            members: Vec::new(),
            rect: Some(Rect::from_min_max(Pos2::new(-60.0, -60.0), Pos2::new(200.0, 100.0))),
            pad: BLOB_PAD,
            label_h: TITLE_H,
            color: [1, 2, 3],
        });
        g.migrate_groups();
        let grp = &g.groups[0];
        assert!(grp.rect.is_none());
        assert!(grp.members.contains(&a) && grp.members.contains(&b));
        assert!(!grp.members.contains(&far));
    }

    /// The headline of the routed layout. One long edge alone gets a
    /// STRAIGHT channel (the virtuals outrank real nodes, so the chain
    /// steps aside and no dots are needed). Two long edges CROSSING —
    /// each source parked over its own chain, each blend under the other's
    /// — cannot both run straight: the router must bend them, and a bend
    /// is a routing dot. Every routed hop must still run downhill and keep
    /// clear of the six chain nodes it bypasses.
    #[test]
    fn crossing_long_edges_are_routed_through_dots() {
        let mut g = EditorGraph::default();
        let s1 = g.add(NodeKind::read(0, ""), Pos2::new(-200.0, 0.0));
        let s2 = g.add(NodeKind::read(1, ""), Pos2::new(200.0, 0.0));
        let chain = |g: &mut EditorGraph, src: u64, x: f32| -> u64 {
            let mut prev = src;
            for k in 1..=3 {
                let n = g.add(grade(), Pos2::new(x, 100.0 * k as f32));
                g.connect(n, 0, prev);
                prev = n;
            }
            prev
        };
        let c1 = chain(&mut g, s1, -200.0);
        let c2 = chain(&mut g, s2, 200.0);
        let m1 = g.add(NodeKind::blend(), Pos2::new(-200.0, 400.0));
        let m2 = g.add(NodeKind::blend(), Pos2::new(200.0, 400.0));
        g.connect(m1, 0, c1);
        g.connect(m1, 1, s2); // crosses left
        g.connect(m2, 0, c2);
        g.connect(m2, 1, s1); // crosses right
        assert!(auto_layout_for(&mut g, &[], false));

        let obstacles: Vec<u64> = g
            .nodes
            .iter()
            .filter(|(id, n)| {
                !matches!(n.kind, NodeKind::Dot { .. })
                    && ![s1, s2, m1, m2].contains(id)
            })
            .map(|(id, _)| *id)
            .collect();
        let mut total_dots = 0;
        for (m, src) in [(m1, s2), (m2, s1)] {
            let mut path = vec![g.nodes[&m].pos];
            let mut cur = g.input_of(m, 1).expect("the long edge is still wired");
            while g.is_dot(cur) {
                total_dots += 1;
                path.push(g.nodes[&cur].pos);
                cur = g.input_of(cur, 0).expect("a routed dot is wired");
            }
            assert_eq!(cur, src, "the chain resolves to its source");
            path.push(g.nodes[&src].pos);
            for w in path.windows(2) {
                assert!(w[0].y > w[1].y, "routed hops must run downhill: {path:?}");
                for ob in &obstacles {
                    let d = seg_dist(g.nodes[ob].pos, w[0], w[1]);
                    assert!(
                        d > NODE_R + 4.0,
                        "a routed hop passes {d:.0} units from node {ob} — through it"
                    );
                }
            }
        }
        assert!(
            total_dots >= 2,
            "two crossing four-rank edges cannot both be straight; the router \
             must have bent them through dots (got {total_dots})"
        );
        assert!(total_dots <= 8, "bend simplification must not spray dots");
    }

    /// Unrelated islands are ZONES: each connected component lays out on
    /// its own and they pack side by side with a wide gutter — never
    /// interleaved into each other's rows.
    #[test]
    fn disconnected_components_pack_apart() {
        let mut g = EditorGraph::default();
        // Island 1: a chain of 3. Island 2: a chain of 2. Scrambled so the
        // naive result would interleave them.
        let a0 = g.add(grade(), Pos2::new(500.0, 0.0));
        let a1 = g.add(grade(), Pos2::new(-500.0, 80.0));
        let a2 = g.add(grade(), Pos2::new(490.0, 160.0));
        let b0 = g.add(grade(), Pos2::new(-490.0, 10.0));
        let b1 = g.add(grade(), Pos2::new(505.0, 90.0));
        g.connect(a1, 0, a0);
        g.connect(a2, 0, a1);
        g.connect(b1, 0, b0);
        assert!(auto_layout_for(&mut g, &[], false));
        let ax: Vec<f32> = [a0, a1, a2].iter().map(|i| g.nodes[i].pos.x).collect();
        let bx: Vec<f32> = [b0, b1].iter().map(|i| g.nodes[i].pos.x).collect();
        let (a_lo, a_hi) = (ax.iter().cloned().fold(f32::MAX, f32::min), ax.iter().cloned().fold(f32::MIN, f32::max));
        let (b_lo, b_hi) = (bx.iter().cloned().fold(f32::MAX, f32::min), bx.iter().cloned().fold(f32::MIN, f32::max));
        let gap = if a_hi < b_lo { b_lo - a_hi } else { a_lo - b_hi };
        assert!(
            gap > 2.0 * COL_GAP,
            "islands must sit apart, not interleave (gap {gap:.0}; a {a_lo:.0}..{a_hi:.0}, b {b_lo:.0}..{b_hi:.0})"
        );
        // Both islands top-aligned: their first rows share a y.
        assert!((g.nodes[&a0].pos.y - g.nodes[&b0].pos.y).abs() < 1.0);
    }

    /// Tidy owns its routing: running it again dissolves its own dots and
    /// reproduces the same picture instead of stacking dot upon dot.
    #[test]
    fn tidy_restrings_a_harp() {
        use engine::graph::{BreakoutParams, NodeKind, RecombineParams};
        let mut g = EditorGraph::default();
        let bar = g.add(
            NodeKind::Breakout(BreakoutParams {
                layers: vec!["a".into(), "b".into(), "c".into()],
                ..Default::default()
            }),
            egui::pos2(0.0, 0.0),
        );
        let rec = g.add(
            NodeKind::Recombine(RecombineParams { n: 3, pair: bar, ..Default::default() }),
            egui::pos2(0.0, 300.0),
        );
        if let Some(NodeKind::Breakout(p)) = g.nodes.get_mut(&bar).map(|n| &mut n.kind) {
            p.pair = rec;
        }
        // A grade spliced into string 1, thrown somewhere silly.
        let grade =
            g.add(NodeKind::Grade(Default::default()), egui::pos2(500.0, -200.0));
        g.connect_out(grade, 0, bar, 1);
        for k in 0..3usize {
            if k == 1 {
                g.connect(rec, k, grade);
            } else {
                g.connect_out(rec, k, bar, k as u32);
            }
        }
        assert!(auto_layout_for(&mut g, &[], false));
        let (bn, gn) = (g.node(bar).unwrap().clone(), g.node(grade).unwrap().clone());
        let want =
            crate::breakout::string_x(&bn.kind, bn.pos, 1, bn.size).unwrap();
        assert!(
            (gn.pos.x - want).abs() < 0.5,
            "the spliced node sits ON its string: x {} want {want}",
            gn.pos.x
        );
        let rn = g.node(rec).unwrap();
        assert_eq!(rn.pos.x, bn.pos.x, "bars share a column");
        assert!(rn.pos.y > bn.pos.y, "receiver hangs below");
        assert!(gn.pos.y > bn.pos.y && gn.pos.y < rn.pos.y, "chain between bars");
    }

    #[test]
    fn tidy_is_rerunnable() {
        let mut g = EditorGraph::default();
        let r = g.add(NodeKind::read(0, ""), Pos2::new(13.0, -7.0));
        let a = g.add(grade(), Pos2::new(300.0, 40.0));
        let b = g.add(grade(), Pos2::new(-200.0, 90.0));
        let c = g.add(grade(), Pos2::new(80.0, 500.0));
        let m = g.add(NodeKind::blend(), Pos2::new(0.0, 220.0));
        g.connect(a, 0, r);
        g.connect(b, 0, a);
        g.connect(c, 0, b);
        g.connect(m, 0, c);
        g.connect(m, 1, r);
        assert!(auto_layout_for(&mut g, &[], false));
        let count1 = g.nodes.len();
        let pos1: Vec<Pos2> = [r, a, b, c, m].iter().map(|i| g.nodes[i].pos).collect();
        assert!(auto_layout_for(&mut g, &[], false));
        assert_eq!(g.nodes.len(), count1, "second tidy must not stack dots");
        for (i, id) in [r, a, b, c, m].iter().enumerate() {
            assert!(
                g.nodes[id].pos.distance(pos1[i]) < 1.0,
                "second tidy moved node {id}: {:?} -> {:?}",
                pos1[i],
                g.nodes[id].pos
            );
        }
    }

    /// Rank tightening: a source feeding only the BOTTOM of a chain falls
    /// to sit just above its consumer, instead of jamming the top row with
    /// a four-rank edge.
    #[test]
    fn sources_fall_toward_their_children() {
        let mut g = EditorGraph::default();
        let a0 = g.add(grade(), Pos2::new(0.0, 0.0));
        let a1 = g.add(grade(), Pos2::new(0.0, 100.0));
        let a2 = g.add(grade(), Pos2::new(0.0, 200.0));
        let m = g.add(NodeKind::blend(), Pos2::new(0.0, 300.0));
        let s = g.add(NodeKind::read(0, ""), Pos2::new(200.0, 0.0));
        g.connect(a1, 0, a0);
        g.connect(a2, 0, a1);
        g.connect(m, 0, a2);
        g.connect(m, 1, s);
        assert!(auto_layout_for(&mut g, &[], false));
        let (sy, a2y, a0y) = (g.nodes[&s].pos.y, g.nodes[&a2].pos.y, g.nodes[&a0].pos.y);
        assert!(
            (sy - a2y).abs() < 1.0,
            "the source should share the row just above its consumer (s.y {sy}, a2.y {a2y})"
        );
        assert!(sy > a0y + 100.0, "and be nowhere near the top row");
        // Short edge now: no dots were needed.
        assert!(g.input_of(m, 1) == Some(s));
    }

    /// Dissolving a dot heals its consumers with the SOURCE-OUTPUT index
    /// intact — the generic remove() heal drops it.
    #[test]
    fn dissolve_dot_keeps_the_output_index() {
        let mut g = EditorGraph::default();
        let a = g.add(grade(), Pos2::new(0.0, 0.0));
        let b = g.add(grade(), Pos2::new(0.0, 100.0));
        let d = g.add(NodeKind::dot(engine::graph::PortKind::Image), Pos2::new(0.0, 50.0));
        g.connect_out(d, 0, a, 2);
        g.connect_out(b, 0, d, 2);
        assert!(g.dissolve_dot(d));
        assert_eq!(g.input_of(b, 0), Some(a));
        assert_eq!(g.out_of(b, 0), 2, "the output index must survive the heal");
        assert!(!g.nodes.contains_key(&d));
    }

    /// Not an assertion so much as a fixture: writes a ~400-node scrambled
    /// comp to target/messy_setup.wbl (⌘O it, then press T) — six reads,
    /// thirty-six branching lanes in five merge zones, mask wiring, cross-zone
    /// long edges, three disconnected islands, two backdrops, and a dozen
    /// pre-scattered dots for the tidy pass to dissolve. Deterministic, so
    /// the file is stable run to run — and the test also tidies a CLONE,
    /// so layout at this scale is proven every run, not just handed over.
    #[test]
    fn a_messy_setup_for_hand_testing_is_generated() {
        use engine::graph::{BlurParams, CurveParams, LumaKeyParams, TransformParams};
        let mut g = EditorGraph::default();
        g.working = engine::color::Colorspace::ACESCG;
        // Tiny LCG: scrambled, but the same scramble every time.
        let mut seed: u64 = 0x5DEECE66D;
        let mut rng = move |lo: f32, hi: f32| -> f32 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            lo + (seed >> 33) as f32 / (u32::MAX >> 1) as f32 * (hi - lo)
        };
        let mut scatter = |g: &mut EditorGraph, id: u64| {
            let p = Pos2::new(rng(-2800.0, 2800.0), rng(-1700.0, 1700.0));
            g.nodes.get_mut(&id).unwrap().pos = p;
        };
        let kind = |k: usize| -> NodeKind {
            match k % 6 {
                0 => NodeKind::Grade(Default::default()),
                1 => NodeKind::Blur(BlurParams::default()),
                2 => NodeKind::Mult(Default::default()),
                3 => NodeKind::Saturation(Default::default()),
                4 => NodeKind::Transform(TransformParams::default()),
                _ => NodeKind::Curve(CurveParams::default()),
            }
        };
        let reads: Vec<u64> = (0..6)
            .map(|k| {
                let path = if k % 2 == 0 { "data/2k_zip_bg" } else { "data/2k_zip_fg" };
                let id = g.add(NodeKind::read(k, path), Pos2::ZERO);
                scatter(&mut g, id);
                id
            })
            .collect();
        // Thirty lanes; the last five branch off earlier lanes instead of a
        // read, so the zones share upstream history.
        let mut kk = 0usize;
        let mut lanes: Vec<Vec<u64>> = Vec::new();
        for lane in 0..36usize {
            let src = if lane < 25 {
                reads[lane % reads.len()]
            } else {
                let donor = &lanes[lane - 25];
                donor[donor.len() / 2]
            };
            let mut chain = vec![src];
            let len = 6 + (lane * 7 + 3) % 8;
            for _ in 0..len {
                let id = g.add(kind(kk), Pos2::ZERO);
                scatter(&mut g, id);
                kk += 1;
                g.connect(id, 0, *chain.last().unwrap());
                chain.push(id);
            }
            lanes.push(chain);
        }
        // Five zones of six lanes, each merged by a little blend tree.
        let mut zone_outs: Vec<u64> = Vec::new();
        for z in 0..5usize {
            let mut heads: Vec<u64> =
                (0..6).map(|k| *lanes[z * 6 + k].last().unwrap()).collect();
            while heads.len() > 1 {
                let b = g.add(NodeKind::blend(), Pos2::ZERO);
                scatter(&mut g, b);
                let a0 = heads.remove(0);
                let a1 = heads.remove(0);
                g.connect(b, 0, a0);
                g.connect(b, 1, a1);
                heads.push(b);
            }
            zone_outs.push(heads[0]);
        }
        // The spine: zones fold into one final blend; a read reaches the
        // end directly — a proper cross-graph long edge.
        let mut spine = zone_outs[0];
        for zo in &zone_outs[1..] {
            let b = g.add(NodeKind::blend(), Pos2::ZERO);
            scatter(&mut g, b);
            g.connect(b, 0, spine);
            g.connect(b, 1, *zo);
            spine = b;
        }
        let fin = g.add(NodeKind::blend(), Pos2::ZERO);
        scatter(&mut g, fin);
        g.connect(fin, 0, spine);
        g.connect(fin, 1, reads[1]);
        // Mask wiring: LumaKeys off one zone gate blends in ANOTHER —
        // cross-zone mask edges, the routing pass's favourite food.
        for m in 0..6usize {
            let tap = {
                let lane = &lanes[(m * 5 + 2) % lanes.len()];
                lane[lane.len() / 2]
            };
            let lk = g.add(NodeKind::LumaKey(LumaKeyParams::default()), Pos2::ZERO);
            scatter(&mut g, lk);
            g.connect(lk, 0, tap);
            if m < zone_outs.len() {
                g.connect(zone_outs[(m + 2) % zone_outs.len()], 2, lk);
            } else {
                let msk = g.add(NodeKind::Mask, Pos2::ZERO);
                scatter(&mut g, msk);
                let img = *lanes[(m * 7 + 1) % lanes.len()].last().unwrap();
                g.connect(msk, 0, img);
                g.connect(msk, 1, lk);
            }
        }
        // Three disconnected ISLANDS — zones with no wire to the comp at
        // all, for the component packer to isolate.
        for isl in 0..3usize {
            let mut prev = None;
            for k in 0..(4 + isl) {
                let id = g.add(kind(k + isl), Pos2::ZERO);
                scatter(&mut g, id);
                if let Some(p) = prev {
                    g.connect(id, 0, p);
                }
                prev = Some(id);
            }
        }
        // Pre-scattered dots on a dozen wires, for tidy to dissolve.
        let edges_snapshot: Vec<(u64, usize, u64)> = g.edges.clone();
        for k in 0..12usize {
            let (dst, input, src) = edges_snapshot[(k * 37) % edges_snapshot.len()];
            if g.is_dot(src) || g.input_of(dst, input) != Some(src) {
                continue;
            }
            let d = g.add(NodeKind::dot(engine::graph::PortKind::Image), Pos2::ZERO);
            scatter(&mut g, d);
            g.connect(d, 0, src);
            g.connect(dst, input, d);
        }
        // Two backdrops: a lane in the comp, and an island.
        for (title, members) in
            [("lane", lanes[3][1..].to_vec()), ("spare parts", {
                // The last island's nodes are the highest ids before dots…
                // easier: everything unreachable from `fin` that is not a
                // dot — the islands exactly.
                let mut on_comp = std::collections::HashSet::new();
                let mut stack = vec![fin];
                while let Some(v) = stack.pop() {
                    if !on_comp.insert(v) {
                        continue;
                    }
                    for (d, _, s) in &g.edges {
                        if *d == v {
                            stack.push(*s);
                        }
                    }
                }
                g.nodes
                    .iter()
                    .filter(|(id, n)| {
                        !on_comp.contains(id) && !matches!(n.kind, NodeKind::Dot { .. })
                    })
                    .map(|(id, _)| *id)
                    .collect()
            })]
        {
            let gid = g.next_id;
            g.next_id += 1;
            let mut grp = group_over(&members, gid);
            grp.title = title.into();
            g.groups.push(grp);
        }
        g.view = Some(fin);

        assert!(g.nodes.len() >= 350, "meant to be GIANT ({} nodes)", g.nodes.len());
        // Layout at this scale must terminate, route, and stay downhill.
        let mut tidy = g.clone();
        assert!(auto_layout_for(&mut tidy, &[], false));
        for (d, _, s) in &tidy.edges {
            let (Some(dn), Some(sn)) = (tidy.node(*d), tidy.node(*s)) else { continue };
            assert!(dn.pos.y > sn.pos.y, "tidied edge {s}->{d} runs uphill");
        }
        let doc = serde_json::json!({ "version": 2u32, "fps": 24.0f32, "graph": g });
        let text = serde_json::to_string_pretty(&doc).unwrap();
        // Prove the file round-trips through the loader's graph field
        // before handing it to a human.
        let back: serde_json::Value = serde_json::from_str(&text).unwrap();
        let g2: EditorGraph = serde_json::from_value(back["graph"].clone()).unwrap();
        assert_eq!(g2.nodes.len(), g.nodes.len());
        assert_eq!(g2.edges.len(), g.edges.len());
        let out = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("target/messy_setup.wbl");
        std::fs::write(&out, text).unwrap();
        println!(
            "messy setup written to {} ({} nodes, {} edges)",
            out.display(),
            g.nodes.len(),
            g.edges.len()
        );
    }

    /// A cycle cannot be authored, but a malformed setup must still lay out
    /// rather than hang or panic.
    #[test]
    fn a_cycle_still_terminates() {
        let mut g = EditorGraph::default();
        let a = g.add(grade(), Pos2::new(0.0, 0.0));
        let b = g.add(grade(), Pos2::new(10.0, 10.0));
        g.edges.push((a, 0, b));
        g.edges.push((b, 0, a));
        assert!(auto_layout_for(&mut g, &[], false));
    }
}
