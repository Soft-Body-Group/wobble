//! Regions — render-layer-scale grouping (FG / ENV / CHAR).
//!
//! A region is a named, tinted wash UNDER a stretch of the graph — higher
//! scope than a backdrop and deliberately quieter to look at. Unlike a
//! backdrop (`nodeux::Group`, decoration that owns nothing) a region OWNS
//! things: its member list and a registry of COMMON SOURCES — nodes (the
//! crypto Read, the util Read) that the backtick pie offers while the view
//! centre stands inside the region. Picking one spawns a SOURCE CHIP: a Dot
//! hidden-wired to the source, so the plumbing exists without a noodle
//! crossing the comp.
//!
//! Hidden wires are ordinary [`EditorGraph::edges`] entries with a
//! presentation flag (`EditorGraph::hidden_wires`), so the engine, the
//! cost/licence taint, cycle checks and the caches never learn the concept.
//! The readability contract lives in the editor: a hidden-fed node wears an
//! amber bead, hovering it draws the ghost noodle to the origin, and
//! node-patch mode (shift+`) shows and edits the whole hidden layer while
//! the ordinary wiring locks grey.

use crate::editor::EditorGraph;
use egui::{Pos2, Rect, Vec2};

/// Muted washes — regions are for orientation at arm's length, so these run
/// dimmer than the backdrop palette on purpose.
pub const REGION_COLORS: [[u8; 3]; 6] = [
    [0x1d, 0x9e, 0x75], // teal
    [0xd8, 0x5a, 0x30], // ember
    [0x37, 0x8a, 0xdd], // blue
    [0xd4, 0x53, 0x7e], // pink
    [0xba, 0x75, 0x17], // amber
    [0x7f, 0x77, 0xdd], // violet
];

/// How far the wash stands off the members' bounding box, in graph units.
pub const PAD: f32 = 70.0;

/// Backtick-pie ring radius, screen px — a small orbit around the cursor
/// (about three cursor-widths across), not a menu. Shared with the
/// editor's test accessor so a suite aims where the drawing puts a disc.
pub const PIE_R: f32 = 21.0;

/// A source disc on the ring: twice a socket nodule's diameter
/// (`editor::SOCKET_R` = 5 → nodule ⌀10 → disc r 10).
pub const PIE_DISC_R: f32 = 10.0;

#[derive(Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Region {
    pub id: u64,
    pub name: String,
    pub color: [u8; 3],
    /// The nodes the wash is derived from. One node, one region — joining
    /// this region leaves any other, same rule as backdrops.
    #[serde(default)]
    pub members: Vec<u64>,
    /// Common sources: node ids the backtick pie offers while the view is
    /// inside this region. A source is usually a member, but nothing
    /// requires it.
    #[serde(default)]
    pub sources: Vec<u64>,
}

/// Create a region over `ids`, named and tinted by rotation.
pub fn create(graph: &mut EditorGraph, ids: &[u64]) -> u64 {
    let id = graph.next_id;
    graph.next_id += 1;
    let n = graph.regions.len();
    graph.regions.push(Region {
        id,
        name: format!("region {}", n + 1),
        color: REGION_COLORS[n % REGION_COLORS.len()],
        members: Vec::new(),
        sources: Vec::new(),
    });
    adopt(graph, id, ids);
    id
}

/// The selection joins `rid` and leaves any other region.
pub fn adopt(graph: &mut EditorGraph, rid: u64, ids: &[u64]) {
    for r in &mut graph.regions {
        if r.id == rid {
            for id in ids {
                if !r.members.contains(id) {
                    r.members.push(*id);
                }
            }
        } else {
            r.members.retain(|m| !ids.contains(m));
        }
    }
    prune(graph);
}

/// Drop dead members and sources; a region emptied of members disappears
/// (its name has nothing left to name).
pub fn prune(graph: &mut EditorGraph) {
    graph.global_sources.retain(|s| graph.nodes.contains_key(s));
    let regions = std::mem::take(&mut graph.regions);
    graph.regions = regions
        .into_iter()
        .filter_map(|mut r| {
            r.members.retain(|m| graph.nodes.contains_key(m));
            r.sources.retain(|s| graph.nodes.contains_key(s));
            (!r.members.is_empty()).then_some(r)
        })
        .collect();
}

/// Register `node` as one of `rid`'s common sources (idempotent) — what
/// the backtick add-pie commits.
pub fn add_source(graph: &mut EditorGraph, rid: u64, node: u64) {
    if let Some(r) = graph.regions.iter_mut().find(|r| r.id == rid) {
        if !r.sources.contains(&node) {
            r.sources.push(node);
        }
    }
}

/// Register `node` as a GLOBAL common source (idempotent) — what the
/// plain backtick pie offers, wherever you are in the graph.
pub fn add_global_source(graph: &mut EditorGraph, node: u64) {
    if !graph.global_sources.contains(&node) {
        graph.global_sources.push(node);
    }
}

pub fn region_of(graph: &EditorGraph, node: u64) -> Option<u64> {
    graph.regions.iter().find(|r| r.members.contains(&node)).map(|r| r.id)
}

/// The wash, in graph space: a SHRINK WRAP around the members — the convex
/// hull of their footprints, stood off by [`PAD`] with rounded corners.
/// Houdini's fit, not the backdrop blob's: one smooth lobe, no internal
/// gaps, because a region is orientation at arm's length and a concave
/// silhouette at that scale is just noise. Derived, never stored.
pub fn wash(graph: &EditorGraph, r: &Region) -> Option<Vec<Pos2>> {
    let mut pts: Vec<Pos2> = Vec::new();
    for m in &r.members {
        if let Some(n) = graph.node(*m) {
            // A wide bar's footprint is the bar, not a node-sized square —
            // otherwise a harp pokes through its own region's wash.
            let (half, dx) = match crate::breakout::wide_rect(&n.kind) {
                Some((h, dx)) => ((h + Vec2::splat(20.0)) * n.size, dx * n.size),
                None => (Vec2::splat(90.0 * n.size), 0.0),
            };
            let c = n.pos + Vec2::new(dx, 0.0);
            for (sx, sy) in [(-1.0f32, -1.0f32), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                pts.push(c + Vec2::new(sx * half.x, sy * half.y));
            }
        }
    }
    let hull = convex_hull(&mut pts);
    (hull.len() >= 3).then(|| offset_rounded(&hull, PAD))
}

/// Bounding box of the wash (framing, tests). The wash is the truth; this
/// is its extent.
pub fn bounds(graph: &EditorGraph, r: &Region) -> Option<Rect> {
    let w = wash(graph, r)?;
    let mut bb = Rect::NOTHING;
    for p in w {
        bb.extend_with(p);
    }
    bb.is_positive().then_some(bb)
}

/// Andrew's monotone chain, counter-clockwise in a y-down space. Sorts
/// `pts` in place; duplicates and collinear runs fall out.
fn convex_hull(pts: &mut Vec<Pos2>) -> Vec<Pos2> {
    pts.sort_by(|a, b| a.x.total_cmp(&b.x).then(a.y.total_cmp(&b.y)));
    pts.dedup();
    if pts.len() < 3 {
        return pts.clone();
    }
    let cross = |o: Pos2, a: Pos2, b: Pos2| {
        (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
    };
    let mut hull: Vec<Pos2> = Vec::with_capacity(pts.len() * 2);
    for pass in 0..2 {
        let start = hull.len() + 1;
        let it: Box<dyn Iterator<Item = &Pos2>> =
            if pass == 0 { Box::new(pts.iter()) } else { Box::new(pts.iter().rev()) };
        for &p in it {
            while hull.len() > start
                && cross(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0
            {
                hull.pop();
            }
            hull.push(p);
        }
        hull.pop();
    }
    hull
}

/// Offset a convex polygon outward by `pad`, corners rounded with sampled
/// arcs — the Minkowski sum with a disc, which is what "stood off by PAD"
/// means when the silhouette must stay smooth.
fn offset_rounded(hull: &[Pos2], pad: f32) -> Vec<Pos2> {
    let n = hull.len();
    let centroid = hull
        .iter()
        .fold(Vec2::ZERO, |a, p| a + p.to_vec2())
        / n as f32;
    let mut out = Vec::with_capacity(n * 6);
    for i in 0..n {
        let v = hull[i];
        let (prev, next) = (hull[(i + n - 1) % n], hull[(i + 1) % n]);
        let out_normal = |a: Pos2, b: Pos2| {
            let e = (b - a).normalized();
            let m = Vec2::new(e.y, -e.x);
            // Whichever perpendicular points away from the middle.
            if (a + m * 1.0 - centroid.to_pos2()).length_sq()
                >= (a - m * 1.0 - centroid.to_pos2()).length_sq()
            {
                m
            } else {
                -m
            }
        };
        let n0 = out_normal(prev, v);
        let n1 = out_normal(v, next);
        let (a0, a1) = (n0.y.atan2(n0.x), n1.y.atan2(n1.x));
        let mut sweep = a1 - a0;
        while sweep < 0.0 {
            sweep += std::f32::consts::TAU;
        }
        while sweep >= std::f32::consts::TAU {
            sweep -= std::f32::consts::TAU;
        }
        let steps = (sweep / 0.45).ceil().max(1.0) as usize;
        for k in 0..=steps {
            let a = a0 + sweep * k as f32 / steps as f32;
            out.push(v + Vec2::new(a.cos(), a.sin()) * pad);
        }
    }
    out
}

/// Point-in-convex-polygon (the wash is convex by construction).
pub fn wash_contains(poly: &[Pos2], p: Pos2) -> bool {
    if poly.len() < 3 {
        return false;
    }
    let mut sign = 0.0f32;
    for i in 0..poly.len() {
        let (a, b) = (poly[i], poly[(i + 1) % poly.len()]);
        let c = (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
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

/// Shoelace area, for the smallest-wash tiebreak.
pub fn wash_area(poly: &[Pos2]) -> f32 {
    let mut a = 0.0;
    for i in 0..poly.len() {
        let (p, q) = (poly[i], poly[(i + 1) % poly.len()]);
        a += p.x * q.y - q.x * p.y;
    }
    (a * 0.5).abs()
}

/// The region whose wash holds `at` — what the backtick pie offers and the
/// corner HUD names. Overlapping washes tiebreak to the SMALLEST, so an
/// enclosing region never shadows the one you are actually working in.
pub fn active(graph: &EditorGraph, at: Pos2) -> Option<u64> {
    graph
        .regions
        .iter()
        .filter_map(|r| {
            let w = wash(graph, r)?;
            wash_contains(&w, at).then(|| (r.id, wash_area(&w)))
        })
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(id, _)| id)
}

// ---- pie geometry --------------------------------------------------------
// Blender-style: slice 0 sits at 12 o'clock and they run clockwise. The
// same maths places the labels and picks the click, so what you read is
// what you hit.

/// Label/slice centre for slice `i` of `n`, `radius` px from `c`.
pub fn pie_slice_pos(c: Pos2, i: usize, n: usize, radius: f32) -> Pos2 {
    let a = std::f32::consts::TAU * i as f32 / n.max(1) as f32
        - std::f32::consts::FRAC_PI_2;
    c + Vec2::new(a.cos(), a.sin()) * radius
}

/// A ring of labelled slices, the ops pie's look: each label in a small
/// box on the orbit, the hot one ringed. `ring` is the (animated) radius.
pub fn draw_label_pie(
    painter: &egui::Painter,
    c: Pos2,
    labels: &[String],
    hot: Option<usize>,
    ring: f32,
) {
    use crate::theme;
    painter.circle_filled(c, 2.0, theme::node_label());
    for (i, name) in labels.iter().enumerate() {
        let at = pie_slice_pos(c, i, labels.len(), ring);
        let is_hot = hot == Some(i);
        let col = if is_hot { theme::text_hot() } else { theme::text() };
        let galley = painter.layout_no_wrap(name.clone(), egui::FontId::proportional(12.0), col);
        let r = Rect::from_center_size(at, galley.size() + Vec2::new(10.0, 4.0));
        painter.rect(
            r,
            4.0,
            theme::graph_bg(),
            egui::Stroke::new(if is_hot { 2.0 } else { 1.0 }, col.gamma_multiply(0.7)),
            egui::StrokeKind::Outside,
        );
        painter.galley(r.center() - galley.size() * 0.5, galley, col);
    }
}

/// Which slice `p` lands in, if outside the dead centre. The zone is
/// small: the orbit is only [`PIE_R`] out, and a disc's inner edge must
/// still be pickable.
pub fn pie_pick(c: Pos2, p: Pos2, n: usize) -> Option<usize> {
    if n == 0 || (p - c).length() < 8.0 {
        return None;
    }
    let v = p - c;
    let a = v.y.atan2(v.x) + std::f32::consts::FRAC_PI_2; // 0 at 12 o'clock
    let a = (a + std::f32::consts::TAU) % std::f32::consts::TAU;
    let step = std::f32::consts::TAU / n as f32;
    // Slice centres sit ON the spoke angles; the boundary is halfway between.
    Some((((a + step * 0.5) / step) as usize) % n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::graph::NodeKind;

    fn mult() -> NodeKind {
        NodeKind::Mult(engine::graph::MultParams::default())
    }

    #[test]
    fn adopt_moves_a_node_between_regions() {
        let mut g = EditorGraph::default();
        let a = g.add(mult(), egui::pos2(0.0, 0.0));
        let b = g.add(mult(), egui::pos2(400.0, 0.0));
        let r1 = create(&mut g, &[a]);
        let r2 = create(&mut g, &[b]);
        adopt(&mut g, r2, &[a]);
        assert_eq!(region_of(&g, a), Some(r2));
        // r1 emptied and pruned away.
        assert!(!g.regions.iter().any(|r| r.id == r1));
    }

    #[test]
    fn active_prefers_the_smallest_wash() {
        let mut g = EditorGraph::default();
        let a = g.add(mult(), egui::pos2(0.0, 0.0));
        let b = g.add(mult(), egui::pos2(600.0, 0.0));
        let big = create(&mut g, &[a, b]);
        let small = create(&mut g, &[a]);
        // b stayed only in big (adopt pulled a out of it).
        assert_eq!(region_of(&g, b), Some(big));
        assert_eq!(active(&g, egui::pos2(0.0, 0.0)), Some(small));
        assert_eq!(active(&g, egui::pos2(600.0, 0.0)), Some(big));
        assert_eq!(active(&g, egui::pos2(4000.0, 0.0)), None);
    }

    #[test]
    fn pie_pick_matches_slice_positions() {
        let c = egui::pos2(100.0, 100.0);
        for n in 1..=8 {
            for i in 0..n {
                let p = pie_slice_pos(c, i, n, 40.0);
                assert_eq!(pie_pick(c, p, n), Some(i), "slice {i} of {n}");
            }
        }
        assert_eq!(pie_pick(c, c, 4), None); // dead centre
    }

    #[test]
    fn a_hidden_chip_carries_the_fat_cable() {
        use engine::graph::{ChannelSelectParams, PortKind};
        let mut g = EditorGraph::default();
        let rd = g.add(NodeKind::read(0, "plate.exr"), egui::pos2(0.0, -200.0));
        if let Some(NodeKind::Read { multi, .. }) = g.node_mut(rd).map(|n| &mut n.kind) {
            *multi = true;
        }
        // The pie's spawn: a Dot hidden-wired to the multi Read.
        let chip = g.add(
            NodeKind::dot(engine::graph::PortKind::Image),
            egui::pos2(0.0, 0.0),
        );
        g.connect_hidden(chip, 0, rd);
        g.sync_dot_kinds();
        let chip_kind = g.node(chip).unwrap().kind.clone();
        assert_eq!(chip_kind.output_kind(), PortKind::Channels);
        assert_eq!(chip_kind.input_kind(0), PortKind::Channels);
        // It plugs a Channels consumer DIRECTLY — no bridge node minted.
        let cs = g.add(
            NodeKind::ChannelSelect(ChannelSelectParams {
                source: 7,
                layer: "crypto00".into(),
            }),
            egui::pos2(0.0, 200.0),
        );
        let before = g.nodes.len();
        crate::editor::connect_or_bridge(&mut g, chip, 0, cs, 0, 1.0);
        assert_eq!(g.nodes.len(), before, "a channels chip must not bridge");
        assert_eq!(g.input_of(cs, 0), Some(chip));
        // The select resolves its Read THROUGH the hidden chip, and the
        // engine sees it lowered to a per-layer Read like any other.
        assert_eq!(g.channel_source(cs), Some(rd));
        let eg = g.to_engine_graph();
        match eg.nodes.get(&cs) {
            Some(NodeKind::Read { layer: Some(l), source, .. }) => {
                assert_eq!(l, "crypto00");
                assert_eq!(*source, 7);
            }
            other => panic!("select lowered to {other:?}"),
        }
    }

    /// View-scoped decode rides on this: every viewer slot reaches the
    /// engine graph LOWERED — a harp string as its hidden kid, a dot as
    /// what feeds it — and the rooted walk from those slots finds exactly
    /// the readers they sample, while the whole-graph walk still finds all.
    #[test]
    fn viewer_slots_lower_into_decode_roots() {
        let mut g = EditorGraph::default();
        let rd = g.add(NodeKind::read(0, "plate.exr"), egui::pos2(0.0, -200.0));
        if let Some(NodeKind::Read { multi, .. }) = g.node_mut(rd).map(|n| &mut n.kind) {
            *multi = true;
        }
        let (b, _r) = crate::breakout::spawn_pair(
            &mut g,
            egui::pos2(0.0, 0.0),
            vec!["diff".into(), "spec".into(), "gi".into()],
        );
        g.connect(b, 0, rd);
        crate::breakout::sync(&mut g);
        let kids = match g.node(b).map(|n| &n.kind) {
            Some(NodeKind::Breakout(p)) => p.kids.clone(),
            _ => panic!("breakout lost"),
        };
        for (i, k) in kids.iter().enumerate() {
            if let Some(NodeKind::ChannelSelect(p)) = g.node_mut(*k).map(|n| &mut n.kind) {
                p.source = 10 + i;
            }
        }
        // A on string 1, B on string 2 through a dot.
        let dot = g.add(
            NodeKind::dot(engine::graph::PortKind::Image),
            egui::pos2(300.0, 100.0),
        );
        g.connect_out(dot, 0, b, 2);
        g.view = Some(b);
        g.view_out = 1;
        g.view_b = Some(dot);
        let eg = g.to_engine_graph();
        assert_eq!(eg.view, Some(kids[1]));
        assert_eq!(eg.hot_roots, vec![kids[2]]);
        let roots: Vec<u64> = eg.view.into_iter().chain(eg.hot_roots.iter().copied()).collect();
        let mut scoped: Vec<u64> =
            eg.reader_instances_from(&roots).iter().map(|i| i.node).collect();
        scoped.sort_unstable();
        assert_eq!(scoped, vec![kids[1], kids[2]], "only what the slots sample");
        // The rig itself still holds every string.
        let all: std::collections::HashSet<u64> =
            eg.reader_instances().iter().map(|i| i.node).collect();
        assert!(kids.iter().all(|k| all.contains(k)), "the whole harp stays instanced");
    }

    #[test]
    fn read_names_guess_sensibly() {
        use crate::editor::guess_read_label as g;
        assert_eq!(g("data/crypto/frame.0000.exr"), "crypto");
        assert_eq!(g("shots/sh010/sh010_cryptomatte_v003.1001.exr"), "sh010_cryptomatte");
        assert_eq!(g("renders/bty_key_v012/frame.####.exr"), "bty_key");
        assert_eq!(g("plates/2k_zip_bg/frame.0001.exr"), "2k_zip_bg");
    }

    #[test]
    fn hidden_wires_are_presentation_only() {
        let mut g = EditorGraph::default();
        let a = g.add(mult(), egui::pos2(0.0, 0.0));
        let d = g.add(
            NodeKind::dot(engine::graph::PortKind::Image),
            egui::pos2(0.0, 200.0),
        );
        let m = g.add(mult(), egui::pos2(0.0, 400.0));
        g.connect_hidden(d, 0, a);
        g.connect(m, 0, d);
        assert!(g.is_hidden(d, 0));
        // The engine resolves through the dot and sees an ordinary edge —
        // the flag never reaches it.
        let eg = g.to_engine_graph();
        assert_eq!(eg.edges.get(&(m, 0)).copied(), Some(a));
        // A plain connect over the same input SURFACES the wire.
        g.connect(d, 0, a);
        assert!(!g.is_hidden(d, 0));
        // Removing the feed sweeps the flag with it.
        g.connect_hidden(d, 0, a);
        g.disconnect_input(d, 0);
        assert!(!g.is_hidden(d, 0));
    }
}
