//! Lowering a multi-output node into single-output ones.
//!
//! A [`NodeKind::LumaKey`] can carry several independent bands, each one a
//! separate matte on a separate output port. Nothing downstream of this
//! module knows that. Codegen, the compositor, the parameter layout, the
//! render cache and the disk cache all still see a graph in which every
//! node has exactly one output — because [`expand`] rewrites a multi-band
//! key into one ordinary single-band key per band that something actually
//! consumes, before any of them look.
//!
//! This is deliberately the same trick two other lanes already took:
//!
//! - a **tool** node is a subgraph that expands into the flat graph before
//!   the engine sees one, so `NodeKind` never grew an arm for it;
//! - a **Dot** is a routing anchor that wires resolve THROUGH and that never
//!   reaches codegen at all.
//!
//! The alternative was threading an output index through `emit`, the stage
//! walk, `StageId`, the param layout and every cache key — about 50 call
//! sites in `codegen.rs` alone, in a file another lane owns, to express
//! something that is exactly equivalent to "these are N keys sharing an
//! input". The equivalence is real, so the cheap version is the honest one.
//!
//! **Cost.** One extra node per CONSUMED band; an unwired band costs
//! nothing at all. Each expanded key re-evaluates the shared input subtree,
//! which is what already happens for any diamond in this graph (`emit` has
//! no common-subexpression pass), and exactly what it would cost to author
//! the same thing as N separate LumaKey nodes by hand — which is what you
//! have to do today.

use crate::graph::{Graph, NodeId, NodeKind};
use std::collections::HashMap;

/// Rewrite every multi-band [`NodeKind::LumaKey`] in `g` into one
/// single-band key per consumed band, and clear the `edge_out` entries that
/// pointed at them. A graph in which no node has more than one output comes
/// out byte-identical.
///
/// Band 0 keeps the ORIGINAL node id. That is not an implementation detail
/// you can change freely: the id is what animation curves, per-node formats,
/// pinned property panels and the disk cache are all keyed by, so moving it
/// would silently orphan all of them.
///
/// New ids are minted above the highest in use, in band order, over nodes
/// visited in sorted id order — deterministic, because the topology hash and
/// the render cache key are computed from the result.
pub fn expand(g: &mut Graph) {
    let mut multi: Vec<NodeId> = g
        .nodes
        .iter()
        .filter(|(_, k)| k.output_count() > 1)
        .map(|(id, _)| *id)
        .collect();
    if multi.is_empty() {
        return;
    }
    multi.sort_unstable();
    let mut next = g.nodes.keys().copied().max().unwrap_or(0) + 1;

    for src in multi {
        // Which bands anything asks for, and who asks. The view target
        // counts as a consumer — viewing band 2 has to work even when
        // nothing is wired to it.
        let mut consumers: HashMap<u32, Vec<(NodeId, usize)>> = HashMap::new();
        for ((dst, input), s) in g.edges.iter() {
            if *s != src {
                continue;
            }
            let out = g.edge_out.get(&(*dst, *input)).copied().unwrap_or(0);
            consumers.entry(out).or_default().push((*dst, *input));
        }
        let viewed = (g.view == Some(src)).then_some(g.view_out);
        if let Some(v) = viewed {
            consumers.entry(v).or_default();
        }

        let Some(NodeKind::LumaKey(params)) = g.nodes.get(&src).cloned() else {
            continue;
        };
        let bands = params.bands.len();

        // Band 0 stays on the original id, always — even when nothing
        // consumes it. Dropping it would strand the node's animation.
        g.nodes.insert(src, NodeKind::LumaKey(params.only(0)));

        let mut bands_wanted: Vec<u32> = consumers.keys().copied().collect();
        bands_wanted.sort_unstable();
        for band in bands_wanted {
            if band == 0 {
                continue;
            }
            // A band index past the end reads as the last band, matching
            // `LumaKeyParams::band` — an edge left behind by a param edit
            // that removed a band gives you a matte, not a black frame.
            let b = (band as usize).min(bands.saturating_sub(1));
            let id = next;
            next += 1;
            g.nodes.insert(id, NodeKind::LumaKey(params.only(b)));
            // Same inputs as the original: the bands are N reads of one
            // image, so they share the whole upstream subtree.
            for i in 0..params_input_count() {
                if let Some(up) = g.input_of(src, i) {
                    let up_out = g.edge_out.get(&(src, i)).copied().unwrap_or(0);
                    g.connect_out(id, i, up, up_out);
                }
            }
            // Side tables the clone has to inherit or it behaves differently
            // from the node it came from.
            if g.disabled.contains(&src) {
                g.disabled.insert(id);
            }
            if let Some(f) = g.formats.get(&src).copied() {
                g.formats.insert(id, f);
            }
            for (dst, input) in consumers.get(&band).into_iter().flatten() {
                g.connect_out(*dst, *input, id, 0);
            }
            if viewed == Some(band) {
                g.view = Some(id);
                g.view_out = 0;
            }
        }
        // Every edge off this node now points at a single-output node.
        if viewed == Some(0) {
            g.view_out = 0;
        }
    }
    // Every LumaKey is single-band now, so an `edge_out` entry pointing at
    // one is stale. Anything still naming a genuinely multi-output source
    // is left alone: a kind this pass does not know how to lower is a bug
    // to find later, not one to silently rewire now.
    let stale: Vec<(NodeId, usize)> = g
        .edge_out
        .keys()
        .filter(|k| g.edges.get(k).map(|s| g.outputs_of(*s)).unwrap_or(1) <= 1)
        .copied()
        .collect();
    for k in stale {
        g.edge_out.remove(&k);
    }
}

/// A LumaKey's input count. Written as a function rather than the literal 1
/// so that giving the node a mask input later cannot silently leave the
/// clones unwired.
fn params_input_count() -> usize {
    NodeKind::LumaKey(Default::default()).input_count()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{LumaBand, LumaKeyParams};

    fn band(lo: f32, hi: f32) -> LumaBand {
        LumaBand { lo, hi, ..Default::default() }
    }

    /// read -> key(3 bands) -> three Grades, one per band.
    fn three_band_graph() -> Graph {
        let mut g = Graph::default();
        g.nodes.insert(0, NodeKind::read(0, ""));
        g.nodes.insert(
            1,
            NodeKind::LumaKey(LumaKeyParams {
                bands: vec![band(0.0, 0.3), band(0.3, 0.6), band(0.6, 1.0)],
                ranges: None,
            }),
        );
        g.edges.insert((1, 0), 0);
        for (i, out) in [(10u64, 0u32), (11, 1), (12, 2)] {
            g.nodes.insert(i, NodeKind::Grade(Default::default()));
            g.connect_out(i, 0, 1, out);
        }
        g.view = Some(10);
        g
    }

    #[test]
    fn a_single_band_graph_is_untouched() {
        let mut g = Graph::default();
        g.nodes.insert(0, NodeKind::read(0, ""));
        // One band, spelled out: a fresh key is a two-band partition (a
        // threshold is below + above), so `Default` is not the case here.
        // This is the shape `expand` itself produces (`LumaKeyParams::only`).
        let one = LumaKeyParams { bands: vec![band(0.2, 0.8)], ranges: None };
        g.nodes.insert(1, NodeKind::LumaKey(one));
        g.edges.insert((1, 0), 0);
        g.view = Some(1);
        let before = g.clone();
        expand(&mut g);
        assert_eq!(g, before);
    }

    #[test]
    fn each_consumed_band_becomes_its_own_node_sharing_the_input() {
        let mut g = three_band_graph();
        expand(&mut g);
        // Three keys now, all single-band, all reading node 0.
        let keys: Vec<NodeId> = {
            let mut v: Vec<_> = g
                .nodes
                .iter()
                .filter(|(_, k)| matches!(k, NodeKind::LumaKey(_)))
                .map(|(id, _)| *id)
                .collect();
            v.sort_unstable();
            v
        };
        assert_eq!(keys.len(), 3, "{keys:?}");
        for k in &keys {
            assert_eq!(g.outputs_of(*k), 1, "node {k} still multi-output");
            assert_eq!(g.input_of(*k, 0), Some(0), "node {k} lost its input");
        }
        // Band 0 kept the original id — animation and caches hang off it.
        assert!(keys.contains(&1));
        // Each Grade reads a DIFFERENT key, and each key is the right band.
        let mut seen = Vec::new();
        for (grade, want_lo) in [(10u64, 0.0f32), (11, 0.3), (12, 0.6)] {
            let src = g.input_of(grade, 0).expect("grade lost its input");
            assert!(!seen.contains(&src), "two grades share key {src}");
            seen.push(src);
            let Some(NodeKind::LumaKey(k)) = g.nodes.get(&src) else {
                panic!("grade {grade} does not read a LumaKey")
            };
            assert_eq!(k.bands.len(), 1);
            assert!(
                (k.band(0).lo - want_lo).abs() < 1e-6,
                "grade {grade} got band lo {} not {want_lo}",
                k.band(0).lo
            );
        }
        // No edge names an output any more.
        assert!(g.edge_out.is_empty(), "{:?}", g.edge_out);
    }

    #[test]
    fn an_unwired_band_costs_nothing() {
        let mut g = three_band_graph();
        // Nobody reads band 1 any more.
        g.edges.remove(&(11, 0));
        g.edge_out.remove(&(11, 0));
        expand(&mut g);
        let keys = g.nodes.values().filter(|k| matches!(k, NodeKind::LumaKey(_))).count();
        assert_eq!(keys, 2, "an unconsumed band was still expanded");
    }

    #[test]
    fn viewing_a_band_follows_it_to_its_new_node() {
        let mut g = three_band_graph();
        g.view = Some(1);
        g.view_out = 2;
        expand(&mut g);
        assert_eq!(g.view_out, 0);
        let v = g.view.expect("view target vanished");
        let Some(NodeKind::LumaKey(k)) = g.nodes.get(&v) else { panic!("view is not a key") };
        assert!((k.band(0).lo - 0.6).abs() < 1e-6, "viewing the wrong band");
    }

    #[test]
    fn two_consumers_of_one_band_share_a_single_node() {
        let mut g = three_band_graph();
        g.nodes.insert(13, NodeKind::Grade(Default::default()));
        g.connect_out(13, 0, 1, 2); // also band 2, like node 12
        expand(&mut g);
        assert_eq!(
            g.input_of(12, 0),
            g.input_of(13, 0),
            "the same band was expanded twice"
        );
        let keys = g.nodes.values().filter(|k| matches!(k, NodeKind::LumaKey(_))).count();
        assert_eq!(keys, 3);
    }

    #[test]
    fn the_clone_inherits_disabled_and_format() {
        let mut g = three_band_graph();
        g.disabled.insert(1);
        g.formats.insert(1, (640, 480));
        expand(&mut g);
        for (id, k) in g.nodes.iter() {
            if matches!(k, NodeKind::LumaKey(_)) {
                assert!(g.disabled.contains(id), "clone {id} is not bypassed");
                assert_eq!(g.formats.get(id).copied(), Some((640, 480)), "clone {id}");
            }
        }
    }

    /// An edge naming a band that a param edit has since deleted must land
    /// on the last band rather than dangling.
    #[test]
    fn an_out_of_range_band_clamps_instead_of_dangling() {
        let mut g = three_band_graph();
        if let Some(NodeKind::LumaKey(k)) = g.nodes.get_mut(&1) {
            k.bands.truncate(2); // node 12 still asks for band 2
        }
        expand(&mut g);
        let src = g.input_of(12, 0).expect("edge dangled");
        let Some(NodeKind::LumaKey(k)) = g.nodes.get(&src) else { panic!() };
        assert!((k.band(0).lo - 0.3).abs() < 1e-6, "did not clamp to the last band");
    }

    /// Deterministic ids: the render cache key is computed from the result,
    /// so expanding the same graph twice has to give the same graph.
    #[test]
    fn expansion_is_deterministic() {
        let (mut a, mut b) = (three_band_graph(), three_band_graph());
        expand(&mut a);
        expand(&mut b);
        assert_eq!(a, b);
        assert_eq!(a.topology_hash(), b.topology_hash());
    }

    /// Expanding is idempotent — the app lowers on every graph change and
    /// must not mint a new node each time.
    #[test]
    fn expanding_twice_changes_nothing() {
        let mut g = three_band_graph();
        expand(&mut g);
        let once = g.clone();
        expand(&mut g);
        assert_eq!(g, once);
    }

    /// The band index is part of the shader, so two graphs differing only
    /// in which band a Grade reads must not share a compiled pipeline.
    #[test]
    fn the_band_an_edge_takes_is_structural() {
        let mut a = three_band_graph();
        let mut b = three_band_graph();
        b.connect_out(10, 0, 1, 1); // node 10 reads band 1 instead of band 0
        assert_ne!(a.topology_hash(), b.topology_hash(), "before expansion");
        expand(&mut a);
        expand(&mut b);
        assert_ne!(a.topology_hash(), b.topology_hash(), "after expansion");
    }
}
