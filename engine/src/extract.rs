//! Lowering an Extract into the Mask nodes it stands for.
//!
//! A [`NodeKind::Extract`] is a convenience: one image on input 0, a matte
//! on each input after it, and output k is the image multiplied by matte
//! k's coverage — exactly what N separate Mask nodes sharing an input do.
//! So that is what [`expand`] turns it into, before codegen, the
//! compositor or any cache key sees the graph (the [`crate::lumabands`]
//! precedent). An Extract costs exactly the Masks it replaces, and an
//! output nothing consumes costs nothing.

use crate::graph::{Graph, NodeId, NodeKind};
use std::collections::HashMap;

/// Rewrite every Extract in `g` into one Mask per consumed output.
///
/// Output 0 keeps the ORIGINAL id (animation, formats and the disk cache
/// are keyed by it); the rest get fresh ids above the highest in use, in
/// output order over sorted Extract ids — deterministic, because the
/// topology hash is computed from the result.
pub fn expand(g: &mut Graph) {
    let mut ids: Vec<NodeId> = g
        .nodes
        .iter()
        .filter(|(_, k)| matches!(k, NodeKind::Extract(_)))
        .map(|(id, _)| *id)
        .collect();
    if ids.is_empty() {
        return;
    }
    ids.sort_unstable();
    let mut next = g.nodes.keys().copied().max().unwrap_or(0) + 1;

    for src in ids {
        let Some(NodeKind::Extract(p)) = g.nodes.get(&src).cloned() else { continue };
        let pairs = p.pairs();
        // Who consumes which output. The view target counts, so viewing
        // output 2 works with nothing wired to it.
        let mut consumers: HashMap<u32, Vec<(NodeId, usize)>> = HashMap::new();
        for ((dst, input), s) in g.edges.iter() {
            if *s == src {
                let out = g.edge_out.get(&(*dst, *input)).copied().unwrap_or(0);
                consumers.entry(out).or_default().push((*dst, *input));
            }
        }
        let viewed = (g.view == Some(src)).then_some(g.view_out);
        if let Some(v) = viewed {
            consumers.entry(v).or_default();
        }
        // The ports feeding it, captured before its own edges are trimmed.
        let ins: Vec<Option<(NodeId, u32)>> =
            (0..=pairs).map(|i| g.input_port(src, i)).collect();

        // Output 0 IS a Mask of (image, matte 0) — inputs 0 and 1 already
        // sit where a Mask reads them. Drop the rest.
        g.nodes.insert(src, NodeKind::Mask);
        for i in 2..=pairs {
            g.edges.remove(&(src, i));
            g.edge_out.remove(&(src, i));
        }

        let mut wanted: Vec<u32> = consumers.keys().copied().collect();
        wanted.sort_unstable();
        for out in wanted {
            // An output past the end (an edge left by shrinking the node)
            // reads as the last one — a picture, not a black frame.
            let k = (out as usize).min(pairs - 1);
            let id = if k == 0 {
                src
            } else {
                let id = next;
                next += 1;
                g.nodes.insert(id, NodeKind::Mask);
                for (slot, from) in [(0, ins[0]), (1, ins[1 + k])] {
                    if let Some((up, up_out)) = from {
                        g.connect_out(id, slot, up, up_out);
                    }
                }
                if g.disabled.contains(&src) {
                    g.disabled.insert(id);
                }
                if let Some(f) = g.formats.get(&src).copied() {
                    g.formats.insert(id, f);
                }
                id
            };
            for (dst, input) in consumers.get(&out).into_iter().flatten() {
                g.connect_out(*dst, *input, id, 0);
            }
            if viewed == Some(out) {
                g.view = Some(id);
                g.view_out = 0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::ExtractParams;

    /// read(1) + three mattes(2,3,4) -> extract(5) -> a grade on each output.
    fn graph() -> Graph {
        let mut g = Graph::default();
        g.nodes.insert(1, NodeKind::read(0, ""));
        for m in 2..=4 {
            g.nodes.insert(m, NodeKind::read(0, ""));
        }
        g.nodes.insert(5, NodeKind::Extract(ExtractParams { n: 3 }));
        g.connect_out(5, 0, 1, 0);
        for (i, m) in (2..=4).enumerate() {
            g.connect_out(5, 1 + i, m, 0);
        }
        for (i, dst) in (10..=12).enumerate() {
            g.nodes.insert(dst, NodeKind::Grade(Default::default()));
            g.connect_out(dst, 0, 5, i as u32);
        }
        g.view = Some(10);
        g
    }

    #[test]
    fn each_output_becomes_a_mask_of_the_image_by_its_matte() {
        let mut g = graph();
        expand(&mut g);
        assert!(!g.nodes.values().any(|k| matches!(k, NodeKind::Extract(_))));
        for (i, dst) in (10..=12).enumerate() {
            let (m, out) = g.input_port(dst, 0).unwrap();
            assert_eq!(out, 0);
            assert!(matches!(g.nodes[&m], NodeKind::Mask));
            assert_eq!(g.input_of(m, 0), Some(1), "output {i} reads the image");
            assert_eq!(g.input_of(m, 1), Some(2 + i as u64), "output {i} reads its matte");
        }
        // Output 0 stays on the original id, and it has exactly two inputs.
        assert_eq!(g.input_of(10, 0), Some(5));
        assert_eq!(g.input_of(5, 2), None);
        assert!(g.edge_out.is_empty());
    }

    #[test]
    fn unconsumed_outputs_cost_nothing_and_a_view_counts() {
        let mut g = graph();
        g.edges.retain(|(d, _), _| *d < 10);
        g.edge_out.clear();
        g.view = Some(5);
        g.view_out = 2;
        let before = g.nodes.len();
        expand(&mut g);
        assert_eq!(g.nodes.len(), before + 1, "only the viewed output is minted");
        let v = g.view.unwrap();
        assert_eq!(g.view_out, 0);
        assert_eq!(g.input_of(v, 1), Some(4));
    }

    #[test]
    fn lowering_is_deterministic() {
        let (mut a, mut b) = (graph(), graph());
        expand(&mut a);
        expand(&mut b);
        assert_eq!(a.topology_hash(), b.topology_hash());
    }
}
