//! A Layer stack's per-layer outputs, lowered.
//!
//! With [`LayerParams::outputs`] on, a stack has `1 + n` outputs: the comp,
//! then each layer ALONE. The engine only ever sees single-output nodes
//! (see [`crate::lumabands`]), so a consumed output `1 + i` becomes a
//! one-layer stack of its own, wired to layer `i`'s input. That clone is
//! still a Layer, so the time walk gives its input the layer's own map
//! (`NodeKind::input_time_map`: start and speed) and the trim gate the app
//! stamped rides along — the layer's time operations, nothing of the comp.
//!
//! Cost: one node per CONSUMED layer output; an unwired one costs nothing.
//! A layer read both through the comp and through its own output is two
//! branches of time (the clone has its own id), so its plate decodes twice.

use crate::graph::{Graph, LayerParams, NodeId, NodeKind};
use std::collections::HashMap;

/// Rewrite every Layer stack with outputs on into the comp (on the original
/// id, so its animation and formats stay put) plus one single-layer stack
/// per consumed layer output. A graph with no such stack is untouched.
pub fn expand(g: &mut Graph) {
    let mut multi: Vec<NodeId> = g
        .nodes
        .iter()
        .filter(|(_, k)| matches!(k, NodeKind::Layer(p) if p.outputs))
        .map(|(id, _)| *id)
        .collect();
    if multi.is_empty() {
        return;
    }
    multi.sort_unstable();
    let mut next = g.nodes.keys().copied().max().unwrap_or(0) + 1;
    for src in multi {
        let Some(NodeKind::Layer(params)) = g.nodes.get(&src).cloned() else { continue };
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
        // The host is the comp alone from here on.
        g.nodes.insert(src, NodeKind::Layer(LayerParams { outputs: false, ..params.clone() }));
        let n = params.layers.len().min(crate::graph::LAYER_MAX);
        let mut wanted: Vec<u32> = consumers.keys().copied().collect();
        wanted.sort_unstable();
        for out in wanted {
            if out == 0 {
                continue;
            }
            let id = next;
            next += 1;
            let layer = out as usize - 1;
            // An output past the stack (a layer removed under a wire that
            // the editor did not drop) is transparent, never another layer.
            if layer < n {
                g.nodes.insert(id, NodeKind::Layer(params.only(layer)));
                if let Some(up) = g.input_of(src, layer) {
                    let up_out = g.edge_out.get(&(src, layer)).copied().unwrap_or(0);
                    g.connect_out(id, 0, up, up_out);
                }
            } else {
                g.nodes.insert(id, NodeKind::Layer(LayerParams::default()));
            }
            if g.disabled.contains(&src) {
                g.disabled.insert(id);
            }
            // The layer's OWN format (the picture on its wire), not the
            // stack's canvas; the stack's only when the wire has none.
            let own = if layer < n { g.input_of(src, layer) } else { None };
            if let Some(f) = own.and_then(|u| g.format_of(u)).or_else(|| g.format_of(src)) {
                g.formats.insert(id, f);
            }
            for (dst, input) in consumers.get(&out).into_iter().flatten() {
                g.connect_out(*dst, *input, id, 0);
            }
            if viewed == Some(out) {
                g.view = Some(id);
                g.view_out = 0;
            }
        }
        if viewed == Some(0) {
            g.view_out = 0;
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{BlendOp, LayerItem, TIME_KEY_ROOT};

    fn stack() -> Graph {
        let mut g = Graph::default();
        g.nodes.insert(1, NodeKind::read(0, ""));
        g.nodes.insert(2, NodeKind::read(1, ""));
        g.nodes.insert(
            100,
            NodeKind::Layer(LayerParams {
                layers: vec![
                    LayerItem { start: 5, speed: 2.0, opacity: 0.3, op: BlendOp::Screen, ..Default::default() },
                    LayerItem::default(),
                ],
                outputs: true,
            }),
        );
        g.edges.insert((100, 0), 1);
        g.edges.insert((100, 1), 2);
        g
    }

    /// Viewing output 1 is layer 1 alone: a one-layer stack on the layer's
    /// wire, opaque and Over, carrying the layer's time map — and nothing of
    /// the other layer reaches it.
    #[test]
    fn a_layer_output_is_the_layer_alone_on_its_own_time() {
        let mut g = stack();
        g.view = Some(100);
        g.view_out = 1;
        expand(&mut g);
        let v = g.view.unwrap();
        assert_ne!(v, 100);
        let Some(NodeKind::Layer(p)) = g.nodes.get(&v) else { panic!("not a stack") };
        assert_eq!(p.layers.len(), 1);
        assert_eq!((p.layers[0].opacity, p.layers[0].op), (1.0, BlendOp::Over));
        assert_eq!(g.input_of(v, 0), Some(1));
        // What the viewer decodes (the rig still holds the whole graph).
        let inst = g.reader_instances_from(&[v]);
        assert_eq!(inst.len(), 1, "only layer 1's plate is read: {inst:?}");
        let k = crate::graph::input_time_key(TIME_KEY_ROOT, v, 0);
        let i = inst.iter().find(|i| i.key == k).expect("the layer's own time branch");
        assert_eq!(i.node, 1);
        // source = (comp − 5) × 2
        assert_eq!(i.map.apply(10), 10);
        assert_eq!(i.map.apply(7), 4);
        // The host is the plain comp again.
        assert!(matches!(g.nodes.get(&100), Some(NodeKind::Layer(p)) if !p.outputs));
    }

    /// A consumer wired to output 2 gets layer 2; output 0 stays the comp;
    /// nothing unconsumed is minted.
    #[test]
    fn consumers_get_their_layer_and_the_comp_stays() {
        let mut g = stack();
        g.nodes.insert(200, NodeKind::Grade(Default::default()));
        g.nodes.insert(201, NodeKind::Grade(Default::default()));
        g.connect_out(200, 0, 100, 2);
        g.connect_out(201, 0, 100, 0);
        g.view = Some(200);
        let before = g.nodes.len();
        expand(&mut g);
        assert_eq!(g.nodes.len(), before + 1, "one clone, for the one consumed layer");
        let c = g.input_of(200, 0).unwrap();
        assert_eq!(g.input_of(c, 0), Some(2));
        assert_eq!(g.input_of(201, 0), Some(100));
        assert!(g.edge_out.is_empty());
    }

    /// A layer output lives in its LAYER's format (the picture on its
    /// wire), not the stack's canvas.
    #[test]
    fn a_layer_output_takes_its_layers_format() {
        let mut g = stack();
        g.formats.insert(1, (640, 360));
        g.formats.insert(2, (2048, 1080));
        g.formats.insert(100, (2048, 1080));
        g.view = Some(100);
        g.view_out = 1;
        expand(&mut g);
        let v = g.view.unwrap();
        assert_eq!(g.format_of(v), Some((640, 360)));
        assert_eq!(g.format_of(100), Some((2048, 1080)), "the comp keeps the canvas");
    }

    /// Off, the pass does nothing at all.
    #[test]
    fn off_is_untouched() {
        let mut g = stack();
        if let Some(NodeKind::Layer(p)) = g.nodes.get_mut(&100) {
            p.outputs = false;
        }
        g.view = Some(100);
        let h = g.topology_hash();
        expand(&mut g);
        assert_eq!(g.topology_hash(), h);
        assert_eq!(g.nodes.len(), 3);
    }
}
