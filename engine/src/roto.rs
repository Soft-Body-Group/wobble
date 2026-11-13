//! Lowering a roto CONTAINER into the nodes it stands for.
//!
//! A Spline with more than one shape, or with combine ops, holds a small
//! graph of its own (`SplineParams::more`/`ops`, edited in the node by the
//! app's `innards`). What it renders is exactly: one plain Spline per
//! shape, one MatteCombine per op, the loose ends unioned, and the result
//! combined with the matte on the node's input by the node's `combine` —
//! so that is what [`expand`] turns it into, before codegen, the
//! compositor or any cache key sees the graph (the `extract` precedent).
//! The shader never learns about containers, and a container costs what
//! the same shapes chained by hand would.
//!
//! The container's id becomes its FINAL node (animation, formats and the
//! disk cache are keyed by it). Minted ids go above the highest in use, in
//! a fixed order over sorted containers — deterministic, because the
//! topology hash is computed from the result — and each minted Spline
//! reads the container's format through `Graph::format_alias`.

use crate::graph::{Graph, MatteCombineParams, MatteOp, NodeId, NodeKind, ROTO_OP};

pub fn expand(g: &mut Graph) {
    let mut ids: Vec<NodeId> = g
        .nodes
        .iter()
        .filter(|(_, k)| matches!(k, NodeKind::Spline(sp) if sp.is_container()))
        .map(|(id, _)| *id)
        .collect();
    if ids.is_empty() {
        return;
    }
    ids.sort_unstable();
    let mut next = g.nodes.keys().copied().max().unwrap_or(0) + 1;
    let mut mint = |g: &mut Graph, k: NodeKind| {
        let id = next;
        next += 1;
        g.nodes.insert(id, k);
        id
    };

    for host in ids {
        let Some(NodeKind::Spline(sp)) = g.nodes.get(&host).cloned() else { continue };
        // Only what reaches Out is built: an unconnected shape (or an op
        // nothing reads) is drawn in the node and costs the render nothing.
        let mut live = std::collections::BTreeSet::new();
        let mut stack = sp.out_inputs();
        while let Some(r) = stack.pop() {
            if live.insert(r) {
                if let Some(o) = r.checked_sub(ROTO_OP).and_then(|k| sp.ops.get(k)) {
                    stack.extend([o.a, o.b].into_iter().flatten());
                }
            }
        }
        // Every live shape as a bare Spline: nothing on its input, so its
        // own union with "nothing" is exactly the shape. Minted in id order
        // (the BTreeSet), so the result is deterministic.
        let mut built: std::collections::HashMap<usize, NodeId> = Default::default();
        for &r in &live {
            let id = if r >= ROTO_OP {
                let Some(o) = sp.ops.get(r - ROTO_OP) else { continue };
                mint(g, NodeKind::MatteCombine(MatteCombineParams { op: o.op, mix: o.mix }))
            } else {
                let mut one = sp.single(r);
                one.combine = MatteOp::Union;
                one.format = None;
                let id = mint(g, NodeKind::Spline(one));
                g.format_alias.insert(id, host);
                id
            };
            built.insert(r, id);
        }
        let node_of = |r: usize| built.get(&r).copied();
        for &r in &live {
            let Some(o) = r.checked_sub(ROTO_OP).and_then(|k| sp.ops.get(k)) else { continue };
            // A wire that would loop (hand-edited data) is dropped rather
            // than handed to codegen.
            for (port, x) in [(0usize, o.a), (1, o.b)] {
                if let (Some(dst), Some(src)) = (node_of(r), x.filter(|x| !sp.feeds(r, *x)).and_then(node_of)) {
                    g.edges.insert((dst, port), src);
                }
            }
        }
        // What is wired into Out, unioned left to right.
        let mut result: Option<NodeId> = None;
        for r in sp.out_inputs() {
            let Some(n) = node_of(r) else { continue };
            result = Some(match result {
                None => n,
                Some(acc) => {
                    let u = mint(
                        g,
                        NodeKind::MatteCombine(MatteCombineParams { op: MatteOp::Union, mix: 1.0 }),
                    );
                    g.edges.insert((u, 0), acc);
                    g.edges.insert((u, 1), n);
                    u
                }
            });
        }
        // The container itself: its input matte (A, where it already is
        // on port 0) met by the finished roto (B) under its own combine —
        // what a one-shape Spline does with its shape.
        g.nodes.insert(
            host,
            NodeKind::MatteCombine(MatteCombineParams { op: sp.combine, mix: 1.0 }),
        );
        match result {
            Some(r) => {
                g.edges.insert((host, 1), r);
            }
            None => {
                g.edges.remove(&(host, 1));
            }
        }
    }
}

/// The box (normalized, `[x0, y0, x1, y1]`, clamped to the frame) around
/// what a Spline DRAWS — every shape of a container carried through its
/// ops: a union (or add, or difference) spans both boxes, intersect and
/// min only their overlap, a cut keeps its A's, replace its B's; an op
/// mixed back toward A also keeps A's; what is wired into Out unions (an
/// unconnected shape draws nothing, so it adds nothing). An inverted
/// shape covers the frame. The outline's box, as the gizmo draws it (the
/// soft edge past it is the crop's own softness to set). `None`: nothing
/// is drawn. What a SoftCrop around the roto should span.
pub fn bounds(sp: &crate::graph::SplineParams, size: [f32; 2]) -> Option<[f32; 4]> {
    type B = Option<[f32; 4]>;
    let union = |a: B, b: B| match (a, b) {
        (Some(a), Some(b)) => Some([a[0].min(b[0]), a[1].min(b[1]), a[2].max(b[2]), a[3].max(b[3])]),
        (a, None) => a,
        (None, b) => b,
    };
    let overlap = |a: B, b: B| {
        let (a, b) = (a?, b?);
        let r = [a[0].max(b[0]), a[1].max(b[1]), a[2].min(b[2]), a[3].min(b[3])];
        (r[0] < r[2] && r[1] < r[3]).then_some(r)
    };
    let shape = |s: usize| -> B {
        let one = sp.single(s);
        if one.invert {
            return Some([0.0, 0.0, 1.0, 1.0]);
        }
        let poly = crate::codegen::spline_polyline(&one, size);
        let mut it = poly.iter().map(|(p, _)| *p);
        let first = it.next()?;
        let mut r = [first[0], first[1], first[0], first[1]];
        for p in it {
            r = [r[0].min(p[0]), r[1].min(p[1]), r[2].max(p[0]), r[3].max(p[1])];
        }
        Some(r)
    };
    fn walk(
        sp: &crate::graph::SplineParams,
        r: usize,
        depth: usize,
        shape: &dyn Fn(usize) -> Option<[f32; 4]>,
        union: &dyn Fn(Option<[f32; 4]>, Option<[f32; 4]>) -> Option<[f32; 4]>,
        overlap: &dyn Fn(Option<[f32; 4]>, Option<[f32; 4]>) -> Option<[f32; 4]>,
    ) -> Option<[f32; 4]> {
        if r < ROTO_OP {
            return shape(r);
        }
        // Hand-edited data could loop; a loop draws nothing (the lowering
        // drops those wires too).
        let o = sp.ops.get(r - ROTO_OP).filter(|_| depth < 64)?;
        let sub = |x: Option<usize>| x.and_then(|x| walk(sp, x, depth + 1, shape, union, overlap));
        let (a, b) = (sub(o.a), sub(o.b));
        let out = match o.op {
            MatteOp::Union | MatteOp::Add | MatteOp::Difference => union(a, b),
            MatteOp::Intersect | MatteOp::Min => overlap(a, b),
            MatteOp::Subtract => a,
            MatteOp::Replace => b,
        };
        if o.mix < 1.0 { union(out, a) } else { out }
    }
    let r = sp
        .out_inputs()
        .into_iter()
        .map(|r| walk(sp, r, 0, &shape, &union, &overlap))
        .fold(None, union)?;
    Some(r.map(|v| v.clamp(0.0, 1.0)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{RotoOp, SplineParams, SplinePoint, SplineShape};

    fn tri(dx: f32) -> Vec<SplinePoint> {
        [[0.2 + dx, 0.2], [0.4 + dx, 0.2], [0.3 + dx, 0.4]]
            .iter()
            .map(|p| SplinePoint::new(*p, 0.0))
            .collect()
    }

    #[test]
    fn a_plain_spline_is_left_alone() {
        let mut g = Graph::default();
        g.nodes.insert(1, NodeKind::Spline(SplineParams { points: tri(0.0), ..Default::default() }));
        let before = g.nodes.clone();
        expand(&mut g);
        assert_eq!(g.nodes, before);
    }

    #[test]
    fn a_container_lowers_to_shapes_ops_and_a_union_of_the_loose_ends() {
        let mut sp = SplineParams { points: tri(0.0), ..Default::default() };
        sp.more.push(SplineShape { points: tri(0.3), ..Default::default() });
        sp.more.push(SplineShape { points: tri(0.5), ..Default::default() });
        // shape 1 cut out of shape 0; shape 2 loose.
        sp.add_op(RotoOp { op: MatteOp::Subtract, a: Some(0), b: Some(1), ..Default::default() });
        let mut g = Graph::default();
        g.nodes.insert(5, NodeKind::Spline(sp));
        g.view = Some(5);
        expand(&mut g);
        // 3 shapes + 1 op + 1 union of (op, shape 2), and the host.
        assert_eq!(g.nodes.len(), 6);
        assert!(matches!(g.nodes[&5], NodeKind::MatteCombine(_)));
        let spl = g.nodes.values().filter(|k| matches!(k, NodeKind::Spline(_))).count();
        assert_eq!(spl, 3);
        // Every minted spline reads the host's format.
        assert_eq!(g.format_alias.len(), 3);
        assert!(g.format_alias.values().all(|h| *h == 5));
        // The host's B is the union of the loose ends — shape 2, then the
        // op (shapes come first).
        let u = g.input_of(5, 1).unwrap();
        let (a, b) = (g.input_of(u, 0).unwrap(), g.input_of(u, 1).unwrap());
        assert!(matches!(g.nodes[&a], NodeKind::Spline(_)));
        assert!(matches!(g.nodes[&b], NodeKind::MatteCombine(MatteCombineParams { op: MatteOp::Subtract, .. })));
    }

    #[test]
    fn bounds_follow_the_ops() {
        const SZ: [f32; 2] = [100.0, 100.0];
        let rect = |x0: f32, y0: f32, x1: f32, y1: f32| -> Vec<SplinePoint> {
            [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
                .iter()
                .map(|p| SplinePoint::new(*p, 0.0))
                .collect()
        };
        let lin = crate::graph::SplineMode::Linear;
        let mut sp = SplineParams { points: rect(0.1, 0.1, 0.5, 0.5), mode: lin, ..Default::default() };
        sp.more.push(SplineShape { points: rect(0.3, 0.3, 0.9, 0.7), mode: lin, ..Default::default() });
        let near = |a: [f32; 4], b: [f32; 4]| a.iter().zip(b).all(|(a, b)| (a - b).abs() < 1e-4);
        // Loose: the union of both.
        assert!(near(bounds(&sp, SZ).unwrap(), [0.1, 0.1, 0.9, 0.7]));
        // Intersected: the overlap only.
        sp.add_op(RotoOp { op: MatteOp::Intersect, a: Some(0), b: Some(1), ..Default::default() });
        assert!(near(bounds(&sp, SZ).unwrap(), [0.3, 0.3, 0.5, 0.5]));
        // Cut: A's box, whatever B took out of it.
        sp.ops[0].op = MatteOp::Subtract;
        assert!(near(bounds(&sp, SZ).unwrap(), [0.1, 0.1, 0.5, 0.5]));
        // Half-mixed intersect keeps A showing through.
        sp.ops[0] = RotoOp { op: MatteOp::Intersect, mix: 0.5, a: Some(1), b: Some(0), ..Default::default() };
        assert!(near(bounds(&sp, SZ).unwrap(), [0.3, 0.3, 0.9, 0.7]));
        // An inverted shape is the whole frame; nothing drawn is None.
        sp.more[0].invert = true;
        sp.ops.clear();
        assert!(near(bounds(&sp, SZ).unwrap(), [0.0, 0.0, 1.0, 1.0]));
        assert!(bounds(&SplineParams::default(), SZ).is_none());
    }

    #[test]
    fn an_unconnected_shape_builds_nothing() {
        let mut sp = SplineParams { points: tri(0.0), ..Default::default() };
        sp.more.push(SplineShape { points: tri(0.3), ..Default::default() });
        sp.out = Some(vec![0]);
        let mut g = Graph::default();
        g.nodes.insert(5, NodeKind::Spline(sp));
        expand(&mut g);
        // Shape 0 and the host; shape 1 is not in Out, so not built.
        assert_eq!(g.nodes.len(), 2);
        let mut sp = SplineParams { points: tri(0.0), ..Default::default() };
        sp.more.push(SplineShape::default());
        sp.out = Some(vec![]);
        assert!(bounds(&sp, [100.0, 100.0]).is_none(), "nothing wired, nothing drawn");
    }

    #[test]
    fn minting_is_deterministic() {
        let make = || {
            let mut sp = SplineParams { points: tri(0.0), ..Default::default() };
            sp.more.push(SplineShape { points: tri(0.3), ..Default::default() });
            let mut g = Graph::default();
            g.nodes.insert(2, NodeKind::Spline(sp.clone()));
            g.nodes.insert(9, NodeKind::Spline(sp));
            expand(&mut g);
            let mut e: Vec<_> = g.edges.iter().map(|(k, v)| (*k, *v)).collect();
            e.sort_unstable();
            e
        };
        assert_eq!(make(), make());
    }
}
