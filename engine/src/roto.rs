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

use crate::graph::{
    Graph, MatteCombineParams, MatteOp, NodeId, NodeKind, RotoCam, RotoRef, ShapeProj, SplineParams,
};
use crate::scene3d::{identity, mat_mul, Mat4};

/// The Roto's camera port (`NodeKind::Spline` input 1).
pub const CAMERA_INPUT: usize = 1;

pub fn expand(g: &mut Graph) {
    let mut ids: Vec<NodeId> = g
        .nodes
        .iter()
        .filter(|(_, k)| matches!(k, NodeKind::Spline(_)))
        .map(|(id, _)| *id)
        .collect();
    if ids.is_empty() {
        return;
    }
    ids.sort_unstable();
    let mut next = g.nodes.keys().copied().max().unwrap_or(0) + 1;

    for host in ids {
        // The camera, read off its port and the wire dropped: the scene pipe
        // is CPU data and never reaches codegen.
        let cam = match g.input_of(host, CAMERA_INPUT).and_then(|c| g.nodes.get(&c)) {
            Some(NodeKind::Camera3D(c)) => Some(RotoCam::of(c)),
            _ => None,
        };
        g.edges.remove(&(host, CAMERA_INPUT));
        let Some(NodeKind::Spline(sp)) = g.nodes.get(&host).cloned() else { continue };
        if !sp.is_container() {
            // A plain Spline: only a camera changes anything.
            if let (Some(cam), Some(NodeKind::Spline(one))) = (cam, g.nodes.get_mut(&host)) {
                one.proj = Some(ShapeProj { world: identity(), cam: Some(cam) });
            }
            continue;
        }
        let mut b = Builder { g: &mut *g, sp: &sp, host, cam, next: &mut next, memo: Default::default(), stack: Vec::new() };
        // Only what reaches Out is built: an unconnected shape (or an op
        // nothing reads) is drawn in the node and costs the render nothing.
        // What is wired into Out, unioned left to right.
        let parts: Vec<NodeId> =
            sp.out_inputs().into_iter().filter_map(|r| b.build(r, identity())).collect();
        let result = b.union(parts);
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

/// One container's lowering: every mini node reached from Out under the
/// moves of the axes above it, built once per (node, move).
struct Builder<'a> {
    g: &'a mut Graph,
    sp: &'a SplineParams,
    host: NodeId,
    cam: Option<RotoCam>,
    next: &'a mut NodeId,
    memo: std::collections::HashMap<(usize, [u32; 16]), Option<NodeId>>,
    /// The mini nodes on the way here: a wire that would loop (hand-edited
    /// data) is dropped rather than handed to codegen.
    stack: Vec<usize>,
}

impl Builder<'_> {
    fn mint(&mut self, k: NodeKind) -> NodeId {
        let id = *self.next;
        *self.next += 1;
        self.g.nodes.insert(id, k);
        id
    }

    fn union(&mut self, parts: Vec<NodeId>) -> Option<NodeId> {
        let mut acc: Option<NodeId> = None;
        for n in parts {
            acc = Some(match acc {
                None => n,
                Some(a) => {
                    let u = self.mint(NodeKind::MatteCombine(MatteCombineParams {
                        op: MatteOp::Union,
                        mix: 1.0,
                    }));
                    self.g.edges.insert((u, 0), a);
                    self.g.edges.insert((u, 1), n);
                    u
                }
            });
        }
        acc
    }

    fn build(&mut self, r: usize, m: Mat4) -> Option<NodeId> {
        if self.stack.contains(&r) || self.stack.len() > 64 || !self.sp.mini_exists(r) {
            return None;
        }
        let mut bits = [0u32; 16];
        for (i, v) in m.iter().flatten().enumerate() {
            bits[i] = v.to_bits();
        }
        let key = (r, bits);
        if let Some(done) = self.memo.get(&key) {
            return *done;
        }
        self.stack.push(r);
        let id = match RotoRef::of(r) {
            RotoRef::Shape(s) => {
                // A bare Spline: nothing on its input, so its own union with
                // "nothing" is exactly the shape.
                let mut one = self.sp.single(s);
                one.combine = MatteOp::Union;
                one.format = None;
                let p = ShapeProj { world: m, cam: self.cam };
                one.proj = (!p.is_identity()).then_some(p);
                let id = self.mint(NodeKind::Spline(one));
                self.g.format_alias.insert(id, self.host);
                Some(id)
            }
            RotoRef::Op(k) => {
                let o = self.sp.ops[k].clone();
                let id = self.mint(NodeKind::MatteCombine(MatteCombineParams { op: o.op, mix: o.mix }));
                for (port, x) in [(0usize, o.a), (1, o.b)] {
                    if let Some(src) = x.and_then(|x| self.build(x, m)) {
                        self.g.edges.insert((id, port), src);
                    }
                }
                Some(id)
            }
            RotoRef::Axis(k) => {
                let a = self.sp.axes[k].clone();
                let m2 = mat_mul(&m, &a.matrix());
                let parts: Vec<NodeId> =
                    a.inputs.iter().flatten().filter_map(|x| self.build(*x, m2)).collect();
                self.union(parts)
            }
        };
        self.stack.pop();
        self.memo.insert(key, id);
        id
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
    let shape = |s: usize, m: &Mat4| -> B {
        let mut one = sp.single(s);
        if one.invert {
            return Some([0.0, 0.0, 1.0, 1.0]);
        }
        let p = ShapeProj { world: *m, cam: None };
        one.proj = (!p.is_identity()).then_some(p);
        let poly = crate::codegen::spline_polyline(&one, size);
        let mut it = poly.iter().map(|(p, _)| *p);
        let first = it.next()?;
        let mut r = [first[0], first[1], first[0], first[1]];
        for p in it {
            r = [r[0].min(p[0]), r[1].min(p[1]), r[2].max(p[0]), r[3].max(p[1])];
        }
        Some(r)
    };
    struct Cx<'a> {
        sp: &'a SplineParams,
        shape: &'a dyn Fn(usize, &Mat4) -> B,
        union: &'a dyn Fn(B, B) -> B,
        overlap: &'a dyn Fn(B, B) -> B,
    }
    fn walk(cx: &Cx, r: usize, m: Mat4, depth: usize) -> B {
        // Hand-edited data could loop; a loop draws nothing (the lowering
        // drops those wires too).
        if depth >= 64 {
            return None;
        }
        match RotoRef::of(r) {
            RotoRef::Shape(s) => (cx.shape)(s, &m),
            RotoRef::Op(k) => {
                let o = cx.sp.ops.get(k)?;
                let sub = |x: Option<usize>| x.and_then(|x| walk(cx, x, m, depth + 1));
                let (a, b) = (sub(o.a), sub(o.b));
                let out = match o.op {
                    MatteOp::Union | MatteOp::Add | MatteOp::Difference => (cx.union)(a, b),
                    MatteOp::Intersect | MatteOp::Min => (cx.overlap)(a, b),
                    MatteOp::Subtract => a,
                    MatteOp::Replace => b,
                };
                if o.mix < 1.0 { (cx.union)(out, a) } else { out }
            }
            RotoRef::Axis(k) => {
                let a = cx.sp.axes.get(k)?;
                let m2 = mat_mul(&m, &a.matrix());
                a.inputs.iter().flatten().map(|x| walk(cx, *x, m2, depth + 1)).fold(None, cx.union)
            }
        }
    }
    let cx = Cx { sp, shape: &shape, union: &union, overlap: &overlap };
    let r = sp
        .out_inputs()
        .into_iter()
        .map(|r| walk(&cx, r, identity(), 0))
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

    fn first_spline(g: &Graph) -> SplineParams {
        let mut v: Vec<_> = g.nodes.iter().filter_map(|(id, k)| match k {
            NodeKind::Spline(sp) => Some((*id, sp.clone())),
            _ => None,
        }).collect();
        v.sort_by_key(|(id, _)| *id);
        v.remove(0).1
    }

    fn centroid(sp: &SplineParams, size: [f32; 2]) -> [f32; 2] {
        let poly = crate::codegen::spline_polyline(sp, size);
        let n = poly.len() as f32;
        let s = poly.iter().fold([0.0, 0.0], |a, (p, _)| [a[0] + p[0], a[1] + p[1]]);
        [s[0] / n, s[1] / n]
    }

    fn axis_over(inputs: Vec<Option<usize>>, translate: [f32; 3]) -> crate::graph::RotoAxis {
        crate::graph::RotoAxis {
            xform: crate::scene3d::Xform { translate, ..Default::default() },
            inputs,
            ..Default::default()
        }
    }

    #[test]
    fn an_axis_moves_its_shapes_and_nests() {
        const SZ: [f32; 2] = [200.0, 100.0];
        let mut sp = SplineParams { points: tri(0.0), ..Default::default() };
        let c0 = centroid(&sp, SZ);
        // Axis 0 holds shape 0, axis 1 holds axis 0: both moves add up.
        let a0 = sp.add_axis(axis_over(vec![Some(0)], [0.1, 0.0, 0.0])).unwrap();
        sp.add_axis(axis_over(vec![Some(a0)], [0.05, 0.1, 0.0])).unwrap();
        assert_eq!(sp.out_inputs(), vec![crate::graph::ROTO_AXIS + 1], "only the top axis is loose");
        let mut g = Graph::default();
        g.nodes.insert(5, NodeKind::Spline(sp.clone()));
        g.view = Some(5);
        expand(&mut g);
        let one = first_spline(&g);
        assert!(one.proj.is_some());
        let c1 = centroid(&one, SZ);
        // Scene units: the frame is one unit wide, Y up. 0.15 across is
        // 0.15 of the width; 0.1 up is 0.1 * 200 px = 0.2 of the height.
        assert!((c1[0] - c0[0] - 0.15).abs() < 1e-4, "{c0:?} -> {c1:?}");
        assert!((c1[1] - c0[1] + 0.2).abs() < 1e-4, "{c0:?} -> {c1:?}");
        // The gizmo's view of the same shape agrees.
        let pr = sp.shape_proj(0, None).unwrap();
        let p = pr.point(c0, SZ);
        assert!((p[0] - c1[0]).abs() < 1e-3 && (p[1] - c1[1]).abs() < 1e-3);
    }

    #[test]
    fn a_default_camera_sees_the_shape_where_it_was_drawn() {
        const SZ: [f32; 2] = [192.0, 108.0];
        let sp = SplineParams { points: tri(0.1), ..Default::default() };
        let c0 = centroid(&sp, SZ);
        let mut g = Graph::default();
        g.nodes.insert(5, NodeKind::Spline(sp));
        g.nodes.insert(6, NodeKind::Camera3D(Default::default()));
        g.edges.insert((5, CAMERA_INPUT), 6);
        g.view = Some(5);
        expand(&mut g);
        assert!(g.input_of(5, CAMERA_INPUT).is_none(), "the scene wire is lowered away");
        let NodeKind::Spline(one) = &g.nodes[&5] else { panic!() };
        assert!(one.proj.is_some());
        let c1 = centroid(one, SZ);
        assert!((c0[0] - c1[0]).abs() < 1e-4 && (c0[1] - c1[1]).abs() < 1e-4, "{c0:?} {c1:?}");
    }

    #[test]
    fn pushing_an_axis_away_from_the_camera_shrinks_it_toward_the_centre() {
        const SZ: [f32; 2] = [100.0, 100.0];
        let mut sp = SplineParams { points: tri(0.0), ..Default::default() };
        let c0 = centroid(&sp, SZ);
        sp.add_axis(axis_over(vec![Some(0)], [0.0, 0.0, -50.0 / 36.0])).unwrap();
        let mut g = Graph::default();
        g.nodes.insert(5, NodeKind::Spline(sp));
        g.nodes.insert(6, NodeKind::Camera3D(Default::default()));
        g.edges.insert((5, CAMERA_INPUT), 6);
        expand(&mut g);
        let c1 = centroid(&first_spline(&g), SZ);
        // Twice as far: half as far from the centre.
        assert!((c1[0] - (0.5 + (c0[0] - 0.5) * 0.5)).abs() < 1e-3, "{c0:?} {c1:?}");
        assert!((c1[1] - (0.5 + (c0[1] - 0.5) * 0.5)).abs() < 1e-3, "{c0:?} {c1:?}");
    }

    #[test]
    fn removing_an_op_does_not_slide_axis_ids() {
        let mut sp = SplineParams { points: tri(0.0), ..Default::default() };
        sp.more.push(SplineShape { points: tri(0.3), ..Default::default() });
        let op = sp.add_op(RotoOp { a: Some(0), b: Some(1), ..Default::default() });
        let ax = sp.add_axis(axis_over(vec![Some(op)], [0.0; 3])).unwrap();
        sp.out = Some(vec![ax]);
        sp.remove_op(0);
        assert_eq!(sp.out, Some(vec![ax]));
        assert_eq!(sp.axes[0].inputs, vec![None]);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn axes_round_trip_and_old_rotos_load_without_them() {
        let mut sp = SplineParams { points: tri(0.0), ..Default::default() };
        sp.add_axis(axis_over(vec![Some(0)], [0.1, 0.2, 0.3])).unwrap();
        let j = serde_json::to_string(&sp).unwrap();
        assert!(!j.contains("proj"));
        let back: SplineParams = serde_json::from_str(&j).unwrap();
        assert_eq!(back, sp);
        let old = r#"{"points":[{"p":[0.2,0.2]},{"p":[0.4,0.2]},{"p":[0.3,0.4]}],"more":[{"points":[]}]}"#;
        let o: SplineParams = serde_json::from_str(old).unwrap();
        assert!(o.axes.is_empty() && o.is_container());
    }
}
