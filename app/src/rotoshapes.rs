//! A roto node's inner graph: its shapes, the ops that combine them, and
//! the Out they all end in — the `innards::Provider` for a Spline.
//!
//! The data is `SplineParams` (`more`, `ops`, `layout`, `out_at`); what it
//! renders is `engine::roto`'s business. Here: how it reads as mini nodes,
//! and what a move, a wire, an unplug, an add, a remove and a SWIPE do to
//! it. Shapes are items (selecting one hands the viewer gizmo and the
//! properties panel to it); ops and Out are not (an op picked shows its
//! own row in the panel instead).

use crate::editor::{EditorGraph, PDst, PSrc, SwipeOp};
use crate::innards::{Mini, MiniEdge, MiniNode, Provider};
use engine::graph::{MatteOp, NodeKind, RotoOp, SplineParams, ROTO_AXIS, ROTO_OP};

pub struct Shapes;
pub static SHAPES: Shapes = Shapes;

/// The Out node's mini id: above every shape and op.
pub const OUT: usize = 1 << 16;

/// An op's label in its mini node — short words, since the default fonts
/// carry none of the set symbols.
pub fn op_label(op: MatteOp) -> &'static str {
    match op {
        MatteOp::Union => "or",
        MatteOp::Add => "add",
        MatteOp::Replace => "B",
        MatteOp::Subtract => "cut",
        MatteOp::Intersect => "in",
        MatteOp::Min => "min",
        MatteOp::Difference => "xor",
    }
}

fn sp_of(kind: &NodeKind) -> Option<&SplineParams> {
    match kind {
        NodeKind::Spline(sp) => Some(sp),
        _ => None,
    }
}

fn sp_mut(kind: &mut NodeKind) -> Option<&mut SplineParams> {
    match kind {
        NodeKind::Spline(sp) => Some(sp),
        _ => None,
    }
}

impl Provider for Shapes {
    fn mini(&self, kind: &NodeKind) -> Mini {
        let Some(sp) = sp_of(kind) else { return Mini::default() };
        let n = sp.shape_count();
        let mut m = Mini::default();
        for s in 0..n {
            m.nodes.push(MiniNode {
                id: s,
                at: sp.shape_at(s),
                label: format!("{}", s + 1),
                inputs: 0,
                output: true,
                item: true,
                removable: n > 1,
            });
        }
        for (k, o) in sp.ops.iter().enumerate() {
            let id = ROTO_OP + k;
            m.nodes.push(MiniNode {
                id,
                at: o.at,
                label: op_label(o.op).into(),
                inputs: 2,
                output: true,
                item: false,
                removable: true,
            });
            for (port, r) in [(0, o.a), (1, o.b)] {
                if let Some(from) = r {
                    m.edges.push(MiniEdge { from, to: id, port });
                }
            }
        }
        // Axes (a Roto's dive makes them; here they are shown and wired
        // like ops): a port per thing in it, and a free one.
        for (k, a) in sp.axes.iter().enumerate() {
            let id = ROTO_AXIS + k;
            let ins: Vec<usize> = a.inputs.iter().flatten().copied().collect();
            m.nodes.push(MiniNode {
                id,
                at: a.at,
                label: if a.name.is_empty() { "axis".into() } else { a.name.clone() },
                inputs: ins.len() + 1,
                output: true,
                item: false,
                removable: true,
            });
            for (port, from) in ins.into_iter().enumerate() {
                m.edges.push(MiniEdge { from, to: id, port });
            }
        }
        // Out: what is wired in, a port each — the union the node outputs,
        // drawn as wires so nothing is secretly combined — and one free
        // port to plug the next into. A shape wired nowhere is drawn here
        // and rendered nowhere.
        let ins = sp.out_inputs();
        m.nodes.push(MiniNode {
            id: OUT,
            at: sp.out_node_at(),
            label: "out".into(),
            inputs: ins.len() + 1,
            output: false,
            item: false,
            removable: false,
        });
        for (port, from) in ins.into_iter().enumerate() {
            m.edges.push(MiniEdge { from, to: OUT, port });
        }
        m
    }

    fn can_add(&self, kind: &NodeKind) -> bool {
        sp_of(kind).is_some_and(|sp| sp.shape_count() < engine::graph::ROTO_SHAPES_MAX)
    }

    fn add(&self, graph: &mut EditorGraph, node: u64) -> Option<usize> {
        sp_mut(&mut graph.node_mut(node)?.kind)?.add_shape()
    }

    fn remove(&self, graph: &mut EditorGraph, node: u64, id: usize) -> bool {
        let Some(n) = graph.node_mut(node) else { return false };
        let Some(sp) = sp_mut(&mut n.kind) else { return false };
        if (ROTO_AXIS..OUT).contains(&id) {
            return sp.remove_axis(id - ROTO_AXIS);
        }
        if id >= ROTO_OP {
            return id < OUT && sp.remove_op(id - ROTO_OP);
        }
        if !sp.remove_shape(id) {
            return false;
        }
        // The shape's curves go and later shapes' slide down — and so do
        // the cables reading or driving them.
        let rename = crate::anim::shape_removed(&mut n.anim, id);
        graph.patch.links.retain_mut(|l| {
            if let PSrc::Chan(sn, sc) = &mut l.from {
                if *sn == node {
                    match rename(*sc) {
                        Some(c) => *sc = c,
                        None => return false,
                    }
                }
            }
            if let PDst::Chan(dn, dc) = &mut l.to {
                if *dn == node {
                    match rename(*dc) {
                        Some(c) => *dc = c,
                        None => return false,
                    }
                }
            }
            true
        });
        true
    }

    fn move_to(&self, kind: &mut NodeKind, id: usize, at: [f32; 2]) {
        let Some(sp) = sp_mut(kind) else { return };
        if id == OUT {
            sp.out_at = Some(at);
        } else if id >= ROTO_AXIS {
            if let Some(a) = sp.axes.get_mut(id - ROTO_AXIS) {
                a.at = at;
            }
        } else if id >= ROTO_OP {
            if let Some(o) = sp.ops.get_mut(id - ROTO_OP) {
                o.at = at;
            }
        } else {
            sp.set_shape_at(id, at);
        }
    }

    fn connect(&self, kind: &mut NodeKind, from: usize, to: usize, port: usize) -> bool {
        let Some(sp) = sp_mut(kind) else { return false };
        if from >= OUT {
            return false;
        }
        if to == OUT {
            // Into Out: onto that port (the free one appends), and only once.
            let list = sp.out_list();
            list.retain(|x| *x != from);
            let at = port.min(list.len());
            list.insert(at, from);
            return true;
        }
        // Otherwise only an op or an axis takes wires, never from
        // something it feeds.
        if !(ROTO_OP..OUT).contains(&to) || sp.feeds(to, from) {
            return false;
        }
        if to >= ROTO_AXIS {
            let Some(a) = sp.axes.get_mut(to - ROTO_AXIS) else { return false };
            let mut list: Vec<usize> = a.inputs.iter().flatten().copied().filter(|x| *x != from).collect();
            list.insert(port.min(list.len()), from);
            a.inputs = list.into_iter().map(Some).collect();
            return true;
        }
        let Some(o) = sp.ops.get_mut(to - ROTO_OP) else { return false };
        match port {
            0 => o.a = Some(from),
            1 => o.b = Some(from),
            _ => return false,
        }
        true
    }

    fn disconnect(&self, kind: &mut NodeKind, to: usize, port: usize) {
        let Some(sp) = sp_mut(kind) else { return };
        if to == OUT {
            let list = sp.out_list();
            if port < list.len() {
                list.remove(port);
            }
            return;
        }
        if let Some(a) = to.checked_sub(ROTO_AXIS).and_then(|k| sp.axes.get_mut(k)) {
            let mut list: Vec<usize> = a.inputs.iter().flatten().copied().collect();
            if port < list.len() {
                list.remove(port);
            }
            a.inputs = list.into_iter().map(Some).collect();
            return;
        }
        if let Some(o) = to.checked_sub(ROTO_OP).and_then(|k| sp.ops.get_mut(k)) {
            match port {
                0 => o.a = None,
                1 => o.b = None,
                _ => {}
            }
        }
    }

    /// The graph's M / I / O swipe, inside the node: `dragged` swiped onto
    /// `under` makes an op of the two that TAKES `under`'s place in
    /// whatever read it — the Blend swipe's rule. M unions them, I keeps
    /// what they share (`under` isolated by `dragged`, a mask), O cuts
    /// `dragged` out of `under` (a stencil).
    fn swipe(&self, kind: &mut NodeKind, dragged: usize, under: usize, op: SwipeOp) -> Option<usize> {
        let sp = sp_mut(kind)?;
        if dragged >= OUT || under >= OUT || dragged == under || sp.feeds(dragged, under) {
            return None;
        }
        let op = match op {
            SwipeOp::Blend | SwipeOp::Multiply => MatteOp::Union,
            SwipeOp::In => MatteOp::Intersect,
            SwipeOp::Out => MatteOp::Subtract,
        };
        let at = mini_at(sp, under);
        // Out's list as it stands, written down before the op exists (or
        // the op would count as one more loose end).
        sp.out_list();
        let id = sp.add_op(RotoOp {
            op,
            a: Some(under),
            b: Some(dragged),
            at: [at[0], (at[1] + 0.32).min(0.8)],
            ..RotoOp::default()
        });
        // Take `under`'s place in Out, and `dragged` leaves it — it is
        // consumed by the op now (left in, the union would put back what
        // a cut took out).
        let out = sp.out_list();
        for x in out.iter_mut() {
            if *x == under {
                *x = id;
            }
        }
        out.retain(|x| *x != dragged);
        let read_elsewhere = sp
            .ops
            .iter()
            .enumerate()
            .any(|(k, o)| ROTO_OP + k != id && (o.a == Some(under) || o.b == Some(under)));
        if !sp.out_list().contains(&id) && !read_elsewhere {
            // `under` fed nothing: the op is a new loose end, so wire it.
            sp.out_list().push(id);
        }
        // ...and in every other op reading `under`, unless that op feeds
        // `dragged`, which would loop.
        for k in 0..sp.ops.len() {
            let me = ROTO_OP + k;
            if me == id || sp.feeds(me, dragged) {
                continue;
            }
            let o = &mut sp.ops[k];
            if o.a == Some(under) {
                o.a = Some(id);
            }
            if o.b == Some(under) {
                o.b = Some(id);
            }
        }
        Some(id)
    }
}

fn mini_at(sp: &SplineParams, id: usize) -> [f32; 2] {
    if id >= ROTO_AXIS {
        sp.axes.get(id - ROTO_AXIS).map(|a| a.at).unwrap_or([0.0, 0.0])
    } else if id >= ROTO_OP {
        sp.ops.get(id - ROTO_OP).map(|o| o.at).unwrap_or([0.0, 0.0])
    } else {
        sp.shape_at(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anim::{for_shape as chan, Channel};
    use engine::graph::{SplinePoint, SplineShape};

    fn pts(n: usize) -> Vec<SplinePoint> {
        (0..n).map(|i| SplinePoint::new([0.1 * i as f32, 0.2], 0.0)).collect()
    }

    fn roto(shapes: usize) -> (EditorGraph, u64) {
        let mut g = EditorGraph::default();
        let mut sp = SplineParams { points: pts(3), ..Default::default() };
        for _ in 1..shapes {
            sp.more.push(SplineShape { points: pts(3), ..Default::default() });
        }
        let id = g.add(NodeKind::Spline(sp), egui::Pos2::ZERO);
        (g, id)
    }

    fn sp(g: &EditorGraph, id: u64) -> SplineParams {
        match &g.node(id).unwrap().kind {
            NodeKind::Spline(sp) => sp.clone(),
            _ => unreachable!(),
        }
    }

    #[test]
    fn loose_ends_are_wired_to_out_and_an_op_takes_its_inputs() {
        let (mut g, id) = roto(3);
        let m = SHAPES.mini(&g.node(id).unwrap().kind);
        assert_eq!(m.edges.iter().filter(|e| e.to == OUT).count(), 3);
        let op = SHAPES.swipe(&mut g.node_mut(id).unwrap().kind, 1, 0, SwipeOp::Out).unwrap();
        let s = sp(&g, id);
        assert_eq!(s.ops[0].op, MatteOp::Subtract);
        assert_eq!((s.ops[0].a, s.ops[0].b), (Some(0), Some(1)), "cut 1 out of 0");
        // Out now reads the op (in shape 0's place) and shape 2.
        assert_eq!(s.out_inputs(), vec![op, 2]);
    }

    #[test]
    fn a_swipe_takes_the_place_of_what_it_swiped_onto() {
        let (mut g, id) = roto(3);
        let k = &mut g.node_mut(id).unwrap().kind;
        let u = SHAPES.swipe(k, 1, 0, SwipeOp::Blend).unwrap();
        // Shape 2 cut into shape 0, which the union already reads: the new
        // op slots in between.
        let c = SHAPES.swipe(k, 2, 0, SwipeOp::Out).unwrap();
        let s = sp(&g, id);
        assert_eq!(s.ops[0].a, Some(c), "the union reads the cut now");
        assert_eq!(s.out_inputs(), vec![u]);
    }

    #[test]
    fn wires_refuse_loops_and_only_land_on_ops() {
        let (mut g, id) = roto(2);
        let k = &mut g.node_mut(id).unwrap().kind;
        let op = SHAPES.swipe(k, 1, 0, SwipeOp::Blend).unwrap();
        assert!(!SHAPES.connect(k, op, op, 0), "into itself");
        assert!(!SHAPES.connect(k, 0, 1, 0), "into a shape");
        SHAPES.disconnect(k, op, 1);
        assert!(SHAPES.connect(k, 1, op, 1));
    }

    /// A shape can sit unconnected: unplugged from Out it stays in the
    /// node and renders nowhere; plugged back into the free port, it is
    /// back. A new shape arrives wired.
    #[test]
    fn a_shape_unplugged_from_out_stays_unconnected() {
        let (mut g, id) = roto(2);
        let k = &mut g.node_mut(id).unwrap().kind;
        assert_eq!(sp_of(k).unwrap().out_inputs(), vec![0, 1]);
        SHAPES.disconnect(k, OUT, 1);
        assert_eq!(sp_of(k).unwrap().out_inputs(), vec![0], "shape 2 unconnected");
        assert_eq!(sp_of(k).unwrap().shape_count(), 2, "and still there");
        let m = SHAPES.mini(k);
        assert_eq!(m.node(OUT).unwrap().inputs, 2, "one wired port and a free one");
        assert!(SHAPES.connect(k, 1, OUT, 1));
        assert_eq!(sp_of(k).unwrap().out_inputs(), vec![0, 1]);
        let s = SHAPES.add(&mut g, id).unwrap();
        assert_eq!(sp(&g, id).out_inputs(), vec![0, 1, s], "a new shape is wired in");
    }

    #[test]
    fn removing_a_shape_slides_its_successors_curves_and_cables() {
        let (mut g, id) = roto(3);
        {
            let n = g.node_mut(id).unwrap();
            n.anim.entry(Channel::SplineTx).or_default().set_key(0.0, 0.1);
            n.anim.entry(chan(1, Channel::SplineTx)).or_default().set_key(0.0, 0.2);
            n.anim.entry(chan(2, Channel::SplinePtX(0))).or_default().set_key(0.0, 0.3);
        }
        g.patch.links.push(crate::editor::PatchLink {
            from: PSrc::Chan(id, chan(2, Channel::SplineTy)),
            to: PDst::Chan(id, chan(1, Channel::SplineRot)),
        });
        assert!(SHAPES.remove(&mut g, id, 1));
        let n = g.node(id).unwrap();
        assert_eq!(n.anim.len(), 2, "shape 1's curve went");
        assert!(n.anim.contains_key(&Channel::SplineTx));
        assert!(n.anim.contains_key(&chan(1, Channel::SplinePtX(0))), "shape 2 is shape 1 now");
        // The cable read shape 2 (now 1) and drove shape 1 (gone): it goes.
        assert!(g.patch.links.is_empty());
    }

    /// A ctrl-slice across the wires into Out unplugs them (the shapes
    /// stay, unconnected); Delete over the node removes the picked one.
    #[test]
    fn a_slice_cuts_inner_wires_and_delete_takes_the_picked_node() {
        let (mut g, id) = roto(2);
        let frame = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(200.0, 200.0));
        let hosts = vec![(id, frame, Vec::new(), true)];
        // Shapes sit at y = -0.5, Out at 0.62: a stroke across y = 0.05.
        let y = frame.center().y + 0.05 * frame.width() * 0.5;
        crate::innards::slice(&mut g, &hosts, egui::pos2(0.0, y), egui::pos2(200.0, y), false);
        assert!(sp(&g, id).out_inputs().is_empty(), "both wires cut: {:?}", sp(&g, id).out);
        assert_eq!(sp(&g, id).shape_count(), 2, "the shapes stay");
        let ctx = egui::Context::default();
        crate::innards::select(&ctx, id, 1);
        assert!(crate::innards::delete_picked(&ctx, &mut g, &hosts, frame.center()));
        assert_eq!(sp(&g, id).shape_count(), 1);
        // Outside the frame, Delete is the graph's own (nothing here).
        crate::innards::select(&ctx, id, 0);
        assert!(!crate::innards::delete_picked(&ctx, &mut g, &hosts, egui::pos2(500.0, 500.0)));
    }

    /// A container survives a save: shapes, ops, layout and all. A plain
    /// Spline writes none of the container fields, so old readers of a
    /// setup see exactly what they always did.
    #[test]
    fn a_container_round_trips_and_a_plain_spline_writes_nothing_new() {
        let (mut g, id) = roto(3);
        let k = &mut g.node_mut(id).unwrap().kind;
        SHAPES.swipe(k, 2, 0, SwipeOp::In).unwrap();
        SHAPES.move_to(k, 1, [0.3, -0.2]);
        let json = serde_json::to_string(&g.node(id).unwrap().kind).unwrap();
        let back: NodeKind = serde_json::from_str(&json).unwrap();
        assert_eq!(&back, &g.node(id).unwrap().kind);
        let plain = serde_json::to_string(&NodeKind::Spline(SplineParams::default())).unwrap();
        for f in ["more", "ops", "layout", "out_at"] {
            assert!(!plain.contains(f), "{f} leaked into a plain Spline: {plain}");
        }
    }

    #[test]
    fn removing_shape_zero_promotes_shape_one() {
        let (mut g, id) = roto(2);
        g.node_mut(id).unwrap().anim.entry(chan(1, Channel::SplineOpacity)).or_default().set_key(0.0, 0.5);
        assert!(SHAPES.remove(&mut g, id, 0));
        assert!(g.node(id).unwrap().anim.contains_key(&Channel::SplineOpacity));
        assert_eq!(sp(&g, id).shape_count(), 1);
    }
}
