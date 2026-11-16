//! A Roto's DIVE: Enter on a Spline (the Roto node) opens its shapes as a
//! graph of their own.
//!
//! Inside, every shape is an ordinary `NodeKind::Spline` — so the viewer
//! gizmo, the pen tool and the panel edit it exactly as they edit any
//! Spline — every combine op an ordinary `MatteCombine`, and the two kinds
//! only a roto has are `NodeKind::RotoNode`: an **Axis** (a group that
//! unions what is wired into it and moves it by a 3D transform; axes nest,
//! Flame-style) and **Out** (what the node renders, unioned).
//!
//! The data never stops being the owner's `SplineParams` (`more`, `ops`,
//! `axes`, `out`, `layout`): [`to_editor`] builds the body from it and
//! [`from_editor`] writes it back, curves included (a shape's `RotoPt` /
//! `RotoSh` on the owner are its own `Spline*` channels in here, an axis's
//! `RotoAx` its `Card3d*` transform channels). So a setup saved before the
//! dive existed opens unchanged, and the in-place `innards` view of the
//! same data keeps working beside it.

use crate::anim::{self, Anim, Channel};
use crate::editor::{ENode, EditorGraph};
use engine::graph::{
    MatteCombineParams, NodeKind, RotoAxis, RotoInner, RotoOp, RotoRef, SplineParams, SplineShape,
    ROTO_AXIS, ROTO_OP,
};

/// Mini-node [-1, 1] square -> graph units.
const SPREAD: f32 = 320.0;

/// The nine axis transform channels inside the dive, in `RotoAx` order:
/// translate, rotate, scale (the Card's, so the panel rows are shared).
pub const AXIS_CHANNELS: [Channel; 9] = [
    Channel::Card3dTx,
    Channel::Card3dTy,
    Channel::Card3dTz,
    Channel::Card3dRx,
    Channel::Card3dRy,
    Channel::Card3dRz,
    Channel::Card3dSx,
    Channel::Card3dSy,
    Channel::Card3dSz,
];

/// A fresh Axis for the dive's add menu.
pub fn new_axis() -> NodeKind {
    NodeKind::RotoNode(RotoInner::Axis(RotoAxis { inputs: vec![None], ..Default::default() }))
}

/// A Spline's channel that belongs to one of its shapes (or axes), as
/// `(owner mini id, the channel as that item names it inside the dive)`.
fn item_of(ch: Channel) -> Option<(usize, Channel)> {
    if let Channel::RotoAx(k, w) = ch {
        return AXIS_CHANNELS.get(w as usize).map(|c| (ROTO_AXIS + k as usize, *c));
    }
    if let Some((s, base)) = anim::shape_base(ch) {
        return Some((s, base));
    }
    anim::is_shape_chan(ch).then_some((0, ch))
}

/// The body for a Roto owner `sp` with curves `anim`.
pub fn to_editor(sp: &SplineParams, anim: &Anim) -> EditorGraph {
    let mut g = EditorGraph::default();
    let at = |p: [f32; 2]| egui::pos2(p[0] * SPREAD, p[1] * SPREAD);
    let mut ids: std::collections::HashMap<usize, u64> = Default::default();
    for s in 0..sp.shape_count() {
        let mut one = sp.single(s);
        // Inside, a shape is a bare shape: the owner's combine and format
        // are the owner's, and stay on its panel.
        one.combine = Default::default();
        one.format = None;
        ids.insert(s, g.add(NodeKind::Spline(one), at(sp.shape_at(s))));
    }
    for (k, o) in sp.ops.iter().enumerate() {
        let id = g.add(NodeKind::MatteCombine(MatteCombineParams { op: o.op, mix: o.mix }), at(o.at));
        ids.insert(ROTO_OP + k, id);
    }
    for (k, a) in sp.axes.iter().enumerate() {
        let mut a2 = a.clone();
        a2.inputs = vec![None; a.inputs.iter().flatten().count() + 1];
        ids.insert(ROTO_AXIS + k, g.add(NodeKind::RotoNode(RotoInner::Axis(a2)), at(a.at)));
    }
    let outs = sp.out_inputs();
    let out = g.add(NodeKind::RotoNode(RotoInner::Out { inputs: outs.len() + 1 }), at(sp.out_node_at()));
    // Wires: what each item reads, packed onto its ports (an axis or Out is
    // one bar with no holes).
    for (k, o) in sp.ops.iter().enumerate() {
        for (port, x) in [(0usize, o.a), (1, o.b)] {
            if let (Some(src), Some(dst)) = (x.and_then(|x| ids.get(&x)), ids.get(&(ROTO_OP + k))) {
                g.edges.push((*dst, port, *src));
            }
        }
    }
    for (k, a) in sp.axes.iter().enumerate() {
        let dst = ids[&(ROTO_AXIS + k)];
        for (port, src) in a.inputs.iter().flatten().filter_map(|x| ids.get(x)).enumerate() {
            g.edges.push((dst, port, *src));
        }
    }
    for (port, src) in outs.iter().filter_map(|x| ids.get(x)).enumerate() {
        g.edges.push((out, port, *src));
    }
    // Curves: each item's own, under the names it has in here.
    for (ch, curve) in anim {
        if let Some((item, local)) = item_of(*ch) {
            if let Some(n) = ids.get(&item).and_then(|id| g.nodes.get_mut(id)) {
                n.anim.insert(local, curve.clone());
            }
        }
    }
    g.view = Some(out);
    g
}

/// The other direction: the owner `host` with its shapes, ops, axes and
/// Out taken from `body`, and its curves `anim` with every shape and axis
/// curve replaced by the body's. Anything else in the body (a node pasted
/// in from the comp) is dropped.
pub fn from_editor(body: &EditorGraph, host: &SplineParams, anim: &Anim) -> (SplineParams, Anim) {
    let mut sp = host.clone();
    sp.more.clear();
    sp.ops.clear();
    sp.axes.clear();
    sp.layout.clear();
    let mut mini: std::collections::HashMap<u64, usize> = Default::default();
    let mut order: Vec<u64> = body.nodes.keys().copied().collect();
    order.sort_unstable();
    let norm = |n: &ENode| [n.pos.x / SPREAD, n.pos.y / SPREAD];
    let mut shapes = 0usize;
    let mut out_node = None;
    for id in &order {
        let n = &body.nodes[id];
        match &n.kind {
            NodeKind::Spline(one) => {
                let sh = SplineShape {
                    points: one.points.clone(),
                    mode: one.mode,
                    opacity: one.opacity,
                    expand: one.expand,
                    falloff: one.falloff,
                    invert: one.invert,
                    xf: one.xf,
                    feather_out: one.feather_out,
                    feather_in: one.feather_in,
                    open: one.open,
                };
                if shapes == 0 {
                    sp.set_shape(0, sh);
                } else {
                    sp.more.push(sh);
                }
                sp.layout.push(norm(n));
                mini.insert(*id, shapes);
                shapes += 1;
            }
            NodeKind::MatteCombine(c) => {
                mini.insert(*id, ROTO_OP + sp.ops.len());
                sp.ops.push(RotoOp { op: c.op, mix: c.mix, a: None, b: None, at: norm(n) });
            }
            NodeKind::RotoNode(RotoInner::Axis(a)) => {
                mini.insert(*id, ROTO_AXIS + sp.axes.len());
                sp.axes.push(RotoAxis { inputs: Vec::new(), at: norm(n), ..a.clone() });
            }
            NodeKind::RotoNode(RotoInner::Out { .. }) if out_node.is_none() => {
                out_node = Some(*id);
                sp.out_at = Some(norm(n));
            }
            _ => {}
        }
    }
    if shapes == 0 {
        // A roto always has a shape 0: an empty one draws nothing.
        sp.set_shape(0, SplineShape::default());
        sp.layout.push(sp.shape_at(0));
    }
    // Edges, port by port.
    let ins = |dst: u64| -> Vec<(usize, usize)> {
        let mut v: Vec<(usize, usize)> = body
            .edges
            .iter()
            .filter(|(d, _, _)| *d == dst)
            .filter_map(|(_, p, s)| body.resolve_dot(*s).and_then(|s| mini.get(&s)).map(|m| (*p, *m)))
            .collect();
        v.sort_unstable();
        v
    };
    let mut out = Vec::new();
    for id in &order {
        let Some(m) = mini.get(id).copied() else {
            if Some(*id) == out_node {
                out = ins(*id).into_iter().map(|(_, m)| m).collect();
            }
            continue;
        };
        match RotoRef::of(m) {
            RotoRef::Op(k) => {
                for (p, src) in ins(*id) {
                    match p {
                        0 => sp.ops[k].a = Some(src),
                        1 => sp.ops[k].b = Some(src),
                        _ => {}
                    }
                }
            }
            RotoRef::Axis(k) => sp.axes[k].inputs = ins(*id).into_iter().map(|(_, m)| Some(m)).collect(),
            RotoRef::Shape(_) => {}
        }
    }
    sp.out = Some(out);
    // Curves: everything that is not a shape's or an axis's stays; those
    // come from the body.
    let mut a2: Anim = anim.iter().filter(|(c, _)| item_of(**c).is_none()).map(|(c, v)| (*c, v.clone())).collect();
    for (id, m) in &mini {
        let Some(n) = body.nodes.get(id) else { continue };
        for (ch, curve) in &n.anim {
            let named = match RotoRef::of(*m) {
                RotoRef::Shape(s) => Some(anim::for_shape(s, *ch)),
                RotoRef::Axis(k) => AXIS_CHANNELS
                    .iter()
                    .position(|c| c == ch)
                    .map(|w| Channel::RotoAx(k.min(u16::MAX as usize) as u16, w as u8)),
                RotoRef::Op(_) => None,
            };
            if let Some(c) = named {
                a2.insert(c, curve.clone());
            }
        }
    }
    (sp, a2)
}

/// Where the body's shape `node` lands — through the axes on its way to Out
/// and the owner's camera (`cam`, evaluated for the frame) — as an
/// OUTPUT-PIXEL homography for the gizmo. `None`: not moved.
pub fn shape_homography(
    body: &EditorGraph,
    node: u64,
    host: &SplineParams,
    cam: Option<engine::graph::RotoCam>,
    size: [f32; 2],
) -> Option<[[f32; 3]; 3]> {
    let (sp, _) = from_editor(body, host, &Anim::new());
    let mut order: Vec<u64> = body
        .nodes
        .iter()
        .filter(|(_, n)| matches!(n.kind, NodeKind::Spline(_)))
        .map(|(id, _)| *id)
        .collect();
    order.sort_unstable();
    let s = order.iter().position(|id| *id == node)?;
    sp.shape_proj(s, cam).map(|p| p.homography(size))
}

/// The Axis panel inside the dive: its transform (keyable, the Card's
/// rows), its pivot and its name.
pub fn axis_panel(
    ui: &mut egui::Ui,
    rcx: &mut crate::widgets::RowCtx,
    a: &mut RotoAxis,
) -> Option<crate::widgets::HudSlider> {
    ui.horizontal(|ui| {
        ui.add_sized([crate::widgets::LABEL_W, 16.0], egui::Label::new(egui::RichText::new("name").small()));
        ui.add(egui::TextEdit::singleline(&mut a.name).desired_width(140.0).hint_text("axis"));
    });
    let h = crate::nodes::scene3d::xform_rows(ui, rcx, &mut a.xform, &AXIS_CHANNELS, true);
    for (i, l) in ["pivot x", "pivot y", "pivot z"].iter().enumerate() {
        crate::widgets::plain_slider(ui, l, &mut a.pivot[i], -1.0, 1.0, false);
    }
    let n = a.inputs.len().saturating_sub(1);
    ui.weak(format!(
        "moves the {n} thing{} wired into it (shapes, combines, axes); scene units: the frame is one unit wide, Y up. \
         Z needs a Camera on the Roto's camera input",
        if n == 1 { "" } else { "s" }
    ));
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::graph::{SplinePoint, SplineShape};

    fn tri(dx: f32) -> Vec<SplinePoint> {
        [[0.2 + dx, 0.2], [0.4 + dx, 0.2], [0.3 + dx, 0.4]]
            .iter()
            .map(|p| SplinePoint::new(*p, 0.0))
            .collect()
    }

    #[test]
    fn a_roto_round_trips_through_its_dive() {
        let mut sp = SplineParams { points: tri(0.0), ..Default::default() };
        sp.more.push(SplineShape { points: tri(0.3), ..Default::default() });
        sp.more.push(SplineShape { points: tri(0.5), ..Default::default() });
        let op = sp.add_op(RotoOp { op: engine::graph::MatteOp::Subtract, a: Some(0), b: Some(1), ..Default::default() });
        let ax = sp
            .add_axis(RotoAxis {
                xform: engine::scene3d::Xform { translate: [0.1, 0.0, 0.0], ..Default::default() },
                inputs: vec![Some(op), Some(2)],
                ..Default::default()
            })
            .unwrap();
        sp.out = Some(vec![ax]);
        let mut anim = Anim::new();
        let mut c = anim::AnimCurve::default();
        c.set_key(1.0, 0.5);
        anim.insert(Channel::RotoPt(2, 1, 0), c.clone());
        anim.insert(Channel::SplineOpacity, c.clone());
        anim.insert(Channel::RotoAx(0, 3), c.clone());
        let body = to_editor(&sp, &anim);
        // 3 shapes, an op, an axis and Out.
        assert_eq!(body.nodes.len(), 6);
        let (back, a2) = from_editor(&body, &sp, &anim);
        assert_eq!(back.shape_count(), 3);
        assert_eq!(back.points, sp.points);
        assert_eq!(back.more, sp.more);
        assert_eq!(back.ops.len(), 1);
        assert_eq!((back.ops[0].a, back.ops[0].b), (Some(0), Some(1)));
        assert_eq!(back.axes.len(), 1);
        assert_eq!(back.axes[0].inputs, vec![Some(op), Some(2)]);
        assert_eq!(back.axes[0].xform, sp.axes[0].xform);
        assert_eq!(back.out, Some(vec![ax]));
        assert_eq!(a2.len(), 3, "{:?}", a2.keys().collect::<Vec<_>>());
        assert!(a2.contains_key(&Channel::RotoPt(2, 1, 0)));
        assert!(a2.contains_key(&Channel::SplineOpacity));
        assert!(a2.contains_key(&Channel::RotoAx(0, 3)));
    }

    #[test]
    fn an_old_one_shape_spline_opens_onto_one_shape_and_out() {
        let sp = SplineParams { points: tri(0.0), ..Default::default() };
        let body = to_editor(&sp, &Anim::new());
        assert_eq!(body.nodes.len(), 2);
        assert_eq!(body.edges.len(), 1, "the loose shape is wired into Out");
        let (back, _) = from_editor(&body, &sp, &Anim::new());
        assert_eq!(back.points, sp.points);
        assert_eq!(back.out, Some(vec![0]));
    }
}

/// The Camera wired into Roto `host`'s camera port in `g`, at frame `t`.
pub fn camera_in(g: &EditorGraph, host: u64, t: u64) -> Option<engine::graph::RotoCam> {
    let c = g.input_of(host, engine::roto::CAMERA_INPUT).and_then(|s| g.resolve_dot(s))?;
    let mut k = g.node(c)?.kind_at(t);
    let _ = engine::scene3d::apply_file_source(&mut k, t as f32);
    match k {
        NodeKind::Camera3D(c) => Some(engine::graph::RotoCam::of(&c)),
        _ => None,
    }
}

impl crate::Gfx {
    /// The viewer gizmo's extra step for a roto shape that its axes or a
    /// camera move: shape `shape_sel` of Roto `node` in the open graph, or
    /// — inside a Roto's dive — the shape `node` IS. The output-pixel
    /// homography from where its points are authored to where it renders
    /// at `t`, and the camera it went through. `None`: drawn where authored.
    pub(crate) fn roto_gizmo_h(
        &self,
        node: u64,
        shape_sel: usize,
        t: u64,
        size: [f32; 2],
    ) -> Option<([[f32; 3]; 3], Option<engine::graph::RotoCam>)> {
        if let Some(d) = self.dive.last().filter(|d| d.kind == crate::DiveKind::Roto) {
            let owner = d.from?;
            let NodeKind::Spline(host) = d.graph.node(owner)?.kind_at(t) else { return None };
            let cam = camera_in(&d.graph, owner, t);
            // The axes as they are at `t`.
            let mut body = self.egraph.clone();
            for n in body.nodes.values_mut() {
                if matches!(n.kind, NodeKind::RotoNode(_)) && !n.anim.is_empty() {
                    n.kind = n.kind_at(t);
                }
            }
            return shape_homography(&body, node, &host, cam, size).map(|h| (h, cam));
        }
        let NodeKind::Spline(sp) = self.egraph.node(node)?.kind_at(t) else { return None };
        let cam = camera_in(&self.egraph, node, t);
        sp.shape_proj(shape_sel, cam).map(|p| (p.homography(size), cam))
    }
}
