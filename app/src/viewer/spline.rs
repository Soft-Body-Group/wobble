//! Spline (roto) and field-shape gizmo: the outline, the points, the handles and the feather rings.
//! Moved out of `viewer_ui` (lib.rs) unchanged.

use crate::*;

impl Gfx {
    /// Spline (roto) and field-shape gizmo: the outline, the points, the handles and the feather rings.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gizmo_spline(
        &mut self,
        ui: &mut egui::Ui,
        cmd: bool,
        scrub_held: bool,
        cur_t: u64,
        owf: f32,
        ohf: f32,
        pointer_pos: Option<egui::Pos2>,
        out_to_pts: &impl Fn(egui::Vec2) -> egui::Pos2,
        pts_to_out: &impl Fn(egui::Pos2) -> egui::Vec2,
        rect: egui::Rect,
        selected: &Vec<u64>,
        patch_driven: &Vec<(u64, anim::Channel)>,
        nav: &egui::Response,
        ink: &mut ink::InkShapes,
        hud_key_list: &mut Vec<hudkeys::HudKey>,
    ) {
        // Spline mask gizmo: the exact outline the GPU walks, plus
        // draggable points, Bezier handles and per-point feather rings.
        // Empty splines start life here — click to lay points down.
        for node in selected.iter().copied() {
            ink.set_tint(self.hud_ink(node));
            // Like the transform/pin gizmos: the mask lives in the masked
            // node's space, so everything downstream of it (transforms AND
            // corner pins) deforms the drawn spline — and drags invert
            // through the same chain.
            // Affine space = SHADER space, which now anchors every
            // spatial node to ITS OWN format (stamped per-node), so the
            // deformation chain resolves each node's local resolution.
            let hd = downstream_proj(
                &self.egraph,
                node,
                &|id| self.node_res_f32(id).unwrap_or((owf, ohf)),
                cur_t,
            );
            let hd_inv = engine::codegen::mat3_inv(&hd);
            // Draw/drag against EVALUATED params (animated points move
            // with the playhead); edits go back keyed, below.
            let Some(ekind) = self.live_kind(node, cur_t) else {
                continue;
            };
            // A field Roto is the same outline, so it borrows this whole
            // gizmo: it becomes a SplineParams for the frame and is
            // written back below. Its points live in FIELD space (y in
            // WIDTH units, not height), which is the one place the two
            // differ — hence `ydiv`.
            // A roto CONTAINER edits one shape at a time: the one picked
            // in its inner graph (`innards`). The others are drawn faint
            // and a click on one picks it.
            let mut shape_sel = 0usize;
            let mut others: Vec<(usize, Vec<([f32; 2], f32)>)> = Vec::new();
            // An open geometry curve (a Stroke, an unclosed Spline): its
            // overlay is the open outline the drawing uses, and the wrap
            // leg back to the first point is neither drawn nor clickable.
            let mut open_curve = false;
            // An SDF spline's `offset` (pixels) moves the drawing, so
            // the gizmo draws its points moved and takes the move back
            // off on the way out: the overlay sits on what renders.
            let mut sdf_off = [0.0f32; 2];
            let (mut spw, is_field) = match ekind {
                NodeKind::Spline(sp) => {
                    shape_sel = innards::selected(ui.ctx(), node, sp.shape_count());
                    // A shape still being drawn has an end (pen tool).
                    open_curve = sp.single(shape_sel).open;
                    for k in (0..sp.shape_count()).filter(|k| *k != shape_sel) {
                        others.push((
                            k,
                            engine::codegen::spline_polyline(&sp.single(k), [owf, ohf]),
                        ));
                    }
                    (sp.single(shape_sel), false)
                }
                NodeKind::Field(engine::field::FieldNode::Spline { points, mode, .. }) => (
                    engine::graph::SplineParams { points, mode, ..Default::default() },
                    true,
                ),
                // A geometry Spline lives in the same space as a field
                // Roto (x across the width, y in width units).
                NodeKind::GeoNode(engine::geo::GeoNode::Spline { points, mode, closed, .. }) => {
                    open_curve = !closed;
                    (engine::graph::SplineParams { points, mode, ..Default::default() }, true)
                }
                // A Stroke is the same outline, always open.
                NodeKind::GeoNode(engine::geo::GeoNode::Stroke { points, mode, .. }) => {
                    open_curve = true;
                    (engine::graph::SplineParams { points, mode, ..Default::default() }, true)
                }
                // An SDF spline: field space too, open or closed.
                NodeKind::SdfNode(engine::sdf::SdfNode::Spline {
                    mut points,
                    mode,
                    closed,
                    offset,
                }) => {
                    open_curve = !closed || points.len() < 3;
                    let k = self.proxy.min(1.0) / owf.max(1.0);
                    sdf_off = [offset[0] * k, offset[1] * k];
                    for q in &mut points {
                        q.p[0] += sdf_off[0];
                        q.p[1] += sdf_off[1];
                    }
                    (engine::graph::SplineParams { points, mode, ..Default::default() }, true)
                }
                _ => continue,
            };
            let ydiv = if is_field { owf } else { ohf };
            let before_sp = spw.clone();
            let sp = &mut spw;
            let poly = engine::codegen::spline_polyline(sp, [owf, ydiv]);
            // The shape transform sits between AUTHORED space (where the
            // control points live, and where every edit lands) and what
            // is drawn. Snapshotted by value — it is `Copy` precisely so
            // these closures can hold it while the points are edited.
            let xf = sp.xf;
            let sz = [owf, ydiv];
            // Authored normalized -> screen, through the transform and
            // then the downstream chain.
            let to_scr = |q: [f32; 2]| {
                let t = xf.point(q, sz);
                out_to_pts(apply_h(&hd, egui::vec2(t[0] * owf, t[1] * ydiv)))
            };
            // Screen -> AUTHORED pixels: the space the group box, its
            // handles and every drag delta are computed in, so a shape
            // that has been moved or rotated still edits point by point.
            let from_screen = |p: egui::Pos2| {
                let q = apply_h(&hd_inv, pts_to_out(p));
                let a = xf.point_inv([q.x / owf, q.y / ydiv], sz);
                egui::vec2(a[0] * owf, a[1] * ydiv)
            };
            let norm = |p: egui::Pos2| {
                let q = from_screen(p);
                [q.x / owf, q.y / ydiv]
            };
            // `spline_polyline` already applied the transform, so the
            // outline goes to screen WITHOUT it — `to_scr` is for
            // authored points only.
            let out_scr =
                |q: [f32; 2]| out_to_pts(apply_h(&hd, egui::vec2(q[0] * owf, q[1] * ydiv)));
            if open_curve && sp.points.len() > 1 {
                let pl = engine::geo::spline_outline(&sp.points, sp.mode, false);
                for w in pl.windows(2) {
                    ink.seg(out_scr(w[0].0), out_scr(w[1].0), 1.2, 0.8, false);
                }
            } else if poly.len() > 1 {
                for i in 0..poly.len() {
                    let a = out_scr(poly[i].0);
                    let b = out_scr(poly[(i + 1) % poly.len()].0);
                    ink.seg(a, b, 1.2, 0.8, false);
                }
            } else if let Some((p0, _)) = poly.first() {
                // A lone point still needs a mark to grab.
                ink.circle_stroke(out_scr(*p0), 7.0, 1.2, 0.8, false);
            }

            // Feather band: both edges of the soft ramp, drawn
            // through the same chain as the outline.
            let ring = spline_band_ring(
                &engine::codegen::spline_band(sp, [owf, ydiv]),
                owf,
                ydiv,
                sp.expand,
            );
            {
                let map = |v: egui::Vec2| out_to_pts(apply_h(&hd, v));
                for i in 0..ring.len() {
                    let k = (i + 1) % ring.len();
                    ink.seg(map(ring[i].0), map(ring[k].0), 0.8, 0.35, false);
                    ink.seg(map(ring[i].1), map(ring[k].1), 0.8, 0.35, false);
                }
            }

            // One spline owns the point selection at a time — indices
            // mean nothing in another shape — and the owner keeps it
            // until it stops being selected.
            if !selected.contains(&self.spline.sel_node.unwrap_or(u64::MAX)) {
                self.spline.sel_node = Some(node);
                self.spline.sel.clear();
            }
            let group_ui = self.spline.sel_node == Some(node);
            // Indices mean nothing in another SHAPE either.
            if group_ui && self.spline.sel_shape != shape_sel {
                self.spline.sel_shape = shape_sel;
                self.spline.sel.clear();
                self.spline.sel_n = sp.points.len();
            }
            // The container's other shapes, faint; the one a click lands
            // on (near its outline) is picked instead of the canvas
            // taking the click.
            let mut other_hit: Option<usize> = None;
            for (k, poly_k) in &others {
                let m = poly_k.len();
                for i in 0..m {
                    let a = out_scr(poly_k[i].0);
                    let b = out_scr(poly_k[(i + 1) % m].0);
                    ink.seg(a, b, 0.8, 0.3, false);
                    if let Some(p) = pointer_pos {
                        let ab = b - a;
                        let t = ((p - a).dot(ab) / ab.length_sq().max(1e-6)).clamp(0.0, 1.0);
                        if (a + ab * t - p).length() < 7.0 && other_hit.is_none() {
                            other_hit = Some(*k);
                        }
                    }
                }
            }
            // Pen-tool drawing: a roto shape with no points, or one not yet
            // closed, takes a plain click as a new point at its end; a click
            // on its first point closes it. Until then it renders no matte.
            let drawing = !is_field && (sp.points.is_empty() || sp.open);
            if group_ui && hud_key_list.is_empty() && drawing {
                use hudkeys::chord;
                hud_key_list.push(chord("click", "add point"));
                hud_key_list.push(chord("drag", if sp.mode == engine::graph::SplineMode::Bezier {
                    "pull handles"
                } else {
                    "set tension"
                }));
                if sp.points.len() >= 3 {
                    hud_key_list.push(chord("1st point", "click: close"));
                }
            }
            if group_ui && hud_key_list.is_empty() {
                use hudkeys::{chord, key};
                hud_key_list.push(key("S", "click: add point", egui::Key::S));
                hud_key_list.push(key("D", "click: remove point", egui::Key::D));
                hud_key_list.push(key("O", "drag point: feather out", egui::Key::O));
                hud_key_list.push(key("I", "drag point: feather in", egui::Key::I));
                if sp.mode == engine::graph::SplineMode::CatmullRom {
                    hud_key_list.push(key("K", "drag point: tension", egui::Key::K));
                }
                if !self.spline.sel.is_empty() {
                    hud_key_list.push(key("E", "scale about click", egui::Key::E));
                    hud_key_list.push(key("R", "rotate: circle the click", egui::Key::R));
                    hud_key_list.push(key("T", "move pick", egui::Key::T));
                    hud_key_list.push(chord("alt", "drag point: feather"));
                }
                if self.spline.sel.len() >= 2 {
                    hud_key_list.push(key("D", "drag centre: pivot", egui::Key::D));
                    hud_key_list.push(chord("ctrl", "drag corner: distort"));
                    hud_key_list.push(chord("alt", "drag edge: from pivot"));
                }
            }
            if group_ui {
                // The properties panel can add or drop points behind our
                // back; a stale index would put the box round the wrong
                // points, so a count change we did not make drops the pick.
                if self.spline.sel_n != sp.points.len() {
                    self.spline.sel.clear();
                    self.spline.sel_n = sp.points.len();
                }
                self.spline.sel.retain(|k| *k < sp.points.len());
            }

            // Adding or removing a point renumbers every per-point
            // channel after it, so the write-back has to slide the
            // curves along with the points.
            let mut struct_edit: Option<SplineEdit> = None;
            // Screen positions the group tools own this frame: a click
            // that lands on one of them is not a click on the canvas.
            let mut hot: Vec<egui::Pos2> = Vec::new();

            // E / R / T held at the press: scale, rotate or translate the
            // pick about a TEMPORARY pivot where the press landed — no
            // box, no handle to find, any number of points. Horizontal
            // travel drives scale and rotate. Snapshotted at the press
            // and computed absolutely, so an animated shape keys cleanly.
            if self.spline.temp.as_ref().is_some_and(|t| t.0 != node) {
                self.spline.temp = None;
            }
            if group_ui && !self.spline.sel.is_empty() && self.spline.temp.is_none() {
                let held = ui.input(|i| {
                    [egui::Key::E, egui::Key::R, egui::Key::T]
                        .into_iter()
                        .find(|k| i.key_down(*k))
                });
                let origin = ui.input(|i| {
                    i.pointer.primary_pressed().then(|| i.pointer.press_origin()).flatten()
                });
                if let (Some(k), Some(o)) = (held, origin.filter(|o| rect.contains(*o))) {
                    self.spline.temp = Some((
                        node,
                        k,
                        o,
                        from_screen(o),
                        self.spline.sel
                            .iter()
                            .filter(|i| **i < sp.points.len())
                            .map(|i| {
                                let q = sp.points[*i];
                                (*i, q.p, q.h_in, q.h_out)
                            })
                            .collect(),
                    ));
                }
            }
            let temp_live = self.spline.temp.is_some();

            // Marquee: a plain background drag bands points (S held keeps
            // what was already picked). cmd/ctrl-drag is the pan and
            // Tab-drag is the scrub, so neither can start one.
            if group_ui {
                let banding = nav.dragged_by(egui::PointerButton::Primary)
                    && self.spline.pen.is_none()
                    && !drawing
                    && !cmd
                    && !scrub_held
                    && !temp_live;
                if banding {
                    if let (Some(o), Some(p)) =
                        (ui.input(|i| i.pointer.press_origin()), pointer_pos)
                    {
                        self.spline.marquee = Some(egui::Rect::from_two_pos(o, p));
                    }
                }
                if let Some(mq) = self.spline.marquee {
                    ink.rect(mq, 0.0, false, 1.0, 0.5, false);
                    if !banding {
                        if !ui.input(|i| i.key_down(egui::Key::S)) {
                            self.spline.sel.clear();
                        }
                        for (i, q) in sp.points.iter().enumerate() {
                            if mq.contains(to_scr(q.p)) {
                                self.spline.sel.insert(i);
                            }
                        }
                        self.spline.marquee = None;
                    }
                }
            }

            // Group transform: the same handle language as the transform
            // gizmo, over the picked points' bounding box. Everything is
            // computed in the node's own pixel space and drawn through
            // the downstream chain, so the box deforms with the shape.
            let sel: Vec<usize> = if group_ui {
                self.spline.sel.iter().copied().collect()
            } else {
                Vec::new()
            };
            let mut box_quad: Option<[egui::Vec2; 4]> = None;
            if sel.len() >= 2 {
                let px = |q: [f32; 2]| egui::vec2(q[0] * owf, q[1] * ydiv);
                let nrm = |v: egui::Vec2| [v.x / owf, v.y / ydiv];
                let (mut lo, mut hi) = (egui::Vec2::splat(f32::MAX), egui::Vec2::splat(f32::MIN));
                for k in &sel {
                    let v = px(sp.points[*k].p);
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
                // A flat pick still needs a box with an inside.
                let pad = (egui::Vec2::splat(2.0) - (hi - lo)).max(egui::Vec2::ZERO) * 0.5;
                let (lo, hi) = (lo - pad, hi + pad);
                // The box is a QUAD that every group edit carries, so a
                // distort bends it and a rotate turns it. It survives
                // while the pick is the one it was made for and the
                // points are where the last edit left them (a common
                // shift — a group point drag, an undo of a move — just
                // moves it); anything else rebuilds it from the bounds.
                let now: Vec<egui::Vec2> = sel.iter().map(|k| px(sp.points[*k].p)).collect();
                let mut quad = [lo, egui::vec2(hi.x, lo.y), hi, egui::vec2(lo.x, hi.y)];
                if let Some((n, s, q, was)) = &self.spline.box_ {
                    if *n == node && *s == sel && was.len() == now.len() {
                        let d = now[0] - was[0];
                        if now.iter().zip(was).all(|(a, b)| (*a - *b - d).length() < 0.01) {
                            quad = q.map(|c| c + d);
                        }
                    }
                }
                let quad_h = |q: &[egui::Vec2; 4]| {
                    engine::codegen::corner_pin_forward_px(
                        &engine::graph::CornerPinParams::pinned(q.map(|v| [v.x, v.y])),
                        [1.0, 1.0],
                    )
                };
                let h = quad_h(&quad);
                let h_inv = engine::codegen::mat3_inv(&h);
                let unit = |u: egui::Vec2| apply_h(&h, u);
                let c = unit(egui::vec2(0.5, 0.5));
                // The pivot: the box centre unless a D-drag moved it for
                // THIS pick. Rotate and alt-scale turn about it, and it
                // rides every edit the points get, so it stays where it
                // sits on the shape.
                if self
                    .spline.pivot
                    .as_ref()
                    .is_some_and(|(n, s, _)| *n != node || *s != sel)
                {
                    self.spline.pivot = None;
                }
                let mut pv = self
                    .spline.pivot
                    .as_ref()
                    .map(|(_, _, q)| px(*q))
                    .unwrap_or(c);
                // Carry the pick, its bezier handles, the pivot and the
                // box through one map of authored pixels.
                let apply = |sp: &mut engine::graph::SplineParams,
                                 pv: &mut egui::Vec2,
                                 quad: &mut [egui::Vec2; 4],
                                 m: &dyn Fn(egui::Vec2) -> egui::Vec2| {
                    for i in &sel {
                        let q = sp.points[*i];
                        let a = px(q.p);
                        let w = m(a);
                        sp.points[*i].p = nrm(w);
                        sp.points[*i].h_in = nrm(m(a + px(q.h_in)) - w);
                        sp.points[*i].h_out = nrm(m(a + px(q.h_out)) - w);
                    }
                    *pv = m(*pv);
                    *quad = quad.map(|v| m(v));
                };
                let edge = |hx: i8, hy: i8| {
                    unit(egui::vec2((hx as f32 + 1.0) * 0.5, (hy as f32 + 1.0) * 0.5))
                };
                // AUTHORED pixels -> screen, through the shape transform
                // and then the downstream chain. The box, its handles
                // and the knob are computed from authored point
                // positions (so `from_screen`, which lands in authored
                // space, is their exact inverse); drawing them without
                // the transform would leave the whole rig sitting where
                // the shape ISN'T, and every grab would then resolve
                // against a position the handle is not at.
                let scr = |v: egui::Vec2| {
                    let t = xf.point([v.x / owf, v.y / ohf], sz);
                    out_to_pts(apply_h(&hd, egui::vec2(t[0] * owf, t[1] * ohf)))
                };
                let cpts = scr(c);
                let ppts = scr(pv);
                for k in 0..4 {
                    ink.seg(scr(quad[k]), scr(quad[(k + 1) % 4]), 1.0, 0.45, false);
                }
                // Handles ride 18px OUTSIDE the box: a corner handle sits
                // on the very point that defines it otherwise, and the
                // point (registered later) would eat every grab.
                let pushed = |hp: egui::Pos2| {
                    let d = hp - cpts;
                    if d.length() > 1e-3 { hp + d.normalized() * 18.0 } else { hp }
                };
                // Alt anchors the scale at the pivot; without it the
                // opposite handle stays pinned, Illustrator-style.
                let about_centre = ui.input(|i| i.modifiers.alt);
                let ctrl = ui.input(|i| i.modifiers.ctrl);
                const BOX_HANDLES: [(i8, i8); 8] = [
                    (-1, -1), (0, -1), (1, -1), (1, 0),
                    (1, 1), (0, 1), (-1, 1), (-1, 0),
                ];
                let (pv0, quad0) = (pv, quad);
                for (k, (hx, hy)) in BOX_HANDLES.iter().enumerate() {
                    // Handle positions come from the box as it was at the
                    // top of the frame: one drag edits per frame.
                    let hp = pushed(scr(edge(*hx, *hy)));
                    hot.push(hp);
                    let rs = ui.interact(
                        egui::Rect::from_center_size(hp, egui::vec2(22.0, 22.0)),
                        ui.id().with(("spline-scale", node, k)),
                        egui::Sense::drag(),
                    );
                    ink.rect(
                        egui::Rect::from_center_size(hp, egui::vec2(11.0, 11.0)),
                        2.0,
                        true,
                        0.0,
                        1.0,
                        rs.hovered(),
                    );
                    // Ctrl on a CORNER is the Transform's distort: that
                    // corner follows the pointer, the other three stay,
                    // and the pick — and the box — ride the homography
                    // between the two quads, handles too.
                    let corner = match (hx, hy) {
                        (-1, -1) => Some(0),
                        (1, -1) => Some(1),
                        (1, 1) => Some(2),
                        (-1, 1) => Some(3),
                        _ => None,
                    };
                    if let (Some(ci), true) = (corner, ctrl || self.spline.distort.is_some()) {
                        if rs.drag_started() && ctrl {
                            let origin = ui
                                .input(|i| i.pointer.press_origin())
                                .or(pointer_pos)
                                .unwrap_or(hp);
                            self.spline.distort = Some((
                                origin,
                                quad0,
                                ci,
                                sel.iter()
                                    .map(|i| {
                                        let q = sp.points[*i];
                                        (*i, q.p, q.h_in, q.h_out)
                                    })
                                    .collect(),
                            ));
                            self.spline.distort_pivot = pv0;
                        }
                        let live = self
                            .spline.distort
                            .as_ref()
                            .is_some_and(|(_, _, dci, _)| *dci == ci);
                        if live && rs.dragged() {
                            if let (Some(p), Some((origin, q0, _, start))) =
                                (pointer_pos, self.spline.distort.clone())
                            {
                                let mut q1 = q0;
                                q1[ci] += from_screen(p) - from_screen(origin);
                                let h1 = quad_h(&q1);
                                let h0_inv = engine::codegen::mat3_inv(&quad_h(&q0));
                                let warp = |v: egui::Vec2| apply_h(&h1, apply_h(&h0_inv, v));
                                for (i, p0, hin, hout) in &start {
                                    let a = px(*p0);
                                    let w = warp(a);
                                    sp.points[*i].p = nrm(w);
                                    sp.points[*i].h_in = nrm(warp(a + px(*hin)) - w);
                                    sp.points[*i].h_out = nrm(warp(a + px(*hout)) - w);
                                }
                                pv = warp(self.spline.distort_pivot);
                                quad = q1;
                            }
                        }
                        if live && rs.drag_stopped() {
                            self.spline.distort = None;
                        }
                        if live || ctrl {
                            continue;
                        }
                    }
                    if rs.dragged() && !rs.drag_started() {
                        if let Some(p) = pointer_pos {
                            // Scale in the BOX's own square: across a
                            // turned or bent box, "wider" means along
                            // its edges, not along the screen's axes.
                            let to_u = |v: egui::Vec2| apply_h(&h_inv, v);
                            let ua = if about_centre {
                                to_u(pv0)
                            } else {
                                egui::vec2((1 - *hx) as f32 * 0.5, (1 - *hy) as f32 * 0.5)
                            };
                            let un = to_u(from_screen(p)) - ua;
                            let up =
                                to_u(from_screen(p - widgets::rated_delta(ui, &rs))) - ua;
                            // A ratio only means anything along the axis
                            // the handle actually drives, and only once
                            // the anchor is far enough away to divide by.
                            let fx = if *hx != 0 && up.x.abs() > 0.01 { un.x / up.x } else { 1.0 };
                            let fy = if *hy != 0 && up.y.abs() > 0.01 { un.y / up.y } else { 1.0 };
                            let f = egui::vec2(fx, fy);
                            apply(sp, &mut pv, &mut quad, &|v| {
                                apply_h(&h, ua + (to_u(v) - ua) * f)
                            });
                        }
                    }
                }
                // Rotate: a knob on a stalk off the top edge.
                let top = scr(edge(0, -1));
                let up = top - cpts;
                let dir = if up.length() > 1e-3 {
                    up.normalized()
                } else {
                    egui::vec2(0.0, -1.0)
                };
                let knob = top + dir * 34.0;
                hot.push(knob);
                let rr = ui.interact(
                    egui::Rect::from_center_size(knob, egui::vec2(22.0, 22.0)),
                    ui.id().with(("spline-rot", node)),
                    egui::Sense::drag(),
                );
                ink.seg(top, knob, 1.0, 0.5, false);
                ink.circle_stroke(knob, 6.0, 1.6, 1.0, rr.hovered());
                if rr.dragged() && !rr.drag_started() {
                    if let Some(p) = pointer_pos {
                        let prev = p - widgets::rated_delta(ui, &rr);
                        let o = pv;
                        let mut d =
                            (from_screen(p) - o).angle() - (from_screen(prev) - o).angle();
                        use std::f32::consts::{PI, TAU};
                        if d > PI {
                            d -= TAU;
                        } else if d < -PI {
                            d += TAU;
                        }
                        let (sn, cs) = d.sin_cos();
                        apply(sp, &mut pv, &mut quad, &|v| {
                            let r = v - o;
                            o + egui::vec2(cs * r.x - sn * r.y, sn * r.x + cs * r.y)
                        });
                    }
                }
                // Translate: the open circle at the pivot (dragging any
                // selected point moves the group too). D held moves the
                // PIVOT alone, as on the Transform gizmo.
                hot.push(ppts);
                let rt = ui.interact(
                    egui::Rect::from_center_size(ppts, egui::vec2(26.0, 26.0)),
                    ui.id().with(("spline-move", node)),
                    egui::Sense::drag(),
                );
                ink.circle_stroke(ppts, 10.0, 1.6, 0.9, rt.hovered());
                let mut pivot_moved = false;
                if rt.dragged() && !rt.drag_started() {
                    if let Some(p) = pointer_pos {
                        let d = from_screen(p)
                            - from_screen(p - widgets::rated_delta(ui, &rt));
                        if ui.input(|i| i.key_down(egui::Key::D)) {
                            pv += d;
                            pivot_moved = true;
                        } else {
                            apply(sp, &mut pv, &mut quad, &|v| v + d);
                        }
                    }
                }
                // The pivot is kept once moved off the centre, and
                // follows the box from then on.
                if self.spline.pivot.is_some() || pivot_moved {
                    self.spline.pivot = Some((node, sel.clone(), nrm(pv)));
                }
                box_quad = Some(quad);
            }

            // S-press lays points down: near the outline it splits that
            // segment, anywhere else it appends. While the shape is being
            // drawn a plain press does too, always at the end. The point
            // lands on the PRESS, so the drag that follows (pen tool) pulls
            // its handles out (Bezier) or dials its tension (Catmull-Rom).
            // A plain click on empty canvas otherwise clears the pick — it
            // used to add a point, which left no way to start a marquee.
            let near_pt_at = |p: egui::Pos2| {
                sp.points.iter().any(|q| {
                    (to_scr(q.p) - p).length() < 12.0
                        || (to_scr([q.p[0] + q.h_in[0], q.p[1] + q.h_in[1]]) - p).length() < 10.0
                        || (to_scr([q.p[0] + q.h_out[0], q.p[1] + q.h_out[1]]) - p).length()
                            < 10.0
                }) || hot.iter().any(|h| (*h - p).length() < 12.0)
            };
            let press = ui.input(|i| {
                i.pointer.primary_pressed().then(|| i.pointer.press_origin()).flatten()
            });
            // Asked BEFORE anything below adds a point under the cursor.
            let press_near = press.is_some_and(near_pt_at);
            let click_near = pointer_pos.is_some_and(near_pt_at);
            if let Some(p) = press.filter(|p| rect.contains(*p) && !cmd && !scrub_held && !temp_live) {
                let shift = ui.input(|i| i.key_down(egui::Key::S));
                let first_hit = sp.points.first().is_some_and(|q| (to_scr(q.p) - p).length() < 12.0);
                if drawing && sp.open && sp.points.len() >= 3 && first_hit {
                    // Back on the first point: the shape closes.
                    sp.open = false;
                } else if (shift || drawing) && !press_near {
                    let mut seg: Option<(f32, usize)> = None;
                    if poly.len() > 1 && !drawing {
                        // Open: the last leg (last point back to the
                        // first) is not part of the curve.
                        let legs = if open_curve {
                            (sp.points.len().saturating_sub(1))
                                * engine::codegen::spline_subd(sp.points.len())
                        } else {
                            poly.len()
                        };
                        for j in 0..legs.min(poly.len()) {
                            // The outline is already transformed, so it
                            // goes to screen WITHOUT `to_scr` — that one
                            // is for authored points.
                            let a = out_scr(poly[j].0);
                            let b = out_scr(poly[(j + 1) % poly.len()].0);
                            let ab = b - a;
                            let t = ((p - a).dot(ab) / ab.length_sq().max(1e-6))
                                .clamp(0.0, 1.0);
                            let d = (a + ab * t - p).length();
                            if seg.is_none_or(|(bd, _)| d < bd) {
                                seg = Some((d, j));
                            }
                        }
                    }
                    // A new shape starts HARD; later points take the
                    // last one's feather.
                    let feather = sp.points.last().map(|q| q.feather).unwrap_or(0.0);
                    let at = match seg {
                        Some((d, j)) if d < 14.0 => {
                            let ci = j / engine::codegen::spline_subd(sp.points.len());
                            (ci + 1).min(sp.points.len())
                        }
                        _ => sp.points.len(),
                    };
                    // The first point of a roto shape opens it: it stays
                    // an open outline until it is closed on purpose.
                    if sp.points.is_empty() && !is_field {
                        sp.open = true;
                    }
                    sp.points.insert(at, engine::graph::SplinePoint::new(norm(p), feather));
                    struct_edit = Some(SplineEdit::Inserted(at));
                    self.spline.pen = Some((node, shape_sel, at, p));
                    if group_ui {
                        self.spline.sel.clear();
                    }
                }
            }
            if ui.input(|i| i.pointer.primary_clicked()) && struct_edit.is_none() {
                if pointer_pos.is_some_and(|p| rect.contains(p)) {
                    let shift = ui.input(|i| i.key_down(egui::Key::S));
                    let near_pt = click_near;
                    if let (false, false, false, Some(k)) = (shift, drawing, near_pt, other_hit) {
                        // Another shape of the container: pick it.
                        innards::select(ui.ctx(), node, k);
                    } else if !shift && !drawing && !near_pt && group_ui && !temp_live {
                        self.spline.sel.clear();
                    }
                }
            }
            // The pen drag: the point just laid down follows the press, and
            // the hand pulls its tangents. Absolute from the press, so an
            // animated shape keys cleanly.
            if let Some((pn, ps, pi, po)) = self.spline.pen {
                let down = ui.input(|i| i.pointer.primary_down());
                if pn != node || ps != shape_sel || pi >= sp.points.len() || !down {
                    if pn == node || !down {
                        self.spline.pen = None;
                    }
                } else if let Some(p) = pointer_pos {
                    let d = (p - po).length();
                    if d > 3.0 {
                        match sp.mode {
                            engine::graph::SplineMode::Bezier => {
                                let q = norm(p);
                                let c = sp.points[pi].p;
                                let h = [q[0] - c[0], q[1] - c[1]];
                                sp.points[pi].h_out = h;
                                sp.points[pi].h_in = [-h[0], -h[1]];
                            }
                            engine::graph::SplineMode::CatmullRom => {
                                // A short pull is a corner, ~80 px the
                                // ordinary curve, further out looser.
                                sp.points[pi].tension = (1.0 - d / 80.0).clamp(-1.0, 1.0);
                                ink.seg(po, p, 1.0, 0.6, true);
                                ink.circle_stroke(p, 3.5, 1.2, 0.9, true);
                            }
                            _ => {}
                        }
                    }
                }
            }

            let n_pts = sp.points.len();
            let mut remove: Option<usize> = None;
            let bezier = sp.mode == engine::graph::SplineMode::Bezier;
            for i in 0..n_pts {
                let pt = sp.points[i];
                let hp = to_scr(pt.p);
                // Bezier handles: two little knobs off the point.
                // Registered BEFORE the point: egui gives the drag to
                // the last widget under the cursor, and a handle still
                // sitting on its point (zero offset) would otherwise
                // swallow every attempt to move the point itself.
                if bezier {
                    for (hi, off) in [(false, pt.h_in), (true, pt.h_out)] {
                        let hpos =
                            to_scr([pt.p[0] + off[0], pt.p[1] + off[1]]);
                        let hres = ui.interact(
                            egui::Rect::from_center_size(hpos, egui::vec2(14.0, 14.0)),
                            ui.id().with(("spline-h", node, i, hi)),
                            egui::Sense::drag(),
                        );
                        ink.seg(hp, hpos, 0.9, 0.5, false);
                        ink.circle_stroke(hpos, 3.5, 1.2, 0.9, hres.hovered());
                        if hres.dragged() && !hres.drag_started() {
                            if let Some(p) = pointer_pos {
                                let d = from_screen(p)
                                    - from_screen(p - widgets::rated_delta(ui, &hres));
                                let (dx, dy) = (d.x / owf, d.y / ohf);
                                if hi {
                                    sp.points[i].h_out[0] += dx;
                                    sp.points[i].h_out[1] += dy;
                                    if !ui.input(|inp| inp.modifiers.alt) {
                                        // Mirror for a smooth join.
                                        sp.points[i].h_in = [
                                            -sp.points[i].h_out[0],
                                            -sp.points[i].h_out[1],
                                        ];
                                    }
                                } else {
                                    sp.points[i].h_in[0] += dx;
                                    sp.points[i].h_in[1] += dy;
                                    if !ui.input(|inp| inp.modifiers.alt) {
                                        sp.points[i].h_out = [
                                            -sp.points[i].h_in[0],
                                            -sp.points[i].h_in[1],
                                        ];
                                    }
                                }
                            }
                        }
                    }
                }
                let hr = ui.interact(
                    egui::Rect::from_center_size(hp, egui::vec2(18.0, 18.0)),
                    ui.id().with(("spline-pt", node, i)),
                    egui::Sense::click_and_drag(),
                );
                let picked = group_ui && self.spline.sel.contains(&i);
                ink.circle_fill(hp, if picked { 5.0 } else { 4.0 }, 1.0, hr.hovered());
                if picked {
                    ink.circle_stroke(hp, 8.5, 1.2, 0.9, hr.hovered());
                }
                // D-click removes the point (ctrl/cmd-click are the
                // viewer's layer pick now).
                if hr.clicked() && ui.input(|inp| inp.key_down(egui::Key::D)) {
                    remove = Some(i);
                } else if hr.clicked() && group_ui && !temp_live {
                    // S toggles the point in the pick; a plain click
                    // makes it the whole pick.
                    if ui.input(|inp| inp.key_down(egui::Key::S)) {
                        if !self.spline.sel.remove(&i) {
                            self.spline.sel.insert(i);
                        }
                    } else {
                        self.spline.sel.clear();
                        self.spline.sel.insert(i);
                    }
                }
                if hr.drag_started() && !temp_live {
                    // Grabbing a point outside the pick makes it the
                    // pick: a drag moves what is under the cursor.
                    if group_ui && !self.spline.sel.contains(&i) {
                        self.spline.sel.clear();
                        self.spline.sel.insert(i);
                    }
                    let group: Vec<usize> = if group_ui && self.spline.sel.len() > 1 {
                        self.spline.sel.iter().copied().filter(|k| *k < n_pts).collect()
                    } else {
                        vec![i]
                    };
                    // Measure from the press point, not from egui's first
                    // drag_delta — that one spans motion from before the
                    // press, which is why this used to skip the first
                    // frame (and lose a frame of travel with it).
                    let origin = ui
                        .input(|inp| inp.pointer.press_origin())
                        .or(pointer_pos)
                        .unwrap_or(hp);
                    self.spline.tension0 = group
                        .iter()
                        .filter(|k| **k < sp.points.len())
                        .map(|k| (*k, sp.points[*k].tension))
                        .collect();
                    self.spline.drag = Some((
                        origin,
                        group
                            .iter()
                            .filter(|k| **k < sp.points.len())
                            .map(|k| {
                                let q = sp.points[*k];
                                (*k, q.p, q.feather, q.bias)
                            })
                            .collect(),
                    ));
                }
                if hr.dragged() {
                    if let (Some(p), Some((origin, start))) =
                        (pointer_pos, self.spline.drag.clone())
                    {
                        let band_key = ui.input(|inp| {
                            [egui::Key::O, egui::Key::I]
                                .into_iter()
                                .find(|k| inp.key_down(*k))
                        });
                        if ui.input(|inp| inp.key_down(egui::Key::K)) {
                            // K held: tension. Right tightens toward a
                            // corner, left loosens; 200 screen px is the
                            // whole 0..1.
                            let dv = (p.x - origin.x) / 200.0;
                            for (k, t0) in &self.spline.tension0 {
                                if let Some(q) = sp.points.get_mut(*k) {
                                    q.tension = (t0 + dv).clamp(-1.0, 1.0);
                                }
                            }
                        } else if let Some(bk) = band_key {
                            // O / I held: this point's feather grows
                            // OUTWARD (its inner edge stays) or INWARD
                            // (its outer edge does) — the band's centre
                            // moves by half of what the width gains.
                            // Image pixels of horizontal travel.
                            let dv = (from_screen(p) - from_screen(origin)).x;
                            let s = if bk == egui::Key::O { 0.5 } else { -0.5 };
                            for (k, _, f0, b0) in &start {
                                let f = (f0 + dv).clamp(0.0, 500.0);
                                sp.points[*k].feather = f;
                                sp.points[*k].bias = b0 + s * (f - f0);
                            }
                        } else if ui.input(|inp| inp.modifiers.alt) {
                            // Alt-drag: feather, for the whole pick.
                            // Right/up softens.
                            let d = p - origin;
                            let dv = (d.x - d.y) * 0.5;
                            for (k, _, f0, _) in &start {
                                sp.points[*k].feather = (f0 + dv).clamp(0.0, 500.0);
                            }
                        } else {
                            let d_out = from_screen(p) - from_screen(origin);
                            for (k, p0, _, _) in &start {
                                sp.points[*k].p[0] = p0[0] + d_out.x / owf;
                                sp.points[*k].p[1] = p0[1] + d_out.y / ohf;
                            }
                        }
                    }
                }
                if hr.drag_stopped() {
                    self.spline.drag = None;
                }
            }
            // The held-key edit lands LAST, so it wins over any widget the
            // press happened to land on.
            if let Some((_, k, o, pv, start)) = self.spline.temp.clone() {
                let p = pointer_pos.unwrap_or(o);
                let dx = p.x - o.x;
                let n = |v: egui::Vec2| [v.x / owf, v.y / ydiv];
                let px = |q: [f32; 2]| egui::vec2(q[0] * owf, q[1] * ydiv);
                // Point about the pivot, and a handle (an offset) alone.
                let (s, (sn, cs), d) = match k {
                    egui::Key::E => ((dx / 150.0).exp(), (0.0, 1.0), egui::Vec2::ZERO),
                    egui::Key::R => (1.0, self.spline_temp_turn(from_screen(p) - pv, p - o), egui::Vec2::ZERO),
                    _ => (1.0, (0.0, 1.0), from_screen(p) - from_screen(o)),
                };
                let lin = |v: egui::Vec2| egui::vec2(cs * v.x - sn * v.y, sn * v.x + cs * v.y) * s;
                for (i, p0, hin, hout) in &start {
                    if let Some(q) = sp.points.get_mut(*i) {
                        q.p = n(pv + lin(px(*p0) - pv) + d);
                        q.h_in = n(lin(px(*hin)));
                        q.h_out = n(lin(px(*hout)));
                    }
                }
                // The group box and its pivot ride the same map.
                if self.spline.temp_box.is_none() {
                    if let Some(q) = box_quad {
                        let piv = self.spline.pivot.as_ref().map(|(_, _, q)| px(*q));
                        self.spline.temp_box = Some((q, piv));
                    }
                }
                if let Some((q0, piv0)) = self.spline.temp_box {
                    let m = |v: egui::Vec2| pv + lin(v - pv) + d;
                    box_quad = Some(q0.map(m));
                    if let (Some(p0), Some((_, _, q))) = (piv0, self.spline.pivot.as_mut()) {
                        *q = n(m(p0));
                    }
                }
                // The temporary pivot, and the throw from it.
                let ps = to_scr(n(pv));
                ink.seg(ps - egui::vec2(7.0, 0.0), ps + egui::vec2(7.0, 0.0), 1.2, 1.0, true);
                ink.seg(ps - egui::vec2(0.0, 7.0), ps + egui::vec2(0.0, 7.0), 1.2, 1.0, true);
                ink.seg(o, p, 1.0, 0.5, false);
                if !ui.input(|i| i.pointer.primary_down()) {
                    self.spline.temp = None;
                    self.spline.temp_box = None;
                    self.spline.temp_rot = None;
                }
            }
            // Remember the box around where the pick now is, so the next
            // frame finds it bent, turned or moved the way it was left.
            if let Some(q) = box_quad {
                let now = sel
                    .iter()
                    .filter_map(|k| sp.points.get(*k))
                    .map(|q| egui::vec2(q.p[0] * owf, q.p[1] * ydiv))
                    .collect();
                self.spline.box_ = Some((node, sel.clone(), q, now));
            }
            if let Some(i) = remove {
                sp.points.remove(i);
                struct_edit = Some(SplineEdit::Removed(i, n_pts));
            }
            // The pick is by index, so a structural edit has to slide
            // it exactly like the curves; and the count it was made
            // against moves with it.
            if group_ui {
                if let Some(e) = struct_edit {
                    remap_spline_sel(&mut self.spline.sel, e);
                }
                self.spline.sel_n = spw.points.len();
            }

            // Write back: point channels key like any other param when
            // animated (or under autokey); structure changes (add /
            // remove / handles) always land on the authored node.
            if spw != before_sp && is_field {
                if let Some(editor::ENode { kind, .. }) = self.egraph.node_mut(node) {
                    if let NodeKind::Field(engine::field::FieldNode::Spline {
                        points,
                        mode,
                        ..
                    }) = kind
                    {
                        *points = spw.points.clone();
                        *mode = spw.mode;
                    }
                    if let NodeKind::GeoNode(
                        engine::geo::GeoNode::Spline { points, mode, .. }
                        | engine::geo::GeoNode::Stroke { points, mode, .. },
                    ) = kind
                    {
                        *points = spw.points.clone();
                        *mode = spw.mode;
                    }
                    if let NodeKind::SdfNode(engine::sdf::SdfNode::Spline {
                        points, mode, ..
                    }) = kind
                    {
                        *points = spw
                            .points
                            .iter()
                            .cloned()
                            .map(|mut q| {
                                q.p[0] -= sdf_off[0];
                                q.p[1] -= sdf_off[1];
                                q
                            })
                            .collect();
                        *mode = spw.mode;
                    }
                }
            } else if spw != before_sp {
                let autokey = self.autokey;
                // The patch-driven channels, under shape-0 names for the
                // shape being edited (the write-back runs in its terms).
                let driven: Vec<(u64, anim::Channel)> = if shape_sel == 0 {
                    patch_driven.clone()
                } else {
                    patch_driven
                        .iter()
                        .filter_map(|(n, c)| match anim::shape_base(*c) {
                            Some((s, b)) if *n == node && s == shape_sel => Some((*n, b)),
                            _ => None,
                        })
                        .collect()
                };
                if let Some(n) = self.egraph.node_mut(node) {
                    let editor::ENode { kind, anim, .. } = n;
                    if let NodeKind::Spline(full) = kind {
                        // The edited shape, authored, as a one-shape
                        // Spline: every helper below speaks that.
                        let mut auth = full.single(shape_sel);
                        let structural = auth.points.len() != spw.points.len();
                        auth.mode = spw.mode;
                        // Open/closed is authored, never keyed.
                        auth.open = spw.open;
                        if structural {
                            // POINTS only. `spw` came from `live_kind`,
                            // so its matte controls and shape transform
                            // are EVALUATED values — writing those back
                            // would bake a curve into authored data (and
                            // flood undo) the moment a point was added
                            // to an animated shape.
                            auth.points = spw.points.clone();
                        } else {
                            for (i, q) in spw.points.iter().enumerate() {
                                auth.points[i].h_in = q.h_in;
                                auth.points[i].h_out = q.h_out;
                                // Tension is authored like the handles.
                                auth.points[i].tension = q.tension;
                            }
                        }
                        let mut auth_kind = NodeKind::Spline(auth);
                        anim::in_shape(anim, shape_sel, |anim| {
                            if structural {
                                match struct_edit {
                                    Some(SplineEdit::Inserted(at)) => {
                                        anim::spline_point_inserted(
                                            anim,
                                            &spw.points,
                                            at,
                                            cur_t,
                                        )
                                    }
                                    Some(SplineEdit::Removed(at, was)) => {
                                        anim::spline_point_removed(anim, at, was)
                                    }
                                    None => {}
                                }
                            } else {
                                let before = NodeKind::Spline(before_sp);
                                let after = NodeKind::Spline(spw);
                                spline_keyed_write_back(
                                    anim, &mut auth_kind, &before, &after, cur_t, autokey,
                                    &driven, node,
                                );
                            }
                        });
                        if let NodeKind::Spline(auth) = auth_kind {
                            full.apply_single(shape_sel, &auth);
                        }
                    }
                }
            }
        }
    }
}
