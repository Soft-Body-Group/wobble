//! The per-node gizmo pass: Transform's manipulator, Gradient, Warp, GridWarp, SoftCrop, Mirror, the lens nodes and Godrays, with motion paths.
//! Moved out of `viewer_ui` (lib.rs) unchanged.

use crate::*;

impl Gfx {
    /// The per-node gizmo pass: Transform's manipulator, Gradient, Warp, GridWarp, SoftCrop, Mirror, the lens nodes and Godrays, with motion paths.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gizmo_transform(
        &mut self,
        ui: &mut egui::Ui,
        cmd: bool,
        scrub_held: bool,
        cur_t: u64,
        owf: f32,
        ohf: f32,
        out_to_pts: &impl Fn(egui::Vec2) -> egui::Pos2,
        pts_to_out: &impl Fn(egui::Pos2) -> egui::Vec2,
        rect: egui::Rect,
        selected: &Vec<u64>,
        patch_driven: &Vec<(u64, anim::Channel)>,
        nav: &egui::Response,
        path_span: (u64, u64),
        show_path: bool,
        ink: &mut ink::InkShapes,
        hud_key_list: &mut Vec<hudkeys::HudKey>,
        focus_node: &mut Option<(u64, layerpick::Reveal)>,
        path_edit: &mut Option<(u64, f64, [f32; 2], (f32, f32))>,
        path_points: &mut usize,
    ) {
        for node in selected.iter().copied() {
            ink.set_tint(self.hud_ink(node));
            // Affine space = SHADER space, which anchors every spatial
            // node to ITS OWN format (stamped per-node), so the
            // deformation chain resolves each node's local resolution.
            // Through a UV Warp the chain is not one matrix: `DownMap`
            // bends each point through the warp's read-back map.
            let dm = downmap::downstream_map(
                &self.egraph,
                node,
                &|id| self.node_res_f32(id).unwrap_or((owf, ohf)),
                cur_t,
                &|id| self.uv_grids.get(&id).map(|g| g.1.clone()),
            );
            let to_screen = |o: egui::Vec2| out_to_pts(dm.fwd(o));
            let from_screen = |p: egui::Pos2| dm.inv(pts_to_out(p));
            // Gizmos draw from and drag against the EVALUATED params —
            // what is actually on screen — then write back below (keys
            // on animated channels, authored values otherwise).
            let Some(ekind) = self.live_kind(node, cur_t) else {
                continue;
            };
            // The image this transform moves is ITS upstream's format —
            // through a fixed-format merge that is larger than the view,
            // the outline and corner handles must span it, not the view.
            let size = self.node_res_f32(node).unwrap_or((owf, ohf));
            // THE PATH the element follows, when asked for (the
            // transport's `path` chip). Drawn before the manipulator so
            // the handles stay on top of it, and sampled over the
            // viewer's range — the frames you are working in.
            if show_path {
                // Params are FULL-res pixels on a possibly-proxy canvas,
                // exactly as the Transform arm below handles it.
                let k = if self.proxy > 0.0 { self.proxy } else { 1.0 };
                let full = (size.0 / k, size.1 / k);
                let pts = self
                    .egraph
                    .node(node)
                    .map(|n| {
                        anim::motion_path(&n.kind, &n.anim, path_span, &|kd| {
                            element_point(kd, full)
                        })
                    })
                    .unwrap_or_default();
                if pts.len() >= 2 {
                    *path_points += pts.len();
                    let col = self.hud_ink_c32(node).unwrap_or_else(theme::animated);
                    let scr = |a: [f32; 2]| to_screen(egui::vec2(a[0], a[1]) * k);
                    ui.painter().add(egui::Shape::line(
                        pts.iter().map(|q| scr(q.at)).collect(),
                        egui::Stroke::new(1.0, col.gamma_multiply(0.65)),
                    ));
                    // Keys are where the shape was decided: mark them,
                    // and let them be GRABBED — dragging one moves that
                    // keyframe's position, at its own frame, wherever
                    // the playhead happens to be.
                    //
                    // Registered BEFORE the manipulators below, so a key
                    // sitting under the pivot handle loses the tie: egui
                    // gives a press to the last widget that claimed the
                    // spot, and the handle you are looking at should win.
                    for q in pts.iter().filter(|q| q.key) {
                        let at = scr(q.at);
                        let r = ui.interact(
                            egui::Rect::from_center_size(at, egui::vec2(13.0, 13.0)),
                            ui.id().with(("path-key", node, q.frame.to_bits())),
                            egui::Sense::click_and_drag(),
                        );
                        let live = self
                            .path_drag
                            .is_some_and(|(n, f, _)| n == node && f == q.frame);
                        if r.drag_started() {
                            // The offset is measured from the PRESS
                            // ORIGIN, not from where the pointer is now:
                            // egui reports a drag only once it has moved
                            // past its threshold, so "now" is already a
                            // dozen pixels in and the key would trail the
                            // cursor by exactly that much for the whole
                            // gesture. (Same shape as the summed-delta
                            // trap the spline gizmo hit: anchor at the
                            // press, then follow the pointer absolutely.)
                            if let Some(o) = ui.input(|i| i.pointer.press_origin()) {
                                self.path_drag = Some((node, q.frame, at - o));
                            }
                        }
                        let hot = live || r.hovered();
                        ui.painter().circle_filled(
                            at,
                            if hot { 4.5 } else { 2.5 },
                            if hot { theme::selection() } else { theme::keyframe() },
                        );
                        if hot {
                            let f = q.frame;
                            let label = if (f - f.round()).abs() < 1e-6 {
                                format!("{}", f.round() as i64)
                            } else {
                                format!("{f:.2}")
                            };
                            r.on_hover_text(format!(
                                "keyframe at {label} — drag to move it, at its own frame"
                            ));
                        }
                    }
                    // And where it is NOW, so the path reads as a clock.
                    if let Some(at) = element_point(&ekind, full) {
                        ui.painter().circle_filled(scr(at), 3.0, theme::playhead());
                    }
                    // Apply a live drag: the pointer plus the grab
                    // offset, back through the same map, written as a
                    // key at the frame that key is on.
                    if let Some((n, frame, grab)) = self.path_drag.filter(|(n, ..)| *n == node)
                    {
                        if ui.input(|i| i.pointer.primary_down()) {
                            if let Some(p) = ui.ctx().pointer_interact_pos() {
                                let want = from_screen(p + grab) / k;
                                *path_edit = Some((n, frame, [want.x, want.y], full));
                            }
                        } else {
                            self.path_drag = None;
                        }
                    }
                }
            }
            match ekind {
                NodeKind::Transform(mut tw) => {
                    // Params are FULL-res pixels (codegen scales translate
                    // by `proxy_scale` at pack) but this canvas is the
                    // proxy one. Run the manipulator in full-res space —
                    // widen the size, scale to and from the screen — or a
                    // drag writes proxy pixels into a full-res param and
                    // the image moves half as far as the pointer. Exact:
                    // the factor is a power of two.
                    let k = if self.proxy > 0.0 { self.proxy } else { 1.0 };
                    let size = (size.0 / k, size.1 / k);
                    let to_screen = |o: egui::Vec2| to_screen(o * k);
                    let from_screen = |p: egui::Pos2| from_screen(p) / k;
                    let before_t = tw;
                    if let Some(r) = pivot_pick(ui, &tw, size, &to_screen) {
                        *focus_node = Some((node, r));
                    }
                    // A SoftCrop upstream sizes the handles to its box.
                    let frame = self.soft_frame(node, cur_t).unwrap_or(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(size.0, size.1),
                    ));
                    let moved = layerui::transform_gizmo(
                        ui,
                        ink,
                        ui.id().with(node),
                        &mut tw,
                        size,
                        frame,
                        &to_screen,
                        &from_screen,
                    );
                    if moved {
                        if let Some(n) = self.egraph.node_mut(node) {
                            let editor::ENode { kind, anim, .. } = n;
                            if let NodeKind::Transform(auth) = kind {
                                auth.scale_gang = tw.scale_gang;
                            }
                            keyed_write_back(
                                anim,
                                kind,
                                &NodeKind::Transform(before_t),
                                &NodeKind::Transform(tw),
                                cur_t,
                                anim::TRANSFORM,
                                self.autokey,
                                &patch_driven,
                                node,
                            );
                        }
                    }
                }
                // The soft crop's own box: four corners and four edges,
                // each placed ABSOLUTELY under the pointer (an animated
                // edge must not sum per-frame deltas), in the node's own
                // normalised format.
                // The gradient's axis on the picture: the ramp ribbon,
                // its two ends (rotate + scale about the centre, shift
                // snaps to 15°), the centre, and a marker per stop
                // (drag slides it along the axis, alt-click removes,
                // double-click the axis adds one). The selected stop
                // carries its swatch — the same selection the panel's
                // bar shows. Every drag is absolute under the pointer.
                // The godrays' source: a sun on the picture, dragged
                // ABSOLUTELY under the pointer (press offset kept, so a
                // grab off-centre does not jump), keyed through the
                // channels like any other value. A few faint spokes show
                // which way the shafts stream and how far they reach.
                NodeKind::Godrays(mut gr) => {
                    let before = gr;
                    let (w, h) = size;
                    let at = to_screen(egui::vec2(gr.center[0] * w, gr.center[1] * h));
                    let rs = ui.interact(
                        egui::Rect::from_center_size(at, egui::vec2(30.0, 30.0)),
                        ui.id().with(("godrays-src", node)),
                        egui::Sense::drag(),
                    );
                    let hot = rs.hovered() || rs.dragged();
                    // Spokes: from each frame corner towards the source,
                    // the stretch a ray at that corner gathers over.
                    for c in [[0.0, 0.0], [w, 0.0], [0.0, h], [w, h]] {
                        let pc = to_screen(egui::vec2(c[0], c[1]));
                        let reach = pc + (at - pc) * gr.length.clamp(0.0, 1.0);
                        ink.seg(pc, reach, 1.0, 0.35, false);
                    }
                    ink.circle_stroke(at, 7.0, 1.5, 0.9, hot);
                    for k in 0..8 {
                        let a = k as f32 * std::f32::consts::TAU / 8.0;
                        let d = egui::vec2(a.cos(), a.sin());
                        ink.seg(at + d * 10.0, at + d * 15.0, 1.5, 0.9, hot);
                    }
                    if rs.drag_started() {
                        if let Some(o) = ui.input(|i| i.pointer.press_origin()) {
                            ui.ctx().data_mut(|d| {
                                d.insert_temp(ui.id().with(("godrays-grab", node)), at - o)
                            });
                        }
                    }
                    if rs.dragged() {
                        let grab: egui::Vec2 = ui
                            .ctx()
                            .data(|d| d.get_temp(ui.id().with(("godrays-grab", node))))
                            .unwrap_or_default();
                        if let Some(p) = ui.ctx().pointer_latest_pos() {
                            let o = from_screen(p + grab);
                            gr.center = [o.x / w, o.y / h];
                        }
                    }
                    if hot {
                        rs.on_hover_text("godrays source — drag to move the light");
                    }
                    if gr != before {
                        if let Some(n) = self.egraph.node_mut(node) {
                            let editor::ENode { kind, anim, .. } = n;
                            keyed_write_back(
                                anim,
                                kind,
                                &NodeKind::Godrays(before),
                                &NodeKind::Godrays(gr),
                                cur_t,
                                nodes::godrays::CHANNELS,
                                self.autokey,
                                &patch_driven,
                                node,
                            );
                        }
                    }
                }
                // The lens flare's light: the same sun handle as the
                // Godrays' source (absolute under the pointer, press
                // offset kept, keyed through its channels), plus the
                // axis through the frame centre the ghosts sit on and a
                // ring at each ghost's place, so moving the light shows
                // where the ghosts will land.
                NodeKind::LensFlare(mut lf) => {
                    let before = lf;
                    let (w, h) = size;
                    let at = to_screen(egui::vec2(lf.center[0] * w, lf.center[1] * h));
                    let mid = to_screen(egui::vec2(0.5 * w, 0.5 * h));
                    let rs = ui.interact(
                        egui::Rect::from_center_size(at, egui::vec2(30.0, 30.0)),
                        ui.id().with(("lensflare-src", node)),
                        egui::Sense::drag(),
                    );
                    let hot = rs.hovered() || rs.dragged();
                    if lf.ghosts > 0.0 && lf.ghost_n() > 0 {
                        ink.seg(at, at + (mid - at) * 2.0, 1.0, 0.3, false);
                        let scale = (to_screen(egui::vec2(0.0, h)) - to_screen(egui::vec2(0.0, 0.0))).length();
                        for i in 0..lf.ghost_n() {
                            let (t, s, _) = lf.ghost(i);
                            let c = at + (mid - at) * t;
                            ink.circle_stroke(c, (lf.ghost_size * s * scale).max(2.0), 1.0, 0.35, false);
                        }
                    }
                    ink.circle_stroke(at, 7.0, 1.5, 0.9, hot);
                    for k in 0..8 {
                        let a = k as f32 * std::f32::consts::TAU / 8.0;
                        let d = egui::vec2(a.cos(), a.sin());
                        ink.seg(at + d * 10.0, at + d * 15.0, 1.5, 0.9, hot);
                    }
                    if rs.drag_started() {
                        if let Some(o) = ui.input(|i| i.pointer.press_origin()) {
                            ui.ctx().data_mut(|d| {
                                d.insert_temp(ui.id().with(("lensflare-grab", node)), at - o)
                            });
                        }
                    }
                    if rs.dragged() {
                        let grab: egui::Vec2 = ui
                            .ctx()
                            .data(|d| d.get_temp(ui.id().with(("lensflare-grab", node))))
                            .unwrap_or_default();
                        if let Some(p) = ui.ctx().pointer_latest_pos() {
                            let o = from_screen(p + grab);
                            lf.center = [o.x / w, o.y / h];
                        }
                    }
                    if hot {
                        rs.on_hover_text("lens flare light — drag to move it");
                    }
                    if lf != before {
                        if let Some(n) = self.egraph.node_mut(node) {
                            let editor::ENode { kind, anim, .. } = n;
                            keyed_write_back(
                                anim,
                                kind,
                                &NodeKind::LensFlare(before),
                                &NodeKind::LensFlare(lf),
                                cur_t,
                                &nodes::lensflare::dyn_channels(),
                                self.autokey,
                                &patch_driven,
                                node,
                            );
                        }
                    }
                }
                NodeKind::Gradient(mut gp) => {
                    let before = gp.clone();
                    let (w, h) = size;
                    let working = self.egraph.working;
                    let spec = pick_spec(working, true);
                    let pt = |v: [f32; 2]| to_screen(egui::vec2(v[0], v[1]));
                    let (a0, a1) = gp.ends(w, h);
                    let (p0, p1) = (pt(a0), pt(a1));
                    let pc = pt([gp.center[0] * w, gp.center[1] * h]);
                    let axis = p1 - p0;
                    let dir = axis / axis.length().max(1.0);
                    let nrm = egui::vec2(-dir.y, dir.x);
                    let mods = ui.input(|i| i.modifiers);
                    let pointer = ui.ctx().pointer_latest_pos();
                    // The ramp as a ribbon beside the axis: what the
                    // stops are ON, read without looking at the panel.
                    {
                        let painter = ui.painter();
                        let mut mesh = egui::Mesh::default();
                        const N: usize = 48;
                        for k in 0..=N {
                            let t = k as f32 / N as f32;
                            let c = nodes::gradient::swatch(gp.sample(t, working), &spec);
                            let at = p0 + axis * t;
                            mesh.colored_vertex(at - nrm * 2.0, c);
                            mesh.colored_vertex(at - nrm * 7.0, c);
                            if k > 0 {
                                let b = (2 * k) as u32;
                                mesh.add_triangle(b - 2, b - 1, b);
                                mesh.add_triangle(b - 1, b + 1, b);
                            }
                        }
                        painter.add(egui::Shape::mesh(mesh));
                    }
                    ink.seg(p0, p1, 1.5, 0.9, false);
                    let from_px = |p: egui::Pos2| {
                        let o = from_screen(p);
                        [o.x, o.y]
                    };
                    // Ends.
                    for (k, at) in [(0usize, p0), (1, p1)] {
                        let rs = ui.interact(
                            egui::Rect::from_center_size(at, egui::vec2(20.0, 20.0)),
                            ui.id().with(("grad-end", node, k)),
                            egui::Sense::drag(),
                        );
                        ink.circle_stroke(at, 6.0, 1.5, 0.9, rs.hovered() || rs.dragged());
                        if let (true, Some(p)) = (rs.dragged(), pointer) {
                            let o = from_px(p);
                            let c = [gp.center[0] * w, gp.center[1] * h];
                            let s = if k == 0 { -1.0 } else { 1.0 };
                            let v = [(o[0] - c[0]) * s, (o[1] - c[1]) * s];
                            let mut ang = (-v[1]).atan2(v[0]).to_degrees();
                            if mods.shift {
                                ang = (ang / 15.0).round() * 15.0;
                            }
                            // Unwrapped against where it was, so a keyed
                            // angle never spins the long way round.
                            let d = (ang - before.angle).rem_euclid(360.0);
                            gp.angle = before.angle + if d > 180.0 { d - 360.0 } else { d };
                            let unit = engine::graph::GradientParams {
                                scale: 1.0,
                                ..gp.clone()
                            }
                            .half_len(w, h);
                            gp.scale = (v[0].hypot(v[1]) / unit).max(0.01);
                        }
                    }
                    // Centre.
                    let rs = ui.interact(
                        egui::Rect::from_center_size(pc, egui::vec2(18.0, 18.0)),
                        ui.id().with(("grad-center", node)),
                        egui::Sense::drag(),
                    );
                    let hot = rs.hovered() || rs.dragged();
                    ink.seg(pc - egui::vec2(6.0, 0.0), pc + egui::vec2(6.0, 0.0), 1.5, 0.9, hot);
                    ink.seg(pc - egui::vec2(0.0, 6.0), pc + egui::vec2(0.0, 6.0), 1.5, 0.9, hot);
                    if let (true, Some(p)) = (rs.dragged(), pointer) {
                        let o = from_px(p);
                        gp.center = [o[0] / w, o[1] / h];
                    }
                    // Stops, markers pointing at the axis from the side
                    // away from the ribbon.
                    let n = gp.stops.len();
                    let sel = nodes::gradient::selected(ui.ctx(), node, n);
                    let mut remove = None;
                    let mut on_stop = false;
                    for i in 0..n {
                        let s = gp.stops[i];
                        let tip = p0 + axis * s.pos.clamp(0.0, 1.0) + nrm * 2.0;
                        let rs = ui.interact(
                            egui::Rect::from_center_size(tip + nrm * 9.0, egui::vec2(16.0, 18.0)),
                            ui.id().with(("grad-stop", node, i)),
                            egui::Sense::click_and_drag(),
                        );
                        on_stop |= rs.hovered();
                        if rs.clicked() && mods.alt {
                            remove = Some(i);
                        } else if rs.clicked() || rs.drag_started() {
                            nodes::gradient::select(ui.ctx(), node, i);
                        }
                        if let (true, Some(p)) = (rs.dragged(), pointer) {
                            gp.stops[i].pos = gp.t_at(from_px(p), w, h).clamp(0.0, 1.0);
                        }
                        nodes::gradient::paint_marker(
                            ui.painter(),
                            tip,
                            nrm,
                            nodes::gradient::swatch(s.color, &spec),
                            i == sel,
                            rs.hovered(),
                        );
                    }
                    // Double-click on the axis adds a stop there.
                    let dbl = ui.input(|i| i.pointer.button_double_clicked(egui::PointerButton::Primary));
                    let mut added = None;
                    if let (true, false, Some(p)) = (dbl, on_stop, pointer) {
                        let t = ((p - p0).dot(dir) / axis.length().max(1.0)).clamp(0.0, 1.0);
                        let off = (p - (p0 + axis * t)).length();
                        if off < 8.0 && rect.contains(p) {
                            added = Some(t);
                        }
                    }
                    // The selected stop's colour, beside its marker.
                    let mut recolour = None;
                    if let Some(s) = gp.stops.get(sel) {
                        let tip = p0 + axis * s.pos.clamp(0.0, 1.0) + nrm * 2.0;
                        let at = tip + nrm * 32.0;
                        let r = egui::Rect::from_center_size(at, egui::vec2(18.0, 13.0));
                        let mut c = s.color;
                        ui.scope_builder(egui::UiBuilder::new().max_rect(r), |ui| {
                            if colorpicker::button_sized(
                                ui,
                                (node, "grad-hud-stop", sel),
                                &mut c,
                                spec,
                                r.size(),
                                "this stop's colour · alt-click a stop removes it · \
                                 double-click the axis adds one",
                                None,
                            )
                            .changed()
                            {
                                recolour = Some(c);
                            }
                        });
                    }
                    if let Some(c) = recolour {
                        gp.stops[sel].color = c;
                    }
                    if gp != before {
                        let chans = anim::channels_for(&NodeKind::Gradient(before.clone()));
                        if let Some(n) = self.egraph.node_mut(node) {
                            let editor::ENode { kind, anim, .. } = n;
                            keyed_write_back(
                                anim,
                                kind,
                                &NodeKind::Gradient(before.clone()),
                                &NodeKind::Gradient(gp.clone()),
                                cur_t,
                                &chans,
                                self.autokey,
                                &patch_driven,
                                node,
                            );
                        }
                    }
                    // Structural edits go straight to the authored stops
                    // (and, for a removal, their curves).
                    if remove.is_some() || added.is_some() {
                        if let Some(editor::ENode { kind: NodeKind::Gradient(ap), anim, .. }) =
                            self.egraph.node_mut(node)
                        {
                            if let Some(i) = remove {
                                nodes::gradient::remove_stop(ap, anim, i);
                            }
                            if let Some(t) = added {
                                // The colour the ramp has there NOW (curves
                                // evaluated), so nothing jumps.
                                let color = gp.sample(t, working);
                                if ap.stops.len() < engine::graph::GRAD_STOPS {
                                    ap.stops.push(engine::graph::GradStop { pos: t, color });
                                    nodes::gradient::select(ui.ctx(), node, ap.stops.len() - 1);
                                }
                            }
                        }
                    }
                }
                // The mirror's line across the picture: drag the centre
                // to move it, the arrow (pointing at the half that is
                // KEPT) to turn it, shift snapping to 15°. Absolute
                // under the pointer, unwrapped against the press.
                NodeKind::Mirror(mut mp) => {
                    let before = mp;
                    let (w, h) = size;
                    let pt = |v: [f32; 2]| to_screen(egui::vec2(v[0], v[1]));
                    let c = [mp.center[0] * w, mp.center[1] * h];
                    let (d, n) = (mp.direction(), mp.normal());
                    let pc = pt(c);
                    let far = w.hypot(h);
                    let (e0, e1) = (
                        pt([c[0] - d[0] * far, c[1] - d[1] * far]),
                        pt([c[0] + d[0] * far, c[1] + d[1] * far]),
                    );
                    // Clip the line to the viewer so it reads as a line.
                    let (e0, e1) = (rect.clamp(e0), rect.clamp(e1));
                    ink.seg(e0, e1, 1.5, 0.9, false);
                    let ns = (pt([c[0] + n[0], c[1] + n[1]]) - pc).normalized();
                    let tip = pc + ns * 42.0;
                    let mods = ui.input(|i| i.modifiers);
                    let pointer = ui.ctx().pointer_latest_pos();
                    // The arrow: the side the mirror keeps.
                    let rs = ui.interact(
                        egui::Rect::from_center_size(tip, egui::vec2(20.0, 20.0)),
                        ui.id().with(("mirror-turn", node)),
                        egui::Sense::drag(),
                    );
                    let hot = rs.hovered() || rs.dragged();
                    ink.seg(pc, tip, 1.5, 0.9, hot);
                    let side = egui::vec2(-ns.y, ns.x);
                    ink.seg(tip, tip - ns * 8.0 + side * 5.0, 1.5, 0.9, hot);
                    ink.seg(tip, tip - ns * 8.0 - side * 5.0, 1.5, 0.9, hot);
                    if let (true, Some(p)) = (rs.dragged(), pointer) {
                        let o = from_screen(p);
                        let mut ang =
                            engine::graph::MirrorParams::angle_facing([o.x - c[0], o.y - c[1]]);
                        if mods.shift {
                            ang = (ang / 15.0).round() * 15.0;
                        }
                        let dd = (ang - before.angle).rem_euclid(360.0);
                        mp.angle = before.angle + if dd > 180.0 { dd - 360.0 } else { dd };
                    }
                    // The centre.
                    let rs = ui.interact(
                        egui::Rect::from_center_size(pc, egui::vec2(18.0, 18.0)),
                        ui.id().with(("mirror-center", node)),
                        egui::Sense::drag(),
                    );
                    ink.circle_stroke(pc, 6.0, 1.5, 0.9, rs.hovered() || rs.dragged());
                    if let (true, Some(p)) = (rs.dragged(), pointer) {
                        let o = from_screen(p);
                        mp.center = [o.x / w, o.y / h];
                    }
                    if mp != before {
                        if let Some(n) = self.egraph.node_mut(node) {
                            let editor::ENode { kind, anim, .. } = n;
                            keyed_write_back(
                                anim,
                                kind,
                                &NodeKind::Mirror(before),
                                &NodeKind::Mirror(mp),
                                cur_t,
                                nodes::mirror::CHANNELS,
                                self.autokey,
                                &patch_driven,
                                node,
                            );
                        }
                    }
                }
                // Lens Distortion: a grid straight in the undistorted
                // picture and the source frame's edge, both where they
                // land in this node's output; the principal point drags.
                NodeKind::LensDistort(mut lp) => {
                    let before = lp;
                    let src = self.egraph.input_of(node, 0).and_then(|s| self.node_res_f32(s)).unwrap_or(size);
                    let viewer = nodes::lensdistort::Viewer {
                        out: size,
                        src,
                        to_screen: &to_screen,
                        from_screen: &from_screen,
                    };
                    if nodes::lensdistort::gizmo(ui, ink, node, &mut lp, &viewer) && lp != before {
                        if let Some(n) = self.egraph.node_mut(node) {
                            let editor::ENode { kind, anim, .. } = n;
                            keyed_write_back(
                                anim,
                                kind,
                                &NodeKind::LensDistort(before),
                                &NodeKind::LensDistort(lp),
                                cur_t,
                                nodes::lensdistort::CHANNELS,
                                self.autokey,
                                &patch_driven,
                                node,
                            );
                        }
                    }
                }
                // The GridWarp lattice: both grids drawn through their
                // interpolant, handles on the one being edited (the
                // panel picks which), the pick dragged absolutely from
                // its press.
                NodeKind::GridWarp(mut gw) => {
                    let before = gw.clone();
                    let banding = nav.dragged_by(egui::PointerButton::Primary) && !cmd && !scrub_held;
                    let viewer = nodes::gridwarp::Viewer {
                        size,
                        to_screen: &to_screen,
                        from_screen: &from_screen,
                        rect,
                        banding,
                    };
                    if nodes::gridwarp::gizmo(ui, ink, node, &mut gw, &viewer, hud_key_list) {
                        let chans = anim::channels_for(&NodeKind::GridWarp(before.clone()));
                        if let Some(n) = self.egraph.node_mut(node) {
                            let editor::ENode { kind, anim, .. } = n;
                            keyed_write_back(
                                anim,
                                kind,
                                &NodeKind::GridWarp(before),
                                &NodeKind::GridWarp(gw),
                                cur_t,
                                &chans,
                                self.autokey,
                                &patch_driven,
                                node,
                            );
                        }
                    }
                }
                // Warp: both sides of every curve, the edited side's
                // points picked and moved like GridWarp's; S-click draws,
                // D-click removes (structure, straight to the authored
                // curves).
                NodeKind::Warp(mut wp) => {
                    let before = wp.clone();
                    let banding = nav.dragged_by(egui::PointerButton::Primary) && !cmd && !scrub_held;
                    let k = if self.proxy > 0.0 { self.proxy } else { 1.0 };
                    let viewer = nodes::warp::Viewer {
                        size,
                        to_screen: &to_screen,
                        from_screen: &from_screen,
                        rect,
                        banding,
                        falloff: (wp.falloff * k).max(1.0),
                    };
                    let (moved, edits) = nodes::warp::gizmo(ui, ink, node, &mut wp, &viewer, hud_key_list);
                    if let Some(n) = self.egraph.node_mut(node) {
                        let editor::ENode { kind, anim, .. } = n;
                        if moved {
                            let chans = anim::channels_for(&NodeKind::Warp(before.clone()));
                            keyed_write_back(
                                anim,
                                kind,
                                &NodeKind::Warp(before),
                                &NodeKind::Warp(wp),
                                cur_t,
                                &chans,
                                self.autokey,
                                &patch_driven,
                                node,
                            );
                        }
                        if let NodeKind::Warp(auth) = kind {
                            nodes::warp::apply_edits(auth, anim, edits);
                        }
                    }
                }
                NodeKind::SoftCrop(mut sc) => {
                    let before = sc;
                    let (w, h) = size;
                    let [x0, y0, x1, y1] = sc.px(w, h);
                    let q = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
                        .map(|(x, y)| to_screen(egui::vec2(x, y)));
                    for k in 0..4 {
                        ink.seg(q[k], q[(k + 1) % 4], 1.5, 0.9, false);
                    }
                    // (edge index into rect, or two for a corner):
                    // l=0 t=1 r=2 b=3.
                    let grips: [(egui::Pos2, &[usize]); 8] = [
                        (q[0], &[0, 1]),
                        (q[1], &[2, 1]),
                        (q[2], &[2, 3]),
                        (q[3], &[0, 3]),
                        (q[0] + (q[1] - q[0]) * 0.5, &[1]),
                        (q[1] + (q[2] - q[1]) * 0.5, &[2]),
                        (q[2] + (q[3] - q[2]) * 0.5, &[3]),
                        (q[3] + (q[0] - q[3]) * 0.5, &[0]),
                    ];
                    for (gi, (at, edges)) in grips.iter().enumerate() {
                        let rs = ui.interact(
                            egui::Rect::from_center_size(*at, egui::vec2(22.0, 22.0)),
                            ui.id().with(("softcrop", node, gi)),
                            egui::Sense::drag(),
                        );
                        ink.rect(
                            egui::Rect::from_center_size(*at, egui::vec2(10.0, 10.0)),
                            1.0,
                            true,
                            0.0,
                            1.0,
                            rs.hovered(),
                        );
                        if rs.dragged() {
                            if let Some(p) = ui.ctx().pointer_latest_pos() {
                                let o = from_screen(p);
                                for e in edges.iter() {
                                    sc.rect[*e] = if e % 2 == 0 {
                                        (o.x / w).clamp(0.0, 1.0)
                                    } else {
                                        (o.y / h).clamp(0.0, 1.0)
                                    };
                                }
                            }
                        }
                    }
                    if sc != before {
                        if let Some(n) = self.egraph.node_mut(node) {
                            let editor::ENode { kind, anim, .. } = n;
                            keyed_write_back(
                                anim,
                                kind,
                                &NodeKind::SoftCrop(before),
                                &NodeKind::SoftCrop(sc),
                                cur_t,
                                nodes::softcrop::CHANNELS,
                                self.autokey,
                                &patch_driven,
                                node,
                            );
                        }
                    }
                }
                _ => {}
            }
        }
    }
}
