//! `--test-roto`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    /// --test-track, the multi-point half (stages 20..): zoom the graph in on
    /// the Tracker until its innards show, press `+` three times through
    /// the real click path, click point 3 to select it, place the four
    /// points on the plane, track, drive a CornerPin from the solve and
    /// check its pin against the truth homography at the last frame.
    /// --test-roto: a roto container through the gestures a hand makes.
    /// Zoomed out the node is its glyph; zoomed in, its inner graph: `+`
    /// adds shape 2 and hands it the viewer, S-clicks draw it (shape 1
    /// untouched) and leave it OPEN, a plain press-drag adds a point with
    /// its tension pulled, a click on the first point closes it, and a node of the inner graph dragged onto another with
    /// O held makes a cut op of the two — which lowers to a Subtract
    /// MatteCombine for the engine.
    pub(crate) fn drive_test_roto(&mut self, raw: &mut egui::RawInput) {
        let Some((stage, mut frames, sid)) = self.test.roto else { return };
        frames += 1;
        // Wall clock, not frames: the S-click loop re-enters its stages, and
        // the unpaced test loop makes a frame count mean nothing anyway.
        let t0 = *self.test.roto_clock.get_or_insert_with(web_time::Instant::now);
        if t0.elapsed() > web_time::Duration::from_secs(30) {
            let n = match self.egraph.node(sid).map(|n| &n.kind) {
                Some(NodeKind::Spline(sp)) => format!(
                    "{} shapes, shape 1 has {} points, more {:?}, picked shape {}, pick {:?}, selected {:?} (roto {sid}), rect {:?}",
                    sp.shape_count(),
                    sp.points.len(),
                    sp.more.iter().map(|m| m.points.len()).collect::<Vec<_>>(),
                    innards::selected(&self.egui_ctx, sid, sp.shape_count()) + 1,
                    self.spline.sel,
                    self.selected,
                    self.test.rect
                ),
                _ => "no spline".into(),
            };
            println!("ROTO TEST FAIL: timeout at stage {stage} ({n})");
            std::process::exit(1);
        }
        let fail = |msg: String| -> ! {
            println!("ROTO TEST FAIL: {msg}");
            std::process::exit(1);
        };
        let sp = |s: &Self| -> Option<engine::graph::SplineParams> {
            match s.egraph.node(sid).map(|n| &n.kind) {
                Some(NodeKind::Spline(sp)) => Some(sp.clone()),
                _ => None,
            }
        };
        let hits = |s: &Self| -> Vec<(innards::Hit, egui::Rect)> {
            s.node_editor.innard_hits.iter().find(|(i, ..)| *i == sid).map(|(_, _, h, _)| h.clone()).unwrap_or_default()
        };
        let at = |s: &Self, k: innards::Hit| hits(s).iter().find(|(x, _)| *x == k).map(|(_, r)| r.center());
        let key = |raw: &mut egui::RawInput, key: egui::Key, down: bool| {
            raw.events.push(egui::Event::Key {
                key,
                physical_key: None,
                pressed: down,
                repeat: false,
                modifiers: Default::default(),
            });
        };
        let click = |raw: &mut egui::RawInput, p: egui::Pos2, down: bool| {
            raw.events.push(egui::Event::PointerMoved(p));
            raw.events.push(egui::Event::PointerButton {
                pos: p,
                button: egui::PointerButton::Primary,
                pressed: down,
                modifiers: Default::default(),
            });
        };
        // Viewer spots for shape 2's points, anchored at capture time.
        let spot = |s: &Self, k: usize| -> egui::Pos2 {
            let r = s.test.rect;
            let f = [(0.55, 0.3), (0.7, 0.35), (0.6, 0.6)][k];
            egui::pos2(r.min.x + r.width() * f.0, r.min.y + r.height() * f.1)
        };
        let mut next = stage;
        match stage {
            0 if frames >= 6 => {
                if !self.layout_settled() {
                    return;
                }
                self.test.rect = self.image_rect;
                // Shape 1: a square, so the node is a real roto.
                let sq = [[0.2, 0.2], [0.45, 0.2], [0.45, 0.45], [0.2, 0.45]]
                    .iter()
                    .map(|p| engine::graph::SplinePoint::new(*p, 0.0))
                    .collect();
                let id = self.egraph.add(
                    // Linear, so the drawn outline IS the square (a curve
                    // would bulge past its corners, and so would the crop).
                    NodeKind::Spline(engine::graph::SplineParams {
                        points: sq,
                        mode: engine::graph::SplineMode::Linear,
                        ..Default::default()
                    }),
                    egui::Pos2::new(600.0, -300.0),
                );
                let pos = self.egraph.node(id).map(|n| n.pos).unwrap_or_default();
                let z = (innards::FULL_AT * 1.6) / editor::NODE_R;
                self.node_editor.zoom = z;
                self.node_editor.pan = -pos.to_vec2() * z;
                self.selected = vec![id];
                self.playing = false;
                self.test.roto = Some((1, 0, id));
                return;
            }
            // `+`: press on one frame, release on the next.
            1 if frames > 3 => {
                let Some(plus) = at(self, innards::Hit::Add) else {
                    fail(format!("zoomed in, the roto drew no + (hits {:?})", hits(self)));
                };
                click(raw, plus, true);
                next = 2;
            }
            2 => {
                if let Some(plus) = at(self, innards::Hit::Add) {
                    click(raw, plus, false);
                }
                next = 3;
            }
            3 if frames > 3 => {
                let n = sp(self).map(|s| s.shape_count()).unwrap_or(0);
                let ctx = self.egui_ctx.clone();
                if n != 2 || innards::selected(&ctx, sid, n) != 1 {
                    fail(format!(
                        "after +: {n} shapes, shape {} picked (wanted 2, shape 2)",
                        innards::selected(&ctx, sid, n) + 1
                    ));
                }
                next = 4;
            }
            // Three S-clicks in the viewer draw shape 2.
            4 => {
                let k = sp(self).map(|s| s.more[0].points.len()).unwrap_or(0).min(2);
                let p = spot(self, k);
                self.test.roto_drag = (p, p);
                key(raw, egui::Key::S, true);
                click(raw, p, true);
                next = 5;
            }
            5 => {
                // Release where it pressed: anywhere else is a drag.
                click(raw, self.test.roto_drag.0, false);
                next = 6;
            }
            6 => {
                key(raw, egui::Key::S, false);
                next = 7;
            }
            7 if frames > 2 => {
                let s = sp(self).unwrap_or_default();
                let k = s.more.first().map(|m| m.points.len()).unwrap_or(0);
                if k < 3 {
                    next = 4;
                } else if k != 3 || s.points.len() != 4 {
                    fail(format!(
                        "S-clicks gave shape 2 {k} points and left shape 1 with {} (wanted 3 and 4)",
                        s.points.len()
                    ));
                } else if !s.more[0].open {
                    fail("three points closed shape 2 by themselves (a new shape stays open)".into());
                } else {
                    next = 20;
                }
            }
            // Pen tool: a PLAIN press while the shape is open adds a point
            // at its end, and the drag that follows (Catmull-Rom) dials
            // its tension — 40 px of pull is half way to a corner.
            20 => {
                let r = self.test.rect;
                let p = egui::pos2(r.min.x + r.width() * 0.75, r.min.y + r.height() * 0.55);
                if frames == 1 {
                    self.test.roto_drag = (p, p + egui::vec2(40.0, 0.0));
                    click(raw, p, true);
                } else if frames <= 5 {
                    let (a, b) = self.test.roto_drag;
                    raw.events.push(egui::Event::PointerMoved(a + (b - a) * (frames as f32 - 1.0) / 4.0));
                } else if frames == 6 {
                    click(raw, self.test.roto_drag.1, false);
                } else if frames > 9 {
                    let s = sp(self).unwrap_or_default();
                    let m = s.more.first().cloned().unwrap_or_default();
                    let t = m.points.last().map(|q| q.tension).unwrap_or(0.0);
                    if m.points.len() != 4 || !m.open || !(0.3..0.7).contains(&t) {
                        fail(format!(
                            "a pen drag left shape 2 with {} points, open {}, last tension {t} (wanted 4, open, ~0.5)",
                            m.points.len(),
                            m.open
                        ));
                    }
                    next = 21;
                }
            }
            // A click on the first point closes it.
            21 => {
                if frames == 1 {
                    let p = spot(self, 0);
                    self.test.roto_drag = (p, p);
                    click(raw, p, true);
                } else if frames == 2 {
                    click(raw, self.test.roto_drag.0, false);
                } else if frames > 5 {
                    let s = sp(self).unwrap_or_default();
                    let m = s.more.first().cloned().unwrap_or_default();
                    if m.open || m.points.len() != 4 {
                        fail(format!(
                            "a click on the first point left shape 2 open {} with {} points (wanted closed, 4)",
                            m.open,
                            m.points.len()
                        ));
                    }
                    next = 8;
                }
            }
            // Shape 2's node dragged onto shape 1's with O held.
            8 => {
                if frames == 1 {
                    let (Some(a), Some(b)) = (at(self, innards::Hit::Node(1)), at(self, innards::Hit::Node(0)))
                    else {
                        fail("no inner nodes for the shapes".into());
                    };
                    self.test.roto_drag = (a, b);
                    key(raw, egui::Key::O, true);
                    click(raw, a, true);
                } else if frames <= 5 {
                    let (a, b) = self.test.roto_drag;
                    raw.events.push(egui::Event::PointerMoved(a + (b - a) * (frames as f32 - 1.0) / 4.0));
                } else if frames == 6 {
                    click(raw, self.test.roto_drag.1, false);
                } else if frames == 7 {
                    key(raw, egui::Key::O, false);
                } else if frames > 10 {
                    next = 9;
                }
            }
            9 => {
                let s = sp(self).unwrap_or_default();
                let ok = s.ops.len() == 1
                    && s.ops[0].op == engine::graph::MatteOp::Subtract
                    && s.ops[0].a == Some(0)
                    && s.ops[0].b == Some(1);
                if !ok {
                    fail(format!("the O swipe left ops {:?}", s.ops));
                }
                // The node the swipe was dragging is back where it began.
                if (s.shape_at(1)[1] - s.shape_at(0)[1]).abs() < 0.05
                    && (s.shape_at(1)[0] - s.shape_at(0)[0]).abs() < 0.05
                {
                    fail("the swiped shape was left on top of the other".into());
                }
                let g = self.egraph.to_engine_graph_at(Some(self.eval_tick()));
                let cuts = g
                    .nodes
                    .values()
                    .filter(|k| {
                        matches!(k, NodeKind::MatteCombine(m) if m.op == engine::graph::MatteOp::Subtract)
                    })
                    .count();
                let splines = g.nodes.values().filter(|k| matches!(k, NodeKind::Spline(_))).count();
                if cuts != 1 || splines < 2 {
                    fail(format!("lowered to {cuts} cuts and {splines} splines"));
                }
                // The soft crop boxes what the roto DRAWS: shape 2 is cut
                // out of shape 1, so that is shape 1's square — not the
                // union of both.
                self.spline_soft_crop(sid, false);
                let rect = self.egraph.nodes.values().find_map(|n| match &n.kind {
                    NodeKind::SoftCrop(c) => Some(c.rect),
                    _ => None,
                });
                let want = [0.2, 0.2, 0.45, 0.45];
                if !rect.is_some_and(|r| r.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-3)) {
                    fail(format!("the soft crop boxed {rect:?}, wanted shape 1's {want:?}"));
                }
                next = 10;
            }
            // A ctrl-drag across the op's wire into Out cuts it: the op and
            // both shapes stay, wired to nothing that renders.
            10 => {
                let op = engine::graph::ROTO_OP;
                let ctrl = egui::Modifiers::CTRL;
                if frames == 1 {
                    let (Some(a), Some(b)) =
                        (at(self, innards::Hit::Out(op)), at(self, innards::Hit::In(rotoshapes::OUT, 0)))
                    else {
                        fail(format!("no op→out wire to cut (hits {:?})", hits(self)));
                    };
                    let mid = a + (b - a) * 0.5;
                    self.test.roto_drag = (mid - egui::vec2(60.0, 0.0), mid + egui::vec2(60.0, 0.0));
                    let p = self.test.roto_drag.0;
                    raw.events.insert(0, egui::Event::ModifiersChanged(ctrl));
                    raw.events.push(egui::Event::PointerMoved(p));
                    raw.events.push(egui::Event::PointerButton {
                        pos: p,
                        button: egui::PointerButton::Primary,
                        pressed: true,
                        modifiers: ctrl,
                    });
                } else if frames <= 5 {
                    let (a, b) = self.test.roto_drag;
                    raw.events.insert(0, egui::Event::ModifiersChanged(ctrl));
                    raw.events.push(egui::Event::PointerMoved(a + (b - a) * (frames as f32 - 1.0) / 4.0));
                } else if frames == 6 {
                    let p = self.test.roto_drag.1;
                    raw.events.insert(0, egui::Event::ModifiersChanged(ctrl));
                    raw.events.push(egui::Event::PointerButton {
                        pos: p,
                        button: egui::PointerButton::Primary,
                        pressed: false,
                        modifiers: ctrl,
                    });
                } else if frames > 9 {
                    let s = sp(self).unwrap_or_default();
                    if !s.out_inputs().is_empty() || s.ops.len() != 1 || s.shape_count() != 2 {
                        fail(format!(
                            "the ctrl-slice left out {:?}, {} ops, {} shapes",
                            s.out_inputs(),
                            s.ops.len(),
                            s.shape_count()
                        ));
                    }
                    next = 11;
                }
            }
            // Click the op, Delete over the node: the op goes, the node stays.
            11 => {
                let op = engine::graph::ROTO_OP;
                if frames == 1 {
                    let Some(a) = at(self, innards::Hit::Node(op)) else { fail("no op node to pick".into()) };
                    self.test.roto_drag = (a, a);
                    click(raw, a, true);
                } else if frames == 2 {
                    click(raw, self.test.roto_drag.0, false);
                } else if frames == 5 {
                    key(raw, egui::Key::Delete, true);
                } else if frames == 6 {
                    key(raw, egui::Key::Delete, false);
                } else if frames > 9 {
                    let s = sp(self).unwrap_or_default();
                    if !s.ops.is_empty() || s.shape_count() != 2 {
                        fail(format!("Delete over the node left {} ops, {} shapes", s.ops.len(), s.shape_count()));
                    }
                    if self.egraph.node(sid).is_none() {
                        fail("Delete over the inner graph deleted the roto itself".into());
                    }
                    next = 13;
                }
            }
            // Enter: the roto opens as a graph — each shape a Spline, an
            // Out — and an Axis wired between them and Out moves them all.
            13 => {
                if frames == 1 {
                    if let Err(e) = self.dive_into(sid) {
                        fail(format!("the roto would not open: {e}"));
                    }
                    let shapes: Vec<u64> = self
                        .egraph
                        .nodes
                        .iter()
                        .filter(|(_, n)| matches!(n.kind, NodeKind::Spline(_)))
                        .map(|(id, _)| *id)
                        .collect();
                    let out = self
                        .egraph
                        .nodes
                        .iter()
                        .find(|(_, n)| matches!(n.kind, NodeKind::RotoNode(engine::graph::RotoInner::Out { .. })))
                        .map(|(id, _)| *id);
                    let Some(out) = out.filter(|_| shapes.len() == 2) else {
                        fail(format!("inside the roto: {} shapes and no Out", shapes.len()))
                    };
                    if self.selected.len() != 1 || !shapes.contains(&self.selected[0]) {
                        fail(format!("the picked shape did not come in selected: {:?}", self.selected));
                    }
                    let mut axis = crate::rotodive::new_axis();
                    if let NodeKind::RotoNode(engine::graph::RotoInner::Axis(a)) = &mut axis {
                        a.xform.translate = [0.1, 0.0, 0.0];
                    }
                    let ax = self.egraph.add(axis, egui::pos2(0.0, 100.0));
                    self.egraph.edges.retain(|(d, _, _)| *d != out);
                    self.egraph.edges.push((ax, 0, shapes[0]));
                    self.egraph.edges.push((ax, 1, shapes[1]));
                    self.egraph.edges.push((out, 0, ax));
                } else if frames > 6 {
                    // A few frames drawn dived (preview + gizmo), then out.
                    if let Err(e) = self.dive_out() {
                        fail(format!("could not come back out: {e}"));
                    }
                    let s = sp(self).unwrap_or_default();
                    let ax = engine::graph::ROTO_AXIS;
                    if s.axes.len() != 1 || s.out != Some(vec![ax]) || s.axes[0].inputs.iter().flatten().count() != 2 {
                        fail(format!("back out: axes {:?}, out {:?}", s.axes, s.out));
                    }
                    // Lowered, both shapes ride the axis; with a Camera on
                    // port 1 they go through it, and the wire is gone.
                    let cam = self.egraph.add(NodeKind::Camera3D(Default::default()), egui::pos2(-200.0, 0.0));
                    self.egraph.edges.push((sid, engine::roto::CAMERA_INPUT, cam));
                    let g = self.egraph.to_engine_graph();
                    let projs: Vec<_> = g
                        .nodes
                        .values()
                        .filter_map(|k| match k {
                            NodeKind::Spline(one) => Some(one.proj),
                            _ => None,
                        })
                        .collect();
                    if projs.len() != 2 || !projs.iter().all(|p| p.is_some_and(|p| p.cam.is_some() && p.world[0][3] == 0.1)) {
                        fail(format!("lowered roto shapes did not ride the axis through the camera: {projs:?}"));
                    }
                    if g.input_of(sid, engine::roto::CAMERA_INPUT) == Some(cam) {
                        fail("the camera wire reached the engine".into());
                    }
                    next = 12;
                    self.test.roto_dived = true;
                }
            }
            12 if !self.test.roto_dived => next = 13,
            12 => {
                println!(
                    "ROTO TEST PASS: zoomed in, + added shape 2 and picked it; S-clicks drew it \
                     with shape 1 untouched, left open; a plain pen press-drag added a point and \
                     set its tension; a click on the first point closed it; an O swipe inside the node cut shape 2 out of shape 1 \
                     and the node lowers to one Subtract over the two splines; the soft crop boxed \
                     what the roto draws (shape 1, the cut taken out of it); a ctrl-drag cut the \
                     op's wire into Out (all three left unconnected); Delete over the node took \
                     the picked op and left the roto; Enter opened it as Splines + Out with the \
                     picked shape selected, an Axis wired in moved both shapes, and back out a \
                     Camera on its port projected them"
                );
                std::process::exit(0);
            }
            _ => {}
        }
        if next != stage {
            frames = 0;
        }
        self.test.roto = Some((next, frames, sid));
    }
}
