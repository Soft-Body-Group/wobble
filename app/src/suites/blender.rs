//! `--test-blender`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    /// --test-blender: the Blender node end to end against a stand-in
    /// Blender (`blenderui::fake`) on a real socket and a real ring.
    ///
    /// Connect; the spec goes out at the comp's format; a seek asks Blender
    /// for that frame (offset applied) and the frame that comes back is ON
    /// THE CANVAS (its colour encodes the frame); Blender's own scrub moves
    /// our playhead; a scrub back over a settled frame is served from the
    /// cache and Blender is told not to draw; an edit in Blender empties
    /// the cache, redraws, and the next request is a real draw again.
    /// Every wait is wall-clock.
    pub(crate) fn drive_test_blender(&mut self) {
        let Some(mut t) = self.test_blender.take() else { return };
        let fail = |m: String| -> ! {
            println!("BLENDER TEST FAIL: {m}");
            std::process::exit(1);
        };
        if t.since.elapsed().as_secs_f32() > 20.0 {
            fail(format!("timeout at stage {} ({:.0}s in)", t.stage, t.t0.elapsed().as_secs_f32()));
        }
        let stamp = self.comp.blender_stamp(t.node);
        let probe = |s: &Self| {
            let r = s.probe_output_box(8, 8, 8, 8, 0);
            let g = s.probe_output_box(8, 8, 8, 8, 1);
            r.zip(g)
        };
        let close = |a: f32, b: f32| (a - b).abs() < 0.006;
        let advance = |t: &mut TestBlender, stage: u32| {
            t.stage = stage;
            t.since = Instant::now();
        };
        match t.stage {
            0 if self.playback.is_some() && self.displayed_total >= 2 => {
                let (w, h) = self.comp.output_size().unwrap_or((64, 32));
                let params = engine::blender::BlenderParams {
                    space: engine::color::InputSpace::Working,
                    offset: 1000,
                    ..Default::default()
                };
                let node = self.egraph.add(NodeKind::Blender(params), egui::pos2(320.0, 40.0));
                self.egraph.view = Some(node);
                let (port, state) = blenderui::fake::start(self.blender_dir(), w, h);
                self.blender_action(node, blenderui::Action::Connect(port));
                t.node = node;
                t.fake = Some((w, h, state));
                advance(&mut t, 1);
            }
            1 => {
                let (w, h, st) = t.fake.as_ref().unwrap();
                let hello = self.blender.get(&t.node).is_some_and(|l| l.hello.is_some());
                let spec_ok = st.specs.lock().unwrap().iter().any(|s| s.contains(&format!("\"w\":{w},\"h\":{h}")));
                if hello && spec_ok {
                    self.seek_to(1007); // Blender frame 7 under the offset
                    advance(&mut t, 2);
                }
            }
            2 => {
                if stamp.is_some_and(|s| s.frame == 7 && s.settled) {
                    if let Some((r, g)) = probe(self) {
                        if close(r, 0.07) && close(g, 0.1) {
                            let st = &t.fake.as_ref().unwrap().2;
                            if st.last_frame_request() != Some((7, false)) {
                                fail(format!("frame 7 was requested as {:?}", st.last_frame_request()));
                            }
                            st.frame_changed(12);
                            advance(&mut t, 3);
                        } else if t.since.elapsed().as_secs_f32() > 5.0 {
                            fail(format!("frame 7 landed but the canvas reads r={r} g={g}"));
                        }
                    }
                }
            }
            3 => {
                let followed = self.timeline_pos == 1012;
                if followed && stamp.is_some_and(|s| s.frame == 12 && s.settled) {
                    if let Some((r, _)) = probe(self) {
                        if close(r, 0.12) {
                            self.seek_to(1007);
                            advance(&mut t, 4);
                        }
                    }
                }
            }
            4 => {
                if stamp.is_some_and(|s| s.frame == 7 && s.settled) {
                    let st = &t.fake.as_ref().unwrap().2;
                    match st.last_frame_request() {
                        Some((7, true)) => {}
                        other => fail(format!("a scrub back over a settled frame asked Blender to draw: {other:?}")),
                    }
                    if let Some((r, g)) = probe(self) {
                        if close(r, 0.07) && close(g, 0.1) {
                            st.edit();
                            advance(&mut t, 5);
                        }
                    }
                }
            }
            5 => {
                if stamp.is_some_and(|s| s.frame == 7 && s.scene == 2) {
                    if let Some((_, g)) = probe(self) {
                        if close(g, 0.2) {
                            self.seek_to(1012);
                            advance(&mut t, 6);
                        }
                    }
                }
            }
            6 => {
                if stamp.is_some_and(|s| s.frame == 12 && s.scene == 2) {
                    let st = &t.fake.as_ref().unwrap().2;
                    match st.last_frame_request() {
                        Some((12, false)) => {}
                        other => fail(format!("after an edit the cache still served frame 12: {other:?}")),
                    }
                    if !self.cost_flow.is_slow(t.node) {
                        fail("a live Blender node is not marked slow".into());
                    }
                    // 3D cards from the Blender scene: a camera and a card
                    // both reading the node, rendered.
                    let pic = self.egraph.add(
                        NodeKind::Grid(engine::graph::GridParams { inherit: true, ..Default::default() }),
                        egui::pos2(500.0, 40.0),
                    );
                    let blender = |object: &str| engine::scene3d::Source::Blender { node: t.node, object: object.into() };
                    let card = self.egraph.add(
                        NodeKind::Card3D(engine::graph::CardParams { source: blender("Plane"), ..Default::default() }),
                        egui::pos2(500.0, 140.0),
                    );
                    self.egraph.connect(card, 0, pic);
                    let cam = self.egraph.add(
                        NodeKind::Camera3D(engine::graph::CameraParams { source: blender(""), ..Default::default() }),
                        egui::pos2(620.0, 140.0),
                    );
                    let render = self.egraph.add(NodeKind::Render3D(Default::default()), egui::pos2(560.0, 240.0));
                    self.egraph.connect(render, 0, cam);
                    self.egraph.connect(render, 1, card);
                    self.egraph.view = Some(render);
                    t.render = render;
                    advance(&mut t, 7);
                }
            }
            7 | 8 => {
                // The cards read Blender's export: the camera's x is the
                // scene version, so an edit there moves it here.
                let want = if t.stage == 7 { 2.0 } else { 3.0 };
                let g = self.egraph.to_engine_graph_at_f(Some(self.timeline_pos as f32));
                let cam_x = match g.nodes.get(&t.render) {
                    Some(NodeKind::Render3D(p)) => {
                        p.lowered.as_ref().and_then(|l| l.cams.first()).map(|c| c.xform.translate[0])
                    }
                    _ => None,
                };
                let cards = match g.nodes.get(&t.render) {
                    Some(NodeKind::Render3D(p)) => p.lowered.as_ref().map_or(0, |l| l.cards.len()),
                    _ => 0,
                };
                if cam_x == Some(want) && cards == 1 {
                    let st = &t.fake.as_ref().unwrap().2;
                    if t.stage == 7 {
                        st.edit();
                        advance(&mut t, 8);
                    } else {
                        let asks = st.scene3d_asks.load(std::sync::atomic::Ordering::Relaxed);
                        if asks < 2 {
                            fail(format!("the edit re-exported the scene {asks} times"));
                        }
                        println!(
                            "BLENDER TEST PASS: connected to a bridge, the spec went out at the comp \
                             format, a seek put Blender's frame on the canvas under the offset, \
                             Blender's own scrub moved the playhead, a scrub back was served from the \
                             cache without a draw, an edit in Blender emptied it, and a 3D camera and \
                             card read Blender's scene, following an edit there"
                        );
                        std::process::exit(0);
                    }
                }
            }
            _ => {}
        }
        self.test_blender = Some(t);
    }
}
