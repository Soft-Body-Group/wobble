//! `--test-track-ui`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    pub(crate) fn drive_test_track_ui(&mut self, raw: &mut egui::RawInput) {
        use engine::planar::{h_apply, h_inv, h_mul, synth};
        let Some((stage, frames, tid)) = self.test.track else { return };
        if stage < 20 {
            return;
        }
        const PW: usize = 960;
        const PH: usize = 540;
        const PN: usize = 48;
        const PLACE: u64 = 1001;
        // Four texels on the plane, spread like a corner pin's.
        const PTS: [[f32; 2]; 4] = [[0.42, 0.47], [0.58, 0.44], [0.57, 0.62], [0.41, 0.6]];
        let fail = |msg: String| -> ! {
            println!("TRACK TEST FAIL: {msg}");
            std::process::exit(1);
        };
        let count = |s: &Self| match s.egraph.node(tid).map(|n| &n.kind) {
            Some(NodeKind::Tracker(t)) => t.count(),
            _ => 0,
        };
        let hits = |s: &Self| -> Vec<(innards::Hit, egui::Rect)> {
            s.node_editor.innard_hits.iter().find(|(i, ..)| *i == tid).map(|(_, _, h, _)| h.clone()).unwrap_or_default()
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
        let mut next = stage;
        match stage {
            // Zoomed out the Tracker is its glyph: no innards.
            20 => {
                if !hits(self).is_empty() {
                    fail("innards drawn at the default zoom".into());
                }
                let pos = self.egraph.node(tid).map(|n| n.pos).unwrap_or_default();
                let z = (innards::FULL_AT * 1.6) / editor::NODE_R;
                self.node_editor.zoom = z;
                self.node_editor.pan = -pos.to_vec2() * z;
                self.selected = vec![tid];
                next = 21;
            }
            // `+` three times: press on one frame, release on the next.
            21 if frames > 3 => {
                let h = hits(self);
                let Some(plus) = h.iter().find(|(k, _)| *k == innards::Hit::Add).map(|(_, r)| r.center()) else {
                    fail(format!("zoomed in, the Tracker drew no + (hits {h:?})"));
                };
                click(raw, plus, true);
                next = 22;
            }
            22 => {
                let h = hits(self);
                if let Some(plus) = h.iter().find(|(k, _)| *k == innards::Hit::Add).map(|(_, r)| r.center()) {
                    click(raw, plus, false);
                }
                next = 23;
            }
            23 if frames > 3 => {
                let n = count(self);
                if n < 4 {
                    next = 21;
                } else if n != 4 {
                    fail(format!("{n} points after three + clicks"));
                } else {
                    let h = hits(self);
                    let Some(third) = h.iter().find(|(k, _)| *k == innards::Hit::Node(2)).map(|(_, r)| r.center())
                    else {
                        fail("no innard for point 3".into());
                    };
                    click(raw, third, true);
                    next = 24;
                }
            }
            24 => {
                let h = hits(self);
                if let Some(third) = h.iter().find(|(k, _)| *k == innards::Hit::Node(2)).map(|(_, r)| r.center()) {
                    click(raw, third, false);
                }
                next = 25;
            }
            25 if frames > 3 => {
                let ctx = self.egui_ctx.clone();
                if innards::selected(&ctx, tid, 4) != 2 {
                    fail(format!("clicking point 3 selected {}", innards::selected(&ctx, tid, 4) + 1));
                }
                next = 30;
            }
            // The inner graph's gestures, as a hand makes them: press,
            // travel over a few frames, release. 30: drag point 2's node.
            30 | 32 | 34 => {
                let h = hits(self);
                let solve = tracker::SOLVE;
                if frames == 1 {
                    let at = |k: innards::Hit| h.iter().find(|(x, _)| *x == k).map(|(_, r)| r.center());
                    let (a, b) = match stage {
                        30 => {
                            let Some(a) = at(innards::Hit::Node(1)) else { fail("no node for point 2".into()) };
                            self.test.track_move_from = self.tracker_node_at(tid, 1);
                            (a, a + egui::vec2(40.0, 20.0))
                        }
                        // Point 4's wire dragged off Solve onto nothing.
                        32 => {
                            let Some(a) = at(innards::Hit::In(solve, 3)) else { fail("Solve has no port 4".into()) };
                            (a, a + egui::vec2(-80.0, 60.0))
                        }
                        // ...and plugged back in from point 4's output.
                        _ => {
                            let (Some(a), Some(b)) = (at(innards::Hit::Out(3)), at(innards::Hit::In(solve, 3))) else {
                                fail("no output on point 4 or no free port on Solve".into())
                            };
                            (a, b)
                        }
                    };
                    self.test.track_drag = (a, b);
                    click(raw, a, true);
                } else if frames <= 4 {
                    let (a, b) = self.test.track_drag;
                    raw.events.push(egui::Event::PointerMoved(a + (b - a) * (frames as f32 - 1.0) / 3.0));
                } else if frames == 5 {
                    click(raw, self.test.track_drag.1, false);
                } else if frames > 8 {
                    next = stage + 1;
                }
            }
            31 => {
                let (was, now) = (self.test.track_move_from, self.tracker_node_at(tid, 1));
                let frame_w = self
                    .node_editor
                    .innard_hits
                    .iter()
                    .find(|(i, ..)| *i == tid)
                    .map(|(_, f, ..)| f.width())
                    .unwrap_or(1.0);
                let want = [40.0 / (frame_w * 0.5), 20.0 / (frame_w * 0.5)];
                let got = [now[0] - was[0], now[1] - was[1]];
                if (got[0] - want[0]).abs() > 0.01 || (got[1] - want[1]).abs() > 0.01 {
                    fail(format!("dragging point 2's node moved it {got:?}, wanted {want:?}"));
                }
                if count(self) != 4 {
                    fail("a node drag changed the point count".into());
                }
                next = 32;
            }
            33 => {
                if self.tracker_solve_in(tid) != vec![0, 1, 2] {
                    fail(format!("dragging point 4's wire off left Solve fed by {:?}", self.tracker_solve_in(tid)));
                }
                next = 34;
            }
            35 => {
                if self.tracker_solve_in(tid) != vec![0, 1, 2, 3] {
                    fail(format!("re-plugging point 4 left Solve fed by {:?}", self.tracker_solve_in(tid)));
                }
                next = 36;
            }
            // WOBBLE_TEST_HOLD=1 parks here (the inner graph on screen) for
            // a look.
            36 if std::env::var_os("WOBBLE_TEST_HOLD").is_some() && frames < 400_000 => {
                self.test.track_clock = web_time::Instant::now();
            }
            36 => {
                // Place the points (the viewer drag is the click suite's
                // gesture) and track from the first frame.
                if let Some(editor::ENode { kind: NodeKind::Tracker(t), .. }) = self.egraph.node_mut(tid) {
                    for (i, c) in PTS.iter().enumerate() {
                        if let Some(mut p) = t.point(i) {
                            p.center = *c;
                            p.size = [20.0 / PW as f32, 20.0 / PH as f32];
                            p.search = 0.05;
                            p.track = *c;
                            t.set_point(i, p);
                        }
                    }
                }
                self.seek_to(PLACE);
                next = 26;
            }
            26 if self.eval_tick() == PLACE && !self.force_display => {
                let Some(NodeKind::Tracker(p)) = self.egraph.node(tid).map(|n| n.kind.clone()) else { return };
                self.test.track_progress = 0;
                self.start_track(tid, p);
                if self.track_job.is_none() {
                    fail("the four-point track did not start".into());
                }
                next = 27;
            }
            27 if self.track_job.is_some() => {
                if let Some((_, j)) = &self.track_job {
                    if j.done > self.test.track_progress {
                        self.test.track_progress = j.done;
                        self.test.track_clock = web_time::Instant::now();
                    }
                }
            }
            27 => {
                let truth = |f: u64| {
                    h_mul(&synth::path(PW, PH, (f - PLACE) as usize), &h_inv(&synth::path(PW, PH, 0)).unwrap())
                };
                // Every point, every frame, on the truth.
                let Some(n) = self.egraph.node(tid) else { return };
                let mut worst = 0.0f64;
                for (i, c) in PTS.iter().enumerate() {
                    let [chx, chy] = tracker::chans(i);
                    let (Some(cx), Some(cy)) = (n.anim.get(&chx), n.anim.get(&chy)) else {
                        fail(format!("point {} keyed nothing", i + 1));
                    };
                    if cx.keys.len() != PN {
                        fail(format!("point {} keyed {} frames, wanted {PN}", i + 1, cx.keys.len()));
                    }
                    for (a, b) in cx.keys.iter().zip(cy.keys.iter()) {
                        let (x, y) = h_apply(&truth(a.frame as u64), c[0] as f64 * PW as f64, c[1] as f64 * PH as f64);
                        let e = (a.value as f64 * PW as f64 - x).hypot(b.value as f64 * PH as f64 - y);
                        worst = worst.max(e);
                    }
                }
                if worst > 0.5 {
                    fail(format!("a point of four sits {worst:.3} px off the truth"));
                }
                self.test.track_points_worst = worst;
                // Cable the solve into a new CornerPin.
                self.selected = vec![tid];
                self.drive_pins(tid, planarui::DriveTarget::CornerPin, &tracker::PIN_CHANNELS, "tracked points");
                next = 28;
            }
            28 if frames > 2 => {
                let end = PLACE + PN as u64 - 1;
                let Some(&cp) = self.selected.first() else { fail("drive selected nothing".into()) };
                let g = self.egraph.to_engine_graph_at(Some(end));
                let Some(NodeKind::CornerPin(p)) = g.nodes.get(&cp) else {
                    fail("drive made no CornerPin".into());
                };
                // Judge the pin where the points are, not at the frame's
                // corners: those are an extrapolation several times further
                // out, where a 0.2 px track reads as pixels (the planar
                // suite's lesson). The warp the pin makes, applied to the
                // four points, against the truth there.
                let h = truth_at(end - PLACE, PW, PH);
                let px = |q: [f32; 2]| [q[0] as f64 * PW as f64, q[1] as f64 * PH as f64];
                let unit = engine::graph::UNIT_QUAD.map(px);
                let Some(warp) = engine::planar::h_from_quad(&unit, &p.to.map(px)) else {
                    fail("the driven pin is degenerate".into());
                };
                let mut worst = 0.0f64;
                for c in PTS {
                    let [x0, y0] = px(c);
                    let (x, y) = h_apply(&h, x0, y0);
                    let (gx, gy) = h_apply(&warp, x0, y0);
                    worst = worst.max((gx - x).hypot(gy - y));
                }
                if worst > 0.75 {
                    fail(format!("the driven CornerPin carries the points {worst:.3} px off the truth at {end}"));
                }
                // Stabilise on the same solve: the points, where the truth
                // puts them on a frame, go back through the CornerPin's
                // from -> to warp to where they sat at the seed frame.
                self.selected = vec![tid];
                self.drive_pins(tid, planarui::DriveTarget::Stabilize, &tracker::PIN_CHANNELS, "tracked points");
                let Some(&sp) = self.selected.first() else { fail("stabilise selected nothing".into()) };
                let mut stab = 0.0f64;
                for t in [PLACE + PN as u64 / 3, PLACE + 2 * PN as u64 / 3, end] {
                    let g = self.egraph.to_engine_graph_at(Some(t));
                    let Some(NodeKind::CornerPin(p)) = g.nodes.get(&sp) else {
                        fail("stabilise made no CornerPin".into());
                    };
                    let Some(shown) = engine::planar::h_from_quad(&unit, &p.effective_to().map(px)) else {
                        fail("the stabilise pin is degenerate".into());
                    };
                    let h = truth_at(t - PLACE, PW, PH);
                    for c in PTS {
                        let [x0, y0] = px(c);
                        let (x, y) = h_apply(&h, x0, y0);
                        let (gx, gy) = h_apply(&shown, x, y);
                        stab = stab.max((gx - x0).hypot(gy - y0));
                    }
                }
                if stab > 0.75 {
                    fail(format!("the stabilise CornerPin leaves the points {stab:.3} px off their seed positions"));
                }
                println!(
                    "TRACK TEST PASS: blob follows the disc and re-tracks from the playhead; pattern tracks \
                     {PN} frames placed at {PLACE} to {:.3} px; zoomed in, + added three points, a click \
                     picked one, a node drag moved one, point 4's wire unplugged and re-plugged; four points track to {:.3} px and the driven CornerPin carries them \
                     {worst:.3} px from the truth, and a stabilise CornerPin holds them within {stab:.3} px",
                    self.test.track_worst, self.test.track_points_worst
                );
                std::process::exit(0);
            }
            _ => {}
        }
        if next != stage {
            self.test.track_clock = web_time::Instant::now();
            self.test.track = Some((next, 0, tid));
        }
    }
}
