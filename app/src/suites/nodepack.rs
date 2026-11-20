//! `--test-nodepack`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    /// --test-nodepack driver: build a chain out of the new nodes, view it,
    /// and prove three things the unit tests cannot — that the pack compiles
    /// into a real fused pass and renders (not black), that a Reformat's
    /// format actually drives the output resolution, and that a TimeOffset
    /// moves which source frame the reader decodes.
    pub(crate) fn drive_test_nodepack(&mut self) {
        let Some((stage, mut frames, chain)) = self.test.nodepack else { return };
        // `frames` counts frames IN THIS STAGE, reset on every advance: a
        // global counter would make every later `frames > N` already true
        // after a slow first load, and the whole state machine would run in
        // three frames — probing the previous topology's output.
        frames += 1;
        let limit = if std::env::var("WOBBLE_TEST_HOLD").is_ok() { u32::MAX } else { 1500 };
        if frames > limit {
            println!(
                "NODEPACK TEST FAIL: timeout at stage {stage} (pos {}, fd {}, ring free {:?}, state {:?})",
                self.timeline_pos,
                self.force_display,
                self.playback.as_ref().map(|pb| pb.ring.free_count()),
                self.playback.as_ref().map(|pb| pb.player.debug_state()),
            );
            std::process::exit(1);
        }
        let mut next = stage;
        match stage {
            // Wait for real frames, then hang the whole pack off Read A.
            0 if self.playback.is_some() && self.displayed_total >= 3 => {
                let at = egui::pos2(300.0, 0.0);
                let un = self.egraph.add(
                    NodeKind::Premult { premult: false },
                    at,
                );
                self.egraph.connect(un, 0, 0); // Read A
                self.egraph.view = Some(un);
                self.test.nodepack = Some((1, 0, un));
                return;
            }
            // The chain must render, and must not be black.
            1 if frames > 20 => {
                match self.probe_output_mean() {
                    Some(m) if m > 0.001 => {
                        self.test_dump("nodepack-chain");
                        next = 2
                    }
                    Some(m) => {
                        println!("NODEPACK TEST FAIL: chain renders black ({m:.4})");
                        std::process::exit(1);
                    }
                    None => {}
                }
            }
            // A Reformat's format drives the comp resolution.
            2 => {
                let rf = self.egraph.add(
                    NodeKind::Reformat(engine::graph::ReformatParams {
                        size: (640, 360),
                        ..Default::default()
                    }),
                    egui::pos2(300.0, 320.0),
                );
                self.egraph.connect(rf, 0, chain);
                self.egraph.view = Some(rf);
                self.test.nodepack = Some((3, 0, rf));
                return;
            }
            3 if frames > 60 => {
                match self.comp.output_size() {
                    Some((640, 360)) => {
                        self.test_dump("nodepack-reformat");
                        next = 30
                    }
                    Some(other) if frames > 240 => {
                        println!(
                            "NODEPACK TEST FAIL: reformat to 640x360 left the output at {other:?}"
                        );
                        std::process::exit(1);
                    }
                    _ => {}
                }
            }
            // A matte keeps its format through a Dilate and a Gamma of it:
            // a Dilate once took the CANVAS (the plate's 2K) because the
            // format walk skipped matte inputs, so growing a 640x360 matte
            // changed its resolution.
            30 => {
                let tm = self.egraph.add(
                    NodeKind::ToMatte(Default::default()),
                    egui::pos2(300.0, 440.0),
                );
                let dl = self.egraph.add(
                    NodeKind::Dilate(engine::graph::DilateParams { amount: 2.0 }),
                    egui::pos2(300.0, 560.0),
                );
                let gm = self.egraph.add(
                    NodeKind::Gamma(engine::graph::GammaParams { master: 2.0, ..Default::default() }),
                    egui::pos2(300.0, 680.0),
                );
                self.egraph.connect(tm, 0, chain); // the Reformat
                self.egraph.connect(dl, 0, tm);
                self.egraph.connect(gm, 0, dl); // a matte: the Gamma follows it
                if !self.egraph.node(gm).is_some_and(|n| n.kind.matte_mode()) {
                    println!("NODEPACK TEST FAIL: a Gamma wired a matte did not take the matte pipe");
                    std::process::exit(1);
                }
                self.egraph.view = Some(gm);
                self.test.nodepack = Some((31, 0, chain));
                return;
            }
            31 if frames > 60 => {
                let sized = self.comp.output_size() == Some((640, 360));
                match self.probe_output_mean() {
                    Some(m) if sized && m > 0.001 => {
                        self.test_dump("nodepack-dilate");
                        next = 4
                    }
                    Some(m) if frames > 240 => {
                        println!(
                            "NODEPACK TEST FAIL: Dilate -> Gamma of a 640x360 matte renders at {:?} (mean {m:.4})",
                            self.comp.output_size()
                        );
                        std::process::exit(1);
                    }
                    _ => {}
                }
            }
            // A Grid generates its own image at its own format, with no input.
            4 => {
                let grid = self.egraph.add(
                    NodeKind::Grid(engine::graph::GridParams {
                        size: (512, 512),
                        spacing: [32.0, 32.0],
                        width: 8.0,
                        ..Default::default()
                    }),
                    egui::pos2(560.0, 0.0),
                );
                self.egraph.view = Some(grid);
                self.test.nodepack = Some((5, 0, grid));
                return;
            }
            5 if frames > 60 => {
                let sized = self.comp.output_size() == Some((512, 512));
                match self.probe_output_mean() {
                    Some(m) if sized && m > 0.001 => {
                        self.test_dump("nodepack-grid");
                        next = 6
                    }
                    Some(m) if frames > 300 => {
                        println!(
                            "NODEPACK TEST FAIL: grid at {:?} probed {m:.4} (wanted 512x512, lines visible)",
                            self.comp.output_size()
                        );
                        std::process::exit(1);
                    }
                    _ => {}
                }
            }
            // A TimeOffset moves which SOURCE frame the readers above it
            // decode, while the timeline itself stays put — and the SAME
            // Read feeding a merge both straight and through the offset is
            // two readers holding two frames at once, not a conflict.
            6 => {
                let off = self.egraph.add(
                    NodeKind::Slip(engine::graph::SlipParams { delay: -3 }),
                    egui::pos2(-300.0, 0.0),
                );
                // DIFFERENCE, so the assertion is about pixels: if both
                // branches were reading the same frame the output would be
                // exactly black, whatever the bookkeeping said.
                let mg = self.egraph.add(
                    NodeKind::Blend { op: engine::graph::BlendOp::Difference, mix: 1.0, gain: [1.0, 1.0] },
                    egui::pos2(-300.0, 120.0),
                );
                self.egraph.connect(off, 0, 0); // Read A, retimed
                self.egraph.connect(mg, 0, off);
                self.egraph.connect(mg, 1, 0); // the same Read, un-retimed
                self.egraph.view = Some(mg);
                self.test.nodepack = Some((7, 0, off));
                return;
            }
            7 if frames > 60 => {
                let Some(pb) = &self.playback else { return };
                // Two instances of one source: the un-retimed slot sits on
                // the timeline frame, the retimed one three ahead.
                let frames_at = |tick: u64| -> Vec<u64> {
                    let mut v: Vec<u64> = (0..pb.slots.len())
                        .map(|r| pb.player.source_frame(r, tick))
                        .collect();
                    v.sort_unstable();
                    v
                };
                let at0 = frames_at(0);
                let len = pb.player.timeline_len();
                // The two branches must actually DIFFER on screen.
                let diff = self.probe_output_mean().unwrap_or(0.0);
                if at0.len() >= 2
                    && at0.contains(&0)
                    && at0.contains(&3)
                    && len > 3
                    && diff > 0.002
                {
                    self.test_dump("nodepack-two-timebases");
                    println!(
                        "NODEPACK TEST: two timebases ok ({at0:?}, diff {diff:.4}) — \
                         now the playhead-stability contract"
                    );
                    // Park the playhead mid-timeline, paused.
                    self.playing = false;
                    self.seek_to(7);
                    self.test.nodepack = Some((8, 0, chain));
                    return;
                }
                if frames > 400 {
                    println!(
                        "NODEPACK TEST FAIL: readers held {at0:?} at tick 0 and the \
                         branches differ by {diff:.4} (wanted 0 and 3 live at once, \
                         visibly different)"
                    );
                    std::process::exit(1);
                }
            }
            // Slip contract: with the playhead parked at 7, dragging the
            // slip re-decodes THAT tick under the new mapping and refreshes
            // the display — the transport must not move a single frame.
            8 if !self.force_display && self.timeline_pos == 7 => {
                let shown = self.displayed_total;
                if let Some(editor::ENode { kind: NodeKind::Slip(p), .. }) =
                    self.egraph.node_mut(chain)
                {
                    p.delay = -9;
                }
                // `chain` becomes the displayed-total mark from here on.
                self.test.nodepack = Some((9, 0, shown));
                return;
            }
            9 => {
                let Some(pb) = &self.playback else { return };
                // The slipped reader must land on 7+9 while its sibling
                // stays on 7; the playhead itself must still read 7; and the
                // image must have REFRESHED without any transport input.
                let frames_now: Vec<u64> = (0..pb.slots.len())
                    .map(|r| pb.player.source_frame(r, 7))
                    .collect();
                let mark = chain;
                let refreshed = self.displayed_total > mark && !self.force_display;
                if self.timeline_pos != 7 {
                    println!(
                        "NODEPACK TEST FAIL: slip edit moved the playhead to {} (must stay 7)",
                        self.timeline_pos
                    );
                    std::process::exit(1);
                }
                if refreshed && frames_now.contains(&16) && frames_now.contains(&7) {
                    println!(
                        "NODEPACK TEST PASS: fused pack + two timebases verified; slip 3->9 \
                         re-decoded tick 7 as sources {frames_now:?} and refreshed the \
                         display with the playhead pinned at 7"
                    );
                    std::process::exit(0);
                }
                if frames > 600 {
                    println!(
                        "NODEPACK TEST FAIL: after slip edit at tick 7: sources {frames_now:?} \
                         (want 7 and 16), refreshed={refreshed} (displayed {} vs mark {})",
                        self.displayed_total, mark
                    );
                    std::process::exit(1);
                }
            }
            _ => {}
        }
        let frames = if next == stage { frames } else { 0 };
        self.test.nodepack = Some((next, frames, chain));
    }
}
