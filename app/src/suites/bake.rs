//! `--test-bake`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    /// Baked and live must be the SAME PICTURE at the same frame.
    ///
    /// A GraphBake renders its subtree to disk and then plays those files
    /// back through a reader instead of evaluating the comp. That swap is
    /// only honest if frame N of the bake is frame N of the comp — so this
    /// probes the live subtree over a window of frames, bakes it, scrubs
    /// back over the same frames and demands the same pixels. When it does
    /// not match it says WHICH live frame the bake is showing, because "off
    /// by one" and "stale picture" look identical in a single value.
    pub(crate) fn drive_test_bake(&mut self) {
        let Some((stage, mut frames)) = self.test_bake else { return };
        frames += 1;
        // Real decodes and a real bake: time out on a CLOCK, never on a
        // redraw budget (CLAUDE.md, "Rules for suites").
        let elapsed = self.test_bake_t0.get_or_insert_with(Instant::now).elapsed();
        if elapsed > web_time::Duration::from_secs(180) {
            println!(
                "BAKE TEST FAIL: timeout at stage {stage} (frame {}, chasing {}, baking {}, \
                 live {:?}, status {:?}, canvas {:?})",
                self.timeline_pos,
                self.force_display,
                self.graph_bake.is_some(),
                self.test_bake_live,
                self.render_status,
                self.comp.output_size(),
            );
            std::process::exit(1);
        }
        // Frames to compare, and where to look: a box well inside the plate,
        // on the green channel of the graded branch.
        // Offsets into the shot, so the placed pass probes 1011.. — the
        // frames that comp actually has.
        // The test plate's content is CYCLIC over its 24 frames, so the
        // probed frames stay inside half a cycle where they read apart.
        const OFFSETS: [u64; 4] = [0, 3, 6, 9];
        let base = self.shot_span().0;
        let probes: Vec<u64> = OFFSETS.iter().map(|o| base + o).collect();
        let probe = |s: &Self| s.probe_output_box(200, 200, 32, 32, 1);
        let settled = |s: &Self| !s.force_display && s.eval_tick() == s.timeline_pos;
        // A believable reading: the playhead has landed AND two successive
        // probes agree. Black means "nothing uploaded yet" here — every real
        // value in this plate is far from zero.
        let stable = |s: &mut Self| -> Option<f32> {
            if !settled(s) {
                s.test_bake_last = None;
                return None;
            }
            let v = probe(s)?;
            let same = s.test_bake_last.is_some_and(|p| (p - v).abs() < 1e-4);
            s.test_bake_last = Some(v);
            (same && v > 1e-4).then_some(v)
        };
        let bake_node = |s: &Self| {
            s.egraph
                .nodes
                .iter()
                .find(|(_, n)| matches!(n.kind, NodeKind::GraphBake { .. }))
                .map(|(id, _)| *id)
        };
        let mut next = stage;
        match stage {
            // LIVE pass: view the Grade (node 3 of the default graph).
            0 if self.playback.is_some() && self.displayed_total >= 3 => {
                self.playing = false;
                self.egraph.view = Some(3);
                self.seek_to(probes[0]);
                frames = 0;
                next = 1;
            }
            1 => {
                let Some(v) = stable(self) else {
                    self.test_bake = Some((next, frames));
                    return;
                };
                let i = self.test_bake_live.len();
                self.test_bake_live.push((probes[i], v));
                match probes.get(i + 1) {
                    Some(f) => {
                        self.seek_to(*f);
                        self.test_bake_last = None;
                        frames = 0;
                    }
                    None => next = 2,
                }
            }
            2 => {
                // Placing a plate renames its frames; it must not change
                // what any of them SHOW.
                if !self.test_bake_ref.is_empty() {
                    let off = |t: &Vec<(u64, f32)>| -> Vec<(u64, f32)> {
                        let b = t.first().map(|(f, _)| *f).unwrap_or(0);
                        t.iter().map(|(f, v)| (f - b, *v)).collect()
                    };
                    let (a, b) = (off(&self.test_bake_ref), off(&self.test_bake_live));
                    if let Some(((o, va), (_, vb))) =
                        a.iter().zip(&b).find(|((_, va), (_, vb))| (va - vb).abs() > 1e-2)
                    {
                        println!(
                            "BAKE TEST FAIL: placing the plate changed the PICTURE — \
                             offset {o} reads {vb:.4} placed, {va:.4} unplaced \
                             (unplaced {:?}, placed {:?})",
                            self.test_bake_ref, self.test_bake_live
                        );
                        std::process::exit(1);
                    }
                }
                // Every probed frame must be distinguishable, or a bake
                // showing the wrong one would pass.
                for (i, (fa, va)) in self.test_bake_live.iter().enumerate() {
                    for (fb, vb) in self.test_bake_live.iter().skip(i + 1) {
                        if (va - vb).abs() < 5e-3 {
                            println!(
                                "BAKE TEST FAIL: frames {fa} and {fb} look the same live \
                                 ({va:.4} / {vb:.4}) — pass {}, table {:?}",
                                self.test_bake_pass,
                                self.test_bake_live
                            );
                            std::process::exit(1);
                        }
                    }
                }
                // A reader-bearing node takes a real slot, exactly as the
                // add menu hands one out.
                let slot = self.next_read_slot;
                self.next_read_slot += 1;
                let bk = self.egraph.add(
                    NodeKind::GraphBake { source: slot, baked: false, start: None, mask: false, window: None, matte_half: false, single: false },
                    egui::pos2(-140.0, 60.0),
                );
                self.egraph.connect(bk, 0, 3);
                self.egraph.view = Some(bk);
                self.playback_dirty = true;
                frames = 0;
                next = 3;
            }
            3 if frames > 5 => {
                let Some(bk) = bake_node(self) else {
                    println!("BAKE TEST FAIL: the Bake node vanished");
                    std::process::exit(1);
                };
                self.start_graph_bake(bk);
                if self.graph_bake.is_none() && !self.loaded.contains_key(&bk) {
                    println!(
                        "BAKE TEST FAIL: the bake did not start ({:?})",
                        self.load_errors.get(&bk)
                    );
                    std::process::exit(1);
                }
                frames = 0;
                next = 4;
            }
            // Baking: wait for it to land and be adopted.
            4 if self.graph_bake.is_none() => {
                if !bake_node(self).is_some_and(|bk| self.loaded.contains_key(&bk)) {
                    println!(
                        "BAKE TEST FAIL: the bake finished but was never adopted ({:?})",
                        bake_node(self).and_then(|bk| self.load_errors.get(&bk))
                    );
                    std::process::exit(1);
                }
                self.seek_to(probes[0]);
                self.test_bake_last = None;
                frames = 0;
                next = 5;
            }
            // BAKED pass: scrub the same frames on the baked node.
            5 => {
                let Some(v) = stable(self) else {
                    self.test_bake = Some((next, frames));
                    return;
                };
                let i = self.test_bake_seen.len();
                let (want_f, _) = self.test_bake_live[i];
                self.test_bake_seen.push((want_f, v));
                if let Some(f) = probes.get(i + 1) {
                    self.seek_to(*f);
                    self.test_bake_last = None;
                    frames = 0;
                    self.test_bake = Some((next, frames));
                    return;
                }
                // Every frame probed: compare the two passes as a table, so
                // a frame shift and a value shift cannot look alike.
                let bad: Vec<String> = self
                    .test_bake_live
                    .iter()
                    .zip(&self.test_bake_seen)
                    .filter(|((_, a), (_, b))| (a - b).abs() > 1e-2)
                    .map(|((f, a), (_, b))| format!("{f}: live {a:.4} baked {b:.4}"))
                    .collect();
                if !bad.is_empty() {
                    println!(
                        "BAKE TEST FAIL: the bake is not the live picture — {}",
                        bad.join("; ")
                    );
                    std::process::exit(1);
                }
                // Frame for frame it is right when you let it settle. Now
                // SCRUB it: re-target every redraw, the way a drag does,
                // and demand that the picture ends up on the frame the hand
                // stopped on rather than wherever the decoder got to.
                self.seek_to(probes[0]);
                self.test_bake_last = None;
                frames = 0;
                next = 6;
            }
            // A DRAG: one new target per redraw, no waiting, ending on the
            // last probed frame.
            6 => {
                let step = frames as u64;
                let last = *probes.last().unwrap();
                if step <= last {
                    self.seek_to(step);
                    self.test_bake_last = None;
                    self.test_bake = Some((next, frames));
                    return;
                }
                frames = 0;
                next = 7;
            }
            7 => {
                let Some(v) = stable(self) else {
                    self.test_bake = Some((next, frames));
                    return;
                };
                let (f, want) = *self.test_bake_live.last().unwrap();
                if (v - want).abs() > 1e-2 {
                    let shows = self
                        .test_bake_live
                        .iter()
                        .find(|(_, lv)| (lv - v).abs() < 1e-2)
                        .map(|(lf, _)| format!("frame {lf}"))
                        .unwrap_or_else(|| "no probed frame".into());
                    println!(
                        "BAKE TEST FAIL: a scrub ending on {f} left the bake showing \
                         {shows} (read {v:.4}, wanted {want:.4})"
                    );
                    std::process::exit(1);
                }
                if self.test_bake_pass == 0 {
                    // Round two: PLACE the plate at 1011 and do it all
                    // again. A bake of a placed shot must be baked, named
                    // and played back at the comp's own frames.
                    self.place_read(0, 1011);
                    if let Some(bk) = bake_node(self) {
                        self.egraph.remove(bk);
                        self.loaded.remove(&bk);
                    }
                    self.egraph.view = Some(3);
                    self.playback_dirty = true;
                    self.test_bake_pass = 1;
                    self.test_bake_ref = self.test_bake_live.clone();
                    self.test_bake_live.clear();
                    self.test_bake_seen.clear();
                    self.test_bake_last = None;
                    self.test_bake = Some((0, 0));
                    return;
                }
                // Round three: PROXY. Switch it on exactly as the toolbar
                // chip does; the full bake must keep playing. A proxy bake
                // a previous run left under the same key would be played
                // instead (proxy prefers its own), so it goes first.
                if let Some(k) = bake_node(self).and_then(|bk| self.subtree_bake_key(bk)) {
                    let _ = std::fs::remove_dir_all(self.graph_bake_proxy_dir(k));
                }
                self.proxy = 0.5;
                self.playback_dirty = true;
                self.last_engine_graph = None;
                self.test_bake_last = None;
                frames = 0;
                next = 8;
            }
            // The canvas has reached half: the full bake still plays, from
            // the FULL directory, and counts as current.
            8 if frames > 5 => {
                let half = self.key_canvas().map(|(w, h)| (w.div_ceil(2), h.div_ceil(2)));
                if self.comp.output_size() != half || self.desired_output() != half {
                    self.test_bake = Some((next, frames));
                    return;
                }
                let Some(bk) = bake_node(self) else {
                    println!("BAKE TEST FAIL: the Bake node vanished under proxy");
                    std::process::exit(1);
                };
                let full = self.subtree_bake_key(bk).map(|k| self.graph_bake_dir(k));
                let playing = self.loaded.get(&bk).map(|l| std::path::PathBuf::from(&l.path));
                if playing.is_none() || playing != full || !self.graph_bake_current(bk) {
                    println!(
                        "BAKE TEST FAIL: under proxy the full bake should still play — \
                         playing {playing:?}, full {full:?}, current {}",
                        self.graph_bake_current(bk)
                    );
                    std::process::exit(1);
                }
                // Now bake UNDER proxy with the full bake still on disk:
                // the proxy bake lands beside it and proxy plays it.
                self.start_graph_bake(bk);
                if self.graph_bake.is_none() {
                    println!(
                        "BAKE TEST FAIL: the proxy bake did not start ({:?})",
                        self.load_errors.get(&bk)
                    );
                    std::process::exit(1);
                }
                frames = 0;
                next = 9;
            }
            // The proxy bake landed: half res, its own directory, and the
            // canvas still half (not a quarter: no second halving).
            9 if self.graph_bake.is_none() => {
                let Some(bk) = bake_node(self) else { std::process::exit(1) };
                let half = self.key_canvas().map(|(w, h)| (w.div_ceil(2), h.div_ceil(2)));
                let l = self.loaded.get(&bk);
                let path = l.map(|l| l.path.clone()).unwrap_or_default();
                let dims = l.map(|l| (l.seq.info.width, l.seq.info.height));
                if !is_proxy_bake(&path) || dims != half {
                    println!(
                        "BAKE TEST FAIL: the proxy bake landed wrong — path {path:?}, \
                         frames {dims:?}, wanted {half:?} ({:?})",
                        self.load_errors.get(&bk)
                    );
                    std::process::exit(1);
                }
                frames = 0;
                next = 10;
            }
            10 if frames > 5 => {
                let half = self.key_canvas().map(|(w, h)| (w.div_ceil(2), h.div_ceil(2)));
                // (A quarter canvas would wait here until the wall-clock
                // timeout, which prints the stage.)
                if self.comp.output_size() != half {
                    self.test_bake = Some((next, frames));
                    return;
                }
                let (f, _) = *self.test_bake_live.last().unwrap();
                self.seek_to(f);
                self.test_bake_last = None;
                frames = 0;
                next = 11;
            }
            // Its picture is the live one: the same region of the plate,
            // at half the pixels (a box filter of the full-res probe).
            11 => {
                let v = if settled(self) {
                    self.probe_output_box(100, 100, 16, 16, 1)
                } else {
                    None
                };
                let Some(v) = v else {
                    self.test_bake_last = None;
                    self.test_bake = Some((next, frames));
                    return;
                };
                let same = self.test_bake_last.is_some_and(|p| (p - v).abs() < 1e-4);
                self.test_bake_last = Some(v);
                if !(same && v > 1e-4) {
                    self.test_bake = Some((next, frames));
                    return;
                }
                let (f, want) = *self.test_bake_live.last().unwrap();
                if (v - want).abs() > 2e-2 {
                    println!(
                        "BAKE TEST FAIL: the proxy bake at {f} reads {v:.4}, live read {want:.4}"
                    );
                    std::process::exit(1);
                }
                // Both bakes kept: the full one is still on disk.
                let full_on_disk = bake_node(self)
                    .and_then(|bk| self.subtree_bake_key(bk))
                    .is_some_and(|k| engine::bakes::is_baked(&self.graph_bake_dir(k)));
                if !full_on_disk {
                    println!("BAKE TEST FAIL: the proxy bake removed the full bake");
                    std::process::exit(1);
                }
                // Back to full res: the full bake plays again, never the
                // proxy one.
                self.proxy = 1.0;
                self.playback_dirty = true;
                self.last_engine_graph = None;
                frames = 0;
                next = 12;
            }
            12 if frames > 5 => {
                let full = self.key_canvas();
                if self.comp.output_size() != full {
                    self.test_bake = Some((next, frames));
                    return;
                }
                let Some(bk) = bake_node(self) else { std::process::exit(1) };
                let full = self.subtree_bake_key(bk).map(|k| self.graph_bake_dir(k));
                let playing = self.loaded.get(&bk).map(|l| std::path::PathBuf::from(&l.path));
                if playing.is_none() || playing != full {
                    println!(
                        "BAKE TEST FAIL: at full res the node should play the full bake — \
                         playing {playing:?}, full {full:?}"
                    );
                    std::process::exit(1);
                }
                // Round four: a CROP that bakes itself. Same machinery, no
                // second node: the box alone lands on disk and plays back
                // where it sits.
                let crop = self.egraph.add(
                    NodeKind::SoftCrop(engine::graph::SoftCropParams {
                        rect: [0.25, 0.25, 0.75, 0.75],
                        mode: engine::graph::CropMode::Crop,
                        ..Default::default()
                    }),
                    egui::pos2(140.0, 60.0),
                );
                self.egraph.connect(crop, 0, 3);
                self.egraph.view = Some(crop);
                self.playback_dirty = true;
                self.test_bake_ref.clear();
                self.test_bake_last = None;
                frames = 0;
                next = 13;
            }
            // Live crop: the middle reads the plate, a corner is cut away.
            13 | 15 if frames > 10 && settled(self) => {
                let crop_node = |s: &Self| {
                    s.egraph
                        .nodes
                        .iter()
                        .find(|(_, n)| matches!(n.kind, NodeKind::SoftCrop(_)))
                        .map(|(id, _)| *id)
                };
                let Some(crop) = crop_node(self) else {
                    println!("BAKE TEST FAIL: the Crop vanished");
                    std::process::exit(1);
                };
                let Some((w, h)) = self.comp.output_size() else {
                    self.test_bake = Some((next, frames));
                    return;
                };
                let inside = self.probe_output_box(w / 2 - 8, h / 2 - 8, 16, 16, 1);
                let outside = self.probe_output_box(16, 16, 16, 16, 1);
                let (Some(inside), Some(outside)) = (inside, outside) else {
                    self.test_bake = Some((next, frames));
                    return;
                };
                let same = self.test_bake_last.is_some_and(|p| (p - inside).abs() < 1e-4);
                self.test_bake_last = Some(inside);
                if !(same && inside > 1e-4) {
                    self.test_bake = Some((next, frames));
                    return;
                }
                if outside.abs() > 1e-3 {
                    println!(
                        "BAKE TEST FAIL: outside the crop's box reads {outside:.4} ({})",
                        if stage == 13 { "live" } else { "baked" }
                    );
                    std::process::exit(1);
                }
                if stage == 13 {
                    self.test_bake_ref.push((self.timeline_pos, inside));
                    let bakes = self
                        .egraph
                        .nodes
                        .values()
                        .filter(|n| matches!(n.kind, NodeKind::GraphBake { .. }))
                        .count();
                    self.test_bake_ref.push((1, bakes as f32));
                    // The bake tab's button.
                    self.press_bake(crop);
                    frames = 0;
                    next = 14;
                } else {
                    let want = self.test_bake_ref[0].1;
                    if (inside - want).abs() > 1e-2 {
                        println!(
                            "BAKE TEST FAIL: the Crop's bake reads {inside:.4} in its box, live read {want:.4}"
                        );
                        std::process::exit(1);
                    }
                    // Finished and loaded, it is the CURRENT bake: its own
                    // frames once filed it under a new key and read STALE.
                    if !self.graph_bake_current(crop) {
                        println!("BAKE TEST FAIL: the Crop's fresh bake reads as stale");
                        std::process::exit(1);
                    }
                    println!(
                        "BAKE TEST PASS: baked and live agree frame for frame ({probes:?}), a \
                         drag lands where it stopped, a PLACED plate's bake plays at the \
                         comp's own frames, a full bake plays under proxy, a proxy bake \
                         lands at half res beside it and plays, full res plays the full bake again; a Crop baked \
                         itself (no new node), stored only its box and plays it back where \
                         it sits"
                    );
                    std::process::exit(0);
                }
            }
            // The Crop's bake landed: on the Crop, the box's size, played
            // through the Bake it lowers to.
            14 if self.auto_bake.is_none() && self.graph_bake.is_none() && frames > 5 => {
                let Some(crop) = self
                    .egraph
                    .nodes
                    .iter()
                    .find(|(_, n)| matches!(n.kind, NodeKind::SoftCrop(_)))
                    .map(|(id, _)| *id)
                else {
                    std::process::exit(1)
                };
                let bakes = self
                    .egraph
                    .nodes
                    .values()
                    .filter(|n| matches!(n.kind, NodeKind::GraphBake { .. }))
                    .count();
                if bakes as f32 != self.test_bake_ref[1].1 {
                    println!("BAKE TEST FAIL: the Crop's bake added a Bake node");
                    std::process::exit(1);
                }
                let Some(l) = self.loaded.get(&crop) else {
                    println!(
                        "BAKE TEST FAIL: the Crop's bake never landed ({:?})",
                        self.load_errors.get(&crop)
                    );
                    std::process::exit(1);
                };
                let dims = (l.seq.info.width, l.seq.info.height);
                let full = self.key_canvas().unwrap_or((0, 0));
                if dims.0.abs_diff(full.0 / 2) > 2 || dims.1.abs_diff(full.1 / 2) > 2 {
                    println!(
                        "BAKE TEST FAIL: the Crop's bake is {dims:?}, wanted its box (half of {full:?})"
                    );
                    std::process::exit(1);
                }
                let mut g = self.egraph.to_engine_graph_at(Some(self.eval_tick()));
                self.apply_bake_state(&mut g);
                let played = matches!(
                    g.nodes.get(&crop),
                    Some(NodeKind::GraphBake { baked: true, window: Some(_), .. })
                ) && matches!(g.nodes.get(&editor::crop_live_id(crop)), Some(NodeKind::SoftCrop(_)));
                if !played {
                    println!(
                        "BAKE TEST FAIL: the Crop does not play its bake: {:?}",
                        g.nodes.get(&crop)
                    );
                    std::process::exit(1);
                }
                // A bake walks the playhead to its end: back to the frame
                // the live crop was read at.
                self.seek_to(self.test_bake_ref[0].0);
                self.test_bake_last = None;
                frames = 0;
                next = 15;
            }
            _ => {}
        }
        self.test_bake = Some((next, frames));
    }
}
