//! `--test-range`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    /// Frame ranges as noodle metadata, end to end: a placed plate's noodle
    /// and everything downstream carry its frames; viewing it puts them on
    /// the bar and the playhead on them; a generator runs for the script
    /// range until it defines its own; and a viewer range pinned past a
    /// plate's end gets that plate's edge behaviour — black, hold, loop.
    pub(crate) fn drive_test_range(&mut self) {
        let Some((stage, mut frames)) = self.test.range else { return };
        frames += 1;
        let elapsed = self.test.range_t0.get_or_insert_with(Instant::now).elapsed();
        if elapsed > web_time::Duration::from_secs(90) {
            println!(
                "RANGE TEST FAIL: timeout at stage {stage} (frame {}, span {:?}, seen {:?})",
                self.timeline_pos,
                self.shot_span(),
                self.test.range_seen
            );
            std::process::exit(1);
        }
        let fail = |msg: String| -> ! {
            println!("RANGE TEST FAIL: {msg}");
            std::process::exit(1);
        };
        const PLACE: u64 = 1011;
        let media = self.loaded.get(&0).map(|l| l.seq.frames.len() as u64).unwrap_or(0);
        let last = PLACE + media.saturating_sub(1);
        // A reading is believed once the playhead has landed and two
        // successive probes agree (black is a legitimate answer here).
        let settled = |s: &Self| !s.force_display && s.eval_tick() == s.timeline_pos;
        let stable = |s: &mut Self| -> Option<f32> {
            if !settled(s) {
                s.test.range_last = None;
                return None;
            }
            let v = s.probe_output_box(200, 200, 32, 32, 1)?;
            let same = s.test.range_last.is_some_and(|p| (p - v).abs() < 1e-4);
            s.test.range_last = Some(v);
            same.then_some(v)
        };
        let set_after = |s: &mut Self, e: engine::graph::ReadEdge| {
            if let Some(NodeKind::Read { edges, .. }) = s.egraph.node_mut(0).map(|n| &mut n.kind) {
                edges.after = e;
            }
        };
        let mut next = stage;
        match stage {
            // Place the plate at 1011 and view its graded branch, with the
            // playhead parked at 0 — the "fresh clip, nothing plays" repro.
            0 if self.playback.is_some() && self.displayed_total >= 3 && media > 8 => {
                // FIRST / LAST come filled in from the files.
                match self.egraph.node(0).map(|n| &n.kind) {
                    Some(NodeKind::Read { first, last, start, .. })
                        if *first == Some(start.unwrap_or(0))
                            && *last == Some(start.unwrap_or(0) + media as i64 - 1) => {}
                    other => fail(format!(
                        "a loaded Read's first/last were not filled from its files: {:?}",
                        other.map(|k| match k {
                            NodeKind::Read { first, last, start, .. } => (*start, *first, *last),
                            _ => (None, None, None),
                        })
                    )),
                }
                // The RAM tier on, so "never cached" has something to prove.
                if let Some(NodeKind::Read { ram_cache, .. }) =
                    self.egraph.node_mut(0).map(|n| &mut n.kind)
                {
                    *ram_cache = true;
                }
                self.playing = false;
                self.seek_to(0);
                self.place_read(0, PLACE as i64);
                self.egraph.view = Some(3);
                self.playback_dirty = true;
                // The range view on, so stage 1 can read its tags.
                self.prefs.graph_ranges = Some(true);
                frames = 0;
                next = 1;
            }
            1 if frames > 10 => {
                // The noodles: the plate's and every one downstream of it.
                for (id, what) in [(0u64, "the Read"), (2, "the Transform"), (3, "the Grade")] {
                    let got = self.node_editor.wire_ranges.get(&id).copied();
                    if got != Some((PLACE, last)) {
                        fail(format!("{what}'s noodle carries {got:?}, wanted {:?}", (PLACE, last)));
                    }
                }
                // The range view draws the plate's range on its noodle, and
                // says the Read SETS it while the Transform below carries it.
                let tags = &self.node_editor.shown_range_tags;
                if !tags.contains(&(0, (PLACE, last))) {
                    fail(format!("the range view drew {tags:?}, wanted the Read's {:?}", (PLACE, last)));
                }
                if !self.node_editor.range_set_here(&self.egraph, 0)
                    || self.node_editor.range_set_here(&self.egraph, 2)
                {
                    fail("the range view should mark the Read as setting its range, the Transform as carrying it".into());
                }
                let detail = self.node_editor.range_detail(&self.egraph, 0, (PLACE, last));
                if !detail.contains("from media") || !detail.contains("outside:") {
                    fail(format!("the Read noodle's hover says {detail:?}"));
                }
                self.prefs.graph_ranges = Some(false);
                // The viewer follows the noodle it shows, and the playhead
                // was moved onto it.
                if self.shot_span() != (PLACE, last) {
                    fail(format!("viewing it spans {:?}, wanted {:?}", self.shot_span(), (PLACE, last)));
                }
                if !(PLACE..=last).contains(&self.timeline_pos) {
                    fail(format!("the playhead stayed at {} outside {PLACE}..{last}", self.timeline_pos));
                }
                // A generator runs for the SCRIPT range...
                self.comp_range = Some((100, 149));
                let grid = self.egraph.add(
                    NodeKind::Grid(Default::default()),
                    egui::pos2(300.0, -200.0),
                );
                self.egraph.view = Some(grid);
                frames = 0;
                next = 2;
            }
            2 if frames > 4 => {
                let grid = self.egraph.view.unwrap_or(0);
                if self.shot_span() != (100, 149) {
                    fail(format!("a generator viewed spans {:?}, wanted the script's (100, 149)", self.shot_span()));
                }
                // ...until it defines its own.
                if let Some(n) = self.egraph.node_mut(grid) {
                    n.range = Some((5, 9));
                }
                frames = 0;
                next = 3;
            }
            3 if frames > 4 => {
                if self.shot_span() != (5, 9) {
                    fail(format!("a generator with its own range spans {:?}, wanted (5, 9)", self.shot_span()));
                }
                if self.node_editor.wire_ranges.get(&self.egraph.view.unwrap_or(0)) != Some(&(5, 9)) {
                    fail("the generator's noodle does not carry its own range".into());
                }
                // Back to the plate, with the viewer PINNED six frames past
                // its end: those frames are the plate's edge behaviour.
                self.egraph.view = Some(3);
                self.comp_range = None;
                self.view_range = Some((PLACE, last + 6));
                set_after(self, engine::graph::ReadEdge::Black);
                self.seek_to(last + 3);
                self.test.range_last = None;
                frames = 0;
                next = 4;
            }
            4 if frames > 6 => {
                let Some(v) = stable(self) else { self.test.range = Some((next, frames)); return };
                if v.abs() > 1e-3 {
                    fail(format!("past the end with a BLACK edge the plate reads {v:.4}"));
                }
                self.test.range_seen.push((last + 3, v));
                set_after(self, engine::graph::ReadEdge::Hold);
                self.test.range_last = None;
                frames = 0;
                next = 5;
            }
            5 if frames > 6 => {
                let Some(v) = stable(self) else { self.test.range = Some((next, frames)); return };
                self.test.range_seen.push((last + 3, v));
                // What the last real frame shows, to compare the hold with.
                self.seek_to(last);
                self.test.range_last = None;
                frames = 0;
                next = 6;
            }
            6 if frames > 6 => {
                let Some(v) = stable(self) else { self.test.range = Some((next, frames)); return };
                let held = self.test.range_seen[1].1;
                if (v - held).abs() > 1e-3 || v < 1e-3 {
                    fail(format!("a HELD frame past the end reads {held:.4}, the last frame {v:.4}"));
                }
                self.test.range_seen.push((last, v));
                // Loop: three past the end is the clip's frame 2 again.
                set_after(self, engine::graph::ReadEdge::Loop);
                self.seek_to(last + 3);
                self.test.range_last = None;
                frames = 0;
                next = 7;
            }
            7 if frames > 6 => {
                let Some(v) = stable(self) else { self.test.range = Some((next, frames)); return };
                self.test.range_seen.push((last + 3, v));
                self.seek_to(PLACE + 2);
                self.test.range_last = None;
                frames = 0;
                next = 8;
            }
            8 if frames > 6 => {
                let Some(v) = stable(self) else { self.test.range = Some((next, frames)); return };
                let looped = self.test.range_seen[3].1;
                if (v - looped).abs() > 1e-3 {
                    fail(format!(
                        "a LOOPED frame past the end reads {looped:.4}, the clip's frame 2 {v:.4}"
                    ));
                }
                // The Read's LAST pushed four frames past its files: those
                // frames are INSIDE its range with nothing on disk — black
                // (not held, whatever the edge says), and never cached.
                set_after(self, engine::graph::ReadEdge::Hold);
                if let Some(NodeKind::Read { last: l, .. }) =
                    self.egraph.node_mut(0).map(|n| &mut n.kind)
                {
                    *l = Some(last as i64 + 4);
                }
                self.playback_dirty = true;
                self.seek_to(last + 2);
                self.test.range_last = None;
                frames = 0;
                next = 9;
            }
            9 if frames > 8 => {
                let Some(v) = stable(self) else { self.test.range = Some((next, frames)); return };
                if v.abs() > 1e-3 {
                    fail(format!("a frame in the Read's range with no file reads {v:.4}, not black"));
                }
                if self.node_editor.wire_ranges.get(&0) != Some(&(PLACE, last + 4)) {
                    fail(format!(
                        "the Read's noodle carries {:?} after LAST moved, wanted {:?}",
                        self.node_editor.wire_ranges.get(&0),
                        (PLACE, last + 4)
                    ));
                }
                // Never cached: the missing frame's slot in the reader's RAM
                // residency is empty while a real frame it showed is held.
                let res = self.playback.as_ref().and_then(|pb| {
                    let i = pb.slots.iter().position(|s| *s == 0)?;
                    Some(pb.player.ram_residency(i).0)
                });
                let missing_idx = (last + 2 - PLACE) as usize;
                if res.as_ref().and_then(|r| r.get(missing_idx)).copied() == Some(true) {
                    fail("a frame with no file on disk went into the RAM cache".into());
                }
                println!(
                    "RANGE TEST PASS: noodles carry {PLACE}..{last} downstream of a placed plate, \
                     the viewer follows them (and moved the playhead onto them), a generator runs \
                     for the script range until it sets its own, past the end the plate answers \
                     black, held and looped, and a Read's first/last fill from its files and read \
                     black — uncached — where there is no file"
                );
                std::process::exit(0);
            }
            _ => {}
        }
        self.test.range = Some((next, frames));
    }
}
