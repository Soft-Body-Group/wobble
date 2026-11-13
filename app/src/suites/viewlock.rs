//! `--test-viewlock`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    /// --test-viewlock: Read -> Grade (collapsed into a subwob) -> Transform
    /// tail. Lock viewer 1 on the tail (the alt+1 request), dive into the
    /// subwob, halve the Grade's gain inside: the viewer still shows the
    /// TAIL, and its pixels halve. Step out: the lock is still there.
    pub(crate) fn drive_test_viewlock(&mut self, raw: &mut egui::RawInput) {
        let Some((stage, mut frames, ids, before)) = self.test_viewlock else { return };
        let [inst, tail] = ids;
        frames += 1;
        if frames > 1200 {
            println!("VIEWLOCK TEST FAIL: timeout at stage {stage}");
            std::process::exit(1);
        }
        let fail = |m: String| -> ! {
            println!("VIEWLOCK TEST FAIL: {m}");
            std::process::exit(1);
        };
        let spot = self.image_rect.min + egui::vec2(0.41, 0.47) * self.image_rect.size();
        if stage >= 1 {
            raw.events.push(egui::Event::PointerMoved(spot));
        }
        let (mut ids, mut before) = (ids, before);
        let mut next = stage;
        match stage {
            0 if self.playback.is_some() && self.displayed_total >= 2 => {
                let read = self
                    .egraph
                    .nodes
                    .iter()
                    .find(|(_, n)| matches!(n.kind, NodeKind::Read { .. }))
                    .map(|(id, _)| *id)
                    .unwrap_or_else(|| fail("needs a Read".into()));
                let grade =
                    self.egraph.add(NodeKind::Grade(Default::default()), egui::pos2(300.0, 120.0));
                let tail =
                    self.egraph.add(NodeKind::Transform(Default::default()), egui::pos2(300.0, 220.0));
                self.egraph.connect(grade, 0, read);
                self.egraph.connect(tail, 0, grade);
                self.selected = vec![grade];
                self.collapse_selection();
                let inst = self.selected.first().copied().unwrap_or(0);
                if self.egraph.input_of(tail, 0) != Some(inst) {
                    fail("collapse did not leave the subwob feeding the tail".into());
                }
                self.playing = false;
                self.seek_to(5);
                // What alt+1 over the tail asks for.
                self.node_editor.view_lock_req = Some(Some((tail, 0)));
                self.picker.size = 0;
                self.picker.on = true;
                ids = [inst, tail];
                next = 1;
            }
            1 if frames > 60 && self.picker.live.is_some() => {
                if self.view_lock.map(|l| l.node) != Some(tail) {
                    fail(format!("alt+1 did not lock the tail: {:?}", self.view_lock));
                }
                before = self.picker.live.unwrap().rgba;
                if before[..3].iter().all(|v| v.abs() < 0.02) {
                    fail(format!("the plate reads black at the spot: {before:?}"));
                }
                if let Err(e) = self.dive_into(inst) {
                    fail(format!("dive: {e}"));
                }
                next = 2;
            }
            2 if frames > 5 => {
                if self.view_lock.is_none() {
                    fail("the lock let go on diving in".into());
                }
                let g = self
                    .egraph
                    .nodes
                    .iter()
                    .find(|(_, n)| matches!(n.kind, NodeKind::Grade(_)))
                    .map(|(id, _)| *id)
                    .unwrap_or_else(|| fail("no Grade inside the subwob".into()));
                if let Some(editor::ENode { kind: NodeKind::Grade(p), .. }) = self.egraph.node_mut(g) {
                    p.gain = [0.5; 3];
                }
                self.playback_dirty = true;
                let t = self.eval_tick();
                match self.render_preview(t) {
                    Some(p) if p.view == Some(tail) => {}
                    Some(p) => fail(format!("inside, the viewer shows {:?}, not the tail", p.view)),
                    None => fail("inside, nothing is spliced for the lock".into()),
                }
                next = 3;
            }
            3 if frames > 90 && self.picker.live.is_some() => {
                let now = self.picker.live.unwrap().rgba;
                for k in 0..3 {
                    if before[k] > 0.02 && ((now[k] / before[k]) - 0.5).abs() > 0.03 {
                        fail(format!(
                            "the locked tail read {before:?}, then {now:?} after halving the inner gain"
                        ));
                    }
                }
                if let Err(e) = self.dive_out() {
                    fail(format!("dive out: {e}"));
                }
                next = 4;
            }
            4 if frames > 5 => {
                if self.view_lock.map(|l| (l.depth, l.node)) != Some((0, tail)) {
                    fail(format!("stepping out lost the lock: {:?}", self.view_lock));
                }
                println!(
                    "VIEWLOCK TEST PASS: the locked tail stayed on screen inside the subwob and halved with the inner gain ({before:?})"
                );
                std::process::exit(0);
            }
            _ => {}
        }
        self.test_viewlock = Some((next, if next == stage { frames } else { 0 }, ids, before));
    }
}
