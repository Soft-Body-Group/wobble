//! `--test-cost`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    /// --test-cost: the slow-path colouring, in the LIVE app.
    ///
    /// The propagation itself is unit-tested in `costviz`; what only a real
    /// run shows is the wiring — that the app recomputes the taint every
    /// frame from its own bake state and hands it to the editor, so the
    /// noodles are coloured from the same answer the Write warning reads.
    /// A stale `cost_flow` would look perfectly fine in a screenshot.
    pub(crate) fn drive_test_cost(&mut self) {
        let Some((stage, mut frames, slow)) = self.test_cost else { return };
        frames += 1;
        // Stages 3+ wait on the wall clock (their own budgets).
        if stage < 3 && frames > 2400 {
            println!("COST TEST FAIL: timeout at stage {stage}");
            std::process::exit(1);
        }
        let fail = |m: String| -> ! {
            println!("COST TEST FAIL: {m}");
            std::process::exit(1);
        };
        match stage {
            0 if self.playback.is_some() && self.displayed_total >= 2 => {
                // Blur (slow) -> Grade (fast in itself) off the default
                // graph's Read, viewed through the Grade.
                let b = self
                    .egraph
                    .add(NodeKind::Blur(Default::default()), egui::pos2(320.0, 40.0));
                let g = self
                    .egraph
                    .add(NodeKind::Grade(Default::default()), egui::pos2(320.0, 120.0));
                self.egraph.connect(b, 0, 0);
                self.egraph.connect(g, 0, b);
                self.egraph.view = Some(g);
                self.test_cost = Some((1, frames, b));
            }
            1 if frames > 10 => {
                let f = self.cost_flow.clone();
                if !f.is_slow(slow) {
                    fail("the Blur itself is not marked slow".into());
                }
                let g = self.egraph.view.unwrap();
                if !f.is_slow(g) {
                    fail("the Grade downstream of the Blur is not tainted".into());
                }
                // Slow only on its own account (not cached), never the Blur's.
                if f.is_slow(0) && !f.uncached.contains(&0) {
                    fail("the Read upstream of the Blur was tainted".into());
                }
                // The editor must be reading the SAME answer, this frame.
                if self.node_editor.cost_flow != f {
                    fail("the editor's flow is not the app's flow".into());
                }
                // Bypass the Blur: no work, so no red.
                self.egraph.nodes.get_mut(&slow).unwrap().disabled = true;
                self.test_cost = Some((2, frames, slow));
            }
            2 if frames > 4 => {
                if self.cost_flow.is_slow(self.egraph.view.unwrap())
                    && !self.cost_flow.uncached.contains(&0)
                {
                    fail("a BYPASSED Blur still taints downstream".into());
                }
                // A Load with no cache tier at all decodes from the source
                // every time: it goes red, and its wires with it.
                if let Some(NodeKind::Read { ram_cache, nvme_cache, .. }) =
                    self.egraph.node_mut(0).map(|n| &mut n.kind)
                {
                    *ram_cache = false;
                    *nvme_cache = false;
                }
                self.playback_dirty = true;
                self.test_cost = Some((3, 0, slow));
            }
            // Wall-clock waits: the answer is refreshed twice a second.
            3 if frames > 4 => {
                static T0: std::sync::Mutex<Option<Instant>> = std::sync::Mutex::new(None);
                let t0 = *T0.lock().unwrap().get_or_insert_with(Instant::now);
                if !self.cost_flow.uncached.contains(&0) || !self.cost_flow.is_slow(0) {
                    if t0.elapsed().as_secs() > 5 {
                        fail("a Load with no cache tier is not red".into());
                    }
                    self.test_cost = Some((3, frames, slow));
                    return;
                }
                if !self.cost_flow.is_slow(self.egraph.view.unwrap()) {
                    fail("an uncached Load does not taint below it".into());
                }
                if self.cost_flow.served.get(&0) != Some(&false) {
                    fail("an uncached Load is not drawn red".into());
                }
                // Precache it: every frame resident, the red clears.
                self.precache_start(vec![0]);
                self.test_cost = Some((4, 0, slow));
            }
            4 => {
                static T0: std::sync::Mutex<Option<Instant>> = std::sync::Mutex::new(None);
                let t0 = *T0.lock().unwrap().get_or_insert_with(Instant::now);
                if self.cost_flow.uncached.contains(&0)
                    || self.cost_flow.is_slow(0)
                    || self.cost_flow.served.get(&0) != Some(&true)
                {
                    if t0.elapsed().as_secs() > 20 {
                        fail(format!(
                            "a precached Load is still red (served {:?})",
                            self.cost_flow.served.get(&0)
                        ));
                    }
                    self.test_cost = Some((4, frames, slow));
                    return;
                }
                println!(
                    "COST TEST PASS: a Blur marks itself and everything downstream slow, the \
                     Read above it stays fast, bypassing it clears the taint, an uncached \
                     Load is red until a precache fills its in..out and then green, and the editor draws from \
                     the same per-frame answer the panels read"
                );
                std::process::exit(0);
            }
            _ => self.test_cost = Some((stage, frames, slow)),
        }
    }
}
