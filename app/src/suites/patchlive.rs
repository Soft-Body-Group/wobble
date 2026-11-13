//! `--test-patchlive`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    /// --test-anim driver: key TranslateX at frames 0 and 12 on the default
    /// transform, play, and verify the values reaching the engine graph
    /// interpolate with time (including a mid-curve sample).
    /// --test-patchlive: wire ScaleX -> GainR in the patch bay, then edit
    /// the authored scale the way a widget drag would, and assert the LIVE
    /// engine graph (what write_params consumed) carries the driven gain.
    pub(crate) fn drive_test_patchlive(&mut self) {
        let Some((stage, mut frames)) = self.test.patchlive else {
            return;
        };
        frames += 1;
        if frames > 1200 {
            println!("PATCHLIVE TEST FAIL: timeout at stage {stage}");
            std::process::exit(1);
        }
        let gain_r = |g: &Option<engine::graph::Graph>| {
            g.as_ref()
                .and_then(|g| g.nodes.get(&3))
                .and_then(|k| anim::get(k, anim::Channel::GainR))
        };
        match stage {
            // Default comp: 2 = Transform, 3 = Grade (in the viewed chain).
            0 if self.displayed_total >= 3 => {
                if let Some(n) = self.egraph.node_mut(2) {
                    if let NodeKind::Transform(t) = &mut n.kind {
                        t.scale = [3.0, 1.0];
                    }
                }
                self.egraph.patch.link(
                    editor::PSrc::Chan(2, anim::Channel::ScaleX),
                    editor::PDst::Chan(3, anim::Channel::GainR),
                );
                self.test.patchlive = Some((1, frames));
                return;
            }
            1 => {
                if gain_r(&self.last_engine_graph) == Some(3.0) {
                    // Now the "user drags the source slider" step.
                    if let Some(n) = self.egraph.node_mut(2) {
                        if let NodeKind::Transform(t) = &mut n.kind {
                            t.scale = [5.0, 1.0];
                        }
                    }
                    self.test.patchlive = Some((2, frames));
                    return;
                }
            }
            2 => {
                if gain_r(&self.last_engine_graph) == Some(5.0) {
                    // Authored grade must still be untouched.
                    let authored = self
                        .egraph
                        .node(3)
                        .and_then(|n| anim::get(&n.kind, anim::Channel::GainR));
                    println!(
                        "PATCHLIVE TEST PASS: live gain follows the patched scale (3 -> 5), authored gain {:?}",
                        authored
                    );
                    std::process::exit(0);
                }
            }
            _ => {}
        }
        self.test.patchlive = Some((stage, frames));
    }
}
