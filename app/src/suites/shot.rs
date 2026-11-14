//! `--test-shot`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    /// --test-box driver. Default graph ids: ra=0 rb=1 tf=2 gr=3 mg=4.
    /// Drag a selection box around tf+gr on the background, verify both
    /// selected; then click gr and ⌘↑ / ⌘↓ it through the chain, verifying
    /// the rewire both ways.
    /// --test-shot driver: with NO footage loaded, the shot's own range has
    /// to give the comp a timeline — a running clock, a moving playhead and
    /// animation that follows it.
    pub(crate) fn drive_test_shot(&mut self) {
        let Some((stage, mut frames, mark)) = self.test.shot else { return };
        frames += 1;
        if frames > 1200 {
            println!(
                "SHOT TEST FAIL: timeout at stage {stage} (pos {}, len {}, playing {})",
                self.timeline_pos,
                self.shot_len(),
                self.playing
            );
            std::process::exit(1);
        }
        let mut next = stage;
        match stage {
            0 if frames >= 5 => {
                if self.playback.is_some() {
                    println!("SHOT TEST FAIL: expected no footage (run without plate args)");
                    std::process::exit(1);
                }
                // A comp with no readers still has a duration.
                if self.shot_len() != Self::DEFAULT_SHOT_FRAMES {
                    println!("SHOT TEST FAIL: default shot len {}", self.shot_len());
                    std::process::exit(1);
                }
                // Animate a node so we can prove the clock drives evaluation.
                if let Some(n) = self.egraph.node_mut(2) {
                    let c = n.anim.entry(anim::Channel::TranslateX).or_default();
                    c.set_key(0, 0.0);
                    c.set_key(20, 200.0);
                }
                self.comp_range = Some((0, 23)); // the shot range wins
                self.playing = true;
                self.test.shot = Some((1, 0, self.timeline_pos));
                return;
            }
            1 if frames >= 40 => {
                if self.shot_len() != 24 {
                    println!("SHOT TEST FAIL: comp_range len {}", self.shot_len());
                    std::process::exit(1);
                }
                // The clock must have moved on its own, and stayed inside the
                // range rather than running off past it.
                if self.timeline_pos == mark {
                    println!("SHOT TEST FAIL: clock never advanced (pos {mark})");
                    std::process::exit(1);
                }
                if self.timeline_pos >= 24 {
                    println!("SHOT TEST FAIL: pos {} outside the range", self.timeline_pos);
                    std::process::exit(1);
                }
                self.playing = false;
                self.seek_to(10);
                next = 2;
            }
            2 if frames >= 3 => {
                // Paused seek lands, and animation evaluates there: the
                // curve is 0 at frame 0 and 200 at 20, so frame 10 is 100.
                let x = match self.egraph.to_engine_graph_at(Some(self.eval_tick())).nodes.get(&2)
                {
                    Some(engine::graph::NodeKind::Transform(t)) => t.translate[0],
                    _ => f32::NAN,
                };
                if self.timeline_pos != 10 || (x - 100.0).abs() > 1e-3 {
                    println!(
                        "SHOT TEST FAIL: pos {} eval_tick {} translate.x {x}",
                        self.timeline_pos,
                        self.eval_tick()
                    );
                    std::process::exit(1);
                }
                println!(
                    "SHOT TEST PASS: no footage — default {} frames, range 0-23 honoured, \
                     clock ran, paused seek to 10 evaluated animation at {x}",
                    Self::DEFAULT_SHOT_FRAMES
                );
                std::process::exit(0);
            }
            _ => {}
        }
        self.test.shot = Some((next, frames, mark));
    }
}
