//! `--test-anim`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    pub(crate) fn drive_test_anim(&mut self) {
        let Some((stage, mut frames, last_t, mut good, mut interp)) = self.test_anim else {
            return;
        };
        frames += 1;
        if frames > 2400 {
            println!("ANIM TEST FAIL: timeout at stage {stage} (good {good}, interp {interp})");
            std::process::exit(1);
        }
        match stage {
            0 if self.playback.is_some() && self.displayed_total >= 3 => {
                if let Some(n) = self.egraph.node_mut(2) {
                    let c = n.anim.entry(anim::Channel::TranslateX).or_default();
                    c.set_key(0, 0.0);
                    c.set_key(12, 240.0);
                }
                self.playing = true;
                self.fps = 60.0;
                self.test_anim = Some((1, frames, u64::MAX, 0, false));
                return;
            }
            1 => {
                let t = self.eval_tick();
                let len = self.active_len().unwrap_or(1);
                if t != last_t {
                    let expected = if t >= 12 {
                        240.0
                    } else {
                        t as f32 / 12.0 * 240.0
                    };
                    let actual = self
                        .last_engine_graph
                        .as_ref()
                        .and_then(|g| g.nodes.get(&2))
                        .and_then(|k| match k {
                            NodeKind::Transform(tr) => Some(tr.translate[0]),
                            _ => None,
                        });
                    match actual {
                        Some(a) if (a - expected).abs() < 0.51 => {
                            good += 1;
                            if expected > 1.0 && expected < 239.0 {
                                interp = true;
                            }
                        }
                        Some(a) => {
                            println!(
                                "ANIM TEST FAIL: tick {t} (len {len}) expected {expected:.1}, engine saw {a:.1}"
                            );
                            std::process::exit(1);
                        }
                        None => {}
                    }
                    if good >= 4 && interp {
                        let authored = match self.egraph.node(2).map(|n| &n.kind) {
                            Some(NodeKind::Transform(tr)) => tr.translate[0],
                            _ => f32::NAN,
                        };
                        if authored != 0.0 {
                            println!(
                                "ANIM TEST FAIL: evaluation leaked into authored params ({authored})"
                            );
                            std::process::exit(1);
                        }
                        println!(
                            "ANIM TEST PASS: {good} ticks matched the curve (incl. mid-curve interpolation), authored params untouched"
                        );
                        std::process::exit(0);
                    }
                    self.test_anim = Some((1, frames, t, good, interp));
                    return;
                }
            }
            _ => {}
        }
        self.test_anim = Some((stage, frames, last_t, good, interp));
    }
}
