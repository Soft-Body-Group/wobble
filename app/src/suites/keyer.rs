//! `--test-keyer`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    pub(crate) fn drive_test_keyer(&mut self) {
        let Some((stage, mut frames, id)) = self.test.keyer else { return };
        frames += 1;
        if frames > 1200 {
            println!("KEYER TEST FAIL: timeout at stage {stage}");
            std::process::exit(1);
        }
        match stage {
            0 if self.playback.is_some() && self.displayed_total >= 3 => {
                // A topology change that does NOT move the view target, so
                // it compiles ASYNCHRONOUSLY — the path where params were
                // left packed for the outgoing layout. LumaKey (lo/hi below
                // any real luma => mask fully ON) drives the Grade's mask.
                let id = self.egraph.add(
                    NodeKind::LumaKey(engine::graph::LumaKeyParams {
                        bands: vec![engine::graph::LumaBand {
                            lo: 0.0,
                            hi: 0.02,
                            invert: false,
                            ..Default::default()
                        }],
                        // The suite authors this band's ends directly, so
                        // it is the legacy (unpartitioned) shape.
                        ranges: None,
                    }),
                    egui::Pos2::new(300.0, 0.0),
                );
                self.egraph.connect(id, 0, 0); // LumaKey <- Read A
                self.egraph.connect(3, 1, id); // Grade.mask <- LumaKey
                self.test.keyer = Some((1, frames, id));
                return;
            }
            1 => {
                // Let the compile land and a few frames render through it.
                if frames > 40 {
                    self.test.keyer = Some((2, frames, id));
                    return;
                }
            }
            2 => {
                // The mechanism check: once a topology compile has landed,
                // the uniform buffer must belong to the LIVE layout. (The
                // rendered symptom is luck-dependent — a foreign slot can
                // hold values that happen to look right — so assert the
                // invariant as well as the pixels.)
                if self.comp.params_stale() {
                    println!(
                        "KEYER TEST FAIL: params still packed for the outgoing layout after the compile landed"
                    );
                    std::process::exit(1);
                }
                let mean = self.probe_output_mean();
                match mean {
                    Some(m) if m > 0.5 => {
                        println!(
                            "KEYER TEST PASS: fresh LumaKey topology renders its own params (mask {m:.3}, uniforms match the live layout)"
                        );
                        std::process::exit(0);
                    }
                    Some(m) => {
                        println!(
                            "KEYER TEST FAIL: mask reads {m:.3} (expected ~1.0) — params stale for the live layout?"
                        );
                        std::process::exit(1);
                    }
                    None => {
                        println!("KEYER TEST FAIL: no output to probe");
                        std::process::exit(1);
                    }
                }
            }
            _ => {}
        }
        self.test.keyer = Some((stage, frames, id));
    }
}
