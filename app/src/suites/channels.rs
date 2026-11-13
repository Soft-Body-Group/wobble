//! `--test-channels`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    pub(crate) fn drive_test_channels(&mut self) {
        let Some((stage, mut frames, sid)) = self.test.channels else { return };
        // Per-stage frame count (see the nodepack driver).
        frames += 1;
        if frames > 1500 {
            println!("CHANNELS TEST FAIL: timeout at stage {stage}");
            std::process::exit(1);
        }
        let mut next = stage;
        let passes = std::path::Path::new("data/2k_passes");
        match stage {
            0 if self.playback.is_some() && self.displayed_total >= 3 => {
                if !passes.is_dir() {
                    println!(
                        "CHANNELS TEST FAIL: {} missing — generate it first:\n  \
                         target/release/gen_exr data/2k_passes 512 288 8 zip passes",
                        passes.display()
                    );
                    std::process::exit(1);
                }
                self.load_read(0, &passes.to_string_lossy());
                next = 1;
            }
            // The layer list must come back from the file itself.
            1 if frames > 20 => {
                let Some(l) = self.loaded.get(&0) else { return };
                let names: Vec<String> =
                    l.layers.iter().map(|d| d.label().to_string()).collect();
                if names != ["rgba", "N", "depth", "diffuse"] {
                    println!("CHANNELS TEST FAIL: layers read as {names:?}");
                    std::process::exit(1);
                }
                // View the Read on its own and record the base rgba.
                self.egraph.view = Some(0);
                next = 2;
            }
            2 if frames > 30 => {
                // Wait for the newly loaded plate to actually be ON SCREEN
                // (a frame count alone catches the empty output between the
                // load landing and the rig rebuilding).
                let Some(base) = self.probe_output_mean() else { return };
                if base <= 0.01 {
                    if frames > 600 {
                        println!(
                            "CHANNELS TEST FAIL: base rgba still {base:.4} after the load"
                        );
                        std::process::exit(1);
                    }
                    return;
                }
                self.test_dump("channels-base-rgba");
                // Switch to the `diffuse` layer: different pixels entirely.
                if let Some(editor::ENode {
                    kind: NodeKind::Read { layer, .. }, ..
                }) = self.egraph.node_mut(0)
                {
                    *layer = Some("diffuse".into());
                }
                self.playback_dirty = true;
                self.test.channels = Some((3, 0, base.to_bits() as u64));
                return;
            }
            3 if frames > 60 => {
                let Some(diffuse) = self.probe_output_mean() else { return };
                let base = f32::from_bits(sid as u32);
                // The diffuse pass is a ramp, the beauty a lit gradient: if
                // the selection did nothing, these would be identical.
                if (diffuse - base).abs() < 0.01 {
                    if frames > 300 {
                        println!(
                            "CHANNELS TEST FAIL: layer 'diffuse' probed {diffuse:.4}, \
                             base rgba {base:.4} — the layer selection did nothing"
                        );
                        std::process::exit(1);
                    }
                    return;
                }
                self.test_dump("channels-diffuse");
                log::info!("channels: base rgba {base:.4}, diffuse {diffuse:.4}");
                // Now a Shuffle on top, wired to show alpha as grey. The
                // plate is opaque, so the result must be white.
                let sh = self.egraph.add(
                    NodeKind::Shuffle(engine::graph::ShuffleParams {
                        out: [
                            engine::graph::ShuffleSrc::A,
                            engine::graph::ShuffleSrc::A,
                            engine::graph::ShuffleSrc::A,
                            engine::graph::ShuffleSrc::One,
                        ],
                    }),
                    egui::pos2(300.0, 0.0),
                );
                self.egraph.connect(sh, 0, 0);
                self.egraph.view = Some(sh);
                self.test.channels = Some((4, 0, sh));
                return;
            }
            4 if frames > 60 => {
                let Some(m) = self.probe_output_mean() else { return };
                // Alpha is 1 across the plate, so alpha-as-grey is white.
                // A display transform sits between, so allow headroom, but
                // this is unmistakably not the gradient it replaced.
                if m > 0.9 {
                    // ...and killing a channel really removes it.
                    if let Some(editor::ENode {
                        kind: NodeKind::Shuffle(sh), ..
                    }) = self.egraph.node_mut(sid)
                    {
                        *sh = engine::graph::ShuffleParams {
                            out: [
                                engine::graph::ShuffleSrc::Zero,
                                engine::graph::ShuffleSrc::Zero,
                                engine::graph::ShuffleSrc::Zero,
                                engine::graph::ShuffleSrc::One,
                            ],
                        };
                    }
                    self.test.channels = Some((5, 0, sid));
                    return;
                }
                if frames > 300 {
                    println!(
                        "CHANNELS TEST FAIL: alpha-as-grey probed {m:.4} (wanted ~1.0)"
                    );
                    std::process::exit(1);
                }
            }
            5 if frames > 60 => {
                let Some(m) = self.probe_output_mean() else { return };
                if m < 0.02 {
                    println!(
                        "CHANNELS TEST PASS: read 4 layers from the file, 'diffuse' \
                         decoded different pixels from the beauty, shuffle showed \
                         alpha as white ({m:.4} once the colour was zeroed)"
                    );
                    std::process::exit(0);
                }
                if frames > 300 {
                    println!(
                        "CHANNELS TEST FAIL: zeroed colour still probes {m:.4}"
                    );
                    std::process::exit(1);
                }
            }
            _ => {}
        }
        let frames = if next == stage { frames } else { 0 };
        self.test.channels = Some((next, frames, sid));
    }
}
