//! `--test-cachemeter`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    /// The Read cache meter reports REAL residency: play a while with the
    /// RAM tier on, and the reported occupancy has to climb from nothing to
    /// the frames actually decoded — and never exceed the sequence length.
    pub(crate) fn drive_test_cachemeter(&mut self) {
        let Some((stage, mut frames)) = self.test_cachemeter else { return };
        frames += 1;
        if frames > 2400 {
            println!("CACHEMETER TEST FAIL: timeout at stage {stage}");
            std::process::exit(1);
        }
        // Node 0 is a Read; find its player-side residency the same way the
        // meter does.
        let residency = |s: &Self| -> Option<(usize, usize)> {
            let NodeKind::Read { source, .. } = s.egraph.node(0).map(|n| n.kind.clone())?
            else {
                return None;
            };
            let pb = s.playback.as_ref()?;
            let i = pb.slots.iter().position(|x| *x == source)?;
            let (flags, _) = pb.player.ram_residency(i);
            Some((flags.iter().filter(|b| **b).count(), flags.len()))
        };
        match stage {
            0 if self.playback.is_some() && self.displayed_total >= 2 => {
                // RAM tier is off by default now: the meter test opts in,
                // exactly as a user would on the Read panel.
                if let Some(editor::ENode {
                    kind: NodeKind::Read { ram_cache, .. }, ..
                }) = self.egraph.node_mut(0)
                {
                    *ram_cache = true;
                }
                self.playback_dirty = true;
                let Some((n, len)) = residency(self) else {
                    println!("CACHEMETER TEST FAIL: no residency for node 0");
                    std::process::exit(1);
                };
                if len == 0 {
                    println!("CACHEMETER TEST FAIL: sequence reported 0 frames");
                    std::process::exit(1);
                }
                self.playing = true;
                self.fps = 60.0;
                self.test_cachemeter = Some((1, frames));
                log::info!("cachemeter: start {n}/{len}");
            }
            // Gate on the reported residency itself, not on how many frames
            // the headless run manages to display (that is throughput, and
            // it varies with machine load).
            1 => {
                let Some((n, len)) = residency(self) else {
                    println!("CACHEMETER TEST FAIL: residency vanished mid-play");
                    std::process::exit(1);
                };
                if n > len {
                    println!("CACHEMETER TEST FAIL: {n} resident > {len} frames");
                    std::process::exit(1);
                }
                if n >= 4 {
                    println!(
                        "CACHEMETER TEST PASS: RAM residency climbed to {n}/{len} frames ({} displayed)",
                        self.displayed_total
                    );
                    std::process::exit(0);
                }
                self.test_cachemeter = Some((1, frames));
            }
            _ => self.test_cachemeter = Some((stage, frames)),
        }
    }
}
