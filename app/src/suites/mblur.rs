//! `--test-mblur`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    /// --test-channels driver: point Read A at the multi-layer plate, read a
    /// NAMED layer through it, and prove a Shuffle rewires channels — both
    /// against real pixels, since either could look plausible and be wrong.
    /// --test-mblur driver: does an animated parameter UPSTREAM of a
    /// MotionBlur node get averaged across its shutter — and one downstream
    /// of it not?
    ///
    /// The gain is keyed as a triangle — 0 at frame 5, 4 at 6, 0 at 7 — and
    /// the playhead parks on the peak. Sampled once, the frame is gain 4.
    /// Averaged over a whole-frame shutter centred on 6, the mean of that
    /// triangle across [5.5, 6.5] is 3. So with the node BELOW the Grade the
    /// frame is exactly three quarters as bright — a number rather than a
    /// vague "it looks softer" — and with the node moved ABOVE the Grade it
    /// is exactly as bright as the unblurred one, because the Grade is now
    /// downstream and runs once, at the frame.
    pub(crate) fn drive_test_mblur(&mut self) {
        let Some((stage, mut frames, bits)) = self.test.mblur else { return };
        frames += 1;
        if frames > 900 {
            println!("MBLUR TEST FAIL: timeout at stage {stage}");
            std::process::exit(1);
        }
        // Node 3 is the default graph's Grade.
        const GRADE: u64 = 3;
        let blur = |sh: engine::graph::MotionBlurParams| {
            NodeKind::MotionBlur(engine::graph::MotionBlurParams { samples: 16, angle: 1.0, ..sh })
        };
        // Stages 2 and 3 changed TOPOLOGY, which compiles asynchronously:
        // the count starts once the node is in the live plan, or the probe
        // reads the old pipeline's picture (a frame count is not a duration).
        let live = self.last_pipeline_ready
            && self.egraph.nodes.iter().any(|(id, n)| {
                matches!(n.kind, NodeKind::MotionBlur(_))
                    && self.comp.motion_blur_stages().iter().any(|s| s.0 == *id)
            });
        if stage >= 2 && !live {
            self.test.mblur = Some((stage, 0, bits));
            return;
        }
        let mut next = stage;
        match stage {
            0 if self.playback.is_some() && self.displayed_total >= 3 => {
                // Key around wherever the playhead already IS. Seeking it
                // somewhere first is a second thing to get right, and this
                // test is not about seeking — stage 1 checks the tick has
                // not moved, which is the part that would invalidate it.
                let t0 = self.eval_tick().max(1);
                self.mblur_peak = t0;
                self.playing = false;
                if let Some(n) = self.egraph.node_mut(GRADE) {
                    for ch in [
                        anim::Channel::GainR,
                        anim::Channel::GainG,
                        anim::Channel::GainB,
                    ] {
                        let c = n.anim.entry(ch).or_default();
                        c.set_key(t0 - 1, 0.0);
                        c.set_key(t0, 4.0);
                        c.set_key(t0 + 1, 0.0);
                    }
                } else {
                    println!("MBLUR TEST FAIL: no Grade to animate");
                    std::process::exit(1);
                }
                self.playback_dirty = true;
                next = 1;
            }
            // Unblurred: the instantaneous value at the peak. Then a
            // MotionBlur spliced in BELOW the Grade.
            1 if frames > 30 => {
                let Some(m) = self.probe_output_mean() else { return };
                if self.eval_tick() != self.mblur_peak {
                    println!(
                        "MBLUR TEST FAIL: the playhead moved off the keyed peak ({} != {})",
                        self.eval_tick(),
                        self.mblur_peak
                    );
                    std::process::exit(1);
                }
                if m <= 1e-4 {
                    println!("MBLUR TEST FAIL: the unblurred frame is black at the peak");
                    std::process::exit(1);
                }
                let mb = self.egraph.add(blur(Default::default()), egui::Pos2::ZERO);
                let consumers: Vec<(u64, usize)> = self
                    .egraph
                    .edges
                    .iter()
                    .filter(|(_, _, s)| *s == GRADE)
                    .map(|(d, i, _)| (*d, *i))
                    .collect();
                for (d, i) in consumers {
                    self.egraph.connect(d, i, mb);
                }
                self.egraph.connect(mb, 0, GRADE);
                self.playback_dirty = true;
                self.test.mblur = Some((2, 0, m.to_bits() as u64));
                return;
            }
            // Below: the Grade is upstream, so it smears. Then move the node
            // ABOVE the Grade.
            2 if frames > 30 => {
                let Some(m) = self.probe_output_mean() else { return };
                let plain = f32::from_bits(bits as u32);
                let ratio = m / plain.max(1e-6);
                // The triangle's mean across the open shutter is 3/4 of its
                // peak. Slack for the sample count and the probe.
                if (ratio - 0.75).abs() > 0.06 {
                    println!(
                        "MBLUR TEST FAIL: upstream gain blurred/unblurred is {ratio:.4} \
                         ({m:.4} / {plain:.4}), wanted 0.75"
                    );
                    std::process::exit(1);
                }
                let Some(mb) = self
                    .egraph
                    .nodes
                    .iter()
                    .find(|(_, n)| matches!(n.kind, NodeKind::MotionBlur(_)))
                    .map(|(id, _)| *id)
                else {
                    println!("MBLUR TEST FAIL: the MotionBlur node went missing");
                    std::process::exit(1);
                };
                // `remove` heals the chain; the new one goes between the
                // Grade and whatever fed it.
                self.egraph.remove(mb);
                let Some(src) = self.egraph.input_of(GRADE, 0) else {
                    println!("MBLUR TEST FAIL: the Grade has no input to blur");
                    std::process::exit(1);
                };
                let mb = self.egraph.add(blur(Default::default()), egui::Pos2::ZERO);
                self.egraph.connect(mb, 0, src);
                self.egraph.connect(GRADE, 0, mb);
                self.playback_dirty = true;
                next = 3;
            }
            // Above: the Grade is downstream and runs once, at the frame.
            3 if frames > 30 => {
                let Some(m) = self.probe_output_mean() else { return };
                let plain = f32::from_bits(bits as u32);
                let ratio = m / plain.max(1e-6);
                if (ratio - 1.0).abs() > 0.02 {
                    println!(
                        "MBLUR TEST FAIL: a gain DOWNSTREAM of the node came out \
                         {ratio:.4} of the unblurred frame, wanted 1.0"
                    );
                    std::process::exit(1);
                }
                println!(
                    "MBLUR TEST PASS: an animated gain upstream of the node averaged \
                     across the shutter (0.75), and the same gain downstream stayed sharp \
                     ({ratio:.3})"
                );
                std::process::exit(0);
            }
            _ => {}
        }
        self.test.mblur = Some((next, if next == stage { frames } else { 0 }, bits));
    }
}
