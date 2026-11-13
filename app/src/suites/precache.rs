//! `--test-precache`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    /// Precache through the graph menu's ask: the Read's RAM toggle turns
    /// on, the run restarts on the rebuilt rig, and when it finishes EVERY
    /// frame of the Read is in the RAM tier with nothing played; then a
    /// second run is cancelled and must actually stop. Wall-clock waits.
    pub(crate) fn drive_test_precache(&mut self) {
        let Some((stage, t0)) = self.test_precache else { return };
        let fail = |m: String| -> ! {
            println!("PRECACHE TEST FAIL: {m}");
            std::process::exit(1);
        };
        if t0.elapsed().as_secs_f32() > 30.0 {
            fail(format!("timeout at stage {stage}"));
        }
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
            0 if self.playback.is_some() && self.displayed_total >= 1 => {
                self.playing = false;
                let before = residency(self).map(|r| r.0).unwrap_or(0);
                if before > 2 {
                    fail(format!("{before} frames resident before the precache"));
                }
                self.node_editor.precache_req = Some(vec![0]);
                self.test_precache = Some((1, Instant::now()));
            }
            1 => {
                let Some(run) = &self.precache else { fail("no run".into()) };
                let Some(job) = run.job() else { return };
                if !job.finished() || run.rig() != self.rig_builds || self.playback_dirty {
                    return;
                }
                if !matches!(
                    self.egraph.node(0).map(|n| &n.kind),
                    Some(NodeKind::Read { ram_cache: true, .. })
                ) {
                    fail("the Read's RAM toggle is still off".into());
                }
                let Some((n, len)) = residency(self) else { fail("no residency".into()) };
                if n != len || job.done() != job.total() || job.total() < len {
                    fail(format!("{n}/{len} resident, job {}/{}", job.done(), job.total()));
                }
                println!("precache: {n}/{len} frames resident, {} displayed", self.displayed_total);
                // Second run, every Read, cancelled at once.
                let reads: Vec<u64> = self.egraph.nodes.keys().copied().collect();
                self.precache_start(reads);
                self.test_precache = Some((2, Instant::now()));
            }
            2 => {
                let Some(run) = &self.precache else { fail("no second run".into()) };
                let Some(job) = run.job() else { return };
                job.stop();
                self.test_precache = Some((3, Instant::now()));
            }
            3 => {
                let Some(job) = self.precache.as_ref().and_then(|r| r.job()) else {
                    fail("run vanished".into())
                };
                if !job.finished() {
                    if t0.elapsed().as_secs_f32() > 5.0 {
                        fail("cancel did not stop the workers".into());
                    }
                    return;
                }
                println!(
                    "PRECACHE TEST PASS: every frame resident via the menu ask; cancel stopped at {}/{}",
                    job.done(),
                    job.total()
                );
                std::process::exit(0);
            }
            _ => {}
        }
    }
}
