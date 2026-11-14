//! `--test-play`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    /// --test-play driver: warm the cache with two loops, then measure.
    pub(crate) fn drive_test_play(&mut self) {
        let Some((stage, mark, started)) = self.test.play else { return };
        if started.elapsed().as_secs() > 120 {
            println!("PLAY TEST FAIL: timed out at stage {stage}");
            std::process::exit(1);
        }
        match stage {
            // The play suite measures the RAM-warm pipeline, and the RAM
            // tier is opt-in now: enable it on both reads, rebuild, and let
            // one loop pass warm the cache before measuring.
            0 if self.displayed_total >= 24 => {
                for id in [0u64, 1] {
                    if let Some(editor::ENode {
                        kind: NodeKind::Read { ram_cache, .. }, ..
                    }) = self.egraph.node_mut(id)
                    {
                        *ram_cache = true;
                    }
                }
                self.playback_dirty = true;
                self.test.play = Some((10, self.displayed_total, Instant::now()));
            }
            10 if self.displayed_total >= mark + 30 => {
                self.fps = 120.0;
                self.skipped = 0;
                self.test.play = Some((1, self.displayed_total, Instant::now()));
            }
            1 if self.displayed_total >= mark + 96 => {
                let secs = started.elapsed().as_secs_f32();
                let displayed_fps = 96.0 / secs;
                // Skipped frames were fetched and paced too — the pipeline
                // delivered them; the display just can't show them all.
                let pipeline_fps = (96.0 + self.skipped as f32) / secs;
                let cache = self
                    .playback
                    .as_ref()
                    .map(|pb| pb.player.cache_stats())
                    .unwrap_or_default();
                let pass = pipeline_fps > 100.0;
                println!(
                    "PLAY TEST {}: pipeline {pipeline_fps:.1} sets/s at 120 target, displayed {displayed_fps:.1} fps (vsync-bound), {} skipped, cache {} frames / {} hits",
                    if pass { "PASS" } else { "FAIL" },
                    self.skipped,
                    cache.1,
                    cache.2,
                );
                std::process::exit(if pass { 0 } else { 1 });
            }
            _ => {}
        }
    }
}
