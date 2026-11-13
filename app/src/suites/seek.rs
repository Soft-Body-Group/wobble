//! `--test-seek`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    /// --test-seek driver. Seek settles when force_display clears (the seek's
    /// landing frame actually displayed).
    pub(crate) fn drive_test_seek(&mut self) {
        let Some((stage, frames, mark)) = &mut self.test.seek else { return };
        *frames += 1;
        let t0 = *self.test.seek_t0.get_or_insert_with(web_time::Instant::now);
        if t0.elapsed().as_secs_f32() > 30.0 {
            println!(
                "SEEK TEST FAIL: timed out at stage {stage} (pos {}, ring free {:?}, state {:?}, decoded {:?})",
                self.timeline_pos,
                self.playback.as_ref().map(|pb| pb.ring.free_count()),
                self.playback.as_ref().map(|pb| pb.player.debug_state()),
                self.playback.as_ref().map(|pb| pb.player.stats.frames_decoded),
            );
            std::process::exit(1);
        }
        let (stage, mark) = (*stage, *mark);
        let landed = !self.force_display;
        match stage {
            0 if self.displayed_total >= 3 => {
                self.playing = false;
                self.seek_to(17);
                self.test.seek = Some((1, 0, 0));
            }
            1 if landed && self.timeline_pos == 17 => {
                self.seek_to(5); // backward: pending + in-flight all stale
                self.test.seek = Some((2, 0, 0));
            }
            2 if landed && self.timeline_pos == 5 => {
                self.seek_to(20); // rapid double-seek: overridden next frame
                self.test.seek = Some((3, 0, 0));
            }
            3 => {
                self.seek_to(9);
                self.test.seek = Some((4, 0, 0));
            }
            4 if landed && self.timeline_pos == 9 => {
                self.playing = true;
                self.test.seek = Some((5, 0, self.displayed_total));
            }
            5 if self.displayed_total >= mark + 24 => {
                let cache = self
                    .playback
                    .as_ref()
                    .map(|pb| pb.player.cache_stats())
                    .unwrap_or_default();
                println!(
                    "SEEK TEST PASS: paused seeks 17/5/20->9 landed, then played {} frames (skipped {}, ring free {:?}, cache {:.2} GB/{} frames, {} hits)",
                    self.displayed_total - mark,
                    self.skipped,
                    self.playback.as_ref().map(|pb| pb.ring.free_count()),
                    cache.0 as f64 / 1e9,
                    cache.1,
                    cache.2,
                );
                std::process::exit(0);
            }
            _ => {}
        }
    }
}
