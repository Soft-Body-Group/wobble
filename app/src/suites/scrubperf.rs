//! `--test-scrubperf`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    /// How many frames a CONTINUOUS scrub actually puts on screen.
    ///
    /// Playback and scrubbing use completely different paths — playback
    /// pipelines decodes ahead of the playhead, a scrub re-targets it every
    /// UI frame — so "playback is smooth" says nothing about scrubbing. This
    /// drags the playhead at a fixed rate for a fixed WALL time (the loop
    /// spins far faster than real time under test, so iterations are not a
    /// duration) and reports frames displayed per second. It is a
    /// measurement, not a pass/fail threshold: the number is the point.
    pub(crate) fn drive_test_scrubperf(&mut self) {
        let Some((stage, shown0, pos)) = self.test_scrubperf else { return };
        let t0 = *self.test_scrub_t0.get_or_insert_with(web_time::Instant::now);
        let el = t0.elapsed().as_secs_f32();
        if el > 30.0 {
            println!("SCRUBPERF TEST FAIL: timeout at stage {stage}");
            std::process::exit(1);
        }
        match stage {
            0 if self.playback.is_some() && self.displayed_total >= 3 => {
                self.playing = false;
                self.seek_to(0);
                self.test_scrubperf = Some((1, 0, 0));
            }
            // Settle, then start the clock.
            1 if el > 1.0 => {
                self.test_scrubperf = Some((2, self.displayed_total, 0));
                self.test_scrub_t0 = Some(web_time::Instant::now());
            }
            // Drag: one frame per UI frame, exactly as the viewer's
            // space-drag does when swept across the whole timeline.
            2 => {
                let len = self
                    .playback
                    .as_ref()
                    .map(|pb| pb.player.timeline_len())
                    .unwrap_or(1);
                if el < 3.0 {
                    // Pace the drag at 60Hz on the WALL clock. Under test
                    // the loop spins thousands of times a second; stepping
                    // per iteration would re-target the playhead far faster
                    // than any hand ever could and measure a fantasy.
                    let want = (el * 60.0) as u64 % len.max(1);
                    if want != pos {
                        self.seek_to(want);
                    }
                    self.test_scrubperf = Some((2, shown0, want));
                    return;
                }
                let shown = self.displayed_total - shown0;
                let decoded = self
                    .playback
                    .as_ref()
                    .map(|pb| pb.player.stats.frames_decoded)
                    .unwrap_or(0);
                let fps = shown as f32 / el;
                // The floor is deliberately far below what it measures
                // (~53fps here): this is guarding against the REGRESSION,
                // which is total — a seek that discards work in flight
                // displays literally nothing during a drag, because a
                // 27ms decode can never outrun a 60Hz re-target. Anything
                // in double figures means work is surviving the seek.
                if fps < 15.0 {
                    println!(
                        "SCRUBPERF TEST FAIL: only {shown} frames displayed in {el:.1}s \
                         = {fps:.1} fps while scrubbing ({decoded} decodes issued) — \
                         a seek is throwing away work it cannot replace in time"
                    );
                    std::process::exit(1);
                }
                println!(
                    "SCRUBPERF TEST PASS: {shown} frames displayed in {el:.1}s = \
                     {fps:.1} fps while scrubbing ({decoded} decodes issued, {} readers)",
                    self.playback.as_ref().map(|pb| pb.player.sequences().len()).unwrap_or(0),
                );
                std::process::exit(0);
            }
            _ => {}
        }
    }
}
