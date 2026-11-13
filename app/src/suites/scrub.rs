//! `--test-scrub`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    /// The browser's preview transport: playing a selected sequence walks
    /// it, and the inspector actually SHOWS the frames it walks past.
    ///
    /// "Distinct frames decoded" is the assertion that matters. A transport
    /// whose position advances while the picture never changes looks fine
    /// in a screenshot and is completely broken.
    pub(crate) fn drive_test_scrub(&mut self) {
        let Some((stage, mut frames, len)) = self.test_scrub else { return };
        frames += 1;
        let t0 = *self.test_scrub_t0.get_or_insert_with(web_time::Instant::now);
        if t0.elapsed().as_secs_f32() > 20.0 {
            println!("SCRUB TEST FAIL: timeout at stage {stage}");
            std::process::exit(1);
        }
        match stage {
            0 if self.playback.is_some() && self.displayed_total >= 2 => {
                self.browser = Some(browser::SeqBrowser::new(
                    None,
                    std::path::PathBuf::from("data/2k_zip_bg"),
                    self.decoder.clone(),
                ));
                self.test_scrub = Some((1, frames, 0));
            }
            // Select the sequence and read how many frames it offers.
            1 if frames > 60 => {
                let Some(b) = &mut self.browser else {
                    println!("SCRUB TEST FAIL: browser closed");
                    std::process::exit(1);
                };
                let (len, playing) = b.test_select_and_len(0);
                if len < 4 {
                    println!("SCRUB TEST FAIL: sequence offers {len} frames, want >= 4");
                    std::process::exit(1);
                }
                if playing {
                    println!("SCRUB TEST FAIL: preview running before it was asked for");
                    std::process::exit(1);
                }
                b.test_play(len);
                self.test_scrub = Some((2, frames, len));
            }
            // Poll until the transport has moved off frame 0, THEN check
            // the picture moved with it. Sampling one instant instead would
            // be a coin flip: playback loops, so "frame 0" is a position it
            // legitimately returns to.
            2 => {
                let Some(b) = &mut self.browser else {
                    println!("SCRUB TEST FAIL: browser closed mid-play");
                    std::process::exit(1);
                };
                let (at, playing) = b.test_preview_state(len);
                if !playing {
                    println!("SCRUB TEST FAIL: transport stopped on its own");
                    std::process::exit(1);
                }
                let seen = b.test_thumb_frames_seen();
                // Both have to happen, and they happen at different times:
                // the position moves on a 24fps clock, the picture follows
                // as decodes land. Give them a second of REAL time and then
                // say which one failed to show up.
                if t0.elapsed().as_secs_f32() < 1.5 || (at > 0 && seen < 3) {
                    self.test_scrub = Some((2, frames, len));
                    return;
                }
                if at == 0 {
                    println!(
                        "SCRUB TEST FAIL: after {:.1}s the transport is still on frame \
                         0 of {len}",
                        t0.elapsed().as_secs_f32()
                    );
                    std::process::exit(1);
                }
                if seen < 3 {
                    println!(
                        "SCRUB TEST FAIL: transport reached frame {at} but only {seen} \
                         distinct frame(s) ever decoded — the position moves and the \
                         picture does not"
                    );
                    std::process::exit(1);
                }
                // Pausing must actually stop it.
                b.test_select_and_len(0);
                let (_, still_playing) = b.test_preview_state(len);
                if still_playing {
                    println!("SCRUB TEST FAIL: re-selecting did not stop the transport");
                    std::process::exit(1);
                }
                println!(
                    "SCRUB TEST PASS: preview walked a {len}-frame sequence to frame \
                     {at}, decoding {seen} distinct frames; re-selecting stopped it"
                );
                std::process::exit(0);
            }
            _ => self.test_scrub = Some((stage, frames, len)),
        }
    }
}
