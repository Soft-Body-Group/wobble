//! `--test-frames`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    /// A PLACED read is the whole frame-numbering convention in one test:
    /// files numbered from 1011 occupy comp frames 1011.., and the playhead,
    /// the readout, the reader's own source frame, animation and the in/out
    /// loop must all agree about that. Before this, the transport drew the
    /// playhead at `comp_range.0 + tick % media_len` — off the left edge of
    /// a 1011-based track — and footage playback ignored in/out entirely.
    pub(crate) fn drive_test_frames(&mut self, raw: &mut egui::RawInput) {
        let Some((stage, mut frames)) = self.test.frames else { return };
        frames += 1;
        let elapsed = self.test.frames_t0.get_or_insert_with(Instant::now).elapsed();
        if elapsed > web_time::Duration::from_secs(60) {
            println!(
                "FRAMES TEST FAIL: timeout at stage {stage} (playhead {} shown {} chasing {} \
                 playing {} span {:?} loop {:?} seen {:?} player {:?} pressure {:?})",
                self.timeline_pos,
                self.shown_tick,
                self.force_display,
                self.playing,
                self.play_span(),
                self.playback.as_ref().and_then(|pb| pb.player.loop_span()),
                self.test.frames_seen,
                self.playback.as_ref().map(|pb| pb.player.debug_state()),
                self.playback.as_ref().map(|pb| pb.player.debug_pressure()),
            );
            std::process::exit(1);
        }
        const PLACE: u64 = 1011;
        let media = self.loaded.get(&0).map(|l| l.seq.frames.len() as u64).unwrap_or(0);
        // A key TAP: down and up (a press never released arrives as a
        // repeat next time).
        let tap = |raw: &mut egui::RawInput, k: egui::Key| {
            for pressed in [true, false] {
                raw.events.push(egui::Event::Key {
                    key: k,
                    physical_key: None,
                    pressed,
                    repeat: false,
                    modifiers: egui::Modifiers::NONE,
                });
            }
        };
        let mut next = stage;
        match stage {
            0 if self.playback.is_some() && self.displayed_total >= 3 && media > 8 => {
                self.place_read(0, PLACE as i64);
                // View the placed Read itself: the case the user hit.
                self.egraph.view = Some(0);
                self.comp_range = None;
                self.handles = (0, 0);
                self.loop_span = LoopSpan::Cut;
                self.play_range = None;
                self.playing = false;
                self.playback_dirty = true;
                next = 1;
            }
            1 if frames >= 8 => {
                let want = (PLACE, PLACE + media - 1);
                if self.shot_span() != want {
                    println!(
                        "FRAMES TEST FAIL: shot span {:?}, wanted {want:?}",
                        self.shot_span()
                    );
                    std::process::exit(1);
                }
                self.seek_to(PLACE + 9);
                next = 2;
            }
            2 if !self.force_display && self.eval_tick() == PLACE + 9 => {
                // The playhead is drawn where the frame is.
                let (ha, hb) = self.shot_span_handles_i();
                let total = (hb - ha + 1).max(1) as f32;
                let r = self.track_rect;
                let px = r.min.x + ((self.timeline_pos as i64 - ha) as f32 / total) * r.width();
                if r.width() > 20.0 && !(r.min.x..=r.max.x).contains(&px) {
                    println!(
                        "FRAMES TEST FAIL: playhead at x {px:.0} is outside the track \
                         ({:.0}..{:.0}) at frame {}",
                        r.min.x,
                        r.max.x,
                        self.timeline_pos
                    );
                    std::process::exit(1);
                }
                // The reader is on ITS frame 9, because the plate is placed.
                let src = self.playback.as_ref().and_then(|pb| {
                    let i = pb.slots.iter().position(|s| *s == 0)?;
                    Some(pb.player.source_frame(i, self.timeline_pos))
                });
                if src != Some(9) {
                    println!(
                        "FRAMES TEST FAIL: comp frame {} reads source frame {src:?}, wanted 9",
                        self.timeline_pos
                    );
                    std::process::exit(1);
                }
                // Arrow keys next: playing, so the step must also pause.
                self.playing = true;
                tap(raw, egui::Key::ArrowRight);
                frames = 0;
                next = 10;
            }
            // → stepped one frame on from where the hand was (the playhead
            // may have played on a frame or two before the tap landed:
            // what is pinned is "paused, and on the frame it stepped to").
            10 if frames >= 3 => {
                if self.playing {
                    println!("FRAMES TEST FAIL: → did not pause playback");
                    std::process::exit(1);
                }
                let at = self.timeline_pos;
                self.test.frames_seen = vec![at];
                tap(raw, egui::Key::ArrowLeft);
                frames = 0;
                next = 11;
            }
            11 if frames >= 3 => {
                let was = self.test.frames_seen[0];
                if self.timeline_pos + 1 != was {
                    println!(
                        "FRAMES TEST FAIL: ← went from {was} to {}, wanted {}",
                        self.timeline_pos,
                        was - 1
                    );
                    std::process::exit(1);
                }
                tap(raw, egui::Key::ArrowRight);
                frames = 0;
                next = 12;
            }
            12 if frames >= 3 => {
                let was = self.test.frames_seen[0];
                if self.timeline_pos != was {
                    println!("FRAMES TEST FAIL: → went to {}, wanted {was}", self.timeline_pos);
                    std::process::exit(1);
                }
                let (_, b) = self.shot_span();
                self.seek_to(b);
                frames = 0;
                next = 13;
            }
            13 if frames >= 3 => {
                tap(raw, egui::Key::ArrowRight);
                frames = 0;
                next = 14;
            }
            14 if frames >= 3 => {
                // At the end of the range → holds; it does not wrap.
                let (_, b) = self.shot_span();
                if self.timeline_pos != b {
                    println!(
                        "FRAMES TEST FAIL: → at the last frame {b} went to {}",
                        self.timeline_pos
                    );
                    std::process::exit(1);
                }
                // Now play a short in/out span and watch it loop THERE.
                self.play_range = Some((PLACE + 3, PLACE + 6));
                self.loop_span = LoopSpan::InOut;
                self.seek_to(PLACE + 3);
                self.playing = true;
                self.test.frames_seen.clear();
                next = 3;
            }
            3 => {
                // Only SETTLED frames: while a seek is still landing the
                // viewer deliberately shows the nearest decoded frame it
                // has, and the playhead follows the picture — that is the
                // scrub rule, not a playback frame.
                if self.force_display {
                    self.test.frames = Some((stage, frames));
                    return;
                }
                let t = self.timeline_pos;
                if self.test.frames_seen.last() != Some(&t) {
                    self.test.frames_seen.push(t);
                }
                let seen = &self.test.frames_seen;
                if seen.len() >= 10 {
                    let (lo, hi) = (PLACE + 3, PLACE + 6);
                    if let Some(bad) = seen.iter().find(|f| **f < lo || **f > hi) {
                        println!(
                            "FRAMES TEST FAIL: playback left the in/out span at frame {bad} \
                             (in {lo}, out {hi}) — saw {seen:?}"
                        );
                        std::process::exit(1);
                    }
                    let looped = seen.windows(2).any(|w| w[1] < w[0]);
                    if !looped {
                        println!("FRAMES TEST FAIL: playback never looped: {seen:?}");
                        std::process::exit(1);
                    }
                    // Animation evaluates at the frame on screen, not at a
                    // free-running counter.
                    if self.eval_tick() != self.shown_tick {
                        println!(
                            "FRAMES TEST FAIL: eval {} != shown {}",
                            self.eval_tick(),
                            self.shown_tick
                        );
                        std::process::exit(1);
                    }
                    println!(
                        "FRAMES TEST PASS: read placed at {PLACE} spans {:?}, playhead on the \
                         track, comp {} = source 9, ←/→ step ±1 (pausing, holding at the end), \
                         in/out loop played {:?}",
                        self.shot_span(),
                        PLACE + 9,
                        seen
                    );
                    std::process::exit(0);
                }
            }
            _ => {}
        }
        self.test.frames = Some((next, frames));
    }
}
