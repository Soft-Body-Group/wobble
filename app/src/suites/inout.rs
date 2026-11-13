//! `--test-inout`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    /// --test-inout: ctrl-click and alt-click the timeline track and check
    /// they set the playback in and out points at the frames clicked.
    pub(crate) fn drive_test_inout(&mut self, raw: &mut egui::RawInput) {
        let Some((stage, mut frames)) = self.test_inout else { return };
        frames += 1;
        // When the current stage began, for the waits that need real time.
        static SINCE: std::sync::Mutex<Option<(u32, web_time::Instant)>> = std::sync::Mutex::new(None);
        let since = {
            let mut g = SINCE.lock().unwrap();
            match *g {
                Some((st, t)) if st == stage => t,
                _ => {
                    let t = web_time::Instant::now();
                    *g = Some((stage, t));
                    t
                }
            }
        };
        // A frame count is not a duration under test: time out on the clock.
        if since.elapsed().as_secs() > 20 {
            println!("IN/OUT TEST FAIL: timeout at stage {stage}");
            std::process::exit(1);
        }
        // A click with modifiers: egui reads them off the event itself.
        let click = |raw: &mut egui::RawInput, pos: egui::Pos2, m: egui::Modifiers, down: bool| {
            raw.events.insert(0, egui::Event::ModifiersChanged(m));
            raw.events.push(egui::Event::PointerMoved(pos));
            raw.events.push(egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed: down,
                modifiers: m,
            });
        };
        // Where a frame sits on the painted track.
        // The track spans what plays (`track_span`), so it re-maps when
        // in/out moves: aim off the span it was LAST painted with.
        let x_of = |s: &Self, f: u64| {
            let (ta, tb) = s.track_span;
            let total = (tb - ta + 1).max(1) as f32;
            let r = s.track_rect;
            r.min.x + ((f as i64 - ta) as f32 / total) * r.width() + r.width() / total * 0.5
        };
        let mut next = stage;
        match stage {
            0 if frames >= 6 && self.track_rect.width() > 20.0 => {
                // A known VIEWER range to aim inside of — pinned, so the
                // track cannot re-map under the clicks while plates load.
                self.view_range = Some((0, 23));
                next = 1;
            }
            1 if frames >= 4 => {
                let y = self.track_rect.center().y;
                click(raw, egui::pos2(x_of(self, 6), y), egui::Modifiers::CTRL, true);
                next = 2;
            }
            2 => {
                let y = self.track_rect.center().y;
                click(raw, egui::pos2(x_of(self, 6), y), egui::Modifiers::CTRL, false);
                next = 3;
            }
            // The track re-maps once in is set, and two clicks close in time
            // would read as a double-click (which clears in/out): let it
            // settle on the wall clock first.
            3 if since.elapsed().as_millis() > 500 => {
                let y = self.track_rect.center().y;
                click(raw, egui::pos2(x_of(self, 17), y), egui::Modifiers::ALT, true);
                next = 4;
            }
            4 => {
                let y = self.track_rect.center().y;
                click(raw, egui::pos2(x_of(self, 17), y), egui::Modifiers::ALT, false);
                next = 5;
            }
            5 if frames >= 4 => {
                raw.events.insert(0, egui::Event::ModifiersChanged(egui::Modifiers::NONE));
                let got = self.play_range;
                // Landing within a frame of the click is the honest bar: the
                // track maps a whole shot into a few hundred pixels.
                let ok = got.is_some_and(|(i, o)| {
                    i.abs_diff(6) <= 1 && o.abs_diff(17) <= 1
                });
                if !ok {
                    println!("IN/OUT TEST FAIL: ctrl/alt clicks gave {got:?}, wanted ~(6, 17)");
                    std::process::exit(1);
                }
                // And the play span is what playback actually loops over.
                let span = self.play_span();
                if span != got.unwrap() {
                    println!("IN/OUT TEST FAIL: play_span {span:?} != in/out {got:?}");
                    std::process::exit(1);
                }
                // Each loop mode must select the band it is named after.
                self.handles = (3, 4);
                let (in_pt, out_pt) = got.unwrap();
                let checks = [
                    (LoopSpan::InOut, (in_pt, out_pt)),
                    (LoopSpan::Cut, (0, 23)),
                    (LoopSpan::Handles, (0, 27)), // head clamps at frame 0
                ];
                for (mode, want) in checks {
                    self.loop_span = mode;
                    let got = self.play_span();
                    if got != want {
                        println!(
                            "IN/OUT TEST FAIL: loop {} spans {got:?}, wanted {want:?}",
                            mode.label()
                        );
                        std::process::exit(1);
                    }
                }
                self.loop_span = LoopSpan::InOut;
                self.handles = (0, 0);
                // The track spans the whole range, in/out drawn on it, not
                // stretched to it.
                if self.track_span != self.shot_span_handles_i() {
                    println!(
                        "IN/OUT TEST FAIL: the track spans {:?}, not the whole range {:?} (in/out {:?})",
                        self.track_span,
                        self.shot_span_handles_i(),
                        (in_pt, out_pt)
                    );
                    std::process::exit(1);
                }
                self.playing = false;
                self.seek_to(10);
                next = 6;
            }
            // I over the timeline: the playhead becomes the in point.
            6 | 7 if since.elapsed().as_millis() > 500 => {
                let p = self.track_rect.center();
                raw.events.push(egui::Event::PointerMoved(p));
                raw.events.push(egui::Event::Key {
                    key: egui::Key::I,
                    physical_key: None,
                    pressed: stage == 6,
                    repeat: false,
                    modifiers: egui::Modifiers::NONE,
                });
                next = stage + 1;
            }
            8 if frames >= 4 => {
                let got = self.play_range;
                if got.map(|r| r.0) != Some(10) {
                    println!("IN/OUT TEST FAIL: I at frame 10 gave in/out {got:?} (pos {} target {:?} playing {})", self.timeline_pos, self.seek_target, self.playing);
                    std::process::exit(1);
                }
                if self.track_span != self.shot_span_handles_i() {
                    println!("IN/OUT TEST FAIL: after I the track spans {:?}", self.track_span);
                    std::process::exit(1);
                }
                // Now look straight at the Load: the track spans its whole
                // clip and the in/out is the Load's own.
                self.egraph.view = Some(0);
                // A pinned viewer range is the user's: it wins over the clip.
                self.view_range = None;
                next = 9;
            }
            9 if since.elapsed().as_millis() > 300 => {
                let Some(v) = self.viewed_load() else {
                    println!("IN/OUT TEST FAIL: viewing the Load is not a viewed Load (view {:?} active {:?} range {:?} loaded {:?})", self.egraph.view, self.egraph.active_view(), self.view_range, self.loaded.keys().collect::<Vec<_>>());
                    std::process::exit(1);
                };
                if self.track_span != v.full {
                    println!("IN/OUT TEST FAIL: viewing the Load the track spans {:?}, not its clip {:?}", self.track_span, v.full);
                    std::process::exit(1);
                }
                let want = Some((v.inout.0 as u64, v.inout.1 as u64));
                if self.play_range != want {
                    println!("IN/OUT TEST FAIL: the timeline's in/out {:?} is not the Load's {want:?}", self.play_range);
                    std::process::exit(1);
                }
                // An in/out set on the timeline lands on the Load.
                self.play_range = Some((v.full.0 as u64 + 3, v.full.0 as u64 + 9));
                next = 10;
            }
            10 if since.elapsed().as_millis() > 300 => {
                let got = self.viewed_load().map(|v| v.inout);
                let a = self.viewed_load().map(|v| v.full.0).unwrap_or(0);
                if got != Some((a + 3, a + 9)) {
                    println!("IN/OUT TEST FAIL: the timeline's in/out did not reach the Load: {got:?}");
                    std::process::exit(1);
                }
                // The pie's in–out pick: the bar glides onto the in/out.
                self.track_move = self.track_shown.map(|s| (s, web_time::Instant::now()));
                self.track_zoom = TrackZoom::InOut;
                next = 11;
            }
            11 if since.elapsed().as_millis() > 500 => {
                let want = self.play_range.map(|(a, b)| (a as f32, b as f32));
                if self.track_shown != want || self.track_move.is_some() {
                    println!(
                        "IN/OUT TEST FAIL: zoomed to in–out the bar shows {:?}, not {want:?}",
                        self.track_shown
                    );
                    std::process::exit(1);
                }
                self.track_zoom = TrackZoom::Node;
                next = 12;
            }
            // Drag the IN mark three frames right: it is a handle now.
            12 if since.elapsed().as_millis() > 500 => {
                let Some((pa, _)) = self.play_range else {
                    println!("IN/OUT TEST FAIL: no in/out to drag");
                    std::process::exit(1);
                };
                let half = self.track_rect.width() / (self.track_span.1 - self.track_span.0 + 1).max(1) as f32 * 0.5;
                let y = self.track_rect.center().y;
                click(raw, egui::pos2(x_of(self, pa) - half + 1.0, y), egui::Modifiers::NONE, true);
                self.test_inout_mark = Some(pa);
                next = 13;
            }
            13 if frames >= 3 => {
                let pa = self.test_inout_mark.unwrap_or(0);
                let half = self.track_rect.width() / (self.track_span.1 - self.track_span.0 + 1).max(1) as f32 * 0.5;
                raw.events.push(egui::Event::PointerMoved(egui::pos2(
                    x_of(self, pa + 3) - half + 1.0,
                    self.track_rect.center().y,
                )));
                next = 14;
            }
            14 if frames >= 3 => {
                let pa = self.test_inout_mark.unwrap_or(0);
                let half = self.track_rect.width() / (self.track_span.1 - self.track_span.0 + 1).max(1) as f32 * 0.5;
                let y = self.track_rect.center().y;
                click(raw, egui::pos2(x_of(self, pa + 3) - half + 1.0, y), egui::Modifiers::NONE, false);
                next = 15;
            }
            15 if frames >= 4 => {
                let pa = self.test_inout_mark.unwrap_or(0);
                if self.play_range.map(|r| r.0) != Some(pa + 3) {
                    println!(
                        "IN/OUT TEST FAIL: dragging the in mark from {pa} gave in/out {:?}, not in {}",
                        self.play_range,
                        pa + 3
                    );
                    std::process::exit(1);
                }
                next = 16;
            }
            // A click on the playhead block types a frame: Enter applies it.
            16 if since.elapsed().as_millis() > 300 => {
                let y = self.track_rect.center().y;
                click(raw, egui::pos2(x_of(self, self.timeline_pos), y), egui::Modifiers::NONE, true);
                next = 17;
            }
            17 if frames >= 2 => {
                let y = self.track_rect.center().y;
                click(raw, egui::pos2(x_of(self, self.timeline_pos), y), egui::Modifiers::NONE, false);
                next = 18;
            }
            18 if frames >= 4 => {
                match self.track_edit.as_mut() {
                    Some((TrackEdit::Playhead, text, _, _)) => *text = "12".into(),
                    other => {
                        println!("IN/OUT TEST FAIL: a click on the playhead opened {other:?}");
                        std::process::exit(1);
                    }
                }
                raw.events.push(egui::Event::Key {
                    key: egui::Key::Enter,
                    physical_key: None,
                    pressed: true,
                    repeat: false,
                    modifiers: egui::Modifiers::NONE,
                });
                next = 19;
            }
            19 if since.elapsed().as_millis() > 400 => {
                let at = self.seek_target.unwrap_or(self.timeline_pos);
                if at != 12 || self.track_edit.is_some() {
                    println!(
                        "IN/OUT TEST FAIL: typing 12 on the playhead went to {at} (field open {})",
                        self.track_edit.is_some()
                    );
                    std::process::exit(1);
                }
                println!(
                    "IN/OUT TEST PASS: ctrl-click set in, alt-click set out, the track spans \
                     the whole range with in/out on it, I at the playhead set in; loop modes select in/out, cut and \
                     handles; viewing a Load the track spans its clip and its in/out is the \
                     Load's, both ways; the pie's in–out zoom glides the bar onto it; the in \
                     mark drags; a frame typed on the playhead lands"
                );
                std::process::exit(0);
            }
            _ => {}
        }
        if next != stage {
            frames = 0;
        }
        self.test_inout = Some((next, frames));
    }
}
