//! `--test-viewpie`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    /// --test-cpick driver: the colour picker (`colorpicker.rs`) under real
    /// input, on a Grid's line swatch. What only a live run can show: the
    /// swatch opens a popup that takes clicks over the panels, at the default
    /// size; the lch L lane puts the knob under the pointer and lands exactly
    /// that lightness through the comp's working-space matrices; a rate key
    /// held BEFORE the press drags relatively (no jump) at its rate, while
    /// the app's own V hotkey — which would hide the very panel the popup
    /// belongs to — stands down; the old chip restores exactly; dragging the
    /// right edge resizes it with the edge under the pointer and the
    /// swatch-side edges pinned; the size reaches prefs, survives a close and
    /// reopen, and a pinch scales it without zooming what is under it; a
    /// double-click on the corner restores the default; Escape closes.
    /// --test-propspie: pin a Grid, hold the pie key over its "line" row,
    /// release on the top slice and the row's channel is LOCKED; do it again
    /// and it is unlocked. The pie is found by what the rows registered, so
    /// this also proves the registry the pie reads.
    /// --test-viewpie: hold the pie key over the viewer and release on a
    /// slice, four times: 1:1 puts one image pixel on one screen pixel,
    /// green shows the green channel, rgb puts all three back, frame fits
    /// the image again. Picks are found by slice position, so this also
    /// proves the orbit the pie draws is the one it picks from.
    pub(crate) fn drive_test_viewpie(&mut self, raw: &mut egui::RawInput) {
        let Some((stage, mut frames, t0)) = self.test.viewpie else { return };
        frames += 1;
        let fail = |why: String| -> ! {
            println!("VIEWPIE TEST FAIL: {why}");
            std::process::exit(1);
        };
        if t0.elapsed().as_secs() > 60 {
            fail(format!("timeout at stage {stage}"));
        }
        // (slice, what the pick must have done).
        const PICKS: [ViewOp; 4] = [ViewOp::Zoom1, ViewOp::Channel(1), ViewOp::Rgb, ViewOp::Frame];
        let key = |raw: &mut egui::RawInput, down: bool| {
            raw.events.push(egui::Event::Key {
                key: egui::Key::Backtick,
                physical_key: None,
                pressed: down,
                repeat: false,
                modifiers: Default::default(),
            });
        };
        // The pie opens a little off centre, so 1:1 must zoom about it.
        let c = self.viewer_rect.center() + egui::vec2(40.0, 30.0);
        let check = |me: &Self, op: ViewOp| match op {
            ViewOp::Zoom1 => {
                let (ow, oh) = me.comp.output_size().unwrap_or((1920, 1080));
                let r = me.viewer_rect;
                let fit = (r.width() / ow as f32).min(r.height() / oh as f32);
                let px = me.view_zoom * fit * me.egui_ctx.pixels_per_point();
                if (px - 1.0).abs() > 1e-3 {
                    fail(format!("1:1 left {px} screen pixels per image pixel"));
                }
                if me.view_pan == egui::Vec2::ZERO {
                    fail("1:1 did not zoom about the pie's centre".into());
                }
            }
            ViewOp::Channel(ch) if me.view_channel != Some(ch) => {
                fail(format!("the channel pick left {:?}", me.view_channel))
            }
            ViewOp::Rgb if me.view_channel.is_some() || me.view_alpha => {
                fail("rgb did not bring all channels back".into())
            }
            ViewOp::Frame if me.view_zoom != 1.0 || me.view_pan != egui::Vec2::ZERO => {
                fail("frame did not fit the image".into())
            }
            _ => {}
        };
        let mut next = stage;
        let round = (stage.saturating_sub(1) / 4) as usize;
        match stage {
            0 if self.playback.is_some() && self.displayed_total >= 3 && frames > 30 => {
                self.playing = false;
                self.view_alpha = true; // rgb must clear this too
                next = 1;
            }
            s if s >= 1 && (s - 1) % 4 == 0 && frames > 3 => {
                if round > 0 {
                    check(self, PICKS[round - 1]);
                }
                if round == PICKS.len() {
                    println!(
                        "VIEWPIE TEST PASS: the viewer pie opened on backtick and 1:1, green, \
                         rgb and frame each did what they say"
                    );
                    std::process::exit(0);
                }
                raw.events.push(egui::Event::PointerMoved(c));
                next = stage + 1;
            }
            s if s >= 1 && (s - 1) % 4 == 1 && frames > 3 => {
                key(raw, true);
                next = stage + 1;
            }
            s if s >= 1 && (s - 1) % 4 == 2 && frames > 3 => {
                if self.viewer_pie.is_none() {
                    fail("holding the pie key over the viewer opened no pie".into());
                }
                let i = VIEWER_PIE.iter().position(|(op, _)| *op == PICKS[round]).unwrap();
                let at = regions::pie_slice_pos(c, i, VIEWER_PIE.len(), 62.0);
                raw.events.push(egui::Event::PointerMoved(at));
                next = stage + 1;
            }
            s if s >= 1 && (s - 1) % 4 == 3 && frames > 3 => {
                key(raw, false);
                next = stage + 1;
            }
            _ => {}
        }
        self.test.viewpie = Some((next, if next == stage { frames } else { 0 }, t0));
    }
}
