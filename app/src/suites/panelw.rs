//! `--test-panelw`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    /// Click panel 0's NAME: it selects. Drag its right edge 100px: it
    /// alone widens by 100px, panel 1 keeps its width. Double-click the
    /// edge: back to the shared width. Real input throughout.
    pub(crate) fn drive_test_panelw(&mut self, raw: &mut egui::RawInput) {
        let Some(mut t) = self.test_panelw.take() else { return };
        t.frames += 1;
        let fail = |m: String| -> ! {
            println!("PANEL WIDTH TEST FAIL: {m}");
            std::process::exit(1);
        };
        if t.start.is_some_and(|s| s.elapsed().as_secs_f32() > 20.0) {
            fail(format!("timeout at stage {}", t.stage));
        }
        let btn = |raw: &mut egui::RawInput, pos: egui::Pos2, down: bool| {
            raw.events.push(egui::Event::PointerMoved(pos));
            raw.events.push(egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed: down,
                modifiers: Default::default(),
            });
        };
        let rect = |s: &Self, n: u64| s.prop_panel_rects.iter().find(|(i, _)| *i == n).map(|(_, r)| *r);
        let next = |t: &mut PanelWTest| {
            t.stage += 1;
            t.frames = 0;
        };
        match t.stage {
            0 if t.frames > 12 && self.prop_panel_rects.len() >= 2 => {
                let (a, ra) = self.prop_panel_rects[0];
                let (b, rb) = self.prop_panel_rects[1];
                (t.a, t.b, t.w_a, t.w_b, t.ra) = (a, b, ra.width(), rb.width(), Some(ra));
                btn(raw, ra.min + egui::vec2(32.0, 15.0), true);
                next(&mut t);
            }
            1 => {
                btn(raw, t.ra.unwrap().min + egui::vec2(32.0, 15.0), false);
                next(&mut t);
            }
            2 if t.frames > 3 => {
                if self.prop_selected != Some(t.a) {
                    fail(format!("a click on the name selected {:?}, want {}", self.prop_selected, t.a));
                }
                let ra = t.ra.unwrap();
                t.edge = egui::pos2(ra.max.x - 1.0, ra.center().y);
                btn(raw, t.edge, true);
                next(&mut t);
            }
            3..=12 => {
                // Anchored to the press, ten steps to +100px.
                let k = (t.stage - 2) as f32;
                raw.events.push(egui::Event::PointerMoved(t.edge + egui::vec2(10.0 * k, 0.0)));
                next(&mut t);
            }
            13 => {
                btn(raw, t.edge + egui::vec2(100.0, 0.0), false);
                t.since = Some(web_time::Instant::now());
                next(&mut t);
            }
            // Past egui's triple-click window (2 x 0.3s from the name
            // click), or the double-click below counts as a third.
            14 if t.since.is_some_and(|s| s.elapsed().as_secs_f32() > 0.8) => {
                let (Some(ra), Some(rb)) = (rect(self, t.a), rect(self, t.b)) else {
                    fail("a panel went missing".into())
                };
                if (ra.width() - (t.w_a + 100.0)).abs() > 2.0 {
                    fail(format!("dragged panel {:.0} -> {:.0}, want +100", t.w_a, ra.width()));
                }
                if (rb.width() - t.w_b).abs() > 0.5 {
                    fail(format!("the OTHER panel changed {:.0} -> {:.0}", t.w_b, rb.width()));
                }
                t.w_drag = ra.width();
                // The edge now sits 100px right: double-click it.
                t.edge = egui::pos2(ra.max.x - 1.0, ra.center().y);
                btn(raw, t.edge, true);
                next(&mut t);
            }
            15 | 17 => {
                btn(raw, t.edge, false);
                next(&mut t);
            }
            16 => {
                btn(raw, t.edge, true);
                next(&mut t);
            }
            18 if t.frames > 6 => {
                let Some(ra) = rect(self, t.a) else { fail("panel went missing".into()) };
                if (ra.width() - t.w_a).abs() > 0.5 {
                    fail(format!("double-click left the panel at {:.0}, want {:.0}", ra.width(), t.w_a));
                }
                println!(
                    "PANEL WIDTH TEST PASS: a click on the name selected panel {}, its edge dragged 100px widened it alone ({:.0} -> {:.0}, the next stayed {:.0}), a double-click gave the shared width back",
                    t.a, t.w_a, t.w_drag, t.w_b
                );
                std::process::exit(0);
            }
            _ => {}
        }
        self.test_panelw = Some(t);
    }
}
