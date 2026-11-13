//! `--test-curveedit`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    /// --test-curves driver: animate two channels on the Transform, then
    /// drag one key +4 frames / +20 units in the curve editor and assert the
    /// move landed exactly, the neighbouring key held, and only the animated
    /// channels are listed.
    /// --test-curveedit driver: the shared curve editor's gesture layer.
    /// Adds a point with shift-click, marquees the lot, drags them as ONE
    /// group, then alt-clicks one away — the four things that are the whole
    /// reason the widget exists, and none of which a unit test can reach.
    pub(crate) fn drive_test_curveedit(&mut self, raw: &mut egui::RawInput) {
        let Some((stage, mut frames, cid)) = self.test_curveedit else { return };
        frames += 1;
        if frames > 900 {
            println!("CURVEEDIT TEST FAIL: timeout at stage {stage}");
            std::process::exit(1);
        }
        let press = |raw: &mut egui::RawInput,
                     pos: egui::Pos2,
                     down: bool,
                     mods: egui::Modifiers| {
            raw.events.insert(0, egui::Event::ModifiersChanged(mods));
            raw.events.push(egui::Event::PointerMoved(pos));
            raw.events.push(egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed: down,
                modifiers: mods,
            });
        };
        let mv = |raw: &mut egui::RawInput, pos: egui::Pos2| {
            raw.events.push(egui::Event::PointerMoved(pos));
        };
        let shift = egui::Modifiers { shift: true, ..Default::default() };
        let alt = egui::Modifiers { alt: true, ..Default::default() };
        let points = |s: &Self| -> Vec<[f32; 2]> {
            match s.egraph.node(cid).map(|n| &n.kind) {
                Some(NodeKind::Curve(cv)) => cv.points.clone(),
                _ => Vec::new(),
            }
        };
        let mut next = stage;
        match stage {
            // Drop a Curve in, select it, and make sure its panel is up.
            0 if self.playback.is_some() && self.displayed_total >= 3 => {
                let id = self.egraph.add(
                    NodeKind::Curve(Default::default()),
                    egui::Pos2::new(360.0, 120.0),
                );
                self.selected = vec![id];
                self.prop_order = vec![id];
                self.props_visible = true;
                self.bottom = BottomTab::Properties;
                self.test_curveedit = Some((1, 0, id));
                return;
            }
            // Shift-click in an empty part of the canvas: a third point.
            1 if frames > 8 && self.curve_rect.is_some() => {
                let r = self.curve_rect.unwrap();
                let p = egui::pos2(r.center().x, r.min.y + r.height() * 0.25);
                press(raw, p, true, shift);
                next = 2;
            }
            2 if frames > 2 => {
                let r = self.curve_rect.unwrap();
                let p = egui::pos2(r.center().x, r.min.y + r.height() * 0.25);
                press(raw, p, false, shift);
                next = 3;
            }
            3 if frames > 2 => {
                let n = points(self).len();
                if n != 3 {
                    println!("CURVEEDIT TEST FAIL: shift-click left {n} points, wanted 3");
                    std::process::exit(1);
                }
                // Marquee the whole canvas: every point selected.
                let r = self.curve_rect.unwrap();
                press(raw, r.min + egui::vec2(2.0, 2.0), true, Default::default());
                next = 4;
            }
            4 if frames > 2 => {
                let r = self.curve_rect.unwrap();
                mv(raw, r.center());
                next = 41;
            }
            41 if frames > 2 => {
                let r = self.curve_rect.unwrap();
                mv(raw, r.max - egui::vec2(2.0, 2.0));
                next = 5;
            }
            5 if frames > 2 => {
                let r = self.curve_rect.unwrap();
                press(raw, r.max - egui::vec2(2.0, 2.0), false, Default::default());
                self.test_curveedit = Some((6, 0, cid));
                // Remember the shape before the group drag.
                self.curve_before = points(self);
                return;
            }
            // Grab the middle point and pull UP: all three must ride along,
            // by the same amount.
            6 if frames > 2 => {
                let before = self.curve_before.clone();
                let _ = &before;
                let p = self.curve_screen[1];
                self.curve_grab = p;
                press(raw, p, true, Default::default());
                next = 7;
            }
            7 if frames > 2 => {
                // Two moves: egui needs one to call it a drag at all, and
                // the second is the move the widget acts on.
                mv(raw, self.curve_grab - egui::vec2(0.0, 8.0));
                next = 71;
            }
            71 if frames > 2 => {
                mv(raw, self.curve_grab - egui::vec2(0.0, 24.0));
                next = 8;
            }
            8 if frames > 2 => {
                let p = self.curve_grab - egui::vec2(0.0, 24.0);
                press(raw, p, false, Default::default());
                next = 9;
            }
            9 if frames > 4 => {
                let before = self.curve_before.clone();
                let after = points(self);
                let lifts: Vec<f32> = before
                    .iter()
                    .zip(after.iter())
                    .map(|(b, a)| a[1] - b[1])
                    .collect();
                if lifts.len() != 3 || lifts.iter().any(|l| *l < 0.05) {
                    println!(
                        "CURVEEDIT TEST FAIL: the group drag moved {lifts:?}, wanted all three up"
                    );
                    std::process::exit(1);
                }
                let spread = lifts
                    .iter()
                    .cloned()
                    .fold(f32::MIN, f32::max)
                    - lifts.iter().cloned().fold(f32::MAX, f32::min);
                if spread > 0.02 {
                    println!(
                        "CURVEEDIT TEST FAIL: selection moved apart ({lifts:?}), not as a group"
                    );
                    std::process::exit(1);
                }
                println!("CURVEEDIT TEST: 3 points selected and lifted together (+{:.3})", lifts[1]);
                // Clear the selection so the alt-click below takes one point,
                // not the set.
                let r = self.curve_rect.unwrap();
                press(raw, egui::pos2(r.min.x + 3.0, r.max.y - 3.0), true, Default::default());
                next = 10;
            }
            10 if frames > 2 => {
                let r = self.curve_rect.unwrap();
                press(raw, egui::pos2(r.min.x + 3.0, r.max.y - 3.0), false, Default::default());
                next = 11;
            }
            // Alt-click the middle point away.
            11 if frames > 2 => {
                let p = self.curve_screen[1];
                self.curve_grab = p;
                press(raw, p, true, alt);
                next = 12;
            }
            12 if frames > 2 => {
                press(raw, self.curve_grab, false, alt);
                next = 13;
            }
            13 if frames > 4 => {
                let after = points(self);
                if after.len() != 2 {
                    println!(
                        "CURVEEDIT TEST FAIL: alt-click left {} points, wanted 2",
                        after.len()
                    );
                    std::process::exit(1);
                }
                println!(
                    "CURVEEDIT TEST PASS: shift-click added, marquee selected 3, they dragged \
                     as a group, alt-click removed one (now {} points)",
                    after.len()
                );
                std::process::exit(0);
            }
            _ => {}
        }
        self.test_curveedit = Some((next, if next == stage { frames } else { 0 }, cid));
    }
}
