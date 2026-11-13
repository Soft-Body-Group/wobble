//! `--test-box`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    pub(crate) fn drive_test_box(&mut self, raw: &mut egui::RawInput) {
        let Some((stage, mut frames)) = self.test_box else { return };
        frames += 1;
        if frames > 900 {
            println!("BOX TEST FAIL: timeout at stage {stage}");
            std::process::exit(1);
        }
        let center = |s: &Self, id: u64| -> Option<egui::Pos2> {
            s.node_rects.iter().find(|(i, _)| *i == id).map(|(_, r)| r.center())
        };
        let press = |raw: &mut egui::RawInput, pos: egui::Pos2, down: bool| {
            raw.events.push(egui::Event::PointerMoved(pos));
            raw.events.push(egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed: down,
                modifiers: Default::default(),
            });
        };
        let key = |raw: &mut egui::RawInput, k: egui::Key, down: bool| {
            raw.events.insert(0, egui::Event::ModifiersChanged(egui::Modifiers { command: true, ..Default::default() }));
            raw.events.push(egui::Event::Key {
                key: k,
                physical_key: None,
                pressed: down,
                repeat: false,
                modifiers: egui::Modifiers::COMMAND,
            });
        };
        let sel_rect = |s: &Self| -> Option<egui::Rect> {
            Some(egui::Rect::from_two_pos(center(s, 2)?, center(s, 3)?).expand(50.0))
        };
        let mut next = stage;
        match stage {
            0 if frames >= 6 => {
                if let Some(r) = sel_rect(self) {
                    press(raw, r.min, true);
                    next = 1;
                }
            }
            1 => {
                let r = sel_rect(self).unwrap();
                raw.events.push(egui::Event::PointerMoved(r.center()));
                next = 2;
            }
            2 => {
                let r = sel_rect(self).unwrap();
                raw.events.push(egui::Event::PointerMoved(r.max));
                next = 3;
            }
            3 => {
                let r = sel_rect(self).unwrap();
                press(raw, r.max, false);
                next = 4;
            }
            4 => {
                let mut got = self.selected.clone();
                got.sort_unstable();
                if got != vec![2, 3] {
                    println!("BOX TEST FAIL: box selected {got:?}, wanted [2, 3]");
                    std::process::exit(1);
                }
                println!("BOX TEST: marquee selected tf+gr ok");
                let c = center(self, 3).unwrap();
                press(raw, c, true);
                next = 5;
            }
            5 => {
                let c = center(self, 3).unwrap();
                press(raw, c, false);
                next = 6;
            }
            6 => next = 7, // settle
            7 => {
                if self.selected != vec![3] {
                    println!("BOX TEST FAIL: click selected {:?}", self.selected);
                    std::process::exit(1);
                }
                key(raw, egui::Key::ArrowUp, true);
                next = 8;
            }
            8 => {
                key(raw, egui::Key::ArrowUp, false);
                next = 9;
            }
            9 => {
                let ok = self.egraph.input_of(3, 0) == Some(0)
                    && self.egraph.input_of(2, 0) == Some(3)
                    && self.egraph.input_of(4, 0) == Some(2);
                let pos_swapped = self
                    .egraph
                    .node(3)
                    .is_some_and(|n| (n.pos.y - (-120.0)).abs() < 1e-3);
                if !ok || !pos_swapped {
                    println!(
                        "BOX TEST FAIL: ⌘↑ rewire wrong (gr.in={:?} tf.in={:?} mg.in={:?} pos_swapped={pos_swapped})",
                        self.egraph.input_of(3, 0),
                        self.egraph.input_of(2, 0),
                        self.egraph.input_of(4, 0),
                    );
                    std::process::exit(1);
                }
                println!("BOX TEST: ⌘↑ swapped grade above transform ok");
                key(raw, egui::Key::ArrowDown, true);
                next = 10;
            }
            10 => {
                key(raw, egui::Key::ArrowDown, false);
                next = 11;
            }
            11 => {
                let ok = self.egraph.input_of(2, 0) == Some(0)
                    && self.egraph.input_of(3, 0) == Some(2)
                    && self.egraph.input_of(4, 0) == Some(3);
                if ok {
                    println!("BOX TEST: swaps ok — rotating the selection");
                    // Rotate phase: with tf(2)+gr(3) still selected, hold R
                    // and drag to orbit their POSITIONS ~90°.
                    self.selected = vec![2, 3];
                    next = 12;
                } else {
                println!(
                    "BOX TEST FAIL: ⌘↓ didn't restore (tf.in={:?} gr.in={:?} mg.in={:?})",
                    self.egraph.input_of(2, 0),
                    self.egraph.input_of(3, 0),
                    self.egraph.input_of(4, 0),
                );
                std::process::exit(1);
                }
            }
            // Hold R and drag about the selection's centre: a quarter turn
            // from above it to beside it. The swap stages latched ⌘ into
            // raw.modifiers — clear it, or the press classifies as a pan.
            12 => {
                raw.events.insert(0, egui::Event::ModifiersChanged(Default::default()));
                let Some(r) = sel_rect(self) else { return };
                let d = center(self, 2)
                    .zip(center(self, 3))
                    .map(|(a, c)| c - a)
                    .unwrap_or(egui::Vec2::ZERO);
                // Stash the pre-rotate relative vector (min) and the pivot
                // (max) — the centre a rotation does not move.
                self.test_rect = egui::Rect {
                    min: egui::pos2(d.x, d.y),
                    max: r.center(),
                };
                raw.events.push(egui::Event::Key {
                    key: egui::Key::R,
                    physical_key: None,
                    pressed: true,
                    repeat: false,
                    modifiers: Default::default(),
                });
                press(raw, r.center() + egui::vec2(0.0, -120.0), true);
                next = 13;
            }
            13 | 14 => {
                let c = self.test_rect.max;
                let p = if stage == 13 {
                    c + egui::vec2(90.0, -90.0)
                } else {
                    c + egui::vec2(380.0, 0.0)
                };
                raw.events.push(egui::Event::PointerMoved(p));
                next = stage + 1;
            }
            15 => {
                let c = self.test_rect.max;
                press(raw, c + egui::vec2(380.0, 0.0), false);
                raw.events.push(egui::Event::Key {
                    key: egui::Key::R,
                    physical_key: None,
                    pressed: false,
                    repeat: false,
                    modifiers: Default::default(),
                });
                next = 16;
            }
            16 => {
                // The tf->gr relative vector must have turned ~90° (was
                // mostly vertical, now mostly horizontal), same length.
                let before = egui::vec2(self.test_rect.min.x, self.test_rect.min.y);
                let Some(after) = center(self, 2)
                    .zip(center(self, 3))
                    .map(|(a, c)| c - a)
                else {
                    return;
                };
                let cross = before.x * after.y - before.y * after.x;
                let dot = before.dot(after);
                let ang = cross.atan2(dot).to_degrees();
                let len_ratio = after.length() / before.length().max(1e-3);
                if ang.abs() < 45.0 || (len_ratio - 1.0).abs() > 0.15 {
                    println!(
                        "BOX TEST FAIL: rotate moved tf->gr by {ang:.0}° (len x{len_ratio:.2}), want a clean turn"
                    );
                    std::process::exit(1);
                }
                println!(
                    "BOX TEST PASS: marquee select + ⌘↑/⌘↓ chain swaps + R-drag rotate ({ang:.0}°, len x{len_ratio:.2}) verified"
                );
                std::process::exit(0);
            }
            _ => {}
        }
        self.test_box = Some((next, frames));
    }
}
