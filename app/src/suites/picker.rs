//! `--test-picker`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    /// --test-picker driver: the pixel readout end to end — P toggles it,
    /// hovering the image lands a sample that matches what the comp actually
    /// holds there, shift+P pins it, and P again clears out.
    ///
    /// The unit tests cover the state machine; this covers the wiring the
    /// unit tests cannot see — the key reaching the app, the pointer mapping
    /// to comp pixels, and the readback resolving.
    pub(crate) fn drive_test_picker(&mut self, raw: &mut egui::RawInput) {
        let Some((stage, mut frames)) = self.test_picker else { return };
        frames += 1;
        if frames > 900 {
            println!("PICKER TEST FAIL: timeout at stage {stage}");
            std::process::exit(1);
        }
        let key = |raw: &mut egui::RawInput, k: egui::Key, shift: bool, down: bool| {
            raw.events.push(egui::Event::Key {
                key: k,
                physical_key: None,
                pressed: down,
                repeat: false,
                modifiers: egui::Modifiers { shift, ..Default::default() },
            });
        };
        let centre = self.image_rect.center();
        let mut next = stage;
        match stage {
            // Let the plates land and the comp render before sampling it.
            0 if frames >= 30 && self.comp.output_size().is_some() => {
                if self.picker.on {
                    println!("PICKER TEST FAIL: readout on before P");
                    std::process::exit(1);
                }
                key(raw, egui::Key::P, false, true);
                next = 1;
            }
            1 => {
                key(raw, egui::Key::P, false, false);
                raw.events.push(egui::Event::PointerMoved(centre));
                next = 2;
            }
            // The readback is async: give it frames to land.
            2 if frames >= 60 => {
                if !self.picker.on {
                    println!("PICKER TEST FAIL: P did not turn the readout on");
                    std::process::exit(1);
                }
                raw.events.push(egui::Event::PointerMoved(centre));
                let Some(s) = self.picker.live else { return };
                let (ow, oh) = self.comp.output_size().unwrap();
                // The sample must name a real pixel near where we pointed.
                if s.x < 0 || s.y < 0 || s.x >= ow as i32 || s.y >= oh as i32 {
                    println!("PICKER TEST FAIL: sample at {},{} outside {ow}x{oh}", s.x, s.y);
                    std::process::exit(1);
                }
                let dx = (s.x - ow as i32 / 2).abs();
                let dy = (s.y - oh as i32 / 2).abs();
                if dx > 2 || dy > 2 {
                    println!(
                        "PICKER TEST FAIL: pointed at the centre, sampled {},{} of {ow}x{oh}",
                        s.x, s.y
                    );
                    std::process::exit(1);
                }
                if !s.rgba.iter().all(|v| v.is_finite()) {
                    println!("PICKER TEST FAIL: non-finite sample {:?}", s.rgba);
                    std::process::exit(1);
                }
                // Hold it, then look somewhere else: the two have to differ,
                // which is the thing a readout wired to the wrong texture (or
                // to nothing) cannot fake.
                key(raw, egui::Key::P, true, true);
                next = 3;
            }
            3 => {
                key(raw, egui::Key::P, true, false);
                next = 4;
            }
            4 if frames >= 70 => {
                if self.picker.pinned.len() != 1 {
                    println!(
                        "PICKER TEST FAIL: shift+P pinned {} samples, want 1",
                        self.picker.pinned.len()
                    );
                    std::process::exit(1);
                }
                // A third of the frame away, on the axis 2k_zip_bg's sine
                // scrolls along.
                let away = egui::pos2(
                    centre.x - self.image_rect.width() * 0.3,
                    centre.y - self.image_rect.height() * 0.3,
                );
                raw.events.push(egui::Event::PointerMoved(away));
                next = 5;
            }
            5 if frames >= 110 => {
                let away = egui::pos2(
                    centre.x - self.image_rect.width() * 0.3,
                    centre.y - self.image_rect.height() * 0.3,
                );
                raw.events.push(egui::Event::PointerMoved(away));
                let (Some(a), Some(b)) = (self.picker.pinned.first().copied(), self.picker.live)
                else {
                    return;
                };
                if a.x == b.x && a.y == b.y {
                    return; // the new sample has not landed yet
                }
                let moved = (0..4).any(|i| (a.rgba[i] - b.rgba[i]).abs() > 1e-4);
                if !moved {
                    println!(
                        "PICKER TEST FAIL: two points of the plate read identically \
                         ({:?} at {},{} and {},{}) — the readout is not tracking the image",
                        a.rgba, a.x, a.y, b.x, b.y
                    );
                    std::process::exit(1);
                }
                key(raw, egui::Key::P, false, true);
                next = 6;
            }
            6 => {
                key(raw, egui::Key::P, false, false);
                next = 7;
            }
            7 if frames >= 130 => {
                if self.picker.on || !self.picker.pinned.is_empty() {
                    println!("PICKER TEST FAIL: P did not clear the readout");
                    std::process::exit(1);
                }
                println!("PICKER TEST PASS: P toggles, hover samples the right pixel, shift+P pins");
                std::process::exit(0);
            }
            _ => {}
        }
        self.test_picker = Some((next, frames));
    }
}
