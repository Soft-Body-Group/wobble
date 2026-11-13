//! `--test-slice`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    /// --test-slice driver. Default graph ids: ra=0 rb=1 tf=2 gr=3 mg=4 vw=5.
    /// Phase 1: ctrl-drag across the grade->merge wire, expect it cut.
    /// Phase 2: drop a fresh Blur onto the transform->grade wire, expect a
    /// splice (tf -> blur -> gr).
    pub(crate) fn drive_test_slice(&mut self, raw: &mut egui::RawInput) {
        let Some((stage, mut frames, blur_id)) = self.test.slice else { return };
        frames += 1;
        if frames > 900 {
            println!("SLICE TEST FAIL: timeout at stage {stage}");
            std::process::exit(1);
        }
        let center = |s: &Self, id: u64| -> Option<egui::Pos2> {
            s.node_rects.iter().find(|(i, _)| *i == id).map(|(_, r)| r.center())
        };
        let stalk = editor::SOCKET_OFFSET;
        let press = |raw: &mut egui::RawInput, pos: egui::Pos2, ctrl: bool, down: bool| {
            raw.events.insert(0, egui::Event::ModifiersChanged(egui::Modifiers { ctrl: ctrl, ..Default::default() }));
            raw.events.push(egui::Event::PointerMoved(pos));
            raw.events.push(egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed: down,
                modifiers: egui::Modifiers { ctrl, ..Default::default() },
            });
        };
        let mv = |raw: &mut egui::RawInput, pos: egui::Pos2, ctrl: bool| {
            raw.events.insert(0, egui::Event::ModifiersChanged(egui::Modifiers { ctrl: ctrl, ..Default::default() }));
            raw.events.push(egui::Event::PointerMoved(pos));
        };
        // Wire midpoint between gr(3) output and mg(4) input 0 (x offset -20).
        let cut_mid = |s: &Self| -> Option<egui::Pos2> {
            let (a, b) = (center(s, 3)?, center(s, 4)?);
            let from = a + egui::vec2(0.0, stalk);
            let to = b + egui::vec2(-20.0, -stalk);
            Some(from + (to - from) * 0.5)
        };
        match stage {
            0 if frames >= 6 => {
                let Some(m) = cut_mid(self) else { return };
                press(raw, m + egui::vec2(-50.0, 0.0), true, true);
                self.test.slice = Some((1, frames, 0));
            }
            1 => {
                // Trip the drag threshold while still left of the wire...
                let m = cut_mid(self).unwrap();
                mv(raw, m + egui::vec2(-25.0, 0.0), true);
                self.test.slice = Some((2, frames, 0));
            }
            2 => {
                // ...then cross it in one clean segment.
                let m = cut_mid(self).unwrap();
                mv(raw, m + egui::vec2(40.0, 0.0), true);
                self.test.slice = Some((3, frames, 0));
            }
            3 => {
                let m = cut_mid(self).unwrap();
                press(raw, m + egui::vec2(40.0, 0.0), true, false);
                self.test.slice = Some((4, frames, 0));
            }
            4 => {
                raw.events.insert(0, egui::Event::ModifiersChanged(egui::Modifiers { ctrl: false, ..Default::default() }));
                if self.egraph.input_of(4, 0).is_some() {
                    println!("SLICE TEST FAIL: wire survived the slicer");
                    std::process::exit(1);
                }
                println!("SLICE TEST: cut ok");
                // Beside the chain rather than at a fixed coordinate: panel
                // chrome shifts the viewport, and an off-screen node can't be
                // grabbed.
                let at = self
                    .egraph
                    .node(2)
                    .map(|n| n.pos + egui::vec2(260.0, 0.0))
                    .unwrap_or(egui::Pos2::new(260.0, 0.0));
                let id = self
                    .egraph
                    .add(NodeKind::Blur(engine::graph::BlurParams::default()), at);
                self.test.slice = Some((5, frames, id));
            }
            5 => {
                // let the new node get a rect + sim entry
                if frames > 8 {
                    self.test.slice = Some((6, frames, blur_id));
                }
            }
            6 => {
                let Some(c) = center(self, blur_id) else { return };
                press(raw, c, false, true);
                self.test.slice = Some((7, frames, blur_id));
            }
            7 | 8 => {
                // Drag onto the tf(2) -> gr(3) wire midpoint, two steps so the
                // drag threshold trips.
                let (a, b) = (center(self, 2).unwrap(), center(self, 3).unwrap());
                let from = a + egui::vec2(0.0, stalk);
                let to = b + egui::vec2(0.0, -stalk);
                let m = from + (to - from) * 0.5;
                let p = if stage == 7 { m + egui::vec2(40.0, 0.0) } else { m };
                mv(raw, p, false);
                self.test.slice = Some((stage + 1, frames, blur_id));
            }
            9 => {
                let (a, b) = (center(self, 2).unwrap(), center(self, 3).unwrap());
                let from = a + egui::vec2(0.0, stalk);
                let to = b + egui::vec2(0.0, -stalk);
                let m = from + (to - from) * 0.5;
                press(raw, m, false, false);
                self.test.slice = Some((10, frames, blur_id));
            }
            10 => {
                let ok = self.egraph.input_of(blur_id, 0) == Some(2)
                    && self.egraph.input_of(3, 0) == Some(blur_id);
                if ok {
                    println!("SLICE TEST PASS: sliced grade->merge, spliced blur into tf->grade");
                    std::process::exit(0);
                } else {
                    println!(
                        "SLICE TEST FAIL: splice wrong (blur.in={:?} grade.in={:?})",
                        self.egraph.input_of(blur_id, 0),
                        self.egraph.input_of(3, 0)
                    );
                    std::process::exit(1);
                }
            }
            _ => {}
        }
        if self.test.slice.is_some_and(|(s, _, _)| s == stage) {
            self.test.slice = Some((stage, frames, blur_id));
        }
    }
}
