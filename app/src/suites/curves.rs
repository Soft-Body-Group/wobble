//! `--test-curves`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    pub(crate) fn drive_test_curves(&mut self, raw: &mut egui::RawInput) {
        let Some((stage, mut frames)) = self.test_curves else { return };
        frames += 1;
        if frames > 900 {
            println!("CURVES TEST FAIL: timeout at stage {stage}");
            std::process::exit(1);
        }
        let press = |raw: &mut egui::RawInput, pos: egui::Pos2, down: bool| {
            raw.events.push(egui::Event::PointerMoved(pos));
            raw.events.push(egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed: down,
                modifiers: Default::default(),
            });
        };
        let key_of = |s: &Self, ch: anim::Channel, f: u64| -> Option<(u64, f32)> {
            s.egraph
                .node(2)
                .and_then(|n| n.anim.get(&ch))
                .and_then(|c| c.keys.iter().find(|k| k.frame == f as f64).map(|k| (k.frame as u64, k.value)))
        };
        let mut next = stage;
        match stage {
            0 if frames >= 4 => {
                if let Some(n) = self.egraph.node_mut(2) {
                    let c = n.anim.entry(anim::Channel::TranslateX).or_default();
                    c.set_key(0, 0.0);
                    c.set_key(12, 240.0);
                    let r = n.anim.entry(anim::Channel::Rotate).or_default();
                    r.set_key(0, 0.0);
                    r.set_key(12, 45.0);
                }
                self.selected = vec![2];
                self.bottom = BottomTab::Curves;
                self.props_visible = true;
                next = 1;
            }
            1 if frames >= 8 => {
                // Transform has 7 channels; exactly the 2 keyed ones show.
                if self.curves.last_tracks != 2 {
                    println!(
                        "CURVES TEST FAIL: {} tracks listed, expected 2 (animated only)",
                        self.curves.last_tracks
                    );
                    std::process::exit(1);
                }
                press(raw, self.curves.key_pos(12.0, 240.0), true);
                next = 2;
            }
            2 => {
                raw.events.push(egui::Event::PointerMoved(self.curves.key_pos(14.0, 250.0)));
                next = 3;
            }
            3 => {
                raw.events.push(egui::Event::PointerMoved(self.curves.key_pos(16.0, 260.0)));
                next = 4;
            }
            4 => {
                press(raw, self.curves.key_pos(16.0, 260.0), false);
                next = 5;
            }
            5 => {
                let moved = key_of(self, anim::Channel::TranslateX, 16);
                let gone = key_of(self, anim::Channel::TranslateX, 12).is_none();
                let held = key_of(self, anim::Channel::TranslateX, 0);
                let other = key_of(self, anim::Channel::Rotate, 12);
                let ok_val = moved.is_some_and(|(_, v)| (v - 260.0).abs() < 1.0);
                let ok_held = held.is_some_and(|(_, v)| v.abs() < 1e-4);
                let ok_other = other.is_some_and(|(_, v)| (v - 45.0).abs() < 1e-4);
                if !(gone && ok_val && ok_held && ok_other) {
                    println!(
                        "CURVES TEST FAIL: moved={moved:?} gone={gone} held={held:?} \
                         other={other:?}"
                    );
                    std::process::exit(1);
                }
                // Interpolation modes must reach the engine graph, not just
                // the editor: hold the first key and sample mid-segment.
                if let Some(c) =
                    self.egraph.node_mut(2).and_then(|n| n.anim.get_mut(&anim::Channel::TranslateX))
                {
                    c.set_interp(0, anim::Interp::Hold);
                }
                next = 6;
            }
            6 => {
                let at = |s: &Self, t: u64| -> Option<f32> {
                    match s.egraph.to_engine_graph_at(Some(t)).nodes.get(&2) {
                        Some(engine::graph::NodeKind::Transform(tr)) => Some(tr.translate[0]),
                        _ => None,
                    }
                };
                // Key 0 = 0.0 held until key 16 = 260.0.
                let (mid, on_key) = (at(self, 8), at(self, 16));
                let held = mid.is_some_and(|v| v.abs() < 1e-4);
                let landed = on_key.is_some_and(|v| (v - 260.0).abs() < 1.0);
                if !(held && landed) {
                    println!("CURVES TEST FAIL: hold not applied — f8={mid:?} f16={on_key:?}");
                    std::process::exit(1);
                }
                // Now the group tools: select the whole channel so the box
                // appears, then scale it from its right-hand edge.
                self.curves.select_channel(&self.egraph, 2, anim::Channel::TranslateX);
                next = 7;
            }
            7 if frames >= 4 => {
                let right = self
                    .curves
                    .box_handles
                    .iter()
                    .find(|(_, _, ax, ay)| *ax && !*ay)
                    .map(|(r, ..)| r.center());
                let Some(rp) = right else {
                    println!("CURVES TEST FAIL: no group box for the selected keys");
                    std::process::exit(1);
                };
                press(raw, rp, true);
                self.test_scale_from = Some(rp);
                self.test_curves = Some((8, 0));
                return;
            }
            8 => {
                // Two moves: egui needs a little travel before a press
                // becomes a drag. Pull the edge 120px to the right.
                let from = self.test_scale_from.unwrap();
                raw.events.push(egui::Event::PointerMoved(from + egui::vec2(180.0, 0.0)));
                next = 9;
            }
            9 => {
                let from = self.test_scale_from.unwrap();
                raw.events.push(egui::Event::PointerMoved(from + egui::vec2(380.0, 0.0)));
                next = 10;
            }
            10 => {
                let from = self.test_scale_from.unwrap();
                press(raw, from + egui::vec2(380.0, 0.0), false);
                next = 11;
            }
            11 => {
                let now: Vec<u64> = self
                    .egraph
                    .node(2)
                    .and_then(|n| n.anim.get(&anim::Channel::TranslateX))
                    .map(|c| c.keys.iter().map(|k| k.frame as u64).collect())
                    .unwrap_or_default();
                // Anchored on the left edge, frame 0 holds and 16 stretches
                // by exactly the ratio the drag describes — derived from the
                // view's own mapping rather than guessed.
                let from = self.test_scale_from.unwrap();
                let f_was = self.curves.frame_at_x(from.x);
                let f_now = self.curves.frame_at_x(from.x + 380.0);
                let want = (16.0 * (f_now / f_was.max(1e-3))).round() as u64;
                let ok = now.len() == 2
                    && now[0] == 0
                    && now[1].abs_diff(want) <= 1
                    // A real stretch, not a one-frame nudge.
                    && now[1] >= 20;
                if !ok {
                    println!("CURVES TEST FAIL: group scale gave {now:?}, wanted [0, {want}]");
                    std::process::exit(1);
                }
                println!(
                    "CURVES TEST PASS: key 12->16 val 260.00, {} animated tracks shown, \
                     Hold reaches the engine graph, group scale stretched 0..16 to {now:?}",
                    self.curves.last_tracks
                );
                std::process::exit(0);
            }
            _ => {}
        }
        self.test_curves = Some((next, frames));
    }
}
