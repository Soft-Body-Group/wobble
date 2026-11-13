//! `--test-cornerpin`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    pub(crate) fn drive_test_cornerpin(&mut self, raw: &mut egui::RawInput) {
        let Some((stage, mut frames, id, press, snap)) = self.test.cornerpin else { return };
        frames += 1;
        let fail = |m: String| -> ! {
            println!("CORNERPIN TEST FAIL: {m}");
            std::process::exit(1);
        };
        if frames > 1200 {
            fail(format!("timeout at stage {stage}"));
        }
        let click = |raw: &mut egui::RawInput, pos: egui::Pos2, down: bool| {
            raw.events.push(egui::Event::PointerMoved(pos));
            raw.events.push(egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed: down,
                modifiers: Default::default(),
            });
        };
        let key = |raw: &mut egui::RawInput, key: egui::Key, down: bool| {
            raw.events.push(egui::Event::Key {
                key,
                physical_key: None,
                pressed: down,
                repeat: false,
                modifiers: Default::default(),
            });
        };
        let at = |s: &Self, f: [f32; 2]| {
            let r = s.test.rect;
            egui::pos2(r.min.x + r.width() * f[0], r.min.y + r.height() * f[1])
        };
        let cp = |s: &Self| match s.egraph.node(id).map(|n| n.kind_at(s.eval_tick())) {
            Some(NodeKind::CornerPin(cp)) => cp,
            _ => Default::default(),
        };
        let near = |a: egui::Pos2, b: egui::Pos2| (a - b).length() < 2.0;
        let mut st = (stage + 1, frames, id, press, snap);
        match stage {
            0 => {
                if frames < 6 || !self.layout_settled() {
                    st.0 = 0;
                } else {
                    self.test.rect = self.image_rect;
                    let n = self.egraph.add(
                        NodeKind::CornerPin(engine::graph::CornerPinParams::pinned([
                            [0.25, 0.25],
                            [0.75, 0.25],
                            [0.75, 0.75],
                            [0.25, 0.75],
                        ])),
                        egui::Pos2::new(240.0, 0.0),
                    );
                    self.selected = vec![n];
                    self.playing = false;
                    st.2 = n;
                }
            }
            // A marquee from empty canvas round the top two pins.
            1 => click(raw, at(self, [0.1, 0.1]), true),
            2 => raw.events.push(egui::Event::PointerMoved(at(self, [0.5, 0.3]))),
            3 => raw.events.push(egui::Event::PointerMoved(at(self, [0.9, 0.35]))),
            4 => click(raw, at(self, [0.9, 0.35]), false),
            5 => {
                if self.pin_sel != Some((id, [true, true, false, false])) {
                    fail(format!("the marquee picked {:?}", self.pin_sel));
                }
                if cp(self).to != [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]] {
                    fail("the marquee moved a pin".into());
                }
            }
            // E held at a press on empty canvas: scale about it.
            6 | 12 => {
                let o = at(self, [0.5, 0.6]);
                key(raw, if stage == 6 { egui::Key::E } else { egui::Key::R }, true);
                click(raw, o, true);
                st.3 = o;
                st.4 = cp(self).to;
            }
            7 | 8 => raw.events.push(egui::Event::PointerMoved(press + egui::vec2(52.0 * (stage - 6) as f32, 0.0))),
            13 => raw.events.push(egui::Event::PointerMoved(press + egui::vec2(40.0, 0.0))),
            14 => raw.events.push(egui::Event::PointerMoved(press + egui::vec2(0.0, 60.0))),
            9 => click(raw, press + egui::vec2(104.0, 0.0), false),
            15 => click(raw, press + egui::vec2(0.0, 60.0), false),
            10 => key(raw, egui::Key::E, false),
            16 => key(raw, egui::Key::R, false),
            11 | 17 => {
                let now = cp(self).to;
                for k in 0..4 {
                    let (p0, p1) = (at(self, snap[k]), at(self, now[k]));
                    let v = p0 - press;
                    let want = if k >= 2 {
                        p0
                    } else if stage == 11 {
                        press + v * (104.0f32 / 150.0).exp()
                    } else {
                        press + egui::vec2(-v.y, v.x)
                    };
                    if !near(p1, want) {
                        fail(format!("stage {stage}: pin {k} at {p1:?}, wanted {want:?}"));
                    }
                }
                if self.pin_grab.is_some() {
                    fail(format!("stage {stage}: the held-key edit outlived its release"));
                }
            }
            // Arm `from`, give it a quad inside the frame, and drag its BR
            // corner: only `from` moves, and the pin grabbed becomes the pick.
            18 => {
                self.pin_from.insert(id);
                if let Some(NodeKind::CornerPin(c)) = self.egraph.node_mut(id).map(|n| &mut n.kind) {
                    c.from = [[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7]];
                }
                st.4 = cp(self).to;
            }
            19 => {}
            20 => {
                st.3 = at(self, [0.7, 0.7]);
                click(raw, st.3, true);
            }
            21 | 22 => raw.events.push(egui::Event::PointerMoved(press + egui::vec2(-15.0, -10.0) * (stage - 20) as f32)),
            23 => click(raw, press + egui::vec2(-30.0, -20.0), false),
            24 => {
                let c = cp(self);
                if c.to != snap {
                    fail(format!("dragging a from pin moved the pin: {:?}", c.to));
                }
                let want = press + egui::vec2(-30.0, -20.0);
                if !near(at(self, c.from[2]), want) || c.from[0] != [0.3, 0.3] {
                    fail(format!("from is {:?}, wanted corner 2 at {want:?}", c.from));
                }
                if self.pin_sel != Some((id, [false, false, true, false])) {
                    fail(format!("grabbing a pin outside the pick left the pick {:?}", self.pin_sel));
                }
            }
            // X held (the 0.1 rate): a 100px drag moves the pin 10px.
            25 => {
                key(raw, egui::Key::X, true);
                st.3 = at(self, cp(self).from[2]);
                st.4 = cp(self).from;
                click(raw, st.3, true);
            }
            26 | 27 => raw.events.push(egui::Event::PointerMoved(press + egui::vec2(50.0 * (stage - 25) as f32, 0.0))),
            28 => click(raw, press + egui::vec2(100.0, 0.0), false),
            29 => key(raw, egui::Key::X, false),
            30 => {
                let got = at(self, cp(self).from[2]);
                let want = press + egui::vec2(10.0, 0.0);
                if (got - want).length() > 1.0 {
                    fail(format!("with X held a 100px drag put the pin at {got:?}, wanted {want:?}"));
                }
                // Back to editing the pin, three picked, a fresh quad.
                self.pin_from.remove(&id);
                if let Some(NodeKind::CornerPin(c)) = self.egraph.node_mut(id).map(|n| &mut n.kind) {
                    // The frame's own corners: the box's top edge IS the
                    // frame's, so a knob above it would be off the viewer.
                    *c = Default::default();
                }
                self.pin_sel = Some((id, [true, true, true, false]));
            }
            31 => {}
            // The box's rotate knob: 34px above the middle of its top edge.
            // Dragged from straight above the centre to straight right of
            // it: a quarter turn about the centre.
            32 => {
                let q = cp(self).to;
                let (mut lo, mut hi) = (egui::pos2(f32::MAX, f32::MAX), egui::pos2(f32::MIN, f32::MIN));
                for k in 0..3 {
                    lo = lo.min(at(self, q[k]));
                    hi = hi.max(at(self, q[k]));
                }
                let c = lo + (hi - lo) * 0.5;
                let Some(knob) = self.pin_knob else { fail("no rotate knob drawn".into()) };
                if !self.viewer_rect.contains(knob) {
                    fail(format!("the rotate knob is off the viewer at {knob:?} ({:?})", self.viewer_rect));
                }
                st.3 = c;
                st.4 = q;
                self.test.cornerpin_knob = knob;
                click(raw, knob, true);
            }
            33..=38 => {
                // Round the arc, a step a frame.
                let r = (self.test.cornerpin_knob - press).length();
                let a0 = (self.test.cornerpin_knob - press).angle();
                let a = a0 + std::f32::consts::FRAC_PI_2 * (stage - 32) as f32 / 6.0;
                raw.events.push(egui::Event::PointerMoved(press + egui::vec2(a.cos(), a.sin()) * r));
            }
            39 => {
                let v = self.test.cornerpin_knob - press;
                click(raw, press + egui::vec2(-v.y, v.x), false);
            }
            40 => {
                let now = cp(self).to;
                for k in 0..4 {
                    let (p0, p1) = (at(self, snap[k]), at(self, now[k]));
                    let v = p0 - press;
                    let want = if k == 3 { p0 } else { press + egui::vec2(-v.y, v.x) };
                    if !near(p1, want) {
                        fail(format!("the knob put pin {k} at {p1:?}, wanted {want:?} (pick {:?})", self.pin_sel));
                    }
                }
                println!(
                    "CORNERPIN TEST PASS: a marquee picked the top two pins; E held scaled them about \
                     the press and R held turned them a quarter round it (the other two held); with \
                     from armed a drag moved from's corner and left the pin alone; X held drags at 0.1; the box knob turned a pick of three a quarter round its centre"
                );
                std::process::exit(0);
            }
            _ => {}
        }
        self.test.cornerpin = Some(st);
    }
}
