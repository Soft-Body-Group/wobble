//! `--test-eyedrop`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    /// --test-layer driver: build a two-layer stack out of the two plates,
    /// then drive the mini timeline's gestures — slide a bar along to
    /// re-time a layer, drag one past its neighbour and see it NOT restack
    /// (bars move left and right only), and ctrl-click the picture to
    /// select a layer.
    /// --test-eyedrop: hover a Blank's colour swatch and press E, click the
    /// viewer: the swatch takes the comp's scene-linear value under the
    /// click — the pixel readout's number at the same spot. Then E + Esc
    /// arms and stands down without touching the colour.
    pub(crate) fn drive_test_eyedrop(&mut self, raw: &mut egui::RawInput) {
        let Some((stage, mut frames, bid)) = self.test.eyedrop else { return };
        frames += 1;
        if frames > 900 {
            println!("EYEDROP TEST FAIL: timeout at stage {stage}");
            std::process::exit(1);
        }
        let fail = |m: String| -> ! {
            println!("EYEDROP TEST FAIL: {m}");
            std::process::exit(1);
        };
        let key = |raw: &mut egui::RawInput, k: egui::Key| {
            for pressed in [true, false] {
                raw.events.push(egui::Event::Key {
                    key: k,
                    physical_key: None,
                    pressed,
                    repeat: false,
                    modifiers: Default::default(),
                });
            }
        };
        let button = |raw: &mut egui::RawInput, pos: egui::Pos2, down: bool| {
            raw.events.push(egui::Event::PointerMoved(pos));
            raw.events.push(egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed: down,
                modifiers: Default::default(),
            });
        };
        let colour = |s: &Self| match s.egraph.node(bid).map(|n| &n.kind) {
            Some(NodeKind::Blank(p)) => p.color,
            _ => [f32::NAN; 4],
        };
        const START: [f32; 4] = [0.9, 0.1, 0.8, 0.5];
        // A spot on the plate, away from the HUD corners.
        let spot = |s: &Self| {
            let r = s.image_rect;
            r.min + egui::vec2(0.41, 0.47) * r.size()
        };
        let spot_a = |s: &Self| s.image_rect.min + egui::vec2(0.25, 0.30) * s.image_rect.size();
        let spot_b = |s: &Self| s.image_rect.min + egui::vec2(0.62, 0.66) * s.image_rect.size();
        let spot_c = |s: &Self| s.image_rect.min + egui::vec2(0.75, 0.40) * s.image_rect.size();
        let swatch = |s: &Self| {
            colorpicker::last_geom(&s.egui_ctx, (bid, "blank-color"))
                .map(|g| g.swatch)
                .filter(|r| r.width() > 0.0)
        };
        let mut next = stage;
        match stage {
            // View the plate; raise a Blank's panel for its swatch.
            0 if self.playback.is_some() && self.displayed_total >= 2 => {
                let read = self
                    .egraph
                    .nodes
                    .iter()
                    .find(|(_, n)| matches!(n.kind, NodeKind::Read { .. }))
                    .map(|(id, _)| *id)
                    .unwrap_or_else(|| fail("needs a Read".into()));
                let b = self.egraph.add(
                    NodeKind::Blank(engine::graph::BlankParams {
                        color: START,
                        ..Default::default()
                    }),
                    egui::pos2(360.0, 120.0),
                );
                self.egraph.view = Some(read);
                self.playing = false;
                self.seek_to(5);
                self.selected = vec![b];
                self.prop.order = vec![b];
                self.props_visible = true;
                self.bottom = BottomTab::Properties;
                self.test.eyedrop = Some((1, 0, b));
                return;
            }
            // E over the swatch arms the viewer.
            1 if frames > 30 && self.layout_settled() && swatch(self).is_some() => {
                let r = swatch(self).unwrap();
                raw.events.push(egui::Event::PointerMoved(r.center()));
                next = 2;
            }
            2 if frames > 2 => {
                raw.events.push(egui::Event::PointerMoved(swatch(self).unwrap().center()));
                key(raw, egui::Key::E);
                next = 3;
            }
            3 if frames > 2 => {
                if !eyedrop::armed(&self.egui_ctx) {
                    fail("E over the swatch did not arm the eyedropper".into());
                }
                raw.events.push(egui::Event::PointerMoved(spot(self)));
                next = 4;
            }
            4 if frames > 2 => {
                button(raw, spot(self), true);
                next = 5;
            }
            5 if frames > 2 => {
                button(raw, spot(self), false);
                next = 6;
            }
            // The sample lands a frame or two later. Alpha is the swatch's.
            6 if colour(self) != START && !eyedrop::armed(&self.egui_ctx) => {
                let c = colour(self);
                if c[3] != START[3] {
                    fail(format!("alpha changed to {} — it is never sampled", c[3]));
                }
                // The readout at the same spot is the number to match.
                self.picker.size = 0;
                self.picker.on = true;
                next = 7;
            }
            // Hold the pointer on the spot: a real mouse over the window
            // would otherwise move the readout off it.
            7 if frames <= 10 => {
                raw.events.push(egui::Event::PointerMoved(spot(self)));
            }
            7 if self.picker.live.is_some() => {
                let want = self.picker.live.unwrap().rgba;
                let got = colour(self);
                for k in 0..3 {
                    if (got[k] - want[k]).abs() > 2e-3 {
                        fail(format!("sampled {got:?} but the readout there says {want:?}"));
                    }
                }
                self.picker.on = false;
                raw.events.push(egui::Event::PointerMoved(swatch(self).unwrap().center()));
                next = 8;
            }
            // E, then Esc: armed and stood down, colour untouched.
            8 if frames > 2 => {
                raw.events.push(egui::Event::PointerMoved(swatch(self).unwrap().center()));
                key(raw, egui::Key::E);
                next = 9;
            }
            9 if frames > 2 => {
                if !eyedrop::armed(&self.egui_ctx) {
                    fail("second E did not arm".into());
                }
                key(raw, egui::Key::Escape);
                next = 10;
            }
            10 if frames > 2 => {
                if eyedrop::armed(&self.egui_ctx) {
                    fail("Esc did not stand the eyedropper down".into());
                }
                raw.events.push(egui::Event::PointerMoved(swatch(self).unwrap().center()));
                key(raw, egui::Key::E);
                next = 11;
            }
            // A plain drag FOLLOWS the pointer: press at A, move to B, and
            // the swatch takes B's pixel while the button is still down.
            11 if frames > 2 => {
                if !eyedrop::armed(&self.egui_ctx) {
                    fail("E did not arm for the drag".into());
                }
                self.test.rect.min = egui::pos2(colour(self)[0], colour(self)[1]);
                button(raw, spot_a(self), true);
                next = 12;
            }
            12 if frames > 2 => {
                raw.events.push(egui::Event::PointerMoved(spot_b(self)));
                self.picker.on = true;
                next = 13;
            }
            13 if frames <= 20 => {
                raw.events.push(egui::Event::PointerMoved(spot_b(self)));
            }
            13 if self.picker.live.is_some() => {
                let want = self.picker.live.unwrap().rgba;
                let got = colour(self);
                if (0..3).any(|k| (got[k] - want[k]).abs() > 2e-3) {
                    fail(format!("mid-drag swatch {got:?}, the pixel under the pointer is {want:?}"));
                }
                if !eyedrop::armed(&self.egui_ctx) {
                    fail("a live sample disarmed mid-drag".into());
                }
                self.picker.on = false;
                button(raw, spot_b(self), false);
                next = 14;
            }
            14 if frames > 10 => {
                if eyedrop::armed(&self.egui_ctx) {
                    fail("still armed after the drag released".into());
                }
                self.test.rect.max = egui::pos2(colour(self)[0], colour(self)[1]);
                raw.events.push(egui::Event::PointerMoved(swatch(self).unwrap().center()));
                next = 15;
            }
            // Shift-drag draws a box: no live updates, the mean on release.
            15 if frames > 2 => {
                raw.events.push(egui::Event::PointerMoved(swatch(self).unwrap().center()));
                key(raw, egui::Key::E);
                next = 16;
            }
            16 if frames > 2 => {
                raw.events.insert(0, egui::Event::ModifiersChanged(egui::Modifiers::SHIFT));
                raw.events.push(egui::Event::PointerMoved(spot_a(self)));
                raw.events.push(egui::Event::PointerButton {
                    pos: spot_a(self),
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::SHIFT,
                });
                next = 17;
            }
            17 if frames > 2 => {
                raw.events.push(egui::Event::PointerMoved(spot_c(self)));
                if frames > 20 {
                    let c = colour(self);
                    if (c[0], c[1]) != (self.test.rect.max.x, self.test.rect.max.y) {
                        fail(format!("a shift-drag sampled live ({c:?})"));
                    }
                    next = 18;
                }
            }
            18 => {
                raw.events.push(egui::Event::PointerButton {
                    pos: spot_c(self),
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers: egui::Modifiers::SHIFT,
                });
                next = 19;
            }
            19 if frames > 10 => {
                raw.events.insert(0, egui::Event::ModifiersChanged(Default::default()));
                if eyedrop::armed(&self.egui_ctx) {
                    fail("still armed after the box".into());
                }
                let c = colour(self);
                if (c[0], c[1]) == (self.test.rect.max.x, self.test.rect.max.y) {
                    fail("the box did not land".into());
                }
                println!("EYEDROP TEST PASS: click samples the readout's pixel, drag follows the pointer live, shift-drag averages a box, Esc stands down");
                std::process::exit(0);
            }
            _ => {}
        }
        self.test.eyedrop = Some((next, if next == stage { frames } else { 0 }, bid));
    }
}
