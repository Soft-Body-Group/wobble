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
            6 if colour(self) != START => {
                let c = colour(self);
                if c[3] != START[3] {
                    fail(format!("alpha changed to {} — it is never sampled", c[3]));
                }
                if eyedrop::armed(&self.egui_ctx) {
                    fail("still armed after the sample landed".into());
                }
                // The readout at the same spot is the number to match.
                self.picker.size = 0;
                self.picker.on = true;
                next = 7;
            }
            7 if frames > 10 && self.picker.live.is_some() => {
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
                println!("EYEDROP TEST PASS: {:?}", colour(self));
                std::process::exit(0);
            }
            _ => {}
        }
        self.test.eyedrop = Some((next, if next == stage { frames } else { 0 }, bid));
    }
}
