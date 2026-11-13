//! `--test-tv`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    /// --test-tv driver. See `Options::test_tv`.
    pub(crate) fn drive_test_tv(&mut self, raw: &mut egui::RawInput) {
        let Some((stage, mut frames, mut wait)) = self.test_tv else { return };
        frames += 1;
        if frames > 2400 {
            println!("TV TEST FAIL: timeout at stage {stage} (tv {:?})", self.node_editor.test_tv(0));
            std::process::exit(1);
        }
        let fail = |m: String| -> ! {
            println!("TV TEST FAIL: {m}");
            std::process::exit(1);
        };
        let ra = 0u64;
        let key = |raw: &mut egui::RawInput, k: egui::Key, m: egui::Modifiers| {
            raw.events.insert(0, egui::Event::ModifiersChanged(m));
            for pressed in [true, false] {
                raw.events.push(egui::Event::Key {
                    key: k,
                    physical_key: None,
                    pressed,
                    repeat: false,
                    modifiers: m,
                });
            }
        };
        let mv = |raw: &mut egui::RawInput, p: egui::Pos2| {
            raw.events.insert(0, egui::Event::ModifiersChanged(egui::Modifiers::NONE));
            raw.events.push(egui::Event::PointerMoved(p));
        };
        // Where a scrub at fraction `f` across the screen should land.
        let aim = |s: &Self, f: f32| -> Option<(egui::Pos2, usize)> {
            let (_, _, _, screen) = s.node_editor.test_tv(ra)?;
            let len = s.node_editor.tv_feeds.get(&ra)?.len();
            let x = screen.min.x + f * screen.width();
            Some((egui::pos2(x, screen.center().y), tv::scrub_index(x, screen.min.x, screen.max.x, len)))
        };
        let mut next = stage;
        match stage {
            0 => {
                if self.layout_settled() && frames >= 8 && self.loaded.contains_key(&ra) {
                    self.selected = vec![ra];
                    if let Some((_, r)) = self.node_rects.iter().find(|(i, _)| *i == ra) {
                        mv(raw, r.center());
                    }
                    next = 1;
                }
            }
            1 => {
                if let Some((_, r)) = self.node_rects.iter().find(|(i, _)| *i == ra) {
                    mv(raw, r.center());
                }
                key(raw, egui::Key::H, egui::Modifiers::SHIFT);
                next = 2;
            }
            2 => {
                if !self.egraph.node(ra).is_some_and(|n| n.tv) {
                    fail("shift+H did not turn the selected Read into a TV".into());
                }
                next = 3;
            }
            3 => {
                // Drawn as a TV: its hit rect is the cabinet, not a circle.
                let Some((_, r)) = self.node_rects.iter().find(|(i, _)| *i == ra) else { return };
                let sz = self.egraph.node(ra).map(|n| n.size).unwrap_or(1.0);
                let want = tv::TV_HALF.x * 2.0 * self.node_editor.zoom * sz;
                if (r.width() - want).abs() > 1.0 {
                    fail(format!("the TV's node rect is {:.1} wide, want the cabinet's {want:.1}", r.width()));
                }
                // Zoom in on it: a screen past BIG_AT asks for the big tier.
                let pa = self.egraph.node(ra).map(|n| n.pos).unwrap_or_default();
                self.node_editor.zoom = 4.6;
                self.node_editor.pan = -pa.to_vec2() * 4.6;
                next = 4;
            }
            4..=9 => next = stage + 1, // let the view land
            10 => {
                let Some((p, _)) = aim(self, 0.25) else { return };
                mv(raw, p);
                wait = 0;
                next = 11;
            }
            11 => {
                // Hover-scrub at 25%: the index follows the pointer at once,
                // and the exact BIG frame lands once its decode does.
                let Some((p, want)) = aim(self, 0.25) else { return };
                mv(raw, p);
                let Some((idx, big, exact, screen)) = self.node_editor.test_tv(ra) else { return };
                if !big {
                    fail(format!("a {:.0}px screen should ask the big tier", screen.width()));
                }
                if idx != want {
                    fail(format!("scrub at 25% asked for frame {idx}, want {want}"));
                }
                wait += 1;
                if exact {
                    wait = 0;
                    next = 12;
                } else if wait > 900 {
                    fail(format!("frame {want} never decoded (big tier)"));
                }
            }
            12 => {
                let Some((p, want)) = aim(self, 0.75) else { return };
                mv(raw, p);
                let Some((idx, _, exact, _)) = self.node_editor.test_tv(ra) else { return };
                wait += 1;
                if idx == want && exact {
                    wait = 0;
                    next = 13;
                } else if wait > 900 {
                    fail(format!("scrub at 75% stuck on frame {idx} (want {want}, exact {exact})"));
                }
            }
            13 => {
                // Leave the screen: the TV follows the playhead again.
                mv(raw, self.graph_panel_rect.left_top() + egui::vec2(8.0, 8.0));
                let Some((idx, _, _, _)) = self.node_editor.test_tv(ra) else { return };
                let now = self.node_editor.tv_feeds.get(&ra).map(|f| f.now);
                wait += 1;
                // Following is throttled to FOLLOW_HZ on WALL time, and a
                // suite's frames sit on dt's 1 ms floor: budget ~0.6 s.
                if Some(idx) == now {
                    wait = 0;
                    next = 14;
                } else if wait > 900 {
                    fail(format!("off the screen the TV shows {idx}, want the playhead's {now:?}"));
                }
            }
            14 => {
                if let Some((_, r)) = self.node_rects.iter().find(|(i, _)| *i == ra) {
                    mv(raw, r.center());
                }
                key(raw, egui::Key::H, egui::Modifiers::SHIFT);
                next = 15;
            }
            15 => {
                if self.egraph.node(ra).is_some_and(|n| n.tv) {
                    fail("a second shift+H did not turn the TV back into a Read".into());
                }
                println!(
                    "TV TEST PASS: shift+H made Read A a CRT with a cabinet-wide hit rect; \
                     zoomed in it asked the big tier, hover-scrub at 25% and 75% landed the \
                     exact decoded frames, leaving it followed the playhead, shift+H undid it"
                );
                std::process::exit(0);
            }
            _ => {}
        }
        self.test_tv = Some((next, frames, wait));
    }
}
