//! `--test-tab`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    pub(crate) fn drive_test_tab(&mut self, raw: &mut egui::RawInput) {
        let Some((stage, mut frames)) = self.test.tab else { return };
        frames += 1;
        if frames > 600 {
            println!("TAB TEST FAIL: timeout at stage {stage}");
            std::process::exit(1);
        }
        let key = |raw: &mut egui::RawInput, k: egui::Key, down: bool| {
            raw.events.push(egui::Event::Key {
                key: k,
                physical_key: None,
                pressed: down,
                repeat: false,
                modifiers: Default::default(),
            });
        };
        let fail = |m: &str| -> ! {
            println!("TAB TEST FAIL: {m}");
            std::process::exit(1);
        };
        // Default graph: Read B = 1, Transform = 2, Grade = 3, Blend = 4.
        let (rb, tf, gr) = (1u64, 2u64, 3u64);
        // The most recently added node of a kind: ids only grow.
        let newest = |s: &Self, want: &dyn Fn(&NodeKind) -> bool| -> Option<u64> {
            s.egraph.nodes.iter().filter(|(_, n)| want(&n.kind)).map(|(id, _)| *id).max()
        };
        // Background point left of all nodes.
        let bg = self
            .node_rects
            .iter()
            .map(|(_, r)| *r)
            .reduce(|a, b| a.union(b))
            .map(|b| egui::pos2(b.min.x - 40.0, b.center().y));
        let mut next = stage;
        match stage {
            0 if frames >= 6 => {
                let Some(p) = bg else { return };
                raw.events.push(egui::Event::PointerMoved(p));
                next = 1;
            }
            1 => {
                key(raw, egui::Key::Tab, true);
                next = 2;
            }
            2 => {
                key(raw, egui::Key::Tab, false);
                raw.events.push(egui::Event::Text("lum".into()));
                next = 3;
            }
            3 => {
                key(raw, egui::Key::Enter, true);
                next = 4;
            }
            4 => {
                key(raw, egui::Key::Enter, false);
                next = 5;
            }
            5 => next = 6, // settle
            6 => {
                let Some(lk) = newest(self, &|k| matches!(k, NodeKind::LumaKey(_))) else {
                    fail("no LumaKey node after Tab/type/Enter")
                };
                if self.egraph.input_of(lk, 0).is_some() {
                    fail("with nothing selected, the new node must arrive unwired");
                }
                // One node selected: it feeds the new node's input 0.
                self.selected = vec![tf];
                next = 10;
            }
            // Tab, type, Enter, settle — the same four beats per add.
            10..=14 | 20..=24 | 30..=34 | 40..=45 => {
                let (base, text) = match stage {
                    10..=14 => (10, "lum"),
                    20..=24 => (20, "blend"),
                    30..=34 => (30, "blur"),
                    // Neither a prefix nor a substring of anything: "grd"
                    // reaches Grid and Grade by subsequence alone.
                    _ => (40, "grd"),
                };
                match stage - base {
                    0 => {
                        // Inside the graph panel, low and left, a different
                        // spot per add. (The "left of every node" point the
                        // first add uses walks off the panel once nodes land
                        // there, and Tab only opens over the graph.)
                        let pr = self.graph_panel_rect;
                        let dx = match base {
                            10 => 70.0,
                            20 => 170.0,
                            30 => 270.0,
                            _ => 370.0,
                        };
                        let p = egui::pos2(pr.min.x + dx, pr.max.y - 70.0);
                        raw.events.push(egui::Event::PointerMoved(p));
                        key(raw, egui::Key::Tab, true);
                    }
                    1 => {
                        key(raw, egui::Key::Tab, false);
                        raw.events.push(egui::Event::Text(text.into()));
                    }
                    // ↓ walks to the second hit before Enter takes it —
                    // a tap, down AND up, because a second press of a key
                    // never released arrives flagged `repeat`.
                    2 if base == 40 => {
                        key(raw, egui::Key::ArrowDown, true);
                        key(raw, egui::Key::ArrowDown, false);
                    }
                    2 => key(raw, egui::Key::Enter, true),
                    3 if base == 40 => key(raw, egui::Key::Enter, true),
                    3 => key(raw, egui::Key::Enter, false),
                    4 if base == 40 => key(raw, egui::Key::Enter, false),
                    _ => {}
                }
                // One beat longer for the arrow walk.
                let last = if base == 40 { 5 } else { 4 };
                next = if stage - base == last { base + last + 1 } else { stage + 1 };
            }
            15 => {
                let lk = newest(self, &|k| matches!(k, NodeKind::LumaKey(_))).unwrap();
                if self.egraph.input_of(lk, 0) != Some(tf) {
                    let n = self
                        .egraph
                        .nodes
                        .values()
                        .filter(|n| matches!(n.kind, NodeKind::LumaKey(_)))
                        .count();
                    fail(&format!(
                        "the selected Transform should feed the new LumaKey (input 0 = {:?}; \
                         LumaKeys {n}; wants keyboard {}; selected {:?})",
                        self.egraph.input_of(lk, 0),
                        self.egui_ctx.egui_wants_keyboard_input(),
                        self.selected,
                    ));
                }
                if self.selected != vec![lk] {
                    fail("the new node should be what is selected after the add");
                }
                // Two selected, in pick order: the Grade first, Read B second.
                self.selected = vec![gr, rb];
                next = 20;
            }
            25 => {
                let Some(bl) = newest(self, &|k| matches!(k, NodeKind::Blend { .. })) else {
                    fail("typing \"blend\" + Enter added no Blend")
                };
                if bl == 4 {
                    fail("typing \"blend\" + Enter added no NEW Blend");
                }
                let (fg, bgi) = (self.egraph.input_of(bl, 0), self.egraph.input_of(bl, 1));
                if fg != Some(gr) || bgi != Some(rb) {
                    fail(&format!(
                        "the Blend should take the first pick (Grade) as fg and the second \
                         (Read B) as bg; got + {fg:?}, - {bgi:?}"
                    ));
                }
                // One selected again: the new node must take its PLACE —
                // what the Transform fed (the Grade, and the LumaKey from
                // the stages above) must feed from the Blur instead.
                self.selected = vec![tf];
                next = 30;
            }
            35 => {
                let Some(bl) = newest(self, &|k| matches!(k, NodeKind::Blur(_))) else {
                    fail("typing \"blur\" + Enter added no Blur")
                };
                let lk = newest(self, &|k| matches!(k, NodeKind::LumaKey(_))).unwrap();
                if self.egraph.input_of(bl, 0) != Some(tf) {
                    fail(&format!(
                        "the selected Transform should feed the new Blur (input 0 = {:?})",
                        self.egraph.input_of(bl, 0)
                    ));
                }
                for (id, what) in [(gr, "the Grade"), (lk, "the LumaKey")] {
                    if self.egraph.input_of(id, 0) != Some(bl) {
                        fail(&format!(
                            "{what} fed off the Transform should now feed off the new Blur \
                             (input 0 = {:?})",
                            self.egraph.input_of(id, 0)
                        ));
                    }
                }
                // Nothing selected for the last add: this one is about the
                // SEARCH, and an unwired arrival keeps it to that.
                self.selected.clear();
                next = 40;
            }
            46 => {
                // "grd" is in no name as a run of letters: matching it at
                // all is the fuzzy pass. The list is [Grid, Gradient,
                // Grade] (equal scores, catalog order), so one ↓ before
                // Enter must land the SECOND row.
                if newest(self, &|k| matches!(k, NodeKind::Grid(_))).is_some() {
                    fail("↓ then Enter took the FIRST hit (Grid), not the second");
                }
                if newest(self, &|k| matches!(k, NodeKind::Gradient(_))).is_none() {
                    fail(
                        "a fuzzy \"grd\" + down + Enter should have added a Gradient (the \
                         second hit: is the catalog still listing Grid, Gradient first?)",
                    );
                }
                println!(
                    "TAB TEST PASS: search opened, filtered, Enter added LumaKey; nothing \
                     selected arrives unwired, one selected feeds input 0 AND the new node \
                     takes its place downstream, two selected feed a Blend first-as-fg, \
                     a fuzzy query matches by subsequence and ↓ walks the hits"
                );
                std::process::exit(0);
            }
            _ => {}
        }
        self.test.tab = Some((next, frames));
    }
}
