//! `--test-quickadd`: ctrl+Tab quick create through raw input.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    pub(crate) fn drive_test_quickadd(&mut self, raw: &mut egui::RawInput) {
        let Some((stage, start)) = self.test.quickadd else { return };
        if start.elapsed() > std::time::Duration::from_secs(30) {
            println!("QUICKADD TEST FAIL: timeout at stage {stage}");
            std::process::exit(1);
        }
        let key = |raw: &mut egui::RawInput, k: egui::Key, down: bool, m: egui::Modifiers| {
            raw.events.push(egui::Event::Key {
                key: k,
                physical_key: None,
                pressed: down,
                repeat: false,
                modifiers: m,
            });
        };
        let none = egui::Modifiers::NONE;
        let ctrl = egui::Modifiers::CTRL;
        let fail = |m: &str| -> ! {
            println!("QUICKADD TEST FAIL: {m}");
            std::process::exit(1);
        };
        let newest = |s: &Self, want: &dyn Fn(&NodeKind) -> bool| -> Option<u64> {
            s.egraph.nodes.iter().filter(|(_, n)| want(&n.kind)).map(|(id, _)| *id).max()
        };
        let count = |s: &Self| s.egraph.nodes.len();
        // Inside the graph panel, low and left: quick create only arms
        // over the graph.
        let pr = self.graph_panel_rect;
        let spot = egui::pos2(pr.min.x + 90.0, pr.max.y - 70.0);
        let next = stage + 1;
        match stage {
            // Let the layout settle, then put the pointer on the graph and
            // give the editor a pass to stamp its hover.
            0..=5 => {}
            6 => {
                raw.events.push(egui::Event::PointerMoved(spot));
                self.selected.clear();
            }
            7..=8 => {}
            // ctrl+Tab arms; it must not open the Tab search.
            9 => key(raw, egui::Key::Tab, true, ctrl),
            10 => key(raw, egui::Key::Tab, false, ctrl),
            11 => {
                if !keymap::quick_armed() {
                    fail("ctrl+Tab over the graph did not arm quick create");
                }
                if self.node_editor.test_search_open() {
                    fail("ctrl+Tab opened the Tab search too");
                }
                self.test.quickadd_n = count(self);
            }
            // G (ctrl released) adds a Grade, unwired: nothing selected.
            12 => {
                key(raw, egui::Key::G, true, none);
                raw.events.push(egui::Event::Text("g".into()));
            }
            13 => key(raw, egui::Key::G, false, none),
            14 => {}
            15 => {
                if keymap::quick_armed() {
                    fail("still armed after G");
                }
                if count(self) != self.test.quickadd_n + 1 {
                    fail(&format!(
                        "G should add exactly one node ({} -> {})",
                        self.test.quickadd_n,
                        count(self)
                    ));
                }
                let Some(g) = newest(self, &|k| matches!(k, NodeKind::Grade(_))) else {
                    fail("ctrl+Tab, G added no Grade")
                };
                if newest(self, &|_| true) != Some(g) {
                    fail("the node G added is not a Grade");
                }
                if self.selected != vec![g] {
                    fail("the new Grade should be selected after the add");
                }
                self.test.quickadd_n = count(self);
            }
            // Escape cancels — and is EATEN: over the graph a plain Escape
            // clears the selection, so the Grade must still be picked.
            16 => key(raw, egui::Key::Tab, true, ctrl),
            17 => key(raw, egui::Key::Tab, false, ctrl),
            18 => {
                if !keymap::quick_armed() {
                    fail("second ctrl+Tab did not arm");
                }
                key(raw, egui::Key::Escape, true, none);
            }
            19 => key(raw, egui::Key::Escape, false, none),
            20 => {}
            21 => {
                if keymap::quick_armed() {
                    fail("Escape did not cancel quick create");
                }
                if count(self) != self.test.quickadd_n {
                    fail("Escape added a node");
                }
                if self.selected.len() != 1 {
                    fail("Escape reached the graph (the selection was cleared)");
                }
            }
            // An unmapped key (J) cancels and adds nothing.
            22 => key(raw, egui::Key::Tab, true, ctrl),
            23 => key(raw, egui::Key::Tab, false, ctrl),
            24 => {
                key(raw, egui::Key::J, true, none);
                key(raw, egui::Key::J, false, none);
            }
            25 => {}
            26 => {
                if keymap::quick_armed() || count(self) != self.test.quickadd_n {
                    fail("an unmapped key should cancel and add nothing");
                }
            }
            // ctrl still held: ctrl+Tab, ctrl+B adds a Blur fed by the
            // selected Grade.
            27 => key(raw, egui::Key::Tab, true, ctrl),
            28 => {
                key(raw, egui::Key::Tab, false, ctrl);
                key(raw, egui::Key::B, true, ctrl);
            }
            29 => key(raw, egui::Key::B, false, ctrl),
            30 => {}
            31 => {
                let g = newest(self, &|k| matches!(k, NodeKind::Grade(_))).unwrap();
                let Some(b) = newest(self, &|k| matches!(k, NodeKind::Blur(_))) else {
                    fail("ctrl+Tab, ctrl+B added no Blur")
                };
                if count(self) != self.test.quickadd_n + 1 {
                    fail("ctrl+B should add exactly one node");
                }
                if self.egraph.input_of(b, 0) != Some(g) {
                    fail(&format!(
                        "the selected Grade should feed the new Blur (input 0 = {:?})",
                        self.egraph.input_of(b, 0)
                    ));
                }
                println!(
                    "QUICKADD TEST PASS: ctrl+Tab arms without opening the search, G adds a \
                     Grade, Escape and an unmapped key cancel (Escape eaten), ctrl held + B \
                     adds a Blur fed by the selection"
                );
                std::process::exit(0);
            }
            _ => {}
        }
        self.test.quickadd = Some((next, start));
    }
}
