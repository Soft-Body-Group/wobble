//! `--test-magnet`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    /// --test-magnet driver: drop a fresh Grade just under Read A (their
    /// sockets land inside the magnet's reach), hover the graph with shift
    /// held, and expect the pair to wire itself up.
    pub(crate) fn drive_test_magnet(&mut self, raw: &mut egui::RawInput) {
        let Some((stage, mut frames, gid)) = self.test_magnet else { return };
        frames += 1;
        if frames > 600 {
            println!("MAGNET TEST FAIL: timeout at stage {stage}");
            std::process::exit(1);
        }
        let mut next = stage;
        match stage {
            0 if frames >= 6 => {
                let Some(ra) = self.egraph.node(0).map(|n| n.pos) else { return };
                let id = self.egraph.add(
                    NodeKind::Grade(Default::default()),
                    ra + egui::vec2(0.0, 100.0),
                );
                self.test_magnet = Some((1, frames, id));
                return;
            }
            1 | 2 | 3 => {
                // Hover the graph panel with shift down; the magnet fires on
                // hovered() + shift.
                if let Some((_, r)) = self.node_rects.iter().find(|(i, _)| *i == 0) {
                    let p = r.center();
                    raw.events.insert(0, egui::Event::ModifiersChanged(egui::Modifiers { shift: true, ..Default::default() }));
                    raw.events.push(egui::Event::PointerMoved(p));
                }
                next = stage + 1;
            }
            4 => {
                let wired = self.egraph.input_of(gid, 0);
                if wired == Some(0) {
                    println!(
                        "MAGNET TEST PASS: free input courted Read A's output and shift wired it"
                    );
                    std::process::exit(0);
                }
                println!("MAGNET TEST FAIL: grade {gid} input = {wired:?} (wanted Some(0))");
                std::process::exit(1);
            }
            _ => {}
        }
        self.test_magnet = Some((next, frames, gid));
    }
}
