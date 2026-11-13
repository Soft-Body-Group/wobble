//! `--test-propspie`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    pub(crate) fn drive_test_propspie(&mut self, raw: &mut egui::RawInput) {
        let Some((stage, mut frames, id)) = self.test.propspie else { return };
        frames += 1;
        if frames > 900 {
            println!("PROPSPIE TEST FAIL: timeout at stage {stage}");
            std::process::exit(1);
        }
        let key = |raw: &mut egui::RawInput, down: bool| {
            raw.events.push(egui::Event::Key {
                key: egui::Key::Backtick,
                physical_key: None,
                pressed: down,
                repeat: false,
                modifiers: Default::default(),
            });
        };
        let locked = |s: &Self| {
            s.egraph.node(id).is_some_and(|n| anim::is_locked(&n.anim, anim::Channel::GridWidth))
        };
        let mut next = stage;
        match stage {
            0 if self.playback.is_some() && self.displayed_total >= 3 => {
                let nid = self.egraph.add(
                    NodeKind::Grid(engine::graph::GridParams::default()),
                    egui::Pos2::new(300.0, 0.0),
                );
                self.prop_order = vec![nid];
                self.props_visible = true;
                self.bottom = BottomTab::Properties;
                self.playing = false;
                self.test.propspie = Some((1, 0, nid));
                return;
            }
            // Wait for the row to hold still (panels animate in).
            1 | 5 => {
                let row = widgets::rows(&self.egui_ctx)
                    .into_iter()
                    .find(|r| r.node == id && r.label == "line");
                if let Some(r) = row {
                    let (was, still) = self.test.propspie_at;
                    self.test.propspie_at = (r.rect, if was == r.rect { still + 1 } else { 0 });
                    if still >= 5 {
                        raw.events.push(egui::Event::PointerMoved(r.rect.center()));
                        next = stage + 1;
                    }
                }
            }
            2 | 6 if frames > 2 => {
                key(raw, true);
                next = stage + 1;
            }
            3 | 7 if frames > 3 => {
                if self.props_pie.is_none() {
                    println!("PROPSPIE TEST FAIL: holding the pie key over a row opened no pie");
                    std::process::exit(1);
                }
                // Slice 0 sits at 12 o'clock, on the orbit.
                let c = self.test.propspie_at.0.center();
                raw.events.push(egui::Event::PointerMoved(c - egui::vec2(0.0, 62.0)));
                next = stage + 1;
            }
            4 | 8 if frames > 2 => {
                key(raw, false);
                next = stage + 1;
            }
            9 if frames > 3 => {
                if locked(self) {
                    println!("PROPSPIE TEST FAIL: the second pick did not unlock the row");
                    std::process::exit(1);
                }
                println!(
                    "PROPSPIE TEST PASS: the pie opened on the row under the pointer, \
                     \"lock keyframes\" locked its channel, and the same pick unlocked it"
                );
                std::process::exit(0);
            }
            _ => {}
        }
        // Between the two rounds: the first pick must have locked it.
        if stage == 5 && frames == 1 && !locked(self) {
            println!("PROPSPIE TEST FAIL: \"lock keyframes\" did not lock the row");
            std::process::exit(1);
        }
        self.test.propspie = Some((next, if next == stage { frames } else { 0 }, id));
    }
}
