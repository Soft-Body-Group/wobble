//! `--test-subwob`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    /// --test-subwob driver: Read -> Grade -> Blur, the Grade and Blur
    /// selected, the ` pie held over the graph and released on "subwob".
    /// They must leave and ONE subwob node take their place — fed by the
    /// Read, viewed — with the picture on screen unchanged (a collapse is a
    /// reorganisation, never an edit). Undo brings the two nodes back, redo
    /// the subwob; Enter opens it and inside, the input is a PORTAL node (no
    /// input socket, an image output) feeding the Grade.
    pub(crate) fn drive_test_subwob(&mut self, raw: &mut egui::RawInput) {
        let Some((stage, mut frames, since)) = self.test_subwob else { return };
        frames += 1;
        // WALL CLOCK: the unpaced test loop spins thousands of frames a
        // second, and two stages wait on a render.
        if since.elapsed() > std::time::Duration::from_secs(20) {
            println!("SUBWOB TEST FAIL: timeout at stage {stage}");
            std::process::exit(1);
        }
        let fail = |m: String| -> ! {
            println!("SUBWOB TEST FAIL: {m}");
            std::process::exit(1);
        };
        let key = |raw: &mut egui::RawInput, k: egui::Key, down: bool| {
            raw.events.push(egui::Event::Key {
                key: k,
                physical_key: None,
                pressed: down,
                repeat: false,
                modifiers: egui::Modifiers::NONE,
            });
        };
        let (at, mean0, [read, grade, blur, inst]) = self.test_subwob_at;
        let settled = at.elapsed() > std::time::Duration::from_millis(900);
        let mut next = stage;
        match stage {
            0 if self.playback.is_some() && self.displayed_total >= 2 => {
                if !self.layout_settled() {
                    return;
                }
                let read = self
                    .egraph
                    .nodes
                    .iter()
                    .find(|(_, n)| matches!(n.kind, NodeKind::Read { source: 0, .. }))
                    .map(|(id, _)| *id)
                    .unwrap_or_else(|| fail("needs a Read".into()));
                let base = self.egraph.node(read).map(|n| n.pos).unwrap_or_default();
                let mut gk = NodeKind::Grade(Default::default());
                for ch in [anim::Channel::GainR, anim::Channel::GainB] {
                    anim::set(&mut gk, ch, 1.6);
                }
                let grade = self.egraph.add(gk, base + egui::vec2(0.0, 110.0));
                let blur = self.egraph.add(
                    NodeKind::Blur(engine::graph::BlurParams { size: 6.0, ..Default::default() }),
                    base + egui::vec2(0.0, 220.0),
                );
                self.egraph.edges.push((grade, 0, read));
                self.egraph.edges.push((blur, 0, grade));
                self.egraph.view = Some(blur);
                self.egraph.view_active_b = false;
                self.playing = false;
                self.playback_dirty = true;
                self.selected = vec![grade, blur];
                self.test_subwob_at = (web_time::Instant::now(), [0.0; 3], [read, grade, blur, 0]);
                next = 1;
            }
            // The picture before, once the new chain has rendered.
            1 if settled => {
                let Some(m) = self.output_mean_rgb() else { return };
                self.test_subwob_at.1 = m;
                next = 2;
            }
            // Hover the graph (graph keys and the pie need it under the
            // pointer), then hold ` for the ops pie.
            2 => {
                let Some((_, r)) = self.node_rects.iter().find(|(id, _)| *id == read) else {
                    return;
                };
                raw.events.push(egui::Event::PointerMoved(r.center()));
                next = 3;
            }
            3 if frames > 2 => {
                key(raw, egui::Key::Backtick, true);
                next = 4;
            }
            4 => {
                let Some(p) = self.node_editor.test_ops_slice(keymap::Act::GraphSubwob) else {
                    if frames > 30 {
                        fail("` did not open the ops pie over the graph".into());
                    }
                    return;
                };
                raw.events.push(egui::Event::PointerMoved(p));
                next = 5;
            }
            // Two frames for the spring to land the slice under the pointer.
            5 | 6 => next = stage + 1,
            7 => {
                key(raw, egui::Key::Backtick, false);
                next = 8;
            }
            8 if frames > 3 => {
                if self.egraph.nodes.contains_key(&grade) || self.egraph.nodes.contains_key(&blur) {
                    fail(format!("the Grade and Blur are still in the comp ({:?})", self.tool_error));
                }
                let subs: Vec<u64> = self
                    .egraph
                    .nodes
                    .iter()
                    .filter(|(_, n)| matches!(&n.kind,
                        NodeKind::Custom { type_id, .. } if type_id.starts_with(tools::LOCAL_PREFIX)))
                    .map(|(id, _)| *id)
                    .collect();
                let [inst] = subs[..] else {
                    fail(format!("{} subwob nodes, want exactly one", subs.len()))
                };
                if self.egraph.input_of(inst, 0) != Some(read) {
                    fail("the subwob is not fed by the Read".into());
                }
                if self.egraph.active_view() != Some(inst) {
                    fail("the viewer is not on the subwob".into());
                }
                if self.egraph.subwobs.len() != 1 {
                    fail(format!("{} definitions in the comp, want 1", self.egraph.subwobs.len()));
                }
                self.test_subwob_at.0 = web_time::Instant::now();
                self.test_subwob_at.2[3] = inst;
                next = 9;
            }
            // The same picture: a collapse reorganises, it never edits.
            9 if settled => {
                let Some(m) = self.output_mean_rgb() else { return };
                if (0..3).any(|c| (m[c] - mean0[c]).abs() > 1.5 / 255.0) {
                    fail(format!("the picture changed across the collapse: {mean0:?} -> {m:?}"));
                }
                // Undo: the two nodes are back and the subwob is gone; redo
                // puts it back.
                self.undo();
                let back = self.egraph.nodes.contains_key(&grade)
                    && self.egraph.nodes.contains_key(&blur)
                    && !self.egraph.nodes.contains_key(&inst);
                if !back {
                    fail("undo did not bring the Grade and Blur back".into());
                }
                self.redo();
                if !self.egraph.nodes.contains_key(&inst) || self.egraph.nodes.contains_key(&grade) {
                    fail("redo did not bring the subwob back".into());
                }
                self.selected = vec![inst];
                next = 10;
            }
            // Enter opens it — over the graph: the pie's slice can sit past
            // the pane's edge, so bring the pointer back to the graph first.
            10 if frames > 2 => {
                raw.events.push(egui::Event::PointerMoved(self.graph_panel_rect.center()));
                key(raw, egui::Key::Enter, true);
                key(raw, egui::Key::Enter, false);
                next = 11;
            }
            11 if frames > 3 => {
                if self.dive.len() != 1 {
                    fail("Enter over the graph did not open the subwob".into());
                }
                let ports = tools::body_ports(&self.egraph);
                let [(portal, engine::graph::PortKind::Image)] = ports[..] else {
                    fail(format!("inside: ports {ports:?}, want one image input"))
                };
                let pk = &self.egraph.nodes[&portal].kind;
                if pk.input_count() != 0 || pk.output_kind() != engine::graph::PortKind::Image {
                    fail("the input node is not a portal (it has an input socket, or no image out)".into());
                }
                if self.egraph.input_of(grade, 0) != Some(portal) {
                    fail("inside, the Grade is not fed by the input node".into());
                }
                if let Err(e) = self.dive_out() {
                    fail(format!("dive out: {e}"));
                }
                if !self.dive.is_empty() || !self.egraph.nodes.contains_key(&inst) {
                    fail("stepping out did not come back to the comp".into());
                }
                println!(
                    "SUBWOB TEST PASS: the ` pie's subwob collapsed Grade+Blur into one node fed by the Read, picture unchanged ({:.4} {:.4} {:.4}); undo/redo carry it; inside, the input is a portal node feeding the Grade",
                    mean0[0], mean0[1], mean0[2]
                );
                std::process::exit(0);
            }
            _ => {}
        }
        if next != stage {
            self.test_subwob = Some((next, 0, web_time::Instant::now()));
        } else {
            self.test_subwob = Some((stage, frames, since));
        }
    }
}
