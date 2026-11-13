//! `--test-mswipe`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    /// --test-mswipe driver. See `Options::test_mswipe`. Default graph:
    /// Read A 0, Read B 1, Transform 2 (A's chain), Grade 3, Blend 4.
    pub(crate) fn drive_test_mswipe(&mut self, raw: &mut egui::RawInput) {
        let Some((stage, mut frames, home, drop)) = self.test_mswipe else { return };
        frames += 1;
        if frames > 1200 {
            println!("MSWIPE TEST FAIL: timeout at stage {stage}");
            std::process::exit(1);
        }
        let fail = |m: String| -> ! {
            println!("MSWIPE TEST FAIL: {m}");
            std::process::exit(1);
        };
        let (rb, tf, gr) = (1u64, 2u64, 3u64);
        let rect_of = |s: &Self, id: u64| -> Option<egui::Rect> {
            s.node_rects.iter().find(|(i, _)| *i == id).map(|(_, r)| *r)
        };
        let spline_of = |s: &Self| -> Option<u64> {
            s.egraph
                .nodes
                .iter()
                .find(|(_, n)| matches!(n.kind, NodeKind::Spline(_)))
                .map(|(id, _)| *id)
        };
        let m_key = |raw: &mut egui::RawInput, down: bool| {
            raw.events.push(egui::Event::Key {
                key: egui::Key::M,
                physical_key: None,
                pressed: down,
                repeat: false,
                modifiers: Default::default(),
            });
        };
        let stencil_of = |s: &Self| -> Option<u64> {
            s.egraph.nodes.iter().find(|(_, n)| n.ui_label == "stencil").map(|(id, _)| *id)
        };
        let b_key = |raw: &mut egui::RawInput, down: bool| {
            raw.events.push(egui::Event::Key {
                key: egui::Key::B,
                physical_key: None,
                pressed: down,
                repeat: false,
                modifiers: Default::default(),
            });
        };
        let btn = |raw: &mut egui::RawInput, p: egui::Pos2, down: bool| {
            raw.events.push(egui::Event::PointerMoved(p));
            raw.events.push(egui::Event::PointerButton {
                pos: p,
                button: egui::PointerButton::Primary,
                pressed: down,
                modifiers: Default::default(),
            });
        };
        let blends = |s: &Self| -> Vec<u64> {
            s.egraph
                .nodes
                .iter()
                .filter(|(id, n)| **id != 4 && matches!(n.kind, NodeKind::Blend { .. }))
                .map(|(id, _)| *id)
                .collect()
        };
        let mut next = stage;
        // The path: Read B's centre -> the Transform's centre -> on past it.
        let path = |s: &Self, k: u32| -> Option<egui::Pos2> {
            let (a, b) = (rect_of(s, rb)?.center(), rect_of(s, tf)?.center());
            Some(if k <= 10 {
                a + (b - a) * (k as f32 / 10.0)
            } else {
                b + egui::vec2(50.0, 45.0) * ((k - 10) as f32 / 4.0)
            })
        };
        match stage {
            0 => {
                if self.layout_settled() && frames >= 8 {
                    let Some(p) = self.egraph.node(rb).map(|n| n.pos) else { return };
                    self.selected = vec![rb];
                    if let Some(r) = rect_of(self, rb) {
                        raw.events.push(egui::Event::PointerMoved(r.center()));
                    }
                    self.test_mswipe = Some((1, frames, p, drop));
                    return;
                }
            }
            1 => {
                // Hold M, then press on Read B.
                m_key(raw, true);
                let Some(p) = path(self, 0) else { return };
                btn(raw, p, true);
                next = 100;
            }
            // Drag along the path, one step a frame (100 -> k = 1..=14).
            100..=113 => {
                let k = stage - 99;
                let Some(p) = path(self, k) else { return };
                raw.events.push(egui::Event::PointerMoved(p));
                if k == 14 {
                    self.test_mswipe = Some((114, frames, home, p));
                    return;
                }
                next = stage + 1;
            }
            114 => {
                btn(raw, drop, false);
                next = 115;
            }
            115 => {
                m_key(raw, false);
                next = 116;
            }
            116 | 117 => next = stage + 1,
            118 => {
                let made = blends(self);
                let [b] = made[..] else {
                    fail(format!("the swipe should make exactly one Blend, made {made:?}"))
                };
                if self.egraph.input_of(b, 0) != Some(rb) || self.egraph.input_of(b, 1) != Some(tf) {
                    fail(format!(
                        "Blend inputs {:?}/{:?}, want Read B over the Transform",
                        self.egraph.input_of(b, 0),
                        self.egraph.input_of(b, 1)
                    ));
                }
                if self.egraph.input_of(gr, 0) != Some(b) {
                    fail("the Blend should take the Transform's place above the Grade".into());
                }
                let back = self.egraph.node(rb).map(|n| n.pos);
                if back != Some(home) {
                    fail(format!("Read B should go home to {home:?}, is at {back:?}"));
                }
                // Under the hand where it was let go. (The rect, not its
                // centre: a merge bar's slab is offset from its anchor by
                // its porch, while the anchor itself sits on the pointer.)
                let Some(r) = rect_of(self, b) else { fail("the Blend is not drawn".into()) };
                if !r.contains(drop) || (r.center().y - drop.y).abs() > 3.0 {
                    fail(format!(
                        "the Blend should land where it was let go ({drop:?}), its rect is {r:?}"
                    ));
                }
                if self.node_editor.test_region_menu_open() {
                    fail("a spent M hold must not open the region popup".into());
                }
                // A plain TAP still opens it.
                raw.events.push(egui::Event::PointerMoved(
                    self.graph_panel_rect.left_top() + egui::vec2(30.0, 300.0),
                ));
                m_key(raw, true);
                m_key(raw, false);
                next = 119;
            }
            119 => {
                if !self.node_editor.test_region_menu_open() {
                    fail("a tap of M should still open the region popup".into());
                }
                raw.events.push(egui::Event::Key {
                    key: egui::Key::Escape,
                    physical_key: None,
                    pressed: true,
                    repeat: false,
                    modifiers: Default::default(),
                });
                next = 200;
            }
            // --- I and O: a Spline (a matte) swiped over the Transform
            // isolates it (mask), then over Read A cuts it out (stencil) ---
            200 => {
                let Some(tp) = self.egraph.node(tf).map(|n| n.pos) else { return };
                let sp = self.egraph.add(
                    NodeKind::Spline(Default::default()),
                    tp + egui::vec2(120.0, -40.0),
                );
                self.selected = vec![sp];
                let home = self.egraph.node(sp).map(|n| n.pos).unwrap_or_default();
                self.test_mswipe = Some((201, frames, home, drop));
                return;
            }
            201 | 202 => next = stage + 1, // let the new node be drawn
            // 203 swipes onto the Transform with I; 303 onto Read A with O.
            203 | 303 => {
                let key = if stage == 203 { egui::Key::I } else { egui::Key::O };
                let Some(sp) = spline_of(self) else { fail("the Spline vanished".into()) };
                let Some(r) = rect_of(self, sp) else { return };
                raw.events.push(egui::Event::Key {
                    key,
                    physical_key: None,
                    pressed: true,
                    repeat: false,
                    modifiers: Default::default(),
                });
                self.selected = vec![sp];
                btn(raw, r.center(), true);
                next = stage + 1;
            }
            204..=217 | 304..=317 => {
                let base = if stage < 300 { 203 } else { 303 };
                let k = stage - base;
                let target = if stage < 300 { tf } else { 0u64 };
                let Some(sp) = spline_of(self) else { return };
                // The spline goes home after the swipe, so aim from HOME.
                let a = rect_of(self, sp).map(|r| r.center());
                let b = rect_of(self, target).map(|r| r.center());
                let (Some(a), Some(b)) = (a, b) else { return };
                let p = if k <= 10 {
                    a + (b - a) * (k as f32 / 10.0)
                } else {
                    b + egui::vec2(40.0, 30.0) * ((k - 10) as f32 / 4.0)
                };
                let p = if k > 10 { self.test_mswipe.map(|t| t.3).unwrap_or(p) + egui::vec2(10.0, 7.5) } else { p };
                raw.events.push(egui::Event::PointerMoved(p));
                self.test_mswipe = Some((stage + 1, frames, home, p));
                return;
            }
            218 | 318 => {
                btn(raw, drop, false);
                next = stage + 1;
            }
            219 | 319 => {
                let key = if stage == 219 { egui::Key::I } else { egui::Key::O };
                raw.events.push(egui::Event::Key {
                    key,
                    physical_key: None,
                    pressed: false,
                    repeat: false,
                    modifiers: Default::default(),
                });
                next = stage + 1;
            }
            220 | 221 | 320 | 321 => next = stage + 1,
            222 | 322 => {
                let (label, target, img_port) =
                    if stage == 222 { ("mask", tf, 0usize) } else { ("stencil", 0u64, 1usize) };
                let Some(sp) = spline_of(self) else { fail("the Spline vanished".into()) };
                let made: Vec<u64> = self
                    .egraph
                    .nodes
                    .iter()
                    .filter(|(_, n)| n.ui_label == label)
                    .map(|(id, _)| *id)
                    .collect();
                let [b] = made[..] else {
                    fail(format!("holding the {label} key should make ONE {label} node, made {made:?}"))
                };
                let ok = matches!(
                    self.egraph.node(b).map(|n| &n.kind),
                    Some(NodeKind::Blend { op: engine::graph::BlendOp::Replace, .. })
                ) && self.egraph.input_of(b, img_port) == Some(target)
                    && self.egraph.input_of(b, 1 - img_port).is_none()
                    && self.egraph.input_of(b, 2) == Some(sp);
                if !ok {
                    fail(format!(
                        "{label}: inputs {:?}/{:?}/{:?}, want the image on {img_port}, the \
                         other empty, the Spline on the matte input",
                        self.egraph.input_of(b, 0),
                        self.egraph.input_of(b, 1),
                        self.egraph.input_of(b, 2)
                    ));
                }
                // It took the image's place downstream.
                let took = self.egraph.edges.iter().any(|(d, _, s)| *s == b && *d != b);
                if !took {
                    fail(format!("the {label} node should take its image's place downstream"));
                }
                let back = self.egraph.node(sp).map(|n| n.pos);
                if stage == 222 && back != Some(home) {
                    fail(format!("the Spline should go home to {home:?}, is at {back:?}"));
                }
                if stage == 222 {
                    next = 303;
                } else {
                    next = 400;
                }
            }
            // --- B: the same blend swipe, hold-only. Read B over the stencil ---
            400 => {
                let Some(r) = rect_of(self, rb) else { return };
                let Some(p) = self.egraph.node(rb).map(|n| n.pos) else { return };
                b_key(raw, true);
                self.selected = vec![rb];
                btn(raw, r.center(), true);
                self.test_mswipe = Some((401, frames, p, drop));
                return;
            }
            401..=414 => {
                let k = stage - 400;
                let a = rect_of(self, rb).map(|r| r.center());
                let b = stencil_of(self).and_then(|t| rect_of(self, t)).map(|r| r.center());
                let (Some(a), Some(b)) = (a, b) else { return };
                let p = if k <= 10 {
                    a + (b - a) * (k as f32 / 10.0)
                } else {
                    drop + egui::vec2(10.0, 7.5)
                };
                raw.events.push(egui::Event::PointerMoved(p));
                self.test_mswipe = Some((stage + 1, frames, home, p));
                return;
            }
            415 => {
                btn(raw, drop, false);
                next = 416;
            }
            416 => {
                b_key(raw, false);
                next = 417;
            }
            417 | 418 => next = stage + 1,
            419 => {
                let Some(st) = stencil_of(self) else { fail("the stencil vanished".into()) };
                let made: Vec<u64> = self
                    .egraph
                    .nodes
                    .iter()
                    .filter(|(id, n)| {
                        matches!(n.kind, NodeKind::Blend { .. })
                            && self.egraph.input_of(**id, 0) == Some(rb)
                            && self.egraph.input_of(**id, 1) == Some(st)
                    })
                    .map(|(id, _)| *id)
                    .collect();
                let [b] = made[..] else {
                    fail(format!("holding B should blend Read B over the stencil, made {made:?}"))
                };
                if !self.egraph.edges.iter().any(|(d, _, s)| *s == b && *d != b) {
                    fail("the B Blend should take the stencil's place downstream".into());
                }
                let back = self.egraph.node(rb).map(|n| n.pos);
                if back != Some(home) {
                    fail(format!("Read B should go home to {home:?} after the B swipe, is at {back:?}"));
                }
                if self.node_editor.test_region_menu_open() {
                    fail("a B swipe must not open the region popup".into());
                }
                // A plain tap of B over the graph does nothing.
                raw.events.push(egui::Event::PointerMoved(
                    self.graph_panel_rect.left_top() + egui::vec2(30.0, 300.0),
                ));
                b_key(raw, true);
                b_key(raw, false);
                let n = self.egraph.nodes.len() as f32;
                self.test_mswipe = Some((420, frames, egui::pos2(n, 0.0), drop));
                return;
            }
            420 | 421 => next = stage + 1,
            422 => {
                if self.node_editor.test_region_menu_open() {
                    fail("a tap of B must not open the region popup".into());
                }
                if self.egraph.nodes.len() as f32 != home.x {
                    fail("a tap of B must not add or remove nodes".into());
                }
                println!(
                    "MSWIPE TEST PASS: holding M and swiping Read B across the Transform \
                     made a Blend (B over, Transform under) that took the Transform's \
                     place above the Grade, Read B went home, the Blend landed where it \
                     was let go, the spent M opened no popup and a tap of M still does; \
                     I swiped a Spline over the Transform into a MASK and O over Read A \
                     into a STENCIL (Replace Blends with the matte on its matte input, \
                     each taking its image's place downstream, the Spline going home); \
                     holding B made the same Blend of Read B over the stencil, and a tap of B \
                     does nothing"
                );
                std::process::exit(0);
            }
            _ => {}
        }
        self.test_mswipe = Some((next, frames, home, drop));
    }
}
