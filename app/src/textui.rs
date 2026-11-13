//! `--test-text`: the Text node through the real app. A Text over the
//! plate draws its glyphs where its layout says, in its colour; an animator
//! lifts exactly the glyphs its range selects without recompiling; a key
//! on the animator's offset (a `text.a0.offset` channel) moves the
//! selection with the playhead; and S-clicks in the viewer lay down a path
//! the glyphs then follow; a Prompt wired into the words port replaces
//! the words. Waits are wall-clock.

use crate::Gfx;
use engine::graph::{NodeKind, TextParams};

pub struct TestText {
    stage: u32,
    frames: u32,
    t0: web_time::Instant,
    since: web_time::Instant,
    node: u64,
    /// Glyph quads before the animator (canvas px).
    base: Vec<[[f32; 2]; 4]>,
    topo: Option<u64>,
    click: [egui::Pos2; 2],
}

impl Default for TestText {
    fn default() -> Self {
        Self {
            stage: 0,
            frames: 0,
            t0: web_time::Instant::now(),
            since: web_time::Instant::now(),
            node: 0,
            base: Vec::new(),
            topo: None,
            click: [egui::Pos2::ZERO; 2],
        }
    }
}

fn centre(q: &[[f32; 2]; 4]) -> [f32; 2] {
    [(q[0][0] + q[2][0]) * 0.5, (q[0][1] + q[2][1]) * 0.5]
}

impl Gfx {
    pub(crate) fn drive_test_text(&mut self, raw: &mut egui::RawInput) {
        let Some(mut t) = self.test_text.take() else { return };
        t.frames += 1;
        let fail = |m: String| -> ! {
            println!("TEXT TEST FAIL: {m}");
            std::process::exit(1);
        };
        if t.t0.elapsed().as_secs() > 90 {
            fail(format!("timeout at stage {}", t.stage));
        }
        // Each gesture step gets passes of its own.
        if t.since.elapsed().as_millis() < 150 {
            self.test_text = Some(t);
            return;
        }
        let next = |t: &mut TestText, s: u32| {
            t.stage = s;
            t.since = web_time::Instant::now();
        };
        let pk = self.proxy.min(1.0);
        let quads = |s: &Self| s.comp.text_quads(t.node).map(|q| q.to_vec()).unwrap_or_default();
        // The picture is current: the playhead shown, no seek in flight.
        let settled = |s: &Self| !s.force_display && s.eval_tick() == s.timeline_pos;
        // Channel `ch` over a small box round a canvas point.
        let probe = |s: &Self, p: [f32; 2], ch: usize| -> Option<f32> {
            let (x, y) = ((p[0] * pk) as u32, (p[1] * pk) as u32);
            s.probe_output_box(x.saturating_sub(1), y.saturating_sub(1), 3, 3, ch)
        };
        let press = |raw: &mut egui::RawInput, pos: egui::Pos2, down: bool| {
            raw.events.push(egui::Event::PointerMoved(pos));
            raw.events.push(egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed: down,
                modifiers: Default::default(),
            });
        };
        let key = |raw: &mut egui::RawInput, down: bool| {
            raw.events.push(egui::Event::Key {
                key: egui::Key::S,
                physical_key: None,
                pressed: down,
                repeat: false,
                modifiers: Default::default(),
            });
        };
        let topo = |s: &Self| s.last_engine_graph.as_ref().map(|g| g.topology_hash());
        match t.stage {
            0 if t.frames >= 6 && self.playback.is_some() && self.displayed_total >= 2 && self.layout_settled() => {
                let read = self
                    .egraph
                    .nodes
                    .iter()
                    .find(|(_, n)| matches!(n.kind, NodeKind::Read { .. }))
                    .map(|(id, _)| *id)
                    .unwrap_or_else(|| fail("no Read in the default graph".into()));
                let (w, h) = self.node_res_f32(read).unwrap_or((2048.0, 1080.0));
                let p = TextParams {
                    text: "IIII".into(),
                    position: [w / pk * 0.5, h / pk * 0.5],
                    size: (h / pk) * 0.25,
                    color: [0.0, 1.0, 0.0, 1.0],
                    valign: engine::textfx::VAlign::Middle,
                    tracking: 400.0,
                    ..Default::default()
                };
                let pos = self.egraph.node(read).map(|n| n.pos).unwrap_or_default();
                let n = self.egraph.add(NodeKind::Text(p), pos + egui::vec2(0.0, 120.0));
                self.egraph.connect(n, 0, read);
                self.egraph.view = Some(n);
                self.selected = vec![n];
                self.playing = false;
                self.seek_to(5);
                t.node = n;
                next(&mut t, 1);
            }
            // Four glyphs, each solid green at its middle (an I's stem).
            1 if settled(self) && quads(self).len() == 4 => {
                let q = quads(self);
                let c = centre(&q[0]);
                let (g, r) = (probe(self, c, 1), probe(self, c, 0));
                if g.is_some_and(|g| g > 0.99) && r.is_some_and(|r| r < 0.01) {
                    t.base = q;
                    t.topo = topo(self);
                    next(&mut t, 2);
                } else if t.since.elapsed().as_secs() > 5 {
                    fail(format!("no green ink at the first glyph's middle {c:?}: g {g:?} r {r:?}"));
                }
            }
            // An animator over the first half lifts glyphs 0 and 1.
            2 => {
                if let Some(NodeKind::Text(p)) = self.egraph.node_mut(t.node).map(|n| &mut n.kind) {
                    p.animators.push(engine::textfx::TextAnimator {
                        end: 50.0,
                        position: [0.0, -150.0],
                        ..Default::default()
                    });
                }
                next(&mut t, 3);
            }
            3 if settled(self) => {
                let q = quads(self);
                let lifted = q.len() == 4
                    && (0..4).all(|i| {
                        let dy = centre(&q[i])[1] - centre(&t.base[i])[1];
                        (dy - if i < 2 { -150.0 } else { 0.0 }).abs() < 0.5
                    });
                if lifted {
                    let c = centre(&q[0]);
                    if !probe(self, c, 1).is_some_and(|g| g > 0.99) {
                        if t.since.elapsed().as_secs() > 5 {
                            fail(format!("the lifted glyph is not on screen at {c:?}"));
                        }
                    } else {
                        if topo(self) != t.topo {
                            fail("an animator recompiled the comp".into());
                        }
                        // Key the offset: -100 at 0, 0 at 5, +100 at 10.
                        if let Some(n) = self.egraph.node_mut(t.node) {
                            let mut c = crate::anim::AnimCurve::default();
                            for (f, v) in [(0u64, -100.0f32), (10, 100.0)] {
                                c.set_key(f, v);
                            }
                            n.anim.insert(crate::nodes::text::chan("a0.offset"), c);
                        }
                        self.seek_to(10);
                        next(&mut t, 4);
                    }
                } else if t.since.elapsed().as_secs() > 5 {
                    fail(format!("animator did not lift glyphs 0-1 by 150: {:?} vs {:?}", q, t.base));
                }
            }
            // At 10 the window has slid off the end: nothing lifted.
            4 if settled(self) && self.timeline_pos == 10 => {
                let q = quads(self);
                if q.len() == 4 && (0..4).all(|i| (centre(&q[i])[1] - centre(&t.base[i])[1]).abs() < 0.5) {
                    self.seek_to(5);
                    next(&mut t, 5);
                } else if t.since.elapsed().as_secs() > 5 {
                    fail("a keyed offset of 100 still lifts glyphs".into());
                }
            }
            // At 5 the offset passes 0: the first half is up again.
            5 if settled(self) && self.timeline_pos == 5 => {
                let q = quads(self);
                if q.len() == 4 && (centre(&q[0])[1] - centre(&t.base[0])[1] + 150.0).abs() < 0.5 {
                    // Two path points, right of the text, one above the other.
                    let a = [t.base[3][2][0] + 100.0, t.base[0][0][1] - 100.0];
                    let b = [a[0], a[1] + 500.0];
                    t.click = [
                        self.node_point_on_screen(t.node, (a[0] * pk, a[1] * pk)),
                        self.node_point_on_screen(t.node, (b[0] * pk, b[1] * pk)),
                    ];
                    key(raw, true);
                    next(&mut t, 6);
                } else if t.since.elapsed().as_secs() > 5 {
                    fail(format!("the key at 5 did not bring the lift back: {q:?}"));
                }
            }
            6 => {
                press(raw, t.click[0], true);
                next(&mut t, 7);
            }
            7 => {
                press(raw, t.click[0], false);
                next(&mut t, 8);
            }
            8 => {
                press(raw, t.click[1], true);
                next(&mut t, 9);
            }
            9 => {
                press(raw, t.click[1], false);
                next(&mut t, 10);
            }
            10 => {
                key(raw, false);
                next(&mut t, 11);
            }
            // The path took both clicks, and the glyphs now run down it.
            11 if settled(self) => {
                let pts = match self.egraph.node(t.node).map(|n| &n.kind) {
                    Some(NodeKind::Text(p)) => p.path.points.clone(),
                    _ => Vec::new(),
                };
                if pts.len() != 2 {
                    fail(format!("two S-clicks made {} path points", pts.len()));
                }
                let q = quads(self);
                let c: Vec<[f32; 2]> = q.iter().map(centre).collect();
                let down = c.len() == 4 && c.windows(2).all(|w| w[1][1] > w[0][1] + 20.0);
                let on = c.iter().all(|p| (p[0] - pts[0].p[0]).abs() < 300.0);
                if down && on {
                    // A Prompt wired into the words port says what it says.
                    let pos = self.egraph.node(t.node).map(|n| n.pos).unwrap_or_default();
                    let pr = self.egraph.add(
                        NodeKind::Prompt(engine::graph::PromptParams { text: "II".into() }),
                        pos + egui::vec2(-160.0, 0.0),
                    );
                    self.egraph.connect(t.node, engine::nodes::text::WORDS_INPUT, pr);
                    next(&mut t, 12);
                } else if t.since.elapsed().as_secs() > 5 {
                    fail(format!("the glyphs do not follow the path {pts:?}: {c:?}"));
                }
            }
            12 if settled(self) => {
                if quads(self).len() == 2 {
                    println!("TEXT TEST PASS: path, animator, key and a wired prompt");
                    std::process::exit(0);
                } else if t.since.elapsed().as_secs() > 5 {
                    fail(format!("a wired Prompt of two letters drew {} glyphs", quads(self).len()));
                }
            }
            _ => {}
        }
        self.test_text = Some(t);
    }
}
