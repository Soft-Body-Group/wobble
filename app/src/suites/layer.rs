//! `--test-layer`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    pub(crate) fn drive_test_layer(&mut self, raw: &mut egui::RawInput) {
        let Some((stage, mut frames, lid)) = self.test_layer else { return };
        frames += 1;
        if frames > 900 {
            println!("LAYER TEST FAIL: timeout at stage {stage}");
            std::process::exit(1);
        }
        let press = |raw: &mut egui::RawInput, pos: egui::Pos2, down: bool| {
            raw.events.push(egui::Event::PointerMoved(pos));
            raw.events.push(egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed: down,
                modifiers: Default::default(),
            });
        };
        let mv = |raw: &mut egui::RawInput, pos: egui::Pos2| {
            raw.events.push(egui::Event::PointerMoved(pos));
        };
        let layers = |s: &Self| -> Vec<engine::graph::LayerItem> {
            match s.egraph.node(lid).map(|n| &n.kind) {
                Some(NodeKind::Layer(p)) => p.layers.clone(),
                _ => Vec::new(),
            }
        };
        let mut next = stage;
        match stage {
            // Wire the two plates into a fresh stack and put it on screen.
            0 if self.playback.is_some() && self.displayed_total >= 3 => {
                let reads: Vec<u64> = self
                    .egraph
                    .nodes
                    .iter()
                    .filter(|(_, n)| matches!(n.kind, NodeKind::Read { .. }))
                    .map(|(id, _)| *id)
                    .collect();
                if reads.len() < 2 {
                    println!("LAYER TEST FAIL: needs two Reads, found {}", reads.len());
                    std::process::exit(1);
                }
                let id = self.egraph.add(NodeKind::layer(), egui::Pos2::new(360.0, 120.0));
                self.egraph.connect(id, 0, reads[0]);
                self.egraph.connect(id, 1, reads[1]);
                self.egraph.view = Some(id);
                self.selected = vec![id];
                self.prop_order = vec![id];
                self.props_visible = true;
                self.bottom = BottomTab::Properties;
                self.test_layer = Some((1, 0, id));
                return;
            }
            // Slide layer 2's bar to the right: its START moves with it, and
            // nothing else about the stack does.
            1 if frames > 20 && self.layer_bars.len() == 2 && self.layout_settled() => {
                self.layer_grab = self.layer_bars[1].center();
                press(raw, self.layer_grab, true);
                next = 2;
            }
            2 if frames > 2 => {
                // Two moves: egui needs one to call it a drag at all, and the
                // second is the one the widget acts on.
                mv(raw, self.layer_grab + egui::vec2(20.0, 0.0));
                next = 3;
            }
            3 if frames > 2 => {
                mv(raw, self.layer_grab + egui::vec2(60.0, 0.0));
                next = 4;
            }
            4 if frames > 2 => {
                press(raw, self.layer_grab + egui::vec2(60.0, 0.0), false);
                next = 5;
            }
            5 if frames > 4 => {
                let l = layers(self);
                if l.len() != 2 || l[1].start <= 0 {
                    println!(
                        "LAYER TEST FAIL: dragging a bar right left start at {:?}",
                        l.get(1).map(|x| x.start)
                    );
                    std::process::exit(1);
                }
                if l[0].start != 0 {
                    println!("LAYER TEST FAIL: the other layer moved too ({})", l[0].start);
                    std::process::exit(1);
                }
                self.layer_slid = l[1].start;
                // ALT-drag layer 1's right end: a STRETCH (AE's ctrl+alt) —
                // the footage still plays whole, it just takes longer.
                // Alt goes down before the press and stays down (egui's
                // modifiers are sticky) until the release has landed.
                self.layer_grab =
                    egui::pos2(self.layer_bars[0].max.x - 2.0, self.layer_bars[0].center().y);
                raw.events.insert(0, egui::Event::ModifiersChanged(egui::Modifiers { alt: true, ..Default::default() }));
                press(raw, self.layer_grab, true);
                next = 51;
            }
            51 if frames > 2 => {
                mv(raw, self.layer_grab + egui::vec2(20.0, 0.0));
                next = 52;
            }
            52 if frames > 2 => {
                mv(raw, self.layer_grab + egui::vec2(90.0, 0.0));
                next = 53;
            }
            53 if frames > 2 => {
                press(raw, self.layer_grab + egui::vec2(90.0, 0.0), false);
                next = 54;
            }
            54 if frames > 4 => {
                raw.events.insert(0, egui::Event::ModifiersChanged(Default::default()));
                let l = layers(self);
                if !(l[0].speed < 0.98) {
                    println!(
                        "LAYER TEST FAIL: stretching a bar did not slow the layer \
                         (speed {})",
                        l[0].speed
                    );
                    std::process::exit(1);
                }
                if l[0].trim.is_some() {
                    println!("LAYER TEST FAIL: a stretch trimmed the layer instead");
                    std::process::exit(1);
                }
                self.layer_speed = l[0].speed;
                // A plain drag on layer 2's LEFT end TRIMS (AE): the in
                // point moves, the footage and its rate stay put.
                self.layer_grab =
                    egui::pos2(self.layer_bars[1].min.x + 2.0, self.layer_bars[1].center().y);
                press(raw, self.layer_grab, true);
                next = 55;
            }
            55 if frames > 2 => {
                mv(raw, self.layer_grab + egui::vec2(10.0, 0.0));
                next = 56;
            }
            56 if frames > 2 => {
                mv(raw, self.layer_grab + egui::vec2(40.0, 0.0));
                next = 57;
            }
            57 if frames > 2 => {
                press(raw, self.layer_grab + egui::vec2(40.0, 0.0), false);
                next = 58;
            }
            58 if frames > 4 => {
                let l = layers(self);
                match l[1].trim {
                    Some((a, _)) if a > self.layer_slid => {}
                    t => {
                        println!(
                            "LAYER TEST FAIL: dragging an end did not trim the in point \
                             (trim {t:?}, start {})",
                            l[1].start
                        );
                        std::process::exit(1);
                    }
                }
                if l[1].start != self.layer_slid || (l[1].speed - 1.0).abs() > 1e-6 {
                    println!(
                        "LAYER TEST FAIL: a trim moved the footage (start {} speed {})",
                        l[1].start, l[1].speed
                    );
                    std::process::exit(1);
                }
                // A range set on the Load hides part of the clip: the bar
                // is the range, the clip is what a trim can reach, and a
                // trim out past the range widens the Load to show it.
                let src = self.egraph.input_of(lid, 1).unwrap();
                let clip = self.read_clip(src).unwrap();
                if clip.1 - clip.0 < 6 {
                    println!("LAYER TEST FAIL: plate too short for the range stage ({clip:?})");
                    std::process::exit(1);
                }
                if let Some(NodeKind::Read { first, last, .. }) =
                    self.egraph.node_mut(src).map(|n| &mut n.kind)
                {
                    *first = Some(clip.0 + 2);
                    *last = Some(clip.1 - 2);
                }
                let s = self.layer_src(src).unwrap();
                if s.play != (clip.0 + 2, clip.1 - 2) || s.clip != clip {
                    println!("LAYER TEST FAIL: a ranged Load reported {s:?}, clip {clip:?}");
                    std::process::exit(1);
                }
                let st = l[1].start;
                if let Some(NodeKind::Layer(p)) = self.egraph.node_mut(lid).map(|n| &mut n.kind) {
                    p.layers[1].trim = Some((clip.0 + st - 5, clip.1 + st - 1));
                }
                self.widen_layer_reads(lid);
                match self.egraph.node(src).map(|n| &n.kind) {
                    Some(NodeKind::Read { first: Some(f), last: Some(la), .. })
                        if (*f, *la) == (clip.0, clip.1 - 1) => {}
                    k => {
                        println!(
                            "LAYER TEST FAIL: a trim past the range did not widen the Load to \
                             it within the clip {clip:?}: {k:?}"
                        );
                        std::process::exit(1);
                    }
                }
                if let Some(NodeKind::Read { first, last, .. }) =
                    self.egraph.node_mut(src).map(|n| &mut n.kind)
                {
                    *first = Some(clip.0);
                    *last = Some(clip.1);
                }
                self.playback_dirty = true;
                // Untrim it again: the pick below clicks the playhead's
                // frame, which the trim just took this layer off.
                if let Some(NodeKind::Layer(p)) = self.egraph.node_mut(lid).map(|n| &mut n.kind) {
                    p.layers[1].trim = None;
                }
                // A vertical drag must NOT restack (bars move left and right only):
                // drag layer 1's bar DOWN past layer 2's centre.
                self.layer_grab = self.layer_bars[0].center();
                press(raw, self.layer_grab, true);
                next = 6;
            }
            6 if frames > 2 => {
                let d = self.layer_bars[1].center().y - self.layer_grab.y;
                mv(raw, self.layer_grab + egui::vec2(0.0, d * 0.5));
                next = 7;
            }
            7 if frames > 2 => {
                let d = self.layer_bars[1].center().y - self.layer_grab.y;
                mv(raw, self.layer_grab + egui::vec2(0.0, d));
                next = 8;
            }
            8 if frames > 2 => {
                press(raw, self.layer_grab, false);
                next = 9;
            }
            9 if frames > 4 => {
                let l = layers(self);
                // The slid layer is still layer 2, on the wire it came in on.
                if l[1].start != self.layer_slid || self.egraph.input_of(lid, 1) != Some(self.layer_wire1) {
                    println!(
                        "LAYER TEST FAIL: a vertical drag restacked the layers (layer 2 start {},                          wanted {}; input {:?}, wanted {:?})",
                        l[1].start,
                        self.layer_slid,
                        self.egraph.input_of(lid, 1),
                        Some(self.layer_wire1)
                    );
                    std::process::exit(1);
                }
                // Finally the OTHER way into a layer: ctrl-click ON the
                // picture. Nothing upstream places these plates, so the
                // viewer's pick lands on the topmost layer itself.
                self.layer_sel.insert(lid, 1);
                // Well inside the frame but away from the pivot (the image
                // centre) and from every handle that orbits it.
                self.layer_grab = self.image_rect.min
                    + egui::vec2(
                        self.image_rect.width() * 0.2,
                        self.image_rect.height() * 0.2,
                    );
                raw.events.insert(0, egui::Event::ModifiersChanged(egui::Modifiers { ctrl: true, ..Default::default() }));
                raw.events.push(egui::Event::PointerMoved(self.layer_grab));
                raw.events.push(egui::Event::PointerButton {
                    pos: self.layer_grab,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers { ctrl: true, ..Default::default() },
                });
                next = 111;
            }
            111 if frames > 2 => {
                raw.events.insert(0, egui::Event::ModifiersChanged(egui::Modifiers { ctrl: true, ..Default::default() }));
                raw.events.push(egui::Event::PointerButton {
                    pos: self.layer_grab,
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers: egui::Modifiers { ctrl: true, ..Default::default() },
                });
                next = 112;
            }
            112 if frames > 4 => {
                // Pop the stack out into its own window and check it really
                // laid out — a window that draws nothing is the failure this
                // catches, and it looks like nothing at all.
                self.layer_popout = Some(lid);
                next = 113;
            }
            113 if frames > 12 => {
                if self.last_layer_window.is_none() {
                    println!("LAYER TEST FAIL: the popped-out window never laid out");
                    std::process::exit(1);
                }
                if self.last_layer_window.map(|r: egui::Rect| r.width() < 100.0) == Some(true) {
                    println!("LAYER TEST FAIL: the popped-out window came up empty");
                    std::process::exit(1);
                }
                self.layer_popout = None;
                next = 114;
            }
            114 if frames > 4 => {
                let sel = self.layer_sel.get(&lid).copied().unwrap_or(9);
                if sel != 0 {
                    println!(
                        "LAYER TEST FAIL: ctrl-clicking the image picked layer {} , \
                         wanted the top one",
                        sel + 1
                    );
                    std::process::exit(1);
                }
                // Last: the claim the whole per-layer timebase rests on —
                // ONE plate on two layers at two starts is two live readers,
                // not one frame drawn twice.
                let top = self.egraph.input_of(lid, 0).unwrap();
                self.egraph.connect(lid, 1, top);
                next = 12;
            }
            12 if frames > 20 => {
                let top = self.egraph.input_of(lid, 0).unwrap();
                // Only the instances reached THROUGH the stack: the startup
                // graph's own Blend is a second root, and it reaches the
                // same plate on the plain root timebase.
                let want: Vec<u64> = (0..2)
                    .map(|i| {
                        engine::graph::input_time_key(
                            engine::graph::TIME_KEY_ROOT,
                            lid,
                            i,
                        )
                    })
                    .collect();
                let mut slots: Vec<usize> = self
                    .reader_slots
                    .iter()
                    .filter(|(i, _)| i.node == top && want.contains(&i.key))
                    .map(|(_, s)| *s)
                    .collect();
                slots.sort_unstable();
                slots.dedup();
                if slots.len() != 2 {
                    println!(
                        "LAYER TEST FAIL: one plate on two layers landed on {} reader \
                         slot(s), wanted 2",
                        slots.len()
                    );
                    std::process::exit(1);
                }
                self.layer_slots = slots.len();
                // Then the per-layer outputs: view layer 1's own output,
                // with a smaller picture on its wire than the stack's canvas.
                let small = self.egraph.add(
                    NodeKind::Reformat(engine::graph::ReformatParams {
                        size: (640, 360),
                        ..Default::default()
                    }),
                    egui::Pos2::ZERO,
                );
                self.egraph.connect(small, 0, top);
                self.egraph.connect(lid, 0, small);
                if let Some(NodeKind::Layer(p)) = self.egraph.node_mut(lid).map(|n| &mut n.kind) {
                    p.outputs = true;
                }
                self.egraph.view = Some(lid);
                self.egraph.view_out = 1;
                self.egraph.view_active_b = false;
                next = 13;
            }
            // Layer 1 alone is a one-layer stack on the layer's wire, read on
            // its OWN timebase. Waited for on the wall clock.
            13 => {
                // The wall clock is this stage's budget, not the frame count.
                frames = 1;
                static T0: std::sync::Mutex<Option<std::time::Instant>> = std::sync::Mutex::new(None);
                let t0 = *T0.lock().unwrap().get_or_insert_with(std::time::Instant::now);
                let small = self.egraph.input_of(lid, 0).unwrap();
                let top = self.egraph.input_of(small, 0).unwrap();
                let lowered = self.last_engine_graph.as_ref().and_then(|g| {
                    let v = g.view?;
                    match g.nodes.get(&v) {
                        Some(NodeKind::Layer(p)) if v != lid && p.layers.len() == 1 => {
                            (g.input_of(v, 0) == Some(small)).then_some(v)
                        }
                        _ => None,
                    }
                });
                let read = lowered.is_some_and(|v| {
                    let k = engine::graph::input_time_key(engine::graph::TIME_KEY_ROOT, v, 0);
                    self.reader_slots.iter().any(|(i, _)| i.node == top && i.key == k)
                });
                // The viewer takes the LAYER's format, not the stack's.
                let want = self.proxy_dims((640, 360));
                if read && self.desired_output() != Some(want) {
                    println!(
                        "LAYER TEST FAIL: layer 1's output sized the viewer {:?}, wanted its \
                         layer's {want:?}",
                        self.desired_output()
                    );
                    std::process::exit(1);
                }
                if read {
                    // Off: the wire (here the viewer) on the layer's output
                    // falls back to the comp rather than staying on a ghost.
                    if let Some(NodeKind::Layer(p)) = self.egraph.node_mut(lid).map(|n| &mut n.kind) {
                        p.outputs = false;
                    }
                    layerui::prune_outputs(&mut self.egraph, lid);
                    if self.egraph.view_out != 0 {
                        println!("LAYER TEST FAIL: switching layer outputs off left the viewer on output {}", self.egraph.view_out);
                        std::process::exit(1);
                    }
                    // Back on the comp, the viewer takes the stack's canvas.
                    if self.desired_output() == Some(want) {
                        println!("LAYER TEST FAIL: the comp took a layer's format");
                        std::process::exit(1);
                    }
                    println!(
                        "LAYER TEST PASS: bar drag slid a layer to start {}, stretching one \
                         retimed it to {:.2}x, a vertical drag left the stack alone, a ctrl-click on the \
                         picture selected a layer, the pop-out window laid out at \
                         {:.0}x{:.0}, one plate on two layers decoded into {} reader slots, \
                         and layer 1's own output viewed it alone on its own timebase at \
                         its own 640x360 format",
                        self.layer_slid,
                        self.layer_speed,
                        self.last_layer_window.map(|r| r.width()).unwrap_or(0.0),
                        self.last_layer_window.map(|r| r.height()).unwrap_or(0.0),
                        self.layer_slots
                    );
                    std::process::exit(0);
                }
                if t0.elapsed() > std::time::Duration::from_secs(10) {
                    println!(
                        "LAYER TEST FAIL: layer 1's output never lowered to the layer alone \
                         (lowered {lowered:?})"
                    );
                    std::process::exit(1);
                }
            }
            _ => {}
        }
        // The wire the slid layer arrived on, captured before the restack.
        if stage == 5 {
            self.layer_wire1 = self.egraph.input_of(lid, 1).unwrap_or(0);
        }
        self.test_layer = Some((next, if next == stage { frames } else { 0 }, lid));
    }
}
