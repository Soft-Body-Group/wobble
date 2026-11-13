//! `--test-region`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    /// --test-region: regions and the hidden-wire layer in the LIVE app.
    ///
    /// The model (adopt/prune/active/pie maths, flags-follow-edges) is
    /// unit-tested in `regions`; what only a live run shows is the wiring —
    /// the view centre making a region ACTIVE, F3/F2 opening the source
    /// pies, a pie click spawning a chip hidden-wired to the registered
    /// source, the hidden edge reaching the engine as an ordinary edge,
    /// shift+backtick flipping patch mode through the real input path,
    /// plain ` the ops pie, and F1 the most-used-nodes pie spawning its
    /// top slice.
    pub(crate) fn drive_test_region(&mut self, raw: &mut egui::RawInput) {
        let Some((stage, mut frames)) = self.test.region else { return };
        frames += 1;
        if frames > 900 {
            println!("REGION TEST FAIL: timeout at stage {stage}");
            std::process::exit(1);
        }
        let key = |raw: &mut egui::RawInput,
                   k: egui::Key,
                   modifiers: egui::Modifiers,
                   down: bool| {
            raw.events.push(egui::Event::Key {
                key: k,
                physical_key: None,
                pressed: down,
                repeat: false,
                modifiers,
            });
        };
        let plain = egui::Modifiers::NONE;
        let shift = egui::Modifiers::SHIFT;
        let find = |g: &editor::EditorGraph, f: &dyn Fn(&NodeKind) -> bool| {
            g.nodes.iter().find(|(_, n)| f(&n.kind)).map(|(id, _)| *id)
        };
        let read_b = find(&self.egraph, &|k| {
            matches!(k, NodeKind::Read { source: 1, .. })
        });
        let chip = find(&self.egraph, &|k| matches!(k, NodeKind::Dot { .. }));
        let mut next = stage;
        match stage {
            // Build the fixture: a region over the A branch, the B Read
            // registered as its common source, the view centred inside it.
            0 if frames >= 10 => {
                let (Some(ra), Some(tf), Some(rb)) = (
                    find(&self.egraph, &|k| {
                        matches!(k, NodeKind::Read { source: 0, .. })
                    }),
                    find(&self.egraph, &|k| matches!(k, NodeKind::Transform(_))),
                    read_b,
                ) else {
                    return;
                };
                let rid = regions::create(&mut self.egraph, &[ra, tf]);
                if let Some(r) = self.egraph.regions.iter_mut().find(|r| r.id == rid) {
                    r.name = "FG".into();
                    r.sources.push(rb);
                }
                if let Some(b) = self
                    .egraph
                    .regions
                    .iter()
                    .find(|r| r.id == rid)
                    .and_then(|r| regions::bounds(&self.egraph, r))
                {
                    self.node_editor.pan =
                        -b.center().to_vec2() * self.node_editor.zoom;
                }
                next = 1;
            }
            // Hover a node in the region (the graph pane must be under the
            // cursor for graph keys), then check the region went active.
            1 => {
                let Some(ra) = find(&self.egraph, &|k| {
                    matches!(k, NodeKind::Read { source: 0, .. })
                }) else {
                    return;
                };
                let Some((_, r)) =
                    self.node_rects.iter().find(|(id, _)| *id == ra)
                else {
                    return;
                };
                raw.events.push(egui::Event::PointerMoved(r.center()));
                next = 2;
            }
            2 => {
                if self.node_editor.active_region.is_none() {
                    println!("REGION TEST FAIL: view centred in the wash but no active region");
                    std::process::exit(1);
                }
                // HOLD F3: the REGION pie, alive while it is down.
                key(raw, egui::Key::F3, plain, true);
                next = 3;
            }
            3 => {
                if self.node_editor.test_pie_scope()
                    != Some(editor::PieScope::Region(self.node_editor.active_region))
                {
                    println!(
                        "REGION TEST FAIL: F3 opened {:?}, want the region's pie",
                        self.node_editor.test_pie_scope()
                    );
                    std::process::exit(1);
                }
                // Aim at the one source's disc while still holding.
                let Some(p) = self.node_editor.test_pie_slice(0, 1) else { return };
                raw.events.push(egui::Event::PointerMoved(p));
                next = 4;
            }
            // A couple of frames for the orbit's spring to open out.
            4 => next = 5,
            5 => next = 6,
            6 => {
                // RELEASE over the disc: that is the pick.
                key(raw, egui::Key::F3, plain, false);
                next = 7;
            }
            7 => next = 8, // settle: the spawn lands on the release frame
            8 => {
                let (Some(chip), Some(rb)) = (chip, read_b) else { return };
                if self.egraph.input_of(chip, 0) != Some(rb) {
                    println!("REGION TEST FAIL: chip not wired to the common source");
                    std::process::exit(1);
                }
                if !self.egraph.is_hidden(chip, 0) {
                    println!("REGION TEST FAIL: the chip's feed is not drawn hidden");
                    std::process::exit(1);
                }
                // The hidden edge is an ORDINARY edge to the engine: wire
                // the chip into the Transform and the engine sees B's Read
                // there, resolved through the Dot.
                let Some(tf) = find(&self.egraph, &|k| {
                    matches!(k, NodeKind::Transform(_))
                }) else {
                    return;
                };
                self.egraph.connect(tf, 0, chip);
                let eg = self.egraph.to_engine_graph();
                if eg.edges.get(&(tf, 0)).copied() != Some(rb) {
                    println!("REGION TEST FAIL: hidden edge did not reach the engine");
                    std::process::exit(1);
                }
                // GLOBAL roster: select exactly Read A and hold F2 —
                // that is the ADD pie, whose up disc registers it.
                let Some(ra) = find(&self.egraph, &|k| {
                    matches!(k, NodeKind::Read { source: 0, .. })
                }) else {
                    return;
                };
                self.selected = vec![ra];
                key(raw, egui::Key::F2, plain, true);
                next = 9;
            }
            9 => {
                if self.node_editor.test_pie_scope() != Some(editor::PieScope::Global) {
                    println!(
                        "REGION TEST FAIL: F2 opened {:?}, want the global pie",
                        self.node_editor.test_pie_scope()
                    );
                    std::process::exit(1);
                }
                // Two discs in add mode; the up one is "register as is".
                let Some(p) = self.node_editor.test_pie_slice(0, 2) else { return };
                raw.events.push(egui::Event::PointerMoved(p));
                next = 10;
            }
            10 => next = 11,
            11 => {
                key(raw, egui::Key::F2, plain, false);
                next = 12;
            }
            12 => {
                let Some(ra) = find(&self.egraph, &|k| {
                    matches!(k, NodeKind::Read { source: 0, .. })
                }) else {
                    return;
                };
                if self.egraph.global_sources != vec![ra] {
                    println!(
                        "REGION TEST FAIL: the add pie left global sources {:?}, want [{ra}]",
                        self.egraph.global_sources
                    );
                    std::process::exit(1);
                }
                // OUTSIDE every region now: pan far away, drop the
                // selection, and hold F2 over bare graph.
                self.node_editor.pan = egui::vec2(-20_000.0, 0.0);
                self.selected.clear();
                next = 13;
            }
            13 => {
                // The pointer is where a node USED to be: bare canvas in
                // the graph pane, which is what the pie needs under it.
                if self.node_editor.active_region.is_some() {
                    println!("REGION TEST FAIL: still inside a region after panning away");
                    std::process::exit(1);
                }
                key(raw, egui::Key::F2, plain, true);
                next = 14;
            }
            14 => {
                if self.node_editor.test_pie_scope() != Some(editor::PieScope::Global) {
                    println!("REGION TEST FAIL: F2 did not open the global pie outside a region");
                    std::process::exit(1);
                }
                let Some(p) = self.node_editor.test_pie_slice(0, 1) else { return };
                raw.events.push(egui::Event::PointerMoved(p));
                next = 15;
            }
            15 => next = 16,
            16 => {
                key(raw, egui::Key::F2, plain, false);
                next = 17;
            }
            17 => next = 18,
            18 => {
                let Some(ra) = find(&self.egraph, &|k| {
                    matches!(k, NodeKind::Read { source: 0, .. })
                }) else {
                    return;
                };
                let global_chip = self.egraph.nodes.iter().any(|(id, n)| {
                    matches!(n.kind, NodeKind::Dot { .. })
                        && self.egraph.input_of(*id, 0) == Some(ra)
                        && self.egraph.is_hidden(*id, 0)
                });
                if !global_chip {
                    println!("REGION TEST FAIL: the global pie spawned no chip wired to the global source");
                    std::process::exit(1);
                }
                key(raw, egui::Key::Backtick, shift, true);
                next = 19;
            }
            19 => {
                key(raw, egui::Key::Backtick, shift, false);
                next = 20;
            }
            20 => {
                if !self.node_editor.patch_mode {
                    println!("REGION TEST FAIL: shift+backtick did not enter patch mode");
                    std::process::exit(1);
                }
                // The OPS pie: plain ` held, release over "bypass" with
                // Read A selected — the op fires through its own key site.
                let Some(ra) = find(&self.egraph, &|k| {
                    matches!(k, NodeKind::Read { source: 0, .. })
                }) else {
                    return;
                };
                if self.egraph.node(ra).is_some_and(|n| n.disabled) {
                    println!("REGION TEST FAIL: Read A bypassed before the ops pie");
                    std::process::exit(1);
                }
                self.selected = vec![ra];
                key(raw, egui::Key::Backtick, plain, true);
                next = 21;
            }
            21 => {
                if !self.node_editor.test_ops_pie_open() {
                    println!("REGION TEST FAIL: plain backtick did not open the ops pie");
                    std::process::exit(1);
                }
                let Some(p) = self.node_editor.test_ops_slice(keymap::Act::GraphBypass)
                else {
                    return;
                };
                raw.events.push(egui::Event::PointerMoved(p));
                next = 22;
            }
            // Two frames for the spring to land the slice under the pointer.
            22 | 23 => next = stage + 1,
            24 => {
                key(raw, egui::Key::Backtick, plain, false);
                next = 25;
            }
            25 => next = 26,
            26 => {
                let Some(ra) = find(&self.egraph, &|k| {
                    matches!(k, NodeKind::Read { source: 0, .. })
                }) else {
                    return;
                };
                if self.node_editor.test_ops_pie_open() {
                    println!("REGION TEST FAIL: the ops pie stayed open after release");
                    std::process::exit(1);
                }
                if !self.egraph.node(ra).is_some_and(|n| n.disabled) {
                    println!("REGION TEST FAIL: the ops pie's bypass slice did not bypass Read A");
                    std::process::exit(1);
                }
                // F1: the most-used-nodes pie. Fresh prefs = no history,
                // so the roster is the catalog's front rank; releasing over
                // slice 0 must SPAWN that node and score one use.
                self.test.rect =
                    egui::Rect::from_min_max(
                        egui::pos2(self.egraph.nodes.len() as f32, 0.0),
                        egui::Pos2::ZERO,
                    );
                key(raw, egui::Key::F1, plain, true);
                next = 27;
            }
            27 => {
                if !self.node_editor.test_node_pie_open() {
                    println!("REGION TEST FAIL: F1 did not open the node pie");
                    std::process::exit(1);
                }
                let Some((p, _)) = self.node_editor.test_node_slice(0) else { return };
                raw.events.push(egui::Event::PointerMoved(p));
                next = 28;
            }
            28 | 29 => next = stage + 1, // the orbit's spring
            30 => {
                key(raw, egui::Key::F1, plain, false);
                next = 31;
            }
            31 => next = 32,
            32 => {
                if self.node_editor.test_node_pie_open() {
                    println!("REGION TEST FAIL: the node pie stayed open after release");
                    std::process::exit(1);
                }
                let before = self.test.rect.min.x as usize;
                if self.egraph.nodes.len() != before + 1 {
                    println!(
                        "REGION TEST FAIL: the node pie's slice spawned {} nodes, want 1",
                        self.egraph.nodes.len() as i64 - before as i64
                    );
                    std::process::exit(1);
                }
                let total: u32 = self.node_editor.use_counts.values().sum();
                if total != 1 {
                    println!(
                        "REGION TEST FAIL: the spawn scored {total} uses, want exactly 1"
                    );
                    std::process::exit(1);
                }
                println!(
                    "REGION TEST PASS: active region, F3 spawned a hidden-wired chip \
                     from the region's roster, the engine saw an ordinary edge, F2 \
                     registered a GLOBAL source and spawned its chip outside every \
                     region, patch mode toggles, plain ` ops pie bypasses the \
                     selection, F1 spawned the top most-used node and scored it"
                );
                std::process::exit(0);
            }
            _ => {}
        }
        self.test.region = Some((next, frames));
    }
}
