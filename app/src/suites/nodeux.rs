//! `--test-nodeux`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    /// --test-nodeux driver: the node-UX gestures end to end, in one run —
    /// the magnet's one-port-per-output rule on a Blend, S input swap, dots
    /// with `.` held — a tap alone cuts nothing, press+drag on a wire cuts
    /// one and pulls its tension, a click places one at the automatic curve,
    /// a drag on a dot re-pulls it (all PROVEN elided: same topology hash),
    /// Z / X taps pinning the whole selection, a G-grab noodle landed on a
    /// far port, ⌘G backdrop, the A-swipe align, and T tidy. State asserts
    /// only — no pixels — so it runs the same on any machine.
    pub(crate) fn drive_test_nodeux(&mut self, raw: &mut egui::RawInput) {
        let Some((stage, mut frames, bid, aux)) = self.test.nodeux else { return };
        frames += 1;
        // Generous: the late stages wait on eased animations that run on
        // wall time, and a suite's frames are faster than a person's.
        if frames > 3600 {
            println!("NODEUX TEST FAIL: timeout at stage {stage}");
            std::process::exit(1);
        }
        let fail = |msg: String| -> ! {
            println!("NODEUX TEST FAIL: {msg}");
            std::process::exit(1);
        };
        // Default graph: Read A = 0, Read B = 1, Transform = 2, Grade = 3,
        // Blend = 4 (A <- grade chain, B <- Read B).
        let (ra, rb, tf, gr, mg) = (0u64, 1u64, 2u64, 3u64, 4u64);
        let rect_of = |s: &Self, id: u64| -> Option<egui::Rect> {
            s.node_rects.iter().find(|(i, _)| *i == id).map(|(_, r)| *r)
        };
        // A socket's screen spot, from the node's recorded rect: the stalks
        // rest at SOCKET_OFFSET graph units, and the rect is 2·NODE_R tall.
        let sock = |s: &Self, id: u64, out: bool| -> Option<egui::Pos2> {
            let r = rect_of(s, id)?;
            let dy = r.height() * (editor::SOCKET_OFFSET / (2.0 * editor::NODE_R));
            Some(egui::pos2(r.center().x, r.center().y + if out { dy } else { -dy }))
        };
        let mv = |raw: &mut egui::RawInput, p: egui::Pos2| {
            raw.events.push(egui::Event::PointerMoved(p));
        };
        // The dot stages: the transform -> grade wire's midpoint (both
        // ends are stacked, so the wire runs straight through it), the one
        // dot in the graph, and `.` held or let go.
        let wire_mid = |s: &Self| -> Option<egui::Pos2> {
            let (a, b) = (rect_of(s, tf)?, rect_of(s, gr)?);
            Some((a.center() + b.center().to_vec2()) * 0.5)
        };
        let the_dot = |s: &Self| -> Option<u64> {
            s.egraph.nodes.iter().find(|(_, n)| matches!(n.kind, NodeKind::Dot { .. })).map(|(id, _)| *id)
        };
        let hold_dot = |raw: &mut egui::RawInput, down: bool| {
            raw.events.push(egui::Event::Key {
                key: egui::Key::Period,
                physical_key: None,
                pressed: down,
                repeat: false,
                modifiers: Default::default(),
            });
        };
        // What a pull to `p` must set: distance from the bead over the
        // reference handle length its press recorded.
        let pull_want = |s: &Self, dot: u64, p: egui::Pos2, reference: f32| -> f32 {
            let bead = rect_of(s, dot).map(|r| r.center()).unwrap_or(p);
            editor::dot_tension_for(p.distance(bead), reference)
        };
        let spliced = |s: &Self, dot: u64| {
            if s.egraph.input_of(gr, 0) != Some(dot) || s.egraph.input_of(dot, 0) != Some(tf) {
                fail("the dot is not spliced into transform -> grade".into());
            }
        };
        let elided = |s: &Self, dot: u64, hash: u64| {
            let e = s.egraph.to_engine_graph();
            if e.nodes.contains_key(&dot) {
                fail("the dot leaked into the engine graph".into());
            }
            if e.edges.get(&(gr, 0)) != Some(&tf) {
                fail("the engine graph does not resolve through the dot".into());
            }
            if e.topology_hash() != hash {
                fail("a dot (or its tension) changed the topology hash".into());
            }
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
        let key = |raw: &mut egui::RawInput, k: egui::Key, m: egui::Modifiers| {
            // Down and up in one frame: key_pressed and key_released both
            // fire, which is exactly what a tap is.
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
        let mut next = stage;
        match stage {
            // --- settle, then park a fresh Blend's ports beside Read A ---
            0 => {
                if self.layout_settled() && frames >= 6 {
                    let Some(pos) = self.egraph.node(ra).map(|n| n.pos) else { return };
                    // Its + and − straddle Read A's output a few units
                    // below it: both in reach, neither much nearer.
                    let b = self
                        .egraph
                        .add(NodeKind::blend(), pos + egui::vec2(0.0, 77.0));
                    self.test.nodeux = Some((1, frames, b, aux));
                    return;
                }
            }
            // --- hold shift over the graph: the magnet may wire ONE port ---
            1..=8 => {
                if let Some(r) = rect_of(self, ra) {
                    raw.events.insert(0, egui::Event::ModifiersChanged(egui::Modifiers { shift: true, ..Default::default() }));
                    mv(raw, r.center());
                }
                next = stage + 1;
            }
            9 => {
                let (a, b) = (self.egraph.input_of(bid, 0), self.egraph.input_of(bid, 1));
                let hits = [a, b].iter().filter(|s| **s == Some(ra)).count();
                if hits != 1 {
                    fail(format!(
                        "one-port-per-output: blend inputs {a:?}/{b:?}, wanted Read A on exactly one"
                    ));
                }
                self.egraph.remove(bid);
                next = 10;
            }
            // --- S swaps the Blend's + and − ---
            10 => {
                self.selected = vec![mg];
                if let Some(r) = rect_of(self, mg) {
                    mv(raw, r.center());
                }
                key(raw, egui::Key::S, Default::default());
                next = 11;
            }
            11 => {
                if self.egraph.input_of(mg, 0) != Some(rb)
                    || self.egraph.input_of(mg, 1) != Some(gr)
                {
                    fail("S did not swap the blend's inputs".into());
                }
                key(raw, egui::Key::S, Default::default());
                next = 12;
            }
            12 => {
                if self.egraph.input_of(mg, 0) != Some(gr)
                    || self.egraph.input_of(mg, 1) != Some(rb)
                {
                    fail("second S did not swap the blend back".into());
                }
                next = 13;
            }
            // --- `.` is HELD: a tap alone cuts nothing; hold + press ON the
            // transform -> grade wire cuts a dot there and the drag pulls
            // its tension; hold + click cuts one at the automatic curve;
            // hold + drag on a dot re-pulls it. The engine graph must not
            // change AT ALL ---
            13 => {
                let hash = self.egraph.to_engine_graph().topology_hash();
                let Some(m) = wire_mid(self) else { return };
                mv(raw, m);
                key(raw, egui::Key::Period, Default::default());
                self.test.nodeux = Some((60, frames, hash, aux));
                return;
            }
            60 => {
                if the_dot(self).is_some() {
                    fail("tapping `.` alone cut a dot — it is a held key now".into());
                }
                hold_dot(raw, true);
                next = 61;
            }
            61 | 66 => {
                let Some(m) = wire_mid(self) else { return };
                btn(raw, m, true);
                next = stage + 1;
            }
            62 => {
                let Some(dot) = the_dot(self) else {
                    fail("hold `.` + press on the wire cut no dot".into())
                };
                let Some((pulled, r)) = self.node_editor.dot_pull() else {
                    fail("the press did not start a pull".into())
                };
                if pulled != dot {
                    fail(format!("the pull holds {pulled}, not the new dot {dot}"));
                }
                if self.egraph.node(dot).unwrap().tension != 1.0 {
                    fail("a press that has not moved must not pull".into());
                }
                spliced(self, dot);
                self.test.rect = egui::Rect::from_min_max(
                    egui::pos2(r, 0.0),
                    self.egraph.node(dot).unwrap().pos,
                );
                let Some(m) = wire_mid(self) else { return };
                mv(raw, m + egui::vec2(40.0, 0.0));
                next = 63;
            }
            63 | 64 => {
                let (dot, m) = (the_dot(self).unwrap(), wire_mid(self).unwrap());
                let reach = if stage == 63 { 40.0 } else { 90.0 };
                let want = pull_want(self, dot, m + egui::vec2(reach, 0.0), self.test.rect.min.x);
                let got = self.egraph.node(dot).unwrap().tension;
                if (got - want).abs() > 1e-3 {
                    fail(format!(
                        "pulled {reach}px: tension {got:.4}, wanted {want:.4} \
                         (distance over the bead's automatic handle)"
                    ));
                }
                if stage == 63 {
                    mv(raw, m + egui::vec2(90.0, 0.0));
                } else {
                    btn(raw, m + egui::vec2(90.0, 0.0), false);
                }
                next = stage + 1;
            }
            65 => {
                let (dot, m) = (the_dot(self).unwrap(), wire_mid(self).unwrap());
                if self.node_editor.dot_pull().is_some() {
                    fail("the pull outlived the release".into());
                }
                let want = pull_want(self, dot, m + egui::vec2(90.0, 0.0), self.test.rect.min.x);
                let got = self.egraph.node(dot).unwrap().tension;
                if (got - want).abs() > 1e-3 || got == 1.0 {
                    fail(format!("tension after release {got:.4}, wanted {want:.4}"));
                }
                if self.selected != vec![dot] {
                    fail(format!(
                        "the pull's press leaked into a click or marquee: selection {:?}",
                        self.selected
                    ));
                }
                if self.egraph.node(dot).unwrap().pos.distance(self.test.rect.max) > 1e-3 {
                    fail("the pull moved the dot — the press became a node drag".into());
                }
                spliced(self, dot);
                elided(self, dot, bid);
                // Out again (a delete heals the chain) for the click stages.
                self.egraph.remove(dot);
                self.selected.clear();
                if self.egraph.input_of(gr, 0) != Some(tf) {
                    fail("removing the pulled dot did not heal transform -> grade".into());
                }
                next = 66;
            }
            67 => {
                let Some(m) = wire_mid(self) else { return };
                btn(raw, m, false);
                next = 68;
            }
            68 => {
                let Some(dot) = the_dot(self) else {
                    fail("hold `.` + click on the wire cut no dot".into())
                };
                if self.egraph.node(dot).unwrap().tension != 1.0 {
                    fail("a click-placed dot must sit at the automatic curve".into());
                }
                spliced(self, dot);
                elided(self, dot, bid);
                next = 69;
            }
            69 => {
                let dot = the_dot(self).unwrap();
                let Some(bead) = rect_of(self, dot).map(|r| r.center()) else { return };
                btn(raw, bead, true);
                self.test.rect = egui::Rect::from_min_max(bead, bead);
                next = 70;
            }
            70 => {
                let dot = the_dot(self).unwrap();
                let Some((pulled, r)) = self.node_editor.dot_pull() else {
                    fail("hold `.` + press on a dot did not take it".into())
                };
                if pulled != dot {
                    fail(format!("the re-pull holds {pulled}, not the dot {dot}"));
                }
                // Bead in min (from 69), this pull's reference in max.x.
                self.test.rect.max.x = r;
                let n = self.egraph.nodes.values().filter(|n| matches!(n.kind, NodeKind::Dot { .. })).count();
                if n != 1 {
                    fail(format!("pressing a dot with `.` held left {n} dots — it cut another"));
                }
                mv(raw, self.test.rect.min + egui::vec2(70.0, 0.0));
                next = 71;
            }
            71 => {
                let dot = the_dot(self).unwrap();
                let want = pull_want(self, dot, self.test.rect.min + egui::vec2(70.0, 0.0), self.test.rect.max.x);
                let got = self.egraph.node(dot).unwrap().tension;
                if (got - want).abs() > 1e-3 {
                    fail(format!("re-pulled tension {got:.4}, wanted {want:.4}"));
                }
                btn(raw, self.test.rect.min + egui::vec2(70.0, 0.0), false);
                next = 72;
            }
            72 => {
                hold_dot(raw, false);
                let dot = the_dot(self).unwrap();
                if self.node_editor.dot_pull().is_some() {
                    fail("the re-pull outlived the release".into());
                }
                spliced(self, dot);
                // Back to a plain dot for the stages that drag it about.
                self.egraph.node_mut(dot).unwrap().tension = 1.0;
                next = 15;
            }
            // --- Z tap pins the selection's panels; shift+Z replaces ---
            15 => {
                self.selected = vec![tf, gr];
                self.prop_order.clear();
                self.gizmo_pinned.clear();
                key(raw, egui::Key::Z, Default::default());
                next = 16;
            }
            16 => {
                if !(self.prop_order.contains(&tf) && self.prop_order.contains(&gr)) {
                    fail(format!("Z tap pinned {:?}, wanted tf+gr", self.prop_order));
                }
                self.selected = vec![mg];
                key(raw, egui::Key::Z, egui::Modifiers::SHIFT);
                next = 17;
            }
            17 => {
                if self.prop_order != vec![mg] {
                    fail(format!("shift+Z tap left {:?}, wanted [blend]", self.prop_order));
                }
                self.selected = vec![tf, gr];
                key(raw, egui::Key::X, Default::default());
                next = 18;
            }
            18 => {
                if !(self.gizmo_pinned.contains(&tf) && self.gizmo_pinned.contains(&gr)) {
                    fail(format!("X tap pinned {:?}, wanted tf+gr", self.gizmo_pinned));
                }
                self.selected.clear();
                key(raw, egui::Key::X, egui::Modifiers::SHIFT);
                next = 19;
            }
            19 => {
                if !self.gizmo_pinned.is_empty() {
                    fail("shift+X tap with nothing selected must unpin all".into());
                }
                // A Grade for the noodle to land on — parked below Read B,
                // inside the stretch of canvas the default graph proves is
                // on screen (the suite clicks rb and mg rects directly).
                let g2 = self
                    .egraph
                    .add(NodeKind::Grade(Default::default()), egui::pos2(140.0, 60.0));
                self.test.nodeux = Some((20, frames, bid, g2));
                return;
            }
            // --- G-grab: select Read B's output, grab, land on the Grade ---
            20 => {
                if let Some(p) = sock(self, rb, true) {
                    btn(raw, p, true);
                }
                next = 21;
            }
            21 => {
                if let Some(p) = sock(self, rb, true) {
                    btn(raw, p, false);
                }
                next = 22;
            }
            22 => {
                key(raw, egui::Key::G, Default::default());
                next = 23;
            }
            23 => {
                if let Some(p) = sock(self, aux, false) {
                    mv(raw, p);
                }
                next = 24;
            }
            24 => {
                if let Some(p) = sock(self, aux, false) {
                    btn(raw, p, true);
                }
                next = 25;
            }
            25 => {
                if let Some(p) = sock(self, aux, false) {
                    btn(raw, p, false);
                }
                next = 26;
            }
            26 => {
                if self.egraph.input_of(aux, 0) != Some(rb) {
                    fail(format!(
                        "G-grab noodle did not land: grade input = {:?}",
                        self.egraph.input_of(aux, 0)
                    ));
                }
                next = 27;
            }
            // --- ⌘G wraps the selection in a backdrop ---
            27 => {
                self.selected = vec![tf, gr];
                key(raw, egui::Key::G, egui::Modifiers::COMMAND);
                next = 28;
            }
            28 => {
                let Some(g) = self.egraph.groups.first() else {
                    fail("⌘G made no backdrop".into())
                };
                if !(g.members.contains(&tf) && g.members.contains(&gr)) {
                    fail(format!("backdrop holds {:?}, wanted tf+gr", g.members));
                }
                // Membership must be exclusive: nothing else joined.
                if g.members.len() != 2 {
                    fail(format!("backdrop holds {:?}, wanted exactly tf+gr", g.members));
                }
                next = 40;
            }
            // --- the backdrop's grip scales its MEMBERS (positions and
            // node sizes), not just a rectangle ---
            40 => {
                let Some(g) = self.egraph.groups.first() else {
                    fail("backdrop vanished before the scale test".into())
                };
                let Some(gp) = self.node_editor.test_group_grip(g.id) else {
                    return; // blob not drawn yet — come back next frame
                };
                btn(raw, gp, true);
                let packed = ((gp.x.to_bits() as u64) << 32) | gp.y.to_bits() as u64;
                self.test.nodeux = Some((41, frames, packed, aux));
                return;
            }
            41 => {
                let gp = egui::pos2(
                    f32::from_bits((bid >> 32) as u32),
                    f32::from_bits(bid as u32),
                );
                mv(raw, gp + egui::vec2(120.0, 90.0));
                next = 42;
            }
            42 => {
                let gp = egui::pos2(
                    f32::from_bits((bid >> 32) as u32),
                    f32::from_bits(bid as u32),
                );
                btn(raw, gp + egui::vec2(120.0, 90.0), false);
                next = 43;
            }
            43 => {
                // tf and gr started 120 units apart at size 1; the grip
                // drag must have spread them AND grown them.
                let (a, b) = (self.egraph.node(tf).unwrap(), self.egraph.node(gr).unwrap());
                let d = a.pos.distance(b.pos);
                if d < 138.0 {
                    fail(format!("grip scale left members {d:.0} apart, wanted > 138"));
                }
                if a.size < 1.1 || b.size < 1.1 {
                    fail(format!(
                        "grip scale left sizes {:.2}/{:.2}, wanted > 1.1 — the nodes \
                         must resize WITH the group",
                        a.size, b.size
                    ));
                }
                next = 44;
            }
            // --- selecting a backdrop selects its members for S + drag:
            // held S, a drag away from the centre spreads them (positions
            // only — sizes hold), and S's release swaps nothing ---
            44 => {
                let Some(g) = self.egraph.groups.first() else {
                    fail("backdrop vanished before the S-scale test".into())
                };
                self.selected.clear();
                self.node_editor.sel_groups = vec![g.id];
                next = 45;
            }
            45 => {
                let Some(g) = self.egraph.groups.first() else { return };
                let cs: Vec<egui::Pos2> =
                    g.members.iter().filter_map(|m| rect_of(self, *m).map(|r| r.center())).collect();
                if cs.len() < 2 {
                    return;
                }
                if !self.group_panel_shown {
                    fail(
                        "selecting a backdrop must put its panel in the \
                         properties rack (the empty rack used to early-out \
                         past it)"
                            .into(),
                    );
                }
                let c = cs.iter().fold(egui::Rect::NOTHING, |r, p| r.union(egui::Rect::from_min_max(*p, *p))).center();
                // Baselines ride the shared test_rect scratch: member
                // distance, tf's size, and grade's input wiring.
                let (a, b) = (self.egraph.node(tf).unwrap(), self.egraph.node(gr).unwrap());
                self.test.rect = egui::Rect::from_min_max(
                    egui::pos2(a.pos.distance(b.pos), a.size),
                    egui::pos2(0.0, 0.0),
                );
                raw.events.insert(0, egui::Event::ModifiersChanged(Default::default()));
                raw.events.push(egui::Event::Key {
                    key: egui::Key::S,
                    physical_key: None,
                    pressed: true,
                    repeat: false,
                    modifiers: Default::default(),
                });
                let h = c + egui::vec2(40.0, 30.0);
                btn(raw, h, true);
                self.test.nodeux = Some((
                    46,
                    frames,
                    ((c.x.to_bits() as u64) << 32) | c.y.to_bits() as u64,
                    aux,
                ));
                return;
            }
            46 => {
                let c = egui::pos2(
                    f32::from_bits((bid >> 32) as u32),
                    f32::from_bits(bid as u32),
                );
                mv(raw, c + egui::vec2(100.0, 75.0));
                next = 47;
            }
            47 => {
                let c = egui::pos2(
                    f32::from_bits((bid >> 32) as u32),
                    f32::from_bits(bid as u32),
                );
                btn(raw, c + egui::vec2(100.0, 75.0), false);
                raw.events.push(egui::Event::Key {
                    key: egui::Key::S,
                    physical_key: None,
                    pressed: false,
                    repeat: false,
                    modifiers: Default::default(),
                });
                next = 48;
            }
            48 => {
                let (before, size0) = (self.test.rect.min.x, self.test.rect.min.y);
                let (a, b) = (self.egraph.node(tf).unwrap(), self.egraph.node(gr).unwrap());
                let after = a.pos.distance(b.pos);
                if after < before * 1.5 {
                    fail(format!(
                        "S + drag left members {after:.0} apart (was {before:.0}) — \
                         a held S drag must scale the selection's spacing"
                    ));
                }
                if (a.size - size0).abs() > 1e-4 {
                    fail(format!(
                        "S + drag changed a node's size ({size0:.2} -> {:.2}) — it \
                         scales positions only",
                        a.size
                    ));
                }
                self.node_editor.sel_groups.clear();
                next = 49;
            }
            // --- a dot can be MOVED by its body (its sockets sit on the
            // bead and used to eat the press), and moving it rewires
            // nothing ---
            49 => {
                let Some(dot) = self
                    .egraph
                    .nodes
                    .iter()
                    .find(|(_, n)| matches!(n.kind, NodeKind::Dot { .. }))
                    .map(|(id, _)| *id)
                else {
                    fail("the dot from stage 13 is gone".into())
                };
                let Some((_, r)) = self.node_rects.iter().find(|(i, _)| *i == dot) else {
                    return;
                };
                let p = self.egraph.node(dot).unwrap().pos;
                self.test.rect = egui::Rect::from_min_max(p, p);
                btn(raw, r.center(), true);
                self.test.nodeux = Some((50, frames, dot, aux));
                return;
            }
            50 => {
                if let Some((_, r)) = self.node_rects.iter().find(|(i, _)| *i == bid) {
                    mv(raw, r.center() + egui::vec2(40.0, 25.0));
                }
                next = 51;
            }
            51 => {
                if let Some((_, r)) = self.node_rects.iter().find(|(i, _)| *i == bid) {
                    btn(raw, r.center(), false);
                }
                next = 52;
            }
            52 => {
                let dot = bid;
                let moved = self.egraph.node(dot).unwrap().pos.distance(self.test.rect.min);
                if moved < 20.0 {
                    fail(format!(
                        "dragging a dot's body moved it {moved:.1} units — the bead \
                         must be movable, not eaten by its own sockets"
                    ));
                }
                if self.egraph.input_of(gr, 0) != Some(dot)
                    || self.egraph.input_of(dot, 0) != Some(tf)
                {
                    fail("moving a dot must not rewire it".into());
                }
                // Let the stalk springs settle before aiming at a wire: the
                // scale stages just yanked their nodes, and the click's
                // target is computed from REST positions.
                self.test.rect.max.x = 0.0;
                next = 56;
            }
            56 => {
                self.test.rect.max.x += 1.0;
                if self.test.rect.max.x > 20.0 {
                    next = 53;
                }
            }
            // --- shift-click ON a wire cuts a dot into it. The target is
            // the Read B -> grab-test Grade wire: both ends are unscaled,
            // so the midpoint sits well clear of any node's hit circle (a
            // node under the pointer wins the click, by design) ---
            53 => {
                // Screen positions derived from Read A's rect (known graph
                // pos): the wire's vertical-tangent bezier passes exactly
                // through the midpoint of its endpoints.
                let Some((_, ra_r)) = self.node_rects.iter().find(|(i, _)| *i == ra) else {
                    return;
                };
                let z = ra_r.height() / (2.0 * editor::NODE_R);
                let anchor = ra_r.center()
                    - egui::vec2(-140.0, -240.0) * z; // Read A never moves
                let to_screen = |g: egui::Pos2| anchor + g.to_vec2() * z;
                let (rn, g2n) =
                    (self.egraph.node(rb).unwrap(), self.egraph.node(aux).unwrap());
                let from = to_screen(rn.pos + egui::vec2(0.0, 42.0 * rn.size));
                let to = to_screen(g2n.pos - egui::vec2(0.0, 42.0 * g2n.size));
                let mid = ((from.to_vec2() + to.to_vec2()) * 0.5).to_pos2();
                raw.events.insert(0, egui::Event::ModifiersChanged(egui::Modifiers { shift: true, ..Default::default() }));
                raw.events.push(egui::Event::PointerMoved(mid));
                raw.events.push(egui::Event::PointerButton {
                    pos: mid,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::SHIFT,
                });
                self.test.nodeux = Some((
                    54,
                    frames,
                    ((mid.x.to_bits() as u64) << 32) | mid.y.to_bits() as u64,
                    aux,
                ));
                return;
            }
            54 => {
                let mid = egui::pos2(
                    f32::from_bits((bid >> 32) as u32),
                    f32::from_bits(bid as u32),
                );
                raw.events.insert(0, egui::Event::ModifiersChanged(egui::Modifiers { shift: true, ..Default::default() }));
                raw.events.push(egui::Event::PointerButton {
                    pos: mid,
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers: egui::Modifiers::SHIFT,
                });
                next = 55;
            }
            55 => {
                let d2 = self.egraph.input_of(aux, 0);
                let ok = d2.is_some_and(|d| {
                    self.egraph.is_dot(d) && self.egraph.input_of(d, 0) == Some(rb)
                });
                if !ok {
                    let mid = egui::pos2(
                        f32::from_bits((bid >> 32) as u32),
                        f32::from_bits(bid as u32),
                    );
                    let dots = self
                        .egraph
                        .nodes
                        .values()
                        .filter(|n| matches!(n.kind, NodeKind::Dot { .. }))
                        .count();
                    fail(format!(
                        "shift-click on the ReadB->grade wire should cut a dot in \
                         (grade input 0 = {d2:?}; clicked {mid:?}, panel {:?}, \
                         dots in graph = {dots})",
                        self.graph_panel_rect,
                    ));
                }
                next = 29;
            }
            // --- A + swipe: the selection lands on the line, in a row ---
            29 => {
                self.selected = vec![tf, gr, aux];
                raw.events.push(egui::Event::Key {
                    key: egui::Key::A,
                    physical_key: None,
                    pressed: true,
                    repeat: false,
                    modifiers: Default::default(),
                });
                next = 30;
            }
            30 | 31 | 32 => {
                // Swipe along a clear line low in the panel, left to right.
                let pr = self.graph_panel_rect;
                let y = pr.max.y - 40.0;
                let (x0, x1) = (pr.min.x + 60.0, pr.max.x - 60.0);
                match stage {
                    30 => btn(raw, egui::pos2(x0, y), true),
                    31 => mv(raw, egui::pos2((x0 + x1) * 0.5, y + 2.0)),
                    _ => {
                        btn(raw, egui::pos2(x1, y + 4.0), false);
                        raw.events.push(egui::Event::Key {
                            key: egui::Key::A,
                            physical_key: None,
                            pressed: false,
                            repeat: false,
                            modifiers: Default::default(),
                        });
                    }
                }
                next = stage + 1;
            }
            33 => {
                let p = |id: u64| self.egraph.node(id).unwrap().pos;
                let (a, b, c) = (p(tf), p(gr), p(aux));
                if (a.y - b.y).abs() > 0.5 || (b.y - c.y).abs() > 0.5 {
                    fail(format!("swipe: y not aligned ({}, {}, {})", a.y, b.y, c.y));
                }
                let step1 = b.x - a.x;
                let step2 = c.x - b.x;
                if step1 <= 0.0 || step2 <= 0.0 || (step1 - step2).abs() > 1.0 {
                    fail(format!("swipe: spacing uneven ({step1} vs {step2})"));
                }
                next = 34;
            }
            // --- T: tidy the whole comp back into a layered flow ---
            34 => {
                self.selected.clear();
                if let Some(r) = rect_of(self, mg) {
                    mv(raw, r.center());
                }
                key(raw, egui::Key::T, Default::default());
                next = 35;
            }
            35 => {
                // Every wire must run downhill after the layout.
                for (d, _, s) in self.egraph.edges.clone() {
                    let (Some(dn), Some(sn)) = (self.egraph.node(d), self.egraph.node(s))
                    else {
                        continue;
                    };
                    if dn.pos.y <= sn.pos.y {
                        fail(format!("tidy: edge {s}->{d} runs uphill"));
                    }
                }
                next = 200;
            }
            // --- renaming. Let the tidy cascade land first: the name's
            // hit quad is last frame's drawing, and it must be at rest ---
            200..=229 => next = stage + 1,
            230 => {
                // Home the view on the Grade so its name is drawn on screen
                // (names only draw above zoom·size 0.4).
                let Some(gp) = self.egraph.node(gr).map(|n| n.pos) else { return };
                self.node_editor.zoom = 1.0;
                self.node_editor.pan = -gp.to_vec2();
                self.selected = vec![tf];
                mv(raw, self.graph_panel_rect.center());
                next = 231;
            }
            231 => {
                // N over the graph: rename the one selected node.
                mv(raw, self.graph_panel_rect.center());
                key(raw, egui::Key::N, Default::default());
                next = 232;
            }
            232 => {
                if self.node_editor.test_renaming() != Some(tf) {
                    fail(format!(
                        "N did not open a rename on the selected node (renaming {:?})",
                        self.node_editor.test_renaming()
                    ));
                }
                next = 233; // the field takes focus on its first frame
            }
            233 => {
                raw.events.push(egui::Event::Text("hero".into()));
                next = 234;
            }
            234 => {
                key(raw, egui::Key::Enter, Default::default());
                next = 235;
            }
            235 => {
                let got = self.egraph.node(tf).map(|n| n.ui_label.clone());
                if got.as_deref() != Some("hero") || self.node_editor.test_renaming().is_some() {
                    fail(format!(
                        "Enter must commit the rename: label {got:?}, still renaming {:?}",
                        self.node_editor.test_renaming()
                    ));
                }
                next = 236;
            }
            // Double-click the Grade's NAME: a rename opens on it. Four
            // frames: press, release, press, release.
            236..=239 => {
                let Some(c) = self.node_editor.test_label_centre(gr) else {
                    fail("the Grade's name was not drawn — nothing to double-click".into())
                };
                let down = stage == 236 || stage == 238;
                raw.events.push(egui::Event::PointerMoved(c));
                raw.events.push(egui::Event::PointerButton {
                    pos: c,
                    button: egui::PointerButton::Primary,
                    pressed: down,
                    modifiers: Default::default(),
                });
                next = stage + 1;
            }
            240 => {
                if self.node_editor.test_renaming() != Some(gr) {
                    let c = self.node_editor.test_label_centre(gr);
                    let under: Vec<u64> = c
                        .map(|c| {
                            self.node_rects
                                .iter()
                                .filter(|(_, r)| r.expand(4.0).contains(c))
                                .map(|(i, _)| *i)
                                .collect()
                        })
                        .unwrap_or_default();
                    fail(format!(
                        "double-clicking the Grade's name did not open its rename \
                         (renaming {:?}; name centre {c:?}, panel {:?}, node rects under it \
                         {under:?}, gr rect {:?}, groups {})",
                        self.node_editor.test_renaming(),
                        self.graph_panel_rect,
                        self.node_rects.iter().find(|(i, _)| *i == gr).map(|(_, r)| *r),
                        self.egraph.groups.len(),
                    ));
                }
                // Esc CANCELS: nothing typed lands.
                raw.events.push(egui::Event::Text("junk".into()));
                next = 241;
            }
            241 => {
                key(raw, egui::Key::Escape, Default::default());
                next = 242;
            }
            242 => {
                let got = self.egraph.node(gr).map(|n| n.ui_label.clone());
                if self.node_editor.test_renaming().is_some()
                    || got.as_deref().is_some_and(|l| l.contains("junk"))
                {
                    fail(format!(
                        "Esc must cancel a rename: label {got:?}, still renaming {:?}",
                        self.node_editor.test_renaming()
                    ));
                }
                next = 243;
            }
            // --- the noodle legend: hovering its dots brings the words out,
            // leaving puts them away (eased on wall time: see 951) ---
            243 => {
                let Some(row) = self.node_editor.test_legend_row() else {
                    fail("the noodle legend's dots were not drawn".into())
                };
                if self.node_editor.test_legend_reveal() > 0.01 {
                    fail("the legend's words should be away until hovered".into());
                }
                mv(raw, row.center());
                self.test.rect.max.x = 0.0;
                next = 244;
            }
            244 => {
                let Some(row) = self.node_editor.test_legend_row() else { return };
                mv(raw, row.center());
                self.test.rect.max.x += 1.0;
                if self.node_editor.test_legend_reveal() > 0.97 {
                    next = 245;
                } else if self.test.rect.max.x > 900.0 {
                    fail(format!(
                        "hovering the dots should bring the words out (reveal {:.2})",
                        self.node_editor.test_legend_reveal()
                    ));
                }
            }
            245 => {
                mv(raw, self.graph_panel_rect.center());
                self.test.rect.max.x = 0.0;
                next = 246;
            }
            246 => {
                mv(raw, self.graph_panel_rect.center());
                self.test.rect.max.x += 1.0;
                if self.node_editor.test_legend_reveal() < 0.01
                    && self.node_editor.test_keys_reveal() < 0.01
                {
                    next = 248;
                } else if self.test.rect.max.x > 900.0 {
                    fail(format!(
                        "leaving the dots should put the words away (reveal {:.2})",
                        self.node_editor.test_legend_reveal()
                    ));
                }
            }
            248 => {
                // --- shift+Z+X-click: ONE click solos a node's panel AND its
                // manipulators (both keys held is both gestures; it used to
                // be else-if, and the hand had to do Z then X). Something is
                // pinned in both lists first, so "solo" has something to
                // clear. The node is whichever the graph has on screen.
                self.prop_order = vec![mg];
                self.gizmo_pinned = vec![gr];
                let Some((id, r)) = self
                    .node_rects
                    .iter()
                    .find(|(i, _)| *i == tf)
                    .or_else(|| self.node_rects.first())
                    .copied()
                else {
                    fail("no node on screen to Z+X-click".into())
                };
                raw.events.insert(0, egui::Event::ModifiersChanged(egui::Modifiers::SHIFT));
                for k in [egui::Key::Z, egui::Key::X] {
                    raw.events.push(egui::Event::Key {
                        key: k,
                        physical_key: None,
                        pressed: true,
                        repeat: false,
                        modifiers: egui::Modifiers::SHIFT,
                    });
                }
                btn(raw, r.center(), true);
                self.test.nodeux = Some((900, frames, bid, id));
                return;
            }
            900 => {
                let Some(r) = rect_of(self, aux) else { fail("the Z+X-clicked node vanished".into()) };
                // The click resolves on the RELEASE, and raw input is
                // per frame: shift has to be held here too.
                raw.events.insert(0, egui::Event::ModifiersChanged(egui::Modifiers::SHIFT));
                btn(raw, r.center(), false);
                next = 901;
            }
            901 => {
                for k in [egui::Key::Z, egui::Key::X] {
                    raw.events.push(egui::Event::Key {
                        key: k,
                        physical_key: None,
                        pressed: false,
                        repeat: false,
                        modifiers: egui::Modifiers::SHIFT,
                    });
                }
                raw.events.insert(0, egui::Event::ModifiersChanged(Default::default()));
                next = 902;
            }
            902 => {
                if self.prop_order != vec![aux] || self.gizmo_pinned != vec![aux] {
                    fail(format!(
                        "shift+Z+X-click on {aux} left panels {:?} and manipulators {:?}, wanted [{aux}] in both",
                        self.prop_order, self.gizmo_pinned
                    ));
                }
                next = 950;
            }
            // --- the key strip: hovering its caps brings the words out,
            // leaving puts them away ---
            950 => {
                let Some(row) = self.node_editor.test_keys_row() else {
                    fail("the key strip was not drawn".into())
                };
                if self.node_editor.test_keys_reveal() > 0.05 {
                    fail(format!(
                        "the key strip's words should be away until hovered (reveal {:.2})",
                        self.node_editor.test_keys_reveal()
                    ));
                }
                mv(raw, row.left_center() + egui::vec2(8.0, 0.0));
                self.test.rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::Vec2::ZERO);
                next = 951;
            }
            951 => {
                let Some(row) = self.node_editor.test_keys_row() else { return };
                mv(raw, row.left_center() + egui::vec2(8.0, 0.0));
                self.test.rect.max.x += 1.0;
                // The reveal eases on WALL time (1 − e^(−12·dt) a frame), and
                // a suite's frames come fast enough to sit on dt's 1 ms
                // floor — so the budget is in frames enough for ~0.6 s.
                if self.node_editor.test_keys_reveal() > 0.97 {
                    next = 952;
                } else if self.test.rect.max.x > 900.0 {
                    fail(format!(
                        "hovering the caps should bring the words out (reveal {:.2})",
                        self.node_editor.test_keys_reveal()
                    ));
                }
            }
            952 => {
                mv(raw, self.graph_panel_rect.center());
                self.test.rect.max.x = 0.0;
                next = 953;
            }
            953 => {
                mv(raw, self.graph_panel_rect.center());
                self.test.rect.max.x += 1.0;
                if self.node_editor.test_keys_reveal() < 0.03 {
                    next = 954;
                } else if self.test.rect.max.x > 900.0 {
                    fail(format!(
                        "leaving the caps should put the words away (reveal {:.2})",
                        self.node_editor.test_keys_reveal()
                    ));
                }
            }
            // --- the graph's strips answer to the GRAPH bar's toggles, not
            // the viewer's "hud keys" ---
            954 => {
                self.prefs.hud_keys = Some(false);
                next = 955;
            }
            955 | 956 => next = stage + 1,
            957 => {
                if self.node_editor.test_keys_row().is_none() {
                    fail("turning the VIEWER's hud keys off hid the graph's key strip".into());
                }
                self.prefs.hud_keys = None;
                self.prefs.graph_keys = Some(false);
                next = 958;
            }
            958 | 959 => next = stage + 1,
            960 => {
                if self.node_editor.test_keys_row().is_some() {
                    fail("the graph bar's keys toggle should hide the key strip".into());
                }
                if self.node_editor.test_legend_row().is_none() {
                    fail("hiding the keys must leave the noodle legend up".into());
                }
                self.prefs.graph_keys = None;
                next = 961;
            }
            961 => {
                println!(
                    "NODEUX TEST PASS: magnet one-port rule, S swap, hold-. dots (tap cuts nothing, \
                     press+drag pulls tension, click places, drag re-pulls) + elide, \
                     Z/X taps, G-grab landing, ⌘G shaped backdrop (grip scales members, held S + drag spreads them), dot body-drag + shift-click cut, A-swipe align, T tidy, \
                     N + double-click-the-name rename (Enter commits, Esc cancels), noodle legend words on hover, \
                     shift+Z+X-click solos a node's panel and manipulators in one click, \
                     key strip words come out on hover and go away after, \
                     the graph's strips answer to its own bar, not the viewer's hud keys"
                );
                std::process::exit(0);
            }
            _ => {}
        }
        self.test.nodeux = Some((next, frames, bid, aux));
    }
}
