//! `--test-geotools`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    /// --test-geo driver: the whole 2dini authoring loop — dive into a
    /// 2dini node wired to a Noise, build a noise-driven scatter + instance
    /// drawing with an exposed Slider and a viewer-drawn Spline inside it,
    /// step back out, render it, and drive the Slider through the channel
    /// the node exposes. The conversions and the evaluator have their own
    /// unit tests; this is the machinery around them.
    /// --test-geotools: the 2dini tool bar, handles, click-through and the
    /// big dot, every step through real pointer events. Waits are on the
    /// wall clock.
    pub(crate) fn drive_test_geotools(&mut self, raw: &mut egui::RawInput) {
        use engine::geo::GeoNode as G;
        let Some((stage, mut frames, t0)) = self.test.geotools else { return };
        frames += 1;
        let fail = |msg: String| -> ! {
            println!("GEOTOOLS TEST FAIL: {msg}");
            std::process::exit(1);
        };
        if t0.elapsed().as_secs_f32() > 60.0 {
            fail(format!("timeout at stage {stage}"));
        }
        let press = |raw: &mut egui::RawInput, p: egui::Pos2, down: bool| {
            raw.events.push(egui::Event::PointerMoved(p));
            raw.events.push(egui::Event::PointerButton {
                pos: p,
                button: egui::PointerButton::Primary,
                pressed: down,
                modifiers: Default::default(),
            });
        };
        let mv = |raw: &mut egui::RawInput, p: egui::Pos2| raw.events.push(egui::Event::PointerMoved(p));
        // Geo space (y in width units) to the screen, through the image.
        let g2s = |s: &Self, p: [f32; 2]| {
            let r = s.image_rect;
            r.min + egui::vec2(p[0], p[1]) * r.width()
        };
        let tool = |s: &Self, t: GeoTool| s.geo.tool_spots.iter().find(|(k, _)| *k == t).map(|(_, r)| r.center());
        let kind = |s: &Self, id: u64| match s.egraph.node(id).map(|n| &n.kind) {
            Some(NodeKind::GeoNode(k)) => Some(k.clone()),
            _ => None,
        };
        let find = |s: &Self, f: &dyn Fn(&G) -> bool| {
            s.egraph.nodes.iter().find_map(|(id, n)| match &n.kind {
                NodeKind::GeoNode(k) if f(k) => Some(*id),
                _ => None,
            })
        };
        let out_sock = |s: &Self, id: u64| -> Option<egui::Pos2> {
            let r = s.node_rects.iter().find(|(i, _)| *i == id).map(|(_, r)| *r)?;
            let dy = r.height() * (editor::SOCKET_OFFSET / (2.0 * editor::NODE_R));
            Some(egui::pos2(r.center().x, r.center().y + dy))
        };
        let id = |s: &Self, k: &str| s.test.geotools_ids.get(k).copied().unwrap_or(0);
        let mut next = stage;
        // Every gesture is press / moves / release on separate passes, a
        // few passes apart, so egui never folds two into one.
        let step = frames > 4;
        match stage {
            0 if self.playback.is_some() && self.displayed_total >= 3 => {
                if !self.layout_settled() {
                    return;
                }
                let g = self.egraph.add(NodeKind::Geo(Default::default()), egui::Pos2::new(320.0, 40.0));
                self.selected = vec![g];
                self.playing = false;
                if let Err(e) = self.dive_into(g) {
                    fail(format!("could not open the 2dini: {e}"));
                }
                next = 1;
            }
            1 if frames > 10 => {
                if self.dive.last().map(|d| d.kind) != Some(DiveKind::Geo) {
                    fail("the dive did not open a geometry graph".into());
                }
                // Start from an empty drawing: just the Output.
                let keep: Vec<u64> = self
                    .egraph
                    .nodes
                    .iter()
                    .filter(|(_, n)| matches!(n.kind, NodeKind::GeoNode(G::Output)))
                    .map(|(i, _)| *i)
                    .collect();
                self.egraph.nodes.retain(|i, _| keep.contains(i));
                self.egraph.edges.clear();
                self.selected.clear();
                // In PIN mode, as it is usually run: a viewer pick must still bring
                // its manipulators up (WOBBLE_GEOTOOLS_FOLLOW=1: follow mode).
                self.prefs.hud_follow = Some(std::env::var_os("WOBBLE_GEOTOOLS_FOLLOW").is_some());
                next = 2;
            }
            // Arm the circle tool from the bar.
            2 if step => match tool(self, GeoTool::Circle) {
                Some(p) => {
                    press(raw, p, true);
                    next = 3;
                }
                None => fail("no tool bar in the 2dini".into()),
            },
            3 => {
                press(raw, tool(self, GeoTool::Circle).unwrap_or_default(), false);
                next = 4;
            }
            4 if step => {
                if self.geo.tool != Some(GeoTool::Circle) {
                    fail(format!("the circle button did not arm the tool ({:?})", self.geo.tool));
                }
                press(raw, g2s(self, [0.4, 0.25]), true);
                next = 5;
            }
            5 if frames > 2 => {
                // Drag out from the centre, in a few moves.
                let k = (frames - 2).min(4) as f32 / 4.0;
                mv(raw, g2s(self, [0.4 + 0.1 * k, 0.25]));
                if frames > 7 {
                    next = 6;
                }
            }
            6 => {
                press(raw, g2s(self, [0.5, 0.25]), false);
                next = 7;
            }
            7 if step => {
                let Some(c) = find(self, &|k| matches!(k, G::Circle { .. })) else {
                    fail("drawing with the circle tool made no Circle".into())
                };
                let Some(G::Circle { center, radius, .. }) = kind(self, c) else { unreachable!() };
                if (center[0] - 0.4).abs() > 0.01 || (center[1] - 0.25).abs() > 0.01 || (radius - 0.1).abs() > 0.01 {
                    fail(format!("circle at {center:?} r {radius}: wanted (0.4, 0.25) r 0.1"));
                }
                let g = self.live_geo();
                if g.output().is_none() || engine::geo::eval_node(&g, g.output().unwrap(), &engine::geo::EvalCx {
                    tick: 0.0, field: None, seed: 0.0, aspect: 0.5, px_per_unit: 1000.0,
                }).prims.is_empty() {
                    fail("the circle is not wired into the drawing".into());
                }
                if self.selected != [c] || self.geo.tool.is_some() {
                    fail(format!("after drawing: selected {:?}, tool {:?}", self.selected, self.geo.tool));
                }
                self.test.geotools_ids.insert("circle", c);
                // Its centre handle: drag it right by 0.1.
                press(raw, g2s(self, center), true);
                next = 8;
            }
            8 if frames > 2 => {
                let k = (frames - 2).min(4) as f32 / 4.0;
                mv(raw, g2s(self, [0.4 + 0.1 * k, 0.25]));
                if frames > 7 {
                    next = 9;
                }
            }
            9 => {
                press(raw, g2s(self, [0.5, 0.25]), false);
                next = 10;
            }
            10 if step => {
                let Some(G::Circle { center, .. }) = kind(self, id(self, "circle")) else { fail("circle gone".into()) };
                if (center[0] - 0.5).abs() > 0.01 {
                    fail(format!("the centre handle did not move the circle: centre {center:?}"));
                }
                // The star tool, and a plain click: the default size there.
                match tool(self, GeoTool::Star) {
                    Some(p) => press(raw, p, true),
                    None => fail("no star tool".into()),
                }
                next = 11;
            }
            11 => {
                press(raw, tool(self, GeoTool::Star).unwrap_or_default(), false);
                next = 12;
            }
            12 if step => {
                press(raw, g2s(self, [0.75, 0.3]), true);
                next = 13;
            }
            13 if frames > 2 => {
                press(raw, g2s(self, [0.75, 0.3]), false);
                next = 14;
            }
            14 if step => {
                let Some(st) = find(self, &|k| matches!(k, G::Star { .. })) else { fail("the star tool made no Star".into()) };
                let Some(G::Star { center, outer, .. }) = kind(self, st) else { unreachable!() };
                if (center[0] - 0.75).abs() > 0.01 || outer < 0.05 {
                    fail(format!("star at {center:?} outer {outer}"));
                }
                // Two shapes: the Output is fed a Merge now.
                let out = find(self, &|k| matches!(k, G::Output)).unwrap_or(0);
                let fed = self.egraph.input_of(out, 0).and_then(|f| kind(self, f));
                if !matches!(fed, Some(G::Merge { .. })) {
                    fail(format!("two shapes but the Output is fed {:?}", fed.map(|k| k.name())));
                }
                self.test.geotools_ids.insert("star", st);
                if self.selected != [st] {
                    fail(format!("the new star is not selected: {:?}", self.selected));
                }
                match tool(self, GeoTool::Transform) {
                    Some(p) => press(raw, p, true),
                    None => fail("no transform tool".into()),
                }
                next = 15;
            }
            15 => {
                press(raw, tool(self, GeoTool::Transform).unwrap_or_default(), false);
                next = 16;
            }
            16 if step => {
                let st = id(self, "star");
                let Some(xf) = find(self, &|k| matches!(k, G::Transform { .. })) else {
                    fail("the transform tool made no Transform".into())
                };
                if self.egraph.input_of(xf, 0) != Some(st) || self.selected != [xf] {
                    fail("the Transform is not after the star, selected".into());
                }
                if self.egraph.edges.iter().any(|(d, _, s)| *s == st && *d != xf) {
                    fail("the star still feeds something past the Transform".into());
                }
                if !matches!(kind(self, xf), Some(G::Transform { inherit: true, .. })) {
                    fail("a tool Transform should inherit the pivot".into());
                }
                self.test.geotools_ids.insert("xf", xf);
                // Its handles sit on the star's centre: drag that down.
                press(raw, g2s(self, [0.75, 0.3]), true);
                next = 17;
            }
            17 if frames > 2 => {
                let k = (frames - 2).min(4) as f32 / 4.0;
                mv(raw, g2s(self, [0.75, 0.3 + 0.05 * k]));
                if frames > 7 {
                    next = 18;
                }
            }
            18 => {
                press(raw, g2s(self, [0.75, 0.35]), false);
                next = 19;
            }
            19 if step => {
                let Some(G::Transform { translate, .. }) = kind(self, id(self, "xf")) else { fail("transform gone".into()) };
                if (translate[1] - 0.05).abs() > 0.008 || translate[0].abs() > 0.008 {
                    fail(format!("the Transform's centre handle (on the star) did not move it: {translate:?}"));
                }
                // Select the circle, then click the moved star: merged, so
                // the click lands on its branch's Transform.
                self.geo_select(id(self, "circle"));
                press(raw, g2s(self, [0.75, 0.35]), true);
                next = 20;
            }
            20 if frames > 2 => {
                press(raw, g2s(self, [0.75, 0.35]), false);
                next = 21;
            }
            21 if step => {
                if self.selected != [id(self, "xf")] {
                    fail(format!("a click on the merged star selected {:?}, not its Transform", self.selected));
                }
                // The big dot: a Slider beside the Transform, dragged in.
                let at = self.egraph.node(id(self, "xf")).map(|n| n.pos).unwrap_or_default();
                let sl = self.egraph.add(NodeKind::GeoNode(G::Slider { value: 0.5, name: "turn".into() }), at + egui::vec2(130.0, -90.0));
                self.test.geotools_ids.insert("slider", sl);
                next = 22;
            }
            22 if frames > 8 => {
                let (Some(a), Some(_)) = (out_sock(self, id(self, "slider")), self.node_editor.test_big_dot(id(self, "xf"))) else {
                    fail("the slider or the Transform's big dot is not on screen".into())
                };
                press(raw, a, true);
                next = 23;
            }
            23 if frames > 2 => {
                let a = out_sock(self, id(self, "slider")).unwrap_or_default();
                let b = self.node_editor.test_big_dot(id(self, "xf")).unwrap_or_default();
                let k = (frames - 2).min(6) as f32 / 6.0;
                mv(raw, a + (b - a) * k);
                if frames > 9 {
                    next = 24;
                }
            }
            24 => {
                press(raw, self.node_editor.test_big_dot(id(self, "xf")).unwrap_or_default(), false);
                next = 25;
            }
            25 if step && self.test.geotools_mark.get_or_insert_with(web_time::Instant::now).elapsed().as_secs_f32() > 0.8 => {
                // The names splay out over half a second: let them settle.
                if std::env::var_os("WOBBLE_GEOTOOLS_SHOT_MENU").is_some() {
                    SHOT_NOW.store(true, std::sync::atomic::Ordering::Relaxed);
                    return self.test.geotools = Some((stage, frames, t0));
                }
                let Some(m) = self.node_editor.test_big_dot_menu() else {
                    fail("a number dropped on the big dot asked nothing".into())
                };
                if m.options.len() != 7 || m.options.contains(&0) {
                    fail(format!("the list offered {:?}, wanted the seven numbers", m.options));
                }
                match self.node_editor.test_big_dot_row(3) {
                    Some(p) => press(raw, p, true),
                    None => {
                        if frames > 30 {
                            fail("no rotate row in the list".into());
                        }
                        return self.test.geotools = Some((stage, frames, t0));
                    }
                }
                next = 26;
            }
            26 => {
                press(raw, self.node_editor.test_big_dot_row(3).unwrap_or_default(), false);
                next = 27;
            }
            27 if step => {
                if self.egraph.input_of(id(self, "xf"), 3) != Some(id(self, "slider")) {
                    fail("picking rotate did not wire the slider into rotate".into());
                }
                if self.node_editor.test_big_dot_menu().is_some() {
                    fail("the list stayed up after a pick".into());
                }
                // Geometry onto a fresh, unwired Transform's dot: its one
                // geometry input takes it, no list.
                let at = self.egraph.node(id(self, "circle")).map(|n| n.pos).unwrap_or_default();
                let t2 = self.egraph.add(
                    NodeKind::GeoNode(G::Transform { translate: [0.0; 2], rotate: 0.0, scale: [1.0; 2], pivot: [0.0; 2], inherit: true }),
                    at + egui::vec2(-160.0, 40.0),
                );
                self.test.geotools_ids.insert("t2", t2);
                next = 28;
            }
            28 if frames > 8 => {
                let (Some(a), Some(_)) = (out_sock(self, id(self, "circle")), self.node_editor.test_big_dot(id(self, "t2"))) else {
                    fail("the circle or the new Transform's big dot is not on screen".into())
                };
                press(raw, a, true);
                next = 29;
            }
            29 if frames > 2 => {
                let a = out_sock(self, id(self, "circle")).unwrap_or_default();
                let b = self.node_editor.test_big_dot(id(self, "t2")).unwrap_or_default();
                let k = (frames - 2).min(6) as f32 / 6.0;
                mv(raw, a + (b - a) * k);
                if frames > 9 {
                    next = 30;
                }
            }
            30 => {
                press(raw, self.node_editor.test_big_dot(id(self, "t2")).unwrap_or_default(), false);
                next = 31;
            }
            31 if step => {
                if self.egraph.input_of(id(self, "t2"), 0) != Some(id(self, "circle")) {
                    let c = id(self, "circle");
                    fail(format!(
                        "geometry on the big dot did not land in the geometry input: circle feeds {:?}, t2 is fed {:?}",
                        self.egraph.edges.iter().filter(|e| e.2 == c).collect::<Vec<_>>(),
                        self.egraph.edges.iter().filter(|e| e.0 == id(self, "t2")).collect::<Vec<_>>()
                    ));
                }
                if self.node_editor.test_big_dot_menu().is_some() {
                    fail("one fitting input, yet the list came up".into());
                }
                next = 32;
            }
            // The brush: its tool makes a Brush, a drag paints a stroke.
            32 if step => match tool(self, GeoTool::Brush) {
                Some(p) => {
                    press(raw, p, true);
                    next = 33;
                }
                None => fail("no brush tool".into()),
            },
            33 => {
                press(raw, tool(self, GeoTool::Brush).unwrap_or_default(), false);
                next = 34;
            }
            34 if step => {
                let Some(b) = find(self, &|k| matches!(k, G::Brush { .. })) else { fail("the brush tool made no Brush".into()) };
                if self.selected != [b] {
                    fail("the new Brush is not selected".into());
                }
                self.test.geotools_ids.insert("brush", b);
                press(raw, g2s(self, [0.2, 0.45]), true);
                next = 35;
            }
            35 if frames > 1 => {
                let k = (frames - 1).min(12) as f32 / 12.0;
                mv(raw, g2s(self, [0.2 + 0.2 * k, 0.45 + 0.03 * (k * 6.0).sin()]));
                if frames > 14 {
                    next = 36;
                }
            }
            36 => {
                press(raw, g2s(self, [0.4, 0.45]), false);
                next = 37;
            }
            37 if step => {
                let Some(G::Brush { strokes, .. }) = kind(self, id(self, "brush")) else { fail("brush gone".into()) };
                let n = strokes.first().map_or(0, |s| s.pts.len());
                if strokes.len() != 1 || n < 6 {
                    fail(format!("a drag painted {} strokes, the first {n} points", strokes.len()));
                }
                let g = self.live_geo();
                let cx = engine::geo::EvalCx { tick: 0.0, field: None, seed: 0.0, aspect: 0.5, px_per_unit: 1000.0 };
                if engine::geo::eval_node(&g, id(self, "brush"), &cx).prims.len() != 1 {
                    fail("the stroke drew no shape".into());
                }
                // Snapping: a circle dropped near the star's centre lands on it.
                self.prefs.geo_viz.snap = true;
                match tool(self, GeoTool::Circle) {
                    Some(p) => press(raw, p, true),
                    None => fail("no circle tool".into()),
                }
                next = 38;
            }
            38 => {
                press(raw, tool(self, GeoTool::Circle).unwrap_or_default(), false);
                next = 39;
            }
            39 if step => {
                press(raw, g2s(self, [0.753, 0.302]), true);
                next = 40;
            }
            40 if frames > 2 => {
                press(raw, g2s(self, [0.753, 0.302]), false);
                next = 41;
            }
            41 if step => {
                let c2 = self.egraph.nodes.iter().filter_map(|(i, n)| match &n.kind {
                    NodeKind::GeoNode(G::Circle { center, .. }) if *i != id(self, "circle") => Some(*center),
                    _ => None,
                }).next();
                match c2 {
                    Some(c) if (c[0] - 0.75).abs() < 1e-5 && (c[1] - 0.3).abs() < 1e-5 => {}
                    other => fail(format!("snapping did not land the circle on the star's centre: {other:?}")),
                }
                // z: ] over the circle's noodle into the Merge raises it.
                let out = find(self, &|k| matches!(k, G::Output)).unwrap_or(0);
                let m = self.egraph.input_of(out, 0).unwrap_or(0);
                let c = id(self, "circle");
                let Some(i) = self.egraph.edges.iter().find(|e| e.0 == m && e.2 == c).map(|e| e.1) else {
                    fail("the circle does not feed the Merge".into())
                };
                let (Some(a), Some(r)) = (out_sock(self, c), self.node_rects.iter().find(|(n, _)| *n == m).map(|(_, r)| *r)) else {
                    fail("the circle or the Merge is off screen".into())
                };
                let mid = a + (egui::pos2(r.center().x, r.top()) - a) * 0.5;
                mv(raw, mid);
                self.test.geotools_ids.insert("merge", m);
                self.test.geotools_ids.insert("merge_in", i as u64);
                next = 42;
            }
            42 if frames > 3 => {
                raw.events.push(egui::Event::Key {
                    key: egui::Key::CloseBracket,
                    physical_key: None,
                    pressed: true,
                    repeat: false,
                    modifiers: Default::default(),
                });
                next = 43;
            }
            43 => {
                raw.events.push(egui::Event::Key {
                    key: egui::Key::CloseBracket,
                    physical_key: None,
                    pressed: false,
                    repeat: false,
                    modifiers: Default::default(),
                });
                next = 44;
            }
            44 if step => {
                let l = self.egraph.wire_layering(id(self, "merge"), id(self, "merge_in") as usize);
                if l.z != 1.0 {
                    fail(format!("] over the circle's noodle left its z at {}", l.z));
                }
                // A click on the painted stroke selects the Brush, and its
                // handles come up: drag its centre down a little.
                self.prefs.geo_viz.snap = false;
                press(raw, g2s(self, [0.3, 0.45]), true);
                next = 45;
            }
            45 if frames > 2 => {
                press(raw, g2s(self, [0.3, 0.45]), false);
                next = 46;
            }
            46 if step => {
                let b = id(self, "brush");
                // Selecting it through the viewer: the Brush or what places it.
                let sel_ok = self.selected == [b]
                    || self.selected.first().is_some_and(|s| self.egraph.input_of(*s, 0) == Some(b));
                if !sel_ok {
                    fail(format!("a click on the stroke selected {:?}", self.selected));
                }
                let Some(k) = self.selected.first().and_then(|s| kind(self, *s)) else { fail("nothing selected".into()) };
                let Some(fr) = geo_frame(&k, [0.0, 0.0]) else { fail(format!("{} has no handles", k.name())) };
                self.test.geotools_ids.insert("pick", self.selected[0]);
                self.test.geotools_ids.insert("fx", (fr.centre[0] * 1e6) as u64);
                self.test.geotools_ids.insert("fy", (fr.centre[1] * 1e6) as u64);
                press(raw, g2s(self, fr.centre), true);
                next = 47;
            }
            47 if frames > 2 => {
                let c = [id(self, "fx") as f32 / 1e6, id(self, "fy") as f32 / 1e6];
                let k = (frames - 2).min(4) as f32 / 4.0;
                mv(raw, g2s(self, [c[0], c[1] + 0.05 * k]));
                if frames > 7 {
                    next = 48;
                }
            }
            48 => {
                let c = [id(self, "fx") as f32 / 1e6, id(self, "fy") as f32 / 1e6];
                press(raw, g2s(self, [c[0], c[1] + 0.05]), false);
                next = 49;
            }
            49 if step => {
                let c = [id(self, "fx") as f32 / 1e6, id(self, "fy") as f32 / 1e6];
                let k = kind(self, id(self, "pick"));
                let now = k.as_ref().and_then(|k| geo_frame(k, [0.0, 0.0])).map(|f| f.centre);
                if !now.is_some_and(|n| (n[1] - c[1] - 0.05).abs() < 0.01) {
                    fail(format!("the stroke's handles did not move it: centre {c:?} -> {now:?}"));
                }
                if matches!(k, Some(G::Brush { ref strokes, .. }) if strokes.len() != 1) {
                    fail("dragging the handle painted another stroke".into());
                }
                // Viewing the first circle, a new rectangle joins it in a
                // Merge that is then what is viewed.
                self.egraph.view = Some(id(self, "circle"));
                match tool(self, GeoTool::Rectangle) {
                    Some(p) => press(raw, p, true),
                    None => fail("no rectangle tool".into()),
                }
                next = 50;
            }
            50 => {
                press(raw, tool(self, GeoTool::Rectangle).unwrap_or_default(), false);
                next = 51;
            }
            51 if step => {
                press(raw, g2s(self, [0.2, 0.15]), true);
                next = 52;
            }
            52 if frames > 2 => {
                press(raw, g2s(self, [0.2, 0.15]), false);
                next = 53;
            }
            53 if step => {
                let Some(rc) = find(self, &|k| matches!(k, G::Rectangle { .. })) else { fail("no Rectangle drawn".into()) };
                let c = id(self, "circle");
                let v = self.egraph.view;
                let viewed_merge = v.is_some_and(|m| {
                    matches!(kind(self, m), Some(G::Merge { .. }))
                        && self.egraph.input_of(m, 0) == Some(c)
                        && self.egraph.input_of(m, 1) == Some(rc)
                });
                if !viewed_merge {
                    fail(format!("viewing the circle, a new rectangle did not land in a viewed Merge with it (view {v:?})"));
                }
                if self.egraph.edges.iter().any(|e| e.2 == c && Some(e.0) != v) {
                    fail("the circle still feeds past the new Merge".into());
                }
                self.test.geotools_ids.insert("rect", rc);
                // The fill swatch: a Style after the rectangle, its panel up.
                match tool(self, GeoTool::Fill) {
                    Some(p) => press(raw, p, true),
                    None => fail("no fill swatch on the tool bar".into()),
                }
                next = 54;
            }
            54 => {
                press(raw, tool(self, GeoTool::Fill).unwrap_or_default(), false);
                next = 55;
            }
            55 if step => {
                let rc = id(self, "rect");
                let Some(st) = find(self, &|k| matches!(k, G::Style { .. })) else { fail("the fill swatch made no Style".into()) };
                if self.egraph.input_of(st, 0) != Some(rc) || self.selected != [st] {
                    fail("the Style is not after the rectangle, selected".into());
                }
                if !(self.props_visible && self.bottom == BottomTab::Properties && self.prop_order.first() == Some(&st)) {
                    fail("the Style's properties did not open".into());
                }
                let Some(G::Style { style }) = kind(self, st) else { unreachable!() };
                let Some(G::Rectangle { style: rs, .. }) = kind(self, rc) else { unreachable!() };
                if style != rs {
                    fail("the new Style should start as the shape's own colours".into());
                }
                // Shift-connect to a big dot: a fresh Transform parked with
                // its dot on the circle's output takes it into geometry.
                let c = id(self, "circle");
                let cp = self.egraph.node(c).map(|n| n.pos).unwrap_or_default();
                let t3 = self.egraph.add(
                    NodeKind::GeoNode(G::Transform { translate: [0.0; 2], rotate: 0.0, scale: [1.0; 2], pivot: [0.0; 2], inherit: true }),
                    cp + egui::vec2(0.0, 2.0 * editor::SOCKET_OFFSET + 4.0),
                );
                self.test.geotools_ids.insert("t3", t3);
                self.test.geotools_mark = Some(web_time::Instant::now());
                next = 56;
            }
            56 => {
                // Shift held over the graph for a moment: the dot courts the
                // output and shift lands it.
                if let Some((_, r)) = self.node_rects.iter().find(|(i, _)| *i == id(self, "t3")) {
                    raw.events.insert(0, egui::Event::ModifiersChanged(egui::Modifiers { shift: true, ..Default::default() }));
                    raw.events.push(egui::Event::PointerMoved(r.center() + egui::vec2(40.0, 0.0)));
                }
                if self.test.geotools_mark.is_some_and(|m| m.elapsed().as_secs_f32() > 1.0) {
                    raw.events.push(egui::Event::ModifiersChanged(egui::Modifiers::default()));
                    next = 57;
                }
            }
            57 if step => {
                let (t3, c) = (id(self, "t3"), id(self, "circle"));
                if self.egraph.input_of(t3, 0) != Some(c) {
                    fail(format!(
                        "shift did not land the circle on the big dot: t3 fed {:?}",
                        self.egraph.edges.iter().filter(|e| e.0 == t3).collect::<Vec<_>>()
                    ));
                }
                // A linear gradient on the rectangle's Style: its handles
                // land across the shape, and the end one drags.
                let Some(st) = find(self, &|k| matches!(k, G::Style { .. })) else { fail("the Style is gone".into()) };
                if let Some(NodeKind::GeoNode(G::Style { style })) = self.egraph.node_mut(st).map(|n| &mut n.kind) {
                    style.grad.kind = engine::geo::GradKind::Linear;
                    style.grad.to = [0.0, 0.0, 1.0, 1.0];
                }
                self.geo_select(st);
                self.test.geotools_ids.insert("style", st);
                next = 58;
            }
            58 if step => {
                let Some(G::Style { style }) = kind(self, id(self, "style")) else { fail("style gone".into()) };
                let dflt = engine::geo::Gradient::default();
                if style.grad.b == dflt.b {
                    if frames > 200 {
                        fail("the gradient's handles were never laid across the shape".into());
                    }
                    return self.test.geotools = Some((stage, frames, t0));
                }
                press(raw, g2s(self, style.grad.b), true);
                self.test.geotools_ids.insert("gbx", (style.grad.b[0] * 1e6) as u64);
                self.test.geotools_ids.insert("gby", (style.grad.b[1] * 1e6) as u64);
                next = 59;
            }
            59 if frames > 2 => {
                let b = [id(self, "gbx") as f32 / 1e6, id(self, "gby") as f32 / 1e6];
                let k = (frames - 2).min(4) as f32 / 4.0;
                mv(raw, g2s(self, [b[0] - 0.05 * k, b[1] + 0.05 * k]));
                if frames > 7 {
                    next = 60;
                }
            }
            60 => {
                let b = [id(self, "gbx") as f32 / 1e6, id(self, "gby") as f32 / 1e6];
                press(raw, g2s(self, [b[0] - 0.05, b[1] + 0.05]), false);
                next = 61;
            }
            61 if step => {
                let b = [id(self, "gbx") as f32 / 1e6, id(self, "gby") as f32 / 1e6];
                let Some(G::Style { style }) = kind(self, id(self, "style")) else { fail("style gone".into()) };
                let nb = style.grad.b;
                if (nb[0] - (b[0] - 0.05)).abs() > 0.006 || (nb[1] - (b[1] + 0.05)).abs() > 0.006 {
                    fail(format!("the gradient's end handle did not follow the drag: {b:?} -> {nb:?}"));
                }
                // The gradient tab: the Gradient node's ramp editor, and a
                // click on its bar adds a stop.
                self.props_visible = true;
                self.bottom = BottomTab::Properties;
                next = 62;
            }
            62 if step => match widgets::noted_rect(&self.egui_ctx, egui::Id::new(("panel-tab-rect", "gradient"))) {
                Some((r, clip)) if clip.contains(r.center()) => {
                    press(raw, r.center(), true);
                    next = 63;
                }
                _ if frames > 400 => fail("no gradient tab on the Style's panel".into()),
                _ => {}
            },
            63 => {
                let r = widgets::noted_rect(&self.egui_ctx, egui::Id::new(("panel-tab-rect", "gradient")));
                press(raw, r.map(|r| r.0.center()).unwrap_or_default(), false);
                next = 64;
            }
            64 if step => match widgets::noted_rect(&self.egui_ctx, egui::Id::new(("grad-ramp", id(self, "style")))) {
                Some((r, clip)) if clip.contains(r.center()) => {
                    // Midway along the bar, away from both end stops.
                    let p = egui::pos2(r.left() + r.width() * 0.5, r.top() + 10.0);
                    press(raw, p, true);
                    self.test.geotools_ids.insert("rampx", (p.x * 10.0) as u64);
                    self.test.geotools_ids.insert("rampy", (p.y * 10.0) as u64);
                    next = 65;
                }
                _ if frames > 400 => fail("the gradient tab shows no ramp editor".into()),
                _ => {}
            },
            65 if frames > 2 => {
                let p = egui::pos2(id(self, "rampx") as f32 / 10.0, id(self, "rampy") as f32 / 10.0);
                press(raw, p, false);
                next = 66;
            }
            66 if step => {
                let Some(G::Style { style }) = kind(self, id(self, "style")) else { fail("style gone".into()) };
                if style.grad.stops.n != 3 {
                    fail(format!("a click on the ramp left {} stops (want 3)", style.grad.stops.n));
                }
                // Direct select on the circle: an Edit after it, and its
                // first point dragged by hand.
                self.geo_select(id(self, "circle"));
                match tool(self, GeoTool::Direct) {
                    Some(p) => press(raw, p, true),
                    None => fail("no direct-select tool".into()),
                }
                next = 67;
            }
            67 => {
                press(raw, tool(self, GeoTool::Direct).unwrap_or_default(), false);
                next = 68;
            }
            68 if step => {
                let c = id(self, "circle");
                let Some(e) = find(self, &|k| matches!(k, G::Edit { .. })) else { fail("direct select made no Edit".into()) };
                if self.egraph.input_of(e, 0) != Some(c) || self.selected != [e] {
                    fail("the Edit is not after the circle, selected".into());
                }
                self.test.geotools_ids.insert("edit", e);
                // Its points come from the gizmo's own evaluation.
                let p0 = self
                    .geo.hud_cache
                    .as_ref()
                    .and_then(|cc| cc.geos.iter().find(|(n, _)| *n == e))
                    .and_then(|(_, g)| g.prims.first().and_then(|pr| pr.verts.first()).and_then(|v| g.points.get(*v as usize)))
                    .map(|q| q.p);
                let Some(p0) = p0 else {
                    if frames > 300 {
                        fail("the Edit's points never came up".into());
                    }
                    return self.test.geotools = Some((stage, frames, t0));
                };
                self.test.geotools_ids.insert("px", (p0[0] * 1e6) as u64);
                self.test.geotools_ids.insert("py", (p0[1] * 1e6) as u64);
                press(raw, g2s(self, p0), true);
                next = 69;
            }
            69 if frames > 2 => {
                let p0 = [id(self, "px") as f32 / 1e6, id(self, "py") as f32 / 1e6];
                let k = (frames - 2).min(4) as f32 / 4.0;
                mv(raw, g2s(self, [p0[0] + 0.05 * k, p0[1]]));
                if frames > 7 {
                    next = 70;
                }
            }
            70 => {
                let p0 = [id(self, "px") as f32 / 1e6, id(self, "py") as f32 / 1e6];
                press(raw, g2s(self, [p0[0] + 0.05, p0[1]]), false);
                next = 71;
            }
            71 if step => {
                let Some(G::Edit { edits }) = kind(self, id(self, "edit")) else { fail("edit gone".into()) };
                let ok = edits.len() == 1
                    && edits[0].prim == 0
                    && edits[0].vert == 0
                    && (edits[0].offset[0] - 0.05).abs() < 0.004
                    && edits[0].offset[1].abs() < 0.004;
                if !ok {
                    fail(format!("dragging the first point left edits {edits:?}"));
                }
                // Alt-click puts it back.
                let p0 = [id(self, "px") as f32 / 1e6 + 0.05, id(self, "py") as f32 / 1e6];
                raw.events.push(egui::Event::PointerMoved(g2s(self, p0)));
                raw.events.push(egui::Event::PointerButton {
                    pos: g2s(self, p0),
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers { alt: true, ..Default::default() },
                });
                raw.events.insert(0, egui::Event::ModifiersChanged(egui::Modifiers { alt: true, ..Default::default() }));
                next = 72;
            }
            72 => {
                let p0 = [id(self, "px") as f32 / 1e6 + 0.05, id(self, "py") as f32 / 1e6];
                raw.events.insert(0, egui::Event::ModifiersChanged(egui::Modifiers { alt: true, ..Default::default() }));
                raw.events.push(egui::Event::PointerButton {
                    pos: g2s(self, p0),
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers: egui::Modifiers { alt: true, ..Default::default() },
                });
                next = 73;
            }
            73 if step => {
                raw.events.push(egui::Event::ModifiersChanged(egui::Modifiers::default()));
                let Some(G::Edit { edits }) = kind(self, id(self, "edit")) else { fail("edit gone".into()) };
                if !edits.is_empty() {
                    fail(format!("alt-click did not put the point back: {edits:?}"));
                }
                // Particles merged into the drawing, and onion skins on:
                // at frame 20 the skins either side are drawn.
                let kind = GeoTool::Brush.make(); // any catalog lookup works; build the node directly
                let _ = kind;
                let parts = editor::node_catalog()
                    .into_iter()
                    .find(|e| e.cat.is_geo() && e.label == "Particles")
                    .and_then(|e| match (e.make)(0) {
                        NodeKind::GeoNode(k) => Some(k),
                        _ => None,
                    });
                let Some(parts) = parts else { fail("no Particles in the catalog".into()) };
                self.egraph.view = None;
                let p = self.geo_tool_add(parts);
                self.test.geotools_ids.insert("parts", p);
                self.prefs.geo_viz.onion = true;
                self.seek_to(1020);
                self.test.geotools_mark = Some(web_time::Instant::now());
                next = 74;
            }
            74 if self.test.geotools_mark.is_some_and(|m| m.elapsed().as_secs_f32() > 1.0) => {
                let near: Vec<u64> = self.geo.onion_cache.iter().map(|(f, ..)| *f).collect();
                if !(near.contains(&1018) && near.contains(&1022)) {
                    fail(format!("onion skins were not drawn either side of 1020: {near:?}"));
                }
                let pts = |f: u64| self.geo.onion_cache.iter().find(|(ff, ..)| *ff == f).map_or(0, |(_, _, g)| g.points.len());
                if pts(1018) == 0 || pts(1018) == pts(1022) && pts(1018) == 0 {
                    fail("the onion skins hold no particles".into());
                }
                self.prefs.geo_viz.onion = false;
                // A Skeleton skinning a bar, the skeleton picked.
                let cat = |label: &str| {
                    editor::node_catalog()
                        .into_iter()
                        .find(|e| e.cat.is_geo() && e.label == label)
                        .and_then(|e| match (e.make)(0) {
                            NodeKind::GeoNode(k) => Some(k),
                            _ => None,
                        })
                };
                let (Some(sk), Some(skin)) = (cat("Skeleton"), cat("Skin")) else { fail("no Skeleton / Skin in the catalog".into()) };
                self.egraph.view = None;
                let skel = self.geo_tool_add(sk);
                let bar = self.egraph.add(
                    NodeKind::GeoNode(G::Rectangle { center: [0.495, 0.35], size: [0.39, 0.03], corner: 0.0, rotation: 0.0, style: Default::default() }),
                    egui::Pos2::new(-200.0, 300.0),
                );
                let sn = self.egraph.add(NodeKind::GeoNode(skin), egui::Pos2::new(-100.0, 400.0));
                self.egraph.connect(sn, 0, bar);
                self.egraph.connect(sn, 1, skel);
                self.egraph.view = Some(sn);
                self.selected = vec![skel];
                self.test.geotools_ids.insert("skel", skel);
                self.test.geotools_ids.insert("skin", sn);
                self.playback_dirty = true;
                next = 75;
            }
            // Turn bone 1 a quarter round its head by dragging its body.
            75 if frames == 11 => press(raw, g2s(self, [0.5, 0.35]), true),
            75 if frames > 14 => next = 76,
            76 | 77 if step => {
                let pv = g2s(self, [0.43, 0.35]);
                let r = g2s(self, [0.5, 0.35]) - pv;
                let a = std::f32::consts::FRAC_PI_4 * (stage - 75) as f32;
                let (sa, ca) = a.sin_cos();
                mv(raw, pv + egui::vec2(r.x * ca - r.y * sa, r.x * sa + r.y * ca));
                next = stage + 1;
            }
            78 if step => {
                let pv = g2s(self, [0.43, 0.35]);
                let r = g2s(self, [0.5, 0.35]) - pv;
                press(raw, pv + egui::vec2(-r.y, r.x), false);
                next = 79;
            }
            79 if frames > 10 => {
                let Some(G::Skeleton { pose, .. }) = kind(self, id(self, "skel")) else { fail("the skeleton went".into()) };
                let turn = pose.get(1).copied().unwrap_or(0.0);
                if (turn - std::f32::consts::FRAC_PI_2).abs() > 0.06 || pose[0].abs() > 1e-4 {
                    fail(format!("a quarter-turn drag on bone 1 posed {pose:?}"));
                }
                let g = editor_to_geo(&self.egraph);
                let (w, h) = self.comp.output_size().unwrap_or((1920, 1080));
                let cx = |t: f32| engine::geo::EvalCx { tick: t, field: None, seed: 0.0, aspect: h as f32 / w.max(1) as f32, px_per_unit: w as f32 };
                let bent = engine::geo::eval_node(&g, id(self, "skin"), &cx(0.0));
                let drop = bent.points.iter().map(|q| q.p[1] - 0.35).fold(0.0f32, f32::max);
                if drop < 0.1 {
                    fail(format!("the skinned bar did not bend with the bone (lowest point {drop} under the line)"));
                }
                // A Ragdoll off the skeleton, baked: it falls.
                let rk = editor::node_catalog()
                    .into_iter()
                    .find(|e| e.cat.is_geo() && e.label == "Ragdoll")
                    .and_then(|e| match (e.make)(0) {
                        NodeKind::GeoNode(k) => Some(k),
                        _ => None,
                    });
                let Some(rk) = rk else { fail("no Ragdoll in the catalog".into()) };
                let mut rk = rk;
                if let G::Ragdoll { walls, .. } = &mut rk {
                    *walls = false; // the bent arm reaches past the floor
                }
                let rd = self.egraph.add(NodeKind::GeoNode(rk), egui::Pos2::new(100.0, 400.0));
                self.egraph.connect(rd, 0, id(self, "skel"));
                self.ragdoll_bake(rd, false);
                match kind(self, rd) {
                    Some(G::Ragdoll { bake, .. }) if !bake.is_empty() => {}
                    _ => fail("the ragdoll did not bake".into()),
                }
                let g = editor_to_geo(&self.egraph);
                let root_y = |t: f32| {
                    let b = engine::geo::eval_node(&g, rd, &cx(t)).bones;
                    b.first().map_or(f32::NAN, |b| engine::georig::apply(&b.xf, b.head)[1])
                };
                if !(root_y(60.0) > root_y(0.0) + 0.05) {
                    fail(format!("the ragdoll did not fall: root at {} then {}", root_y(0.0), root_y(60.0)));
                }
                // Under WOBBLE_SHOT, hold the finished drawing a moment
                // and take the shot of it (WOBBLE_SHOT_AFTER large).
                if std::env::var_os("WOBBLE_SHOT").is_some() {
                    next = 99;
                }
                println!(
                    "GEOTOOLS TEST PASS: drew a circle and a star from the tool bar, moved them by their handles, \
                     a Transform tool turned about the star and a click on it picked the Transform, the big dot \
                     asked which number and took the geometry straight in, a brush drag painted a stroke, snapping \
                     landed a circle on the star's centre, ] raised a noodle's z, a shape drawn while viewing \
                     another merged with it on screen, the fill swatch opened a Style, shift landed geometry \
                     on a big dot, a linear gradient's end handle dragged, a click added a ramp stop and an Edit dragged \
                     a circle's point by hand and alt-click put it back; onion skins drew particles either side \
                     of the playhead; dragging a bone a quarter round its head posed it and bent the skinned bar, \
                     and a baked Ragdoll fell"
                );
                if next != 99 {
                    std::process::exit(0);
                }
            }
            99 if frames > 40 => SHOT_NOW.store(true, std::sync::atomic::Ordering::Relaxed),
            _ => {}
        }
        self.test.geotools = Some((next, if next == stage { frames } else { 0 }, t0));
    }
}
