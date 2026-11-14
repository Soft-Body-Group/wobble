//! `--test-paint`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    /// --test-paint driver: splice a Paint node above the merge, select it,
    /// drag a stroke across the viewer, verify points + a live render.
    pub(crate) fn drive_test_paint(&mut self, raw: &mut egui::RawInput) {
        let Some((stage, mut frames, pid)) = self.test.paint else { return };
        frames += 1;
        if frames > 900 {
            println!("PAINT TEST FAIL: timeout at stage {stage}");
            std::process::exit(1);
        }
        let press = |raw: &mut egui::RawInput, pos: egui::Pos2, down: bool| {
            // Clears any modifier a previous stage set: `raw.modifiers` is
            // sticky state, not a per-event flag.
            raw.events.insert(0, egui::Event::ModifiersChanged(Default::default()));
            raw.events.push(egui::Event::PointerMoved(pos));
            raw.events.push(egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed: down,
                modifiers: Default::default(),
            });
        };
        let press_mod = |raw: &mut egui::RawInput,
                         pos: egui::Pos2,
                         down: bool,
                         modifiers: egui::Modifiers| {
            // `InputState::modifiers` reads the RawInput field, not the one
            // riding the event — the paint gesture asks `i.modifiers.alt`.
            raw.events.insert(0, egui::Event::ModifiersChanged(modifiers));
            raw.events.push(egui::Event::PointerMoved(pos));
            raw.events.push(egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed: down,
                modifiers,
            });
        };
        let brush_key = |raw: &mut egui::RawInput, k: egui::Key, m: egui::Modifiers| {
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
        let strokes = |s: &Self| -> Vec<engine::graph::PaintStroke> {
            match s.egraph.node(pid).map(|n| &n.kind) {
                Some(NodeKind::Paint(pp)) => pp.strokes.clone(),
                _ => Vec::new(),
            }
        };
        let c = self.image_rect.center();
        let mut next = stage;
        match stage {
            0 if frames >= 6 => {
                // Default graph: gr=3 -> mg=4 input 0. Splice paint between.
                let id = self.egraph.add(
                    NodeKind::Paint(Default::default()),
                    egui::Pos2::new(60.0, 60.0),
                );
                self.egraph.connect(id, 0, 3);
                self.egraph.connect(4, 0, id);
                self.selected = vec![id];
                self.test.paint = Some((1, frames, id));
                return;
            }
            1 => {
                press(raw, c - egui::vec2(380.0, 0.0), true);
                next = 2;
            }
            2 | 3 | 4 | 5 => {
                let t = (stage - 1) as f32;
                raw.events.push(egui::Event::PointerMoved(
                    c - egui::vec2(380.0, 0.0) + egui::vec2(60.0 * t, 8.0 * t),
                ));
                next = stage + 1;
            }
            6 => {
                press(raw, c + egui::vec2(120.0, 32.0), false);
                next = 7;
            }
            7 => next = 8, // settle: layer records + fused pass samples it
            8 => {
                let stroke = match self.egraph.node(pid).map(|n| &n.kind) {
                    Some(NodeKind::Paint(pp)) => pp.strokes.first(),
                    _ => None,
                };
                let pts = stroke.map(|st| st.points.len()).unwrap_or(0);
                if pts < 3 {
                    println!("PAINT TEST FAIL: stroke has {pts} points");
                    std::process::exit(1);
                }
                // The brush keys: [ shrinks, ] grows, shift softens, alt
                // fades. All four off ONE key pair, and only while a Paint
                // node is selected.
                self.brush.radius = 40.0;
                self.brush.soft = 0.5;
                self.brush.color[3] = 1.0;
                brush_key(raw, egui::Key::OpenBracket, egui::Modifiers::NONE);
                next = 9;
            }
            9 => {
                brush_key(raw, egui::Key::CloseBracket, egui::Modifiers::SHIFT);
                next = 10;
            }
            10 => {
                // alt+[ fades — `]` is the GROW key and would clamp at 1.0.
                brush_key(raw, egui::Key::OpenBracket, egui::Modifiers::ALT);
                next = 11;
            }
            11 => {
                let (r, sf, op) = (self.brush.radius, self.brush.soft, self.brush.color[3]);
                if (r - 32.0).abs() > 0.01 || (sf - 0.6).abs() > 0.01 || (op - 0.9).abs() > 0.01
                {
                    println!(
                        "PAINT TEST FAIL: brush keys gave radius {r:.2} (want 32), \
                         soft {sf:.2} (want 0.60), opacity {op:.2} (want 0.90)"
                    );
                    std::process::exit(1);
                }
                // Per-frame strokes: author on tick 3 with the node set to
                // "this frame", and the stroke must carry that frame.
                self.seek_to(3);
                if let Some(editor::ENode { kind: NodeKind::Paint(pp), .. }) =
                    self.egraph.node_mut(pid)
                {
                    pp.life_default = engine::graph::StrokeLife::Single(0);
                }
                next = 12;
            }
            12 => {
                if self.force_display || self.eval_tick() != 3 {
                    return; // wait for the seek to land
                }
                press(raw, c + egui::vec2(-40.0, 90.0), true);
                next = 13;
            }
            13 => {
                raw.events
                    .push(egui::Event::PointerMoved(c + egui::vec2(40.0, 90.0)));
                next = 14;
            }
            14 => {
                press(raw, c + egui::vec2(40.0, 90.0), false);
                next = 15;
            }
            15 => next = 16, // settle
            16 => {
                use engine::graph::StrokeLife;
                let ss = strokes(self);
                let life = ss.last().map(|s| s.life);
                if life != Some(StrokeLife::Single(3)) {
                    println!(
                        "PAINT TEST FAIL: per-frame stroke landed on {life:?}, \
                         wanted Single(3)"
                    );
                    std::process::exit(1);
                }
                // ...and it is only THERE on that frame, which is what the
                // rasterizer keys its layer on.
                let at = |f: u64| {
                    ss.iter().filter(|s| s.life.covers(f)).count()
                };
                if at(3) != 2 || at(0) != 1 {
                    println!(
                        "PAINT TEST FAIL: {} strokes on frame 3, {} on frame 0; \
                         wanted 2 and 1",
                        at(3),
                        at(0)
                    );
                    std::process::exit(1);
                }
                // Clone: alt-click sets the source, the next stroke fixes
                // the offset against its own first point.
                self.brush.mode = engine::graph::BrushMode::Clone;
                self.brush.clone_src = None;
                self.brush.clone_off = None;
                press_mod(raw, c + egui::vec2(150.0, -60.0), true, egui::Modifiers::ALT);
                next = 17;
            }
            17 => {
                press_mod(raw, c + egui::vec2(150.0, -60.0), false, egui::Modifiers::ALT);
                next = 18;
            }
            18 => {
                if self.brush.clone_src.is_none() {
                    println!("PAINT TEST FAIL: alt-click did not set the clone source");
                    std::process::exit(1);
                }
                press(raw, c + egui::vec2(-150.0, -60.0), true);
                next = 19;
            }
            19 => {
                raw.events
                    .push(egui::Event::PointerMoved(c + egui::vec2(-90.0, -60.0)));
                next = 20;
            }
            20 => {
                press(raw, c + egui::vec2(-90.0, -60.0), false);
                next = 21;
            }
            21 => next = 22, // settle
            22 => {
                let ss = strokes(self);
                let Some(off) = ss.last().and_then(|s| s.clone_off) else {
                    println!("PAINT TEST FAIL: the clone stroke carries no offset");
                    std::process::exit(1);
                };
                // Source was 300px right of where the stroke started, so the
                // offset is the source minus the stroke's FIRST point, in the
                // node's own normalized space. (It used to be judged against
                // `300 / image_rect.width()`, which is not the node's format
                // width on screen, and failed at 0.8 of it.)
                let want = match (self.brush.clone_src, ss.last().and_then(|s| s.points.first())) {
                    (Some(src), Some(p0)) => src[0] - p0[0],
                    _ => f32::NAN,
                };
                if !(want > 0.0) || (off[0] - want).abs() > want * 0.02 || off[1].abs() > 0.02 {
                    println!(
                        "PAINT TEST FAIL: clone offset {off:?}, wanted about \
                         ({want:.4}, 0)"
                    );
                    std::process::exit(1);
                }
                // A clone stroke is a different SHADER, not just different
                // stamps: that is what makes the second read of the input
                // exist at all.
                let g = self.egraph.to_engine_graph_at(Some(self.eval_tick()));
                let h = g.topology_hash();
                let mut plain = g.clone();
                if let Some(NodeKind::Paint(pp)) = plain.nodes.get_mut(&pid) {
                    for st in pp.strokes.iter_mut() {
                        st.clone_off = None;
                        st.mode = engine::graph::BrushMode::Color;
                    }
                }
                if plain.topology_hash() == h {
                    println!("PAINT TEST FAIL: cloning never reached the shader");
                    std::process::exit(1);
                }
                // ...while more strokes never do.
                let mut more = g.clone();
                if let Some(NodeKind::Paint(pp)) = more.nodes.get_mut(&pid) {
                    let extra = pp.strokes[0].clone();
                    pp.strokes.push(extra);
                }
                if more.topology_hash() != h {
                    println!("PAINT TEST FAIL: painting a stroke recompiled the shader");
                    std::process::exit(1);
                }
                // HELD F + motion: Blender's brush gesture. Hold the key,
                // move right 40px, let go: the radius grew and nothing was
                // painted. Ctrl held with it moves the aspect instead.
                self.brush.mode = engine::graph::BrushMode::Color;
                self.brush.radius = 40.0;
                self.brush.aspect = 1.0;
                raw.events.push(egui::Event::PointerMoved(c));
                raw.events.push(egui::Event::Key {
                    key: egui::Key::F,
                    physical_key: None,
                    pressed: true,
                    repeat: false,
                    modifiers: Default::default(),
                });
                next = 23;
            }
            23 => {
                raw.events.push(egui::Event::PointerMoved(c + egui::vec2(40.0, 0.0)));
                next = 24;
            }
            24 => {
                raw.events.push(egui::Event::Key {
                    key: egui::Key::F,
                    physical_key: None,
                    pressed: false,
                    repeat: false,
                    modifiers: Default::default(),
                });
                next = 25;
            }
            25 => {
                let n_before = strokes(self).len();
                if self.brush.radius <= 40.0 * 1.2 {
                    println!(
                        "PAINT TEST FAIL: held F + 40px right left the radius at {:.1}",
                        self.brush.radius
                    );
                    std::process::exit(1);
                }
                // One all-frames stroke, one on frame 3, the clone: three.
                if n_before != 3 {
                    println!("PAINT TEST FAIL: the F gesture painted ({n_before} strokes, want 3)");
                    std::process::exit(1);
                }
                // A DODGE stroke carries its mode, its stops and its shape.
                self.brush.mode = engine::graph::BrushMode::Dodge;
                self.brush.stops = 0.75;
                self.brush.aspect = 0.5;
                self.brush.angle = 30.0;
                press(raw, c + egui::vec2(-60.0, 40.0), true);
                next = 26;
            }
            26 => {
                raw.events.push(egui::Event::PointerMoved(c + egui::vec2(0.0, 40.0)));
                next = 27;
            }
            27 => {
                press(raw, c + egui::vec2(0.0, 40.0), false);
                next = 28;
            }
            28 => next = 29, // settle
            29 => {
                use engine::graph::{BrushMode, PaintOutput};
                let ss = strokes(self);
                let Some(last) = ss.last() else {
                    println!("PAINT TEST FAIL: the dodge stroke never landed");
                    std::process::exit(1);
                };
                if last.mode() != BrushMode::Dodge
                    || (last.strength - 0.75).abs() > 1e-4
                    || (last.aspect - 0.5).abs() > 1e-4
                    || (last.angle - 30.0).abs() > 1e-4
                {
                    println!(
                        "PAINT TEST FAIL: dodge stroke is {:?} strength {} aspect {} angle {}",
                        last.mode(),
                        last.strength,
                        last.aspect,
                        last.angle
                    );
                    std::process::exit(1);
                }
                // Dodge is its own topology bit (an exposure read), and the
                // matte output is another — and it turns the socket green.
                let g = self.egraph.to_engine_graph_at(Some(self.eval_tick()));
                let h = g.topology_hash();
                let mut no_tone = g.clone();
                if let Some(NodeKind::Paint(pp)) = no_tone.nodes.get_mut(&pid) {
                    pp.strokes.retain(|s| s.mode() != BrushMode::Dodge);
                }
                if no_tone.topology_hash() == h {
                    println!("PAINT TEST FAIL: a dodge stroke never reached the shader");
                    std::process::exit(1);
                }
                if let Some(editor::ENode { kind: NodeKind::Paint(pp), .. }) =
                    self.egraph.node_mut(pid)
                {
                    pp.output = PaintOutput::Matte;
                }
                let g2 = self.egraph.to_engine_graph_at(Some(self.eval_tick()));
                if g2.topology_hash() == h {
                    println!("PAINT TEST FAIL: matte output did not change the shader");
                    std::process::exit(1);
                }
                if g2.nodes[&pid].output_kind() != engine::graph::PortKind::Mask {
                    println!("PAINT TEST FAIL: matte output did not turn the socket into a mask");
                    std::process::exit(1);
                }
                next = 30;
            }
            30 => {
                // PEN + BLEND: a multiply stroke painted with a pen whose
                // pressure rises from 0.3 to 0.9. The samples go through
                // pen.rs exactly as winit's CursorMoved would put them
                // there (physical px), several per frame, so the stroke
                // must carry one pressure per kept point and the blend
                // must be its own topology bit.
                if let Some(editor::ENode { kind: NodeKind::Paint(pp), .. }) =
                    self.egraph.node_mut(pid)
                {
                    pp.output = engine::graph::PaintOutput::Rgb;
                }
                self.brush.mode = engine::graph::BrushMode::Color;
                self.brush.blend = engine::graph::PaintBlend::Multiply;
                self.brush.press = engine::graph::PressureMap { size: 1.0, opacity: 0.0, gamma: 1.0 };
                self.brush.radius = 20.0;
                pen::drain();
                pen::set_pressure(Some(0.3));
                press(raw, c + egui::vec2(-120.0, 120.0), true);
                next = 40;
            }
            40 => {
                let ppp = self.egui_state.egui_ctx().pixels_per_point();
                for (i, x) in [-90.0f32, -60.0, -30.0, 0.0].iter().enumerate() {
                    pen::set_pressure(Some(0.3 + 0.2 * (i + 1) as f32 * 0.75));
                    let p = c + egui::vec2(*x, 120.0);
                    pen::note_cursor((p.x * ppp) as f64, (p.y * ppp) as f64);
                }
                raw.events.push(egui::Event::PointerMoved(c + egui::vec2(0.0, 120.0)));
                next = 41;
            }
            41 => {
                press(raw, c + egui::vec2(0.0, 120.0), false);
                pen::set_pressure(None);
                next = 42;
            }
            42 => next = 43, // settle
            43 => {
                let ss = strokes(self);
                let Some(st) = ss.last() else { return };
                let ok = st.blend == engine::graph::PaintBlend::Multiply
                    && st.pressure.len() == st.points.len()
                    && st.points.len() >= 4
                    && st.pressure[0] < 0.5
                    && st.pressure.windows(2).all(|w| w[1] >= w[0])
                    && st.pressure.last().is_some_and(|p| (*p - 0.9).abs() < 1e-3)
                    && st.press.size == 1.0;
                if !ok {
                    println!(
                        "PAINT TEST FAIL: pen stroke: blend {:?}, {} points, pressure {:?}, press {:?}",
                        st.blend,
                        st.points.len(),
                        st.pressure,
                        st.press
                    );
                    std::process::exit(1);
                }
                let g = self.egraph.to_engine_graph_at(Some(self.eval_tick()));
                let h = g.topology_hash();
                let mut normal = g.clone();
                if let Some(NodeKind::Paint(pp)) = normal.nodes.get_mut(&pid) {
                    for s in pp.strokes.iter_mut() {
                        s.blend = engine::graph::PaintBlend::Normal;
                    }
                }
                if normal.topology_hash() == h {
                    println!("PAINT TEST FAIL: a multiply stroke never reached the shader");
                    std::process::exit(1);
                }
                self.brush.blend = engine::graph::PaintBlend::Normal;
                if let Some(editor::ENode { kind: NodeKind::Paint(pp), .. }) =
                    self.egraph.node_mut(pid)
                {
                    pp.output = engine::graph::PaintOutput::Matte;
                }
                next = 44;
            }
            44 => next = 31, // settle: the matte compiles and draws
            31 => {
                // BAKE: the panel's button puts a GraphBake on the output
                // and rewires the consumer (the Blend at node 4 took the
                // paint's output at stage 0) to read it.
                let Some(bk) = self.bake_paint_node(pid) else {
                    println!("PAINT TEST FAIL: bake_paint_node found no paint node");
                    std::process::exit(1);
                };
                let wired = self.egraph.input_of(bk, 0) == Some(pid)
                    && self.egraph.input_of(4, 0) == Some(bk);
                if !wired {
                    println!(
                        "PAINT TEST FAIL: bake wiring — bake<-{:?}, blend<-{:?} (want {pid}, {bk})",
                        self.egraph.input_of(bk, 0),
                        self.egraph.input_of(4, 0)
                    );
                    std::process::exit(1);
                }
                self.test.paint_bake = Some(bk);
                next = 32;
            }
            32 => {
                // The start is deferred until the rig has rebuilt; wait for
                // it, then it must have either started or adopted a cache.
                if self.auto_bake.is_some() {
                    return;
                }
                let bk = self.test.paint_bake.unwrap_or(0);
                if self.graph_bake.is_none() && !self.loaded.contains_key(&bk) {
                    println!(
                        "PAINT TEST FAIL: the bake did not start ({:?})",
                        self.load_errors.get(&bk)
                    );
                    std::process::exit(1);
                }
                next = 33;
            }
            33 => {
                let n = strokes(self).len();
                println!(
                    "PAINT TEST PASS: a stroke captured and rendered live; [ / ] with \
                     shift and alt drove size/softness/opacity; a stroke authored on \
                     frame 3 lives only there; alt-click set a clone source and the \
                     next stroke took the offset; cloning is topology, painting is \
                     not; held F + motion resized the brush without painting; a dodge \
                     stroke carried mode/stops/aspect/angle and its own topology bit; \
                     matte output recompiled and turned the socket into a mask; the \
                     bake button put a Bake on the output, rewired the consumer and \
                     started it; a pen stroke kept a pressure per point and its multiply \
                     blend is topology ({n} strokes)"
                );
                std::process::exit(0);
            }
            _ => {}
        }
        self.test.paint = Some((next, frames, pid));
    }
}
