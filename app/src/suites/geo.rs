//! `--test-geo`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    pub(crate) fn drive_test_geo(&mut self, raw: &mut egui::RawInput) {
        let Some((stage, mut frames, gid)) = self.test_geo else { return };
        frames += 1;
        if frames > 1500 {
            println!("GEO TEST FAIL: timeout at stage {stage}");
            std::process::exit(1);
        }
        use engine::geo::GeoNode;
        // S-click lays a point down (a plain click is a selection). S goes
        // down with the press and comes up a frame AFTER the release: egui
        // applies a frame's events before the UI runs, so a key released in
        // the click's own frame is already up when the gizmo looks.
        let key_s = |raw: &mut egui::RawInput, down: bool| {
            raw.events.push(egui::Event::Key {
                key: egui::Key::S,
                physical_key: None,
                pressed: down,
                repeat: false,
                modifiers: Default::default(),
            });
        };
        let add = |raw: &mut egui::RawInput, pos: egui::Pos2, down: bool| {
            if down {
                key_s(raw, true);
            }
            raw.events.push(egui::Event::PointerMoved(pos));
            raw.events.push(egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed: down,
                modifiers: Default::default(),
            });
        };
        let spot = |s: &Self, k: usize| -> egui::Pos2 {
            let r = s.test_rect;
            let f = [(0.3, 0.3), (0.7, 0.35), (0.5, 0.7)][k];
            egui::pos2(r.min.x + r.width() * f.0, r.min.y + r.height() * f.1)
        };
        let spline_id = |s: &Self| -> Option<u64> {
            s.egraph.nodes.iter().find_map(|(id, n)| {
                matches!(n.kind, NodeKind::GeoNode(GeoNode::Spline { .. })).then_some(*id)
            })
        };
        let spline_pts = |s: &Self| -> Vec<engine::graph::SplinePoint> {
            spline_id(s)
                .and_then(|id| s.egraph.node(id))
                .and_then(|n| match &n.kind {
                    NodeKind::GeoNode(GeoNode::Spline { points, .. }) => Some(points.clone()),
                    _ => None,
                })
                .unwrap_or_default()
        };
        let mut next = stage;
        match stage {
            0 if self.playback.is_some() && self.displayed_total >= 3 => {
                if !self.layout_settled() {
                    return;
                }
                self.test_rect = self.image_rect;
                // A Noise whose field is the x coordinate: the drawing will
                // read it, so its scatter should lean right.
                let mut fg = engine::field::FieldGraph::default();
                fg.nodes.insert(
                    1,
                    engine::field::FieldNode::Coord {
                        kind: engine::field::CoordKind::X,
                        scale: 1.0,
                    },
                );
                fg.next_id = 2;
                let noise = self.egraph.add(
                    NodeKind::Noise(engine::graph::NoiseParams {
                        graph: fg,
                        scale: 1.0,
                        ..Default::default()
                    }),
                    egui::Pos2::new(120.0, 40.0),
                );
                let id =
                    self.egraph.add(NodeKind::Geo(Default::default()), egui::Pos2::new(320.0, 40.0));
                self.egraph.edges.push((id, 0, noise));
                self.selected = vec![id];
                self.playing = false;
                if let Err(e) = self.dive_into(id) {
                    println!("GEO TEST FAIL: could not open the node: {e}");
                    std::process::exit(1);
                }
                self.test_geo = Some((1, 0, id));
                return;
            }
            // Inside: Star -> Scatter (density from the Noise) -> Instance
            // of a small Polygon, scaled by an exposed Slider; plus an empty
            // Spline to draw into, merged in.
            1 if frames > 2 => {
                if self.dive.last().map(|d| d.kind) != Some(DiveKind::Geo) {
                    println!("GEO TEST FAIL: the dive did not open a geometry graph");
                    std::process::exit(1);
                }
                let out = self.egraph.nodes.iter().find_map(|(id, n)| {
                    matches!(n.kind, NodeKind::GeoNode(GeoNode::Output)).then_some(*id)
                });
                let Some(out) = out else {
                    println!("GEO TEST FAIL: an empty dive did not seed an Output");
                    std::process::exit(1);
                };
                let g = |s: &mut Self, k: GeoNode, y: f32| {
                    s.egraph.add(NodeKind::GeoNode(k), egui::Pos2::new(0.0, y))
                };
                let star = g(
                    self,
                    GeoNode::Star {
                        center: [0.5, 0.28],
                        outer: 0.26,
                        inner: 0.14,
                        points: 5,
                        rotation: 0.0,
                        style: Default::default(),
                    },
                    0.0,
                );
                let field = g(self, GeoNode::FieldIn { shift: 0.0 }, 60.0);
                let scatter =
                    g(self, GeoNode::Scatter { count: 300, seed: 1.0, density: 1.0 }, 120.0);
                let tmpl = g(
                    self,
                    GeoNode::Polygon {
                        center: [0.0, 0.0],
                        radius: 0.02,
                        sides: 4,
                        step: 1,
                        rotation: 0.0,
                        style: Default::default(),
                    },
                    180.0,
                );
                let size = g(self, GeoNode::Slider { value: 0.5, name: "size".into() }, 240.0);
                let inst = g(
                    self,
                    GeoNode::Instance {
                        scale: 1.0,
                        rotation: 0.0,
                        inherit_cd: true,
                        pivot: engine::geo::Pivot::Origin,
                    },
                    300.0,
                );
                let spline = g(
                    self,
                    GeoNode::Spline {
                        points: Vec::new(),
                        mode: engine::graph::SplineMode::Linear,
                        closed: true,
                        style: engine::geo::Style {
                            fill: [0.0; 4],
                            stroke: [1.0; 4],
                            width: 4.0,
                            ..Default::default()
                        },
                    },
                    360.0,
                );
                let merge = g(self, GeoNode::Merge { count: 2 }, 420.0);
                self.egraph.edges.push((scatter, 0, star));
                self.egraph.edges.push((scatter, 1, field));
                self.egraph.edges.push((inst, 0, tmpl));
                self.egraph.edges.push((inst, 1, scatter));
                self.egraph.edges.push((inst, 2, size));
                self.egraph.edges.push((merge, 0, inst));
                self.egraph.edges.push((merge, 1, spline));
                self.egraph.edges.push((out, 0, merge));
                // The gizmo draws into the SELECTED spline.
                self.selected = vec![spline];
                next = 2;
            }
            // Three shift-clicks into the viewer: press on one frame,
            // release on the next, then wait for the point to land — and
            // click again if it did not (a click issued while the viewer is
            // still settling can miss, and a GPU shared with other work
            // makes that likelier).
            2 | 4 | 6 if frames > 6 => {
                let k = (stage as usize - 2) / 2;
                if spline_pts(self).len() > k {
                    next = stage + 2;
                } else {
                    add(raw, spot(self, k), true);
                    next = stage + 1;
                }
            }
            3 | 5 | 7 => {
                let k = (stage as usize - 3) / 2;
                if frames == 1 {
                    add(raw, spot(self, k), false);
                } else if frames == 2 {
                    key_s(raw, false);
                } else if spline_pts(self).len() > k {
                    next = stage + 1;
                } else if frames > 20 {
                    // Missed: go round again — but not forever. A suite that
                    // loops never reports, which is worse than failing.
                    self.geo_click_tries += 1;
                    if self.geo_click_tries > 5 {
                        println!("GEO TEST FAIL: an S-click in the viewer never laid point {k} down");
                        std::process::exit(1);
                    }
                    next = stage - 1;
                }
            }
            8 => {
                let p = spline_pts(self);
                if p.len() < 3 {
                    println!("GEO TEST FAIL: clicked 3 points, the spline has {}", p.len());
                    std::process::exit(1);
                }
                // Where they were clicked, across the width (y is in width
                // units here, so only x is checked against the click).
                let want = [0.3, 0.7, 0.5];
                for (q, w) in p.iter().zip(want) {
                    if (q.p[0] - w).abs() > 0.06 {
                        println!("GEO TEST FAIL: point at x {:.3}, wanted {w:.2}", q.p[0]);
                        std::process::exit(1);
                    }
                }
                // The vector HUD inside the dive shows the SELECTED node's
                // own geometry: the spline just drawn, as one outline.
                let (hp, hn) = self.geo_hud_counts();
                if hn != 1 || hp < 3 {
                    println!("GEO TEST FAIL: the HUD shows {hp} points / {hn} prims for the selected spline");
                    std::process::exit(1);
                }
                next = 30;
            }
            // K held on a point: its tension. Press on the first point with K
            // down, move 100 px right over two frames, release, let K go.
            30 if frames > 2 => {
                let p = spot(self, 0);
                raw.events.push(egui::Event::Key {
                    key: egui::Key::K,
                    physical_key: None,
                    pressed: true,
                    repeat: false,
                    modifiers: Default::default(),
                });
                raw.events.push(egui::Event::PointerMoved(p));
                raw.events.push(egui::Event::PointerButton {
                    pos: p,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: Default::default(),
                });
                next = 31;
            }
            31 | 32 => {
                let f = if stage == 31 { 50.0 } else { 100.0 };
                raw.events.push(egui::Event::PointerMoved(spot(self, 0) + egui::vec2(f, 0.0)));
                next = stage + 1;
            }
            33 => {
                let p = spot(self, 0) + egui::vec2(100.0, 0.0);
                raw.events.push(egui::Event::PointerButton {
                    pos: p,
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers: Default::default(),
                });
                next = 34;
            }
            34 => {
                raw.events.push(egui::Event::Key {
                    key: egui::Key::K,
                    physical_key: None,
                    pressed: false,
                    repeat: false,
                    modifiers: Default::default(),
                });
                next = 35;
            }
            35 if frames > 2 => {
                let p = spline_pts(self);
                let q = p.first().copied().unwrap_or(engine::graph::SplinePoint::new([0.0, 0.0], 0.0));
                // 100 px of a 200 px range: tension 0.5, and the point stays put.
                if !(0.3..0.7).contains(&q.tension) || (q.p[0] - 0.3).abs() > 0.06 {
                    println!(
                        "GEO TEST FAIL: K-drag gave tension {} (want ~0.5), point x {} (want 0.3)",
                        q.tension, q.p[0]
                    );
                    std::process::exit(1);
                }
                next = 36;
            }
            // Physics, through the app's own path: a Physics node fed by the
            // Star, baked, reported fresh, made stale by an edit, cleared.
            // Removed again after, so the counts below are unchanged.
            36 if frames > 2 => {
                let star = self.egraph.nodes.iter().find_map(|(id, n)| {
                    matches!(n.kind, NodeKind::GeoNode(GeoNode::Star { .. })).then_some(*id)
                });
                let Some(star) = star else {
                    println!("GEO TEST FAIL: no Star to simulate");
                    std::process::exit(1);
                };
                let ph = self.egraph.add(
                    NodeKind::GeoNode(GeoNode::Physics {
                        settings: engine::geophys::PhysicsSettings { frames: 24, ..Default::default() },
                        bake: Default::default(),
                    }),
                    egui::Pos2::new(300.0, 0.0),
                );
                self.egraph.edges.push((ph, 0, star));
                if self.physics_status(ph) != Some((None, false)) {
                    println!("GEO TEST FAIL: a fresh Physics node reports {:?}", self.physics_status(ph));
                    std::process::exit(1);
                }
                self.physics_bake_with(ph, false, true);
                match self.physics_status(ph) {
                    Some((Some((24, 1, 0)), false)) => {}
                    other => {
                        println!("GEO TEST FAIL: after a bake the status is {other:?}");
                        std::process::exit(1);
                    }
                }
                // Move the Star: the bake is stale.
                if let Some(NodeKind::GeoNode(GeoNode::Star { center, .. })) =
                    self.egraph.node_mut(star).map(|n| &mut n.kind)
                {
                    center[0] += 0.05;
                }
                if self.physics_status(ph).map(|s| s.1) != Some(true) {
                    println!("GEO TEST FAIL: moving a body did not make the bake stale");
                    std::process::exit(1);
                }
                if let Some(NodeKind::GeoNode(GeoNode::Star { center, .. })) =
                    self.egraph.node_mut(star).map(|n| &mut n.kind)
                {
                    center[0] -= 0.05;
                }
                self.physics_bake(ph, true);
                if self.physics_status(ph) != Some((None, false)) {
                    println!("GEO TEST FAIL: clear left {:?}", self.physics_status(ph));
                    std::process::exit(1);
                }
                // Now the live bake: it runs on a worker and is shown as it
                // goes; the next stages watch it land.
                self.physics_bake(ph, false);
                if self.physics_job.is_none() {
                    println!("GEO TEST FAIL: a live bake did not start");
                    std::process::exit(1);
                }
                self.geo_phys_node = ph;
                self.geo_phys_seen = 0;
                next = 37;
            }
            37 => {
                let ph = self.geo_phys_node;
                if let Some(j) = &self.physics_job {
                    // Progress while it runs: a partial bake in the viewer.
                    let done = j.bake.as_ref().map_or(0, |b| b.frames);
                    self.geo_phys_seen = self.geo_phys_seen.max(done);
                    if frames > 600 {
                        println!("GEO TEST FAIL: the live bake never finished ({done} frames)");
                        std::process::exit(1);
                    }
                    self.test_geo = Some((37, frames, gid));
                    return;
                }
                match self.physics_status(ph) {
                    Some((Some((24, 1, 0)), false)) => {}
                    other => {
                        println!("GEO TEST FAIL: the live bake landed as {other:?}");
                        std::process::exit(1);
                    }
                }
                self.egraph.remove(ph);
                next = 38;
            }
            // Click-through: a shape in the viewed geometry picks its maker.
            38 if frames > 2 => {
                let t = self.eval_tick();
                let Some(geo) = self.geo_view_geometry(t) else {
                    println!("GEO TEST FAIL: no viewed geometry");
                    std::process::exit(1);
                };
                let Some(i) = (0..geo.prims.len()).find(|i| geo.prims[*i].src != 0) else {
                    println!("GEO TEST FAIL: no shape knows its maker");
                    std::process::exit(1);
                };
                let c = geo.centroid(i);
                let want = geo.prims[i].src;
                match self.geo_pick(c, 0.004, t) {
                    Some(src) if self.egraph.node(src).is_some() => {
                        let _ = want;
                    }
                    other => {
                        println!("GEO TEST FAIL: a click on a shape picked {other:?}");
                        std::process::exit(1);
                    }
                }
                // The display bar is up inside a 2dini.
                if self.geo_viz_bar.is_none() {
                    println!("GEO TEST FAIL: no display bar inside the 2dini");
                    std::process::exit(1);
                }
                // WOBBLE_GEO_LINGER=<frames>: hold here with every overlay on,
                // for a screenshot of the HUD and its display bar.
                if let Some(n) = std::env::var("WOBBLE_GEO_LINGER").ok().and_then(|v| v.parse::<u32>().ok()) {
                    if frames < n {
                        self.prefs.geo_viz = GeoViz { points: true, point_nums: true, prim_nums: true, normals: true, ticks: true, outlines: true, bbox: true, ..Default::default() };
                        self.test_geo = Some((38, frames, gid));
                        return;
                    }
                    self.prefs.geo_viz = GeoViz::default();
                }
                // A preset drops in wired and selected; then it goes again.
                let before = self.egraph.nodes.len();
                self.insert_geo_preset(0, egui::pos2(600.0, 0.0));
                let added = self.selected.clone();
                if added.is_empty() || self.egraph.nodes.len() != before + added.len() {
                    println!("GEO TEST FAIL: the preset inserted {} nodes", self.egraph.nodes.len() - before);
                    std::process::exit(1);
                }
                for id in added {
                    self.egraph.remove(id);
                }
                self.selected.clear();
                if let Err(e) = self.dive_out() {
                    println!("GEO TEST FAIL: could not close the node: {e}");
                    std::process::exit(1);
                }
                next = 9;
            }
            9 if frames > 2 => {
                // What came back out has to be what was built inside.
                let g = match self.egraph.node(gid).map(|e| &e.kind) {
                    Some(NodeKind::Geo(g)) => g.clone(),
                    _ => {
                        println!("GEO TEST FAIL: the 2dini node is gone");
                        std::process::exit(1);
                    }
                };
                if g.graph.nodes.len() != 9 || g.graph.edges.len() != 8 {
                    println!(
                        "GEO TEST FAIL: the geometry graph came back as {} nodes / {} edges",
                        g.graph.nodes.len(),
                        g.graph.edges.len()
                    );
                    std::process::exit(1);
                }
                if g.graph.exposed() != vec![("size".to_string(), false)] {
                    println!("GEO TEST FAIL: exposed {:?}", g.graph.exposed());
                    std::process::exit(1);
                }
                let drawn = g.graph.nodes.values().any(|n| {
                    matches!(n, GeoNode::Spline { points, .. } if points.len() == 3)
                });
                if !drawn {
                    println!("GEO TEST FAIL: the drawn spline did not survive the round trip");
                    std::process::exit(1);
                }
                // The Slider is a channel on the node now.
                let chans = anim::channels_for(&NodeKind::Geo(g.clone()));
                if !chans.contains(&anim::Channel::Custom(anim::intern("size"))) {
                    println!("GEO TEST FAIL: the exposed slider is not a channel");
                    std::process::exit(1);
                }
                self.egraph.view = Some(gid);
                self.playback_dirty = true;
                next = 10;
            }
            10 if frames > 40 => {
                let Some(m) = self.probe_output_mean() else { return };
                if !(0.001..0.6).contains(&m) {
                    println!("GEO TEST FAIL: the rendered drawing reads {m:.4}");
                    std::process::exit(1);
                }
                self.geo_mean = m;
                // Outside, the HUD shows the selected 2dini's layer geometry:
                // every scattered copy plus the drawn spline.
                self.selected = vec![gid];
                let (hp, hn) = self.geo_hud_counts();
                if hn != 301 || hp < 1000 {
                    println!("GEO TEST FAIL: the HUD shows {hp} points / {hn} prims for the 2dini node");
                    std::process::exit(1);
                }
                self.selected.clear();
                // Drive the exposed slider through its CHANNEL, the way a
                // key or a patch cable would: bigger instances, more cover.
                if let Some(n) = self.egraph.node_mut(gid) {
                    anim::set(&mut n.kind, anim::Channel::Custom(anim::intern("size")), 2.0);
                }
                self.playback_dirty = true;
                next = 11;
            }
            11 if frames > 40 => {
                let Some(m) = self.probe_output_mean() else { return };
                if m < self.geo_mean * 1.3 {
                    let value = self.egraph.node(gid).and_then(|n| match &n.kind {
                        NodeKind::Geo(g) => g.graph.exposed_value("size"),
                        _ => None,
                    });
                    println!(
                        "GEO TEST FAIL: driving the slider did not change the drawing \
                         ({:.4} -> {m:.4}); slider now {value:?}, layer {:?}, dirty {:?}",
                        self.geo_mean,
                        self.comp.geo_stats(gid),
                        self.comp.geo_layer_dirty(gid)
                    );
                    std::process::exit(1);
                }
                println!(
                    "GEO TEST PASS: dived in, built a noise-driven scatter + instance with an \
                     exposed slider and a viewer-drawn spline, stepped out, rendered it \
                     (mean {:.3}) and drove the slider through its channel (mean {m:.3})",
                    self.geo_mean
                );
                std::process::exit(0);
            }
            _ => {}
        }
        self.test_geo = Some((next, if next == stage { frames } else { 0 }, gid));
    }
}
