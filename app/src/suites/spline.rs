//! `--test-spline`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    /// --test-spline driver: a fresh Spline starts empty; click three
    /// points into the viewer, check they land where clicked, then key one
    /// point and confirm the EVALUATED outline follows the playhead while
    /// the authored point stays put.
    pub(crate) fn drive_test_spline(&mut self, raw: &mut egui::RawInput) {
        let Some((stage, mut frames, sid, stash)) = self.test.spline else { return };
        frames += 1;
        if frames > 1200 {
            println!("SPLINE TEST FAIL: timeout at stage {stage}");
            std::process::exit(1);
        }
        let click_mod = |raw: &mut egui::RawInput,
                         pos: egui::Pos2,
                         down: bool,
                         modifiers: egui::Modifiers| {
            // `InputState::modifiers` reads the RawInput field, not the one
            // riding the event — a gizmo asking `i.modifiers.shift` sees
            // nothing unless this is set too.
            raw.events.insert(0, egui::Event::ModifiersChanged(modifiers));
            raw.events.push(egui::Event::PointerMoved(pos));
            raw.events.push(egui::Event::PointerButton {
                pos,
                button: egui::PointerButton::Primary,
                pressed: down,
                modifiers,
            });
        };
        let click = |raw: &mut egui::RawInput, pos: egui::Pos2, down: bool| {
            click_mod(raw, pos, down, Default::default())
        };
        // A letter going down or up. It must not come up in the SAME frame
        // as the click it modifies: egui applies a frame's events before the
        // UI runs, so the click would see the key already released.
        let key = |raw: &mut egui::RawInput, key: egui::Key, down: bool| {
            raw.events.push(egui::Event::Key {
                key,
                physical_key: None,
                pressed: down,
                repeat: false,
                modifiers: Default::default(),
            });
        };
        // A click with a letter held: the key goes down with the press and
        // stays down — the stage after the gesture lets it go.
        let key_click = |raw: &mut egui::RawInput, pos: egui::Pos2, down: bool, k: egui::Key| {
            if down {
                key(raw, k, true);
            }
            click_mod(raw, pos, down, Default::default());
        };
        // Points go down on S-click; a plain click is a selection.
        let add = |raw: &mut egui::RawInput, pos: egui::Pos2, down: bool| {
            key_click(raw, pos, down, egui::Key::S)
        };
        // Three spots inside the image, well clear of each other.
        let spot = |s: &Self, k: usize| -> egui::Pos2 {
            // Anchored at capture time: a press and its release must land on
            // the same spot even while panels are still animating.
            let r = s.test.rect;
            let f = [(0.3, 0.3), (0.7, 0.35), (0.5, 0.7)][k];
            egui::pos2(r.min.x + r.width() * f.0, r.min.y + r.height() * f.1)
        };
        let pts = |s: &Self| -> Vec<engine::graph::SplinePoint> {
            match s.egraph.node(sid).map(|n| &n.kind) {
                Some(NodeKind::Spline(sp)) => sp.points.clone(),
                _ => Vec::new(),
            }
        };
        // Screen box the group tools put round every point (all three are
        // picked by then): min corner, max corner, centre.
        let bounds = |s: &Self| -> Option<(egui::Pos2, egui::Pos2, egui::Pos2)> {
            let n = s.egraph.node(sid)?;
            let NodeKind::Spline(sp) = n.kind_at(s.eval_tick()) else { return None };
            let r = s.test.rect;
            let (mut lo, mut hi) =
                (egui::pos2(f32::MAX, f32::MAX), egui::pos2(f32::MIN, f32::MIN));
            for q in &sp.points {
                let p = egui::pos2(
                    r.min.x + q.p[0] * r.width(),
                    r.min.y + q.p[1] * r.height(),
                );
                lo = lo.min(p);
                hi = hi.max(p);
            }
            Some((lo, hi, lo + (hi - lo) * 0.5))
        };
        // The group box as the gizmo last drew it (TL TR BR BL, screen) —
        // it turns and bends with the pick, so it is not the points' bounds.
        let quad_scr = |s: &Self| -> Option<[egui::Pos2; 4]> {
            let (n, _, q, _) = s.spline.box_.as_ref()?;
            if *n != sid {
                return None;
            }
            let [w, h] = s
                .node_res_f32(sid)
                .or_else(|| s.comp.output_size().map(|(w, h)| (w as f32, h as f32)))
                .map(|(w, h)| [w, h])?;
            let r = s.test.rect;
            Some(q.map(|v| {
                egui::pos2(r.min.x + v.x / w * r.width(), r.min.y + v.y / h * r.height())
            }))
        };
        // A corner's scale handle: the box corner, pushed 18px clear of the
        // points that define it (away from the centre).
        let corner_at = |s: &Self, i: usize| -> Option<egui::Pos2> {
            let q = quad_scr(s)?;
            let c = egui::pos2(
                (q[0].x + q[1].x + q[2].x + q[3].x) * 0.25,
                (q[0].y + q[1].y + q[2].y + q[3].y) * 0.25,
            );
            Some(q[i] + (q[i] - c).normalized() * 18.0)
        };
        let corner_handle = |s: &Self| corner_at(s, 2);
        let xs = |s: &Self| -> [f32; 2] {
            match s.egraph.node(sid).map(|n| n.kind_at(s.eval_tick())) {
                Some(NodeKind::Spline(sp)) => {
                    let mut lo = f32::MAX;
                    let mut hi = f32::MIN;
                    for q in &sp.points {
                        lo = lo.min(q.p[0]);
                        hi = hi.max(q.p[0]);
                    }
                    [lo, hi]
                }
                _ => [f32::NAN; 2],
            }
        };
        // A corner of the viewer well clear of every point: the marquee has
        // to start on empty canvas or a point handle eats the drag.
        let band = |s: &Self, f: f32| -> egui::Pos2 {
            let r = s.test.rect;
            egui::pos2(r.min.x + r.width() * f, r.min.y + r.height() * f)
        };
        // Where a control point actually is now (the group scale moved
        // them all), in screen points.
        let pt_scr = |s: &Self, i: usize| -> Option<egui::Pos2> {
            let NodeKind::Spline(sp) = s.egraph.node(sid)?.kind_at(s.eval_tick()) else {
                return None;
            };
            let q = sp.points.get(i)?.p;
            let r = s.test.rect;
            Some(egui::pos2(r.min.x + q[0] * r.width(), r.min.y + q[1] * r.height()))
        };
        // Where a control point is DRAWN once the shape transform is on it:
        // authored -> transform -> screen, the same chain the gizmo uses.
        // An unwired Spline has no resolution of its own (its one input is a
        // matte, which resolution does not flow through), so these fall back
        // to the canvas — the SAME fallback the gizmo under test uses. Without
        // it they return None forever, and stage 35's early return skips the
        // frame write, so the suite can neither pass nor time out.
        let canvas = |s: &Self| -> Option<[f32; 2]> {
            s.node_res_f32(sid)
                .or_else(|| s.comp.output_size().map(|(w, h)| (w as f32, h as f32)))
                .map(|(w, h)| [w, h])
        };
        let pt_xf_scr = |s: &Self, i: usize| -> Option<egui::Pos2> {
            let NodeKind::Spline(sp) = s.egraph.node(sid)?.kind_at(s.eval_tick()) else {
                return None;
            };
            let sz = canvas(s)?;
            let q = sp.xf.point(sp.points.get(i)?.p, sz);
            let r = s.test.rect;
            Some(egui::pos2(r.min.x + q[0] * r.width(), r.min.y + q[1] * r.height()))
        };
        // ...and where it is in normalized space after the transform.
        let pt_xf = |s: &Self, i: usize| -> Option<[f32; 2]> {
            let NodeKind::Spline(sp) = s.egraph.node(sid)?.kind_at(s.eval_tick()) else {
                return None;
            };
            let sz = canvas(s)?;
            Some(sp.xf.point(sp.points.get(i)?.p, sz))
        };
        // The rotate knob: 34px off the middle of the box's top edge.
        let rot_knob = |s: &Self| -> Option<egui::Pos2> {
            let q = quad_scr(s)?;
            let c = egui::pos2(
                (q[0].x + q[1].x + q[2].x + q[3].x) * 0.25,
                (q[0].y + q[1].y + q[2].y + q[3].y) * 0.25,
            );
            let top = q[0] + (q[1] - q[0]) * 0.5;
            let up = top - c;
            let dir = if up.length() > 1e-3 { up.normalized() } else { egui::vec2(0.0, -1.0) };
            Some(top + dir * 34.0)
        };
        // Distance of each surviving point from the box centre.
        let radii = |s: &Self| -> [f32; 2] {
            match (bounds(s), s.egraph.node(sid).map(|n| n.kind_at(s.eval_tick()))) {
                (Some((_, _, c)), Some(NodeKind::Spline(sp))) => {
                    let r = s.test.rect;
                    let d = |i: usize| {
                        sp.points
                            .get(i)
                            .map(|q| {
                                (egui::pos2(
                                    r.min.x + q.p[0] * r.width(),
                                    r.min.y + q.p[1] * r.height(),
                                ) - c)
                                    .length()
                            })
                            .unwrap_or(f32::NAN)
                    };
                    [d(0), d(1)]
                }
                _ => [f32::NAN; 2],
            }
        };
        let mut next_stash = stash;
        let mut next = stage;
        match stage {
            0 if frames >= 6 => {
                if !self.layout_settled() {
                    return;
                }
                self.test.rect = self.image_rect;
                let id = self.egraph.add(
                    NodeKind::Spline(Default::default()),
                    egui::Pos2::new(240.0, 0.0),
                );
                self.selected = vec![id];
                self.playing = false;
                self.test.spline = Some((1, frames, id, stash));
                return;
            }
            1 => {
                if !pts(self).is_empty() {
                    println!("SPLINE TEST FAIL: fresh spline came with a shape");
                    std::process::exit(1);
                }
                add(raw, spot(self, 0), true);
                next = 2;
            }
            2 | 4 | 6 => {
                let k = (stage as usize - 2) / 2;
                add(raw, spot(self, k), false);
                next = stage + 1;
            }
            3 | 5 => {
                let k = (stage as usize - 1) / 2;
                add(raw, spot(self, k), true);
                next = stage + 1;
            }
            7 => {
                key(raw, egui::Key::S, false);
                let p = pts(self);
                if p.len() < 3 {
                    // The click lands during the egui pass; give it a frame.
                    self.test.spline = Some((7, frames, sid, stash));
                    return;
                }
                if p.len() != 3 {
                    println!("SPLINE TEST FAIL: clicked 3 points, got {}", p.len());
                    std::process::exit(1);
                }
                // Points should sit where they were clicked (normalized).
                let want = [(0.3, 0.3), (0.7, 0.35), (0.5, 0.7)];
                for (q, w) in p.iter().zip(want) {
                    if (q.p[0] - w.0).abs() > 0.02 || (q.p[1] - w.1).abs() > 0.02 {
                        println!("SPLINE TEST FAIL: point {:?} wanted {w:?}", q.p);
                        std::process::exit(1);
                    }
                }
                // Per-point feather is independent.
                if let Some(editor::ENode { kind: NodeKind::Spline(sp), .. }) =
                    self.egraph.node_mut(sid)
                {
                    // Three points leave a drawn shape OPEN (pen tool);
                    // closing it by a click on point 0 is --test-roto's.
                    if !sp.open {
                        println!("SPLINE TEST FAIL: three points closed the shape by themselves");
                        std::process::exit(1);
                    }
                    sp.open = false;
                    sp.points[0].feather = 4.0;
                    sp.points[1].feather = 80.0;
                    sp.mode = engine::graph::SplineMode::Bezier;
                    sp.points[0].h_out = [0.05, -0.05];
                }
                // Key point 0's X at frames 0 and 10.
                if let Some(n) = self.egraph.node_mut(sid) {
                    let c = n.anim.entry(anim::Channel::SplinePtX(0)).or_default();
                    c.set_key(0, 0.3);
                    c.set_key(10, 0.9);
                }
                self.seek_to(5);
                next = 8;
            }
            8 => {
                if !self.force_display && self.eval_tick() == 5 {
                    next = 9;
                }
            }
            9 => {
                let n = self.egraph.node(sid).unwrap();
                let evaluated = match n.kind_at(self.eval_tick()) {
                    NodeKind::Spline(sp) => sp.points[0].p[0],
                    _ => f32::NAN,
                };
                let authored = n.kind_at(u64::MAX); // clamps to the last key
                let _ = authored;
                let stored = match &n.kind {
                    NodeKind::Spline(sp) => sp.points[0].p[0],
                    _ => f32::NAN,
                };
                let feathers = match &n.kind {
                    NodeKind::Spline(sp) => {
                        (sp.points[0].feather, sp.points[1].feather)
                    }
                    _ => (0.0, 0.0),
                };
                // Halfway between 0.3 and 0.9 at frame 5.
                let ok = (evaluated - 0.6).abs() < 0.02
                    && (stored - 0.3).abs() < 0.02
                    && feathers == (4.0, 80.0);
                if !ok {
                    println!(
                        "SPLINE TEST FAIL: eval {evaluated:.3} stored {stored:.3} feathers {feathers:?}"
                    );
                    std::process::exit(1);
                }
                next = 10;
            }
            // Drag point 1 while the shape is animated (point 0 carries keys
            // from stage 7). The whole shape must land on frame 5, not just
            // the point under the cursor.
            10 => {
                click(raw, spot(self, 1), true);
                next = 11;
            }
            11 | 12 => {
                let d = if stage == 11 { 12.0 } else { 24.0 };
                raw.events
                    .push(egui::Event::PointerMoved(spot(self, 1) + egui::vec2(d, 0.0)));
                next = stage + 1;
            }
            13 => {
                click(raw, spot(self, 1) + egui::vec2(24.0, 0.0), false);
                next = 14;
            }
            14 => next = 15, // settle
            15 => {
                let n = self.egraph.node(sid).unwrap();
                let at5 = |ch: anim::Channel| {
                    n.anim
                        .get(&ch)
                        .and_then(|c| c.keys.iter().find(|k| k.frame == 5.0))
                        .map(|k| k.value)
                };
                let shape = anim::spline_shape(3);
                let missing: Vec<_> =
                    shape.iter().filter(|ch| at5(**ch).is_none()).collect();
                // The dragged point moved; the others were keyed where they
                // already were, so the shape holds instead of tearing.
                let dragged = at5(anim::Channel::SplinePtX(1)).unwrap_or(f32::NAN);
                let held = at5(anim::Channel::SplinePtX(0)).unwrap_or(f32::NAN);
                let keys0 = n
                    .anim
                    .get(&anim::Channel::SplinePtX(0))
                    .map(|c| c.keys.len())
                    .unwrap_or(0);
                if !(missing.is_empty()
                    && dragged > 0.705
                    && (held - 0.6).abs() < 0.02
                    && keys0 == 3)
                {
                    println!(
                        "SPLINE TEST FAIL: shape key missing {missing:?} \
                         dragged {dragged:.3} held {held:.3} keys0 {keys0}"
                    );
                    std::process::exit(1);
                }
                let _ = shape;
                next = 16;
            }
            // Marquee: a plain background drag from clear of the shape to
            // past it should band every point.
            16 => {
                click(raw, band(self, 0.04), true);
                next = 17;
            }
            17 | 18 => {
                let d = if stage == 17 { 0.5 } else { 0.96 };
                let a = band(self, 0.04);
                let b = band(self, 0.96);
                raw.events.push(egui::Event::PointerMoved(a + (b - a) * (d / 0.96)));
                next = stage + 1;
            }
            19 => {
                click(raw, band(self, 0.96), false);
                next = 20;
            }
            20 => next = 21, // settle
            21 => {
                if self.spline.sel.len() != 3 {
                    println!(
                        "SPLINE TEST FAIL: marquee picked {:?}, wanted all 3",
                        self.spline.sel
                    );
                    std::process::exit(1);
                }
                // WOBBLE_TEST_HOLD parks here, every point picked, for a
                // look at the group box and the hotkey strip.
                if std::env::var("WOBBLE_TEST_HOLD").is_ok() {
                    return;
                }
                // Grab the bottom-right scale handle of the group box.
                let Some(h) = corner_handle(self) else {
                    println!("SPLINE TEST FAIL: no group box to scale");
                    std::process::exit(1);
                };
                next_stash = xs(self);
                click(raw, h, true);
                next = 22;
            }
            22 | 23 => {
                let Some(h) = corner_handle(self) else { return };
                let d = if stage == 22 { 20.0 } else { 40.0 };
                raw.events.push(egui::Event::PointerMoved(h + egui::vec2(d, d)));
                next = stage + 1;
            }
            24 => {
                let Some(h) = corner_handle(self) else { return };
                click(raw, h + egui::vec2(40.0, 40.0), false);
                next = 25;
            }
            25 => next = 26, // settle
            26 => {
                // Scaling from the bottom-right handle pins the opposite
                // corner: the leftmost point holds, the rightmost travels.
                let [lo0, hi0] = stash;
                let [lo1, hi1] = xs(self);
                if (lo1 - lo0).abs() > 0.005 || hi1 <= hi0 + 0.002 {
                    println!(
                        "SPLINE TEST FAIL: scale handle left x {lo0:.4}->{lo1:.4} \
                         (anchor should hold) and {hi0:.4}->{hi1:.4} (should grow)"
                    );
                    std::process::exit(1);
                }
                // D-click the last point: that is the remove gesture.
                let Some(p) = pt_scr(self, 2) else { return };
                key_click(raw, p, true, egui::Key::D);
                next = 27;
            }
            27 => {
                let Some(p) = pt_scr(self, 2) else { return };
                key_click(raw, p, false, egui::Key::D);
                next = 28;
            }
            28 => {
                key(raw, egui::Key::D, false);
                next = 29; // settle
            }
            29 => {
                let left = match self.egraph.node(sid).map(|n| &n.kind) {
                    Some(NodeKind::Spline(sp)) => sp.points.len(),
                    _ => 0,
                };
                if left != 2 {
                    println!("SPLINE TEST FAIL: D-click left {left} points, wanted 2");
                    std::process::exit(1);
                }
                // The pick slid with the removal, so the two survivors are
                // still picked and the rotate knob is still up.
                if self.spline.sel.len() != 2 {
                    println!(
                        "SPLINE TEST FAIL: removal left the pick as {:?}",
                        self.spline.sel
                    );
                    std::process::exit(1);
                }
                next_stash = radii(self);
                let Some(k) = rot_knob(self) else { return };
                click(raw, k, true);
                next = 30;
            }
            30 | 31 => {
                let Some(k) = rot_knob(self) else { return };
                let d = if stage == 30 { 25.0 } else { 50.0 };
                raw.events.push(egui::Event::PointerMoved(k + egui::vec2(d, 0.0)));
                next = stage + 1;
            }
            32 => {
                let Some(k) = rot_knob(self) else { return };
                click(raw, k + egui::vec2(50.0, 0.0), false);
                next = 33;
            }
            33 => next = 34, // settle
            34 => {
                // A rotation keeps every point the same distance from the
                // box centre; it just moves them round it.
                let [b0, b1] = stash;
                let [a0, a1] = radii(self);
                let moved = self
                    .egraph
                    .node(sid)
                    .map(|n| match n.kind_at(self.eval_tick()) {
                        NodeKind::Spline(sp) => sp.points[0].p[1],
                        _ => f32::NAN,
                    })
                    .unwrap_or(f32::NAN);
                if (a0 - b0).abs() > b0.max(1.0) * 0.06
                    || (a1 - b1).abs() > b1.max(1.0) * 0.06
                {
                    println!(
                        "SPLINE TEST FAIL: rotate changed the radii {b0:.1}->{a0:.1}, \
                         {b1:.1}->{a1:.1}"
                    );
                    std::process::exit(1);
                }
                let _ = moved;
                next = 200;
            }
            // D-drag on the group box's centre circle moves the PIVOT and
            // leaves the points alone (the Transform gizmo's gesture).
            200 => {
                let Some((_, _, c)) = bounds(self) else { return };
                self.test.spline_press = Some(c);
                next_stash = [c.x, c.y];
                key_click(raw, c, true, egui::Key::D);
                next = 201;
            }
            201 | 202 => {
                let Some(c) = self.test.spline_press else { return };
                let f = if stage == 201 { 0.5 } else { 1.0 };
                raw.events.push(egui::Event::PointerMoved(c + egui::vec2(30.0, 20.0) * f));
                next = stage + 1;
            }
            203 => {
                let Some(c) = self.test.spline_press else { return };
                key_click(raw, c + egui::vec2(30.0, 20.0), false, egui::Key::D);
                next = 204;
            }
            204 => {
                key(raw, egui::Key::D, false);
                next = 205;
            }
            205 => {
                let Some((_, _, c)) = bounds(self) else { return };
                let r = self.test.rect;
                let pv = self.spline.pivot.as_ref().map(|(_, _, q)| {
                    egui::pos2(r.min.x + q[0] * r.width(), r.min.y + q[1] * r.height())
                });
                let want = egui::pos2(stash[0], stash[1]) + egui::vec2(30.0, 20.0);
                if (c - egui::pos2(stash[0], stash[1])).length() > 0.5
                    || pv.is_none_or(|p| (p - want).length() > 1.5)
                {
                    println!(
                        "SPLINE TEST FAIL: D-drag on the centre moved the box centre \
                         {:?}->{c:?} (should hold) and put the pivot at {pv:?} (wanted {want:?})",
                        stash
                    );
                    std::process::exit(1);
                }
                // Now the knob turns the pick about THAT point.
                let (Some(a), Some(b), Some(p)) = (pt_scr(self, 0), pt_scr(self, 1), pv) else {
                    return;
                };
                self.test.spline_aux = vec![p];
                next_stash = [(a - p).length(), (b - p).length()];
                let Some(k) = rot_knob(self) else { return };
                self.test.spline_press = Some(k);
                click(raw, k, true);
                next = 206;
            }
            206 | 207 => {
                let Some(k) = self.test.spline_press else { return };
                let d = if stage == 206 { 20.0 } else { 40.0 };
                raw.events.push(egui::Event::PointerMoved(k + egui::vec2(d, 0.0)));
                next = stage + 1;
            }
            208 => {
                let Some(k) = self.test.spline_press else { return };
                click(raw, k + egui::vec2(40.0, 0.0), false);
                next = 209;
            }
            209 => next = 210, // settle
            210 => {
                let p = self.test.spline_aux[0];
                let (Some(a), Some(b)) = (pt_scr(self, 0), pt_scr(self, 1)) else { return };
                let [d0, d1] = stash;
                let (e0, e1) = ((a - p).length(), (b - p).length());
                if (e0 - d0).abs() > d0.max(1.0) * 0.03 || (e1 - d1).abs() > d1.max(1.0) * 0.03 {
                    println!(
                        "SPLINE TEST FAIL: rotate after a pivot move changed the distances \
                         to the pivot {d0:.1}->{e0:.1}, {d1:.1}->{e1:.1}"
                    );
                    std::process::exit(1);
                }
                next = 211;
            }
            // Ctrl-drag on a CORNER distorts: the point on that corner
            // follows the pointer 1:1, the point on the opposite corner holds.
            211 => {
                let (Some((lo, hi, c)), Some(a), Some(b)) =
                    (bounds(self), pt_scr(self, 0), pt_scr(self, 1))
                else {
                    return;
                };
                if (hi - lo).min_elem() < 10.0 {
                    println!("SPLINE TEST FAIL: group box too thin to distort ({lo:?}..{hi:?})");
                    std::process::exit(1);
                }
                let _ = c;
                let Some(quad) = quad_scr(self) else { return };
                let ci = (0..4)
                    .min_by(|i, j| (quad[*i] - a).length().total_cmp(&(quad[*j] - a).length()))
                    .unwrap_or(0);
                let Some(h) = corner_at(self, ci) else { return };
                self.test.spline_press = Some(h);
                // Both points, then the box corner and the one opposite it.
                self.test.spline_aux = vec![a, b, quad[ci], quad[(ci + 2) % 4]];
                click_mod(raw, h, true, egui::Modifiers::CTRL);
                next = 212;
            }
            212 | 213 => {
                let Some(h) = self.test.spline_press else { return };
                let f = if stage == 212 { 0.5 } else { 1.0 };
                raw.events.insert(0, egui::Event::ModifiersChanged(egui::Modifiers::CTRL));
                raw.events.push(egui::Event::PointerMoved(h + egui::vec2(-30.0, 25.0) * f));
                next = stage + 1;
            }
            214 => {
                let Some(h) = self.test.spline_press else { return };
                click_mod(raw, h + egui::vec2(-30.0, 25.0), false, egui::Modifiers::CTRL);
                next = 215;
            }
            215 => next = 216, // settle
            216 => {
                let (Some(a), Some(b)) = (pt_scr(self, 0), pt_scr(self, 1)) else { return };
                let (a0, b0) = (self.test.spline_aux[0], self.test.spline_aux[1]);
                let want = a0 + egui::vec2(-30.0, 25.0);
                if (a - want).length() > 1.5 || (b - b0).length() > 0.5 {
                    println!(
                        "SPLINE TEST FAIL: ctrl-drag on a corner put its point at {a:?} \
                         (wanted {want:?}) and moved the opposite one {b0:?}->{b:?}"
                    );
                    std::process::exit(1);
                }
                if self.spline.distort.is_some() {
                    println!("SPLINE TEST FAIL: the distort outlived its drag");
                    std::process::exit(1);
                }
                // The BOX bends with it: the dragged corner went with the
                // pointer and the opposite one held — it did not snap back
                // to the points' bounds (the user's complaint).
                let (k0, o0) = (self.test.spline_aux[2], self.test.spline_aux[3]);
                let want_k = k0 + egui::vec2(-30.0, 25.0);
                let quad = quad_scr(self).unwrap_or([egui::Pos2::ZERO; 4]);
                let ki = (0..4)
                    .min_by(|i, j| (quad[*i] - want_k).length().total_cmp(&(quad[*j] - want_k).length()))
                    .unwrap_or(0);
                if (quad[ki] - want_k).length() > 2.0 || (quad[(ki + 2) % 4] - o0).length() > 1.0 {
                    println!(
                        "SPLINE TEST FAIL: after the distort the box is {quad:?}; wanted a \
                         corner at {want_k:?} and the opposite one held at {o0:?}"
                    );
                    std::process::exit(1);
                }
                next = 240;
            }
            // E / R / T held at the press: scale, rotate, translate the
            // pick about wherever the press landed. 240 T, 250 E, 260 R.
            240 | 250 | 260 => {
                let (Some(a), Some(b)) = (pt_scr(self, 0), pt_scr(self, 1)) else { return };
                let (k, o) = match stage {
                    240 => (egui::Key::T, band(self, 0.1)),
                    250 => (egui::Key::E, band(self, 0.12)),
                    _ => (egui::Key::R, band(self, 0.12)),
                };
                self.test.spline_press = Some(o);
                self.test.spline_aux = vec![a, b];
                key_click(raw, o, true, k);
                next = stage + 1;
            }
            241 | 242 | 251 | 252 | 261 | 262 => {
                let Some(o) = self.test.spline_press else { return };
                let f = if stage % 10 == 1 { 0.5 } else { 1.0 };
                let d = match stage {
                    241 | 242 => egui::vec2(40.0, -20.0) * f,
                    251 | 252 => egui::vec2(104.0, 0.0) * f,
                    // R: out to the right (sets the reference), then round
                    // to straight below the pivot — a quarter turn.
                    261 => egui::vec2(40.0, 0.0),
                    _ => egui::vec2(0.0, 60.0),
                };
                raw.events.push(egui::Event::PointerMoved(o + d));
                next = stage + 1;
            }
            243 | 253 | 263 => {
                let Some(o) = self.test.spline_press else { return };
                let d = match stage / 10 {
                    24 => egui::vec2(40.0, -20.0),
                    25 => egui::vec2(104.0, 0.0),
                    _ => egui::vec2(0.0, 60.0),
                };
                let k = [egui::Key::T, egui::Key::E, egui::Key::R][(stage / 10 - 24) as usize];
                key_click(raw, o + d, false, k);
                next = stage + 1;
            }
            244 | 254 | 264 => {
                let k = [egui::Key::T, egui::Key::E, egui::Key::R][(stage / 10 - 24) as usize];
                key(raw, k, false);
                next = stage + 1;
            }
            245 | 255 | 265 => {
                let (Some(a), Some(b), Some(o)) =
                    (pt_scr(self, 0), pt_scr(self, 1), self.test.spline_press)
                else {
                    return;
                };
                let (a0, b0) = (self.test.spline_aux[0], self.test.spline_aux[1]);
                let want = |q: egui::Pos2| -> egui::Pos2 {
                    let v = q - o;
                    match stage {
                        245 => q + egui::vec2(40.0, -20.0),
                        255 => o + v * (104.0f32 / 150.0).exp(),
                        // The cursor went from right of the pivot to below
                        // it: a quarter turn.
                        _ => o + egui::vec2(-v.y, v.x),
                    }
                };
                let (wa, wb) = (want(a0), want(b0));
                if (a - wa).length() > 2.0 || (b - wb).length() > 2.0 || self.spline.temp.is_some() {
                    println!(
                        "SPLINE TEST FAIL: held-key edit (stage {stage}) put the pick at \
                         {a:?} {b:?}, wanted {wa:?} {wb:?} (live after release: {})",
                        self.spline.temp.is_some()
                    );
                    std::process::exit(1);
                }
                if self.spline.sel.len() != 2 {
                    println!("SPLINE TEST FAIL: held-key edit changed the pick to {:?}", self.spline.sel);
                    std::process::exit(1);
                }
                next = match stage {
                    245 => 250,
                    255 => 260,
                    _ => 268,
                };
            }
            // O / I held on a POINT: that point's feather grows outward then
            // inward — the band's centre (bias) moving by half of what the
            // width gains, so the hard side stays where it was.
            // The held-key edits above flung the pick off screen: bring the
            // two points back where a press can reach them.
            268 => {
                if let Some(n) = self.egraph.node_mut(sid) {
                    anim::spline_channels_cleared(&mut n.anim);
                    if let NodeKind::Spline(sp) = &mut n.kind {
                        sp.points[0].p = [0.4, 0.45];
                        sp.points[1].p = [0.6, 0.55];
                    }
                }
                next = 269;
            }
            269 => next = 270, // settle
            270 | 280 => {
                let (Some(NodeKind::Spline(sp)), Some(p)) =
                    (self.egraph.node(sid).map(|n| n.kind_at(self.eval_tick())), pt_scr(self, 0))
                else {
                    return;
                };
                next_stash = [sp.points[0].feather, sp.points[0].bias];
                self.test.spline_press = Some(p);
                key_click(raw, p, true, if stage == 270 { egui::Key::O } else { egui::Key::I });
                next = stage + 1;
            }
            271 | 272 | 281 | 282 => {
                let Some(o) = self.test.spline_press else { return };
                let f = if stage % 10 == 1 { 0.5 } else { 1.0 };
                raw.events.push(egui::Event::PointerMoved(o + egui::vec2(60.0 * f, 0.0)));
                next = stage + 1;
            }
            273 | 283 => {
                let Some(o) = self.test.spline_press else { return };
                let k = if stage == 273 { egui::Key::O } else { egui::Key::I };
                key_click(raw, o + egui::vec2(60.0, 0.0), false, k);
                next = stage + 1;
            }
            274 | 284 => {
                key(raw, if stage == 274 { egui::Key::O } else { egui::Key::I }, false);
                next = stage + 1;
            }
            275 | 285 => {
                let Some(NodeKind::Spline(sp)) = self.egraph.node(sid).map(|n| n.kind_at(self.eval_tick())) else {
                    return;
                };
                let [f0, b0] = stash;
                let q = sp.points[0];
                let s = if stage == 275 { 0.5 } else { -0.5 };
                let moved = (q.p[0] - 0.0).is_finite() && pt_scr(self, 0) == self.test.spline_press;
                if q.feather <= f0 + 1.0 || (q.bias - (b0 + s * (q.feather - f0))).abs() > 0.01 || !moved {
                    println!(
                        "SPLINE TEST FAIL: O/I drag on a point (stage {stage}): feather {f0}->{}, \
                         bias {b0}->{} (wanted {}), point still under the press: {moved}; \
                         pick {:?}, points {:?}, press {:?}",
                        q.feather,
                        q.bias,
                        b0 + s * (q.feather - f0),
                        self.spline.sel,
                        sp.points.iter().map(|q| q.p).collect::<Vec<_>>(),
                        self.test.spline_press
                    );
                    std::process::exit(1);
                }
                next = if stage == 275 { 280 } else { 230 };
            }
            230 => {
                // A shape with no format inherits (the canvas); one with a
                // format declares it, exactly like a Grid.
                let inherits = self.node_res_f32(sid).is_none();
                if let Some(NodeKind::Spline(sp)) = self.egraph.node_mut(sid).map(|n| &mut n.kind) {
                    sp.format = Some((640, 360));
                }
                let declared = self.node_res_f32(sid);
                if let Some(NodeKind::Spline(sp)) = self.egraph.node_mut(sid).map(|n| &mut n.kind) {
                    sp.format = None;
                }
                if !inherits || declared != Some((640.0, 360.0)) {
                    println!(
                        "SPLINE TEST FAIL: format — inherits {inherits}, declared {declared:?} \
                         (wanted 640x360)"
                    );
                    std::process::exit(1);
                }
                // A clean shape with no curves on it, under a shape
                // transform: what the rest of the suite has been building
                // toward is a shape you can still edit point by point once
                // the whole thing is moved.
                if let Some(n) = self.egraph.node_mut(sid) {
                    anim::spline_channels_cleared(&mut n.anim);
                    if let NodeKind::Spline(sp) = &mut n.kind {
                        sp.points = [[0.35, 0.35], [0.65, 0.35], [0.5, 0.6]]
                            .iter()
                            .map(|p| engine::graph::SplinePoint::new(*p, 20.0))
                            .collect();
                        sp.mode = engine::graph::SplineMode::Linear;
                        sp.xf = engine::graph::ShapeXf {
                            translate: [0.05, 0.0],
                            rotate: 30.0,
                            ..Default::default()
                        };
                        // Matte controls are uniforms, so setting them here
                        // must not cost a recompile — stage 41 checks that.
                        sp.opacity = 0.5;
                        sp.expand = 3.0;
                        sp.falloff = 1.0;
                    }
                }
                self.spline.sel.clear();
                next = 35;
            }
            35 => {
                // Grab point 0 WHERE IT IS DRAWN, not where it is authored:
                // if the gizmo's hit test skipped the transform this press
                // lands on empty canvas and nothing moves.
                let Some(p) = pt_xf_scr(self, 0) else { return };
                next_stash = pt_xf(self, 0).unwrap_or([f32::NAN; 2]);
                self.test.spline_press = Some(p);
                click(raw, p, true);
                next = 36;
            }
            36 | 37 => {
                let Some(p) = self.test.spline_press else { return };
                let d = if stage == 36 { 20.0 } else { 40.0 };
                raw.events.push(egui::Event::PointerMoved(p + egui::vec2(d, 0.0)));
                next = stage + 1;
            }
            38 => {
                let Some(p) = self.test.spline_press else { return };
                click(raw, p + egui::vec2(40.0, 0.0), false);
                next = 39;
            }
            39 => next = 40, // settle
            40 => {
                let want_dx = 40.0 / self.test.rect.width();
                let now = pt_xf(self, 0).unwrap_or([f32::NAN; 2]);
                let (dx, dy) = (now[0] - stash[0], now[1] - stash[1]);
                // The point followed the cursor: straight along +x by the
                // drag, with no sideways component. A drag applied WITHOUT
                // the inverse transform would come out 30 degrees off, i.e.
                // dy about half of dx.
                if (dx - want_dx).abs() > want_dx * 0.25 || dy.abs() > want_dx * 0.2 {
                    println!(
                        "SPLINE TEST FAIL: drag on a transformed shape moved the \
                         point ({dx:.4}, {dy:.4}), wanted ({want_dx:.4}, 0)"
                    );
                    std::process::exit(1);
                }
                // And it edited the AUTHORED point, through the inverse — so
                // the authored delta is the rotated one, not the screen one.
                let auth = match self.egraph.node(sid).map(|n| &n.kind) {
                    Some(NodeKind::Spline(sp)) => sp.points[0].p,
                    _ => [f32::NAN; 2],
                };
                if (auth[1] - 0.35).abs() < want_dx * 0.2 {
                    println!(
                        "SPLINE TEST FAIL: authored point {auth:?} moved as if the \
                         shape transform were not there"
                    );
                    std::process::exit(1);
                }
                next = 41;
            }
            41 => {
                // Dialling the matte controls (stage 34 set them) must not
                // have changed the topology: parameters are uniforms.
                let g = self.egraph.to_engine_graph_at(Some(self.eval_tick()));
                let h = g.topology_hash();
                let mut g2 = g.clone();
                if let Some(NodeKind::Spline(sp)) = g2.nodes.get_mut(&sid) {
                    sp.opacity = 0.2;
                    sp.expand = -8.0;
                    sp.falloff = 0.25;
                    sp.invert = true;
                    sp.xf.rotate = -12.0;
                    sp.xf.translate = [-0.2, 0.3];
                }
                if g2.topology_hash() != h {
                    println!(
                        "SPLINE TEST FAIL: a matte/transform edit recompiled the shader"
                    );
                    std::process::exit(1);
                }
                // ...while the combine op IS structural, like a Blend's.
                let mut g3 = g.clone();
                if let Some(NodeKind::Spline(sp)) = g3.nodes.get_mut(&sid) {
                    sp.combine = engine::graph::MatteOp::Subtract;
                }
                if g3.topology_hash() == h {
                    println!("SPLINE TEST FAIL: the combine op never reached the shader");
                    std::process::exit(1);
                }
                // The soft crop button, baked: a SoftCrop boxing the shape,
                // keyed on every frame of the shot, its box holding every
                // drawn point.
                self.spline_soft_crop(sid, true);
                let Some((cid, crop)) = self.egraph.nodes.iter().find_map(|(id, n)| match &n.kind {
                    NodeKind::SoftCrop(p) => Some((*id, *p)),
                    _ => None,
                }) else {
                    println!("SPLINE TEST FAIL: the soft crop button made no SoftCrop");
                    std::process::exit(1);
                };
                let (a, b) = self.shot_span();
                let keyed = self.egraph.node(cid).map_or(0, |n| {
                    nodes::softcrop::CHANNELS
                        .iter()
                        .filter(|c| n.anim.get(c).is_some_and(|k| k.keys.len() as u64 == b - a + 1))
                        .count()
                });
                let (w, h) = self.node_res_f32(sid).unwrap_or((1920.0, 1080.0));
                let drawn = match self.egraph.node(sid).map(|n| n.kind_at(self.eval_tick())) {
                    Some(NodeKind::Spline(sp)) => engine::codegen::spline_polyline(&sp, [w, h]),
                    _ => Vec::new(),
                };
                let [l, t, r, bt] = crop.rect;
                let holds = drawn.iter().all(|(p, _)| {
                    p[0] >= l - 1e-4 && p[0] <= r + 1e-4 && p[1] >= t - 1e-4 && p[1] <= bt + 1e-4
                });
                if keyed != 4 || !holds || r - l < 0.01 || bt - t < 0.01 {
                    println!(
                        "SPLINE TEST FAIL: soft crop box {:?} (holds the shape {holds}, \
                         {keyed}/4 edges keyed on all {} frames)",
                        crop.rect,
                        b - a + 1
                    );
                    std::process::exit(1);
                }
                // Extract -> layers: the crop rides the matte into an
                // Extract, and the Transform made for that output is
                // conformed to it — pivot on the crop's centre, picture
                // exactly where it was.
                let plate = self.egraph.add(NodeKind::Grid(Default::default()), egui::pos2(-200.0, 300.0));
                let ex = self.egraph.add(
                    NodeKind::Extract(engine::graph::ExtractParams { n: 1 }),
                    egui::pos2(0.0, 400.0),
                );
                self.egraph.connect(ex, 0, plate);
                self.egraph.connect(ex, 1, cid);
                self.extract_to_layers(ex);
                let stack = self.selected.first().copied();
                let xf = stack.and_then(|l| self.egraph.input_of(l, 0));
                let Some(NodeKind::Transform(t)) = xf.and_then(|x| self.egraph.node(x)).map(|n| n.kind.clone())
                else {
                    println!("SPLINE TEST FAIL: extract -> layers made no Layer fed by a Transform");
                    std::process::exit(1);
                };
                if xf.map(|x| self.egraph.input_of(x, 0)) != Some(Some(ex)) {
                    println!("SPLINE TEST FAIL: the layer's Transform is not fed by the Extract");
                    std::process::exit(1);
                }
                let moved = (t.center[0] - 0.5).abs() + (t.center[1] - 0.5).abs() > 1e-3;
                let still = t.translate[0].abs() < 1e-2 && t.translate[1].abs() < 1e-2;
                if !moved || !still {
                    println!(
                        "SPLINE TEST FAIL: conform left centre {:?} translate {:?} \
                         (wanted the crop's centre, picture unmoved)",
                        t.center, t.translate
                    );
                    std::process::exit(1);
                }
                let now = pt_xf(self, 0).unwrap_or([f32::NAN; 2]);
                println!(
                    "SPLINE TEST PASS: S-click drew 3 points with independent \
                     feather; a keyed point evaluates 0.60 at frame 5 while the \
                     authored value holds at 0.30; dragging ONE point keyed the \
                     whole shape; marquee picked all 3; the corner handle scaled \
                     them about the opposite corner; D-click removed one \
                     (2 left, pick slid with it); the knob rotated the rest \
                     about the box centre; a D-drag on the centre moved the pivot alone and \
                     the knob then turned about it; a ctrl-drag on a corner distorted the pick; \
                     T/E/R held moved, scaled and turned the pick about the press; \
                     a format declares the shape's own resolution; \
                     O/I held on a point feathered it outward and inward; \
                     under a shape transform a point still grabs where it is \
                     DRAWN and follows the cursor (point 0 now at \
                     {:.3},{:.3}); opacity/expand/falloff/transform are uniforms \
                     and the combine op is topology; the soft crop button baked a \
                     SoftCrop round it; Extract -> layers conformed its Transform to it",
                    now[0], now[1]
                );
                std::process::exit(0);
            }
            _ => {}
        }
        self.test.spline = Some((next, frames, sid, next_stash));
    }
}
