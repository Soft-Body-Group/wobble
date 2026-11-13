//! `--test-gizmokey`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    /// --test-gizmokey driver: with TranslateX keyed 0->0 / 12->240, pause
    /// at frame 6 (evaluated x = 120), drag the pivot right, and verify the
    /// drag became a NEW key at frame 6 while authored params stayed 0.
    pub(crate) fn drive_test_gizmokey(&mut self, raw: &mut egui::RawInput) {
        let Some((stage, mut frames)) = self.test_gizmokey else { return };
        frames += 1;
        if frames > 1800 {
            println!("GIZMOKEY TEST FAIL: timeout at stage {stage}");
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
        // The pivot where it is DRAWN: centre + the EVALUATED translate, so
        // this follows the curve instead of assuming a value (phase two
        // drags at a different frame, and a locked x sits wherever its curve
        // left it).
        let pivot = |s: &Self| -> egui::Pos2 {
            let (ow, oh) = s.comp.output_size().unwrap_or((1920, 1080));
            let (ow, oh) = (ow as f32, oh as f32);
            // The gizmo draws in the NODE's own pixels and maps them through
            // the OUTPUT's — which is the 4K branch here, not the 2K plate
            // this Transform sits on. Aiming in output pixels lands on a
            // scale handle instead of the pivot (it did, silently, and phase
            // one was keying scale.x as well as translate.x).
            let (nw, nh) = s.node_res_f32(2).unwrap_or((ow, oh));
            let r = s.image_rect;
            let (tx, ty) = match s.egraph.node(2).map(|n| n.kind_at(s.eval_tick())) {
                Some(NodeKind::Transform(t)) => (t.translate[0], t.translate[1]),
                _ => (0.0, 0.0),
            };
            egui::pos2(
                r.min.x + (0.5 * nw + tx) / ow * r.width(),
                r.min.y + (0.5 * nh + ty) / oh * r.height(),
            )
        };
        let mut next = stage;
        match stage {
            0 if self.playback.is_some() && self.displayed_total >= 3 => {
                if let Some(n) = self.egraph.node_mut(2) {
                    let c = n.anim.entry(anim::Channel::TranslateX).or_default();
                    c.set_key(0, 0.0);
                    c.set_key(12, 240.0);
                }
                self.playing = false;
                self.selected = vec![2];
                self.seek_to(6);
                next = 1;
            }
            1 => {
                if !self.force_display && self.eval_tick() == 6 {
                    next = 2;
                }
            }
            2 => {
                press(raw, pivot(self), true);
                next = 3;
            }
            3 | 4 => {
                let d = if stage == 3 { 15.0 } else { 30.0 };
                raw.events.push(egui::Event::PointerMoved(pivot(self) + egui::vec2(d, 0.0)));
                next = stage + 1;
            }
            5 => {
                press(raw, pivot(self) + egui::vec2(30.0, 0.0), false);
                next = 6;
            }
            6 => next = 7, // settle
            7 => {
                let n = self.egraph.node(2).expect("transform node");
                let authored = match &n.kind {
                    NodeKind::Transform(t) => t.translate[0],
                    _ => f32::NAN,
                };
                let curve = n.anim.get(&anim::Channel::TranslateX);
                let key6 = curve.and_then(|c| {
                    c.keys.iter().find(|k| k.frame == 6.0).map(|k| k.value)
                });
                let nkeys = curve.map(|c| c.keys.len()).unwrap_or(0);
                // The drag moved the pivot by its pointer travel mapped into
                // the node's own pixels (the first dragged frame is the
                // drag_started skip): clearly past the curve's 120 at this
                // frame, and nowhere near flying off.
                match key6 {
                    Some(v) if authored == 0.0 && nkeys == 3 && v > 121.0 && v < 400.0 => {
                        self.test_gizmokey_key6 = v;
                    }
                    _ => {
                        println!(
                            "GIZMOKEY TEST FAIL: key6={key6:?} nkeys={nkeys} authored={authored}"
                        );
                        std::process::exit(1);
                    }
                }
                // PHASE 2: lock translate.x and drag the same gizmo
                // DIAGONALLY with autokey on. x must not move — no key, no
                // authored write — while y keys like any other channel.
                if let Some(n) = self.egraph.node_mut(2) {
                    anim::set_locked(&mut n.anim, anim::Channel::TranslateX, true);
                }
                self.autokey = true;
                self.seek_to(8);
                next = 8;
            }
            8 => {
                if !self.force_display && self.eval_tick() == 8 {
                    next = 9;
                }
            }
            9 => {
                press(raw, pivot(self), true);
                next = 10;
            }
            10 | 11 => {
                let d = if stage == 10 { 15.0 } else { 30.0 };
                raw.events
                    .push(egui::Event::PointerMoved(pivot(self) + egui::vec2(d, d)));
                next = stage + 1;
            }
            12 => {
                press(raw, pivot(self) + egui::vec2(30.0, 30.0), false);
                next = 13;
            }
            13 => next = 14, // settle
            14 => {
                let n = self.egraph.node(2).expect("transform node");
                let (ax, ay) = match &n.kind {
                    NodeKind::Transform(t) => (t.translate[0], t.translate[1]),
                    _ => (f32::NAN, f32::NAN),
                };
                let x = n.anim.get(&anim::Channel::TranslateX);
                let y = n.anim.get(&anim::Channel::TranslateY);
                let x_keys = x.map(|c| c.keys.len()).unwrap_or(0);
                let x_at_6 =
                    x.and_then(|c| c.keys.iter().find(|k| k.frame == 6.0).map(|k| k.value));
                let x_at_8 = x.and_then(|c| c.keys.iter().find(|k| k.frame == 8.0).map(|k| k.value));
                let y_at_8 = y.and_then(|c| c.keys.iter().find(|k| k.frame == 8.0).map(|k| k.value));
                let locked = x.is_some_and(|c| c.locked);
                let ok = locked
                    && x_keys == 3
                    && x_at_8.is_none()
                    && x_at_6 == Some(self.test_gizmokey_key6)
                    && ax == 0.0
                    && y_at_8.is_some_and(|v| v.abs() > 1.0);
                if !ok {
                    println!(
                        "GIZMOKEY TEST FAIL: locked x took an edit — locked={locked} \
                         x_keys={x_keys} x@6={x_at_6:?} x@8={x_at_8:?} authored_x={ax} \
                         y@8={y_at_8:?} authored_y={ay}"
                    );
                    std::process::exit(1);
                }
                self.test_gizmokey_y8 = y_at_8.unwrap_or(f32::NAN);
                // PHASE 3: the motion path. Off, nothing is drawn; on, the
                // animated pivot's path goes on screen.
                if self.last_path_points != 0 {
                    println!(
                        "GIZMOKEY TEST FAIL: {} path samples drawn with the path OFF",
                        self.last_path_points
                    );
                    std::process::exit(1);
                }
                self.prefs.motion_path = Some(true);
                frames = 0;
                next = 15;
            }
            15 if frames > 3 => {
                if self.last_path_points < 8 {
                    println!(
                        "GIZMOKEY TEST FAIL: the motion path drew {} samples with it ON \
                         (translate.x is keyed at 0/6/12 — it moves)",
                        self.last_path_points
                    );
                    std::process::exit(1);
                }
                // PHASE 4: GRAB the key at frame 0 on the path and drag it.
                // x is unlocked again first, so both axes may move.
                if let Some(n) = self.egraph.node_mut(2) {
                    anim::set_locked(&mut n.anim, anim::Channel::TranslateX, false);
                }
                let (k0, size) = {
                    let n = self.egraph.node(2).expect("transform node");
                    let mut k = n.kind.clone();
                    anim::apply_at(&mut k, &n.anim, 0.0);
                    (k, self.node_res_f32(2).unwrap_or((2048.0, 1080.0)))
                };
                let at = element_point(&k0, size).expect("a transform has a point");
                self.test_gizmokey_before = match &k0 {
                    NodeKind::Transform(t) => [t.translate[0], t.translate[1]],
                    _ => [f32::NAN; 2],
                };
                let c = self.node_point_on_screen(2, (at[0], at[1]));
                press(raw, c, true);
                self.test_pos = Some(c);
                frames = 0;
                next = 16;
            }
            16 | 17 => {
                let d = if stage == 16 { 20.0 } else { 40.0 };
                raw.events
                    .push(egui::Event::PointerMoved(self.test_pos.unwrap() + egui::vec2(d, d / 2.0)));
                next = stage + 1;
            }
            18 => {
                press(raw, self.test_pos.unwrap() + egui::vec2(40.0, 20.0), false);
                next = 19;
            }
            19 if frames > 2 => {
                let n = self.egraph.node(2).expect("transform node");
                let x = n.anim.get(&anim::Channel::TranslateX).expect("x curve");
                let y = n.anim.get(&anim::Channel::TranslateY).expect("y curve");
                let key = |c: &anim::AnimCurve, f: f64| {
                    c.keys.iter().find(|k| (k.frame - f).abs() < 1e-6).map(|k| k.value)
                };
                // Screen pixels -> this node's pixels, the way the path maps.
                let (ow, _) = self.comp.output_size().unwrap_or((4096, 2160));
                let per_px = ow as f32 / self.image_rect.width().max(1.0);
                let want_x = self.test_gizmokey_before[0] + 40.0 * per_px;
                let want_y = self.test_gizmokey_before[1] + 20.0 * per_px;
                let got_x = key(x, 0.0);
                let got_y = key(y, 0.0);
                let near = |a: Option<f32>, b: f32| a.is_some_and(|v| (v - b).abs() < 8.0);
                // The key MOVED, at its own frame, and the rest stayed put.
                let ok = near(got_x, want_x)
                    && near(got_y, want_y)
                    && x.keys.len() == 3
                    && key(x, 6.0).is_some_and(|v| (v - self.test_gizmokey_key6).abs() < 1e-3)
                    && key(x, 12.0).is_some_and(|v| (v - 240.0).abs() < 1e-3);
                if !ok {
                    println!(
                        "GIZMOKEY TEST FAIL: dragging the key on the path gave x@0={got_x:?} \
                         y@0={got_y:?} (wanted ~{want_x:.1} / ~{want_y:.1}), x keys \
                         {:?}",
                        x.keys.iter().map(|k| (k.frame, k.value)).collect::<Vec<_>>()
                    );
                    std::process::exit(1);
                }
                println!(
                    "GIZMOKEY TEST PASS: pivot drag keyed frame 6 at {:.1} (3 keys, authored \
                     still 0); with translate.x LOCKED a diagonal drag left x untouched and \
                     keyed y at 8 ({:.1}); the motion path drew {} samples once switched on \
                     (none before); and dragging its key at frame 0 moved THAT key to \
                     ({:.1}, {:.1}) with frames 6 and 12 untouched",
                    self.test_gizmokey_key6,
                    self.test_gizmokey_y8,
                    self.last_path_points,
                    got_x.unwrap_or(f32::NAN),
                    got_y.unwrap_or(f32::NAN),
                );
                std::process::exit(0);
            }
            _ => {}
        }
        self.test_gizmokey = Some((next, frames));
    }
}
