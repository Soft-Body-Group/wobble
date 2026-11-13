//! `--test-click`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    /// --test-click driver: wait for node rects, inject press then release at
    /// the first Read node's center, then assert it ended up selected.
    pub(crate) fn drive_test_click(&mut self, stage: u32, raw: &mut egui::RawInput) {
        if stage < 1000 {
            // Pin the target once rects exist and the graph has settled.
            let found = self
                .node_rects
                .iter()
                .find(|(id, _)| {
                    matches!(self.egraph.node(*id).map(|n| &n.kind), Some(NodeKind::Read { .. }))
                })
                .map(|(id, r)| (*id, r.center()));
            match found {
                Some(t) if stage >= 3 => {
                    self.test_target = Some(t);
                    self.test_click = Some(1000);
                }
                _ if stage > 240 => {
                    println!("CLICK TEST FAIL: no node rects after {stage} frames");
                    std::process::exit(1);
                }
                _ => self.test_click = Some(stage + 1),
            }
            return;
        }
        let Some((tid, pos)) = self.test_target else {
            println!("CLICK TEST FAIL: target vanished");
            std::process::exit(1);
        };
        match stage {
            1000 | 1001 => {
                raw.events.push(egui::Event::PointerMoved(pos));
                raw.events.push(egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed: stage == 1000,
                    modifiers: Default::default(),
                });
                self.test_click = Some(stage + 1);
            }
            1002 => self.test_click = Some(1003), // settle frame
            1003 => {
                let pass = self.selected.contains(&tid);
                println!(
                    "CLICK TEST {}: clicked node {:?} at {:?}, selected={:?}",
                    if pass { "PASS" } else { "FAIL" },
                    tid,
                    pos,
                    self.selected
                );
                if !pass {
                    std::process::exit(1);
                }
                // Phase 2: click empty graph background left of all nodes.
                let bb = self
                    .node_rects
                    .iter()
                    .map(|(_, r)| *r)
                    .reduce(|a, b| a.union(b))
                    .unwrap();
                self.test_target = Some((tid, egui::pos2(bb.min.x - 40.0, bb.center().y)));
                self.test_click = Some(2000);
            }
            2000 | 2001 => {
                raw.events.push(egui::Event::PointerMoved(pos));
                raw.events.push(egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed: stage == 2000,
                    modifiers: Default::default(),
                });
                self.test_click = Some(stage + 1);
            }
            2002 => self.test_click = Some(2003),
            2003 => {
                let pass = self.selected.is_empty();
                println!(
                    "DESELECT TEST {}: background click at {:?}, selected={:?}",
                    if pass { "PASS" } else { "FAIL" },
                    pos,
                    self.selected
                );
                if !pass {
                    std::process::exit(1);
                }
                // Phase 3: click the Transform node so its gizmo (and the
                // per-pixel ink pipeline) actually renders for a few frames —
                // a pipeline validation error would abort the process.
                let target = self
                    .node_rects
                    .iter()
                    .find(|(id, _)| {
                        matches!(
                            self.egraph.node(*id).map(|n| &n.kind),
                            Some(NodeKind::Transform(_))
                        )
                    })
                    .map(|(id, r)| (*id, r.center()));
                let Some(t) = target else {
                    println!("INK TEST FAIL: no transform node");
                    std::process::exit(1);
                };
                self.test_target = Some(t);
                self.test_click = Some(3000);
            }
            3000 | 3001 => {
                let (_, tpos) = self.test_target.unwrap();
                raw.events.push(egui::Event::PointerMoved(tpos));
                raw.events.push(egui::Event::PointerButton {
                    pos: tpos,
                    button: egui::PointerButton::Primary,
                    pressed: stage == 3000,
                    modifiers: Default::default(),
                });
                self.test_click = Some(stage + 1);
            }
            n @ 3002..=3011 => self.test_click = Some(n + 1),
            3012 => {
                let (tid, _) = self.test_target.unwrap();
                let pass = self.selected.contains(&tid);
                println!(
                    "INK TEST {}: transform gizmo + per-pixel ink rendered 10 frames (selected={:?})",
                    if pass { "PASS" } else { "FAIL" },
                    self.selected
                );
                if !pass {
                    std::process::exit(1);
                }
                self.test_click = Some(4000);
            }
            // Phase 4: drag the pivot handle in the viewer; the transform's
            // translate must change.
            4000 => {
                if !self.layout_settled() {
                    return; // panels still animating: the pivot is moving
                }
                // The NODE's centre, where its pivot is drawn — the proxy
                // phase repeats this exact gesture and compares distances,
                // so both must aim the same way.
                let (tid, _) = self.test_target.unwrap();
                let (nw, nh) = self.node_res_f32(tid).unwrap_or((2048.0, 1080.0));
                let c = self.node_point_on_screen(tid, (0.5 * nw, 0.5 * nh));
                self.test_pos = Some(c);
                raw.events.push(egui::Event::PointerMoved(c));
                raw.events.push(egui::Event::PointerButton {
                    pos: c,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: Default::default(),
                });
                self.test_click = Some(4001);
            }
            4001 | 4002 => {
                let c = self.test_pos.unwrap()
                    + egui::vec2(if stage == 4001 { 15.0 } else { 30.0 }, 0.0);
                raw.events.push(egui::Event::PointerMoved(c));
                self.test_click = Some(stage + 1);
            }
            4003 => {
                let c = self.test_pos.unwrap() + egui::vec2(30.0, 0.0);
                raw.events.push(egui::Event::PointerButton {
                    pos: c,
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers: Default::default(),
                });
                self.test_click = Some(4004);
            }
            4004 => {
                let (tid, _) = self.test_target.unwrap();
                let tx = match self.egraph.node(tid).map(|n| &n.kind) {
                    Some(NodeKind::Transform(t)) => t.translate[0],
                    _ => 0.0,
                };
                if tx.abs() <= 1.0 {
                    println!("GIZMO DRAG TEST FAIL: pivot drag moved translate.x to {tx:.1}");
                    std::process::exit(1);
                }
                println!("GIZMO DRAG TEST PASS: pivot drag moved translate.x to {tx:.1}");
                self.test_proxy_drag.0 = tx * self.image_rect.width();
                // hud off phase: the same drag on the same spot must now do
                // NOTHING — including for X-pinned nodes. Hidden through
                // the KEY (shift+W), not the pref, so the hotkey is driven.
                raw.events.insert(0, egui::Event::ModifiersChanged(egui::Modifiers::SHIFT));
                for pressed in [true, false] {
                    raw.events.push(egui::Event::Key {
                        key: egui::Key::W,
                        physical_key: None,
                        pressed,
                        repeat: false,
                        modifiers: egui::Modifiers::SHIFT,
                    });
                }
                self.gizmo_pinned = vec![tid];
                self.test_pos = None;
                self.test_click = Some(4005);
            }
            // Re-run the exact pivot gesture with the HUD hidden.
            4005 => {
                if !self.layout_settled() {
                    return;
                }
                if self.prefs.hud_hide != Some(true) {
                    println!("HUD OFF TEST FAIL: shift+W did not hide the HUD (hud_hide {:?})", self.prefs.hud_hide);
                    std::process::exit(1);
                }
                let (tid, _) = self.test_target.unwrap();
                let tx0 = match self.egraph.node(tid).map(|n| &n.kind) {
                    Some(NodeKind::Transform(t)) => t.translate[0],
                    _ => 0.0,
                };
                self.hud_off_tx0 = tx0;
                let c = {
                    let r = self.image_rect;
                    egui::pos2(
                        r.min.x + (0.5 + tx0 / 2048.0) * r.width(),
                        r.min.y + 0.5 * r.height(),
                    )
                };
                self.test_pos = Some(c);
                raw.events.push(egui::Event::PointerMoved(c));
                raw.events.push(egui::Event::PointerButton {
                    pos: c,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: Default::default(),
                });
                self.test_click = Some(4006);
            }
            4006 | 4007 => {
                let c = self.test_pos.unwrap()
                    + egui::vec2(if self.test_click == Some(4006) { 15.0 } else { 30.0 }, 0.0);
                raw.events.push(egui::Event::PointerMoved(c));
                self.test_click = Some(if self.test_click == Some(4006) { 4007 } else { 4008 });
            }
            4008 => {
                let c = self.test_pos.unwrap() + egui::vec2(30.0, 0.0);
                raw.events.push(egui::Event::PointerButton {
                    pos: c,
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers: Default::default(),
                });
                self.test_click = Some(4009);
            }
            4009 => {
                let (tid, _) = self.test_target.unwrap();
                let tx = match self.egraph.node(tid).map(|n| &n.kind) {
                    Some(NodeKind::Transform(t)) => t.translate[0],
                    _ => 0.0,
                };
                let moved = (tx - self.hud_off_tx0).abs();
                if moved > 0.5 {
                    println!(
                        "HUD OFF TEST FAIL: hidden gizmo still dragged (moved {moved:.1})"
                    );
                    std::process::exit(1);
                }
                println!("HUD OFF TEST PASS: hidden manipulators (incl. pinned) are inert");
                // Focus phase: the HUD comes back, and the node is left
                // UNSELECTED with no properties panel, so the ctrl-click has
                // something to actually change.
                self.prefs.hud_hide = Some(false);
                self.gizmo_pinned = vec![tid];
                self.selected.clear();
                self.prop_order.clear();
                self.props_visible = false;
                self.test_pos = None;
                self.test_click = Some(4010);
            }
            // Ctrl+cmd-CLICK (press and release without moving) on the pivot:
            // select, HUD AND properties.
            4010 => {
                if !self.layout_settled() {
                    return;
                }
                let (tid, _) = self.test_target.unwrap();
                let (cx, cy, tx, ty) = match self.egraph.node(tid).map(|n| &n.kind) {
                    Some(NodeKind::Transform(t)) => {
                        (t.center[0], t.center[1], t.translate[0], t.translate[1])
                    }
                    _ => (0.5, 0.5, 0.0, 0.0),
                };
                // A gizmo draws in its NODE's pixels and maps them through
                // the OUTPUT's, which here is the 4K branch, not the 2K plate
                // this Transform sits on. Aiming in output pixels lands on a
                // scale handle instead of the pivot — which is exactly why
                // this stage has been failing.
                let (nw, nh) = self.node_res_f32(tid).unwrap_or((2048.0, 1080.0));
                let c = self.node_point_on_screen(tid, (cx * nw + tx, cy * nh + ty));
                self.test_pos = Some(c);
                let ctrl = egui::Modifiers { ctrl: true, mac_cmd: true, command: true, ..Default::default() };
                raw.events.insert(0, egui::Event::ModifiersChanged(ctrl));
                raw.events.push(egui::Event::PointerMoved(c));
                raw.events.push(egui::Event::PointerButton {
                    pos: c,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: ctrl,
                });
                self.test_click = Some(4011);
            }
            4011 => {
                let c = self.test_pos.unwrap();
                let ctrl = egui::Modifiers { ctrl: true, mac_cmd: true, command: true, ..Default::default() };
                raw.events.insert(0, egui::Event::ModifiersChanged(ctrl));
                raw.events.push(egui::Event::PointerButton {
                    pos: c,
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers: ctrl,
                });
                self.test_click = Some(4012);
            }
            // D-DRAG the pivot: the centre moves under the pointer and the
            // picture does not move at all.
            5000 => {
                if !self.layout_settled() {
                    return;
                }
                let (tid, _) = self.test_target.unwrap();
                let Some(NodeKind::Transform(t)) = self.egraph.node(tid).map(|n| &n.kind)
                else {
                    println!("PIVOT DRAG TEST FAIL: transform node vanished");
                    std::process::exit(1);
                };
                let (nw, nh) = self.node_res_f32(tid).unwrap_or((2048.0, 1080.0));
                let c = self.node_point_on_screen(
                    tid,
                    (t.center[0] * nw + t.translate[0], t.center[1] * nh + t.translate[1]),
                );
                self.test_pos = Some(c);
                raw.events.push(egui::Event::Key {
                    key: egui::Key::D,
                    physical_key: None,
                    pressed: true,
                    repeat: false,
                    modifiers: Default::default(),
                });
                raw.events.push(egui::Event::PointerMoved(c));
                raw.events.push(egui::Event::PointerButton {
                    pos: c,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: Default::default(),
                });
                self.test_click = Some(5001);
            }
            5001 | 5002 => {
                let c = self.test_pos.unwrap()
                    + egui::vec2(if stage == 5001 { 15.0 } else { 30.0 }, 0.0);
                raw.events.push(egui::Event::PointerMoved(c));
                self.test_click = Some(stage + 1);
            }
            5003 => {
                let c = self.test_pos.unwrap() + egui::vec2(30.0, 0.0);
                raw.events.push(egui::Event::PointerButton {
                    pos: c,
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers: Default::default(),
                });
                raw.events.push(egui::Event::Key {
                    key: egui::Key::D,
                    physical_key: None,
                    pressed: false,
                    repeat: false,
                    modifiers: Default::default(),
                });
                self.test_click = Some(5004);
            }
            5004 => {
                let (tid, _) = self.test_target.unwrap();
                let Some(NodeKind::Transform(t)) = self.egraph.node(tid).map(|n| &n.kind)
                else {
                    println!("PIVOT DRAG TEST FAIL: transform node vanished");
                    std::process::exit(1);
                };
                let (w, h) = self.node_res_f32(tid).unwrap_or((2048.0, 1080.0));
                let aff = engine::codegen::forward_affine(t, [w, h]);
                // The picture: every affine term within half a pixel.
                let shift = (0..6)
                    .map(|k| (aff[k] - self.test_pivot_aff[k]).abs())
                    .fold(0.0f32, f32::max);
                // The pivot: it was pressed at the old centre (0.5, 0.5) and
                // dragged along +x, so it must now sit to the right of where
                // it was and on the same row. (The first frame's delta is
                // skipped by design, so the travel is less than the 30 px.)
                let pivot = egui::vec2(
                    t.center[0] * w + t.translate[0],
                    t.center[1] * h + t.translate[1],
                );
                // The old pivot in OUTPUT space is where the source centre
                // landed: the stored affine applied to (0.5, 0.5).
                let was = crate::apply_aff(&self.test_pivot_aff, egui::vec2(0.5 * w, 0.5 * h));
                let travel = pivot - was;
                let centred = (t.center[0] - 0.5).abs() > 0.002 || (t.center[1] - 0.5).abs() > 0.002;
                let pass = shift < 0.5 && travel.x > 5.0 && travel.y.abs() < 1.0 && centred;
                println!(
                    "PIVOT DRAG TEST {}: D-drag moved the pivot by ({:.1}, {:.1}) px to centre \
                     ({:.3}, {:.3}); the picture moved {shift:.3} px (want < 0.5)",
                    if pass { "PASS" } else { "FAIL" },
                    travel.x,
                    travel.y,
                    t.center[0],
                    t.center[1],
                );
                if !pass {
                    std::process::exit(1);
                }
                // A fit image's corner sits on the viewer's edge, where the
                // panel next door takes the press: zoom out a touch first.
                self.view_zoom = 0.7;
                self.view_pan = egui::Vec2::ZERO;
                // Reset now, not at the press: egui hit-tests against LAST
                // frame's widgets, so the handle must already be drawn where
                // the press will land.
                let (tid, _) = self.test_target.unwrap();
                if let Some(NodeKind::Transform(t)) =
                    self.egraph.node_mut(tid).map(|n| &mut n.kind)
                {
                    *t = engine::graph::TransformParams::default();
                }
                self.selected = vec![tid];
                self.test_click = Some(5097);
            }
            5097..=5099 => self.test_click = Some(stage + 1),
            // Distort phase: ctrl-drag the bottom-right CORNER handle. That
            // corner follows the pointer; the other three stay where they
            // were — a corner pin on the Transform, not a scale.
            5100 => {
                if !self.layout_settled() {
                    return;
                }
                // Fresh from the reset the frame's corner is the NODE's
                // bottom-right — which is not the image's when the viewed
                // output is bigger than this node (the 4K branch here).
                let (tid, _) = self.test_target.unwrap();
                let (nw, nh) = self.node_res_f32(tid).unwrap_or((2048.0, 1080.0));
                let c = self.node_point_on_screen(tid, (nw, nh));
                self.test_pos = Some(c);
                let ctrl = egui::Modifiers { ctrl: true, ..Default::default() };
                raw.events.insert(0, egui::Event::ModifiersChanged(ctrl));
                raw.events.push(egui::Event::PointerMoved(c));
                raw.events.push(egui::Event::PointerButton {
                    pos: c,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: ctrl,
                });
                self.test_click = Some(5101);
            }
            5101 | 5102 => {
                raw.events.insert(0, egui::Event::ModifiersChanged(egui::Modifiers { ctrl: true, ..Default::default() }));
                let c = self.test_pos.unwrap()
                    - if stage == 5101 { egui::vec2(20.0, 10.0) } else { egui::vec2(60.0, 40.0) };
                raw.events.push(egui::Event::PointerMoved(c));
                self.test_click = Some(stage + 1);
            }
            5103 => {
                let ctrl = egui::Modifiers { ctrl: true, ..Default::default() };
                raw.events.insert(0, egui::Event::ModifiersChanged(ctrl));
                let c = self.test_pos.unwrap() - egui::vec2(60.0, 40.0);
                raw.events.push(egui::Event::PointerButton {
                    pos: c,
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers: ctrl,
                });
                self.test_click = Some(5104);
            }
            5104 => {
                let (tid, _) = self.test_target.unwrap();
                let Some(NodeKind::Transform(t)) = self.egraph.node(tid).map(|n| n.kind.clone())
                else {
                    println!("DISTORT TEST FAIL: transform node vanished");
                    std::process::exit(1);
                };
                // Screen pixels -> this node's unit quad: through the
                // output's pixels, then the node's own size.
                let (nw, nh) = self.node_res_f32(tid).unwrap_or((2048.0, 1080.0));
                let (ow, oh) = self
                    .comp
                    .output_size()
                    .map(|(w, h)| (w as f32, h as f32))
                    .unwrap_or((nw, nh));
                let r = self.image_rect;
                let want = [
                    1.0 - 60.0 / r.width() * ow / nw,
                    1.0 - 40.0 / r.height() * oh / nh,
                ];
                let br = t.pin[2];
                let rest_still = [0usize, 1, 3].iter().all(|&i| {
                    (t.pin[i][0] - engine::graph::UNIT_QUAD[i][0]).abs() < 1e-3
                        && (t.pin[i][1] - engine::graph::UNIT_QUAD[i][1]).abs() < 1e-3
                });
                let untouched = t.translate == [0.0, 0.0] && t.scale == [1.0, 1.0];
                let pass = (br[0] - want[0]).abs() < 0.01
                    && (br[1] - want[1]).abs() < 0.01
                    && rest_still
                    && untouched;
                println!(
                    "DISTORT TEST {}: ctrl-drag on the corner pinned it to ({:.3}, {:.3}), \
                     wanted ({:.3}, {:.3}); other corners still {rest_still}, \
                     translate/scale untouched {untouched}",
                    if pass { "PASS" } else { "FAIL" },
                    br[0],
                    br[1],
                    want[0],
                    want[1],
                );
                if !pass {
                    std::process::exit(1);
                }
                self.test_click = Some(6000);
            }
            // Proxy phase: the SAME pivot drag with proxy on. Params are
            // full-res pixels and the proxy canvas is half size, so a
            // manipulator working in canvas pixels moved the image half as
            // far as the pointer — invisible to every other assertion here.
            6000 => {
                let (tid, _) = self.test_target.unwrap();
                if let Some(NodeKind::Transform(t)) =
                    self.egraph.node_mut(tid).map(|n| &mut n.kind)
                {
                    // Pivot back on the image centre, no rotate/scale (the
                    // pivot phase left both behind).
                    *t = engine::graph::TransformParams::default();
                }
                self.test_proxy_drag.1 = self.comp.output_size().map(|s| s.0).unwrap_or(0);
                self.test_proxy_drag.2 = Some(web_time::Instant::now());
                self.prefs.hud_hide = Some(false);
                self.selected = vec![tid];
                // Exactly what the toolbar chip does.
                self.proxy = 0.5;
                self.playback_dirty = true;
                self.last_engine_graph = None;
                self.test_click = Some(6001);
            }
            6001 => {
                // Wait for the half-size canvas to land, then settle. On the
                // WALL CLOCK: under test the loop is unpaced, so a frame count
                // is not a duration (CLAUDE.md).
                let w = self.comp.output_size().map(|s| s.0).unwrap_or(0);
                let settled = self.layout_settled();
                if w == 0 || w * 5 > self.test_proxy_drag.1 * 3 || !settled {
                    if self.test_proxy_drag.2.is_some_and(|t| t.elapsed().as_secs() > 30) {
                        println!(
                            "PROXY GIZMO TEST FAIL: timeout waiting for the proxy canvas \
                             (output width {w}, full {}, layout settled {settled})",
                            self.test_proxy_drag.1
                        );
                        std::process::exit(1);
                    }
                    return;
                }
                // The NODE's centre (its pivot), not the image's.
                let (nw, nh) = self.node_res_f32(tid).unwrap_or((2048.0, 1080.0));
                let c = self.node_point_on_screen(tid, (0.5 * nw, 0.5 * nh));
                self.test_pos = Some(c);
                raw.events.push(egui::Event::PointerMoved(c));
                raw.events.push(egui::Event::PointerButton {
                    pos: c,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: Default::default(),
                });
                self.test_click = Some(6002);
            }
            6002 | 6003 => {
                let c = self.test_pos.unwrap()
                    + egui::vec2(if stage == 6002 { 15.0 } else { 30.0 }, 0.0);
                raw.events.push(egui::Event::PointerMoved(c));
                self.test_click = Some(stage + 1);
            }
            6004 => {
                let c = self.test_pos.unwrap() + egui::vec2(30.0, 0.0);
                raw.events.push(egui::Event::PointerButton {
                    pos: c,
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers: Default::default(),
                });
                self.test_click = Some(6005);
            }
            6005 => {
                let (tid, _) = self.test_target.unwrap();
                let tx = match self.egraph.node(tid).map(|n| &n.kind) {
                    Some(NodeKind::Transform(t)) => t.translate[0],
                    _ => 0.0,
                };
                let ratio = tx * self.image_rect.width() / self.test_proxy_drag.0.max(1e-3);
                let pass = (ratio - 1.0).abs() < 0.05;
                println!(
                    "PROXY GIZMO TEST {}: the same pivot drag moved {:.2}x as far under proxy \
                     (translate.x {tx:.1}, want 1.00x)",
                    if pass { "PASS" } else { "FAIL" },
                    ratio
                );
                std::process::exit(if pass { 0 } else { 1 });
            }
            _ => {
                let (tid, _) = self.test_target.unwrap();
                if self.selected != vec![tid] {
                    println!(
                        "GIZMO FOCUS TEST FAIL: ctrl+cmd-click left the selection {:?}, wanted [{tid}] \
                         (frame {} shown {} playing {} at {:?})",
                        self.selected,
                        self.eval_tick(),
                        self.shown_tick,
                        self.playing,
                        self.test_pos,
                    );
                    std::process::exit(1);
                }
                if self.prop_order.first() != Some(&tid) || !self.props_visible {
                    println!(
                        "GIZMO FOCUS TEST FAIL: properties are {:?} (visible {}), wanted {tid} in front",
                        self.prop_order, self.props_visible
                    );
                    std::process::exit(1);
                }
                println!(
                    "GIZMO FOCUS TEST PASS: ctrl+cmd-clicking the pivot selected node {tid} and \
                     raised its properties"
                );
                // Pivot phase: give the transform a rotation and a
                // non-uniform scale first, so a centre that moved WITHOUT
                // translate compensating would visibly shift the picture.
                let sz = self.node_res_f32(tid).unwrap_or((2048.0, 1080.0));
                if let Some(NodeKind::Transform(t)) =
                    self.egraph.node_mut(tid).map(|n| &mut n.kind)
                {
                    t.rotate = 30.0;
                    t.scale = [1.5, 0.8];
                    self.test_pivot_aff = engine::codegen::forward_affine(t, [sz.0, sz.1]);
                }
                self.test_pos = None;
                self.test_click = Some(5000);
            }

        }
    }
}
