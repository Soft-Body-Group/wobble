//! Paint capture: with a Paint node selected, a plain left-drag brushes into it.
//! Moved out of `viewer_ui` (lib.rs) unchanged.

use crate::*;

impl Gfx {
    /// Paint capture: with a Paint node selected, a plain left-drag brushes into it.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn viewer_paint_capture(
        &mut self,
        ui: &mut egui::Ui,
        cmd: bool,
        cur_t: u64,
        dropping: bool,
        nav: &egui::Response,
        owf: f32,
        ohf: f32,
        rect: egui::Rect,
        scrubbing: bool,
    ) {
        // Paint capture: with a Paint node selected, plain left-drag brushes
        // into it (strokes stored normalized, inverted through the
        // downstream chain like every other gizmo).
        let paint_sel = self.selected.iter().copied().filter(|_| !dropping).find(|id| {
            matches!(self.egraph.node(*id).map(|n| &n.kind), Some(NodeKind::Paint(_)))
        });
        if let Some(pid) = paint_sel {
            use engine::graph::BrushMode;
            let started = nav.drag_started_by(egui::PointerButton::Primary);
            let dragging = nav.dragged_by(egui::PointerButton::Primary);
            let mods = ui.input(|i| i.modifiers);
            let clone_mode = self.brush.mode == BrushMode::Clone;
            // The brush chain, shared by the gesture and the cursor ring:
            // screen point -> the node's own normalized space, inverted
            // through everything downstream of it.
            let hd = downstream_proj(
                &self.egraph,
                pid,
                &|id| self.node_res_f32(id).unwrap_or((owf, ohf)),
                cur_t,
            );
            let hd_inv = engine::codegen::mat3_inv(&hd);
            let to_norm = |p: egui::Pos2| {
                let out_px = egui::vec2(
                    (p.x - rect.min.x) / rect.width() * owf,
                    (p.y - rect.min.y) / rect.height() * ohf,
                );
                let np = apply_h(&hd_inv, out_px);
                [np.x / owf, np.y / ohf]
            };
            // HELD F: pointer MOTION sets the brush — Blender's F, so the
            // hand painting never leaves the tablet. Plain: x is size, y is
            // softness. Shift: x opacity, y angle. Ctrl: x the ellipse's
            // aspect. Size moves multiplicatively, like the keys — a 4px
            // brush and a 200px one both want proportional steps. Nothing
            // paints while it is held, and a focused text field owns F.
            let pens = pen::drain();
            let adjusting = !ui.ctx().egui_wants_keyboard_input()
                && ui.input(|i| keymap::down(i, keymap::Act::BrushAdjust));
            // HELD alt (or cmd on a Mac) + clone: the brush freezes where it
            // is and pointer travel moves the SOURCE; let go and it paints
            // again with the new offset. Nothing to click, nothing to exit.
            let clone_hold = clone_mode && (mods.alt || mods.mac_cmd);
            let held = adjusting || clone_hold;
            let hover = ui.ctx().pointer_latest_pos().filter(|p| rect.contains(*p));
            let primary_down = ui.input(|i| i.pointer.primary_down());
            if !primary_down {
                self.brush.swallow = false;
            } else if held && (started || ui.input(|i| i.pointer.primary_pressed())) {
                self.brush.swallow = true;
            }
            if held {
                // The ring stays where the gesture began, whatever the hand
                // does while the key is down.
                if self.brush.pin.is_none() {
                    self.brush.pin = hover.map(to_norm);
                }
            } else {
                self.brush.pin = None;
            }
            if !clone_hold || adjusting {
                self.brush.grab = None;
            }
            if adjusting {
                let d = ui.input(|i| i.pointer.delta());
                if mods.ctrl {
                    self.brush.aspect = (self.brush.aspect + d.x * 0.005).clamp(0.05, 1.0);
                } else if mods.shift {
                    self.brush.color[3] = (self.brush.color[3] + d.x * 0.005).clamp(0.0, 1.0);
                    // Kept in (-180, 180] so the panel's field reads the
                    // same number the gesture made.
                    self.brush.angle = (self.brush.angle - d.y * 0.5 + 180.0).rem_euclid(360.0) - 180.0;
                } else {
                    self.brush.radius =
                        (self.brush.radius * (1.0 + d.x * 0.01)).clamp(1.0, 500.0);
                    self.brush.soft = (self.brush.soft - d.y * 0.005).clamp(0.0, 1.0);
                }
            } else if clone_hold {
                // The offset moves by the pointer's travel since the grab,
                // so a second hold nudges the source from where it was
                // instead of snapping it onto the brush. The first hold
                // starts from zero: the source leaves the brush with the hand.
                if let Some(n) = hover.map(to_norm) {
                    let (p0, o0) = *self
                        .brush
                        .grab
                        .get_or_insert((n, self.brush.clone_off.unwrap_or([0.0, 0.0])));
                    self.brush.clone_off = Some([o0[0] + n[0] - p0[0], o0[1] + n[1] - p0[1]]);
                }
            } else if !scrubbing && !cmd && !self.brush.swallow && (started || dragging) {
                // Every cursor sample since last frame (pen.rs), with its
                // pressure — a fast pen stroke is many samples a frame, and
                // one per frame draws polygons. A stroke starts where the
                // PRESS was (egui starts a drag once the pointer has moved,
                // and the samples of that frame are the stroke's first
                // stretch); a frame with no samples (a synthetic gesture, a
                // pointer egui moved itself) falls back to egui's one.
                let ppp = ui.ctx().pixels_per_point();
                let mut hand: Vec<(egui::Pos2, Option<f32>)> =
                    pens.iter().map(|s| (s.pos(ppp), s.pressure)).collect();
                if started {
                    if let Some(o) = ui.input(|i| i.pointer.press_origin()) {
                        let p = hand.first().map(|h| h.1).unwrap_or_else(pen::pressure);
                        hand.insert(0, (o, p));
                    }
                }
                if hand.is_empty() {
                    if let Some(p) = ui.ctx().pointer_latest_pos() {
                        hand.push((p, pen::pressure()));
                    }
                }
                if let Some(&(p0, press0)) = hand.first() {
                    let norm = to_norm(p0);
                    // Aligned clone: every stroke reuses the offset the last
                    // alt/cmd hold left, so a patch stays registered with its
                    // source instead of jumping per stroke.
                    let clone_off = if clone_mode { self.brush.clone_off } else { None };
                    // Cloning with no source set would lay down a
                    // zero-offset stroke, i.e. an invisible no-op.
                    let armed = !clone_mode || clone_off.is_some();
                    let life = self
                        .egraph
                        .node(pid)
                        .and_then(|n| match &n.kind {
                            NodeKind::Paint(pp) => Some(pp.life_default),
                            _ => None,
                        })
                        .unwrap_or_default()
                        .at(cur_t);
                    // Read before the node borrow below: these are methods.
                    let strength = self.brush_strength();
                    let mode = self.brush.mode;
                    if let (true, Some(editor::ENode { kind: NodeKind::Paint(pp), .. })) =
                        (armed, self.egraph.node_mut(pid))
                    {
                        if started {
                            // A pen stroke records its pressure, a mouse
                            // stroke none (full pressure, as ever).
                            pp.strokes.push(engine::graph::PaintStroke {
                                points: vec![norm],
                                radius: self.brush.radius,
                                softness: self.brush.soft,
                                color: self.brush.color,
                                life,
                                clone_off,
                                mode,
                                aspect: self.brush.aspect,
                                angle: self.brush.angle,
                                strength,
                                blend: self.brush.blend,
                                pressure: press0.into_iter().collect(),
                                press: self.brush.press,
                            });
                        }
                        if let Some(stroke) = pp.strokes.last_mut() {
                            let pen_stroke = !stroke.pressure.is_empty();
                            for &(p, press) in hand.iter().skip(usize::from(started)) {
                                let norm = to_norm(p);
                                // Thin while the pen is light: the spacing
                                // of kept points follows the pressured size.
                                let f = stroke.press.factors(press.unwrap_or(1.0)).0;
                                let min_px = (stroke.radius * f * 0.25).max(1.5);
                                let far = stroke.points.last().map(|l| {
                                    let dx = (norm[0] - l[0]) * owf;
                                    let dy = (norm[1] - l[1]) * ohf;
                                    (dx * dx + dy * dy).sqrt() >= min_px
                                });
                                if far.unwrap_or(true) {
                                    stroke.points.push(norm);
                                    if pen_stroke {
                                        // A pen that lifts mid-sample reads
                                        // None: keep its last pressure.
                                        let last = stroke.pressure.last().copied().unwrap_or(1.0);
                                        stroke.pressure.push(press.unwrap_or(last));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
