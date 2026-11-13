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
            } else if clone_mode && mods.alt {
                // ALT + clone = picking the source, not painting. It holds
                // for the WHOLE gesture, not just the press: an alt-drag
                // that fell through to the paint branch would quietly
                // extend the previous stroke.
                if ui.input(|i| i.pointer.button_pressed(egui::PointerButton::Primary)) {
                    if let Some(p) =
                        ui.ctx().pointer_latest_pos().filter(|p| rect.contains(*p))
                    {
                        self.brush.clone_src = Some(to_norm(p));
                        // A fresh anchor re-derives the offset on the next
                        // stroke; until then there is nothing to paint with.
                        self.brush.clone_off = None;
                    }
                }
            } else if !scrubbing && !cmd && (started || dragging) {
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
                    // Aligned clone: the offset is fixed when the first
                    // stroke after an alt-click starts, and every later
                    // stroke reuses it, so a patch stays registered with its
                    // source instead of jumping per stroke.
                    if started && clone_mode && self.brush.clone_off.is_none() {
                        if let Some(src) = self.brush.clone_src {
                            self.brush.clone_off =
                                Some([src[0] - norm[0], src[1] - norm[1]]);
                        }
                    }
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
