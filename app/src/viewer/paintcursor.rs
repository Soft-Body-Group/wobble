//! Paint: the brush cursor, at the size and softness it will land at.
//! Moved out of `viewer_ui` (lib.rs) unchanged.

use crate::*;

impl Gfx {
    /// Paint: the brush cursor, at the size and softness it will land at.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gizmo_paint_cursor(
        &mut self,
        ui: &mut egui::Ui,
        avail: egui::Rect,
        cur_t: u64,
        owf: f32,
        ohf: f32,
        pointer_pos: Option<egui::Pos2>,
        out_to_pts: &impl Fn(egui::Vec2) -> egui::Pos2,
        pts_to_out: &impl Fn(egui::Pos2) -> egui::Vec2,
        rect: egui::Rect,
        ink: &mut ink::InkShapes,
        deferred: &mut Vec<Box<dyn FnOnce(&egui::Painter)>>,
    ) {
        // The brush cursor for a selected Paint node: the disc that is
        // actually about to land, at the size and softness it will land
        // at, drawn through the same downstream chain as the strokes (so
        // it squashes with a transform below the node, exactly as the
        // paint will).
        if let Some(pid) = self.selected.iter().copied().find(|id| {
            matches!(self.egraph.node(*id).map(|n| &n.kind), Some(NodeKind::Paint(_)))
        }) {
            let hd = downstream_proj(
                &self.egraph,
                pid,
                &|id| self.node_res_f32(id).unwrap_or((owf, ohf)),
                cur_t,
            );
            let hd_inv = engine::codegen::mat3_inv(&hd);
            let to_norm = |p: egui::Pos2| {
                let q = apply_h(&hd_inv, pts_to_out(p));
                [q.x / owf, q.y / ohf]
            };
            let to_scr = |q: [f32; 2]| {
                out_to_pts(apply_h(&hd, egui::vec2(q[0] * owf, q[1] * ohf)))
            };
            if let Some(p) = pointer_pos.filter(|p| rect.contains(*p)) {
                use engine::graph::BrushMode;
                // The radius is in FULL-res px; `owf` is the proxied
                // canvas, so scale back or the ring doubles in proxy.
                let r = (self.brush.radius * self.proxy.min(1.0) * rect.width() / owf).max(1.5);
                let clone_mode = self.brush.mode == BrushMode::Clone;
                // Clone armed with nowhere to clone FROM paints nothing.
                // The ring goes faint rather than letting the brush look
                // ready and then quietly do nothing.
                let inert = clone_mode
                    && self.brush.clone_off.is_none()
                    && self.brush.clone_src.is_none();
                // The outline is the brush's unit disc through the same
                // squash-and-turn the stamp shader applies, so what is
                // drawn is exactly the stamp about to land.
                let (sn, cs) = self.brush.angle.to_radians().sin_cos();
                let asp = self.brush.aspect.clamp(0.02, 1.0);
                let ellipse = |ink: &mut ink::InkShapes,
                               c: egui::Pos2,
                               r: f32,
                               w: f32,
                               alpha: f32| {
                    const N: usize = 40;
                    let at = |k: usize| {
                        let t = (k % N) as f32 / N as f32 * std::f32::consts::TAU;
                        let (x, y) = (t.cos() * r, t.sin() * r * asp);
                        c + egui::vec2(cs * x - sn * y, sn * x + cs * y)
                    };
                    for k in 0..N {
                        ink.seg(at(k), at(k + 1), w, alpha, false);
                    }
                };
                ellipse(ink, p, r, 1.2, if inert { 0.3 } else { 0.85 });
                // Inner ring = where the falloff starts. The two meet
                // when the brush is hard, so a hard brush reads as one.
                let inner = r * (1.0 - self.brush.soft).clamp(0.0, 1.0);
                if inner > 2.0 && self.brush.soft > 0.02 {
                    ellipse(ink, p, inner, 0.8, 0.4);
                }
                // A turned ellipse shows its major axis, or a thin brush
                // at 45 degrees is just a slanted blob with no
                // handedness.
                if asp < 0.98 {
                    let ax = egui::vec2(cs, sn) * r;
                    ink.seg(p - ax, p + ax, 0.6, 0.3, false);
                }
                if clone_mode {
                    // Where the pixels come FROM: the anchor until the
                    // offset is fixed, then the live source under the
                    // brush.
                    let src = match (self.brush.clone_off, self.brush.clone_src) {
                        (Some(o), _) => {
                            let n = to_norm(p);
                            Some(to_scr([n[0] + o[0], n[1] + o[1]]))
                        }
                        (None, Some(a)) => Some(to_scr(a)),
                        _ => None,
                    };
                    if let Some(sp) = src {
                        ellipse(ink, sp, r, 0.9, 0.5);
                        ink.seg(sp, p, 0.8, 0.35, false);
                        // A cross marks it a source, not a second brush.
                        let k = (r * 0.35).clamp(4.0, 14.0);
                        ink.seg(
                            sp - egui::vec2(k, 0.0),
                            sp + egui::vec2(k, 0.0),
                            0.9,
                            0.6,
                            false,
                        );
                        ink.seg(
                            sp - egui::vec2(0.0, k),
                            sp + egui::vec2(0.0, k),
                            0.9,
                            0.6,
                            false,
                        );
                    }
                }
            }
            // KEY LEGEND, in the corner while a Paint node is selected,
            // with the line for whatever is held LIT — which is also the
            // cheap way to see that a held key is being heard at all.
            // ASCII and the middle dot only: the default font has no
            // arrows, and a missing glyph draws as a box.
            {
                let typing = ui.ctx().egui_wants_keyboard_input();
                let (down, m) =
                    ui.input(|i| (keymap::down(i, keymap::Act::BrushAdjust), i.modifiers));
                let held = down && !typing;
                let mut lines: Vec<(&str, bool)> = vec![
                    ("[ ]   size  ·  shift: softness  ·  alt: opacity", false),
                    ("hold F + move    x: size    y: softness", held && !m.shift && !m.ctrl),
                    ("shift + F        x: opacity    y: angle", held && m.shift),
                    ("ctrl + F         x: aspect", held && m.ctrl),
                ];
                if self.brush.mode == engine::graph::BrushMode::Clone {
                    lines.push(("alt-click        set the clone source", m.alt));
                }
                let (lh, pad, w) = (14.0, 8.0, 296.0);
                let h = lh * lines.len() as f32 + pad * 2.0;
                let origin = egui::pos2(avail.min.x + 12.0, avail.max.y - 12.0 - h);
                ink.frost(
                    egui::Rect::from_min_size(origin, egui::vec2(w, h)),
                    6.0,
                    self.prefs.frost_opacity.unwrap_or(0.96),
                );
                for (k, (text, lit)) in lines.into_iter().enumerate() {
                    let p = origin + egui::vec2(pad, pad + lh * k as f32);
                    let col = if lit {
                        egui::Color32::WHITE
                    } else {
                        egui::Color32::from_gray(0xb4)
                    };
                    let text = text.to_string();
                    deferred.push(Box::new(move |pt: &egui::Painter| {
                        pt.text(
                            p,
                            egui::Align2::LEFT_TOP,
                            text,
                            egui::FontId::proportional(11.0),
                            col,
                        );
                    }));
                }
            }
        }
    }
}
