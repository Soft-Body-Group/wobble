//! The floating channel HUD sliders (ctrl-dragged out of the properties panel).
//! Moved out of `viewer_ui` (lib.rs) unchanged.

use crate::*;

impl Gfx {
    /// The floating channel HUD sliders (ctrl-dragged out of the properties panel).
    pub(crate) fn gizmo_hud_sliders(
        &mut self,
        ui: &mut egui::Ui,
        avail: egui::Rect,
        cur_t: u64,
        patch_driven: &Vec<(u64, anim::Channel)>,
        ink: &mut ink::InkShapes,
        deferred: &mut Vec<Box<dyn FnOnce(&egui::Painter)>>,
    ) {
        // --- floating channel HUD sliders (ctrl-dragged out of the
        // properties panel). Frosted, solid ink, keyed writes. ---
        {
            let autokey = self.autokey;
            let frost_op = self.prefs.frost_opacity.unwrap_or(0.96);
            let mut remove: Option<usize> = None;
            for hidx in 0..self.huds.len() {
                if self.prefs.hud_hide.unwrap_or(false) {
                    break;
                }
                let (node, chans, label, lo, hiv, mut pos, scale, is_rgb, rgb_scale, hgang) = {
                    let h = &self.huds[hidx];
                    (
                        h.node,
                        h.chans.clone(),
                        h.label.clone(),
                        h.lo,
                        h.hi,
                        h.pos,
                        h.scale,
                        h.rgb && h.chans.len() == 3,
                        h.rgb_scale,
                        h.gang,
                    )
                };
                ink.set_tint(self.hud_ink(node));
                // Labels beside a coloured node's marks take its colour too.
                let hud_c32 = self.hud_ink_c32(node);
                let sol = |on: bool| {
                    if on {
                        egui::Color32::WHITE
                    } else {
                        hud_c32.unwrap_or(egui::Color32::from_gray(0xd4))
                    }
                };
                // An rgb HUD is the properties slider in the viewer: the
                // same gang, the same held-key colour axes.
                let mods = rgbslider::Mods::read(ui);
                let ganged = is_rgb && (hgang != mods.gang_invert);
                let hluma = self.egraph.working.primaries.luma_weights();
                let spec = if rgb_scale {
                    rgbslider::RgbSpec::scale(lo, hiv, hluma)
                } else {
                    rgbslider::RgbSpec::offset(lo, hiv, hluma)
                }
                .with_display(engine::color::primaries_matrix(
                    self.egraph.working.primaries,
                    engine::color::Primaries::Rec709,
                ));
                if self.egraph.node(node).is_none() {
                    remove = Some(hidx);
                    continue;
                };
                let Some(ekind) = self.live_kind(node, cur_t) else {
                    remove = Some(hidx);
                    continue;
                };
                let w = 170.0 * scale.x;
                let row_h = 16.0 * scale.y;
                let hgt = 18.0 + chans.len() as f32 * row_h + 4.0;
                pos.x = pos.x.clamp(4.0, (avail.width() - w - 8.0).max(4.0));
                pos.y = pos.y.clamp(4.0, (avail.height() - hgt - 8.0).max(4.0));
                let origin = avail.min + pos;
                ink.frost(
                    egui::Rect::from_min_size(
                        origin - egui::vec2(8.0, 5.0),
                        egui::vec2(w + 16.0, hgt + 10.0),
                    ),
                    7.0,
                    frost_op,
                );
                let header =
                    egui::Rect::from_min_size(origin, egui::vec2(w - 22.0, 14.0));
                let hg = ui.interact(
                    header,
                    ui.id().with(("chud-grip", node, hidx)),
                    egui::Sense::drag(),
                );
                if hg.dragged() && !hg.drag_started() {
                    pos += hg.drag_delta();
                }
                {
                    let title = format!("{label} · {node}");
                    let tp = origin + egui::vec2(2.0, 7.0);
                    let col = sol(hg.hovered());
                    deferred.push(Box::new(move |pt: &egui::Painter| {
                        pt.text(
                            tp,
                            egui::Align2::LEFT_CENTER,
                            title,
                            egui::FontId::proportional(10.0),
                            col,
                        );
                    }));
                }
                if is_rgb {
                    let gr_rect = egui::Rect::from_center_size(
                        egui::pos2(origin.x + w - 24.0, origin.y + 7.0),
                        egui::vec2(14.0, 12.0),
                    );
                    let gr = ui.interact(
                        gr_rect,
                        ui.id().with(("chud-gang", node, hidx)),
                        egui::Sense::click(),
                    );
                    let c = sol(gr.hovered());
                    let on = ganged;
                    deferred.push(Box::new(move |pt: &egui::Painter| {
                        if on {
                            pt.rect_filled(gr_rect, 3.0, c);
                        } else {
                            pt.rect_stroke(
                                gr_rect,
                                3.0,
                                egui::Stroke::new(1.2, c),
                                egui::StrokeKind::Outside,
                            );
                        }
                        pt.text(
                            gr_rect.center(),
                            egui::Align2::CENTER_CENTER,
                            "G",
                            egui::FontId::proportional(9.0),
                            if on { egui::Color32::from_gray(0x18) } else { c },
                        );
                    }));
                    if gr.clicked() {
                        self.huds[hidx].gang = !hgang;
                    }
                    gr.on_hover_text(
                        "gang r/g/b (relative)\nhold G to invert · S sat · D orange↔cyan · F green↔magenta",
                    );
                }
                let xr = egui::Rect::from_center_size(
                    egui::pos2(origin.x + w - 8.0, origin.y + 7.0),
                    egui::vec2(14.0, 14.0),
                );
                let xresp = ui.interact(
                    xr,
                    ui.id().with(("chud-x", node, hidx)),
                    egui::Sense::click(),
                );
                {
                    let c = sol(xresp.hovered());
                    let xc = xr.center();
                    deferred.push(Box::new(move |pt: &egui::Painter| {
                        pt.text(
                            xc,
                            egui::Align2::CENTER_CENTER,
                            "x",
                            egui::FontId::proportional(10.0),
                            c,
                        );
                    }));
                }
                if xresp.clicked() {
                    remove = Some(hidx);
                }
                // Current triple, for the ganged/axis maths and the band.
                let cur: [f32; 3] = if is_rgb {
                    std::array::from_fn(|i| {
                        chans.get(i).and_then(|c| anim::get(&ekind, *c)).unwrap_or(0.0)
                    })
                } else {
                    [0.0; 3]
                };
                let hud_sw = if is_rgb { 16.0 } else { 0.0 };
                let (lx0, lx1) = (origin.x + 2.0, origin.x + w - 34.0 - hud_sw);
                let lanes_rect = egui::Rect::from_min_max(
                    egui::pos2(lx0 - 6.0, origin.y + 18.0),
                    egui::pos2(lx1 + 6.0, origin.y + 18.0 + 3.0 * row_h),
                );
                let armed = mods.axis(0, hgang);
                let show_axis =
                    is_rgb && armed.is_colour() && ui.rect_contains_pointer(lanes_rect);
                let lane_tw = (lx1 - lx0).max(1.0);
                if is_rgb {
                    let chip = egui::Rect::from_min_max(
                        egui::pos2(origin.x + w - hud_sw - 2.0, origin.y + 19.0),
                        egui::pos2(origin.x + w - 2.0, origin.y + 17.0 + 3.0 * row_h),
                    );
                    let c = rgbslider::value_swatch(cur, spec);
                    deferred.push(Box::new(move |pt: &egui::Painter| {
                        pt.rect_filled(chip, 3.0, c);
                    }));
                }
                let mut knob_x = [lx0; 3];
                for (li, ch) in chans.iter().enumerate() {
                    let y = origin.y + 18.0 + (li as f32 + 0.5) * row_h;
                    let (x0, x1) = (lx0, lx1);
                    {
                        let (a2, b2) = (egui::pos2(x0, y), egui::pos2(x1, y));
                        if show_axis {
                            deferred.push(Box::new(move |pt: &egui::Painter| {
                                rgbslider::paint_axis_line(
                                    pt, a2, b2, 4.0, cur, armed, lane_tw, spec,
                                );
                            }));
                        } else {
                            let c = if is_rgb {
                                sol(false)
                                    .lerp_to_gamma(rgbslider::LANE_TINT[li.min(2)], 0.55)
                                    .gamma_multiply(0.8)
                            } else {
                                sol(false).gamma_multiply(0.7)
                            };
                            deferred.push(Box::new(move |pt: &egui::Painter| {
                                pt.line_segment([a2, b2], egui::Stroke::new(1.2, c));
                            }));
                        }
                    }
                    let v = anim::get(&ekind, *ch).unwrap_or(0.0);
                    let tpos = ((v - lo) / (hiv - lo).max(1e-6)).clamp(0.0, 1.0);
                    let hx = x0 + tpos * (x1 - x0);
                    if li < 3 {
                        knob_x[li] = hx;
                    }
                    let hr = ui.interact(
                        egui::Rect::from_center_size(
                            egui::pos2(hx, y),
                            egui::vec2(14.0, 13.0),
                        ),
                        ui.id().with(("chud-s", node, hidx, li)),
                        egui::Sense::drag(),
                    );
                    let driven = patch_driven.contains(&(node, *ch));
                    {
                        let kp = egui::pos2(hx, y);
                        let c = sol(hr.hovered());
                        deferred.push(Box::new(move |pt: &egui::Painter| {
                            // Driven knob: a ring, not a dot — the cable
                            // owns it, dragging does nothing.
                            if driven {
                                pt.circle_stroke(kp, 4.0, egui::Stroke::new(1.4, c));
                            } else {
                                pt.circle_filled(kp, 4.0, c);
                            }
                        }));
                        let vt = egui::pos2(x1 + 4.0, y);
                        let txt = format!("{v:.2}");
                        let c2 = sol(false).gamma_multiply(0.85);
                        deferred.push(Box::new(move |pt: &egui::Painter| {
                            pt.text(
                                vt,
                                egui::Align2::LEFT_CENTER,
                                txt,
                                egui::FontId::proportional(9.0),
                                c2,
                            );
                        }));
                    }
                    if driven && hr.hovered() {
                        hr.clone().on_hover_text("driven by patch cable");
                    }
                    if !driven && hr.dragged() {
                        rgbslider::claim_keyboard(ui.ctx(), true);
                    }
                    if !driven && hr.dragged() && !hr.drag_started() {
                        if is_rgb {
                            let axis = mods.axis(li, hgang);
                            let dv = rgbslider::drag_units_rated(
                                ui,
                                axis,
                                hr.drag_delta().x,
                                (x1 - x0).max(1.0),
                                spec,
                            );
                            let nv3 = rgbslider::nudge(cur, axis, dv, spec);
                            write_hud_triple(
                                &mut self.egraph,
                                node,
                                &chans,
                                nv3,
                                cur_t,
                                autokey,
                                &patch_driven,
                            );
                        } else {
                            // Scalar HUDs ride the z/x/c/v rate keys
                            // like every other draggable value.
                            let dv = hr.drag_delta().x / (x1 - x0).max(1.0)
                                * (hiv - lo)
                                * rgbslider::rate(ui).unwrap_or(1.0);
                            let nv = (v + dv).clamp(lo, hiv);
                            if let Some(nm) = self.egraph.node_mut(node) {
                                let editor::ENode { kind, anim, .. } = nm;
                                let animated = anim
                                    .get(ch)
                                    .is_some_and(|c| !c.keys.is_empty());
                                if animated || autokey {
                                    anim.entry(*ch).or_default().set_key(cur_t, nv);
                                } else {
                                    anim::set(kind, *ch, nv);
                                }
                            }
                        }
                    }
                }
                // Ganged: the box that hugs the three knobs, exactly as
                // in the properties panel and the Grade HUD. Registered
                // after the knobs, so a grab inside it gangs.
                if is_rgb && ganged {
                    let (kmin, kmax) = knob_x
                        .iter()
                        .fold((f32::MAX, f32::MIN), |(a, b), &x| (a.min(x), b.max(x)));
                    let brect = egui::Rect::from_min_max(
                        egui::pos2(kmin - 7.0, origin.y + 19.0),
                        egui::pos2(kmax + 7.0, origin.y + 17.0 + 3.0 * row_h),
                    );
                    let br = ui.interact(
                        brect,
                        ui.id().with(("chud-gangbox", node, hidx)),
                        egui::Sense::drag(),
                    );
                    let c = sol(br.hovered()).gamma_multiply(0.9);
                    deferred.push(Box::new(move |pt: &egui::Painter| {
                        pt.rect_stroke(
                            brect,
                            3.0,
                            egui::Stroke::new(1.3, c),
                            egui::StrokeKind::Outside,
                        );
                    }));
                    if br.dragged() {
                        rgbslider::claim_keyboard(ui.ctx(), true);
                    }
                    if br.dragged() && !br.drag_started() {
                        let lane = br
                            .interact_pointer_pos()
                            .map(|p| {
                                (((p.y - origin.y - 18.0) / row_h) as i32).clamp(0, 2)
                                    as usize
                            })
                            .unwrap_or(0);
                        let axis = mods.axis(lane, hgang);
                        let dv = rgbslider::drag_units_rated(
                            ui,
                            axis,
                            br.drag_delta().x,
                            (lx1 - lx0).max(1.0),
                            spec,
                        );
                        let nv3 = rgbslider::nudge(cur, axis, dv, spec);
                        write_hud_triple(
                            &mut self.egraph,
                            node,
                            &chans,
                            nv3,
                            cur_t,
                            autokey,
                            &patch_driven,
                        );
                    }
                }

                // Corner resize (alt = uniform), same idiom as the
                // grade HUD — every HUD element is resizable.
                let rh = egui::pos2(origin.x + w, origin.y + hgt);
                let rr = ui.interact(
                    egui::Rect::from_center_size(rh, egui::vec2(16.0, 16.0)),
                    ui.id().with(("chud-size", node, hidx)),
                    egui::Sense::drag(),
                );
                {
                    let c = sol(rr.hovered());
                    deferred.push(Box::new(move |pt: &egui::Painter| {
                        let st = egui::Stroke::new(1.4, c);
                        pt.line_segment(
                            [rh + egui::vec2(-8.0, 0.0), rh + egui::vec2(0.0, -8.0)],
                            st,
                        );
                        pt.line_segment(
                            [rh + egui::vec2(-3.5, 0.0), rh + egui::vec2(0.0, -3.5)],
                            st,
                        );
                    }));
                }
                if rr.dragged() && !rr.drag_started() {
                    let d = rr.drag_delta();
                    let mut sc = scale;
                    if ui.input(|i| i.modifiers.alt) {
                        let f = 1.0 + (d.x + d.y) / 220.0;
                        sc.x *= f;
                        sc.y *= f;
                    } else {
                        sc.x += d.x / 170.0;
                        sc.y += d.y / (chans.len() as f32 * 16.0).max(16.0);
                    }
                    sc.x = sc.x.clamp(0.5, 4.0);
                    sc.y = sc.y.clamp(0.6, 3.0);
                    self.huds[hidx].scale = sc;
                }
                self.huds[hidx].pos = pos;
            }
            ink.set_tint(None);
            if let Some(i) = remove {
                self.huds.remove(i);
            }
        }
    }
}
