//! The viewer HUDs of the selected nodes: Grade's wheels and every node's pinned sliders.
//! Moved out of `viewer_ui` (lib.rs) unchanged.

use crate::*;

impl Gfx {
    /// The viewer HUDs of the selected nodes: Grade's wheels and every node's pinned sliders.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gizmo_huds(
        &mut self,
        ui: &mut egui::Ui,
        avail: egui::Rect,
        cur_t: u64,
        selected: Vec<u64>,
        patch_driven: &Vec<(u64, anim::Channel)>,
        ink: &mut ink::InkShapes,
        deferred: &mut Vec<Box<dyn FnOnce(&egui::Painter)>>,
    ) {
    // Grade HUDs: monochrome vector controls — pure black over bright
        // image regions, pure white over dark ones (a tiny GPU luma probe
        // under each HUD decides).
        let mut hud_index = 0;
        // The colour axes preserve the working space's luma, and the
        // preview swatch has to leave it to reach the screen.
        let hud_luma = self.egraph.working.primaries.luma_weights();
        let hud_display = engine::color::primaries_matrix(
            self.egraph.working.primaries,
            engine::color::Primaries::Rec709,
        );
        for node in selected {
            ink.set_tint(self.hud_ink(node));
            let Some(ekind) = self.live_kind(node, cur_t) else {
                continue;
            };
            let NodeKind::Grade(mut gw) = ekind else {
                continue;
            };
            let before_g = gw.clone();
            let g = &mut gw;
            if self.hud.hidden.contains(&node) {
                continue;
            }
            let default_off = egui::vec2(16.0, 16.0 + hud_index as f32 * 150.0);
            hud_index += 1;
            let scale = *self.hud.scale.entry(node).or_insert(egui::vec2(1.0, 1.0));
            // Same gesture vocabulary as the properties rgb slider: G
            // inverts the gang while held, S/D/F arm the colour axes.
            let mods = rgbslider::Mods::read(ui);
            let gang = g.gang != mods.gang_invert;
            let w = 200.0 * scale.x;
            let header_h = 22.0;
            // Three rgb lanes per adjustment, grouped tight with clear
            // air between adjustments (gang never collapses them).
            let lane_h = 13.0 * scale.y;
            let row_gap = 8.0 * scale.y;
            let row_h = 3.0 * lane_h + row_gap;
            let h = header_h + 4.0 * row_h - row_gap + 10.0;
            // Window-locked: the offset is relative to the viewer panel, not
            // the image, so the HUD holds its screen position while the
            // canvas pans/zooms under it. Clamped to stay reachable.
            let off = self.hud.pos.entry(node).or_insert(default_off);
            off.x = off.x.clamp(4.0, (avail.width() - w - 8.0).max(4.0));
            off.y = off.y.clamp(4.0, (avail.height() - h - 8.0).max(4.0));
            let origin = avail.min + *off;
            // Frosted panel behind the whole HUD (old-iOS heavy blur),
            // per-HUD toggle in the header.
            let frosted = !self.hud.no_frost.contains(&node);
            if frosted {
                ink.frost(
                    egui::Rect::from_min_size(
                        origin - egui::vec2(10.0, 6.0),
                        egui::vec2(w + 20.0, h + 12.0),
                    ),
                    8.0,
                    self.prefs.frost_opacity.unwrap_or(0.96),
                );
            }
            // Over frost the background is normalized, so controls use
            // one solid tone instead of the per-pixel adaptive ink.
            // A node that chose a HUD colour gets its labels in it too;
            // hover still snaps white.
            let hud_c32 = self.hud_ink_c32(node);
            let sol = |on: bool| {
                if on {
                    egui::Color32::WHITE
                } else {
                    hud_c32.unwrap_or(egui::Color32::from_gray(0xd4))
                }
            };
            // Layout stretches with hud_scale; fonts/handles stay
            // constant (like node labels under editor zoom).
            let fsz = 1.0f32;
            let dark_bg = self.hud.luma.get(&node).copied().unwrap_or(0.0) <= 0.45;
            // Frosted: fixed tones (the panel normalizes the bg).
            // Otherwise match the ink shader's luma-adaptive greys.
            let (mono, anti) = if frosted {
                (egui::Color32::from_gray(0xd4), egui::Color32::from_gray(0x22))
            } else if dark_bg {
                (egui::Color32::from_gray(0xc7), egui::Color32::from_gray(0x1f))
            } else {
                (egui::Color32::from_gray(0x1f), egui::Color32::from_gray(0xc7))
            };
            let mono = hud_c32.unwrap_or(mono);
            let label = |d: &mut Vec<Box<dyn FnOnce(&egui::Painter)>>,
                         p: egui::Pos2,
                         anchor: egui::Align2,
                         s: String,
                         size: f32,
                         col: egui::Color32| {
                let size = size * fsz;
                d.push(Box::new(move |pt: &egui::Painter| {
                    pt.text(p, anchor, s, egui::FontId::proportional(size), col);
                }));
            };

            // Grip: two short lines (the trigram glyph is tofu).
            for dy in [-2.0f32, 2.0] {
                let seg = [
                    origin + egui::vec2(0.0, 11.0 + dy),
                    origin + egui::vec2(10.0 * fsz.max(1.0), 11.0 + dy),
                ];
                let st = egui::Stroke::new(1.2, mono.gamma_multiply(0.7));
                deferred.push(Box::new(move |pt: &egui::Painter| {
                    pt.line_segment(seg, st);
                }));
            }
            label(
                deferred,
                origin + egui::vec2(16.0 * fsz.max(1.0), 11.0),
                egui::Align2::LEFT_CENTER,
                format!("grade {node}"),
                11.0,
                mono,
            );
            // Frost toggle sits left of the gang button.
            let frect = egui::Rect::from_center_size(
                egui::pos2(origin.x + w - 56.0, origin.y + 11.0),
                egui::vec2(16.0, 14.0),
            );
            let f_hover = ui.rect_contains_pointer(frect.expand(2.0));
            if frosted {
                let c = sol(f_hover);
                deferred.push(Box::new(move |pt: &egui::Painter| {
                    pt.rect_filled(frect, 3.0, c);
                }));
            } else {
                ink.rect(frect, 3.0, false, 1.2, 1.0, f_hover);
            }
            label(
                deferred,
                frect.center(),
                egui::Align2::CENTER_CENTER,
                "~".into(),
                10.0,
                if frosted { anti } else { mono },
            );
            if ui
                .interact(
                    frect.expand(2.0),
                    ui.id().with(("hud-frost", node)),
                    egui::Sense::click(),
                )
                .on_hover_text("frosted background")
                .clicked()
            {
                if frosted {
                    self.hud.no_frost.insert(node);
                } else {
                    self.hud.no_frost.remove(&node);
                }
            }
            let header = egui::Rect::from_min_size(origin, egui::vec2(w - 78.0, header_h));
            let hg = ui.interact(header, ui.id().with(("hud-grip", node)), egui::Sense::drag());
            if hg.dragged() && !hg.drag_started() {
                *self.hud.pos.get_mut(&node).unwrap() += hg.drag_delta();
            }

            let grect = egui::Rect::from_center_size(
                egui::pos2(origin.x + w - 34.0, origin.y + 11.0),
                egui::vec2(16.0, 14.0),
            );
            let gang_hover = ui.rect_contains_pointer(grect.expand(2.0));
            if frosted {
                let c = sol(gang_hover);
                deferred.push(Box::new(move |pt: &egui::Painter| {
                    if gang {
                        pt.rect_filled(grect, 3.0, c);
                    } else {
                        pt.rect_stroke(
                            grect,
                            3.0,
                            egui::Stroke::new(1.2, c),
                            egui::StrokeKind::Outside,
                        );
                    }
                }));
            } else {
                ink.rect(grect, 3.0, gang, 1.2, 1.0, gang_hover);
            }
            label(
                deferred,
                grect.center(),
                egui::Align2::CENTER_CENTER,
                "G".into(),
                10.0,
                if gang { anti } else { mono },
            );
            if ui
                .interact(grect.expand(2.0), ui.id().with(("hud-gang", node)), egui::Sense::click())
                .on_hover_text("gang r/g/b")
                .clicked()
            {
                g.gang = !g.gang;
            }

            let hide_r = egui::Rect::from_center_size(
                egui::pos2(origin.x + w - 12.0, origin.y + 11.0),
                egui::vec2(14.0, 14.0),
            );
            label(
                deferred,
                hide_r.center(),
                egui::Align2::CENTER_CENTER,
                "x".into(),
                11.0,
                mono,
            );
            if ui
                .interact(hide_r, ui.id().with(("hud-hide", node)), egui::Sense::click())
                .clicked()
            {
                self.hud.hidden.insert(node);
            }

            // Lift and offset live around zero (additive); gamma and
            // gain around one (ratios) — which is exactly the split the
            // rgb slider's colour axes need to move them correctly.
            for (ri, (name, lo, hi, mode)) in [
                ("lift", -1.0f32, 1.0f32, rgbslider::RgbMode::Offset),
                ("gamma", 0.05, 4.0, rgbslider::RgbMode::Scale),
                ("gain", 0.0, 4.0, rgbslider::RgbMode::Scale),
                ("offset", -1.0, 1.0, rgbslider::RgbMode::Offset),
            ]
            .iter()
            .enumerate()
            {
                let spec = match mode {
                    rgbslider::RgbMode::Scale => rgbslider::RgbSpec::scale(*lo, *hi, hud_luma),
                    rgbslider::RgbMode::Offset => {
                        rgbslider::RgbSpec::offset(*lo, *hi, hud_luma)
                    }
                }
                .with_display(hud_display);
                let v: &mut [f32; 3] = match ri {
                    0 => &mut g.lift,
                    1 => &mut g.gamma,
                    2 => &mut g.gain,
                    _ => &mut g.offset,
                };
                let y0 = origin.y + header_h + ri as f32 * row_h;
                label(
                    deferred,
                    egui::pos2(origin.x, y0 + 1.5 * lane_h),
                    egui::Align2::LEFT_CENTER,
                    name.to_string(),
                    10.0,
                    mono,
                );
                // The row's colour chip sits at the right end, the same
                // instrument as the properties slider.
                let sw_w = rgbslider::SWATCH_W.min(w * 0.15);
                let (x0, x1) = (
                    origin.x + 52.0 * fsz.max(1.0),
                    origin.x + w - 14.0 - sw_w - 4.0,
                );
                let track_w = (x1 - x0).max(8.0);
                {
                    let chip = egui::Rect::from_min_max(
                        egui::pos2(x1 + 6.0, y0 + 1.0),
                        egui::pos2(x1 + 6.0 + sw_w, y0 + 3.0 * lane_h - 1.0),
                    );
                    let c = rgbslider::value_swatch(*v, spec);
                    deferred.push(Box::new(move |pt: &egui::Painter| {
                        pt.rect_filled(chip, 3.0, c);
                    }));
                }
                let mut knob_x = [x0; 3];
                // The colour-op preview, on the row under the hand: what
                // the armed axis does to mid grey across a full drag.
                let row_rect = egui::Rect::from_min_max(
                    egui::pos2(x0 - 7.0, y0),
                    egui::pos2(x1 + 7.0, y0 + 3.0 * lane_h),
                );
                let armed = mods.axis(0, g.gang);
                // On the LINES, not over the row: a HUD sits on the shot,
                // and a filled block hides both the knobs and the picture.
                let show_axis = armed.is_colour() && ui.rect_contains_pointer(row_rect);
                let vv = *v;
                for lane in 0..3 {
                    let y = y0 + (lane as f32 + 0.5) * lane_h;
                    let tint = [
                        egui::Color32::from_rgb(0xc9, 0x5c, 0x55),
                        egui::Color32::from_rgb(0x63, 0xb3, 0x62),
                        egui::Color32::from_rgb(0x5f, 0x83, 0xd4),
                    ][lane];
                    if show_axis {
                        let (a2, b2) = (egui::pos2(x0, y), egui::pos2(x1, y));
                        deferred.push(Box::new(move |pt: &egui::Painter| {
                            rgbslider::paint_axis_line(
                                pt, a2, b2, 4.0, vv, armed, track_w, spec,
                            );
                        }));
                    } else if frosted {
                        let c = sol(false).lerp_to_gamma(tint, 0.55).gamma_multiply(0.75);
                        let (a2, b2) = (egui::pos2(x0, y), egui::pos2(x1, y));
                        deferred.push(Box::new(move |pt: &egui::Painter| {
                            pt.line_segment([a2, b2], egui::Stroke::new(1.2, c));
                        }));
                    } else {
                        // Tinted per channel in the ink (luma-adaptive).
                        ink.seg_lane(
                            egui::pos2(x0, y),
                            egui::pos2(x1, y),
                            1.2,
                            0.75,
                            false,
                            lane,
                        );
                    }
                    label(
                        deferred,
                        egui::pos2(x0 - 8.0 * fsz, y),
                        egui::Align2::CENTER_CENTER,
                        ["r", "g", "b"][lane].to_string(),
                        9.0,
                        mono.lerp_to_gamma(tint, 0.55),
                    );
                    let tpos = ((v[lane] - lo) / (hi - lo)).clamp(0.0, 1.0);
                    let hx = x0 + tpos * track_w;
                    knob_x[lane] = hx;
                    let hr = ui.interact(
                        egui::Rect::from_center_size(
                            egui::pos2(hx, y),
                            egui::vec2(14.0, 12.0) * fsz.max(1.0),
                        ),
                        ui.id().with(("hud-s", node, ri, lane)),
                        egui::Sense::drag(),
                    );
                    let hrad = 4.5 * fsz.max(1.0);
                    if frosted {
                        let c = if hr.hovered() {
                            egui::Color32::WHITE
                        } else {
                            sol(false).lerp_to_gamma(tint, 0.6)
                        };
                        let kp = egui::pos2(hx, y);
                        deferred.push(Box::new(move |pt: &egui::Painter| {
                            pt.circle_filled(kp, hrad, c);
                        }));
                    } else {
                        ink.circle_fill_lane(egui::pos2(hx, y), hrad, 1.0, hr.hovered(), lane);
                    }
                    if hr.dragged() && !hr.drag_started() {
                        let axis = mods.axis(lane, g.gang);
                        let dv = rgbslider::drag_units_rated(
                            ui,
                            axis,
                            hr.drag_delta().x,
                            track_w,
                            spec,
                        );
                        *v = rgbslider::nudge(*v, axis, dv, spec);
                    }
                    if hr.dragged() {
                        // The colour axes are held letters; the app's
                        // hotkeys have to stand down for the gesture.
                        rgbslider::claim_keyboard(ui.ctx(), true);
                    }
                }
                // Ganged: a box hugs the three knobs wherever they sit;
                // sliding it moves all three together (relative). It sits
                // on top of the knobs, so any grab inside it gangs.
                if gang {
                    let (kmin, kmax) = knob_x
                        .iter()
                        .fold((f32::MAX, f32::MIN), |(a, b), &x| (a.min(x), b.max(x)));
                    let brect = egui::Rect::from_min_max(
                        egui::pos2(kmin - 9.0, y0 + 1.0),
                        egui::pos2(kmax + 9.0, y0 + 3.0 * lane_h - 1.0),
                    );
                    let br = ui.interact(
                        brect,
                        ui.id().with(("hud-gangbox", node, ri)),
                        egui::Sense::drag(),
                    );
                    if frosted {
                        let c = sol(br.hovered()).gamma_multiply(0.9);
                        deferred.push(Box::new(move |pt: &egui::Painter| {
                            pt.rect_stroke(
                                brect,
                                3.0,
                                egui::Stroke::new(1.3, c),
                                egui::StrokeKind::Outside,
                            );
                        }));
                    } else {
                        ink.rect(brect, 3.0, false, 1.3, 0.9, br.hovered());
                    }
                    if br.dragged() && !br.drag_started() {
                        // Hold G mid-drag and the grab peels off to the
                        // lane under the hand, not always red.
                        let lane = br
                            .interact_pointer_pos()
                            .map(|p| {
                                (((p.y - y0) / lane_h) as i32).clamp(0, 2) as usize
                            })
                            .unwrap_or(0);
                        let axis = mods.axis(lane, g.gang);
                        let dv = rgbslider::drag_units_rated(
                            ui,
                            axis,
                            br.drag_delta().x,
                            track_w,
                            spec,
                        );
                        *v = rgbslider::nudge(*v, axis, dv, spec);
                    }
                    if br.dragged() {
                        rgbslider::claim_keyboard(ui.ctx(), true);
                    }
                }
            }

            // Corner resize: scales the controls per axis.
            let rh = egui::pos2(origin.x + w, origin.y + h);
            let rr = ui.interact(
                egui::Rect::from_center_size(rh, egui::vec2(18.0, 18.0)),
                ui.id().with(("hud-size", node)),
                egui::Sense::drag(),
            );
            if frosted {
                let c = sol(rr.hovered());
                deferred.push(Box::new(move |pt: &egui::Painter| {
                    let st = egui::Stroke::new(1.5, c);
                    pt.line_segment([rh + egui::vec2(-9.0, 0.0), rh + egui::vec2(0.0, -9.0)], st);
                    pt.line_segment([rh + egui::vec2(-4.0, 0.0), rh + egui::vec2(0.0, -4.0)], st);
                }));
            } else {
                ink.seg(rh + egui::vec2(-9.0, 0.0), rh + egui::vec2(0.0, -9.0), 1.5, 1.0, rr.hovered());
                ink.seg(rh + egui::vec2(-4.0, 0.0), rh + egui::vec2(0.0, -4.0), 1.5, 1.0, rr.hovered());
            }
            if rr.dragged() && !rr.drag_started() {
                let sc = self.hud.scale.get_mut(&node).unwrap();
                let d = rr.drag_delta();
                if ui.input(|i| i.modifiers.alt) {
                    // Alt: uniform.
                    let f = 1.0 + (d.x + d.y) / 260.0;
                    sc.x = (sc.x * f).max(0.05);
                    sc.y = (sc.y * f).max(0.05);
                } else {
                    // Free per-axis stretch; proportions unbounded.
                    sc.x = (sc.x + d.x / 200.0).max(0.05);
                    sc.y = (sc.y + d.y / 70.0).max(0.05);
                }
            }
            if gw != before_g {
                if let Some(n) = self.egraph.node_mut(node) {
                    let editor::ENode { kind, anim, .. } = n;
                    if let NodeKind::Grade(auth) = kind {
                        auth.gang = gw.gang;
                    }
                    keyed_write_back(
                        anim,
                        kind,
                        &NodeKind::Grade(before_g),
                        &NodeKind::Grade(gw),
                        cur_t,
                        anim::GRADE,
                        self.autokey,
                        &patch_driven,
                        node,
                    );
                }
            }
        }
    }
}
