//! CornerPin: the four pins and their grab gestures.
//! Moved out of `viewer_ui` (lib.rs) unchanged.

use crate::*;

impl Gfx {
    /// CornerPin: the four pins and their grab gestures.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn gizmo_corner_pin(
        &mut self,
        ui: &mut egui::Ui,
        cmd: bool,
        scrub_held: bool,
        cur_t: u64,
        owf: f32,
        ohf: f32,
        pointer_pos: Option<egui::Pos2>,
        out_to_pts: &impl Fn(egui::Vec2) -> egui::Pos2,
        pts_to_out: &impl Fn(egui::Pos2) -> egui::Vec2,
        rect: egui::Rect,
        selected: &Vec<u64>,
        patch_driven: &Vec<(u64, anim::Channel)>,
        nav: &egui::Response,
        ink: &mut ink::InkShapes,
        hud_key_list: &mut Vec<hudkeys::HudKey>,
    ) {
        for node in selected.iter().copied() {
            ink.set_tint(self.hud_ink(node));
            // Affine space = SHADER space, which now anchors every
            // spatial node to ITS OWN format (stamped per-node), so the
            // deformation chain resolves each node's local resolution.
            let hd = downstream_proj(
                &self.egraph,
                node,
                &|id| self.node_res_f32(id).unwrap_or((owf, ohf)),
                cur_t,
            );
            let hd_inv = engine::codegen::mat3_inv(&hd);
            let Some(ekind) = self.live_kind(node, cur_t) else {
                continue;
            };
            let NodeKind::CornerPin(mut cpw) = ekind else {
                continue;
            };
            // Pins are normalized in the NODE's own format (the shader
            // now anchors them there too).
            let (lwf, lhf) = self.node_res_f32(node).unwrap_or((owf, ohf));
            let before_cp = cpw;
            let edit_from = self.pin_from.contains(&node);
            // Under a SoftCrop the pins are the CROP's corners (where
            // they land), and a drag re-solves `to` around them. The
            // `from` quad is a plain quad in the node's space.
            let crop = self.soft_frame_px(node, cur_t).map(|r| {
                [r.min.x / lwf, r.min.y / lhf, r.max.x / lwf, r.max.y / lhf]
            });
            let pin_quad = |cp: &engine::graph::CornerPinParams| match crop {
                Some(c) => nodes::softcrop::pins_on_crop(cp.effective_to(), c),
                None => cp.to,
            };
            let shown = if edit_from { cpw.from } else { pin_quad(&cpw) };
            let set_shown = |cp: &mut engine::graph::CornerPinParams, q: [[f32; 2]; 4]| {
                if edit_from {
                    cp.from = q;
                } else if let Some(c) = crop {
                    cp.to = cp.to_for_effective(nodes::softcrop::to_from_crop_pins(q, c));
                } else {
                    cp.to = q;
                }
            };
            let px = |q: [f32; 2]| egui::vec2(q[0] * lwf, q[1] * lhf);
            let nrm = |v: egui::Vec2| [v.x / lwf, v.y / lhf];
            let scr = |q: [f32; 2]| out_to_pts(apply_h(&hd, px(q)));
            let scr_px = |v: egui::Vec2| out_to_pts(apply_h(&hd, v));
            let from_screen = |p: egui::Pos2| apply_h(&hd_inv, pts_to_out(p));
            let pts: Vec<egui::Pos2> = shown.iter().map(|c| scr(*c)).collect();

            // The quad NOT being edited, faintly: the pin while `from` is
            // armed; `from` while the pin is, unless it is still the frame.
            let other = if edit_from {
                Some(pin_quad(&cpw))
            } else {
                (cpw.from != engine::graph::UNIT_QUAD).then_some(cpw.from)
            };
            if let Some(o) = other {
                for k in 0..4 {
                    ink.seg(scr(o[k]), scr(o[(k + 1) % 4]), 1.0, 0.35, false);
                }
                for c in o {
                    ink.circle_stroke(scr(c), 4.0, 1.0, 0.35, false);
                }
            }
            for k in 0..4 {
                ink.seg(pts[k], pts[(k + 1) % 4], 1.5, 0.8, false);
            }

            let group_ui = self.pin_sel.is_some_and(|(n, _)| n == node);
            let mut pick = self.pin_sel.filter(|(n, _)| *n == node).map_or([false; 4], |(_, s)| s);
            let n_pick = pick.iter().filter(|b| **b).count();
            let grab_live = self.pin_grab.as_ref().is_some_and(|g| g.node == node);
            if group_ui && hud_key_list.is_empty() {
                use hudkeys::{chord, key};
                hud_key_list.push(key("S", "click: add to pick", egui::Key::S));
                if n_pick > 0 {
                    hud_key_list.push(key("E", "scale about click", egui::Key::E));
                    hud_key_list.push(key("R", "rotate: circle the click", egui::Key::R));
                    hud_key_list.push(key("T", "move pick", egui::Key::T));
                }
                if n_pick >= 2 {
                    hud_key_list.push(chord("alt", "drag edge: from centre"));
                }
            }
            // Positions the group tools own this frame: a click there is
            // not a click on the canvas.
            let mut hot: Vec<egui::Pos2> = pts.clone();

            // E / R / T held at the press: a TEMPORARY pivot where the
            // press landed. Taken before the widgets so it wins over
            // whichever one the press happened to land on.
            if group_ui && n_pick > 0 && !grab_live {
                let held = ui.input(|i| {
                    [egui::Key::E, egui::Key::R, egui::Key::T].into_iter().find(|k| i.key_down(*k))
                });
                let origin = ui.input(|i| {
                    i.pointer.primary_pressed().then(|| i.pointer.press_origin()).flatten()
                });
                if let (Some(k), Some(o)) = (held, origin.filter(|o| rect.contains(*o))) {
                    self.pin_grab = Some(PinGrab {
                        node,
                        mode: match k {
                            egui::Key::E => PinMode::KeyScale,
                            egui::Key::R => PinMode::KeyRotate,
                            _ => PinMode::Move,
                        },
                        origin: o,
                        pivot: from_screen(o),
                        start: shown,
                        moving: pick,
                        a0: None,
                        by_key: true,
                        vp: egui::Pos2::ZERO,
                        last: egui::Pos2::ZERO,
                    });
                }
            }
            // Anything already in flight (a held-key edit taken above, a
            // drag from an earlier frame) keeps the widgets' hands off.
            let key_grab = self.pin_grab.as_ref().is_some_and(|g| g.node == node);

            // The box round a pick of two or more: the spline's handle
            // language, axis-aligned in the node's pixels.
            if group_ui && n_pick >= 2 {
                let (mut lo, mut hi) = (egui::Vec2::splat(f32::MAX), egui::Vec2::splat(f32::MIN));
                for k in 0..4 {
                    if pick[k] {
                        lo = lo.min(px(shown[k]));
                        hi = hi.max(px(shown[k]));
                    }
                }
                let pad = (egui::Vec2::splat(2.0) - (hi - lo)).max(egui::Vec2::ZERO) * 0.5;
                let (lo, hi) = (lo - pad, hi + pad);
                let at = |hx: i8, hy: i8| {
                    egui::vec2(
                        lo.x + (hi.x - lo.x) * (hx as f32 + 1.0) * 0.5,
                        lo.y + (hi.y - lo.y) * (hy as f32 + 1.0) * 0.5,
                    )
                };
                let c = at(0, 0);
                let cpts = scr_px(c);
                let corners = [at(-1, -1), at(1, -1), at(1, 1), at(-1, 1)];
                for k in 0..4 {
                    ink.seg(scr_px(corners[k]), scr_px(corners[(k + 1) % 4]), 1.0, 0.45, false);
                }
                // Handles ride 18px OUTSIDE the box, or a corner handle
                // would sit on the pin that defines it.
                // A pin on the frame's edge (the default) puts anything
                // outside the box off the viewer, or under a floating
                // panel over it, where no hand can press it. A handle
                // that would land there comes INSIDE the box instead.
                let safe = self.viewer_rect.intersect(ui.clip_rect()).shrink(12.0);
                let my_layer = ui.layer_id();
                let free = |p: egui::Pos2| {
                    safe.contains(p)
                        && ui.ctx().layer_id_at(p).is_none_or(|l| l == my_layer || l.order == egui::Order::Background)
                };
                let pushed = |hp: egui::Pos2| {
                    let d = hp - cpts;
                    if d.length() < 1e-3 {
                        return hp;
                    }
                    let out = hp + d.normalized() * 18.0;
                    if free(out) { out } else { hp - d.normalized() * 18.0 }
                };
                let about_centre = ui.input(|i| i.modifiers.alt);
                let press = ui.input(|i| i.pointer.press_origin()).or(pointer_pos);
                const BOX_HANDLES: [(i8, i8); 8] =
                    [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)];
                for (k, (hx, hy)) in BOX_HANDLES.iter().enumerate() {
                    let hp = pushed(scr_px(at(*hx, *hy)));
                    hot.push(hp);
                    let rs = ui.interact(
                        egui::Rect::from_center_size(hp, egui::vec2(22.0, 22.0)),
                        ui.id().with(("pin-scale", node, k)),
                        egui::Sense::drag(),
                    );
                    ink.rect(egui::Rect::from_center_size(hp, egui::vec2(11.0, 11.0)), 2.0, true, 0.0, 1.0, rs.hovered());
                    if rs.drag_started() && !key_grab {
                        self.pin_grab = Some(PinGrab {
                            node,
                            mode: PinMode::Scale(*hx != 0, *hy != 0),
                            origin: press.unwrap_or(hp),
                            pivot: if about_centre { c } else { at(-*hx, -*hy) },
                            start: shown,
                            moving: pick,
                            a0: None,
                            by_key: false,
                            vp: egui::Pos2::ZERO,
                            last: egui::Pos2::ZERO,
                        });
                    }
                }
                // Rotate: a knob on a stalk off the top edge — or the
                // bottom one when the top is off the viewer, as it is
                // for pins on the frame's corners.
                let stalk = |edge: egui::Pos2| {
                    let up = edge - cpts;
                    let dir = if up.length() > 1e-3 { up.normalized() } else { egui::vec2(0.0, -1.0) };
                    (edge, edge + dir * 34.0)
                };
                // Last resort: inside the box, a stalk DOWN from the top.
                let (top, knob) = [stalk(scr_px(at(0, -1))), stalk(scr_px(at(0, 1)))]
                    .into_iter()
                    .find(|(_, k)| free(*k))
                    .unwrap_or_else(|| {
                        let (t, k) = stalk(scr_px(at(0, -1)));
                        (t, t - (k - t) * 1.4)
                    });
                self.pin_knob = Some(knob);
                hot.push(knob);
                let rr = ui.interact(
                    egui::Rect::from_center_size(knob, egui::vec2(22.0, 22.0)),
                    ui.id().with(("pin-rot", node)),
                    egui::Sense::drag(),
                );
                ink.seg(top, knob, 1.0, 0.5, false);
                ink.circle_stroke(knob, 6.0, 1.6, 1.0, rr.hovered());
                if rr.drag_started() && !key_grab {
                    self.pin_grab = Some(PinGrab {
                        node,
                        mode: PinMode::Rotate,
                        origin: press.unwrap_or(knob),
                        pivot: c,
                        start: shown,
                        moving: pick,
                        a0: None,
                        by_key: false,
                        vp: egui::Pos2::ZERO,
                        last: egui::Pos2::ZERO,
                    });
                }
                // Translate: the open circle at the centre.
                hot.push(cpts);
                let rt = ui.interact(
                    egui::Rect::from_center_size(cpts, egui::vec2(26.0, 26.0)),
                    ui.id().with(("pin-move", node)),
                    egui::Sense::drag(),
                );
                ink.circle_stroke(cpts, 10.0, 1.6, 0.9, rt.hovered());
                if rt.drag_started() && !key_grab {
                    self.pin_grab = Some(PinGrab {
                        node,
                        mode: PinMode::Move,
                        origin: press.unwrap_or(cpts),
                        pivot: c,
                        start: shown,
                        moving: pick,
                        a0: None,
                        by_key: false,
                        vp: egui::Pos2::ZERO,
                        last: egui::Pos2::ZERO,
                    });
                }
            }

            // The pins, registered last so they win over the box.
            for (i, p0) in pts.iter().enumerate() {
                let rs = ui.interact(
                    egui::Rect::from_center_size(*p0, egui::vec2(22.0, 22.0)),
                    ui.id().with(("g-pin", node, i)),
                    egui::Sense::click_and_drag(),
                );
                let picked = group_ui && pick[i];
                ink.rect(egui::Rect::from_center_size(*p0, egui::vec2(12.0, 12.0)), 2.0, true, 0.0, 1.0, rs.hovered());
                if picked {
                    ink.circle_stroke(*p0, 11.0, 1.2, 0.9, rs.hovered());
                }
                if rs.clicked() && group_ui && !key_grab {
                    // S toggles the pin in the pick; a plain click makes
                    // it the whole pick.
                    if ui.input(|inp| inp.key_down(egui::Key::S)) {
                        pick[i] = !pick[i];
                    } else {
                        pick = [false; 4];
                        pick[i] = true;
                    }
                }
                if rs.drag_started() && !key_grab {
                    // Grabbing a pin outside the pick makes it the pick:
                    // a drag moves what is under the cursor.
                    if !pick[i] {
                        pick = [false; 4];
                        pick[i] = true;
                    }
                    let origin =
                        ui.input(|inp| inp.pointer.press_origin()).or(pointer_pos).unwrap_or(*p0);
                    self.pin_grab = Some(PinGrab {
                        node,
                        mode: PinMode::Move,
                        origin,
                        pivot: px(shown[i]),
                        start: shown,
                        moving: if group_ui { pick } else { std::array::from_fn(|k| k == i) },
                        a0: None,
                        by_key: false,
                        vp: egui::Pos2::ZERO,
                        last: egui::Pos2::ZERO,
                    });
                }
            }

            // The gesture in flight, from its press.
            let rate = widgets::sens_factor(ui).unwrap_or(1.0);
            if let Some(g) = self.pin_grab.as_mut().filter(|g| g.node == node) {
                // A fresh grab starts both at the press, so the travel
                // past egui's drag threshold is not lost.
                if g.vp == egui::Pos2::ZERO && g.last == egui::Pos2::ZERO {
                    (g.vp, g.last) = (g.origin, g.origin);
                }
                let raw_p = pointer_pos.unwrap_or(g.last);
                g.vp += (raw_p - g.last) * rate;
                g.last = raw_p;
                let p = g.vp;
                let (cur, o, pv) = (from_screen(p), from_screen(g.origin), g.pivot);
                let turn = |a: f32, v: egui::Vec2| {
                    let (sn, cs) = a.sin_cos();
                    pv + egui::vec2(cs * v.x - sn * v.y, sn * v.x + cs * v.y)
                };
                let map: Box<dyn Fn(egui::Vec2) -> egui::Vec2> = match g.mode {
                    PinMode::Move => {
                        let d = cur - o;
                        Box::new(move |v| v + d)
                    }
                    PinMode::Scale(ax, ay) => {
                        let den = o - pv;
                        let f = egui::vec2(
                            if ax && den.x.abs() > 0.5 { (cur.x - pv.x) / den.x } else { 1.0 },
                            if ay && den.y.abs() > 0.5 { (cur.y - pv.y) / den.y } else { 1.0 },
                        );
                        Box::new(move |v| pv + (v - pv) * f)
                    }
                    PinMode::Rotate => {
                        let a = (cur - pv).angle() - (o - pv).angle();
                        Box::new(move |v| turn(a, v - pv))
                    }
                    PinMode::KeyScale => {
                        let s = ((p.x - g.origin.x) / 150.0).exp();
                        Box::new(move |v| pv + (v - pv) * s)
                    }
                    PinMode::KeyRotate => {
                        // A dead zone round the pivot, where the angle
                        // means nothing.
                        if (p - g.origin).length() >= 12.0 && g.a0.is_none() {
                            g.a0 = Some((cur - pv).angle());
                        }
                        let a = g.a0.map_or(0.0, |a0| (cur - pv).angle() - a0);
                        Box::new(move |v| turn(a, v - pv))
                    }
                };
                let mut q = g.start;
                for k in 0..4 {
                    if g.moving[k] {
                        q[k] = nrm(map(px(g.start[k])));
                    }
                }
                set_shown(&mut cpw, q);
                if g.by_key {
                    // The temporary pivot, and the throw from it.
                    let ps = scr_px(pv);
                    ink.seg(ps - egui::vec2(7.0, 0.0), ps + egui::vec2(7.0, 0.0), 1.2, 1.0, true);
                    ink.seg(ps - egui::vec2(0.0, 7.0), ps + egui::vec2(0.0, 7.0), 1.2, 1.0, true);
                    ink.seg(g.origin, p, 1.0, 0.5, false);
                }
                if rate != 1.0 {
                    widgets::claim_sens(ui.ctx());
                }
                if !ui.input(|i| i.pointer.primary_down()) {
                    self.pin_grab = None;
                }
            }
            let grab_live = self.pin_grab.as_ref().is_some_and(|g| g.node == node);

            // Marquee: a plain background drag bands pins (S held keeps
            // what was picked). cmd/ctrl-drag is the pan and Tab-drag the
            // scrub, so neither starts one. A plain click on empty canvas
            // drops the pick.
            if group_ui {
                let banding = nav.dragged_by(egui::PointerButton::Primary)
                    && !cmd
                    && !scrub_held
                    && !grab_live;
                if banding {
                    if let (Some(o), Some(p)) = (ui.input(|i| i.pointer.press_origin()), pointer_pos) {
                        self.pin_marquee = Some(egui::Rect::from_two_pos(o, p));
                    }
                }
                if let Some(mq) = self.pin_marquee {
                    ink.rect(mq, 0.0, false, 1.0, 0.5, false);
                    if !banding {
                        if !ui.input(|i| i.key_down(egui::Key::S)) {
                            pick = [false; 4];
                        }
                        for (k, p) in pts.iter().enumerate() {
                            if mq.contains(*p) {
                                pick[k] = true;
                            }
                        }
                        self.pin_marquee = None;
                    }
                }
                if ui.input(|i| i.pointer.primary_clicked()) && !grab_live {
                    if let Some(p) = pointer_pos.filter(|p| rect.contains(*p)) {
                        if !hot.iter().any(|h| (*h - p).length() < 12.0)
                            && !ui.input(|i| i.key_down(egui::Key::S))
                        {
                            pick = [false; 4];
                        }
                    }
                }
                self.pin_sel = Some((node, pick));
            }

            if cpw != before_cp {
                if let Some(n) = self.egraph.node_mut(node) {
                    let editor::ENode { kind, anim, .. } = n;
                    keyed_write_back(
                        anim,
                        kind,
                        &NodeKind::CornerPin(before_cp),
                        &NodeKind::CornerPin(cpw),
                        cur_t,
                        if edit_from { anim::FROMS } else { anim::PINS },
                        self.autokey,
                        &patch_driven,
                        node,
                    );
                }
            }
        }
    }
}
