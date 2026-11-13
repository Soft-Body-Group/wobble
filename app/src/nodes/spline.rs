//! Spline (roto): the shape list, feather, transform and the soft-crop buttons.
//! Moved out of `properties_ui` (lib.rs) unchanged.

use crate::*;

/// Everything the spline panel borrows from the properties pass.
pub(crate) struct Cx<'a> {
    pub full: &'a mut engine::graph::SplineParams,
    pub canvas_res: (u32, u32),
    pub cur_t: u64,
    pub node: u64,
    pub node_px: Option<[f32; 2]>,
    pub spline_sel_node: Option<u64>,
    pub spline_sel: &'a std::collections::BTreeSet<usize>,
    pub roto_click: &'a mut Option<RotoAction>,
    pub softcrop_req: &'a mut Option<(u64, bool)>,
}

/// Spline (roto): the shape list, feather, transform and the soft-crop buttons.
pub(crate) fn panel(
    ui: &mut egui::Ui,
    rcx: &mut widgets::RowCtx,
    cx: Cx,
) -> Option<widgets::HudSlider> {
    let mut hud_drag_out: Option<widgets::HudSlider> = None;
    let Cx {
        full,
        canvas_res,
        cur_t,
        node,
        node_px,
        spline_sel_node,
        spline_sel,
        roto_click,
        softcrop_req,
    } = cx;
    use engine::graph::ROTO_OP;
    // A roto CONTAINER: its inner graph's shapes and ops.
    // The panel shows whichever mini node is picked — a
    // shape's full panel (the one a plain Spline always
    // had, on that shape's channels), or an op's row.
    let n_sh = full.shape_count();
    let shape_sel = innards::selected(ui.ctx(), node, n_sh);
    let op_sel = innards::picked(ui.ctx(), node)
        .filter(|p| (ROTO_OP..rotoshapes::OUT).contains(p))
        .map(|p| p - ROTO_OP)
        .filter(|k| *k < full.ops.len());
    let ops_now: Vec<engine::graph::MatteOp> = full.ops.iter().map(|o| o.op).collect();
    // The soft crop boxes the whole roto, so any shape
    // with points is enough to offer it.
    let roto_drawn = (0..n_sh).any(|s| full.shape(s).is_some_and(|x| !x.points.is_empty()));
    // One row: the shapes (click picks, alt-click removes),
    // `+`, then the ops.
    let roto_strip = |ui: &mut egui::Ui, act: &mut Option<RotoAction>| {
        ui.horizontal_wrapped(|ui| {
            ui.label(egui::RichText::new("shapes").small());
            for s in 0..n_sh {
                let on = op_sel.is_none() && s == shape_sel;
                let r = ui.add(egui::Button::selectable(on, format!("{}", s + 1)).small());
                if r.clicked() {
                    *act = Some(if ui.input(|i| i.modifiers.alt) && n_sh > 1 {
                        RotoAction::RemoveShape(s)
                    } else {
                        RotoAction::Pick(s)
                    });
                }
            }
            if n_sh < engine::graph::ROTO_SHAPES_MAX
                && ui.small_button("+").on_hover_text("a new shape").clicked()
            {
                *act = Some(RotoAction::Add);
            }
            for (k, op) in ops_now.iter().enumerate() {
                let r = ui.add(
                    egui::Button::selectable(op_sel == Some(k), rotoshapes::op_label(*op)).small(),
                );
                if r.clicked() {
                    *act = Some(RotoAction::Pick(ROTO_OP + k));
                }
            }
        });
    };
    if let Some(k) = op_sel {
        // An op: what it does, and how much.
        roto_strip(ui, roto_click);
        let o = &mut full.ops[k];
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("op").small());
            egui::ComboBox::from_id_salt(("roto-op", node, k))
                .selected_text(o.op.name())
                .show_ui(ui, |ui| {
                    for c in engine::graph::MatteOp::ALL {
                        ui.selectable_value(&mut o.op, c, c.name());
                    }
                });
            if ui.small_button("remove").clicked() {
                *roto_click = Some(RotoAction::RemoveOp(k));
            }
        });
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new("mix").small());
            ui.add(egui::Slider::new(&mut o.mix, 0.0..=1.0));
        });
        ui.weak("A is its left port, B its right: cut takes B out of A");
    } else {
    let ch = |c: anim::Channel| anim::for_shape(shape_sel, c);
    let mut one = full.single(shape_sel);
    {
    let sp = &mut one;
    use anim::Channel::*;
    use engine::graph::{MatteOp, SplineMode};
    let mode_before = sp.mode;
    // TWO COLUMNS. This panel is two tiles WIDE and — like
    // every panel that is not a tool — ONE tile TALL, so a
    // single stacked run of shape, matte, transform and
    // point rows falls off the bottom into the tile's own
    // scroll area, where the buttons cannot be reached.
    // Exactly the trap the Grade's slider mode hit; the
    // room here is sideways, so: shape + matte left,
    // transform + point tools right.
    let mut h: Option<widgets::HudSlider> = None;
    ui.horizontal_top(|ui| {
    let col_w = ((ui.available_width() - 14.0) * 0.5).max(190.0);
    let col_h = ui.available_height();
    ui.allocate_ui(egui::vec2(col_w, col_h), |ui| {
    ui.vertical(|ui| {
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("shape").small());
        egui::ComboBox::from_id_salt(("spline-mode", node))
            .selected_text(sp.mode.name())
            .show_ui(ui, |ui| {
                for m in SplineMode::ALL {
                    ui.selectable_value(&mut sp.mode, m, m.name());
                }
            });
        ui.separator();
        ui.label(egui::RichText::new("combine").small());
        egui::ComboBox::from_id_salt(("spline-combine", node))
            .selected_text(sp.combine.name())
            .show_ui(ui, |ui| {
                for c in MatteOp::ALL {
                    ui.selectable_value(&mut sp.combine, c, c.name());
                }
            })
            .response
            .on_hover_text(
                "what this shape does to the matte on the input: \
                 chain Splines to build a roto out of several \
                 shapes (subtract punches a hole)",
            );
    });
    if sp.mode != mode_before && sp.mode == SplineMode::Bezier {
        seed_bezier_handles(&mut sp.points);
    }
    // The shape keys as one: this dot puts a keyframe on
    // every control point at once. Points cannot be keyed
    // apart yet — a shape animated a point at a time drifts
    // into a different shape between keys.
    ui.horizontal(|ui| {
        ui.add_sized(
            [widgets::LABEL_W, 16.0],
            egui::Label::new(egui::RichText::new("points").small()),
        );
        ui.weak(format!("{}", sp.points.len()));
        let chans: Vec<_> = anim::spline_shape(sp.points.len()).into_iter().map(ch).collect();
        ui.with_layout(
            egui::Layout::right_to_left(egui::Align::Center),
            |ui| widgets::key_dot(ui, rcx, &chans),
        );
    });
    // Feather-all: one control that writes every point,
    // and keys them all when the curve is animated.
    let mut fall = sp.points.first().map(|q| q.feather).unwrap_or(0.0);
    let pre_f = fall;
    // The standard row, so it lines up with the matte
    // rows under it; its dot keys every point's feather.
    let fchans: Vec<_> = anim::spline_feather(sp.points.len()).into_iter().map(ch).collect();
    h = h.take().or(widgets::slider(
        ui, rcx, "uniform", &mut fall, 0.0, 500.0, &fchans,
    ));
    if fall != pre_f {
        for q in sp.points.iter_mut() {
            q.feather = fall;
        }
    }
    // Shape-wide feather on ONE side of every point's
    // band: out softens past the outline with the inner
    // edge held, in softens inside it.
    h = h.take().or(widgets::drags(
        ui,
        rcx,
        "out / in",
        &mut [&mut sp.feather_out, &mut sp.feather_in],
        0.5,
        (0.0, 200.0),
        &[ch(SplineFeatherOut), ch(SplineFeatherIn)],
    ));
    sp.feather_out = sp.feather_out.max(0.0);
    sp.feather_in = sp.feather_in.max(0.0);
    // Matte controls. Each is a plain uniform — none of
    // them recompiles, so all of them are scrubbable at
    // frame rate on a shape of any size.
    h = h.take().or(widgets::slider(
        ui,
        rcx,
        "opacity",
        &mut sp.opacity,
        0.0,
        1.0,
        &[ch(SplineOpacity)],
    ));
    h = h.take().or(widgets::drags(
        ui,
        rcx,
        "expand",
        &mut [&mut sp.expand],
        0.25,
        (-100.0, 100.0),
        &[ch(SplineExpand)],
    ));
    h = h.take().or(widgets::slider(
        ui,
        rcx,
        "falloff",
        &mut sp.falloff,
        0.0,
        1.0,
        &[ch(SplineFalloff)],
    ));
    ui.horizontal(|ui| {
        ui.checkbox(&mut sp.invert, "invert").on_hover_text(
            "everything the shape is not — an empty Spline \
             inverts to a full matte",
        );
        ui.weak("· expand grows (+) or shrinks (−) the matte in px");
    });
    widgets::format_row(ui, ("spline-format", node), &mut sp.format, canvas_res);
    });
    });
    ui.separator();
    // The shape transform: what a whole shape rides on.
    // Five curves for a move, instead of two per point —
    // and the jacks a Tracker's TrackX/TrackY patch into.
    ui.allocate_ui(egui::vec2(col_w, col_h), |ui| {
    ui.vertical(|ui| {
    roto_strip(ui, roto_click);
    h = h.take().or(widgets::drags(
        ui,
        rcx,
        "shape pos",
        {
            let [tx, ty] = &mut sp.xf.translate;
            &mut [tx, ty]
        },
        0.002,
        (-1.0, 1.0),
        &[ch(SplineTx), ch(SplineTy)],
    ));
    h = h.take().or(widgets::drags(
        ui,
        rcx,
        "rotate",
        &mut [&mut sp.xf.rotate],
        0.25,
        (-180.0, 180.0),
        &[ch(SplineRot)],
    ));
    h = h.take().or(widgets::drags_linked(
        ui,
        rcx,
        "scale",
        {
            let [sx, sy] = &mut sp.xf.scale;
            &mut [sx, sy]
        },
        0.005,
        (0.0, 4.0),
        &[ch(SplineSx), ch(SplineSy)],
    ));
    h = h.take().or(widgets::drags(
        ui,
        rcx,
        "centre",
        {
            let [cx0, cy0] = &mut sp.xf.center;
            &mut [cx0, cy0]
        },
        0.002,
        (0.0, 1.0),
        &[ch(SplineCx), ch(SplineCy)],
    ));
    ui.horizontal(|ui| {
        if ui
            .button("bake transform")
            .on_hover_text(
                "fold the shape transform into the control \
                 points and reset it — the shape does not move",
            )
            .clicked()
            && !sp.xf.is_identity()
        {
            let sz = node_px.unwrap_or([1.0, 1.0]);
            let xf = sp.xf;
            for q in sp.points.iter_mut() {
                let p = xf.point(q.p, sz);
                let tip = |hnd: [f32; 2]| {
                    let t =
                        xf.point([q.p[0] + hnd[0], q.p[1] + hnd[1]], sz);
                    [t[0] - p[0], t[1] - p[1]]
                };
                q.h_in = tip(q.h_in);
                q.h_out = tip(q.h_out);
                q.p = p;
            }
            sp.xf = Default::default();
        }
        if ui
            .button("reset transform")
            .on_hover_text("back to the authored shape")
            .clicked()
        {
            sp.xf = Default::default();
        }
    });
    // A SoftCrop around the shape's box, downstream: this
    // frame's box, or one keyed on every frame of the shot.
    ui.horizontal(|ui| {
        let has_shape = roto_drawn;
        ui.label("soft crop");
        if ui
            .add_enabled(has_shape, egui::Button::new("this frame"))
            .on_hover_text("a SoftCrop downstream, boxed to what the roto draws now")
            .on_disabled_hover_text("draw the shape first")
            .clicked()
        {
            *softcrop_req = Some((node, false));
        }
        if ui
            .add_enabled(has_shape, egui::Button::new("bake"))
            .on_hover_text(
                "a SoftCrop downstream, keyed to the roto's box on \
                 every frame of the shot",
            )
            .on_disabled_hover_text("draw the shape first")
            .clicked()
        {
            *softcrop_req = Some((node, true));
        }
    });
    ui.separator();
    ui.horizontal(|ui| {
        let n = sp.points.len();
        if n >= 2 && ui.button("+ point").clicked() {
            // Split the longest segment.
            let (mut bi, mut bd) = (0usize, -1.0f32);
            for i in 0..n {
                let a = sp.points[i].p;
                let b = sp.points[(i + 1) % n].p;
                let d = (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2);
                if d > bd {
                    bd = d;
                    bi = i;
                }
            }
            let a = sp.points[bi];
            let b = sp.points[(bi + 1) % n];
            sp.points.insert(
                bi + 1,
                engine::graph::SplinePoint::new(
                    [(a.p[0] + b.p[0]) * 0.5, (a.p[1] + b.p[1]) * 0.5],
                    (a.feather + b.feather) * 0.5,
                ),
            );
            let pts = sp.points.clone();
            anim::in_shape(rcx.anim, shape_sel, |a| {
                anim::spline_point_inserted(a, &pts, bi + 1, cur_t)
            });
        }
        if n > 0 && ui.button("- point").clicked() {
            sp.points.pop();
            anim::in_shape(rcx.anim, shape_sel, |a| {
                anim::spline_point_removed(a, n - 1, n)
            });
        }
        if n > 0 && ui.button("clear").clicked() {
            sp.points.clear();
            anim::in_shape(rcx.anim, shape_sel, anim::spline_channels_cleared);
        }
    });
    if sp.points.is_empty() {
        ui.weak("S-click in the viewer to lay points down");
        ui.weak("(an empty shape + invert = a full matte)");
    } else {
        ui.weak(format!("{} points", sp.points.len()));
        ui.weak(
            "S-click = add · D-click = remove · drag = move",
        );
        ui.weak(
            "drag bg = marquee · S-click = extend · \
             alt-drag = feather",
        );
        ui.weak(
            "the shape transform moves the WHOLE shape — key it, \
             or patch a Tracker into it (via an add node: the \
             track is a position, this is an offset), instead \
             of keying every point",
        );
        let picked: Vec<usize> = if spline_sel_node == Some(node) {
            spline_sel.iter().copied().collect()
        } else {
            Vec::new()
        };
        tension_row(ui, &mut sp.points, sp.mode, &picked);
        if spline_sel_node == Some(node)
            && spline_sel.len() >= 2
        {
            ui.weak(format!(
                "{} picked — box handles scale, the knob rotates, \
                 the centre moves (alt-scale from the centre)",
                spline_sel.len()
            ));
        }
    }
    });
    });
    });
    if h.is_some() {
        hud_drag_out = h;
    }
    }
    full.apply_single(shape_sel, &one);
    }
    hud_drag_out
}
