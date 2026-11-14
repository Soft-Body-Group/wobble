//! One node of a 2dini graph: every shape, modifier and solver's page.
//! Moved out of `properties_ui` (lib.rs) unchanged.

use crate::*;

/// Everything the geonode panel borrows from the properties pass.
pub(crate) struct Cx<'a> {
    pub gn: &'a mut engine::geo::GeoNode,
    pub cur_t: u64,
    pub node: u64,
    pub tab: &'a String,
    pub wired: [bool; 16],
    pub working: engine::color::Colorspace,
    pub merge_before: &'a Vec<(usize, String, engine::geo::Layering)>,
    pub physics_live: Option<(u32, u32)>,
    pub physics_status: Option<(Option<(u32, usize, u32)>, bool)>,
    pub spline_sel_node: Option<u64>,
    pub spline_sel: &'a std::collections::BTreeSet<usize>,
    pub geo_edit_req: &'a mut Option<u64>,
    pub merge_edit: &'a mut Option<(u64, Vec<(usize, engine::geo::Layering)>)>,
    pub physics_cancel_req: &'a mut bool,
    pub physics_req: &'a mut Option<(u64, bool)>,
    pub ragdoll_req: &'a mut Option<(u64, bool)>,
    pub merge_rows: &'a mut Vec<(usize, String, engine::geo::Layering)>,
}

/// One node of a 2dini graph: every shape, modifier and solver's page.
pub(crate) fn panel(
    ui: &mut egui::Ui,
    rcx: &mut widgets::RowCtx,
    cx: Cx,
) -> Option<widgets::HudSlider> {
    let mut hud_drag_out: Option<widgets::HudSlider> = None;
    let Cx {
        gn,
        cur_t,
        node,
        tab,
        wired,
        working,
        merge_before,
        physics_live,
        physics_status,
        spline_sel_node,
        spline_sel,
        geo_edit_req,
        merge_edit,
        physics_cancel_req,
        physics_req,
        ragdoll_req,
        merge_rows,
    } = cx;
    // The points picked in the viewer, for the tension row.
    let geo_picked: Vec<usize> = if spline_sel_node == Some(node) {
        spline_sel.iter().copied().collect()
    } else {
        Vec::new()
    };
    use engine::geo::*;
    // Same contract as a field node: a socket that is
    // WIRED overrides its literal, so its row is greyed
    // and says so; every other channel is a standard
    // slider row with a key dot and a HUD drag-out.
    let sockets = gn.inputs();
    let driven: Vec<bool> = (0..gn.channel_count())
        .map(|i| {
            wired.get(i).copied().unwrap_or(false)
                && sockets.get(i).is_some_and(|s| s.kind == engine::graph::PortKind::Field)
        })
        .collect();
    let mut row = |ui: &mut egui::Ui,
                   rcx: &mut widgets::RowCtx,
                   label: &str,
                   v: &mut f32,
                   lo: f32,
                   hi: f32,
                   idx: Option<usize>| {
        let on = idx.is_some_and(|i| driven.get(i).copied().unwrap_or(false));
        if on {
            ui.horizontal(|ui| {
                ui.add_sized(
                    [76.0, 16.0],
                    egui::Label::new(
                        egui::RichText::new(label)
                            .small()
                            .color(theme::cable()),
                    ),
                );
                ui.add_enabled(false, egui::DragValue::new(v).update_while_editing(false));
                ui.weak("driven by a socket");
            });
            return;
        }
        match idx {
            Some(i) => {
                if let Some(h) = widgets::slider(
                    ui,
                    rcx,
                    label,
                    v,
                    lo,
                    hi,
                    &[anim::Channel::Field(i as u16)],
                ) {
                    hud_drag_out = Some(h);
                }
            }
            None => {
                ui.horizontal(|ui| {
                    ui.add_sized(
                        [76.0, 16.0],
                        egui::Label::new(egui::RichText::new(label).small()),
                    );
                    ui.add(
                        egui::DragValue::new(v).update_while_editing(false).speed(widgets::rated(ui, (hi - lo) * 0.002)),
                    );
                });
            }
        }
    };
    // A generator's style is a plain param: colours as
    // swatches, the width a number, the rule a choice.
    let style_rows = |ui: &mut egui::Ui, st: &mut Style, salt: &str| {
        ui.horizontal(|ui| {
            ui.add_sized(
                [52.0, 16.0],
                egui::Label::new(egui::RichText::new("fill").small()),
            );
            colorpicker::button(
                ui,
                (node, salt, "fill"),
                &mut st.fill,
                pick_spec(working, true),
            );
            ui.add_sized(
                [44.0, 16.0],
                egui::Label::new(egui::RichText::new("stroke").small()),
            );
            colorpicker::button(
                ui,
                (node, salt, "stroke"),
                &mut st.stroke,
                pick_spec(working, true),
            );
        });
        ui.horizontal(|ui| {
            ui.add_sized(
                [52.0, 16.0],
                egui::Label::new(egui::RichText::new("width").small()),
            );
            ui.add(
                egui::DragValue::new(&mut st.width).update_while_editing(false)
                    .speed(widgets::rated(ui, 0.1))
                    .range(0.0..=200.0)
                    .suffix(" px"),
            );
            let mut rule = st.rule;
            egui::ComboBox::from_id_salt((salt, "rule", node))
                .width(84.0)
                .selected_text(rule.name())
                .show_ui(ui, |ui| {
                    for r in FillRule::ALL {
                        ui.selectable_value(&mut rule, r, r.name());
                    }
                });
            st.rule = rule;
        });
    };
    // The gradient is its own tab (`panel_tabs`): the
    // ramp editor does not fit under a shape's rows.
    if tab == "style" {
        // A 2dini Text's typography and styling: the
        // standalone Text node's, keyed by literal index.
        if let GeoNode::Text { fx, .. } = gn {
            let ch = |s: &str| {
                let k = engine::textfx::GEO_STYLE_PARAMS.iter().position(|p| *p == s).unwrap_or(0);
                anim::Channel::Field(text_style_index(k) as u16)
            };
            if let Some(h) = nodes::text::geo_style_page(ui, rcx, fx, working, &ch) {
                hud_drag_out = Some(h);
            }
        }
    } else if tab == "animate" {
        // A 2dini Text's animators: the Text node's own
        // page, keyed by literal index.
        if let GeoNode::Text { fx, .. } = gn {
            let ch = |i: usize, s: &str| {
                let k = engine::textfx::ANIM_PARAMS.iter().position(|p| *p == s).unwrap_or(0);
                anim::Channel::Field(text_anim_index(i, k) as u16)
            };
            if let Some(h) =
                nodes::text::animators_page(ui, rcx, &mut fx.animators, node, working, &ch)
            {
                hud_drag_out = Some(h);
            }
        }
    } else if tab == "gradient" {
        if let Some(st) = geo_style_mut(gn) {
            geo_gradient_rows(ui, node, "geo-grad", st, working);
        }
    } else {
    // Every shape maker can have its points dragged by
    // hand: an Edit after it (Houdini's Edit SOP).
    if gn.is_geometry() && !matches!(gn, GeoNode::Edit { .. } | GeoNode::Output) {
        if ui
            .small_button("edit points")
            .on_hover_text("drag this shape's points by hand in the viewer (an Edit node after it)")
            .clicked()
        {
            *geo_edit_req = Some(node);
        }
    }
    match gn {
        GeoNode::Spline { mode, closed, style, points } => {
            ui.horizontal(|ui| {
                egui::ComboBox::from_id_salt(("geo-spline-mode", node))
                    .width(96.0)
                    .selected_text(mode.name())
                    .show_ui(ui, |ui| {
                        for m in engine::graph::SplineMode::ALL {
                            ui.selectable_value(mode, m, m.name());
                        }
                    });
                ui.checkbox(closed, "closed");
            });
            style_rows(ui, style, "geo-spline");
            tension_row(ui, points, *mode, &geo_picked);
            ui.weak(if points.is_empty() {
                "click in the viewer to lay points down"
            } else {
                "drag points in the viewer; shift-click adds, alt-click removes"
            });
        }
        GeoNode::Text { text, font, size, position, align, tracking, leading, rotation, style, fx } => {
            use engine::geotext::{TextAlign, TextFont};
            // A String on the words socket says what is written.
            let words_wired = wired.get(TEXT_WORDS_INPUT).copied().unwrap_or(false);
            ui.add_enabled(
                !words_wired,
                egui::TextEdit::multiline(text)
                    .desired_rows(2)
                    .desired_width(f32::INFINITY),
            )
            .on_disabled_hover_text("from the String wired into the words input");
            ui.horizontal(|ui| {
                let mut pick = match font {
                    TextFont::Sans => 0,
                    TextFont::Mono => 1,
                    TextFont::File(_) => 2,
                };
                let before = pick;
                ui.selectable_value(&mut pick, 0, "sans");
                ui.selectable_value(&mut pick, 1, "mono");
                if ui.selectable_label(pick == 2, format!("file: {}", if pick == 2 { font.name() } else { "…".into() })).clicked() {
                    #[cfg(not(target_arch = "wasm32"))]
                    if let Some(p) = rfd::FileDialog::new().add_filter("font", &["ttf", "otf", "ttc"]).pick_file() {
                        *font = TextFont::File(p.to_string_lossy().into_owned());
                    }
                }
                if pick != before {
                    match pick {
                        0 => *font = TextFont::Sans,
                        1 => *font = TextFont::Mono,
                        _ => {}
                    }
                }
            });
            ui.horizontal(|ui| {
                for a in TextAlign::ALL {
                    ui.selectable_value(align, a, a.name());
                }
            });
            row(ui, rcx, "x", &mut position[0], -0.5, 1.5, Some(0));
            row(ui, rcx, "y", &mut position[1], -0.5, 1.5, Some(1));
            row(ui, rcx, "size", size, 0.005, 0.6, Some(2));
            row(ui, rcx, "rotation", rotation, -std::f32::consts::PI, std::f32::consts::PI, Some(3));
            row(ui, rcx, "tracking", tracking, -0.2, 1.0, Some(4));
            widgets::plain_slider(ui, "leading", leading, 0.5, 3.0, false);
            ui.horizontal(|ui| {
                for c in engine::textfx::TextCase::ALL {
                    ui.selectable_value(&mut fx.case, c, egui::RichText::new(c.name()).small());
                }
                ui.add(egui::DragValue::new(&mut fx.wrap).speed(0.005).range(0.0..=2.0).prefix("wrap "))
                    .on_hover_text("paragraph box width, canvas widths (0 = none)");
                ui.checkbox(&mut fx.justify, egui::RichText::new("justify").small());
            });
            if wired.get(5).copied().unwrap_or(false) {
                ui.horizontal(|ui| {
                    ui.checkbox(&mut fx.perpendicular, egui::RichText::new("turn with the path").small());
                    ui.checkbox(&mut fx.reverse, egui::RichText::new("reverse").small());
                });
                row(ui, rcx, "path offset", &mut fx.path_offset, -1.0, 1.0, Some(TEXT_PATH_OFFSET));
            }
            style_rows(ui, style, "geo-text");
            ui.weak("one shape per glyph outline; counters are holes · size is the em, the canvas is 1 wide · x/y is the first baseline · wire a shape into path to run the text along it · animators on their own tab");
        }
        GeoNode::Stroke { points, mode, color, width } => {
            ui.horizontal(|ui| {
                egui::ComboBox::from_id_salt(("geo-stroke-mode", node))
                    .width(96.0)
                    .selected_text(mode.name())
                    .show_ui(ui, |ui| {
                        for m in engine::graph::SplineMode::ALL {
                            ui.selectable_value(mode, m, m.name());
                        }
                    });
                ui.add_sized(
                    [44.0, 16.0],
                    egui::Label::new(egui::RichText::new("colour").small()),
                );
                colorpicker::button(
                    ui,
                    (node, "geo-stroke", "colour"),
                    color,
                    pick_spec(working, true),
                );
            });
            row(ui, rcx, "width", width, 0.0, 60.0, Some(0));
            tension_row(ui, points, *mode, &geo_picked);
            ui.weak(if points.is_empty() {
                "a line, never filled — click in the viewer to lay points down"
            } else {
                "drag points in the viewer; shift-click adds, alt-click removes"
            });
        }
        GeoNode::Polygon { center, radius, sides, step, rotation, style } => {
            row(ui, rcx, "centre x", &mut center[0], 0.0, 1.0, Some(0));
            row(ui, rcx, "centre y", &mut center[1], 0.0, 1.0, Some(1));
            row(ui, rcx, "radius", radius, 0.0, 1.0, Some(2));
            row(ui, rcx, "rotation", rotation, -std::f32::consts::PI, std::f32::consts::PI, Some(3));
            ui.horizontal(|ui| {
                ui.add_sized(
                    [52.0, 16.0],
                    egui::Label::new(egui::RichText::new("sides").small()),
                );
                ui.add(egui::DragValue::new(sides).update_while_editing(false).speed(widgets::rated(ui, 1.0)).range(3..=256));
                ui.add_sized(
                    [40.0, 16.0],
                    egui::Label::new(egui::RichText::new("step").small()),
                );
                ui.add(egui::DragValue::new(step).update_while_editing(false).speed(widgets::rated(ui, 1.0)).range(1..=128))
                    .on_hover_text("vertex stride: 2 on 5 sides is the pentagram");
            });
            style_rows(ui, style, "geo-poly");
        }
        GeoNode::Skeleton { bones, pose } => {
            pose.resize(bones.len(), 0.0);
            for i in 0..bones.len() {
                let label = if bones[i].name.is_empty() { format!("bone {i}") } else { bones[i].name.clone() };
                row(ui, rcx, &label, &mut pose[i], -std::f32::consts::PI, std::f32::consts::PI, Some(i));
            }
            ui.horizontal(|ui| {
                if ui.button("add bone").on_hover_text("a child of the last bone, carrying on its line").clicked() {
                    let (head, dir, parent) = match bones.last() {
                        Some(l) => (l.tail, [l.tail[0] - l.head[0], l.tail[1] - l.head[1]], Some(bones.len() as u32 - 1)),
                        None => ([0.4, 0.4], [0.12, 0.0], None),
                    };
                    bones.push(engine::georig::Bone {
                        name: String::new(),
                        parent,
                        head,
                        tail: [head[0] + dir[0], head[1] + dir[1]],
                    });
                    pose.push(0.0);
                }
                if bones.len() > 1 && ui.button("remove last").clicked() {
                    bones.pop();
                    pose.pop();
                }
                if ui.button("rest").on_hover_text("every turn back to 0").clicked() {
                    pose.iter_mut().for_each(|p| *p = 0.0);
                }
            });
            ui.weak("viewer: drag a bone to turn it (keys the pose); alt-drag a joint to move it at rest · a turn is relative to the parent");
        }
        GeoNode::Skin { smooth, falloff } => {
            ui.checkbox(smooth, "smooth")
                .on_hover_text("off: each point follows its nearest bone alone (rigid parts)");
            if *smooth {
                widgets::plain_slider(ui, "falloff", falloff, 1.0, 12.0, false)
                    .on_hover_text("how fast a bone lets go with distance: high is sharp joints, low is rubber");
            }
            ui.weak("input 1: the shapes · input 2: a Skeleton (or a Ragdoll); weights are taken with the bones at rest · a shape bends only at its points: Resample it first for a smooth bend");
        }
        GeoNode::Ragdoll { gravity, limit, frames, start, walls, kick, bake } => {
            widgets::plain_slider(ui, "gravity", gravity, -30.0, 30.0, false)
                .on_hover_text("m/s² downward; the canvas is 10 m wide");
            widgets::plain_slider(ui, "joint limit", limit, 0.0, std::f32::consts::PI, false)
                .on_hover_text("radians either way from rest each joint may bend");
            widgets::plain_slider(ui, "push x", &mut kick[0], -3.0, 3.0, false);
            widgets::plain_slider(ui, "push y", &mut kick[1], -3.0, 3.0, false);
            widgets::plain_slider(ui, "spin", &mut kick[2], -20.0, 20.0, false)
                .on_hover_text("the root bone's start velocity (widths a second) and spin (radians a second)");
            ui.horizontal(|ui| {
                ui.checkbox(walls, "canvas walls");
                ui.add_sized([32.0, 16.0], egui::Label::new(egui::RichText::new("start").small()));
                ui.add(egui::DragValue::new(start).speed(1.0).update_while_editing(false));
                if ui.small_button("here").on_hover_text("start at the frame on screen").clicked() {
                    *start = cur_t as f32;
                }
                ui.add_sized([40.0, 16.0], egui::Label::new(egui::RichText::new("frames").small()));
                ui.add(egui::DragValue::new(frames).update_while_editing(false).range(1..=2000));
            });
            ui.horizontal(|ui| {
                if ui.button("bake").on_hover_text("let the skeleton fall from its pose at the start frame").clicked() {
                    *ragdoll_req = Some((node, true));
                }
                if !bake.is_empty() && ui.button("clear").clicked() {
                    *ragdoll_req = Some((node, false));
                }
            });
            if bake.is_empty() {
                ui.weak("not baked: the skeleton passes through as posed");
            } else {
                ui.weak(format!("baked: {} bones, {} frames from {}", bake.bones, bake.frames, start));
            }
            ui.weak("input: a Skeleton; feed this into a Skin's second input to make shapes fall with it");
        }
        GeoNode::Particles {
            rate, life, life_var, speed, speed_var, direction, spread, gravity, wind, drag, size, size_end, fade, start, seed,
        } => {
            row(ui, rcx, "rate", rate, 0.0, 30.0, Some(1));
            row(ui, rcx, "life", life, 1.0, 400.0, Some(5));
            widgets::plain_slider(ui, "life ±", life_var, 0.0, 1.0, false);
            row(ui, rcx, "speed", speed, 0.0, 80.0, Some(2));
            widgets::plain_slider(ui, "speed ±", speed_var, 0.0, 1.0, false);
            row(ui, rcx, "direction", direction, -std::f32::consts::PI, std::f32::consts::PI, Some(3));
            row(ui, rcx, "spread", spread, 0.0, std::f32::consts::PI, Some(6));
            row(ui, rcx, "gravity", gravity, -3.0, 3.0, Some(4));
            row(ui, rcx, "wind", wind, -3.0, 3.0, Some(7));
            row(ui, rcx, "drag", drag, 0.0, 0.5, Some(8));
            row(ui, rcx, "size", size, 0.0, 60.0, Some(9));
            row(ui, rcx, "size at death", size_end, 0.0, 60.0, Some(10));
            ui.horizontal(|ui| {
                ui.checkbox(fade, "fade out");
                ui.label(egui::RichText::new("start").small());
                ui.add(egui::DragValue::new(start).speed(1.0));
                ui.label(egui::RichText::new("seed").small());
                ui.add(egui::DragValue::new(seed).speed(0.1));
            });
            ui.weak("born on the emitter's points or inside its closed shapes (the canvas centre unwired); px a frame; solved per frame, no bake · an Instance stamps shapes onto them, a Displace with a moving Noise adds turbulence");
        }
        GeoNode::Brush { strokes, color, width, taper, smooth } => {
            ui.horizontal(|ui| {
                ui.add_sized(
                    [52.0, 16.0],
                    egui::Label::new(egui::RichText::new("colour").small()),
                );
                colorpicker::button(ui, (node, "geo-brush", "colour"), color, pick_spec(working, true));
                if ui.button("clear").on_hover_text("remove every stroke").clicked() {
                    strokes.clear();
                }
                if ui.button("undo stroke").clicked() {
                    strokes.pop();
                }
            });
            row(ui, rcx, "width", width, 0.0, 120.0, Some(0));
            widgets::plain_slider(ui, "taper", taper, 0.0, 1.0, false);
            widgets::plain_slider(ui, "smooth", smooth, 0.0, 1.0, false);
            ui.weak(format!(
                "{} strokes — drag in the viewer to paint; a quick hand draws thinner",
                strokes.len()
            ));
        }
        GeoNode::Circle { center, radius, style } => {
            row(ui, rcx, "centre x", &mut center[0], 0.0, 1.0, Some(0));
            row(ui, rcx, "centre y", &mut center[1], 0.0, 1.0, Some(1));
            row(ui, rcx, "radius", radius, 0.0, 1.0, Some(2));
            style_rows(ui, style, "geo-circle");
        }
        GeoNode::Rectangle { center, size, corner, rotation, style } => {
            row(ui, rcx, "centre x", &mut center[0], 0.0, 1.0, Some(0));
            row(ui, rcx, "centre y", &mut center[1], 0.0, 1.0, Some(1));
            row(ui, rcx, "width", &mut size[0], 0.0, 1.0, Some(2));
            row(ui, rcx, "height", &mut size[1], 0.0, 1.0, Some(3));
            row(ui, rcx, "corner", corner, 0.0, 0.25, Some(4));
            row(ui, rcx, "rotation", rotation, -std::f32::consts::PI, std::f32::consts::PI, Some(5));
            style_rows(ui, style, "geo-rect");
        }
        GeoNode::Star { center, outer, inner, points, rotation, style } => {
            row(ui, rcx, "centre x", &mut center[0], 0.0, 1.0, Some(0));
            row(ui, rcx, "centre y", &mut center[1], 0.0, 1.0, Some(1));
            row(ui, rcx, "outer", outer, 0.0, 1.0, Some(2));
            row(ui, rcx, "inner", inner, 0.0, 1.0, Some(3));
            row(ui, rcx, "rotation", rotation, -std::f32::consts::PI, std::f32::consts::PI, Some(4));
            ui.horizontal(|ui| {
                ui.add_sized(
                    [52.0, 16.0],
                    egui::Label::new(egui::RichText::new("points").small()),
                );
                ui.add(egui::DragValue::new(points).update_while_editing(false).speed(widgets::rated(ui, 1.0)).range(2..=128));
            });
            style_rows(ui, style, "geo-star");
        }
        GeoNode::Resample { mode, spacing, count } => {
            ui.horizontal(|ui| {
                for m in ResampleMode::ALL {
                    ui.selectable_value(mode, m, m.name());
                }
            });
            match mode {
                ResampleMode::Spacing => {
                    row(ui, rcx, "spacing", spacing, 1.0, 200.0, Some(1));
                    ui.weak("output pixels between vertices");
                }
                ResampleMode::Count => {
                    ui.horizontal(|ui| {
                        ui.add_sized(
                            [52.0, 16.0],
                            egui::Label::new(egui::RichText::new("count").small()),
                        );
                        ui.add(egui::DragValue::new(count).update_while_editing(false).speed(widgets::rated(ui, 1.0)).range(2..=512));
                    });
                }
            }
        }
        GeoNode::Scatter { count, seed, density } => {
            ui.horizontal(|ui| {
                ui.add_sized(
                    [52.0, 16.0],
                    egui::Label::new(egui::RichText::new("count").small()),
                );
                ui.add(egui::DragValue::new(count).update_while_editing(false).range(0..=4096));
                ui.add_sized(
                    [40.0, 16.0],
                    egui::Label::new(egui::RichText::new("seed").small()),
                );
                ui.add(egui::DragValue::new(seed).update_while_editing(false).speed(widgets::rated(ui, 0.1)));
            });
            row(ui, rcx, "density", density, 0.0, 1.0, Some(1));
            ui.weak("inside the closed shapes wired in (the canvas when none); wire a Field In at density to follow a noise");
        }
        GeoNode::Displace { mode, amount, x, y } => {
            ui.horizontal(|ui| {
                for m in engine::field::WarpMode::ALL {
                    ui.selectable_value(mode, m, m.name());
                }
            });
            match mode {
                engine::field::WarpMode::Offset => {
                    row(ui, rcx, "x", x, 0.0, 1.0, Some(1));
                    row(ui, rcx, "y", y, 0.0, 1.0, Some(2));
                    row(ui, rcx, "amount", amount, -1.0, 1.0, Some(3));
                    ui.weak("moves every point by (x, y) − 0.5, times amount");
                }
                engine::field::WarpMode::Curl => {
                    row(ui, rcx, "potential", x, 0.0, 1.0, Some(1));
                    row(ui, rcx, "amount", amount, -1.0, 1.0, Some(2));
                    ui.weak("moves every point along the curl of the potential");
                }
            }
        }
        GeoNode::Instance { scale, rotation, inherit_cd, pivot } => {
            row(ui, rcx, "scale", scale, 0.0, 2.0, Some(2));
            row(ui, rcx, "rotation", rotation, -std::f32::consts::PI, std::f32::consts::PI, Some(3));
            ui.horizontal(|ui| {
                ui.checkbox(inherit_cd, "inherit colour");
                egui::ComboBox::from_id_salt(("geo-pivot", node))
                    .width(96.0)
                    .selected_text(pivot.name())
                    .show_ui(ui, |ui| {
                        for p in Pivot::ALL {
                            ui.selectable_value(pivot, p, p.name());
                        }
                    });
            });
            ui.weak("input 1 is the template, input 2 the points it lands on");
        }
        GeoNode::Transform { translate, rotate, scale, pivot, inherit } => {
            ui.checkbox(inherit, "pivot from the shape")
                .on_hover_text("turn and scale about the incoming shape's own centre; pivot x/y then nudge it");
            row(ui, rcx, "translate x", &mut translate[0], -1.0, 1.0, Some(1));
            row(ui, rcx, "translate y", &mut translate[1], -1.0, 1.0, Some(2));
            row(ui, rcx, "rotate", rotate, -std::f32::consts::PI, std::f32::consts::PI, Some(3));
            row(ui, rcx, "scale x", &mut scale[0], 0.0, 4.0, Some(4));
            row(ui, rcx, "scale y", &mut scale[1], 0.0, 4.0, Some(5));
            let (lo, hi) = if *inherit { (-1.0, 1.0) } else { (-1.0, 2.0) };
            row(ui, rcx, "pivot x", &mut pivot[0], lo, hi, Some(6));
            row(ui, rcx, "pivot y", &mut pivot[1], lo, hi, Some(7));
            ui.weak("drag the handles in the viewer: centre moves, ring rotates, corners scale, alt-drag places the pivot");
        }
        GeoNode::Merge { count } => {
            // Each noodle's z (stack order: lowest under,
            // wiring order breaking ties) and blend.
            for (i, name, l) in merge_rows.iter_mut() {
                ui.horizontal(|ui| {
                    ui.add_sized(
                        [76.0, 16.0],
                        egui::Label::new(egui::RichText::new(format!("{} {name}", *i + 1)).small()).truncate(),
                    );
                    ui.label(egui::RichText::new("z").small());
                    ui.add(egui::DragValue::new(&mut l.z).update_while_editing(false).speed(widgets::rated(ui, 0.05)));
                    egui::ComboBox::from_id_salt(("merge-blend", node, *i))
                        .width(76.0)
                        .selected_text(l.blend.name())
                        .show_ui(ui, |ui| {
                            for m in BlendMode::ALL {
                                ui.selectable_value(&mut l.blend, m, m.name());
                            }
                        });
                });
            }
            if merge_rows != merge_before {
                *merge_edit = Some((node, merge_rows.iter().map(|(i, _, l)| (*i, *l)).collect()));
            }
            ui.weak(format!("{} wired — drop wires anywhere on the bar; draw order is wiring order", count.saturating_sub(1)));
        }
        GeoNode::Boolean { op } => {
            ui.horizontal(|ui| {
                for m in engine::geobool::BoolOp::ALL {
                    ui.selectable_value(op, m, m.name());
                }
            });
            ui.weak(match op {
                engine::geobool::BoolOp::Union => "a and b fused (a alone: its overlaps dissolved)",
                engine::geobool::BoolOp::Intersect => "only where a and b overlap; lines in a keep their inside",
                engine::geobool::BoolOp::Subtract => "a with b cut out; each shape keeps its style, lines are cut",
                engine::geobool::BoolOp::Exclude => "where exactly one of a and b is",
            });
        }
        GeoNode::Style { style } => {
            ui.horizontal(|ui| {
                ui.add_sized(
                    [52.0, 16.0],
                    egui::Label::new(egui::RichText::new("fill").small()),
                );
                colorpicker::button(
                    ui,
                    (node, "style-fill"),
                    &mut style.fill,
                    pick_spec(working, true),
                );
                ui.add_sized(
                    [44.0, 16.0],
                    egui::Label::new(egui::RichText::new("stroke").small()),
                );
                colorpicker::button(
                    ui,
                    (node, "style-stroke"),
                    &mut style.stroke,
                    pick_spec(working, true),
                );
                let mut rule = style.rule;
                egui::ComboBox::from_id_salt(("geo-style-rule", node))
                    .width(84.0)
                    .selected_text(rule.name())
                    .show_ui(ui, |ui| {
                        for r in FillRule::ALL {
                            ui.selectable_value(&mut rule, r, r.name());
                        }
                    });
                style.rule = rule;
            });
            // Colour as the house rgb slider (ganged, keyed,
            // the S/D/F colour axes), alpha its own row.
            // A wired lane stays a plain greyed row.
            let spec = rgbslider::RgbSpec::scale(0.0, 1.0, working.primaries.luma_weights())
                .with_display(engine::color::primaries_matrix(working.primaries, engine::color::Primaries::Rec709));
            for (base, label, c) in [(1usize, "fill", &mut style.fill), (5, "stroke", &mut style.stroke)] {
                let lanes_wired = (0..3).any(|k| driven.get(base + k).copied().unwrap_or(false));
                if lanes_wired {
                    for (k, lane) in ["r", "g", "b"].iter().enumerate() {
                        row(ui, rcx, &format!("{label} {lane}"), &mut c[k], 0.0, 1.0, Some(base + k));
                    }
                } else {
                    let gid = egui::Id::new(("geo-style-gang", node, label));
                    let mut gang = ui.data(|d| d.get_temp::<bool>(gid)).unwrap_or(true);
                    let mut v = [c[0], c[1], c[2]];
                    let chans = [
                        anim::Channel::Field(base as u16),
                        anim::Channel::Field(base as u16 + 1),
                        anim::Channel::Field(base as u16 + 2),
                    ];
                    // (The row closure holds the drag-out slot,
                    // so this row offers none.)
                    let _ = widgets::rgb_row(ui, rcx, label, &mut v, &mut gang, spec, &chans);
                    ui.data_mut(|d| d.insert_temp(gid, gang));
                    c[..3].copy_from_slice(&v);
                }
                row(ui, rcx, &format!("{label} alpha"), &mut c[3], 0.0, 1.0, Some(base + 3));
            }
            row(ui, rcx, "width", &mut style.width, 0.0, 200.0, Some(9));
            ui.weak("sockets are evaluated at each shape's centre");
        }
        GeoNode::Output => {
            ui.weak("what the 2dini node draws");
        }
        GeoNode::Edit { edits } => {
            ui.horizontal(|ui| {
                ui.label(format!("{} points moved by hand", edits.len()));
                if ui.add_enabled(!edits.is_empty(), egui::Button::new("reset all")).clicked() {
                    edits.clear();
                }
            });
            ui.weak("drag points in the viewer · shift-click picks more · drag on nothing boxes them · alt-click puts one back");
            ui.weak("edits are kept by shape and vertex number: change the count upstream and the ones past it rest");
        }
        GeoNode::String { text } => {
            ui.add(
                egui::TextEdit::multiline(text)
                    .id(egui::Id::new(("geo-string-text", node)))
                    .hint_text("words")
                    .desired_rows(4)
                    .desired_width(f32::INFINITY),
            );
            ui.weak("wire into a Text's words socket: it writes these");
        }
        GeoNode::Slider { value, name } => {
            ui.horizontal(|ui| {
                ui.add_sized(
                    [52.0, 16.0],
                    egui::Label::new(egui::RichText::new("name").small()),
                );
                ui.add(egui::TextEdit::singleline(name).desired_width(120.0));
            });
            row(ui, rcx, "value", value, -4.0, 4.0, Some(0));
            ui.weak("a widget on the 2dini node's panel, by this name");
        }
        GeoNode::Input { name, value } => {
            ui.horizontal(|ui| {
                ui.add_sized(
                    [52.0, 16.0],
                    egui::Label::new(egui::RichText::new("name").small()),
                );
                ui.add(egui::TextEdit::singleline(name).desired_width(120.0));
            });
            row(ui, rcx, "value", value, -4.0, 4.0, Some(0));
            ui.weak("a jack on the 2dini node, cabled in link mode");
        }
        GeoNode::FieldIn { shift } => {
            row(ui, rcx, "seed shift", shift, 0.0, 64.0, Some(0));
            ui.weak("the Noise on the 2dini node's field port, sampled at the point; 0.5 when unwired");
        }
        GeoNode::Math { op, a, b } => {
            ui.horizontal(|ui| {
                egui::ComboBox::from_id_salt(("geo-math", node))
                    .width(96.0)
                    .selected_text(op.name())
                    .show_ui(ui, |ui| {
                        for m in engine::field::MathOp::ALL {
                            ui.selectable_value(op, m, m.name());
                        }
                    });
            });
            row(ui, rcx, "a", a, -4.0, 4.0, Some(0));
            row(ui, rcx, "b", b, -4.0, 4.0, Some(1));
        }
        GeoNode::RigidBody { density, friction, bounce, pinned, velocity, spin } => {
            egui::ComboBox::from_id_salt(("rigid-mat", node))
                .selected_text("material…")
                .show_ui(ui, |ui| {
                    for (name, d, f, b) in engine::geophys::RIGID_MATERIALS {
                        if ui.selectable_label(false, name).clicked() {
                            *density = d;
                            *friction = f;
                            *bounce = b;
                        }
                    }
                });
            row(ui, rcx, "density", density, 0.05, 10.0, Some(1));
            row(ui, rcx, "friction", friction, 0.0, 1.0, Some(2));
            row(ui, rcx, "bounce", bounce, 0.0, 1.0, Some(3));
            row(ui, rcx, "velocity x", &mut velocity[0], -20.0, 20.0, Some(4));
            row(ui, rcx, "velocity y", &mut velocity[1], -20.0, 20.0, Some(5));
            row(ui, rcx, "spin", spin, -20.0, 20.0, Some(6));
            ui.checkbox(pinned, "pinned").on_hover_text("never moves: a collider that is still drawn");
            ui.weak("tags every shape for a Physics node downstream; sockets are read at each shape's centre · velocity m/s, spin rad/s");
        }
        GeoNode::Light { strength, tint, occlude } => {
            row(ui, rcx, "strength", strength, 0.0, 16.0, Some(1));
            for (k, lane) in ["tint r", "tint g", "tint b"].into_iter().enumerate() {
                widgets::plain_slider(ui, lane, &mut tint[k], 0.0, 2.0, false);
            }
            ui.checkbox(occlude, "casts shadows");
            ui.weak("its shapes give off their own colour × tint × strength, once the 2dini node's light tab has light on");
        }
        GeoNode::SoftBody { blobbiness, pressure, friction, velocity, spin } => {
            egui::ComboBox::from_id_salt(("soft-mat", node))
                .selected_text("material…")
                .show_ui(ui, |ui| {
                    for (name, b, p, f) in engine::geophys::SOFT_MATERIALS {
                        if ui.selectable_label(false, name).clicked() {
                            *blobbiness = b;
                            *pressure = p;
                            *friction = f;
                        }
                    }
                });
            row(ui, rcx, "blobbiness", blobbiness, 0.0, 1.0, Some(1));
            row(ui, rcx, "pressure", pressure, 0.2, 2.0, Some(2));
            row(ui, rcx, "friction", friction, 0.0, 1.0, Some(3));
            row(ui, rcx, "velocity x", &mut velocity[0], -20.0, 20.0, Some(4));
            row(ui, rcx, "velocity y", &mut velocity[1], -20.0, 20.0, Some(5));
            row(ui, rcx, "spin", spin, -20.0, 20.0, Some(6));
            ui.weak("tags every shape for a Physics node downstream (soft mode) · pressure > 1 inflates");
        }
        GeoNode::Physics { settings, bake } => {
            use engine::geophys::{PhysicsMode, MAX_FRAMES};
            ui.horizontal(|ui| {
                for m in PhysicsMode::ALL {
                    ui.selectable_value(&mut settings.mode, m, m.name());
                }
                egui::ComboBox::from_id_salt(("phys-preset", node))
                    .selected_text("preset…")
                    .show_ui(ui, |ui| {
                        for p in &engine::geophys::WORLD_PRESETS {
                            if ui.selectable_label(false, p.name).clicked() {
                                (p.apply)(settings);
                            }
                        }
                    });
            });
            widgets::plain_slider(ui, "gravity", &mut settings.gravity, -30.0, 30.0, false)
                .on_hover_text("m/s² downward; the canvas is 10 m wide");
            widgets::plain_slider(ui, "friction", &mut settings.friction, 0.0, 1.0, false);
            match settings.mode {
                PhysicsMode::Rigid => {
                    widgets::plain_slider(ui, "bounce", &mut settings.bounce, 0.0, 1.0, false);
                    widgets::plain_slider(ui, "density", &mut settings.density, 0.1, 10.0, false);
                }
                PhysicsMode::Soft => {
                    widgets::plain_slider(ui, "blobbiness", &mut settings.blobbiness, 0.0, 1.0, false)
                        .on_hover_text("0 firm and shape-holding, 1 a loose jiggly blob");
                }
                PhysicsMode::Fluid => geofluidui::rows(ui, &mut settings.fluid),
            }
            let mut sub = settings.substeps as f32;
            if widgets::plain_slider(ui, "substeps", &mut sub, 1.0, 32.0, true).changed() {
                settings.substeps = sub.round() as u32;
            }
            ui.horizontal(|ui| {
                ui.checkbox(&mut settings.walls, "canvas walls");
                ui.checkbox(&mut settings.show_colliders, "draw colliders");
            });
            ui.horizontal(|ui| {
                ui.add_sized([52.0, 16.0], egui::Label::new(egui::RichText::new("start").small()));
                ui.add(egui::DragValue::new(&mut settings.start).update_while_editing(false));
                if ui.small_button("here").on_hover_text("start at the frame on screen").clicked() {
                    settings.start = cur_t as i64;
                }
                ui.add_sized([44.0, 16.0], egui::Label::new(egui::RichText::new("frames").small()));
                ui.add(egui::DragValue::new(&mut settings.frames).update_while_editing(false).range(1..=MAX_FRAMES));
            });
            if let Some((done, total)) = physics_live {
                ui.horizontal(|ui| {
                    ui.add(
                        egui::ProgressBar::new(done as f32 / total.max(1) as f32)
                            .desired_width(140.0)
                            .text(format!("simulating {done}/{total}")),
                    );
                    if ui.button("cancel").clicked() {
                        *physics_cancel_req = true;
                    }
                });
                ui.ctx().request_repaint();
            } else {
                ui.horizontal(|ui| {
                    if ui.button("bake").on_hover_text("simulate from the start frame — you watch it run").clicked() {
                        *physics_req = Some((node, true));
                    }
                    if bake.get().is_some() && ui.button("clear").clicked() {
                        *physics_req = Some((node, false));
                    }
                });
            }
            match physics_status {
                Some((Some((frames, bodies, skipped)), stale)) => {
                    ui.weak(format!(
                        "baked: {bodies} bodies, {frames} frames from {}{}",
                        settings.start,
                        if skipped > 0 { format!(" ({skipped} open lines skipped)") } else { String::new() }
                    ));
                    if stale {
                        ui.label(
                            egui::RichText::new("STALE — inputs or settings changed; bake again")
                                .small()
                                .color(egui::Color32::from_rgb(0xff, 0x8a, 0x4d)),
                        );
                    }
                }
                _ => {
                    ui.weak("not baked: the bodies pass through where they stand");
                }
            }
            ui.weak("input 1: bodies (closed shapes, points) · input 2: colliders (closed = solid, open = wall)");
        }
        // A Noise node inside the drawing: its numbers
        // here, its graph behind ⏎ (the field nodes live
        // in there). Its Sliders show as rows, as on the
        // comp's Noise node.
        GeoNode::Noise { graph, scale, seed, speed } => {
            noise_preset_row(ui, ("geo", node), graph);
            row(ui, rcx, "scale", scale, 0.05, 8.0, Some(0));
            row(ui, rcx, "seed", seed, 0.0, 256.0, Some(1));
            row(ui, rcx, "speed", speed, 0.0, 4.0, Some(2));
            for (name, is_input) in graph.exposed() {
                if is_input {
                    continue;
                }
                let mut v = graph.exposed_value(&name).unwrap_or(0.0);
                if widgets::plain_slider(ui, &name, &mut v, -4.0, 4.0, false).changed() {
                    graph.set_exposed(&name, v);
                }
            }
            ui.weak(format!(
                "{} field nodes — open it (↩) to edit the noise",
                graph.nodes.len()
            ));
        }
        GeoNode::Remap { in_lo, in_hi, out_lo, out_hi, clamp } => {
            row(ui, rcx, "in lo", in_lo, -4.0, 4.0, Some(1));
            row(ui, rcx, "in hi", in_hi, -4.0, 4.0, Some(2));
            row(ui, rcx, "out lo", out_lo, -4.0, 4.0, Some(3));
            row(ui, rcx, "out hi", out_hi, -4.0, 4.0, Some(4));
            ui.checkbox(clamp, "clamp");
        }
    }
    }
    hud_drag_out
}
