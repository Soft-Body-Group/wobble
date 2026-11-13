//! One node of a 2dini field graph.
//! Moved out of `properties_ui` (lib.rs) unchanged.

use crate::*;

/// One node of a 2dini field graph.
pub(crate) fn panel(
    ui: &mut egui::Ui,
    rcx: &mut widgets::RowCtx,
    f: &mut engine::field::FieldNode,
    node: u64,
    wired: [bool; 16],
    spline_sel_node: Option<u64>,
    spline_sel: &std::collections::BTreeSet<usize>,
) -> Option<widgets::HudSlider> {
    let mut hud_drag_out: Option<widgets::HudSlider> = None;
    // The points picked in the viewer, for the tension row.
    let roto_picked: Vec<usize> = if spline_sel_node == Some(node) {
        spline_sel.iter().copied().collect()
    } else {
        Vec::new()
    };
    use engine::field::*;
    // A socket that is WIRED overrides its literal, so
    // the number below it is no longer being read. Say
    // so rather than leaving a live-looking control that
    // does nothing — that confusion is the one this
    // paradigm invites.
    let names: Vec<&'static str> =
        f.inputs().iter().map(|i| i.name).collect();
    let driven: Vec<bool> = (0..names.len())
        .map(|i| wired.get(i).copied().unwrap_or(false))
        .collect();
    // Sockets ARE channels here (`Channel::Field(i)`), so
    // a socket row is the standard slider row: key dot,
    // HUD drag-out, patch jack — an fBm's gain animates
    // and cables exactly like a Grade's. A socket wired
    // to another FIELD is different again: that wire
    // overrides the literal per pixel, so the row is
    // greyed and says so rather than sitting there live.
    let mut row = |ui: &mut egui::Ui,
                   rcx: &mut widgets::RowCtx,
                   label: &str,
                   v: &mut f32,
                   lo: f32,
                   hi: f32,
                   idx: Option<usize>| {
        let on = idx.is_some_and(|i| driven[i]);
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
                ui.weak("driven by a field");
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
            // No socket, no channel: a plain number.
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
    match f {
        FieldNode::Coord { kind, scale } => {
            ui.horizontal(|ui| {
                for k in CoordKind::ALL {
                    ui.selectable_value(kind, k, k.name());
                }
            });
            row(ui, rcx, "scale", scale, 0.001, 2.0, None);
        }
        // The data input: no widget of its own — outside,
        // it is a jack on the Noise node.
        FieldNode::Input { name, value } => {
            ui.horizontal(|ui| {
                ui.add_sized(
                    [76.0, 16.0],
                    egui::Label::new(egui::RichText::new("name").small()),
                );
                ui.text_edit_singleline(name);
            });
            ui.add_enabled(false, egui::DragValue::new(value).update_while_editing(false));
            ui.weak("a jack on the Noise node, driven from the comp's link mode");
        }
        // The slider: its value is channel 0, so it keys
        // like any param, and it wires into any socket.
        // Its NAME is how it appears on the Noise node.
        FieldNode::Const { value, name } => {
            ui.horizontal(|ui| {
                ui.add_sized(
                    [76.0, 16.0],
                    egui::Label::new(egui::RichText::new("name").small()),
                );
                ui.text_edit_singleline(name);
            });
            if let Some(h) = widgets::slider(
                ui,
                rcx,
                "value",
                value,
                -4.0,
                4.0,
                &[anim::Channel::Field(0)],
            ) {
                hud_drag_out = Some(h);
            }
            ui.weak("wire the output into any socket");
        }
        FieldNode::Fbm {
            mode,
            scale,
            offset,
            lacunarity,
            gain,
            seed,
            octaves,
            speed,
            uniform,
        } => {
            ui.horizontal(|ui| {
                for m in FieldFbmMode::ALL {
                    ui.selectable_value(mode, m, m.name());
                }
            });
            row(ui, rcx, "scale", uniform, 0.001, 8.0, Some(7));
            row(ui, rcx, "scale x", &mut scale[0], 0.001, 2.0, Some(0));
            row(ui, rcx, "scale y", &mut scale[1], 0.001, 2.0, Some(1));
            row(ui, rcx, "offset x", &mut offset[0], -4.0, 4.0, Some(2));
            row(ui, rcx, "offset y", &mut offset[1], -4.0, 4.0, Some(3));
            row(ui, rcx, "lacunarity", lacunarity, 1.0, 4.0, Some(4));
            row(ui, rcx, "gain", gain, 0.0, 1.0, Some(5));
            row(ui, rcx, "seed", seed, 0.0, 256.0, Some(6));
            row(ui, rcx, "speed x", speed, -4.0, 4.0, None);
            {
                let mut o = *octaves as f32;
                if widgets::plain_slider(
                    ui,
                    "octaves",
                    &mut o,
                    1.0,
                    engine::field::FBM_MAX_OCTAVES as f32,
                    true,
                )
                    .on_hover_text(
                        "not drivable: it is the loop bound, and a \
                         per-pixel octave count costs the worst case \
                         anyway and bands visibly",
                    )
                    .changed()
                {
                    *octaves = o.round() as u32;
                }
            }
        }
        FieldNode::Voronoi { mode, scale, jitter, seed, speed } => {
            ui.horizontal(|ui| {
                for m in VoronoiMode::ALL {
                    ui.selectable_value(mode, m, m.name());
                }
            });
            row(ui, rcx, "scale x", &mut scale[0], 0.001, 2.0, Some(0));
            row(ui, rcx, "scale y", &mut scale[1], 0.001, 2.0, Some(1));
            row(ui, rcx, "jitter", jitter, 0.0, 1.0, Some(2));
            row(ui, rcx, "seed", seed, 0.0, 256.0, Some(3));
            row(ui, rcx, "speed x", speed, -4.0, 4.0, None);
        }
        FieldNode::Math { op, a, b } => {
            egui::ComboBox::from_id_salt(("fld-op", node))
                .selected_text(op.name())
                .show_ui(ui, |ui| {
                    for m in MathOp::ALL {
                        ui.selectable_value(op, m, m.name());
                    }
                });
            row(ui, rcx, "a", a, -4.0, 4.0, Some(0));
            row(ui, rcx, "b", b, -4.0, 4.0, Some(1));
        }
        FieldNode::Mix { a, b, t } => {
            row(ui, rcx, "a", a, -4.0, 4.0, Some(0));
            row(ui, rcx, "b", b, -4.0, 4.0, Some(1));
            row(ui, rcx, "t", t, 0.0, 1.0, Some(2));
        }
        FieldNode::Remap { in_lo, in_hi, out_lo, out_hi, clamp } => {
            row(ui, rcx, "in low", in_lo, -4.0, 4.0, None);
            row(ui, rcx, "in high", in_hi, -4.0, 4.0, None);
            row(ui, rcx, "out low", out_lo, -4.0, 4.0, None);
            row(ui, rcx, "out high", out_hi, -4.0, 4.0, None);
            ui.checkbox(clamp, "clamp");
        }
        FieldNode::Warp { mode, amount } => {
            ui.horizontal(|ui| {
                for m in WarpMode::ALL {
                    ui.selectable_value(mode, m, m.name());
                }
            });
            row(ui, rcx, "amount", amount, -2.0, 2.0, Some(if *mode == WarpMode::Curl { 2 } else { 3 }));
            ui.weak(if *mode == WarpMode::Curl {
                "input 2 is a POTENTIAL: the offset is its curl, so the \
                 domain swirls rather than pooling"
            } else {
                "inputs 2 and 3 offset x and y"
            });
        }
        FieldNode::Seed { shift } => {
            row(ui, rcx, "shift", shift, 0.0, 256.0, Some(1));
            ui.weak("reseeds everything above it");
        }
        FieldNode::Spline { points, mode, feather, offset } => {
            use engine::graph::SplineMode;
            ui.horizontal(|ui| {
                egui::ComboBox::from_id_salt(("roto-mode", node))
                    .selected_text(mode.name())
                    .show_ui(ui, |ui| {
                        for m in SplineMode::ALL {
                            ui.selectable_value(mode, m, m.name());
                        }
                    });
                ui.weak(format!("{} points", points.len()));
            });
            row(ui, rcx, "offset x", &mut offset[0], -1.0, 1.0, Some(0));
            row(ui, rcx, "offset y", &mut offset[1], -1.0, 1.0, Some(1));
            row(ui, rcx, "feather", feather, 0.0, 4.0, Some(2));
            tension_row(ui, points, *mode, &roto_picked);
            ui.weak(if points.is_empty() {
                "select this node and click in the viewer to draw"
            } else {
                "1 inside, 0 outside · edit points in the viewer"
            });
        }
        FieldNode::Shape { kind, center, size, feather } => {
            ui.horizontal(|ui| {
                for k in ShapeKind::ALL {
                    ui.selectable_value(kind, k, k.name());
                }
            });
            row(ui, rcx, "centre x", &mut center[0], -1.0, 2.0, Some(0));
            row(ui, rcx, "centre y", &mut center[1], -1.0, 2.0, Some(1));
            row(ui, rcx, "size x", &mut size[0], 0.0, 2.0, Some(2));
            row(ui, rcx, "size y", &mut size[1], 0.0, 2.0, Some(3));
            row(ui, rcx, "feather", feather, 0.0, 1.0, Some(4));
            ui.weak("1 inside, 0 outside — a mask made here");
        }
        FieldNode::MaskIn => {
            ui.weak(
                "the matte wired into this Noise node, read \
                 per pixel — mix through it and the field \
                 changes only where the mask is",
            );
        }
        FieldNode::Output => {
            ui.weak(
                "what this graph evaluates to — the Noise \
                 node renders whatever reaches here",
            );
        }
        FieldNode::Gradient { axis, step } => {
            ui.horizontal(|ui| {
                for a in Axis::ALL {
                    ui.selectable_value(axis, a, a.name());
                }
            });
            row(ui, rcx, "step", step, 0.0001, 0.05, None);
        }
    }
    if !names.is_empty() {
        ui.weak(format!("sockets: {}", names.join(", ")));
    }
    hud_drag_out
}
