//! Paint: the brush, its presets, the stroke list and the bake.
//! Moved out of `properties_ui` (lib.rs) unchanged.

use crate::*;

/// Paint: the brush, its presets, the stroke list and the bake.
#[allow(clippy::too_many_arguments)]
pub(crate) fn panel(
    ui: &mut egui::Ui,
    pp: &mut engine::graph::PaintParams,
    cur_t: u64,
    node: u64,
    tab: &String,
    working: engine::color::Colorspace,
    brush: &mut state::BrushState,
    prefs: &mut Prefs,
    paint_bake_click: &mut bool,
) {
    use engine::graph::{BrushMode, PaintBlend, PaintOutput, StrokeLife};
    // TWO COLUMNS, like the Spline: two tiles wide, one
    // tall, and the brush alone is six rows. Brush on the
    // left, the node (output, lifetimes, stroke tools)
    // on the right.
    ui.horizontal_top(|ui| {
    let col_w = ((ui.available_width() - 14.0) * 0.5).max(190.0);
    let col_h = ui.available_height();
    ui.allocate_ui(egui::vec2(col_w, col_h), |ui| {
    ui.vertical(|ui| {
    if tab == "pen" {
    // PEN: what pressure does to a NEW stroke (each
    // stroke keeps the map it was painted with). The
    // plot is the response: solid size, dashed opacity,
    // the live line where the pen is pressing now.
    let live = pen::pressure();
    brushes::curve_plot(
        ui,
        &brush.press,
        live,
        egui::vec2(ui.available_width().min(220.0), 90.0),
    );
    match live {
        Some(p) => ui.monospace(format!("pen {:.0}%", p * 100.0)),
        None => ui.weak("no pen pressure: a mouse paints at full"),
    };
    ui.weak(if cfg!(target_os = "linux") {
        "this platform reports no tablet pressure yet"
    } else {
        "Wacom, and Apple Pencil over Sidecar"
    });
    } else if tab == "strokes" {
    // OUTPUT: pixels, or the strokes' coverage as a matte.
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("output").small());
        for (o, name, tip) in [
            (PaintOutput::Rgb, "rgb", "the input with the strokes applied"),
            (
                PaintOutput::Matte,
                "matte",
                "the strokes' coverage, unioned with the input's \
                 alpha — paint over a keyer's matte by hand",
            ),
        ] {
            if ui
                .selectable_label(pp.output == o, name)
                .on_hover_text(tip)
                .clicked()
            {
                pp.output = o;
            }
        }
    });
    // LIFETIME of new strokes: leave it on all frames and
    // the node is a matte painting; set it to this frame
    // and it is a per-frame dust bust.
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("new strokes last").small());
        let cur = pp.life_default;
        egui::ComboBox::from_id_salt(("paint-life", node))
            .selected_text(cur.name())
            .show_ui(ui, |ui| {
                for l in [
                    StrokeLife::All,
                    StrokeLife::Single(cur_t),
                    StrokeLife::Range(cur_t, cur_t + 9),
                ] {
                    let sel = std::mem::discriminant(&cur)
                        == std::mem::discriminant(&l);
                    if ui.selectable_label(sel, l.name()).clicked() {
                        pp.life_default = l;
                    }
                }
            });
        if let StrokeLife::Range(a, b) = &mut pp.life_default {
            crate::widgets::num_field(ui, egui::DragValue::new(a).update_while_editing(false).speed(widgets::rated(ui, 1.0)).prefix("in "),
            );
            crate::widgets::num_field(ui, egui::DragValue::new(b).update_while_editing(false).speed(widgets::rated(ui, 1.0)).prefix("out "),
            );
        }
    });
    let here = pp.strokes.iter().filter(|s| s.life.covers(cur_t)).count();
    let per_frame = pp
        .strokes
        .iter()
        .filter(|s| !matches!(s.life, StrokeLife::All))
        .count();
    ui.horizontal(|ui| {
        ui.monospace(format!(
            "{} strokes · {here} on this frame",
            pp.strokes.len()
        ));
        if per_frame > 0 {
            ui.weak(format!("({per_frame} timed)"));
        }
    });
    ui.horizontal(|ui| {
        if ui.button("undo stroke").clicked() {
            pp.strokes.pop();
        }
        if ui
            .button("clear frame")
            .on_hover_text("drop the strokes visible on THIS frame")
            .clicked()
        {
            pp.strokes.retain(|s| !s.life.covers(cur_t));
        }
        if ui.button("clear").clicked() {
            pp.strokes.clear();
        }
    });
    // Re-time what is already painted: the common fix
    // after painting a dust bust on the wrong setting.
    ui.horizontal(|ui| {
        if ui
            .button("last ➡ this frame")
            .on_hover_text("pin the newest stroke to the playhead")
            .clicked()
        {
            if let Some(s) = pp.strokes.last_mut() {
                s.life = StrokeLife::Single(cur_t);
            }
        }
        if ui
            .button("last ➡ all frames")
            .clicked()
        {
            if let Some(s) = pp.strokes.last_mut() {
                s.life = StrokeLife::All;
            }
        }
    });
    ui.weak("select this node and left-drag in the viewer to paint");
    } else {
    // MODE. One row of six; the cost tier of each is the
    // panel's business further down, not the button's.
    ui.horizontal_wrapped(|ui| {
        for m in BrushMode::ALL {
            if ui
                .selectable_label(brush.mode == m, m.name())
                .clicked()
            {
                brush.mode = m;
            }
        }
    });
    ui.horizontal(|ui| {
        ui.label("brush");
        crate::widgets::num_field_w(ui, widgets::NUM_W + 10.0, egui::DragValue::new(&mut brush.radius).update_while_editing(false)
                .speed(widgets::rated(ui, 0.5))
                .range(1.0..=500.0)
                .max_decimals(1)
                .suffix(" px"),
        )
        .on_hover_text("[ and ] resize · hold F and move");
        ui.add(
            egui::Slider::new(&mut brush.soft, 0.0..=1.0)
                .text("soft"),
        )
        .on_hover_text("shift-[ / shift-] · hold F and move up/down");
        if brush.mode == BrushMode::Color {
            colorpicker::button(
                ui,
                "paint-brush",
                &mut brush.color,
                pick_spec(working, true),
            );
        }
    });
    ui.horizontal(|ui| {
        ui.add_sized(
            [50.0, 16.0],
            egui::Label::new(
                egui::RichText::new("opacity").small(),
            ),
        );
        ui.spacing_mut().slider_width =
            (ui.available_width() - 66.0).max(40.0);
        ui.add(egui::Slider::new(
            &mut brush.color[3],
            0.0..=1.0,
        ))
        .on_hover_text("alt-[ / alt-] · hold shift+F and move");
    });
    // The mode's second knob, labelled with what it is.
    if let Some(label) = brush.mode.strength_label() {
        let (v, range, speed): (&mut f32, _, f32) = match brush.mode {
            BrushMode::Blur => (&mut brush.blur_px, 0.5..=200.0, 0.25),
            BrushMode::Smear => (&mut brush.smear_px, 0.0..=400.0, 0.5),
            _ => (&mut brush.stops, 0.0..=4.0, 0.01),
        };
        ui.horizontal(|ui| {
            ui.add_sized(
                [50.0, 16.0],
                egui::Label::new(egui::RichText::new(label).small()),
            );
            crate::widgets::num_field_w(ui, widgets::NUM_W + 10.0, egui::DragValue::new(v).update_while_editing(false)
                    .speed(widgets::rated(ui, speed))
                    .range(range)
                    .max_decimals(2),
            );
        });
    }
    // SHAPE: an ellipse, turned. Round at aspect 1.
    ui.horizontal(|ui| {
        ui.add_sized(
            [50.0, 16.0],
            egui::Label::new(egui::RichText::new("shape").small()),
        );
        crate::widgets::num_field_w(ui, widgets::NUM_W + 10.0, egui::DragValue::new(&mut brush.aspect).update_while_editing(false)
                .speed(widgets::rated(ui, 0.005))
                .range(0.05..=1.0)
                .max_decimals(2)
                .prefix("aspect "),
        )
        .on_hover_text("minor / major axis · hold ctrl+F and move");
        crate::widgets::num_field_w(ui, widgets::NUM_W + 10.0, egui::DragValue::new(&mut brush.angle).update_while_editing(false)
                .speed(widgets::rated(ui, 0.5))
                .range(-180.0..=180.0)
                .max_decimals(1)
                .suffix("°"),
        )
        .on_hover_text("hold shift+F and move up/down");
    });
    if brush.mode == BrushMode::Clone {
        ui.horizontal(|ui| {
            match brush.clone_off {
                Some(o) => {
                    ui.monospace(format!(
                        "offset {:+.3}, {:+.3}",
                        o[0], o[1]
                    ));
                    if ui.button("clear").clicked() {
                        brush.clone_off = None;
                    }
                }
                None => {
                    ui.weak("hold alt/cmd and move to pull the source off the brush");
                }
            }
        });
    }
    // COST. Colour, dodge and burn are pointwise and
    // free; clone and smear read the input again; blur
    // reads it a ring of taps more. Said here in amber
    // rather than quietly costing frame rate.
    if pp.has_blur() {
        ui.colored_label(
            theme::cost_note(),
            format!(
                "⚠ blur strokes: this node reads its input {}× \
                 per pixel",
                engine::paint::PAINT_BLUR_TAPS + 1 + usize::from(pp.has_source())
            ),
        )
        .on_hover_text(
            "every read is the whole chain above this node \
             again. Bake or pre-render the input if it is \
             heavy; the tap count is fixed, so a wider blur \
             is sparser, not slower.",
        );
    } else if pp.has_source() {
        ui.colored_label(
            theme::cost_note(),
            "⚠ clone/smear strokes: this node reads its input \
             TWICE per pixel",
        )
        .on_hover_text(
            "the source is read live from upstream, so the \
             whole chain above this node runs a second time. \
             Bake or pre-render the input if it is heavy.",
        );
    }
    // BAKE. Slow is fine once it is baked: a GraphBake
    // goes on the output, downstream reads the cache,
    // the red clears, and the strokes stay editable.
    ui.horizontal(|ui| {
        if ui
            .button("bake")
            .on_hover_text(
                "put a Bake on this node's output and start it: \
                 everything downstream reads the cache, the red \
                 clears, and the strokes stay yours to edit (an \
                 edit marks the bake stale; nothing rebakes \
                 behind your back)",
            )
            .clicked()
        {
            *paint_bake_click = true;
        }
        if pp.has_source() || pp.has_effect() {
            ui.weak("slow is fine — bake it");
        }
    });
    ui.weak("[ / ] size · shift soft · alt opacity · hold F + move");
    }
    });
    });
    ui.separator();
    ui.allocate_ui(egui::vec2(col_w, col_h), |ui| {
    ui.vertical(|ui| {
    if tab == "pen" {
    for (label, v, tip) in [
        ("size", &mut brush.press.size, "how far a light touch thins the brush"),
        ("opacity", &mut brush.press.opacity, "how far a light touch fades it (flow)"),
    ] {
        ui.horizontal(|ui| {
            ui.add_sized([50.0, 16.0], egui::Label::new(egui::RichText::new(label).small()));
            ui.spacing_mut().slider_width = (ui.available_width() - 66.0).max(40.0);
            ui.add(egui::Slider::new(v, 0.0..=1.0)).on_hover_text(tip);
        });
    }
    ui.horizontal(|ui| {
        ui.add_sized([50.0, 16.0], egui::Label::new(egui::RichText::new("curve").small()));
        crate::widgets::num_field_w(
            ui,
            widgets::NUM_W + 10.0,
            egui::DragValue::new(&mut brush.press.gamma).update_while_editing(false)
                .speed(widgets::rated(ui, 0.01))
                .range(0.2..=5.0)
                .max_decimals(2),
        )
        .on_hover_text("above 1 the light end is gentler; below 1 a light touch lands hard");
    });
    if ui.small_button("off").on_hover_text("pressure does nothing").clicked() {
        brush.press = engine::graph::PressureMap::OFF;
    }
    } else if tab != "strokes" {
    // BLEND: how a colour stroke lands. Ordered with the
    // rest (a multiply after a stroke darkens it), all in
    // one memoized layer; anything but normal and erase
    // adds one texture read.
    ui.label(egui::RichText::new("blend").small());
    ui.add_enabled_ui(brush.mode == BrushMode::Color, |ui| {
        ui.horizontal_wrapped(|ui| {
            for b in PaintBlend::ALL {
                if ui.selectable_label(brush.blend == b, b.name()).clicked() {
                    brush.blend = b;
                }
            }
        });
    })
    .response
    .on_disabled_hover_text("blend modes are for colour strokes");
    ui.add_space(4.0);
    // BRUSHES: presets, app-wide.
    ui.label(egui::RichText::new("brushes").small());
    let now = brushes::BrushPreset {
        name: String::new(),
        mode: brush.mode,
        radius: brush.radius,
        soft: brush.soft,
        opacity: brush.color[3],
        aspect: brush.aspect,
        angle: brush.angle,
        strength: brush.mode.strength_label().map(|_| match brush.mode {
            BrushMode::Blur => brush.blur_px,
            BrushMode::Smear => brush.smear_px,
            _ => brush.stops,
        }),
        blend: brush.blend,
        press: brush.press,
    };
    let current = brush.presets
        .iter()
        .position(|b| brushes::BrushPreset { name: String::new(), ..b.clone() } == now);
    if let Some(act) = brushes::row(ui, &brush.presets, current) {
        match act {
            brushes::RowAct::Use(i) => {
                let b = brush.presets[i].clone();
                brush.mode = b.mode;
                brush.radius = b.radius;
                brush.soft = b.soft;
                brush.color[3] = b.opacity;
                brush.aspect = b.aspect;
                brush.angle = b.angle;
                brush.blend = b.blend;
                brush.press = b.press;
                if let Some(v) = b.strength {
                    match b.mode {
                        BrushMode::Blur => brush.blur_px = v,
                        BrushMode::Smear => brush.smear_px = v,
                        BrushMode::Dodge | BrushMode::Burn => brush.stops = v,
                        _ => {}
                    }
                }
            }
            act => {
                match act {
                    brushes::RowAct::Replace(i) => {
                        let name = brush.presets[i].name.clone();
                        brush.presets[i] = brushes::BrushPreset { name, ..now };
                    }
                    brushes::RowAct::Remove(i) => {
                        brush.presets.remove(i);
                    }
                    _ => {
                        let name = brushes::fresh_name(&brush.presets, now.mode);
                        brush.presets.push(brushes::BrushPreset { name, ..now });
                    }
                }
                prefs.brush_presets = Some(brush.presets.clone());
                save_prefs(&prefs);
            }
        }
    }
    }
    });
    });
    });
}
