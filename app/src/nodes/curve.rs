//! Curve: the rgb curve canvas.
//! Moved out of `properties_ui` (lib.rs) unchanged.

use crate::*;

/// Curve: the rgb curve canvas.
pub(crate) fn panel(
    ui: &mut egui::Ui,
    cv: &mut engine::graph::CurveParams,
    node: u64,
    curve_rect_out: &mut Option<(egui::Rect, Vec<egui::Pos2>)>,
) {
    // Full-panel curve editor: drag points (several at
    // once after a marquee), shift-click adds, right-click
    // or Delete removes, linear or monotone-cubic. With
    // "split rgb" each channel gets its own curve and the
    // other two show as ghosts.
    use engine::graph::CurveInterp;
    let cid = ui.id().with(("curve", node));
    let mut chan =
        ui.data_mut(|d| d.get_temp::<usize>(cid).unwrap_or(0));
    ui.horizontal(|ui| {
        egui::ComboBox::from_id_salt(("curve-interp", node))
            .width(78.0)
            .selected_text(cv.interp.name())
            .show_ui(ui, |ui| {
                for m in CurveInterp::ALL {
                    ui.selectable_value(&mut cv.interp, m, m.name());
                }
            });
        if ui
            .selectable_label(cv.split, "split rgb")
            .on_hover_text(
                "give each channel its own curve\n\
                 (it starts from the ganged one)",
            )
            .clicked()
        {
            cv.split = !cv.split;
            if cv.split {
                for c in cv.rgb.iter_mut() {
                    if c.is_empty() {
                        *c = cv.points.clone();
                    }
                }
            }
        }
        if cv.split {
            ui.separator();
            for (i, name) in ["r", "g", "b"].iter().enumerate() {
                if ui.selectable_label(chan == i, *name).clicked() {
                    chan = i;
                }
            }
        }
    });
    let cols = [
        egui::Color32::from_rgb(0xe2, 0x5c, 0x5c),
        egui::Color32::from_rgb(0x5c, 0xd6, 0x7a),
        egui::Color32::from_rgb(0x62, 0x9c, 0xe8),
    ];
    let interp = cv.interp;
    let split = cv.split;
    let ghost_pts: Vec<(Vec<[f32; 2]>, egui::Color32)> = if split {
        (0..3)
            .filter(|i| *i != chan)
            .map(|i| (cv.channel(i).clone(), cols[i].gamma_multiply(0.5)))
            .collect()
    } else {
        Vec::new()
    };
    let ghosts: Vec<curveui::Ghost<'_>> = ghost_pts
        .iter()
        .map(|(p, c)| curveui::Ghost { pts: p, color: *c, fill: false })
        .collect();
    let cfg = curveui::CurveCfg {
        size: egui::vec2(
            ui.available_width(),
            (ui.available_height() - 48.0).max(120.0),
        ),
        // Framed on 0..1, but nothing is clamped there:
        // scroll out to grade the highlights.
        view: [0.0, cv.high.max(1.0), 0.0, 1.0],
        interp,
        // A colour curve extrapolates past its ends.
        flat_ends: false,
        min_points: 2,
        tint: if split {
            cols[chan]
        } else {
            theme::curve_line()
        },
        ..Default::default()
    };
    let target = if split {
        &mut cv.rgb[chan]
    } else {
        &mut cv.points
    };
    if target.is_empty() {
        *target = engine::graph::CurveParams::default().points;
    }
    let out =
        curveui::curve_edit(ui, cid.with(chan), &cfg, &ghosts, target);
    *curve_rect_out = Some((out.response.rect, out.screen.clone()));
    // The high point: type where the top of the curve
    // should sit and "set" rebuilds it as a straight
    // line to there. It THROWS AWAY the shape — that is
    // the point of it, a one-click white point.
    ui.horizontal(|ui| {
        ui.add_sized(
            [46.0, 16.0],
            egui::Label::new(egui::RichText::new("high").small()),
        );
        ui.add(
            egui::DragValue::new(&mut cv.high).update_while_editing(false)
                .speed(widgets::rated(ui, 0.05))
                .range(1e-3..=1e6),
        )
        .on_hover_text("the input value that should map to 1.0");
        if ui
            .button("set")
            .on_hover_text("replace the curve with a straight line\n\
                            from 0 to the high point (drops its points)")
            .clicked()
        {
            let line = engine::graph::CurveParams::ramp(cv.high);
            if split {
                cv.rgb[chan] = line;
            } else {
                cv.points = line;
                cv.rgb = Default::default();
            }
        }
        if ui.button("reset").clicked() {
            if split {
                cv.rgb[chan] =
                    engine::graph::CurveParams::default().points;
            } else {
                // A reset keeps the pipe it sits in.
                *cv = engine::graph::CurveParams {
                    interp,
                    matte: cv.matte,
                    ..Default::default()
                };
            }
        }
    });
    ui.weak("shift-click add · alt-click remove · drag bg selects");
    ui.data_mut(|d| d.insert_temp(cid, chan));
}
