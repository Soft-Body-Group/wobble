//! RangeRemap: a source range laid onto a comp range.
//! Moved out of `properties_ui` (lib.rs) unchanged.

use crate::*;

/// RangeRemap: a source range laid onto a comp range.
#[allow(clippy::too_many_arguments)]
pub(crate) fn panel(
    ui: &mut egui::Ui,
    rp: &mut engine::graph::RangeRemapParams,
    cur_t: u64,
    extra_readers: Option<usize>,
    input_range: Option<(u64, u64)>,
    node: u64,
    script: (i64, i64),
    remap_seek: &mut Option<i64>,
) {
    // Source in..out played over target in..out.
    *remap_seek = remap_bars(ui, node, rp, input_range, script, cur_t as i64);
    for (label, pair, tip) in [
        ("source", &mut rp.src, "the upstream frames, in and out"),
        ("target", &mut rp.dst, "where they play on this timeline, in and out"),
    ] {
        ui.horizontal(|ui| {
            ui.label(label).on_hover_text(tip);
            for (end, v) in [("in", &mut pair.0), ("out", &mut pair.1)] {
                ui.label(end);
                crate::widgets::num_field_w(
                    ui,
                    widgets::NUM_W + 14.0,
                    egui::DragValue::new(v)
                        .update_while_editing(false)
                        .speed(widgets::rated(ui, 0.5)),
                );
            }
        });
    }
    ui.horizontal(|ui| {
        if let Some((a, b)) = input_range {
            if ui
                .small_button("source from input")
                .on_hover_text(format!("the input's frames: {a}–{b}"))
                .clicked()
            {
                rp.src = (a as f32, b as f32);
            }
        }
        if ui
            .small_button("target = source")
            .on_hover_text("play it at its own speed, where it was")
            .clicked()
        {
            rp.dst = rp.src;
        }
    });
    time_interp_row(ui, &mut rp.interp);
    let (k, b) = rp.affine();
    let src = k * cur_t as f32 + b;
    if rp.in_off || rp.out_off {
        ui.weak(format!(
            "slip {b:+} (no retime) · frame {cur_t} reads source {src:.2}"
        ));
    } else {
        ui.weak(format!(
            "speed {:.0}% · frame {cur_t} reads source {src:.2}",
            k * 100.0
        ));
    }
    if let Some(n) = extra_readers {
        ui.weak(format!(
            "{n} extra reader{} — the sources above this feed \
             other branches at other frames too",
            if n == 1 { "" } else { "s" }
        ));
    }
}
