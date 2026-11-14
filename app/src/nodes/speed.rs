//! Speed: a rate about an anchor frame.
//! Moved out of `properties_ui` (lib.rs) unchanged.

use crate::*;

/// Speed: a rate about an anchor frame.
pub(crate) fn panel(
    ui: &mut egui::Ui,
    sp: &mut engine::graph::SpeedParams,
    cur_t: u64,
    extra_readers: Option<usize>,
) {
    // A constant rate around a frame that stays put.
    ui.horizontal(|ui| {
        ui.label("speed");
        crate::widgets::num_field_w(ui, widgets::NUM_W + 14.0, egui::DragValue::new(&mut sp.speed).update_while_editing(false)
                .speed(widgets::rated(ui, 0.01))
                .range(-8.0..=8.0),
        )
        .on_hover_text("0.5 = half speed; negative runs backwards");
        ui.label("anchor");
        crate::widgets::num_field_w(ui, widgets::NUM_W + 14.0, egui::DragValue::new(&mut sp.anchor).update_while_editing(false).speed(widgets::rated(ui, 0.25)),
        )
        .on_hover_text("the frame that stays put while the rest stretches");
    });
    time_interp_row(ui, &mut sp.interp);
    let src = sp.anchor + (cur_t as f32 - sp.anchor) * sp.speed;
    let frac = src - src.floor();
    if sp.interp == engine::graph::TimeInterp::Blend && frac > 1e-3 {
        ui.weak(format!(
            "frame {cur_t} reads source {src:.2} — {:.0}% of the way \
             from {} to {}",
            frac * 100.0,
            src.floor(),
            src.floor() + 1.0
        ));
    } else {
        ui.weak(format!("frame {cur_t} reads source {src:.2}"));
    }
    if let Some(n) = extra_readers {
        ui.weak(format!(
            "{n} extra reader{} — the sources above this feed \
             other branches at other frames too",
            if n == 1 { "" } else { "s" }
        ));
    }
}
