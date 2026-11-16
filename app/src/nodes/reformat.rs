//! Reformat: the target format and how the picture fits it.
//! The format is absolute (`size`) or a factor of the input's (`by_factor`).

use crate::*;

/// Reformat: the target format and how the picture fits it.
pub(crate) fn panel(
    ui: &mut egui::Ui,
    rcx: &mut widgets::RowCtx,
    rf: &mut engine::graph::ReformatParams,
    key: u64,
    reformat_src: Option<(u32, u32)>,
) -> Option<widgets::HudSlider> {
    let mut hud_drag_out: Option<widgets::HudSlider> = None;
    ui.horizontal(|ui| {
        ui.label("size by");
        ui.selectable_value(&mut rf.by_factor, false, "format")
            .on_hover_text("an absolute width x height");
        ui.selectable_value(&mut rf.by_factor, true, "factor")
            .on_hover_text("the input's format times a number: 0.5 is half res");
    });
    if rf.by_factor {
        ui.horizontal(|ui| {
            ui.label("factor");
            widgets::num_field(
                ui,
                egui::DragValue::new(&mut rf.factor).speed(0.01).range(0.01..=16.0).max_decimals(4),
            )
            .on_hover_text("0.5 = half res, 2 = double; arithmetic works (1/3)");
            for (label, f) in [("0.25", 0.25), ("0.5", 0.5), ("1", 1.0), ("2", 2.0)] {
                if ui.small_button(label).clicked() {
                    rf.factor = f;
                }
            }
        });
    } else {
        ui.horizontal(|ui| {
            ui.label("format");
            widgets::size_fields(ui, egui::Id::new(("rf-size", key)), &mut rf.size);
        });
    }
    ui.horizontal(|ui| {
        if !rf.by_factor {
            ui.label("resize");
            egui::ComboBox::from_id_salt(("rf-mode", key))
                .selected_text(rf.mode.name())
                .width(96.0)
                .show_ui(ui, |ui| {
                    for m in engine::graph::ResizeMode::ALL {
                        ui.selectable_value(&mut rf.mode, m, m.name());
                    }
                });
        }
        ui.checkbox(&mut rf.center, "center")
            .on_hover_text("else the image anchors top-left");
    });
    if let Some(h) = widgets::slider(
        ui,
        rcx,
        "scale",
        &mut rf.scale,
        0.05,
        4.0,
        &[anim::Channel::ReformatScale],
    ) {
        hud_drag_out = Some(h);
    }
    // Source dimensions make the fit legible without
    // making the user do the arithmetic.
    if let Some((sw, sh)) = reformat_src {
        let (ow, oh) = rf.out_size((sw, sh));
        let [fx, fy] = engine::codegen::reformat_scale(
            rf,
            [sw as f32, sh as f32],
            [ow.max(1) as f32, oh.max(1) as f32],
        );
        ui.monospace(format!("{sw}x{sh} ➡ {ow}x{oh}  ({fx:.3}× {fy:.3}×)"));
    } else {
        ui.weak("nothing loaded upstream");
    }
    ui.weak("one sample, no resampling pass — the map is a coordinate scale");
    hud_drag_out
}
