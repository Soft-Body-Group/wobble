//! Reformat: the target format and how the picture fits it.
//! Moved out of `properties_ui` (lib.rs) unchanged.

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
        ui.label("format");
        widgets::size_fields(ui, egui::Id::new(("rf-size", key)), &mut rf.size);
    });
    ui.horizontal(|ui| {
        ui.label("resize");
        egui::ComboBox::from_id_salt(("rf-mode", key))
            .selected_text(rf.mode.name())
            .width(96.0)
            .show_ui(ui, |ui| {
                for m in engine::graph::ResizeMode::ALL {
                    ui.selectable_value(&mut rf.mode, m, m.name());
                }
            });
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
        let [fx, fy] = engine::codegen::reformat_scale(
            rf,
            [sw as f32, sh as f32],
            [rf.size.0.max(1) as f32, rf.size.1.max(1) as f32],
        );
        ui.monospace(format!(
            "{sw}x{sh} ➡ {}x{}  ({:.3}× {:.3}×)",
            rf.size.0, rf.size.1, fx, fy
        ));
    } else {
        ui.weak("nothing loaded upstream");
    }
    ui.weak("one sample, no resampling pass — the map is a coordinate scale");
    hud_drag_out
}
