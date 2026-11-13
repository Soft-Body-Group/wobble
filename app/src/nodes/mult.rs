//! Mult: one rgb row.
//! Moved out of `properties_ui` (lib.rs) unchanged.

use crate::*;

/// Mult: one rgb row.
pub(crate) fn panel(
    ui: &mut egui::Ui,
    rcx: &mut widgets::RowCtx,
    m: &mut engine::graph::MultParams,
    working: engine::color::Colorspace,
    ungang: &mut std::collections::BTreeSet<String>,
) -> Option<widgets::HudSlider> {
    let mut hud_drag_out: Option<widgets::HudSlider> = None;
    // The rgb slider: ganged it is the single number a
    // Mult always was; ungang (or hold G) for lanes,
    // and S/D/F slide saturation, orange<->cyan and
    // green<->magenta about the working-space luma.
    let mut gang = !ungang.contains("value");
    if let Some(h) = widgets::rgb_row(
        ui,
        rcx,
        "value",
        &mut m.value,
        &mut gang,
        rgbslider::RgbSpec::scale(
            0.0,
            16.0,
            working.primaries.luma_weights(),
        )
        .with_display(engine::color::primaries_matrix(
            working.primaries,
            engine::color::Primaries::Rec709,
        )),
        &[
            anim::Channel::MultR,
            anim::Channel::MultG,
            anim::Channel::MultB,
        ],
    ) {
        hud_drag_out = Some(h);
    }
    if gang {
        ungang.remove("value");
    } else {
        ungang.insert("value".to_string());
    }
    ui.checkbox(&mut m.alpha, "alpha too")
        .on_hover_text(
            "off: colour only, which is what makes something \
             hot. on: colour and alpha together, a fade",
        );
    ui.weak("image on input 1, matte on input 2 — unwired = everywhere");
    hud_drag_out
}
