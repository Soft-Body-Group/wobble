//! Mix: the amount.
//! Moved out of `properties_ui` (lib.rs) unchanged.

use crate::*;

/// Mix: the amount.
pub(crate) fn panel(
    ui: &mut egui::Ui,
    rcx: &mut widgets::RowCtx,
    m: &mut engine::graph::MixParams,
) -> Option<widgets::HudSlider> {
    let mut hud_drag_out: Option<widgets::HudSlider> = None;
    // One number: 0 is all of input 1, 1 all of input 2.
    if let Some(h) = widgets::slider(
        ui,
        rcx,
        "mix",
        &mut m.amount,
        0.0,
        1.0,
        &[anim::Channel::MixAmount],
    ) {
        hud_drag_out = Some(h);
    }
    ui.weak("crossfades input 1 ➡ input 2");
    hud_drag_out
}
