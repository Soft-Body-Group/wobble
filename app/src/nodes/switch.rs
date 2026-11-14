//! Switch: which input is live, and how many there are.
//! Moved out of `properties_ui` (lib.rs) unchanged.

use crate::*;

/// Switch: which input is live, and how many there are.
pub(crate) fn panel(
    ui: &mut egui::Ui,
    rcx: &mut widgets::RowCtx,
    sw: &mut engine::graph::SwitchParams,
) -> Option<widgets::HudSlider> {
    let mut hud_drag_out: Option<widgets::HudSlider> = None;
    // `which` is animatable, and that is the point: key
    // it to cut between sources on a frame.
    let last = (sw.n() - 1) as f32;
    let mut w = sw.which as f32;
    let h = widgets::drags(
        ui,
        rcx,
        "which",
        &mut [&mut w],
        0.05,
        (0.0, last),
        &[anim::Channel::SwitchWhich],
    );
    // An input index is a whole number: a curve through it
    // snaps at the midpoint, and Hold keys make the cut
    // land exactly where it is keyed.
    sw.which = w.round().clamp(0.0, last) as u32;
    if h.is_some() {
        hud_drag_out = h;
    }
    ui.horizontal(|ui| {
        ui.label("inputs");
        let mut n = sw.inputs;
        if ui
            .add(egui::DragValue::new(&mut n).update_while_editing(false).speed(widgets::rated(ui, 0.05)).range(2..=16))
            .changed()
        {
            sw.inputs = n;
        }
    });
    ui.weak(
        "only the live input is compiled, so the others cost \
         nothing — no decode, no filter passes",
    );
    hud_drag_out
}
