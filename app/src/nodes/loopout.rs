//! LoopOut: the iteration count, and a plain statement of what will run.
//! Moved out of `properties_ui` (lib.rs) unchanged.

use crate::*;

/// LoopOut: the iteration count, and a plain statement of what will run.
pub(crate) fn panel(
    ui: &mut egui::Ui,
    rcx: &mut widgets::RowCtx,
    lp: &mut engine::graph::LoopParams,
    loop_state: Option<u8>,
) -> Option<widgets::HudSlider> {
    let mut hud_drag_out: Option<widgets::HudSlider> = None;
    let h = widgets::drags(
        ui,
        rcx,
        "iterations",
        &mut [&mut lp.n],
        0.25,
        (1.0, engine::graph::LOOP_MAX),
        &[anim::Channel::LoopN],
    );
    if h.is_some() {
        hud_drag_out = h;
    }
    // Say exactly what will run. The engine degrades a
    // loop it cannot iterate; the panel must not let
    // that pass silently.
    if loop_state == Some(0) {
        ui.weak(format!(
            "runs the body {}x per frame (a full pass each — \
             watch the cost)",
            lp.n.round().clamp(1.0, engine::graph::LOOP_MAX)
        ));
    } else if loop_state == Some(1) {
        ui.colored_label(
            theme::error(),
            "no LoopIn paired — body runs ONCE",
        );
        ui.weak(
            "drag a wire from a downstream output back into an \
             upstream input and the pair is made for you",
        );
    } else {
        ui.colored_label(
            theme::error(),
            "a pass node (blur/glow/…) sits between LoopIn and \
             here — body runs ONCE",
        );
        ui.weak(
            "a pass can only run once a frame; keep the tap's \
             chain pointwise (grade, transform, merge, distort)",
        );
    }
    ui.weak("input 1 seeds iteration 0; unwired = transparent");
    hud_drag_out
}
