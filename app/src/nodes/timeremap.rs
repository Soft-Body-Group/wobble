//! TimeRemap: the lookup curve from comp time to source time.
//! Moved out of `properties_ui` (lib.rs) unchanged.

use crate::*;

/// TimeRemap: the lookup curve from comp time to source time.
pub(crate) fn panel(
    ui: &mut egui::Ui,
    rcx: &mut widgets::RowCtx,
    tr: &mut engine::graph::TimeRemapParams,
    extra_readers: Option<usize>,
) -> Option<widgets::HudSlider> {
    let mut hud_drag_out: Option<widgets::HudSlider> = None;
    // One animatable number: which source frame this
    // timeline frame shows. Key it and the CURVE is the
    // remap — shaped in the curve editor (C) like any
    // other animation, with the same interpolation modes.
    let h = widgets::drags(
        ui,
        rcx,
        "source frame",
        &mut [&mut tr.frame],
        0.25,
        (0.0, 100.0),
        &[anim::Channel::TimeRemapFrame],
    );
    if h.is_some() {
        hud_drag_out = h;
    }
    let animated = rcx
        .anim
        .get(&anim::Channel::TimeRemapFrame)
        .is_some_and(|c| c.keys.len() > 1);
    time_interp_row(ui, &mut tr.interp);
    if animated {
        ui.weak("remapped by its curve — edit the shape in C");
    } else {
        ui.weak(
            "one key holds a freeze; key it twice and the curve \
             between them is the retime (C to shape it)",
        );
    }
    if let Some(n) = extra_readers {
        ui.weak(format!(
            "{n} extra reader{} — the sources above this feed \
             other branches at other frames too",
            if n == 1 { "" } else { "s" }
        ));
    }
    ui.weak(
        "NEAREST source frame, no motion estimation and no \
         blending — RIFE Retime is the interpolating one",
    );
    hud_drag_out
}
