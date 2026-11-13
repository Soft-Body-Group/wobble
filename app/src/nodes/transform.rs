//! Transform: translate, rotate, scale, pivot and the filter.
//! Moved out of `properties_ui` (lib.rs) unchanged.

use crate::*;

/// Transform: translate, rotate, scale, pivot and the filter.
#[allow(clippy::too_many_arguments)]
pub(crate) fn panel(
    ui: &mut egui::Ui,
    rcx: &mut widgets::RowCtx,
    t: &mut engine::graph::TransformParams,
    node: u64,
    wired: [bool; 16],
    cropped: &std::collections::HashSet<u64>,
    conform_req: &mut Option<u64>,
    track_import_req: &mut Option<bool>,
) -> Option<widgets::HudSlider> {
    let mut hud_drag_out: Option<widgets::HudSlider> = None;
    use anim::Channel::*;
    let mut h = widgets::drags(
        ui,
        rcx,
        "translate",
        {
            let [tx, ty] = &mut t.translate;
            &mut [tx, ty]
        },
        1.0,
        (-500.0, 500.0),
        &[TranslateX, TranslateY],
    );
    h = h.or(widgets::slider(
        ui,
        rcx,
        "rotate",
        &mut t.rotate,
        -180.0,
        180.0,
        &[Rotate],
    ));
    // Locked (the saved `scale_gang`), x and y keep their
    // ratio — the HUD's ganged scale multiplies both too.
    h = h.or(widgets::drags_locked(
        ui,
        rcx,
        "scale",
        {
            let [sx, sy] = &mut t.scale;
            &mut [sx, sy]
        },
        0.01,
        (0.0, 4.0),
        &[ScaleX, ScaleY],
        &mut t.scale_gang,
    ));
    h = h.or(widgets::drags(
        ui,
        rcx,
        "center",
        {
            let [cx2, cy2] = &mut t.center;
            &mut [cx2, cy2]
        },
        0.005,
        (0.0, 1.0),
        &[CenterX, CenterY],
    ));
    // Shear and taper bend the picture about the pivot,
    // before it turns and scales.
    h = h.or(widgets::drags(
        ui,
        rcx,
        "shear",
        {
            let [sx, sy] = &mut t.shear;
            &mut [sx, sy]
        },
        0.005,
        (-1.0, 1.0),
        &[ShearX, ShearY],
    ));
    h = h.or(widgets::slider(
        ui,
        rcx,
        "taper",
        &mut t.taper,
        -0.95,
        0.95,
        &[Taper],
    ));
    if t.taper != 0.0 {
        h = h.or(widgets::slider(
            ui,
            rcx,
            "taper axis",
            &mut t.taper_angle,
            -180.0,
            180.0,
            &[TaperAngle],
        ));
    }
    ui.horizontal(|ui| {
        if ui
            .add_enabled(
                cropped.contains(&node),
                egui::Button::new("conform to soft crop"),
            )
            .on_hover_text(
                "pivot to the centre of the SoftCrop's box upstream; \
                 the picture does not move",
            )
            .on_disabled_hover_text("no SoftCrop upstream")
            .clicked()
        {
            *conform_req = Some(node);
        }
        // The corner pin a ctrl-drag on a corner handle
        // leaves behind.
        if t.pin != engine::graph::UNIT_QUAD
            && ui
                .button("reset pin")
                .on_hover_text("drop the corner pin (ctrl-drag a corner to make one)")
                .clicked()
        {
            t.pin = engine::graph::UNIT_QUAD;
        }
        if ui.button("import track…").on_hover_text("key the corner pin from exported corner pin or transform text").clicked() {
            *track_import_req = Some(false);
        }
    });
    // A matte on the modulate (~) port says how much of
    // the move applies at each pixel; how it blends in is
    // this choice, a uniform (no recompile).
    if wired[1] {
        modulate_blend_row(ui, &mut t.mod_blend);
    }
    if h.is_some() {
        hud_drag_out = h;
    }
    hud_drag_out
}
