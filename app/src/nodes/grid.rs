//! Grid: the lines, their spacing and the two colours.
//! Moved out of `properties_ui` (lib.rs) unchanged.

use crate::*;

/// Grid: the lines, their spacing and the two colours.
pub(crate) fn panel(
    ui: &mut egui::Ui,
    rcx: &mut widgets::RowCtx,
    g: &mut engine::graph::GridParams,
    node: u64,
    working: engine::color::Colorspace,
) -> Option<widgets::HudSlider> {
    let mut hud_drag_out: Option<widgets::HudSlider> = None;
    use anim::Channel::*;
    widgets::gen_format_row(ui, ("grid-format", node), &mut g.size, &mut g.inherit);
    let mut h = widgets::drags_linked(
        ui,
        rcx,
        "spacing",
        {
            let [sx, sy] = &mut g.spacing;
            &mut [sx, sy]
        },
        0.5,
        (1.0, 512.0),
        &[GridSpacingX, GridSpacingY],
    );
    h = h.or(widgets::slider(
        ui,
        rcx,
        "line",
        &mut g.width,
        0.0,
        32.0,
        &[GridWidth],
    ));
    h = h.or(widgets::drags(
        ui,
        rcx,
        "offset",
        {
            let [ox, oy] = &mut g.offset;
            &mut [ox, oy]
        },
        0.5,
        (-256.0, 256.0),
        &[GridOffsetX, GridOffsetY],
    ));
    ui.horizontal(|ui| {
        ui.label("colours");
        let spec = pick_spec(working, true);
        colorpicker::button(ui, (node, "grid-line"), &mut g.line, spec)
            .on_hover_text("line");
        colorpicker::button(ui, (node, "grid-back"), &mut g.back, spec)
            .on_hover_text("background");
    });
    if h.is_some() {
        hud_drag_out = h;
    }
    ui.weak("generated at its own format — a source, like a Load");
    hud_drag_out
}
