//! Shader: the user shader's declared params and its code-window buttons.
//! Moved out of `properties_ui` (lib.rs) unchanged.

use crate::*;

/// Shader: the user shader's declared params and its code-window buttons.
pub(crate) fn panel(
    ui: &mut egui::Ui,
    rcx: &mut widgets::RowCtx,
    sp: &mut engine::graph::ShaderParams,
    node: u64,
    shader_dirty: bool,
    shader_req: &mut Option<shaderui::Req>,
) -> Option<widgets::HudSlider> {
    let mut hud_drag_out: Option<widgets::HudSlider> = None;
    let h = shaderui::params_ui(
        ui,
        rcx,
        node,
        sp,
        shader_dirty,
        shader_req,
    );
    if h.is_some() {
        hud_drag_out = h;
    }
    hud_drag_out
}
