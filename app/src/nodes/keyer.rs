//! Keyer: hands the colour-keyer panel what it needs.
//! Moved out of `properties_ui` (lib.rs) unchanged.

use crate::*;

/// Keyer: hands the colour-keyer panel what it needs.
#[allow(clippy::too_many_arguments)]
pub(crate) fn panel(
    ui: &mut egui::Ui,
    rcx: &mut widgets::RowCtx,
    kp: &mut engine::keyer::KeyerParams,
    node: u64,
    viewing: bool,
    working: engine::color::Colorspace,
    zoom: f32,
    keyer: &mut keyerui::KeyerState,
) -> Option<widgets::HudSlider> {
    let mut hud_drag_out: Option<widgets::HudSlider> = None;
    let mut sel =
        keyer.sel.get(&node).copied().unwrap_or(0);
    let out = keyerui::keyer_ui(
        ui,
        kp,
        rcx,
        &mut sel,
        keyer.cloud.get(&node),
        viewing,
        zoom,
        working,
    );
    keyer.sel.insert(node, sel);
    if out.hud.is_some() {
        hud_drag_out = out.hud;
    }
    if out.want_cloud {
        keyer.want_cloud = Some(node);
    }
    if out.want_view {
        keyer.want_view = Some(node);
    }
    hud_drag_out
}
