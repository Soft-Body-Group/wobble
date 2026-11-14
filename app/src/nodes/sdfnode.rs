//! One node of an SDF graph, edited from the rack.
//! Moved out of `properties_ui` (lib.rs) unchanged.

use crate::*;

/// One node of an SDF graph, edited from the rack.
#[allow(clippy::too_many_arguments)]
pub(crate) fn panel(
    ui: &mut egui::Ui,
    rcx: &mut widgets::RowCtx,
    n: &mut engine::sdf::SdfNode,
    node: u64,
    wired: [bool; 16],
    working: engine::color::Colorspace,
    spline_sel_node: Option<u64>,
    spline_sel: &std::collections::BTreeSet<usize>,
) -> Option<widgets::HudSlider> {
    let mut hud_drag_out: Option<widgets::HudSlider> = None;
    // The points picked in the viewer, for the tension row.
    let picked: Vec<usize> = if spline_sel_node == Some(node) {
        spline_sel.iter().copied().collect()
    } else {
        Vec::new()
    };
    let spec = pick_spec(working, true);
    if let Some(h) = sdfui::node_panel(ui, rcx, node, n, &wired, &picked, spec) {
        hud_drag_out = Some(h);
    }
    hud_drag_out
}
