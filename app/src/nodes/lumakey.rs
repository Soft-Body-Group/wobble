//! LumaKey: the band strip and the selected band's curve.
//! Moved out of `properties_ui` (lib.rs) unchanged.

use crate::*;

/// LumaKey: the band strip and the selected band's curve.
pub(crate) fn panel(
    ui: &mut egui::Ui,
    k: &mut engine::graph::LumaKeyParams,
    node: u64,
    zoom: f32,
    view_band_req: &mut Option<(u64, u32)>,
) {
    // Locked bands: one partition of the luma axis, every
    // boundary shared with the band next to it. There is
    // no second model — a key loaded from an older setup
    // has already been migrated (`adopt_ranges`), and one
    // that somehow has not is migrated here rather than
    // shown a different editor.
    if k.ranges.is_none() {
        k.adopt_ranges(engine::graph::LUMA_SPAN);
    }
    let bid = ui.id().with(("lk-band", node));
    let mut sel =
        ui.data_mut(|d| d.get_temp::<usize>(bid).unwrap_or(0));
    let out = bandui::luma_panel(
        ui,
        k,
        node,
        &mut sel,
        zoom,
        &bandui::BandOpts::KEYER,
    );
    ui.data_mut(|d| d.insert_temp(bid, sel));
    if let Some(b) = out.view_band {
        *view_band_req = Some((node, b));
    }
}
