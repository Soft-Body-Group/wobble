//! Bake: the bake rows, where it lands and what is on disk.
//! Moved out of `properties_ui` (lib.rs) unchanged.

use crate::*;

/// Everything the graphbake panel borrows from the properties pass.
pub(crate) struct Cx<'a> {
    pub mask: &'a mut bool,
    pub matte_half: &'a mut bool,
    pub key: u64,
    pub gb_current: bool,
    pub gb_key: Option<u64>,
    pub gb_dir: &'a Option<std::path::PathBuf>,
    pub gb_proxy_dir: &'a Option<std::path::PathBuf>,
    pub gb_target: &'a Option<std::path::PathBuf>,
    pub playback_dirty: &'a mut bool,
    pub proxy: f32,
    pub loaded: &'a HashMap<u64, LoadedSeq>,
    pub load_errors: &'a HashMap<u64, String>,
    pub graph_bake: &'a Option<SubtreeBake>,
    pub graph_bake_cancel: &'a mut bool,
    pub graph_bake_clear: &'a mut bool,
    pub graph_bake_click: &'a mut bool,
}

/// Bake: the bake rows, where it lands and what is on disk.
pub(crate) fn panel(
    ui: &mut egui::Ui,
    cx: Cx,
) {
    let Cx {
        mask,
        matte_half,
        key,
        gb_current,
        gb_key,
        gb_dir,
        gb_proxy_dir,
        gb_target,
        playback_dirty,
        proxy,
        loaded,
        load_errors,
        graph_bake,
        graph_bake_cancel,
        graph_bake_clear,
        graph_bake_click,
    } = cx;
    if *mask
        && ui
            .checkbox(matte_half, "16-bit matte")
            .on_hover_text(
                "play the matte back as half float, not 8-bit: for a soft \
                 gradient over a large area that would band (no re-bake)",
            )
            .changed()
    {
        *playback_dirty = true;
    }
    let clicks = bake_rows(ui, &BakeRows {
        running: graph_bake
            .as_ref()
            .filter(|b| b.node == key)
            .map(|b| (b.next, b.len)),
        playing: loaded.get(&key).map(|l| {
            (l.seq.proxy, l.seq.frames.len(), l.seq.info.width, l.seq.info.height)
        }),
        current: gb_current,
        on_disk: gb_target.is_some(),
        half_waiting: proxy >= 1.0
            && gb_proxy_dir.as_ref().is_some_and(|d| engine::bakes::is_baked(d)),
        proxy_on: proxy < 1.0,
        can_bake: gb_key.is_some(),
    });
    *graph_bake_click |= clicks.bake;
    *graph_bake_clear |= clicks.clear;
    *graph_bake_cancel |= clicks.cancel;
    if let Some(e) = load_errors.get(&key) {
        ui.colored_label(theme::error(), e);
    }
    if let Some(d) = &gb_dir {
        ui.add(
            egui::Label::new(
                egui::RichText::new(d.to_string_lossy()).weak().small(),
            )
            .truncate(),
        );
    }
    ui.weak(
        "caches this point of the graph to NVMe (EXR half,                              full alpha); an explicit bake, keyed on the whole                              upstream state",
    );
}
