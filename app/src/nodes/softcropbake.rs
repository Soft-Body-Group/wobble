//! SoftCrop's bake page: the same rows a Bake draws.
//! Moved out of `properties_ui` (lib.rs) unchanged.

use crate::*;

/// Everything the softcropbake panel borrows from the properties pass.
pub(crate) struct Cx<'a> {
    pub p: &'a mut engine::graph::SoftCropParams,
    pub key: u64,
    pub gb_current: bool,
    pub gb_key: Option<u64>,
    pub gb_proxy_dir: &'a Option<std::path::PathBuf>,
    pub gb_target: &'a Option<std::path::PathBuf>,
    pub playback_dirty: &'a mut bool,
    pub auto_bake: &'a mut Option<(u64, u8)>,
    pub next_read_slot: &'a mut usize,
    pub proxy: f32,
    pub loaded: &'a HashMap<u64, LoadedSeq>,
    pub load_errors: &'a HashMap<u64, String>,
    pub graph_bake: &'a Option<SubtreeBake>,
    pub graph_bake_cancel: &'a mut bool,
    pub graph_bake_clear: &'a mut bool,
    pub graph_bake_click: &'a mut bool,
}

/// SoftCrop's bake page: the same rows a Bake draws.
pub(crate) fn panel(
    ui: &mut egui::Ui,
    cx: Cx,
) {
    let Cx {
        p,
        key,
        gb_current,
        gb_key,
        gb_proxy_dir,
        gb_target,
        playback_dirty,
        auto_bake,
        next_read_slot,
        proxy,
        loaded,
        load_errors,
        graph_bake,
        graph_bake_cancel,
        graph_bake_clear,
        graph_bake_click,
    } = cx;
    // The Crop's own bake: the Bake node's machinery on
    // the Crop itself (`editor::lower_crop_bakes`).
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
    if clicks.bake {
        if p.bake.is_none() {
            // First bake: the Crop takes a reader slot,
            // and the bake starts once the rig has
            // rebuilt around it (the Paint precedent).
            p.bake = Some(*next_read_slot);
            *next_read_slot += 1;
            *playback_dirty = true;
            *auto_bake = Some((key, 6));
        } else {
            *graph_bake_click = true;
        }
    }
    if clicks.clear {
        // Back to the plain Crop it was: no slot, nothing lowered.
        p.bake = None;
        *graph_bake_clear = true;
    }
    *graph_bake_cancel |= clicks.cancel;
    if let Some(e) = load_errors.get(&key) {
        ui.colored_label(theme::error(), e);
    }
    ui.weak(match p.mode() {
        engine::graph::CropMode::Crop => {
            "stores only the box, played back where it sits"
        }
        engine::graph::CropMode::Reformat => "stores the box's format",
        engine::graph::CropMode::BoundingBox => "stores the whole frame",
    });
}
