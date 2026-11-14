//! Layer: hands the stack's mini timeline what it needs.
//! Moved out of `properties_ui` (lib.rs) unchanged.

use crate::*;

/// Everything the layer panel borrows from the properties pass.
pub(crate) struct Cx<'a> {
    pub lp: &'a mut engine::graph::LayerParams,
    pub cur_t: u64,
    pub node: u64,
    pub blend_res: &'a mut Option<(u32, u32)>,
    pub global_fmt: Option<(u32, u32)>,
    pub layer_meta: &'a Option<((u64, u64), Vec<String>, Vec<Option<layerui::Src>>)>,
    pub layer_sel_now: usize,
    pub script: (i64, i64),
    pub play_range: &'a mut Option<(u64, u64)>,
    pub compositor: &'a Compositor,
    pub layer_act: &'a mut Option<layerui::TimelineAction>,
    pub layer_add: &'a mut bool,
    pub layer_bars: &'a mut Option<Vec<egui::Rect>>,
    pub layer_changed: &'a mut bool,
    pub layer_del: &'a mut Option<usize>,
    pub layer_popout: &'a mut bool,
}

/// Layer: hands the stack's mini timeline what it needs.
pub(crate) fn panel(
    ui: &mut egui::Ui,
    rcx: &mut widgets::RowCtx,
    cx: Cx,
) -> Option<widgets::HudSlider> {
    let mut hud_drag_out: Option<widgets::HudSlider> = None;
    let Cx {
        lp,
        cur_t,
        node,
        blend_res,
        global_fmt,
        layer_meta,
        layer_sel_now,
        script,
        play_range,
        compositor,
        layer_act,
        layer_add,
        layer_bars,
        layer_changed,
        layer_del,
        layer_popout,
    } = cx;
    // The whole panel lives in layerui, because the
    // pop-out window draws exactly the same thing.
    if let Some((comp, names, lens)) = &layer_meta {
        let o = layerui::panel(
            ui,
            lp,
            rcx,
            node,
            layer_sel_now,
            names,
            lens,
            *comp,
            cur_t,
            minibar::Ranges {
                comp: script,
                node: (comp.0 as i64, comp.1 as i64),
                inout: play_range.map(|(a, b)| (a as i64, b as i64)),
            },
            blend_res,
            compositor.output_size().unwrap_or((1920, 1080)),
            global_fmt,
            true,
        );
        *layer_act = o.action;
        *layer_changed |= o.changed;
        *layer_bars = Some(o.bars);
        *layer_add = o.add;
        *layer_del = o.del;
        *layer_popout = o.popout;
        if o.hud.is_some() {
            hud_drag_out = o.hud;
        }
    }
    hud_drag_out
}
