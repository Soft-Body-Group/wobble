//! Blend: the operator, the mix and the per-input gains.
//! Moved out of `properties_ui` (lib.rs) unchanged.

use crate::*;

/// Blend: the operator, the mix and the per-input gains.
#[allow(clippy::too_many_arguments)]
pub(crate) fn panel(
    ui: &mut egui::Ui,
    rcx: &mut widgets::RowCtx,
    op: &mut engine::graph::BlendOp,
    mix: &mut f32,
    gain: &mut [f32; 2],
    blend_res: &mut Option<(u32, u32)>,
    node: u64,
    comp: &Compositor,
) -> Option<widgets::HudSlider> {
    let mut hud_drag_out: Option<widgets::HudSlider> = None;
    ui.horizontal(|ui| {
        ui.label("operation");
        egui::ComboBox::from_id_salt(("merge-op", node))
            .selected_text(op.name())
            .show_ui(ui, |ui| {
                for o in engine::graph::BlendOp::ALL {
                    ui.selectable_value(op, o, o.name());
                }
            });
    });
    ui.weak(if *op == engine::graph::BlendOp::Replace {
        "+ (input 1) replaces − (input 2) where the matte covers"
    } else {
        "+ (input 1) blended onto − (input 2) · matte limits it"
    });
    if let Some(h) = widgets::slider(
        ui,
        rcx,
        "mix",
        mix,
        0.0,
        1.0,
        &[anim::Channel::BlendMix],
    ) {
        hud_drag_out = Some(h);
    }
    ui.weak("0 dissolves the whole blend back to − (bypass)");
    // The graph's horseshoe sliders, as ordinary
    // keyframable rows — one truth, two handles.
    let [ga, gb] = gain;
    if let Some(h) = widgets::slider(
        ui,
        rcx,
        "gain +",
        ga,
        0.0,
        2.0,
        &[anim::Channel::BlendGainA],
    ) {
        hud_drag_out = Some(h);
    }
    if let Some(h) = widgets::slider(
        ui,
        rcx,
        "gain −",
        gb,
        0.0,
        2.0,
        &[anim::Channel::BlendGainB],
    ) {
        hud_drag_out = Some(h);
    }
    ui.weak("per-input multipliers — draggable on the arcs in the graph");
    ui.horizontal(|ui| {
        let mut on = blend_res.is_some();
        if ui
            .checkbox(&mut on, "resolution override")
            .on_hover_text(
                "fix this merge's output size; else it inherits \
                 the background (−) input's resolution",
            )
            .changed()
        {
            *blend_res = on
                .then(|| comp.output_size().unwrap_or((1920, 1080)));
        }
        if let Some((w, h)) = blend_res {
            crate::widgets::num_field(ui, egui::DragValue::new(w).update_while_editing(false).speed(widgets::rated(ui, 4)).range(16..=16384),
            );
            crate::widgets::num_field(ui, egui::DragValue::new(h).update_while_editing(false).speed(widgets::rated(ui, 4)).range(16..=16384),
            );
        }
    });
    hud_drag_out
}
