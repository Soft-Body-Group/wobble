//! A plugin node or a subwob tool: its declared params, or its laid-out panel.
//! Moved out of `properties_ui` (lib.rs) unchanged.

use crate::*;

/// A plugin node or a subwob tool: its declared params, or its laid-out panel.
#[allow(clippy::too_many_arguments)]
pub(crate) fn panel(
    ui: &mut egui::Ui,
    rcx: &mut widgets::RowCtx,
    type_id: &mut String,
    params: &mut std::collections::BTreeMap<String, f32>,
    node: u64,
    prop_zoom: f32,
    ui_label: &mut String,
    ungang: &mut std::collections::BTreeSet<String>,
    working: engine::color::Colorspace,
) -> (Option<widgets::HudSlider>, bool) {
    let mut hud_drag_out: Option<widgets::HudSlider> = None;
    // A CONTROL is captioned here; that caption is what
    // its knob is called on every instance's panel.
    if tools::ControlKind::from_type(type_id).is_some() {
        ui.horizontal(|ui| {
            ui.label("caption");
            ui.text_edit_singleline(ui_label);
        });
    }
    match engine::plugins::get(type_id) {
        None => {
            ui.colored_label(
                theme::warn(),
                format!("plugin '{type_id}' not installed"),
            );
            ui.weak(
                "values are preserved; the node passes through",
            );
        }
        Some(def) => {
            // Never silently wrong: a body that reads
            // footage does not unfold yet, so say so.
            if let Some(why) =
                tools::get(type_id).and_then(|t| tools::body_reads_footage(&t.body))
            {
                ui.colored_label(
                    theme::warn(),
                    format!(
                        "{why} reads footage — not supported inside a subwob yet; \
                         this node passes its input through"
                    ),
                );
            }
            // A TOOL with an authored panel draws that
            // instead of a row per knob. An empty layout
            // — every tool before anyone arranges one —
            // falls through to the standard rows.
            // A TOOL draws its control nodes; a shader
            // plugin falls through to the rows below.
            let laid_out = tools::get(type_id)
                .map(|t| {
                    egui::ScrollArea::vertical()
                        .id_salt(("tool-panel", node))
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            tool_panel_ui(
                                ui,
                                rcx,
                                &t.body,
                                tools::cols(t.tiles),
                                tools::rows(t.tiles_h),
                                params,
                                ungang,
                                working,
                                prop_zoom,
                                None,
                            )
                        })
                        .inner
                });
            if let Some(h) = laid_out {
                if h.is_some() {
                    hud_drag_out = h;
                }
                ui.weak(format!("subwob · {}", def.type_id));
                return (hud_drag_out, true);
            }
            for spec in &def.params {
                let chan =
                    anim::Channel::Custom(anim::intern(&spec.name));
                let mut v = params
                    .get(&spec.name)
                    .copied()
                    .unwrap_or(spec.default);
                let h = if spec.min < spec.max {
                    widgets::slider(
                        ui,
                        rcx,
                        &spec.label,
                        &mut v,
                        spec.min,
                        spec.max,
                        &[chan],
                    )
                } else {
                    widgets::drags(
                        ui,
                        rcx,
                        &spec.label,
                        &mut [&mut v],
                        0.01,
                        (0.0, 1.0),
                        &[chan],
                    )
                };
                params.insert(spec.name.clone(), v);
                if h.is_some() {
                    hud_drag_out = h;
                }
            }
            ui.weak(format!("plugin · {}", def.type_id));
        }
    }
    (hud_drag_out, false)
}
