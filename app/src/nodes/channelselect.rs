//! ChannelSelect: which layer of the input this node reads.
//! Moved out of `properties_ui` (lib.rs) unchanged.

use crate::*;

/// ChannelSelect: which layer of the input this node reads.
pub(crate) fn panel(
    ui: &mut egui::Ui,
    csp: &mut engine::graph::ChannelSelectParams,
    node: u64,
    cs_layers: &Option<Vec<(String, String)>>,
    playback_dirty: &mut bool,
) {
    match &cs_layers {
        Some(layers) if !layers.is_empty() => {
            let shown = if csp.layer.is_empty() {
                "rgba".to_string()
            } else {
                csp.layer.clone()
            };
            let mut pick: Option<String> = None;
            ui.horizontal(|ui| {
                ui.label("channel");
                egui::ComboBox::from_id_salt(("cs-layer", node))
                    .selected_text(shown)
                    .width(180.0)
                    .show_ui(ui, |ui| {
                        for (name, label) in layers {
                            let sel = *name == csp.layer;
                            if ui
                                .selectable_label(sel, label.as_str())
                                .clicked()
                                && !sel
                            {
                                pick = Some(name.clone());
                            }
                        }
                    });
            });
            if let Some(newl) = pick {
                csp.layer = newl;
                *playback_dirty = true;
            }
            ui.weak(
                "one layer off the cable, as a hot wire; only \
                 selected layers decode and cache",
            );
        }
        Some(_) => {
            ui.weak("upstream file has no extra layers");
        }
        None => {
            ui.colored_label(
                theme::error(),
                "not wired to a Load's channel output",
            );
            ui.weak(
                "wire a Read (set to all-channels) into this \
                 node's input",
            );
        }
    }
}
