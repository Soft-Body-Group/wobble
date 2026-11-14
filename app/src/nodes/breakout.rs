//! Breakout: which layers of the input fan out, one output each.
//! Moved out of `properties_ui` (lib.rs) unchanged.

use crate::*;

/// Breakout: which layers of the input fan out, one output each.
pub(crate) fn panel(
    ui: &mut egui::Ui,
    bp: &mut engine::graph::BreakoutParams,
    node: u64,
    cs_layers: &Option<Vec<(String, String)>>,
    playback_dirty: &mut bool,
) {
    // The horizontal scale — the string pitch both bars
    // share (sync mirrors it onto the receiver, so it is
    // LOCKED between them). Always visible: it is the
    // node's one layout knob.
    widgets::plain_slider(ui, "h spacing", &mut bp.spacing, 40.0, 260.0, false);
    if ui
        .checkbox(&mut bp.remainder, "remainder")
        .on_hover_text(
            "one more string: the base rgba minus every \
             chosen layer — whatever the renderer failed \
             to distribute into the AOVs. Signed, so the \
             receiver's sum takes it back out exactly.",
        )
        .changed()
    {
        *playback_dirty = true;
    }
    match &cs_layers {
        Some(layers) if !layers.is_empty() => {
            // The strings, in nodule order — each one a
            // SHUFFLE you can re-point after the fact:
            // swapping a string's layer keeps its kid,
            // its slot and everything spliced into it,
            // so a graded string re-aims without
            // rebuilding the chain.
            ui.weak("strings — re-point or drop in place");
            let mut changed = false;
            let mut repoint: Option<(usize, String)> = None;
            let mut drop_at: Option<usize> = None;
            for (i, cur) in bp.layers.iter().enumerate() {
                ui.horizontal(|ui| {
                    let shown = layers
                        .iter()
                        .find(|(n, _)| n == cur)
                        .map(|(_, l)| l.clone())
                        .unwrap_or_else(|| cur.clone());
                    egui::ComboBox::from_id_salt((
                        "bo-string",
                        node,
                        i,
                    ))
                    .selected_text(shown)
                    .width(150.0)
                    .show_ui(ui, |ui| {
                        for (name, label) in layers {
                            // A layer already on another
                            // string stays pickable —
                            // two strings of one pass is
                            // odd but not wrong.
                            let sel = name == cur;
                            if ui
                                .selectable_label(
                                    sel,
                                    label.as_str(),
                                )
                                .clicked()
                                && !sel
                            {
                                repoint =
                                    Some((i, name.clone()));
                            }
                        }
                    });
                    if ui
                        .small_button("×")
                        .on_hover_text("drop this string")
                        .clicked()
                    {
                        drop_at = Some(i);
                    }
                });
            }
            if let Some((i, name)) = repoint {
                bp.layers[i] = name;
                changed = true;
            }
            if let Some(i) = drop_at {
                bp.layers.remove(i);
                changed = true;
            }
            let unused: Vec<(String, String)> = layers
                .iter()
                .filter(|(n, _)| !bp.layers.contains(n))
                .cloned()
                .collect();
            if !unused.is_empty() {
                ui.weak("tick to add");
                for (name, label) in unused {
                    let mut on = false;
                    if ui
                        .checkbox(&mut on, label.as_str())
                        .changed()
                        && on
                    {
                        bp.layers.push(name);
                        changed = true;
                    }
                }
            }
            if changed {
                *playback_dirty = true;
            }
            ui.weak(
                "the paired Recombine sums the strings back; \
                 splice grades into them like a harp",
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
