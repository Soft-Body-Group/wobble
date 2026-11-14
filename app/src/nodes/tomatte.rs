//! ToMatte: which channel becomes the matte.
//! Moved out of `properties_ui` (lib.rs) unchanged.

use crate::*;

/// ToMatte: which channel becomes the matte.
pub(crate) fn panel(
    ui: &mut egui::Ui,
    t: &mut engine::graph::ToMatteParams,
) {
    ui.horizontal(|ui| {
        ui.add_sized(
            [50.0, 16.0],
            egui::Label::new(egui::RichText::new("channel").small()),
        );
        for c in engine::graph::MatteChannel::ALL {
            ui.selectable_value(&mut t.channel, c, c.name());
        }
    });
    ui.checkbox(&mut t.invert, "invert");
    ui.horizontal(|ui| {
        ui.add_sized(
            [50.0, 16.0],
            egui::Label::new(egui::RichText::new("range").small()),
        );
        crate::widgets::num_field(ui, egui::DragValue::new(&mut t.lo).update_while_editing(false).speed(widgets::rated(ui, 0.01)).prefix("lo "),
        );
        crate::widgets::num_field(ui, egui::DragValue::new(&mut t.hi).update_while_editing(false).speed(widgets::rated(ui, 0.01)).prefix("hi "),
        );
    })
    .response
    .on_hover_text(
        "what counts as 0 and 1 — a scene-linear pass that runs \
         past 1 needs telling",
    );
    ui.weak(
        "image in, MATTE out: the bridge every mask input needs",
    );
}
