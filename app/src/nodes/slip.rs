//! Slip: a constant offset in time.
//! Moved out of `properties_ui` (lib.rs) unchanged.

use crate::*;

/// Slip: a constant offset in time.
pub(crate) fn panel(
    ui: &mut egui::Ui,
    to: &mut engine::graph::SlipParams,
    extra_readers: Option<usize>,
) {
    ui.horizontal(|ui| {
        ui.label("offset");
        crate::widgets::num_field_w(ui, widgets::NUM_W + 14.0, egui::DragValue::new(&mut to.offset).update_while_editing(false)
                .speed(widgets::rated(ui, 0.25))
                .suffix(" f"),
        )
        .on_hover_text("positive = later footage at the same tick");
    });
    if let Some(n) = extra_readers {
        ui.weak(format!(
            "{n} extra reader{} — the sources above this feed \
             other branches at other frames too",
            if n == 1 { "" } else { "s" }
        ));
    }
    ui.weak(
        "shifts which frame the sources above it read; no pixels \
         move, so it costs nothing to render",
    );
}
