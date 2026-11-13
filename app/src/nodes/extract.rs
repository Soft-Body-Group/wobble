//! Extract: pull one layer, or all of them into a Layer stack.
//! Moved out of `properties_ui` (lib.rs) unchanged.

/// Extract: pull one layer, or all of them into a Layer stack.
pub(crate) fn panel(
    ui: &mut egui::Ui,
    x: &mut engine::graph::ExtractParams,
    node: u64,
    extract_layers_req: &mut Option<u64>,
) {
    ui.horizontal(|ui| {
        ui.label(format!("{} outputs", x.pairs()));
        if ui.small_button("−").on_hover_text("drop the last pair").clicked() {
            x.n = (x.pairs() as u32).saturating_sub(1).max(1);
        }
        if ui.small_button("+").on_hover_text("one more matte + output").clicked() {
            x.n = (x.pairs() as u32 + 1).min(engine::graph::EXTRACT_MAX);
        }
    });
    if ui
        .button("layers")
        .on_hover_text(
            "a Layer stack below with a Transform on every output, \
             each conformed to its matte's soft crop",
        )
        .clicked()
    {
        *extract_layers_req = Some(node);
    }
    ui.weak("image on the left, a matte on each top input;");
    ui.weak("each output is the image isolated by the matte above it");
}
