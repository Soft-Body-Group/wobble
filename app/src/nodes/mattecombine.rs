//! MatteCombine: how the mattes combine.
//! Moved out of `properties_ui` (lib.rs) unchanged.

use crate::*;

/// MatteCombine: how the mattes combine.
pub(crate) fn panel(
    ui: &mut egui::Ui,
    rcx: &mut widgets::RowCtx,
    mc: &mut engine::graph::MatteCombineParams,
    node: u64,
) -> Option<widgets::HudSlider> {
    let mut hud_drag_out: Option<widgets::HudSlider> = None;
    ui.horizontal(|ui| {
        ui.label("operation");
        egui::ComboBox::from_id_salt(("matte-op", node))
            .selected_text(mc.op.name())
            .show_ui(ui, |ui| {
                for o in engine::graph::MatteOp::ALL {
                    ui.selectable_value(&mut mc.op, o, o.name());
                }
            });
    });
    if let Some(h) = widgets::slider(
        ui,
        rcx,
        "mix",
        &mut mc.mix,
        0.0,
        1.0,
        &[anim::Channel::MatteMix],
    ) {
        hud_drag_out = Some(h);
    }
    ui.weak(match mc.op {
        engine::graph::MatteOp::Union => "max(A, B) — either, no accumulation",
        engine::graph::MatteOp::Add => "A + B clamped — soft edges ADD up",
        engine::graph::MatteOp::Replace => "B alone; A is dropped",
        engine::graph::MatteOp::Subtract => "A punched out by B",
        engine::graph::MatteOp::Intersect => "A × B — both, soft",
        engine::graph::MatteOp::Min => "min(A, B) — both, keeps soft edges",
        engine::graph::MatteOp::Difference => "|A − B| — where they disagree",
    });
    ui.weak("an unwired input is EMPTY coverage, so one input alone");
    ui.weak("passes through. mix 0 bypasses to A.");
    hud_drag_out
}
