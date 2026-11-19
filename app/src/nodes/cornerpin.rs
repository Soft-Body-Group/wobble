//! CornerPin: the four corners, to and from.
//! Moved out of `properties_ui` (lib.rs) unchanged.

use crate::*;

/// CornerPin: the four corners, to and from.
pub(crate) fn panel(
    ui: &mut egui::Ui,
    rcx: &mut widgets::RowCtx,
    cp: &mut engine::graph::CornerPinParams,
    node: u64,
    wired: [bool; 16],
    pin_from: &mut std::collections::HashSet<u64>,
    track_import_req: &mut Option<bool>,
) -> Option<widgets::HudSlider> {
    let mut hud_drag_out: Option<widgets::HudSlider> = None;
    use anim::Channel::*;
    const PIN_CH: [[anim::Channel; 2]; 4] = [
        [Pin0X, Pin0Y],
        [Pin1X, Pin1Y],
        [Pin2X, Pin2Y],
        [Pin3X, Pin3Y],
    ];
    const FROM_CH: [[anim::Channel; 2]; 4] = [
        [From0X, From0Y],
        [From1X, From1Y],
        [From2X, From2Y],
        [From3X, From3Y],
    ];
    // Which quad the viewer's handles edit.
    let from_on = pin_from.contains(&node);
    ui.horizontal(|ui| {
        ui.label("viewer edits");
        if ui.add(egui::Button::selectable(!from_on, "pin")).clicked() {
            pin_from.remove(&node);
        }
        if ui.add(egui::Button::selectable(from_on, "from")).clicked() {
            pin_from.insert(node);
        }
    });
    // Pin and from side by side: eight rows stacked
    // are taller than the tile.
    ui.columns(2, |cols| {
        for (col, (head, quad, chans)) in cols.iter_mut().zip([
            ("pin", &mut cp.to, &PIN_CH),
            ("from", &mut cp.from, &FROM_CH),
        ]) {
            col.weak(head);
            for (i, name) in ["TL", "TR", "BR", "BL"].iter().enumerate() {
                let [mut x, mut y] = quad[i];
                let h = widgets::drags(
                    col,
                    rcx,
                    name,
                    &mut [&mut x, &mut y],
                    0.002,
                    (0.0, 1.0),
                    &chans[i],
                );
                quad[i] = [x, y];
                if h.is_some() {
                    hud_drag_out = h;
                }
            }
        }
    });
    ui.horizontal(|ui| {
        if ui.button("reset").clicked() {
            *cp = engine::graph::CornerPinParams {
                mod_blend: cp.mod_blend,
                ..Default::default()
            };
        }
        // `from` = the pin: the warp is nothing until
        // the pin moves off it again.
        if ui.button("from = pin").on_hover_text("rest the pin where it is: no warp until it moves").clicked() {
            cp.from = cp.to;
        }
        if ui.button("from = frame").clicked() {
            cp.from = engine::graph::UNIT_QUAD;
        }
        if ui.button("import track…").on_hover_text("key the pin from exported corner pin or transform text").clicked() {
            *track_import_req = Some(true);
        }
    });
    ui.checkbox(&mut cp.invert, "invert")
        .on_hover_text("apply the inverse warp, pin back onto from: undo a matchmove, then redo it with a second pin");
    ui.weak("the warp carries from onto pin; S / marquee pick pins, E R T scale rotate move");
    if wired[1] {
        modulate_blend_row(ui, &mut cp.mod_blend);
    }
    hud_drag_out
}
