//! Recombine: the harp's spacing and the layer each input lands in.
//! Moved out of `properties_ui` (lib.rs) unchanged.

use crate::*;

/// Recombine: the harp's spacing and the layer each input lands in.
pub(crate) fn panel(
    ui: &mut egui::Ui,
    rcx: &mut widgets::RowCtx,
    rp: &mut engine::graph::RecombineParams,
    node: u64,
    harp_spacing: &mut Option<(u64, f32)>,
) -> Option<widgets::HudSlider> {
    let mut hud_drag_out: Option<widgets::HudSlider> = None;
    ui.weak(format!(
        "sums its {} strings — the harp's receiver",
        rp.strings()
    ));
    // Summing AOVs rebuilds COLOUR; their alphas are
    // per-pass coverage and adding those means nothing.
    // So the rebuild wears the ORIGINAL rgba's alpha by
    // default, off the porch input the harp wires for
    // it (hand-wire that port to override the source).
    ui.horizontal(|ui| {
        ui.label("alpha");
        egui::ComboBox::from_id_salt(("rc-alpha", node))
            .selected_text(rp.alpha.label())
            .width(120.0)
            .show_ui(ui, |ui| {
                for a in engine::graph::AlphaFrom::ALL {
                    if ui
                        .selectable_label(rp.alpha == a, a.label())
                        .clicked()
                    {
                        rp.alpha = a;
                    }
                }
            });
    });
    if rp.alpha == engine::graph::AlphaFrom::Base {
        ui.weak(
            "rgb from the sum, alpha off the rgba on the \
             left bookend",
        );
    }
    // One keyframable row per string, mirroring the
    // graph's horseshoe arcs — same channels, one truth.
    let n = rp.n.clamp(1, 16) as usize;
    if rp.gain.len() < n {
        rp.gain.resize(n, 1.0);
    }
    for i in 0..n {
        let (head, tail) = rp.gain.split_at_mut(i);
        let _ = head;
        if let Some(h) = widgets::slider(
            ui,
            rcx,
            &format!("s{} gain", i + 1),
            &mut tail[0],
            0.0,
            2.0,
            &[anim::Channel::RecombineGain(i as u16)],
        ) {
            hud_drag_out = Some(h);
        }
    }
    if rp.pair != 0 {
        // The same locked knob as on the Breakout: an
        // edit here writes through to the pair, and
        // sync mirrors it straight back onto this bar.
        let mut sp = rp.spacing;
        if widgets::plain_slider(ui, "h spacing", &mut sp, 40.0, 260.0, false)
            .changed()
        {
            rp.spacing = sp;
            *harp_spacing = Some((rp.pair, sp));
        }
        ui.weak(
            "x-locked under its Breakout: drag it vertically, \
             the column holds",
        );
    } else {
        let mut n = rp.n.clamp(1, 16) as f32;
        if widgets::plain_slider(ui, "inputs", &mut n, 1.0, 16.0, true).changed() {
            rp.n = n.round() as _;
        }
    }
    hud_drag_out
}
