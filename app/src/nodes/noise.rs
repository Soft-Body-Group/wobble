//! Noise: the fBm controls and its presets.
//! Moved out of `properties_ui` (lib.rs) unchanged.

use crate::*;

/// Noise: the fBm controls and its presets.
pub(crate) fn panel(
    ui: &mut egui::Ui,
    rcx: &mut widgets::RowCtx,
    n: &mut engine::graph::NoiseParams,
    cur_t: u64,
    node: u64,
) -> Option<widgets::HudSlider> {
    let mut hud_drag_out: Option<widgets::HudSlider> = None;
    widgets::gen_format_row(ui, ("noise-format", node), &mut n.size, &mut n.inherit);
    noise_preset_row(ui, node, &mut n.graph);
    if let Some(hd) = widgets::slider(
        ui,
        rcx,
        "scale",
        &mut n.scale,
        0.05,
        8.0,
        &[anim::Channel::NoiseScale],
    ) {
        hud_drag_out = Some(hd);
    }
    if let Some(hd) = widgets::slider(
        ui,
        rcx,
        "seed",
        &mut n.seed,
        0.0,
        256.0,
        &[anim::Channel::NoiseSeed],
    ) {
        hud_drag_out = Some(hd);
    }
    // Two dimensions: green is the same graph at a
    // moved seed — x and y displaced alike, not in
    // lockstep.
    ui.checkbox(&mut n.two_d, "2d").on_hover_text(
        "red = the field, green = the same field at seed + offset: \
         one character, two values (evaluates the graph twice)",
    );
    if n.two_d {
        if let Some(hd) = widgets::slider(
            ui,
            rcx,
            "seed offset",
            &mut n.seed_offset,
            0.0,
            256.0,
            &[anim::Channel::NoiseSeedOffset],
        ) {
            hud_drag_out = Some(hd);
        }
    }
    if let Some(hd) = widgets::slider(
        ui,
        rcx,
        "speed",
        &mut n.speed,
        0.0,
        4.0,
        &[anim::Channel::NoiseSpeed],
    ) {
        hud_drag_out = Some(hd);
    }
    ui.weak(if n.speed_unit == 1.0 {
        "cycles per frame (a setup from before the 0.1x rate) — every generator scales this by its own"
    } else {
        "tenths of a cycle per frame — every generator scales this by its own"
    });
    ui.horizontal(|ui| {
        ui.checkbox(&mut n.single, "single channel").on_hover_text(
            "the field in all four channels, like a Load's single channel \
             (the 2d second field is not evaluated); a Bake of it stores one \
             channel: a quarter of the memory",
        );
        ui.checkbox(&mut n.eight_bit, "8-bit").on_hover_text(
            "clamp to 0..1 and round to 256 steps; a Bake of it stores 8 bits",
        );
    });
    ui.separator();
    // The field AS DATA. The probe is a point in the
    // field; `noise value` is what the graph says there,
    // and a patch cable out of that channel drives any
    // parameter in the comp.
    ui.horizontal(|ui| {
        ui.add_sized(
            [52.0, 16.0],
            egui::Label::new(egui::RichText::new("probe").small()),
        );
        ui.add(
            egui::DragValue::new(&mut n.probe[0]).update_while_editing(false).speed(widgets::rated(ui, 0.002)).range(-64.0..=64.0),
        );
        ui.add(
            egui::DragValue::new(&mut n.probe[1]).update_while_editing(false).speed(widgets::rated(ui, 0.002)).range(-64.0..=64.0),
        );
        widgets::key_dot(
            ui,
            rcx,
            &[anim::Channel::NoiseProbeX, anim::Channel::NoiseProbeY],
        );
    });
    let at = |seed: f32| {
        engine::field::eval_cpu(&n.graph, n.probe, seed, cur_t as f32 * n.speed * n.speed_unit, 1.0)
    };
    let shown = if n.two_d {
        format!("{:.4}  {:.4}", at(n.seed), at(n.seed + n.seed_offset))
    } else {
        format!("{:.4}", at(n.seed))
    };
    ui.horizontal(|ui| {
        ui.add_sized(
            [52.0, 16.0],
            egui::Label::new(egui::RichText::new("value").small()),
        );
        ui.label(egui::RichText::new(shown).monospace().color(theme::cable()));
        ui.weak("⬅ patch this to drive anything");
    });
    // The sub-graph's Sliders are THIS node's widgets, and
    // its Inputs are jacks with no widget: both are
    // `Custom` channels named after the node, so the key
    // dot, HUD drag-out and link-mode jack are the ones
    // every other row has.
    let exposed = n.graph.exposed();
    if !exposed.is_empty() {
        ui.separator();
    }
    for (name, is_input) in &exposed {
        if *is_input {
            continue;
        }
        let ch = anim::Channel::Custom(anim::intern(name));
        let mut val = n.graph.exposed_value(name).unwrap_or(0.0);
        let before = val;
        if let Some(h) =
            widgets::slider(ui, rcx, name, &mut val, -4.0, 4.0, &[ch])
        {
            hud_drag_out = Some(h);
        }
        if val != before {
            n.graph.set_exposed(name, val);
        }
    }
    let inputs: Vec<&str> = exposed
        .iter()
        .filter(|(_, i)| *i)
        .map(|(nm, _)| nm.as_str())
        .collect();
    if !inputs.is_empty() {
        ui.weak(format!(
            "data inputs: {} — cable them in link mode (E)",
            inputs.join(", ")
        ));
    }
    ui.weak(if n.graph.nodes.is_empty() {
        "empty — open it to build the field"
    } else {
        "open the node to edit its field graph"
    });
    hud_drag_out
}
