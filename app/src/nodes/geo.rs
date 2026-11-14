//! 2dini: the container's page (stats, the web export).
//! Moved out of `properties_ui` (lib.rs) unchanged.

use crate::*;

/// 2dini: the container's page (stats, the web export).
#[allow(clippy::too_many_arguments)]
pub(crate) fn panel(
    ui: &mut egui::Ui,
    rcx: &mut widgets::RowCtx,
    g: &mut engine::graph::GeoParams,
    node: u64,
    wired: [bool; 16],
    geo_stats: Option<(engine::geo::GeoStats, f32, u64)>,
    geo_web_msg: &Option<(u64, String)>,
    geo_web_live: &mut bool,
    geo_web_req: &mut Option<u64>,
) -> Option<widgets::HudSlider> {
    let mut hud_drag_out: Option<widgets::HudSlider> = None;
    widgets::gen_format_row(ui, ("geo-format", node), &mut g.size, &mut g.inherit);
    if let Some(hd) = widgets::slider(
        ui,
        rcx,
        "seed",
        &mut g.seed,
        0.0,
        256.0,
        &[anim::Channel::GeoSeed],
    ) {
        hud_drag_out = Some(hd);
    }
    // The field port: what every Field In inside reads.
    let field_wired = wired.first().copied().unwrap_or(false);
    ui.weak(match (field_wired, g.graph.has_field_in()) {
        (true, true) => "field port: a Noise drives the Field In nodes",
        (true, false) => "field port: wired, but nothing inside reads it",
        (false, true) => "field port: wire a Noise — Field In reads 0.5 until then",
        (false, false) => "field port: wire a Noise to drive a Field In node",
    });
    ui.horizontal(|ui| {
        if ui
            .button("export web page…")
            .on_hover_text("every frame of the play range as one static HTML page: the shapes as data, drawn with Canvas2D, looping")
            .clicked()
        {
            *geo_web_req = Some(node);
        }
        ui.checkbox(geo_web_live, "play forever")
            .on_hover_text("the page evaluates the drawing itself, every frame, so moving noise never repeats (a baked simulation plays once and holds). Off: loop the play range's frames");
    });
    if let Some((n, msg)) = &geo_web_msg {
        if *n == node {
            ui.weak(msg);
        }
    }
    // The sub-graph's Sliders are THIS node's widgets and
    // its Inputs are jacks, exactly as a Noise node's are.
    let exposed = g.graph.exposed();
    if !exposed.is_empty() {
        ui.separator();
    }
    for (name, is_input) in &exposed {
        if *is_input {
            continue;
        }
        let ch = anim::Channel::Custom(anim::intern(name));
        let mut val = g.graph.exposed_value(name).unwrap_or(0.0);
        let before = val;
        if let Some(h) =
            widgets::slider(ui, rcx, name, &mut val, -4.0, 4.0, &[ch])
        {
            hud_drag_out = Some(h);
        }
        if val != before {
            g.graph.set_exposed(name, val);
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
    ui.separator();
    match geo_stats {
        Some((st, ms, cost)) => {
            ui.weak(format!(
                "{} prims / {} verts / {} points — evaluated in {ms:.1} ms{}",
                st.prims,
                st.verts,
                st.points,
                if st.truncated { " (CAPPED)" } else { "" }
            ));
            // The re-record cost, so a drawing that is
            // expensive to redraw is never expensive
            // silently (manifesto rule 8): pixels times
            // segments, which is what the raster does.
            // Static, it is paid once; animated, every
            // frame.
            let m = cost as f32 / 1e6;
            if cost > 200_000_000 {
                ui.label(
                    egui::RichText::new(format!(
                        "raster ≈ {m:.0} M edge tests — HEAVY when animated"
                    ))
                    .small()
                    .color(theme::cost_note()),
                );
            } else {
                ui.weak(format!(
                    "raster ≈ {m:.1} M edge tests; re-recorded only when the geometry changes"
                ));
            }
        }
        None => {
            ui.weak("not rendered yet");
        }
    }
    ui.weak(if g.graph.nodes.is_empty() {
        "empty — open it (↩) to draw"
    } else {
        "open the node (↩) to edit its geometry"
    });
    hud_drag_out
}
