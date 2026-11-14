//! Defocus: the lens, the depth modes and the focus pick.
//! Moved out of `properties_ui` (lib.rs) unchanged.

use crate::*;

/// Defocus: the lens, the depth modes and the focus pick.
#[allow(clippy::too_many_arguments)]
pub(crate) fn panel(
    ui: &mut egui::Ui,
    rcx: &mut widgets::RowCtx,
    d: &mut engine::graph::DefocusParams,
    node: u64,
    wired: [bool; 16],
    focus_armed: Option<u64>,
    focus_pick_req: &mut Option<u64>,
    view_band_req: &mut Option<(u64, u32)>,
) -> Option<widgets::HudSlider> {
    let mut hud_drag_out: Option<widgets::HudSlider> = None;
    use engine::graph::DefocusMode;
    let sl = |ui: &mut egui::Ui,
                  rcx: &mut _,
                  label,
                  v: &mut f32,
                  lo,
                  hi,
                  ch| {
        widgets::slider(ui, rcx, label, v, lo, hi, &[ch])
    };
    let has_depth = wired[2];
    // Four ports and a hexagon that says nothing about
    // which is which: spell them out, and show what is
    // actually wired so a missed connection is visible
    // here rather than only as a picture that looks odd.
    for (i, tag) in [(0usize, "·"), (1, "·"), (2, "Z"), (3, "B")] {
        let name = NodeKind::Defocus(Default::default()).input_name(i);
        let (mark, col) = if wired[i] {
            ("⏺", ui.visuals().strong_text_color())
        } else {
            ("○", ui.visuals().weak_text_color())
        };
        ui.horizontal(|ui| {
            ui.colored_label(col, mark);
            ui.colored_label(col, format!("{}  {name}", tag));
        });
    }
    ui.separator();
    ui.horizontal(|ui| {
        ui.label("mode");
        for m in [DefocusMode::Fast, DefocusMode::Accurate] {
            let name = match m {
                DefocusMode::Fast => "fast",
                DefocusMode::Accurate => "accurate (slow)",
            };
            if ui
                .add(egui::Button::selectable(d.mode == m, name))
                .clicked()
            {
                d.mode = m;
            }
        }
    });
    for h in [
        sl(ui, rcx, "max coc", &mut d.radius, 0.0, 256.0,
            anim::Channel::DefocusRadius),
    ]
    .into_iter()
    .flatten()
    {
        hud_drag_out = Some(h);
    }
    if has_depth {
        // Focus is a raw DEPTH-PASS value: unbounded
        // (world-space passes run to thousands) and
        // dragged at a rate proportional to itself, so
        // 3.2 and 3200 both pull by feel.
        let speed = (d.focus.abs() * 0.004).max(1e-3);
        let hud = (0.0, (d.focus.abs() * 2.0).max(1.0));
        if let Some(h) = widgets::drags(
            ui,
            rcx,
            "focus",
            &mut [&mut d.focus],
            speed,
            hud,
            &[anim::Channel::DefocusFocus],
        ) {
            hud_drag_out = Some(h);
        }
        ui.horizontal(|ui| {
            let armed = focus_armed == Some(node);
            if ui
                .add(egui::Button::selectable(armed, "pick depth"))
                .on_hover_text(
                    "then click or drag on the viewer: the depth pass \
                     under the pointer becomes the focus",
                )
                .clicked()
            {
                *focus_pick_req = Some(node);
            }
            let was = d.focal_view;
            ui.toggle_value(&mut d.focal_view, "focal plane")
                .on_hover_text(
                    "view the node as green in focus, red in front, \
                     blue behind",
                );
            if d.focal_view && !was {
                // The view only shows on the view
                // target: asking for it means looking.
                *view_band_req = Some((node, 0));
            }
        });
        if focus_armed == Some(node) {
            ui.weak("click / drag the viewer to set focus");
        }
        for h in [
            sl(ui, rcx, "f-stop", &mut d.f_stop, 0.7, 32.0,
                anim::Channel::DefocusFStop),
        ]
        .into_iter()
        .flatten()
        {
            hud_drag_out = Some(h);
        }
        ui.horizontal(|ui| {
            ui.label("lens");
            crate::widgets::num_field(ui, egui::DragValue::new(&mut d.focal_len).update_while_editing(false)
                    .speed(widgets::rated(ui, 0.5))
                    .range(1.0..=1000.0)
                    .suffix("mm"),
            )
            .on_hover_text("focal length");
            crate::widgets::num_field(ui, egui::DragValue::new(&mut d.filmback).update_while_editing(false)
                    .speed(widgets::rated(ui, 0.1))
                    .range(1.0..=200.0)
                    .suffix("mm"),
            )
            .on_hover_text("filmback / sensor width");
        });
        ui.horizontal(|ui| {
            ui.label("1 unit =");
            crate::widgets::num_field(ui, egui::DragValue::new(&mut d.unit).update_while_editing(false)
                    .speed(widgets::rated(ui, 0.001))
                    .range(1e-6..=1e6)
                    .suffix("m"),
            )
            .on_hover_text(
                "metres per depth unit (0.01 for cm). Scales the \
                 blur, never the focal plane",
            );
            for (lbl, u) in [("m", 1.0f32), ("cm", 0.01), ("mm", 0.001)] {
                if ui.add(egui::Button::selectable(d.unit == u, lbl)).clicked() {
                    d.unit = u;
                }
            }
        });
        // No channel pick here any more: Z is a DATA
        // port, and the one channel on it was picked by
        // whatever put it there (a ToData). Two picks in
        // series is a way to get the wrong number.
        ui.horizontal(|ui| {
            ui.checkbox(&mut d.inverse_depth, "1/Z");
            ui.weak("channel: on the ToData feeding Z");
        });
        // The depth pass is one sample a pixel under an
        // antialiased colour: an edge pixel wears the
        // wrong depth and halos. `edge` takes the
        // NEAREST depth within n px, so the rim travels
        // with its object.
        for h in [sl(ui, rcx, "edge", &mut d.edge, 0.0, 4.0,
            anim::Channel::DefocusEdge)]
        .into_iter()
        .flatten()
        {
            hud_drag_out = Some(h);
        }
        ui.weak("edge: take the nearer depth within n px —");
        ui.weak("stops halos on antialiased edges.");
    } else {
        ui.weak("no depth wired — uniform disc at max coc.");
        ui.weak("wire a depth branch into the third input");
        ui.weak("for a focal plane.");
    }
    if d.mode == DefocusMode::Accurate {
        ui.horizontal(|ui| {
            ui.label("samples");
            crate::widgets::num_field(ui, egui::DragValue::new(&mut d.samples).update_while_editing(false)
                    .speed(widgets::rated(ui, 4))
                    .range(8..=1024),
            )
            .on_hover_text("taps per pixel — rings mean too few");
            ui.label("blades");
            crate::widgets::num_field(ui, egui::DragValue::new(&mut d.blades).update_while_editing(false)
                    .speed(widgets::rated(ui, 1))
                    .range(0..=11),
            )
            .on_hover_text("0 = circular iris");
            // 1 and 2 are not apertures.
            if (1..3).contains(&d.blades) {
                d.blades = 3;
            }
        });
        for h in [
            sl(ui, rcx, "iris rot", &mut d.rotation, 0.0, 180.0,
                anim::Channel::DefocusRotation),
            sl(ui, rcx, "cat's-eye", &mut d.catseye, 0.0, 1.0,
                anim::Channel::DefocusCatseye),
            sl(ui, rcx, "clamp", &mut d.bright_clamp, 0.0, 16.0,
                anim::Channel::DefocusClamp),
        ]
        .into_iter()
        .flatten()
        {
            hud_drag_out = Some(h);
        }
        if wired[3] {
            ui.weak("bokeh shape from B (blades are ignored).");
        } else {
            ui.weak(if d.blades == 0 {
                "circular iris. wire a Bokeh Kernel, or any"
            } else {
                "polygonal iris. wire a Bokeh Kernel, or any"
            });
            ui.weak("picture, into B for its shape instead.");
        }
        ui.weak("clamp: soft-limit a tap's brightness before it");
        ui.weak("spreads (0 = off) — tames dotted HDR highlights.");
        ui.colored_label(
            theme::cost_note(),
            "SLOW PATH: taps per pixel, every pixel.",
        );
        ui.weak("Arbitrary kernel + occlusion test. `samples` is");
        ui.weak("the count at max coc; smaller discs take fewer.");
        ui.weak("4K: ~32ms at 128 samples with the frame at max coc.");
    } else {
        if wired[3] {
            ui.weak("B (the kernel) is used in accurate mode only:");
            ui.weak("fast is always a flat disc.");
        }
        ui.weak("exact flat disc via row prefix sums: O(r), not");
        ui.weak("O(r²). 4K: 7ms at r=8, 20ms at 32, 73ms at 128.");
        ui.weak("Two layers with depth: foreground bleeds out over");
        ui.weak("the background, which is filled from around it.");
        ui.weak("Without depth, gather only — nothing spreads out of a blurry");
        ui.weak("foreground onto a sharp background.");
    }
    hud_drag_out
}
