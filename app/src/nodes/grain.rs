//! Grain's app half: the panel, the glyph and the channel bindings. The
//! model, the WGSL and the params are `engine::nodes::grain`.

use crate::anim::Channel;
use crate::rgbslider::RgbSpec;
use crate::widgets::{self, HudSlider, RowCtx};
use egui::{Color32, Painter, Pos2, Stroke};
use engine::color::{primaries_matrix, Colorspace, Primaries};
use engine::graph::{GrainApply, GrainParams, GrainStock};
use std::collections::BTreeSet;

pub const CHANNELS: &[Channel] = &[
    Channel::GrainSizeR,
    Channel::GrainSizeG,
    Channel::GrainSizeB,
    Channel::GrainAmountR,
    Channel::GrainAmountG,
    Channel::GrainAmountB,
    Channel::GrainIrrR,
    Channel::GrainIrrG,
    Channel::GrainIrrB,
    Channel::GrainSoftness,
    Channel::GrainMono,
    Channel::GrainAspect,
    Channel::GrainBase,
    Channel::GrainShadows,
    Channel::GrainMids,
    Channel::GrainHighs,
    Channel::GrainSeed,
];

/// Three rgb rows need track to aim along...
pub const TILES: usize = 2;
/// ...and the rows under them need a second tile of height.
pub const TILE_ROWS: usize = 2;

const SIZE: [Channel; 3] = [
    Channel::GrainSizeR,
    Channel::GrainSizeG,
    Channel::GrainSizeB,
];
const AMOUNT: [Channel; 3] = [
    Channel::GrainAmountR,
    Channel::GrainAmountG,
    Channel::GrainAmountB,
];
const IRR: [Channel; 3] = [Channel::GrainIrrR, Channel::GrainIrrG, Channel::GrainIrrB];

pub fn get(p: &GrainParams, ch: Channel) -> Option<f32> {
    use Channel::*;
    Some(match ch {
        GrainSizeR => p.grain[0],
        GrainSizeG => p.grain[1],
        GrainSizeB => p.grain[2],
        GrainAmountR => p.amount[0],
        GrainAmountG => p.amount[1],
        GrainAmountB => p.amount[2],
        GrainIrrR => p.irregularity[0],
        GrainIrrG => p.irregularity[1],
        GrainIrrB => p.irregularity[2],
        GrainSoftness => p.softness,
        GrainMono => p.mono,
        GrainAspect => p.aspect,
        GrainBase => p.base,
        GrainShadows => p.response[0],
        GrainMids => p.response[1],
        GrainHighs => p.response[2],
        GrainSeed => p.seed,
        _ => return None,
    })
}

pub fn set(p: &mut GrainParams, ch: Channel, v: f32) {
    use Channel::*;
    match ch {
        GrainSizeR => p.grain[0] = v,
        GrainSizeG => p.grain[1] = v,
        GrainSizeB => p.grain[2] = v,
        GrainAmountR => p.amount[0] = v,
        GrainAmountG => p.amount[1] = v,
        GrainAmountB => p.amount[2] = v,
        GrainIrrR => p.irregularity[0] = v,
        GrainIrrG => p.irregularity[1] = v,
        GrainIrrB => p.irregularity[2] = v,
        GrainSoftness => p.softness = v,
        GrainMono => p.mono = v,
        GrainAspect => p.aspect = v,
        GrainBase => p.base = v,
        GrainShadows => p.response[0] = v,
        GrainMids => p.response[1] = v,
        GrainHighs => p.response[2] = v,
        GrainSeed => p.seed = v,
        _ => {}
    }
}

pub fn label(ch: Channel) -> Option<&'static str> {
    use Channel::*;
    Some(match ch {
        GrainSizeR => "grain size.r",
        GrainSizeG => "grain size.g",
        GrainSizeB => "grain size.b",
        GrainAmountR => "grain amount.r",
        GrainAmountG => "grain amount.g",
        GrainAmountB => "grain amount.b",
        GrainIrrR => "grain irregularity.r",
        GrainIrrG => "grain irregularity.g",
        GrainIrrB => "grain irregularity.b",
        GrainSoftness => "grain softness",
        GrainMono => "grain mono",
        GrainAspect => "grain aspect",
        GrainBase => "grain base",
        GrainShadows => "grain shadows",
        GrainMids => "grain mids",
        GrainHighs => "grain highlights",
        GrainSeed => "grain seed",
        _ => return None,
    })
}

/// One rgb row in its own child, narrower by the chip's overhang — the
/// ColorCompress layout, for the same reason (see its `row`).
#[allow(clippy::too_many_arguments)]
fn row(
    ui: &mut egui::Ui,
    rcx: &mut RowCtx,
    ungang: &mut BTreeSet<String>,
    label: &str,
    v: &mut [f32; 3],
    spec: RgbSpec,
    chans: &[Channel; 3],
    out: &mut Option<HudSlider>,
) {
    let w = (ui.available_width() - ROW_OVERHANG).max(240.0);
    let mut c = ui.new_child(
        egui::UiBuilder::new()
            .max_rect(egui::Rect::from_min_size(
                ui.cursor().min,
                egui::vec2(w, 0.0),
            ))
            .layout(egui::Layout::top_down(egui::Align::Min)),
    );
    let mut gang = !ungang.contains(label);
    if let Some(h) = widgets::rgb_row(&mut c, rcx, label, v, &mut gang, spec, chans) {
        *out = Some(h);
    }
    ui.advance_cursor_after_rect(c.min_rect());
    if gang {
        ungang.remove(label);
    } else {
        ungang.insert(label.to_string());
    }
}

const ROW_OVERHANG: f32 = crate::rgbslider::SWATCH_W + 4.0;

/// The properties panel. `ungang` is the node's set of un-ganged rows;
/// `wired` is whether a plate feeds input 0 — grain ON a plate, or grain
/// generated on `base`.
pub fn panel(
    ui: &mut egui::Ui,
    rcx: &mut RowCtx,
    p: &mut GrainParams,
    working: Colorspace,
    ungang: &mut BTreeSet<String>,
    wired: bool,
) -> Option<HudSlider> {
    use Channel::*;
    let luma = working.primaries.luma_weights();
    let disp = primaries_matrix(working.primaries, Primaries::Rec709);
    let mut h: Option<HudSlider> = None;
    // A plate brings its own format; the row only means something unwired.
    if !wired {
        widgets::gen_format_row(ui, ("grain-format", rcx.node), &mut p.size, &mut p.inherit);
    }
    ui.horizontal(|ui| {
        ui.add_sized(
            [50.0, 16.0],
            egui::Label::new(egui::RichText::new("stock").small()),
        );
        egui::ComboBox::from_id_salt(("grain-stock", rcx.node))
            .selected_text("load a look…")
            .show_ui(ui, |ui| {
                for s in GrainStock::ALL {
                    if ui.selectable_label(false, s.label()).clicked() {
                        // Sizes are per pixel, so they scale with the
                        // format the grain is drawn at.
                        let w = if p.inherit || wired {
                            widgets::inherit_canvas().0
                        } else {
                            p.size.0
                        };
                        p.apply_stock(s, w);
                    }
                }
            })
            .response
            .on_hover_text(
                "sets size, amount, irregularity, softness and mono to a look shaped \
                 like that stock, scaled to this format",
            );
    });
    // Sizes and irregularity scale; amount is an offset from base.
    row(
        ui,
        rcx,
        ungang,
        "size",
        &mut p.grain,
        RgbSpec::scale(0.1, 12.0, luma).with_display(disp),
        &SIZE,
        &mut h,
    );
    row(
        ui,
        rcx,
        ungang,
        "amount",
        &mut p.amount,
        RgbSpec::scale(0.0, 0.5, luma).with_display(disp),
        &AMOUNT,
        &mut h,
    );
    row(
        ui,
        rcx,
        ungang,
        "irregular",
        &mut p.irregularity,
        RgbSpec::scale(0.0, 1.0, luma).with_display(disp),
        &IRR,
        &mut h,
    );
    for got in [
        widgets::slider(
            ui,
            rcx,
            "softness",
            &mut p.softness,
            0.05,
            1.0,
            &[GrainSoftness],
        ),
        widgets::slider(ui, rcx, "mono", &mut p.mono, 0.0, 1.0, &[GrainMono]),
        widgets::slider(ui, rcx, "aspect", &mut p.aspect, 0.25, 4.0, &[GrainAspect]),
        widgets::slider(ui, rcx, "seed", &mut p.seed, 0.0, 100.0, &[GrainSeed]),
    ]
    .into_iter()
    .flatten()
    {
        h = Some(got);
    }
    if wired {
        ui.horizontal(|ui| {
            ui.add_sized(
                [50.0, 16.0],
                egui::Label::new(egui::RichText::new("apply").small()),
            );
            egui::ComboBox::from_id_salt(("grain-apply", rcx.node))
                .selected_text(p.apply.label())
                .show_ui(ui, |ui| {
                    for a in GrainApply::ALL {
                        ui.selectable_value(&mut p.apply, a, a.label());
                    }
                })
                .response
                .on_hover_text(
                    "film: grain scales with the plate, as a negative's does — black stays \
                     black, amount is the grain at 18% grey. add: the same grain everywhere",
                );
        });
        if let Some(got) = widgets::drags(
            ui,
            rcx,
            "shad · mid · high",
            {
                let [a, b, c] = &mut p.response;
                &mut [a, b, c]
            },
            0.01,
            (0.0, 2.0),
            &[GrainShadows, GrainMids, GrainHighs],
        ) {
            h = Some(got);
        }
    } else if let Some(got) = widgets::slider(ui, rcx, "base", &mut p.base, 0.0, 1.0, &[GrainBase])
    {
        h = Some(got);
    }
    ui.horizontal(|ui| {
        ui.checkbox(&mut p.animated, "new grain every frame")
            .on_hover_text("off: one pattern held for the whole shot");
    });
    ui.weak(if wired {
        "amount is the grain's sd at 18% grey"
    } else {
        "no plate wired: grain on base, amount is its sd"
    });
    h
}

/// The graph glyph: a disc speckled with grains.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.circle(c, r * 0.9, fill, stroke);
    let ink = crate::theme::node_ink();
    // A fixed scatter, so the glyph is the same every frame.
    const DOTS: [(f32, f32, f32); 9] = [
        (-0.42, -0.30, 0.13),
        (0.10, -0.50, 0.09),
        (0.45, -0.18, 0.12),
        (-0.15, -0.05, 0.10),
        (0.25, 0.20, 0.08),
        (-0.50, 0.25, 0.09),
        (-0.05, 0.45, 0.12),
        (0.50, 0.40, 0.07),
        (0.05, 0.10, 0.06),
    ];
    for (x, y, s) in DOTS {
        painter.circle_filled(c + egui::vec2(x, y) * r, (s * r).max(0.8), ink);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::graph::NodeKind;

    /// The panel must fit the tiles it asks for, both ways — see
    /// ColorCompress's twin of this test for why it measures.
    #[test]
    fn the_panel_fits_the_tiles_it_asks_for() {
        let p = GrainParams::default();
        let mut ungang = BTreeSet::new();
        let mut anim = Default::default();
        let ekind = NodeKind::Grain(p.clone());
        for wired in [false, true] {
            let mut p = p.clone();
            let over = std::cell::Cell::new(0.0f32);
            let wide = std::cell::Cell::new(0.0f32);
            let span = |n: usize| {
                crate::widgets::TILE * n as f32 + crate::widgets::TILE_GAP * (n as f32 - 1.0)
            };
            let (w, hgt) = (span(TILES) - 14.0, span(TILE_ROWS) - 14.0 - 22.0);
            let shot = crate::snapshot::shoot(w, hgt, |ui| {
                let out = egui::ScrollArea::vertical()
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        let mut rcx = RowCtx {
                            node: 1,
                            anim: &mut anim,
                            ekind: &ekind,
                            t: 0,
                            hud_live: false,
                            driven: Vec::new(),
                        };
                        let _ = panel(ui, &mut rcx, &mut p, Colorspace::ACESCG, &mut ungang, wired);
                    });
                over.set(out.content_size.y - out.inner_rect.height());
                wide.set(out.content_size.x - out.inner_rect.width());
            });
            shot.save(if wired {
                "grain-panel-wired"
            } else {
                "grain-panel"
            });
            assert!(
                over.get() <= 0.5,
                "wired {wired}: the panel overflows its tiles by {}pt",
                over.get()
            );
            assert!(
                wide.get() <= 0.5,
                "wired {wired}: the panel overflows SIDEWAYS by {}pt",
                wide.get()
            );
        }
    }

    #[test]
    fn every_channel_round_trips() {
        let mut p = GrainParams::default();
        for (i, ch) in CHANNELS.iter().enumerate() {
            set(&mut p, *ch, 0.01 * (i + 1) as f32);
            assert_eq!(get(&p, *ch), Some(0.01 * (i + 1) as f32), "{ch:?}");
            assert!(label(*ch).is_some());
        }
    }
}
