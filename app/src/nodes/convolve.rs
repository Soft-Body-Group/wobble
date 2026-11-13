//! Convolve's app half: size + preset picks, the weight grid, a mix slider
//! and a 3x3-dots glyph.

use crate::anim::Channel;
use crate::widgets::{self, HudSlider, RowCtx};
use egui::{Color32, Painter, Pos2, Stroke};
use engine::graph::{ConvolveParams, ConvolvePreset, ConvolveRes, CONVOLVE_MAX};

/// Only the mix is a channel: 49 keyable weights would bury the curve
/// editor, and a kernel is typed, not animated.
pub const CHANNELS: &[Channel] = &[Channel::ConvolveMix];

pub const TILES: usize = 2;
/// Tall enough for the 7x7 grid under its header (the fit test checks).
pub const TILE_ROWS: usize = 1;

/// One weight box's width: seven of them fit across two tiles.
const CELL_W: f32 = 40.0;

pub fn get(p: &ConvolveParams, ch: Channel) -> Option<f32> {
    match ch {
        Channel::ConvolveMix => Some(p.mix),
        _ => None,
    }
}

pub fn set(p: &mut ConvolveParams, ch: Channel, v: f32) {
    if ch == Channel::ConvolveMix {
        p.mix = v;
    }
}

pub fn label(ch: Channel) -> Option<&'static str> {
    match ch {
        Channel::ConvolveMix => Some("convolve mix"),
        _ => None,
    }
}

/// `kernel`: a picture is wired into K, so it is the kernel (an FFT) and
/// the typed grid is not used.
pub fn panel(ui: &mut egui::Ui, rcx: &mut RowCtx, p: &mut ConvolveParams, kernel: bool) -> Option<HudSlider> {
    if kernel {
        ui.horizontal(|ui| {
            ui.label("grid").on_hover_text(
                "the resolution the FFT runs at; it steps down further on its own \
                 when a side would not fit",
            );
            for r in ConvolveRes::ALL {
                if ui.selectable_label(p.res == r, r.label()).clicked() {
                    p.res = r;
                }
            }
        });
        ui.horizontal(|ui| {
            ui.checkbox(&mut p.normalize, "normalise")
                .on_hover_text("divide by the kernel's sum, so the picture keeps its level");
            ui.checkbox(&mut p.alpha, "alpha").on_hover_text("convolve alpha too; off, it passes");
        });
        let h = widgets::slider(ui, rcx, "mix", &mut p.mix, 0.0, 1.0, &[Channel::ConvolveMix]);
        ui.weak("kernel: the picture on K, centred on its middle,");
        ui.weak("one pixel per pixel. An FFT: size costs nothing.");
        ui.weak("~ scales the mix per pixel.");
        return h;
    }
    ui.horizontal(|ui| {
        ui.label("size").on_hover_text(
            "or wire a picture into K (a Glint Kernel, a Bokeh Kernel) and it is the \
             kernel, at any size",
        );
        for s in [3u32, 5, 7] {
            if ui.selectable_label(p.side() == s as usize, format!("{s}x{s}")).clicked() {
                p.size = s;
            }
        }
        ui.add_space(8.0);
        egui::ComboBox::from_id_salt(("convolve-preset", rcx.node))
            .selected_text("preset")
            .show_ui(ui, |ui| {
                for pr in ConvolvePreset::ALL {
                    if ui.selectable_label(false, pr.name()).clicked() {
                        let keep = (p.mix, p.alpha, p.matte, p.res);
                        *p = ConvolveParams::preset(pr);
                        (p.mix, p.alpha, p.matte, p.res) = keep;
                    }
                }
            });
    });
    if p.kernel.len() < CONVOLVE_MAX * CONVOLVE_MAX {
        p.kernel.resize(CONVOLVE_MAX * CONVOLVE_MAX, 0.0);
    }
    ui.scope(|ui| {
        ui.spacing_mut().item_spacing.y = 2.0;
        for y in 0..CONVOLVE_MAX {
            ui.horizontal(|ui| {
                ui.spacing_mut().item_spacing.x = 3.0;
                for x in 0..CONVOLVE_MAX {
                    let used = p.in_use(x, y);
                    let w = &mut p.kernel[y * CONVOLVE_MAX + x];
                    ui.add_enabled_ui(used, |ui| {
                        widgets::num_field_w(ui, CELL_W, egui::DragValue::new(w).speed(0.05));
                    });
                }
            });
        }
    });
    ui.horizontal(|ui| {
        ui.checkbox(&mut p.normalize, "normalise").on_hover_text(
            "divide by the weights' sum (left alone when it is 0, as for an edge kernel)",
        );
        ui.checkbox(&mut p.alpha, "alpha").on_hover_text("convolve alpha too; off, it passes");
    });
    let h = widgets::slider(ui, rcx, "mix", &mut p.mix, 0.0, 1.0, &[Channel::ConvolveMix]);
    ui.weak(format!("{} taps a pixel, edges clamped", p.side() * p.side()));
    h
}

/// A 3x3 grid of dots, the centre one filled: a kernel.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let s = r * 0.55;
    for y in -1..=1 {
        for x in -1..=1 {
            let q = Pos2::new(c.x + x as f32 * s, c.y + y as f32 * s);
            if x == 0 && y == 0 {
                painter.circle(q, r * 0.22, fill, stroke);
            } else {
                painter.circle_stroke(q, r * 0.14, stroke);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::graph::NodeKind;

    /// The full 7x7 grid, header rows and mix at the declared tiles: no
    /// overflow either way (an overflowed row cannot be clicked).
    #[test]
    fn the_panel_fits_the_tiles_it_asks_for() {
        let mut p = ConvolveParams { size: 7, ..Default::default() };
        let mut anim = Default::default();
        let ekind = NodeKind::Convolve(p.clone());
        let over = std::cell::Cell::new(0.0f32);
        let wide = std::cell::Cell::new(0.0f32);
        let span = |n: usize| {
            crate::widgets::TILE * n as f32 + crate::widgets::TILE_GAP * (n as f32 - 1.0)
        };
        let (w, hgt) = (span(TILES) - 14.0, span(TILE_ROWS) - 14.0 - 22.0);
        let shot = crate::snapshot::shoot(w, hgt, |ui| {
            let out = egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                let mut rcx = RowCtx {
                    node: 1,
                    anim: &mut anim,
                    ekind: &ekind,
                    t: 0,
                    hud_live: false,
                    driven: Vec::new(),
                };
                let _ = panel(ui, &mut rcx, &mut p, false);
            });
            over.set(out.content_size.y - out.inner_rect.height());
            wide.set(out.content_size.x - out.inner_rect.width());
        });
        shot.save("convolve-panel");
        assert!(over.get() <= 0.5, "overflows its tiles by {}pt", over.get());
        assert!(wide.get() <= 0.5, "overflows its tiles SIDEWAYS by {}pt", wide.get());
    }
}
