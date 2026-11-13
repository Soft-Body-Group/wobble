//! The Physics node's fluid rows (`engine::geophys::fluid`): the water's
//! resolution and thickness, then how it is drawn. The first three change
//! the simulation (bake again); the look rows are read at playback.

use engine::geophys::{FluidLook, FluidSettings};

use crate::widgets;

pub fn rows(ui: &mut egui::Ui, f: &mut FluidSettings) {
    // Resolution as particles across the canvas: a number people can feel,
    // where a spacing in canvas widths is not.
    let mut across = 1.0 / f.spacing.max(1e-4);
    if widgets::plain_slider(ui, "resolution", &mut across, 30.0, 400.0, true)
        .on_hover_text("particles across the canvas width — finer is slower (the count goes as the square)")
        .changed()
    {
        f.spacing = 1.0 / across.max(1.0);
    }
    widgets::plain_slider(ui, "viscosity", &mut f.viscosity, 0.0, 1.0, false)
        .on_hover_text("0 splashy water, 1 thick honey");
    let mut it = f.iterations as f32;
    if widgets::plain_slider(ui, "iterations", &mut it, 1.0, 8.0, true)
        .on_hover_text("density corrections a substep: more is less squashy, and slower")
        .changed()
    {
        f.iterations = it.round() as u32;
    }
    ui.horizontal(|ui| {
        ui.add_sized([52.0, 16.0], egui::Label::new(egui::RichText::new("look").small()));
        for l in FluidLook::ALL {
            ui.selectable_value(&mut f.look, l, l.name());
        }
    });
    if f.look != FluidLook::Particles {
        widgets::plain_slider(ui, "blob", &mut f.blob, 0.8, 4.0, false)
            .on_hover_text("how far the surface reaches round each particle — no re-bake");
    }
    if f.look != FluidLook::Surface {
        widgets::plain_slider(ui, "dot size", &mut f.dot, 0.2, 3.0, false).on_hover_text("no re-bake");
    }
    ui.weak("bodies input: closed shapes fill with water in their own colour · pinned shapes stay solid");
}
