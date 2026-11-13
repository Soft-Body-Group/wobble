//! Bokeh Kernel's app half: the format row, the aperture's shape and a
//! framed-hexagon glyph. The params and the WGSL are
//! `engine::nodes::bokehkernel`.

use crate::anim::Channel;
use crate::widgets::{self, HudSlider, RowCtx};
use egui::{Color32, Painter, Pos2, Stroke};
use engine::graph::BokehKernelParams;

pub const CHANNELS: &[Channel] = &[
    Channel::BokehKernelBlades,
    Channel::BokehKernelRoundness,
    Channel::BokehKernelRotation,
    Channel::BokehKernelRadius,
    Channel::BokehKernelSoftness,
    Channel::BokehKernelCatseye,
    Channel::BokehKernelCatseyeAngle,
    Channel::BokehKernelRing,
    Channel::BokehKernelFringe,
    Channel::BokehKernelSqueeze,
    Channel::BokehKernelIntensity,
];

pub const TILES: usize = 2;
/// The format row and eleven sliders.
pub const TILE_ROWS: usize = 2;

pub fn get(p: &BokehKernelParams, ch: Channel) -> Option<f32> {
    use Channel::*;
    Some(match ch {
        BokehKernelBlades => p.blades,
        BokehKernelRoundness => p.roundness,
        BokehKernelRotation => p.rotation,
        BokehKernelRadius => p.radius,
        BokehKernelSoftness => p.softness,
        BokehKernelCatseye => p.catseye,
        BokehKernelCatseyeAngle => p.catseye_angle,
        BokehKernelRing => p.ring,
        BokehKernelFringe => p.fringe,
        BokehKernelSqueeze => p.squeeze,
        BokehKernelIntensity => p.intensity,
        _ => return None,
    })
}

pub fn set(p: &mut BokehKernelParams, ch: Channel, v: f32) {
    use Channel::*;
    match ch {
        BokehKernelBlades => p.blades = v,
        BokehKernelRoundness => p.roundness = v,
        BokehKernelRotation => p.rotation = v,
        BokehKernelRadius => p.radius = v,
        BokehKernelSoftness => p.softness = v,
        BokehKernelCatseye => p.catseye = v,
        BokehKernelCatseyeAngle => p.catseye_angle = v,
        BokehKernelRing => p.ring = v,
        BokehKernelFringe => p.fringe = v,
        BokehKernelSqueeze => p.squeeze = v,
        BokehKernelIntensity => p.intensity = v,
        _ => {}
    }
}

pub fn label(ch: Channel) -> Option<&'static str> {
    use Channel::*;
    Some(match ch {
        BokehKernelBlades => "bokeh blades",
        BokehKernelRoundness => "bokeh roundness",
        BokehKernelRotation => "bokeh rotation",
        BokehKernelRadius => "bokeh radius",
        BokehKernelSoftness => "bokeh softness",
        BokehKernelCatseye => "bokeh cat's-eye",
        BokehKernelCatseyeAngle => "bokeh cat's-eye angle",
        BokehKernelRing => "bokeh ring",
        BokehKernelFringe => "bokeh fringe",
        BokehKernelSqueeze => "bokeh squeeze",
        BokehKernelIntensity => "bokeh intensity",
        _ => return None,
    })
}

pub fn panel(ui: &mut egui::Ui, rcx: &mut RowCtx, p: &mut BokehKernelParams) -> Option<HudSlider> {
    use Channel::*;
    widgets::gen_format_row(ui, ("bokeh-kernel-format", rcx.node), &mut p.size, &mut p.inherit);
    let mut h = None;
    for got in [
        widgets::slider(ui, rcx, "blades", &mut p.blades, 0.0, 16.0, &[BokehKernelBlades]),
        widgets::slider(ui, rcx, "roundness", &mut p.roundness, 0.0, 1.0, &[BokehKernelRoundness]),
        widgets::slider(ui, rcx, "rotation", &mut p.rotation, -180.0, 180.0, &[BokehKernelRotation]),
        widgets::slider(ui, rcx, "radius", &mut p.radius, 0.0, 1.0, &[BokehKernelRadius]),
        widgets::slider(ui, rcx, "softness", &mut p.softness, 0.3, 16.0, &[BokehKernelSoftness]),
        widgets::slider(ui, rcx, "cat's-eye", &mut p.catseye, 0.0, 1.0, &[BokehKernelCatseye]),
        widgets::slider(ui, rcx, "eye angle", &mut p.catseye_angle, -180.0, 180.0, &[BokehKernelCatseyeAngle]),
        widgets::slider(ui, rcx, "ring", &mut p.ring, -1.0, 1.0, &[BokehKernelRing]),
        widgets::slider(ui, rcx, "fringe", &mut p.fringe, -1.0, 1.0, &[BokehKernelFringe]),
        widgets::slider(ui, rcx, "squeeze", &mut p.squeeze, 0.2, 1.0, &[BokehKernelSqueeze]),
        widgets::slider(ui, rcx, "intensity", &mut p.intensity, 0.0, 4.0, &[BokehKernelIntensity]),
    ]
    .into_iter()
    .flatten()
    {
        h = Some(got);
    }
    ui.weak("an aperture for a Defocus's B port (accurate)");
    ui.weak("or a Convolve's K port. blades < 3 = round.");
    h
}

/// A hexagonal iris in a frame.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let frame = egui::Rect::from_center_size(c, egui::vec2(r * 1.8, r * 1.8));
    painter.rect(frame, r * 0.15, Color32::TRANSPARENT, stroke, egui::StrokeKind::Middle);
    let pts: Vec<Pos2> = (0..6)
        .map(|k| {
            let a = k as f32 * std::f32::consts::TAU / 6.0;
            Pos2::new(c.x + a.cos() * r * 0.7, c.y + a.sin() * r * 0.7)
        })
        .collect();
    painter.add(egui::Shape::convex_polygon(pts, fill, stroke));
}

#[cfg(test)]
mod fit_tests {
    use super::*;
    use engine::graph::NodeKind;

    /// Hosted at its declared tiles, the panel fits both ways (an
    /// overflowed row cannot be clicked).
    #[test]
    fn the_panel_fits_the_tiles_it_asks_for() {
        let mut p = BokehKernelParams::default();
        let mut anim = Default::default();
        let ekind = NodeKind::BokehKernel(p.clone());
        let over = std::cell::Cell::new(0.0f32);
        let wide = std::cell::Cell::new(0.0f32);
        let span = |n: usize| {
            crate::widgets::TILE * n as f32 + crate::widgets::TILE_GAP * (n as f32 - 1.0)
        };
        let rows = crate::widgets::tile_rows(&ekind);
        let (w, hgt) = (span(crate::widgets::tiles_for(&ekind)) - 14.0, span(rows) - 14.0 - 22.0);
        let shot = crate::snapshot::shoot(w, hgt, |ui| {
            let out = egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                let mut rcx = crate::widgets::RowCtx {
                    node: 1,
                    anim: &mut anim,
                    ekind: &ekind,
                    t: 0,
                    hud_live: false,
                    driven: Vec::new(),
                };
                let _ = panel(ui, &mut rcx, &mut p);
            });
            over.set(out.content_size.y - out.inner_rect.height());
            wide.set(out.content_size.x - out.inner_rect.width());
        });
        shot.save("bokehkernel-panel");
        assert!(over.get() <= 0.5, "overflows its tiles by {}pt", over.get());
        assert!(wide.get() <= 0.5, "overflows its tiles SIDEWAYS by {}pt", wide.get());
    }
}
