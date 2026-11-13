//! Glint Kernel's app half: the format row, the star's shape, a tint row
//! and a framed-star glyph. The params and the WGSL are
//! `engine::nodes::glintkernel`.

use crate::anim::Channel;
use crate::rgbslider::RgbSpec;
use crate::widgets::{self, HudSlider, RowCtx};
use egui::{Color32, Painter, Pos2, Stroke};
use engine::color::{primaries_matrix, Colorspace, Primaries};
use engine::graph::GlintKernelParams;
use engine::nodes::glintkernel::MAX_RAYS;
use std::collections::BTreeSet;

pub const CHANNELS: &[Channel] = &[
    Channel::GlintKernelRays,
    Channel::GlintKernelLength,
    Channel::GlintKernelThickness,
    Channel::GlintKernelFalloff,
    Channel::GlintKernelRotation,
    Channel::GlintKernelAlternate,
    Channel::GlintKernelCore,
    Channel::GlintKernelCoreSize,
    Channel::GlintKernelChroma,
    Channel::GlintKernelIntensity,
    Channel::GlintKernelTintR,
    Channel::GlintKernelTintG,
    Channel::GlintKernelTintB,
];

/// An rgb row needs two tiles of track.
pub const TILES: usize = 2;
/// The format row, ten sliders and the tint.
pub const TILE_ROWS: usize = 2;

pub fn get(p: &GlintKernelParams, ch: Channel) -> Option<f32> {
    use Channel::*;
    Some(match ch {
        GlintKernelRays => p.rays,
        GlintKernelLength => p.length,
        GlintKernelThickness => p.thickness,
        GlintKernelFalloff => p.falloff,
        GlintKernelRotation => p.rotation,
        GlintKernelAlternate => p.alternate,
        GlintKernelCore => p.core,
        GlintKernelCoreSize => p.core_size,
        GlintKernelChroma => p.chroma,
        GlintKernelIntensity => p.intensity,
        GlintKernelTintR => p.tint[0],
        GlintKernelTintG => p.tint[1],
        GlintKernelTintB => p.tint[2],
        _ => return None,
    })
}

pub fn set(p: &mut GlintKernelParams, ch: Channel, v: f32) {
    use Channel::*;
    match ch {
        GlintKernelRays => p.rays = v,
        GlintKernelLength => p.length = v,
        GlintKernelThickness => p.thickness = v,
        GlintKernelFalloff => p.falloff = v,
        GlintKernelRotation => p.rotation = v,
        GlintKernelAlternate => p.alternate = v,
        GlintKernelCore => p.core = v,
        GlintKernelCoreSize => p.core_size = v,
        GlintKernelChroma => p.chroma = v,
        GlintKernelIntensity => p.intensity = v,
        GlintKernelTintR => p.tint[0] = v,
        GlintKernelTintG => p.tint[1] = v,
        GlintKernelTintB => p.tint[2] = v,
        _ => {}
    }
}

pub fn label(ch: Channel) -> Option<&'static str> {
    use Channel::*;
    Some(match ch {
        GlintKernelRays => "glint kernel rays",
        GlintKernelLength => "glint kernel length",
        GlintKernelThickness => "glint kernel thickness",
        GlintKernelFalloff => "glint kernel falloff",
        GlintKernelRotation => "glint kernel rotation",
        GlintKernelAlternate => "glint kernel alternate",
        GlintKernelCore => "glint kernel core",
        GlintKernelCoreSize => "glint kernel core size",
        GlintKernelChroma => "glint kernel chroma",
        GlintKernelIntensity => "glint kernel intensity",
        GlintKernelTintR => "glint kernel tint.r",
        GlintKernelTintG => "glint kernel tint.g",
        GlintKernelTintB => "glint kernel tint.b",
        _ => return None,
    })
}

pub fn panel(
    ui: &mut egui::Ui,
    rcx: &mut RowCtx,
    p: &mut GlintKernelParams,
    working: Colorspace,
    ungang: &mut BTreeSet<String>,
) -> Option<HudSlider> {
    use Channel::*;
    widgets::gen_format_row(ui, ("glint-kernel-format", rcx.node), &mut p.size, &mut p.inherit);
    let mut h = None;
    for got in [
        widgets::slider(ui, rcx, "rays", &mut p.rays, 1.0, MAX_RAYS as f32, &[GlintKernelRays]),
        widgets::slider(ui, rcx, "length", &mut p.length, 0.0, 1.0, &[GlintKernelLength]),
        widgets::slider(ui, rcx, "thickness", &mut p.thickness, 0.3, 12.0, &[GlintKernelThickness]),
        widgets::slider(ui, rcx, "falloff", &mut p.falloff, 0.0, 1.0, &[GlintKernelFalloff]),
        widgets::slider(ui, rcx, "rotation", &mut p.rotation, -180.0, 180.0, &[GlintKernelRotation]),
        widgets::slider(ui, rcx, "alternate", &mut p.alternate, 0.0, 1.0, &[GlintKernelAlternate]),
        widgets::slider(ui, rcx, "core", &mut p.core, 0.0, 4.0, &[GlintKernelCore]),
        widgets::slider(ui, rcx, "core size", &mut p.core_size, 0.0, 0.3, &[GlintKernelCoreSize]),
        widgets::slider(ui, rcx, "chroma", &mut p.chroma, -1.0, 1.0, &[GlintKernelChroma]),
        widgets::slider(ui, rcx, "intensity", &mut p.intensity, 0.0, 4.0, &[GlintKernelIntensity]),
    ]
    .into_iter()
    .flatten()
    {
        h = Some(got);
    }
    let mut gang = !ungang.contains("tint");
    if let Some(got) = widgets::rgb_row(
        ui,
        rcx,
        "tint",
        &mut p.tint,
        &mut gang,
        RgbSpec::scale(0.0, 4.0, working.primaries.luma_weights())
            .with_display(primaries_matrix(working.primaries, Primaries::Rec709)),
        &[GlintKernelTintR, GlintKernelTintG, GlintKernelTintB],
    ) {
        h = Some(got);
    }
    if gang {
        ungang.remove("tint");
    } else {
        ungang.insert("tint".to_string());
    }
    ui.weak("a star to wire into a Convolve's K port, under a");
    ui.weak("highlight key: every ping wears it.");
    h
}

/// A star in a frame: a kernel is a picture of the effect.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let frame = egui::Rect::from_center_size(c, egui::vec2(r * 1.8, r * 1.8));
    painter.rect(frame, r * 0.15, fill, stroke, egui::StrokeKind::Middle);
    for k in 0..4 {
        let a = k as f32 * std::f32::consts::TAU / 4.0;
        let d = egui::vec2(a.cos(), a.sin());
        painter.line_segment([c - d * r * 0.75, c + d * r * 0.75], stroke);
    }
}

#[cfg(test)]
mod fit_tests {
    use super::*;
    use engine::graph::NodeKind;

    /// Hosted at its declared tiles, the panel fits both ways (an
    /// overflowed row cannot be clicked).
    #[test]
    fn the_panel_fits_the_tiles_it_asks_for() {
        let mut p = GlintKernelParams::default();
        let mut anim = Default::default();
        let ekind = NodeKind::GlintKernel(p.clone());
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
                let _ = panel(ui, &mut rcx, &mut p, engine::color::Colorspace::new(engine::color::Primaries::Rec709, engine::color::Curve::Linear), &mut Default::default());
            });
            over.set(out.content_size.y - out.inner_rect.height());
            wide.set(out.content_size.x - out.inner_rect.width());
        });
        shot.save("glintkernel-panel");
        assert!(over.get() <= 0.5, "overflows its tiles by {}pt", over.get());
        assert!(wide.get() <= 0.5, "overflows its tiles SIDEWAYS by {}pt", wide.get());
    }
}
