//! Glint's app half: the key, the star's shape, an rgb tint row, the
//! combine mode, and a star glyph.

use crate::anim::Channel;
use crate::rgbslider::RgbSpec;
use crate::widgets::{self, HudSlider, RowCtx};
use egui::{Color32, Painter, Pos2, Stroke};
use engine::color::{primaries_matrix, Colorspace, Primaries};
use engine::graph::{GlintParams, GlowMode};
use engine::nodes::glint::MAX_RAYS;
use std::collections::BTreeSet;

pub const CHANNELS: &[Channel] = &[
    Channel::GlintThreshold,
    Channel::GlintKnee,
    Channel::GlintRays,
    Channel::GlintLength,
    Channel::GlintRotation,
    Channel::GlintFalloff,
    Channel::GlintIntensity,
    Channel::GlintTintR,
    Channel::GlintTintG,
    Channel::GlintTintB,
    Channel::GlintErode,
    Channel::GlintSpread,
    Channel::GlintShift,
];

/// A row of rgb lanes needs track to aim along.
pub const TILES: usize = 2;
/// Eight sliders, the isolate row, the tint and the combine pick.
pub const TILE_ROWS: usize = 2;

pub fn get(p: &GlintParams, ch: Channel) -> Option<f32> {
    use Channel::*;
    Some(match ch {
        GlintThreshold => p.threshold,
        GlintKnee => p.knee,
        GlintRays => p.rays,
        GlintLength => p.length,
        GlintRotation => p.rotation,
        GlintFalloff => p.falloff,
        GlintIntensity => p.intensity,
        GlintTintR => p.tint[0],
        GlintTintG => p.tint[1],
        GlintTintB => p.tint[2],
        GlintErode => p.erode,
        GlintSpread => p.spread,
        GlintShift => p.shift,
        _ => return None,
    })
}

pub fn set(p: &mut GlintParams, ch: Channel, v: f32) {
    use Channel::*;
    match ch {
        GlintThreshold => p.threshold = v,
        GlintKnee => p.knee = v,
        GlintRays => p.rays = v,
        GlintLength => p.length = v,
        GlintRotation => p.rotation = v,
        GlintFalloff => p.falloff = v,
        GlintIntensity => p.intensity = v,
        GlintTintR => p.tint[0] = v,
        GlintTintG => p.tint[1] = v,
        GlintTintB => p.tint[2] = v,
        GlintErode => p.erode = v,
        GlintSpread => p.spread = v,
        GlintShift => p.shift = v,
        _ => {}
    }
}

pub fn label(ch: Channel) -> Option<&'static str> {
    use Channel::*;
    Some(match ch {
        GlintThreshold => "glint threshold",
        GlintKnee => "glint knee",
        GlintRays => "glint rays",
        GlintLength => "glint length",
        GlintRotation => "glint rotation",
        GlintFalloff => "glint falloff",
        GlintIntensity => "glint intensity",
        GlintTintR => "glint tint.r",
        GlintTintG => "glint tint.g",
        GlintTintB => "glint tint.b",
        GlintErode => "glint isolate size",
        GlintSpread => "glint spread",
        GlintShift => "glint hue shift",
        _ => return None,
    })
}

/// `ungang` is the node's set of rows whose lanes are un-ganged; the tint
/// row keeps its flag there under "tint".
pub fn panel(
    ui: &mut egui::Ui,
    rcx: &mut RowCtx,
    p: &mut GlintParams,
    working: Colorspace,
    ungang: &mut BTreeSet<String>,
) -> Option<HudSlider> {
    use Channel::*;
    let mut h = None;
    for got in [
        widgets::slider(ui, rcx, "threshold", &mut p.threshold, 0.0, 8.0, &[GlintThreshold]),
        widgets::slider(ui, rcx, "knee", &mut p.knee, 0.0, 2.0, &[GlintKnee]),
        widgets::slider(ui, rcx, "rays", &mut p.rays, 0.0, MAX_RAYS as f32, &[GlintRays]),
        widgets::slider(ui, rcx, "length", &mut p.length, 0.0, 1000.0, &[GlintLength]),
        widgets::slider(ui, rcx, "rotation", &mut p.rotation, -180.0, 180.0, &[GlintRotation]),
        widgets::slider(ui, rcx, "falloff", &mut p.falloff, 0.0, 1.0, &[GlintFalloff]),
        widgets::slider(ui, rcx, "intensity", &mut p.intensity, 0.0, 4.0, &[GlintIntensity]),
        widgets::slider(ui, rcx, "spread", &mut p.spread, -1.0, 1.0, &[GlintSpread]),
        widgets::slider(ui, rcx, "hue shift", &mut p.shift, -1.0, 1.0, &[GlintShift]),
    ]
    .into_iter()
    .flatten()
    {
        h = Some(got);
    }
    ui.horizontal(|ui| {
        ui.checkbox(&mut p.isolate, "isolate pings").on_hover_text(
            "take away every highlight wider than the size before the rays are cast, \
             so the glints come off small specks and not off a bright window",
        );
    });
    if p.isolate {
        if let Some(got) = widgets::slider(ui, rcx, "size", &mut p.erode, 0.0, 64.0, &[GlintErode]) {
            h = Some(got);
        }
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
        &[GlintTintR, GlintTintG, GlintTintB],
    ) {
        h = Some(got);
    }
    if gang {
        ungang.remove("tint");
    } else {
        ungang.insert("tint".to_string());
    }
    ui.horizontal(|ui| {
        ui.label("combine");
        egui::ComboBox::from_id_salt(("glint-mode", rcx.node))
            .selected_text(p.mode.name())
            .show_ui(ui, |ui| {
                for m in GlowMode::ALL {
                    ui.selectable_value(&mut p.mode, m, m.name());
                }
            });
    });
    ui.weak("rays off everything over the threshold, at half");
    ui.weak("res; length costs no extra taps. spread / hue");
    ui.weak("shift colour the rays (3 taps a step instead of 1).");
    h
}

/// A star: four long rays on the axes, four short ones between.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.circle(c, r * 0.32, fill, stroke);
    for k in 0..8 {
        let a = k as f32 * std::f32::consts::TAU / 8.0;
        let d = egui::vec2(a.cos(), a.sin());
        // Four long rays on the axes, four short ones between.
        let reach = if k % 2 == 0 { 1.25 } else { 0.75 };
        painter.line_segment([c + d * r * 0.42, c + d * r * reach], stroke);
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
        let mut p = GlintParams { isolate: true, ..Default::default() };
        let mut anim = Default::default();
        let ekind = NodeKind::Glint(p.clone());
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
        shot.save("glint-panel");
        assert!(over.get() <= 0.5, "overflows its tiles by {}pt", over.get());
        assert!(wide.get() <= 0.5, "overflows its tiles SIDEWAYS by {}pt", wide.get());
    }
}
