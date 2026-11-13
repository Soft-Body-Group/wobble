//! Glow's app half: the pyramid's five sliders, an rgb tint row, the
//! combine mode, and a light-source glyph.

use crate::anim::Channel;
use crate::rgbslider::RgbSpec;
use crate::widgets::{self, HudSlider, RowCtx};
use egui::{Color32, Painter, Pos2, Stroke};
use engine::color::{primaries_matrix, Colorspace, Primaries};
use engine::graph::{GlowMode, GlowParams};
use std::collections::BTreeSet;

pub const CHANNELS: &[Channel] = &[
    Channel::GlowSize,
    Channel::GlowFalloff,
    Channel::GlowThreshold,
    Channel::GlowKnee,
    Channel::GlowIntensity,
    Channel::GlowTintR,
    Channel::GlowTintG,
    Channel::GlowTintB,
];

/// A row of rgb lanes needs track to aim along.
pub const TILES: usize = 2;

pub fn get(p: &GlowParams, ch: Channel) -> Option<f32> {
    use Channel::*;
    Some(match ch {
        GlowSize => p.size,
        GlowFalloff => p.falloff,
        GlowThreshold => p.threshold,
        GlowKnee => p.knee,
        GlowIntensity => p.intensity,
        GlowTintR => p.tint[0],
        GlowTintG => p.tint[1],
        GlowTintB => p.tint[2],
        _ => return None,
    })
}

pub fn set(p: &mut GlowParams, ch: Channel, v: f32) {
    use Channel::*;
    match ch {
        GlowSize => p.size = v,
        GlowFalloff => p.falloff = v,
        GlowThreshold => p.threshold = v,
        GlowKnee => p.knee = v,
        GlowIntensity => p.intensity = v,
        GlowTintR => p.tint[0] = v,
        GlowTintG => p.tint[1] = v,
        GlowTintB => p.tint[2] = v,
        _ => {}
    }
}

pub fn label(ch: Channel) -> Option<&'static str> {
    use Channel::*;
    Some(match ch {
        GlowSize => "glow size",
        GlowFalloff => "glow falloff",
        GlowThreshold => "glow threshold",
        GlowKnee => "glow knee",
        GlowIntensity => "glow intensity",
        GlowTintR => "glow tint.r",
        GlowTintG => "glow tint.g",
        GlowTintB => "glow tint.b",
        _ => return None,
    })
}

/// `ungang` is the node's set of rows whose lanes are un-ganged; the tint
/// row keeps its flag there under "tint".
pub fn panel(
    ui: &mut egui::Ui,
    rcx: &mut RowCtx,
    p: &mut GlowParams,
    working: Colorspace,
    ungang: &mut BTreeSet<String>,
) -> Option<HudSlider> {
    use Channel::*;
    let mut h = None;
    for got in [
        widgets::slider(ui, rcx, "size", &mut p.size, 1.0, 512.0, &[GlowSize]),
        widgets::slider(ui, rcx, "falloff", &mut p.falloff, 0.0, 1.0, &[GlowFalloff]),
        widgets::slider(ui, rcx, "threshold", &mut p.threshold, 0.0, 8.0, &[GlowThreshold]),
        widgets::slider(ui, rcx, "knee", &mut p.knee, 0.0, 2.0, &[GlowKnee]),
        widgets::slider(ui, rcx, "intensity", &mut p.intensity, 0.0, 4.0, &[GlowIntensity]),
    ]
    .into_iter()
    .flatten()
    {
        h = Some(got);
    }
    // The tint is a per-channel multiplier, which is exactly what the rgb
    // slider is for — a colour button could not gang it, key it or reach
    // the S/D/F axes.
    let mut gang = !ungang.contains("tint");
    if let Some(got) = widgets::rgb_row(
        ui,
        rcx,
        "tint",
        &mut p.tint,
        &mut gang,
        RgbSpec::scale(0.0, 4.0, working.primaries.luma_weights())
            .with_display(primaries_matrix(working.primaries, Primaries::Rec709)),
        &[GlowTintR, GlowTintG, GlowTintB],
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
        egui::ComboBox::from_id_salt(("glow-mode", rcx.node))
            .selected_text(p.mode.name())
            .show_ui(ui, |ui| {
                for m in GlowMode::ALL {
                    ui.selectable_value(&mut p.mode, m, m.name());
                }
            });
    });
    ui.weak("mip-pyramid bloom: octaves of 2, 4, 8… px summed");
    ui.weak("with falling weights. Size picks how far up the");
    ui.weak("pyramid the sum reaches — it costs no extra taps.");
    h
}

/// Core plus rays: a light source blooming.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.circle(c, r * 0.45, fill, stroke);
    for k in 0..8 {
        let a = k as f32 * std::f32::consts::TAU / 8.0;
        let d = egui::vec2(a.cos(), a.sin());
        painter.line_segment([c + d * r * 0.65, c + d * r * 1.15], stroke);
    }
}
