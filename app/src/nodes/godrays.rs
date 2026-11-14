//! Godrays' app half: the source point, the ray sliders, an rgb tint row,
//! the combine mode, and a sun-behind-a-ridge glyph. The source point's
//! viewer handle lives with the other gizmos in `lib.rs`.

use crate::anim::Channel;
use crate::rgbslider::RgbSpec;
use crate::widgets::{self, HudSlider, RowCtx};
use egui::{Color32, Painter, Pos2, Stroke};
use engine::color::{primaries_matrix, Colorspace, Primaries};
use engine::graph::{GlowMode, GodraysParams};
use engine::nodes::godrays;
use std::collections::BTreeSet;

pub const CHANNELS: &[Channel] = &[
    Channel::GodraysX,
    Channel::GodraysY,
    Channel::GodraysLength,
    Channel::GodraysFalloff,
    Channel::GodraysThreshold,
    Channel::GodraysKnee,
    Channel::GodraysIntensity,
    Channel::GodraysTintR,
    Channel::GodraysTintG,
    Channel::GodraysTintB,
];

/// A row of rgb lanes needs track to aim along.
pub const TILES: usize = 2;
/// Two tall: the ray rows, the tint, the quality rows and the combine.
pub const TILE_ROWS: usize = 2;

pub fn get(p: &GodraysParams, ch: Channel) -> Option<f32> {
    use Channel::*;
    Some(match ch {
        GodraysX => p.center[0],
        GodraysY => p.center[1],
        GodraysLength => p.length,
        GodraysFalloff => p.falloff,
        GodraysThreshold => p.threshold,
        GodraysKnee => p.knee,
        GodraysIntensity => p.intensity,
        GodraysTintR => p.tint[0],
        GodraysTintG => p.tint[1],
        GodraysTintB => p.tint[2],
        _ => return None,
    })
}

pub fn set(p: &mut GodraysParams, ch: Channel, v: f32) {
    use Channel::*;
    match ch {
        GodraysX => p.center[0] = v,
        GodraysY => p.center[1] = v,
        GodraysLength => p.length = v,
        GodraysFalloff => p.falloff = v,
        GodraysThreshold => p.threshold = v,
        GodraysKnee => p.knee = v,
        GodraysIntensity => p.intensity = v,
        GodraysTintR => p.tint[0] = v,
        GodraysTintG => p.tint[1] = v,
        GodraysTintB => p.tint[2] = v,
        _ => {}
    }
}

pub fn label(ch: Channel) -> Option<&'static str> {
    use Channel::*;
    Some(match ch {
        GodraysX => "godrays source.x",
        GodraysY => "godrays source.y",
        GodraysLength => "godrays length",
        GodraysFalloff => "godrays falloff",
        GodraysThreshold => "godrays threshold",
        GodraysKnee => "godrays knee",
        GodraysIntensity => "godrays intensity",
        GodraysTintR => "godrays tint.r",
        GodraysTintG => "godrays tint.g",
        GodraysTintB => "godrays tint.b",
        _ => return None,
    })
}

/// `ungang` is the node's set of rows whose lanes are un-ganged; the tint
/// row keeps its flag there under "tint".
pub fn panel(
    ui: &mut egui::Ui,
    rcx: &mut RowCtx,
    p: &mut GodraysParams,
    working: Colorspace,
    ungang: &mut BTreeSet<String>,
    source_wired: bool,
) -> Option<HudSlider> {
    use Channel::*;
    let mut h = widgets::drags(
        ui,
        rcx,
        "source",
        {
            let [x, y] = &mut p.center;
            &mut [x, y]
        },
        0.002,
        (-0.5, 1.5),
        &[GodraysX, GodraysY],
    );
    for got in [
        widgets::slider(ui, rcx, "length", &mut p.length, 0.0, 1.0, &[GodraysLength]),
        widgets::slider(ui, rcx, "falloff", &mut p.falloff, 0.0, 8.0, &[GodraysFalloff]),
        widgets::slider(ui, rcx, "threshold", &mut p.threshold, 0.0, 8.0, &[GodraysThreshold]),
        widgets::slider(ui, rcx, "knee", &mut p.knee, 0.0, 2.0, &[GodraysKnee]),
        widgets::slider(ui, rcx, "intensity", &mut p.intensity, 0.0, 8.0, &[GodraysIntensity]),
    ]
    .into_iter()
    .flatten()
    {
        h = Some(got);
    }
    // Quality: taps per radial pass (the ray is samples³ points) and how
    // far the taps are dithered. Both uniforms: no recompile, only time.
    let mut n = p.samples as f32;
    if widgets::plain_slider(
        ui,
        "samples",
        &mut n,
        godrays::MIN_TAPS as f32,
        godrays::MAX_TAPS as f32,
        true,
    )
    .changed()
    {
        p.samples = n.round() as u32;
    }
    widgets::plain_slider(ui, "dither", &mut p.dither, 0.0, 1.0, false);
    let mut gang = !ungang.contains("tint");
    if let Some(got) = widgets::rgb_row(
        ui,
        rcx,
        "tint",
        &mut p.tint,
        &mut gang,
        RgbSpec::scale(0.0, 4.0, working.primaries.luma_weights())
            .with_display(primaries_matrix(working.primaries, Primaries::Rec709)),
        &[GodraysTintR, GodraysTintG, GodraysTintB],
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
        egui::ComboBox::from_id_salt(("godrays-mode", rcx.node))
            .selected_text(match p.mode {
                GlowMode::Only => "rays only",
                m => m.name(),
            })
            .show_ui(ui, |ui| {
                for m in GlowMode::ALL {
                    let name = if m == GlowMode::Only { "rays only" } else { m.name() };
                    ui.selectable_value(&mut p.mode, m, name);
                }
            });
    });
    if source_wired {
        ui.weak("rays cast from the highlights the source matte covers");
    } else {
        ui.weak("rays cast from every highlight — wire a matte into");
        ui.weak("the source port to choose what casts them");
    }
    ui.weak("drag the sun on the viewer to move the source");
    h
}

/// A sun on the horizon with shafts fanning down from it.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    let sun = c + egui::vec2(0.0, -r * 0.45);
    painter.circle(sun, r * 0.32, fill, stroke);
    for k in 0..5 {
        let a = std::f32::consts::FRAC_PI_2 + (k as f32 - 2.0) * 0.38;
        let d = egui::vec2(a.cos(), a.sin());
        painter.line_segment([sun + d * r * 0.5, sun + d * r * 1.45], stroke);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::graph::NodeKind;

    /// The panel must FIT the tiles it asks for: a row pushed into the
    /// tile's scroll area cannot be clicked.
    #[test]
    fn the_panel_fits_the_tiles_it_asks_for() {
        let mut p = GodraysParams::default();
        let mut ungang = BTreeSet::new();
        let mut anim = Default::default();
        let ekind = NodeKind::Godrays(p);
        let rows = crate::widgets::tile_rows(&ekind);
        let over = std::cell::Cell::new(0.0f32);
        let wide = std::cell::Cell::new(0.0f32);
        let span = |n: usize| {
            crate::widgets::TILE * n as f32 + crate::widgets::TILE_GAP * (n as f32 - 1.0)
        };
        let (w, hgt) = (span(TILES) - 14.0, span(rows) - 14.0 - 22.0);
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
                let _ = panel(ui, &mut rcx, &mut p, Colorspace::ACESCG, &mut ungang, false);
            });
            over.set(out.content_size.y - out.inner_rect.height());
            wide.set(out.content_size.x - out.inner_rect.width());
        });
        shot.save("godrays-panel");
        assert!(over.get() <= 0.5, "the panel overflows its tiles by {}pt", over.get());
        assert!(wide.get() <= 0.5, "the panel overflows SIDEWAYS by {}pt", wide.get());
    }
}
