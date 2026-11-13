//! Diffuse's app half: the scatter, the veil, the mist colour as an rgb
//! row, the depth ramp, and a sun behind bands of mist for a glyph.

use crate::anim::Channel;
use crate::rgbslider::RgbSpec;
use crate::widgets::{self, HudSlider, RowCtx};
use egui::{Color32, Painter, Pos2, Stroke};
use engine::color::{primaries_matrix, Colorspace, Primaries};
use engine::graph::DiffuseParams;
use std::collections::BTreeSet;

pub const CHANNELS: &[Channel] = &[
    Channel::DiffuseSize,
    Channel::DiffuseFalloff,
    Channel::DiffuseAmount,
    Channel::DiffuseHaze,
    Channel::DiffuseMistR,
    Channel::DiffuseMistG,
    Channel::DiffuseMistB,
    Channel::DiffuseNear,
    Channel::DiffuseFar,
];

/// A row of rgb lanes needs track to aim along.
pub const TILES: usize = 2;

pub fn get(p: &DiffuseParams, ch: Channel) -> Option<f32> {
    use Channel::*;
    Some(match ch {
        DiffuseSize => p.size,
        DiffuseFalloff => p.falloff,
        DiffuseAmount => p.amount,
        DiffuseHaze => p.haze,
        DiffuseMistR => p.mist[0],
        DiffuseMistG => p.mist[1],
        DiffuseMistB => p.mist[2],
        DiffuseNear => p.near,
        DiffuseFar => p.far,
        _ => return None,
    })
}

pub fn set(p: &mut DiffuseParams, ch: Channel, v: f32) {
    use Channel::*;
    match ch {
        DiffuseSize => p.size = v,
        DiffuseFalloff => p.falloff = v,
        DiffuseAmount => p.amount = v,
        DiffuseHaze => p.haze = v,
        DiffuseMistR => p.mist[0] = v,
        DiffuseMistG => p.mist[1] = v,
        DiffuseMistB => p.mist[2] = v,
        DiffuseNear => p.near = v,
        DiffuseFar => p.far = v,
        _ => {}
    }
}

pub fn label(ch: Channel) -> Option<&'static str> {
    use Channel::*;
    Some(match ch {
        DiffuseSize => "diffuse size",
        DiffuseFalloff => "diffuse falloff",
        DiffuseAmount => "diffuse amount",
        DiffuseHaze => "diffuse haze",
        DiffuseMistR => "diffuse mist.r",
        DiffuseMistG => "diffuse mist.g",
        DiffuseMistB => "diffuse mist.b",
        DiffuseNear => "diffuse near",
        DiffuseFar => "diffuse far",
        _ => return None,
    })
}

/// `ungang` is the node's set of rows whose lanes are un-ganged; the mist
/// row keeps its flag there under "mist". `depth` is whether the depth
/// port is wired: the ramp rows only mean something then.
pub fn panel(
    ui: &mut egui::Ui,
    rcx: &mut RowCtx,
    p: &mut DiffuseParams,
    working: Colorspace,
    ungang: &mut BTreeSet<String>,
    depth: bool,
) -> Option<HudSlider> {
    use Channel::*;
    let mut h = None;
    for got in [
        widgets::slider(ui, rcx, "size", &mut p.size, 1.0, 512.0, &[DiffuseSize]),
        widgets::slider(ui, rcx, "falloff", &mut p.falloff, 0.0, 1.0, &[DiffuseFalloff]),
        widgets::slider(ui, rcx, "amount", &mut p.amount, 0.0, 1.0, &[DiffuseAmount]),
        widgets::slider(ui, rcx, "haze", &mut p.haze, 0.0, 1.0, &[DiffuseHaze]),
    ]
    .into_iter()
    .flatten()
    {
        h = Some(got);
    }
    let mut gang = !ungang.contains("mist");
    if let Some(got) = widgets::rgb_row(
        ui,
        rcx,
        "mist",
        &mut p.mist,
        &mut gang,
        RgbSpec::scale(0.0, 4.0, working.primaries.luma_weights())
            .with_display(primaries_matrix(working.primaries, Primaries::Rec709)),
        &[DiffuseMistR, DiffuseMistG, DiffuseMistB],
    ) {
        h = Some(got);
    }
    if gang {
        ungang.remove("mist");
    } else {
        ungang.insert("mist".to_string());
    }
    ui.add_enabled_ui(depth, |ui| {
        for got in [
            widgets::slider(ui, rcx, "near", &mut p.near, 0.0, 1.0, &[DiffuseNear]),
            widgets::slider(ui, rcx, "far", &mut p.far, 0.0, 1.0, &[DiffuseFar]),
        ]
        .into_iter()
        .flatten()
        {
            h = Some(got);
        }
        ui.checkbox(&mut p.far_low, "far is low (depth falls with distance)");
    });
    if depth {
        ui.weak("depth ramps it near to far: the near end stays sharp");
        ui.weak("and its light never haloes over the mist behind it.");
    } else {
        ui.weak("wire a depth pass (Z) to ramp it with distance.");
    }
    ui.weak("light scattered, not added: one pyramid, size is free.");
    h
}

/// A sun behind two bands of mist.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.circle(c + egui::vec2(0.0, -r * 0.2), r * 0.5, fill, stroke);
    for (dy, w) in [(0.25, 1.1), (0.65, 0.8)] {
        let y = c.y + r * dy;
        painter.line_segment([Pos2::new(c.x - r * w, y), Pos2::new(c.x + r * w, y)], stroke);
    }
}
