//! Saturation's app half: the panel, the glyph and the channel bindings.
//! The maths, the WGSL and the params are `engine::nodes::saturation`.

use crate::anim::Channel;
use crate::widgets::{self, HudSlider, RowCtx};
use egui::{Color32, Painter, Pos2, Stroke, Vec2};
use engine::graph::{SatDomain, SatLuma, SatParams};

/// Every animatable param, in panel order. The domain and the luma basis
/// are not channels: they are shader shape, not values.
pub const CHANNELS: &[Channel] = &[
    Channel::SatAmount,
    Channel::SatBoost,
    Channel::SatShadow,
    Channel::SatMid,
    Channel::SatHigh,
    Channel::SatPivot,
    Channel::SatWidth,
    Channel::SatPreserve,
    Channel::SatCompress,
];

/// Panel width in tiles: the three-way zone row needs the room.
pub const TILES: usize = 2;

pub fn get(p: &SatParams, ch: Channel) -> Option<f32> {
    use Channel::*;
    Some(match ch {
        SatAmount => p.sat,
        SatBoost => p.boost,
        SatShadow => p.zones[0],
        SatMid => p.zones[1],
        SatHigh => p.zones[2],
        SatPivot => p.pivot,
        SatWidth => p.width,
        SatPreserve => p.preserve,
        SatCompress => p.compress,
        _ => return None,
    })
}

pub fn set(p: &mut SatParams, ch: Channel, v: f32) {
    use Channel::*;
    match ch {
        SatAmount => p.sat = v,
        SatBoost => p.boost = v,
        SatShadow => p.zones[0] = v,
        SatMid => p.zones[1] = v,
        SatHigh => p.zones[2] = v,
        SatPivot => p.pivot = v,
        SatWidth => p.width = v,
        SatPreserve => p.preserve = v,
        SatCompress => p.compress = v,
        _ => {}
    }
}

/// The curve editor's name for one of this node's channels.
pub fn label(ch: Channel) -> Option<&'static str> {
    use Channel::*;
    Some(match ch {
        SatAmount => "saturation",
        SatBoost => "colour boost",
        SatShadow => "sat shadows",
        SatMid => "sat midtones",
        SatHigh => "sat highlights",
        SatPivot => "sat pivot",
        SatWidth => "sat zone width",
        SatPreserve => "keep luminance",
        SatCompress => "gamut compress",
        _ => return None,
    })
}

/// The properties panel. Hands back the row the user ctrl-dragged out as a
/// HUD slider, if one was.
pub fn panel(ui: &mut egui::Ui, rcx: &mut RowCtx, p: &mut SatParams) -> Option<HudSlider> {
    use Channel::*;
    let mut h = None;
    h = h.or(widgets::slider(
        ui,
        rcx,
        "saturation",
        &mut p.sat,
        0.0,
        3.0,
        &[SatAmount],
    ));
    h = h.or(widgets::slider(
        ui,
        rcx,
        "colour boost",
        &mut p.boost,
        -1.0,
        2.0,
        &[SatBoost],
    ));
    // Where the arithmetic happens. Both of these are shader shape, so they
    // recompile — which is why they are combos and not sliders.
    ui.horizontal(|ui| {
        ui.label("domain");
        egui::ComboBox::from_id_salt(("sat-domain", rcx.node))
            .width(148.0)
            .selected_text(p.domain.label())
            .show_ui(ui, |ui| {
                for d in SatDomain::ALL {
                    ui.selectable_value(&mut p.domain, d, d.label());
                }
            });
    });
    ui.horizontal(|ui| {
        ui.label("luma");
        let e = p.domain != SatDomain::Perceptual;
        ui.add_enabled_ui(e, |ui| {
            egui::ComboBox::from_id_salt(("sat-luma", rcx.node))
                .width(148.0)
                .selected_text(p.luma.label())
                .show_ui(ui, |ui| {
                    for l in SatLuma::ALL {
                        ui.selectable_value(&mut p.luma, l, l.label());
                    }
                });
        })
        .response
        .on_hover_text(if e {
            "which luminance the lerp holds still"
        } else {
            "Oklab holds its own lightness — no weights to pick"
        });
    });
    ui.separator();
    // Sat vs Lum, three zones instead of a curve.
    h = h.or(widgets::drags(
        ui,
        rcx,
        "shad · mid · high",
        {
            let [a, b, c] = &mut p.zones;
            &mut [a, b, c]
        },
        0.01,
        (0.0, 3.0),
        &[SatShadow, SatMid, SatHigh],
    ));
    h = h.or(widgets::drags(
        ui,
        rcx,
        "pivot · stops",
        {
            let SatParams { pivot, width, .. } = p;
            &mut [pivot, width]
        },
        0.01,
        (0.001, 8.0),
        &[SatPivot, SatWidth],
    ));
    ui.separator();
    h = h.or(widgets::slider(
        ui,
        rcx,
        "keep luma",
        &mut p.preserve,
        0.0,
        1.0,
        &[SatPreserve],
    ));
    h = h.or(widgets::slider(
        ui,
        rcx,
        "gamut",
        &mut p.compress,
        0.0,
        1.0,
        &[SatCompress],
    ));
    ui.weak(match p.domain {
        SatDomain::Linear => {
            "scene-linear: honest, and hard on highlights \
             — they sit furthest from neutral"
        }
        SatDomain::Log => {
            "log, the way a Resolve timeline grades: \
             highlights and midtones gain alike"
        }
        SatDomain::Perceptual => {
            "Oklab chroma: lightness and hue hold, and \
             no other tool will match the numbers"
        }
    });
    ui.weak(
        "boost favours the colours that have least — \
         vibrance, under Resolve's name for it",
    );
    h
}

/// The graph glyph: a disc with the colour drained out of one half — the
/// before/after of the only control that matters.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.circle(c, r * 0.9, fill, stroke);
    let grey =
        Color32::from_gray(((fill.r() as u32 + fill.g() as u32 + fill.b() as u32) / 3) as u8);
    let mut half = vec![Pos2::new(c.x, c.y - r * 0.9)];
    for k in 0..=8 {
        let a = std::f32::consts::PI * (k as f32 / 8.0 - 0.5);
        half.push(c + Vec2::new(a.cos(), a.sin()) * r * 0.9);
    }
    painter.add(egui::Shape::convex_polygon(half, grey, Stroke::NONE));
}
