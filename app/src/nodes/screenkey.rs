//! ScreenKey's app half: the panel, the glyph and the channel bindings.
//! The maths, the WGSL and the params are `engine::nodes::screenkey`.

use crate::anim::Channel;
use crate::rgbslider::RgbSpec;
use crate::widgets::{self, HudSlider, RowCtx};
use egui::{Color32, Painter, Pos2, Rect, Stroke, Vec2};
use engine::color::{Colorspace, Primaries};
use engine::graph::{ScreenKeyParams, ScreenOut};
use std::collections::BTreeSet;

/// Every animatable param, in panel order. The output mode is a pick,
/// not a value.
pub const CHANNELS: &[Channel] = &[
    Channel::ScreenKeyR,
    Channel::ScreenKeyG,
    Channel::ScreenKeyB,
    Channel::ScreenKeyBalance,
    Channel::ScreenKeyGain,
    Channel::ScreenKeyClipBlack,
    Channel::ScreenKeyClipWhite,
    Channel::ScreenKeyDespill,
    Channel::ScreenKeyRestore,
];

/// The screen colour is an rgb row, and an rgb row needs track to aim
/// along: two tiles.
pub const TILES: usize = 2;

pub fn get(p: &ScreenKeyParams, ch: Channel) -> Option<f32> {
    use Channel::*;
    Some(match ch {
        ScreenKeyR => p.screen[0],
        ScreenKeyG => p.screen[1],
        ScreenKeyB => p.screen[2],
        ScreenKeyBalance => p.balance,
        ScreenKeyGain => p.gain,
        ScreenKeyClipBlack => p.clip_black,
        ScreenKeyClipWhite => p.clip_white,
        ScreenKeyDespill => p.despill,
        ScreenKeyRestore => p.restore,
        _ => return None,
    })
}

pub fn set(p: &mut ScreenKeyParams, ch: Channel, v: f32) {
    use Channel::*;
    match ch {
        ScreenKeyR => p.screen[0] = v,
        ScreenKeyG => p.screen[1] = v,
        ScreenKeyB => p.screen[2] = v,
        ScreenKeyBalance => p.balance = v,
        ScreenKeyGain => p.gain = v,
        ScreenKeyClipBlack => p.clip_black = v,
        ScreenKeyClipWhite => p.clip_white = v,
        ScreenKeyDespill => p.despill = v,
        ScreenKeyRestore => p.restore = v,
        _ => {}
    }
}

/// The curve editor's name for one of this node's channels.
pub fn label(ch: Channel) -> Option<&'static str> {
    use Channel::*;
    Some(match ch {
        ScreenKeyR => "screen.r",
        ScreenKeyG => "screen.g",
        ScreenKeyB => "screen.b",
        ScreenKeyBalance => "screen balance",
        ScreenKeyGain => "screen gain",
        ScreenKeyClipBlack => "clip black",
        ScreenKeyClipWhite => "clip white",
        ScreenKeyDespill => "despill",
        ScreenKeyRestore => "despill restore",
        _ => return None,
    })
}

/// The properties panel. Hands back the row the user ctrl-dragged out as a
/// HUD slider, if one was.
pub fn panel(
    ui: &mut egui::Ui,
    rcx: &mut RowCtx,
    p: &mut ScreenKeyParams,
    working: Colorspace,
    ungang: &mut BTreeSet<String>,
) -> Option<HudSlider> {
    use Channel::*;
    let luma = working.primaries.luma_weights();
    let disp = engine::color::primaries_matrix(working.primaries, Primaries::Rec709);
    let mut h: Option<HudSlider> = None;
    // The screen colour: an rgb row, so it keys, patches and — the point —
    // takes the eyedropper (E over it, then click or drag a box on the
    // screen in the viewer).
    let mut gang = !ungang.contains("screen");
    if let Some(got) = widgets::rgb_row(
        ui,
        rcx,
        "screen",
        &mut p.screen,
        &mut gang,
        RgbSpec::scale(0.0, 4.0, luma).with_display(disp),
        &[ScreenKeyR, ScreenKeyG, ScreenKeyB],
    ) {
        h = Some(got);
    }
    if gang {
        ungang.remove("screen");
    } else {
        ungang.insert("screen".to_string());
    }
    let which = match p.dominant() {
        0 => "red",
        1 => "green",
        _ => "blue",
    };
    ui.weak(format!("hold E over the row and click the screen in the viewer · keying {which}"));
    ui.separator();
    for got in [
        widgets::slider(ui, rcx, "balance", &mut p.balance, 0.0, 1.0, &[ScreenKeyBalance]),
        widgets::slider(ui, rcx, "gain", &mut p.gain, 0.0, 4.0, &[ScreenKeyGain]),
        widgets::slider(ui, rcx, "clip black", &mut p.clip_black, 0.0, 1.0, &[ScreenKeyClipBlack]),
        widgets::slider(ui, rcx, "clip white", &mut p.clip_white, 0.0, 1.0, &[ScreenKeyClipWhite]),
    ]
    .into_iter()
    .flatten()
    {
        h = Some(got);
    }
    ui.separator();
    for got in [
        widgets::slider(ui, rcx, "despill", &mut p.despill, 0.0, 1.0, &[ScreenKeyDespill]),
        widgets::slider(ui, rcx, "restore", &mut p.restore, 0.0, 1.0, &[ScreenKeyRestore]),
    ]
    .into_iter()
    .flatten()
    {
        h = Some(got);
    }
    ui.horizontal(|ui| {
        ui.label("output");
        egui::ComboBox::from_id_salt(("screenkey-out", rcx.node))
            .width(190.0)
            .selected_text(p.output.label())
            .show_ui(ui, |ui| {
                for m in ScreenOut::ALL {
                    ui.selectable_value(&mut p.output, m, m.label());
                }
            });
        if ui.small_button("reset").on_hover_text("a fresh key, screen colour kept").clicked() {
            *p = ScreenKeyParams { screen: p.screen, ..Default::default() };
        }
    });
    ui.weak("balance: toward 1 only a pixel greener than BOTH other");
    ui.weak("channels is screen. gain calls more of the edge screen;");
    ui.weak("the clips solidify the core and clean the screen.");
    ui.weak("holdout keeps, garbage drops; restore puts the light");
    ui.weak("despill took back, so skin keeps its brightness.");
    h
}

/// The graph glyph: a screen with a figure cut out of it.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.circle(c, r * 0.9, fill, stroke);
    let ink = crate::theme::node_ink();
    let screen = Rect::from_center_size(c, Vec2::new(r * 1.1, r * 0.9));
    painter.rect_filled(screen, r * 0.12, ink);
    // Head, then shoulders rising from the screen's bottom edge.
    painter.circle_filled(c + Vec2::new(0.0, -r * 0.12), r * 0.17, fill);
    let base = Pos2::new(c.x, screen.bottom());
    let pts: Vec<Pos2> = (0..=10)
        .map(|k| {
            let a = std::f32::consts::PI * (1.0 + k as f32 / 10.0);
            base + Vec2::new(a.cos(), a.sin()) * r * 0.36
        })
        .collect();
    painter.add(egui::Shape::convex_polygon(pts, fill, Stroke::NONE));
}
