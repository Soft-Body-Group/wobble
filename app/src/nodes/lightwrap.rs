//! Lightwrap's app half: width, background blur, intensity, the combine
//! mode, and the glyph. The model, the WGSL and the params are
//! `engine::nodes::lightwrap`.

use crate::anim::Channel;
use crate::widgets::{self, HudSlider, RowCtx};
use egui::{Color32, Painter, Pos2, Stroke, Vec2};
use engine::graph::{LightwrapParams, WrapMode};

pub const CHANNELS: &[Channel] =
    &[Channel::LightwrapWidth, Channel::LightwrapBgBlur, Channel::LightwrapIntensity];

/// Three sliders and a picker.
pub const TILES: usize = 2;

pub fn get(p: &LightwrapParams, ch: Channel) -> Option<f32> {
    Some(match ch {
        Channel::LightwrapWidth => p.width,
        Channel::LightwrapBgBlur => p.bg_blur,
        Channel::LightwrapIntensity => p.intensity,
        _ => return None,
    })
}

pub fn set(p: &mut LightwrapParams, ch: Channel, v: f32) {
    match ch {
        Channel::LightwrapWidth => p.width = v,
        Channel::LightwrapBgBlur => p.bg_blur = v,
        Channel::LightwrapIntensity => p.intensity = v,
        _ => {}
    }
}

/// The curve editor's name for a channel.
pub fn label(ch: Channel) -> Option<&'static str> {
    Some(match ch {
        Channel::LightwrapWidth => "lightwrap width",
        Channel::LightwrapBgBlur => "lightwrap bg blur",
        Channel::LightwrapIntensity => "lightwrap intensity",
        _ => return None,
    })
}

/// `bg_wired`: whether the background port has a wire — without one there
/// is nothing to wrap, and the panel says so rather than looking broken.
pub fn panel(
    ui: &mut egui::Ui,
    rcx: &mut RowCtx,
    p: &mut LightwrapParams,
    bg_wired: bool,
) -> Option<HudSlider> {
    use Channel::*;
    let mut h = None;
    for got in [
        widgets::slider(ui, rcx, "width", &mut p.width, 0.0, 100.0, &[LightwrapWidth]),
        widgets::slider(ui, rcx, "bg blur", &mut p.bg_blur, 0.0, 200.0, &[LightwrapBgBlur]),
        widgets::slider(ui, rcx, "intensity", &mut p.intensity, 0.0, 4.0, &[LightwrapIntensity]),
    ]
    .into_iter()
    .flatten()
    {
        h = Some(got);
    }
    ui.horizontal(|ui| {
        ui.add_sized([widgets::LABEL_W, 16.0], egui::Label::new("combine"));
        egui::ComboBox::from_id_salt(("lightwrap-mode", rcx.node))
            .selected_text(p.mode.name())
            .show_ui(ui, |ui| {
                for m in WrapMode::ALL {
                    ui.selectable_value(&mut p.mode, m, m.name());
                }
            });
    });
    ui.add_space(4.0);
    if bg_wired {
        ui.weak("the background, blurred, in a band inside the");
        ui.weak("foreground's edge. Merge the result over the bg.");
        ui.weak("Slow path: a separable blur, three passes.");
    } else {
        ui.colored_label(crate::theme::warn(), "no background wired: nothing to wrap");
    }
    h
}

/// The graph glyph: a disc with light hugging part of its rim.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.circle(c, r * 0.9, fill, stroke);
    painter.circle_filled(c, r * 0.42, crate::theme::node_ink());
    let ink = Stroke::new((r * 0.14).max(1.0), crate::theme::node_ink());
    let n = 16;
    let (a0, a1) = (-0.95 * std::f32::consts::PI, -0.05 * std::f32::consts::PI);
    let pts: Vec<Pos2> = (0..=n)
        .map(|k| {
            let a = a0 + (a1 - a0) * k as f32 / n as f32;
            c + Vec2::new(a.cos(), a.sin()) * r * 0.62
        })
        .collect();
    painter.add(egui::Shape::line(pts, ink));
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::graph::NodeKind;

    #[test]
    fn the_channels_round_trip() {
        let mut p = LightwrapParams::default();
        for (i, ch) in CHANNELS.iter().enumerate() {
            set(&mut p, *ch, 0.5 + i as f32);
            assert_eq!(get(&p, *ch), Some(0.5 + i as f32));
            assert!(label(*ch).is_some());
        }
    }

    /// The panel fits its tiles, both ways, wired or not.
    #[test]
    fn the_panel_fits_its_tiles() {
        for bg in [true, false] {
            let mut p = LightwrapParams::default();
            let mut anim = Default::default();
            let ekind = NodeKind::Lightwrap(p);
            let (over, wide) = (std::cell::Cell::new(0.0f32), std::cell::Cell::new(0.0f32));
            let t = crate::widgets::TILE;
            let shot = crate::snapshot::shoot(t * TILES as f32 + crate::widgets::TILE_GAP * (TILES as f32 - 1.0) - 14.0, t - 14.0 - 22.0, |ui| {
                let out = egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                    let mut rcx = RowCtx {
                        node: 1,
                        anim: &mut anim,
                        ekind: &ekind,
                        t: 0,
                        hud_live: false,
                        driven: Vec::new(),
                    };
                    let _ = panel(ui, &mut rcx, &mut p, bg);
                });
                over.set(out.content_size.y - out.inner_rect.height());
                wide.set(out.content_size.x - out.inner_rect.width());
            });
            shot.save("lightwrap-panel");
            assert!(over.get() <= 0.5, "overflows its tiles DOWN by {}pt (bg {bg})", over.get());
            assert!(wide.get() <= 0.5, "overflows its tiles SIDEWAYS by {}pt (bg {bg})", wide.get());
        }
    }
}
