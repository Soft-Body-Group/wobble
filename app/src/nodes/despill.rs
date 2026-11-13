//! Despill's app half: the screen and the method, three sliders, a picture
//! of what it does, and the glyph. The model, the WGSL and the params are
//! `engine::nodes::despill`.

use crate::anim::Channel;
use crate::widgets::{self, HudSlider, RowCtx};
use egui::{Color32, Painter, Pos2, Stroke, Vec2};
use engine::color::{Colorspace, Curve};
use engine::graph::{DespillMethod, DespillParams, DespillScreen};

pub const CHANNELS: &[Channel] =
    &[Channel::DespillStrength, Channel::DespillBias, Channel::DespillRestore];

/// Two pickers, three sliders and the strips.
pub const TILES: usize = 2;

pub fn get(p: &DespillParams, ch: Channel) -> Option<f32> {
    Some(match ch {
        Channel::DespillStrength => p.strength,
        Channel::DespillBias => p.bias,
        Channel::DespillRestore => p.restore,
        _ => return None,
    })
}

pub fn set(p: &mut DespillParams, ch: Channel, v: f32) {
    match ch {
        Channel::DespillStrength => p.strength = v,
        Channel::DespillBias => p.bias = v,
        Channel::DespillRestore => p.restore = v,
        _ => {}
    }
}

/// The curve editor's name for a channel.
pub fn label(ch: Channel) -> Option<&'static str> {
    Some(match ch {
        Channel::DespillStrength => "despill strength",
        Channel::DespillBias => "despill bias",
        Channel::DespillRestore => "despill restore",
        _ => return None,
    })
}

fn shown(c: [f32; 3]) -> Color32 {
    let e = |v: f32| (Curve::Srgb.from_linear(v.clamp(0.0, 1.0)) * 255.0).round() as u8;
    Color32::from_rgb(e(c[0]), e(c[1]), e(c[2]))
}

/// A ramp from a neutral skin tone into the screen colour: the spill a
/// keyed edge carries, from none to all of it.
fn spill_ramp(screen: DespillScreen, t: f32) -> [f32; 3] {
    let skin = [0.45, 0.3, 0.22];
    let scr = match screen {
        DespillScreen::Green => [0.08, 0.5, 0.1],
        DespillScreen::Blue => [0.06, 0.12, 0.55],
    };
    std::array::from_fn(|i| skin[i] + (scr[i] - skin[i]) * t)
}

/// A horizontal strip whose colour at each point is `f` of the point's
/// place along it, 0..1.
fn strip(ui: &egui::Ui, r: egui::Rect, f: impl Fn(f32) -> Color32) {
    let n = 48;
    let mut mesh = egui::Mesh::default();
    for k in 0..=n {
        let t = k as f32 / n as f32;
        let x = r.left() + r.width() * t;
        let c = f(t);
        mesh.colored_vertex(Pos2::new(x, r.top()), c);
        mesh.colored_vertex(Pos2::new(x, r.bottom()), c);
        if k > 0 {
            let i = 2 * k as u32;
            mesh.add_triangle(i - 2, i - 1, i);
            mesh.add_triangle(i - 1, i + 1, i);
        }
    }
    ui.painter().add(egui::Shape::mesh(mesh));
}

pub fn panel(
    ui: &mut egui::Ui,
    rcx: &mut RowCtx,
    p: &mut DespillParams,
    working: Colorspace,
) -> Option<HudSlider> {
    use Channel::*;
    ui.horizontal(|ui| {
        ui.add_sized([widgets::LABEL_W, 16.0], egui::Label::new("screen"));
        egui::ComboBox::from_id_salt(("despill-screen", rcx.node))
            .selected_text(p.screen.name())
            .show_ui(ui, |ui| {
                for s in DespillScreen::ALL {
                    ui.selectable_value(&mut p.screen, s, s.name());
                }
            });
        ui.add_space(8.0);
        ui.label("limit");
        egui::ComboBox::from_id_salt(("despill-method", rcx.node))
            .selected_text(p.method.name())
            .show_ui(ui, |ui| {
                for m in DespillMethod::ALL {
                    ui.selectable_value(&mut p.method, m, m.name());
                }
            });
    });
    let mut h = None;
    for got in [
        widgets::slider(ui, rcx, "strength", &mut p.strength, 0.0, 1.0, &[DespillStrength]),
        widgets::slider(ui, rcx, "bias", &mut p.bias, 0.0, 2.0, &[DespillBias]),
        widgets::slider(ui, rcx, "restore", &mut p.restore, 0.0, 1.0, &[DespillRestore]),
    ]
    .into_iter()
    .flatten()
    {
        h = Some(got);
    }
    ui.add_space(6.0);
    let w = ui.available_width();
    for (name, despilled) in [("in", false), ("out", true)] {
        ui.horizontal(|ui| {
            ui.add_sized(
                [widgets::LABEL_W, 12.0],
                egui::Label::new(egui::RichText::new(name).small().weak()),
            );
            let (r, _) = ui.allocate_exact_size(
                egui::vec2((w - widgets::LABEL_W - 8.0).max(40.0), 10.0),
                egui::Sense::hover(),
            );
            // The model paints the bottom strip: the picture is the node.
            let q = *p;
            strip(ui, r, |t| {
                let c = spill_ramp(q.screen, t);
                shown(if despilled { q.apply(c, working) } else { c })
            });
        });
    }
    ui.add_space(4.0);
    ui.weak("the screen channel limited to the other two");
    h
}

/// The graph glyph: a drop, crossed out — spill taken away.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.circle(c, r * 0.9, fill, stroke);
    let ink = Stroke::new((r * 0.14).max(1.0), crate::theme::node_ink());
    // The drop: a point on top, a round belly below.
    let n = 14;
    let mut pts = vec![c + Vec2::new(0.0, -r * 0.6)];
    for k in 0..=n {
        let a = std::f32::consts::PI * (-0.15 + 1.3 * k as f32 / n as f32);
        pts.push(c + Vec2::new(a.cos(), a.sin()) * r * 0.32 + Vec2::new(0.0, r * 0.15));
    }
    pts.push(c + Vec2::new(0.0, -r * 0.6));
    painter.add(egui::Shape::line(pts, ink));
    painter.line_segment([c + Vec2::new(-r * 0.5, r * 0.5), c + Vec2::new(r * 0.5, -r * 0.5)], ink);
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::graph::NodeKind;

    #[test]
    fn the_channels_round_trip() {
        let mut p = DespillParams::default();
        for (i, ch) in CHANNELS.iter().enumerate() {
            set(&mut p, *ch, 0.25 + i as f32);
            assert_eq!(get(&p, *ch), Some(0.25 + i as f32));
            assert!(label(*ch).is_some());
        }
    }

    /// The panel fits its tiles, both ways.
    #[test]
    fn the_panel_fits_its_tiles() {
        let mut p = DespillParams::default();
        let mut anim = Default::default();
        let ekind = NodeKind::Despill(p);
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
                let _ = panel(ui, &mut rcx, &mut p, Colorspace::ACESCG);
            });
            over.set(out.content_size.y - out.inner_rect.height());
            wide.set(out.content_size.x - out.inner_rect.width());
        });
        shot.save("despill-panel");
        assert!(over.get() <= 0.5, "overflows its tiles DOWN by {}pt", over.get());
        assert!(wide.get() <= 0.5, "overflows its tiles SIDEWAYS by {}pt", wide.get());
    }
}
