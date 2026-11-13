//! HueRotate's app half: the panel, the glyph and the channel binding. The
//! model, the WGSL and the params are `engine::nodes::huerotate`.

use crate::anim::Channel;
use crate::widgets::{self, HudSlider, RowCtx};
use egui::{Color32, Painter, Pos2, Stroke, Vec2};
use engine::color::{Colorspace, Curve};
use engine::graph::HueRotateParams;

pub const CHANNELS: &[Channel] = &[Channel::HueRotateAngle];

/// One slider and its picture: a single tile.
pub const TILES: usize = 1;

pub fn get(p: &HueRotateParams, ch: Channel) -> Option<f32> {
    (ch == Channel::HueRotateAngle).then_some(p.angle)
}

pub fn set(p: &mut HueRotateParams, ch: Channel, v: f32) {
    if ch == Channel::HueRotateAngle {
        p.angle = v;
    }
}

/// The curve editor's name for the channel.
pub fn label(ch: Channel) -> Option<&'static str> {
    (ch == Channel::HueRotateAngle).then_some("hue rotate")
}

/// The hue circle at `deg`, as a display colour: fully colourful at a
/// middling level, so the strips show hue and nothing else.
fn wheel(deg: f32) -> [f32; 3] {
    let h = deg.rem_euclid(360.0) / 60.0;
    let x = 1.0 - (h % 2.0 - 1.0).abs();
    let (r, g, b) = match h as i32 {
        0 => (1.0, x, 0.0),
        1 => (x, 1.0, 0.0),
        2 => (0.0, 1.0, x),
        3 => (0.0, x, 1.0),
        4 => (x, 0.0, 1.0),
        _ => (1.0, 0.0, x),
    };
    [r * 0.45, g * 0.45, b * 0.45]
}

fn shown(c: [f32; 3]) -> Color32 {
    let e = |v: f32| (Curve::Srgb.from_linear(v.clamp(0.0, 1.0)) * 255.0).round() as u8;
    Color32::from_rgb(e(c[0]), e(c[1]), e(c[2]))
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

/// The panel: the angle, then the hue circle before and after it — each
/// colour on the top strip lands as the colour under it on the bottom one.
pub fn panel(
    ui: &mut egui::Ui,
    rcx: &mut RowCtx,
    p: &mut HueRotateParams,
    working: Colorspace,
) -> Option<HudSlider> {
    let h = widgets::slider(ui, rcx, "angle", &mut p.angle, -180.0, 180.0, &[Channel::HueRotateAngle]);
    ui.add_space(6.0);
    let w = ui.available_width();
    for (name, turned) in [("in", false), ("out", true)] {
        ui.horizontal(|ui| {
            ui.add_sized(
                [widgets::LABEL_W, 12.0],
                egui::Label::new(egui::RichText::new(name).small().weak()),
            );
            let (r, _) = ui.allocate_exact_size(
                egui::vec2((w - widgets::LABEL_W - 8.0).max(40.0), 10.0),
                egui::Sense::hover(),
            );
            // The model itself paints the bottom strip, in the comp's own
            // working space: the picture is what the node does.
            let q = *p;
            strip(ui, r, |t| {
                let c = wheel(t * 360.0);
                shown(if turned { q.apply(c, working) } else { c })
            });
        });
    }
    ui.add_space(4.0);
    ui.weak("every colour turns about grey; brightness holds");
    h
}

/// The graph glyph: a ring with an arrowhead — a turn.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.circle(c, r * 0.9, fill, stroke);
    let ink = Stroke::new((r * 0.14).max(1.0), crate::theme::node_ink());
    let rr = r * 0.52;
    let (a0, a1) = (-0.35 * std::f32::consts::PI, 1.25 * std::f32::consts::PI);
    let n = 20;
    let pts: Vec<Pos2> = (0..=n)
        .map(|k| {
            let a = a0 + (a1 - a0) * k as f32 / n as f32;
            c + Vec2::new(a.cos(), a.sin()) * rr
        })
        .collect();
    let tip = *pts.last().unwrap();
    painter.add(egui::Shape::line(pts, ink));
    // The head points along the turn at its end.
    let t = Vec2::new(-a1.sin(), a1.cos());
    let side = Vec2::new(t.y, -t.x);
    let s = r * 0.26;
    painter.add(egui::Shape::convex_polygon(
        vec![tip + t * s, tip - t * s * 0.2 + side * s * 0.8, tip - t * s * 0.2 - side * s * 0.8],
        ink.color,
        Stroke::NONE,
    ));
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::graph::NodeKind;

    #[test]
    fn the_channel_round_trips() {
        let mut p = HueRotateParams::default();
        set(&mut p, Channel::HueRotateAngle, 42.0);
        assert_eq!(get(&p, Channel::HueRotateAngle), Some(42.0));
        assert_eq!(p.angle, 42.0);
        assert!(label(Channel::HueRotateAngle).is_some());
    }

    /// The panel fits its one tile, both ways.
    #[test]
    fn the_panel_fits_its_tile() {
        let mut p = HueRotateParams { angle: 60.0 };
        let mut anim = Default::default();
        let ekind = NodeKind::HueRotate(p);
        let (over, wide) = (std::cell::Cell::new(0.0f32), std::cell::Cell::new(0.0f32));
        let t = crate::widgets::TILE;
        let shot = crate::snapshot::shoot(t - 14.0, t - 14.0 - 22.0, |ui| {
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
        shot.save("huerotate-panel");
        assert!(over.get() <= 0.5, "overflows its tile DOWN by {}pt", over.get());
        assert!(wide.get() <= 0.5, "overflows its tile SIDEWAYS by {}pt", wide.get());
    }
}
