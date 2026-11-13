//! Lens Distortion's app half: the panel (two pages, lens and frame), the
//! channel bindings, the glyph and the viewer overlay. The model, its
//! inverse and the WGSL are `engine::nodes::lensdistort`.
//!
//! The overlay is a judging tool, not a manipulator: a grid that is
//! STRAIGHT in the undistorted picture, drawn where it lands in this node's
//! output (straight when undistorting, bent when distorting), plus the
//! source frame's edge mapped the same way (where the black corners begin).
//! Lines in the plate that should be straight should follow the grid. The
//! one handle is the principal point.

use crate::anim::Channel;
use crate::ink::InkShapes;
use crate::widgets::{self, HudSlider, RowCtx};
use egui::{Color32, Painter, Pos2, Stroke, Vec2};
use engine::graph::{LensDistortParams, LensMode, LensOutput};

pub const CHANNELS: &[Channel] = &[
    Channel::LensK1,
    Channel::LensK2,
    Channel::LensK3,
    Channel::LensP1,
    Channel::LensP2,
    Channel::LensCenterX,
    Channel::LensCenterY,
    Channel::LensSqueeze,
    Channel::LensZoom,
    Channel::LensCaR,
    Channel::LensCaB,
];

pub const TILES: usize = 2;

pub fn get(p: &LensDistortParams, ch: Channel) -> Option<f32> {
    use Channel::*;
    Some(match ch {
        LensK1 => p.k1,
        LensK2 => p.k2,
        LensK3 => p.k3,
        LensP1 => p.p1,
        LensP2 => p.p2,
        LensCenterX => p.center[0],
        LensCenterY => p.center[1],
        LensSqueeze => p.squeeze,
        LensZoom => p.zoom,
        LensCaR => p.ca[0],
        LensCaB => p.ca[1],
        _ => return None,
    })
}

pub fn set(p: &mut LensDistortParams, ch: Channel, v: f32) {
    use Channel::*;
    match ch {
        LensK1 => p.k1 = v,
        LensK2 => p.k2 = v,
        LensK3 => p.k3 = v,
        LensP1 => p.p1 = v,
        LensP2 => p.p2 = v,
        LensCenterX => p.center[0] = v,
        LensCenterY => p.center[1] = v,
        LensSqueeze => p.squeeze = v,
        LensZoom => p.zoom = v,
        LensCaR => p.ca[0] = v,
        LensCaB => p.ca[1] = v,
        _ => {}
    }
}

pub fn label(ch: Channel) -> Option<&'static str> {
    use Channel::*;
    Some(match ch {
        LensK1 => "lens k1",
        LensK2 => "lens k2",
        LensK3 => "lens k3",
        LensP1 => "lens p1",
        LensP2 => "lens p2",
        LensCenterX => "lens centre x",
        LensCenterY => "lens centre y",
        LensSqueeze => "lens squeeze",
        LensZoom => "lens zoom",
        LensCaR => "lens ca red",
        LensCaB => "lens ca blue",
        _ => return None,
    })
}

/// The pages: the model, then what happens to the frame.
pub fn tabs() -> Vec<widgets::PanelTab> {
    vec![widgets::PanelTab::new("lens", "lens"), widgets::PanelTab::new("frame", "frame")]
}

pub fn panel(ui: &mut egui::Ui, rcx: &mut RowCtx, p: &mut LensDistortParams, tab: &str) -> Option<HudSlider> {
    use Channel::*;
    let mut h = None;
    if tab == "frame" {
        h = h.or(widgets::slider(ui, rcx, "squeeze", &mut p.squeeze, 0.5, 2.5, &[LensSqueeze]));
        h = h.or(widgets::slider(ui, rcx, "zoom", &mut p.zoom, 0.5, 2.0, &[LensZoom]));
        widgets::plain_slider(ui, "overscan", &mut p.overscan, 1.0, 2.0, false)
            .on_hover_text("grows the format undistorting, shrinks it distorting: the same value on both ends of a round trip");
        ui.checkbox(&mut p.chromatic, "chromatic aberration")
            .on_hover_text("red and blue at their own magnification (three lookups)");
        let ca = ui
            .add_enabled_ui(p.chromatic, |ui| {
                let [r, b] = &mut p.ca;
                widgets::drags(ui, rcx, "ca red, blue", &mut [r, b], 0.0002, (-0.05, 0.05), &[LensCaR, LensCaB])
            })
            .inner;
        h = h.or(ca);
        ui.weak("ca: magnification minus one (0.002 ≈ 2 px at 2K)");
        return h;
    }
    ui.horizontal(|ui| {
        egui::ComboBox::from_id_salt(("lens-mode", rcx.node))
            .width(90.0)
            .selected_text(p.mode.name())
            .show_ui(ui, |ui| {
                for m in LensMode::ALL {
                    ui.selectable_value(&mut p.mode, m, m.name());
                }
            });
        egui::ComboBox::from_id_salt(("lens-out", rcx.node))
            .width(70.0)
            .selected_text(p.output.name())
            .show_ui(ui, |ui| {
                for o in LensOutput::ALL {
                    ui.selectable_value(&mut p.output, o, o.name());
                }
            })
            .response
            .on_hover_text("stmap: the lookup as a UV map (UV Warp reads it)");
    });
    h = h.or(widgets::slider(ui, rcx, "k1", &mut p.k1, -0.5, 0.5, &[LensK1]));
    h = h.or(widgets::slider(ui, rcx, "k2", &mut p.k2, -0.2, 0.2, &[LensK2]));
    h = h.or(widgets::slider(ui, rcx, "k3", &mut p.k3, -0.1, 0.1, &[LensK3]));
    h = h.or(widgets::slider(ui, rcx, "p1", &mut p.p1, -0.05, 0.05, &[LensP1]));
    h = h.or(widgets::slider(ui, rcx, "p2", &mut p.p2, -0.05, 0.05, &[LensP2]));
    let [x, y] = &mut p.center;
    h = h.or(widgets::drags(ui, rcx, "centre", &mut [x, y], 0.001, (0.0, 1.0), &[LensCenterX, LensCenterY]));
    ui.weak("k1 < 0 barrel · r = 1 at the corners");
    h
}

/// What the overlay needs from the viewer.
pub struct Viewer<'a> {
    /// The node's OUTPUT format, in the pixels `to_screen` takes.
    pub out: (f32, f32),
    /// Its input's format.
    pub src: (f32, f32),
    pub to_screen: &'a dyn Fn(Vec2) -> Pos2,
    pub from_screen: &'a dyn Fn(Pos2) -> Vec2,
}

/// Grid cells across the undistorted picture.
const GRID: usize = 12;
/// Samples along each grid line.
const SAMPLES: usize = 40;

/// Draw the overlay; the principal point drags. True when it moved.
pub fn gizmo(ui: &mut egui::Ui, ink: &mut InkShapes, node: u64, p: &mut LensDistortParams, v: &Viewer) -> bool {
    let g = p.geom([v.src.0, v.src.1], [v.out.0, v.out.1]);
    // The undistorted picture: the output undistorting, the input
    // distorting. `to_out` carries its pixels into the output.
    let (usize_, undist_is_out) = match p.mode {
        LensMode::Undistort => (v.out, true),
        LensMode::Distort => (v.src, false),
    };
    let to_out = |q: [f32; 2]| -> Option<[f32; 2]> { if undist_is_out { Some(q) } else { p.out_of(&g, q) } };
    let scr = |q: [f32; 2]| (v.to_screen)(Vec2::new(q[0], q[1]));
    let polyline = |ink: &mut InkShapes, pts: &mut dyn Iterator<Item = Option<[f32; 2]>>, w: f32, a: f32| {
        let mut last: Option<Pos2> = None;
        for q in pts {
            let s = q.map(scr);
            if let (Some(a0), Some(b0)) = (last, s) {
                ink.seg(a0, b0, w, a, false);
            }
            last = s;
        }
    };
    let (w, h) = usize_;
    let nx = GRID;
    let ny = ((GRID as f32 * h / w.max(1.0)).round() as usize).max(2);
    let cell = [w / nx as f32, h / ny as f32];
    for i in 0..=nx {
        let x = i as f32 * cell[0];
        let mut it = (0..=SAMPLES).map(|k| to_out([x, h * k as f32 / SAMPLES as f32]));
        polyline(ink, &mut it, 1.0, 0.45);
    }
    for j in 0..=ny {
        let y = j as f32 * cell[1];
        let mut it = (0..=SAMPLES).map(|k| to_out([w * k as f32 / SAMPLES as f32, y]));
        polyline(ink, &mut it, 1.0, 0.45);
    }
    // The source frame's edge, where it lands: inside it the output has
    // picture, outside it is black.
    let (sw, sh) = v.src;
    let edge = |t: f32| -> [f32; 2] {
        let t = t * 4.0;
        match t as usize {
            0 => [sw * t, 0.0],
            1 => [sw, sh * (t - 1.0)],
            2 => [sw * (3.0 - t), sh],
            _ => [0.0, sh * (4.0 - t).max(0.0)],
        }
    };
    let n = SAMPLES * 4;
    let mut it = (0..=n).map(|k| p.out_of(&g, edge(k as f32 / n as f32)));
    polyline(ink, &mut it, 2.0, 0.9);
    // The principal point.
    let (co, _) = p.centres(&g);
    let pc = scr(co);
    let rs = ui.interact(
        egui::Rect::from_center_size(pc, egui::vec2(18.0, 18.0)),
        ui.id().with(("lens-center", node)),
        egui::Sense::drag(),
    );
    let hot = rs.hovered() || rs.dragged();
    ink.circle_stroke(pc, 6.0, 1.5, 0.9, hot);
    ink.seg(pc - Vec2::X * 10.0, pc + Vec2::X * 10.0, 1.0, 0.9, hot);
    ink.seg(pc - Vec2::Y * 10.0, pc + Vec2::Y * 10.0, 1.0, 0.9, hot);
    let mut moved = false;
    if let (true, Some(ptr)) = (rs.dragged(), ui.ctx().pointer_latest_pos()) {
        // Absolute under the pointer: the centre is wherever it is.
        let o = (v.from_screen)(ptr);
        p.center = [
            0.5 + (o.x - v.out.0 * 0.5) / g.plate[0].max(1.0),
            0.5 + (o.y - v.out.1 * 0.5) / g.plate[1].max(1.0),
        ];
        moved = true;
    }
    moved
}

/// The graph glyph: a barrel-bent grid in a circle — a lens.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.circle(c, r * 0.9, fill, stroke);
    let ink = Stroke::new((r * 0.1).max(1.0), crate::theme::node_ink());
    let h = r * 0.55;
    let n = 10;
    let bend = |q: Vec2| -> Vec2 {
        let r2 = (q / h).length_sq();
        q * (1.0 - 0.22 * r2)
    };
    for k in [-1.0f32, 0.0, 1.0] {
        let across: Vec<Pos2> =
            (0..=n).map(|i| c + bend(Vec2::new((i as f32 / n as f32 * 2.0 - 1.0) * h, k * h))).collect();
        let down: Vec<Pos2> =
            (0..=n).map(|i| c + bend(Vec2::new(k * h, (i as f32 / n as f32 * 2.0 - 1.0) * h))).collect();
        painter.add(egui::Shape::line(across, ink));
        painter.add(egui::Shape::line(down, ink));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::graph::NodeKind;

    #[test]
    fn channels_round_trip() {
        let mut p = LensDistortParams::default();
        for (i, ch) in CHANNELS.iter().enumerate() {
            set(&mut p, *ch, i as f32 + 0.5);
            assert_eq!(get(&p, *ch), Some(i as f32 + 0.5), "{ch:?}");
            assert!(label(*ch).is_some());
        }
    }

    /// Both pages fit the declared tiles, both ways.
    #[test]
    fn the_panel_fits_its_tiles() {
        for tab in ["lens", "frame"] {
            let mut p = LensDistortParams { chromatic: true, ..Default::default() };
            let mut anim = Default::default();
            let ekind = NodeKind::LensDistort(p);
            let (over, wide) = (std::cell::Cell::new(0.0f32), std::cell::Cell::new(0.0f32));
            let t = crate::widgets::TILE;
            let w = t * TILES as f32 + crate::widgets::TILE_GAP * (TILES as f32 - 1.0);
            let shot = crate::snapshot::shoot(w - 14.0, t - 14.0 - 22.0, |ui| {
                let out = egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                    let mut rcx = RowCtx {
                        node: 1,
                        anim: &mut anim,
                        ekind: &ekind,
                        t: 0,
                        hud_live: false,
                        driven: Vec::new(),
                    };
                    let _ = panel(ui, &mut rcx, &mut p, tab);
                });
                over.set(out.content_size.y - out.inner_rect.height());
                wide.set(out.content_size.x - out.inner_rect.width());
            });
            shot.save(&format!("lensdistort-panel-{tab}"));
            assert!(over.get() <= 0.5, "{tab}: overflows DOWN by {}pt", over.get());
            assert!(wide.get() <= 0.5, "{tab}: overflows SIDEWAYS by {}pt", wide.get());
        }
    }
}
