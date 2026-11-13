//! Gradient's app half: the ramp editor (the panel's bar of stops), the
//! viewer HUD's pieces, the glyph and the channel bindings. The maths, the
//! WGSL and the params are `engine::nodes::gradient`.
//!
//! The panel and the viewer HUD edit the same stops and share one stop
//! SELECTION (egui temp memory, keyed by node), so picking a stop on the
//! picture puts its swatch in the panel and the other way round.
//!
//! Stops are indexed channels (`GradPos(i)`, `GradR(i)`…), so a stop's
//! position and colour key, patch and HUD-slide like any other value. The
//! index is the stop's identity — the engine sorts at pack time — which is
//! why removing one shifts every later stop's curves down with it
//! ([`remove_stop`]).

use crate::anim::{AnimCurve, Channel};
use crate::colorpicker::{self, PickSpec};
use crate::widgets::{self, HudSlider, RowCtx};
use egui::{Color32, Painter, Pos2, Rect, Sense, Stroke, Vec2};
use engine::color::Colorspace;
use engine::graph::{GradInterp, GradientParams, GRAD_STOPS};
use std::collections::BTreeMap;

/// The fixed channels, in panel order. The stops' own are indexed
/// ([`stop_channels`]).
pub const CHANNELS: &[Channel] =
    &[Channel::GradAngle, Channel::GradCenterX, Channel::GradCenterY, Channel::GradScale];

/// The ramp wants width to aim along.
pub const TILES: usize = 2;

/// Stop `i`'s channels: position, then straight r g b a.
pub fn stop_channels(i: u16) -> [Channel; 5] {
    [Channel::GradPos(i), Channel::GradR(i), Channel::GradG(i), Channel::GradB(i), Channel::GradA(i)]
}

pub fn get(p: &GradientParams, ch: Channel) -> Option<f32> {
    use Channel::*;
    let stop = |i: u16| p.stops.get(i as usize);
    Some(match ch {
        GradAngle => p.angle,
        GradCenterX => p.center[0],
        GradCenterY => p.center[1],
        GradScale => p.scale,
        GradPos(i) => stop(i)?.pos,
        GradR(i) => stop(i)?.color[0],
        GradG(i) => stop(i)?.color[1],
        GradB(i) => stop(i)?.color[2],
        GradA(i) => stop(i)?.color[3],
        _ => return None,
    })
}

pub fn set(p: &mut GradientParams, ch: Channel, v: f32) {
    use Channel::*;
    let (i, lane) = match ch {
        GradAngle => return p.angle = v,
        GradCenterX => return p.center[0] = v,
        GradCenterY => return p.center[1] = v,
        GradScale => return p.scale = v,
        GradPos(i) => (i, 4),
        GradR(i) => (i, 0),
        GradG(i) => (i, 1),
        GradB(i) => (i, 2),
        GradA(i) => (i, 3),
        _ => return,
    };
    if let Some(s) = p.stops.get_mut(i as usize) {
        if lane == 4 {
            s.pos = v;
        } else {
            s.color[lane] = v;
        }
    }
}

/// The curve editor's name for a fixed channel.
pub fn label(ch: Channel) -> Option<&'static str> {
    use Channel::*;
    Some(match ch {
        GradAngle => "angle",
        GradCenterX => "center x",
        GradCenterY => "center y",
        GradScale => "ramp scale",
        _ => return None,
    })
}

/// The curve editor's name for a stop's channel.
pub fn stop_label(ch: Channel) -> Option<String> {
    use Channel::*;
    let (i, what) = match ch {
        GradPos(i) => (i, "position"),
        GradR(i) => (i, "r"),
        GradG(i) => (i, "g"),
        GradB(i) => (i, "b"),
        GradA(i) => (i, "alpha"),
        _ => return None,
    };
    Some(format!("stop{} {what}", i + 1))
}

/// Remove stop `i`, sliding every later stop's curves down one index so
/// each keeps its own animation. Refuses to leave the ramp empty.
pub fn remove_stop(p: &mut GradientParams, anim: &mut BTreeMap<Channel, AnimCurve>, i: usize) {
    if p.stops.len() <= 1 || i >= p.stops.len() {
        return;
    }
    let n = p.stops.len();
    p.stops.remove(i);
    for ch in stop_channels(i as u16) {
        anim.remove(&ch);
    }
    for k in i + 1..n {
        for (to, from) in stop_channels(k as u16 - 1).into_iter().zip(stop_channels(k as u16)) {
            if let Some(c) = anim.remove(&from) {
                anim.insert(to, c);
            }
        }
    }
}

/// The selected stop, shared by the panel and the viewer HUD.
pub fn selected(ctx: &egui::Context, node: u64, n: usize) -> usize {
    ctx.data(|d| d.get_temp::<usize>(sel_id(node))).unwrap_or(0).min(n.saturating_sub(1))
}

pub fn select(ctx: &egui::Context, node: u64, i: usize) {
    ctx.data_mut(|d| d.insert_temp(sel_id(node), i));
}

fn sel_id(node: u64) -> egui::Id {
    egui::Id::new(("grad-sel", node))
}

/// A stop's straight colour on screen.
pub fn swatch(c: [f32; 4], spec: &PickSpec) -> Color32 {
    colorpicker::preview([c[0], c[1], c[2]], c[3], spec)
}

/// Paint the ramp itself into `r` along `from -> to` (the panel's bar is
/// left to right; the viewer's follows the axis). Over a checker, so alpha
/// reads as alpha.
pub fn paint_ramp(painter: &Painter, r: Rect, p: &GradientParams, working: Colorspace, spec: &PickSpec) {
    colorpicker::paint_checker(painter, r, 5.0);
    const N: usize = 64;
    let mut mesh = egui::Mesh::default();
    for k in 0..=N {
        let t = k as f32 / N as f32;
        let c = swatch(p.sample(t, working), spec);
        let x = r.left() + t * r.width();
        mesh.colored_vertex(Pos2::new(x, r.top()), c);
        mesh.colored_vertex(Pos2::new(x, r.bottom()), c);
        if k > 0 {
            let b = (2 * k) as u32;
            mesh.add_triangle(b - 2, b - 1, b);
            mesh.add_triangle(b - 1, b + 1, b);
        }
    }
    painter.add(egui::Shape::mesh(mesh));
}

/// A stop marker: a small house shape pointing at `tip`, filled with the
/// stop's colour. `up` points it up (the panel's markers sit under the bar).
pub fn paint_marker(painter: &Painter, tip: Pos2, dir: Vec2, fill: Color32, sel: bool, hot: bool) {
    let n = Vec2::new(-dir.y, dir.x);
    let s = 6.0;
    let pts = vec![
        tip,
        tip + dir * s + n * s,
        tip + dir * (2.2 * s) + n * s,
        tip + dir * (2.2 * s) - n * s,
        tip + dir * s - n * s,
    ];
    let ink = if sel {
        Color32::WHITE
    } else if hot {
        Color32::from_gray(220)
    } else {
        Color32::from_gray(20)
    };
    painter.add(egui::Shape::convex_polygon(pts, fill, Stroke::new(if sel { 2.0 } else { 1.0 }, ink)));
}

/// What the ramp editor did this frame.
#[derive(Default)]
pub struct RampOut {
    pub changed: bool,
    /// Alt-clicked a stop: the caller removes it (it moves curves, so it
    /// needs the node's animation, which the editor does not hold).
    pub remove: Option<usize>,
}

/// The panel's ramp editor: the bar, and the markers under it. Click the
/// bar to add a stop (it takes the colour the ramp already has there),
/// drag a marker to slide it, alt-click one to remove it. Edits `e` — the
/// EVALUATED params; the caller writes what changed back.
pub fn ramp_editor(
    ui: &mut egui::Ui,
    node: u64,
    e: &mut GradientParams,
    working: Colorspace,
    spec: &PickSpec,
) -> RampOut {
    let mut out = RampOut::default();
    let w = ui.available_width().max(60.0);
    let (rect, resp) = ui.allocate_exact_size(Vec2::new(w, 38.0), Sense::click_and_drag());
    widgets::note_rect(ui, egui::Id::new(("grad-ramp", node)), rect);
    let bar = Rect::from_min_size(rect.min + Vec2::new(7.0, 2.0), Vec2::new(w - 14.0, 20.0));
    let x_of = |t: f32| bar.left() + t.clamp(0.0, 1.0) * bar.width();
    let t_of = |x: f32| ((x - bar.left()) / bar.width()).clamp(0.0, 1.0);
    let sel = selected(ui.ctx(), node, e.stops.len());
    let drag_id = egui::Id::new(("grad-ramp-drag", node));

    // The marker nearest the pointer within reach, by x.
    let hit = |p: Pos2, e: &GradientParams| {
        e.stops
            .iter()
            .enumerate()
            .map(|(i, s)| (i, (x_of(s.pos) - p.x).abs()))
            .filter(|(_, d)| *d < 7.0)
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(i, _)| i)
    };
    let alt = ui.input(|i| i.modifiers.alt);
    if resp.drag_started() || resp.clicked() {
        if let Some(p) = resp.interact_pointer_pos() {
            match hit(p, e) {
                Some(i) if alt => out.remove = Some(i),
                Some(i) => {
                    select(ui.ctx(), node, i);
                    ui.ctx().data_mut(|d| d.insert_temp(drag_id, i));
                }
                None if resp.clicked() => {
                    if let Some(i) = e.add_stop(t_of(p.x), working) {
                        select(ui.ctx(), node, i);
                        out.changed = true;
                    }
                }
                None => {}
            }
        }
    }
    if resp.dragged() {
        let d: Option<usize> = ui.ctx().data(|d| d.get_temp(drag_id));
        if let (Some(i), Some(p)) = (d, resp.interact_pointer_pos()) {
            if let Some(s) = e.stops.get_mut(i) {
                let t = t_of(p.x);
                if s.pos != t {
                    s.pos = t;
                    out.changed = true;
                }
            }
        }
    }
    if resp.drag_stopped() {
        ui.ctx().data_mut(|d| d.remove::<usize>(drag_id));
    }

    let painter = ui.painter_at(rect.expand(2.0));
    paint_ramp(&painter, bar, e, working, spec);
    painter.rect_stroke(bar, 0.0, Stroke::new(1.0, Color32::from_gray(60)), egui::StrokeKind::Outside);
    let hover = resp.hover_pos().and_then(|p| hit(p, e));
    for (i, s) in e.stops.iter().enumerate() {
        if i == sel {
            continue;
        }
        let tip = Pos2::new(x_of(s.pos), bar.bottom() + 1.0);
        paint_marker(&painter, tip, Vec2::DOWN, swatch(s.color, spec), false, hover == Some(i));
    }
    if let Some(s) = e.stops.get(sel) {
        let tip = Pos2::new(x_of(s.pos), bar.bottom() + 1.0);
        paint_marker(&painter, tip, Vec2::DOWN, swatch(s.color, spec), true, false);
    }
    resp.on_hover_text("click to add a stop · drag a stop to slide it · alt-click removes");
    out
}

/// Write the fields of `e` that differ from `ev` into the AUTHORED params
/// `p` — so a keyed channel sees exactly its own change, and the props
/// write-back turns it into a key. Stops past `ev`'s end are new: they
/// are appended as they are.
pub fn write_changes(p: &mut GradientParams, ev: &GradientParams, e: &GradientParams) {
    for (i, s) in e.stops.iter().enumerate() {
        match ev.stops.get(i) {
            Some(b) => {
                if let Some(a) = p.stops.get_mut(i) {
                    if s.pos != b.pos {
                        a.pos = s.pos;
                    }
                    for c in 0..4 {
                        if s.color[c] != b.color[c] {
                            a.color[c] = s.color[c];
                        }
                    }
                }
            }
            None => p.stops.push(*s),
        }
    }
}

/// What the panel hands back.
#[derive(Default)]
pub struct PanelOut {
    pub hud: Option<HudSlider>,
    /// A stop to remove once the node borrow is released.
    pub remove: Option<usize>,
}

/// The properties panel.
pub fn panel(
    ui: &mut egui::Ui,
    rcx: &mut RowCtx,
    p: &mut GradientParams,
    working: Colorspace,
    spec: PickSpec,
) -> PanelOut {
    use Channel::*;
    let mut out = PanelOut::default();
    let node = rcx.node;
    let ev = match rcx.ekind {
        engine::graph::NodeKind::Gradient(g) => g.clone(),
        _ => p.clone(),
    };
    widgets::gen_format_row(ui, ("grad-format", node), &mut p.size, &mut p.inherit);
    ui.horizontal(|ui| {
        ui.label("mix");
        egui::ComboBox::from_id_salt(("grad-interp", node))
            .width(64.0)
            .selected_text(p.interp.label())
            .show_ui(ui, |ui| {
                for m in GradInterp::ALL {
                    ui.selectable_value(&mut p.interp, m, m.label());
                }
            })
            .response
            .on_hover_text(
                "linear: the numbers mixed, the way light mixes · \
                 lch: lightness, chroma and hue mixed (OkLCh), so the \
                 middle keeps its colour",
            );
    });

    // The bar.
    let mut e = ev.clone();
    let r = ramp_editor(ui, node, &mut e, working, &spec);
    out.remove = r.remove;
    if r.changed {
        write_changes(p, &ev, &e);
    }

    // The selected stop: its colour, its position, remove / reverse.
    let n = p.stops.len();
    let sel = selected(ui.ctx(), node, n);
    if let (Some(es), true) = (e.stops.get(sel).copied(), sel < n) {
        ui.horizontal(|ui| {
            ui.label(format!("stop {}/{}", sel + 1, n));
            let mut c = es.color;
            if colorpicker::button(ui, (node, "grad-stop", sel), &mut c, spec).changed() {
                let before = e.clone();
                e.stops[sel].color = c;
                write_changes(p, &before, &e);
            }
            if ui
                .add_enabled(n > 1, egui::Button::new("remove"))
                .on_hover_text("alt-click a stop does the same")
                .clicked()
            {
                out.remove = Some(sel);
            }
            if ui.button("reverse").on_hover_text("mirror every stop along the ramp").clicked() {
                let before = e.clone();
                for s in e.stops.iter_mut() {
                    s.pos = 1.0 - s.pos;
                }
                write_changes(p, &before, &e);
            }
        });
        let i = sel as u16;
        let mut pos = p.stops[sel].pos;
        out.hud = out.hud.or(widgets::slider(ui, rcx, "position", &mut pos, 0.0, 1.0, &[GradPos(i)]));
        p.stops[sel].pos = pos;
    }
    if n >= GRAD_STOPS {
        ui.weak(format!("{GRAD_STOPS} stops is the most a ramp holds"));
    }
    ui.separator();
    out.hud = out.hud.or(widgets::slider(ui, rcx, "angle", &mut p.angle, -180.0, 180.0, &[GradAngle]));
    out.hud = out.hud.or(widgets::drags(
        ui,
        rcx,
        "center",
        {
            let [cx, cy] = &mut p.center;
            &mut [cx, cy]
        },
        0.002,
        (0.0, 1.0),
        &[GradCenterX, GradCenterY],
    ));
    out.hud = out.hud.or(widgets::slider(ui, rcx, "scale", &mut p.scale, 0.05, 4.0, &[GradScale]));
    ui.weak("scale 1 runs corner to corner at any angle · drag the ends in the viewer");
    out
}

/// The graph glyph: a disc ramped dark to light, left to right.
pub fn glyph(painter: &Painter, c: Pos2, r: f32, fill: Color32, stroke: Stroke) {
    painter.circle(c, r * 0.9, Color32::from_gray(20), stroke);
    const N: usize = 5;
    let rr = r * 0.9;
    for k in 0..N {
        let x0 = -rr + 2.0 * rr * k as f32 / N as f32;
        let x1 = -rr + 2.0 * rr * (k + 1) as f32 / N as f32;
        let y0 = (rr * rr - x0 * x0).max(0.0).sqrt();
        let y1 = (rr * rr - x1 * x1).max(0.0).sqrt();
        let a = (k + 1) as f32 / N as f32;
        let col = Color32::from_rgb(
            (fill.r() as f32 * a) as u8,
            (fill.g() as f32 * a) as u8,
            (fill.b() as f32 * a) as u8,
        );
        let pts = vec![
            Pos2::new(c.x + x0, c.y - y0),
            Pos2::new(c.x + x1, c.y - y1),
            Pos2::new(c.x + x1, c.y + y1),
            Pos2::new(c.x + x0, c.y + y0),
        ];
        painter.add(egui::Shape::convex_polygon(pts, col, Stroke::NONE));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::graph::GradStop;

    #[test]
    fn channels_round_trip_on_every_stop() {
        let mut p = GradientParams::default();
        p.stops.push(GradStop { pos: 0.4, color: [0.1, 0.2, 0.3, 0.5] });
        for i in 0..p.stops.len() as u16 {
            for (k, ch) in stop_channels(i).into_iter().enumerate() {
                let v = 0.1 * (k as f32 + 1.0) + i as f32;
                set(&mut p, ch, v);
                assert_eq!(get(&p, ch), Some(v), "{ch:?}");
            }
        }
        assert_eq!(get(&p, Channel::GradPos(7)), None, "no such stop");
    }

    #[test]
    fn removing_a_stop_slides_later_curves_down_with_their_stops() {
        let mut p = GradientParams::default();
        p.stops.push(GradStop { pos: 0.5, color: [1.0, 0.0, 0.0, 1.0] });
        let mut anim = BTreeMap::new();
        let mut c = AnimCurve::default();
        c.set_key(3, 0.25);
        anim.insert(Channel::GradPos(2), c);
        anim.insert(Channel::GradR(1), AnimCurve::default());
        remove_stop(&mut p, &mut anim, 1);
        assert_eq!(p.stops.len(), 2);
        assert_eq!(p.stops[1].color[0], 1.0, "the red stop moved down");
        assert!(anim.contains_key(&Channel::GradPos(1)), "its curve came with it");
        assert!(!anim.contains_key(&Channel::GradPos(2)));
        assert!(!anim.contains_key(&Channel::GradR(1)), "the removed stop's curve is gone");
    }

    #[test]
    fn the_last_stop_stays() {
        let mut p = GradientParams::default();
        let mut anim = BTreeMap::new();
        remove_stop(&mut p, &mut anim, 0);
        remove_stop(&mut p, &mut anim, 0);
        assert_eq!(p.stops.len(), 1);
    }

    #[test]
    fn a_write_back_touches_only_what_moved() {
        let mut authored = GradientParams::default();
        authored.stops[1].color = [0.9, 0.9, 0.9, 1.0];
        // The evaluated copy differs from the authored one (a keyed green),
        // and the edit moves only stop 0's position.
        let mut ev = authored.clone();
        ev.stops[1].color[1] = 0.2;
        let mut e = ev.clone();
        e.stops[0].pos = 0.3;
        write_changes(&mut authored, &ev, &e);
        assert_eq!(authored.stops[0].pos, 0.3);
        assert_eq!(authored.stops[1].color[1], 0.9, "the keyed lane is left to its curve");
    }
}
