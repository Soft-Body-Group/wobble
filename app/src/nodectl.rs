//! CONTROLS inside a node, revealed by zoom — the innards precedent
//! (`crate::innards`) with knobs instead of a graph.
//!
//! Zoom in on a node that offers controls and its glyph gives way to them,
//! drawn inside its outline: an outer DIAL riding the rim (a Transform's
//! rotate — drag around it) and a column of draggable VALUES (drag
//! sideways; shift is fine, Blender's precision key; double-click resets).
//!
//! Every edit is a CHANNEL value, handed back to the app through
//! `NodeEditor::ctl_edits` and applied by the SAME keyed write-back the
//! panel rows use — so autokey, keys at the playhead, locks and patch
//! cables all behave as they do on the row. Values are shown EVALUATED at
//! the playhead, and a drag is computed ABSOLUTELY from its press (the
//! pointer's travel since the press, never "last value + delta"), which is
//! what keeps it right on an animated channel.
//!
//! This module knows nothing about the editor. A kind joins with one arm in
//! `controls()`.

use crate::anim::{self, Channel};
use crate::theme;
use egui::{Color32, Pos2, Rect, Stroke, Vec2};
use engine::graph::NodeKind;

/// How a value responds to a sideways drag of `dx` screen px.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Resp {
    /// `start + dx × rate`.
    Add(f32),
    /// `start × e^(dx × rate)` — a scale, which stays positive and feels
    /// the same at 0.1 as at 10.
    Mul(f32),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Ctl {
    /// The outer ring: degrees, clockwise on screen (the viewer gizmo's
    /// sense, y-down).
    Dial { ch: Channel, reset: f32 },
    /// One row. Several channels move together (a ganged scale), each
    /// from its own press value.
    Value { label: &'static str, chans: Vec<Channel>, resp: Resp, reset: f32, digits: usize },
}

/// The one registry: which kinds offer controls, and which.
pub fn controls(kind: &NodeKind) -> Option<Vec<Ctl>> {
    use Channel::*;
    let v = |label, chans: &[Channel], resp, reset, digits| Ctl::Value {
        label,
        chans: chans.to_vec(),
        resp,
        reset,
        digits,
    };
    match kind {
        NodeKind::Transform(t) => {
            let mut c = vec![
                Ctl::Dial { ch: Rotate, reset: 0.0 },
                v("x", &[TranslateX], Resp::Add(1.0), 0.0, 1),
                v("y", &[TranslateY], Resp::Add(1.0), 0.0, 1),
            ];
            if t.scale_gang {
                c.push(v("scale", &[ScaleX, ScaleY], Resp::Mul(0.005), 1.0, 3));
            } else {
                c.push(v("scale x", &[ScaleX], Resp::Mul(0.005), 1.0, 3));
                c.push(v("scale y", &[ScaleY], Resp::Mul(0.005), 1.0, 3));
            }
            c.push(v("pivot x", &[CenterX], Resp::Add(0.002), 0.5, 3));
            c.push(v("pivot y", &[CenterY], Resp::Add(0.002), 0.5, 3));
            Some(c)
        }
        _ => None,
    }
}

/// What a pointer is over: the dial, or row `i` of the controls (its index
/// in `controls()`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Hit {
    Dial,
    Value(usize),
}

/// Where the controls of a host drawn at `c` with radius `r` sit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Place {
    pub c: Pos2,
    pub r: f32,
}

impl Place {
    /// The dial band: an annulus hugging the node's rim.
    fn dial_r(&self) -> f32 {
        self.r * 0.9
    }

    fn in_dial(&self, p: Pos2) -> bool {
        let d = (p - self.c).length();
        d >= self.r * 0.76 && d <= self.r * 1.08
    }

    /// Row `k` of `n`, stacked in the square inside the dial.
    fn row(&self, k: usize, n: usize) -> Rect {
        // The stack's corners stay inside the dial band's inner edge.
        let w = self.r * 1.1;
        let h = (self.r * 0.9 / n.max(1) as f32).min(self.r * 0.2);
        let top = self.c.y - h * n as f32 * 0.5;
        Rect::from_min_size(Pos2::new(self.c.x - w * 0.5, top + h * k as f32), Vec2::new(w, h))
    }
}

/// A gesture in progress. `acc` is the pointer's travel since the press —
/// px sideways for a value, degrees round for the dial — with shift's fine
/// rate folded in per frame, so pressing shift mid-drag never jumps.
#[derive(Clone, Debug, PartialEq)]
pub struct Drag {
    pub host: u64,
    pub hit: Hit,
    pub place: Place,
    /// The channels' values at the press, evaluated at the playhead.
    start: Vec<(Channel, f32)>,
    acc: f32,
    last: Pos2,
}

/// 0 (glyph) → 1 (controls): the innards' reveal, so both kinds of node
/// open up at the same zoom.
pub fn reveal(r: f32) -> f32 {
    crate::innards::reveal(r)
}

pub fn live(r: f32) -> bool {
    crate::innards::live(r)
}

fn chans_of(ctl: &Ctl) -> Vec<Channel> {
    match ctl {
        Ctl::Dial { ch, .. } => vec![*ch],
        Ctl::Value { chans, .. } => chans.clone(),
    }
}

fn rows(ctls: &[Ctl]) -> Vec<usize> {
    ctls.iter()
        .enumerate()
        .filter(|(_, c)| matches!(c, Ctl::Value { .. }))
        .map(|(i, _)| i)
        .collect()
}

/// The rows' rects for a host at `place` — what `paint` draws and returns.
pub fn layout(place: Place, kind: &NodeKind) -> Vec<(Hit, Rect)> {
    let Some(ctls) = controls(kind) else { return Vec::new() };
    let ids = rows(&ctls);
    ids.iter().enumerate().map(|(k, &i)| (Hit::Value(i), place.row(k, ids.len()))).collect()
}

/// What the pointer at `p` is over, for the highlight.
pub fn hover(place: Place, kind: &NodeKind, p: Pos2) -> Option<Hit> {
    hit_at(place, &layout(place, kind), p)
}

/// Draw the controls of a node whose EVALUATED kind is `live_kind` (its
/// curves at the playhead), `animated` naming the channels that carry
/// keys. Returns the hit rects (the dial is tested as an annulus, not a
/// rect — see `hit_at`).
#[allow(clippy::too_many_arguments)]
pub fn paint(
    painter: &egui::Painter,
    place: Place,
    live_kind: &NodeKind,
    animated: &dyn Fn(Channel) -> bool,
    hot: Option<Hit>,
    alpha: f32,
    stroke: Stroke,
) -> Vec<(Hit, Rect)> {
    let mut hits = Vec::new();
    let Some(ctls) = controls(live_kind) else { return hits };
    if alpha <= 0.0 {
        return hits;
    }
    let fade = |c: Color32| c.gamma_multiply(alpha);
    let Place { c, r } = place;
    // A floor, so the host's glyph and through-wire stay out of the numbers.
    painter.circle(c, r * 0.98, fade(theme::graph_bg()), Stroke::new(stroke.width.max(1.0), fade(stroke.color)));
    for (i, ctl) in ctls.iter().enumerate() {
        match ctl {
            Ctl::Dial { ch, .. } => {
                let deg = anim::get(live_kind, *ch).unwrap_or(0.0);
                let is_hot = hot == Some(Hit::Dial);
                let keyed = animated(*ch);
                let dr = place.dial_r();
                let w = r * if is_hot { 0.1 } else { 0.07 };
                painter.circle_stroke(c, dr, Stroke::new(w, fade(theme::lift(theme::graph_bg(), 22))));
                // The sweep from 12 o'clock to the angle, then the knob.
                use std::f32::consts::FRAC_PI_2;
                let a0 = -FRAC_PI_2;
                let a1 = a0 + deg.to_radians();
                let ink = if keyed { theme::keyframe() } else if is_hot { theme::text_hot() } else { theme::text() };
                let steps = ((a1 - a0).abs() * 12.0).ceil().clamp(2.0, 400.0) as usize;
                let pts: Vec<Pos2> = (0..=steps)
                    .map(|k| {
                        let a = a0 + (a1 - a0) * k as f32 / steps as f32;
                        c + Vec2::angled(a) * dr
                    })
                    .collect();
                painter.add(egui::Shape::line(pts, Stroke::new(w * 0.6, fade(ink.gamma_multiply(0.7)))));
                painter.line_segment(
                    [c + Vec2::angled(a0) * (dr - w), c + Vec2::angled(a0) * (dr + w)],
                    Stroke::new((w * 0.25).max(1.0), fade(theme::text())),
                );
                painter.circle_filled(c + Vec2::angled(a1) * dr, w * 1.1, fade(ink));
                if is_hot {
                    painter.text(
                        c + Vec2::angled(a1) * (dr + w * 3.5),
                        egui::Align2::CENTER_CENTER,
                        format!("{deg:.1}°"),
                        egui::FontId::proportional((r * 0.14).clamp(9.0, 14.0)),
                        fade(ink),
                    );
                }
            }
            Ctl::Value { label, chans, digits, .. } => {
                let Some(rr) = layout(place, live_kind).into_iter().find(|(h, _)| *h == Hit::Value(i)).map(|(_, r)| r)
                else {
                    continue;
                };
                let is_hot = hot == Some(Hit::Value(i));
                if is_hot {
                    painter.rect_filled(rr, rr.height() * 0.3, fade(theme::lift(theme::graph_bg(), 18)));
                }
                let keyed = chans.iter().any(|ch| animated(*ch));
                let v = anim::get(live_kind, chans[0]).unwrap_or(0.0);
                let size = (rr.height() * 0.7).clamp(7.0, 15.0);
                let font = egui::FontId::proportional(size);
                painter.text(
                    rr.left_center() + Vec2::new(rr.height() * 0.2, 0.0),
                    egui::Align2::LEFT_CENTER,
                    *label,
                    font.clone(),
                    fade(theme::text().gamma_multiply(0.75)),
                );
                let ink = if keyed { theme::keyframe() } else if is_hot { theme::text_hot() } else { theme::text() };
                painter.text(
                    rr.right_center() - Vec2::new(rr.height() * 0.2, 0.0),
                    egui::Align2::RIGHT_CENTER,
                    format!("{v:.*}", *digits),
                    egui::FontId::monospace(size),
                    fade(ink),
                );
                hits.push((Hit::Value(i), rr));
            }
        }
    }
    hits
}

/// What is under `p`: a row first, then the dial band.
pub fn hit_at(place: Place, hits: &[(Hit, Rect)], p: Pos2) -> Option<Hit> {
    if let Some((h, _)) = hits.iter().find(|(_, r)| r.contains(p)) {
        return Some(*h);
    }
    place.in_dial(p).then_some(Hit::Dial)
}

/// A press at `p` on `host`, whose evaluated kind is `live_kind`.
pub fn begin(host: u64, live_kind: &NodeKind, place: Place, hits: &[(Hit, Rect)], p: Pos2) -> Option<Drag> {
    let hit = hit_at(place, hits, p)?;
    let ctls = controls(live_kind)?;
    let ctl = match hit {
        Hit::Dial => ctls.iter().find(|c| matches!(c, Ctl::Dial { .. }))?,
        Hit::Value(i) => ctls.get(i)?,
    };
    let start = chans_of(ctl).into_iter().filter_map(|ch| Some((ch, anim::get(live_kind, ch)?))).collect();
    Some(Drag { host, hit, place, start, acc: 0.0, last: p })
}

/// The pointer is at `p`: the channel values the drag now asks for.
pub fn update(d: &mut Drag, kind: &NodeKind, p: Pos2, fine: bool) -> Vec<(Channel, f32)> {
    let k = if fine { 0.1 } else { 1.0 };
    let Some(ctls) = controls(kind) else { return Vec::new() };
    match d.hit {
        Hit::Dial => {
            let c = d.place.c;
            let mut a = ((p - c).angle() - (d.last - c).angle()).to_degrees();
            if a > 180.0 {
                a -= 360.0;
            } else if a < -180.0 {
                a += 360.0;
            }
            // Too near the centre the angle is noise.
            if (p - c).length() > d.place.r * 0.15 {
                d.acc += a * k;
            }
            d.last = p;
            d.start.iter().map(|&(ch, v)| (ch, v + d.acc)).collect()
        }
        Hit::Value(i) => {
            d.acc += (p.x - d.last.x) * k;
            d.last = p;
            let Some(Ctl::Value { resp, .. }) = ctls.get(i) else { return Vec::new() };
            d.start
                .iter()
                .map(|&(ch, v)| {
                    let nv = match *resp {
                        Resp::Add(rate) => v + d.acc * rate,
                        Resp::Mul(rate) => v * (d.acc * rate).exp(),
                    };
                    (ch, nv)
                })
                .collect()
        }
    }
}

/// A double-click on `hit`: its channels back to their reset value.
pub fn reset(kind: &NodeKind, hit: Hit) -> Vec<(Channel, f32)> {
    let Some(ctls) = controls(kind) else { return Vec::new() };
    let ctl = match hit {
        Hit::Dial => ctls.iter().find(|c| matches!(c, Ctl::Dial { .. })),
        Hit::Value(i) => ctls.get(i),
    };
    match ctl {
        Some(c @ Ctl::Dial { reset, .. }) | Some(c @ Ctl::Value { reset, .. }) => {
            chans_of(c).into_iter().map(|ch| (ch, *reset)).collect()
        }
        None => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::graph::TransformParams;

    fn place() -> Place {
        Place { c: Pos2::new(200.0, 200.0), r: 100.0 }
    }

    fn xf(t: TransformParams) -> NodeKind {
        NodeKind::Transform(t)
    }

    fn hits(k: &NodeKind) -> Vec<(Hit, Rect)> {
        let ctx = egui::Context::default();
        let mut out = Vec::new();
        let _ = ctx.run_ui(Default::default(), |ui| {
            out = paint(ui.painter(), place(), k, &|_| false, None, 1.0, Stroke::NONE);
        });
        out
    }

    #[test]
    fn a_transform_has_a_dial_and_rows_and_a_grade_has_none() {
        let k = xf(Default::default());
        let c = controls(&k).unwrap();
        assert_eq!(c.iter().filter(|c| matches!(c, Ctl::Dial { .. })).count(), 1);
        assert!(controls(&NodeKind::Grade(Default::default())).is_none());
        // Every row is hit-testable and inside the dial.
        let h = hits(&k);
        assert_eq!(h.len(), c.len() - 1);
        for (_, r) in &h {
            for p in [r.left_top(), r.right_top(), r.left_bottom(), r.right_bottom()] {
                assert!((p - place().c).length() < place().r * 0.76, "row {r:?} runs into the dial");
            }
        }
    }

    #[test]
    fn the_dial_follows_the_pointer_round_from_its_press() {
        let k = xf(TransformParams { rotate: 10.0, ..Default::default() });
        let c = place().c;
        let press = c + Vec2::new(0.0, -90.0); // 12 o'clock, on the band
        let mut d = begin(1, &k, place(), &hits(&k), press).expect("the band is the dial");
        assert_eq!(d.hit, Hit::Dial);
        // A quarter turn clockwise on screen (y-down), in small steps, and
        // on past 180 without a wrap jump.
        let mut last = Vec::new();
        for s in 1..=27 {
            let a = -std::f32::consts::FRAC_PI_2 + (s as f32 * 10.0).to_radians();
            last = update(&mut d, &k, c + Vec2::angled(a) * 90.0, false);
        }
        assert!((last[0].1 - 280.0).abs() < 1e-2, "{last:?}");
    }

    #[test]
    fn a_value_drags_from_its_press_and_a_ganged_scale_moves_both() {
        let k = xf(TransformParams { translate: [5.0, 0.0], scale: [2.0, 1.0], scale_gang: true, ..Default::default() });
        let h = hits(&k);
        let row = |label: &str| {
            let c = controls(&k).unwrap();
            let i = c.iter().position(|c| matches!(c, Ctl::Value { label: l, .. } if *l == label)).unwrap();
            h.iter().find(|(x, _)| *x == Hit::Value(i)).unwrap().1.center()
        };
        let p = row("x");
        let mut d = begin(1, &k, place(), &h, p).unwrap();
        update(&mut d, &k, p + Vec2::new(10.0, 0.0), false);
        let v = update(&mut d, &k, p + Vec2::new(30.0, 5.0), false);
        assert_eq!(v, vec![(Channel::TranslateX, 35.0)], "30 px of travel from 5, once");
        let v = update(&mut d, &k, p + Vec2::new(40.0, 5.0), true);
        assert_eq!(v, vec![(Channel::TranslateX, 36.0)], "shift: a tenth, and no jump");

        let p = row("scale");
        let mut d = begin(1, &k, place(), &h, p).unwrap();
        let v = update(&mut d, &k, p + Vec2::new(200.0, 0.0), false);
        let e = 1.0f32.exp();
        assert!((v[0].1 - 2.0 * e).abs() < 1e-4 && (v[1].1 - e).abs() < 1e-4, "{v:?}");
        assert_eq!(reset(&k, Hit::Value(3)), vec![(Channel::ScaleX, 1.0), (Channel::ScaleY, 1.0)]);
    }

    #[test]
    fn the_centre_is_neither_dial_nor_row_gap() {
        let k = xf(Default::default());
        let h = hits(&k);
        assert_eq!(hit_at(place(), &h, place().c + Vec2::new(0.0, place().r * 1.5)), None);
    }
}

/// For LOOKING: a Transform at three zooms through the real editor.
/// `cargo test -p app nodectl_render -- --ignored` writes
/// `target/nodectl-*.png`.
#[cfg(test)]
mod render {
    use crate::editor::{EditorGraph, NodeEditor};
    use egui::{Pos2, Vec2};
    use engine::graph::{NodeKind, TransformParams};

    /// The real editor's input path, headless: a zoomed-in Transform's
    /// dial takes a quarter turn from a press on its band, a row takes a
    /// sideways drag, and neither drags the node.
    #[test]
    fn the_editor_routes_a_dial_and_a_row_drag_into_channel_edits() {
        use crate::anim::Channel;
        use egui::{Event, PointerButton, RawInput};
        let mut g = EditorGraph::default();
        g.nodes.clear();
        g.edges.clear();
        let id = g.add(NodeKind::Transform(TransformParams::default()), Pos2::ZERO);
        let home = g.node(id).unwrap().pos;
        let mut ed = NodeEditor::default();
        ed.zoom = 6.0;
        ed.pan = Vec2::ZERO;
        let ctx = egui::Context::default();
        let (mut sel, mut act, mut pin, mut slot, mut rects) = (Vec::new(), Vec::new(), Vec::new(), 2usize, Vec::new());
        let mut t = 0.0;
        let mut frame = |ed: &mut NodeEditor, g: &mut EditorGraph, events: Vec<Event>| {
            t += 1.0 / 60.0;
            let input = RawInput {
                screen_rect: Some(egui::Rect::from_min_size(Pos2::ZERO, Vec2::new(900.0, 800.0))),
                time: Some(t),
                events,
                ..Default::default()
            };
            let _ = ctx.run_ui(input, |ui| {
                ed.show(ui, g, &mut sel, &mut act, &mut pin, &mut slot, &mut rects);
            });
            std::mem::take(&mut ed.ctl_edits)
        };
        let mv = |p: Pos2| Event::PointerMoved(p);
        let btn = |p: Pos2, pressed| Event::PointerButton { pos: p, button: PointerButton::Primary, pressed, modifiers: Default::default() };
        let gesture = |frame: &mut dyn FnMut(&mut NodeEditor, &mut EditorGraph, Vec<Event>) -> Vec<(u64, Channel, f32)>,
                       ed: &mut NodeEditor,
                       g: &mut EditorGraph,
                       path: &[Pos2]| {
            let mut out = Vec::new();
            frame(ed, g, vec![mv(path[0])]);
            frame(ed, g, vec![btn(path[0], true)]);
            for p in &path[1..] {
                out.extend(frame(ed, g, vec![mv(*p)]));
            }
            let end = *path.last().unwrap();
            out.extend(frame(ed, g, vec![btn(end, false)]));
            frame(ed, g, vec![]);
            out
        };
        for _ in 0..3 {
            frame(&mut ed, &mut g, vec![]);
        }
        let (host, place, rows, live) = ed.ctl_hits.iter().find(|(h, ..)| *h == id).cloned().expect("controls drawn at zoom 6");
        assert!(live, "zoom 6 is far enough in to edit");
        assert_eq!(host, id);

        // The dial: 12 o'clock to 3 o'clock round the band, in steps.
        let dr = place.r * 0.9;
        let path: Vec<Pos2> = (0..=9)
            .map(|k| place.c + Vec2::angled(-std::f32::consts::FRAC_PI_2 + (k as f32 * 10.0).to_radians()) * dr)
            .collect();
        let edits = gesture(&mut frame, &mut ed, &mut g, &path);
        let last = edits.iter().rev().find(|(n, ch, _)| *n == id && *ch == Channel::Rotate).expect("the dial edits rotate");
        assert!((last.2 - 90.0).abs() < 1.0, "a quarter turn: {edits:?}");

        // The "x" row, dragged 40 px right from its middle.
        let x_row = rows.iter().find(|(h, _)| *h == super::Hit::Value(1)).unwrap().1.center();
        let path: Vec<Pos2> = (0..=8).map(|k| x_row + Vec2::new(k as f32 * 5.0, 0.0)).collect();
        let edits = gesture(&mut frame, &mut ed, &mut g, &path);
        let last = edits.iter().rev().find(|(_, ch, _)| *ch == Channel::TranslateX).expect("the row edits translate x");
        assert!(last.2 > 20.0 && last.2 <= 40.0, "{edits:?}");
        assert_eq!(g.node(id).unwrap().pos, home, "the host never moved");
    }

    #[test]
    #[ignore]
    fn nodectl_render() {
        for (name, z) in [("zoom1", 1.0f32), ("half", 2.5), ("in", 6.0)] {
            let mut g = EditorGraph::default();
            g.nodes.clear();
            g.edges.clear();
            let id = g.add(
                NodeKind::Transform(TransformParams { translate: [12.0, -4.5], rotate: 35.0, scale: [1.25, 1.25], ..Default::default() }),
                Pos2::ZERO,
            );
            let mut ed = NodeEditor::default();
            ed.zoom = z;
            ed.pan = Vec2::ZERO;
            let (mut sel, mut act, mut pin, mut slot, mut rects) = (vec![id], Vec::new(), Vec::new(), 2usize, Vec::new());
            crate::uishot::render(&format!("nodectl-{name}"), 700.0, 600.0, 2.0, 4, |ui, _| {
                ed.show(ui, &mut g, &mut sel, &mut act, &mut pin, &mut slot, &mut rects);
            });
        }
    }
}
