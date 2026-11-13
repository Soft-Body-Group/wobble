//! Picking and moving a set of points on the viewer, shared by the gizmos
//! whose data is "some points" (GridWarp's lattice, Warp's curves).
//!
//! A click picks a point, S-click adds or drops one, a drag on the
//! background bands a box of them, and a drag on a picked point moves the
//! pick. Two or more picked wear an affine box: drag inside it to move,
//! corner and edge handles scale about the opposite side (alt: about the
//! pivot), ctrl on an edge handle skews, the knob on its stalk turns about
//! the pivot, and the pivot (a crosshair, the box centre until moved)
//! drags anywhere.
//!
//! The module never touches the caller's data. It hands back an [`Affine`]
//! from the PRESS, which the caller lays on its own snapshot of the points
//! as they were at the press ([`Pick::snap`], filled when
//! [`Drag::started`]) — the gizmo rule: absolute from the press, never
//! summed per frame, because the points may be animated channels that are
//! re-evaluated at the playhead every frame. Coordinates are the node's
//! pixels; `to_screen` / `from_screen` carry them through whatever deforms
//! the picture downstream.

use crate::hudkeys;
use crate::ink::InkShapes;
use crate::widgets;
use egui::{Pos2, Rect, Vec2};

/// An affine map of node pixels: `pivot + m (p - pivot) + t`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Affine {
    pub m: [f32; 4],
    pub pivot: Vec2,
    pub t: Vec2,
}

impl Affine {
    pub const IDENTITY: Affine = Affine { m: [1.0, 0.0, 0.0, 1.0], pivot: Vec2::ZERO, t: Vec2::ZERO };

    pub fn apply(&self, p: Vec2) -> Vec2 {
        let q = p - self.pivot;
        self.pivot + Vec2::new(self.m[0] * q.x + self.m[1] * q.y, self.m[2] * q.x + self.m[3] * q.y) + self.t
    }

    /// The same map on a normalised point of a `size` format.
    pub fn apply_norm(&self, q: [f32; 2], size: (f32, f32)) -> [f32; 2] {
        let p = self.apply(Vec2::new(q[0] * size.0, q[1] * size.1));
        [p.x / size.0.max(1.0), p.y / size.1.max(1.0)]
    }
}

/// What a drag does. Every mode is one affine map, computed from the press.
#[derive(Clone, Copy, Debug)]
enum Mode {
    Move,
    /// Scale about a pivot, on x and/or y.
    Scale(bool, bool, Vec2),
    Rotate(Vec2),
    /// Shear: `true` slides x along y (a top or bottom edge dragged
    /// sideways), `false` slides y along x.
    Skew(bool, Vec2),
    /// Not the points at all: the pivot itself.
    Pivot,
}

/// The map a drag lays on the points: from the press `o` to the pointer
/// `cur`, both in node pixels.
fn affine(mode: Mode, o: Vec2, cur: Vec2) -> Affine {
    let lin = |m: [f32; 4], pivot: Vec2| Affine { m, pivot, t: Vec2::ZERO };
    match mode {
        Mode::Move | Mode::Pivot => Affine { t: cur - o, ..Affine::IDENTITY },
        Mode::Scale(ax, ay, pv) => {
            let den = o - pv;
            let fx = if ax && den.x.abs() > 0.5 { (cur.x - pv.x) / den.x } else { 1.0 };
            let fy = if ay && den.y.abs() > 0.5 { (cur.y - pv.y) / den.y } else { 1.0 };
            lin([fx, 0.0, 0.0, fy], pv)
        }
        Mode::Rotate(pv) => {
            let a = (cur - pv).angle() - (o - pv).angle();
            let (s, c) = a.sin_cos();
            lin([c, -s, s, c], pv)
        }
        Mode::Skew(along_x, pv) => {
            let d = cur - o;
            if along_x {
                let den = o.y - pv.y;
                lin([1.0, if den.abs() > 0.5 { d.x / den } else { 0.0 }, 0.0, 1.0], pv)
            } else {
                let den = o.x - pv.x;
                lin([1.0, 0.0, if den.abs() > 0.5 { d.y / den } else { 0.0 }, 1.0], pv)
            }
        }
    }
}

/// A drag in flight, from its press.
#[derive(Clone, Debug)]
struct Grab {
    mode: Mode,
    origin: Pos2,
    /// A pivot the user placed, as it was at the press: it rides along.
    pivot0: Option<Vec2>,
    moving: Vec<bool>,
    /// The rated pointer (z/x/c/v), and the raw one last frame.
    vp: Pos2,
    last: Pos2,
}

/// The pick and the gesture in flight. Keep one per gizmo (egui temp
/// memory, keyed by node) and hand it to [`run`] every frame.
#[derive(Clone, Debug, Default)]
pub struct Pick {
    pub pick: Vec<bool>,
    grab: Option<Grab>,
    marquee: Option<Rect>,
    /// The pick's pivot (node pixels), for the pick it was set on; `None`
    /// is the box centre.
    pivot: Option<(Vec<bool>, Vec2)>,
    /// The caller's data at the press of the drag in flight: whatever it
    /// needs to lay the map on (filled when [`Drag::started`]).
    pub snap: Vec<[f32; 2]>,
    /// The grips the box drew this frame (for suites).
    pub hot: Option<BoxHot>,
}

/// Where the pick's box drew its grips, in screen points: the eight scale
/// handles (corner (-1, -1) first, clockwise), the rotate knob, the pivot.
#[derive(Clone, Copy, Debug)]
pub struct BoxHot {
    pub scale: [Pos2; 8],
    pub knob: Pos2,
    pub pivot: Pos2,
}

impl Pick {
    /// Match the pick to `n` points; a change of count drops it (the
    /// points were renumbered).
    pub fn fit(&mut self, n: usize) {
        if self.pick.len() != n {
            self.pick = vec![false; n];
            self.grab = None;
            self.pivot = None;
        }
    }

    pub fn set(&mut self, n: usize, picked: &[usize]) {
        self.fit(n);
        self.pick = (0..n).map(|k| picked.contains(&k)).collect();
    }

    pub fn count(&self) -> usize {
        self.pick.iter().filter(|b| **b).count()
    }

    pub fn first(&self) -> Option<usize> {
        self.pick.iter().position(|b| *b)
    }

    pub fn grabbing(&self) -> bool {
        self.grab.is_some()
    }
}

/// The viewer as the gizmo sees it.
pub struct Frame<'a> {
    /// Node pixels to screen points and back.
    pub to_screen: &'a dyn Fn(Vec2) -> Pos2,
    pub from_screen: &'a dyn Fn(Pos2) -> Vec2,
    /// The viewer's rect: handles that would land off it come inside.
    pub rect: Rect,
    /// A plain primary drag on the viewer background this frame (no pan,
    /// no scrub): it bands points.
    pub banding: bool,
    /// Salt for the widget ids (the node).
    pub id: egui::Id,
}

/// What a frame of gestures came to.
#[derive(Default)]
pub struct Out {
    /// A drag in flight: the points it moves and the map from the press.
    pub drag: Option<Drag>,
    /// The pointer is over something of ours (a point, the box): a click
    /// there is not a click on the canvas.
    pub hot: bool,
    /// A click on a point, with the keys held then (S toggled the pick
    /// already; D, say, the caller may answer).
    pub clicked: Option<(usize, egui::Modifiers)>,
}

pub struct Drag {
    /// This frame is the press: fill [`Pick::snap`] now.
    pub started: bool,
    pub moving: Vec<bool>,
    pub map: Affine,
}

/// Draw the handles of `pts` (node pixels), run every gesture, and say
/// what to do to the points. Pushes its keys into the HUD strip.
pub fn run(
    ui: &mut egui::Ui,
    ink: &mut InkShapes,
    st: &mut Pick,
    pts: &[Vec2],
    f: &Frame,
    hud_keys: &mut Vec<hudkeys::HudKey>,
) -> Out {
    let n = pts.len();
    st.fit(n);
    let mut out = Out::default();
    let ctx = ui.ctx().clone();
    let sp = |q: Vec2| (f.to_screen)(q);
    let scr: Vec<Pos2> = pts.iter().map(|p| sp(*p)).collect();
    let pointer = ctx.pointer_latest_pos();
    let press = ui.input(|i| i.pointer.press_origin()).or(pointer);
    let mods = ui.input(|i| i.modifiers);
    let s_held = ui.input(|i| i.key_down(egui::Key::S));
    let n_pick = st.count();
    if hud_keys.is_empty() {
        hud_keys.push(hudkeys::key("S", "click: add to pick", egui::Key::S));
        hud_keys.push(hudkeys::chord("drag", "box: pick points"));
        if n_pick >= 2 {
            hud_keys.push(hudkeys::chord("alt", "scale about the pivot"));
            hud_keys.push(hudkeys::chord("ctrl", "drag an edge: skew"));
        }
    }
    let mut started = false;
    let begin = |st: &mut Pick, mode: Mode, at: Pos2, started: &mut bool| {
        let origin = press.unwrap_or(at);
        let pivot0 = st.pivot.as_ref().filter(|(p, _)| *p == st.pick).map(|(_, v)| *v);
        st.grab = Some(Grab { mode, origin, pivot0, moving: st.pick.clone(), vp: origin, last: origin });
        *started = true;
    };

    // The box, registered BEFORE the points so a point under it still wins.
    st.hot = None;
    if n_pick >= 2 {
        let (mut lo, mut hi) = (Vec2::splat(f32::MAX), Vec2::splat(f32::MIN));
        for k in (0..n).filter(|k| st.pick[*k]) {
            lo = lo.min(pts[k]);
            hi = hi.max(pts[k]);
        }
        let pad = (Vec2::splat(2.0) - (hi - lo)).max(Vec2::ZERO) * 0.5;
        let (lo, hi) = (lo - pad, hi + pad);
        let at = |hx: i8, hy: i8| {
            Vec2::new(lo.x + (hi.x - lo.x) * (hx as f32 + 1.0) * 0.5, lo.y + (hi.y - lo.y) * (hy as f32 + 1.0) * 0.5)
        };
        if st.pivot.as_ref().is_some_and(|(p, _)| *p != st.pick) {
            st.pivot = None;
        }
        let pivot = st.pivot.as_ref().map_or(at(0, 0), |(_, p)| *p);
        let corners = [at(-1, -1), at(1, -1), at(1, 1), at(-1, 1)];
        for k in 0..4 {
            ink.seg(sp(corners[k]), sp(corners[(k + 1) % 4]), 1.0, 0.45, false);
        }
        let quad: Vec<Pos2> = corners.iter().map(|c| sp(*c)).collect();
        let side = |p: Pos2, k: usize| {
            let (a, b) = (quad[k], quad[(k + 1) % 4]);
            (b - a).x * (p - a).y - (b - a).y * (p - a).x
        };
        let inside = |p: Pos2| (0..4).all(|k| side(p, k) >= 0.0) || (0..4).all(|k| side(p, k) <= 0.0);
        // The body: a move grip under everything else.
        let body = Rect::from_points(&quad);
        let rb = ui.interact(body, f.id.with("pp-box"), egui::Sense::click_and_drag());
        let in_body = press.is_some_and(inside);
        if rb.drag_started() && st.grab.is_none() && in_body {
            begin(st, Mode::Move, body.center(), &mut started);
        }
        if rb.clicked() && in_body && !s_held && st.grab.is_none() {
            st.pick.iter_mut().for_each(|b| *b = false);
        }
        out.hot |= pointer.is_some_and(inside);
        let centre = sp(at(0, 0));
        let safe = f.rect.shrink(10.0);
        let pushed = |hp: Pos2| {
            let d = hp - centre;
            if d.length() < 1e-3 {
                return hp;
            }
            let o = hp + d.normalized() * 14.0;
            if safe.contains(o) { o } else { hp - d.normalized() * 14.0 }
        };
        const BOX: [(i8, i8); 8] = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)];
        let mut hot = BoxHot { scale: [Pos2::ZERO; 8], knob: Pos2::ZERO, pivot: Pos2::ZERO };
        for (k, (hx, hy)) in BOX.iter().copied().enumerate() {
            let hp = pushed(sp(at(hx, hy)));
            hot.scale[k] = hp;
            let rs = ui.interact(Rect::from_center_size(hp, Vec2::splat(20.0)), f.id.with(("pp-scale", k)), egui::Sense::drag());
            out.hot |= rs.hovered();
            ink.rect(Rect::from_center_size(hp, Vec2::splat(9.0)), 1.5, false, 1.2, 1.0, rs.hovered());
            if rs.drag_started() && st.grab.is_none() {
                let edge = (hx == 0) != (hy == 0);
                let about = if mods.alt { pivot } else { at(-hx, -hy) };
                let mode = if edge && mods.command { Mode::Skew(hx == 0, about) } else { Mode::Scale(hx != 0, hy != 0, about) };
                begin(st, mode, hp, &mut started);
            }
        }
        // Rotate: a knob on a stalk off the top edge (the bottom one when
        // the top is off the viewer).
        let stalk = |edge: Pos2| {
            let d = edge - centre;
            let dir = if d.length() > 1e-3 { d.normalized() } else { Vec2::new(0.0, -1.0) };
            (edge, edge + dir * 30.0)
        };
        let (top, knob) = [stalk(sp(at(0, -1))), stalk(sp(at(0, 1)))]
            .into_iter()
            .find(|(_, k)| safe.contains(*k))
            .unwrap_or_else(|| stalk(sp(at(0, -1))));
        let rr = ui.interact(Rect::from_center_size(knob, Vec2::splat(20.0)), f.id.with("pp-rot"), egui::Sense::drag());
        out.hot |= rr.hovered();
        ink.seg(top, knob, 1.0, 0.5, false);
        ink.circle_stroke(knob, 5.5, 1.5, 1.0, rr.hovered());
        if rr.drag_started() && st.grab.is_none() {
            begin(st, Mode::Rotate(pivot), knob, &mut started);
        }
        // The pivot: a crosshair, dragged anywhere.
        let pp = sp(pivot);
        let rp = ui.interact(Rect::from_center_size(pp, Vec2::splat(16.0)), f.id.with("pp-pivot"), egui::Sense::drag());
        out.hot |= rp.hovered();
        let lit = rp.hovered() || rp.dragged();
        ink.circle_stroke(pp, 4.0, 1.2, 0.9, lit);
        ink.seg(pp - Vec2::new(8.0, 0.0), pp + Vec2::new(8.0, 0.0), 1.2, 0.9, lit);
        ink.seg(pp - Vec2::new(0.0, 8.0), pp + Vec2::new(0.0, 8.0), 1.2, 0.9, lit);
        if rp.drag_started() && st.grab.is_none() {
            begin(st, Mode::Pivot, pp, &mut started);
            if let Some(g) = st.grab.as_mut() {
                g.pivot0 = Some(pivot);
            }
            // Moving the pivot moves no point.
            started = false;
        }
        hot.knob = knob;
        hot.pivot = pp;
        st.hot = Some(hot);
    }

    // The points, registered last so they win.
    for (k, p0) in scr.iter().enumerate() {
        let rs = ui.interact(Rect::from_center_size(*p0, Vec2::splat(16.0)), f.id.with(("pp-pt", k)), egui::Sense::click_and_drag());
        out.hot |= rs.hovered() || rs.is_pointer_button_down_on();
        let picked = st.pick[k];
        ink.rect(Rect::from_center_size(*p0, Vec2::splat(if picked { 9.0 } else { 7.0 })), 1.5, true, 0.0, 1.0, rs.hovered());
        if picked {
            ink.circle_stroke(*p0, 9.0, 1.2, 0.9, rs.hovered());
        }
        if rs.clicked() && st.grab.is_none() {
            if s_held {
                st.pick[k] = !st.pick[k];
            } else {
                st.pick.iter_mut().for_each(|b| *b = false);
                st.pick[k] = true;
            }
            out.clicked = Some((k, mods));
        }
        if rs.drag_started() && st.grab.is_none() {
            // Grabbing a point outside the pick makes it the pick: a drag
            // moves what is under the cursor.
            if !st.pick[k] {
                st.pick.iter_mut().for_each(|b| *b = false);
                st.pick[k] = true;
            }
            begin(st, Mode::Move, *p0, &mut started);
        }
    }

    // The gesture in flight, from its press.
    let rate = widgets::sens_factor(ui).unwrap_or(1.0);
    if let Some(g) = st.grab.as_mut() {
        let raw = pointer.unwrap_or(g.last);
        g.vp += (raw - g.last) * rate;
        g.last = raw;
        let (cur, o) = ((f.from_screen)(g.vp), (f.from_screen)(g.origin));
        if let Mode::Pivot = g.mode {
            let p0 = g.pivot0.unwrap_or(o);
            st.pivot = Some((st.pick.clone(), p0 + (cur - o)));
        } else {
            let map = affine(g.mode, o, cur);
            // A pivot the user placed rides with the points.
            if let Some(p0) = g.pivot0 {
                st.pivot = Some((g.moving.clone(), map.apply(p0)));
            }
            out.drag = Some(Drag { started, moving: g.moving.clone(), map });
        }
        if rate != 1.0 {
            widgets::claim_sens(&ctx);
        }
        if !ui.input(|i| i.pointer.primary_down()) {
            st.grab = None;
        }
    }

    // A box on the background picks what it covers (S keeps the pick).
    if f.banding && st.grab.is_none() && !out.hot {
        if let (Some(o), Some(p)) = (press, pointer) {
            st.marquee = Some(Rect::from_two_pos(o, p));
        }
    }
    if let Some(mq) = st.marquee {
        ink.rect(mq, 0.0, false, 1.0, 0.5, false);
        if !f.banding {
            if !s_held {
                st.pick.iter_mut().for_each(|b| *b = false);
            }
            for (k, p) in scr.iter().enumerate() {
                if mq.contains(*p) {
                    st.pick[k] = true;
                }
            }
            st.marquee = None;
        }
    }
    // A plain click on empty picture drops the pick.
    if ui.input(|i| i.pointer.primary_clicked()) && st.grab.is_none() && !out.hot && !s_held {
        if pointer.is_some_and(|p| f.rect.contains(p) && !scr.iter().any(|h| (*h - p).length() < 10.0)) {
            st.pick.iter_mut().for_each(|b| *b = false);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_maps_do_what_the_handles_say() {
        let o = Vec2::new(10.0, 0.0);
        let rot = affine(Mode::Rotate(Vec2::ZERO), o, Vec2::new(0.0, 10.0));
        let p = rot.apply(Vec2::new(5.0, 0.0));
        assert!((p - Vec2::new(0.0, 5.0)).length() < 1e-4, "{p:?}");
        let sc = affine(Mode::Scale(true, false, Vec2::ZERO), o, Vec2::new(20.0, 7.0));
        assert_eq!(sc.apply(Vec2::new(3.0, 4.0)), Vec2::new(6.0, 4.0));
        let sk = affine(Mode::Skew(true, Vec2::ZERO), Vec2::new(0.0, 10.0), Vec2::new(5.0, 10.0));
        assert_eq!(sk.apply(Vec2::new(0.0, 4.0)), Vec2::new(2.0, 4.0));
        let mv = affine(Mode::Move, o, Vec2::new(13.0, -2.0));
        assert_eq!(mv.apply(Vec2::new(1.0, 1.0)), Vec2::new(4.0, -1.0));
        assert_eq!(mv.apply_norm([0.5, 0.5], (10.0, 10.0)), [0.8, 0.3]);
    }
}
