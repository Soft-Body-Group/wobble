//! A painter that remembers what it painted: every primitive names the
//! theme slot it wears and a few words for what it is, so the mock layouts
//! can be clicked. Each primitive reserves a shape slot UNDER itself; the
//! highlight (selected colour, hovered colour, what follows it) is written
//! into those slots after the frame, so it reads as a halo around the very
//! thing it marks and never covers it.

use crate::glyphs::{self, Icon};
use crate::theme::{self, Slot};
use egui::{vec2, Align2, Color32, FontId, Painter, Pos2, Rect, Shape, Stroke};

#[derive(Clone)]
pub enum HitShape {
    Rect(Rect, f32),
    /// An outline: hit near the edge only, so what sits inside stays clickable.
    Ring(Rect, f32, f32),
    Line(Vec<Pos2>, f32),
    Circle(Pos2, f32),
}

impl HitShape {
    fn dist(&self, p: Pos2) -> f32 {
        match self {
            HitShape::Rect(r, _) => r.distance_to_pos(p),
            HitShape::Ring(r, _, w) => {
                let out = r.distance_to_pos(p);
                if out > 0.0 {
                    out
                } else {
                    // Inside: how far from the nearest edge, minus the stroke.
                    let d = (p.x - r.min.x).min(r.max.x - p.x).min(p.y - r.min.y).min(r.max.y - p.y);
                    (d - w * 0.5 - 3.0).max(0.0)
                }
            }
            HitShape::Line(pts, w) => {
                let mut best = f32::MAX;
                for s in pts.windows(2) {
                    best = best.min(seg_dist(p, s[0], s[1]));
                }
                (best - w * 0.5).max(0.0)
            }
            HitShape::Circle(c, r) => ((p - *c).length() - r).max(0.0),
        }
    }

    /// The halo drawn under the element: the element grown by `grow`.
    fn halo(&self, grow: f32, color: Color32) -> Shape {
        match self {
            // An outline, not a fill: text and translucent fills must stay legible.
            // A thin ring at a fixed offset; `grow` only sets its weight.
            HitShape::Rect(r, round) => Shape::rect_stroke(
                r.expand(2.5),
                round + 2.5,
                Stroke::new(grow.min(3.5) * 0.6, color),
                egui::StrokeKind::Middle,
            ),
            HitShape::Ring(r, round, w) => Shape::rect_stroke(
                r.expand(w * 0.5 + 2.5),
                round + w * 0.5 + 2.5,
                Stroke::new(grow.min(3.5) * 0.6, color),
                egui::StrokeKind::Middle,
            ),
            HitShape::Line(pts, w) => Shape::line(pts.clone(), Stroke::new(w + grow * 2.0, color)),
            HitShape::Circle(c, r) => Shape::circle_filled(*c, r + grow, color),
        }
    }
}

fn seg_dist(p: Pos2, a: Pos2, b: Pos2) -> f32 {
    let ab = b - a;
    let t = if ab.length_sq() > 0.0 { ((p - a).dot(ab) / ab.length_sq()).clamp(0.0, 1.0) } else { 0.0 };
    (p - (a + ab * t)).length()
}

pub struct Hit {
    pub shape: HitShape,
    pub slot: Slot,
    /// What the thing is, in a few words (reads well in the scene code).
    #[allow(dead_code)]
    pub what: &'static str,
    under: egui::layers::ShapeIdx,
}

/// Mock layouts are drawn in a fixed 1200 x 760 design space, fitted to
/// whatever room the window gives them.
pub const W: f32 = 1200.0;
pub const H: f32 = 760.0;

pub struct Canvas {
    pub painter: Painter,
    o: Pos2,
    pub s: f32,
    pub hits: Vec<Hit>,
    pub time: f64,
}

impl Canvas {
    pub fn new(painter: Painter, room: Rect, time: f64) -> Self {
        let s = (room.width() / W).min(room.height() / H);
        let o = room.center() - vec2(W, H) * s * 0.5;
        Self { painter, o, s, hits: Vec::new(), time }
    }

    pub fn p(&self, x: f32, y: f32) -> Pos2 {
        self.o + vec2(x, y) * self.s
    }
    pub fn r(&self, x: f32, y: f32, w: f32, h: f32) -> Rect {
        Rect::from_min_size(self.p(x, y), vec2(w, h) * self.s)
    }

    fn reserve(&self) -> egui::layers::ShapeIdx {
        self.painter.add(Shape::Noop)
    }
    fn hit(&mut self, shape: HitShape, slot: Slot, what: &'static str, under: egui::layers::ShapeIdx) {
        self.hits.push(Hit { shape, slot, what, under });
    }

    // --- primitives, all in design-space units --------------------------

    pub fn fill(&mut self, x: f32, y: f32, w: f32, h: f32, round: f32, slot: Slot, what: &'static str) -> Rect {
        let r = self.r(x, y, w, h);
        self.fill_px(r, round, theme::get(slot), slot, what);
        r
    }

    /// A fill in a colour derived from a slot (faded, lifted): still that
    /// slot when clicked.
    pub fn fill_px(&mut self, r: Rect, round: f32, c: Color32, slot: Slot, what: &'static str) {
        let under = self.reserve();
        let round = round * self.s;
        self.painter.rect_filled(r, round, c);
        self.hit(HitShape::Rect(r, round), slot, what, under);
    }

    pub fn outline(&mut self, x: f32, y: f32, w: f32, h: f32, round: f32, width: f32, slot: Slot, what: &'static str) {
        let r = self.r(x, y, w, h);
        self.outline_px(r, round * self.s, width, theme::get(slot), slot, what);
    }

    pub fn outline_px(&mut self, r: Rect, round: f32, width: f32, c: Color32, slot: Slot, what: &'static str) {
        let under = self.reserve();
        let w = (width * self.s).max(1.0);
        self.painter.rect_stroke(r, round, Stroke::new(w, c), egui::StrokeKind::Middle);
        self.hit(HitShape::Ring(r, round, w), slot, what, under);
    }

    pub fn text(&mut self, x: f32, y: f32, s: &str, size: f32, slot: Slot, what: &'static str) -> Rect {
        self.text_a(x, y, Align2::LEFT_TOP, s, size, theme::get(slot), slot, what)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn text_a(&mut self, x: f32, y: f32, align: Align2, s: &str, size: f32, c: Color32, slot: Slot, what: &'static str) -> Rect {
        let under = self.reserve();
        let r = self.painter.text(self.p(x, y), align, s, FontId::proportional(size * self.s), c);
        self.hit(HitShape::Rect(r.expand(1.0), 2.0), slot, what, under);
        r
    }

    pub fn mono(&mut self, x: f32, y: f32, align: Align2, s: &str, size: f32, slot: Slot, what: &'static str) -> Rect {
        let under = self.reserve();
        let r = self.painter.text(self.p(x, y), align, s, FontId::monospace(size * self.s), theme::get(slot));
        self.hit(HitShape::Rect(r.expand(1.0), 2.0), slot, what, under);
        r
    }

    pub fn line(&mut self, pts: &[(f32, f32)], width: f32, slot: Slot, what: &'static str) {
        let c = theme::get(slot);
        self.line_c(pts, width, c, slot, what);
    }

    pub fn line_c(&mut self, pts: &[(f32, f32)], width: f32, c: Color32, slot: Slot, what: &'static str) {
        let under = self.reserve();
        let pts: Vec<Pos2> = pts.iter().map(|&(x, y)| self.p(x, y)).collect();
        let w = (width * self.s).max(1.0);
        self.painter.add(Shape::line(pts.clone(), Stroke::new(w, c)));
        self.hit(HitShape::Line(pts, w), slot, what, under);
    }

    /// A graph noodle, top to bottom: out of `a` downward, into `b` from above.
    pub fn wire(&mut self, a: (f32, f32), b: (f32, f32), width: f32, slot: Slot, what: &'static str) {
        let pts = wire_points(a, b);
        self.line(&pts, width, slot, what);
    }

    pub fn dot(&mut self, x: f32, y: f32, r: f32, slot: Slot, what: &'static str) {
        let c = theme::get(slot);
        self.dot_c(x, y, r, c, slot, what);
    }

    pub fn dot_c(&mut self, x: f32, y: f32, r: f32, c: Color32, slot: Slot, what: &'static str) {
        let under = self.reserve();
        let p = self.p(x, y);
        let r = r * self.s;
        self.painter.circle_filled(p, r, c);
        self.hit(HitShape::Circle(p, r), slot, what, under);
    }

    pub fn ring(&mut self, x: f32, y: f32, r: f32, width: f32, slot: Slot, what: &'static str) {
        let under = self.reserve();
        let p = self.p(x, y);
        let r = r * self.s;
        let w = (width * self.s).max(1.0);
        self.painter.circle_stroke(p, r, Stroke::new(w, theme::get(slot)));
        // Hit as a thick line around the circle.
        let pts: Vec<Pos2> = (0..=32)
            .map(|i| {
                let a = i as f32 / 32.0 * std::f32::consts::TAU;
                p + vec2(a.cos(), a.sin()) * r
            })
            .collect();
        self.hit(HitShape::Line(pts, w), slot, what, under);
    }

    /// A diamond (keyframe).
    pub fn diamond(&mut self, x: f32, y: f32, r: f32, slot: Slot, what: &'static str) {
        let under = self.reserve();
        let c = self.p(x, y);
        let r = r * self.s;
        let pts = vec![c + vec2(0.0, -r), c + vec2(r, 0.0), c + vec2(0.0, r), c + vec2(-r, 0.0)];
        self.painter.add(Shape::convex_polygon(pts, theme::get(slot), Stroke::NONE));
        self.hit(HitShape::Circle(c, r * 0.8), slot, what, under);
    }

    /// A graph node, drawn by Wobble's own shapes (`glyphs`): `r` is its
    /// radius in design units. Hit as its body. Returns where it drew, in
    /// design units, so a name can sit past its right end.
    #[allow(clippy::too_many_arguments)]
    pub fn node(&mut self, x: f32, y: f32, r: f32, icon: Icon, slot: Slot, selected: bool, what: &'static str) -> Rect {
        let under = self.reserve();
        let p = self.p(x, y);
        let scale = r * self.s / glyphs::R;
        let stroke = if selected {
            Stroke::new(2.5, theme::selection())
        } else {
            Stroke::new(1.0, theme::node_outline())
        };
        let e = glyphs::paint(&self.painter, icon, p, scale, theme::get(slot), stroke);
        let shape = if e.width() > e.height() * 1.5 {
            HitShape::Rect(e, e.height() * 0.5)
        } else {
            HitShape::Circle(p, e.width() * 0.5)
        };
        self.hit(shape, slot, what, under);
        if selected {
            // The outline is the selection colour's, not the node's.
            let pts: Vec<Pos2> = (0..=32)
                .map(|i| {
                    let a = i as f32 / 32.0 * std::f32::consts::TAU;
                    p + vec2(a.cos(), a.sin()) * e.width() * 0.5
                })
                .collect();
            let under = self.reserve();
            self.hit(HitShape::Line(pts, 2.5), Slot::selection, "selected node outline", under);
        }
        Rect::from_min_max(self.design(e.min), self.design(e.max))
    }

    /// A screen point back in design units.
    pub fn design(&self, p: Pos2) -> Pos2 {
        Pos2::ZERO + (p - self.o) / self.s
    }

    /// Paint something that is data, not chrome (the image in the viewer):
    /// never a hit.
    pub fn plain(&self, shape: Shape) {
        self.painter.add(shape);
    }

    // --- after the frame ------------------------------------------------

    /// The topmost hit under `p`, within a couple of pixels.
    pub fn pick(&self, p: Pos2) -> Option<usize> {
        let slack = 2.0;
        // Later hits sit on top; among those touching, the closest wins so
        // a thin wire over a panel is easy to grab.
        let mut best: Option<(usize, f32)> = None;
        for (i, h) in self.hits.iter().enumerate().rev() {
            let d = h.shape.dist(p);
            if d <= slack {
                match best {
                    Some((_, bd)) if bd <= d => {}
                    _ => best = Some((i, d)),
                }
                if d == 0.0 {
                    break;
                }
            }
        }
        best.map(|(i, _)| i)
    }

    /// Write the halos: everything `sel` colours pulses, everything `hover`
    /// colours rings. Both are indexed by slot.
    pub fn halos(&self, sel: &[bool], hover: &[bool]) {
        let pulse = (self.time * 3.0).sin() as f32 * 0.5 + 0.5;
        let hot = Color32::from_rgb(255, 225, 60);
        let cool = Color32::from_rgb(90, 220, 255);
        for h in &self.hits {
            // Specks (grid dots) stay quiet: hundreds of them would glow.
            if matches!(h.shape, HitShape::Circle(_, r) if r < 3.0) {
                continue;
            }
            let shape = if sel[h.slot as usize] {
                let g = 2.0 + pulse * 1.5;
                Shape::Vec(vec![h.shape.halo(g + 1.5, Color32::BLACK), h.shape.halo(g, hot)])
            } else if hover[h.slot as usize] {
                Shape::Vec(vec![h.shape.halo(3.0, Color32::BLACK), h.shape.halo(2.0, cool)])
            } else {
                continue;
            };
            self.painter.set(h.under, shape);
        }
    }
}

pub fn wire_points(a: (f32, f32), b: (f32, f32)) -> Vec<(f32, f32)> {
    let dy = ((b.1 - a.1).abs() * 0.5).max(30.0);
    let (p0, p1, p2, p3) = (a, (a.0, a.1 + dy), (b.0, b.1 - dy), b);
    (0..=24)
        .map(|i| {
            let t = i as f32 / 24.0;
            let u = 1.0 - t;
            let w = [u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t];
            (
                w[0] * p0.0 + w[1] * p1.0 + w[2] * p2.0 + w[3] * p3.0,
                w[0] * p0.1 + w[1] * p1.1 + w[2] * p2.1 + w[3] * p3.1,
            )
        })
        .collect()
}
