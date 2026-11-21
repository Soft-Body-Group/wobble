//! Shared control-point curve editor.
//!
//! Every curve-shaped property authors the same thing — a short list of
//! (x, y) points joined linearly or by a monotone cubic — so they all get the
//! same editor: shift-click adds, alt-click removes, a marquee selects, and a
//! selection of two or more raises the same scale box the keyframe curve
//! editor uses. The view navigates with Blender's hands (`crate::nav`) like
//! every canvas, with DECOUPLED axes: ctrl+MMB / ctrl+alt+LMB / alt+RMB
//! drag stretches x and y separately (right = x, up = y), alt+wheel zooms y
//! alone, double-click frames, Q frames the picked points (all of them
//! with none picked). A colour curve in a scene-linear comp has
//! interesting business well above 1.0.
//!
//! `eval()` mirrors `curve_eval` in `engine/src/codegen.rs` line for line —
//! including the extrapolation past the end points — so the drawn response IS
//! the rendered one.

use crate::theme;
use egui::{Color32, Pos2, Rect, Stroke, Vec2};
use engine::graph::CurveInterp;

/// Secant of segment i -> i+1.
fn sec(p: &[[f32; 2]], i: usize) -> f32 {
    (p[i + 1][1] - p[i][1]) / (p[i + 1][0] - p[i][0]).max(1e-5)
}

/// Fritsch-Carlson tangent. Same clamp as the WGSL. At the ends it is flat
/// when `flat_ends` (so two smooth points are a smoothstep — a band's
/// crossfade), otherwise the end segment's secant: the slope the
/// extrapolation carries on with, so a colour curve meets it without a kink.
fn tan_at(p: &[[f32; 2]], i: usize, flat_ends: bool) -> f32 {
    if i == 0 || i + 1 >= p.len() {
        if flat_ends || p.len() < 2 {
            return 0.0;
        }
        return sec(p, if i == 0 { 0 } else { p.len() - 2 });
    }
    let (a, b) = (sec(p, i - 1), sec(p, i));
    if a * b <= 0.0 {
        return 0.0;
    }
    let lim = 3.0 * a.abs().min(b.abs());
    (0.5 * (a + b)).clamp(-lim, lim)
}

/// The curve's value at x. `pts` must be sorted by x.
///
/// Outside the end points the curve EXTRAPOLATES along the end segment's
/// slope rather than flattening: a colour curve that clamps at its last
/// point silently crushes every highlight above it. `flat_ends` picks the
/// smooth curve's end tangents (see `tan_at`): true for a response curve
/// that is clamped to 0..1 anyway, false for a colour curve, whose ends
/// must run into the extrapolation smoothly.
pub fn eval(pts: &[[f32; 2]], x: f32, interp: CurveInterp, flat_ends: bool) -> f32 {
    match pts.len() {
        0 => return x,
        1 => return pts[0][1],
        _ => {}
    }
    let last = pts.len() - 1;
    if x <= pts[0][0] {
        return pts[0][1] + (x - pts[0][0]) * sec(pts, 0);
    }
    if x >= pts[last][0] {
        return pts[last][1] + (x - pts[last][0]) * sec(pts, last - 1);
    }
    for i in 1..pts.len() {
        let (prev, p) = (pts[i - 1], pts[i]);
        if x <= p[0] {
            let h = (p[0] - prev[0]).max(1e-5);
            let t = (x - prev[0]) / h;
            if interp == CurveInterp::Linear {
                return prev[1] + (p[1] - prev[1]) * t;
            }
            let (m0, m1) = (tan_at(pts, i - 1, flat_ends) * h, tan_at(pts, i, flat_ends) * h);
            let (t2, t3) = (t * t, t * t * t);
            return (2.0 * t3 - 3.0 * t2 + 1.0) * prev[1]
                + (t3 - 2.0 * t2 + t) * m0
                + (-2.0 * t3 + 3.0 * t2) * p[1]
                + (t3 - t2) * m1;
        }
    }
    pts[last][1]
}

pub struct CurveCfg<'a> {
    pub size: Vec2,
    /// The range framed on first show and by "frame" (x0, x1, y0, y1).
    pub view: [f32; 4],
    pub interp: CurveInterp,
    /// Smooth end tangents are FLAT (a band's crossfade, clamped to 0..1)
    /// rather than the end segment's slope (a colour curve, which
    /// extrapolates past its ends and must meet that line without a kink).
    pub flat_ends: bool,
    /// Never delete below this many points.
    pub min_points: usize,
    /// The first and last point keep their y (LumaKey's ramp ends ARE 0 and
    /// 1) and cannot be removed.
    pub pin_ends: bool,
    /// Hold y inside these bounds (coverage is 0..1; colour is not).
    pub clamp_y: Option<(f32, f32)>,
    pub tint: Color32,
    /// Draw, but take no edits — the caller has claimed the gesture (a
    /// ctrl-drag pulling the key out to a viewer HUD, say).
    pub frozen: bool,
    /// Paint the plot with the colours its axes MEAN, behind the grid: a
    /// luma ramp under a luma key, a hue wheel under a hue curve, the graded
    /// result under a colour curve. Takes curve-space (x, y).
    ///
    /// This is why it belongs to the editor rather than to one node: a curve
    /// drawn on a bare dark panel makes you hold the axis in your head, and
    /// every curve in this app is a curve over something with a colour.
    pub backdrop: Option<&'a dyn Fn(f32, f32) -> Color32>,
    /// Points whose Y is STRUCTURAL: they move in x only, and cannot be
    /// removed. `pin_ends` is the special case of this for the two ends.
    ///
    /// It exists for curves where some points are not free-floating shape
    /// but the joints of a model — a band boundary is at full coverage or at
    /// none, never at 0.4, and letting one be dragged off 0 or 1 makes a
    /// near-vertical drop that is easy to create and painful to repair.
    pub pin_y: &'a [usize],
    /// Where a new point may be added, in curve x. `None` = anywhere.
    /// Adding one inside a crossfade asks a boundary to be in two places at
    /// once, so a model-backed curve says where its slack actually is.
    pub add_ok: Option<&'a dyn Fn(f32) -> bool>,
    /// Groups of point indices that move together in X. Dragging any member
    /// drags the whole group.
    ///
    /// It exists for boundaries that are SHARED: the point where one band
    /// stops and the next starts is one place, not two that have to be kept
    /// in agreement by hand. Y stays independent, so a mirrored pair can
    /// still cross at different heights.
    pub mirror: &'a [Vec<usize>],
    /// A dragged point STOPS at its neighbour instead of passing it and
    /// re-sorting to the other side. For a curve whose points are the
    /// joints of a model (a band's corners and its slope) swapping two of
    /// them silently re-means them; stopping is the honest answer.
    pub keep_order: bool,
}

impl Default for CurveCfg<'_> {
    fn default() -> Self {
        Self {
            size: egui::vec2(240.0, 160.0),
            view: [0.0, 1.0, 0.0, 1.0],
            interp: CurveInterp::Smooth,
            flat_ends: true,
            min_points: 2,
            pin_ends: false,
            clamp_y: None,
            tint: theme::curve_line(),
            frozen: false,
            backdrop: None,
            pin_y: &[],
            add_ok: None,
            mirror: &[],
            keep_order: false,
        }
    }
}

#[derive(Clone, Debug)]
enum Drag {
    Marquee(Pos2),
    /// Moving the selection: where the drag began, and the points as they
    /// were (so the move is absolute, never an accumulation of deltas).
    Points { origin: Pos2, start: Vec<[f32; 2]> },
    Pan,
    /// Scaling the selection about the opposite edge of the box.
    Scale { origin: Pos2, anchor: (f32, f32), ax: bool, ay: bool, start: Vec<[f32; 2]> },
}

/// Selection, view and gesture, remembered per widget id between frames.
#[derive(Clone)]
struct State {
    sel: Vec<usize>,
    drag: Option<Drag>,
    /// Data value at the left / bottom edge, and pixels per unit.
    x_off: f32,
    y_off: f32,
    x_scale: f32,
    y_scale: f32,
    framed: bool,
}

impl Default for State {
    fn default() -> Self {
        Self {
            sel: Vec::new(),
            drag: None,
            x_off: 0.0,
            y_off: 0.0,
            x_scale: 1.0,
            y_scale: 1.0,
            framed: false,
        }
    }
}

impl State {
    fn load(ctx: &egui::Context, id: egui::Id) -> Self {
        ctx.data_mut(|d| d.get_temp::<Self>(id).unwrap_or_default())
    }
    fn store(self, ctx: &egui::Context, id: egui::Id) {
        ctx.data_mut(|d| d.insert_temp(id, self));
    }
    fn frame(&mut self, r: &Rect, view: [f32; 4]) {
        let (x0, x1, y0, y1) = (view[0], view[1], view[2], view[3]);
        let mx = (x1 - x0).max(1e-4) * 0.06;
        let my = (y1 - y0).max(1e-4) * 0.12;
        self.x_scale = r.width() / ((x1 - x0).max(1e-4) + mx * 2.0);
        self.y_scale = r.height() / ((y1 - y0).max(1e-4) + my * 2.0);
        self.x_off = x0 - mx;
        self.y_off = y0 - my;
        self.framed = true;
    }
    /// Fit the picked points (all of them with none picked) with the same
    /// margins as `frame`. An axis the points do not span — one point, or
    /// a row at one height — keeps its zoom and is centred on them.
    fn frame_points(&mut self, r: &Rect, pts: &[[f32; 2]]) {
        let picked: Vec<[f32; 2]> = self.sel.iter().filter_map(|&i| pts.get(i).copied()).collect();
        let set = if picked.is_empty() { pts } else { &picked[..] };
        let Some(first) = set.first() else { return };
        let (mut x0, mut x1, mut y0, mut y1) = (first[0], first[0], first[1], first[1]);
        for p in set {
            x0 = x0.min(p[0]);
            x1 = x1.max(p[0]);
            y0 = y0.min(p[1]);
            y1 = y1.max(p[1]);
        }
        let (sx, sy) = (self.x_scale, self.y_scale);
        self.frame(r, [x0, x1, y0, y1]);
        const FLAT: f32 = 1e-4;
        if x1 - x0 < FLAT {
            self.x_scale = sx;
            self.x_off = (x0 + x1) * 0.5 - r.width() * 0.5 / sx;
        }
        if y1 - y0 < FLAT {
            self.y_scale = sy;
            self.y_off = (y0 + y1) * 0.5 - r.height() * 0.5 / sy;
        }
    }
    fn x_of(&self, r: &Rect, v: f32) -> f32 {
        r.min.x + (v - self.x_off) * self.x_scale
    }
    fn y_of(&self, r: &Rect, v: f32) -> f32 {
        r.max.y - (v - self.y_off) * self.y_scale
    }
    fn x_at(&self, r: &Rect, px: f32) -> f32 {
        self.x_off + (px - r.min.x) / self.x_scale
    }
    fn y_at(&self, r: &Rect, py: f32) -> f32 {
        self.y_off + (r.max.y - py) / self.y_scale
    }
    fn pos(&self, r: &Rect, p: [f32; 2]) -> Pos2 {
        Pos2::new(self.x_of(r, p[0]), self.y_of(r, p[1]))
    }
}

/// A faint curve drawn behind the editable one (the other channels).
pub struct Ghost<'a> {
    pub pts: &'a [[f32; 2]],
    pub color: Color32,
    /// Fill the area under the curve as well as stroking it. A set of bands
    /// reads as bands when it is filled and as spaghetti when it is not.
    pub fill: bool,
}

/// What the editor did this frame, plus its canvas response so the caller
/// can claim the gesture (see `CurveCfg::frozen`).
pub struct CurveOut {
    pub changed: bool,
    pub response: egui::Response,
    /// Where each point landed on screen, in point order — the self-test
    /// clicks these rather than mirroring the view maths.
    pub screen: Vec<Pos2>,
}

/// Nice round grid step near `target`, in 1/2/5 decades.
fn nice_step(target: f32) -> f32 {
    if !(target.is_finite() && target > 0.0) {
        return 1.0;
    }
    let d = 10f32.powf(target.log10().floor());
    let m = target / d;
    d * if m < 1.5 {
        1.0
    } else if m < 3.5 {
        2.0
    } else if m < 7.5 {
        5.0
    } else {
        10.0
    }
}

/// Draws and edits `pts` in place.
pub fn curve_edit(
    ui: &mut egui::Ui,
    id: egui::Id,
    cfg: &CurveCfg<'_>,
    ghosts: &[Ghost<'_>],
    pts: &mut Vec<[f32; 2]>,
) -> CurveOut {
    let (resp, painter) = ui.allocate_painter(cfg.size, egui::Sense::click_and_drag());
    let full = resp.rect;
    let r = full.shrink(8.0);
    let mut st = State::load(ui.ctx(), id);
    let mut changed = false;
    if !st.framed || !st.x_scale.is_finite() || st.x_scale <= 0.0 {
        st.frame(&r, cfg.view);
    }
    painter.rect_filled(full, 4.0, theme::well());
    let clip = painter.with_clip_rect(full);
    // The axes, painted as what they mean. Drawn first so the grid, the
    // ghosts and the curve all sit ON it.
    if let Some(bg) = cfg.backdrop {
        // A coarse mesh: this is a backdrop, and it is rebuilt every frame
        // while the view pans, so it stays cheap on purpose.
        const NX: usize = 40;
        const NY: usize = 24;
        let mut mesh = egui::epaint::Mesh::default();
        for iy in 0..=NY {
            for ix in 0..=NX {
                let (u, v) = (ix as f32 / NX as f32, iy as f32 / NY as f32);
                let sp = Pos2::new(full.min.x + u * full.width(), full.min.y + v * full.height());
                mesh.colored_vertex(sp, bg(st.x_at(&r, sp.x), st.y_at(&r, sp.y)));
            }
        }
        let w = NX as u32 + 1;
        for iy in 0..NY as u32 {
            for ix in 0..NX as u32 {
                let i = iy * w + ix;
                mesh.indices.extend([i, i + 1, i + w, i + 1, i + w + 1, i + w]);
            }
        }
        clip.add(egui::Shape::mesh(mesh));
        // Knock it back. A backdrop is reference, not subject: at full
        // strength a light one swallows the grid, the ghosts and the curve,
        // and the caller should not have to pre-dim its own colours to stop
        // that happening.
        clip.rect_filled(full, 0.0, theme::hud_bg().gamma_multiply_u8(118));
    }

    pts.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap_or(std::cmp::Ordering::Equal));
    let pointer = resp.hover_pos().or_else(|| ui.ctx().pointer_latest_pos());
    let mods = ui.input(|i| i.modifiers);

    // --- navigation (decoupled axes) --------------------------------------
    if resp.hovered() && !cfg.frozen {
        // The canvas OWNS the wheel while hovered: the properties grid it
        // sits in would otherwise scroll under it (egui's scroll area reads
        // the delta after its content, so clearing it here is in time) and
        // zoom itself on a pinch (read BEFORE this draws — hence the claim
        // left for next frame, see `owns_pointer`).
        claim_wheel(ui, full);
        // Blender's wheel rules (`crate::nav`), same as the keyframe
        // editor: wheel zooms, trackpad fingers pan, ctrl/shift+wheel pan
        // by axis, alt+wheel keeps the y-only zoom.
        let w = crate::nav::wheel_nav(ui);
        st.x_off -= w.pan.x / st.x_scale;
        st.y_off += w.pan.y / st.y_scale;
        let factor = w.zoom;
        if (factor - 1.0).abs() > 1e-4 {
            if let Some(c) = pointer {
                let (fx, fy) = if w.alt { (1.0, factor) } else { (factor, factor) };
                let (dx, dy) = (st.x_at(&r, c.x), st.y_at(&r, c.y));
                st.x_scale = (st.x_scale * fx).clamp(1.0, 1e6);
                st.y_scale = (st.y_scale * fy).clamp(1.0, 1e6);
                st.x_off = dx - (c.x - r.min.x) / st.x_scale;
                st.y_off = dy - (r.max.y - c.y) / st.y_scale;
            }
        }
    }
    if resp.double_clicked() {
        st.frame(&r, cfg.view);
    }
    // Q: frame the picked points, every point with none picked.
    if resp.hovered()
        && !ui.ctx().egui_wants_keyboard_input()
        && ui.input(|i| crate::keymap::pressed(i, crate::keymap::Act::PropsCurveFrame))
    {
        st.frame_points(&r, pts);
    }

    // --- grid --------------------------------------------------------------
    let xstep = nice_step(90.0 / st.x_scale);
    let ystep = nice_step(60.0 / st.y_scale);
    let grid = Stroke::new(0.5, theme::grid_line());
    let axis = Stroke::new(1.0, theme::axis_line());
    let mut gx = (st.x_at(&r, r.min.x) / xstep).ceil() * xstep;
    while st.x_of(&r, gx) <= r.max.x {
        let x = st.x_of(&r, gx);
        clip.line_segment(
            [Pos2::new(x, full.min.y), Pos2::new(x, full.max.y)],
            if gx.abs() < xstep * 0.01 { axis } else { grid },
        );
        clip.text(
            Pos2::new(x + 2.0, full.max.y - 2.0),
            egui::Align2::LEFT_BOTTOM,
            fmt(gx, xstep),
            egui::FontId::proportional(9.0),
            theme::text_weak(),
        );
        gx += xstep;
    }
    let mut gy = (st.y_at(&r, r.max.y) / ystep).ceil() * ystep;
    while st.y_of(&r, gy) >= r.min.y {
        let y = st.y_of(&r, gy);
        clip.line_segment(
            [Pos2::new(full.min.x, y), Pos2::new(full.max.x, y)],
            if gy.abs() < ystep * 0.01 { axis } else { grid },
        );
        clip.text(
            Pos2::new(full.min.x + 2.0, y - 1.0),
            egui::Align2::LEFT_BOTTOM,
            fmt(gy, ystep),
            egui::FontId::proportional(9.0),
            theme::text_weak(),
        );
        gy += ystep;
    }

    // --- curves ------------------------------------------------------------
    let sample = |src: &[[f32; 2]], w: f32, color: Color32, interp: CurveInterp| {
        let n = (r.width().max(16.0) as usize).min(360);
        let line: Vec<Pos2> = (0..=n)
            .map(|i| {
                let x = st.x_at(&r, r.min.x + r.width() * i as f32 / n as f32);
                let mut y = eval(src, x, interp, cfg.flat_ends);
                if let Some((lo, hi)) = cfg.clamp_y {
                    y = y.clamp(lo, hi);
                }
                Pos2::new(st.x_of(&r, x), st.y_of(&r, y))
            })
            .collect();
        egui::Shape::line(line, Stroke::new(w, color))
    };
    // The 1:1 diagonal, so a colour curve reads against "no change".
    // Linear, whatever the curve is: smoothed, two points are an S.
    clip.add(sample(&[[0.0, 0.0], [1.0, 1.0]], 0.5, theme::axis_line(), CurveInterp::Linear));
    for g in ghosts {
        if g.pts.len() < 2 {
            continue;
        }
        if g.fill {
            // Area under the curve, in the ghost's own colour: N bands read
            // as bands, and where two of them cross you can SEE the
            // crossfade instead of inferring it from a number.
            let n = (r.width().max(16.0) as usize).min(360);
            let mut mesh = egui::epaint::Mesh::default();
            let base = st.y_of(&r, cfg.clamp_y.map(|c| c.0).unwrap_or(0.0));
            for i in 0..=n {
                let x = st.x_at(&r, r.min.x + r.width() * i as f32 / n as f32);
                let mut y = eval(g.pts, x, cfg.interp, cfg.flat_ends);
                if let Some((lo, hi)) = cfg.clamp_y {
                    y = y.clamp(lo, hi);
                }
                let sx = st.x_of(&r, x);
                mesh.colored_vertex(Pos2::new(sx, base), Color32::TRANSPARENT);
                mesh.colored_vertex(Pos2::new(sx, st.y_of(&r, y)), g.color);
            }
            for i in 0..n as u32 {
                let k = i * 2;
                mesh.indices.extend([k, k + 1, k + 2, k + 1, k + 3, k + 2]);
            }
            clip.add(egui::Shape::mesh(mesh));
        }
        clip.add(sample(g.pts, 1.0, g.color, cfg.interp));
    }
    clip.add(sample(pts, 1.8, cfg.tint, cfg.interp));

    if cfg.frozen {
        for p in pts.iter() {
            clip.circle_filled(st.pos(&r, *p), 4.0, cfg.tint);
        }
        let screen = pts.iter().map(|p| st.pos(&r, *p)).collect();
        st.store(ui.ctx(), id);
        return CurveOut { changed: false, response: resp, screen };
    }

    // --- selection box (two or more points) --------------------------------
    let n = pts.len();
    st.sel.retain(|i| *i < n);
    let sel_pts: Vec<[f32; 2]> = st.sel.iter().map(|i| pts[*i]).collect();
    let mut handles: Vec<(Rect, (f32, f32), bool, bool)> = Vec::new();
    if sel_pts.len() >= 2 && !matches!(st.drag, Some(Drag::Points { .. })) {
        let (mut x0, mut x1) = (f32::MAX, f32::MIN);
        let (mut y0, mut y1) = (f32::MAX, f32::MIN);
        for p in &sel_pts {
            x0 = x0.min(p[0]);
            x1 = x1.max(p[0]);
            y0 = y0.min(p[1]);
            y1 = y1.max(p[1]);
        }
        let bx = Rect::from_two_pos(
            Pos2::new(st.x_of(&r, x0), st.y_of(&r, y1)),
            Pos2::new(st.x_of(&r, x1), st.y_of(&r, y0)),
        )
        .expand(9.0)
        .intersect(full.shrink(3.0));
        let scaling = matches!(st.drag, Some(Drag::Scale { .. }));
        if bx.width() > 2.0 && bx.height() > 2.0 {
            clip.rect_stroke(
                bx,
                0.0,
                Stroke::new(1.0, if scaling { theme::handle() } else { theme::selection_box() }),
                egui::StrokeKind::Outside,
            );
            // Each handle scales about the OPPOSITE edge, so the far side
            // stays pinned — the same manipulator as the keyframe editor.
            for (hx, hy) in
                [(-1i8, -1i8), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]
            {
                let hp = Pos2::new(
                    bx.center().x + hx as f32 * bx.width() * 0.5,
                    bx.center().y + hy as f32 * bx.height() * 0.5,
                );
                let anchor = (
                    if hx > 0 { x0 } else { x1 },
                    if hy > 0 { y1 } else { y0 },
                );
                clip.rect_filled(
                    Rect::from_center_size(hp, egui::vec2(7.0, 7.0)),
                    1.0,
                    theme::handle(),
                );
                handles.push((
                    Rect::from_center_size(hp, egui::vec2(13.0, 13.0)),
                    anchor,
                    hx != 0,
                    hy != 0,
                ));
            }
        }
    }

    // --- points ------------------------------------------------------------
    // Free fn, not a closure: the gesture block below needs `st` mutable
    // while it is still hit-testing.
    fn hit_point(st: &State, r: &Rect, pts: &[[f32; 2]], p: Pos2) -> Option<usize> {
        (0..pts.len()).find(|i| st.pos(r, pts[*i]).distance(p) <= 8.0)
    }
    for i in 0..n {
        let hp = st.pos(&r, pts[i]);
        let selected = st.sel.contains(&i);
        let hot = pointer.is_some_and(|p| hp.distance(p) <= 8.0) && resp.hovered();
        let pinned =
            (cfg.pin_ends && (i == 0 || i + 1 == n)) || cfg.pin_y.contains(&i);
        let col = if selected {
            theme::curve_selected()
        } else if hot {
            theme::handle_hot()
        } else if pinned {
            theme::lift(theme::handle(), -72)
        } else {
            cfg.tint
        };
        // A dark rim so a point reads against a bright backdrop as well as
        // against the dark panel — the tint alone disappears on a luma ramp.
        clip.circle_filled(hp, if selected { 6.0 } else { 5.0 }, theme::handle_rim());
        clip.circle_filled(hp, if selected { 4.5 } else { 3.5 }, col);
        if selected {
            clip.circle_stroke(hp, 7.5, Stroke::new(1.0, col));
        }
    }

    // --- gestures ----------------------------------------------------------
    // Blender's hands (`crate::nav`): MMB / alt+LMB pan, ctrl+MMB and
    // ctrl+alt+LMB stretch the view per axis about the press point;
    // cmd/ctrl+LMB stays this canvas's old pan chord.
    let ctrlish = mods.command || mods.ctrl;
    let zoom_btn = (resp.dragged_by(egui::PointerButton::Middle) && ctrlish)
        || (resp.dragged_by(egui::PointerButton::Primary) && mods.alt && ctrlish)
        // The dolly every other canvas has (`nav::drag_nav`). A plain
        // right-CLICK still removes a point; only a modified right-DRAG zooms.
        || (resp.dragged_by(egui::PointerButton::Secondary) && (mods.alt || mods.command));
    let pan_btn = (resp.dragged_by(egui::PointerButton::Middle) && !ctrlish)
        || (resp.dragged_by(egui::PointerButton::Primary)
            && !zoom_btn
            && (ctrlish || mods.alt));
    if zoom_btn && !cfg.frozen {
        let d = resp.drag_delta();
        if let Some(a) = ui.input(|i| i.pointer.press_origin()) {
            let (fx, fy) = crate::nav::drag_zoom_xy(d);
            let (dx, dy) = (st.x_at(&r, a.x), st.y_at(&r, a.y));
            st.x_scale = (st.x_scale * fx).clamp(1.0, 1e6);
            st.y_scale = (st.y_scale * fy).clamp(1.0, 1e6);
            st.x_off = dx - (a.x - r.min.x) / st.x_scale;
            st.y_off = dy - (r.max.y - a.y) / st.y_scale;
        }
    }
    if resp.drag_started() {
        // The PRESS origin, not where the pointer has already got to: egui
        // only calls it a drag once it has moved, so `interact_pointer_pos`
        // here is a few pixels along — enough to miss the point you grabbed
        // and to eat the first part of the move.
        let p = ui
            .input(|i| i.pointer.press_origin())
            .or_else(|| resp.interact_pointer_pos())
            .unwrap_or_default();
        st.drag = if zoom_btn {
            // The stateless per-axis dolly above owns this drag; letting it
            // fall through would marquee (or grab a point) with ctrl+MMB.
            None
        } else if pan_btn {
            Some(Drag::Pan)
        } else if let Some((_, anchor, ax, ay)) =
            handles.iter().find(|(hr, ..)| hr.contains(p)).copied()
        {
            Some(Drag::Scale { origin: p, anchor, ax, ay, start: pts.clone() })
        } else if let Some(i) = hit_point(&st, &r, pts, p) {
            if mods.shift {
                // Shift EXTENDS a selection when starting a drag; it never
                // removes, or a shift-drag would deselect the very point it
                // grabbed and then move everything except it. Toggling is a
                // click-time behaviour and stays in the click handler.
                if !st.sel.contains(&i) {
                    st.sel.push(i);
                }
            } else if !st.sel.contains(&i) {
                st.sel = vec![i];
            }
            Some(Drag::Points { origin: p, start: pts.clone() })
        } else {
            Some(Drag::Marquee(p))
        };
    }
    if resp.dragged() {
        let p = pointer.unwrap_or_default();
        match st.drag.clone() {
            Some(Drag::Pan) => {
                let d = resp.drag_delta();
                st.x_off -= d.x / st.x_scale;
                st.y_off += d.y / st.y_scale;
            }
            Some(Drag::Marquee(a)) => {
                let m = Rect::from_two_pos(a, p);
                clip.rect_filled(m, 0.0, theme::marquee_fill());
                clip.rect_stroke(
                    m,
                    0.0,
                    Stroke::new(1.0, theme::marquee()),
                    egui::StrokeKind::Outside,
                );
                st.sel = (0..n).filter(|i| m.contains(st.pos(&r, pts[*i]))).collect();
            }
            Some(Drag::Points { origin, start }) => {
                let dx = (p.x - origin.x) / st.x_scale;
                let dy = -(p.y - origin.y) / st.y_scale;
                let moved: Vec<usize> = st.sel.clone();
                let mut next = start.clone();
                for i in moved.iter().copied() {
                    let pinned = (cfg.pin_ends && (i == 0 || i + 1 == start.len()))
                        || cfg.pin_y.contains(&i);
                    next[i][0] = start[i][0] + dx;
                    if !pinned {
                        next[i][1] = start[i][1] + dy;
                    }
                }
                mirror_x(cfg, &mut next, &start, &moved);
                if cfg.keep_order {
                    hold_order(pts, &mut next);
                }
                changed |= apply(pts, &mut st.sel, next, &moved, cfg);
            }
            Some(Drag::Scale { origin, anchor, ax, ay, start }) => {
                let (a_x, a_y) = anchor;
                let sx = if ax {
                    let (now, was) =
                        (st.x_at(&r, p.x) - a_x, st.x_at(&r, origin.x) - a_x);
                    if was.abs() > 1e-5 { now / was } else { 1.0 }
                } else {
                    1.0
                };
                let sy = if ay {
                    let (now, was) =
                        (st.y_at(&r, p.y) - a_y, st.y_at(&r, origin.y) - a_y);
                    if was.abs() > 1e-5 { now / was } else { 1.0 }
                } else {
                    1.0
                };
                let moved: Vec<usize> = st.sel.clone();
                let mut next = start.clone();
                for i in moved.iter().copied() {
                    let pinned = (cfg.pin_ends && (i == 0 || i + 1 == start.len()))
                        || cfg.pin_y.contains(&i);
                    next[i][0] = a_x + (start[i][0] - a_x) * sx;
                    if !pinned {
                        next[i][1] = a_y + (start[i][1] - a_y) * sy;
                    }
                }
                if cfg.keep_order {
                    hold_order(pts, &mut next);
                }
                changed |= apply(pts, &mut st.sel, next, &moved, cfg);
            }
            None => {}
        }
    }
    if resp.drag_stopped() {
        st.drag = None;
    }

    // --- click: add, remove, select ----------------------------------------
    if resp.clicked() || resp.secondary_clicked() {
        if let Some(p) = resp.interact_pointer_pos() {
            let hit = hit_point(&st, &r, pts, p);
            match hit {
                // Alt-click (or right-click, which does the same thing)
                // removes; the ends of a pinned curve are not deletable.
                Some(i) if mods.alt || resp.secondary_clicked() => {
                    let pinned = cfg.pin_ends && (i == 0 || i + 1 == n);
                    let mut kill: Vec<usize> =
                        if st.sel.contains(&i) { st.sel.clone() } else { vec![i] };
                    kill.retain(|k| {
                        !(cfg.pin_ends && (*k == 0 || *k + 1 == n)) && !cfg.pin_y.contains(k)
                    });
                    if !pinned && !kill.is_empty() {
                        changed |= remove(pts, &mut st.sel, kill, cfg);
                    }
                }
                Some(i) => {
                    if mods.shift {
                        if let Some(at) = st.sel.iter().position(|s| *s == i) {
                            st.sel.remove(at);
                        } else {
                            st.sel.push(i);
                        }
                    } else {
                        st.sel = vec![i];
                    }
                }
                None if mods.shift && cfg.add_ok.is_none_or(|f| f(st.x_at(&r, p.x))) => {
                    let mut v = [st.x_at(&r, p.x), st.y_at(&r, p.y)];
                    if let Some((lo, hi)) = cfg.clamp_y {
                        v[1] = v[1].clamp(lo, hi);
                    }
                    pts.push(v);
                    pts.sort_by(|a, b| {
                        a[0].partial_cmp(&b[0]).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    st.sel = pts.iter().position(|q| *q == v).map(|i| vec![i]).unwrap_or_default();
                    changed = true;
                }
                None => st.sel.clear(),
            }
        }
    }
    if resp.hovered()
        && ui.input(|i| i.key_pressed(egui::Key::Delete) || i.key_pressed(egui::Key::Backspace))
        && !st.sel.is_empty()
    {
        let mut kill = st.sel.clone();
        kill.retain(|k| {
            !(cfg.pin_ends && (*k == 0 || *k + 1 == n)) && !cfg.pin_y.contains(k)
        });
        changed |= remove(pts, &mut st.sel, kill, cfg);
    }

    // --- readout -----------------------------------------------------------
    let mut status = String::new();
    if st.sel.len() > 1 {
        status = format!("{} points", st.sel.len());
    }
    if let Some(p) = pointer.filter(|p| full.contains(*p)) {
        status = format!(
            "{status}   {} ➡ {}",
            fmt(st.x_at(&r, p.x), xstep),
            fmt(st.y_at(&r, p.y), ystep)
        );
    }
    if !status.is_empty() {
        clip.text(
            Pos2::new(full.max.x - 5.0, full.min.y + 2.0),
            egui::Align2::RIGHT_TOP,
            status,
            egui::FontId::proportional(10.0),
            theme::text(),
        );
    }

    let screen = pts.iter().map(|p| st.pos(&r, *p)).collect();
    st.store(ui.ctx(), id);
    CurveOut { changed, response: resp, screen }
}

/// The wheel, trackpad and pinch over `rect` belong to the canvas drawing
/// it this frame: the scroll area around it stops scrolling now, and a
/// container that reads them before drawing stands down next frame
/// (`owns_pointer`). Call while hovered.
pub fn claim_wheel(ui: &egui::Ui, rect: Rect) {
    ui.input_mut(|i| i.smooth_scroll_delta = Vec2::ZERO);
    // Read the frame number BEFORE taking the data lock: egui's context
    // lock is not re-entrant, and asking for it inside `data_mut` froze
    // the app the moment a band plot was hovered.
    let nr = ui.ctx().cumulative_frame_nr();
    ui.ctx().data_mut(|d| d.insert_temp(claim_id(), (nr, rect)));
}

fn claim_id() -> egui::Id {
    egui::Id::new("curveui-wheel-claim")
}

/// A curve canvas was under the pointer last frame and still is: the
/// wheel, trackpad and pinch are its navigation, so a container that reads
/// them BEFORE the canvas draws (the properties grid's zoom) must stand
/// down. One frame stale by construction, which a hover cannot notice.
pub fn owns_pointer(ctx: &egui::Context) -> bool {
    let Some((frame, rect)) = ctx.data(|d| d.get_temp::<(u64, Rect)>(claim_id())) else {
        return false;
    };
    ctx.cumulative_frame_nr() <= frame + 1
        && ctx.pointer_latest_pos().is_some_and(|p| rect.contains(p))
}

fn fmt(v: f32, step: f32) -> String {
    let dp = if step >= 1.0 { 0 } else { (-step.log10().floor()) as usize };
    format!("{v:.dp$}")
}

/// Write moved points back, keeping them sorted by x and following the
/// selection through the re-sort.
/// Move a group RIGIDLY in x: partners take the same delta as the member the
/// gesture grabbed, so the offsets inside the group survive.
///
/// It has to be a delta and not a shared value. Forcing partners onto the
/// moved point's x collapses any group whose members are at DIFFERENT x —
/// which is every crossfade, whose two corners are a width apart — into a
/// vertical drop the moment you touch it. Two points that genuinely are one
/// place have offset zero and are unaffected by the distinction.
///
/// The moved points keep their own y, and partners are deliberately not
/// added to `moved`: they follow the gesture, they are not part of the
/// selection it grabbed.
pub(crate) fn mirror_x(
    cfg: &CurveCfg<'_>,
    next: &mut [[f32; 2]],
    start: &[[f32; 2]],
    moved: &[usize],
) {
    for g in cfg.mirror {
        let Some(src) = g.iter().find(|i| moved.contains(i)) else { continue };
        let (Some(a), Some(b)) = (next.get(*src), start.get(*src)) else { continue };
        let dx = a[0] - b[0];
        for j in g.iter().copied() {
            if j < next.len() && j < start.len() && !moved.contains(&j) {
                next[j][0] = start[j][0] + dx;
            }
        }
    }
}

/// Closest two neighbours may come in x under [`CurveCfg::keep_order`].
const ORDER_GAP: f32 = 1e-4;

/// Stop points at their neighbours rather than let them pass
/// ([`CurveCfg::keep_order`]). `now` is the curve as it stands (sorted),
/// `next` where the gesture wants it, index for index. Every x travels the
/// same FRACTION of the way from `now` to `next` — the largest at which no
/// two neighbours close past [`ORDER_GAP`] — so a dragged point, or a
/// rigidly moving group, stops against the first point in its way, and a
/// scale box stops when any member meets one. Y is left alone: a point
/// stopped against a wall still follows the hand up and down.
///
/// Fractional rather than per-point so a group keeps its own spacing; and
/// measured from where the points ARE, not where the drag began, so a fast
/// flick lands against the neighbour instead of short of it.
pub(crate) fn hold_order(now: &[[f32; 2]], next: &mut [[f32; 2]]) {
    let n = now.len().min(next.len());
    let mut u = 1.0f32;
    for i in 0..n.saturating_sub(1) {
        let g0 = now[i + 1][0] - now[i][0];
        let g1 = next[i + 1][0] - next[i][0];
        // Pairs already closer than the gap (a hard boundary's corners)
        // may stay where they are, never close further.
        let floor = g0.min(ORDER_GAP);
        if g1 >= floor {
            continue;
        }
        let f = if g0 - g1 > 1e-12 { ((g0 - floor) / (g0 - g1)).clamp(0.0, 1.0) } else { 0.0 };
        u = u.min(f);
    }
    if u < 1.0 {
        for i in 0..n {
            next[i][0] = now[i][0] + (next[i][0] - now[i][0]) * u;
        }
    }
}

fn apply(
    pts: &mut Vec<[f32; 2]>,
    sel: &mut Vec<usize>,
    next: Vec<[f32; 2]>,
    moved: &[usize],
    cfg: &CurveCfg<'_>,
) -> bool {
    let mut tagged: Vec<([f32; 2], bool)> = next
        .into_iter()
        .enumerate()
        .map(|(i, mut p)| {
            if let Some((lo, hi)) = cfg.clamp_y {
                p[1] = p[1].clamp(lo, hi);
            }
            (p, moved.contains(&i))
        })
        .collect();
    tagged.sort_by(|a, b| a.0[0].partial_cmp(&b.0[0]).unwrap_or(std::cmp::Ordering::Equal));
    let now: Vec<[f32; 2]> = tagged.iter().map(|(p, _)| *p).collect();
    if now == *pts {
        return false;
    }
    *sel = tagged.iter().enumerate().filter(|(_, (_, m))| *m).map(|(i, _)| i).collect();
    *pts = now;
    true
}

fn remove(
    pts: &mut Vec<[f32; 2]>,
    sel: &mut Vec<usize>,
    mut kill: Vec<usize>,
    cfg: &CurveCfg,
) -> bool {
    kill.sort_unstable();
    kill.dedup();
    // Never below the floor: drop the extras from the end of the list.
    while pts.len().saturating_sub(kill.len()) < cfg.min_points && !kill.is_empty() {
        kill.pop();
    }
    if kill.is_empty() {
        return false;
    }
    let mut i = 0;
    pts.retain(|_| {
        let keep = !kill.contains(&i);
        i += 1;
        keep
    });
    sel.clear();
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_smooth_points_are_a_smoothstep() {
        let p = [[0.2, 0.0], [0.8, 1.0]];
        for i in 0..=20 {
            let x = 0.2 + i as f32 / 20.0 * 0.6;
            let t = ((x - 0.2) / 0.6).clamp(0.0, 1.0);
            let want = t * t * (3.0 - 2.0 * t);
            let got = eval(&p, x, CurveInterp::Smooth, true);
            assert!((got - want).abs() < 1e-5, "x={x} got {got} want {want}");
        }
    }

    #[test]
    fn linear_is_linear() {
        let p = [[0.0, 0.0], [1.0, 1.0]];
        assert!((eval(&p, 0.25, CurveInterp::Linear, true) - 0.25).abs() < 1e-6);
        // The same two points read as a smoothstep when smoothed.
        assert!(eval(&p, 0.25, CurveInterp::Smooth, true) < 0.2);
    }

    /// The reported bug: a colour curve must not flatten past its last point,
    /// or every value above it is crushed to the same output.
    #[test]
    fn the_default_curve_extrapolates_past_its_ends() {
        let p = [[0.0, 0.0], [1.0, 1.0]];
        for interp in CurveInterp::ALL {
            for flat in [true, false] {
                assert!((eval(&p, 4.0, interp, flat) - 4.0).abs() < 1e-4, "{interp:?} clamped at the top");
                assert!((eval(&p, -2.0, interp, flat) + 2.0).abs() < 1e-4, "{interp:?} clamped at the bottom");
            }
        }
        // A curve whose last segment climbs steeply keeps climbing.
        let steep = [[0.0, 0.0], [0.5, 0.5], [1.0, 2.0]];
        assert!(eval(&steep, 2.0, CurveInterp::Linear, false) > 4.0);
    }

    /// The reported bug: a smoothed colour curve flattened INTO its last
    /// point (a flat end tangent) and then carried on along the end
    /// segment's slope — a kink right after the point, the curve rising
    /// again from a plateau. Its ends must meet the extrapolation smoothly.
    #[test]
    fn a_colour_curve_meets_its_extrapolation_without_a_kink() {
        let p = [[0.0, 0.0], [0.4, 0.5], [1.0, 0.6]];
        let slope = |x: f32, h: f32| {
            (eval(&p, x + h, CurveInterp::Smooth, false) - eval(&p, x - h, CurveInterp::Smooth, false)) / (2.0 * h)
        };
        let h = 1e-3;
        for end in [0.0f32, 1.0] {
            let inside = slope(end + if end == 0.0 { h } else { -h }, h);
            let outside = slope(end + if end == 0.0 { -2.0 * h } else { 2.0 * h }, h);
            assert!((inside - outside).abs() < 0.02, "kink at x={end}: {inside} inside vs {outside} outside");
        }
        // Monotone points stay monotone right up to the end (no dip, no hump).
        let mut prev = eval(&p, 0.0, CurveInterp::Smooth, false);
        for i in 1..=200 {
            let y = eval(&p, i as f32 / 200.0, CurveInterp::Smooth, false);
            assert!(y >= prev - 1e-6, "dips at {}", i as f32 / 200.0);
            prev = y;
        }
        // And the default ramp, smoothed, is the straight line it looks like.
        let ramp = [[0.0, 0.0], [1.0, 1.0]];
        for i in 0..=10 {
            let x = i as f32 / 10.0;
            assert!((eval(&ramp, x, CurveInterp::Smooth, false) - x).abs() < 1e-5);
        }
    }

    #[test]
    fn smooth_never_overshoots_between_its_points() {
        let p = [[0.0, 0.0], [0.4, 0.0], [0.5, 1.0], [1.0, 1.0]];
        for i in 0..=200 {
            let x = i as f32 / 200.0;
            let y = eval(&p, x, CurveInterp::Smooth, true);
            assert!((-1e-5..=1.0 + 1e-5).contains(&y), "x={x} y={y}");
        }
    }

    #[test]
    fn passes_through_every_point() {
        let p = [[0.0, 0.1], [0.3, 0.7], [0.6, 0.2], [1.0, 0.9]];
        for q in p {
            for interp in CurveInterp::ALL {
                assert!((eval(&p, q[0], interp, true) - q[1]).abs() < 1e-5);
                assert!((eval(&p, q[0], interp, false) - q[1]).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn a_dragged_point_keeps_its_place_in_the_selection() {
        // Drag the middle point past the last one: the list re-sorts and the
        // selection must follow it, not the index it used to sit at.
        let mut pts = vec![[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]];
        let mut sel = vec![1usize];
        let next = vec![[0.0, 0.0], [1.5, 0.9], [1.0, 1.0]];
        assert!(apply(&mut pts, &mut sel, next, &[1], &CurveCfg::default()));
        assert_eq!(sel, vec![2], "the moved point is now last");
        assert_eq!(pts[2], [1.5, 0.9]);
    }

    /// `keep_order`: a point dragged past its neighbour stops against it,
    /// keeps following the hand in y, and the order never changes.
    #[test]
    fn a_held_point_stops_at_its_neighbour() {
        let now = vec![[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]];
        let mut next = vec![[0.0, 0.0], [1.5, 0.9], [1.0, 1.0]];
        hold_order(&now, &mut next);
        assert!(next[1][0] < next[2][0], "it passed: {next:?}");
        assert!((next[2][0] - next[1][0] - ORDER_GAP).abs() < 1e-5, "stopped short: {next:?}");
        assert_eq!(next[1][1], 0.9, "y stopped following the hand");
        // And the other way, against the first point.
        let mut back = vec![[0.0, 0.0], [-3.0, 0.2], [1.0, 1.0]];
        hold_order(&now, &mut back);
        assert!(back[1][0] > back[0][0], "{back:?}");
        // A move that crosses nothing is untouched.
        let mut free = vec![[0.0, 0.0], [0.7, 0.5], [1.0, 1.0]];
        hold_order(&now, &mut free);
        assert_eq!(free[1][0], 0.7);
    }

    /// A rigid group (a crossfade dragged whole) stops as ONE: its spacing
    /// survives the stop.
    #[test]
    fn a_held_group_keeps_its_spacing() {
        let now = vec![[0.0, 1.0], [0.4, 0.0], [0.6, 1.0], [1.0, 1.0]];
        let mut next = now.clone();
        next[1][0] += 0.9;
        next[2][0] += 0.9;
        hold_order(&now, &mut next);
        assert!((next[2][0] - next[1][0] - 0.2).abs() < 1e-5, "{next:?}");
        assert!(next[2][0] < next[3][0]);
    }

    /// Two corners of a hard cut sit closer than the gap; they may stay, and
    /// moving something else is not blocked by them.
    #[test]
    fn a_hard_cut_does_not_jam_the_curve() {
        let now = vec![[0.0, 1.0], [0.5, 1.0], [0.50001, 0.0], [1.0, 0.0]];
        let mut next = now.clone();
        next[3][0] = 0.8;
        hold_order(&now, &mut next);
        assert_eq!(next[3][0], 0.8);
    }

    #[test]
    fn q_frames_the_picked_points_or_all_of_them() {
        let r = Rect::from_min_size(Pos2::ZERO, egui::vec2(200.0, 100.0));
        let pts = vec![[0.0, 0.0], [0.2, 0.5], [0.4, 0.9], [1.0, 1.0]];
        let inside = |st: &State, p: [f32; 2]| r.expand(0.01).contains(st.pos(&r, p));
        // Picked: those two fill the plot, the others fall outside it.
        let mut st = State { sel: vec![1, 2], ..Default::default() };
        st.frame(&r, [0.0, 1.0, 0.0, 1.0]);
        st.frame_points(&r, &pts);
        assert!(inside(&st, pts[1]) && inside(&st, pts[2]));
        assert!(!inside(&st, pts[0]) && !inside(&st, pts[3]));
        assert!((st.x_of(&r, 0.3) - r.center().x).abs() < 0.01);
        // None picked: every point.
        st.sel.clear();
        st.frame_points(&r, &pts);
        assert!(pts.iter().all(|&p| inside(&st, p)));
        // One point: centred, zoom kept.
        let before = (st.x_scale, st.y_scale);
        st.sel = vec![1];
        st.frame_points(&r, &pts);
        assert_eq!((st.x_scale, st.y_scale), before);
        assert!((st.pos(&r, pts[1]) - r.center()).length() < 0.01);
    }

    #[test]
    fn removal_respects_the_floor() {
        let cfg = CurveCfg { min_points: 2, ..Default::default() };
        let mut pts = vec![[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]];
        let mut sel = vec![0, 1, 2];
        assert!(remove(&mut pts, &mut sel, vec![0, 1, 2], &cfg));
        assert_eq!(pts.len(), 2, "a curve is never left with fewer than two points");
    }
}
