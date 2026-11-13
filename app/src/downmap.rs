//! The map from a node's own output pixels to the VIEWED node's pixels,
//! when something downstream is not a straight-line map.
//!
//! `downstream_proj` composes every Transform / CornerPin / Reformat
//! between a node and the viewer into one homography, and that is all a
//! gizmo needs while every step keeps lines straight. A UV Warp does not:
//! each output pixel is the source sampled where a PICTURE says, so the
//! step is data, not a matrix. Here the chain is a list of steps — runs
//! of homographies with UV Warps between them — and a point goes forward
//! (node -> viewer, for drawing) and back (pointer -> node, for dragging)
//! through it.
//!
//! Back through a UV Warp is the easy direction: the map is sampled where
//! the pointer is, exactly as the shader does. Forward is the warp's
//! INVERSE, solved per point: the nearest of a coarse lattice of the map,
//! then Newton on the full-resolution map. Where the map folds (two output
//! pixels show the same source pixel) the solve lands on one of them; off
//! the map's picture it extrapolates the edge linearly.
//!
//! The map's pixels come from one render + readback of its input
//! (`Gfx::refresh_uv_grids`), cached on the map's subtree key — so a
//! still map is read once, not per frame.

use std::sync::Arc;

use crate::{H3, IDENTITY_H};

/// A UV Warp's map read back to the CPU, as SOURCE pixels per map pixel
/// (the lookup already applied: V orientation, the source's format).
pub(crate) struct UvGrid {
    w: usize,
    h: usize,
    /// Row-major, top row first, the source pixel each map pixel samples.
    q: Vec<[f32; 2]>,
    /// Every `stride`-th pixel, for the first guess of an inverse.
    coarse: Vec<([f32; 2], [f32; 2])>,
}

impl UvGrid {
    /// From the map's red/green (U, V normalised over the source's
    /// format), as `UvWarpParams::lookup` reads them.
    pub(crate) fn new(
        w: usize,
        h: usize,
        uv: &[[f32; 2]],
        p: &engine::graph::UvWarpParams,
        src: [f32; 2],
    ) -> Option<Self> {
        if w == 0 || h == 0 || uv.len() != w * h {
            return None;
        }
        // Bilinear lookup only: the nearest snap is a sampling choice, and a
        // handle is a continuous position.
        let lp = engine::graph::UvWarpParams { filter: Default::default(), ..*p };
        let q: Vec<[f32; 2]> = uv
            .iter()
            .map(|c| {
                let c = if c[0].is_finite() && c[1].is_finite() { *c } else { [0.0, 0.0] };
                lp.lookup(c, src)
            })
            .collect();
        let stride = (w.max(h) / 96).max(1);
        let mut coarse = Vec::new();
        for y in (stride / 2..h).step_by(stride) {
            for x in (stride / 2..w).step_by(stride) {
                coarse.push(([x as f32 + 0.5, y as f32 + 0.5], q[y * w + x]));
            }
        }
        Some(Self { w, h, q, coarse })
    }

    fn size(&self) -> [f32; 2] {
        [self.w as f32, self.h as f32]
    }

    fn texel(&self, x: usize, y: usize) -> [f32; 2] {
        self.q[y.min(self.h - 1) * self.w + x.min(self.w - 1)]
    }

    /// Bilinear inside the pixel centres.
    fn bilerp(&self, o: [f32; 2]) -> [f32; 2] {
        let fx = (o[0] - 0.5).clamp(0.0, (self.w - 1) as f32);
        let fy = (o[1] - 0.5).clamp(0.0, (self.h - 1) as f32);
        let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
        let (tx, ty) = (fx - x0 as f32, fy - y0 as f32);
        let (a, b) = (self.texel(x0, y0), self.texel(x0 + 1, y0));
        let (c, d) = (self.texel(x0, y0 + 1), self.texel(x0 + 1, y0 + 1));
        let l = |a: f32, b: f32, t: f32| a + (b - a) * t;
        [
            l(l(a[0], b[0], tx), l(c[0], d[0], tx), ty),
            l(l(a[1], b[1], tx), l(c[1], d[1], tx), ty),
        ]
    }

    /// The source pixel map pixel `o` samples (map pixels, centres at
    /// +0.5). Past the edge centres the edge's slope carries on, so a
    /// handle off the picture still moves smoothly and the solve below
    /// still has a gradient.
    pub(crate) fn sample(&self, o: [f32; 2]) -> [f32; 2] {
        let lo = [0.5, 0.5];
        let hi = [self.w as f32 - 0.5, self.h as f32 - 0.5];
        let oc = [o[0].clamp(lo[0], hi[0]), o[1].clamp(lo[1], hi[1])];
        let v = self.bilerp(oc);
        let d = [o[0] - oc[0], o[1] - oc[1]];
        if d == [0.0, 0.0] {
            return v;
        }
        // One-sided slope, stepping INTO the map.
        let sx = if oc[0] > lo[0] + 1.0 { -1.0 } else { 1.0 };
        let sy = if oc[1] > lo[1] + 1.0 { -1.0 } else { 1.0 };
        let vx = self.bilerp([oc[0] + sx, oc[1]]);
        let vy = self.bilerp([oc[0], oc[1] + sy]);
        let jx = [(vx[0] - v[0]) * sx, (vx[1] - v[1]) * sx];
        let jy = [(vy[0] - v[0]) * sy, (vy[1] - v[1]) * sy];
        [v[0] + jx[0] * d[0] + jy[0] * d[1], v[1] + jx[1] * d[0] + jy[1] * d[1]]
    }

    /// The map pixel that samples source pixel `s`: the warp's inverse.
    pub(crate) fn solve(&self, s: [f32; 2]) -> [f32; 2] {
        let d2 = |a: [f32; 2]| (a[0] - s[0]).powi(2) + (a[1] - s[1]).powi(2);
        let mut o = self
            .coarse
            .iter()
            .min_by(|a, b| d2(a.1).total_cmp(&d2(b.1)))
            .map(|c| c.0)
            .unwrap_or([self.w as f32 * 0.5, self.h as f32 * 0.5]);
        let mut f = self.sample(o);
        let mut best = (d2(f), o);
        // Newton, finite-difference Jacobian, steps capped so a fold or a
        // flat patch cannot throw the point across the frame.
        let cap = self.w.max(self.h) as f32 * 0.25;
        for _ in 0..24 {
            if best.0 < 1e-6 {
                break;
            }
            let e = 0.5;
            let fx = self.sample([o[0] + e, o[1]]);
            let fy = self.sample([o[0], o[1] + e]);
            let (a, c) = ((fx[0] - f[0]) / e, (fx[1] - f[1]) / e);
            let (b, dd) = ((fy[0] - f[0]) / e, (fy[1] - f[1]) / e);
            let det = a * dd - b * c;
            if det.abs() < 1e-9 {
                break;
            }
            let (rx, ry) = (s[0] - f[0], s[1] - f[1]);
            let mut step = [(dd * rx - b * ry) / det, (-c * rx + a * ry) / det];
            let len = (step[0] * step[0] + step[1] * step[1]).sqrt();
            if len > cap {
                step = [step[0] * cap / len, step[1] * cap / len];
            }
            o = [o[0] + step[0], o[1] + step[1]];
            f = self.sample(o);
            let e2 = d2(f);
            if e2 < best.0 {
                best = (e2, o);
            }
        }
        best.1
    }
}

/// One step of the chain, in its own pixels.
#[derive(Clone)]
pub(crate) enum Step {
    H(H3),
    /// A UV Warp: `out` is its own format, `src` its source's (what the
    /// node's params are in), and the grid may be read at another size.
    Uv { grid: Arc<UvGrid>, out: [f32; 2] },
}

/// Node pixels -> viewed pixels, step by step (first step nearest the node).
#[derive(Clone)]
pub(crate) struct DownMap {
    steps: Vec<Step>,
}

impl DownMap {
    fn push_h(&mut self, h: H3) {
        if h == IDENTITY_H {
            return;
        }
        if let Some(Step::H(prev)) = self.steps.last_mut() {
            // Applied after the previous one.
            *prev = crate::h_mul(&h, prev);
        } else {
            self.steps.push(Step::H(h));
        }
    }

    pub(crate) fn fwd(&self, mut p: egui::Vec2) -> egui::Vec2 {
        for s in &self.steps {
            p = match s {
                Step::H(h) => crate::apply_h(h, p),
                Step::Uv { grid, out } => {
                    let k = [grid.size()[0] / out[0], grid.size()[1] / out[1]];
                    let o = grid.solve([p.x, p.y]);
                    egui::vec2(o[0] / k[0], o[1] / k[1])
                }
            };
        }
        p
    }

    pub(crate) fn inv(&self, mut p: egui::Vec2) -> egui::Vec2 {
        for s in self.steps.iter().rev() {
            p = match s {
                Step::H(h) => crate::apply_h(&engine::codegen::mat3_inv(h), p),
                Step::Uv { grid, out } => {
                    let k = [grid.size()[0] / out[0], grid.size()[1] / out[1]];
                    let q = grid.sample([p.x * k[0], p.y * k[1]]);
                    egui::vec2(q[0], q[1])
                }
            };
        }
        p
    }
}

/// `downstream_proj`, with UV Warps in it. `uv(id)` answers a UV Warp's
/// read-back map, or `None` (not read yet, no map wired: the warp is
/// skipped, as `downstream_proj` always skips it). Only a path through
/// the warp's SOURCE (input 0) bends; one through its map or mask does not
/// carry the node's pixels anywhere.
pub(crate) fn downstream_map(
    g: &crate::editor::EditorGraph,
    target: u64,
    res: &dyn Fn(u64) -> (f32, f32),
    t: u64,
    uv: &dyn Fn(u64) -> Option<Arc<UvGrid>>,
) -> DownMap {
    // The path, viewer first: (node, the input the walk left it through).
    #[allow(clippy::too_many_arguments)]
    fn dfs(
        g: &crate::editor::EditorGraph,
        cur: u64,
        out: usize,
        target: u64,
        visited: &mut std::collections::HashSet<(u64, usize)>,
        path: &mut Vec<(u64, usize)>,
    ) -> bool {
        if cur == target {
            return true;
        }
        if !visited.insert((cur, out)) {
            return false;
        }
        let Some(n) = g.node(cur) else { return false };
        for i in 0..n.kind.input_count() {
            if !crate::feeds_output(&n.kind, i, out) {
                continue;
            }
            if let Some(srcn) = g.input_of(cur, i) {
                path.push((cur, i));
                if dfs(g, srcn, g.out_of(cur, i) as usize, target, visited, path) {
                    return true;
                }
                path.pop();
            }
        }
        false
    }
    let mut m = DownMap { steps: Vec::new() };
    let Some(viewer) = g.active_view() else { return m };
    let mut path = Vec::new();
    let mut visited = std::collections::HashSet::new();
    if !dfs(g, viewer, g.active_view_out() as usize, target, &mut visited, &mut path) {
        return m;
    }
    // Nearest the node first.
    for &(id, input) in path.iter().rev() {
        let warp = matches!(
            g.node(id),
            Some(n) if !n.disabled && matches!(n.kind, engine::graph::NodeKind::UvWarp(_))
        );
        if warp {
            if input == 0 {
                if let Some(grid) = uv(id) {
                    let (w, h) = res(id);
                    m.steps.push(Step::Uv { grid, out: [w, h] });
                }
            }
            continue;
        }
        m.push_h(crate::node_step_h(g, id, res, t));
    }
    m
}

/// The UV Warps a gizmo on `target` is seen through — the ones whose maps
/// are worth reading back.
pub(crate) fn warps_between(
    g: &crate::editor::EditorGraph,
    target: u64,
    res: &dyn Fn(u64) -> (f32, f32),
    t: u64,
) -> Vec<u64> {
    let asked = std::cell::RefCell::new(Vec::new());
    downstream_map(g, target, res, t, &|id| {
        if g.input_of(id, 1).is_some() {
            asked.borrow_mut().push(id);
        }
        None
    });
    asked.into_inner()
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::graph::UvWarpParams;

    /// A smooth swirl-ish map: forward then back lands where it started.
    fn swirl(w: usize, h: usize) -> (Vec<[f32; 2]>, UvWarpParams) {
        let p = UvWarpParams { v_up: false, ..Default::default() };
        let mut uv = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                let (u, v) = ((x as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32);
                let du = 0.04 * (v * 6.0).sin();
                let dv = 0.03 * (u * 5.0).cos();
                uv.push([u + du, v + dv]);
            }
        }
        (uv, p)
    }

    #[test]
    fn the_solve_inverts_the_sample() {
        let (w, h) = (160, 90);
        let (uv, p) = swirl(w, h);
        let g = UvGrid::new(w, h, &uv, &p, [320.0, 180.0]).unwrap();
        for s in [[160.0, 90.0], [40.0, 30.0], [300.0, 170.0], [10.0, 150.0]] {
            let o = g.solve(s);
            let back = g.sample(o);
            assert!((back[0] - s[0]).abs() < 0.01 && (back[1] - s[1]).abs() < 0.01, "{s:?} {o:?} {back:?}");
        }
    }

    /// Off the map's picture the edge carries on, and the inverse still
    /// round-trips.
    #[test]
    fn off_the_picture_extrapolates() {
        let (w, h) = (64, 36);
        let (uv, p) = swirl(w, h);
        let g = UvGrid::new(w, h, &uv, &p, [64.0, 36.0]).unwrap();
        for s in [[-20.0, 18.0], [80.0, -10.0]] {
            let back = g.sample(g.solve(s));
            assert!((back[0] - s[0]).abs() < 0.05 && (back[1] - s[1]).abs() < 0.05, "{s:?} {back:?}");
        }
    }

    /// The identity map (V up, the standard) is the identity, and a map
    /// read at half size still answers in the node's pixels.
    #[test]
    fn an_identity_map_at_half_size_maps_nothing() {
        let (w, h) = (50, 30);
        let mut uv = Vec::new();
        for y in 0..h {
            for x in 0..w {
                uv.push([(x as f32 + 0.5) / w as f32, 1.0 - (y as f32 + 0.5) / h as f32]);
            }
        }
        let grid = Arc::new(UvGrid::new(w, h, &uv, &UvWarpParams::default(), [100.0, 60.0]).unwrap());
        let m = DownMap { steps: vec![Step::Uv { grid, out: [100.0, 60.0] }] };
        for p in [egui::vec2(10.0, 7.0), egui::vec2(77.0, 41.0)] {
            let f = m.fwd(p);
            let b = m.inv(p);
            assert!((f - p).length() < 0.05 && (b - p).length() < 0.05, "{p:?} {f:?} {b:?}");
        }
    }
}
