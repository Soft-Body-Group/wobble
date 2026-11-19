//! 2dini water: the Physics node's FLUID mode. Ported from graav's CPU fluid
//! (`graav_core::fluid`): position-based fluids (Macklin & Muller 2013), a
//! counting-sort spatial hash, a neighbour list per particle per step, and
//! XSPH viscosity. Every per-particle pass reads the state as it stood, so
//! rayon may run them in any order with the same answer: a bake is
//! deterministic, like the rigid and soft ones.
//!
//! What changed on the way over: graav's gravity points at a planet's
//! centre, here it points down the canvas; graav's terrain distance field is
//! here one built from the Physics node's colliders (closed shapes solid,
//! open lines walls, points discs) plus the canvas walls; sleeping pools,
//! pushers and the recycle shell are gone (a bake is offline).
//!
//! Units. The solver runs in graav's own units, where the particle spacing is
//! `SOLVER_SPACING` and the smoothing radius `H`: its tuned constants carry
//! over unchanged and the look does not depend on the canvas. Geo space is
//! scaled by `k = SOLVER_SPACING / spacing` on the way in and back out.
//!
//! A bake stores one body per water volume (a closed shape on the bodies
//! input), its particles' positions per frame. `draw` turns a frame into
//! geometry: a smooth surface (marching squares over a metaball field), the
//! particles as dots, or both. The look is read at playback, so changing it
//! never makes the bake stale.

use rayon::prelude::*;

use super::{BakeMsg, BakedBody, BodyIn, BodyShape, PhysicsBake, PhysicsSettings, Statics, Track, METRES_PER_WIDTH};
use crate::geo::{FillRule, Geometry, Point, Prim, PrimKind, Style};

/// The most particles a bake keeps, and the most particle-frames: past
/// either the spacing is coarsened until it fits (scale down, never refuse).
pub const MAX_PARTICLES: usize = 20_000;
const MAX_PARTICLE_FRAMES: usize = 6_000_000;

/// graav's particle spacing and smoothing radius: its constants are tuned
/// for these.
const SOLVER_SPACING: f32 = 0.135;
const H: f32 = 0.35;
/// How far past a solid's surface a particle is put when pushed out of it.
/// graav's 0.01 turned into an upward kick ten times gravity's on every
/// substep a particle rested on something (the push becomes velocity), and
/// a dam break climbed the canvas.
const SKIN: f32 = 1e-4;

/// How water looks when played back.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FluidLook {
    /// A smooth filled surface traced round the particles.
    #[default]
    Surface,
    /// Each particle a dot.
    Particles,
    Both,
}

impl FluidLook {
    pub const ALL: [FluidLook; 3] = [FluidLook::Surface, FluidLook::Particles, FluidLook::Both];
    pub fn name(self) -> &'static str {
        match self {
            FluidLook::Surface => "surface",
            FluidLook::Particles => "particles",
            FluidLook::Both => "both",
        }
    }
}

/// The fluid's own settings, inside `PhysicsSettings`. `spacing`,
/// `viscosity` and `iterations` shape the simulation (they are in the bake's
/// input hash); `look`, `blob` and `dot` only shape the drawing.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct FluidSettings {
    /// Particle spacing, geo units (canvas widths). Smaller is finer and
    /// slower: the count goes as 1 / spacing².
    pub spacing: f32,
    /// XSPH: how far each particle's velocity is drawn to its neighbours'.
    /// 0 splashy water, 1 thick honey.
    pub viscosity: f32,
    /// Density corrections a substep. More is less squashy.
    pub iterations: u32,
    pub look: FluidLook,
    /// The surface's reach round each particle, in spacings.
    pub blob: f32,
    /// A particle dot's radius, in half-spacings.
    pub dot: f32,
}

impl Default for FluidSettings {
    fn default() -> Self {
        Self { spacing: 0.006, viscosity: 0.1, iterations: 3, look: FluidLook::Surface, blob: 1.8, dot: 1.0 }
    }
}

impl FluidSettings {
    /// What goes into the bake's input hash.
    pub(super) fn hash_bytes(&self) -> Vec<u8> {
        let mut b = vec![b'F'];
        for f in [self.spacing, self.viscosity] {
            b.extend_from_slice(&f.to_le_bytes());
        }
        b.extend_from_slice(&self.iterations.to_le_bytes());
        b
    }
}

type V = [f32; 2];

// ---------------------------------------------------------------------------
// Kernels (graav's `kernels.rs`): each adds up to 1 over its disc.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct Kernels {
    h: f32,
    poly6: f32,
    spiky2: f32,
    spiky2_d: f32,
}

impl Kernels {
    fn new(h: f32) -> Self {
        use std::f32::consts::PI;
        Self { h, poly6: 4.0 / (PI * h.powi(8)), spiky2: 6.0 / (PI * h.powi(4)), spiky2_d: 12.0 / (PI * h.powi(4)) }
    }
    fn poly6(&self, d: f32) -> f32 {
        if d < self.h {
            let v = self.h * self.h - d * d;
            v * v * v * self.poly6
        } else {
            0.0
        }
    }
    fn spiky2(&self, d: f32) -> f32 {
        if d < self.h {
            let v = self.h - d;
            v * v * self.spiky2
        } else {
            0.0
        }
    }
    fn spiky2_d(&self, d: f32) -> f32 {
        if d <= self.h {
            -(self.h - d) * self.spiky2_d
        } else {
            0.0
        }
    }
}

/// The density a particle has inside a still lattice at the solver spacing:
/// the fluid's rest density, so the spawn neither explodes nor collapses.
fn rest_density(k: &Kernels) -> f32 {
    let reach = (k.h / SOLVER_SPACING).ceil() as i32 + 1;
    let mut sum = 0.0;
    for j in -reach..=reach {
        for i in -reach..=reach {
            sum += k.spiky2((i as f32 * SOLVER_SPACING).hypot(j as f32 * SOLVER_SPACING));
        }
    }
    sum
}

// ---------------------------------------------------------------------------
// The spatial hash (graav's `hash.rs`): a counting sort by cell, stable.
// ---------------------------------------------------------------------------

fn cell_of(p: V, h: f32) -> [i32; 2] {
    [(p[0] / h).floor() as i32, (p[1] / h).floor() as i32]
}

fn key_of(cell: [i32; 2], n: u32) -> u32 {
    ((cell[0] as u32).wrapping_mul(15823).wrapping_add((cell[1] as u32).wrapping_mul(9737333))) % n
}

#[derive(Default)]
struct Hash {
    h: f32,
    keys: Vec<u32>,
    starts: Vec<u32>,
    order: Vec<u32>,
    unsorted: Vec<u32>,
}

impl Hash {
    fn build(&mut self, at: &[V], h: f32) -> &[u32] {
        let n = at.len();
        self.h = h;
        self.unsorted.clear();
        self.unsorted.extend(at.iter().map(|p| key_of(cell_of(*p, h), n as u32)));
        self.starts.clear();
        self.starts.resize(n + 1, 0);
        for &key in &self.unsorted {
            self.starts[key as usize + 1] += 1;
        }
        for k in 0..n {
            self.starts[k + 1] += self.starts[k];
        }
        let mut next = self.starts.clone();
        self.order.resize(n, 0);
        self.keys.resize(n, 0);
        for (i, &key) in self.unsorted.iter().enumerate() {
            let place = next[key as usize] as usize;
            next[key as usize] += 1;
            self.order[place] = i as u32;
            self.keys[place] = key;
        }
        &self.order
    }

    #[inline]
    fn near(&self, p: V, mut visit: impl FnMut(usize)) {
        let n = self.keys.len();
        if n == 0 {
            return;
        }
        let cell = cell_of(p, self.h);
        let mut seen = [u32::MAX; 9];
        let mut count = 0;
        for dy in -1..=1 {
            for dx in -1..=1 {
                let key = key_of([cell[0] + dx, cell[1] + dy], n as u32);
                if seen[..count].contains(&key) {
                    continue;
                }
                seen[count] = key;
                count += 1;
                let mut at = self.starts[key as usize] as usize;
                while at < n && self.keys[at] == key {
                    visit(at);
                    at += 1;
                }
            }
        }
    }
}

fn permute<T: Copy>(values: &mut Vec<T>, order: &[u32], scratch: &mut Vec<T>) {
    scratch.clear();
    scratch.extend(order.iter().map(|&was| values[was as usize]));
    std::mem::swap(values, scratch);
}

// ---------------------------------------------------------------------------
// Neighbour lists and the PBF sums (graav's `pbf.rs`).
// ---------------------------------------------------------------------------

const MOST: usize = 64;
const SLACK: f32 = 1.2;
const NO_LIST: u32 = u32::MAX;

#[derive(Default)]
struct Near {
    lists: Vec<u32>,
    counts: Vec<u32>,
}

impl Near {
    fn build(&mut self, hash: &Hash, at: &[V], h: f32) {
        let reach = (h * SLACK) * (h * SLACK);
        self.lists.resize(at.len() * MOST, 0);
        self.lists
            .par_chunks_mut(MOST)
            .enumerate()
            .map(|(i, list)| {
                let (pos, mut count) = (at[i], 0usize);
                hash.near(pos, |n| {
                    let off = [at[n][0] - pos[0], at[n][1] - pos[1]];
                    if off[0] * off[0] + off[1] * off[1] <= reach {
                        if count < MOST {
                            list[count] = n as u32;
                        }
                        count += 1;
                    }
                });
                if count <= MOST { count as u32 } else { NO_LIST }
            })
            .collect_into_vec(&mut self.counts);
    }
}

/// The solver's numbers (graav's `Params`, less what a bake does not use).
struct Params {
    rest: f32,
    epsilon: f32,
    s_corr_k: f32,
    s_corr_n: f32,
    s_corr_dq: f32,
    xsph: f32,
}

struct State<'a> {
    p: &'a Params,
    k: &'a Kernels,
    hash: &'a Hash,
    near: &'a Near,
    predicted: &'a [V],
}

impl State<'_> {
    #[inline]
    fn each(&self, i: usize, mut visit: impl FnMut(usize)) {
        match self.near.counts[i] {
            NO_LIST => self.hash.near(self.predicted[i], visit),
            count => self.near.lists[i * MOST..i * MOST + count as usize].iter().for_each(|&n| visit(n as usize)),
        }
    }

    fn lambda(&self, i: usize) -> f32 {
        let (pos, h) = (self.predicted[i], self.k.h);
        let (mut density, mut grad_i, mut sum_grad_j) = (0.0f32, [0.0f32; 2], 0.0f32);
        self.each(i, |n| {
            let off = [self.predicted[n][0] - pos[0], self.predicted[n][1] - pos[1]];
            let sqr = off[0] * off[0] + off[1] * off[1];
            if sqr > h * h {
                return;
            }
            let r = sqr.sqrt();
            density += self.k.spiky2(r);
            if n != i && r > 0.0 {
                let slope = self.k.spiky2_d(r);
                grad_i = [grad_i[0] - off[0] / r * slope, grad_i[1] - off[1] / r * slope];
                sum_grad_j += slope * slope;
            }
        });
        let inv_rest = 1.0 / self.p.rest;
        let sum_grad = (grad_i[0] * grad_i[0] + grad_i[1] * grad_i[1] + sum_grad_j) * (inv_rest * inv_rest);
        -(density * inv_rest - 1.0) / (sum_grad + self.p.epsilon)
    }

    fn delta(&self, lambdas: &[f32], i: usize) -> V {
        let (pos, h, mine) = (self.predicted[i], self.k.h, lambdas[i]);
        let at_dq = self.k.spiky2(self.p.s_corr_dq * h).max(1e-12);
        let mut dp = [0.0f32; 2];
        self.each(i, |n| {
            if n == i {
                return;
            }
            let off = [self.predicted[n][0] - pos[0], self.predicted[n][1] - pos[1]];
            let sqr = off[0] * off[0] + off[1] * off[1];
            if sqr > h * h || sqr <= 0.0 {
                return;
            }
            let r = sqr.sqrt();
            let s_corr = -self.p.s_corr_k * (self.k.spiky2(r) / at_dq).powi(self.p.s_corr_n as i32);
            let w = -(mine + lambdas[n] + s_corr) * self.k.spiky2_d(r) / r;
            dp = [dp[0] + off[0] * w, dp[1] + off[1] * w];
        });
        let dp = [dp[0] / self.p.rest, dp[1] / self.p.rest];
        let (size, most) = (dp[0].hypot(dp[1]), 0.25 * h);
        if size > most { [dp[0] * most / size, dp[1] * most / size] } else { dp }
    }

    fn xsph(&self, velocities: &[V], i: usize) -> V {
        let (pos, h, mine) = (self.predicted[i], self.k.h, velocities[i]);
        let (mut pull, mut weight) = ([0.0f32; 2], self.k.poly6(0.0));
        self.each(i, |n| {
            if n == i {
                return;
            }
            let off = [self.predicted[n][0] - pos[0], self.predicted[n][1] - pos[1]];
            let sqr = off[0] * off[0] + off[1] * off[1];
            if sqr > h * h {
                return;
            }
            let w = self.k.poly6(sqr.sqrt());
            pull = [pull[0] + (velocities[n][0] - mine[0]) * w, pull[1] + (velocities[n][1] - mine[1]) * w];
            weight += w;
        });
        if weight > 1e-6 {
            [mine[0] + pull[0] / weight * self.p.xsph, mine[1] + pull[1] / weight * self.p.xsph]
        } else {
            mine
        }
    }
}

// ---------------------------------------------------------------------------
// What the water cannot go through: the colliders as a distance field.
// ---------------------------------------------------------------------------

/// The colliders baked into a grid of depths (geo units, + inside a solid),
/// band-limited, plus the canvas walls analytically. Built once a bake.
struct Solids {
    origin: V,
    cell: f32,
    w: usize,
    h: usize,
    depth: Vec<f32>,
    band: f32,
    walls: Option<V>,
}

fn seg_dist(p: V, a: V, b: V) -> f32 {
    let ab = [b[0] - a[0], b[1] - a[1]];
    let ap = [p[0] - a[0], p[1] - a[1]];
    let l = ab[0] * ab[0] + ab[1] * ab[1];
    let t = if l > 0.0 { ((ap[0] * ab[0] + ap[1] * ab[1]) / l).clamp(0.0, 1.0) } else { 0.0 };
    (ap[0] - ab[0] * t).hypot(ap[1] - ab[1] * t)
}

impl Solids {
    fn build(st: &Statics, aspect: f32, walls: bool, spacing: f32) -> Self {
        let pad = 0.25;
        let origin = [-pad, -pad];
        let span = [1.0 + 2.0 * pad, aspect + 2.0 * pad];
        let cell = (spacing * 0.5).max(span[0].max(span[1]) / 1024.0);
        let (w, h) = ((span[0] / cell).ceil() as usize + 1, (span[1] / cell).ceil() as usize + 1);
        let band = spacing * 8.0;
        let thick = spacing * 0.75;
        let at = |x: usize, y: usize| [origin[0] + x as f32 * cell, origin[1] + y as f32 * cell];
        let range = |lo: f32, hi: f32, o: f32, n: usize| {
            let a = (((lo - band - o) / cell).floor().max(0.0)) as usize;
            let b = (((hi + band - o) / cell).ceil().max(0.0) as usize).min(n.saturating_sub(1));
            (a, b)
        };
        // Distance to the solids' edges, and which cells are inside one.
        let mut edge = vec![band; w * h];
        let mut other = vec![-band; w * h];
        let mut segs: Vec<(V, V)> = Vec::new();
        for poly in st.solids.iter().flatten() {
            for i in 0..poly.len() {
                segs.push((poly[i], poly[(i + 1) % poly.len()]));
            }
        }
        for (a, b) in &segs {
            let (x0, x1) = range(a[0].min(b[0]), a[0].max(b[0]), origin[0], w);
            let (y0, y1) = range(a[1].min(b[1]), a[1].max(b[1]), origin[1], h);
            for y in y0..=y1 {
                for x in x0..=x1 {
                    let d = seg_dist(at(x, y), *a, *b);
                    let e = &mut edge[y * w + x];
                    *e = e.min(d);
                }
            }
        }
        // Inside: even-odd per solid (over all its rings, so holes are
        // open) along each row, unioned.
        let mut inside = vec![false; w * h];
        for solid in &st.solids {
            for y in 0..h {
                let py = origin[1] + y as f32 * cell;
                let mut xs: Vec<f32> = Vec::new();
                for poly in solid {
                    let n = poly.len();
                    for i in 0..n {
                        let (a, b) = (poly[i], poly[(i + 1) % n]);
                        if (a[1] > py) != (b[1] > py) {
                            xs.push(a[0] + (py - a[1]) / (b[1] - a[1]) * (b[0] - a[0]));
                        }
                    }
                }
                xs.sort_by(|a, b| a.total_cmp(b));
                for pair in xs.chunks(2) {
                    if let [l, r] = *pair {
                        let a = ((l - origin[0]) / cell).ceil().max(0.0) as usize;
                        let b = ((r - origin[0]) / cell).floor();
                        if b >= 0.0 {
                            for x in a..=(b as usize).min(w - 1) {
                                inside[y * w + x] = true;
                            }
                        }
                    }
                }
            }
        }
        // Open lines are walls of a thickness; points are discs.
        for line in &st.lines {
            for s in line.windows(2) {
                let (a, b) = (s[0], s[1]);
                let (x0, x1) = range(a[0].min(b[0]), a[0].max(b[0]), origin[0], w);
                let (y0, y1) = range(a[1].min(b[1]), a[1].max(b[1]), origin[1], h);
                for y in y0..=y1 {
                    for x in x0..=x1 {
                        let d = thick - seg_dist(at(x, y), a, b);
                        let o = &mut other[y * w + x];
                        *o = o.max(d);
                    }
                }
            }
        }
        for (c, r) in &st.discs {
            let (x0, x1) = range(c[0] - r, c[0] + r, origin[0], w);
            let (y0, y1) = range(c[1] - r, c[1] + r, origin[1], h);
            for y in y0..=y1 {
                for x in x0..=x1 {
                    let p = at(x, y);
                    let d = r - (p[0] - c[0]).hypot(p[1] - c[1]);
                    let o = &mut other[y * w + x];
                    *o = o.max(d);
                }
            }
        }
        let depth = (0..w * h).map(|i| (if inside[i] { edge[i] } else { -edge[i] }).max(other[i])).collect();
        Self { origin, cell, w, h, depth, band, walls: walls.then_some([1.0, aspect]) }
    }

    /// The grid's depth at a geo point, bilinear.
    fn grid(&self, p: V) -> f32 {
        let fx = (p[0] - self.origin[0]) / self.cell;
        let fy = (p[1] - self.origin[1]) / self.cell;
        if fx < 0.0 || fy < 0.0 || fx >= (self.w - 1) as f32 || fy >= (self.h - 1) as f32 {
            return -self.band;
        }
        let (x, y) = (fx as usize, fy as usize);
        let (tx, ty) = (fx - x as f32, fy - y as f32);
        let d = |x: usize, y: usize| self.depth[y * self.w + x];
        let top = d(x, y) + (d(x + 1, y) - d(x, y)) * tx;
        let bot = d(x, y + 1) + (d(x + 1, y + 1) - d(x, y + 1)) * tx;
        top + (bot - top) * ty
    }

    /// The canvas walls: how far outside the canvas, and which way is out.
    fn wall(&self, p: V) -> Option<(f32, V)> {
        let [wx, wy] = self.walls?;
        let cands = [(-p[0], [-1.0, 0.0]), (p[0] - wx, [1.0, 0.0]), (-p[1], [0.0, -1.0]), (p[1] - wy, [0.0, 1.0])];
        cands.into_iter().max_by(|a, b| a.0.total_cmp(&b.0))
    }

    /// Depth (+ inside) and the unit way further in, at a geo point.
    fn probe(&self, p: V) -> (f32, V) {
        let d = self.grid(p);
        let mut best = (d, [0.0, 1.0]);
        if d > 0.0 {
            let e = self.cell * 2.0;
            let dx = self.grid([p[0] + e, p[1]]) - self.grid([p[0] - e, p[1]]);
            let dy = self.grid([p[0], p[1] + e]) - self.grid([p[0], p[1] - e]);
            let steep = dx.hypot(dy);
            // In a sharp corner the slopes cancel: down into it then.
            best.1 = if steep > 1e-6 { [dx / steep, dy / steep] } else { [0.0, 1.0] };
        }
        if let Some(w) = self.wall(p) {
            if w.0 > best.0 {
                best = w;
            }
        }
        best
    }
}

/// The solids as the solver sees them: in solver units.
struct Solid<'a> {
    s: &'a Solids,
    k: f32,
}

impl Solid<'_> {
    fn probe(&self, p: V) -> (f32, V) {
        let (d, n) = self.s.probe([p[0] / self.k, p[1] / self.k]);
        (d * self.k, n)
    }

    /// Out of the solids, position only (the correction loop).
    fn out(&self, p: V, reach: f32) -> V {
        let (depth, n) = self.probe(p);
        if depth > 0.0 {
            let push = (depth + SKIN).min(reach);
            [p[0] - n[0] * push, p[1] - n[1] * push]
        } else {
            p
        }
    }

    /// Out of the solids, with the speed into them taken out and the rest
    /// damped (`damping`: the share kept).
    /// `from` is where the particle was: the move is walked in short steps
    /// and stopped at the first solid it meets, so a fast particle cannot
    /// tunnel through a thin wall (an open line is only a spacing or so
    /// wide).
    fn collide(&self, from: V, mut pos: V, mut vel: V, damping: f32, reach: f32) -> (V, V) {
        let step = SOLVER_SPACING * 0.4;
        let travel = (pos[0] - from[0]).hypot(pos[1] - from[1]);
        if travel > step {
            let n = (travel / step).ceil() as usize;
            for k in 1..n {
                let t = k as f32 / n as f32;
                let q = [from[0] + (pos[0] - from[0]) * t, from[1] + (pos[1] - from[1]) * t];
                if self.probe(q).0 > 0.0 {
                    pos = q;
                    break;
                }
            }
        }
        let (depth, n) = self.probe(pos);
        if depth > 0.0 {
            let push = (depth + SKIN).min(reach);
            pos = [pos[0] - n[0] * push, pos[1] - n[1] * push];
            let into = vel[0] * n[0] + vel[1] * n[1];
            if into > 0.0 {
                vel = [(vel[0] - n[0] * into) * damping, (vel[1] - n[1] * into) * damping];
            }
        }
        (pos, vel)
    }
}

// ---------------------------------------------------------------------------
// The particles.
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Fluid {
    pos: Vec<V>,
    vel: Vec<V>,
    pred: Vec<V>,
    /// Each particle's place in the bake's record (sorting reorders them).
    id: Vec<u32>,
    hash: Hash,
    near: Near,
    lambdas: Vec<f32>,
    scratch: (Vec<V>, Vec<f32>, Vec<u32>),
}

impl Fluid {
    #[allow(clippy::too_many_arguments)]
    fn substep(&mut self, p: &Params, k: &Kernels, solid: &Solid, gravity: f32, iterations: u32, damping: f32, dt: f32) {
        for i in 0..self.pos.len() {
            let v = [self.vel[i][0], self.vel[i][1] + gravity * dt];
            self.vel[i] = v;
            self.pred[i] = [self.pos[i][0] + v[0] * dt, self.pos[i][1] + v[1] * dt];
        }
        let order = self.hash.build(&self.pred, k.h).to_vec();
        permute(&mut self.pos, &order, &mut self.scratch.0);
        permute(&mut self.pred, &order, &mut self.scratch.0);
        permute(&mut self.vel, &order, &mut self.scratch.0);
        permute(&mut self.id, &order, &mut self.scratch.2);
        self.near.build(&self.hash, &self.pred, k.h);

        let reach = k.h;
        for _ in 0..iterations.max(1) {
            let state = State { p, k, hash: &self.hash, near: &self.near, predicted: &self.pred };
            (0..self.pred.len()).into_par_iter().map(|i| state.lambda(i)).collect_into_vec(&mut self.lambdas);
            let lambdas = &self.lambdas;
            (0..self.pred.len())
                .into_par_iter()
                .map(|i| {
                    let (at, dp) = (state.predicted[i], state.delta(lambdas, i));
                    solid.out([at[0] + dp[0], at[1] + dp[1]], reach)
                })
                .collect_into_vec(&mut self.scratch.0);
            std::mem::swap(&mut self.pred, &mut self.scratch.0);
        }

        for i in 0..self.pos.len() {
            self.vel[i] = [(self.pred[i][0] - self.pos[i][0]) / dt, (self.pred[i][1] - self.pos[i][1]) / dt];
        }
        let state = State { p, k, hash: &self.hash, near: &self.near, predicted: &self.pred };
        let vel = &self.vel;
        (0..self.pred.len()).into_par_iter().map(|i| state.xsph(vel, i)).collect_into_vec(&mut self.scratch.0);
        let fastest = k.h / dt;
        for i in 0..self.pos.len() {
            let v = self.scratch.0[i];
            let speed = v[0].hypot(v[1]);
            let v = if speed > fastest { [v[0] * fastest / speed, v[1] * fastest / speed] } else { v };
            (self.pos[i], self.vel[i]) = solid.collide(self.pos[i], self.pred[i], v, damping, reach);
        }
    }

    /// Particles that have left the canvas for good are dropped (their
    /// record reads NaN from then on, which draws nothing).
    fn cull(&mut self, lo: V, hi: V) {
        let mut i = 0;
        while i < self.pos.len() {
            let q = self.pos[i];
            if !(q[0] >= lo[0] && q[0] <= hi[0] && q[1] >= lo[1] && q[1] <= hi[1]) {
                self.pos.swap_remove(i);
                self.vel.swap_remove(i);
                self.pred.swap_remove(i);
                self.id.swap_remove(i);
            } else {
                i += 1;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Inputs -> particles, and the bake.
// ---------------------------------------------------------------------------

/// Even-odd inside test over a shape's rings.
fn inside(rings: &[&[V]], p: V) -> bool {
    let mut odd = false;
    for r in rings {
        let n = r.len();
        for i in 0..n {
            let (a, b) = (r[i], r[(i + 1) % n]);
            if (a[1] > p[1]) != (b[1] > p[1]) && p[0] < a[0] + (p[1] - a[1]) / (b[1] - a[1]) * (b[0] - a[0]) {
                odd = !odd;
            }
        }
    }
    odd
}

/// A water volume's particles: a lattice at `spacing` over its area (aligned
/// to the canvas, so neighbouring volumes do not overlap), nudged off the
/// exact grid so the first step has something to break symmetry with.
fn fill(b: &BodyIn, spacing: f32) -> Vec<V> {
    let (outline, holes): (Vec<V>, Vec<Vec<V>>) = match b {
        BodyIn::Poly { pts, holes, .. } => (pts.clone(), holes.clone()),
        BodyIn::Disc { .. } => (b.outline(), Vec::new()),
    };
    let rings: Vec<&[V]> = std::iter::once(outline.as_slice()).chain(holes.iter().map(|h| h.as_slice())).collect();
    let (mut lo, mut hi) = ([f32::MAX; 2], [f32::MIN; 2]);
    for q in &outline {
        lo = [lo[0].min(q[0]), lo[1].min(q[1])];
        hi = [hi[0].max(q[0]), hi[1].max(q[1])];
    }
    let mut out = Vec::new();
    let (i0, i1) = ((lo[0] / spacing).floor() as i64, (hi[0] / spacing).ceil() as i64);
    let (j0, j1) = ((lo[1] / spacing).floor() as i64, (hi[1] / spacing).ceil() as i64);
    for j in j0..=j1 {
        for i in i0..=i1 {
            let h = ((i.wrapping_mul(73856093) ^ j.wrapping_mul(19349663)) & 1023) as f32 / 1023.0 - 0.5;
            let p = [(i as f32 + 0.5 + h * 0.04) * spacing, (j as f32 + 0.5 - h * 0.04) * spacing];
            if inside(&rings, p) {
                out.push(p);
            }
        }
    }
    out
}

fn area(b: &BodyIn) -> f32 {
    let ring_area = |p: &[V]| {
        let n = p.len();
        (0..n).map(|i| p[i][0] * p[(i + 1) % n][1] - p[(i + 1) % n][0] * p[i][1]).sum::<f32>().abs() * 0.5
    };
    match b {
        BodyIn::Poly { pts, holes, .. } => (ring_area(pts) - holes.iter().map(|h| ring_area(h)).sum::<f32>()).max(0.0),
        BodyIn::Disc { r, .. } => std::f32::consts::PI * r * r,
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run(
    ins: &[BodyIn],
    track: &Track,
    s: &PhysicsSettings,
    aspect: f32,
    frames: u32,
    fps: f32,
    head: &mut dyn FnMut(Vec<BakedBody>, u32) -> PhysicsBake,
    sink: &mut dyn FnMut(BakeMsg) -> bool,
) -> bool {
    let f = &s.fluid;
    // Pinned shapes are solid, and drawn where they stand.
    let mut pinned_solids: Vec<Vec<Vec<[f32; 2]>>> = Vec::new();
    let mut pinned_rec: Vec<(u32, [f32; 3])> = Vec::new();
    let mut baked = Vec::new();
    let mut stride = 0u32;
    let water: Vec<&BodyIn> = ins.iter().filter(|b| !b.pinned()).collect();
    for b in ins.iter().filter(|b| b.pinned()) {
        let outline = b.outline();
        pinned_solids.push(vec![outline.clone()]);
        let c = super::centroid(&outline);
        let local = outline.iter().map(|p| [p[0] - c[0], p[1] - c[1]]).collect();
        let shape = BodyShape::Rigid { local, rings: Vec::new() };
        let off = stride;
        stride += 3;
        pinned_rec.push((off, [c[0], c[1], 0.0]));
        baked.push(BakedBody { shape, style: style_of(b), offset: off });
    }

    // The spacing, coarsened until the particles fit the caps.
    let total_area: f32 = water.iter().map(|b| area(b)).sum();
    let cap = MAX_PARTICLES.min(MAX_PARTICLE_FRAMES / frames.max(1) as usize).max(256) as f32;
    let spacing = f.spacing.max(1e-4).max((total_area / cap).sqrt() * 1.05);

    let mut fl = Fluid::default();
    for b in &water {
        let pts = fill(b, spacing);
        let v0 = b.attrs().map_or([0.0, 0.0], |a| [a.velocity[0] / METRES_PER_WIDTH, a.velocity[1] / METRES_PER_WIDTH]);
        let first = fl.pos.len() as u32;
        for p in pts {
            fl.id.push(fl.pos.len() as u32);
            fl.pos.push(p);
            fl.vel.push(v0);
        }
        let n = fl.pos.len() as u32 - first;
        let shape = BodyShape::Fluid { n, spacing };
        baked.push(BakedBody { shape, style: style_of(b), offset: stride });
        stride += 2 * n;
    }
    let fluid_off = stride - 2 * fl.pos.len() as u32;
    if !sink(BakeMsg::Header(head(baked, stride))) {
        return false;
    }

    // Into solver units.
    let kscale = SOLVER_SPACING / spacing;
    for (p, v) in fl.pos.iter_mut().zip(fl.vel.iter_mut()) {
        *p = [p[0] * kscale, p[1] * kscale];
        *v = [v[0] * kscale, v[1] * kscale];
    }
    fl.pred = fl.pos.clone();
    // The solids' depth grid; an animated collider input rebuilds it each
    // frame it changes.
    let build = |k: usize| {
        let mut st: Statics = track.frames[k.min(track.frames.len() - 1)].clone();
        st.solids.extend(pinned_solids.iter().cloned());
        Solids::build(&st, aspect, s.walls, spacing)
    };
    let mut solids = build(0);
    let mut built_at = 0usize;
    let kern = Kernels::new(H);
    let params = Params {
        rest: rest_density(&kern),
        epsilon: 20.0,
        s_corr_k: 0.001,
        s_corr_n: 4.0,
        s_corr_dq: 0.2,
        xsph: f.viscosity.clamp(0.0, 1.0),
    };
    let gravity = s.gravity / METRES_PER_WIDTH * kscale;
    let damping = (1.0 - 0.7 * s.friction).clamp(0.05, 1.0);
    let substeps = s.substeps.max(1);
    let dt = 1.0 / fps / substeps as f32;
    let (lo, hi) = ([-0.5 * kscale, -2.0 * kscale], [1.5 * kscale, (aspect + 0.5) * kscale]);

    let record = |fl: &Fluid| {
        let mut rec = vec![f32::NAN; stride as usize];
        for (off, pose) in &pinned_rec {
            rec[*off as usize..*off as usize + 3].copy_from_slice(pose);
        }
        for (p, &id) in fl.pos.iter().zip(&fl.id) {
            let o = (fluid_off + 2 * id) as usize;
            rec[o] = p[0] / kscale;
            rec[o + 1] = p[1] / kscale;
        }
        rec
    };
    for frame in 0..frames {
        if frame > 0 {
            let k = (frame as usize).min(track.frames.len() - 1);
            if k != built_at {
                if track.frames[k] != track.frames[built_at] {
                    solids = build(k);
                }
                built_at = k;
            }
            let solid = Solid { s: &solids, k: kscale };
            for _ in 0..substeps {
                fl.substep(&params, &kern, &solid, gravity, f.iterations, damping, dt);
            }
            fl.cull(lo, hi);
        }
        if !sink(BakeMsg::Frame(record(&fl))) {
            return false;
        }
    }
    true
}

fn style_of(b: &BodyIn) -> Style {
    match b {
        BodyIn::Poly { style, .. } | BodyIn::Disc { style, .. } => *style,
    }
}

// ---------------------------------------------------------------------------
// Drawing a frame.
// ---------------------------------------------------------------------------

/// One water volume's frame as geometry: `xy` is its record (x y x y ...,
/// NaN for a particle that has left).
pub(super) fn draw(xy: &[f32], spacing: f32, style: Style, id: u32, look: &FluidSettings, px_per_unit: f32, g: &mut Geometry) {
    let pts: Vec<V> = xy.chunks_exact(2).filter(|c| c[0].is_finite() && c[1].is_finite()).map(|c| [c[0], c[1]]).collect();
    if pts.is_empty() {
        return;
    }
    if matches!(look.look, super::FluidLook::Surface | super::FluidLook::Both) {
        let mut st = style;
        st.rule = FillRule::EvenOdd;
        for group in shapes(surface(&pts, spacing, look.blob.max(0.5)), spacing * 0.5) {
            let first = g.points.len() as u32;
            let mut starts = Vec::new();
            for r in &group {
                if g.points.len() as u32 > first {
                    starts.push(g.points.len() as u32 - first);
                }
                g.points.extend(r.iter().map(|&p| Point { p, id, ..Default::default() }));
            }
            let verts = (first..g.points.len() as u32).collect();
            g.prims.push(Prim { layer: Default::default(), kind: PrimKind::Poly { closed: true }, verts, style: st, phys: None, src: 0, rings: starts });
        }
    }
    if matches!(look.look, super::FluidLook::Particles | super::FluidLook::Both) {
        let r = (look.dot.max(0.05) * spacing * 0.5 * px_per_unit).max(0.5);
        let mut st = style;
        if st.fill[3] == 0.0 {
            st.fill = st.stroke;
        }
        st.stroke[3] = 0.0;
        // 2dini draws at most `geo::MAX_PRIMS` shapes: past `MAX_DOTS` an
        // even share of the particles is drawn (the panel says so).
        let every = pts.len().div_ceil(MAX_DOTS).max(1);
        for p in pts.into_iter().step_by(every) {
            let v = g.points.len() as u32;
            g.points.push(Point { p, pscale: r, id, ..Default::default() });
            g.prims.push(Prim { layer: Default::default(), kind: PrimKind::Point, verts: vec![v], style: st, phys: None, src: 0, rings: Vec::new() });
        }
    }
}

/// The most particle dots one water volume draws (`geo::MAX_PRIMS` and
/// `MAX_POINTS` are 4096 for a whole drawing).
pub const MAX_DOTS: usize = 2500;
/// The most outline points one water volume draws, and one shape holds
/// (`geo::MAX_VERTS_PER_PRIM`): the outline is simplified until it fits.
const MAX_SURFACE_POINTS: usize = 2400;
const MAX_SHAPE_POINTS: usize = 500;

/// Rings into shapes: each outer ring with the holes inside it, smoothed
/// and resampled to fit the caps above (`cell`: the field's cell).
fn shapes(rings: Vec<Vec<V>>, cell: f32) -> Vec<Vec<Vec<V>>> {
    let signed = |r: &[V]| {
        let n = r.len();
        (0..n).map(|i| r[i][0] * r[(i + 1) % n][1] - r[(i + 1) % n][0] * r[i][1]).sum::<f32>() * 0.5
    };
    // Marching squares' rings come in no particular winding: a ring inside
    // an odd number of others is a hole, and goes with the smallest even
    // one that holds it.
    let depth: Vec<usize> = (0..rings.len())
        .map(|i| (0..rings.len()).filter(|&j| j != i && inside(&[rings[j].as_slice()], rings[i][0])).count())
        .collect();
    let mut groups: Vec<Vec<Vec<V>>> = Vec::new();
    let mut home: Vec<Option<usize>> = vec![None; rings.len()];
    for i in (0..rings.len()).filter(|&i| depth[i] % 2 == 0) {
        home[i] = Some(groups.len());
        groups.push(vec![rings[i].clone()]);
    }
    for i in (0..rings.len()).filter(|&i| depth[i] % 2 == 1) {
        let holder = (0..rings.len())
            .filter(|&j| depth[j] % 2 == 0 && inside(&[rings[j].as_slice()], rings[i][0]))
            .min_by(|&a, &b| signed(&rings[a]).abs().total_cmp(&signed(&rings[b]).abs()));
        if let Some(g) = holder.and_then(|j| home[j]) {
            groups[g].push(rings[i].clone());
        }
    }
    // Smoothed (marching squares' staircase off), then resampled evenly:
    // under a cell apart where the caps allow, further where they do not.
    let perim = |r: &[V]| (0..r.len()).map(|i| dist(r[i], r[(i + 1) % r.len()])).sum::<f32>();
    let total: f32 = groups.iter().flatten().map(|r| perim(r)).sum();
    let base = (cell * 0.75).max(total / (MAX_SURFACE_POINTS as f32 - 8.0 * groups.len() as f32).max(64.0));
    groups
        .into_iter()
        .map(|gr| {
            let own: f32 = gr.iter().map(|r| perim(r)).sum();
            let step = base.max(own / (MAX_SHAPE_POINTS as f32 - 8.0 * gr.len() as f32).max(32.0));
            gr.iter().map(|r| resample(&smooth(r), step)).filter(|r| r.len() >= 3).collect::<Vec<_>>()
        })
        .filter(|g: &Vec<Vec<V>>| !g.is_empty())
        .collect()
}

fn dist(a: V, b: V) -> f32 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

/// Two passes of neighbour averaging round a closed ring.
fn smooth(r: &[V]) -> Vec<V> {
    let mut a = r.to_vec();
    let n = a.len();
    if n < 5 {
        return a;
    }
    for _ in 0..2 {
        let b = a.clone();
        for i in 0..n {
            let (p, q) = (b[(i + n - 1) % n], b[(i + 1) % n]);
            a[i] = [0.5 * b[i][0] + 0.25 * (p[0] + q[0]), 0.5 * b[i][1] + 0.25 * (p[1] + q[1])];
        }
    }
    a
}

/// A closed ring resampled at even arc-length steps (at least six points).
fn resample(r: &[V], step: f32) -> Vec<V> {
    let n = r.len();
    let seg: Vec<f32> = (0..n).map(|i| dist(r[i], r[(i + 1) % n])).collect();
    let total: f32 = seg.iter().sum();
    let m = ((total / step.max(1e-9)).round() as usize).max(6);
    if m >= n {
        return r.to_vec();
    }
    let mut out = Vec::with_capacity(m);
    let (mut i, mut acc) = (0usize, 0.0f32);
    for k in 0..m {
        let target = total * k as f32 / m as f32;
        while i < n - 1 && acc + seg[i] < target {
            acc += seg[i];
            i += 1;
        }
        let t = if seg[i] > 0.0 { ((target - acc) / seg[i]).clamp(0.0, 1.0) } else { 0.0 };
        let (p, q) = (r[i], r[(i + 1) % n]);
        out.push([p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t]);
    }
    out
}

/// The water's outline: marching squares over a metaball field of the
/// particles (each reaches `blob` spacings), at half a spacing a cell.
/// Closed rings, outer and holes alike (drawn even-odd).
fn surface(pts: &[V], spacing: f32, blob: f32) -> Vec<Vec<V>> {
    let r = blob * spacing;
    let (mut lo, mut hi) = ([f32::MAX; 2], [f32::MIN; 2]);
    for q in pts {
        lo = [lo[0].min(q[0]), lo[1].min(q[1])];
        hi = [hi[0].max(q[0]), hi[1].max(q[1])];
    }
    let mut cell = spacing * 0.5;
    let span = [hi[0] - lo[0] + 2.0 * r, hi[1] - lo[1] + 2.0 * r];
    cell = cell.max(span[0].max(span[1]) / 1500.0);
    // One empty node of margin all round, so every contour closes.
    let origin = [lo[0] - r - cell, lo[1] - r - cell];
    let w = (span[0] / cell).ceil() as usize + 3;
    let h = (span[1] / cell).ceil() as usize + 3;
    let mut f = vec![0.0f32; w * h];
    let (rr, reach) = (r * r, (r / cell).ceil() as isize);
    for q in pts {
        let (cx, cy) = (((q[0] - origin[0]) / cell).round() as isize, ((q[1] - origin[1]) / cell).round() as isize);
        for y in (cy - reach).max(1)..=(cy + reach).min(h as isize - 2) {
            for x in (cx - reach).max(1)..=(cx + reach).min(w as isize - 2) {
                let dx = origin[0] + x as f32 * cell - q[0];
                let dy = origin[1] + y as f32 * cell - q[1];
                let t = 1.0 - (dx * dx + dy * dy) / rr;
                if t > 0.0 {
                    f[y as usize * w + x as usize] += t * t * t;
                }
            }
        }
    }
    const ISO: f32 = 0.5;
    let node = |x: usize, y: usize| f[y * w + x];
    let pos_of = |e: u32| -> V {
        let (k, vert) = ((e >> 1) as usize, e & 1 == 1);
        let (x, y) = (k % w, k / w);
        let (a, (x2, y2)) = (node(x, y), if vert { (x, y + 1) } else { (x + 1, y) });
        let b = node(x2, y2);
        let t = if (b - a).abs() > 1e-12 { ((ISO - a) / (b - a)).clamp(0.0, 1.0) } else { 0.5 };
        [origin[0] + (x as f32 + t * (x2 - x) as f32) * cell, origin[1] + (y as f32 + t * (y2 - y) as f32) * cell]
    };
    let hz = |x: usize, y: usize| ((y * w + x) as u32) << 1;
    let vt = |x: usize, y: usize| (((y * w + x) as u32) << 1) | 1;
    // Segments, by the edge points they join.
    let mut segs: Vec<[u32; 2]> = Vec::new();
    for y in 0..h - 1 {
        for x in 0..w - 1 {
            let (a, b, c, d) = (node(x, y) > ISO, node(x + 1, y) > ISO, node(x + 1, y + 1) > ISO, node(x, y + 1) > ISO);
            let (top, right, bottom, left) = (hz(x, y), vt(x + 1, y), hz(x, y + 1), vt(x, y));
            let mut cut = Vec::with_capacity(4);
            if a != b {
                cut.push(top);
            }
            if b != c {
                cut.push(right);
            }
            if c != d {
                cut.push(bottom);
            }
            if d != a {
                cut.push(left);
            }
            match cut.len() {
                2 => segs.push([cut[0], cut[1]]),
                4 => {
                    let centre = (node(x, y) + node(x + 1, y) + node(x + 1, y + 1) + node(x, y + 1)) * 0.25 > ISO;
                    // a and c alike: the diagonal a-c is joined through the
                    // centre exactly when the centre agrees with a.
                    if centre == a {
                        segs.push([top, right]);
                        segs.push([bottom, left]);
                    } else {
                        segs.push([left, top]);
                        segs.push([right, bottom]);
                    }
                }
                _ => {}
            }
        }
    }
    // Join them into rings: every edge point has exactly two segments.
    let mut by: std::collections::HashMap<u32, [u32; 2]> = std::collections::HashMap::with_capacity(segs.len() * 2);
    for (i, s) in segs.iter().enumerate() {
        for &e in s {
            by.entry(e).and_modify(|v| v[1] = i as u32).or_insert([i as u32, u32::MAX]);
        }
    }
    let mut used = vec![false; segs.len()];
    let mut rings = Vec::new();
    for start in 0..segs.len() {
        if used[start] {
            continue;
        }
        let mut ring = Vec::new();
        let (mut seg, mut at) = (start, segs[start][0]);
        loop {
            used[seg] = true;
            ring.push(pos_of(at));
            let next_pt = if segs[seg][0] == at { segs[seg][1] } else { segs[seg][0] };
            let pair = by[&next_pt];
            let next = if pair[0] as usize == seg { pair[1] } else { pair[0] };
            at = next_pt;
            if next == u32::MAX || used[next as usize] {
                break;
            }
            seg = next as usize;
        }
        if ring.len() >= 3 {
            rings.push(ring);
        }
    }
    rings
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geophys::{bake, frame_geometry_look, PhysicsMode};

    fn rect(x0: f32, y0: f32, x1: f32, y1: f32) -> Geometry {
        let mut g = Geometry::default();
        for p in [[x0, y0], [x1, y0], [x1, y1], [x0, y1]] {
            g.points.push(Point::at(p));
        }
        let mut style = Style::default();
        style.fill = [0.2, 0.5, 1.0, 1.0];
        g.prims.push(Prim { layer: Default::default(), kind: PrimKind::Poly { closed: true }, verts: vec![0, 1, 2, 3], style, phys: None, src: 0, rings: Vec::new() });
        g
    }

    fn settings(frames: u32) -> PhysicsSettings {
        let mut s = PhysicsSettings { mode: PhysicsMode::Fluid, frames, substeps: 4, ..Default::default() };
        s.fluid.spacing = 0.02;
        s
    }

    fn live(b: &PhysicsBake, frame: usize) -> Vec<V> {
        let rec = &b.data[frame * b.stride as usize..(frame + 1) * b.stride as usize];
        rec.chunks_exact(2).filter(|c| c[0].is_finite()).map(|c| [c[0], c[1]]).collect()
    }

    #[test]
    fn the_rest_density_is_what_a_lattice_particle_feels() {
        let k = Kernels::new(H);
        let r = rest_density(&k);
        // graav tuned 55 particles/unit² at about this spacing.
        assert!(r > 30.0 && r < 90.0, "{r}");
    }

    #[test]
    fn a_block_of_water_falls_and_spreads_over_the_floor_and_stays_in() {
        let aspect = 0.5625;
        let water = rect(0.05, 0.05, 0.3, 0.3);
        let s = settings(60);
        let b = bake(&water, &Geometry::default(), &s, aspect, 640.0, 24.0);
        assert_eq!(b.frames, 60);
        let start = live(&b, 0);
        let end = live(&b, 59);
        assert!(start.len() > 100, "{}", start.len());
        assert_eq!(end.len(), start.len(), "the walls keep every particle");
        let mean = |v: &[V], i: usize| v.iter().map(|p| p[i]).sum::<f32>() / v.len() as f32;
        assert!(mean(&end, 1) > mean(&start, 1) + 0.1, "it fell: {} -> {}", mean(&start, 1), mean(&end, 1));
        let width = |v: &[V]| v.iter().map(|p| p[0]).fold(f32::MIN, f32::max) - v.iter().map(|p| p[0]).fold(f32::MAX, f32::min);
        assert!(width(&end) > width(&start) * 1.5, "it spread: {} -> {}", width(&start), width(&end));
        for p in &end {
            assert!(p[0] >= -0.01 && p[0] <= 1.01 && p[1] >= -0.01 && p[1] <= aspect + 0.01, "{p:?}");
        }
    }

    #[test]
    fn a_solid_collider_holds_the_water_up() {
        let aspect = 0.5625;
        let water = rect(0.4, 0.05, 0.6, 0.15);
        let shelf = rect(0.2, 0.3, 0.8, 0.35);
        let s = settings(48);
        let b = bake(&water, &shelf, &s, aspect, 640.0, 24.0);
        let end = live(&b, 47);
        let over = end.iter().filter(|p| p[0] > 0.25 && p[0] < 0.75 && p[1] < 0.31).count();
        assert!(over as f32 > end.len() as f32 * 0.6, "{over} of {} on the shelf", end.len());
        assert!(end.iter().all(|p| !(p[0] > 0.22 && p[0] < 0.78 && p[1] > 0.305 && p[1] < 0.345)), "none inside it");
    }

    #[test]
    fn the_same_inputs_bake_the_same_frames() {
        let water = rect(0.1, 0.1, 0.25, 0.25);
        let s = settings(12);
        let a = bake(&water, &Geometry::default(), &s, 0.5625, 640.0, 24.0);
        let b = bake(&water, &Geometry::default(), &s, 0.5625, 640.0, 24.0);
        assert!(a.data.iter().zip(&b.data).all(|(x, y)| x.to_bits() == y.to_bits()));
    }

    #[test]
    fn a_bake_round_trips_through_bytes_and_draws_a_closed_surface() {
        let water = rect(0.1, 0.1, 0.3, 0.2);
        let s = settings(4);
        let b = bake(&water, &Geometry::default(), &s, 0.5625, 640.0, 24.0);
        let back = PhysicsBake::from_bytes(&b.to_bytes()).expect("reads back");
        assert_eq!(back, b);
        let g = frame_geometry_look(&b, 0.0, &s.fluid, 640.0);
        assert_eq!(g.prims.len(), 1, "one surface");
        // The surface encloses the block (within a blob's reach).
        assert!(g.prims[0].verts.len() <= MAX_SHAPE_POINTS);
        let rings = g.rings(0);
        let (lo, hi) = rings.iter().flatten().fold(([f32::MAX; 2], [f32::MIN; 2]), |(l, h), p| ([l[0].min(p[0]), l[1].min(p[1])], [h[0].max(p[0]), h[1].max(p[1])]));
        assert!(lo[0] < 0.11 && hi[0] > 0.29 && lo[1] < 0.11 && hi[1] > 0.19, "{lo:?} {hi:?}");
        assert!(lo[0] > 0.05 && hi[0] < 0.35, "{lo:?} {hi:?}");
        let mut dots = s.fluid;
        dots.look = FluidLook::Particles;
        let g = frame_geometry_look(&b, 0.0, &dots, 640.0);
        assert_eq!(g.prims.len(), live(&b, 0).len());
    }

    #[test]
    fn a_ring_of_particles_traces_a_hole() {
        // A ring of points: the surface is an annulus, two rings.
        let mut pts = Vec::new();
        for k in 0..200 {
            let a = std::f32::consts::TAU * k as f32 / 200.0;
            for rad in [0.2f32, 0.21, 0.22] {
                pts.push([0.5 + rad * a.cos(), 0.5 + rad * a.sin()]);
            }
        }
        let rings = surface(&pts, 0.01, 1.8);
        assert_eq!(rings.len(), 2, "outer and hole");
    }
}

#[cfg(test)]
mod shape_tests {
    use super::*;

    #[test]
    fn a_long_wavy_outline_is_simplified_into_the_caps_and_keeps_its_shape() {
        // A wavy band across the canvas: thousands of outline points.
        let mut pts = Vec::new();
        for i in 0..3000 {
            let x = i as f32 / 3000.0;
            for j in 0..6 {
                pts.push([x, 0.3 + 0.05 * (x * 40.0).sin() + j as f32 * 0.004]);
            }
        }
        let rings = surface(&pts, 0.004, 1.8);
        assert!(rings.iter().map(|r| r.len()).sum::<usize>() > MAX_SHAPE_POINTS, "the test needs a long outline");
        let groups = shapes(rings, 0.002);
        let total: usize = groups.iter().flatten().map(|r| r.len()).sum();
        assert!(total <= MAX_SURFACE_POINTS, "{total}");
        for g in &groups {
            assert!(g.iter().map(|r| r.len()).sum::<usize>() <= MAX_SHAPE_POINTS);
        }
        // Still wavy: the outline reaches both crests.
        let ys: Vec<f32> = groups.iter().flatten().flatten().map(|p| p[1]).collect();
        let (lo, hi) = (ys.iter().cloned().fold(f32::MAX, f32::min), ys.iter().cloned().fold(f32::MIN, f32::max));
        assert!(lo < 0.255 && hi > 0.37, "{lo} {hi}");
    }

    #[test]
    fn a_hole_rides_with_the_ring_round_it() {
        let mut pts = Vec::new();
        for k in 0..200 {
            let a = std::f32::consts::TAU * k as f32 / 200.0;
            for rad in [0.2f32, 0.21, 0.22] {
                pts.push([0.5 + rad * a.cos(), 0.5 + rad * a.sin()]);
            }
        }
        pts.push([0.9, 0.9]);
        let groups = shapes(surface(&pts, 0.01, 1.8), 0.005);
        let mut sizes: Vec<usize> = groups.iter().map(|g| g.len()).collect();
        sizes.sort();
        assert_eq!(sizes, vec![1, 2], "a lone drop, and the annulus with its hole");
    }
}
