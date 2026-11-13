//! GridWarp: two lattices of control points over the node's format, a
//! SOURCE grid and a DESTINATION grid with the same shape, and the picture
//! moved so whatever sat under each source point lands on its destination
//! point. Between the points the warp is the lattice's interpolant —
//! bilinear per cell, or a Catmull-Rom patch net (`smooth`, the default),
//! which bends across cell edges instead of creasing at them.
//!
//! Both grids start as the same regular lattice, which is the identity.
//! Pull the destination to push pixels around; pull the source to choose
//! WHAT gets pushed. Moving both together is a no-op, on purpose.
//!
//! It fuses. Per output pixel the shader inverts the destination lattice —
//! Newton's method on the patch net, seeded at the identity and, if that
//! does not converge, again from the nearest destination point — then
//! evaluates the source lattice at the same lattice coordinate and samples
//! the input there (`Emit::input_at`, as Transform and UV Warp do). No
//! pass, no texture, no stage. Outside the destination lattice the warp
//! extends linearly off its edge rather than stopping. A pixel the inverse
//! cannot find — a destination grid folded over itself — renders
//! TRANSPARENT, never a plausible wrong sample.
//!
//! Every point is a uniform (and an animatable channel in the app); the
//! lattice SIZE is the only structure, because it sets the uniform span.

use crate::codegen::Emit;
use crate::nodes::NodeDef;

const TOPO_BASE: u64 = 0x6a17_0000;

/// The largest lattice, in cells per side. 16 x 16 cells is 289 points, one
/// vec4 each: a fifth of the uniform block, which is already a lot of
/// points to drag.
pub const GRIDWARP_MAX: u32 = 16;

/// Header vec4s before the points.
const HEADER: usize = 2;

/// Newton steps per seed. A lattice that has not converged in this many
/// is folded or wildly stretched; the second seed handles the rest.
const ITERS: u32 = 10;

/// A [`crate::graph::NodeKind::GridWarp`]'s lattices.
///
/// Points are NORMALISED over the node's format (0..1, y down), row-major,
/// `(cols + 1) * (rows + 1)` of them in each grid, so a warp survives a
/// resolution change and a proxy.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(from = "GridWarpCompat"))]
pub struct GridWarpParams {
    /// Cells across and down.
    pub cols: u32,
    pub rows: u32,
    /// Where each point sits in the SOURCE: what it picks up.
    pub src: Vec<[f32; 2]>,
    /// Where each point sits in the OUTPUT: where it puts it down.
    pub dst: Vec<[f32; 2]>,
    /// Catmull-Rom between points (bends through them); off = bilinear.
    pub smooth: bool,
    /// 0 = untouched, 1 = the full warp; the destination grid is scaled
    /// toward the source by it.
    pub amount: f32,
}

/// The saved shape, which also reads a GridWarp from before the prune:
/// one lattice of `points`, regular in the output, each holding the source
/// position it pulled from — exactly a source grid over an untouched
/// destination grid.
#[cfg(feature = "serde")]
#[derive(serde::Deserialize)]
struct GridWarpCompat {
    cols: u32,
    rows: u32,
    #[serde(default, alias = "points")]
    src: Vec<[f32; 2]>,
    #[serde(default)]
    dst: Vec<[f32; 2]>,
    #[serde(default = "yes")]
    smooth: bool,
    #[serde(default = "one")]
    amount: f32,
}

#[cfg(feature = "serde")]
fn yes() -> bool {
    true
}

#[cfg(feature = "serde")]
fn one() -> f32 {
    1.0
}

#[cfg(feature = "serde")]
impl From<GridWarpCompat> for GridWarpParams {
    fn from(c: GridWarpCompat) -> Self {
        let mut p = GridWarpParams::identity(c.cols, c.rows);
        let n = p.src.len();
        if c.src.len() == n {
            p.src = c.src;
        }
        if c.dst.len() == n {
            p.dst = c.dst;
        }
        p.smooth = c.smooth;
        p.amount = c.amount;
        p
    }
}

impl Default for GridWarpParams {
    fn default() -> Self {
        Self::identity(4, 4)
    }
}

/// What the lattice is at one lattice coordinate: both grids' positions
/// (normalised), and the destination's derivatives along the two lattice
/// axes (normalised per cell).
#[derive(Clone, Copy, Debug)]
pub struct GwEval {
    pub src: [f32; 2],
    pub dst: [f32; 2],
    pub du: [f32; 2],
    pub dv: [f32; 2],
}

/// The four tap weights along one axis at `t` in [0, 1] for taps -1, 0, 1,
/// 2, and their derivatives — the CPU mirror of `gw_w`.
fn weights(t: f32, smooth: bool) -> ([f32; 4], [f32; 4]) {
    if !smooth {
        return ([0.0, 1.0 - t, t, 0.0], [0.0, -1.0, 1.0, 0.0]);
    }
    let (t2, t3) = (t * t, t * t * t);
    (
        [
            0.5 * (-t3 + 2.0 * t2 - t),
            0.5 * (3.0 * t3 - 5.0 * t2 + 2.0),
            0.5 * (-3.0 * t3 + 4.0 * t2 + t),
            0.5 * (t3 - t2),
        ],
        [
            0.5 * (-3.0 * t2 + 4.0 * t - 1.0),
            0.5 * (9.0 * t2 - 10.0 * t),
            0.5 * (-9.0 * t2 + 8.0 * t + 1.0),
            0.5 * (3.0 * t2 - 2.0 * t),
        ],
    )
}

impl GridWarpParams {
    /// The identity: both grids the same regular `cols` x `rows` lattice.
    pub fn identity(cols: u32, rows: u32) -> Self {
        let (cols, rows) = (cols.clamp(1, GRIDWARP_MAX), rows.clamp(1, GRIDWARP_MAX));
        let mut src = Vec::with_capacity(((cols + 1) * (rows + 1)) as usize);
        for j in 0..=rows {
            for i in 0..=cols {
                src.push([i as f32 / cols as f32, j as f32 / rows as f32]);
            }
        }
        Self { cols, rows, dst: src.clone(), src, smooth: true, amount: 1.0 }
    }

    /// Points in each grid.
    pub fn count(&self) -> usize {
        ((self.cols + 1) * (self.rows + 1)) as usize
    }

    /// Point (i, j)'s index in either grid.
    pub fn index(&self, i: u32, j: u32) -> usize {
        (j * (self.cols + 1) + i) as usize
    }

    /// Where point `k` sits on the regular lattice, normalised.
    pub fn rest(&self, k: usize) -> [f32; 2] {
        let w = self.cols as usize + 1;
        [(k % w) as f32 / self.cols as f32, (k / w) as f32 / self.rows as f32]
    }

    /// The uniform span this lattice needs.
    pub fn slots(&self) -> usize {
        HEADER + self.count()
    }

    /// The destination as the shader sees it: scaled toward the source by
    /// `amount`.
    fn dst_eff(&self, k: usize) -> [f32; 2] {
        let (s, d) = (self.src[k], self.dst[k]);
        [s[0] + (d[0] - s[0]) * self.amount, s[1] + (d[1] - s[1]) * self.amount]
    }

    /// Point (i, j) as (src, dst), linearly extended one ring past the
    /// edge so the outer cells have neighbours to bend against.
    fn point(&self, i: i32, j: i32) -> [f32; 4] {
        let (c, r) = (self.cols as i32, self.rows as i32);
        let at = |i: i32, j: i32| {
            let k = self.index(i as u32, j as u32);
            let (s, d) = (self.src[k], self.dst_eff(k));
            [s[0], s[1], d[0], d[1]]
        };
        let (ci, cj) = (i.clamp(0, c), j.clamp(0, r));
        let v = at(ci, cj);
        let mut out = v;
        if i != ci {
            let n = at(if i < ci { 1 } else { c - 1 }, cj);
            for l in 0..4 {
                out[l] += v[l] - n[l];
            }
        }
        if j != cj {
            let n = at(ci, if j < cj { 1 } else { r - 1 });
            for l in 0..4 {
                out[l] += v[l] - n[l];
            }
        }
        out
    }

    /// Both grids at lattice coordinate `g` (0..cols, 0..rows), extended
    /// linearly outside it — the CPU mirror of `gw_eval`.
    pub fn eval(&self, g: [f32; 2]) -> GwEval {
        let gc = [g[0].clamp(0.0, self.cols as f32), g[1].clamp(0.0, self.rows as f32)];
        let i = (gc[0].floor() as i32).min(self.cols as i32 - 1);
        let j = (gc[1].floor() as i32).min(self.rows as i32 - 1);
        let (wx, dx) = weights(gc[0] - i as f32, self.smooth);
        let (wy, dy) = weights(gc[1] - j as f32, self.smooth);
        let mut e = [[0.0f32; 2]; 6]; // src, dst, du, dv, src du, src dv
        for b in 0..4 {
            for a in 0..4 {
                let v = self.point(i - 1 + a as i32, j - 1 + b as i32);
                let (w, wu, wv) = (wx[a] * wy[b], dx[a] * wy[b], wx[a] * dy[b]);
                for l in 0..2 {
                    e[0][l] += w * v[l];
                    e[1][l] += w * v[2 + l];
                    e[2][l] += wu * v[2 + l];
                    e[3][l] += wv * v[2 + l];
                    e[4][l] += wu * v[l];
                    e[5][l] += wv * v[l];
                }
            }
        }
        let ex = [g[0] - gc[0], g[1] - gc[1]];
        let ext = |p: [f32; 2], u: [f32; 2], v: [f32; 2]| {
            [p[0] + u[0] * ex[0] + v[0] * ex[1], p[1] + u[1] * ex[0] + v[1] * ex[1]]
        };
        GwEval { src: ext(e[0], e[4], e[5]), dst: ext(e[1], e[2], e[3]), du: e[2], dv: e[3] }
    }

    /// Newton from `g` toward the lattice coordinate whose destination is
    /// `q` (normalised). Residual measured in `fmt` pixels.
    fn newton(&self, q: [f32; 2], mut g: [f32; 2], fmt: [f32; 2]) -> ([f32; 2], f32) {
        let mut r2 = f32::MAX;
        for _ in 0..ITERS {
            let e = self.eval(g);
            let r = [(e.dst[0] - q[0]) * fmt[0], (e.dst[1] - q[1]) * fmt[1]];
            r2 = r[0] * r[0] + r[1] * r[1];
            if r2 < 1e-6 {
                break;
            }
            let (a, b) = (e.du[0] * fmt[0], e.dv[0] * fmt[0]);
            let (c, d) = (e.du[1] * fmt[1], e.dv[1] * fmt[1]);
            let det = a * d - b * c;
            if det.abs() < 1e-9 {
                break;
            }
            let mut s = [(d * r[0] - b * r[1]) / det, (-c * r[0] + a * r[1]) / det];
            let len = (s[0] * s[0] + s[1] * s[1]).sqrt();
            if len > 1.0 {
                s = [s[0] / len, s[1] / len];
            }
            g = [g[0] - s[0], g[1] - s[1]];
        }
        let e = self.eval(g);
        let r = [(e.dst[0] - q[0]) * fmt[0], (e.dst[1] - q[1]) * fmt[1]];
        if r2 > 1e-6 {
            r2 = r[0] * r[0] + r[1] * r[1];
        }
        (g, r2)
    }

    /// Where the output pixel `p` (pixels of a `fmt` format) samples the
    /// source, in the same pixels — or `None` where the destination
    /// lattice cannot be inverted there. The CPU mirror of `gridwarp`.
    pub fn sample_at(&self, p: [f32; 2], fmt: [f32; 2]) -> Option<[f32; 2]> {
        let q = [p[0] / fmt[0], p[1] / fmt[1]];
        let seed = [q[0] * self.cols as f32, q[1] * self.rows as f32];
        let (mut g, mut r2) = self.newton(q, seed, fmt);
        if r2 > CONVERGED {
            let mut best = (f32::MAX, seed);
            for k in 0..self.count() {
                let d = self.dst_eff(k);
                let dd = ((d[0] - q[0]) * fmt[0]).powi(2) + ((d[1] - q[1]) * fmt[1]).powi(2);
                if dd < best.0 {
                    let w = self.cols as usize + 1;
                    best = (dd, [(k % w) as f32, (k / w) as f32]);
                }
            }
            (g, r2) = self.newton(q, best.1, fmt);
        }
        (r2 <= CONVERGED).then(|| {
            let s = self.eval(g).src;
            [s[0] * fmt[0], s[1] * fmt[1]]
        })
    }

    /// Re-grid to `cols` x `rows`, keeping the warp: each new point takes
    /// both grids' values at its lattice coordinate, so a doubling is exact
    /// for a bilinear lattice and close for a smooth one.
    pub fn resize(&mut self, cols: u32, rows: u32) {
        let mut out = Self::identity(cols, rows);
        out.smooth = self.smooth;
        out.amount = self.amount;
        let full = Self { amount: 1.0, ..self.clone() };
        for k in 0..out.count() {
            let r = out.rest(k);
            let e = full.eval([r[0] * self.cols as f32, r[1] * self.rows as f32]);
            out.src[k] = e.src;
            out.dst[k] = e.dst;
        }
        *self = out;
    }

    /// The span with the node's format in it: the codegen's central packer
    /// knows the per-node formats, `pack` does not.
    pub fn pack_fmt(&self, fmt: [f32; 2], v: &mut [f32]) {
        self.pack(v);
        v[2] = fmt[0];
        v[3] = fmt[1];
    }
}

/// Squared pixels of residual that count as found.
const CONVERGED: f32 = 0.01;

impl NodeDef for GridWarpParams {
    const NAME: &'static str = "GridWarp";
    /// The source, then the mask.
    const INPUTS: usize = 2;
    const MASK_INPUT: Option<usize> = Some(1);
    /// The HEADER only: the points follow, one vec4 each, and the codegen
    /// reads the whole span from [`GridWarpParams::slots`].
    const VEC4_SLOTS: usize = HEADER;

    /// The lattice's shape sets the span; every value is a uniform.
    fn topo_tag(&self) -> u64 {
        TOPO_BASE + (self.cols as u64) * (GRIDWARP_MAX as u64 + 1) + self.rows as u64
    }

    /// (cols, rows, fmt w, fmt h), (smooth, -, -, -), then a vec4 per point:
    /// (src x, src y, dst x, dst y). The format is written by
    /// [`GridWarpParams::pack_fmt`].
    fn pack(&self, v: &mut [f32]) {
        v[0] = self.cols as f32;
        v[1] = self.rows as f32;
        v[4] = f32::from(u8::from(self.smooth));
        let n = self.count().min(self.src.len()).min(self.dst.len());
        for k in 0..n {
            let o = (HEADER + k) * 4;
            if o + 4 > v.len() {
                break;
            }
            let (s, d) = (self.src[k], self.dst_eff(k));
            v[o..o + 4].copy_from_slice(&[s[0], s[1], d[0], d[1]]);
        }
    }

    fn emit(&self, e: &mut Emit) -> String {
        if !e.has_input(0) {
            return "vec4f(0.0)".into();
        }
        let name = wgsl(e);
        let slot = e.slot();
        let pos = e.pos().to_string();
        let w = e.fresh();
        e.line(format_args!("let {w} = {name}({pos}, {slot}u);"));
        let q = e.fresh();
        e.line(format_args!("let {q} = {w}.xy;"));
        let v = e.input_at(0, &q);
        let found = format!("select(vec4f(0.0), {v}, {w}.z > 0.5)");
        if e.has_input(1) {
            let orig = e.input(0);
            e.masked(Self::MASK_INPUT, &orig, found)
        } else {
            found
        }
    }
}

/// Emit the lattice functions once per shader; returns the entry's name.
fn wgsl(e: &mut Emit) -> String {
    let name = "gridwarp".to_string();
    let (_, fns) = e.color_helpers();
    if fns.contains_key(&name) {
        return name;
    }
    fns.insert(
        name.clone(),
        format!(
            "// GridWarp: point k of the lattice is P.v[slot + {HEADER} + k] =
// (src.xy, dst.xy), normalised, row-major. One ring past the edge is
// extended linearly so the outer cells have neighbours.
fn gw_pt(slot: u32, cols: i32, rows: i32, i: i32, j: i32) -> vec4f {{
    let ci = clamp(i, 0, cols);
    let cj = clamp(j, 0, rows);
    let w = u32(cols + 1);
    let v = P.v[slot + {HEADER}u + u32(cj) * w + u32(ci)];
    var o = v;
    if (i != ci) {{
        let ni = select(cols - 1, 1, i < ci);
        o = o + v - P.v[slot + {HEADER}u + u32(cj) * w + u32(ni)];
    }}
    if (j != cj) {{
        let nj = select(rows - 1, 1, j < cj);
        o = o + v - P.v[slot + {HEADER}u + u32(nj) * w + u32(ci)];
    }}
    return o;
}}
// Tap weights for taps -1..2 at t, and their derivatives: Catmull-Rom, or
// bilinear (the outer two zero).
struct GwW {{ w: vec4f, d: vec4f }}
fn gw_w(t: f32, sm: bool) -> GwW {{
    if (!sm) {{
        return GwW(vec4f(0.0, 1.0 - t, t, 0.0), vec4f(0.0, -1.0, 1.0, 0.0));
    }}
    let t2 = t * t;
    let t3 = t2 * t;
    return GwW(
        0.5 * vec4f(-t3 + 2.0 * t2 - t, 3.0 * t3 - 5.0 * t2 + 2.0, -3.0 * t3 + 4.0 * t2 + t, t3 - t2),
        0.5 * vec4f(-3.0 * t2 + 4.0 * t - 1.0, 9.0 * t2 - 10.0 * t, -9.0 * t2 + 8.0 * t + 1.0, 3.0 * t2 - 2.0 * t),
    );
}}
// Both grids at lattice coordinate g, extended linearly outside the
// lattice: both positions and the destination's two lattice derivatives.
struct GwE {{ s: vec2f, d: vec2f, du: vec2f, dv: vec2f }}
fn gw_eval(slot: u32, cols: i32, rows: i32, sm: bool, g: vec2f) -> GwE {{
    let gc = clamp(g, vec2f(0.0), vec2f(f32(cols), f32(rows)));
    let i = min(i32(floor(gc.x)), cols - 1);
    let j = min(i32(floor(gc.y)), rows - 1);
    let wx = gw_w(gc.x - f32(i), sm);
    let wy = gw_w(gc.y - f32(j), sm);
    // Bilinear needs only the middle two taps each way.
    let lo = select(1, 0, sm);
    let hi = select(3, 4, sm);
    var s = vec2f(0.0);
    var d = vec2f(0.0);
    var du = vec2f(0.0);
    var dv = vec2f(0.0);
    var su = vec2f(0.0);
    var sv = vec2f(0.0);
    for (var b = lo; b < hi; b = b + 1) {{
        for (var a = lo; a < hi; a = a + 1) {{
            let v = gw_pt(slot, cols, rows, i - 1 + a, j - 1 + b);
            let w = wx.w[a] * wy.w[b];
            let wu = wx.d[a] * wy.w[b];
            let wv = wx.w[a] * wy.d[b];
            s = s + w * v.xy;
            d = d + w * v.zw;
            du = du + wu * v.zw;
            dv = dv + wv * v.zw;
            su = su + wu * v.xy;
            sv = sv + wv * v.xy;
        }}
    }}
    let ex = g - gc;
    return GwE(s + su * ex.x + sv * ex.y, d + du * ex.x + dv * ex.y, du, dv);
}}
// Newton toward the lattice coordinate whose destination is q, residual
// in pixels. Returns (g, squared residual).
fn gw_newton(slot: u32, cols: i32, rows: i32, sm: bool, q: vec2f, fmt: vec2f, g0: vec2f) -> vec3f {{
    var g = g0;
    var r2 = 1e30;
    for (var k = 0u; k < {ITERS}u; k = k + 1u) {{
        let e = gw_eval(slot, cols, rows, sm, g);
        let r = (e.d - q) * fmt;
        r2 = dot(r, r);
        if (r2 < 1e-6) {{ break; }}
        let ju = e.du * fmt;
        let jv = e.dv * fmt;
        let det = ju.x * jv.y - jv.x * ju.y;
        if (abs(det) < 1e-9) {{ break; }}
        var st = vec2f(jv.y * r.x - jv.x * r.y, -ju.y * r.x + ju.x * r.y) / det;
        let len = length(st);
        if (len > 1.0) {{ st = st / len; }}
        g = g - st;
    }}
    if (r2 >= 1e-6) {{
        let r = (gw_eval(slot, cols, rows, sm, g).d - q) * fmt;
        r2 = dot(r, r);
    }}
    return vec3f(g, r2);
}}
// The source position (pixels) output pixel p samples, and 1 in z where
// the destination lattice could be inverted there (0 = transparent).
fn gridwarp(p: vec2f, slot: u32) -> vec3f {{
    let h = P.v[slot];
    let cols = max(i32(h.x), 1);
    let rows = max(i32(h.y), 1);
    let fmt = max(h.zw, vec2f(1.0));
    let sm = P.v[slot + 1u].x > 0.5;
    let q = p / fmt;
    let seed = q * vec2f(f32(cols), f32(rows));
    var n = gw_newton(slot, cols, rows, sm, q, fmt, seed);
    if (n.z > {CONVERGED:?}) {{
        // A big warp: start again from the nearest destination point.
        var best = 1e30;
        var g0 = seed;
        let w = u32(cols + 1);
        let all = w * u32(rows + 1);
        for (var k = 0u; k < all; k = k + 1u) {{
            let r = (P.v[slot + {HEADER}u + k].zw - q) * fmt;
            let dd = dot(r, r);
            if (dd < best) {{
                best = dd;
                g0 = vec2f(f32(k % w), f32(k / w));
            }}
        }}
        n = gw_newton(slot, cols, rows, sm, q, fmt, g0);
    }}
    let s = gw_eval(slot, cols, rows, sm, n.xy).s * fmt;
    return vec3f(s, select(0.0, 1.0, n.z <= {CONVERGED:?}));
}}
"
        ),
    );
    name
}

#[cfg(test)]
mod tests {
    use super::*;

    const FMT: [f32; 2] = [200.0, 100.0];

    fn close(a: [f32; 2], b: [f32; 2], tol: f32) -> bool {
        (a[0] - b[0]).abs() < tol && (a[1] - b[1]).abs() < tol
    }

    #[test]
    fn the_identity_samples_where_it_lands() {
        for smooth in [false, true] {
            let p = GridWarpParams { smooth, ..GridWarpParams::identity(4, 3) };
            for q in [[0.5, 0.5], [37.2, 81.0], [199.5, 99.5], [-20.0, 130.0]] {
                let s = p.sample_at(q, FMT).expect("identity always inverts");
                assert!(close(s, q, 1e-2), "{smooth} {q:?} -> {s:?}");
            }
        }
    }

    /// Moving both grids by the same amount is still the identity.
    #[test]
    fn moving_both_grids_together_does_nothing() {
        let mut p = GridWarpParams::identity(3, 3);
        let k = p.index(1, 2);
        p.src[k] = [0.4, 0.71];
        p.dst[k] = [0.4, 0.71];
        for q in [[30.0, 20.0], [80.0, 70.0], [150.0, 40.0]] {
            let s = p.sample_at(q, FMT).unwrap();
            assert!(close(s, q, 1e-2), "{q:?} -> {s:?}");
        }
    }

    /// A destination point's pixel samples its source point, exactly —
    /// that is the contract a user drags against.
    #[test]
    fn a_destination_point_samples_its_source_point() {
        for smooth in [false, true] {
            let mut p = GridWarpParams { smooth, ..GridWarpParams::identity(4, 4) };
            let k = p.index(2, 2);
            p.dst[k] = [0.62, 0.41];
            p.src[k] = [0.45, 0.55];
            let k2 = p.index(1, 3);
            p.dst[k2] = [0.2, 0.8];
            for k in [k, k2] {
                let d = p.dst[k];
                let s = p.sample_at([d[0] * FMT[0], d[1] * FMT[1]], FMT).unwrap();
                let want = [p.src[k][0] * FMT[0], p.src[k][1] * FMT[1]];
                assert!(close(s, want, 0.05), "{smooth}: {s:?} vs {want:?}");
            }
        }
    }

    /// A pull most of the way across its neighbours' cells (but not past
    /// them — that folds the lattice, and a fold has two answers).
    #[test]
    fn a_big_pull_still_inverts() {
        let mut p = GridWarpParams::identity(2, 2);
        let k = p.index(1, 1);
        p.dst[k] = [0.8, 0.25];
        let d = p.dst[k];
        let s = p.sample_at([d[0] * FMT[0], d[1] * FMT[1]], FMT).unwrap();
        let want = [p.src[k][0] * FMT[0], p.src[k][1] * FMT[1]];
        assert!(close(s, want, 0.05), "{s:?} vs {want:?}");
    }

    #[test]
    fn amount_zero_is_the_identity() {
        let mut p = GridWarpParams::identity(2, 2);
        p.dst[4] = [0.8, 0.2];
        p.amount = 0.0;
        let s = p.sample_at([100.0, 50.0], FMT).unwrap();
        assert!(close(s, [100.0, 50.0], 1e-2), "{s:?}");
    }

    #[test]
    fn resizing_keeps_the_warp() {
        let mut p = GridWarpParams { smooth: false, ..GridWarpParams::identity(2, 2) };
        p.dst[4] = [0.6, 0.45];
        let before: Vec<_> =
            [[60.0, 30.0], [120.0, 45.0], [150.0, 80.0]].iter().map(|q| p.sample_at(*q, FMT).unwrap()).collect();
        p.resize(4, 4);
        assert_eq!(p.count(), 25);
        for (q, b) in [[60.0, 30.0], [120.0, 45.0], [150.0, 80.0]].iter().zip(before) {
            let a = p.sample_at(*q, FMT).unwrap();
            assert!(close(a, b, 0.05), "{q:?}: {a:?} vs {b:?}");
        }
    }

    #[test]
    fn only_the_lattice_shape_is_topology() {
        let a = GridWarpParams::identity(3, 3);
        let mut b = a.clone();
        b.dst[5] = [0.1, 0.9];
        b.smooth = false;
        b.amount = 0.3;
        assert_eq!(a.topo_tag(), b.topo_tag());
        assert_ne!(a.topo_tag(), GridWarpParams::identity(4, 3).topo_tag());
        assert_ne!(GridWarpParams::identity(3, 4).topo_tag(), GridWarpParams::identity(4, 3).topo_tag());
        let max = GridWarpParams::identity(GRIDWARP_MAX, GRIDWARP_MAX);
        assert!(max.slots() * 16 <= 16384 / 3, "the biggest lattice takes {} bytes", max.slots() * 16);
    }

    #[test]
    fn pack_lays_out_the_points() {
        let mut p = GridWarpParams::identity(1, 1);
        p.dst[3] = [0.9, 0.8];
        let mut v = vec![0.0; p.slots() * 4];
        p.pack_fmt([64.0, 32.0], &mut v);
        assert_eq!(&v[0..5], &[1.0, 1.0, 64.0, 32.0, 1.0]);
        assert_eq!(&v[(HEADER + 3) * 4..(HEADER + 4) * 4], &[1.0, 1.0, 0.9, 0.8]);
    }

    /// The fused shader a GridWarp lands in parses and validates, masked
    /// and not.
    #[test]
    fn the_generated_shader_validates() {
        use crate::graph::{Graph, NodeKind};
        for masked in [false, true] {
            let mut g = Graph::default();
            g.nodes.insert(0, NodeKind::read(0, ""));
            g.nodes.insert(1, NodeKind::read(1, ""));
            g.nodes.insert(2, NodeKind::GridWarp(GridWarpParams::default()));
            g.edges.insert((2, 0), 0);
            if masked {
                g.edges.insert((2, 1), 1);
            }
            g.view = Some(2);
            let src = crate::codegen::generate(&g).main.wgsl;
            assert!(src.contains("fn gridwarp("), "{src}");
            let module = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| panic!("parse: {e}\n{src}"));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                .validate(&module)
                .unwrap_or_else(|e| panic!("validate: {e:?}"));
        }
    }

    /// A comp saved before the prune stored one lattice of `points`: the
    /// source positions over a regular output.
    #[cfg(feature = "serde")]
    #[test]
    fn an_old_single_lattice_loads_as_its_source_grid() {
        let old = r#"{"cols":1,"rows":1,"points":[[0.1,0.0],[1.0,0.0],[0.0,1.0],[1.0,1.0]],"smooth":false}"#;
        let p: GridWarpParams = serde_json::from_str(old).unwrap();
        assert_eq!(p.src[0], [0.1, 0.0]);
        assert_eq!(p.dst, GridWarpParams::identity(1, 1).dst);
        assert!(!p.smooth);
        assert_eq!(p.amount, 1.0);
        let back: GridWarpParams = serde_json::from_str(&serde_json::to_string(&p).unwrap()).unwrap();
        assert_eq!(back, p);
    }
}
