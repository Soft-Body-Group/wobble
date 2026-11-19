//! Planar tracking on the CPU — the Mocha paradigm.
//!
//! A region of one frame (a quad, in pixels) is assumed to lie on a plane.
//! Its motion through the sequence is then a homography per frame, and the
//! tracker finds it by DIRECT DENSE ALIGNMENT: every pixel of the region
//! votes, not a handful of corners, which is what makes a planar track hold
//! on soft, low-texture surfaces where a point tracker slides. The solver
//! is ESM (Benhimane & Malis: Gauss-Newton with the mean of the template
//! and warped-image gradients, second-order convergence) run coarse to fine
//! over a pyramid, with zero-mean/unit-variance normalisation (a lighting
//! change is not motion) and Tukey reweighting (an occluder is not either).
//!
//! Two paradigms come out of one solve:
//!
//! * **Planar** — the homography itself, constrained to a `Model`
//!   (translation / similarity / affine / perspective). Its picture is the
//!   region's four corners carried through the sequence: a corner pin.
//! * **Mesh** — a triangle mesh over the region, each vertex refined off the
//!   planar prediction by a local translation-only alignment and smoothed
//!   against its neighbours. That is the deformation a plane cannot express
//!   (cloth, a face, a bending page): PowerMesh, roughly.
//!
//! Everything here is `f32`/`f64` and `Vec`; nothing touches wgpu. The
//! app's job (`planarui.rs`) decodes frames on a worker and calls
//! [`Tracker::step`] once per frame; `bin/planar.rs` runs the same code on
//! a directory of frames and draws the result.
//!
//! Coordinates: continuous pixel coordinates, pixel (i, j) covering
//! [i, i+1) × [j, j+1) with its centre at (i+0.5, j+0.5) — the viewer's
//! output space, so a normalised position is `(x / w, y / h)` exactly.

use rayon::prelude::*;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Homographies
// ---------------------------------------------------------------------------

/// A 3×3 homography, row-major, mapping `(x, y, 1)` to `(u, v, w)`.
pub type H = [f64; 9];

pub const IDENTITY: H = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

pub fn h_mul(a: &H, b: &H) -> H {
    let mut r = [0.0; 9];
    for i in 0..3 {
        for j in 0..3 {
            r[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
        }
    }
    r
}

pub fn h_inv(h: &H) -> Option<H> {
    let [a, b, c, d, e, f, g, hh, i] = *h;
    let det = a * (e * i - f * hh) - b * (d * i - f * g) + c * (d * hh - e * g);
    if det.abs() < 1e-300 || !det.is_finite() {
        return None;
    }
    let inv = 1.0 / det;
    Some([
        (e * i - f * hh) * inv,
        (c * hh - b * i) * inv,
        (b * f - c * e) * inv,
        (f * g - d * i) * inv,
        (a * i - c * g) * inv,
        (c * d - a * f) * inv,
        (d * hh - e * g) * inv,
        (b * g - a * hh) * inv,
        (a * e - b * d) * inv,
    ])
}

/// Apply to a point. A point at infinity (w ≈ 0) comes back huge, never NaN.
pub fn h_apply(h: &H, x: f64, y: f64) -> (f64, f64) {
    let u = h[0] * x + h[1] * y + h[2];
    let v = h[3] * x + h[4] * y + h[5];
    let w = h[6] * x + h[7] * y + h[8];
    let w = if w.abs() < 1e-12 { 1e-12f64.copysign(w) } else { w };
    (u / w, v / w)
}

pub fn h_translation(tx: f64, ty: f64) -> H {
    [1.0, 0.0, tx, 0.0, 1.0, ty, 0.0, 0.0, 1.0]
}

pub fn h_scale(s: f64) -> H {
    [s, 0.0, 0.0, 0.0, s, 0.0, 0.0, 0.0, 1.0]
}

/// Normalise so the bottom-right entry is 1 (a homography is defined up to
/// scale; keeping it there makes the entries comparable across frames).
pub fn h_norm(h: &H) -> H {
    if h[8].abs() > 1e-12 {
        let inv = 1.0 / h[8];
        let mut r = *h;
        for v in r.iter_mut() {
            *v *= inv;
        }
        r
    } else {
        *h
    }
}

/// The homography taking four source points to four destination points
/// (DLT on the 8×8 system). `None` for a degenerate quad.
pub fn h_from_quad(src: &[[f64; 2]; 4], dst: &[[f64; 2]; 4]) -> Option<H> {
    let mut a = [0.0f64; 64];
    let mut b = [0.0f64; 8];
    for k in 0..4 {
        let (x, y) = (src[k][0], src[k][1]);
        let (u, v) = (dst[k][0], dst[k][1]);
        let r0 = 2 * k;
        let r1 = 2 * k + 1;
        a[r0 * 8..r0 * 8 + 8].copy_from_slice(&[x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y]);
        a[r1 * 8..r1 * 8 + 8].copy_from_slice(&[0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y]);
        b[r0] = u;
        b[r1] = v;
    }
    let s = solve_linear(&mut a, &mut b, 8)?;
    Some([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], 1.0])
}

/// Gaussian elimination with partial pivoting on an n×n row-major system.
pub(crate) fn solve_linear(a: &mut [f64], b: &mut [f64], n: usize) -> Option<Vec<f64>> {
    for col in 0..n {
        // Pivot.
        let mut best = col;
        let mut best_v = a[col * n + col].abs();
        for r in col + 1..n {
            let v = a[r * n + col].abs();
            if v > best_v {
                best = r;
                best_v = v;
            }
        }
        if best_v < 1e-14 || !best_v.is_finite() {
            return None;
        }
        if best != col {
            for c in 0..n {
                a.swap(col * n + c, best * n + c);
            }
            b.swap(col, best);
        }
        let p = a[col * n + col];
        for r in col + 1..n {
            let f = a[r * n + col] / p;
            if f == 0.0 {
                continue;
            }
            for c in col..n {
                a[r * n + c] -= f * a[col * n + c];
            }
            b[r] -= f * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for r in (0..n).rev() {
        let mut s = b[r];
        for c in r + 1..n {
            s -= a[r * n + c] * x[c];
        }
        x[r] = s / a[r * n + r];
    }
    Some(x)
}

// ---------------------------------------------------------------------------
// Motion models
// ---------------------------------------------------------------------------

/// What the planar solve is allowed to do. Each is a subgroup of the
/// homographies, and each is parameterised by the GENERATORS below: the
/// solve's update is `H ← H · (I + Σ δₖ Gₖ)`, so the model is the list of
/// generators and nothing else knows which one is running.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Model {
    /// 2 DOF: position.
    Translation,
    /// 4 DOF: position, rotation, uniform scale.
    Similarity,
    /// 6 DOF: + shear / anisotropic scale.
    Affine,
    /// 8 DOF: + perspective. The corner pin.
    #[default]
    Perspective,
}

impl Model {
    pub const ALL: [Model; 4] =
        [Model::Translation, Model::Similarity, Model::Affine, Model::Perspective];

    pub fn label(self) -> &'static str {
        match self {
            Model::Translation => "translation",
            Model::Similarity => "similarity",
            Model::Affine => "affine",
            Model::Perspective => "perspective",
        }
    }

    pub fn dof(self) -> usize {
        self.generators().len()
    }

    /// The Lie-algebra basis of the model, as 3×3 matrices: the update
    /// direction for each parameter.
    pub fn generators(self) -> &'static [H] {
        const TX: H = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        const TY: H = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        const SCALE: H = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        const ROT: H = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        const E11: H = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        const E12: H = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        const E21: H = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        const E22: H = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        const E31: H = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        const E32: H = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        const TRANSLATION: [H; 2] = [TX, TY];
        const SIMILARITY: [H; 4] = [TX, TY, SCALE, ROT];
        const AFFINE: [H; 6] = [TX, TY, E11, E12, E21, E22];
        const PERSPECTIVE: [H; 8] = [TX, TY, E11, E12, E21, E22, E31, E32];
        match self {
            Model::Translation => &TRANSLATION,
            Model::Similarity => &SIMILARITY,
            Model::Affine => &AFFINE,
            Model::Perspective => &PERSPECTIVE,
        }
    }
}

/// Where a generator sends the point `(x, y)` to first order: the
/// derivative of the dehomogenised warp `(I + εG)(x, y)` at ε = 0.
#[inline]
fn generator_flow(g: &H, x: f64, y: f64) -> (f64, f64) {
    let wz = g[6] * x + g[7] * y + g[8];
    (
        g[0] * x + g[1] * y + g[2] - x * wz,
        g[3] * x + g[4] * y + g[5] - y * wz,
    )
}

// ---------------------------------------------------------------------------
// Images
// ---------------------------------------------------------------------------

/// One grayscale float plane.
#[derive(Clone, Debug)]
pub struct Plane {
    pub w: usize,
    pub h: usize,
    pub px: Vec<f32>,
}

impl Plane {
    pub fn new(w: usize, h: usize, px: Vec<f32>) -> Self {
        assert_eq!(px.len(), w * h);
        Self { w, h, px }
    }

    pub fn filled(w: usize, h: usize, v: f32) -> Self {
        Self { w, h, px: vec![v; w * h] }
    }

    #[inline]
    pub fn at(&self, x: usize, y: usize) -> f32 {
        self.px[y * self.w + x]
    }

    /// A tracking plane from an RGBA-f16 frame (row padding dropped):
    /// Rec.709 luma through a soft log, so a scene-linear plate does not
    /// hand every vote to its highlights. `ln(1 + 16v) / ln 17` maps 0..1
    /// to 0..1 and stays monotone above it.
    pub fn from_rgba_f16(src: &[half::f16], w: usize, h: usize, row_stride: usize) -> Self {
        let mut px = vec![0.0f32; w * h];
        let k = 1.0 / 17f32.ln();
        if w > 0 {
            px.par_chunks_mut(w).enumerate().for_each(|(y, out)| {
                let row = y * row_stride;
                for (x, o) in out.iter_mut().enumerate() {
                    let i = row + x * 4;
                    if i + 2 < src.len() {
                        let r = src[i].to_f32();
                        let g = src[i + 1].to_f32();
                        let b = src[i + 2].to_f32();
                        let l = (0.2126 * r + 0.7152 * g + 0.0722 * b).max(0.0);
                        *o = (1.0 + 16.0 * l).ln() * k;
                    }
                }
            });
        }
        Self { w, h, px }
    }

    /// Bilinear sample at a continuous coordinate; `None` outside the
    /// pixel-centre hull.
    #[inline]
    pub fn sample(&self, x: f64, y: f64) -> Option<f32> {
        let fx = x - 0.5;
        let fy = y - 0.5;
        if !(fx >= 0.0 && fy >= 0.0 && fx <= (self.w - 1) as f64 && fy <= (self.h - 1) as f64) {
            return None;
        }
        let ix = fx as usize;
        let iy = fy as usize;
        let tx = (fx - ix as f64) as f32;
        let ty = (fy - iy as f64) as f32;
        let ix1 = (ix + 1).min(self.w - 1);
        let iy1 = (iy + 1).min(self.h - 1);
        let a = self.px[iy * self.w + ix];
        let b = self.px[iy * self.w + ix1];
        let c = self.px[iy1 * self.w + ix];
        let d = self.px[iy1 * self.w + ix1];
        Some((a * (1.0 - tx) + b * tx) * (1.0 - ty) + (c * (1.0 - tx) + d * tx) * ty)
    }

    /// The plane minus its local mean over a (2r+1)² window: the DETAIL
    /// layer. Shading that varies slowly across the surface — a sheen that
    /// moves with the view, a fold's shadow, a light sweeping past —
    /// cancels, and what is left is the texture the solve can align. On a
    /// cloth shot the raw residuals of that shading looked like a giant
    /// occluder and every frame was declared covered.
    pub fn highpass(&self, r: usize) -> Plane {
        let (w, h) = (self.w, self.h);
        if w == 0 || h == 0 {
            return self.clone();
        }
        let mut sum = self.px.clone();
        let mut tmp = Vec::new();
        box_blur(&mut sum, &mut tmp, w, h, r);
        let mut out = vec![0.0f32; w * h];
        out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            let y0 = y.saturating_sub(r);
            let y1 = (y + r).min(h - 1);
            let ny = (y1 - y0 + 1) as f32;
            for (x, o) in row.iter_mut().enumerate() {
                let x0 = x.saturating_sub(r);
                let x1 = (x + r).min(w - 1);
                let n = ny * (x1 - x0 + 1) as f32;
                *o = self.px[y * w + x] - sum[y * w + x] / n;
            }
        });
        Plane { w, h, px: out }
    }

    /// Central-difference gradient planes, edges clamped.
    pub fn gradients(&self) -> (Plane, Plane) {
        let (w, h) = (self.w, self.h);
        let mut gx = vec![0.0f32; w * h];
        let mut gy = vec![0.0f32; w * h];
        if w > 0 && h > 0 {
            gx.par_chunks_mut(w).zip(gy.par_chunks_mut(w)).enumerate().for_each(|(y, (rx, ry))| {
                let y0 = y.saturating_sub(1);
                let y1 = (y + 1).min(h - 1);
                for x in 0..w {
                    let x0 = x.saturating_sub(1);
                    let x1 = (x + 1).min(w - 1);
                    rx[x] = 0.5 * (self.px[y * w + x1] - self.px[y * w + x0]);
                    ry[x] = 0.5 * (self.px[y1 * w + x] - self.px[y0 * w + x]);
                }
            });
        }
        (Plane { w, h, px: gx }, Plane { w, h, px: gy })
    }

    /// Half-size plane through a [1 3 3 1]/8 binomial filter centred
    /// BETWEEN pixel pairs, so a full-res coordinate `p` lands at exactly
    /// `p / 2` on the new plane — the pyramid needs no half-pixel bookkeeping.
    pub fn downsample(&self) -> Plane {
        let (w, h) = (self.w, self.h);
        let (w2, h2) = ((w / 2).max(1), (h / 2).max(1));
        let cl = |i: isize, n: usize| i.clamp(0, n as isize - 1) as usize;
        let mut tmp = vec![0.0f32; w2 * h];
        tmp.par_chunks_mut(w2).enumerate().for_each(|(y, t)| {
            let row = &self.px[y * w..y * w + w];
            for (x, o) in t.iter_mut().enumerate() {
                let c = 2 * x as isize;
                *o = 0.125
                    * (row[cl(c - 1, w)] + 3.0 * row[cl(c, w)] + 3.0 * row[cl(c + 1, w)] + row[cl(c + 2, w)]);
            }
        });
        let mut out = vec![0.0f32; w2 * h2];
        out.par_chunks_mut(w2).enumerate().for_each(|(y, o)| {
            let c = 2 * y as isize;
            let (r0, r1, r2, r3) = (cl(c - 1, h), cl(c, h), cl(c + 1, h), cl(c + 2, h));
            for (x, v) in o.iter_mut().enumerate() {
                *v = 0.125
                    * (tmp[r0 * w2 + x] + 3.0 * tmp[r1 * w2 + x] + 3.0 * tmp[r2 * w2 + x] + tmp[r3 * w2 + x]);
            }
        });
        Plane { w: w2, h: h2, px: out }
    }
}

/// One pyramid level: the plane and its gradients.
#[derive(Clone, Debug)]
pub struct Level {
    pub img: Plane,
    pub gx: Plane,
    pub gy: Plane,
    /// The IGNORE matte at this level (1 = not the plane: a foreground
    /// occluder the user roto'd), or none. A pixel under it is treated as
    /// off the frame: no vote in the solve, the statistics or the score,
    /// and never cut into a template.
    pub ignore: Option<Plane>,
}

/// Above this (bilinear, at the level's own scale) an ignore matte keeps a
/// pixel out. Low on purpose: an occluder's soft edge and the gradients it
/// smears into its neighbours are not the plane either, and the pyramid's
/// blur spreads the edge a little wider at each level, which is the
/// dilation a coarse level wants.
pub const IGNORE_ABOVE: f32 = 0.25;

impl Level {
    /// Whether the ignore matte covers level coordinate `(x, y)`.
    #[inline]
    pub fn ignored(&self, x: f64, y: f64) -> bool {
        self.ignore.as_ref().and_then(|m| m.sample(x, y)).is_some_and(|v| v > IGNORE_ABOVE)
    }
}

/// A Gaussian-ish pyramid; level `l` is the full image at scale `2^-l`.
#[derive(Clone, Debug)]
pub struct Pyramid {
    pub levels: Vec<Level>,
}

impl Pyramid {
    pub fn build(img: Plane, n_levels: usize) -> Self {
        Self::build_with(img, n_levels, None)
    }

    /// With `detail`, every level is its own high-passed detail layer
    /// (radius in that level's pixels — one band per octave); the next
    /// level is still downsampled from the FULL image, or the coarse levels
    /// would be empty.
    pub fn build_with(img: Plane, n_levels: usize, detail: Option<usize>) -> Self {
        let mut levels = Vec::with_capacity(n_levels.max(1));
        let mut cur = img;
        for l in 0..n_levels.max(1) {
            let next = if l + 1 < n_levels && cur.w >= 4 && cur.h >= 4 {
                Some(cur.downsample())
            } else {
                None
            };
            let img = match detail {
                Some(r) => cur.highpass(r),
                None => cur,
            };
            let (gx, gy) = img.gradients();
            levels.push(Level { img, gx, gy, ignore: None });
            match next {
                Some(n) => cur = n,
                None => break,
            }
        }
        Self { levels }
    }

    pub fn base(&self) -> &Plane {
        &self.levels[0].img
    }

    /// Attach an ignore matte (frame-sized, 1 = not the plane), downsampled
    /// alongside the picture. A matte of another size is ignored itself.
    pub fn with_ignore(mut self, matte: Plane) -> Self {
        let Some(l0) = self.levels.first() else { return self };
        if matte.w != l0.img.w || matte.h != l0.img.h {
            return self;
        }
        let mut cur = matte;
        let n = self.levels.len();
        for (l, lev) in self.levels.iter_mut().enumerate() {
            let next = (l + 1 < n).then(|| cur.downsample());
            lev.ignore = Some(cur);
            match next {
                Some(m) => cur = m,
                None => break,
            }
        }
        self
    }
}

// ---------------------------------------------------------------------------
// Polygons
// ---------------------------------------------------------------------------

pub fn poly_bbox(poly: &[[f64; 2]]) -> ([f64; 2], [f64; 2]) {
    let mut lo = [f64::INFINITY; 2];
    let mut hi = [f64::NEG_INFINITY; 2];
    for p in poly {
        for k in 0..2 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    (lo, hi)
}

/// Even-odd point-in-polygon.
pub fn poly_contains(poly: &[[f64; 2]], x: f64, y: f64) -> bool {
    let n = poly.len();
    let mut inside = false;
    let mut j = n.wrapping_sub(1);
    for i in 0..n {
        let (xi, yi) = (poly[i][0], poly[i][1]);
        let (xj, yj) = (poly[j][0], poly[j][1]);
        if (yi > y) != (yj > y) {
            let xc = xj + (y - yj) * (xi - xj) / (yi - yj);
            if x < xc {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}

pub fn poly_centroid(poly: &[[f64; 2]]) -> [f64; 2] {
    let n = poly.len().max(1) as f64;
    let mut c = [0.0; 2];
    for p in poly {
        c[0] += p[0] / n;
        c[1] += p[1] / n;
    }
    c
}

pub fn poly_map(poly: &[[f64; 2]], h: &H) -> Vec<[f64; 2]> {
    poly.iter()
        .map(|p| {
            let (x, y) = h_apply(h, p[0], p[1]);
            [x, y]
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Template
// ---------------------------------------------------------------------------

/// The region's pixels at every pyramid level, in NORMALISED template
/// coordinates (`(p - origin) / scale`, roughly the unit square), each with
/// its zero-mean unit-variance value and the gradient of that value.
/// Normalising keeps the 8×8 normal equations conditioned whatever the
/// region's size in pixels.
#[derive(Clone, Debug)]
pub struct Template {
    /// The region polygon, full-res pixels of the frame it was cut from.
    pub polygon: Vec<[f64; 2]>,
    /// Full-res pixels → template coordinates.
    pub norm: H,
    pub norm_inv: H,
    pub scale: f64,
    pub levels: Vec<TemplateLevel>,
}

#[derive(Clone, Debug, Default)]
pub struct TemplateLevel {
    pub xs: Vec<f32>,
    pub ys: Vec<f32>,
    /// Zero-mean, unit-variance.
    pub val: Vec<f32>,
    /// Gradient of `val` per template unit.
    pub gx: Vec<f32>,
    pub gy: Vec<f32>,
    /// Each pixel's cell in a `gw`×`gh` grid over the region's bounding
    /// box (one cell per stride step), for the residual blur that tells an
    /// occluder from a misaligned edge.
    pub gi: Vec<u32>,
    pub gw: usize,
    pub gh: usize,
    /// The grid's origin in this level's pixels.
    pub ox: usize,
    pub oy: usize,
}

impl TemplateLevel {
    pub fn len(&self) -> usize {
        self.xs.len()
    }
    pub fn is_empty(&self) -> bool {
        self.xs.is_empty()
    }
}

/// Fewest pixels a level may hold and still be solved on.
const MIN_LEVEL_PIXELS: usize = 24;
/// Most pixels a level is allowed: above this the region is subsampled —
/// at RANDOM, never on a lattice. A stride-2 lattice at the finest level
/// aliased the texture and moved a converged fit from 0.07 px to 0.5 px,
/// with the objective preferring the wrong answer.
const MAX_LEVEL_PIXELS: usize = 250_000;

/// A cheap hash in [0, 1) for the subsample lottery.
#[inline]
fn lottery(x: usize, y: usize) -> f32 {
    let mut h = (x as u32).wrapping_mul(0x8da6_b343) ^ (y as u32).wrapping_mul(0xd816_3841);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2c1b_3c6d);
    h ^= h >> 12;
    (h & 0xffff) as f32 / 65536.0
}

impl Template {
    /// Cut the region out of `pyr`. Levels stop where the region falls
    /// under `min_side` pixels across (so the coarsest level still has
    /// something to align) or at the pyramid's depth.
    pub fn build(pyr: &Pyramid, polygon: &[[f64; 2]], min_side: f64) -> Self {
        Self::build_masked(pyr, polygon, min_side, &|_, _| true)
    }

    /// `build`, keeping only the pixels `keep` accepts (full-res
    /// coordinates). A re-anchor cuts its template through the solve's own
    /// weights, so whatever was in front of the plane — a ball, a hand —
    /// is left out rather than baked in for the rest of the shot.
    pub fn build_masked(
        pyr: &Pyramid,
        polygon: &[[f64; 2]],
        min_side: f64,
        keep: &dyn Fn(f64, f64) -> bool,
    ) -> Self {
        let origin = poly_centroid(polygon);
        let (lo, hi) = poly_bbox(polygon);
        let ext = (hi[0] - lo[0]).max(hi[1] - lo[1]).max(1.0);
        let scale = 0.5 * ext;
        let norm: H = [1.0 / scale, 0.0, -origin[0] / scale, 0.0, 1.0 / scale, -origin[1] / scale, 0.0, 0.0, 1.0];
        let norm_inv = h_inv(&norm).unwrap_or(IDENTITY);
        let mut levels = Vec::new();
        let short = (hi[0] - lo[0]).min(hi[1] - lo[1]);
        for (l, lev) in pyr.levels.iter().enumerate() {
            let f = (1u64 << l) as f64;
            if l > 0 && short / f < min_side {
                break;
            }
            let poly_l: Vec<[f64; 2]> = polygon.iter().map(|p| [p[0] / f, p[1] / f]).collect();
            let (llo, lhi) = poly_bbox(&poly_l);
            let x0 = llo[0].floor().max(0.0) as usize;
            let y0 = llo[1].floor().max(0.0) as usize;
            let x1 = (lhi[0].ceil() as usize).min(lev.img.w);
            let y1 = (lhi[1].ceil() as usize).min(lev.img.h);
            if x1 <= x0 || y1 <= y0 {
                break;
            }
            let est = (x1 - x0) * (y1 - y0);
            let density = (MAX_LEVEL_PIXELS as f32 / est.max(1) as f32).min(1.0);
            let mut tl = TemplateLevel::default();
            tl.gw = x1 - x0;
            tl.gh = y1 - y0;
            tl.ox = x0;
            tl.oy = y0;
            let gscale = (scale / f) as f32;
            for y in y0..y1 {
                for x in x0..x1 {
                    if density < 1.0 && lottery(x, y) >= density {
                        continue;
                    }
                    let cx = x as f64 + 0.5;
                    let cy = y as f64 + 0.5;
                    if !poly_contains(&poly_l, cx, cy) || lev.ignored(cx, cy) || !keep(cx * f, cy * f) {
                        continue;
                    }
                    let (tx, ty) = h_apply(&norm, cx * f, cy * f);
                    tl.gi.push(((y - y0) * tl.gw + (x - x0)) as u32);
                    tl.xs.push(tx as f32);
                    tl.ys.push(ty as f32);
                    tl.val.push(lev.img.at(x, y));
                    tl.gx.push(lev.gx.at(x, y) * gscale);
                    tl.gy.push(lev.gy.at(x, y) * gscale);
                }
            }
            if tl.len() < MIN_LEVEL_PIXELS {
                break;
            }
            // Zero mean, unit variance — and the gradients scaled the same.
            let n = tl.len() as f64;
            let mean = tl.val.iter().map(|&v| v as f64).sum::<f64>() / n;
            let var = tl.val.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
            let inv = 1.0 / var.sqrt().max(1e-6);
            for v in tl.val.iter_mut() {
                *v = ((*v as f64 - mean) * inv) as f32;
            }
            for v in tl.gx.iter_mut().chain(tl.gy.iter_mut()) {
                *v = (*v as f64 * inv) as f32;
            }
            levels.push(tl);
        }
        Self { polygon: polygon.to_vec(), norm, norm_inv, scale, levels }
    }

    /// How much texture the region has for a translation solve: the
    /// smaller eigenvalue of the 2×2 gradient covariance at the finest
    /// level, per pixel. Shi–Tomasi's measure; ~0 on a flat patch.
    pub fn texture(&self) -> f64 {
        let Some(l) = self.levels.first() else { return 0.0 };
        let n = l.len().max(1) as f64;
        let (mut xx, mut xy, mut yy) = (0.0f64, 0.0f64, 0.0f64);
        for k in 0..l.len() {
            let (gx, gy) = (l.gx[k] as f64, l.gy[k] as f64);
            xx += gx * gx;
            xy += gx * gy;
            yy += gy * gy;
        }
        let (xx, xy, yy) = (xx / n, xy / n, yy / n);
        let tr = xx + yy;
        let det = xx * yy - xy * xy;
        let disc = (tr * tr * 0.25 - det).max(0.0).sqrt();
        (tr * 0.5 - disc).max(0.0)
    }
}

// ---------------------------------------------------------------------------
// The solve
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct SolveOpts {
    pub model: Model,
    /// Gauss-Newton iterations per level.
    pub max_iters: usize,
    /// Stop when no parameter moves the region by more than this many
    /// full-res pixels.
    pub eps_px: f64,
    /// Tukey-reweight residuals (occlusion, specular hits).
    pub robust: bool,
}

impl Default for SolveOpts {
    fn default() -> Self {
        Self { model: Model::Perspective, max_iters: 20, eps_px: 0.02, robust: true }
    }
}

/// The robust weights of the finest level, on a grid over the region's
/// bounding box in TEMPLATE-frame pixels: what the planar solve decided
/// was not the plane (an occluder, a specular hit). −1 where the grid has
/// no sample.
#[derive(Clone, Debug, Default)]
pub struct WeightMap {
    pub ox: usize,
    pub oy: usize,
    pub w: usize,
    pub h: usize,
    pub cells: Vec<f32>,
}

impl WeightMap {
    /// Mean weight over the window of half-size `r` around `(x, y)`
    /// (template-frame pixels); 1 where nothing is known.
    pub fn around(&self, x: f64, y: f64, r: f64) -> f32 {
        if self.w == 0 || self.h == 0 {
            return 1.0;
        }
        let x0 = ((x - r).floor() as i64 - self.ox as i64).clamp(0, self.w as i64 - 1) as usize;
        let x1 = ((x + r).ceil() as i64 - self.ox as i64).clamp(0, self.w as i64 - 1) as usize;
        let y0 = ((y - r).floor() as i64 - self.oy as i64).clamp(0, self.h as i64 - 1) as usize;
        let y1 = ((y + r).ceil() as i64 - self.oy as i64).clamp(0, self.h as i64 - 1) as usize;
        let mut sum = 0.0;
        let mut n = 0.0;
        for yy in y0..=y1 {
            for xx in x0..=x1 {
                let v = self.cells[yy * self.w + xx];
                if v >= 0.0 {
                    sum += v;
                    n += 1.0;
                }
            }
        }
        if n > 0.0 {
            sum / n
        } else {
            1.0
        }
    }
}

/// What a solve came back with.
#[derive(Clone, Debug)]
pub struct Fit {
    /// Template frame pixels → current frame pixels.
    pub h: H,
    /// Zero-normalised cross-correlation of the region at the fit, −1..1.
    /// 1 is a perfect match; below ~0.5 the region is probably not there.
    pub score: f32,
    /// Fraction of the region's pixels that landed inside the frame.
    pub coverage: f32,
    pub iters: usize,
    /// The finest level's robust weights (robust solves only).
    pub weights: Option<WeightMap>,
    /// Fraction of the visible region the reweighting rejected: what the
    /// solve decided was not the plane. Above ~half, the fit is a guess.
    pub rejected: f32,
}

/// One template pixel's view of the current frame, refreshed per
/// iteration. One array rather than six so a chunk of pixels is one slice
/// for rayon.
#[derive(Clone, Copy, Debug)]
struct Sample {
    valid: bool,
    /// The warped image, normalised over the voting set.
    iw: f32,
    /// Gradient of `iw` per template unit.
    gtx: f32,
    gty: f32,
    res: f32,
    /// Robust weight (1 until the weighted phase).
    wt: f32,
}

impl Default for Sample {
    fn default() -> Self {
        Self { valid: false, iw: 0.0, gtx: 0.0, gty: 0.0, res: 0.0, wt: 1.0 }
    }
}

/// Scratch for one solve, kept across levels and iterations.
#[derive(Default)]
struct Scratch {
    px: Vec<Sample>,
    acc: Vec<f32>,
    cnt: Vec<f32>,
    tmp: Vec<f32>,
    /// The correlation gate per cell, computed once per level (the first
    /// weighted iteration) — it barely moves between iterations and costs
    /// five blurs.
    gate: Vec<f32>,
}

/// Pixels per rayon task. A mesh patch is one task; a full-frame region
/// is a hundred.
const CHUNK: usize = 4096;
/// Radius, in a level's pixels, of the window the robust weights judge a
/// residual in: its local mean and its local correlation. Wider than an
/// edge, narrower than the shading of a surface.
const SHADE_RADIUS: usize = 8;
/// Template variance (normalised units) under which a window has nothing
/// to align and the correlation gate stays out of it.
const FLAT_VAR: f32 = 0.05;

impl Scratch {
    /// Size the sample array for a level, every weight back to 1.
    fn reset(&mut self, n: usize) {
        self.px.clear();
        self.px.resize(n, Sample::default());
        self.gate.clear();
    }
}

/// Separable box blur of radius `r` over a `w`×`h` grid, in place via `tmp`.
fn box_blur(grid: &mut [f32], tmp: &mut Vec<f32>, w: usize, h: usize, r: usize) {
    tmp.clear();
    tmp.resize(w * h, 0.0);
    tmp.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, o) in row.iter_mut().enumerate() {
            let x0 = x.saturating_sub(r);
            let x1 = (x + r).min(w - 1);
            let mut s = 0.0;
            for xx in x0..=x1 {
                s += grid[y * w + xx];
            }
            *o = s;
        }
    });
    grid.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let y0 = y.saturating_sub(r);
        let y1 = (y + r).min(h - 1);
        for (x, o) in row.iter_mut().enumerate() {
            let mut s = 0.0;
            for yy in y0..=y1 {
                s += tmp[yy * w + x];
            }
            *o = s;
        }
    });
}

/// Sample the frame under the current warp for one template level. Fills
/// each `Sample` with the warped image, normalised to zero mean and unit
/// variance over the valid pixels (weighted by the previous iteration's
/// robust weights when `use_wts`), and the gradient of that normalised
/// image with respect to TEMPLATE coordinates. Returns the valid count.
fn warp_level(tl: &TemplateLevel, lev: &Level, hl: &H, s: &mut Scratch, use_wts: bool) -> usize {
    let n = tl.len();
    if s.px.len() != n {
        s.reset(n);
    }
    let hl = *hl;
    let (count, wsum, sum, sum2) = s
        .px
        .par_chunks_mut(CHUNK)
        .zip(tl.xs.par_chunks(CHUNK))
        .zip(tl.ys.par_chunks(CHUNK))
        .map(|((px, xs), ys)| {
            let mut count = 0usize;
            let mut wsum = 0.0f64;
            let mut sum = 0.0f64;
            let mut sum2 = 0.0f64;
            for (k, p) in px.iter_mut().enumerate() {
                let prev_w = if use_wts { p.wt as f64 } else { 1.0 };
                p.valid = false;
                let x = xs[k] as f64;
                let y = ys[k] as f64;
                let u = hl[0] * x + hl[1] * y + hl[2];
                let v = hl[3] * x + hl[4] * y + hl[5];
                let w = hl[6] * x + hl[7] * y + hl[8];
                if w.abs() < 1e-9 {
                    continue;
                }
                let sx = u / w;
                let sy = v / w;
                if lev.ignored(sx, sy) {
                    continue;
                }
                let (Some(i), Some(gx), Some(gy)) =
                    (lev.img.sample(sx, sy), lev.gx.sample(sx, sy), lev.gy.sample(sx, sy))
                else {
                    continue;
                };
                // Chain rule through the warp: d(frame px)/d(template).
                let iw = 1.0 / w;
                let dxx = (hl[0] - sx * hl[6]) * iw;
                let dxy = (hl[1] - sx * hl[7]) * iw;
                let dyx = (hl[3] - sy * hl[6]) * iw;
                let dyy = (hl[4] - sy * hl[7]) * iw;
                p.valid = true;
                p.iw = i;
                p.gtx = (gx as f64 * dxx + gy as f64 * dyx) as f32;
                p.gty = (gx as f64 * dxy + gy as f64 * dyy) as f32;
                count += 1;
                // Statistics over the pixels that still have a vote: an
                // occluder that has been weighted out must not set the
                // gain and bias the survivors are judged against (it did,
                // and it bent the fit).
                wsum += prev_w;
                sum += prev_w * i as f64;
                sum2 += prev_w * (i as f64) * (i as f64);
            }
            (count, wsum, sum, sum2)
        })
        .reduce(|| (0, 0.0, 0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2, a.3 + b.3));
    if count == 0 || wsum <= 0.0 {
        return 0;
    }
    let mean = sum / wsum;
    let var = (sum2 / wsum - mean * mean).max(0.0);
    let inv = (1.0 / var.sqrt().max(1e-6)) as f32;
    let mean = mean as f32;
    s.px.par_iter_mut().for_each(|p| {
        if p.valid {
            p.iw = (p.iw - mean) * inv;
            p.gtx *= inv;
            p.gty *= inv;
        }
    });
    count
}

/// ZNCC between the template level and the warped image in `s` (which
/// `warp_level` has already normalised over the valid set; the template is
/// re-normalised over that same set so a region half off-frame is judged
/// on what is visible). Deliberately the RAW correlation: a version with
/// the smooth part of the difference taken out (shading-invariant, like
/// the solve's residual) scored a 123 px miss at 0.86 — the smooth part
/// of a misalignment is most of it.
fn zncc(tl: &TemplateLevel, s: &Scratch) -> f32 {
    let (n, st, st2, sti) = s
        .px
        .par_chunks(CHUNK)
        .zip(tl.val.par_chunks(CHUNK))
        .map(|(px, val)| {
            let (mut n, mut st, mut st2, mut sti) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for (p, &t) in px.iter().zip(val) {
                if p.valid {
                    let t = t as f64;
                    n += 1.0;
                    st += t;
                    st2 += t * t;
                    sti += t * p.iw as f64;
                }
            }
            (n, st, st2, sti)
        })
        .reduce(|| (0.0, 0.0, 0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2, a.3 + b.3));
    if n < 2.0 {
        return 0.0;
    }
    let mt = st / n;
    let vt = (st2 / n - mt * mt).max(1e-12);
    // iw is zero-mean unit-var over the valid set already.
    (sti / n / vt.sqrt()) as f32
}

/// Subtract from every residual its mean over the shading window around
/// it. Shading that changes with the view (a sheen sliding over cloth, a
/// fold's shadow, a light sweeping past) is a broad, smooth residual; at a
/// coarse level it dominates the texture, and a Gauss-Newton step reads it
/// as motion and slides — so it is removed from what the solve minimises,
/// in every phase, not only from what the weights judge.
fn centre_residuals(tl: &TemplateLevel, s: &mut Scratch) {
    let (w, h) = (tl.gw, tl.gh);
    // Only where the region dwarfs the shading window: on a mesh vertex's
    // 29 px patch the window IS the patch, and centring took the signal
    // with the shading (vertices 2.4 px off on a plane).
    if w < 4 * SHADE_RADIUS || h < 4 * SHADE_RADIUS {
        return;
    }
    let cells = w * h;
    let (mut acc, mut cnt, mut tmp) =
        (std::mem::take(&mut s.acc), std::mem::take(&mut s.cnt), std::mem::take(&mut s.tmp));
    acc.clear();
    acc.resize(cells, 0.0);
    cnt.clear();
    cnt.resize(cells, 0.0);
    for (k, p) in s.px.iter().enumerate() {
        if p.valid {
            let c = tl.gi[k] as usize;
            acc[c] += p.res;
            cnt[c] += 1.0;
        }
    }
    box_blur(&mut acc, &mut tmp, w, h, SHADE_RADIUS);
    box_blur(&mut cnt, &mut tmp, w, h, SHADE_RADIUS);
    let gi = &tl.gi;
    s.px.par_iter_mut().enumerate().for_each(|(k, p)| {
        if p.valid {
            let c = gi[k] as usize;
            if cnt[c] > 0.0 {
                p.res -= acc[c] / cnt[c];
            }
        }
    });
    s.acc = acc;
    s.cnt = cnt;
    s.tmp = tmp;
}

/// Robust weights, three ideas in one pass over the region's grid:
///
/// 1. The residual is CENTRED on its local mean (`centre_residuals`, in
///    every iteration of every phase). Shading that changes with the view
///    (a sheen sliding over cloth, a fold's shadow) is a broad, smooth
///    residual that a raw magnitude test read as a giant occluder — it
///    declared every frame of a cloth shot covered — and that a raw
///    Gauss-Newton step reads as motion.
/// 2. The magnitude is BLURRED before the Tukey cut: a misaligned edge is
///    a thin line that blurs away, an occluder a solid patch that does not.
///    Rejecting edges is rejecting the position signal.
/// 3. A LOCAL VARIANCE gate: where the warped picture has no structure at
///    all while the template has some (a flat grey ball, a hand, a card)
///    the pixels lose their vote whatever their residual — a smooth
///    occluder has a smooth residual, and idea 1 alone waved the ball
///    through. Foreign TEXTURE keeps its variance and is left to idea 2.
///    Not a correlation gate: on fine cloth texture the correlation length
///    is a couple of pixels, the folds' parallax misaligns that much
///    everywhere, and a correlation test threw away 90% of a region whose
///    coarse structure matched fine. Windows where the template itself is
///    flat get no opinion (nothing to align there).
fn robust_weights(tl: &TemplateLevel, s: &mut Scratch, radius: usize) {
    let (w, h) = (tl.gw, tl.gh);
    if w == 0 || h == 0 {
        s.px.iter_mut().for_each(|p| p.wt = 1.0);
        return;
    }
    let cells = w * h;
    let mut tmp = std::mem::take(&mut s.tmp);
    // The variance gate, on the level's first weighted iteration: per-cell
    // sums of count, template, image, T², I², blurred over the shading
    // window.
    if s.gate.len() != cells {
        const F: usize = 5;
        let mut g = vec![0.0f32; cells * F];
        for (k, p) in s.px.iter().enumerate() {
            if p.valid {
                let c = tl.gi[k] as usize;
                let t = tl.val[k];
                let i = p.iw;
                let row = &mut g[c * F..c * F + F];
                row[0] += 1.0;
                row[1] += t;
                row[2] += i;
                row[3] += t * t;
                row[4] += i * i;
            }
        }
        let mut field = vec![0.0f32; cells];
        let mut blurred = vec![0.0f32; cells * F];
        for f in 0..F {
            for c in 0..cells {
                field[c] = g[c * F + f];
            }
            box_blur(&mut field, &mut tmp, w, h, SHADE_RADIUS);
            for c in 0..cells {
                blurred[c * F + f] = field[c];
            }
        }
        s.gate.clear();
        s.gate.resize(cells, 1.0);
        for c in 0..cells {
            let b = &blurred[c * F..c * F + F];
            let n = b[0];
            if n <= 0.0 {
                continue;
            }
            let mt = b[1] / n;
            let mi = b[2] / n;
            let vt = (b[3] / n - mt * mt).max(0.0);
            let vi = (b[4] / n - mi * mi).max(0.0);
            if vt < FLAT_VAR {
                continue; // nothing to align here: no opinion
            }
            let ratio = vi / vt;
            s.gate[c] = ((ratio - 0.04) / 0.16).clamp(0.0, 1.0);
        }
    }
    let gate = std::mem::take(&mut s.gate);
    // The residual arrives centred on its local mean (`centre_residuals`);
    // its blurred magnitude decides the Tukey cut.
    let mut acc = std::mem::take(&mut s.acc);
    let mut cnt = std::mem::take(&mut s.cnt);
    acc.clear();
    acc.resize(cells, 0.0);
    cnt.clear();
    cnt.resize(cells, 0.0);
    for (k, p) in s.px.iter().enumerate() {
        if p.valid {
            let c = tl.gi[k] as usize;
            acc[c] += p.res.abs();
            cnt[c] += 1.0;
        }
    }
    box_blur(&mut acc, &mut tmp, w, h, radius);
    box_blur(&mut cnt, &mut tmp, w, h, radius);
    let mut vals: Vec<f32> = Vec::with_capacity(tl.len());
    // The blurred magnitude rides in `wt` for the cutoff pass.
    for (k, p) in s.px.iter_mut().enumerate() {
        if p.valid {
            let c = tl.gi[k] as usize;
            let b = if cnt[c] > 0.0 { acc[c] / cnt[c] } else { 0.0 };
            p.wt = b;
            vals.push(b);
        }
    }
    s.acc = acc;
    s.cnt = cnt;
    s.tmp = tmp;
    if vals.len() < 8 {
        s.px.iter_mut().for_each(|p| p.wt = 1.0);
        return;
    }
    let mid = vals.len() / 2;
    let med = *vals.select_nth_unstable_by(mid, |a, b| a.total_cmp(b)).1;
    let mut dev: Vec<f32> = vals.iter().map(|v| (v - med).abs()).collect();
    let mad = *dev.select_nth_unstable_by(mid, |a, b| a.total_cmp(b)).1;
    // Cutoff: well past the inliers' spread, but never so high that a
    // patch of foreign texture (two unit-variance signals: |r| ≈ 1.1 on
    // average) gets a vote, and never so low that resampling noise on an
    // edge loses one.
    let c = (med + 5.0 * 1.4826 * mad).clamp(0.3, 1.0);
    let gi = &tl.gi;
    s.px.par_iter_mut().enumerate().for_each(|(k, p)| {
        p.wt = if !p.valid {
            1.0
        } else {
            let tukey = if p.wt < c {
                let q = 1.0 - (p.wt / c) * (p.wt / c);
                q * q
            } else {
                0.0
            };
            tukey * gate[gi[k] as usize]
        };
    });
    s.gate = gate;
}

/// Solve one level from `hl` (template → level pixels). Returns the refined
/// warp and the iteration count.
#[allow(clippy::too_many_arguments)]
fn solve_level(
    tl: &TemplateLevel,
    lev: &Level,
    mut hl: H,
    opts: &SolveOpts,
    eps_level: f64,
    blur_radius: usize,
    start_weighted: bool,
    s: &mut Scratch,
) -> (H, usize) {
    let gens = opts.model.generators();
    let n = gens.len();
    let mut iters = 0;
    s.reset(tl.len());
    // Phase 1 converges with every pixel voting; phase 2 (robust only)
    // re-converges with the occluders weighted out. Weighting from the
    // start would judge residuals that are mostly misalignment — except
    // on a level that inherits a converged coarser one, which must NOT
    // begin by letting the occluder pull the fit off again.
    let mut weighted = start_weighted && opts.robust;
    // Weighted residual energy of the previous iteration, to notice a fit
    // that has stopped improving: on a surface that is not quite a plane
    // the step never falls under `eps` and every level ran to its cap.
    let mut last_energy = f64::INFINITY;
    let mut stalled = 0;
    for it in 0..opts.max_iters {
        iters = it + 1;
        let count = warp_level(tl, lev, &hl, s, weighted);
        if count < MIN_LEVEL_PIXELS {
            break;
        }
        // The template's own statistics over the same weighted set, so
        // both sides are normalised over the pixels being compared.
        let (wsum, ts, ts2) = s
            .px
            .par_chunks(CHUNK)
            .zip(tl.val.par_chunks(CHUNK))
            .map(|(px, val)| {
                let (mut wsum, mut ts, mut ts2) = (0.0f64, 0.0f64, 0.0f64);
                for (p, &t) in px.iter().zip(val) {
                    if p.valid {
                        let wk = if weighted { p.wt as f64 } else { 1.0 };
                        let t = t as f64;
                        wsum += wk;
                        ts += wk * t;
                        ts2 += wk * t * t;
                    }
                }
                (wsum, ts, ts2)
            })
            .reduce(|| (0.0, 0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2));
        let t_mean = if wsum > 0.0 { ts / wsum } else { 0.0 };
        let t_var = if wsum > 0.0 { (ts2 / wsum - t_mean * t_mean).max(1e-12) } else { 1.0 };
        let t_inv = (1.0 / t_var.sqrt()) as f32;
        let t_mean = t_mean as f32;
        s.px.par_chunks_mut(CHUNK).zip(tl.val.par_chunks(CHUNK)).for_each(|(px, val)| {
            for (p, &t) in px.iter_mut().zip(val) {
                if p.valid {
                    p.res = p.iw - (t - t_mean) * t_inv;
                }
            }
        });
        centre_residuals(tl, s);
        if weighted {
            robust_weights(tl, s, blur_radius);
        } else {
            s.px.par_iter_mut().for_each(|p| p.wt = 1.0);
        }
        // Normal equations, accumulated a chunk at a time.
        let ((mut a, b), energy) = s
            .px
            .par_chunks(CHUNK)
            .zip(tl.xs.par_chunks(CHUNK))
            .zip(tl.ys.par_chunks(CHUNK))
            .zip(tl.gx.par_chunks(CHUNK))
            .zip(tl.gy.par_chunks(CHUNK))
            .map(|((((px, xs), ys), tgx), tgy)| {
                let mut a = vec![0.0f64; n * n];
                let mut b = vec![0.0f64; n];
                let mut jrow = vec![0.0f64; n];
                let mut energy = 0.0f64;
                for (k, p) in px.iter().enumerate() {
                    if !p.valid || p.wt == 0.0 {
                        continue;
                    }
                    let wk = p.wt as f64;
                    energy += wk * (p.res as f64) * (p.res as f64);
                    let x = xs[k] as f64;
                    let y = ys[k] as f64;
                    // ESM: the mean of the template and warped-image gradients.
                    let gx = 0.5 * ((tgx[k] * t_inv) as f64 + p.gtx as f64);
                    let gy = 0.5 * ((tgy[k] * t_inv) as f64 + p.gty as f64);
                    for (i, g) in gens.iter().enumerate() {
                        let (du, dv) = generator_flow(g, x, y);
                        jrow[i] = gx * du + gy * dv;
                    }
                    let r = p.res as f64;
                    for i in 0..n {
                        let ji = jrow[i] * wk;
                        b[i] += ji * r;
                        for j in i..n {
                            a[i * n + j] += ji * jrow[j];
                        }
                    }
                }
                ((a, b), energy)
            })
            .reduce(
                || ((vec![0.0f64; n * n], vec![0.0f64; n]), 0.0),
                |((mut a1, mut b1), e1), ((a2, b2), e2)| {
                    for (x, y) in a1.iter_mut().zip(&a2) {
                        *x += y;
                    }
                    for (x, y) in b1.iter_mut().zip(&b2) {
                        *x += y;
                    }
                    ((a1, b1), e1 + e2)
                },
            );
        // Stalled: two iterations without a thousandth of improvement.
        // (The weights change the energy's meaning between phases, so the
        // count restarts when the weighted phase begins.)
        if energy > last_energy * 0.999 {
            stalled += 1;
        } else {
            stalled = 0;
        }
        last_energy = energy;
        for i in 0..n {
            for j in 0..i {
                a[i * n + j] = a[j * n + i];
            }
            // A whisper of Levenberg damping so a flat direction cannot
            // blow the step up.
            a[i * n + i] += 1e-9 * (1.0 + a[i * n + i]);
        }
        let mut bm: Vec<f64> = b.iter().map(|v| -v).collect();
        let Some(delta) = solve_linear(&mut a, &mut bm, n) else { break };
        // H ← H · (I + Σ δ G)
        let mut inc = IDENTITY;
        for (i, g) in gens.iter().enumerate() {
            for e in 0..9 {
                inc[e] += delta[i] * g[e];
            }
        }
        hl = h_norm(&h_mul(&hl, &inc));
        // Step size in template units: a translation parameter moves the
        // region by exactly δ; the others by at most that over the unit box.
        let step = delta.iter().fold(0.0f64, |m, d| m.max(d.abs()));
        let converged = step < eps_level || stalled >= 2 || it + 1 >= opts.max_iters / 2;
        if converged {
            if opts.robust && !weighted {
                weighted = true;
                last_energy = f64::INFINITY;
                stalled = 0;
                continue;
            }
            if step < eps_level || stalled >= 2 {
                break;
            }
        }
    }
    (hl, iters)
}

/// Align `tpl` to `cur` starting from `init` (template-frame pixels → cur
/// pixels), coarse to fine.
pub fn solve(tpl: &Template, cur: &Pyramid, init: &H, opts: &SolveOpts) -> Fit {
    let mut s = Scratch::default();
    // Template coordinates → full-res pixels.
    let mut h = h_mul(init, &tpl.norm_inv);
    let mut iters = 0;
    let nl = tpl.levels.len().min(cur.levels.len());
    if nl == 0 {
        return Fit { h: *init, score: 0.0, coverage: 0.0, iters: 0, weights: None, rejected: 0.0 };
    }
    for l in (0..nl).rev() {
        let f = (1u64 << l) as f64;
        let down = h_scale(1.0 / f);
        let hl = h_mul(&down, &h);
        // A template unit is `scale / f` level pixels.
        let eps_level = opts.eps_px / tpl.scale;
        let radius = if l + 1 == nl { 1 } else { 2 };
        let (hl, it) =
            solve_level(&tpl.levels[l], &cur.levels[l], hl, opts, eps_level, radius, l + 1 < nl, &mut s);
        iters += it;
        h = h_mul(&h_scale(f), &hl);
    }
    // The finest level's weights, before they are refreshed for the score.
    let tl0 = &tpl.levels[0];
    let rejected = if opts.robust && s.px.len() == tl0.len() {
        let (mut n, mut out) = (0usize, 0usize);
        for p in &s.px {
            if p.valid {
                n += 1;
                if p.wt < 0.5 {
                    out += 1;
                }
            }
        }
        if n > 0 { out as f32 / n as f32 } else { 0.0 }
    } else {
        0.0
    };
    let weights = (opts.robust && s.px.len() == tl0.len() && tl0.gw > 0).then(|| {
        let mut cells = vec![-1.0f32; tl0.gw * tl0.gh];
        for (k, p) in s.px.iter().enumerate() {
            if p.valid {
                cells[tl0.gi[k] as usize] = p.wt;
            }
        }
        WeightMap { ox: tl0.ox, oy: tl0.oy, w: tl0.gw, h: tl0.gh, cells }
    });
    // Score at the finest level actually used.
    let count = warp_level(tl0, &cur.levels[0], &h, &mut s, false);
    let score = if count >= MIN_LEVEL_PIXELS { zncc(tl0, &s) } else { 0.0 };
    let coverage = count as f32 / tl0.len().max(1) as f32;
    Fit { h: h_norm(&h_mul(&h, &tpl.norm)), score, coverage, iters, weights, rejected }
}

/// Score a warp without refining it.
pub fn score(tpl: &Template, cur: &Pyramid, h: &H) -> (f32, f32) {
    if tpl.levels.is_empty() {
        return (0.0, 0.0);
    }
    let mut s = Scratch::default();
    let hh = h_mul(h, &tpl.norm_inv);
    let count = warp_level(&tpl.levels[0], &cur.levels[0], &hh, &mut s, false);
    let score = if count >= MIN_LEVEL_PIXELS { zncc(&tpl.levels[0], &s) } else { 0.0 };
    (score, count as f32 / tpl.levels[0].len().max(1) as f32)
}

/// Exhaustive translation search at the coarsest level: the escape hatch
/// for a jump the pyramid cannot reach. Returns the best full-res warp
/// found within ±`radius` coarse pixels of `init`, and its coarse score.
pub fn coarse_search(tpl: &Template, cur: &Pyramid, init: &H, radius: usize) -> (H, f32) {
    if tpl.levels.is_empty() || cur.levels.is_empty() {
        return (*init, 0.0);
    }
    let l = tpl.levels.len().min(cur.levels.len()).saturating_sub(1);
    let f = (1u64 << l) as f64;
    let tl = &tpl.levels[l];
    let lev = &cur.levels[l];
    let base = h_mul(&h_scale(1.0 / f), &h_mul(init, &tpl.norm_inv));
    let r = radius as i64;
    // Every shift is independent: one task per row of shifts.
    let best = (-r..=r)
        .into_par_iter()
        .map(|dy| {
            let mut s = Scratch::default();
            let mut best = (0.0f64, 0.0f64, f32::NEG_INFINITY);
            for dx in -r..=r {
                let hl = h_mul(&h_translation(dx as f64, dy as f64), &base);
                let count = warp_level(tl, lev, &hl, &mut s, false);
                if count < MIN_LEVEL_PIXELS || (count as f32) < 0.5 * tl.len() as f32 {
                    continue;
                }
                let sc = zncc(tl, &s);
                if sc > best.2 {
                    best = (dx as f64, dy as f64, sc);
                }
            }
            best
        })
        .reduce(|| (0.0, 0.0, f32::NEG_INFINITY), |a, b| if b.2 > a.2 { b } else { a });
    if best.2 == f32::NEG_INFINITY {
        return (*init, 0.0);
    }
    let h = h_mul(&h_translation(best.0 * f, best.1 * f), init);
    (h_norm(&h), best.2)
}

// ---------------------------------------------------------------------------
// Mesh
// ---------------------------------------------------------------------------

/// A triangle mesh over the region, in the REFERENCE frame's pixels.
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Mesh {
    pub verts: Vec<[f32; 2]>,
    pub tris: Vec<[u32; 3]>,
}

impl Mesh {
    /// A grid of `cells` cells along the polygon's longer side, keeping
    /// every triangle whose centroid the polygon contains.
    pub fn grid(polygon: &[[f64; 2]], cells: usize) -> Mesh {
        let cells = cells.clamp(1, 200);
        let (lo, hi) = poly_bbox(polygon);
        let (w, h) = (hi[0] - lo[0], hi[1] - lo[1]);
        if !(w > 0.0 && h > 0.0) {
            return Mesh::default();
        }
        let (cols, rows) = if w >= h {
            (cells, ((cells as f64 * h / w).round() as usize).max(1))
        } else {
            (((cells as f64 * w / h).round() as usize).max(1), cells)
        };
        let vert = |i: usize, j: usize| -> [f64; 2] {
            [lo[0] + w * i as f64 / cols as f64, lo[1] + h * j as f64 / rows as f64]
        };
        let idx = |i: usize, j: usize| (j * (cols + 1) + i) as u32;
        let mut tris_all: Vec<[u32; 3]> = Vec::new();
        for j in 0..rows {
            for i in 0..cols {
                // Alternate the diagonal so the mesh has no grain.
                let (a, b, c, d) = (idx(i, j), idx(i + 1, j), idx(i, j + 1), idx(i + 1, j + 1));
                if (i + j) % 2 == 0 {
                    tris_all.push([a, b, d]);
                    tris_all.push([a, d, c]);
                } else {
                    tris_all.push([a, b, c]);
                    tris_all.push([b, d, c]);
                }
            }
        }
        let all_verts: Vec<[f64; 2]> =
            (0..=rows).flat_map(|j| (0..=cols).map(move |i| vert(i, j))).collect();
        let mut keep = vec![false; all_verts.len()];
        let mut tris = Vec::new();
        for t in tris_all {
            let c = [
                (all_verts[t[0] as usize][0] + all_verts[t[1] as usize][0] + all_verts[t[2] as usize][0]) / 3.0,
                (all_verts[t[0] as usize][1] + all_verts[t[1] as usize][1] + all_verts[t[2] as usize][1]) / 3.0,
            ];
            if poly_contains(polygon, c[0], c[1]) {
                for &v in &t {
                    keep[v as usize] = true;
                }
                tris.push(t);
            }
        }
        let mut remap = vec![u32::MAX; all_verts.len()];
        let mut verts = Vec::new();
        for (i, k) in keep.iter().enumerate() {
            if *k {
                remap[i] = verts.len() as u32;
                verts.push([all_verts[i][0] as f32, all_verts[i][1] as f32]);
            }
        }
        for t in tris.iter_mut() {
            for v in t.iter_mut() {
                *v = remap[*v as usize];
            }
        }
        Mesh { verts, tris }
    }

    /// Vertex adjacency from the triangle edges.
    pub fn neighbours(&self) -> Vec<Vec<u32>> {
        let mut n: Vec<Vec<u32>> = vec![Vec::new(); self.verts.len()];
        for t in &self.tris {
            for k in 0..3 {
                let (a, b) = (t[k], t[(k + 1) % 3]);
                if !n[a as usize].contains(&b) {
                    n[a as usize].push(b);
                }
                if !n[b as usize].contains(&a) {
                    n[b as usize].push(a);
                }
            }
        }
        n
    }
}

/// Mesh refinement settings.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MeshOpts {
    /// Cells along the region's longer side.
    pub cells: usize,
    /// Half-size of the patch each vertex aligns, in pixels.
    pub patch: usize,
    /// Neighbour pull in the smoothing pass (0 = raw per-vertex flow).
    pub smooth: f32,
}

impl Default for MeshOpts {
    fn default() -> Self {
        Self { cells: 12, patch: 14, smooth: 0.6 }
    }
}

/// A vertex's alignment patch, cut from the anchor frame.
struct Patch {
    tpl: Template,
    /// Position in the anchor frame.
    at: [f64; 2],
    /// Texture measure of the patch (0 = flat: this vertex only follows).
    conf: f32,
}

struct MeshTrack {
    mesh: Mesh,
    neighbours: Vec<Vec<u32>>,
    patches: Vec<Patch>,
    /// Vertex positions in the anchor frame (= reference until a re-anchor).
    anchor_pos: Vec<[f64; 2]>,
    /// Where each vertex was last frame, and that frame's anchor → frame
    /// warp: the seed for the next local solve is the vertex's OWN last
    /// position carried by the plane's frame-to-frame motion, not the
    /// plane's prediction (which wobbles on a surface that is not one).
    last: Option<(Vec<[f64; 2]>, H)>,
    /// Why vertices did not vote last step: flat, occluded, coverage,
    /// score, too far, no gain. Diagnostics for the probes.
    pub(crate) rejects: [usize; 6],
}

/// The texture measure at which a patch's vote reaches full weight; below
/// it the vote scales down, at 0 the vertex only follows its neighbours.
const PATCH_FLAT: f64 = 0.05;
/// A vertex leaves the plane only when its patch matches better off it by
/// at least this much.
const PATCH_GAIN: f32 = 0.002;

impl MeshTrack {
    fn new(mesh: Mesh, pyr: &Pyramid, opts: &MeshOpts) -> Self {
        let neighbours = mesh.neighbours();
        let anchor_pos: Vec<[f64; 2]> = mesh.verts.iter().map(|v| [v[0] as f64, v[1] as f64]).collect();
        let patches = Self::cut_patches(pyr, &anchor_pos, opts);
        Self { mesh, neighbours, patches, anchor_pos, last: None, rejects: [0; 6] }
    }

    fn cut_patches(pyr: &Pyramid, at: &[[f64; 2]], opts: &MeshOpts) -> Vec<Patch> {
        let r = opts.patch.max(3) as f64;
        at.iter()
            .map(|p| {
                let poly = [[p[0] - r, p[1] - r], [p[0] + r, p[1] - r], [p[0] + r, p[1] + r], [p[0] - r, p[1] + r]];
                let tpl = Template::build(pyr, &poly, 6.0);
                let tex = if tpl.levels.is_empty() { 0.0 } else { tpl.texture() };
                let conf = ((tex / PATCH_FLAT).min(1.0)) as f32;
                Patch { tpl, at: *p, conf }
            })
            .collect()
    }

    /// Re-cut every patch from the current frame at the positions just
    /// tracked: the mesh's own re-anchor.
    fn reanchor(
        &mut self,
        pyr: &Pyramid,
        positions: &[[f64; 2]],
        opts: &MeshOpts,
        keep: &dyn Fn(f64, f64) -> bool,
    ) {
        self.anchor_pos = positions.to_vec();
        self.patches = Self::cut_patches(pyr, positions, opts);
        // A vertex under whatever was in front of the plane gets a patch
        // of that thing: no vote until the next re-anchor.
        for p in self.patches.iter_mut() {
            if !keep(p.at[0], p.at[1]) {
                p.conf = 0.0;
            }
        }
        self.last = Some((positions.to_vec(), IDENTITY));
    }

    /// Per-vertex local alignment off the planar prediction `h` (anchor →
    /// frame), then a confidence-weighted Laplacian smooth of the residuals.
    /// Returns the vertex positions in the frame and the confidence used.
    fn step(
        &mut self,
        pyr: &Pyramid,
        h: &H,
        opts: &MeshOpts,
        occlusion: Option<&WeightMap>,
    ) -> (Vec<[f64; 2]>, Vec<f32>) {
        let n = self.anchor_pos.len();
        let mut raw = vec![[0.0f64; 2]; n];
        let mut conf = vec![0.0f32; n];
        let sopts = SolveOpts { model: Model::Translation, max_iters: 12, eps_px: 0.02, robust: false };
        let pred: Vec<(f64, f64)> = self.anchor_pos.iter().map(|p| h_apply(h, p[0], p[1])).collect();
        // Frame-to-frame planar motion, to carry last frame's positions.
        let carry = self.last.as_ref().and_then(|(_, hl)| h_inv(hl).map(|inv| h_mul(h, &inv)));
        let r = opts.patch.max(3) as f64;
        // Every vertex is its own small solve: one rayon task each.
        let last = self.last.as_ref();
        let results: Vec<(Option<([f64; 2], f32)>, usize)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let patch = &self.patches[i];
                if patch.conf <= 0.0 || patch.tpl.levels.is_empty() {
                    return (None, 0);
                }
                // The plane's solve already knows where the plane is NOT: a
                // vertex whose patch it weighted out has nothing to align.
                if let Some(m) = occlusion {
                    if m.around(patch.at[0], patch.at[1], r) < 0.5 {
                        return (None, 1);
                    }
                }
                // Seed: the vertex's own last position carried by the plane's
                // motion since (a bulge does not vanish between frames);
                // the plane's prediction on the first frame.
                let seed = match (last, &carry) {
                    (Some((pos, _)), Some(c)) => h_apply(c, pos[i][0], pos[i][1]),
                    _ => pred[i],
                };
                let init = h_mul(&h_translation(seed.0 - pred[i].0, seed.1 - pred[i].1), h);
                let fit = solve(&patch.tpl, pyr, &init, &sopts);
                // A patch has to be nearly all in frame and well matched to
                // vote; otherwise the vertex follows its neighbours. Half a
                // patch at the frame's edge measured 2 px off on a plane.
                if fit.coverage < 0.9 {
                    return (None, 2);
                }
                if fit.score < 0.6 {
                    return (None, 3);
                }
                let (fx, fy) = h_apply(&fit.h, patch.at[0], patch.at[1]);
                // A vertex that claims to have moved further than its patch is
                // wide has matched something else.
                let d = ((fx - seed.0).powi(2) + (fy - seed.1).powi(2)).sqrt();
                if d > r {
                    return (None, 4);
                }
                // Leaving the plane needs a reason the pixels can show: the
                // patch must match better where the local solve put it than
                // where the plane predicts it. Otherwise the plane wins — a
                // small patch on fine repetitive texture settles into the
                // nearest wrong minimum with full confidence.
                let (at_plane, _) = score(&patch.tpl, pyr, h);
                if fit.score < at_plane + PATCH_GAIN {
                    return (None, 5);
                }
                (Some(([fx - pred[i].0, fy - pred[i].1], patch.conf * fit.score)), 0)
            })
            .collect();
        self.rejects = [0; 6];
        for (i, (vote, why)) in results.into_iter().enumerate() {
            match vote {
                Some((d, c)) => {
                    raw[i] = d;
                    conf[i] = c;
                }
                None => self.rejects[why] += 1,
            }
        }
        // Smooth: each vertex leans on its neighbours in proportion to how
        // little it trusts its own patch.
        let lambda = opts.smooth.max(0.0) as f64;
        let mut cur = raw.clone();
        for _ in 0..6 {
            let mut next = cur.clone();
            for i in 0..n {
                let nb = &self.neighbours[i];
                if nb.is_empty() {
                    continue;
                }
                let c = conf[i] as f64;
                let mut sx = c * raw[i][0];
                let mut sy = c * raw[i][1];
                for &j in nb {
                    sx += lambda * cur[j as usize][0];
                    sy += lambda * cur[j as usize][1];
                }
                let d = c + lambda * nb.len() as f64;
                if d > 1e-9 {
                    next[i] = [sx / d, sy / d];
                }
            }
            cur = next;
        }
        let out: Vec<[f64; 2]> = (0..n).map(|i| [pred[i].0 + cur[i][0], pred[i].1 + cur[i][1]]).collect();
        self.last = Some((out.clone(), *h));
        (out, conf)
    }
}

// ---------------------------------------------------------------------------
// The sequence tracker
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TrackOpts {
    pub model: Model,
    /// Below this score the template is re-cut from the current frame
    /// (appearance has drifted: perspective, lighting, blur). 0 never
    /// re-anchors; 1 tracks frame to frame.
    pub reanchor_below: f32,
    /// Below this score the frame is LOST: the previous warp is held and
    /// nothing is learned from the frame.
    pub lost_below: f32,
    /// Track a mesh as well as the plane.
    pub mesh: Option<MeshOpts>,
    /// Track the detail layer (high-pass of this radius, per level) rather
    /// than the raw picture: shading that moves with the view cancels.
    pub detail: Option<usize>,
    /// How far (full-res pixels) the re-acquiring coarse search reaches
    /// from the prediction. 0 = the default, `COARSE_RADIUS` pixels at the
    /// coarsest level — which is all a region that big needs, and not
    /// enough for a small point pattern moving further than it is wide.
    #[cfg_attr(feature = "serde", serde(default))]
    pub search: f64,
    /// A fit scoring this well is trusted wherever it lands. 0 = the
    /// default, `CONFIDENT`, right for a region; a small pattern needs it
    /// higher, because a patch of a few dozen pixels has lookalikes that
    /// score 0.85 a region never meets.
    #[cfg_attr(feature = "serde", serde(default))]
    pub confident: f32,
}

impl Default for TrackOpts {
    fn default() -> Self {
        Self { model: Model::Perspective, reanchor_below: 0.8, lost_below: 0.35, mesh: None, detail: None, search: 0.0, confident: 0.0 }
    }
}

/// The coarse search's reach at the coarsest level when `TrackOpts.search`
/// asks for no more.
const COARSE_RADIUS: usize = 12;

/// Detail radius when the layer is on, pixels at each level. Off by
/// default: on clean footage it costs accuracy (0.15 → 0.44 px on the
/// synthetic shot, and the band-passed scores sit under `CONFIDENT`), and
/// the shading problem it was written for is handled in the weights.
pub const DETAIL_RADIUS: usize = 6;

/// Why a frame gave no trusted answer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lost {
    /// The match score fell under `lost_below`.
    Score,
    /// More than half the region was weighted out.
    Covered,
    /// The fit landed far from the prediction without a confident score.
    Jump,
    /// The warp went non-finite.
    Diverged,
}

impl Lost {
    pub fn label(self) -> &'static str {
        match self {
            Lost::Score => "score",
            Lost::Covered => "covered",
            Lost::Jump => "jump",
            Lost::Diverged => "diverged",
        }
    }
}

/// One frame's answer.
#[derive(Clone, Debug)]
pub struct FrameResult {
    /// Reference frame pixels → this frame's pixels.
    pub h: H,
    pub score: f32,
    pub coverage: f32,
    pub lost: bool,
    /// Why, when lost.
    pub why: Option<Lost>,
    pub reanchored: bool,
    /// Fraction of the region the solve weighted out (occlusion).
    pub rejected: f32,
    /// How far the fit landed from the velocity's prediction, in units of
    /// the region's extent (the consistency test's input).
    pub jump: f32,
    /// Mesh vertex positions in this frame (empty without a mesh).
    pub mesh: Vec<[f32; 2]>,
    pub mesh_conf: Vec<f32>,
}

/// Coarsest level: the region must still be this wide.
const MIN_SIDE: f64 = 24.0;
static TRACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
/// Pyramid depth cap.
const MAX_LEVELS: usize = 6;

pub struct Tracker {
    opts: TrackOpts,
    polygon_ref: Vec<[f64; 2]>,
    tpl: Template,
    /// Reference → anchor (identity until a re-anchor).
    anchor: H,
    /// The last two frames' reference → frame warps, for the prediction.
    prev: Option<H>,
    prev2: Option<H>,
    /// Last frame's score, to tell a drift from a dip.
    last_score: Option<f32>,
    /// Consecutive frames without a trusted answer.
    lost_run: usize,
    mesh: Option<MeshTrack>,
    n_levels: usize,
}

/// A fit this good is taken whatever the prediction said.
const CONFIDENT: f32 = 0.85;
/// An unconfident fit must land within this fraction of the region's
/// extent of where the velocity put it.
const CONSISTENT: f64 = 0.15;
/// Lost frames coast on the velocity this long, then hold still.
const COAST_MAX: usize = 12;

impl Tracker {
    /// Levels a frame of this size gets.
    pub fn levels_for(w: usize, h: usize) -> usize {
        let mut n = 1;
        let mut s = w.min(h);
        while s >= 2 * MIN_SIDE as usize && n < MAX_LEVELS {
            s /= 2;
            n += 1;
        }
        n
    }

    /// Start a track on `reference` over `polygon` (reference pixels).
    pub fn new(reference: Plane, polygon: Vec<[f64; 2]>, opts: TrackOpts) -> Self {
        Self::new_masked(reference, None, polygon, opts)
    }

    /// `new` with the reference frame's ignore matte (frame-sized, 1 = not
    /// the plane): what it covers is never cut into the template.
    pub fn new_masked(reference: Plane, ignore: Option<Plane>, polygon: Vec<[f64; 2]>, opts: TrackOpts) -> Self {
        let n_levels = Self::levels_for(reference.w, reference.h);
        let mut pyr = Pyramid::build_with(reference, n_levels, opts.detail);
        if let Some(m) = ignore {
            pyr = pyr.with_ignore(m);
        }
        let tpl = Template::build(&pyr, &polygon, MIN_SIDE);
        let mesh = opts.mesh.map(|m| MeshTrack::new(Mesh::grid(&polygon, m.cells), &pyr, &m));
        Self {
            opts,
            polygon_ref: polygon,
            tpl,
            anchor: IDENTITY,
            prev: None,
            prev2: None,
            last_score: None,
            lost_run: 0,
            mesh,
            n_levels,
        }
    }

    pub fn mesh(&self) -> Option<&Mesh> {
        self.mesh.as_ref().map(|m| &m.mesh)
    }

    /// Whether the region left anything to track: an ignore matte over the
    /// whole region at the reference frame leaves no template.
    pub fn has_template(&self) -> bool {
        !self.tpl.levels.is_empty()
    }

    /// Move the trust bar (`TrackOpts.confident`) between frames: a point
    /// tracker judges a fit against the scores its pattern has been making.
    pub fn set_confident(&mut self, c: f32) {
        self.opts.confident = c;
    }

    pub fn polygon(&self) -> &[[f64; 2]] {
        &self.polygon_ref
    }

    /// The region's corners under `h`.
    pub fn corners(&self, h: &H) -> Vec<[f64; 2]> {
        poly_map(&self.polygon_ref, h)
    }

    /// Track one more frame. Frames must arrive in order (either direction
    /// — a backward track is a forward track over a reversed list).
    /// `WOBBLE_PLANAR_TRACE=1` prints where the frame's time went.
    pub fn step(&mut self, frame: Plane) -> FrameResult {
        self.step_masked(frame, None)
    }

    /// `step` with this frame's ignore matte (frame-sized, 1 = not the
    /// plane): its pixels sit out the solve, the score and any re-anchor,
    /// as if they were off the picture.
    pub fn step_masked(&mut self, frame: Plane, ignore: Option<Plane>) -> FrameResult {
        let mut pyr = self.pyramid(frame);
        if let Some(m) = ignore {
            pyr = pyr.with_ignore(m);
        }
        self.step_pyramid(&pyr)
    }

    /// The pyramid `step` builds from a frame. Trackers of the same frame
    /// size and detail setting build the SAME one, so several (a
    /// multi-point Tracker's points) share it through `step_pyramid`.
    pub fn pyramid(&self, frame: Plane) -> Pyramid {
        Pyramid::build_with(frame, self.n_levels, self.opts.detail)
    }

    /// `step` on a pyramid from `pyramid()` (of this tracker or of one with
    /// the same frame size and detail).
    pub fn step_pyramid(&mut self, pyr: &Pyramid) -> FrameResult {
        let trace = *TRACE.get_or_init(|| std::env::var_os("WOBBLE_PLANAR_TRACE").is_some());
        let t_start = web_time::Instant::now();
        let mut marks: Vec<(&'static str, f32)> = Vec::new();
        let mut mark = |name: &'static str, since: web_time::Instant| {
            if trace {
                marks.push((name, since.elapsed().as_secs_f32() * 1000.0));
            }
        };
        let sopts = SolveOpts { model: self.opts.model, ..Default::default() };
        // A frame after a lost one gets shorter solves: with the region
        // half off the picture nothing converges, and every level ran to
        // its iteration cap — a lost frame cost eight tracked ones.
        let cheap = SolveOpts { max_iters: 14, ..sopts };
        let sopts = if self.lost_run > 0 { cheap } else { sopts };
        let anchor_inv = h_inv(&self.anchor).unwrap_or(IDENTITY);
        let last = self.prev.unwrap_or(IDENTITY);
        // Where the plane would be by velocity alone (reference → frame):
        // the corners extrapolated linearly. Composing the frame-to-frame
        // homography instead compounds its perspective terms, and twelve
        // coasted frames of that sent a region a hundred thousand pixels
        // away.
        let (fw, fh) = (pyr.base().w as f64, pyr.base().h as f64);
        let coast = match (self.prev, self.prev2) {
            (Some(p1), Some(p2)) if self.polygon_ref.len() == 4 => {
                let a = poly_map(&self.polygon_ref, &p2);
                let b = poly_map(&self.polygon_ref, &p1);
                let mut dst = [[0.0f64; 2]; 4];
                let mut sane = true;
                for k in 0..4 {
                    dst[k] = [2.0 * b[k][0] - a[k][0], 2.0 * b[k][1] - a[k][1]];
                    sane &= dst[k][0].abs() < 3.0 * fw && dst[k][1].abs() < 3.0 * fh;
                }
                let src = [self.polygon_ref[0], self.polygon_ref[1], self.polygon_ref[2], self.polygon_ref[3]];
                if sane { h_from_quad(&src, &dst) } else { None }
            }
            _ => None,
        };
        let pred = coast.unwrap_or(last);
        // Past the coast a lost region is on HOLD: one solve from where it
        // was, and the coarse search every few frames — a lost frame at
        // full effort cost four times a tracked one, for the rest of the
        // shot.
        let holding = self.lost_run > COAST_MAX;
        let extent = 2.0 * self.tpl.scale;
        let corners_apart = |a: &H, b: &H| -> f64 {
            let pa = poly_map(&self.polygon_ref, a);
            let pb = poly_map(&self.polygon_ref, b);
            pa.iter()
                .zip(&pb)
                .map(|(u, v)| ((u[0] - v[0]).powi(2) + (u[1] - v[1]).powi(2)).sqrt())
                .fold(0.0, f64::max)
                / extent
        };
        let sure = if self.opts.confident > 0.0 { self.opts.confident } else { CONFIDENT };
        let confident = |f: &Fit| f.h.iter().all(|v| v.is_finite()) && f.score >= sure && f.rejected < 0.35;
        // The solve starts where the velocity says the plane is (the last
        // frame when there is none). A second solve from the last frame
        // runs only on a sign of trouble — a score that dropped, or a fit
        // that disagrees with the prediction — not on every frame under a
        // fixed score, which doubled the cost of a whole cloth shot whose
        // honest scores sit at 0.85.
        let from_prev = h_mul(&last, &anchor_inv);
        let from_pred = h_mul(&pred, &anchor_inv);
        let t = web_time::Instant::now();
        let mut best = solve(&self.tpl, &pyr, if holding { &from_prev } else { &from_pred }, &sopts);
        mark("solve", t);
        let dropped = self.last_score.is_some_and(|l| best.score < l - 0.1);
        let fitted0 = h_norm(&h_mul(&best.h, &self.anchor));
        if !holding && coast.is_some() && !confident(&best) && (dropped || corners_apart(&fitted0, &pred) > CONSISTENT) {
            let t = web_time::Instant::now();
            let f = solve(&self.tpl, &pyr, &from_prev, &sopts);
            mark("solve-prev", t);
            if f.score > best.score {
                best = f;
            }
        }
        // Trust: a confident fit is taken as it is. An unconfident one is
        // taken only where the velocity expected it — a lattice of bricks
        // has a fair-scoring minimum one period over, and that is how a
        // hand crossing the plane turned into a slide onto the next row.
        // Otherwise the frame is LOST: the plane coasts on its velocity
        // (the surface did not stop because something crossed it) while
        // the coarse search looks for it, and after `COAST_MAX` frames it
        // holds still rather than run off the picture.
        let finite = best.h.iter().all(|v| v.is_finite());
        let fitted = h_norm(&h_mul(&best.h, &self.anchor));
        let jump = corners_apart(&fitted, &pred);
        let mut why = if !finite {
            Some(Lost::Diverged)
        } else if best.score < self.opts.lost_below {
            Some(Lost::Score)
        } else if best.rejected > 0.5 && best.score < sure {
            // Half the region weighted out AND a poor match: a fit this
            // good on what is left is trusted (the weights are conservative
            // on a surface whose shading is changing).
            Some(Lost::Covered)
        } else if !confident(&best) && jump > CONSISTENT {
            Some(Lost::Jump)
        } else {
            None
        };
        if why.is_some() && (!holding || self.lost_run % 4 == 0) {
            let t = web_time::Instant::now();
            let f = (1u64 << self.tpl.levels.len().saturating_sub(1)) as f64;
            let radius = COARSE_RADIUS.max((self.opts.search / f).ceil() as usize);
            let (h0, _) = coarse_search(&self.tpl, &pyr, &h_mul(&pred, &anchor_inv), radius);
            mark("coarse", t);
            let t = web_time::Instant::now();
            let f = solve(&self.tpl, &pyr, &h0, &sopts);
            mark("solve-coarse", t);
            if confident(&f) {
                best = f;
                why = None;
            }
        }
        let lost = why.is_some();
        let fitted = h_norm(&h_mul(&best.h, &self.anchor));
        let h_ref = if !lost {
            self.lost_run = 0;
            fitted
        } else {
            self.lost_run += 1;
            if self.lost_run <= COAST_MAX {
                pred
            } else {
                last
            }
        };
        let may_reanchor = !lost && best.score < self.opts.reanchor_below && self.may_reanchor(&best);
        // The re-anchor's mask: a current-frame pixel is kept where the
        // solve's weights, over the old template's frame, kept it.
        let back = h_inv(&best.h).unwrap_or(IDENTITY);
        let weights = best.weights.clone();
        let keep = move |x: f64, y: f64| -> bool {
            match &weights {
                Some(m) => {
                    let (ax, ay) = h_apply(&back, x, y);
                    m.around(ax, ay, 1.0) >= 0.5
                }
                None => true,
            }
        };
        let mut mesh_pos = Vec::new();
        let mut mesh_conf = Vec::new();
        let mut reanchored = false;
        if let Some(m) = &mut self.mesh {
            if lost {
                mesh_pos = m.anchor_pos.iter().map(|p| {
                    let (x, y) = h_apply(&h_mul(&h_ref, &anchor_inv), p[0], p[1]);
                    [x as f32, y as f32]
                }).collect();
                mesh_conf = vec![0.0; mesh_pos.len()];
            } else {
                let mopts = self.opts.mesh.unwrap_or_default();
                let t = web_time::Instant::now();
                let (pos, conf) = m.step(&pyr, &best.h, &mopts, best.weights.as_ref());
                mark("mesh", t);
                mesh_pos = pos.iter().map(|p| [p[0] as f32, p[1] as f32]).collect();
                mesh_conf = conf;
                if may_reanchor {
                    m.reanchor(&pyr, &pos, &mopts, &keep);
                }
            }
        }
        if !lost {
            // Re-anchor on a DRIFT of appearance, never on a dip: a sudden
            // drop is something in front of the plane, and a template cut
            // then carries it along for the rest of the shot.
            if may_reanchor {
                let t = web_time::Instant::now();
                let poly = poly_map(&self.polygon_ref, &h_ref);
                let tpl = Template::build_masked(&pyr, &poly, MIN_SIDE, &keep);
                if !tpl.levels.is_empty() {
                    self.tpl = tpl;
                    self.anchor = h_ref;
                    reanchored = true;
                }
                mark("re-anchor", t);
            }
            self.prev2 = self.prev;
            self.prev = Some(h_ref);
            self.last_score = Some(best.score);
        } else if self.lost_run <= COAST_MAX && coast.is_some() {
            // Keep coasting: the velocity stands, the position advances.
            self.prev2 = self.prev;
            self.prev = Some(h_ref);
        } else {
            // Holding: no velocity to speak of.
            self.prev2 = None;
            self.prev = Some(h_ref);
        }
        if trace {
            let parts: Vec<String> = marks.iter().map(|(n, ms)| format!("{n} {ms:.0}")).collect();
            eprintln!(
                "planar: {:.0} ms = {} | iters {} px {} lost {:?}",
                t_start.elapsed().as_secs_f32() * 1000.0,
                parts.join(", "),
                best.iters,
                self.tpl.levels.first().map(|l| l.len()).unwrap_or(0),
                why
            );
        }
        FrameResult {
            h: h_ref,
            score: best.score,
            coverage: best.coverage,
            lost,
            why,
            reanchored,
            rejected: best.rejected,
            jump: jump as f32,
            mesh: mesh_pos,
            mesh_conf,
        }
    }

    /// A template may be re-cut only when the score slid there gradually
    /// and little of the region is weighted out.
    fn may_reanchor(&self, fit: &Fit) -> bool {
        // `rejected` counts the correlation gate too, so a surface whose
        // shading is changing runs at 0.3–0.5 rejected while perfectly
        // tracked; the re-anchor cuts through the weights anyway, so only
        // "too little left to cut" refuses.
        let dip = self.last_score.map(|l| l - fit.score).unwrap_or(0.0);
        dip < 0.1 && fit.rejected < 0.7
    }
}

// ---------------------------------------------------------------------------
// Synthetic footage: a textured plane under a known homography
// ---------------------------------------------------------------------------

pub mod synth {
    use super::*;

    fn hash(x: u32, y: u32, seed: u32) -> f32 {
        let mut h = x.wrapping_mul(0x8da6_b343) ^ y.wrapping_mul(0xd816_3841) ^ seed.wrapping_mul(0x9e37_79b9);
        h ^= h >> 15;
        h = h.wrapping_mul(0x2c1b_3c6d);
        h ^= h >> 12;
        h = h.wrapping_mul(0x297a_2d39);
        h ^= h >> 15;
        (h & 0xffff) as f32 / 65535.0
    }

    /// A plane with structure at every scale: value noise over five
    /// octaves plus a scatter of hard-edged rectangles.
    pub fn texture(w: usize, h: usize, seed: u32) -> Plane {
        let mut px = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let mut v = 0.0;
                let mut amp = 0.5;
                let mut cell = 64.0f32;
                for o in 0..5 {
                    let fx = x as f32 / cell;
                    let fy = y as f32 / cell;
                    let (ix, iy) = (fx.floor() as u32, fy.floor() as u32);
                    let (tx, ty) = (fx - fx.floor(), fy - fy.floor());
                    let sx = tx * tx * (3.0 - 2.0 * tx);
                    let sy = ty * ty * (3.0 - 2.0 * ty);
                    let s = seed.wrapping_add(o * 7919);
                    let a = hash(ix, iy, s);
                    let b = hash(ix + 1, iy, s);
                    let c = hash(ix, iy + 1, s);
                    let d = hash(ix + 1, iy + 1, s);
                    v += amp * ((a * (1.0 - sx) + b * sx) * (1.0 - sy) + (c * (1.0 - sx) + d * sx) * sy);
                    amp *= 0.5;
                    cell *= 0.5;
                }
                px[y * w + x] = v;
            }
        }
        // Rectangles: edges and flat areas, like print on a poster.
        for k in 0..24u32 {
            let rx = (hash(k, 1, seed) * w as f32) as usize;
            let ry = (hash(k, 2, seed) * h as f32) as usize;
            let rw = (8.0 + hash(k, 3, seed) * w as f32 * 0.15) as usize;
            let rh = (8.0 + hash(k, 4, seed) * h as f32 * 0.15) as usize;
            let val = hash(k, 5, seed);
            for y in ry..(ry + rh).min(h) {
                for x in rx..(rx + rw).min(w) {
                    px[y * w + x] = 0.7 * val + 0.3 * px[y * w + x];
                }
            }
        }
        Plane { w, h, px }
    }

    /// Rendering controls for one frame.
    #[derive(Clone, Copy, Debug)]
    pub struct RenderOpts {
        pub gain: f32,
        pub bias: f32,
        /// Uniform noise amplitude.
        pub noise: f32,
        pub seed: u32,
        /// Paint a disc (centre, radius) of foreign texture over the frame.
        pub occluder: Option<([f64; 2], f64)>,
        /// Add a smooth non-planar displacement to the plane (frame px):
        /// a bulge of this amplitude, one period across the frame.
        pub bulge: f32,
        /// A sheen: multiply by `1 + amplitude · sin(2πx/w + phase)`, a
        /// band of light that slides across the surface with the view.
        pub sheen: (f32, f32),
    }

    impl Default for RenderOpts {
        fn default() -> Self {
            Self { gain: 1.0, bias: 0.0, noise: 0.0, seed: 1, occluder: None, bulge: 0.0, sheen: (0.0, 0.0) }
        }
    }

    /// Where a bulge of amplitude `a` moves the texture point that would
    /// otherwise land at frame `(x, y)`: the displacement field
    /// `d(x, y)`, so the rendered frame shows texture from `(x, y) - d`.
    pub fn bulge_offset(a: f32, x: f64, y: f64, w: usize, h: usize) -> (f64, f64) {
        let fx = 2.0 * PI * x / w as f64;
        let fy = 2.0 * PI * y / h as f64;
        (a as f64 * fx.sin() * fy.cos(), a as f64 * (fx * 1.3).cos() * (fy * 0.7).sin())
    }

    /// Render a `w`×`h` frame of `tex` seen through `h_tex_to_frame`.
    pub fn render(tex: &Plane, h_tex_to_frame: &H, w: usize, h: usize, o: &RenderOpts) -> Plane {
        let inv = h_inv(h_tex_to_frame).expect("invertible view");
        let mut px = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let mut cx = x as f64 + 0.5;
                let mut cy = y as f64 + 0.5;
                if o.bulge != 0.0 {
                    let (dx, dy) = bulge_offset(o.bulge, cx, cy, w, h);
                    cx -= dx;
                    cy -= dy;
                }
                let (tx, ty) = h_apply(&inv, cx, cy);
                let mut v = tex.sample(tx, ty).unwrap_or(0.05);
                if let Some((c, r)) = o.occluder {
                    let (dx, dy) = (x as f64 + 0.5 - c[0], y as f64 + 0.5 - c[1]);
                    if dx * dx + dy * dy < r * r {
                        v = 0.2 + 0.6 * hash(x as u32, y as u32, o.seed ^ 0xabcd);
                    }
                }
                if o.sheen.0 != 0.0 {
                    v *= 1.0 + o.sheen.0 * (2.0 * PI as f32 * x as f32 / w as f32 + o.sheen.1).sin();
                }
                v = v * o.gain + o.bias;
                if o.noise > 0.0 {
                    v += o.noise * (hash(x as u32, y as u32, o.seed) - 0.5);
                }
                px[y * w + x] = v;
            }
        }
        Plane { w, h, px }
    }

    /// A view homography: the texture centred in the frame, then rotated,
    /// scaled, translated and tilted about the frame centre.
    pub fn view(w: usize, h: usize, rot_deg: f64, scale: f64, tx: f64, ty: f64, px: f64, py: f64) -> H {
        let (cx, cy) = (w as f64 * 0.5, h as f64 * 0.5);
        let (s, c) = (rot_deg.to_radians().sin(), rot_deg.to_radians().cos());
        let rs = [c * scale, -s * scale, 0.0, s * scale, c * scale, 0.0, px, py, 1.0];
        let to_centre = h_translation(-cx, -cy);
        let back = h_translation(cx + tx, cy + ty);
        h_norm(&h_mul(&back, &h_mul(&rs, &to_centre)))
    }

    /// The region `write_sequence` reports for its shot: a quad in the
    /// middle of the plane, frame-0 pixels.
    pub fn region(w: usize, h: usize) -> [[f64; 2]; 4] {
        [
            [w as f64 * 0.32, h as f64 * 0.30],
            [w as f64 * 0.68, h as f64 * 0.31],
            [w as f64 * 0.67, h as f64 * 0.70],
            [w as f64 * 0.33, h as f64 * 0.69],
        ]
    }

    /// Write a synthetic shot to `dir` as 16-bit greyscale PNGs
    /// (`frame.NNNN.png`) plus `truth.txt`: the region on its first line
    /// and, per frame, the texture→frame view and the bulge amplitude. The
    /// motion is `path`; `bulge` is the amplitude of a slow sinusoidal
    /// bulge (0 = a true plane), `occluder` drifts a disc of foreign
    /// texture across the region. Returns the region.
    pub fn write_sequence(
        dir: &std::path::Path,
        w: usize,
        h: usize,
        frames: usize,
        bulge: f32,
        occluder: bool,
        noise: f32,
    ) -> anyhow::Result<[[f64; 2]; 4]> {
        std::fs::create_dir_all(dir)?;
        let tex = texture(w, h, 7);
        let base = view(w, h, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
        let quad = region(w, h);
        let mut truth = String::new();
        truth.push_str(&format!(
            "quad {} {} {} {} {} {} {} {}\n",
            quad[0][0], quad[0][1], quad[1][0], quad[1][1], quad[2][0], quad[2][1], quad[3][0], quad[3][1]
        ));
        for t in 0..frames {
            let view = h_mul(&path(w, h, t), &base);
            let occ = occluder.then(|| {
                let c = [w as f64 * (0.35 + 0.4 * t as f64 / frames as f64), h as f64 * 0.5];
                (c, h as f64 * 0.09)
            });
            let amp = bulge * ((t as f32) * 0.2).sin();
            let o = RenderOpts { noise, seed: t as u32 + 1, occluder: occ, bulge: amp, ..Default::default() };
            let frame = render(&tex, &view, w, h, &o);
            let mut data = Vec::with_capacity(w * h * 2);
            for &v in &frame.px {
                let q = (v.clamp(0.0, 1.0) * 65535.0).round() as u16;
                data.extend_from_slice(&q.to_be_bytes());
            }
            let file = std::io::BufWriter::new(std::fs::File::create(dir.join(format!("frame.{t:04}.png")))?);
            let mut enc = png::Encoder::new(file, w as u32, h as u32);
            enc.set_color(png::ColorType::Grayscale);
            enc.set_depth(png::BitDepth::Sixteen);
            enc.write_header()?.write_image_data(&data)?;
            truth.push_str(&format!("{t}"));
            for v in view {
                truth.push_str(&format!(" {v}"));
            }
            truth.push_str(&format!(" {amp}\n"));
        }
        std::fs::write(dir.join("truth.txt"), truth)?;
        Ok(quad)
    }

    /// Frame-to-frame motion for a synthetic shot: reference → frame `t`.
    pub fn path(w: usize, h: usize, t: usize) -> H {
        let tf = t as f64;
        let rot = 0.4 * tf;
        let scale = 1.0 + 0.006 * tf;
        let tx = 3.0 * tf + 6.0 * (tf * 0.3).sin();
        let ty = -2.0 * tf + 4.0 * (tf * 0.21).cos() - 4.0;
        let px = 1.2e-5 * tf;
        let py = -0.8e-5 * tf;
        h_mul(&view(w, h, rot, scale, tx, ty, px, py), &h_inv(&view(w, h, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)).unwrap())
    }
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::synth::*;
    use super::*;

    const W: usize = 512;
    const HT: usize = 384;

    fn scene(seed: u32) -> Plane {
        texture(W, HT, seed)
    }

    /// The region: a quad in reference pixels.
    fn quad() -> Vec<[f64; 2]> {
        vec![[150.0, 110.0], [370.0, 118.0], [362.0, 268.0], [158.0, 262.0]]
    }

    fn corner_err(tr_h: &H, truth: &H, poly: &[[f64; 2]]) -> f64 {
        poly.iter()
            .map(|p| {
                let (a, b) = h_apply(tr_h, p[0], p[1]);
                let (c, d) = h_apply(truth, p[0], p[1]);
                ((a - c).powi(2) + (b - d).powi(2)).sqrt()
            })
            .fold(0.0, f64::max)
    }

    fn track_pair(motion: &H, model: Model, o: &RenderOpts) -> (Fit, f64) {
        let tex = scene(3);
        let base = view(W, HT, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
        let reference = render(&tex, &base, W, HT, &RenderOpts::default());
        let moved = render(&tex, &h_mul(motion, &base), W, HT, o);
        let n = Tracker::levels_for(W, HT);
        let rp = Pyramid::build(reference, n);
        let cp = Pyramid::build(moved, n);
        let tpl = Template::build(&rp, &quad(), MIN_SIDE);
        let fit = solve(&tpl, &cp, &IDENTITY, &SolveOpts { model, ..Default::default() });
        let err = corner_err(&fit.h, motion, &quad());
        (fit, err)
    }

    #[test]
    fn homography_algebra_holds() {
        let src = [[0.0, 0.0], [100.0, 0.0], [100.0, 80.0], [0.0, 80.0]];
        let dst = [[10.0, 5.0], [120.0, 12.0], [110.0, 90.0], [4.0, 84.0]];
        let h = h_from_quad(&src, &dst).unwrap();
        for k in 0..4 {
            let (u, v) = h_apply(&h, src[k][0], src[k][1]);
            assert!((u - dst[k][0]).abs() < 1e-9 && (v - dst[k][1]).abs() < 1e-9, "{k}: {u} {v}");
        }
        let inv = h_inv(&h).unwrap();
        let id = h_norm(&h_mul(&h, &inv));
        for e in 0..9 {
            assert!((id[e] - IDENTITY[e]).abs() < 1e-9, "{id:?}");
        }
        assert!(h_from_quad(&src, &[[0.0, 0.0]; 4]).is_none());
    }

    #[test]
    fn generator_flows_match_finite_differences() {
        for m in Model::ALL {
            for g in m.generators() {
                let (x, y) = (0.37, -0.61);
                let eps = 1e-6;
                let mut w = IDENTITY;
                for e in 0..9 {
                    w[e] += eps * g[e];
                }
                let (u, v) = h_apply(&w, x, y);
                let (du, dv) = generator_flow(g, x, y);
                assert!(((u - x) / eps - du).abs() < 1e-4, "{m:?} {g:?}");
                assert!(((v - y) / eps - dv).abs() < 1e-4, "{m:?} {g:?}");
            }
        }
    }

    #[test]
    fn pyramid_geometry_is_exactly_half() {
        // A vertical edge at x = 100 in a 200×64 plane lands at x = 50.
        let mut p = Plane::filled(200, 64, 0.0);
        for y in 0..64 {
            for x in 100..200 {
                p.px[y * 200 + x] = 1.0;
            }
        }
        let d = p.downsample();
        assert_eq!((d.w, d.h), (100, 32));
        // The half-way crossing of the ramp sits at coordinate 50.0.
        let a = d.sample(49.5, 16.0).unwrap();
        let b = d.sample(50.5, 16.0).unwrap();
        assert!((a + b - 1.0).abs() < 1e-5 && a < 0.5 && b > 0.5, "{a} {b}");
    }

    #[test]
    fn recovers_a_small_translation_to_a_hundredth_of_a_pixel() {
        let motion = h_translation(3.3, -1.7);
        let (fit, err) = track_pair(&motion, Model::Translation, &RenderOpts::default());
        assert!(err < 0.02, "err {err} fit {:?}", fit);
        assert!(fit.score > 0.98, "{}", fit.score);
    }

    #[test]
    fn recovers_a_perspective_warp_through_the_pyramid() {
        let motion = view(W, HT, 5.0, 1.06, 25.0, -18.0, 2e-4, -1e-4);
        let motion = h_mul(&motion, &h_inv(&view(W, HT, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)).unwrap());
        let (fit, err) = track_pair(&motion, Model::Perspective, &RenderOpts::default());
        assert!(err < 0.1, "err {err} fit {:?}", fit);
        assert!(fit.score > 0.97, "{}", fit.score);
    }

    #[test]
    fn constrained_models_fit_their_own_motion_and_not_more() {
        let sim = h_mul(
            &view(W, HT, 4.0, 1.05, 12.0, 7.0, 0.0, 0.0),
            &h_inv(&view(W, HT, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)).unwrap(),
        );
        let (_, e_sim) = track_pair(&sim, Model::Similarity, &RenderOpts::default());
        assert!(e_sim < 0.05, "similarity on a similarity: {e_sim}");
        let (_, e_tr) = track_pair(&sim, Model::Translation, &RenderOpts::default());
        assert!(e_tr > 1.0, "translation cannot express a rotation: {e_tr}");
        // Affine: a shear.
        let shear: H = [1.0, 0.12, 8.0, -0.05, 0.96, -6.0, 0.0, 0.0, 1.0];
        let (_, e_aff) = track_pair(&shear, Model::Affine, &RenderOpts::default());
        assert!(e_aff < 0.05, "affine on a shear: {e_aff}");
        let (_, e_persp) = track_pair(&shear, Model::Perspective, &RenderOpts::default());
        assert!(e_persp < 0.08, "perspective contains affine: {e_persp}");
    }

    #[test]
    fn a_lighting_change_is_not_motion() {
        let motion = h_translation(7.25, 4.5);
        let o = RenderOpts { gain: 1.8, bias: 0.2, noise: 0.02, ..Default::default() };
        let (fit, err) = track_pair(&motion, Model::Perspective, &o);
        assert!(err < 0.08, "err {err} {:?}", fit);
    }

    #[test]
    fn an_occluder_is_outvoted() {
        let motion = h_mul(
            &view(W, HT, 2.0, 1.02, 9.0, -5.0, 0.0, 0.0),
            &h_inv(&view(W, HT, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)).unwrap(),
        );
        // A disc over roughly a fifth of the region.
        let o = RenderOpts { occluder: Some(([300.0, 190.0], 42.0)), ..Default::default() };
        let (fit, err) = track_pair(&motion, Model::Perspective, &o);
        assert!(err < 0.25, "robust: err {err} {:?}", fit);
        // Without the reweighting the occluder pulls the fit.
        let tex = scene(3);
        let base = view(W, HT, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
        let reference = render(&tex, &base, W, HT, &RenderOpts::default());
        let moved = render(&tex, &h_mul(&motion, &base), W, HT, &o);
        let n = Tracker::levels_for(W, HT);
        let rp = Pyramid::build(reference, n);
        let cp = Pyramid::build(moved, n);
        let tpl = Template::build(&rp, &quad(), MIN_SIDE);
        let plain = solve(&tpl, &cp, &IDENTITY, &SolveOpts { robust: false, ..Default::default() });
        let e_plain = corner_err(&plain.h, &motion, &quad());
        assert!(e_plain > err, "plain {e_plain} vs robust {err}");
    }

    #[test]
    fn a_big_jump_is_caught_by_the_coarse_search() {
        // 70 px on a 220 px region: past the pyramid's reach from identity.
        let motion = h_translation(70.0, -22.0);
        let tex = scene(5);
        let base = view(W, HT, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
        let reference = render(&tex, &base, W, HT, &RenderOpts::default());
        let moved = render(&tex, &h_mul(&motion, &base), W, HT, &RenderOpts::default());
        let mut tr = Tracker::new(reference, quad(), TrackOpts::default());
        let r = tr.step(moved);
        let err = corner_err(&r.h, &motion, &quad());
        assert!(!r.lost && err < 0.1, "err {err} score {} lost {}", r.score, r.lost);
    }

    #[test]
    fn a_sequence_tracks_without_drift() {
        let tex = scene(9);
        let base = view(W, HT, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
        let n = 30;
        let frames: Vec<Plane> = (0..n)
            .map(|t| {
                let o = RenderOpts { noise: 0.01, seed: t as u32, ..Default::default() };
                render(&tex, &h_mul(&path(W, HT, t), &base), W, HT, &o)
            })
            .collect();
        for (reanchor, tol) in [(0.0f32, 0.15f64), (0.9, 0.35), (1.0, 0.6)] {
            let opts = TrackOpts { reanchor_below: reanchor, ..Default::default() };
            let mut tr = Tracker::new(frames[0].clone(), quad(), opts);
            let mut worst = 0.0f64;
            for t in 0..n {
                let r = tr.step(frames[t].clone());
                assert!(!r.lost, "frame {t} lost at reanchor {reanchor}: score {}", r.score);
                let truth = h_mul(&path(W, HT, t), &h_inv(&path(W, HT, 0)).unwrap());
                worst = worst.max(corner_err(&r.h, &truth, &quad()));
            }
            assert!(worst < tol, "reanchor {reanchor}: worst corner error {worst} px");
        }
    }

    #[test]
    fn a_lost_region_holds_and_recovers() {
        let tex = scene(11);
        let base = view(W, HT, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
        let reference = render(&tex, &base, W, HT, &RenderOpts::default());
        let mut tr = Tracker::new(reference, quad(), TrackOpts::default());
        // A frame of pure noise: nothing to find.
        let junk = Plane { w: W, h: HT, px: (0..W * HT).map(|i| ((i * 7919) % 1000) as f32 / 1000.0).collect() };
        let r = tr.step(junk);
        assert!(r.lost, "score {}", r.score);
        assert_eq!(r.h, IDENTITY);
        // The plate comes back, a little moved: picked up again.
        let motion = h_translation(4.0, 2.0);
        let moved = render(&tex, &h_mul(&motion, &base), W, HT, &RenderOpts::default());
        let r = tr.step(moved);
        assert!(!r.lost && corner_err(&r.h, &motion, &quad()) < 0.05, "{:?}", r.h);
    }

    #[test]
    fn a_big_occluder_coasts_and_never_becomes_the_template() {
        let tex = scene(17);
        let base = view(W, HT, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
        let step = h_translation(4.0, -1.5);
        let mut tr = Tracker::new(render(&tex, &base, W, HT, &RenderOpts::default()), quad(), TrackOpts::default());
        let mut motion = IDENTITY;
        // Two clean frames establish a velocity...
        for _ in 0..2 {
            motion = h_mul(&step, &motion);
            let r = tr.step(render(&tex, &h_mul(&motion, &base), W, HT, &RenderOpts::default()));
            assert!(!r.lost);
        }
        // ...then a disc covers most of the region for three frames.
        for _ in 0..3 {
            motion = h_mul(&step, &motion);
            let c = poly_centroid(&poly_map(&quad(), &motion));
            let o = RenderOpts { occluder: Some((c, 95.0)), ..Default::default() };
            let r = tr.step(render(&tex, &h_mul(&motion, &base), W, HT, &o));
            assert!(r.lost && r.rejected > 0.5, "covered frame: lost {} rejected {:.2} score {:.3}", r.lost, r.rejected, r.score);
            assert!(!r.reanchored, "a covered frame must never become the template");
            // Coasting on the velocity keeps the plane within a pixel or so.
            let err = corner_err(&r.h, &motion, &quad());
            assert!(err < 1.5, "coasting error {err}");
        }
        // The disc leaves: picked up again exactly, from where the coast put it.
        motion = h_mul(&step, &motion);
        let r = tr.step(render(&tex, &h_mul(&motion, &base), W, HT, &RenderOpts::default()));
        assert!(!r.lost, "score {}", r.score);
        assert!(corner_err(&r.h, &motion, &quad()) < 0.1, "{:?}", r.h);
    }

    /// An ignore matte over a disc (centre, radius), 1 inside.
    fn disc_matte(c: [f64; 2], r: f64) -> Plane {
        let mut m = Plane::filled(W, HT, 0.0);
        for y in 0..HT {
            for x in 0..W {
                let (dx, dy) = (x as f64 + 0.5 - c[0], y as f64 + 0.5 - c[1]);
                if dx * dx + dy * dy <= r * r {
                    m.px[y * W + x] = 1.0;
                }
            }
        }
        m
    }

    /// The occluder that `a_big_occluder_coasts…` loses the plane under,
    /// roto'd: with the matte it is no longer there as far as the solve
    /// is concerned, and the frames under it are TRACKED, not coasted.
    #[test]
    fn an_ignore_matte_tracks_through_a_big_occluder() {
        let tex = scene(17);
        let base = view(W, HT, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
        let step = h_mul(&h_translation(4.0, -1.5), &view(W, HT, 0.6, 1.004, 0.0, 0.0, 0.0, 0.0));
        let step = h_mul(&step, &h_inv(&base).unwrap());
        let mut tr = Tracker::new(render(&tex, &base, W, HT, &RenderOpts::default()), quad(), TrackOpts::default());
        let mut motion = IDENTITY;
        for t in 0..6 {
            motion = h_mul(&step, &motion);
            let c = poly_centroid(&poly_map(&quad(), &motion));
            let o = RenderOpts { occluder: Some((c, 95.0)), ..Default::default() };
            let frame = render(&tex, &h_mul(&motion, &base), W, HT, &o);
            let r = tr.step_masked(frame, Some(disc_matte(c, 97.0)));
            let err = corner_err(&r.h, &motion, &quad());
            assert!(!r.lost, "frame {t}: lost ({:?}) score {:.3}", r.why, r.score);
            assert!(err < 0.25, "frame {t}: corner error {err}");
        }
    }

    /// An occluder that sits still on the screen while the plane slides
    /// under it, already there in the reference frame. Matted, it is never
    /// cut into the template, and so never holds the track back.
    #[test]
    fn an_ignore_matte_keeps_a_reference_occluder_out_of_the_template() {
        let tex = scene(29);
        let base = view(W, HT, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
        let disc = ([260.0, 190.0], 60.0);
        let o = RenderOpts { occluder: Some(disc), ..Default::default() };
        let matte = || Some(disc_matte(disc.0, disc.1 + 2.0));
        let run = |masked: bool| -> f64 {
            let reference = render(&tex, &base, W, HT, &o);
            let mut tr = if masked {
                Tracker::new_masked(reference, matte(), quad(), TrackOpts::default())
            } else {
                Tracker::new(reference, quad(), TrackOpts::default())
            };
            let mut worst = 0.0f64;
            for t in 1..8 {
                let motion = h_translation(3.0 * t as f64, 1.0 * t as f64);
                let frame = render(&tex, &h_mul(&motion, &base), W, HT, &o);
                let r = if masked { tr.step_masked(frame, matte()) } else { tr.step(frame) };
                worst = worst.max(corner_err(&r.h, &motion, &quad()));
            }
            worst
        };
        let masked = run(true);
        assert!(masked < 0.2, "masked: worst corner error {masked}");
        println!("reference occluder: masked {masked:.3} px, unmasked {:.3} px", run(false));
    }

    /// A cloth shot's failure, pinned: a band of light sliding over the
    /// surface as the camera moves. The raw residual of that is broad and
    /// smooth and once read as an occluder covering the region — every
    /// frame lost as "covered" with the fit still fine.
    #[test]
    fn a_sliding_sheen_is_not_an_occluder() {
        let tex = scene(23);
        let base = view(W, HT, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
        let mut tr = Tracker::new(render(&tex, &base, W, HT, &RenderOpts::default()), quad(), TrackOpts::default());
        let mut worst = 0.0f64;
        for t in 1..24 {
            let motion = h_translation(2.5 * t as f64, -1.0 * t as f64);
            // ±30% of the light, a full period sliding across in 24 frames.
            let o = RenderOpts { sheen: (0.3, 0.26 * t as f32), ..Default::default() };
            let r = tr.step(render(&tex, &h_mul(&motion, &base), W, HT, &o));
            assert!(!r.lost, "frame {t}: lost as {:?}, rejected {:.2} score {:.3}", r.why, r.rejected, r.score);
            assert!(r.rejected < 0.5, "frame {t}: {:.2} of the region rejected under a sheen", r.rejected);
            worst = worst.max(corner_err(&r.h, &motion, &quad()));
        }
        assert!(worst < 1.0, "worst corner error {worst:.3} px under a sliding sheen");
    }

    #[test]
    fn a_re_anchor_leaves_the_occluder_out_of_the_template() {
        let tex = scene(21);
        let base = view(W, HT, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
        // Re-anchor on every frame: the template is re-cut from each one.
        let opts = TrackOpts { reanchor_below: 1.01, ..Default::default() };
        let mut tr = Tracker::new(render(&tex, &base, W, HT, &RenderOpts::default()), quad(), opts);
        let full = tr.tpl.levels[0].len();
        let motion = h_translation(3.0, 1.0);
        // A disc over roughly a fifth of the region.
        let o = RenderOpts { occluder: Some(([300.0, 190.0], 42.0)), ..Default::default() };
        let r = tr.step(render(&tex, &h_mul(&motion, &base), W, HT, &o));
        assert!(!r.lost && r.reanchored, "lost {} reanchored {} rejected {:.2}", r.lost, r.reanchored, r.rejected);
        let cut = tr.tpl.levels[0].len();
        // The disc's pixels were left out — and not much else.
        let frac = 1.0 - cut as f64 / full as f64;
        assert!((0.10..0.35).contains(&frac), "template kept {cut} of {full} pixels ({frac:.2} dropped)");
        assert!(corner_err(&r.h, &motion, &quad()) < 0.3);
        // The disc gone, the track carries on from the cut template.
        let motion2 = h_mul(&h_translation(3.0, 1.0), &motion);
        let r = tr.step(render(&tex, &h_mul(&motion2, &base), W, HT, &RenderOpts::default()));
        assert!(!r.lost && corner_err(&r.h, &motion2, &quad()) < 0.3, "{:?} lost {}", r.h, r.lost);
    }

    #[test]
    fn mesh_grid_stays_inside_the_region_and_alternates_diagonals() {
        let poly = quad();
        let m = Mesh::grid(&poly, 10);
        assert!(!m.tris.is_empty() && m.verts.len() > 50, "{} verts {} tris", m.verts.len(), m.tris.len());
        for t in &m.tris {
            let c = [
                (m.verts[t[0] as usize][0] + m.verts[t[1] as usize][0] + m.verts[t[2] as usize][0]) as f64 / 3.0,
                (m.verts[t[0] as usize][1] + m.verts[t[1] as usize][1] + m.verts[t[2] as usize][1]) as f64 / 3.0,
            ];
            assert!(poly_contains(&poly, c[0], c[1]));
            assert!(t.iter().all(|&v| (v as usize) < m.verts.len()));
        }
        let nb = m.neighbours();
        assert!(nb.iter().all(|n| !n.is_empty()));
        // A flat quad: a triangle is never degenerate.
        for t in &m.tris {
            let a = m.verts[t[0] as usize];
            let b = m.verts[t[1] as usize];
            let c = m.verts[t[2] as usize];
            let area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
            assert!(area.abs() > 1.0, "{area}");
        }
    }

    #[test]
    fn a_flat_patch_has_no_texture_and_a_busy_one_does() {
        let flat = Plane::filled(64, 64, 0.5);
        let pyr = Pyramid::build(flat, 2);
        let poly = [[20.0, 20.0], [44.0, 20.0], [44.0, 44.0], [20.0, 44.0]];
        let t = Template::build(&pyr, &poly, 6.0);
        assert!(t.texture() < 1e-6);
        let busy = Pyramid::build(texture(64, 64, 2), 2);
        let t = Template::build(&busy, &poly, 6.0);
        assert!(t.texture() > PATCH_FLAT, "{}", t.texture());
    }

    #[test]
    fn the_mesh_follows_a_bulge_the_plane_cannot() {
        let tex = scene(13);
        let base = view(W, HT, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
        let reference = render(&tex, &base, W, HT, &RenderOpts::default());
        let motion = h_translation(6.0, -3.0);
        let amp = 3.0f32;
        let bent = render(&tex, &h_mul(&motion, &base), W, HT, &RenderOpts { bulge: amp, ..Default::default() });
        let mopts = MeshOpts { cells: 10, patch: 10, smooth: 0.4 };
        let mut tr = Tracker::new(reference, quad(), TrackOpts { mesh: Some(mopts), ..Default::default() });
        let r = tr.step(bent);
        assert!(!r.lost);
        let mesh = tr.mesh().unwrap().clone();
        assert_eq!(r.mesh.len(), mesh.verts.len());
        // Truth: the reference vertex v lands where the bulge PUTS the
        // planar image of v: frame point q with q - d(q) = H v. Solve by
        // fixed-point iteration (d is smooth and small).
        let mut mesh_err = 0.0f64;
        let mut plane_err = 0.0f64;
        let mut n = 0.0;
        for (i, v) in mesh.verts.iter().enumerate() {
            let (hx, hy) = h_apply(&motion, v[0] as f64, v[1] as f64);
            let (mut qx, mut qy) = (hx, hy);
            for _ in 0..20 {
                let (dx, dy) = bulge_offset(amp, qx, qy, W, HT);
                qx = hx + dx;
                qy = hy + dy;
            }
            let m = r.mesh[i];
            let me = ((m[0] as f64 - qx).powi(2) + (m[1] as f64 - qy).powi(2)).sqrt();
            let pe = ((hx - qx).powi(2) + (hy - qy).powi(2)).sqrt();
            mesh_err += me;
            plane_err += pe;
            n += 1.0;
        }
        mesh_err /= n;
        plane_err /= n;
        assert!(plane_err > 1.0, "the bulge should be worth chasing: {plane_err}");
        assert!(mesh_err < 0.5 && mesh_err < 0.35 * plane_err, "mesh {mesh_err} vs plane {plane_err}");
    }

    #[test]
    fn luma_plane_is_log_shaped_and_drops_padding() {
        let w = 2;
        let h = 2;
        let stride = 12; // 2 px * 4 + padding
        let mut src = vec![half::f16::ZERO; stride * h];
        src[stride + 4 + 1] = half::f16::from_f32(1.0); // pixel (1,1) green
        let p = Plane::from_rgba_f16(&src, w, h, stride);
        assert_eq!(p.px.len(), 4);
        assert_eq!(p.px[0], 0.0);
        let expect = (1.0f32 + 16.0 * 0.7152).ln() / 17f32.ln();
        assert!((p.px[3] - expect).abs() < 1e-5, "{}", p.px[3]);
    }
}

/// Diagnostics, ignored by default:
/// `cargo test -p engine --lib planar::probe -- --ignored --nocapture`.
/// `probe_robust` prints the solver's error under gain, noise and occlusion
/// for every model with and without the reweighting, then the per-level fit
/// on the occluded case with a census of which pixels lost their vote;
/// `probe_scale` the per-level fit at 960x540 (where a strided finest level
/// once cost 0.5 px); `probe_mesh_*` the worst mesh vertex per frame and
/// why vertices declined to vote on the bulged shot. Rerun after touching
/// `solve_level`, `robust_weights` or `MeshTrack::step`.
#[cfg(test)]
mod probe {
    use super::synth::*;
    use super::*;

    const W: usize = 512;
    const HT: usize = 384;
    fn quad() -> Vec<[f64; 2]> {
        vec![[150.0, 110.0], [370.0, 118.0], [362.0, 268.0], [158.0, 262.0]]
    }
    fn corner_err(tr_h: &H, truth: &H, poly: &[[f64; 2]]) -> f64 {
        poly.iter()
            .map(|p| {
                let (a, b) = h_apply(tr_h, p[0], p[1]);
                let (c, d) = h_apply(truth, p[0], p[1]);
                ((a - c).powi(2) + (b - d).powi(2)).sqrt()
            })
            .fold(0.0, f64::max)
    }

    #[test]
    #[ignore]
    fn probe_robust() {
        let tex = texture(W, HT, 3);
        let base = view(W, HT, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
        let reference = render(&tex, &base, W, HT, &RenderOpts::default());
        let n = Tracker::levels_for(W, HT);
        let rp = Pyramid::build(reference, n);
        let tpl = Template::build(&rp, &quad(), MIN_SIDE);
        eprintln!("levels {} sizes {:?}", tpl.levels.len(), tpl.levels.iter().map(|l| l.len()).collect::<Vec<_>>());
        let motion = h_translation(7.25, 4.5);
        let occ = ([300.0f64, 190.0f64], 42.0f64);
        for (name, o) in [
            ("plain", RenderOpts::default()),
            ("gain", RenderOpts { gain: 1.8, bias: 0.2, ..Default::default() }),
            ("noise", RenderOpts { noise: 0.02, ..Default::default() }),
            ("both", RenderOpts { gain: 1.8, bias: 0.2, noise: 0.02, ..Default::default() }),
            ("occl", RenderOpts { occluder: Some(occ), ..Default::default() }),
        ] {
            let moved = render(&tex, &h_mul(&motion, &base), W, HT, &o);
            let cp = Pyramid::build(moved, n);
            for robust in [false, true] {
                for model in [Model::Translation, Model::Affine, Model::Perspective] {
                    let fit = solve(&tpl, &cp, &IDENTITY, &SolveOpts { model, robust, ..Default::default() });
                    let err = corner_err(&fit.h, &motion, &quad());
                    eprintln!("{name:6} robust={robust:5} {:12} err {err:.4} score {:.4} iters {}", model.label(), fit.score, fit.iters);
                }
            }
        }
        // Per level, on the occluded frame.
        let o = RenderOpts { occluder: Some(occ), ..Default::default() };
        let moved = render(&tex, &h_mul(&motion, &base), W, HT, &o);
        let cp = Pyramid::build(moved, n);
        let opts = SolveOpts { model: Model::Perspective, robust: true, ..Default::default() };
        let mut s = Scratch::default();
        let mut h = h_mul(&IDENTITY, &tpl.norm_inv);
        let nl = tpl.levels.len();
        for l in (0..nl).rev() {
            let f = (1u64 << l) as f64;
            let hl = h_mul(&h_scale(1.0 / f), &h);
            let eps_level = opts.eps_px / tpl.scale;
            let radius = if l + 1 == nl { 1 } else { 2 };
            let (hl2, it) = solve_level(&tpl.levels[l], &cp.levels[l], hl, &opts, eps_level, radius, l + 1 < nl, &mut s);
            h = h_mul(&h_scale(f), &hl2);
            let err = corner_err(&h_mul(&h, &tpl.norm), &motion, &quad());
            let tl = &tpl.levels[l];
            let (mut rej_in, mut rej_out, mut in_occ, mut n_valid) = (0, 0, 0, 0);
            for k in 0..tl.len() {
                if !s.px[k].valid {
                    continue;
                }
                n_valid += 1;
                let (px, py) = h_apply(&tpl.norm_inv, tl.xs[k] as f64, tl.ys[k] as f64);
                let (mx, my) = h_apply(&motion, px, py);
                let inside = (mx - occ.0[0]).powi(2) + (my - occ.0[1]).powi(2) < occ.1 * occ.1;
                let w = s.px[k].wt;
                if inside {
                    in_occ += 1;
                    if w == 0.0 {
                        rej_in += 1;
                    }
                } else if w == 0.0 {
                    rej_out += 1;
                }
            }
            eprintln!("level {l}: iters {it} err {err:.3} valid {n_valid} occluded {in_occ} rejected {rej_in}, clean rejected {rej_out}");
        }
    }


    #[test]
    #[ignore]
    fn probe_scale() {
        let (w, h) = (960usize, 540usize);
        let tex = texture(w, h, 7);
        let base = view(w, h, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
        let quad = vec![[307.2, 162.0], [652.8, 167.4], [643.2, 378.0], [316.8, 372.6]];
        let truth = h_mul(&path(w, h, 1), &h_inv(&path(w, h, 0)).unwrap());
        let err_of = |hh: &H| {
            quad.iter()
                .map(|p| {
                    let (a, b) = h_apply(hh, p[0], p[1]);
                    let (c, d) = h_apply(&truth, p[0], p[1]);
                    ((a - c).powi(2) + (b - d).powi(2)).sqrt()
                })
                .fold(0.0, f64::max)
        };
        for noise in [0.0f32, 0.01] {
            let mk = |t: usize| {
                let o = RenderOpts { noise, seed: t as u32 + 1, ..Default::default() };
                render(&tex, &h_mul(&path(w, h, t), &base), w, h, &o)
            };
            let f0 = mk(0);
            let f1 = mk(1);
            for n in [5usize, 3, 2] {
                let rp = Pyramid::build(f0.clone(), n);
                let cp = Pyramid::build(f1.clone(), n);
                let tpl = Template::build(&rp, &quad, MIN_SIDE);
                for robust in [false, true] {
                    for model in [Model::Affine, Model::Perspective] {
                        let fit = solve(&tpl, &cp, &IDENTITY, &SolveOpts { model, robust, ..Default::default() });
                        eprintln!("noise {noise} levels {n} ({} used) robust {robust:5} {:11} err {:.4} score {:.4} iters {}", tpl.levels.len(), model.label(), err_of(&fit.h), fit.score, fit.iters);
                    }
                }
            }
            // Per-level trace, perspective, robust, 5 levels.
            let n = 5;
            let rp = Pyramid::build(f0.clone(), n);
            let cp = Pyramid::build(f1.clone(), n);
            let tpl = Template::build(&rp, &quad, MIN_SIDE);
            let opts = SolveOpts::default();
            let mut s = Scratch::default();
            let mut hh = h_mul(&IDENTITY, &tpl.norm_inv);
            let nl = tpl.levels.len();
            for l in (0..nl).rev() {
                let f = (1u64 << l) as f64;
                let hl = h_mul(&h_scale(1.0 / f), &hh);
                let (hl2, it) = solve_level(&tpl.levels[l], &cp.levels[l], hl, &opts, opts.eps_px / tpl.scale, if l + 1 == nl { 1 } else { 2 }, l + 1 < nl, &mut s);
                hh = h_mul(&h_scale(f), &hl2);
                let rejected = s.px.iter().filter(|p| p.valid && p.wt == 0.0).count();
                eprintln!("  level {l}: iters {it} err {:.4} rejected {rejected}/{}", err_of(&h_mul(&hh, &tpl.norm)), tpl.levels[l].len());
            }
            // And the truth's own score: is the optimum even at the truth?
            let (sc, _) = score(&tpl, &cp, &truth);
            let (sf, _) = score(&tpl, &cp, &solve(&tpl, &cp, &IDENTITY, &opts).h);
            eprintln!("  score at truth {sc:.5} vs at fit {sf:.5}");
        }
    }


    #[test]
    #[ignore]
    fn probe_mesh_bulge_rejects() {
        let (w, h) = (960usize, 540usize);
        let tex = texture(w, h, 7);
        let base = view(w, h, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
        let quad = region(w, h).to_vec();
        let mk = |t: usize| {
            let amp = 6.0 * ((t as f32) * 0.2).sin();
            let o = RenderOpts { noise: 0.01, seed: t as u32 + 1, bulge: amp, ..Default::default() };
            let p = render(&tex, &h_mul(&path(w, h, t), &base), w, h, &o);
            Plane { w, h, px: p.px.iter().map(|v| (1.0 + 16.0 * v.max(0.0)).ln() / 17f32.ln()).collect() }
        };
        let mut tr = Tracker::new(mk(0), quad.clone(), TrackOpts { mesh: Some(MeshOpts::default()), ..Default::default() });
        for t in 0..30 {
            let r = tr.step(mk(t));
            let voted = r.mesh_conf.iter().filter(|c| **c > 0.0).count();
            let rej = tr.mesh.as_ref().unwrap().rejects;
            eprintln!("frame {t:2}: voted {voted}/{} rejects flat {} occl {} cover {} score {} far {} gain {}", r.mesh.len(), rej[0], rej[1], rej[2], rej[3], rej[4], rej[5]);
        }
    }

    #[test]
    #[ignore]
    fn probe_mesh_worst_vertex() {
        let (w, h) = (960usize, 540usize);
        let tex = texture(w, h, 7);
        let base = view(w, h, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
        let quad = region(w, h).to_vec();
        let mk = |t: usize| {
            let o = RenderOpts { noise: 0.01, seed: t as u32 + 1, ..Default::default() };
            let p = render(&tex, &h_mul(&path(w, h, t), &base), w, h, &o);
            Plane { w, h, px: p.px.iter().map(|v| (1.0 + 16.0 * v.max(0.0)).ln() / 17f32.ln()).collect() }
        };
        for cells in [10usize, 12] {
            let mopts = MeshOpts { cells, ..Default::default() };
            let mut tr = Tracker::new(mk(0), quad.clone(), TrackOpts { mesh: Some(mopts), ..Default::default() });
            let mesh = tr.mesh().unwrap().clone();
            let nb = mesh.neighbours();
            let mut worst_all = (0.0f64, 0usize, 0usize);
            for t in 0..48 {
                let r = tr.step(mk(t));
                let truth = h_mul(&path(w, h, t), &h_inv(&path(w, h, 0)).unwrap());
                let mut worst = (0.0f64, 0usize);
                for (i, v) in mesh.verts.iter().enumerate() {
                    let (x, y) = h_apply(&truth, v[0] as f64, v[1] as f64);
                    let g = r.mesh[i];
                    let e = ((g[0] as f64 - x).powi(2) + (g[1] as f64 - y).powi(2)).sqrt();
                    if e > worst.0 {
                        worst = (e, i);
                    }
                }
                if worst.0 > worst_all.0 {
                    worst_all = (worst.0, worst.1, t);
                }
                if t % 8 == 0 || worst.0 > 1.0 {
                    let i = worst.1;
                    let v = mesh.verts[i];
                    let (x, y) = h_apply(&truth, v[0] as f64, v[1] as f64);
                    let voted = r.mesh_conf.iter().filter(|c| **c > 0.0).count();
                    eprintln!(
                        "cells {cells} frame {t:2}: worst vertex {i} at ref ({:.0},{:.0}) truth ({x:.1},{y:.1}) got ({:.1},{:.1}) err {:.2} conf {:.2} neighbours {} | {voted}/{} voted, score {:.3} reanchor {}",
                        v[0], v[1], r.mesh[i][0], r.mesh[i][1], worst.0, r.mesh_conf[i], nb[i].len(), mesh.verts.len(), r.score, r.reanchored
                    );
                }
            }
            eprintln!("cells {cells}: worst {:.2} at vertex {} frame {}", worst_all.0, worst_all.1, worst_all.2);
        }
    }
}


