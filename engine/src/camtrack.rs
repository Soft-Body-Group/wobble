//! Camera tracking: feature tracks off the picture, and a 3D camera solved
//! from them (structure from motion), for the CameraTrack node.
//!
//! Two halves, both ours, no new dependencies:
//!
//! - **Tracks** ([`track`]): SEA-RAFT's flow between every neighbouring pair,
//!   forward AND backward, at `1 / divisor` of the size. A grid of points is
//!   carried along the forward flow; a point whose backward flow does not
//!   bring it home (`FB_TOL`) is dropped, as is one that leaves the frame
//!   or lands under the matte; cells left empty are re-seeded on the
//!   strongest gradient in them. A track is its start frame and its
//!   positions, in FULL-size pixels (centres at +0.5).
//! - **Solve** ([`solve`]): the usual incremental pipeline in `f64`.
//!   Two keyframes with parallax initialise (essential matrix by RANSAC
//!   over the eight-point fit, the pose with the most points in front of
//!   both cameras, DLT triangulation); every other frame is placed by a
//!   robust pose refinement against the points so far (consecutive frames
//!   start from their neighbour's pose); new tracks are triangulated as
//!   their baseline allows; keyframes are bundle-adjusted (Levenberg
//!   Marquardt, Huber, the points eliminated through their 3x3 blocks so
//!   the system solved is cameras-only) in a sliding window as they
//!   arrive and globally at the end, the focal length optionally with
//!   them. Non-keyframes are then re-fitted against the final points.
//!
//! Inside the solver the convention is the computer-vision one (camera
//! looks down +Z, image y down, `x_c = R x + t`); the world is the FIRST
//! camera's frame, the scale set so the median point depth from it is 1.
//! [`Solve::camera`] / [`Solve::point`] turn that into the 3D cards'
//! convention (Y up, the camera down −Z, `scene3d`) — the one place the
//! flip happens.
//!
//! Lens distortion is not modelled: undistort upstream (Lens Distortion).
//! `WOBBLE_CAMTRACK_TRACE=1` prints every bundle step's cost.
//!
//! Measured on the synthetic three-plane dolly (`synth`): the fit is at
//! the noise floor, and what limits the camera is the COUNT of tracks —
//! 170 points under 0.4 px of jitter leave a 1° / 10%-of-travel
//! ambiguity that explains the data better than the truth does; 450
//! points bring it to 0.03° / 0.4%. The tracker seeds a grid for exactly
//! that reason, and a bundle over every frame costs little (the system
//! solved is cameras-only).

use crate::decode::ExrDecoder as _;
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

pub type V3 = [f64; 3];
pub type M3 = [[f64; 3]; 3];

// ------------------------------------------------------------ small maths

pub fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

pub fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

pub fn scl(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

pub fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}

pub fn unit(a: V3) -> V3 {
    let n = norm(a);
    if n > 1e-300 {
        scl(a, 1.0 / n)
    } else {
        a
    }
}

pub const I3: M3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

pub fn m_mul(a: &M3, b: &M3) -> M3 {
    let mut m = [[0.0; 3]; 3];
    for (r, row) in m.iter_mut().enumerate() {
        for (c, v) in row.iter_mut().enumerate() {
            *v = a[r][0] * b[0][c] + a[r][1] * b[1][c] + a[r][2] * b[2][c];
        }
    }
    m
}

pub fn m_t(a: &M3) -> M3 {
    let mut m = [[0.0; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            m[r][c] = a[c][r];
        }
    }
    m
}

pub fn m_v(a: &M3, v: V3) -> V3 {
    [dot(a[0], v), dot(a[1], v), dot(a[2], v)]
}

fn skew(w: V3) -> M3 {
    [[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]]
}

/// Rodrigues: the rotation of angle `|w|` about `w`.
pub fn so3_exp(w: V3) -> M3 {
    let th = norm(w);
    let k = skew(w);
    let kk = m_mul(&k, &k);
    let (a, b) = if th < 1e-9 { (1.0, 0.5) } else { (th.sin() / th, (1.0 - th.cos()) / (th * th)) };
    let mut r = I3;
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] += a * k[i][j] + b * kk[i][j];
        }
    }
    r
}

/// The inverse of [`so3_exp`].
pub fn so3_log(r: &M3) -> V3 {
    let c = ((r[0][0] + r[1][1] + r[2][2] - 1.0) * 0.5).clamp(-1.0, 1.0);
    let th = c.acos();
    let v = [r[2][1] - r[1][2], r[0][2] - r[2][0], r[1][0] - r[0][1]];
    if th < 1e-9 {
        return scl(v, 0.5);
    }
    if (std::f64::consts::PI - th) < 1e-6 {
        // Near a half turn the axis comes off the symmetric part.
        let mut best = 0;
        for i in 1..3 {
            if r[i][i] > r[best][best] {
                best = i;
            }
        }
        let i = best;
        let (j, k) = ((i + 1) % 3, (i + 2) % 3);
        let s = (r[i][i] - r[j][j] - r[k][k] + 1.0).max(0.0).sqrt();
        let mut ax = [0.0; 3];
        ax[i] = s * 0.5;
        ax[j] = (r[i][j] + r[j][i]) / (2.0 * s.max(1e-12));
        ax[k] = (r[i][k] + r[k][i]) / (2.0 * s.max(1e-12));
        return scl(unit(ax), th);
    }
    scl(v, th / (2.0 * th.sin()))
}

/// Eigen-decomposition of a symmetric `n×n` matrix (row-major) by Jacobi
/// rotations: the values ASCENDING and the matching unit vectors, each a
/// row of the result. Small matrices only (the solver's are 4 and 9).
pub fn sym_eig(a: &[f64], n: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    let mut m = a.to_vec();
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    for _sweep in 0..60 {
        let mut off = 0.0;
        for p in 0..n {
            for q in p + 1..n {
                off += m[p * n + q] * m[p * n + q];
            }
        }
        if off < 1e-30 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = m[p * n + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let theta = (m[q * n + q] - m[p * n + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (mkp, mkq) = (m[k * n + p], m[k * n + q]);
                    m[k * n + p] = c * mkp - s * mkq;
                    m[k * n + q] = s * mkp + c * mkq;
                }
                for k in 0..n {
                    let (mpk, mqk) = (m[p * n + k], m[q * n + k]);
                    m[p * n + k] = c * mpk - s * mqk;
                    m[q * n + k] = s * mpk + c * mqk;
                }
                for k in 0..n {
                    let (vkp, vkq) = (v[k * n + p], v[k * n + q]);
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| m[i * n + i].total_cmp(&m[j * n + j]));
    let vals = order.iter().map(|&i| m[i * n + i]).collect();
    let vecs = order.iter().map(|&i| (0..n).map(|k| v[k * n + i]).collect()).collect();
    (vals, vecs)
}

/// SVD of a 3x3: `(u, s, v)` with `a = u · diag(s) · vᵀ`, `s` descending.
fn svd3(a: &M3) -> (M3, V3, M3) {
    let ata = m_mul(&m_t(a), a);
    let flat: Vec<f64> = ata.iter().flatten().copied().collect();
    let (vals, vecs) = sym_eig(&flat, 3);
    // Descending.
    let mut v = [[0.0; 3]; 3];
    let mut s = [0.0; 3];
    for k in 0..3 {
        let src = 2 - k;
        s[k] = vals[src].max(0.0).sqrt();
        for r in 0..3 {
            v[r][k] = vecs[src][r];
        }
    }
    let mut u = [[0.0; 3]; 3];
    for k in 0..3 {
        let col = m_v(a, [v[0][k], v[1][k], v[2][k]]);
        let col = if s[k] > 1e-12 { scl(col, 1.0 / s[k]) } else { [0.0; 3] };
        for r in 0..3 {
            u[r][k] = col[r];
        }
    }
    // A null column of u: complete the frame.
    for k in 0..3 {
        let c = [u[0][k], u[1][k], u[2][k]];
        if norm(c) < 0.5 {
            let (j, l) = ((k + 1) % 3, (k + 2) % 3);
            let n = unit(cross([u[0][j], u[1][j], u[2][j]], [u[0][l], u[1][l], u[2][l]]));
            // Keep u right-handed: n = u_j × u_l means u_k = ±n by parity.
            let sign = if (j + 1) % 3 == l { 1.0 } else { -1.0 };
            for r in 0..3 {
                u[r][k] = sign * n[r];
            }
        }
    }
    (u, s, v)
}

fn det3(m: &M3) -> f64 {
    dot(m[0], cross(m[1], m[2]))
}

// ------------------------------------------------------------- the model

/// A camera pose, world → camera: `x_c = R x + t`.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Pose {
    pub r: M3,
    pub t: V3,
}

impl Pose {
    pub const IDENTITY: Pose = Pose { r: I3, t: [0.0; 3] };

    pub fn apply(&self, x: V3) -> V3 {
        add(m_v(&self.r, x), self.t)
    }

    /// The camera centre in world coordinates.
    pub fn centre(&self) -> V3 {
        scl(m_v(&m_t(&self.r), self.t), -1.0)
    }

    pub fn inverse(&self) -> Pose {
        let rt = m_t(&self.r);
        Pose { r: rt, t: scl(m_v(&rt, self.t), -1.0) }
    }

    /// `self ∘ other`: apply `other`, then `self`.
    pub fn then(&self, other: &Pose) -> Pose {
        Pose { r: m_mul(&self.r, &other.r), t: add(m_v(&self.r, other.t), self.t) }
    }
}

/// One feature's path: its position in every frame from `start`, in
/// full-size pixels (centres at +0.5).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Track {
    pub start: usize,
    pub pts: Vec<[f64; 2]>,
}

impl Track {
    pub fn end(&self) -> usize {
        self.start + self.pts.len()
    }

    pub fn at(&self, f: usize) -> Option<[f64; 2]> {
        (f >= self.start && f < self.end()).then(|| self.pts[f - self.start])
    }
}

/// A pinhole: focal length and principal point, pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Intrinsics {
    pub f: f64,
    pub c: [f64; 2],
}

impl Intrinsics {
    pub fn for_size(f: f64, size: [u32; 2]) -> Self {
        Self { f, c: [size[0] as f64 * 0.5, size[1] as f64 * 0.5] }
    }

    /// Pixel → normalised image plane (z = 1).
    pub fn normalise(&self, p: [f64; 2]) -> [f64; 2] {
        [(p[0] - self.c[0]) / self.f, (p[1] - self.c[1]) / self.f]
    }

    /// Camera-space point → pixel; None behind the camera.
    pub fn project(&self, p: V3) -> Option<[f64; 2]> {
        (p[2] > 1e-9).then(|| [self.c[0] + self.f * p[0] / p[2], self.c[1] + self.f * p[1] / p[2]])
    }
}

/// How to solve.
#[derive(Clone, Debug, PartialEq)]
pub struct SolveOpts {
    pub size: [u32; 2],
    /// The focal length in pixels to start from (and to hold, unless
    /// refined).
    pub focal_px: f64,
    pub refine_focal: bool,
    /// Keyframes bundle-adjusted together, at most.
    pub max_keyframes: usize,
}

/// A solved camera: a pose per frame, the points, and how well they fit.
#[derive(Clone, Debug, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize), serde(default))]
pub struct Solve {
    pub size: [u32; 2],
    pub focal_px: f64,
    /// Per frame, world (= the first camera's frame) → camera, CV
    /// convention. A frame the solve could not place takes its neighbour's
    /// pose and is listed in `unplaced`.
    pub poses: Vec<Pose>,
    pub unplaced: Vec<usize>,
    pub points: Vec<V3>,
    /// Per point, the track it came from.
    pub point_track: Vec<usize>,
    /// Per frame, the mean reprojection error of the points it sees, px.
    pub errors: Vec<f64>,
}

/// The CV → cards flip: x right, y UP, the camera looking down −Z.
const FLIP: M3 = [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]];

impl Solve {
    pub fn is_empty(&self) -> bool {
        self.poses.is_empty()
    }

    /// The camera at `frame` as a cards-convention local → world matrix
    /// (`scene3d::Mat4`), the world being the first camera's frame with Y
    /// up and −Z forward, scaled by `scale`.
    pub fn camera(&self, frame: usize, scale: f64) -> crate::scene3d::Mat4 {
        let i = frame.min(self.poses.len().saturating_sub(1));
        let Some(p) = self.poses.get(i) else { return crate::scene3d::identity() };
        // Local (flipped camera) → world (flipped first camera):
        // F · Rᵀ · F on the axes, −F · Rᵀ · t for the origin.
        let rt = m_t(&p.r);
        let r = m_mul(&FLIP, &m_mul(&rt, &FLIP));
        let c = scl(m_v(&FLIP, m_v(&rt, p.t)), -scale);
        let mut m = crate::scene3d::identity();
        for a in 0..3 {
            for b in 0..3 {
                m[a][b] = r[a][b] as f32;
            }
            m[a][3] = c[a] as f32;
        }
        m
    }

    /// A point in the cards' world, scaled.
    pub fn point(&self, k: usize, scale: f64) -> [f32; 3] {
        let p = scl(m_v(&FLIP, self.points[k]), scale);
        [p[0] as f32, p[1] as f32, p[2] as f32]
    }

    /// The mean reprojection error over every placed frame, px.
    pub fn mean_error(&self) -> f64 {
        let v: Vec<f64> = self.errors.iter().copied().filter(|e| e.is_finite()).collect();
        if v.is_empty() {
            0.0
        } else {
            v.iter().sum::<f64>() / v.len() as f64
        }
    }
}

// ------------------------------------------------------------ two views

/// `x2ᵀ E x1 = 0` by the eight-point fit on normalised coordinates;
/// rank two enforced. None with fewer than eight.
fn eight_point(pairs: &[([f64; 2], [f64; 2])]) -> Option<M3> {
    if pairs.len() < 8 {
        return None;
    }
    let mut ata = vec![0.0; 81];
    for (a, b) in pairs {
        let row = [b[0] * a[0], b[0] * a[1], b[0], b[1] * a[0], b[1] * a[1], b[1], a[0], a[1], 1.0];
        for i in 0..9 {
            for j in 0..9 {
                ata[i * 9 + j] += row[i] * row[j];
            }
        }
    }
    let (_, vecs) = sym_eig(&ata, 9);
    let e = &vecs[0];
    let e = [[e[0], e[1], e[2]], [e[3], e[4], e[5]], [e[6], e[7], e[8]]];
    let (u, s, v) = svd3(&e);
    let sm = 0.5 * (s[0] + s[1]);
    let d = [[sm, 0.0, 0.0], [0.0, sm, 0.0], [0.0, 0.0, 0.0]];
    Some(m_mul(&u, &m_mul(&d, &m_t(&v))))
}

/// Sampson distance of a pair to `e`, in normalised units.
fn sampson(e: &M3, a: [f64; 2], b: [f64; 2]) -> f64 {
    let x1 = [a[0], a[1], 1.0];
    let x2 = [b[0], b[1], 1.0];
    let ex1 = m_v(e, x1);
    let etx2 = m_v(&m_t(e), x2);
    let num = dot(x2, ex1);
    let den = ex1[0] * ex1[0] + ex1[1] * ex1[1] + etx2[0] * etx2[0] + etx2[1] * etx2[1];
    if den < 1e-300 {
        f64::INFINITY
    } else {
        num * num / den
    }
}

/// A small deterministic generator (the solve must be repeatable).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
}

/// The essential matrix by RANSAC over the eight-point fit (`thr` a Sampson
/// bound in normalised units), re-fitted on its inliers. Returns it with
/// the inlier flags.
fn essential_ransac(pairs: &[([f64; 2], [f64; 2])], thr: f64, iters: usize) -> Option<(M3, Vec<bool>)> {
    if pairs.len() < 8 {
        return None;
    }
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let mut best: Option<(usize, M3)> = None;
    for _ in 0..iters {
        let mut pick = Vec::with_capacity(8);
        while pick.len() < 8 {
            let k = rng.below(pairs.len());
            if !pick.contains(&k) {
                pick.push(k);
            }
        }
        let sample: Vec<_> = pick.iter().map(|&k| pairs[k]).collect();
        let Some(e) = eight_point(&sample) else { continue };
        let n = pairs.iter().filter(|(a, b)| sampson(&e, *a, *b) < thr).count();
        if best.as_ref().is_none_or(|(bn, _)| n > *bn) {
            best = Some((n, e));
        }
    }
    let (_, mut e) = best?;
    // Refit on the inliers, twice.
    let mut inl = vec![false; pairs.len()];
    for _ in 0..2 {
        for (k, (a, b)) in pairs.iter().enumerate() {
            inl[k] = sampson(&e, *a, *b) < thr;
        }
        let set: Vec<_> = pairs.iter().zip(&inl).filter(|(_, &i)| i).map(|(p, _)| *p).collect();
        if let Some(e2) = eight_point(&set) {
            e = e2;
        }
    }
    for (k, (a, b)) in pairs.iter().enumerate() {
        inl[k] = sampson(&e, *a, *b) < thr;
    }
    Some((e, inl))
}

/// Linear (DLT) triangulation of a point seen at `obs` (normalised
/// coordinates) by cameras `poses` (world → camera). None when degenerate.
fn triangulate(poses: &[Pose], obs: &[[f64; 2]]) -> Option<V3> {
    if poses.len() < 2 {
        return None;
    }
    let mut ata = [0.0; 16];
    for (p, o) in poses.iter().zip(obs) {
        let row = |a: usize, s: f64, b: usize| -> [f64; 4] {
            [
                s * p.r[a][0] - p.r[b][0],
                s * p.r[a][1] - p.r[b][1],
                s * p.r[a][2] - p.r[b][2],
                s * p.t[a] - p.t[b],
            ]
        };
        for r in [row(2, o[0], 0), row(2, o[1], 1)] {
            for i in 0..4 {
                for j in 0..4 {
                    ata[i * 4 + j] += r[i] * r[j];
                }
            }
        }
    }
    let (_, vecs) = sym_eig(&ata, 4);
    let v = &vecs[0];
    if v[3].abs() < 1e-12 {
        return None;
    }
    let x = [v[0] / v[3], v[1] / v[3], v[2] / v[3]];
    x.iter().all(|c| c.is_finite()).then_some(x)
}

/// The angle at the point between the rays from two camera centres,
/// radians.
fn parallax(a: &Pose, b: &Pose, x: V3) -> f64 {
    let ra = unit(sub(x, a.centre()));
    let rb = unit(sub(x, b.centre()));
    dot(ra, rb).clamp(-1.0, 1.0).acos()
}

/// The relative pose of the second view from `e`: of the four candidates,
/// the one with the most of `pairs` in front of both cameras, and the
/// points it triangulates (None where behind or degenerate).
fn pose_from_essential(e: &M3, pairs: &[([f64; 2], [f64; 2])]) -> (Pose, Vec<Option<V3>>) {
    let (mut u, _, mut v) = svd3(e);
    if det3(&u) < 0.0 {
        for r in u.iter_mut() {
            r[2] = -r[2];
        }
    }
    if det3(&v) < 0.0 {
        for r in v.iter_mut() {
            r[2] = -r[2];
        }
    }
    let w = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
    let r1 = m_mul(&u, &m_mul(&w, &m_t(&v)));
    let r2 = m_mul(&u, &m_mul(&m_t(&w), &m_t(&v)));
    let t = [u[0][2], u[1][2], u[2][2]];
    let mut best: Option<(usize, Pose, Vec<Option<V3>>)> = None;
    for r in [r1, r2] {
        for tt in [t, scl(t, -1.0)] {
            let p = Pose { r, t: tt };
            let pts: Vec<Option<V3>> = pairs
                .iter()
                .map(|(a, b)| {
                    let x = triangulate(&[Pose::IDENTITY, p], &[*a, *b])?;
                    (x[2] > 0.0 && p.apply(x)[2] > 0.0).then_some(x)
                })
                .collect();
            let n = pts.iter().filter(|x| x.is_some()).count();
            if best.as_ref().is_none_or(|(bn, _, _)| n > *bn) {
                best = Some((n, p, pts));
            }
        }
    }
    let (_, p, pts) = best.expect("four candidates");
    (p, pts)
}

// ------------------------------------------------------- the bundle solve

/// Huber: unit weight inside `delta` px, falling off past it.
fn huber_w(r: f64, delta: f64) -> f64 {
    if r <= delta {
        1.0
    } else {
        delta / r
    }
}

fn huber_cost(r: f64, delta: f64) -> f64 {
    if r <= delta {
        0.5 * r * r
    } else {
        delta * (r - 0.5 * delta)
    }
}

const HUBER: f64 = 2.0;

/// An observation for the bundle: camera index, point index, pixel.
#[derive(Clone, Copy)]
struct Obs {
    cam: usize,
    pt: usize,
    px: [f64; 2],
}

/// Residual and Jacobians of one observation: `(r, J_cam 2x6, J_pt 2x3,
/// J_f 2x1)`, residual = projection − pixel. None behind the camera.
#[allow(clippy::type_complexity)]
fn obs_jac(k: &Intrinsics, pose: &Pose, x: V3, px: [f64; 2]) -> Option<([f64; 2], [[f64; 6]; 2], [[f64; 3]; 2], [f64; 2])> {
    let pc = pose.apply(x);
    if pc[2] <= 1e-6 {
        return None;
    }
    let (xx, yy, z) = (pc[0], pc[1], pc[2]);
    let iz = 1.0 / z;
    let proj = [k.c[0] + k.f * xx * iz, k.c[1] + k.f * yy * iz];
    let r = [proj[0] - px[0], proj[1] - px[1]];
    // ∂π/∂p_c (2x3), scaled by f.
    let dp = [[k.f * iz, 0.0, -k.f * xx * iz * iz], [0.0, k.f * iz, -k.f * yy * iz * iz]];
    // ∂p_c/∂δ = −[p_c]×, ∂p_c/∂t = I.
    let dw = [[0.0, z, -yy], [-z, 0.0, xx], [yy, -xx, 0.0]];
    let mut jc = [[0.0; 6]; 2];
    for i in 0..2 {
        for j in 0..3 {
            jc[i][j] = dp[i][0] * dw[0][j] + dp[i][1] * dw[1][j] + dp[i][2] * dw[2][j];
            jc[i][3 + j] = dp[i][j];
        }
    }
    // ∂p_c/∂x = R.
    let mut jp = [[0.0; 3]; 2];
    for i in 0..2 {
        for j in 0..3 {
            jp[i][j] = dp[i][0] * pose.r[0][j] + dp[i][1] * pose.r[1][j] + dp[i][2] * pose.r[2][j];
        }
    }
    let jf = [xx * iz, yy * iz];
    Some((r, jc, jp, jf))
}

fn apply_delta(pose: &Pose, d: &[f64]) -> Pose {
    let dr = so3_exp([d[0], d[1], d[2]]);
    Pose { r: m_mul(&dr, &pose.r), t: add(m_v(&dr, pose.t), [d[3], d[4], d[5]]) }
}

fn inv3(m: &M3) -> Option<M3> {
    let d = det3(m);
    if d.abs() < 1e-300 || !d.is_finite() {
        return None;
    }
    let c = |a: usize, b: usize| {
        let (r1, r2) = ((a + 1) % 3, (a + 2) % 3);
        let (c1, c2) = ((b + 1) % 3, (b + 2) % 3);
        m[r1][c1] * m[r2][c2] - m[r1][c2] * m[r2][c1]
    };
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[j][i] = c(i, j) / d;
        }
    }
    Some(out)
}

/// The total robust cost of `obs` under `poses`, `points`, `k`.
fn cost_of(k: &Intrinsics, poses: &[Pose], points: &[V3], obs: &[Obs]) -> f64 {
    obs.iter()
        .map(|o| match k.project(poses[o.cam].apply(points[o.pt])) {
            Some(p) => huber_cost((p[0] - o.px[0]).hypot(p[1] - o.px[1]), HUBER),
            None => huber_cost(1e3, HUBER),
        })
        .sum()
}

/// Bundle adjustment: Levenberg–Marquardt over the free cameras
/// (`free[i]`), every point, and the focal (if `refine_focal`), the
/// points eliminated through their 3x3 blocks (Schur) so the linear
/// system is `6 · free (+1)` square. Edits in place.
fn bundle(
    k: &mut Intrinsics,
    poses: &mut [Pose],
    points: &mut [V3],
    obs: &[Obs],
    free: &[bool],
    refine_focal: bool,
    iters: usize,
) {
    let cam_slot: Vec<Option<usize>> = {
        let mut n = 0;
        free.iter()
            .map(|f| {
                if *f {
                    n += 1;
                    Some(n - 1)
                } else {
                    None
                }
            })
            .collect()
    };
    let nfree = cam_slot.iter().filter(|s| s.is_some()).count();
    let dim = 6 * nfree + usize::from(refine_focal);
    if dim == 0 || obs.is_empty() {
        return;
    }
    let np = points.len();
    // Observations grouped by point.
    let mut by_pt: Vec<Vec<usize>> = vec![Vec::new(); np];
    for (i, o) in obs.iter().enumerate() {
        by_pt[o.pt].push(i);
    }
    let mut lambda = 1e-3;
    let mut cur = cost_of(k, poses, points, obs);
    for _ in 0..iters {
        // Accumulate.
        let mut s = vec![0.0; dim * dim];
        let mut g = vec![0.0; dim];
        let mut vblk = vec![[[0.0; 3]; 3]; np];
        let mut gp = vec![[0.0; 3]; np];
        // Per observation: W (dim-rows x 3) is sparse: 6 rows at the camera
        // slot (+1 at the focal). Keep per-obs (jc, jp, jf, w, r).
        struct Term {
            slot: Option<usize>,
            jc: [[f64; 6]; 2],
            jp: [[f64; 3]; 2],
            jf: [f64; 2],
            w: f64,
        }
        let mut terms: Vec<Option<Term>> = Vec::with_capacity(obs.len());
        for o in obs {
            let Some((r, jc, jp, jf)) = obs_jac(k, &poses[o.cam], points[o.pt], o.px) else {
                terms.push(None);
                continue;
            };
            let w = huber_w((r[0] * r[0] + r[1] * r[1]).sqrt(), HUBER);
            let slot = cam_slot[o.cam];
            // Camera block + gradient.
            if let Some(sl) = slot {
                for i in 0..6 {
                    for j in 0..6 {
                        s[(6 * sl + i) * dim + 6 * sl + j] += w * (jc[0][i] * jc[0][j] + jc[1][i] * jc[1][j]);
                    }
                    g[6 * sl + i] -= w * (jc[0][i] * r[0] + jc[1][i] * r[1]);
                    if refine_focal {
                        let v = w * (jc[0][i] * jf[0] + jc[1][i] * jf[1]);
                        s[(6 * sl + i) * dim + dim - 1] += v;
                        s[(dim - 1) * dim + 6 * sl + i] += v;
                    }
                }
            }
            if refine_focal {
                s[(dim - 1) * dim + dim - 1] += w * (jf[0] * jf[0] + jf[1] * jf[1]);
                g[dim - 1] -= w * (jf[0] * r[0] + jf[1] * r[1]);
            }
            // Point block + gradient.
            let vb = &mut vblk[o.pt];
            for i in 0..3 {
                for j in 0..3 {
                    vb[i][j] += w * (jp[0][i] * jp[0][j] + jp[1][i] * jp[1][j]);
                }
                gp[o.pt][i] -= w * (jp[0][i] * r[0] + jp[1][i] * r[1]);
            }
            terms.push(Some(Term { slot, jc, jp, jf, w }));
        }
        // Damp the diagonals.
        for i in 0..dim {
            s[i * dim + i] *= 1.0 + lambda;
            s[i * dim + i] += 1e-9;
        }
        let vinv: Vec<Option<M3>> = vblk
            .iter()
            .map(|v| {
                let mut d = *v;
                for i in 0..3 {
                    d[i][i] = d[i][i] * (1.0 + lambda) + 1e-9;
                }
                inv3(&d)
            })
            .collect();
        // Schur: S −= Σ_j W_j V_j⁻¹ W_jᵀ, g −= Σ_j W_j V_j⁻¹ g_pj, where W_j
        // is the stack of the point's observations' camera rows.
        // A row of W_j for observation o: index (slot rows or focal row) →
        // the 3-vector w · J_cᵀ J_p (6 of them) or w · J_fᵀ J_p.
        for (j, list) in by_pt.iter().enumerate() {
            let Some(vi) = vinv[j] else { continue };
            // Collect the W rows of this point: (row index, [f64;3]).
            let mut rows: Vec<(usize, [f64; 3])> = Vec::new();
            for &oi in list {
                let Some(t) = &terms[oi] else { continue };
                if let Some(sl) = t.slot {
                    for i in 0..6 {
                        let mut v = [0.0; 3];
                        for c in 0..3 {
                            v[c] = t.w * (t.jc[0][i] * t.jp[0][c] + t.jc[1][i] * t.jp[1][c]);
                        }
                        rows.push((6 * sl + i, v));
                    }
                }
                if refine_focal {
                    let mut v = [0.0; 3];
                    for c in 0..3 {
                        v[c] = t.w * (t.jf[0] * t.jp[0][c] + t.jf[1] * t.jp[1][c]);
                    }
                    rows.push((dim - 1, v));
                }
            }
            if rows.is_empty() {
                continue;
            }
            let vg = m_v(&vi, gp[j]);
            for (ra, wa) in &rows {
                let wav = m_v(&vi, *wa);
                g[*ra] -= dot(wav, gp[j]);
                let _ = vg;
                for (rb, wb) in &rows {
                    s[*ra * dim + *rb] -= dot(wav, *wb);
                }
            }
        }
        let mut sm = s.clone();
        let mut gm = g.clone();
        let Some(dc) = crate::planar::solve_linear(&mut sm, &mut gm, dim) else {
            lambda *= 10.0;
            if lambda > 1e8 {
                break;
            }
            continue;
        };
        if dc.iter().any(|v| !v.is_finite()) {
            lambda *= 10.0;
            continue;
        }
        // Back-substitute the points.
        let mut dp = vec![[0.0; 3]; np];
        for (j, list) in by_pt.iter().enumerate() {
            let Some(vi) = vinv[j] else { continue };
            let mut rhs = gp[j];
            for &oi in list {
                let Some(t) = &terms[oi] else { continue };
                if let Some(sl) = t.slot {
                    for i in 0..6 {
                        for c in 0..3 {
                            rhs[c] -= t.w * (t.jc[0][i] * t.jp[0][c] + t.jc[1][i] * t.jp[1][c]) * dc[6 * sl + i];
                        }
                    }
                }
                if refine_focal {
                    for c in 0..3 {
                        rhs[c] -= t.w * (t.jf[0] * t.jp[0][c] + t.jf[1] * t.jp[1][c]) * dc[dim - 1];
                    }
                }
            }
            dp[j] = m_v(&vi, rhs);
        }
        // Try the step.
        let new_poses: Vec<Pose> = poses
            .iter()
            .enumerate()
            .map(|(i, p)| match cam_slot[i] {
                Some(sl) => apply_delta(p, &dc[6 * sl..6 * sl + 6]),
                None => *p,
            })
            .collect();
        let new_points: Vec<V3> = points.iter().zip(&dp).map(|(p, d)| add(*p, *d)).collect();
        let mut nk = *k;
        if refine_focal {
            nk.f = (k.f + dc[dim - 1]).max(k.f * 0.5);
        }
        let next = cost_of(&nk, &new_poses, &new_points, obs);
        if std::env::var_os("WOBBLE_CAMTRACK_TRACE").is_some() {
            eprintln!("bundle: {} cams {} pts {} obs: cost {cur:.6} -> {next:.6} lambda {lambda:.2e} f {:.2}", nfree, np, obs.len(), nk.f);
        }
        if next < cur && next.is_finite() {
            let gain = (cur - next) / cur.max(1e-300);
            poses.copy_from_slice(&new_poses);
            points.copy_from_slice(&new_points);
            *k = nk;
            cur = next;
            lambda = (lambda / 3.0).max(1e-9);
            if gain < 1e-8 {
                break;
            }
        } else {
            lambda *= 4.0;
            if lambda > 1e8 {
                break;
            }
        }
    }
}

/// Refine one pose against fixed points (a bundle with everything else
/// held): the camera-only Levenberg–Marquardt.
fn refine_pose(k: &Intrinsics, pose: &mut Pose, pts: &[(V3, [f64; 2])], iters: usize) -> f64 {
    let cost = |p: &Pose| -> f64 {
        pts.iter()
            .map(|(x, px)| match k.project(p.apply(*x)) {
                Some(q) => huber_cost((q[0] - px[0]).hypot(q[1] - px[1]), HUBER),
                None => huber_cost(1e3, HUBER),
            })
            .sum()
    };
    let mut lambda = 1e-3;
    let mut cur = cost(pose);
    for _ in 0..iters {
        let mut a = [0.0; 36];
        let mut b = [0.0; 6];
        for (x, px) in pts {
            let Some((r, jc, _, _)) = obs_jac(k, pose, *x, *px) else { continue };
            let w = huber_w((r[0] * r[0] + r[1] * r[1]).sqrt(), HUBER);
            for i in 0..6 {
                for j in 0..6 {
                    a[i * 6 + j] += w * (jc[0][i] * jc[0][j] + jc[1][i] * jc[1][j]);
                }
                b[i] -= w * (jc[0][i] * r[0] + jc[1][i] * r[1]);
            }
        }
        for i in 0..6 {
            a[i * 6 + i] = a[i * 6 + i] * (1.0 + lambda) + 1e-12;
        }
        let Some(d) = crate::planar::solve_linear(&mut a, &mut b, 6) else {
            lambda *= 10.0;
            if lambda > 1e8 {
                break;
            }
            continue;
        };
        let next_pose = apply_delta(pose, &d);
        let next = cost(&next_pose);
        if next < cur && next.is_finite() {
            let gain = (cur - next) / cur.max(1e-300);
            *pose = next_pose;
            cur = next;
            lambda = (lambda / 3.0).max(1e-9);
            if gain < 1e-7 {
                break;
            }
        } else {
            lambda *= 4.0;
            if lambda > 1e8 {
                break;
            }
        }
    }
    cur
}

/// The mean pixel error of `pts` under `pose`, and the share within 4 px.
fn pose_fit(k: &Intrinsics, pose: &Pose, pts: &[(V3, [f64; 2])]) -> (f64, f64) {
    let mut sum = 0.0;
    let mut good = 0usize;
    for (x, px) in pts {
        let e = match k.project(pose.apply(*x)) {
            Some(q) => (q[0] - px[0]).hypot(q[1] - px[1]),
            None => 1e3,
        };
        sum += e;
        if e < 4.0 {
            good += 1;
        }
    }
    let n = pts.len().max(1) as f64;
    (sum / n, good as f64 / n)
}

/// Solve the camera from `tracks` (see the module docs). Errors when the
/// tracks hold no initial pair with parallax.
pub fn solve(tracks: &[Track], opts: &SolveOpts, progress: &mut dyn FnMut(usize, usize), cancel: &AtomicBool) -> Result<Solve> {
    let n = tracks.iter().map(|t| t.end()).max().unwrap_or(0);
    anyhow::ensure!(n >= 2, "the tracks cover {n} frame(s); a camera needs at least two");
    let mut k = Intrinsics::for_size(opts.focal_px.max(1.0), opts.size);
    let width = opts.size[0] as f64;
    let check = |done: usize, progress: &mut dyn FnMut(usize, usize)| -> Result<()> {
        progress(done, n);
        anyhow::ensure!(!cancel.load(Ordering::Relaxed), "cancelled");
        Ok(())
    };
    // Keyframes: evenly spaced, the first and last always.
    let stride = ((n - 1) as f64 / (opts.max_keyframes.max(2) - 1) as f64).ceil().max(1.0) as usize;
    let mut keyframes: Vec<usize> = (0..n).step_by(stride).collect();
    if *keyframes.last().unwrap() != n - 1 {
        keyframes.push(n - 1);
    }
    let is_key = |f: usize| keyframes.contains(&f);
    // ---- initial pair: an early keyframe against a later one, the pair
    // with the most parallax among those sharing enough tracks. The world
    // is the first of the pair's camera until the end, when everything is
    // re-based onto frame 0.
    let shared = |a: usize, b: usize| -> Vec<(usize, [f64; 2], [f64; 2])> {
        tracks
            .iter()
            .enumerate()
            .filter_map(|(i, t)| Some((i, t.at(a)?, t.at(b)?)))
            .collect()
    };
    let median = |mut v: Vec<f64>| -> f64 {
        if v.is_empty() {
            return 0.0;
        }
        v.sort_by(|a, b| a.total_cmp(b));
        v[v.len() / 2]
    };
    let mut init: Option<(usize, usize, Pose, Vec<(usize, Option<V3>)>)> = None;
    let mut best_score = 0.0f64;
    let firsts: Vec<usize> = keyframes.iter().copied().take(4).collect();
    'pairs: for &a in &firsts {
        for &b in keyframes.iter().filter(|&&b| b > a).take(12) {
            let sh = shared(a, b);
            if sh.len() < 24 {
                continue;
            }
            let disp = median(sh.iter().map(|(_, p, q)| (q[0] - p[0]).hypot(q[1] - p[1])).collect());
            if disp < width * 0.01 {
                continue;
            }
            let pairs: Vec<([f64; 2], [f64; 2])> = sh.iter().map(|(_, p, q)| (k.normalise(*p), k.normalise(*q))).collect();
            let Some((e, inl)) = essential_ransac(&pairs, (1.5 / k.f).powi(2), 400) else { continue };
            let inliers: Vec<([f64; 2], [f64; 2])> = pairs.iter().zip(&inl).filter(|(_, i)| **i).map(|(p, _)| *p).collect();
            if inliers.len() < 16 {
                continue;
            }
            let (pose, pts) = pose_from_essential(&e, &inliers);
            // Score: parallax at the triangulated points (a pure rotation
            // scores nothing), over the share of inliers.
            let angles: Vec<f64> = pts.iter().flatten().map(|x| parallax(&Pose::IDENTITY, &pose, *x)).collect();
            let front = angles.len() as f64 / inliers.len().max(1) as f64;
            let score = median(angles) * front;
            if score > best_score {
                best_score = score;
                let mut idx = sh.iter().zip(&inl).filter(|(_, i)| **i).map(|((i, _, _), _)| *i);
                let placed: Vec<(usize, Option<V3>)> = pts.iter().map(|x| (idx.next().unwrap(), *x)).collect();
                init = Some((a, b, pose, placed));
            }
            // Two degrees of median parallax is plenty.
            if best_score > 2f64.to_radians() {
                break 'pairs;
            }
        }
    }
    let (a0, b0, pose_b, placed) = init.context(
        "no pair of frames with enough parallax: the camera has to MOVE (a pan or a lock-off has no 3D to solve)",
    )?;
    // ---- state
    let mut poses: Vec<Option<Pose>> = vec![None; n];
    poses[a0] = Some(Pose::IDENTITY);
    poses[b0] = Some(pose_b);
    // Per track: its 3D point, if triangulated.
    let mut pt_of: Vec<Option<usize>> = vec![None; tracks.len()];
    let mut points: Vec<V3> = Vec::new();
    let mut point_track: Vec<usize> = Vec::new();
    for (ti, x) in placed {
        if let Some(x) = x {
            pt_of[ti] = Some(points.len());
            points.push(x);
            point_track.push(ti);
        }
    }
    // The points of a frame: (3D, pixel) for every track with a point.
    let pts_in = |f: usize, pt_of: &[Option<usize>], points: &[V3]| -> Vec<(V3, [f64; 2])> {
        tracks
            .iter()
            .zip(pt_of)
            .filter_map(|(t, p)| Some((points[(*p)?], t.at(f)?)))
            .collect()
    };
    let obs_for = |frames: &[usize], pt_of: &[Option<usize>]| -> Vec<Obs> {
        let mut out = Vec::new();
        for (ti, t) in tracks.iter().enumerate() {
            let Some(pt) = pt_of[ti] else { continue };
            for &f in frames {
                if let Some(px) = t.at(f) {
                    out.push(Obs { cam: f, pt, px });
                }
            }
        }
        out
    };
    // Triangulate every track seen by at least two placed frames with a
    // baseline, that has no point yet (or re-triangulate all when `all`).
    let triangulate_new = |pt_of: &mut Vec<Option<usize>>, points: &mut Vec<V3>, point_track: &mut Vec<usize>, poses: &[Option<Pose>], k: &Intrinsics, all: bool| {
        for (ti, t) in tracks.iter().enumerate() {
            if pt_of[ti].is_some() && !all {
                continue;
            }
            let views: Vec<(usize, Pose, [f64; 2])> = (t.start..t.end())
                .filter_map(|f| Some((f, poses[f]?, k.normalise(t.pts[f - t.start]))))
                .collect();
            if views.len() < 2 {
                continue;
            }
            // Widest baseline pair first, then all views.
            let mut best = (0.0, 0usize, 0usize);
            for a in 0..views.len() {
                for b in a + 1..views.len() {
                    let d = norm(sub(views[a].1.centre(), views[b].1.centre()));
                    if d > best.0 {
                        best = (d, a, b);
                    }
                }
            }
            let (a, b) = (best.1, best.2);
            let Some(x) = triangulate(&[views[a].1, views[b].1], &[views[a].2, views[b].2]) else { continue };
            if parallax(&views[a].1, &views[b].1, x) < 0.5f64.to_radians() {
                continue;
            }
            let ps: Vec<Pose> = views.iter().map(|v| v.1).collect();
            let os: Vec<[f64; 2]> = views.iter().map(|v| v.2).collect();
            let x = triangulate(&ps, &os).unwrap_or(x);
            // In front of every view and reprojecting well.
            let ok = views.iter().all(|(f, p, _)| {
                let pc = p.apply(x);
                pc[2] > 1e-6 && k.project(pc).is_some_and(|q| {
                    let px = t.pts[f - t.start];
                    (q[0] - px[0]).hypot(q[1] - px[1]) < 6.0
                })
            });
            if !ok {
                continue;
            }
            match pt_of[ti] {
                Some(i) => points[i] = x,
                None => {
                    pt_of[ti] = Some(points.len());
                    points.push(x);
                    point_track.push(ti);
                }
            }
        }
    };
    // Drop points that reproject badly in any placed keyframe.
    let prune = |pt_of: &mut Vec<Option<usize>>, points: &mut Vec<V3>, point_track: &mut Vec<usize>, poses: &[Option<Pose>], k: &Intrinsics, frames: &[usize], tol: f64| {
        let mut keep = vec![true; points.len()];
        for (pi, &ti) in point_track.iter().enumerate() {
            let t = &tracks[ti];
            let mut worst = 0.0f64;
            let mut seen = 0;
            for &f in frames {
                let (Some(px), Some(p)) = (t.at(f), poses[f]) else { continue };
                seen += 1;
                let e = match k.project(p.apply(points[pi])) {
                    Some(q) => (q[0] - px[0]).hypot(q[1] - px[1]),
                    None => 1e3,
                };
                worst = worst.max(e);
            }
            if seen >= 2 && worst > tol {
                keep[pi] = false;
            }
        }
        let mut remap = vec![None; points.len()];
        let mut np = Vec::new();
        let mut nt = Vec::new();
        for pi in 0..points.len() {
            if keep[pi] {
                remap[pi] = Some(np.len());
                np.push(points[pi]);
                nt.push(point_track[pi]);
            } else {
                pt_of[point_track[pi]] = None;
            }
        }
        for p in pt_of.iter_mut() {
            if let Some(i) = *p {
                *p = remap[i];
            }
        }
        *points = np;
        *point_track = nt;
    };
    // A frame placed from its nearest placed neighbour's pose.
    let place = |f: usize, poses: &mut Vec<Option<Pose>>, pt_of: &[Option<usize>], points: &[V3], k: &Intrinsics, unplaced: &mut Vec<usize>| {
        let seed = (1..n)
            .flat_map(|d| [f.checked_sub(d), (f + d < n).then_some(f + d)])
            .flatten()
            .find_map(|g| poses[g]);
        let Some(seed) = seed else { return };
        let pts = pts_in(f, pt_of, points);
        if pts.len() < 6 {
            poses[f] = Some(seed);
            unplaced.push(f);
            return;
        }
        let mut p = seed;
        refine_pose(k, &mut p, &pts, 30);
        let (_, good) = pose_fit(k, &p, &pts);
        if good < 0.5 {
            poses[f] = Some(seed);
            unplaced.push(f);
        } else {
            poses[f] = Some(p);
        }
    };
    let mut unplaced = Vec::new();
    // ---- the first stretch: 1..b0, then onwards, bundling keyframes as
    // they arrive (a window of the last WINDOW keyframes).
    const WINDOW: usize = 8;
    let mut done_keys: Vec<usize> = vec![a0, b0];
    // (bundle over Option poses: a helper copies in and out.)
    fn poses_vec(p: &[Option<Pose>]) -> Vec<Pose> {
        p.iter().map(|q| q.unwrap_or(Pose::IDENTITY)).collect()
    }
    fn poses_back(p: &mut [Option<Pose>], v: &[Pose]) {
        for (a, b) in p.iter_mut().zip(v) {
            if a.is_some() {
                *a = Some(*b);
            }
        }
    }
    let bundle_window = |poses: &mut Vec<Option<Pose>>, points: &mut Vec<V3>, pt_of: &[Option<usize>], k: &mut Intrinsics, keys: &[usize], free_from: usize, refine_focal: bool, iters: usize| {
        let obs = obs_for(keys, pt_of);
        let mut v = poses_vec(poses);
        let mut free = vec![false; n];
        for &f in keys {
            if f >= free_from && f != a0 {
                free[f] = true;
            }
        }
        bundle(k, &mut v, points, &obs, &free, refine_focal, iters);
        poses_back(poses, &v);
    };
    // Outward from the pair: between them, before, then after.
    let order: Vec<usize> = (a0 + 1..b0).chain((0..a0).rev()).chain(b0 + 1..n).collect();
    for (step, f) in order.iter().copied().enumerate() {
        check(step + 1, progress)?;
        place(f, &mut poses, &pt_of, &points, &k, &mut unplaced);
        if is_key(f) {
            done_keys.push(f);
            done_keys.sort_unstable();
            done_keys.dedup();
            triangulate_new(&mut pt_of, &mut points, &mut point_track, &poses, &k, false);
            let from = done_keys.len().saturating_sub(WINDOW);
            let window: Vec<usize> = done_keys[from..].to_vec();
            let free_from = window[0];
            bundle_window(&mut poses, &mut points, &pt_of, &mut k, &window, free_from, false, 8);
            prune(&mut pt_of, &mut points, &mut point_track, &poses, &k, &window, 8.0);
        } else if f > b0 && (f - b0) % 4 == 0 {
            // Between keyframes new tracks still want points, so the next
            // frames have something to hold onto.
            triangulate_new(&mut pt_of, &mut points, &mut point_track, &poses, &k, false);
        }
    }
    // ---- global passes over every keyframe.
    for pass in 0..2 {
        triangulate_new(&mut pt_of, &mut points, &mut point_track, &poses, &k, pass == 0);
        bundle_window(&mut poses, &mut points, &pt_of, &mut k, &keyframes, 0, opts.refine_focal, 50);
        prune(&mut pt_of, &mut points, &mut point_track, &poses, &k, &keyframes, if pass == 0 { 4.0 } else { 3.0 });
        check(n, progress)?;
    }
    // ---- every other frame against the final points.
    unplaced.clear();
    for f in 0..n {
        if is_key(f) {
            continue;
        }
        let pts = pts_in(f, &pt_of, &points);
        let Some(mut p) = poses[f] else {
            place(f, &mut poses, &pt_of, &points, &k, &mut unplaced);
            continue;
        };
        if pts.len() < 6 {
            unplaced.push(f);
            continue;
        }
        refine_pose(&k, &mut p, &pts, 30);
        poses[f] = Some(p);
    }
    // Frames never placed take a neighbour's pose.
    for f in 0..n {
        if poses[f].is_none() {
            let seed = (1..n)
                .flat_map(|d| [f.checked_sub(d), (f + d < n).then_some(f + d)])
                .flatten()
                .find_map(|g| poses[g]);
            poses[f] = Some(seed.unwrap_or(Pose::IDENTITY));
            if !unplaced.contains(&f) {
                unplaced.push(f);
            }
        }
    }
    unplaced.sort_unstable();
    unplaced.dedup();
    let mut poses: Vec<Pose> = poses.into_iter().map(|p| p.unwrap()).collect();
    // ---- re-base: the world is frame 0's camera.
    let base = poses[0].inverse();
    for p in &mut poses {
        *p = p.then(&base);
    }
    let base_inv = base.inverse();
    for x in &mut points {
        *x = base_inv.apply(*x);
    }
    // ---- scale: the median depth from the first camera is 1.
    let depth = median(points.iter().map(|x| x[2]).filter(|z| *z > 0.0).collect());
    if depth > 1e-9 {
        for p in &mut poses {
            p.t = scl(p.t, 1.0 / depth);
        }
        for x in &mut points {
            *x = scl(*x, 1.0 / depth);
        }
    }
    // ---- errors per frame.
    let errors: Vec<f64> = (0..n)
        .map(|f| {
            let pts = pts_in(f, &pt_of, &points);
            if pts.is_empty() {
                f64::NAN
            } else {
                pose_fit(&k, &poses[f], &pts).0
            }
        })
        .collect();
    Ok(Solve { size: opts.size, focal_px: k.f, poses, unplaced, points, point_track, errors })
}

/// A lens guess: [`solve`] from each of `starts_px` with the focal free,
/// keeping the best fit (mean error, an unplaced frame counting as a
/// pixel). `progress` counts the starts.
pub fn solve_guess(
    tracks: &[Track],
    opts: &SolveOpts,
    starts_px: &[f64],
    progress: &mut dyn FnMut(usize, usize),
    cancel: &AtomicBool,
) -> Result<Solve> {
    anyhow::ensure!(!starts_px.is_empty(), "no focal lengths to try");
    let mut best: Option<(f64, Solve)> = None;
    let mut last_err = None;
    for (i, f) in starts_px.iter().enumerate() {
        progress(i, starts_px.len());
        let o = SolveOpts { focal_px: *f, refine_focal: true, ..opts.clone() };
        match solve(tracks, &o, &mut |_, _| {}, cancel) {
            Ok(s) => {
                let score = s.mean_error() + s.unplaced.len() as f64;
                if best.as_ref().is_none_or(|(b, _)| score < *b) {
                    best = Some((score, s));
                }
            }
            Err(e) => {
                if cancel.load(Ordering::Relaxed) {
                    return Err(e);
                }
                last_err = Some(e);
            }
        }
    }
    progress(starts_px.len(), starts_px.len());
    match best {
        Some((_, s)) => Ok(s),
        None => Err(last_err.unwrap_or_else(|| anyhow::anyhow!("no start solved"))),
    }
}

// ------------------------------------------------------------- tracking

/// Forward–backward tolerance: a point whose round trip lands further
/// than this (full-size px, plus a share of the move) is dropped.
const FB_TOL: f64 = 0.75;
const FB_SHARE: f64 = 0.02;
/// Seeding grid cell, in ANALYSED pixels.
const CELL: usize = 24;
/// Border left alone, as a share of each side.
const MARGIN: f64 = 0.03;

/// What to track.
pub struct Job {
    pub weights: PathBuf,
    /// The frames, in time order (linear Rec.709 EXRs).
    pub frames: Vec<PathBuf>,
    /// Per frame, a matte whose red > 0.5 keeps a pixel out.
    pub mattes: Option<Vec<PathBuf>>,
    /// Where `tracks.txt` and the done-marker go.
    pub out: PathBuf,
    pub divisor: usize,
}

/// Bilinear sample of plane `c` of `[2][h][w]` at analysed coordinates.
fn sample(flow: &[f32], w: usize, h: usize, c: usize, x: f64, y: f64) -> f64 {
    let fx = (x - 0.5).clamp(0.0, (w - 1) as f64);
    let fy = (y - 0.5).clamp(0.0, (h - 1) as f64);
    let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
    let (x1, y1) = ((x0 + 1).min(w - 1), (y0 + 1).min(h - 1));
    let (tx, ty) = (fx - x0 as f64, fy - y0 as f64);
    let at = |xx: usize, yy: usize| flow[c * w * h + yy * w + xx] as f64;
    (at(x0, y0) * (1.0 - tx) + at(x1, y0) * tx) * (1.0 - ty) + (at(x0, y1) * (1.0 - tx) + at(x1, y1) * tx) * ty
}

/// Carry `live` points (analysed px) through `fwd`, checked against
/// `bwd`; a point that fails is closed (returns None in its slot).
fn carry(fwd: &[f32], bwd: &[f32], w: usize, h: usize, p: [f64; 2], d: f64) -> Option<[f64; 2]> {
    let (mx, my) = (w as f64 * MARGIN, h as f64 * MARGIN);
    let q = [p[0] + sample(fwd, w, h, 0, p[0], p[1]), p[1] + sample(fwd, w, h, 1, p[0], p[1])];
    if !(q[0] > mx && q[0] < w as f64 - mx && q[1] > my && q[1] < h as f64 - my) {
        return None;
    }
    let back = [q[0] + sample(bwd, w, h, 0, q[0], q[1]), q[1] + sample(bwd, w, h, 1, q[0], q[1])];
    let err = (back[0] - p[0]).hypot(back[1] - p[1]) * d;
    let moved = (q[0] - p[0]).hypot(q[1] - p[1]) * d;
    (err < FB_TOL + FB_SHARE * moved).then_some(q)
}

/// Seed one point per empty cell on the strongest gradient in it.
fn seed(gray: &[f32], w: usize, h: usize, live: &[[f64; 2]], keep: &dyn Fn(f64, f64) -> bool) -> Vec<[f64; 2]> {
    let (cx, cy) = (w.div_ceil(CELL), h.div_ceil(CELL));
    let mut taken = vec![false; cx * cy];
    for p in live {
        let (i, j) = ((p[0] as usize / CELL).min(cx - 1), (p[1] as usize / CELL).min(cy - 1));
        taken[j * cx + i] = true;
    }
    let (mx, my) = ((w as f64 * MARGIN) as usize + 1, (h as f64 * MARGIN) as usize + 1);
    let mut out = Vec::new();
    for j in 0..cy {
        for i in 0..cx {
            if taken[j * cx + i] {
                continue;
            }
            let mut best = (0.0f32, None);
            for y in (j * CELL).max(my)..((j + 1) * CELL).min(h - my) {
                for x in (i * CELL).max(mx)..((i + 1) * CELL).min(w - mx) {
                    let gx = gray[y * w + x + 1] - gray[y * w + x - 1];
                    let gy = gray[(y + 1) * w + x] - gray[(y - 1) * w + x];
                    let g = gx * gx + gy * gy;
                    if g > best.0 {
                        best = (g, Some([x as f64 + 0.5, y as f64 + 0.5]));
                    }
                }
            }
            if let (g, Some(p)) = best {
                if g > 1e-6 && keep(p[0], p[1]) {
                    out.push(p);
                }
            }
        }
    }
    out
}

/// Track `job`'s frames, reporting `(done, total)` per pair. Returns the
/// tracks (full-size px) and the frames' size.
pub fn track(
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    job: &Job,
    progress: &mut dyn FnMut(usize, usize),
    cancel: &AtomicBool,
) -> Result<(Vec<Track>, [u32; 2])> {
    let n_frames = job.frames.len();
    anyhow::ensure!(n_frames >= 2, "tracking needs at least two frames, the range has {n_frames}");
    std::fs::create_dir_all(&job.out).with_context(|| format!("creating {}", job.out.display()))?;
    let dec = crate::decode::AnyDecoder::default();
    let first = dec
        .read_header(&job.frames[0])
        .with_context(|| format!("reading {}", job.frames[0].display()))?;
    let (w, h) = (first.width as usize, first.height as usize);
    let mut d = job.divisor.max(1);
    let max = device.limits().max_storage_buffer_binding_size as u64;
    while crate::searaft::largest_buffer(w.div_ceil(d), h.div_ceil(d)) > max && d < 16 {
        d *= 2;
    }
    while d > 1 && (w.div_ceil(d) < 128 || h.div_ceil(d) < 128) {
        d /= 2;
    }
    let (sw, sh) = (w.div_ceil(d), h.div_ceil(d));
    let get = |i: usize| -> Result<Vec<f32>> {
        let (fw, fh, rgb) = crate::flowsolve::load(&dec, &job.frames[i])?;
        anyhow::ensure!((fw, fh) == (w, h), "frame {i} is {fw}x{fh}, the first was {w}x{h}");
        Ok(crate::flowsolve::shrink(&rgb, 3, w, h, d).2)
    };
    let gray_of = |rgb: &[f32]| -> Vec<f32> {
        let n = sw * sh;
        (0..n).map(|i| 0.299 * rgb[i] + 0.587 * rgb[n + i] + 0.114 * rgb[2 * n + i]).collect()
    };
    let m = crate::searaft::Model::load(&job.weights).context("loading the SEA-RAFT weights")?;
    let net = crate::searaft::Gpu::new(device, queue, &m, sw, sh)?;
    let k = d as f64;
    let matte_at = |i: usize| -> Result<Option<(usize, usize, Vec<f32>)>> {
        match &job.mattes {
            Some(ms) => Ok(Some(crate::stabilize::load_matte(&dec, &ms[i])?)),
            None => Ok(None),
        }
    };
    let keep_of = |m: &Option<(usize, usize, Vec<f32>)>, x: f64, y: f64| -> bool {
        match m {
            Some((mw, mh, m)) => {
                let (ix, iy) = (((x * k) as usize).min(mw - 1), ((y * k) as usize).min(mh - 1));
                m[iy * mw + ix] <= 0.5
            }
            None => true,
        }
    };
    let mut tracks: Vec<Track> = Vec::new();
    // Live tracks: index into `tracks`, position in analysed px.
    let mut live: Vec<(usize, [f64; 2])> = Vec::new();
    let mut cur = get(0)?;
    let m0 = matte_at(0)?;
    for p in seed(&gray_of(&cur), sw, sh, &[], &|x, y| keep_of(&m0, x, y)) {
        live.push((tracks.len(), p));
        tracks.push(Track { start: 0, pts: vec![[p[0] * k, p[1] * k]] });
    }
    for i in 0..n_frames - 1 {
        if cancel.load(Ordering::Relaxed) {
            anyhow::bail!("cancelled");
        }
        let next = get(i + 1)?;
        let mut fwd = net.run(&cur, &next)?;
        fwd.truncate(2 * sw * sh);
        let mut bwd = net.run(&next, &cur)?;
        bwd.truncate(2 * sw * sh);
        let matte = matte_at(i + 1)?;
        let mut still: Vec<(usize, [f64; 2])> = Vec::new();
        for (ti, p) in &live {
            if let Some(q) = carry(&fwd, &bwd, sw, sh, *p, k) {
                if keep_of(&matte, q[0], q[1]) {
                    tracks[*ti].pts.push([q[0] * k, q[1] * k]);
                    still.push((*ti, q));
                }
            }
        }
        live = still;
        let have: Vec<[f64; 2]> = live.iter().map(|l| l.1).collect();
        for p in seed(&gray_of(&next), sw, sh, &have, &|x, y| keep_of(&matte, x, y)) {
            live.push((tracks.len(), p));
            tracks.push(Track { start: i + 1, pts: vec![[p[0] * k, p[1] * k]] });
        }
        cur = next;
        progress(i + 1, n_frames - 1);
    }
    // A track of two frames or fewer says nothing a longer one does not.
    tracks.retain(|t| t.pts.len() >= 3);
    write_tracks(&job.out, &tracks, [w as u32, h as u32])?;
    crate::bakes::mark_baked(&job.out).context("marking the tracking done")?;
    Ok((tracks, [w as u32, h as u32]))
}

/// `tracks.txt`: `size w h`, then per track `start x y x y …`.
pub fn write_tracks(dir: &Path, tracks: &[Track], size: [u32; 2]) -> Result<()> {
    let mut s = format!("size {} {}\n", size[0], size[1]);
    for t in tracks {
        s.push_str(&t.start.to_string());
        for p in &t.pts {
            s.push_str(&format!(" {:.3} {:.3}", p[0], p[1]));
        }
        s.push('\n');
    }
    std::fs::write(dir.join("tracks.txt"), s).with_context(|| format!("writing {}", dir.display()))
}

pub fn read_tracks(dir: &Path) -> Result<(Vec<Track>, [u32; 2])> {
    let text = std::fs::read_to_string(dir.join("tracks.txt")).with_context(|| format!("reading {}", dir.display()))?;
    let mut lines = text.lines();
    let head: Vec<u32> =
        lines.next().unwrap_or("").split_whitespace().skip(1).filter_map(|v| v.parse().ok()).collect();
    anyhow::ensure!(head.len() == 2, "tracks.txt has no size");
    let mut tracks = Vec::new();
    for l in lines {
        let v: Vec<f64> = l.split_whitespace().filter_map(|v| v.parse().ok()).collect();
        anyhow::ensure!(v.len() >= 3 && v.len() % 2 == 1, "a track row of {} numbers", v.len());
        let pts = v[1..].chunks(2).map(|c| [c[0], c[1]]).collect();
        tracks.push(Track { start: v[0] as usize, pts });
    }
    Ok((tracks, [head[0], head[1]]))
}

/// `solve.txt`: `size w h`, `focal f`, `unplaced …`, then per frame
/// `pose r00 … r22 t0 t1 t2 err`, then `points n` and one `x y z` a line.
pub fn write_solve(dir: &Path, s: &Solve) -> Result<()> {
    let mut out = format!("size {} {}\nfocal {:e}\nunplaced", s.size[0], s.size[1], s.focal_px);
    for u in &s.unplaced {
        out.push_str(&format!(" {u}"));
    }
    out.push('\n');
    for (p, e) in s.poses.iter().zip(&s.errors) {
        out.push_str("pose");
        for v in p.r.iter().flatten().chain(p.t.iter()) {
            out.push_str(&format!(" {v:e}"));
        }
        out.push_str(&format!(" {e:e}\n"));
    }
    out.push_str(&format!("points {}\n", s.points.len()));
    for (x, t) in s.points.iter().zip(&s.point_track) {
        out.push_str(&format!("{:e} {:e} {:e} {t}\n", x[0], x[1], x[2]));
    }
    std::fs::write(dir.join("solve.txt"), out).with_context(|| format!("writing {}", dir.display()))
}

pub fn read_solve(dir: &Path) -> Result<Solve> {
    let text = std::fs::read_to_string(dir.join("solve.txt")).with_context(|| format!("reading {}", dir.display()))?;
    let mut s = Solve::default();
    let mut in_points = false;
    for l in text.lines() {
        let mut it = l.split_whitespace();
        let Some(tag) = it.next() else { continue };
        let nums: Vec<f64> = it.filter_map(|v| v.parse().ok()).collect();
        if in_points {
            let all: Vec<f64> = std::iter::once(tag.parse::<f64>().ok()).flatten().chain(nums).collect();
            anyhow::ensure!(all.len() == 4, "a point of {} numbers", all.len());
            s.points.push([all[0], all[1], all[2]]);
            s.point_track.push(all[3] as usize);
            continue;
        }
        match tag {
            "size" => {
                anyhow::ensure!(nums.len() == 2, "solve.txt has no size");
                s.size = [nums[0] as u32, nums[1] as u32];
            }
            "focal" => s.focal_px = nums.first().copied().unwrap_or(0.0),
            "unplaced" => s.unplaced = nums.iter().map(|v| *v as usize).collect(),
            "pose" => {
                anyhow::ensure!(nums.len() == 13, "a pose of {} numbers", nums.len());
                let r = [[nums[0], nums[1], nums[2]], [nums[3], nums[4], nums[5]], [nums[6], nums[7], nums[8]]];
                s.poses.push(Pose { r, t: [nums[9], nums[10], nums[11]] });
                s.errors.push(nums[12]);
            }
            "points" => in_points = true,
            _ => {}
        }
    }
    Ok(s)
}

// ------------------------------------------------------------- synthetic

/// A synthetic scene for tests: points on three textured planes at
/// different depths (a far wall, a floor, a near card), a camera that
/// dollies sideways and turns a little. `truth` is the camera per frame
/// (world → camera, CV, world = the first camera's frame) and the focal.
pub mod synth {
    use super::*;

    pub const F: f64 = 1100.0;

    /// The true camera at frame `t` of `n`.
    pub fn camera(t: usize, n: usize) -> Pose {
        let s = t as f64 / (n.max(2) - 1) as f64;
        let c = [0.9 * s, 0.15 * (s * 6.0).sin(), 0.25 * s];
        let r = so3_exp([0.03 * (s * 4.0).sin(), -0.22 * s, 0.02 * s]);
        // Pose: x_c = R (x − c).
        Pose { r, t: scl(m_v(&r, c), -1.0) }
    }

    /// Points on the three planes, in the first camera's frame.
    pub fn points(seed: u64) -> Vec<V3> {
        let mut rng = Rng(seed | 1);
        let mut out = Vec::new();
        let unif = |rng: &mut Rng| (rng.next() % 1_000_000) as f64 / 1_000_000.0;
        for _ in 0..600 {
            let u = unif(&mut rng) * 2.0 - 1.0;
            let v = unif(&mut rng) * 2.0 - 1.0;
            let which = rng.below(3);
            out.push(match which {
                0 => [u * 3.5, v * 2.0, 6.0],         // the far wall
                1 => [u * 2.5, 1.2, 2.5 + (v + 1.0) * 2.0], // the floor
                _ => [u * 0.6 - 0.8, v * 0.5, 2.2],   // a near card
            });
        }
        out
    }

    /// Tracks of `points` over `n` frames with `noise` px of jitter, each
    /// visible over a random interval and only while inside the frame.
    pub fn tracks(pts: &[V3], n: usize, size: [u32; 2], noise: f64, seed: u64) -> Vec<Track> {
        tracks_from(pts, n, size, noise, seed).into_iter().map(|(t, _)| t).collect()
    }

    /// `tracks`, each with the index of the point it is of.
    pub fn tracks_from(pts: &[V3], n: usize, size: [u32; 2], noise: f64, seed: u64) -> Vec<(Track, usize)> {
        let k = Intrinsics::for_size(F, size);
        let mut rng = Rng(seed.wrapping_mul(31) | 1);
        let unif = |rng: &mut Rng| (rng.next() % 1_000_000) as f64 / 1_000_000.0 - 0.5;
        let mut out = Vec::new();
        for (pi, x) in pts.iter().enumerate() {
            let a = rng.below(n / 3);
            let b = n - rng.below(n / 3);
            let mut t = Track { start: a, pts: Vec::new() };
            for f in a..b {
                let p = k.project(camera(f, n).apply(*x));
                let inside = p.is_some_and(|p| p[0] > 2.0 && p[1] > 2.0 && p[0] < size[0] as f64 - 2.0 && p[1] < size[1] as f64 - 2.0);
                match (p, inside) {
                    (Some(p), true) => {
                        t.pts.push([p[0] + noise * unif(&mut rng), p[1] + noise * unif(&mut rng)]);
                    }
                    _ => {
                        if t.pts.len() >= 3 {
                            out.push((std::mem::take(&mut t), pi));
                        }
                        t = Track { start: f + 1, pts: Vec::new() };
                    }
                }
            }
            if t.pts.len() >= 3 {
                out.push((t, pi));
            }
        }
        out
    }

    /// The three planes as surfaces, for a rendered shot.
    pub struct Scene {
        tex: [crate::planar::Plane; 3],
    }

    /// Texture pixels per scene unit.
    const PX_PER_UNIT: f64 = 180.0;

    impl Scene {
        pub fn new() -> Self {
            use crate::planar::synth::texture;
            Self { tex: [texture(1024, 1024, 21), texture(1024, 1024, 22), texture(512, 512, 23)] }
        }

        fn shade(&self, which: usize, uv: [f64; 2]) -> f32 {
            let t = &self.tex[which];
            let x = (uv[0] * PX_PER_UNIT).rem_euclid(t.w as f64 - 1.0) + 0.5;
            let y = (uv[1] * PX_PER_UNIT).rem_euclid(t.h as f64 - 1.0) + 0.5;
            let v = t.sample(x, y).unwrap_or(0.5);
            // A different level per plane, so the planes read apart.
            [0.9, 0.7, 1.0][which] * v + [0.0, 0.1, 0.0][which]
        }

        /// Where the ray through `(x, y)` (pixels) meets the scene under
        /// `pose`: the nearest of the three planes, as a texture lookup
        /// (which plane, its 2D coordinate in units).
        fn hit(&self, pose: &Pose, k: &Intrinsics, x: f64, y: f64) -> Option<(usize, [f64; 2])> {
            // The ray in the world: from the centre, along Rᵀ (u, v, 1).
            let c = pose.centre();
            let d = m_v(&m_t(&pose.r), [(x - k.c[0]) / k.f, (y - k.c[1]) / k.f, 1.0]);
            let mut best: Option<(f64, usize, [f64; 2])> = None;
            // The near card: z = 2.2, |x + 0.8| < 0.6, |y| < 0.5.
            if d[2].abs() > 1e-9 {
                let s = (2.2 - c[2]) / d[2];
                if s > 1e-6 {
                    let p = add(c, scl(d, s));
                    if (p[0] + 0.8).abs() < 0.6 && p[1].abs() < 0.5 {
                        best = Some((s, 2, [p[0], p[1]]));
                    }
                }
            }
            // The floor: y = 1.2, in front of the wall.
            if d[1].abs() > 1e-9 {
                let s = (1.2 - c[1]) / d[1];
                if s > 1e-6 {
                    let p = add(c, scl(d, s));
                    if p[2] < 6.0 && best.is_none_or(|b| s < b.0) {
                        best = Some((s, 1, [p[0], p[2]]));
                    }
                }
            }
            // The wall: z = 6.
            if d[2].abs() > 1e-9 {
                let s = (6.0 - c[2]) / d[2];
                if s > 1e-6 && best.is_none_or(|b| s < b.0) {
                    let p = add(c, scl(d, s));
                    best = Some((s, 0, [p[0], p[1]]));
                }
            }
            best.map(|(_, which, uv)| (which, uv))
        }

        /// Frame `t` of `n` at `w × h`, through the true camera.
        pub fn render(&self, t: usize, n: usize, w: usize, h: usize) -> crate::planar::Plane {
            let pose = camera(t, n);
            let k = Intrinsics::for_size(F * w as f64 / 1280.0, [w as u32, h as u32]);
            let mut px = vec![0.0f32; w * h];
            for y in 0..h {
                for x in 0..w {
                    px[y * w + x] = match self.hit(&pose, &k, x as f64 + 0.5, y as f64 + 0.5) {
                        Some((which, uv)) => self.shade(which, uv),
                        None => 0.05,
                    };
                }
            }
            crate::planar::Plane::new(w, h, px)
        }
    }

    impl Default for Scene {
        fn default() -> Self {
            Self::new()
        }
    }

    /// Write a synthetic shot to `dir` as 16-bit greyscale PNGs
    /// (`frame.NNNN.png`): the three planes under the true camera. Returns
    /// the camera per frame (world → camera, CV, world = frame 0) and the
    /// focal length in pixels for that width.
    pub fn write_shot(dir: &std::path::Path, w: usize, h: usize, frames: usize) -> Result<(Vec<Pose>, f64)> {
        std::fs::create_dir_all(dir)?;
        let scene = Scene::new();
        let mut truth = Vec::with_capacity(frames);
        for t in 0..frames {
            truth.push(camera(t, frames));
            let f = scene.render(t, frames, w, h);
            let mut data = Vec::with_capacity(w * h * 2);
            for &px in &f.px {
                data.extend_from_slice(&((px.clamp(0.0, 1.0) * 65535.0).round() as u16).to_be_bytes());
            }
            let file = std::io::BufWriter::new(std::fs::File::create(dir.join(format!("frame.{t:04}.png")))?);
            let mut enc = png::Encoder::new(file, w as u32, h as u32);
            enc.set_color(png::ColorType::Grayscale);
            enc.set_depth(png::BitDepth::Sixteen);
            enc.write_header()?.write_image_data(&data)?;
        }
        Ok((truth, F * w as f64 / 1280.0))
    }
}

/// How far `solve` is from `truth`, both relative to their first camera:
/// the solve's scale fitted to the truth's, then the worst camera-centre
/// error (truth units) and the worst rotation error (degrees).
pub fn compare(solve: &[Pose], truth: &[Pose]) -> (f64, f64) {
    let n = solve.len().min(truth.len());
    let (mut num, mut den) = (0.0, 0.0);
    for i in 0..n {
        let (a, b) = (solve[i].centre(), truth[i].centre());
        num += dot(a, b);
        den += dot(a, a);
    }
    let s = if den > 1e-12 { num / den } else { 1.0 };
    let mut worst_c = 0.0f64;
    let mut worst_r = 0.0f64;
    for i in 0..n {
        let c = scl(solve[i].centre(), s);
        worst_c = worst_c.max(norm(sub(c, truth[i].centre())));
        let dr = m_mul(&solve[i].r, &m_t(&truth[i].r));
        worst_r = worst_r.max(norm(so3_log(&dr)).to_degrees());
    }
    (worst_c, worst_r)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIZE: [u32; 2] = [1280, 720];

    #[test]
    fn rotations_round_trip() {
        for w in [[0.1, -0.2, 0.3], [1e-12, 0.0, 0.0], [2.0, 1.0, -1.5], [0.0, 3.1, 0.0]] {
            let r = so3_exp(w);
            let back = so3_log(&r);
            let again = so3_exp(back);
            for i in 0..3 {
                for j in 0..3 {
                    assert!((r[i][j] - again[i][j]).abs() < 1e-9, "{w:?}");
                }
            }
        }
    }

    #[test]
    fn the_eigen_solver_and_svd_agree_with_the_matrix() {
        let a = [[2.0, -1.0, 0.5], [0.3, 1.5, -0.7], [1.1, 0.2, 0.9]];
        let (u, s, v) = svd3(&a);
        let d = [[s[0], 0.0, 0.0], [0.0, s[1], 0.0], [0.0, 0.0, s[2]]];
        let back = m_mul(&u, &m_mul(&d, &m_t(&v)));
        for i in 0..3 {
            for j in 0..3 {
                assert!((a[i][j] - back[i][j]).abs() < 1e-9, "{back:?}");
            }
        }
        assert!(s[0] >= s[1] && s[1] >= s[2]);
    }

    #[test]
    fn two_exact_views_give_the_pose_and_the_points() {
        let n = 24;
        let pts = synth::points(3);
        let k = Intrinsics::for_size(synth::F, SIZE);
        let b = synth::camera(n - 1, n);
        let pairs: Vec<([f64; 2], [f64; 2])> = pts
            .iter()
            .filter_map(|x| Some((k.normalise(k.project(*x)?), k.normalise(k.project(b.apply(*x))?))))
            .collect();
        let (e, inl) = essential_ransac(&pairs, (1.0 / k.f).powi(2), 200).unwrap();
        assert!(inl.iter().filter(|i| **i).count() > pairs.len() * 9 / 10);
        let (pose, tri) = pose_from_essential(&e, &pairs);
        // Up to scale: the translation direction and the rotation.
        let dir_err = 1.0 - dot(unit(pose.t), unit(b.t)).abs();
        let rot_err = norm(so3_log(&m_mul(&pose.r, &m_t(&b.r)))).to_degrees();
        assert!(dir_err < 1e-4 && rot_err < 0.05, "dir {dir_err} rot {rot_err}");
        assert!(tri.iter().filter(|x| x.is_some()).count() > pairs.len() * 9 / 10);
    }

    #[test]
    fn a_dolly_past_three_planes_is_recovered() {
        let n = 48;
        let pts = synth::points(5);
        let with = synth::tracks_from(&pts, n, SIZE, 0.4, 9);
        let tracks: Vec<Track> = with.iter().map(|(t, _)| t.clone()).collect();
        let opts = SolveOpts { size: SIZE, focal_px: synth::F, refine_focal: false, max_keyframes: 48 };
        let s = solve(&tracks, &opts, &mut |_, _| {}, &AtomicBool::new(false)).unwrap();
        let truth: Vec<Pose> = (0..n).map(|t| synth::camera(t, n)).collect();
        // Is the solve a better explanation of its own observations than
        // the truth? If not, it is stuck, not uncertain.
        let k = Intrinsics::for_size(synth::F, SIZE);
        let mut obs = Vec::new();
        for (pi, &ti) in s.point_track.iter().enumerate() {
            let t = &tracks[ti];
            for f in t.start..t.end() {
                obs.push(Obs { cam: f, pt: pi, px: t.pts[f - t.start] });
            }
        }
        let true_pts: Vec<V3> = s.point_track.iter().map(|&ti| pts[with[ti].1]).collect();
        let (c_solve, c_truth) = (cost_of(&k, &s.poses, &s.points, &obs), cost_of(&k, &truth, &true_pts, &obs));
        eprintln!("cost: solve {c_solve:.3}, truth {c_truth:.3}");
        assert!(c_solve < c_truth, "the solve explains its data worse than the truth: stuck ({c_solve:.2} vs {c_truth:.2})");
        let (wc, wr) = compare(&s.poses, &truth);
        let travel = norm(sub(truth[n - 1].centre(), truth[0].centre()));
        eprintln!("centres within {wc:.4} of a {travel:.3} travel, rotation within {wr:.3}°, {} points, {:.2} px mean error, {} unplaced",
            s.points.len(), s.mean_error(), s.unplaced.len());
        assert!(s.unplaced.is_empty(), "{:?}", s.unplaced);
        assert!(wc < 0.01 * travel, "centres {wc} of {travel}");
        assert!(wr < 0.1, "rotation {wr}°");
        assert!(s.mean_error() < 0.3, "{}", s.mean_error());
        assert!(s.points.len() > 400, "{} points", s.points.len());
    }

    #[test]
    fn the_focal_is_refined_from_a_wrong_start() {
        let n = 48;
        let pts = synth::points(7);
        let tracks = synth::tracks(&pts, n, SIZE, 0.3, 11);
        let opts = SolveOpts { size: SIZE, focal_px: synth::F * 1.15, refine_focal: true, max_keyframes: 48 };
        let s = solve(&tracks, &opts, &mut |_, _| {}, &AtomicBool::new(false)).unwrap();
        let err = (s.focal_px - synth::F).abs() / synth::F;
        eprintln!("focal {:.1} for {:.1}: {:.2}% off", s.focal_px, synth::F, err * 100.0);
        assert!(err < 0.02, "focal {} for {}", s.focal_px, synth::F);
        let truth: Vec<Pose> = (0..n).map(|t| synth::camera(t, n)).collect();
        let (_, wr) = compare(&s.poses, &truth);
        assert!(wr < 0.5, "rotation {wr}°");
    }

    /// Starting from wide to long, the guess lands on the true lens.
    #[test]
    fn the_lens_is_guessed_from_nothing() {
        let n = 48;
        let pts = synth::points(13);
        let tracks = synth::tracks(&pts, n, SIZE, 0.3, 17);
        let opts = SolveOpts { size: SIZE, focal_px: 0.0, refine_focal: true, max_keyframes: 48 };
        let starts: Vec<f64> = [14.0, 24.0, 35.0, 50.0, 85.0, 135.0].iter().map(|mm| mm / 36.0 * SIZE[0] as f64).collect();
        let s = solve_guess(&tracks, &opts, &starts, &mut |_, _| {}, &AtomicBool::new(false)).unwrap();
        let err = (s.focal_px - synth::F).abs() / synth::F;
        eprintln!("guessed {:.1} px for {:.1}: {:.2}% off", s.focal_px, synth::F, err * 100.0);
        assert!(err < 0.03, "guessed {} for {}", s.focal_px, synth::F);
    }

    #[test]
    fn a_pan_is_refused_with_words() {
        let k = Intrinsics::for_size(synth::F, SIZE);
        let pts = synth::points(1);
        let n = 20;
        let tracks: Vec<Track> = pts
            .iter()
            .map(|x| Track {
                start: 0,
                pts: (0..n)
                    .map(|t| {
                        let r = so3_exp([0.0, 0.004 * t as f64, 0.0]);
                        k.project(m_v(&r, *x)).unwrap_or([1.0, 1.0])
                    })
                    .collect(),
            })
            .collect();
        let opts = SolveOpts { size: SIZE, focal_px: synth::F, refine_focal: false, max_keyframes: 10 };
        let r = solve(&tracks, &opts, &mut |_, _| {}, &AtomicBool::new(false));
        // Either refused outright or solved with next to no parallax; a
        // refusal is the honest answer and the message must say MOVE.
        if let Err(e) = r {
            assert!(format!("{e}").contains("MOVE"), "{e}");
        }
    }

    #[test]
    fn the_files_round_trip() {
        let dir = std::env::temp_dir().join(format!("wobble-camtrack-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let tracks = vec![Track { start: 2, pts: vec![[1.5, 2.25], [3.0, 4.0], [5.0, 6.0]] }];
        write_tracks(&dir, &tracks, [640, 360]).unwrap();
        let (back, size) = read_tracks(&dir).unwrap();
        assert_eq!(size, [640, 360]);
        assert_eq!(back, tracks);
        let s = Solve {
            size: [640, 360],
            focal_px: 812.5,
            poses: vec![Pose::IDENTITY, Pose { r: so3_exp([0.1, 0.2, 0.3]), t: [1.0, -2.0, 0.5] }],
            unplaced: vec![1],
            points: vec![[0.1, 0.2, 3.0], [-1.0, 2.0, 4.5]],
            point_track: vec![4, 9],
            errors: vec![0.3, 0.7],
        };
        write_solve(&dir, &s).unwrap();
        let back = read_solve(&dir).unwrap();
        assert_eq!(back.size, s.size);
        assert_eq!(back.unplaced, s.unplaced);
        assert_eq!(back.points.len(), 2);
        assert_eq!(back.point_track, s.point_track);
        for (a, b) in back.poses.iter().zip(&s.poses) {
            for i in 0..3 {
                assert!((a.t[i] - b.t[i]).abs() < 1e-12);
                for j in 0..3 {
                    assert!((a.r[i][j] - b.r[i][j]).abs() < 1e-12);
                }
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_cards_camera_looks_at_the_points_it_solved() {
        // A point the CV camera sees at the centre sits straight down −Z
        // in the cards' camera space.
        let s = Solve {
            size: SIZE,
            focal_px: synth::F,
            poses: vec![Pose::IDENTITY, synth::camera(5, 10)],
            unplaced: vec![],
            points: vec![[0.0, 0.0, 3.0]],
            point_track: vec![0],
            errors: vec![0.0, 0.0],
        };
        let m = s.camera(1, 1.0);
        let p = s.point(0, 1.0);
        let inv = crate::scene3d::Xform::from_matrix(&m).inverse_scaled([1.0; 3]);
        let local = [
            inv[0][0] * p[0] + inv[0][1] * p[1] + inv[0][2] * p[2] + inv[0][3],
            inv[1][0] * p[0] + inv[1][1] * p[1] + inv[1][2] * p[2] + inv[1][3],
            inv[2][0] * p[0] + inv[2][1] * p[1] + inv[2][2] * p[2] + inv[2][3],
        ];
        let pc = s.poses[1].apply(s.points[0]);
        assert!((local[0] as f64 - pc[0]).abs() < 1e-4 && (local[1] as f64 + pc[1]).abs() < 1e-4 && (local[2] as f64 + pc[2]).abs() < 1e-4, "{local:?} vs {pc:?}");
    }
}
