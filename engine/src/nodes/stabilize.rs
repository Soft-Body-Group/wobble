//! Stabilize: the camera's shake measured off the picture and taken out —
//! or put back.
//!
//! The analysis is explicit and runs on a click (the Flow precedent): the
//! app renders the node's input (and its optional matte) over the input's
//! range, `crate::stabilize::run` takes SEA-RAFT's flow between each pair of
//! neighbouring frames at a reduced size, fits ONE global motion to a grid
//! of it (robustly — a moving subject is outvoted, and a matte wired into
//! port 1 takes its pixels out of the vote entirely) and the motions
//! accumulate into a CAMERA PATH: per comp frame, where the first frame's
//! pixels are on that frame.
//!
//! Everything after that is CPU maths on a few numbers a frame, evaluated
//! at the playhead: lock-off (hold a reference frame) or a Gaussian smooth
//! of the path, per axis, then reverse and zoom. The result is a corner pin
//! (`pin`, transient, set by the app per evaluated frame) the node applies
//! exactly as a CornerPin does — fused into the main pass, one resample, no
//! stage — so a Stabilize is cyan, and editing the smoothing never re-runs
//! the analysis.
//!
//! Unanalysed it is the identity pin: the picture passes through.

use crate::planar::{h_apply, h_from_quad, h_inv, h_mul, h_translation, H};

/// What the path is held to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum StabMode {
    /// Every frame is warped onto the reference frame: a locked-off camera.
    #[default]
    LockOff,
    /// The path is smoothed (a Gaussian, `smooth` frames wide): the camera
    /// keeps its intended move and loses the shake.
    Smooth,
}

impl StabMode {
    pub const ALL: [StabMode; 2] = [StabMode::LockOff, StabMode::Smooth];
    pub fn label(self) -> &'static str {
        match self {
            StabMode::LockOff => "lock-off",
            StabMode::Smooth => "smooth",
        }
    }
}

/// The motion fitted between neighbouring frames.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum StabModel {
    Translate,
    /// Move, turn and scale: what a hand-held camera mostly does.
    #[default]
    Similarity,
    Affine,
    Perspective,
}

impl StabModel {
    pub const ALL: [StabModel; 4] =
        [StabModel::Translate, StabModel::Similarity, StabModel::Affine, StabModel::Perspective];
    pub fn label(self) -> &'static str {
        match self {
            StabModel::Translate => "translate",
            StabModel::Similarity => "similarity",
            StabModel::Affine => "affine",
            StabModel::Perspective => "perspective",
        }
    }
}

/// The analysis' result: the camera path, keyed by comp frame.
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize), serde(default))]
pub struct StabPath {
    /// The comp frame `motion[0]` is.
    pub first: i64,
    /// The analysed picture's size, in pixels: the path's units.
    pub size: [u32; 2],
    /// Per frame, the homography carrying the FIRST frame's pixels onto
    /// this frame's (`motion[0]` is the identity). Row-major, pixels, y
    /// down, centres at +0.5.
    pub motion: Vec<H>,
    /// The key of the run that made it (what it was analysed from and how):
    /// the app compares it with the current one to say STALE.
    pub recipe: u64,
    /// The path as similarity parameters about the frame centre (x, y,
    /// radians, log scale), rotation unwrapped — derived once, on first use.
    #[cfg_attr(feature = "serde", serde(skip))]
    params: std::sync::OnceLock<Vec<[f64; 4]>>,
}

impl PartialEq for StabPath {
    fn eq(&self, o: &Self) -> bool {
        self.first == o.first && self.size == o.size && self.motion == o.motion && self.recipe == o.recipe
    }
}

impl StabPath {
    pub fn new(first: i64, size: [u32; 2], motion: Vec<H>, recipe: u64) -> Self {
        Self { first, size, motion, recipe, params: Default::default() }
    }

    pub fn is_empty(&self) -> bool {
        self.motion.is_empty()
    }

    /// The comp frames it covers, inclusive.
    pub fn span(&self) -> Option<(i64, i64)> {
        (!self.motion.is_empty()).then(|| (self.first, self.first + self.motion.len() as i64 - 1))
    }

    fn centre(&self) -> [f64; 2] {
        [self.size[0] as f64 * 0.5, self.size[1] as f64 * 0.5]
    }

    fn params(&self) -> &[[f64; 4]] {
        self.params.get_or_init(|| {
            let c = self.centre();
            let (w, h) = (self.size[0] as f64, self.size[1] as f64);
            let mut out: Vec<[f64; 4]> = self.motion.iter().map(|m| sim_params(m, c, w, h)).collect();
            for i in 1..out.len() {
                let prev = out[i - 1][2];
                let mut d = out[i][2] - prev;
                d -= (d / std::f64::consts::TAU).round() * std::f64::consts::TAU;
                out[i][2] = prev + d;
            }
            out
        })
    }

    fn index(&self, t: i64) -> usize {
        (t - self.first).clamp(0, self.motion.len() as i64 - 1) as usize
    }
}

/// The path behind an `Arc` (the `graph::TrackRef` pattern): the graph is
/// cloned, compared and keyed per frame, and a path is a matrix a frame.
#[derive(Clone, Debug, Default)]
pub struct StabPathRef(std::sync::Arc<StabPath>);

impl StabPathRef {
    pub fn new(p: StabPath) -> Self {
        Self(std::sync::Arc::new(p))
    }
}

impl std::ops::Deref for StabPathRef {
    type Target = StabPath;
    fn deref(&self) -> &StabPath {
        &self.0
    }
}

impl PartialEq for StabPathRef {
    fn eq(&self, o: &Self) -> bool {
        std::sync::Arc::ptr_eq(&self.0, &o.0) || *self.0 == *o.0
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for StabPathRef {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        (*self.0).serialize(s)
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for StabPathRef {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        StabPath::deserialize(d).map(StabPathRef::new)
    }
}

/// The node's authored state, plus the transient pin.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize), serde(default))]
pub struct StabilizeParams {
    pub mode: StabMode,
    /// What the analysis fits between frames. Part of the recipe.
    pub model: StabModel,
    /// Analyse at 1 / this of the size (1, 2 or 4). Part of the recipe.
    pub divisor: u8,
    /// Smooth: the Gaussian's sigma, in frames.
    pub smooth: f32,
    /// Which parts of the motion are taken out: x, y, rotate, scale. An
    /// axis left off keeps the camera's own move. All four on in lock-off
    /// holds the reference frame exactly, perspective included.
    pub axes: [bool; 4],
    /// Lock-off's reference, a comp frame. None: the first analysed frame.
    pub reference: Option<i64>,
    /// Put the shake BACK: the inverse of what this node would take out,
    /// with the same settings — for matching CG to a plate, or undoing a
    /// stabilise after the work is done.
    pub reverse: bool,
    /// Scale about the centre after the correction, to push the edges the
    /// warp drags in off the frame. 1 = none.
    pub zoom: f32,
    pub path: StabPathRef,
    /// TRANSIENT: the corner pin for the evaluated frame, normalised to the
    /// node's format (TL, TR, BR, BL). Set by the app on the engine-graph
    /// copy, the identity everywhere else.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub pin: [[f32; 2]; 4],
}

impl Default for StabilizeParams {
    fn default() -> Self {
        Self {
            mode: StabMode::LockOff,
            model: StabModel::Similarity,
            divisor: 2,
            smooth: 12.0,
            axes: [true; 4],
            reference: None,
            reverse: false,
            zoom: 1.0,
            path: StabPathRef::default(),
            pin: crate::graph::UNIT_QUAD,
        }
    }
}

impl StabilizeParams {
    pub const NAME: &'static str = "Stabilize";
    /// The picture, then the matte that keeps its pixels out of the solve.
    pub const INPUTS: usize = 2;
    pub const MATTE_INPUT: usize = 1;
    /// Three rows of the inverse homography, like a CornerPin.
    pub const VEC4_SLOTS: usize = 3;

    /// The warp is always there (the identity until analysed); everything
    /// about it is a uniform.
    pub fn topo_tag(&self) -> u64 {
        0x57AB
    }

    /// What shapes the analysis, for the app's result key.
    pub fn recipe(&self) -> (&'static str, u8, StabModel) {
        ("searaft-m", self.divisor.clamp(1, 4), self.model)
    }

    pub fn models(&self) -> &'static [&'static str] {
        &["searaft-m"]
    }

    pub fn analysed(&self) -> bool {
        !self.path.is_empty()
    }

    /// The correction at comp frame `t`, in the path's pixels: the picture
    /// at `p` is shown at `W·p`. None until analysed.
    pub fn correction_at(&self, t: i64) -> Option<H> {
        let path = &*self.path;
        if path.is_empty() {
            return None;
        }
        let c = path.centre();
        let i = path.index(t);
        let r = path.index(self.reference.unwrap_or(path.first));
        let w = if self.mode == StabMode::LockOff && self.axes.iter().all(|a| *a) {
            // The whole fitted motion, perspective and all.
            h_mul(&path.motion[r], &h_inv(&path.motion[i])?)
        } else {
            let ps = path.params();
            let raw = ps[i];
            let mut target = raw;
            for k in 0..4 {
                if !self.axes[k] {
                    continue;
                }
                target[k] = match self.mode {
                    StabMode::LockOff => ps[r][k],
                    StabMode::Smooth => gauss(ps, i, k, self.smooth as f64),
                };
            }
            h_mul(&sim_h(&target, c), &h_inv(&sim_h(&raw, c))?)
        };
        let z = self.zoom.clamp(0.1, 10.0) as f64;
        let zh = h_mul(&h_translation(c[0], c[1]), &h_mul(&[z, 0.0, 0.0, 0.0, z, 0.0, 0.0, 0.0, 1.0], &h_translation(-c[0], -c[1])));
        let m = h_mul(&zh, &w);
        if self.reverse {
            h_inv(&m)
        } else {
            Some(m)
        }
    }

    /// The corner pin for comp frame `t`: where the frame's corners go,
    /// normalised. The identity until analysed.
    pub fn pin_at(&self, t: i64) -> [[f32; 2]; 4] {
        let Some(m) = self.correction_at(t) else { return crate::graph::UNIT_QUAD };
        let (w, h) = (self.path.size[0] as f64, self.path.size[1] as f64);
        crate::graph::UNIT_QUAD.map(|q| {
            let (x, y) = h_apply(&m, q[0] as f64 * w, q[1] as f64 * h);
            [(x / w) as f32, (y / h) as f32]
        })
    }

    /// The pin as the CornerPin the shader runs.
    pub fn as_corner_pin(&self) -> crate::graph::CornerPinParams {
        crate::graph::CornerPinParams::pinned(self.pin)
    }
}

/// A Gaussian of `sigma` frames over parameter `k`, centred on `i`, the
/// kernel cut at 3 sigma and renormalised where it runs off either end.
fn gauss(ps: &[[f64; 4]], i: usize, k: usize, sigma: f64) -> f64 {
    if sigma < 1e-3 {
        return ps[i][k];
    }
    let reach = (3.0 * sigma).ceil() as i64;
    let (mut acc, mut wsum) = (0.0, 0.0);
    for d in -reach..=reach {
        let j = i as i64 + d;
        if j < 0 || j >= ps.len() as i64 {
            continue;
        }
        let wt = (-0.5 * (d as f64 / sigma).powi(2)).exp();
        acc += wt * ps[j as usize][k];
        wsum += wt;
    }
    acc / wsum
}

/// The similarity that best matches `m` on the frame's corners, about the
/// centre `c`: (x, y) the corners' centroid's move, then radians and log
/// scale (complex least squares, as `planarui::drive_of`).
pub fn sim_params(m: &H, c: [f64; 2], w: f64, h: f64) -> [f64; 4] {
    let src = [[0.0, 0.0], [w, 0.0], [w, h], [0.0, h]];
    let dst = src.map(|p| {
        let (x, y) = h_apply(m, p[0], p[1]);
        [x, y]
    });
    let cd = [dst.iter().map(|p| p[0]).sum::<f64>() / 4.0, dst.iter().map(|p| p[1]).sum::<f64>() / 4.0];
    let (mut a, mut b, mut pp) = (0.0, 0.0, 0.0);
    for k in 0..4 {
        let p = [src[k][0] - c[0], src[k][1] - c[1]];
        let q = [dst[k][0] - cd[0], dst[k][1] - cd[1]];
        a += p[0] * q[0] + p[1] * q[1];
        b += p[0] * q[1] - p[1] * q[0];
        pp += p[0] * p[0] + p[1] * p[1];
    }
    let s = ((a * a + b * b).sqrt() / pp.max(1e-12)).max(1e-9);
    [cd[0] - c[0], cd[1] - c[1], b.atan2(a), s.ln()]
}

/// The similarity `sim_params` describes, as a homography.
pub fn sim_h(p: &[f64; 4], c: [f64; 2]) -> H {
    let s = p[3].exp();
    let (sn, cs) = p[2].sin_cos();
    let rs = [cs * s, -sn * s, 0.0, sn * s, cs * s, 0.0, 0.0, 0.0, 1.0];
    h_mul(&h_translation(c[0] + p[0], c[1] + p[1]), &h_mul(&rs, &h_translation(-c[0], -c[1])))
}

/// The homography a pin makes, frame pixels to frame pixels (for tests and
/// gizmos): the unit square onto `pin`.
pub fn pin_h(pin: &[[f32; 2]; 4], w: f64, h: f64) -> Option<H> {
    let px = |q: [f32; 2]| [q[0] as f64 * w, q[1] as f64 * h];
    h_from_quad(&crate::graph::UNIT_QUAD.map(px), &pin.map(px))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planar::synth::view;

    const W: usize = 640;
    const HT: usize = 360;

    /// A shaky path: a slow pan with a jitter on every axis.
    fn shaky(n: usize) -> StabPath {
        let base = view(W, HT, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
        let inv = h_inv(&base).unwrap();
        let motion = (0..n)
            .map(|t| {
                let tf = t as f64;
                let v = view(
                    W,
                    HT,
                    1.5 * (tf * 1.3).sin(),
                    1.0 + 0.01 * (tf * 0.9).cos(),
                    2.0 * tf + 5.0 * (tf * 1.7).sin(),
                    3.0 * (tf * 2.3).cos() - 3.0,
                    0.0,
                    0.0,
                );
                h_mul(&v, &inv)
            })
            .collect();
        StabPath::new(1001, [W as u32, HT as u32], motion, 0)
    }

    fn params(path: StabPath) -> StabilizeParams {
        StabilizeParams { path: StabPathRef::new(path), ..Default::default() }
    }

    fn dist(m: &H, n: &H) -> f64 {
        let mut worst = 0.0f64;
        for p in [[0.0, 0.0], [W as f64, 0.0], [W as f64, HT as f64], [0.0, HT as f64], [320.0, 180.0]] {
            let (a, b) = h_apply(m, p[0], p[1]);
            let (c, d) = h_apply(n, p[0], p[1]);
            worst = worst.max((a - c).hypot(b - d));
        }
        worst
    }

    #[test]
    fn lock_off_holds_the_reference_frame() {
        let mut p = params(shaky(24));
        p.reference = Some(1005);
        for t in 1001..1025 {
            let w = p.correction_at(t).unwrap();
            // A point of the reference frame, carried to frame t by the
            // camera and back by the correction, is where it was.
            let cam = h_mul(&p.path.motion[(t - 1001) as usize], &h_inv(&p.path.motion[4]).unwrap());
            let id = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
            assert!(dist(&h_mul(&w, &cam), &id) < 1e-6, "frame {t}");
        }
        // The reference itself passes through.
        let pin = p.pin_at(1005);
        for k in 0..4 {
            assert!((pin[k][0] - crate::graph::UNIT_QUAD[k][0]).abs() < 1e-5);
            assert!((pin[k][1] - crate::graph::UNIT_QUAD[k][1]).abs() < 1e-5);
        }
    }

    #[test]
    fn reverse_undoes_the_stabilise() {
        let mut p = params(shaky(30));
        p.mode = StabMode::Smooth;
        p.zoom = 1.1;
        let mut r = p.clone();
        r.reverse = true;
        let id = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        for t in 1001..1031 {
            let both = h_mul(&r.correction_at(t).unwrap(), &p.correction_at(t).unwrap());
            assert!(dist(&both, &id) < 1e-6, "frame {t}");
        }
    }

    #[test]
    fn smoothing_takes_the_jitter_and_keeps_the_pan() {
        let mut p = params(shaky(60));
        p.mode = StabMode::Smooth;
        p.smooth = 6.0;
        // The stabilised camera: the correction after the camera.
        let path = |p: &StabilizeParams, t: i64| -> [f64; 4] {
            let m = h_mul(&p.correction_at(t).unwrap(), &p.path.motion[(t - 1001) as usize]);
            sim_params(&m, [W as f64 * 0.5, HT as f64 * 0.5], W as f64, HT as f64)
        };
        // Second differences (acceleration) of x: the shake is in them.
        let accel = |f: &dyn Fn(i64) -> f64| -> f64 {
            (1016..1046).map(|t| (f(t + 1) - 2.0 * f(t) + f(t - 1)).abs()).sum::<f64>() / 30.0
        };
        let raw = |t: i64| p.path.params()[(t - 1001) as usize][0];
        let smooth = |t: i64| path(&p, t)[0];
        let (a_raw, a_smooth) = (accel(&raw), accel(&smooth));
        assert!(a_smooth < a_raw * 0.1, "jitter {a_raw:.3} -> {a_smooth:.3}");
        // The pan (2 px a frame) survives.
        let pan = (smooth(1045) - smooth(1015)) / 30.0;
        assert!((pan - 2.0).abs() < 0.3, "pan {pan:.3}");
        // An axis left off keeps its shake.
        p.axes[0] = false;
        let kept = |t: i64| path(&p, t)[0];
        assert!((accel(&kept) - a_raw).abs() < 1e-6);
    }

    #[test]
    fn unanalysed_is_the_identity_and_cheap_to_compare() {
        let p = StabilizeParams::default();
        assert_eq!(p.pin_at(10), crate::graph::UNIT_QUAD);
        assert!(p.correction_at(0).is_none());
        let a = params(shaky(5));
        let b = a.clone();
        assert!(std::sync::Arc::ptr_eq(&a.path.0, &b.path.0));
        assert_eq!(a, b);
        // A setting is not a recipe; the model is.
        let mut c = a.clone();
        c.smooth = 3.0;
        c.reverse = true;
        assert_eq!(a.recipe(), c.recipe());
        c.model = StabModel::Perspective;
        assert_ne!(a.recipe(), c.recipe());
        assert_eq!(a.topo_tag(), c.topo_tag());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn the_path_saves_and_the_pin_does_not() {
        let mut p = params(shaky(3));
        p.pin = [[9.0; 2]; 4];
        let j = serde_json::to_string(&p).unwrap();
        assert!(!j.contains("\"pin\""), "{j}");
        let back: StabilizeParams = serde_json::from_str(&j).unwrap();
        assert_eq!(back.path, p.path);
        assert_eq!(back.pin, crate::graph::UNIT_QUAD);
    }
}
