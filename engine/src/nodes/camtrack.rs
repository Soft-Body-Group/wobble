//! CameraTrack: a 3D camera solved off the picture, for the 3D cards.
//!
//! The work is explicit and runs on a click (the Stabilize precedent): the
//! app renders the node's input (and its optional matte, which keeps a
//! moving subject out) over the input's range, `crate::camtrack::track`
//! follows a grid of features along SEA-RAFT's flow, and
//! `crate::camtrack::solve` fits a camera path and a point cloud to them.
//! The result lands ON THE NODE (`CamSolve`, behind an Arc — the
//! `StabPath` pattern), keyed by comp frame, saved with the setup.
//!
//! The node itself changes no pixels: it is a passthrough with a matte
//! port, cyan, and its panel draws the solved points over the viewer. A
//! Camera node with `Source::Tracker { node }` reads its pose and lens per
//! frame (`camera_at`), so a Render3D follows a re-solve without a bake,
//! and "bake to keys" works as for any source.
//!
//! Two recipes, so a re-solve never re-tracks: the TRACKS are keyed on the
//! inputs, the detail and the range; the SOLVE on those plus the lens
//! settings. Changing the focal length re-runs the solve alone (seconds).

use crate::camtrack::Solve;

/// The solve as it sits on the node: comp-frame keyed.
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize), serde(default))]
pub struct CamSolve {
    /// The comp frame `solve.poses[0]` is.
    pub first: i64,
    pub solve: Solve,
    /// The key of the run that made it (inputs, range, detail, lens
    /// settings): the app compares it with the current one to say STALE.
    pub recipe: u64,
}

impl CamSolve {
    pub fn is_empty(&self) -> bool {
        self.solve.is_empty()
    }

    pub fn span(&self) -> Option<(i64, i64)> {
        (!self.is_empty()).then(|| (self.first, self.first + self.solve.poses.len() as i64 - 1))
    }
}

/// The solve behind an `Arc`: the graph is cloned, compared and keyed per
/// frame, and a solve is a pose a frame plus a cloud.
#[derive(Clone, Debug, Default)]
pub struct CamSolveRef(std::sync::Arc<CamSolve>);

impl CamSolveRef {
    pub fn new(s: CamSolve) -> Self {
        Self(std::sync::Arc::new(s))
    }

    pub fn ptr_eq(&self, o: &Self) -> bool {
        std::sync::Arc::ptr_eq(&self.0, &o.0)
    }
}

impl std::ops::Deref for CamSolveRef {
    type Target = CamSolve;
    fn deref(&self) -> &CamSolve {
        &self.0
    }
}

impl PartialEq for CamSolveRef {
    fn eq(&self, o: &Self) -> bool {
        std::sync::Arc::ptr_eq(&self.0, &o.0) || *self.0 == *o.0
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for CamSolveRef {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        (*self.0).serialize(s)
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for CamSolveRef {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        CamSolve::deserialize(d).map(CamSolveRef::new)
    }
}

/// The node's authored state.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize), serde(default))]
pub struct CamTrackParams {
    /// Track at 1 / this of the size (1, 2 or 4). Part of the tracks recipe.
    /// Full size by default: at half, the flow carries a bias the solve
    /// cannot tell from motion (measured: 0.6° off against 0.1° at full).
    /// Half and quarter are for a quick look.
    pub divisor: u8,
    /// The lens: focal length and filmback width, mm. The focal is the
    /// solve's starting point (and its answer, unless refined); the
    /// filmback only turns pixels into millimetres for the Camera.
    pub focal: f32,
    pub aperture: f32,
    /// Let the solve adjust the focal length.
    pub refine_focal: bool,
    /// Guess the lens: solve from every start in [`GUESS_STARTS`] with the
    /// focal free and keep the best fit; `focal` and `refine_focal` are
    /// then ignored. Part of the solve recipe.
    pub guess: bool,
    /// Scene units per solve unit (the solve puts the median point depth
    /// from the first camera at 1). Read when a Camera asks; never a re-run.
    pub scale: f32,
    /// Draw the solved points over the viewer.
    pub show: bool,
    pub solve: CamSolveRef,
    /// Points picked over the viewer (indices into the solve's cloud),
    /// and the solve recipe they belong to: a new solve has new points.
    pub picked: Vec<usize>,
    pub picked_for: u64,
    /// The alignment laid over the solve, read at use (never a re-run):
    /// the world is `R · (p − scale · off)` — `align_off` in solve units
    /// (the origin), `align_rot` Euler degrees (the ground's up to +Y).
    pub align_rot: [f32; 3],
    pub align_off: [f32; 3],
    /// The distance two picked points are set apart, scene units.
    pub ruler: f32,
}

impl Default for CamTrackParams {
    fn default() -> Self {
        Self {
            divisor: 1,
            focal: 50.0,
            aperture: 36.0,
            refine_focal: true,
            guess: false,
            scale: 1.0,
            show: true,
            solve: CamSolveRef::default(),
            picked: Vec::new(),
            picked_for: 0,
            align_rot: [0.0; 3],
            align_off: [0.0; 3],
            ruler: 1.0,
        }
    }
}

/// Lens presets, as (name, focal mm, filmback width mm). Phones are in
/// 35 mm-equivalent terms (what the solve needs is the ratio). Video
/// stabilisation crops a phone's frame a little, so a preset is a start
/// and refine takes it the rest of the way.
pub const LENS_PRESETS: &[(&str, f32, f32)] = &[
    ("iPhone main (24 mm)", 24.0, 36.0),
    ("iPhone main, 11–13 (26 mm)", 26.0, 36.0),
    ("iPhone ultra wide (13 mm)", 13.0, 36.0),
    ("iPhone 2× (48 mm)", 48.0, 36.0),
    ("iPhone telephoto 3× (77 mm)", 77.0, 36.0),
    ("iPhone telephoto 5× (120 mm)", 120.0, 36.0),
    ("14 mm full frame", 14.0, 36.0),
    ("24 mm full frame", 24.0, 36.0),
    ("35 mm full frame", 35.0, 36.0),
    ("50 mm full frame", 50.0, 36.0),
    ("85 mm full frame", 85.0, 36.0),
    ("135 mm full frame", 135.0, 36.0),
];

/// Where a lens guess starts from, mm on a 36 mm back: wide to long, so
/// one of them is within a stop of any real lens and refine does the rest.
pub const GUESS_STARTS: &[f32] = &[14.0, 24.0, 35.0, 50.0, 85.0, 135.0];

impl CamTrackParams {
    pub const NAME: &'static str = "CameraTrack";
    /// The picture, then the matte that keeps its pixels out of the track.
    pub const INPUTS: usize = 2;
    pub const MATTE_INPUT: usize = 1;
    pub const VEC4_SLOTS: usize = 0;

    /// A passthrough: nothing about it is topology.
    pub fn topo_tag(&self) -> u64 {
        0xCA_3D
    }

    /// What shapes the tracks, for the app's result key.
    pub fn recipe_tracks(&self) -> (&'static str, u8) {
        ("searaft-m", self.divisor.clamp(1, 4))
    }

    /// What shapes the solve on top of the tracks.
    pub fn recipe_solve(&self) -> (i32, bool, bool) {
        if self.guess {
            return (0, true, true);
        }
        // The focal to a tenth of a millimetre: a drag through the slider
        // is not a hundred runs.
        ((self.focal * 10.0).round() as i32, self.refine_focal, false)
    }

    /// The focal lengths in pixels a guess starts from, for `width`.
    pub fn guess_starts_px(width: f64) -> Vec<f64> {
        GUESS_STARTS.iter().map(|mm| *mm as f64 / 36.0 * width).collect()
    }

    pub fn models(&self) -> &'static [&'static str] {
        &["searaft-m"]
    }

    pub fn solved(&self) -> bool {
        !self.solve.is_empty()
    }

    /// The focal length in pixels for a picture `width` wide.
    pub fn focal_px(&self, width: f64) -> f64 {
        self.focal.max(0.1) as f64 / self.aperture.max(0.1) as f64 * width
    }

    /// The solved camera at comp frame `t`: its transform (cards
    /// convention) and lens. None until solved; held past either end.
    pub fn camera_at(&self, t: f32) -> Option<(crate::scene3d::Xform, crate::scene3d::import::Lens)> {
        let s = &*self.solve;
        if s.is_empty() {
            return None;
        }
        let n = s.solve.poses.len();
        let f = t - s.first as f32;
        // Between frames: a blend of the two matrices' transforms (the
        // poses are a frame apart, so a linear blend is honest enough).
        let fl = f.floor().clamp(0.0, (n - 1) as f32);
        let i = fl as usize;
        let j = (i + 1).min(n - 1);
        let w = (f - fl).clamp(0.0, 1.0);
        let a = crate::scene3d::Xform::from_matrix(&self.camera_mat(i));
        let x = if w > 0.0 && j != i {
            let b = crate::scene3d::Xform::from_matrix(&self.camera_mat(j));
            let mut r = b.rotate;
            for k in 0..3 {
                // The short way round.
                let mut d = r[k] - a.rotate[k];
                d -= (d / 360.0).round() * 360.0;
                r[k] = a.rotate[k] + d;
            }
            crate::scene3d::Xform {
                translate: std::array::from_fn(|k| a.translate[k] + (b.translate[k] - a.translate[k]) * w),
                rotate: std::array::from_fn(|k| a.rotate[k] + (r[k] - a.rotate[k]) * w),
                scale: [1.0; 3],
            }
        } else {
            crate::scene3d::Xform { scale: [1.0; 3], ..a }
        };
        let (wpx, hpx) = (s.solve.size[0].max(1) as f32, s.solve.size[1].max(1) as f32);
        let focal = s.solve.focal_px as f32 / wpx * self.aperture;
        let lens = crate::scene3d::import::Lens { focal, aperture: [self.aperture, self.aperture * hpx / wpx], near: 0.01, far: 10000.0 };
        Some((x, lens))
    }
}

/// The alignment and the picks: everything that turns a solve into a
/// scene without re-running it.
impl CamTrackParams {
    /// The alignment as a matrix over the SCALED solve: `R · T(−scale·off)`.
    pub fn align_mat(&self) -> crate::scene3d::Mat4 {
        let r = crate::scene3d::rotation(self.align_rot);
        let s = self.scale.max(1e-6);
        let mut m = crate::scene3d::identity();
        for i in 0..3 {
            for j in 0..3 {
                m[i][j] = r[i][j];
            }
            m[i][3] = -(0..3).map(|k| r[i][k] * self.align_off[k] * s).sum::<f32>();
        }
        m
    }

    /// Solved camera `i` (an index into the poses) as a cards local → world
    /// matrix, aligned and scaled.
    pub fn camera_mat(&self, i: usize) -> crate::scene3d::Mat4 {
        crate::scene3d::mat_mul(&self.align_mat(), &self.solve.solve.camera(i, self.scale.max(1e-6) as f64))
    }

    /// Solved point `k` in the scene, aligned and scaled.
    pub fn point_world(&self, k: usize) -> [f32; 3] {
        let p = self.solve.solve.point(k, self.scale.max(1e-6) as f64);
        let a = self.align_mat();
        std::array::from_fn(|i| a[i][0] * p[0] + a[i][1] * p[1] + a[i][2] * p[2] + a[i][3])
    }

    /// The picks that belong to the solve on the node (a re-solve has new
    /// points, so older picks mean nothing).
    pub fn picks(&self) -> &[usize] {
        if self.picked_for == self.solve.recipe && !self.solve.is_empty() {
            &self.picked
        } else {
            &[]
        }
    }

    /// Toggle a pick.
    pub fn toggle_pick(&mut self, k: usize) {
        if self.picked_for != self.solve.recipe {
            self.picked.clear();
            self.picked_for = self.solve.recipe;
        }
        match self.picked.iter().position(|p| *p == k) {
            Some(i) => {
                self.picked.remove(i);
            }
            None => self.picked.push(k),
        }
    }

    /// The picked points in the UNSCALED, unaligned cards frame (the
    /// solve's own, flipped): what the alignment ops are computed from.
    fn pick_pts(&self) -> Vec<[f32; 3]> {
        let s = &self.solve.solve;
        self.picks().iter().filter(|&&k| k < s.points.len()).map(|&k| s.point(k, 1.0)).collect()
    }

    /// Put the origin on the picked points' centroid.
    pub fn set_origin(&mut self) -> bool {
        let pts = self.pick_pts();
        if pts.is_empty() {
            return false;
        }
        self.align_off = centroid(&pts);
        true
    }

    /// Fit a plane to three or more picked points and turn the scene so it
    /// is the ground (its normal +Y, the first camera above it), with the
    /// origin on their centroid.
    pub fn set_ground(&mut self) -> bool {
        let pts = self.pick_pts();
        if pts.len() < 3 {
            return false;
        }
        let c = centroid(&pts);
        let Some(mut n) = plane_normal(&pts) else { return false };
        // The first camera is at the solve's origin: it must look down on
        // the ground, so the normal points its way.
        if n[0] * (0.0 - c[0]) + n[1] * (0.0 - c[1]) + n[2] * (0.0 - c[2]) < 0.0 {
            n = [-n[0], -n[1], -n[2]];
        }
        self.align_rot = crate::scene3d::euler_of(&rotation_to_y(n));
        self.align_off = c;
        true
    }

    /// Set the scale so two picked points are `ruler` units apart.
    pub fn set_distance(&mut self) -> bool {
        let pts = self.pick_pts();
        if pts.len() != 2 || self.ruler <= 0.0 {
            return false;
        }
        let d = ((pts[0][0] - pts[1][0]).powi(2) + (pts[0][1] - pts[1][1]).powi(2) + (pts[0][2] - pts[1][2]).powi(2)).sqrt();
        if d < 1e-9 {
            return false;
        }
        self.scale = self.ruler / d;
        true
    }

    /// A card on the picked points, in the scene: centred on them, on
    /// their plane (three or more) or facing the camera at comp frame `t`
    /// (fewer), sized to cover them. None without picks or a solve.
    pub fn card_from_picks(&self, t: f32) -> Option<crate::scene3d::Xform> {
        let picks = self.picks();
        if picks.is_empty() {
            return None;
        }
        let pts: Vec<[f32; 3]> = picks.iter().filter(|&&k| k < self.solve.solve.points.len()).map(|&k| self.point_world(k)).collect();
        if pts.is_empty() {
            return None;
        }
        let c = centroid(&pts);
        let (cam, _) = self.camera_at(t)?;
        let to_cam = [cam.translate[0] - c[0], cam.translate[1] - c[1], cam.translate[2] - c[2]];
        let mut n = if pts.len() >= 3 { plane_normal(&pts).unwrap_or(unit3(to_cam)) } else { unit3(to_cam) };
        if n[0] * to_cam[0] + n[1] * to_cam[1] + n[2] * to_cam[2] < 0.0 {
            n = [-n[0], -n[1], -n[2]];
        }
        // Local +Z is the normal; local X level where it can be.
        let up = if n[1].abs() > 0.99 { [0.0, 0.0, 1.0] } else { [0.0, 1.0, 0.0] };
        let x = unit3(cross3(up, n));
        let y = cross3(n, x);
        let r = [[x[0], y[0], n[0]], [x[1], y[1], n[1]], [x[2], y[2], n[2]]];
        let mut reach = 0.0f32;
        for p in &pts {
            let d = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
            let px = d[0] * x[0] + d[1] * x[1] + d[2] * x[2];
            let py = d[0] * y[0] + d[1] * y[1] + d[2] * y[2];
            reach = reach.max(px.abs()).max(py.abs());
        }
        let size = (reach * 2.2).max(0.1);
        Some(crate::scene3d::Xform { translate: c, rotate: crate::scene3d::euler_of(&r), scale: [size, size, 1.0] })
    }
}

fn centroid(pts: &[[f32; 3]]) -> [f32; 3] {
    let n = pts.len().max(1) as f32;
    pts.iter().fold([0.0; 3], |a, p| [a[0] + p[0] / n, a[1] + p[1] / n, a[2] + p[2] / n])
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn unit3(a: [f32; 3]) -> [f32; 3] {
    let n = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    if n > 1e-12 {
        [a[0] / n, a[1] / n, a[2] / n]
    } else {
        [0.0, 0.0, 1.0]
    }
}

/// The unit normal of the plane through `pts` (least squares: the
/// smallest axis of their scatter). None when they do not span one.
pub fn plane_normal(pts: &[[f32; 3]]) -> Option<[f32; 3]> {
    if pts.len() < 3 {
        return None;
    }
    let c = centroid(pts);
    let mut cov = [0.0f64; 9];
    for p in pts {
        let d = [(p[0] - c[0]) as f64, (p[1] - c[1]) as f64, (p[2] - c[2]) as f64];
        for i in 0..3 {
            for j in 0..3 {
                cov[i * 3 + j] += d[i] * d[j];
            }
        }
    }
    let (vals, vecs) = crate::camtrack::sym_eig(&cov, 3);
    // A line (two equal smallest values) has no plane.
    if vals[1] < 1e-12 * vals[2].max(1e-30) {
        return None;
    }
    let v = &vecs[0];
    Some(unit3([v[0] as f32, v[1] as f32, v[2] as f32]))
}

/// The rotation taking unit `n` onto +Y, as the smallest turn.
pub fn rotation_to_y(n: [f32; 3]) -> [[f32; 3]; 3] {
    let y = [0.0, 1.0, 0.0];
    let axis = cross3(n, y);
    let s = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    let c = n[1].clamp(-1.0, 1.0);
    if s < 1e-9 {
        // Already up, or straight down: a half turn about X.
        return if c > 0.0 { crate::scene3d::identity_3() } else { [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]] };
    }
    let a = [axis[0] / s, axis[1] / s, axis[2] / s];
    let w = crate::camtrack::so3_exp([a[0] as f64 * s.atan2(c) as f64, a[1] as f64 * s.atan2(c) as f64, a[2] as f64 * s.atan2(c) as f64]);
    let mut r = [[0.0f32; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = w[i][j] as f32;
        }
    }
    r
}

impl crate::nodes::NodeDef for CamTrackParams {
    const NAME: &'static str = CamTrackParams::NAME;
    const INPUTS: usize = CamTrackParams::INPUTS;
    const MASK_INPUT: Option<usize> = None;
    const VEC4_SLOTS: usize = CamTrackParams::VEC4_SLOTS;
    fn topo_tag(&self) -> u64 {
        CamTrackParams::topo_tag(self)
    }
    fn pack(&self, _span: &mut [f32]) {}
    /// The picture passes through; the matte is read by the track alone.
    fn emit(&self, e: &mut crate::codegen::Emit) -> String {
        e.input(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camtrack::{synth, Pose, V3};

    fn solved() -> CamTrackParams {
        let n = 12;
        let solve = Solve {
            size: [1280, 720],
            focal_px: synth::F,
            poses: (0..n).map(|t| synth::camera(t, n)).collect(),
            unplaced: vec![],
            points: vec![[0.0, 0.0, 1.0]],
            point_track: vec![0],
            errors: vec![0.1; n],
        };
        CamTrackParams { solve: CamSolveRef::new(CamSolve { first: 1001, solve, recipe: 7 }), ..Default::default() }
    }

    #[test]
    fn the_camera_is_keyed_by_comp_frame_and_held_past_the_ends() {
        let p = solved();
        assert!(CamTrackParams::default().camera_at(0.0).is_none());
        let (x0, lens) = p.camera_at(1001.0).unwrap();
        assert!(x0.translate.iter().all(|v| v.abs() < 1e-6), "{x0:?}");
        // 1100 px on 1280 px wide with a 36 mm back: 30.9 mm.
        assert!((lens.focal - 1100.0 / 1280.0 * 36.0).abs() < 1e-3, "{}", lens.focal);
        assert!((lens.aperture[1] - 36.0 * 720.0 / 1280.0).abs() < 1e-3);
        let (x5, _) = p.camera_at(1006.0).unwrap();
        let truth = synth::camera(5, 12).centre();
        // Cards convention: y and z flipped.
        assert!((x5.translate[0] as f64 - truth[0]).abs() < 1e-4 && (x5.translate[1] as f64 + truth[1]).abs() < 1e-4 && (x5.translate[2] as f64 + truth[2]).abs() < 1e-4, "{x5:?} vs {truth:?}");
        let (before, _) = p.camera_at(900.0).unwrap();
        let (after, _) = p.camera_at(2000.0).unwrap();
        assert_eq!(before.translate, x0.translate);
        assert_eq!(after.translate, p.camera_at(1012.0).unwrap().0.translate);
        // Scale multiplies positions, never the lens.
        let mut q = p.clone();
        q.scale = 2.0;
        let (y5, lens2) = q.camera_at(1006.0).unwrap();
        assert!((y5.translate[0] - 2.0 * x5.translate[0]).abs() < 1e-5);
        assert_eq!(lens2.focal, lens.focal);
        // Halfway between frames is halfway between the positions.
        let (mid, _) = p.camera_at(1005.5).unwrap();
        let (x4, _) = p.camera_at(1005.0).unwrap();
        assert!((mid.translate[0] - 0.5 * (x4.translate[0] + x5.translate[0])).abs() < 1e-5);
    }

    #[test]
    fn the_alignment_ops_put_the_picks_where_they_say() {
        let mut p = solved();
        // Picks mean nothing until they belong to this solve.
        p.picked = vec![0];
        p.picked_for = 99;
        assert!(p.picks().is_empty());
        // Three points on a tilted floor, one off it, the first camera at
        // the origin above them (CV y down: the floor is at y = +1.2, so in
        // the cards frame y = −1.2).
        let floor: Vec<V3> = vec![[-1.0, 1.2, 3.0], [1.0, 1.2, 3.0], [0.0, 1.2, 5.0], [0.5, 1.2, 4.0]];
        let mut s = p.solve.solve.clone();
        s.points = floor.clone();
        s.point_track = vec![0, 1, 2, 3];
        p.solve = CamSolveRef::new(CamSolve { first: 1001, solve: s, recipe: 7 });
        for k in 0..3 {
            p.toggle_pick(k);
        }
        assert_eq!(p.picks(), &[0, 1, 2]);
        p.toggle_pick(1);
        p.toggle_pick(1);
        assert_eq!(p.picks(), &[0, 2, 1]);
        // Tilt the solve first so the ground has something to undo.
        p.align_rot = [20.0, 0.0, 10.0];
        assert!(p.set_ground());
        for k in 0..4 {
            let w = p.point_world(k);
            assert!(w[1].abs() < 1e-4, "point {k} at y {}", w[1]);
        }
        let (cam, _) = p.camera_at(1001.0).unwrap();
        assert!(cam.translate[1] > 1.0, "the camera is above the ground: {:?}", cam.translate);
        // The origin: the picks' centroid.
        let c = p.point_world(0);
        let d = p.point_world(2);
        let mid: [f32; 3] = std::array::from_fn(|i| (c[i] + d[i] + p.point_world(1)[i]) / 3.0);
        assert!(mid.iter().all(|v| v.abs() < 1e-4), "{mid:?}");
        // Two picks a known distance apart set the scale.
        p.picked = vec![0, 1];
        p.ruler = 4.0;
        assert!(p.set_distance());
        assert!((p.scale - 2.0).abs() < 1e-5, "{}", p.scale);
        let (a, b) = (p.point_world(0), p.point_world(1));
        assert!(((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt() - 4.0 < 1e-4);
        // A card on three floor picks lies on the floor, facing up at the
        // camera, wide enough to cover them.
        p.picked = vec![0, 1, 2];
        let card = p.card_from_picks(1001.0).unwrap();
        let r = crate::scene3d::rotation(card.rotate);
        let nz = [r[0][2], r[1][2], r[2][2]];
        assert!(nz[1] > 0.999, "card normal {nz:?}");
        assert!(card.scale[0] > 2.0, "{:?}", card.scale);
        // One pick: a card facing the camera.
        p.picked = vec![3];
        let card = p.card_from_picks(1001.0).unwrap();
        let (cam, _) = p.camera_at(1001.0).unwrap();
        let r = crate::scene3d::rotation(card.rotate);
        let nz = [r[0][2], r[1][2], r[2][2]];
        let to = unit3([cam.translate[0] - card.translate[0], cam.translate[1] - card.translate[1], cam.translate[2] - card.translate[2]]);
        assert!(nz[0] * to[0] + nz[1] * to[1] + nz[2] * to[2] > 0.999, "{nz:?} vs {to:?}");
    }

    #[test]
    fn settings_split_into_the_two_recipes_and_none_is_topology() {
        let a = CamTrackParams::default();
        let mut b = a.clone();
        b.focal = 35.0;
        b.refine_focal = false;
        b.scale = 3.0;
        b.show = false;
        assert_eq!(a.recipe_tracks(), b.recipe_tracks());
        assert_ne!(a.recipe_solve(), b.recipe_solve());
        assert_eq!(a.topo_tag(), b.topo_tag());
        b.divisor = 4;
        assert_ne!(a.recipe_tracks(), b.recipe_tracks());
        // Guessing ignores the focal: two guessers share a recipe.
        let mut g1 = a.clone();
        g1.guess = true;
        let mut g2 = g1.clone();
        g2.focal = 85.0;
        g2.refine_focal = false;
        assert_eq!(g1.recipe_solve(), g2.recipe_solve());
        assert_ne!(g1.recipe_solve(), a.recipe_solve());
        let c = solved();
        let d = c.clone();
        assert!(c.solve.ptr_eq(&d.solve));
        assert_eq!(c, d);
        let _ = Pose::IDENTITY;
    }

    #[cfg(feature = "serde")]
    #[test]
    fn the_solve_saves_with_the_node() {
        let p = solved();
        let j = serde_json::to_string(&p).unwrap();
        let back: CamTrackParams = serde_json::from_str(&j).unwrap();
        // JSON keeps a float to within its last digit, not bit for bit.
        let (a, b) = (&p.solve.solve, &back.solve.solve);
        assert_eq!(a.poses.len(), b.poses.len());
        for (x, y) in a.poses.iter().zip(&b.poses) {
            for i in 0..3 {
                assert!((x.t[i] - y.t[i]).abs() < 1e-12);
                for j in 0..3 {
                    assert!((x.r[i][j] - y.r[i][j]).abs() < 1e-12);
                }
            }
        }
        assert_eq!((a.points.clone(), a.errors.clone()), (b.points.clone(), b.errors.clone()));
        assert_eq!(back.solve.first, 1001);
    }
}
