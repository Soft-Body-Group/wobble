//! A 3D view as Blender's `RegionView3D` keeps it — a pivot, an
//! orientation, a distance — and Blender's turntable hands on it. The
//! same record travels both ways on the socket: Blender's viewport (or
//! camera) as the user left it, and wobble's orbit as the hand moves it.
//!
//! Pure maths, no egui, so it is unit-tested here and the app only
//! measures pointer deltas. The projection is Blender's business (the
//! script builds it from `lens`/`ortho` and the draw size), which keeps
//! one formula in one place.

/// Blender's `view_location` / `view_rotation` / `view_distance`, plus the
/// lens the projection uses.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewState {
    /// The pivot the view orbits about, world space.
    pub loc: [f32; 3],
    /// View → world orientation, quaternion (w, x, y, z): what the eye's
    /// axes are in the world. The eye sits at `loc + rot * (0, 0, dist)`.
    pub rot: [f32; 4],
    pub dist: f32,
    /// Focal length in mm (the 3D view's, or the camera's).
    pub lens: f32,
    pub ortho: bool,
}

impl Default for ViewState {
    /// Blender's default viewport: looking at the origin from above and
    /// in front, ten units off.
    fn default() -> Self {
        ViewState {
            loc: [0.0, 0.0, 0.0],
            rot: [0.716, 0.439, 0.291, 0.459],
            dist: 10.0,
            lens: 50.0,
            ortho: false,
        }
    }
}

/// Blender's turntable sensitivity (`view_rotate_sensitivity_turntable`
/// default 0.4° per pixel).
pub const TURNTABLE_RAD_PER_PX: f32 = 0.4 * std::f32::consts::PI / 180.0;

fn qmul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    let [aw, ax, ay, az] = a;
    let [bw, bx, by, bz] = b;
    [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]
}

fn qaxis(axis: [f32; 3], angle: f32) -> [f32; 4] {
    let (s, c) = (angle * 0.5).sin_cos();
    [c, axis[0] * s, axis[1] * s, axis[2] * s]
}

fn qnorm(q: [f32; 4]) -> [f32; 4] {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt().max(1e-9);
    [q[0] / n, q[1] / n, q[2] / n, q[3] / n]
}

/// Rotate `v` by `q`.
pub fn qrot(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    let [w, x, y, z] = q;
    // v + 2w(u×v) + 2u×(u×v), u = (x,y,z)
    let cross = |a: [f32; 3], b: [f32; 3]| {
        [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
    };
    let u = [x, y, z];
    let uv = cross(u, v);
    let uuv = cross(u, uv);
    [
        v[0] + 2.0 * (w * uv[0] + uuv[0]),
        v[1] + 2.0 * (w * uv[1] + uuv[1]),
        v[2] + 2.0 * (w * uv[2] + uuv[2]),
    ]
}

impl ViewState {
    /// Where the eye is.
    pub fn eye(&self) -> [f32; 3] {
        let d = qrot(self.rot, [0.0, 0.0, self.dist]);
        [self.loc[0] + d[0], self.loc[1] + d[1], self.loc[2] + d[2]]
    }

    /// The view's right and up axes in the world.
    pub fn axes(&self) -> ([f32; 3], [f32; 3]) {
        (qrot(self.rot, [1.0, 0.0, 0.0]), qrot(self.rot, [0.0, 1.0, 0.0]))
    }

    /// Blender's turntable: a horizontal drag turns the view about the
    /// WORLD's up axis, a vertical one tilts it about the view's own
    /// sideways axis, so the horizon never rolls. The hand grabs the
    /// scene: drag right and the scene turns right under it, so the eye
    /// travels LEFT around the pivot and the scene's left side comes into
    /// view; drag down and its top comes into view (the eye rises).
    pub fn orbit(&mut self, dx_px: f32, dy_px: f32) {
        let yaw = -dx_px * TURNTABLE_RAD_PER_PX;
        let pitch = -dy_px * TURNTABLE_RAD_PER_PX;
        // World Z first (applied on the world side of the orientation),
        // then the view's local X (on the view side).
        let q = qmul(qaxis([0.0, 0.0, 1.0], yaw), self.rot);
        self.rot = qnorm(qmul(q, qaxis([1.0, 0.0, 0.0], pitch)));
    }

    /// Move the pivot so the picture follows the hand: `units_per_px` is
    /// what one screen pixel is at the pivot's depth.
    pub fn pan(&mut self, dx_px: f32, dy_px: f32, units_per_px: f32) {
        let (right, up) = self.axes();
        let (sx, sy) = (-dx_px * units_per_px, dy_px * units_per_px);
        for i in 0..3 {
            self.loc[i] += right[i] * sx + up[i] * sy;
        }
    }

    /// Dolly: scale the distance (a factor above 1 backs off).
    pub fn dolly(&mut self, factor: f32) {
        self.dist = (self.dist * factor).clamp(1e-3, 1e6);
    }

    /// Half the vertical field of view's tangent, for a draw `w` by `h`:
    /// Blender's 3D view fits a 36mm sensor to the larger side and doubles
    /// the view (its perspective zoom), so a 50mm viewport lens sees
    /// wider than a 50mm camera; a camera fits its 36mm sensor plainly.
    pub fn tan_half_y(&self, w: u32, h: u32, camera: bool) -> f32 {
        let vf = w.max(h).max(1) as f32;
        let sensor = if camera { 18.0 } else { 36.0 };
        (h as f32 / vf) * sensor / self.lens.max(1.0)
    }

    /// World units per screen pixel at the pivot, for a picture drawn `w`
    /// by `h` and shown `px_high` pixels tall.
    pub fn units_per_px(&self, w: u32, h: u32, camera: bool, px_high: f32) -> f32 {
        let half = if self.ortho {
            // Ortho: the view is `dist * 36 / lens` units across the
            // larger side, doubled like the perspective.
            (h as f32 / w.max(h).max(1) as f32) * self.dist * 36.0 / self.lens.max(1.0)
        } else {
            self.dist * self.tan_half_y(w, h, camera)
        };
        2.0 * half / px_high.max(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: [f32; 3], b: [f32; 3]) -> bool {
        (0..3).all(|i| (a[i] - b[i]).abs() < 1e-4)
    }

    #[test]
    fn the_eye_sits_dist_behind_the_pivot_along_the_view_axis() {
        // Identity: looking down -Z from +Z.
        let v = ViewState { rot: [1.0, 0.0, 0.0, 0.0], dist: 5.0, ..Default::default() };
        assert!(close(v.eye(), [0.0, 0.0, 5.0]));
        let (r, u) = v.axes();
        assert!(close(r, [1.0, 0.0, 0.0]) && close(u, [0.0, 1.0, 0.0]));
    }

    #[test]
    fn a_horizontal_orbit_turns_about_world_z_and_keeps_the_horizon() {
        // Looking along +Y from -Y (a front view): eye at (0,-d,0), up = +Z.
        let front = qaxis([1.0, 0.0, 0.0], std::f32::consts::FRAC_PI_2);
        let mut v = ViewState { rot: front, dist: 4.0, ..Default::default() };
        assert!(close(v.eye(), [0.0, -4.0, 0.0]), "{:?}", v.eye());
        let (_, up0) = v.axes();
        assert!(close(up0, [0.0, 0.0, 1.0]));
        // A quarter turn's worth of drag to the right.
        let px = std::f32::consts::FRAC_PI_2 / TURNTABLE_RAD_PER_PX;
        v.orbit(px, 0.0);
        let e = v.eye();
        assert!((e[2]).abs() < 1e-3, "height unchanged: {e:?}");
        assert!(((e[0] * e[0] + e[1] * e[1]).sqrt() - 4.0).abs() < 1e-3, "same distance: {e:?}");
        assert!(e[0] < -3.9, "the eye travelled to the left (-X): {e:?}");
        let (_, up) = v.axes();
        assert!(close(up, [0.0, 0.0, 1.0]), "no roll: up {up:?}");
    }

    #[test]
    fn a_vertical_orbit_tilts_about_the_views_own_x() {
        let front = qaxis([1.0, 0.0, 0.0], std::f32::consts::FRAC_PI_2);
        let mut v = ViewState { rot: front, dist: 4.0, ..Default::default() };
        let px = std::f32::consts::FRAC_PI_4 / TURNTABLE_RAD_PER_PX;
        v.orbit(0.0, px); // drag down: the eye rises over the pivot
        let e = v.eye();
        assert!(e[2] > 2.5 && e[1] < -2.5, "{e:?}");
        let (r, _) = v.axes();
        assert!(close(r, [1.0, 0.0, 0.0]), "sideways axis unchanged: {r:?}");
    }

    #[test]
    fn pan_moves_the_pivot_in_the_view_plane_and_dolly_scales_the_distance() {
        let mut v = ViewState { rot: [1.0, 0.0, 0.0, 0.0], dist: 2.0, ..Default::default() };
        v.pan(10.0, 0.0, 0.1);
        assert!(close(v.loc, [-1.0, 0.0, 0.0]), "{:?}", v.loc);
        v.dolly(2.0);
        assert_eq!(v.dist, 4.0);
    }

    #[test]
    fn the_viewport_sees_twice_as_wide_as_a_camera_of_the_same_lens() {
        let v = ViewState { lens: 50.0, ..Default::default() };
        let free = v.tan_half_y(1920, 1080, false);
        let cam = v.tan_half_y(1920, 1080, true);
        assert!((free / cam - 2.0).abs() < 1e-5);
        assert!((free - (1080.0 / 1920.0) * 36.0 / 50.0).abs() < 1e-5);
    }
}
