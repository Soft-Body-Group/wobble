//! A view of things at infinity: a camera's rotation and lens, nothing
//! else. The Env node (a latlong background) and the Starfield read the
//! sky through it.
//!
//! The Camera3D convention (`crate::scene3d`): Y up, right-handed, the
//! camera looks down −Z, rotations in degrees applied X then Y then Z
//! (`R = Rz · Ry · Rx`), the filmback's WIDTH fitted to the picture's
//! width. Translation means nothing at infinity, so a [`SkyCam`] has none:
//! pan a tracked camera and the sky slides exactly as a real one would,
//! dolly it and nothing moves.
//!
//! One packing, three vec4s ([`pack_rows`]), and one WGSL function over it
//! ([`WGSL`], `sky_ray(p, slot)`), so every node that looks at the sky
//! agrees about which way a pixel points.

use crate::graph::{Graph, NodeId, NodeKind};

/// A camera at the centre of the sky.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SkyCam {
    /// Euler degrees, the Camera3D convention.
    pub rotate: [f32; 3],
    /// Focal length and filmback width, mm.
    pub focal: f32,
    pub aperture: f32,
}

impl SkyCam {
    /// Focal length in pixels for a picture `w` wide.
    pub fn focal_px(&self, w: f32) -> f32 {
        self.focal.max(1e-3) / self.aperture.max(1e-3) * w.max(1.0)
    }

    /// The unit world direction through pixel `p` (pixels, y down, centres
    /// at +0.5) of a `w` x `h` picture — the CPU mirror of `sky_ray`.
    pub fn ray(&self, p: [f32; 2], w: f32, h: f32) -> [f32; 3] {
        let f = self.focal_px(w);
        let d = [(p[0] - 0.5 * w) / f, -(p[1] - 0.5 * h) / f, -1.0];
        let r = crate::scene3d::rotation(self.rotate);
        let o = [0, 1, 2].map(|i| r[i][0] * d[0] + r[i][1] * d[1] + r[i][2] * d[2]);
        let l = (o[0] * o[0] + o[1] * o[1] + o[2] * o[2]).sqrt();
        o.map(|x| x / l)
    }

    /// The pixel a world direction lands on, if it is in front of the
    /// camera — the inverse of [`ray`](Self::ray).
    pub fn project(&self, d: [f32; 3], w: f32, h: f32) -> Option<[f32; 2]> {
        let f = self.focal_px(w);
        let r = crate::scene3d::rotation(self.rotate);
        let c = [0, 1, 2].map(|j| r[0][j] * d[0] + r[1][j] * d[1] + r[2][j] * d[2]);
        if c[2] >= -1e-6 {
            return None;
        }
        Some([0.5 * w + f * c[0] / -c[2], 0.5 * h - f * c[1] / -c[2]])
    }

    /// The view of a Camera3D (its source already applied: a CameraTrack's
    /// solve, a file, Blender all arrive as the node's values).
    pub fn of_camera(c: &crate::scene3d::CameraParams) -> Self {
        SkyCam { rotate: c.xform.rotate, focal: c.focal, aperture: c.aperture[0] }
    }
}

/// The Camera3D wired into `node`'s `port`, as a view. Run after animation
/// and camera sources are applied (next to `scene3d::expand`).
pub fn camera_on(g: &Graph, node: NodeId, port: usize) -> Option<SkyCam> {
    match g.input_of(node, port).and_then(|c| g.nodes.get(&c)) {
        Some(NodeKind::Camera3D(c)) => Some(SkyCam::of_camera(c)),
        _ => None,
    }
}

/// Three vec4s: the rotation's rows in `.xyz`, then focal px, format
/// width, format height in the `.w`s. `rot` is applied after the camera's
/// own (a node turning the sky under the camera passes it; identity
/// otherwise).
pub fn pack_rows_turned(v: &mut [f32], cam: &SkyCam, fmt: [f32; 2], rot: &[[f32; 3]; 3]) {
    let c = crate::scene3d::rotation(cam.rotate);
    for i in 0..3 {
        for j in 0..3 {
            v[i * 4 + j] = (0..3).map(|k| rot[i][k] * c[k][j]).sum();
        }
    }
    v[3] = cam.focal_px(fmt[0]);
    v[7] = fmt[0];
    v[11] = fmt[1];
}

/// [`pack_rows_turned`] with nothing turned.
pub fn pack_rows(v: &mut [f32], cam: &SkyCam, fmt: [f32; 2]) {
    pack_rows_turned(v, cam, fmt, &crate::scene3d::identity_3());
}

/// The helper's name in the shader's function map.
pub const WGSL_NAME: &str = "sky_ray";

/// `sky_ray(p, slot)`: the unit world direction through pixel `p`, read
/// off the three vec4s [`pack_rows`] wrote at `slot`.
pub const WGSL: &str = "// The world direction through a pixel (skyview::pack_rows at slot).
fn sky_ray(p: vec2f, slot: u32) -> vec3f {
    let r0 = P.v[slot];
    let r1 = P.v[slot + 1u];
    let r2 = P.v[slot + 2u];
    let fmt = vec2f(r1.w, r2.w);
    let dc = vec3f((p.x - 0.5 * fmt.x) / r0.w, -(p.y - 0.5 * fmt.y) / r0.w, -1.0);
    return normalize(vec3f(dot(r0.xyz, dc), dot(r1.xyz, dc), dot(r2.xyz, dc)));
}
";

/// Put `sky_ray` in a shader's function map once; its name back.
pub fn wgsl(fns: &mut std::collections::BTreeMap<String, String>) -> &'static str {
    fns.entry(WGSL_NAME.into()).or_insert_with(|| WGSL.into());
    WGSL_NAME
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ray_and_project_are_inverses() {
        let c = SkyCam { rotate: [12.0, -40.0, 7.0], focal: 50.0, aperture: 36.0 };
        for px in [[0.0, 0.0], [960.0, 540.0], [1900.0, 100.0]] {
            let d = c.ray(px, 1920.0, 1080.0);
            let q = c.project(d, 1920.0, 1080.0).unwrap();
            assert!((q[0] - px[0]).abs() < 1e-2 && (q[1] - px[1]).abs() < 1e-2, "{px:?} -> {q:?}");
        }
    }

    /// The packed rows ARE the rotation the CPU ray uses.
    #[test]
    fn the_packed_rows_give_the_cpu_ray() {
        let c = SkyCam { rotate: [20.0, 70.0, -10.0], focal: 24.0, aperture: 36.0 };
        let mut v = [0.0f32; 12];
        pack_rows(&mut v, &c, [640.0, 360.0]);
        let p = [100.0f32, 300.0];
        let f = v[3];
        let dc = [(p[0] - 320.0) / f, -(p[1] - 180.0) / f, -1.0];
        let d = [0, 1, 2].map(|i| (0..3).map(|j| v[i * 4 + j] * dc[j]).sum::<f32>());
        let l = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        let want = c.ray(p, 640.0, 360.0);
        for i in 0..3 {
            assert!((d[i] / l - want[i]).abs() < 1e-5, "{d:?} {want:?}");
        }
    }
}
