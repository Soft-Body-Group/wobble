//! Env: a latlong (equirectangular) picture as a background at infinity,
//! seen through a camera.
//!
//! Every output pixel is a ray (`crate::skyview`: the camera's rotation
//! and lens, the Camera3D convention), the ray is turned into a longitude
//! and latitude, and the latlong is sampled there. It is at infinity, so a
//! camera's TRANSLATION does nothing: pan a tracked camera and the
//! background slides exactly as a distant one would, dolly it and the
//! background holds.
//!
//! **The latlong convention is Blender's** (Environment Texture,
//! equirectangular), so an HDRI lined up in Blender lines up here with
//! the camera a Blender sync or a USD export hands over. In Blender's
//! Z-up space `u = (π − atan2(y, −x)) / 2π`, `v = ½ + asin(z) / π` (v
//! up); through the importer's −90° about X (our `z = −y_b`) that is
//! `u = (π − atan2(−z, −x)) / 2π`. The picture's centre is therefore −X,
//! which is why the manual view starts turned 90° about Y — looking down
//! −X, at the middle of the latlong.
//!
//! **It fuses.** The latlong's subtree is re-evaluated at the looked-up
//! position (`Emit::input_at`, the UV Warp pattern): no pass, no texture,
//! no stage, cyan. Sampling is the leaves' bilinear; the seam at u = 0
//! wraps by `fract`, so it is exact to within the half pixel a clamped
//! bilinear read leaves at the picture's edge.
//!
//! **A camera on input 1 drives the view.** The scene pipe (a Camera3D,
//! including one sourced from a CameraTrack, a file or Blender); [`lower`]
//! — after animation and sources, next to `scene3d::expand` — copies its
//! rotation and lens into the transient [`EnvParams::cam`] and drops the
//! wire. Unwired, the node's own `rotate` and `focal` are the view. `spin`
//! turns the latlong about the vertical under either view (lining up an
//! HDRI's sun with the plate's). Everything is a uniform.

use crate::codegen::Emit;
use crate::graph::{Graph, NodeId, NodeKind};
use crate::nodes::NodeDef;
use crate::skyview::SkyCam;

const TOPO: u64 = 0xE7_0000;

/// The latlong.
pub const IMAGE_INPUT: usize = 0;
/// The camera, on the scene pipe (lowered away before codegen).
pub const CAMERA_INPUT: usize = 1;

/// Channel prefix for the keyable numbers (`env.<name>` on
/// `Channel::Custom`, the Text / Starfield precedent).
pub const PREFIX: &str = "env.";

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct EnvParams {
    /// The picture's format (see `inherit`): a camera frame, not the
    /// latlong's.
    pub size: (u32, u32),
    /// Take the script's canvas instead of `size`.
    pub inherit: bool,
    /// The view when no camera is wired: Euler degrees (Camera3D's).
    pub rotate: [f32; 3],
    /// Focal length, mm, on `aperture` mm of filmback width.
    pub focal: f32,
    pub aperture: f32,
    /// Turns the latlong about the vertical, degrees.
    pub spin: f32,
    /// Multiplies the picture (stops, 2^x): HDRIs are rarely exposed for
    /// the plate.
    pub exposure: f32,
    /// Set by [`lower`] when a camera is wired; never authored.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub cam: Option<SkyCam>,
}

impl Default for EnvParams {
    fn default() -> Self {
        Self {
            size: (1920, 1080),
            inherit: false,
            rotate: [0.0, 90.0, 0.0],
            focal: 35.0,
            aperture: 36.0,
            spin: 0.0,
            exposure: 0.0,
            cam: None,
        }
    }
}

const NAMES: &[&str] = &["rotate.x", "rotate.y", "rotate.z", "focal", "spin", "exposure"];

impl EnvParams {
    /// Every keyable number's name (after [`PREFIX`]).
    pub fn names() -> &'static [&'static str] {
        NAMES
    }

    fn field(&mut self, name: &str) -> Option<&mut f32> {
        Some(match name {
            "rotate.x" => &mut self.rotate[0],
            "rotate.y" => &mut self.rotate[1],
            "rotate.z" => &mut self.rotate[2],
            "focal" => &mut self.focal,
            "spin" => &mut self.spin,
            "exposure" => &mut self.exposure,
            _ => return None,
        })
    }

    pub fn get(&self, name: &str) -> Option<f32> {
        self.clone().field(name).map(|v| *v)
    }

    pub fn set(&mut self, name: &str, v: f32) -> bool {
        match self.field(name) {
            Some(f) => {
                *f = v;
                true
            }
            None => false,
        }
    }

    /// The view this frame: the wired camera's, else the node's own.
    pub fn view(&self) -> SkyCam {
        self.cam.unwrap_or(SkyCam { rotate: self.rotate, focal: self.focal, aperture: self.aperture })
    }

    /// The turn `spin` applies to a world direction before the lookup.
    fn spin_rot(&self) -> [[f32; 3]; 3] {
        crate::scene3d::rotation([0.0, -self.spin, 0.0])
    }

    /// Where pixel `p` of a `fmt` picture samples a `src` latlong, in the
    /// latlong's pixels (y down) — the CPU mirror of the shader.
    pub fn lookup(&self, p: [f32; 2], fmt: [f32; 2], src: [f32; 2]) -> [f32; 2] {
        let d = self.view().ray(p, fmt[0], fmt[1]);
        let r = self.spin_rot();
        let d = [0, 1, 2].map(|i| r[i][0] * d[0] + r[i][1] * d[1] + r[i][2] * d[2]);
        latlong_px(d, src)
    }

    /// [`NodeDef::pack`] with the node's format and the latlong's.
    pub fn pack_fmt(&self, v: &mut [f32], fmt: [f32; 2], src: [f32; 2]) {
        crate::skyview::pack_rows_turned(v, &self.view(), fmt, &self.spin_rot());
        v[12] = src[0];
        v[13] = src[1];
        v[14] = self.exposure.exp2();
        v[15] = 0.0;
    }
}

/// The latlong pixel (y down) a unit direction lands on, in a `src`
/// picture — Blender's equirectangular mapping (see the module docs).
pub fn latlong_px(d: [f32; 3], src: [f32; 2]) -> [f32; 2] {
    use std::f32::consts::PI;
    let u = (PI - (-d[2]).atan2(-d[0])) / (2.0 * PI);
    let v = 0.5 - d[1].clamp(-1.0, 1.0).asin() / PI;
    [(u - u.floor()) * src[0], v * src[1]]
}

/// Copy each Env's wired camera into its transient `cam` and drop the
/// scene wire, so codegen sees a node with one picture input. Run after
/// animation and camera sources are applied.
pub fn lower(g: &mut Graph) {
    let ids: Vec<NodeId> =
        g.nodes.iter().filter(|(_, k)| matches!(k, NodeKind::Env(_))).map(|(id, _)| *id).collect();
    for id in ids {
        let cam = crate::skyview::camera_on(g, id, CAMERA_INPUT);
        g.edges.remove(&(id, CAMERA_INPUT));
        g.edge_out.remove(&(id, CAMERA_INPUT));
        if let Some(NodeKind::Env(p)) = g.nodes.get_mut(&id) {
            p.cam = cam;
        }
    }
}

impl NodeDef for EnvParams {
    const NAME: &'static str = "Env";
    /// The latlong, then the camera.
    const INPUTS: usize = 2;
    const MASK_INPUT: Option<usize> = None;
    /// Rotation rows + focal/format (`skyview::pack_rows`), then (latlong
    /// w, h, gain, -).
    const VEC4_SLOTS: usize = 4;

    /// One topology whatever the node holds.
    fn topo_tag(&self) -> u64 {
        TOPO
    }

    fn pack(&self, v: &mut [f32]) {
        let fmt = [self.size.0 as f32, self.size.1 as f32];
        self.pack_fmt(v, fmt, fmt);
    }

    fn emit(&self, e: &mut Emit) -> String {
        if !e.has_input(IMAGE_INPUT) {
            return "vec4f(0.0)".into();
        }
        let slot = e.slot();
        let pos = e.pos().to_string();
        let (_, fns) = e.color_helpers();
        let ray = crate::skyview::wgsl(fns);
        let (d, q, v) = (e.fresh(), e.fresh(), e.fresh());
        e.line(format_args!("let {d} = {ray}({pos}, {slot}u);"));
        // Blender's equirectangular: u = (pi - atan2(-z, -x)) / 2pi, v down.
        e.line(format_args!(
            "let {q} = vec2f(fract((3.14159265 - atan2(-{d}.z, -{d}.x)) * 0.15915494), \
             0.5 - asin(clamp({d}.y, -1.0, 1.0)) * 0.31830989) * P.v[{slot}u + 3u].xy;"
        ));
        let s = e.input_at(IMAGE_INPUT, &q);
        e.line(format_args!("let {v} = {s} * P.v[{slot}u + 3u].z;"));
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_edit_is_a_uniform() {
        let a = EnvParams::default();
        let mut b = a.clone();
        b.rotate = [10.0, 20.0, 30.0];
        b.spin = 45.0;
        b.exposure = 2.0;
        b.cam = Some(SkyCam { rotate: [1.0; 3], focal: 80.0, aperture: 24.0 });
        assert_eq!(a.topo_tag(), b.topo_tag());
    }

    #[test]
    fn every_name_reads_and_writes() {
        let mut p = EnvParams::default();
        for (i, n) in EnvParams::names().iter().enumerate() {
            assert!(p.set(n, i as f32 + 0.5), "{n}");
            assert_eq!(p.get(n), Some(i as f32 + 0.5), "{n}");
        }
        assert!(!p.set("nope", 1.0));
    }

    /// Blender's mapping: the centre is −X, a quarter in is −Z (Blender's
    /// +Y), up is the top row.
    #[test]
    fn the_latlong_is_blenders() {
        let s = [400.0, 200.0];
        let near = |a: [f32; 2], b: [f32; 2]| (a[0] - b[0]).abs() < 1e-3 && (a[1] - b[1]).abs() < 1e-3;
        assert!(near(latlong_px([-1.0, 0.0, 0.0], s), [200.0, 100.0]));
        assert!(near(latlong_px([0.0, 0.0, -1.0], s), [100.0, 100.0]));
        assert!(near(latlong_px([0.0, 0.0, 1.0], s), [300.0, 100.0]));
        assert!(latlong_px([0.0, 1.0, 0.0], s)[1].abs() < 1e-3);
        assert!((latlong_px([0.0, -1.0, 0.0], s)[1] - 200.0).abs() < 1e-3);
    }

    /// The default manual view looks at the middle of the latlong.
    #[test]
    fn the_default_view_looks_at_the_centre() {
        let q = EnvParams::default().lookup([960.0, 540.0], [1920.0, 1080.0], [4096.0, 2048.0]);
        assert!((q[0] - 2048.0).abs() < 0.5 && (q[1] - 1024.0).abs() < 0.5, "{q:?}");
    }

    /// Spin slides the lookup along the latlong; a full turn is no turn.
    #[test]
    fn spin_slides_the_longitude() {
        let mut p = EnvParams::default();
        let (fmt, src) = ([1920.0, 1080.0], [3600.0, 1800.0]);
        let c = [960.0, 540.0];
        p.spin = 90.0;
        let q = p.lookup(c, fmt, src);
        assert!(((q[0] - 1800.0).abs() - 900.0).abs() < 0.5, "{q:?}");
        p.spin = 360.0;
        let q = p.lookup(c, fmt, src);
        assert!((q[0] - 1800.0).abs() < 0.5 || (q[0] - 1800.0).abs() > 3599.0, "{q:?}");
    }

    /// Translation means nothing at infinity; rotation and lens are all.
    #[test]
    fn lowering_takes_the_cameras_rotation_and_lens() {
        let mut g = Graph::default();
        let cam = crate::scene3d::CameraParams {
            xform: crate::scene3d::Xform {
                translate: [10.0, -4.0, 99.0],
                rotate: [5.0, 6.0, 7.0],
                ..Default::default()
            },
            focal: 85.0,
            ..Default::default()
        };
        g.nodes.insert(1, NodeKind::Camera3D(cam));
        g.nodes.insert(2, NodeKind::Env(EnvParams::default()));
        g.connect_out(2, CAMERA_INPUT, 1, 0);
        lower(&mut g);
        assert!(g.input_of(2, CAMERA_INPUT).is_none());
        let Some(NodeKind::Env(p)) = g.nodes.get(&2) else { panic!() };
        assert_eq!(p.view(), SkyCam { rotate: [5.0, 6.0, 7.0], focal: 85.0, aperture: 36.0 });
    }
}
