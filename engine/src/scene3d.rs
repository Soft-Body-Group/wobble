//! 3D cards: textured planes seen through a camera — projection
//! compositing, and nothing more. No meshes, no lights, no shading.
//!
//! Four nodes on one new pipe (`PortKind::Scene`):
//!
//! - **Camera** — a transform, a focal length, a filmback, near/far;
//! - **Card** — an image on a plane, a transform, opacity;
//! - **Scene** — gathers cards (and other Scenes) so a big setup stays a
//!   tree rather than a fan;
//! - **Render3D** — a camera and a scene in, the picture (and its depth)
//!   out.
//!
//! The scene pipe is CPU data, like the text pipe: nothing on it is ever
//! sampled. [`expand`] lowers each Render3D, after animation, into ONE
//! stage whose aux ports are the card images and whose uniforms are the
//! card and camera matrices for the frame (`nodes::render3d`), so moving
//! a card or the camera is a uniform write and only adding or removing a
//! card changes topology. Camera, Card and Scene nodes stay in the engine
//! graph as orphans: viewed directly a Card shows its image, the others
//! transparent black.
//!
//! Conventions, fixed — imports and the Blender sync convert to them:
//! Y up, right-handed, the camera looks down −Z, lengths in scene units,
//! rotations in degrees applied X then Y then Z (`R = Rz · Ry · Rx`). A
//! card is one unit wide before its scale, its height the width over its
//! aspect (the image's own unless set), centred on its origin, facing +Z.

use crate::graph::{Graph, NodeId, NodeKind};

pub mod import;

/// Cards one Render3D draws. Each is a texture binding, and a stage gets
/// sixteen: the cap is the device's, not a choice. Past it the panel says
/// how many were left out.
pub const MAX_CARDS: usize = 16;
/// Shutter samples for motion blur.
pub const MAX_SAMPLES: usize = 16;

/// Where a camera or card's transform comes from.
#[derive(Clone, Debug, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Source {
    /// The node's own (keyable) values.
    #[default]
    Wobble,
    /// An imported file (`scene3d::import`): JSON, .chan or .usda. `object`
    /// names the camera or transform in it; empty takes the first. The
    /// file's frame is the comp frame plus `offset`.
    File {
        path: String,
        object: String,
        #[cfg_attr(feature = "serde", serde(default))]
        offset: f32,
    },
    /// A Blender node's live scene: `object` names the Blender object;
    /// empty on a Camera takes the scene camera.
    Blender { node: u64, object: String },
    /// A CameraTrack node's solve (cameras only): the pose and lens it
    /// solved for the frame.
    Tracker { node: u64 },
}

/// A transform: translate, then rotate (degrees, X then Y then Z), then
/// scale, about the origin.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct Xform {
    pub translate: [f32; 3],
    pub rotate: [f32; 3],
    pub scale: [f32; 3],
}

impl Default for Xform {
    fn default() -> Self {
        Self { translate: [0.0; 3], rotate: [0.0; 3], scale: [1.0; 3] }
    }
}

/// 4x4, row-major, acting on column vectors.
pub type Mat4 = [[f32; 4]; 4];

pub fn mat_mul(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut m = [[0.0f32; 4]; 4];
    for (r, row) in m.iter_mut().enumerate() {
        for (c, v) in row.iter_mut().enumerate() {
            *v = (0..4).map(|k| a[r][k] * b[k][c]).sum();
        }
    }
    m
}

pub fn identity_3() -> [[f32; 3]; 3] {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

pub fn identity() -> Mat4 {
    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
}

/// The rotation part, `Rz · Ry · Rx`, degrees.
pub fn rotation(r: [f32; 3]) -> [[f32; 3]; 3] {
    let (sx, cx) = r[0].to_radians().sin_cos();
    let (sy, cy) = r[1].to_radians().sin_cos();
    let (sz, cz) = r[2].to_radians().sin_cos();
    let rx = [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]];
    let ry = [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]];
    let rz = [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]];
    m3_mul(&rz, &m3_mul(&ry, &rx))
}

fn m3_mul(a: &[[f32; 3]; 3], b: &[[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let mut m = [[0.0f32; 3]; 3];
    for (r, row) in m.iter_mut().enumerate() {
        for (c, v) in row.iter_mut().enumerate() {
            *v = (0..3).map(|k| a[r][k] * b[k][c]).sum();
        }
    }
    m
}

/// Euler degrees (X then Y then Z) back out of a rotation matrix: what an
/// importer hands a node. The inverse of [`rotation`] away from gimbal
/// lock (|ry| = 90°), where it puts all the turn in X.
pub fn euler_of(m: &[[f32; 3]; 3]) -> [f32; 3] {
    // m = Rz Ry Rx: m[2][0] = -sin(ry).
    let sy = (-m[2][0]).clamp(-1.0, 1.0);
    let ry = sy.asin();
    let (rx, rz) = if sy.abs() < 0.99999 {
        (m[2][1].atan2(m[2][2]), m[1][0].atan2(m[0][0]))
    } else {
        ((-m[1][2]).atan2(m[1][1]), 0.0)
    };
    [rx.to_degrees(), ry.to_degrees(), rz.to_degrees()]
}

impl Xform {
    /// Local → world, scale included.
    pub fn matrix(&self) -> Mat4 {
        self.matrix_scaled([1.0; 3])
    }

    /// Local → world with an extra scale applied first (a card's aspect).
    pub fn matrix_scaled(&self, pre: [f32; 3]) -> Mat4 {
        let r = rotation(self.rotate);
        let s = [self.scale[0] * pre[0], self.scale[1] * pre[1], self.scale[2] * pre[2]];
        let mut m = identity();
        for i in 0..3 {
            for j in 0..3 {
                m[i][j] = r[i][j] * s[j];
            }
            m[i][3] = self.translate[i];
        }
        m
    }

    /// World → local, the exact inverse of [`matrix_scaled`](Self::matrix_scaled)
    /// (a zero scale is held at a tiny one rather than dividing by it).
    pub fn inverse_scaled(&self, pre: [f32; 3]) -> Mat4 {
        let r = rotation(self.rotate);
        let s = [self.scale[0] * pre[0], self.scale[1] * pre[1], self.scale[2] * pre[2]]
            .map(|v| if v.abs() < 1e-12 { 1e-12f32.copysign(v) } else { v });
        let mut m = identity();
        // (T R S)^-1 = S^-1 R^T T^-1
        for i in 0..3 {
            for j in 0..3 {
                m[i][j] = r[j][i] / s[i];
            }
            m[i][3] = -(0..3).map(|k| r[k][i] * self.translate[k]).sum::<f32>() / s[i];
        }
        m
    }

    /// A transform back out of a local → world matrix (no shear): what an
    /// importer reads a 4x4 into.
    pub fn from_matrix(m: &Mat4) -> Self {
        let col = |j: usize| [m[0][j], m[1][j], m[2][j]];
        let len = |v: [f32; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        let mut s = [len(col(0)), len(col(1)), len(col(2))];
        let mut r = [[0.0f32; 3]; 3];
        for j in 0..3 {
            let c = col(j);
            let l = s[j].max(1e-12);
            for i in 0..3 {
                r[i][j] = c[i] / l;
            }
        }
        // A mirrored matrix: put the flip in the scale, keep R a rotation.
        let det = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
            - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
            + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
        if det < 0.0 {
            s[2] = -s[2];
            for row in r.iter_mut() {
                row[2] = -row[2];
            }
        }
        Xform { translate: [m[0][3], m[1][3], m[2][3]], rotate: euler_of(&r), scale: s }
    }
}

/// A camera: where it is, and how it sees.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct CameraParams {
    /// Scale is ignored: a camera is a point and a direction.
    pub xform: Xform,
    /// Focal length, mm.
    pub focal: f32,
    /// Filmback (aperture) width and height, mm. The width is fitted to
    /// the render's width; the height is for the record (square pixels).
    pub aperture: [f32; 2],
    pub near: f32,
    pub far: f32,
    pub source: Source,
}

impl Default for CameraParams {
    /// A 50mm on a 36mm back, standing back exactly as far as makes a
    /// default card (one unit wide, at the origin) fill the frame.
    fn default() -> Self {
        Self {
            xform: Xform { translate: [0.0, 0.0, 50.0 / 36.0], ..Default::default() },
            focal: 50.0,
            aperture: [36.0, 24.0],
            near: 0.01,
            far: 10000.0,
            source: Source::Wobble,
        }
    }
}

impl CameraParams {
    /// Focal length in pixels for a picture `width` wide.
    pub fn focal_px(&self, width: f32) -> f32 {
        self.focal.max(1e-3) / self.aperture[0].max(1e-3) * width
    }
}

/// A card: a picture on a plane.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct CardParams {
    pub xform: Xform,
    /// Width over height; 0 takes the image's own.
    pub aspect: f32,
    pub opacity: f32,
    pub source: Source,
}

impl Default for CardParams {
    fn default() -> Self {
        Self { xform: Xform::default(), aspect: 0.0, opacity: 1.0, source: Source::Wobble }
    }
}

/// Gathers cards.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct SceneParams {
    /// Scene ports, each a card or another Scene.
    pub inputs: usize,
}

impl Default for SceneParams {
    fn default() -> Self {
        Self { inputs: 2 }
    }
}

/// Most Scene ports a node offers.
pub const MAX_SCENE_INPUTS: usize = 16;

impl SceneParams {
    pub fn input_count(&self) -> usize {
        self.inputs.clamp(1, MAX_SCENE_INPUTS)
    }
}

macro_rules! orphan_def {
    ($t:ty, $name:expr, $inputs:expr, $tag:expr) => {
        impl crate::nodes::NodeDef for $t {
            const NAME: &'static str = $name;
            const INPUTS: usize = $inputs;
            const MASK_INPUT: Option<usize> = None;
            const VEC4_SLOTS: usize = 0;
            fn topo_tag(&self) -> u64 {
                $tag
            }
            fn pack(&self, _span: &mut [f32]) {}
            fn emit(&self, _e: &mut crate::codegen::Emit) -> String {
                "vec4f(0.0)".into()
            }
        }
    };
}

orphan_def!(CameraParams, "Camera", 0, 0x3D_CA);
orphan_def!(SceneParams, "Blend 3D", 0, 0x3D_5C);

impl crate::nodes::NodeDef for CardParams {
    const NAME: &'static str = "Card";
    const INPUTS: usize = 1;
    const MASK_INPUT: Option<usize> = None;
    const VEC4_SLOTS: usize = 0;
    fn topo_tag(&self) -> u64 {
        0x3D_CD
    }
    fn pack(&self, _span: &mut [f32]) {}
    /// Viewed on its own, a card shows its picture.
    fn emit(&self, e: &mut crate::codegen::Emit) -> String {
        e.input(0)
    }
}

impl Source {
    pub fn is_wobble(&self) -> bool {
        matches!(self, Source::Wobble)
    }
}

type BlenderScenes = std::sync::Mutex<std::collections::HashMap<u64, (String, f32)>>;

fn blender_scenes() -> &'static BlenderScenes {
    static S: std::sync::OnceLock<BlenderScenes> = std::sync::OnceLock::new();
    S.get_or_init(Default::default)
}

/// Where a Blender node's scene export is (`path`, the JSON Blender wrote)
/// and how its frames sit against the comp's (Blender frame = comp frame +
/// `offset`). Set by the app as exports land; read by a `Source::Blender`.
pub fn set_blender_scene(node: u64, path: String, offset: f32) {
    blender_scenes().lock().unwrap().insert(node, (path, offset));
}

/// The export registered for a Blender node.
pub fn blender_scene(node: u64) -> Option<(String, f32)> {
    blender_scenes().lock().unwrap().get(&node).cloned()
}

type TrackerSolves = std::sync::Mutex<std::collections::HashMap<u64, crate::graph::CamTrackParams>>;

fn tracker_solves() -> &'static TrackerSolves {
    static S: std::sync::OnceLock<TrackerSolves> = std::sync::OnceLock::new();
    S.get_or_init(Default::default)
}

/// What a CameraTrack node holds (its solve, scale and lens settings), for
/// a `Source::Tracker` to read. Set by the app whenever the node's solve
/// or scale changes; a node with no solve is removed.
pub fn set_tracker_solve(node: u64, params: Option<crate::graph::CamTrackParams>) {
    let mut m = tracker_solves().lock().unwrap();
    match params {
        Some(p) => {
            m.insert(node, p);
        }
        None => {
            m.remove(&node);
        }
    }
}

/// The CameraTrack registered under `node`, if it has a solve.
pub fn tracker_solve(node: u64) -> Option<crate::graph::CamTrackParams> {
    tracker_solves().lock().unwrap().get(&node).cloned()
}

/// A camera's or card's pose read from its FILE source at comp frame
/// `frame`, written into the node: a camera takes the object's pose and
/// optics, a card keeps its own transform as a child of the object. A node
/// keyed here is untouched; a file that cannot be read leaves the node as
/// it was and says why.
pub fn apply_file_source(kind: &mut NodeKind, frame: f32) -> Result<(), String> {
    use import::TrackKind;
    let (source, want) = match kind {
        NodeKind::Camera3D(p) => (&p.source, TrackKind::Camera),
        NodeKind::Card3D(p) => (&p.source, TrackKind::Xform),
        _ => return Ok(()),
    };
    let (path, object, offset) = match source {
        Source::Wobble => return Ok(()),
        Source::File { path, object, offset } => (path.clone(), object.clone(), *offset),
        Source::Blender { node, object } => {
            let (p, o) = blender_scene(*node).ok_or("waiting for the Blender scene")?;
            (p, object.clone(), o)
        }
        Source::Tracker { node } => {
            let t = tracker_solve(*node).ok_or("the CameraTrack has no solve yet")?;
            let (xf, lens) = t.camera_at(frame).ok_or("the CameraTrack has no solve yet")?;
            let NodeKind::Camera3D(p) = kind else { return Err("a tracker drives a Camera, not a card".into()) };
            p.xform = xf;
            p.focal = lens.focal;
            p.aperture = lens.aperture;
            return Ok(());
        }
    };
    if path.is_empty() {
        return Err("no file picked".into());
    }
    let file = import::load(&path)?;
    let object = &object;
    let track = file
        .track(object, want)
        .or_else(|| if object.is_empty() { file.tracks.first() } else { None })
        .ok_or_else(|| format!("no object named {object:?} in the file"))?;
    let (xf, lens) = track.sample(frame + offset).ok_or("the object has no frames")?;
    match kind {
        NodeKind::Camera3D(p) => {
            p.xform = Xform { scale: [1.0; 3], ..xf };
            if let Some(l) = lens {
                p.focal = l.focal;
                p.aperture = l.aperture;
                p.near = l.near;
                p.far = l.far;
            }
        }
        // A card's own transform rides on the object's, as its child: an
        // imported plane two units wide takes a card scale of 2.
        NodeKind::Card3D(p) => p.xform = Xform::from_matrix(&mat_mul(&xf.matrix(), &p.xform.matrix())),
        _ => {}
    }
    Ok(())
}

/// The cards behind a scene port, in the order they are reached (a port
/// at a time, depth first), each once, at most [`MAX_CARDS`]. Also the
/// count reached, so a panel can say how many were left out.
pub fn cards_of(g: &Graph, render: NodeId) -> (Vec<NodeId>, usize) {
    fn walk(g: &Graph, n: NodeId, seen: &mut Vec<NodeId>, out: &mut Vec<NodeId>) {
        if seen.contains(&n) {
            return;
        }
        seen.push(n);
        match g.nodes.get(&n) {
            Some(NodeKind::Card3D(_)) => out.push(n),
            Some(NodeKind::Scene3D(p)) => {
                for i in 0..p.input_count() {
                    if let Some(s) = g.input_of(n, i) {
                        walk(g, s, seen, out);
                    }
                }
            }
            _ => {}
        }
    }
    let mut out = Vec::new();
    if let Some(s) = g.input_of(render, crate::nodes::render3d::SCENE_PORT) {
        walk(g, s, &mut Vec::new(), &mut out);
    }
    let reached = out.len();
    out.truncate(MAX_CARDS);
    (out, reached)
}

/// The camera on a Render3D's camera port, if one is wired.
pub fn camera_of(g: &Graph, render: NodeId) -> Option<NodeId> {
    let c = g.input_of(render, crate::nodes::render3d::CAMERA_PORT)?;
    matches!(g.nodes.get(&c), Some(NodeKind::Camera3D(_))).then_some(c)
}

/// Lower every Render3D in `g` (see the module docs). `at(node, dt)` is
/// the node `dt` frames from now — fully resolved, its source applied —
/// for motion blur's shutter samples; `None` uses the node as it is in
/// `g`. Run after animation, like the other lowerings.
pub fn expand(g: &mut Graph, at: &dyn Fn(NodeId, f32) -> Option<NodeKind>) {
    use crate::nodes::render3d::{Lowered, LoweredCard};
    let mut renders: Vec<NodeId> = g
        .nodes
        .iter()
        .filter(|(_, k)| matches!(k, NodeKind::Render3D(p) if p.lowered.is_none()))
        .map(|(id, _)| *id)
        .collect();
    if renders.is_empty() {
        return;
    }
    renders.sort_unstable();
    let mut next = g.nodes.keys().copied().max().unwrap_or(0) + 1;
    for r in renders {
        let Some(NodeKind::Render3D(params)) = g.nodes.get(&r).cloned() else { continue };
        let times = params.shutter_times();
        let pose = |id: NodeId, dt: f32| -> Option<NodeKind> {
            if dt == 0.0 {
                g.nodes.get(&id).cloned()
            } else {
                at(id, dt).or_else(|| g.nodes.get(&id).cloned())
            }
        };
        let cams: Vec<CameraParams> = match camera_of(g, r) {
            Some(c) => times
                .iter()
                .filter_map(|dt| match pose(c, *dt) {
                    Some(NodeKind::Camera3D(p)) => Some(p),
                    _ => None,
                })
                .collect(),
            None => Vec::new(),
        };
        let (cards, reached) = cards_of(g, r);
        let mut lowered_cards = Vec::new();
        let mut images = Vec::new();
        for c in &cards {
            // A card with no picture draws nothing: it takes no port.
            let Some(img) = g.input_port(*c, 0) else { continue };
            let poses: Vec<CardParams> = times
                .iter()
                .filter_map(|dt| match pose(*c, *dt) {
                    Some(NodeKind::Card3D(p)) => Some(p),
                    _ => None,
                })
                .collect();
            if poses.len() != times.len() {
                continue;
            }
            lowered_cards.push(LoweredCard { poses });
            images.push(img);
        }
        // The scene wiring goes; the card pictures take ports 1..=n.
        for port in 0..2 {
            g.edges.remove(&(r, port));
            g.edge_out.remove(&(r, port));
        }
        let ok = !cams.is_empty();
        for (i, (src, out)) in images.iter().enumerate() {
            if ok {
                g.connect_out(r, 1 + i, *src, *out);
            }
        }
        let lowered = Lowered {
            cams,
            cards: if ok { lowered_cards } else { Vec::new() },
            reached,
            depth: false,
        };
        if let Some(NodeKind::Render3D(p)) = g.nodes.get_mut(&r) {
            p.lowered = Some(lowered.clone());
        }
        // The depth output is the same render keeping its depth: a node
        // of its own, so the picture's consumers never pay for it.
        let depth_users: Vec<(NodeId, usize)> = g
            .edges
            .iter()
            .filter(|(k, s)| **s == r && g.edge_out.get(k).copied() == Some(1))
            .map(|(k, _)| *k)
            .collect();
        let viewed = g.view == Some(r) && g.view_out == 1;
        if depth_users.is_empty() && !viewed {
            continue;
        }
        let id = next;
        next += 1;
        let mut kid = params.clone();
        kid.lowered = Some(Lowered { depth: true, ..lowered });
        g.nodes.insert(id, NodeKind::Render3D(kid));
        for (i, (src, out)) in images.iter().enumerate() {
            if ok {
                g.connect_out(id, 1 + i, *src, *out);
            }
        }
        g.format_alias.insert(id, r);
        if g.disabled.contains(&r) {
            g.disabled.insert(id);
        }
        for (dst, input) in depth_users {
            g.connect_out(dst, input, id, 0);
        }
        if viewed {
            g.view = Some(id);
            g.view_out = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: &Mat4, b: &Mat4) -> bool {
        a.iter().flatten().zip(b.iter().flatten()).all(|(x, y)| (x - y).abs() < 1e-4)
    }

    #[test]
    fn the_inverse_undoes_the_matrix() {
        let x = Xform { translate: [1.0, -2.0, 3.0], rotate: [20.0, -35.0, 70.0], scale: [2.0, 0.5, 1.5] };
        let pre = [1.0, 0.75, 1.0];
        assert!(close(&mat_mul(&x.inverse_scaled(pre), &x.matrix_scaled(pre)), &identity()));
    }

    #[test]
    fn euler_round_trips_through_a_matrix() {
        let x = Xform { translate: [4.0, 5.0, -6.0], rotate: [10.0, -40.0, 120.0], scale: [1.0, 2.0, 3.0] };
        let back = Xform::from_matrix(&x.matrix());
        assert!(close(&back.matrix(), &x.matrix()), "{back:?}");
    }

    #[test]
    fn rotation_is_x_then_y_then_z() {
        // +Y -> (Rx 90) +Z -> (Ry 90) +X.
        let r = rotation([90.0, 90.0, 0.0]);
        let v = [r[0][1], r[1][1], r[2][1]];
        assert!((v[0] - 1.0).abs() < 1e-5 && v[1].abs() < 1e-5 && v[2].abs() < 1e-5, "{v:?}");
    }

    #[test]
    fn the_default_camera_frames_the_default_card() {
        // The card's right edge (x = 0.5, at the origin) lands on the
        // picture's right edge.
        let cam = CameraParams::default();
        let w = 1920.0;
        let z = cam.xform.translate[2];
        let x_px = 0.5 / z * cam.focal_px(w);
        assert!((x_px - w / 2.0).abs() < 1e-2, "{x_px}");
    }
}
