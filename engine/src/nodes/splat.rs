//! Splat: a 3D Gaussian splat capture (`.ply`, `.spz`) seen through a
//! Camera — the picture, and its depth.
//!
//! **A camera on input 0 drives the view** (a Camera3D, so a CameraTrack's
//! solve, a file's or Blender's too). [`lower`] — after animation and
//! camera sources, next to `scene3d::expand` — copies the camera into the
//! transient [`SplatParams::cam`] and drops the scene wire, so the engine
//! sees a generator. Unwired, the default Camera is the view.
//!
//! **The capture is a LAYER, never topology** (the Plasma / Text
//! precedent): `splatgpu` projects, sorts and draws it into a per-node
//! texture outside the fused pass, and the fused pass samples it. Every
//! edit — the camera, the placement, opacity — is a uniform; `topo_tag`
//! only tells the picture from the depth.
//!
//! **Depth is a second lowered node** (Render3D's pattern): [`lower`] adds
//! a copy with `depth_of` set when output 1 is consumed or viewed, which
//! reads the same layer's depth target; a picture-only comp never draws
//! it. Depth is the coverage-weighted distance along the camera's −Z
//! where the splats are at least half opaque, the far clip elsewhere —
//! Render3D's convention.
//!
//! **Colour**: a capture's colours are what it was trained on, display
//! sRGB. They are blended as trained, then the fused pass decodes the
//! result and takes it to the working primaries; `linear` skips the
//! decode for a capture trained on linear data.

use crate::codegen::Emit;
use crate::graph::{Graph, NodeId, NodeKind};
use crate::nodes::NodeDef;
use crate::scene3d::{mat_mul, CameraParams, Mat4, Xform};
use crate::splat::{Cloud, Native, State};

/// Topology tag, clear of every other node's span.
const TOPO: u64 = 0x5B1A_7000;

/// Channel prefix for the keyable numbers (`splat.<name>` on
/// `Channel::Custom`, the Starfield precedent).
pub const PREFIX: &str = "splat.";

/// The camera port.
pub const CAMERA_INPUT: usize = 0;

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct SplatParams {
    /// The capture: a `.ply` or `.spz`.
    pub path: String,
    /// The node's format (see `inherit`).
    pub size: (u32, u32),
    pub inherit: bool,
    /// Where the capture sits in the scene: degrees X then Y then Z, like
    /// a Card.
    pub translate: [f32; 3],
    pub rotate: [f32; 3],
    pub scale: f32,
    /// Turn a `.ply`'s right-down-forward into wobble's Y up (an `.spz`
    /// is already). Off: the file's numbers as they are.
    pub convert: bool,
    /// Multiplies every splat's opacity.
    pub opacity: f32,
    /// Multiplies every splat's size (a look, not a placement).
    pub splat_size: f32,
    /// View-dependent colour (the SH bands past the first).
    pub sh: bool,
    /// The capture's colours are already scene-linear: no sRGB decode.
    pub linear: bool,
    /// Set by [`lower`]: the wired camera, if any. Never authored.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub cam: Option<CameraParams>,
    /// Set by [`lower`]: the cloud's load generation, so a capture that
    /// finishes loading changes the graph (and the viewer redraws).
    #[cfg_attr(feature = "serde", serde(skip))]
    pub gen: u64,
    /// Set by [`lower`] on the depth copy: the node whose layer it reads.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub depth_of: Option<NodeId>,
    /// Set by [`lower`] on the picture node: a depth copy reads it.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub depth_used: bool,
    /// Set by [`lower`]: Rec.709 to the working primaries.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub to_working: [[f32; 3]; 3],
}

impl Default for SplatParams {
    fn default() -> Self {
        Self {
            path: String::new(),
            size: (1920, 1080),
            inherit: true,
            translate: [0.0; 3],
            rotate: [0.0; 3],
            scale: 1.0,
            convert: true,
            opacity: 1.0,
            splat_size: 1.0,
            sh: true,
            linear: false,
            cam: None,
            gen: 0,
            depth_of: None,
            depth_used: false,
            to_working: crate::color::IDENTITY,
        }
    }
}

const NAMES: &[&str] = &[
    "translate.x",
    "translate.y",
    "translate.z",
    "rotate.x",
    "rotate.y",
    "rotate.z",
    "scale",
    "opacity",
    "size",
];

impl SplatParams {
    /// Every keyable number's name (after [`PREFIX`]).
    pub fn names() -> &'static [&'static str] {
        NAMES
    }

    fn field(&mut self, name: &str) -> Option<&mut f32> {
        Some(match name {
            "translate.x" => &mut self.translate[0],
            "translate.y" => &mut self.translate[1],
            "translate.z" => &mut self.translate[2],
            "rotate.x" => &mut self.rotate[0],
            "rotate.y" => &mut self.rotate[1],
            "rotate.z" => &mut self.rotate[2],
            "scale" => &mut self.scale,
            "opacity" => &mut self.opacity,
            "size" => &mut self.splat_size,
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

    /// The camera this frame: the wired one, else the default.
    pub fn camera(&self) -> CameraParams {
        self.cam.clone().unwrap_or_default()
    }

    /// The node's placement as a transform.
    pub fn xform(&self) -> Xform {
        let s = if self.scale.abs() < 1e-6 { 1e-6f32.copysign(self.scale) } else { self.scale };
        Xform { translate: self.translate, rotate: self.rotate, scale: [s; 3] }
    }

    /// The file's frame into wobble's.
    fn convention(&self, native: Native) -> Mat4 {
        let mut m = crate::scene3d::identity();
        if self.convert && native == Native::Rdf {
            m[1][1] = -1.0;
            m[2][2] = -1.0;
        }
        m
    }

    /// Native → world.
    pub fn model(&self, native: Native) -> Mat4 {
        mat_mul(&self.xform().matrix(), &self.convention(native))
    }

    /// Fit the capture to a unit-sized box at the origin: what a freshly
    /// loaded capture gets, so the default camera sees it.
    pub fn fit(&mut self, cloud: &Cloud) {
        let c = self.convention(cloud.native);
        let ctr = [0, 1, 2].map(|i| (0..3).map(|j| c[i][j] * cloud.center[j]).sum::<f32>());
        let s = 0.5 / cloud.radius.max(1e-6);
        self.rotate = [0.0; 3];
        self.scale = s;
        self.translate = ctr.map(|v| -v * s);
    }

    /// The layer's uniform for a `w` x `h` picture (see `struct U` in
    /// `splatgpu`).
    pub fn uniform(&self, cloud: &Cloud, w: u32, h: u32) -> [u8; crate::splatgpu::UNIFORM_BYTES] {
        let (wf, hf) = (w.max(1) as f32, h.max(1) as f32);
        let cam = self.camera();
        let model = self.model(cloud.native);
        let view = Xform { scale: [1.0; 3], ..cam.xform }.inverse_scaled([1.0; 3]);
        let mv = mat_mul(&view, &model);
        // The camera in the file's frame: C^-1 X^-1 eye (C is its own
        // inverse).
        let inv = mat_mul(&self.convention(cloud.native), &self.xform().inverse_scaled([1.0; 3]));
        let eye = cam.xform.translate;
        let cn = [0, 1, 2].map(|i| (0..3).map(|j| inv[i][j] * eye[j]).sum::<f32>() + inv[i][3]);
        let f = cam.focal_px(wf);
        let near = cam.near.max(1e-4);
        let far = cam.far.max(near * 1.001);
        let mut v = [0f32; 28];
        for r in 0..3 {
            v[r * 4..r * 4 + 4].copy_from_slice(&mv[r]);
        }
        v[12..15].copy_from_slice(&cn);
        v[16..20].copy_from_slice(&[f, 0.5 * wf, 0.5 * hf, near]);
        v[20..24].copy_from_slice(&[wf, hf, far, (far / near).ln()]);
        v[24..28].copy_from_slice(&[
            self.opacity.max(0.0),
            self.splat_size.max(0.0),
            1.3 * 0.5 * wf / f,
            1.3 * 0.5 * hf / f,
        ]);
        let tail = [
            cloud.count as u32,
            if self.sh { cloud.sh_degree } else { 0 },
            cloud.sh_words as u32,
            cloud.antialiased as u32,
        ];
        let mut out = [0u8; crate::splatgpu::UNIFORM_BYTES];
        out[..112].copy_from_slice(bytemuck::cast_slice(&v));
        out[112..].copy_from_slice(bytemuck::cast_slice(&tail));
        out
    }

    /// What the panel shows under the file row.
    pub fn status(&self) -> String {
        match crate::splat::peek(&self.path) {
            _ if self.path.is_empty() => "pick a .ply or .spz capture".into(),
            None | Some((State::Loading, _)) => "loading…".into(),
            Some((State::Failed(e), _)) => e,
            Some((State::Ready(c), _)) => {
                let mut s = format!("{} splats, SH band {}", fmt_count(c.count), c.sh_degree);
                for n in &c.notes {
                    s += "; ";
                    s += n;
                }
                s
            }
        }
    }
}

fn fmt_count(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{:.2}M", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.1}k", n as f64 / 1e3)
    } else {
        n.to_string()
    }
}

/// Copy each Splat's camera onto it, drop the scene wire, ask for its
/// cloud, and split off a depth copy where output 1 is used (see the
/// module docs). Run after animation and camera sources are applied.
pub fn lower(g: &mut Graph) {
    let mut ids: Vec<NodeId> = g
        .nodes
        .iter()
        .filter(|(_, k)| matches!(k, NodeKind::Splat(p) if p.depth_of.is_none()))
        .map(|(id, _)| *id)
        .collect();
    if ids.is_empty() {
        return;
    }
    ids.sort_unstable();
    let to_working = crate::color::primaries_matrix(crate::color::Primaries::Rec709, g.working.primaries);
    let mut next = g.nodes.keys().copied().max().unwrap_or(0) + 1;
    for id in ids {
        let cam = match g.input_of(id, CAMERA_INPUT).and_then(|c| g.nodes.get(&c)) {
            Some(NodeKind::Camera3D(c)) => Some(c.clone()),
            _ => None,
        };
        g.edges.remove(&(id, CAMERA_INPUT));
        g.edge_out.remove(&(id, CAMERA_INPUT));
        let depth_users: Vec<(NodeId, usize)> = g
            .edges
            .iter()
            .filter(|(k, s)| **s == id && g.edge_out.get(k).copied() == Some(1))
            .map(|(k, _)| *k)
            .collect();
        let viewed = g.view == Some(id) && g.view_out == 1;
        let Some(NodeKind::Splat(p)) = g.nodes.get_mut(&id) else { continue };
        p.cam = cam;
        p.gen = crate::splat::request(&p.path).1;
        p.to_working = to_working;
        p.depth_used = !depth_users.is_empty() || viewed;
        if !p.depth_used {
            continue;
        }
        let mut kid = p.clone();
        kid.depth_of = Some(id);
        let kid_id = next;
        next += 1;
        g.nodes.insert(kid_id, NodeKind::Splat(kid));
        g.format_alias.insert(kid_id, id);
        if g.disabled.contains(&id) {
            g.disabled.insert(kid_id);
        }
        for (dst, input) in depth_users {
            g.connect_out(dst, input, kid_id, 0);
        }
        if viewed {
            g.view = Some(kid_id);
            g.view_out = 0;
        }
    }
}

impl NodeDef for SplatParams {
    const NAME: &'static str = "Splat";
    /// The camera, on the scene pipe (lowered away before codegen).
    const INPUTS: usize = 1;
    const MASK_INPUT: Option<usize> = None;
    /// Rec.709 → working rows; decode flag and far clip in the w lanes.
    const VEC4_SLOTS: usize = 3;

    fn topo_tag(&self) -> u64 {
        TOPO ^ self.depth_of.is_some() as u64
    }

    fn pack(&self, v: &mut [f32]) {
        for r in 0..3 {
            v[r * 4..r * 4 + 3].copy_from_slice(&self.to_working[r]);
        }
        v[3] = if self.linear { 0.0 } else { 1.0 };
        v[7] = self.camera().far.max(self.camera().near.max(1e-4) * 1.001);
        v[11] = 0.0;
    }

    fn emit(&self, e: &mut Emit) -> String {
        let slot = e.slot();
        let pos = e.pos().to_string();
        let host = self.depth_of.unwrap_or(e.node());
        let t = e.splat_layer(host, self.depth_of.is_some());
        let (_, fns) = e.color_helpers();
        let f = wgsl(fns);
        let out = e.fresh();
        if self.depth_of.is_some() {
            e.line(format_args!("let {out} = splat_depth(sample_px({t}, {pos}), {slot}u);"));
        } else {
            e.line(format_args!("let {out} = {f}(sample_px({t}, {pos}), {slot}u);"));
        }
        out
    }
}

/// The fused half: the blended capture, decoded and into the working
/// primaries; and the depth target's weighted distance.
fn wgsl(fns: &mut std::collections::BTreeMap<String, String>) -> &'static str {
    const NAME: &str = "splat_px";
    if fns.contains_key(NAME) {
        return NAME;
    }
    fns.insert(
        NAME.into(),
        "fn splat_px(s: vec4f, slot: u32) -> vec4f {
    let a = s.a;
    if (a <= 1e-6) { return vec4f(0.0); }
    var c = max(s.rgb / a, vec3f(0.0));
    if (P.v[slot].w > 0.5) {
        c = select(pow((c + 0.055) / 1.055, vec3f(2.4)), c / 12.92, c <= vec3f(0.04045));
    }
    let m = vec3f(dot(P.v[slot].xyz, c), dot(P.v[slot + 1u].xyz, c), dot(P.v[slot + 2u].xyz, c));
    return vec4f(m * a, a);
}
fn splat_depth(s: vec4f, slot: u32) -> vec4f {
    let far = P.v[slot + 1u].w;
    let d = select(far, s.r / max(s.a, 1e-6), s.a >= 0.5);
    return vec4f(d, d, d, 1.0);
}
"
        .into(),
    );
    NAME
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_edit_is_a_uniform() {
        let a = SplatParams::default();
        let mut b = a.clone();
        b.translate = [1.0, 2.0, 3.0];
        b.opacity = 0.2;
        b.cam = Some(CameraParams { focal: 80.0, ..Default::default() });
        b.path = "x.ply".into();
        assert_eq!(a.topo_tag(), b.topo_tag());
        b.depth_of = Some(4);
        assert_ne!(a.topo_tag(), b.topo_tag());
    }

    #[test]
    fn a_ply_turns_y_up() {
        let p = SplatParams::default();
        let m = p.model(Native::Rdf);
        // File +Y (down) is wobble -Y.
        assert_eq!(m[1][1], -1.0);
        assert_eq!(p.model(Native::Rub)[1][1], 1.0);
    }

    #[test]
    fn a_wired_camera_lands_and_the_wire_goes() {
        let mut g = Graph::default();
        g.nodes.insert(1, NodeKind::Camera3D(CameraParams { focal: 85.0, ..Default::default() }));
        g.nodes.insert(2, NodeKind::Splat(SplatParams::default()));
        g.edges.insert((2, 0), 1);
        lower(&mut g);
        assert!(g.input_of(2, 0).is_none());
        let Some(NodeKind::Splat(p)) = g.nodes.get(&2) else { panic!() };
        assert_eq!(p.camera().focal, 85.0);
        assert!(!p.depth_used);
    }

    #[test]
    fn a_used_depth_output_is_its_own_node() {
        let mut g = Graph::default();
        g.nodes.insert(2, NodeKind::Splat(SplatParams::default()));
        g.nodes.insert(3, NodeKind::Gamma(Default::default()));
        g.connect_out(3, 0, 2, 1);
        lower(&mut g);
        let kid = g.input_of(3, 0).unwrap();
        assert_ne!(kid, 2);
        let Some(NodeKind::Splat(k)) = g.nodes.get(&kid) else { panic!() };
        assert_eq!(k.depth_of, Some(2));
        assert_eq!(g.format_alias.get(&kid), Some(&2));
        let Some(NodeKind::Splat(p)) = g.nodes.get(&2) else { panic!() };
        assert!(p.depth_used);
    }
}
