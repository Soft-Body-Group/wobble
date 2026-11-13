//! 2dini rigs: a skeleton of bones posed by forward kinematics, shapes
//! skinned to it, and a ragdoll that hands the pose to a physics bake.
//!
//! Everything is in geo space (x across the width, y down, in widths). A
//! bone's REST is its head and tail as drawn; its POSE is a rotation about
//! its head, relative to its parent. What flows downstream is, per bone,
//! the rest -> pose affine ([`Affine`]), carried on the geometry
//! (`Geometry::bones`), so a Skin node needs nothing but its two inputs.

use std::sync::Arc;

/// One bone at rest. A child's head is meant to sit on its parent's tail
/// (the viewer keeps it there); parents come before their children.
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Bone {
    #[cfg_attr(feature = "serde", serde(default))]
    pub name: String,
    pub parent: Option<u32>,
    pub head: [f32; 2],
    pub tail: [f32; 2],
}

/// `p' = [a·x + b·y + tx, c·x + d·y + ty]`, as `[a, b, c, d, tx, ty]`.
pub type Affine = [f32; 6];

pub const IDENTITY: Affine = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];

pub fn apply(m: &Affine, p: [f32; 2]) -> [f32; 2] {
    [m[0] * p[0] + m[1] * p[1] + m[4], m[2] * p[0] + m[3] * p[1] + m[5]]
}

/// `a ∘ b`: b first.
pub fn compose(a: &Affine, b: &Affine) -> Affine {
    [
        a[0] * b[0] + a[1] * b[2],
        a[0] * b[1] + a[1] * b[3],
        a[2] * b[0] + a[3] * b[2],
        a[2] * b[1] + a[3] * b[3],
        a[0] * b[4] + a[1] * b[5] + a[4],
        a[2] * b[4] + a[3] * b[5] + a[5],
    ]
}

/// A turn of `r` radians about `c`.
pub fn rot_about(c: [f32; 2], r: f32) -> Affine {
    let (s, k) = r.sin_cos();
    [k, -s, s, k, c[0] - k * c[0] + s * c[1], c[1] - s * c[0] - k * c[1]]
}

/// What flows down a skeleton wire: a bone at rest and where its pose
/// puts it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SkelBone {
    pub head: [f32; 2],
    pub tail: [f32; 2],
    pub parent: Option<u32>,
    /// Rest -> pose.
    pub xf: Affine,
}

/// Forward kinematics: each bone's rest -> pose affine, its own turn about
/// its rest head composed onto its parent's. A parent that comes later in
/// the list (or names itself) is treated as none.
pub fn pose(bones: &[Bone], angles: &[f32]) -> Vec<SkelBone> {
    let mut out: Vec<SkelBone> = Vec::with_capacity(bones.len());
    for (i, b) in bones.iter().enumerate() {
        let own = rot_about(b.head, angles.get(i).copied().unwrap_or(0.0));
        let parent = b.parent.filter(|p| (*p as usize) < i);
        let xf = match parent {
            Some(p) => compose(&out[p as usize].xf, &own),
            None => own,
        };
        out.push(SkelBone { head: b.head, tail: b.tail, parent, xf });
    }
    out
}

/// Distance from `p` to the segment a-b.
fn seg_dist(p: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
    let e = [b[0] - a[0], b[1] - a[1]];
    let w = [p[0] - a[0], p[1] - a[1]];
    let t = ((w[0] * e[0] + w[1] * e[1]) / (e[0] * e[0] + e[1] * e[1]).max(1e-12)).clamp(0.0, 1.0);
    ((w[0] - e[0] * t).powi(2) + (w[1] - e[1] * t).powi(2)).sqrt()
}

/// Skinning weights for a point at rest: rigid = all on the nearest bone;
/// smooth = inverse distance to the power `falloff`, normalised.
pub fn weights(p: [f32; 2], bones: &[SkelBone], smooth: bool, falloff: f32) -> Vec<f32> {
    let d: Vec<f32> = bones.iter().map(|b| seg_dist(p, b.head, b.tail)).collect();
    let mut w = vec![0.0; bones.len()];
    if bones.is_empty() {
        return w;
    }
    let near = d.iter().enumerate().min_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i).unwrap_or(0);
    if !smooth || d[near] < 1e-6 {
        w[near] = 1.0;
        return w;
    }
    let k = falloff.max(0.5);
    let mut sum = 0.0;
    for (i, di) in d.iter().enumerate() {
        w[i] = 1.0 / di.max(1e-5).powf(k);
        sum += w[i];
    }
    for x in &mut w {
        *x /= sum;
    }
    w
}

/// A rest point carried to the pose: the weighted blend of every bone's
/// rest -> pose (linear blend skinning).
pub fn deform(p: [f32; 2], bones: &[SkelBone], w: &[f32]) -> [f32; 2] {
    let mut q = [0.0, 0.0];
    for (b, wi) in bones.iter().zip(w) {
        if *wi == 0.0 {
            continue;
        }
        let r = apply(&b.xf, p);
        q[0] += wi * r[0];
        q[1] += wi * r[1];
    }
    q
}

/// A ragdoll's baked motion: per frame, per bone, (head x, head y, turn)
/// of the body — the bone's rest head carried to there and turned. Shared,
/// like a Physics bake, so graph clones do not copy it.
#[derive(Clone, Debug, Default)]
pub struct RagdollBake {
    pub bones: u32,
    pub frames: u32,
    pub data: Arc<Vec<f32>>,
}

impl PartialEq for RagdollBake {
    fn eq(&self, o: &Self) -> bool {
        self.bones == o.bones && self.frames == o.frames && (Arc::ptr_eq(&self.data, &o.data) || self.data == o.data)
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for RagdollBake {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        (self.bones, self.frames, self.data.as_slice()).serialize(s)
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for RagdollBake {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let (bones, frames, data): (u32, u32, Vec<f32>) = serde::Deserialize::deserialize(d)?;
        Ok(RagdollBake { bones, frames, data: Arc::new(data) })
    }
}

impl RagdollBake {
    pub fn is_empty(&self) -> bool {
        self.frames == 0
    }

    /// The bones' rest -> pose at frame `f` (held at the ends).
    pub fn pose_at(&self, rest: &[SkelBone], f: f32) -> Vec<SkelBone> {
        if self.frames == 0 || self.bones as usize != rest.len() {
            return rest.to_vec();
        }
        let fi = (f.floor().max(0.0) as u32).min(self.frames - 1);
        let base = (fi * self.bones * 3) as usize;
        rest.iter()
            .enumerate()
            .map(|(i, b)| {
                let r = &self.data[base + i * 3..base + i * 3 + 3];
                // Translate the rest head to the body, turned about it.
                let turn = rot_about(b.head, r[2]);
                let mv: Affine = [1.0, 0.0, 0.0, 1.0, r[0] - b.head[0], r[1] - b.head[1]];
                SkelBone { xf: compose(&mv, &turn), ..*b }
            })
            .collect()
    }
}

/// What a ragdoll is simulated with.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RagdollSettings {
    /// m/s², down.
    pub gravity: f32,
    /// How far a joint bends either way from rest, radians.
    pub limit: f32,
    pub frames: u32,
    pub fps: f32,
    /// The canvas edges are walls and floor.
    pub walls: bool,
    /// Height over width.
    pub aspect: f32,
    /// Start spin of the root, radians/s, and start velocity (widths/s).
    pub kick: [f32; 3],
}

/// Simulate: each bone a capsule body, each child pinned to its parent at
/// its head by a hinge limited to `limit`, starting from `start` (the pose
/// the skeleton is in). Returns the bake.
pub fn ragdoll_bake(start: &[SkelBone], s: &RagdollSettings) -> RagdollBake {
    use rapier2d::prelude::*;
    let m = crate::geophys::METRES_PER_WIDTH;
    let mut w = PhysicsWorld::new();
    w.gravity = Vector::new(0.0, s.gravity);
    let sub = 4u32;
    w.integration_parameters.dt = 1.0 / (s.fps.max(1.0) * sub as f32);
    // A bone's body: origin at its rest head, rotated by its pose's turn.
    let mut handles = Vec::new();
    for (i, b) in start.iter().enumerate() {
        let head = apply(&b.xf, b.head);
        let turn = b.xf[2].atan2(b.xf[0]);
        let len = ((b.tail[0] - b.head[0]).powi(2) + (b.tail[1] - b.head[1]).powi(2)).sqrt().max(1e-4);
        let mut rb = RigidBodyBuilder::dynamic().translation(Vector::new(head[0] * m, head[1] * m)).rotation(turn);
        if i == 0 {
            rb = rb.linvel(Vector::new(s.kick[0] * m, s.kick[1] * m)).angvel(s.kick[2]);
        }
        let h = w.bodies.insert(rb.build());
        let a = Vector::new(0.0, 0.0);
        let t = Vector::new((b.tail[0] - b.head[0]) * m, (b.tail[1] - b.head[1]) * m);
        let col = ColliderBuilder::capsule_from_endpoints(a, t, (len * m * 0.12).max(0.02)).friction(0.6).density(1.0);
        w.colliders.insert_with_parent(col.build(), h, &mut w.bodies);
        handles.push(h);
    }
    for (i, b) in start.iter().enumerate() {
        let Some(p) = b.parent.map(|p| p as usize).filter(|p| *p < i) else { continue };
        let pb = &start[p];
        let anchor1 = Vector::new((b.head[0] - pb.head[0]) * m, (b.head[1] - pb.head[1]) * m);
        let lim = s.limit.max(0.0);
        let j = RevoluteJointBuilder::new()
            .local_anchor1(anchor1)
            .local_anchor2(Vector::new(0.0, 0.0))
            .contacts_enabled(false)
            .limits([-lim, lim]);
        w.impulse_joints.insert(handles[p], handles[i], j, true);
    }
    if s.walls {
        let v = |x: f32, y: f32| Vector::new(x * m, y * m);
        let pts = vec![v(0.0, 0.0), v(1.0, 0.0), v(1.0, s.aspect), v(0.0, s.aspect), v(0.0, 0.0)];
        w.colliders.insert(ColliderBuilder::polyline(pts, None).friction(0.8).build());
    }
    let n = start.len() as u32;
    let mut data = Vec::with_capacity((s.frames * n * 3) as usize);
    let record = |w: &PhysicsWorld, data: &mut Vec<f32>| {
        for h in &handles {
            let b = &w.bodies[*h];
            let t = b.translation();
            data.extend_from_slice(&[t.x / m, t.y / m, b.rotation().angle()]);
        }
    };
    record(&w, &mut data);
    for _ in 1..s.frames.max(1) {
        for _ in 0..sub {
            w.step();
        }
        record(&w, &mut data);
    }
    RagdollBake { bones: n, frames: s.frames.max(1), data: Arc::new(data) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arm() -> Vec<Bone> {
        vec![
            Bone { name: "upper".into(), parent: None, head: [0.2, 0.2], tail: [0.4, 0.2] },
            Bone { name: "lower".into(), parent: Some(0), head: [0.4, 0.2], tail: [0.6, 0.2] },
        ]
    }

    /// Turning the root swings the child with it; turning the child moves
    /// only the child, about its own head.
    #[test]
    fn forward_kinematics_carries_children() {
        let b = arm();
        let rest = pose(&b, &[0.0, 0.0]);
        assert_eq!(apply(&rest[1].xf, b[1].tail), b[1].tail);
        let quarter = std::f32::consts::FRAC_PI_2;
        let p = pose(&b, &[quarter, 0.0]);
        let tip = apply(&p[1].xf, b[1].tail);
        assert!((tip[0] - 0.2).abs() < 1e-5 && (tip[1] - 0.6).abs() < 1e-5, "{tip:?}");
        let p = pose(&b, &[0.0, quarter]);
        assert_eq!(apply(&p[0].xf, b[0].tail), b[0].tail);
        let tip = apply(&p[1].xf, b[1].tail);
        assert!((tip[0] - 0.4).abs() < 1e-5 && (tip[1] - 0.4).abs() < 1e-5, "{tip:?}");
    }

    /// Rigid weights follow the nearest bone; smooth ones blend and sum to 1.
    #[test]
    fn skinning_follows_the_bones() {
        let b = arm();
        let p = pose(&b, &[0.0, std::f32::consts::FRAC_PI_2]);
        let w = weights([0.55, 0.21], &p, false, 2.0);
        assert_eq!(w, vec![0.0, 1.0]);
        let q = deform([0.55, 0.21], &p, &w);
        assert!((q[0] - 0.39).abs() < 1e-4 && (q[1] - 0.35).abs() < 1e-4, "{q:?}");
        let w = weights([0.4, 0.25], &p, true, 2.0);
        assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-5 && w[0] > 0.2 && w[1] > 0.2);
    }

    /// A ragdoll falls under gravity, stays joined at its hinge, and its
    /// first frame is the pose it started in.
    #[test]
    fn a_ragdoll_falls_and_stays_joined() {
        let b = arm();
        let start = pose(&b, &[0.0, 0.0]);
        let s = RagdollSettings { gravity: 9.8, limit: 1.0, frames: 30, fps: 24.0, walls: false, aspect: 1.0, kick: [0.0; 3] };
        let bake = ragdoll_bake(&start, &s);
        let f0 = bake.pose_at(&start, 0.0);
        assert!((apply(&f0[1].xf, b[1].tail)[0] - 0.6).abs() < 1e-3);
        let f = bake.pose_at(&start, 29.0);
        let head = apply(&f[0].xf, b[0].head);
        assert!(head[1] > 0.3, "it fell: {head:?}");
        let (elbow0, elbow1) = (apply(&f[0].xf, b[0].tail), apply(&f[1].xf, b[1].head));
        let gap = ((elbow0[0] - elbow1[0]).powi(2) + (elbow0[1] - elbow1[1]).powi(2)).sqrt();
        assert!(gap < 0.01, "the hinge holds: {gap}");
    }
}
