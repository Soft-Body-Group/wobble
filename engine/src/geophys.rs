//! 2dini physics: the Physics node's simulation, run once by an explicit
//! BAKE and then played back per frame.
//!
//! Two solvers behind one node. RIGID drops the shapes through rapier2d:
//! they fall, stack, bounce and topple without deforming. SOFT runs a small
//! XPBD ring solver (squish2d's approach — a ring of particles per body held
//! by edge, bending and area constraints plus shape matching, many substeps
//! of one iteration each) so shapes squash and wobble. Both collide with the
//! same static colliders: closed shapes are solid, open lines are walls a
//! body cannot cross, and the canvas edges can be walls too.
//!
//! A bake is DATA: the bodies' start shapes and styles, then one small record
//! per body per frame (a rigid pose, or a soft body's particle ring). It is
//! small enough to live inside the node, so it saves with the setup and the
//! collaborator who opens it plays the same motion without re-simulating.
//! Nothing here runs per frame except `frame_geometry`, a lookup.
//!
//! Units: geo space (x 0..1 across the width, y in width units, y DOWN).
//! Physical settings are in metres with the canvas `METRES_PER_WIDTH` wide,
//! so a gravity of 9.81 looks like gravity whatever the resolution.

use std::sync::Arc;

use crate::geo::{FillRule, Geometry, Point, Prim, PrimKind, Style};

/// FLUID mode: graav's water, ported (`geofluid.rs`).
#[path = "geofluid.rs"]
pub mod fluid;
pub use fluid::{FluidLook, FluidSettings};

/// How wide the canvas is in metres, for the physical settings.
pub const METRES_PER_WIDTH: f32 = 10.0;
/// Caps, so a stray Scatter cannot turn a bake into an afternoon.
pub const MAX_BODIES: usize = 2048;
pub const MAX_FRAMES: u32 = 2400;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PhysicsMode {
    /// Shapes keep their shape: rapier2d rigid bodies.
    #[default]
    Rigid,
    /// Shapes squash and wobble: an XPBD particle ring per body.
    Soft,
    /// Shapes are water: filled with particles that flow (`fluid`).
    Fluid,
}

impl PhysicsMode {
    pub const ALL: [PhysicsMode; 3] = [PhysicsMode::Rigid, PhysicsMode::Soft, PhysicsMode::Fluid];
    pub fn name(self) -> &'static str {
        match self {
            PhysicsMode::Rigid => "rigid",
            PhysicsMode::Soft => "soft",
            PhysicsMode::Fluid => "fluid",
        }
    }
}

/// Everything a bake is run with. None of it is read per frame: change a
/// setting and the bake is stale until it is run again.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct PhysicsSettings {
    pub mode: PhysicsMode,
    /// Downward, m/s² (the canvas is `METRES_PER_WIDTH` wide).
    pub gravity: f32,
    /// 0 slides, 1 grips.
    pub friction: f32,
    /// Rigid only: 0 dead, 1 bounces back as high as it fell.
    pub bounce: f32,
    /// Rigid only: mass per area, relative between bodies.
    pub density: f32,
    /// Soft only: 0 firm and shape-holding, 1 a loose jiggly blob.
    pub blobbiness: f32,
    /// Solver steps per frame. More is stiffer and steadier, and slower.
    pub substeps: u32,
    /// The canvas edges are walls.
    pub walls: bool,
    /// Draw the collider shapes into the output too.
    pub show_colliders: bool,
    /// The frame the simulation starts at (the bodies' shapes at the bake).
    pub start: i64,
    /// How many frames to simulate.
    pub frames: u32,
    /// Fluid only.
    pub fluid: FluidSettings,
}

impl Default for PhysicsSettings {
    fn default() -> Self {
        Self {
            mode: PhysicsMode::Rigid,
            gravity: 9.81,
            friction: 0.5,
            bounce: 0.2,
            density: 1.0,
            blobbiness: 0.3,
            substeps: 8,
            walls: true,
            show_colliders: true,
            start: 0,
            frames: 120,
            fluid: FluidSettings::default(),
        }
    }
}

/// Which solver a shape's own settings are for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BodyKind {
    #[default]
    Rigid,
    Soft,
}

/// ONE shape's physics, set by a Rigid Body or Soft Body node upstream of
/// the Physics node (Houdini's per-primitive attributes). Where a shape has
/// none, the Physics node's own settings apply. A rigid tag in a soft
/// simulation keeps friction, velocity and spin, and a PINNED shape is a
/// static solid in either; a soft tag in a rigid one keeps the same three.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct PhysAttrs {
    pub kind: BodyKind,
    /// Rigid: mass per area.
    pub density: f32,
    pub friction: f32,
    /// Rigid: restitution.
    pub bounce: f32,
    /// Rigid: never moves — a collider that is also drawn as a body.
    pub pinned: bool,
    /// Soft: 0 firm .. 1 jiggly.
    pub blobbiness: f32,
    /// Soft: the area it wants, as a multiple of its drawn area.
    pub pressure: f32,
    /// Start velocity, m/s (y down).
    pub velocity: [f32; 2],
    /// Start spin, radians/s.
    pub spin: f32,
}

impl Default for PhysAttrs {
    fn default() -> Self {
        Self {
            kind: BodyKind::Rigid,
            density: 1.0,
            friction: 0.5,
            bounce: 0.2,
            pinned: false,
            blobbiness: 0.3,
            pressure: 1.0,
            velocity: [0.0, 0.0],
            spin: 0.0,
        }
    }
}

/// A rigid material: (name, density, friction, bounce).
pub const RIGID_MATERIALS: [(&str, f32, f32, f32); 7] = [
    ("stone", 2.5, 0.8, 0.05),
    ("wood", 0.7, 0.6, 0.25),
    ("metal", 7.8, 0.4, 0.15),
    ("rubber", 1.1, 0.9, 0.8),
    ("ice", 0.9, 0.02, 0.1),
    ("cardboard", 0.2, 0.7, 0.1),
    ("balloon", 0.05, 0.3, 0.6),
];

/// A soft material: (name, blobbiness, pressure, friction).
pub const SOFT_MATERIALS: [(&str, f32, f32, f32); 6] = [
    ("jelly", 0.8, 1.0, 0.4),
    ("water balloon", 0.95, 1.0, 0.15),
    ("dough", 0.6, 0.85, 0.9),
    ("stiff rubber", 0.15, 1.0, 0.7),
    ("slime", 1.0, 0.7, 0.05),
    ("pillow", 0.7, 1.15, 0.6),
];

/// Whole-simulation starting points for the Physics node. Each writes the
/// settings it is about and leaves the frame range alone.
pub struct WorldPreset {
    pub name: &'static str,
    pub apply: fn(&mut PhysicsSettings),
}

pub const WORLD_PRESETS: [WorldPreset; 10] = [
    WorldPreset { name: "falling blocks", apply: |s| {
        s.mode = PhysicsMode::Rigid; s.gravity = 9.81; s.density = 2.5; s.friction = 0.8; s.bounce = 0.05; s.walls = true; s.substeps = 8;
    } },
    WorldPreset { name: "bouncy balls", apply: |s| {
        s.mode = PhysicsMode::Rigid; s.gravity = 9.81; s.density = 1.1; s.friction = 0.9; s.bounce = 0.8; s.walls = true; s.substeps = 8;
    } },
    WorldPreset { name: "ice rink", apply: |s| {
        s.mode = PhysicsMode::Rigid; s.gravity = 9.81; s.density = 0.9; s.friction = 0.02; s.bounce = 0.1; s.walls = true; s.substeps = 8;
    } },
    WorldPreset { name: "moon", apply: |s| {
        s.mode = PhysicsMode::Rigid; s.gravity = 1.62; s.bounce = 0.3; s.walls = true;
    } },
    WorldPreset { name: "zero-g", apply: |s| {
        s.gravity = 0.0; s.walls = true;
    } },
    WorldPreset { name: "jelly pile", apply: |s| {
        s.mode = PhysicsMode::Soft; s.gravity = 9.81; s.blobbiness = 0.8; s.friction = 0.4; s.walls = true; s.substeps = 8;
    } },
    WorldPreset { name: "water balloons", apply: |s| {
        s.mode = PhysicsMode::Soft; s.gravity = 9.81; s.blobbiness = 0.95; s.friction = 0.15; s.walls = true; s.substeps = 12;
    } },
    WorldPreset { name: "water", apply: |s| {
        s.mode = PhysicsMode::Fluid; s.gravity = 9.81; s.friction = 0.3; s.walls = true; s.substeps = 6;
        s.fluid.viscosity = 0.1; s.fluid.iterations = 3;
    } },
    WorldPreset { name: "honey", apply: |s| {
        s.mode = PhysicsMode::Fluid; s.gravity = 9.81; s.friction = 0.9; s.walls = true; s.substeps = 6;
        s.fluid.viscosity = 0.7; s.fluid.iterations = 4;
    } },
    WorldPreset { name: "stiff rubber", apply: |s| {
        s.mode = PhysicsMode::Soft; s.gravity = 9.81; s.blobbiness = 0.15; s.friction = 0.7; s.walls = true; s.substeps = 10;
    } },
];

/// One simulated body as the bake remembers it.
#[derive(Clone, Debug, PartialEq)]
pub struct BakedBody {
    pub shape: BodyShape,
    pub style: Style,
    /// Where this body's record starts inside a frame, in floats.
    pub offset: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub enum BodyShape {
    /// A rigid outline, as offsets from its centre; a frame holds (x, y,
    /// angle). `rings` are where any counters start in `local` (a glyph's
    /// holes ride along; collision uses the outline).
    Rigid { local: Vec<[f32; 2]>, rings: Vec<u32> },
    /// A rigid disc of this radius (geo units); a frame holds (x, y, angle).
    Disc { radius: f32 },
    /// A soft ring of `n` particles; a frame holds all of them, x y x y ...
    Soft { n: u32 },
    /// A water volume of `n` particles `spacing` apart at rest; a frame
    /// holds all of them, x y x y ..., NaN for one that has left.
    Fluid { n: u32, spacing: f32 },
}

impl BodyShape {
    fn floats(&self) -> u32 {
        match self {
            BodyShape::Rigid { .. } | BodyShape::Disc { .. } => 3,
            BodyShape::Soft { n } | BodyShape::Fluid { n, .. } => 2 * n,
        }
    }
}

/// A finished simulation: bodies, then `frames` records of `stride` floats.
#[derive(Clone, Debug, PartialEq)]
pub struct PhysicsBake {
    pub start: i64,
    pub frames: u32,
    pub fps: f32,
    pub stride: u32,
    pub bodies: Vec<BakedBody>,
    pub data: Vec<f32>,
    /// What it was baked FROM (inputs + settings + canvas), so the panel can
    /// say when an edit has made it stale.
    pub input_hash: u64,
    /// Shapes in the input that could not be bodies (open lines).
    pub skipped: u32,
}

impl PhysicsBake {
    /// The frame record for a tick: before the start it is the first frame,
    /// past the end the last — a bake holds its final pose.
    fn frame_of(&self, tick: f32) -> usize {
        let f = (tick.round() as i64 - self.start).clamp(0, self.frames.max(1) as i64 - 1);
        f as usize
    }
}

/// The bodies at a tick, as geometry, each in its own style.
pub fn frame_geometry(b: &PhysicsBake, tick: f32) -> Geometry {
    frame_geometry_look(b, tick, &FluidSettings::default(), 1920.0)
}

/// The same, with a fluid drawn the way its settings say (the look is read
/// here, at playback, so changing it never makes the bake stale).
pub fn frame_geometry_look(b: &PhysicsBake, tick: f32, look: &FluidSettings, px_per_unit: f32) -> Geometry {
    let mut g = Geometry::default();
    if b.frames == 0 || b.data.len() < (b.frames as usize) * b.stride as usize {
        return g;
    }
    let base = b.frame_of(tick) * b.stride as usize;
    let rec = &b.data[base..base + b.stride as usize];
    for (i, body) in b.bodies.iter().enumerate() {
        let o = body.offset as usize;
        let first = g.points.len() as u32;
        match &body.shape {
            BodyShape::Rigid { local, rings } => {
                let (x, y, a) = (rec[o], rec[o + 1], rec[o + 2]);
                let (s, c) = a.sin_cos();
                for q in local {
                    g.points.push(Point {
                        p: [x + c * q[0] - s * q[1], y + s * q[0] + c * q[1]],
                        orient: a,
                        id: i as u32,
                        ..Default::default()
                    });
                }
                let verts = (first..g.points.len() as u32).collect();
                g.prims.push(Prim { layer: Default::default(), kind: PrimKind::Poly { closed: true }, verts, style: body.style, phys: None, src: 0, rings: rings.clone() });
            }
            BodyShape::Disc { radius } => {
                // A disc prim draws at pscale PIXELS; the bake keeps geo
                // units and the pixel size rides in the style's width slot
                // being irrelevant — so keep the disc as a poly ring, which
                // also shows it spinning.
                let (x, y, a) = (rec[o], rec[o + 1], rec[o + 2]);
                let n = 24;
                for k in 0..n {
                    let t = a + std::f32::consts::TAU * k as f32 / n as f32;
                    g.points.push(Point {
                        p: [x + radius * t.cos(), y + radius * t.sin()],
                        orient: a,
                        id: i as u32,
                        ..Default::default()
                    });
                }
                let verts = (first..g.points.len() as u32).collect();
                g.prims.push(Prim { layer: Default::default(), kind: PrimKind::Poly { closed: true }, verts, style: body.style, phys: None, src: 0, rings: Vec::new() });
            }
            BodyShape::Soft { n } => {
                for k in 0..*n as usize {
                    g.points.push(Point {
                        p: [rec[o + 2 * k], rec[o + 2 * k + 1]],
                        id: i as u32,
                        ..Default::default()
                    });
                }
                let verts = (first..g.points.len() as u32).collect();
                g.prims.push(Prim { layer: Default::default(), kind: PrimKind::Poly { closed: true }, verts, style: body.style, phys: None, src: 0, rings: Vec::new() });
            }
            BodyShape::Fluid { n, spacing } => {
                fluid::draw(&rec[o..o + 2 * *n as usize], *spacing, body.style, i as u32, look, px_per_unit, &mut g);
            }
        }
    }
    g
}

// ---------------------------------------------------------------------------
// Inputs -> bodies and colliders.
// ---------------------------------------------------------------------------

/// A body to simulate, in geo space.
#[derive(Clone, Debug)]
enum BodyIn {
    Poly { pts: Vec<[f32; 2]>, holes: Vec<Vec<[f32; 2]>>, style: Style, attrs: Option<PhysAttrs> },
    Disc { c: [f32; 2], r: f32, style: Style, attrs: Option<PhysAttrs> },
}

impl BodyIn {
    fn attrs(&self) -> Option<PhysAttrs> {
        match self {
            BodyIn::Poly { attrs, .. } | BodyIn::Disc { attrs, .. } => *attrs,
        }
    }
    fn pinned(&self) -> bool {
        self.attrs().is_some_and(|a| a.pinned)
    }
    fn outline(&self) -> Vec<[f32; 2]> {
        match self {
            BodyIn::Poly { pts, .. } => pts.clone(),
            BodyIn::Disc { c, r, .. } => (0..24)
                .map(|k| {
                    let a = std::f32::consts::TAU * k as f32 / 24.0;
                    [c[0] + r * a.cos(), c[1] + r * a.sin()]
                })
                .collect(),
        }
    }
}

/// Static collision, in geo space.
#[derive(Clone, Debug, Default)]
struct Statics {
    /// Closed, solid.
    solids: Vec<Vec<[f32; 2]>>,
    /// Open: segments a body cannot cross.
    lines: Vec<Vec<[f32; 2]>>,
    discs: Vec<([f32; 2], f32)>,
}

fn bodies_of(g: &Geometry, px_per_unit: f32) -> (Vec<BodyIn>, u32) {
    let mut out = Vec::new();
    let mut skipped = 0;
    for (i, pr) in g.prims.iter().enumerate() {
        match pr.kind {
            PrimKind::Poly { closed: true } => {
                // The outline: a glyph's counters do not change how it
                // collides.
                let mut rings = g.rings(i).into_iter();
                let pts = rings.next().unwrap_or_default();
                let holes: Vec<Vec<[f32; 2]>> = rings.filter(|r| r.len() >= 3).collect();
                if pts.len() >= 3 && signed_area(&pts).abs() > 1e-9 {
                    out.push(BodyIn::Poly { pts, holes, style: pr.style, attrs: pr.phys });
                } else {
                    skipped += 1;
                }
            }
            PrimKind::Point => {
                if let Some(q) = pr.verts.first().and_then(|&v| g.points.get(v as usize)) {
                    let r = (q.pscale / px_per_unit.max(1.0)).max(1e-4);
                    // A bare point draws as a disc in its stroke/fill; keep it.
                    let mut style = pr.style;
                    if style.fill[3] == 0.0 && style.stroke[3] == 0.0 {
                        style = Style::default();
                    }
                    out.push(BodyIn::Disc { c: q.p, r, style, attrs: pr.phys });
                }
            }
            PrimKind::Poly { closed: false } => skipped += 1,
        }
        if out.len() >= MAX_BODIES {
            break;
        }
    }
    (out, skipped)
}

fn statics_of(g: &Geometry, px_per_unit: f32) -> Statics {
    let mut s = Statics::default();
    for (i, pr) in g.prims.iter().enumerate() {
        match pr.kind {
            PrimKind::Poly { closed: true } => {
                let pts = g.rings(i).into_iter().next().unwrap_or_default();
                if pts.len() >= 3 {
                    s.solids.push(pts);
                }
            }
            PrimKind::Poly { closed: false } => {
                for pts in g.rings(i) {
                    if pts.len() >= 2 {
                        s.lines.push(pts);
                    }
                }
            }
            PrimKind::Point => {
                if let Some(q) = pr.verts.first().and_then(|&v| g.points.get(v as usize)) {
                    s.discs.push((q.p, (q.pscale / px_per_unit.max(1.0)).max(1e-4)));
                }
            }
        }
    }
    s
}

fn signed_area(p: &[[f32; 2]]) -> f32 {
    let n = p.len();
    let mut a = 0.0;
    for i in 0..n {
        let (u, v) = (p[i], p[(i + 1) % n]);
        a += u[0] * v[1] - v[0] * u[1];
    }
    0.5 * a
}

fn centroid(p: &[[f32; 2]]) -> [f32; 2] {
    // Area centroid, so a rigid body turns about its centre of mass.
    let n = p.len();
    let (mut cx, mut cy, mut a2) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let (u, v) = (p[i], p[(i + 1) % n]);
        let c = (u[0] * v[1] - v[0] * u[1]) as f64;
        a2 += c;
        cx += (u[0] + v[0]) as f64 * c;
        cy += (u[1] + v[1]) as f64 * c;
    }
    if a2.abs() < 1e-12 {
        let m = p.iter().fold([0.0f32; 2], |s, q| [s[0] + q[0], s[1] + q[1]]);
        return [m[0] / n as f32, m[1] / n as f32];
    }
    [(cx / (3.0 * a2)) as f32, (cy / (3.0 * a2)) as f32]
}

/// What a bake would be made from, hashed: the bodies and colliders as they
/// stand, the settings and the canvas. The panel compares it with the bake's
/// to say "stale".
pub fn input_hash(bodies: &Geometry, colliders: &Geometry, s: &PhysicsSettings, aspect: f32, fps: f32) -> u64 {
    let mut b: Vec<u8> = Vec::new();
    for g in [bodies, colliders] {
        for (i, pr) in g.prims.iter().enumerate() {
            b.extend_from_slice(&(i as u32).to_le_bytes());
            b.push(match pr.kind {
                PrimKind::Poly { closed: true } => 1,
                PrimKind::Poly { closed: false } => 2,
                PrimKind::Point => 3,
            });
            for q in pr.verts.iter().filter_map(|&v| g.points.get(v as usize)) {
                for f in [q.p[0], q.p[1], q.pscale] {
                    b.extend_from_slice(&f.to_le_bytes());
                }
            }
            for f in pr.style.fill.iter().chain(&pr.style.stroke).chain(std::iter::once(&pr.style.width)) {
                b.extend_from_slice(&f.to_le_bytes());
            }
            if let Some(a) = pr.phys {
                b.push(1 + matches!(a.kind, BodyKind::Soft) as u8 + 2 * a.pinned as u8);
                for f in [a.density, a.friction, a.bounce, a.blobbiness, a.pressure, a.velocity[0], a.velocity[1], a.spin] {
                    b.extend_from_slice(&f.to_le_bytes());
                }
            }
        }
        b.push(0xff);
    }
    for f in [s.gravity, s.friction, s.bounce, s.density, s.blobbiness, aspect, fps] {
        b.extend_from_slice(&f.to_le_bytes());
    }
    b.extend_from_slice(&s.substeps.to_le_bytes());
    b.extend_from_slice(&s.start.to_le_bytes());
    b.extend_from_slice(&s.frames.to_le_bytes());
    b.push(s.walls as u8);
    b.push(matches!(s.mode, PhysicsMode::Soft) as u8);
    // (Only a fluid bake hashes these, so rigid and soft bakes made before
    // the fluid mode existed are not stale.)
    if s.mode == PhysicsMode::Fluid {
        b.extend_from_slice(&s.fluid.hash_bytes());
    }
    crate::diskcache::content_hash(&[&b])
}

/// What a streaming bake hands its sink: the bodies first (a bake with no
/// frames yet), then one record per frame as it is simulated.
pub enum BakeMsg {
    Header(PhysicsBake),
    Frame(Vec<f32>),
}

/// Run the simulation, handing each frame to `sink` as it is made — so a
/// caller can show it while it runs. `sink` returning false stops it.
/// Returns whether every frame was simulated. Deterministic: the same
/// inputs give the same frames.
pub fn bake_with(
    bodies: &Geometry,
    colliders: &Geometry,
    s: &PhysicsSettings,
    aspect: f32,
    px_per_unit: f32,
    fps: f32,
    sink: &mut dyn FnMut(BakeMsg) -> bool,
) -> bool {
    let (ins, skipped) = bodies_of(bodies, px_per_unit);
    let st = statics_of(colliders, px_per_unit);
    let frames = s.frames.clamp(1, MAX_FRAMES);
    let fps = if fps.is_finite() && fps > 0.0 { fps } else { 24.0 };
    let hash = input_hash(bodies, colliders, s, aspect, fps);
    // The solvers build their bodies, then ask for the header around them.
    let mut head = |bodies: Vec<BakedBody>, stride: u32| PhysicsBake {
        start: s.start,
        frames: 0,
        fps,
        stride,
        bodies,
        data: Vec::new(),
        input_hash: hash,
        skipped,
    };
    match s.mode {
        PhysicsMode::Rigid => rigid::run(&ins, &st, s, aspect, frames, fps, &mut head, sink),
        PhysicsMode::Soft => soft::run(&ins, &st, s, aspect, px_per_unit, frames, fps, &mut head, sink),
        PhysicsMode::Fluid => fluid::run(&ins, &st, s, aspect, frames, fps, &mut head, sink),
    }
}

/// Run the whole simulation and return the bake.
pub fn bake(
    bodies: &Geometry,
    colliders: &Geometry,
    s: &PhysicsSettings,
    aspect: f32,
    px_per_unit: f32,
    fps: f32,
) -> PhysicsBake {
    let mut out: Option<PhysicsBake> = None;
    bake_with(bodies, colliders, s, aspect, px_per_unit, fps, &mut |m| {
        match m {
            BakeMsg::Header(h) => out = Some(h),
            BakeMsg::Frame(rec) => {
                if let Some(b) = out.as_mut() {
                    b.data.extend_from_slice(&rec);
                    b.frames += 1;
                }
            }
        }
        true
    });
    out.unwrap_or(PhysicsBake {
        start: s.start,
        frames: 0,
        fps,
        stride: 0,
        bodies: Vec::new(),
        data: Vec::new(),
        input_hash: 0,
        skipped: 0,
    })
}

// ---------------------------------------------------------------------------
// Rigid: rapier2d.
// ---------------------------------------------------------------------------

mod rigid {
    use super::*;
    use rapier2d::prelude::*;

    #[allow(clippy::too_many_arguments)]
    pub(super) fn run(
        ins: &[BodyIn],
        st: &Statics,
        s: &PhysicsSettings,
        aspect: f32,
        frames: u32,
        fps: f32,
        head: &mut dyn FnMut(Vec<BakedBody>, u32) -> PhysicsBake,
        sink: &mut dyn FnMut(BakeMsg) -> bool,
    ) -> bool {
        let m = METRES_PER_WIDTH;
        let v = |p: [f32; 2]| Vector::new(p[0] * m, p[1] * m);
        let mut w = PhysicsWorld::new();
        // Geo space is y-DOWN, so gravity is +y.
        w.gravity = Vector::new(0.0, s.gravity);
        let sub = s.substeps.clamp(1, 64);
        w.integration_parameters.dt = 1.0 / (fps * sub as f32);

        let mut handles = Vec::new();
        let mut baked = Vec::new();
        let mut stride = 0u32;
        for b in ins {
            // The shape's own material where it has one, else the node's.
            let a = b.attrs();
            let fric = a.map_or(s.friction, |a| a.friction).clamp(0.0, 4.0);
            let rest = match a {
                Some(a) if a.kind == BodyKind::Rigid => a.bounce,
                _ => s.bounce,
            }
            .clamp(0.0, 1.0);
            let dens = match a {
                Some(a) if a.kind == BodyKind::Rigid => a.density,
                _ => s.density,
            }
            .max(1e-3);
            let builder = if b.pinned() { RigidBodyBuilder::fixed() } else { RigidBodyBuilder::dynamic() };
            let (vel, spin) = a.map_or(([0.0, 0.0], 0.0), |a| (a.velocity, a.spin));
            let (col, shape, centre) = match b {
                BodyIn::Poly { pts, holes, .. } => {
                    let c = centroid(pts);
                    let local: Vec<[f32; 2]> = pts.iter().map(|p| [p[0] - c[0], p[1] - c[1]]).collect();
                    let verts: Vec<Vector> = local.iter().map(|q| Vector::new(q[0] * m, q[1] * m)).collect();
                    let Some(col) = shape_collider(&verts) else { continue };
                    // Counters ride along with the outline they belong to.
                    let mut all = local;
                    let mut rings = Vec::new();
                    for h in holes {
                        rings.push(all.len() as u32);
                        all.extend(h.iter().map(|p| [p[0] - c[0], p[1] - c[1]]));
                    }
                    (col, BodyShape::Rigid { local: all, rings }, c)
                }
                BodyIn::Disc { c, r, .. } => (ColliderBuilder::ball(r * m), BodyShape::Disc { radius: *r }, *c),
            };
            let body = builder
                .translation(v(centre))
                .linvel(Vector::new(vel[0], vel[1]))
                .angvel(spin)
                .build();
            let h = w.bodies.insert(body);
            let col = col.friction(fric).restitution(rest).density(dens).build();
            w.colliders.insert_with_parent(col, h, &mut w.bodies);
            handles.push(h);
            let off = stride;
            stride += shape.floats();
            let style = match b {
                BodyIn::Poly { style, .. } | BodyIn::Disc { style, .. } => *style,
            };
            baked.push(BakedBody { shape, style, offset: off });
        }
        let (fric, rest) = (s.friction.clamp(0.0, 4.0), s.bounce.clamp(0.0, 1.0));
        // Statics: solid shapes, walls you cannot cross, discs, the canvas.
        for pts in &st.solids {
            let verts: Vec<Vector> = pts.iter().map(|p| v(*p)).collect();
            if let Some(col) = shape_collider(&verts) {
                w.colliders.insert(col.friction(fric).restitution(rest).build());
            }
        }
        for pts in &st.lines {
            let verts: Vec<Vector> = pts.iter().map(|p| v(*p)).collect();
            w.colliders.insert(ColliderBuilder::polyline(verts, None).friction(fric).restitution(rest).build());
        }
        for (c, r) in &st.discs {
            w.colliders.insert(ColliderBuilder::ball(r * m).translation(v(*c)).friction(fric).build());
        }
        if s.walls {
            let box_pts = vec![v([0.0, 0.0]), v([1.0, 0.0]), v([1.0, aspect]), v([0.0, aspect]), v([0.0, 0.0])];
            w.colliders.insert(ColliderBuilder::polyline(box_pts, None).friction(fric).restitution(rest).build());
        }

        if !sink(BakeMsg::Header(head(baked, stride))) {
            return false;
        }
        let record = |w: &PhysicsWorld| {
            let mut rec = Vec::with_capacity(stride as usize);
            for h in &handles {
                let b = &w.bodies[*h];
                let t = b.translation();
                rec.extend_from_slice(&[t.x / m, t.y / m, b.rotation().angle()]);
            }
            rec
        };
        if !sink(BakeMsg::Frame(record(&w))) {
            return false;
        }
        for _ in 1..frames {
            for _ in 0..sub {
                w.step();
            }
            if !sink(BakeMsg::Frame(record(&w))) {
                return false;
            }
        }
        true
    }

    /// A polygon's collider: its hull when convex, else a convex
    /// decomposition. None for a degenerate outline.
    fn shape_collider(verts: &[Vector]) -> Option<ColliderBuilder> {
        if verts.len() < 3 {
            return None;
        }
        if is_convex(verts) {
            return ColliderBuilder::convex_hull(verts);
        }
        let n = verts.len() as u32;
        let idx: Vec<[u32; 2]> = (0..n).map(|i| [i, (i + 1) % n]).collect();
        Some(ColliderBuilder::convex_decomposition(verts, &idx))
    }

    fn is_convex(v: &[Vector]) -> bool {
        let n = v.len();
        let mut sign = 0.0f32;
        for i in 0..n {
            let (a, b, c) = (v[i], v[(i + 1) % n], v[(i + 2) % n]);
            let cross = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x);
            if cross.abs() < 1e-9 {
                continue;
            }
            if sign == 0.0 {
                sign = cross.signum();
            } else if cross.signum() != sign {
                return false;
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Soft: XPBD particle rings (squish2d's approach).
// ---------------------------------------------------------------------------

mod soft {
    use super::*;

    type V = [f32; 2];
    fn sub(a: V, b: V) -> V {
        [a[0] - b[0], a[1] - b[1]]
    }
    fn add(a: V, b: V) -> V {
        [a[0] + b[0], a[1] + b[1]]
    }
    fn mul(a: V, s: f32) -> V {
        [a[0] * s, a[1] * s]
    }
    fn dot(a: V, b: V) -> f32 {
        a[0] * b[0] + a[1] * b[1]
    }
    fn cross(a: V, b: V) -> f32 {
        a[0] * b[1] - a[1] * b[0]
    }
    fn len(a: V) -> f32 {
        dot(a, a).sqrt()
    }

    struct Body {
        start: usize,
        n: usize,
        rest_edge: Vec<f32>,
        rest_bend: Vec<f32>,
        rest_area: f32,
        /// Spawn ring about its centroid: the shape-matching reference.
        rest_local: Vec<V>,
        lo: V,
        hi: V,
        // This body's own material (its PhysAttrs, else the node's).
        edge_c: f32,
        bend_c: f32,
        area_c: f32,
        memory: f32,
        friction: f32,
    }

    /// Compliances and shape memory for a blobbiness 0 (firm) .. 1 (jiggly).
    fn material(t: f32) -> (f32, f32, f32, f32) {
        let t = t.clamp(0.0, 1.0);
        (2e-7 + t * 5e-5, 1e-6 + t * 2e-3, 1e-7 + t * 1e-4, 40.0 * (1.0 - t) * (1.0 - t) + 0.5)
    }

    /// Resample a closed outline to `n` particles evenly spaced along it,
    /// counter-clockwise in the maths sense (positive signed area), so the
    /// area constraint pushes outward.
    fn ring(pts: &[V], n: usize) -> Vec<V> {
        let mut p = pts.to_vec();
        if signed_area(&p) < 0.0 {
            p.reverse();
        }
        let m = p.len();
        let seg: Vec<f32> = (0..m).map(|i| len(sub(p[(i + 1) % m], p[i]))).collect();
        let total: f32 = seg.iter().sum();
        let mut out = Vec::with_capacity(n);
        let (mut i, mut acc) = (0usize, 0.0f32);
        for k in 0..n {
            let target = total * k as f32 / n as f32;
            while i < m - 1 && acc + seg[i] < target {
                acc += seg[i];
                i += 1;
            }
            let t = if seg[i] > 0.0 { ((target - acc) / seg[i]).clamp(0.0, 1.0) } else { 0.0 };
            out.push(add(p[i], mul(sub(p[(i + 1) % m], p[i]), t)));
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn run(
        ins: &[BodyIn],
        st: &Statics,
        s: &PhysicsSettings,
        aspect: f32,
        px_per_unit: f32,
        frames: u32,
        fps: f32,
        head: &mut dyn FnMut(Vec<BakedBody>, u32) -> PhysicsBake,
        sink: &mut dyn FnMut(BakeMsg) -> bool,
    ) -> bool {
        let ppu = px_per_unit.max(1.0);
        // Contact thickness: a couple of pixels.
        let r = 1.5 / ppu;
        let g = s.gravity / METRES_PER_WIDTH;
        // The most a contact moves a particle in one substep.
        let max_push = 4.0 / ppu;

        // Pinned shapes do not move: they are static solids, and a constant
        // pose in the bake so they are still drawn.
        let mut st = st.clone();
        let mut pos: Vec<V> = Vec::new();
        let mut vel: Vec<V> = Vec::new();
        let mut owner: Vec<usize> = Vec::new();
        let mut bodies: Vec<Body> = Vec::new();
        let mut baked = Vec::new();
        let mut pinned_rec: Vec<(u32, [f32; 3])> = Vec::new();
        let mut stride = 0u32;
        for b in ins {
            let a = b.attrs();
            let style = match b {
                BodyIn::Poly { style, .. } | BodyIn::Disc { style, .. } => *style,
            };
            let outline = b.outline();
            if b.pinned() {
                let c = centroid(&outline);
                st.solids.push(outline.clone());
                let local = outline.iter().map(|p| [p[0] - c[0], p[1] - c[1]]).collect();
                let shape = BodyShape::Rigid { local, rings: Vec::new() };
                let off = stride;
                stride += shape.floats();
                pinned_rec.push((off, [c[0], c[1], 0.0]));
                baked.push(BakedBody { shape, style, offset: off });
                continue;
            }
            // The shape's own softness where it is tagged soft, else the node's.
            let (blob, pressure) = match a {
                Some(a) if a.kind == BodyKind::Soft => (a.blobbiness, a.pressure.max(0.05)),
                Some(_) => (0.0, 1.0),
                None => (s.blobbiness, 1.0),
            };
            let (edge_c, bend_c, area_c, memory) = material(blob);
            let friction = a.map_or(s.friction, |a| a.friction).clamp(0.0, 1.0);
            let perim: f32 = (0..outline.len())
                .map(|i| len(sub(outline[(i + 1) % outline.len()], outline[i])))
                .sum();
            let n = ((perim * ppu / 10.0).round() as usize).clamp(12, 64);
            let rp = ring(&outline, n);
            let c = mul(rp.iter().fold([0.0, 0.0], |a, q| add(a, *q)), 1.0 / n as f32);
            let start = pos.len();
            let rest_edge = (0..n).map(|i| len(sub(rp[(i + 1) % n], rp[i]))).collect();
            let rest_bend = (0..n).map(|i| len(sub(rp[(i + 2) % n], rp[i]))).collect();
            let rest_area = signed_area(&rp).abs().max(1e-9) * pressure;
            let rest_local = rp.iter().map(|q| sub(*q, c)).collect();
            // Start velocity (m/s) and spin (rad/s), in geo units per second.
            let (v0, spin) = a.map_or(([0.0, 0.0], 0.0), |a| (a.velocity, a.spin));
            let v0 = mul(v0, 1.0 / METRES_PER_WIDTH);
            for q in &rp {
                let d = sub(*q, c);
                vel.push(add(v0, [-spin * d[1], spin * d[0]]));
                owner.push(bodies.len());
            }
            pos.extend_from_slice(&rp);
            bodies.push(Body {
                start,
                n,
                rest_edge,
                rest_bend,
                rest_area,
                rest_local,
                lo: c,
                hi: c,
                edge_c,
                bend_c,
                area_c,
                memory,
                friction,
            });
            let shape = BodyShape::Soft { n: n as u32 };
            let off = stride;
            stride += shape.floats();
            baked.push(BakedBody { shape, style, offset: off });
        }
        // Where each soft body's particles sit in a frame record.
        let soft_offsets: Vec<u32> = baked
            .iter()
            .filter(|b| matches!(b.shape, BodyShape::Soft { .. }))
            .map(|b| b.offset)
            .collect();
        if !sink(BakeMsg::Header(head(baked, stride))) {
            return false;
        }
        let np = pos.len();
        let mut prev = pos.clone();
        let mut touched: Vec<Option<V>> = vec![None; np];

        let sub_n = (s.substeps.clamp(1, 64) * 2).max(4);
        let h = 1.0 / (fps * sub_n as f32);
        let record = |pos: &[V], bodies: &[Body]| {
            let mut rec = vec![0.0f32; stride as usize];
            for (off, pose) in &pinned_rec {
                rec[*off as usize..*off as usize + 3].copy_from_slice(pose);
            }
            for (b, off) in bodies.iter().zip(&soft_offsets) {
                for k in 0..b.n {
                    let q = pos[b.start + k];
                    rec[*off as usize + 2 * k] = q[0];
                    rec[*off as usize + 2 * k + 1] = q[1];
                }
            }
            rec
        };
        if !sink(BakeMsg::Frame(record(&pos, &bodies))) {
            return false;
        }
        for _ in 1..frames {
            for _ in 0..sub_n {
                // Integrate.
                for i in 0..np {
                    vel[i][1] += g * h;
                    let damp = (-0.2 * h).exp();
                    vel[i] = mul(vel[i], damp);
                    prev[i] = pos[i];
                    pos[i] = add(pos[i], mul(vel[i], h));
                    touched[i] = None;
                }
                // Each body's own constraints.
                for b in &bodies {
                    solve_body(b, &mut pos, h);
                }
                // Bounds for the broadphase.
                for b in bodies.iter_mut() {
                    let (mut lo, mut hi) = ([f32::MAX; 2], [f32::MIN; 2]);
                    for q in &pos[b.start..b.start + b.n] {
                        lo = [lo[0].min(q[0]), lo[1].min(q[1])];
                        hi = [hi[0].max(q[0]), hi[1].max(q[1])];
                    }
                    b.lo = lo;
                    b.hi = hi;
                }
                // Body against body: a particle inside (or within r of)
                // another ring is pushed out to its nearest edge, the edge
                // taking its share.
                for a in 0..bodies.len() {
                    for bi in 0..bodies.len() {
                        if a == bi {
                            continue;
                        }
                        let (ba, bb) = (&bodies[a], &bodies[bi]);
                        if ba.hi[0] + r < bb.lo[0]
                            || bb.hi[0] + r < ba.lo[0]
                            || ba.hi[1] + r < bb.lo[1]
                            || bb.hi[1] + r < ba.lo[1]
                        {
                            continue;
                        }
                        let (sa, na, sb, nb) = (ba.start, ba.n, bb.start, bb.n);
                        for i in sa..sa + na {
                            let p = pos[i];
                            let ring_b: Vec<V> = pos[sb..sb + nb].to_vec();
                            if let Some((k, t, q, inside)) = nearest_on_ring(&ring_b, p) {
                                let d = sub(p, q);
                                let dist = len(d);
                                let pen = if inside { dist + r } else { r - dist };
                                if pen <= 0.0 {
                                    continue;
                                }
                                // Deep overlaps (shapes spawned inside each
                                // other) come apart over a few substeps, not
                                // in one shove that becomes a launch.
                                let pen = pen.min(max_push);
                                let nrm = if dist > 1e-9 {
                                    mul(d, if inside { -1.0 / dist } else { 1.0 / dist })
                                } else {
                                    edge_normal(&ring_b, k)
                                };
                                let corr = mul(nrm, pen);
                                // Particle 1, edge ends (1-t) and t.
                                let (w0, w1) = (1.0 - t, t);
                                let wsum = 1.0 + w0 * w0 + w1 * w1;
                                pos[i] = add(pos[i], mul(corr, 1.0 / wsum));
                                let e0 = sb + k;
                                let e1 = sb + (k + 1) % nb;
                                pos[e0] = sub(pos[e0], mul(corr, w0 / wsum));
                                pos[e1] = sub(pos[e1], mul(corr, w1 / wsum));
                                touched[i] = Some(nrm);
                            }
                        }
                    }
                }
                // Statics.
                for i in 0..np {
                    for solid in &st.solids {
                        if let Some((k, _, q, inside)) = nearest_on_ring(solid, pos[i]) {
                            let d = sub(pos[i], q);
                            let dist = len(d);
                            let pen = if inside { dist + r } else { r - dist };
                            if pen > 0.0 {
                                let nrm = if dist > 1e-9 {
                                    mul(d, if inside { -1.0 / dist } else { 1.0 / dist })
                                } else {
                                    edge_normal(solid, k)
                                };
                                pos[i] = add(pos[i], mul(nrm, pen));
                                touched[i] = Some(nrm);
                            }
                        }
                    }
                    for line in &st.lines {
                        if let Some(nrm) = line_contact(line, prev[i], &mut pos[i], r) {
                            touched[i] = Some(nrm);
                        }
                    }
                    for (c, rad) in &st.discs {
                        let d = sub(pos[i], *c);
                        let dist = len(d);
                        let pen = rad + r - dist;
                        if pen > 0.0 && dist > 1e-9 {
                            let nrm = mul(d, 1.0 / dist);
                            pos[i] = add(pos[i], mul(nrm, pen));
                            touched[i] = Some(nrm);
                        }
                    }
                    if s.walls {
                        let lim = [(r, 1.0 - r), (r, aspect - r)];
                        for ax in 0..2 {
                            if pos[i][ax] < lim[ax].0 {
                                pos[i][ax] = lim[ax].0;
                                let mut nrm = [0.0, 0.0];
                                nrm[ax] = 1.0;
                                touched[i] = Some(nrm);
                            } else if pos[i][ax] > lim[ax].1 {
                                pos[i][ax] = lim[ax].1;
                                let mut nrm = [0.0, 0.0];
                                nrm[ax] = -1.0;
                                touched[i] = Some(nrm);
                            }
                        }
                    }
                }
                // Friction at the position level, then velocities. A contact
                // push-out is not a bounce: most of the outward speed a
                // correction created is taken back (position-based contacts
                // otherwise pump energy into bodies pressed together).
                for i in 0..np {
                    if let Some(nrm) = touched[i] {
                        let dp = sub(pos[i], prev[i]);
                        let tang = sub(dp, mul(nrm, dot(dp, nrm)));
                        pos[i] = sub(pos[i], mul(tang, bodies[owner[i]].friction));
                    }
                    vel[i] = mul(sub(pos[i], prev[i]), 1.0 / h);
                    if let Some(nrm) = touched[i] {
                        let vn = dot(vel[i], nrm);
                        if vn > 0.0 {
                            vel[i] = sub(vel[i], mul(nrm, vn * 0.9));
                        }
                    }
                }
            }
            if !sink(BakeMsg::Frame(record(&pos, &bodies))) {
                return false;
            }
        }
        true
    }

    fn solve_body(b: &Body, pos: &mut [V], h: f32) {
        let (edge_c, bend_c, area_c, memory) = (b.edge_c, b.bend_c, b.area_c, b.memory);
        let (s, n) = (b.start, b.n);
        let h2 = h * h;
        // Edge and bend lengths (normalised, so compliance is scale-free).
        for (dist_rest, step, comp) in [(&b.rest_edge, 1usize, edge_c), (&b.rest_bend, 2usize, bend_c)] {
            for i in 0..n {
                let (a, c) = (s + i, s + (i + step) % n);
                let d = sub(pos[c], pos[a]);
                let l = len(d);
                let l0 = dist_rest[i].max(1e-9);
                if l < 1e-12 {
                    continue;
                }
                let cval = (l - l0) / l0;
                let grad = mul(d, 1.0 / (l * l0));
                let wsum = 2.0 * dot(grad, grad);
                let lam = -cval / (wsum + comp / h2);
                pos[a] = sub(pos[a], mul(grad, lam));
                pos[c] = add(pos[c], mul(grad, lam));
            }
        }
        // Area (pressure), normalised by the rest area.
        let ring = &pos[s..s + n];
        let area = signed_area(ring);
        let cval = (area - b.rest_area) / b.rest_area;
        let mut grads = Vec::with_capacity(n);
        let mut wsum = 0.0;
        for i in 0..n {
            let (pp, pn) = (ring[(i + n - 1) % n], ring[(i + 1) % n]);
            let gi = mul([pn[1] - pp[1], pp[0] - pn[0]], 0.5 / b.rest_area);
            wsum += dot(gi, gi);
            grads.push(gi);
        }
        let lam = -cval / (wsum + area_c / h2);
        for i in 0..n {
            pos[s + i] = add(pos[s + i], mul(grads[i], lam));
        }
        // Shape matching: pull toward the best-fit rotated rest shape.
        let ring = &pos[s..s + n];
        let c = mul(ring.iter().fold([0.0, 0.0], |a, q| add(a, *q)), 1.0 / n as f32);
        let (mut sd, mut sc) = (0.0f32, 0.0f32);
        for i in 0..n {
            let q = sub(ring[i], c);
            sd += cross(b.rest_local[i], q);
            sc += dot(b.rest_local[i], q);
        }
        let a = sd.atan2(sc);
        let (sn, cs) = a.sin_cos();
        let k = (memory * h).min(1.0);
        for i in 0..n {
            let r = b.rest_local[i];
            let goal = add(c, [cs * r[0] - sn * r[1], sn * r[0] + cs * r[1]]);
            let p = pos[s + i];
            pos[s + i] = add(p, mul(sub(goal, p), k));
        }
    }

    /// Nearest point on a closed ring: (edge index, t along it, the point,
    /// is `p` inside the ring).
    fn nearest_on_ring(ring: &[V], p: V) -> Option<(usize, f32, V, bool)> {
        let n = ring.len();
        if n < 3 {
            return None;
        }
        let (mut best, mut bk, mut bt, mut bq) = (f32::MAX, 0, 0.0, p);
        let mut wn = 0i32;
        for k in 0..n {
            let (a, b) = (ring[k], ring[(k + 1) % n]);
            let e = sub(b, a);
            let t = (dot(sub(p, a), e) / dot(e, e).max(1e-18)).clamp(0.0, 1.0);
            let q = add(a, mul(e, t));
            let d = dot(sub(p, q), sub(p, q));
            if d < best {
                best = d;
                bk = k;
                bt = t;
                bq = q;
            }
            if (a[1] <= p[1]) != (b[1] <= p[1]) {
                let s = cross(e, sub(p, a));
                if b[1] > a[1] && s > 0.0 {
                    wn += 1;
                } else if b[1] <= a[1] && s < 0.0 {
                    wn -= 1;
                }
            }
        }
        Some((bk, bt, bq, wn != 0))
    }

    fn edge_normal(ring: &[V], k: usize) -> V {
        let n = ring.len();
        let e = sub(ring[(k + 1) % n], ring[k]);
        let l = len(e).max(1e-12);
        // Outward for a positive-area ring.
        [e[1] / l, -e[0] / l]
    }

    /// Keep a particle off an open line: pushed out if within `r`, and put
    /// back on its own side if it crossed the line this substep.
    fn line_contact(line: &[V], prev: V, p: &mut V, r: f32) -> Option<V> {
        let mut hit = None;
        for w in line.windows(2) {
            let (a, b) = (w[0], w[1]);
            let e = sub(b, a);
            let ee = dot(e, e);
            if ee < 1e-18 {
                continue;
            }
            // Which side the particle came from.
            let side0 = cross(e, sub(prev, a));
            let side1 = cross(e, sub(*p, a));
            let t = (dot(sub(*p, a), e) / ee).clamp(0.0, 1.0);
            let q = add(a, mul(e, t));
            let el = ee.sqrt();
            let nrm0 = mul([-e[1], e[0]], 1.0 / el);
            let nrm = if side0 >= 0.0 { nrm0 } else { mul(nrm0, -1.0) };
            let crossed = side0.signum() != side1.signum() && side0 != 0.0 && {
                let tp = (dot(sub(prev, a), e) / ee).clamp(0.0, 1.0);
                (0.0..=1.0).contains(&t) || (0.0..=1.0).contains(&tp)
            };
            let dist = dot(sub(*p, q), nrm);
            if crossed || (dist < r && len(sub(*p, q)) < r + 1e-6) {
                *p = add(q, mul(nrm, r));
                hit = Some(nrm);
            }
        }
        hit
    }
}

// ---------------------------------------------------------------------------
// Keeping a bake inside the node.
// ---------------------------------------------------------------------------

/// The bake a Physics node holds: shared, so the undo stack and the memo
/// keys cloning the graph do not copy it, and compared by content hash.
#[derive(Clone, Default)]
pub struct BakeSlot(pub Option<Arc<PhysicsBake>>);

impl BakeSlot {
    pub fn get(&self) -> Option<&PhysicsBake> {
        self.0.as_deref()
    }
    fn key(&self) -> Option<(u64, usize)> {
        self.0.as_ref().map(|b| (b.input_hash, b.data.len()))
    }
}

impl PartialEq for BakeSlot {
    fn eq(&self, other: &Self) -> bool {
        match (&self.0, &other.0) {
            (Some(a), Some(b)) => Arc::ptr_eq(a, b) || (self.key() == other.key() && a == b),
            (None, None) => true,
            _ => false,
        }
    }
}

impl std::fmt::Debug for BakeSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.0 {
            Some(b) => write!(f, "Bake({} bodies, {} frames)", b.bodies.len(), b.frames),
            None => write!(f, "NoBake"),
        }
    }
}

/// v2 adds a rigid body's counter rings; a v1 bake still reads.
const MAGIC: &[u8; 4] = b"WPB2";
const MAGIC_V1: &[u8; 4] = b"WPB1";

impl PhysicsBake {
    /// Little-endian bytes: header, bodies, frame data.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(64 + self.data.len() * 4);
        b.extend_from_slice(MAGIC);
        b.extend_from_slice(&self.start.to_le_bytes());
        for u in [self.frames, self.stride, self.skipped, self.bodies.len() as u32] {
            b.extend_from_slice(&u.to_le_bytes());
        }
        b.extend_from_slice(&self.fps.to_le_bytes());
        b.extend_from_slice(&self.input_hash.to_le_bytes());
        for body in &self.bodies {
            for f in body.style.fill.iter().chain(&body.style.stroke) {
                b.extend_from_slice(&f.to_le_bytes());
            }
            b.extend_from_slice(&body.style.width.to_le_bytes());
            b.push(matches!(body.style.rule, FillRule::EvenOdd) as u8);
            b.extend_from_slice(&body.offset.to_le_bytes());
            match &body.shape {
                BodyShape::Rigid { local, rings } => {
                    b.push(0);
                    b.extend_from_slice(&(local.len() as u32).to_le_bytes());
                    for q in local {
                        b.extend_from_slice(&q[0].to_le_bytes());
                        b.extend_from_slice(&q[1].to_le_bytes());
                    }
                    b.extend_from_slice(&(rings.len() as u32).to_le_bytes());
                    for r in rings {
                        b.extend_from_slice(&r.to_le_bytes());
                    }
                }
                BodyShape::Disc { radius } => {
                    b.push(1);
                    b.extend_from_slice(&radius.to_le_bytes());
                }
                BodyShape::Soft { n } => {
                    b.push(2);
                    b.extend_from_slice(&n.to_le_bytes());
                }
                BodyShape::Fluid { n, spacing } => {
                    b.push(3);
                    b.extend_from_slice(&n.to_le_bytes());
                    b.extend_from_slice(&spacing.to_le_bytes());
                }
            }
        }
        b.extend_from_slice(&(self.data.len() as u32).to_le_bytes());
        for f in &self.data {
            b.extend_from_slice(&f.to_le_bytes());
        }
        b
    }

    /// The inverse; None for anything malformed (a bake that does not read
    /// back is simply not baked).
    pub fn from_bytes(b: &[u8]) -> Option<Self> {
        let mut r = Reader { b, i: 0 };
        let magic = r.take(4)?;
        let v2 = magic == MAGIC;
        if !v2 && magic != MAGIC_V1 {
            return None;
        }
        let start = i64::from_le_bytes(r.take(8)?.try_into().ok()?);
        let (frames, stride, skipped, nb) = (r.u32()?, r.u32()?, r.u32()?, r.u32()?);
        let fps = r.f32()?;
        let input_hash = u64::from_le_bytes(r.take(8)?.try_into().ok()?);
        if nb as usize > MAX_BODIES || frames > MAX_FRAMES {
            return None;
        }
        let mut bodies = Vec::with_capacity(nb as usize);
        for _ in 0..nb {
            let mut st = Style::default();
            for k in 0..4 {
                st.fill[k] = r.f32()?;
            }
            for k in 0..4 {
                st.stroke[k] = r.f32()?;
            }
            st.width = r.f32()?;
            st.rule = if r.take(1)?[0] == 1 { FillRule::EvenOdd } else { FillRule::NonZero };
            let offset = r.u32()?;
            let shape = match r.take(1)?[0] {
                0 => {
                    let n = r.u32()? as usize;
                    if n > 1 << 16 {
                        return None;
                    }
                    let mut local = Vec::with_capacity(n);
                    for _ in 0..n {
                        local.push([r.f32()?, r.f32()?]);
                    }
                    let mut rings = Vec::new();
                    if v2 {
                        let k = r.u32()? as usize;
                        if k > n {
                            return None;
                        }
                        for _ in 0..k {
                            rings.push(r.u32()?);
                        }
                    }
                    BodyShape::Rigid { local, rings }
                }
                1 => BodyShape::Disc { radius: r.f32()? },
                2 => BodyShape::Soft { n: r.u32()? },
                3 => BodyShape::Fluid { n: r.u32()?, spacing: r.f32()? },
                _ => return None,
            };
            bodies.push(BakedBody { shape, style: st, offset });
        }
        let nd = r.u32()? as usize;
        if nd != frames as usize * stride as usize {
            return None;
        }
        let mut data = Vec::with_capacity(nd);
        for _ in 0..nd {
            data.push(r.f32()?);
        }
        Some(PhysicsBake { start, frames, fps, stride, bodies, data, input_hash, skipped })
    }
}

struct Reader<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let s = self.b.get(self.i..self.i + n)?;
        self.i += n;
        Some(s)
    }
    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }
    fn f32(&mut self) -> Option<f32> {
        Some(f32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }
}

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

pub(crate) fn b64_encode(b: &[u8]) -> String {
    let mut s = String::with_capacity(b.len().div_ceil(3) * 4);
    for c in b.chunks(3) {
        let n = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
        for k in 0..4 {
            if k <= c.len() {
                s.push(B64[(n >> (18 - 6 * k) & 63) as usize] as char);
            } else {
                s.push('=');
            }
        }
    }
    s
}

pub(crate) fn b64_decode(s: &str) -> Option<Vec<u8>> {
    let val = |c: u8| B64.iter().position(|&x| x == c).map(|v| v as u32);
    let bytes: Vec<u8> = s.bytes().filter(|c| !c.is_ascii_whitespace()).collect();
    if bytes.len() % 4 != 0 {
        return None;
    }
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    for c in bytes.chunks(4) {
        let pad = c.iter().filter(|&&x| x == b'=').count();
        let mut n = 0u32;
        for (k, &ch) in c.iter().enumerate() {
            let v = if ch == b'=' { 0 } else { val(ch)? };
            n |= v << (18 - 6 * k);
        }
        let got = [(n >> 16) as u8, (n >> 8) as u8, n as u8];
        out.extend_from_slice(&got[..3 - pad]);
    }
    Some(out)
}

/// In a setup the bake is one string: lz4-compressed bytes, base64.
#[cfg(feature = "serde")]
impl serde::Serialize for BakeSlot {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match &self.0 {
            Some(b) => {
                let z = lz4_flex::compress_prepend_size(&b.to_bytes());
                s.serialize_some(&b64_encode(&z))
            }
            None => s.serialize_none(),
        }
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for BakeSlot {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s: Option<String> = serde::Deserialize::deserialize(d)?;
        Ok(BakeSlot(s.and_then(|s| {
            let z = b64_decode(&s)?;
            let raw = lz4_flex::decompress_size_prepended(&z).ok()?;
            PhysicsBake::from_bytes(&raw).map(Arc::new)
        })))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(c: [f32; 2], h: f32) -> Vec<[f32; 2]> {
        vec![[c[0] - h, c[1] - h], [c[0] + h, c[1] - h], [c[0] + h, c[1] + h], [c[0] - h, c[1] + h]]
    }

    fn geo_of(polys: &[Vec<[f32; 2]>], closed: bool) -> Geometry {
        let mut g = Geometry::default();
        for pts in polys {
            let first = g.points.len() as u32;
            for p in pts {
                g.points.push(Point::at(*p));
            }
            g.prims.push(Prim {
                layer: Default::default(),
                kind: PrimKind::Poly { closed },
                verts: (first..g.points.len() as u32).collect(),
                style: Style::default(),
                phys: None, src: 0, rings: Vec::new() });
        }
        g
    }

    fn lowest(g: &Geometry) -> f32 {
        g.points.iter().map(|q| q.p[1]).fold(f32::MIN, f32::max)
    }

    /// Both solvers: a square dropped onto a ground line falls, stops ON the
    /// line (not through it), and the bake plays back what it simulated.
    #[test]
    fn a_dropped_square_lands_on_the_ground_in_both_modes() {
        let bodies = geo_of(&[square([0.5, 0.1], 0.04)], true);
        let ground = geo_of(&[vec![[0.0, 0.4], [1.0, 0.4]]], false);
        for mode in [PhysicsMode::Rigid, PhysicsMode::Soft] {
            let s = PhysicsSettings { mode, frames: 72, walls: false, ..Default::default() };
            let b = bake(&bodies, &ground, &s, 0.5625, 1920.0, 24.0);
            assert_eq!(b.frames, 72);
            assert_eq!(b.bodies.len(), 1, "{mode:?}");
            let first = frame_geometry(&b, 0.0);
            let last = frame_geometry(&b, 71.0);
            assert!((lowest(&first) - 0.14).abs() < 0.01, "{mode:?} starts where it was drawn: {}", lowest(&first));
            let rest = lowest(&last);
            assert!(rest > 0.35, "{mode:?} did not fall: {rest}");
            assert!(rest < 0.41, "{mode:?} went through the ground: {rest}");
            // Before the start it is the first frame, past the end the last.
            assert_eq!(frame_geometry(&b, -10.0), first);
            assert_eq!(frame_geometry(&b, 500.0), last);
        }
    }

    /// Deterministic: the same inputs bake the same numbers.
    #[test]
    fn a_bake_is_deterministic() {
        let bodies = geo_of(&[square([0.3, 0.1], 0.03), square([0.32, 0.2], 0.03)], true);
        for mode in PhysicsMode::ALL {
            let s = PhysicsSettings { mode, frames: 30, ..Default::default() };
            let a = bake(&bodies, &Geometry::default(), &s, 0.5625, 1920.0, 24.0);
            let b = bake(&bodies, &Geometry::default(), &s, 0.5625, 1920.0, 24.0);
            assert_eq!(a, b, "{mode:?}");
        }
    }

    /// Rigid bodies stack; soft ones squash. Two squares dropped onto the
    /// floor one above the other: rigid keeps each its size, soft flattens
    /// the lower one under the upper when blobby.
    #[test]
    fn rigid_keeps_its_shape_and_soft_gives() {
        let bodies = geo_of(&[square([0.5, 0.40], 0.05), square([0.5, 0.25], 0.05)], true);
        let rigid = bake(&bodies, &Geometry::default(), &PhysicsSettings { frames: 96, ..Default::default() }, 0.5625, 1920.0, 24.0);
        let g = frame_geometry(&rigid, 95.0);
        for i in 0..2 {
            let pl = g.polyline(i);
            let d = ((pl[0][0] - pl[2][0]).powi(2) + (pl[0][1] - pl[2][1]).powi(2)).sqrt();
            assert!((d - 0.1414).abs() < 1e-3, "a rigid square changed size: diagonal {d}");
        }
        // Resting on the floor (canvas bottom 0.5625), stacked, not merged.
        assert!(lowest(&g) > 0.55 && lowest(&g) < 0.5626, "floor contact {}", lowest(&g));
        let tops: Vec<f32> = (0..2).map(|i| g.polyline(i).iter().map(|q| q[1]).fold(f32::MAX, f32::min)).collect();
        assert!((tops[0] - tops[1]).abs() > 0.08, "the squares did not stack: {tops:?}");

        let soft = bake(
            &bodies,
            &Geometry::default(),
            &PhysicsSettings { mode: PhysicsMode::Soft, blobbiness: 0.9, frames: 96, ..Default::default() },
            0.5625,
            1920.0,
            24.0,
        );
        let g = frame_geometry(&soft, 95.0);
        assert!(lowest(&g) <= 0.5626, "a soft body left the canvas: {}", lowest(&g));
        let span = |i: usize| {
            let pl = g.polyline(i);
            let (lo, hi) = pl.iter().fold((f32::MAX, f32::MIN), |(l, h), q| (l.min(q[0]), h.max(q[0])));
            hi - lo
        };
        assert!(span(0) > 0.105, "a blobby body did not squash: width {}", span(0));
    }

    /// Discs roll; a static solid is solid; open input shapes are skipped.
    #[test]
    fn discs_solids_and_skipped_lines() {
        let mut bodies = geo_of(&[vec![[0.1, 0.1], [0.2, 0.1]]], false);
        bodies.points.push(Point { p: [0.5, 0.1], pscale: 20.0, ..Default::default() });
        bodies.prims.push(Prim { layer: Default::default(), kind: PrimKind::Point, verts: vec![2], style: Style::default(), phys: None, src: 0, rings: Vec::new() });
        let block = geo_of(&[square([0.5, 0.35], 0.05)], true);
        for mode in [PhysicsMode::Rigid, PhysicsMode::Soft] {
            let b = bake(&bodies, &block, &PhysicsSettings { mode, frames: 60, walls: false, ..Default::default() }, 0.5625, 1000.0, 24.0);
            assert_eq!(b.skipped, 1);
            assert_eq!(b.bodies.len(), 1);
            let g = frame_geometry(&b, 59.0);
            // A 20 px disc on a 1000 px canvas has radius 0.02; the block's top is 0.30.
            assert!(lowest(&g) < 0.302, "{mode:?} sank into the solid: {}", lowest(&g));
            assert!(lowest(&g) > 0.2, "{mode:?} did not fall: {}", lowest(&g));
        }
    }

    fn tag(mut g: Geometry, a: PhysAttrs) -> Geometry {
        for pr in &mut g.prims {
            pr.phys = Some(a);
        }
        g
    }

    /// Per-shape settings reach the solver: a rubber square bounces higher
    /// than a stone one, a pinned one never moves and still stops others,
    /// and a start velocity carries it sideways.
    #[test]
    fn per_shape_attributes_are_honoured() {
        let peak_after_bounce = |a: PhysAttrs| {
            let g = tag(geo_of(&[square([0.5, 0.1], 0.03)], true), a);
            let b = bake(&g, &Geometry::default(), &PhysicsSettings { frames: 96, ..Default::default() }, 0.5625, 1920.0, 24.0);
            // Lowest point per frame; after first touching the floor, how
            // high does it come back up?
            let lows: Vec<f32> = (0..96).map(|f| lowest(&frame_geometry(&b, f as f32))).collect();
            let touch = lows.iter().position(|y| *y > 0.555).unwrap_or(95);
            lows[touch..].iter().copied().fold(f32::MAX, f32::min)
        };
        let rubber = peak_after_bounce(PhysAttrs { bounce: 0.9, ..Default::default() });
        let stone = peak_after_bounce(PhysAttrs { bounce: 0.0, ..Default::default() });
        assert!(rubber < stone - 0.02, "rubber did not bounce higher: rubber {rubber} stone {stone}");

        // Pinned: the lower square stays put and the upper one lands on it.
        for mode in [PhysicsMode::Rigid, PhysicsMode::Soft] {
            let floor = tag(geo_of(&[square([0.5, 0.35], 0.05)], true), PhysAttrs { pinned: true, ..Default::default() });
            let mut g = geo_of(&[square([0.5, 0.1], 0.03)], true);
            g.merge(floor);
            let b = bake(&g, &Geometry::default(), &PhysicsSettings { mode, frames: 72, walls: false, ..Default::default() }, 0.5625, 1920.0, 24.0);
            let last = frame_geometry(&b, 71.0);
            let first = frame_geometry(&b, 0.0);
            assert_eq!(last.polyline(1), first.polyline(1), "{mode:?}: the pinned square moved");
            let top_of_falling = last.polyline(0).iter().map(|q| q[1]).fold(f32::MIN, f32::max);
            assert!(top_of_falling < 0.305 && top_of_falling > 0.25, "{mode:?}: did not land on the pinned one: {top_of_falling}");
        }

        // Velocity: thrown right in both modes, with no gravity.
        for (mode, kind) in [(PhysicsMode::Rigid, BodyKind::Rigid), (PhysicsMode::Soft, BodyKind::Soft)] {
            let g = tag(geo_of(&[square([0.3, 0.25], 0.03)], true), PhysAttrs { kind, velocity: [2.0, 0.0], ..Default::default() });
            let b = bake(&g, &Geometry::default(), &PhysicsSettings { mode, gravity: 0.0, frames: 13, walls: false, ..Default::default() }, 0.5625, 1920.0, 24.0);
            let cx = |f: f32| {
                let pl = frame_geometry(&b, f).polyline(0);
                pl.iter().map(|q| q[0]).sum::<f32>() / pl.len() as f32
            };
            // 2 m/s for half a second on a 10 m canvas: 0.1 across.
            let moved = cx(12.0) - cx(0.0);
            assert!((moved - 0.1).abs() < 0.02, "{mode:?}: moved {moved}");
        }
    }

    /// Streaming hands out exactly the frames a full bake holds, in order,
    /// and a sink that says stop stops it.
    #[test]
    fn streaming_matches_the_full_bake_and_stops_on_request() {
        let g = geo_of(&[square([0.5, 0.1], 0.04)], true);
        for mode in PhysicsMode::ALL {
            let s = PhysicsSettings { mode, frames: 20, ..Default::default() };
            let full = bake(&g, &Geometry::default(), &s, 0.5625, 1920.0, 24.0);
            let mut frames = Vec::new();
            let mut header = None;
            let done = bake_with(&g, &Geometry::default(), &s, 0.5625, 1920.0, 24.0, &mut |m| {
                match m {
                    BakeMsg::Header(h) => header = Some(h),
                    BakeMsg::Frame(r) => frames.push(r),
                }
                true
            });
            assert!(done);
            assert_eq!(header.unwrap().bodies, full.bodies);
            assert_eq!(frames.concat(), full.data);
            let mut n = 0;
            let done = bake_with(&g, &Geometry::default(), &s, 0.5625, 1920.0, 24.0, &mut |m| {
                if matches!(m, BakeMsg::Frame(_)) {
                    n += 1;
                }
                n < 5
            });
            assert!(!done);
            assert_eq!(n, 5);
        }
    }

    /// Every preset is a working simulation.
    #[test]
    fn every_world_preset_simulates() {
        let g = geo_of(&[square([0.5, 0.1], 0.04), square([0.3, 0.15], 0.03)], true);
        for p in &WORLD_PRESETS {
            let mut s = PhysicsSettings { frames: 24, ..Default::default() };
            (p.apply)(&mut s);
            let b = bake(&g, &Geometry::default(), &s, 0.5625, 1920.0, 24.0);
            assert_eq!(b.frames, 24, "{}", p.name);
            assert!(b.data.iter().all(|v| v.is_finite()), "{} went non-finite", p.name);
        }
        assert!(RIGID_MATERIALS.iter().all(|m| !m.0.is_empty() && m.1 > 0.0));
        assert!(SOFT_MATERIALS.iter().all(|m| !m.0.is_empty() && m.2 > 0.0));
    }

    /// A rigid glyph keeps its counter through the bake, and a v1 bake
    /// (before counters) still reads.
    #[test]
    fn a_rigid_glyph_keeps_its_hole_and_v1_bakes_read() {
        let o = crate::geotext::layout(&crate::geotext::TextSpec {
            text: "o",
            font: &crate::geotext::TextFont::Sans,
            size: 0.15,
            position: [0.4, 0.2],
            align: Default::default(),
            tracking: 0.0,
            leading: 1.2,
            rotation: 0.0,
            style: Style::default(),
            px_per_unit: 1920.0,
        });
        let b = bake(&o, &Geometry::default(), &PhysicsSettings { frames: 24, ..Default::default() }, 0.5625, 1920.0, 24.0);
        let g = frame_geometry(&b, 23.0);
        assert_eq!(g.rings(0).len(), 2, "the counter was lost");
        let (lo, hi) = g.bbox().unwrap();
        assert!(!g.contains([(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5]), "the hole filled in");
        // v1: the same bytes without the ring counts.
        let mut v1 = PhysicsBake { bodies: b.bodies.iter().map(|x| BakedBody {
            shape: match &x.shape { BodyShape::Rigid { local, .. } => BodyShape::Rigid { local: local.clone(), rings: Vec::new() }, s => s.clone() },
            ..x.clone()
        }).collect(), ..b.clone() };
        let mut bytes = v1.to_bytes();
        bytes[..4].copy_from_slice(b"WPB1");
        // Strip the (zero) ring counts a v2 writer added after each rigid body.
        let back = {
            let mut out = Vec::new();
            let mut i = 0;
            out.extend_from_slice(&bytes[..4 + 8 + 16 + 4 + 8]);
            i += 4 + 8 + 16 + 4 + 8;
            for body in &v1.bodies {
                let head = 36 + 1 + 4;
                out.extend_from_slice(&bytes[i..i + head]);
                i += head;
                out.push(bytes[i]);
                i += 1;
                if let BodyShape::Rigid { local, .. } = &body.shape {
                    let n = 4 + 8 * local.len();
                    out.extend_from_slice(&bytes[i..i + n]);
                    i += n + 4;
                } else {
                    out.extend_from_slice(&bytes[i..i + 4]);
                    i += 4;
                }
            }
            out.extend_from_slice(&bytes[i..]);
            out
        };
        let read = PhysicsBake::from_bytes(&back).expect("a v1 bake reads");
        v1.bodies.iter_mut().for_each(|_| {});
        assert_eq!(read.bodies.len(), v1.bodies.len());
        assert_eq!(read.data, v1.data);
    }

    #[test]
    fn a_bake_round_trips_through_bytes_and_base64() {
        let bodies = geo_of(&[square([0.5, 0.1], 0.04)], true);
        for mode in PhysicsMode::ALL {
            let b = bake(&bodies, &Geometry::default(), &PhysicsSettings { mode, frames: 10, ..Default::default() }, 0.5625, 1920.0, 24.0);
            let back = PhysicsBake::from_bytes(&b.to_bytes()).expect("reads back");
            assert_eq!(back, b);
            let z = lz4_flex::compress_prepend_size(&b.to_bytes());
            let s = b64_encode(&z);
            let raw = lz4_flex::decompress_size_prepended(&b64_decode(&s).unwrap()).unwrap();
            assert_eq!(PhysicsBake::from_bytes(&raw).unwrap(), b);
        }
        for s in ["", "QQ==", "QUI=", "QUJD", "aGVsbG8gd29ybGQ="] {
            assert_eq!(b64_encode(&b64_decode(s).unwrap()), s);
        }
        assert!(PhysicsBake::from_bytes(b"nope").is_none());
    }

    #[test]
    fn the_input_hash_sees_inputs_and_settings() {
        let a = geo_of(&[square([0.5, 0.1], 0.04)], true);
        let b = geo_of(&[square([0.5, 0.11], 0.04)], true);
        let s = PhysicsSettings::default();
        let h = |g: &Geometry, s: &PhysicsSettings| input_hash(g, &Geometry::default(), s, 0.5625, 24.0);
        assert_eq!(h(&a, &s), h(&a, &s));
        assert_ne!(h(&a, &s), h(&b, &s));
        assert_ne!(h(&a, &s), h(&a, &PhysicsSettings { gravity: 5.0, ..s }));
        // Showing the colliders changes nothing about the simulation.
        assert_eq!(h(&a, &s), h(&a, &PhysicsSettings { show_colliders: false, ..s }));
    }
}
