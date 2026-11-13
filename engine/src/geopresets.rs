//! Ready-wired 2dini networks: the add menu's "presets" inside a 2dini.
//!
//! Each preset is a small `GeoGraph` — nodes, edges and a canvas layout —
//! whose `out` names the node to wire into the Output. The app copies it
//! into the graph being edited (fresh ids, at the cursor); the tests render
//! every one, so a preset that stops drawing anything fails a build rather
//! than a demo. Physics presets arrive UNBAKED: press bake on the Physics
//! node to watch them run.
//!
//! Coordinates are geo space: x 0..1 across the canvas, y in width units
//! (a 16:9 canvas is 0.5625 tall).

use crate::field::{FieldGraph, FieldNode, MathOp, WarpMode};
use crate::geo::{GeoGraph, GeoNode, GradKind, GradStops, Gradient, Pivot, ResampleMode, Style};
use crate::geophys::{PhysicsMode, PhysicsSettings};
use crate::graph::{NodeId, SplineMode, SplinePoint};

pub struct GeoPreset {
    pub name: &'static str,
    /// Physics presets are grouped apart in the menu.
    pub physics: bool,
    pub build: fn() -> GeoGraph,
}

pub const PRESETS: [GeoPreset; 14] = [
    GeoPreset { name: "star field", physics: false, build: star_field },
    GeoPreset { name: "wobbly title", physics: false, build: wobbly_title },
    GeoPreset { name: "wobbly ring", physics: false, build: wobbly_ring },
    GeoPreset { name: "rosette", physics: false, build: rosette },
    GeoPreset { name: "noise dots", physics: false, build: noise_dots },
    GeoPreset { name: "fountain", physics: false, build: fountain },
    GeoPreset { name: "stencil", physics: false, build: stencil },
    GeoPreset { name: "falling blocks", physics: true, build: falling_blocks },
    GeoPreset { name: "bouncy balls", physics: true, build: bouncy_balls },
    GeoPreset { name: "jelly pile", physics: true, build: jelly_pile },
    GeoPreset { name: "plinko", physics: true, build: plinko },
    GeoPreset { name: "falling letters", physics: true, build: falling_letters },
    GeoPreset { name: "jelly title", physics: true, build: jelly_title },
    GeoPreset { name: "dam break", physics: true, build: dam_break },
];

/// A graph under construction: nodes laid out in columns, left to right.
struct B {
    g: GeoGraph,
}

impl B {
    fn new() -> Self {
        Self { g: GeoGraph::default() }
    }
    fn add(&mut self, n: GeoNode, col: f32, row: f32) -> NodeId {
        self.g.next_id += 1;
        let id = self.g.next_id;
        self.g.nodes.insert(id, n);
        self.g.layout.insert(id, [col * 190.0, row * 90.0]);
        id
    }
    fn wire(&mut self, dst: NodeId, port: usize, src: NodeId) {
        self.g.edges.insert((dst, port), src);
    }
    fn done(mut self, out: NodeId) -> GeoGraph {
        self.g.out = Some(out);
        self.g.next_id += 1;
        self.g
    }
}

fn style(fill: [f32; 4], stroke: [f32; 4], width: f32) -> Style {
    Style { fill, stroke, width, ..Default::default() }
}

/// A Noise subnode: one fBm, scaled, seeded.
fn noise(scale: f32, seed: f32, speed: f32) -> GeoNode {
    let mut graph = FieldGraph::default();
    graph.nodes.insert(1, FieldNode::default());
    graph.nodes.insert(2, FieldNode::Output);
    graph.edges.insert((2, 0), 1);
    graph.layout.insert(1, [0.0, 0.0]);
    graph.layout.insert(2, [0.0, 220.0]);
    graph.next_id = 3;
    GeoNode::Noise { graph, scale, seed, speed }
}

/// A Noise subnode with only two octaves: smooth, for shape wobble.
fn smooth_noise(scale: f32, seed: f32, speed: f32) -> GeoNode {
    let mut n = noise(scale, seed, speed);
    if let GeoNode::Noise { graph, .. } = &mut n {
        if let Some(FieldNode::Fbm { octaves, .. }) = graph.nodes.get_mut(&1) {
            *octaves = 2;
        }
    }
    n
}

fn ngon(center: [f32; 2], radius: f32, sides: u32, st: Style) -> GeoNode {
    GeoNode::Polygon { center, radius, sides, step: 1, rotation: 0.0, style: st }
}

/// A rectangle, as a closed linear Spline (a scatter region, a floor).
fn rect(x0: f32, y0: f32, x1: f32, y1: f32, st: Style) -> GeoNode {
    GeoNode::Spline {
        points: [[x0, y0], [x1, y0], [x1, y1], [x0, y1]].iter().map(|p| SplinePoint::new(*p, 0.0)).collect(),
        mode: SplineMode::Linear,
        closed: true,
        style: st,
    }
}

fn instance(scale: f32) -> GeoNode {
    GeoNode::Instance { scale, rotation: 0.0, inherit_cd: true, pivot: Pivot::Origin }
}

const GOLD: [f32; 4] = [1.0, 0.82, 0.35, 1.0];
const NONE: [f32; 4] = [0.0; 4];

/// Particles from a small disc at the bottom, thrown up and falling back,
/// each one a fading dot stamped by an Instance.
fn fountain() -> GeoGraph {
    let mut b = B::new();
    let src = b.add(ngon([0.5, 0.5], 0.02, 24, style(NONE, NONE, 0.0)), 0.0, 0.0);
    let p = b.add(
        GeoNode::Particles {
            rate: 3.0,
            life: 48.0,
            life_var: 0.25,
            speed: 14.0,
            speed_var: 0.3,
            direction: -std::f32::consts::FRAC_PI_2,
            spread: 0.35,
            gravity: 0.35,
            wind: 0.0,
            drag: 0.01,
            size: 6.0,
            size_end: 1.0,
            fade: true,
            start: 0.0,
            seed: 0.0,
        },
        0.0,
        1.0,
    );
    b.wire(p, 0, src);
    let dot = b.add(ngon([0.0, 0.0], 0.006, 12, style([0.55, 0.8, 1.0, 1.0], NONE, 0.0)), 1.0, 0.0);
    let inst = b.add(instance(1.0), 1.0, 1.0);
    b.wire(inst, 0, dot);
    b.wire(inst, 1, p);
    b.done(inst)
}

fn star_field() -> GeoGraph {
    let mut b = B::new();
    let pts = b.add(GeoNode::Scatter { count: 140, seed: 3.0, density: 1.0 }, 0.0, 0.0);
    let star = b.add(
        GeoNode::Star { center: [0.0, 0.0], outer: 0.012, inner: 0.005, points: 5, rotation: 0.0, style: style(GOLD, NONE, 0.0) },
        0.0,
        1.0,
    );
    // Sizes vary with a noise: some stars big, most small.
    let n = b.add(noise(3.0, 7.0, 0.0), 0.0, 2.0);
    let size = b.add(GeoNode::Remap { in_lo: 0.35, in_hi: 0.65, out_lo: 0.4, out_hi: 2.2, clamp: true }, 1.0, 2.0);
    b.wire(size, 0, n);
    let inst = b.add(instance(1.0), 1.0, 0.0);
    b.wire(inst, 0, star);
    b.wire(inst, 1, pts);
    b.wire(inst, 2, size);
    b.done(inst)
}

fn wobbly_ring() -> GeoGraph {
    let mut b = B::new();
    let ring = b.add(ngon([0.5, 0.28], 0.2, 64, style(NONE, [1.0; 4], 3.0)), 0.0, 0.0);
    let fine = b.add(GeoNode::Resample { mode: ResampleMode::Spacing, spacing: 6.0, count: 64 }, 1.0, 0.0);
    b.wire(fine, 0, ring);
    // Broad, smooth noise: a wobble, not a fringe.
    let nx = b.add(smooth_noise(1.2, 0.0, 0.02), 0.0, 1.0);
    let ny = b.add(smooth_noise(1.2, 11.0, 0.02), 0.0, 2.0);
    let wob = b.add(GeoNode::Displace { mode: WarpMode::Offset, amount: 0.15, x: 0.5, y: 0.5 }, 2.0, 0.0);
    b.wire(wob, 0, fine);
    b.wire(wob, 1, nx);
    b.wire(wob, 2, ny);
    b.done(wob)
}

fn rosette() -> GeoGraph {
    let mut b = B::new();
    let circle = b.add(ngon([0.5, 0.28], 0.14, 12, Style::default()), 0.0, 0.0);
    let spots = b.add(GeoNode::Resample { mode: ResampleMode::Count, spacing: 10.0, count: 12 }, 1.0, 0.0);
    b.wire(spots, 0, circle);
    // A petal: a disc off the origin, squashed along the tangent the
    // instance points it with.
    let disc = b.add(ngon([0.0, 0.1], 0.1, 24, style([0.95, 0.45, 0.6, 0.85], [0.6, 0.15, 0.3, 1.0], 1.5)), 0.0, 1.0);
    let petal = b.add(
        GeoNode::Transform { translate: [0.0, 0.0], rotate: 0.0, scale: [0.35, 1.0], pivot: [0.0, 0.0], inherit: false },
        1.0,
        1.0,
    );
    b.wire(petal, 0, disc);
    let inst = b.add(instance(1.0), 2.0, 0.0);
    b.wire(inst, 0, petal);
    b.wire(inst, 1, spots);
    let heart = b.add(ngon([0.5, 0.28], 0.045, 24, style(GOLD, [0.7, 0.5, 0.1, 1.0], 1.5)), 2.0, 1.0);
    let all = b.add(GeoNode::Merge { count: 3 }, 3.0, 0.0);
    b.wire(all, 0, inst);
    b.wire(all, 1, heart);
    b.done(all)
}

fn noise_dots() -> GeoGraph {
    let mut b = B::new();
    let n = b.add(noise(2.5, 5.0, 0.01), 0.0, 1.0);
    let dens = b.add(GeoNode::Remap { in_lo: 0.45, in_hi: 0.6, out_lo: 0.0, out_hi: 1.0, clamp: true }, 1.0, 1.0);
    b.wire(dens, 0, n);
    let pts = b.add(GeoNode::Scatter { count: 900, seed: 1.0, density: 1.0 }, 1.0, 0.0);
    b.wire(pts, 1, dens);
    let dot = b.add(ngon([0.0, 0.0], 0.004, 12, style([0.4, 0.8, 1.0, 1.0], NONE, 0.0)), 1.0, 2.0);
    let inst = b.add(instance(1.0), 2.0, 0.0);
    b.wire(inst, 0, dot);
    b.wire(inst, 1, pts);
    b.done(inst)
}

/// Shapes scattered in a band across the top, instanced from `template`,
/// tagged by `tag`, dropped by a Physics node with `settings` onto
/// `colliders`. Returns the builder and the Physics node.
fn drop_scene(
    b: &mut B,
    count: u32,
    template: GeoNode,
    tag: GeoNode,
    settings: PhysicsSettings,
) -> (NodeId, NodeId) {
    let band = b.add(rect(0.08, 0.03, 0.92, 0.2, Style::default()), 0.0, 0.0);
    let pts = b.add(GeoNode::Scatter { count, seed: 2.0, density: 1.0 }, 1.0, 0.0);
    b.wire(pts, 0, band);
    let shape = b.add(template, 1.0, 1.0);
    // Each copy turned by a noise, so they do not land flat.
    let turn = b.add(noise(5.0, 9.0, 0.0), 1.0, 2.0);
    let angle = b.add(GeoNode::Math { op: MathOp::Mul, a: 0.0, b: std::f32::consts::TAU }, 2.0, 2.0);
    b.wire(angle, 0, turn);
    let inst = b.add(instance(1.0), 2.0, 0.0);
    b.wire(inst, 0, shape);
    b.wire(inst, 1, pts);
    b.wire(inst, 3, angle);
    let tagged = b.add(tag, 3.0, 0.0);
    b.wire(tagged, 0, inst);
    let phys = b.add(GeoNode::Physics { settings, bake: Default::default() }, 4.0, 0.0);
    b.wire(phys, 0, tagged);
    (phys, tagged)
}

fn rigid(density: f32, friction: f32, bounce: f32) -> GeoNode {
    GeoNode::RigidBody { density, friction, bounce, pinned: false, velocity: [0.0, 0.0], spin: 0.0 }
}

fn title(text: &str, st: Style) -> GeoNode {
    GeoNode::Text {
        text: text.into(),
        font: Default::default(),
        size: 0.16,
        position: [0.5, 0.33],
        align: crate::geotext::TextAlign::Centre,
        tracking: 0.02,
        leading: 1.1,
        rotation: 0.0,
        style: st,
        fx: Default::default(),
    }
}

/// A plate with a word and a row of portholes cut through it: Merge the
/// cutters, Boolean-subtract them from the plate.
fn stencil() -> GeoGraph {
    let mut b = B::new();
    let plate = b.add(rect(0.12, 0.08, 0.88, 0.48, style(GOLD, NONE, 0.0)), 0.0, 0.0);
    let word = b.add(title("stencil", style([1.0; 4], NONE, 0.0)), 1.0, 0.0);
    let hole = b.add(ngon([0.2, 0.42], 0.018, 32, style([1.0; 4], NONE, 0.0)), 2.0, 0.0);
    let row = b.add(
        GeoNode::Transform { translate: [0.6, 0.0], rotate: 0.0, scale: [1.0, 1.0], pivot: [0.0, 0.0], inherit: false },
        2.0,
        1.0,
    );
    b.wire(row, 0, hole);
    let cutters = b.add(GeoNode::Merge { count: 4 }, 1.5, 2.0);
    b.wire(cutters, 0, word);
    b.wire(cutters, 1, hole);
    b.wire(cutters, 2, row);
    let cut = b.add(GeoNode::Boolean { op: crate::geobool::BoolOp::Subtract }, 0.5, 3.0);
    b.wire(cut, 0, plate);
    b.wire(cut, 1, cutters);
    b.done(cut)
}

fn wobbly_title() -> GeoGraph {
    let mut b = B::new();
    let t = b.add(title("wobble", style([1.0, 0.95, 0.85, 1.0], [0.9, 0.4, 0.2, 1.0], 2.0)), 0.0, 0.0);
    // Dense enough that the wobble bends the letters, not just moves them.
    let fine = b.add(GeoNode::Resample { mode: ResampleMode::Spacing, spacing: 4.0, count: 64 }, 1.0, 0.0);
    b.wire(fine, 0, t);
    let nx = b.add(smooth_noise(2.0, 3.0, 0.03), 0.0, 1.0);
    let ny = b.add(smooth_noise(2.0, 17.0, 0.03), 0.0, 2.0);
    let wob = b.add(GeoNode::Displace { mode: WarpMode::Offset, amount: 0.05, x: 0.5, y: 0.5 }, 2.0, 0.0);
    b.wire(wob, 0, fine);
    b.wire(wob, 1, nx);
    b.wire(wob, 2, ny);
    b.done(wob)
}

fn falling_letters() -> GeoGraph {
    let mut b = B::new();
    let t = b.add(
        GeoNode::Text {
            text: "PHYSICS".into(),
            font: Default::default(),
            size: 0.14,
            position: [0.5, 0.16],
            align: crate::geotext::TextAlign::Centre,
            tracking: 0.08,
            leading: 1.1,
            rotation: 0.0,
            style: style([0.95, 0.8, 0.3, 1.0], [0.4, 0.3, 0.05, 1.0], 1.5),
            fx: Default::default(),
        },
        0.0,
        0.0,
    );
    let tag = b.add(rigid(1.0, 0.6, 0.3), 1.0, 0.0);
    b.wire(tag, 0, t);
    let phys = b.add(
        GeoNode::Physics { settings: PhysicsSettings { mode: PhysicsMode::Rigid, frames: 120, ..Default::default() }, bake: Default::default() },
        2.0,
        0.0,
    );
    b.wire(phys, 0, tag);
    // A peg under the middle so they tumble rather than just drop.
    let peg = b.add(ngon([0.5, 0.42], 0.03, 16, style([0.5, 0.5, 0.55, 1.0], NONE, 0.0)), 1.0, 1.0);
    b.wire(phys, 1, peg);
    b.done(phys)
}

fn falling_blocks() -> GeoGraph {
    let mut b = B::new();
    let s = PhysicsSettings { mode: PhysicsMode::Rigid, frames: 144, ..Default::default() };
    let block = ngon([0.0, 0.0], 0.03, 4, style([0.75, 0.72, 0.68, 1.0], [0.3, 0.28, 0.26, 1.0], 1.5));
    let (phys, _) = drop_scene(&mut b, 24, block, rigid(2.5, 0.8, 0.05), s);
    // A ramp to tumble down: an open Stroke is a wall.
    let ramp = b.add(
        GeoNode::Stroke {
            points: vec![SplinePoint::new([0.1, 0.32], 0.0), SplinePoint::new([0.6, 0.45], 0.0)],
            mode: SplineMode::Linear,
            color: [0.6, 0.6, 0.6, 1.0],
            width: 4.0,
        },
        3.0,
        1.0,
    );
    b.wire(phys, 1, ramp);
    b.done(phys)
}

/// Water: a column of it let go at the left, over a block and against a
/// ramp (an open Stroke is a wall to it as to a rigid body).
fn dam_break() -> GeoGraph {
    let mut b = B::new();
    let mut s = PhysicsSettings { mode: PhysicsMode::Fluid, frames: 120, substeps: 6, friction: 0.3, ..Default::default() };
    s.fluid.spacing = 1.0 / 150.0;
    let water = b.add(rect(0.02, 0.12, 0.3, 0.55, style([0.2, 0.55, 1.0, 0.9], NONE, 0.0)), 0.0, 0.0);
    let phys = b.add(GeoNode::Physics { settings: s, bake: Default::default() }, 2.0, 0.0);
    b.wire(phys, 0, water);
    let block = b.add(rect(0.5, 0.46, 0.6, 0.5625, style([0.5, 0.5, 0.55, 1.0], NONE, 0.0)), 0.0, 1.0);
    let ramp = b.add(
        GeoNode::Stroke {
            points: vec![SplinePoint::new([0.75, 0.5625], 0.0), SplinePoint::new([1.0, 0.36], 0.0)],
            mode: SplineMode::Linear,
            color: [0.6, 0.6, 0.6, 1.0],
            width: 4.0,
        },
        0.0,
        2.0,
    );
    let both = b.add(GeoNode::Merge { count: 2 }, 1.0, 1.0);
    b.wire(both, 0, block);
    b.wire(both, 1, ramp);
    b.wire(phys, 1, both);
    b.done(phys)
}

fn bouncy_balls() -> GeoGraph {
    let mut b = B::new();
    let s = PhysicsSettings { mode: PhysicsMode::Rigid, bounce: 0.8, frames: 168, ..Default::default() };
    let ball = ngon([0.0, 0.0], 0.022, 24, style([1.0, 0.35, 0.3, 1.0], [0.5, 0.1, 0.1, 1.0], 1.5));
    let (phys, _) = drop_scene(&mut b, 18, ball, rigid(1.1, 0.9, 0.85), s);
    b.done(phys)
}

fn jelly_pile() -> GeoGraph {
    let mut b = B::new();
    let s = PhysicsSettings { mode: PhysicsMode::Soft, blobbiness: 0.8, frames: 144, ..Default::default() };
    // Sized so the scatter's spacing keeps them apart at the start.
    let blob = ngon([0.0, 0.0], 0.04, 24, style([0.35, 0.85, 0.45, 0.9], [0.1, 0.4, 0.15, 1.0], 2.0));
    let tag = GeoNode::SoftBody { blobbiness: 0.8, pressure: 1.0, friction: 0.4, velocity: [0.0, 0.0], spin: 0.0 };
    let (phys, _) = drop_scene(&mut b, 8, blob, tag, s);
    b.done(phys)
}

/// "wobble" as jelly: circles scattered inside the letters, each pulsing
/// with a moving Noise, fused by a Boolean union into blobs, dropped as
/// soft bodies. The sim takes the blobs as they are at the bake's start
/// frame (a bake reads its input once), so the pulse shows unbaked only.
fn jelly_title() -> GeoGraph {
    let mut b = B::new();
    let pink = [1.0, 0.42, 0.62, 1.0];
    let deep = [0.55, 0.12, 0.3, 1.0];
    let word = b.add(
        GeoNode::Text {
            text: "wobble".into(),
            font: Default::default(),
            size: 0.24,
            position: [0.5, 0.25],
            align: crate::geotext::TextAlign::Centre,
            tracking: 0.04,
            leading: 1.1,
            rotation: 0.0,
            style: style([1.0; 4], NONE, 0.0),
            fx: Default::default(),
        },
        0.0,
        0.0,
    );
    let spots = b.add(GeoNode::Scatter { count: 260, seed: 2.0, density: 1.0 }, 0.0, 1.0);
    b.wire(spots, 0, word);
    let circle = b.add(ngon([0.0, 0.0], 0.0075, 16, style(pink, deep, 2.0)), 1.0, 0.0);
    // The pulse: a smooth moving fBm read at each circle, remapped to a
    // size multiplier, so neighbours breathe together and far ones don't.
    let pulse = b.add(smooth_noise(6.0, 9.0, 0.05), 1.0, 1.0);
    let size = b.add(GeoNode::Remap { in_lo: 0.3, in_hi: 0.7, out_lo: 0.5, out_hi: 1.7, clamp: true }, 1.0, 2.0);
    b.wire(size, 0, pulse);
    let circles = b.add(GeoNode::Instance { scale: 1.0, rotation: 0.0, inherit_cd: false, pivot: Pivot::Origin }, 2.0, 0.0);
    b.wire(circles, 0, circle);
    b.wire(circles, 1, spots);
    b.wire(circles, 2, size);
    let fused = b.add(GeoNode::Boolean { op: crate::geobool::BoolOp::Union }, 3.0, 0.0);
    b.wire(fused, 0, circles);
    // The blobs' look: an edge gradient, green at the rim through blue
    // and violet to cyan in the core.
    let grad = Gradient {
        kind: GradKind::Edge,
        stops: GradStops::from_slice(&[
            [0.0, 0.017405, 1.0, 0.0, 1.0],
            [0.715428, 0.0, 1.0, 0.790768, 1.0],
            [0.248636, 0.959454, 0.0, 0.507681, 1.0],
            [0.500354, 0.0, 0.005511, 0.569305, 1.0],
        ]),
        ..Default::default()
    };
    let look = b.add(GeoNode::Style { style: Style { grad, ..Default::default() } }, 4.0, 0.0);
    b.wire(look, 0, fused);
    let soft = b.add(
        GeoNode::SoftBody { blobbiness: 0.75, pressure: 1.0, friction: 0.4, velocity: [0.0, 0.0], spin: 0.0 },
        5.0,
        0.0,
    );
    b.wire(soft, 0, look);
    let s = PhysicsSettings {
        mode: PhysicsMode::Soft,
        blobbiness: 0.75,
        frames: 200,
        show_colliders: false,
        ..Default::default()
    };
    let phys = b.add(GeoNode::Physics { settings: s, bake: Default::default() }, 6.0, 0.0);
    b.wire(phys, 0, soft);
    b.done(phys)
}

fn plinko() -> GeoGraph {
    let mut b = B::new();
    let s = PhysicsSettings { mode: PhysicsMode::Rigid, bounce: 0.5, frames: 192, ..Default::default() };
    let ball = ngon([0.0, 0.0], 0.012, 16, style([0.95, 0.85, 0.3, 1.0], [0.5, 0.4, 0.1, 1.0], 1.0));
    let (phys, _) = drop_scene(&mut b, 30, ball, rigid(1.0, 0.3, 0.5), s);
    // Pegs: evenly scattered small discs across the middle — colliders.
    let field = b.add(rect(0.05, 0.24, 0.95, 0.5, Style::default()), 3.0, 1.0);
    let spots = b.add(GeoNode::Scatter { count: 70, seed: 4.0, density: 1.0 }, 4.0, 1.0);
    b.wire(spots, 0, field);
    let peg = b.add(ngon([0.0, 0.0], 0.008, 12, style([0.8, 0.8, 0.85, 1.0], NONE, 0.0)), 4.0, 2.0);
    let pegs = b.add(instance(1.0), 5.0, 1.0);
    b.wire(pegs, 0, peg);
    b.wire(pegs, 1, spots);
    b.wire(phys, 1, pegs);
    b.done(phys)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geo::{eval, EvalCx};

    /// Every preset is a well-formed graph that draws something, and every
    /// physics preset bakes to something that moves.
    #[test]
    fn every_preset_draws_and_every_physics_preset_moves() {
        let cx = EvalCx { tick: 0.0, field: None, seed: 0.0, aspect: 0.5625, px_per_unit: 1920.0 };
        for p in &PRESETS {
            let g = (p.build)();
            for ((dst, _), src) in &g.edges {
                assert!(g.nodes.contains_key(dst) && g.nodes.contains_key(src), "{}: dangling edge", p.name);
            }
            let out = g.out.expect("a preset names its output");
            assert!(g.nodes.contains_key(&out), "{}", p.name);
            let geo = eval(&g, &cx);
            assert!(!geo.prims.is_empty(), "{} draws nothing", p.name);
            let dl = crate::geo::flatten(&geo, 1920, 1080);
            assert!(!dl.prims.is_empty(), "{} flattens to nothing", p.name);
            assert_eq!(p.physics, g.nodes.values().any(|n| matches!(n, GeoNode::Physics { .. })), "{}", p.name);
            if p.physics {
                let (phys, settings) = g
                    .nodes
                    .iter()
                    .find_map(|(id, n)| match n {
                        GeoNode::Physics { settings, .. } => Some((*id, *settings)),
                        _ => None,
                    })
                    .unwrap();
                let bodies = g.input_of(phys, 0).map(|s| crate::geo::eval_node(&g, s, &cx)).unwrap_or_default();
                let colliders = g.input_of(phys, 1).map(|s| crate::geo::eval_node(&g, s, &cx)).unwrap_or_default();
                let short = PhysicsSettings { frames: 48, ..settings };
                let b = crate::geophys::bake(&bodies, &colliders, &short, 0.5625, 1920.0, 24.0);
                assert!(!b.bodies.is_empty(), "{} has no bodies", p.name);
                assert!(b.data.iter().all(|v| v.is_finite()), "{} went non-finite", p.name);
                assert_ne!(
                    crate::geophys::frame_geometry(&b, 0.0),
                    crate::geophys::frame_geometry(&b, 47.0),
                    "{} did not move",
                    p.name
                );
            }
        }
    }
}
