//! Particles: a general 2D particle system, simulated on the GPU.
//!
//! A Particles node OWNS a small graph of [`PNode`]s — the SDF node's
//! arrangement — and opens onto it (↩). Particle STREAMS flow along its
//! wires:
//!
//!  - **Emit** makes a stream: particles born on a point, a line, a ring, a
//!    disc or a box, at a rate, with a life, a speed and a direction.
//!  - **Force** acts on every stream that flows through it: gravity, a
//!    radial push or pull, a vortex, drag, curl-noise turbulence, FIELD
//!    LINES (particles ride the field of ± pole pairs, which is how the
//!    Plasma node's loops and streamers come back as a preset), and a kill
//!    region.
//!  - **Merge** joins two streams so the forces after it act on both.
//!  - **Render** draws the streams that reach it as streaks of light, added
//!    up in one layer the fused pass samples (`SourceBinding::Particles`).
//!
//! **What must stay true** (the GPU half is [`crate::particlegpu`]):
//!
//!  - **A frame is a pure function of the authored data and the frame
//!    number.** Births are scheduled by the frame (slot `i` of an emitter is
//!    reborn every `cap / rate` frames, offset by `i / rate`), every random
//!    number is a hash of (slot, generation, seed), and the simulation
//!    always starts empty at `start - preroll`. So the compositor may jump
//!    anywhere: it restores the nearest checkpoint at or before the frame
//!    and steps forward, and the pixels are the same as if it had played.
//!  - **The simulation runs in the node's FULL-RESOLUTION pixels.** Proxy
//!    scales the drawing only, so a proxy flip never re-simulates and never
//!    changes where a particle goes.
//!  - **Nothing here recompiles.** The emitters, forces and renders are
//!    records in a table the compute shader loops over; rewiring the inner
//!    graph changes the table, not the shader. The comp side is one texture
//!    read (`topo_tag` is a constant).
//!  - **Sim and look are hashed apart** ([`Compiled::sim_hash`] vs the
//!    render records): a colour or streak drag redraws the particles where
//!    they are; an emitter or force edit throws the checkpoints away.

use crate::codegen::Emit;
use crate::graph::NodeId;
use crate::nodes::NodeDef;
use std::collections::BTreeMap;

/// Topology tag: the whole comp side is one code path.
const TOPO: u64 = 0x9A27_1C1E;

/// Channel prefix for the owner's keyable numbers (`particles.<name>`).
pub const PREFIX: &str = "particles.";

/// Emitters, forces, renders and field charges one node may hold. The
/// table is fixed-size so the shader never changes.
pub const MAX_EMITTERS: usize = 32;
pub const MAX_FORCES: usize = 32;
pub const MAX_RENDERS: usize = 16;
pub const MAX_CHARGES: usize = 512;
/// Particle slots across every emitter of one node.
pub const MAX_SLOTS: u32 = 4_000_000;

/// vec4 layout of the simulation table ([`Compiled::sim`]).
pub const SIM_HEADER: usize = 2;
pub const EMIT_VEC4S: usize = 6;
pub const FORCE_VEC4S: usize = 4;
pub const EMIT_BASE: usize = SIM_HEADER;
pub const FORCE_BASE: usize = EMIT_BASE + EMIT_VEC4S * MAX_EMITTERS;
pub const CHARGE_BASE: usize = FORCE_BASE + FORCE_VEC4S * MAX_FORCES;
/// A charge is two vec4s: `(x, y, z, q)` and `(link, -, -, -)`.
pub const SIM_VEC4S: usize = CHARGE_BASE + 2 * MAX_CHARGES;
/// vec4s in one Render's uniform.
pub const RENDER_VEC4S: usize = 6;

// ---------------------------------------------------------------------------
// The inner graph
// ---------------------------------------------------------------------------

/// The graph a Particles node owns. DATA: it serializes inside the node.
#[derive(Clone, Debug, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct ParticleGraph {
    pub nodes: BTreeMap<NodeId, PNode>,
    /// (consumer, socket) -> producer.
    #[cfg_attr(feature = "serde", serde(with = "crate::field::edge_map"))]
    pub edges: BTreeMap<(NodeId, usize), NodeId>,
    /// Canvas positions — the editor's business, but authored state.
    pub layout: BTreeMap<NodeId, [f32; 2]>,
    /// Highest id ever handed out.
    pub next_id: NodeId,
}

impl ParticleGraph {
    pub fn input_of(&self, node: NodeId, port: usize) -> Option<NodeId> {
        self.edges.get(&(node, port)).copied()
    }
}

/// Where an emitter's particles are born.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum EmitShape {
    #[default]
    Point,
    /// `size.x` long, turned by `angle`.
    Line,
    /// The edge of a circle `size.x` in radius.
    Ring,
    /// Anywhere inside a circle `size.x` in radius.
    Disc,
    /// Anywhere inside a `size` box, turned by `angle`.
    Box,
}

impl EmitShape {
    pub const ALL: [EmitShape; 5] =
        [EmitShape::Point, EmitShape::Line, EmitShape::Ring, EmitShape::Disc, EmitShape::Box];

    pub fn name(self) -> &'static str {
        match self {
            EmitShape::Point => "point",
            EmitShape::Line => "line",
            EmitShape::Ring => "ring",
            EmitShape::Disc => "disc",
            EmitShape::Box => "box",
        }
    }

    fn code(self) -> f32 {
        self as u32 as f32
    }
}

/// An emitter. Positions and lengths are pixels of the node's format,
/// angles degrees (0 = right, 90 = up), times frames.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize), serde(default))]
pub struct EmitParams {
    pub shape: EmitShape,
    pub center: [f32; 2],
    pub size: [f32; 2],
    pub angle: f32,
    /// Particles born a frame (times the owner's `quality`).
    pub rate: f32,
    /// Frames a particle lives, and how much that varies (0..1).
    pub life: f32,
    pub life_var: f32,
    /// Launch speed, pixels a frame, and its variation (0..1).
    pub speed: f32,
    pub speed_var: f32,
    /// Launch direction and the cone about it, degrees.
    pub direction: f32,
    pub spread: f32,
    /// 0 = launch along `direction`; 1 = along the shape's outward normal
    /// (out of a ring, up off a line).
    pub normal: f32,
    /// Hot spots along the shape (0 = even), how wide each is (a fraction
    /// of the shape) and what share of the particles they take.
    pub spots: f32,
    pub spot_size: f32,
    pub clump: f32,
    pub seed: f32,
}

impl Default for EmitParams {
    fn default() -> Self {
        Self {
            shape: EmitShape::Point,
            center: [960.0, 540.0],
            size: [300.0, 300.0],
            angle: 0.0,
            rate: 2000.0,
            life: 60.0,
            life_var: 0.3,
            speed: 4.0,
            speed_var: 0.5,
            direction: 90.0,
            spread: 360.0,
            normal: 0.0,
            spots: 0.0,
            spot_size: 0.05,
            clump: 1.0,
            seed: 0.0,
        }
    }
}

/// What a Force does.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ForceKind {
    /// A constant pull along `direction`, `strength` px a frame a frame.
    #[default]
    Gravity,
    /// Away from `center` (negative pulls in), fading to nothing at
    /// `radius` (0 = everywhere).
    Radial,
    /// Round `center`, counter-clockwise for a positive strength.
    Vortex,
    /// Velocity lost a frame, as a fraction (`strength`).
    Drag,
    /// Curl noise: a swirling, divergence-free flow `scale` px across,
    /// evolving at `speed`. `follow` 1 carries particles along it (their
    /// own velocity untouched); 0 pushes them.
    Turbulence,
    /// The field of `pairs` ± poles sat just under a ring (`center`,
    /// `radius`). `follow` 1 locks a particle's velocity onto its field line
    /// at `strength` px a frame — loops between poles, streamers off the
    /// unpaired ones; 0 pushes along the line.
    FieldLines,
    /// Kill what is inside the circle (`invert`: outside it).
    Kill,
}

impl ForceKind {
    pub const ALL: [ForceKind; 7] = [
        ForceKind::Gravity,
        ForceKind::Radial,
        ForceKind::Vortex,
        ForceKind::Drag,
        ForceKind::Turbulence,
        ForceKind::FieldLines,
        ForceKind::Kill,
    ];

    pub fn name(self) -> &'static str {
        match self {
            ForceKind::Gravity => "gravity",
            ForceKind::Radial => "radial",
            ForceKind::Vortex => "vortex",
            ForceKind::Drag => "drag",
            ForceKind::Turbulence => "turbulence",
            ForceKind::FieldLines => "field lines",
            ForceKind::Kill => "kill",
        }
    }

    fn code(self) -> f32 {
        self as u32 as f32
    }
}

/// A force. One struct for every kind (the panel shows the fields a kind
/// reads), so switching kind keeps the numbers.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize), serde(default))]
pub struct ForceParams {
    pub kind: ForceKind,
    pub strength: f32,
    pub direction: f32,
    pub center: [f32; 2],
    pub radius: f32,
    /// Turbulence: feature size (px) and how fast it evolves.
    pub scale: f32,
    pub speed: f32,
    /// 0 = a push (acceleration), 1 = a carry (velocity), between = a mix.
    pub follow: f32,
    pub seed: f32,
    /// Kill: kill outside the circle instead.
    pub invert: bool,
    /// Field lines: particles flow either way along their line, half each.
    pub both_ways: bool,
    /// Field lines: pole pairs, the regions they gather in and how hard
    /// (0 = anywhere on the ring), a region's width (radians).
    pub pairs: f32,
    pub regions: f32,
    pub cluster: f32,
    pub region_size: f32,
    /// Field lines: a pair's pole separation (radians on the ring) and
    /// how much it varies (0..1).
    pub span: f32,
    pub span_var: f32,
    /// Field lines: how deep under the ring the poles sit (fraction of
    /// the radius): deeper = rounder, taller loops.
    pub depth: f32,
    /// Field lines: share of pairs with no partner — open lines,
    /// streamers.
    pub open: f32,
    /// Field lines: re-home every newborn particle onto a pole, on a
    /// ring this many px round it (0 = leave births where the emitter put them).
    /// A particle born on a pole flows AWAY from it along its line, so
    /// the loops are bundles rooted at their footpoints.
    pub foot: f32,
    /// Field lines: with `foot`, how many lines leave each pole (0 = any
    /// direction, a haze). Many particles on the SAME line is what draws a
    /// crisp thread.
    pub strands: f32,
    /// Field lines: with `foot`, the cone (degrees) a pole's lines leave
    /// in, measured from the line to its partner, on the outward side. A
    /// lone pair's lines are circles through both poles, so this bounds
    /// the loop height: 90 = a semicircle, the tallest loop half the
    /// pair's separation.
    pub fan: f32,
    /// Field lines: share of pairs sat INSIDE the ring at any angle —
    /// loops across the face, as if seen from above.
    pub fill: f32,
    /// Field lines: with `foot`, a particle feels only the pole it was
    /// born on and that pole's partner, so its line is a clean arc (or a
    /// straight ray off an unpaired pole) however crowded the ring is.
    /// Off, every pole pulls on it and the lines tangle like a real field.
    pub own_pair: bool,
    /// Field lines: 0 = every pole on the ring (flat, side-on arches);
    /// 1 = poles all over a SPHERE of that radius, seen from the front.
    /// Particles then carry a depth: loops stand off the surface, lie
    /// foreshortened across the face, and a Render's `occlude` hides the
    /// ones behind the disc. (Plasma's `depth`.)
    pub sphere: f32,
}

impl Default for ForceParams {
    fn default() -> Self {
        Self {
            kind: ForceKind::Gravity,
            strength: 0.1,
            direction: -90.0,
            center: [960.0, 540.0],
            radius: 300.0,
            scale: 200.0,
            speed: 0.02,
            follow: 1.0,
            seed: 0.0,
            invert: false,
            both_ways: true,
            pairs: 24.0,
            regions: 6.0,
            cluster: 0.8,
            region_size: 0.3,
            span: 0.25,
            span_var: 0.6,
            depth: 0.04,
            open: 0.2,
            foot: 0.0,
            strands: 0.0,
            fan: 100.0,
            fill: 0.0,
            own_pair: false,
            sphere: 0.0,
        }
    }
}

impl ForceParams {
    pub fn of(kind: ForceKind) -> Self {
        let mut f = ForceParams { kind, ..Default::default() };
        match kind {
            ForceKind::Gravity => f.strength = 0.1,
            ForceKind::Radial => f.strength = 0.2,
            ForceKind::Vortex => f.strength = 0.2,
            ForceKind::Drag => f.strength = 0.05,
            ForceKind::Turbulence => {
                f.strength = 2.0;
                f.follow = 1.0;
            }
            ForceKind::FieldLines => f.strength = 1.5,
            ForceKind::Kill => f.radius = 280.0,
        }
        f
    }
}

/// How a stream is drawn.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize), serde(default))]
pub struct RenderParams {
    /// Colour at birth and at death, working space.
    pub color: [f32; 3],
    pub color_end: [f32; 3],
    /// Light per particle (divided by the owner's `quality`, so quality
    /// is not an exposure).
    pub brightness: f32,
    /// A particle's streak: length (px) plus `stretch` times its speed,
    /// and its width (px).
    pub streak: f32,
    pub stretch: f32,
    pub width: f32,
    /// Share of the life spent fading in and out.
    pub fade_in: f32,
    pub fade_out: f32,
    /// How much light varies line to line (0 = even): a few bright
    /// threads among many faint ones.
    pub vary: f32,
    /// Hide particles BEHIND a disc (negative depth, inside its outline):
    /// the star itself, for Field Lines on a sphere. Canvas pixels.
    pub occlude: bool,
    pub disc_center: [f32; 2],
    pub disc_radius: f32,
}

impl Default for RenderParams {
    fn default() -> Self {
        Self {
            color: [1.0, 0.8, 0.5],
            color_end: [1.0, 0.25, 0.05],
            brightness: 1.0,
            streak: 2.0,
            stretch: 1.0,
            width: 1.3,
            fade_in: 0.1,
            fade_out: 0.4,
            vary: 0.0,
            occlude: false,
            disc_center: [960.0, 540.0],
            disc_radius: 300.0,
        }
    }
}

/// One node of a particle graph.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PNode {
    Emit(EmitParams),
    Force(ForceParams),
    Merge,
    Render(RenderParams),
}

impl Default for PNode {
    fn default() -> Self {
        PNode::Emit(EmitParams::default())
    }
}

impl PNode {
    /// Every kind, freshly made — what the add menu and the tests walk.
    pub fn catalog() -> Vec<PNode> {
        let mut v = vec![PNode::Emit(EmitParams::default())];
        v.extend(ForceKind::ALL.iter().map(|k| PNode::Force(ForceParams::of(*k))));
        v.push(PNode::Merge);
        v.push(PNode::Render(RenderParams::default()));
        v
    }

    pub fn name(&self) -> &'static str {
        match self {
            PNode::Emit(_) => "Emit",
            PNode::Force(f) => match f.kind {
                ForceKind::Gravity => "Gravity",
                ForceKind::Radial => "Radial",
                ForceKind::Vortex => "Vortex",
                ForceKind::Drag => "Drag",
                ForceKind::Turbulence => "Turbulence",
                ForceKind::FieldLines => "Field Lines",
                ForceKind::Kill => "Kill",
            },
            PNode::Merge => "Merge",
            PNode::Render(_) => "Render",
        }
    }

    /// Socket names, in order.
    pub fn inputs(&self) -> &'static [&'static str] {
        match self {
            PNode::Emit(_) => &[],
            PNode::Force(_) | PNode::Render(_) => &["particles"],
            PNode::Merge => &["a", "b"],
        }
    }

    pub fn topo_tag(&self) -> u64 {
        match self {
            PNode::Emit(_) => 0x51,
            PNode::Force(_) => 0x52,
            PNode::Merge => 0x53,
            PNode::Render(_) => 0x54,
        }
    }
}

// ---------------------------------------------------------------------------
// The owner node
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize), serde(default))]
pub struct ParticlesParams {
    /// The node's format (see `inherit`).
    pub size: (u32, u32),
    pub inherit: bool,
    /// The tick being rendered, stamped per frame. TRANSIENT.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub time: f32,
    /// Salts every emitter's and force's own seed.
    pub seed: f32,
    /// The first frame, and how many frames are simulated before it so the
    /// picture is already full there.
    pub start: f32,
    pub preroll: f32,
    /// Simulation steps a frame (1..8): more for fast particles in tight
    /// forces.
    pub substeps: f32,
    /// Multiplies every rate and divides every Render's light: more
    /// particles, same exposure.
    pub quality: f32,
    /// How much of the light counts as alpha (0 = pure emission).
    pub alpha: f32,
    pub graph: ParticleGraph,
}

impl Default for ParticlesParams {
    fn default() -> Self {
        Self {
            size: (1920, 1080),
            inherit: false,
            time: 0.0,
            seed: 0.0,
            start: 1.0,
            preroll: 0.0,
            substeps: 1.0,
            quality: 1.0,
            alpha: 0.0,
            graph: ParticleGraph::default(),
        }
    }
}

const NAMES: &[&str] = &["seed", "start", "preroll", "substeps", "quality", "alpha"];

/// Fields a compile reads, clamped and finite.
fn fin(x: f32, lo: f32, hi: f32) -> f32 {
    if x.is_finite() { x.clamp(lo, hi) } else { lo }
}

/// A deterministic 0..1 for (thing, which number) — the CPU twin of the
/// shader's `rnd`, used to place field poles.
fn rnd(a: u32, k: u32) -> f32 {
    fn pcg(x: u32) -> u32 {
        let s = x.wrapping_mul(747796405).wrapping_add(2891336453);
        let w = ((s >> ((s >> 28) + 4)) ^ s).wrapping_mul(277803737);
        (w >> 22) ^ w
    }
    (pcg(a ^ pcg(k.wrapping_mul(0x9e3779b9).wrapping_add(0x85ebca6b))) >> 8) as f32 / 16777216.0
}

/// The inner graph boiled down to what the GPU reads.
#[derive(Clone, Debug, PartialEq)]
pub struct Compiled {
    /// The simulation table, `SIM_VEC4S` vec4s.
    pub sim: Vec<f32>,
    /// Hash of everything the simulation depends on (not the look).
    pub sim_hash: u64,
    /// One uniform per drawing Render, `RENDER_VEC4S` vec4s each.
    pub renders: Vec<[f32; RENDER_VEC4S * 4]>,
    /// Slots across every emitter.
    pub slots: u32,
    /// The frame the simulation starts empty at.
    pub origin: i64,
    pub substeps: u32,
}

impl ParticlesParams {
    pub fn names() -> &'static [&'static str] {
        NAMES
    }

    fn field(&mut self, name: &str) -> Option<&mut f32> {
        Some(match name {
            "seed" => &mut self.seed,
            "start" => &mut self.start,
            "preroll" => &mut self.preroll,
            "substeps" => &mut self.substeps,
            "quality" => &mut self.quality,
            "alpha" => &mut self.alpha,
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

    /// The frame being rendered, as a whole simulation step.
    pub fn frame(&self) -> i64 {
        if self.time.is_finite() { (self.time + 1e-4).floor() as i64 } else { 0 }
    }

    /// The emitter bits upstream of every node (cycles and dangling wires
    /// contribute nothing), and the emitters in table order.
    fn masks(&self) -> (BTreeMap<NodeId, u32>, Vec<NodeId>) {
        let g = &self.graph;
        let emitters: Vec<NodeId> = g
            .nodes
            .iter()
            .filter(|(_, n)| matches!(n, PNode::Emit(_)))
            .map(|(id, _)| *id)
            .take(MAX_EMITTERS)
            .collect();
        fn walk(
            g: &ParticleGraph,
            emitters: &[NodeId],
            id: NodeId,
            memo: &mut BTreeMap<NodeId, u32>,
            depth: usize,
        ) -> u32 {
            if let Some(m) = memo.get(&id) {
                return *m;
            }
            if depth > 256 {
                return 0;
            }
            let m = match g.nodes.get(&id) {
                Some(PNode::Emit(_)) => {
                    emitters.iter().position(|e| *e == id).map_or(0, |i| 1u32 << i)
                }
                Some(PNode::Force(_)) | Some(PNode::Render(_)) => g
                    .input_of(id, 0)
                    .map_or(0, |s| walk(g, emitters, s, memo, depth + 1)),
                Some(PNode::Merge) => {
                    let a = g.input_of(id, 0).map_or(0, |s| walk(g, emitters, s, memo, depth + 1));
                    let b = g.input_of(id, 1).map_or(0, |s| walk(g, emitters, s, memo, depth + 1));
                    a | b
                }
                None => 0,
            };
            memo.insert(id, m);
            m
        }
        let mut memo = BTreeMap::new();
        for id in g.nodes.keys() {
            walk(g, &emitters, *id, &mut memo, 0);
        }
        (memo, emitters)
    }

    /// How far downstream of an emitter a node sits: forces apply in this
    /// order, so a chain Gravity -> Drag runs gravity first.
    fn depth_of(&self, id: NodeId) -> usize {
        let g = &self.graph;
        let mut best = 0;
        let mut stack = vec![(id, 0usize)];
        let mut seen = 0usize;
        while let Some((n, d)) = stack.pop() {
            seen += 1;
            if seen > 4096 || d > 256 {
                break;
            }
            best = best.max(d);
            for p in 0..2 {
                if let Some(s) = g.input_of(n, p) {
                    stack.push((s, d + 1));
                }
            }
        }
        best
    }

    /// Boil the graph down to the GPU's tables.
    pub fn compile(&self, proxy: f32) -> Compiled {
        let (masks, emitters) = self.masks();
        let quality = fin(self.quality, 0.01, 64.0);
        let salt = fin(self.seed, -1e6, 1e6).floor() as i64 as u32;
        let substeps = fin(self.substeps, 1.0, 8.0).round() as u32;
        let origin = fin(self.start, -1e6, 1e6).floor() as i64 - fin(self.preroll, 0.0, 1e5).floor() as i64;
        let mut sim = vec![0.0f32; SIM_VEC4S * 4];
        let put = |sim: &mut Vec<f32>, vec4: usize, v: [f32; 4]| {
            sim[vec4 * 4..vec4 * 4 + 4].copy_from_slice(&v);
        };
        let deg = |x: f32| fin(x, -1e5, 1e5).to_radians();

        // Emitters: each owns a run of slots.
        let mut slots = 0u32;
        for (i, id) in emitters.iter().enumerate() {
            let Some(PNode::Emit(e)) = self.graph.nodes.get(id) else { continue };
            let rate = fin(e.rate, 0.0, 1e7) * quality;
            let life = fin(e.life, 0.0, 1e5);
            let var = fin(e.life_var, 0.0, 1.0);
            let want = (rate * life * (1.0 + var)).ceil();
            let cap = (want.min((MAX_SLOTS - slots) as f32)).max(0.0) as u32;
            let period = if rate > 0.0 { (cap as f32 / rate).max(1e-3) } else { 1.0 };
            let b = EMIT_BASE + i * EMIT_VEC4S;
            put(&mut sim, b, [f32::from_bits(slots), f32::from_bits(cap), rate, period]);
            put(&mut sim, b + 1, [
                fin(e.center[0], -1e6, 1e6),
                fin(e.center[1], -1e6, 1e6),
                fin(e.size[0], 0.0, 1e6),
                fin(e.size[1], 0.0, 1e6),
            ]);
            put(&mut sim, b + 2, [deg(e.angle), e.shape.code(), life, var]);
            put(&mut sim, b + 3, [
                fin(e.speed, -1e5, 1e5),
                fin(e.speed_var, 0.0, 1.0),
                deg(e.direction),
                fin(e.spread, 0.0, 360.0).to_radians(),
            ]);
            put(&mut sim, b + 4, [
                fin(e.normal, 0.0, 1.0),
                fin(e.spots, 0.0, 4096.0).floor(),
                fin(e.spot_size, 0.0, 1.0),
                fin(e.clump, 0.0, 1.0),
            ]);
            let seed = (fin(e.seed, -1e6, 1e6).floor() as i64 as u32)
                .wrapping_mul(7919)
                .wrapping_add(salt.wrapping_mul(104729))
                .wrapping_add(i as u32 * 31);
            put(&mut sim, b + 5, [f32::from_bits(seed), 0.0, 0.0, 0.0]);
            slots += cap;
        }

        // Forces, in flow order.
        let mut forces: Vec<(usize, NodeId)> = self
            .graph
            .nodes
            .iter()
            .filter(|(id, n)| matches!(n, PNode::Force(_)) && masks.get(id).copied().unwrap_or(0) != 0)
            .map(|(id, _)| (self.depth_of(*id), *id))
            .collect();
        forces.sort();
        forces.truncate(MAX_FORCES);
        let mut charges: Vec<[f32; 5]> = Vec::new();
        for (j, (_, id)) in forces.iter().enumerate() {
            let Some(PNode::Force(f)) = self.graph.nodes.get(id) else { continue };
            let seed = (fin(f.seed, -1e6, 1e6).floor() as i64 as u32)
                .wrapping_mul(7919)
                .wrapping_add(salt.wrapping_mul(104729));
            let c0 = charges.len();
            if f.kind == ForceKind::FieldLines {
                field_charges(f, seed, &mut charges);
            }
            let b = FORCE_BASE + j * FORCE_VEC4S;
            put(&mut sim, b, [
                f.kind.code(),
                f32::from_bits(masks[id]),
                fin(f.strength, -1e5, 1e5),
                fin(f.follow, 0.0, 1.0),
            ]);
            put(&mut sim, b + 1, [
                fin(f.center[0], -1e6, 1e6),
                fin(f.center[1], -1e6, 1e6),
                if f.kind == ForceKind::FieldLines {
                    if f.own_pair { 1.0 } else { 0.0 }
                } else {
                    deg(f.direction)
                },
                fin(f.radius, 0.0, 1e6),
            ]);
            let field = f.kind == ForceKind::FieldLines;
            put(&mut sim, b + 2, [
                if field { fin(f.strands, 0.0, 4096.0).floor() } else { fin(f.scale, 1.0, 1e6) },
                if field { fin(f.fan, 0.0, 180.0).to_radians() } else { fin(f.speed, -100.0, 100.0) },
                f32::from_bits(seed),
                f32::from_bits(c0 as u32),
            ]);
            put(&mut sim, b + 3, [
                f32::from_bits((charges.len() - c0) as u32),
                if field {
                    fin(f.sphere, 0.0, 1.0)
                } else if f.invert {
                    1.0
                } else {
                    0.0
                },
                if f.both_ways { 1.0 } else { 0.0 },
                if f.kind == ForceKind::FieldLines { fin(f.foot, 0.0, 1e4) } else { 0.0 },
            ]);
        }
        for (k, c) in charges.iter().enumerate() {
            put(&mut sim, CHARGE_BASE + 2 * k, [c[0], c[1], c[2], c[3]]);
            put(&mut sim, CHARGE_BASE + 2 * k + 1, [c[4], 0.0, 0.0, 0.0]);
        }
        put(&mut sim, 0, [
            f32::from_bits(emitters.len() as u32),
            f32::from_bits(forces.len() as u32),
            f32::from_bits(slots),
            f32::from_bits(substeps),
        ]);
        put(&mut sim, 1, [origin as f32, 1.0 / substeps as f32, 0.0, 0.0]);
        let sim_hash = crate::diskcache::content_hash(&[bytemuck::cast_slice(&sim)]);

        // Renders that have something to draw.
        let mut renders = Vec::new();
        for (id, n) in &self.graph.nodes {
            let PNode::Render(r) = n else { continue };
            let mask = masks.get(id).copied().unwrap_or(0);
            if mask == 0 || renders.len() >= MAX_RENDERS {
                continue;
            }
            let light = fin(r.brightness, 0.0, 1e6) * 0.02 / quality;
            let mut u = [0.0f32; RENDER_VEC4S * 4];
            u[0] = 0.0; // canvas w/h: filled by the compositor
            u[1] = 0.0;
            u[2] = proxy;
            u[3] = f32::from_bits(mask);
            for c in 0..3 {
                u[4 + c] = fin(r.color[c], -1e6, 1e6) * light;
                u[8 + c] = fin(r.color_end[c], -1e6, 1e6) * light;
            }
            u[7] = light;
            u[11] = light;
            u[12] = fin(r.streak, 0.0, 1024.0) * proxy;
            u[13] = fin(r.stretch, 0.0, 100.0) * proxy;
            u[14] = (fin(r.width, 0.0, 256.0) * proxy).max(0.35);
            u[15] = fin(r.fade_in, 0.0, 1.0);
            u[16] = fin(r.fade_out, 0.0, 1.0);
            u[17] = f32::from_bits(slots);
            u[18] = fin(r.vary, 0.0, 1.0);
            u[19] = if r.occlude { 1.0 } else { 0.0 };
            u[20] = fin(r.disc_center[0], -1e6, 1e6) * proxy;
            u[21] = fin(r.disc_center[1], -1e6, 1e6) * proxy;
            u[22] = fin(r.disc_radius, 0.0, 1e6) * proxy;
            renders.push(u);
        }
        Compiled { sim, sim_hash, renders, slots, origin, substeps }
    }
}

/// Place a Field Lines force's poles: `pairs` ± pairs under the ring,
/// gathered into regions, a share of them left unpaired (open lines).
fn field_charges(f: &ForceParams, seed: u32, out: &mut Vec<[f32; 5]>) {
    let pairs = fin(f.pairs, 0.0, 256.0).floor() as u32;
    let regions = fin(f.regions, 1.0, 256.0).floor() as u32;
    let r = fin(f.radius, 1.0, 1e6) * (1.0 - fin(f.depth, 0.0, 0.95));
    let (cx, cy) = (fin(f.center[0], -1e6, 1e6), fin(f.center[1], -1e6, 1e6));
    let sphere = fin(f.sphere, 0.0, 1.0);
    let tau = std::f32::consts::TAU;
    // A root on the unit sphere (y down the screen, z towards the eye): on
    // the limb ring when flat, anywhere at sphere 1. Plasma's `root`.
    let root = |a: u32, k: u32| -> [f32; 3] {
        let th = rnd(a, k) * tau;
        let z = (rnd(a, k + 1) * 2.0 - 1.0) * sphere;
        let rr = (1.0 - z * z).max(0.0).sqrt();
        [rr * th.cos(), -rr * th.sin(), z]
    };
    let norm = |v: [f32; 3]| -> [f32; 3] {
        let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-6);
        [v[0] / l, v[1] / l, v[2] / l]
    };
    let cross = |a: [f32; 3], b: [f32; 3]| -> [f32; 3] {
        [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
    };
    for p in 0..pairs {
        if out.len() + 2 > MAX_CHARGES {
            break;
        }
        let id = p.wrapping_mul(2654435761).wrapping_add(seed);
        let mut d = root(id, 30);
        if rnd(id, 2) < fin(f.cluster, 0.0, 1.0) {
            let reg = (rnd(id, 3) * regions as f32) as u32;
            let c = root(reg.wrapping_add(seed.wrapping_mul(31)), 20);
            // A blob about the region's centre: along the limb when flat.
            let g = (rnd(id, 5) + rnd(id, 6) - 1.0) * fin(f.region_size, 0.0, 6.3);
            let t = norm([-c[1], c[0], 0.0]);
            let gz = (rnd(id, 14) + rnd(id, 15) - 1.0) * fin(f.region_size, 0.0, 6.3) * sphere;
            let bz = cross(c, t);
            d = norm([
                c[0] * g.cos() + t[0] * g.sin() + bz[0] * gz,
                c[1] * g.cos() + t[1] * g.sin() + bz[1] * gz,
                c[2] * g.cos() + t[2] * g.sin() + bz[2] * gz,
            ]);
        }
        let sv = 1.0 + fin(f.span_var, 0.0, 1.0) * (rnd(id, 7) * 2.0 - 1.0);
        let half = fin(f.span, 0.0, 3.1) * 0.5 * sv.max(0.05);
        let q = if rnd(id, 8) < 0.5 { 1.0 } else { -1.0 };
        // The pair's axis: along the limb when flat, any way on a sphere.
        let mut t0 = [-d[1], d[0], 0.0];
        if t0[0].abs() + t0[1].abs() < 1e-4 {
            t0 = [1.0, 0.0, 0.0];
        }
        let t0 = norm(t0);
        let b0 = cross(d, t0);
        let chi = rnd(id, 16) * tau * sphere;
        let t = [0, 1, 2].map(|i| t0[i] * chi.cos() + b0[i] * chi.sin());
        let foot = |s: f32| -> [f32; 3] {
            let v = [0, 1, 2].map(|i| d[i] * half.cos() + t[i] * half.sin() * s);
            [cx + r * v[0], cy + r * v[1], r * v[2]]
        };
        let paired = rnd(id, 9) >= fin(f.open, 0.0, 1.0);
        // The link lane says where a pole's partner is in the table (+1 /
        // -1; doubled for a flat pair on the face; 0 = no partner).
        let (mut a, mut b, mut link) = (foot(-1.0), foot(1.0), 1.0);
        if paired && rnd(id, 10) < fin(f.fill, 0.0, 1.0) {
            let (pr, pa) = (r * 0.92 * rnd(id, 11).sqrt(), rnd(id, 12) * tau);
            let ax = rnd(id, 13) * tau;
            let (mx, my) = (cx + pr * pa.cos(), cy - pr * pa.sin());
            let dd = half * r;
            a = [mx - dd * ax.cos(), my - dd * ax.sin(), 0.0];
            b = [mx + dd * ax.cos(), my + dd * ax.sin(), 0.0];
            link = 2.0;
        }
        out.push([a[0], a[1], a[2], q, if paired { link } else { 0.0 }]);
        if paired {
            out.push([b[0], b[1], b[2], -q, -link]);
        }
    }
}

impl NodeDef for ParticlesParams {
    const NAME: &'static str = "Particles";
    const INPUTS: usize = 0;
    const MASK_INPUT: Option<usize> = None;
    /// The alpha gain.
    const VEC4_SLOTS: usize = 1;

    fn topo_tag(&self) -> u64 {
        TOPO
    }

    fn pack(&self, v: &mut [f32]) {
        v[0] = fin(self.alpha, 0.0, 1e6);
    }

    fn emit(&self, e: &mut Emit) -> String {
        let slot = e.slot();
        let pos = e.pos().to_string();
        let t = e.particles_layer();
        let ink = e.fresh();
        e.line(format_args!("let {ink} = sample_px({t}, {pos});"));
        let out = e.fresh();
        e.line(format_args!(
            "let {out} = vec4f({ink}.rgb, clamp({ink}.a * P.v[{slot}u].x, 0.0, 1.0));"
        ));
        out
    }
}

// ---------------------------------------------------------------------------
// Presets
// ---------------------------------------------------------------------------

/// A graph builder: node ids from 1, laid out down the canvas.
struct Builder {
    g: ParticleGraph,
}

impl Builder {
    fn new() -> Self {
        Builder { g: ParticleGraph { next_id: 1, ..Default::default() } }
    }
    fn add(&mut self, n: PNode, at: [f32; 2]) -> NodeId {
        let id = self.g.next_id;
        self.g.next_id += 1;
        self.g.nodes.insert(id, n);
        self.g.layout.insert(id, at);
        id
    }
    fn wire(&mut self, from: NodeId, to: NodeId, port: usize) {
        self.g.edges.insert((to, port), from);
    }
    /// A chain: each node into the next's socket 0.
    fn chain(&mut self, nodes: Vec<PNode>, x: f32) -> Vec<NodeId> {
        let mut ids = Vec::new();
        for (i, n) in nodes.into_iter().enumerate() {
            let id = self.add(n, [x, i as f32 * 110.0]);
            if let Some(prev) = ids.last() {
                self.wire(*prev, id, 0);
            }
            ids.push(id);
        }
        ids
    }
}

/// A ready-made graph.
pub struct ParticlePreset {
    pub name: &'static str,
    pub hint: &'static str,
    pub build: fn() -> ParticleGraph,
}

pub const PRESETS: [ParticlePreset; 4] = [
    ParticlePreset {
        name: "fountain",
        hint: "a point spraying upwards under gravity, cooling as it falls",
        build: fountain,
    },
    ParticlePreset {
        name: "smoke",
        hint: "a soft rising column bent by turbulence",
        build: smoke,
    },
    ParticlePreset { name: "vortex", hint: "a ring of particles drawn into a spiral", build: vortex },
    ParticlePreset {
        name: "corona",
        hint: "the Plasma node's look from the parts: loops along field lines off a star's limb, streamers off the open poles",
        build: corona,
    },
];

/// The empty-node starter: one emitter, one force, one render.
pub fn starter() -> ParticleGraph {
    fountain()
}

fn fountain() -> ParticleGraph {
    let mut b = Builder::new();
    b.chain(
        vec![
            PNode::Emit(EmitParams {
                center: [960.0, 800.0],
                rate: 1500.0,
                life: 90.0,
                speed: 9.0,
                speed_var: 0.3,
                direction: 90.0,
                spread: 25.0,
                ..Default::default()
            }),
            PNode::Force(ForceParams { kind: ForceKind::Gravity, strength: 0.15, direction: -90.0, ..Default::default() }),
            PNode::Force(ForceParams::of(ForceKind::Drag)),
            PNode::Render(RenderParams { fade_in: 0.02, ..Default::default() }),
        ],
        0.0,
    );
    b.g
}

fn smoke() -> ParticleGraph {
    let mut b = Builder::new();
    b.chain(
        vec![
            PNode::Emit(EmitParams {
                shape: EmitShape::Line,
                center: [960.0, 900.0],
                size: [120.0, 0.0],
                rate: 3000.0,
                life: 160.0,
                speed: 2.5,
                direction: 90.0,
                spread: 20.0,
                ..Default::default()
            }),
            PNode::Force(ForceParams {
                kind: ForceKind::Turbulence,
                strength: 2.5,
                scale: 260.0,
                speed: 0.015,
                follow: 1.0,
                ..Default::default()
            }),
            PNode::Force(ForceParams { kind: ForceKind::Drag, strength: 0.02, ..Default::default() }),
            PNode::Render(RenderParams {
                color: [0.7, 0.75, 0.8],
                color_end: [0.2, 0.22, 0.25],
                brightness: 0.4,
                streak: 1.0,
                stretch: 0.5,
                width: 3.0,
                fade_in: 0.2,
                fade_out: 0.6,
                ..Default::default()
            }),
        ],
        0.0,
    );
    b.g
}

fn vortex() -> ParticleGraph {
    let mut b = Builder::new();
    b.chain(
        vec![
            PNode::Emit(EmitParams {
                shape: EmitShape::Ring,
                size: [420.0, 420.0],
                rate: 4000.0,
                life: 140.0,
                speed: 0.5,
                ..Default::default()
            }),
            PNode::Force(ForceParams { kind: ForceKind::Vortex, strength: 0.08, radius: 0.0, ..Default::default() }),
            PNode::Force(ForceParams { kind: ForceKind::Radial, strength: -0.05, radius: 0.0, ..Default::default() }),
            PNode::Force(ForceParams { kind: ForceKind::Drag, strength: 0.03, ..Default::default() }),
            PNode::Render(RenderParams {
                color: [0.5, 0.8, 1.0],
                color_end: [0.2, 0.3, 1.0],
                brightness: 0.6,
                ..Default::default()
            }),
        ],
        0.0,
    );
    b.g
}

/// The Plasma node's corona, rebuilt: loops ride the field of pole pairs
/// sat just under the limb, the unpaired poles throw streamers, a little
/// turbulence sways them, and anything that falls back into the star dies.
fn corona() -> ParticleGraph {
    let (c, r) = ([960.0, 540.0], 300.0);
    let mut b = Builder::new();
    b.chain(
        vec![
            // Where a particle is born does not matter: the field re-homes
            // each one onto a pole (`foot`).
            PNode::Emit(EmitParams {
                shape: EmitShape::Ring,
                center: c,
                size: [r, r],
                rate: 8000.0,
                life: 170.0,
                life_var: 0.7,
                speed: 0.0,
                speed_var: 0.0,
                spread: 0.0,
                ..Default::default()
            }),
            PNode::Force(ForceParams {
                kind: ForceKind::FieldLines,
                center: c,
                radius: r,
                strength: 1.2,
                follow: 1.0,
                pairs: 230.0,
                regions: 14.0,
                cluster: 0.7,
                region_size: 0.3,
                span: 0.2,
                span_var: 0.8,
                depth: 0.01,
                open: 0.36,
                foot: 3.0,
                strands: 12.0,
                fan: 125.0,
                sphere: 1.0,
                own_pair: true,
                seed: 3.0,
                ..Default::default()
            }),
            PNode::Force(ForceParams {
                kind: ForceKind::Turbulence,
                // Slow: a particle keeps the push it was given, so a field
                // that changes within a line's travel time reads as a
                // ripple along the line rather than a sway of it.
                strength: 0.07,
                scale: 500.0,
                speed: 0.002,
                follow: 1.0,
                ..Default::default()
            }),
            PNode::Render(RenderParams {
                color: [1.0, 0.36, 0.06],
                color_end: [1.0, 0.24, 0.03],
                brightness: 0.7,
                streak: 7.0,
                stretch: 0.0,
                width: 1.2,
                fade_in: 0.06,
                fade_out: 0.55,
                vary: 0.9,
                occlude: true,
                disc_center: c,
                disc_radius: r,
            }),
        ],
        0.0,
    );
    b.g
}

#[cfg(test)]
mod tests {
    use super::*;

    fn with(g: ParticleGraph) -> ParticlesParams {
        ParticlesParams { graph: g, ..Default::default() }
    }

    #[test]
    fn every_name_reads_and_writes() {
        let mut p = ParticlesParams::default();
        for (i, n) in ParticlesParams::names().iter().enumerate() {
            let v = i as f32 + 0.5;
            assert!(p.set(n, v), "{n}");
            assert_eq!(p.get(n), Some(v));
        }
        assert!(!p.set("nope", 1.0));
    }

    #[test]
    fn the_look_does_not_touch_the_sim() {
        let a = with(fountain());
        let mut b = a.clone();
        for n in b.graph.nodes.values_mut() {
            if let PNode::Render(r) = n {
                r.color = [0.0, 1.0, 0.0];
                r.streak = 40.0;
            }
        }
        b.time = 50.0;
        assert_eq!(a.compile(1.0).sim_hash, b.compile(1.0).sim_hash);
        assert_ne!(a.compile(1.0).renders, b.compile(1.0).renders);
        // Proxy scales the drawing, never the simulation.
        assert_eq!(a.compile(1.0).sim_hash, a.compile(0.5).sim_hash);
        // An emitter edit does.
        let mut c = a.clone();
        if let Some(PNode::Emit(e)) = c.graph.nodes.get_mut(&1) {
            e.speed += 1.0;
        }
        assert_ne!(a.compile(1.0).sim_hash, c.compile(1.0).sim_hash);
    }

    #[test]
    fn forces_act_on_what_flows_through_them() {
        // Two emitters; gravity on the first only, a merge, drag on both.
        let mut b = Builder::new();
        let e1 = b.add(PNode::Emit(EmitParams::default()), [0.0, 0.0]);
        let e2 = b.add(PNode::Emit(EmitParams::default()), [200.0, 0.0]);
        let g = b.add(PNode::Force(ForceParams::of(ForceKind::Gravity)), [0.0, 100.0]);
        let m = b.add(PNode::Merge, [100.0, 200.0]);
        let d = b.add(PNode::Force(ForceParams::of(ForceKind::Drag)), [100.0, 300.0]);
        let r = b.add(PNode::Render(RenderParams::default()), [100.0, 400.0]);
        b.wire(e1, g, 0);
        b.wire(g, m, 0);
        b.wire(e2, m, 1);
        b.wire(m, d, 0);
        b.wire(d, r, 0);
        let p = with(b.g);
        let (masks, em) = p.masks();
        assert_eq!(em, vec![e1, e2]);
        assert_eq!(masks[&g], 0b01);
        assert_eq!(masks[&d], 0b11);
        assert_eq!(masks[&r], 0b11);
        let c = p.compile(1.0);
        // Gravity runs before drag.
        let kind = |j: usize| c.sim[(FORCE_BASE + j * FORCE_VEC4S) * 4];
        assert_eq!(kind(0), ForceKind::Gravity.code());
        assert_eq!(kind(1), ForceKind::Drag.code());
        assert_eq!(c.renders.len(), 1);
    }

    #[test]
    fn quality_is_not_an_exposure() {
        let a = with(fountain());
        let b = ParticlesParams { quality: 4.0, ..a.clone() };
        let (ca, cb) = (a.compile(1.0), b.compile(1.0));
        assert!(cb.slots > ca.slots * 3);
        let la = ca.renders[0][7] * ca.slots as f32;
        let lb = cb.renders[0][7] * cb.slots as f32;
        assert!((la / lb - 1.0).abs() < 0.01, "{la} vs {lb}");
    }

    #[test]
    fn a_cycle_or_a_dangling_wire_is_harmless() {
        let mut b = Builder::new();
        let f1 = b.add(PNode::Force(ForceParams::default()), [0.0, 0.0]);
        let f2 = b.add(PNode::Force(ForceParams::default()), [0.0, 0.0]);
        b.wire(f1, f2, 0);
        b.wire(f2, f1, 0);
        b.g.edges.insert((99, 0), 1234);
        let c = with(b.g).compile(1.0);
        assert_eq!(c.slots, 0);
        assert!(c.renders.is_empty());
    }

    #[test]
    fn nonsense_never_reaches_the_shader() {
        let mut g = fountain();
        if let Some(PNode::Emit(e)) = g.nodes.get_mut(&1) {
            e.rate = f32::NAN;
            e.center = [f32::INFINITY, 0.0];
        }
        let p = ParticlesParams { quality: f32::NAN, start: f32::NAN, ..with(g) };
        let c = p.compile(1.0);
        assert_eq!(c.slots, 0);
        // A NaN start clamps to the low end: finite, just early.
        assert!(c.origin == -1_000_000 && c.sim[4].is_finite() && c.sim[5].is_finite());
        assert!(c.renders.iter().flatten().enumerate().all(|(i, x)| i % 24 == 3 || i % 24 == 17 || x.is_finite()));
    }

    #[test]
    fn field_poles_are_paired_unless_open() {
        let f = ForceParams { pairs: 10.0, open: 0.0, ..ForceParams::of(ForceKind::FieldLines) };
        let mut v = Vec::new();
        field_charges(&f, 1, &mut v);
        assert_eq!(v.len(), 20);
        assert!(v.iter().map(|c| c[3]).sum::<f32>().abs() < 1e-6, "balanced");
        assert!(v.iter().all(|c| c[2] == 0.0), "flat unless on a sphere");
        let f = ForceParams { open: 1.0, ..f };
        let mut v = Vec::new();
        field_charges(&f, 1, &mut v);
        assert_eq!(v.len(), 10);
    }

    #[test]
    fn every_preset_draws() {
        for p in PRESETS {
            let c = with((p.build)()).compile(1.0);
            assert!(c.slots > 0, "{}", p.name);
            assert_eq!(c.renders.len(), 1, "{}", p.name);
        }
    }
}
