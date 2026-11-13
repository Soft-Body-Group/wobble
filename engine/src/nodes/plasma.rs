//! Plasma: a GPU particle system whose particles flow along magnetic
//! field lines off a star — the closed LOOPS that arch between two
//! footpoints and the open STREAMERS that leave for good. Enough of them,
//! added up, is a corona.
//!
//! **Every particle is a pure function of its index and the frame.**
//! Nothing is simulated and nothing is carried frame to frame: the vertex
//! shader ([`crate::plasmagpu`]) hashes a particle's index into a field
//! line, a strand of that line and a place along it, and moves the place
//! with the clock. So any frame draws on its own, scrubbing is live, a
//! million particles cost no CPU at all, and every number here is a
//! uniform — dragging one never recompiles and never uploads a particle.
//!
//! **The star is a sphere, the picture is 2D.** Field lines are rooted on
//! a unit sphere and projected straight down the view axis onto the disc
//! (`center`, `radius`, canvas pixels). `depth` 0 keeps every root on the
//! limb, so the loops are flat arches seen side-on; `depth` 1 spreads them
//! over the whole sphere, so some lie foreshortened across the face and
//! the ones behind are hidden by the disc (`occlude`). `spin` turns the
//! sphere about the picture's vertical.
//!
//! **Loops come in bundles.** A loop is `strands` nested field lines
//! (`thickness` apart), which is what makes the arcades of a real active
//! region; loops gather into `regions` by `cluster`. Each loop lives
//! `life` frames, fades in and out, and is reborn somewhere else.
//! `turbulence` bends the lines with a smooth moving field that is zero at
//! the feet, so a loop sways without leaving the surface.
//!
//! The pixels are a per-node additive layer the fused pass samples
//! (`SourceBinding::Plasma`); this node's WGSL is one texture read.
//! Output is light: rgb adds up, alpha is the light's own coverage scaled
//! by `alpha` (0 = pure emission, for an Add or a Screen).

use crate::codegen::Emit;
use crate::nodes::NodeDef;

/// Topology tag; the whole node is one code path.
const TOPO: u64 = 0x91A5_0000;

/// Channel prefix for the node's keyable numbers (`plasma.<name>` on
/// `Channel::Custom`, the Text node's precedent).
pub const PREFIX: &str = "plasma.";

/// Most particles one node draws.
pub const MAX_COUNT: f32 = 4_000_000.0;

/// vec4s in the layer's uniform ([`PlasmaParams::uniform`]).
pub const UNIFORM_VEC4S: usize = 12;

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize), serde(default))]
pub struct PlasmaParams {
    /// The node's format (see `inherit`).
    pub size: (u32, u32),
    /// Take the comp's canvas instead of `size`.
    pub inherit: bool,
    /// The tick being rendered, stamped per frame like a Noise's. TRANSIENT.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub time: f32,
    pub seed: f32,
    /// The star's disc, canvas pixels.
    pub center: [f32; 2],
    pub radius: f32,
    /// Particles drawn.
    pub count: f32,
    /// Loop bundles alive at once.
    pub loops: f32,
    /// Nested field lines per loop.
    pub strands: f32,
    /// Active regions the loops gather into, and how hard (0 = anywhere).
    pub regions: f32,
    pub cluster: f32,
    /// How wide a region is, radians on the sphere.
    pub region_size: f32,
    /// 0 = every root on the limb (flat, side-on arches); 1 = all over the
    /// sphere.
    pub depth: f32,
    /// Footpoint separation, as an angle on the surface (radians), and how
    /// much it varies loop to loop (0..1).
    pub span: f32,
    pub span_var: f32,
    /// Apex height, in radii, and how much it varies.
    pub height: f32,
    pub height_var: f32,
    /// How far apart a loop's strands sit, as a fraction of its size.
    pub thickness: f32,
    /// How far loops lean out of their own plane (0..1).
    pub lean: f32,
    /// How far a loop's strands spread side by side along its axis
    /// (radians on the surface): an arcade, a tunnel of arches.
    pub arcade: f32,
    /// Frames a loop lives before it is reborn elsewhere; 0 = forever.
    pub life: f32,
    /// Flow along the line, pixels a frame.
    pub flow: f32,
    /// Brightness waves travelling with the flow (0 = an even line).
    pub pulse: f32,
    /// A particle's streak: its length and width, pixels.
    pub streak: f32,
    pub width: f32,
    /// Turbulence: how far lines are pushed (pixels), the field's size
    /// (features across a radius) and how fast it moves.
    pub turbulence: f32,
    pub turb_scale: f32,
    pub turb_speed: f32,
    /// Random scatter of each particle off its line, pixels.
    pub fuzz: f32,
    /// Fraction of the particles on open lines.
    pub streamers: f32,
    /// How far streamers reach, radii.
    pub reach: f32,
    /// How fast a streamer's light dies with height (higher = a tighter
    /// glow about the limb; the tail is long either way).
    pub falloff: f32,
    /// How hard a region's streamers bend in towards its axis as they
    /// climb (0 = straight out; 1 = they close into a helmet with a cusp).
    pub pinch: f32,
    /// Extra light at the footpoints (0 = even along the loop).
    pub feet: f32,
    /// Sphere turn about the vertical, degrees a frame.
    pub spin: f32,
    /// Hide what is behind the disc.
    pub occlude: bool,
    /// An eclipse: a black disc IN FRONT of everything (the moon), so
    /// only what stands clear of the limb shows. Replaces `body`.
    pub eclipse: bool,
    /// Light per particle and its colour, working space.
    pub brightness: f32,
    pub color: [f32; 3],
    /// How much of the light counts as alpha (0 = pure emission).
    pub alpha: f32,
    /// The star itself: a disc of `color` this bright (0 = none), darker
    /// towards its limb by `limb` (0..1) or brighter (below 0, the way a
    /// thin glowing shell looks). Drawn in the fused pass, under the
    /// particles; where it is, alpha is 1.
    pub body: f32,
    pub limb: f32,
    /// The disc's granulation: how strong (0 = a smooth disc), how many
    /// cells across a radius, and how fast it boils.
    pub mottle: f32,
    pub mottle_scale: f32,
    pub mottle_speed: f32,
}

impl Default for PlasmaParams {
    fn default() -> Self {
        Self {
            size: (1920, 1080),
            inherit: false,
            time: 0.0,
            seed: 0.0,
            center: [960.0, 540.0],
            radius: 300.0,
            count: 400_000.0,
            loops: 160.0,
            strands: 12.0,
            regions: 9.0,
            cluster: 0.8,
            region_size: 0.3,
            depth: 0.85,
            span: 0.32,
            span_var: 0.7,
            height: 0.3,
            height_var: 0.7,
            thickness: 0.45,
            lean: 0.3,
            arcade: 0.08,
            life: 240.0,
            flow: 1.2,
            pulse: 0.5,
            streak: 7.0,
            width: 1.3,
            turbulence: 6.0,
            turb_scale: 3.0,
            turb_speed: 0.02,
            fuzz: 0.5,
            streamers: 0.25,
            reach: 0.7,
            falloff: 2.5,
            pinch: 0.0,
            feet: 2.0,
            spin: 0.0,
            occlude: true,
            eclipse: false,
            brightness: 1.0,
            color: [1.0, 0.3, 0.04],
            alpha: 0.0,
            body: 0.0,
            limb: 0.6,
            mottle: 0.6,
            mottle_scale: 9.0,
            mottle_speed: 0.01,
        }
    }
}

/// A ready-made look: its name and the params it sets (the format, the
/// disc and the seed are the node's own and are kept).
pub struct PlasmaPreset {
    pub name: &'static str,
    pub hint: &'static str,
    pub build: fn() -> PlasmaParams,
}

/// The three layers of a corona — add them up (`examples/corona.wbl`) —
/// and the flat, side-on arches of a 2D sun.
pub const PRESETS: [PlasmaPreset; 6] = [
    PlasmaPreset {
        name: "active regions",
        hint: "big loop arcades gathered into a few regions, with a fan of open lines",
        build: PlasmaParams::default,
    },
    PlasmaPreset {
        name: "surface",
        hint: "the star's disc under a carpet of thousands of tiny loops and a fringe of spicules",
        build: surface,
    },
    PlasmaPreset {
        name: "streamers",
        hint: "open field lines only: long faint rays leaving the star",
        build: streamers,
    },
    PlasmaPreset {
        name: "flat arches",
        hint: "every loop on the limb, side-on: a 2D sun with nothing across its face",
        build: flat_arches,
    },
    PlasmaPreset {
        name: "eclipse",
        hint: "the white-light corona of a total eclipse: fine soft rays gathered into helmets, behind a black disc",
        build: eclipse,
    },
    PlasmaPreset {
        name: "prominences",
        hint: "a few small bright loops right on the limb, pink: lay them over an eclipse",
        build: prominences,
    },
];

fn eclipse() -> PlasmaParams {
    PlasmaParams {
        count: 1_500_000.0,
        loops: 3000.0,
        strands: 8.0,
        regions: 9.0,
        cluster: 0.8,
        region_size: 0.4,
        depth: 0.8,
        streamers: 1.0,
        reach: 2.4,
        falloff: 5.0,
        pinch: 0.55,
        flow: 0.5,
        pulse: 0.15,
        streak: 22.0,
        width: 2.4,
        turbulence: 5.0,
        turb_scale: 1.5,
        turb_speed: 0.008,
        fuzz: 2.5,
        brightness: 1.4,
        color: [1.0, 1.0, 1.0],
        eclipse: true,
        ..Default::default()
    }
}

fn prominences() -> PlasmaParams {
    PlasmaParams {
        count: 60_000.0,
        loops: 16.0,
        strands: 10.0,
        regions: 5.0,
        cluster: 0.7,
        region_size: 0.2,
        depth: 0.12,
        span: 0.07,
        span_var: 0.6,
        height: 0.035,
        height_var: 0.8,
        thickness: 0.9,
        lean: 0.8,
        arcade: 0.03,
        life: 400.0,
        flow: 0.3,
        streak: 3.0,
        width: 1.2,
        turbulence: 2.5,
        turb_scale: 8.0,
        fuzz: 1.2,
        streamers: 0.0,
        feet: 1.0,
        brightness: 1.2,
        color: [1.0, 0.12, 0.3],
        eclipse: true,
        ..Default::default()
    }
}

fn surface() -> PlasmaParams {
    PlasmaParams {
        count: 700_000.0,
        loops: 7000.0,
        strands: 3.0,
        regions: 90.0,
        cluster: 0.75,
        region_size: 0.12,
        depth: 1.0,
        span: 0.06,
        span_var: 0.8,
        height: 0.03,
        height_var: 0.9,
        thickness: 0.6,
        lean: 0.8,
        arcade: 0.02,
        life: 90.0,
        flow: 0.4,
        streak: 3.0,
        width: 1.0,
        turbulence: 2.0,
        turb_scale: 9.0,
        fuzz: 0.8,
        streamers: 0.3,
        reach: 0.07,
        falloff: 2.0,
        feet: 0.5,
        brightness: 0.9,
        body: 0.22,
        limb: -0.6,
        mottle: 0.75,
        mottle_scale: 11.0,
        ..Default::default()
    }
}

fn streamers() -> PlasmaParams {
    PlasmaParams {
        count: 250_000.0,
        loops: 400.0,
        strands: 8.0,
        regions: 14.0,
        cluster: 0.6,
        region_size: 0.22,
        depth: 0.35,
        streamers: 1.0,
        reach: 1.0,
        falloff: 5.0,
        flow: 2.5,
        streak: 16.0,
        width: 1.6,
        turbulence: 16.0,
        turb_scale: 2.0,
        fuzz: 1.5,
        brightness: 0.4,
        ..Default::default()
    }
}

fn flat_arches() -> PlasmaParams {
    PlasmaParams { depth: 0.0, cluster: 0.6, regions: 12.0, body: 0.4, ..Default::default() }
}

/// The keyable numbers, by channel name after [`PREFIX`].
const NAMES: &[&str] = &[
    "seed", "center.x", "center.y", "radius", "count", "loops", "strands", "regions", "cluster",
    "region.size", "depth", "span", "span.var", "height", "height.var", "thickness", "lean", "arcade",
    "life", "flow", "pulse", "streak", "width", "turbulence", "turb.scale", "turb.speed", "fuzz",
    "streamers", "reach", "falloff", "pinch", "feet", "spin", "brightness", "color.r", "color.g",
    "color.b", "alpha", "body", "limb", "mottle", "mottle.scale", "mottle.speed",
];

impl PlasmaParams {
    /// Every keyable number's name (after [`PREFIX`]).
    pub fn names() -> &'static [&'static str] {
        NAMES
    }

    fn field(&mut self, name: &str) -> Option<&mut f32> {
        Some(match name {
            "seed" => &mut self.seed,
            "center.x" => &mut self.center[0],
            "center.y" => &mut self.center[1],
            "radius" => &mut self.radius,
            "count" => &mut self.count,
            "loops" => &mut self.loops,
            "strands" => &mut self.strands,
            "regions" => &mut self.regions,
            "cluster" => &mut self.cluster,
            "region.size" => &mut self.region_size,
            "depth" => &mut self.depth,
            "span" => &mut self.span,
            "span.var" => &mut self.span_var,
            "height" => &mut self.height,
            "height.var" => &mut self.height_var,
            "thickness" => &mut self.thickness,
            "lean" => &mut self.lean,
            "arcade" => &mut self.arcade,
            "life" => &mut self.life,
            "flow" => &mut self.flow,
            "pulse" => &mut self.pulse,
            "streak" => &mut self.streak,
            "width" => &mut self.width,
            "turbulence" => &mut self.turbulence,
            "turb.scale" => &mut self.turb_scale,
            "turb.speed" => &mut self.turb_speed,
            "fuzz" => &mut self.fuzz,
            "streamers" => &mut self.streamers,
            "reach" => &mut self.reach,
            "falloff" => &mut self.falloff,
            "pinch" => &mut self.pinch,
            "feet" => &mut self.feet,
            "spin" => &mut self.spin,
            "brightness" => &mut self.brightness,
            "color.r" => &mut self.color[0],
            "color.g" => &mut self.color[1],
            "color.b" => &mut self.color[2],
            "alpha" => &mut self.alpha,
            "body" => &mut self.body,
            "limb" => &mut self.limb,
            "mottle" => &mut self.mottle,
            "mottle.scale" => &mut self.mottle_scale,
            "mottle.speed" => &mut self.mottle_speed,
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

    /// [`NodeDef::pack`] with the proxy scale (the disc is in pixels).
    pub fn pack_fmt(&self, v: &mut [f32], proxy: f32) {
        let f = |x: f32, lo: f32, hi: f32| if x.is_finite() { x.clamp(lo, hi) } else { lo };
        let t = if self.time.is_finite() { self.time } else { 0.0 };
        v[0] = f(self.alpha, 0.0, 1e6);
        v[1] = f(self.body, 0.0, 1e6);
        v[2] = f(self.limb, -4.0, 1.0);
        v[3] = if self.eclipse { 1.0 } else { 0.0 };
        v[4] = f(self.center[0], -1e6, 1e6) * proxy;
        v[5] = f(self.center[1], -1e6, 1e6) * proxy;
        v[6] = f(self.radius, 1.0, 1e6) * proxy;
        v[7] = f(self.spin, -360.0, 360.0).to_radians() * t;
        v[8..11].copy_from_slice(&self.color);
        v[12] = f(self.mottle, 0.0, 1.0);
        v[13] = f(self.mottle_scale, 0.0, 512.0);
        v[14] = f(self.mottle_speed, -10.0, 10.0) * t;
        v[15] = f(self.seed, -1e6, 1e6).floor();
    }

    /// Take a preset's look, keeping this node's format, disc and seed.
    pub fn apply_preset(&mut self, preset: &PlasmaPreset) {
        let keep = self.clone();
        *self = PlasmaParams {
            size: keep.size,
            inherit: keep.inherit,
            time: keep.time,
            seed: keep.seed,
            center: keep.center,
            radius: keep.radius,
            ..(preset.build)()
        };
    }

    /// Particles this node draws, clamped.
    pub fn particles(&self) -> u32 {
        if self.count.is_finite() { self.count.clamp(0.0, MAX_COUNT) as u32 } else { 0 }
    }

    /// The layer's uniform for a `w` x `h` layer (the node's format, proxy
    /// included): everything the shader reads, pixel lengths scaled by the
    /// proxy. Its bytes are also the layer's memo key.
    pub fn uniform(&self, w: u32, h: u32, proxy: f32) -> [f32; UNIFORM_VEC4S * 4] {
        let f = |x: f32, lo: f32, hi: f32| if x.is_finite() { x.clamp(lo, hi) } else { lo };
        let n = self.particles() as f32;
        let mut v = [0.0f32; UNIFORM_VEC4S * 4];
        v[0] = w as f32;
        v[1] = h as f32;
        v[2] = if self.time.is_finite() { self.time } else { 0.0 };
        v[3] = f(self.seed, -1e6, 1e6).floor();
        v[4] = f(self.center[0], -1e6, 1e6) * proxy;
        v[5] = f(self.center[1], -1e6, 1e6) * proxy;
        v[6] = f(self.radius, 1.0, 1e6) * proxy;
        v[7] = proxy;
        v[8] = n;
        v[9] = f(self.loops, 1.0, 100_000.0).floor();
        v[10] = f(self.regions, 1.0, 4096.0).floor();
        v[11] = f(self.strands, 1.0, 4096.0).floor();
        v[12] = f(self.cluster, 0.0, 1.0);
        v[13] = f(self.depth, 0.0, 1.0);
        v[14] = f(self.span, 0.0, 3.0);
        v[15] = f(self.span_var, 0.0, 1.0);
        v[16] = f(self.height, 0.0, 20.0);
        v[17] = f(self.height_var, 0.0, 1.0);
        v[18] = f(self.thickness, 0.0, 2.0);
        v[19] = f(self.lean, 0.0, 2.0);
        v[20] = f(self.life, 0.0, 1e6);
        v[21] = f(self.flow, -1e3, 1e3) * proxy;
        v[22] = f(self.streak, 0.0, 512.0) * proxy;
        v[23] = (f(self.width, 0.0, 256.0) * proxy).max(0.35);
        v[24] = f(self.turbulence, 0.0, 1e4) * proxy;
        v[25] = f(self.turb_scale, 0.0, 100.0);
        v[26] = f(self.turb_speed, -10.0, 10.0);
        v[27] = f(self.fuzz, 0.0, 1e3) * proxy;
        v[28] = f(self.streamers, 0.0, 1.0);
        v[29] = f(self.reach, 0.0, 50.0);
        v[30] = f(self.falloff, 0.0, 50.0);
        v[31] = f(self.feet, 0.0, 100.0);
        // Light per particle is authored for 100k of them, so the count is
        // a quality control and not an exposure.
        let per = f(self.brightness, 0.0, 1e6) * 0.08 * (100_000.0 / n.max(1.0));
        v[32] = self.color[0] * per;
        v[33] = self.color[1] * per;
        v[34] = self.color[2] * per;
        v[35] = per;
        v[36] = f(self.spin, -360.0, 360.0).to_radians();
        v[37] = if self.occlude { 1.0 } else { 0.0 };
        v[38] = f(self.pulse, 0.0, 1.0);
        v[39] = f(self.region_size, 0.0, 3.2);
        v[40] = f(self.arcade, 0.0, 3.2);
        v[41] = f(self.pinch, 0.0, 1.0);
        v
    }
}

impl NodeDef for PlasmaParams {
    const NAME: &'static str = "Plasma";
    const INPUTS: usize = 0;
    const MASK_INPUT: Option<usize> = None;
    /// Alpha gain + body, the disc, the colour, the mottle.
    const VEC4_SLOTS: usize = 4;

    /// The particles are a layer: one code path.
    fn topo_tag(&self) -> u64 {
        TOPO
    }

    fn pack(&self, v: &mut [f32]) {
        self.pack_fmt(v, 1.0);
    }

    fn emit(&self, e: &mut Emit) -> String {
        let slot = e.slot();
        let pos = e.pos().to_string();
        let t = e.plasma_layer();
        let (_, fns) = e.color_helpers();
        let f = wgsl(fns);
        let out = e.fresh();
        e.line(format_args!("let {out} = {f}({pos}, {slot}u, sample_px({t}, {pos}));"));
        out
    }
}

/// The fused half, once per shader: the particle layer's light, and under
/// it the star's own disc — one pixel of edge, shaded towards the limb,
/// mottled by a noise that lives ON the sphere (so it turns with `spin`
/// and bunches up at the limb the way a real surface does).
fn wgsl(fns: &mut std::collections::BTreeMap<String, String>) -> &'static str {
    const NAME: &str = "plasma_px";
    if fns.contains_key(NAME) {
        return NAME;
    }
    fns.insert(
        NAME.into(),
        "fn plasma_hash(p: vec3i, s: u32) -> f32 {
    var h = bitcast<u32>(p.x) * 0x8da6b343u ^ bitcast<u32>(p.y) * 0xd8163841u
        ^ bitcast<u32>(p.z) * 0xcb1ab31fu ^ s * 0x9e3779b9u;
    h ^= h >> 15u; h *= 0x2c1b3c6du; h ^= h >> 12u; h *= 0x297a2d39u; h ^= h >> 15u;
    return f32(h >> 8u) * (1.0 / 16777216.0);
}
// Value noise, 0..1.
fn plasma_noise(p: vec3f, s: u32) -> f32 {
    let i = vec3i(floor(p));
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let x00 = mix(plasma_hash(i, s), plasma_hash(i + vec3i(1, 0, 0), s), u.x);
    let x10 = mix(plasma_hash(i + vec3i(0, 1, 0), s), plasma_hash(i + vec3i(1, 1, 0), s), u.x);
    let x01 = mix(plasma_hash(i + vec3i(0, 0, 1), s), plasma_hash(i + vec3i(1, 0, 1), s), u.x);
    let x11 = mix(plasma_hash(i + vec3i(0, 1, 1), s), plasma_hash(i + vec3i(1, 1, 1), s), u.x);
    return mix(mix(x00, x10, u.y), mix(x01, x11, u.y), u.z);
}
fn plasma_px(pos: vec2f, slot: u32, ink: vec4f) -> vec4f {
    let a = P.v[slot];
    let c = P.v[slot + 1u];
    let col = P.v[slot + 2u];
    let m = P.v[slot + 3u];
    let light = vec4f(ink.rgb, clamp(ink.a * a.x, 0.0, 1.0));
    if (a.y <= 0.0 && a.w < 0.5) { return light; }
    let q = (pos - c.xy) / c.z;
    let rr = length(q);
    let disc = clamp((1.0 - rr) * c.z + 0.5, 0.0, 1.0);
    if (disc <= 0.0) { return light; }
    // An eclipse: the moon, black and in front of the lot.
    if (a.w > 0.5) { return vec4f(light.rgb * (1.0 - disc), max(light.a * (1.0 - disc), disc)); }
    // mu: the cosine of the angle between the surface and the eye.
    let mu = sqrt(max(1.0 - rr * rr, 0.0));
    var tex = 1.0;
    if (m.x > 0.0) {
        // The point on the sphere, turned back by the spin so the
        // surface carries its pattern round with it.
        let cs = cos(c.w);
        let sn = sin(c.w);
        let n = vec3f(q.x * cs - mu * sn, -q.y, q.x * sn + mu * cs);
        let seed = u32(i32(m.w));
        var sum = 0.0;
        var amp = 0.5;
        var fr = m.y;
        for (var o = 0u; o < 4u; o = o + 1u) {
            sum += amp * plasma_noise(n * fr + vec3f(0.0, m.z * (1.0 + f32(o)), f32(o) * 7.3), seed + o);
            amp *= 0.5;
            fr *= 2.1;
        }
        // Granules: bright cells with dark lanes between.
        let cells = smoothstep(0.3, 0.72, sum / 0.9375);
        tex = mix(1.0, 0.15 + cells * 1.9, m.x);
    }
    let body = col.rgb * (a.y * disc * (1.0 - a.z * (1.0 - mu)) * tex);
    return vec4f(light.rgb + body, clamp(light.a + disc, 0.0, 1.0));
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
        let a = PlasmaParams::default();
        let mut b = a.clone();
        b.count = 10.0;
        b.loops = 3.0;
        b.occlude = false;
        b.time = 99.0;
        b.color = [0.0, 1.0, 0.0];
        assert_eq!(a.topo_tag(), b.topo_tag());
    }

    #[test]
    fn every_name_reads_and_writes() {
        let mut p = PlasmaParams::default();
        for (i, n) in PlasmaParams::names().iter().enumerate() {
            let v = i as f32 * 0.25 + 0.125;
            assert!(p.set(n, v), "{n} does not set");
            assert_eq!(p.get(n), Some(v), "{n} does not read back");
        }
        assert!(!p.set("nope", 1.0));
    }

    #[test]
    fn the_count_is_not_an_exposure() {
        let a = PlasmaParams::default();
        let b = PlasmaParams { count: a.count * 4.0, ..a.clone() };
        let (ua, ub) = (a.uniform(64, 64, 1.0), b.uniform(64, 64, 1.0));
        assert!((ua[35] * a.count - ub[35] * b.count).abs() < 1e-3);
    }

    #[test]
    fn proxy_scales_pixels_not_shape() {
        let p = PlasmaParams::default();
        let (full, half) = (p.uniform(1920, 1080, 1.0), p.uniform(960, 540, 0.5));
        assert_eq!(half[6], full[6] * 0.5, "radius");
        assert_eq!(half[22], full[22] * 0.5, "streak");
        assert_eq!(half[14], full[14], "span is an angle");
        assert_eq!(half[8], full[8], "same particles");
    }

    #[test]
    fn nonsense_never_reaches_the_shader() {
        let p = PlasmaParams { count: f32::NAN, radius: f32::NAN, time: f32::NAN, ..Default::default() };
        assert!(p.uniform(8, 8, 1.0).iter().all(|x| x.is_finite()));
        assert_eq!(p.particles(), 0);
    }
}
