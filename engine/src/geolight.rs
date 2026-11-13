//! Light and SDF effects for a 2dini layer, all on the GPU.
//!
//! When a 2dini node's LOOK is on (`GeoParams::look`), its shapes are drawn
//! twice in one pass (`georaster`, the "lit" pipelines): the picture as
//! always, and a LIGHT target holding what each pixel gives off (rgb, from a
//! Light node upstream) and how much it stands in the way of light (a).
//! From there, per layer:
//!
//! 1. **The scene SDF** (`SEED`, `JFA`, `SDF`): a jump flood over the light
//!    target's coverage gives every pixel its signed distance to the
//!    nearest outline, in pixels (negative inside), and where that nearest
//!    outline point is. ONE field, read by both of what follows.
//! 2. **Holographic radiance cascades** (`CASCADE`, `RESOLVE`): a port of
//!    graav_rs's `hrc.wgsl` (the same author's, after Yaazarai's Volumetric-HRC).
//!    A square grid of cells, four "frustums" (light from the right, below,
//!    left, above), cascade n has a plane of probes every 2^n cells and 2^n
//!    cones per probe; rays march the scene SDF, a ray that hits a shape
//!    takes what that shape gives off, one that gets away sees the sky.
//!    Top cascade first; the resolve sums cascade 0's four frustums into
//!    the mean light arriving at every cell.
//! 3. **The finish** (`FINISH`): the shapes lit by that light, then the SDF
//!    effects (drop shadow, outer glow, offset, inner shadow, bevel,
//!    outline), into the texture the fused pass samples.
//!
//! What reruns is decided by what changed (`Need`): a new drawing reruns
//! all three; the cell size or the sky reruns 2 and 3; any other look
//! param (intensity, every effect) only the finish — a single pass. A
//! still drawing with still params costs nothing per frame, like the layer
//! itself. Params are uniforms: nothing here ever recompiles.

/// The look's params are channels of the 2dini node named `look.<param>`
/// (`Channel::Custom`), so they key, cable and drive like any value.
pub const PREFIX: &str = "look.";

/// The grid is never more cells a side than this (graav's measurement: a
/// 1024 grid is ~13 ms on an M4 Pro, 512 about 5): coarser cells instead.
pub const MAX_EXTENT: u32 = 1024;
pub const MIN_EXTENT: u32 = 32;

/// Lighting + effects of one 2dini node. Colours are straight rgba in the
/// working space; lengths are output pixels (y down).
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct GeoLook {
    // --- light ---
    pub light: bool,
    /// Scales all light: what shapes give off and what reaches them.
    pub intensity: f32,
    /// Light everywhere, before any shape gives off any.
    pub ambient: f32,
    /// How much of the light in EMPTY space is drawn (as added light, alpha
    /// 0): the glow of the air round a lamp. 0 = only shapes are lit.
    pub air: f32,
    /// Pixels per cell of the light grid: smaller is finer and dearer.
    pub spacing: f32,
    /// What a ray that escapes the drawing sees.
    pub sky: [f32; 3],
    // --- SDF effects ---
    /// Grows (+) or shrinks (-) every shape by this many pixels.
    pub offset: f32,
    pub shadow: bool,
    pub shadow_offset: [f32; 2],
    pub shadow_soft: f32,
    pub shadow_color: [f32; 4],
    pub glow: bool,
    pub glow_size: f32,
    pub glow_color: [f32; 4],
    pub inner: bool,
    pub inner_offset: [f32; 2],
    pub inner_soft: f32,
    pub inner_color: [f32; 4],
    pub bevel: bool,
    pub bevel_depth: f32,
    /// Where the light comes from, degrees (0 = from the right, 90 = from above).
    pub bevel_angle: f32,
    pub bevel_strength: f32,
    pub outline: bool,
    pub outline_width: f32,
    /// -1 inside the edge, 0 centred on it, 1 outside.
    pub outline_pos: f32,
    pub outline_color: [f32; 4],
}

impl Default for GeoLook {
    fn default() -> Self {
        Self {
            light: false,
            intensity: 1.0,
            ambient: 0.05,
            air: 0.0,
            spacing: 4.0,
            sky: [0.0; 3],
            offset: 0.0,
            shadow: false,
            shadow_offset: [8.0, 8.0],
            shadow_soft: 8.0,
            shadow_color: [0.0, 0.0, 0.0, 0.6],
            glow: false,
            glow_size: 24.0,
            glow_color: [1.0, 0.8, 0.3, 0.8],
            inner: false,
            inner_offset: [4.0, 4.0],
            inner_soft: 6.0,
            inner_color: [0.0, 0.0, 0.0, 0.6],
            bevel: false,
            bevel_depth: 10.0,
            bevel_angle: 120.0,
            bevel_strength: 0.6,
            outline: false,
            outline_width: 4.0,
            outline_pos: 1.0,
            outline_color: [1.0, 1.0, 1.0, 1.0],
        }
    }
}

/// The keyable params, by channel name (after [`PREFIX`]), in panel order.
pub const PARAMS: &[&str] = &[
    "intensity", "ambient", "air", "spacing", "sky.r", "sky.g", "sky.b",
    "offset",
    "shadow.x", "shadow.y", "shadow.soft", "shadow.r", "shadow.g", "shadow.b", "shadow.a",
    "glow.size", "glow.r", "glow.g", "glow.b", "glow.a",
    "inner.x", "inner.y", "inner.soft", "inner.r", "inner.g", "inner.b", "inner.a",
    "bevel.depth", "bevel.angle", "bevel.strength",
    "outline.width", "outline.pos", "outline.r", "outline.g", "outline.b", "outline.a",
];

impl GeoLook {
    /// Anything to do at all? Off, the layer is drawn exactly as before:
    /// one target, no passes.
    pub fn active(&self) -> bool {
        self.light || self.shadow || self.glow || self.inner || self.bevel || self.outline || self.offset != 0.0
    }

    fn slot(&mut self, name: &str) -> Option<&mut f32> {
        let lane = |k: &str| -> Option<usize> { Some(match k { "r" => 0, "g" => 1, "b" => 2, "a" => 3, _ => return None }) };
        let (head, tail) = name.split_once('.').unwrap_or((name, ""));
        Some(match (head, tail) {
            ("intensity", "") => &mut self.intensity,
            ("ambient", "") => &mut self.ambient,
            ("air", "") => &mut self.air,
            ("spacing", "") => &mut self.spacing,
            ("sky", k) => &mut self.sky[lane(k).filter(|&i| i < 3)?],
            ("offset", "") => &mut self.offset,
            ("shadow", "x") => &mut self.shadow_offset[0],
            ("shadow", "y") => &mut self.shadow_offset[1],
            ("shadow", "soft") => &mut self.shadow_soft,
            ("shadow", k) => &mut self.shadow_color[lane(k)?],
            ("glow", "size") => &mut self.glow_size,
            ("glow", k) => &mut self.glow_color[lane(k)?],
            ("inner", "x") => &mut self.inner_offset[0],
            ("inner", "y") => &mut self.inner_offset[1],
            ("inner", "soft") => &mut self.inner_soft,
            ("inner", k) => &mut self.inner_color[lane(k)?],
            ("bevel", "depth") => &mut self.bevel_depth,
            ("bevel", "angle") => &mut self.bevel_angle,
            ("bevel", "strength") => &mut self.bevel_strength,
            ("outline", "width") => &mut self.outline_width,
            ("outline", "pos") => &mut self.outline_pos,
            ("outline", k) => &mut self.outline_color[lane(k)?],
            _ => return None,
        })
    }

    /// A param's value by its name after [`PREFIX`].
    pub fn get(&self, name: &str) -> Option<f32> {
        self.clone().slot(name).map(|v| *v)
    }

    /// Set a param by its name after [`PREFIX`]; false if there is none.
    pub fn set(&mut self, name: &str, v: f32) -> bool {
        match self.slot(name) {
            Some(s) => {
                *s = v;
                true
            }
            None => false,
        }
    }

    /// What the cascades depend on (besides the drawing): a change reruns them.
    fn hrc_key(&self) -> [u32; 5] {
        [
            self.light as u32,
            self.spacing.to_bits(),
            self.sky[0].to_bits(),
            self.sky[1].to_bits(),
            self.sky[2].to_bits(),
        ]
    }

    /// The finish pass's uniform: 12 vec4s, `Look` in [`FINISH`].
    fn uniform(&self, grid: Grid, size: (u32, u32)) -> [f32; 48] {
        let b = |v: bool| if v { 1.0 } else { 0.0 };
        let a = self.bevel_angle.to_radians();
        let mut u = [0.0f32; 48];
        let rows: [[f32; 4]; 12] = [
            [b(self.light), self.intensity, self.ambient, self.air],
            [self.offset, grid.extent as f32, grid.spacing, 1.5 * grid.spacing + 1.0],
            [b(self.shadow), self.shadow_offset[0], self.shadow_offset[1], self.shadow_soft.max(0.0)],
            self.shadow_color,
            [b(self.glow), self.glow_size.max(1e-3), 0.0, 0.0],
            self.glow_color,
            [b(self.inner), self.inner_offset[0], self.inner_offset[1], self.inner_soft.max(0.0)],
            self.inner_color,
            [b(self.bevel), self.bevel_depth.max(1e-3), a, self.bevel_strength],
            [b(self.outline), self.outline_width.max(0.0), self.outline_pos.clamp(-1.0, 1.0), 0.0],
            self.outline_color,
            [size.0 as f32, size.1 as f32, 0.0, 0.0],
        ];
        for (i, r) in rows.iter().enumerate() {
            u[i * 4..i * 4 + 4].copy_from_slice(r);
        }
        u
    }
}

/// The light grid over a layer: a square of `extent` cells (a power of
/// two, the same grid turned four ways), `spacing` pixels each, its corner
/// on the layer's. Cells past the layer's edge are open space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Grid {
    pub extent: u32,
    pub count: u32,
    pub spacing: f32,
}

impl Grid {
    /// The cell size asked for, if the grid it makes fits `MAX_EXTENT`;
    /// the finest that does if not.
    pub fn new(w: u32, h: u32, spacing: f32) -> Self {
        let longest = w.max(h).max(1) as f32;
        let spacing = if spacing.is_finite() { spacing } else { 4.0 };
        let spacing = spacing.max(1.0).max(longest / MAX_EXTENT as f32);
        let extent = ((longest / spacing).ceil() as u32).next_power_of_two().clamp(MIN_EXTENT, MAX_EXTENT);
        Self { extent, count: extent.ilog2(), spacing }
    }
}

const SEED: &str = r#"
struct U { size: vec4i }
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var light_tex: texture_2d<f32>;
@group(0) @binding(2) var<storage, read_write> seeds: array<vec4f>;

fn cov(p: vec2i) -> f32 {
    if (any(p < vec2i(0)) || any(p >= u.size.xy)) { return 0.0; }
    return textureLoad(light_tex, p, 0).a;
}

// A seed is a pixel on an outline: in the anti-aliasing ramp, or solid
// with a neighbour on the other side. It stores where it is and its own
// signed distance to the edge (exact on a straight edge in the ramp).
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let p = vec2i(id.xy);
    if (any(p >= u.size.xy)) { return; }
    let a = cov(p);
    let inside = a >= 0.5;
    var seed = false;
    var sd0 = 0.0;
    if (a > 0.004 && a < 0.996) {
        seed = true;
        sd0 = 0.5 - a;
    } else {
        let o = array<vec2i, 4>(vec2i(1, 0), vec2i(-1, 0), vec2i(0, 1), vec2i(0, -1));
        for (var k = 0; k < 4; k++) {
            if ((cov(p + o[k]) >= 0.5) != inside) { seed = true; }
        }
        sd0 = select(0.5, -0.5, inside);
    }
    let i = u32(p.y * u.size.x + p.x);
    seeds[i] = select(vec4f(-1e7, -1e7, 0.0, 0.0), vec4f(vec2f(p) + 0.5, sd0, 1.0), seed);
}
"#;

const JFA: &str = r#"
struct U { size: vec4i } // xy size, z the step
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> src: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4f>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let p = vec2i(id.xy);
    if (any(p >= u.size.xy)) { return; }
    let c = vec2f(p) + 0.5;
    var best = src[u32(p.y * u.size.x + p.x)];
    var bd = select(1e20, distance(c, best.xy), best.w > 0.5);
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let q = p + vec2i(dx, dy) * u.size.z;
            if (any(q < vec2i(0)) || any(q >= u.size.xy)) { continue; }
            let s = src[u32(q.y * u.size.x + q.x)];
            if (s.w < 0.5) { continue; }
            let d = distance(c, s.xy);
            if (d < bd) { bd = d; best = s; }
        }
    }
    dst[u32(p.y * u.size.x + p.x)] = best;
}
"#;

const SDF: &str = r#"
struct U { size: vec4i }
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var light_tex: texture_2d<f32>;
@group(0) @binding(2) var<storage, read> seeds: array<vec4f>;
@group(0) @binding(3) var out_tex: texture_storage_2d<rgba16float, write>;

// r: signed distance to the nearest outline, px (negative inside);
// gb: that nearest outline point, px.
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let p = vec2i(id.xy);
    if (any(p >= u.size.xy)) { return; }
    let a = textureLoad(light_tex, p, 0).a;
    let s = seeds[u32(p.y * u.size.x + p.x)];
    let c = vec2f(p) + 0.5;
    var sd: f32;
    if (s.w < 0.5) {
        sd = 60000.0; // nothing stands anywhere
    } else if (a > 0.004 && a < 0.996) {
        sd = 0.5 - a;
    } else if (a >= 0.5) {
        sd = min(s.z - distance(c, s.xy), -0.5);
    } else {
        sd = max(distance(c, s.xy) + s.z, 0.5);
    }
    textureStore(out_tex, p, vec4f(clamp(sd, -60000.0, 60000.0), s.xy, 0.0));
}
"#;

/// graav_rs's `hrc.wgsl`, the ray march moved onto the scene SDF and the
/// world/screen/bounce parts left out.
const CASCADE: &str = r#"
struct Hrc {
    extent: f32,
    index: f32,
    count: f32,
    spacing: f32,
    size: vec2f,  // the layer, px
    pad: vec2f,
    sky: vec4f,
}
@group(0) @binding(0) var<uniform> h: Hrc;
@group(0) @binding(1) var above_tex: texture_2d<f32>;
@group(0) @binding(2) var sdf_tex: texture_2d<f32>;
@group(0) @binding(3) var light_tex: texture_2d<f32>;
@group(0) @binding(4) var samp: sampler;
@group(0) @binding(5) var out_tex: texture_storage_2d<rgba16float, write>;

const HIT = 0.35;      // px: nearer than this is a hit
const MAX_STEPS = 64;

// A frustum's own frame <-> the grid's (both 0..1). Its own inverse.
fn turn(uv: vec2f, frustum: f32) -> vec2f {
    if (frustum < 0.5) { return uv; }
    if (frustum < 1.5) { return 1.0 - uv.yx; }
    if (frustum < 2.5) { return 1.0 - uv; }
    return uv.yx;
}

fn to_px(cell: vec2f, frustum: f32) -> vec2f {
    return turn(cell / h.extent, frustum) * h.extent * h.spacing;
}

// How far the nearest shape is, px: off the layer at least as far as the
// layer's edge (no shape lives out there), and never more than the field
// at the edge allows.
fn clearance(p: vec2f) -> f32 {
    let c = clamp(p, vec2f(0.5), h.size - 0.5);
    let d = textureSampleLevel(sdf_tex, samp, c / h.size, 0.0).r;
    let out = length(p - c);
    return max(out, d - out);
}

// What a shape gives off where a ray lands on it.
fn emitted(p: vec2f) -> vec3f {
    let l = textureSampleLevel(light_tex, samp, p / h.size, 0.0);
    return l.rgb / max(l.a, 0.1);
}

struct Seen {
    light: vec3f,
    through: f32,
}

// Walk a ray's stretch: `reach` cells along its longer axis, each step as
// long as the nearest shape is far (half a cell at least, or a thin wall
// leaves a dotted line of light behind it).
fn trace(start: vec2f, toward: vec2f, reach: f32, frustum: f32) -> Seen {
    let delta = toward / max(abs(toward.x), abs(toward.y));
    let per_step = length(delta);
    var t = 0.0;
    for (var i = 0; i < MAX_STEPS && t < reach; i++) {
        let at = start + delta * min(t, reach - 1.0);
        if (any(floor(at / h.extent) != vec2f(0.0))) {
            break; // off the grid
        }
        let px = to_px(at, frustum);
        let open = clearance(px);
        if (open < HIT) {
            return Seen(emitted(px), 0.0);
        }
        t += max(open / h.spacing, 0.5) / per_step;
    }
    return Seen(vec3f(0.0), 1.0);
}

fn cone_above(probe: vec2f, ray: f32, weight: f32, frustum: f32) -> vec3f {
    let at = vec2f(probe.x + ray, probe.y);
    if (h.index > h.count - 1.5 || any(floor(at / h.extent) != vec2f(0.0))) {
        return h.sky.rgb * weight;
    }
    let inside = clamp(at, vec2f(0.5), vec2f(h.extent - 0.5));
    let uv = (inside + vec2f(frustum * h.extent, 0.0)) / vec2f(4.0 * h.extent, h.extent);
    return textureSampleLevel(above_tex, samp, uv, 0.0).rgb;
}

fn trace_and_merge(probe: vec2f, toward: vec2f, reach: f32, weight: f32, ray: f32, frustum: f32) -> vec3f {
    let seen = trace(probe, toward, reach, frustum);
    return seen.light * weight + cone_above(probe + toward, ray, weight, frustum) * seen.through;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let e = u32(h.extent);
    if (id.x >= 4u * e || id.y >= e) { return; }
    let frag = vec2f(id.xy) + 0.5;
    let frustum = floor(frag.x / h.extent);
    let texel = vec2f(frag.x - frustum * h.extent, frag.y);
    let interval = pow(2.0, h.index);
    let plane = floor(texel.x / interval);
    let cone = floor(texel.x - plane * interval);
    let probe = vec2f(plane * interval + 0.5, texel.y);

    // Odd planes sit between the cascade above's: their rays go one
    // interval. Even ones sit ON one, go two, and are mixed with it.
    let even = (plane % 2.0) < 0.5;
    let align = select(1.0, 2.0, even);
    let left = align * vec2f(interval, -interval + 2.0 * cone);
    let right = align * vec2f(interval, -interval + 2.0 * (cone + 1.0));
    let weight = 0.5 * (atan2(right.y, right.x) - atan2(left.y, left.x));

    var l = trace_and_merge(probe, left, interval * align, weight, cone * 2.0, frustum);
    var r = trace_and_merge(probe, right, interval * align, weight, cone * 2.0 + 1.0, frustum);
    if (even) {
        // (Mixed AFTER merging: before would break the shadows.)
        l = mix(l, cone_above(probe, cone * 2.0, weight, frustum), 0.5);
        r = mix(r, cone_above(probe, cone * 2.0 + 1.0, weight, frustum), 0.5);
    }
    textureStore(out_tex, vec2i(id.xy), vec4f(l + r, 1.0));
}
"#;

/// graav_rs's `hrc_resolve.wgsl` without the temporal blend (a comp frame
/// is rendered once, not smoothed over the next).
const RESOLVE: &str = r#"
struct Hrc {
    extent: f32,
    index: f32,
    count: f32,
    spacing: f32,
    size: vec2f,
    pad: vec2f,
    sky: vec4f,
}
@group(0) @binding(0) var<uniform> h: Hrc;
@group(0) @binding(1) var cascade_tex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var out_tex: texture_storage_2d<rgba16float, write>;

fn turn(uv: vec2f, frustum: f32) -> vec2f {
    if (frustum < 0.5) { return uv; }
    if (frustum < 1.5) { return 1.0 - uv.yx; }
    if (frustum < 2.5) { return 1.0 - uv; }
    return uv.yx;
}

fn frustum_at(uv: vec2f, frustum: f32) -> vec3f {
    let cell = clamp(turn(uv, frustum) * h.extent, vec2f(0.5), vec2f(h.extent - 0.5));
    let at = (cell + vec2f(frustum * h.extent, 0.0)) / vec2f(4.0 * h.extent, h.extent);
    return textureSampleLevel(cascade_tex, samp, at, 0.0).rgb;
}

// Each side read a cell further out its own way: neighbouring sides share
// the ray along their seam and would count it twice.
fn four_sides(uv: vec2f) -> vec3f {
    let px = 1.0 / h.extent;
    return frustum_at(uv + vec2f(px, 0.0), 0.0) + frustum_at(uv + vec2f(0.0, -px), 1.0)
        + frustum_at(uv + vec2f(-px, 0.0), 2.0) + frustum_at(uv + vec2f(0.0, px), 3.0);
}

// Read at the cell's four corners: cascade 0 works odd and even cells out
// differently, which shows as a chequer read cell by cell.
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let e = u32(h.extent);
    if (id.x >= e || id.y >= e) { return; }
    let uv = (vec2f(id.xy) + 0.5) / h.extent;
    let k = 0.5 / h.extent;
    let sum = four_sides(uv + vec2f(-k, -k)) + four_sides(uv + vec2f(k, -k))
        + four_sides(uv + vec2f(-k, k)) + four_sides(uv + vec2f(k, k));
    // (Each side holds its light summed over a quarter turn.)
    textureStore(out_tex, vec2i(id.xy), vec4f(sum / (4.0 * 6.283185), 1.0));
}
"#;

const FINISH: &str = r#"
struct Look {
    light: vec4f,   // on, intensity, ambient, air
    grid: vec4f,    // offset, extent, spacing, reach
    shadow: vec4f,  // on, dx, dy, soft
    shadow_c: vec4f,
    glow: vec4f,    // on, size
    glow_c: vec4f,
    inner: vec4f,   // on, dx, dy, soft
    inner_c: vec4f,
    bevel: vec4f,   // on, depth, angle, strength
    outline: vec4f, // on, width, pos
    outline_c: vec4f,
    size: vec4f,
}
@group(0) @binding(0) var<uniform> k: Look;
@group(0) @binding(1) var color_tex: texture_2d<f32>;
@group(0) @binding(2) var light_tex: texture_2d<f32>;
@group(0) @binding(3) var sdf_tex: texture_2d<f32>;
@group(0) @binding(4) var radiance_tex: texture_2d<f32>;
@group(0) @binding(5) var samp: sampler;
@group(0) @binding(6) var out_tex: texture_storage_2d<rgba16float, write>;

fn sd_at(p: vec2f) -> f32 {
    return textureSampleLevel(sdf_tex, samp, p / k.size.xy, 0.0).r - k.grid.x;
}

fn radiance(p: vec2f) -> vec3f {
    return textureSampleLevel(radiance_tex, samp, p / (k.grid.y * k.grid.z), 0.0).rgb;
}

fn over(top: vec4f, under: vec4f) -> vec4f {
    return top + under * (1.0 - top.a);
}

// A straight colour at alpha a, premultiplied.
fn pm(c: vec4f, a: f32) -> vec4f {
    let t = c.a * clamp(a, 0.0, 1.0);
    return vec4f(c.rgb * t, t);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let size = vec2u(k.size.xy);
    if (id.x >= size.x || id.y >= size.y) { return; }
    let p = vec2f(id.xy) + 0.5;
    let sd = sd_at(p);
    // Outward normal: the field's slope.
    let g = vec2f(sd_at(p + vec2f(1.0, 0.0)) - sd_at(p - vec2f(1.0, 0.0)),
                  sd_at(p + vec2f(0.0, 1.0)) - sd_at(p - vec2f(0.0, 1.0)));
    let n = select(vec2f(0.0), normalize(g), dot(g, g) > 1e-8);

    // The shapes: as drawn, or grown / shrunk along the field. A grown
    // pixel takes the colour of the outline point nearest it.
    var c = textureLoad(color_tex, vec2i(id.xy), 0);
    var e = textureLoad(light_tex, vec2i(id.xy), 0).rgb;
    if (k.grid.x != 0.0) {
        let cov = clamp(0.5 - sd, 0.0, 1.0);
        var src = c;
        var se = e;
        if (src.a < 0.5) {
            let near = textureSampleLevel(sdf_tex, samp, p / k.size.xy, 0.0).gb;
            let q = vec2i(clamp(near - n * 1.0, vec2f(0.0), k.size.xy - 1.0));
            src = textureLoad(color_tex, q, 0);
            se = textureLoad(light_tex, q, 0).rgb;
        }
        let s = src.rgb / max(src.a, 1e-4);
        c = vec4f(s * cov, cov);
        e = se / max(src.a, 1e-4) * cov;
    }

    // Light: what arrives at the shape's surface (read just outside it, as
    // inside a shape every ray stops at once), and what the shape gives off.
    var air = vec3f(0.0);
    if (k.light.x > 0.5) {
        var q = p;
        if (sd < k.grid.w) {
            q = p - n * sd + n * k.grid.w;
        }
        let l = radiance(q) * k.light.y;
        c = vec4f(c.rgb * (k.light.z + l) + e * k.light.y, c.a);
        air = radiance(p) * k.light.y * k.light.w * (1.0 - c.a);
    }

    // Inner shadow: where the shape, moved by the offset, no longer covers.
    if (k.inner.x > 0.5) {
        let v = sd_at(p - k.inner.yz);
        let amt = smoothstep(-k.inner.w - 0.5, 0.5, v) * k.inner_c.a;
        c = vec4f(mix(c.rgb, k.inner_c.rgb * c.a, amt), c.a);
    }
    // Bevel: a ridge `depth` px in from the edge, lit from `angle`.
    if (k.bevel.x > 0.5 && sd < 0.0) {
        let t = clamp(-sd / k.bevel.y, 0.0, 1.0);
        let slope = (1.0 - t) * 2.0; // a rounded profile's steepness
        let up = normalize(vec3f(n * slope, 1.0));
        let to_light = normalize(vec3f(cos(k.bevel.z), -sin(k.bevel.z), 1.0));
        let shade = (dot(up, to_light) - to_light.z) * k.bevel.w * 2.0;
        c = vec4f(max(c.rgb + vec3f(shade) * c.a, vec3f(0.0)), c.a);
    }

    // Under the shape: drop shadow, outer glow.
    var under = vec4f(0.0);
    if (k.shadow.x > 0.5) {
        let v = sd_at(p - k.shadow.yz);
        let s = max(k.shadow.w, 0.5);
        under = pm(k.shadow_c, 1.0 - smoothstep(-s, s, v));
    }
    if (k.glow.x > 0.5) {
        let f = 1.0 - clamp(sd / k.glow.y, 0.0, 1.0);
        under = over(pm(k.glow_c, f * f * step(-0.5, sd)), under);
    }
    var o = over(c, under);
    // Over everything: the outline.
    if (k.outline.x > 0.5) {
        let w = k.outline.y;
        let centre = k.outline.z * w * 0.5;
        o = over(pm(k.outline_c, clamp(0.5 + w * 0.5 - abs(sd - centre), 0.0, 1.0)), o);
    }
    textureStore(out_tex, vec2i(id.xy), vec4f(o.rgb + air, o.a));
}
"#;

/// Every WGSL module here, for the validation test.
pub const MODULES: &[(&str, &str)] =
    &[("seed", SEED), ("jfa", JFA), ("sdf", SDF), ("cascade", CASCADE), ("resolve", RESOLVE), ("finish", FINISH)];

/// What a frame has to redo, worst first.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Need {
    Nothing,
    Finish,
    Light,
    All,
}

/// The pipelines, shared by every layer.
pub struct GeoLit {
    seed: wgpu::ComputePipeline,
    jfa: wgpu::ComputePipeline,
    sdf: wgpu::ComputePipeline,
    cascade: wgpu::ComputePipeline,
    resolve: wgpu::ComputePipeline,
    finish: wgpu::ComputePipeline,
    sampler: wgpu::Sampler,
}

/// One layer's textures, buffers and bind groups.
pub struct LitLayer {
    pub width: u32,
    pub height: u32,
    pub grid: Grid,
    /// The light target the rasteriser draws into: rgb given off
    /// (premultiplied by coverage), a = coverage of what stands in the way.
    pub light: wgpu::TextureView,
    /// What the fused pass samples instead of the bare layer.
    pub out: wgpu::TextureView,
    /// What has to be redone at the next encode.
    pub need: Need,
    /// The look the uniforms were written from.
    look: Option<GeoLook>,
    size_u: wgpu::Buffer,
    jfa_u: Vec<wgpu::Buffer>,
    level_u: Vec<wgpu::Buffer>,
    look_u: wgpu::Buffer,
    seed_bg: wgpu::BindGroup,
    jfa_bg: Vec<wgpu::BindGroup>,
    sdf_bg: wgpu::BindGroup,
    cascade_bg: Vec<wgpu::BindGroup>,
    resolve_bg: wgpu::BindGroup,
    finish_bg: wgpu::BindGroup,
    /// Which seed buffer the flood ends in.
    jfa_steps: usize,
}

fn groups(n: u32) -> u32 {
    n.div_ceil(8)
}

impl GeoLit {
    pub fn new(device: &wgpu::Device) -> Self {
        let pipe = |label: &str, wgsl: &str| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(wgsl.into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("geolit"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Self {
            seed: pipe("geolit-seed", SEED),
            jfa: pipe("geolit-jfa", JFA),
            sdf: pipe("geolit-sdf", SDF),
            cascade: pipe("geolit-cascade", CASCADE),
            resolve: pipe("geolit-resolve", RESOLVE),
            finish: pipe("geolit-finish", FINISH),
            sampler,
        }
    }

    /// A layer's worth of everything, for a `w` x `h` drawing whose picture
    /// is `color`, lit on `grid`.
    pub fn make_layer(&self, device: &wgpu::Device, color: &wgpu::TextureView, w: u32, h: u32, grid: Grid) -> LitLayer {
        let tex = |label: &str, tw: u32, th: u32, usage: wgpu::TextureUsages| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d { width: tw, height: th, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba16Float,
                    usage: usage | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        let st = wgpu::TextureUsages::STORAGE_BINDING;
        let light = tex("geolit-light", w, h, wgpu::TextureUsages::RENDER_ATTACHMENT);
        let sdf = tex("geolit-sdf", w, h, st);
        let out = tex("geolit-out", w, h, st);
        let e = grid.extent;
        let cascades = [tex("geolit-cascade-a", 4 * e, e, st), tex("geolit-cascade-b", 4 * e, e, st)];
        let radiance = tex("geolit-radiance", e, e, st);
        let ubuf = |label: &str, size: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let seeds: Vec<wgpu::Buffer> = (0..2)
            .map(|_| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("geolit-seeds"),
                    size: (w as u64 * h as u64 * 16).max(16),
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                })
            })
            .collect();
        // The flood: steps from half the longer side down to 1, then 1
        // again (JFA+1 mends the few pixels the halving gets wrong).
        let mut steps = Vec::new();
        let mut s = (w.max(h).next_power_of_two() / 2).max(1);
        while s >= 1 {
            steps.push(s);
            s /= 2;
        }
        steps.push(1);
        let size_u = ubuf("geolit-size", 16);
        let jfa_u: Vec<wgpu::Buffer> = steps.iter().map(|_| ubuf("geolit-jfa", 16)).collect();
        let level_u: Vec<wgpu::Buffer> = (0..grid.count).map(|_| ubuf("geolit-level", 48)).collect();
        let look_u = ubuf("geolit-look", 48 * 4);
        let entry = |binding: u32, resource| wgpu::BindGroupEntry { binding, resource };
        let tv = wgpu::BindingResource::TextureView;
        let bg = |label: &str, pipeline: &wgpu::ComputePipeline, entries: &[wgpu::BindGroupEntry]| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &pipeline.get_bind_group_layout(0),
                entries,
            })
        };
        let seed_bg = bg(
            "geolit-seed",
            &self.seed,
            &[entry(0, size_u.as_entire_binding()), entry(1, tv(&light)), entry(2, seeds[0].as_entire_binding())],
        );
        let jfa_bg: Vec<wgpu::BindGroup> = jfa_u
            .iter()
            .enumerate()
            .map(|(i, u)| {
                bg(
                    "geolit-jfa",
                    &self.jfa,
                    &[
                        entry(0, u.as_entire_binding()),
                        entry(1, seeds[i % 2].as_entire_binding()),
                        entry(2, seeds[(i + 1) % 2].as_entire_binding()),
                    ],
                )
            })
            .collect();
        let sdf_bg = bg(
            "geolit-sdf",
            &self.sdf,
            &[
                entry(0, size_u.as_entire_binding()),
                entry(1, tv(&light)),
                entry(2, seeds[steps.len() % 2].as_entire_binding()),
                entry(3, tv(&sdf)),
            ],
        );
        // Cascade i writes cascades[i % 2] and reads the other: cascade 0
        // always lands in the first.
        let cascade_bg: Vec<wgpu::BindGroup> = level_u
            .iter()
            .enumerate()
            .map(|(i, u)| {
                bg(
                    "geolit-cascade",
                    &self.cascade,
                    &[
                        entry(0, u.as_entire_binding()),
                        entry(1, tv(&cascades[(i + 1) % 2])),
                        entry(2, tv(&sdf)),
                        entry(3, tv(&light)),
                        entry(4, wgpu::BindingResource::Sampler(&self.sampler)),
                        entry(5, tv(&cascades[i % 2])),
                    ],
                )
            })
            .collect();
        let resolve_bg = bg(
            "geolit-resolve",
            &self.resolve,
            &[
                entry(0, level_u[0].as_entire_binding()),
                entry(1, tv(&cascades[0])),
                entry(2, wgpu::BindingResource::Sampler(&self.sampler)),
                entry(3, tv(&radiance)),
            ],
        );
        let finish_bg = bg(
            "geolit-finish",
            &self.finish,
            &[
                entry(0, look_u.as_entire_binding()),
                entry(1, tv(color)),
                entry(2, tv(&light)),
                entry(3, tv(&sdf)),
                entry(4, tv(&radiance)),
                entry(5, wgpu::BindingResource::Sampler(&self.sampler)),
                entry(6, tv(&out)),
            ],
        );
        LitLayer {
            width: w,
            height: h,
            grid,
            light,
            out,
            need: Need::All,
            look: None,
            size_u,
            jfa_steps: steps.len(),
            jfa_u,
            level_u,
            look_u,
            seed_bg,
            jfa_bg,
            sdf_bg,
            cascade_bg,
            resolve_bg,
            finish_bg,
        }
    }

    /// Write what the passes read for this look, and mark what must rerun.
    /// The drawing changing is the caller's (`need = Need::All`).
    pub fn set_look(&self, queue: &wgpu::Queue, layer: &mut LitLayer, look: &GeoLook) {
        if layer.look.as_ref() == Some(look) {
            return;
        }
        let first = layer.look.is_none();
        let light_changed = layer.look.as_ref().is_none_or(|l| l.hrc_key() != look.hrc_key());
        if first {
            let (w, h) = (layer.width as i32, layer.height as i32);
            queue.write_buffer(&layer.size_u, 0, bytemuck::cast_slice(&[w, h, 0, 0]));
            let mut s = (layer.width.max(layer.height).next_power_of_two() / 2).max(1) as i32;
            for (i, u) in layer.jfa_u.iter().enumerate() {
                let step = if i + 1 == layer.jfa_steps { 1 } else { s };
                queue.write_buffer(u, 0, bytemuck::cast_slice(&[w, h, step, 0]));
                s = (s / 2).max(1);
            }
        }
        if light_changed {
            let g = layer.grid;
            for (i, u) in layer.level_u.iter().enumerate() {
                let v: [f32; 12] = [
                    g.extent as f32,
                    i as f32,
                    g.count as f32,
                    g.spacing,
                    layer.width as f32,
                    layer.height as f32,
                    0.0,
                    0.0,
                    look.sky[0],
                    look.sky[1],
                    look.sky[2],
                    1.0,
                ];
                queue.write_buffer(u, 0, bytemuck::cast_slice(&v));
            }
            layer.need = layer.need.max(Need::Light);
        }
        queue.write_buffer(&layer.look_u, 0, bytemuck::cast_slice(&look.uniform(layer.grid, (layer.width, layer.height))));
        layer.need = layer.need.max(Need::Finish);
        layer.look = Some(look.clone());
    }

    /// Record what the layer needs, after its drawing was recorded this
    /// frame (both targets). Light off skips the cascades: the finish reads
    /// whatever the radiance last held and multiplies it by nothing.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder, layer: &mut LitLayer) {
        let need = layer.need;
        if need == Need::Nothing {
            return;
        }
        let lit = layer.look.as_ref().is_some_and(|l| l.light);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("geolit"), timestamp_writes: None });
        let (gw, gh) = (groups(layer.width), groups(layer.height));
        if need >= Need::All {
            pass.set_pipeline(&self.seed);
            pass.set_bind_group(0, &layer.seed_bg, &[]);
            pass.dispatch_workgroups(gw, gh, 1);
            pass.set_pipeline(&self.jfa);
            for bg in &layer.jfa_bg {
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(gw, gh, 1);
            }
            pass.set_pipeline(&self.sdf);
            pass.set_bind_group(0, &layer.sdf_bg, &[]);
            pass.dispatch_workgroups(gw, gh, 1);
        }
        if need >= Need::Light && lit {
            let e = layer.grid.extent;
            pass.set_pipeline(&self.cascade);
            for i in (0..layer.grid.count as usize).rev() {
                pass.set_bind_group(0, &layer.cascade_bg[i], &[]);
                pass.dispatch_workgroups(groups(4 * e), groups(e), 1);
            }
            pass.set_pipeline(&self.resolve);
            pass.set_bind_group(0, &layer.resolve_bg, &[]);
            pass.dispatch_workgroups(groups(e), groups(e), 1);
        }
        pass.set_pipeline(&self.finish);
        pass.set_bind_group(0, &layer.finish_bg, &[]);
        pass.dispatch_workgroups(gw, gh, 1);
        layer.need = Need::Nothing;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_module_validates() {
        for (name, src) in MODULES {
            let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(src)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                .validate(&module)
                .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        }
    }

    #[test]
    fn the_grid_is_a_power_of_two_and_never_too_big() {
        let g = Grid::new(1920, 1080, 4.0);
        assert_eq!((g.extent, g.count), (512, 9));
        let fine = Grid::new(1920, 1080, 1.0);
        assert_eq!(fine.extent, MAX_EXTENT);
        assert!((fine.spacing - 1920.0 / 1024.0).abs() < 1e-4, "cells coarsened to fit: {}", fine.spacing);
        assert_eq!(Grid::new(16, 16, 4.0).extent, MIN_EXTENT);
    }

    #[test]
    fn every_param_has_a_slot_and_round_trips() {
        let mut l = GeoLook::default();
        for (i, name) in PARAMS.iter().enumerate() {
            let v = 0.25 + i as f32;
            assert!(l.set(name, v), "{name} has no slot");
            assert_eq!(l.get(name), Some(v), "{name}");
        }
        assert!(!l.set("nope", 1.0) && !l.set("sky.a", 1.0));
    }

    #[test]
    fn off_is_inactive_and_any_effect_turns_it_on() {
        let mut l = GeoLook::default();
        assert!(!l.active());
        l.outline = true;
        assert!(l.active());
        let l = GeoLook { offset: 2.0, ..Default::default() };
        assert!(l.active());
    }
}
