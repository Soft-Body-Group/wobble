//! Grain: procedural film grain. With a plate on input 0 it GRAINS the
//! plate; with nothing wired it is a generator that fills its format with
//! grain on a flat `base`, to look at or to Blend in by hand. Input 1 is
//! the usual mask.
//!
//! **The model is sparse convolution, not filtered noise.** Each channel is
//! a field of GRAINS: one per cell of a lattice at the grain size, jittered
//! anywhere in its cell, each a soft disc with its own radius and its own
//! signed density. Film grain is exactly that — silver or dye clumps of
//! varying size, overlapping — and it is why the result reads as grain
//! rather than as the smooth blobs a blurred white noise gives. Three layers
//! per channel:
//!   - the grains at `size`;
//!   - a finer layer at roughly half the size, weighted by `irregularity`
//!     (real grain is a spread of sizes, not one);
//!   - a coarse CLUMPING field, also weighted by `irregularity`, which
//!     modulates the grain's strength across the frame (film clumps).
//!
//! `irregularity` also widens the spread of grain radii. Each channel's
//! size, strength and irregularity are separate, because a colour negative's
//! three dye layers are: blue is typically the coarsest and loudest.
//! `mono` blends each channel towards the channels' mean — 0 is three
//! independent dye layers, 1 is black-and-white grain — renormalised so the
//! strength does not sag as it goes.
//!
//! The field is NORMALISED to unit standard deviation (measured, see
//! [`layer_std`]) so `amount` IS the grain's standard deviation in
//! scene-linear units, whatever the softness or irregularity.
//!
//! **Applied to a plate**, the grain is weighted by the plate's luminance
//! through `response` (shadows · mids · highlights, crossfaded over stops
//! around 18% grey) and then goes on one of two ways:
//!   - `Film`: MULTIPLICATIVE, `in · exp(d / 0.18)`. A negative's grain is a
//!     density (log) fluctuation, so in linear light it scales with the
//!     signal: black stays black and highlights get the same grain in
//!     stops. `amount` is the sd at 18% grey.
//!   - `Add`: `in + d · alpha`, the same grain everywhere — video noise, or
//!     a look that wants grain in the blacks.
//!
//! Unwired, the output is `base + amount · grain`, alpha 1.
//!
//! **Time**: `animated` grain is re-seeded every frame from the tick, which
//! the app resolves into the transient `time` exactly as it does a Noise's —
//! the engine has no idea what time it is. Static grain holds one pattern.
//!
//! Everything is a uniform — sizes, strengths, the seed, the frame: one
//! topology whatever the node holds. Grain sizes are in pixels of the
//! the full-res comp and scale with the proxy, so a proxy shows the same
//! grain, smaller.

use crate::codegen::Emit;
use crate::nodes::NodeDef;

/// Topology tag, clear of every other node's span.
const TOPO: u64 = 0x6D00;

/// A grain's base radius, in cells. With the radius spread at full
/// irregularity it stays under one cell, so a 3x3 neighbourhood sees every
/// grain that can reach a pixel.
const R_BASE: f32 = 0.68;
/// The finer layer's scale and weight at full irregularity.
const FINE_SCALE: f32 = 2.1;
const FINE_W: f32 = 0.7;
/// The clumping field's scale (coarser than the grain) and depth at full
/// irregularity.
const CLUMP_SCALE: f32 = 0.3;
const CLUMP_DEPTH: f32 = 0.5;
/// Softness is clamped off zero: a hard disc aliases, and smoothstep needs
/// its edges apart.
const SOFT_MIN: f32 = 0.05;

/// How grain goes onto a plate. A uniform, not a recompile.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum GrainApply {
    /// Multiplicative, as a negative's density grain is in linear light.
    #[default]
    Film,
    /// Additive: the same grain at every level.
    Add,
}

impl GrainApply {
    pub const ALL: [GrainApply; 2] = [GrainApply::Film, GrainApply::Add];

    pub fn label(self) -> &'static str {
        match self {
            GrainApply::Film => "film (multiply)",
            GrainApply::Add => "add",
        }
    }
}

/// A film stock the panel can load. The values are a starting point shaped
/// like the stock, not a measurement of any one — they are there so the
/// first click lands somewhere film-like.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GrainStock {
    Fine35,
    Mid35,
    Fast35,
    S16,
    Super8,
    Bw400,
}

impl GrainStock {
    pub const ALL: [GrainStock; 6] = [
        GrainStock::Fine35,
        GrainStock::Mid35,
        GrainStock::Fast35,
        GrainStock::S16,
        GrainStock::Super8,
        GrainStock::Bw400,
    ];

    pub fn label(self) -> &'static str {
        match self {
            GrainStock::Fine35 => "35mm 50D",
            GrainStock::Mid35 => "35mm 250D",
            GrainStock::Fast35 => "35mm 500T",
            GrainStock::S16 => "16mm 500T",
            GrainStock::Super8 => "super 8",
            GrainStock::Bw400 => "b&w 400",
        }
    }

    /// (size px at 2K, amount, irregularity, softness, mono), per channel
    /// where it is a triple.
    #[allow(clippy::type_complexity)]
    fn values(self) -> ([f32; 3], [f32; 3], [f32; 3], f32, f32) {
        match self {
            GrainStock::Fine35 => (
                [0.9, 0.8, 1.1],
                [0.012, 0.010, 0.018],
                [0.35, 0.35, 0.45],
                0.6,
                0.15,
            ),
            GrainStock::Mid35 => (
                [1.1, 1.0, 1.4],
                [0.020, 0.017, 0.030],
                [0.45, 0.45, 0.55],
                0.55,
                0.2,
            ),
            GrainStock::Fast35 => (
                [1.4, 1.2, 1.8],
                [0.030, 0.026, 0.045],
                [0.55, 0.55, 0.65],
                0.5,
                0.2,
            ),
            GrainStock::S16 => (
                [2.2, 2.0, 2.8],
                [0.045, 0.040, 0.065],
                [0.6, 0.6, 0.7],
                0.45,
                0.25,
            ),
            GrainStock::Super8 => (
                [3.4, 3.0, 4.2],
                [0.070, 0.060, 0.095],
                [0.75, 0.75, 0.8],
                0.4,
                0.3,
            ),
            GrainStock::Bw400 => (
                [1.6, 1.6, 1.6],
                [0.040, 0.040, 0.040],
                [0.7, 0.7, 0.7],
                0.35,
                1.0,
            ),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(default)
)]
pub struct GrainParams {
    /// The node's format, when it declares one (see `inherit`).
    pub size: (u32, u32),
    /// Take the comp's canvas instead of `size`.
    pub inherit: bool,
    /// Grain size per channel, in pixels of the declared format.
    pub grain: [f32; 3],
    /// Standard deviation per channel, scene-linear.
    pub amount: [f32; 3],
    /// Spread of grain sizes and clumping per channel, 0..1.
    pub irregularity: [f32; 3],
    /// Grain edge: small is crisp, 1 is a soft cone.
    pub softness: f32,
    /// 0 = independent channels (colour grain), 1 = one grain for all (b&w).
    pub mono: f32,
    /// Horizontal stretch of the grain; 1 is round.
    pub aspect: f32,
    /// The level the grain sits on when nothing is wired.
    pub base: f32,
    /// How grain goes onto a wired plate.
    pub apply: GrainApply,
    /// Grain strength in the plate's shadows, mids and highlights.
    pub response: [f32; 3],
    /// Picks the pattern. Whole numbers are distinct patterns.
    pub seed: f32,
    /// A new pattern every frame, as film has.
    pub animated: bool,
    /// The tick being rendered, resolved per frame by the app. TRANSIENT:
    /// never authored, never saved.
    #[cfg_attr(feature = "serde", serde(skip))]
    pub time: f32,
}

impl Default for GrainParams {
    fn default() -> Self {
        let mut p = Self {
            size: (1920, 1080),
            inherit: false,
            grain: [1.0; 3],
            amount: [0.02; 3],
            irregularity: [0.5; 3],
            softness: 0.5,
            mono: 0.2,
            aspect: 1.0,
            base: 0.18,
            apply: GrainApply::Film,
            response: [1.0; 3],
            seed: 0.0,
            animated: true,
            time: 0.0,
        };
        p.apply_stock(GrainStock::Mid35, 1920);
        p
    }
}

impl GrainParams {
    /// Load a stock's look, with grain sizes scaled from 2K to a format
    /// `width` px wide — a 4K scan has twice the pixels per grain. Leaves
    /// the format, base, seed, aspect and animation alone.
    pub fn apply_stock(&mut self, s: GrainStock, width: u32) {
        let (g, a, i, soft, mono) = s.values();
        let k = width.max(1) as f32 / 2048.0;
        self.grain = g.map(|v| v * k);
        self.amount = a;
        self.irregularity = i;
        self.softness = soft;
        self.mono = mono;
    }

    /// The frame number that picks this frame's pattern.
    fn frame(&self) -> i32 {
        if self.animated {
            self.time.round() as i32
        } else {
            0
        }
    }

    /// [`NodeDef::pack`] at the comp's proxy scale, which only the codegen
    /// arm knows: grain sizes are pixels, so a half-res proxy halves them.
    pub fn pack_scaled(&self, v: &mut [f32], proxy: f32) {
        let k = proxy;
        for c in 0..3 {
            v[c] = (self.grain[c] * k).max(0.05);
            v[4 + c] = self.amount[c];
            v[8 + c] = self.irregularity[c].clamp(0.0, 1.0);
        }
        v[3] = self.aspect.max(0.05);
        v[7] = self.base;
        v[11] = self.softness.clamp(SOFT_MIN, 1.0);
        v[12] = self.mono.clamp(0.0, 1.0);
        v[13] = self.seed.floor();
        v[14] = self.frame() as f32;
        v[15] = if self.apply == GrainApply::Film {
            1.0
        } else {
            0.0
        };
        v[16..19].copy_from_slice(&self.response);
    }

    /// The pixel at `p` (pixel centres at +0.5) of a render at `proxy`
    /// scale — the CPU mirror of the shader, for tests and anything that
    /// wants to reason about the grain without a GPU.
    pub fn sample(&self, p: [f32; 2], proxy: f32) -> [f32; 3] {
        let g = self.field(p, proxy);
        [0, 1, 2].map(|c| self.base + self.amount[c] * g[c])
    }

    /// The pixel at `p` of `input` grained — the CPU mirror of the applied
    /// shader. `luma` is the working space's luminance weights.
    pub fn apply_to(&self, input: [f32; 4], p: [f32; 2], proxy: f32, luma: [f32; 3]) -> [f32; 4] {
        let g = self.field(p, proxy);
        let l = input[0] * luma[0] + input[1] * luma[1] + input[2] * luma[2];
        let w = response_weight(self.response, l);
        let mut out = input;
        for c in 0..3 {
            let d = self.amount[c] * g[c] * w;
            out[c] = match self.apply {
                GrainApply::Film => input[c] * (d / 0.18).exp(),
                GrainApply::Add => input[c] + d * input[3],
            };
        }
        out
    }

    /// The normalised grain at `p`, unit sd per channel, mono applied.
    fn field(&self, p: [f32; 2], proxy: f32) -> [f32; 3] {
        let mut v = [0.0; 20];
        self.pack_scaled(&mut v, proxy);
        let seed = v[13] as i32 as u32;
        let frame = v[14] as i32 as u32;
        let (soft, mono) = (v[11], v[12]);
        let mut g = [0.0f32; 3];
        for c in 0..3 {
            let q = [p[0] / (v[c] * v[3]), p[1] / v[c]];
            g[c] = channel(q, key(seed, frame, c as u32), v[8 + c], soft);
        }
        mono_mix(g, mono)
    }
}

/// How much grain a pixel of luminance `l` gets: shadows below ~2.5
/// stops under 18% grey, highlights above ~2.5 over, crossfaded.
fn response_weight(r: [f32; 3], l: f32) -> f32 {
    let x = (l.max(1e-6) / 0.18).log2();
    let sh = 1.0 - smoothstep(-4.0, -1.0, x);
    let hi = smoothstep(1.0, 4.0, x);
    r[0] * sh + r[1] * (1.0 - sh - hi) + r[2] * hi
}

fn key(seed: u32, frame: u32, c: u32) -> [u32; 2] {
    [
        seed.wrapping_mul(0x9E37_79B9) ^ frame.wrapping_mul(0x85EB_CA6B),
        c,
    ]
}

/// PCG4D (Jarzynski & Olano 2020): four well-mixed u32s from four.
fn pcg4d(mut v: [u32; 4]) -> [u32; 4] {
    for x in &mut v {
        *x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    }
    let mix = |v: &mut [u32; 4]| {
        v[0] = v[0].wrapping_add(v[1].wrapping_mul(v[3]));
        v[1] = v[1].wrapping_add(v[2].wrapping_mul(v[0]));
        v[2] = v[2].wrapping_add(v[0].wrapping_mul(v[1]));
        v[3] = v[3].wrapping_add(v[1].wrapping_mul(v[2]));
    };
    mix(&mut v);
    for x in &mut v {
        *x ^= *x >> 16;
    }
    mix(&mut v);
    v
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// One layer of grains at lattice coordinate `q`: zero mean, UNnormalised.
fn layer(q: [f32; 2], k: [u32; 2], irr: f32, soft: f32) -> f32 {
    let c = [q[0].floor(), q[1].floor()];
    let f = [q[0] - c[0], q[1] - c[1]];
    let mut s = 0.0;
    for j in -1..=1 {
        for i in -1..=1 {
            let h = pcg4d([
                (c[0] as i32 + i) as u32,
                (c[1] as i32 + j) as u32,
                k[0],
                k[1],
            ])
            .map(|u| (u >> 8) as f32 * (1.0 / 16_777_216.0));
            let dx = f[0] - i as f32 - h[0];
            let dy = f[1] - j as f32 - h[1];
            let d = (dx * dx + dy * dy).sqrt();
            let r = R_BASE * (1.0 + irr * (0.55 + 0.9 * h[3] - 1.0));
            let kern = 1.0 - smoothstep(r * (1.0 - soft), r, d);
            s += (h[2] * 2.0 - 1.0) * kern;
        }
    }
    s
}

/// The standard deviation of one [`layer`] at this softness and
/// irregularity — exact, not fitted (`layer_std_matches_measurement`).
/// Dividing by it is what makes `amount` mean what it says.
///
/// One grain per unit cell, density uniform in -1..1 (E[a²] = 1/3), so the
/// variance is E[r²]/3 times the kernel's energy on a unit disc,
/// J = 2π[(1-s)²/2 + (13/35)s(1-s) + (3/35)s²] (the smoothstep rim
/// integrates in closed form), and the radius spread gives
/// E[r²] = R²(1 + 0.0675·irr²).
pub fn layer_std(irr: f32, soft: f32) -> f32 {
    let s = soft;
    let j = std::f32::consts::TAU
        * (0.5 * (1.0 - s) * (1.0 - s) + 13.0 / 35.0 * s * (1.0 - s) + 3.0 / 35.0 * s * s);
    R_BASE * (j / 3.0 * (1.0 + 0.0675 * irr * irr)).sqrt()
}

/// One channel's grain at `q`: normalised to unit standard deviation.
fn channel(q: [f32; 2], k: [u32; 2], irr: f32, soft: f32) -> f32 {
    let base = layer(q, k, irr, soft);
    let fine = layer(
        [q[0] * FINE_SCALE + 0.37, q[1] * FINE_SCALE + 0.61],
        [k[0], k[1] + 3],
        irr,
        soft,
    );
    let clump = layer(
        [q[0] * CLUMP_SCALE, q[1] * CLUMP_SCALE],
        [k[0], k[1] + 6],
        0.0,
        1.0,
    );
    let w = FINE_W * irr;
    let sd = layer_std(irr, soft) * (1.0 + w * w).sqrt();
    let depth = CLUMP_DEPTH * irr / layer_std(0.0, 1.0);
    // The clump modulation's mean square, so the product keeps unit sd.
    let m = (1.0 + depth * depth * layer_std(0.0, 1.0).powi(2)).sqrt();
    (base + w * fine) / sd * (1.0 + depth * clump) / m
}

/// Each channel pulled towards the three's mean by `m`, keeping unit sd.
fn mono_mix(g: [f32; 3], m: f32) -> [f32; 3] {
    let mean = (g[0] + g[1] + g[2]) / 3.0;
    let a = 1.0 - 2.0 * m / 3.0;
    let n = (a * a + 2.0 * (m / 3.0) * (m / 3.0)).sqrt();
    g.map(|x| (x + (mean - x) * m) / n)
}

const WGSL_FN: &str = "grain_field";
const WGSL_APPLY: &str = "grain_apply";

fn wgsl() -> String {
    format!(
        "// Grain: sparse convolution — a jittered soft disc per lattice cell,
// each with its own radius and signed density. See engine::nodes::grain.
fn grain_pcg4d(v0: vec4u) -> vec4u {{
    var v = v0 * 1664525u + 1013904223u;
    v.x += v.y * v.w; v.y += v.z * v.x; v.z += v.x * v.y; v.w += v.y * v.z;
    v = v ^ (v >> vec4u(16u));
    v.x += v.y * v.w; v.y += v.z * v.x; v.z += v.x * v.y; v.w += v.y * v.z;
    return v;
}}

fn grain_layer(q: vec2f, k: vec2u, irr: f32, soft: f32) -> f32 {{
    let c = floor(q);
    let f = q - c;
    let ci = vec2i(c);
    var s = 0.0;
    for (var j = -1; j <= 1; j = j + 1) {{
        for (var i = -1; i <= 1; i = i + 1) {{
            let u = grain_pcg4d(vec4u(bitcast<vec2u>(ci + vec2i(i, j)), k));
            let h = vec4f(u >> vec4u(8u)) * (1.0 / 16777216.0);
            let d = length(f - vec2f(f32(i), f32(j)) - h.xy);
            let r = {R_BASE:?} * (1.0 + irr * (0.55 + 0.9 * h.w - 1.0));
            let kern = 1.0 - smoothstep(r * (1.0 - soft), r, d);
            s += (h.z * 2.0 - 1.0) * kern;
        }}
    }}
    return s;
}}

fn grain_std(irr: f32, soft: f32) -> f32 {{
    let j = 6.2831853 * (0.5 * (1.0 - soft) * (1.0 - soft) + 13.0 / 35.0 * soft * (1.0 - soft)
        + 3.0 / 35.0 * soft * soft);
    return {R_BASE:?} * sqrt(j / 3.0 * (1.0 + 0.0675 * irr * irr));
}}

fn grain_channel(q: vec2f, k: vec2u, irr: f32, soft: f32) -> f32 {{
    let base = grain_layer(q, k, irr, soft);
    let fine = grain_layer(q * {FINE_SCALE:?} + vec2f(0.37, 0.61), k + vec2u(0u, 3u), irr, soft);
    let clump = grain_layer(q * {CLUMP_SCALE:?}, k + vec2u(0u, 6u), 0.0, 1.0);
    let w = {FINE_W:?} * irr;
    let sd = grain_std(irr, soft) * sqrt(1.0 + w * w);
    let s1 = grain_std(0.0, 1.0);
    let depth = {CLUMP_DEPTH:?} * irr / s1;
    let m = sqrt(1.0 + depth * depth * s1 * s1);
    return (base + w * fine) / sd * (1.0 + depth * clump) / m;
}}

// The normalised grain at p: unit sd per channel, mono applied.
fn {WGSL_FN}(p: vec2f, slot: u32) -> vec3f {{
    let sz = P.v[slot];
    let amt = P.v[slot + 1u];
    let irr = P.v[slot + 2u];
    let e = P.v[slot + 3u];
    let seed = bitcast<u32>(i32(e.y));
    let frame = bitcast<u32>(i32(e.z));
    let k0 = seed * 0x9E3779B9u ^ frame * 0x85EBCA6Bu;
    let soft = irr.w;
    var g = vec3f(0.0);
    for (var c = 0u; c < 3u; c = c + 1u) {{
        let q = vec2f(p.x / (sz[c] * sz.w), p.y / sz[c]);
        g[c] = grain_channel(q, vec2u(k0, c), irr[c], soft);
    }}
    let m = e.x;
    let a = 1.0 - 2.0 * m / 3.0;
    let n = sqrt(a * a + 2.0 * (m / 3.0) * (m / 3.0));
    let mean = (g.x + g.y + g.z) / 3.0;
    return mix(g, vec3f(mean), m) / n;
}}

// A plate grained: weighted by its luminance through the response, then
// multiplied on (film) or added (scaled by alpha, so clear stays clear).
fn {WGSL_APPLY}(i: vec4f, p: vec2f, slot: u32, luma: vec3f) -> vec4f {{
    let g = {WGSL_FN}(p, slot);
    let amt = P.v[slot + 1u].rgb;
    let film = P.v[slot + 3u].w;
    let r = P.v[slot + 4u];
    let x = log2(max(dot(i.rgb, luma), 1e-6) / 0.18);
    let sh = 1.0 - smoothstep(-4.0, -1.0, x);
    let hi = smoothstep(1.0, 4.0, x);
    let w = r.x * sh + r.y * (1.0 - sh - hi) + r.z * hi;
    let d = amt * g * w;
    let rgb = select(i.rgb + d * i.a, i.rgb * exp(d / 0.18), film > 0.5);
    return vec4f(rgb, i.a);
}}
"
    )
}

impl NodeDef for GrainParams {
    const NAME: &'static str = "Grain";
    /// The plate, then the mask.
    const INPUTS: usize = 2;
    const MASK_INPUT: Option<usize> = Some(1);
    /// (size rgb, aspect), (amount rgb, base), (irregularity rgb, softness),
    /// (mono, seed, frame, film), (response shadows/mids/highs, -).
    const VEC4_SLOTS: usize = 5;

    fn topo_tag(&self) -> u64 {
        TOPO
    }

    fn pack(&self, v: &mut [f32]) {
        self.pack_scaled(v, 1.0);
    }

    fn emit(&self, e: &mut Emit) -> String {
        let slot = e.slot();
        let pos = e.pos().to_string();
        let (_, fns) = e.color_helpers();
        fns.entry(WGSL_FN.to_string()).or_insert_with(wgsl);
        let v = e.fresh();
        // Whether a plate is wired picks the code; everything else is a
        // uniform.
        if !e.has_input(0) {
            e.line(format_args!(
                "let {v} = vec4f(P.v[{slot}u + 1u].w + P.v[{slot}u + 1u].rgb * {WGSL_FN}({pos}, {slot}u), 1.0);"
            ));
            return v;
        }
        let input = e.input(0);
        let [lr, lg, lb] = e.working().primaries.luma_weights();
        e.line(format_args!(
            "let {v} = {WGSL_APPLY}({input}, {pos}, {slot}u, vec3f({lr:?}, {lg:?}, {lb:?}));"
        ));
        e.masked(Self::MASK_INPUT, &input, v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mean and sd of each channel over an `n`-px square.
    fn stats(p: &GrainParams, n: usize) -> ([f32; 3], [f32; 3]) {
        let mut s = [0.0f64; 3];
        let mut s2 = [0.0f64; 3];
        for y in 0..n {
            for x in 0..n {
                let v = p.sample([x as f32 + 0.5, y as f32 + 0.5], 1.0);
                for c in 0..3 {
                    s[c] += v[c] as f64;
                    s2[c] += (v[c] as f64).powi(2);
                }
            }
        }
        let t = (n * n) as f64;
        let mean = s.map(|v| (v / t) as f32);
        let sd = [0, 1, 2].map(|c| ((s2[c] / t - (s[c] / t).powi(2)).sqrt()) as f32);
        (mean, sd)
    }

    #[test]
    fn layer_std_matches_measurement() {
        for soft in [0.05, 0.3, 0.6, 1.0] {
            for irr in [0.0, 0.5, 1.0] {
                let (mut s, mut s2) = (0.0f64, 0.0f64);
                let n = 400;
                for y in 0..n {
                    for x in 0..n {
                        let v =
                            layer([x as f32 * 0.173, y as f32 * 0.171], [7, 1], irr, soft) as f64;
                        s += v;
                        s2 += v * v;
                    }
                }
                let t = (n * n) as f64;
                let sd = (s2 / t - (s / t).powi(2)).sqrt() as f32;
                let fit = layer_std(irr, soft);
                assert!(
                    (sd / fit - 1.0).abs() < 0.04,
                    "soft {soft} irr {irr}: measured {sd}, fit {fit}"
                );
            }
        }
    }

    #[test]
    fn amount_is_the_standard_deviation_and_base_the_mean() {
        for stock in GrainStock::ALL {
            let mut p = GrainParams {
                size: (2048, 1080),
                ..Default::default()
            };
            p.apply_stock(stock, 2048);
            p.amount = [0.05, 0.1, 0.2];
            let (mean, sd) = stats(&p, 256);
            for c in 0..3 {
                assert!(
                    (mean[c] - p.base).abs() < 0.1 * p.amount[c],
                    "{stock:?} c{c}: mean {}",
                    mean[c]
                );
                assert!(
                    (sd[c] / p.amount[c] - 1.0).abs() < 0.12,
                    "{stock:?} c{c}: sd {} vs {}",
                    sd[c],
                    p.amount[c]
                );
            }
        }
    }

    #[test]
    fn mono_one_is_black_and_white_and_zero_is_colour() {
        let mut p = GrainParams {
            mono: 1.0,
            amount: [0.1; 3],
            ..Default::default()
        };
        for i in 0..50 {
            let v = p.sample([i as f32 * 3.7 + 0.5, i as f32 * 1.3 + 0.5], 1.0);
            assert!((v[0] - v[1]).abs() < 1e-5 && (v[1] - v[2]).abs() < 1e-5);
        }
        p.mono = 0.0;
        let v = p.sample([10.5, 20.5], 1.0);
        assert!((v[0] - v[1]).abs() > 1e-4 || (v[1] - v[2]).abs() > 1e-4);
    }

    #[test]
    fn animated_grain_changes_every_frame_and_static_grain_holds() {
        let at = |p: &GrainParams| p.sample([100.5, 50.5], 1.0);
        let mut p = GrainParams::default();
        let a = at(&p);
        p.time = 1.0;
        assert_ne!(a, at(&p), "a new frame is a new pattern");
        p.animated = false;
        let b = at(&p);
        p.time = 7.0;
        assert_eq!(b, at(&p), "static grain ignores the frame");
    }

    #[test]
    fn bigger_grain_is_smoother_between_neighbours() {
        // Neighbour correlation rises with size: that is what size means.
        let corr = |g: f32| {
            let p = GrainParams {
                grain: [g; 3],
                mono: 0.0,
                ..Default::default()
            };
            let (mut num, mut den) = (0.0f64, 0.0f64);
            for y in 0..64 {
                for x in 0..64 {
                    let a = p.sample([x as f32 + 0.5, y as f32 + 0.5], 1.0)[1] - p.base;
                    let b = p.sample([x as f32 + 1.5, y as f32 + 0.5], 1.0)[1] - p.base;
                    num += (a * b) as f64;
                    den += (a * a) as f64;
                }
            }
            num / den
        };
        assert!(
            corr(3.0) > corr(1.0) + 0.2,
            "{} vs {}",
            corr(3.0),
            corr(1.0)
        );
    }

    #[test]
    fn film_grain_scales_with_the_plate_and_add_does_not() {
        let luma = [0.2126, 0.7152, 0.0722];
        let p = GrainParams {
            amount: [0.05; 3],
            ..Default::default()
        };
        let sd = |p: &GrainParams, level: f32| {
            let (mut s, mut s2) = (0.0f64, 0.0f64);
            let n = 128;
            for y in 0..n {
                for x in 0..n {
                    let v = p.apply_to(
                        [level, level, level, 1.0],
                        [x as f32 + 0.5, y as f32 + 0.5],
                        1.0,
                        luma,
                    )[1] as f64;
                    s += v;
                    s2 += v * v;
                }
            }
            let t = (n * n) as f64;
            (s2 / t - (s / t).powi(2)).sqrt() as f32
        };
        // Film: the sd at 18% grey is `amount`, and it scales with the
        // signal — black stays black.
        assert!((sd(&p, 0.18) / 0.05 - 1.0).abs() < 0.15, "{}", sd(&p, 0.18));
        assert!(sd(&p, 0.72) > 3.0 * sd(&p, 0.18));
        assert_eq!(sd(&p, 0.0), 0.0);
        // Add: the same everywhere in the mids.
        let a = GrainParams {
            apply: GrainApply::Add,
            ..p.clone()
        };
        assert!((sd(&a, 0.36) / sd(&a, 0.18) - 1.0).abs() < 0.05);
    }

    #[test]
    fn the_response_weights_shadows_mids_and_highlights() {
        let r = [0.0, 1.0, 2.0];
        assert!(response_weight(r, 0.18 / 64.0) < 1e-3);
        assert!((response_weight(r, 0.18) - 1.0).abs() < 1e-5);
        assert!((response_weight(r, 0.18 * 64.0) - 2.0).abs() < 1e-3);
    }

    #[test]
    fn a_clear_pixel_stays_clear() {
        let luma = [0.2126, 0.7152, 0.0722];
        for apply in GrainApply::ALL {
            let p = GrainParams {
                apply,
                amount: [0.2; 3],
                ..Default::default()
            };
            assert_eq!(
                p.apply_to([0.0; 4], [3.5, 7.5], 1.0, luma),
                [0.0; 4],
                "{apply:?}"
            );
        }
    }

    #[test]
    fn nothing_moves_the_topology() {
        let a = GrainParams::default();
        let mut b = a.clone();
        b.apply_stock(GrainStock::Bw400, 4096);
        b.animated = false;
        b.seed = 9.0;
        b.time = 12.0;
        b.apply = GrainApply::Add;
        b.response = [0.3, 1.0, 0.5];
        assert_eq!(a.topo_tag(), b.topo_tag());
    }

    #[test]
    fn a_4k_stock_has_twice_the_grain_size_of_2k() {
        let mut a = GrainParams::default();
        let mut b = a.clone();
        a.apply_stock(GrainStock::Fast35, 2048);
        b.apply_stock(GrainStock::Fast35, 4096);
        assert_eq!(b.grain, a.grain.map(|v| v * 2.0));
    }
}
