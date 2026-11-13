//! The Particles node's GPU half: a compute simulation with a checkpoint
//! cache, and the streaks it draws.
//!
//! **State lives on the GPU and never comes back.** One storage buffer per
//! node holds the clock (vec4 0, `x` = the frame the state is AT) and two
//! vec4s a slot (three, with `(z, vz, -, -)` for depth): `(pos, vel)` and `(age, life, rnd, emitter + 1)` — 0 in
//! that last lane is a dead slot. Three kernels touch it, all ordered by
//! the encoder (nothing is written with `queue.write_buffer` mid-sim, so a
//! catch-up of a thousand frames is one encoder and one submit):
//!
//!  - `reset`: every slot dead, the clock at the sim's origin;
//!  - `step`: births scheduled by the frame, then the forces, `substeps`
//!    times;
//!  - `tick`: the clock moves on one frame.
//!
//! **Scrubbing is checkpoints.** Every `interval` frames (from the origin)
//! the state is copied aside. A frame is reached from the newest of
//! (the live state, the best checkpoint) at or before it, stepping
//! forward; going back restores a checkpoint, never re-runs from the top
//! unless there is none. The checkpoints fit a byte budget by THINNING —
//! double the interval, drop every other one — so a long shot keeps even
//! coverage instead of losing its far end, and a node too big for any
//! checkpoint still works, just re-simulating from the origin.
//!
//! Anything the simulation reads changing ([`Compiled::sim_hash`]) throws
//! the state and the checkpoints away; a Render edit only re-draws.

use crate::nodes::particles::{
    Compiled, CHARGE_BASE, EMIT_BASE, EMIT_VEC4S, FORCE_BASE, FORCE_VEC4S, RENDER_VEC4S, SIM_VEC4S,
};
use std::collections::BTreeMap;

/// Bytes of checkpoints one node may keep.
pub const CHECKPOINT_BUDGET: u64 = 512 << 20;
/// Frames between checkpoints before any thinning.
pub const CHECKPOINT_INTERVAL: i64 = 8;

const COMMON_WGSL: &str = r#"
const PI: f32 = 3.14159265;
const TAU: f32 = 6.2831853;

fn pcg(x: u32) -> u32 {
    let s = x * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}
fn rnd(a: u32, k: u32) -> f32 {
    return f32(pcg(a ^ pcg(k * 0x9e3779b9u + 0x85ebca6bu)) >> 8u) / 16777216.0;
}
"#;

fn sim_wgsl() -> String {
    format!(
        "const EB: u32 = {EMIT_BASE}u;\nconst EV: u32 = {EMIT_VEC4S}u;\nconst FB: u32 = {FORCE_BASE}u;\n\
         const FV: u32 = {FORCE_VEC4S}u;\nconst CB: u32 = {CHARGE_BASE}u;\n{COMMON_WGSL}{SIM_BODY}"
    )
}

const SIM_BODY: &str = r#"
@group(0) @binding(0) var<storage, read> sim: array<vec4f>;
@group(0) @binding(1) var<storage, read_write> st: array<vec4f>;

fn slots() -> u32 { return bitcast<u32>(sim[0].z); }

@compute @workgroup_size(256)
fn reset(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i == 0u) { st[0] = vec4f(sim[1].x, 0.0, 0.0, 0.0); }
    if (i >= slots()) { return; }
    st[1u + 3u * i] = vec4f(0.0);
    st[2u + 3u * i] = vec4f(0.0);
    st[3u + 3u * i] = vec4f(0.0);
}

@compute @workgroup_size(1)
fn tick() {
    st[0].x = st[0].x + 1.0;
}

// Rotate a y-down pixel vector counter-clockwise ON SCREEN by `a`.
fn turn(v: vec2f, a: f32) -> vec2f {
    let c = cos(a);
    let s = sin(a);
    return vec2f(v.x * c + v.y * s, -v.x * s + v.y * c);
}

// A screen direction from degrees-as-radians (0 right, +90 up).
fn dir(a: f32) -> vec2f { return vec2f(cos(a), -sin(a)); }

// Curl noise: the perpendicular gradient of a sum of travelling waves, so
// the flow neither gathers nor spreads.
fn curl(p: vec2f, t: f32, scale: f32, speed: f32, seed: u32) -> vec2f {
    var g = vec2f(0.0);
    for (var k = 0u; k < 6u; k = k + 1u) {
        let a = rnd(seed, k) * TAU;
        let hi = k >= 3u;
        let oct = select(1.0, 2.3, hi);
        let kf = TAU / scale * oct * (0.7 + 0.6 * rnd(seed, k + 10u));
        let kv = vec2f(cos(a), sin(a)) * kf;
        let ph = rnd(seed, k + 20u) * TAU + t * speed * TAU * (0.5 + rnd(seed, k + 30u));
        let amp = select(1.0, 0.45, hi);
        g += (kv / kf) * cos(dot(kv, p) + ph) * amp;
    }
    return vec2f(g.y, -g.x) * 0.6;
}

@compute @workgroup_size(256)
fn step(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= slots()) { return; }
    let ne = bitcast<u32>(sim[0].x);
    let nf = bitcast<u32>(sim[0].y);
    var e = 0u;
    for (var k = 0u; k < ne; k = k + 1u) {
        let a = sim[EB + k * EV];
        let s = bitcast<u32>(a.x);
        if (i >= s && i < s + bitcast<u32>(a.y)) { e = k; break; }
    }
    let eb = EB + e * EV;
    let ea = sim[eb];
    let local = i - bitcast<u32>(ea.x);
    let f = st[0].x + 1.0;
    var p0 = st[1u + 3u * i];
    var p1 = st[2u + 3u * i];
    // Depth: (z towards the eye, its velocity). Only Field Lines on a
    // sphere moves it; everything else is flat.
    var p2 = st[3u + 3u * i];

    // --- birth: slot `local` is reborn every `period` frames, offset by
    // local / rate, so the schedule is a function of the frame alone.
    let o = f32(local) / ea.z;
    let g1 = floor((f - o) / ea.w);
    let g0 = floor((f - 1.0 - o) / ea.w);
    let newborn = g1 != g0;
    if (newborn) {
        let frac = clamp((f - o) - g1 * ea.w, 0.0, 1.0);
        let eseed = bitcast<u32>(sim[eb + 5u].x);
        let gen = u32(i32(g1) + 1048576);
        let h = (local * 2654435761u) ^ pcg(gen * 0x9e3779b9u + eseed);
        let b = sim[eb + 1u];
        let c = sim[eb + 2u];
        let d = sim[eb + 3u];
        let q = sim[eb + 4u];
        let life = c.z * (1.0 + c.w * (rnd(h, 1u) * 2.0 - 1.0));
        var u = rnd(h, 2u);
        let v = rnd(h, 3u);
        if (q.y > 0.0 && rnd(h, 4u) < q.w) {
            let j = u32(rnd(h, 5u) * q.y);
            u = fract(rnd(eseed, j + 1000u) + (rnd(h, 6u) + rnd(h, 7u) - 1.0) * q.z);
        }
        let shape = u32(c.y + 0.5);
        let ed = dir(d.z);
        var lp = vec2f(0.0);
        var nrm = ed;
        let th = u * TAU;
        if (shape == 1u) {
            lp = vec2f((u - 0.5) * b.z, 0.0);
            nrm = vec2f(0.0, -1.0);
        } else if (shape == 2u) {
            nrm = vec2f(cos(th), -sin(th));
            lp = nrm * b.z;
        } else if (shape == 3u) {
            nrm = vec2f(cos(th), -sin(th));
            lp = nrm * b.z * sqrt(v);
        } else if (shape == 4u) {
            lp = vec2f((u - 0.5) * b.z, (v - 0.5) * b.w);
            nrm = vec2f(0.0, -1.0);
        }
        lp = turn(lp, c.x);
        if (shape == 1u || shape == 4u) { nrm = turn(nrm, c.x); }
        var base = mix(ed, nrm, q.x);
        if (length(base) < 1e-4) { base = ed; }
        let ang = atan2(base.y, base.x) + (rnd(h, 8u) - 0.5) * d.w;
        let spd = d.x * (1.0 + d.y * (rnd(h, 9u) * 2.0 - 1.0));
        let vel = vec2f(cos(ang), sin(ang)) * spd;
        p0 = vec4f(b.xy + lp + vel * frac, vel);
        p1 = vec4f(frac, life, rnd(h, 10u), select(0.0, f32(e + 1u), life > 0.0));
        p2 = vec4f(0.0);
    }

    if (p1.w > 0.0) {
        let ns = bitcast<u32>(sim[0].w);
        let dt = sim[1].y;
        var pos = p0.xy;
        var vel = p0.zw;
        var age = p1.x;
        var zz = p2.x;
        var vz = p2.y;
        var alive = true;
        for (var s = 0u; s < ns; s = s + 1u) {
            let t = f - 1.0 + (f32(s) + 0.5) * dt;
            var adv = vec2f(0.0);
            for (var j = 0u; j < nf; j = j + 1u) {
                let fb = FB + j * FV;
                let fa = sim[fb];
                if (((bitcast<u32>(fa.y) >> e) & 1u) == 0u) { continue; }
                let fx = sim[fb + 1u];
                let fy = sim[fb + 2u];
                let fz = sim[fb + 3u];
                let kind = u32(fa.x + 0.5);
                let str = fa.z;
                let d = pos - fx.xy;
                let l = max(length(d), 1e-3);
                var fall = 1.0;
                if (fx.w > 0.0) { fall = clamp(1.0 - l / fx.w, 0.0, 1.0); }
                if (kind == 0u) {
                    vel += dir(fx.z) * str * dt;
                } else if (kind == 1u) {
                    vel += d / l * str * fall * dt;
                } else if (kind == 2u) {
                    vel += vec2f(d.y, -d.x) / l * str * fall * dt;
                } else if (kind == 3u) {
                    vel *= pow(max(1.0 - clamp(str, 0.0, 1.0), 0.0), dt);
                } else if (kind == 4u) {
                    let cv = curl(pos, t, fy.x, fy.y, bitcast<u32>(fy.z)) * str;
                    vel += cv * (1.0 - fa.w) * dt;
                    adv += cv * fa.w;
                } else if (kind == 5u) {
                    var ef = vec3f(0.0);
                    let c0 = bitcast<u32>(fy.w);
                    let cn = bitcast<u32>(fz.x);
                    let mid = vec3f(fx.xy, 0.0);
                    // Born on a pole: moved to a random one, a little off
                    // it, set to flow away from it.
                    if (newborn && s == 0u && fz.w > 0.0 && cn > 0u) {
                        let h = bitcast<u32>(p1.z * 16777216.0) ^ (i * 747796405u);
                        let k = min(u32(rnd(h, 1u) * f32(cn)), cn - 1u);
                        let ch = sim[CB + 2u * (c0 + k)];
                        let link = sim[CB + 2u * (c0 + k) + 1u].x;
                        // Which of the pole's lines: one of `strands`,
                        // a hair of jitter so a thread has some body.
                        var u = rnd(h, 2u);
                        var lid = h;
                        if (fy.x > 0.0) {
                            let si = min(floor(u * fy.x), fy.x - 1.0);
                            // Lines are spread unevenly (each has its own offset in
                            // its slot), so a bundle is not a ruled fan.
                            lid = (k * 8191u + u32(si)) ^ bitcast<u32>(fy.z);
                            u = (si + 0.15 + 0.7 * rnd(lid, 5u) + (rnd(h, 3u) - 0.5) * 0.1) / fy.x;
                        }
                        var outw = ch.xyz - mid;
                        let ol = length(outw);
                        if (ol > 1e-3) { outw = outw / ol; } else { outw = vec3f(0.0, -1.0, 0.0); }
                        var dv = outw;
                        if (link == 0.0) {
                            // Unpaired: a narrow fan straight out — across
                            // the ring when flat, a cone on a sphere.
                            var t = cross(outw, vec3f(0.0, 0.0, 1.0));
                            let tl = length(t);
                            if (tl > 1e-3) { t = t / tl; } else { t = vec3f(1.0, 0.0, 0.0); }
                            let bn = cross(outw, t);
                            let a = tan((u * 2.0 - 1.0) * fy.y * 0.12);
                            let ph = rnd(lid, 6u) * PI * fz.y;
                            dv = normalize(outw + (t * cos(ph) + bn * sin(ph)) * a);
                        } else {
                            let pk = u32(i32(k) + i32(sign(link)));
                            var ax = sim[CB + 2u * (c0 + pk)].xyz - ch.xyz;
                            let al = length(ax);
                            if (al > 1e-3) { ax = ax / al; } else { ax = vec3f(1.0, 0.0, 0.0); }
                            // Off the chord by at least a quarter of the fan:
                            // the flattest lines would run under the surface.
                            var a = (0.25 + 0.75 * u) * fy.y;
                            var perp = vec3f(-ax.y, ax.x, 0.0);
                            if (abs(link) > 1.5) {
                                // Flat, on the face: seen from above, so
                                // shallower, and the pair leans to one side.
                                a = a * 0.6;
                                // (`ax` points the other way from the partner, so
                                // its perp is flipped back to agree.)
                                if (link < 0.0) { perp = -perp; }
                                if (rnd(min(k, pk) ^ bitcast<u32>(fy.z), 9u) < 0.5) { perp = -perp; }
                            } else {
                                // The loop stands off the surface: its plane
                                // holds the chord and the outward normal,
                                // leaning a little on a sphere.
                                var up = outw - ax * dot(outw, ax);
                                let ul = length(up);
                                if (ul > 1e-3) { up = up / ul; } else { up = perp; }
                                let lean = (rnd(min(k, pk) ^ bitcast<u32>(fy.z), 9u) * 2.0 - 1.0) * 0.5 * fz.y;
                                perp = normalize(up + cross(ax, up) * lean);
                            }
                            dv = ax * cos(a) + perp * sin(a);
                        }
                        pos = ch.xy + dv.xy * fz.w;
                        zz = ch.z + dv.z * fz.w;
                        // Sign in which half, the line's own random in the rest.
                        let lr = rnd(lid, 7u) * 0.499;
                        p1.z = select(0.5 + lr, lr, ch.w < 0.0);
                        // Remember the pole (its place in the table),
                        // above the emitter's six bits.
                        p1.w = f32(e + 1u) + 64.0 * f32(c0 + k + 1u);
                    }
                    // A particle that reaches a pole after leaving its own
                    // is absorbed (it would stall there, a bright dot).
                    let sink = max(2.0, fz.w * 0.35);
                    let root = u32(p1.w + 0.5) >> 6u;
                    let p3 = vec3f(pos, zz);
                    if (fx.z > 0.5 && root > c0 && root <= c0 + cn) {
                        // Its own pair only: the lines are circles through
                        // both poles (a ray, off a pole with no partner).
                        let ch = sim[CB + 2u * (root - 1u)];
                        let link = sim[CB + 2u * (root - 1u) + 1u].x;
                        let r = p3 - ch.xyz;
                        ef = ch.w * r / (dot(r, r) + 4.0);
                        if (link != 0.0) {
                            let pc = sim[CB + 2u * u32(i32(root) - 1 + i32(sign(link)))];
                            let rp = p3 - pc.xyz;
                            let r2 = dot(rp, rp);
                            if (r2 < sink * sink) { alive = false; }
                            ef += pc.w * rp / (r2 + 4.0);
                        }
                    } else {
                        for (var k = 0u; k < cn; k = k + 1u) {
                            let ch = sim[CB + 2u * (c0 + k)];
                            let r = p3 - ch.xyz;
                            let r2 = dot(r, r);
                            if (r2 < sink * sink && age > 3.0) { alive = false; }
                            ef += ch.w * r / (r2 + 4.0);
                        }
                    }
                    let el = length(ef);
                    if (el > 1e-12) {
                        var sg = 1.0;
                        if ((fz.z > 0.5 || fz.w > 0.0) && p1.z < 0.5) { sg = -1.0; }
                        let fd = ef / el * sg;
                        let v3 = mix(vec3f(vel, vz) + fd * str * dt, fd * str, fa.w);
                        vel = v3.xy;
                        vz = v3.z;
                    }
                } else if (kind == 6u) {
                    let inside = l < fx.w;
                    if (inside != (fz.y > 0.5)) { alive = false; }
                }
            }
            pos += (vel + adv) * dt;
            zz += vz * dt;
            age += dt;
        }
        if (age >= p1.y) { alive = false; }
        p0 = vec4f(pos, vel);
        p2 = vec4f(zz, vz, 0.0, 0.0);
        p1 = vec4f(age, p1.y, p1.z, select(0.0, p1.w, alive));
    }
    st[1u + 3u * i] = p0;
    st[2u + 3u * i] = p1;
    st[3u + 3u * i] = p2;
}
"#;

const DRAW_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read> st: array<vec4f>;
struct R { v: array<vec4f, 6> };
@group(0) @binding(1) var<uniform> ru: R;

struct VO {
    @builtin(position) pos: vec4f,
    @location(0) q: vec2f,
    @location(1) @interpolate(flat) col: vec4f,
    @location(2) @interpolate(flat) lw: vec2f,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VO {
    var o: VO;
    o.pos = vec4f(2.0, 2.0, 0.0, 1.0);
    o.q = vec2f(0.0);
    o.col = vec4f(0.0);
    o.lw = vec2f(1.0);
    let p0 = st[1u + 3u * ii];
    let p1 = st[2u + 3u * ii];
    let p2 = st[3u + 3u * ii];
    if (p1.w <= 0.0) { return o; }
    let e = (u32(p1.w + 0.5) & 63u) - 1u;
    if (((bitcast<u32>(ru.v[0].w) >> e) & 1u) == 0u) { return o; }
    let t = clamp(p1.x / max(p1.y, 1e-3), 0.0, 1.0);
    let fi = ru.v[3].w;
    let fo = ru.v[4].x;
    var fade = 1.0;
    if (fi > 0.0) { fade *= smoothstep(0.0, 1.0, t / fi); }
    if (fo > 0.0) { fade *= smoothstep(0.0, 1.0, (1.0 - t) / fo); }
    let lr = fract(p1.z * 2.0);
    let col = mix(ru.v[1], ru.v[2], t) * fade * mix(1.0, 0.12 + lr * lr * lr * 3.2, ru.v[4].z);
    if (col.w <= 1e-7) { return o; }
    let proxy = ru.v[0].z;
    let p = p0.xy * proxy;
    // Behind the disc: hidden, with a pixel of edge at the limb.
    if (ru.v[4].w > 0.5 && p2.x < 0.0) {
        let dr = length(p - ru.v[5].xy);
        if (dr < ru.v[5].z) { return o; }
    }
    let sp = length(p0.zw);
    var along = vec2f(1.0, 0.0);
    if (sp > 1e-5) { along = p0.zw / sp; }
    let across = vec2f(-along.y, along.x);
    let len = ru.v[3].x + ru.v[3].y * sp;
    let w = ru.v[3].z;
    let hw = w * 1.5;
    let hl = len * 0.5 + hw;
    let corner = vec2f(f32(vi & 1u) * 2.0 - 1.0, f32(vi >> 1u) * 2.0 - 1.0);
    let qp = p + along * (corner.x * hl) + across * (corner.y * hw);
    o.pos = vec4f(qp.x / ru.v[0].x * 2.0 - 1.0, 1.0 - qp.y / ru.v[0].y * 2.0, 0.0, 1.0);
    o.q = vec2f(corner.x * hl, corner.y * 1.5);
    o.col = col;
    o.lw = vec2f(len, w);
    return o;
}

@fragment
fn fs(o: VO) -> @location(0) vec4f {
    let d = max(abs(o.q.x) - o.lw.x * 0.5, 0.0) / o.lw.y;
    return o.col * exp(-(o.q.y * o.q.y + d * d) * 2.2);
}
"#;

/// The shared pipelines; per-node state is a [`ParticleLayer`].
pub struct ParticleRaster {
    reset: wgpu::ComputePipeline,
    step: wgpu::ComputePipeline,
    tick: wgpu::ComputePipeline,
    sim_layout: wgpu::BindGroupLayout,
    draw: wgpu::RenderPipeline,
    draw_layout: wgpu::BindGroupLayout,
}

struct RenderSlot {
    buf: wgpu::Buffer,
    bind: wgpu::BindGroup,
}

pub struct ParticleLayer {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub width: u32,
    pub height: u32,
    pub dirty: bool,
    sim_buf: wgpu::Buffer,
    state: Option<wgpu::Buffer>,
    sim_bind: Option<wgpu::BindGroup>,
    slots: u32,
    origin: i64,
    sim_hash: u64,
    renders: Vec<RenderSlot>,
    render_hash: u64,
    /// The frame `state` holds, when it holds one for `sim_hash`.
    at: Option<i64>,
    /// The frame asked for.
    want: i64,
    /// What the texture shows: (frame, render hash).
    drawn: Option<(i64, u64)>,
    checkpoints: BTreeMap<i64, wgpu::Buffer>,
    interval: i64,
    /// Steps run by the last encode, for tests.
    pub last_steps: u64,
}

impl ParticleLayer {
    fn state_bytes(&self) -> u64 {
        16 + self.slots as u64 * 48
    }

    /// Checkpoints held, for tests.
    pub fn checkpoints(&self) -> Vec<i64> {
        self.checkpoints.keys().copied().collect()
    }
}

fn storage(binding: u32, read_only: bool, vis: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: vis,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

impl ParticleRaster {
    pub fn new(device: &wgpu::Device) -> Self {
        let sim_src = sim_wgsl();
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("particles-sim"),
            source: wgpu::ShaderSource::Wgsl(sim_src.into()),
        });
        let sim_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("particles-sim"),
            entries: &[
                storage(0, true, wgpu::ShaderStages::COMPUTE),
                storage(1, false, wgpu::ShaderStages::COMPUTE),
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("particles-sim"),
            bind_group_layouts: &[Some(&sim_layout)],
            immediate_size: 0,
        });
        let kernel = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("particles-sim"),
                layout: Some(&pl),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let (reset, step, tick) = (kernel("reset"), kernel("step"), kernel("tick"));

        let dmod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("particles-draw"),
            source: wgpu::ShaderSource::Wgsl(format!("{COMMON_WGSL}{DRAW_WGSL}").into()),
        });
        let draw_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("particles-draw"),
            entries: &[
                storage(0, true, wgpu::ShaderStages::VERTEX),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let dpl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("particles-draw"),
            bind_group_layouts: &[Some(&draw_layout)],
            immediate_size: 0,
        });
        let add = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        };
        let draw = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("particles-draw"),
            layout: Some(&dpl),
            vertex: wgpu::VertexState {
                module: &dmod,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &dmod,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: Some(wgpu::BlendState { color: add, alpha: add }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        Self { reset, step, tick, sim_layout, draw, draw_layout }
    }

    pub fn make_layer(&self, device: &wgpu::Device, w: u32, h: u32) -> ParticleLayer {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("particles-layer"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        let sim_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particles-table"),
            size: (SIM_VEC4S * 16) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ParticleLayer {
            texture,
            view,
            width: w,
            height: h,
            dirty: true,
            sim_buf,
            state: None,
            sim_bind: None,
            slots: 0,
            origin: 0,
            sim_hash: u64::MAX,
            renders: Vec::new(),
            render_hash: u64::MAX,
            at: None,
            want: 0,
            drawn: None,
            checkpoints: BTreeMap::new(),
            interval: CHECKPOINT_INTERVAL,
            last_steps: 0,
        }
    }

    /// Hand the layer what to simulate and draw, and the frame; marks it
    /// dirty when the picture would differ from what it holds.
    pub fn set(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layer: &mut ParticleLayer,
        c: &Compiled,
        frame: i64,
    ) {
        if c.sim_hash != layer.sim_hash {
            queue.write_buffer(&layer.sim_buf, 0, bytemuck::cast_slice(&c.sim));
            if c.slots != layer.slots || layer.state.is_none() {
                layer.slots = c.slots;
                let state = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("particles-state"),
                    size: layer.state_bytes(),
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                layer.sim_bind = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("particles-sim"),
                    layout: &self.sim_layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: layer.sim_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: state.as_entire_binding() },
                    ],
                }));
                layer.state = Some(state);
                // The draw bind groups point at the old state buffer.
                layer.renders.clear();
                layer.render_hash = u64::MAX;
            }
            layer.origin = c.origin;
            layer.sim_hash = c.sim_hash;
            layer.checkpoints.clear();
            layer.interval = CHECKPOINT_INTERVAL;
            layer.at = None;
            layer.drawn = None;
        }
        // The look, with the layer's size filled in.
        let mut rs = c.renders.clone();
        for r in &mut rs {
            r[0] = layer.width as f32;
            r[1] = layer.height as f32;
        }
        let rh = crate::diskcache::content_hash(&[bytemuck::cast_slice(&rs)]);
        if rh != layer.render_hash {
            if let Some(state) = &layer.state {
                while layer.renders.len() < rs.len() {
                    let buf = device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("particles-render-u"),
                        size: (RENDER_VEC4S * 16) as u64,
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("particles-draw"),
                        layout: &self.draw_layout,
                        entries: &[
                            wgpu::BindGroupEntry { binding: 0, resource: state.as_entire_binding() },
                            wgpu::BindGroupEntry { binding: 1, resource: buf.as_entire_binding() },
                        ],
                    });
                    layer.renders.push(RenderSlot { buf, bind });
                }
                layer.renders.truncate(rs.len());
                for (slot, r) in layer.renders.iter().zip(&rs) {
                    queue.write_buffer(&slot.buf, 0, bytemuck::cast_slice(r));
                }
            }
            layer.render_hash = rh;
        }
        layer.want = frame;
        if layer.drawn != Some((frame, layer.render_hash)) {
            layer.dirty = true;
        }
    }

    fn dispatch(&self, encoder: &mut wgpu::CommandEncoder, layer: &ParticleLayer, pipe: &wgpu::ComputePipeline, n: u32) {
        let Some(bind) = &layer.sim_bind else { return };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("particles-sim"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipe);
        pass.set_bind_group(0, bind, &[]);
        pass.dispatch_workgroups(n, 1, 1);
    }

    /// Keep a checkpoint of frame `f`, thinning the set to fit the budget.
    fn checkpoint(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, layer: &mut ParticleLayer, f: i64) {
        let bytes = layer.state_bytes();
        if bytes > CHECKPOINT_BUDGET || (f - layer.origin) % layer.interval != 0 || layer.checkpoints.contains_key(&f) {
            return;
        }
        while (layer.checkpoints.len() as u64 + 1) * bytes > CHECKPOINT_BUDGET {
            layer.interval *= 2;
            let (iv, o) = (layer.interval, layer.origin);
            layer.checkpoints.retain(|k, _| (k - o) % iv == 0);
            if (f - o) % iv != 0 {
                return;
            }
        }
        let Some(state) = &layer.state else { return };
        let ck = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("particles-checkpoint"),
            size: bytes,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(state, 0, &ck, 0, bytes);
        layer.checkpoints.insert(f, ck);
    }

    /// Bring the state to the frame asked for and draw it.
    pub fn encode(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, layer: &mut ParticleLayer) {
        layer.last_steps = 0;
        let groups = layer.slots.div_ceil(256).max(1);
        let t = layer.want;
        if layer.slots > 0 && layer.state.is_some() {
            if t <= layer.origin {
                if layer.at != Some(layer.origin) {
                    self.dispatch(encoder, layer, &self.reset, groups);
                    layer.at = Some(layer.origin);
                }
            } else {
                let live = layer.at.filter(|a| *a <= t && *a >= layer.origin);
                let ck = layer.checkpoints.range(..=t).next_back().map(|(k, _)| *k);
                let mut cur = match (live, ck) {
                    (Some(a), Some(k)) if k > a => {
                        self.restore(encoder, layer, k);
                        k
                    }
                    (Some(a), _) => a,
                    (None, Some(k)) => {
                        self.restore(encoder, layer, k);
                        k
                    }
                    (None, None) => {
                        self.dispatch(encoder, layer, &self.reset, groups);
                        layer.origin
                    }
                };
                while cur < t {
                    self.dispatch(encoder, layer, &self.step, groups);
                    self.dispatch(encoder, layer, &self.tick, 1);
                    cur += 1;
                    layer.last_steps += 1;
                    self.checkpoint(device, encoder, layer, cur);
                }
                layer.at = Some(t);
            }
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("particles-draw"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &layer.view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if layer.slots > 0 {
                pass.set_pipeline(&self.draw);
                for r in &layer.renders {
                    pass.set_bind_group(0, &r.bind, &[]);
                    pass.draw(0..4, 0..layer.slots);
                }
            }
        }
        layer.drawn = Some((t, layer.render_hash));
        layer.dirty = false;
    }

    fn restore(&self, encoder: &mut wgpu::CommandEncoder, layer: &ParticleLayer, k: i64) {
        if let (Some(state), Some(ck)) = (&layer.state, layer.checkpoints.get(&k)) {
            encoder.copy_buffer_to_buffer(ck, 0, state, 0, layer.state_bytes());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn validate(src: &str) {
        let module = naga::front::wgsl::parse_str(src).expect("parses");
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&module)
            .expect("validates");
    }

    #[test]
    fn sim_wgsl_validates() {
        validate(&sim_wgsl());
    }

    #[test]
    fn draw_wgsl_validates() {
        validate(&format!("{COMMON_WGSL}{DRAW_WGSL}"));
    }
}
