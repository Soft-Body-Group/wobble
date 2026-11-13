//! The Plasma node's layer: one instanced quad per particle, drawn
//! ADDITIVELY into a per-node texture the fused pass samples
//! (`SourceBinding::Plasma`).
//!
//! There is no particle buffer. The vertex shader works a particle's
//! place out of its instance index and the uniform
//! ([`crate::nodes::plasma::PlasmaParams::uniform`]): which field line, which
//! strand, how far along, where that is on the sphere, where the sphere
//! lands on the picture. So the only thing uploaded per frame is 192 bytes,
//! and the layer is re-drawn only when those bytes change (a still frame
//! with still params draws nothing).

use crate::nodes::plasma::UNIFORM_VEC4S;

pub const PLASMA_WGSL: &str = r#"
struct U { v: array<vec4f, 12> };
@group(0) @binding(0) var<uniform> u: U;

struct VO {
    @builtin(position) pos: vec4f,
    @location(0) q: vec2f,
    @location(1) @interpolate(flat) col: vec4f,
};

const PI: f32 = 3.14159265;
const TAU: f32 = 6.2831853;

fn pcg(x: u32) -> u32 {
    let s = x * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}
// A number in 0..1 for (thing, which number of that thing).
fn rnd(a: u32, k: u32) -> f32 {
    return f32(pcg(a ^ pcg(k * 0x9e3779b9u + 0x85ebca6bu)) >> 8u) / 16777216.0;
}

// A root on the unit sphere: on the limb ring at depth 0, anywhere at 1.
// z is towards the viewer.
fn root(a: u32, k: u32, depth: f32) -> vec3f {
    let th = rnd(a, k) * TAU;
    let z = (rnd(a, k + 1u) * 2.0 - 1.0) * depth;
    let r = sqrt(max(1.0 - z * z, 0.0));
    return vec3f(r * cos(th), r * sin(th), z);
}

// The smooth moving field that bends the lines: three sines a lane, two
// octaves. `ph` decorrelates one line family from the next.
fn turb(p: vec3f, t: f32, ph: f32) -> vec3f {
    let f = u.v[6].y;
    let a = p * f + vec3f(ph, ph * 1.7, ph * 0.6);
    let o1 = vec3f(
        sin(a.y * 1.9 + a.z * 1.3 + t),
        sin(a.z * 1.7 + a.x * 2.1 - t * 0.8),
        sin(a.x * 1.5 + a.y * 2.3 + t * 1.2));
    let b = a * 2.3 + o1 * 0.6;
    let o2 = vec3f(
        sin(b.y * 1.3 + b.z * 2.1 - t * 1.7),
        sin(b.z * 1.9 + b.x * 1.1 + t * 1.4),
        sin(b.x * 2.2 + b.y * 1.6 + t * 2.1));
    return o1 + o2 * 0.45;
}

struct Line {
    d: vec3f,      // root direction (a loop's midpoint)
    t: vec3f,      // along the loop
    b: vec3f,      // out of its plane
    alpha: f32,    // half the footpoint separation, radians
    h: f32,        // apex height, radii
    lean: f32,
    bright: f32,
    ph: f32,
};

// Where a field-line family is rooted: in a region (by `cluster`) or
// anywhere. `id` is the family, already salted by its generation.
struct Root {
    d: vec3f, // where the line leaves the surface
    c: vec3f, // its region's centre (itself, outside a region)
};

fn rooted(id: u32, spread: f32) -> Root {
    let depth = u.v[3].y;
    let seed = u32(i32(u.v[0].w));
    var d = root(id, 11u, depth);
    var axis = d;
    if (rnd(id, 13u) < u.v[3].x) {
        let reg = u32(rnd(id, 14u) * u.v[2].z) + seed * 7919u;
        let c = root(reg, 21u, depth);
        // A gaussian-ish blob about the region's centre.
        let j = vec3f(
            rnd(id, 15u) + rnd(id, 16u) - 1.0,
            rnd(id, 17u) + rnd(id, 18u) - 1.0,
            (rnd(id, 19u) + rnd(id, 20u) - 1.0) * depth);
        d = normalize(c + j * spread);
        axis = c;
    }
    return Root(d, axis);
}

fn frame_of(d: vec3f, id: u32) -> array<vec3f, 2> {
    let depth = u.v[3].y;
    // The limb's own tangent: at depth 0 every loop lies in the picture.
    var t0 = vec3f(-d.y, d.x, 0.0);
    let l = length(t0);
    if (l < 1e-4) { t0 = vec3f(1.0, 0.0, 0.0); } else { t0 = t0 / l; }
    let b0 = cross(d, t0);
    let chi = rnd(id, 31u) * TAU * depth;
    let t = t0 * cos(chi) + b0 * sin(chi);
    return array<vec3f, 2>(t, cross(d, t));
}

fn loop_line(id: u32) -> Line {
    var L: Line;
    L.d = rooted(id, u.v[9].w).d;
    let f = frame_of(L.d, id);
    L.t = f[0];
    L.b = f[1];
    let sv = 1.0 + u.v[3].w * (rnd(id, 41u) * 2.0 - 1.0);
    L.alpha = u.v[3].z * 0.5 * sv;
    // Wider loops stand taller: height follows the span, then varies.
    // Mostly low, a few tall: the variation is skewed upwards.
    let hr = rnd(id, 42u);
    L.h = u.v[4].x * mix(1.0, sv, 0.6) * (1.0 + u.v[4].y * (hr * hr * 3.0 - 1.0));
    L.lean = u.v[4].w * (rnd(id, 43u) * 2.0 - 1.0);
    let r = rnd(id, 44u);
    L.bright = 0.12 + r * r * r * 1.6;
    L.ph = rnd(id, 45u) * 40.0;
    return L;
}

// A point of a loop at `s` (0 = one foot, 1 = the other), `k` scaling
// the strand inside the bundle.
fn loop_pos(L: Line, s: f32, k: f32, off: f32, t: f32) -> vec3f {
    let ang = L.alpha * k * -cos(PI * s);
    let up = sin(PI * s);
    let el = L.h * k * pow(up, 0.85);
    // An arcade: strands side by side along the loop's axis.
    let d = normalize(L.d + L.b * off);
    var p = (d * cos(ang) + L.t * sin(ang)) * (1.0 + el) + L.b * (L.lean * el);
    p += turb(p, t, L.ph) * (u.v[6].x / u.v[1].z) * up * up;
    return p;
}

// A point of an open line at `s` (0 = the surface).
// `pinch` bends it towards its region's axis as it climbs, so a region's
// lines close into a helmet with a cusp.
fn stream_pos(r: Root, len: f32, s: f32, t: f32, ph: f32) -> vec3f {
    let up = len * s;
    let d = normalize(mix(r.d, r.c, u.v[10].y * (1.0 - exp(-up * 2.5))));
    var p = d * (1.0 + up);
    p += turb(p * 0.7, t, ph) * (u.v[6].x / u.v[1].z) * s * (0.5 + s) * 1.5;
    return p;
}

fn spin(p: vec3f, a: f32) -> vec3f {
    let c = cos(a);
    let s = sin(a);
    return vec3f(p.x * c + p.z * s, p.y, -p.x * s + p.z * c);
}

fn to_px(p: vec3f) -> vec2f {
    return u.v[1].xy + vec2f(p.x, -p.y) * u.v[1].z;
}

// 1 in front of the disc or clear of it, 0 behind it.
fn seen(p: vec3f) -> f32 {
    if (u.v[9].y < 0.5 || p.z >= 0.0) { return 1.0; }
    return smoothstep(1.0, 1.012, length(p.xy));
}

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VO {
    var o: VO;
    o.pos = vec4f(2.0, 2.0, 0.0, 1.0);
    o.q = vec2f(0.0);
    o.col = vec4f(0.0);
    let time = u.v[0].z;
    let seed = u32(i32(u.v[0].w));
    let n = u.v[2].x;
    let pid = ii * 2654435761u + seed * 97u;
    let tt = time * u.v[6].z;
    let rot = time * u.v[9].x;
    let R = u.v[1].z;
    var a: vec3f;
    var c: vec3f;
    var light: f32;
    let eps = 0.004;
    if (f32(ii) < n * u.v[7].x) {
        // --- an open line ---------------------------------------------
        let lines = max(u.v[2].y * u.v[2].w * 0.5, 1.0);
        let lid = u32(rnd(pid, 1u) * lines) + seed * 131u + 0x51ed27u;
        let rt = rooted(lid, u.v[9].w * 1.4);
        let lr = rnd(lid, 61u);
        let len = u.v[7].y * (0.25 + 1.25 * lr * lr);
        let speed = u.v[5].y * (0.5 + rnd(lid, 62u)) / max(len * R, 1.0);
        // Light dies with HEIGHT (radii) as (1 + falloff * up)^-2.5:
        // steeply at first, then a long tail, as a real corona's does.
        // HALF the law is in where the particles are and half in how
        // bright each one is. All in the brightness, and a half-float
        // layer cannot add up light a thousand times dimmer than what
        // the bright pixels by the limb already hold (it plateaus); all
        // in the placement, and the few far particles show one by one.
        // (So a particle also speeds up as it leaves, like the wind.)
        let k = max(u.v[7].z, 1e-3);
        let area = 1.0 - pow(1.0 + k * len, -0.25);
        let r = fract(rnd(pid, 2u) + time * speed);
        let s = clamp((pow(1.0 - r * area, -4.0) - 1.0) / (k * len), 0.0, 1.0);
        let ph = rnd(lid, 63u) * 40.0;
        a = stream_pos(rt, len, s, tt, ph);
        c = stream_pos(rt, len, s + eps, tt, ph);
        let lb = rnd(lid, 64u);
        // The line's own end is soft.
        light = (0.1 + lb * lb * 1.4) * pow(1.0 + k * len * s, -1.25) * smoothstep(1.0, 0.55, s);
        let w = 0.5 + 0.5 * sin(TAU * (s * 2.0 - time * speed * 3.0 + rnd(lid, 65u)));
        light *= mix(1.0, w * 1.6, u.v[9].z);
    } else {
        // --- a loop -----------------------------------------------------
        let lp = u32(rnd(pid, 1u) * u.v[2].y);
        var gen = 0u;
        var env = 1.0;
        if (u.v[5].x > 0.5) {
            let ph = time / u.v[5].x + rnd(lp + seed * 31u, 51u);
            gen = u32(i32(floor(ph)) + 4096);
            let fr = fract(ph);
            env = pow(sin(PI * fr), 0.7);
        }
        let lid = lp + gen * 104729u + seed * 31u;
        let L = loop_line(lid);
        let strands = u.v[2].w;
        let st = u32(rnd(pid, 3u) * strands);
        let sid = lid * 8191u + st;
        // Strands nest: each a scaled copy of the loop.
        let k = max(1.0 + u.v[4].z * ((f32(st) + rnd(sid, 71u)) / strands - 0.5), 0.05);
        let px_len = max((L.alpha * 2.0 + L.h * 2.0) * k * R, 1.0);
        let dir = select(-1.0, 1.0, rnd(sid, 72u) < 0.5);
        let speed = u.v[5].y * (0.5 + rnd(sid, 73u)) / px_len * dir;
        let s = fract(rnd(pid, 2u) + time * speed);
        let off = u.v[10].x * (rnd(sid, 76u) - 0.5);
        a = loop_pos(L, s, k, off, tt);
        c = loop_pos(L, s + eps, k, off, tt);
        let sb = rnd(sid, 74u);
        light = L.bright * (0.25 + 0.75 * sb * sb) * env;
        light *= 1.0 + u.v[7].w * (exp(-s * 14.0) + exp(-(1.0 - s) * 14.0));
        let w = 0.5 + 0.5 * sin(TAU * (s * 3.0 - time * speed * 4.0 + rnd(sid, 75u)));
        light *= mix(1.0, w * 1.6, u.v[9].z);
    }
    a = spin(a, rot);
    c = spin(c, rot);
    light *= seen(a);
    if (light <= 1e-5) { return o; }
    let p = to_px(a);
    var along = to_px(c) - p;
    let al = length(along);
    if (al < 1e-6) { along = vec2f(1.0, 0.0); } else { along = along / al; }
    let across = vec2f(-along.y, along.x);
    // Scatter off the line.
    let fz = (vec2f(rnd(pid, 5u), rnd(pid, 6u)) + vec2f(rnd(pid, 7u), rnd(pid, 8u)) - 1.0) * u.v[6].w;
    let hw = u.v[5].w * 1.5;
    let hl = u.v[5].z * 0.5 + hw;
    let corner = vec2f(f32(vi & 1u) * 2.0 - 1.0, f32(vi >> 1u) * 2.0 - 1.0);
    let q = p + fz + along * (corner.x * hl) + across * (corner.y * hw);
    o.pos = vec4f(q.x / u.v[0].x * 2.0 - 1.0, 1.0 - q.y / u.v[0].y * 2.0, 0.0, 1.0);
    // The fragment wants px along (from the centre) and a unit across.
    o.q = vec2f(corner.x * hl, corner.y * 1.5);
    o.col = vec4f(u.v[8].rgb, u.v[8].w) * light;
    return o;
}

@fragment
fn fs(o: VO) -> @location(0) vec4f {
    // A capsule of light: gaussian across, and along past the streak's
    // two ends.
    let hw = u.v[5].w;
    let d = max(abs(o.q.x) - u.v[5].z * 0.5, 0.0) / hw;
    let e = exp(-(o.q.y * o.q.y + d * d) * 2.2);
    return o.col * e;
}
"#;

/// The shared pipeline; per-node state lives in [`PlasmaLayer`].
pub struct PlasmaRaster {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
}

pub struct PlasmaLayer {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub width: u32,
    pub height: u32,
    /// The uniform's bytes hashed; `u64::MAX` until the first set.
    pub hash: u64,
    pub dirty: bool,
    /// Particles the last set asked for.
    pub count: u32,
    uniform: wgpu::Buffer,
    bind: wgpu::BindGroup,
}

impl PlasmaRaster {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("plasma-layer"),
            source: wgpu::ShaderSource::Wgsl(PLASMA_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("plasma-layer"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("plasma-layer"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let add = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        };
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("plasma-layer"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: Some(wgpu::BlendState { color: add, alpha: add }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleStrip, ..Default::default() },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        Self { pipeline, layout }
    }

    pub fn make_layer(&self, device: &wgpu::Device, w: u32, h: u32) -> PlasmaLayer {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("plasma-layer"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("plasma-layer-u"),
            size: (UNIFORM_VEC4S * 16) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("plasma-layer"),
            layout: &self.layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() }],
        });
        PlasmaLayer { texture, view, width: w, height: h, hash: u64::MAX, dirty: true, count: 0, uniform, bind }
    }

    /// Hand the layer what it should draw; marks it dirty when that
    /// differs from what it holds.
    pub fn set(&self, queue: &wgpu::Queue, layer: &mut PlasmaLayer, uni: &[f32; UNIFORM_VEC4S * 4], count: u32) {
        let hash = crate::diskcache::content_hash(&[bytemuck::cast_slice(uni)]);
        if layer.hash == hash {
            return;
        }
        queue.write_buffer(&layer.uniform, 0, bytemuck::cast_slice(uni));
        layer.count = count;
        layer.hash = hash;
        layer.dirty = true;
    }

    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder, layer: &mut PlasmaLayer) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("plasma-layer"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &layer.view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        if layer.count > 0 {
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &layer.bind, &[]);
            pass.draw(0..4, 0..layer.count);
        }
        layer.dirty = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plasma_wgsl_validates() {
        let module = naga::front::wgsl::parse_str(PLASMA_WGSL).expect("parses");
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&module)
            .expect("validates");
    }
}
