//! The Splat node's layer: a Gaussian splat capture drawn through a camera
//! into a per-node texture the fused pass samples (`SourceBinding::Splat`).
//!
//! Everything per frame is on the GPU, in one compute pass and one render
//! pass, and nothing comes back to the CPU:
//!
//! 1. **project** — a thread per splat: cull behind the camera and off
//!    frame, take its 3D covariance to the picture (the EWA Jacobian), add
//!    the 0.3 px low-pass, evaluate its SH for this view, and append the
//!    survivors (an atomic counter) with a 24-bit log-depth key;
//! 2. **prepare** — one thread turns the count into the sort's dispatch
//!    and the draw's instance count (both indirect);
//! 3. **sort** — an LSD radix sort, 4 bits a pass, six passes: per-block
//!    digit counts, one scan of them, a stable scatter (each block sorted
//!    in workgroup memory by four one-bit splits, kept under 8 KiB);
//! 4. **draw** — an instanced quad per splat, nearest first, blended
//!    "under" (`1 - dst.a`), so the layer comes out premultiplied. A second
//!    target accumulates depth the same way when the depth output is used.
//!
//! The layer is re-drawn only when its uniform (camera, transform, look)
//! or its cloud changes; a still camera costs nothing. The colours are the
//! capture's (display-referred, as trained); the fused pass turns them
//! scene-linear (`nodes::splat`).

use crate::splat::{Cloud, SPLAT_WORDS};

/// Uniform bytes; see `struct U` in [`PROJECT_WGSL`].
pub const UNIFORM_BYTES: usize = 128;
/// Splats a sort block holds.
const BLOCK: u32 = 512;
/// Sort passes: 24-bit keys, 4 bits each.
const PASSES: u32 = 6;

const COMMON: &str = r#"
struct U {
    mv0: vec4f, mv1: vec4f, mv2: vec4f,
    cam: vec4f,   // camera position, native frame
    lens: vec4f,  // focal px, cx, cy, near
    img: vec4f,   // w, h, far, log(far / near)
    look: vec4f,  // opacity gain, size gain, x and y frustum limits (tan)
    n: u32, sh_deg: u32, sh_words: u32, aa: u32,
};
"#;

pub const PROJECT_WGSL: &str = r#"
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> splats: array<u32>;
@group(0) @binding(2) var<storage, read> sh: array<u32>;
@group(0) @binding(3) var<storage, read_write> proj: array<vec4u>;
@group(0) @binding(4) var<storage, read_write> keys: array<u32>;
@group(0) @binding(5) var<storage, read_write> vals: array<u32>;
@group(0) @binding(6) var<storage, read_write> ctl: array<atomic<u32>>;

fn hh(w: u32) -> vec2f { return unpack2x16float(w); }

fn shh(i: u32, j: u32) -> f32 {
    let w = hh(sh[i * u.sh_words + j / 2u]);
    return select(w.x, w.y, (j & 1u) == 1u);
}
fn shc(i: u32, k: u32) -> vec3f {
    let o = k * 3u;
    return vec3f(shh(i, o), shh(i, o + 1u), shh(i, o + 2u));
}

fn sh_color(i: u32, d: vec3f) -> vec3f {
    var c = vec3f(0.0);
    let x = d.x; let y = d.y; let z = d.z;
    if (u.sh_deg >= 1u) {
        let C1 = 0.4886025119029199;
        c += -C1 * y * shc(i, 0u) + C1 * z * shc(i, 1u) - C1 * x * shc(i, 2u);
    }
    if (u.sh_deg >= 2u) {
        let xx = x * x; let yy = y * y; let zz = z * z;
        c += 1.0925484305920792 * x * y * shc(i, 3u)
           - 1.0925484305920792 * y * z * shc(i, 4u)
           + 0.31539156525252005 * (2.0 * zz - xx - yy) * shc(i, 5u)
           - 1.0925484305920792 * x * z * shc(i, 6u)
           + 0.5462742152960396 * (xx - yy) * shc(i, 7u);
        if (u.sh_deg >= 3u) {
            c += -0.5900435899266435 * y * (3.0 * xx - yy) * shc(i, 8u)
               + 2.890611442640554 * x * y * z * shc(i, 9u)
               - 0.4570457994644658 * y * (4.0 * zz - xx - yy) * shc(i, 10u)
               + 0.3731763325901154 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * shc(i, 11u)
               - 0.4570457994644658 * x * (4.0 * zz - xx - yy) * shc(i, 12u)
               + 1.445305721320277 * z * (xx - yy) * shc(i, 13u)
               - 0.5900435899266435 * x * (xx - 3.0 * yy) * shc(i, 14u);
        }
    }
    return c;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(num_workgroups) nwg: vec3u) {
    let i = gid.x + gid.y * nwg.x * 256u;
    if (i >= u.n) { return; }
    let b = i * 9u;
    let p = vec3f(bitcast<f32>(splats[b]), bitcast<f32>(splats[b + 1u]), bitcast<f32>(splats[b + 2u]));
    let v = vec3f(dot(u.mv0.xyz, p) + u.mv0.w, dot(u.mv1.xyz, p) + u.mv1.w, dot(u.mv2.xyz, p) + u.mv2.w);
    let t = -v.z;
    if (t < u.lens.w || t > u.img.z) { return; }
    let f = u.lens.x;
    let sx = u.lens.y + f * v.x / t;
    let sy = u.lens.z - f * v.y / t;
    let s2o = hh(splats[b + 6u]);
    var op = s2o.y * u.look.x;
    if (op < 1.0 / 255.0) { return; }
    let q = normalize(vec4f(hh(splats[b + 3u]), hh(splats[b + 4u]))); // w x y z
    let w = q.x; let x = q.y; let y = q.z; let z = q.w;
    let R = mat3x3f(
        vec3f(1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z), 2.0 * (x * z - w * y)),
        vec3f(2.0 * (x * y - w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + w * x)),
        vec3f(2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)));
    let s = exp(vec3f(hh(splats[b + 5u]), s2o.x)) * u.look.y;
    // The Jacobian of the projection, its tangents clamped a little past
    // the frame so a splat far off the edge cannot blow up.
    let tx = clamp(v.x / t, -u.look.z, u.look.z);
    let ty = clamp(v.y / t, -u.look.w, u.look.w);
    let j0 = vec3f(f / t, 0.0, f * tx / t);
    let j1 = vec3f(0.0, -f / t, -f * ty / t);
    // T = J W (W the model-view's linear part); cov2 = T R S S R^T T^T.
    let t0 = j0.x * u.mv0.xyz + j0.y * u.mv1.xyz + j0.z * u.mv2.xyz;
    let t1 = j1.x * u.mv0.xyz + j1.y * u.mv1.xyz + j1.z * u.mv2.xyz;
    let m0 = (transpose(R) * t0) * s;
    let m1 = (transpose(R) * t1) * s;
    var a = dot(m0, m0);
    let bb = dot(m0, m1);
    var c = dot(m1, m1);
    let det0 = a * c - bb * bb;
    a += 0.3;
    c += 0.3;
    let det = a * c - bb * bb;
    if (det <= 0.0) { return; }
    if (u.aa == 1u) { op *= sqrt(max(det0, 0.0) / det); }
    let mid = 0.5 * (a + c);
    let lam = mid + sqrt(max(0.1, mid * mid - det));
    let r = min(ceil(3.0 * sqrt(lam)), max(u.img.x, u.img.y));
    if (sx + r < 0.0 || sx - r > u.img.x || sy + r < 0.0 || sy - r > u.img.y) { return; }
    var col = vec3f(hh(splats[b + 7u]), hh(splats[b + 8u]).x);
    if (u.sh_deg > 0u) {
        col += sh_color(i, normalize(p - u.cam.xyz));
    }
    col = max(col, vec3f(0.0));
    let slot = atomicAdd(&ctl[0], 1u);
    proj[slot * 2u] = vec4u(bitcast<u32>(sx), bitcast<u32>(sy), bitcast<u32>(r), pack2x16float(col.rg));
    proj[slot * 2u + 1u] = vec4u(bitcast<vec3u>(vec3f(c / det, -bb / det, a / det)), pack2x16float(vec2f(col.b, op)));
    let k = clamp(log(t / u.lens.w) / u.img.w, 0.0, 1.0);
    keys[slot] = u32(k * 16777215.0);
    vals[slot] = slot;
}
"#;

pub const PREPARE_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> ctl: array<u32>;
@compute @workgroup_size(1)
fn main() {
    let n = ctl[0];
    ctl[4] = max((n + 511u) / 512u, 1u);
    ctl[5] = 1u;
    ctl[6] = 1u;
    ctl[8] = 4u;
    ctl[9] = n;
    ctl[10] = 0u;
    ctl[11] = 0u;
}
"#;

const RADIX_COMMON: &str = r#"
struct P { shift: u32, a: u32, b: u32, c: u32 };
@group(0) @binding(0) var<uniform> pp: P;
@group(0) @binding(1) var<storage, read> ctl: array<u32>;
@group(0) @binding(2) var<storage, read> kin: array<u32>;
@group(0) @binding(3) var<storage, read> vin: array<u32>;
@group(0) @binding(4) var<storage, read_write> kout: array<u32>;
@group(0) @binding(5) var<storage, read_write> vout: array<u32>;
@group(0) @binding(6) var<storage, read_write> hist: array<u32>;
fn blocks() -> u32 { return (ctl[0] + 511u) / 512u; }
fn digit(k: u32) -> u32 { return (k >> pp.shift) & 15u; }
"#;

pub const HIST_WGSL: &str = r#"
var<workgroup> cnt: array<atomic<u32>, 16>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3u) {
    let n = ctl[0];
    let nb = blocks();
    if (li < 16u) { atomicStore(&cnt[li], 0u); }
    workgroupBarrier();
    for (var k = 0u; k < 2u; k++) {
        let i = wg.x * 512u + li + k * 256u;
        if (i < n) { atomicAdd(&cnt[digit(kin[i])], 1u); }
    }
    workgroupBarrier();
    if (li < 16u && wg.x < nb) { hist[li * nb + wg.x] = atomicLoad(&cnt[li]); }
}
"#;

pub const SCAN_WGSL: &str = r#"
var<workgroup> part: array<u32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32) {
    let total = blocks() * 16u;
    let per = (total + 255u) / 256u;
    let s = min(li * per, total);
    let e = min(s + per, total);
    var sum = 0u;
    for (var i = s; i < e; i++) { sum += hist[i]; }
    part[li] = sum;
    workgroupBarrier();
    for (var off = 1u; off < 256u; off *= 2u) {
        var v = 0u;
        if (li >= off) { v = part[li - off]; }
        workgroupBarrier();
        part[li] += v;
        workgroupBarrier();
    }
    var run = part[li] - sum;
    for (var i = s; i < e; i++) {
        let h = hist[i];
        hist[i] = run;
        run += h;
    }
}
"#;

pub const SCATTER_WGSL: &str = r#"
const SENT: u32 = 0xffffffffu;
var<workgroup> sk: array<u32, 512>;
var<workgroup> sv: array<u32, 512>;
var<workgroup> part: array<u32, 256>;
var<workgroup> dstart: array<atomic<u32>, 16>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32, @builtin(workgroup_id) wg: vec3u) {
    let n = ctl[0];
    let nb = blocks();
    let base = wg.x * 512u;
    for (var k = 0u; k < 2u; k++) {
        let l = li + k * 256u;
        let i = base + l;
        if (i < n) { sk[l] = kin[i]; sv[l] = vin[i]; } else { sk[l] = SENT; sv[l] = 0u; }
    }
    if (li < 16u) { atomicStore(&dstart[li], 0u); }
    workgroupBarrier();
    // Sort the block by its digit, stably: four one-bit splits.
    for (var bit = 0u; bit < 4u; bit++) {
        let l0 = li * 2u;
        let l1 = l0 + 1u;
        let k0 = sk[l0]; let k1 = sk[l1];
        let v0 = sv[l0]; let v1 = sv[l1];
        let f0 = 1u - ((digit(k0) >> bit) & 1u);
        let f1 = 1u - ((digit(k1) >> bit) & 1u);
        let ts = f0 + f1;
        part[li] = ts;
        workgroupBarrier();
        for (var off = 1u; off < 256u; off *= 2u) {
            var v = 0u;
            if (li >= off) { v = part[li - off]; }
            workgroupBarrier();
            part[li] += v;
            workgroupBarrier();
        }
        let ex = part[li] - ts;
        let zeros = part[255];
        workgroupBarrier();
        let p0 = select(zeros + l0 - ex, ex, f0 == 1u);
        let p1 = select(zeros + l1 - (ex + f0), ex + f0, f1 == 1u);
        sk[p0] = k0; sv[p0] = v0;
        sk[p1] = k1; sv[p1] = v1;
        workgroupBarrier();
    }
    for (var k = 0u; k < 2u; k++) {
        let key = sk[li + k * 256u];
        if (key != SENT) { atomicAdd(&dstart[digit(key)], 1u); }
    }
    workgroupBarrier();
    if (li == 0u) {
        var run = 0u;
        for (var d = 0u; d < 16u; d++) {
            let c = atomicLoad(&dstart[d]);
            atomicStore(&dstart[d], run);
            run += c;
        }
    }
    workgroupBarrier();
    for (var k = 0u; k < 2u; k++) {
        let l = li + k * 256u;
        let key = sk[l];
        if (key != SENT) {
            let d = digit(key);
            let g = hist[d * nb + wg.x] + (l - atomicLoad(&dstart[d]));
            kout[g] = key;
            vout[g] = sv[l];
        }
    }
}
"#;

pub const DRAW_WGSL: &str = r#"
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> proj: array<vec4u>;
@group(0) @binding(2) var<storage, read> keys: array<u32>;
@group(0) @binding(3) var<storage, read> vals: array<u32>;

struct VO {
    @builtin(position) pos: vec4f,
    @location(0) d: vec2f,
    @location(1) @interpolate(flat) conic: vec3f,
    @location(2) @interpolate(flat) col: vec4f,
    @location(3) @interpolate(flat) z: f32,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VO {
    var o: VO;
    let s = vals[ii];
    let a = proj[s * 2u];
    let b = proj[s * 2u + 1u];
    let c = vec2f(bitcast<f32>(a.x), bitcast<f32>(a.y));
    let r = bitcast<f32>(a.z);
    let corner = vec2f(f32(vi & 1u) * 2.0 - 1.0, f32(vi >> 1u) * 2.0 - 1.0);
    let q = c + corner * r;
    o.pos = vec4f(q.x / u.img.x * 2.0 - 1.0, 1.0 - q.y / u.img.y * 2.0, 0.0, 1.0);
    o.d = corner * r;
    o.conic = bitcast<vec3f>(b.xyz);
    o.col = vec4f(unpack2x16float(a.w), unpack2x16float(b.w));
    o.z = u.lens.w * exp(f32(keys[ii]) / 16777215.0 * u.img.w);
    return o;
}

fn alpha_of(o: VO) -> f32 {
    let d = o.d;
    let power = -0.5 * (o.conic.x * d.x * d.x + o.conic.z * d.y * d.y) - o.conic.y * d.x * d.y;
    if (power > 0.0) { return 0.0; }
    return min(0.99, o.col.a * exp(power));
}

@fragment
fn fs(o: VO) -> @location(0) vec4f {
    let al = alpha_of(o);
    if (al < 1.0 / 255.0) { discard; }
    return vec4f(o.col.rgb * al, al);
}

struct Two { @location(0) c: vec4f, @location(1) z: vec4f };

@fragment
fn fs_depth(o: VO) -> Two {
    let al = alpha_of(o);
    if (al < 1.0 / 255.0) { discard; }
    return Two(vec4f(o.col.rgb * al, al), vec4f(o.z * al, 0.0, 0.0, al));
}
"#;

/// Every module, for the validation test.
pub fn modules() -> Vec<(&'static str, String)> {
    vec![
        ("project", format!("{COMMON}{PROJECT_WGSL}")),
        ("prepare", PREPARE_WGSL.to_string()),
        ("hist", format!("{RADIX_COMMON}{HIST_WGSL}")),
        ("scan", format!("{RADIX_COMMON}{SCAN_WGSL}")),
        ("scatter", format!("{RADIX_COMMON}{SCATTER_WGSL}")),
        ("draw", format!("{COMMON}{DRAW_WGSL}")),
    ]
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

fn uniform(binding: u32, vis: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: vis,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// The shared pipelines; per-node state lives in [`SplatLayer`].
pub struct SplatRaster {
    project: wgpu::ComputePipeline,
    prepare: wgpu::ComputePipeline,
    hist: wgpu::ComputePipeline,
    scan: wgpu::ComputePipeline,
    scatter: wgpu::ComputePipeline,
    draw: wgpu::RenderPipeline,
    draw_depth: wgpu::RenderPipeline,
    project_l: wgpu::BindGroupLayout,
    prepare_l: wgpu::BindGroupLayout,
    radix_l: wgpu::BindGroupLayout,
    draw_l: wgpu::BindGroupLayout,
}

/// The buffers for one uploaded cloud.
struct Gpu {
    gen: u64,
    count: u32,
    pre: wgpu::BindGroup,
    prep: wgpu::BindGroup,
    radix: Vec<wgpu::BindGroup>,
    draw: wgpu::BindGroup,
    ctl: wgpu::Buffer,
    _bufs: Vec<wgpu::Buffer>,
}

pub struct SplatLayer {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub depth_texture: wgpu::Texture,
    pub depth_view: wgpu::TextureView,
    pub width: u32,
    pub height: u32,
    /// What the last set asked for, hashed; `u64::MAX` before one.
    pub hash: u64,
    pub dirty: bool,
    /// Draw the depth target too.
    pub depth: bool,
    /// Times the layer was drawn (tests: a still camera draws once).
    pub draws: u64,
    uniform: wgpu::Buffer,
    gpu: Option<Gpu>,
}

impl SplatLayer {
    /// Splats in the uploaded cloud.
    pub fn splats(&self) -> u32 {
        self.gpu.as_ref().map_or(0, |g| g.count)
    }
    /// The cloud generation on the GPU (0: none).
    pub fn gen(&self) -> u64 {
        self.gpu.as_ref().map_or(0, |g| g.gen)
    }
}

impl SplatRaster {
    pub fn new(device: &wgpu::Device) -> Self {
        use wgpu::ShaderStages as S;
        let m = |label: &str, src: String| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            })
        };
        let mods = modules();
        let module = |name: &str| m(name, mods.iter().find(|(n, _)| *n == name).unwrap().1.clone());
        let layout = |label: &str, entries: &[wgpu::BindGroupLayoutEntry]| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some(label), entries })
        };
        let c = S::COMPUTE;
        let project_l = layout(
            "splat-project",
            &[
                uniform(0, c),
                storage(1, true, c),
                storage(2, true, c),
                storage(3, false, c),
                storage(4, false, c),
                storage(5, false, c),
                storage(6, false, c),
            ],
        );
        let prepare_l = layout("splat-prepare", &[storage(0, false, c)]);
        let radix_l = layout(
            "splat-radix",
            &[
                uniform(0, c),
                storage(1, true, c),
                storage(2, true, c),
                storage(3, true, c),
                storage(4, false, c),
                storage(5, false, c),
                storage(6, false, c),
            ],
        );
        let vf = S::VERTEX | S::FRAGMENT;
        let draw_l = layout(
            "splat-draw",
            &[uniform(0, vf), storage(1, true, S::VERTEX), storage(2, true, S::VERTEX), storage(3, true, S::VERTEX)],
        );
        let compute = |label: &str, l: &wgpu::BindGroupLayout, name: &str| {
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(l)],
                immediate_size: 0,
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pl),
                module: &module(name),
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let project = compute("splat-project", &project_l, "project");
        let prepare = compute("splat-prepare", &prepare_l, "prepare");
        let hist = compute("splat-hist", &radix_l, "hist");
        let scan = compute("splat-scan", &radix_l, "scan");
        let scatter = compute("splat-scatter", &radix_l, "scatter");
        let draw_mod = module("draw");
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("splat-draw"),
            bind_group_layouts: &[Some(&draw_l)],
            immediate_size: 0,
        });
        // Nearest first, each splat laid UNDER what is already there.
        let under = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::OneMinusDstAlpha,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        };
        let target = Some(wgpu::ColorTargetState {
            format: wgpu::TextureFormat::Rgba16Float,
            blend: Some(wgpu::BlendState { color: under, alpha: under }),
            write_mask: wgpu::ColorWrites::ALL,
        });
        let render = |label: &str, fs: &str, targets: &[Option<wgpu::ColorTargetState>]| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pl),
                vertex: wgpu::VertexState {
                    module: &draw_mod,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &draw_mod,
                    entry_point: Some(fs),
                    compilation_options: Default::default(),
                    targets,
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };
        let draw = render("splat-draw", "fs", &[target.clone()]);
        let draw_depth = render("splat-draw-depth", "fs_depth", &[target.clone(), target]);
        Self { project, prepare, hist, scan, scatter, draw, draw_depth, project_l, prepare_l, radix_l, draw_l }
    }

    fn tex(device: &wgpu::Device, w: u32, h: u32, label: &str) -> (wgpu::Texture, wgpu::TextureView) {
        let t = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let v = t.create_view(&Default::default());
        (t, v)
    }

    pub fn make_layer(&self, device: &wgpu::Device, w: u32, h: u32) -> SplatLayer {
        let (texture, view) = Self::tex(device, w, h, "splat-layer");
        let (depth_texture, depth_view) = Self::tex(device, w, h, "splat-depth");
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("splat-u"),
            size: UNIFORM_BYTES as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        SplatLayer {
            texture,
            view,
            depth_texture,
            depth_view,
            width: w,
            height: h,
            hash: u64::MAX,
            dirty: true,
            depth: false,
            draws: 0,
            uniform,
            gpu: None,
        }
    }

    /// Put a cloud on the GPU (or take it off, with `None`).
    fn upload(&self, device: &wgpu::Device, queue: &wgpu::Queue, layer: &mut SplatLayer, cloud: Option<(&Cloud, u64)>) {
        let Some((cloud, gen)) = cloud else {
            layer.gpu = None;
            return;
        };
        let n = cloud.count as u64;
        let buf = |label: &str, size: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size.max(16).div_ceil(4) * 4,
                usage,
                mapped_at_creation: false,
            })
        };
        use wgpu::BufferUsages as B;
        let st = B::STORAGE | B::COPY_DST;
        let splats = buf("splat-data", n * SPLAT_WORDS as u64 * 4, st);
        queue.write_buffer(&splats, 0, bytemuck::cast_slice(&cloud.splats));
        let sh = buf("splat-sh", cloud.sh.len() as u64 * 4, st);
        if !cloud.sh.is_empty() {
            queue.write_buffer(&sh, 0, bytemuck::cast_slice(&cloud.sh));
        }
        let proj = buf("splat-proj", n * 32, B::STORAGE);
        let ka = buf("splat-keys-a", n * 4, B::STORAGE);
        let va = buf("splat-vals-a", n * 4, B::STORAGE);
        let kb = buf("splat-keys-b", n * 4, B::STORAGE);
        let vb = buf("splat-vals-b", n * 4, B::STORAGE);
        let hist = buf("splat-hist", 16 * n.div_ceil(BLOCK as u64).max(1) * 4, B::STORAGE);
        let ctl = buf("splat-ctl", 64, B::STORAGE | B::INDIRECT | B::COPY_DST);
        fn e(b: u32, buf: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry { binding: b, resource: buf.as_entire_binding() }
        }
        let bg = |label: &str, l: &wgpu::BindGroupLayout, entries: &[wgpu::BindGroupEntry]| {
            device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some(label), layout: l, entries })
        };
        let pre = bg(
            "splat-project",
            &self.project_l,
            &[e(0, &layer.uniform), e(1, &splats), e(2, &sh), e(3, &proj), e(4, &ka), e(5, &va), e(6, &ctl)],
        );
        let prep = bg("splat-prepare", &self.prepare_l, &[e(0, &ctl)]);
        let mut shifts = Vec::new();
        let mut radix = Vec::new();
        for p in 0..PASSES {
            let ub = buf("splat-radix-u", 16, B::UNIFORM | B::COPY_DST);
            queue.write_buffer(&ub, 0, bytemuck::cast_slice(&[p * 4, 0, 0, 0]));
            let (ki, vi, ko, vo) = if p % 2 == 0 { (&ka, &va, &kb, &vb) } else { (&kb, &vb, &ka, &va) };
            radix.push(bg(
                "splat-radix",
                &self.radix_l,
                &[e(0, &ub), e(1, &ctl), e(2, ki), e(3, vi), e(4, ko), e(5, vo), e(6, &hist)],
            ));
            shifts.push(ub);
        }
        // An even number of passes leaves the order in A.
        let draw = bg("splat-draw", &self.draw_l, &[e(0, &layer.uniform), e(1, &proj), e(2, &ka), e(3, &va)]);
        let mut bufs = vec![splats, sh, proj, ka, va, kb, vb, hist];
        bufs.extend(shifts);
        layer.gpu = Some(Gpu { gen, count: cloud.count as u32, pre, prep, radix, draw, ctl, _bufs: bufs });
    }

    /// Hand the layer what it should draw: the cloud (re-uploaded when its
    /// generation moves) and the uniform. Marks it dirty when either, or
    /// the depth want, changed.
    #[allow(clippy::too_many_arguments)]
    pub fn set(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layer: &mut SplatLayer,
        cloud: Option<(&Cloud, u64)>,
        uni: &[u8; UNIFORM_BYTES],
        depth: bool,
    ) {
        let gen = cloud.map_or(0, |c| c.1);
        if layer.gen() != gen {
            self.upload(device, queue, layer, cloud);
            layer.hash = u64::MAX;
        }
        let hash = crate::diskcache::content_hash(&[uni, &gen.to_le_bytes(), &[depth as u8]]);
        if layer.hash == hash {
            return;
        }
        queue.write_buffer(&layer.uniform, 0, uni);
        layer.hash = hash;
        layer.depth = depth;
        layer.dirty = true;
    }

    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder, layer: &mut SplatLayer) {
        if let Some(g) = &layer.gpu {
            encoder.clear_buffer(&g.ctl, 0, None);
            let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("splat-sort"),
                timestamp_writes: None,
            });
            let groups = g.count.div_ceil(256).max(1);
            let gx = groups.min(65_535);
            cp.set_pipeline(&self.project);
            cp.set_bind_group(0, &g.pre, &[]);
            cp.dispatch_workgroups(gx, groups.div_ceil(gx), 1);
            cp.set_pipeline(&self.prepare);
            cp.set_bind_group(0, &g.prep, &[]);
            cp.dispatch_workgroups(1, 1, 1);
            for bgp in &g.radix {
                cp.set_bind_group(0, bgp, &[]);
                cp.set_pipeline(&self.hist);
                cp.dispatch_workgroups_indirect(&g.ctl, 16);
                cp.set_pipeline(&self.scan);
                cp.dispatch_workgroups(1, 1, 1);
                cp.set_pipeline(&self.scatter);
                cp.dispatch_workgroups_indirect(&g.ctl, 16);
            }
        }
        let clear = wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store };
        fn att(v: &wgpu::TextureView, ops: wgpu::Operations<wgpu::Color>) -> Option<wgpu::RenderPassColorAttachment<'_>> {
            Some(wgpu::RenderPassColorAttachment { view: v, depth_slice: None, resolve_target: None, ops })
        }
        let color = [att(&layer.view, clear)];
        let both = [att(&layer.view, clear), att(&layer.depth_view, clear)];
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("splat-draw"),
            color_attachments: if layer.depth { &both } else { &color },
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        if let Some(g) = &layer.gpu {
            pass.set_pipeline(if layer.depth { &self.draw_depth } else { &self.draw });
            pass.set_bind_group(0, &g.draw, &[]);
            pass.draw_indirect(&g.ctl, 32);
        }
        drop(pass);
        layer.dirty = false;
        layer.draws += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splat_wgsl_validates() {
        for (name, src) in modules() {
            let module = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| panic!("{name}: {e}"));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                .validate(&module)
                .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        }
    }

    #[test]
    fn scatter_stays_under_eight_kib_of_workgroup_memory() {
        // sk + sv (512 u32 each) + part (256) + dstart (16): Apple's
        // 16 KiB cliff starts well above this.
        assert!((512 * 2 + 256 + 16) * 4 <= 8 * 1024);
    }
}
