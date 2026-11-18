//! The viewer's GPU side: a decoded frame's planes go up as they are and
//! one render pass turns them into a display-encoded RGBA texture, which
//! is then drawn fit to the viewer.
//!
//! Both happen inside egui's paint callback (`prepare` uploads and
//! converts, `paint` draws), so a frame chosen during this redraw's UI is
//! on screen in this redraw: a scrub never waits a vsync for its picture.
//!
//! Planes keep their native sample type: 8-bit planes are `R8Unorm`,
//! 10/12/16-bit planes `R16Uint` (read with `textureLoad` and divided by
//! the bit depth's maximum), interleaved RGBA is `Rgba8Unorm`,
//! `Rgba16Uint` or `Rgba16Float`. Chroma is upsampled bilinearly, centre
//! sited. The matrix and range are uniforms, the sample type is a pipeline.

use egui_wgpu::wgpu;
use media::{FrameRef, Layout};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use wgpu::util::DeviceExt as _;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Kind {
    Yuv8,
    Yuv16,
    /// Semi-planar: luma, then one plane of Cb Cr pairs (NV12, P010).
    Nv8,
    Nv16,
    Rgba8,
    Rgba16,
    RgbaF16,
}

impl Kind {
    fn of(l: Layout) -> Kind {
        match l {
            Layout::Yuv { bits: 8, semi: true, .. } => Kind::Nv8,
            Layout::Yuv { semi: true, .. } => Kind::Nv16,
            Layout::Yuv { bits: 8, .. } => Kind::Yuv8,
            Layout::Yuv { .. } => Kind::Yuv16,
            Layout::Rgba8 => Kind::Rgba8,
            Layout::Rgba16 => Kind::Rgba16,
            Layout::RgbaF16 => Kind::RgbaF16,
        }
    }

    fn yuv(self) -> bool {
        matches!(self, Kind::Yuv8 | Kind::Yuv16)
    }

    fn semi(self) -> bool {
        matches!(self, Kind::Nv8 | Kind::Nv16)
    }

    /// Bindings after the uniforms.
    fn textures(self) -> u32 {
        if self.yuv() { 4 } else if self.semi() { 2 } else { 1 }
    }

    /// Texture format of plane `i`.
    fn format(self, i: usize) -> wgpu::TextureFormat {
        match self {
            Kind::Nv8 if i == 1 => wgpu::TextureFormat::Rg8Unorm,
            Kind::Nv16 if i == 1 => wgpu::TextureFormat::Rg16Uint,
            Kind::Yuv8 | Kind::Nv8 => wgpu::TextureFormat::R8Unorm,
            Kind::Yuv16 | Kind::Nv16 => wgpu::TextureFormat::R16Uint,
            Kind::Rgba8 => wgpu::TextureFormat::Rgba8Unorm,
            Kind::Rgba16 => wgpu::TextureFormat::Rgba16Uint,
            Kind::RgbaF16 => wgpu::TextureFormat::Rgba16Float,
        }
    }

    fn uint(self) -> bool {
        matches!(self, Kind::Yuv16 | Kind::Nv16 | Kind::Rgba16)
    }
}

/// The convert pass's uniforms; one 16-byte row per line.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ConvertU {
    /// kr, kb, 1 / max code value, scene-linear (1) or code values (0).
    kr: f32,
    kb: f32,
    inv_max: f32,
    linear: f32,
    /// Luma offset + scale, chroma offset + scale (range expansion).
    yoff: f32,
    yscale: f32,
    coff: f32,
    cscale: f32,
    /// Chroma subsampling shifts, has alpha, right shift of u16 samples.
    cx: u32,
    cy: u32,
    alpha: u32,
    shift: u32,
}

const CONVERT_COMMON: &str = r#"
struct U {
    kr: f32, kb: f32, inv_max: f32, linear: f32,
    yoff: f32, yscale: f32, coff: f32, cscale: f32,
    cx: u32, cy: u32, alpha: u32, shift: u32,
};
@group(0) @binding(0) var<uniform> u: U;

@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}

fn srgb_oetf(c: vec3<f32>) -> vec3<f32> {
    let x = clamp(c, vec3<f32>(0.0), vec3<f32>(1.0));
    return select(1.055 * pow(x, vec3<f32>(1.0 / 2.4)) - 0.055, x * 12.92, x <= vec3<f32>(0.0031308));
}
"#;

const CONVERT_YUV: &str = r#"
@group(0) @binding(1) var ty: texture_2d<TEX>;
@group(0) @binding(2) var tu: texture_2d<TEX>;
@group(0) @binding(3) var tv: texture_2d<TEX>;
@group(0) @binding(4) var ta: texture_2d<TEX>;

fn ld(t: texture_2d<TEX>, p: vec2<i32>) -> f32 {
    let d = vec2<i32>(textureDimensions(t)) - 1;
    return LOAD(textureLoad(t, clamp(p, vec2<i32>(0), d), 0).r);
}

// Bilinear chroma at the luma pixel `p`, centre sited.
fn chroma(t: texture_2d<TEX>, p: vec2<i32>) -> f32 {
    let s = vec2<f32>(f32(1u << u.cx), f32(1u << u.cy));
    let c = (vec2<f32>(p) + 0.5) / s - 0.5;
    let i = vec2<i32>(floor(c));
    let f = c - floor(c);
    let a = mix(ld(t, i), ld(t, i + vec2<i32>(1, 0)), f.x);
    let b = mix(ld(t, i + vec2<i32>(0, 1)), ld(t, i + vec2<i32>(1, 1)), f.x);
    return mix(a, b, f.y);
}

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let p = vec2<i32>(pos.xy);
    let y = (ld(ty, p) - u.yoff) * u.yscale;
    let cb = (chroma(tu, p) - u.coff) * u.cscale;
    let cr = (chroma(tv, p) - u.coff) * u.cscale;
    let kg = 1.0 - u.kr - u.kb;
    let r = y + 2.0 * (1.0 - u.kr) * cr;
    let b = y + 2.0 * (1.0 - u.kb) * cb;
    let g = (y - u.kr * r - u.kb * b) / kg;
    var a = 1.0;
    if (u.alpha != 0u) { a = ld(ta, p); }
    return vec4<f32>(idt(clamp(vec3<f32>(r, g, b), vec3<f32>(0.0), vec3<f32>(1.0))), a);
}
"#;

const CONVERT_NV: &str = r#"
@group(0) @binding(1) var ty: texture_2d<TEX>;
@group(0) @binding(2) var tc: texture_2d<TEX>;

fn ld(p: vec2<i32>) -> f32 {
    let d = vec2<i32>(textureDimensions(ty)) - 1;
    return LOAD(textureLoad(ty, clamp(p, vec2<i32>(0), d), 0).r);
}

fn ldc(p: vec2<i32>) -> vec2<f32> {
    let d = vec2<i32>(textureDimensions(tc)) - 1;
    return LOAD2(textureLoad(tc, clamp(p, vec2<i32>(0), d), 0).rg);
}

fn chroma(p: vec2<i32>) -> vec2<f32> {
    let s = vec2<f32>(f32(1u << u.cx), f32(1u << u.cy));
    let c = (vec2<f32>(p) + 0.5) / s - 0.5;
    let i = vec2<i32>(floor(c));
    let f = c - floor(c);
    let a = mix(ldc(i), ldc(i + vec2<i32>(1, 0)), f.x);
    let b = mix(ldc(i + vec2<i32>(0, 1)), ldc(i + vec2<i32>(1, 1)), f.x);
    return mix(a, b, f.y);
}

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let p = vec2<i32>(pos.xy);
    let y = (ld(p) - u.yoff) * u.yscale;
    let c = (chroma(p) - u.coff) * u.cscale;
    let kg = 1.0 - u.kr - u.kb;
    let r = y + 2.0 * (1.0 - u.kr) * c.y;
    let b = y + 2.0 * (1.0 - u.kb) * c.x;
    let g = (y - u.kr * r - u.kb * b) / kg;
    return vec4<f32>(idt(clamp(vec3<f32>(r, g, b), vec3<f32>(0.0), vec3<f32>(1.0))), 1.0);
}
"#;

const CONVERT_RGBA: &str = r#"
@group(0) @binding(1) var t: texture_2d<TEX>;

@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let v = LOAD4(textureLoad(t, vec2<i32>(pos.xy), 0));
    return vec4<f32>(idt(v.rgb), clamp(v.a, 0.0, 1.0));
}
"#;

/// The display pass, wobble's blit tail: exposure, the view transform,
/// gamma, the encode. `TRANSFORM` is the view transform's WGSL (it defines
/// `view_transform`), `ENCODE` the inverse EOTF over `c`; `viewtf`'s
/// uniform struct `U` is prepended.
const DISPLAY: &str = r#"
// xf: where the picture sits in the drawn rect (offset xy, size zw, in
// 0..1 of the rect); outside it is black (a reframed clip in the raster).
struct D { nearest: f32, mixv: f32, pad0: f32, pad1: f32, xf: vec4<f32>, xb: vec4<f32> };
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var smp: sampler;
@group(0) @binding(2) var<uniform> d: D;
@group(0) @binding(3) var<uniform> u: U;
// A 3D LUT for transforms built on one (AgX); a 1^3 placeholder otherwise.
@group(0) @binding(4) var lut3d: texture_3d<f32>;
@group(0) @binding(5) var texb: texture_2d<f32>;

struct V { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };

@vertex
fn vs(@builtin(vertex_index) i: u32) -> V {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var o: V;
    o.pos = vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
    o.uv = vec2<f32>(p.x, 1.0 - p.y);
    return o;
}

TRANSFORM

fn shade(rgb: vec3f) -> vec3f {
    let c = pow(max(view_transform(rgb * u.mode.y), vec3f(0.0)), vec3f(u.mode.z));
    return ENCODE;
}

fn pick(t: texture_2d<f32>, uv0: vec2<f32>, xf: vec4<f32>) -> vec3<f32> {
    let uv = (uv0 - xf.xy) / xf.zw;
    if (any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0))) {
        return vec3<f32>(-1.0);
    }
    if (d.nearest != 0.0) {
        let sz = vec2<f32>(textureDimensions(t));
        return textureLoad(t, vec2<i32>(min(uv * sz, sz - 1.0)), 0).rgb;
    }
    return textureSampleLevel(t, smp, uv, 0.0).rgb;
}

fn shown(t: texture_2d<f32>, uv0: vec2<f32>, xf: vec4<f32>) -> vec3<f32> {
    let c = pick(t, uv0, xf);
    if (c.x < -0.5) {
        return vec3<f32>(0.0);
    }
    return shade(c);
}

@fragment
fn fs(v: V) -> @location(0) vec4<f32> {
    _ = textureDimensions(lut3d);
    _ = textureDimensions(texb);
    if (d.mixv < 0.0) {
        return vec4<f32>(shown(tex, v.uv, d.xf), 1.0);
    }
    // A dissolve: both pictures through the view, mixed as displayed (as
    // NLEs' default dissolve does).
    return vec4<f32>(mix(shown(tex, v.uv, d.xf), shown(texb, v.uv, d.xb), d.mixv), 1.0);
}
"#;

/// An input transform: the clip's colourspace into the working space, as
/// the convert pass's `idt` function.
#[derive(Clone, Debug)]
pub struct Idt {
    pub key: u64,
    pub wgsl: String,
}

impl Idt {
    pub fn new(src: colour::color::Colorspace, working: colour::color::Colorspace) -> Self {
        let t = colour::color::Transform::new(src, working);
        let wgsl = if t.is_identity() {
            "fn idt(c: vec3f) -> vec3f { return c; }\n".to_string()
        } else {
            let mut helpers = std::collections::BTreeMap::new();
            let f = t.wgsl("idt", &mut helpers);
            let mut out: String = helpers.into_values().collect::<Vec<_>>().join("\n");
            out.push('\n');
            out.push_str(&f);
            out
        };
        let mut h = 0xcbf29ce484222325u64;
        for b in wgsl.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        Idt { key: h, wgsl }
    }

    /// Code values straight through (the convert's old behaviour).
    pub fn identity() -> Self {
        Self::new(colour::color::Colorspace::LINEAR_REC709, colour::color::Colorspace::LINEAR_REC709)
    }
}

/// The display side's view, per window: its pipelines by shader key, the
/// uniform, the AgX cube.
struct ViewGpu {
    layout: wgpu::BindGroupLayout,
    pipelines: HashMap<u64, wgpu::RenderPipeline>,
    uniform: wgpu::Buffer,
    lut: wgpu::TextureView,
    lut_for: Option<colour::viewtf::Gamut>,
    lut_gen: u64,
    target: wgpu::TextureFormat,
}

impl ViewGpu {
    fn new(device: &wgpu::Device, target: wgpu::TextureFormat) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("display"),
            entries: &[
                tex_entry(0, wgpu::TextureSampleType::Float { filterable: true }),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                uniform_entry(2),
                uniform_entry(3),
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D3, multisampled: false },
                    count: None,
                },
                tex_entry(5, wgpu::TextureSampleType::Float { filterable: true }),
            ],
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("view uniforms"),
            size: std::mem::size_of::<colour::viewtf::BlitUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let lut = lut_texture(device, None, [1, 1, 1], &[[0.0; 3]]);
        ViewGpu { layout, pipelines: HashMap::new(), uniform, lut, lut_for: None, lut_gen: 0, target }
    }

    /// The pipeline for `view` (built on first use), the uniform written
    /// and the cube it needs uploaded.
    fn prepare(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, view: &colour::viewtf::ViewSettings) -> u64 {
        let key = view.shader_key();
        if !self.pipelines.contains_key(&key) {
            let src = format!(
                "{}{}",
                colour::viewtf::uniform_wgsl(),
                DISPLAY.replace("TRANSFORM", &view.wgsl()).replace("ENCODE", view.encode_expr(self.target.is_srgb()))
            );
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("display"), source: wgpu::ShaderSource::Wgsl(src.into()) });
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("display"), bind_group_layouts: &[Some(&self.layout)], immediate_size: 0 });
            self.pipelines.insert(key, pipeline(device, &pl, &module, self.target));
        }
        let mut u = colour::viewtf::BlitUniform { mode: [0.0, view.exposure.exp2(), view.inv_gamma(), 0.0], ..Default::default() };
        view.fill_uniform(&mut u);
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&u));
        if let Some(d) = view.lut_display().filter(|d| Some(*d) != self.lut_for) {
            if let Some(l) = colour::viewtf::agx::lut_stack(d) {
                self.lut = lut_texture(device, Some(queue), l.extent, &l.data);
                self.lut_for = Some(d);
                self.lut_gen += 1;
            }
        }
        key
    }
}

fn lut_texture(device: &wgpu::Device, queue: Option<&wgpu::Queue>, size: [u32; 3], data: &[[f32; 3]]) -> wgpu::TextureView {
    let extent = wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: size[2] };
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("view lut3d"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    if let Some(q) = queue {
        let rgba: Vec<[f32; 4]> = data.iter().map(|c| [c[0], c[1], c[2], 1.0]).collect();
        q.write_texture(
            tex.as_image_copy(),
            bytemuck::cast_slice(&rgba),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(size[0] * 16), rows_per_image: Some(size[1]) },
            extent,
        );
    }
    tex.create_view(&Default::default())
}

struct Convert {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
}

/// The planes currently allocated, by what they can hold.
struct Planes {
    key: (Kind, u32, u32, bool),
    textures: Vec<wgpu::Texture>,
    views: Vec<wgpu::TextureView>,
}

/// Lives in egui-wgpu's callback resources.
pub struct ViewerGpu {
    /// Convert pipelines by plane kind and input transform.
    converts: HashMap<(Kind, u64), Convert>,
    /// EXPERIMENT (fold): pipelines that bring a texture already on the GPU
    /// (a wobble comp's output) into the working space, by transform.
    ext: HashMap<u64, Convert>,
    convert_u: wgpu::Buffer,
    display_u: wgpu::Buffer,
    sampler: wgpu::Sampler,
    planes: Option<Planes>,
    /// The scene-linear picture, in the working space.
    out: Option<(wgpu::Texture, wgpu::TextureView)>,
    /// The display bind group this redraw (built in `prepare`).
    bind: Option<wgpu::BindGroup>,
    dummy: wgpu::TextureView,
    dummy_u: wgpu::TextureView,
    /// Which frame (and input transform) `out` holds.
    pub shown: Option<u64>,
    /// CPU time of the last upload (`write_texture` copies), ms.
    pub upload_ms: f32,
    pub size: (u32, u32),
}

impl ViewerGpu {
    pub fn new(device: &wgpu::Device) -> Self {
        let convert_u = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("convert uniforms"),
            size: std::mem::size_of::<ConvertU>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let display_u = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("display uniforms"),
            contents: bytemuck::cast_slice(&[0f32; 12]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("display"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let tiny = |format, label| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        Self {
            converts: HashMap::new(),
            ext: HashMap::new(),
            convert_u,
            display_u,
            sampler,
            planes: None,
            out: None,
            bind: None,
            dummy: tiny(wgpu::TextureFormat::R8Unorm, "dummy plane"),
            dummy_u: tiny(wgpu::TextureFormat::R16Uint, "dummy plane u16"),
            shown: None,
            upload_ms: 0.0,
            size: (0, 0),
        }
    }

    /// EXPERIMENT (fold): a picture that is already a texture on this device
    /// (scene-linear RGBA, `w`x`h`) through `idt` into `out`. No upload.
    fn show_ext(&mut self, device: &wgpu::Device, enc: &mut wgpu::CommandEncoder, src: &wgpu::TextureView, (w, h): (u32, u32), idt: &Idt) {
        if self.size != (w, h) || self.out.is_none() {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("viewer picture"),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = tex.create_view(&Default::default());
            self.out = Some((tex, view));
            self.size = (w, h);
        }
        let conv = self.ext.entry(idt.key).or_insert_with(|| {
            let src = format!("{EXT_VS}{}{EXT_FS}", idt.wgsl);
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("ext convert"), source: wgpu::ShaderSource::Wgsl(src.into()) });
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("ext convert"), entries: &[tex_entry(0, wgpu::TextureSampleType::Float { filterable: false })] });
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("ext convert"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
            Convert { pipeline: pipeline(device, &pl, &module, wgpu::TextureFormat::Rgba16Float), layout }
        });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ext convert"),
            layout: &conv.layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(src) }],
        });
        let target = &self.out.as_ref().expect("allocated above").1;
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("ext convert"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
            })],
            ..Default::default()
        });
        pass.set_pipeline(&conv.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.draw(0..3, 0..1);
    }

    /// Upload `frame` and convert it into `out`: decoded to code values,
    /// then the clip's input transform into the working space.
    fn show(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, enc: &mut wgpu::CommandEncoder, frame: &media::Frame, idt: &Idt) {
        let kind = Kind::of(frame.layout);
        let (w, h) = (frame.width, frame.height);
        let has_alpha = matches!(frame.layout, Layout::Yuv { alpha: true, .. });
        let key = (kind, w, h, has_alpha);
        if self.planes.as_ref().is_none_or(|p| p.key != key) {
            let mut textures = Vec::new();
            for i in 0..frame.layout.plane_count() {
                let (pw, ph) = frame.layout.plane_size(i, w, h);
                textures.push(device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("plane"),
                    size: wgpu::Extent3d { width: pw, height: ph, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: kind.format(i),
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                }));
            }
            let views = textures.iter().map(|t| t.create_view(&Default::default())).collect();
            self.planes = Some(Planes { key, textures, views });
        }
        if self.size != (w, h) || self.out.is_none() {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("viewer picture"),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = tex.create_view(&Default::default());
            self.out = Some((tex, view));
            self.size = (w, h);
        }

        // Upload: the plane bytes as they are, the decoder's stride and all.
        let t0 = Instant::now();
        let planes = self.planes.as_ref().expect("allocated above");
        for (i, tex) in planes.textures.iter().enumerate() {
            let p = frame.plane(i);
            queue.write_texture(
                tex.as_image_copy(),
                p.data,
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(p.stride as u32), rows_per_image: None },
                wgpu::Extent3d { width: p.width, height: p.height, depth_or_array_layers: 1 },
            );
        }
        self.upload_ms = t0.elapsed().as_secs_f32() * 1000.0;

        queue.write_buffer(&self.convert_u, 0, bytemuck::bytes_of(&convert_uniforms(frame.layout)));
        let conv = self.converts.entry((kind, idt.key)).or_insert_with(|| make_convert(device, kind, &idt.wgsl));
        let mut entries = vec![wgpu::BindGroupEntry { binding: 0, resource: self.convert_u.as_entire_binding() }];
        let n = kind.textures() as usize;
        let dummy = if kind.uint() { &self.dummy_u } else { &self.dummy };
        for i in 0..n {
            let v = planes.views.get(i).unwrap_or(dummy);
            entries.push(wgpu::BindGroupEntry { binding: 1 + i as u32, resource: wgpu::BindingResource::TextureView(v) });
        }
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("convert"), layout: &conv.layout, entries: &entries });
        let target = &self.out.as_ref().expect("allocated above").1;
        let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("convert"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
            })],
            ..Default::default()
        });
        pass.set_pipeline(&conv.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.draw(0..3, 0..1);
    }
}

fn convert_uniforms(l: Layout) -> ConvertU {
    let mut u = ConvertU {
        kr: 0.2126,
        kb: 0.0722,
        inv_max: 1.0,
        linear: 0.0,
        yoff: 0.0,
        yscale: 1.0,
        coff: 0.5,
        cscale: 1.0,
        cx: 0,
        cy: 0,
        alpha: 0,
        shift: 0,
    };
    match l {
        Layout::Yuv { bits, cx, cy, alpha, matrix, full_range, shift, .. } => {
            u.shift = shift as u32;
            let max = ((1u32 << bits) - 1) as f32;
            let s = (1u32 << (bits - 8)) as f32;
            (u.kr, u.kb) = matrix.kr_kb();
            u.inv_max = 1.0 / max;
            if full_range {
                u.coff = (1u32 << (bits - 1)) as f32 / max;
            } else {
                u.yoff = 16.0 * s / max;
                u.yscale = max / (219.0 * s);
                u.coff = 128.0 * s / max;
                u.cscale = max / (224.0 * s);
            }
            (u.cx, u.cy, u.alpha) = (cx as u32, cy as u32, alpha as u32);
        }
        Layout::Rgba16 => u.inv_max = 1.0 / 65535.0,
        Layout::RgbaF16 => u.linear = 1.0,
        Layout::Rgba8 => {}
    }
    u
}

fn tex_entry(binding: u32, sample_type: wgpu::TextureSampleType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture { sample_type, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }
}

fn pipeline(device: &wgpu::Device, layout: &wgpu::PipelineLayout, module: &wgpu::ShaderModule, format: wgpu::TextureFormat) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(layout),
        vertex: wgpu::VertexState { module, entry_point: Some("vs"), buffers: &[], compilation_options: Default::default() },
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some("fs"),
            targets: &[Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

fn make_convert(device: &wgpu::Device, k: Kind, idt: &str) -> Convert {
    let (tex, load, load2, load4) = if k.uint() {
        (
            "u32",
            "(f32(X >> u.shift) * u.inv_max)",
            "(vec2<f32>(X >> vec2<u32>(u.shift)) * u.inv_max)",
            "(vec4<f32>(X) * u.inv_max)",
        )
    } else {
        ("f32", "(X)", "(X)", "(X)")
    };
    let body = if k.yuv() {
        CONVERT_YUV
    } else if k.semi() {
        CONVERT_NV
    } else {
        CONVERT_RGBA
    };
    // LOAD(expr) -> the expression normalised to 0..1.
    let body = body.replace("TEX", tex);
    let body = expand_macro(&body, "LOAD4", load4);
    let body = expand_macro(&body, "LOAD2", load2);
    let body = expand_macro(&body, "LOAD", load);
    let src = format!("{CONVERT_COMMON}{idt}{body}");
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("convert"),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    });
    let sample = if k.uint() { wgpu::TextureSampleType::Uint } else { wgpu::TextureSampleType::Float { filterable: false } };
    let mut entries = vec![uniform_entry(0)];
    for i in 0..k.textures() {
        entries.push(tex_entry(1 + i, sample));
    }
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("convert"), entries: &entries });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("convert"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    Convert { pipeline: pipeline(device, &pl, &module, wgpu::TextureFormat::Rgba16Float), layout }
}

/// Replace `NAME(arg)` with `template` where `X` stands for `arg`
/// (arguments may hold balanced parentheses).
fn expand_macro(src: &str, name: &str, template: &str) -> String {
    let pat = format!("{name}(");
    let mut out = String::new();
    let mut rest = src;
    while let Some(i) = rest.find(&pat) {
        // Not part of a longer name (LOAD inside LOAD4).
        let prev = rest[..i].chars().last();
        if prev.is_some_and(|c| c.is_alphanumeric() || c == '_') {
            out.push_str(&rest[..i + pat.len()]);
            rest = &rest[i + pat.len()..];
            continue;
        }
        out.push_str(&rest[..i]);
        let args = &rest[i + pat.len()..];
        let mut depth = 1;
        let mut end = 0;
        for (j, c) in args.char_indices() {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        end = j;
                        break;
                    }
                }
                _ => {}
            }
        }
        out.push_str(&template.replace('X', &args[..end]));
        rest = &args[end + 1..];
    }
    out.push_str(rest);
    out
}

/// Every viewer's GPU state, by slot (0 source, 1 record), and the view
/// (display transform) they share. One per window: a torn-off window has
/// its own.
pub struct Viewers {
    pub slots: Vec<ViewerGpu>,
    view: ViewGpu,
}

impl Viewers {
    pub fn new(device: &wgpu::Device, target: wgpu::TextureFormat, n: usize) -> Self {
        Self { slots: (0..n).map(|_| ViewerGpu::new(device)).collect(), view: ViewGpu::new(device, target) }
    }
}

/// The paint callback: carries the frame to show this redraw.
pub struct ViewerPaint {
    pub slot: usize,
    pub frame: Option<(u64, FrameRef)>,
    /// The frame's input transform (its colourspace into the working one).
    pub idt: Arc<Idt>,
    /// The project's view: transform, display, exposure, gamma.
    pub view: colour::viewtf::ViewSettings,
    /// Nothing to show here (a gap in the timeline): draw no picture.
    pub black: bool,
    /// Where the picture sits in the drawn rect (offset, size; 0..1).
    pub xf: [f32; 4],
    /// A dissolve: the incoming picture, converted in slot `.0`, its input
    /// transform, how much of it to show, and where it sits.
    pub mix: Option<(usize, u64, FrameRef, Arc<Idt>, f32, [f32; 4])>,
    /// Magnified past 1:1: show pixels, not a bilinear ramp between them.
    pub nearest: bool,
    /// EXPERIMENT (fold): show this texture (a wobble comp's output, its
    /// size, its space into the working one) in place of `frame`.
    pub ext: Option<(wgpu::TextureView, (u32, u32), Arc<Idt>)>,
}

impl egui_wgpu::CallbackTrait for ViewerPaint {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen: &egui_wgpu::ScreenDescriptor,
        enc: &mut wgpu::CommandEncoder,
        res: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let Some(all) = res.get_mut::<Viewers>() else { return vec![] };
        let vkey = all.view.prepare(device, queue, &self.view);
        if let Some((view, size, idt)) = &self.ext {
            if let Some(v) = all.slots.get_mut(self.slot) {
                // Live: its pixels change under the same view, so every
                // redraw converts; the next footage frame uploads afresh.
                v.show_ext(device, enc, view, *size, idt);
                v.shown = None;
            }
        }
        let mut show = |v: &mut ViewerGpu, key: u64, frame: &FrameRef, idt: &Idt| {
            let k = key ^ idt.key.rotate_left(17);
            if v.shown != Some(k) {
                v.show(device, queue, enc, frame, idt);
                v.shown = Some(k);
            }
        };
        if self.ext.is_some() {
        } else if let Some((key, frame)) = &self.frame {
            if let Some(v) = all.slots.get_mut(self.slot) {
                show(v, *key, frame, &self.idt);
            }
        }
        if let Some((bslot, key, frame, idt, _, _)) = &self.mix {
            if let Some(b) = all.slots.get_mut(*bslot) {
                show(b, *key, frame, idt);
            }
        }
        let x = self.xf;
        let (m, xb, bslot) = match &self.mix {
            Some((bs, _, _, _, m, xb)) => (*m, *xb, *bs),
            None => (-1.0, [0.0, 0.0, 1.0, 1.0], self.slot),
        };
        let d = [self.nearest as u32 as f32, m, 0.0, 0.0, x[0], x[1], x[2], x[3], xb[0], xb[1], xb[2], xb[3]];
        let _ = vkey;
        let (Some(a_view), b_view) = (all.slots.get(self.slot).and_then(|s| s.out.as_ref().map(|o| o.1.clone())), all.slots.get(bslot).and_then(|s| s.out.as_ref().map(|o| o.1.clone()))) else {
            return vec![];
        };
        let b_view = b_view.unwrap_or_else(|| a_view.clone());
        let Some(v) = all.slots.get_mut(self.slot) else { return vec![] };
        queue.write_buffer(&v.display_u, 0, bytemuck::cast_slice(&d));
        v.bind = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("display"),
            layout: &all.view.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&a_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&v.sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: v.display_u.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: all.view.uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&all.view.lut) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&b_view) },
            ],
        }));
        vec![]
    }

    fn paint(&self, _info: egui::PaintCallbackInfo, pass: &mut wgpu::RenderPass<'static>, res: &egui_wgpu::CallbackResources) {
        if self.black {
            return;
        }
        let Some(all) = res.get::<Viewers>() else { return };
        let Some(p) = all.view.pipelines.get(&self.view.shader_key()) else { return };
        let Some(v) = all.slots.get(self.slot) else { return };
        let Some(bg) = &v.bind else { return };
        pass.set_pipeline(p);
        pass.set_bind_group(0, bg, &[]);
        pass.draw(0..3, 0..1);
    }
}

const EXT_VS: &str = "
@group(0) @binding(0) var t: texture_2d<f32>;
@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
}
";
const EXT_FS: &str = "
@fragment
fn fs(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let c = textureLoad(t, vec2<i32>(pos.xy), 0);
    return vec4<f32>(idt(c.rgb), c.a);
}
";

#[cfg(feature = "fold")]
/// EXPERIMENT (fold): a texture drawn as it is into a rect of the UI (the
/// hosted wobble window). Lives in egui-wgpu's callback resources.
pub struct TexBlit {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    bind: Option<wgpu::BindGroup>,
}

#[cfg(feature = "fold")]
impl TexBlit {
    pub fn new(device: &wgpu::Device, target: wgpu::TextureFormat) -> Self {
        let src = "
@group(0) @binding(0) var t: texture_2d<f32>;
@group(0) @binding(1) var s: sampler;
struct V { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex
fn vs(@builtin(vertex_index) i: u32) -> V {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    var o: V;
    o.pos = vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
    o.uv = vec2<f32>(p.x, 1.0 - p.y);
    return o;
}
@fragment
fn fs(v: V) -> @location(0) vec4<f32> {
    return vec4<f32>(textureSampleLevel(t, s, v.uv, 0.0).rgb, 1.0);
}
";
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("tex blit"), source: wgpu::ShaderSource::Wgsl(src.into()) });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tex blit"),
            entries: &[
                tex_entry(0, wgpu::TextureSampleType::Float { filterable: true }),
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("tex blit"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor { label: Some("tex blit"), mag_filter: wgpu::FilterMode::Nearest, min_filter: wgpu::FilterMode::Linear, ..Default::default() });
        TexBlit { pipeline: pipeline(device, &pl, &module, target), layout, sampler, bind: None }
    }
}

#[cfg(feature = "fold")]
pub struct TexPaint {
    pub view: wgpu::TextureView,
}

#[cfg(feature = "fold")]
impl egui_wgpu::CallbackTrait for TexPaint {
    fn prepare(&self, device: &wgpu::Device, _queue: &wgpu::Queue, _screen: &egui_wgpu::ScreenDescriptor, _enc: &mut wgpu::CommandEncoder, res: &mut egui_wgpu::CallbackResources) -> Vec<wgpu::CommandBuffer> {
        if let Some(b) = res.get_mut::<TexBlit>() {
            b.bind = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tex blit"),
                layout: &b.layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&b.sampler) },
                ],
            }));
        }
        vec![]
    }

    fn paint(&self, _info: egui::PaintCallbackInfo, pass: &mut wgpu::RenderPass<'static>, res: &egui_wgpu::CallbackResources) {
        let Some(b) = res.get::<TexBlit>() else { return };
        let Some(bg) = &b.bind else { return };
        pass.set_pipeline(&b.pipeline);
        pass.set_bind_group(0, bg, &[]);
        pass.draw(0..3, 0..1);
    }
}

/// The viewer's passes offscreen, for export and stills: a picture (or a
/// dissolve's two) through its input transform and the view into an RGBA8
/// frame of the output's size, read back. Its own device, so a render on a
/// worker never touches the window's.
pub struct Offscreen {
    gpu: imfram::shell::Gpu,
    viewers: Viewers,
    target: wgpu::Texture,
    view: wgpu::TextureView,
    buf: wgpu::Buffer,
    w: u32,
    h: u32,
    row: u32,
    count: u64,
}

/// One picture to draw: the frame, its input transform, and where it sits
/// in the output (offset, size; 0..1 of the output).
pub struct Layer<'a> {
    pub frame: &'a media::Frame,
    pub idt: &'a Idt,
    pub xf: [f32; 4],
}

impl Offscreen {
    pub fn new(w: u32, h: u32) -> Option<Self> {
        let gpu = imfram::shell::Gpu::standalone()?;
        let fmt = wgpu::TextureFormat::Rgba8Unorm;
        let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("render frame"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: fmt,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&Default::default());
        let row = (w * 4).div_ceil(256) * 256;
        let buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("render readback"),
            size: (row * h) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let viewers = Viewers::new(&gpu.device, fmt, 2);
        Some(Offscreen { gpu, viewers, target, view, buf, w, h, row, count: 0 })
    }

    /// `a` (and `b` mixed in by `m`, a dissolve) through `view`: tight
    /// RGBA8 rows, encoded for the display `view` names.
    pub fn frame(&mut self, a: Option<Layer>, b: Option<Layer>, m: f32, view: &colour::viewtf::ViewSettings) -> Vec<u8> {
        let (device, queue) = (&self.gpu.device, &self.gpu.queue);
        let mut enc = device.create_command_encoder(&Default::default());
        let key = self.viewers.view.prepare(device, queue, view);
        self.count += 1;
        let mut out = [None, None];
        for (i, l) in [a.as_ref(), b.as_ref()].into_iter().enumerate() {
            if let Some(l) = l {
                let v = &mut self.viewers.slots[i];
                v.show(device, queue, &mut enc, l.frame, l.idt);
                v.shown = None;
                out[i] = v.out.as_ref().map(|o| o.1.clone());
            }
        }
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("render frame"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                ..Default::default()
            });
            if let Some(av) = out[0].clone() {
                let bv = out[1].clone().unwrap_or_else(|| av.clone());
                let (xa, xb) = (a.as_ref().map_or([0.0, 0.0, 1.0, 1.0], |l| l.xf), b.as_ref().map_or([0.0, 0.0, 1.0, 1.0], |l| l.xf));
                let mv = if out[1].is_some() { m } else { -1.0 };
                let d = [0.0f32, mv, 0.0, 0.0, xa[0], xa[1], xa[2], xa[3], xb[0], xb[1], xb[2], xb[3]];
                let v = &self.viewers.slots[0];
                queue.write_buffer(&v.display_u, 0, bytemuck::cast_slice(&d));
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("render display"),
                    layout: &self.viewers.view.layout,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&av) },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&v.sampler) },
                        wgpu::BindGroupEntry { binding: 2, resource: v.display_u.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: self.viewers.view.uniform.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&self.viewers.view.lut) },
                        wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&bv) },
                    ],
                });
                pass.set_pipeline(&self.viewers.view.pipelines[&key]);
                pass.set_bind_group(0, &bg, &[]);
                pass.draw(0..3, 0..1);
            }
        }
        enc.copy_texture_to_buffer(
            self.target.as_image_copy(),
            wgpu::TexelCopyBufferInfo { buffer: &self.buf, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(self.row), rows_per_image: Some(self.h) } },
            wgpu::Extent3d { width: self.w, height: self.h, depth_or_array_layers: 1 },
        );
        queue.submit([enc.finish()]);
        let slice = self.buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let mut px = Vec::with_capacity((self.w * self.h * 4) as usize);
        {
            let Ok(data) = slice.get_mapped_range() else { return vec![0; (self.w * self.h * 4) as usize] };
            for y in 0..self.h as usize {
                let o = y * self.row as usize;
                px.extend_from_slice(&data[o..o + self.w as usize * 4]);
            }
        }
        self.buf.unmap();
        px
    }
}

#[cfg(test)]
mod tests {
    use super::{expand_macro, Idt, Layer, Offscreen};
    use colour::color::Colorspace;
    use colour::viewtf::{Eotf, Gamut, ViewSettings, ViewTransform};

    #[allow(dead_code, clippy::excessive_precision)]
    mod ocio {
        include!("../../../engine/tests/aces12_ref/tables.rs");
    }

    type Table = &'static [([f32; 3], [f32; 3])];

    /// The real passes, convert then display, over a row of float pixels:
    /// the worst difference from `want`, in 8-bit codes.
    fn worst(input: Colorspace, eotf: Eotf, t: Table, want: impl Fn(usize) -> [f32; 3]) -> Option<i32> {
        let n = t.len() as u32;
        let mut off = Offscreen::new(n, 1)?;
        let bytes: Vec<u8> = t
            .iter()
            .flat_map(|(c, _)| [c[0], c[1], c[2], 1.0])
            .flat_map(|v| half::f16::from_f32(v).to_le_bytes())
            .collect();
        let frame = media::Frame::owned(n, 1, media::Layout::RgbaF16, vec![bytes]);
        let idt = Idt::new(input, Colorspace::ACESCG);
        let view = ViewSettings { transform: ViewTransform::Aces12, working: Gamut::AcesCg, display: Gamut::Rec709, eotf, ..Default::default() };
        let px = off.frame(Some(Layer { frame: &frame, idt: &idt, xf: [0.0, 0.0, 1.0, 1.0] }), None, 0.0, &view);
        let mut w = 0;
        for i in 0..t.len() {
            for k in 0..3 {
                w = w.max((px[i * 4 + k] as i32 - (want(i)[k] * 255.0).round() as i32).abs());
            }
        }
        Some(w)
    }

    /// `sRGB (ACES)` and `Rec.709 (ACES)` on screen are OCIO's numbers.
    #[test]
    fn the_aces12_views_match_ocio_through_the_viewer() {
        for (eotf, t) in [(Eotf::Rec1886, ocio::ANALYTIC_FWD_REC709), (Eotf::Srgb, ocio::ANALYTIC_FWD_SRGB)] {
            let Some(w) = worst(Colorspace::ACESCG, eotf, t, |i| t[i].1) else { return };
            assert!(w <= 1, "{eotf:?}: {w}/255");
        }
    }

    /// A plate read as an Output space and shown through that view is the
    /// plate: the input transform in the convert pass, the view after it.
    #[test]
    fn an_output_input_comes_back_through_its_view() {
        for (cs, eotf, t) in [
            (Colorspace::ACES_OUTPUT_REC709, Eotf::Rec1886, ocio::ANALYTIC_INV_REC709),
            (Colorspace::ACES_OUTPUT_SRGB, Eotf::Srgb, ocio::ANALYTIC_INV_SRGB),
        ] {
            let Some(w) = worst(cs, eotf, t, |i| t[i].0) else { return };
            assert!(w <= 2, "{eotf:?}: {w}/255");
        }
    }

    #[test]
    fn a_macro_expands_with_nested_parentheses_and_skips_longer_names() {
        let s = expand_macro("a = LOAD(f(x, y)); b = LOAD4(z);", "LOAD", "[X]");
        assert_eq!(s, "a = [f(x, y)]; b = LOAD4(z);");
    }
}
