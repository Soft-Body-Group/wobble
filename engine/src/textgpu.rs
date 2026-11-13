//! The Text node's layer: glyph instances (`crate::textfx`) drawn as quads
//! over the font's distance-field atlas into a per-node premultiplied
//! texture the fused pass samples (`SourceBinding::Text`).
//!
//! Two draws: first every glyph's shadow and glow (so one glyph's shadow
//! never lands on its neighbour's face), then every face — fill and outline
//! from one distance read. The layer is re-recorded only when what it would
//! draw changes (`TextLayer::hash`: the instances, the uniform, the atlas,
//! the size), on the Geo layer's memo.

use crate::textfx::{Atlas, Instance, Uniform, SPREAD};
use std::sync::Arc;

pub const TEXT_LAYER_WGSL: &str = r#"
struct Inst { m: vec4f, t: vec4f, c: vec4f, fill: vec4f, line: vec4f, x: vec4f };
struct U { dims: vec4f, shadow: vec4f, shadow_col: vec4f, glow: vec4f, glow_col: vec4f, pad: vec4f };
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> inst: array<Inst>;
@group(0) @binding(2) var atlas: texture_2d<f32>;
@group(0) @binding(3) var samp: sampler;

struct VO {
    @builtin(position) pos: vec4f,
    @location(0) q: vec2f,
    @location(1) @interpolate(flat) i: u32,
    @location(2) @interpolate(flat) kind: u32,
};

const SPREAD2: f32 = SPREADX2;

fn quad(v: u32, i: u32, kind: u32, off: vec2f) -> VO {
    let g = inst[i];
    let corner = vec2f(f32(v & 1u), f32(v >> 1u));
    let q = corner * g.c.xy;
    let p = g.t.xy + vec2f(g.m.x * q.x + g.m.z * q.y, g.m.y * q.x + g.m.w * q.y) + off;
    var o: VO;
    o.pos = vec4f(p.x / u.dims.x * 2.0 - 1.0, 1.0 - p.y / u.dims.y * 2.0, 0.0, 1.0);
    o.q = q;
    o.i = i;
    o.kind = kind;
    return o;
}

@vertex fn vs_under(@builtin(vertex_index) v: u32, @builtin(instance_index) ii: u32) -> VO {
    let n = u32(u.glow.z);
    if (ii < n) { return quad(v, ii, 0u, u.shadow.xy); }
    return quad(v, ii - n, 1u, vec2f(0.0));
}

@vertex fn vs_face(@builtin(vertex_index) v: u32, @builtin(instance_index) ii: u32) -> VO {
    return quad(v, ii, 2u, vec2f(0.0));
}

fn dist(g: Inst, q: vec2f) -> f32 {
    let a = textureSampleLevel(atlas, samp, (g.t.zw + q) / u.dims.zw, 0.0).r;
    return (a - 0.5) * SPREAD2;
}

fn px_per_atlas(g: Inst) -> f32 {
    return sqrt(max(abs(g.m.x * g.m.w - g.m.y * g.m.z), 1e-12));
}

fn edge(d: f32, th: f32, w: f32) -> f32 {
    return clamp((d - th) / w + 0.5, 0.0, 1.0);
}

fn pm(c: vec4f) -> vec4f { return vec4f(c.rgb * c.a, c.a); }

@fragment fn fs_under(o: VO) -> @location(0) vec4f {
    let g = inst[o.i];
    let d = dist(g, o.q);
    let s = px_per_atlas(g);
    let th = -(g.c.z + g.c.w);
    let ink = max(g.fill.a, select(0.0, g.line.a, g.c.z > 0.0));
    if (o.kind == 0u) {
        let a = edge(d, th, 1.0 / s + g.x.y + u.shadow.z);
        return pm(u.shadow_col) * a * g.x.x * ink;
    }
    let gw = max(u.glow.x, 1e-3);
    let t = clamp(1.0 + (d - th) / gw, 0.0, 1.0);
    return pm(u.glow_col) * (t * t) * g.x.x * ink;
}

@fragment fn fs_face(o: VO) -> @location(0) vec4f {
    let g = inst[o.i];
    let d = dist(g, o.q);
    let w = 1.0 / px_per_atlas(g) + g.x.y;
    let cf = edge(d, -g.c.w, w);
    let co = select(cf, edge(d, -g.c.w - g.c.z, w), g.c.z > 0.0);
    return (pm(g.fill) * cf + pm(g.line) * max(co - cf, 0.0)) * g.x.x;
}
"#;

fn wgsl() -> String {
    TEXT_LAYER_WGSL.replace("SPREADX2", &format!("{:.1}", 2.0 * SPREAD))
}

pub struct TextRaster {
    under: wgpu::RenderPipeline,
    face: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
}

/// One Text node's layer and what it was drawn from.
pub struct TextLayer {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub width: u32,
    pub height: u32,
    /// What the layer holds (instances + uniform + atlas); `u64::MAX` until
    /// the first upload.
    pub hash: u64,
    pub dirty: bool,
    /// Each drawn glyph's quad in canvas px, for the viewer.
    pub quads: Vec<[[f32; 2]; 4]>,
    atlas: Option<(u64, wgpu::Texture, wgpu::TextureView)>,
    uniform: wgpu::Buffer,
    inst: Option<(wgpu::Buffer, usize)>,
    bind: Option<wgpu::BindGroup>,
    count: u32,
    under: bool,
}

impl TextRaster {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("text-layer"),
            source: wgpu::ShaderSource::Wgsl(wgsl().into()),
        });
        let vf = wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT;
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("text-layer"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: vf,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: vf,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("text-layer"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let make = |vs: &str, fs: &str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("text-layer"),
                layout: Some(&pl),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some(vs),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(fs),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
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
            })
        };
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("text-atlas"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Self { under: make("vs_under", "fs_under"), face: make("vs_face", "fs_face"), layout, sampler }
    }

    pub fn make_layer(&self, device: &wgpu::Device, w: u32, h: u32) -> TextLayer {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("text-layer"),
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
            label: Some("text-layer-u"),
            size: std::mem::size_of::<Uniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        TextLayer {
            texture,
            view,
            width: w,
            height: h,
            hash: u64::MAX,
            dirty: true,
            quads: Vec::new(),
            atlas: None,
            uniform,
            inst: None,
            bind: None,
            count: 0,
            under: false,
        }
    }

    /// Upload what the layer should draw; marks it dirty when it differs
    /// from what it holds. `hash` covers everything passed.
    pub fn set(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layer: &mut TextLayer,
        atlas: &Arc<Atlas>,
        inst: &[Instance],
        mut uni: Uniform,
        hash: u64,
    ) {
        if layer.hash == hash {
            return;
        }
        let mut rebind = false;
        if layer.atlas.as_ref().is_none_or(|a| a.0 != atlas.hash) {
            let size = wgpu::Extent3d { width: atlas.width, height: atlas.height, depth_or_array_layers: 1 };
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("text-glyphs"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(&atlas.texels),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(atlas.width * 2),
                    rows_per_image: Some(atlas.height),
                },
                size,
            );
            let view = texture.create_view(&Default::default());
            layer.atlas = Some((atlas.hash, texture, view));
            rebind = true;
        }
        let n = inst.len().max(1);
        if layer.inst.as_ref().is_none_or(|(_, cap)| *cap < n) {
            let cap = n.next_power_of_two();
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("text-glyph-instances"),
                size: (cap * std::mem::size_of::<Instance>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            layer.inst = Some((buf, cap));
            rebind = true;
        }
        if !inst.is_empty() {
            queue.write_buffer(&layer.inst.as_ref().unwrap().0, 0, bytemuck::cast_slice(inst));
        }
        uni.dims = [layer.width as f32, layer.height as f32, atlas.width as f32, atlas.height as f32];
        uni.glow[2] = inst.len() as f32;
        queue.write_buffer(&layer.uniform, 0, bytemuck::bytes_of(&uni));
        if rebind || layer.bind.is_none() {
            layer.bind = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("text-layer"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: layer.uniform.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: layer.inst.as_ref().unwrap().0.as_entire_binding() },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&layer.atlas.as_ref().unwrap().2),
                    },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                ],
            }));
        }
        layer.count = inst.len() as u32;
        layer.under = uni.shadow[3] > 0.0 || uni.glow[1] > 0.0;
        layer.hash = hash;
        layer.dirty = true;
    }

    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder, layer: &mut TextLayer) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("text-layer"),
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
        if let (Some(bind), true) = (&layer.bind, layer.count > 0) {
            pass.set_bind_group(0, bind, &[]);
            if layer.under {
                // Shadows then glows, every glyph: instances 0..2n.
                pass.set_pipeline(&self.under);
                pass.draw(0..4, 0..layer.count * 2);
            }
            pass.set_pipeline(&self.face);
            pass.draw(0..4, 0..layer.count);
        }
        layer.dirty = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_layer_wgsl_validates() {
        let module = naga::front::wgsl::parse_str(&wgsl()).expect("parses");
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&module)
            .expect("validates");
    }

    #[test]
    fn the_structs_are_the_size_the_shader_reads() {
        assert_eq!(std::mem::size_of::<Instance>(), 96);
        assert_eq!(std::mem::size_of::<Uniform>(), 96);
    }
}
