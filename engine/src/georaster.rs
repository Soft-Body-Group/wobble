//! Vector rasteriser for 2dini geometry: the flattened [`DrawList`] is
//! drawn into a persistent per-node LAYER texture, exactly as Paint strokes
//! are. The layer is a memoised rasterisation — re-recorded only when the
//! draw list's bytes (or the output size) change — so a static drawing costs
//! nothing per frame and editing a shape never recompiles a shader. The
//! fused pass samples the layer like any other source.
//!
//! One instanced quad per shape, covering its padded bbox (a big shape is
//! several, tiled by `geo::flatten` so its empty corners and flat interior
//! skip the walk); the fragment shader walks the shape's segments (a
//! storage buffer) for the signed distance and the integer winding, so
//! fill is exact under either rule and the stroke is a band about the
//! outline. Anti-aliasing is a one-pixel ramp on the distance, which is
//! the exact box filter on a straight edge. Shapes composite in draw
//! order, premultiplied, stroke over fill.

use crate::geo::{DrawList, FieldCx, PrimGpu};
use crate::graph::GeoParams;

pub const GEO_WGSL: &str = r#"
struct U { size: vec4f } // xy = layer size in px
struct Prim {
    bbox: vec4f,
    fill: vec4f,
    stroke: vec4f,
    first: u32,
    count: u32,
    kind: u32,
    rule: u32,
    width: f32,
    radius: f32,
    blend: u32,
    pad: f32,
    grad_first: u32,
    grad_n: u32,
    pad3: vec2f,
    grad_ab: vec4f,
    grad_kind: u32,
    grad_depth: f32,
    pad2: vec2f,
}

// The fill at `p`: flat, or run from the fill to the gradient's colour.
fn fill_at(pr: Prim, p: vec2f, sd: f32) -> vec4f {
    var t = 0.0;
    if (pr.grad_kind == 0u) { return pr.fill; }
    if (pr.grad_kind == 1u) {
        let e = pr.grad_ab.zw - pr.grad_ab.xy;
        t = dot(p - pr.grad_ab.xy, e) / max(dot(e, e), 1e-6);
    } else if (pr.grad_kind == 2u) {
        t = length(p - pr.grad_ab.xy) / max(length(pr.grad_ab.zw - pr.grad_ab.xy), 1e-3);
    } else {
        t = -sd / pr.grad_depth;
    }
    // The ramp's samples, premultiplied, lerped.
    let x = clamp(t, 0.0, 1.0) * f32(pr.grad_n - 1u);
    let i = min(u32(x), pr.grad_n - 2u);
    let f = x - f32(i);
    let at = pr.grad_first + 2u * i;
    let c0 = vec4f(verts[at], verts[at + 1u]);
    let c1 = vec4f(verts[at + 2u], verts[at + 3u]);
    return mix(c0, c1, f);
}
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> prims: array<Prim>;
@group(0) @binding(2) var<storage, read> verts: array<vec2f>;

struct VOut {
    @builtin(position) pos: vec4f,
    @location(0) px: vec2f,
    @location(1) @interpolate(flat) prim: u32,
}

@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
    let b = prims[ii].bbox;
    // Triangle-strip quad over the bbox.
    let c = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u)) * 0.5;
    let px = mix(b.xy, b.zw, c);
    var o: VOut;
    o.pos = vec4f(px / u.size.xy * 2.0 - 1.0, 0.0, 1.0);
    o.pos.y = -o.pos.y;
    o.px = px;
    o.prim = ii;
    return o;
}

// The picture, and for a lit layer (`geolight`) what the pixel gives off
// (rgb, premultiplied by coverage) and how much it stands in light's way
// (a): packed in the pad lanes by `geo::flatten`.
struct Px {
    color: vec4f,
    light: vec4f,
}

fn emit_of(pr: Prim, cov: f32) -> vec4f {
    return vec4f(pr.pad3.x, pr.pad3.y, pr.pad2.x, pr.pad2.y) * cov;
}

@fragment
fn fs(in: VOut) -> @location(0) vec4f {
    return shade(in).color;
}

struct Lit {
    @location(0) color: vec4f,
    @location(1) light: vec4f,
}

@fragment
fn fs_lit(in: VOut) -> Lit {
    let s = shade(in);
    return Lit(s.color, s.light);
}

fn shade(in: VOut) -> Px {
    let pr = prims[in.prim];
    // A tile wholly inside a filled shape: flatten proved every pixel is
    // past the stroke and the AA ramp, so this is what the walk would say.
    if (pr.kind == 3u) { return Px(fill_at(pr, in.px, 1e9), emit_of(pr, 1.0)); }
    let p = in.px;
    var sd: f32;
    if (pr.kind == 2u) {
        sd = length(p - verts[pr.first]) - pr.radius;
    } else {
        var d = 1e20;
        var wn: i32 = 0;
        // Rings one after another, a sentinel (x < -1e29) between: each
        // closes on its own start. One ring is the old single loop.
        let end = pr.first + pr.count;
        var rs = pr.first;
        var i = pr.first;
        loop {
            if (i >= end) { break; }
            let ni = i + 1u;
            var closing = false;
            var b: vec2f;
            if (ni >= end) {
                closing = true;
            } else if (verts[ni].x < -1e29) {
                closing = true;
            }
            if (closing) { b = verts[rs]; } else { b = verts[ni]; }
            let a = verts[i];
            // An open outline has no closing segment.
            if (!(closing && pr.kind == 0u)) {
                let e = a - b;
                let w = p - b;
                let t = clamp(dot(w, e) / max(dot(e, e), 1e-12), 0.0, 1.0);
                let q = w - e * t;
                d = min(d, dot(q, q));
                let up = (b.y <= p.y) && (a.y > p.y);
                let dn = (a.y <= p.y) && (b.y > p.y);
                let side = e.x * w.y - e.y * w.x;
                if (up && (side > 0.0)) { wn += 1; }
                if (dn && (side < 0.0)) { wn -= 1; }
            }
            if (closing) {
                rs = ni + 1u;
                i = ni + 1u;
            } else {
                i = ni;
            }
        }
        var inside = false;
        if (pr.kind == 1u) {
            if (pr.rule == 1u) { inside = (wn & 1) != 0; } else { inside = wn != 0; }
        }
        sd = select(sqrt(d), -sqrt(d), inside);
    }
    var fc = 0.0;
    if (pr.kind != 0u) { fc = clamp(0.5 - sd, 0.0, 1.0); }
    var sc = 0.0;
    if (pr.width > 0.0) { sc = clamp(0.5 + pr.width * 0.5 - abs(sd), 0.0, 1.0); }
    let f = fill_at(pr, p, sd) * fc;
    let s = pr.stroke * sc;
    let filled = select(0.0, fc, pr.fill.a > 0.0 || pr.grad_kind != 0u);
    return Px(s + f * (1.0 - s.a), emit_of(pr, max(filled, sc)));
}
"#;

/// Shared pipelines, one per [`crate::geo::BlendMode`] (same shader, same
/// layout, a different blend state); per-layer state lives in [`GeoLayer`].
pub struct GeoRaster {
    pipelines: Vec<wgpu::RenderPipeline>,
    /// The same, drawing a second target too: a lit layer's light
    /// (`geolight`), always composited over.
    lit_pipelines: Vec<wgpu::RenderPipeline>,
    layout: wgpu::BindGroupLayout,
}

/// The fixed-function blend for a mode, premultiplied. Where a mode has
/// nothing to work on (multiply, darken and subtract onto an empty layer)
/// it leaves nothing: the alpha is the layer's, not the shape's.
fn blend_state(mode: crate::geo::BlendMode) -> wgpu::BlendState {
    use crate::geo::BlendMode as M;
    use wgpu::{BlendComponent as C, BlendFactor as F, BlendOperation as O};
    let c = |src_factor, dst_factor, operation| C { src_factor, dst_factor, operation };
    let over = C { src_factor: F::One, dst_factor: F::OneMinusSrcAlpha, operation: O::Add };
    let keep = c(F::Zero, F::One, O::Add);
    let (color, alpha) = match mode {
        M::Over => (over, over),
        M::Add => (c(F::One, F::One, O::Add), over),
        M::Multiply => (c(F::Dst, F::OneMinusSrcAlpha, O::Add), keep),
        M::Screen => (c(F::One, F::OneMinusSrc, O::Add), over),
        M::Lighten => (c(F::One, F::One, O::Max), c(F::One, F::One, O::Max)),
        M::Darken => (c(F::One, F::One, O::Min), c(F::One, F::One, O::Min)),
        M::Subtract => (c(F::One, F::One, O::ReverseSubtract), keep),
    };
    wgpu::BlendState { color, alpha }
}

pub struct GeoLayer {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub width: u32,
    pub height: u32,
    /// Hash of the draw list on the GPU; `u64::MAX` until the first upload.
    pub hash: u64,
    pub dirty: bool,
    /// What the geometry was last evaluated FROM, so an unchanged node is
    /// not re-evaluated every frame (the CPU half of the memo).
    pub last_key: Option<(GeoParams, Option<FieldCx>)>,
    /// What the last evaluation reported, for the panel.
    pub stats: crate::geo::GeoStats,
    /// How long the last evaluation + flatten took, for the panel.
    pub eval_ms: f32,
    /// `DrawList::edge_tests` of what is on the GPU: the re-record cost.
    pub raster_cost: u64,
    /// The geometry the layer was drawn from — what the viewer's vector
    /// HUD overlays, so the picture and the data can never disagree.
    pub geometry: crate::geo::Geometry,
    uniform: wgpu::Buffer,
    prims: Option<wgpu::Buffer>,
    verts: Option<wgpu::Buffer>,
    bind: Option<wgpu::BindGroup>,
    count: u32,
    /// Consecutive shapes of one blend mode: (mode index, instances).
    runs: Vec<(u32, std::ops::Range<u32>)>,
}

impl GeoRaster {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("geo"),
            source: wgpu::ShaderSource::Wgsl(GEO_WGSL.into()),
        });
        let vf = wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT;
        let storage = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: vf,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("geo"),
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
                storage(1),
                storage(2),
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("geo"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let make = |m: &crate::geo::BlendMode, lit: bool| device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("geo"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some(if lit { "fs_lit" } else { "fs" }),
                compilation_options: Default::default(),
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: Some(blend_state(*m)),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    lit.then(|| wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: Some(blend_state(crate::geo::BlendMode::Over)),
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ][..if lit { 2 } else { 1 }],
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
        let pipelines = crate::geo::BlendMode::ALL.iter().map(|m| make(m, false)).collect();
        let lit_pipelines = crate::geo::BlendMode::ALL.iter().map(|m| make(m, true)).collect();
        Self { pipelines, lit_pipelines, layout }
    }

    pub fn make_layer(&self, device: &wgpu::Device, w: u32, h: u32) -> GeoLayer {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("geo-layer"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("geo-u"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        GeoLayer {
            texture,
            view,
            width: w,
            height: h,
            hash: u64::MAX, // never matches: the first upload always records
            dirty: true,
            last_key: None,
            stats: Default::default(),
            eval_ms: 0.0,
            raster_cost: 0,
            geometry: Default::default(),
            uniform,
            prims: None,
            verts: None,
            bind: None,
            count: 0,
            runs: Vec::new(),
        }
    }

    /// Upload a draw list into the layer and mark it for re-record. Buffers
    /// grow to the next power of two and are never shrunk; the bind group
    /// is rebuilt only when a buffer is reallocated.
    pub fn set_draw_list(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layer: &mut GeoLayer,
        dl: &DrawList,
        hash: u64,
    ) {
        queue.write_buffer(
            &layer.uniform,
            0,
            bytemuck::cast_slice(&[layer.width as f32, layer.height as f32, 0.0, 0.0]),
        );
        let pbytes: &[u8] = bytemuck::cast_slice(&dl.prims);
        let vbytes: &[u8] = bytemuck::cast_slice(&dl.verts);
        let pneed = pbytes.len().max(std::mem::size_of::<PrimGpu>()) as u64;
        let vneed = vbytes.len().max(8) as u64;
        let mut realloc = false;
        if layer.prims.as_ref().is_none_or(|b| b.size() < pneed) {
            layer.prims = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("geo-prims"),
                size: pneed.next_power_of_two(),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            realloc = true;
        }
        if layer.verts.as_ref().is_none_or(|b| b.size() < vneed) {
            layer.verts = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("geo-verts"),
                size: vneed.next_power_of_two(),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            realloc = true;
        }
        if realloc || layer.bind.is_none() {
            layer.bind = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("geo-bg"),
                layout: &self.layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: layer.uniform.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: layer.prims.as_ref().unwrap().as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: layer.verts.as_ref().unwrap().as_entire_binding(),
                    },
                ],
            }));
        }
        if !pbytes.is_empty() {
            queue.write_buffer(layer.prims.as_ref().unwrap(), 0, pbytes);
        }
        if !vbytes.is_empty() {
            queue.write_buffer(layer.verts.as_ref().unwrap(), 0, vbytes);
        }
        layer.count = dl.prims.len() as u32;
        layer.runs.clear();
        for (i, p) in dl.prims.iter().enumerate() {
            let i = i as u32;
            match layer.runs.last_mut() {
                Some((m, r)) if *m == p.blend => r.end = i + 1,
                _ => layer.runs.push((p.blend, i..i + 1)),
            }
        }
        layer.hash = hash;
        layer.dirty = true;
    }

    /// Record the layer: clear to transparent, draw every shape in order.
    /// `light`: a lit layer's light target (`geolight`), drawn in the same
    /// pass.
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder, layer: &mut GeoLayer, light: Option<&wgpu::TextureView>) {
        let attach = |view| {
            Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })
        };
        let attachments = [attach(&layer.view), light.and_then(attach)];
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("geo-layer"),
            color_attachments: &attachments[..if light.is_some() { 2 } else { 1 }],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        if let (Some(bind), true) = (&layer.bind, layer.count > 0) {
            pass.set_bind_group(0, bind, &[]);
            let pipes = if light.is_some() { &self.lit_pipelines } else { &self.pipelines };
            for (m, r) in &layer.runs {
                let pipe = pipes.get(*m as usize).unwrap_or(&pipes[0]);
                pass.set_pipeline(pipe);
                pass.draw(0..4, r.clone());
            }
        }
        layer.dirty = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rasteriser's WGSL parses and validates, so a typo here fails a
    /// unit test rather than the first render.
    #[test]
    fn geo_wgsl_validates() {
        let module = naga::front::wgsl::parse_str(GEO_WGSL).expect("GEO_WGSL parses");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("GEO_WGSL validates");
    }

    #[test]
    fn prim_gpu_is_the_size_the_shader_expects() {
        assert_eq!(std::mem::size_of::<PrimGpu>(), 128);
    }
}
