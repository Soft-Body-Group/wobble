//! Paint-stroke rasterizer: strokes (vector params on the Paint node) are
//! resampled into stamps and splatted as instanced soft ellipses into
//! persistent per-node layer textures. A layer is a MEMOIZED rasterization —
//! re-recorded only when its stamp set (or output size) changes — so painting
//! never recompiles shaders and playback pays nothing while strokes are
//! static. The fused pass samples the layers like any other source.
//!
//! One node can own up to three layers, one per [`BrushClass`]:
//!
//! * **colour** — premultiplied rgba, composited over the input. Blended
//!   `over`, so a later stroke covers an earlier one.
//! * **source** — clone and smear: `rg` = where to fetch the input FROM
//!   (an offset in layer px), `a` = coverage. Also `over`, so overlapping
//!   strokes land as the alpha-weighted AVERAGE of their offsets, which the
//!   fused pass divides back out; offsets must never add.
//! * **effect** — blur, dodge and burn: `r` = exposure in stops, `g` = blur
//!   radius in px, `b` = blur coverage, `a` = coverage. Blended ADDITIVELY,
//!   because these are amounts and a brush that builds up under repeated
//!   strokes is the whole feel of a dodge brush. Each is premultiplied by
//!   its stamp's alpha, so `r` IS the exposure to apply and `g / b` the
//!   mean blur radius under the pixel.
//!
//! A layer exists only while the node has a stroke of its class, and the
//! same flag is what the fused shader was compiled against — so a node that
//! has only ever painted colour costs exactly what it always did.

use crate::graph::{BrushClass, BrushMode, PaintBlend, PaintParams};

/// How many extra reads of the input a blur stroke costs the fused pass,
/// per pixel: one ring of taps around the centre pixel (which is already
/// there). Fixed, so the cost of a blur brush is bounded by design rather
/// than by its radius — a big radius spreads the same taps wider and gets
/// sparser, it does not get slower.
pub const PAINT_BLUR_TAPS: usize = 8;

const PAINT_WGSL: &str = r#"
struct U { size: vec4f } // xy = layer size in px
@group(0) @binding(0) var<uniform> u: U;

struct VOut {
    @builtin(position) pos: vec4f,
    @location(0) local: vec2f,
    @location(1) color: vec4f,
    @location(2) soft: f32,
    @location(3) @interpolate(flat) mode: u32,
}

@vertex
fn vs(
    @builtin(vertex_index) vi: u32,
    @location(0) center: vec2f,   // px
    @location(1) rad_soft: vec2f, // major radius px, softness 0..1
    @location(2) shape: vec2f,    // aspect (minor/major), angle radians
    @location(3) color: vec4f,    // payload, straight
    @location(4) extra: vec2f,    // x = blend id
) -> VOut {
    // Triangle-strip quad, padded a touch for the soft edge. The quad is
    // built in the brush's OWN frame — the unit disc, `local` — and the
    // ellipse is entirely in how that frame lands on the layer: squash y
    // by the aspect, turn by the angle. The fragment never knows.
    let corner = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u)) - 1.0; // -1/+1
    let local = corner * 1.1;
    let scaled = vec2f(local.x, local.y * shape.x) * rad_soft.x;
    let c = cos(shape.y);
    let s = sin(shape.y);
    let px = center + vec2f(c * scaled.x - s * scaled.y, s * scaled.x + c * scaled.y);
    var o: VOut;
    o.pos = vec4f(px / u.size.xy * 2.0 - 1.0, 0.0, 1.0);
    o.pos.y = -o.pos.y;
    o.local = local;
    o.color = color;
    o.soft = rad_soft.y;
    o.mode = u32(extra.x + 0.5);
    return o;
}

// The stroke's blend (`graph::PaintBlend::id`) picks what the fragment
// hands the blend unit; the blend STATES (`PaintRaster::new`) do the rest.
// `B` is the additive term of the layer's affine map, `M` the multiplier:
// the fused pass computes `below * M + B`. Coverage `c`, colour `k`.
fn coverage(in: VOut) -> f32 {
    let d = length(in.local);
    let edge0 = clamp(1.0 - in.soft, 0.0, 0.999);
    return (1.0 - smoothstep(edge0, 1.0, d)) * in.color.a;
}

fn out_b(mode: u32, c: f32, k: vec3f) -> vec4f {
    switch mode {
        case 1u: { return vec4f(0.0, 0.0, 0.0, c); }            // erase
        case 2u, 3u: { return vec4f(k * c, 0.0); }              // add, subtract
        case 4u: { return vec4f(1.0 - c + c * k, 1.0); }        // multiply
        case 5u: { return vec4f(k * c, 0.0); }                  // screen
        default: { return vec4f(k * c, c); }                    // normal: premultiplied
    }
}

fn out_m(mode: u32, c: f32, k: vec3f) -> vec4f {
    switch mode {
        case 4u: { return vec4f(1.0 - c + c * k, 1.0); }
        case 5u: { return vec4f(k * c, 0.0); }
        default: { return vec4f(c); }
    }
}

@fragment
fn fs(in: VOut) -> @location(0) vec4f {
    return out_b(in.mode, coverage(in), in.color.rgb);
}

struct Two {
    @location(0) b: vec4f,
    @location(1) m: vec4f,
}

@fragment
fn fs2(in: VOut) -> Two {
    let c = coverage(in);
    var o: Two;
    o.b = out_b(in.mode, c, in.color.rgb);
    o.m = out_m(in.mode, c, in.color.rgb);
    return o;
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Stamp {
    pub center: [f32; 2],
    /// Major radius px, softness 0..1.
    pub rad_soft: [f32; 2],
    /// Aspect (minor/major), angle in RADIANS.
    pub shape: [f32; 2],
    /// The class's payload — see the module doc.
    pub color: [f32; 4],
    /// `[blend id, 0]` — `graph::PaintBlend::id`, 0 (normal) outside the
    /// colour set.
    pub extra: [f32; 2],
}

/// The stamp sets a Paint node rasterizes, one per layer. Empty sets mean
/// no layer.
#[derive(Default)]
pub struct PaintStamps {
    pub color: Vec<Stamp>,
    /// `color` = `(offset.x px, offset.y px, 0, alpha)`.
    pub source: Vec<Stamp>,
    /// `color` = `(stops, blur px, is_blur, alpha)`.
    pub effect: Vec<Stamp>,
}

/// Resample strokes into stamps for a layer of `w`x`h` px, at `frame` (a
/// stroke's lifetime decides whether it is there at all). `scale` is layer
/// px per AUTHORED px — the proxy factor: a stroke's radius, blur and smear
/// are authored in full-res pixels, so at half res they halve with the
/// canvas and the stroke looks the same in proxy. Deterministic: the
/// stamp bytes are what the layer cache is keyed on, so a scrub across
/// frames with no per-frame strokes never re-records a thing.
pub fn stamps_for(params: &PaintParams, w: u32, h: u32, frame: u64, scale: f32) -> PaintStamps {
    let (wf, hf) = (w as f32, h as f32);
    let scale = if scale > 0.0 { scale } else { 1.0 };
    let mut out = PaintStamps::default();
    for s in params.live_at(frame) {
        let radius = s.radius.clamp(0.5, 4096.0) * scale;
        let shape = [s.aspect.clamp(0.02, 1.0), s.angle.to_radians()];
        let mode = s.mode();
        let alpha = s.color[3];
        // Everything but smear has one payload for the whole stroke. A
        // smear's is per stamp: the offset points BACK along the stroke,
        // so the pixels under the brush are the ones it just passed over,
        // dragged forward — which is what a smudge is.
        let payload = match mode {
            BrushMode::Color => s.color,
            BrushMode::Clone => {
                let o = s.clone_off.unwrap_or([0.0, 0.0]);
                [o[0] * wf, o[1] * hf, 0.0, alpha]
            }
            BrushMode::Smear => [0.0, 0.0, 0.0, alpha],
            BrushMode::Blur => [0.0, s.strength.max(0.0) * scale, 1.0, alpha],
            BrushMode::Dodge => [s.strength.max(0.0), 0.0, 0.0, alpha],
            BrushMode::Burn => [-s.strength.max(0.0), 0.0, 0.0, alpha],
        };
        let class = mode.class();
        let blend = if class == BrushClass::Color { s.blend.id() as f32 } else { 0.0 };
        let smear_px = if mode == BrushMode::Smear { s.strength.max(0.0) * scale } else { 0.0 };
        // Pressure scales the radius and the opacity per stamp, through the
        // map the stroke was painted with. A stroke with no samples is at
        // full pressure, i.e. factors of exactly 1 — the stamps of every
        // stroke from before pressure are byte-for-byte what they were.
        let press = |i: usize| s.press.factors(s.pressure_at(i));
        let emit = |p: [f32; 2], dir: [f32; 2], (fr, fa): (f32, f32), out: &mut PaintStamps| {
            let mut color = payload;
            color[3] *= fa;
            if mode == BrushMode::Smear {
                color[0] = -dir[0] * smear_px;
                color[1] = -dir[1] * smear_px;
            }
            let stamp = Stamp {
                center: p,
                rad_soft: [(radius * fr).max(0.5), s.softness.clamp(0.0, 1.0)],
                shape,
                color,
                extra: [blend, 0.0],
            };
            match class {
                BrushClass::Color => out.color.push(stamp),
                BrushClass::Source => out.source.push(stamp),
                BrushClass::Effect => out.effect.push(stamp),
            }
        };
        let mut last: Option<([f32; 2], (f32, f32))> = None;
        // The stroke's direction at its last stamp: a lone point has none,
        // so a single-click smear moves nothing.
        let mut dir = [0.0f32, 0.0];
        for (i, pt) in s.points.iter().enumerate() {
            let p = [pt[0] * wf, pt[1] * hf];
            let f = press(i);
            match last {
                None => emit(p, dir, f, &mut out),
                Some((l, lf)) => {
                    let d = ((p[0] - l[0]).powi(2) + (p[1] - l[1]).powi(2)).sqrt();
                    if d > 1e-6 {
                        dir = [(p[0] - l[0]) / d, (p[1] - l[1]) / d];
                    }
                    // Spacing follows the brush's SHORT axis (a thin brush
                    // at a steep angle would otherwise leave gaps), at the
                    // segment's SMALLER pressure radius, so a taper stays
                    // closed where it is thinnest.
                    let r = radius * lf.0.min(f.0);
                    let spacing = (r * shape[0] * 0.35).max(0.75);
                    let steps = (d / spacing).floor() as usize;
                    for k in 1..=steps {
                        let t = k as f32 / (steps.max(1)) as f32 * (steps as f32 * spacing / d);
                        let t = t.min(1.0);
                        let ft = (lf.0 + (f.0 - lf.0) * t, lf.1 + (f.1 - lf.1) * t);
                        emit([l[0] + (p[0] - l[0]) * t, l[1] + (p[1] - l[1]) * t], dir, ft, &mut out);
                    }
                }
            }
            last = Some((p, f));
        }
    }
    out
}

/// Shared pipelines + per-layer GPU state lives in the compositor; this
/// module only knows how to build the pipelines and record one layer.
pub struct PaintRaster {
    /// One-texture colour layers and the source layer, per blend that can
    /// live in one texture: `[normal (premultiplied over), erase]`.
    single: [wgpu::RenderPipeline; 2],
    /// Additive: the effect layer, where amounts build up.
    add: wgpu::RenderPipeline,
    /// Two-texture colour layers (`B` + `M`, see `PaintBlend`), one per
    /// blend in `PaintBlend::ALL` order.
    dual: Vec<wgpu::RenderPipeline>,
    /// The one bind group layout EVERY pipeline is built against, so a
    /// layer's bind group binds under any. It has to be explicit: with
    /// `layout: None` each pipeline derives its own EXCLUSIVE layout, and a
    /// group made from `over`'s is a validation error under `add` (the
    /// effect strokes — blur, dodge, burn — all died on it).
    bgl: wgpu::BindGroupLayout,
}

/// How a layer blends its stamps.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerBlend {
    /// Colour and source: each stamp by its own blend id, in order.
    Over,
    Add,
}

pub struct PaintLayer {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    /// The colour layer's MULTIPLIER (`M`), only while the node has a blend
    /// that needs one (`PaintParams::has_blend`). Cleared to 1.
    pub mul: Option<(wgpu::Texture, wgpu::TextureView)>,
    pub width: u32,
    pub height: u32,
    pub hash: u64,
    pub dirty: bool,
    uniform: wgpu::Buffer,
    bind: wgpu::BindGroup,
    vbuf: Option<wgpu::Buffer>,
    count: u32,
    /// Consecutive stamps with one blend id: `(first, count, blend id)`.
    runs: Vec<(u32, u32, u32)>,
}

fn comp(src: wgpu::BlendFactor, dst: wgpu::BlendFactor) -> wgpu::BlendComponent {
    wgpu::BlendComponent { src_factor: src, dst_factor: dst, operation: wgpu::BlendOperation::Add }
}

/// The blend states that make the fragment's outputs (`out_b`, `out_m`)
/// compose the layer's affine map, per `PaintBlend`: `(B state, M state)`.
fn blend_states(b: PaintBlend) -> (wgpu::BlendState, wgpu::BlendState) {
    use wgpu::BlendFactor::{One, OneMinusSrc, OneMinusSrcAlpha, Src, Zero};
    // Leave the destination alone (alpha of the colour-only modes, `M` of
    // add / subtract).
    let keep = comp(Zero, One);
    match b {
        PaintBlend::Normal => (
            wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING,
            // M *= 1 - c
            wgpu::BlendState { color: comp(Zero, OneMinusSrc), alpha: comp(Zero, OneMinusSrc) },
        ),
        PaintBlend::Erase => (
            // B *= 1 - c
            wgpu::BlendState {
                color: comp(Zero, OneMinusSrcAlpha),
                alpha: comp(Zero, OneMinusSrcAlpha),
            },
            // M = M (1 - c) + c: back towards 1, the input
            wgpu::BlendState { color: comp(One, OneMinusSrc), alpha: comp(One, OneMinusSrc) },
        ),
        PaintBlend::Add => (
            wgpu::BlendState { color: comp(One, One), alpha: keep },
            wgpu::BlendState { color: keep, alpha: keep },
        ),
        PaintBlend::Subtract => (
            wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: One,
                    dst_factor: One,
                    operation: wgpu::BlendOperation::ReverseSubtract,
                },
                alpha: keep,
            },
            wgpu::BlendState { color: keep, alpha: keep },
        ),
        // B *= m, M *= m — everything below darkens, earlier paint too.
        PaintBlend::Multiply => (
            wgpu::BlendState { color: comp(Zero, Src), alpha: keep },
            wgpu::BlendState { color: comp(Zero, Src), alpha: keep },
        ),
        // x -> x (1 - ck) + ck
        PaintBlend::Screen => (
            wgpu::BlendState { color: comp(One, OneMinusSrc), alpha: keep },
            wgpu::BlendState { color: comp(Zero, OneMinusSrc), alpha: keep },
        ),
    }
}

impl PaintRaster {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("paint"),
            source: wgpu::ShaderSource::Wgsl(PAINT_WGSL.into()),
        });
        // Every pipeline is the one module with the one vertex layout; only
        // the blend (and, for the two-texture family, the entry point)
        // differs, so they share a bind group layout — an EXPLICIT one
        // (see `bgl`) — and a layer records with any.
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("paint-bgl"),
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
            label: Some("paint-pl"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let target = |blend: wgpu::BlendState| {
            Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba16Float,
                blend: Some(blend),
                write_mask: wgpu::ColorWrites::ALL,
            })
        };
        let make = |label: &str, entry: &str, targets: &[Option<wgpu::ColorTargetState>]| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pl),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[Some(wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<Stamp>() as u64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array![
                            0 => Float32x2, 1 => Float32x2, 2 => Float32x2, 3 => Float32x4,
                            4 => Float32x2
                        ],
                    })],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(entry),
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
        let add = wgpu::BlendState {
            color: comp(wgpu::BlendFactor::One, wgpu::BlendFactor::One),
            alpha: comp(wgpu::BlendFactor::One, wgpu::BlendFactor::One),
        };
        let one = |b: PaintBlend| make("paint-1", "fs", &[target(blend_states(b).0)]);
        Self {
            single: [one(PaintBlend::Normal), one(PaintBlend::Erase)],
            add: make("paint-add", "fs", &[target(add)]),
            dual: PaintBlend::ALL
                .iter()
                .map(|&b| {
                    let (sb, sm) = blend_states(b);
                    make("paint-2", "fs2", &[target(sb), target(sm)])
                })
                .collect(),
            bgl,
        }
    }

    fn texture(device: &wgpu::Device, w: u32, h: u32, label: &str) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
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
        (texture, view)
    }

    /// A layer of `w`x`h`; `mul` adds the multiplier texture a colour
    /// layer with colour blends needs.
    pub fn make_layer(&self, device: &wgpu::Device, w: u32, h: u32, mul: bool) -> PaintLayer {
        let (texture, view) = Self::texture(device, w, h, "paint-layer");
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("paint-u"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("paint-bg"),
            layout: &self.bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform.as_entire_binding(),
            }],
        });
        PaintLayer {
            texture,
            view,
            mul: mul.then(|| Self::texture(device, w, h, "paint-mul")),
            width: w,
            height: h,
            hash: u64::MAX, // never matches: first update always records
            dirty: true,
            uniform,
            bind,
            vbuf: None,
            count: 0,
            runs: Vec::new(),
        }
    }

    /// Upload a new stamp set into the layer and mark it for re-record.
    pub fn set_stamps(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layer: &mut PaintLayer,
        stamps: &[Stamp],
        hash: u64,
    ) {
        queue.write_buffer(
            &layer.uniform,
            0,
            bytemuck::cast_slice(&[layer.width as f32, layer.height as f32, 0.0, 0.0]),
        );
        let bytes: &[u8] = bytemuck::cast_slice(stamps);
        let need = bytes.len().max(std::mem::size_of::<Stamp>()) as u64;
        let recreate = layer.vbuf.as_ref().map(|b| b.size() < need).unwrap_or(true);
        if recreate {
            layer.vbuf = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("paint-stamps"),
                size: need.next_power_of_two(),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        if !bytes.is_empty() {
            queue.write_buffer(layer.vbuf.as_ref().unwrap(), 0, bytes);
        }
        layer.count = stamps.len() as u32;
        layer.runs = blend_runs(stamps);
        layer.hash = hash;
        layer.dirty = true;
    }

    /// Record the layer: clear (B to 0, M to 1), splat every stamp in
    /// order — one draw per run of one blend, since the blend is pipeline
    /// state.
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        layer: &mut PaintLayer,
        blend: LayerBlend,
    ) {
        fn attach(view: &wgpu::TextureView, clear: wgpu::Color) -> Option<wgpu::RenderPassColorAttachment<'_>> {
            Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Clear(clear), store: wgpu::StoreOp::Store },
            })
        }
        let dual = layer.mul.is_some() && blend == LayerBlend::Over;
        let mut atts = vec![attach(&layer.view, wgpu::Color::TRANSPARENT)];
        if let (true, Some((_, mv))) = (dual, &layer.mul) {
            atts.push(attach(mv, wgpu::Color::WHITE));
        }
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("paint-layer"),
            color_attachments: &atts,
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        if layer.count > 0 {
            pass.set_bind_group(0, &layer.bind, &[]);
            pass.set_vertex_buffer(0, layer.vbuf.as_ref().unwrap().slice(..));
            for &(first, n, id) in &layer.runs {
                let pipe = match blend {
                    LayerBlend::Add => &self.add,
                    LayerBlend::Over if dual => &self.dual[(id as usize).min(self.dual.len() - 1)],
                    // One texture holds normal and erase; a colour blend
                    // reaching here (it cannot: `has_blend` allocates the
                    // multiplier) lands as normal rather than corrupting.
                    LayerBlend::Over => &self.single[usize::from(id == PaintBlend::Erase.id())],
                };
                pass.set_pipeline(pipe);
                pass.draw(0..4, first..first + n);
            }
        }
        drop(pass);
        layer.dirty = false;
    }
}

/// Split a stamp set into runs of one blend id, in order.
fn blend_runs(stamps: &[Stamp]) -> Vec<(u32, u32, u32)> {
    let mut runs: Vec<(u32, u32, u32)> = Vec::new();
    for (i, st) in stamps.iter().enumerate() {
        let id = st.extra[0].round().max(0.0) as u32;
        match runs.last_mut() {
            Some((_, n, last)) if *last == id => *n += 1,
            _ => runs.push((i as u32, 1, id)),
        }
    }
    runs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::PaintStroke;

    use crate::graph::StrokeLife;

    fn one(points: Vec<[f32; 2]>, radius: f32, color: [f32; 4]) -> PaintStroke {
        PaintStroke::new(points, radius, 0.5, color)
    }

    #[test]
    fn stamps_cover_stroke_at_spacing() {
        let params = PaintParams {
            strokes: vec![one(vec![[0.0, 0.5], [1.0, 0.5]], 20.0, [1.0, 0.0, 0.0, 1.0])],
            ..Default::default()
        };
        let stamps = stamps_for(&params, 1000, 1000, 0, 1.0).color;
        // 1000px stroke at spacing 7px -> ~143 stamps, endpoints included.
        assert!(stamps.len() > 100 && stamps.len() < 200, "{}", stamps.len());
        assert_eq!(stamps[0].center, [0.0, 500.0]);
        let last = stamps.last().unwrap().center;
        assert!((last[0] - 1000.0).abs() < 8.0, "end near stroke end: {last:?}");
        // Deterministic: same input, same bytes (the layer cache key).
        let again = stamps_for(&params, 1000, 1000, 0, 1.0).color;
        assert_eq!(bytemuck::cast_slice::<_, u8>(&stamps), bytemuck::cast_slice::<_, u8>(&again));
    }

    #[test]
    fn single_point_stroke_is_one_stamp() {
        let params = PaintParams {
            strokes: vec![PaintStroke::new(
                vec![[0.5, 0.5]],
                10.0,
                0.0,
                [0.0, 1.0, 0.0, 0.5],
            )],
            ..Default::default()
        };
        let stamps = stamps_for(&params, 200, 100, 0, 1.0).color;
        assert_eq!(stamps.len(), 1);
        assert_eq!(stamps[0].center, [100.0, 50.0]);
    }

    /// A stroke's lifetime decides whether it is rasterized at all, and the
    /// stamp bytes are the layer's cache key — so a frame with no per-frame
    /// strokes on it produces exactly the bytes it did before they existed.
    #[test]
    fn a_strokes_lifetime_decides_which_frames_it_lands_on() {
        let mut always = one(vec![[0.2, 0.2]], 10.0, [1.0; 4]);
        let mut single = one(vec![[0.4, 0.4]], 10.0, [1.0; 4]);
        single.life = StrokeLife::Single(7);
        let mut ranged = one(vec![[0.6, 0.6]], 10.0, [1.0; 4]);
        ranged.life = StrokeLife::Range(5, 9);
        always.life = StrokeLife::All;
        let p = PaintParams {
            strokes: vec![always, single, ranged],
            ..Default::default()
        };
        let at = |f: u64| stamps_for(&p, 100, 100, f, 1.0).color.len();
        assert_eq!(at(0), 1, "only the all-frames stroke");
        assert_eq!(at(5), 2, "the range opens");
        assert_eq!(at(7), 3, "everything, on the single stroke's frame");
        assert_eq!(at(9), 2, "the range's last frame");
        assert_eq!(at(10), 1, "and it closes");

        // Nothing per-frame in the node = identical bytes on every frame, so
        // scrubbing never re-records the layer.
        let plain = PaintParams {
            strokes: vec![one(vec![[0.1, 0.1], [0.9, 0.9]], 8.0, [1.0; 4])],
            ..Default::default()
        };
        let a = stamps_for(&plain, 100, 100, 0, 1.0).color;
        let b = stamps_for(&plain, 100, 100, 999, 1.0).color;
        assert_eq!(
            bytemuck::cast_slice::<_, u8>(&a),
            bytemuck::cast_slice::<_, u8>(&b)
        );
    }

    /// Clone strokes go to the source set, carrying the offset in PIXELS
    /// where a colour stroke carries its colour.
    #[test]
    fn clone_strokes_carry_a_pixel_offset_in_their_own_set() {
        let mut c = one(vec![[0.5, 0.5]], 10.0, [0.0, 0.0, 0.0, 0.8]);
        c.mode = BrushMode::Clone;
        c.clone_off = Some([0.25, -0.5]);
        let p = PaintParams {
            strokes: vec![one(vec![[0.1, 0.1]], 10.0, [1.0, 0.0, 0.0, 1.0]), c],
            ..Default::default()
        };
        assert!(p.has_source());
        let st = stamps_for(&p, 200, 100, 0, 1.0);
        assert_eq!(st.color.len(), 1, "the colour stroke stays a colour stroke");
        assert_eq!(st.source.len(), 1);
        assert!(st.effect.is_empty());
        // Offset in the same space as `center`, so the shader just adds it.
        assert_eq!(st.source[0].color, [50.0, -50.0, 0.0, 0.8]);
        assert_eq!(st.source[0].center, [100.0, 50.0]);

        // No source strokes: no source set at all, and nothing to pay for.
        let plain = PaintParams {
            strokes: vec![one(vec![[0.1, 0.1]], 10.0, [1.0; 4])],
            ..Default::default()
        };
        assert!(!plain.has_source());
        assert!(stamps_for(&plain, 200, 100, 0, 1.0).source.is_empty());
    }

    /// A smear is a clone whose offset points BACK along the stroke, per
    /// stamp — the pixels it just passed over, dragged forward.
    #[test]
    fn smear_offsets_point_back_along_the_stroke() {
        let mut sm = one(vec![[0.1, 0.5], [0.9, 0.5]], 10.0, [0.0, 0.0, 0.0, 1.0]);
        sm.mode = BrushMode::Smear;
        sm.strength = 30.0;
        let p = PaintParams { strokes: vec![sm], ..Default::default() };
        assert!(p.has_source(), "smear is a source read, like clone");
        let st = stamps_for(&p, 1000, 1000, 0, 1.0);
        assert!(st.source.len() > 10 && st.color.is_empty() && st.effect.is_empty());
        // The stroke runs +x, so every stamp after the first pulls from -x
        // by the drag distance; the first has no direction yet and pulls
        // from nowhere.
        assert_eq!(st.source[0].color, [0.0, 0.0, 0.0, 1.0]);
        for s in &st.source[1..] {
            assert!((s.color[0] + 30.0).abs() < 1e-4 && s.color[1].abs() < 1e-4, "{:?}", s.color);
        }
        // A single-click smear is inert, not a stamp with a stale direction.
        let mut dot = one(vec![[0.5, 0.5]], 10.0, [0.0, 0.0, 0.0, 1.0]);
        dot.mode = BrushMode::Smear;
        dot.strength = 30.0;
        let st = stamps_for(&PaintParams { strokes: vec![dot], ..Default::default() }, 100, 100, 0, 1.0);
        assert_eq!(st.source[0].color[0], 0.0);
    }

    /// Blur, dodge and burn share the effect set, each in its own channel,
    /// so one layer carries all three and the fused pass reads each once.
    #[test]
    fn effect_strokes_pack_stops_and_blur_into_one_set() {
        let mk = |mode: BrushMode, strength: f32| {
            let mut s = one(vec![[0.5, 0.5]], 10.0, [0.0, 0.0, 0.0, 0.5]);
            s.mode = mode;
            s.strength = strength;
            s
        };
        let p = PaintParams {
            strokes: vec![
                mk(BrushMode::Dodge, 0.75),
                mk(BrushMode::Burn, 1.5),
                mk(BrushMode::Blur, 12.0),
            ],
            ..Default::default()
        };
        assert!(p.has_effect() && p.has_blur() && p.has_tone() && !p.has_source());
        let st = stamps_for(&p, 100, 100, 0, 1.0);
        assert_eq!(st.effect.len(), 3);
        assert!(st.color.is_empty() && st.source.is_empty());
        assert_eq!(st.effect[0].color, [0.75, 0.0, 0.0, 0.5], "dodge: +stops");
        assert_eq!(st.effect[1].color, [-1.5, 0.0, 0.0, 0.5], "burn: -stops");
        assert_eq!(st.effect[2].color, [0.0, 12.0, 1.0, 0.5], "blur: radius, flagged");
        // A negative strength cannot flip a dodge into a burn from the
        // panel's drag field.
        let st = stamps_for(
            &PaintParams { strokes: vec![mk(BrushMode::Dodge, -2.0)], ..Default::default() },
            100,
            100,
            0,
            1.0,
        );
        assert_eq!(st.effect[0].color[0], 0.0);
    }

    /// The brush shape rides on every stamp, angle in radians, and a thin
    /// brush stamps more densely so its short axis still overlaps.
    #[test]
    fn brush_shape_reaches_the_stamp_and_tightens_the_spacing() {
        let mut round = one(vec![[0.0, 0.5], [1.0, 0.5]], 20.0, [1.0; 4]);
        round.aspect = 1.0;
        let mut thin = round.clone();
        thin.aspect = 0.25;
        thin.angle = 90.0;
        let r = stamps_for(&PaintParams { strokes: vec![round], ..Default::default() }, 1000, 1000, 0, 1.0);
        let t = stamps_for(&PaintParams { strokes: vec![thin], ..Default::default() }, 1000, 1000, 0, 1.0);
        assert_eq!(r.color[0].shape, [1.0, 0.0]);
        assert!((t.color[0].shape[0] - 0.25).abs() < 1e-6);
        assert!((t.color[0].shape[1] - std::f32::consts::FRAC_PI_2).abs() < 1e-6);
        assert!(
            t.color.len() > r.color.len() * 3,
            "thin brush {} vs round {}: spacing must follow the short axis",
            t.color.len(),
            r.color.len()
        );
    }

    /// Radius, blur and smear are authored in full-res px: at a proxy
    /// scale they shrink with the canvas, so a stroke looks the same.
    #[test]
    fn proxy_scale_shrinks_radius_with_the_canvas() {
        let p = PaintParams { strokes: vec![one(vec![[0.5, 0.5]], 20.0, [1.0; 4])], ..Default::default() };
        let full = stamps_for(&p, 1000, 1000, 0, 1.0).color;
        let half = stamps_for(&p, 500, 500, 0, 0.5).color;
        assert_eq!(full[0].rad_soft[0], 20.0);
        assert_eq!(half[0].rad_soft[0], 10.0);
        assert_eq!(half[0].center, [250.0, 250.0]);
    }

    /// Pressure scales each stamp's radius and opacity through the
    /// stroke's map, interpolated along a segment; a stroke with no samples
    /// is byte-identical to one painted before pressure existed.
    #[test]
    fn pressure_tapers_radius_and_opacity() {
        use crate::graph::PressureMap;
        let mut s = one(vec![[0.0, 0.5], [1.0, 0.5]], 20.0, [1.0, 1.0, 1.0, 1.0]);
        let plain = stamps_for(&PaintParams { strokes: vec![s.clone()], ..Default::default() }, 1000, 1000, 0, 1.0).color;
        s.press = PressureMap { size: 1.0, opacity: 0.5, gamma: 1.0 };
        let mapped_no_samples =
            stamps_for(&PaintParams { strokes: vec![s.clone()], ..Default::default() }, 1000, 1000, 0, 1.0).color;
        assert_eq!(
            bytemuck::cast_slice::<_, u8>(&plain),
            bytemuck::cast_slice::<_, u8>(&mapped_no_samples),
            "no samples = full pressure, whatever the map"
        );
        s.pressure = vec![0.25, 1.0];
        let st = stamps_for(&PaintParams { strokes: vec![s], ..Default::default() }, 1000, 1000, 0, 1.0).color;
        assert!((st[0].rad_soft[0] - 5.0).abs() < 1e-4, "{:?}", st[0]);
        assert!((st[0].color[3] - 0.625).abs() < 1e-4, "{:?}", st[0]);
        let last = st.last().unwrap();
        assert!(last.rad_soft[0] > 19.0 && last.color[3] > 0.99, "{last:?}");
        // Thin end = tighter spacing: more stamps than the full-size stroke.
        assert!(st.len() > plain.len(), "{} vs {}", st.len(), plain.len());
        // Monotone along the taper.
        assert!(st.windows(2).all(|w| w[1].rad_soft[0] >= w[0].rad_soft[0] - 1e-4));
    }

    #[test]
    fn blend_runs_split_on_the_blend_and_keep_order() {
        let mk = |b: f32| Stamp { extra: [b, 0.0], ..bytemuck::Zeroable::zeroed() };
        let r = blend_runs(&[mk(0.0), mk(0.0), mk(4.0), mk(0.0), mk(1.0), mk(1.0)]);
        assert_eq!(r, vec![(0, 2, 0), (2, 1, 4), (3, 1, 0), (4, 2, 1)]);
        assert!(blend_runs(&[]).is_empty());
    }
}
