//! Owns the comp machinery: pipeline cache keyed by topology hash, reader
//! textures, blur intermediates, output storage texture, shared uniforms.
//!
//! A compiled topology is a PLAN: zero or more blur stages (subtree pass +
//! separable H/V passes into per-node intermediate textures) followed by the
//! main fused pass. All passes share one packed uniform buffer; blur sigma
//! and direction live in tiny per-node uniforms. Parameter edits are buffer
//! writes only.

use crate::codegen::{self, FilterKind, GeneratedPlan, SourceBinding};
use crate::decode::ImageInfo;
use crate::graph::{Graph, NodeKind};
use crate::nodes::stage::{Bind, Dispatch, StageCx, StageProgram, TexFormat};
use crate::timing::GpuTimer;
use std::collections::HashMap;
use std::sync::Arc;
use web_time::Instant;

// 1024 vec4 slots. 4096 was enough until splines got a real sample budget:
// one dense mask can want ~145 slots, and two of them tipped a whole graph
// into the rejection path below. Native adapters guarantee far more than
// this (the WebGL2 floor is exactly 16 KiB), and the buffer is written only
// as far as the live layout reaches, so the extra costs nothing per frame.
const UNIFORM_BYTES: u64 = 16384;

#[derive(Clone, Copy, Debug, Default)]
pub struct CompileTiming {
    pub codegen_ms: f32,
    pub module_ms: f32,
    pub pipeline_ms: f32,
    pub total_ms: f32,
    pub cache_hit: bool,
}

struct Compiled {
    main_pipeline: wgpu::ComputePipeline,
    /// Parallel to plan.blurs (subtree pipelines).
    blur_pipelines: Vec<wgpu::ComputePipeline>,
    /// Parallel to plan.blurs: one pipeline per AUX subtree the stage
    /// declares (Defocus's depth; a Shader node's inputs 1..). Empty for
    /// stages that take one image input.
    aux_pipelines: Vec<Vec<wgpu::ComputePipeline>>,
    /// Parallel to plan.blurs: a user Shader stage's pass pipelines, all
    /// built from ONE module (the module compile is the expensive half, and
    /// the passes differ only in entry point).
    shader_pipelines: Vec<Vec<wgpu::ComputePipeline>>,
    plan: GeneratedPlan,
}

pub struct ReaderTexture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    /// The TEXTURE's size: the reader's window, else its whole frame.
    pub info: ImageInfo,
    /// The frame the window is a box of (equal to `info` when whole).
    pub plate: ImageInfo,
    /// How the reader's frames are stored (`decode::Pack`): the texture's
    /// format follows it, and its size follows the window.
    pub pack: crate::decode::Pack,
}

/// The texture format a reader's frames upload into. Every one of them is
/// float-filterable, so the bind group layout and the shader are the same
/// whatever the pack: an R texture samples as (r, 0, 0, 1) and codegen
/// broadcasts it, unorm samples as 0..1.
pub fn reader_format(pack: crate::decode::Pack) -> wgpu::TextureFormat {
    match (pack.lanes(), pack.eight) {
        (4, false) => wgpu::TextureFormat::Rgba16Float,
        (2, false) => wgpu::TextureFormat::Rg16Float,
        (1, false) => wgpu::TextureFormat::R16Float,
        (4, true) => wgpu::TextureFormat::Rgba8Unorm,
        (2, true) => wgpu::TextureFormat::Rg8Unorm,
        _ => wgpu::TextureFormat::R8Unorm,
    }
}

struct OutputTex {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    width: u32,
    height: u32,
}

/// What a Blender node's layer holds: the last picture that landed and the
/// stamp it came with, so the app can tell whether the frame on the
/// canvas is the one the playhead asks for.
pub struct BlenderLayer {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub width: u32,
    pub height: u32,
    pub format: wgpu::TextureFormat,
    pub stamp: Option<BlenderStamp>,
}

/// The texture format a ring pixel format lands in.
pub fn blender_format(fmt: u32) -> Option<wgpu::TextureFormat> {
    match fmt {
        crate::blender::shm::FMT_RGBA16F => Some(wgpu::TextureFormat::Rgba16Float),
        crate::blender::shm::FMT_RGBA32F => Some(wgpu::TextureFormat::Rgba32Float),
        _ => None,
    }
}

/// The provenance of the pixels in a Blender layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BlenderStamp {
    /// The BLENDER frame it was drawn at.
    pub frame: i64,
    /// The session's scene version when it was drawn.
    pub scene: u64,
    /// Nothing left to refine (viewport settled, or a final render).
    pub settled: bool,
    pub is_final: bool,
}

/// Intermediates for one Blur node: A holds the subtree render and, after the
/// V pass, the final blurred result consumers sample; B is the H scratch.
/// One loop stage's dispatch set: the first pass reads the SEED (aux 0) and
/// writes whichever of a/b makes the alternation END in b; later passes
/// ping-pong. Four groups cover every n.
struct LoopBgs {
    seed_to_a: wgpu::BindGroup,
    seed_to_b: wgpu::BindGroup,
    ab: wgpu::BindGroup,
    ba: wgpu::BindGroup,
}

struct BlurTex {
    /// The stage's input: its subtree, rendered at the canvas size.
    a: (wgpu::Texture, wgpu::TextureView),
    /// A second full-canvas target, for the kinds that still build their
    /// passes by hand (Defocus writes into it; a Loop ping-pongs a/b).
    b: Option<(wgpu::Texture, wgpu::TextureView)>,
    /// Row prefix sums (rgba32float), allocated for Fast-mode defocus only.
    p: Option<(wgpu::Texture, wgpu::TextureView)>,
    /// A depth-wired Defocus's CoC map, its tile dilation, and (Fast) the
    /// near layer's prefix sums.
    maps: Option<DefocusMaps>,
    /// AUX subtree renders: further image ports resolved to their own
    /// full-canvas textures (Defocus's depth branch; a Shader node's inputs
    /// 1..). Index i matches plan aux shader i.
    aux: Vec<(wgpu::Texture, wgpu::TextureView)>,
    /// User Shader stages only: one result texture per pass (the last is
    /// what downstream samples) plus one uniform per pass.
    shader: Option<ShaderTex>,
    /// The hand-built kinds' one uniform (Defocus's DP block).
    params: wgpu::Buffer,
    width: u32,
    height: u32,
}

/// What a depth-wired Defocus derives from its depth branch each frame:
/// the signed CoC per pixel (`cmap`, rg32float: coc, z), the 8x8 tile max
/// of it dilated by the max radius (`tiles` -> `tiles_h` -> `tiles_v`; the
/// last is what the passes read), and in Fast mode the near layer's prefix
/// sums (`pn`: rgba * near weight, `pw`: the weight).
struct DefocusMaps {
    cmap: (wgpu::Texture, wgpu::TextureView),
    tiles: (wgpu::Texture, wgpu::TextureView),
    tiles_h: (wgpu::Texture, wgpu::TextureView),
    tiles_v: (wgpu::Texture, wgpu::TextureView),
    pn: Option<(wgpu::Texture, wgpu::TextureView)>,
    pw: Option<(wgpu::Texture, wgpu::TextureView)>,
    tile_dims: (u32, u32),
}

/// A Defocus stage's dispatch set: the map passes (depth wired), the
/// prefix pass (Fast), then the gather.
struct DefocusBgs {
    maps: Option<[wgpu::BindGroup; 4]>,
    prefix: Option<wgpu::BindGroup>,
    gather: wgpu::BindGroup,
    accurate: bool,
}

/// Intermediates for one user Shader stage.
///
/// Every pass keeps its OWN result texture for the whole chain rather than
/// ping-ponging, because the contract says pass 4 may sample pass 1 — which
/// is the entire reason multi-pass exists. That makes this node a real
/// memory cost (4 passes at 4K is ~283 MB of rgba16f) and it is meant to
/// look like one.
struct ShaderTex {
    passes: Vec<(wgpu::Texture, wgpu::TextureView)>,
    /// One uniform per pass: identical but for `pass_index`, which is what
    /// lets all passes share a single module.
    params: Vec<wgpu::Buffer>,
}

/// A node's own chain, materialised: the textures its program declared and
/// one uniform buffer per pass that binds one. Built for one canvas size
/// and one plan; either changing rebuilds it.
struct StageRt {
    program: StageProgram,
    canvas: (u32, u32),
    hash: u64,
    texes: HashMap<String, (wgpu::Texture, wgpu::TextureView)>,
    uniforms: Vec<Option<wgpu::Buffer>>,
}

/// Identity of a pass module: the hash of its text.
fn wgsl_hash(wgsl: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    wgsl.hash(&mut h);
    h.finish()
}

struct StageBindGroups {
    subtree: wgpu::BindGroup,
    aux: Vec<wgpu::BindGroup>,
    shader: Option<Vec<wgpu::BindGroup>>,
    defocus: Option<DefocusBgs>,
    /// A node's own chain (`FilterKind::Node`): one bind group per pass, in
    /// the program's order.
    passes: Vec<wgpu::BindGroup>,
}



struct FrameBindGroups {
    main: wgpu::BindGroup,
    stages: Vec<StageBindGroups>,
}

struct InflightCompile {
    hash: u64,
    started: Instant,
    rx: std::sync::mpsc::Receiver<Option<(Compiled, CompileTiming)>>,
}

/// A LUT node's table on the GPU (`Compositor::update_lut_tables`).
struct LutTex {
    hash: u64,
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
}

/// A Text or Burnin node's glyph atlas on the GPU
/// (`Compositor::update_text_tables`).
struct TextTex {
    hash: u64,
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
}

pub struct Compositor {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    sampler: wgpu::Sampler,
    uniform: wgpu::Buffer,
    dummy: ReaderTexture,
    defocus_prefix_pipeline: wgpu::ComputePipeline,
    defocus_disc_pipeline: wgpu::ComputePipeline,
    defocus_bokeh_pipeline: wgpu::ComputePipeline,
    defocus_coc_pipeline: wgpu::ComputePipeline,
    defocus_tile_pipeline: wgpu::ComputePipeline,
    defocus_tile_h_pipeline: wgpu::ComputePipeline,
    defocus_tile_v_pipeline: wgpu::ComputePipeline,
    /// 1x1 stand-ins for a depthless Defocus's unused map bindings, in the
    /// storage formats those bindings declare.
    dummy_rg32: (wgpu::Texture, wgpu::TextureView),
    dummy_rgba32: (wgpu::Texture, wgpu::TextureView),
    dummy_r32: (wgpu::Texture, wgpu::TextureView),
    /// Motion blur: sub-frame samples are summed here and resolved back into
    /// the output. Allocated only when a blurred frame is actually rendered.
    accum_pipeline: wgpu::ComputePipeline,
    accum_resolve_pipeline: wgpu::ComputePipeline,
    /// A MotionBlur stage's accumulator, f32 per channel: a storage BUFFER
    /// because a read-write rgba32float storage texture needs an adapter
    /// feature not every target has.
    accum_bufs: HashMap<codegen::StageId, (wgpu::Buffer, u32, u32)>,
    /// MotionBlur stages resolved since the last `encode_comp`: their `a`
    /// holds the average, so nothing re-renders into it and nothing that
    /// only feeds them runs again.
    motion_done: std::collections::HashSet<codegen::StageId>,
    accum_params: wgpu::Buffer,
    /// One intermediate per FILTER STAGE — (node, time path), so a blur
    /// upstream of two different retimes gets a texture per frame it has to
    /// filter rather than one they fight over.
    blur_texes: HashMap<codegen::StageId, BlurTex>,
    /// A node's own chain, per stage: its textures and uniforms, built for
    /// one canvas size and one plan.
    stage_rts: HashMap<codegen::StageId, StageRt>,
    /// Compiled pass modules, keyed by the WGSL text's hash. Built-ins are
    /// compiled up front in `new`; anything else the first time it runs.
    pass_pipelines: HashMap<u64, wgpu::ComputePipeline>,
    /// (frame, fps) the comp is being evaluated at. Only user Shader nodes
    /// read it — it is what backs their `P.time` / `P.frame` built-ins, the
    /// one thing a Matchbox-style shader expects that the graph model has no
    /// other reason to carry down here.
    clock: (f32, f32),
    paint_raster: crate::paint::PaintRaster,
    paint_layers: HashMap<crate::graph::NodeId, crate::paint::PaintLayer>,
    /// 2dini geometry layers, one per Geo node: the same memoised-layer
    /// arrangement as paint, with a CPU evaluation in front of the upload.
    geo_raster: crate::georaster::GeoRaster,
    geo_layers: HashMap<crate::graph::NodeId, crate::georaster::GeoLayer>,
    /// One table texture per LUT node, keyed on the file's content hash.
    lut_tables: HashMap<crate::graph::NodeId, LutTex>,
    text_tables: HashMap<crate::graph::NodeId, TextTex>,
    /// A Text node's layer (`textgpu`): glyph quads drawn over the font's
    /// atlas, re-recorded only when what it draws changes.
    text_raster: crate::textgpu::TextRaster,
    text_layers: HashMap<crate::graph::NodeId, crate::textgpu::TextLayer>,
    /// A Plasma node's layer (`plasmagpu`): its particles, drawn
    /// additively, re-drawn only when its uniform changes.
    plasma_raster: crate::plasmagpu::PlasmaRaster,
    plasma_layers: HashMap<crate::graph::NodeId, crate::plasmagpu::PlasmaLayer>,
    /// A Splat node's layer (`splatgpu`): its capture projected, sorted
    /// and drawn, re-drawn only when its uniform or cloud changes, and
    /// only while the plan samples it.
    splat_raster: crate::splatgpu::SplatRaster,
    splat_layers: HashMap<crate::graph::NodeId, crate::splatgpu::SplatLayer>,
    /// A Particles node's simulation and layer (`particlegpu`).
    particle_raster: crate::particlegpu::ParticleRaster,
    particle_layers: HashMap<crate::graph::NodeId, crate::particlegpu::ParticleLayer>,
    /// A 2dini node's light and SDF effects (`geolight`), only while its
    /// look is on; the fused pass then samples the finished picture.
    geo_lit: crate::geolight::GeoLit,
    geo_lit_layers: HashMap<crate::graph::NodeId, crate::geolight::LitLayer>,
    /// Blender layers, one per Blender node: the live picture the app
    /// uploads straight from the session's ring (`blender_upload`). No CPU
    /// evaluation, no re-record — a texture and a stamp saying which frame
    /// it holds.
    blender_layers: HashMap<(crate::graph::NodeId, u8), BlenderLayer>,
    /// Loop stages: per-stage iteration count (an encode-time pass count,
    /// written by `write_params`), the LoopIn -> stage map (so a tap sampled
    /// OUTSIDE its loop resolves to the finished result), and the two
    /// ping-pong + two seed-primed bind groups per stage.
    loop_counts: HashMap<codegen::StageId, u32>,
    loop_stages: HashMap<crate::graph::NodeId, codegen::StageId>,
    loop_bgs: HashMap<codegen::StageId, LoopBgs>,
    /// Clone and smear strokes' offset layer, only for nodes that have any.
    paint_source_layers: HashMap<crate::graph::NodeId, crate::paint::PaintLayer>,
    /// Blur / dodge / burn amounts, additive, only for nodes that have any.
    paint_effect_layers: HashMap<crate::graph::NodeId, crate::paint::PaintLayer>,
    /// The tick stroke lifetimes are resolved against (see `set_paint_frame`).
    paint_frame: u64,
    pipelines: HashMap<u64, Compiled>,
    current: Option<u64>,
    /// Topology the uniform buffer was last packed for. Params are packed
    /// into the LIVE pipeline's layout, so when an async compile lands the
    /// buffer belongs to the outgoing topology and MUST be rewritten — even
    /// though the graph itself hasn't changed.
    params_hash: Option<u64>,
    readers: Vec<Option<ReaderTexture>>,
    output: Option<OutputTex>,
    frame_bgs: Option<FrameBindGroups>,
    inflight: Option<InflightCompile>,
    /// Topologies that failed validation — don't respawn their compile every
    /// frame (codegen is deterministic, so a failure is permanent).
    failed: std::collections::HashSet<u64>,
    pub last_compile: CompileTiming,
}

fn make_compute_pipeline(
    device: &wgpu::Device,
    label: &str,
    wgsl: &str,
    layout: Option<&wgpu::PipelineLayout>,
) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

/// Explicit layout for fused-style passes. Auto layout would DROP bindings
/// the shader doesn't reference (e.g. the shared uniform in a main pass whose
/// params all live in blur subtrees), making bind-group construction
/// unpredictable — so the layout is derived from the source count instead.
fn fused_bgl(device: &wgpu::Device, n_sources: usize) -> wgpu::BindGroupLayout {
    let mut entries = vec![
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        },
    ];
    for i in 0..n_sources {
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 2 + i as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        });
    }
    entries.push(wgpu::BindGroupLayoutEntry {
        binding: 2 + n_sources as u32,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: wgpu::TextureFormat::Rgba16Float,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    });
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("fused-bgl"),
        entries: &entries,
    })
}

/// A user Shader stage's pass pipelines: ONE module, one layout, N entry
/// points. Module creation is where Metal spends its ~100-150ms on novel
/// WGSL, so building four passes must not mean building it four times.
fn make_shader_pipelines(
    device: &wgpu::Device,
    label: &str,
    wgsl: &str,
    n_sources: usize,
    passes: usize,
) -> Vec<wgpu::ComputePipeline> {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let bgl = fused_bgl(device, n_sources);
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("shader-pl"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    (0..passes)
        .map(|i| {
            let entry = crate::usershader::ShaderDef::entry(i);
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pl),
                module: &module,
                entry_point: Some(&entry),
                compilation_options: Default::default(),
                cache: None,
            })
        })
        .collect()
}

fn make_fused_pipeline(device: &wgpu::Device, label: &str, wgsl: &str, n_sources: usize) -> wgpu::ComputePipeline {
    let bgl = fused_bgl(device, n_sources);
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("fused-pl"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    make_compute_pipeline(device, label, wgsl, Some(&pl))
}

/// Compile every pipeline of a plan. Runs on the caller's thread (sync path)
/// or a worker (async path); the error scope is per-thread in wgpu 29.
/// Returns (compiled, pipeline_ms).
fn compile_plan(device: &wgpu::Device, plan: GeneratedPlan) -> (Option<Compiled>, f32) {
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let m0 = Instant::now();
    let blur_pipelines: Vec<_> = plan
        .blurs
        .iter()
        .map(|(n, _, s, _)| {
            make_fused_pipeline(device, &format!("filter-subtree-{n:?}"), &s.wgsl, s.sources.len())
        })
        .collect();
    let aux_pipelines: Vec<Vec<_>> = plan
        .blurs
        .iter()
        .map(|(n, _, _, aux)| {
            aux.iter()
                .enumerate()
                .map(|(i, a)| {
                    make_fused_pipeline(
                        device,
                        &format!("filter-aux-{n:?}-{i}"),
                        &a.wgsl,
                        a.sources.len(),
                    )
                })
                .collect()
        })
        .collect();
    // One module, N pass pipelines, per user Shader stage.
    let shader_pipelines: Vec<Vec<_>> = plan
        .blurs
        .iter()
        .map(|(n, kind, _, _)| match (kind, plan.shaders.get(n)) {
            (FilterKind::Shader { .. }, Some(def)) => make_shader_pipelines(
                device,
                &format!("user-shader-{n:?}"),
                &def.wgsl,
                def.source_count(),
                def.passes,
            ),
            _ => Vec::new(),
        })
        .collect();
    let main_pipeline =
        make_fused_pipeline(device, "fused-main", &plan.main.wgsl, plan.main.sources.len());
    #[cfg(not(target_arch = "wasm32"))]
    let err = pollster::block_on(scope.pop()).map(|e| e.to_string());
    #[cfg(target_arch = "wasm32")]
    let err = {
        drop(scope);
        web_validate(&plan)
    };
    let total = m0.elapsed().as_secs_f32() * 1e3;
    if let Some(e) = err {
        log::error!("comp plan failed validation: {e}\n--- main WGSL ---\n{}", plan.main.wgsl);
        for (n, _, s, aux) in &plan.blurs {
            log::error!("--- filter {n:?} WGSL ---\n{}", s.wgsl);
            for (i, a) in aux.iter().enumerate() {
                log::error!("--- filter {n:?} AUX {i} WGSL ---\n{}", a.wgsl);
            }
        }
        for (n, def) in &plan.shaders {
            log::error!("--- user shader {n:?} WGSL ---\n{}", def.wgsl);
        }
        return (None, total);
    }
    (
        Some(Compiled { main_pipeline, blur_pipelines, aux_pipelines, shader_pipelines, plan }),
        total,
    )
}

/// wasm32 stand-in for the error scope: a browser resolves `scope.pop()` on
/// its own event loop, so a compile cannot wait on it. Every module of the
/// plan goes through naga instead — the front end wgpu runs natively — so a
/// plan that would fail there is still rejected, not left to render garbage.
#[cfg(target_arch = "wasm32")]
fn web_validate(plan: &GeneratedPlan) -> Option<String> {
    let mut srcs = vec![plan.main.wgsl.as_str()];
    for (_, _, s, aux) in &plan.blurs {
        srcs.push(&s.wgsl);
        srcs.extend(aux.iter().map(|a| a.wgsl.as_str()));
    }
    srcs.extend(plan.shaders.values().map(|d| d.wgsl.as_str()));
    srcs.into_iter().find_map(|src| {
        let module = match naga::front::wgsl::parse_str(src) {
            Ok(m) => m,
            Err(e) => return Some(e.emit_to_string(src)),
        };
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::default(),
        )
        .validate(&module)
        .err()
        .map(|e| e.emit_to_string(src))
    })
}

impl Compositor {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>, reader_slots: usize) -> Self {
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("bilinear-clamp"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fused-params"),
            size: UNIFORM_BYTES,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let dummy = Self::make_texture(
            &device,
            ImageInfo { width: 1, height: 1 },
            Default::default(),
            "dummy-reader",
        );
        let dc = codegen::DEFOCUS_COMMON_WGSL;
        let defocus_prefix_pipeline = make_compute_pipeline(
            &device,
            "defocus-prefix",
            &format!("{dc}{}", codegen::DEFOCUS_PREFIX_WGSL),
            None,
        );
        let defocus_coc_pipeline = make_compute_pipeline(
            &device,
            "defocus-coc",
            &format!("{dc}{}", codegen::DEFOCUS_COC_WGSL),
            None,
        );
        let defocus_tile_pipeline =
            make_compute_pipeline(&device, "defocus-tile", codegen::DEFOCUS_TILE_WGSL, None);
        let defocus_tile_h_pipeline = make_compute_pipeline(
            &device,
            "defocus-tile-h",
            &format!("{dc}{}", codegen::DEFOCUS_TILEMAX_WGSL.replace("DIR", "1, 0")),
            None,
        );
        let defocus_tile_v_pipeline = make_compute_pipeline(
            &device,
            "defocus-tile-v",
            &format!("{dc}{}", codegen::DEFOCUS_TILEMAX_WGSL.replace("DIR", "0, 1")),
            None,
        );
        let dummy_rg32 = Self::make_tex(&device, 1, 1, wgpu::TextureFormat::Rg32Float, "dummy-rg32");
        let dummy_rgba32 =
            Self::make_tex(&device, 1, 1, wgpu::TextureFormat::Rgba32Float, "dummy-rgba32");
        let dummy_r32 = Self::make_tex(&device, 1, 1, wgpu::TextureFormat::R32Float, "dummy-r32");
        let defocus_disc_pipeline = make_compute_pipeline(
            &device,
            "defocus-disc",
            &format!("{}{}", codegen::DEFOCUS_COMMON_WGSL, codegen::DEFOCUS_DISC_WGSL),
            None,
        );
        let defocus_bokeh_pipeline = make_compute_pipeline(
            &device,
            "defocus-bokeh",
            &format!("{}{}", codegen::DEFOCUS_COMMON_WGSL, codegen::DEFOCUS_BOKEH_WGSL),
            None,
        );
        let accum_pipeline =
            make_compute_pipeline(&device, "mblur-accum", codegen::ACCUM_WGSL, None);
        let accum_resolve_pipeline = make_compute_pipeline(
            &device,
            "mblur-resolve",
            codegen::ACCUM_RESOLVE_WGSL,
            None,
        );
        let accum_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mblur-params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let paint_raster = crate::paint::PaintRaster::new(&device);
        let geo_raster = crate::georaster::GeoRaster::new(&device);
        let text_raster = crate::textgpu::TextRaster::new(&device);
        let plasma_raster = crate::plasmagpu::PlasmaRaster::new(&device);
        let splat_raster = crate::splatgpu::SplatRaster::new(&device);
        crate::splat::set_gpu_budget(
            (device.limits().max_storage_buffer_binding_size as u64).min(device.limits().max_buffer_size),
        );
        let particle_raster = crate::particlegpu::ParticleRaster::new(&device);
        let geo_lit = crate::geolight::GeoLit::new(&device);
        // Every built-in stage's pass modules, compiled once here rather
        // than on a node's first use.
        let mut pass_pipelines = HashMap::new();
        let one = crate::nodes::stage::one_channel()
            .then(crate::nodes::builtin_one_channel_wgsl)
            .unwrap_or_default();
        for wgsl in crate::nodes::builtin_pass_wgsl().into_iter().chain(one) {
            pass_pipelines
                .entry(wgsl_hash(wgsl))
                .or_insert_with(|| make_compute_pipeline(&device, "stage-pass", wgsl, None));
        }
        Self {
            device,
            queue,
            sampler,
            uniform,
            dummy,
            defocus_prefix_pipeline,
            defocus_disc_pipeline,
            defocus_bokeh_pipeline,
            defocus_coc_pipeline,
            defocus_tile_pipeline,
            defocus_tile_h_pipeline,
            defocus_tile_v_pipeline,
            dummy_rg32,
            dummy_rgba32,
            dummy_r32,
            accum_pipeline,
            accum_resolve_pipeline,
            accum_bufs: HashMap::new(),
            motion_done: Default::default(),
            accum_params,
            blur_texes: HashMap::new(),
            stage_rts: HashMap::new(),
            pass_pipelines,
            clock: (0.0, 24.0),
            paint_raster,
            paint_layers: HashMap::new(),
            geo_raster,
            geo_layers: HashMap::new(),
            lut_tables: HashMap::new(),
            text_tables: HashMap::new(),
            text_raster,
            text_layers: HashMap::new(),
            plasma_raster,
            plasma_layers: HashMap::new(),
            splat_raster,
            splat_layers: HashMap::new(),
            particle_raster,
            particle_layers: HashMap::new(),
            geo_lit,
            geo_lit_layers: HashMap::new(),
            blender_layers: HashMap::new(),
            loop_counts: HashMap::new(),
            loop_stages: HashMap::new(),
            loop_bgs: HashMap::new(),
            paint_source_layers: HashMap::new(),
            paint_effect_layers: HashMap::new(),
            paint_frame: 0,
            pipelines: HashMap::new(),
            current: None,
            params_hash: None,
            readers: (0..reader_slots).map(|_| None).collect(),
            output: None,
            frame_bgs: None,
            inflight: None,
            failed: std::collections::HashSet::new(),
            last_compile: CompileTiming::default(),
        }
    }

    fn make_texture(
        device: &wgpu::Device,
        info: ImageInfo,
        pack: crate::decode::Pack,
        label: &str,
    ) -> ReaderTexture {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: info.width,
                height: info.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: reader_format(pack),
            // COPY_SRC: dual view's second compositor is fed the same plates
            // by a GPU copy of these.
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        ReaderTexture { texture, view, info, plate: info, pack }
    }

    fn make_storage_tex(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        label: &str,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            // COPY_SRC: a stage's aux may be read back at a pointer (the
            // Defocus focus pick reads its depth texture).
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    fn make_tex(
        device: &wgpu::Device,
        w: u32,
        h: u32,
        format: wgpu::TextureFormat,
        label: &str,
    ) -> (wgpu::Texture, wgpu::TextureView) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        (texture, view)
    }

    /// Small uniform buffer for a filter pass.
    fn make_uniform(device: &wgpu::Device, size: u64, label: &str) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Make sure reader slot `source` has a texture of the right size.
    /// Make sure reader slot `source` has a texture of `info`'s size.
    /// False when the GPU refused one: the slot is left EMPTY (it samples
    /// as the dummy, black) rather than the refusal reaching wgpu's
    /// uncaptured-error handler, which aborts the app. At 6K a reader
    /// texture is ~290 MB of VRAM, and a comp with a harp and two
    /// cryptomattes asks for dozens.
    /// What the compositor holds in VRAM right now, in bytes, by kind:
    /// (label, bytes). Textures only (buffers are small beside them): the
    /// numbers to read when a 6K comp spills out of a 24 GB card.
    pub fn vram_report(&self) -> Vec<(&'static str, u64)> {
        fn tb(t: &wgpu::Texture) -> u64 {
            let s = t.size();
            let bpp = t.format().block_copy_size(None).unwrap_or(8) as u64;
            s.width as u64 * s.height as u64 * s.depth_or_array_layers as u64 * bpp
        }
        let readers: u64 = self.readers.iter().flatten().map(|r| tb(&r.texture)).sum();
        let output = self.output.as_ref().map_or(0, |o| tb(&o.texture));
        let mut subtree = 0u64;
        let mut defocus = 0u64;
        let mut shader = 0u64;
        for bt in self.blur_texes.values() {
            subtree += tb(&bt.a.0);
            subtree += bt.aux.iter().map(|t| tb(&t.0)).sum::<u64>();
            defocus += bt.b.as_ref().map_or(0, |t| tb(&t.0)) + bt.p.as_ref().map_or(0, |t| tb(&t.0));
            if let Some(m) = &bt.maps {
                defocus += tb(&m.cmap.0) + tb(&m.tiles.0) + tb(&m.tiles_h.0) + tb(&m.tiles_v.0);
                defocus += m.pn.as_ref().map_or(0, |t| tb(&t.0)) + m.pw.as_ref().map_or(0, |t| tb(&t.0));
            }
            if let Some(sh) = &bt.shader {
                shader += sh.passes.iter().map(|t| tb(&t.0)).sum::<u64>();
            }
        }
        let stages: u64 = self.stage_rts.values().flat_map(|rt| rt.texes.values()).map(|t| tb(&t.0)).sum();
        let layers: u64 = self
            .paint_layers
            .values()
            .chain(self.paint_source_layers.values())
            .chain(self.paint_effect_layers.values())
            .map(|l| tb(&l.texture) + l.mul.as_ref().map_or(0, |m| tb(&m.0)))
            .sum();
        vec![
            ("readers", readers),
            ("output", output),
            ("stage inputs", subtree),
            ("stage scratch", stages),
            ("defocus", defocus),
            ("shader", shader),
            ("paint", layers),
        ]
    }

    pub fn ensure_reader(&mut self, source: usize, info: ImageInfo) -> bool {
        self.ensure_reader_packed(source, info, Default::default())
    }

    /// `ensure_reader` for frames stored as `pack` (its texture format).
    /// `plate` is the reader's whole frame; the texture is the pack's
    /// window of it (`Pack::dims`).
    pub fn ensure_reader_packed(
        &mut self,
        source: usize,
        plate: ImageInfo,
        pack: crate::decode::Pack,
    ) -> bool {
        if source >= self.readers.len() {
            self.readers.resize_with(source + 1, || None);
        }
        let info = pack.dims(plate);
        let needs = match &self.readers[source] {
            Some(t) => t.plate != plate || t.pack != pack,
            None => true,
        };
        if !needs {
            return true;
        }
        // A WINDOW moved over the same plate (a crop edit settling, or a
        // box dragged out of its window): carry the pixels both textures
        // hold across, so the picture never blinks while the new frames
        // decode. Anything else is a real resize.
        let carry = match self.readers[source].take() {
            Some(old)
                if old.plate == plate
                    && reader_format(old.pack) == reader_format(pack)
                    && old.pack.scale == pack.scale
                    && (old.pack.window.is_some() || pack.window.is_some()) =>
            {
                Some(old)
            }
            // Let go of the old texture (and the bind groups holding it)
            // BEFORE allocating the new one, so a resize — a proxy toggle —
            // never needs both in VRAM at once.
            _ => None,
        };
        self.frame_bgs = None;
        #[cfg(not(target_arch = "wasm32"))]
        let scope = self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let tex = Self::make_texture(&self.device, info, pack, &format!("reader-{source}"));
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(e) = pollster::block_on(scope.pop()) {
            log::error!(
                "reader {source}: no GPU memory for a {}x{} texture ({e})",
                info.width,
                info.height
            );
            return false;
        }
        let mut tex = tex;
        tex.plate = plate;
        if let Some(old) = carry {
            let rect = |p: &crate::decode::Pack, i: ImageInfo| match p.stored_window(plate) {
                Some(w) => (w.x, w.y, w.w, w.h),
                None => (0, 0, i.width, i.height),
            };
            let (ax, ay, aw, ah) = rect(&old.pack, old.info);
            let (bx, by, bw, bh) = rect(&pack, info);
            let (x0, y0) = (ax.max(bx), ay.max(by));
            let (x1, y1) = ((ax + aw).min(bx + bw), (ay + ah).min(by + bh));
            if x1 > x0 && y1 > y0 {
                let mut enc = self.device.create_command_encoder(&Default::default());
                fn at(t: &wgpu::Texture, x: u32, y: u32) -> wgpu::TexelCopyTextureInfo<'_> {
                    wgpu::TexelCopyTextureInfo {
                        texture: t,
                        mip_level: 0,
                        origin: wgpu::Origin3d { x, y, z: 0 },
                        aspect: wgpu::TextureAspect::All,
                    }
                }
                enc.copy_texture_to_texture(
                    at(&old.texture, x0 - ax, y0 - ay),
                    at(&tex.texture, x0 - bx, y0 - by),
                    wgpu::Extent3d { width: x1 - x0, height: y1 - y0, depth_or_array_layers: 1 },
                );
                self.queue.submit([enc.finish()]);
            }
        }
        self.readers[source] = Some(tex);
        true
    }

    /// Drop the textures of reader slots that are not in `live`. Slots
    /// are handed out monotonically (a new Read, a paste, a re-minted
    /// channel or cryptomatte rank all take a fresh one) and nothing used
    /// to free the texture of a slot nobody reads any more, so a long
    /// session leaked one per retired reader — ~290 MB of VRAM each at 6K,
    /// until an allocation failed. Returns the bytes released.
    pub fn retain_readers(&mut self, live: &[usize]) -> u64 {
        let mut freed = 0u64;
        for (slot, tex) in self.readers.iter_mut().enumerate() {
            if tex.is_some() && !live.contains(&slot) {
                let t = tex.take().unwrap();
                freed += t.info.width as u64 * t.info.height as u64 * t.pack.bytes_per_pixel() as u64;
            }
        }
        if freed > 0 {
            // The frame's bind groups hold views of what was just dropped;
            // the memory is only released once they go too.
            self.frame_bgs = None;
        }
        freed
    }

    /// Hand-built stage chains holding scratch textures (Defocus, Loop,
    /// Shader) — what `engine/tests/gpu_memory.rs` watches for a leak.
    pub fn stage_scratch_count(&self) -> usize {
        self.blur_texes.len()
    }

    pub fn reader_texture(&self, source: usize) -> Option<&ReaderTexture> {
        self.readers.get(source).and_then(|t| t.as_ref())
    }

    /// Output resolution follows the largest connected source (fallback HD).
    pub fn ensure_output(&mut self, width: u32, height: u32) {
        let needs = match &self.output {
            Some(o) => o.width != width || o.height != height,
            None => true,
        };
        if needs {
            let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("fused-output"),
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                // COPY_DST: cached-render playback uploads baked frames
                // straight into the output, bypassing the comp passes.
                usage: wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            self.output = Some(OutputTex { texture, view, width, height });
            self.frame_bgs = None;
        }
    }

    pub fn output_view(&self) -> Option<&wgpu::TextureView> {
        self.output.as_ref().map(|o| &o.view)
    }

    pub fn output_texture(&self) -> Option<&wgpu::Texture> {
        self.output.as_ref().map(|o| &o.texture)
    }

    pub fn output_size(&self) -> Option<(u32, u32)> {
        self.output.as_ref().map(|o| (o.width, o.height))
    }

    /// Compile (or fetch from cache) the plan for this topology, blocking.
    /// Returns false if it failed validation (previous plan stays live).
    pub fn set_graph(&mut self, graph: &Graph) -> bool {
        let hash = graph.topology_hash();
        if self.current == Some(hash) && self.pipelines.contains_key(&hash) {
            return true;
        }
        if self.pipelines.contains_key(&hash) {
            self.current = Some(hash);
            self.frame_bgs = None;
            self.last_compile = CompileTiming { cache_hit: true, ..Default::default() };
            return true;
        }
        if self.failed.contains(&hash) {
            return false;
        }

        let t0 = Instant::now();
        let plan = codegen::generate(graph);
        let t1 = Instant::now();
        if plan.layout.byte_size() > UNIFORM_BYTES {
            log::error!(
                "graph needs {} uniform bytes, buffer holds {UNIFORM_BYTES}; rejecting topology",
                plan.layout.byte_size()
            );
            self.failed.insert(hash);
            return false;
        }
        let (compiled, pipeline_ms) = compile_plan(&self.device, plan);
        let Some(compiled) = compiled else {
            self.failed.insert(hash);
            return false;
        };
        self.last_compile = CompileTiming {
            codegen_ms: (t1 - t0).as_secs_f32() * 1e3,
            module_ms: 0.0,
            pipeline_ms,
            total_ms: t0.elapsed().as_secs_f32() * 1e3,
            cache_hit: false,
        };
        self.pipelines.insert(hash, compiled);
        self.current = Some(hash);
        self.frame_bgs = None;
        true
    }

    /// Non-blocking set_graph: cache hits switch immediately; a novel
    /// topology compiles on a worker thread while the previous pipeline keeps
    /// rendering, and swaps in when ready (Metal PSO compiles of novel WGSL
    /// run ~100ms — blocking would hitch playback on every rewire).
    ///
    /// Returns true when the requested topology is the one currently live.
    pub fn set_graph_async(&mut self, graph: &Graph) -> bool {
        // No threads in a browser: compile in place (web/).
        #[cfg(target_arch = "wasm32")]
        return self.set_graph(graph);
        #[cfg_attr(target_arch = "wasm32", allow(unreachable_code))]
        let hash = graph.topology_hash();
        if self.current == Some(hash) {
            return true;
        }
        if self.pipelines.contains_key(&hash) {
            self.current = Some(hash);
            self.frame_bgs = None;
            self.last_compile = CompileTiming { cache_hit: true, ..Default::default() };
            return true;
        }
        if self.failed.contains(&hash) {
            return false;
        }

        // Harvest or discard the in-flight compile.
        if let Some(inflight) = &self.inflight {
            if inflight.hash == hash {
                match inflight.rx.try_recv() {
                    Ok(Some((compiled, mut timing))) => {
                        timing.total_ms = inflight.started.elapsed().as_secs_f32() * 1e3;
                        self.pipelines.insert(hash, compiled);
                        self.current = Some(hash);
                        self.frame_bgs = None;
                        self.last_compile = timing;
                        self.inflight = None;
                        return true;
                    }
                    Ok(None) => {
                        // Compile failed (validation); stay on the old
                        // pipeline and never retry this topology.
                        self.failed.insert(hash);
                        self.inflight = None;
                        return false;
                    }
                    Err(std::sync::mpsc::TryRecvError::Empty) => return false,
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        self.inflight = None;
                        return false;
                    }
                }
            }
            // Requested topology changed again mid-compile: drop the old
            // request and fall through to spawn the new one.
        }

        let t0 = Instant::now();
        let plan = codegen::generate(graph);
        let codegen_ms = t0.elapsed().as_secs_f32() * 1e3;
        if plan.layout.byte_size() > UNIFORM_BYTES {
            log::error!(
                "graph needs {} uniform bytes, buffer holds {UNIFORM_BYTES}; rejecting topology",
                plan.layout.byte_size()
            );
            self.failed.insert(hash);
            return false;
        }
        let device = self.device.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let (compiled, pipeline_ms) = compile_plan(&device, plan);
            let result = compiled.map(|c| {
                (
                    c,
                    CompileTiming {
                        codegen_ms,
                        module_ms: 0.0,
                        pipeline_ms,
                        total_ms: 0.0, // stamped with wall time at swap
                        cache_hit: false,
                    },
                )
            });
            let _ = tx.send(result);
        });
        self.inflight = Some(InflightCompile { hash, started: t0, rx });
        false
    }

    pub fn current_wgsl(&self) -> Option<&str> {
        self.current
            .and_then(|h| self.pipelines.get(&h))
            .map(|c| c.plan.main.wgsl.as_str())
    }

    /// The depth texture a Defocus node's stage resolved this frame (its
    /// Z port, rendered at the canvas), for the focus pick to read one texel
    /// of. `None` when the node has no live stage or nothing on its Z port.
    pub fn defocus_depth_texture(&self, node: crate::graph::NodeId) -> Option<&wgpu::Texture> {
        let compiled = self.current.and_then(|h| self.pipelines.get(&h))?;
        compiled.plan.blurs.iter().find_map(|(sid, kind, _, _)| match kind {
            FilterKind::Defocus { depth: true, .. } if sid.0 == node => {
                self.blur_texes.get(sid).and_then(|bt| bt.aux.first()).map(|t| &t.0)
            }
            _ => None,
        })
    }

    /// Read the comp output back to the CPU as tightly-packed RGB f32.
    ///
    /// A full-frame readback and a GPU stall, so it is `pub` for exactly one
    /// caller: an explicit segment run. Nothing on a per-frame path may use
    /// it (manifesto rule 2 — readbacks are suspect by default).
    pub fn read_output_rgb(&self) -> Option<(Vec<f32>, u32, u32)> {
        let out = self.output.as_ref()?;
        let (w, h) = (out.width, out.height);
        // copy_texture_to_buffer wants 256-byte aligned rows, so the staging
        // buffer is padded and the rows are un-padded on the way out.
        let unpadded = (w as usize) * 8; // rgba16float
        let padded = unpadded.div_ceil(256) * 256;
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sam-readback"),
            size: (padded * h as usize) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("sam-rb") });
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &out.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded as u32),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        self.queue.submit(Some(enc.finish()));
        let (tx, rx) = std::sync::mpsc::channel();
        buf.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        // The map only resolves once the copy has actually run.
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().ok()?.ok()?;
        let mut rgb = vec![0.0f32; (w as usize) * (h as usize) * 3];
        {
            let view = buf.slice(..).get_mapped_range().expect("a mapped readback");
            for y in 0..h as usize {
                let row = &view[y * padded..y * padded + unpadded];
                let px: &[half::f16] = bytemuck::cast_slice(row);
                for x in 0..w as usize {
                    for c in 0..3 {
                        rgb[(y * w as usize + x) * 3 + c] = px[x * 4 + c].to_f32();
                    }
                }
            }
        }
        buf.unmap();
        Some((rgb, w, h))
    }

    /// The output as it is: rgba16f bytes, rows top-down, un-padded — what
    /// a Blender node hands over as a texture. The same stall as
    /// `read_output_rgb`, for the same reason and with the same rule: an
    /// explicit slow path (the node is red for it), never a per-frame
    /// convenience.
    pub fn read_output_rgba16(&self) -> Option<(Vec<u8>, u32, u32)> {
        let out = self.output.as_ref()?;
        let (w, h) = (out.width, out.height);
        let unpadded = (w as usize) * 8;
        let padded = unpadded.div_ceil(256) * 256;
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("blender-tex-readback"),
            size: (padded * h as usize) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("blender-tex-rb") });
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &out.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &buf,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded as u32),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        self.queue.submit(Some(enc.finish()));
        let (tx, rx) = std::sync::mpsc::channel();
        buf.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().ok()?.ok()?;
        let mut px = Vec::with_capacity(unpadded * h as usize);
        {
            let view = buf.slice(..).get_mapped_range().expect("a mapped readback");
            for y in 0..h as usize {
                px.extend_from_slice(&view[y * padded..y * padded + unpadded]);
            }
        }
        buf.unmap();
        Some((px, w, h))
    }

    /// Keep every Paint node's stroke layers in sync: right-sized for the
    /// output, stamps re-uploaded + pass re-recorded only when the stroke
    /// set changes (hash of the resampled stamp bytes).
    ///
    /// TWO layers per node — colour stamps and clone stamps — because they
    /// mean different things downstream. The clone layer only exists while
    /// the node has clone strokes, so a node that has never cloned costs
    /// exactly what it always did.
    ///
    /// The stamp set is FRAME-DEPENDENT now (a stroke's lifetime decides
    /// whether it is there), which is why this reads the clock. It is still
    /// a memoization: `stamps_for` returns identical bytes on every frame
    /// unless a per-frame stroke comes or goes, so an ordinary painted node
    /// re-records exactly never.
    fn update_paint_layers(&mut self, graph: &Graph) {
        let Some((w, h)) = self.output_size() else { return };
        let frame = self.paint_frame;
        let mut live = std::collections::HashSet::new();
        let mut live_source = std::collections::HashSet::new();
        let mut live_effect = std::collections::HashSet::new();
        for (id, kind) in &graph.nodes {
            let NodeKind::Paint(p) = kind else { continue };
            live.insert(*id);
            let stamps = crate::paint::stamps_for(p, w, h, frame, graph.proxy_scale());
            let sync = |layers: &mut HashMap<crate::graph::NodeId, crate::paint::PaintLayer>,
                            set: &[crate::paint::Stamp],
                            tag: u8,
                            mul: bool,
                            raster: &crate::paint::PaintRaster,
                            device: &wgpu::Device,
                            queue: &wgpu::Queue|
             -> bool {
                let hash = crate::diskcache::content_hash(&[
                    bytemuck::cast_slice(set),
                    &w.to_le_bytes(),
                    &h.to_le_bytes(),
                    &[tag],
                ]);
                let resize = layers
                    .get(id)
                    .is_none_or(|l| l.width != w || l.height != h || l.mul.is_some() != mul);
                if resize {
                    layers.insert(*id, raster.make_layer(device, w, h, mul));
                }
                let layer = layers.get_mut(id).unwrap();
                if layer.hash != hash {
                    raster.set_stamps(device, queue, layer, set, hash);
                }
                resize
            };
            if sync(
                &mut self.paint_layers,
                &stamps.color,
                0,
                p.has_blend(),
                &self.paint_raster,
                &self.device,
                &self.queue,
            ) {
                self.frame_bgs = None;
            }
            // The source and effect layers are allocated on a class's first
            // stroke and dropped with its last: `has_source` / `has_effect`
            // are what the shader was compiled against, so the layers and
            // the bindings cannot disagree.
            if p.has_source() {
                live_source.insert(*id);
                if sync(
                    &mut self.paint_source_layers,
                    &stamps.source,
                    1,
                    false,
                    &self.paint_raster,
                    &self.device,
                    &self.queue,
                ) {
                    self.frame_bgs = None;
                }
            }
            if p.has_effect() {
                live_effect.insert(*id);
                if sync(
                    &mut self.paint_effect_layers,
                    &stamps.effect,
                    2,
                    false,
                    &self.paint_raster,
                    &self.device,
                    &self.queue,
                ) {
                    self.frame_bgs = None;
                }
            }
        }
        let before = (
            self.paint_layers.len(),
            self.paint_source_layers.len(),
            self.paint_effect_layers.len(),
        );
        self.paint_layers.retain(|id, _| live.contains(id));
        self.paint_source_layers.retain(|id, _| live_source.contains(id));
        self.paint_effect_layers.retain(|id, _| live_effect.contains(id));
        let after = (
            self.paint_layers.len(),
            self.paint_source_layers.len(),
            self.paint_effect_layers.len(),
        );
        if after != before {
            self.frame_bgs = None;
        }
    }

    /// Keep one glyph atlas per Text / Burnin node, re-uploaded only when
    /// its words change (the atlas is cached by content, so asking each
    /// frame is a hash and a lookup). A font that will not load binds the
    /// dummy, which draws nothing.
    fn update_text_tables(&mut self, graph: &Graph) {
        let mut live = std::collections::HashSet::new();
        for (id, kind) in &graph.nodes {
            let raster = match kind {
                NodeKind::Burnin(p) => p.raster(),
                _ => continue,
            };
            let Some(r) = raster else {
                if self.text_tables.remove(id).is_some() {
                    self.frame_bgs = None;
                }
                continue;
            };
            live.insert(*id);
            if self.text_tables.get(id).is_some_and(|t| t.hash == r.hash) {
                continue;
            }
            let size = wgpu::Extent3d { width: r.width, height: r.height, depth_or_array_layers: 1 };
            let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("text-atlas"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                bytemuck::cast_slice(&r.texels),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(r.width * 2),
                    rows_per_image: Some(r.height),
                },
                size,
            );
            let view = texture.create_view(&Default::default());
            self.text_tables.insert(*id, TextTex { hash: r.hash, _texture: texture, view });
            self.frame_bgs = None;
        }
        let before = self.text_tables.len();
        self.text_tables.retain(|id, _| live.contains(id));
        if self.text_tables.len() != before {
            self.frame_bgs = None;
        }
    }

    /// Keep one layer per Text node at the node's format: lay the text out
    /// and animate it (CPU, cheap), and upload + mark the layer dirty only
    /// when the instances' bytes change, so a still title costs a hash a
    /// frame and an animated one one draw call.
    fn update_text_layers(&mut self, graph: &Graph) {
        let mut live = std::collections::HashSet::new();
        let proxy = graph.proxy_scale();
        for (id, kind) in &graph.nodes {
            let NodeKind::Text(p) = kind else { continue };
            let Some((w, h)) = graph.format_of(*id).or_else(|| self.output_size()) else { continue };
            let (w, h) = (w.max(1), h.max(1));
            live.insert(*id);
            if self.text_layers.get(id).is_none_or(|l| l.width != w || l.height != h) {
                self.text_layers.insert(*id, self.text_raster.make_layer(&self.device, w, h));
                self.frame_bgs = None;
            }
            let layer = self.text_layers.get_mut(id).unwrap();
            match p.build(proxy, self.clock.1) {
                Some(b) => {
                    if layer.hash != b.hash {
                        self.text_raster.set(&self.device, &self.queue, layer, &b.atlas, &b.instances, b.uniform, b.hash);
                        layer.quads = b.quads;
                    }
                }
                None => {
                    // The font will not load: an empty layer.
                    if layer.hash != 0 {
                        layer.hash = 0;
                        layer.quads.clear();
                        layer.dirty = true;
                    }
                }
            }
        }
        let before = self.text_layers.len();
        self.text_layers.retain(|id, _| live.contains(id));
        if self.text_layers.len() != before {
            self.frame_bgs = None;
        }
    }

    /// Keep one particle layer per Plasma node at the node's format. The
    /// particles are worked out in the vertex shader, so all this does is
    /// hand over the uniform; the layer re-draws only when it changed.
    fn update_plasma_layers(&mut self, graph: &Graph) {
        let mut live = std::collections::HashSet::new();
        let proxy = graph.proxy_scale();
        for (id, kind) in &graph.nodes {
            let NodeKind::Plasma(p) = kind else { continue };
            let Some((w, h)) = graph.format_of(*id).or_else(|| self.output_size()) else { continue };
            let (w, h) = (w.max(1), h.max(1));
            live.insert(*id);
            if self.plasma_layers.get(id).is_none_or(|l| l.width != w || l.height != h) {
                self.plasma_layers.insert(*id, self.plasma_raster.make_layer(&self.device, w, h));
                self.frame_bgs = None;
            }
            let layer = self.plasma_layers.get_mut(id).unwrap();
            self.plasma_raster.set(&self.queue, layer, &p.uniform(w, h, proxy), p.particles());
        }
        let before = self.plasma_layers.len();
        self.plasma_layers.retain(|id, _| live.contains(id));
        if self.plasma_layers.len() != before {
            self.frame_bgs = None;
        }
    }

    /// Keep one layer per Splat node at the node's format, its cloud on
    /// the GPU and its uniform current; the encode draws it when dirty.
    fn update_splat_layers(&mut self, graph: &Graph) {
        let mut live = std::collections::HashSet::new();
        for (id, kind) in &graph.nodes {
            let NodeKind::Splat(p) = kind else { continue };
            if p.depth_of.is_some() {
                continue;
            }
            let Some((w, h)) = graph.format_of(*id).or_else(|| self.output_size()) else { continue };
            let (w, h) = (w.max(1), h.max(1));
            live.insert(*id);
            if self.splat_layers.get(id).is_none_or(|l| l.width != w || l.height != h) {
                self.splat_layers.insert(*id, self.splat_raster.make_layer(&self.device, w, h));
                self.frame_bgs = None;
            }
            let layer = self.splat_layers.get_mut(id).unwrap();
            let cloud = match crate::splat::peek(&p.path) {
                Some((crate::splat::State::Ready(c), gen)) => Some((c, gen)),
                _ => None,
            };
            let uni = cloud.as_ref().map(|(c, _)| p.uniform(c, w, h)).unwrap_or([0; crate::splatgpu::UNIFORM_BYTES]);
            self.splat_raster.set(
                &self.device,
                &self.queue,
                layer,
                cloud.as_ref().map(|(c, g)| (&**c, *g)),
                &uni,
                p.depth_used,
            );
        }
        let before = self.splat_layers.len();
        self.splat_layers.retain(|id, _| live.contains(id));
        if self.splat_layers.len() != before {
            self.frame_bgs = None;
        }
    }

    /// A Splat node's layer, for tests: (dirty, times drawn, splats on the
    /// GPU).
    #[doc(hidden)]
    pub fn splat_layer_state(&self, id: crate::graph::NodeId) -> Option<(bool, u64, u32)> {
        self.splat_layers.get(&id).map(|l| (l.dirty, l.draws, l.splats()))
    }

    /// Keep one simulation + layer per Particles node at the node's format,
    /// and hand it the frame; the encode steps it there.
    fn update_particle_layers(&mut self, graph: &Graph) {
        let mut live = std::collections::HashSet::new();
        let proxy = graph.proxy_scale();
        for (id, kind) in &graph.nodes {
            let NodeKind::Particles(p) = kind else { continue };
            let Some((w, h)) = graph.format_of(*id).or_else(|| self.output_size()) else { continue };
            let (w, h) = (w.max(1), h.max(1));
            live.insert(*id);
            if self.particle_layers.get(id).is_none_or(|l| l.width != w || l.height != h) {
                self.particle_layers.insert(*id, self.particle_raster.make_layer(&self.device, w, h));
                self.frame_bgs = None;
            }
            let layer = self.particle_layers.get_mut(id).unwrap();
            let c = p.compile(proxy);
            self.particle_raster.set(&self.device, &self.queue, layer, &c, p.frame());
        }
        let before = self.particle_layers.len();
        self.particle_layers.retain(|id, _| live.contains(id));
        if self.particle_layers.len() != before {
            self.frame_bgs = None;
        }
    }

    /// A Particles node's layer, for tests: (dirty, steps the last encode
    /// ran, checkpoint frames).
    #[doc(hidden)]
    pub fn particle_layer_state(&self, id: crate::graph::NodeId) -> Option<(bool, u64, Vec<i64>)> {
        self.particle_layers.get(&id).map(|l| (l.dirty, l.last_steps, l.checkpoints()))
    }

    /// Is a Plasma node's layer waiting to be re-drawn? For tests.
    #[doc(hidden)]
    pub fn plasma_layer_dirty(&self, id: crate::graph::NodeId) -> Option<bool> {
        self.plasma_layers.get(&id).map(|l| l.dirty)
    }

    /// Each drawn glyph's quad of a Text node, canvas px — the viewer's
    /// box and a test's yardstick. None before its layer exists.
    pub fn text_quads(&self, id: crate::graph::NodeId) -> Option<&[[[f32; 2]; 4]]> {
        self.text_layers.get(&id).map(|l| l.quads.as_slice())
    }

    /// Is a Text node's layer waiting to be re-recorded? For tests.
    #[doc(hidden)]
    pub fn text_layer_dirty(&self, id: crate::graph::NodeId) -> Option<bool> {
        self.text_layers.get(&id).map(|l| l.dirty)
    }

    /// Keep one table texture per LUT node, re-uploaded only when the
    /// file's bytes change (`lut::load` is cached, so asking is free). A
    /// node whose file does not load binds the dummy and passes through.
    fn update_lut_tables(&mut self, graph: &Graph) {
        let mut live = std::collections::HashSet::new();
        for (id, kind) in &graph.nodes {
            let NodeKind::Lut(p) = kind else { continue };
            let Ok(lut) = p.status() else {
                if self.lut_tables.remove(id).is_some() {
                    self.frame_bgs = None;
                }
                continue;
            };
            live.insert(*id);
            if self.lut_tables.get(id).is_some_and(|t| t.hash == lut.hash) {
                continue;
            }
            let (lay, px) = lut.texture();
            // Float32 where it can be bound beside filtered sources; half
            // elsewhere (manifesto 3 — scale down, never refuse).
            let full = self.device.features().contains(wgpu::Features::FLOAT32_FILTERABLE);
            let (format, bytes, bpp): (_, Vec<u8>, u32) = if full {
                (wgpu::TextureFormat::Rgba32Float, bytemuck::cast_slice(&px).to_vec(), 16)
            } else {
                let h: Vec<half::f16> = px.iter().flatten().map(|v| half::f16::from_f32(*v)).collect();
                (wgpu::TextureFormat::Rgba16Float, bytemuck::cast_slice(&h).to_vec(), 8)
            };
            let size = wgpu::Extent3d { width: lay.width, height: lay.height, depth_or_array_layers: 1 };
            let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("lut-table"),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &bytes,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(lay.width * bpp),
                    rows_per_image: Some(lay.height),
                },
                size,
            );
            let view = texture.create_view(&Default::default());
            self.lut_tables.insert(*id, LutTex { hash: lut.hash, _texture: texture, view });
            self.frame_bgs = None;
        }
        let before = self.lut_tables.len();
        self.lut_tables.retain(|id, _| live.contains(id));
        if self.lut_tables.len() != before {
            self.frame_bgs = None;
        }
    }

    /// Keep one rasterised layer per 2dini node: evaluate the geometry when
    /// the node (or the Noise it reads) changed, upload when the drawing's
    /// bytes changed, and leave it to `encode_comp` to re-record. A static
    /// drawing therefore costs nothing per frame on either side.
    fn update_geo_layers(&mut self, graph: &Graph) {
        let Some((w, h)) = self.output_size() else { return };
        let mut live = std::collections::HashSet::new();
        for (id, kind) in &graph.nodes {
            let NodeKind::Geo(g) = kind else { continue };
            live.insert(*id);
            // The Noise on the field port, as the CPU evaluator reads it.
            // Left out of the key when nothing inside reads it, and its
            // phase held still when the field does not move, so scrubbing
            // a static comp never re-evaluates the drawing.
            let noise = if g.graph.has_field_in() {
                graph.input_of(*id, 0).and_then(|s| match graph.nodes.get(&s) {
                    Some(NodeKind::Noise(n)) => Some(n),
                    _ => None,
                })
            } else {
                None
            };
            let phase = |n: &crate::graph::NoiseParams| crate::geo::field_cx_for(n).phase;
            let resize =
                self.geo_layers.get(id).is_none_or(|l| l.width != w || l.height != h);
            if resize {
                let layer = self.geo_raster.make_layer(&self.device, w, h);
                self.geo_layers.insert(*id, layer);
                self.frame_bgs = None;
            }
            let layer = self.geo_layers.get_mut(id).unwrap();
            // Light + SDF effects: their own textures while the look is on,
            // the layer redrawn with its light target when they appear or
            // go. Their params are uniforms; `set_look` marks what reruns.
            if g.look.active() {
                let grid = crate::geolight::Grid::new(w, h, g.look.spacing);
                let stale = resize
                    || self.geo_lit_layers.get(id).is_none_or(|l| l.width != w || l.height != h || l.grid != grid);
                if stale {
                    let lit = self.geo_lit.make_layer(&self.device, &layer.view, w, h, grid);
                    self.geo_lit_layers.insert(*id, lit);
                    layer.dirty = true;
                    self.frame_bgs = None;
                }
                let lit = self.geo_lit_layers.get_mut(id).unwrap();
                self.geo_lit.set_look(&self.queue, lit, &g.look);
            } else if self.geo_lit_layers.remove(id).is_some() {
                layer.dirty = true;
                self.frame_bgs = None;
            }
            // Compared in place, field by field, before anything is cloned:
            // this runs every tick something else in the comp animates, and
            // a static drawing should not pay a deep copy for that. Canvas
            // layout and the id counter are left out — dragging a node
            // across the geo canvas is not an edit of the geometry.
            let same = layer.last_key.as_ref().is_some_and(|(lg, lf)| {
                lg.size == g.size
                    && lg.seed == g.seed
                    // The graph's own noise: its phase, 0 unless something
                    // inside moves — so a still drawing ignores the tick.
                    // A baked simulation plays per tick; nothing else does.
                    && crate::geo::geo_time_key(lg) == crate::geo::geo_time_key(g)
                    && lg.graph.out == g.graph.out
                    && lg.graph.view == g.graph.view
                    && lg.graph.edges == g.graph.edges
                    && lg.graph.wires == g.graph.wires
                    && lg.graph.nodes == g.graph.nodes
                    && match (lf, noise) {
                        (None, None) => true,
                        (Some(f), Some(n)) => {
                            f.scale == n.scale.max(1e-4)
                                && f.seed == n.seed
                                && f.phase == phase(n)
                                && f.graph == n.graph
                        }
                        _ => false,
                    }
            });
            if same {
                continue;
            }
            let field = noise.map(crate::geo::field_cx_for);
            let key = (g.clone(), field);
            let t0 = Instant::now();
            let cx = crate::geo::EvalCx {
                tick: g.time,
                field: key.1.as_ref(),
                seed: g.seed,
                aspect: h as f32 / w as f32,
                px_per_unit: w as f32,
            };
            let (geo, stats) = crate::geo::eval_stats(&g.graph, &cx);
            let dl = crate::geo::flatten(&geo, w, h);
            let hash = crate::geo::hash(&dl, w, h);
            layer.stats = stats;
            layer.eval_ms = t0.elapsed().as_secs_f32() * 1000.0;
            layer.raster_cost = dl.edge_tests();
            layer.geometry = geo;
            if layer.hash != hash {
                self.geo_raster.set_draw_list(&self.device, &self.queue, layer, &dl, hash);
            }
            layer.last_key = Some(key);
        }
        self.geo_lit_layers.retain(|id, _| live.contains(id));
        let before = self.geo_layers.len();
        self.geo_layers.retain(|id, _| live.contains(id));
        if self.geo_layers.len() != before {
            self.frame_bgs = None;
        }
    }

    /// Is a 2dini node's layer waiting to be re-recorded? None when the
    /// node has no layer. For tests and the panel.
    #[doc(hidden)]
    pub fn geo_layer_dirty(&self, id: crate::graph::NodeId) -> Option<bool> {
        self.geo_layers.get(&id).map(|l| l.dirty)
    }

    /// The geometry a 2dini node's layer was drawn from, for the viewer's
    /// vector HUD. None when the node has no layer yet.
    pub fn geo_geometry(&self, id: crate::graph::NodeId) -> Option<&crate::geo::Geometry> {
        self.geo_layers.get(&id).map(|l| &l.geometry)
    }

    /// What a 2dini node's last evaluation produced and cost, for its
    /// panel: (stats, evaluation ms).
    pub fn geo_stats(
        &self,
        id: crate::graph::NodeId,
    ) -> Option<(crate::geo::GeoStats, f32, u64)> {
        self.geo_layers.get(&id).map(|l| (l.stats, l.eval_ms, l.raster_cost))
    }

    /// Which frame a stroke's LIFETIME is judged against. Separate from the
    /// clock on purpose: the clock is what a user shader reports as
    /// `P.frame` and keeps counting through a loop, while this is the tick
    /// the graph was evaluated at — the frame whose pixels are on screen,
    /// which is the one a per-frame stroke belongs to.
    /// Keep a layer per Blender node — created empty on first sight (it
    /// samples as nothing until a frame lands) and dropped with the node.
    fn update_blender_layers(&mut self, graph: &Graph) {
        let mut live = std::collections::HashSet::new();
        for (id, kind) in &graph.nodes {
            if matches!(kind, NodeKind::Blender(_)) {
                live.insert(*id);
            }
        }
        let before = self.blender_layers.len();
        self.blender_layers.retain(|(id, _), _| live.contains(id));
        if self.blender_layers.len() != before {
            self.frame_bgs = None;
        }
    }

    /// The node's layer, right-sized and right-formatted for a `w` x `h`
    /// frame of ring format `fmt`; recreated (bind groups and the size
    /// slot with it) when either changes.
    fn blender_layer(&mut self, id: crate::graph::NodeId, pass: u8, w: u32, h: u32, fmt: u32) -> Option<&mut BlenderLayer> {
        let format = blender_format(fmt)?;
        if w == 0 || h == 0 {
            return None;
        }
        let key = (id, pass);
        let fresh = self
            .blender_layers
            .get(&key)
            .is_none_or(|l| l.width != w || l.height != h || l.format != format);
        if fresh {
            let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("blender-layer"),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            self.blender_layers.insert(
                key,
                BlenderLayer { texture, view, width: w, height: h, format, stamp: None },
            );
            self.frame_bgs = None;
            // The slot carries the layer's size: pack it again.
            self.params_hash = None;
        }
        self.blender_layers.get_mut(&key)
    }

    /// A frame from a Blender session by COPY: `pixels` is `w * h * bpp`
    /// bytes of ring format `fmt`, rows un-padded, bottom-up, in the
    /// node's declared space. One `write_texture`: the copy is staged by
    /// wgpu and lands ahead of the next submit, so the frame that uploads
    /// it samples it.
    #[allow(clippy::too_many_arguments)]
    pub fn blender_upload(
        &mut self,
        id: crate::graph::NodeId,
        pass: u8,
        w: u32,
        h: u32,
        fmt: u32,
        pixels: &[u8],
        stamp: BlenderStamp,
    ) {
        let Some(bpp) = crate::blender::shm::bpp(fmt) else { return };
        if pixels.len() != w as usize * h as usize * bpp {
            return;
        }
        // A GPU that cannot filter float32 textures gets the frame as
        // half: one conversion on the way in, so the node still works
        // there (manifesto 3 — scale down, never refuse).
        let halves: Vec<u8>;
        let (fmt, bpp, pixels) = if fmt == crate::blender::shm::FMT_RGBA32F
            && !self.device.features().contains(wgpu::Features::FLOAT32_FILTERABLE)
        {
            let f: &[f32] = bytemuck::cast_slice(pixels);
            let h16: Vec<half::f16> = f.iter().map(|v| half::f16::from_f32(*v)).collect();
            halves = bytemuck::cast_slice(&h16).to_vec();
            (crate::blender::shm::FMT_RGBA16F, 8usize, halves.as_slice())
        } else {
            (fmt, bpp, pixels)
        };
        let queue = self.queue.clone();
        let Some(layer) = self.blender_layer(id, pass, w, h, fmt) else { return };
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &layer.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(w * bpp as u32),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        layer.stamp = Some(stamp);
    }

    /// A frame from a Blender session by REFERENCE: the ring is on the
    /// GPU (`blender::gpuimport`) and the pixels sit at `offset` in `src`,
    /// `stride` bytes a row. One blit, submitted now so it lands ahead of
    /// this redraw's comp pass; `done` runs once the GPU has finished with
    /// the slot (the ack that lets Blender write it again).
    #[allow(clippy::too_many_arguments)]
    pub fn blender_copy(
        &mut self,
        id: crate::graph::NodeId,
        pass: u8,
        src: &wgpu::Buffer,
        offset: u64,
        stride: u32,
        w: u32,
        h: u32,
        fmt: u32,
        stamp: BlenderStamp,
        done: Box<dyn FnOnce() + Send + 'static>,
    ) -> bool {
        if stride % 256 != 0 || offset + stride as u64 * h as u64 > src.size() {
            done();
            return false;
        }
        let device = self.device.clone();
        let queue = self.queue.clone();
        let Some(layer) = self.blender_layer(id, pass, w, h, fmt) else {
            done();
            return false;
        };
        let mut enc = device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("blender-ring-blit") });
        enc.copy_buffer_to_texture(
            wgpu::TexelCopyBufferInfo {
                buffer: src,
                layout: wgpu::TexelCopyBufferLayout {
                    offset,
                    bytes_per_row: Some(stride),
                    rows_per_image: Some(h),
                },
            },
            wgpu::TexelCopyTextureInfo {
                texture: &layer.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        queue.submit([enc.finish()]);
        queue.on_submitted_work_done(done);
        layer.stamp = Some(stamp);
        true
    }

    /// A frame from a Blender session as a TEXTURE (an IOSurface shared
    /// with Blender, `blender::gpuimport::import_surfaces`): its top-left
    /// `w` x `h` is copied into the node's layer, GPU to GPU. `done` runs
    /// once the GPU is finished with the source.
    #[allow(clippy::too_many_arguments)]
    pub fn blender_copy_texture(
        &mut self,
        id: crate::graph::NodeId,
        pass: u8,
        src: &wgpu::Texture,
        w: u32,
        h: u32,
        fmt: u32,
        stamp: BlenderStamp,
        done: Box<dyn FnOnce() + Send + 'static>,
    ) -> bool {
        if w > src.width() || h > src.height() {
            done();
            return false;
        }
        let device = self.device.clone();
        let queue = self.queue.clone();
        let Some(layer) = self.blender_layer(id, pass, w, h, fmt) else {
            done();
            return false;
        };
        let mut enc = device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("blender-surface-copy") });
        enc.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: src,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: &layer.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        queue.submit([enc.finish()]);
        queue.on_submitted_work_done(done);
        layer.stamp = Some(stamp);
        true
    }

    /// What a Blender node's combined layer holds right now: `None` until
    /// a frame has landed.
    pub fn blender_stamp(&self, id: crate::graph::NodeId) -> Option<BlenderStamp> {
        self.blender_layers.get(&(id, 0)).and_then(|l| l.stamp)
    }

    /// The same for one pass.
    pub fn blender_pass_stamp(&self, id: crate::graph::NodeId, pass: u8) -> Option<BlenderStamp> {
        self.blender_layers.get(&(id, pass)).and_then(|l| l.stamp)
    }

    /// The combined layer's texture, for a readback (tests, the pixel
    /// readout).
    pub fn blender_texture(&self, id: crate::graph::NodeId) -> Option<&wgpu::Texture> {
        self.blender_layers.get(&(id, 0)).map(|l| &l.texture)
    }

    pub fn set_paint_frame(&mut self, frame: u64) {
        self.paint_frame = frame;
    }

    /// The frame the next `write_params` should tell user shaders about.
    /// Set it per tick; it never invalidates a pipeline.
    pub fn set_clock(&mut self, frame: f32, fps: f32) {
        self.clock = (frame, if fps > 0.0 { fps } else { 24.0 });
    }

    /// Upload current parameter values (uniform writes, never a recompile).
    /// Also syncs paint layers (cheap when strokes are unchanged).
    pub fn write_params(&mut self, graph: &Graph) {
        self.params_hash = self.current;
        self.update_paint_layers(graph);
        self.update_geo_layers(graph);
        self.update_lut_tables(graph);
        self.update_text_tables(graph);
        self.update_text_layers(graph);
        self.update_plasma_layers(graph);
        self.update_splat_layers(graph);
        self.update_particle_layers(graph);
        self.update_blender_layers(graph);
        let Some(compiled) = self.current.and_then(|h| self.pipelines.get(&h)) else {
            return;
        };
        let Some(out) = &self.output else { return };
        let mut params = codegen::pack_params(
            graph,
            &compiled.plan.layout,
            [out.width as f32, out.height as f32],
        );
        // A Blender layer's slot: its size over the canvas's, so a frame
        // drawn smaller (half-res while the eye moves) still fills it.
        for (id, kind) in &graph.nodes {
            let NodeKind::Blender(p) = kind else { continue };
            let Some(layer) = self.blender_layers.get(&p.layer_key(*id)) else { continue };
            if let Some(slot) = compiled.plan.layout.node_slot.get(id) {
                if layer.stamp.is_some() && slot * 4 + 1 < params.len() {
                    params[slot * 4] = layer.width as f32 / out.width as f32;
                    params[slot * 4 + 1] = layer.height as f32 / out.height as f32;
                }
            }
        }
        self.queue.write_buffer(&self.uniform, 0, bytemuck::cast_slice(&params));

        let (frame, fps) = self.clock;
        let res = [out.width as f32, out.height as f32];
        for (stage, kind, _, _) in &compiled.plan.blurs {
            let Some(bt) = self.blur_texes.get(stage) else { continue };
            if *kind == FilterKind::Node {
                // A node's own chain: its uniforms, per pass, from the live
                // params — a slider drag lands here and nowhere else.
                let def = graph.nodes.get(&stage.0).and_then(|k| k.stage_def());
                let (Some(rt), Some(def)) = (self.stage_rts.get(stage), def) else { continue };
                let cx = StageCx {
                    canvas: (out.width, out.height),
                    proxy_scale: graph.proxy_scale(),
                    frame,
                    fps,
                    passes: rt.program.passes.len(),
                };
                let wired = codegen::wired_of(graph, stage.0);
                let aux_fmt: Vec<(u32, u32)> = def
                    .aux_ports(crate::nodes::stage::Wired(&wired))
                    .into_iter()
                    .map(|p| {
                        graph
                            .input_of(stage.0, p)
                            .and_then(|src| graph.format_of(src))
                            .unwrap_or((out.width, out.height))
                    })
                    .chain(std::iter::once(graph.format_of(stage.0).unwrap_or((out.width, out.height))))
                    .collect();
                for (i, (pass, buf)) in rt.program.passes.iter().zip(&rt.uniforms).enumerate() {
                    let Some(buf) = buf else { continue };
                    let mut u = def.uniforms_aux(i, &cx, &aux_fmt);
                    u.resize(pass.uniform_floats.max(4), 0.0);
                    self.queue.write_buffer(buf, 0, bytemuck::cast_slice(&u));
                }
                continue;
            }
            match (kind, graph.nodes.get(&stage.0)) {
                (FilterKind::Shader { .. }, Some(NodeKind::Shader(p))) => {
                    // One buffer per pass: identical but for `pass_index`,
                    // which is what lets every pass share one module.
                    let (Some(def), Some(st)) =
                        (compiled.plan.shaders.get(stage), bt.shader.as_ref())
                    else {
                        continue;
                    };
                    for (j, buf) in st.params.iter().enumerate() {
                        let bytes = def.pack(
                            &p.values,
                            res,
                            frame / fps,
                            frame,
                            j,
                            graph.working.primaries.luma_weights(),
                        );
                        self.queue.write_buffer(buf, 0, &bytes);
                    }
                }
                (
                    FilterKind::Defocus { depth, modulate, kernel, kernel_aux, .. },
                    Some(NodeKind::Defocus(d)),
                ) => {
                    // DP: 4 vec4s. Every control here is a uniform — only
                    // `mode` (in the topology hash) ever recompiles.
                    let chan = match d.depth_channel {
                        crate::graph::DepthChannel::R => 0.0f32,
                        crate::graph::DepthChannel::G => 1.0,
                        crate::graph::DepthChannel::B => 2.0,
                        crate::graph::DepthChannel::A => 3.0,
                        crate::graph::DepthChannel::Luma => 4.0,
                    };
                    // The kernel's centre + radius in its texture: a Read's
                    // own texture is its format; an aux kernel sits at the
                    // canvas's top-left in ITS format, like any picture.
                    let read_kernel = kernel.and_then(|k| self.readers.get(k)).and_then(|t| t.as_ref());
                    let aux_kernel = *kernel_aux && bt.aux.len() == 1 + *depth as usize + *modulate as usize;
                    let kfmt = match read_kernel {
                        Some(t) => Some((t.info.width, t.info.height)),
                        None if aux_kernel => Some(
                            graph
                                .input_of(stage.0, codegen::DEFOCUS_KERNEL_INPUT)
                                .and_then(|k| graph.format_of(k))
                                .unwrap_or((out.width, out.height)),
                        ),
                        None => None,
                    };
                    let has_kernel = kfmt.is_some() as u32 as f32;
                    let (kw, kh) = kfmt.unwrap_or((1, 1));
                    let kgeo = [kw as f32 * 0.5, kh as f32 * 0.5, kw.min(kh) as f32 * 0.5, 0.0];
                    self.queue.write_buffer(
                        &bt.params,
                        0,
                        bytemuck::cast_slice(&[
                            // Radius is pixels: scale with the proxy.
                            d.radius.max(0.0) * graph.proxy_scale(),
                            // The map exists: a depth pass or a matte made it.
                            ((*depth || *modulate) && !bt.aux.is_empty()) as u32 as f32,
                            chan,
                            d.inverse_depth as u32 as f32,
                            d.focus,
                            d.f_stop,
                            d.focal_len,
                            d.filmback,
                            d.blades as f32,
                            d.rotation,
                            d.catseye.clamp(0.0, 1.0),
                            d.samples.max(1) as f32,
                            has_kernel,
                            d.unit,
                            d.live_focal_view() as u32 as f32,
                            d.edge.max(0.0) * graph.proxy_scale(),
                            d.bright_clamp.max(0.0),
                            (*depth && !bt.aux.is_empty()) as u32 as f32,
                            (*modulate && !bt.aux.is_empty()) as u32 as f32,
                            0.0,
                            kgeo[0],
                            kgeo[1],
                            kgeo[2],
                            kgeo[3],
                        ]),
                    );
                }
                (FilterKind::Loop, Some(NodeKind::LoopOut(p))) => {
                    let n = p.n.round().clamp(1.0, crate::graph::LOOP_MAX) as u32;
                    self.loop_counts.insert(*stage, n);
                    self.loop_stages.insert(p.pair, *stage);
                }
                _ => {}
            }
        }
    }

    /// True when the live pipeline has no params written for it: a topology
    /// compile landed since the last write, or `encode_comp` just built a
    /// stage chain (first sight, or a canvas change such as the proxy
    /// toggle) whose uniform buffers are fresh and therefore ZERO. Stage
    /// chains are built during encode, after the frame's `write_params`,
    /// so the caller must write again when this is set — writes are staged
    /// ahead of the submit and land before the recorded passes run. Left
    /// unwritten, a new Blur rendered sharp and a proxy toggle dropped every
    /// effect until the next edit (`--test-proxy`).
    pub fn params_stale(&self) -> bool {
        self.current.is_some() && self.current != self.params_hash
    }

    /// The pipeline for a pass module, compiled on first sight.
    fn ensure_pass_pipeline(&mut self, wgsl: &'static str) {
        self.pass_pipelines
            .entry(wgsl_hash(wgsl))
            .or_insert_with(|| make_compute_pipeline(&self.device, "stage-pass", wgsl, None));
    }

    /// Allocate a program's textures and uniforms, and make sure every pass
    /// it dispatches has a pipeline.
    fn build_stage_rt(&mut self, program: StageProgram, canvas: (u32, u32), hash: u64) -> StageRt {
        for pass in &program.passes {
            self.ensure_pass_pipeline(pass.wgsl);
        }
        let texes = program
            .textures
            .iter()
            .map(|t| {
                let (w, h) = t.dims(canvas);
                let format = match t.format {
                    TexFormat::Rgba16Float => wgpu::TextureFormat::Rgba16Float,
                    TexFormat::Rgba32Float => wgpu::TextureFormat::Rgba32Float,
                    TexFormat::R16Float => wgpu::TextureFormat::R16Float,
                };
                (t.name.clone(), Self::make_tex(&self.device, w, h, format, &t.name))
            })
            .collect();
        let uniforms = program
            .passes
            .iter()
            .map(|p| {
                (p.uniform_floats > 0).then(|| {
                    Self::make_uniform(&self.device, (p.uniform_floats * 4).max(16) as u64, p.label)
                })
            })
            .collect();
        StageRt { program, canvas, hash, texes, uniforms }
    }

    /// A stage's texture by the name its program uses: the subtree input,
    /// an aux input, or one it declared. An unknown name reads the dummy —
    /// a wrong picture rather than a crash.
    fn stage_tex<'a>(&'a self, rt: &'a StageRt, bt: &'a BlurTex, name: &str) -> &'a wgpu::TextureView {
        if name == "a" {
            return &bt.a.1;
        }
        if let Some(i) = name.strip_prefix("aux").and_then(|i| i.parse::<usize>().ok()) {
            return bt.aux.get(i).map(|t| &t.1).unwrap_or(&self.dummy.view);
        }
        rt.texes.get(name).map(|t| &t.1).unwrap_or(&self.dummy.view)
    }

    /// The size of a stage texture, for a dispatch over it.
    fn stage_size(&self, rt: &StageRt, name: &str, canvas: (u32, u32)) -> (u32, u32) {
        rt.program
            .textures
            .iter()
            .find(|t| t.name == name)
            .map(|t| t.dims(canvas))
            .unwrap_or(canvas)
    }

    fn source_view<'a>(&'a self, binding: &SourceBinding) -> &'a wgpu::TextureView {
        match binding {
            SourceBinding::Reader(s) => self
                .readers
                .get(*s)
                .and_then(|t| t.as_ref())
                .map(|t| &t.view)
                .unwrap_or(&self.dummy.view),
            SourceBinding::Blur(n) => {
                // Where the result lives depends on the pass chain: gaussian
                // ends V(B->A) = A; disc conv is A->B = B.
                let kind = self
                    .current
                    .and_then(|h| self.pipelines.get(&h))
                    .and_then(|c| c.plan.blurs.iter().find(|(id, _, _, _)| id == n))
                    .map(|(_, k, _, _)| *k);
                self.blur_texes
                    .get(n)
                    .map(|bt| match kind {
                        // A node's own chain says which texture it left the
                        // result in.
                        Some(FilterKind::Node) => self
                            .stage_rts
                            .get(n)
                            .map(|rt| self.stage_tex(rt, bt, &rt.program.result))
                            .unwrap_or(&self.dummy.view),
                        // A user shader's result is its LAST pass.
                        Some(FilterKind::Shader { .. }) => bt
                            .shader
                            .as_ref()
                            .and_then(|st| st.passes.last())
                            .map(|t| &t.1)
                            .unwrap_or(&self.dummy.view),
                        Some(FilterKind::Defocus { .. }) | Some(FilterKind::Loop) => {
                            bt.b.as_ref().map(|t| &t.1).unwrap_or(&self.dummy.view)
                        }
                        _ => &bt.a.1,
                    })
                    .unwrap_or(&self.dummy.view)
            }
            // Outside its own loop, a tap is the loop's finished result.
            SourceBinding::LoopPrev(pair) => self
                .loop_stages
                .get(pair)
                .and_then(|sid| self.blur_texes.get(sid))
                .and_then(|bt| bt.b.as_ref())
                .map(|t| &t.1)
                .unwrap_or(&self.dummy.view),
            SourceBinding::Paint(n) => self
                .paint_layers
                .get(n)
                .map(|l| &l.view)
                .unwrap_or(&self.dummy.view),
            SourceBinding::PaintMul(n) => self
                .paint_layers
                .get(n)
                .and_then(|l| l.mul.as_ref())
                .map(|m| &m.1)
                .unwrap_or(&self.dummy.view),
            SourceBinding::PaintSource(n) => self
                .paint_source_layers
                .get(n)
                .map(|l| &l.view)
                .unwrap_or(&self.dummy.view),
            SourceBinding::PaintEffect(n) => self
                .paint_effect_layers
                .get(n)
                .map(|l| &l.view)
                .unwrap_or(&self.dummy.view),
            SourceBinding::Lut(n) => {
                self.lut_tables.get(n).map(|t| &t.view).unwrap_or(&self.dummy.view)
            }
            SourceBinding::Text(n) => self
                .text_layers
                .get(n)
                .map(|l| &l.view)
                .or_else(|| self.text_tables.get(n).map(|t| &t.view))
                .unwrap_or(&self.dummy.view),
            SourceBinding::Plasma(n) => self.plasma_layers.get(n).map(|l| &l.view).unwrap_or(&self.dummy.view),
            SourceBinding::Splat(n, depth) => self
                .splat_layers
                .get(n)
                .map(|l| if *depth { &l.depth_view } else { &l.view })
                .unwrap_or(&self.dummy.view),
            SourceBinding::Particles(n) => {
                self.particle_layers.get(n).map(|l| &l.view).unwrap_or(&self.dummy.view)
            }
            SourceBinding::Geo(n) => self
                .geo_lit_layers
                .get(n)
                .map(|l| &l.out)
                .or_else(|| self.geo_layers.get(n).map(|l| &l.view))
                .unwrap_or(&self.dummy.view),
            // Nothing landed yet: the dummy, so an unconnected session is
            // black, not a stale texture from another node.
            SourceBinding::Blender(n, pass) => self
                .blender_layers
                .get(&(*n, *pass))
                .filter(|l| l.stamp.is_some())
                .map(|l| &l.view)
                .unwrap_or(&self.dummy.view),
        }
    }

    /// Bind group for one fused-style pass (shared uniform, sampler, sources,
    /// storage target).
    fn fused_bind_group(
        &self,
        pipeline: &wgpu::ComputePipeline,
        sources: &[SourceBinding],
        target: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        let mut entries = vec![
            wgpu::BindGroupEntry { binding: 0, resource: self.uniform.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&self.sampler),
            },
        ];
        for (i, src) in sources.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: 2 + i as u32,
                resource: wgpu::BindingResource::TextureView(self.source_view(src)),
            });
        }
        entries.push(wgpu::BindGroupEntry {
            binding: 2 + sources.len() as u32,
            resource: wgpu::BindingResource::TextureView(target),
        });
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fused-bg"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        })
    }

    /// A fused-layout bind group over EXPLICIT views and an explicit uniform
    /// buffer — what a user Shader pass needs, since its textures are stage
    /// intermediates rather than `SourceBinding`s and its params live in its
    /// own buffer rather than the shared one.
    fn views_bind_group(
        &self,
        pipeline: &wgpu::ComputePipeline,
        uniform: &wgpu::Buffer,
        views: &[&wgpu::TextureView],
        target: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        let mut entries = vec![
            wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&self.sampler),
            },
        ];
        for (i, v) in views.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: 2 + i as u32,
                resource: wgpu::BindingResource::TextureView(v),
            });
        }
        entries.push(wgpu::BindGroupEntry {
            binding: 2 + views.len() as u32,
            resource: wgpu::BindingResource::TextureView(target),
        });
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shader-pass-bg"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        })
    }


    /// Bind group for a filter pass with no sampler: uniform, N source
    /// textures, storage target.
    /// Bind groups for a texture list in `@binding` order, on `pipeline`'s
    /// layout.
    fn labelled_bind_group(
        &self,
        label: &str,
        pipeline: &wgpu::ComputePipeline,
        params: Option<&wgpu::Buffer>,
        views: &[&wgpu::TextureView],
    ) -> wgpu::BindGroup {
        let mut entries = Vec::new();
        if let Some(p) = params {
            entries.push(wgpu::BindGroupEntry { binding: 0, resource: p.as_entire_binding() });
        }
        let base = entries.len() as u32;
        for (i, v) in views.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: base + i as u32,
                resource: wgpu::BindingResource::TextureView(v),
            });
        }
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        })
    }

    /// A Defocus stage's dispatch set. With depth: the CoC map and its
    /// tile dilation (both modes), then Fast's three prefix sums or
    /// Accurate's one gather. Without: the map bindings take the dummies
    /// and `has_depth` keeps the passes off them.
    fn defocus_bind_groups(
        &self,
        bt: &BlurTex,
        mode: crate::graph::DefocusMode,
        depth_view: &wgpu::TextureView,
        mod_view: &wgpu::TextureView,
        kern_view: &wgpu::TextureView,
    ) -> DefocusBgs {
        let b = bt.b.as_ref().map(|t| &t.1).expect("stage b");
        let m = bt.maps.as_ref();
        let cmap = m.map(|m| &m.cmap.1).unwrap_or(&self.dummy_rg32.1);
        let tiles = m.map(|m| &m.tiles_v.1).unwrap_or(&self.dummy_rg32.1);
        let maps = m.map(|m| {
            [
                self.labelled_bind_group(
                    "defocus-coc-bg",
                    &self.defocus_coc_pipeline,
                    Some(&bt.params),
                    &[depth_view, mod_view, &m.cmap.1],
                ),
                self.labelled_bind_group(
                    "defocus-tile-bg",
                    &self.defocus_tile_pipeline,
                    None,
                    &[&m.cmap.1, &m.tiles.1],
                ),
                self.labelled_bind_group(
                    "defocus-tile-h-bg",
                    &self.defocus_tile_h_pipeline,
                    Some(&bt.params),
                    &[&m.tiles.1, &m.tiles_h.1],
                ),
                self.labelled_bind_group(
                    "defocus-tile-v-bg",
                    &self.defocus_tile_v_pipeline,
                    Some(&bt.params),
                    &[&m.tiles_h.1, &m.tiles_v.1],
                ),
            ]
        });
        match mode {
            crate::graph::DefocusMode::Fast => {
                let p = bt.p.as_ref().expect("defocus prefix texture");
                let pn = m.and_then(|m| m.pn.as_ref()).map(|t| &t.1).unwrap_or(&self.dummy_rgba32.1);
                let pw = m.and_then(|m| m.pw.as_ref()).map(|t| &t.1).unwrap_or(&self.dummy_r32.1);
                let prefix = self.labelled_bind_group(
                    "defocus-prefix-bg",
                    &self.defocus_prefix_pipeline,
                    Some(&bt.params),
                    &[&bt.a.1, cmap, &p.1, pn, pw],
                );
                let gather = self.labelled_bind_group(
                    "defocus-disc-bg",
                    &self.defocus_disc_pipeline,
                    Some(&bt.params),
                    &[&p.1, pn, pw, cmap, tiles, b],
                );
                DefocusBgs { maps, prefix: Some(prefix), gather, accurate: false }
            }
            crate::graph::DefocusMode::Accurate => {
                // Same shape as a glow pass: uniform, sampler, sources,
                // storage target. The sampler is what lets a bokeh plate
                // (and now every tap) be read at sub-texel offsets.
                let gather = self.sampled_filter_bind_group(
                    &self.defocus_bokeh_pipeline,
                    &bt.params,
                    &[&bt.a.1, cmap, tiles, kern_view],
                    b,
                );
                DefocusBgs { maps, prefix: None, gather, accurate: true }
            }
        }
    }

    /// Bind group for a filter pass with a sampler: uniform, sampler, N
    /// source textures, storage target.
    fn sampled_filter_bind_group(
        &self,
        pipeline: &wgpu::ComputePipeline,
        params: &wgpu::Buffer,
        srcs: &[&wgpu::TextureView],
        dst: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        let mut entries = vec![
            wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&self.sampler),
            },
        ];
        for (i, v) in srcs.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: 2 + i as u32,
                resource: wgpu::BindingResource::TextureView(v),
            });
        }
        entries.push(wgpu::BindGroupEntry {
            binding: 2 + srcs.len() as u32,
            resource: wgpu::BindingResource::TextureView(dst),
        });
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("glow-bg"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        })
    }

    /// Encode the whole comp: blur prepasses then the main fused pass.
    /// Returns whether any prepass ran (for the timer's resolve flags).
    /// Everything a dispatch needs, made current: dirty layers re-recorded,
    /// stage intermediates sized, bind groups built. `(topology, w, h)`, or
    /// None when there is nothing to render yet.
    fn prepare(&mut self, encoder: &mut wgpu::CommandEncoder) -> Option<(u64, u32, u32)> {
        let hash = self.current?;
        if !self.pipelines.contains_key(&hash) {
            return None;
        }
        let (out_w, out_h) = self.output_size()?;

        // Re-record any paint layer whose strokes changed (before anything
        // samples it this frame).
        let dirty_paints: Vec<crate::graph::NodeId> = self
            .paint_layers
            .iter()
            .filter(|(_, l)| l.dirty)
            .map(|(id, _)| *id)
            .collect();
        for id in dirty_paints {
            let layer = self.paint_layers.get_mut(&id).unwrap();
            self.paint_raster.encode(encoder, layer, crate::paint::LayerBlend::Over);
        }
        let dirty_sources: Vec<crate::graph::NodeId> = self
            .paint_source_layers
            .iter()
            .filter(|(_, l)| l.dirty)
            .map(|(id, _)| *id)
            .collect();
        for id in dirty_sources {
            let layer = self.paint_source_layers.get_mut(&id).unwrap();
            self.paint_raster.encode(encoder, layer, crate::paint::LayerBlend::Over);
        }
        // Amounts build up: the effect layer records additively.
        let dirty_effects: Vec<crate::graph::NodeId> = self
            .paint_effect_layers
            .iter()
            .filter(|(_, l)| l.dirty)
            .map(|(id, _)| *id)
            .collect();
        for id in dirty_effects {
            let layer = self.paint_effect_layers.get_mut(&id).unwrap();
            self.paint_raster.encode(encoder, layer, crate::paint::LayerBlend::Add);
        }
        // Geometry layers likewise: re-record the ones whose drawing
        // changed, before anything samples them this frame.
        let dirty_geo: Vec<crate::graph::NodeId> = self
            .geo_layers
            .iter()
            .filter(|(_, l)| l.dirty)
            .map(|(id, _)| *id)
            .collect();
        for id in dirty_geo {
            let layer = self.geo_layers.get_mut(&id).unwrap();
            let light = self.geo_lit_layers.get_mut(&id).map(|l| {
                l.need = crate::geolight::Need::All;
                &l.light
            });
            self.geo_raster.encode(encoder, layer, light);
        }
        // Text layers whose glyphs moved.
        let dirty_text: Vec<crate::graph::NodeId> =
            self.text_layers.iter().filter(|(_, l)| l.dirty).map(|(id, _)| *id).collect();
        for id in dirty_text {
            let layer = self.text_layers.get_mut(&id).unwrap();
            self.text_raster.encode(encoder, layer);
        }
        // Plasma layers whose uniform moved (the clock is in it).
        let dirty_plasma: Vec<crate::graph::NodeId> =
            self.plasma_layers.iter().filter(|(_, l)| l.dirty).map(|(id, _)| *id).collect();
        for id in dirty_plasma {
            let layer = self.plasma_layers.get_mut(&id).unwrap();
            self.plasma_raster.encode(encoder, layer);
        }
        // Splat layers whose view moved — only those this plan samples: a
        // capture of millions is not drawn for a node nobody is looking at.
        if !self.splat_layers.is_empty() {
            let plan = &self.pipelines[&hash].plan;
            let used: std::collections::HashSet<crate::graph::NodeId> = plan
                .main
                .sources
                .iter()
                .chain(plan.blurs.iter().flat_map(|(_, _, s, aux)| {
                    s.sources.iter().chain(aux.iter().flat_map(|a| a.sources.iter()))
                }))
                .filter_map(|b| match b {
                    SourceBinding::Splat(n, _) => Some(*n),
                    _ => None,
                })
                .collect();
            for (id, layer) in self.splat_layers.iter_mut() {
                if layer.dirty && used.contains(id) {
                    self.splat_raster.encode(encoder, layer);
                }
            }
        }
        // Particle layers whose frame or look moved: step, then draw.
        let dirty_particles: Vec<crate::graph::NodeId> =
            self.particle_layers.iter().filter(|(_, l)| l.dirty).map(|(id, _)| *id).collect();
        for id in dirty_particles {
            let layer = self.particle_layers.get_mut(&id).unwrap();
            self.particle_raster.encode(&self.device, encoder, layer);
        }
        // Then their light and effects, as far as anything changed.
        for lit in self.geo_lit_layers.values_mut() {
            self.geo_lit.encode(encoder, lit);
        }

        // Make sure every blur node in the plan has right-sized intermediates.
        let blur_nodes: Vec<(codegen::StageId, FilterKind)> = self.pipelines[&hash]
            .plan
            .blurs
            .iter()
            .map(|(n, k, _, _)| (*n, *k))
            .collect();
        // Uniform size for each user Shader stage, read out of the plan
        // before the allocation loop takes `&mut self.blur_texes`.
        let shader_ubytes: HashMap<codegen::StageId, (u64, usize)> = self.pipelines[&hash]
            .plan
            .shaders
            .iter()
            .map(|(k, d)| (*k, (d.uniform_bytes(), d.passes)))
            .collect();
        let aux_counts: Vec<usize> =
            self.pipelines[&hash].plan.blurs.iter().map(|(_, _, _, aux)| aux.len()).collect();
        for (i, (n, kind)) in blur_nodes.iter().enumerate() {
            // Fast-mode defocus needs the prefix texture; the hand-built
            // kinds write into `b`; a node's own chain declares its scratch
            // itself.
            let wants_prefix = matches!(
                kind,
                FilterKind::Defocus { mode: crate::graph::DefocusMode::Fast, .. }
            );
            let wants_b = matches!(kind, FilterKind::Defocus { .. } | FilterKind::Loop);
            // A depth-wired Defocus derives its maps; Fast mode also sums
            // the near layer.
            let wants_maps = matches!(kind, FilterKind::Defocus { depth: true, .. })
                || matches!(kind, FilterKind::Defocus { modulate: true, .. });
            let n_aux = aux_counts[i];
            let n_passes = match kind {
                FilterKind::Shader { passes, .. } => *passes,
                _ => 0,
            };
            let needs = match self.blur_texes.get(n) {
                Some(bt) => {
                    bt.width != out_w
                        || bt.height != out_h
                        || (wants_prefix && bt.p.is_none())
                        || (wants_maps != bt.maps.is_some())
                        || (wants_maps && wants_prefix && bt.maps.as_ref().is_some_and(|m| m.pn.is_none()))
                        || (wants_b && bt.b.is_none())
                        || bt.aux.len() != n_aux
                        || bt.shader.as_ref().map_or(0, |s| s.passes.len()) != n_passes
                }
                None => true,
            };
            if needs {
                let a = Self::make_storage_tex(&self.device, out_w, out_h, "stage-a");
                let b = wants_b.then(|| Self::make_storage_tex(&self.device, out_w, out_h, "stage-b"));
                let p = wants_prefix.then(|| {
                    Self::make_tex(
                        &self.device,
                        out_w,
                        out_h,
                        wgpu::TextureFormat::Rgba32Float,
                        "defocus-prefix",
                    )
                });
                let aux: Vec<_> = (0..n_aux)
                    .map(|_| Self::make_storage_tex(&self.device, out_w, out_h, "stage-aux"))
                    .collect();
                let maps = wants_maps.then(|| {
                    use wgpu::TextureFormat as F;
                    let (tw, th) = (out_w.div_ceil(8), out_h.div_ceil(8));
                    let tile = |l| Self::make_tex(&self.device, tw, th, F::Rg32Float, l);
                    DefocusMaps {
                        cmap: Self::make_tex(&self.device, out_w, out_h, F::Rg32Float, "defocus-coc"),
                        tiles: tile("defocus-tiles"),
                        tiles_h: tile("defocus-tiles-h"),
                        tiles_v: tile("defocus-tiles-v"),
                        pn: wants_prefix.then(|| {
                            Self::make_tex(&self.device, out_w, out_h, F::Rgba32Float, "defocus-near")
                        }),
                        pw: wants_prefix.then(|| {
                            Self::make_tex(&self.device, out_w, out_h, F::R32Float, "defocus-near-w")
                        }),
                        tile_dims: (tw, th),
                    }
                });
                // Each user pass keeps its own result for the whole chain,
                // because pass 4 may sample pass 1.
                let shader = (n_passes > 0).then(|| {
                    let (ubytes, _) = shader_ubytes.get(n).copied().unwrap_or((16, 0));
                    ShaderTex {
                        passes: (0..n_passes)
                            .map(|_| {
                                Self::make_storage_tex(&self.device, out_w, out_h, "shader-pass")
                            })
                            .collect(),
                        params: (0..n_passes)
                            .map(|_| {
                                Self::make_uniform(&self.device, ubytes.max(16), "shader-params")
                            })
                            .collect(),
                    }
                });
                self.blur_texes.insert(
                    *n,
                    BlurTex {
                        a,
                        b,
                        p,
                        maps,
                        aux,
                        shader,
                        params: Self::make_uniform(&self.device, 96, "stage-params"),
                        width: out_w,
                        height: out_h,
                    },
                );
                self.frame_bgs = None;
                // Fresh uniform buffers hold zeros: whoever wrote params
                // this frame wrote them into the buffers this replaced.
                self.params_hash = None;
            }
        }
        // A node's own chain: textures and uniforms per its program, rebuilt
        // when the canvas or the plan changes — never on a param edit.
        let programs: Vec<(codegen::StageId, StageProgram)> = {
            let plan = &self.pipelines[&hash].plan;
            plan.blurs
                .iter()
                .zip(plan.stage_nodes.iter().zip(&plan.stage_wired))
                .filter(|((_, k, _, _), _)| *k == FilterKind::Node)
                .filter_map(|((n, _, _, _), (node, wired))| {
                    let current = self
                        .stage_rts
                        .get(n)
                        .is_some_and(|rt| rt.hash == hash && rt.canvas == (out_w, out_h));
                    if current {
                        return None;
                    }
                    node.stage_def().map(|d| {
                        let w = crate::nodes::stage::Wired(wired);
                        // A matte runs on ONE channel where the device can
                        // write one: a quarter of the arithmetic and bytes.
                        let one = node.output_kind() == crate::graph::PortKind::Mask
                            && crate::nodes::stage::one_channel();
                        let p = one
                            .then(|| d.program_one_channel((out_w, out_h), w))
                            .flatten()
                            .unwrap_or_else(|| d.program((out_w, out_h), w));
                        (*n, p)
                    })
                })
                .collect()
        };
        for (n, program) in programs {
            let rt = self.build_stage_rt(program, (out_w, out_h), hash);
            self.stage_rts.insert(n, rt);
            self.frame_bgs = None;
            // Same: a rebuilt chain's uniforms are zero until written.
            self.params_hash = None;
        }
        self.stage_rts.retain(|n, _| blur_nodes.iter().any(|(m, _)| m == n));
        // The hand-built chains' scratch (Defocus, Loop, Shader) follows the
        // plan the same way. It was only ever inserted: every Defocus added
        // and removed kept its canvas-sized textures — at 6K two RGBA16F
        // storage textures and an RGBA32F prefix, ~1.2 GB of VRAM each time
        // — until an allocation failed (the user's "crashes when I add a
        // Defocus", on a 24 GB card).
        let before = self.blur_texes.len();
        self.blur_texes.retain(|n, _| blur_nodes.iter().any(|(m, _)| m == n));
        if self.blur_texes.len() != before {
            // Bind groups hold views of what just went; the memory is only
            // released once they are dropped too.
            self.frame_bgs = None;
        }

        // Loop dispatch sets ride the same lifecycle: rebuilt whenever the
        // intermediates were (the bind groups hold a/b/aux views).
        if self.frame_bgs.is_none() {
            self.loop_bgs.clear();
            let compiled = &self.pipelines[&hash];
            for (i, (sid, kind, shader, _)) in compiled.plan.blurs.iter().enumerate() {
                if *kind != FilterKind::Loop {
                    continue;
                }
                let Some(bt) = self.blur_texes.get(sid) else { continue };
                let Some(seed) = bt.aux.first() else { continue };
                let pl = &compiled.blur_pipelines[i];
                let over = |tap: &wgpu::TextureView, out: &wgpu::TextureView| {
                    let mut entries = vec![
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.uniform.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                    ];
                    for (k, src) in shader.sources.iter().enumerate() {
                        let view = match src {
                            SourceBinding::LoopPrev(_) => tap,
                            other => self.source_view(other),
                        };
                        entries.push(wgpu::BindGroupEntry {
                            binding: 2 + k as u32,
                            resource: wgpu::BindingResource::TextureView(view),
                        });
                    }
                    entries.push(wgpu::BindGroupEntry {
                        binding: 2 + shader.sources.len() as u32,
                        resource: wgpu::BindingResource::TextureView(out),
                    });
                    self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("loop-bg"),
                        layout: &pl.get_bind_group_layout(0),
                        entries: &entries,
                    })
                };
                self.loop_bgs.insert(
                    *sid,
                    LoopBgs {
                        seed_to_a: over(&seed.1, &bt.a.1),
                        seed_to_b: over(&seed.1, bt.b.as_ref().map(|t| &t.1).expect("stage b")),
                        ab: over(&bt.a.1, bt.b.as_ref().map(|t| &t.1).expect("stage b")),
                        ba: over(bt.b.as_ref().map(|t| &t.1).expect("stage b"), &bt.a.1),
                    },
                );
            }
        }

        if self.frame_bgs.is_none() {
            let compiled = &self.pipelines[&hash];
            let out_view = &self.output.as_ref().unwrap().view;
            let stages = compiled
                .plan
                .blurs
                .iter()
                .zip(&compiled.blur_pipelines)
                .zip(&compiled.aux_pipelines)
                .zip(&compiled.shader_pipelines)
                .map(
                    |((((n, kind, shader, aux_shader), pipeline), aux_pipeline), user_pipes)| {
                    let bt = &self.blur_texes[n];
                    // Aux subtrees resolve further image ports into their own
                    // textures (Defocus's depth; a Shader node's inputs 1..).
                    // All three halves have to line up — a plan/texture
                    // mismatch means that aux pass simply does not run.
                    let aux: Vec<_> = aux_shader
                        .iter()
                        .zip(aux_pipeline)
                        .zip(&bt.aux)
                        .map(|((sh, pl), tex)| self.fused_bind_group(pl, &sh.sources, &tex.1))
                        .collect();
                    let depth_view =
                        bt.aux.first().map(|t| &t.1).unwrap_or(&self.dummy.view);
                    // A user shader's passes: every pass sees every input and
                    // every result texture, with its OWN result slot bound to
                    // the dummy (a texture cannot be sampled and written by
                    // the same pass). Constant binding count is what lets all
                    // passes share one module and one layout.
                    let user = match (kind, &bt.shader) {
                        (FilterKind::Shader { inputs, passes }, Some(st))
                            if user_pipes.len() == *passes
                                && st.passes.len() == *passes =>
                        {
                            Some(
                                (0..*passes)
                                    .map(|i| {
                                        let mut views: Vec<&wgpu::TextureView> =
                                            Vec::with_capacity(inputs + passes);
                                        views.push(&bt.a.1);
                                        for k in 1..*inputs {
                                            views.push(
                                                bt.aux
                                                    .get(k - 1)
                                                    .map(|t| &t.1)
                                                    .unwrap_or(&self.dummy.view),
                                            );
                                        }
                                        for j in 0..*passes {
                                            views.push(if j == i {
                                                &self.dummy.view
                                            } else {
                                                &st.passes[j].1
                                            });
                                        }
                                        self.views_bind_group(
                                            &user_pipes[i],
                                            &st.params[i],
                                            &views,
                                            &st.passes[i].1,
                                        )
                                    })
                                    .collect::<Vec<_>>(),
                            )
                        }
                        _ => None,
                    };
                    match kind {
                        // The loop's real dispatch set lives in `loop_bgs`
                        // (rebuilt below); the generic subtree pass is
                        // SKIPPED for it in the encode loop.
                        FilterKind::Node => {
                            let rt = &self.stage_rts[n];
                            let passes = rt
                                .program
                                .passes
                                .iter()
                                .enumerate()
                                .map(|(i, pass)| {
                                    let pl = &self.pass_pipelines[&wgsl_hash(pass.wgsl)];
                                    let entries: Vec<wgpu::BindGroupEntry> = pass
                                        .binds
                                        .iter()
                                        .enumerate()
                                        .map(|(b, bind)| wgpu::BindGroupEntry {
                                            binding: b as u32,
                                            resource: match bind {
                                                Bind::Uniform => rt.uniforms[i]
                                                    .as_ref()
                                                    .expect("a pass that binds a uniform declares its floats")
                                                    .as_entire_binding(),
                                                Bind::Sampler => {
                                                    wgpu::BindingResource::Sampler(&self.sampler)
                                                }
                                                Bind::Tex(name) | Bind::Storage(name) => {
                                                    wgpu::BindingResource::TextureView(
                                                        self.stage_tex(rt, bt, name),
                                                    )
                                                }
                                            },
                                        })
                                        .collect();
                                    self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                                        label: Some(pass.label),
                                        layout: &pl.get_bind_group_layout(0),
                                        entries: &entries,
                                    })
                                })
                                .collect();
                            StageBindGroups {
                                aux,
                                shader: None,
                                subtree: self.fused_bind_group(pipeline, &shader.sources, &bt.a.1),
                                defocus: None,
                                passes,
                            }
                        }
                        FilterKind::Loop => StageBindGroups {
                            aux,
                            shader: None,
                            passes: Vec::new(),
                            subtree: self.fused_bind_group(pipeline, &shader.sources, &bt.a.1),
                            defocus: None,
                        },
                        FilterKind::Shader { .. } => StageBindGroups {
                            aux,
                            shader: user,
                            passes: Vec::new(),
                            subtree: self.fused_bind_group(pipeline, &shader.sources, &bt.a.1),
                            defocus: None,
                        },
                        FilterKind::Defocus { mode, modulate, kernel, kernel_aux, depth } => {
                            // Aux order: depth, the matte, the kernel.
                            let kern_view = kernel
                                .and_then(|k| self.readers.get(k))
                                .and_then(|t| t.as_ref())
                                .map(|t| &t.view)
                                .or_else(|| {
                                    kernel_aux
                                        .then(|| bt.aux.get(*depth as usize + *modulate as usize))
                                        .flatten()
                                        .map(|t| &t.1)
                                })
                                .unwrap_or(&self.dummy.view);
                            // Unwired, the depth view stands in for the matte
                            // and `has_mod` keeps the pass off it.
                            let mod_view = if *modulate {
                                bt.aux.get(*depth as usize).map(|t| &t.1).unwrap_or(depth_view)
                            } else {
                                depth_view
                            };
                            let defocus =
                                self.defocus_bind_groups(bt, *mode, depth_view, mod_view, kern_view);
                            StageBindGroups {
                                aux,
                                shader: None,
                                passes: Vec::new(),
                                subtree: self.fused_bind_group(pipeline, &shader.sources, &bt.a.1),
                                defocus: Some(defocus),
                            }
                        }
                    }
                },
                )
                .collect();
            let main = self.fused_bind_group(
                &compiled.main_pipeline,
                &compiled.plan.main.sources,
                out_view,
            );
            self.frame_bgs = Some(FrameBindGroups { main, stages });
        }
        Some((hash, out_w, out_h))
    }

    /// Record stage `i` of the live plan: its aux subtrees, its subtree (or
    /// a Loop's iterations), then its own passes. `first`/`last_stage` place
    /// the timer's prepass window.
    fn encode_stage(
        &self,
        hash: u64,
        i: usize,
        encoder: &mut wgpu::CommandEncoder,
        timer: &GpuTimer,
        first: bool,
        last_stage: bool,
    ) {
        let compiled = &self.pipelines[&hash];
        let bgs = self.frame_bgs.as_ref().unwrap();
        let stage = &bgs.stages[i];
        let Some((out_w, out_h)) = self.output_size() else { return };
        let (gx, gy) = (out_w.div_ceil(8), out_h.div_ceil(8));
        // Aux subtrees (Defocus's depth branch; a Shader node's inputs
        // 1..) resolve BEFORE the stage's own subtree, so every input
        // texture is ready together.
        if let Some(pls) = compiled.aux_pipelines.get(i) {
            for (bg, pl) in stage.aux.iter().zip(pls) {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("filter-aux-subtree"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(pl);
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(gx, gy, 1);
            }
        }
        // A Loop's "subtree" shader IS its iteration body: instead of the
        // one generic dispatch into `a`, run it n times — seed-primed
        // first pass, then ping-pong, phased so the LAST write lands in
        // `b` (where downstream reads every loop result).
        if let Some((sid, FilterKind::Loop)) = compiled.plan.blurs.get(i).map(|(s, k, _, _)| (s, k)) {
            if let Some(lb) = self.loop_bgs.get(sid) {
                let n = self.loop_counts.get(sid).copied().unwrap_or(1).max(1);
                for k in 0..n {
                    let remaining_after = n - 1 - k;
                    let to_b = remaining_after % 2 == 0;
                    let bg = match (k == 0, to_b) {
                        (true, true) => &lb.seed_to_b,
                        (true, false) => &lb.seed_to_a,
                        (false, true) => &lb.ab,
                        (false, false) => &lb.ba,
                    };
                    let mut pass =
                        encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("loop-body"),
                            timestamp_writes: if first && k == 0 {
                                timer.prepass_begin_writes()
                            } else {
                                None
                            },
                        });
                    pass.set_pipeline(&compiled.blur_pipelines[i]);
                    pass.set_bind_group(0, bg, &[]);
                    pass.dispatch_workgroups(gx, gy, 1);
                }
            }
        } else {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("filter-subtree"),
                timestamp_writes: if first { timer.prepass_begin_writes() } else { None },
            });
            pass.set_pipeline(&compiled.blur_pipelines[i]);
            pass.set_bind_group(0, &stage.subtree, &[]);
            pass.dispatch_workgroups(gx, gy, 1);
        }
        // User shader: passes in order, each reading the results of the
        // ones before it.
        if let (Some(bgs), Some(pls)) = (&stage.shader, compiled.shader_pipelines.get(i)) {
            let last = bgs.len().saturating_sub(1);
            for (j, (bg, pl)) in bgs.iter().zip(pls).enumerate() {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("user-shader-pass"),
                    timestamp_writes: if last_stage && j == last {
                        timer.prepass_end_writes()
                    } else {
                        None
                    },
                });
                pass.set_pipeline(pl);
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(gx, gy, 1);
            }
        }
        if let Some(d) = &stage.defocus {
            let run = |encoder: &mut wgpu::CommandEncoder, label, pl, bg, x: u32, y: u32, last: bool| {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(label),
                    timestamp_writes: if last { timer.prepass_end_writes() } else { None },
                });
                pass.set_pipeline(pl);
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(x, y, 1);
            };
            if let Some(m) = &d.maps {
                let (tw, th) = compiled
                    .plan
                    .blurs
                    .get(i)
                    .and_then(|(sid, ..)| self.blur_texes.get(sid))
                    .and_then(|bt| bt.maps.as_ref())
                    .map(|m| m.tile_dims)
                    .unwrap_or((out_w.div_ceil(8), out_h.div_ceil(8)));
                let (tx, ty) = (tw.div_ceil(8), th.div_ceil(8));
                run(encoder, "defocus-coc", &self.defocus_coc_pipeline, &m[0], gx, gy, false);
                run(encoder, "defocus-tile", &self.defocus_tile_pipeline, &m[1], tx, ty, false);
                run(encoder, "defocus-tile-h", &self.defocus_tile_h_pipeline, &m[2], tx, ty, false);
                run(encoder, "defocus-tile-v", &self.defocus_tile_v_pipeline, &m[3], tx, ty, false);
            }
            if let Some(bg) = &d.prefix {
                run(encoder, "defocus-prefix", &self.defocus_prefix_pipeline, bg, out_h.div_ceil(64), 1, false);
            }
            let pl = if d.accurate { &self.defocus_bokeh_pipeline } else { &self.defocus_disc_pipeline };
            run(encoder, "defocus-gather", pl, &d.gather, gx, gy, last_stage);
        }
        // A node's own chain, in declared order; the timer's prepass
        // window closes on the last pass of the last stage.
        if let Some((sid, FilterKind::Node)) = compiled.plan.blurs.get(i).map(|(s, k, _, _)| (s, k)) {
            if let Some(rt) = self.stage_rts.get(sid) {
                let n_passes = rt.program.passes.len();
                for (j, (pass, bg)) in rt.program.passes.iter().zip(&stage.passes).enumerate() {
                    let pl = &self.pass_pipelines[&wgsl_hash(pass.wgsl)];
                    let (dx, dy) = match &pass.dispatch {
                        Dispatch::Pixels(t) => {
                            let (w, h) = self.stage_size(rt, t, (out_w, out_h));
                            (w.div_ceil(8), h.div_ceil(8))
                        }
                        Dispatch::Rows(t) => {
                            let (_, h) = self.stage_size(rt, t, (out_w, out_h));
                            (h.div_ceil(64), 1)
                        }
                        Dispatch::Groups(x, y) => (*x, *y),
                    };
                    let last = last_stage && j + 1 == n_passes;
                    let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some(pass.label),
                        timestamp_writes: if last { timer.prepass_end_writes() } else { None },
                    });
                    cpass.set_pipeline(pl);
                    cpass.set_bind_group(0, bg, &[]);
                    cpass.dispatch_workgroups(dx, dy, 1);
                }
            }
        }
    }

    pub fn encode_comp(&mut self, encoder: &mut wgpu::CommandEncoder, timer: &GpuTimer) -> bool {
        let Some((hash, out_w, out_h)) = self.prepare(encoder) else { return false };
        // Every stage the picture needs, minus what a resolved MotionBlur
        // already holds: the stages feeding ONLY it rendered once per sample
        // and must not run an (N+1)th time at the frame.
        let run = self.stages_to_run(hash, None);
        for (k, &i) in run.iter().enumerate() {
            self.encode_stage(hash, i, encoder, timer, k == 0, k + 1 == run.len());
        }
        self.motion_done.clear();
        let compiled = &self.pipelines[&hash];
        let bgs = self.frame_bgs.as_ref().unwrap();
        let (gx, gy) = (out_w.div_ceil(8), out_h.div_ceil(8));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fused-main"),
            timestamp_writes: timer.fused_pass_writes(),
        });
        pass.set_pipeline(&compiled.main_pipeline);
        pass.set_bind_group(0, &bgs.main, &[]);
        pass.dispatch_workgroups(gx, gy, 1);
        drop(pass);


        !run.is_empty()
    }

    /// Stage indices to dispatch, in plan order. `Some(m)`: stage `m` and
    /// everything upstream of it (one motion-blur sample). `None`: what the
    /// main pass needs. Either way a resolved MotionBlur is neither run nor
    /// walked through — its `a` is final for this frame.
    fn stages_to_run(&self, hash: u64, only: Option<usize>) -> Vec<usize> {
        let plan = &self.pipelines[&hash].plan;
        let index: HashMap<codegen::StageId, usize> =
            plan.blurs.iter().enumerate().map(|(i, (s, ..))| (*s, i)).collect();
        let deps_of = |sources: &[SourceBinding], own: Option<usize>, out: &mut Vec<usize>| {
            for src in sources {
                let sid = match src {
                    SourceBinding::Blur(s) => Some(*s),
                    // Outside its own loop a tap is the loop's result.
                    SourceBinding::LoopPrev(pair) => self.loop_stages.get(pair).copied(),
                    _ => None,
                };
                if let Some(j) = sid.and_then(|s| index.get(&s).copied()) {
                    if Some(j) != own {
                        out.push(j);
                    }
                }
            }
        };
        let deps = |j: usize, out: &mut Vec<usize>| {
            let (_, _, sh, aux) = &plan.blurs[j];
            deps_of(&sh.sources, Some(j), out);
            for a in aux {
                deps_of(&a.sources, Some(j), out);
            }
        };
        let mut stack = Vec::new();
        match only {
            Some(m) => stack.push(m),
            None => {
                deps_of(&plan.main.sources, None, &mut stack);
                // A stage no other stage reads is a root in its own right
                // (the plan made it for a reason): keep it.
                let mut fed = vec![false; plan.blurs.len()];
                for j in 0..plan.blurs.len() {
                    let mut d = Vec::new();
                    deps(j, &mut d);
                    for k in d {
                        fed[k] = true;
                    }
                }
                stack.extend((0..plan.blurs.len()).filter(|j| !fed[*j]));
            }
        }
        let mut want = vec![false; plan.blurs.len()];
        while let Some(j) = stack.pop() {
            if want[j] || self.motion_done.contains(&plan.blurs[j].0) {
                continue;
            }
            want[j] = true;
            deps(j, &mut stack);
        }
        (0..want.len()).filter(|j| want[*j]).collect()
    }

    /// The MotionBlur stages in the live plan, upstream first — the order
    /// their samples must run in, so a nested one is resolved (and held at
    /// the frame) before the one below it samples through it.
    /// The shutter itself is read from the LIVE graph by the caller — the
    /// plan's copy of the node is as it was at compile time, and turning the
    /// shutter never recompiles.
    pub fn motion_blur_stages(&self) -> Vec<codegen::StageId> {
        let Some(c) = self.current.and_then(|h| self.pipelines.get(&h)) else {
            return Vec::new();
        };
        c.plan
            .blurs
            .iter()
            .zip(&c.plan.stage_nodes)
            .filter(|(_, k)| matches!(k, NodeKind::MotionBlur(_)))
            .map(|((sid, ..), _)| *sid)
            .collect()
    }

    /// One motion-blur sample of stage `sid`: its upstream stages and its
    /// own subtree, under whatever params were written last, added into its
    /// accumulator at `weight` (`first` seeds it, so no clear pass).
    ///
    /// ONE SUBMIT PER SAMPLE: `queue.write_buffer` is staged ahead of the
    /// next submit, not ordered against recorded commands, so N samples in
    /// one encoder would all read the LAST sample's params — invisibly,
    /// since every sample still renders. The caller writes params, encodes
    /// this, checks `params_stale`, submits; `resolve_motion` after the last.
    pub fn encode_motion_sample(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        timer: &GpuTimer,
        sid: codegen::StageId,
        weight: f32,
        first: bool,
    ) -> bool {
        let Some((hash, w, h)) = self.prepare(encoder) else { return false };
        let Some(m) = self.pipelines[&hash].plan.blurs.iter().position(|(s, ..)| *s == sid)
        else {
            return false;
        };
        for i in self.stages_to_run(hash, Some(m)) {
            self.encode_stage(hash, i, encoder, timer, false, false);
        }
        let stale = self.accum_bufs.get(&sid).is_none_or(|(_, bw, bh)| (*bw, *bh) != (w, h));
        if stale {
            let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("mblur-accum"),
                size: (w as u64) * (h as u64) * 16,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            self.accum_bufs.insert(sid, (buf, w, h));
        }
        self.queue.write_buffer(
            &self.accum_params,
            0,
            bytemuck::cast_slice(&[weight, if first { 1.0 } else { 0.0 }, w as f32, h as f32]),
        );
        let (Some((buf, ..)), Some(bt)) = (self.accum_bufs.get(&sid), self.blur_texes.get(&sid))
        else {
            return false;
        };
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mblur-accum-bg"),
            layout: &self.accum_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.accum_params.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&bt.a.1),
                },
                wgpu::BindGroupEntry { binding: 2, resource: buf.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("mblur-accum"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.accum_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(w.div_ceil(8), h.div_ceil(8), 1);
        true
    }

    /// The accumulated average becomes stage `sid`'s `a`, and the stage is
    /// marked done until the next `encode_comp`, which samples it like any
    /// stage result and runs nothing that only fed it.
    pub fn resolve_motion(&mut self, encoder: &mut wgpu::CommandEncoder, sid: codegen::StageId) {
        let (Some((buf, w, h)), Some(bt)) = (self.accum_bufs.get(&sid), self.blur_texes.get(&sid))
        else {
            return;
        };
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mblur-resolve-bg"),
            layout: &self.accum_resolve_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.accum_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buf.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&bt.a.1),
                },
            ],
        });
        let (w, h) = (*w, *h);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("mblur-resolve"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.accum_resolve_pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(w.div_ceil(8), h.div_ceil(8), 1);
        drop(pass);
        self.motion_done.insert(sid);
    }
}

