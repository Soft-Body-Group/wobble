//! Headless integration for the Shader node: user-authored WGSL through the
//! real Compositor, on the GPU, reading back actual pixels.
//!
//! What these lock down, in order of what would hurt most to break:
//! * a user shader RUNS, and its declared params reach it as uniforms;
//! * it can sample its input somewhere other than the current pixel — the
//!   whole reason this node exists rather than a fused plugin expression;
//! * multi-pass works, and pass 2 really reads what pass 1 wrote (a chain
//!   that silently read black would still look plausible in one frame);
//! * a SECOND image input is resolved to its own texture and is samplable;
//! * a shader that does not compile leaves the node a passthrough instead of
//!   taking the comp down;
//! * editing the text recompiles and moving a param does NOT.
//!
//! Self-skips without a GPU (CI).

use engine::codegen::{FilterKind, SourceBinding};
use engine::compositor::Compositor;
use engine::decode::ImageInfo;
use engine::graph::{Graph, NodeKind, ShaderParams};
use engine::timing::GpuTimer;
use half::f16;

const W: u32 = 64;
const H: u32 = 64;

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP usershader test: no adapter ({e})");
            None
        }
    }
}

fn upload(ctx: &engine::GpuContext, comp: &mut Compositor, slot: usize, px: &[[f16; 4]]) {
    comp.ensure_reader(slot, ImageInfo { width: W, height: H });
    ctx.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &comp.reader_texture(slot).unwrap().texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(px),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(W * 8),
            rows_per_image: Some(H),
        },
        wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
    );
}

fn render(ctx: &engine::GpuContext, comp: &mut Compositor, graph: &Graph) -> Vec<[f32; 4]> {
    assert!(comp.set_graph(graph), "fused shader failed to compile");
    let timer = GpuTimer::new(ctx);
    // Warm-up encode: stage textures are allocated inside encode_comp, and
    // write_params only reaches stages that already exist.
    let mut warm = ctx.device.create_command_encoder(&Default::default());
    comp.encode_comp(&mut warm, &timer);
    ctx.queue.submit([warm.finish()]);
    comp.write_params(graph);
    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    comp.encode_comp(&mut encoder, &timer);
    let padded = (W * 8).div_ceil(256) * 256;
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (padded * H) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: comp.output_texture().unwrap(),
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(H),
            },
        },
        wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
    );
    ctx.queue.submit([encoder.finish()]);
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    let data = slice.get_mapped_range().expect("a mapped buffer");
    let mut out = Vec::with_capacity((W * H) as usize);
    for y in 0..H as usize {
        let row = &data[y * padded as usize..][..(W * 8) as usize];
        let halves: &[f16] = bytemuck::cast_slice(row);
        for x in 0..W as usize {
            let p = &halves[x * 4..][..4];
            out.push([p[0].to_f32(), p[1].to_f32(), p[2].to_f32(), p[3].to_f32()]);
        }
    }
    drop(data);
    staging.unmap();
    out
}

fn at(px: &[[f32; 4]], x: u32, y: u32) -> [f32; 4] {
    px[(y * W + x) as usize]
}

fn flat(c: [f32; 4]) -> Vec<[f16; 4]> {
    vec![c.map(f16::from_f32); (W * H) as usize]
}

/// A vertical split: left half `a`, right half `b`. Sampling the "wrong"
/// half is then unmistakable in the readback.
fn split(a: [f32; 4], b: [f32; 4]) -> Vec<[f16; 4]> {
    let mut px = Vec::with_capacity((W * H) as usize);
    for _ in 0..H {
        for x in 0..W {
            px.push(if x < W / 2 { a.map(f16::from_f32) } else { b.map(f16::from_f32) });
        }
    }
    px
}

fn shader(src: &str) -> NodeKind {
    let mut p = ShaderParams::default();
    p.set_source(src.to_string());
    NodeKind::Shader(p)
}

/// reader 0 -> Shader -> view, plus optional reader 1 on input 1.
fn graph_of(src: &str, second_input: bool) -> Graph {
    let mut g = Graph::default();
    g.nodes.insert(1, NodeKind::read(0, ""));
    g.nodes.insert(2, shader(src));
    g.edges.insert((2, 0), 1);
    if second_input {
        g.nodes.insert(3, NodeKind::read(1, ""));
        g.edges.insert((2, 1), 3);
    }
    g.view = Some(2);
    g
}

const MULT: &str = r#"//! input: front
//! param: float gain = 2.0 [0, 8]
fn pass1(uv: vec2f, frag: vec2f) -> vec4f {
    let c = tex(front, uv);
    return vec4f(c.rgb * P.gain, c.a);
}
"#;

#[test]
fn a_user_shader_runs_and_its_params_reach_it() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 2);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp, 0, &flat([0.25, 0.5, 0.125, 1.0]));

    let mut g = graph_of(MULT, false);
    let px = render(&ctx, &mut comp, &g);
    let c = at(&px, W / 2, H / 2);
    assert!((c[0] - 0.5).abs() < 1e-2, "gain 2 on 0.25 -> {c:?}");
    assert!((c[1] - 1.0).abs() < 1e-2, "{c:?}");

    // Moving a param is a UNIFORM write: same topology hash, new pixels.
    let before = g.topology_hash();
    let NodeKind::Shader(p) = g.nodes.get_mut(&2).unwrap() else { unreachable!() };
    p.values.insert("gain".into(), [4.0, 0.0, 0.0, 0.0]);
    assert_eq!(g.topology_hash(), before, "a param moved the topology hash");
    let px = render(&ctx, &mut comp, &g);
    assert!((at(&px, W / 2, H / 2)[0] - 1.0).abs() < 1e-2);
}

#[test]
fn it_can_sample_the_input_somewhere_else() {
    // The thing a fused plugin expression cannot do: read a DIFFERENT pixel.
    // This shader mirrors horizontally, so the left half must come back
    // holding what the right half of the plate had.
    const MIRROR: &str = r#"//! input: front
fn pass1(uv: vec2f, frag: vec2f) -> vec4f {
    return tex(front, vec2f(1.0 - uv.x, uv.y));
}
"#;
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 2);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp, 0, &split([1.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]));

    let px = render(&ctx, &mut comp, &graph_of(MIRROR, false));
    let left = at(&px, 4, H / 2);
    let right = at(&px, W - 5, H / 2);
    assert!(left[2] > 0.8 && left[0] < 0.2, "left should hold the blue half: {left:?}");
    assert!(right[0] > 0.8 && right[2] < 0.2, "right should hold the red half: {right:?}");
}

#[test]
fn pass_two_reads_what_pass_one_wrote() {
    // Pass 1 writes a known constant that has nothing to do with the plate;
    // pass 2 returns `result1` scaled. If the chain were reading black (or
    // the plate) instead, the number would not be this one.
    const CHAIN: &str = r#"//! input: front
//! passes: 2
fn pass1(uv: vec2f, frag: vec2f) -> vec4f {
    return vec4f(0.25, 0.5, 0.75, 1.0);
}
fn pass2(uv: vec2f, frag: vec2f) -> vec4f {
    return tex(result1, uv) * 2.0;
}
"#;
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 2);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp, 0, &flat([0.0, 0.0, 0.0, 1.0]));

    let g = graph_of(CHAIN, false);
    assert!(matches!(
        engine::codegen::generate(&g).blurs[0].1,
        FilterKind::Shader { passes: 2, .. }
    ));
    let px = render(&ctx, &mut comp, &g);
    let c = at(&px, W / 2, H / 2);
    assert!((c[0] - 0.5).abs() < 1e-2, "{c:?}");
    assert!((c[1] - 1.0).abs() < 1e-2, "{c:?}");
    assert!((c[2] - 1.5).abs() < 1e-2, "{c:?}");
}

#[test]
fn a_second_input_gets_its_own_texture() {
    const TWO: &str = r#"//! input: front
//! input: back
fn pass1(uv: vec2f, frag: vec2f) -> vec4f {
    return tex(front, uv) + tex(back, uv);
}
"#;
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 2);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp, 0, &flat([0.25, 0.0, 0.0, 1.0]));
    upload(&ctx, &mut comp, 1, &flat([0.0, 0.5, 0.0, 1.0]));

    let g = graph_of(TWO, true);
    let plan = engine::codegen::generate(&g);
    assert_eq!(plan.blurs.len(), 1);
    assert_eq!(plan.blurs[0].3.len(), 1, "input 1 needs its own aux subtree pass");

    let px = render(&ctx, &mut comp, &g);
    let c = at(&px, W / 2, H / 2);
    assert!((c[0] - 0.25).abs() < 1e-2 && (c[1] - 0.5).abs() < 1e-2, "{c:?}");
}

#[test]
fn a_broken_shader_passes_through_instead_of_failing_the_comp() {
    const BROKEN: &str = r#"//! input: front
fn pass1(uv: vec2f, frag: vec2f) -> vec4f { return no_such_function(uv); }
"#;
    let g = graph_of(BROKEN, false);
    let plan = engine::codegen::generate(&g);
    assert!(plan.blurs.is_empty(), "a broken shader must not get a stage");
    // The main pass reads the READ directly: the node is a passthrough.
    assert_eq!(plan.main.sources, vec![SourceBinding::Reader(0)]);

    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 2);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp, 0, &flat([0.3, 0.6, 0.9, 1.0]));
    let px = render(&ctx, &mut comp, &g);
    let c = at(&px, W / 2, H / 2);
    assert!((c[0] - 0.3).abs() < 1e-2 && (c[2] - 0.9).abs() < 1e-2, "{c:?}");
}

#[test]
fn editing_the_text_recompiles() {
    let mut g = graph_of(MULT, false);
    let before = g.topology_hash();
    let NodeKind::Shader(p) = g.nodes.get_mut(&2).unwrap() else { unreachable!() };
    p.set_source(format!("{MULT}// a comment"));
    assert_ne!(g.topology_hash(), before);
}

#[test]
fn declared_ports_survive_a_shader_that_does_not_compile() {
    // Ports come from the PRAGMAS, which still parse — otherwise a typo
    // mid-edit would drop the node's wires on the floor.
    let k = shader("//! input: front\n//! input: back\nfn pass1(uv: vec2f, frag: vec2f) -> vec4f { return bad(uv); }");
    assert_eq!(k.input_count(), 2);
    assert_eq!(k.input_label(1), "back");
}

#[test]
fn the_generator_case_needs_no_input_at_all() {
    const GEN: &str = r#"//! name: Ramp
fn pass1(uv: vec2f, frag: vec2f) -> vec4f { return vec4f(uv.x, uv.y, 0.0, 1.0); }
"#;
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 2);
    comp.ensure_output(W, H);
    let mut g = Graph::default();
    g.nodes.insert(2, shader(GEN));
    g.view = Some(2);
    let px = render(&ctx, &mut comp, &g);
    let l = at(&px, 2, H / 2);
    let r = at(&px, W - 3, H / 2);
    assert!(r[0] > l[0] + 0.8, "x ramp: {l:?} -> {r:?}");
}

/// The app compiles topology on a WORKER thread (`set_graph_async`) and only
/// falls back to the blocking path on a view hop. A stage that compiles fine
/// synchronously but never lands asynchronously would look, in the app, like
/// a node that simply never comes up — with nothing in the log.
#[test]
fn a_shader_stage_lands_through_the_async_compile_path() {
    const TWO: &str = r#"//! input: front
//! passes: 2
//! param: float gain = 2.0 [0, 4]
fn pass1(uv: vec2f, frag: vec2f) -> vec4f { return tex(front, uv) * P.gain; }
fn pass2(uv: vec2f, frag: vec2f) -> vec4f { return tex(result1, uv); }
"#;
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 2);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp, 0, &flat([0.25, 0.25, 0.25, 1.0]));
    let g = graph_of(TWO, false);
    for _ in 0..600 {
        if comp.set_graph_async(&g) {
            return;
        }
        let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    panic!("async compile never landed for a 2-pass shader");
}
