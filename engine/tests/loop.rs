//! Headless integration for the loop stage (LoopIn -> body -> LoopOut):
//! iterated multiplication has an exact closed form, an iterated translate
//! moves by exactly n steps, and the iteration count must change results
//! without ever recompiling. Self-skips without a GPU (CI).

use engine::compositor::Compositor;
use engine::decode::ImageInfo;
use engine::graph::{Graph, LoopParams, MultParams, NodeKind, TransformParams};
use engine::timing::GpuTimer;
use half::f16;

const W: u32 = 128;
const H: u32 = 32;

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP loop test: no adapter ({e})");
            None
        }
    }
}

/// Constant plate `v` in every channel, alpha 1.
fn upload_flat(ctx: &engine::GpuContext, comp: &mut Compositor, slot: usize, v: f32) {
    comp.ensure_reader(slot, ImageInfo { width: W, height: H });
    let px: Vec<[f16; 4]> = (0..W * H)
        .map(|_| [f16::from_f32(v), f16::from_f32(v), f16::from_f32(v), f16::from_f32(1.0)])
        .collect();
    ctx.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &comp.reader_texture(slot).unwrap().texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&px),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(W * 8),
            rows_per_image: Some(H),
        },
        wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
    );
}

/// A step edge: 1.0 left of `edge` pixels, 0.0 right of it.
fn upload_edge(ctx: &engine::GpuContext, comp: &mut Compositor, slot: usize, edge: u32) {
    comp.ensure_reader(slot, ImageInfo { width: W, height: H });
    let mut px: Vec<[f16; 4]> = Vec::with_capacity((W * H) as usize);
    for _ in 0..H {
        for x in 0..W {
            let v = if x < edge { 1.0 } else { 0.0 };
            px.push([f16::from_f32(v), f16::from_f32(v), f16::from_f32(v), f16::from_f32(1.0)]);
        }
    }
    ctx.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &comp.reader_texture(slot).unwrap().texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&px),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(W * 8),
            rows_per_image: Some(H),
        },
        wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
    );
}

/// Middle row, red channel.
fn row(ctx: &engine::GpuContext, comp: &mut Compositor, graph: &Graph) -> Vec<f32> {
    assert!(comp.set_graph(graph), "fused shader failed to compile");
    let timer = GpuTimer::new(ctx);
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
    let off = (padded * (H / 2)) as usize;
    let halves: &[f16] = bytemuck::cast_slice(&data[off..off + (W * 8) as usize]);
    let out = (0..W as usize).map(|i| halves[i * 4].to_f32()).collect();
    drop(data);
    staging.unmap();
    out
}

/// Read(1) seed -> LoopOut(7); body LoopIn(5) -> `mid` node (6) -> LoopOut.
fn loop_graph(mid: NodeKind, n: f32) -> Graph {
    let mut g = Graph::default();
    g.nodes.insert(1, NodeKind::read(0, ""));
    g.nodes.insert(5, NodeKind::LoopIn);
    g.nodes.insert(6, mid);
    g.edges.insert((6, 0), 5);
    g.nodes.insert(7, NodeKind::LoopOut(LoopParams { pair: 5, n }));
    g.edges.insert((7, 0), 6);
    g.edges.insert((7, 1), 1);
    g.view = Some(7);
    g
}

fn mult(v: f32) -> NodeKind {
    NodeKind::Mult(MultParams { value: [v; 3], ..Default::default() })
}

/// Iterated multiply has an exact closed form: n applications of x0.5 on a
/// plate of 1.0 is 2^-n. Nails the pass count AND the seed priming.
#[test]
fn iterated_mult_hits_the_closed_form() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload_flat(&ctx, &mut comp, 0, 1.0);
    for n in [1u32, 4, 7] {
        let r = row(&ctx, &mut comp, &loop_graph(mult(0.5), n as f32));
        let want = 0.5f32.powi(n as i32);
        assert!(
            (r[(W / 2) as usize] - want).abs() < want * 0.02 + 1e-4,
            "n={n}: got {} want {want}",
            r[(W / 2) as usize]
        );
    }
}

/// An iterated translate moves the edge n steps — geometry accumulates
/// through the ping-pong exactly like colour does.
#[test]
fn iterated_translate_moves_n_steps() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload_edge(&ctx, &mut comp, 0, 32);
    let step = NodeKind::Transform(TransformParams {
        matte: false,
        translate: [10.0, 0.0],
        rotate: 0.0,
        scale: [1.0, 1.0],
        center: [0.5, 0.5],
        scale_gang: false,
        pin: engine::graph::UNIT_QUAD,
        mod_blend: Default::default(),
        shear: [0.0, 0.0],
        taper: 0.0,
        taper_angle: 90.0,
    });
    let r = row(&ctx, &mut comp, &loop_graph(step.clone(), 3.0));
    // The FALLING edge: shifted-in pixels are transparent black, so the
    // leading margin is dark too — find where the ones END.
    let edge = r.iter().rposition(|v| *v > 0.5).map(|i| i + 1).unwrap_or(0);
    assert!(
        (edge as i64 - 62).abs() <= 1,
        "edge at 32 + 3x10 = 62, got {edge}"
    );
    // And the leading margin really did shift in as emptiness.
    assert!(r[5] < 0.5 && r[40] > 0.5, "content occupies [30, 62): {:?}", &r[28..34]);
}

/// Changing n changes the picture and NEVER the pipeline: same topology
/// hash, different result.
#[test]
fn n_is_a_free_knob() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload_flat(&ctx, &mut comp, 0, 1.0);
    let g4 = loop_graph(mult(0.5), 4.0);
    let g6 = loop_graph(mult(0.5), 6.0);
    assert_eq!(g4.topology_hash(), g6.topology_hash());
    let a = row(&ctx, &mut comp, &g4)[(W / 2) as usize];
    let b = row(&ctx, &mut comp, &g6)[(W / 2) as usize];
    assert!((a - 0.0625).abs() < 0.002, "n=4 -> 1/16, got {a}");
    assert!((b - 0.015625).abs() < 0.001, "n=6 -> 1/64, got {b}");
}

/// An unwired seed primes with transparent black: a pure feedback body
/// (tap x0.5) stays black — nothing invents energy.
#[test]
fn an_unwired_seed_is_black() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    let mut g = loop_graph(mult(0.5), 5.0);
    g.edges.remove(&(7, 1));
    let r = row(&ctx, &mut comp, &g);
    assert!(r.iter().all(|v| v.abs() < 1e-5), "no seed, no signal");
}
