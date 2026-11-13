//! The Blur node on the GPU: an impulse spreads into a gaussian whose mass
//! is conserved, and sigma 0 is the identity. The stage runs through the
//! generic `StageDef` executor, so this also pins that machinery.

use engine::compositor::Compositor;
use engine::decode::ImageInfo;
use engine::graph::{BlurParams, Graph, NodeKind};
use engine::timing::GpuTimer;
use half::f16;

const W: u32 = 64;
const H: u32 = 8;

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP blur test: no adapter ({e})");
            None
        }
    }
}

/// One bright column at x = 32, on every row.
fn upload_impulse(ctx: &engine::GpuContext, comp: &mut Compositor, slot: usize) {
    comp.ensure_reader(slot, ImageInfo { width: W, height: H });
    let mut px: Vec<[f16; 4]> = Vec::with_capacity((W * H) as usize);
    for _ in 0..H {
        for x in 0..W {
            let v = f16::from_f32(if x == 32 { 1.0 } else { 0.0 });
            px.push([v, v, v, f16::from_f32(1.0)]);
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
    let off = (padded * 4) as usize;
    let halves: &[f16] = bytemuck::cast_slice(&data[off..off + (W * 8) as usize]);
    let out = (0..W as usize).map(|i| halves[i * 4].to_f32()).collect();
    drop(data);
    staging.unmap();
    out
}

fn graph_with(size: f32) -> Graph {
    let mut g = Graph::default();
    g.nodes.insert(0, NodeKind::read(0, ""));
    g.nodes.insert(1, NodeKind::Blur(BlurParams { size, matte: false }));
    g.edges.insert((1, 0), 0);
    g.view = Some(1);
    g
}

#[test]
fn an_impulse_spreads_and_keeps_its_mass() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 1);
    comp.ensure_output(W, H);
    upload_impulse(&ctx, &mut comp, 0);

    let sharp = row(&ctx, &mut comp, &graph_with(0.0));
    assert!((sharp[32] - 1.0).abs() < 0.01, "sigma 0 is the identity, got {}", sharp[32]);
    assert!(sharp[30].abs() < 0.01 && sharp[34].abs() < 0.01, "sigma 0 must not spread");

    let soft = row(&ctx, &mut comp, &graph_with(2.0));
    let peak = soft[32];
    assert!(peak < 0.35 && peak > 0.12, "sigma 2 peak should be ~0.2, got {peak}");
    assert!((soft[30] - soft[34]).abs() < 0.01, "a gaussian is symmetric");
    assert!(soft[30] > 0.05 && soft[30] < peak, "neighbours rise but stay below the peak");
    let mass: f32 = soft.iter().sum();
    assert!((mass - 1.0).abs() < 0.08, "a normalised kernel keeps the mass, got {mass}");
}
