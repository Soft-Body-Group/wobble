//! Headless integration for Median: a speck vanishes, a step edge stays
//! exactly where it was, on both the exact (small radius) and the separable
//! (large radius) path. Self-skips without a GPU (CI).

use engine::compositor::Compositor;
use engine::decode::ImageInfo;
use engine::graph::{Graph, MedianParams, NodeKind};
use engine::timing::GpuTimer;
use half::f16;

const W: u32 = 64;
const H: u32 = 8;

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP median test: no adapter ({e})");
            None
        }
    }
}

/// rgb 1 left of x = 32, 0 right of it; a dark speck at (10, 4) and a
/// bright one at (50, 4) on row 4, the row the readback takes.
fn plate(x: u32, y: u32) -> f32 {
    match (x, y) {
        (10, 4) => 0.0,
        (50, 4) => 5.0,
        _ if x < 32 => 1.0,
        _ => 0.0,
    }
}

fn upload(ctx: &engine::GpuContext, comp: &mut Compositor, slot: usize) {
    comp.ensure_reader(slot, ImageInfo { width: W, height: H });
    let mut px: Vec<[f16; 4]> = Vec::with_capacity((W * H) as usize);
    for y in 0..H {
        for x in 0..W {
            let v = f16::from_f32(plate(x, y));
            px.push([v, v, v, f16::ONE]);
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

/// One row of the output's red channel.
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
    // Row 4, to stay away from any edge effects at the top.
    let off = (padded * 4) as usize;
    let halves: &[f16] = bytemuck::cast_slice(&data[off..off + (W * 8) as usize]);
    let out = (0..W as usize).map(|i| halves[i * 4].to_f32()).collect();
    drop(data);
    staging.unmap();
    out
}

fn graph_with(radius: f32) -> Graph {
    let mut g = Graph::default();
    g.nodes.insert(0, NodeKind::read(0, ""));
    g.nodes.insert(1, NodeKind::Median(MedianParams { radius, matte: false }));
    g.edges.insert((1, 0), 0);
    g.view = Some(1);
    g
}

fn setup() -> Option<(engine::GpuContext, Compositor)> {
    let ctx = ctx()?;
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp, 0);
    Some((ctx, comp))
}

#[test]
fn specks_vanish_and_the_edge_stays_put() {
    let Some((ctx, mut comp)) = setup() else { return };
    // 1 and 3 take the exact 2D path, 5 and 12 the separable one.
    for radius in [1.0f32, 3.0, 5.0, 12.0] {
        let got = row(&ctx, &mut comp, &graph_with(radius));
        assert!((got[10] - 1.0).abs() < 1e-3, "r {radius}: dark speck left {}", got[10]);
        assert!(got[50].abs() < 1e-3, "r {radius}: bright speck left {}", got[50]);
        assert!((got[31] - 1.0).abs() < 1e-3 && got[32].abs() < 1e-3, "r {radius}: edge moved {:?}", &got[28..36]);
    }
}

#[test]
fn zero_is_a_passthrough() {
    let Some((ctx, mut comp)) = setup() else { return };
    let got = row(&ctx, &mut comp, &graph_with(0.0));
    for (x, v) in got.iter().enumerate() {
        assert!((v - plate(x as u32, 4)).abs() < 1e-3, "px {x}: {v}");
    }
}
