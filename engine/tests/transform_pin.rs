//! A Transform's corner pin (`TransformParams::pin`, what a ctrl-drag on a
//! corner handle makes) renders exactly what a CornerPin with the same
//! corners does, an unpinned Transform is untouched by it, and moving a pin
//! is a uniform write — never a recompile. Self-skips without a GPU (CI).

use engine::compositor::Compositor;
use engine::decode::ImageInfo;
use engine::graph::{CornerPinParams, Graph, NodeKind, TransformParams, UNIT_QUAD};
use engine::timing::GpuTimer;
use half::f16;

const W: u32 = 128;
const H: u32 = 64;

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP loop test: no adapter ({e})");
            None
        }
    }
}

/// A horizontal ramp, 0 at the left edge to 1 at the right, alpha 1: any
/// warp shows up as the values moving.
fn upload_ramp(ctx: &engine::GpuContext, comp: &mut Compositor, slot: usize) {
    comp.ensure_reader(slot, ImageInfo { width: W, height: H });
    let mut px: Vec<[f16; 4]> = Vec::with_capacity((W * H) as usize);
    for y in 0..H {
        for x in 0..W {
            let v = x as f32 / W as f32;
            let g = y as f32 / H as f32;
            px.push([f16::from_f32(v), f16::from_f32(g), f16::from_f32(0.0), f16::from_f32(1.0)]);
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

fn one(kind: NodeKind) -> Graph {
    let mut g = Graph::default();
    g.nodes.insert(1, NodeKind::read(0, ""));
    g.nodes.insert(2, kind);
    g.edges.insert((2, 0), 1);
    g.view = Some(2);
    g
}

const QUAD: [[f32; 2]; 4] = [[0.1, 0.05], [0.8, 0.0], [1.0, 0.9], [0.05, 0.8]];

#[test]
fn a_pinned_transform_renders_what_a_corner_pin_does() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload_ramp(&ctx, &mut comp, 0);
    let pinned = row(
        &ctx,
        &mut comp,
        &one(NodeKind::Transform(TransformParams { pin: QUAD, ..Default::default() })),
    );
    let cp = row(&ctx, &mut comp, &one(NodeKind::CornerPin(CornerPinParams::pinned(QUAD))));
    let worst = pinned.iter().zip(&cp).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(worst < 2e-3, "pinned Transform and CornerPin differ by {worst}");
    // And the pin actually did something.
    let plain = row(&ctx, &mut comp, &one(NodeKind::Transform(TransformParams::default())));
    let moved = pinned.iter().zip(&plain).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(moved > 0.05, "the pin moved nothing ({moved})");
    // An unpinned Transform is the plate.
    let mid = plain[(W / 2) as usize];
    assert!((mid - 0.5).abs() < 0.02, "identity Transform moved the plate: {mid}");
}

#[test]
fn moving_a_pin_is_not_topology() {
    let a = one(NodeKind::Transform(TransformParams::default()));
    let b = one(NodeKind::Transform(TransformParams { pin: QUAD, ..Default::default() }));
    assert_eq!(a.topology_hash(), b.topology_hash());
    assert_eq!(TransformParams::default().pin, UNIT_QUAD);
}
