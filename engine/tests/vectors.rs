//! Motion vectors on real pixels: a Flow node plays its vectors back raw
//! (no colour transform, ever) and is zero motion until loaded; VectorWarp
//! samples where the vectors point, forward or backward by the sign of
//! its amount; VectorBlur spreads a line along its motion without losing
//! or gaining energy, and does nothing where nothing moved. Self-skips
//! without a GPU (CI).

use engine::compositor::Compositor;
use engine::decode::ImageInfo;
use engine::graph::{FlowParams, Graph, NodeKind, VectorBlurParams, VectorWarpParams};
use engine::timing::GpuTimer;
use half::f16;

const W: u32 = 16;
const H: u32 = 8;

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP vectors test: no adapter ({e})");
            None
        }
    }
}

fn upload(ctx: &engine::GpuContext, comp: &mut Compositor, slot: usize, f: impl Fn(u32, u32) -> [f32; 4]) {
    comp.ensure_reader(slot, ImageInfo { width: W, height: H });
    let px: Vec<[f16; 4]> = (0..W * H).map(|i| f(i % W, i / W).map(f16::from_f32)).collect();
    ctx.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &comp.reader_texture(slot).unwrap().texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&px),
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(W * 8), rows_per_image: Some(H) },
        wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
    );
}

fn frame(ctx: &engine::GpuContext, comp: &mut Compositor, graph: &Graph) -> Vec<[f32; 4]> {
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
    let mut out = Vec::new();
    for y in 0..H as usize {
        let row: &[f16] = bytemuck::cast_slice(&data[y * padded as usize..][..(W * 8) as usize]);
        for x in 0..W as usize {
            out.push([
                row[x * 4].to_f32(),
                row[x * 4 + 1].to_f32(),
                row[x * 4 + 2].to_f32(),
                row[x * 4 + 3].to_f32(),
            ]);
        }
    }
    drop(data);
    staging.unmap();
    out
}


fn setup(src: impl Fn(u32, u32) -> [f32; 4]) -> Option<(engine::GpuContext, Compositor)> {
    let ctx = ctx()?;
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp, 0, src);
    Some((ctx, comp))
}

/// Read 0 the picture, a LOADED Flow on slot 1 the vectors, `consumer`
/// over both.
fn graph(consumer: NodeKind) -> Graph {
    let mut g = Graph::default();
    g.nodes.insert(0, NodeKind::read(0, ""));
    g.nodes.insert(1, NodeKind::Flow(FlowParams { loaded: true, ..FlowParams::new(1) }));
    g.nodes.insert(2, consumer);
    g.edges.insert((2, 0), 0);
    g.edges.insert((2, 1), 1);
    g.view = Some(2);
    g
}

fn ramp(x: u32, y: u32) -> [f32; 4] {
    [x as f32 / W as f32, y as f32 / H as f32, 0.5, 1.0]
}

#[test]
fn a_flow_plays_its_vectors_raw_and_is_still_until_loaded() {
    let Some((ctx, mut comp)) = setup(ramp) else { return };
    let vec = |x: u32, y: u32| [x as f32 - 3.5, -(y as f32) * 2.0, 7.25, -40.0];
    upload(&ctx, &mut comp, 1, vec);
    let mut g = Graph { working: engine::color::Colorspace::ACESCG, ..Graph::default() };
    g.nodes.insert(1, NodeKind::Flow(FlowParams { loaded: true, ..FlowParams::new(1) }));
    g.view = Some(1);
    let out = frame(&ctx, &mut comp, &g);
    for y in 0..H {
        for x in 0..W {
            let (got, want) = (out[(y * W + x) as usize], vec(x, y));
            assert!((0..4).all(|c| (got[c] - want[c]).abs() < 0.02), "({x},{y}) {got:?} != {want:?}: vectors were transformed");
        }
    }
    g.nodes.insert(1, NodeKind::Flow(FlowParams::new(1)));
    let out = frame(&ctx, &mut comp, &g);
    assert!(out.iter().all(|p| *p == [0.0; 4]), "an unloaded Flow is no motion");
}

#[test]
fn a_vector_warp_follows_forward_or_backward_vectors() {
    let Some((ctx, mut comp)) = setup(ramp) else { return };
    upload(&ctx, &mut comp, 1, |_, _| [2.0, 1.0, -1.0, 0.0]);
    for amount in [1.0f32, 0.5, -1.0, 0.0] {
        let p = VectorWarpParams { amount };
        let out = frame(&ctx, &mut comp, &graph(NodeKind::VectorWarp(p)));
        for y in 1..H - 2 {
            for x in 2..W - 3 {
                let q = p.lookup([x as f32 + 0.5, y as f32 + 0.5], [2.0, 1.0, -1.0, 0.0]);
                // The ramp is linear, so bilinear sampling is exact.
                let want = [(q[0] - 0.5) / W as f32, (q[1] - 0.5) / H as f32];
                let got = out[(y * W + x) as usize];
                assert!(
                    (got[0] - want[0]).abs() < 3e-3 && (got[1] - want[1]).abs() < 3e-3,
                    "amount {amount}: ({x},{y}) {got:?}, want {want:?}"
                );
            }
        }
    }
}

#[test]
fn a_vector_blur_spreads_a_line_along_its_motion_and_keeps_its_energy() {
    // One bright column, moving 4 px right a frame.
    let line = |x: u32, _y: u32| if x == 8 { [1.0, 1.0, 1.0, 1.0] } else { [0.0, 0.0, 0.0, 1.0] };
    let Some((ctx, mut comp)) = setup(line) else { return };
    upload(&ctx, &mut comp, 1, |_, _| [4.0, 0.0, -4.0, 0.0]);
    let p = VectorBlurParams { amount: 1.0, offset: 0.0, samples: 16, matte: false };
    let out = frame(&ctx, &mut comp, &graph(NodeKind::VectorBlur(p)));
    let row: Vec<f32> = (0..W).map(|x| out[(3 * W + x) as usize][0]).collect();
    let sum: f32 = row.iter().sum();
    assert!((sum - 1.0).abs() < 0.05, "energy {sum} in {row:?}");
    assert!(row[8] < 0.5, "the line stayed sharp: {row:?}");
    assert!(row[6] > 0.05 && row[10] > 0.05, "the blur does not reach two pixels each way: {row:?}");
    assert!(row[3] < 1e-3 && row[13] < 1e-3, "the blur reaches past the motion: {row:?}");

    // The same with the Flow's own input wired, as it always is in a comp
    // (the analysis reads it; the shader must not care).
    let mut g = graph(NodeKind::VectorBlur(p));
    g.edges.insert((1, 0), 0);
    let wired = frame(&ctx, &mut comp, &g);
    for x in 0..W {
        let (a, b) = (wired[(3 * W + x) as usize][0], row[x as usize]);
        assert!((a - b).abs() < 1e-3, "wiring the Flow's input changed the blur at {x}: {a} vs {b}");
    }

    // No motion: the picture, exactly.
    upload(&ctx, &mut comp, 1, |_, _| [0.0; 4]);
    let out = frame(&ctx, &mut comp, &graph(NodeKind::VectorBlur(p)));
    for x in 0..W {
        let want = line(x, 0)[0];
        assert!((out[(3 * W + x) as usize][0] - want).abs() < 1e-3, "still pixels moved at {x}");
    }
}
