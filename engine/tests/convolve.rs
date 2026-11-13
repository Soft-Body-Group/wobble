//! Headless integration for Convolve: identity is a passthrough, a box
//! normalises, an edge kernel (sum 0) is left unnormalised, mix 0 is the
//! input. Self-skips without a GPU (CI).

use engine::compositor::Compositor;
use engine::decode::ImageInfo;
use engine::graph::{ConvolveParams, ConvolvePreset, Graph, NodeKind};
use engine::timing::GpuTimer;
use half::f16;

const W: u32 = 64;
const H: u32 = 8;

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP convolve test: no adapter ({e})");
            None
        }
    }
}

/// rgb 1 left of x = 32, 0 right of it.
fn plate(x: u32, _y: u32) -> f32 {
    if x < 32 { 1.0 } else { 0.0 }
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

fn graph_with(p: ConvolveParams) -> Graph {
    let mut g = Graph::default();
    g.nodes.insert(0, NodeKind::read(0, ""));
    g.nodes.insert(1, NodeKind::Convolve(p));
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

fn near(a: f32, b: f32) -> bool {
    (a - b).abs() < 2e-3
}

#[test]
fn identity_and_mix_zero_pass_through() {
    let Some((ctx, mut comp)) = setup() else { return };
    let id = row(&ctx, &mut comp, &graph_with(ConvolveParams::preset(ConvolvePreset::Identity)));
    let off = row(&ctx, &mut comp, &graph_with(ConvolveParams { mix: 0.0, ..Default::default() }));
    for x in 0..W as usize {
        assert!(near(id[x], plate(x as u32, 4)), "identity px {x}: {}", id[x]);
        assert!(near(off[x], plate(x as u32, 4)), "mix 0 px {x}: {}", off[x]);
    }
}

#[test]
fn a_box_is_normalised_and_its_size_counts() {
    let Some((ctx, mut comp)) = setup() else { return };
    let b3 = row(&ctx, &mut comp, &graph_with(ConvolveParams::preset(ConvolvePreset::Box)));
    assert!(near(b3[31], 2.0 / 3.0) && near(b3[32], 1.0 / 3.0), "{:?}", &b3[29..35]);
    assert!(near(b3[4], 1.0) && near(b3[60], 0.0), "flat parts keep their level");
    // The same box widened to 5x5: fill the ring the 3x3 left at 0.
    let mut b5 = ConvolveParams::preset(ConvolvePreset::Box);
    b5.size = 5;
    for y in 1..6 {
        for x in 1..6 {
            b5.kernel[y * 7 + x] = 1.0;
        }
    }
    let got = row(&ctx, &mut comp, &graph_with(b5));
    assert!(near(got[30], 0.8) && near(got[33], 0.2), "{:?}", &got[29..35]);
}

#[test]
fn a_zero_sum_kernel_is_not_normalised() {
    let Some((ctx, mut comp)) = setup() else { return };
    let got = row(&ctx, &mut comp, &graph_with(ConvolveParams::preset(ConvolvePreset::Edge)));
    // Laplacian of the step: -1 just inside, +1 just outside, 0 elsewhere.
    assert!(near(got[31], -1.0) && near(got[32], 1.0), "{:?}", &got[29..35]);
    assert!(near(got[4], 0.0) && near(got[60], 0.0));
}
