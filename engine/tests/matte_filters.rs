//! Filters on a MATTE run on one channel. Wired a matte, a Blur, Median,
//! Convolve, Dilate or VectorBlur takes its one-channel chain (single-float
//! kernels, `r16float` scratch) where the device can write one, and the
//! answer must be the same coverage the four-channel chain gives — in all
//! four channels, so the matte stays a matte. Self-skips without a GPU.

use engine::compositor::Compositor;
use engine::decode::ImageInfo;
use engine::graph::{
    BlurParams, ConvolveParams, DilateParams, Graph, MatteChannel, MedianParams, NodeKind,
    ToMatteParams, VectorBlurParams,
};
use engine::nodes::stage;
use engine::timing::GpuTimer;
use half::f16;

const W: u32 = 40;
const H: u32 = 24;

/// Hard-edged shapes in red (what the matte is pulled from) and noise in
/// green, so a filter that mixed channels would show.
fn source(x: u32, y: u32) -> [f32; 4] {
    let disc = ((x as f32 - 14.0).powi(2) + (y as f32 - 11.0).powi(2)).sqrt() < 7.0;
    let bar = (26..31).contains(&x);
    let speck = (x * 7 + y * 13) % 11 == 0;
    let r = if disc || bar { 1.0 } else if speck { 0.6 } else { 0.0 };
    [r, ((x * 31 + y * 17) % 7) as f32 / 7.0, 0.25, 1.0]
}

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP matte filters test: no adapter ({e})");
            None
        }
    }
}

fn upload(ctx: &engine::GpuContext, comp: &mut Compositor, slot: usize, px: &[[f32; 4]]) {
    comp.ensure_reader(slot, ImageInfo { width: W, height: H });
    let data: Vec<[f16; 4]> = px.iter().map(|c| c.map(f16::from_f32)).collect();
    ctx.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &comp.reader_texture(slot).unwrap().texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&data),
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(W * 8), rows_per_image: Some(H) },
        wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
    );
}

fn render(ctx: &engine::GpuContext, graph: &Graph) -> Vec<[f32; 4]> {
    // A fresh compositor per render: the chain is chosen when it is built.
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    let plate: Vec<[f32; 4]> = (0..W * H).map(|i| source(i % W, i / W)).collect();
    upload(ctx, &mut comp, 0, &plate);
    // Slot 1: a constant motion field for the VectorBlur, 3 px right.
    upload(ctx, &mut comp, 1, &vec![[3.0, 0.0, -3.0, 0.0]; (W * H) as usize]);
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
            out.push(std::array::from_fn(|c| row[x * 4 + c].to_f32()));
        }
    }
    drop(data);
    staging.unmap();
    out
}

/// Read -> ToMatte (red) -> `filter`, viewed; the filter in the matte pipe
/// unless it is a Dilate (always one).
fn graph(mut filter: NodeKind) -> Graph {
    filter.set_matte_mode(true);
    let mut g = Graph::default();
    g.nodes.insert(0, NodeKind::read(0, ""));
    g.nodes.insert(
        1,
        NodeKind::ToMatte(ToMatteParams { channel: MatteChannel::R, ..Default::default() }),
    );
    g.nodes.insert(2, filter);
    g.nodes.insert(3, NodeKind::read(1, ""));
    g.edges.insert((1, 0), 0);
    g.edges.insert((2, 0), 1);
    if matches!(g.nodes[&2], NodeKind::VectorBlur(_)) {
        g.edges.insert((2, 1), 3);
    }
    g.view = Some(2);
    g
}

fn filters() -> Vec<(&'static str, NodeKind)> {
    vec![
        ("blur", NodeKind::Blur(BlurParams { size: 2.5, ..Default::default() })),
        ("median", NodeKind::Median(MedianParams { radius: 2.0, ..Default::default() })),
        ("median-wide", NodeKind::Median(MedianParams { radius: 5.0, ..Default::default() })),
        ("convolve", NodeKind::Convolve(ConvolveParams::default())),
        ("dilate", NodeKind::Dilate(DilateParams { amount: 1.6 })),
        ("erode", NodeKind::Dilate(DilateParams { amount: -1.3 })),
        ("vectorblur", NodeKind::VectorBlur(VectorBlurParams { amount: 1.0, ..Default::default() })),
    ]
}

#[test]
fn one_channel_chains_agree_with_four_channel_chains() {
    let Some(ctx) = ctx() else { return };
    let can = stage::one_channel();
    eprintln!("one-channel chains on this device: {can}");
    for (name, f) in filters() {
        let g = graph(f);
        stage::set_one_channel(false);
        let four = render(&ctx, &g);
        stage::set_one_channel(can);
        let one = render(&ctx, &g);
        let mut moved = 0;
        for (i, (a, b)) in four.iter().zip(&one).enumerate() {
            let (x, y) = (i as u32 % W, i as u32 / W);
            for c in 0..4 {
                assert!(
                    (a[0] - b[c]).abs() <= 2e-3 + 2e-3 * a[0].abs(),
                    "{name} at ({x},{y}) ch{c}: one-channel {} vs four-channel coverage {}",
                    b[c],
                    a[0]
                );
                assert!((a[0] - a[c]).abs() < 1e-6, "{name}: the four-channel matte is not splatted");
            }
            if (a[0] - source(x, y)[0]).abs() > 0.02 {
                moved += 1;
            }
        }
        assert!(moved > 20, "{name}: only {moved} pixels changed — the filter did nothing");
    }
}

/// Where the device can, this IS the one-channel path: the program a matte
/// Blur builds has single-channel scratch. (On a device that cannot, the
/// four-channel chain is the answer and this says so.)
#[test]
fn a_matte_blur_builds_single_channel_scratch() {
    use engine::nodes::stage::{StageDef, TexFormat, Wired};
    let Some(_ctx) = ctx() else { return };
    let p = BlurParams::default().program_one_channel((64, 64), Wired(&[true, false])).unwrap();
    assert!(p.textures.iter().all(|t| t.format == TexFormat::R16Float));
    assert!(p.passes.iter().all(|pass| pass.wgsl.contains("r16float")));
    if !stage::one_channel() {
        eprintln!("NOTE: this device cannot write r16float; mattes take the four-channel chain");
    }
}
