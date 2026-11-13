//! Warp on real pixels: no curves is the source back pixel for pixel, and
//! two curve pairs sample exactly where the CPU mirror
//! (`WarpParams::sample_at`) says they do. The source is a linear ramp, so a
//! bilinear sample at an interior position reads that position back as
//! colour. Self-skips without a GPU (CI).

use engine::compositor::Compositor;
use engine::decode::ImageInfo;
use engine::graph::{Graph, NodeKind, WarpPair, WarpParams};
use engine::timing::GpuTimer;
use half::f16;

const W: u32 = 64;
const H: u32 = 32;

fn source(x: u32, y: u32) -> [f32; 4] {
    [x as f32 / W as f32, y as f32 / H as f32, 0.5, 1.0]
}

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP warp test: no adapter ({e})");
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

fn graph(p: WarpParams) -> Graph {
    let mut g = Graph::default();
    g.nodes.insert(0, NodeKind::read(0, ""));
    g.nodes.insert(1, NodeKind::Warp(p));
    g.edges.insert((1, 0), 0);
    g.view = Some(1);
    g
}

fn setup() -> Option<(engine::GpuContext, Compositor)> {
    let ctx = ctx()?;
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp, 0, source);
    Some((ctx, comp))
}

#[test]
fn no_curves_is_the_source() {
    let Some((ctx, mut comp)) = setup() else { return };
    let out = frame(&ctx, &mut comp, &graph(WarpParams::default()));
    for y in 0..H {
        for x in 0..W {
            let (got, want) = (out[(y * W + x) as usize], source(x, y));
            assert!((0..4).all(|c| (got[c] - want[c]).abs() <= 2e-3), "({x}, {y}) {got:?} vs {want:?}");
        }
    }
}

#[test]
fn curves_sample_where_the_mirror_says() {
    let Some((ctx, mut comp)) = setup() else { return };
    let p = WarpParams {
        pairs: vec![
            WarpPair {
                src: vec![[0.2, 0.4], [0.5, 0.45], [0.8, 0.4]],
                dst: vec![[0.2, 0.55], [0.5, 0.65], [0.8, 0.55]],
                closed: false,
            },
            WarpPair {
                src: vec![[0.3, 0.2], [0.4, 0.2], [0.4, 0.3]],
                dst: vec![[0.28, 0.18], [0.42, 0.15], [0.45, 0.32]],
                closed: true,
            },
        ],
        falloff: 20.0,
        amount: 1.0,
    };
    let out = frame(&ctx, &mut comp, &graph(p.clone()));
    let mut checked = 0;
    for y in 0..H {
        for x in 0..W {
            let c = [x as f32 + 0.5, y as f32 + 0.5];
            let s = p.sample_at(c, [W as f32, H as f32], 20.0);
            if s[0] < 0.6 || s[1] < 0.6 || s[0] > W as f32 - 0.6 || s[1] > H as f32 - 0.6 {
                continue;
            }
            let want = [(s[0] - 0.5) / W as f32, (s[1] - 0.5) / H as f32];
            let got = out[(y * W + x) as usize];
            assert!(
                (got[0] - want[0]).abs() < 4e-3 && (got[1] - want[1]).abs() < 4e-3,
                "({x}, {y}) rendered {got:?}, want {want:?} (sample {s:?})"
            );
            checked += 1;
        }
    }
    assert!(checked > (W * H / 2) as usize, "only {checked} pixels were checkable");
}

/// The first S-click of a curve is a new topology (one more curve pair),
/// and until its recompile lands the uniforms are packed against the
/// OLD layout. That packed past the end of the mirror and panicked — the
/// crash on drawing a Warp's first curve over a plate. The same for a
/// GridWarp re-grid. Packing a newer graph against an older layout must
/// clamp, not panic.
#[test]
fn packing_ahead_of_a_recompile_does_not_panic() {
    let Some((ctx, mut comp)) = setup() else { return };
    let _ = ctx;
    // The layout of a Warp with no curves at all, as on a fresh node.
    assert!(comp.set_graph(&graph(WarpParams::default())));
    let one = WarpParams {
        pairs: vec![WarpPair { src: vec![[0.5, 0.5]], dst: vec![[0.5, 0.5]], closed: false }],
        ..Default::default()
    };
    comp.write_params(&graph(one.clone()));
    let mut two = one;
    two.pairs.push(two.pairs[0].clone());
    comp.write_params(&graph(two));

    use engine::graph::GridWarpParams;
    let gw = |p: GridWarpParams| {
        let mut g = Graph::default();
        g.nodes.insert(0, NodeKind::read(0, ""));
        g.nodes.insert(1, NodeKind::GridWarp(p));
        g.edges.insert((1, 0), 0);
        g.view = Some(1);
        g
    };
    assert!(comp.set_graph(&gw(GridWarpParams::identity(1, 1))));
    comp.write_params(&gw(GridWarpParams::identity(16, 16)));
}
