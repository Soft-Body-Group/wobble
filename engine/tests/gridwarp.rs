//! GridWarp on real pixels: the identity lattice gives the source back
//! pixel for pixel, and a pulled lattice samples exactly where the CPU
//! mirror (`GridWarpParams::sample_at`) says it does — through Newton's
//! inverse in the shader, smooth and bilinear. The source is a linear ramp,
//! so a bilinear sample at any interior position is that position, read
//! back as colour. Self-skips without a GPU (CI).

use engine::compositor::Compositor;
use engine::decode::ImageInfo;
use engine::graph::{GridWarpParams, Graph, NodeKind};
use engine::timing::GpuTimer;
use half::f16;

const W: u32 = 64;
const H: u32 = 32;

/// x and y as colour, alpha 1: a bilinear sample reads its position back.
fn source(x: u32, y: u32) -> [f32; 4] {
    [x as f32 / W as f32, y as f32 / H as f32, 0.5, 1.0]
}

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP gridwarp test: no adapter ({e})");
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

fn graph(p: GridWarpParams) -> Graph {
    let mut g = Graph::default();
    g.nodes.insert(0, NodeKind::read(0, ""));
    g.nodes.insert(1, NodeKind::GridWarp(p));
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
fn the_identity_lattice_is_the_source() {
    let Some((ctx, mut comp)) = setup() else { return };
    for smooth in [true, false] {
        let p = GridWarpParams { smooth, ..GridWarpParams::identity(4, 3) };
        let out = frame(&ctx, &mut comp, &graph(p));
        for y in 0..H {
            for x in 0..W {
                let (got, want) = (out[(y * W + x) as usize], source(x, y));
                assert!(
                    (0..4).all(|c| (got[c] - want[c]).abs() <= 2e-3),
                    "smooth {smooth}: ({x}, {y}) rendered {got:?}, want {want:?}"
                );
            }
        }
    }
}

#[test]
fn a_pulled_lattice_samples_where_the_mirror_says() {
    let Some((ctx, mut comp)) = setup() else { return };
    for smooth in [true, false] {
        let mut p = GridWarpParams { smooth, ..GridWarpParams::identity(4, 4) };
        let k = p.index(2, 2);
        p.dst[k] = [0.62, 0.38];
        let k = p.index(1, 3);
        p.src[k] = [0.3, 0.68];
        let k = p.index(3, 1);
        p.dst[k] = [0.7, 0.3];
        p.src[k] = [0.78, 0.2];
        let out = frame(&ctx, &mut comp, &graph(p.clone()));
        let mut checked = 0;
        for y in 0..H {
            for x in 0..W {
                let c = [x as f32 + 0.5, y as f32 + 0.5];
                let s = p.sample_at(c, [W as f32, H as f32]).expect("this lattice does not fold");
                // Inside the texels' centres, where the ramp is exact.
                if s[0] < 0.6 || s[1] < 0.6 || s[0] > W as f32 - 0.6 || s[1] > H as f32 - 0.6 {
                    continue;
                }
                let want = [(s[0] - 0.5) / W as f32, (s[1] - 0.5) / H as f32];
                let got = out[(y * W + x) as usize];
                assert!(
                    (got[0] - want[0]).abs() < 4e-3 && (got[1] - want[1]).abs() < 4e-3 && got[3] > 0.999,
                    "smooth {smooth}: ({x}, {y}) rendered {got:?}, want {want:?} (sample {s:?})"
                );
                checked += 1;
            }
        }
        assert!(checked > (W * H / 2) as usize, "only {checked} pixels were checkable");
    }
}
