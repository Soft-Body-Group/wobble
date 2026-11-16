//! Turn on the GPU: every output pixel is exactly the input pixel the
//! CPU mirror (`TurnParams::lookup`) names, in the TURNED format.

use engine::compositor::Compositor;
use engine::decode::ImageInfo;
use engine::graph::{Graph, MatteChannel, NodeKind, ToMatteParams, TurnMode, TurnParams};
use engine::timing::GpuTimer;
use half::f16;

const W: u32 = 40;
const H: u32 = 24;

/// Every pixel a different colour, so a wrong lookup cannot hide.
fn source(x: u32, y: u32) -> [f32; 4] {
    [x as f32 / W as f32, y as f32 / H as f32, ((x * 7 + y * 3) % 5) as f32 / 5.0, 1.0]
}

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP turn test: no adapter ({e})");
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

fn render(ctx: &engine::GpuContext, graph: &Graph, (ow, oh): (u32, u32)) -> Vec<[f32; 4]> {
    // A fresh compositor per render: the chain is chosen when it is built.
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(ow, oh);
    let plate: Vec<[f32; 4]> = (0..W * H).map(|i| source(i % W, i / W)).collect();
    upload(ctx, &mut comp, 0, &plate);
    assert!(comp.set_graph(graph), "fused shader failed to compile");
    let timer = GpuTimer::new(ctx);
    let mut warm = ctx.device.create_command_encoder(&Default::default());
    comp.encode_comp(&mut warm, &timer);
    ctx.queue.submit([warm.finish()]);
    comp.write_params(graph);
    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    comp.encode_comp(&mut encoder, &timer);
    let padded = (ow * 8).div_ceil(256) * 256;
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (padded * oh) as u64,
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
                rows_per_image: Some(oh),
            },
        },
        wgpu::Extent3d { width: ow, height: oh, depth_or_array_layers: 1 },
    );
    ctx.queue.submit([encoder.finish()]);
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    let data = slice.get_mapped_range().expect("a mapped buffer");
    let mut out = Vec::new();
    for y in 0..oh as usize {
        let row: &[f16] = bytemuck::cast_slice(&data[y * padded as usize..][..(ow * 8) as usize]);
        for x in 0..ow as usize {
            out.push(std::array::from_fn(|c| row[x * 4 + c].to_f32()));
        }
    }
    drop(data);
    staging.unmap();
    out
}



/// A plate into a Turn, the formats as the app resolves them.
fn graph(p: TurnParams) -> (Graph, (u32, u32)) {
    let out = p.output_format((W, H));
    let mut g = Graph::default();
    g.nodes.insert(0, NodeKind::read(0, ""));
    g.nodes.insert(1, NodeKind::Turn(p));
    g.edges.insert((1, 0), 0);
    g.formats.insert(0, (W, H));
    g.formats.insert(1, out);
    g.view = Some(1);
    (g, out)
}

#[test]
fn each_turn_lands_every_pixel_and_swaps_the_format() {
    let Some(ctx) = ctx() else { return };
    for mode in TurnMode::ALL {
        let p = TurnParams { mode, matte: false };
        let (g, (ow, oh)) = graph(p);
        assert_eq!((ow, oh), if mode.swaps() { (H, W) } else { (W, H) });
        let out = render(&ctx, &g, (ow, oh));
        for y in 0..oh {
            for x in 0..ow {
                let q = p.lookup([x as f32 + 0.5, y as f32 + 0.5], [ow as f32, oh as f32]);
                let (sx, sy) = ((q[0] - 0.5).round() as u32, (q[1] - 0.5).round() as u32);
                assert!(sx < W && sy < H, "{} at ({x},{y}) left the plate", mode.name());
                let want = source(sx, sy);
                let got = out[(y * ow + x) as usize];
                for c in 0..3 {
                    assert!(
                        (got[c] - want[c]).abs() < 4e-3,
                        "{} at ({x},{y}) ch{c}: {} want {} (from {sx},{sy})",
                        mode.name(),
                        got[c],
                        want[c]
                    );
                }
            }
        }
    }
}

/// On the matte pipe it hands back a matte.
#[test]
fn a_turned_matte_is_a_matte() {
    let Some(ctx) = ctx() else { return };
    let mut op = NodeKind::Turn(Default::default());
    assert!(op.follows_pipe(0));
    op.set_matte_mode(true);
    let mut g = Graph::default();
    g.nodes.insert(0, NodeKind::read(0, ""));
    g.nodes.insert(1, NodeKind::ToMatte(ToMatteParams { channel: MatteChannel::G, ..Default::default() }));
    g.nodes.insert(2, op);
    g.edges.insert((1, 0), 0);
    g.edges.insert((2, 0), 1);
    g.formats.insert(0, (W, H));
    g.formats.insert(1, (W, H));
    g.formats.insert(2, (H, W));
    g.view = Some(2);
    let out = render(&ctx, &g, (H, W));
    for px in &out {
        assert!(px.iter().all(|c| (c - px[0]).abs() < 1e-6), "not a matte: {px:?}");
    }
    assert!(out.iter().any(|p| p[0] > 0.5), "the matte is empty");
}
