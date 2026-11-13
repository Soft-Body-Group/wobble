//! Mirror and Flip/Flop on real pixels: the fused shader samples exactly
//! where the CPU model (`lookup`) says, over the node's own format, and a
//! matte comes out a matte. Self-skips without a GPU.

use engine::compositor::Compositor;
use engine::decode::ImageInfo;
use engine::graph::{
    FlipFlopParams, FlipMode, Graph, MatteChannel, MirrorParams, MirrorSide, NodeKind, ToMatteParams,
};
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
            eprintln!("SKIP mirror test: no adapter ({e})");
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


fn graph(op: NodeKind) -> Graph {
    let mut g = Graph::default();
    g.nodes.insert(0, NodeKind::read(0, ""));
    g.nodes.insert(1, op);
    g.edges.insert((1, 0), 0);
    g.view = Some(1);
    g
}

/// The rendered picture is the plate sampled where `lookup` says: at the
/// pixel centre it lands on (every case here reflects centres onto centres).
fn check(name: &str, out: &[[f32; 4]], lookup: impl Fn([f32; 2]) -> [f32; 2]) {
    for y in 0..H {
        for x in 0..W {
            let q = lookup([x as f32 + 0.5, y as f32 + 0.5]);
            let (sx, sy) = ((q[0] - 0.5).round() as i64, (q[1] - 0.5).round() as i64);
            // Reflected off the frame: nothing there, like a Transform.
            let inside = (0..W as i64).contains(&sx) && (0..H as i64).contains(&sy);
            let want = if inside { source(sx as u32, sy as u32) } else { [0.0; 4] };
            let got = out[(y * W + x) as usize];
            for c in 0..3 {
                assert!(
                    (got[c] - want[c]).abs() < 4e-3,
                    "{name} at ({x},{y}) ch{c}: {} want {} (from {sx},{sy})",
                    got[c],
                    want[c]
                );
            }
        }
    }
}

#[test]
fn flip_flop_turns_the_picture_over() {
    let Some(ctx) = ctx() else { return };
    for mode in FlipMode::ALL {
        let p = FlipFlopParams { mode, matte: false };
        let out = render(&ctx, &graph(NodeKind::FlipFlop(p)));
        check(mode.name(), &out, |q| p.lookup(q, [W as f32, H as f32]));
    }
}

#[test]
fn mirror_reflects_across_its_line() {
    let Some(ctx) = ctx() else { return };
    let cases = [
        MirrorParams::default(),
        MirrorParams { side: MirrorSide::Back, ..Default::default() },
        MirrorParams { side: MirrorSide::Both, center: [0.25, 0.5], ..Default::default() },
        MirrorParams { angle: 0.0, center: [0.5, 0.5], ..Default::default() },
        MirrorParams { angle: 180.0, center: [0.5, 0.25], ..Default::default() },
    ];
    for p in cases {
        let out = render(&ctx, &graph(NodeKind::Mirror(p)));
        check(&format!("{p:?}"), &out, |q| p.lookup(q, [W as f32, H as f32]));
    }
}

/// On the matte pipe both hand back a matte: coverage in all four.
#[test]
fn a_flipped_matte_is_a_matte() {
    let Some(ctx) = ctx() else { return };
    for mut op in [NodeKind::FlipFlop(Default::default()), NodeKind::Mirror(Default::default())] {
        assert!(op.follows_pipe(0));
        op.set_matte_mode(true);
        let mut g = Graph::default();
        g.nodes.insert(0, NodeKind::read(0, ""));
        g.nodes.insert(1, NodeKind::ToMatte(ToMatteParams { channel: MatteChannel::G, ..Default::default() }));
        g.nodes.insert(2, op);
        g.edges.insert((1, 0), 0);
        g.edges.insert((2, 0), 1);
        g.view = Some(2);
        let out = render(&ctx, &g);
        for px in &out {
            assert!(px.iter().all(|c| (c - px[0]).abs() < 1e-6), "not a matte: {px:?}");
        }
        assert!(out.iter().any(|p| p[0] > 0.5), "the matte is empty");
    }
}
