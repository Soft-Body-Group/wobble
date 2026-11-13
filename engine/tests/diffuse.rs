//! Headless integration for the Diffuse node through the real Compositor:
//! nothing at amount 0, light scattered (not added) at amount > 0, the veil
//! reaching the mist colour, and a depth pass that keeps the near half sharp
//! and stops a near highlight haloing over the far half. Self-skips without
//! a GPU (CI).

use engine::compositor::Compositor;
use engine::decode::ImageInfo;
use engine::graph::{DiffuseParams, Graph, NodeKind};
use engine::timing::GpuTimer;
use half::f16;

const W: u32 = 256;
const H: u32 = 256;

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP diffuse test: no adapter ({e})");
            None
        }
    }
}

fn upload(ctx: &engine::GpuContext, comp: &mut Compositor, slot: usize, px: &[[f16; 4]]) {
    comp.ensure_reader(slot, ImageInfo { width: W, height: H });
    ctx.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &comp.reader_texture(slot).unwrap().texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(px),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(W * 8),
            rows_per_image: Some(H),
        },
        wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
    );
}

fn render(ctx: &engine::GpuContext, comp: &mut Compositor, graph: &Graph) -> Vec<[f32; 4]> {
    assert!(comp.set_graph(graph), "fused shader failed to compile");
    let timer = GpuTimer::new(ctx);
    // Warm-up encode: stage textures are allocated inside encode_comp, and
    // write_params only reaches stages that already exist.
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
    let mut out = Vec::with_capacity((W * H) as usize);
    for y in 0..H as usize {
        let row = &data[y * padded as usize..][..(W * 8) as usize];
        let halves: &[f16] = bytemuck::cast_slice(row);
        for x in 0..W as usize {
            let p = &halves[x * 4..][..4];
            out.push([p[0].to_f32(), p[1].to_f32(), p[2].to_f32(), p[3].to_f32()]);
        }
    }
    drop(data);
    staging.unmap();
    out
}



fn px(v: f32) -> [f16; 4] {
    [f16::from_f32(v), f16::from_f32(v), f16::from_f32(v), f16::ONE]
}

/// A dim field with 8x8 hot blocks at the given centres.
fn plate(hot: &[(i32, i32)]) -> Vec<[f16; 4]> {
    let mut out = Vec::new();
    for y in 0..H as i32 {
        for x in 0..W as i32 {
            let on = hot.iter().any(|(cx, cy)| (x - cx).abs() < 4 && (y - cy).abs() < 4);
            out.push(px(if on { 20.0 } else { 0.1 }));
        }
    }
    out
}

/// Near (0) on the left half, far (1) on the right.
fn split_depth() -> Vec<[f16; 4]> {
    let mut out = Vec::new();
    for _ in 0..H {
        for x in 0..W {
            out.push(px(if x < W / 2 { 0.0 } else { 1.0 }));
        }
    }
    out
}

fn graph_of(p: DiffuseParams, depth: bool) -> Graph {
    let mut g = Graph::default();
    g.nodes.insert(1, NodeKind::read(0, ""));
    g.nodes.insert(2, NodeKind::Diffuse(p));
    g.edges.insert((2, 0), 1);
    if depth {
        g.nodes.insert(3, NodeKind::read(1, ""));
        g.nodes.insert(4, NodeKind::ToData(Default::default()));
        g.edges.insert((4, 0), 3);
        g.edges.insert((2, 2), 4);
    }
    g.view = Some(2);
    g
}

fn at(o: &[[f32; 4]], x: i32, y: i32) -> f32 {
    o[(y as u32 * W + x as u32) as usize][0]
}

fn diffuse(amount: f32, haze: f32) -> DiffuseParams {
    DiffuseParams { size: 40.0, amount, haze, ..Default::default() }
}

#[test]
fn amount_zero_is_the_input() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 1);
    comp.ensure_output(W, H);
    let src = plate(&[(128, 128)]);
    upload(&ctx, &mut comp, 0, &src);
    let out = render(&ctx, &mut comp, &graph_of(diffuse(0.0, 0.0), false));
    for (o, s) in out.iter().zip(&src) {
        assert!((o[0] - s[0].to_f32()).abs() < 1e-2, "{} vs {}", o[0], s[0]);
    }
}

#[test]
fn light_scatters_and_is_conserved() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 1);
    comp.ensure_output(W, H);
    let src = plate(&[(128, 128)]);
    upload(&ctx, &mut comp, 0, &src);
    let out = render(&ctx, &mut comp, &graph_of(diffuse(0.5, 0.0), false));
    let core = at(&out, 128, 128);
    assert!(core < 15.0 && core > 5.0, "the core gives up light: {core}");
    let (n8, n20, n60) = (at(&out, 136, 128), at(&out, 148, 128), at(&out, 188, 128));
    assert!(n8 > n20 && n20 > n60 && n60 > 0.1, "a falling halo: {n8} {n20} {n60}");
    let sum = |v: &mut dyn Iterator<Item = f32>| v.sum::<f32>();
    let a = sum(&mut src.iter().map(|p| p[0].to_f32()));
    let b = sum(&mut out.iter().map(|p| p[0]));
    assert!((a - b).abs() / a < 0.05, "scattered, not added: {a} -> {b}");
}

#[test]
fn haze_reaches_the_mist() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 1);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp, 0, &plate(&[(128, 128)]));
    let p = DiffuseParams { mist: [0.3, 0.6, 0.9], ..diffuse(0.0, 1.0) };
    let out = render(&ctx, &mut comp, &graph_of(p, false));
    let o = out[(10 * W + 10) as usize];
    assert!((o[0] - 0.3).abs() < 1e-2 && (o[1] - 0.6).abs() < 1e-2 && (o[2] - 0.9).abs() < 1e-2);
}

#[test]
fn depth_keeps_the_near_half_sharp() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 2);
    comp.ensure_output(W, H);
    // One highlight in the near half, right beside the split; one far.
    let src = plate(&[(116, 128), (200, 128)]);
    upload(&ctx, &mut comp, 0, &src);
    upload(&ctx, &mut comp, 1, &split_depth());
    let out = render(&ctx, &mut comp, &graph_of(diffuse(0.8, 0.0), true));
    // Near: untouched, highlight and all.
    assert!((at(&out, 116, 128) - 20.0).abs() < 0.1, "near stays sharp: {}", at(&out, 116, 128));
    assert!((at(&out, 60, 128) - 0.1).abs() < 1e-2);
    // Far: the far highlight is diffused...
    assert!(at(&out, 200, 128) < 10.0, "far diffuses: {}", at(&out, 200, 128));
    assert!(at(&out, 215, 128) > 0.15, "and haloes: {}", at(&out, 215, 128));
    // ...but the near one does not halo over the far half beside it.
    assert!(at(&out, 136, 128) < 0.5, "near light kept out of the far halo: {}", at(&out, 136, 128));

    // Flipped, the halves swap.
    let p = DiffuseParams { far_low: true, ..diffuse(0.8, 0.0) };
    let out = render(&ctx, &mut comp, &graph_of(p, true));
    assert!((at(&out, 200, 128) - 20.0).abs() < 0.1, "flipped: far half sharp");
    assert!(at(&out, 116, 128) < 15.0, "flipped: near half diffuses: {}", at(&out, 116, 128));
}

#[test]
fn modulate_scales_the_effect_by_hand() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 2);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp, 0, &plate(&[(60, 128), (200, 128)]));
    // The split plate as a matte: 0 on the left, 1 on the right.
    upload(&ctx, &mut comp, 1, &split_depth());
    let mut g = graph_of(diffuse(0.8, 0.0), false);
    g.nodes.insert(5, NodeKind::read(1, ""));
    g.edges.insert((2, 3), 5);
    let out = render(&ctx, &mut comp, &g);
    assert!((at(&out, 60, 128) - 20.0).abs() < 0.1, "matte 0: untouched");
    assert!(at(&out, 200, 128) < 10.0, "matte 1: diffused");
}
