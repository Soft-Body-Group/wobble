//! Headless integration for the Glint node through the real Compositor:
//! rays go where the count and rotation say and nowhere else, stop near
//! their length, fade with falloff, and the threshold keeps dim pixels out.
//! Self-skips without a GPU (CI).

use engine::compositor::Compositor;
use engine::decode::ImageInfo;
use engine::graph::{GlintParams, GlowMode, Graph, NodeKind};
use engine::timing::GpuTimer;
use half::f16;

const W: u32 = 256;
const H: u32 = 256;

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP glint test: no adapter ({e})");
            None
        }
    }
}

fn upload(ctx: &engine::GpuContext, comp: &mut Compositor, px: &[[f16; 4]]) {
    comp.ensure_reader(0, ImageInfo { width: W, height: H });
    ctx.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &comp.reader_texture(0).unwrap().texture,
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
    // Warm-up encode: the pyramid textures are allocated inside encode_comp,
    // and write_params only reaches stages that already exist.
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

/// A 4x4 hot block at the centre on a dim (sub-threshold) grey field.
fn plate(hot_v: f32) -> Vec<[f16; 4]> {
    let mut px = Vec::new();
    for y in 0..H as usize {
        for x in 0..W as usize {
            let hot = (126..130).contains(&x) && (126..130).contains(&y);
            let v = if hot { hot_v } else { 0.2 };
            px.push([f16::from_f32(v), f16::from_f32(v), f16::from_f32(v), f16::ONE]);
        }
    }
    px
}

fn graph(p: GlintParams) -> Graph {
    let mut g = Graph::default();
    g.nodes.insert(1, NodeKind::read(0, ""));
    g.nodes.insert(2, NodeKind::Glint(p));
    g.edges.insert((2, 0), 1);
    g.view = Some(2);
    g
}

fn only() -> GlintParams {
    GlintParams {
        rays: 4.0,
        length: 80.0,
        rotation: 0.0,
        falloff: 0.3,
        mode: GlowMode::Only,
        ..Default::default()
    }
}

/// Red at (x, y), offset from the block's centre (128, 128).
fn at(o: &[[f32; 4]], dx: i32, dy: i32) -> f32 {
    o[((128 + dy) as u32 * W + (128 + dx) as u32) as usize][0]
}

#[test]
fn rays_follow_count_rotation_and_length() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 1);
    upload(&ctx, &mut comp, &plate(16.0));
    comp.ensure_output(W, H);

    let out = render(&ctx, &mut comp, &graph(only()));
    // Four rays on the axes, both ways along each.
    for (dx, dy) in [(40, 0), (-40, 0), (0, 40), (0, -40)] {
        assert!(at(&out, dx, dy) > 0.05, "ray at ({dx},{dy}) got {}", at(&out, dx, dy));
    }
    // Nothing between the rays, and nothing well past the tip.
    assert!(at(&out, 30, 30) < 1e-3, "diagonal got {}", at(&out, 30, 30));
    assert!(at(&out, 110, 0) < 1e-3, "past the tip got {}", at(&out, 110, 0));
    // The tail falls off along the ray.
    assert!(at(&out, 10, 0) > at(&out, 60, 0), "{} vs {}", at(&out, 10, 0), at(&out, 60, 0));
    // The ray is thin: a few pixels off its axis is dark.
    assert!(at(&out, 40, 6) < 0.1 * at(&out, 40, 0), "ray too wide: {}", at(&out, 40, 6));

    // Two rays turned a quarter: up and down only.
    let out = render(&ctx, &mut comp, &graph(GlintParams { rays: 2.0, rotation: 90.0, ..only() }));
    assert!(at(&out, 0, -40) > 0.05, "up ray got {}", at(&out, 0, -40));
    assert!(at(&out, 0, 40) > 0.05, "down ray got {}", at(&out, 0, 40));
    assert!(at(&out, 40, 0) < 1e-3, "no ray to the right, got {}", at(&out, 40, 0));

    // A steeper falloff dims the far end of the ray.
    let soft = render(&ctx, &mut comp, &graph(GlintParams { falloff: 1.0, ..only() }));
    let flat = render(&ctx, &mut comp, &graph(GlintParams { falloff: 0.0, ..only() }));
    assert!(flat[128 * W as usize + 188][0] > 4.0 * soft[128 * W as usize + 188][0]);

    // Zero rays: nothing at all.
    let none = render(&ctx, &mut comp, &graph(GlintParams { rays: 0.0, ..only() }));
    assert!(none.iter().all(|p| p[0].abs() < 1e-4), "zero rays still drew");
}

#[test]
fn threshold_keeps_dim_pixels_out_and_add_keeps_the_plate() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 1);
    // The block is under the threshold: no rays.
    upload(&ctx, &mut comp, &plate(0.4));
    comp.ensure_output(W, H);
    let p = GlintParams { threshold: 1.0, knee: 0.0, ..only() };
    let out = render(&ctx, &mut comp, &graph(p));
    assert!(out.iter().all(|p| p[0].abs() < 1e-4), "sub-threshold pixels made rays");

    // Add over a hot plate: the plate comes through, the rays sit on it.
    upload(&ctx, &mut comp, &plate(16.0));
    let add = render(&ctx, &mut comp, &graph(GlintParams { mode: GlowMode::Add, ..only() }));
    assert!((at(&add, 30, 30) - 0.2).abs() < 0.01, "plate off the rays got {}", at(&add, 30, 30));
    assert!(at(&add, 40, 0) > 0.25, "ray on the plate got {}", at(&add, 40, 0));
}

#[test]
fn isolate_keeps_the_ping_and_drops_the_window() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 1);
    // A 4x4 ping at the centre, and a 40x40 hot "window" in the top-left.
    let mut px = plate(16.0);
    for y in 20..60usize {
        for x in 20..60usize {
            px[y * W as usize + x] = [f16::from_f32(16.0), f16::from_f32(16.0), f16::from_f32(16.0), f16::ONE];
        }
    }
    upload(&ctx, &mut comp, &px);
    comp.ensure_output(W, H);
    let o = |out: &[[f32; 4]], x: usize, y: usize| out[y * W as usize + x][0];
    // Off: both make rays (the window's ray to its right, at y = 40).
    let plain = render(&ctx, &mut comp, &graph(only()));
    assert!(o(&plain, 90, 40) > 0.05, "window ray got {}", o(&plain, 90, 40));
    // On, erode 8 px: the window is wider than that and loses its rays;
    // the 4 px ping keeps them.
    let iso = render(&ctx, &mut comp, &graph(GlintParams { isolate: true, erode: 8.0, ..only() }));
    assert!(o(&iso, 90, 40) < 0.1 * o(&plain, 90, 40), "window ray survived: {}", o(&iso, 90, 40));
    assert!(at(&iso, 40, 0) > 0.5 * at(&plain, 40, 0), "ping ray lost: {} vs {}", at(&iso, 40, 0), at(&plain, 40, 0));
}

#[test]
fn spread_fringes_the_tips_and_shift_warms_the_tail() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 1);
    upload(&ctx, &mut comp, &plate(16.0));
    comp.ensure_output(W, H);
    let px = |o: &[[f32; 4]], dx: i32| {
        let i = (128 * W as i32 + 128 + dx) as usize;
        (o[i][0], o[i][2])
    };
    // Plain: red and blue are the same along the ray.
    let plain = render(&ctx, &mut comp, &graph(only()));
    let (r, b) = px(&plain, 60);
    assert!((r - b).abs() < 0.02 * r.max(1e-3), "plain ray coloured: {r} {b}");
    // Spread: past green's tip red still reaches, blue is gone first.
    let sp = render(&ctx, &mut comp, &graph(GlintParams { spread: 1.0, ..only() }));
    let (r, b) = px(&sp, 76);
    assert!(r > 3.0 * b.max(1e-4) && r > 1e-3, "no red fringe at the tip: {r} {b}");
    // Shift > 0: blue fades first, the tail warms; < 0 the other way.
    let warm = render(&ctx, &mut comp, &graph(GlintParams { shift: 1.0, ..only() }));
    let (r, b) = px(&warm, 50);
    assert!(r > 1.5 * b, "warm tail: {r} {b}");
    let cool = render(&ctx, &mut comp, &graph(GlintParams { shift: -1.0, ..only() }));
    let (r, b) = px(&cool, 50);
    assert!(b > 1.5 * r, "cool tail: {r} {b}");
}
