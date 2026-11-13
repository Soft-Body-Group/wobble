//! Headless integration for the EdgeExtend node through the real
//! Compositor: spread fills the frame with the nearest edge colours, expand
//! stops at its distance, an inset replaces a dark rim with the colour from
//! inside it, and a wired matte replaces the alpha as the edge.
//! Self-skips without a GPU (CI).

use engine::compositor::Compositor;
use engine::decode::ImageInfo;
use engine::graph::{EdgeExtendParams, ExtendMode, Graph, NodeKind};
use engine::timing::GpuTimer;
use half::f16;

const W: u32 = 128;
const H: u32 = 128;
/// The disc's centre and radius, px.
const C: (f32, f32) = (64.3, 64.0);
const R: f32 = 20.0;

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP edgeextend test: no adapter ({e})");
            None
        }
    }
}

fn upload(ctx: &engine::GpuContext, comp: &mut Compositor, slot: usize, px: &[[f32; 4]]) {
    let px: Vec<[f16; 4]> = px.iter().map(|p| p.map(f16::from_f32)).collect();
    comp.ensure_reader(slot, ImageInfo { width: W, height: H });
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

fn render(ctx: &engine::GpuContext, comp: &mut Compositor, graph: &Graph) -> Vec<[f32; 4]> {
    assert!(comp.set_graph(graph), "fused shader failed to compile");
    let timer = GpuTimer::new(ctx);
    // Warm-up encode: stage textures are allocated inside encode_comp.
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

/// Distance of a pixel centre from the disc's centre.
fn dist(x: u32, y: u32) -> f32 {
    ((x as f32 + 0.5 - C.0).powi(2) + (y as f32 + 0.5 - C.1).powi(2)).sqrt()
}

/// Anti-aliased disc coverage.
fn cover(x: u32, y: u32) -> f32 {
    (R + 0.5 - dist(x, y)).clamp(0.0, 1.0)
}

/// Straight colour: red left of centre, blue right.
fn colour(x: u32) -> [f32; 3] {
    if (x as f32) < C.0 { [1.0, 0.0, 0.0] } else { [0.0, 0.0, 1.0] }
}

/// A premultiplied disc, red | blue, over transparent black. `rim` px
/// inside the edge are darkened to a tenth: a dark key edge.
fn disc(rim: f32) -> Vec<[f32; 4]> {
    let mut out = Vec::new();
    for y in 0..H {
        for x in 0..W {
            let a = cover(x, y);
            let dark = if rim > 0.0 && dist(x, y) > R - rim { 0.1 } else { 1.0 };
            let c = colour(x);
            out.push([c[0] * a * dark, c[1] * a * dark, c[2] * a * dark, a]);
        }
    }
    out
}

fn graph(p: EdgeExtendParams, matte: bool) -> Graph {
    let mut g = Graph::default();
    g.nodes.insert(1, NodeKind::read(0, ""));
    g.nodes.insert(2, NodeKind::EdgeExtend(p));
    g.edges.insert((2, 0), 1);
    if matte {
        g.nodes.insert(3, NodeKind::read(1, ""));
        g.edges.insert((2, 1), 3);
    }
    g.view = Some(2);
    g
}

fn at(img: &[[f32; 4]], x: u32, y: u32) -> [f32; 4] {
    img[(y * W + x) as usize]
}

/// A point `d` px outside the edge, straight left or right of centre.
fn left(d: f32) -> u32 {
    (C.0 - R - d) as u32
}
fn right(d: f32) -> u32 {
    (C.0 + R + d) as u32
}

#[test]
fn spread_fills_the_frame_and_expand_stops_at_its_distance() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 2);
    let plate = disc(0.0);
    upload(&ctx, &mut comp, 0, &plate);
    comp.ensure_output(W, H);
    let y = C.1 as u32;

    // Spread: the far corners carry the nearest edge's colour, straight.
    let s = render(&ctx, &mut comp, &graph(EdgeExtendParams::default(), false));
    let l = at(&s, 2, y);
    let r = at(&s, W - 3, y);
    assert!(l[0] > 0.9 && l[2] < 0.1, "left of the disc is red: {l:?}");
    assert!(r[2] > 0.9 && r[0] < 0.1, "right of the disc is blue: {r:?}");
    assert_eq!(l[3], 0.0, "keep alpha keeps the transparent outside");
    // The middle of the disc is untouched.
    let mid = at(&s, 50, y);
    assert!((mid[0] - 1.0).abs() < 1e-3 && mid[3] > 0.999, "inside is the plate: {mid:?}");
    // The anti-aliased rim is straight colour now, not darkened by alpha.
    let rim = (0..W).find(|&x| { let a = cover(x, y); a > 0.1 && a < 0.6 }).unwrap();
    assert!(at(&s, rim, y)[0].max(at(&s, rim, y)[2]) > 0.9, "the rim unpremultiplied: {:?}", at(&s, rim, y));

    // Expand 6 px, hard: filled at 3 px out, the plate's black at 15.
    let p = EdgeExtendParams {
        mode: ExtendMode::Expand,
        distance: 6.0,
        softness: 0.0,
        keep_alpha: false,
        ..Default::default()
    };
    let e = render(&ctx, &mut comp, &graph(p, false));
    let near = at(&e, left(3.0), y);
    let far = at(&e, left(15.0), y);
    assert!(near[0] > 0.9 && near[3] > 0.999, "3 px out is filled, alpha 1: {near:?}");
    assert!(far[0] < 1e-3 && far[3] < 1e-3, "15 px out is untouched: {far:?}");
    assert!(at(&e, right(3.0), y)[2] > 0.9, "the right side spreads blue");

    // Softness ramps the reach instead of cutting it.
    let soft = render(&ctx, &mut comp, &graph(EdgeExtendParams { softness: 8.0, ..p }, false));
    let a = at(&soft, left(10.0), y)[3];
    assert!(a > 0.05 && a < 0.95, "inside the soft band the coverage ramps: {a}");
}

#[test]
fn an_inset_replaces_a_dark_rim_with_the_colour_inside_it() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 2);
    upload(&ctx, &mut comp, 0, &disc(3.0));
    comp.ensure_output(W, H);
    let y = C.1 as u32;
    // 1.5 px inside the edge: in the dark rim, fully opaque.
    let x = left(-1.5);
    assert!(cover(x, y) > 0.999 && dist(x, y) > R - 3.0);

    let p = EdgeExtendParams { mode: ExtendMode::Expand, distance: 0.0, softness: 0.0, ..Default::default() };
    let plain = render(&ctx, &mut comp, &graph(p, false));
    assert!(at(&plain, x, y)[0] < 0.2, "no inset keeps the rim: {:?}", at(&plain, x, y));

    let fixed = EdgeExtendParams { inset: 5.0, premult_out: true, ..p };
    let f = render(&ctx, &mut comp, &graph(fixed, false));
    let v = at(&f, x, y);
    assert!(v[0] > 0.85, "inset 5 pulls the clean red out over the rim: {v:?}");
    assert!((v[3] - 1.0).abs() < 1e-3, "alpha kept");
    // Premultiplied out: the rim pixel's colour times its alpha.
    let edge = (0..W).find(|&x| { let a = cover(x, y); a > 0.3 && a < 0.7 }).unwrap();
    let ev = at(&f, edge, y);
    assert!((ev[0] - 0.9 * ev[3]).abs() < 0.12 * ev[3].max(0.1) + 0.05, "premultiplied by alpha: {ev:?}");
    // Deep inside: untouched.
    assert!((at(&f, 50, y)[0] - 1.0).abs() < 1e-3);
}

#[test]
fn a_wired_matte_is_the_edge_and_the_colour_is_read_straight() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 2);
    // A straight plate: the disc's colours inside, green garbage outside,
    // opaque everywhere.
    let mut plate = Vec::new();
    let mut matte = Vec::new();
    for y in 0..H {
        for x in 0..W {
            let a = cover(x, y);
            let c = if a > 0.0 { colour(x) } else { [0.0, 1.0, 0.0] };
            plate.push([c[0], c[1], c[2], 1.0]);
            matte.push([a, a, a, 1.0]);
        }
    }
    upload(&ctx, &mut comp, 0, &plate);
    upload(&ctx, &mut comp, 1, &matte);
    comp.ensure_output(W, H);
    let y = C.1 as u32;
    let s = render(&ctx, &mut comp, &graph(EdgeExtendParams::default(), true));
    let l = at(&s, 4, y);
    assert!(l[0] > 0.9 && l[1] < 0.1, "the garbage is replaced by the edge's red: {l:?}");
    assert_eq!(l[3], 1.0, "the plate's own alpha is kept");
    let r = at(&s, W - 5, 10);
    assert!(r[1] < 0.1, "no green survives outside the matte: {r:?}");
}

#[test]
fn no_param_is_a_recompile() {
    let a = graph(EdgeExtendParams::default(), false);
    let b = graph(
        EdgeExtendParams {
            mode: ExtendMode::Expand,
            distance: 40.0,
            inset: 3.0,
            softness: 5.0,
            premultiplied: false,
            keep_alpha: false,
            premult_out: true,
        },
        false,
    );
    assert_eq!(a.topology_hash(), b.topology_hash());
}
