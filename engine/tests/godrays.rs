//! Headless integration for the Godrays node through the real Compositor:
//! shafts stream from the source point, length 0 casts none, the source
//! matte decides what casts them, and moving the source moves them.
//! Self-skips without a GPU (CI).

use engine::compositor::Compositor;
use engine::decode::ImageInfo;
use engine::graph::{GlowMode, GodraysParams, Graph, NodeKind};
use engine::timing::GpuTimer;
use half::f16;

const W: u32 = 256;
const H: u32 = 256;

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP godrays test: no adapter ({e})");
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


fn px(v: f32) -> [f16; 4] {
    [f16::from_f32(v), f16::from_f32(v), f16::from_f32(v), f16::ONE]
}

/// Two bright discs high in a black frame, at x = 64 and x = 192.
fn plate() -> Vec<[f16; 4]> {
    let mut out = Vec::new();
    for y in 0..H as i32 {
        for x in 0..W as i32 {
            let hot = [(64, 48), (192, 48)]
                .iter()
                .any(|(cx, cy)| (x - cx).pow(2) + (y - cy).pow(2) < 64);
            out.push(px(if hot { 8.0 } else { 0.0 }));
        }
    }
    out
}

/// White on the left half, black on the right.
fn left_matte() -> Vec<[f16; 4]> {
    (0..H).flat_map(|_| (0..W).map(|x| px(if x < W / 2 { 1.0 } else { 0.0 }))).collect()
}

fn graph(p: GodraysParams, matte: bool) -> Graph {
    let mut g = Graph::default();
    g.nodes.insert(1, NodeKind::read(0, ""));
    g.nodes.insert(2, NodeKind::Godrays(p));
    g.edges.insert((2, 0), 1);
    if matte {
        g.nodes.insert(3, NodeKind::read(1, ""));
        g.edges.insert((2, 1), 3);
    }
    g.view = Some(2);
    g
}

fn at(img: &[[f32; 4]], x: usize, y: usize) -> f32 {
    img[y * W as usize + x][0]
}

/// The source sits above both discs, so each disc's shafts run straight
/// down from it.
fn base() -> GodraysParams {
    GodraysParams {
        center: [0.5, -0.5],
        length: 0.8,
        falloff: 0.0,
        threshold: 0.5,
        knee: 0.1,
        intensity: 4.0,
        tint: [1.0; 3],
        mode: GlowMode::Only,
        ..Default::default()
    }
}

#[test]
fn shafts_stream_from_the_source_and_the_matte_picks_what_casts() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 2);
    upload(&ctx, &mut comp, 0, &plate());
    upload(&ctx, &mut comp, 1, &left_matte());
    comp.ensure_output(W, H);

    // Under each disc, on the line away from the source: lit.
    let rays = render(&ctx, &mut comp, &graph(base(), false));
    // The line from (0.5, -0.5) through the left disc continues down-left.
    let (lx, ly) = (54, 90);
    let (rx, ry) = (202, 90);
    let lit_l = at(&rays, lx, ly);
    let lit_r = at(&rays, rx, ry);
    assert!(lit_l > 0.02 && lit_r > 0.02, "shafts below both discs: {lit_l} {lit_r}");
    // Beside a shaft, between the two: dark.
    let gap = at(&rays, 128, 90);
    assert!(gap < lit_l * 0.1, "between the shafts stays dark: {gap} vs {lit_l}");

    // Length 0 gathers nothing beyond the pixel itself.
    let none = render(&ctx, &mut comp, &graph(GodraysParams { length: 0.0, ..base() }, false));
    assert!(at(&none, lx, ly) < 1e-3, "length 0 casts no shaft: {}", at(&none, lx, ly));

    // The matte covers the left disc only: the right shaft goes.
    let m = render(&ctx, &mut comp, &graph(base(), true));
    assert!(at(&m, lx, ly) > lit_l * 0.9, "the matted disc still casts: {}", at(&m, lx, ly));
    assert!(at(&m, rx, ry) < lit_r * 0.05, "the unmatted disc casts nothing: {}", at(&m, rx, ry));

    // A source moved to the far left sends the left disc's shaft right.
    let moved = render(&ctx, &mut comp, &graph(GodraysParams { center: [-1.0, 0.18], ..base() }, false));
    assert!(at(&moved, lx, ly) < lit_l * 0.2, "the old shaft moved away: {}", at(&moved, lx, ly));
    assert!(at(&moved, 100, 48) > 0.02, "a shaft now runs right of the disc: {}", at(&moved, 100, 48));

    // Add mode lays the shafts over the plate.
    let add = render(&ctx, &mut comp, &graph(GodraysParams { mode: GlowMode::Add, ..base() }, false));
    assert!((at(&add, 64, 48) - (8.0 + at(&rays, 64, 48))).abs() < 0.1, "add = plate + rays");
}

#[test]
fn moving_the_source_is_not_a_recompile() {
    let a = graph(base(), false);
    let b = graph(GodraysParams { center: [0.1, 0.9], length: 0.3, ..base() }, false);
    assert_eq!(a.topology_hash(), b.topology_hash());
}

#[test]
fn quality_is_a_uniform_and_any_count_casts_the_same_shafts() {
    let a = graph(base(), false);
    let b = graph(GodraysParams { samples: 48, dither: 0.0, ..base() }, false);
    assert_eq!(a.topology_hash(), b.topology_hash());

    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 2);
    upload(&ctx, &mut comp, 0, &plate());
    upload(&ctx, &mut comp, 1, &left_matte());
    comp.ensure_output(W, H);
    let fine = render(&ctx, &mut comp, &graph(GodraysParams { samples: 48, ..base() }, false));
    for (samples, dither) in [(4, 0.0), (4, 1.0), (16, 0.0)] {
        let q = render(&ctx, &mut comp, &graph(GodraysParams { samples, dither, ..base() }, false));
        // A shaft's level does not depend on how finely it is gathered.
        let (f, g) = (at(&fine, 54, 90), at(&q, 54, 90));
        assert!((f - g).abs() < f * 0.25, "samples {samples} dither {dither}: {g} vs {f}");
        assert!(at(&q, 128, 90) < f * 0.1, "between the shafts stays dark at {samples}");
    }
}
