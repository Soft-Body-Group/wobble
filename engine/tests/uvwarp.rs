//! UV Warp on real pixels: an identity map built the standard way gives the
//! source back pixel for pixel (alpha from the SOURCE, the map's own alpha
//! ignored), a mirrored map mirrors, V runs up unless told otherwise, a
//! lookup between two pixels is their bilinear blend, and Nearest snaps it
//! onto one. Self-skips without a GPU (CI).

use engine::compositor::Compositor;
use engine::decode::ImageInfo;
use engine::graph::{Graph, NodeKind, UvWarpFilter, UvWarpParams};
use engine::timing::GpuTimer;
use half::f16;

const W: u32 = 16;
const H: u32 = 8;

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP uvwarp test: no adapter ({e})");
            None
        }
    }
}

/// Every pixel different, alpha included.
fn source(x: u32, y: u32) -> [f32; 4] {
    [
        x as f32 / W as f32,
        y as f32 / H as f32,
        ((x * 3 + y * 5) % 7) as f32 / 7.0,
        0.25 + ((x + 2 * y) % 4) as f32 * 0.25,
    ]
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

fn graph(p: UvWarpParams) -> Graph {
    let mut g = Graph::default();
    g.nodes.insert(0, NodeKind::read(0, ""));
    g.nodes.insert(1, NodeKind::read(1, ""));
    g.nodes.insert(2, NodeKind::UvWarp(p));
    g.edges.insert((2, 0), 0);
    g.edges.insert((2, 1), 1);
    g.view = Some(2);
    g
}

fn close(a: [f32; 4], b: [f32; 4]) -> bool {
    (0..4).all(|c| (a[c] - b[c]).abs() <= 3e-3)
}

/// Pixel centres, the standard way: u = (x + 0.5) / w, v = 1 - (y + 0.5) / h.
fn u_of(x: f32) -> f32 {
    (x + 0.5) / W as f32
}
fn v_of(y: f32) -> f32 {
    1.0 - (y + 0.5) / H as f32
}

/// Render a map `m(x, y) -> (u, v)` and check every pixel against `want`.
fn check(
    ctx: &engine::GpuContext,
    comp: &mut Compositor,
    p: UvWarpParams,
    m: impl Fn(u32, u32) -> [f32; 2],
    want: impl Fn(u32, u32) -> [f32; 4],
    what: &str,
) {
    // The map's alpha is 0 everywhere: if it leaked, alpha would read 0.
    upload(ctx, comp, 1, |x, y| {
        let [u, v] = m(x, y);
        [u, v, 0.0, 0.0]
    });
    let out = frame(ctx, comp, &graph(p));
    for y in 0..H {
        for x in 0..W {
            let (got, want) = (out[(y * W + x) as usize], want(x, y));
            assert!(close(got, want), "{what}: ({x}, {y}) rendered {got:?}, want {want:?}");
        }
    }
}

fn setup() -> Option<(engine::GpuContext, Compositor)> {
    let ctx = ctx()?;
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp, 0, source);
    Some((ctx, comp))
}

#[test]
fn an_identity_map_gives_the_source_back() {
    let Some((ctx, mut comp)) = setup() else { return };
    for filter in [UvWarpFilter::Bilinear, UvWarpFilter::Nearest] {
        let p = UvWarpParams { filter, ..Default::default() };
        check(&ctx, &mut comp, p, |x, y| [u_of(x as f32), v_of(y as f32)], source, "identity");
    }
}

#[test]
fn a_mirrored_map_mirrors_and_v_runs_up() {
    let Some((ctx, mut comp)) = setup() else { return };
    let p = UvWarpParams::default();
    check(
        &ctx,
        &mut comp,
        p,
        |x, y| [1.0 - u_of(x as f32), v_of(y as f32)],
        |x, y| source(W - 1 - x, y),
        "mirror in x",
    );
    // A map written top-down (v = (y + 0.5) / h) read the standard way is
    // upside down ...
    let top_down = |x: u32, y: u32| [u_of(x as f32), (y as f32 + 0.5) / H as f32];
    check(&ctx, &mut comp, p, top_down, |x, y| source(x, H - 1 - y), "v up flips a top-down map");
    // ... and read with v_up off it is the identity.
    let down = UvWarpParams { v_up: false, ..p };
    check(&ctx, &mut comp, down, top_down, source, "v down");
}

#[test]
fn between_two_pixels_is_their_blend_and_nearest_picks_one() {
    let Some((ctx, mut comp)) = setup() else { return };
    // Halfway between pixel x and x + 1 (stop one short of the edge).
    let half = |x: u32, y: u32| [(x.min(W - 2) as f32 + 1.0) / W as f32, v_of(y as f32)];
    let blend = |x: u32, y: u32| {
        let (a, b) = (source(x.min(W - 2), y), source(x.min(W - 2) + 1, y));
        std::array::from_fn(|c| (a[c] + b[c]) * 0.5)
    };
    check(&ctx, &mut comp, UvWarpParams::default(), half, blend, "bilinear");
    let near = UvWarpParams { filter: UvWarpFilter::Nearest, ..Default::default() };
    check(&ctx, &mut comp, near, half, |x, y| source(x.min(W - 2) + 1, y), "nearest");
}

#[test]
fn no_map_passes_the_source() {
    let Some((ctx, mut comp)) = setup() else { return };
    let mut g = graph(UvWarpParams::default());
    g.edges.remove(&(2, 1));
    let out = frame(&ctx, &mut comp, &g);
    for y in 0..H {
        for x in 0..W {
            assert!(close(out[(y * W + x) as usize], source(x, y)), "({x}, {y}) moved");
        }
    }
}
