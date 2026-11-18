//! Morph Dissolve on real pixels, with vectors we write ourselves so the
//! right answer is known: each end is its input untouched, unanalysed it is
//! a plain dissolve, a picture translated by `d` meets itself half way, and
//! a square moving over a still background neither ghosts nor leaves a
//! hole. Self-skips without a GPU.

use engine::compositor::Compositor;
use engine::decode::ImageInfo;
use engine::graph::{Graph, MorphDissolveParams, NodeKind};
use engine::timing::GpuTimer;
use half::f16;

const W: u32 = 96;
const H: u32 = 48;
const A: usize = 0;
const B: usize = 1;
const VECTORS: usize = 2;

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP morph dissolve test: no adapter ({e})");
            None
        }
    }
}

fn upload(ctx: &engine::GpuContext, comp: &mut Compositor, slot: usize, f: impl Fn(f32, f32) -> [f32; 4]) {
    comp.ensure_reader(slot, ImageInfo { width: W, height: H });
    let px: Vec<[f16; 4]> =
        (0..W * H).map(|i| f((i % W) as f32 + 0.5, (i / W) as f32 + 0.5).map(f16::from_f32)).collect();
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
            out.push([row[x * 4].to_f32(), row[x * 4 + 1].to_f32(), row[x * 4 + 2].to_f32(), row[x * 4 + 3].to_f32()]);
        }
    }
    drop(data);
    staging.unmap();
    out
}

fn graph(p: MorphDissolveParams) -> Graph {
    let mut g = Graph::default();
    g.nodes.insert(0, NodeKind::read(A, ""));
    g.nodes.insert(1, NodeKind::read(B, ""));
    g.nodes.insert(2, NodeKind::MorphDissolve(p));
    g.edges.insert((2, 0), 0);
    g.edges.insert((2, 1), 1);
    g.view = Some(2);
    g
}

fn loaded(mix: f32) -> MorphDissolveParams {
    MorphDissolveParams { mix, loaded: true, ..MorphDissolveParams::new(VECTORS) }
}

/// Smooth, and different everywhere, so a wrong lookup cannot hide.
fn texture(x: f32, y: f32) -> [f32; 4] {
    let u = x / W as f32;
    let v = y / H as f32;
    [
        0.5 + 0.4 * (u * 9.0).sin() * (v * 4.0).cos(),
        0.5 + 0.4 * (u * 5.0 + v * 7.0).cos(),
        0.5 + 0.3 * (v * 11.0 - u * 3.0).sin(),
        1.0,
    ]
}

/// Largest channel difference, away from the borders a shift exposes.
fn worst(got: &[[f32; 4]], want: impl Fn(f32, f32) -> [f32; 4], margin: u32) -> f32 {
    let mut w = 0f32;
    for y in margin..H - margin {
        for x in margin..W - margin {
            let g = got[(y * W + x) as usize];
            let t = want(x as f32 + 0.5, y as f32 + 0.5);
            for c in 0..4 {
                w = w.max((g[c] - t[c]).abs());
            }
        }
    }
    w
}

const D: [f32; 2] = [8.0, -4.0];

fn translated(ctx: &engine::GpuContext) -> Compositor {
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload(ctx, &mut comp, A, texture);
    upload(ctx, &mut comp, B, |x, y| texture(x - D[0], y - D[1]));
    upload(ctx, &mut comp, VECTORS, |_, _| [D[0], D[1], -D[0], -D[1]]);
    comp
}

#[test]
fn each_end_is_its_input_exactly_whatever_the_vectors_say() {
    let Some(ctx) = ctx() else { return };
    let mut comp = translated(&ctx);
    let a = frame(&ctx, &mut comp, &graph(loaded(0.0)));
    assert!(worst(&a, texture, 0) < 2e-3, "mix 0 is A: off by {}", worst(&a, texture, 0));
    let b = frame(&ctx, &mut comp, &graph(loaded(1.0)));
    let e = worst(&b, |x, y| texture(x - D[0], y - D[1]), 0);
    assert!(e < 2e-3, "mix 1 is B: off by {e}");
}

#[test]
fn unanalysed_it_is_a_plain_dissolve() {
    let Some(ctx) = ctx() else { return };
    let mut comp = translated(&ctx);
    let p = MorphDissolveParams { mix: 0.25, ..MorphDissolveParams::new(VECTORS) };
    let got = frame(&ctx, &mut comp, &graph(p));
    let e = worst(
        &got,
        |x, y| {
            let (a, b) = (texture(x, y), texture(x - D[0], y - D[1]));
            std::array::from_fn(|c| a[c] * 0.75 + b[c] * 0.25)
        },
        0,
    );
    assert!(e < 2e-3, "off a dissolve by {e}");
}

#[test]
fn a_translated_picture_meets_itself_half_way() {
    let Some(ctx) = ctx() else { return };
    let mut comp = translated(&ctx);
    for t in [0.25f32, 0.5, 0.75] {
        let got = frame(&ctx, &mut comp, &graph(loaded(t)));
        let e = worst(&got, |x, y| texture(x - t * D[0], y - t * D[1]), 10);
        assert!(e < 0.01, "mix {t}: off the in-between picture by {e}");
    }
    // Warp off: the same vectors, and the two pictures simply dissolve.
    let got = frame(&ctx, &mut comp, &graph(MorphDissolveParams { warp: 0.0, ..loaded(0.5) }));
    let e = worst(
        &got,
        |x, y| {
            let (a, b) = (texture(x, y), texture(x - D[0], y - D[1]));
            std::array::from_fn(|c| (a[c] + b[c]) * 0.5)
        },
        0,
    );
    assert!(e < 2e-3, "warp 0 is the dissolve: off by {e}");
}

/// A bright square moving right over a still dark background, flows as an
/// ideal solver gives them. Half way the square is whole, at the middle
/// position; behind it the background is back and ahead of it too.
#[test]
fn a_moving_square_neither_ghosts_nor_leaves_a_hole() {
    let Some(ctx) = ctx() else { return };
    const X0: f32 = 24.0;
    const SIDE: f32 = 24.0;
    const MOVE: f32 = 16.0;
    let inside = |x: f32, y: f32, at: f32| x >= at && x < at + SIDE && y >= 12.0 && y < 36.0;
    let bg = [0.1, 0.1, 0.1, 1.0];
    let fg = [1.0, 0.8, 0.6, 1.0];
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp, A, |x, y| if inside(x, y, X0) { fg } else { bg });
    upload(&ctx, &mut comp, B, |x, y| if inside(x, y, X0 + MOVE) { fg } else { bg });
    upload(&ctx, &mut comp, VECTORS, |x, y| {
        let ab = if inside(x, y, X0) { MOVE } else { 0.0 };
        let ba = if inside(x, y, X0 + MOVE) { -MOVE } else { 0.0 };
        [ab, 0.0, ba, 0.0]
    });
    let got = frame(&ctx, &mut comp, &graph(loaded(0.5)));
    let at = |x: f32, y: f32| got[(y as u32 * W + x as u32) as usize];
    let mid = X0 + MOVE / 2.0;
    // Well inside the square at its half-way place.
    for x in [mid + 3.0, mid + SIDE / 2.0, mid + SIDE - 4.0] {
        let v = at(x, 24.0);
        assert!((v[0] - fg[0]).abs() < 0.05, "x {x}: the square is {v:?} at its middle position");
    }
    // Just uncovered behind it, and far from it: background, no ghost of
    // where it was or where it is going.
    for x in [X0 + 2.0, X0 + MOVE + SIDE - 2.0, 8.0, 88.0] {
        let v = at(x, 24.0);
        assert!((v[0] - bg[0]).abs() < 0.05, "x {x}: background reads {v:?}");
    }
}
