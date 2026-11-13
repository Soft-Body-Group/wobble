//! ColorCompress on real pixels: the fused shader has to draw the same
//! straight line the params describe.
//!
//! The node solves for that line on the CPU (`pack` writes a slope and an
//! intercept, not the four knobs), so the thing worth pinning end to end is
//! that the picture agrees with `ColorCompressParams::apply` — if the two
//! ever drift, every panel readout in the app is lying about the frame.
//! Self-skips without a GPU (CI).

use engine::compositor::Compositor;
use engine::decode::ImageInfo;
use engine::graph::{ColorCompressParams, Graph, NodeKind};
use engine::timing::GpuTimer;
use half::f16;

const W: u32 = 32;
const H: u32 = 4;

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP colorcompress test: no adapter ({e})");
            None
        }
    }
}

/// A ramp with a different slope per channel, so a per-channel line cannot
/// hide behind a grey one: r rises to 2, g to 1, b falls from 1 to 0.
fn upload_ramp(ctx: &engine::GpuContext, comp: &mut Compositor) {
    let info = ImageInfo { width: W, height: H };
    comp.ensure_reader(0, info);
    let px: Vec<[f16; 4]> = (0..W * H)
        .map(|i| {
            let t = (i % W) as f32 / (W - 1) as f32;
            [
                f16::from_f32(t * 2.0),
                f16::from_f32(t),
                f16::from_f32(1.0 - t),
                f16::ONE,
            ]
        })
        .collect();
    ctx.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &comp.reader_texture(0).unwrap().texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&px),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(W * 8),
            rows_per_image: Some(H),
        },
        wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
    );
}

fn row(ctx: &engine::GpuContext, comp: &mut Compositor, graph: &Graph) -> Vec<[f32; 4]> {
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
    let halves: &[f16] = bytemuck::cast_slice(&data[..(W * 8) as usize]);
    let out: Vec<[f32; 4]> = (0..W as usize)
        .map(|x| {
            [
                halves[x * 4].to_f32(),
                halves[x * 4 + 1].to_f32(),
                halves[x * 4 + 2].to_f32(),
                halves[x * 4 + 3].to_f32(),
            ]
        })
        .collect();
    drop(data);
    staging.unmap();
    out
}

fn graph(p: ColorCompressParams) -> Graph {
    let mut g = Graph::default();
    g.nodes.insert(0, NodeKind::read(0, ""));
    g.nodes.insert(1, NodeKind::ColorCompress(p));
    g.edges.insert((1, 0), 0);
    g.view = Some(1);
    g
}

/// The plate as it went in, so the test knows what it is grading.
fn source(x: usize) -> [f32; 3] {
    let t = x as f32 / (W - 1) as f32;
    [t * 2.0, t, 1.0 - t]
}

fn close(a: f32, b: f32) -> bool {
    // Half floats: a tenth of a percent of the value, plus a floor.
    (a - b).abs() <= 0.004 + 0.004 * b.abs()
}

#[test]
fn the_rendered_picture_is_the_line_the_params_describe() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload_ramp(&ctx, &mut comp);

    // A different line per channel, and one of them inverted.
    let p = ColorCompressParams {
        black: [0.1, 0.0, 0.8],
        white: [1.8, 0.5, 0.2],
        lift: [0.0, 0.25, 0.0],
        mult: [1.0, 0.75, 1.0],
    };
    let out = row(&ctx, &mut comp, &graph(p));
    for x in 0..W as usize {
        let want = p.apply(source(x));
        for c in 0..3 {
            assert!(
                close(out[x][c], want[c]),
                "x={x} ch{c}: rendered {} vs the params' own {} (src {:?})",
                out[x][c],
                want[c],
                source(x)
            );
        }
        assert!(close(out[x][3], 1.0), "alpha is not a colour: {:?}", out[x]);
    }

    // Two levels the ramp hits EXACTLY, so they can be read off the picture
    // rather than interpolated: channel 1 is 0.0 at x=0 — its black point,
    // which must land on its lift — and 1.0 at the far end, which is twice
    // its white point (0.5) and must therefore land twice as far past the
    // lift as the mult does.
    assert!(close(out[0][1], 0.25), "black did not land on lift: {:?}", out[0]);
    let last = W as usize - 1;
    assert!(close(out[last][1], 1.25), "the line stopped at the white point: {:?}", out[last]);
}

/// A fresh node is an identity — a node dropped into a chain must not move
/// the picture until it is asked to.
#[test]
fn a_default_node_renders_the_plate_unchanged() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload_ramp(&ctx, &mut comp);
    let out = row(&ctx, &mut comp, &graph(ColorCompressParams::default()));
    for x in 0..W as usize {
        let src = source(x);
        for c in 0..3 {
            assert!(close(out[x][c], src[c]), "x={x} ch{c}: {:?} vs {src:?}", out[x]);
        }
    }
}

/// Nothing is clipped at either end: the plate reaches 2.0 in red, and a
/// node compressing 0..1 has to carry that above its white point rather
/// than flattening it.
#[test]
fn highlights_above_the_white_point_survive() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload_ramp(&ctx, &mut comp);
    let p = ColorCompressParams {
        lift: [0.1; 3],
        mult: [0.6; 3],
        ..Default::default()
    };
    let out = row(&ctx, &mut comp, &graph(p));
    let last = W as usize - 1;
    // Red at the far end is 2.0 in, well past the white point of 1.0.
    assert!(out[last][0] > 0.6, "a highlight was clipped at the white point: {:?}", out[last]);
    assert!(close(out[last][0], p.apply(source(last))[0]), "{:?}", out[last]);
    // And the slope is the same on both sides of the white point.
    let step = |x: usize| out[x + 1][0] - out[x][0];
    assert!(close(step(2), step(last - 2)), "the line bent: {} vs {}", step(2), step(last - 2));
}

/// Upload any plate, one colour per column, the same on every row.
fn upload_px(ctx: &engine::GpuContext, comp: &mut Compositor, f: impl Fn(usize) -> [f32; 3]) {
    let info = ImageInfo { width: W, height: H };
    comp.ensure_reader(0, info);
    let px: Vec<[f16; 4]> = (0..W * H)
        .map(|i| {
            let c = f((i % W) as usize);
            [f16::from_f32(c[0]), f16::from_f32(c[1]), f16::from_f32(c[2]), f16::ONE]
        })
        .collect();
    ctx.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &comp.reader_texture(0).unwrap().texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&px),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(W * 8),
            rows_per_image: Some(H),
        },
        wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
    );
}

/// Negatives are values, not errors — below-black in a scene-linear plate,
/// out-of-gamut after a primaries change, or somewhere a line was ASKED to
/// go. Both ways through the node on real pixels: negative pixels in come
/// out on the line (not at 0), and a line that runs below zero writes
/// negatives out. A `max(…, 0)` anywhere in the fused pass fails both.
#[test]
fn negatives_are_not_clamped_on_the_way_in_or_out() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    // A plate that crosses zero in every channel: r -1..1, g -0.5..0.5,
    // b 0.2 down to -0.8.
    let plate = |x: usize| {
        let t = x as f32 / (W - 1) as f32;
        [2.0 * t - 1.0, t - 0.5, 0.2 - t]
    };
    upload_px(&ctx, &mut comp, plate);

    // In: the identity must hand the negatives back untouched.
    let out = row(&ctx, &mut comp, &graph(ColorCompressParams::default()));
    for x in [0, 3, W as usize / 2 - 2] {
        let want = plate(x);
        for i in 0..3 {
            assert!(
                close(out[x][i], want[i]),
                "a negative input was changed at column {x}: {:?}, want {want:?}",
                out[x]
            );
        }
    }
    assert!(out[0][0] < -0.9, "negative red came out at {}", out[0][0]);

    // Out: lift below zero puts black below zero, and it must stay there.
    let p = ColorCompressParams {
        lift: [-0.25; 3],
        mult: [0.75; 3],
        ..Default::default()
    };
    let out = row(&ctx, &mut comp, &graph(p));
    for x in 0..W as usize {
        let want = p.apply(plate(x));
        for i in 0..3 {
            assert!(
                close(out[x][i], want[i]),
                "column {x} channel {i}: rendered {}, the line says {}",
                out[x][i],
                want[i]
            );
        }
    }
    assert!(out[0][0] < -1.0, "the line runs to -1.25 at red's end, rendered {}", out[0][0]);
}
