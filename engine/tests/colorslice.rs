//! ColorSlice on real pixels: the fused shader must draw exactly what the
//! CPU model (`ColorSliceParams::apply`) says, across the whole hue circle,
//! at two exposures and on greys — and the highlight must show the vector,
//! and only while the node is armed as the view target.
//!
//! The model is where the node's behaviour is argued and unit-tested; this
//! pins that the WGSL is that model and not a cousin of it. Self-skips
//! without a GPU (CI).

use engine::color::Colorspace;
use engine::compositor::Compositor;
use engine::decode::ImageInfo;
use engine::graph::{ColorSliceParams, Graph, NodeKind, SliceVector};
use engine::timing::GpuTimer;
use half::f16;

const W: u32 = 72;
const H: u32 = 3;
const WS: Colorspace = Colorspace::ACESCG;

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP colorslice test: no adapter ({e})");
            None
        }
    }
}

/// A colour on the RGB hue circle at `deg`, with the given brightest
/// channel and colourfulness.
fn on_circle(deg: f32, mx: f32, q: f32) -> [f32; 3] {
    let mn = mx * (1.0 - q);
    let x = (deg / 60.0).rem_euclid(6.0);
    let f = x - x.floor();
    let (up, down) = (mn + (mx - mn) * f, mx - (mx - mn) * f);
    match x as i32 {
        0 => [mx, up, mn],
        1 => [down, mx, mn],
        2 => [mn, mx, up],
        3 => [mn, down, mx],
        4 => [up, mn, mx],
        _ => [mx, mn, down],
    }
}

/// The plate: row 0 walks the hue circle at a normal exposure, row 1 does
/// it four stops up (where the depths act), row 2 is a grey ramp.
fn source(x: u32, y: u32) -> [f32; 3] {
    let deg = x as f32 / W as f32 * 360.0;
    match y {
        0 => on_circle(deg, 0.6, 0.7),
        1 => on_circle(deg, 9.6, 0.55),
        _ => [(x as f32 + 1.0) / W as f32 * 2.0; 3],
    }
}

fn upload(ctx: &engine::GpuContext, comp: &mut Compositor) {
    comp.ensure_reader(0, ImageInfo { width: W, height: H });
    let px: Vec<[f16; 4]> = (0..W * H)
        .map(|i| {
            let c = source(i % W, i / W);
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

/// Render `graph` and read the whole frame back, row by row.
fn frame(ctx: &engine::GpuContext, comp: &mut Compositor, graph: &Graph) -> Vec<Vec<[f32; 4]>> {
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
    let out = (0..H as usize)
        .map(|y| {
            let row: &[f16] = bytemuck::cast_slice(&data[y * padded as usize..][..(W * 8) as usize]);
            (0..W as usize)
                .map(|x| {
                    [
                        row[x * 4].to_f32(),
                        row[x * 4 + 1].to_f32(),
                        row[x * 4 + 2].to_f32(),
                        row[x * 4 + 3].to_f32(),
                    ]
                })
                .collect()
        })
        .collect();
    drop(data);
    staging.unmap();
    out
}

fn graph(p: ColorSliceParams) -> Graph {
    let mut g = Graph::default();
    g.working = WS;
    g.nodes.insert(0, NodeKind::read(0, ""));
    g.nodes.insert(1, NodeKind::ColorSlice(p));
    g.edges.insert((1, 0), 0);
    g.view = Some(1);
    g
}

/// Half floats, plus the Oklab round trip: a small floor and a relative
/// share of the value.
fn close(a: f32, b: f32) -> bool {
    (a - b).abs() <= 0.006 + 0.006 * b.abs()
}

/// A setting that exercises every control at once.
fn busy() -> ColorSliceParams {
    let mut p = ColorSliceParams {
        sat: 1.2,
        density: 0.3,
        sat_depth: 0.6,
        den_depth: 1.4,
        sat_balance: 0.5,
        ..Default::default()
    };
    p.vectors[0] = SliceVector { center: 8.0, hue: -15.0, sat: 1.6, density: 0.5 }; // red
    p.vectors[1] = SliceVector { center: -5.0, hue: 6.0, sat: 0.8, density: -0.4 }; // skin
    p.vectors[3] = SliceVector { center: 12.0, hue: 25.0, sat: 0.5, density: 0.0 }; // green
    p.vectors[5] = SliceVector { center: -10.0, hue: 0.0, sat: 2.0, density: 1.0 }; // blue
    p
}

#[test]
fn the_rendered_picture_is_the_model() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp);
    let p = busy();
    let out = frame(&ctx, &mut comp, &graph(p));
    let mut moved = 0;
    for y in 0..H {
        for x in 0..W {
            let src = source(x, y);
            let want = p.apply(src, WS);
            let got = out[y as usize][x as usize];
            for c in 0..3 {
                assert!(
                    close(got[c], want[c]),
                    "row {y} x={x} ch{c}: rendered {} vs the model's {} (src {src:?})",
                    got[c],
                    want[c]
                );
            }
            assert!(close(got[3], 1.0), "alpha is not a colour");
            if (0..3).any(|c| (want[c] - src[c]).abs() > 0.01) {
                moved += 1;
            }
        }
    }
    // A test that passes because nothing moved proves nothing.
    assert!(moved > W as usize, "only {moved} pixels were graded at all");
    // And the grey row must not have moved in the MODEL either.
    for x in 0..W {
        let g = source(x, 2);
        let o = out[2][x as usize];
        assert!(close(o[0], g[0]) && close(o[1], g[1]) && close(o[2], g[2]), "grey moved: {o:?}");
    }
}

#[test]
fn a_default_node_is_an_identity() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp);
    let out = frame(&ctx, &mut comp, &graph(ColorSliceParams::default()));
    for y in 0..H {
        for x in 0..W {
            let src = source(x, y);
            for c in 0..3 {
                assert!(close(out[y as usize][x as usize][c], src[c]), "row {y} x={x}");
            }
        }
    }
}

/// The highlight keeps the vector's pixels and sinks the rest to a dim
/// grey — while armed. Unarmed, the same authored highlight is the graded
/// picture, which is what keeps it out of renders.
#[test]
fn the_highlight_shows_the_vector_only_when_armed() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp);
    let mut p = busy();
    p.highlight = Some(5); // blue
    let graded = frame(&ctx, &mut comp, &graph(p));
    p.previewing = true;
    let lit = frame(&ctx, &mut comp, &graph(p));
    let lw = WS.primaries.luma_weights();
    let (mut kept, mut sunk) = (0, 0);
    for x in 0..W {
        let src = source(x, 0);
        let wgt = p.highlight_weight(src, WS);
        let g = p.apply(src, WS);
        let y = g[0] * lw[0] + g[1] * lw[1] + g[2] * lw[2];
        let o = lit[0][x as usize];
        for c in 0..3 {
            let want = y * 0.22 + (g[c] - y * 0.22) * wgt;
            assert!(close(o[c], want), "x={x} ch{c}: {} vs {want} (w={wgt})", o[c]);
        }
        if wgt > 0.99 {
            kept += 1;
        }
        if wgt < 0.01 {
            sunk += 1;
        }
        // Unarmed: exactly the graded picture.
        for c in 0..3 {
            assert!(close(graded[0][x as usize][c], g[c]), "an unarmed highlight changed x={x}");
        }
    }
    assert!(kept > 0 && sunk > W as usize / 2, "kept {kept}, sunk {sunk}");
}
