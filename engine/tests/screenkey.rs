//! ScreenKey on real pixels: the fused shader must key, despill and
//! premultiply exactly what the CPU model (`ScreenKeyParams::apply`) says,
//! in every output mode, with the holdout and garbage mattes wired and
//! not. Self-skips without a GPU (CI).

use engine::color::Colorspace;
use engine::compositor::Compositor;
use engine::decode::ImageInfo;
use engine::graph::{Graph, NodeKind, ScreenKeyParams, ScreenOut};
use engine::timing::GpuTimer;
use half::f16;

const W: u32 = 48;
const H: u32 = 2;

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP huerotate test: no adapter ({e})");
            None
        }
    }
}

/// Row 0: the screen colour at exposures from a dark corner to a hot
/// spot, then a grey ramp. Row 1: a walk from a foreground colour (a red
/// coat, then a blue one, then skin) into the screen, so every degree of
/// edge and spill is on the plate.
fn source(x: u32, y: u32) -> [f32; 3] {
    let t = x as f32 / W as f32;
    let s = SCREEN;
    if y == 0 {
        if t < 0.5 {
            let g = 0.05 + t * 2.0 * 3.0;
            return s.map(|v| v * g);
        }
        return [(t - 0.5) * 2.0 * 2.0 + 0.01; 3];
    }
    let fg = match (t * 3.0) as i32 {
        0 => [0.6, 0.1, 0.1],
        1 => [0.05, 0.1, 0.6],
        _ => [0.5, 0.32, 0.22],
    };
    let m = (t * 3.0).fract();
    std::array::from_fn(|i| fg[i] * (1.0 - m) + s[i] * m)
}

const SCREEN: [f32; 3] = [0.16, 0.55, 0.14];

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


/// The plate through a ScreenKey; `hold` / `garbage` wire a Blank of that
/// coverage into the matte port (the whole frame, so the wiring is what is
/// under test — the maths is the model's).
fn graph(p: ScreenKeyParams, working: Colorspace, hold: Option<f32>, garbage: Option<f32>) -> Graph {
    let mut g = Graph::default();
    g.working = working;
    g.nodes.insert(0, NodeKind::read(0, ""));
    g.nodes.insert(1, NodeKind::ScreenKey(p));
    g.edges.insert((1, 0), 0);
    let blank = |v: f32| {
        NodeKind::Blank(engine::graph::BlankParams { color: [v, v, v, 1.0], ..Default::default() })
    };
    if let Some(h) = hold {
        g.nodes.insert(2, blank(h));
        g.edges.insert((1, 1), 2);
    }
    if let Some(gb) = garbage {
        g.nodes.insert(3, blank(gb));
        g.edges.insert((1, 2), 3);
    }
    g.view = Some(1);
    g
}

/// Half floats: a small floor and a relative share of the value.
fn close(a: f32, b: f32) -> bool {
    (a - b).abs() <= 0.004 + 0.004 * b.abs()
}

#[test]
fn the_rendered_picture_is_the_model_in_every_mode() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp);
    for working in [Colorspace::ACESCG, Colorspace::LINEAR_REC709] {
        for output in ScreenOut::ALL {
            let mut p = ScreenKeyParams::default();
            p.screen = SCREEN;
            p.output = output;
            p.balance = 0.35;
            p.gain = 1.2;
            p.clip_black = 0.05;
            p.clip_white = 0.9;
            let out = frame(&ctx, &mut comp, &graph(p, working, None, None));
            let (mut keyed, mut kept) = (0, 0);
            for y in 0..H {
                for x in 0..W {
                    let src = source(x, y);
                    let want = p.apply([src[0], src[1], src[2], 1.0], 0.0, 0.0, working);
                    let got = out[(y * W + x) as usize];
                    for c in 0..4 {
                        assert!(
                            close(got[c], want[c]),
                            "{output:?} in {working:?}, row {y} x={x} ch{c}: rendered {} vs the model's {}",
                            got[c],
                            want[c]
                        );
                    }
                    // The matte mode carries the key in rgb, alpha 1.
                    let key = if output == ScreenOut::Matte { want[0] } else { want[3] };
                    if key < 0.01 {
                        keyed += 1;
                    } else if key > 0.99 {
                        kept += 1;
                    }
                }
            }
            if output != ScreenOut::Despill {
                // The plate has both a screen and a foreground on it, or
                // the agreement above proves nothing.
                assert!(keyed >= 20 && kept >= 20, "{output:?}: keyed {keyed}, kept {kept}");
            }
        }
    }
}

#[test]
fn holdout_keeps_and_garbage_drops() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp);
    let working = Colorspace::ACESCG;
    let mut p = ScreenKeyParams::default();
    p.screen = SCREEN;
    p.output = ScreenOut::Matte;
    // (holdout, garbage) coverages; None = unwired.
    for (hold, garbage) in [(Some(1.0), None), (None, Some(1.0)), (Some(0.5), Some(0.25))] {
        let out = frame(&ctx, &mut comp, &graph(p, working, hold, garbage));
        for y in 0..H {
            for x in 0..W {
                let src = source(x, y);
                let want = p.apply(
                    [src[0], src[1], src[2], 1.0],
                    hold.unwrap_or(0.0),
                    garbage.unwrap_or(0.0),
                    working,
                )[0];
                let got = out[(y * W + x) as usize][0];
                assert!(close(got, want), "{hold:?}/{garbage:?}: row {y} x={x} {got} vs {want}");
            }
        }
        // ...and the coverage did something: a full holdout is all
        // foreground, a full garbage all hole.
        let a: Vec<f32> = out.iter().map(|px| px[0]).collect();
        match (hold, garbage) {
            (Some(1.0), None) => assert!(a.iter().all(|v| close(*v, 1.0))),
            (None, Some(1.0)) => assert!(a.iter().all(|v| close(*v, 0.0))),
            _ => assert!(a.iter().all(|v| *v <= 0.76 && *v >= 0.37), "{:?}", &a[..8]),
        }
    }
}
