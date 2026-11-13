//! Despill and Lightwrap on real pixels: each fused / staged result must be
//! what its CPU model (`DespillParams::apply`, `LightwrapParams::combine`
//! over a CPU gaussian) says. Self-skips without a GPU (CI).

use engine::color::Colorspace;
use engine::compositor::Compositor;
use engine::decode::ImageInfo;
use engine::graph::{
    DespillMethod, DespillParams, DespillScreen, Graph, LightwrapParams, NodeKind, WrapMode,
};
use engine::timing::GpuTimer;
use half::f16;

const W: u32 = 64;
const H: u32 = 4;

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP despill/lightwrap test: no adapter ({e})");
            None
        }
    }
}

fn upload(ctx: &engine::GpuContext, comp: &mut Compositor, slot: usize, px: impl Fn(u32, u32) -> [f32; 4]) {
    comp.ensure_reader(slot, ImageInfo { width: W, height: H });
    let data: Vec<[f16; 4]> = (0..W * H).map(|i| px(i % W, i / W).map(f16::from_f32)).collect();
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

fn frame(ctx: &engine::GpuContext, comp: &mut Compositor, graph: &Graph) -> Vec<[f32; 4]> {
    assert!(comp.set_graph(graph), "shader failed to compile");
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

/// Half floats: a small floor and a relative share of the value.
fn close(a: f32, b: f32) -> bool {
    (a - b).abs() <= 0.004 + 0.004 * b.abs()
}

/// A green-spilled ramp along x, a blue-spilled one down the rows.
fn spilled(x: u32, y: u32) -> [f32; 4] {
    let t = x as f32 / (W - 1) as f32;
    let base = [0.45, 0.3, 0.22];
    let scr = if y % 2 == 0 { [0.08, 0.9, 0.1] } else { [0.06, 0.12, 0.9] };
    let c: [f32; 3] = std::array::from_fn(|i| base[i] + (scr[i] - base[i]) * t);
    [c[0], c[1], c[2], 1.0]
}

#[test]
fn despill_renders_its_model() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp, 0, spilled);
    for working in [Colorspace::LINEAR_REC709, Colorspace::ACESCG] {
        for screen in DespillScreen::ALL {
            for method in DespillMethod::ALL {
                let p = DespillParams { screen, method, strength: 0.8, bias: 1.1, restore: 0.6 };
                let mut g = Graph::default();
                g.working = working;
                g.nodes.insert(0, NodeKind::read(0, ""));
                g.nodes.insert(1, NodeKind::Despill(p));
                g.edges.insert((1, 0), 0);
                g.view = Some(1);
                let out = frame(&ctx, &mut comp, &g);
                for (i, got) in out.iter().enumerate() {
                    let src = spilled(i as u32 % W, i as u32 / W);
                    let want = p.apply([src[0], src[1], src[2]], working);
                    for c in 0..3 {
                        assert!(
                            close(got[c], want[c]),
                            "{screen:?} {method:?} px {i} ch {c}: got {} want {}",
                            got[c],
                            want[c]
                        );
                    }
                    assert!(close(got[3], 1.0), "alpha moved");
                }
            }
        }
    }
}

/// The compositor's separable gaussian along one row, edge-clamped, at
/// `sigma` — the same taps the pass takes.
fn blur_row(row: &[f32], sigma: f32) -> Vec<f32> {
    if sigma < 0.05 {
        return row.to_vec();
    }
    let r = ((sigma * 3.0).ceil() as i32).min(128);
    (0..row.len() as i32)
        .map(|x| {
            let (mut acc, mut ws) = (0.0, 0.0);
            for i in -r..=r {
                let w = (-((i * i) as f32) / (2.0 * sigma * sigma)).exp();
                acc += row[(x + i).clamp(0, row.len() as i32 - 1) as usize] * w;
                ws += w;
            }
            acc / ws
        })
        .collect()
}

/// Foreground: the left half, premultiplied grey at full alpha. Background:
/// a bright ramp. Both vary only along x, so the vertical blur is exact
/// and the CPU reference is one row.
fn fg(x: u32, _: u32) -> [f32; 4] {
    if x < W / 2 { [0.2, 0.25, 0.3, 1.0] } else { [0.0; 4] }
}
fn bg(x: u32, _: u32) -> [f32; 4] {
    let t = x as f32 / (W - 1) as f32;
    [0.5 + 3.0 * t, 1.0 + t, 2.0 - t, 1.0]
}

fn wrap_graph(p: LightwrapParams, with_bg: bool) -> Graph {
    let mut g = Graph::default();
    g.nodes.insert(0, NodeKind::read(0, ""));
    g.nodes.insert(1, NodeKind::read(1, ""));
    g.nodes.insert(2, NodeKind::Lightwrap(p));
    g.edges.insert((2, 0), 0);
    if with_bg {
        g.edges.insert((2, 1), 1);
    }
    g.view = Some(2);
    g
}

#[test]
fn lightwrap_renders_its_model() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp, 0, fg);
    upload(&ctx, &mut comp, 1, bg);
    for mode in WrapMode::ALL {
        let p = LightwrapParams { width: 4.0, bg_blur: 6.0, intensity: 0.7, mode };
        let out = frame(&ctx, &mut comp, &wrap_graph(p, true));
        let alpha: Vec<f32> = (0..W).map(|x| fg(x, 0)[3]).collect();
        let ab = blur_row(&alpha, p.width);
        let bgb: Vec<Vec<f32>> = (0..3)
            .map(|c| blur_row(&(0..W).map(|x| bg(x, 0)[c]).collect::<Vec<_>>(), p.bg_blur))
            .collect();
        for (i, got) in out.iter().enumerate() {
            let x = i as u32 % W;
            let want = p.combine(fg(x, 0), [bgb[0][x as usize], bgb[1][x as usize], bgb[2][x as usize]], ab[x as usize]);
            for c in 0..4 {
                assert!(close(got[c], want[c]), "{mode:?} x {x} ch {c}: got {} want {}", got[c], want[c]);
            }
        }
        // The wrap is really there: just inside the edge, more light than
        // the plain foreground; deep inside, none.
        if mode == WrapMode::Add {
            assert!(out[(W / 2 - 1) as usize][0] > 0.4, "no light at the edge: {:?}", out[(W / 2 - 1) as usize]);
            assert!(close(out[2][0], 0.2), "light in the core: {:?}", out[2]);
        }
    }
}

/// No background, nothing to wrap: the foreground passes through untouched.
#[test]
fn lightwrap_without_a_background_is_the_foreground() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp, 0, fg);
    upload(&ctx, &mut comp, 1, bg);
    let out = frame(&ctx, &mut comp, &wrap_graph(LightwrapParams::default(), false));
    for (i, got) in out.iter().enumerate() {
        let want = fg(i as u32 % W, 0);
        for c in 0..4 {
            assert!(close(got[c], want[c]), "px {i} ch {c}: got {} want {}", got[c], want[c]);
        }
    }
}
