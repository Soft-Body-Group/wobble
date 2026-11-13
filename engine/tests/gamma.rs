//! Gamma on real pixels: the fused shader must draw exactly what the CPU
//! model (`GammaParams::apply`) says — master and per-channel gammas, alpha,
//! negatives passed through, the mix — and a fresh node must change nothing.
//! Self-skips without a GPU (CI).

use engine::compositor::Compositor;
use engine::decode::ImageInfo;
use engine::graph::{GammaParams, Graph, NodeKind};
use engine::timing::GpuTimer;
use half::f16;

const W: u32 = 48;
const H: u32 = 2;

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP gamma test: no adapter ({e})");
            None
        }
    }
}

/// Row 0 runs every channel from below zero into highlights, at different
/// rates so the channels differ; row 1 is a mid-grey with alpha ramping 0..1.
fn source(x: u32, y: u32) -> [f32; 4] {
    let t = x as f32 / (W - 1) as f32;
    if y == 1 {
        return [0.18, 0.18, 0.18, t];
    }
    [t * 4.5 - 0.5, t * 2.0 - 0.25, t * t * 3.0, 1.0]
}

fn upload(ctx: &engine::GpuContext, comp: &mut Compositor) {
    comp.ensure_reader(0, ImageInfo { width: W, height: H });
    let px: Vec<[f16; 4]> = (0..W * H)
        .map(|i| {
            let c = source(i % W, i / W);
            c.map(f16::from_f32)
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

fn graph(p: GammaParams) -> Graph {
    let mut g = Graph::default();
    g.nodes.insert(0, NodeKind::read(0, ""));
    g.nodes.insert(1, NodeKind::Gamma(p));
    g.edges.insert((1, 0), 0);
    g.view = Some(1);
    g
}

/// Half floats: a small floor and a relative share of the value.
fn close(a: f32, b: f32) -> bool {
    (a - b).abs() <= 0.004 + 0.004 * b.abs()
}

fn setup() -> Option<(engine::GpuContext, Compositor)> {
    let ctx = ctx()?;
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp);
    Some((ctx, comp))
}

#[test]
fn the_rendered_picture_is_the_model() {
    let Some((ctx, mut comp)) = setup() else { return };
    let cases = [
        GammaParams { master: 2.2, ..Default::default() },
        GammaParams { master: 0.5, ..Default::default() },
        GammaParams { master: 1.5, rgb: [1.0, 0.6, 2.0], alpha: 0.5, mix: 1.0, matte: false },
        GammaParams { master: 2.0, mix: 0.3, ..Default::default() },
    ];
    for p in cases {
        let out = frame(&ctx, &mut comp, &graph(p));
        let mut moved = 0;
        for y in 0..H {
            for x in 0..W {
                let src = source(x, y);
                let want = p.apply(src);
                let got = out[(y * W + x) as usize];
                for c in 0..4 {
                    assert!(
                        close(got[c], want[c]),
                        "{p:?}, row {y} x={x} ch{c}: rendered {} vs the model's {}",
                        got[c],
                        want[c]
                    );
                    // Negatives and zero come through exactly.
                    if src[c] <= 0.0 {
                        assert_eq!(got[c], f16::from_f32(src[c]).to_f32(), "{p:?} touched {}", src[c]);
                    }
                }
                if (0..4).any(|c| (want[c] - src[c]).abs() > 0.01) {
                    moved += 1;
                }
            }
        }
        // A pass on an unmoved picture proves nothing.
        assert!(moved >= W as usize, "{p:?}: only {moved} pixels moved");
    }
}

#[test]
fn a_default_node_is_an_identity() {
    let Some((ctx, mut comp)) = setup() else { return };
    let out = frame(&ctx, &mut comp, &graph(GammaParams::default()));
    for y in 0..H {
        for x in 0..W {
            let src = source(x, y);
            for c in 0..4 {
                assert!(close(out[(y * W + x) as usize][c], src[c]), "row {y} x={x} ch{c} moved");
            }
        }
    }
}
