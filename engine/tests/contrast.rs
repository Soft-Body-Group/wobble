//! Contrast on real pixels: the fused shader must draw what the CPU model
//! (`ContrastParams::apply`) says — log and linear, per side, toe and
//! shoulder, per channel and on luminance, the mix — a fresh node must
//! change nothing, and local contrast must push an edge apart while leaving
//! a flat field alone. Self-skips without a GPU (CI).

use engine::compositor::Compositor;
use engine::decode::ImageInfo;
use engine::graph::{ContrastDomain, ContrastParams, Graph, NodeKind};
use engine::timing::GpuTimer;
use half::f16;

const W: u32 = 64;
const H: u32 = 16;

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP contrast test: no adapter ({e})");
            None
        }
    }
}

/// Every row a different colour ramp from below zero into highlights.
fn ramp(x: u32, y: u32) -> [f32; 4] {
    let t = x as f32 / (W - 1) as f32;
    let k = 1.0 + y as f32 * 0.15;
    [t * 4.5 * k - 0.3, t * t * 2.0, (t * 3.0).exp2() * 0.05 * k, 0.25 + 0.75 * t]
}

/// A hard vertical edge: a dark wall left, a bright one right.
fn edge(x: u32, _y: u32) -> [f32; 4] {
    if x < W / 2 { [0.06, 0.05, 0.04, 1.0] } else { [0.6, 0.5, 0.4, 1.0] }
}

fn flat(_x: u32, _y: u32) -> [f32; 4] {
    [0.3, 0.2, 0.1, 1.0]
}

fn upload(ctx: &engine::GpuContext, comp: &mut Compositor, source: fn(u32, u32) -> [f32; 4]) {
    comp.ensure_reader(0, ImageInfo { width: W, height: H });
    let px: Vec<[f16; 4]> = (0..W * H)
        .map(|i| source(i % W, i / W).map(f16::from_f32))
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

fn graph(p: ContrastParams) -> Graph {
    let mut g = Graph::default();
    g.nodes.insert(0, NodeKind::read(0, ""));
    g.nodes.insert(1, NodeKind::Contrast(p));
    g.edges.insert((1, 0), 0);
    g.view = Some(1);
    g
}

/// Half floats: a small floor and a relative share of the value.
fn close(a: f32, b: f32) -> bool {
    (a - b).abs() <= 0.004 + 0.006 * b.abs()
}

fn setup(source: fn(u32, u32) -> [f32; 4]) -> Option<(engine::GpuContext, Compositor, [f32; 3])> {
    let ctx = ctx()?;
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp, source);
    let w = Graph::default().working.primaries.luma_weights();
    Some((ctx, comp, w))
}

/// What the GPU actually read: the source through half floats.
fn src16(source: fn(u32, u32) -> [f32; 4], x: u32, y: u32) -> [f32; 4] {
    source(x, y).map(|v| f16::from_f32(v).to_f32())
}

#[test]
fn the_rendered_picture_is_the_model() {
    let Some((ctx, mut comp, w)) = setup(ramp) else { return };
    let d = ContrastParams::default();
    let cases = [
        ContrastParams { contrast: 1.6, ..d },
        ContrastParams { contrast: 1.6, preserve: 0.0, ..d },
        ContrastParams { contrast: 0.6, preserve: 0.4, pivot: 0.3, ..d },
        ContrastParams { contrast: 2.0, toe: 1.0, shoulder: 0.5, range: 2.0, ..d },
        ContrastParams { shadows: 1.8, highlights: 0.7, preserve: 0.0, ..d },
        ContrastParams { contrast: 1.4, domain: ContrastDomain::Linear, preserve: 0.0, ..d },
        ContrastParams { contrast: 1.4, domain: ContrastDomain::Linear, toe: 0.7, ..d },
        ContrastParams { contrast: 2.0, mix: 0.3, ..d },
    ];
    for p in cases {
        let out = frame(&ctx, &mut comp, &graph(p));
        let mut moved = 0;
        for y in 0..H {
            for x in 0..W {
                let src = src16(ramp, x, y);
                let want = p.apply(src, w);
                let got = out[(y * W + x) as usize];
                for c in 0..4 {
                    assert!(
                        close(got[c], want[c]),
                        "{p:?}, row {y} x={x} ch{c}: rendered {} vs the model's {}",
                        got[c],
                        want[c]
                    );
                }
                if (0..3).any(|c| (want[c] - src[c]).abs() > 0.01) {
                    moved += 1;
                }
            }
        }
        // A pass on an unmoved picture proves nothing.
        assert!(moved >= (W * H / 2) as usize, "{p:?}: only {moved} pixels moved");
    }
}

#[test]
fn a_default_node_is_an_identity() {
    let Some((ctx, mut comp, _)) = setup(ramp) else { return };
    let out = frame(&ctx, &mut comp, &graph(ContrastParams::default()));
    for y in 0..H {
        for x in 0..W {
            let src = src16(ramp, x, y);
            for c in 0..4 {
                assert!(close(out[(y * W + x) as usize][c], src[c]), "row {y} x={x} ch{c} moved");
            }
        }
    }
}

/// On a flat field every pixel IS its base, so clarity does nothing and
/// the local path draws exactly what the global one does.
#[test]
fn local_contrast_leaves_a_flat_field_alone() {
    let Some((ctx, mut comp, w)) = setup(flat) else { return };
    let p = ContrastParams { contrast: 1.5, local: true, clarity: 1.0, radius: 8.0, ..Default::default() };
    let out = frame(&ctx, &mut comp, &graph(p));
    let want = p.apply(src16(flat, 0, 0), w);
    for (i, got) in out.iter().enumerate() {
        for c in 0..4 {
            assert!(close(got[c], want[c]), "px {i} ch{c}: {} vs {}", got[c], want[c]);
        }
    }
}

/// Across an edge the bright side gets brighter and the dark side darker
/// near it, and far from it (past the radius) nothing much moves.
#[test]
fn local_contrast_pushes_an_edge_apart() {
    let Some((ctx, mut comp, _)) = setup(edge) else { return };
    let p = ContrastParams { local: true, clarity: 0.8, radius: 4.0, limit: 2.0, ..Default::default() };
    let out = frame(&ctx, &mut comp, &graph(p));
    let row = H / 2;
    let at = |x: u32| out[(row * W + x) as usize];
    let src = |x: u32| src16(edge, x, row);
    let (dark, bright) = (W / 2 - 1, W / 2);
    assert!(at(dark)[1] < src(dark)[1] * 0.9, "dark side {:?} vs {:?}", at(dark), src(dark));
    assert!(at(bright)[1] > src(bright)[1] * 1.1, "bright side {:?} vs {:?}", at(bright), src(bright));
    for x in [2, W - 3] {
        assert!(close(at(x)[1], src(x)[1]) || (at(x)[1] / src(x)[1] - 1.0).abs() < 0.08, "far x={x} moved: {:?}", at(x));
    }
    // Alpha is never touched, and hue holds (it scales rgb together).
    let b = at(bright);
    assert_eq!(b[3], 1.0);
    assert!((b[0] / b[2] - 1.5).abs() < 0.03, "{b:?}");
}
