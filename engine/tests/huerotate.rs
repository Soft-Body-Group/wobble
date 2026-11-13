//! HueRotate on real pixels: the fused shader must draw exactly what the
//! CPU model (`HueRotateParams::apply`) says — round the whole hue circle, in
//! two working spaces, greys and a highlight included — and a fresh node
//! must change nothing. Self-skips without a GPU (CI).

use engine::color::Colorspace;
use engine::compositor::Compositor;
use engine::decode::ImageInfo;
use engine::graph::{Graph, HueRotateParams, NodeKind};
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

/// Row 0 walks the RGB hue circle at a normal exposure; row 1 is a grey
/// ramp that runs into highlights.
fn source(x: u32, y: u32) -> [f32; 3] {
    let t = x as f32 / W as f32;
    if y == 1 {
        return [t * 4.0; 3];
    }
    let h = t * 6.0;
    let f = h - h.floor();
    let (hi, lo) = (0.7, 0.1);
    let (up, down) = (lo + (hi - lo) * f, hi - (hi - lo) * f);
    match h as i32 {
        0 => [hi, up, lo],
        1 => [down, hi, lo],
        2 => [lo, hi, up],
        3 => [lo, down, hi],
        4 => [up, lo, hi],
        _ => [hi, lo, down],
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

fn graph(p: HueRotateParams, working: Colorspace) -> Graph {
    let mut g = Graph::default();
    g.working = working;
    g.nodes.insert(0, NodeKind::read(0, ""));
    g.nodes.insert(1, NodeKind::HueRotate(p));
    g.edges.insert((1, 0), 0);
    g.view = Some(1);
    g
}

/// Half floats: a small floor and a relative share of the value.
fn close(a: f32, b: f32) -> bool {
    (a - b).abs() <= 0.004 + 0.004 * b.abs()
}

#[test]
fn the_rendered_picture_is_the_model() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp);
    for working in [Colorspace::ACESCG, Colorspace::LINEAR_REC709] {
        for angle in [-120.0, 35.0, 180.0] {
            let p = HueRotateParams { angle };
            let out = frame(&ctx, &mut comp, &graph(p, working));
            let mut moved = 0;
            for y in 0..H {
                for x in 0..W {
                    let src = source(x, y);
                    let want = p.apply(src, working);
                    let got = out[(y * W + x) as usize];
                    for c in 0..3 {
                        assert!(
                            close(got[c], want[c]),
                            "{angle}° in {working:?}, row {y} x={x} ch{c}: rendered {} vs the model's {}",
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
            // A pass on an unmoved picture proves nothing — and the grey
            // row is the half that must NOT move.
            assert!(moved >= W as usize - 2, "{angle}°: only {moved} pixels turned");
        }
    }
}

#[test]
fn a_default_node_is_an_identity() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp);
    let out = frame(&ctx, &mut comp, &graph(HueRotateParams::default(), Colorspace::ACESCG));
    for y in 0..H {
        for x in 0..W {
            let src = source(x, y);
            for c in 0..3 {
                assert!(close(out[(y * W + x) as usize][c], src[c]), "row {y} x={x} moved");
            }
        }
    }
}
