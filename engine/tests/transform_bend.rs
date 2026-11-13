//! A Transform's shear and taper on real pixels: the fused shader samples
//! where the inverse of `codegen::transform_forward_px` says. Self-skips
//! without a GPU.

use engine::compositor::Compositor;
use engine::decode::ImageInfo;
use engine::graph::{Graph, NodeKind, TransformParams};
use engine::timing::GpuTimer;
use half::f16;

const W: u32 = 48;
const H: u32 = 32;

/// Smooth, so a bilinear lookup on the CPU agrees with the GPU's.
fn source(x: u32, y: u32) -> [f32; 4] {
    let (u, v) = (x as f32 / W as f32, y as f32 / H as f32);
    [u, v, (u * 3.0 + v * 2.0).sin() * 0.5 + 0.5, 1.0]
}

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP transform bend test: no adapter ({e})");
            None
        }
    }
}

fn upload(ctx: &engine::GpuContext, comp: &mut Compositor, slot: usize, px: &[[f32; 4]]) {
    comp.ensure_reader(slot, ImageInfo { width: W, height: H });
    let data: Vec<[f16; 4]> = px.iter().map(|c| c.map(f16::from_f32)).collect();
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

fn render(ctx: &engine::GpuContext, graph: &Graph) -> Vec<[f32; 4]> {
    // A fresh compositor per render: the chain is chosen when it is built.
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    let plate: Vec<[f32; 4]> = (0..W * H).map(|i| source(i % W, i / W)).collect();
    upload(ctx, &mut comp, 0, &plate);
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
            out.push(std::array::from_fn(|c| row[x * 4 + c].to_f32()));
        }
    }
    drop(data);
    staging.unmap();
    out
}



fn bilinear(p: [f32; 2]) -> [f32; 4] {
    let (x, y) = (p[0] - 0.5, p[1] - 0.5);
    let (x0, y0) = (x.floor(), y.floor());
    let (fx, fy) = (x - x0, y - y0);
    let at = |dx: f32, dy: f32| source((x0 + dx) as u32, (y0 + dy) as u32);
    let (a, b, c, d) = (at(0.0, 0.0), at(1.0, 0.0), at(0.0, 1.0), at(1.0, 1.0));
    std::array::from_fn(|i| {
        (a[i] * (1.0 - fx) + b[i] * fx) * (1.0 - fy) + (c[i] * (1.0 - fx) + d[i] * fx) * fy
    })
}

#[test]
fn shear_and_taper_sample_where_the_matrix_says() {
    let Some(ctx) = ctx() else { return };
    for t in [
        TransformParams { shear: [0.3, -0.2], ..Default::default() },
        TransformParams { taper: 0.4, ..Default::default() },
        TransformParams { taper: -0.3, taper_angle: 20.0, shear: [0.1, 0.0], rotate: 10.0, ..Default::default() },
    ] {
        let mut g = Graph::default();
        g.nodes.insert(0, NodeKind::read(0, ""));
        g.nodes.insert(1, NodeKind::Transform(t.clone()));
        g.edges.insert((1, 0), 0);
        g.view = Some(1);
        let out = render(&ctx, &g);
        let inv = engine::codegen::mat3_inv(&engine::codegen::transform_forward_px(&t, [W as f32, H as f32]));
        let mut checked = 0;
        for y in 0..H {
            for x in 0..W {
                let p = [x as f32 + 0.5, y as f32 + 0.5];
                let w = inv[2][0] * p[0] + inv[2][1] * p[1] + inv[2][2];
                let q = [
                    (inv[0][0] * p[0] + inv[0][1] * p[1] + inv[0][2]) / w,
                    (inv[1][0] * p[0] + inv[1][1] * p[1] + inv[1][2]) / w,
                ];
                // Well inside the plate only: the edge is a sampler matter.
                if q[0] < 2.0 || q[1] < 2.0 || q[0] > W as f32 - 2.0 || q[1] > H as f32 - 2.0 {
                    continue;
                }
                let want = bilinear(q);
                let got = out[(y * W + x) as usize];
                for c in 0..3 {
                    assert!(
                        (got[c] - want[c]).abs() < 0.01,
                        "{t:?} at ({x},{y}) ch{c}: {} want {}",
                        got[c],
                        want[c]
                    );
                }
                checked += 1;
            }
        }
        assert!(checked > (W * H / 3) as usize, "only {checked} pixels inside");
    }
}
