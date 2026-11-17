//! SoftCrop on real pixels: a soft crop passes every pixel, a FULL crop
//! keeps exactly the pixels `keeps` says and zeroes the rest, and moving
//! its box moves the cut; a REFORMAT is the box at its corner and nothing
//! past its own format. Self-skips without a GPU.

use engine::compositor::Compositor;
use engine::decode::ImageInfo;
use engine::graph::{Graph, NodeKind, SoftCropParams};
use engine::timing::GpuTimer;
use half::f16;

const W: u32 = 40;
const H: u32 = 24;

/// Every pixel a different colour, so a wrong lookup cannot hide.
fn source(x: u32, y: u32) -> [f32; 4] {
    [x as f32 / W as f32, y as f32 / H as f32, ((x * 7 + y * 3) % 5) as f32 / 5.0, 1.0]
}

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP softcrop test: no adapter ({e})");
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


fn graph(op: NodeKind) -> Graph {
    let mut g = Graph::default();
    g.nodes.insert(0, NodeKind::read(0, ""));
    g.nodes.insert(1, op);
    g.edges.insert((1, 0), 0);
    g.view = Some(1);
    g
}

fn check(name: &str, out: &[[f32; 4]], p: &SoftCropParams) {
    for y in 0..H {
        for x in 0..W {
            let keep = p.keeps([x as f32 + 0.5, y as f32 + 0.5], [W as f32, H as f32]);
            let want = if keep { source(x, y) } else { [0.0; 4] };
            let got = out[(y * W + x) as usize];
            for c in 0..4 {
                assert!(
                    (got[c] - want[c]).abs() < 4e-3,
                    "{name} at ({x},{y}) ch{c}: {} want {}",
                    got[c],
                    want[c]
                );
            }
        }
    }
}

#[test]
fn a_full_crop_removes_what_lies_outside_the_box() {
    let Some(ctx) = ctx() else { return };
    let rect = [0.25, 0.125, 0.75, 0.625];
    let soft = SoftCropParams { rect, ..Default::default() };
    check("soft", &render(&ctx, &graph(NodeKind::SoftCrop(soft))), &soft);
    let full = SoftCropParams { mode: engine::nodes::softcrop::CropMode::Crop, ..soft };
    let out = render(&ctx, &graph(NodeKind::SoftCrop(full)));
    assert!(out.iter().any(|c| c[3] == 0.0), "a full crop cut nothing");
    check("full", &out, &full);
    let moved = SoftCropParams { rect: [0.0, 0.5, 0.4, 1.0], ..full };
    check("moved", &render(&ctx, &graph(NodeKind::SoftCrop(moved))), &moved);
}

/// A reformat's pixel p is the input's p + the box's corner, and past its
/// own format it is transparent black: a Transform or a Layer that reads
/// beyond the box (crops into a Layer) must not find the plate there.
#[test]
fn a_reformat_is_the_box_and_nothing_past_it() {
    let Some(ctx) = ctx() else { return };
    let p = SoftCropParams {
        rect: [0.25, 0.125, 0.75, 0.625],
        mode: engine::nodes::softcrop::CropMode::Reformat,
        ..Default::default()
    };
    let ((x0, y0), (bw, bh)) = p.px_box(W, H);
    let mut g = graph(NodeKind::SoftCrop(p));
    g.formats.insert(0, (W, H));
    g.formats.insert(1, (bw, bh));
    let out = render(&ctx, &g);
    for y in 0..H {
        for x in 0..W {
            let want = if x < bw && y < bh { source(x + x0, y + y0) } else { [0.0; 4] };
            let got = out[(y * W + x) as usize];
            for c in 0..4 {
                assert!(
                    (got[c] - want[c]).abs() < 4e-3,
                    "reformat at ({x},{y}) ch{c}: {} want {}",
                    got[c],
                    want[c]
                );
            }
        }
    }
}

/// Crop (reformat) then Uncrop with the same box is the plate inside the
/// box and transparent black outside it, at the full format.
#[test]
fn an_uncrop_puts_a_reformat_back() {
    let Some(ctx) = ctx() else { return };
    let rect = [0.25, 0.125, 0.75, 0.625];
    let p = SoftCropParams { rect, mode: engine::nodes::softcrop::CropMode::Reformat, ..Default::default() };
    let ((x0, y0), (bw, bh)) = p.px_box(W, H);
    let mut g = graph(NodeKind::SoftCrop(p));
    g.nodes.insert(2, NodeKind::Uncrop(engine::graph::UncropParams { rect, crop: None }));
    g.edges.insert((2, 0), 1);
    g.view = Some(2);
    g.formats.insert(0, (W, H));
    g.formats.insert(1, (bw, bh));
    g.formats.insert(2, (W, H));
    let out = render(&ctx, &g);
    for y in 0..H {
        for x in 0..W {
            let inside = x >= x0 && x < x0 + bw && y >= y0 && y < y0 + bh;
            let want = if inside { source(x, y) } else { [0.0; 4] };
            let got = out[(y * W + x) as usize];
            for c in 0..4 {
                assert!((got[c] - want[c]).abs() < 4e-3, "uncrop at ({x},{y}) ch{c}: {} want {}", got[c], want[c]);
            }
        }
    }
}
