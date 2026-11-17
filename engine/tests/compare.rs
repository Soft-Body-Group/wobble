//! Compare on real pixels: the wipe splits A and B along its line, A / B
//! show one input whole, and a half-size B is scaled up into the larger
//! format pixel for pixel. Self-skips without a GPU.

use engine::compositor::Compositor;
use engine::decode::ImageInfo;
use engine::graph::{CompareMode, CompareParams, Graph, NodeKind, SoftCropParams};
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
            eprintln!("SKIP compare test: no adapter ({e})");
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

fn compare(p: CompareParams, b: NodeKind, b_fmt: (u32, u32)) -> Graph {
    let mut g = graph(b);
    g.nodes.insert(2, NodeKind::Compare(p));
    g.edges.insert((2, 0), 0);
    g.edges.insert((2, 1), 1);
    g.view = Some(2);
    g.formats.insert(0, (W, H));
    g.formats.insert(1, b_fmt);
    g.formats.insert(2, (W, H));
    g
}

fn assert_px(name: &str, got: [f32; 4], want: [f32; 4], x: u32, y: u32) {
    for c in 0..4 {
        assert!((got[c] - want[c]).abs() < 4e-3, "{name} at ({x},{y}) ch{c}: {} want {}", got[c], want[c]);
    }
}

/// B is the plate flipped left to right: a vertical wipe through the middle
/// is the plate on the left and its mirror image on the right.
#[test]
fn the_wipe_splits_a_and_b() {
    let Some(ctx) = ctx() else { return };
    let flip = NodeKind::FlipFlop(engine::graph::FlipFlopParams { mode: engine::graph::FlipMode::Flop, matte: false });
    for mode in CompareMode::ALL {
        let p = CompareParams { mode, ..Default::default() };
        let out = render(&ctx, &compare(p, flip.clone(), (W, H)));
        for y in 0..H {
            for x in 0..W {
                let a = p.shows_a([x as f32 + 0.5, y as f32 + 0.5], [W as f32, H as f32]);
                let want = if a { source(x, y) } else { source(W - 1 - x, y) };
                assert_px(mode.name(), out[(y * W + x) as usize], want, x, y);
            }
        }
    }
}

/// B is the plate's top-left quarter at half size (a reformatting Crop):
/// shown alone it fills the frame, centre on centre.
#[test]
fn a_half_size_b_is_scaled_up_to_fill() {
    let Some(ctx) = ctx() else { return };
    let crop = SoftCropParams {
        rect: [0.0, 0.0, 0.5, 0.5],
        mode: engine::nodes::softcrop::CropMode::Reformat,
        ..Default::default()
    };
    let p = CompareParams { mode: CompareMode::B, ..Default::default() };
    let out = render(&ctx, &compare(p, NodeKind::SoftCrop(crop), (W / 2, H / 2)));
    // Reads filter bilinearly: B's pixel centres land between ours, so the
    // want is the quarter-plate interpolated there (interior, away from the
    // clamped edge).
    let lerp = |a: [f32; 4], b: [f32; 4], t: f32| -> [f32; 4] { std::array::from_fn(|c| a[c] + (b[c] - a[c]) * t) };
    for y in 1..H - 1 {
        for x in 1..W - 1 {
            let (u, v) = ((x as f32 + 0.5) * 0.5 - 0.5, (y as f32 + 0.5) * 0.5 - 0.5);
            let (i, j) = (u.floor() as u32, v.floor() as u32);
            let (fu, fv) = (u - u.floor(), v - v.floor());
            let top = lerp(source(i, j), source(i + 1, j), fu);
            let bot = lerp(source(i, j + 1), source(i + 1, j + 1), fu);
            assert_px("half-size B", out[(y * W + x) as usize], lerp(top, bot, fv), x, y);
        }
    }
}
