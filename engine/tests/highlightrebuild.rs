//! Highlight Rebuild on the GPU: a pixel with no channel near the clip
//! comes out bit for bit as it went in, and a red-clipped patch inside a
//! surround of one hue gets its true red back. Self-skips without a GPU.

use engine::compositor::Compositor;
use engine::decode::ImageInfo;
use engine::graph::{Graph, HighlightRebuildParams, NodeKind};
use engine::timing::GpuTimer;
use half::f16;

const W: u32 = 64;
const H: u32 = 48;

/// The surface's colour: red twice green, green twice blue.
const HUE: [f32; 3] = [1.0, 0.5, 0.25];

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP highlight rebuild test: no adapter ({e})");
            None
        }
    }
}

fn in_patch(x: u32, y: u32) -> bool {
    (24..40).contains(&x) && (24..36).contains(&y)
}

/// What the scene really was: the surround at 0.7 of the hue, a patch at
/// 1.6 — whose red the camera clipped at 1.0. The top rows are a ramp of
/// arbitrary, unclipped colours, for the pass-through check, out of the
/// patch's search radius (a reference of another hue would bend it).
fn truth(x: u32, y: u32) -> [f32; 4] {
    if y < 4 {
        let t = (x + y * W) as f32 / (W * 4) as f32;
        return [0.05 + 0.6 * t, 0.7 - 0.5 * t, 0.123 + 0.3 * (t * 7.0).sin().abs(), 0.37];
    }
    let k = if in_patch(x, y) { 1.6 } else { 0.7 };
    [HUE[0] * k, HUE[1] * k, HUE[2] * k, 1.0]
}

fn plate(x: u32, y: u32) -> [f16; 4] {
    let t = truth(x, y);
    [f16::from_f32(t[0].min(1.0)), f16::from_f32(t[1].min(1.0)), f16::from_f32(t[2].min(1.0)), f16::from_f32(t[3])]
}

fn upload(ctx: &engine::GpuContext, comp: &mut Compositor) {
    comp.ensure_reader(0, ImageInfo { width: W, height: H });
    let px: Vec<[f16; 4]> = (0..H).flat_map(|y| (0..W).map(move |x| plate(x, y))).collect();
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

/// The whole output, as the half floats it is stored in.
fn render(ctx: &engine::GpuContext, comp: &mut Compositor, graph: &Graph) -> Vec<[f16; 4]> {
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
            layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(padded), rows_per_image: Some(H) },
        },
        wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
    );
    ctx.queue.submit([encoder.finish()]);
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    let data = slice.get_mapped_range().expect("a mapped buffer");
    let mut out = Vec::with_capacity((W * H) as usize);
    for y in 0..H as usize {
        let row: &[[f16; 4]] = bytemuck::cast_slice(&data[y * padded as usize..y * padded as usize + (W * 8) as usize]);
        out.extend_from_slice(row);
    }
    drop(data);
    staging.unmap();
    out
}

fn graph_with(p: HighlightRebuildParams) -> Graph {
    let mut g = Graph::default();
    g.nodes.insert(0, NodeKind::read(0, ""));
    g.nodes.insert(1, NodeKind::HighlightRebuild(p));
    g.edges.insert((1, 0), 0);
    g.view = Some(1);
    g
}

fn setup() -> Option<(engine::GpuContext, Compositor)> {
    let ctx = ctx()?;
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp);
    Some((ctx, comp))
}

#[test]
fn unclipped_pixels_pass_through_bit_exact() {
    let Some((ctx, mut comp)) = setup() else { return };
    let got = render(&ctx, &mut comp, &graph_with(HighlightRebuildParams::default()));
    for y in 0..H {
        for x in 0..W {
            if in_patch(x, y) {
                continue;
            }
            let want = plate(x, y);
            let have = got[(y * W + x) as usize];
            let bits = |p: [f16; 4]| p.map(f16::to_bits);
            assert_eq!(bits(have), bits(want), "({x}, {y}) changed: {have:?} for {want:?}");
        }
    }
}

#[test]
fn a_red_clipped_patch_gets_its_red_back() {
    let Some((ctx, mut comp)) = setup() else { return };
    // The radius has to reach the patch's middle from its edge (6 rows).
    let p = HighlightRebuildParams { radius: 16.0, ..Default::default() };
    let got = render(&ctx, &mut comp, &graph_with(p));
    let mut worst = 0.0f32;
    for y in 24..36 {
        for x in 24..40 {
            let t = truth(x, y);
            let have = got[(y * W + x) as usize].map(f16::to_f32);
            // Green and blue were never clipped: untouched.
            assert_eq!(have[1], plate(x, y)[1].to_f32(), "({x}, {y}) green moved");
            assert_eq!(have[2], plate(x, y)[2].to_f32(), "({x}, {y}) blue moved");
            worst = worst.max((have[0] - t[0]).abs() / t[0]);
        }
    }
    assert!(worst < 0.01, "rebuilt red off by {:.2}% at worst", worst * 100.0);
}

#[test]
fn out_of_reach_or_all_clipped_is_left_alone() {
    let Some((ctx, mut comp)) = setup() else { return };
    // A radius that cannot reach the patch's middle from its edge.
    let p = HighlightRebuildParams { radius: 2.0, ..Default::default() };
    let got = render(&ctx, &mut comp, &graph_with(p));
    let mid = got[(30 * W + 32) as usize].map(f16::to_f32);
    assert_eq!(mid[0], 1.0, "no reference in reach: red stays at the clip");
    // A clip level under every channel of the patch: all three clipped,
    // nothing to rebuild from, so the plate itself.
    let p = HighlightRebuildParams { clip: 0.3, softness: 0.0, ..Default::default() };
    let got = render(&ctx, &mut comp, &graph_with(p));
    let mid = got[(30 * W + 32) as usize];
    assert_eq!(mid.map(f16::to_bits), plate(32, 30).map(f16::to_bits));
}
