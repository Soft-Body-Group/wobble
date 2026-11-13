//! Headless integration for Edge Detect: a hard step of height 1 reads 1
//! under every operator, flat parts read 0, and the matte output is the
//! threshold applied to that. Self-skips without a GPU (CI).

use engine::compositor::Compositor;
use engine::decode::ImageInfo;
use engine::graph::{EdgeDetectParams, EdgeOp, EdgeOutput, Graph, NodeKind};
use engine::timing::GpuTimer;
use half::f16;

const W: u32 = 64;
const H: u32 = 8;

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP edge detect test: no adapter ({e})");
            None
        }
    }
}

/// rgb 1 left of x = 32, 0 right of it, no anti-aliasing; alpha 1 throughout.
fn upload_step(ctx: &engine::GpuContext, comp: &mut Compositor, slot: usize) {
    comp.ensure_reader(slot, ImageInfo { width: W, height: H });
    let mut px: Vec<[f16; 4]> = Vec::with_capacity((W * H) as usize);
    for _ in 0..H {
        for x in 0..W {
            let v = f16::from_f32(if x < 32 { 1.0 } else { 0.0 });
            px.push([v, v, v, f16::ONE]);
        }
    }
    ctx.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &comp.reader_texture(slot).unwrap().texture,
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

/// One row of the output's red channel.
fn row(ctx: &engine::GpuContext, comp: &mut Compositor, graph: &Graph) -> Vec<f32> {
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
    // Row 4, to stay away from any edge effects at the top.
    let off = (padded * 4) as usize;
    let halves: &[f16] = bytemuck::cast_slice(&data[off..off + (W * 8) as usize]);
    let out = (0..W as usize).map(|i| halves[i * 4].to_f32()).collect();
    drop(data);
    staging.unmap();
    out
}

fn graph_with(p: EdgeDetectParams) -> Graph {
    let mut g = Graph::default();
    g.nodes.insert(0, NodeKind::read(0, ""));
    g.nodes.insert(1, NodeKind::EdgeDetect(p));
    g.edges.insert((1, 0), 0);
    g.view = Some(1);
    g
}

fn setup() -> Option<(engine::GpuContext, Compositor)> {
    let ctx = ctx()?;
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload_step(&ctx, &mut comp, 0);
    Some((ctx, comp))
}

#[test]
fn a_unit_step_reads_one_under_every_operator() {
    let Some((ctx, mut comp)) = setup() else { return };
    for op in EdgeOp::ALL {
        let got = row(&ctx, &mut comp, &graph_with(EdgeDetectParams { op, ..Default::default() }));
        let peak = got.iter().cloned().fold(0.0f32, f32::max);
        assert!((peak - 1.0).abs() < 0.02, "{}: the step peaks at {peak}", op.name());
        assert!(got[4].abs() < 1e-3, "{}: flat solid reads {}", op.name(), got[4]);
        assert!(got[60].abs() < 1e-3, "{}: flat empty reads {}", op.name(), got[60]);
        // Only the two pixels either side of the step see it.
        assert!(got[29].abs() < 1e-3 && got[34].abs() < 1e-3, "{}: {:?}", op.name(), &got[28..36]);
    }
}

#[test]
fn gain_scales_the_magnitude() {
    let Some((ctx, mut comp)) = setup() else { return };
    let got = row(&ctx, &mut comp, &graph_with(EdgeDetectParams { gain: 0.5, ..Default::default() }));
    let peak = got.iter().cloned().fold(0.0f32, f32::max);
    assert!((peak - 0.5).abs() < 0.02, "gain 0.5 peaks at {peak}");
}

#[test]
fn the_matte_is_the_threshold_of_the_strength() {
    let Some((ctx, mut comp)) = setup() else { return };
    let matte = |threshold| EdgeDetectParams {
        output: EdgeOutput::Matte,
        threshold,
        softness: 0.0,
        ..Default::default()
    };
    let got = row(&ctx, &mut comp, &graph_with(matte(0.5)));
    assert!((got[31] - 1.0).abs() < 1e-3 && (got[32] - 1.0).abs() < 1e-3, "{:?}", &got[28..36]);
    assert!(got[4].abs() < 1e-3 && got[60].abs() < 1e-3, "flat parts are not edges");
    // Over the strength, nothing is an edge.
    let none = row(&ctx, &mut comp, &graph_with(matte(1.5)));
    assert!(none.iter().all(|v| v.abs() < 1e-3), "{:?}", &none[28..36]);
}
