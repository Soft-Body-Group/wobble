//! A roto CONTAINER on real pixels: several shapes in one Spline, combined
//! by its inner ops and the union of its loose ends, render what the same
//! shapes chained by hand would. `engine::roto::expand` is the lowering the
//! app runs in `to_engine_graph_at`; here it is called directly.
//! Self-skips without a GPU (CI).

use engine::compositor::Compositor;
use engine::graph::{
    Graph, MatteOp, NodeKind, RotoOp, SplineMode, SplineParams, SplinePoint, SplineShape,
};
use engine::timing::GpuTimer;
use half::f16;

const W: u32 = 64;
const H: u32 = 32;

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP roto test: no adapter ({e})");
            None
        }
    }
}

fn render(ctx: &engine::GpuContext, comp: &mut Compositor, graph: &Graph) -> Vec<[f32; 4]> {
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
    let (tx, rx) = std::sync::mpsc::channel();
    staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().unwrap().unwrap();
    let view = staging.slice(..).get_mapped_range().expect("a mapped buffer");
    let mut out = Vec::with_capacity((W * H) as usize);
    for y in 0..H as usize {
        let row: &[f16] = bytemuck::cast_slice(
            &view[y * padded as usize..y * padded as usize + (W * 8) as usize],
        );
        for c in row.chunks_exact(4) {
            out.push([c[0].to_f32(), c[1].to_f32(), c[2].to_f32(), c[3].to_f32()]);
        }
    }
    out
}

/// A hard rectangle, normalized corners.
fn rect(x0: f32, y0: f32, x1: f32, y1: f32) -> Vec<SplinePoint> {
    [[x0, y0], [x1, y0], [x1, y1], [x0, y1]].iter().map(|p| SplinePoint::new(*p, 0.0)).collect()
}

fn alpha(out: &[[f32; 4]], x: u32, y: u32) -> f32 {
    out[(y * W + x) as usize][3]
}

fn draw(ctx: &engine::GpuContext, sp: SplineParams) -> Vec<[f32; 4]> {
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    let mut g = Graph::default_poc();
    g.nodes.insert(7, NodeKind::Spline(sp));
    g.view = Some(7);
    engine::roto::expand(&mut g);
    render(ctx, &mut comp, &g)
}

#[test]
fn loose_shapes_union_and_an_op_cuts() {
    let Some(ctx) = ctx() else { return };
    // Two shapes, no ops: both show (the loose ends are unioned).
    let mut sp = SplineParams {
        points: rect(0.05, 0.25, 0.4, 0.75),
        mode: SplineMode::Linear,
        ..Default::default()
    };
    sp.more.push(SplineShape {
        points: rect(0.6, 0.25, 0.95, 0.75),
        mode: SplineMode::Linear,
        ..Default::default()
    });
    let out = draw(&ctx, sp.clone());
    assert!(alpha(&out, 14, 16) > 0.99, "shape 0 missing");
    assert!(alpha(&out, 50, 16) > 0.99, "shape 1 missing");
    assert!(alpha(&out, 32, 16) < 0.01, "between the shapes");

    // A third shape inside the first, cut out of it by a Subtract op.
    sp.more.push(SplineShape {
        points: rect(0.15, 0.4, 0.3, 0.6),
        mode: SplineMode::Linear,
        ..Default::default()
    });
    sp.add_op(RotoOp { op: MatteOp::Subtract, a: Some(0), b: Some(2), ..Default::default() });
    let out = draw(&ctx, sp);
    assert!(alpha(&out, 14, 16) < 0.01, "the hole: {}", alpha(&out, 14, 16));
    assert!(alpha(&out, 5, 16) > 0.99, "the rim of shape 0");
    assert!(alpha(&out, 50, 16) > 0.99, "shape 1 still loose, still shown");
}
