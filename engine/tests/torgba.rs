//! ToRgba on real pixels: a matte comes back onto the IMAGE pipe as
//! (m, m, m, m), and an image node downstream treats it as a picture.
//! Self-skips without a GPU (CI).

use engine::compositor::Compositor;
use engine::graph::{Graph, MultParams, NodeKind, SplineMode, SplineParams, SplinePoint};
use engine::timing::GpuTimer;
use half::f16;

const W: u32 = 64;
const H: u32 = 32;

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP torgba test: no adapter ({e})");
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

/// A hard-edged rectangle at x 16..48, y 8..24 in output pixels.
fn rect() -> SplineParams {
    let (lo, hi) = (0.25, 0.75);
    SplineParams {
        points: [[lo, lo], [hi, lo], [hi, hi], [lo, hi]]
            .iter()
            .map(|p| SplinePoint::new(*p, 0.0))
            .collect(),
        mode: SplineMode::Linear,
        ..Default::default()
    }
}

fn px(out: &[[f32; 4]], x: u32, y: u32) -> [f32; 4] {
    out[(y * W + x) as usize]
}

fn near(a: [f32; 4], b: [f32; 4]) -> bool {
    a.iter().zip(b).all(|(a, b)| (a - b).abs() < 0.01)
}

#[test]
fn a_matte_arrives_as_rgba_and_an_image_node_can_use_it() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);

    // Spline -> ToRgba, viewed: white inside, transparent black outside.
    let mut g = Graph::default_poc();
    g.nodes.insert(7, NodeKind::Spline(rect()));
    g.nodes.insert(8, NodeKind::ToRgba(Default::default()));
    g.edges.insert((8, 0), 7);
    g.view = Some(8);
    let out = render(&ctx, &mut comp, &g);
    assert!(near(px(&out, 32, 16), [1.0; 4]), "inside: {:?}", px(&out, 32, 16));
    assert!(near(px(&out, 4, 4), [0.0; 4]), "outside: {:?}", px(&out, 4, 4));

    // ...and on into a Mult, which only takes an image: the matte is a
    // picture now, so half of it is half as bright (alpha untouched).
    g.nodes.insert(9, NodeKind::Mult(MultParams { value: [0.5; 3], ..Default::default() }));
    g.edges.insert((9, 0), 8);
    g.view = Some(9);
    let out = render(&ctx, &mut comp, &g);
    assert!(
        near(px(&out, 32, 16), [0.5, 0.5, 0.5, 1.0]),
        "through a Mult: {:?}",
        px(&out, 32, 16)
    );
}

/// Unwired, the matte is coverage 1 — the convention of every mask port.
#[test]
fn an_unwired_to_rgba_is_solid_white() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    let mut g = Graph::default_poc();
    g.nodes.insert(8, NodeKind::ToRgba(Default::default()));
    g.view = Some(8);
    let out = render(&ctx, &mut comp, &g);
    assert!(near(px(&out, 10, 10), [1.0; 4]), "{:?}", px(&out, 10, 10));
}

/// Its port is a MATTE port: a matte wires straight in, and an image does
/// not (that wire goes through a ToMatte, like every other matte input).
#[test]
fn its_input_takes_a_matte() {
    let t = NodeKind::ToRgba(Default::default());
    assert_eq!(t.input_kind(0), engine::graph::PortKind::Mask);
    assert_eq!(t.output_kind(), engine::graph::PortKind::Image);
}
