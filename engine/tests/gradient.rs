//! Gradient on real pixels: the fused shader matches the CPU mirror the
//! editor draws from, in both mixes, at an angle, with alpha, with the
//! stops authored out of order. Self-skips without a GPU (CI).

use engine::compositor::Compositor;
use engine::graph::{GradInterp, GradStop, GradientParams, Graph, NodeKind};
use engine::timing::GpuTimer;
use half::f16;

const W: u32 = 64;
const H: u32 = 32;

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP gradient test: no adapter ({e})");
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

fn params(interp: GradInterp) -> GradientParams {
    GradientParams {
        size: (W, H),
        inherit: false,
        // Authored out of order: the index is identity, the engine sorts.
        stops: vec![
            GradStop { pos: 0.9, color: [0.1, 0.2, 1.0, 1.0] },
            GradStop { pos: 0.1, color: [1.0, 0.1, 0.05, 1.0] },
            GradStop { pos: 0.5, color: [0.2, 0.9, 0.3, 0.5] },
        ],
        angle: 30.0,
        center: [0.45, 0.55],
        scale: 0.8,
        interp,
    }
}

#[test]
fn the_shader_draws_what_the_editor_shows() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    for interp in GradInterp::ALL {
        let p = params(interp);
        let mut g = Graph::default_poc();
        g.nodes.insert(7, NodeKind::Gradient(p.clone()));
        g.formats.insert(7, (W, H));
        g.view = Some(7);
        let out = render(&ctx, &mut comp, &g);
        let mut worst = 0.0f32;
        for y in 0..H {
            for x in 0..W {
                let t = p.t_at([x as f32 + 0.5, y as f32 + 0.5], W as f32, H as f32);
                let c = p.sample(t, g.working);
                let want = [c[0] * c[3], c[1] * c[3], c[2] * c[3], c[3]];
                let got = out[(y * W + x) as usize];
                for k in 0..4 {
                    worst = worst.max((got[k] - want[k]).abs());
                }
            }
        }
        assert!(worst < 4e-3, "{interp:?}: worst error {worst}");
    }
}

#[test]
fn moving_a_stop_or_adding_one_is_not_a_recompile() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    let mut g = Graph::default_poc();
    g.nodes.insert(7, NodeKind::Gradient(params(GradInterp::Lch)));
    g.formats.insert(7, (W, H));
    g.view = Some(7);
    let before = g.topology_hash();
    if let Some(NodeKind::Gradient(p)) = g.nodes.get_mut(&7) {
        p.stops[0].pos = 0.3;
        p.stops.push(GradStop { pos: 0.7, color: [1.0; 4] });
        p.angle = -120.0;
    }
    assert_eq!(g.topology_hash(), before);
    // And it still renders the new ramp.
    let out = render(&ctx, &mut comp, &g);
    assert!(out.iter().any(|c| c[3] > 0.99));
}
