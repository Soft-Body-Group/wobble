//! Grain on real pixels: the fused shader draws what the CPU mirror
//! computes, a new frame is a new pattern with no recompile, and the
//! statistics hold on the GPU. Self-skips without a GPU (CI).

use engine::compositor::Compositor;
use engine::graph::{GrainParams, GrainStock, Graph, NodeKind};
use engine::timing::GpuTimer;
use half::f16;

const W: u32 = 64;
const H: u32 = 32;

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP grain test: no adapter ({e})");
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
        wgpu::Extent3d {
            width: W,
            height: H,
            depth_or_array_layers: 1,
        },
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

fn graph(p: GrainParams) -> Graph {
    let mut g = Graph::default_poc();
    g.nodes.insert(7, NodeKind::Grain(p));
    g.formats.insert(7, (W, H));
    g.view = Some(7);
    g
}

fn params() -> GrainParams {
    let mut p = GrainParams {
        size: (W, H),
        aspect: 1.3,
        seed: 4.0,
        time: 3.0,
        ..Default::default()
    };
    p.apply_stock(GrainStock::S16, W * 32);
    p.amount = [0.05, 0.08, 0.11];
    p
}

#[test]
fn the_shader_draws_what_the_cpu_mirror_computes() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    let p = params();
    let out = render(&ctx, &mut comp, &graph(p.clone()));
    let mut worst = 0.0f32;
    for y in 0..H {
        for x in 0..W {
            let want = p.sample([x as f32 + 0.5, y as f32 + 0.5], 1.0);
            let got = out[(y * W + x) as usize];
            for k in 0..3 {
                worst = worst.max((got[k] - want[k]).abs());
            }
            assert_eq!(got[3], 1.0);
        }
    }
    // f16 readback around 0.18 is ~1e-4 a step.
    assert!(worst < 2e-3, "worst error {worst}");
}

#[test]
fn a_new_frame_is_a_new_pattern_and_not_a_recompile() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    let mut g = graph(params());
    let before = g.topology_hash();
    let a = render(&ctx, &mut comp, &g);
    if let Some(NodeKind::Grain(p)) = g.nodes.get_mut(&7) {
        p.time = 4.0;
    }
    assert_eq!(g.topology_hash(), before);
    let b = render(&ctx, &mut comp, &g);
    let differ = a
        .iter()
        .zip(&b)
        .filter(|(x, y)| (x[1] - y[1]).abs() > 1e-3)
        .count();
    assert!(
        differ > a.len() / 2,
        "only {differ} of {} pixels moved",
        a.len()
    );
}

#[test]
fn a_wired_plate_is_grained_as_the_cpu_mirror_says_in_both_modes() {
    use engine::graph::{BlankParams, GrainApply};
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    let plate = [0.4, 0.2, 0.05, 1.0];
    for apply in GrainApply::ALL {
        let mut p = params();
        p.apply = apply;
        p.response = [0.5, 1.0, 1.5];
        let mut g = graph(p.clone());
        g.nodes.insert(
            3,
            NodeKind::Blank(BlankParams {
                size: (W, H),
                color: plate,
                ..Default::default()
            }),
        );
        g.formats.insert(3, (W, H));
        g.edges.insert((7, 0), 3);
        let out = render(&ctx, &mut comp, &g);
        let luma = g.working.primaries.luma_weights();
        let mut worst = 0.0f32;
        for y in 0..H {
            for x in 0..W {
                let want = p.apply_to(plate, [x as f32 + 0.5, y as f32 + 0.5], 1.0, luma);
                let got = out[(y * W + x) as usize];
                for k in 0..4 {
                    worst = worst.max((got[k] - want[k]).abs());
                }
            }
        }
        assert!(worst < 2e-3, "{apply:?}: worst error {worst}");
        // And it is actually grained, not passed through.
        assert!(
            out.iter().any(|c| (c[0] - plate[0]).abs() > 0.02),
            "{apply:?}"
        );
    }
}
