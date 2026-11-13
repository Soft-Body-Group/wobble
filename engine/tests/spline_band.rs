//! The spline's soft BAND on real pixels: a point's bias moves its
//! feather out or in, and the shape-wide `feather_out` / `feather_in` widen
//! the band on their own side — with the ramp running from full coverage
//! at the band's inner edge to none at its outer, exactly as
//! `codegen::spline_band` says (which is what the gizmo's rings draw).
//! Self-skips without a GPU (CI).

use engine::compositor::Compositor;
use engine::graph::{Graph, NodeKind, SplineMode, SplineParams, SplinePoint};
use engine::timing::GpuTimer;
use half::f16;

const W: u32 = 64;
const H: u32 = 32;

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP spline_band test: no adapter ({e})");
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

/// A rectangle x 16..48, y 8..24 px; every point carries `feather` about
/// `bias`.
fn rect(feather: f32, bias: f32) -> SplineParams {
    let (lo, hi) = (0.25, 0.75);
    SplineParams {
        points: [[lo, lo], [hi, lo], [hi, hi], [lo, hi]]
            .iter()
            .map(|p| SplinePoint { bias, ..SplinePoint::new(*p, feather) })
            .collect(),
        mode: SplineMode::Linear,
        ..Default::default()
    }
}

/// Along the middle row, left of and just inside the left edge (x = 16),
/// coverage against the band the params describe.
fn check(ctx: &engine::GpuContext, sp: SplineParams, band: [f32; 2], what: &str) {
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    let mut g = Graph::default_poc();
    g.nodes.insert(7, NodeKind::Spline(sp));
    g.view = Some(7);
    let out = render(ctx, &mut comp, &g);
    // Stop 7px inside: past that the top/bottom edges are nearer.
    for x in 0..23u32 {
        let sd = 16.0 - (x as f32 + 0.5); // + outside
        let want = ((band[1] - sd) / (band[1] - band[0]).max(1e-3)).clamp(0.0, 1.0);
        let got = out[(16 * W + x) as usize][3];
        assert!(
            (got - want).abs() < 0.03,
            "{what}: x={x} (sd {sd}) coverage {got}, band {band:?} wants {want}"
        );
    }
}

#[test]
fn bias_and_the_shape_wide_in_out_move_the_band() {
    let Some(ctx) = ctx() else { return };
    // Centred: the old symmetric feather.
    check(&ctx, rect(8.0, 0.0), [-4.0, 4.0], "centred");
    // Bias +feather/2: all OUTSIDE, the hard edge on the outline.
    check(&ctx, rect(8.0, 4.0), [0.0, 8.0], "outward point");
    // Bias -feather/2: all inside.
    check(&ctx, rect(8.0, -4.0), [-8.0, 0.0], "inward point");
    // A hard shape feathered by the shape-wide sliders alone.
    check(&ctx, SplineParams { feather_out: 6.0, ..rect(0.0, 0.0) }, [0.0, 6.0], "global out");
    check(&ctx, SplineParams { feather_in: 6.0, ..rect(0.0, 0.0) }, [-6.0, 0.0], "global in");
    // ...and they stack on a point's own band.
    check(
        &ctx,
        SplineParams { feather_out: 2.0, feather_in: 3.0, ..rect(4.0, 1.0) },
        [-4.0, 5.0],
        "stacked",
    );
}

/// The lane is exact where it matters: quarter pixels round-trip.
#[test]
fn the_band_lane_round_trips_quarter_pixels() {
    for b in [[-4.0f32, 4.0], [0.0, 8.0], [-511.75, 511.75], [-0.25, 0.5]] {
        let v = engine::codegen::spline_band_lane(b) as u32;
        let back = [((v >> 12) as f32 - 2048.0) * 0.25, ((v & 4095) as f32 - 2048.0) * 0.25];
        assert_eq!(back, b);
    }
}
