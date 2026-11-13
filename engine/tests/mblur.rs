//! Headless integration for the MotionBlur node's accumulator.
//!
//! Motion blur here is sub-frame sampling SCOPED to a node: the node's
//! upstream renders several times with the animated params evaluated at
//! fractional ticks, the results blend into its texture, and everything
//! downstream runs once. This file tests the blending and the scoping —
//! that N samples average to what they should, that a node downstream sees
//! the average once rather than being averaged itself, and that a stage
//! upstream really is re-run per sample. Self-skips without a GPU (CI).

use engine::compositor::Compositor;
use engine::graph::{BlurParams, Graph, GradeParams, MotionBlurParams, NodeKind};
use engine::timing::GpuTimer;
use half::f16;

const W: u32 = 32;
const H: u32 = 16;

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP mblur test: no adapter ({e})");
            None
        }
    }
}

fn upload(ctx: &engine::GpuContext, comp: &mut Compositor, v: f32) {
    comp.ensure_reader(0, engine::decode::ImageInfo { width: W, height: H });
    let px: Vec<[f16; 4]> = (0..W * H)
        .map(|_| [f16::from_f32(v), f16::from_f32(v), f16::from_f32(v), f16::ONE])
        .collect();
    ctx.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &comp.reader_texture(0).unwrap().texture,
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

fn read_px(ctx: &engine::GpuContext, comp: &Compositor) -> f32 {
    let padded = (W * 8).div_ceil(256) * 256;
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (padded * H) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    enc.copy_texture_to_buffer(
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
    ctx.queue.submit([enc.finish()]);
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    let data = slice.get_mapped_range().expect("a mapped buffer");
    let halves: &[f16] = bytemuck::cast_slice(&data[..(W * 8) as usize]);
    let v = halves[0].to_f32();
    drop(data);
    staging.unmap();
    v
}

const MB: u64 = 2;

/// Read -> Grade(`gain`) -> MotionBlur -> Grade(`after`): the upstream
/// gain is "the param the animation moves between sub-frames"; the
/// downstream one runs at the frame.
fn graded(gain: f32, after: f32) -> Graph {
    let grade = |g| NodeKind::Grade(GradeParams { gain: [g; 3], ..Default::default() });
    let mut g = Graph::default();
    g.nodes.insert(0, NodeKind::read(0, ""));
    g.nodes.insert(1, grade(gain));
    g.nodes.insert(MB, NodeKind::MotionBlur(MotionBlurParams::default()));
    g.nodes.insert(3, grade(after));
    g.edges.insert((1, 0), 0);
    g.edges.insert((MB, 0), 1);
    g.edges.insert((3, 0), MB);
    g.view = Some(3);
    g
}

/// The frame as the app renders it: one submit per sample graph, a resolve
/// after the last, then the comp at `frame`'s params.
fn render(ctx: &engine::GpuContext, comp: &mut Compositor, samples: &[Graph], frame: &Graph) -> f32 {
    assert!(comp.set_graph(frame), "fused shader failed to compile");
    let sid = *comp.motion_blur_stages().first().expect("the MotionBlur is a stage");
    assert_eq!(sid.0, MB);
    let timer = GpuTimer::new(ctx);
    let n = samples.len();
    for (i, g) in samples.iter().enumerate() {
        comp.write_params(g);
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        assert!(comp.encode_motion_sample(&mut enc, &timer, sid, 1.0 / n as f32, i == 0));
        if comp.params_stale() {
            comp.write_params(g);
        }
        if i + 1 == n {
            comp.resolve_motion(&mut enc, sid);
        }
        ctx.queue.submit([enc.finish()]);
    }
    comp.write_params(frame);
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    comp.encode_comp(&mut enc, &timer);
    if comp.params_stale() {
        comp.write_params(frame);
    }
    ctx.queue.submit([enc.finish()]);
    read_px(ctx, comp)
}

fn setup(ctx: &engine::GpuContext) -> Compositor {
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload(ctx, &mut comp, 0.5);
    comp
}

/// One sample of a still branch is exactly the unblurred frame — and so is
/// the node with NO samples (a closed shutter): a passthrough.
#[test]
fn a_single_sample_is_exactly_the_unblurred_frame() {
    let Some(ctx) = ctx() else { return };
    let mut comp = setup(&ctx);
    let g = graded(1.6, 1.0);
    let plain = render(&ctx, &mut comp, &[], &g);
    assert!((plain - 0.8).abs() < 5e-3, "the test comp itself is wrong: {plain:.5}");
    let one = render(&ctx, &mut comp, &[g.clone()], &g);
    assert!((plain - one).abs() < 1e-3, "one sample changed the frame: {plain:.5} -> {one:.5}");
}

/// Several samples blend. Two samples at half weight each land exactly
/// between them, which is what a shutter open across a moving value does.
#[test]
fn samples_blend_to_their_average() {
    let Some(ctx) = ctx() else { return };
    let mut comp = setup(&ctx);
    let got = render(&ctx, &mut comp, &[graded(0.4, 1.0), graded(1.6, 1.0)], &graded(1.0, 1.0));
    // 0.5 * 0.4 = 0.2 and 0.5 * 1.6 = 0.8, averaged.
    assert!((got - 0.5).abs() < 5e-3, "two samples should average to 0.5, got {got:.5}");
}

/// Downstream of the node runs ONCE, at the frame's params, on the average:
/// the samples' own values for the downstream Grade must not leak in.
#[test]
fn downstream_runs_once_at_the_frame() {
    let Some(ctx) = ctx() else { return };
    let mut comp = setup(&ctx);
    // The samples carry a wild downstream gain; the frame says 2.
    let got = render(
        &ctx,
        &mut comp,
        &[graded(0.4, 9.0), graded(1.6, 0.0)],
        &graded(1.0, 2.0),
    );
    assert!((got - 1.0).abs() < 6e-3, "average 0.5 x the frame's 2 = 1.0, got {got:.5}");
}

/// Eight samples across a ramp of values: the accumulator must not drift,
/// double-count or lose the first sample as the count goes up.
#[test]
fn many_samples_keep_their_mean() {
    let Some(ctx) = ctx() else { return };
    let mut comp = setup(&ctx);
    let gains: Vec<f32> = (0..8).map(|i| 0.2 + i as f32 * 0.2).collect();
    let samples: Vec<Graph> = gains.iter().map(|g| graded(*g, 1.0)).collect();
    let got = render(&ctx, &mut comp, &samples, &graded(1.0, 1.0));
    let want: f32 = gains.iter().map(|g| g * 0.5).sum::<f32>() / gains.len() as f32;
    assert!((got - want).abs() < 6e-3, "8 samples: got {got:.5}, wanted {want:.5}");
}

/// The accumulator is seeded, not appended to: rendering a second blurred
/// frame must not carry the first one's light into it.
#[test]
fn a_new_frame_does_not_inherit_the_last_one() {
    let Some(ctx) = ctx() else { return };
    let mut comp = setup(&ctx);
    let two = |g| [graded(g, 1.0), graded(g, 1.0)];
    let bright = render(&ctx, &mut comp, &two(1.8), &graded(1.8, 1.0));
    let dark = render(&ctx, &mut comp, &two(0.2), &graded(0.2, 1.0));
    assert!((bright - 0.9).abs() < 6e-3, "bright frame reads {bright:.5}");
    assert!((dark - 0.1).abs() < 6e-3, "the previous frame leaked in: {dark:.5}");
}

/// A STAGE upstream of the node is re-dispatched every sample: a Blur
/// between the Grade and the node on a flat plate is still flat, so the
/// average is the Grades' average — and a Blur that did not run would read
/// black.
#[test]
fn an_upstream_stage_runs_per_sample() {
    let Some(ctx) = ctx() else { return };
    let mut comp = setup(&ctx);
    let with_blur = |gain| {
        let mut g = graded(gain, 1.0);
        g.nodes.insert(4, NodeKind::Blur(BlurParams { size: 2.0, matte: false }));
        g.edges.insert((4, 0), 1);
        g.edges.insert((MB, 0), 4);
        g
    };
    let got = render(&ctx, &mut comp, &[with_blur(0.4), with_blur(1.6)], &with_blur(1.0));
    assert!((got - 0.5).abs() < 6e-3, "blurred flat samples should average to 0.5, got {got:.5}");
}
