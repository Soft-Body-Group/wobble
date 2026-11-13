//! Headless integration for the spline mask: what the shader actually
//! rasterizes, checked against the contract the viewer's gizmo draws.
//!
//! Two claims live here. The matte has to reach ALPHA (the viewer's alpha
//! toggle and every downstream composite read it there, not just in rgb),
//! and the soft edge has to run from full coverage at `-feather/2` of the
//! signed distance to zero at `+feather/2` — which is where the gizmo puts
//! its two feather rings. If the shader ever ramps over a different width,
//! the overlay is lying about the render.
//!
//! Self-skips without a GPU (CI).

use engine::compositor::Compositor;
use engine::decode::ImageInfo;
use engine::graph::{
    Graph, NodeKind, ShapeXf, MatteOp, SplineMode, SplineParams, SplinePoint,
};
use engine::timing::GpuTimer;
use half::f16;

const W: u32 = 64;
const H: u32 = 32;
const FEATHER: f32 = 8.0;

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP spline test: no adapter ({e})");
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

/// A rectangle in normalized space, Linear mode so the polyline IS the
/// rectangle and the expected signed distance is arithmetic rather than an
/// approximation of a curve.
fn rect(lo: f32, hi: f32, feather: f32) -> SplineParams {
    SplineParams {
        points: [[lo, lo], [hi, lo], [hi, hi], [lo, hi]]
            .iter()
            .map(|p| SplinePoint::new(*p, feather))
            .collect(),
        mode: SplineMode::Linear,
        ..Default::default()
    }
}

/// That rectangle at x 16..48, y 8..24 in output pixels, as the view.
/// NOTHING is wired to the Spline's matte input, so this is the bare shape.
fn spline_graph(sp: SplineParams) -> Graph {
    let mut g = Graph::default_poc();
    g.nodes.insert(7, NodeKind::Spline(sp));
    g.view = Some(7);
    g
}

fn rect_graph() -> Graph {
    spline_graph(rect(0.25, 0.75, FEATHER))
}

#[test]
fn matte_lands_in_alpha_and_the_feather_ramp_is_where_the_gizmo_draws_it() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 2);
    comp.ensure_output(W, H);
    comp.ensure_reader(0, ImageInfo { width: W, height: H });
    let out = render(&ctx, &mut comp, &rect_graph());
    let at = |x: u32, y: u32| out[(y * W + x) as usize];

    // Alpha is not a stand-in for 1.0 any more: it carries the matte.
    assert!((at(32, 16)[3] - 1.0).abs() < 0.01, "inside alpha {:?}", at(32, 16));
    assert!(at(1, 1)[3] < 0.01, "outside alpha {:?}", at(1, 1));
    for p in &out {
        assert!((p[3] - p[0]).abs() < 0.01, "alpha must match the matte in rgb: {p:?}");
    }

    // Across the left edge, where that edge is the nearest one: coverage
    // 1 at -feather/2, 0.5 on the outline, 0 at +feather/2.
    for x in 10..=22u32 {
        let px = x as f32 + 0.5; // the shader samples pixel centres
        let sd = 16.0 - px; // positive outside
        let want = (0.5 - sd / FEATHER).clamp(0.0, 1.0);
        let got = at(x, 16)[0];
        assert!(
            (got - want).abs() < 0.02,
            "x={x}: coverage {got:.3}, the ramp says {want:.3}"
        );
    }
    // The band really does end: half a feather out is black, half in is solid.
    assert!(at(11, 16)[0] < 0.01, "outside the band");
    assert!(at(21, 16)[0] > 0.99, "inside the band");
}

#[test]
fn a_shape_with_no_points_renders_nothing() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 2);
    comp.ensure_output(W, H);
    comp.ensure_reader(0, ImageInfo { width: W, height: H });
    let out = render(&ctx, &mut comp, &spline_graph(SplineParams::default()));
    assert!(
        out.iter().all(|p| p[0] < 0.01 && p[3] < 0.01),
        "a fresh spline draws nothing at all"
    );
}

/// Coverage the linear ramp predicts across the rect's left edge, for a
/// shape grown (or shrunk) by `expand`: 1 at `-feather/2` of the signed
/// distance, 0.5 ON the outline, 0 at `+feather/2`.
fn want_left_edge(x: u32, expand: f32) -> f32 {
    let px = x as f32 + 0.5; // the shader samples pixel centres
    let sd = (16.0 - px) - expand; // positive outside
    (0.5 - sd / FEATHER).clamp(0.0, 1.0)
}

#[test]
fn invert_is_the_complement_and_opacity_scales_what_is_left() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 2);
    comp.ensure_output(W, H);
    comp.ensure_reader(0, ImageInfo { width: W, height: H });

    let inv = spline_graph(SplineParams { invert: true, ..rect(0.25, 0.75, FEATHER) });
    let out = render(&ctx, &mut comp, &inv);
    let at = |o: &Vec<[f32; 4]>, x: u32, y: u32| o[(y * W + x) as usize];
    assert!(at(&out, 32, 16)[0] < 0.01, "inside an inverted shape is empty");
    assert!((at(&out, 1, 1)[0] - 1.0).abs() < 0.01, "outside it is solid");
    assert!((at(&out, 1, 1)[3] - 1.0).abs() < 0.01, "and alpha carries it too");
    // The soft edge is the same band, running the other way.
    for x in 12..=20u32 {
        let want = 1.0 - want_left_edge(x, 0.0);
        let got = at(&out, x, 16)[0];
        assert!((got - want).abs() < 0.02, "x={x}: {got:.3} vs {want:.3}");
    }

    // Opacity is a gain on the finished matte.
    let half = spline_graph(SplineParams { opacity: 0.5, ..rect(0.25, 0.75, FEATHER) });
    let out = render(&ctx, &mut comp, &half);
    assert!((at(&out, 32, 16)[0] - 0.5).abs() < 0.01, "inside is half covered");
    assert!((at(&out, 32, 16)[3] - 0.5).abs() < 0.01, "in alpha as well as rgb");
    assert!(at(&out, 1, 1)[0] < 0.01, "outside stays empty");
    // Half of a ramp is still the ramp: the soft edge scales with it.
    let got = at(&out, 14, 16)[0];
    let want = want_left_edge(14, 0.0) * 0.5;
    assert!((got - want).abs() < 0.02, "{got:.3} vs {want:.3}");

    // An EMPTY shape inverts to a full matte — that is what makes invert
    // usable as a garbage switch rather than a no-op.
    let empty = spline_graph(SplineParams { invert: true, ..SplineParams::default() });
    let out = render(&ctx, &mut comp, &empty);
    assert!(
        out.iter().all(|p| (p[0] - 1.0).abs() < 0.01 && (p[3] - 1.0).abs() < 0.01),
        "an inverted empty shape covers everything"
    );
}

#[test]
fn expand_moves_the_outline_and_takes_the_soft_edge_with_it() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 2);
    comp.ensure_output(W, H);
    comp.ensure_reader(0, ImageInfo { width: W, height: H });
    let at = |o: &Vec<[f32; 4]>, x: u32, y: u32| o[(y * W + x) as usize];

    // Sampled along y=16, where the LEFT edge is the nearest one for
    // roughly 9 < x < 23 — outside that the corner takes over and the
    // one-edge arithmetic below stops describing the shape.

    // Grown by 2px: the 0.5 crossing that sat at x=16 now sits at x=14, and
    // the band is still FEATHER wide around it (a matte that grew by eating
    // its own feather would be a different, worse thing).
    let g = spline_graph(SplineParams { expand: 2.0, ..rect(0.25, 0.75, FEATHER) });
    let out = render(&ctx, &mut comp, &g);
    for x in 10..=18u32 {
        let want = want_left_edge(x, 2.0);
        let got = at(&out, x, 16)[0];
        assert!((got - want).abs() < 0.02, "grown x={x}: {got:.3} vs {want:.3}");
    }
    assert!((at(&out, 18, 16)[0] - 1.0).abs() < 0.01, "what was a ramp is now solid");

    // Shrunk by 4px: the same crossing moves the other way, to x=20.
    let g = spline_graph(SplineParams { expand: -4.0, ..rect(0.25, 0.75, FEATHER) });
    let out = render(&ctx, &mut comp, &g);
    for x in 16..=22u32 {
        let want = want_left_edge(x, -4.0);
        let got = at(&out, x, 16)[0];
        assert!((got - want).abs() < 0.02, "shrunk x={x}: {got:.3} vs {want:.3}");
    }
    assert!(at(&out, 15, 16)[0] < 0.01, "what was inside the band is now out of it");
    // The middle is still covered — but only just: this rect is 16px tall,
    // so shrinking it by 4 pulls the top and bottom feather in to meet.
    assert!(at(&out, 32, 16)[0] > 0.9, "the middle went with it");
}

#[test]
fn falloff_shapes_the_band_without_moving_the_outline() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 2);
    comp.ensure_output(W, H);
    comp.ensure_reader(0, ImageInfo { width: W, height: H });
    let at = |o: &Vec<[f32; 4]>, x: u32, y: u32| o[(y * W + x) as usize];

    let g = spline_graph(SplineParams { falloff: 1.0, ..rect(0.25, 0.75, FEATHER) });
    let out = render(&ctx, &mut comp, &g);
    for x in 12..=20u32 {
        let lin = want_left_edge(x, 0.0);
        let want = lin * lin * (3.0 - 2.0 * lin); // smoothstep
        let got = at(&out, x, 16)[0];
        assert!((got - want).abs() < 0.02, "x={x}: {got:.3} vs smooth {want:.3}");
    }
    // 0.5 still lands ON the outline: smoothstep fixes 0.5, so the crossing
    // does not move, and the gizmo's rings mean the same thing at every
    // falloff — the whole reason the ramp is shaped this way and not with a
    // gamma.
    assert!(at(&out, 15, 16)[0] < 0.5, "x=15.5 is outside the outline at 16");
    assert!(at(&out, 16, 16)[0] > 0.5, "x=16.5 is inside it");
    // Inside and outside are untouched — this only bends the middle.
    assert!((at(&out, 32, 16)[0] - 1.0).abs() < 0.01);
    assert!(at(&out, 1, 1)[0] < 0.01);
}

#[test]
fn the_shape_transform_moves_the_matte_the_gpu_rasterizes() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 2);
    comp.ensure_output(W, H);
    comp.ensure_reader(0, ImageInfo { width: W, height: H });
    let at = |o: &Vec<[f32; 4]>, x: u32, y: u32| o[(y * W + x) as usize];

    // A quarter of the frame to the right: x 16..48 becomes x 32..64.
    let g = spline_graph(SplineParams {
        xf: ShapeXf { translate: [0.25, 0.0], ..Default::default() },
        ..rect(0.25, 0.75, FEATHER)
    });
    let out = render(&ctx, &mut comp, &g);
    assert!((at(&out, 48, 16)[0] - 1.0).abs() < 0.01, "the shape moved with it");
    assert!(at(&out, 20, 16)[0] < 0.01, "and left where it was");
    for x in 26..=38u32 {
        let px = x as f32 + 0.5;
        let want = (0.5 - (32.0 - px) / FEATHER).clamp(0.0, 1.0);
        let got = at(&out, x, 16)[0];
        assert!((got - want).abs() < 0.02, "moved edge x={x}: {got:.3} vs {want:.3}");
    }

    // Scaling about the frame centre: half size puts the left edge at 24.
    let g = spline_graph(SplineParams {
        xf: ShapeXf { scale: [0.5, 0.5], ..Default::default() },
        ..rect(0.25, 0.75, FEATHER)
    });
    let out = render(&ctx, &mut comp, &g);
    // Half size makes the rect 16x8, so the centre sits 3.5px from the
    // nearest (top) edge and its feather reaches it — covered, not quite 1.
    assert!(at(&out, 32, 16)[0] > 0.9, "still covered at the centre");
    assert!(at(&out, 18, 16)[0] < 0.01, "but no longer out at x=18");
    let on = at(&out, 24, 16)[0];
    assert!((on - 0.5625).abs() < 0.05, "the new edge is at x=24, not {on:.3}");
}

#[test]
fn chained_splines_combine_into_one_matte() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 2);
    comp.ensure_output(W, H);
    comp.ensure_reader(0, ImageInfo { width: W, height: H });
    let at = |o: &Vec<[f32; 4]>, x: u32, y: u32| o[(y * W + x) as usize];

    // A hole: the big rect on the input, a small hard-edged one subtracted.
    // This is what roto with several shapes looks like here — one node per
    // shape, wired in a chain, no multi-shape editor to learn.
    let hole = |c: MatteOp| {
        let mut g = Graph::default();
        g.nodes.insert(8, NodeKind::Spline(rect(0.25, 0.75, FEATHER)));
        g.nodes
            .insert(7, NodeKind::Spline(SplineParams { combine: c, ..rect(0.4, 0.6, 2.0) }));
        g.edges.insert((7, 0), 8);
        g.view = Some(7);
        g
    };

    let out = render(&ctx, &mut comp, &hole(MatteOp::Subtract));
    assert!(at(&out, 32, 16)[0] < 0.01, "the small shape punched a hole");
    assert!((at(&out, 20, 16)[0] - 1.0).abs() < 0.01, "the big one survives around it");
    assert!(at(&out, 1, 1)[0] < 0.01, "and nothing leaks outside");
    for p in &out {
        assert!((p[3] - p[0]).abs() < 0.01, "combined matte reaches alpha: {p:?}");
    }

    // Intersect keeps only the overlap — here, the small shape.
    let out = render(&ctx, &mut comp, &hole(MatteOp::Intersect));
    assert!((at(&out, 32, 16)[0] - 1.0).abs() < 0.01, "the overlap stays");
    assert!(at(&out, 20, 16)[0] < 0.01, "what only the input had is gone");

    // Replace ignores the input entirely: just the small shape.
    let out = render(&ctx, &mut comp, &hole(MatteOp::Replace));
    assert!((at(&out, 32, 16)[0] - 1.0).abs() < 0.01);
    assert!(at(&out, 20, 16)[0] < 0.01, "the input matte was dropped");

    // Union is the default, and an unwired Union is the bare shape — the
    // contract that keeps every setup written before the input existed
    // rendering exactly as it did.
    let out = render(&ctx, &mut comp, &hole(MatteOp::Union));
    assert!((at(&out, 32, 16)[0] - 1.0).abs() < 0.01);
    assert!((at(&out, 20, 16)[0] - 1.0).abs() < 0.01, "both shapes are there");
    let bare = render(&ctx, &mut comp, &spline_graph(rect(0.25, 0.75, FEATHER)));
    let wired_only = render(
        &ctx,
        &mut comp,
        &spline_graph(SplineParams {
            combine: MatteOp::Union,
            ..rect(0.25, 0.75, FEATHER)
        }),
    );
    for (a, b) in bare.iter().zip(&wired_only) {
        assert!((a[0] - b[0]).abs() < 1e-3, "unwired union changed the shape");
    }
}

/// Per-point TENSION reaches the matte the GPU draws, and it is a param,
/// not topology. A Catmull-Rom rectangle bulges past its own corners'
/// box; with every point at tension 1 it is the rectangle again. Hard
/// edged, so any coverage outside the box is the curve, not a feather.
#[test]
fn tension_reaches_the_rendered_matte_without_a_recompile() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 2);
    comp.ensure_output(W, H);
    comp.ensure_reader(0, ImageInfo { width: W, height: H });
    let shape = |t: f32| {
        let mut sp = rect(0.25, 0.75, 0.0);
        sp.mode = SplineMode::CatmullRom;
        for q in &mut sp.points {
            q.tension = t;
        }
        spline_graph(sp)
    };
    // Alpha outside x 16..48, y 8..24 (with a pixel of AA slack).
    let outside = |img: &[[f32; 4]]| -> f32 {
        let mut sum = 0.0;
        for y in 0..H as usize {
            for x in 0..W as usize {
                let inside = (15..=48).contains(&x) && (7..=24).contains(&y);
                if !inside {
                    sum += img[y * W as usize + x][3];
                }
            }
        }
        sum
    };
    let loose = shape(0.0);
    let tight = shape(1.0);
    assert_eq!(loose.topology_hash(), tight.topology_hash(), "tension recompiled");
    let a = render(&ctx, &mut comp, &loose);
    let b = render(&ctx, &mut comp, &tight);
    assert!(outside(&a) > 2.0, "an untensioned Catmull-Rom rect should bulge ({})", outside(&a));
    assert!(outside(&b) < 0.5, "tension 1 should sit on the rect ({})", outside(&b));
}
