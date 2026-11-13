//! Headless integration for the noise-field paradigm.
//!
//! The unit tests in `field.rs` prove the right WGSL comes out. These prove
//! it MEANS something once a GPU runs it — above all that a parameter driven
//! by another field really does vary across the picture, which is the entire
//! point of the thing and the one claim that a shader could quietly fail to
//! honour while still compiling. Self-skips without a GPU (CI).

use engine::compositor::Compositor;
use engine::field::{CoordKind, FieldFbmMode, FieldGraph, FieldNode, VoronoiMode, WarpMode};
use engine::graph::{Graph, NodeKind, NoiseParams};
use engine::timing::GpuTimer;
use half::f16;

const W: u32 = 256;
const H: u32 = 128;

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP field test: no adapter ({e})");
            None
        }
    }
}

/// The whole red channel, row-major.
fn render(ctx: &engine::GpuContext, comp: &mut Compositor, graph: &Graph) -> Vec<f32> {
    render_ch(ctx, comp, graph, 0)
}

/// One channel (0 = red, 1 = green, …), row-major.
fn render_ch(ctx: &engine::GpuContext, comp: &mut Compositor, graph: &Graph, ch: usize) -> Vec<f32> {
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
    let mut out = Vec::with_capacity((W * H) as usize);
    for y in 0..H as usize {
        let off = y * padded as usize;
        let halves: &[f16] = bytemuck::cast_slice(&data[off..off + (W * 8) as usize]);
        for x in 0..W as usize {
            out.push(halves[x * 4 + ch].to_f32());
        }
    }
    drop(data);
    staging.unmap();
    out
}

/// A comp graph holding one Noise node with `sub` inside it.
fn comp_of(sub: FieldGraph) -> Graph {
    let mut g = Graph::default();
    g.nodes.insert(
        1,
        NodeKind::Noise(NoiseParams {
            size: (W, H),
            graph: sub,
            scale: 1.0,
            seed: 0.0,
            probe: [0.0, 0.0],
            ..Default::default()
        }),
    );
    g.view = Some(1);
    g
}

fn fbm(scale: f32) -> FieldNode {
    FieldNode::Fbm {
        mode: FieldFbmMode::Fbm,
        scale: [scale, scale],
        offset: [0.0, 0.0],
        lacunarity: 2.0,
        gain: 0.5,
        seed: 0.0,
        octaves: 4,
        speed: 1.0,
        uniform: 1.0,
    }
}

/// Mean absolute difference between horizontally adjacent pixels inside a
/// column band — a cheap proxy for how FINE the detail is there.
fn detail(img: &[f32], x0: usize, x1: usize) -> f32 {
    let mut acc = 0.0;
    let mut n = 0.0f32;
    for y in 0..H as usize {
        for x in x0..x1.saturating_sub(1) {
            let i = y * W as usize + x;
            acc += (img[i + 1] - img[i]).abs();
            n += 1.0;
        }
    }
    acc / n.max(1.0)
}

fn stats(img: &[f32]) -> (f32, f32, f32) {
    let lo = img.iter().cloned().fold(f32::MAX, f32::min);
    let hi = img.iter().cloned().fold(f32::MIN, f32::max);
    let mean = img.iter().sum::<f32>() / img.len() as f32;
    (lo, hi, mean)
}

#[test]
fn a_bare_fbm_field_renders_a_field() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    let mut sub = FieldGraph::default();
    sub.nodes.insert(1, fbm(0.25));
    sub.out = Some(1);
    let (lo, hi, mean) = stats(&render(&ctx, &mut comp, &comp_of(sub)));
    assert!(hi - lo > 0.4, "the field is flat: {lo:.3}..{hi:.3}");
    assert!((0.2..0.8).contains(&mean), "mean {mean:.3} is stuck at an end");
}

/// THE test. An fBm whose feature size is driven by a Coord field must be
/// coarse where that field says coarse and fine where it says fine. A shader
/// that read the literal instead would produce one uniform frequency across
/// the frame and pass every other test in this file.
#[test]
fn a_field_driven_parameter_varies_across_the_picture() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);

    let mut sub = FieldGraph::default();
    // x (0..1 across the frame) -> feature size 0.5 on the left, 0.02 on the
    // right, and that number becomes the fBm's scale on both axes.
    sub.nodes.insert(1, FieldNode::Coord { kind: CoordKind::X, scale: 1.0 });
    sub.nodes.insert(
        2,
        FieldNode::Remap {
            in_lo: 0.0,
            in_hi: 1.0,
            out_lo: 0.5,
            out_hi: 0.02,
            clamp: true,
        },
    );
    sub.nodes.insert(3, fbm(0.25));
    sub.edges.insert((2, 0), 1);
    sub.edges.insert((3, 0), 2); // scale x
    sub.edges.insert((3, 1), 2); // scale y
    sub.out = Some(3);
    let driven = render(&ctx, &mut comp, &comp_of(sub));

    let left = detail(&driven, 8, 72);
    let right = detail(&driven, 184, 248);
    assert!(
        right > left * 3.0,
        "the driven scale did nothing: detail {left:.4} left vs {right:.4} right"
    );

    // And the same graph with the driver UNWIRED is uniform across the
    // frame, which is what makes the comparison above mean something.
    let mut flat = FieldGraph::default();
    flat.nodes.insert(3, fbm(0.1));
    flat.out = Some(3);
    let plain = render(&ctx, &mut comp, &comp_of(flat));
    let (l, r) = (detail(&plain, 8, 72), detail(&plain, 184, 248));
    assert!(
        (l - r).abs() < l.max(r) * 0.5,
        "an undriven fBm should be statistically the same everywhere: {l:.4} vs {r:.4}"
    );
}

/// A Warp rewrites the coordinate its source is evaluated at — the gesture
/// the whole "coordinates flow upstream" design exists for.
#[test]
fn a_warp_moves_the_field_it_reads() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);

    let plain = {
        let mut sub = FieldGraph::default();
        sub.nodes.insert(1, fbm(0.2));
        sub.out = Some(1);
        render(&ctx, &mut comp, &comp_of(sub))
    };
    // A constant offset of 0.25 in x: the same field, displaced.
    let warped = {
        let mut sub = FieldGraph::default();
        sub.nodes.insert(1, fbm(0.2));
        sub.nodes.insert(2, FieldNode::Const { value: 0.25, name: String::new() });
        sub.nodes.insert(
            3,
            FieldNode::Warp { mode: WarpMode::Offset, amount: 1.0 },
        );
        sub.edges.insert((3, 0), 1); // field
        sub.edges.insert((3, 1), 2); // x offset
        sub.out = Some(3);
        render(&ctx, &mut comp, &comp_of(sub))
    };
    let diff: f32 = plain
        .iter()
        .zip(warped.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / plain.len() as f32;
    assert!(diff > 0.05, "the warp did not move anything (mean diff {diff:.4})");

    // Displaced, not destroyed: the warped field has the same statistics.
    let (_, _, m0) = stats(&plain);
    let (_, _, m1) = stats(&warped);
    assert!((m0 - m1).abs() < 0.08, "the warp changed the field itself: {m0:.3} vs {m1:.3}");
}

/// Seed flows upstream too: the same sub-graph at a different seed is a
/// different field, and nothing else about it changes.
#[test]
fn a_seed_shift_reshuffles_everything_above_it() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    let build = |shift: f32| {
        let mut sub = FieldGraph::default();
        sub.nodes.insert(1, fbm(0.2));
        sub.nodes.insert(2, FieldNode::Seed { shift });
        sub.edges.insert((2, 0), 1);
        sub.out = Some(2);
        comp_of(sub)
    };
    let a = render(&ctx, &mut comp, &build(0.0));
    let b = render(&ctx, &mut comp, &build(64.0));
    let diff: f32 =
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum::<f32>() / a.len() as f32;
    assert!(diff > 0.05, "the seed shift changed nothing (mean diff {diff:.4})");
    let (_, _, ma) = stats(&a);
    let (_, _, mb) = stats(&b);
    assert!((ma - mb).abs() < 0.08, "a reseed should not change the statistics");
}

/// Voronoi's modes are genuinely different fields, and `cell id` is flat
/// within a cell — the property that makes it useful for regions.
#[test]
fn voronoi_modes_differ_and_cells_are_flat() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    let build = |mode: VoronoiMode| {
        let mut sub = FieldGraph::default();
        sub.nodes.insert(
            1,
            FieldNode::Voronoi {
                mode,
                scale: [0.2, 0.2],
                jitter: 1.0,
                seed: 3.0,
                speed: 1.0,
            },
        );
        sub.out = Some(1);
        comp_of(sub)
    };
    let f1 = render(&ctx, &mut comp, &build(VoronoiMode::F1));
    let ridges = render(&ctx, &mut comp, &build(VoronoiMode::Ridges));
    let cells = render(&ctx, &mut comp, &build(VoronoiMode::Cell));
    let d: f32 = f1
        .iter()
        .zip(ridges.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / f1.len() as f32;
    assert!(d > 0.05, "F1 and F2-F1 should not be the same picture");
    // Cell ids are piecewise constant: almost every neighbouring pair is
    // identical, and the few that differ are the cell borders.
    let flat = (0..cells.len() - 1)
        .filter(|i| (i + 1) % W as usize != 0)
        .filter(|i| (cells[*i] - cells[i + 1]).abs() < 1e-3)
        .count() as f32;
    let frac = flat / cells.len() as f32;
    assert!(frac > 0.85, "cell ids should be flat inside a cell, only {frac:.2} were");
}

/// The invariant the whole engine runs on, extended to the sub-graph: a
/// number changing anywhere inside a noise graph is a param write, and only
/// rewiring recompiles.
#[test]
fn a_number_inside_the_sub_graph_is_not_a_recompile() {
    let mut sub = FieldGraph::default();
    sub.nodes.insert(1, fbm(0.25));
    sub.nodes.insert(2, FieldNode::Const { value: 1.0, name: String::new() });
    sub.edges.insert((1, 0), 2);
    sub.out = Some(1);
    let a = comp_of(sub.clone());

    let mut sub2 = sub.clone();
    if let Some(FieldNode::Const { value, .. }) = sub2.nodes.get_mut(&2) {
        *value = 42.0;
    }
    let b = comp_of(sub2);
    assert_eq!(a.topology_hash(), b.topology_hash(), "a knob must not recompile");
    assert_eq!(
        engine::codegen::generate(&a).main.wgsl,
        engine::codegen::generate(&b).main.wgsl,
        "the same shape must produce the same shader"
    );

    let mut sub3 = sub.clone();
    sub3.edges.remove(&(1, 0));
    let c = comp_of(sub3);
    assert_ne!(a.topology_hash(), c.topology_hash(), "rewiring must recompile");
}

/// The CPU evaluator and the shader must agree, or a noise graph means one
/// thing when it drives a linked value and another when it renders. Every
/// hash, fade and per-mode normalisation is duplicated between the two, so
/// this is the test that stops them drifting apart.
#[test]
fn cpu_matches_the_gpu() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);

    // Something with a driven parameter, a warp and a voronoi in it — not
    // just a bare fBm, so the agreement covers the composition too.
    let mut sub = FieldGraph::default();
    sub.nodes.insert(1, FieldNode::Coord { kind: CoordKind::X, scale: 1.0 });
    sub.nodes.insert(
        2,
        FieldNode::Remap { in_lo: 0.0, in_hi: 1.0, out_lo: 0.3, out_hi: 0.05, clamp: true },
    );
    sub.nodes.insert(3, fbm(0.25));
    sub.nodes.insert(
        4,
        FieldNode::Voronoi {
            mode: VoronoiMode::Ridges,
            scale: [0.15, 0.15],
            jitter: 0.8,
            seed: 2.0,
            speed: 1.0,
        },
    );
    sub.nodes.insert(5, FieldNode::Math { op: engine::field::MathOp::Mul, a: 1.0, b: 1.0 });
    sub.edges.insert((2, 0), 1);
    sub.edges.insert((3, 0), 2);
    sub.edges.insert((3, 1), 2);
    sub.edges.insert((5, 0), 3);
    sub.edges.insert((5, 1), 4);
    sub.out = Some(5);

    let img = render(&ctx, &mut comp, &comp_of(sub.clone()));
    // `pos` in the shader is the pixel CENTRE, and field space is normalized
    // on the width, so both axes divide by W.
    let mut worst = 0.0f32;
    for (x, y) in [(3, 5), (40, 17), (128, 64), (200, 100), (255, 127), (77, 3)] {
        let p = [(x as f32 + 0.5) / W as f32, (y as f32 + 0.5) / W as f32];
        let cpu = engine::field::eval_cpu(&sub, p, 0.0, 0.0, 1.0);
        let gpu = img[y * W as usize + x];
        worst = worst.max((cpu - gpu).abs());
        assert!(
            (cpu - gpu).abs() < 6e-3,
            "pixel ({x},{y}): cpu {cpu:.5} vs gpu {gpu:.5}"
        );
    }
    // Not vacuously equal: the field has to actually be doing something.
    assert!(img.iter().cloned().fold(f32::MIN, f32::max) > 0.1, "the field is empty");
    println!("worst cpu/gpu disagreement: {worst:.5}");
}
/// Animation: one global speed on the Noise node, scaled by each
/// generator's own multiplier — so a single node can be FROZEN while the
/// rest of the graph moves, or run at double.
#[test]
fn each_generator_scales_the_global_speed() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);

    // The same field at two moments, with the generator's multiplier varied.
    let at = |comp: &mut Compositor, mult: f32, time: f32| {
        let mut sub = FieldGraph::default();
        sub.nodes.insert(
            1,
            FieldNode::Fbm {
                mode: FieldFbmMode::Fbm,
                scale: [0.25, 0.25],
                offset: [0.0, 0.0],
                lacunarity: 2.0,
                gain: 0.5,
                seed: 0.0,
                octaves: 4,
                speed: mult,
                uniform: 1.0,
            },
        );
        sub.out = Some(1);
        let mut g = Graph::default();
        g.nodes.insert(
            1,
            NodeKind::Noise(NoiseParams {
                size: (W, H),
                graph: sub,
                scale: 1.0,
                speed: 1.0,
                time,
                ..Default::default()
            }),
        );
        g.view = Some(1);
        render(&ctx, comp, &g)
    };
    let diff = |a: &[f32], b: &[f32]| {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum::<f32>() / a.len() as f32
    };

    let still = at(&mut comp, 1.0, 0.0);

    // Frozen: the global speed is 1, but this generator's multiplier is 0.
    let frozen_a = at(&mut comp, 0.0, 0.0);
    let frozen_b = at(&mut comp, 0.0, 9.0);
    assert!(
        diff(&frozen_a, &frozen_b) < 1e-4,
        "a multiplier of 0 must hold the field still while time passes"
    );

    // Running: the same field, the same clock, moves.
    let d1 = diff(&still, &at(&mut comp, 1.0, 0.05));
    assert!(d1 > 2e-3, "speed 1 should evolve the field (moved {d1:.4})");

    // And the multiplier SCALES rather than merely gating: twice the rate
    // has gone further by the same moment.
    //
    // Measured over a SMALL step on purpose. `z` advances in noise-lattice
    // units (the same convention the image-side fBm's `evolve` uses), so a
    // step approaching 1 has already decorrelated the field — past that the
    // difference saturates and every rate looks alike, which says nothing
    // about the rate.
    let d2 = diff(&still, &at(&mut comp, 2.0, 0.05));
    assert!(
        d2 > d1 * 1.2,
        "double speed should move further by the same tick ({d2:.4} vs {d1:.4})"
    );
}

/// The reported use: a matte from the comp drives the field, so the noise is
/// FINER where the mask is and unchanged where it is not. This is the only
/// thing in a field graph that comes from outside it, so it gets its own
/// test end to end — plate to matte to field to pixels.
#[test]
fn a_mask_from_the_comp_changes_the_field_only_where_it_is() {
    use engine::decode::ImageInfo;
    use engine::graph::{MatteChannel, ToMatteParams};
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);

    // A plate that is white on the LEFT half and black on the right.
    comp.ensure_reader(0, ImageInfo { width: W, height: H });
    let px: Vec<[f16; 4]> = (0..W * H)
        .map(|i| {
            let on = (i % W) < W / 2;
            let v = f16::from_f32(if on { 1.0 } else { 0.0 });
            [v, v, v, f16::ONE]
        })
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

    // Inside: feature size mixes from coarse (mask 0) to fine (mask 1), and
    // that number is the fBm's scale. "Leave it as it was where the mask is
    // 0, apply it where the mask is 1" is exactly a Mix through the mask.
    let mut sub = FieldGraph::default();
    sub.nodes.insert(1, FieldNode::MaskIn);
    sub.nodes.insert(2, FieldNode::Mix { a: 0.5, b: 0.03, t: 0.0 });
    sub.nodes.insert(3, fbm(0.25));
    sub.edges.insert((2, 2), 1); // the mask is the blend amount
    sub.edges.insert((3, 0), 2); // scale x
    sub.edges.insert((3, 1), 2); // scale y
    sub.out = Some(3);

    let mut g = Graph::default();
    g.nodes.insert(0, NodeKind::read(0, ""));
    g.nodes.insert(
        1,
        NodeKind::ToMatte(ToMatteParams { channel: MatteChannel::R, ..Default::default() }),
    );
    g.edges.insert((1, 0), 0);
    g.nodes.insert(
        2,
        NodeKind::Noise(NoiseParams { size: (W, H), graph: sub, scale: 1.0, ..Default::default() }),
    );
    g.edges.insert((2, 0), 1); // the matte into the Noise
    g.view = Some(2);
    let masked = render(&ctx, &mut comp, &g);

    let inside = detail(&masked, 8, 112);   // mask 1: fine
    let outside = detail(&masked, 144, 248); // mask 0: coarse
    assert!(
        inside > outside * 3.0,
        "the mask should make the field finer where it is: {inside:.4} in vs {outside:.4} out"
    );

    // Unwire it and the mask reads 1 everywhere — the same convention every
    // other matte input here follows — so the whole frame goes fine.
    g.edges.remove(&(2, 0));
    let unwired = render(&ctx, &mut comp, &g);
    let l = detail(&unwired, 8, 112);
    let r = detail(&unwired, 144, 248);
    assert!(
        (l - r).abs() < l.max(r) * 0.5,
        "an unwired mask should read 1 everywhere: {l:.4} vs {r:.4}"
    );
}

/// A mask made INSIDE the field graph: no comp input, no round trip. The
/// noise is finer inside the shape and coarse outside it, and the shape's
/// own numbers are drivable like any other socket.
#[test]
fn a_shape_masks_the_field_without_leaving_the_graph() {
    use engine::field::ShapeKind;
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);

    // A box covering the LEFT half of the frame. Field space is normalized
    // on the width, so the frame runs 0..1 across and 0..H/W down.
    let mut sub = FieldGraph::default();
    sub.nodes.insert(
        1,
        FieldNode::Shape {
            kind: ShapeKind::Box,
            center: [0.25, 0.25],
            size: [0.25, 1.0],
            feather: 0.02,
        },
    );
    sub.nodes.insert(2, FieldNode::Mix { a: 0.5, b: 0.03, t: 0.0 });
    sub.nodes.insert(3, fbm(0.25));
    sub.edges.insert((2, 2), 1);
    sub.edges.insert((3, 0), 2);
    sub.edges.insert((3, 1), 2);
    sub.out = Some(3);

    let img = render(&ctx, &mut comp, &comp_of(sub.clone()));
    let inside = detail(&img, 8, 112);
    let outside = detail(&img, 144, 248);
    assert!(
        inside > outside * 3.0,
        "the shape should make the field finer inside it: {inside:.4} vs {outside:.4}"
    );

    // The shape itself, straight out: 1 inside, 0 outside, and it is the
    // same field the CPU evaluator reports.
    let mut only = sub.clone();
    only.out = Some(1);
    assert!(engine::field::eval_cpu(&only, [0.1, 0.2], 0.0, 0.0, 1.0) > 0.99);
    assert!(engine::field::eval_cpu(&only, [0.9, 0.2], 0.0, 0.0, 1.0) < 0.01);

    // Driving a socket moves the shape: a centre pushed right takes the
    // fine noise with it.
    let mut driven = sub.clone();
    driven.nodes.insert(4, FieldNode::Const { value: 0.75, name: String::new() });
    driven.edges.insert((1, 0), 4); // centre x
    let moved = render(&ctx, &mut comp, &comp_of(driven));
    let now_in = detail(&moved, 144, 248);
    let now_out = detail(&moved, 8, 112);
    assert!(
        now_in > now_out * 3.0,
        "driving the centre should move the mask: {now_in:.4} right vs {now_out:.4} left"
    );
}

/// A roto drawn IN the field graph: the comp Spline's outline in field
/// units. Inside is 1, outside 0, the offset socket moves it, and the CPU
/// evaluator agrees with the GPU about where the edge is.
#[test]
fn a_field_roto_masks_and_moves() {
    use engine::graph::{SplineMode, SplinePoint};
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    // A square over the left half of the frame, in field units (y in width
    // units, so the frame is 0..0.5 tall).
    let pt = |x: f32, y: f32| SplinePoint {
        p: [x, y],
        feather: 0.01,
        bias: 0.0,
        h_in: [0.0, 0.0],
        h_out: [0.0, 0.0],
        tension: 0.0,
    };
    let roto = FieldNode::Spline {
        points: vec![pt(0.05, 0.05), pt(0.45, 0.05), pt(0.45, 0.45), pt(0.05, 0.45)],
        mode: SplineMode::Linear,
        feather: 1.0,
        offset: [0.0, 0.0],
    };
    let mut sub = FieldGraph::default();
    sub.nodes.insert(1, roto.clone());
    sub.nodes.insert(2, FieldNode::Mix { a: 0.5, b: 0.03, t: 0.0 });
    sub.nodes.insert(3, fbm(0.25));
    sub.edges.insert((2, 2), 1);
    sub.edges.insert((3, 0), 2);
    sub.edges.insert((3, 1), 2);
    sub.out = Some(3);
    let img = render(&ctx, &mut comp, &comp_of(sub.clone()));
    let (inside, outside) = (detail(&img, 20, 108), detail(&img, 148, 240));
    assert!(inside > outside * 3.0, "roto should make the field finer inside: {inside:.4} vs {outside:.4}");

    // The roto alone, CPU and GPU: 1 well inside, 0 well outside.
    let mut only = FieldGraph::default();
    only.nodes.insert(1, roto.clone());
    only.out = Some(1);
    let gpu = render(&ctx, &mut comp, &comp_of(only.clone()));
    let at = |x: usize, y: usize| gpu[y * W as usize + x];
    assert!(at(64, 64) > 0.98 && at(200, 64) < 0.02, "gpu: {} inside, {} outside", at(64, 64), at(200, 64));
    let cpu_in = engine::field::eval_cpu(&only, [0.25, 0.25], 0.0, 0.0, 1.0);
    let cpu_out = engine::field::eval_cpu(&only, [0.8, 0.25], 0.0, 0.0, 1.0);
    assert!(cpu_in > 0.98 && cpu_out < 0.02, "cpu: {cpu_in} inside, {cpu_out} outside");

    // The offset socket slides the whole roto: driven by a Const, the square
    // moves to the right half.
    let mut moved = only.clone();
    moved.nodes.insert(9, FieldNode::Const { value: 0.5, name: String::new() });
    moved.edges.insert((1, 0), 9); // offset x
    let img2 = render(&ctx, &mut comp, &comp_of(moved));
    let at2 = |x: usize, y: usize| img2[y * W as usize + x];
    assert!(at2(200, 64) > 0.98 && at2(64, 64) < 0.02, "moved: {} right, {} left", at2(200, 64), at2(64, 64));
}

/// Not an assertion — a way to LOOK at it. `cargo test -p engine --test
/// field -- --ignored dump` writes a PPM per graph into `$WOBBLE_DUMP_DIR`.
#[test]
#[ignore]
fn dump_graphs_for_eyeballing() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    let dir = std::env::var("WOBBLE_DUMP_DIR").unwrap_or_else(|_| "/tmp".into());

    // 1: an fBm whose feature size is itself a field — coarse to fine.
    let mut a = FieldGraph::default();
    a.nodes.insert(1, FieldNode::Coord { kind: CoordKind::X, scale: 1.0 });
    a.nodes.insert(
        2,
        FieldNode::Remap {
            in_lo: 0.0,
            in_hi: 1.0,
            out_lo: 0.4,
            out_hi: 0.02,
            clamp: true,
        },
    );
    a.nodes.insert(3, fbm(0.25));
    a.edges.insert((2, 0), 1);
    a.edges.insert((3, 0), 2);
    a.edges.insert((3, 1), 2);
    a.out = Some(3);

    // 2: a curl warp driven by a coarse noise — one noise stirring another.
    let mut b = FieldGraph::default();
    b.nodes.insert(1, fbm(0.06));
    b.nodes.insert(2, fbm(0.4));
    b.nodes.insert(3, FieldNode::Warp { mode: WarpMode::Curl, amount: 0.02 });
    b.edges.insert((3, 0), 1);
    b.edges.insert((3, 1), 2);
    b.out = Some(3);

    // 3: voronoi cells whose jitter is driven by noise.
    let mut c = FieldGraph::default();
    c.nodes.insert(1, fbm(0.3));
    c.nodes.insert(
        2,
        FieldNode::Voronoi {
            mode: VoronoiMode::Ridges,
            scale: [0.12, 0.12],
            jitter: 1.0,
            seed: 5.0,
            speed: 1.0,
        },
    );
    c.edges.insert((2, 2), 1); // jitter
    c.out = Some(2);

    for (name, sub) in [("driven_scale", a), ("curl_warp", b), ("voronoi_jitter", c)] {
        let img = render(&ctx, &mut comp, &comp_of(sub));
        let mut out = format!("P2\n{W} {H}\n255\n");
        for v in &img {
            out.push_str(&format!("{} ", (v.clamp(0.0, 1.0) * 255.0) as u32));
        }
        let path = format!("{dir}/field_{name}.pgm");
        std::fs::write(&path, out).unwrap();
        println!("wrote {path}");
    }
}

/// Every stock preset the Noise panel loads builds a shader the GPU takes and
/// draws a texture with it — not a flat card. `--ignored dump_presets` writes
/// one PGM each into `$WOBBLE_DUMP_DIR` for looking at.
#[test]
fn every_preset_renders_a_texture() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    for p in engine::field::presets::NoisePreset::ALL {
        let img = render(&ctx, &mut comp, &comp_of(p.graph()));
        let (lo, hi, _) = stats(&img);
        assert!(hi - lo > 0.3, "{}: flat on the GPU ({lo}..{hi})", p.label());
        assert!(detail(&img, 0, W as usize) > 1e-3, "{}: no detail", p.label());
    }
}

#[test]
#[ignore]
fn dump_presets() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    let dir = std::env::var("WOBBLE_DUMP_DIR").unwrap_or_else(|_| "/tmp".into());
    for p in engine::field::presets::NoisePreset::ALL {
        let img = render(&ctx, &mut comp, &comp_of(p.graph()));
        let mut out = format!("P2\n{W} {H}\n255\n");
        for v in &img {
            out.push_str(&format!("{} ", (v.clamp(0.0, 1.0) * 255.0) as u32));
        }
        let path = format!("{dir}/preset_{}.pgm", p.label().replace(' ', "_"));
        std::fs::write(&path, out).unwrap();
        println!("wrote {path}");
    }
}

/// Two dimensions: red is exactly the one-dimensional field, green the same
/// graph at seed + offset — the CPU evaluator's answer there, not red again —
/// and moving the offset is a uniform, not a recompile.
#[test]
fn a_two_d_noise_puts_the_offset_seed_in_green() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    let sub = engine::field::presets::NoisePreset::Liquid.graph();
    let one = comp_of(sub.clone());
    let red_1d = render(&ctx, &mut comp, &one);
    let mut two = one.clone();
    if let Some(NodeKind::Noise(n)) = two.nodes.get_mut(&1) {
        n.two_d = true;
        n.seed_offset = 57.0;
    }
    assert_ne!(one.topology_hash(), two.topology_hash(), "2d emits a second call");
    let red = render_ch(&ctx, &mut comp, &two, 0);
    let green = render_ch(&ctx, &mut comp, &two, 1);
    let blue = render_ch(&ctx, &mut comp, &two, 2);
    let same = red.iter().zip(&red_1d).all(|(a, b)| (a - b).abs() < 1e-3);
    assert!(same, "red must be the 1-d field unchanged");
    assert!(blue.iter().all(|b| *b == 0.0), "blue is empty in 2d");
    for (x, y) in [(3, 5), (40, 17), (128, 64), (200, 100), (255, 127)] {
        let p = [(x as f32 + 0.5) / W as f32, (y as f32 + 0.5) / W as f32];
        let cpu = engine::field::eval_cpu(&sub, p, 57.0, 0.0, 1.0);
        let gpu = green[y * W as usize + x];
        assert!((cpu - gpu).abs() < 6e-3, "green ({x},{y}): cpu {cpu:.5} vs gpu {gpu:.5}");
    }
    let diff = red.iter().zip(&green).map(|(a, b)| (a - b).abs()).sum::<f32>() / red.len() as f32;
    assert!(diff > 0.05, "green is red again (mean |r-g| {diff})");
    // Same character: the two channels spread alike.
    let (rl, rh, _) = stats(&red);
    let (gl, gh, _) = stats(&green);
    assert!(((rh - rl) - (gh - gl)).abs() < 0.35, "ranges {rl}..{rh} vs {gl}..{gh}");
    let mut moved = two.clone();
    if let Some(NodeKind::Noise(n)) = moved.nodes.get_mut(&1) {
        n.seed_offset = 58.0;
    }
    assert_eq!(two.topology_hash(), moved.topology_hash(), "the offset is a knob");
    let green2 = render_ch(&ctx, &mut comp, &moved, 1);
    assert!(green2 != green, "a new offset reshuffles green");
}
