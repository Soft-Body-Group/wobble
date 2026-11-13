//! Headless integration for the Paint node: what the stroke layers actually
//! rasterize and what the fused pass does with them.
//!
//! Three claims live here. A colour stroke lands where it was painted, at
//! the alpha it was painted with. A CLONE stroke lands the input's pixels
//! from the offset instead — which is the one thing that costs a second
//! evaluation of the whole upstream chain, so it had better be the pixels
//! and not something approximating them. And a stroke's LIFETIME decides
//! which frames it appears on at all, in the render, not just in the data.
//!
//! The upstream plate is a Spline rectangle rather than a decoded frame:
//! its coverage is exactly known (see `spline.rs`), so "the clone moved
//! these pixels" is an arithmetic claim, not an eyeball.
//!
//! Self-skips without a GPU (CI).

use engine::compositor::Compositor;
use engine::decode::ImageInfo;
use engine::graph::{
    BrushMode, Graph, NodeKind, PaintBlend, PaintOutput, PaintParams, PaintStroke, SplineMode,
    SplineParams, SplinePoint, StrokeLife,
};
use engine::timing::GpuTimer;
use half::f16;

const W: u32 = 64;
const H: u32 = 32;

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP paint test: no adapter ({e})");
            None
        }
    }
}

fn render(
    ctx: &engine::GpuContext,
    comp: &mut Compositor,
    graph: &Graph,
    frame: u64,
) -> Vec<[f32; 4]> {
    assert!(comp.set_graph(graph), "fused shader failed to compile");
    let timer = GpuTimer::new(ctx);
    comp.set_paint_frame(frame);
    comp.write_params(graph);
    // One encode to record the stroke layers, one to read.
    let mut warm = ctx.device.create_command_encoder(&Default::default());
    comp.encode_comp(&mut warm, &timer);
    ctx.queue.submit([warm.finish()]);
    comp.set_paint_frame(frame);
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

/// Node 8 = a hard-edged rectangle at x 16..48, y 8..24 (coverage 1 inside,
/// 0 outside). Node 7 = the Paint node under test, fed by it, viewed.
fn graph_with(strokes: Vec<PaintStroke>) -> Graph {
    let pts = [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]]
        .iter()
        .map(|p| SplinePoint::new(*p, 0.5))
        .collect();
    let mut g = Graph::default();
    g.nodes.insert(
        8,
        NodeKind::Spline(SplineParams {
            points: pts,
            mode: SplineMode::Linear,
            ..Default::default()
        }),
    );
    g.nodes
        .insert(7, NodeKind::Paint(PaintParams { strokes, ..Default::default() }));
    g.edges.insert((7, 0), 8);
    g.view = Some(7);
    g
}

fn comp_for(ctx: &engine::GpuContext) -> Compositor {
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 2);
    comp.ensure_output(W, H);
    comp.ensure_reader(0, ImageInfo { width: W, height: H });
    comp
}

/// A dot: one point, no falloff, so the covered pixels are unambiguous.
fn dot(at: [f32; 2], radius: f32, color: [f32; 4]) -> PaintStroke {
    PaintStroke::new(vec![at], radius, 0.0, color)
}

#[test]
fn a_colour_stroke_lands_where_it_was_painted() {
    let Some(ctx) = ctx() else { return };
    let mut comp = comp_for(&ctx);
    // Dead centre of the rectangle, in green, at half alpha: it composites
    // OVER coverage 1, so the result is halfway between the two.
    let g = graph_with(vec![dot([0.5, 0.5], 6.0, [0.0, 1.0, 0.0, 0.5])]);
    let out = render(&ctx, &mut comp, &g, 0);
    let at = |x: u32, y: u32| out[(y * W + x) as usize];
    let c = at(32, 16);
    // Premultiplied green at 0.5 over a white plate: green saturates
    // (0.5 + 0.5), red and blue keep only what shows through (0 + 0.5).
    assert!((c[1] - 1.0).abs() < 0.05, "green over white: {c:?}");
    assert!((c[0] - 0.5).abs() < 0.05, "and the plate shows through: {c:?}");
    // Outside the brush the plate is untouched, inside and out of the rect.
    assert!((at(20, 16)[0] - 1.0).abs() < 0.02, "plate {:?}", at(20, 16));
    assert!(at(2, 2)[0] < 0.02, "nothing outside the rectangle");
}

#[test]
fn a_clone_stroke_paints_the_input_from_the_offset() {
    let Some(ctx) = ctx() else { return };
    let mut comp = comp_for(&ctx);
    // A dot at pixel (8, 4) — well outside the rectangle, where the plate is
    // black — cloning from +24px x, +12px y, i.e. from (32, 16): the middle
    // of the rectangle, where the plate is 1.
    let mut c = dot([8.0 / W as f32, 4.0 / H as f32], 3.0, [0.0, 0.0, 0.0, 1.0]);
    c.clone_off = Some([24.0 / W as f32, 12.0 / H as f32]);
    let g = graph_with(vec![c]);
    let out = render(&ctx, &mut comp, &g, 0);
    let at = |x: u32, y: u32| out[(y * W + x) as usize];
    assert!(
        (at(8, 4)[0] - 1.0).abs() < 0.05,
        "the clone should have brought the rectangle here: {:?}",
        at(8, 4)
    );
    assert!(
        (at(8, 4)[3] - 1.0).abs() < 0.05,
        "alpha comes with it: {:?}",
        at(8, 4)
    );
    // Just outside the brush, the plate is still black — a clone stroke is
    // a stroke, not a global offset.
    assert!(at(16, 4)[0] < 0.05, "outside the brush: {:?}", at(16, 4));
    assert!(at(2, 2)[0] < 0.05);
    // And the source region is untouched: cloning reads, it does not move.
    assert!((at(32, 16)[0] - 1.0).abs() < 0.02);

    // The same stroke WITHOUT the offset is a black dot, not a copy — i.e.
    // the pixels really came from the offset and not from the brush colour.
    let mut plain = dot([8.0 / W as f32, 4.0 / H as f32], 3.0, [0.0, 0.0, 0.0, 1.0]);
    plain.clone_off = None;
    let out = render(&ctx, &mut comp, &graph_with(vec![plain]), 0);
    assert!(out[(4 * W + 8) as usize][0] < 0.05, "colour stroke, not a clone");
}

#[test]
fn a_strokes_lifetime_decides_which_frames_it_renders_on() {
    let Some(ctx) = ctx() else { return };
    let mut comp = comp_for(&ctx);
    let mut always = dot([0.25, 0.5], 4.0, [1.0, 0.0, 0.0, 1.0]);
    always.life = StrokeLife::All;
    let mut single = dot([0.5, 0.5], 4.0, [1.0, 0.0, 0.0, 1.0]);
    single.life = StrokeLife::Single(5);
    let g = graph_with(vec![always, single]);

    // x=16 is the all-frames dot, x=32 the one that only exists on frame 5.
    let red = |out: &[[f32; 4]], x: u32| {
        let p = out[(16 * W + x) as usize];
        p[0] > 0.9 && p[1] < 0.1
    };
    let f0 = render(&ctx, &mut comp, &g, 0);
    assert!(red(&f0, 16), "the all-frames stroke is always there");
    assert!(!red(&f0, 32), "the single-frame stroke is not on frame 0");

    let f5 = render(&ctx, &mut comp, &g, 5);
    assert!(red(&f5, 16), "the all-frames stroke did not go anywhere");
    assert!(red(&f5, 32), "and the single-frame stroke arrived on frame 5");

    let f6 = render(&ctx, &mut comp, &g, 6);
    assert!(red(&f6, 16));
    assert!(!red(&f6, 32), "and left again");
}

/// A hard dot in a given mode, strength set, full opacity unless told.
fn brush(mode: BrushMode, at: [f32; 2], radius: f32, strength: f32) -> PaintStroke {
    let mut s = dot(at, radius, [0.0, 0.0, 0.0, 1.0]);
    s.mode = mode;
    s.strength = strength;
    s
}

#[test]
fn dodge_and_burn_are_exposure_and_build_up() {
    let Some(ctx) = ctx() else { return };
    let mut comp = comp_for(&ctx);
    let at = |o: &Vec<[f32; 4]>, x: u32, y: u32| o[(y * W + x) as usize];
    // One stop up at the rectangle's centre (plate = 1): exactly 2.
    let out = render(&ctx, &mut comp, &graph_with(vec![brush(BrushMode::Dodge, [0.5, 0.5], 4.0, 1.0)]), 0);
    assert!((at(&out, 32, 16)[0] - 2.0).abs() < 0.05, "dodge one stop: {:?}", at(&out, 32, 16));
    assert!((at(&out, 32, 16)[3] - 1.0).abs() < 0.02, "alpha is not exposed");
    assert!((at(&out, 20, 16)[0] - 1.0).abs() < 0.02, "outside the brush the plate is untouched");
    // One stop down: exactly a half.
    let out = render(&ctx, &mut comp, &graph_with(vec![brush(BrushMode::Burn, [0.5, 0.5], 4.0, 1.0)]), 0);
    assert!((at(&out, 32, 16)[0] - 0.5).abs() < 0.03, "burn one stop: {:?}", at(&out, 32, 16));
    // Two dodge strokes on the same spot ADD: the effect layer is additive,
    // so a dodge brush builds up under repeated strokes the way one should.
    let out = render(
        &ctx,
        &mut comp,
        &graph_with(vec![
            brush(BrushMode::Dodge, [0.5, 0.5], 4.0, 1.0),
            brush(BrushMode::Dodge, [0.5, 0.5], 4.0, 1.0),
        ]),
        0,
    );
    assert!((at(&out, 32, 16)[0] - 4.0).abs() < 0.1, "two stops: {:?}", at(&out, 32, 16));
    // Half opacity halves the exposure, not the coverage — it is premultiplied.
    let mut half = brush(BrushMode::Dodge, [0.5, 0.5], 4.0, 1.0);
    half.color[3] = 0.5;
    let out = render(&ctx, &mut comp, &graph_with(vec![half]), 0);
    let want = 2f32.powf(0.5);
    assert!((at(&out, 32, 16)[0] - want).abs() < 0.05, "half a stop: {:?}", at(&out, 32, 16));
}

#[test]
fn a_blur_stroke_softens_the_edge_under_it() {
    let Some(ctx) = ctx() else { return };
    let mut comp = comp_for(&ctx);
    let at = |o: &Vec<[f32; 4]>, x: u32, y: u32| o[(y * W + x) as usize];
    // A hard dot straddling the rectangle's left edge (x = 16), blurring
    // over 4px: a ring of eight taps at radius 4 plus the centre. Just
    // inside the edge, 5 of the 8 taps land inside; just outside, 3 do.
    let g = graph_with(vec![brush(BrushMode::Blur, [16.0 / W as f32, 0.5], 6.0, 4.0)]);
    let out = render(&ctx, &mut comp, &g, 0);
    let inside = at(&out, 16, 16)[0];
    let outside = at(&out, 15, 16)[0];
    assert!((0.55..0.78).contains(&inside), "just inside the edge: {inside:.3} (want ~0.667)");
    assert!((0.2..0.45).contains(&outside), "just outside the edge: {outside:.3} (want ~0.333)");
    // Beyond the brush the edge is as hard as ever.
    assert!(at(&out, 16, 4)[0] < 0.02 && at(&out, 20, 4)[0] < 0.02, "above the rect, off-brush");
    assert!((at(&out, 30, 16)[0] - 1.0).abs() < 0.02, "deep inside, off-brush");
}

#[test]
fn a_smear_drags_pixels_along_the_stroke() {
    let Some(ctx) = ctx() else { return };
    let mut comp = comp_for(&ctx);
    let at = |o: &Vec<[f32; 4]>, x: u32, y: u32| o[(y * W + x) as usize];
    // A stroke from x=16 to x=32 along the rectangle's middle, dragging
    // 8px: every pixel under it shows what was 8px BEHIND it. At x=20 that
    // is x=12 — black, outside the rectangle — pulled in over the white.
    let mut sm = brush(BrushMode::Smear, [16.0 / W as f32, 0.5], 4.0, 8.0);
    sm.points.push([32.0 / W as f32, 0.5]);
    let out = render(&ctx, &mut comp, &graph_with(vec![sm]), 0);
    assert!(at(&out, 20, 16)[0] < 0.05, "black dragged into the rect: {:?}", at(&out, 20, 16));
    assert!((at(&out, 30, 16)[0] - 1.0).abs() < 0.05, "8px behind x=30 is still white");
    assert!((at(&out, 40, 16)[0] - 1.0).abs() < 0.02, "past the stroke, untouched");
}

#[test]
fn an_elliptical_brush_covers_along_its_major_axis() {
    let Some(ctx) = ctx() else { return };
    let mut comp = comp_for(&ctx);
    let at = |o: &Vec<[f32; 4]>, x: u32, y: u32| o[(y * W + x) as usize];
    let red = |p: [f32; 4]| p[0] > 0.9 && p[1] < 0.1;
    // Major radius 10, minor 3, lying along x: 8px right is covered, 6px
    // down is not.
    let mut flat = dot([0.5, 0.5], 10.0, [1.0, 0.0, 0.0, 1.0]);
    flat.aspect = 0.3;
    flat.angle = 0.0;
    let out = render(&ctx, &mut comp, &graph_with(vec![flat.clone()]), 0);
    assert!(red(at(&out, 40, 16)), "along the major axis: {:?}", at(&out, 40, 16));
    assert!(!red(at(&out, 32, 22)), "across the minor axis: {:?}", at(&out, 32, 22));
    // Turned 90 degrees, the two swap.
    flat.angle = 90.0;
    let out = render(&ctx, &mut comp, &graph_with(vec![flat]), 0);
    assert!(!red(at(&out, 40, 16)), "turned: x is now the minor axis {:?}", at(&out, 40, 16));
    assert!(red(at(&out, 32, 22)), "turned: y is now the major axis {:?}", at(&out, 32, 22));
}

#[test]
fn matte_output_is_coverage_unioned_with_the_inputs_alpha() {
    let Some(ctx) = ctx() else { return };
    let mut comp = comp_for(&ctx);
    let at = |o: &Vec<[f32; 4]>, x: u32, y: u32| o[(y * W + x) as usize];
    // A colour dot outside the rectangle and a dodge dot outside it too:
    // in matte mode both are just coverage, and the rectangle's own alpha
    // comes through unioned.
    let mut g = graph_with(vec![
        dot([8.0 / W as f32, 4.0 / H as f32], 3.0, [0.0, 1.0, 0.0, 1.0]),
        brush(BrushMode::Dodge, [56.0 / W as f32, 4.0 / H as f32], 3.0, 1.0),
    ]);
    if let Some(NodeKind::Paint(p)) = g.nodes.get_mut(&7) {
        p.output = PaintOutput::Matte;
    }
    assert_eq!(g.nodes[&7].output_kind(), engine::graph::PortKind::Mask);
    let out = render(&ctx, &mut comp, &g, 0);
    let m = at(&out, 8, 4);
    assert!((m[0] - 1.0).abs() < 0.02 && (m[3] - 1.0).abs() < 0.02, "colour stroke = coverage: {m:?}");
    assert!(m[1] > 0.98, "a matte is grey, not green: {m:?}");
    let d = at(&out, 56, 4);
    assert!((d[0] - 1.0).abs() < 0.02, "a dodge stroke counts as coverage too: {d:?}");
    assert!((at(&out, 32, 16)[0] - 1.0).abs() < 0.02, "the input's alpha comes through");
    assert!(at(&out, 2, 2)[0] < 0.02 && at(&out, 2, 2)[3] < 0.02, "nothing elsewhere");
}

#[test]
fn a_clone_picks_up_the_strokes_already_painted_at_its_source() {
    let Some(ctx) = ctx() else { return };
    let mut comp = comp_for(&ctx);
    let at = |o: &Vec<[f32; 4]>, x: u32, y: u32| o[(y * W + x) as usize];
    // A green dot at the rectangle's centre, and a clone dot at (8, 4) —
    // outside the rectangle, where the plate is black — whose source is
    // that centre. What arrives is the GREEN, not the plate's white: a
    // clone samples the painted picture, not the bare input.
    let green = dot([0.5, 0.5], 4.0, [0.0, 1.0, 0.0, 1.0]);
    let mut c = brush(BrushMode::Clone, [8.0 / W as f32, 4.0 / H as f32], 3.0, 0.0);
    c.clone_off = Some([24.0 / W as f32, 12.0 / H as f32]);
    let out = render(&ctx, &mut comp, &graph_with(vec![green.clone(), c.clone()]), 0);
    let p = at(&out, 8, 4);
    assert!(p[1] > 0.95 && p[0] < 0.05, "the clone brought the green stroke: {p:?}");
    // The source's dodge comes along too: with a stop of dodge under the
    // centre and NO green, the clone lands the plate at 2.0.
    let dodge = brush(BrushMode::Dodge, [0.5, 0.5], 4.0, 1.0);
    let out = render(&ctx, &mut comp, &graph_with(vec![dodge, c]), 0);
    let p = at(&out, 8, 4);
    assert!((p[0] - 2.0).abs() < 0.06, "the clone brought the dodged plate: {p:?}");
}

/// A hard dot with a blend mode.
fn blended(at: [f32; 2], color: [f32; 4], blend: PaintBlend) -> PaintStroke {
    PaintStroke { blend, ..dot(at, 6.0, color) }
}

fn close(a: [f32; 4], b: [f32; 4]) -> bool {
    a.iter().zip(b).all(|(x, y)| (x - y).abs() < 0.03)
}

/// Every blend against the plate, inside the rectangle (white, alpha 1) and
/// outside it (black, alpha 0). The arithmetic is the claim: each mode is
/// the affine map its doc says, on the PLATE, not on the paint alone.
#[test]
fn blend_modes_act_on_the_plate() {
    let Some(ctx) = ctx() else { return };
    let mut comp = comp_for(&ctx);
    let k = [0.5, 0.25, 1.0, 1.0];
    let half = [0.5, 0.25, 1.0, 0.5];
    // (blend, colour, inside, outside)
    let cases = [
        (PaintBlend::Multiply, k, [0.5, 0.25, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]),
        (PaintBlend::Add, k, [1.5, 1.25, 2.0, 1.0], [0.5, 0.25, 1.0, 0.0]),
        (PaintBlend::Subtract, k, [0.5, 0.75, 0.0, 1.0], [-0.5, -0.25, -1.0, 0.0]),
        (PaintBlend::Screen, k, [1.0, 1.0, 1.0, 1.0], [0.5, 0.25, 1.0, 0.0]),
        // Half opacity: multiply mixes halfway to the product.
        (PaintBlend::Multiply, half, [0.75, 0.625, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0]),
        // Normal still covers, alpha and all.
        (PaintBlend::Normal, k, [0.5, 0.25, 1.0, 1.0], [0.5, 0.25, 1.0, 1.0]),
    ];
    for (b, c, inside, outside) in cases {
        // One dot in the rectangle (32,16), one outside it (6,16).
        let g = graph_with(vec![
            blended([0.5, 0.5], c, b),
            blended([6.0 / W as f32, 0.5], c, b),
        ]);
        let out = render(&ctx, &mut comp, &g, 0);
        let at = |x: u32, y: u32| out[(y * W + x) as usize];
        assert!(close(at(32, 16), inside), "{b:?} inside: {:?} want {inside:?}", at(32, 16));
        assert!(close(at(6, 16), outside), "{b:?} outside: {:?} want {outside:?}", at(6, 16));
        // Off the brush, untouched.
        assert!(close(at(20, 10), [1.0, 1.0, 1.0, 1.0]), "{b:?} plate {:?}", at(20, 10));
    }
}

/// Stroke ORDER holds across modes: a multiply laid after a normal stroke
/// darkens the paint, one laid before is covered by it. And erase takes
/// earlier paint away, back to the plate — never the plate itself.
#[test]
fn blend_order_holds_and_erase_restores_the_plate() {
    let Some(ctx) = ctx() else { return };
    let mut comp = comp_for(&ctx);
    let grey = [0.5, 0.5, 0.5, 1.0];
    let at = |out: &[[f32; 4]]| out[(16 * W + 32) as usize];
    let normal = blended([0.5, 0.5], grey, PaintBlend::Normal);
    let mul = blended([0.5, 0.5], grey, PaintBlend::Multiply);

    let after = render(&ctx, &mut comp, &graph_with(vec![normal.clone(), mul.clone()]), 0);
    assert!(close(at(&after), [0.25, 0.25, 0.25, 1.0]), "multiply after: {:?}", at(&after));
    let before = render(&ctx, &mut comp, &graph_with(vec![mul.clone(), normal.clone()]), 0);
    assert!(close(at(&before), [0.5, 0.5, 0.5, 1.0]), "multiply before: {:?}", at(&before));

    let erase = blended([0.5, 0.5], [0.0, 0.0, 0.0, 1.0], PaintBlend::Erase);
    // In the one-texture path (normal + erase) and the two-texture one.
    for strokes in [vec![normal.clone(), erase.clone()], vec![normal.clone(), mul.clone(), erase.clone()]] {
        let out = render(&ctx, &mut comp, &graph_with(strokes), 0);
        assert!(close(at(&out), [1.0, 1.0, 1.0, 1.0]), "erased to the plate: {:?}", at(&out));
    }
    // Erase over nothing painted leaves the plate alone, and outside the
    // rectangle it does not cut a hole in black either.
    let out = render(&ctx, &mut comp, &graph_with(vec![erase]), 0);
    assert!(close(at(&out), [1.0, 1.0, 1.0, 1.0]), "{:?}", at(&out));
}
