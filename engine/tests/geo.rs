//! Headless integration for 2dini, the geometry paradigm.
//!
//! The unit tests in `geo.rs` prove the evaluator produces the right points
//! and `georaster.rs` proves the WGSL validates. These prove the two agree
//! once a GPU draws the result — that a fill is a fill, a stroke is as wide
//! as it says, draw order is Merge order, premultiplication happened once,
//! and above all that a Noise wired into the node's field port really does
//! move geometry across the picture. Then the two invariants the whole
//! design rests on: a geometry edit is a layer re-record and never a
//! recompile, and an unchanged graph records nothing. Last, volume on a 2K
//! frame with the times printed. All four channels are read back, since
//! colour and coverage are separate claims here. Self-skips without a GPU
//! (CI).

use engine::compositor::Compositor;
use engine::field::{CoordKind, FieldFbmMode, FieldGraph, FieldNode};
use engine::geo::{FillRule, GeoGraph, GeoNode, Style};
use engine::graph::{GeoParams, Graph, NodeId, NodeKind, NoiseParams, SplineMode, SplinePoint};
use engine::timing::GpuTimer;
use half::f16;

const W: u32 = 256;
const H: u32 = 128;
/// The comp-level ids: the 2dini node and the Noise that may feed it.
const GEO: NodeId = 1;
const NOISE: NodeId = 2;

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP geo test: no adapter ({e})");
            None
        }
    }
}

fn comp(ctx: &engine::GpuContext) -> Compositor {
    let mut c = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    c.ensure_output(W, H);
    c
}

/// Every channel of every pixel, row-major, at the harness size.
fn render(ctx: &engine::GpuContext, comp: &mut Compositor, graph: &Graph) -> Vec<[f32; 4]> {
    render_at(ctx, comp, graph, W, H)
}

/// The same at any size; the second encode is the one that records the
/// geometry layer, so its wall time is the raster (plus the fused pass).
fn render_at(ctx: &engine::GpuContext, comp: &mut Compositor, graph: &Graph, w: u32, h: u32) -> Vec<[f32; 4]> {
    assert!(comp.set_graph(graph), "fused shader failed to compile");
    let timer = GpuTimer::new(ctx);
    let mut warm = ctx.device.create_command_encoder(&Default::default());
    comp.encode_comp(&mut warm, &timer);
    ctx.queue.submit([warm.finish()]);
    comp.write_params(graph);
    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    comp.encode_comp(&mut encoder, &timer);
    let padded = (w * 8).div_ceil(256) * 256;
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (padded * h) as u64,
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
                rows_per_image: Some(h),
            },
        },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    let t0 = std::time::Instant::now();
    ctx.queue.submit([encoder.finish()]);
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    eprintln!("render {w}x{h}: encode+submit {:.2} ms", t0.elapsed().as_secs_f32() * 1e3);
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    let data = slice.get_mapped_range().expect("a mapped buffer");
    let mut out = Vec::with_capacity((w * h) as usize);
    for y in 0..h as usize {
        let off = y * padded as usize;
        let halves: &[f16] = bytemuck::cast_slice(&data[off..off + (w * 8) as usize]);
        for x in 0..w as usize {
            let i = x * 4;
            out.push([
                halves[i].to_f32(),
                halves[i + 1].to_f32(),
                halves[i + 2].to_f32(),
                halves[i + 3].to_f32(),
            ]);
        }
    }
    drop(data);
    staging.unmap();
    out
}

fn at(img: &[[f32; 4]], x: usize, y: usize) -> [f32; 4] {
    img[y * W as usize + x]
}

fn close(a: [f32; 4], b: [f32; 4], tol: f32) -> bool {
    a.iter().zip(b.iter()).all(|(p, q)| (p - q).abs() < tol)
}

/// Sum of alpha over a column band — coverage, in pixels.
fn coverage(img: &[[f32; 4]], x0: usize, x1: usize) -> f32 {
    let mut acc = 0.0;
    for y in 0..H as usize {
        for x in x0..x1 {
            acc += at(img, x, y)[3];
        }
    }
    acc
}

/// A comp graph holding one 2dini node with `sub` inside it.
fn comp_of(sub: GeoGraph) -> Graph {
    let mut g = Graph::default();
    g.nodes.insert(GEO, NodeKind::Geo(GeoParams { size: (W, H), inherit: false, graph: sub, seed: 0.0, ..Default::default() }));
    g.view = Some(GEO);
    g
}

/// A Noise on the 2dini node's field port, so a FieldIn inside reads it.
fn wire_noise(g: &mut Graph, field: FieldGraph, speed: f32, time: f32) {
    g.nodes.insert(
        NOISE,
        NodeKind::Noise(NoiseParams {
            size: (W, H),
            graph: field,
            scale: 1.0,
            seed: 0.0,
            speed,
            time,
            ..Default::default()
        }),
    );
    g.edges.insert((GEO, 0), NOISE);
}

/// A field that is just x: 0 at the left edge, 1 at the right.
fn coord_x() -> FieldGraph {
    let mut f = FieldGraph::default();
    f.nodes.insert(1, FieldNode::Coord { kind: CoordKind::X, scale: 1.0 });
    f.out = Some(1);
    f
}

/// A field that consumes its phase, so `field_moves` says yes.
fn moving_fbm() -> FieldGraph {
    let mut f = FieldGraph::default();
    f.nodes.insert(
        1,
        FieldNode::Fbm {
            mode: FieldFbmMode::Fbm,
            scale: [0.25, 0.25],
            offset: [0.0, 0.0],
            lacunarity: 2.0,
            gain: 0.5,
            seed: 0.0,
            octaves: 4,
            speed: 1.0,
            uniform: 1.0,
        },
    );
    f.out = Some(1);
    f
}

fn fill(c: [f32; 4]) -> Style {
    Style { fill: c, ..Default::default() }
}

/// An axis-aligned rectangle in geo space (x in 0..1, y in width units),
/// as a straight-segment closed Spline — no rotation convention to get
/// wrong, and every edge lands where the numbers say.
fn rect(x0: f32, y0: f32, x1: f32, y1: f32, style: Style) -> GeoNode {
    let pt = |x: f32, y: f32| SplinePoint { p: [x, y], feather: 0.0, bias: 0.0, h_in: [0.0; 2], h_out: [0.0; 2], tension: 0.0 };
    GeoNode::Spline {
        points: vec![pt(x0, y0), pt(x1, y0), pt(x1, y1), pt(x0, y1)],
        mode: SplineMode::Linear,
        closed: true,
        style,
    }
}

/// The reference square: x 64.5..192 px, y 25.6..102.4 px. Its left edge
/// sits on a pixel CENTRE so the AA ramp has a genuine half-covered pixel.
const SQ_X0: f32 = 64.5 / W as f32;
const SQ_X1: f32 = 0.75;
const SQ_Y0: f32 = 0.1;
const SQ_Y1: f32 = 0.4;

fn square(style: Style) -> GeoNode {
    rect(SQ_X0, SQ_Y0, SQ_X1, SQ_Y1, style)
}

/// One generator straight into an Output.
fn single(node: GeoNode) -> GeoGraph {
    let mut g = GeoGraph::default();
    g.nodes.insert(1, node);
    g.nodes.insert(2, GeoNode::Output);
    g.edges.insert((2, 0), 1);
    g.next_id = 3;
    g
}

// ---------------------------------------------------------------------------

/// A closed outline fills: 1 inside, 0 outside, and the edge is a single
/// monotone anti-aliasing ramp that passes through half coverage.
#[test]
fn a_closed_spline_fills_its_interior() {
    let Some(ctx) = ctx() else { return };
    let mut comp = comp(&ctx);
    let img = render(&ctx, &mut comp, &comp_of(single(square(Style::default()))));

    let white = [1.0; 4];
    for (x, y) in [(128, 64), (70, 30), (186, 98)] {
        let c = at(&img, x, y);
        assert!(close(c, white, 1e-2), "interior ({x},{y}) = {c:?}, want white");
    }
    for (x, y) in [(20, 64), (240, 64), (128, 5), (128, 120), (60, 64)] {
        let c = at(&img, x, y);
        assert!(close(c, [0.0; 4], 1e-2), "exterior ({x},{y}) = {c:?}, want clear");
    }
    // Across the left edge (x = 64.5 px) on the middle row: never dips,
    // and somewhere in the ramp is a pixel about half covered.
    let row: Vec<f32> = (56..80).map(|x| at(&img, x, 64)[3]).collect();
    assert!(
        row.windows(2).all(|w| w[1] >= w[0] - 1e-3),
        "coverage across the edge is not monotone: {row:?}"
    );
    // The edge runs through the centre of pixel 64, so that one is half.
    let half = at(&img, 64, 64)[3];
    assert!((0.45..=0.55).contains(&half), "pixel under the edge = {half}, want 0.5: {row:?}");
    assert!(row[0] < 1e-2 && row[row.len() - 1] > 0.99, "ramp ends wrong: {row:?}");
}

/// A pentagram ({5/2}) has winding 2 at its centre: filled under non-zero,
/// a hole under even-odd. The one place the two rules disagree.
#[test]
fn fill_rule_decides_the_pentagram_centre() {
    let Some(ctx) = ctx() else { return };
    let mut comp = comp(&ctx);
    let star = |rule: FillRule| {
        single(GeoNode::Polygon {
            center: [0.5, 0.25],
            radius: 0.2,
            sides: 5,
            step: 2,
            rotation: 0.0,
            style: Style { rule, ..Default::default() },
        })
    };
    let nz = render(&ctx, &mut comp, &comp_of(star(FillRule::NonZero)));
    let c = at(&nz, 128, 64);
    assert!(close(c, [1.0; 4], 1e-2), "non-zero centre = {c:?}, want filled");
    // And it IS a star: something got drawn away from the centre too.
    assert!(coverage(&nz, 0, W as usize) > 1000.0, "the pentagram is too small to be one");

    let eo = render(&ctx, &mut comp, &comp_of(star(FillRule::EvenOdd)));
    let c = at(&eo, 128, 64);
    assert!(close(c, [0.0; 4], 1e-2), "even-odd centre = {c:?}, want empty");
    // The points are winding 1 under both rules, so they stay filled.
    let tip = at(&eo, 128, 64 - 48);
    assert!(tip[3] > 0.9, "even-odd lost the star's tip: {tip:?}");
}

/// A stroke is centred on the outline and as wide as it says, and leaves
/// the interior alone when there is no fill.
#[test]
fn a_stroke_is_as_wide_as_it_says() {
    let Some(ctx) = ctx() else { return };
    let mut comp = comp(&ctx);
    let style = Style { fill: [0.0; 4], stroke: [1.0; 4], width: 6.0, ..Default::default() };
    let img = render(&ctx, &mut comp, &comp_of(single(square(style))));

    // Interior empty: an unfilled outline is hollow.
    for (x, y) in [(128, 64), (80, 40), (180, 90)] {
        let c = at(&img, x, y);
        assert!(close(c, [0.0; 4], 1e-2), "hollow square has paint inside at ({x},{y}): {c:?}");
    }
    // Across the left edge, mid-row: alpha sums to the width (the AA ramps
    // on either side add to one pixel between them), and the solid core is
    // about width pixels wide.
    let band: Vec<f32> = (50..80).map(|x| at(&img, x, 64)[3]).collect();
    let total: f32 = band.iter().sum();
    assert!((total - 6.0).abs() < 0.5, "stroke integrates to {total:.2} px, want 6");
    let solid = band.iter().filter(|&&a| a > 0.5).count();
    assert!((5..=7).contains(&solid), "stroke core is {solid} px wide, want 6 ± 1");
    // Centred on the edge (x = 64.5): symmetric either side of it.
    let inner: f32 = (65..80).map(|x| at(&img, x, 64)[3]).sum();
    let outer: f32 = (50..64).map(|x| at(&img, x, 64)[3]).sum();
    assert!((inner - outer).abs() < 1.0, "stroke is off-centre: {outer:.2} out vs {inner:.2} in");
    // No double counting where two segments meet: nothing exceeds one.
    let peak = img.iter().map(|c| c[3]).fold(0.0f32, f32::max);
    assert!(peak <= 1.0 + 1e-2, "stroke alpha peaks at {peak}");
    // Corners are round. Outside the top-left corner (64.5, 25.6) the
    // stroke is a quarter disc of r = 3; the window x 54..64, y 15..25
    // catches that disc minus its 0.5 and 0.6 px strips along the edges
    // — about 4.1 px of coverage. A mitre would make it a 2.5 × 2.4
    // square (6.0), a bevel roughly half of that.
    let mut corner = 0.0;
    for y in 15..25 {
        for x in 54..64 {
            corner += at(&img, x, y)[3];
        }
    }
    assert!((3.6..=4.7).contains(&corner), "corner covers {corner:.2} px, want a round join (~4.1)");
}

/// Colours are premultiplied ONCE, in flatten: a half-transparent red
/// reads back as (0.5, 0, 0, 0.5), never (1, 0, 0, 0.5) or (0.25, ...).
#[test]
fn fills_are_premultiplied() {
    let Some(ctx) = ctx() else { return };
    let mut comp = comp(&ctx);
    let img = render(&ctx, &mut comp, &comp_of(single(square(fill([1.0, 0.0, 0.0, 0.5])))));
    let c = at(&img, 128, 64);
    assert!(close(c, [0.5, 0.0, 0.0, 0.5], 1e-2), "half red reads {c:?}");
    // And nothing anywhere exceeds its alpha.
    let over = img.iter().filter(|c| c[0] > c[3] + 1e-2 || c[1] > c[3] + 1e-2 || c[2] > c[3] + 1e-2).count();
    assert_eq!(over, 0, "{over} pixels have colour brighter than their alpha");
}

/// Merge is draw order: input 1 lands on top of input 0 where they overlap.
#[test]
fn merge_draws_its_second_input_on_top() {
    let Some(ctx) = ctx() else { return };
    let mut comp = comp(&ctx);
    let red = [1.0, 0.0, 0.0, 1.0];
    let blue = [0.0, 0.0, 1.0, 1.0];
    let build = |first: [f32; 4], second: [f32; 4]| {
        let mut g = GeoGraph::default();
        g.nodes.insert(1, rect(0.2, 0.1, 0.6, 0.4, fill(first)));
        g.nodes.insert(2, rect(0.4, 0.1, 0.8, 0.4, fill(second)));
        g.nodes.insert(3, GeoNode::Merge { count: 2 });
        g.nodes.insert(4, GeoNode::Output);
        g.edges.insert((3, 0), 1);
        g.edges.insert((3, 1), 2);
        g.edges.insert((4, 0), 3);
        g.next_id = 5;
        comp_of(g)
    };
    let img = render(&ctx, &mut comp, &build(red, blue));
    let only_a = at(&img, 70, 64);
    let overlap = at(&img, 128, 64);
    let only_b = at(&img, 190, 64);
    assert!(close(only_a, red, 1e-2), "first shape alone = {only_a:?}");
    assert!(close(only_b, blue, 1e-2), "second shape alone = {only_b:?}");
    assert!(close(overlap, blue, 1e-2), "overlap = {overlap:?}, want the SECOND input");

    // Swap the inputs and the overlap swaps with them — so the result above
    // came from order, not from which colour happens to win a blend.
    let img = render(&ctx, &mut comp, &build(blue, red));
    let overlap = at(&img, 128, 64);
    assert!(close(overlap, red, 1e-2), "swapped overlap = {overlap:?}, want red");
}

/// A z on a noodle restacks a Merge (lower z under), and a blend on it
/// changes how its shapes land on the ones below.
#[test]
fn a_noodles_z_and_blend_reach_the_picture() {
    use engine::geo::{BlendMode, Layering};
    let Some(ctx) = ctx() else { return };
    let mut comp = comp(&ctx);
    let red = [1.0, 0.0, 0.0, 1.0];
    let half = [0.5, 0.5, 0.5, 1.0];
    let build = |second: [f32; 4], wires: Vec<((u64, usize), Layering)>| {
        let mut g = GeoGraph::default();
        g.nodes.insert(1, rect(0.2, 0.1, 0.6, 0.4, fill(red)));
        g.nodes.insert(2, rect(0.4, 0.1, 0.8, 0.4, fill(second)));
        g.nodes.insert(3, GeoNode::Merge { count: 2 });
        g.nodes.insert(4, GeoNode::Output);
        g.edges.insert((3, 0), 1);
        g.edges.insert((3, 1), 2);
        g.edges.insert((4, 0), 3);
        g.wires = wires;
        g.next_id = 5;
        comp_of(g)
    };
    // Input 1 is on top by order; z -1 on its noodle puts it under.
    let img = render(&ctx, &mut comp, &build(half, vec![((3, 1), Layering { z: -1.0, ..Default::default() })]));
    assert!(close(at(&img, 128, 64), red, 1e-2), "z did not restack: {:?}", at(&img, 128, 64));
    let lay = |blend| vec![((3, 1), Layering { z: 0.0, blend })];
    // Multiply: red times grey where they overlap; on nothing, nothing.
    let img = render(&ctx, &mut comp, &build(half, lay(BlendMode::Multiply)));
    assert!(close(at(&img, 128, 64), [0.5, 0.0, 0.0, 1.0], 2e-2), "multiply = {:?}", at(&img, 128, 64));
    assert!(at(&img, 190, 64)[3] < 1e-3, "multiply onto nothing drew {:?}", at(&img, 190, 64));
    // Add and screen brighten, subtract darkens.
    let img = render(&ctx, &mut comp, &build(half, lay(BlendMode::Add)));
    assert!(close(at(&img, 128, 64), [1.5, 0.5, 0.5, 1.0], 2e-2), "add = {:?}", at(&img, 128, 64));
    let img = render(&ctx, &mut comp, &build(half, lay(BlendMode::Screen)));
    assert!(close(at(&img, 128, 64), [1.0, 0.5, 0.5, 1.0], 2e-2), "screen = {:?}", at(&img, 128, 64));
    let img = render(&ctx, &mut comp, &build(half, lay(BlendMode::Subtract)));
    assert!(close(at(&img, 128, 64), [0.5, -0.5, -0.5, 1.0], 2e-2), "subtract = {:?}", at(&img, 128, 64));
    let img = render(&ctx, &mut comp, &build(half, lay(BlendMode::Lighten)));
    assert!(close(at(&img, 128, 64), [1.0, 0.5, 0.5, 1.0], 2e-2), "lighten = {:?}", at(&img, 128, 64));
    let img = render(&ctx, &mut comp, &build(half, lay(BlendMode::Darken)));
    assert!(close(at(&img, 128, 64), [0.5, 0.0, 0.0, 1.0], 2e-2), "darken = {:?}", at(&img, 128, 64));
}

/// Gradient fills: linear runs fill -> colour along its line, radial from
/// its centre out, from-edges from the outline inward by `depth` px.
#[test]
fn gradient_fills_run_from_the_fill_to_their_colour() {
    use engine::geo::{GradKind, Gradient};
    let Some(ctx) = ctx() else { return };
    let mut comp = comp(&ctx);
    let black = [0.0, 0.0, 0.0, 1.0];
    let white = [1.0, 1.0, 1.0, 1.0];
    let shape = |grad: Gradient| {
        let style = Style { fill: black, grad, ..Default::default() };
        comp_of(single(rect(0.25, 0.1, 0.75, 0.4, style)))
    };
    let grey = |c: [f32; 4]| c[0];
    // Linear across the rectangle, left (black) to right (white).
    let g = Gradient { kind: GradKind::Linear, to: white, a: [0.25, 0.2], b: [0.75, 0.2], depth: 1.0, ..Default::default() };
    let img = render(&ctx, &mut comp, &shape(g));
    let (l, m, r) = (grey(at(&img, 66, 64)), grey(at(&img, 128, 64)), grey(at(&img, 190, 64)));
    assert!(l < 0.05 && (m - 0.5).abs() < 0.05 && r > 0.95, "linear: {l} {m} {r}");
    // Radial from the middle.
    let g = Gradient { kind: GradKind::Radial, to: white, a: [0.5, 0.25], b: [0.75, 0.25], depth: 1.0, ..Default::default() };
    let img = render(&ctx, &mut comp, &shape(g));
    let (c, e) = (grey(at(&img, 128, 64)), grey(at(&img, 190, 64)));
    assert!(c < 0.05 && e > 0.9, "radial: centre {c} rim {e}");
    // Three stops: red, green in the middle, blue — the stops, not the fill.
    let mut g = Gradient { kind: GradKind::Linear, a: [0.25, 0.2], b: [0.75, 0.2], ..Default::default() };
    g.stops = engine::geo::GradStops::from_slice(&[
        [0.0, 1.0, 0.0, 0.0, 1.0],
        [0.5, 0.0, 1.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0, 1.0],
    ]);
    let img = render(&ctx, &mut comp, &shape(g));
    let (l, m, r) = (at(&img, 66, 64), at(&img, 128, 64), at(&img, 190, 64));
    assert!(l[0] > 0.9 && m[1] > 0.9 && m[0] < 0.1 && r[2] > 0.9, "three stops: {l:?} {m:?} {r:?}");
    // From the edges: black on the outline, white 20 px in.
    let g = Gradient { kind: GradKind::Edge, to: white, depth: 20.0, ..Default::default() };
    let img = render(&ctx, &mut comp, &shape(g));
    let (edge, deep) = (grey(at(&img, 66, 64)), grey(at(&img, 128, 64)));
    assert!(edge < 0.15 && deep > 0.95, "from edges: edge {edge} deep {deep}");
}

/// A bare point is a disc of radius `pscale` px: N points scattered where
/// they cannot touch add up to N·π·r² of coverage.
#[test]
fn bare_points_draw_as_discs() {
    let Some(ctx) = ctx() else { return };
    let mut comp = comp(&ctx);
    const N: u32 = 40;
    const R: f32 = 2.0;
    // Scatter in a small rectangle, then Transform scales it up about the
    // origin: that spreads the points AND sets pscale (a Transform scales
    // pscale by its mean scale) without anything else in the graph
    // knowing about radii. The region scales into x 64..192, y 32..96 px,
    // so no disc touches the canvas edge.
    let mut g = GeoGraph::default();
    g.nodes.insert(1, rect(0.125, 0.0625, 0.375, 0.1875, Style::default()));
    g.nodes.insert(2, GeoNode::Scatter { count: N, seed: 0.0, density: 1.0 });
    g.nodes.insert(3, GeoNode::Transform { translate: [0.0, 0.0], rotate: 0.0, scale: [R, R], pivot: [0.0, 0.0], inherit: false });
    g.nodes.insert(4, GeoNode::Output);
    g.edges.insert((2, 0), 1);
    g.edges.insert((3, 0), 2);
    g.edges.insert((4, 0), 3);
    g.next_id = 5;
    let img = render(&ctx, &mut comp, &comp_of(g));

    let total = coverage(&img, 0, W as usize);
    let want = N as f32 * std::f32::consts::PI * R * R;
    assert!(
        (total - want).abs() < want * 0.05,
        "{N} discs of r={R} cover {total:.1} px, want {want:.1} ± 5%"
    );
    // Discs, not squares or points: a disc of r=2 peaks at full alpha and
    // its coverage is white (the scatter's default Cd).
    let peak = img.iter().cloned().fold([0.0f32; 4], |m, c| if c[3] > m[3] { c } else { m });
    assert!(close(peak, [1.0; 4], 2e-2), "brightest disc pixel = {peak:?}");
    // The region itself is not drawn: Scatter emits only points.
    let inside_region_only: f32 = coverage(&img, 0, 60) + coverage(&img, 196, W as usize);
    assert!(inside_region_only < 1e-2, "paint outside the scaled region: {inside_region_only}");
}

/// THE test. A Noise wired into the node's field port drives a Scatter's
/// density through a FieldIn: with the field being x, copies pile up on
/// the right; unwired, a FieldIn reads 0.5 and the halves match.
#[test]
fn a_wired_noise_moves_geometry_across_the_picture() {
    let Some(ctx) = ctx() else { return };
    let mut comp = comp(&ctx);
    // FieldIn → Remap(0..1 → -0.5..1.5, clamped) → density, so the left
    // quarter accepts nothing and the right quarter everything: 7:1 in
    // expectation, where a plain x would give 3:1 and miss the bar.
    let mut g = GeoGraph::default();
    g.nodes.insert(1, GeoNode::FieldIn { shift: 0.0 });
    g.nodes.insert(
        2,
        GeoNode::Remap { in_lo: 0.0, in_hi: 1.0, out_lo: -0.5, out_hi: 1.5, clamp: true },
    );
    g.nodes.insert(3, GeoNode::Scatter { count: 400, seed: 0.0, density: 1.0 });
    g.nodes.insert(
        4,
        GeoNode::Polygon {
            center: [0.0, 0.0],
            radius: 0.008,
            sides: 6,
            step: 1,
            rotation: 0.0,
            style: Style::default(),
        },
    );
    g.nodes.insert(
        5,
        GeoNode::Instance {
            scale: 1.0,
            rotation: 0.0,
            inherit_cd: true,
            pivot: engine::geo::Pivot::BboxCentre,
        },
    );
    g.nodes.insert(6, GeoNode::Output);
    g.edges.insert((2, 0), 1); // remap reads the field
    g.edges.insert((3, 1), 2); // density socket
    g.edges.insert((5, 0), 4); // template
    g.edges.insert((5, 1), 3); // points
    g.edges.insert((6, 0), 5);
    g.next_id = 7;

    let mut driven = comp_of(g.clone());
    wire_noise(&mut driven, coord_x(), 0.0, 0.0);
    let img = render(&ctx, &mut comp, &driven);
    let left = coverage(&img, 0, (W / 2) as usize);
    let right = coverage(&img, (W / 2) as usize, W as usize);
    assert!(right > 50.0, "nothing was instanced on the right: {right:.1}");
    assert!(
        right > left * 4.0,
        "the field did not shape the scatter: {left:.1} left vs {right:.1} right"
    );

    // Unwired: 0.5 everywhere, so the two halves fill alike — which is
    // what makes the comparison above mean something.
    let plain = comp_of(g);
    let img = render(&ctx, &mut comp, &plain);
    let left = coverage(&img, 0, (W / 2) as usize);
    let right = coverage(&img, (W / 2) as usize, W as usize);
    assert!(left > 50.0 && right > 50.0, "an unwired FieldIn should still scatter: {left:.1}/{right:.1}");
    assert!(
        (left - right).abs() < left.max(right) * 0.25,
        "an unwired FieldIn should be even across the frame: {left:.1} vs {right:.1}"
    );
}

/// The invariant the engine runs on, extended to geometry: moving a slider
/// inside a 2dini node re-records its layer and never touches a pipeline.
#[test]
fn a_geometry_edit_is_not_a_recompile() {
    let Some(ctx) = ctx() else { return };
    let mut comp = comp(&ctx);
    let mut g = GeoGraph::default();
    g.nodes.insert(1, GeoNode::Slider { value: 0.1, name: "radius".into() });
    g.nodes.insert(
        2,
        GeoNode::Polygon {
            center: [0.5, 0.25],
            radius: 0.05,
            sides: 32,
            step: 1,
            rotation: 0.0,
            style: Style::default(),
        },
    );
    g.nodes.insert(3, GeoNode::Output);
    g.edges.insert((2, 2), 1); // radius socket
    g.edges.insert((3, 0), 2);
    g.next_id = 4;
    let small = comp_of(g);
    let before = render(&ctx, &mut comp, &small);
    assert!(at(&before, 128, 64)[3] > 0.99, "the disc did not draw");

    // The edit goes through the same door the panel and animation use.
    let mut big = small.clone();
    let NodeKind::Geo(p) = big.nodes.get_mut(&GEO).unwrap() else { unreachable!() };
    assert!(p.graph.set_exposed("radius", 0.2), "the slider is exposed by name");

    // Same topology, same shader: nothing to compile. (The compositor
    // exposes no pipeline count; the topology hash and the generated WGSL
    // are what it keys pipelines on, so they are the same claim.)
    assert_eq!(small.topology_hash(), big.topology_hash(), "a slider edit recompiled");
    assert_eq!(
        engine::codegen::generate(&small).main.wgsl,
        engine::codegen::generate(&big).main.wgsl,
        "a slider edit changed the shader"
    );
    // And the second write_params sees the change as a layer to re-record.
    comp.write_params(&big);
    assert_eq!(comp.geo_layer_dirty(GEO), Some(true), "the edit did not mark the layer");

    let after = render(&ctx, &mut comp, &big);
    assert!(before != after, "the slider edit did not change the picture");
    // Radius doubled: a pixel that was outside the small disc but inside
    // the big one flipped.
    let r_small = 0.1 * W as f32;
    let probe = at(&after, 128 + (r_small * 1.5) as usize, 64);
    let was = at(&before, 128 + (r_small * 1.5) as usize, 64);
    assert!(was[3] < 1e-2 && probe[3] > 0.99, "radius edit: {was:?} -> {probe:?}");
}

/// A static drawing costs nothing per frame: the second write_params with
/// the same graph leaves the layer clean, scrubbing a Noise that does not
/// move leaves it clean, and only a field that actually moves re-records.
#[test]
fn an_unchanged_graph_records_nothing() {
    let Some(ctx) = ctx() else { return };
    let mut comp = comp(&ctx);
    // Density read from the field, so the Noise is part of the memo key.
    let mut g = GeoGraph::default();
    g.nodes.insert(1, GeoNode::FieldIn { shift: 0.0 });
    g.nodes.insert(2, GeoNode::Scatter { count: 200, seed: 0.0, density: 1.0 });
    g.nodes.insert(3, GeoNode::Output);
    g.edges.insert((2, 1), 1);
    g.edges.insert((3, 0), 2);
    g.next_id = 4;
    let mut graph = comp_of(g);
    wire_noise(&mut graph, moving_fbm(), 0.0, 0.0);

    let first = render(&ctx, &mut comp, &graph);
    assert!(coverage(&first, 0, W as usize) > 10.0, "nothing was scattered");
    assert_eq!(comp.geo_layer_dirty(GEO), Some(false), "encode_comp left the layer dirty");

    comp.write_params(&graph);
    assert_eq!(comp.geo_layer_dirty(GEO), Some(false), "an identical graph re-recorded");

    // A still Noise (speed 0) scrubbed to another time is the same field.
    let scrub = |graph: &mut Graph, speed: f32, time: f32| {
        let NodeKind::Noise(n) = graph.nodes.get_mut(&NOISE).unwrap() else { unreachable!() };
        n.speed = speed;
        n.time = time;
    };
    scrub(&mut graph, 0.0, 7.0);
    comp.write_params(&graph);
    assert_eq!(comp.geo_layer_dirty(GEO), Some(false), "scrubbing a still Noise re-recorded");

    // A moving one is a different field at a different time.
    scrub(&mut graph, 1.0, 3.0);
    comp.write_params(&graph);
    assert_eq!(comp.geo_layer_dirty(GEO), Some(true), "a moving Noise did not re-record");
    let later = render(&ctx, &mut comp, &graph);
    assert!(first != later, "the moving field did not move the scatter");
}


/// A Noise node INSIDE the drawing (its own field graph, no Noise in the
/// comp): it scatters by density on the GPU picture, it holds still with no
/// speed however the tick moves, and with a speed the tick re-records it.
#[test]
fn an_internal_noise_drives_the_drawing_and_animates_with_speed() {
    let Some(ctx) = ctx() else { return };
    let mut comp = comp(&ctx);
    let mut fg = FieldGraph::default();
    fg.nodes.insert(1, FieldNode::default());
    let mut g = GeoGraph::default();
    g.nodes.insert(1, GeoNode::Noise { graph: fg, scale: 1.0, seed: 0.0, speed: 0.0 });
    g.nodes.insert(
        2,
        GeoNode::Remap { in_lo: 0.45, in_hi: 0.55, out_lo: 0.0, out_hi: 1.0, clamp: true },
    );
    g.edges.insert((2, 0), 1);
    g.nodes.insert(3, GeoNode::Scatter { count: 300, seed: 0.0, density: 1.0 });
    g.edges.insert((3, 1), 2);
    g.nodes.insert(4, GeoNode::Output);
    g.edges.insert((4, 0), 3);
    g.next_id = 5;
    let mut graph = comp_of(g);
    let set = |graph: &mut Graph, speed: f32, time: f32| {
        let NodeKind::Geo(p) = graph.nodes.get_mut(&GEO).unwrap() else { unreachable!() };
        p.time = time;
        if let Some(GeoNode::Noise { speed: s, .. }) = p.graph.nodes.get_mut(&1) {
            *s = speed;
        }
    };
    let first = render(&ctx, &mut comp, &graph);
    assert!(coverage(&first, 0, W as usize) > 5.0, "the internal noise scattered nothing");
    assert_eq!(comp.geo_layer_dirty(GEO), Some(false));

    // No speed: another tick is the same drawing.
    set(&mut graph, 0.0, 30.0);
    comp.write_params(&graph);
    assert_eq!(comp.geo_layer_dirty(GEO), Some(false), "a still drawing re-recorded on a scrub");

    // A speed: the tick moves the noise, and the scatter with it.
    set(&mut graph, 0.05, 30.0);
    comp.write_params(&graph);
    assert_eq!(comp.geo_layer_dirty(GEO), Some(true), "the Noise's speed did not reach the memo");
    let later = render(&ctx, &mut comp, &graph);
    assert!(first != later, "the internal noise did not move the scatter");
}

/// A Stroke draws a LINE: a vee through three points, as wide as it says
/// along its length, nothing filled inside the vee, and no leg back from
/// the last point to the first.
#[test]
fn a_stroke_draws_a_line_and_fills_nothing() {
    let Some(ctx) = ctx() else { return };
    let mut comp = comp(&ctx);
    // y in WIDTH units: the canvas is 256 wide, so 0.1 is 25.6 px down.
    let pts = vec![
        SplinePoint::new([0.1, 0.1], 0.0),
        SplinePoint::new([0.5, 0.4], 0.0),
        SplinePoint::new([0.9, 0.1], 0.0),
    ];
    let img = render(
        &ctx,
        &mut comp,
        &comp_of(single(GeoNode::Stroke {
            points: pts,
            mode: SplineMode::Linear,
            color: [1.0, 1.0, 1.0, 1.0],
            width: 6.0,
        })),
    );
    // Down a column through the first leg (x = 64, where the leg is at
    // y = 54.4 px): the leg slopes 3:4, so a vertical cut through a band
    // six pixels wide is 6 / 0.8 = 7.5 px deep.
    let band: f32 = (35..75).map(|y| at(&img, 64, y)[3]).sum();
    assert!((band - 7.5).abs() < 0.5, "the stroke cuts {band} px deep, wanted 7.5");
    // Inside the vee, above the point, is empty: a stroke is never filled.
    assert!(at(&img, 128, 60)[3] < 1e-3, "the vee was filled");
    // No closing leg: the chord between the first and last points (y = 25.6)
    // is empty midway.
    assert!(at(&img, 128, 26)[3] < 1e-3, "the stroke closed back on itself");
}

/// A big shape is drawn as tiles: the ones the outline touches walk the
/// segments, the interior ones are flat quads. The seams must be invisible
/// — a half-transparent fill larger than the canvas reads the same
/// premultiplied colour at every pixel, and a half-transparent stroke
/// crossing tile boundaries never doubles up.
#[test]
fn tiled_shapes_have_no_seams() {
    let Some(ctx) = ctx() else { return };
    let mut comp = comp(&ctx);
    // Bigger than the canvas: every tile is interior (kind 3).
    let img = render(&ctx, &mut comp, &comp_of(single(rect(-0.5, -0.5, 1.5, 1.0, fill([1.0, 0.0, 0.0, 0.5])))));
    let off = img.iter().filter(|c| !close(**c, [0.5, 0.0, 0.0, 0.5], 1e-2)).count();
    assert_eq!(off, 0, "{off} pixels of a canvas-sized fill are not the fill");

    // A translucent 6 px stroke along a many-vertex diagonal, spanning
    // every tile column: alpha never exceeds its own, and the total is the
    // band's area (length × width, plus the two round caps) at half.
    let pt = |x: f32, y: f32| SplinePoint { p: [x, y], feather: 0.0, bias: 0.0, h_in: [0.0; 2], h_out: [0.0; 2], tension: 0.0 };
    let (a, b) = ([0.05, 0.05], [0.95, 0.45]);
    let points: Vec<SplinePoint> =
        (0..=24).map(|i| i as f32 / 24.0).map(|t| pt(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)).collect();
    let stroke = Style { fill: [0.0; 4], stroke: [1.0, 1.0, 1.0, 0.5], width: 6.0, ..Default::default() };
    let line = GeoNode::Spline { points, mode: SplineMode::Linear, closed: false, style: stroke };
    let img = render(&ctx, &mut comp, &comp_of(single(line)));
    let peak = img.iter().map(|c| c[3]).fold(0.0f32, f32::max);
    assert!(peak <= 0.5 + 1e-2, "a translucent stroke doubled up to {peak}");
    let len = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt() * W as f32;
    let want = (len * 6.0 + std::f32::consts::PI * 9.0) * 0.5;
    let total = coverage(&img, 0, W as usize);
    assert!((total - want).abs() < want * 0.03, "stroke integrates to {total:.1}, want {want:.1}");
}

/// An OPEN spline is an open outline: the leg back to the first point is
/// not drawn. An L stroked open is two legs, not a triangle.
#[test]
fn an_open_spline_has_no_return_leg() {
    let Some(ctx) = ctx() else { return };
    let mut comp = comp(&ctx);
    let pt = |x: f32, y: f32| SplinePoint { p: [x, y], feather: 0.0, bias: 0.0, h_in: [0.0; 2], h_out: [0.0; 2], tension: 0.0 };
    let stroke = Style { fill: [0.0; 4], stroke: [1.0; 4], width: 4.0, ..Default::default() };
    let l = GeoNode::Spline {
        points: vec![pt(0.1, 0.1), pt(0.9, 0.1), pt(0.9, 0.45)],
        mode: SplineMode::Linear,
        closed: false,
        style: stroke,
    };
    let img = render(&ctx, &mut comp, &comp_of(single(l)));
    // On the first leg (y = 0.1 → 25.6 px), and on the second (x = 0.9 → 230 px).
    assert!(at(&img, 128, 25)[3] > 0.9, "the first leg is missing");
    assert!(at(&img, 230, 70)[3] > 0.9, "the second leg is missing");
    // The midpoint of the would-be return leg (0.5, 0.275) → (128, 70): clear.
    let c = at(&img, 128, 70);
    assert!(c[3] < 1e-2, "the return leg was drawn: {c:?}");
    // Closed, the same points ARE a triangle.
    let tri = GeoNode::Spline {
        points: vec![pt(0.1, 0.1), pt(0.9, 0.1), pt(0.9, 0.45)],
        mode: SplineMode::Linear,
        closed: true,
        style: stroke,
    };
    let img = render(&ctx, &mut comp, &comp_of(single(tri)));
    assert!(at(&img, 128, 70)[3] > 0.9, "the closed outline lost its third side");
}

/// Volume on a 2K frame: as many instanced stars as the caps allow. Every
/// one lands where the CPU put it (fifty centres sampled), nothing was
/// capped, and the eval and raster times are printed so a regression shows
/// up in the log before it shows up in the panel.
///
/// The design asks for 2000 stars, but `cap` holds every node's POINTS to
/// MAX_POINTS (4096) and an Instance mints ten fresh points per star, so
/// 409 is the most that can exist; 400 keeps clear of the cap.
#[test]
fn four_hundred_stars_at_2k_land_where_the_cpu_put_them() {
    let Some(ctx) = ctx() else { return };
    const BW: u32 = 2048;
    const BH: u32 = 1080;
    const N: u32 = 400;
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(BW, BH);
    // Scatter over the whole canvas (no region), Instance a small star.
    let mut g = GeoGraph::default();
    g.nodes.insert(1, GeoNode::Scatter { count: N, seed: 0.0, density: 1.0 });
    g.nodes.insert(
        2,
        GeoNode::Star {
            center: [0.0, 0.0],
            outer: 0.006,
            inner: 0.003,
            points: 5,
            rotation: 0.0,
            style: Style::default(),
        },
    );
    g.nodes.insert(
        3,
        GeoNode::Instance {
            scale: 1.0,
            rotation: 0.0,
            inherit_cd: true,
            pivot: engine::geo::Pivot::BboxCentre,
        },
    );
    g.nodes.insert(4, GeoNode::Output);
    g.edges.insert((3, 0), 2);
    g.edges.insert((3, 1), 1);
    g.edges.insert((4, 0), 3);
    g.next_id = 5;

    // Where the CPU put the points: the same Scatter, evaluated alone with
    // the context the compositor uses at this size.
    let mut pts = g.clone();
    pts.edges.insert((4, 0), 1);
    let cx = engine::geo::EvalCx { tick: 0.0, field: None, seed: 0.0, aspect: BH as f32 / BW as f32, px_per_unit: BW as f32 };
    let centres = engine::geo::eval(&pts, &cx).points;
    assert_eq!(centres.len(), N as usize, "the scatter starved");

    let mut graph = Graph::default();
    graph.nodes.insert(GEO, NodeKind::Geo(GeoParams { size: (BW, BH), inherit: false, graph: g, seed: 0.0, ..Default::default() }));
    graph.view = Some(GEO);
    let img = render_at(&ctx, &mut comp, &graph, BW, BH);
    let (stats, eval_ms, raster_cost) = comp.geo_stats(GEO).expect("the node has a layer");
    assert!(raster_cost > 0, "a drawing that draws something has a re-record cost");
    eprintln!("{N} stars at {BW}x{BH}: {stats:?}, eval+flatten {eval_ms:.2} ms");
    assert_eq!(stats.prims, N as usize, "one prim per star");
    assert!(!stats.truncated, "a cap bit at {N} stars");

    // A star's centre has winding one, so every sampled centre is solid.
    for q in centres.iter().take(50) {
        let (x, y) = ((q.p[0] * BW as f32) as usize, (q.p[1] * BW as f32) as usize);
        let c = img[y * BW as usize + x];
        assert!(c[3] > 0.9, "star at ({x},{y}) reads {c:?}");
    }
}


/// A baked Physics node plays back through the compositor: the body is
/// where it was drawn at the start, lower later, the same tick twice does
/// not re-record, and another tick does.
#[test]
fn a_baked_physics_node_plays_back_per_tick() {
    let Some(ctx) = ctx() else { return };
    let mut comp = comp(&ctx);
    let mut g = GeoGraph::default();
    g.nodes.insert(1, rect(0.4, 0.05, 0.6, 0.15, fill([1.0; 4])));
    g.nodes.insert(
        2,
        GeoNode::Physics {
            settings: engine::geophys::PhysicsSettings { frames: 48, ..Default::default() },
            bake: Default::default(),
        },
    );
    g.edges.insert((2, 0), 1);
    g.nodes.insert(3, GeoNode::Output);
    g.edges.insert((3, 0), 2);
    g.next_id = 4;
    // Bake it the way the panel does: the input evaluated, then simulated.
    let aspect = H as f32 / W as f32;
    let cx = engine::geo::EvalCx { tick: 0.0, field: None, seed: 0.0, aspect, px_per_unit: W as f32 };
    let bodies = engine::geo::eval_node(&g, 1, &cx);
    let s = engine::geophys::PhysicsSettings { frames: 48, ..Default::default() };
    let b = engine::geophys::bake(&bodies, &Default::default(), &s, aspect, W as f32, 24.0);
    if let Some(GeoNode::Physics { bake, .. }) = g.nodes.get_mut(&2) {
        bake.0 = Some(std::sync::Arc::new(b));
    }
    let mut graph = comp_of(g);
    let at_tick = |graph: &mut Graph, t: f32| {
        let NodeKind::Geo(p) = graph.nodes.get_mut(&GEO).unwrap() else { unreachable!() };
        p.time = t;
    };
    // Rows with any coverage, top to bottom.
    let lowest_row = |img: &[[f32; 4]]| {
        (0..H as usize).rev().find(|y| (0..W as usize).any(|x| at(img, x, *y)[3] > 0.5)).unwrap_or(0)
    };
    let start = render(&ctx, &mut comp, &graph);
    // Drawn at y 0.15 * 256 = 38 px.
    assert!((lowest_row(&start) as i32 - 38).abs() <= 2, "start at row {}", lowest_row(&start));
    comp.write_params(&graph);
    assert_eq!(comp.geo_layer_dirty(GEO), Some(false), "the same tick re-recorded");
    at_tick(&mut graph, 40.0);
    comp.write_params(&graph);
    assert_eq!(comp.geo_layer_dirty(GEO), Some(true), "another tick did not re-record");
    let later = render(&ctx, &mut comp, &graph);
    // Fallen to the canvas floor (128 px), not through it.
    let low = lowest_row(&later);
    assert!(low > 100, "it did not fall: row {low}");
    assert!(low < 128, "row {low}");
}

/// Render every 2dini preset to PNG for a look: `GEO_DUMP=<dir> cargo test
/// -p engine --test geo dump_presets -- --ignored`. Physics presets are
/// baked and shown at their last frame, or at each of `GEO_DUMP_FRAMES=0,12,24`
/// (`<name>_<frame>.png`); `GEO_DUMP_ONLY=dam` keeps the presets whose name
/// holds it.
#[test]
#[ignore]
fn dump_presets() {
    let Some(ctx) = ctx() else { return };
    let dir = std::env::var("GEO_DUMP").unwrap_or_else(|_| "target/geo_presets".into());
    std::fs::create_dir_all(&dir).unwrap();
    let (w, h) = (960u32, 540u32);
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 1);
    comp.ensure_output(w, h);
    let only = std::env::var("GEO_DUMP_ONLY").unwrap_or_default();
    let frames: Vec<f32> = std::env::var("GEO_DUMP_FRAMES")
        .map(|v| v.split(',').filter_map(|f| f.trim().parse().ok()).collect())
        .unwrap_or_default();
    for p in engine::geopresets::PRESETS.iter().filter(|p| p.name.contains(only.as_str())) {
        let mut g = (p.build)();
        let aspect = h as f32 / w as f32;
        let cx = engine::geo::EvalCx { tick: 0.0, field: None, seed: 0.0, aspect, px_per_unit: w as f32 };
        let mut last = 0.0;
        let phys: Vec<(NodeId, engine::geophys::PhysicsSettings)> = g
            .nodes
            .iter()
            .filter_map(|(id, n)| match n {
                GeoNode::Physics { settings, .. } => Some((*id, *settings)),
                _ => None,
            })
            .collect();
        for (id, s) in phys {
            let bodies = g.input_of(id, 0).map(|n| engine::geo::eval_node(&g, n, &cx)).unwrap_or_default();
            let cols = g.input_of(id, 1).map(|n| engine::geo::eval_node(&g, n, &cx)).unwrap_or_default();
            let b = engine::geophys::bake(&bodies, &cols, &s, aspect, w as f32, 24.0);
            last = (b.frames as f32 - 1.0).max(0.0);
            if let Some(GeoNode::Physics { bake, .. }) = g.nodes.get_mut(&id) {
                bake.0 = Some(std::sync::Arc::new(b));
            }
        }
        let out = g.out;
        g.nodes.insert(999, GeoNode::Output);
        if let Some(o) = out {
            g.edges.insert((999, 0), o);
        }
        let ticks: Vec<(f32, String)> = if frames.is_empty() {
            vec![(last, String::new())]
        } else {
            frames.iter().map(|&f| (f, format!("_{f}"))).collect()
        };
        for (time, suffix) in ticks {
        let mut graph = Graph::default();
        graph.nodes.insert(GEO, NodeKind::Geo(GeoParams { size: (w, h), graph: g.clone(), time, ..Default::default() }));
        graph.view = Some(GEO);
        let img = render_at(&ctx, &mut comp, &graph, w, h);
        let mut bytes = Vec::with_capacity((w * h * 4) as usize);
        for px in &img {
            // Over a dark grey so strokes and fills both read.
            let a = px[3].clamp(0.0, 1.0);
            for c in 0..3 {
                let v = px[c].clamp(0.0, 1.0) + 0.12 * (1.0 - a);
                bytes.push((v.powf(1.0 / 2.2) * 255.0).round() as u8);
            }
            bytes.push(255);
        }
        let path = format!("{dir}/{}{suffix}.png", p.name.replace(' ', "_"));
        let f = std::fs::File::create(&path).unwrap();
        let mut enc = png::Encoder::new(std::io::BufWriter::new(f), w, h);
        enc.set_color(png::ColorType::Rgba);
        enc.set_depth(png::BitDepth::Eight);
        enc.write_header().unwrap().write_image_data(&bytes).unwrap();
        eprintln!("wrote {path}");
        }
    }
}

/// Text draws with its counters as holes: the middle of a big "o" is
/// empty, its ring is ink.
#[test]
fn text_draws_with_holes() {
    let Some(ctx) = ctx() else { return };
    let mut comp = comp(&ctx);
    let g = single(GeoNode::Text {
        text: "o".into(),
        font: Default::default(),
        size: 0.5,
        position: [0.35, 0.4],
        align: Default::default(),
        tracking: 0.0,
        leading: 1.2,
        rotation: 0.0,
        style: fill([1.0; 4]),
        fx: Default::default(),
    });
    // Where the glyph landed, from the CPU side.
    let cx = engine::geo::EvalCx { tick: 0.0, field: None, seed: 0.0, aspect: H as f32 / W as f32, px_per_unit: W as f32 };
    let geo = engine::geo::eval(&g, &cx);
    let (lo, hi) = geo.bbox().unwrap();
    let img = render(&ctx, &mut comp, &comp_of(g));
    let px = |p: [f32; 2]| at(&img, (p[0] * W as f32) as usize, (p[1] * W as f32) as usize);
    let mid = [(lo[0] + hi[0]) * 0.5, (lo[1] + hi[1]) * 0.5];
    assert!(px(mid)[3] < 0.05, "the counter is filled: {:?}", px(mid));
    let ink = [lo[0] + (hi[0] - lo[0]) * 0.06, mid[1]];
    assert!(px(ink)[3] > 0.9, "the ring is empty: {:?}", px(ink));
}

/// `GEO_DUMP=<dir> cargo test -p engine --test geo text_style_showcase --
/// --ignored`: a 2dini Text wearing the Text node's styling (outline,
/// weight, italic, drop shadow) and a gradient fill, for a look.
#[test]
#[ignore]
fn text_style_showcase() {
    let Some(ctx) = ctx() else { return };
    let dir = std::env::var("GEO_DUMP").unwrap_or_else(|_| "target/geo_presets".into());
    std::fs::create_dir_all(&dir).unwrap();
    let (w, h) = (960u32, 360u32);
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 1);
    comp.ensure_output(w, h);
    let g = single(GeoNode::Text {
        text: "Styled 2dini".into(),
        font: Default::default(),
        size: 0.13,
        position: [0.5, 0.24],
        align: engine::geotext::TextAlign::Centre,
        tracking: 0.02,
        leading: 1.2,
        rotation: 0.0,
        style: fill([1.0, 0.8, 0.25, 1.0]),
        fx: engine::textfx::GeoTextFx {
            skew: 12.0,
            weight: 0.015,
            outline: 0.03,
            outline_color: [0.1, 0.05, 0.3, 1.0],
            shadow: true,
            shadow_offset: [10.0, 12.0],
            ..Default::default()
        },
    });
    let mut graph = Graph::default();
    graph.nodes.insert(GEO, NodeKind::Geo(GeoParams { size: (w, h), graph: g, ..Default::default() }));
    graph.view = Some(GEO);
    let img = render_at(&ctx, &mut comp, &graph, w, h);
    let mut bytes = Vec::with_capacity((w * h * 4) as usize);
    for px in &img {
        let a = px[3].clamp(0.0, 1.0);
        for c in 0..3 {
            let v = px[c].clamp(0.0, 1.0) + 0.3 * (1.0 - a);
            bytes.push((v.powf(1.0 / 2.2) * 255.0).round() as u8);
        }
        bytes.push(255);
    }
    let f = std::fs::File::create(format!("{dir}/text_style.png")).unwrap();
    let mut enc = png::Encoder::new(std::io::BufWriter::new(f), w, h);
    enc.set_color(png::ColorType::Rgba);
    enc.write_header().unwrap().write_image_data(&bytes).unwrap();
}
