//! Headless integration for the SDF sub-graph: the unit tests in `sdf.rs`
//! prove the right WGSL comes out and that naga accepts it; these prove the
//! distances MEAN what they say once a GPU runs them — a circle is its
//! radius, a closed spline has an inside, an open one does not, the layer
//! sits over the input, a refraction bends it, and a proxy halves it all.
//! Self-skips without a GPU (CI).

use engine::compositor::Compositor;
use engine::decode::ImageInfo;
use engine::graph::{Graph, NodeKind, SdfParams, SplineMode, SplinePoint};
use engine::sdf::{CombineOp, SdfGraph, SdfNode};
use engine::timing::GpuTimer;
use half::f16;

const W: u32 = 256;
const H: u32 = 128;
const WHITE: [f32; 4] = [1.0, 1.0, 1.0, 1.0];

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP sdf test: no adapter ({e})");
            None
        }
    }
}

/// Every pixel's rgba, row-major, at `w`x`h`.
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
    ctx.queue.submit([encoder.finish()]);
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    let data = slice.get_mapped_range().expect("a mapped buffer");
    let mut out = Vec::with_capacity((w * h) as usize);
    for y in 0..h as usize {
        let off = y * padded as usize;
        let halves: &[f16] = bytemuck::cast_slice(&data[off..off + (w * 8) as usize]);
        for x in 0..w as usize {
            let q = &halves[x * 4..x * 4 + 4];
            out.push([q[0].to_f32(), q[1].to_f32(), q[2].to_f32(), q[3].to_f32()]);
        }
    }
    drop(data);
    staging.unmap();
    out
}

fn render(ctx: &engine::GpuContext, comp: &mut Compositor, graph: &Graph) -> Vec<[f32; 4]> {
    render_at(ctx, comp, graph, W, H)
}

fn at(img: &[[f32; 4]], x: u32, y: u32) -> [f32; 4] {
    img[(y * W + x) as usize]
}

/// A graph whose Output is `effect` fed by `shape`, as nodes 1 (shape),
/// 2 (effect), 3 (Output).
fn chain(shape: SdfNode, effect: SdfNode) -> SdfGraph {
    let mut g = SdfGraph::default();
    g.nodes.insert(1, shape);
    g.nodes.insert(2, effect);
    g.nodes.insert(3, SdfNode::Output);
    g.edges.insert((2, 0), 1);
    g.edges.insert((3, 0), 2);
    g
}

/// A comp with one SDF node (id 5), nothing on its input, viewed.
fn comp_of(sub: SdfGraph) -> Graph {
    let mut g = Graph::default();
    g.nodes.insert(5, NodeKind::Sdf(SdfParams { size: (W, H), graph: sub, ..Default::default() }));
    g.formats.insert(5, (W, H));
    g.view = Some(5);
    g
}

fn fill() -> SdfNode {
    SdfNode::Fill { color: WHITE, soft: 0.0 }
}

fn circle(x: f32, y: f32, r: f32) -> SdfNode {
    SdfNode::Circle { center: [x, y], radius: r }
}

/// A plate on reader 0: red ramps left to right, green is 0.25, alpha 1.
fn ramp_plate(ctx: &engine::GpuContext, comp: &mut Compositor) {
    comp.ensure_reader(0, ImageInfo { width: W, height: H });
    let px: Vec<[f16; 4]> = (0..W * H)
        .map(|i| {
            let x = (i % W) as f32 / (W - 1) as f32;
            [f16::from_f32(x), f16::from_f32(0.25), f16::ZERO, f16::ONE]
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
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(W * 8), rows_per_image: Some(H) },
        wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
    );
}

/// A circle IS its radius: full inside, empty two pixels past the edge,
/// half on it.
#[test]
fn a_filled_circle_is_its_radius() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    // Radius 40.5 puts the edge on the CENTRE of column 168 (pixel centres
    // sit at +0.5), where coverage must be a half.
    let img = render(&ctx, &mut comp, &comp_of(chain(circle(128.0, 64.0, 40.5), fill())));
    assert!(at(&img, 128, 64)[0] > 0.99, "centre {:?}", at(&img, 128, 64));
    assert!(at(&img, 128 + 37, 64)[0] > 0.99, "just inside {:?}", at(&img, 165, 64));
    assert!(at(&img, 128 + 43, 64)[3] < 0.01, "just outside {:?}", at(&img, 171, 64));
    let edge = at(&img, 168, 64)[3];
    assert!((0.4..0.6).contains(&edge), "the edge is antialiased: {edge}");
}

/// A closed spline has an inside; the same points OPEN are a curve with no
/// inside at all, and a stroke draws along it.
#[test]
fn a_closed_spline_has_an_inside_and_an_open_one_is_a_path() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    // Field space: x across the width, y in WIDTH units (so 128px tall is
    // 0.5). A square from (64,32) to (192,96) in pixels.
    let pts = [[0.25, 0.125], [0.75, 0.125], [0.75, 0.375], [0.25, 0.375]]
        .map(|p| SplinePoint::new(p, 0.0))
        .to_vec();
    let spline = |closed| SdfNode::Spline {
        points: pts.clone(),
        mode: SplineMode::Linear,
        closed,
        offset: [0.0, 0.0],
    };
    let img = render(&ctx, &mut comp, &comp_of(chain(spline(true), fill())));
    assert!(at(&img, 128, 64)[3] > 0.99, "inside the closed square");
    assert!(at(&img, 20, 64)[3] < 0.01, "outside it");
    assert!(at(&img, 100, 40)[3] > 0.99, "near a corner, inside");

    // Open, the path runs top, right, bottom; the left side was the
    // closing leg and is gone.
    let stroke = SdfNode::Stroke { color: WHITE, width: 4.0, soft: 0.0 };
    let img = render(&ctx, &mut comp, &comp_of(chain(spline(false), stroke)));
    assert!(at(&img, 128, 64)[3] < 0.01, "an open path has no inside");
    assert!(at(&img, 128, 32)[3] > 0.9, "the stroke lies on the top leg");
    assert!(at(&img, 192, 64)[3] > 0.9, "and on the right one");
    assert!(at(&img, 64, 64)[3] < 0.01, "but not on the closing leg");
}

/// The layer goes OVER the input: the plate shows around the shape, the
/// fill covers it inside, and a glow adds light without covering.
#[test]
fn the_layer_sits_over_the_input_and_a_glow_is_light() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    ramp_plate(&ctx, &mut comp);
    let mut g = comp_of(chain(circle(128.0, 64.0, 20.0), fill()));
    g.nodes.insert(0, NodeKind::read(0, ""));
    g.edges.insert((5, 0), 0);
    let img = render(&ctx, &mut comp, &g);
    assert!((at(&img, 20, 10)[1] - 0.25).abs() < 0.01, "the plate outside: {:?}", at(&img, 20, 10));
    assert!(at(&img, 128, 64)[1] > 0.99, "the fill inside: {:?}", at(&img, 128, 64));

    let glow = SdfNode::Glow { color: [1.0, 0.0, 0.0, 1.0], radius: 10.0, intensity: 1.0 };
    if let Some(NodeKind::Sdf(p)) = g.nodes.get_mut(&5) {
        p.graph.nodes.insert(2, glow);
    }
    let img = render(&ctx, &mut comp, &g);
    let plate = 150.0 / 255.0;
    let near = at(&img, 150, 64); // 2px past the edge
    let far = at(&img, 250, 64);
    assert!(near[0] > plate + 0.5, "light near the edge adds to the plate: {near:?}");
    assert!((near[1] - 0.25).abs() < 0.01, "and covers nothing: {near:?}");
    assert!(far[0] - 250.0 / 255.0 < 0.01, "and has died away far off: {far:?}");
}

/// Smooth union melts two discs together: the gap between them fills in.
#[test]
fn a_smooth_union_fills_the_seam() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    let mut sub = SdfGraph::default();
    sub.nodes.insert(1, circle(100.0, 64.0, 24.0));
    sub.nodes.insert(2, circle(156.0, 64.0, 24.0));
    sub.nodes.insert(3, SdfNode::Combine { op: CombineOp::Union, smooth: 0.0 });
    sub.nodes.insert(4, fill());
    sub.nodes.insert(6, SdfNode::Output);
    sub.edges.insert((3, 0), 1);
    sub.edges.insert((3, 1), 2);
    sub.edges.insert((4, 0), 3);
    sub.edges.insert((6, 0), 4);
    let mut g = comp_of(sub);
    let hard = render(&ctx, &mut comp, &g);
    assert!(at(&hard, 128, 64)[3] < 0.01, "a hard union leaves the gap");
    if let Some(NodeKind::Sdf(p)) = g.nodes.get_mut(&5) {
        p.graph.nodes.insert(3, SdfNode::Combine { op: CombineOp::Union, smooth: 40.0 });
    }
    let topo = g.topology_hash();
    let soft = render(&ctx, &mut comp, &g);
    assert!(at(&soft, 128, 64)[3] > 0.99, "a smooth one bridges it");
    if let Some(NodeKind::Sdf(p)) = g.nodes.get_mut(&5) {
        p.graph.nodes.insert(3, SdfNode::Combine { op: CombineOp::Subtract, smooth: 40.0 });
    }
    assert_eq!(g.topology_hash(), topo, "the op is a uniform, not a recompile");
    let sub = render(&ctx, &mut comp, &g);
    assert!(at(&sub, 100, 64)[3] > 0.99 && at(&sub, 156, 64)[3] < 0.01, "a minus b");
}

/// Refraction samples the input somewhere else inside the rim and nowhere
/// else outside it.
#[test]
fn a_refraction_bends_the_input_only_inside() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    ramp_plate(&ctx, &mut comp);
    let mut g = comp_of(chain(circle(128.0, 64.0, 40.0), SdfNode::Refract { amount: 20.0, width: 30.0 }));
    g.nodes.insert(0, NodeKind::read(0, ""));
    g.edges.insert((5, 0), 0);
    let img = render(&ctx, &mut comp, &g);
    let plate = |x: u32| x as f32 / (W - 1) as f32;
    // Just inside the right rim the outward normal is +x; sampling back
    // along it reads the plate further LEFT.
    let rim = at(&img, 128 + 36, 64)[0];
    assert!(rim < plate(164) - 0.03, "the rim is bent: {rim} vs plate {}", plate(164));
    assert!((at(&img, 128, 64)[0] - plate(128)).abs() < 0.01, "the flat middle is not");
    assert!((at(&img, 200, 64)[0] - plate(200)).abs() < 0.01, "outside is untouched");
}

/// Ripples travel with the tick.
#[test]
fn ripples_move_with_time() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    let rip = SdfNode::Ripples {
        color: WHITE,
        spacing: 16.0,
        width: 3.0,
        speed: 2.0,
        falloff: 0.0,
        side: engine::sdf::Side::Outside,
    };
    let mut g = comp_of(chain(circle(128.0, 64.0, 10.0), rip));
    let row = |img: &[[f32; 4]]| (140..200).map(|x| at(img, x, 64)[3]).collect::<Vec<_>>();
    let a = render(&ctx, &mut comp, &g);
    assert!(at(&a, 128, 64)[3] < 0.01, "outside only");
    if let Some(NodeKind::Sdf(p)) = g.nodes.get_mut(&5) {
        p.time = 4.0; // 8 px: half a spacing
    }
    let b = render(&ctx, &mut comp, &g);
    let (ra, rb) = (row(&a), row(&b));
    let lit_a: f32 = ra.iter().sum();
    let lit_b: f32 = rb.iter().sum();
    assert!(lit_a > 3.0 && lit_b > 3.0, "there are rings: {lit_a} {lit_b}");
    let moved: f32 = ra.iter().zip(&rb).map(|(x, y)| (x - y).abs()).sum();
    assert!(moved > lit_a, "half a spacing on, the rings sit between the old ones: {moved}");
}

/// Under a half proxy the whole drawing halves: a Slider wired into the
/// radius scales exactly like the literal did.
#[test]
fn a_proxy_halves_the_drawing_driven_or_not() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    let (w, h) = (W / 2, H / 2);
    comp.ensure_output(w, h);
    let mut sub = chain(circle(128.0, 64.0, 40.0), fill());
    sub.nodes.insert(7, SdfNode::Const { value: 40.0, name: "r".into() });
    let mut g = comp_of(sub.clone());
    g.proxy = 0.5;
    g.formats.insert(5, (w, h));
    let lit = render_at(&ctx, &mut comp, &g, w, h);
    sub.edges.insert((1, 2), 7);
    let mut g2 = comp_of(sub);
    g2.proxy = 0.5;
    g2.formats.insert(5, (w, h));
    let driven = render_at(&ctx, &mut comp, &g2, w, h);
    let px = |img: &[[f32; 4]], x: u32| img[(32 * w + x) as usize][3];
    for img in [&lit, &driven] {
        assert!(px(img, 64 + 18) > 0.99, "inside the halved radius");
        assert!(px(img, 64 + 22) < 0.01, "outside it");
    }
}
