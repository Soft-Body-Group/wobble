//! 2dini light (holographic radiance cascades) and SDF effects on a GPU.
//!
//! A lamp, a wall, and the claims the feature rests on: light reaches what
//! the lamp can see and not what the wall hides; a shape is lit on the
//! side facing the lamp; the effects land where their params say (an
//! outline outside the edge, a grown shape, a drop shadow behind it); and
//! a look edit is never a recompile. Self-skips without a GPU (CI).

use engine::compositor::Compositor;
use engine::geo::{GeoGraph, GeoNode, Style};
use engine::geolight::GeoLook;
use engine::graph::{GeoParams, Graph, NodeId, NodeKind, SplineMode, SplinePoint};
use engine::timing::GpuTimer;
use half::f16;

const W: u32 = 256;
const H: u32 = 128;
const GEO: NodeId = 1;

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP geolight test: no adapter ({e})");
            None
        }
    }
}

fn render(ctx: &engine::GpuContext, comp: &mut Compositor, graph: &Graph) -> Vec<[f32; 4]> {
    render_at(ctx, comp, graph, W, H)
}

fn render_at(ctx: &engine::GpuContext, comp: &mut Compositor, graph: &Graph, w: u32, h: u32) -> Vec<[f32; 4]> {
    assert!(comp.set_graph(graph), "fused shader failed to compile");
    let timer = GpuTimer::new(ctx);
    comp.write_params(graph);
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
            layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(padded), rows_per_image: Some(h) },
        },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    let t0 = std::time::Instant::now();
    ctx.queue.submit([encoder.finish()]);
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    eprintln!("render {w}x{h}: {:.2} ms", t0.elapsed().as_secs_f32() * 1e3);
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
            out.push([halves[i].to_f32(), halves[i + 1].to_f32(), halves[i + 2].to_f32(), halves[i + 3].to_f32()]);
        }
    }
    drop(data);
    staging.unmap();
    out
}

fn at(img: &[[f32; 4]], x: usize, y: usize) -> [f32; 4] {
    img[y * W as usize + x]
}

fn luma(c: [f32; 4]) -> f32 {
    (c[0] + c[1] + c[2]) / 3.0
}

/// A rectangle in PIXELS (geo space is the canvas width wide).
fn rect(x0: f32, y0: f32, x1: f32, y1: f32, style: Style) -> GeoNode {
    let w = W as f32;
    let pt = |x: f32, y: f32| SplinePoint { p: [x / w, y / w], feather: 0.0, bias: 0.0, h_in: [0.0; 2], h_out: [0.0; 2], tension: 0.0 };
    GeoNode::Spline {
        points: vec![pt(x0, y0), pt(x1, y0), pt(x1, y1), pt(x0, y1)],
        mode: SplineMode::Linear,
        closed: true,
        style,
    }
}

fn fill(c: f32) -> Style {
    Style { fill: [c, c, c, 1.0], ..Default::default() }
}

/// Shapes merged in order into the Output; `lamp` (if any) is wrapped in a
/// Light node first.
fn scene(shapes: Vec<GeoNode>, lamp: Option<GeoNode>) -> GeoGraph {
    let mut g = GeoGraph::default();
    let mut id = 1;
    let mut feeds = Vec::new();
    for s in shapes {
        g.nodes.insert(id, s);
        feeds.push(id);
        id += 1;
    }
    if let Some(l) = lamp {
        g.nodes.insert(id, l);
        g.nodes.insert(id + 1, GeoNode::Light { strength: 4.0, tint: [1.0; 3], occlude: true });
        g.edges.insert((id + 1, 0), id);
        feeds.push(id + 1);
        id += 2;
    }
    let merge = id;
    g.nodes.insert(merge, GeoNode::Merge { count: feeds.len() as u32 });
    for (i, f) in feeds.iter().enumerate() {
        g.edges.insert((merge, i), *f);
    }
    g.nodes.insert(merge + 1, GeoNode::Output);
    g.edges.insert((merge + 1, 0), merge);
    g.next_id = merge + 2;
    g
}

fn comp_of(sub: GeoGraph, look: GeoLook) -> Graph {
    comp_sized(sub, look, W, H)
}

fn comp_sized(sub: GeoGraph, look: GeoLook, w: u32, h: u32) -> Graph {
    let mut g = Graph::default();
    g.nodes.insert(GEO, NodeKind::Geo(GeoParams { size: (w, h), inherit: false, graph: sub, look, ..Default::default() }));
    g.view = Some(GEO);
    g
}

fn lit() -> GeoLook {
    GeoLook { light: true, ambient: 0.0, air: 1.0, spacing: 2.0, ..Default::default() }
}

/// The lamp at the left (30, 64), a wall at x 120..130 spanning y 44..84.
fn lamp_and_wall() -> GeoGraph {
    scene(vec![rect(120.0, 44.0, 130.0, 84.0, fill(0.5))], Some(rect(24.0, 58.0, 36.0, 70.0, fill(1.0))))
}

#[test]
fn the_wall_casts_a_shadow_and_the_air_round_the_lamp_glows() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    let img = render(&ctx, &mut comp, &comp_of(lamp_and_wall(), lit()));
    // Empty space: the air holds the light, alpha 0.
    let near = at(&img, 60, 64);
    let hidden = at(&img, 200, 64);
    let seen = at(&img, 200, 6);
    eprintln!("near {near:?} hidden {hidden:?} seen {seen:?}");
    assert!(luma(near) > 0.02 && near[3] < 1e-3, "no light in the air by the lamp: {near:?}");
    assert!(luma(seen) > 3.0 * luma(hidden), "the wall casts no shadow: seen {seen:?} hidden {hidden:?}");
    assert!(luma(near) > luma(seen), "light does not fall off with distance");
    // The wall: lit on the lamp's side, dark on the far one.
    let front = at(&img, 121, 64);
    let back = at(&img, 129, 64);
    eprintln!("front {front:?} back {back:?}");
    assert!(front[3] > 0.99 && back[3] > 0.99, "the wall is not opaque");
    assert!(luma(front) > 2.0 * luma(back), "the wall's lit side is not the lamp's: {front:?} vs {back:?}");
    // The lamp itself is bright.
    assert!(luma(at(&img, 30, 64)) > 1.0, "the lamp gives off nothing: {:?}", at(&img, 30, 64));
}

#[test]
fn light_off_leaves_the_drawing_as_it_was() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    let plain = render(&ctx, &mut comp, &comp_of(lamp_and_wall(), GeoLook::default()));
    assert_eq!(at(&plain, 125, 64), [0.5, 0.5, 0.5, 1.0]);
    assert_eq!(at(&plain, 200, 64), [0.0; 4]);
    // An effect on, light off: the shapes keep their colour.
    let fx = render(&ctx, &mut comp, &comp_of(lamp_and_wall(), GeoLook { outline: true, ..Default::default() }));
    assert_eq!(at(&fx, 125, 64), [0.5, 0.5, 0.5, 1.0]);
}

#[test]
fn the_effects_land_where_their_params_say() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    // A square x 96.5..160, y 32..96.
    let sq = || scene(vec![rect(96.5, 32.0, 160.0, 96.0, fill(0.5))], None);
    // Outline outside, 6 px, red: 3 px out of the left edge is red, solid.
    let red = [1.0, 0.0, 0.0, 1.0];
    let look = GeoLook { outline: true, outline_width: 6.0, outline_pos: 1.0, outline_color: red, ..Default::default() };
    let img = render(&ctx, &mut comp, &comp_of(sq(), look));
    let o = at(&img, 93, 64);
    assert!((o[0] - 1.0).abs() < 0.02 && o[1] < 0.02 && o[3] > 0.98, "outline 3 px out = {o:?}");
    assert!(at(&img, 85, 64)[3] < 0.02, "outline wider than asked: {:?}", at(&img, 85, 64));
    assert_eq!(at(&img, 128, 64), [0.5, 0.5, 0.5, 1.0], "inside untouched");
    // Offset +5: the shape grows 5 px, in its own colour.
    let img = render(&ctx, &mut comp, &comp_of(sq(), GeoLook { offset: 5.0, ..Default::default() }));
    let g = at(&img, 93, 64);
    assert!((g[0] - 0.5).abs() < 0.02 && g[3] > 0.98, "grown pixel = {g:?}");
    assert!(at(&img, 90, 64)[3] < 0.05, "grew too far: {:?}", at(&img, 90, 64));
    // Offset -5 shrinks it.
    let img = render(&ctx, &mut comp, &comp_of(sq(), GeoLook { offset: -5.0, ..Default::default() }));
    assert!(at(&img, 99, 64)[3] < 0.05 && at(&img, 104, 64)[3] > 0.98, "did not shrink");
    // Drop shadow 10 px right and down, hard: below-right of the square is
    // shadow, above-left is not.
    let look = GeoLook { shadow: true, shadow_offset: [10.0, 10.0], shadow_soft: 0.5, shadow_color: [0.0, 0.0, 0.0, 0.5], ..Default::default() };
    let img = render(&ctx, &mut comp, &comp_of(sq(), look));
    let s = at(&img, 165, 100);
    assert!((s[3] - 0.5).abs() < 0.05 && s[0] < 0.01, "shadow = {s:?}");
    assert!(at(&img, 92, 28)[3] < 0.01, "shadow on the wrong side");
    // Glow: falls off away from the edge.
    let look = GeoLook { glow: true, glow_size: 20.0, glow_color: [1.0, 1.0, 0.0, 1.0], ..Default::default() };
    let img = render(&ctx, &mut comp, &comp_of(sq(), look));
    let (a, b, c) = (at(&img, 92, 64)[3], at(&img, 84, 64)[3], at(&img, 70, 64)[3]);
    assert!(a > b && b > c && c < 0.01, "glow falloff {a} {b} {c}");
    // Bevel: lit from upper left, the top edge is brighter than the bottom.
    let look = GeoLook { bevel: true, bevel_depth: 8.0, bevel_angle: 90.0, bevel_strength: 0.5, ..Default::default() };
    let img = render(&ctx, &mut comp, &comp_of(sq(), look));
    let (top, bottom) = (luma(at(&img, 128, 34)), luma(at(&img, 128, 93)));
    assert!(top > 0.55 && bottom < 0.45, "bevel top {top} bottom {bottom}");
}

#[test]
fn a_look_edit_is_not_a_recompile() {
    let a = comp_of(lamp_and_wall(), lit());
    let b = comp_of(lamp_and_wall(), GeoLook { intensity: 3.0, outline: true, ..lit() });
    assert_eq!(a.topology_hash(), b.topology_hash());
}

/// A look at it: three coloured lamps among walls and pillars, lit, with a
/// bevel and an outline, written to `geolight-demo.png` in cargo's test tmp dir (display
/// sRGB, clipped). `cargo test --release -p engine --test geolight demo -- --ignored`
#[test]
#[ignore]
fn demo() {
    let Some(ctx) = ctx() else { return };
    let (w, h) = (768u32, 432u32);
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(w, h);
    let s = w as f32 / W as f32; // rect() is in 256-wide pixels
    let px = |x0: f32, y0: f32, x1: f32, y1: f32, st: Style| rect(x0 / s, y0 / s, x1 / s, y1 / s, st);
    let grey = fill(0.6);
    let shapes = vec![
        px(300.0, 60.0, 320.0, 250.0, grey),
        px(420.0, 200.0, 600.0, 220.0, grey),
        px(150.0, 300.0, 260.0, 320.0, grey),
        px(520.0, 60.0, 560.0, 100.0, grey),
        px(620.0, 300.0, 660.0, 340.0, grey),
        px(80.0, 120.0, 120.0, 160.0, grey),
    ];
    // Three lamps, each through its own Light node.
    let mut g = GeoGraph::default();
    let mut id = 1;
    let mut feeds = Vec::new();
    for sh in shapes {
        g.nodes.insert(id, sh);
        feeds.push(id);
        id += 1;
    }
    for (x, y, c) in [(200.0, 200.0, [1.0, 0.45, 0.2]), (460.0, 120.0, [0.25, 0.55, 1.0]), (560.0, 330.0, [0.4, 1.0, 0.45])] {
        g.nodes.insert(id, px(x - 9.0, y - 9.0, x + 9.0, y + 9.0, Style { fill: [c[0], c[1], c[2], 1.0], ..Default::default() }));
        g.nodes.insert(id + 1, GeoNode::Light { strength: 6.0, tint: [1.0; 3], occlude: true });
        g.edges.insert((id + 1, 0), id);
        feeds.push(id + 1);
        id += 2;
    }
    g.nodes.insert(id, GeoNode::Merge { count: feeds.len() as u32 });
    for (i, f) in feeds.iter().enumerate() {
        g.edges.insert((id, i), *f);
    }
    g.nodes.insert(id + 1, GeoNode::Output);
    g.edges.insert((id + 1, 0), id);
    g.next_id = id + 2;
    let look = GeoLook {
        light: true,
        intensity: 1.5,
        ambient: 0.02,
        air: 1.0,
        spacing: 2.0,
        bevel: true,
        bevel_depth: 6.0,
        bevel_strength: 0.3,
        ..Default::default()
    };
    let t0 = std::time::Instant::now();
    let img = render_at(&ctx, &mut comp, &comp_sized(g, look, w, h), w, h);
    eprintln!("demo {w}x{h} in {:.1} ms (two renders)", t0.elapsed().as_secs_f32() * 1e3);
    let enc = |v: f32| {
        let v = v.clamp(0.0, 1.0);
        let e = if v <= 0.0031308 { 12.92 * v } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 };
        (e * 255.0 + 0.5) as u8
    };
    let mut bytes = Vec::with_capacity((w * h * 3) as usize);
    for p in &img {
        bytes.extend([enc(p[0]), enc(p[1]), enc(p[2])]);
    }
    let path = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("geolight-demo.png");
    let file = std::fs::File::create(&path).unwrap();
    let mut e = png::Encoder::new(std::io::BufWriter::new(file), w, h);
    e.set_color(png::ColorType::Rgb);
    e.set_depth(png::BitDepth::Eight);
    e.write_header().unwrap().write_image_data(&bytes).unwrap();
    eprintln!("wrote {}", path.display());
}
