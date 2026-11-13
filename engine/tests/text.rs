//! The Text and Burnin nodes on real pixels: the atlas uploaded, sampled in
//! the fused pass and read back. Ink must land where the node says, move
//! with its position, take its colour, and a Burnin must print the same
//! glyphs as a Text of the same words — with a new frame or colour never
//! touching the topology hash. Self-skips without a GPU (CI).

use engine::color::Colorspace;
use engine::compositor::Compositor;
use engine::decode::ImageInfo;
use engine::geotext::{TextAlign, TextFont};
use engine::graph::{BurninCorner, BurninParams, Graph, NodeKind, TextParams};
use engine::timing::GpuTimer;
use half::f16;

const W: u32 = 320;
const H: u32 = 96;

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP text test: no adapter ({e})");
            None
        }
    }
}

fn comp(ctx: &engine::GpuContext) -> Compositor {
    let mut c = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 1);
    c.ensure_output(W, H);
    c
}

fn frame(ctx: &engine::GpuContext, comp: &mut Compositor, graph: &Graph) -> Vec<[f32; 4]> {
    assert!(comp.set_graph(graph), "fused shader failed to compile");
    let timer = GpuTimer::new(ctx);
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
            layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(padded), rows_per_image: Some(H) },
        },
        wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
    );
    ctx.queue.submit([encoder.finish()]);
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    let data = slice.get_mapped_range().expect("a mapped buffer");
    let mut out = Vec::new();
    for y in 0..H as usize {
        let row: &[f16] = bytemuck::cast_slice(&data[y * padded as usize..][..(W * 8) as usize]);
        for x in 0..W as usize {
            out.push([0, 1, 2, 3].map(|c| row[x * 4 + c].to_f32()));
        }
    }
    drop(data);
    staging.unmap();
    out
}

/// `WOBBLE_TEXT_DUMP=<dir>` writes a render as an 8-bit PNG (alpha over
/// mid grey) for a look.
fn dump(name: &str, px: &[[f32; 4]]) {
    let Ok(dir) = std::env::var("WOBBLE_TEXT_DUMP") else { return };
    let rgb: Vec<u8> = px
        .iter()
        .flat_map(|p| (0..3).map(move |k| ((p[k] + 0.2 * (1.0 - p[3])).clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8))
        .collect();
    let f = std::fs::File::create(std::path::Path::new(&dir).join(format!("{name}.png"))).unwrap();
    let mut e = png::Encoder::new(std::io::BufWriter::new(f), W, H);
    e.set_color(png::ColorType::Rgb);
    e.write_header().unwrap().write_image_data(&rgb).unwrap();
}

fn alone(kind: NodeKind) -> Graph {
    let mut g = Graph::default();
    g.working = Colorspace::LINEAR_REC709;
    g.nodes.insert(1, kind);
    g.view = Some(1);
    g
}

/// The inked pixels' bounding box (alpha over a half), or None.
fn ink_box(px: &[[f32; 4]]) -> Option<([u32; 2], [u32; 2])> {
    let mut b: Option<([u32; 2], [u32; 2])> = None;
    for (i, p) in px.iter().enumerate() {
        if p[3] > 0.5 {
            let (x, y) = (i as u32 % W, i as u32 / W);
            let e = b.get_or_insert(([x, y], [x, y]));
            e.0 = [e.0[0].min(x), e.0[1].min(y)];
            e.1 = [e.1[0].max(x), e.1[1].max(y)];
        }
    }
    b
}

fn text(s: &str) -> TextParams {
    TextParams {
        text: s.into(),
        font: TextFont::Sans,
        align: TextAlign::Left,
        position: [20.0, 10.0],
        size: 48.0,
        color: [1.0, 0.0, 0.0, 1.0],
        ..Default::default()
    }
}

#[test]
fn text_lands_in_its_rect_moves_with_its_position_and_takes_its_colour() {
    let Some(ctx) = ctx() else { return };
    let mut c = comp(&ctx);
    let p = text("Hello");
    let g = alone(NodeKind::Text(p.clone()));
    let px = frame(&ctx, &mut c, &g);
    dump("text", &px);
    let (lo, hi) = ink_box(&px).expect("the text drew something");
    let (tl, wh) = p.rect().unwrap();
    assert!(lo[0] as f32 >= tl[0] - 3.0 && hi[0] as f32 <= tl[0] + wh[0] + 3.0, "x {lo:?}..{hi:?} vs {tl:?} {wh:?}");
    assert!(lo[1] as f32 >= tl[1] - 3.0 && hi[1] as f32 <= tl[1] + wh[1] + 3.0, "y {lo:?}..{hi:?} vs {tl:?} {wh:?}");
    assert!(hi[0] - lo[0] > 80, "five letters at 48px are wide: {lo:?}..{hi:?}");
    // Solid ink is the colour, premultiplied; empty pixels are empty.
    let solid: Vec<_> = px.iter().filter(|p| p[3] > 0.99).collect();
    assert!(!solid.is_empty());
    for s in &solid {
        assert!((s[0] - 1.0).abs() < 1e-2 && s[1].abs() < 1e-3 && s[2].abs() < 1e-3, "{s:?}");
    }
    assert_eq!(px[(H - 1) as usize * W as usize + (W - 1) as usize], [0.0; 4]);

    // Moved: the same ink, shifted by exactly the move.
    let topo = g.topology_hash();
    let moved = alone(NodeKind::Text(TextParams { position: [70.0, 30.0], color: [0.0, 1.0, 0.0, 1.0], ..p.clone() }));
    assert_eq!(moved.topology_hash(), topo, "place and colour are uniforms");
    let px2 = frame(&ctx, &mut c, &moved);
    let (lo2, hi2) = ink_box(&px2).unwrap();
    assert_eq!([lo2[0] - lo[0], lo2[1] - lo[1]], [50, 20], "{lo:?} -> {lo2:?}");
    assert_eq!([hi2[0] - hi[0], hi2[1] - hi[1]], [50, 20], "{hi:?} -> {hi2:?}");
    assert!(px2.iter().filter(|p| p[3] > 0.99).all(|p| p[1] > 0.99 && p[0] < 1e-3), "now green");

    // Twice the size: about twice the ink, from the same corner.
    let big = alone(NodeKind::Text(TextParams { size: 96.0, position: [0.0, 0.0], ..p.clone() }));
    let small = alone(NodeKind::Text(TextParams { size: 48.0, position: [0.0, 0.0], ..p }));
    let (_, hs) = ink_box(&frame(&ctx, &mut c, &small)).unwrap();
    let (_, hb) = ink_box(&frame(&ctx, &mut c, &big)).unwrap();
    let ratio = hb[1] as f32 / hs[1] as f32;
    assert!((ratio - 2.0).abs() < 0.15, "height {} -> {} ({ratio})", hs[1], hb[1]);
}

#[test]
fn text_lays_over_its_input() {
    let Some(ctx) = ctx() else { return };
    let mut c = comp(&ctx);
    c.ensure_reader(0, ImageInfo { width: W, height: H });
    let grey: Vec<[f16; 4]> = vec![[0.25, 0.25, 0.25, 1.0].map(f16::from_f32); (W * H) as usize];
    ctx.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &c.reader_texture(0).unwrap().texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&grey),
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(W * 8), rows_per_image: Some(H) },
        wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
    );
    let mut g = Graph::default();
    g.working = Colorspace::LINEAR_REC709;
    g.nodes.insert(0, NodeKind::read(0, ""));
    g.nodes.insert(1, NodeKind::Text(TextParams { color: [0.0, 0.0, 1.0, 0.5], ..text("Hi") }));
    g.edges.insert((1, 0), 0);
    g.view = Some(1);
    let px = frame(&ctx, &mut c, &g);
    // Off the text: the plate. On it: half blue over the plate.
    assert!((px[(H as usize - 1) * W as usize][0] - 0.25).abs() < 1e-3);
    let on = px.iter().copied().fold([0.0f32; 4], |a, p| if p[2] > a[2] { p } else { a });
    assert!((on[2] - (0.5 + 0.125)).abs() < 1e-2 && (on[0] - 0.125).abs() < 1e-2 && (on[3] - 1.0).abs() < 1e-3, "{on:?}");
}

#[test]
fn burnin_prints_what_a_text_of_its_words_prints_and_a_new_frame_is_a_uniform() {
    let Some(ctx) = ctx() else { return };
    let mut c = comp(&ctx);
    let b = BurninParams {
        corner: BurninCorner::Custom,
        position: [16.0, 16.0],
        size: 40.0,
        boxed: false,
        font: TextFont::Mono,
        time: 1001.0,
        ..Default::default()
    };
    let topo = alone(NodeKind::Burnin(b.clone())).topology_hash();
    for f in [1001i64, 1002, 37, -5] {
        let bf = BurninParams { time: f as f32, ..b.clone() };
        let g = alone(NodeKind::Burnin(bf.clone()));
        assert_eq!(g.topology_hash(), topo, "frame {f} recompiled");
        let got = frame(&ctx, &mut c, &g);
        let words = bf.text_at(f);
        let want = frame(
            &ctx,
            &mut c,
            &alone(NodeKind::Text(TextParams {
                text: words.clone(),
                font: TextFont::Mono,
                align: TextAlign::Left,
                position: [16.0, 16.0],
                size: 40.0,
                color: [1.0; 4],
                ..Default::default()
            })),
        );
        let (mut sum, mut worst, mut ink) = (0.0f32, 0.0f32, 0usize);
        for (a, w) in got.iter().zip(&want) {
            let d = (a[3] - w[3]).abs();
            sum += d;
            worst = worst.max(d);
            ink += (w[3] > 0.5) as usize;
        }
        let mean = sum / (W * H) as f32;
        assert!(ink > 200, "{words:?} drew nothing");
        assert!(mean < 0.01 && worst < 0.5, "{words:?}: mean {mean}, worst {worst}");
    }
    // Colour is a uniform too.
    let red = alone(NodeKind::Burnin(BurninParams { color: [1.0, 0.0, 0.0, 1.0], ..b }));
    assert_eq!(red.topology_hash(), topo);
}

#[test]
fn burnin_sits_in_its_corner_on_its_box() {
    let Some(ctx) = ctx() else { return };
    let mut c = comp(&ctx);
    let b = BurninParams { margin: 8.0, size: 24.0, time: 1001.0, ..Default::default() };
    // Format comes from the graph's stamped formats in the app; here the
    // node takes the output's.
    let mut g = alone(NodeKind::Burnin(b.clone()));
    g.formats.insert(1, (W, H));
    let px = frame(&ctx, &mut c, &g);
    dump("burnin", &px);
    // The box: alpha 0.6 in the bottom-left corner, just inside the margin.
    let at = |x: u32, y: u32| px[(y * W + x) as usize];
    assert!((at(9, H - 9)[3] - 0.6).abs() < 1e-2, "{:?}", at(9, H - 9));
    assert_eq!(at(3, H - 3), [0.0; 4], "the margin is clear");
    assert_eq!(at(W - 2, 2), [0.0; 4], "the far corner is clear");
    // White text in the box.
    let (lo, hi) = ink_box(&px.iter().map(|p| if p[0] > 0.9 { *p } else { [0.0; 4] }).collect::<Vec<_>>()).unwrap();
    assert!(lo[0] >= 8 && hi[1] <= H - 8 && lo[1] > H / 2, "{lo:?}..{hi:?}");
    // Top right puts it there instead.
    let mut g = alone(NodeKind::Burnin(BurninParams { corner: BurninCorner::TopRight, ..b }));
    g.formats.insert(1, (W, H));
    let px = frame(&ctx, &mut c, &g);
    assert!((px[(9 * W + W - 10) as usize][3] - 0.6).abs() < 1e-2);
    assert_eq!(px[((H - 9) * W + 9) as usize], [0.0; 4]);
}

/// Alpha at (x, y).
fn alpha(px: &[[f32; 4]], x: u32, y: u32) -> f32 {
    px[(y * W + x) as usize][3]
}

/// An outline with no fill is a RING: ink on the stroke, nothing in the
/// middle of an O; a shadow lands at its offset; a glow lights the air
/// round the glyph that a bare glyph leaves empty.
#[test]
fn outline_is_a_ring_and_shadow_and_glow_land_outside_the_glyph() {
    let Some(ctx) = ctx() else { return };
    let mut c = comp(&ctx);
    let base = TextParams { size: 80.0, position: [40.0, 4.0], color: [1.0, 1.0, 1.0, 1.0], ..text("O") };
    let plain = frame(&ctx, &mut c, &alone(NodeKind::Text(base.clone())));
    let (lo, hi) = ink_box(&plain).unwrap();
    let mid = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2];
    assert_eq!(alpha(&plain, mid[0], mid[1]), 0.0, "an O's counter is empty");

    let ring = TextParams { color: [1.0, 1.0, 1.0, 0.0], outline: 0.06, outline_color: [0.0, 1.0, 0.0, 1.0], ..base.clone() };
    let px = frame(&ctx, &mut c, &alone(NodeKind::Text(ring)));
    dump("outline", &px);
    assert_eq!(alpha(&px, mid[0], mid[1]), 0.0, "the ring leaves the counter");
    let inked: Vec<_> = px.iter().filter(|p| p[3] > 0.99).collect();
    assert!(!inked.is_empty() && inked.iter().all(|p| p[1] > 0.99 && p[0] < 0.01), "a green ring");
    // The ring is wider than the bare glyph's ink box.
    let (rlo, rhi) = ink_box(&px).unwrap();
    assert!(rhi[0] - rlo[0] > hi[0] - lo[0] + 4, "{rlo:?}..{rhi:?} vs {lo:?}..{hi:?}");

    // A hard shadow 30 px right: ink to the right of where the glyph ends.
    let shadow = TextParams {
        shadow: true,
        shadow_offset: [30.0, 0.0],
        shadow_soft: 0.0,
        shadow_color: [0.0, 0.0, 0.0, 1.0],
        ..base.clone()
    };
    let px = frame(&ctx, &mut c, &alone(NodeKind::Text(shadow)));
    dump("shadow", &px);
    let (_, shi) = ink_box(&px).unwrap();
    assert!(shi[0] >= hi[0] + 25, "the shadow reaches past the glyph: {} vs {}", shi[0], hi[0]);
    // The bare glyph's rightmost ink on the middle row, carried 30 px.
    let x = (0..W).rev().find(|x| alpha(&plain, *x, mid[1]) > 0.99).unwrap();
    let right = px[(mid[1] * W + x + 30) as usize];
    assert!(right[3] > 0.9 && right[0] < 0.01, "black shadow: {right:?}");

    // A glow: the air just outside the glyph is lit.
    let glow = TextParams { glow: true, glow_size: 0.2, glow_color: [1.0, 0.0, 0.0, 1.0], ..base };
    let px = frame(&ctx, &mut c, &alone(NodeKind::Text(glow)));
    dump("glow", &px);
    let air = px[(mid[1] * W + hi[0] + 4) as usize];
    assert_eq!(plain[(mid[1] * W + hi[0] + 4) as usize][3], 0.0);
    assert!(air[3] > 0.05 && air[0] > air[1], "glowing red: {air:?}");
}

/// A range selector on the GPU: the first half of the letters lifts, the
/// rest stay — and nothing recompiles.
#[test]
fn an_animator_lifts_the_glyphs_it_selects() {
    let Some(ctx) = ctx() else { return };
    let mut c = comp(&ctx);
    let base = TextParams { size: 40.0, position: [20.0, 40.0], ..text("IIIIIIII") };
    let g0 = alone(NodeKind::Text(base.clone()));
    let px0 = frame(&ctx, &mut c, &g0);
    let mut lifted = base.clone();
    lifted.animators.push(engine::textfx::TextAnimator { end: 50.0, position: [0.0, -30.0], ..Default::default() });
    let g1 = alone(NodeKind::Text(lifted));
    assert_eq!(g1.topology_hash(), g0.topology_hash(), "an animator is not topology");
    let px1 = frame(&ctx, &mut c, &g1);
    dump("animator", &px1);
    let top_of = |px: &[[f32; 4]], x0: u32, x1: u32| {
        (0..H).find(|y| (x0..x1).any(|x| alpha(px, x, *y) > 0.5)).unwrap_or(H)
    };
    let (lo, hi) = ink_box(&px0).unwrap();
    let midx = (lo[0] + hi[0]) / 2;
    let (l0, r0) = (top_of(&px0, lo[0], midx), top_of(&px0, midx, hi[0] + 1));
    let (l1, r1) = (top_of(&px1, lo[0], midx), top_of(&px1, midx, hi[0] + 1));
    assert_eq!(l0, r0);
    assert!((l0 as i32 - l1 as i32 - 30).abs() <= 1, "left half up 30: {l0} -> {l1}");
    assert_eq!(r0, r1, "right half stays");
}

/// A still title records its layer once; a wiggle re-records it each new
/// frame.
#[test]
fn a_still_title_is_drawn_once_and_a_wiggle_redraws() {
    let Some(ctx) = ctx() else { return };
    let mut c = comp(&ctx);
    let mut p = text("still");
    let g = alone(NodeKind::Text(p.clone()));
    frame(&ctx, &mut c, &g);
    assert_eq!(c.text_layer_dirty(1), Some(false));
    c.write_params(&g);
    assert_eq!(c.text_layer_dirty(1), Some(false), "nothing changed, nothing to draw");
    p.animators.push(engine::textfx::TextAnimator { position: [0.0, 10.0], wiggle: 1.0, ..Default::default() });
    p.time = 3.0;
    frame(&ctx, &mut c, &alone(NodeKind::Text(p.clone())));
    p.time = 9.0;
    c.write_params(&alone(NodeKind::Text(p)));
    assert_eq!(c.text_layer_dirty(1), Some(true), "a new frame of a wiggle");
}

/// Text on a path: along a diagonal, the ink runs down-right with it.
#[test]
fn text_follows_its_path() {
    let Some(ctx) = ctx() else { return };
    let mut c = comp(&ctx);
    let mut p = TextParams { size: 20.0, ..text("alongalongalong") };
    p.path = engine::textfx::TextPath {
        on: true,
        perpendicular: true,
        points: vec![
            engine::textfx::PathPoint { p: [10.0, 10.0], ..Default::default() },
            engine::textfx::PathPoint { p: [300.0, 90.0], ..Default::default() },
        ],
        ..Default::default()
    };
    let px = frame(&ctx, &mut c, &alone(NodeKind::Text(p)));
    dump("path", &px);
    // Ink in the left third sits higher than ink in the right third.
    let mean_y = |x0: u32, x1: u32| {
        let (mut s, mut n) = (0.0f32, 0.0f32);
        for y in 0..H {
            for x in x0..x1 {
                if alpha(&px, x, y) > 0.5 {
                    s += y as f32;
                    n += 1.0;
                }
            }
        }
        s / n.max(1.0)
    };
    let (a, b) = (mean_y(20, 60), mean_y(110, 150));
    assert!(b > a + 15.0, "the ink descends with the path: {a} -> {b}");
}

/// `WOBBLE_TEXT_DUMP=<dir> cargo test -p engine --test text text_showcase
/// -- --ignored`: a title with an outline, a shadow, a wave animator and a
/// line on an arc, at 1280x540 — for a look, not a check.
#[test]
#[ignore]
fn text_showcase() {
    let Some(ctx) = ctx() else { return };
    let Ok(dir) = std::env::var("WOBBLE_TEXT_DUMP") else { return };
    let (w, h) = (1280u32, 540u32);
    let mut c = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 1);
    c.ensure_output(w, h);
    let mut title = TextParams {
        text: "WOBBLE TITLES".into(),
        position: [640.0, 300.0],
        size: 120.0,
        color: [1.0, 0.85, 0.3, 1.0],
        outline: 0.04,
        outline_color: [0.05, 0.02, 0.1, 1.0],
        shadow: true,
        shadow_offset: [8.0, 10.0],
        shadow_soft: 0.08,
        tracking: 60.0,
        valign: engine::textfx::VAlign::Baseline,
        ..Default::default()
    };
    title.animators.push(engine::textfx::TextAnimator {
        unit: engine::textfx::TextUnit::Letters,
        shape: engine::textfx::RangeShape::Smooth,
        end: 40.0,
        offset: 20.0,
        position: [0.0, -50.0],
        scale: [1.35, 1.35],
        rotation: -8.0,
        tint: [0.3, 0.9, 1.0, 1.0],
        ..Default::default()
    });
    let r = 420.0;
    let k = 0.5523 * r;
    let arc = TextParams {
        text: "per-letter animation on a path".into(),
        size: 34.0,
        color: [1.0; 4],
        glow: true,
        glow_size: 0.15,
        glow_color: [0.4, 0.6, 1.0, 0.9],
        path: engine::textfx::TextPath {
            on: true,
            perpendicular: true,
            points: vec![
                engine::textfx::PathPoint { p: [640.0 - r, 520.0], h_in: [0.0, k], h_out: [0.0, -k] },
                engine::textfx::PathPoint { p: [640.0, 520.0 - r], h_in: [-k, 0.0], h_out: [k, 0.0] },
                engine::textfx::PathPoint { p: [640.0 + r, 520.0], h_in: [0.0, -k], h_out: [0.0, k] },
            ],
            ..Default::default()
        },
        ..Default::default()
    };
    let mut g = Graph::default();
    g.working = Colorspace::LINEAR_REC709;
    g.nodes.insert(1, NodeKind::Text(arc));
    g.nodes.insert(2, NodeKind::Text(title));
    g.edges.insert((2, 0), 1);
    g.view = Some(2);
    assert!(c.set_graph(&g));
    let timer = GpuTimer::new(&ctx);
    c.write_params(&g);
    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    c.encode_comp(&mut encoder, &timer);
    let padded = (w * 8).div_ceil(256) * 256;
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (padded * h) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: c.output_texture().unwrap(),
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
    ctx.queue.submit([encoder.finish()]);
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    let data = slice.get_mapped_range().expect("mapped");
    let mut rgb = Vec::new();
    for y in 0..h as usize {
        let row: &[f16] = bytemuck::cast_slice(&data[y * padded as usize..][..(w * 8) as usize]);
        for x in 0..w as usize {
            let a = row[x * 4 + 3].to_f32();
            let bg = 0.04 + 0.05 * (y as f32 / h as f32);
            for k in 0..3 {
                let v = row[x * 4 + k].to_f32() + bg * (1.0 - a);
                rgb.push((v.clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8);
            }
        }
    }
    let f = std::fs::File::create(std::path::Path::new(&dir).join("showcase.png")).unwrap();
    let mut e = png::Encoder::new(std::io::BufWriter::new(f), w, h);
    e.set_color(png::ColorType::Rgb);
    e.write_header().unwrap().write_image_data(&rgb).unwrap();
}
