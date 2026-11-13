//! Starfield on real pixels: there are stars, the distributions differ
//! the way they say, a camera turn moves every star to where the CPU's
//! projection says it goes, and a wired camera draws exactly what the same
//! view typed into the node draws. Self-skips without a GPU (CI).

use engine::compositor::Compositor;
use engine::graph::{Graph, NodeKind, StarDist, StarfieldParams};
use engine::scene3d::{CameraParams, Xform};
use engine::timing::GpuTimer;
use half::f16;

const W: u32 = 256;
const H: u32 = 144;

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP starfield test: no adapter ({e})");
            None
        }
    }
}

fn render_wh(ctx: &engine::GpuContext, comp: &mut Compositor, graph: &Graph, w: u32, h: u32) -> Vec<[f32; 4]> {
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
    let (tx, rx) = std::sync::mpsc::channel();
    staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().unwrap().unwrap();
    let view = staging.slice(..).get_mapped_range().expect("a mapped buffer");
    let mut out = Vec::with_capacity((w * h) as usize);
    for y in 0..h as usize {
        let row: &[f16] = bytemuck::cast_slice(
            &view[y * padded as usize..y * padded as usize + (w * 8) as usize],
        );
        for c in row.chunks_exact(4) {
            out.push([c[0].to_f32(), c[1].to_f32(), c[2].to_f32(), c[3].to_f32()]);
        }
    }
    out
}

fn render(ctx: &engine::GpuContext, comp: &mut Compositor, graph: &Graph) -> Vec<[f32; 4]> {
    render_wh(ctx, comp, graph, W, H)
}

fn base() -> StarfieldParams {
    StarfieldParams {
        size: (W, H),
        star_size: 1.0,
        opaque: false,
        glow: 0.0,
        // Two layers: stars far enough apart to count.
        layers: 2.0,
        ..Default::default()
    }
}

fn draw(ctx: &engine::GpuContext, comp: &mut Compositor, p: StarfieldParams) -> Vec<[f32; 4]> {
    let mut g = Graph::default_poc();
    g.nodes.insert(7, NodeKind::Starfield(p));
    g.formats.insert(7, (W, H));
    g.view = Some(7);
    render(ctx, comp, &g)
}

fn lum(c: [f32; 4]) -> f32 {
    c[0] + c[1] + c[2]
}

/// Local maxima brighter than `floor`, with their sub-pixel centroids.
fn stars(img: &[[f32; 4]], floor: f32) -> Vec<([f32; 2], f32)> {
    let at = |x: i32, y: i32| -> f32 {
        if x < 0 || y < 0 || x >= W as i32 || y >= H as i32 { 0.0 } else { lum(img[(y as u32 * W + x as u32) as usize]) }
    };
    let mut out = Vec::new();
    for y in 2..H as i32 - 2 {
        for x in 2..W as i32 - 2 {
            let v = at(x, y);
            if v < floor {
                continue;
            }
            let mut peak = true;
            let (mut sx, mut sy, mut sw) = (0.0, 0.0, 0.0);
            for j in -1..=1 {
                for i in -1..=1 {
                    let u = at(x + i, y + j);
                    if (i, j) != (0, 0) && u >= v {
                        peak = false;
                    }
                    sx += u * (x + i) as f32;
                    sy += u * (y + j) as f32;
                    sw += u;
                }
            }
            if peak {
                out.push(([sx / sw + 0.5, sy / sw + 0.5], v));
            }
        }
    }
    out
}

#[test]
fn there_are_stars_and_the_distributions_differ() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    let count = |comp: &mut Compositor, p: StarfieldParams| stars(&draw(&ctx, comp, p), 0.05).len();
    let random = count(&mut comp, StarfieldParams { density: 0.2, ..base() });
    let even = count(&mut comp, StarfieldParams { density: 0.2, dist: StarDist::Even, ..base() });
    assert!(random > 5, "a random sky has stars: {random}");
    assert!(even > random * 2, "even fills every cell, random a fifth of them: {even} vs {random}");
    // A band through the middle of the view against one turned away.
    let band = |tilt: f32| StarfieldParams {
        dist: StarDist::Band,
        band_tilt: tilt,
        band_floor: 0.0,
        band_width: 8.0,
        ..base()
    };
    // At tilt 0 the band's normal is +Y, so the band is the horizon and
    // runs across the default view (down -Z); at tilt 90 the normal is the
    // view axis and the band rings the edge of the sky, far outside it.
    let across = count(&mut comp, band(0.0));
    let away = count(&mut comp, band(90.0));
    assert!(across > away * 3, "the band runs across the view at tilt 0: {across} vs {away}");
    // Opaque: alpha 1 everywhere.
    let o = draw(&ctx, &mut comp, StarfieldParams { opaque: true, ..base() });
    assert!(o.iter().all(|c| (c[3] - 1.0).abs() < 1e-3));
}

#[test]
fn a_camera_turn_moves_every_star_where_the_projection_says() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    let a = StarfieldParams { rotate: [4.0, -10.0, 2.0], focal: 30.0, ..base() };
    let b = StarfieldParams { rotate: [6.0, -7.5, 3.0], ..a.clone() };
    let sa = stars(&draw(&ctx, &mut comp, a.clone()), 0.3);
    let sb = stars(&draw(&ctx, &mut comp, b.clone()), 0.3);
    assert!(sa.len() > 10, "{}", sa.len());
    let mut matched = 0;
    let mut tried = 0;
    for (p, _) in &sa {
        let d = a.ray(*p, W as f32, H as f32);
        let Some(q) = b.project(d, W as f32, H as f32) else { continue };
        if q[0] < 4.0 || q[1] < 4.0 || q[0] > W as f32 - 4.0 || q[1] > H as f32 - 4.0 {
            continue;
        }
        tried += 1;
        if sb.iter().any(|(s, _)| (s[0] - q[0]).hypot(s[1] - q[1]) < 0.75) {
            matched += 1;
        }
    }
    assert!(tried > 5 && matched * 10 >= tried * 9, "{matched} of {tried} stars where predicted");
}

#[test]
fn a_wired_camera_draws_what_the_typed_view_draws() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    let typed = StarfieldParams { rotate: [10.0, 25.0, -5.0], focal: 50.0, aperture: 36.0, ..base() };
    let want = draw(&ctx, &mut comp, typed);
    let mut g = Graph::default_poc();
    let cam = CameraParams {
        xform: Xform { translate: [3.0, 4.0, 5.0], rotate: [10.0, 25.0, -5.0], ..Default::default() },
        focal: 50.0,
        ..Default::default()
    };
    g.nodes.insert(6, NodeKind::Camera3D(cam));
    g.nodes.insert(7, NodeKind::Starfield(base()));
    g.formats.insert(7, (W, H));
    g.connect_out(7, 0, 6, 0);
    g.view = Some(7);
    engine::nodes::starfield::lower(&mut g);
    let got = render(&ctx, &mut comp, &g);
    let worst = want.iter().zip(&got).map(|(a, b)| (0..4).map(|k| (a[k] - b[k]).abs()).fold(0.0, f32::max)).fold(0.0, f32::max);
    assert!(worst < 1e-3, "worst {worst}");
    assert!(stars(&got, 0.3).len() > 5);
}

/// Looks, not checks: three skies at 1280x720 through a plain sRGB encode
/// to `target/starfield-*.png`.
#[test]
#[ignore]
fn starfield_render() {
    let Some(ctx) = ctx() else { return };
    let (w, h) = (1280u32, 720u32);
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(w, h);
    for (name, dist) in [("random", StarDist::Random), ("even", StarDist::Even), ("clustered", StarDist::Clustered), ("band", StarDist::Band)] {
        let p = StarfieldParams { size: (w, h), dist, band_tilt: 8.0, glow: 0.15, ..Default::default() };
        let mut g = Graph::default_poc();
        g.nodes.insert(7, NodeKind::Starfield(p));
        g.formats.insert(7, (w, h));
        g.view = Some(7);
        let img = render_wh(&ctx, &mut comp, &g, w, h);
        let enc = |v: f32| {
            let v = v.clamp(0.0, 1.0);
            let s = if v <= 0.0031308 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 };
            (s * 255.0).round() as u8
        };
        let bytes: Vec<u8> = img.iter().flat_map(|c| [enc(c[0]), enc(c[1]), enc(c[2])]).collect();
        let path = format!("{}/../target/starfield-{name}.png", env!("CARGO_MANIFEST_DIR"));
        let f = std::fs::File::create(&path).unwrap();
        let mut e = png::Encoder::new(std::io::BufWriter::new(f), w, h);
        e.set_color(png::ColorType::Rgb);
        e.set_depth(png::BitDepth::Eight);
        e.write_header().unwrap().write_image_data(&bytes).unwrap();
        println!("wrote {path}");
    }
}
