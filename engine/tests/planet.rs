//! Planet on real pixels: the disc is where it says and nothing outside
//! it, the latlong is the SAME surface the disc shows (pixel for pixel,
//! through `env::latlong_px`), it has no seam, and a pole looks like an
//! equator — the pinching this node exists to avoid. Writes
//! `target/planet-*.png` to look at. Self-skips without a GPU (CI).

use engine::compositor::Compositor;
use engine::graph::{Graph, NodeKind, PlanetMode, PlanetNoise, PlanetParams};
use engine::timing::GpuTimer;
use half::f16;

const W: u32 = 512;
const H: u32 = 256;

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP planet test: no adapter ({e})");
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

fn draw(ctx: &engine::GpuContext, comp: &mut Compositor, p: PlanetParams) -> Vec<[f32; 4]> {
    let mut g = Graph::default_poc();
    g.nodes.insert(7, NodeKind::Planet(p));
    g.formats.insert(7, (W, H));
    g.view = Some(7);
    render_wh(ctx, comp, &g, W, H)
}

fn base() -> PlanetParams {
    PlanetParams {
        size: (W, H),
        center: [256.0, 128.0],
        radius: 110.0,
        rotate: [0.0; 3],
        spin: 0.0,
        ..Default::default()
    }
}

fn at(img: &[[f32; 4]], x: u32, y: u32) -> [f32; 4] {
    img[(y * W + x) as usize]
}

fn diff(a: [f32; 4], b: [f32; 4]) -> f32 {
    (0..3).map(|i| (a[i] - b[i]).abs()).fold(0.0, f32::max)
}

/// Mean colour step between horizontal neighbours inside a box: how much
/// detail the noise shows there.
fn detail(img: &[[f32; 4]], x0: u32, y0: u32, n: u32) -> f32 {
    let mut s = 0.0;
    for y in y0..y0 + n {
        for x in x0..x0 + n {
            s += diff(at(img, x, y), at(img, x + 1, y));
        }
    }
    s / (n * n) as f32
}

fn save(img: &[[f32; 4]], name: &str) {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../target");
    let _ = std::fs::create_dir_all(&dir);
    let path = dir.join(format!("planet-{name}.png"));
    let enc = |v: f32| {
        let v = v.clamp(0.0, 1.0);
        let s = if v <= 0.0031308 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 };
        (s * 255.0 + 0.5) as u8
    };
    let mut bytes = Vec::with_capacity((W * H * 3) as usize);
    for c in img {
        // Over black: the output is premultiplied.
        bytes.extend([enc(c[0]), enc(c[1]), enc(c[2])]);
    }
    let file = std::fs::File::create(&path).unwrap();
    let mut e = png::Encoder::new(std::io::BufWriter::new(file), W, H);
    e.set_color(png::ColorType::Rgb);
    e.set_depth(png::BitDepth::Eight);
    e.write_header().unwrap().write_image_data(&bytes).unwrap();
    eprintln!("wrote {}", path.display());
}

#[test]
fn the_disc_is_where_it_says() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    let img = draw(&ctx, &mut comp, base());
    save(&img, "star");
    assert!((at(&img, 256, 128)[3] - 1.0).abs() < 1e-3, "inside is solid");
    assert!(at(&img, 256 + 115, 128)[3] < 1e-3, "outside is empty");
    assert!(at(&img, 10, 10) == [0.0; 4], "a corner is clear");
    // Limb darkening: the rim is darker than the middle, on average.
    let lum = |c: [f32; 4]| c[0] + c[1] + c[2];
    let mid: f32 = (0..20).map(|i| lum(at(&img, 246 + i, 128))).sum();
    let rim: f32 = (0..20).map(|i| lum(at(&img, 256 + 104, 118 + i))).sum();
    assert!(rim < mid * 0.8, "limb-darkened: rim {rim} vs middle {mid}");
}

/// The latlong IS the disc's surface: a disc pixel, through the CPU's
/// `surface` and Env's latlong mapping, lands on a latlong pixel of the
/// same colour.
#[test]
fn the_latlong_is_the_surface_the_disc_shows() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    let flat = PlanetParams { limb: 0.0, light: 0.0, warp: 0.0, rotate: [25.0, 40.0, 0.0], ..base() };
    let disc = draw(&ctx, &mut comp, flat.clone());
    let ll = draw(&ctx, &mut comp, PlanetParams { mode: PlanetMode::Latlong, ..flat.clone() });
    save(&ll, "latlong");
    let mut worst = 0.0f32;
    let mut n = 0;
    for (x, y) in [(256, 128), (200, 90), (300, 170), (230, 60), (330, 120), (180, 150)] {
        let p = [x as f32 + 0.5, y as f32 + 0.5];
        let s = flat.surface(p, [W as f32, H as f32]).expect("on the disc");
        // `surface` is in planet space, and the latlong turns by the same
        // pose: the latlong pixel is the one whose view direction the pose
        // carries onto `s`.
        let m = flat.pose();
        let d = [0, 1, 2].map(|r| m[r][0] * s[0] + m[r][1] * s[1] + m[r][2] * s[2]);
        let q = engine::nodes::env::latlong_px(d, [W as f32, H as f32]);
        let c = at(&ll, (q[0] as u32).min(W - 1), (q[1] as u32).min(H - 1));
        worst = worst.max(diff(c, at(&disc, x, y)));
        n += 1;
    }
    assert_eq!(n, 6);
    // Nearest-pixel lookups on a field with detail: close, not exact.
    assert!(worst < 0.12, "latlong and disc disagree by {worst}");
}

#[test]
fn the_latlong_has_no_seam() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    let ll = draw(&ctx, &mut comp, PlanetParams { mode: PlanetMode::Latlong, ..base() });
    let (mut seam, mut inner) = (0.0, 0.0);
    for y in 20..H - 20 {
        seam += diff(at(&ll, W - 1, y), at(&ll, 0, y));
        inner += diff(at(&ll, W / 2, y), at(&ll, W / 2 + 1, y));
    }
    assert!(seam < inner * 2.0 + 1e-3, "the wrap is no rougher than any other column step: {seam} vs {inner}");
}

/// The point of the node: turn a pole to face the viewer and it shows the
/// same amount of detail as the equator did. A wrapped 2D latlong would
/// pinch to a starburst there.
#[test]
fn a_pole_looks_like_an_equator() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    let flat = PlanetParams { limb: 0.0, warp: 0.0, ..base() };
    let eq = draw(&ctx, &mut comp, flat.clone());
    let pole = draw(&ctx, &mut comp, PlanetParams { rotate: [90.0, 0.0, 0.0], ..flat });
    save(&pole, "pole");
    let (a, b) = (detail(&eq, 236, 108, 40), detail(&pole, 236, 108, 40));
    assert!(a > 1e-3 && b > 1e-3, "there is detail: {a} {b}");
    let r = a.max(b) / a.min(b);
    assert!(r < 1.6, "pole detail {b} vs equator {a}");
}

/// A lit, banded planet, for the eye.
#[test]
fn a_lit_banded_planet() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    let p = PlanetParams {
        rotate: [18.0, 0.0, -12.0],
        stretch: 9.0,
        warp: 0.6,
        noise: PlanetNoise::Fbm,
        low: [0.45, 0.28, 0.16],
        high: [0.95, 0.85, 0.7],
        limb: 0.2,
        light: 1.0,
        ..base()
    };
    let img = draw(&ctx, &mut comp, p);
    save(&img, "giant");
    // Lit from the left: the left limb is brighter than the right.
    let lum = |c: [f32; 4]| c[0] + c[1] + c[2];
    assert!(lum(at(&img, 256 - 80, 128)) > lum(at(&img, 256 + 80, 128)) * 2.0);
}
