//! Convolve with a KERNEL PICTURE (port 2), the FFT path: a single hot pixel
//! comes out as the kernel stamped where it was, normalise keeps the level,
//! the half-res grid still lands it, and the modulate port scales the mix.
//! Self-skips without a GPU (CI).

use engine::compositor::Compositor;
use engine::decode::ImageInfo;
use engine::graph::{ConvolveParams, ConvolveRes, Graph, NodeKind};
use engine::timing::GpuTimer;
use half::f16;

const W: u32 = 256;
const H: u32 = 192;

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP convolve kernel test: no adapter ({e})");
            None
        }
    }
}

pub fn upload(ctx: &engine::GpuContext, comp: &mut Compositor, slot: usize, w: u32, h: u32, f: impl Fn(u32, u32) -> [f32; 4]) {
    comp.ensure_reader(slot, ImageInfo { width: w, height: h });
    let mut px: Vec<[f16; 4]> = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            px.push(f(x, y).map(f16::from_f32));
        }
    }
    ctx.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &comp.reader_texture(slot).unwrap().texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&px),
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 8), rows_per_image: Some(h) },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
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

/// A canvas-sized kernel centred on (W/2, H/2): a horizontal bar (rgb 1)
/// through the centre, 21 px long, and a red-only dot 8 px above it.
fn kernel(x: u32, y: u32) -> [f32; 4] {
    let (cx, cy) = (W / 2, H / 2);
    if y == cy && (cx - 10..=cx + 10).contains(&x) {
        [1.0, 1.0, 1.0, 1.0]
    } else if y == cy - 8 && x == cx {
        [1.0, 0.0, 0.0, 1.0]
    } else {
        [0.0, 0.0, 0.0, 1.0]
    }
}

const PX: u32 = 100;
const PY: u32 = 90;

fn graph(p: ConvolveParams, modulate: bool) -> Graph {
    let mut g = Graph::default();
    g.nodes.insert(1, NodeKind::read(0, ""));
    g.nodes.insert(2, NodeKind::read(1, ""));
    g.nodes.insert(3, NodeKind::Convolve(p));
    g.edges.insert((3, 0), 1);
    g.edges.insert((3, 2), 2);
    if modulate {
        g.nodes.insert(4, NodeKind::read(2, ""));
        g.edges.insert((3, 3), 4);
    }
    g.view = Some(3);
    g
}

fn setup() -> Option<(engine::GpuContext, Compositor)> {
    let ctx = ctx()?;
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    // One hot pixel, value 2, alpha 1 everywhere.
    upload(&ctx, &mut comp, 0, W, H, |x, y| {
        let v = if (x, y) == (PX, PY) { 2.0 } else { 0.0 };
        [v, v, v, 1.0]
    });
    upload(&ctx, &mut comp, 1, W, H, kernel);
    // Modulate: 1 on the left half, 0 on the right.
    upload(&ctx, &mut comp, 2, W, H, |x, _| {
        let v = if x < PX { 1.0 } else { 0.0 };
        [v, v, v, 1.0]
    });
    Some((ctx, comp))
}

fn at(o: &[[f32; 4]], x: u32, y: u32) -> [f32; 4] {
    o[(y * W + x) as usize]
}

#[test]
fn a_hot_pixel_comes_out_as_the_kernel() {
    let Some((ctx, mut comp)) = setup() else { return };
    let p = ConvolveParams { normalize: false, res: ConvolveRes::Full, ..Default::default() };
    let o = render(&ctx, &mut comp, &graph(p, false));
    // The bar: 2 x 1 along it, nothing past its ends or off its row.
    for dx in [-10i32, -5, 0, 5, 10] {
        let v = at(&o, (PX as i32 + dx) as u32, PY);
        assert!((v[0] - 2.0).abs() < 0.02 && (v[2] - 2.0).abs() < 0.02, "bar at {dx}: {v:?}");
    }
    assert!(at(&o, PX + 12, PY)[0].abs() < 0.02, "past the bar: {:?}", at(&o, PX + 12, PY));
    assert!(at(&o, PX, PY + 2)[0].abs() < 0.02, "off the bar: {:?}", at(&o, PX, PY + 2));
    // The red dot: above the pixel, red only.
    let d = at(&o, PX, PY - 8);
    assert!((d[0] - 2.0).abs() < 0.02 && d[1].abs() < 0.02, "dot: {d:?}");
    // Alpha off: the plate's alpha passes.
    assert!((at(&o, 10, 10)[3] - 1.0).abs() < 1e-3);
    // Far away, nothing (no wrap round the grid).
    assert!(at(&o, 10, 10)[0].abs() < 0.02, "far: {:?}", at(&o, 10, 10));

    // Normalised, the bar's 21 white px + the red dot sum to ~21.3: the
    // total light is the hot pixel's.
    let n = ConvolveParams { normalize: true, res: ConvolveRes::Full, ..Default::default() };
    let o = render(&ctx, &mut comp, &graph(n, false));
    let sum: f32 = o.iter().map(|p| p[1]).sum();
    assert!((sum - 2.0).abs() < 0.05, "normalised green sum {sum}");
}

#[test]
fn half_res_still_lands_the_kernel() {
    let Some((ctx, mut comp)) = setup() else { return };
    let p = ConvolveParams { normalize: true, res: ConvolveRes::Half, ..Default::default() };
    let o = render(&ctx, &mut comp, &graph(p, false));
    let sum: f32 = o.iter().map(|p| p[1]).sum();
    assert!((sum - 2.0).abs() < 0.1, "half-res green sum {sum}");
    // The light sits on the bar's row, along its length.
    assert!(at(&o, PX + 6, PY)[1] > 4.0 * at(&o, PX + 6, PY + 6)[1], "{:?}", at(&o, PX + 6, PY));
}

#[test]
fn modulate_scales_the_mix() {
    let Some((ctx, mut comp)) = setup() else { return };
    let p = ConvolveParams { normalize: false, res: ConvolveRes::Full, ..Default::default() };
    let o = render(&ctx, &mut comp, &graph(p.clone(), true));
    // Left of the pixel the bar is there; right of it the matte is 0 and
    // the plate (black) shows.
    assert!((at(&o, PX - 5, PY)[0] - 2.0).abs() < 0.02, "left: {:?}", at(&o, PX - 5, PY));
    assert!(at(&o, PX + 5, PY)[0].abs() < 0.02, "right: {:?}", at(&o, PX + 5, PY));
}

#[test]
fn a_small_kernel_is_centred_on_its_own_format() {
    let Some((ctx, mut comp)) = setup() else { return };
    // A 33x33 kernel picture: sits at the canvas's top-left like any plate,
    // its centre at (16, 16). A bar 21 px long through it.
    upload(&ctx, &mut comp, 1, 33, 33, |x, y| {
        let v = if y == 16 && (6..=26).contains(&x) { 1.0 } else { 0.0 };
        [v, v, v, 1.0]
    });
    let mut g = graph(ConvolveParams { normalize: false, res: ConvolveRes::Full, ..Default::default() }, false);
    g.formats.insert(2, (33, 33));
    let o = render(&ctx, &mut comp, &g);
    for dx in [-10i32, 0, 10] {
        let v = at(&o, (PX as i32 + dx) as u32, PY);
        assert!((v[0] - 2.0).abs() < 0.02, "bar at {dx}: {v:?}");
    }
    assert!(at(&o, PX, PY + 1)[0].abs() < 0.02);
}

#[test]
fn a_glint_kernel_node_makes_a_star_through_convolve() {
    let Some((ctx, mut comp)) = setup() else { return };
    use engine::graph::GlintKernelParams;
    let mut g = Graph::default();
    g.nodes.insert(1, NodeKind::read(0, ""));
    g.nodes.insert(2, NodeKind::GlintKernel(GlintKernelParams { rays: 4.0, length: 0.8, core: 0.0, ..Default::default() }));
    g.nodes.insert(3, NodeKind::Convolve(ConvolveParams { normalize: false, res: ConvolveRes::Full, ..Default::default() }));
    g.edges.insert((3, 0), 1);
    g.edges.insert((3, 2), 2);
    // A 128x128 kernel at the canvas's top-left: rays 0.8 * 64 = 51 px.
    g.formats.insert(2, (128, 128));
    g.view = Some(3);
    let o = render(&ctx, &mut comp, &g);
    for (dx, dy) in [(30i32, 0i32), (-30, 0), (0, 30), (0, -30)] {
        let v = at(&o, (PX as i32 + dx) as u32, (PY as i32 + dy) as u32)[1];
        assert!(v > 0.2, "ray at ({dx},{dy}) got {v}");
    }
    assert!(at(&o, PX + 20, PY + 20)[1] < 0.01, "diagonal got {:?}", at(&o, PX + 20, PY + 20));
    assert!(at(&o, PX + 60, PY)[1] < 0.01, "past the tip got {:?}", at(&o, PX + 60, PY));
}

#[test]
fn a_bokeh_kernel_node_shapes_an_accurate_defocus() {
    let Some((ctx, mut comp)) = setup() else { return };
    use engine::graph::{BokehKernelParams, DefocusMode, DefocusParams};
    let run = |comp: &mut Compositor, k: Option<BokehKernelParams>, mode: DefocusMode| {
        let mut g = Graph::default();
        g.nodes.insert(1, NodeKind::read(0, ""));
        g.nodes.insert(3, NodeKind::Defocus(DefocusParams { radius: 20.0, mode, samples: 512, ..Default::default() }));
        g.edges.insert((3, 0), 1);
        if let Some(k) = k {
            g.nodes.insert(2, NodeKind::BokehKernel(k));
            g.edges.insert((3, 3), 2);
            g.formats.insert(2, (64, 64));
        }
        g.view = Some(3);
        render(&ctx, comp, &g)
    };
    // A strong cat's-eye: one side of the bokeh is cut away, the other kept.
    let cat = BokehKernelParams { blades: 0.0, catseye: 0.6, ..Default::default() };
    let o = run(&mut comp, Some(cat), DefocusMode::Accurate);
    let (l, r) = (at(&o, PX - 12, PY)[1], at(&o, PX + 12, PY)[1]);
    assert!(l.max(r) > 1e-3, "no bokeh at all: {l} {r}");
    assert!(l.min(r) < 0.1 * l.max(r), "cat's-eye not one-sided: {l} vs {r}");
    // Round, it is symmetric.
    let o = run(&mut comp, Some(BokehKernelParams { blades: 0.0, ..Default::default() }), DefocusMode::Accurate);
    let (l, r) = (at(&o, PX - 12, PY)[1], at(&o, PX + 12, PY)[1]);
    assert!((l - r).abs() < 0.35 * l.max(r) && l > 1e-3, "round bokeh lopsided: {l} vs {r}");
    // Fast ignores the kernel (and does not render it): a plain disc.
    let fast_k = run(&mut comp, Some(cat), DefocusMode::Fast);
    let fast = run(&mut comp, None, DefocusMode::Fast);
    assert!(fast_k.iter().zip(&fast).all(|(a, b)| (a[1] - b[1]).abs() < 1e-4), "fast changed with a kernel");
}

/// Not a check: draws the two kernels at work to `target/kernel-glint.png`
/// and `target/kernel-bokeh.png` (tonemapped x/(1+x)), to judge by eye.
#[test]
#[ignore]
fn kernel_pictures() {
    let Some(ctx) = ctx() else { return };
    use engine::graph::{BokehKernelParams, DefocusMode, DefocusParams, GlintKernelParams};
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    let dots = [(60u32, 60u32, 8.0f32), (180, 50, 3.0), (120, 130, 20.0), (40, 160, 5.0), (210, 150, 12.0)];
    upload(&ctx, &mut comp, 0, W, H, |x, y| {
        let v = dots.iter().find(|d| d.0 == x && d.1 == y).map(|d| d.2 * 40.0).unwrap_or(0.0);
        [v, v * 0.9, v * 0.8, 1.0]
    });
    let save = |o: &[[f32; 4]], name: &str| {
        let px: Vec<u8> = o
            .iter()
            .flat_map(|p| [p[0], p[1], p[2]].map(|v| ((v / (1.0 + v)).clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8))
            .collect();
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("../target/{name}.png"));
        let f = std::fs::File::create(&path).unwrap();
        let mut e = png::Encoder::new(std::io::BufWriter::new(f), W, H);
        e.set_color(png::ColorType::Rgb);
        e.set_depth(png::BitDepth::Eight);
        e.write_header().unwrap().write_image_data(&px).unwrap();
        eprintln!("wrote {}", path.display());
    };
    let mut g = Graph::default();
    g.nodes.insert(1, NodeKind::read(0, ""));
    g.nodes.insert(2, NodeKind::GlintKernel(GlintKernelParams { rays: 6.0, alternate: 0.5, chroma: 0.6, intensity: 0.02, ..Default::default() }));
    g.nodes.insert(3, NodeKind::Convolve(ConvolveParams { normalize: false, res: ConvolveRes::Full, ..Default::default() }));
    g.edges.insert((3, 0), 1);
    g.edges.insert((3, 2), 2);
    g.formats.insert(2, (160, 160));
    g.view = Some(3);
    save(&render(&ctx, &mut comp, &g), "kernel-glint");
    let mut g = Graph::default();
    g.nodes.insert(1, NodeKind::read(0, ""));
    g.nodes.insert(2, NodeKind::BokehKernel(BokehKernelParams { blades: 6.0, roundness: 0.2, ring: 0.5, fringe: 0.5, catseye: 0.25, catseye_angle: 30.0, ..Default::default() }));
    g.nodes.insert(3, NodeKind::Defocus(DefocusParams { radius: 22.0, mode: DefocusMode::Accurate, samples: 1024, ..Default::default() }));
    g.edges.insert((3, 0), 1);
    g.edges.insert((3, 3), 2);
    g.formats.insert(2, (128, 128));
    g.view = Some(3);
    save(&render(&ctx, &mut comp, &g), "kernel-bokeh");
}
