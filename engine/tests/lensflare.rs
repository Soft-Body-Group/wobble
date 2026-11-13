//! Lens Flare on real pixels: the flare peaks at the light, a ghost sits
//! where the CPU's hash mirror puts it on the axis through the centre, the
//! flare-only mode is the flare alone, occlusion follows the input's
//! brightness AT the light, and intensity 0 is the plate exactly.
//! Self-skips without a GPU (CI).

use engine::compositor::Compositor;
use engine::graph::{BlankParams, GlowMode, Graph, LensFlareParams, NodeKind};
use engine::timing::GpuTimer;
use half::f16;

const W: u32 = 256;
const H: u32 = 144;

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP lensflare test: no adapter ({e})");
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

fn quiet() -> LensFlareParams {
    LensFlareParams {
        glow: 0.0,
        burst: 0.0,
        streak: 0.0,
        halo: 0.0,
        ghosts: 0.0,
        fade: 0.0,
        ..Default::default()
    }
}

/// A Blank of `plate` -> Lens Flare -> view.
fn draw(ctx: &engine::GpuContext, comp: &mut Compositor, p: LensFlareParams, plate: [f32; 4]) -> Vec<[f32; 4]> {
    let mut g = Graph::default_poc();
    g.nodes.insert(3, NodeKind::Blank(BlankParams { size: (W, H), color: plate, ..Default::default() }));
    g.formats.insert(3, (W, H));
    g.nodes.insert(7, NodeKind::LensFlare(p));
    g.formats.insert(7, (W, H));
    g.edges.insert((7, 0), 3);
    g.view = Some(7);
    render(ctx, comp, &g)
}

fn lum(c: [f32; 4]) -> f32 {
    c[0] + c[1] + c[2]
}

fn at(img: &[[f32; 4]], x: f32, y: f32) -> [f32; 4] {
    let xi = (x.floor() as i32).clamp(0, W as i32 - 1) as u32;
    let yi = (y.floor() as i32).clamp(0, H as i32 - 1) as u32;
    img[(yi * W + xi) as usize]
}

#[test]
fn the_flare_peaks_at_the_light_and_zero_is_the_plate() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    let plate = [0.1, 0.2, 0.3, 1.0];
    let p = LensFlareParams { center: [0.25, 0.4], glow: 1.0, mode: GlowMode::Only, ..quiet() };
    let img = draw(&ctx, &mut comp, p, plate);
    let (lx, ly) = (0.25 * W as f32, 0.4 * H as f32);
    let (mut best, mut bi) = (0.0, 0usize);
    for (i, c) in img.iter().enumerate() {
        if lum(*c) > best {
            best = lum(*c);
            bi = i;
        }
    }
    let (bx, by) = ((bi as u32 % W) as f32 + 0.5, (bi as u32 / W) as f32 + 0.5);
    assert!((bx - lx).abs() <= 1.0 && (by - ly).abs() <= 1.0, "peak at {bx},{by}, light at {lx},{ly}");
    // Flare only: the plate is not in it; alpha is the plate's.
    let far = at(&img, W as f32 - 2.0, H as f32 - 2.0);
    assert!(lum(far) < 0.1, "flare only still carries the plate: {far:?}");
    assert_eq!(far[3], 1.0);
    // Intensity 0, add: the plate exactly.
    let img = draw(&ctx, &mut comp, LensFlareParams { intensity: 0.0, ..Default::default() }, plate);
    assert!(img.iter().all(|c| (0..4).all(|k| (c[k] - plate[k]).abs() < 1e-3)));
}

#[test]
fn a_ghost_sits_where_the_hash_says() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    let base = LensFlareParams {
        center: [0.2, 0.3],
        ghosts: 1.0,
        ghost_count: 1.0,
        ghost_color: 0.0,
        ghost_chroma: 0.0,
        ghost_blades: 0.0,
        ghost_size: 0.04,
        mode: GlowMode::Only,
        ..quiet()
    };
    // A seed whose single ghost lands well inside the frame, away from the light.
    let p = (0..64)
        .map(|s| LensFlareParams { ghost_seed: s as f32, ..base })
        .find(|p| {
            let (t, _, _) = p.ghost(0);
            (0.6..1.8).contains(&t)
        })
        .expect("a seed with an in-frame ghost");
    let (t, _, _) = p.ghost(0);
    let l = [p.center[0] * W as f32, p.center[1] * H as f32];
    let g = [l[0] + (0.5 * W as f32 - l[0]) * t, l[1] + (0.5 * H as f32 - l[1]) * t];
    let img = draw(&ctx, &mut comp, p, [0.0, 0.0, 0.0, 1.0]);
    let on = lum(at(&img, g[0], g[1]));
    let off = lum(at(&img, W as f32 - g[0], g[1]));
    assert!(on > 0.2, "nothing at the ghost ({g:?}): {on}");
    assert!(off < 0.02, "light off the axis: {off}");
}

#[test]
fn occlusion_follows_the_input_at_the_light() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    let p = LensFlareParams { glow: 1.0, occlude: true, threshold: 1.0, knee: 0.25, mode: GlowMode::Only, ..quiet() };
    let (lx, ly) = (p.center[0] * W as f32, p.center[1] * H as f32);
    let dark = draw(&ctx, &mut comp, p, [0.1, 0.1, 0.1, 1.0]);
    assert!(lum(at(&dark, lx, ly)) < 1e-3, "a dark light still flares");
    let bright = draw(&ctx, &mut comp, p, [4.0, 4.0, 4.0, 1.0]);
    assert!(lum(at(&bright, lx, ly)) > 1.0, "a bright light does not flare");
}

/// The default flare over a dark plate, to `target/lensflare-look.png`
/// (Reinhard + 2.2, a look only). `cargo test -p engine --test lensflare
/// lens_flare_look -- --ignored`.
#[test]
#[ignore]
fn lens_flare_look() {
    let Some(ctx) = ctx() else { return };
    let (w, h) = (960u32, 540u32);
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(w, h);
    let mut g = Graph::default_poc();
    g.nodes.insert(3, NodeKind::Blank(BlankParams { size: (w, h), color: [0.01, 0.015, 0.03, 1.0], ..Default::default() }));
    g.formats.insert(3, (w, h));
    g.nodes.insert(7, NodeKind::LensFlare(LensFlareParams { center: [0.28, 0.32], ..Default::default() }));
    g.formats.insert(7, (w, h));
    g.edges.insert((7, 0), 3);
    g.view = Some(7);
    let img = render_wh(&ctx, &mut comp, &g, w, h);
    let px: Vec<u8> = img
        .iter()
        .flat_map(|p| [p[0], p[1], p[2]].map(|v| ((v / (1.0 + v)).clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8))
        .collect();
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../target/lensflare-look.png");
    let f = std::fs::File::create(&path).unwrap();
    let mut e = png::Encoder::new(std::io::BufWriter::new(f), w, h);
    e.set_color(png::ColorType::Rgb);
    e.set_depth(png::BitDepth::Eight);
    e.write_header().unwrap().write_image_data(&px).unwrap();
    eprintln!("wrote {}", path.display());
}
