//! The Plasma node on real pixels: its particle layer drawn, sampled in
//! the fused pass and read back. Light must hug the star (loops near the
//! limb, nothing far away), stay off the far side of the disc, move with
//! the clock, and never touch the topology hash. Self-skips without a GPU
//! (CI).

use engine::color::Colorspace;
use engine::compositor::Compositor;
use engine::graph::{Graph, NodeKind, PlasmaParams};
use engine::timing::GpuTimer;
use half::f16;

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP plasma test: no adapter ({e})");
            None
        }
    }
}

fn frame(ctx: &engine::GpuContext, comp: &mut Compositor, graph: &Graph, w: u32, h: u32) -> Vec<[f32; 4]> {
    comp.ensure_output(w, h);
    assert!(comp.set_graph(graph), "fused shader failed to compile");
    let timer = GpuTimer::new(ctx);
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
    ctx.queue.submit([encoder.finish()]);
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    let data = slice.get_mapped_range().expect("a mapped buffer");
    let mut out = Vec::new();
    for y in 0..h as usize {
        let row: &[f16] = bytemuck::cast_slice(&data[y * padded as usize..][..(w * 8) as usize]);
        for x in 0..w as usize {
            out.push([0, 1, 2, 3].map(|c| row[x * 4 + c].to_f32()));
        }
    }
    drop(data);
    staging.unmap();
    out
}

/// `PLASMA_DUMP=<dir>` writes a render as an 8-bit PNG (a filmic-ish
/// shoulder, so stacked light whitens instead of clipping) for a look.
fn dump(name: &str, px: &[[f32; 4]], w: u32, h: u32) {
    let Ok(dir) = std::env::var("PLASMA_DUMP") else { return };
    let rgb: Vec<u8> = px
        .iter()
        .flat_map(|p| {
            // Hot light spills into the other channels, as a display
            // transform's would.
            let m = p[0].max(p[1]).max(p[2]);
            let spill = (m - 1.0).max(0.0) * 0.06;
            (0..3).map(move |k| {
                let x = p[k] + spill;
                ((1.0 - (-x * 1.4).exp()).clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8
            })
        })
        .collect();
    let f = std::fs::File::create(std::path::Path::new(&dir).join(format!("{name}.png"))).unwrap();
    let mut e = png::Encoder::new(std::io::BufWriter::new(f), w, h);
    e.set_color(png::ColorType::Rgb);
    e.write_header().unwrap().write_image_data(&rgb).unwrap();
}

fn alone(p: PlasmaParams) -> Graph {
    let mut g = Graph::default();
    g.working = Colorspace::LINEAR_REC709;
    g.nodes.insert(1, NodeKind::Plasma(p));
    g.view = Some(1);
    g
}

const W: u32 = 256;
const H: u32 = 256;

fn small() -> PlasmaParams {
    PlasmaParams { center: [128.0, 128.0], radius: 60.0, count: 60_000.0, turbulence: 2.0, ..Default::default() }
}

/// Light summed over the pixels whose distance from the centre is in
/// `lo..hi` radii.
fn light(px: &[[f32; 4]], p: &PlasmaParams, lo: f32, hi: f32) -> f32 {
    let mut sum = 0.0;
    for (i, q) in px.iter().enumerate() {
        let (x, y) = ((i as u32 % W) as f32 + 0.5, (i as u32 / W) as f32 + 0.5);
        let r = ((x - p.center[0]).powi(2) + (y - p.center[1]).powi(2)).sqrt() / p.radius;
        if r >= lo && r < hi {
            sum += q[0] + q[1] + q[2];
        }
    }
    sum
}

#[test]
fn light_hugs_the_limb_and_is_finite() {
    let Some(ctx) = ctx() else { return };
    let mut c = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 1);
    // Flat: every loop an arch on the limb, no streamers.
    let p = PlasmaParams { depth: 0.0, streamers: 0.0, ..small() };
    let px = frame(&ctx, &mut c, &alone(p.clone()), W, H);
    dump("plasma_flat", &px, W, H);
    assert!(px.iter().all(|q| q.iter().all(|v| v.is_finite() && *v >= 0.0)));
    let near = light(&px, &p, 0.95, 1.6);
    let inside = light(&px, &p, 0.0, 0.8);
    let far = light(&px, &p, 2.0, 9.0);
    assert!(near > 1.0, "the loops drew: {near}");
    assert!(inside < near * 0.02, "arches on the limb stand OFF the disc: {inside} vs {near}");
    assert!(far < near * 0.02, "and stay near it: {far} vs {near}");
    // Emission by default: alpha 0. With `alpha` the light is coverage.
    assert!(px.iter().all(|q| q[3] == 0.0));
    let a = frame(&ctx, &mut c, &alone(PlasmaParams { alpha: 1000.0, ..p }), W, H);
    assert!(a.iter().any(|q| q[3] > 0.99) && a.iter().all(|q| q[3] <= 1.0));
}

#[test]
fn the_disc_hides_what_is_behind_it() {
    let Some(ctx) = ctx() else { return };
    let mut c = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 1);
    let p = PlasmaParams { depth: 1.0, streamers: 0.0, cluster: 0.0, ..small() };
    let hidden = frame(&ctx, &mut c, &alone(p.clone()), W, H);
    let shown = frame(&ctx, &mut c, &alone(PlasmaParams { occlude: false, ..p.clone() }), W, H);
    dump("plasma_sphere", &hidden, W, H);
    let (h, s) = (light(&hidden, &p, 0.0, 0.9), light(&shown, &p, 0.0, 0.9));
    assert!(h > 1.0, "loops on the near side lie across the face: {h}");
    assert!(h < s * 0.75, "the far side's are gone: {h} vs {s}");
}

#[test]
fn the_clock_moves_it_and_nothing_recompiles() {
    let Some(ctx) = ctx() else { return };
    let mut c = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 1);
    let p = small();
    let g0 = alone(p.clone());
    let a = frame(&ctx, &mut c, &g0, W, H);
    assert_eq!(c.plasma_layer_dirty(1), Some(false));
    // The same frame again: the layer is not re-drawn.
    let again = frame(&ctx, &mut c, &g0, W, H);
    assert_eq!(a, again);
    let g1 = alone(PlasmaParams { time: 12.0, brightness: 2.0, loops: 40.0, color: [0.2, 0.4, 1.0], ..p });
    assert_eq!(g1.topology_hash(), g0.topology_hash(), "every number is a uniform");
    let b = frame(&ctx, &mut c, &g1, W, H);
    assert_ne!(a, b);
    let (ra, ba): (f32, f32) = (b.iter().map(|q| q[0]).sum(), b.iter().map(|q| q[2]).sum());
    assert!(ba > ra * 2.0, "it took the new colour: r {ra} b {ba}");
}

fn to8(px: &[[f32; 4]]) -> Vec<u8> {
    px.iter()
        .flat_map(|p| {
            let m = p[0].max(p[1]).max(p[2]);
            let spill = (m - 1.0).max(0.0) * 0.06;
            (0..3).map(move |k| ((1.0 - (-(p[k] + spill) * 1.4).exp()).clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8)
        })
        .collect()
}

/// `PLASMA_DUMP=<dir> cargo test --release -p engine --test plasma look --
/// --ignored`: the default node at 1080p, three frames, for a look.
#[test]
#[ignore]
fn look() {
    let Some(ctx) = ctx() else { return };
    let mut c = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 1);
    use engine::graph::{BlendOp, GlowParams};
    use engine::nodes::plasma::PRESETS;
    let frames: Vec<f32> = match std::env::var("PLASMA_FRAMES") {
        Ok(n) => (0..n.parse::<u32>().unwrap()).map(|f| f as f32).collect(),
        Err(_) => vec![0.0, 24.0],
    };
    // PLASMA_LAYERS="eclipse,prominences": which presets to add up.
    let layers = std::env::var("PLASMA_LAYERS").unwrap_or_else(|_| "surface,streamers,active regions".into());
    let layers: Vec<&str> = layers.split(',').collect();
    // PLASMA_CLOSE=1: the limb filling the frame, like a TRACE plate.
    let close = std::env::var("PLASMA_CLOSE").is_ok();
    let mut anim: Vec<Vec<[f32; 4]>> = Vec::new();
    let animate = frames.len() > 2;
    for t in frames {
        let mut g = Graph::default();
        g.working = Colorspace::LINEAR_REC709;
        let mut top = 0u64;
        for (i, name) in layers.iter().enumerate() {
            let pre = PRESETS.iter().find(|p| p.name == *name).unwrap();
            let mut p = PlasmaParams { time: t, seed: i as f32, ..Default::default() };
            p.apply_preset(pre);
            if close {
                p.center = [960.0, 2300.0];
                p.radius = 1500.0;
            }
            if layers[0] == "eclipse" {
                p.radius = 190.0;
            }
            let id = 1 + i as u64;
            g.nodes.insert(id, NodeKind::Plasma(p));
            if top != 0 {
                let m = 10 + i as u64;
                g.nodes.insert(m, NodeKind::Blend { op: BlendOp::Add, mix: 1.0, gain: [1.0, 1.0] });
                g.edges.insert((m, 0), id);
                g.edges.insert((m, 1), top);
                top = m;
            } else {
                top = id;
            }
        }
        let glow = if layers[0] == "eclipse" {
            GlowParams { size: 50.0, falloff: 0.8, threshold: 0.0, knee: 0.0, intensity: 0.5, ..Default::default() }
        } else {
            GlowParams { size: 40.0, threshold: 0.3, intensity: 0.6, ..Default::default() }
        };
        g.nodes.insert(12, NodeKind::Glow(glow));
        g.edges.insert((12, 0), top);
        g.view = Some(if std::env::var("PLASMA_NOGLOW").is_ok() { top } else { 12 });
        let px = frame(&ctx, &mut c, &g, 1920, 1080);
        if std::env::var("PLASMA_PROBE").is_ok() {
            for x in [1200usize, 1300, 1400, 1500, 1600, 1700] {
                eprintln!("x={x} {:?}", px[540 * 1920 + x]);
            }
        }
        if animate {
            // Every other frame, a third the size: an APNG for a look at the motion.
            if t as u32 % 2 == 0 {
                let mut small = Vec::with_capacity(640 * 360);
                for y in 0..360 {
                    for x in 0..640 {
                        let mut acc = [0.0f32; 4];
                        for (dy, dx) in (0..3).flat_map(|dy| (0..3).map(move |dx| (dy, dx))) {
                            let p = px[(y * 3 + dy) * 1920 + x * 3 + dx];
                            for k in 0..4 {
                                acc[k] += p[k] / 9.0;
                            }
                        }
                        small.push(acc);
                    }
                }
                anim.push(small);
            }
        } else {
            dump(&format!("plasma_{}_{t:03}", if close { "close" } else if layers[0] == "eclipse" { "eclipse" } else { "look" }), &px, 1920, 1080);
        }
    }
    if let (true, Ok(dir)) = (animate, std::env::var("PLASMA_DUMP")) {
        let f = std::fs::File::create(std::path::Path::new(&dir).join("plasma_anim.png")).unwrap();
        let mut e = png::Encoder::new(std::io::BufWriter::new(f), 640, 360);
        e.set_color(png::ColorType::Rgb);
        e.set_animated(anim.len() as u32, 0).unwrap();
        e.set_frame_delay(1, 12).unwrap();
        let mut w = e.write_header().unwrap();
        for fr in &anim {
            w.write_image_data(&to8(fr)).unwrap();
        }
        w.finish().unwrap();
    }
}
