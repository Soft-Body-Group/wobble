//! The Particles node on real pixels: its compute simulation stepped,
//! drawn, sampled in the fused pass and read back. A frame must not care
//! how it was reached (played, jumped to, scrubbed back over a
//! checkpoint), a look edit must not re-simulate, nothing may touch the
//! topology hash, and the corona preset's light must hug the limb.
//! Self-skips without a GPU (CI).
//!
//! `PARTICLES_DUMP=<dir> cargo test --release -p engine --test particles look -- --ignored`
//! renders every preset for a look.

use engine::color::Colorspace;
use engine::compositor::Compositor;
use engine::graph::{Graph, NodeKind, PNode, ParticlesParams};
use engine::nodes::particles::{ForceKind, PRESETS};
use engine::timing::GpuTimer;
use half::f16;

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP particles test: no adapter ({e})");
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

fn dump(name: &str, px: &[[f32; 4]], w: u32, h: u32) {
    let Ok(dir) = std::env::var("PARTICLES_DUMP") else { return };
    let rgb: Vec<u8> = px
        .iter()
        .flat_map(|p| {
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

fn alone(p: ParticlesParams, w: u32, h: u32) -> Graph {
    let mut g = Graph::default();
    g.working = Colorspace::LINEAR_REC709;
    g.nodes.insert(1, NodeKind::Particles(ParticlesParams { size: (w, h), ..p }));
    g.formats.insert(1, (w, h));
    g.view = Some(1);
    g
}

fn at(mut g: Graph, t: f32) -> Graph {
    if let Some(NodeKind::Particles(p)) = g.nodes.get_mut(&1) {
        p.time = t;
    }
    g
}

/// The fountain preset scaled into a small canvas.
fn fountain(w: u32, h: u32) -> ParticlesParams {
    let mut graph = (PRESETS[0].build)();
    for n in graph.nodes.values_mut() {
        if let PNode::Emit(e) = n {
            e.center = [w as f32 * 0.5, h as f32 * 0.8];
            e.rate = 400.0;
            e.speed = 4.0;
        }
    }
    // Start at 0 so frame numbers are step counts.
    ParticlesParams { graph, start: 0.0, ..Default::default() }
}

fn total(px: &[[f32; 4]]) -> f32 {
    px.iter().map(|q| q[0] + q[1] + q[2]).sum()
}

const W: u32 = 192;
const H: u32 = 192;

#[test]
fn a_fountain_sprays_up_and_is_finite() {
    let Some(ctx) = ctx() else { return };
    let mut c = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 1);
    let g = alone(fountain(W, H), W, H);
    // Before the first step: nothing.
    let px = frame(&ctx, &mut c, &at(g.clone(), 0.0), W, H);
    assert_eq!(total(&px), 0.0, "the sim starts empty");
    let px = frame(&ctx, &mut c, &at(g, 12.0), W, H);
    dump("fountain12", &px, W, H);
    assert!(px.iter().all(|q| q.iter().all(|v| v.is_finite() && *v >= 0.0)));
    let (mut above, mut below) = (0.0, 0.0);
    for (i, q) in px.iter().enumerate() {
        let y = (i as u32 / W) as f32;
        let l = q[0] + q[1] + q[2];
        if y < H as f32 * 0.78 { above += l } else if y > H as f32 * 0.82 { below += l }
    }
    assert!(above > 1.0, "the spray drew: {above}");
    assert!(below < above * 0.1, "launched upwards: {below} vs {above}");
    // Emission by default: alpha 0.
    assert!(px.iter().all(|q| q[3] == 0.0));
}

#[test]
fn a_frame_does_not_care_how_it_was_reached() {
    let Some(ctx) = ctx() else { return };
    let g = alone(fountain(W, H), W, H);
    // Played: every frame in order.
    let mut played = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 1);
    let mut last = Vec::new();
    for t in 0..=40 {
        last = frame(&ctx, &mut played, &at(g.clone(), t as f32), W, H);
    }
    let (_, steps, cks) = played.particle_layer_state(1).unwrap();
    assert_eq!(steps, 1, "playing forward is one step a frame");
    assert_eq!(cks, vec![8, 16, 24, 32, 40], "checkpoints every 8 frames");
    // Jumped to cold.
    let mut jumped = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 1);
    let cold = frame(&ctx, &mut jumped, &at(g.clone(), 40.0), W, H);
    assert_eq!(cold, last, "a jump lands on the played frame");
    // Scrubbed back over a checkpoint, then forward again.
    let back = frame(&ctx, &mut played, &at(g.clone(), 37.0), W, H);
    let (_, steps, _) = played.particle_layer_state(1).unwrap();
    assert_eq!(steps, 5, "37 is restored from 32, not re-run from the top");
    let again = frame(&ctx, &mut played, &at(g.clone(), 40.0), W, H);
    assert_eq!(again, last, "and forward again lands on the same frame");
    assert_ne!(back, last);
    // A still frame re-shown does nothing.
    frame(&ctx, &mut played, &at(g.clone(), 40.0), W, H);
    assert_eq!(played.particle_layer_state(1).unwrap().0, false);
}

#[test]
fn a_look_edit_redraws_without_simulating_and_nothing_recompiles() {
    let Some(ctx) = ctx() else { return };
    let mut c = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 1);
    let g = at(alone(fountain(W, H), W, H), 30.0);
    let a = frame(&ctx, &mut c, &g, W, H);
    let mut g2 = g.clone();
    if let Some(NodeKind::Particles(p)) = g2.nodes.get_mut(&1) {
        for n in p.graph.nodes.values_mut() {
            if let PNode::Render(r) = n {
                r.color = [0.0, 1.0, 0.0];
                r.color_end = [0.0, 1.0, 0.0];
            }
        }
    }
    let b = frame(&ctx, &mut c, &g2, W, H);
    assert_eq!(c.particle_layer_state(1).unwrap().1, 0, "a colour drag is not a re-simulation");
    assert!(total(&b) > 0.0 && b != a);
    assert!(b.iter().all(|q| q[0] == 0.0 && q[2] == 0.0), "green now");
    // A force edit re-simulates; neither changes the topology.
    let mut g3 = g2.clone();
    if let Some(NodeKind::Particles(p)) = g3.nodes.get_mut(&1) {
        for n in p.graph.nodes.values_mut() {
            if let PNode::Force(f) = n {
                f.strength *= 2.0;
            }
        }
    }
    frame(&ctx, &mut c, &g3, W, H);
    assert_eq!(c.particle_layer_state(1).unwrap().1, 30);
    assert_eq!(g.topology_hash(), g3.topology_hash());
}

/// The corona preset scaled into a 320 canvas: (params, centre, radius).
fn small_corona(sphere: f32, occlude: bool) -> (ParticlesParams, [f32; 2], f32) {
    let (cx, cy, r) = (160.0f32, 160.0f32, 80.0f32);
    let mut graph = (PRESETS[3].build)();
    let k = r / 300.0;
    let scale = |p: [f32; 2]| [cx + (p[0] - 960.0) * k, cy + (p[1] - 540.0) * k];
    for n in graph.nodes.values_mut() {
        match n {
            PNode::Emit(e) => {
                e.center = scale(e.center);
                e.size = [e.size[0] * k, e.size[1] * k];
                e.rate *= 0.3;
            }
            PNode::Force(f) => {
                f.center = scale(f.center);
                f.radius *= k;
                f.scale *= k;
                if f.kind == ForceKind::FieldLines {
                    f.strength *= k;
                    f.sphere = sphere;
                }
            }
            PNode::Render(rn) => {
                rn.occlude = occlude;
                rn.disc_center = [cx, cy];
                rn.disc_radius = r;
            }
            _ => {}
        }
    }
    (ParticlesParams { graph, preroll: 60.0, ..Default::default() }, [cx, cy], r)
}

/// Light summed where the distance from `c` is in `lo..hi` radii.
fn ring(px: &[[f32; 4]], w: u32, c: [f32; 2], r: f32, lo: f32, hi: f32) -> f32 {
    let mut s = 0.0;
    for (i, q) in px.iter().enumerate() {
        let (x, y) = ((i as u32 % w) as f32 + 0.5, (i as u32 / w) as f32 + 0.5);
        let d = ((x - c[0]).powi(2) + (y - c[1]).powi(2)).sqrt() / r;
        if d >= lo && d < hi {
            s += q[0] + q[1] + q[2];
        }
    }
    s
}

#[test]
fn the_corona_hugs_the_limb() {
    let Some(ctx) = ctx() else { return };
    let (w, h) = (320u32, 320u32);
    // Flat: every pole on the ring.
    let (p, c, r) = small_corona(0.0, false);
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 1);
    let px = frame(&ctx, &mut comp, &at(alone(p, w, h), 1.0), w, h);
    dump("corona_small", &px, w, h);
    let near = ring(&px, w, c, r, 0.97, 1.5);
    let inside = ring(&px, w, c, r, 0.0, 0.9);
    assert!(near > 1.0, "the loops drew: {near}");
    assert!(inside < near * 0.05, "limb loops leave on the outward side: {inside} vs {near}");
}

#[test]
fn on_a_sphere_loops_cross_the_face_and_the_disc_hides_the_far_side() {
    let Some(ctx) = ctx() else { return };
    let (w, h) = (320u32, 320u32);
    let face = |occlude: bool| {
        let (p, c, r) = small_corona(1.0, occlude);
        let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 1);
        let px = frame(&ctx, &mut comp, &at(alone(p, w, h), 1.0), w, h);
        dump(if occlude { "corona_sphere" } else { "corona_sphere_through" }, &px, w, h);
        assert!(px.iter().all(|q| q.iter().all(|v| v.is_finite() && *v >= 0.0)));
        (ring(&px, w, c, r, 0.0, 0.9), ring(&px, w, c, r, 1.05, 3.0))
    };
    let (through, out_a) = face(false);
    let (hidden, out_b) = face(true);
    assert!(hidden > 1.0, "loops in front show across the face: {hidden}");
    assert!(hidden < through * 0.8, "the far side is hidden: {hidden} vs {through}");
    // Clear of the disc nothing is hidden.
    assert!((out_a - out_b).abs() <= out_a * 1e-3, "{out_a} vs {out_b}");
}

/// Every preset at frame 90 of a 1920x1080 canvas, for a look.
#[test]
#[ignore]
fn look() {
    let Some(ctx) = ctx() else { return };
    let (w, h) = (1920u32, 1080u32);
    for pre in PRESETS {
        let mut c = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 1);
        let p = ParticlesParams { graph: (pre.build)(), preroll: 120.0, ..Default::default() };
        let g = alone(p, w, h);
        let t = std::time::Instant::now();
        let px = frame(&ctx, &mut c, &at(g, 90.0), w, h);
        eprintln!("{}: {:?}", pre.name, t.elapsed());
        dump(&pre.name.replace(' ', "_"), &px, w, h);
    }
}
