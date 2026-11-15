//! The Splat node on real pixels: a capture drawn through a camera. The
//! nearer splat covers the farther whatever order the file lists them in
//! (the GPU sort), the depth output is the distance, a big random capture
//! renders the same shuffled (the sort is a sort), a still camera draws
//! once, and a multi-million-splat capture renders in usable time.
//! Self-skips without a GPU (CI).

use engine::compositor::Compositor;
use engine::graph::{CameraParams, Graph, NodeKind, SplatParams};
use engine::splat::{Native, Raw, SH_C0};
use engine::timing::GpuTimer;
use half::f16;

const W: u32 = 64;
const H: u32 = 40;

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP splat test: no adapter ({e})");
            None
        }
    }
}

fn readback(ctx: &engine::GpuContext, comp: &Compositor, w: u32, h: u32, encoder: &mut wgpu::CommandEncoder) -> wgpu::Buffer {
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
    staging
}

fn frame(ctx: &engine::GpuContext, comp: &mut Compositor, graph: &Graph) -> Vec<[f32; 4]> {
    assert!(comp.set_graph(graph), "fused shader failed to compile");
    let timer = GpuTimer::new(ctx);
    comp.write_params(graph);
    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    comp.encode_comp(&mut encoder, &timer);
    let staging = readback(ctx, comp, W, H, &mut encoder);
    ctx.queue.submit([encoder.finish()]);
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    let padded = (W * 8).div_ceil(256) * 256;
    let data = slice.get_mapped_range().expect("a mapped buffer");
    let mut out = Vec::new();
    for y in 0..H as usize {
        let row: &[f16] = bytemuck::cast_slice(&data[y * padded as usize..][..(W * 8) as usize]);
        for x in 0..W as usize {
            out.push([0, 1, 2, 3].map(|k| row[x * 4 + k].to_f32()));
        }
    }
    drop(data);
    staging.unmap();
    out
}

fn setup() -> Option<(engine::GpuContext, Compositor)> {
    let ctx = ctx()?;
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 1);
    comp.ensure_output(W, H);
    Some((ctx, comp))
}

/// A splat of `rgb` (0..1), `size` units, opaque.
fn raw(pos: [f32; 3], rgb: [f32; 3], size: f32) -> Raw {
    Raw {
        pos,
        rot: [1.0, 0.0, 0.0, 0.0],
        log_scale: [size.ln(); 3],
        opacity: 1.0,
        dc: rgb.map(|c| (c - 0.5) / SH_C0),
    }
}

/// A Splat (id 1) on `name`, through `cam`, viewed at `out`, lowered.
fn graph(name: &str, cam: Option<CameraParams>, out: u32) -> Graph {
    let mut g = Graph::default();
    let p = SplatParams { path: name.into(), linear: true, ..Default::default() };
    g.nodes.insert(1, NodeKind::Splat(p));
    if let Some(c) = cam {
        g.nodes.insert(2, NodeKind::Camera3D(c));
        g.edges.insert((1, 0), 2);
    }
    g.formats.insert(1, (W, H));
    g.view = Some(1);
    g.view_out = out;
    engine::nodes::splat::lower(&mut g);
    g
}

fn at(out: &[[f32; 4]], x: u32, y: u32) -> [f32; 4] {
    out[(y * W + x) as usize]
}

#[test]
fn the_nearer_splat_wins_whatever_the_file_order() {
    let Some((ctx, mut comp)) = setup() else { return };
    let near = raw([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 0.05);
    let far = raw([0.0, 0.0, -1.0], [0.0, 0.0, 1.0], 0.3);
    for (name, order) in [("mem:near-first", vec![near, far]), ("mem:far-first", vec![far, near])] {
        engine::splat::insert(name, engine::splat::pack(&order, &[], 0, Native::Rub, false));
        let out = frame(&ctx, &mut comp, &graph(name, None, 0));
        let c = at(&out, W / 2, H / 2);
        assert!(c[0] > 0.9 && c[2] < 0.05 && c[3] > 0.95, "{name}: centre {c:?}");
        // Off the red splat but on the big blue one behind it.
        let e = at(&out, W / 2 + 14, H / 2);
        assert!(e[2] > 0.3 && e[0] < 0.05, "{name}: edge {e:?}");
        // Clear of both: nothing.
        assert!(at(&out, 1, 1)[3] < 0.01);
    }
}

#[test]
fn the_depth_output_is_the_distance() {
    let Some((ctx, mut comp)) = setup() else { return };
    let name = "mem:depth";
    engine::splat::insert(name, engine::splat::pack(&[raw([0.0, 0.0, -2.0], [1.0; 3], 0.5)], &[], 0, Native::Rub, false));
    let out = frame(&ctx, &mut comp, &graph(name, None, 1));
    let cam = CameraParams::default();
    let want = cam.xform.translate[2] + 2.0;
    let d = at(&out, W / 2, H / 2)[0];
    assert!((d - want).abs() < want * 0.01, "depth {d}, want {want}");
    let far = at(&out, 0, 0)[0];
    assert!(far > 1000.0, "background {far}");
}

#[test]
fn a_ply_is_turned_y_up() {
    let Some((ctx, mut comp)) = setup() else { return };
    // A .ply's +Y is DOWN: a splat at file y = +0.3 lands below centre.
    let name = "mem:rdf";
    engine::splat::insert(
        name,
        engine::splat::pack(&[raw([0.0, 0.3, 0.0], [1.0; 3], 0.08)], &[], 0, Native::Rdf, false),
    );
    let out = frame(&ctx, &mut comp, &graph(name, None, 0));
    let (mut best, mut by) = (0.0, 0);
    for y in 0..H {
        let a = at(&out, W / 2, y)[3];
        if a > best {
            best = a;
            by = y;
        }
    }
    assert!(by > H / 2 + 3, "brightest row {by}");
}

fn rnd(s: &mut u64) -> f32 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    (*s >> 40) as f32 / (1u64 << 24) as f32
}

fn cloud(n: usize, seed: u64) -> Vec<Raw> {
    cloud_sized(n, seed, 1.0)
}

/// `k` scales every splat: a real capture's splats are a pixel or two.
fn cloud_sized(n: usize, seed: u64, k: f32) -> Vec<Raw> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            let p = [rnd(&mut s) - 0.5, (rnd(&mut s) - 0.5) * 0.6, -rnd(&mut s) * 2.0];
            let mut r = raw(p, [rnd(&mut s), rnd(&mut s), rnd(&mut s)], (0.004 + 0.02 * rnd(&mut s)) * k);
            r.opacity = 0.3 + 0.6 * rnd(&mut s);
            r
        })
        .collect()
}

#[test]
fn a_shuffled_capture_renders_the_same() {
    let Some((ctx, mut comp)) = setup() else { return };
    let a = cloud(150_000, 7);
    let mut b = a.clone();
    let mut s = 99u64;
    for i in (1..b.len()).rev() {
        let j = (rnd(&mut s) * (i + 1) as f32) as usize % (i + 1);
        b.swap(i, j);
    }
    engine::splat::insert("mem:a", engine::splat::pack(&a, &[], 0, Native::Rub, false));
    engine::splat::insert("mem:b", engine::splat::pack(&b, &[], 0, Native::Rub, false));
    let oa = frame(&ctx, &mut comp, &graph("mem:a", None, 0));
    let ob = frame(&ctx, &mut comp, &graph("mem:b", None, 0));
    let mut worst = 0.0f32;
    let mut cover = 0.0f32;
    for (x, y) in oa.iter().zip(&ob) {
        cover += x[3];
        for k in 0..4 {
            worst = worst.max((x[k] - y[k]).abs());
        }
    }
    assert!(cover > (W * H) as f32 * 0.3, "the cloud covers the frame: {cover}");
    // Ties in a 24-bit key may land either way; half floats round.
    assert!(worst < 0.02, "worst {worst}");
}

#[test]
fn a_still_camera_draws_once() {
    let Some((ctx, mut comp)) = setup() else { return };
    engine::splat::insert("mem:still", engine::splat::pack(&cloud(1000, 3), &[], 0, Native::Rub, false));
    let g = graph("mem:still", None, 0);
    frame(&ctx, &mut comp, &g);
    frame(&ctx, &mut comp, &g);
    let (_, draws, n) = comp.splat_layer_state(1).unwrap();
    assert_eq!(n, 1000);
    assert_eq!(draws, 1);
    let moved = graph("mem:still", Some(CameraParams { focal: 70.0, ..Default::default() }), 0);
    frame(&ctx, &mut comp, &moved);
    assert_eq!(comp.splat_layer_state(1).unwrap().1, 2);
}

/// Four million splats through a moving camera at 1080p: the time per
/// frame, printed. Gated on WOBBLE_SPLAT_PERF so the suite stays quick.
#[test]
fn millions_of_splats_stay_usable() {
    if std::env::var("WOBBLE_SPLAT_PERF").is_err() {
        return;
    }
    let Some(ctx) = ctx() else { return };
    let (w, h) = (1920u32, 1080u32);
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 1);
    comp.ensure_output(w, h);
    let n = 4_000_000;
    let mut sh = Vec::with_capacity(n * 45);
    let mut s = 5u64;
    for _ in 0..n * 45 {
        sh.push((rnd(&mut s) - 0.5) * 0.1);
    }
    let t = std::time::Instant::now();
    engine::splat::insert("mem:perf", engine::splat::pack(&cloud_sized(n, 11, 0.06), &sh, 3, Native::Rub, false));
    eprintln!("pack {n} splats: {:.0} ms", t.elapsed().as_secs_f64() * 1e3);
    let timer = GpuTimer::new(&ctx);
    let mut times = Vec::new();
    for i in 0..12 {
        let cam = CameraParams {
            xform: engine::scene3d::Xform { translate: [i as f32 * 0.02, 0.0, 1.4], ..Default::default() },
            ..Default::default()
        };
        let mut g = graph("mem:perf", Some(cam), 0);
        g.formats.insert(1, (w, h));
        assert!(comp.set_graph(&g));
        comp.write_params(&g);
        let t = std::time::Instant::now();
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        comp.encode_comp(&mut enc, &timer);
        ctx.queue.submit([enc.finish()]);
        let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
        times.push(t.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(|a, b| a.total_cmp(b));
    eprintln!("4M splats @1080p, SH3: median {:.1} ms/frame (min {:.1})", times[times.len() / 2], times[0]);
    assert!(times[times.len() / 2] < 1000.0);
}
