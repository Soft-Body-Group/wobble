//! Convolve timing: the gather (3x3, 7x7) and the kernel-image FFT (full,
//! half, quarter), with and without the modulate matte, on a 2K and a 4K
//! canvas. Each case is a Read -> Convolve graph; a bare Read is the
//! baseline subtracted, so the number is what the node costs.
//!
//!     cargo run --release -p engine --example convolve_bench [ITERS=20]
//!
//! Best of ITERS batches of BATCH frames, each batch one submit + wait
//! (the GPU clocks up over the first runs, so best-of is the honest read).

use engine::compositor::Compositor;
use engine::decode::ImageInfo;
use engine::graph::{ConvolveParams, ConvolvePreset, ConvolveRes, Graph, NodeKind};
use engine::timing::GpuTimer;
use std::time::Instant;

const BATCH: usize = 8;

fn upload(ctx: &engine::GpuContext, comp: &mut Compositor, slot: usize, w: u32, h: u32, f: impl Fn(u32, u32) -> [f32; 4]) {
    comp.ensure_reader(slot, ImageInfo { width: w, height: h });
    let mut px: Vec<[half::f16; 4]> = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            px.push(f(x, y).map(half::f16::from_f32));
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

fn time(ctx: &engine::GpuContext, comp: &mut Compositor, g: &Graph, iters: usize) -> f64 {
    assert!(comp.set_graph(g), "compile");
    let timer = GpuTimer::new(ctx);
    let mut best = f64::MAX;
    for i in 0..iters + 2 {
        comp.write_params(g);
        let t = Instant::now();
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        for _ in 0..BATCH {
            comp.encode_comp(&mut enc, &timer);
        }
        ctx.queue.submit([enc.finish()]);
        let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
        if i >= 2 {
            best = best.min(t.elapsed().as_secs_f64() * 1000.0 / BATCH as f64);
        }
    }
    best
}

fn main() {
    let iters: usize = std::env::var("ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(20);
    let only = std::env::args().nth(1);
    let ctx = engine::GpuContext::new().expect("a GPU");
    for (w, h) in [(2048u32, 1080u32), (4096, 2160)] {
        if only.as_deref().is_some_and(|o| !format!("{w}").starts_with(o)) {
            continue;
        }
        let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
        comp.ensure_output(w, h);
        upload(&ctx, &mut comp, 0, w, h, |x, y| {
            let v = ((x * 7 + y * 13) % 97) as f32 / 97.0;
            [v, 1.0 - v, (x % 31) as f32 / 31.0, 1.0]
        });
        // A 255 px star: a kernel picture with a format of its own.
        let k = 255u32;
        upload(&ctx, &mut comp, 1, k, k, |x, y| {
            let (dx, dy) = (x as f32 - 127.0, y as f32 - 127.0);
            let r = (dx * dx + dy * dy).sqrt();
            let s = if dx.abs() < 1.0 || dy.abs() < 1.0 { 1.0 } else { 0.0 };
            [s / (1.0 + r), s / (1.0 + r) * 0.8, s / (1.0 + r) * 0.6, 1.0]
        });
        upload(&ctx, &mut comp, 2, w, h, |x, _| [x as f32 / w as f32; 4]);
        let read = |g: &mut Graph, id, slot| {
            g.nodes.insert(id, NodeKind::read(slot, ""));
        };
        let mut base = Graph::default();
        read(&mut base, 1, 0);
        base.view = Some(1);
        let b = time(&ctx, &mut comp, &base, iters);
        println!("{w}x{h}  baseline (Read only) {b:.3} ms");
        let mut case = |name: &str, p: ConvolveParams, kernel: bool, modulate: bool| {
            let mut g = Graph::default();
            read(&mut g, 1, 0);
            g.nodes.insert(3, NodeKind::Convolve(p));
            g.edges.insert((3, 0), 1);
            if kernel {
                read(&mut g, 2, 1);
                g.edges.insert((3, 2), 2);
                g.formats.insert(2, (k, k));
            }
            if modulate {
                read(&mut g, 4, 2);
                g.edges.insert((3, 3), 4);
            }
            g.view = Some(3);
            let t = time(&ctx, &mut comp, &g, iters);
            println!("{w}x{h}  {name:<28} {:>8.3} ms", t - b);
        };
        let mut g3 = ConvolveParams::preset(ConvolvePreset::Gaussian);
        case("gather 3x3", g3.clone(), false, false);
        case("gather 3x3 + modulate", g3.clone(), false, true);
        g3.size = 7;
        g3.kernel = vec![1.0; 49];
        case("gather 7x7", g3.clone(), false, false);
        case("gather 7x7 + modulate", g3.clone(), false, true);
        for res in ConvolveRes::ALL {
            let p = ConvolveParams { res, ..ConvolveParams::default() };
            case(&format!("fft {}", res.label()), p.clone(), true, false);
            case(&format!("fft {} + modulate", res.label()), p, true, true);
        }
    }
}
