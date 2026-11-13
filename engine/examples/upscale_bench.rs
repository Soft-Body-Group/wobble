//! Upscale timing: a 1080p frame to 4K (each model at x4, then the [1 3 3
//! 1] halving), f16 storage where offered.
//!
//!     cargo run --release -p engine --example upscale_bench [w h] [x4]
//!
//! Needs the weights in `WOBBLE_MODELS_DIR` / `~/.wobble/models`.

use engine::nn::gpu::{Colour, Kernels, Runner};
use engine::upscale::Model;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (w, h) = if args.len() >= 3 {
        (args[1].parse().unwrap(), args[2].parse().unwrap())
    } else {
        (1920u32, 1080u32)
    };
    let out_scale = if args.iter().any(|a| a == "x4") { 4 } else { 2 };
    let ctx = engine::GpuContext::new().expect("a GPU");
    let mut k = Kernels::new(ctx.device.clone(), true);
    println!("{w}x{h} -> {}x{} (f16 {})", w * out_scale, h * out_scale, k.half);
    let tex = |w, h, usage| {
        ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage,
            view_formats: &[],
        })
    };
    let input = tex(w, h, wgpu::TextureUsages::TEXTURE_BINDING);
    let output = tex(w * out_scale, h * out_scale, wgpu::TextureUsages::STORAGE_BINDING);
    let (iv, ov) = (input.create_view(&Default::default()), output.create_view(&Default::default()));
    let sampler = ctx.device.create_sampler(&Default::default());
    let only: Vec<Model> = Model::ALL
        .into_iter()
        .filter(|m| args.iter().any(|a| a.eq_ignore_ascii_case(&format!("{m:?}"))))
        .collect();
    let models = if only.is_empty() { Model::ALL.to_vec() } else { only };
    let iters: usize = std::env::var("ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
    for model in models {
        let Some(net) = engine::upscale::net(model) else {
            println!("{}: weights missing", model.label());
            continue;
        };
        let t0 = Instant::now();
        let mut r = Runner::new(&mut k, &ctx.queue, &net, (w, h), out_scale, 1 << 30);
        r.set_colour(&ctx.queue, Colour { encode: true, ..Colour::RAW });
        let build = t0.elapsed();
        let mut times = Vec::new();
        for _ in 0..iters {
            let t = Instant::now();
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            r.encode(&k, &ctx.queue, &mut enc, &iv, &ov, &sampler);
            ctx.queue.submit([enc.finish()]);
            let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
            times.push(t.elapsed().as_secs_f64() * 1e3);
            eprintln!("  {:?} {:.1} ms", model, times.last().unwrap());
        }
        let best = times.iter().cloned().fold(f64::MAX, f64::min);
        let gmac = net.macs_per_pixel() as f64 * (w * h) as f64 / 1e9;
        println!(
            "{:32} {:8.1} ms/frame  ({} tiles of {:?}, arena {} MB, {:.0} GMAC, {:.2} TFLOP/s, build {} ms)",
            model.label(),
            best,
            r.tile_count(),
            r.tile_size(),
            r.arena_bytes >> 20,
            gmac,
            2.0 * gmac / best,
            build.as_millis()
        );
    }
}
