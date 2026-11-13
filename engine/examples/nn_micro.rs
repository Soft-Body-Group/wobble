//! One conv, timed: what the kernel itself achieves.
use engine::nn::gpu::{Kernels, Runner};
use engine::nn::{Buf, Conv, Epi, Net, Op, Out, Slice};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let c: u32 = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(48);
    let n: u32 = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(1);
    let half = !args.iter().any(|a| a == "f32");
    let ctx = engine::GpuContext::new().unwrap();
    let mut k = Kernels::new(ctx.device.clone(), half);
    let w: std::sync::Arc<[f32]> = vec![0.01f32; (c * c * 9) as usize].into();
    let b: std::sync::Arc<[f32]> = vec![0.0f32; c as usize].into();
    let ops = (0..n)
        .map(|i| {
            let (s, d) = if i % 2 == 0 { (1, 2) } else { (2, 1) };
            Op::Conv(Conv {
                k: 3, cin: c, cout: c, weight: w.clone(), bias: b.clone(),
                src: Slice::new(s, 0, c), dst: Slice::new(d, 0, c), up: 1, silu_in: false, epi: Epi::none(),
            })
        })
        .collect();
    let net = Net {
        bufs: vec![Buf { ch: 4, res: 1, div: 1 }, Buf { ch: c, res: 1, div: 1 }, Buf { ch: c, res: 1, div: 1 }, Buf { ch: 4, res: 4, div: 1 }],
        input: Slice::new(0, 0, 4), mean: [0.0; 3], range: 1.0, ops,
        out: Out::Direct(Slice::new(3, 0, 4)), base: false, scale: 4, halo: 0, io: engine::nn::Io::Upscale,
    };
    let sz = 512u32;
    let tex = |w, h, usage| ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: None, size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float, usage, view_formats: &[] });
    let input = tex(sz, sz, wgpu::TextureUsages::TEXTURE_BINDING);
    let output = tex(sz * 4, sz * 4, wgpu::TextureUsages::STORAGE_BINDING);
    let (iv, ov) = (input.create_view(&Default::default()), output.create_view(&Default::default()));
    let sampler = ctx.device.create_sampler(&Default::default());
    let r = Runner::new(&mut k, &ctx.queue, &net, (sz, sz), 4, 1 << 30);
    let mut best = f64::MAX;
    for _ in 0..20 {
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        r.encode(&k, &ctx.queue, &mut enc, &iv, &ov, &sampler);
        let t = Instant::now();
        ctx.queue.submit([enc.finish()]);
        let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let gmac = (c * c * 9) as f64 * (sz * sz) as f64 * n as f64 / 1e9;
        best = best.min(ms);
        if ms > best { continue; }
        let _ = gmac;
    }
    let gmac = (c * c * 9) as f64 * (sz * sz) as f64 * n as f64 / 1e9;
    println!("{c}ch x{n} half={} best {best:.2} ms  {:.2} TFLOP/s", k.half, 2.0 * gmac / best);
}
