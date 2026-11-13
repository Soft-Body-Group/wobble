//! One FFT pass on its own: a 2048x2048 rgba32float grid, rows then
//! columns, best of ITERS. What `convolve_bench` is made of, separated.
//!
//!     cargo run --release -p engine --example fft_micro [n]

use engine::nodes::fft;
use std::time::Instant;
use wgpu::util::DeviceExt;

fn main() {
    let n: u32 = std::env::args().nth(1).and_then(|v| v.parse().ok()).unwrap_or(2048);
    let iters: usize = std::env::var("ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(20);
    let ctx = engine::GpuContext::new().expect("a GPU");
    let d = &ctx.device;
    let tex = || {
        d.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width: n, height: n, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        })
    };
    let (a, b) = (tex(), tex());
    let (av, bv) = (a.create_view(&Default::default()), b.create_view(&Default::default()));
    let module = d.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(
            std::env::var("FFT_WGSL_FILE").map(|f| std::fs::read_to_string(f).unwrap()).unwrap_or(fft::FFT_WGSL.into()).into(),
        ),
    });
    let pl = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bg = |axis: f32, src: &wgpu::TextureView, dst: &wgpu::TextureView| {
        let u = [n as f32, n.trailing_zeros() as f32, axis, -1.0f32];
        let buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&u),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pl.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(src) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(dst) },
            ],
        })
    };
    let rows = bg(0.0, &av, &bv);
    let cols = bg(1.0, &bv, &av);
    for (name, g) in [("rows", &rows), ("cols", &cols)] {
        let mut best = f64::MAX;
        for i in 0..iters + 2 {
            let t = Instant::now();
            let mut enc = d.create_command_encoder(&Default::default());
            for _ in 0..8 {
                let mut p = enc.begin_compute_pass(&Default::default());
                p.set_pipeline(&pl);
                p.set_bind_group(0, g, &[]);
                p.dispatch_workgroups(fft::groups(n, n), 1, 1);
            }
            ctx.queue.submit([enc.finish()]);
            let _ = d.poll(wgpu::PollType::wait_indefinitely());
            if i >= 2 {
                best = best.min(t.elapsed().as_secs_f64() * 1000.0 / 8.0);
            }
        }
        println!("{n}x{n} fft {name}: {best:.3} ms");
    }
}
