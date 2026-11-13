//! The Upscale models against torch.
//!
//! `upscale_ref.rs` holds what the upstream architectures (spandrel, torch
//! float32) output for a small smooth picture, run on the same `.pth` files
//! wobble reads — generated once by `tools/upscale_reference.py`. Each model
//! is checked twice: the CPU statement of its `Net` (`nn::reference`), which
//! pins the ARCHITECTURE builder, and the GPU runner, which pins the
//! KERNELS — in f32 and f16 storage, whole-picture and tiled.
//!
//! Needs the weights: `WOBBLE_MODELS_DIR` (or `~/.wobble/models`) holding
//! the three files. A model that is not there is skipped and says so;
//! nothing here downloads.

#[path = "upscale_ref.rs"]
mod reference;

use engine::nn::gpu::{Kernels, Runner};
use engine::nn::{reference as cpu, Net};
use engine::upscale::Model;
use half::f16;
use std::sync::Arc;

fn picture(w: usize, h: usize) -> Vec<f32> {
    let mut a = vec![0.0f32; 3 * w * h];
    for c in 0..3 {
        for y in 0..h {
            for x in 0..w {
                a[(c * h + y) * w + x] = (0.5
                    + 0.4 * (0.9 * x as f64 + 1.7 * c as f64).sin() * (0.7 * y as f64 - 0.3 * c as f64).cos())
                    as f32;
            }
        }
    }
    a
}

fn net(model: Model) -> Option<Arc<Net>> {
    let n = engine::upscale::net(model);
    if n.is_none() {
        eprintln!(
            "SKIP {}: weights not in the models dir ({})",
            model.label(),
            engine::upscale::load_error(model).unwrap_or_default()
        );
    }
    n
}

fn expected(model: Model) -> (&'static [f32], f64) {
    match model {
        Model::Compact => (&reference::COMPACT, reference::COMPACT_MEAN),
        Model::X4Plus => (&reference::X4PLUS, reference::X4PLUS_MEAN),
        Model::Span => (&reference::SPAN, reference::SPAN_MEAN),
    }
}

/// Largest error over the sampled values, and the error of the mean.
fn compare(model: Model, out: &[f32]) -> (f32, f64) {
    let (want, mean) = expected(model);
    let n = out.len();
    let worst = (0..reference::N)
        .map(|k| (out[(k as u64 * 2654435761 % n as u64) as usize] - want[k]).abs())
        .fold(0.0f32, f32::max);
    let m = out.iter().map(|v| *v as f64).sum::<f64>() / n as f64;
    (worst, (m - mean).abs())
}

#[test]
fn the_cpu_nets_match_torch() {
    let (w, h) = (reference::W, reference::H);
    for model in Model::ALL {
        let Some(net) = net(model) else { continue };
        let out = cpu::run(&net, &picture(w, h), w, h);
        let (worst, mean) = compare(model, &out);
        eprintln!("{model:?} cpu: worst {worst:.2e}, mean {mean:.2e}");
        assert!(worst < 2e-4 && mean < 2e-5, "{model:?}: worst {worst}, mean {mean}");
    }
}

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP upscale GPU test: no adapter ({e})");
            None
        }
    }
}

fn tex(ctx: &engine::GpuContext, w: u32, h: u32, usage: wgpu::TextureUsages) -> wgpu::Texture {
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
}

/// Run a net on the GPU over `rgb` (planar, 0..1), alpha 1, raw values;
/// returns planar rgb at `out_scale` and the runner's tile count.
fn gpu_run(
    ctx: &engine::GpuContext,
    k: &mut Kernels,
    net: &Net,
    rgb: &[f32],
    (w, h): (usize, usize),
    out_scale: u32,
    budget: u64,
    submit_macs: u64,
) -> (Vec<f32>, usize) {
    let input = tex(ctx, w as u32, h as u32, wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST);
    let px: Vec<[f16; 4]> = (0..w * h)
        .map(|i| [rgb[i], rgb[w * h + i], rgb[2 * w * h + i], 1.0].map(f16::from_f32))
        .collect();
    ctx.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &input,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&px),
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w as u32 * 8), rows_per_image: Some(h as u32) },
        wgpu::Extent3d { width: w as u32, height: h as u32, depth_or_array_layers: 1 },
    );
    let (ow, oh) = (w as u32 * out_scale, h as u32 * out_scale);
    let output = tex(ctx, ow, oh, wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC);
    let sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    let mut runner = Runner::new(k, &ctx.queue, net, (w as u32, h as u32), out_scale, budget);
    runner.submit_macs = submit_macs;
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    runner.encode(
        k,
        &ctx.queue,
        &mut enc,
        &input.create_view(&Default::default()),
        &output.create_view(&Default::default()),
        &sampler,
    );
    let padded = (ow * 8).div_ceil(256) * 256;
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (padded * oh) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    enc.copy_texture_to_buffer(
        output.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(padded), rows_per_image: Some(oh) },
        },
        wgpu::Extent3d { width: ow, height: oh, depth_or_array_layers: 1 },
    );
    ctx.queue.submit([enc.finish()]);
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    let data = slice.get_mapped_range().expect("mapped");
    let mut out = vec![0.0f32; 3 * (ow * oh) as usize];
    for y in 0..oh as usize {
        let row: &[f16] = bytemuck::cast_slice(&data[y * padded as usize..][..ow as usize * 8]);
        for x in 0..ow as usize {
            for c in 0..3 {
                out[(c * oh as usize + y) * ow as usize + x] = row[x * 4 + c].to_f32();
            }
        }
    }
    drop(data);
    staging.unmap();
    (out, runner.tile_count())
}

/// The input as the GPU sees it: through a half-float texture.
fn quantised(rgb: &[f32]) -> Vec<f32> {
    rgb.iter().map(|v| f16::from_f32(*v).to_f32()).collect()
}

#[test]
fn the_gpu_nets_match_torch_in_f32_and_f16() {
    let Some(ctx) = ctx() else { return };
    let (w, h) = (reference::W, reference::H);
    let rgb = picture(w, h);
    for half in [false, true] {
        let mut k = Kernels::new(ctx.device.clone(), half);
        if half && !k.half {
            eprintln!("SKIP f16: the device has no SHADER_F16");
            continue;
        }
        for model in Model::ALL {
            let Some(net) = net(model) else { continue };
            let (out, tiles) = gpu_run(&ctx, &mut k, &net, &rgb, (w, h), 4, 1 << 30, u64::MAX);
            assert_eq!(tiles, 1);
            // Against the CPU net on the same (half-float) input: the
            // kernels alone. Then against torch: what a user gets.
            let mine = cpu::run(&net, &quantised(&rgb), w, h);
            let kern = out.iter().zip(&mine).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
            let (worst, mean) = compare(model, &out);
            eprintln!(
                "{model:?} gpu {}: vs cpu {kern:.2e}; vs torch worst {worst:.2e}, mean {mean:.2e}",
                if half { "f16" } else { "f32" }
            );
            let (tk, tw) = if half { (3e-2, 5e-2) } else { (3e-3, 1e-2) };
            assert!(kern < tk, "{model:?} kernels drift {kern}");
            assert!(worst < tw && mean < 2e-3, "{model:?}: worst {worst}, mean {mean}");
        }
    }
}

#[test]
fn tiles_join_without_seams_and_half_scale_is_the_decimated_grid() {
    let Some(ctx) = ctx() else { return };
    let Some(net) = net(Model::Compact) else { return };
    let mut k = Kernels::new(ctx.device.clone(), false);
    let (w, h) = (150, 90);
    let rgb = picture(w, h);
    let (whole, n1) = gpu_run(&ctx, &mut k, &net, &rgb, (w, h), 4, 1 << 30, u64::MAX);
    // A budget that only fits the smallest tile.
    let small = net.values_per_tile(net.halo * 2 + 32) / 4 * 16;
    // Tiled AND submitted in small pieces: the frame Windows' watchdog
    // would otherwise kill is the same frame.
    let (tiled, n2) = gpu_run(&ctx, &mut k, &net, &rgb, (w, h), 4, small, 1_000_000_000);
    assert_eq!(n1, 1);
    assert!(n2 > 4, "the small budget tiles ({n2})");
    let seam = whole.iter().zip(&tiled).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    eprintln!("tiled ({n2} tiles) vs whole: {seam:.2e}");
    assert!(seam < 1e-2, "tiles disagree with the whole picture by {seam}");

    // 2x: [1 3 3 1] of the 4x grid, premultiplied by alpha 1.
    let (half, _) = gpu_run(&ctx, &mut k, &net, &rgb, (w, h), 2, 1 << 30, u64::MAX);
    let (ow, oh, fw, fh) = (w * 2, h * 2, w * 4, h * 4);
    let wt = [0.125f32, 0.375, 0.375, 0.125];
    let mut worst = 0.0f32;
    for c in 0..3 {
        for y in 0..oh {
            for x in 0..ow {
                let mut acc = 0.0;
                for (j, wj) in wt.iter().enumerate() {
                    let gy = (2 * y + j).saturating_sub(1).min(fh - 1);
                    for (i, wi) in wt.iter().enumerate() {
                        let gx = (2 * x + i).saturating_sub(1).min(fw - 1);
                        acc += wi * wj * whole[(c * fh + gy) * fw + gx];
                    }
                }
                worst = worst.max((acc - half[(c * oh + y) * ow + x]).abs());
            }
        }
    }
    assert!(worst < 3e-3, "2x is not the decimated 4x: {worst}");
}

// ---------------------------------------------- the node's job, end to end

use engine::decode::ExrDecoder as _;
use engine::graph::{UpscaleModel, UpscaleScale};
use std::sync::atomic::AtomicBool;

const NW: u32 = 24;
const NH: u32 = 16;

/// Premultiplied plate: slow gradients (the model should leave them
/// nearly alone), alpha 1 on the left half and 0.5 on the right, and a 2x2
/// highlight at 4.0 on the left.
fn plate() -> Vec<[f32; 4]> {
    (0..(NW * NH) as usize)
        .map(|i| {
            let (x, y) = (i % NW as usize, i / NW as usize);
            let a = if x < NW as usize / 2 { 1.0 } else { 0.5 };
            let hot = (5..7).contains(&x) && (7..9).contains(&y);
            let smooth = |k: usize| {
                let (fx, fy) = (x as f32 / NW as f32, y as f32 / NH as f32);
                0.15 + 0.6 * [fx, fy, 0.5 * (fx + fy)][k]
            };
            let c = |k: usize| if hot { 4.0 } else { smooth(k) };
            [c(0) * a, c(1) * a, c(2) * a, a]
        })
        .collect()
}

fn read_exr(path: &std::path::Path) -> (Vec<[f32; 4]>, u32, u32) {
    let dec = engine::decode::AnyDecoder::default();
    let info = dec.read_header(path).unwrap();
    let stride = info.row_stride_f16();
    let mut buf = vec![f16::ZERO; stride * info.height as usize];
    dec.decode_into(path, info, stride, &mut buf).unwrap();
    let mut out = Vec::new();
    for y in 0..info.height as usize {
        for x in 0..info.width as usize {
            out.push(std::array::from_fn(|c| buf[y * stride + x * 4 + c].to_f32()));
        }
    }
    (out, info.width, info.height)
}

#[test]
fn the_job_upscales_frames_on_disk_keeping_alpha_and_highlights() {
    let Some(ctx) = ctx() else { return };
    if net(UpscaleModel::Compact).is_none() {
        return;
    }
    let dir = std::env::temp_dir().join(format!("wobble-upscale-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("Input")).unwrap();
    let src = plate();
    let px: Vec<f16> = src.iter().flat_map(|c| c.map(f16::from_f32)).collect();
    let frames: Vec<_> = (1001..1003)
        .map(|t| {
            let f = dir.join("Input").join(format!("f_{t:05}.exr"));
            engine::bakes::write_rgba_f16_exr_tagged(&f, NW as usize, NH as usize, NW as usize * 4, &px, None).unwrap();
            f
        })
        .collect();
    for scale in [UpscaleScale::X4, UpscaleScale::X2] {
        let out = dir.join(format!("Upscaled{}", scale.factor()));
        let job = engine::upscale::Job { model: UpscaleModel::Compact, frames: frames.clone(), out: out.clone(), scale: scale.factor() };
        let mut seen = Vec::new();
        engine::upscale::run(ctx.device.clone(), ctx.queue.clone(), &job, &mut |a, b| seen.push((a, b)), &AtomicBool::new(false))
            .unwrap();
        assert_eq!(seen.last(), Some(&(2, 2)));
        assert!(engine::bakes::is_baked(&out), "the done-marker is written last");
        let (img, w, h) = read_exr(&out.join("f_01002.exr"));
        let s = scale.factor();
        assert_eq!((w, h), (NW * s, NH * s));
        let at = |x: u32, y: u32| img[(y * w + x) as usize];
        // Alpha: 1 on the left, 0.5 on the right.
        assert!((at(2 * s, 2 * s)[3] - 1.0).abs() < 1e-3, "{:?}", at(2 * s, 2 * s));
        assert!((at(w - 2 * s, h - 2 * s)[3] - 0.5).abs() < 1e-3, "{:?}", at(w - 2 * s, h - 2 * s));
        // Colour: premultiplied, close to the plate where it is smooth.
        let mut err = 0.0f32;
        let mut n = 0;
        for y in 2..NH - 2 {
            for x in 2..NW - 2 {
                if (4..8).contains(&x) && (6..10).contains(&y) {
                    continue;
                }
                let o = at(x * s + s / 2, y * s + s / 2);
                let i = src[(y * NW + x) as usize];
                for c in 0..3 {
                    err += (o[c] - i[c]).abs();
                    n += 1;
                }
            }
        }
        let mean = err / n as f32;
        eprintln!("{scale:?}: mean |out - plate| {mean:.4}, highlight {:?}", at(6 * s, 8 * s));
        assert!(mean < 0.02, "{scale:?} drifted from the plate by {mean}");
        // A highlight past 1 comes back past 1: nothing is clipped.
        assert!(at(6 * s, 8 * s)[0] > 1.5, "{scale:?} clipped the highlight: {:?}", at(6 * s, 8 * s));
    }
    // A cancelled run stops between frames and leaves no marker.
    let out = dir.join("Cancelled");
    let job = engine::upscale::Job { model: UpscaleModel::Compact, frames, out: out.clone(), scale: 4 };
    let r = engine::upscale::run(ctx.device.clone(), ctx.queue.clone(), &job, &mut |_, _| {}, &AtomicBool::new(true));
    assert!(r.is_err() && !engine::bakes::is_baked(&out));
    let _ = std::fs::remove_dir_all(&dir);
}
