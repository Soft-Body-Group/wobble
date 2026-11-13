//! HDRCNN against the upstream graph.
//!
//! `hdrcnn_ref.rs` holds what the upstream network (`network.py`, written
//! out in NumPy by `tools/hdrcnn_reference.py` — TF1 no longer installs)
//! outputs for a 64x64 picture with a clipped highlight, run on the same
//! `.npz` files wobble reads. The CPU statement of the built `Net`
//! (`nn::reference`) pins the BUILDER; the GPU runner pins the KERNELS
//! (f32 storage: the bottleneck runs past f16); then the method's blend,
//! its pass-through and the tiling.
//!
//! Needs the weights: `WOBBLE_MODELS_DIR` (or `~/.wobble/models`) holding
//! the two `.npz` files. A model that is not there is skipped and says so;
//! nothing here downloads.

#[path = "hdrcnn_ref.rs"]
mod reference;

use engine::hdrcnn::Model;
use engine::nn::gpu::{Colour, Kernels, Runner};
use engine::nn::{reference as cpu, Net};
use half::f16;
use std::sync::Arc;

/// The network input `x` (0..1), planar `[3][h][w]` — the reference
/// script's formula.
fn picture(w: usize, h: usize) -> Vec<f32> {
    let mut a = vec![0.0f32; 3 * w * h];
    for y in 0..h {
        for x in 0..w {
            let d = ((x as f64 - 40.0).powi(2) + (y as f64 - 24.0).powi(2)).sqrt();
            for c in 0..3 {
                let mut v = 0.45 + 0.3 * (0.21 * x as f64 + 1.3 * c as f64).sin() * (0.17 * y as f64 - 0.4 * c as f64).cos();
                v += 1.4 * (-d * d / 60.0).exp() * (1.0 - 0.25 * c as f64);
                a[(c * h + y) * w + x] = v.clamp(0.0, 1.0) as f32;
            }
        }
    }
    a
}

fn net(model: Model) -> Option<Arc<Net>> {
    let n = engine::hdrcnn::net(model);
    if n.is_none() {
        eprintln!("SKIP {}: weights not in the models dir ({})", model.label(), engine::hdrcnn::load_error(model).unwrap_or_default());
    }
    n
}

fn expected(model: Model) -> ((&'static [f32], f64), (&'static [f32], f64)) {
    match model {
        Model::Regularized => ((&reference::REG_Y, reference::REG_Y_MEAN), (&reference::REG_FINAL, reference::REG_FINAL_MEAN)),
        Model::Original => ((&reference::ORIG_Y, reference::ORIG_Y_MEAN), (&reference::ORIG_FINAL, reference::ORIG_FINAL_MEAN)),
    }
}

/// Largest error over the sampled values, and the error of the mean.
fn compare(want: (&[f32], f64), out: &[f32]) -> (f32, f64) {
    let n = out.len();
    let worst = (0..reference::N)
        .map(|k| (out[(k as u64 * 2654435761 % n as u64) as usize] - want.0[k]).abs())
        .fold(0.0f32, f32::max);
    let m = out.iter().map(|v| *v as f64).sum::<f64>() / n as f64;
    (worst, (m - want.1).abs())
}

/// The blend, per pixel, over planar pictures.
fn blend(x: &[f32], y: &[f32], plane: usize) -> Vec<f32> {
    let mut out = vec![0.0; x.len()];
    for i in 0..plane {
        let xi = [x[i], x[plane + i], x[2 * plane + i]];
        let yi = [y[i], y[plane + i], y[2 * plane + i]];
        let f = cpu::hdr_blend(xi, yi);
        for c in 0..3 {
            out[c * plane + i] = f[c];
        }
    }
    out
}

#[test]
fn the_cpu_nets_match_the_upstream_graph() {
    let (w, h) = (reference::W, reference::H);
    let x = picture(w, h);
    for model in Model::ALL {
        let Some(net) = net(model) else { continue };
        let y = cpu::run(&net, &x, w, h);
        let (wy, wf) = expected(model);
        let (worst, mean) = compare(wy, &y);
        eprintln!("{model:?} cpu y: worst {worst:.2e}, mean {mean:.2e}");
        assert!(worst < 2e-3 && mean < 2e-4, "{model:?} y: worst {worst}, mean {mean}");
        let (worst, mean) = compare(wf, &blend(&x, &y, w * h));
        eprintln!("{model:?} cpu final: worst {worst:.2e}, mean {mean:.2e}");
        assert!(worst < 2e-3 && mean < 2e-4, "{model:?} final: worst {worst}, mean {mean}");
    }
}

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP hdrcnn gpu: no adapter ({e})");
            None
        }
    }
}

/// Run on the GPU: `px` is rgba (any alpha) at `(w, h)`, the result read
/// back as planar rgba f32 at the same size, and the tile count.
fn gpu_run(
    ctx: &engine::GpuContext,
    k: &mut Kernels,
    net: &Net,
    px: &[[f32; 4]],
    (w, h): (u32, u32),
    colour: Colour,
    budget: u64,
) -> (Vec<[f32; 4]>, usize) {
    let tex = |usage, label| {
        ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage,
            view_formats: &[],
        })
    };
    let input = tex(wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST, "in");
    let output = tex(wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC, "out");
    let half: Vec<[f16; 4]> = px.iter().map(|p| p.map(f16::from_f32)).collect();
    ctx.queue.write_texture(
        input.as_image_copy(),
        bytemuck::cast_slice(&half),
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 8), rows_per_image: Some(h) },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    let mut r = Runner::new(k, &ctx.queue, net, (w, h), 1, budget);
    r.set_colour(&ctx.queue, colour);
    let sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    let (iv, ov) = (input.create_view(&Default::default()), output.create_view(&Default::default()));
    let row = (w as u64 * 8).div_ceil(256) * 256;
    let readback = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: row * h as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    r.encode(k, &ctx.queue, &mut enc, &iv, &ov, &sampler);
    enc.copy_texture_to_buffer(
        output.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row as u32), rows_per_image: Some(h) },
        },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    ctx.queue.submit([enc.finish()]);
    readback.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    let data = readback.slice(..).get_mapped_range().unwrap();
    let mut out = Vec::with_capacity((w * h) as usize);
    for y in 0..h as usize {
        let r: &[[f16; 4]] = bytemuck::cast_slice(&data[y * row as usize..y * row as usize + w as usize * 8]);
        out.extend(r.iter().map(|p| p.map(f16::to_f32)));
    }
    drop(data);
    readback.unmap();
    (out, r.tile_count())
}

fn interleave(planar: &[f32], plane: usize, alpha: f32) -> Vec<[f32; 4]> {
    (0..plane).map(|i| [planar[i], planar[plane + i], planar[2 * plane + i], alpha]).collect()
}

fn planar(px: &[[f32; 4]]) -> Vec<f32> {
    (0..3).flat_map(|c| px.iter().map(move |p| p[c])).collect()
}

/// The GPU's raw log prediction against the upstream graph's.
#[test]
fn the_gpu_net_matches_the_upstream_graph() {
    let Some(ctx) = ctx() else { return };
    let (w, h) = (reference::W as u32, reference::H as u32);
    let plane = (w * h) as usize;
    let x = picture(w as usize, h as usize);
    let mut k = Kernels::new(ctx.device.clone(), false);
    for model in Model::ALL {
        let Some(net) = net(model) else { continue };
        let (out, tiles) = gpu_run(&ctx, &mut k, &net, &interleave(&x, plane, 1.0), (w, h), Colour::RAW, 1 << 30);
        assert_eq!(tiles, 1);
        let (worst, mean) = compare(expected(model).0, &planar(&out));
        eprintln!("{model:?} gpu y: worst {worst:.2e}, mean {mean:.2e}");
        // The output texture is half float: ~1e-3 on values of a few.
        assert!(worst < 1e-2 && mean < 1e-3, "{model:?}: worst {worst}, mean {mean}");
    }
}

/// The blend through the real colour path: a linear plate in, the method's
/// result out — and every pixel the mask leaves alone, bit for bit.
#[test]
fn the_blend_leaves_unclipped_pixels_as_they_came() {
    let Some(ctx) = ctx() else { return };
    let Some(net) = net(Model::Regularized) else { return };
    let (w, h) = (reference::W as u32, reference::H as u32);
    let plane = (w * h) as usize;
    let x = picture(w as usize, h as usize);
    let clip = 2.0f32;
    // The plate the net is handed `x` from: x² times the clip, half float.
    let plate: Vec<[f32; 4]> = interleave(&x, plane, 1.0)
        .iter()
        .map(|p| [p[0] * p[0] * clip, p[1] * p[1] * clip, p[2] * p[2] * clip, 1.0].map(|v| f16::from_f32(v).to_f32()))
        .collect();
    let id = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let colour = Colour { encode: true, to_709: id, from_709: id, clip };
    let mut k = Kernels::new(ctx.device.clone(), false);
    let (out, _) = gpu_run(&ctx, &mut k, &net, &plate, (w, h), colour, 1 << 30);
    let want = &reference::REG_FINAL;
    let (mut untouched, mut blended, mut worst) = (0, 0, 0.0f32);
    for (i, (o, p)) in out.iter().zip(&plate).enumerate() {
        let xe = [0, 1, 2].map(|c| (p[c] / clip).clamp(0.0, 1.0).sqrt());
        if cpu::hdr_alpha(xe) <= 0.0 {
            assert_eq!(o.map(f32::to_bits), p.map(f32::to_bits), "pixel {i} under the clip moved");
            untouched += 1;
        } else {
            blended += 1;
        }
    }
    // Where the reference sampled a blended pixel, the result is the
    // reference's times the clip.
    let n = 3 * plane;
    let pl = planar(&out);
    for (s, wv) in want.iter().enumerate() {
        let at = (s as u64 * 2654435761 % n as u64) as usize;
        worst = worst.max((pl[at] - wv * clip).abs() / (wv * clip).abs().max(0.05));
    }
    eprintln!("blend: {untouched} untouched, {blended} blended, worst relative {worst:.2e}");
    assert!(blended > 50 && untouched > 2000, "the picture should have both");
    assert!(worst < 2e-2, "the blend is off the reference by {worst}");
}

/// A frame cut into tiles is the whole frame, where the result shows
/// (the blended pixels), within what the halo cap allows.
#[test]
fn tiles_join_where_the_highlights_are() {
    let Some(ctx) = ctx() else { return };
    let Some(net) = net(Model::Regularized) else { return };
    let (w, h) = (300u32, 200u32);
    let plane = (w * h) as usize;
    // Highlights scattered over the frame, several straddling tile seams.
    let mut px = vec![[0.0f32; 4]; plane];
    for y in 0..h {
        for x in 0..w {
            let mut v = [0.0f32; 3];
            for c in 0..3 {
                v[c] = 0.2 + 0.15 * ((0.05 * x as f32 + c as f32).sin() + (0.07 * y as f32).cos());
            }
            for (cx, cy) in [(40.0, 50.0), (150.0, 100.0), (240.0, 60.0), (120.0, 170.0), (270.0, 180.0)] {
                let d2 = (x as f32 - cx).powi(2) + (y as f32 - cy).powi(2);
                for (c, vc) in v.iter_mut().enumerate() {
                    *vc += 3.0 * (-d2 / 200.0).exp() * (1.0 - 0.3 * c as f32);
                }
            }
            px[(y * w + x) as usize] = [v[0], v[1], v[2], 1.0];
        }
    }
    let id = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let colour = Colour { encode: true, to_709: id, from_709: id, clip: 1.0 };
    let mut k = Kernels::new(ctx.device.clone(), false);
    let (whole, n1) = gpu_run(&ctx, &mut k, &net, &px, (w, h), colour, 1 << 31);
    let small = net.values_per_tile(2 * net.halo + 32) / 4 * 16;
    let (tiled, n2) = gpu_run(&ctx, &mut k, &net, &px, (w, h), colour, small);
    assert_eq!(n1, 1, "the whole frame should fit one tile");
    assert!(n2 > 1, "the small budget should tile ({n2})");
    let mut worst = 0.0f32;
    for (a, b) in whole.iter().zip(&tiled) {
        for c in 0..3 {
            worst = worst.max((a[c] - b[c]).abs() / a[c].abs().max(1.0));
        }
    }
    eprintln!("{n2} tiles vs whole (halo {}): worst relative {worst:.2e}", net.halo);
    assert!(worst < 0.1, "tiles disagree with the whole frame by {worst}");
}

/// What a 2K frame costs (ignored: a timing, not a check). Best of five:
/// the GPU clocks up over the first runs.
#[test]
#[ignore]
fn timing_2k() {
    let Some(ctx) = ctx() else { return };
    let Some(net) = net(Model::Regularized) else { return };
    let (w, h) = (2048u32, 1080u32);
    let px: Vec<[f32; 4]> = (0..w * h).map(|i| [(i % 97) as f32 / 97.0, 0.5, 0.3, 1.0]).collect();
    let id = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let colour = Colour { encode: true, to_709: id, from_709: id, clip: 1.0 };
    let mut k = Kernels::new(ctx.device.clone(), false);
    let budget = (ctx.device.limits().max_storage_buffer_binding_size / 4).min(1 << 30);
    let mut best = f64::MAX;
    let mut tiles = 0;
    for _ in 0..5 {
        let t = std::time::Instant::now();
        let (_, n) = gpu_run(&ctx, &mut k, &net, &px, (w, h), colour, budget);
        best = best.min(t.elapsed().as_secs_f64());
        tiles = n;
    }
    eprintln!("2048x1080: {tiles} tiles, best {best:.2} s a frame (incl. upload and readback)");
}
