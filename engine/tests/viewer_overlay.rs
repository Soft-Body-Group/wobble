//! Dual view: two images in ONE target. The live pane's blit clears and
//! draws its half; the held pane's overlay blit draws the other half over
//! it and leaves the first alone. Regression: both used to share one Blit,
//! so the second draw's uniform and bind group won and its clear wiped the
//! first — only one pane ever showed, and on the wrong side.
//!
//! Self-skips without a GPU (CI).

use engine::blit::Blit;
use engine::viewtf::{Eotf, ViewSettings, ViewTransform};
use half::f16;

const FMT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

fn solid(ctx: &engine::GpuContext, v: [f32; 3]) -> wgpu::Texture {
    let tex = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FMT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let px: Vec<f16> = [v[0], v[1], v[2], 1.0].iter().map(|x| f16::from_f32(*x)).collect();
    ctx.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&px),
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(8), rows_per_image: Some(1) },
        wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
    );
    tex
}

#[test]
fn two_panes_each_keep_their_own_picture() {
    let ctx = match engine::GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("SKIP viewer_overlay test: no adapter ({e})");
            return;
        }
    };
    let (w, h) = (64u32, 16u32);
    let dst = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FMT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let dv = dst.create_view(&Default::default());
    let (a, b) = (solid(&ctx, [0.8, 0.1, 0.1]), solid(&ctx, [0.1, 0.1, 0.8]));
    let view = ViewSettings { transform: ViewTransform::None, eotf: Eotf::Linear, ..Default::default() };
    let mut live = Blit::new(&ctx.device, FMT);
    let mut held = Blit::new(&ctx.device, FMT);
    held.set_overlay(true);
    live.set_view(&ctx.device, &view);
    held.set_view(&ctx.device, &view);
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    // Live pane on the RIGHT, held on the LEFT: what 3 over the graph does.
    live.encode(
        &ctx.device, &ctx.queue, &mut enc,
        &b.create_view(&Default::default()), &dv, None,
        [32.0, 0.0, 32.0, 16.0], false, &view,
    );
    held.encode(
        &ctx.device, &ctx.queue, &mut enc,
        &a.create_view(&Default::default()), &dv, None,
        [0.0, 0.0, 32.0, 16.0], false, &view,
    );
    let padded = (w * 8).div_ceil(256) * 256;
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (padded * h) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    enc.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &dst,
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
    ctx.queue.submit([enc.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().unwrap().unwrap();
    let bytes = staging.slice(..).get_mapped_range().expect("a mapped buffer");
    let row: &[f16] = bytemuck::cast_slice(&bytes[8 * padded as usize..8 * padded as usize + (w * 8) as usize]);
    let px = |x: usize| [row[x * 4].to_f32(), row[x * 4 + 1].to_f32(), row[x * 4 + 2].to_f32()];
    let near = |p: [f32; 3], q: [f32; 3]| p.iter().zip(q).all(|(x, y)| (x - y).abs() < 2e-3);
    assert!(near(px(10), [0.8, 0.1, 0.1]), "left (held) pane: {:?}", px(10));
    assert!(near(px(50), [0.1, 0.1, 0.8]), "right (live) pane: {:?}", px(50));
}
