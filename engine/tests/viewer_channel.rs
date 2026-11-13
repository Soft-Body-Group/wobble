//! Single-channel view: shift+R/G/B show one channel as grey.
//!
//! Self-skips without a GPU (CI).

use engine::blit::Blit;
use engine::viewtf::{Eotf, ViewSettings, ViewTransform};
use half::f16;

const FMT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP viewer_channel test: no adapter ({e})");
            None
        }
    }
}

/// Render a `sw x sh` plate into a `dw x dh` target through the blit.
fn blit_to(
    ctx: &engine::GpuContext,
    px: &[f16],
    (sw, sh): (u32, u32),
    (dw, dh): (u32, u32),
    channel: Option<usize>,
) -> Vec<[f32; 3]> {
    let desc = |w, h, usage| wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FMT,
        usage,
        view_formats: &[],
    };
    let src = ctx.device.create_texture(&desc(
        sw,
        sh,
        wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
    ));
    ctx.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &src,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(px),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(sw * 8),
            rows_per_image: Some(sh),
        },
        wgpu::Extent3d { width: sw, height: sh, depth_or_array_layers: 1 },
    );
    let dst = ctx.device.create_texture(&desc(
        dw,
        dh,
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
    ));

    // No transform, no encode: the channel pick is all that is under test.
    let view = ViewSettings {
        transform: ViewTransform::None,
        eotf: Eotf::Linear,
        ..Default::default()
    };
    let mut blit = Blit::new(&ctx.device, FMT);
    blit.set_view(&ctx.device, &view);
    blit.set_channel(channel);
    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    blit.encode(
        &ctx.device,
        &ctx.queue,
        &mut encoder,
        &src.create_view(&Default::default()),
        &dst.create_view(&Default::default()),
        None,
        [0.0, 0.0, dw as f32, dh as f32],
        false,
        &view,
    );

    let padded = (dw * 8).div_ceil(256) * 256;
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (padded * dh) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
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
                rows_per_image: Some(dh),
            },
        },
        wgpu::Extent3d { width: dw, height: dh, depth_or_array_layers: 1 },
    );
    ctx.queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().unwrap().unwrap();
    let bytes = staging.slice(..).get_mapped_range().expect("a mapped buffer");
    let mut out = Vec::with_capacity((dw * dh) as usize);
    for y in 0..dh as usize {
        let row: &[f16] =
            bytemuck::cast_slice(&bytes[y * padded as usize..y * padded as usize + (dw * 8) as usize]);
        for c in row.chunks_exact(4) {
            out.push([c[0].to_f32(), c[1].to_f32(), c[2].to_f32()]);
        }
    }
    out
}

#[test]
fn each_channel_shows_as_grey_and_none_shows_all() {
    let Some(ctx) = ctx() else { return };
    let px: Vec<f16> = [0.25f32, 0.5, 0.75, 1.0].iter().map(|v| f16::from_f32(*v)).collect();
    let near = |a: [f32; 3], b: [f32; 3]| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 2e-3);
    let all = blit_to(&ctx, &px, (1, 1), (4, 4), None)[5];
    assert!(near(all, [0.25, 0.5, 0.75]), "rgb view: {all:?}");
    for (c, v) in [0.25f32, 0.5, 0.75].into_iter().enumerate() {
        let got = blit_to(&ctx, &px, (1, 1), (4, 4), Some(c))[5];
        assert!(near(got, [v; 3]), "channel {c}: {got:?}, want {v} grey");
    }
}
