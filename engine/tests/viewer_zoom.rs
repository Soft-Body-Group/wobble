//! Magnified pixels must have hard edges.
//!
//! Zoomed past 1:1 a compositor is showing you PIXELS, and a bilinear ramp
//! between them is a lie about the data. Minified, the filtering is doing
//! real work and has to stay.
//!
//! Self-skips without a GPU (CI).

use engine::blit::Blit;
use engine::viewtf::{Eotf, ViewSettings, ViewTransform};
use half::f16;

const FMT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// 2x2 source blown up by this much.
const ZOOM: u32 = 16;

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP viewer_zoom test: no adapter ({e})");
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

    // No transform, no encode: this test is about SAMPLING, nothing else.
    let view = ViewSettings {
        transform: ViewTransform::None,
        eotf: Eotf::Linear,
        ..Default::default()
    };
    let mut blit = Blit::new(&ctx.device, FMT);
    blit.set_view(&ctx.device, &view);
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

/// A 2x2 checker: black, white / white, black.
fn checker() -> Vec<f16> {
    let mut v = Vec::new();
    for q in [0.0f32, 1.0, 1.0, 0.0] {
        v.extend([f16::from_f32(q), f16::from_f32(q), f16::from_f32(q), f16::ONE]);
    }
    v
}

#[test]
fn magnified_pixels_have_hard_edges() {
    let Some(ctx) = ctx() else { return };
    let (dw, dh) = (2 * ZOOM, 2 * ZOOM);
    let out = blit_to(&ctx, &checker(), (2, 2), (dw, dh));

    // Every output pixel must be exactly one of the two source values —
    // a bilinear ramp would put intermediate greys between the quadrants.
    let mut between = 0;
    for p in &out {
        if p[0] > 0.02 && p[0] < 0.98 {
            between += 1;
        }
    }
    assert_eq!(between, 0, "{between} interpolated pixels — magnification is filtering");

    // And each quadrant is uniform and correct.
    let at = |x: u32, y: u32| out[(y * dw + x) as usize][0];
    for (qx, qy, want) in [(0, 0, 0.0f32), (1, 0, 1.0), (0, 1, 1.0), (1, 1, 0.0)] {
        for dy in 0..ZOOM {
            for dx in 0..ZOOM {
                let v = at(qx * ZOOM + dx, qy * ZOOM + dy);
                assert!(
                    (v - want).abs() < 0.02,
                    "quadrant ({qx},{qy}) px ({dx},{dy}) = {v}, want {want}"
                );
            }
        }
    }
}

/// Minification must still filter: snapping there would alias.
#[test]
fn minified_pixels_still_filter() {
    let Some(ctx) = ctx() else { return };
    // A 64-wide horizontal ramp squeezed into 8 pixels. With filtering the
    // result is smooth; with nearest it would be 8 sparse samples of the
    // ramp, which is a different (and much harsher) set of values.
    const SW: u32 = 64;
    let mut px = Vec::new();
    for _ in 0..2 {
        for x in 0..SW {
            let v = f16::from_f32(x as f32 / (SW - 1) as f32);
            px.extend([v, v, v, f16::ONE]);
        }
    }
    let out = blit_to(&ctx, &px, (SW, 2), (8, 2));
    // Whatever the filtering, the ramp must still be monotonic across the
    // row and must not reach the extremes it would if it were point-sampled
    // at the very first and last texels.
    for x in 1..8 {
        assert!(
            out[x][0] >= out[x - 1][0] - 1e-3,
            "minified ramp not monotonic at {x}: {:?}",
            &out[..8]
        );
    }
    assert!(out[0][0] > 0.0, "first minified sample should average, not point-sample");
}

/// At exactly 1:1 the two paths agree, so nothing shifts as you cross the
/// threshold — a half-texel offset here would show up as the image jumping.
#[test]
fn one_to_one_is_unchanged() {
    let Some(ctx) = ctx() else { return };
    let out = blit_to(&ctx, &checker(), (2, 2), (2, 2));
    let want = [0.0f32, 1.0, 1.0, 0.0];
    for (i, w) in want.iter().enumerate() {
        assert!((out[i][0] - w).abs() < 0.02, "1:1 px {i} = {}, want {w}", out[i][0]);
    }
}
