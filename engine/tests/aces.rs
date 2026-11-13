//! The ACES 1.3 shader must agree with the Rust mirror in
//! `engine::viewtf::aces`.
//!
//! They are two independent transcriptions of the same ~200 lines of
//! reference CTL. Nothing on screen tells you when they drift — the picture
//! stays plausible — so the only thing that catches it is running both over
//! the same pixels and comparing, which is what this does.
//!
//! Self-skips without a GPU (CI).

use engine::blit::Blit;
use engine::viewtf::{aces, Eotf, Gamut, ViewSettings, ViewTransform};
use half::f16;

const W: u32 = 64;
const H: u32 = 32;
/// Float target: the point is to compare the transform, not an 8-bit encode.
const FMT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP aces test: no adapter ({e})");
            None
        }
    }
}

/// Hue around x, exposure down y — saturated, near-neutral and far above
/// diffuse white all in one pass, which is where a tone curve goes wrong.
fn plate() -> (Vec<f16>, Vec<[f32; 3]>) {
    let mut px = Vec::with_capacity((W * H * 4) as usize);
    let mut lin = Vec::with_capacity((W * H) as usize);
    for y in 0..H {
        let e = 0.002 * (1.0f32 + y as f32 / H as f32 * 6.0).powi(4);
        for x in 0..W {
            let t = x as f32 / (W - 1) as f32 * std::f32::consts::TAU;
            let c = [
                e * (0.5 + 0.5 * t.cos()),
                e * (0.5 + 0.5 * (t + 2.094).cos()),
                e * (0.5 + 0.5 * (t + 4.189).cos()),
            ];
            lin.push(c);
            px.extend([
                f16::from_f32(c[0]),
                f16::from_f32(c[1]),
                f16::from_f32(c[2]),
                f16::ONE,
            ]);
        }
    }
    (px, lin)
}

fn run(ctx: &engine::GpuContext, view: &ViewSettings) -> Vec<[f32; 3]> {
    let (px, _) = plate();
    let desc = |usage| wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FMT,
        usage,
        view_formats: &[],
    };
    let src = ctx.device.create_texture(&desc(
        wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
    ));
    ctx.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &src,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&px),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(W * 8),
            rows_per_image: Some(H),
        },
        wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
    );
    let dst = ctx.device.create_texture(&desc(
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
    ));

    let mut blit = Blit::new(&ctx.device, FMT);
    blit.set_view(&ctx.device, view);
    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    blit.encode(
        &ctx.device,
        &ctx.queue,
        &mut encoder,
        &src.create_view(&Default::default()),
        &dst.create_view(&Default::default()),
        None,
        [0.0, 0.0, W as f32, H as f32],
        false,
        view,
    );

    let padded = (W * 8).div_ceil(256) * 256;
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (padded * H) as u64,
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
                rows_per_image: Some(H),
            },
        },
        wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
    );
    ctx.queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().unwrap().unwrap();
    let view_bytes = staging.slice(..).get_mapped_range().expect("a mapped buffer");
    let mut out = Vec::with_capacity((W * H) as usize);
    for y in 0..H as usize {
        let row: &[f16] = bytemuck::cast_slice(
            &view_bytes[y * padded as usize..y * padded as usize + (W * 8) as usize],
        );
        for c in row.chunks_exact(4) {
            out.push([c[0].to_f32(), c[1].to_f32(), c[2].to_f32()]);
        }
    }
    out
}

fn settings(working: Gamut, display: Gamut) -> ViewSettings {
    ViewSettings {
        transform: ViewTransform::Aces13,
        working,
        display,
        // Linear out: compare the TRANSFORM, not a display encode.
        eotf: Eotf::Linear,
        ..Default::default()
    }
}

#[test]
fn shader_matches_the_cpu_mirror() {
    let Some(ctx) = ctx() else { return };
    let (_, lin) = plate();
    let out = run(&ctx, &settings(Gamut::AcesCg, Gamut::Rec709));

    let mut worst = 0.0f32;
    let mut lit = 0;
    for (o, c) in out.iter().zip(&lin) {
        let want = aces::apply(*c, engine::color::Primaries::Ap1, Gamut::Rec709);
        for i in 0..3 {
            worst = worst.max((o[i] - want[i]).abs());
        }
        if want[1] > 0.02 {
            lit += 1;
        }
    }
    // f16 in and f16 out, through a log-log spline: a few thousandths is the
    // storage floor, not slack.
    assert!(worst < 5e-3, "shader disagrees with the mirror by {worst}");
    // Not vacuous: the plate has to actually reach the lit part of the curve.
    assert!(lit > 400, "plate barely leaves black ({lit} px)");
}

#[test]
fn shader_matches_the_mirror_in_every_display_gamut() {
    let Some(ctx) = ctx() else { return };
    let (_, lin) = plate();
    for display in [Gamut::Rec709, Gamut::P3D65, Gamut::Rec2020] {
        let out = run(&ctx, &settings(Gamut::AcesCg, display));
        let mut worst = 0.0f32;
        for (o, c) in out.iter().zip(&lin) {
            let want = aces::apply(*c, engine::color::Primaries::Ap1, display);
            for i in 0..3 {
                worst = worst.max((o[i] - want[i]).abs());
            }
        }
        assert!(worst < 5e-3, "{display:?}: shader off the mirror by {worst}");
    }
}

/// The working space reaches the shader through the blit uniform, not
/// through an assumption — so the same picture in two working spaces has to
/// come out the same, on the GPU as well as on the CPU.
#[test]
fn the_working_space_reaches_the_shader() {
    let Some(ctx) = ctx() else { return };
    let a = run(&ctx, &settings(Gamut::AcesCg, Gamut::Rec709));
    let b = run(&ctx, &settings(Gamut::Rec709, Gamut::Rec709));
    let differ = a
        .iter()
        .zip(&b)
        .filter(|(x, y)| (0..3).any(|i| (x[i] - y[i]).abs() > 0.01))
        .count();
    assert!(
        differ > 200,
        "treating the plate as ACEScg vs Rec.709 changed only {differ} px — \
         the working space is not reaching the shader"
    );
}
