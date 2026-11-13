//! The Mult node's multiplier is rgb (the three-lane slider in
//! `app/src/rgbslider.rs` drives it). Two things have to hold on the GPU,
//! and neither is visible by eye:
//!
//! 1. each channel gets ITS OWN factor, exactly — a tint that is subtly the
//!    wrong channel still looks like a tint;
//! 2. multiplying alpha follows the WORKING SPACE's luma of the multiplier,
//!    so a tinted mult fades by the amount it brightens. That number moves
//!    when the working space does, which is the whole point of the
//!    working-space invariant and is easy to break by hardcoding Rec.709.
//!
//! Self-skips without a GPU (CI).

use engine::color::Colorspace;
use engine::compositor::Compositor;
use engine::decode::ImageInfo;
use engine::graph::{Graph, MultParams, NodeKind};
use engine::timing::GpuTimer;
use half::f16;

const W: u32 = 16;
const H: u32 = 8;

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP mult-rgb test: no adapter ({e})");
            None
        }
    }
}

/// A flat plate with every channel distinguishable.
fn upload(ctx: &engine::GpuContext, comp: &mut Compositor, px: [f32; 4]) {
    comp.ensure_reader(0, ImageInfo { width: W, height: H });
    let texels: Vec<[f16; 4]> = (0..W * H)
        .map(|_| {
            [
                f16::from_f32(px[0]),
                f16::from_f32(px[1]),
                f16::from_f32(px[2]),
                f16::from_f32(px[3]),
            ]
        })
        .collect();
    ctx.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &comp.reader_texture(0).unwrap().texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&texels),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(W * 8),
            rows_per_image: Some(H),
        },
        wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
    );
}

fn render(ctx: &engine::GpuContext, comp: &mut Compositor, graph: &Graph) -> [f32; 4] {
    assert!(comp.set_graph(graph), "fused shader failed to compile");
    let timer = GpuTimer::new(ctx);
    // Params are packed for the LIVE pipeline's layout, so the first submit
    // warms the compile and the second one is the measurement.
    let mut warm = ctx.device.create_command_encoder(&Default::default());
    comp.encode_comp(&mut warm, &timer);
    ctx.queue.submit([warm.finish()]);
    comp.write_params(graph);
    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    comp.encode_comp(&mut encoder, &timer);
    let padded = (W * 8).div_ceil(256) * 256;
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (padded * H) as u64,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: comp.output_texture().unwrap(),
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
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    let data = slice.get_mapped_range().expect("a mapped buffer");
    let halves: &[f16] = bytemuck::cast_slice(&data[..(W * 8) as usize]);
    let out = [
        halves[0].to_f32(),
        halves[1].to_f32(),
        halves[2].to_f32(),
        halves[3].to_f32(),
    ];
    drop(data);
    staging.unmap();
    out
}

fn graph_with(value: [f32; 3], alpha: bool, working: Colorspace) -> Graph {
    let mut g = Graph::default();
    g.working = working;
    g.nodes.insert(0, NodeKind::read(0, ""));
    g.nodes.insert(1, NodeKind::Mult(MultParams { value, alpha, matte: false }));
    g.edges.insert((1, 0), 0);
    g.view = Some(1);
    g
}

fn close(got: f32, want: f32, what: &str) {
    // f16 output: ~3 decimal digits is all the format has.
    assert!((got - want).abs() < 2e-3, "{what}: got {got}, want {want}");
}

#[test]
fn each_channel_gets_its_own_factor() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp, [0.5, 0.5, 0.5, 0.8]);

    let g = graph_with([2.0, 3.0, 0.5], false, Colorspace::default());
    let out = render(&ctx, &mut comp, &g);
    close(out[0], 1.0, "red");
    close(out[1], 1.5, "green");
    close(out[2], 0.25, "blue");
    // Colour only by default — that is what makes something hot.
    close(out[3], 0.8, "alpha untouched");

    // A neutral rgb multiplier is exactly the old scalar behaviour.
    let g = graph_with([4.0; 3], false, Colorspace::default());
    let out = render(&ctx, &mut comp, &g);
    for (i, c) in ["r", "g", "b"].iter().enumerate() {
        close(out[i], 2.0, c);
    }
}

#[test]
fn alpha_fades_by_the_working_space_luma() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp, [0.5, 0.5, 0.5, 0.8]);

    // Heavily green-weighted multiplier: the two working spaces disagree
    // about how bright it is, and the alpha has to follow whichever one the
    // comp is in.
    let value = [0.0, 2.0, 0.0];
    for working in [Colorspace::LINEAR_REC709, Colorspace::ACESCG] {
        let w = working.primaries.luma_weights();
        let g = graph_with(value, true, working);
        let out = render(&ctx, &mut comp, &g);
        let want = 0.8 * (w[0] * value[0] + w[1] * value[1] + w[2] * value[2]);
        close(out[3], want, "alpha");
        // Colour is unaffected by the alpha switch.
        close(out[1], 1.0, "green");
    }

    // ...and those two expectations are genuinely different numbers, or the
    // test would pass with the weights hardcoded.
    let r709 = Colorspace::LINEAR_REC709.primaries.luma_weights();
    let ap1 = Colorspace::ACESCG.primaries.luma_weights();
    assert!((r709[1] - ap1[1]).abs() > 0.02, "{r709:?} vs {ap1:?}");
}

/// Mult was a single float for a day. Setups written then still have to
/// load — silently, and as a neutral multiplier rather than a tint.
#[cfg(feature = "serde")]
#[test]
fn legacy_scalar_setups_still_load() {
    let m: MultParams = serde_json::from_str(r#"{"value": 2.5}"#).unwrap();
    assert_eq!(m.value, [2.5; 3]);
    assert!(!m.alpha);
    let m: MultParams = serde_json::from_str(r#"{"value": [1.0, 2.0, 3.0], "alpha": true}"#)
        .unwrap();
    assert_eq!(m.value, [1.0, 2.0, 3.0]);
    assert!(m.alpha);
    // Round-trips as rgb from here on.
    let s = serde_json::to_string(&m).unwrap();
    assert_eq!(serde_json::from_str::<MultParams>(&s).unwrap(), m);
}
