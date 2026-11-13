//! Headless integration for the Glow node: the mip-pyramid bloom through the
//! real Compositor. Verifies the pyramid stage runs, that light actually
//! spreads far beyond any single octave's radius, that `falloff` widens the
//! tail, and that the threshold keeps dim pixels out of it.
//! Self-skips without a GPU (CI).

use engine::compositor::Compositor;
use engine::decode::ImageInfo;
use engine::graph::{Graph, GlowMode, GlowParams, NodeKind};
use engine::timing::GpuTimer;
use half::f16;

const W: u32 = 256;
const H: u32 = 256;

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP glow test: no adapter ({e})");
            None
        }
    }
}

fn upload(ctx: &engine::GpuContext, comp: &mut Compositor, px: &[[f16; 4]]) {
    comp.ensure_reader(0, ImageInfo { width: W, height: H });
    ctx.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &comp.reader_texture(0).unwrap().texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(px),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(W * 8),
            rows_per_image: Some(H),
        },
        wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
    );
}

fn render(ctx: &engine::GpuContext, comp: &mut Compositor, graph: &Graph) -> Vec<[f32; 4]> {
    assert!(comp.set_graph(graph), "fused shader failed to compile");
    let timer = GpuTimer::new(ctx);
    // Warm-up encode: the pyramid textures are allocated inside encode_comp,
    // and write_params only reaches stages that already exist.
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
    let mut out = Vec::with_capacity((W * H) as usize);
    for y in 0..H as usize {
        let row = &data[y * padded as usize..][..(W * 8) as usize];
        let halves: &[f16] = bytemuck::cast_slice(row);
        for x in 0..W as usize {
            let p = &halves[x * 4..][..4];
            out.push([p[0].to_f32(), p[1].to_f32(), p[2].to_f32(), p[3].to_f32()]);
        }
    }
    drop(data);
    staging.unmap();
    out
}

/// A small bright square at the centre on a dim (sub-threshold) grey field.
fn plate() -> Vec<[f16; 4]> {
    let mut px = Vec::new();
    for y in 0..H as usize {
        for x in 0..W as usize {
            let hot = (x as i32 - 128).abs() < 4 && (y as i32 - 128).abs() < 4;
            let v = if hot { 16.0 } else { 0.2 };
            px.push([
                f16::from_f32(v),
                f16::from_f32(v),
                f16::from_f32(v),
                f16::ONE,
            ]);
        }
    }
    px
}

fn glow_graph(p: GlowParams) -> Graph {
    let mut g = Graph::default();
    g.nodes.insert(1, NodeKind::read(0, ""));
    g.nodes.insert(2, NodeKind::Glow(p));
    g.edges.insert((2, 0), 1);
    g.view = Some(2);
    g
}

#[test]
fn glow_spreads_far_and_respects_threshold() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 1);
    upload(&ctx, &mut comp, &plate());
    comp.ensure_output(W, H);

    let base = GlowParams {
        size: 120.0,
        falloff: 0.7,
        threshold: 1.0,
        knee: 0.5,
        intensity: 1.0,
        tint: [1.0, 1.0, 1.0],
        mode: GlowMode::Only,
    };
    let out = render(&ctx, &mut comp, &glow_graph(base));
    let row = 128 * W as usize;
    let at = |o: &Vec<[f32; 4]>, x: usize| o[row + x][0];

    // Core glows, and the halo is still alive 100px out — a radius no single
    // separable pass in this comp reaches without 300 taps per pixel.
    assert!(at(&out, 128) > 0.1, "core should glow, got {}", at(&out, 128));
    assert!(at(&out, 168) > 0.0005, "halo at 40px, got {}", at(&out, 168));
    assert!(at(&out, 228) > 0.0001, "halo at 100px, got {}", at(&out, 228));
    // Monotonic falloff away from the core (no octave banding / ringing).
    for x in [140usize, 160, 180, 200, 220] {
        assert!(
            at(&out, x) > at(&out, x + 10),
            "falloff must be monotonic: {} at {x} vs {} at {}",
            at(&out, x),
            at(&out, x + 10),
            x + 10
        );
    }

    // The dim field is below threshold: it must contribute essentially
    // nothing, so a far corner stays near black in glow-only mode.
    assert!(out[8 * W as usize + 8][0] < 0.02, "sub-threshold field leaked");

    // Falloff is the shape control: more weight on the wide octaves must
    // push relatively more energy into the far halo.
    let wide = render(
        &ctx,
        &mut comp,
        &glow_graph(GlowParams { falloff: 1.0, ..base }),
    );
    let ratio = |o: &Vec<[f32; 4]>| at(o, 228) / at(o, 130).max(1e-6);
    assert!(
        ratio(&wide) > ratio(&out) * 1.5,
        "falloff=1.0 should broaden the tail: {} vs {}",
        ratio(&wide),
        ratio(&out)
    );

    // Add mode returns the original plus the glow.
    let add = render(&ctx, &mut comp, &glow_graph(GlowParams { mode: GlowMode::Add, ..base }));
    let far = 8 * W as usize + 8;
    assert!(
        (add[far][0] - (0.2 + out[far][0])).abs() < 0.01,
        "add mode must be original + glow, got {}",
        add[far][0]
    );
    assert!((add[far][3] - 1.0).abs() < 0.01, "alpha must be the original's");
}
