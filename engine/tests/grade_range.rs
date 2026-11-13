//! Headless integration: does a Grade actually push pixels HOT?
//!
//! The comp is scene-linear and the view transform lives at the blit, so
//! "it looks like it stops getting brighter" is exactly what a display
//! transform is for. This test reads the OUTPUT TEXTURE — before any of
//! that — so the question can be answered about the pixels rather than
//! about the picture. Self-skips without a GPU (CI).

use engine::compositor::Compositor;
use engine::decode::ImageInfo;
use engine::graph::{
    Graph, GradeParams, MatteChannel, MultParams, NodeKind,
    ToMatteParams,
};
use engine::timing::GpuTimer;
use half::f16;

const W: u32 = 16;
const H: u32 = 8;
/// Bigger canvas for the noise statistics: a 16x8 sample says nothing about
/// whether a field fills its range.
#[allow(dead_code)]
const W2: u32 = 256;
#[allow(dead_code)]
const H2: u32 = 256;

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP grade-range test: no adapter ({e})");
            None
        }
    }
}

fn upload(
    ctx: &engine::GpuContext,
    comp: &mut Compositor,
    slot: usize,
    v: f32,
    w: u32,
    h: u32,
) {
    let info = ImageInfo { width: w, height: h };
    comp.ensure_reader(slot, info);
    let px: Vec<[f16; 4]> = (0..w * h)
        .map(|_| [f16::from_f32(v), f16::from_f32(v), f16::from_f32(v), f16::ONE])
        .collect();
    ctx.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &comp.reader_texture(slot).unwrap().texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&px),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(w * 8),
            rows_per_image: Some(h),
        },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
}

fn render(ctx: &engine::GpuContext, comp: &mut Compositor, graph: &Graph) -> [f32; 4] {
    assert!(comp.set_graph(graph), "fused shader failed to compile");
    let timer = GpuTimer::new(ctx);
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

/// Render whatever `g` says and hand back min/max/mean of the red channel
/// over the whole frame.
#[allow(dead_code)]
fn stats(
    ctx: &engine::GpuContext,
    comp: &mut Compositor,
    graph: &Graph,
    w: u32,
    h: u32,
) -> (f32, f32, f32) {
    assert!(comp.set_graph(graph), "fused shader failed to compile");
    let timer = GpuTimer::new(ctx);
    let mut warm = ctx.device.create_command_encoder(&Default::default());
    comp.encode_comp(&mut warm, &timer);
    ctx.queue.submit([warm.finish()]);
    comp.write_params(graph);
    let mut encoder = ctx.device.create_command_encoder(&Default::default());
    comp.encode_comp(&mut encoder, &timer);
    let padded = (w * 8).div_ceil(256) * 256;
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (padded * h) as u64,
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
                rows_per_image: Some(h),
            },
        },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    ctx.queue.submit([encoder.finish()]);
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    let data = slice.get_mapped_range().expect("a mapped buffer");
    let (mut lo, mut hi, mut sum) = (f32::MAX, f32::MIN, 0.0f64);
    for y in 0..h as usize {
        let row: &[f16] =
            bytemuck::cast_slice(&data[y * padded as usize..][..(w * 8) as usize]);
        for x in 0..w as usize {
            let v = row[x * 4].to_f32();
            lo = lo.min(v);
            hi = hi.max(v);
            sum += v as f64;
        }
    }
    drop(data);
    staging.unmap();
    (lo, hi, (sum / (w * h) as f64) as f32)
}
/// ToMatte turns any image into coverage. The channel pick has to be exact
/// — a matte that is subtly the wrong channel looks plausible and grades
/// wrong everywhere.
#[test]
fn to_matte_picks_the_channel_it_says() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    // A plate where every channel is distinguishable.
    let info = ImageInfo { width: W, height: H };
    comp.ensure_reader(0, info);
    let px: Vec<[f16; 4]> = (0..W * H)
        .map(|_| {
            [
                f16::from_f32(0.2),
                f16::from_f32(0.4),
                f16::from_f32(0.6),
                f16::from_f32(0.8),
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
        bytemuck::cast_slice(&px),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(W * 8),
            rows_per_image: Some(H),
        },
        wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
    );

    let mut g = Graph::default();
    g.nodes.insert(0, NodeKind::read(0, ""));
    g.nodes.insert(1, NodeKind::ToMatte(ToMatteParams::default()));
    g.edges.insert((1, 0), 0);
    g.view = Some(1);

    let luma = 0.2126 * 0.2 + 0.7152 * 0.4 + 0.0722 * 0.6;
    for (ch, want) in [
        (MatteChannel::R, 0.2),
        (MatteChannel::G, 0.4),
        (MatteChannel::B, 0.6),
        (MatteChannel::A, 0.8),
        (MatteChannel::Luma, luma),
    ] {
        if let Some(NodeKind::ToMatte(t)) = g.nodes.get_mut(&1) {
            t.channel = ch;
        }
        let out = render(&ctx, &mut comp, &g);
        assert!(
            (out[0] - want).abs() < 0.01,
            "{}: got {out:?}, wanted {want}",
            ch.name()
        );
        // Coverage rides in rgb like every other mask source, so a consumer
        // reading .x and a viewer showing it agree.
        assert!((out[1] - out[0]).abs() < 1e-3 && (out[2] - out[0]).abs() < 1e-3);
    }

    // It is a MASK output, which is what lets it reach a matte input.
    assert_eq!(
        g.nodes[&1].output_kind(),
        engine::graph::PortKind::Mask,
        "ToMatte must present as a mask or it cannot be wired to one"
    );

    // Invert, and the range controls for a source that is not already 0..1.
    if let Some(NodeKind::ToMatte(t)) = g.nodes.get_mut(&1) {
        t.channel = MatteChannel::A;
        t.invert = true;
    }
    let out = render(&ctx, &mut comp, &g);
    assert!((out[0] - 0.2).abs() < 0.01, "invert: got {out:?}, wanted 0.2");
    if let Some(NodeKind::ToMatte(t)) = g.nodes.get_mut(&1) {
        t.invert = false;
        t.lo = 0.6;
        t.hi = 1.0;      // 0.8 sits halfway up this window
    }
    let out = render(&ctx, &mut comp, &g);
    assert!((out[0] - 0.5).abs() < 0.01, "range: got {out:?}, wanted 0.5");
    // Out-of-window values clamp rather than running past coverage 1.
    if let Some(NodeKind::ToMatte(t)) = g.nodes.get_mut(&1) {
        t.lo = 0.0;
        t.hi = 0.4;
    }
    let out = render(&ctx, &mut comp, &g);
    assert!((out[0] - 1.0).abs() < 0.01, "clamp: got {out:?}, wanted 1.0");
}

/// MatteIn is ToMatte's mirror: it copies a matte back INTO the stream as
/// alpha, colour untouched. Together they round-trip, which is the property
/// that makes them usable as a pair.
#[test]
fn matte_in_copies_coverage_into_alpha() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    let info = ImageInfo { width: W, height: H };
    comp.ensure_reader(0, info);
    let px: Vec<[f16; 4]> = (0..W * H)
        .map(|_| {
            [
                f16::from_f32(0.25),
                f16::from_f32(0.5),
                f16::from_f32(0.75),
                f16::from_f32(0.4),
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
        bytemuck::cast_slice(&px),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(W * 8),
            rows_per_image: Some(H),
        },
        wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
    );

    let mut g = Graph::default();
    g.nodes.insert(0, NodeKind::read(0, ""));
    // Take the plate's BLUE as a matte, then push it back in as alpha.
    g.nodes.insert(
        1,
        NodeKind::ToMatte(ToMatteParams {
            channel: MatteChannel::B,
            ..Default::default()
        }),
    );
    g.edges.insert((1, 0), 0);
    g.nodes.insert(2, NodeKind::MatteIn);
    g.edges.insert((2, 0), 0); // colour from the plate
    g.edges.insert((2, 1), 1); // alpha from the matte
    g.view = Some(2);

    let out = render(&ctx, &mut comp, &g);
    assert!((out[0] - 0.25).abs() < 0.01, "colour must pass through: {out:?}");
    assert!((out[1] - 0.5).abs() < 0.01, "colour must pass through: {out:?}");
    assert!((out[2] - 0.75).abs() < 0.01, "colour must pass through: {out:?}");
    assert!((out[3] - 0.75).abs() < 0.01, "alpha should be the matte: {out:?}");

    // An unwired matte means opaque, the same convention as every other
    // mask input — not a transparent frame.
    g.edges.remove(&(2, 1));
    let out = render(&ctx, &mut comp, &g);
    assert!((out[3] - 1.0).abs() < 0.01, "unwired matte should read opaque: {out:?}");

    // It hands back an IMAGE, so it can go on down the chain.
    assert_eq!(g.nodes[&2].output_kind(), engine::graph::PortKind::Image);
}

/// The Mult node: one number, applied through a matte. Colour is multiplied
/// and alpha is not, unless asked — the default is for making something hot,
/// which is deliberately not the same as fading it.
#[test]
fn mult_scales_colour_through_its_matte() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp, 0, 0.5, W, H);

    let mut g = Graph::default();
    g.nodes.insert(0, NodeKind::read(0, ""));
    g.nodes.insert(1, NodeKind::Mult(MultParams { value: [8.0; 3], ..Default::default() }));
    g.edges.insert((1, 0), 0);
    g.view = Some(1);

    // Unmasked: the whole image takes the multiply, alpha untouched.
    let px = render(&ctx, &mut comp, &g);
    assert!((px[0] - 4.0).abs() < 0.02, "colour: got {px:?}, wanted 4.0");
    assert_eq!(px[3], 1.0, "alpha must stay put by default");

    // With alpha on it is a fade instead: everything scales together.
    if let Some(NodeKind::Mult(m)) = g.nodes.get_mut(&1) {
        m.value = [0.5; 3];
        m.alpha = true;
    }
    let px = render(&ctx, &mut comp, &g);
    assert!((px[0] - 0.25).abs() < 0.01, "colour: got {px:?}");
    // With an rgb multiplier alpha fades by its working-space luma, which
    // for a neutral 0.5 is still 0.5.
    assert!((px[3] - 0.5).abs() < 0.01, "alpha should follow when asked: {px:?}");

    // Matted by a black mask: the multiply is fully held off, so the pixels
    // come back exactly as they went in.
    if let Some(NodeKind::Mult(m)) = g.nodes.get_mut(&1) {
        m.value = [8.0; 3];
        m.alpha = false;
    }
    g.nodes.insert(
        2,
        NodeKind::Grid(engine::graph::GridParams {
            size: (W, H),
            inherit: false,
            spacing: [4096.0, 4096.0],
            width: 0.0,
            line: [1.0; 4],
            back: [0.0, 0.0, 0.0, 1.0],
            offset: [0.0, 0.0],
        }),
    );
    g.edges.insert((1, 1), 2);
    let px = render(&ctx, &mut comp, &g);
    assert!((px[0] - 0.5).abs() < 0.01, "black matte should hold it off: {px:?}");

    // A white matte lets the whole multiply through.
    if let Some(NodeKind::Grid(gr)) = g.nodes.get_mut(&2) {
        gr.back = [1.0; 4];
    }
    let px = render(&ctx, &mut comp, &g);
    assert!((px[0] - 4.0).abs() < 0.02, "white matte should pass it: {px:?}");

    // The value is a uniform: dialling it never recompiles.
    let h = g.topology_hash();
    if let Some(NodeKind::Mult(m)) = g.nodes.get_mut(&1) {
        m.value = [137.0, 4.0, 0.5];
        m.alpha = true;
    }
    assert_eq!(g.topology_hash(), h, "mult params must stay uniform edits");
}

/// A Grade must be able to take pixels well above 1.0 and keep going. If
/// this ever fails, something in the fused pass is clamping the comp to
/// display range — which would quietly destroy every highlight in the
/// scene-linear pipeline.
#[test]
fn grade_gain_pushes_pixels_far_above_one() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp, 0, 0.5, W, H);

    let mut g = Graph::default();
    g.nodes.insert(0, NodeKind::read(0, ""));
    g.nodes.insert(1, NodeKind::Grade(GradeParams::default()));
    g.edges.insert((1, 0), 0);
    g.view = Some(1);

    // Gain is a straight multiply on a 0.5 plate, so the answer is knowable
    // in advance at every stop.
    for gain in [1.0f32, 8.0, 64.0, 512.0] {
        if let Some(NodeKind::Grade(gr)) = g.nodes.get_mut(&1) {
            gr.gain = [gain; 3];
        }
        let px = render(&ctx, &mut comp, &g);
        let want = 0.5 * gain;
        assert!(
            (px[0] - want).abs() < want * 0.02,
            "gain {gain}: got {px:?}, wanted ~{want} — the comp is being clamped"
        );
        assert_eq!(px[3], 1.0, "alpha must not ride the colour gain");
    }

    // Lift and offset reach past 1.0 too, and gamma below 1 lifts further
    // rather than capping.
    if let Some(NodeKind::Grade(gr)) = g.nodes.get_mut(&1) {
        gr.gain = [1.0; 3];
        gr.offset = [4.0; 3];
        gr.lift = [2.0; 3];
    }
    let px = render(&ctx, &mut comp, &g);
    assert!((px[0] - 6.5).abs() < 0.1, "offset + lift: got {px:?}, wanted ~6.5");

    // The half-float output carries a genuinely large value intact: this is
    // the ceiling that actually exists (rgba16float), ~65504.
    if let Some(NodeKind::Grade(gr)) = g.nodes.get_mut(&1) {
        gr.offset = [0.0; 3];
        gr.lift = [0.0; 3];
        gr.gain = [20000.0; 3];
    }
    let px = render(&ctx, &mut comp, &g);
    assert!(px[0] > 9000.0, "half-float headroom: got {px:?}");
}

/// The Grade's overall saturation: 1 changes nothing, 0 is monochrome at the
/// pixel's own luminance, and past 1 pushes. It must be LUMINANCE-PRESERVING
/// with the working space's weights — the whole reason not to reach for an
/// HSV conversion.
#[test]
fn grade_saturation_pivots_about_luminance() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    let colour = [0.8f32, 0.3, 0.1];
    let info = ImageInfo { width: W, height: H };
    comp.ensure_reader(0, info);
    let px: Vec<[f16; 4]> = (0..W * H)
        .map(|_| {
            [
                f16::from_f32(colour[0]),
                f16::from_f32(colour[1]),
                f16::from_f32(colour[2]),
                f16::ONE,
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
        bytemuck::cast_slice(&px),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(W * 8),
            rows_per_image: Some(H),
        },
        wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
    );

    let build = |saturation: f32| {
        let mut g = Graph::default();
        g.nodes.insert(0, NodeKind::read(0, ""));
        g.nodes.insert(1, NodeKind::Grade(GradeParams { saturation, ..Default::default() }));
        g.edges.insert((1, 0), 0);
        g.view = Some(1);
        g
    };
    let w = Graph::default().working.primaries.luma_weights();
    let y = colour[0] * w[0] + colour[1] * w[1] + colour[2] * w[2];

    // 1.0 is a no-op.
    let one = render(&ctx, &mut comp, &build(1.0));
    for c in 0..3 {
        assert!((one[c] - colour[c]).abs() < 3e-3, "sat 1 changed channel {c}");
    }
    // 0.0 is flat grey AT the luminance.
    let zero = render(&ctx, &mut comp, &build(0.0));
    for c in 0..3 {
        assert!((zero[c] - y).abs() < 4e-3, "sat 0: channel {c} is {} not {y}", zero[c]);
    }
    // Half way, and past 1: still centred on the same luminance.
    for s in [0.5f32, 2.0] {
        let got = render(&ctx, &mut comp, &build(s));
        let gy = got[0] * w[0] + got[1] * w[1] + got[2] * w[2];
        assert!((gy - y).abs() < 5e-3, "sat {s} moved the luminance: {gy} vs {y}");
        for c in 0..3 {
            let want = y + (colour[c] - y) * s;
            assert!((got[c] - want).abs() < 5e-3, "sat {s}, channel {c}: {} vs {want}", got[c]);
        }
    }
}

/// The Grade's contrast: a power about the pivot. 1 changes nothing, the
/// pivot itself never moves, values either side spread apart (or gather) by
/// the exponent, and an old setup's defaults (1, 0.18) are a no-op.
#[test]
fn grade_contrast_turns_about_its_pivot() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    let build = |contrast: f32, pivot: f32| {
        let mut g = Graph::default();
        g.nodes.insert(0, NodeKind::read(0, ""));
        g.nodes.insert(
            1,
            NodeKind::Grade(GradeParams { contrast, pivot, ..Default::default() }),
        );
        g.edges.insert((1, 0), 0);
        g.view = Some(1);
        g
    };
    let d = GradeParams::default();
    assert_eq!((d.contrast, d.pivot), (1.0, 0.18), "neutral defaults");
    for v in [0.05f32, 0.18, 0.6, 2.5] {
        upload(&ctx, &mut comp, 0, v, W, H);
        let one = render(&ctx, &mut comp, &build(1.0, 0.18));
        assert!((one[0] - v).abs() < 2e-3 * v.max(1.0), "contrast 1 moved {v}: {}", one[0]);
        for (k, p) in [(1.5f32, 0.18f32), (0.6, 0.18), (2.0, 0.5)] {
            let got = render(&ctx, &mut comp, &build(k, p));
            let want = p * (v / p).powf(k);
            assert!(
                (got[0] - want).abs() < 4e-3 * want.max(1.0),
                "contrast {k} pivot {p} on {v}: {} vs {want}",
                got[0]
            );
        }
    }
}
