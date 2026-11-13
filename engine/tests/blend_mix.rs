//! Headless integration for the two composite additions: the Mix node's
//! crossfade, and Blend's `Replace` op plus the matte input that decides
//! where any blend lands. Self-skips without a GPU (CI).

use engine::compositor::Compositor;
use engine::decode::ImageInfo;
use engine::graph::{
    BlendOp, Graph, MatteChannel, MixParams, NodeKind, ToMatteParams,
};
use engine::timing::GpuTimer;
use half::f16;

const W: u32 = 16;
const H: u32 = 8;

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP blend/mix test: no adapter ({e})");
            None
        }
    }
}

fn upload(ctx: &engine::GpuContext, comp: &mut Compositor, slot: usize, v: [f32; 4]) {
    comp.ensure_reader(slot, ImageInfo { width: W, height: H });
    let px: Vec<[f16; 4]> = (0..W * H)
        .map(|_| {
            [
                f16::from_f32(v[0]),
                f16::from_f32(v[1]),
                f16::from_f32(v[2]),
                f16::from_f32(v[3]),
            ]
        })
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
            bytes_per_row: Some(W * 8),
            rows_per_image: Some(H),
        },
        wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
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

/// The whole point of Mix: one slider, all the way from one input to the
/// other, with the ends landing exactly on their input.
#[test]
fn mix_crossfades_between_its_two_inputs() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp, 0, [0.0, 0.2, 0.4, 1.0]);
    upload(&ctx, &mut comp, 1, [1.0, 0.6, 0.8, 1.0]);
    for (amount, want) in [
        (0.0f32, [0.0f32, 0.2, 0.4]),
        (1.0, [1.0, 0.6, 0.8]),
        (0.25, [0.25, 0.3, 0.5]),
        (0.5, [0.5, 0.4, 0.6]),
    ] {
        let mut g = Graph::default();
        g.nodes.insert(0, NodeKind::read(0, ""));
        g.nodes.insert(1, NodeKind::read(1, ""));
        g.nodes.insert(2, NodeKind::Mix(MixParams { amount }));
        g.edges.insert((2, 0), 0);
        g.edges.insert((2, 1), 1);
        g.view = Some(2);
        let got = render(&ctx, &mut comp, &g);
        for c in 0..3 {
            assert!(
                (got[c] - want[c]).abs() < 3e-3,
                "amount {amount}, channel {c}: got {} want {}",
                got[c],
                want[c]
            );
        }
    }
}

/// Replace ignores alpha entirely — that is what separates it from Over.
/// The comp is PREMULTIPLIED, so an A with zero alpha contributes nothing
/// under Over; Replace hands it over anyway, because coverage is the matte
/// input's job, not alpha's.
#[test]
fn replace_ignores_alpha_where_over_does_not() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    // A is fully transparent (premultiplied: no colour either); B is
    // opaque grey.
    upload(&ctx, &mut comp, 0, [0.0, 0.0, 0.0, 0.0]);
    upload(&ctx, &mut comp, 1, [0.5, 0.5, 0.5, 1.0]);
    let build = |op| {
        let mut g = Graph::default();
        g.nodes.insert(0, NodeKind::read(0, ""));
        g.nodes.insert(1, NodeKind::read(1, ""));
        g.nodes.insert(2, NodeKind::Blend { op, mix: 1.0, gain: [1.0, 1.0] });
        g.edges.insert((2, 0), 0);
        g.edges.insert((2, 1), 1);
        g.view = Some(2);
        g
    };
    let over = render(&ctx, &mut comp, &build(BlendOp::Over));
    assert!(
        (over[0] - 0.5).abs() < 3e-3 && (over[1] - 0.5).abs() < 3e-3,
        "a transparent A must leave B alone under Over, got {over:?}"
    );
    let replace = render(&ctx, &mut comp, &build(BlendOp::Replace));
    assert!(
        replace[0].abs() < 3e-3 && replace[3].abs() < 3e-3,
        "Replace should hand back A's pixels whatever its alpha, got {replace:?}"
    );
}

/// The matte input decides WHERE a blend lands: half coverage means half of
/// each, for Replace and for Over alike.
#[test]
fn the_matte_input_limits_any_blend() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp, 0, [1.0, 1.0, 1.0, 1.0]); // A: white
    upload(&ctx, &mut comp, 1, [0.0, 0.0, 0.0, 1.0]); // B: black
    upload(&ctx, &mut comp, 2, [0.5, 0.5, 0.5, 1.0]); // matte plate: 0.5 luma
    let mut g = Graph::default();
    g.nodes.insert(0, NodeKind::read(0, ""));
    g.nodes.insert(1, NodeKind::read(1, ""));
    g.nodes.insert(2, NodeKind::read(2, ""));
    g.nodes.insert(
        3,
        NodeKind::ToMatte(ToMatteParams { channel: MatteChannel::Luma, ..Default::default() }),
    );
    g.edges.insert((3, 0), 2);
    g.nodes.insert(4, NodeKind::Blend { op: BlendOp::Replace, mix: 1.0, gain: [1.0, 1.0] });
    g.edges.insert((4, 0), 0);
    g.edges.insert((4, 1), 1);
    g.view = Some(4);

    let unmasked = render(&ctx, &mut comp, &g)[0];
    assert!((unmasked - 1.0).abs() < 3e-3, "unmasked Replace is just A, got {unmasked}");

    g.edges.insert((4, 2), 3);
    let masked = render(&ctx, &mut comp, &g)[0];
    assert!(
        (masked - 0.5).abs() < 6e-3,
        "half coverage should land half of A on B, got {masked}"
    );

    // Same matte, different op: Over is limited the same way. White over
    // black is white, so half coverage reads 0.5 again.
    g.nodes.insert(4, NodeKind::Blend { op: BlendOp::Over, mix: 1.0, gain: [1.0, 1.0] });
    let over = render(&ctx, &mut comp, &g)[0];
    assert!((over - 0.5).abs() < 6e-3, "a masked Over should also halve, got {over}");
}

/// Blend's `mix`: 0 is a bypass to B, 1 is the full blend, and it multiplies
/// with the matte rather than fighting it.
#[test]
fn blend_mix_dissolves_back_to_b() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp, 0, [1.0, 1.0, 1.0, 1.0]); // A: white
    upload(&ctx, &mut comp, 1, [0.0, 0.0, 0.0, 1.0]); // B: black
    upload(&ctx, &mut comp, 2, [0.5, 0.5, 0.5, 1.0]); // matte plate

    let mut g = Graph::default();
    g.nodes.insert(0, NodeKind::read(0, ""));
    g.nodes.insert(1, NodeKind::read(1, ""));
    g.nodes.insert(2, NodeKind::read(2, ""));
    g.nodes.insert(
        3,
        NodeKind::ToMatte(ToMatteParams { channel: MatteChannel::Luma, ..Default::default() }),
    );
    g.edges.insert((3, 0), 2);
    g.nodes.insert(4, NodeKind::Blend { op: BlendOp::Over, mix: 1.0, gain: [1.0, 1.0] });
    g.edges.insert((4, 0), 0);
    g.edges.insert((4, 1), 1);
    g.view = Some(4);

    for (mix, want) in [(1.0f32, 1.0f32), (0.5, 0.5), (0.0, 0.0)] {
        g.nodes.insert(4, NodeKind::Blend { op: BlendOp::Over, mix, gain: [1.0, 1.0] });
        let got = render(&ctx, &mut comp, &g)[0];
        assert!(
            (got - want).abs() < 6e-3,
            "mix {mix} should read {want} (0 = bypass to B), got {got}"
        );
    }

    // Matte and mix compose: half coverage at half mix is a quarter of A.
    g.nodes.insert(4, NodeKind::Blend { op: BlendOp::Over, mix: 0.5, gain: [1.0, 1.0] });
    g.edges.insert((4, 2), 3);
    let both = render(&ctx, &mut comp, &g)[0];
    assert!((both - 0.25).abs() < 6e-3, "matte * mix should multiply, got {both}");
}

/// The matte algebra: every `MatteOp` on the same pair of coverages, plus
/// the two identities that make a chain of these usable — an unwired input
/// is EMPTY coverage (so one operand alone passes through) and `mix` at 0
/// is a bypass to A.
#[test]
fn matte_combine_covers_the_whole_algebra() {
    use engine::graph::{MatteCombineParams, MatteOp};

    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    let (a, b) = (0.4f32, 0.25f32);
    upload(&ctx, &mut comp, 0, [a, a, a, 1.0]);
    upload(&ctx, &mut comp, 1, [b, b, b, 1.0]);

    let mut g = Graph::default();
    g.nodes.insert(0, NodeKind::read(0, ""));
    g.nodes.insert(1, NodeKind::read(1, ""));
    let to_matte =
        || NodeKind::ToMatte(ToMatteParams { channel: MatteChannel::Luma, ..Default::default() });
    g.nodes.insert(2, to_matte());
    g.nodes.insert(3, to_matte());
    g.edges.insert((2, 0), 0);
    g.edges.insert((3, 0), 1);
    g.nodes.insert(4, NodeKind::MatteCombine(Default::default()));
    g.edges.insert((4, 0), 2);
    g.edges.insert((4, 1), 3);
    g.view = Some(4);

    // Values chosen so no two ops agree — a test where `subtract` and
    // `intersect` both read 0.15 would pass with the two wired backwards.
    for (op, want) in [
        (MatteOp::Union, a.max(b)),
        (MatteOp::Add, a + b),
        (MatteOp::Replace, b),
        (MatteOp::Subtract, a * (1.0 - b)),
        (MatteOp::Intersect, a * b),
        (MatteOp::Min, a.min(b)),
        (MatteOp::Difference, (a - b).abs()),
    ] {
        g.nodes.insert(4, NodeKind::MatteCombine(MatteCombineParams { op, mix: 1.0 }));
        let got = render(&ctx, &mut comp, &g)[0];
        assert!(
            (got - want).abs() < 6e-3,
            "{} of {a} and {b} should be {want}, got {got}",
            op.name()
        );
    }

    // Half mix on a subtract lands halfway back to A.
    g.nodes.insert(
        4,
        NodeKind::MatteCombine(MatteCombineParams { op: MatteOp::Subtract, mix: 0.5 }),
    );
    let half = render(&ctx, &mut comp, &g)[0];
    let want = a + (a * (1.0 - b) - a) * 0.5;
    assert!((half - want).abs() < 6e-3, "mix 0.5 should read {want}, got {half}");

    // An unwired B is empty coverage, so a union is just A — NOT a solid
    // frame, which is what "unwired mask = 1" would have given.
    g.nodes.insert(4, NodeKind::MatteCombine(Default::default()));
    g.edges.remove(&(4, 1));
    let alone = render(&ctx, &mut comp, &g)[0];
    assert!((alone - a).abs() < 6e-3, "one operand alone should pass through, got {alone}");

    // And the matte reaches ALPHA as well as rgb, so it composites.
    let px = render(&ctx, &mut comp, &g);
    assert!((px[3] - a).abs() < 6e-3, "matte must be in alpha too, got {}", px[3]);
}
