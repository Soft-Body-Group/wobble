//! Headless integration for the MODULATED Transform: a matte on input 1
//! says how much of the transform applies at each output pixel — the
//! UV map + UV Warp trick, done as one position expression in the fused pass.
//!
//! What these lock down:
//! * modulate 1 is the Transform exactly, in both blend modes, pin included;
//!   modulate 0 is the input;
//! * PARAMS blend at 0.5 is the Transform at half strength (half the move,
//!   half the rotation, the square root of the scale);
//! * COORDS blend is a different picture on a rotation (the chord, not the
//!   arc) — the reason the mode exists;
//! * unwired, the Transform is the plan it always was: three uniform rows
//!   and a plain `persp`, no modulate code.
//!
//! Self-skips without a GPU (CI).

use engine::compositor::Compositor;
use engine::decode::ImageInfo;
use engine::graph::{
    CornerPinParams, Graph, MatteChannel, NodeKind, ToMatteParams, TransformBlend, TransformParams,
};
use engine::timing::GpuTimer;
use half::f16;

const W: u32 = 64;
const H: u32 = 64;

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP transform_mod test: no adapter ({e})");
            None
        }
    }
}

fn upload(ctx: &engine::GpuContext, comp: &mut Compositor, slot: usize, f: impl Fn(u32, u32) -> [f32; 4]) {
    comp.ensure_reader(slot, ImageInfo { width: W, height: H });
    let mut px: Vec<[f16; 4]> = Vec::with_capacity((W * H) as usize);
    for y in 0..H {
        for x in 0..W {
            px.push(f(x, y).map(f16::from_f32));
        }
    }
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

/// Every pixel of the output, rgba.
fn render(ctx: &engine::GpuContext, comp: &mut Compositor, graph: &Graph) -> Vec<[f32; 4]> {
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
    let mut out = Vec::with_capacity((W * H) as usize);
    for y in 0..H as usize {
        let row: &[f16] = bytemuck::cast_slice(&data[y * padded as usize..][..(W * 8) as usize]);
        for x in 0..W as usize {
            out.push(std::array::from_fn(|c| row[x * 4 + c].to_f32()));
        }
    }
    drop(data);
    staging.unmap();
    out
}

fn max_diff(a: &[[f32; 4]], b: &[[f32; 4]]) -> f32 {
    a.iter()
        .zip(b)
        .flat_map(|(x, y)| (0..4).map(move |c| (x[c] - y[c]).abs()))
        .fold(0.0, f32::max)
}

/// A picture with something to see in every direction: ramps in x and y.
fn pattern(x: u32, y: u32) -> [f32; 4] {
    [x as f32 / W as f32, y as f32 / H as f32, 0.5, 1.0]
}

/// Reader 0 -> Transform; with `modulate`, reader 1 through a ToMatte onto
/// the Transform's modulate port.
fn graph_of(t: TransformParams, modulate: bool) -> Graph {
    graph_node(NodeKind::Transform(t), modulate)
}

fn graph_node(node: NodeKind, modulate: bool) -> Graph {
    let mut g = Graph::default();
    g.nodes.insert(0, NodeKind::read(0, ""));
    g.nodes.insert(3, node);
    g.edges.insert((3, 0), 0);
    if modulate {
        g.nodes.insert(1, NodeKind::read(1, ""));
        g.nodes.insert(
            2,
            NodeKind::ToMatte(ToMatteParams { channel: MatteChannel::R, ..Default::default() }),
        );
        g.edges.insert((2, 0), 1);
        g.edges.insert((3, 1), 2);
    }
    g.view = Some(3);
    g
}

const PIN: [[f32; 2]; 4] = [[0.05, 0.02], [0.97, 0.08], [0.9, 0.95], [0.02, 0.9]];

#[test]
fn full_and_zero_strength_are_the_transform_and_the_input() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 2);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp, 0, pattern);
    let t = TransformParams {
        translate: [6.0, -3.0],
        rotate: 20.0,
        scale: [1.2, 0.9],
        pin: PIN,
        ..Default::default()
    };
    let plain = render(&ctx, &mut comp, &graph_of(t, false));
    let input = render(&ctx, &mut comp, &graph_of(TransformParams::default(), false));
    for mode in [TransformBlend::Params, TransformBlend::Coords] {
        let t = TransformParams { mod_blend: mode, ..t };
        upload(&ctx, &mut comp, 1, |_, _| [1.0; 4]);
        let one = render(&ctx, &mut comp, &graph_of(t, true));
        assert!(max_diff(&one, &plain) < 2e-3, "{mode:?}: modulate 1 is not the Transform ({})", max_diff(&one, &plain));
        upload(&ctx, &mut comp, 1, |_, _| [0.0; 4]);
        let zero = render(&ctx, &mut comp, &graph_of(t, true));
        assert!(max_diff(&zero, &input) < 2e-3, "{mode:?}: modulate 0 is not the input");
    }
}

#[test]
fn params_blend_is_the_transform_at_partial_strength() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 2);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp, 0, pattern);
    let full = TransformParams { translate: [8.0, 4.0], rotate: 40.0, scale: [1.44, 1.44], ..Default::default() };
    let half = TransformParams { translate: [4.0, 2.0], rotate: 20.0, scale: [1.2, 1.2], ..Default::default() };
    let want = render(&ctx, &mut comp, &graph_of(half, false));
    upload(&ctx, &mut comp, 1, |_, _| [0.5; 4]);
    let got = render(&ctx, &mut comp, &graph_of(full, true));
    assert!(max_diff(&got, &want) < 3e-3, "params 0.5 is not half the transform: {}", max_diff(&got, &want));

    // The same half-strength rotation through COORDS is the chord, not the
    // arc: a different picture.
    let spin = TransformParams { rotate: 90.0, ..Default::default() };
    let p = render(&ctx, &mut comp, &graph_of(spin, true));
    let c = render(&ctx, &mut comp, &graph_of(TransformParams { mod_blend: TransformBlend::Coords, ..spin }, true));
    assert!(max_diff(&p, &c) > 0.05, "params and coords agree on a half rotation");

    // A varying matte varies the move: 0 at the left edge, 1 at the right,
    // on a pure translate — the left column stays, the right one moves.
    let mv = TransformParams { translate: [0.0, 10.0], ..Default::default() };
    upload(&ctx, &mut comp, 1, |x, _| [x as f32 / (W - 1) as f32; 4]);
    let ramp = render(&ctx, &mut comp, &graph_of(mv, true));
    let g_at = |o: &[[f32; 4]], x: usize| o[32 * W as usize + x][1];
    assert!((g_at(&ramp, 0) - g_at(&input_of(&ctx, &mut comp), 0)).abs() < 2e-3, "the left edge moved");
    assert!(g_at(&ramp, 63) < g_at(&ramp, 0) - 0.1, "the right edge did not move");
}

fn input_of(ctx: &engine::GpuContext, comp: &mut Compositor) -> Vec<[f32; 4]> {
    render(ctx, comp, &graph_of(TransformParams::default(), false))
}

/// Unwired, nothing changes: three uniform rows and the plain `persp`.
#[test]
fn unwired_modulate_costs_nothing() {
    let t = TransformParams { rotate: 10.0, ..Default::default() };
    let plain = engine::codegen::generate(&graph_of(t, false));
    let wired = engine::codegen::generate(&graph_of(t, true));
    assert!(plain.layout.modulated.is_empty());
    assert!(!plain.main.wgsl.contains("= xf_mod("), "unwired Transform emits the modulate path");
    assert!(wired.main.wgsl.contains("= xf_mod("));
    assert!(wired.layout.modulated.contains(&3));
    let span = |p: &engine::codegen::GeneratedPlan| {
        let s = p.layout.node_slot[&3];
        p.layout.node_slot.values().filter(|v| **v > s).min().copied().unwrap_or(p.layout.total_slots) - s
    };
    assert_eq!(span(&plain), 3, "an unwired Transform grew its span");
    assert_eq!(span(&wired), 3 + engine::codegen::MOD_XFORM_SLOTS);
}

/// A CornerPin modulates the same way: 1 is the pin, 0 the input, and
/// PARAMS at 0.5 is the pin with every corner half way there.
#[test]
fn a_corner_pin_modulates_like_a_transform() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 2);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp, 0, pattern);
    let pin = |to, mod_blend| NodeKind::CornerPin(CornerPinParams { to, mod_blend, ..CornerPinParams::default() });
    let plain = render(&ctx, &mut comp, &graph_node(pin(PIN, TransformBlend::Params), false));
    let input = input_of(&ctx, &mut comp);
    for mode in [TransformBlend::Params, TransformBlend::Coords] {
        upload(&ctx, &mut comp, 1, |_, _| [1.0; 4]);
        let one = render(&ctx, &mut comp, &graph_node(pin(PIN, mode), true));
        assert!(max_diff(&one, &plain) < 2e-3, "{mode:?}: modulate 1 is not the pin");
        upload(&ctx, &mut comp, 1, |_, _| [0.0; 4]);
        let zero = render(&ctx, &mut comp, &graph_node(pin(PIN, mode), true));
        assert!(max_diff(&zero, &input) < 2e-3, "{mode:?}: modulate 0 is not the input");
    }
    let unit = CornerPinParams::default().to;
    let half: [[f32; 2]; 4] =
        std::array::from_fn(|i| [0.5 * (unit[i][0] + PIN[i][0]), 0.5 * (unit[i][1] + PIN[i][1])]);
    let want = render(&ctx, &mut comp, &graph_node(pin(half, TransformBlend::Params), false));
    upload(&ctx, &mut comp, 1, |_, _| [0.5; 4]);
    let got = render(&ctx, &mut comp, &graph_node(pin(PIN, TransformBlend::Params), true));
    assert!(max_diff(&got, &want) < 3e-3, "params 0.5 is not the half-way pin: {}", max_diff(&got, &want));
    // Unwired: the plain three rows.
    let plan = engine::codegen::generate(&graph_node(pin(PIN, TransformBlend::Params), false));
    assert!(!plan.main.wgsl.contains("= xf_mod("));
}
