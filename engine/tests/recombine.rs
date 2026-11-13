//! Headless integration for the Recombine node (the harp's receiver): n
//! inputs summed in ONE fused expression. The editor-side lowering that
//! feeds it (Breakout → hidden ChannelSelect kids) is unit-tested in
//! `app::breakout`; what only a GPU run shows is the emitted WGSL
//! compiling and the sum being exact. Self-skips without a GPU (CI).

use engine::compositor::Compositor;
use engine::decode::ImageInfo;
use engine::graph::{AlphaFrom, Graph, NodeKind, RecombineParams};
use engine::timing::GpuTimer;
use half::f16;

const W: u32 = 16;
const H: u32 = 8;

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP recombine test: no adapter ({e})");
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

fn read(slot: usize) -> NodeKind {
    NodeKind::read(slot, "mem")
}

/// Three AOVs summed to the beauty — the rebuilt comb, one expression.
#[test]
fn recombine_sums_its_strings_exactly() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp, 0, [0.1, 0.0, 0.5, 1.0]);
    upload(&ctx, &mut comp, 1, [0.2, 0.3, 0.0, 0.0]);
    upload(&ctx, &mut comp, 2, [0.0, 0.1, 0.25, 0.0]);
    let mut g = Graph::default();
    g.nodes.insert(1, read(0));
    g.nodes.insert(2, read(1));
    g.nodes.insert(3, read(2));
    g.nodes.insert(
        4,
        NodeKind::Recombine(RecombineParams { n: 3, ..Default::default() }),
    );
    g.edges.insert((4, 0), 1);
    g.edges.insert((4, 1), 2);
    g.edges.insert((4, 2), 3);
    g.view = Some(4);
    let out = render(&ctx, &mut comp, &g);
    for (i, (got, want)) in
        out.iter().zip([0.3f32, 0.4, 0.75, 1.0]).enumerate()
    {
        assert!(
            (got - want).abs() < 1e-3,
            "channel {i}: {got} != {want} (out {out:?})"
        );
    }
}

/// The remainder's shape: a signed input SUBTRACTS, exactly — base rgba
/// plus, every AOV minus, negatives allowed and preserved.
#[test]
fn signed_inputs_subtract_exactly() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp, 0, [1.0, 0.5, 0.25, 1.0]);
    upload(&ctx, &mut comp, 1, [0.25, 0.75, 0.5, 0.25]);
    let mut g = Graph::default();
    g.nodes.insert(1, read(0));
    g.nodes.insert(2, read(1));
    g.nodes.insert(
        3,
        NodeKind::Recombine(RecombineParams {
            n: 2,
            sub: vec![false, true],
            ..Default::default()
        }),
    );
    g.edges.insert((3, 0), 1);
    g.edges.insert((3, 1), 2);
    g.view = Some(3);
    let out = render(&ctx, &mut comp, &g);
    for (i, (got, want)) in
        out.iter().zip([0.75f32, -0.25, -0.25, 0.75]).enumerate()
    {
        assert!(
            (got - want).abs() < 1e-3,
            "channel {i}: {got} != {want} (out {out:?})"
        );
    }
}

/// The rebuild wears the ORIGINAL rgba's alpha: rgb from the sum, `a`
/// from the porch input — and the sum's own alpha is ignored, which is the
/// whole point (adding per-pass coverage means nothing).
#[test]
fn alpha_comes_off_the_base_rgba() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    // Two passes, each carrying junk alpha, and the beauty they came from.
    upload(&ctx, &mut comp, 0, [0.25, 0.0, 0.5, 0.75]);
    upload(&ctx, &mut comp, 1, [0.25, 0.5, 0.0, 0.75]);
    upload(&ctx, &mut comp, 2, [9.0, 9.0, 9.0, 0.5]);
    let mut g = Graph::default();
    g.nodes.insert(1, read(0));
    g.nodes.insert(2, read(1));
    g.nodes.insert(3, read(2));
    let p = RecombineParams { n: 2, ..Default::default() };
    let base_port = p.base_input();
    g.nodes.insert(4, NodeKind::Recombine(p));
    g.edges.insert((4, 0), 1);
    g.edges.insert((4, 1), 2);
    g.edges.insert((4, base_port), 3);
    g.view = Some(4);
    let out = render(&ctx, &mut comp, &g);
    // rgb: the sum. a: the beauty's 0.5, not the passes' 1.5.
    for (i, (got, want)) in out.iter().zip([0.5f32, 0.5, 0.5, 0.5]).enumerate() {
        assert!(
            (got - want).abs() < 1e-3,
            "channel {i}: {got} != {want} (out {out:?})"
        );
    }
}

/// Alpha modes: the sum's own alpha, and forced opaque.
#[test]
fn the_other_alpha_modes_do_what_they_say() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp, 0, [0.25, 0.25, 0.25, 0.25]);
    upload(&ctx, &mut comp, 1, [9.0, 9.0, 9.0, 0.5]);
    for (mode, want_a) in [(AlphaFrom::Sum, 0.25f32), (AlphaFrom::Opaque, 1.0)] {
        let mut g = Graph::default();
        g.nodes.insert(1, read(0));
        g.nodes.insert(2, read(1));
        let p = RecombineParams { n: 1, alpha: mode, ..Default::default() };
        let base_port = p.base_input();
        g.nodes.insert(3, NodeKind::Recombine(p));
        g.edges.insert((3, 0), 1);
        g.edges.insert((3, base_port), 2); // wired, and deliberately ignored
        g.view = Some(3);
        let out = render(&ctx, &mut comp, &g);
        assert!(
            (out[3] - want_a).abs() < 1e-3,
            "{mode:?}: alpha {} != {want_a} (out {out:?})",
            out[3]
        );
        assert!((out[0] - 0.25).abs() < 1e-3, "{mode:?} moved rgb: {out:?}");
    }
}

/// A half-plugged harp still sums: unwired strings contribute nothing
/// rather than poisoning the expression.
#[test]
fn unwired_strings_contribute_nothing() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp, 0, [0.25, 0.5, 0.125, 1.0]);
    let mut g = Graph::default();
    g.nodes.insert(1, read(0));
    g.nodes.insert(
        2,
        NodeKind::Recombine(RecombineParams { n: 4, ..Default::default() }),
    );
    g.edges.insert((2, 2), 1); // only string 2 of 4 is plugged
    g.view = Some(2);
    let out = render(&ctx, &mut comp, &g);
    for (i, (got, want)) in
        out.iter().zip([0.25f32, 0.5, 0.125, 1.0]).enumerate()
    {
        assert!(
            (got - want).abs() < 1e-3,
            "channel {i}: {got} != {want} (out {out:?})"
        );
    }
}
