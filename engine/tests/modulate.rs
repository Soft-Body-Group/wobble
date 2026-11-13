//! Headless integration for the MODULATE input (Blur, Dilate): a matte that
//! scales the effect per pixel, as opposed to the mask, which mixes the
//! finished effect back against the input.
//!
//! What these lock down:
//! * modulate 1 is the node exactly as it renders unwired, 0 is the input,
//!   and a flat 0.5 is the node at HALF its amount — not a half-and-half
//!   mix (which a Blur no longer offers at all);
//! * a matte that varies does vary the effect across the frame;
//! * unwired costs nothing: no aux subtree and the plain pass module.
//!
//! Self-skips without a GPU (CI).

use engine::compositor::Compositor;
use engine::decode::ImageInfo;
use engine::graph::{BlurParams, DilateParams, Graph, MatteChannel, NodeKind, ToMatteParams};
use engine::nodes::stage::Wired;
use engine::timing::GpuTimer;
use half::f16;

const W: u32 = 64;
const H: u32 = 8;

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP modulate test: no adapter ({e})");
            None
        }
    }
}

fn upload(ctx: &engine::GpuContext, comp: &mut Compositor, slot: usize, f: impl Fn(u32) -> f32) {
    comp.ensure_reader(slot, ImageInfo { width: W, height: H });
    let mut px: Vec<[f16; 4]> = Vec::with_capacity((W * H) as usize);
    for _ in 0..H {
        for x in 0..W {
            let v = f16::from_f32(f(x));
            px.push([v, v, v, v]);
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

/// A hard vertical edge at x = 32: what a blur visibly softens and a
/// dilate visibly moves.
fn edge(x: u32) -> f32 {
    if x < 32 { 1.0 } else { 0.0 }
}

/// Row 4 of the output's red channel.
fn row(ctx: &engine::GpuContext, comp: &mut Compositor, graph: &Graph) -> Vec<f32> {
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
    let off = (padded * 4) as usize;
    let halves: &[f16] = bytemuck::cast_slice(&data[off..off + (W * 8) as usize]);
    let out = (0..W as usize).map(|i| halves[i * 4].to_f32()).collect();
    drop(data);
    staging.unmap();
    out
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

/// Reader 0 (the edge) -> `node`; with `modulate`, reader 1 through a
/// ToMatte onto the node's MODULATE port; with `mask`, onto its mask port.
fn graph_of(node: NodeKind, modulate: bool, mask: bool) -> Graph {
    let mut g = Graph::default();
    g.nodes.insert(0, NodeKind::read(0, ""));
    let mut src = 0;
    if matches!(node, NodeKind::Dilate(_)) {
        // Dilate's input IS a matte.
        g.nodes.insert(4, NodeKind::ToMatte(ToMatteParams { channel: MatteChannel::R, ..Default::default() }));
        g.edges.insert((4, 0), 0);
        src = 4;
    }
    g.nodes.insert(1, NodeKind::read(1, ""));
    g.nodes.insert(
        2,
        NodeKind::ToMatte(ToMatteParams { channel: MatteChannel::R, ..Default::default() }),
    );
    g.edges.insert((2, 0), 1);
    let port = |on: bool, p: Option<usize>| p.filter(|_| on);
    g.nodes.insert(3, node.clone());
    g.edges.insert((3, 0), src);
    if let Some(p) = port(modulate, node.mod_input()) {
        g.edges.insert((3, p), 2);
    }
    if let Some(p) = port(mask, node.mask_input()) {
        g.edges.insert((3, p), 2);
    }
    g.view = Some(3);
    g
}

fn blur(size: f32) -> NodeKind {
    NodeKind::Blur(BlurParams { size, matte: false })
}

#[test]
fn blur_modulate_scales_the_blur_not_the_mix() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 2);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp, 0, edge);

    let plain6 = row(&ctx, &mut comp, &graph_of(blur(6.0), false, false));
    let plain3 = row(&ctx, &mut comp, &graph_of(blur(3.0), false, false));
    let input = row(&ctx, &mut comp, &graph_of(blur(0.0), false, false));

    upload(&ctx, &mut comp, 1, |_| 1.0);
    let m1 = row(&ctx, &mut comp, &graph_of(blur(6.0), true, false));
    assert!(max_diff(&m1, &plain6) < 2e-3, "modulate 1 is not the plain blur: {}", max_diff(&m1, &plain6));

    upload(&ctx, &mut comp, 1, |_| 0.0);
    let m0 = row(&ctx, &mut comp, &graph_of(blur(6.0), true, false));
    assert!(max_diff(&m0, &input) < 2e-3, "modulate 0 is not the input");

    upload(&ctx, &mut comp, 1, |_| 0.5);
    let half = row(&ctx, &mut comp, &graph_of(blur(6.0), true, false));
    assert!(
        max_diff(&half, &plain3) < 2e-3,
        "modulate 0.5 is not a blur of half the size: {}",
        max_diff(&half, &plain3)
    );
    // A blur's ONE matte port is the modulate; there is no mix-back mask.
    assert_eq!((blur(1.0).mask_input(), blur(1.0).mod_input()), (None, Some(1)));

    // A matte that varies varies the blur: 0 on the left, 1 on the right,
    // with the image edge in the middle of the ramp.
    upload(&ctx, &mut comp, 1, |x| x as f32 / (W - 1) as f32);
    let ramp = row(&ctx, &mut comp, &graph_of(blur(6.0), true, false));
    let width = |r: &[f32]| r.iter().filter(|v| **v > 0.05 && **v < 0.95).count();
    assert!(
        width(&ramp) > width(&input) && width(&ramp) < width(&plain6),
        "a ramped modulate should blur the edge part-way: {} vs {}..{}",
        width(&ramp),
        width(&input),
        width(&plain6)
    );
}

#[test]
fn dilate_modulate_scales_the_amount() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 2);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp, 0, edge);
    let dil = |amount| NodeKind::Dilate(DilateParams { amount });
    let plain4 = row(&ctx, &mut comp, &graph_of(dil(4.0), false, false));
    let plain2 = row(&ctx, &mut comp, &graph_of(dil(2.0), false, false));

    upload(&ctx, &mut comp, 1, |_| 1.0);
    let m1 = row(&ctx, &mut comp, &graph_of(dil(4.0), true, false));
    assert!(max_diff(&m1, &plain4) < 2e-3, "modulate 1 is not the plain dilate");

    upload(&ctx, &mut comp, 1, |_| 0.5);
    let half = row(&ctx, &mut comp, &graph_of(dil(4.0), true, false));
    assert!(max_diff(&half, &plain2) < 2e-3, "modulate 0.5 is not half the dilate");
}

/// The fast path: unwired, a Blur is the plan it always was — no aux
/// subtree to render, and the pass module without the extra binding.
#[test]
fn unwired_modulate_costs_nothing() {
    for (node, n_in) in [(blur(6.0), 2), (NodeKind::Dilate(DilateParams { amount: 2.0 }), 2)] {
        assert_eq!(node.input_count(), n_in);
        let plain = engine::codegen::generate(&graph_of(node.clone(), false, false));
        let wired = engine::codegen::generate(&graph_of(node.clone(), true, false));
        let aux = |p: &engine::codegen::GeneratedPlan| {
            p.blurs.iter().find(|(s, ..)| s.0 == 3).map(|(_, _, _, a)| a.len()).unwrap()
        };
        assert_eq!(aux(&plain), 0, "{}: unwired modulate still renders a subtree", node.name());
        assert_eq!(aux(&wired), 1);
        let def = node.stage_def().unwrap();
        let mut on = vec![false; n_in];
        on[0] = true;
        let off = def.program((W, H), Wired(&on)).passes;
        on[node.mod_input().unwrap()] = true;
        let wired = def.program((W, H), Wired(&on)).passes;
        for (a, b) in off.iter().zip(&wired) {
            assert!(a.binds.len() + 1 == b.binds.len(), "{}: unwired pass binds the matte", node.name());
            assert!(a.wgsl != b.wgsl, "{}: one module for both paths", node.name());
        }
    }
}
