//! Headless integration for the sub-pixel matte dilate / erode.
//!
//! The question a morphology node has to answer is "did the edge move by the
//! amount I asked for", and the interesting half of that is FRACTIONS: an
//! implementation that quietly snaps to whole pixels passes every integer
//! test and is useless for matte work. Self-skips without a GPU (CI).

use engine::compositor::Compositor;
use engine::decode::ImageInfo;
use engine::graph::{DilateParams, Graph, MatteChannel, NodeKind, ToMatteParams};
use engine::timing::GpuTimer;
use half::f16;

const W: u32 = 64;
const H: u32 = 8;

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP dilate test: no adapter ({e})");
            None
        }
    }
}

/// A vertical edge: coverage 1 left of `edge`, 0 right of it, with ONE
/// anti-aliased pixel in between — the ramp a real matte has, and what makes
/// a sub-pixel shift measurable.
fn upload_edge(ctx: &engine::GpuContext, comp: &mut Compositor, slot: usize, edge: f32) {
    comp.ensure_reader(slot, ImageInfo { width: W, height: H });
    let mut px: Vec<[f16; 4]> = Vec::with_capacity((W * H) as usize);
    for _ in 0..H {
        for x in 0..W {
            // Pixel centre at x + 0.5; coverage falls from 1 to 0 across it.
            let c = (edge - (x as f32 + 0.5) + 0.5).clamp(0.0, 1.0);
            let v = f16::from_f32(c);
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

/// One row of the output's red channel.
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
    // Row 4, to stay away from any edge effects at the top.
    let off = (padded * 4) as usize;
    let halves: &[f16] = bytemuck::cast_slice(&data[off..off + (W * 8) as usize]);
    let out = (0..W as usize).map(|i| halves[i * 4].to_f32()).collect();
    drop(data);
    staging.unmap();
    out
}

/// Where the coverage crosses 0.5, in pixels — the matte's actual edge.
fn edge_at(row: &[f32]) -> f32 {
    for i in 1..row.len() {
        if row[i - 1] >= 0.5 && row[i] < 0.5 {
            let t = (row[i - 1] - 0.5) / (row[i - 1] - row[i]).max(1e-6);
            return i as f32 - 0.5 + t;
        }
    }
    f32::NAN
}

fn graph_with(amount: f32) -> Graph {
    let mut g = Graph::default();
    g.nodes.insert(0, NodeKind::read(0, ""));
    // Read -> ToMatte (the input is already coverage) -> Dilate.
    g.nodes.insert(
        1,
        NodeKind::ToMatte(ToMatteParams { channel: MatteChannel::R, ..Default::default() }),
    );
    g.edges.insert((1, 0), 0);
    g.nodes.insert(2, NodeKind::Dilate(DilateParams { amount }));
    g.edges.insert((2, 0), 1);
    g.view = Some(2);
    g
}

#[test]
fn whole_pixel_amounts_move_the_edge_by_that_many_pixels() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload_edge(&ctx, &mut comp, 0, 32.0);
    let base = edge_at(&row(&ctx, &mut comp, &graph_with(0.0)));
    assert!(base.is_finite(), "the test plate has no edge to find");
    for amount in [1.0f32, 2.0, 4.0, -1.0, -3.0] {
        let got = edge_at(&row(&ctx, &mut comp, &graph_with(amount)));
        let moved = got - base;
        assert!(
            (moved - amount).abs() < 0.25,
            "amount {amount}: the edge moved {moved:.3} px (from {base:.3} to {got:.3})"
        );
    }
}

/// The whole point: fractions are real. A 0.25 px dilate has to move the
/// edge about a quarter of a pixel — not zero (snapped down) and not a whole
/// one (snapped up).
#[test]
fn sub_pixel_amounts_move_the_edge_sub_pixel_distances() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload_edge(&ctx, &mut comp, 0, 32.0);
    let base = edge_at(&row(&ctx, &mut comp, &graph_with(0.0)));
    let mut last = 0.0f32;
    for amount in [0.25f32, 0.5, 0.75] {
        let moved = edge_at(&row(&ctx, &mut comp, &graph_with(amount))) - base;
        assert!(
            (moved - amount).abs() < 0.2,
            "dilate {amount}: the edge moved {moved:.3} px"
        );
        assert!(moved > last, "growth must be monotone: {moved:.3} after {last:.3}");
        last = moved;
    }
    // And the same going the other way.
    let shrunk = edge_at(&row(&ctx, &mut comp, &graph_with(-0.5))) - base;
    assert!((shrunk + 0.5).abs() < 0.2, "erode 0.5: the edge moved {shrunk:.3} px");
}

/// Zero is a pass-through, not "an almost pass-through": a matte that drifts
/// when the node is doing nothing is worse than no node.
#[test]
fn zero_is_exactly_a_passthrough() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload_edge(&ctx, &mut comp, 0, 32.0);
    let mut plain = Graph::default();
    plain.nodes.insert(0, NodeKind::read(0, ""));
    plain.nodes.insert(
        1,
        NodeKind::ToMatte(ToMatteParams { channel: MatteChannel::R, ..Default::default() }),
    );
    plain.edges.insert((1, 0), 0);
    plain.view = Some(1);
    let want = row(&ctx, &mut comp, &plain);
    let got = row(&ctx, &mut comp, &graph_with(0.0));
    for (i, (a, b)) in want.iter().zip(got.iter()).enumerate() {
        assert!((a - b).abs() < 1e-3, "px {i}: {b} != {a}");
    }
}

/// A dilate must not push coverage past 1 or an erode below 0, and it must
/// leave the flat interior alone — only the edge is its business.
#[test]
fn the_flat_parts_of_the_matte_are_untouched() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload_edge(&ctx, &mut comp, 0, 32.0);
    for amount in [3.0f32, -3.0] {
        let got = row(&ctx, &mut comp, &graph_with(amount));
        assert!((got[2] - 1.0).abs() < 1e-3, "solid end became {}", got[2]);
        assert!(got[W as usize - 3].abs() < 1e-3, "empty end became {}", got[W as usize - 3]);
        assert!(
            got.iter().all(|v| (-1e-3..=1.0 + 1e-3).contains(v)),
            "coverage left 0..1"
        );
    }
}
