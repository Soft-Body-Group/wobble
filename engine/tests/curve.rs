//! Headless integration for the control-point curve: does the GPU's
//! `curve_eval` agree with the shape the editor draws, does splitting rgb
//! really give each channel its own curve, and does a LumaKey with the
//! default two points still key exactly like the smoothstep it replaced?
//!
//! Self-skips without a GPU (CI).

use engine::compositor::Compositor;
use engine::decode::ImageInfo;
use engine::graph::{CurveInterp, CurveParams, Graph, LumaBand, LumaKeyParams, NodeKind};
use engine::timing::GpuTimer;
use half::f16;

const W: u32 = 16;
const H: u32 = 8;

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP curve test: no adapter ({e})");
            None
        }
    }
}

/// A flat plate of one rgb triple.
fn upload(ctx: &engine::GpuContext, comp: &mut Compositor, slot: usize, v: [f32; 3]) {
    comp.ensure_reader(slot, ImageInfo { width: W, height: H });
    let px: Vec<[f16; 4]> = (0..W * H)
        .map(|_| {
            [
                f16::from_f32(v[0]),
                f16::from_f32(v[1]),
                f16::from_f32(v[2]),
                f16::ONE,
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

/// The reference the app's `curveui::eval` also implements for a Curve node
/// (`flat_ends: false`): monotone cubic whose END tangents are the end
/// segments' slopes — what the extrapolation carries on with, so the curve
/// meets it without a kink — or plain linear.
fn eval(pts: &[[f32; 2]], x: f32, interp: CurveInterp) -> f32 {
    let sec = |i: usize| (pts[i + 1][1] - pts[i][1]) / (pts[i + 1][0] - pts[i][0]).max(1e-5);
    let tan = |i: usize| {
        if i == 0 {
            return sec(0);
        }
        if i + 1 >= pts.len() {
            return sec(pts.len() - 2);
        }
        let (a, b) = (sec(i - 1), sec(i));
        if a * b <= 0.0 {
            return 0.0;
        }
        let lim = 3.0 * a.abs().min(b.abs());
        (0.5 * (a + b)).clamp(-lim, lim)
    };
    let last = pts.len() - 1;
    if x <= pts[0][0] {
        return pts[0][1] + (x - pts[0][0]) * sec(0);
    }
    if x >= pts[last][0] {
        return pts[last][1] + (x - pts[last][0]) * sec(last - 1);
    }
    for i in 1..pts.len() {
        let (p0, p1) = (pts[i - 1], pts[i]);
        if x <= p1[0] {
            let h = (p1[0] - p0[0]).max(1e-5);
            let t = (x - p0[0]) / h;
            if interp == CurveInterp::Linear {
                return p0[1] + (p1[1] - p0[1]) * t;
            }
            let (m0, m1) = (tan(i - 1) * h, tan(i) * h);
            let (t2, t3) = (t * t, t * t * t);
            return (2.0 * t3 - 3.0 * t2 + 1.0) * p0[1]
                + (t3 - 2.0 * t2 + t) * m0
                + (-2.0 * t3 + 3.0 * t2) * p1[1]
                + (t3 - t2) * m1;
        }
    }
    pts[pts.len() - 1][1]
}

fn curve_graph(cv: CurveParams) -> Graph {
    let mut g = Graph::default();
    g.nodes.insert(0, NodeKind::read(0, ""));
    g.nodes.insert(1, NodeKind::Curve(cv));
    g.edges.insert((1, 0), 0);
    g.view = Some(1);
    g
}

/// Both interpolations must match the reference the editor draws — a curve
/// that renders differently from its own preview is worse than no curve.
#[test]
fn gpu_matches_the_reference_shape() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    let pts = vec![[0.0, 0.05], [0.35, 0.55], [0.7, 0.6], [1.0, 0.95]];
    for interp in CurveInterp::ALL {
        for step in 0..=8 {
            let x = step as f32 / 8.0;
            upload(&ctx, &mut comp, 0, [x, x, x]);
            let g = curve_graph(CurveParams {
                points: pts.clone(),
                interp,
                ..Default::default()
            });
            let got = render(&ctx, &mut comp, &g)[0];
            let want = eval(&pts, x, interp);
            assert!(
                (got - want).abs() < 4e-3,
                "{:?} at x={x}: gpu {got:.4} vs reference {want:.4}",
                interp
            );
        }
    }
}

/// Linear and smooth have to actually differ mid-segment, or "smooth
/// interpolation" is a no-op switch.
#[test]
fn smooth_bends_where_linear_does_not() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    // A bend at (0.5, 0.2): at x=0.25 linear reads 0.1, the smoothed one
    // eases under it. (Two points no longer show it: a smoothed colour
    // curve's ends follow its end slopes, so a two-point ramp is a line.)
    let pts = vec![[0.0, 0.0], [0.5, 0.2], [1.0, 1.0]];
    upload(&ctx, &mut comp, 0, [0.25, 0.25, 0.25]);
    let lin = render(
        &ctx,
        &mut comp,
        &curve_graph(CurveParams {
            points: pts.clone(),
            interp: CurveInterp::Linear,
            ..Default::default()
        }),
    )[0];
    let smooth = render(
        &ctx,
        &mut comp,
        &curve_graph(CurveParams {
            points: pts,
            interp: CurveInterp::Smooth,
            ..Default::default()
        }),
    )[0];
    assert!((lin - 0.1).abs() < 3e-3, "linear should read 0.1, got {lin}");
    assert!(smooth < lin - 0.03, "smooth {smooth} should ease under linear {lin}");
}

/// Split rgb: each channel follows its OWN curve, and turning the split off
/// puts every channel back on the master.
#[test]
fn split_rgb_gives_each_channel_its_own_curve() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp, 0, [0.5, 0.5, 0.5]);
    // Master says 0.5 -> 0.5; the channels each say something else, and an
    // odd master point count also proves the vec4 padding lines up.
    let cv = CurveParams {
        points: vec![[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]],
        rgb: [
            vec![[0.0, 0.0], [1.0, 0.0]],           // r -> 0
            vec![[0.0, 0.25], [1.0, 0.25]],         // g -> 0.25
            vec![[0.0, 0.8], [1.0, 0.8]],           // b -> 0.8
        ],
        split: true,
        interp: CurveInterp::Linear,
        ..Default::default()
    };
    let got = render(&ctx, &mut comp, &curve_graph(cv.clone()));
    assert!(got[0].abs() < 3e-3, "r {} should be 0", got[0]);
    assert!((got[1] - 0.25).abs() < 3e-3, "g {} should be 0.25", got[1]);
    assert!((got[2] - 0.8).abs() < 3e-3, "b {} should be 0.8", got[2]);

    let ganged = render(
        &ctx,
        &mut comp,
        &curve_graph(CurveParams { split: false, ..cv }),
    );
    for (i, v) in ganged[..3].iter().enumerate() {
        assert!((v - 0.5).abs() < 3e-3, "ch{i} {v} should follow the master (0.5)");
    }
}

/// A default LumaKey must key EXACTLY as the smoothstep it replaced — every
/// existing setup depends on that — while `Linear` gives the hard ramp.
#[test]
fn lumakey_default_is_the_old_smoothstep() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    let (lo, hi) = (0.2f32, 0.8f32);
    for step in 0..=8 {
        let l = step as f32 / 8.0;
        upload(&ctx, &mut comp, 0, [l, l, l]);
        let mut g = Graph::default();
        g.nodes.insert(0, NodeKind::read(0, ""));
        g.nodes.insert(
            1,
            NodeKind::LumaKey(LumaKeyParams {
                bands: vec![LumaBand { lo, hi, ..Default::default() }],
                ranges: None,
            }),
        );
        g.edges.insert((1, 0), 0);
        g.view = Some(1);
        let got = render(&ctx, &mut comp, &g)[0];
        let t = ((l - lo) / (hi - lo)).clamp(0.0, 1.0);
        let want = t * t * (3.0 - 2.0 * t);
        assert!((got - want).abs() < 3e-3, "luma {l}: got {got:.4} want {want:.4}");
    }
}

/// Custom points shape the key: a point that holds the response down until
/// late must read lower than the plain ramp at the same luma.
#[test]
fn lumakey_custom_points_reshape_the_response() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp, 0, [0.5, 0.5, 0.5]);
    let key = |points: Vec<[f32; 2]>, interp| {
        let mut g = Graph::default();
        g.nodes.insert(0, NodeKind::read(0, ""));
        g.nodes.insert(
            1,
            NodeKind::LumaKey(LumaKeyParams {
                bands: vec![LumaBand {
                    lo: 0.0,
                    hi: 1.0,
                    invert: false,
                    points,
                    interp,
                    ..Default::default()
                }],
                ranges: None,
            }),
        );
        g.edges.insert((1, 0), 0);
        g.view = Some(1);
        g
    };
    let plain = render(&ctx, &mut comp, &key(Vec::new(), CurveInterp::Linear))[0];
    assert!((plain - 0.5).abs() < 3e-3, "a 0..1 linear ramp at 0.5 is 0.5, got {plain}");
    // Hold the response near zero until 0.7, then let it climb.
    let held = render(
        &ctx,
        &mut comp,
        &key(vec![[0.7, 0.05]], CurveInterp::Linear),
    )[0];
    assert!(held < 0.1, "the extra point should hold the key down, got {held}");
    // Points outside (lo, hi) are ignored rather than reordering the ends.
    let outside = render(
        &ctx,
        &mut comp,
        &key(vec![[-0.5, 1.0], [3.0, 0.0]], CurveInterp::Linear),
    )[0];
    assert!((outside - plain).abs() < 3e-3, "out-of-range points changed the key");
}

/// A matte has to read as a matte in the ALPHA channel too, or the viewer's
/// `A` toggle over a LumaKey shows flat white instead of the key. Spline and
/// Keyer already carried coverage in alpha; LumaKey and ToMatte said 1.0.
#[test]
fn a_key_carries_its_coverage_in_alpha() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    for (luma, want) in [(0.0f32, 0.0f32), (0.5, 0.5), (1.0, 1.0)] {
        upload(&ctx, &mut comp, 0, [luma, luma, luma]);
        let mut g = Graph::default();
        g.nodes.insert(0, NodeKind::read(0, ""));
        g.nodes.insert(
            1,
            NodeKind::LumaKey(LumaKeyParams {
                bands: vec![LumaBand {
                    lo: 0.0,
                    hi: 1.0,
                    interp: CurveInterp::Linear,
                    ..Default::default()
                }],
                ranges: None,
            }),
        );
        g.edges.insert((1, 0), 0);
        g.view = Some(1);
        let got = render(&ctx, &mut comp, &g);
        assert!(
            (got[3] - want).abs() < 3e-3,
            "luma {luma}: alpha {} should match the matte {want}",
            got[3]
        );
        assert!(
            (got[0] - got[3]).abs() < 3e-3,
            "rgb {} and alpha {} must agree",
            got[0],
            got[3]
        );
    }
}

/// Same for ToMatte, the bridge every image->mask connection auto-inserts.
#[test]
fn to_matte_carries_its_coverage_in_alpha() {
    use engine::graph::{MatteChannel, ToMatteParams};
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp, 0, [0.25, 0.25, 0.25]);
    let mut g = Graph::default();
    g.nodes.insert(0, NodeKind::read(0, ""));
    g.nodes.insert(
        1,
        NodeKind::ToMatte(ToMatteParams {
            channel: MatteChannel::Luma,
            ..Default::default()
        }),
    );
    g.edges.insert((1, 0), 0);
    g.view = Some(1);
    let got = render(&ctx, &mut comp, &g);
    assert!((got[3] - 0.25).abs() < 3e-3, "alpha {} should be the matte", got[3]);
    assert!((got[0] - got[3]).abs() < 3e-3, "rgb and alpha must agree");
}

/// The reported bug: a colour Curve must not clamp. A default Curve is a
/// straight x = y that keeps going, so a value of 6.0 in a scene-linear comp
/// comes out 6.0 — not pinned to the last point's y.
#[test]
fn the_colour_curve_does_not_clamp_its_range() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    for v in [2.0f32, 6.0, 20.0] {
        upload(&ctx, &mut comp, 0, [v, v, v]);
        let got = render(&ctx, &mut comp, &curve_graph(CurveParams::default()))[0];
        assert!(
            (got - v).abs() < v * 0.01,
            "a default Curve is a no-op: {v} came out {got}"
        );
    }
    // A curve that ends steep keeps climbing past its last point instead of
    // flattening: (1,1) -> (2,4) has slope 3, so 3.0 lands at 7.
    upload(&ctx, &mut comp, 0, [3.0, 3.0, 3.0]);
    let steep = CurveParams {
        points: vec![[0.0, 0.0], [1.0, 1.0], [2.0, 4.0]],
        interp: CurveInterp::Linear,
        ..Default::default()
    };
    let got = render(&ctx, &mut comp, &curve_graph(steep))[0];
    assert!((got - 7.0).abs() < 0.05, "extrapolation should reach 7.0, got {got}");
}

/// The high-point button rebuilds the curve as a straight line to (high, 1):
/// everything at or above `high` maps to 1 and beyond, which is the whole
/// reason for the control.
#[test]
fn the_high_point_ramp_maps_that_input_to_one() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    let cv = CurveParams {
        points: CurveParams::ramp(8.0),
        interp: CurveInterp::Linear,
        ..Default::default()
    };
    for (input, want) in [(0.0f32, 0.0f32), (4.0, 0.5), (8.0, 1.0), (16.0, 2.0)] {
        upload(&ctx, &mut comp, 0, [input, input, input]);
        let got = render(&ctx, &mut comp, &curve_graph(cv.clone()))[0];
        assert!((got - want).abs() < 0.02, "input {input}: got {got} want {want}");
    }
}
