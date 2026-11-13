//! Expression on real pixels: every expression here is compiled into the
//! fused pass, rendered, and read back against the CPU mirror
//! (`ExpressionParams::apply`) — on a plate that carries the values an
//! expression is most often written to catch: exact zeros, infinities both
//! ways and NaN, straight out of a half-float texture the way a render's
//! fireflies arrive. Self-skips without a GPU (CI).

use engine::compositor::Compositor;
use engine::decode::ImageInfo;
use engine::graph::{ExpressionParams, Graph, NodeKind};
use engine::nodes::expression::Env;
use engine::timing::GpuTimer;
use half::f16;

const W: u32 = 12;
const H: u32 = 2;

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP expression test: no adapter ({e})");
            None
        }
    }
}

/// Red runs through the special values; green and blue are ordinary so a
/// field that should pass through visibly does.
fn source(x: u32, y: u32) -> [f32; 4] {
    const R: [f32; 12] = [
        0.0,
        0.3,
        1.0,
        -0.5,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        2.5,
        -0.0,
        0.75,
        65504.0,
        0.125,
    ];
    let r = R[x as usize];
    [r, 0.2 + 0.05 * x as f32, 0.5 + y as f32 * 0.25, 1.0]
}

fn upload(ctx: &engine::GpuContext, comp: &mut Compositor) {
    comp.ensure_reader(0, ImageInfo { width: W, height: H });
    let px: Vec<[f16; 4]> = (0..W * H)
        .map(|i| source(i % W, i / W).map(f16::from_f32))
        .collect();
    ctx.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &comp.reader_texture(0).unwrap().texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(&px),
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(W * 8), rows_per_image: Some(H) },
        wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
    );
}

fn frame(ctx: &engine::GpuContext, comp: &mut Compositor, graph: &Graph) -> Vec<[f32; 4]> {
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
            layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(padded), rows_per_image: Some(H) },
        },
        wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
    );
    ctx.queue.submit([encoder.finish()]);
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    let data = slice.get_mapped_range().expect("a mapped buffer");
    let mut out = Vec::new();
    for y in 0..H as usize {
        let row: &[f16] = bytemuck::cast_slice(&data[y * padded as usize..][..(W * 8) as usize]);
        for x in 0..W as usize {
            out.push([0, 1, 2, 3].map(|c| row[x * 4 + c].to_f32()));
        }
    }
    drop(data);
    staging.unmap();
    out
}

fn graph(p: ExpressionParams) -> Graph {
    let mut g = Graph::default();
    g.nodes.insert(0, NodeKind::read(0, ""));
    g.nodes.insert(1, NodeKind::Expression(p));
    g.edges.insert((1, 0), 0);
    g.view = Some(1);
    g
}

/// Half floats out, so a relative tolerance; infinities and NaN by class.
fn same(got: f32, want: f32) -> bool {
    // Past half float's range the output texture holds the edge — Metal
    // SATURATES a store to 65504 where an IEEE conversion would give inf.
    if want.is_finite() && want.abs() > 65504.0 {
        return got.signum() == want.signum() && (got.abs() == 65504.0 || got.is_infinite());
    }
    // What the output texture can hold of `want`.
    let want = f16::from_f32(want).to_f32();
    if want.is_nan() || got.is_nan() {
        return want.is_nan() && got.is_nan();
    }
    if want.is_infinite() || got.is_infinite() {
        return want == got;
    }
    (got - want).abs() <= 0.002 + 0.002 * want.abs()
}

/// Each expression, and whether it must match the model on the special
/// values too. A TEST of a special value is exact (it is done on bits); the
/// ARITHMETIC of one — `sin(nan)`, `min(nan, 2)` — is whatever the GPU's
/// fast math makes it, so those are judged on ordinary pixels only.
const EXPRS: [(&str, bool); 15] = [
    // The two that asked for the node.
    ("r == 0 ? 1:0", true),
    ("r == inf ? 1 : r", true),
    // Their relatives.
    ("isinf(r) ? 0 : r", true),
    ("isnan(r) ? 0 : r", true),
    ("r == -inf ? 2 : r", true),
    ("r == nan ? 3 : r", true),
    ("r != inf && r == r ? 1 : 0", true),
    ("(r < 1) + (r >= 1) * 2 + (r != r) * 4 + !r * 8", true),
    // Everything else the language does, through the GPU.
    ("k = r * 2 + g; clamp(k, 0, 1) ^ 2 + (x > 6 ? 1 : 0) + y * 0.01 + frame * 0.01", false),
    ("smoothstep(0, 1, r) + smoothstep(1, 1, r)", false),
    ("-7 % 3 + r % 0.5", false),
    ("(r - 0.5) ^ 3 + (-2) ^ 2 + abs(r) ^ 0.5", false),
    ("fract(r * 3.5) + round(r * 4) + sign(r) + trunc(r * 1.5)", false),
    ("atan2(r, g) + step(0.5, r) + min(r, 2) - max(-r, -9) + mix(0, r, 0.5)", false),
    ("!r + (r > 0.2 || g > 0.4) + width * 0.001 - height * 0.001 + exp2(1) + log2(4)", false),
];

#[test]
fn every_expression_renders_what_the_model_says() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp);
    for (src, specials) in EXPRS {
        let p = ExpressionParams {
            exprs: [src.to_string(), String::new(), "b * 2".into(), String::new()],
            time: 7.0,
        };
        let out = frame(&ctx, &mut comp, &graph(p.clone()));
        for y in 0..H {
            for x in 0..W {
                let env = Env {
                    rgba: source(x, y).map(|v| f16::from_f32(v).to_f32()),
                    xy: [x as f32 + 0.5, y as f32 + 0.5],
                    size: [W as f32, H as f32],
                    frame: 7.0,
                };
                if !specials && !env.rgba[0].is_finite() {
                    continue;
                }
                let want = p.apply(&env);
                let got = out[(y * W + x) as usize];
                for c in 0..4 {
                    assert!(
                        same(got[c], want[c]),
                        "`{src}` at x={x} y={y} (r = {}) ch{c}: GPU {} vs the model's {}",
                        env.rgba[0],
                        got[c],
                        want[c]
                    );
                }
            }
        }
    }
}

/// A field that does not parse passes its channel through and the rest of
/// the node still acts — the comp keeps rendering while you type.
#[test]
fn a_broken_field_passes_through() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp);
    let p = ExpressionParams {
        exprs: ["r == inf 0: 1 : r".into(), "1".into(), String::new(), String::new()],
        time: 0.0,
    };
    let out = frame(&ctx, &mut comp, &graph(p));
    for x in [0u32, 1, 3, 7] {
        let src = source(x, 0);
        let got = out[x as usize];
        assert!(same(got[0], src[0]), "x={x}: the broken red field moved red");
        assert!(same(got[1], 1.0), "x={x}: the good green field did not act");
        assert!(same(got[2], src[2]), "x={x}: an empty field moved blue");
    }
}
