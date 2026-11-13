//! The Saturation node on the GPU, against the CPU mirror in
//! `engine::nodes::saturation`.
//!
//! The unit tests in that module prove the MATHS. What they cannot prove is
//! that the generated WGSL is the same maths — and this node generates a
//! different shader for each of the 3 domains x 4 luma bases, three of
//! which pull in helper functions (the ACEScct pair, the Oklab inverse)
//! whose declaration ORDER in the shader is the thing most likely to break.
//! So: render every combination, compare every one to the mirror.
//!
//! Self-skips without a GPU (CI).

use engine::color::Colorspace;
use engine::compositor::Compositor;
use engine::decode::ImageInfo;
use engine::graph::{Graph, NodeKind, SatDomain, SatLuma, SatParams};
use engine::timing::GpuTimer;
use half::f16;

const W: u32 = 16;
const H: u32 = 8;

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP saturation test: no adapter ({e})");
            None
        }
    }
}

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

fn graph_with(p: SatParams, working: Colorspace) -> Graph {
    let mut g = Graph::default();
    g.working = working;
    g.nodes.insert(0, NodeKind::read(0, ""));
    g.nodes.insert(1, NodeKind::Saturation(p));
    g.edges.insert((1, 0), 0);
    g.view = Some(1);
    g
}

/// f16 output with a scene-linear range: relative, with an absolute floor
/// for the values near zero where f16 has no relative precision left.
fn close(got: f32, want: f32, what: &str) {
    let tol = 3e-3 * want.abs().max(1.0);
    assert!((got - want).abs() < tol, "{what}: got {got}, want {want}");
}

/// The whole matrix: every domain x every luma basis, on a colour chosen so
/// the four bases genuinely disagree, in a working space that is NOT
/// Rec.709 (so a hardcoded weight triple shows up).
#[test]
fn every_domain_and_luma_matches_the_cpu_mirror() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    let plate = [0.6, 0.22, 0.06, 1.0];
    upload(&ctx, &mut comp, plate);
    let working = Colorspace::ACESCG;

    for domain in SatDomain::ALL {
        for luma in SatLuma::ALL {
            // Every non-structural control off its default too, so a param
            // packed into the wrong uniform slot cannot pass.
            let p = SatParams {
                sat: 1.7,
                boost: 0.4,
                zones: [0.5, 1.2, 0.9],
                pivot: 0.2,
                width: 2.0,
                domain,
                luma,
                preserve: 0.6,
                compress: 0.5,
            };
            let want = engine::nodes::saturation::apply(&p, working, [plate[0], plate[1], plate[2]]);
            let got = render(&ctx, &mut comp, &graph_with(p, working));
            for i in 0..3 {
                close(got[i], want[i], &format!("{domain:?}/{luma:?} channel {i}"));
            }
            close(got[3], plate[3], "alpha untouched");
        }
    }
}

/// The luma basis has to reach the shader as WEIGHTS, not as a name: the
/// working space decides them, and the same node in two comps is two
/// different corrections.
#[test]
fn the_working_space_picks_the_weights() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    let plate = [0.6, 0.22, 0.06, 1.0];
    upload(&ctx, &mut comp, plate);

    let p = SatParams { sat: 0.0, domain: SatDomain::Linear, ..Default::default() };
    let mut greys = Vec::new();
    for working in [Colorspace::LINEAR_REC709, Colorspace::ACESCG] {
        let want = engine::nodes::saturation::apply(&p, working, [plate[0], plate[1], plate[2]]);
        let got = render(&ctx, &mut comp, &graph_with(p, working));
        close(got[0], want[0], "grey");
        greys.push(got[0]);
    }
    // ...and the two really are different numbers, or a hardcoded Rec.709
    // triple would pass this.
    assert!(
        (greys[0] - greys[1]).abs() > 2e-3,
        "the two working spaces agreed: {greys:?}"
    );
}

/// Every combination has to produce a shader that COMPILES — the helper
/// functions are shared and alphabetically ordered, and WGSL wants them
/// declared before use.
#[test]
fn every_combination_compiles() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp, [0.5, 0.4, 0.3, 1.0]);
    for working in [Colorspace::LINEAR_REC709, Colorspace::ACESCG] {
        for domain in SatDomain::ALL {
            for luma in SatLuma::ALL {
                let p = SatParams { domain, luma, ..Default::default() };
                assert!(
                    comp.set_graph(&graph_with(p, working)),
                    "{domain:?}/{luma:?} in {working:?} failed to compile"
                );
            }
        }
    }
}

/// Two Saturation nodes in one comp share the helper functions. Emitting
/// `sat_lin_to_log` twice is a redefinition and the whole shader dies.
#[test]
fn two_nodes_share_their_helpers() {
    let mut g = Graph::default();
    g.working = Colorspace::ACESCG;
    g.nodes.insert(0, NodeKind::read(0, ""));
    for (i, domain) in [SatDomain::Log, SatDomain::Log, SatDomain::Perceptual]
        .into_iter()
        .enumerate()
    {
        let n = 1 + i as u64;
        g.nodes.insert(n, NodeKind::Saturation(SatParams { domain, ..Default::default() }));
        g.edges.insert((n, 0), n - 1);
    }
    g.view = Some(3);
    let plan = engine::codegen::generate(&g);
    let src = &plan.main.wgsl;
    for f in ["fn sat_lin_to_log", "fn sat_log_to_lin", "fn ok_to_linear"] {
        assert_eq!(src.matches(f).count(), 1, "{f} is not emitted exactly once");
    }
    // ...and each helper is declared before anything calls it.
    for (helper, caller) in [
        ("fn sat_lin_to_log", "fn sat_op_log_"),
        ("fn ok_to_linear", "fn sat_op_ok_"),
    ] {
        assert!(
            src.find(helper).unwrap() < src.find(caller).unwrap(),
            "{helper} is declared after {caller}"
        );
    }
}

/// The domain and the luma basis pick the generated code, so they have to
/// move the topology hash — otherwise changing one would keep the old
/// shader. Everything else on the node is a uniform, so sliding saturation
/// must NOT recompile.
#[test]
fn the_domain_and_luma_are_structural_and_nothing_else_is() {
    let hash = |p: SatParams| graph_with(p, Colorspace::ACESCG).topology_hash();
    let base = hash(SatParams::default());
    assert_ne!(
        base,
        hash(SatParams { domain: SatDomain::Linear, ..Default::default() }),
        "domain did not recompile"
    );
    assert_ne!(
        base,
        hash(SatParams { luma: SatLuma::Rec709, ..Default::default() }),
        "luma basis did not recompile"
    );
    assert_eq!(
        base,
        hash(SatParams {
            sat: 2.4,
            boost: 1.0,
            zones: [0.2, 1.4, 0.7],
            pivot: 0.09,
            width: 4.0,
            preserve: 1.0,
            compress: 1.0,
            ..Default::default()
        }),
        "a value edit recompiled"
    );
}

/// Setups written against an earlier default must load, and a missing field
/// must come back neutral rather than as a silent correction.
#[cfg(feature = "serde")]
#[test]
fn partial_setups_load_as_written() {
    let p: SatParams = serde_json::from_str(r#"{"sat": 1.5}"#).unwrap();
    assert_eq!(p.sat, 1.5);
    assert_eq!(p.zones, [1.0; 3]);
    assert_eq!(p.domain, SatDomain::Log);
    let s = serde_json::to_string(&p).unwrap();
    assert_eq!(serde_json::from_str::<SatParams>(&s).unwrap(), p);
}
