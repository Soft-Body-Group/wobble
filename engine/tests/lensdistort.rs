//! Lens Distortion on real pixels. The source is a ramp (red = x, green =
//! y, blue = x), so a bilinear sample reads its own position back as
//! colour, and every check is a position: undistort reads where the CPU
//! mirror (`LensDistortParams::src_of`) says, chromatic aberration moves red
//! and blue by their own magnification, the STMap output IS the lookup (and
//! a UV Warp fed it reproduces the node), and undistort followed by distort
//! — overscanned in between — comes back to the plate. Self-skips without a
//! GPU (CI).

use engine::compositor::Compositor;
use engine::decode::ImageInfo;
use engine::graph::{Graph, LensDistortParams, LensMode, LensOutput, NodeKind, UvWarpParams};
use engine::timing::GpuTimer;
use half::f16;

const W: u32 = 64;
const H: u32 = 40;

fn source(x: u32, y: u32) -> [f32; 4] {
    [x as f32 / W as f32, y as f32 / H as f32, x as f32 / W as f32, 1.0]
}

/// What the ramp reads at source position `q` (pixels, centres at +0.5).
fn ramp_at(q: [f32; 2]) -> [f32; 2] {
    [(q[0] - 0.5) / W as f32, (q[1] - 0.5) / H as f32]
}

/// Well inside the source, where bilinear has both neighbours.
fn inside(q: [f32; 2]) -> bool {
    q[0] > 1.0 && q[0] < W as f32 - 1.0 && q[1] > 1.0 && q[1] < H as f32 - 1.0
}

fn lens() -> LensDistortParams {
    LensDistortParams {
        k1: -0.15,
        k2: 0.04,
        k3: -0.005,
        p1: 0.003,
        p2: -0.002,
        center: [0.53, 0.46],
        squeeze: 1.2,
        zoom: 1.05,
        ..Default::default()
    }
}

const SIZE: [f32; 2] = [W as f32, H as f32];

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP lensdistort test: no adapter ({e})");
            None
        }
    }
}

fn upload(ctx: &engine::GpuContext, comp: &mut Compositor, slot: usize, f: impl Fn(u32, u32) -> [f32; 4]) {
    comp.ensure_reader(slot, ImageInfo { width: W, height: H });
    let px: Vec<[f16; 4]> = (0..W * H).map(|i| f(i % W, i / W).map(f16::from_f32)).collect();
    ctx.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &comp.reader_texture(slot).unwrap().texture,
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
    let mut out = Vec::new();
    for y in 0..H as usize {
        let row: &[f16] = bytemuck::cast_slice(&data[y * padded as usize..][..(W * 8) as usize]);
        for x in 0..W as usize {
            out.push([
                row[x * 4].to_f32(),
                row[x * 4 + 1].to_f32(),
                row[x * 4 + 2].to_f32(),
                row[x * 4 + 3].to_f32(),
            ]);
        }
    }
    drop(data);
    staging.unmap();
    out
}


fn one(p: LensDistortParams) -> Graph {
    let mut g = Graph::default();
    g.nodes.insert(0, NodeKind::read(0, ""));
    g.nodes.insert(1, NodeKind::LensDistort(p));
    g.edges.insert((1, 0), 0);
    g.formats.insert(0, (W, H));
    g.formats.insert(1, (W, H));
    g.view = Some(1);
    g
}

fn setup() -> Option<(engine::GpuContext, Compositor)> {
    let ctx = ctx()?;
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp, 0, source);
    Some((ctx, comp))
}

fn px(q: [u32; 2]) -> [f32; 2] {
    [q[0] as f32 + 0.5, q[1] as f32 + 0.5]
}

/// Both ways round, the shader samples where the mirror says, and pixels
/// the mirror calls invalid are transparent.
#[test]
fn each_mode_samples_where_the_mirror_says() {
    let Some((ctx, mut comp)) = setup() else { return };
    for mode in LensMode::ALL {
        let p = LensDistortParams { mode, ..lens() };
        let geom = p.geom(SIZE, SIZE);
        let out = frame(&ctx, &mut comp, &one(p));
        let mut checked = 0;
        for y in 0..H {
            for x in 0..W {
                let got = out[(y * W + x) as usize];
                match p.src_of(&geom, px([x, y]), 1.0) {
                    Some(q) if inside(q) => {
                        let want = ramp_at(q);
                        assert!(
                            (got[0] - want[0]).abs() < 2e-3 && (got[1] - want[1]).abs() < 2e-3,
                            "{mode:?} ({x}, {y}): got {got:?}, the mirror reads {q:?} = {want:?}"
                        );
                        checked += 1;
                    }
                    None => assert_eq!(got[3], 0.0, "{mode:?} ({x}, {y}) should be transparent"),
                    _ => {}
                }
            }
        }
        assert!(checked > (W * H / 2) as usize, "{mode:?}: only {checked} pixels checked");
    }
}

/// Red and blue are read at their own magnification, green at none.
#[test]
fn chromatic_aberration_moves_red_and_blue() {
    let Some((ctx, mut comp)) = setup() else { return };
    for mode in LensMode::ALL {
        let p = LensDistortParams { mode, chromatic: true, ca: [0.02, -0.015], ..lens() };
        let geom = p.geom(SIZE, SIZE);
        let out = frame(&ctx, &mut comp, &one(p));
        let mut moved = 0;
        for y in (2..H - 2).step_by(3) {
            for x in (2..W - 2).step_by(3) {
                let got = out[(y * W + x) as usize];
                let (Some(qr), Some(qg), Some(qb)) = (
                    p.src_of(&geom, px([x, y]), 1.02),
                    p.src_of(&geom, px([x, y]), 1.0),
                    p.src_of(&geom, px([x, y]), 0.985),
                ) else {
                    continue;
                };
                if !(inside(qr) && inside(qg) && inside(qb)) {
                    continue;
                }
                assert!((got[0] - ramp_at(qr)[0]).abs() < 2e-3, "{mode:?} red at ({x}, {y})");
                assert!((got[1] - ramp_at(qg)[1]).abs() < 2e-3, "{mode:?} green at ({x}, {y})");
                assert!((got[2] - ramp_at(qb)[0]).abs() < 2e-3, "{mode:?} blue at ({x}, {y})");
                if (got[0] - got[2]).abs() > 0.01 {
                    moved += 1;
                }
            }
        }
        assert!(moved > 20, "{mode:?}: red and blue barely separated ({moved})");
    }
}

/// The STMap output is the lookup in UV Warp's convention, and a UV Warp
/// fed it renders what the node renders.
#[test]
fn the_stmap_drives_a_uv_warp_to_the_same_picture() {
    let Some((ctx, mut comp)) = setup() else { return };
    for mode in LensMode::ALL {
        let p = LensDistortParams { mode, ..lens() };
        let geom = p.geom(SIZE, SIZE);
        let map = frame(&ctx, &mut comp, &one(LensDistortParams { output: LensOutput::StMap, ..p }));
        for &(x, y) in &[(10u32, 7u32), (32, 20), (55, 33)] {
            let q = p.src_of(&geom, px([x, y]), 1.0).unwrap();
            let got = map[(y * W + x) as usize];
            let want = [q[0] / W as f32, 1.0 - q[1] / H as f32];
            assert!(
                (got[0] - want[0]).abs() < 1e-3 && (got[1] - want[1]).abs() < 1e-3 && got[3] == 1.0,
                "{mode:?} map at ({x}, {y}): {got:?} vs {want:?}"
            );
        }
        let direct = frame(&ctx, &mut comp, &one(p));
        let mut g = Graph::default();
        g.nodes.insert(0, NodeKind::read(0, ""));
        g.nodes.insert(1, NodeKind::LensDistort(LensDistortParams { output: LensOutput::StMap, ..p }));
        g.nodes.insert(2, NodeKind::UvWarp(UvWarpParams::default()));
        g.edges.insert((1, 0), 0);
        g.edges.insert((2, 0), 0);
        g.edges.insert((2, 1), 1);
        for id in 0..3 {
            g.formats.insert(id, (W, H));
        }
        g.view = Some(2);
        let warped = frame(&ctx, &mut comp, &g);
        for y in 2..H - 2 {
            for x in 2..W - 2 {
                let (a, b) = (direct[(y * W + x) as usize], warped[(y * W + x) as usize]);
                if a[3] == 1.0 {
                    assert!(
                        (0..3).all(|c| (a[c] - b[c]).abs() < 3e-3),
                        "{mode:?} ({x}, {y}): node {a:?}, uv warp {b:?}"
                    );
                }
            }
        }
    }
}

/// Undistort into an overscanned format, distort back out of it: the
/// plate, pixel for pixel, wherever the round trip is defined.
#[test]
fn undistort_then_distort_comes_back_to_the_plate() {
    let Some((ctx, mut comp)) = setup() else { return };
    for chromatic in [false, true] {
        let un = LensDistortParams { overscan: 1.25, chromatic, ca: [0.01, -0.01], ..lens() };
        let di = LensDistortParams { mode: LensMode::Distort, ..un };
        let mid = un.output_format((W, H));
        assert_eq!(di.output_format(mid), (W, H));
        let mut g = Graph::default();
        g.nodes.insert(0, NodeKind::read(0, ""));
        g.nodes.insert(1, NodeKind::LensDistort(un));
        g.nodes.insert(2, NodeKind::LensDistort(di));
        g.edges.insert((1, 0), 0);
        g.edges.insert((2, 0), 1);
        g.formats.insert(0, (W, H));
        g.formats.insert(1, mid);
        g.formats.insert(2, (W, H));
        g.view = Some(2);
        let out = frame(&ctx, &mut comp, &g);
        let mut worst = 0.0f32;
        let mut checked = 0;
        for y in 1..H - 1 {
            for x in 1..W - 1 {
                let got = out[(y * W + x) as usize];
                if got[3] < 1.0 {
                    continue;
                }
                let want = source(x, y);
                for c in 0..3 {
                    worst = worst.max((got[c] - want[c]).abs());
                }
                checked += 1;
            }
        }
        assert!(checked as u32 > (W - 2) * (H - 2) * 9 / 10, "chromatic {chromatic}: only {checked} valid");
        assert!(worst < 2e-3, "chromatic {chromatic}: round trip off by {worst} (1 px = {})", 1.0 / W as f32);
    }
}

/// The default node is the identity, to the pixel.
#[test]
fn the_default_node_is_the_identity() {
    let Some((ctx, mut comp)) = setup() else { return };
    let out = frame(&ctx, &mut comp, &one(LensDistortParams::default()));
    for y in 0..H {
        for x in 0..W {
            let (got, want) = (out[(y * W + x) as usize], source(x, y));
            assert!((0..4).all(|c| (got[c] - want[c]).abs() <= 2e-3), "({x}, {y}) {got:?}");
        }
    }
}
