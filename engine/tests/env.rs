//! The Env node on real pixels: every output pixel is the latlong sampled
//! where the CPU mirror (`EnvParams::lookup`) says, for the manual view, a
//! wired Camera (lowered the way the app lowers it), spin and exposure; a
//! camera's translation changes nothing. Self-skips without a GPU (CI).

use engine::compositor::Compositor;
use engine::decode::ImageInfo;
use engine::graph::{EnvParams, Graph, NodeKind};
use engine::scene3d::{CameraParams, Xform};
use engine::timing::GpuTimer;
use half::f16;

/// The latlong.
const SW: u32 = 64;
const SH: u32 = 32;
/// The camera frame.
const W: u32 = 32;
const H: u32 = 18;

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP env test: no adapter ({e})");
            None
        }
    }
}

/// Smooth in both directions, so a bilinear read at `q` is the function at
/// `q`: red runs down the latitude, green and blue round the longitude.
fn latlong_at(q: [f32; 2]) -> [f32; 3] {
    let a = std::f32::consts::TAU * q[0] / SW as f32;
    [q[1] / SH as f32, 0.5 + 0.5 * a.cos(), 0.5 + 0.5 * a.sin()]
}

fn upload(ctx: &engine::GpuContext, comp: &mut Compositor) {
    comp.ensure_reader(0, ImageInfo { width: SW, height: SH });
    let px: Vec<[f16; 4]> = (0..SW * SH)
        .map(|i| {
            let c = latlong_at([(i % SW) as f32 + 0.5, (i / SW) as f32 + 0.5]);
            [c[0], c[1], c[2], 1.0].map(f16::from_f32)
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
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(SW * 8), rows_per_image: Some(SH) },
        wgpu::Extent3d { width: SW, height: SH, depth_or_array_layers: 1 },
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

/// Read -> Env, with a Camera on port 1 when given, lowered as the app
/// lowers it.
fn graph(p: EnvParams, cam: Option<CameraParams>) -> Graph {
    let mut g = Graph::default();
    g.nodes.insert(0, NodeKind::read(0, ""));
    g.nodes.insert(1, NodeKind::Env(p));
    g.edges.insert((1, engine::nodes::env::IMAGE_INPUT), 0);
    if let Some(c) = cam {
        g.nodes.insert(2, NodeKind::Camera3D(c));
        g.edges.insert((1, engine::nodes::env::CAMERA_INPUT), 2);
    }
    g.formats.insert(0, (SW, SH));
    g.formats.insert(1, (W, H));
    g.view = Some(1);
    engine::nodes::env::lower(&mut g);
    g
}

fn setup() -> Option<(engine::GpuContext, Compositor)> {
    let ctx = ctx()?;
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp);
    Some((ctx, comp))
}

/// Every pixel against the mirror, away from the seam and the poles
/// (where the bilinear read clamps at the picture's edge). `gain` is the
/// exposure's.
fn check(out: &[[f32; 4]], p: &EnvParams, gain: f32, what: &str) {
    let mut checked = 0;
    for y in 0..H {
        for x in 0..W {
            let q = p.lookup([x as f32 + 0.5, y as f32 + 0.5], [W as f32, H as f32], [SW as f32, SH as f32]);
            if q[0] < 1.0 || q[0] > SW as f32 - 1.0 || q[1] < 1.0 || q[1] > SH as f32 - 1.0 {
                continue;
            }
            let want = latlong_at(q).map(|c| c * gain);
            let got = out[(y * W + x) as usize];
            for c in 0..3 {
                assert!(
                    (got[c] - want[c]).abs() < 6e-3 * gain,
                    "{what}: ({x}, {y}) rendered {got:?}, the mirror reads {q:?} = {want:?}"
                );
            }
            checked += 1;
        }
    }
    assert!(checked > (W * H / 2) as usize, "{what}: only {checked} pixels checked");
}

#[test]
fn the_manual_view_samples_where_the_mirror_says() {
    let Some((ctx, mut comp)) = setup() else { return };
    for p in [
        EnvParams::default(),
        EnvParams { rotate: [25.0, -60.0, 10.0], focal: 18.0, ..Default::default() },
        EnvParams { rotate: [-80.0, 200.0, 0.0], focal: 50.0, spin: 33.0, ..Default::default() },
    ] {
        let out = frame(&ctx, &mut comp, &graph(p.clone(), None));
        check(&out, &p, 1.0, &format!("{p:?}"));
    }
}

#[test]
fn a_wired_camera_drives_it_and_its_translation_does_nothing() {
    let Some((ctx, mut comp)) = setup() else { return };
    let cam = CameraParams {
        xform: Xform { rotate: [12.0, 140.0, -7.0], ..Default::default() },
        focal: 24.0,
        ..Default::default()
    };
    let g = graph(EnvParams::default(), Some(cam.clone()));
    let Some(NodeKind::Env(lowered)) = g.nodes.get(&1).cloned() else { panic!() };
    assert!(lowered.cam.is_some(), "the camera was not lowered onto the node");
    let out = frame(&ctx, &mut comp, &g);
    check(&out, &lowered, 1.0, "wired camera");

    let moved = CameraParams { xform: Xform { translate: [40.0, -3.0, 900.0], ..cam.xform }, ..cam };
    let out2 = frame(&ctx, &mut comp, &graph(EnvParams::default(), Some(moved)));
    assert_eq!(out, out2, "a camera's translation moved the background at infinity");
}

#[test]
fn exposure_is_stops() {
    let Some((ctx, mut comp)) = setup() else { return };
    let p = EnvParams { exposure: 1.0, rotate: [0.0, 45.0, 0.0], ..Default::default() };
    let out = frame(&ctx, &mut comp, &graph(p.clone(), None));
    check(&out, &p, 2.0, "exposure +1");
}
