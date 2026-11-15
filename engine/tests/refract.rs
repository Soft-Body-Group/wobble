//! The Refract node on real pixels: every output pixel is input 0 sampled
//! where the CPU mirror (`RefractParams::lookup`) says, for a normal map
//! and a height map, behind a plate and under a latlong; dispersion splits
//! the channels by Snell at three indices; roughness keeps a flat colour
//! flat. Self-skips without a GPU (CI).

use engine::compositor::Compositor;
use engine::decode::ImageInfo;
use engine::graph::{Graph, NodeKind, NormalY, RefractMap, RefractMode, RefractParams};
use engine::nodes::refract::{CAMERA_INPUT, IMAGE_INPUT, MAP_INPUT};
use engine::timing::GpuTimer;
use half::f16;

/// Input 0 (a plate, or a latlong).
const SW: u32 = 64;
const SH: u32 = 32;
/// The map, and so the picture.
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
/// `q`.
fn source_at(q: [f32; 2]) -> [f32; 3] {
    let a = std::f32::consts::TAU * q[0] / SW as f32;
    [q[1] / SH as f32, 0.5 + 0.5 * a.cos(), 0.5 + 0.5 * a.sin()]
}

fn upload(ctx: &engine::GpuContext, comp: &mut Compositor, slot: usize, w: u32, h: u32, f: impl Fn([f32; 2]) -> [f32; 3]) {
    comp.ensure_reader(slot, ImageInfo { width: w, height: h });
    let px: Vec<[f16; 4]> = (0..w * h)
        .map(|i| {
            let c = f([(i % w) as f32 + 0.5, (i / w) as f32 + 0.5]);
            [c[0], c[1], c[2], 1.0].map(f16::from_f32)
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
        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 8), rows_per_image: Some(h) },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
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

/// Read -> Refract <- Read (the map), lowered as the app lowers it.
fn graph(p: RefractParams) -> Graph {
    let mut g = Graph::default();
    g.nodes.insert(0, NodeKind::read(0, ""));
    g.nodes.insert(1, NodeKind::read(1, ""));
    g.nodes.insert(2, NodeKind::Refract(p));
    g.edges.insert((2, IMAGE_INPUT), 0);
    g.edges.insert((2, MAP_INPUT), 1);
    g.formats.insert(0, (SW, SH));
    g.formats.insert(1, (W, H));
    g.formats.insert(2, (W, H));
    g.view = Some(2);
    engine::nodes::refract::lower(&mut g);
    assert!(g.input_of(2, CAMERA_INPUT).is_none());
    g
}

/// A compositor with the source up and the map `f` in reader 1.
fn setup(f: impl Fn([f32; 2]) -> [f32; 3]) -> Option<(engine::GpuContext, Compositor)> {
    let ctx = ctx()?;
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp, 0, SW, SH, source_at);
    upload(&ctx, &mut comp, 1, W, H, f);
    Some((ctx, comp))
}

const FMT: [f32; 2] = [W as f32, H as f32];
const SRC: [f32; 2] = [SW as f32, SH as f32];

/// Every pixel against the mirror through the tangent normal `n`, away
/// from the source's edge (where a clamped bilinear read differs) and
/// `margin` pixels in from the picture's. `chans` is checked.
fn check(out: &[[f32; 4]], p: &RefractParams, n: [f32; 3], margin: u32, chans: &[usize], what: &str) {
    let mut checked = 0;
    for y in margin..H - margin {
        for x in margin..W - margin {
            let q = p.lookup([x as f32 + 0.5, y as f32 + 0.5], n, FMT, SRC, 1.0);
            if q[0] < 1.0 || q[0] > SRC[0] - 1.0 || q[1] < 1.0 || q[1] > SRC[1] - 1.0 {
                continue;
            }
            let want = source_at(q);
            let got = out[(y * W + x) as usize];
            for &c in chans {
                assert!(
                    (got[c] - want[c]).abs() < 6e-3,
                    "{what}: ({x}, {y}) rendered {got:?}, the mirror reads {q:?} = {want:?}"
                );
            }
            checked += 1;
        }
    }
    assert!(checked > (W * H / 3) as usize, "{what}: only {checked} pixels checked");
}

/// A normal-map colour for a tangent normal, green up.
fn encode(n: [f32; 3]) -> [f32; 3] {
    let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    n.map(|c| c / l * 0.5 + 0.5)
}

#[test]
fn a_normal_map_bends_a_plate_and_a_latlong_where_the_mirror_says() {
    let tilt = [0.25, -0.15, 1.0];
    let Some((ctx, mut comp)) = setup(move |_| encode(tilt)) else { return };
    for mode in [RefractMode::Background, RefractMode::Latlong] {
        for normal_y in [NormalY::Up, NormalY::Down] {
            let p = RefractParams {
                mode,
                map: RefractMap::Normal,
                normal_y,
                ior: 1.45,
                thickness: 12.0,
                rotate: [10.0, 70.0, 0.0],
                spin: 20.0,
                ..Default::default()
            };
            let out = frame(&ctx, &mut comp, &graph(p.clone()));
            // What the shader decodes from the stored half floats.
            let n = p.normal_of_map(encode(tilt).map(|c| f16::from_f32(c).to_f32()));
            check(&out, &p, n, 0, &[0, 1, 2], &format!("{mode:?} {normal_y:?}"));
        }
    }
}

/// Flat glass under a latlong is the Env node, pixel for pixel in the
/// mirror; and with no map wired at all the glass is flat.
#[test]
fn flat_glass_is_the_picture_behind_it() {
    let Some((ctx, mut comp)) = setup(|_| [0.5, 0.5, 1.0]) else { return };
    for mode in [RefractMode::Background, RefractMode::Latlong] {
        let p = RefractParams { mode, map: RefractMap::Normal, ..Default::default() };
        let mut g = graph(p.clone());
        let out = frame(&ctx, &mut comp, &g);
        // 0.5 is not exactly zero tilt in a half float; the mirror gets
        // what the shader decodes.
        let n = p.normal_of_map([0.5, 0.5, 1.0].map(|c| f16::from_f32(c).to_f32()));
        check(&out, &p, n, 0, &[0, 1, 2], "flat map");
        g.edges.remove(&(2, MAP_INPUT));
        let out = frame(&ctx, &mut comp, &g);
        check(&out, &p, [0.0, 0.0, 1.0], 0, &[0, 1, 2], "no map");
    }
}

/// A plane of height has one slope everywhere, so one normal.
#[test]
fn a_height_map_is_read_as_its_slope() {
    let (a, b) = (0.02f32, -0.03f32);
    let Some((ctx, mut comp)) = setup(move |q| [0.5 + a * q[0] + b * q[1]; 3]) else { return };
    for mode in [RefractMode::Background, RefractMode::Latlong] {
        let p = RefractParams { mode, height: 12.0, thickness: 20.0, ..Default::default() };
        let out = frame(&ctx, &mut comp, &graph(p.clone()));
        // The edge column differences against nothing; half floats cost
        // the slope its last digits.
        check(&out, &p, p.normal_of_slope([a, b], 1.0), 2, &[0, 1, 2], &format!("{mode:?}"));
    }
}

/// Three taps, three indices: red through `ior - d`, green `ior`, blue
/// `ior + d`.
#[test]
fn dispersion_splits_the_channels_by_index() {
    let tilt = [0.4, 0.2, 1.0];
    let Some((ctx, mut comp)) = setup(move |_| encode(tilt)) else { return };
    let p = RefractParams {
        map: RefractMap::Normal,
        thickness: 30.0,
        dispersion: 0.2,
        samples: 1,
        ..Default::default()
    };
    assert_eq!(p.taps(), 3);
    let out = frame(&ctx, &mut comp, &graph(p.clone()));
    let n = p.normal_of_map(encode(tilt).map(|c| f16::from_f32(c).to_f32()));
    for (c, ior) in [(0, p.ior - 0.2), (1, p.ior), (2, p.ior + 0.2)] {
        let one = RefractParams { ior, dispersion: 0.0, ..p.clone() };
        check(&out, &one, n, 0, &[c], &format!("channel {c}"));
    }
}

/// Roughness scatters the read but invents nothing: alpha holds, the
/// picture stays inside the source's range, and it is not the polished one.
#[test]
fn roughness_blurs_without_inventing_light() {
    let Some((ctx, mut comp)) = setup(|_| [0.5, 0.5, 1.0]) else { return };
    let sharp = RefractParams { map: RefractMap::Normal, thickness: 40.0, ..Default::default() };
    let rough = RefractParams { roughness: 0.3, samples: 32, ..sharp.clone() };
    let a = frame(&ctx, &mut comp, &graph(sharp));
    let b = frame(&ctx, &mut comp, &graph(rough));
    let mut moved = 0.0;
    for (pa, pb) in a.iter().zip(&b) {
        assert!(pb.iter().all(|c| c.is_finite() && (-1e-3..=1.001).contains(c)), "{pb:?}");
        moved += (pa[1] - pb[1]).abs();
    }
    assert!(moved > 0.5, "roughness changed nothing ({moved})");
    let mid = b[(H / 2 * W + W / 2) as usize];
    assert!((mid[3] - 1.0).abs() < 1e-3, "alpha {mid:?}");
}
