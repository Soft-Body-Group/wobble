//! The LUT node on real pixels: tables written to disk, loaded through the
//! node's path, rendered through the fused pass and read back against the
//! CPU mirror (`LutParams::apply` over `lut::Lut::apply`). Self-skips
//! without a GPU (CI).

use engine::color::Colorspace;
use engine::compositor::Compositor;
use engine::decode::ImageInfo;
use engine::graph::{Graph, LutOut, LutParams, LutShaper, NodeKind};
use engine::lut::Interp;
use engine::timing::GpuTimer;
use engine::viewtf::{Gamut, ViewSettings};
use half::f16;
use std::path::PathBuf;

const W: u32 = 16;
const H: u32 = 4;

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP lut test: no adapter ({e})");
            None
        }
    }
}

/// Values all over 0..1, off the table's nodes, plus a row past both ends
/// (row 3) for the clamps.
fn source(x: u32, y: u32) -> [f32; 4] {
    let t = x as f32 / (W - 1) as f32;
    match y {
        0 => [t, 1.0 - t, 0.5 * t + 0.13, 1.0],
        1 => [(t * 7.3).fract(), (t * 3.1 + 0.2).fract(), (t * 5.7 + 0.6).fract(), 0.5],
        2 => [0.18 * (t * 3.0).exp2(), 0.02 + 0.3 * t, 0.9 - 0.8 * t, 1.0],
        _ => [-0.5 + 2.0 * t, 1.5 - t, 3.0 * t, 1.0],
    }
}

fn upload(ctx: &engine::GpuContext, comp: &mut Compositor) {
    comp.ensure_reader(0, ImageInfo { width: W, height: H });
    let px: Vec<[f16; 4]> = (0..W * H).map(|i| source(i % W, i / W).map(f16::from_f32)).collect();
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

fn graph(p: LutParams) -> Graph {
    let mut g = Graph::default();
    g.working = Colorspace::LINEAR_REC709;
    g.nodes.insert(0, NodeKind::read(0, ""));
    g.nodes.insert(1, NodeKind::Lut(p));
    g.edges.insert((1, 0), 0);
    g.view = Some(1);
    g
}

fn dir() -> PathBuf {
    let d = std::env::temp_dir().join(format!("wobble-lut-gpu-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// A `.cube` of `n`^3 entries from `f`, red fastest.
fn cube(name: &str, n: usize, f: impl Fn([f32; 3]) -> [f32; 3]) -> String {
    let mut t = format!("TITLE \"{name}\"\nLUT_3D_SIZE {n}\n");
    let s = (n - 1) as f32;
    for b in 0..n {
        for g in 0..n {
            for r in 0..n {
                let [x, y, z] = f([r as f32 / s, g as f32 / s, b as f32 / s]);
                t += &format!("{x:.7} {y:.7} {z:.7}\n");
            }
        }
    }
    let p = dir().join(format!("{name}.cube"));
    std::fs::write(&p, t).unwrap();
    p.to_str().unwrap().to_string()
}

/// Every pixel against the CPU mirror, within what a half-float output
/// holds; alpha untouched.
fn check(ctx: &engine::GpuContext, comp: &mut Compositor, p: LutParams, what: &str) {
    let lut = p.status().ok();
    let out = frame(ctx, comp, &graph(p.clone()));
    for (i, got) in out.iter().enumerate() {
        let src = source(i as u32 % W, i as u32 / W).map(|v| f16::from_f32(v).to_f32());
        let want = p.apply(lut.as_deref(), Colorspace::LINEAR_REC709, [src[0], src[1], src[2]]);
        for k in 0..3 {
            let tol = 2e-3 + 2e-3 * want[k].abs();
            assert!(
                (got[k] - want[k]).abs() <= tol,
                "{what}: pixel {i} ch {k}: {src:?} -> got {got:?}, want {want:?}"
            );
        }
        assert_eq!(got[3], src[3], "{what}: alpha moved");
    }
}

/// The table read straight: linear in, raw out.
fn direct() -> LutParams {
    LutParams { input: Colorspace::LINEAR_REC709, output: LutOut::Raw, ..Default::default() }
}

#[test]
fn identity_cube_is_the_identity_and_a_known_cube_matches_the_cpu() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp);

    let id = cube("identity", 17, |c| c);
    let out = frame(&ctx, &mut comp, &graph(LutParams { path: id, ..direct() }));
    for (i, got) in out.iter().enumerate().take((3 * W) as usize) {
        let s = source(i as u32 % W, i as u32 / W);
        for k in 0..3 {
            let want = f16::from_f32(s[k].clamp(0.0, 1.0)).to_f32();
            assert!((got[k] - want).abs() < 2e-3, "identity moved pixel {i}: {s:?} -> {got:?}");
        }
    }

    // Swap red and blue: exact on a 2-point cube, by construction.
    let swap = cube("swap", 2, |[r, g, b]| [b, g, r]);
    let out = frame(&ctx, &mut comp, &graph(LutParams { path: swap, ..direct() }));
    let s = source(5, 0);
    assert!((out[5][0] - s[2]).abs() < 2e-3 && (out[5][2] - s[0]).abs() < 2e-3, "{s:?} -> {:?}", out[5]);

    // Curved on every axis, so the tetrahedra show (trilinear would not
    // match the mirror off the diagonal).
    let bent = cube("bent", 9, |[r, g, b]| [b * b, g.sqrt(), 0.5 * r + 0.25 * g * g + 0.25 * b]);
    check(&ctx, &mut comp, LutParams { path: bent.clone(), ..direct() }, "bent, direct");
    // The same table behind the sRGB shaping, decoded back: values past
    // 1 and below 0 clamp in the table and come back in range.
    check(&ctx, &mut comp, LutParams { path: bent.clone(), ..Default::default() }, "bent, sRGB");
    check(
        &ctx,
        &mut comp,
        LutParams { path: bent.clone(), shaper: LutShaper::Log2, ..Default::default() },
        "bent, log2",
    );
    check(&ctx, &mut comp, LutParams { path: bent.clone(), mix: 0.4, ..direct() }, "bent, mix 0.4");
    // Resolve's default interpolation: the mirror switches with it. The
    // two agree on any table that is a SUM of per-axis curves (like
    // `bent`), so the products make them disagree.
    check(&ctx, &mut comp, LutParams { path: bent.clone(), interp: Interp::Trilinear, ..direct() }, "bent, trilinear");
    let prod = cube("product", 5, |[r, g, b]| [r * g, g * b, b * r]);
    let tri = LutParams { path: prod, interp: Interp::Trilinear, ..direct() };
    check(&ctx, &mut comp, tri.clone(), "product, trilinear");
    check(&ctx, &mut comp, LutParams { interp: Interp::Tetrahedral, ..tri.clone() }, "product, tetrahedral");
    let lut = tri.status().unwrap();
    let c = [0.31, 0.77, 0.12];
    let (t, l) = (lut.apply_with(c, Interp::Trilinear), lut.apply_with(c, Interp::Tetrahedral));
    assert!((0..3).any(|k| (t[k] - l[k]).abs() > 1e-3), "{t:?} vs {l:?}");
    // A Resolve DWG-project LUT: scene-linear into DaVinci Intermediate,
    // looked up, back.
    check(
        &ctx,
        &mut comp,
        LutParams { path: bent, input: Colorspace::DAVINCI_INTERMEDIATE, ..Default::default() },
        "bent, DaVinci Intermediate",
    );
}

#[test]
fn a_cinespace_csp_renders() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp);
    // A non-uniform prelut (squeezed towards black) and a bent 5^3 cube.
    let n = 5;
    let mut t = String::from("CSPLUTV100\n3D\n\nBEGIN METADATA\nmade by a test\nEND METADATA\n\n");
    for _ in 0..3 {
        t += "4\n0.0 0.1 0.4 1.0\n0.0 0.3 0.7 1.0\n";
    }
    t += &format!("\n{n} {n} {n}\n");
    let s = (n - 1) as f32;
    for b in 0..n {
        for g in 0..n {
            for r in 0..n {
                let [r, g, b] = [r as f32 / s, g as f32 / s, b as f32 / s];
                t += &format!("{:.6} {:.6} {:.6}\n", g * g, b, 0.3 * r + 0.7 * b * r);
            }
        }
    }
    let p = dir().join("bent.csp");
    std::fs::write(&p, t).unwrap();
    let params = LutParams { path: p.to_str().unwrap().into(), ..direct() };
    assert!(params.status().unwrap().shaper.is_some(), "the prelut is kept");
    check(&ctx, &mut comp, params.clone(), "csp");
    check(&ctx, &mut comp, LutParams { interp: Interp::Trilinear, ..params }, "csp, trilinear");
}

#[test]
fn a_1d_table_and_a_shaper_cube_render() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp);

    // A gamma table in spi1d over 0..2.
    let mut t = String::from("Version 1\nFrom 0.0 2.0\nLength 64\nComponents 1\n{\n");
    for i in 0..64 {
        t += &format!("  {:.7}\n", (i as f32 / 63.0).powf(0.45));
    }
    t += "}\n";
    let p = dir().join("gamma.spi1d");
    std::fs::write(&p, t).unwrap();
    check(&ctx, &mut comp, LutParams { path: p.to_str().unwrap().into(), ..direct() }, "spi1d");

    // A .cube carrying a 1D shaper and a cube, applied in that order.
    let mut c = String::from("LUT_1D_SIZE 5\nLUT_3D_SIZE 3\nLUT_1D_INPUT_RANGE 0 2\n");
    for i in 0..5 {
        let v = (i as f32 / 4.0).powi(2);
        c += &format!("{v} {v} {v}\n");
    }
    for b in 0..3 {
        for g in 0..3 {
            for r in 0..3 {
                c += &format!("{} {} {}\n", g as f32 / 2.0, b as f32 / 2.0, (r as f32 / 2.0).powi(2));
            }
        }
    }
    let p = dir().join("both.cube");
    std::fs::write(&p, c).unwrap();
    check(&ctx, &mut comp, LutParams { path: p.to_str().unwrap().into(), ..direct() }, "1D + 3D cube");
}

#[test]
fn mix_zero_a_missing_file_and_a_new_file_never_recompile() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp);
    let input: Vec<[f32; 4]> =
        (0..W * H).map(|i| source(i % W, i / W).map(|v| f16::from_f32(v).to_f32())).collect();

    let bent = cube("bent0", 5, |[r, g, b]| [g, b * b, r]);
    let out = frame(&ctx, &mut comp, &graph(LutParams { path: bent.clone(), mix: 0.0, ..Default::default() }));
    assert_eq!(out, input, "mix 0 is the input, exactly");

    let none = graph(LutParams { path: "/nowhere/at/all.cube".into(), ..direct() });
    assert!(LutParams { path: "/nowhere/at/all.cube".into(), ..Default::default() }.status().is_err());
    assert_eq!(frame(&ctx, &mut comp, &none), input, "a missing file passes through");

    // Pointing the node at another table — even another SHAPE of table —
    // is a texture and uniforms, not a new shader.
    let a = graph(LutParams { path: bent, ..direct() });
    let b = graph(LutParams { path: cube("other", 3, |[r, g, b]| [1.0 - r, g, b]), mix: 0.3, ..direct() });
    assert_eq!(a.topology_hash(), b.topology_hash());
    assert_eq!(a.topology_hash(), none.topology_hash());
    let (fa, fb) = (frame(&ctx, &mut comp, &a), frame(&ctx, &mut comp, &b));
    assert_ne!(fa, fb);
    check(&ctx, &mut comp, LutParams { path: dir().join("other.cube").to_str().unwrap().into(), mix: 0.3, ..direct() }, "other");
}

/// Display codes the Inverse View can reach: greys with some colour.
fn code(x: u32, y: u32) -> [f32; 4] {
    let i = y * W + x;
    let t = |k: u32| ((i * k + 3) % 61) as f32 / 60.0;
    let g = 0.05 + 0.9 * t(1);
    let c = [0.05 + 0.9 * t(7), 0.05 + 0.9 * t(13), 0.05 + 0.9 * t(29)];
    [0, 1, 2].map(|k| c[k] * 0.3 + g * 0.7).map(|v| v.clamp(0.05, 0.95)).into_iter().chain([1.0]).collect::<Vec<_>>()
        .try_into()
        .unwrap()
}

fn upload_codes(ctx: &engine::GpuContext, comp: &mut Compositor) {
    comp.ensure_reader(0, ImageInfo { width: W, height: H });
    let px: Vec<[f16; 4]> = (0..W * H).map(|i| code(i % W, i / W).map(f16::from_f32)).collect();
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

/// The Inverse View on the GPU: the CPU mirror within a half float, and
/// the viewer's forward transform hands back the codes it was fed.
#[test]
fn an_inverse_view_round_trips_through_the_viewer() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload_codes(&ctx, &mut comp);
    let working = Colorspace::ACESCG;
    // Input = working: the codes go into the table as they are.
    let p = LutParams { input: working, dither: 0.0, ..LutParams::inverse_view() };
    let lut = p.status().expect("the ACES table builds");
    let mut g = graph(p.clone());
    g.working = working;
    let out = frame(&ctx, &mut comp, &g);
    let view = ViewSettings { working: Gamut::AcesCg, ..Default::default() };
    let mut worst = 0.0f32;
    for (i, got) in out.iter().enumerate() {
        let src = code(i as u32 % W, i as u32 / W).map(|v| f16::from_f32(v).to_f32());
        let want = p.apply(Some(&lut), working, [src[0], src[1], src[2]]);
        for k in 0..3 {
            let tol = 2e-3 + 2e-3 * want[k].abs();
            assert!((got[k] - want[k]).abs() <= tol, "pixel {i} ch {k}: {src:?} -> got {got:?}, want {want:?}");
        }
        let back = view.display_rgb([got[0], got[1], got[2]]).unwrap();
        for k in 0..3 {
            worst = worst.max((back[k] - src[k]).abs());
        }
    }
    eprintln!("GPU inverse view round trip: worst {:.3} code values", worst * 255.0);
    assert!(worst < 1.5 / 255.0, "worst {}", worst * 255.0);

    // Dither on: the GPU matches the mirror at the same pixel, and every
    // pixel moved by at most `dither` code values before the table.
    let pd = LutParams { dither: 1.0, ..p.clone() };
    let mut gd = graph(pd.clone());
    gd.working = working;
    assert_eq!(g.topology_hash(), gd.topology_hash(), "dither is a uniform");
    let out = frame(&ctx, &mut comp, &gd);
    let mut moved = 0;
    for (i, got) in out.iter().enumerate() {
        let (x, y) = (i as u32 % W, i as u32 / W);
        let src = code(x, y).map(|v| f16::from_f32(v).to_f32());
        let c = [src[0], src[1], src[2]];
        let want = pd.apply_at(Some(&lut), working, c, Some([x as f32 + 0.5, y as f32 + 0.5]));
        for k in 0..3 {
            let tol = 2e-3 + 2e-3 * want[k].abs();
            assert!((got[k] - want[k]).abs() <= tol, "dither pixel {i} ch {k}: got {got:?}, want {want:?}");
        }
        let still = pd.apply(Some(&lut), working, c);
        let (a, b) = (view.display_rgb(want).unwrap(), view.display_rgb(still).unwrap());
        for k in 0..3 {
            let d = (a[k] - b[k]).abs();
            assert!(d <= 1.0 / 255.0 + 3e-3, "pixel {i} ch {k} moved {} code values", d * 255.0);
            if d > 1e-4 {
                moved += 1;
            }
        }
    }
    assert!(moved > out.len(), "the dither did nothing ({moved})");
}
