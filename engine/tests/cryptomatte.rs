//! Cryptomatte end to end: a real EXR with f32 id channels and a manifest
//! decodes through the reader as exact chunks, and the fused node turns two
//! rank textures plus a pick list into the coverage sum the spec defines.
//! The GPU half self-skips without an adapter (CI).

use engine::compositor::Compositor;
use engine::cryptomatte::{self, CryptoPick, CryptomatteParams};
use engine::decode::{ExrDecoder, ExrsDecoder, ImageInfo};
use engine::graph::{Graph, NodeKind};
use engine::timing::GpuTimer;
use half::f16;

const W: u32 = 64;
const H: u32 = 8;

/// A rank plate: `(id, coverage)` per column, the same down every row.
fn rank_px(cols: impl Fn(u32) -> (u32, f32)) -> Vec<[f16; 4]> {
    (0..W * H)
        .map(|i| {
            let (id, cov) = cols(i % W);
            let c = cryptomatte::encode_id(id);
            [c[0], c[1], c[2], f16::from_f32(cov)]
        })
        .collect()
}

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP cryptomatte test: no adapter ({e})");
            None
        }
    }
}

fn upload(ctx: &engine::GpuContext, comp: &mut Compositor, slot: usize, px: &[[f16; 4]]) {
    comp.ensure_reader(slot, ImageInfo { width: W, height: H });
    ctx.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &comp.reader_texture(slot).unwrap().texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytemuck::cast_slice(px),
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(W * 8),
            rows_per_image: Some(H),
        },
        wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
    );
}

/// Middle row, all four channels.
fn row(ctx: &engine::GpuContext, comp: &mut Compositor, graph: &Graph) -> Vec<[f32; 4]> {
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
    let off = (padded * (H / 2)) as usize;
    let halves: &[f16] = bytemuck::cast_slice(&data[off..off + (W * 8) as usize]);
    let out = (0..W as usize)
        .map(|i| std::array::from_fn(|k| halves[i * 4 + k].to_f32()))
        .collect();
    drop(data);
    staging.unmap();
    out
}

/// Cryptomatte(4) as the app lowers it: its own reader slots — 0 the
/// beauty, 1 and 2 ranks 0 and 1, the rest unminted (unwired ranks).
fn graph(mut p: CryptomatteParams) -> Graph {
    let mut g = Graph::default();
    p.sources = vec![0, 1, 2, usize::MAX, usize::MAX, usize::MAX, usize::MAX];
    g.nodes.insert(4, NodeKind::Cryptomatte(p));
    g.view = Some(4);
    g
}

fn pick(name: &str) -> CryptoPick {
    CryptoPick { name: name.into(), id: cryptomatte::id_of_name(name) }
}

/// Left third: cube 1.0. Middle third: cube 0.6 over sphere 0.4 (a soft
/// edge, two ranks). Right third: sphere 1.0.
fn scene(ctx: &engine::GpuContext, comp: &mut Compositor) {
    let (cube, sphere) = (cryptomatte::id_of_name("cube"), cryptomatte::id_of_name("sphere"));
    let third = W / 3;
    upload(ctx, comp, 0, &rank_px(|_| (0, 1.0)));
    upload(
        ctx,
        comp,
        1,
        &rank_px(|x| match x / third {
            0 => (cube, 1.0),
            1 => (cube, 0.6),
            _ => (sphere, 1.0),
        }),
    );
    upload(
        ctx,
        comp,
        2,
        &rank_px(|x| match x / third {
            1 => (sphere, 0.4),
            _ => (0, 0.0),
        }),
    );
}

#[test]
fn the_viewed_matte_is_the_coverage_sum_of_the_picked_ids() {
    // The GPU id path runs only while the node is VIEWED, with the matte in
    // alpha; downstream reads the decode-side matte (tested below).
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    scene(&ctx, &mut comp);
    let third = (W / 3) as usize;
    let at = |r: &Vec<[f32; 4]>, t: usize| r[t * third + third / 2];

    let p = CryptomatteParams { picks: vec![pick("cube")], previewing: true, ..Default::default() };
    let r = row(&ctx, &mut comp, &graph(p));
    assert!((at(&r, 0)[3] - 1.0).abs() < 1e-3, "cube solid: {:?}", at(&r, 0));
    assert!((at(&r, 1)[3] - 0.6).abs() < 2e-3, "cube's share of the edge: {:?}", at(&r, 1));
    assert!(at(&r, 2)[3].abs() < 1e-3, "sphere is not picked: {:?}", at(&r, 2));

    // Both picked: the soft edge sums back to 1 across the two ranks.
    let p = CryptomatteParams { picks: vec![pick("cube"), pick("sphere")], previewing: true, ..Default::default() };
    let r = row(&ctx, &mut comp, &graph(p));
    for t in 0..3 {
        assert!((at(&r, t)[3] - 1.0).abs() < 2e-3, "third {t}: {:?}", at(&r, t));
    }

    // Inverted, sphere only.
    let p = CryptomatteParams { picks: vec![pick("sphere")], invert: true, previewing: true, ..Default::default() };
    let r = row(&ctx, &mut comp, &graph(p));
    assert!((at(&r, 0)[3] - 1.0).abs() < 1e-3);
    assert!((at(&r, 1)[3] - 0.6).abs() < 2e-3);
    assert!(at(&r, 2)[3].abs() < 1e-3);

    // A name that is in no rank selects nothing.
    let p = CryptomatteParams { picks: vec![pick("teapot")], previewing: true, ..Default::default() };
    let r = row(&ctx, &mut comp, &graph(p));
    assert!(r.iter().all(|c| c[3].abs() < 1e-3));
}

#[test]
fn ids_view_decodes_back_to_the_id_and_never_leaks_downstream() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    scene(&ctx, &mut comp);
    let third = (W / 3) as usize;

    // Viewed: the ids picture, exact through the f16 output.
    let p = CryptomatteParams { previewing: true, ..Default::default() };
    let r = row(&ctx, &mut comp, &graph(p));
    let c = r[third / 2];
    assert_eq!(cryptomatte::decode_scaled([c[0], c[1], c[2]]), cryptomatte::id_of_name("cube"));
    let c = r[2 * third + third / 2];
    assert_eq!(cryptomatte::decode_scaled([c[0], c[1], c[2]]), cryptomatte::id_of_name("sphere"));
    assert!(r.iter().all(|c| c[3].abs() < 1e-3), "alpha is the matte: nothing picked");

    // A picked object is LIT — bright yellow, red and green at or above 1 —
    // and still decodes to its id; alpha carries the matte.
    let p = CryptomatteParams {
        previewing: true,
        picks: vec![pick("cube")],
        ..Default::default()
    };
    let r = row(&ctx, &mut comp, &graph(p));
    let c = r[third / 2];
    assert!(c[0] >= 1.0 && c[1] >= 1.0 && c[2] < 0.25, "lit: {c:?}");
    assert_eq!(cryptomatte::decode_view([c[0], c[1], c[2]]), cryptomatte::id_of_name("cube"));
    assert!((c[3] - 1.0).abs() < 1e-3, "alpha is the matte");
    let c = r[2 * third + third / 2];
    assert!(c[0] < 1.0, "the sphere is not lit: {c:?}");
    assert_eq!(cryptomatte::decode_view([c[0], c[1], c[2]]), cryptomatte::id_of_name("sphere"));
    assert!(c[3].abs() < 1e-3);

    // Not the view target: the same params render the plain (empty) matte.
    let p = CryptomatteParams { previewing: false, ..Default::default() };
    let r = row(&ctx, &mut comp, &graph(p));
    assert!(r.iter().all(|c| c[0].abs() < 1e-3 && c[3].abs() < 1e-3));
}

/// Not viewed, the node is ONE texture sample: its matte reader, whatever
/// the picks say — they already went into that reader's decode.
#[test]
fn unviewed_the_node_samples_only_its_matte_reader() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    scene(&ctx, &mut comp);
    let half = f16::from_f32(0.5);
    upload(&ctx, &mut comp, 3, &vec![[half; 4]; (W * H) as usize]);
    let mut g = graph(CryptomatteParams { picks: vec![pick("teapot")], ..Default::default() });
    if let Some(NodeKind::Cryptomatte(c)) = g.nodes.get_mut(&4) {
        c.matte_source = 3;
    }
    let inst = g.reader_instances();
    assert_eq!(inst.len(), 1, "one reader in the rig: {inst:?}");
    assert_eq!(inst[0].sub, cryptomatte::MATTE_SUB);
    let r = row(&ctx, &mut comp, &g);
    assert!(r.iter().all(|c| c.iter().all(|v| (v - 0.5).abs() < 1e-3)), "{:?}", r[0]);
    // Viewed, it is the live set instead, and the matte reader is out.
    if let Some(NodeKind::Cryptomatte(c)) = g.nodes.get_mut(&4) {
        c.previewing = true;
    }
    let subs: Vec<usize> = g.reader_instances().iter().map(|i| i.sub).collect();
    assert!(!subs.contains(&cryptomatte::MATTE_SUB) && subs.len() == 3, "{subs:?}");
}

/// Write a cryptomatte EXR the way a renderer does — f32 id/coverage pairs
/// in `CryptoObject00..02` plus the manifest — and read one rank back.
#[test]
fn a_rank_layer_decodes_to_exact_chunks_and_the_manifest_reads() {
    use exr::prelude::*;
    let dir = std::env::temp_dir().join("wobble-cryptomatte-test");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("crypto.exr");
    let (w, h) = (16usize, 4usize);
    let (cube, sphere) = (cryptomatte::id_of_name("cube"), cryptomatte::id_of_name("sphere"));
    let mut ch: Vec<(String, Vec<f32>)> = Vec::new();
    for name in ["R", "G", "B", "A"] {
        ch.push((name.into(), vec![0.5; w * h]));
    }
    let id0: Vec<f32> = (0..w * h)
        .map(|i| f32::from_bits(if i % w < w / 2 { cube } else { sphere }))
        .collect();
    let cov0: Vec<f32> = (0..w * h).map(|i| if i % w < w / 2 { 0.75 } else { 1.0 }).collect();
    let id1: Vec<f32> =
        (0..w * h).map(|i| f32::from_bits(if i % w < w / 2 { sphere } else { 0 })).collect();
    let cov1: Vec<f32> = (0..w * h).map(|i| if i % w < w / 2 { 0.25 } else { 0.0 }).collect();
    ch.push(("CryptoObject00.R".into(), id0));
    ch.push(("CryptoObject00.G".into(), cov0));
    ch.push(("CryptoObject00.B".into(), id1));
    ch.push(("CryptoObject00.A".into(), cov1));
    for l in ["CryptoObject01", "CryptoObject02"] {
        for c in ["R", "G", "B", "A"] {
            ch.push((format!("{l}.{c}"), vec![0.0; w * h]));
        }
    }
    let mut attrs = LayerAttributes::named("rgba");
    let key = cryptomatte::layer_key("CryptoObject");
    for (k, v) in [
        ("name", "CryptoObject".to_string()),
        ("hash", "MurmurHash3_32".to_string()),
        ("conversion", "uint32_to_float32".to_string()),
        ("manifest", cryptomatte::manifest_json(&["cube", "sphere"])),
    ] {
        attrs.other.insert(
            Text::from(format!("cryptomatte/{key}/{k}").as_str()),
            AttributeValue::Text(Text::from(v.as_str())),
        );
    }
    let layer = Layer::new(
        (w, h),
        attrs,
        Encoding::FAST_LOSSLESS,
        AnyChannels::sort(
            ch.into_iter()
                .map(|(n, d)| AnyChannel::new(n.as_str(), FlatSamples::F32(d)))
                .collect(),
        ),
    );
    Image::from_layer(layer).write().non_parallel().to_file(&path).unwrap();

    let defs = cryptomatte::defs(&path).unwrap();
    assert_eq!(defs.len(), 1);
    assert_eq!(defs[0].name, "CryptoObject");
    assert_eq!(defs[0].name_of(sphere), Some("sphere"));

    let dec = ExrsDecoder::default();
    let info = dec.read_header(&path).unwrap();
    let stride = w * 4;
    let mut buf = vec![f16::ZERO; stride * h];
    dec.decode_layer_into(&path, info, stride, Some("CryptoObject00#0"), &mut buf).unwrap();
    let px = |buf: &[f16], x: usize| -> ([f32; 3], f32) {
        let i = 2 * stride + x * 4;
        ([buf[i].to_f32(), buf[i + 1].to_f32(), buf[i + 2].to_f32()], buf[i + 3].to_f32())
    };
    let (c, cov) = px(&buf, 3);
    assert_eq!(cryptomatte::decode_id(c), cube);
    assert_eq!(cov, 0.75);
    let (c, cov) = px(&buf, 12);
    assert_eq!(cryptomatte::decode_id(c), sphere);
    assert_eq!(cov, 1.0);

    dec.decode_layer_into(&path, info, stride, Some("CryptoObject00#1"), &mut buf).unwrap();
    let (c, cov) = px(&buf, 3);
    assert_eq!(cryptomatte::decode_id(c), sphere);
    assert_eq!(cov, 0.25);
    let (c, cov) = px(&buf, 12);
    assert_eq!(cryptomatte::decode_id(c), 0);
    assert_eq!(cov, 0.0);

    // The decode-side matte: the same file, asked for the finished matte.
    let matte = |picks: &[&str], invert: bool| {
        CryptomatteParams {
            layer: "CryptoObject".into(),
            picks: picks.iter().map(|n| pick(n)).collect(),
            invert,
            ..Default::default()
        }
        .matte_layer()
    };
    let at = |buf: &[f16], x: usize| buf[2 * stride + x * 4].to_f32();
    dec.decode_layer_into(&path, info, stride, Some(&matte(&["cube"], false)), &mut buf).unwrap();
    assert_eq!((at(&buf, 3), at(&buf, 12)), (0.75, 0.0), "cube alone");
    assert_eq!(buf[2 * stride + 3 * 4 + 3].to_f32(), 0.75, "in alpha too");
    dec.decode_layer_into(&path, info, stride, Some(&matte(&["sphere", "cube"], false)), &mut buf)
        .unwrap();
    assert_eq!((at(&buf, 3), at(&buf, 12)), (1.0, 1.0), "the soft edge sums back to 1");
    dec.decode_layer_into(&path, info, stride, Some(&matte(&["cube"], true)), &mut buf).unwrap();
    assert_eq!((at(&buf, 3), at(&buf, 12)), (0.25, 1.0), "inverted");
    dec.decode_layer_into(&path, info, stride, Some(&matte(&[], false)), &mut buf).unwrap();
    assert!(buf.iter().all(|v| v.to_f32() == 0.0), "nothing picked");

    // The plain layer still reads as before (raw floats, rounded to half).
    dec.decode_layer_into(&path, info, stride, Some("CryptoObject00"), &mut buf).unwrap();
    assert_eq!(buf[2 * stride + 3 * 4 + 1].to_f32(), 0.75);
}
