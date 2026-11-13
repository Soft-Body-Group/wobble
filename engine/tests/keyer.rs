//! Headless integration for the colour keyer: the WGSL in `engine::keyer`
//! must agree with the Rust mirror the UI's plots are drawn from, pixel for
//! pixel. The plots would otherwise shade one selection while the shader
//! makes another, which is the one bug this node cannot afford.
//!
//! Self-skips without a GPU (CI).

use engine::compositor::Compositor;
use engine::decode::ImageInfo;
use engine::graph::{Graph, NodeKind};
use engine::keyer::{KeyBlob, KeyerParams, KeyerView};
use engine::timing::GpuTimer;
use half::f16;

const W: u32 = 64;
const H: u32 = 32;

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP keyer test: no adapter ({e})");
            None
        }
    }
}

/// A plate that sweeps hue across x and exposure down y, so the test covers
/// saturated, near-neutral and above-white pixels in one pass.
fn plate() -> (Vec<[f16; 4]>, Vec<[f32; 3]>) {
    let mut px = Vec::with_capacity((W * H) as usize);
    let mut lin = Vec::with_capacity((W * H) as usize);
    for y in 0..H {
        // 0.02 .. ~3.0: below, around and well above diffuse white.
        let e = 0.02 * (1.0f32 + y as f32 / H as f32 * 5.0).powi(3);
        for x in 0..W {
            let t = x as f32 / (W - 1) as f32 * std::f32::consts::TAU;
            let c = [
                e * (0.5 + 0.5 * t.cos()),
                e * (0.5 + 0.5 * (t + 2.094).cos()),
                e * (0.5 + 0.5 * (t + 4.189).cos()),
            ];
            lin.push(c);
            px.push([f16::from_f32(c[0]), f16::from_f32(c[1]), f16::from_f32(c[2]), f16::ONE]);
        }
    }
    (px, lin)
}

fn upload(ctx: &engine::GpuContext, comp: &mut Compositor, slot: usize, px: &[[f16; 4]]) {
    comp.ensure_reader(slot, ImageInfo { width: W, height: H });
    let tex = &comp.reader_texture(slot).unwrap().texture;
    ctx.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: tex,
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

fn render(ctx: &engine::GpuContext, comp: &mut Compositor, graph: &Graph) -> Vec<[f32; 4]> {
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
    let (tx, rx) = std::sync::mpsc::channel();
    staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    rx.recv().unwrap().unwrap();
    let view = staging.slice(..).get_mapped_range().expect("a mapped buffer");
    let mut out = Vec::with_capacity((W * H) as usize);
    for y in 0..H as usize {
        let row: &[f16] =
            bytemuck::cast_slice(&view[y * padded as usize..y * padded as usize + (W * 8) as usize]);
        for c in row.chunks_exact(4) {
            out.push([c[0].to_f32(), c[1].to_f32(), c[2].to_f32(), c[3].to_f32()]);
        }
    }
    out
}

/// Read A -> Keyer (node 7), which is the view target.
fn keyer_graph(p: KeyerParams) -> Graph {
    let mut g = Graph::default_poc();
    g.nodes.insert(7, NodeKind::Keyer(p));
    g.edges.insert((7, 0), 1);
    g.view = Some(7);
    g
}

/// The same parity check, but with the comp working in ACEScg.
///
/// Oklab is defined from Rec.709 primaries, so the shader and the CPU mirror
/// each have to reach it through the working gamut — and they have to do it
/// the SAME way. If they drift, a selection lands somewhere other than where
/// the artist sampled, and nothing on screen says so.
#[test]
fn shader_matte_matches_the_cpu_mirror_in_acescg() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 2);
    comp.ensure_output(W, H);
    let (px, lin) = plate();
    upload(&ctx, &mut comp, 0, &px);
    let p = interesting_params();
    let mut g = keyer_graph(p.clone());
    g.working = engine::color::Colorspace::ACESCG;
    let out = render(&ctx, &mut comp, &g);

    let m = engine::keyer::oklab_matrix(g.working);
    let mut worst = 0.0f32;
    let mut nonzero = 0;
    let mut differs = 0;
    for (o, c) in out.iter().zip(&lin) {
        let want = engine::keyer::matte_in(&p, *c, m);
        worst = worst.max((o[3] - want).abs());
        if want > 0.02 {
            nonzero += 1;
        }
        // ...and it must NOT match the Rec.709 mirror everywhere, or the
        // working space is being ignored and this test proves nothing.
        if (engine::keyer::matte(&p, *c) - want).abs() > 0.02 {
            differs += 1;
        }
    }
    assert!(worst < 4e-3, "ACEScg shader disagrees with its mirror by {worst}");
    // Lower than the Rec.709 case on purpose: the blob centres in
    // `interesting_params` are Rec.709 Oklab constants, so read on the AP1
    // frame they sit somewhere slightly different and select a smaller set.
    // That they select a DIFFERENT set is the whole point (see `differs`).
    assert!(nonzero > 100, "plate barely triggers the key ({nonzero} px)");
    assert!(
        differs > 20,
        "ACEScg and Rec.709 mattes are identical on {} px — the working \
         space is not reaching the keyer",
        out.len()
    );
}

fn interesting_params() -> KeyerParams {
    KeyerParams {
        blobs: vec![
            // A warm, mid-lightness grab...
            KeyBlob {
                center: engine::keyer::oklab_from_linear([0.4, 0.22, 0.15]),
                tol: [0.22, 0.07, 0.05],
                soft: 1.3,
                weight: 1.0,
            },
            // ...a second, brighter one that overlaps it (exercises the
            // product union)...
            KeyBlob {
                center: engine::keyer::oklab_from_linear([0.9, 0.55, 0.3]),
                tol: [0.15, 0.06, 0.04],
                soft: 0.7,
                weight: 0.8,
            },
            // ...and one that carves a bite out of both.
            KeyBlob {
                center: engine::keyer::oklab_from_linear([0.5, 0.30, 0.10]),
                tol: [0.10, 0.03, 0.02],
                soft: 0.5,
                weight: -0.9,
            },
        ],
        tilt: 0.35,
        lo: 0.08,
        hi: 0.92,
        gamma: 1.6,
        invert: false,
        view: KeyerView::Source,
        previewing: false,
    }
}

#[test]
fn shader_matte_matches_the_cpu_mirror() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 2);
    comp.ensure_output(W, H);
    let (px, lin) = plate();
    upload(&ctx, &mut comp, 0, &px);
    let p = interesting_params();
    let out = render(&ctx, &mut comp, &keyer_graph(p.clone()));

    let mut worst = 0.0f32;
    let mut nonzero = 0;
    let mut solid = 0;
    for (o, c) in out.iter().zip(&lin) {
        let want = engine::keyer::matte(&p, *c);
        worst = worst.max((o[3] - want).abs());
        // Matte view is off, so rgb is the matte too.
        assert!((o[0] - o[3]).abs() < 2e-3, "rgb must carry the matte: {o:?}");
        if want > 0.02 {
            nonzero += 1;
        }
        if want > 0.98 {
            solid += 1;
        }
    }
    // f16 storage on both ends, so a few thousandths is the floor.
    assert!(worst < 4e-3, "shader disagrees with the mirror by {worst}");
    // Guard against a vacuous pass: the plate has to actually exercise the
    // soft edge AND the solid core.
    assert!(nonzero > 200, "plate barely triggers the key ({nonzero} px)");
    assert!(solid > 20, "no solid core in the plate ({solid} px)");
    assert!(nonzero < (W * H) as usize - 200, "key selects nearly everything");
}

#[test]
fn preview_shows_the_source_only_when_the_node_is_viewed() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 2);
    comp.ensure_output(W, H);
    let (px, lin) = plate();
    upload(&ctx, &mut comp, 0, &px);

    let mut p = interesting_params();
    p.previewing = true; // what `to_engine_graph_at` does for the view node
    let out = render(&ctx, &mut comp, &keyer_graph(p.clone()));
    for (o, c) in out.iter().zip(&lin) {
        let want = engine::keyer::matte(&p, *c);
        assert!((o[3] - want).abs() < 4e-3, "alpha must stay the matte");
        for k in 0..3 {
            assert!(
                (o[k] - c[k]).abs() < 5e-3 * c[k].max(1.0),
                "preview must show the untouched source: {o:?} vs {c:?}"
            );
        }
    }

    // Matte view mode, still the view target: back to the matte in rgb.
    p.view = KeyerView::Matte;
    let out = render(&ctx, &mut comp, &keyer_graph(p.clone()));
    for (o, c) in out.iter().zip(&lin) {
        let want = engine::keyer::matte(&p, *c);
        assert!((o[0] - want).abs() < 4e-3, "matte view must show the matte");
    }
}

#[test]
fn garbage_matte_multiplies_into_the_key() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 2);
    comp.ensure_output(W, H);
    let (px, lin) = plate();
    upload(&ctx, &mut comp, 0, &px);
    // Reader B is a constant 0.25 grey; a LumaKey over it makes a flat 0.5
    // coverage, which is a garbage matte with an unmistakable signature.
    let grey = vec![[f16::from_f32(0.25); 4]; (W * H) as usize];
    upload(&ctx, &mut comp, 1, &grey);

    let p = interesting_params();
    let mut g = keyer_graph(p.clone());
    g.nodes.insert(
        8,
        NodeKind::LumaKey(engine::graph::LumaKeyParams {
            bands: vec![engine::graph::LumaBand {
                lo: 0.0,
                hi: 0.5,
                invert: false,
                ..Default::default()
            }],
            ranges: None,
        }),
    );
    g.edges.insert((8, 0), 2); // lumakey <- read B (flat 0.25 -> mask 0.5)
    g.edges.insert((7, 1), 8); // keyer garbage matte <- lumakey
    let out = render(&ctx, &mut comp, &g);
    for (o, c) in out.iter().zip(&lin) {
        let want = engine::keyer::matte(&p, *c) * 0.5;
        assert!((o[3] - want).abs() < 5e-3, "garbage matte must scale the key");
    }
}
