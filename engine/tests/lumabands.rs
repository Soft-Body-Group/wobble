//! A multi-band LumaKey end to end: lower it, compile it, render it, and
//! check each band really is its own matte.
//!
//! The unit tests in `engine::lumabands` prove the LOWERING — that the right
//! nodes appear, wired to the right things. They say nothing about whether
//! the result compiles or what comes out of it, which is the part that
//! matters: three bands that all render band 0 would pass every one of them.
//!
//! Self-skips without a GPU (CI).

use engine::color::Colorspace;
use engine::compositor::Compositor;
use engine::decode::ImageInfo;
use engine::graph::{Graph, LumaBand, LumaKeyParams, NodeKind};
use engine::timing::GpuTimer;
use half::f16;

const W: u32 = 16;
const H: u32 = 8;

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP lumabands test: no adapter ({e})");
            None
        }
    }
}

/// A flat NEUTRAL plate, so its luminance is `grey` in any working space
/// (the weights sum to 1) and the expected coverage is arithmetic, not a
/// colour-science argument.
fn upload(ctx: &engine::GpuContext, comp: &mut Compositor, grey: f32) {
    comp.ensure_reader(0, ImageInfo { width: W, height: H });
    let texels: Vec<[f16; 4]> = (0..W * H)
        .map(|_| {
            let g = f16::from_f32(grey);
            [g, g, g, f16::from_f32(1.0)]
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

fn band(lo: f32, hi: f32) -> LumaBand {
    LumaBand { lo, hi, ..Default::default() }
}

/// Three bands tiling 0..1, wired to nothing: the view target picks which
/// one is rendered, which is how a band with no consumer is inspected.
fn keyed(view_out: u32) -> Graph {
    let mut g = Graph::default();
    g.working = Colorspace::ACESCG;
    g.nodes.insert(0, NodeKind::read(0, ""));
    g.nodes.insert(
        1,
        NodeKind::LumaKey(LumaKeyParams {
            bands: vec![band(0.0, 0.3), band(0.3, 0.6), band(0.6, 1.0)], ranges: None }),
    );
    g.edges.insert((1, 0), 0);
    g.view = Some(1);
    g.view_out = view_out;
    g
}

/// f16 output: relative, with an absolute floor for coverage values near
/// zero. A flat 3e-3 is one ulp at 4.5 and this test grades past that.
fn close(got: f32, want: f32, what: &str) {
    let tol = 3e-3 * want.abs().max(1.0);
    assert!((got - want).abs() < tol, "{what}: got {got}, want {want}");
}

/// The headline: the same node, three outputs, three different mattes. On a
/// 0.45 grey the low band is fully past its top, the middle is halfway up
/// its ramp, and the high band has not started.
#[test]
fn each_band_renders_its_own_matte() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp, 0.45);

    for (out, want) in [(0u32, 1.0f32), (1, 0.5), (2, 0.0)] {
        let mut g = keyed(out);
        engine::lumabands::expand(&mut g);
        let got = render(&ctx, &mut comp, &g);
        close(got[0], want, &format!("band {out} rgb"));
        // Coverage lands in ALPHA too — the convention Spline and Keyer
        // already follow, and what the viewer's alpha toggle reads.
        close(got[3], want, &format!("band {out} alpha"));
    }
}

/// ...and it is really the BAND that changed, not the plate: sweep the
/// plate instead and every band moves the way its own ends say it should.
#[test]
fn every_band_responds_on_its_own_range() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    // (grey, [band 0, band 1, band 2] expected coverage)
    let cases: [(f32, [f32; 3]); 3] = [
        (0.15, [0.5, 0.0, 0.0]),
        (0.45, [1.0, 0.5, 0.0]),
        (0.80, [1.0, 1.0, 0.5]),
    ];
    for (grey, want) in cases {
        upload(&ctx, &mut comp, grey);
        for (out, w) in want.iter().enumerate() {
            let mut g = keyed(out as u32);
            engine::lumabands::expand(&mut g);
            let got = render(&ctx, &mut comp, &g);
            close(got[0], *w, &format!("grey {grey}, band {out}"));
        }
    }
}

/// Two bands driving two different masks IN ONE COMP — the case the whole
/// feature exists for, and the one where a lowering bug (both consumers
/// landing on the same expanded node) would show up.
#[test]
fn two_bands_mask_two_grades_in_one_comp() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp, 0.45);

    // read -> key(3). Grade A is masked by band 0 (coverage 1 here, so its
    // gain lands in full); Grade B, downstream, is masked by band 2
    // (coverage 0, so its gain must not land at all).
    let mut g = keyed(0);
    let gain = |v: f32| engine::graph::GradeParams {
        gain: [v; 3],
        ..Default::default()
    };
    g.nodes.insert(10, NodeKind::Grade(gain(2.0)));
    g.edges.insert((10, 0), 0);
    g.connect_out(10, 1, 1, 0); // mask <- band 0
    g.nodes.insert(11, NodeKind::Grade(gain(5.0)));
    g.edges.insert((11, 0), 10);
    g.connect_out(11, 1, 1, 2); // mask <- band 2
    g.view = Some(11);
    g.view_out = 0;
    engine::lumabands::expand(&mut g);

    let got = render(&ctx, &mut comp, &g);
    // 0.45 * 2.0 from the first grade; the second is fully masked off.
    close(got[0], 0.9, "two bands, two masks");

    // Point the second grade at band 0 as well and it DOES land — proof the
    // number above came from the band and not from a broken mask edge.
    let mut g2 = g.clone();
    // (post-expansion, band 0 is still node 1)
    g2.connect_out(11, 1, 1, 0);
    let got = render(&ctx, &mut comp, &g2);
    close(got[0], 4.5, "both on band 0");
}

/// A band nothing reads must not reach the shader at all. Three bands with
/// one consumer compile to the same thing as a one-band key.
#[test]
fn an_unconsumed_band_is_not_compiled() {
    let mut three = keyed(0);
    engine::lumabands::expand(&mut three);
    let mut one = Graph::default();
    one.working = Colorspace::ACESCG;
    one.nodes.insert(0, NodeKind::read(0, ""));
    one.nodes.insert(1, NodeKind::LumaKey(LumaKeyParams { bands: vec![band(0.0, 0.3)], ranges: None }));
    one.edges.insert((1, 0), 0);
    one.view = Some(1);
    assert_eq!(
        engine::codegen::generate(&three).main.wgsl,
        engine::codegen::generate(&one).main.wgsl,
        "an unwired band changed the shader"
    );
}

/// Output 0's coverage at luma `x`: band 0's curve, linear between its
/// points and flat past its ends, then its `invert`. Judges a key by what it
/// DOES, which is what a migration has to keep — not by how the partition
/// happens to spell it.
#[cfg(feature = "serde")]
fn coverage(k: &LumaKeyParams, x: f32) -> f32 {
    let b = k.band(0);
    let c = b.curve();
    let (first, last) = (c[0], c[c.len() - 1]);
    let y = if x <= first[0] {
        first[1]
    } else if x >= last[0] {
        last[1]
    } else {
        let i = c.windows(2).position(|w| x >= w[0][0] && x <= w[1][0]).unwrap();
        let (a, z) = (c[i], c[i + 1]);
        a[1] + (z[1] - a[1]) * (x - a[0]) / (z[0] - a[0]).max(1e-6)
    };
    if b.invert {
        1.0 - y
    } else {
        y
    }
}

/// Setups written before bands existed carried the single band's fields
/// FLAT on the node, and a fresh key ramps from `lo` to `hi`. Both broke
/// under the partition/`BandRanges` work (see `.agents/messages.md`,
/// 2026-09-04); this guards the fix. The one loss the partition genuinely
/// cannot avoid has its own test below, ignored with the reason.
#[cfg(feature = "serde")]
#[test]
fn legacy_flat_setups_still_load() {
    // 1. A fresh LumaKey must key SOMETHING. It was a one-band partition —
    //    no boundary, so flat at full coverage: a no-op, and anything masked
    //    by it applied everywhere.
    let d = LumaKeyParams::default();
    let c = d.curve();
    assert!(
        c.first().map(|p| p[1]) != c.last().map(|p| p[1]),
        "a default LumaKey is a FLAT matte ({c:?}) — it keys nothing, so a \
         Grade masked by a fresh key applies to the whole frame"
    );
    // ...and it is the same key an empty setup loads as, not a second idea
    // of "fresh".
    let empty: LumaKeyParams = serde_json::from_str("{}").unwrap();
    assert_eq!(d, empty, "a fresh key and an empty setup disagree");

    // 2. The ends the file asked for are the ends it gets: coverage is flat
    //    outside 0.1..0.9 and moves only inside it. (The partition spells
    //    band 0 from the axis start, so its first POINT is at 0.0 — that is
    //    representation, not response, and output 0 is unchanged.)
    let p: LumaKeyParams =
        serde_json::from_str(r#"{"lo": 0.1, "hi": 0.9, "invert": true}"#).unwrap();
    let at = |x: f32| coverage(&p, x);
    for x in [0.0, 0.05, 0.1] {
        assert!((at(x) - 1.0).abs() < 0.03, "below lo=0.1, x={x}: {} (want 1)", at(x));
    }
    for x in [0.9, 1.2, 2.0] {
        assert!(at(x).abs() < 0.03, "above hi=0.9, x={x}: {} (want 0)", at(x));
    }
    assert!(
        at(0.5) > 0.05 && at(0.5) < 0.95,
        "the ramp must run between the ends, x=0.5: {}",
        at(0.5)
    );

    // 3. The new shape round-trips (this half still holds).
    let p = LumaKeyParams { bands: vec![band(0.0, 0.3), band(0.3, 1.0)], ..Default::default() };
    let s = serde_json::to_string(&p).unwrap();
    let back: LumaKeyParams = serde_json::from_str(&s).unwrap();
    assert_eq!(back.bands.len(), p.bands.len());
    for (a, b) in back.bands.iter().zip(p.bands.iter()) {
        assert!((a.lo - b.lo).abs() < 1e-6 && (a.hi - b.hi).abs() < 1e-6);
    }
}

/// An authored shape should survive the load: the interior point is the
/// user's work. It CANNOT in the locked-band model — the point sits on the
/// ramp, and in a partition the ramp is the crossfade two bands share, which
/// has no room for one. The load warns instead of dropping it silently.
/// Kept, not deleted, so the limitation shows in every test run; the fix is
/// a model change (per-edge shape), not a test change.
#[cfg(feature = "serde")]
#[test]
#[ignore = "known limitation: a pre-partition key's hand-shaped ramp loses its points on \
            migration (logged as a warning) — the partition has no place for a point on a \
            crossfade"]
fn legacy_shaped_key_keeps_its_points() {
    let p: LumaKeyParams =
        serde_json::from_str(r#"{"lo": 0.0, "hi": 1.0, "points": [[0.5, 0.25]]}"#).unwrap();
    let pts = p.curve();
    assert!(
        pts.iter().any(|q| (q[0] - 0.5).abs() < 1e-3 && (q[1] - 0.25).abs() < 1e-3),
        "the authored shaping point (0.5, 0.25) was dropped on load: {pts:?}"
    );
}
