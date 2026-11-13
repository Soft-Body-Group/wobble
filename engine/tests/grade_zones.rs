//! Headless integration: a Grade with zones IS a LumaKey feeding masked
//! Grades in series.
//!
//! That equivalence is the whole design — the zones' partition is the key's
//! model, the zone weights are the key's mattes, and each zone's correction
//! is an ordinary masked grade4 over the running result — so it is pinned
//! pixel for pixel against the graph you would build by hand. Self-skips
//! without a GPU (CI).

use engine::compositor::Compositor;
use engine::decode::ImageInfo;
use engine::graph::{Graph, GradeParams, GradeZone, GradeZones, NodeKind};
use engine::timing::GpuTimer;
use half::f16;

/// A luma ramp: pixel x carries x / (W-1) * 2, black to twice white.
const W: u32 = 64;
const H: u32 = 4;

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP grade-zones test: no adapter ({e})");
            None
        }
    }
}

fn upload_ramp(ctx: &engine::GpuContext, comp: &mut Compositor) {
    let info = ImageInfo { width: W, height: H };
    comp.ensure_reader(0, info);
    let px: Vec<[f16; 4]> = (0..W * H)
        .map(|i| {
            let v = (i % W) as f32 / (W - 1) as f32 * 2.0;
            // A little chroma so the saturation term has something to do.
            [f16::from_f32(v * 1.1), f16::from_f32(v), f16::from_f32(v * 0.9), f16::ONE]
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
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(W * 8),
            rows_per_image: Some(H),
        },
        wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
    );
}

/// Render `graph` and hand back the first row.
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
    let halves: &[f16] = bytemuck::cast_slice(&data[..(W * 8) as usize]);
    let out: Vec<[f32; 4]> = (0..W as usize)
        .map(|x| {
            [
                halves[x * 4].to_f32(),
                halves[x * 4 + 1].to_f32(),
                halves[x * 4 + 2].to_f32(),
                halves[x * 4 + 3].to_f32(),
            ]
        })
        .collect();
    drop(data);
    staging.unmap();
    out
}

fn master() -> GradeParams {
    GradeParams {
        gain: [1.2, 1.0, 0.9],
        lift: [0.02, 0.0, -0.01],
        saturation: 1.3,
        ..Default::default()
    }
}

/// Three zones that each do something visibly different.
fn zones() -> GradeZones {
    let mut z = GradeZones::default();
    z.adjust[0] = GradeZone { lift: [0.05, 0.02, 0.0], gain: [0.8; 3], ..Default::default() };
    z.adjust[1] = GradeZone {
        gamma: [0.8, 1.0, 1.2],
        saturation: 0.5,
        contrast: 1.3,
        pivot: 0.25,
        ..Default::default()
    };
    z.adjust[2] = GradeZone { gain: [2.0, 1.8, 1.6], offset: [0.1; 3], ..Default::default() };
    // A bias point in the midtones' plateau, so the shape-of-the-band path
    // is exercised too, not just the corners.
    z.key.bands[1].points.push([0.3, 0.6]);
    z
}

/// read -> Grade(master + zones) -> view.
fn fused(z: Option<GradeZones>) -> Graph {
    let mut g = Graph::default();
    g.nodes.insert(0, NodeKind::read(0, ""));
    g.nodes.insert(1, NodeKind::Grade(GradeParams { zones: z, ..master() }));
    g.edges.insert((1, 0), 0);
    g.view = Some(1);
    g
}

/// The same thing by hand: read -> LumaKey(bands) and read -> Grade(master)
/// -> Grade(zone 0, masked by band 0) -> Grade(zone 1, masked by band 1) ->
/// Grade(zone 2, masked by band 2) -> view. Lowered exactly as the app lowers
/// a multi-band key.
fn by_hand(z: &GradeZones) -> Graph {
    let mut g = Graph::default();
    g.nodes.insert(0, NodeKind::read(0, ""));
    g.nodes.insert(1, NodeKind::LumaKey(z.key.clone()));
    g.edges.insert((1, 0), 0);
    g.nodes.insert(2, NodeKind::Grade(master()));
    g.edges.insert((2, 0), 0);
    let mut prev = 2;
    for (i, a) in z.adjust.iter().enumerate() {
        let id = 10 + i as u64;
        g.nodes.insert(
            id,
            NodeKind::Grade(GradeParams {
                lift: a.lift,
                gamma: a.gamma,
                gain: a.gain,
                offset: a.offset,
                saturation: a.saturation,
                contrast: a.contrast,
                pivot: a.pivot,
                ..Default::default()
            }),
        );
        g.edges.insert((id, 0), prev);
        g.connect_out(id, 1, 1, i as u32);
        prev = id;
    }
    g.view = Some(prev);
    engine::lumabands::expand(&mut g);
    g
}

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() <= 0.01 + 0.01 * b.abs()
}

#[test]
fn a_zoned_grade_matches_a_key_feeding_masked_grades_in_series() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload_ramp(&ctx, &mut comp);
    let z = zones();
    let a = row(&ctx, &mut comp, &fused(Some(z.clone())));
    let b = row(&ctx, &mut comp, &by_hand(&z));
    for x in 0..W as usize {
        for c in 0..3 {
            assert!(
                close(a[x][c], b[x][c]),
                "x={x} ch{c}: fused {} vs by hand {} (row {:?} vs {:?})",
                a[x][c],
                b[x][c],
                a[x],
                b[x]
            );
        }
    }
    // And the zones actually did something: the two ends of the ramp are
    // not what the master alone makes of them.
    let plain = row(&ctx, &mut comp, &fused(None));
    assert!(!close(a[0][0], plain[0][0]), "shadows zone had no effect");
    assert!(!close(a[W as usize - 1][0], plain[W as usize - 1][0]), "highlights zone had no effect");
}

/// Identity zones are a no-op: the picture is the master's alone, so
/// adding zones to a comp changes nothing until a zone is actually turned.
#[test]
fn identity_zones_change_nothing() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload_ramp(&ctx, &mut comp);
    let a = row(&ctx, &mut comp, &fused(Some(GradeZones::default())));
    let b = row(&ctx, &mut comp, &fused(None));
    for x in 0..W as usize {
        for c in 0..4 {
            assert!(close(a[x][c], b[x][c]), "x={x} ch{c}: {} vs {}", a[x][c], b[x][c]);
        }
    }
}

/// A zone acts in its band and nowhere else: doubling the highlights
/// doubles the top of the ramp and leaves black and the midtones alone.
#[test]
fn a_zone_acts_only_in_its_band() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload_ramp(&ctx, &mut comp);
    let mut z = GradeZones::default();
    z.adjust[2].gain = [2.0; 3];
    let a = row(&ctx, &mut comp, &fused(Some(z)));
    let b = row(&ctx, &mut comp, &fused(None));
    let last = W as usize - 1;
    assert!(close(a[last][1], 2.0 * b[last][1]), "top of ramp: {} vs 2x{}", a[last][1], b[last][1]);
    // x = 8 is luma 0.25: pure midtones under the defaults.
    assert!(close(a[8][1], b[8][1]), "midtones moved: {} vs {}", a[8][1], b[8][1]);
    assert!(close(a[0][1], b[0][1]), "black moved: {} vs {}", a[0][1], b[0][1]);
    // Inside the crossfade the effect is partial: strictly between.
    let mid = (0.75 / 2.0 * last as f32).round() as usize;
    assert!(a[mid][1] > b[mid][1] * 1.2 && a[mid][1] < b[mid][1] * 1.8, "crossfade: {} vs {}", a[mid][1], b[mid][1]);
}

/// The ranges preview, on real pixels. The zone under the hand is shown
/// either as its own coverage (matte) or by holding its pixels and sinking
/// everything else toward grey (highlight) — and neither touches the picture
/// anywhere the zone does not reach.
#[test]
fn the_ranges_preview_shows_the_zone_under_the_hand() {
    use engine::graph::{ZonePreview, ZoneView};
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload_ramp(&ctx, &mut comp);

    // The ramp carries luma ~= x/(W-1)*2, and the default ranges put the
    // midtones' plateau at 0.09..0.5: x=0 is all shadows, x=9 all midtones,
    // x=40 all highlights. `sync_from_ranges` drops the fixture's bias
    // point, which sits at 0.3 and would dip the midtones' coverage right
    // where this reads it — the partition's own shape is what a preview is
    // being asked about.
    let (dark, mid, hot) = (0usize, 9usize, 40usize);
    let clean = || {
        let mut z = zones();
        z.key.sync_from_ranges();
        z
    };
    let graded = row(&ctx, &mut comp, &fused(Some(clean())));

    let armed = |view: ZoneView, zone: u32| {
        let mut z = clean();
        z.preview = ZonePreview { view, zone };
        // The app arms this for the node it is VIEWING; in a headless test
        // we stand in for it.
        z.previewing = true;
        fused(Some(z))
    };

    // MATTE: the zone's coverage, in rgb and in alpha — the convention every
    // matte in this app follows.
    let m = row(&ctx, &mut comp, &armed(ZoneView::Matte, 1));
    assert!(close(m[mid][0], 1.0), "midtones not covered: {:?}", m[mid]);
    assert!(m[dark][0] < 0.02, "shadows leaked into the midtones' matte: {:?}", m[dark]);
    assert!(m[hot][0] < 0.02, "highlights leaked in: {:?}", m[hot]);
    for x in [dark, mid, hot] {
        assert!(close(m[x][3], m[x][0]), "matte not in alpha at {x}: {:?}", m[x]);
        assert!(close(m[x][1], m[x][0]) && close(m[x][2], m[x][0]), "not grey: {:?}", m[x]);
    }
    // A different chip, a different matte — the preview follows the hand.
    let m0 = row(&ctx, &mut comp, &armed(ZoneView::Matte, 0));
    assert!(close(m0[dark][0], 1.0) && m0[mid][0] < 0.02, "zone 0: {:?} {:?}", m0[dark], m0[mid]);

    // HIGHLIGHT: the graded picture inside the zone, sunk toward grey
    // outside it — so the coverage can be read against the shot.
    let h = row(&ctx, &mut comp, &armed(ZoneView::Highlight, 1));
    for c in 0..3 {
        assert!(close(h[mid][c], graded[mid][c]), "the zone's own pixels moved: {:?}", h[mid]);
    }
    assert!(h[hot][1] < graded[hot][1] * 0.5, "outside the zone was not dimmed: {:?}", h[hot]);
    assert!(
        close(h[hot][0], h[hot][1]) && close(h[hot][1], h[hot][2]),
        "outside the zone should go grey: {:?}",
        h[hot]
    );
    assert!(close(h[hot][3], graded[hot][3]), "alpha is not the preview's to change");

    // OFF is the picture, exactly.
    let off = row(&ctx, &mut comp, &armed(ZoneView::Off, 1));
    for x in [dark, mid, hot] {
        for c in 0..4 {
            assert!(close(off[x][c], graded[x][c]), "off changed the picture at {x}");
        }
    }
    // And an armed preview on a node the app has NOT marked as viewed is
    // off too — that is what keeps it out of renders.
    let mut z = clean();
    z.preview = ZonePreview { view: ZoneView::Matte, zone: 1 };
    let unarmed = row(&ctx, &mut comp, &fused(Some(z)));
    for c in 0..3 {
        assert!(close(unarmed[mid][c], graded[mid][c]), "an unviewed node previewed");
    }
}

/// ZONES: every zone at once, each in its own colour, mixed by the zones'
/// own weights. Pinned against the MATTE view of each zone read off the
/// same GPU — the map must be Σ matte_i × colour_i, pixel for pixel — and
/// the ramp is checked to contain each kind of pixel that claim covers: a
/// plateau (one zone's flat colour), a crossfade (two blended, adding to
/// one) and the fixture's bias point (a plateau dipped to 0.6, which must
/// DARKEN, because the weights are summed and never normalised).
#[test]
fn the_zones_preview_is_every_matte_in_its_own_colour() {
    use engine::graph::{band_linear, ZonePreview, ZoneView};
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload_ramp(&ctx, &mut comp);

    let armed = |view: ZoneView, zone: u32| {
        let mut z = zones();
        z.preview = ZonePreview { view, zone };
        z.previewing = true;
        fused(Some(z))
    };
    let n = zones().count();
    let working = fused(None).working;
    let cols: Vec<[f32; 3]> = (0..n).map(|i| band_linear(i, n, working)).collect();
    let mattes: Vec<Vec<[f32; 4]>> =
        (0..n).map(|i| row(&ctx, &mut comp, &armed(ZoneView::Matte, i as u32))).collect();
    let graded = row(&ctx, &mut comp, &fused(Some(zones())));
    let map = row(&ctx, &mut comp, &armed(ZoneView::Zones, 0));

    let (mut flat, mut blend, mut dipped) = (0, 0, 0);
    for x in 0..W as usize {
        let w: Vec<f32> = (0..n).map(|i| mattes[i][x][0]).collect();
        for c in 0..3 {
            let want: f32 = (0..n).map(|i| w[i] * cols[i][c]).sum();
            assert!(
                close(map[x][c], want),
                "x={x} ch{c}: the map shows {} where the zones' weights {w:?} make {want}",
                map[x][c]
            );
        }
        assert!(close(map[x][3], graded[x][3]), "alpha is not the preview's to change");
        let (sum, top) = (w.iter().sum::<f32>(), w.iter().cloned().fold(0.0, f32::max));
        if top > 0.99 {
            flat += 1;
        } else if (sum - 1.0).abs() < 0.02 && top < 0.9 {
            blend += 1;
        } else if sum < 0.9 {
            dipped += 1;
        }
    }
    assert!(flat > 3 && blend > 2 && dipped > 0, "flat {flat}, blend {blend}, dipped {dipped}");
    // Told apart: no two zones anywhere near the same colour.
    for a in 0..n {
        for b in a + 1..n {
            let d: f32 = (0..3).map(|c| (cols[a][c] - cols[b][c]).abs()).sum();
            assert!(d > 0.2, "zones {a} and {b} are too alike: {:?} {:?}", cols[a], cols[b]);
        }
    }
    // It is all of them: which zone is selected does not matter.
    let map2 = row(&ctx, &mut comp, &armed(ZoneView::Zones, 2));
    for x in 0..W as usize {
        for c in 0..4 {
            assert!(close(map2[x][c], map[x][c]), "the selected zone changed the map at {x}");
        }
    }
    // And unarmed — not the view target — it is the graded picture.
    let mut z = zones();
    z.preview = ZonePreview { view: ZoneView::Zones, zone: 0 };
    let unarmed = row(&ctx, &mut comp, &fused(Some(z)));
    for x in 0..W as usize {
        for c in 0..4 {
            assert!(close(unarmed[x][c], graded[x][c]), "an unviewed node previewed at {x}");
        }
    }
}

/// HARD: the zones map with its crossfades cut. Each pixel is the flat colour
/// of the zone whose matte is largest there — pinned against the mattes read
/// off the same GPU — so the ramp crosses each boundary in ONE step (three
/// zones, two steps, however soft the crossfades are), a dipped plateau stays
/// its zone's colour at full strength, and nothing else changes: alpha, the
/// unarmed picture.
#[test]
fn the_hard_map_is_the_heaviest_zone_flat() {
    use engine::graph::{band_linear, ZonePreview, ZoneView};
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload_ramp(&ctx, &mut comp);

    let armed = |view: ZoneView, zone: u32| {
        let mut z = zones();
        z.preview = ZonePreview { view, zone };
        z.previewing = true;
        fused(Some(z))
    };
    let n = zones().count();
    let working = fused(None).working;
    let cols: Vec<[f32; 3]> = (0..n).map(|i| band_linear(i, n, working)).collect();
    let mattes: Vec<Vec<[f32; 4]>> =
        (0..n).map(|i| row(&ctx, &mut comp, &armed(ZoneView::Matte, i as u32))).collect();
    let graded = row(&ctx, &mut comp, &fused(Some(zones())));
    let hard = row(&ctx, &mut comp, &armed(ZoneView::Hard, 1));

    let mut owners = Vec::new();
    for x in 0..W as usize {
        let w: Vec<f32> = (0..n).map(|i| mattes[i][x][0]).collect();
        // The heaviest zone, the lower one on a tie — skipping pixels whose
        // top two are within half-float noise, where either answer is fair.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| w[b].partial_cmp(&w[a]).unwrap().then(a.cmp(&b)));
        let (top, next) = (order[0], order[1]);
        assert!(close(hard[x][3], graded[x][3]), "alpha is not the preview's to change");
        if w[top] - w[next] < 0.01 {
            continue;
        }
        owners.push(top);
        for c in 0..3 {
            assert!(
                close(hard[x][c], cols[top][c]),
                "x={x} ch{c}: {} where zone {top} (weights {w:?}) is {}",
                hard[x][c],
                cols[top][c]
            );
        }
    }
    // One step per boundary: the owners along the ramp run 0, 1, 2 in order.
    owners.dedup();
    assert_eq!(owners, (0..n).collect::<Vec<_>>(), "the ramp should cross each boundary once");
    // Unarmed, it is the graded picture.
    let mut z = zones();
    z.preview = ZonePreview { view: ZoneView::Hard, zone: 0 };
    let unarmed = row(&ctx, &mut comp, &fused(Some(z)));
    for x in 0..W as usize {
        for c in 0..4 {
            assert!(close(unarmed[x][c], graded[x][c]), "an unviewed node previewed at {x}");
        }
    }
}

/// "Adding ranges crashes." Replay exactly what the app does when the
/// ranges tab's button is pressed: a PLAIN grade is on screen, the same node
/// gains zones, the compositor is asked for the new topology ASYNC (the app's
/// path), and frames keep encoding — params written from the NEW graph
/// against whatever pipeline is live — until the compile lands. Then a band
/// is split and dissolved, which changes the zone count (topology) again.
#[test]
fn adding_zones_live_does_not_crash_and_lands() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    upload_ramp(&ctx, &mut comp);
    let timer = GpuTimer::new(&ctx);
    let mut g = fused(None);
    assert!(comp.set_graph(&g));
    let frame = |comp: &mut Compositor, g: &Graph| {
        comp.write_params(g);
        let mut e = ctx.device.create_command_encoder(&Default::default());
        comp.encode_comp(&mut e, &timer);
        ctx.queue.submit([e.finish()]);
    };
    frame(&mut comp, &g);
    let steps: Vec<Box<dyn Fn(&mut GradeParams)>> = vec![
        Box::new(|gp| gp.zones = Some(GradeZones::default())),
        Box::new(|gp| {
            let z = gp.zones.as_mut().unwrap();
            let n = z.key.split_band(1);
            z.band_split(n);
        }),
        Box::new(|gp| {
            let z = gp.zones.as_mut().unwrap();
            z.key.dissolve_band(0);
            z.band_dissolved(0);
        }),
        Box::new(|gp| gp.zones = None),
    ];
    for step in steps {
        if let Some(NodeKind::Grade(gp)) = g.nodes.get_mut(&1) {
            step(gp);
        }
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        loop {
            let landed = comp.set_graph_async(&g);
            frame(&mut comp, &g);
            if landed {
                break;
            }
            assert!(std::time::Instant::now() < deadline, "the zoned topology never landed");
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    }
}
