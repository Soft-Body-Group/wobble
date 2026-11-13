//! Headless integration for the Defocus node: both modes through the real
//! Compositor, with a depth plate wired into the node's depth port.
//!
//! What these lock down, in order of what would hurt most to break:
//! * depth actually drives the circle of confusion, and the GPU's CoC agrees
//!   with `DefocusParams::coc_px` on the CPU (the two are hand-mirrored);
//! * an unwired depth port still means a plain uniform disc, so every setup
//!   authored before depth existed renders the same;
//! * Accurate mode's occlusion test does the thing Fast mode cannot — a
//!   blurry foreground spreads OUT over a sharp background;
//! * the bokeh kernel is really the aperture shape (blades, cat's-eye).
//!
//! Self-skips without a GPU (CI).

use engine::compositor::Compositor;
use engine::decode::ImageInfo;
use engine::graph::{DefocusMode, DefocusParams, DepthChannel, Graph, NodeKind};
use engine::timing::GpuTimer;
use half::f16;

const W: u32 = 256;
const H: u32 = 256;

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP defocus test: no adapter ({e})");
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

fn render(ctx: &engine::GpuContext, comp: &mut Compositor, graph: &Graph) -> Vec<[f32; 4]> {
    assert!(comp.set_graph(graph), "fused shader failed to compile");
    let timer = GpuTimer::new(ctx);
    // Warm-up encode: stage textures are allocated inside encode_comp, and
    // write_params only reaches stages that already exist.
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
    let mut out = Vec::with_capacity((W * H) as usize);
    for y in 0..H as usize {
        let row = &data[y * padded as usize..][..(W * 8) as usize];
        let halves: &[f16] = bytemuck::cast_slice(row);
        for x in 0..W as usize {
            let p = &halves[x * 4..][..4];
            out.push([p[0].to_f32(), p[1].to_f32(), p[2].to_f32(), p[3].to_f32()]);
        }
    }
    drop(data);
    staging.unmap();
    out
}

/// One bright pixel at the centre of black. A flat disc turns it into a disc
/// of exactly the CoC radius, which is what makes the radius measurable.
fn point() -> Vec<[f16; 4]> {
    let mut px = vec![[f16::ZERO, f16::ZERO, f16::ZERO, f16::ONE]; (W * H) as usize];
    px[(H / 2 * W + W / 2) as usize] = [f16::from_f32(1000.0); 4];
    px
}

/// A small bright square at the centre — what a spiral gather can actually
/// find, unlike a single texel.
fn blob(r: i32) -> Vec<[f16; 4]> {
    let mut px = vec![[f16::ZERO, f16::ZERO, f16::ZERO, f16::ONE]; (W * H) as usize];
    for dy in -r..=r {
        for dx in -r..=r {
            let x = (W / 2) as i32 + dx;
            let y = (H / 2) as i32 + dy;
            px[y as usize * W as usize + x as usize] = [f16::from_f32(200.0); 4];
        }
    }
    px
}

/// Constant depth everywhere.
fn flat_depth(z: f32) -> Vec<[f16; 4]> {
    vec![[f16::from_f32(z), f16::ZERO, f16::ZERO, f16::ONE]; (W * H) as usize]
}

/// Reader 0 -> Defocus, optionally with reader 1 on the depth port.
fn graph_of(p: DefocusParams, depth: bool) -> Graph {
    let mut g = Graph::default();
    g.nodes.insert(1, NodeKind::read(0, ""));
    g.nodes.insert(2, NodeKind::Defocus(p));
    g.edges.insert((2, 0), 1);
    if depth {
        g.nodes.insert(3, NodeKind::read(1, ""));
        g.edges.insert((2, 2), 3);
    }
    g.view = Some(2);
    g
}

/// Furthest lit pixel from centre along the middle row — the CoC radius the
/// GPU actually produced.
fn measured_radius(out: &[[f32; 4]]) -> f32 {
    let row = (H / 2) as usize * W as usize;
    let mut r = 0.0f32;
    for x in 0..W as usize {
        if out[row + x][0] > 1e-4 {
            r = r.max((x as f32 - (W / 2) as f32).abs());
        }
    }
    r
}

/// A deliberately shallow rig: 100 mm at f/1.4 focused at 1 m. At a 256 px
/// canvas an ordinary 50mm/f2.8 at 10 m has a SUB-PIXEL circle of confusion,
/// which would make every assertion below pass on a sharp image.
fn base(radius: f32) -> DefocusParams {
    DefocusParams {
        radius,
        mode: DefocusMode::Fast,
        focus: 1.0,
        unit: 1.0,
        focal_view: false,
        previewing: false,
        f_stop: 1.4,
        focal_len: 100.0,
        filmback: 36.0,
        inverse_depth: false,
        depth_channel: DepthChannel::R,
        blades: 0,
        rotation: 0.0,
        catseye: 0.0,
        samples: 256,
        edge: 0.0,
        bright_clamp: 0.0,
    }
}

/// The GPU's circle of confusion has to agree with the CPU mirror, or every
/// focus control in the UI lies about where the focal plane is.
#[test]
fn coc_matches_the_cpu_mirror() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 2);
    upload(&ctx, &mut comp, 0, &point());
    comp.ensure_output(W, H);

    let p = base(60.0);
    // A spread of depths either side of the focal plane, all inside the clamp
    // (14 px to 45 px of CoC) so nothing is measuring the clamp instead.
    for z in [0.6f32, 0.8, 1.4, 2.0, 5.0] {
        upload(&ctx, &mut comp, 1, &flat_depth(z));
        let out = render(&ctx, &mut comp, &graph_of(p, true));
        let want = p.coc_px(z, W as f32);
        let got = measured_radius(&out);
        assert!(want > 8.0, "z={z} must actually defocus, CPU says {want:.2} px");
        assert!(
            (got - want).abs() <= 1.5,
            "z={z}: CPU says {want:.2} px, GPU produced {got:.2} px"
        );
    }

    // At the focal plane nothing moves: the point stays a point.
    upload(&ctx, &mut comp, 1, &flat_depth(p.focus));
    let sharp = render(&ctx, &mut comp, &graph_of(p, true));
    assert!(
        measured_radius(&sharp) <= 1.0,
        "in-focus depth must stay sharp, got {}",
        measured_radius(&sharp)
    );
}

/// Cost is the reason Fast mode exists: the prefix-sum strip walk is O(r),
/// so a huge CoC must not be a different order of work. This asserts the
/// RESULT is right at a radius a tap-per-texel gather could not afford.
#[test]
fn fast_mode_holds_up_at_a_huge_radius() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 2);
    upload(&ctx, &mut comp, 0, &point());
    // z=0.3 in front of a 1 m focal plane is ~132 px of CoC: past the cap,
    // so this measures the clamp AND a radius no per-texel gather would pay.
    upload(&ctx, &mut comp, 1, &flat_depth(0.3));
    comp.ensure_output(W, H);

    let p = base(100.0);
    let out = render(&ctx, &mut comp, &graph_of(p, true));
    let r = measured_radius(&out);
    assert!(r > 90.0, "clamped CoC should reach the radius cap, got {r}");

    // Flatness has to be probed where the disc is FULLY in frame: the pass
    // normalizes by the in-bounds texel count (deliberately — the
    // alternative is a dark border), so a disc hanging off the edge reads
    // brighter. At r=60 both probes below are interior.
    let mid = render(&ctx, &mut comp, &graph_of(base(60.0), true));
    let row = (H / 2) as usize * W as usize;
    let (near, far) = (mid[row + 128 + 10][0], mid[row + 128 + 40][0]);
    assert!(
        (near - far).abs() / near.max(1e-6) < 0.05,
        "flat disc must be flat: {near} at 10px vs {far} at 40px"
    );
}

/// Every setup authored before the depth port existed must render exactly as
/// it did: unwired depth = uniform disc of `radius`.
#[test]
fn unwired_depth_is_a_uniform_disc() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 2);
    upload(&ctx, &mut comp, 0, &point());
    comp.ensure_output(W, H);

    // Both modes: Accurate must not quietly collapse to a sharp image just
    // because there is no depth to compute a CoC from.
    let out = render(&ctx, &mut comp, &graph_of(base(20.0), false));
    let r = measured_radius(&out);
    assert!((r - 20.0).abs() <= 1.5, "fast: expected a 20px disc, got {r}");

    upload(&ctx, &mut comp, 0, &blob(2));
    let slow = render(
        &ctx,
        &mut comp,
        &graph_of(
            DefocusParams { mode: DefocusMode::Accurate, samples: 512, ..base(20.0) },
            false,
        ),
    );
    let rs = measured_radius(&slow);
    assert!(
        (rs - 22.0).abs() <= 3.0,
        "accurate: expected a ~20px disc around a 2px blob, got {rs}"
    );
}

/// The whole point of Accurate mode. Foreground at the left, background at
/// the right, both lit; only the foreground is defocused. Fast mode gathers
/// with the destination's own CoC, so nothing crosses the border; Accurate
/// runs the scatter-as-gather test, so the blurry foreground bleeds OVER the
/// sharp background.
#[test]
fn accurate_mode_bleeds_foreground_over_background() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 2);
    comp.ensure_output(W, H);

    // Left half lit and NEAR (very defocused); right half black and AT the
    // focal plane (perfectly sharp).
    let mut img = vec![[f16::ZERO, f16::ZERO, f16::ZERO, f16::ONE]; (W * H) as usize];
    let mut dep = vec![[f16::ZERO, f16::ZERO, f16::ZERO, f16::ONE]; (W * H) as usize];
    for y in 0..H as usize {
        for x in 0..W as usize {
            let near = x < (W / 2) as usize;
            if near {
                img[y * W as usize + x] = [f16::from_f32(4.0); 4];
            }
            dep[y * W as usize + x][0] = f16::from_f32(if near { 0.5 } else { 1.0 });
        }
    }
    upload(&ctx, &mut comp, 0, &img);
    upload(&ctx, &mut comp, 1, &dep);

    let p = base(30.0);
    let fast = render(&ctx, &mut comp, &graph_of(p, true));
    let slow = render(
        &ctx,
        &mut comp,
        &graph_of(DefocusParams { mode: DefocusMode::Accurate, ..p }, true),
    );

    // 8 px INTO the sharp background, past the border at x=128.
    let probe = (H / 2) as usize * W as usize + 136;
    assert!(
        slow[probe][0] > 0.2,
        "accurate mode should spill foreground onto the background, got {}",
        slow[probe][0]
    );
    assert!(
        fast[probe][0] < slow[probe][0] * 0.5,
        "fast mode gathers only: {} should be well under {}",
        fast[probe][0],
        slow[probe][0]
    );
}

/// The kernel is genuinely the aperture: a 4-blade iris makes a square
/// bokeh, so the diagonal reaches further than the axis. A circle does not.
#[test]
fn blades_shape_the_bokeh() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 2);
    // NOT a single pixel: a fixed-count spiral samples a fraction of its
    // search disc, so a one-texel source is missed by most output pixels.
    // That is a real property of the mode, not a test artefact.
    upload(&ctx, &mut comp, 0, &blob(3));
    upload(&ctx, &mut comp, 1, &flat_depth(2.0));
    comp.ensure_output(W, H);

    let p = DefocusParams {
        mode: DefocusMode::Accurate,
        samples: 1024,
        ..base(40.0)
    };
    let lit = |o: &Vec<[f32; 4]>, dx: i32, dy: i32| {
        let x = (W / 2) as i32 + dx;
        let y = (H / 2) as i32 + dy;
        o[y as usize * W as usize + x as usize][0] > 1e-4
    };

    // A 4-blade iris at rotation 0 puts its CORNERS on the axes and its flat
    // sides across the diagonals, inscribed in the circle it replaces. So a
    // probe out along the diagonal falls inside the circle and outside the
    // square — the shape is anisotropic, which a disc never is.
    let square = render(&ctx, &mut comp, &graph_of(DefocusParams { blades: 4, ..p }, true));
    let round = render(&ctx, &mut comp, &graph_of(p, true));
    let r = measured_radius(&round);
    assert!(r > 8.0, "need a real disc to compare shapes, got {r}");
    let d = (r * 0.62) as i32; // |(d,d)| ~ 0.88r: inside the circle, past the flat
    assert!(lit(&round, d, d), "a circle should reach {d},{d}");
    assert!(!lit(&square, d, d), "4 blades should NOT reach across the flat at {d},{d}");
    // ...but along the axis, where the corner points, it still reaches.
    assert!(lit(&square, d, 0), "the blade corner should still reach {d},0");

    // Optical vignetting clips the disc against a circle displaced toward
    // the frame edge, so a point away from centre loses its outer flank.
    let mut off = vec![[f16::ZERO, f16::ZERO, f16::ZERO, f16::ONE]; (W * H) as usize];
    let cx = (W as f32 * 0.85) as usize;
    for dy in -3i32..=3 {
        for dx in -3i32..=3 {
            let y = (H / 2) as i32 + dy;
            off[y as usize * W as usize + (cx as i32 + dx) as usize] =
                [f16::from_f32(200.0); 4];
        }
    }
    upload(&ctx, &mut comp, 0, &off);
    let cat = render(
        &ctx,
        &mut comp,
        &graph_of(DefocusParams { catseye: 0.9, ..p }, true),
    );
    let row = (H / 2) as usize * W as usize;
    let outward = cat[row + cx + (r as usize * 3 / 4)][0];
    let inward = cat[row + cx - (r as usize * 3 / 4)][0];
    assert!(
        inward > outward * 2.0,
        "cat's-eye should cut the outward flank: in {inward} vs out {outward}"
    );
}

/// Premultiplied throughout: rgb and alpha ride the same weights, so a
/// defocused matte edge keeps colour/alpha in step instead of fringing.
#[test]
fn premultiplied_edge_stays_neutral() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 2);
    comp.ensure_output(W, H);

    // A premultiplied grey square on transparent black.
    let mut img = vec![[f16::ZERO; 4]; (W * H) as usize];
    for y in 64..192usize {
        for x in 64..192usize {
            // Premultiplied: colour 0.5 at alpha 1.
            img[y * W as usize + x] =
                [f16::from_f32(0.5), f16::from_f32(0.5), f16::from_f32(0.5), f16::ONE];
        }
    }
    upload(&ctx, &mut comp, 0, &img);
    upload(&ctx, &mut comp, 1, &flat_depth(2.0));

    for mode in [DefocusMode::Fast, DefocusMode::Accurate] {
        let out = render(
            &ctx,
            &mut comp,
            &graph_of(DefocusParams { mode, ..base(16.0) }, true),
        );
        // Walk across the soft edge: unpremultiplied colour must stay 0.5
        // everywhere alpha is meaningful.
        let row = (H / 2) as usize * W as usize;
        for x in 56..72usize {
            let [r, _, _, a] = out[row + x];
            if a > 0.05 {
                assert!(
                    (r / a - 0.5).abs() < 0.03,
                    "{mode:?}: unpremult colour drifted to {} at x={x} (a={a})",
                    r / a
                );
            }
        }
    }
}

/// The DATA pipe carries a depth exactly as the old direct wire did — past
/// 1, unclamped, from whichever channel it was in. The depth plate here
/// holds garbage in R and the real depths in G, all of them > 1 (a clamp at
/// 1 would put every one of them on the same wrong circle): picking G on a
/// ToData must render what picking G on the Defocus itself rendered, which
/// is exactly the rewrite the loader's migration performs.
#[test]
fn a_todata_depth_renders_what_the_direct_wire_did() {
    use engine::graph::{MatteChannel, ToDataParams};
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 2);
    upload(&ctx, &mut comp, 0, &point());
    comp.ensure_output(W, H);
    for z in [1.4f32, 2.0, 5.0] {
        let plate = vec![
            [f16::from_f32(0.25), f16::from_f32(z), f16::from_f32(9.0), f16::ONE];
            (W * H) as usize
        ];
        upload(&ctx, &mut comp, 1, &plate);

        // Before: the Defocus picks G itself.
        let mut old = base(60.0);
        old.depth_channel = DepthChannel::G;
        let direct = render(&ctx, &mut comp, &graph_of(old, true));

        // After: a ToData picks G onto the data pipe; the Defocus reads R.
        let mut g = graph_of(base(60.0), true);
        g.nodes.insert(4, NodeKind::ToData(ToDataParams { channel: MatteChannel::G }));
        g.edges.insert((4, 0), 3);
        g.edges.insert((2, 2), 4);
        let piped = render(&ctx, &mut comp, &g);

        let want = base(60.0).coc_px(z, W as f32);
        let got = measured_radius(&piped);
        assert!(
            (got - want).abs() <= 1.5,
            "z={z}: the data pipe must carry the depth unclamped — CoC {got:.2} px, want {want:.2}"
        );
        let worst = direct
            .iter()
            .zip(&piped)
            .map(|(a, b)| (a[0] - b[0]).abs().max((a[1] - b[1]).abs()).max((a[2] - b[2]).abs()))
            .fold(0.0f32, f32::max);
        assert!(worst < 1e-3, "z={z}: ToData(G) and the Defocus's own G pick differ by {worst}");
    }
}

/// Focus is a number read off the pass: the same shot with its depth in
/// centimetres (unit 0.01, focus 100) blurs exactly as it does in metres
/// (unit 1, focus 1), and with 1/Z the focus is the 1/Z value.
#[test]
fn focus_is_a_pass_value_in_any_unit() {
    let m = base(60.0);
    let cm = DefocusParams { unit: 0.01, focus: 100.0, ..m };
    for z in [0.6f32, 0.8, 1.4, 2.0, 5.0] {
        let (a, b) = (m.coc_px(z, W as f32), cm.coc_px(z * 100.0, W as f32));
        assert!((a - b).abs() < 1e-3, "z={z}: metres {a}, centimetres {b}");
    }
    // World-space magnitudes are not clamped anywhere.
    let far = DefocusParams { unit: 0.01, focus: 25_000.0, ..m };
    assert_eq!(far.coc_px(25_000.0, W as f32), 0.0);
    assert!(far.coc_px(2_000.0, W as f32) > 1.0);
    // Disparity: focus 0.5 IS the plane at Z = 2.
    let inv = DefocusParams { inverse_depth: true, focus: 0.5, ..m };
    assert_eq!(inv.coc_px(0.5, W as f32), 0.0);
    assert!(inv.coc_px(0.2, W as f32) > 1.0);

    // And the GPU agrees in centimetres.
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 2);
    upload(&ctx, &mut comp, 0, &point());
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp, 1, &flat_depth(200.0));
    let out = render(&ctx, &mut comp, &graph_of(cm, true));
    let want = cm.coc_px(200.0, W as f32);
    let got = measured_radius(&out);
    assert!(want > 8.0 && (got - want).abs() <= 1.5, "cm: CPU {want:.2}, GPU {got:.2}");
    // The depth texture the focus pick reads is there once a frame ran.
    assert!(comp.defocus_depth_texture(2).is_some());
}

/// The focal-plane view: green at the plane, red in front, blue behind —
/// and only while previewing (a render never sees it). Both modes.
#[test]
fn focal_view_colours_the_plane() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 2);
    upload(&ctx, &mut comp, 0, &blob(3));
    comp.ensure_output(W, H);
    for mode in [DefocusMode::Fast, DefocusMode::Accurate] {
        let p = DefocusParams { mode, focal_view: true, previewing: true, ..base(40.0) };
        let px = |z: f32, comp: &mut Compositor| {
            upload(&ctx, comp, 1, &flat_depth(z));
            let o = render(&ctx, comp, &graph_of(p, true));
            o[10 * W as usize + 10]
        };
        let at = px(1.0, &mut comp);
        assert!(at[1] > at[0] && at[1] > at[2], "{mode:?} in focus not green: {at:?}");
        let near = px(0.6, &mut comp);
        assert!(near[0] > near[1] && near[0] > near[2], "{mode:?} near not red: {near:?}");
        let far = px(5.0, &mut comp);
        assert!(far[2] > far[0] && far[2] > far[1], "{mode:?} far not blue: {far:?}");
        // Authored but not the view target: the real defocus, untouched.
        upload(&ctx, &mut comp, 1, &flat_depth(5.0));
        let off = render(&ctx, &mut comp, &graph_of(DefocusParams { previewing: false, ..p }, true));
        let plain = render(&ctx, &mut comp, &graph_of(DefocusParams { focal_view: false, ..p }, true));
        assert_eq!(off, plain, "{mode:?}: focal view leaked into a non-preview render");
    }
}

/// The Z port is what the ToData on it SAYS, not the rgba behind it: a
/// plate whose R is 5 and G is 1, through a ToData picking G, focused at 1,
/// must stay sharp — reading R would blur it. And the focus pick's texel is
/// that same G.
#[test]
fn z_through_a_todata_uses_its_channel() {
    use engine::graph::{MatteChannel, ToDataParams};
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 2);
    upload(&ctx, &mut comp, 0, &point());
    let (r, gg, b) = (5.0f32, 1.0f32, 3.0f32);
    let rg = vec![
        [f16::from_f32(r), f16::from_f32(gg), f16::from_f32(b), f16::ONE];
        (W * H) as usize
    ];
    upload(&ctx, &mut comp, 1, &rg);
    comp.ensure_output(W, H);
    for (ch, z) in [
        (MatteChannel::G, gg),
        (MatteChannel::B, b),
        (MatteChannel::Luma, 0.2126 * r + 0.7152 * gg + 0.0722 * b),
    ] {
        let p = DefocusParams { focus: z, ..base(60.0) };
        let mut g = graph_of(p, false);
        g.nodes.insert(3, NodeKind::read(1, ""));
        g.nodes.insert(4, NodeKind::ToData(ToDataParams { channel: ch }));
        g.edges.insert((4, 0), 3);
        g.edges.insert((2, 2), 4);
        let out = render(&ctx, &mut comp, &g);
        assert!(
            measured_radius(&out) <= 1.0,
            "{ch:?}: {}px of blur — the ToData's value ({z}, in focus) was not what arrived",
            measured_radius(&out)
        );
    }
}

/// A depth plate: `z_bg` everywhere, `z_fg` inside a centred square of
/// half-size `h` (shrunk by `inset` px so a colour edge can outrun it).
fn square_depth(z_bg: f32, z_fg: f32, h: i32, inset: i32) -> Vec<[f16; 4]> {
    let mut px = vec![[f16::from_f32(z_bg), f16::ZERO, f16::ZERO, f16::ONE]; (W * H) as usize];
    let hh = h - inset;
    for dy in -hh..=hh {
        for dx in -hh..=hh {
            let x = (W / 2) as i32 + dx;
            let y = (H / 2) as i32 + dy;
            px[y as usize * W as usize + x as usize] = [f16::from_f32(z_fg), f16::ZERO, f16::ZERO, f16::ONE];
        }
    }
    px
}

/// A bright square of half-size `h` on a dim flat background.
fn square_plate(h: i32, fg: f32, bg: f32) -> Vec<[f16; 4]> {
    let mut px = vec![[f16::from_f32(bg), f16::from_f32(bg), f16::from_f32(bg), f16::ONE]; (W * H) as usize];
    for dy in -h..=h {
        for dx in -h..=h {
            let x = (W / 2) as i32 + dx;
            let y = (H / 2) as i32 + dy;
            px[y as usize * W as usize + x as usize] = [f16::from_f32(fg); 4];
        }
    }
    px
}

fn at(out: &[[f32; 4]], dx: i32, dy: i32) -> [f32; 4] {
    let x = (W / 2) as i32 + dx;
    let y = (H / 2) as i32 + dy;
    out[y as usize * W as usize + x as usize]
}

/// Fast mode is TWO LAYERS now: a blurry foreground square over an in-focus
/// background bleeds OUT past its own edge (a plain gather could not — the
/// background pixel's own CoC is zero), the background far from it is
/// untouched, under the edge the picture is a mix of the two and never
/// darker than either, and deep inside the square the foreground stands.
#[test]
fn fast_mode_bleeds_foreground_over_background() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 2);
    comp.ensure_output(W, H);
    // Square of 1.0 on 0.2, in front (z 0.6: ~45 px of CoC at base(60)'s
    // lens, so a 20 px square is comfortably blurrier than its size).
    let (fg, bg) = (1.0f32, 0.2f32);
    upload(&ctx, &mut comp, 0, &square_plate(20, fg, bg));
    upload(&ctx, &mut comp, 1, &square_depth(1.0, 0.6, 20, 0));
    let p = base(60.0);
    let out = render(&ctx, &mut comp, &graph_of(p, true));
    let coc = p.coc_px(0.6, W as f32);
    assert!(coc > 30.0, "the square must be well out of focus ({coc} px)");
    // 10 px outside the square's edge: foreground light has arrived.
    let near_edge = at(&out, 30, 0)[0];
    assert!(near_edge > bg + 0.05, "no bleed past the edge: {near_edge} vs bg {bg}");
    assert!(near_edge < fg, "bleed cannot exceed the foreground: {near_edge}");
    // Far outside the reach: exactly the background.
    let far = at(&out, 100, 0)[0];
    assert!((far - bg).abs() < 1e-3, "background was disturbed far from the square: {far}");
    // The bleed falls off with distance, like a blur.
    let closer = at(&out, 24, 0)[0];
    let further = at(&out, 40, 0)[0];
    assert!(closer > near_edge && near_edge > further, "bleed does not fall off: {closer} {near_edge} {further}");
    // The centre of a square SMALLER than its CoC is see-through: the
    // foreground's light has spread thin and the background shows behind
    // it — a real lens does that to a small out-of-focus object.
    let centre = at(&out, 0, 0)[0];
    assert!(centre > bg + 0.1 && centre < fg - 0.1, "a small blurry square should thin at its centre: {centre}");
    // A square much bigger than the CoC: nothing but foreground in reach
    // at its centre, so the foreground.
    upload(&ctx, &mut comp, 0, &square_plate(90, fg, bg));
    upload(&ctx, &mut comp, 1, &square_depth(1.0, 0.6, 90, 0));
    let big = render(&ctx, &mut comp, &graph_of(p, true));
    let centre = at(&big, 0, 0)[0];
    assert!((centre - fg).abs() < 0.02, "centre of a big square is not the foreground: {centre}");
}

/// The far layer EXCLUDES the foreground: a sharp background pixel next
/// to a blurry foreground edge is the background mixed with foreground
/// coverage — never the foreground's colour leaking through the far
/// average. With a textured background (bright left, dark right) the far
/// fill under the edge follows the side it is on.
#[test]
fn far_layer_fills_from_the_background_around_it() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 2);
    comp.ensure_output(W, H);
    // Background: 0.8 on the left half, 0.1 on the right; a foreground
    // square of 0.5 (grey, between the two) in the middle.
    let mut px = vec![[f16::ZERO; 4]; (W * H) as usize];
    for y in 0..H as usize {
        for x in 0..W as usize {
            let v = if x < (W / 2) as usize { 0.8 } else { 0.1 };
            px[y * W as usize + x] = [f16::from_f32(v), f16::from_f32(v), f16::from_f32(v), f16::ONE];
        }
    }
    for dy in -12..=12i32 {
        for dx in -12..=12i32 {
            let x = (W / 2) as i32 + dx;
            let y = (H / 2) as i32 + dy;
            px[y as usize * W as usize + x as usize] = [f16::from_f32(0.5); 4];
        }
    }
    upload(&ctx, &mut comp, 0, &px);
    upload(&ctx, &mut comp, 1, &square_depth(1.0, 0.6, 12, 0));
    let out = render(&ctx, &mut comp, &graph_of(base(60.0), true));
    // Just outside the square on each side the far estimate is that
    // side's background; the mix with the 0.5 foreground sits BETWEEN.
    let l = at(&out, -18, 0)[0];
    let r = at(&out, 18, 0)[0];
    assert!(l > 0.5 && l < 0.8, "left of the square should be between fg 0.5 and bg 0.8: {l}");
    assert!(r > 0.1 && r < 0.5, "right of the square should be between bg 0.1 and fg 0.5: {r}");
}

/// `edge`: the nearest depth within n px stands for the pixel. A colour
/// square one pixel LARGER than its depth square (the antialiasing case)
/// wears background depth on its rim at edge 0 — in the focal-plane view
/// the rim reads as far — and foreground depth at edge 1.
#[test]
fn edge_takes_the_nearer_depth() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 2);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp, 0, &square_plate(10, 1.0, 0.2));
    upload(&ctx, &mut comp, 1, &square_depth(1.0, 0.6, 10, 1));
    let view = |edge: f32, comp: &mut Compositor| {
        let p = DefocusParams { focal_view: true, previewing: true, edge, ..base(60.0) };
        let o = render(&ctx, comp, &graph_of(p, true));
        at(&o, 10, 0) // the rim pixel
    };
    let off = view(0.0, &mut comp);
    assert!(off[1] > off[0], "at edge 0 the rim wears the (in-focus) background: {off:?}");
    let on = view(1.0, &mut comp);
    assert!(on[0] > on[1] && on[0] > on[2], "at edge 1 the rim is foreground (red): {on:?}");
    // And it holds in Accurate mode's map too (same pass).
    let p = DefocusParams { mode: DefocusMode::Accurate, focal_view: true, previewing: true, edge: 1.0, ..base(60.0) };
    let o = render(&ctx, &mut comp, &graph_of(p, true));
    let a = at(&o, 10, 0);
    assert!(a[0] > a[1], "accurate: rim not foreground at edge 1: {a:?}");
}

/// Accurate's search radius is per pixel: a small CoC under a big max
/// radius used to spread its taps over the whole max disc and land almost
/// none, so a mildly blurry region stayed sharp (and dotted). It must now
/// blur a checkerboard as much as Fast mode does.
#[test]
fn accurate_blurs_a_small_coc_under_a_big_max_radius() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 2);
    comp.ensure_output(W, H);
    let mut px = vec![[f16::ZERO; 4]; (W * H) as usize];
    for y in 0..H as usize {
        for x in 0..W as usize {
            let v = if ((x / 2) + (y / 2)) % 2 == 0 { 1.0 } else { 0.0 };
            px[y * W as usize + x] = [f16::from_f32(v), f16::from_f32(v), f16::from_f32(v), f16::ONE];
        }
    }
    upload(&ctx, &mut comp, 0, &px);
    // base(200): max radius 200 px; z 1.1 is ~4 px of CoC on this lens.
    let p = DefocusParams { samples: 64, ..base(200.0) };
    let coc = p.coc_px(1.1, W as f32);
    assert!(coc > 2.0 && coc < 8.0, "want a small CoC: {coc}");
    upload(&ctx, &mut comp, 1, &flat_depth(1.1));
    // Spread of the middle row: a 2 px checker is 0.5 of it; blurred it
    // collapses toward grey, with Accurate's tap noise on top.
    let spread = |o: &[[f32; 4]]| {
        let row = (H / 2) as usize * W as usize;
        let xs: Vec<f32> = (40..(W as usize - 40)).map(|x| o[row + x][0]).collect();
        let mean = xs.iter().sum::<f32>() / xs.len() as f32;
        (xs.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / xs.len() as f32).sqrt()
    };
    let sharp = spread(&render(&ctx, &mut comp, &graph_of(DefocusParams { radius: 0.0, ..p }, true)));
    let fast = spread(&render(&ctx, &mut comp, &graph_of(p, true)));
    let acc = spread(&render(&ctx, &mut comp, &graph_of(DefocusParams { mode: DefocusMode::Accurate, ..p }, true)));
    assert!(sharp > 0.4, "the checker should be sharp at radius 0: {sharp}");
    assert!(fast < 0.1, "fast mode did not blur the checker: {fast}");
    assert!(acc < 0.15, "accurate left the checker sharp under a big max radius: {acc} (fast {fast}, sharp {sharp})");
}

/// The highlight clamp: a 1000-unit speck through Accurate at a modest
/// tap count becomes a disc no brighter than the clamp.
#[test]
fn bright_clamp_tames_a_highlight() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 2);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp, 0, &point());
    upload(&ctx, &mut comp, 1, &flat_depth(2.0));
    let p = DefocusParams { mode: DefocusMode::Accurate, samples: 32, ..base(40.0) };
    let raw = render(&ctx, &mut comp, &graph_of(p, true));
    let peak = |o: &[[f32; 4]]| o.iter().map(|v| v[0]).fold(0.0f32, f32::max);
    assert!(peak(&raw) > 1.0, "unclamped: the speck should still be hot somewhere: {}", peak(&raw));
    let tamed = render(&ctx, &mut comp, &graph_of(DefocusParams { bright_clamp: 1.0, ..p }, true));
    assert!(peak(&tamed) <= 1.0 + 1e-3, "clamped at 1: peak {}", peak(&tamed));
    // Deterministic noise: the same frame twice is the same frame.
    let again = render(&ctx, &mut comp, &graph_of(DefocusParams { bright_clamp: 1.0, ..p }, true));
    assert_eq!(tamed, again);
}

/// The MODULATE port scales the circle of confusion by hand: with no depth
/// a flat half matte halves the uniform disc (both modes), with depth it
/// halves what the depth pass asked for, and unwired it is no port at all.
#[test]
fn modulate_scales_the_circle_of_confusion() {
    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 3);
    comp.ensure_output(W, H);
    upload(&ctx, &mut comp, 0, &point());
    upload(&ctx, &mut comp, 2, &flat_depth(0.5));
    let with_mod = |mut g: Graph| {
        g.nodes.insert(5, NodeKind::read(2, ""));
        g.edges.insert((2, 4), 5);
        g
    };

    let out = render(&ctx, &mut comp, &with_mod(graph_of(base(20.0), false)));
    let r = measured_radius(&out);
    assert!((r - 10.0).abs() <= 1.5, "fast, no depth: a half matte is a 10px disc, got {r}");

    upload(&ctx, &mut comp, 0, &blob(2));
    let slow = render(
        &ctx,
        &mut comp,
        &with_mod(graph_of(DefocusParams { mode: DefocusMode::Accurate, samples: 512, ..base(20.0) }, false)),
    );
    let rs = measured_radius(&slow);
    assert!((rs - 12.0).abs() <= 3.0, "accurate, no depth: ~10px around a 2px blob, got {rs}");

    // With depth: half of the depth pass's own CoC.
    upload(&ctx, &mut comp, 0, &point());
    upload(&ctx, &mut comp, 1, &flat_depth(0.5));
    let p = base(60.0);
    let full = measured_radius(&render(&ctx, &mut comp, &graph_of(p, true)));
    let half = measured_radius(&render(&ctx, &mut comp, &with_mod(graph_of(p, true))));
    assert!(full > 8.0, "the rig must blur at z=0.5, got {full}");
    assert!((half - full / 2.0).abs() <= 1.5, "depth + half matte: {full} -> {half}");

    // A black matte is sharp.
    upload(&ctx, &mut comp, 2, &flat_depth(0.0));
    let none = measured_radius(&render(&ctx, &mut comp, &with_mod(graph_of(base(20.0), false))));
    assert!(none <= 0.5, "a black matte leaves it sharp, got {none}");
}
