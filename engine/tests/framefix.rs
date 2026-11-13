//! Freeze and Frame Fix: two time nodes on the reader-remap machinery.
//!
//! What is pinned: a Freeze reads the held frame at every tick and moving it
//! is no recompile; a Frame Fix in blend mode puts its input on two time
//! branches that read the good neighbours of a bad frame, dissolves them to
//! their exact average there, and passes every other frame through bit for
//! bit; loading flow frames swaps the branches for its reader. The flow
//! job (RIFE) makes a bad frame from non-adjacent neighbours along the
//! motion. The GPU halves self-skip without an adapter (CI), the flow one
//! without `data/rife/flownet.pkl`.

use engine::compositor::Compositor;
use engine::decode::ImageInfo;
use engine::graph::{FrameFixParams, FreezeParams, Graph, NodeKind};
use engine::timing::GpuTimer;
use half::f16;

const W: u32 = 16;
const H: u32 = 8;

fn ctx() -> Option<engine::GpuContext> {
    match engine::GpuContext::new() {
        Ok(c) => Some(c),
        Err(e) => {
            eprintln!("SKIP framefix test: no adapter ({e})");
            None
        }
    }
}

/// The colour a plate shows on its frame `f`: distinct per frame, and not
/// a round number in half floats, so an average is visibly an average.
fn plate(f: u64) -> [f32; 4] {
    let x = f as f32;
    [0.01 * x, 0.5 + 0.003 * x, 2.0 - 0.007 * x, 1.0]
}

fn upload(ctx: &engine::GpuContext, comp: &mut Compositor, slot: usize, v: [f32; 4]) {
    comp.ensure_reader(slot, ImageInfo { width: W, height: H });
    let px: Vec<[f16; 4]> = (0..W * H).map(|_| v.map(f16::from_f32)).collect();
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
            layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(padded), rows_per_image: Some(H) },
        },
        wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
    );
    ctx.queue.submit([encoder.finish()]);
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    let data = slice.get_mapped_range().expect("a mapped buffer");
    let halves: &[f16] = bytemuck::cast_slice(&data[..(W * 8) as usize]);
    let out = [halves[0].to_f32(), halves[1].to_f32(), halves[2].to_f32(), halves[3].to_f32()];
    drop(data);
    staging.unmap();
    out
}

/// Read (node 1) -> `kind` (node 2), viewed, each reader instance on its
/// own slot the way the app assigns them.
fn graph(kind: NodeKind) -> Graph {
    let mut g = Graph::default();
    g.nodes.insert(1, NodeKind::read(0, ""));
    g.nodes.insert(2, kind);
    g.edges.insert((2, 0), 1);
    g.view = Some(2);
    g.time_slots = g.reader_instances().iter().enumerate().map(|(i, inst)| ((inst.node, inst.key), i)).collect();
    g
}

/// What the player would put in every slot at `tick`: each instance's own
/// source frame of the plate.
fn feed(ctx: &engine::GpuContext, comp: &mut Compositor, g: &Graph, tick: u64) {
    for (i, inst) in g.reader_instances().iter().enumerate() {
        upload(ctx, comp, i, plate(inst.map.apply(tick)));
    }
}

fn h(v: [f32; 4]) -> [f32; 4] {
    v.map(|x| f16::from_f32(x).to_f32())
}

#[test]
fn a_freeze_reads_the_held_frame_at_every_tick() {
    let g = graph(NodeKind::Freeze(FreezeParams::at(1012.0)));
    let inst = g.reader_instances();
    assert_eq!(inst.len(), 1, "one branch of time, one reader");
    for t in [0u64, 1001, 1012, 1013, 2000] {
        assert_eq!(inst[0].map.apply(t), 1012, "tick {t}");
    }
    // Moving the frame is a time-map edit, never a recompile.
    let h0 = g.topology_hash();
    let moved = graph(NodeKind::Freeze(FreezeParams::at(1020.0)));
    assert_eq!(moved.topology_hash(), h0, "moving the freeze recompiled");
    assert_eq!(moved.reader_instances()[0].map.apply(5), 1020);

    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    for t in [1001u64, 1030] {
        feed(&ctx, &mut comp, &g, t);
        assert_eq!(render(&ctx, &mut comp, &g), h(plate(1012)), "tick {t}");
    }
}

#[test]
fn frame_fix_blend_is_the_neighbours_average_and_leaves_other_frames_alone() {
    let fix = FrameFixParams::new(0, 1012);
    let mut g = graph(NodeKind::FrameFix(fix.clone()));
    // The node's own (empty until flow frames load) reader aside.
    let inst: Vec<_> = g.reader_instances().into_iter().filter(|i| i.node == 1).collect();
    assert_eq!(inst.len(), 2, "two branches of time: before and after");
    let mut at_bad: Vec<u64> = inst.iter().map(|i| i.map.apply(1012)).collect();
    at_bad.sort_unstable();
    assert_eq!(at_bad, vec![1011, 1013]);
    assert!(inst.iter().all(|i| i.map.apply(1020) == 1020));

    let Some(ctx) = ctx() else { return };
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 4);
    comp.ensure_output(W, H);
    for t in [1010u64, 1011, 1012, 1013, 1014] {
        if let Some(NodeKind::FrameFix(p)) = g.nodes.get_mut(&2) {
            p.blend_w = fix.weight_at(t as i64);
        }
        feed(&ctx, &mut comp, &g, t);
        let got = render(&ctx, &mut comp, &g);
        if t == 1012 {
            let (a, b) = (h(plate(1011)), h(plate(1013)));
            for c in 0..4 {
                let want = 0.5 * (a[c] + b[c]);
                assert!((got[c] - want).abs() <= 2e-3 * want.abs().max(1.0), "ch {c}: {} vs {want}", got[c]);
            }
        } else {
            assert_eq!(got, h(plate(t)), "tick {t} must pass through exactly");
        }
    }
}

#[test]
fn loaded_flow_frames_replace_the_branches_with_the_reader() {
    let loaded = FrameFixParams { loaded: true, source: 3, ..FrameFixParams::new(3, 1012) };
    let g = graph(NodeKind::FrameFix(loaded));
    let inst = g.reader_instances();
    assert_eq!(inst.len(), 1, "the input is not sampled once loaded: {inst:?}");
    assert_eq!(inst[0].node, 2, "the node's own reader plays the frames");
    assert_ne!(
        g.topology_hash(),
        graph(NodeKind::FrameFix(FrameFixParams::new(3, 1012))).topology_hash(),
        "loading is topology"
    );
    // The list and the mode are not.
    let a = graph(NodeKind::FrameFix(FrameFixParams::new(0, 5)));
    let b = graph(NodeKind::FrameFix(FrameFixParams {
        mode: engine::graph::FrameFixMode::Flow,
        ..FrameFixParams { frames: vec![5, 9, 10], ..FrameFixParams::new(0, 5) }
    }));
    assert_eq!(a.topology_hash(), b.topology_hash(), "editing the list recompiled");
}

/// Flow mode's job: frame 1 of 0..=2 is bad, made by RIFE from frames 0
/// and 2 (NOT adjacent) at the halfway point: a texture moving 4 px a frame
/// comes back moved 4 px (halfway to frame 2), far closer than a cross-fade.
#[test]
fn the_flow_job_makes_a_bad_frame_from_its_neighbours() {
    use engine::rifetime::{frame_name, run, Job};
    use std::path::Path;
    let weights = Path::new("../data/rife/flownet.pkl");
    if !weights.exists() {
        eprintln!("SKIP: no data/rife/flownet.pkl");
        return;
    }
    let Some(ctx) = ctx() else { return };
    let dir = std::env::temp_dir().join(format!("wobble-framefix-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("in")).unwrap();
    let (w, hgt) = (192usize, 128usize);
    let tex = |x: f32, y: f32| {
        let v = 0.5 + 0.25 * (x * 0.13 + y * 0.05).sin() + 0.2 * (x * 0.041 - y * 0.11 + 1.0).sin();
        v * v * 2.5
    };
    let shot = |shift: f32| {
        let mut px = vec![f16::ZERO; w * hgt * 4];
        for y in 0..hgt {
            for x in 0..w {
                let v = tex(x as f32 - shift, y as f32);
                let i = (y * w + x) * 4;
                px[i] = f16::from_f32(v);
                px[i + 1] = f16::from_f32(tex(x as f32 - shift + 30.0, y as f32 + 9.0));
                px[i + 2] = f16::from_f32(v * 0.4);
                px[i + 3] = f16::ONE;
            }
        }
        px
    };
    let mut frames = vec![];
    for f in 0..3u64 {
        let p = dir.join("in").join(frame_name(100 + f));
        // Frame 1 is junk: a flash the fix must not look at.
        let px = if f == 1 { vec![f16::from_f32(9.0); w * hgt * 4] } else { shot(4.0 * f as f32) };
        engine::bakes::write_rgba_f16_exr_tagged(&p, w, hgt, w * 4, &px, None).unwrap();
        frames.push(p);
    }
    let fix = FrameFixParams::new(0, 101);
    let job = Job { weights: weights.into(), plan: fix.plan(100, 3), divisor: 1, frames, first: 100, out: dir.join("out") };
    assert_eq!(job.plan[1], (0, 2, 0.5));
    run(ctx.device.clone(), ctx.queue.clone(), &job, &mut |_, _| {}, &std::sync::atomic::AtomicBool::new(false)).unwrap();
    assert!(engine::bakes::is_baked(&job.out));
    use engine::decode::ExrDecoder as _;
    let dec = engine::decode::AnyDecoder::default();
    let read = |p: &Path| {
        let info = dec.read_header(p).unwrap();
        let mut buf = vec![f16::ZERO; info.row_stride_f16() * hgt];
        dec.decode_into(p, info, info.row_stride_f16(), &mut buf).unwrap();
        buf
    };
    let mid = read(&job.out.join(frame_name(101)));
    let (want, a, b) = (shot(4.0), shot(0.0), shot(8.0));
    let (mut made, mut blend, mut n) = (0.0f32, 0.0f32, 0.0f32);
    for y in 16..hgt - 16 {
        for x in 16..w - 16 {
            let i = (y * w + x) * 4;
            made += (mid[i].to_f32() - want[i].to_f32()).abs();
            blend += (0.5 * (a[i].to_f32() + b[i].to_f32()) - want[i].to_f32()).abs();
            n += 1.0;
        }
    }
    let (made, blend) = (made / n, blend / n);
    eprintln!("mean error of the fixed frame {made:.4}, of a cross-fade {blend:.4}");
    assert!(made < blend * 0.35, "the fix is no better than a cross-fade ({made} vs {blend})");
    // Good frames are the input byte for byte.
    assert_eq!(std::fs::read(&job.frames[2]).unwrap(), std::fs::read(job.out.join(frame_name(102))).unwrap());
    let _ = std::fs::remove_dir_all(&dir);
}

/// A stage (a Blur) above a blending time node is sampled on BOTH branches'
/// keys, so the plan must hold it on both: walking only the plain key left
/// the two samples bound to nothing.
#[test]
fn a_stage_above_the_blend_runs_on_both_branches() {
    let mut g = Graph::default();
    g.nodes.insert(1, NodeKind::read(0, ""));
    g.nodes.insert(3, NodeKind::Blur(engine::graph::BlurParams { size: 4.0, matte: false }));
    g.nodes.insert(2, NodeKind::FrameFix(FrameFixParams::new(0, 1012)));
    g.edges.insert((3, 0), 1);
    g.edges.insert((2, 0), 3);
    g.view = Some(2);
    let plan = engine::codegen::generate(&g);
    let keys: Vec<_> = plan.blurs.iter().filter(|b| b.0 .0 == 3).map(|b| b.0 .1).collect();
    let want = [0u64, 1].map(|ph| engine::graph::time_key_push_phase(engine::graph::TIME_KEY_ROOT, 2, ph));
    assert_eq!(keys.len(), 2, "{keys:?}");
    assert!(want.iter().all(|k| keys.contains(k)), "{keys:?} vs {want:?}");
}
