//! Headless throughput bench: decode -> staging ring -> copies -> fused pass
//! -> blit (offscreen). No window, no pacing — reports the steady-state cost
//! of every stage.
//!
//! Usage: bench <seqA_dir> <seqB_dir> [--frames N] [--slots N] [--ahead N]
//!              [--parallel-decode] [--label TEXT]

use engine::blit::Blit;
use engine::viewtf::ViewSettings;
use engine::compositor::Compositor;
use engine::decode::{ExrDecoder, ExrsDecoder};
use engine::graph::Graph;
use engine::player::{Player, Sequence};
use engine::staging::{CopyUploader, StagingRing, Uploader};
use engine::timing::GpuTimer;
use engine::GpuContext;
use std::sync::Arc;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let mut args = std::env::args().skip(1);
    let seq_a = std::path::PathBuf::from(args.next().expect("seqA dir"));
    let seq_b = std::path::PathBuf::from(args.next().expect("seqB dir"));
    let mut frames = 200u64;
    let mut slots = 6usize;
    let mut ahead = 3u64;
    let mut parallel_decode = false;
    let mut with_blur = false;
    let mut with_defocus = false;
    let mut with_defocus_slow = false;
    let mut defocus_samples: u32 = 128;
    let mut defocus_radius: f32 = 32.0;
    let mut with_glow = false;
    let with_node: Option<engine::graph::NodeKind> = None;
    let mut label = String::new();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--frames" => frames = args.next().unwrap().parse()?,
            "--slots" => slots = args.next().unwrap().parse()?,
            "--ahead" => ahead = args.next().unwrap().parse()?,
            "--parallel-decode" => parallel_decode = true,
            "--blur" => with_blur = true,
            "--defocus" => with_defocus = true,
            "--defocus-accurate" => with_defocus_slow = true,
            "--defocus-samples" => defocus_samples = args.next().unwrap().parse()?,
            "--defocus-radius" => defocus_radius = args.next().unwrap().parse()?,
            "--glow" => with_glow = true,
            "--label" => label = args.next().unwrap(),
            other => anyhow::bail!("unknown arg {other}"),
        }
    }

    let decoder: Arc<dyn ExrDecoder> =
        Arc::new(ExrsDecoder { single_threaded: !parallel_decode });
    let sa = Arc::new(Sequence::from_dir(&seq_a, decoder.as_ref())?);
    let sb = Arc::new(Sequence::from_dir(&seq_b, decoder.as_ref())?);
    println!(
        "seqA: {} frames {}x{} | seqB: {} frames {}x{}",
        sa.frames.len(),
        sa.info.width,
        sa.info.height,
        sb.frames.len(),
        sb.info.width,
        sb.info.height
    );

    let ctx = GpuContext::new()?;
    let slot_bytes = sa.info.staging_bytes().max(sb.info.staging_bytes());
    let ring = Arc::new(StagingRing::new(&ctx.device, slots, slot_bytes));
    let sequences = vec![sa.clone(), sb.clone()];
    // Cache disabled: the bench measures the decode path honestly.
    let mut player = Player::new(sequences, ring.clone(), decoder, ahead, 0);

    let mut graph = Graph::default_poc();
    if with_blur {
        // grade -> blur -> blend.+ (sigma from BlurParams::default = 10 px)
        graph.nodes.insert(7, engine::graph::NodeKind::Blur(Default::default()));
        graph.edges.insert((7, 0), 4);
        graph.edges.insert((5, 0), 7);
    } else if with_defocus || with_defocus_slow {
        graph.nodes.insert(
            7,
            engine::graph::NodeKind::Defocus(engine::graph::DefocusParams {
                radius: defocus_radius,
                samples: defocus_samples,
                mode: if with_defocus_slow {
                    engine::graph::DefocusMode::Accurate
                } else {
                    engine::graph::DefocusMode::Fast
                },
                ..Default::default()
            }),
        );
        graph.edges.insert((7, 0), 4);
        graph.edges.insert((5, 0), 7);
    } else if let Some(k) = with_node {
        graph.nodes.insert(7, k);
        graph.edges.insert((7, 0), 4);
        graph.edges.insert((5, 0), 7);
    } else if with_glow {
        // Same slot as the blur bench, so the two numbers are comparable.
        graph.nodes.insert(7, engine::graph::NodeKind::Glow(Default::default()));
        graph.edges.insert((7, 0), 4);
        graph.edges.insert((5, 0), 7);
    }
    let mut comp = Compositor::new(ctx.device.clone(), ctx.queue.clone(), 2);
    comp.ensure_reader(0, sa.info);
    comp.ensure_reader(1, sb.info);
    let (ow, oh) = (
        sa.info.width.max(sb.info.width),
        sa.info.height.max(sb.info.height),
    );
    comp.ensure_output(ow, oh);
    assert!(comp.set_graph(&graph), "fused shader failed to compile");
    // A filter stage's uniform buffer is created inside encode_comp, and
    // write_params only reaches stages that already exist — so a bare
    // write_params here left EVERY filter param at zero for the whole run
    // (sigma 0, radius 0, glow size 0). Warm up first, then write.
    {
        let warm = GpuTimer::new(&ctx);
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        comp.encode_comp(&mut enc, &warm);
        ctx.queue.submit([enc.finish()]);
        let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    }
    comp.write_params(&graph);
    println!(
        "fused compile: codegen {:.2}ms  naga {:.2}ms  pipeline {:.2}ms  total {:.2}ms",
        comp.last_compile.codegen_ms,
        comp.last_compile.module_ms,
        comp.last_compile.pipeline_ms,
        comp.last_compile.total_ms
    );

    // Offscreen blit target stands in for the swapchain.
    let target = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("bench-target"),
        size: wgpu::Extent3d { width: ow, height: oh, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());
    let mut blit = Blit::new(&ctx.device, wgpu::TextureFormat::Rgba8Unorm);
    // The bench measures throughput, not look: default view transform.
    let view = ViewSettings::default();
    let mut timer = GpuTimer::new(&ctx);
    println!(
        "timestamps: {} (inside encoders: {})",
        timer.enabled(),
        timer.inside_encoders
    );
    let uploader = CopyUploader;

    // Warmup: let the ring fill before the clock starts.
    let mut displayed = 0u64;
    let mut encode_cpu = engine::timing::RollingAvg::default();
    let mut ahead_sum = 0u64;
    let warmup = 8u64.min(frames / 10).max(1);
    let mut t_start = Instant::now();
    let bench_t0 = Instant::now();

    while displayed < frames {
        player.pump();
        let _ = ctx.device.poll(wgpu::PollType::Poll);
        timer.collect();

        let Some(set) = player.show_display_set() else {
            std::thread::yield_now();
            continue;
        };

        let e0 = Instant::now();
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });
        timer.stamp_upload_begin(&mut encoder);
        for (reader, slot, info) in &set.uploads {
            let tex = comp.reader_texture(*reader).expect("reader texture");
            uploader.encode_upload(
                &mut encoder,
                &ring,
                *slot,
                &tex.texture,
                info.width,
                info.height,
                info.padded_row_bytes(),
            );
        }
        let has_prepass = comp.encode_comp(&mut encoder, &timer);
        blit.encode(
            &ctx.device,
            &ctx.queue,
            &mut encoder,
            comp.output_view().unwrap(),
            &target_view,
            timer.blit_pass_writes(),
            [0.0, 0.0, ow as f32, oh as f32],
            false,
            &view,
        );
        timer.resolve(&mut encoder, true, has_prepass, true);
        ctx.queue.submit([encoder.finish()]);
        timer.after_submit();
        player.frame_submitted();
        encode_cpu.push(e0.elapsed().as_secs_f32() * 1e3);
        ahead_sum += player.frames_ahead() as u64;

        displayed += 1;
        if displayed == warmup {
            t_start = Instant::now();
        }
    }

    // Drain GPU + timing readbacks.
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());
    timer.collect();

    let timed_frames = displayed.saturating_sub(warmup).max(1);
    let wall = t_start.elapsed().as_secs_f32();
    let g = timer.averages();
    let d = &player.stats;
    println!();
    if !label.is_empty() {
        println!("== {label} ==");
    }
    println!(
        "frames: {timed_frames} (+{warmup} warmup) in {:.2}s -> {:.1} fps ({:.2} ms/frame)",
        wall,
        timed_frames as f32 / wall,
        wall * 1e3 / timed_frames as f32
    );
    println!(
        "decode: {:.2} ms/frame avg across {} rayon threads ({} frames decoded, per-file {})",
        d.per_frame_ms.avg(),
        d.workers,
        d.frames_decoded,
        if parallel_decode { "parallel" } else { "single-thread" },
    );
    println!(
        "gpu: upload {:.3} ms | prepass {:.3} ms | fused {:.3} ms | blit {:.3} ms  (averages{})",
        g.upload_ms,
        g.prepass_ms,
        g.fused_ms,
        g.blit_ms,
        if timer.inside_encoders { "" } else { ", upload via sentinel-pass gap" }
    );
    println!(
        "cpu encode+submit: {:.3} ms/frame | decoded-ahead avg: {:.1} | ring slots: {} | stall-polls: {} | total wall: {:.2}s",
        encode_cpu.avg(),
        ahead_sum as f32 / displayed.max(1) as f32,
        ring.slot_count(),
        d.stalls,
        bench_t0.elapsed().as_secs_f32()
    );

    // --- M3: live-rewire cost, measured warm ---
    // Each variant is a different topology (different hash -> full codegen +
    // naga + PSO). This is the "graph edit -> new fused shader" number.
    let mut v1 = Graph::default_poc(); // bypass the grade: transform -> blend.+
    v1.edges.insert((5, 0), 3);
    v1.nodes.remove(&4);
    let mut v2 = Graph::default_poc(); // grade B instead of A
    v2.edges.insert((4, 0), 2);
    v2.edges.insert((5, 1), 4);
    v2.edges.insert((5, 0), 3);
    let mut v3 = Graph::default_poc(); // view read A directly
    v3.view = Some(1);
    println!("\nrewire-to-live-shader (warm):");
    for (name, g) in [("bypass-grade", &v1), ("grade-on-B", &v2), ("viewer-direct", &v3)] {
        assert!(comp.set_graph(g), "variant {name} failed to compile");
        let c = comp.last_compile;
        println!(
            "  {name:>14}: total {:6.2} ms (codegen {:.2} + naga {:.2} + pso {:.2})",
            c.total_ms, c.codegen_ms, c.module_ms, c.pipeline_ms
        );
    }
    // Back to the original topology: should be a pure cache hit.
    assert!(comp.set_graph(&graph));
    println!(
        "  back-to-cached: cache_hit={} ({:.3} ms)",
        comp.last_compile.cache_hit, comp.last_compile.total_ms
    );
    Ok(())
}
