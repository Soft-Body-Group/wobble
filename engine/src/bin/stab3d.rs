//! 3D stabilisation harness (a prototype; see `engine::stab3d`): a shot
//! directory with `Input/*.exr` (linear plates) and `Depth/*.exr` (a Depth
//! node's `Z`, same names) goes in; the camera is tracked and solved
//! (`camtrack`), smoothed, and every frame re-drawn through its smoothed
//! camera as a cloud of splats. Prints how big the holes are.
//!
//! ```text
//! stab3d <shot> [--smooth FRAMES] [--neighbours 2,5] [--focal PX] [--divisor D]
//!        [--frames N] [--sigma PX] [--edge SHARE] [--out DIR] [--eval]
//! ```
//!
//! Writes `<out>/stab/f_NNNNN.png` (sRGB, straight alpha: a hole is
//! transparent) and `<out>/own/…` (the frame's own cloud only, no fill).
//! `--eval` also re-draws frame t from frame t+k's cloud through t's SOLVED
//! camera and reports the error against the real frame: how well depth and
//! camera agree, before any smoothing.

use engine::camtrack::{self, Pose, SolveOpts};
use engine::stab3d::{self, CloudOpts, Frame, Renderer};
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).map(|s| s.as_str())
}

fn srgb(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.003_130_8 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

fn write_png(path: &Path, px: &[f32], w: usize, h: usize) -> anyhow::Result<()> {
    let mut enc = png::Encoder::new(std::io::BufWriter::new(std::fs::File::create(path)?), w as u32, h as u32);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    enc.set_compression(png::Compression::Fast);
    let mut bytes = Vec::with_capacity(w * h * 4);
    for p in px.chunks_exact(4) {
        let a = p[3].clamp(0.0, 1.0);
        let k = if a > 1e-4 { 1.0 / a } else { 0.0 };
        for c in 0..3 {
            bytes.push((srgb(p[c] * k) * 255.0 + 0.5) as u8);
        }
        bytes.push((a * 255.0 + 0.5) as u8);
    }
    enc.write_header()?.write_image_data(&bytes)?;
    Ok(())
}

/// Mean absolute sRGB difference where `a` is covered, and the coverage.
fn diff(a: &[f32], b: &[f32]) -> (f64, f64) {
    let (mut sum, mut n) = (0.0f64, 0usize);
    for (p, q) in a.chunks_exact(4).zip(b.chunks_exact(4)) {
        if p[3] < 0.9 {
            continue;
        }
        for c in 0..3 {
            sum += (srgb(p[c] / p[3]) - srgb(q[c])).abs() as f64;
        }
        n += 1;
    }
    (sum / (3 * n.max(1)) as f64, n as f64 / (a.len() / 4) as f64)
}

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(shot) = args.first().map(PathBuf::from) else {
        eprintln!("usage: stab3d <shot> [--smooth FRAMES] [--neighbours 2,5] [--focal PX] [--divisor D] [--frames N] [--sigma PX] [--edge SHARE] [--out DIR] [--eval]");
        std::process::exit(2);
    };
    let num = |name: &str, d: f64| -> f64 { flag(&args, name).and_then(|s| s.parse().ok()).unwrap_or(d) };
    let smooth = num("--smooth", 12.0);
    let sigma = num("--sigma", 0.45) as f32;
    let edge = num("--edge", 0.08) as f32;
    let divisor = num("--divisor", 1.0) as usize;
    let neighbours: Vec<isize> = flag(&args, "--neighbours")
        .unwrap_or("2,5")
        .split(',')
        .filter_map(|s| s.parse().ok())
        .filter(|d: &isize| *d != 0)
        .collect();
    let out = flag(&args, "--out").map(PathBuf::from).unwrap_or_else(|| shot.join("Out"));
    let list = |sub: &str| -> anyhow::Result<Vec<PathBuf>> {
        let mut v: Vec<PathBuf> = std::fs::read_dir(shot.join(sub))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|e| e == "exr"))
            .collect();
        v.sort();
        Ok(v)
    };
    let mut plates = list("Input")?;
    let mut depths = list("Depth")?;
    let n = (num("--frames", 1e9) as usize).min(plates.len()).min(depths.len());
    plates.truncate(n);
    depths.truncate(n);
    anyhow::ensure!(n >= 3, "need at least three frames with depth, have {n}");

    let ctx = engine::GpuContext::new()?;
    let cancel = AtomicBool::new(false);
    let t0 = std::time::Instant::now();

    // Tracks, kept beside the shot: the slow part, and the same for any
    // smoothing.
    let tdir = shot.join(format!("Tracks-{n}-d{divisor}"));
    let (tracks, size) = if engine::bakes::is_baked(&tdir) {
        camtrack::read_tracks(&tdir)?
    } else {
        let weights = flag(&args, "--weights").map(PathBuf::from).unwrap_or_else(|| "data/searaft/C-T-M.safetensors".into());
        let job = camtrack::Job { weights, frames: plates.clone(), mattes: None, out: tdir.clone(), divisor };
        let r = camtrack::track(ctx.device.clone(), ctx.queue.clone(), &job, &mut |d, t| eprint!("\rtracking {d}/{t}"), &cancel)?;
        eprintln!();
        r
    };
    println!("tracks: {} over {n} frames, {}x{} ({:.0} s)", tracks.len(), size[0], size[1], t0.elapsed().as_secs_f32());

    let focal = flag(&args, "--focal").and_then(|s| s.parse::<f64>().ok());
    let sdir = tdir.join(format!("Solve-{}", focal.map_or("guess".into(), |f| format!("{f:.0}"))));
    let solve = if sdir.join("solve.txt").exists() {
        camtrack::read_solve(&sdir)?
    } else {
        let t1 = std::time::Instant::now();
        let opts = SolveOpts { size, focal_px: focal.unwrap_or(size[0] as f64), refine_focal: true, max_keyframes: 100 };
        let s = match focal {
            Some(_) => camtrack::solve(&tracks, &opts, &mut |d, t| eprint!("\rsolving {d}/{t}"), &cancel)?,
            None => {
                let starts: Vec<f64> = [18.0, 24.0, 35.0, 50.0, 85.0].iter().map(|mm| mm / 36.0 * size[0] as f64).collect();
                camtrack::solve_guess(&tracks, &opts, &starts, &mut |d, t| eprint!("\rguessing {d}/{t}"), &cancel)?
            }
        };
        eprintln!();
        std::fs::create_dir_all(&sdir)?;
        camtrack::write_solve(&sdir, &s)?;
        println!("solved in {:.0} s", t1.elapsed().as_secs_f32());
        s
    };
    let f = solve.focal_px;
    println!(
        "solve: focal {f:.1} px, mean error {:.3} px, {} points, {} unplaced frames",
        solve.mean_error(),
        solve.points.len(),
        solve.unplaced.len()
    );

    let dec = engine::decode::AnyDecoder::default();
    let load = |i: usize| -> anyhow::Result<Frame> { stab3d::read_frame(&dec, &plates[i], &depths[i]) };

    // Depth map units → solve units, per frame.
    let mut scales = Vec::with_capacity(n);
    let mut misfits = Vec::new();
    let mut zmed = Vec::new();
    for i in 0..n {
        let fr = load(i)?;
        let s = stab3d::depth_scale(&solve, &tracks, i, &fr);
        if let Some((_, m)) = s {
            misfits.push(m);
        }
        let mut zs: Vec<f32> = fr.z.iter().step_by(97).copied().collect();
        zs.sort_by(|a, b| a.total_cmp(b));
        zmed.push((zs[zs.len() / 20] as f64, zs[zs.len() / 2] as f64));
        scales.push(s.map(|s| s.0));
    }
    let known: Vec<f64> = scales.iter().flatten().copied().collect();
    anyhow::ensure!(!known.is_empty(), "no frame has enough solved points to scale its depth");
    let mut sorted = known.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let global = sorted[sorted.len() / 2];
    let scales: Vec<f64> = if args.iter().any(|a| a == "--global-scale") {
        vec![global; n]
    } else {
        scales.iter().map(|s| s.unwrap_or(global)).collect()
    };
    misfits.sort_by(|a, b| a.total_cmp(b));
    println!(
        "depth scale (solve units per depth unit): median {global:.4}, range {:.4}..{:.4} ({:+.1}% / {:+.1}%), {} frames unscaled",
        sorted[0],
        sorted[sorted.len() - 1],
        (sorted[0] / global - 1.0) * 100.0,
        (sorted[sorted.len() - 1] / global - 1.0) * 100.0,
        n - known.len()
    );
    if !misfits.is_empty() {
        println!(
            "depth vs solved points, after scaling: median misfit {:.1}% (worst frame {:.1}%)",
            misfits[misfits.len() / 2] * 100.0,
            misfits[misfits.len() - 1] * 100.0
        );
    }
    let near = zmed.iter().zip(&scales).map(|(z, s)| z.0 * s).fold(f64::MAX, f64::min) * 0.2;
    let far = engine::nodes::depth::FAR as f64 * sorted[sorted.len() - 1] * 2.0;
    let scene = zmed.iter().zip(&scales).map(|(z, s)| z.1 * s).sum::<f64>() / n as f64;

    let (w, h) = (size[0] as usize, size[1] as usize);
    let mut rend = Renderer::new(ctx.device.clone(), ctx.queue.clone(), size[0], size[1]);
    let own = CloudOpts { sigma_px: sigma, edge: 0.0, step: 1 };
    let fill = CloudOpts { sigma_px: sigma, edge, step: 1 };

    if args.iter().any(|a| a == "--eval") {
        println!("\nre-drawing frame t from frame t+k, through t's solved camera (mean abs sRGB error, 0..1):");
        for k in [0usize, 1, 4, 8, 16] {
            let (mut e, mut base, mut cov, mut m) = (0.0, 0.0, 0.0, 0);
            for t in (0..n.saturating_sub(k)).step_by(12) {
                let truth = load(t)?;
                let src = load(t + k)?;
                let c = stab3d::cloud(&src, scales[t + k], &solve.poses[t + k], f, &fill);
                let px = rend.render(&c, &solve.poses[t], f, near, far);
                let (d, cv) = diff(&px, &truth.rgba);
                let mut plain = src.rgba.clone();
                plain.chunks_exact_mut(4).for_each(|p| p[3] = 1.0);
                e += d;
                cov += cv;
                base += diff(&plain, &truth.rgba).0;
                m += 1;
            }
            let m = m.max(1) as f64;
            println!("  k={k:>2}: {:.4} (not re-drawn: {:.4}), {:.1}% of the frame covered", e / m, base / m, cov / m * 100.0);
        }
    }

    let smoothed: Vec<Pose> = stab3d::smooth_path(&solve.poses, smooth);
    let (dc, dr) = stab3d::path_delta(&solve.poses, &smoothed);
    println!(
        "\nsmoothing {smooth} frames: the camera moves {:.2}% of the scene's median depth on average (max {:.2}%), turns {:.2}° (max {:.2}°)",
        dc[0] / scene * 100.0,
        dc[1] / scene * 100.0,
        dr[0],
        dr[1]
    );
    // The same rotation as picture travel, for scale.
    println!("  ({:.0} px of picture at the mean turn, {:.0} px at the worst)", dr[0].to_radians().tan() * f, dr[1].to_radians().tan() * f);

    for d in ["stab", "own"] {
        std::fs::create_dir_all(out.join(d))?;
    }
    let t2 = std::time::Instant::now();
    let mut tot = [[0.0f64; 3]; 2];
    let mut worst = [[0.0f64, 0.0, 1.0]; 2];
    for t in 0..n {
        let fr = load(t)?;
        let c = stab3d::cloud(&fr, scales[t], &solve.poses[t], f, &own);
        let mut px = rend.render(&c, &smoothed[t], f, near, far);
        let a = stab3d::holes(&px, w, h);
        write_png(&out.join("own").join(format!("f_{t:05}.png")), &px, w, h)?;
        let mut clouds = Vec::new();
        for d in neighbours.iter().flat_map(|d| [-*d, *d]) {
            let j = t as isize + d;
            if j >= 0 && (j as usize) < n {
                let j = j as usize;
                clouds.push(stab3d::cloud(&load(j)?, scales[j], &solve.poses[j], f, &fill));
            }
        }
        if !clouds.is_empty() {
            let under = rend.render(&stab3d::merge(clouds), &smoothed[t], f, near, far);
            stab3d::over(&mut px, &under);
        }
        let b = stab3d::holes(&px, w, h);
        write_png(&out.join("stab").join(format!("f_{t:05}.png")), &px, w, h)?;
        for (k, r) in [a, b].iter().enumerate() {
            let r = [r.0, r.1, r.2];
            for i in 0..3 {
                tot[k][i] += r[i] / n as f64;
            }
            worst[k] = [worst[k][0].max(r[0]), worst[k][1].max(r[1]), worst[k][2].min(r[2])];
        }
        eprint!("\rdrawing {}/{n}", t + 1);
    }
    eprintln!();
    println!("drew {n} frames in {:.0} s ({:.2} s a frame, reads and PNGs included)", t2.elapsed().as_secs_f32(), t2.elapsed().as_secs_f32() / n as f32);
    for (k, name) in ["own cloud only", "with neighbours"].iter().enumerate() {
        println!(
            "{name}: border holes {:.2}% of the frame (worst {:.2}%), inner holes {:.2}% (worst {:.2}%), crop to clear the border {:.1}% (worst {:.1}%)",
            tot[k][0] * 100.0,
            worst[k][0] * 100.0,
            tot[k][1] * 100.0,
            worst[k][1] * 100.0,
            tot[k][2] * 100.0,
            worst[k][2] * 100.0
        );
    }
    Ok(())
}
