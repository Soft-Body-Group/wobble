//! The camera tracker end to end with the REAL network: the synthetic
//! three-plane dolly (`camtrack::synth::write_shot`) rendered to PNGs,
//! tracked along SEA-RAFT's flow and solved. Needs
//! `data/searaft/C-T-M.safetensors` (the catalog's file); skips, saying
//! so, without it.

use engine::camtrack::{compare, norm, solve, sub, track, Job, SolveOpts};
use std::sync::atomic::AtomicBool;

#[test]
fn a_three_plane_dolly_is_tracked_and_solved_through_the_network() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let weights = root.join("data/searaft/C-T-M.safetensors");
    if !weights.exists() {
        eprintln!("SKIP camtrack: no {}", weights.display());
        return;
    }
    let ctx = match engine::GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("SKIP camtrack: no adapter ({e})");
            return;
        }
    };
    let (w, h, n) = (960usize, 540usize, 36usize);
    let dir = std::env::temp_dir().join(format!("wobble-camtrack-test-{}", std::process::id()));
    let (truth, focal_px) = engine::camtrack::synth::write_shot(&dir, w, h, n).unwrap();
    let frames: Vec<_> = (0..n).map(|t| dir.join(format!("frame.{t:04}.png"))).collect();
    let job = Job { weights, frames, mattes: None, out: dir.join("Tracks"), divisor: 1 };
    let cancel = AtomicBool::new(false);
    let t0 = std::time::Instant::now();
    let (tracks, size) = track(ctx.device.clone(), ctx.queue.clone(), &job, &mut |_, _| {}, &cancel).unwrap();
    eprintln!("{} tracks in {:.1} s", tracks.len(), t0.elapsed().as_secs_f32());
    assert_eq!(size, [w as u32, h as u32]);
    assert!(tracks.len() > 500, "{} tracks", tracks.len());
    let t0 = std::time::Instant::now();
    let opts = SolveOpts { size, focal_px, refine_focal: false, max_keyframes: 100 };
    let s = solve(&tracks, &opts, &mut |_, _| {}, &cancel).unwrap();
    let (wc, wr) = compare(&s.poses, &truth);
    let travel = norm(sub(truth[n - 1].centre(), truth[0].centre()));
    eprintln!(
        "solved in {:.1} s: centres within {:.2}% of the travel, rotation within {wr:.3}°, {} points, {:.2} px, {} unplaced",
        t0.elapsed().as_secs_f32(),
        wc / travel * 100.0,
        s.points.len(),
        s.mean_error(),
        s.unplaced.len()
    );
    assert!(s.unplaced.is_empty(), "{:?}", s.unplaced);
    assert!(wc < 0.03 * travel, "centres {wc} of {travel}");
    assert!(wr < 0.5, "rotation {wr}°");
    // A wrong lens, refined.
    let opts = SolveOpts { size, focal_px: focal_px * 1.1, refine_focal: true, max_keyframes: 100 };
    let s = solve(&tracks, &opts, &mut |_, _| {}, &cancel).unwrap();
    let ferr = (s.focal_px - focal_px).abs() / focal_px;
    eprintln!("refined focal {:.1} for {focal_px:.1}: {:.2}% off", s.focal_px, ferr * 100.0);
    assert!(ferr < 0.03, "focal {} for {focal_px}", s.focal_px);
    let _ = std::fs::remove_dir_all(&dir);
}
