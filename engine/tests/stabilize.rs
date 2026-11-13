//! The Stabilize analysis with the REAL network: a synthetic shot under a
//! known shaky camera (`stabilize::write_shaky`), solved by SEA-RAFT at
//! half size, fitted and chained. Needs `data/searaft/C-T-M.safetensors`
//! (the catalog's file); skips, saying so, without it.

use engine::nodes::stabilize::{StabMode, StabModel, StabPath, StabPathRef, StabilizeParams};
use engine::planar::{h_apply, h_mul};
use std::sync::atomic::AtomicBool;

#[test]
fn a_shaky_shot_locks_off_to_a_fraction_of_a_pixel() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let weights = root.join("data/searaft/C-T-M.safetensors");
    if !weights.exists() {
        eprintln!("SKIP stabilize: no {}", weights.display());
        return;
    }
    let ctx = match engine::GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("SKIP stabilize: no adapter ({e})");
            return;
        }
    };
    let (w, h, n) = (640usize, 360usize, 12usize);
    let dir = std::env::temp_dir().join(format!("wobble-stab-test-{}", std::process::id()));
    let truth = engine::stabilize::write_shaky(&dir, w, h, n).unwrap();
    let frames: Vec<_> = (0..n).map(|t| dir.join(format!("frame.{t:04}.png"))).collect();
    let job = engine::stabilize::Job {
        weights,
        frames,
        mattes: None,
        out: dir.join("Motion"),
        divisor: 2,
        model: StabModel::Similarity,
    };
    let (path, size) = engine::stabilize::run(
        ctx.device.clone(),
        ctx.queue.clone(),
        &job,
        &mut |_, _| {},
        &AtomicBool::new(false),
    )
    .unwrap();
    assert_eq!(size, [w as u32, h as u32]);
    // The path against the truth, judged inside the frame.
    let pts = [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8], [0.5, 0.5]].map(|p| [p[0] * w as f64, p[1] * h as f64]);
    let mut worst = 0.0f64;
    for t in 0..n {
        for p in pts {
            let (a, b) = h_apply(&path[t], p[0], p[1]);
            let (c, d) = h_apply(&truth[t], p[0], p[1]);
            worst = worst.max((a - c).hypot(b - d));
        }
    }
    eprintln!("stabilize: path within {worst:.3} px of the truth over {n} frames");
    assert!(worst < 0.75, "path {worst:.3} px off");
    // Lock-off through the node's maths: the corrected camera is still.
    let p = StabilizeParams {
        mode: StabMode::LockOff,
        path: StabPathRef::new(StabPath::new(0, size, path, 0)),
        ..Default::default()
    };
    let mut still = 0.0f64;
    for t in 0..n {
        let shown = h_mul(&p.correction_at(t as i64).unwrap(), &truth[t]);
        for q in pts {
            let (a, b) = h_apply(&shown, q[0], q[1]);
            still = still.max((a - q[0]).hypot(b - q[1]));
        }
    }
    eprintln!("stabilize: lock-off residual {still:.3} px");
    assert!(still < 0.5, "lock-off leaves {still:.3} px");
    let _ = std::fs::remove_dir_all(&dir);
}
