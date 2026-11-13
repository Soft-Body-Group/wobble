//! `--test-track`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    /// --test-track driver: point Read A at the orbiting-discs plate, drop a
    /// Tracker on the first disc, run the bake, and expect a key per frame
    /// that actually FOLLOWS the disc (a tracker that sticks still would key
    /// exactly the same count).
    pub(crate) fn drive_test_track(&mut self) {
        let Some((stage, mut frames, tid)) = self.test_track else { return };
        // Per-stage frame count, for short settles only; the timeout is
        // the wall clock since the last stage change or bake progress.
        frames += 1;
        if self.test_track_clock.elapsed().as_secs_f32() > 60.0 {
            println!("TRACK TEST FAIL: timeout at stage {stage}");
            std::process::exit(1);
        }
        let mut next = stage;
        // The discs plate: generated, not committed (see CLAUDE.md).
        let fg = std::path::Path::new("data/2k_zip_fg");
        use engine::planar::{h_apply, h_inv, h_mul, synth};
        const PW: usize = 960;
        const PH: usize = 540;
        const PN: usize = 48;
        // A point on the plane, away from the region's corners.
        const TRACK_PT: [f32; 2] = [0.42, 0.47];
        const PLACE: u64 = 1001;
        let planar_dir = std::path::Path::new("data/planar_synth");
        match stage {
            0 if self.playback.is_some() && self.displayed_total >= 3 => {
                if !fg.is_dir() {
                    println!(
                        "TRACK TEST FAIL: {} missing — generate it first:\n  \
                         target/release/gen_exr data/2k_zip_fg 2048 1080 24 zip fg",
                        fg.display()
                    );
                    std::process::exit(1);
                }
                self.load_read(0, &fg.to_string_lossy());
                next = 1;
            }
            1 if frames > 20 => {
                // A track runs from the frame ON SCREEN, so park the
                // playhead on 0 for the reference pass: at t=0 the first
                // disc sits at (0.8, 0.5) and orbits from there.
                self.playing = false;
                self.seek_to(0);
                next = 2;
            }
            // Seed only once frame 0 is the one actually displayed — the
            // pattern is authored against those pixels, and so is the bake.
            2 if self.eval_tick() == 0 => {
                let tk = self.egraph.add(
                    NodeKind::Tracker(engine::graph::TrackerParams {
                        center: [0.8, 0.5],
                        size: [0.04, 0.04],
                        search: 0.09,
                        threshold: 0.5,
                        invert: false,
                        track: [0.8, 0.5],
                        // The discs are a BLOB plate: soft domes whose
                        // speckle changes every frame, ~100 px a frame on a
                        // curve, crossing each other. The pattern pass runs
                        // on the planar plate afterwards (stage 10).
                        mode: engine::graph::TrackMode::Blob,
                        ..Default::default()
                    }),
                    egui::pos2(300.0, 0.0),
                );
                self.egraph.connect(tk, 0, 0);
                self.egraph.view = Some(tk);
                let Some(NodeKind::Tracker(p)) =
                    self.egraph.node(tk).map(|n| n.kind.clone())
                else {
                    return;
                };
                self.start_track(tk, p);
                if self.track_job.is_none() {
                    println!(
                        "TRACK TEST FAIL: track did not start ({})",
                        self.render_status.as_deref().unwrap_or("no reason given")
                    );
                    std::process::exit(1);
                }
                self.test_track_clock = web_time::Instant::now();
                self.test_track = Some((3, 0, tk));
                return;
            }
            // Waiting on the bake: any progress rearms the clock, so only a
            // track that has genuinely stalled can time this out.
            3 if self.track_job.is_some() => {
                if let Some((_, j)) = &self.track_job {
                    if j.done > self.test_track_progress {
                        self.test_track_progress = j.done;
                        self.test_track_clock = web_time::Instant::now();
                    }
                }
            }
            3 => {
                let Some(n) = self.egraph.node(tid) else { return };
                let (Some(cx), Some(cy)) = (
                    n.anim.get(&anim::Channel::TrackX),
                    n.anim.get(&anim::Channel::TrackY),
                ) else {
                    println!("TRACK TEST FAIL: no track channels were keyed");
                    std::process::exit(1);
                };
                let want = self.playback.as_ref().map(|p| p.player.timeline_len()).unwrap_or(0);
                if cx.keys.len() as u64 != want || cx.keys.first().map(|k| k.frame) != Some(0.0) {
                    println!(
                        "TRACK TEST FAIL: keyed {} frames from {:?}, wanted {want} from 0",
                        cx.keys.len(),
                        cx.keys.first().map(|k| k.frame),
                    );
                    std::process::exit(1);
                }
                // It must MOVE — a stuck tracker keys the same count.
                let path: f32 = cx
                    .keys
                    .windows(2)
                    .zip(cy.keys.windows(2))
                    .map(|(a, b)| {
                        let dx = a[1].value - a[0].value;
                        let dy = b[1].value - b[0].value;
                        (dx * dx + dy * dy).sqrt()
                    })
                    .sum();
                // ...and stay ON the disc: its orbit is a circle of radius
                // 0.3 about the centre, so every key should sit near it.
                let off_orbit = cx
                    .keys
                    .iter()
                    .zip(cy.keys.iter())
                    .filter(|(a, b)| {
                        let (dx, dy) = (a.value - 0.5, b.value - 0.5);
                        ((dx * dx + dy * dy).sqrt() - 0.3).abs() > 0.12
                    })
                    .count();
                if path <= 0.5 || off_orbit > 0 {
                    println!(
                        "TRACK TEST FAIL: path {path:.3} (wanted > 0.5), {off_orbit} keys off the disc's orbit"
                    );
                    std::process::exit(1);
                }
                // Keep the reference track, then run a SECOND one from a
                // playhead parked mid-sequence: it must key that frame
                // onward (never frame 0 again) and, seeded on the same
                // point the reference reached there, retrace it exactly.
                self.test_track_ref = cx
                    .keys
                    .iter()
                    .zip(cy.keys.iter())
                    .map(|(a, b)| (a.frame as u64, a.value, b.value))
                    .collect();
                let from = want / 3;
                let Some(&(_, sx, sy)) =
                    self.test_track_ref.iter().find(|(f, _, _)| *f == from)
                else {
                    println!("TRACK TEST FAIL: reference track has no key at {from}");
                    std::process::exit(1);
                };
                self.test_track_seed = [sx, sy];
                self.seek_to(from);
                next = 4;
            }
            4 if self.eval_tick() == self.test_track_ref.len() as u64 / 3 => {
                let Some(NodeKind::Tracker(mut p)) =
                    self.egraph.node(tid).map(|n| n.kind.clone())
                else {
                    return;
                };
                p.center = self.test_track_seed;
                p.track = self.test_track_seed;
                self.test_track_progress = 0;
                self.start_track(tid, p);
                self.test_track_clock = web_time::Instant::now();
                self.test_track = Some((5, 0, tid));
                return;
            }
            5 if self.track_job.is_some() => {
                if let Some((_, j)) = &self.track_job {
                    if j.done > self.test_track_progress {
                        self.test_track_progress = j.done;
                        self.test_track_clock = web_time::Instant::now();
                    }
                }
            }
            5 => {
                let len = self.playback.as_ref().map(|p| p.player.timeline_len()).unwrap_or(0);
                let from = len / 3;
                let Some(n) = self.egraph.node(tid) else { return };
                let (Some(cx), Some(cy)) = (
                    n.anim.get(&anim::Channel::TrackX),
                    n.anim.get(&anim::Channel::TrackY),
                ) else {
                    println!("TRACK TEST FAIL: the second track keyed nothing");
                    std::process::exit(1);
                };
                // Keyed from the playhead the bake was launched at, to the
                // end — and NOT one frame before it.
                let want = len - from;
                if cx.keys.len() as u64 != want || cx.keys.first().map(|k| k.frame) != Some(from as f64) {
                    println!(
                        "TRACK TEST FAIL: second track keyed {} frames from {:?}, \
                         wanted {want} from {from}",
                        cx.keys.len(),
                        cx.keys.first().map(|k| k.frame),
                    );
                    std::process::exit(1);
                }
                // Same seed, same frames: the same feature. Not bit-equal —
                // the reference reached `from` with its window centred on
                // `from - 1`, so the two centroids differ slightly — but
                // far inside the disc (radius 0.16) it is following.
                let drift = cx
                    .keys
                    .iter()
                    .zip(cy.keys.iter())
                    .filter_map(|(a, b)| {
                        self.test_track_ref
                            .iter()
                            .find(|(f, _, _)| *f as f64 == a.frame)
                            .map(|(_, rx, ry)| (a.value - rx).abs().max((b.value - ry).abs()))
                    })
                    .fold(0.0f32, f32::max);
                if drift > 0.05 {
                    println!(
                        "TRACK TEST FAIL: the mid-sequence track drifted {drift:.5} from \
                         the reference over the same frames — a different feature"
                    );
                    std::process::exit(1);
                }
                log::info!(
                    "track: blob keyed {want} frames from the playhead ({from}), \
                     retracing the reference (drift {drift:.4})"
                );
                next = 10;
            }
            // PATTERN mode on the planar suite's plate (the same directory
            // and parameters, so either suite may write it): a textured
            // plane under a known homography, so every key has a truth.
            10 => {
                if !planar_dir.join(format!("frame.{:04}.png", PN - 1)).exists() {
                    if let Err(e) = synth::write_sequence(planar_dir, PW, PH, PN, 0.0, false, 0.01) {
                        println!("TRACK TEST FAIL: writing {}: {e:#}", planar_dir.display());
                        std::process::exit(1);
                    }
                }
                self.load_read(0, &planar_dir.to_string_lossy());
                next = 11;
            }
            11 if frames > 20 => {
                // PLACED at 1001, as a real plate is: the track is seeded
                // from the playhead's TICK, which is not an index into the
                // frame list (using it as one tracked a single frame).
                self.place_read(0, PLACE as i64);
                self.playing = false;
                self.playback_dirty = true;
                self.seek_to(PLACE);
                next = 12;
            }
            12 if self.eval_tick() == PLACE
                && !self.force_display
                && self.loaded.get(&0).map(|l| l.seq.info.width) == Some(PW as u32) =>
            {
                let Some(NodeKind::Tracker(mut p)) =
                    self.egraph.node(tid).map(|n| n.kind.clone())
                else {
                    return;
                };
                p.mode = engine::graph::TrackMode::Pattern;
                p.center = TRACK_PT;
                // A 40 px square pattern, normalised per axis.
                p.size = [20.0 / PW as f32, 20.0 / PH as f32];
                p.search = 0.05;
                p.track = p.center;
                self.test_track_progress = 0;
                self.start_track(tid, p);
                if self.track_job.is_none() {
                    println!("TRACK TEST FAIL: the pattern track did not start");
                    std::process::exit(1);
                }
                next = 13;
            }
            13 if self.track_job.is_some() => {
                if let Some((_, j)) = &self.track_job {
                    if j.done > self.test_track_progress {
                        self.test_track_progress = j.done;
                        self.test_track_clock = web_time::Instant::now();
                    }
                    if j.lost > 0 {
                        println!("TRACK TEST FAIL: the pattern track lost {} frames of a clean plate", j.lost);
                        std::process::exit(1);
                    }
                }
            }
            13 => {
                let Some(n) = self.egraph.node(tid) else { return };
                let (Some(cx), Some(cy)) = (
                    n.anim.get(&anim::Channel::TrackX),
                    n.anim.get(&anim::Channel::TrackY),
                ) else {
                    println!("TRACK TEST FAIL: the pattern track keyed nothing");
                    std::process::exit(1);
                };
                if cx.keys.len() != PN || cx.keys.first().map(|k| k.frame) != Some(PLACE as f64) {
                    println!(
                        "TRACK TEST FAIL: pattern keyed {} frames from {:?}, wanted {PN} from {PLACE}",
                        cx.keys.len(),
                        cx.keys.first().map(|k| k.frame)
                    );
                    std::process::exit(1);
                }
                // The viewer followed the bake to its last frame.
                let end = PLACE + PN as u64 - 1;
                if self.timeline_pos != end {
                    println!(
                        "TRACK TEST FAIL: the viewer did not follow the track: playhead at {}, wanted {end}",
                        self.timeline_pos
                    );
                    std::process::exit(1);
                }
                // Every key against where the synth put that texel.
                let (px, py) = (TRACK_PT[0] as f64 * PW as f64, TRACK_PT[1] as f64 * PH as f64);
                let mut worst = (0.0f64, 0u64);
                for (a, b) in cx.keys.iter().zip(cy.keys.iter()) {
                    let t = (a.frame - PLACE as f64).round() as usize;
                    let h = h_mul(&synth::path(PW, PH, t), &h_inv(&synth::path(PW, PH, 0)).unwrap());
                    let (x, y) = h_apply(&h, px, py);
                    let e = ((a.value as f64 * PW as f64 - x).powi(2) + (b.value as f64 * PH as f64 - y).powi(2)).sqrt();
                    if e > worst.0 {
                        worst = (e, a.frame as u64);
                    }
                }
                if worst.0 > 0.5 {
                    println!(
                        "TRACK TEST FAIL: pattern track {:.3} px off the truth at frame {}",
                        worst.0, worst.1
                    );
                    std::process::exit(1);
                }
                log::info!(
                    "track: pattern tracks {PN} frames of a turning, growing plane placed at {PLACE} to {:.3} px worst",
                    worst.0
                );
                self.test_track_worst = worst.0;
                // The multi-point pass drives the graph: see
                // `drive_test_track_ui`.
                next = 20;
            }
            _ => {}
        }
        let frames = if next == stage {
            frames
        } else {
            self.test_track_clock = web_time::Instant::now();
            0
        };
        self.test_track = Some((next, frames, tid));
    }
}
