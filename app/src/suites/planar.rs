//! `--test-planar`: the suite's driver, moved out of main.rs unchanged.
//! What it covers: docs/claude/testing.md.
use crate::*;

impl Gfx {
    /// --test-planar driver: a synthetic textured plane under a known
    /// homography (written to `data/planar_synth` on first run, like the
    /// crypto plate), a PlanarTrack on it, a forward bake scored against
    /// the truth per frame; then the same with the mesh on (a planar shot:
    /// every vertex must stay on the plane); then the region gizmo: clear
    /// the track, drag a corner handle, expect that corner and only that
    /// corner to follow. Waits are WALL CLOCK: a bake is real work.
    pub(crate) fn drive_test_planar(&mut self, raw: &mut egui::RawInput) {
        use engine::planar::{h_apply, h_inv, h_mul, synth};
        let Some((stage, mut frames, mut node)) = self.test.planar else { return };
        frames += 1;
        if self.test.planar_t0.elapsed().as_secs_f32() > 900.0 {
            println!("PLANAR TEST FAIL: timeout at stage {stage}");
            std::process::exit(1);
        }
        const W: usize = 960;
        const HT: usize = 540;
        // WOBBLE_PLANAR_FRAMES=240 is the TIMING mode: a longer shot in its
        // own directory, the accuracy checks off (the synthetic path runs
        // off the picture past ~150 frames), the pass durations printed.
        // It is how "starts quick, gets super slow" was measured.
        let timing: Option<usize> =
            std::env::var("WOBBLE_PLANAR_FRAMES").ok().and_then(|v| v.parse().ok());
        #[allow(non_snake_case)]
        let N: usize = timing.unwrap_or(48);
        let dir_owned = match timing {
            Some(n) => format!("data/planar_synth_{n}"),
            None => "data/planar_synth".to_string(),
        };
        let dir = std::path::Path::new(&dir_owned);
        let fail = |msg: String| -> ! {
            println!("PLANAR TEST FAIL: {msg}");
            std::process::exit(1);
        };
        // Reference → frame t, the truth the synth wrote the frames with.
        let truth = |t: usize| h_mul(&synth::path(W, HT, t), &h_inv(&synth::path(W, HT, 0)).unwrap());
        let mut next = stage;
        match stage {
            0 if self.playback.is_some() && self.displayed_total >= 3 => {
                if !dir.join(format!("frame.{:04}.png", N - 1)).exists() {
                    if let Err(e) = synth::write_sequence(dir, W, HT, N, 0.0, false, 0.01) {
                        fail(format!("writing {}: {e:#}", dir.display()));
                    }
                }
                self.load_read(0, &dir.to_string_lossy());
                next = 1;
            }
            1 if frames > 20 => {
                self.playing = false;
                self.seek_to(0);
                next = 2;
            }
            2 if self.eval_tick() == 0 => {
                let Some(l) = self.loaded.get(&0) else { return };
                if (l.seq.info.width, l.seq.info.height) != (W as u32, HT as u32) {
                    fail(format!("plate is {}x{}", l.seq.info.width, l.seq.info.height));
                }
                let q = synth::region(W, HT);
                let mut p = engine::graph::PlanarTrackParams::default();
                for k in 0..4 {
                    p.corners[k] = [(q[k][0] / W as f64) as f32, (q[k][1] / HT as f64) as f32];
                }
                let id = self.egraph.add(NodeKind::PlanarTrack(p), egui::pos2(300.0, 0.0));
                self.egraph.connect(id, 0, 0);
                self.egraph.view = Some(id);
                self.selected = vec![id];
                self.start_planar(id, false);
                if self.planar_job.is_none() {
                    fail("the track did not start".into());
                }
                node = id;
                next = 3;
            }
            3 if self.planar_job.is_none() && frames > 2 => {
                // The viewer followed the bake to its last frame.
                if self.timeline_pos != (N - 1) as u64 {
                    fail(format!("the viewer did not follow the bake: playhead at {}", self.timeline_pos));
                }
                let Some(NodeKind::PlanarTrack(p)) = self.egraph.node(node).map(|n| &n.kind) else {
                    fail("node gone".into())
                };
                let t = &p.track;
                if t.frames.len() != N || (timing.is_none() && t.lost() > 0) || t.reference != Some(0) {
                    fail(format!(
                        "planar: {} frames tracked (wanted {N}), {} lost, reference {:?}",
                        t.frames.len(),
                        t.lost(),
                        t.reference
                    ));
                }
                println!(
                    "planar: {N} frames tracked in {:.1}s through the app",
                    self.test.planar_t0.elapsed().as_secs_f32()
                );
                let q = synth::region(W, HT);
                let mut worst = 0.0f64;
                for f in &t.frames {
                    let h = truth(f.frame as usize);
                    for k in 0..4 {
                        let (x, y) = h_apply(&h, q[k][0], q[k][1]);
                        let got = f.corners[k];
                        let e = ((got[0] as f64 * W as f64 - x).powi(2) + (got[1] as f64 * HT as f64 - y).powi(2)).sqrt();
                        worst = worst.max(e);
                    }
                }
                if worst > 0.5 && timing.is_none() {
                    fail(format!("planar: worst corner error {worst:.3}px against the truth"));
                }
                log::info!("planar: {N} frames, worst corner error {worst:.3}px");
                self.test.planar_t0 = web_time::Instant::now();
                // Back to the reference frame for the mesh pass (a track is
                // seeded from the frame on screen).
                self.seek_to(0);
                next = 30;
            }
            30 if self.eval_tick() == 0 => {
                // Now with the mesh.
                if let Some(NodeKind::PlanarTrack(p)) = self.egraph.node_mut(node).map(|n| &mut n.kind) {
                    p.mesh = true;
                    p.mesh_cells = 10;
                }
                self.start_planar(node, false);
                next = 4;
            }
            4 if self.planar_job.is_none() && frames > 2 => {
                let Some(NodeKind::PlanarTrack(p)) = self.egraph.node(node).map(|n| &n.kind) else {
                    fail("node gone".into())
                };
                let t = &p.track;
                let nv = t.mesh.verts.len();
                if nv < 20 || t.mesh.tris.is_empty() || t.frames.len() != N {
                    fail(format!("mesh: {nv} verts, {} tris, {} frames", t.mesh.tris.len(), t.frames.len()));
                }
                let mut worst = 0.0f64;
                for f in &t.frames {
                    if f.mesh.len() != nv {
                        fail(format!("mesh: frame {} carries {} verts of {nv}", f.frame, f.mesh.len()));
                    }
                    let h = truth(f.frame as usize);
                    for (v, got) in t.mesh.verts.iter().zip(f.mesh.iter()) {
                        let (x, y) = h_apply(&h, v[0] as f64 * W as f64, v[1] as f64 * HT as f64);
                        let e = ((got[0] as f64 * W as f64 - x).powi(2) + (got[1] as f64 * HT as f64 - y).powi(2)).sqrt();
                        worst = worst.max(e);
                    }
                }
                if worst > 1.0 && timing.is_none() {
                    fail(format!("mesh: worst vertex error {worst:.3}px on a planar shot"));
                }
                println!(
                    "planar: mesh pass of {N} frames ({nv} verts) in {:.1}s through the app",
                    self.test.planar_t0.elapsed().as_secs_f32()
                );
                log::info!("planar: mesh of {nv} verts, worst vertex error {worst:.3}px");
                // Drive a Transform: a new one appears, cabled on its eight
                // pin channels, and evaluated at the last frame its pin is the
                // frame's corners under the truth.
                self.selected = vec![node];
                self.drive_planar(node, planarui::DriveTarget::Transform);
                let Some(&tf) = self.selected.first() else { fail("no transform selected".into()) };
                if !matches!(self.egraph.node(tf).map(|n| &n.kind), Some(NodeKind::Transform(_))) {
                    fail("drive did not make a Transform".into());
                }
                let wired = [
                    anim::Channel::Pin0X,
                    anim::Channel::Pin0Y,
                    anim::Channel::Pin1X,
                    anim::Channel::Pin1Y,
                    anim::Channel::Pin2X,
                    anim::Channel::Pin2Y,
                    anim::Channel::Pin3X,
                    anim::Channel::Pin3Y,
                ]
                .iter()
                .filter(|ch| self.egraph.patch.link_to(editor::PDst::Chan(tf, **ch)).is_some())
                .count();
                if wired != 8 {
                    fail(format!("drive cabled {wired} of 8 pin channels"));
                }
                let last = (N - 1) as u64;
                let Some(NodeKind::Transform(tp)) = self.live_kind(tf, last) else {
                    fail("no live transform".into())
                };
                // The pin is a homography of the whole frame; judge it where
                // the element would sit — the region's corners through it —
                // not at the frame's corners, an extrapolation three times
                // further out where 0.2 px in the region reads as 2.
                let h = truth(N - 1);
                let unit_px: [[f64; 2]; 4] = std::array::from_fn(|k| {
                    [engine::graph::UNIT_QUAD[k][0] as f64 * W as f64, engine::graph::UNIT_QUAD[k][1] as f64 * HT as f64]
                });
                let pin_px: [[f64; 2]; 4] =
                    std::array::from_fn(|k| [tp.pin[k][0] as f64 * W as f64, tp.pin[k][1] as f64 * HT as f64]);
                let Some(hp) = engine::planar::h_from_quad(&unit_px, &pin_px) else { fail("degenerate pin".into()) };
                let q = synth::region(W, HT);
                let mut worst = 0.0f64;
                for c in &q {
                    let (x, y) = h_apply(&h, c[0], c[1]);
                    let (u, v) = h_apply(&hp, c[0], c[1]);
                    worst = worst.max(((u - x).powi(2) + (v - y).powi(2)).sqrt());
                }
                if worst > 0.75 {
                    fail(format!("driven Transform's pin puts the region {worst:.2}px off the truth at frame {last}"));
                }
                log::info!("planar: driven Transform pin within {worst:.3}px at frame {last}");
                // Stabilise: a new CornerPin, the track cabled into its FROM
                // corners. Pushed through its warp, the region on each frame
                // must land back where it sat on the reference frame.
                self.selected = vec![node];
                self.drive_planar(node, planarui::DriveTarget::Stabilize);
                let Some(&cp) = self.selected.first() else { fail("stabilise selected nothing".into()) };
                let wired = planarui::drive_channels(planarui::DriveTarget::Stabilize)
                    .iter()
                    .filter(|ch| self.egraph.patch.link_to(editor::PDst::Chan(cp, **ch)).is_some())
                    .count();
                if wired != 8 {
                    fail(format!("stabilise cabled {wired} of 8 from channels"));
                }
                let mut worst = 0.0f64;
                for f in [0, N / 4, N / 2, 3 * N / 4, N - 1] {
                    let Some(NodeKind::CornerPin(p)) = self.live_kind(cp, f as u64) else {
                        fail("stabilise made no CornerPin".into())
                    };
                    let px = |q: [[f32; 2]; 4]| -> [[f64; 2]; 4] {
                        q.map(|c| [c[0] as f64 * W as f64, c[1] as f64 * HT as f64])
                    };
                    let Some(shown) = engine::planar::h_from_quad(&px(engine::graph::UNIT_QUAD), &px(p.effective_to()))
                    else {
                        fail(format!("degenerate stabilise pin at frame {f}"))
                    };
                    let h = truth(f);
                    for c in &q {
                        let (x, y) = h_apply(&h, c[0], c[1]);
                        let (u, v) = h_apply(&shown, x, y);
                        worst = worst.max((u - c[0]).hypot(v - c[1]));
                    }
                }
                if worst > 0.75 && timing.is_none() {
                    fail(format!("stabilised region wanders {worst:.2}px off its reference corners"));
                }
                self.test.planar_stab = worst;
                log::info!("planar: stabilised region within {worst:.3}px of the reference");
                // Similarity channels: the reference frame is at rest.
                let Some(NodeKind::PlanarTrack(p0)) = self.live_kind(node, 0) else { fail("no live planar".into()) };
                if (p0.drive[0].abs() > 1e-3) || (p0.drive[3] - 1.0).abs() > 1e-3 {
                    fail(format!("drive channels at the reference frame: {:?}", p0.drive));
                }
                self.selected = vec![node];
                // The gizmo: clear the track so the region's handles show.
                if let Some(NodeKind::PlanarTrack(p)) = self.egraph.node_mut(node).map(|n| &mut n.kind) {
                    p.track = Default::default();
                }
                next = 5;
            }
            5 if frames > 5 => {
                let Some(NodeKind::PlanarTrack(p)) = self.egraph.node(node).map(|n| &n.kind) else {
                    fail("node gone".into())
                };
                let c = p.corners[0];
                let pos = self.node_point_on_screen(node, (c[0] * W as f32, c[1] * HT as f32));
                self.test.planar_press = pos;
                raw.events.push(egui::Event::PointerMoved(pos));
                raw.events.push(egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: Default::default(),
                });
                next = 6;
            }
            6 if frames > 2 => {
                raw.events.push(egui::Event::PointerMoved(self.test.planar_press + egui::vec2(30.0, 10.0)));
                next = 7;
            }
            7 if frames > 2 => {
                raw.events.push(egui::Event::PointerMoved(self.test.planar_press + egui::vec2(60.0, 25.0)));
                next = 8;
            }
            8 if frames > 2 => {
                let pos = self.test.planar_press + egui::vec2(60.0, 25.0);
                raw.events.push(egui::Event::PointerMoved(pos));
                raw.events.push(egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers: Default::default(),
                });
                next = 9;
            }
            9 if frames > 3 => {
                let Some(NodeKind::PlanarTrack(p)) = self.egraph.node(node).map(|n| &n.kind) else {
                    fail("node gone".into())
                };
                let q = synth::region(W, HT);
                let was = [(q[0][0] / W as f64) as f32, (q[0][1] / HT as f64) as f32];
                let got = p.corners[0];
                // 60x25 screen px through the viewer's scale, in the node's
                // normalised units (the node is viewed directly, so its
                // pixels are the output's).
                let r = self.image_rect;
                let want = [60.0 / r.width(), 25.0 / r.height()];
                let d = [got[0] - was[0], got[1] - was[1]];
                let ok = |k: usize| (d[k] - want[k]).abs() < 0.25 * want[k].abs() + 0.002;
                if !(ok(0) && ok(1)) {
                    fail(format!("gizmo: corner 0 moved {d:?}, wanted {want:?}"));
                }
                for k in 1..4 {
                    let w = [(q[k][0] / W as f64) as f32, (q[k][1] / HT as f64) as f32];
                    if (p.corners[k][0] - w[0]).abs() > 1e-4 || (p.corners[k][1] - w[1]).abs() > 1e-4 {
                        fail(format!("gizmo: corner {k} moved too: {:?} from {w:?}", p.corners[k]));
                    }
                }
                println!(
                    "PLANAR TEST PASS: {N} frames tracked to the truth within 0.5px (planar) and 1px \
                     (mesh vertices) with the viewer following, a driven Transform's pin lands on the \
                     truth, a stabilise CornerPin holds the region within {:.3}px of its reference \
                     corners, and a corner drag of 60x25 screen px moved corner 0 by {d:?} \
                     (wanted {want:?}) leaving the other three",
                    self.test.planar_stab
                );
                std::process::exit(0);
            }
            _ => {}
        }
        let frames = if next == stage { frames } else { 0 };
        self.test.planar = Some((next, frames, node));
    }
}
