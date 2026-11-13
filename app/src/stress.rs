//! `--test-stress`: the crop-into-Layer workflow under load, MEASURED.
//!
//! One plate (the first positional directory; give it something big) is
//! cut into sections by Crop nodes in reformat mode, and the sections go
//! into a Layer stack as separate media, each slipped and stretched. The
//! suite then plays it cold and warm, scrubs it, edits a slip, a speed and
//! a crop box, and flips proxy both ways — timing every step on the wall
//! clock and counting what the player decoded and uploaded to do it.
//!
//! `WOBBLE_STRESS_SECTIONS` (default 4) is how many sections;
//! `WOBBLE_STRESS_MEDIA=two` cuts the sections out of the two plates in
//! turn (separate input media); `=whole` stacks the two plates whole, no
//! crops — the worst case, a full frame a layer. It prints `STRESS:` lines and one
//! `STRESS TEST PASS`; the floors are deliberately loose — this is the
//! instrument, the numbers are the point.

use crate::editor;
use crate::Gfx;
use engine::graph::{CropMode, LayerItem, LayerParams, NodeKind, SoftCropParams};
use web_time::Instant;

pub struct StressTest {
    stage: u32,
    t0: Instant,
    clock: Instant,
    layer: u64,
    crops: Vec<u64>,
    /// Counters at the start of the stage being measured.
    shown: u64,
    uploaded: u64,
    decoded: u64,
    rigs: u64,
    real: u64,
    real_ms: u64,
    /// Latency samples of the edit being repeated (ms).
    lat: Vec<f32>,
    reps: u32,
    waiting: bool,
    two: bool,
    whole: bool,
    bake: u64,
    /// (tick, live reading) and the same ticks read off the bake.
    live: Vec<(u64, f32)>,
    baked: Vec<f32>,
    last: Option<f32>,
    stale_at: u64,
}

impl StressTest {
    pub fn new() -> Self {
        let now = Instant::now();
        Self {
            stage: 0,
            t0: now,
            clock: now,
            layer: 0,
            crops: Vec::new(),
            shown: 0,
            uploaded: 0,
            decoded: 0,
            rigs: 0,
            real: 0,
            real_ms: 0,
            lat: Vec::new(),
            reps: 0,
            waiting: false,
            two: std::env::var("WOBBLE_STRESS_MEDIA").is_ok_and(|v| v == "two" || v == "whole"),
            whole: std::env::var("WOBBLE_STRESS_MEDIA").is_ok_and(|v| v == "whole"),
            bake: 0,
            live: Vec::new(),
            baked: Vec::new(),
            last: None,
            stale_at: 0,
        }
    }
}

fn real_decodes() -> (u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (engine::player::FULL_DECODES.load(Relaxed), engine::player::FULL_DECODE_MS.load(Relaxed))
}

/// Ticks the bake is held to the live picture at.
const PROBES: [u64; 3] = [2, 9, 15];

fn sections() -> usize {
    std::env::var("WOBBLE_STRESS_SECTIONS").ok().and_then(|v| v.parse().ok()).unwrap_or(4).clamp(1, 16)
}

fn summary(v: &[f32]) -> String {
    if v.is_empty() {
        return "no samples".into();
    }
    let max = v.iter().cloned().fold(0.0f32, f32::max);
    let avg = v.iter().sum::<f32>() / v.len() as f32;
    format!("avg {avg:.0} ms, worst {max:.0} ms over {}", v.len())
}

impl Gfx {
    fn stress_decoded(&self) -> u64 {
        self.playback.as_ref().map(|pb| pb.player.stats.frames_decoded).unwrap_or(0)
    }

    /// The comp's reading at the playhead once it has landed and two
    /// successive probes agree.
    fn stress_stable(&mut self, st: &mut StressTest) -> Option<f32> {
        if self.force_display || self.refresh_display || self.eval_tick() != self.timeline_pos {
            st.last = None;
            return None;
        }
        let (w, h) = self.comp.output_size()?;
        let v = self.probe_output_box(w / 2 - 16, h / 2 - 16, 32, 32, 1)?;
        let same = st.last.is_some_and(|p| (p - v).abs() < 1e-5);
        st.last = Some(v);
        same.then_some(v)
    }

    fn stress_looped(&self, st: &StressTest) -> bool {
        let len = self.playback.as_ref().map(|pb| pb.player.timeline_len()).unwrap_or(1);
        self.displayed_total - st.shown >= len
    }

    fn stress_mark(&mut self, st: &mut StressTest) {
        st.shown = self.displayed_total;
        st.uploaded = self.uploaded_total;
        st.decoded = self.stress_decoded();
        st.rigs = self.rig_builds;
        st.real = real_decodes().0;
        st.real_ms = real_decodes().1;
        st.clock = Instant::now();
    }

    fn stress_report(&self, st: &StressTest, what: &str) {
        let el = st.clock.elapsed().as_secs_f32();
        let shown = self.displayed_total - st.shown;
        println!(
            "STRESS: {what}: {:.1} pictures/s ({shown} in {el:.1}s), {} reader frames uploaded, \
             {} frames delivered, {} files decoded ({:.0} ms each), {} rig rebuilds",
            shown as f32 / el,
            self.uploaded_total - st.uploaded,
            self.stress_decoded() - st.decoded,
            real_decodes().0 - st.real,
            (real_decodes().1 - st.real_ms) as f32 / (real_decodes().0 - st.real).max(1) as f32,
            self.rig_builds - st.rigs,
        );
    }

    /// One repeated edit: apply `edit`, then time how long until a new
    /// picture is on screen. Returns true once `n` samples are in.
    fn stress_edit(
        &mut self,
        st: &mut StressTest,
        n: u32,
        edit: impl FnOnce(&mut Self, &StressTest),
    ) -> bool {
        if st.waiting {
            if self.displayed_total > st.shown {
                st.lat.push(st.clock.elapsed().as_secs_f32() * 1000.0);
                st.waiting = false;
                st.reps += 1;
            } else if st.clock.elapsed().as_secs_f32() > 10.0 {
                st.lat.push(10_000.0);
                st.waiting = false;
                st.reps += 1;
            }
            return st.reps >= n;
        }
        // A beat between edits, so each starts from a settled picture.
        if st.clock.elapsed().as_secs_f32() < 0.4 && st.reps > 0 {
            return false;
        }
        edit(self, st);
        st.shown = self.displayed_total;
        st.clock = Instant::now();
        st.waiting = true;
        false
    }

    pub(crate) fn drive_test_stress(&mut self) {
        let Some(mut st) = self.test_stress.take() else { return };
        if st.t0.elapsed().as_secs_f32() > 240.0 {
            println!("STRESS TEST FAIL: timeout at stage {}", st.stage);
            std::process::exit(1);
        }
        let el = st.clock.elapsed().as_secs_f32();
        match st.stage {
            0 if self.playback.is_some() && self.displayed_total >= 3 => {
                let n = sections();
                let mut layers = Vec::new();
                let id = self.egraph.add(NodeKind::layer(), egui::pos2(520.0, 160.0));
                for i in 0..n {
                    let plate = if st.two { (i % 2) as u64 } else { 0 };
                    let src = if st.whole {
                        plate
                    } else {
                        // Vertical strips of the one plate, as their own formats.
                        let (a, b) = (i as f32 / n as f32, (i + 1) as f32 / n as f32);
                        let c = self.egraph.add(
                            NodeKind::SoftCrop(SoftCropParams {
                                rect: [a, 0.0, b, 1.0],
                                mode: CropMode::Reformat,
                                ..Default::default()
                            }),
                            egui::pos2(320.0, 40.0 + 60.0 * i as f32),
                        );
                        self.egraph.connect(c, 0, plate);
                        st.crops.push(c);
                        c
                    };
                    layers.push(LayerItem {
                        // See-through, so a probe of the comp reads EVERY
                        // layer's frame, not just the top one's.
                        opacity: 0.6,
                        start: i as i64 * 3,
                        speed: [1.0, 0.5, 2.0, 1.0][i % 4],
                        ..Default::default()
                    });
                    let _ = src;
                }
                if let Some(editor::ENode { kind: NodeKind::Layer(p), .. }) = self.egraph.node_mut(id) {
                    *p = LayerParams { layers, outputs: false };
                }
                for i in 0..n {
                    let src = if st.whole { (i % 2) as u64 } else { st.crops[i] };
                    self.egraph.connect(id, i, src);
                }
                // `WOBBLE_STRESS_NVME=0` takes the NVMe tier out, to tell a
                // decode-bound pass from one waiting on cache writes.
                if std::env::var("WOBBLE_STRESS_NVME").is_ok_and(|v| v == "0") {
                    for n in self.egraph.nodes.values_mut() {
                        if let NodeKind::Read { nvme_cache, .. } = &mut n.kind {
                            *nvme_cache = false;
                        }
                    }
                }
                self.egraph.view = Some(id);
                self.playing = false;
                self.fps = 120.0;
                st.layer = id;
                st.stage = 1;
                st.clock = Instant::now();
            }
            1 if el > 2.0 => {
                if let Some(pb) = &self.playback {
                    let seqs = pb.player.sequences();
                    let info = seqs.first().map(|s| s.info);
                    println!(
                        "STRESS: rig: {} sections of {}, {} readers, {:?} per frame, {:.0} MB a reader frame, \
                         timeline {} frames, {} rig rebuilds to get here",
                        sections(),
                        if st.whole { "two plates, whole" } else if st.two { "two plates" } else { "one plate" },
                        seqs.len(),
                        info.map(|i| (i.width, i.height)),
                        seqs.first().map(|s| s.slot_bytes()).unwrap_or(0) as f64 / 1e6,
                        pb.player.timeline_len(),
                        self.rig_builds,
                    );
                }
                self.seek_to(0);
                self.playing = true;
                self.stress_mark(&mut st);
                st.stage = 2;
            }
            // One whole pass of the timeline cold, then a second one warm
            // (a 4 s sample of a cold pass says nothing about the caches).
            2 if self.stress_looped(&st) || el > 40.0 => {
                self.stress_report(&st, "play, cold, one pass");
                self.stress_mark(&mut st);
                st.stage = 3;
            }
            3 if self.stress_looped(&st) || el > 40.0 => {
                self.stress_report(&st, "play, warm, one pass");
                self.playing = false;
                self.seek_to(0);
                self.stress_mark(&mut st);
                st.stage = 4;
            }
            4 if el > 1.0 => {
                self.stress_mark(&mut st);
                st.stage = 5;
            }
            // The hand sweeping the whole timeline at a frame a 60Hz redraw.
            5 => {
                let len = self.playback.as_ref().map(|pb| pb.player.timeline_len()).unwrap_or(1).max(1);
                if el < 3.0 {
                    if std::env::var_os("WOBBLE_STRESS_TRACE").is_some() && (el * 20.0) as u32 != st.reps {
                        st.reps = (el * 20.0) as u32;
                        if let Some(pb) = &self.playback {
                            println!(
                                "STRESS trace {el:.2}s: hand {:?} playhead {} shown {} force {} (pending, inflight, free, complete) {:?} ring {:.1}/{:.1} GB",
                                self.seek_target,
                                self.timeline_pos,
                                self.shown_tick,
                                self.force_display,
                                pb.player.debug_pressure(),
                                pb.ring.created_bytes() as f64 / 1e9,
                                pb.ring.budget_bytes() as f64 / 1e9,
                            );
                        }
                    }
                    // Back and forth, as a hand goes: a sweep that wraps
                    // from the last frame to the first is a teleport no
                    // drag makes (and every frame in flight is then behind
                    // the picture).
                    let span = (len - 1).max(1);
                    let k = (el * 60.0) as u64 % (2 * span);
                    let want = if k <= span { k } else { 2 * span - k };
                    if self.seek_target != Some(want) && self.timeline_pos != want {
                        self.seek_to(want);
                    }
                } else {
                    self.stress_report(&st, "scrub, 60Hz back and forth");
                    st.lat.clear();
                    st.reps = 0;
                    st.stage = 6;
                    self.stress_mark(&mut st);
                }
            }
            // Slip one layer by a frame, six times.
            6 => {
                let (rigs, dec) = (st.rigs, st.decoded);
                if self.stress_edit(&mut st, 6, |s, st| {
                    if let Some(editor::ENode { kind: NodeKind::Layer(p), .. }) = s.egraph.node_mut(st.layer) {
                        if let Some(l) = p.layers.get_mut(0) {
                            l.start += 1;
                        }
                    }
                }) {
                    println!(
                        "STRESS: slip a layer one frame: {} to the new picture, {} decodes, {} rig rebuilds",
                        summary(&st.lat),
                        self.stress_decoded().saturating_sub(dec),
                        self.rig_builds - rigs
                    );
                    st.lat.clear();
                    st.reps = 0;
                    st.stage = 7;
                    self.stress_mark(&mut st);
                }
                st.rigs = rigs;
                st.decoded = dec;
            }
            // Stretch one layer, six times.
            7 => {
                let (rigs, dec) = (st.rigs, st.decoded);
                if self.stress_edit(&mut st, 6, |s, st| {
                    if let Some(editor::ENode { kind: NodeKind::Layer(p), .. }) = s.egraph.node_mut(st.layer) {
                        if let Some(l) = p.layers.get_mut(1) {
                            l.speed += 0.1;
                        }
                    }
                }) {
                    println!(
                        "STRESS: stretch a layer: {} to the new picture, {} decodes, {} rig rebuilds",
                        summary(&st.lat),
                        self.stress_decoded().saturating_sub(dec),
                        self.rig_builds - rigs
                    );
                    st.lat.clear();
                    st.reps = 0;
                    st.stage = 8;
                    self.stress_mark(&mut st);
                }
                st.rigs = rigs;
                st.decoded = dec;
            }
            // Move a crop box: a uniform, so nothing should decode or rebuild.
            8 => {
                if let Some(&c) = st.crops.first() {
                    if let Some(editor::ENode { kind: NodeKind::SoftCrop(p), .. }) = self.egraph.node_mut(c) {
                        p.rect[1] = 0.05;
                    }
                }
                st.stage = 9;
                st.clock = Instant::now();
            }
            9 if el > 1.0 => {
                println!(
                    "STRESS: move a crop box: {} decodes, {} rig rebuilds, {} pictures in the second after",
                    self.stress_decoded().saturating_sub(st.decoded),
                    self.rig_builds - st.rigs,
                    self.displayed_total - st.shown,
                );
                st.lat.clear();
                st.reps = 0;
                st.stage = 10;
                self.stress_mark(&mut st);
            }
            // Proxy on and off, three times each way.
            10 => {
                let (rigs, dec) = (st.rigs, st.decoded);
                let before = st.lat.len();
                let on_now = self.proxy < 1.0;
                if self.stress_edit(&mut st, 6, |s, _| {
                    let on = s.proxy < 1.0;
                    s.proxy = if on { 1.0 } else { 0.5 };
                    s.playback_dirty = true;
                    s.last_engine_graph = None;
                }) {
                    println!(
                        "STRESS: proxy toggles (to full, to proxy alternating): {:?} ms, {} decodes, {} rig rebuilds",
                        st.lat.iter().map(|v| v.round() as u32).collect::<Vec<_>>(),
                        self.stress_decoded().saturating_sub(dec),
                        self.rig_builds - rigs
                    );
                    st.stage = 11;
                    self.playing = true;
                    self.stress_mark(&mut st);
                } else if st.lat.len() > before {
                    let _ = on_now;
                }
                if st.stage == 10 {
                    st.rigs = rigs;
                    st.decoded = dec;
                }
            }
            11 if el > 4.0 => {
                self.stress_report(&st, if self.proxy < 1.0 { "play, in proxy" } else { "play, after proxy round trips" });
                let cache = self.playback.as_ref().map(|pb| pb.player.cache_stats()).unwrap_or_default();
                println!("STRESS: RAM tier at the end: {:?}", cache);
                if self.proxy < 1.0 {
                    self.proxy = 1.0;
                    self.playback_dirty = true;
                    self.last_engine_graph = None;
                }
                self.playing = false;
                self.seek_to(PROBES[0]);
                st.last = None;
                st.stage = 12;
                st.clock = Instant::now();
            }
            // LIVE readings of the comp at a few ticks, to hold the bake to.
            12 if el > 0.3 => {
                if let Some(v) = self.stress_stable(&mut st) {
                    st.live.push((PROBES[st.live.len()], v));
                    match PROBES.get(st.live.len()) {
                        Some(t) => {
                            self.seek_to(*t);
                            st.last = None;
                            st.clock = Instant::now();
                        }
                        None => {
                            // A Bake under the stack, baked now.
                            let slot = self.next_read_slot;
                            self.next_read_slot += 1;
                            let b = self.egraph.add(editor::crop_bake_kind(slot, false), egui::pos2(700.0, 160.0));
                            self.egraph.connect(b, 0, st.layer);
                            self.egraph.view = Some(b);
                            st.bake = b;
                            st.stage = 13;
                            st.clock = Instant::now();
                        }
                    }
                }
            }
            13 if el > 0.5 => {
                self.start_graph_bake(st.bake);
                self.stress_mark(&mut st);
                st.stage = 14;
            }
            14 if self.graph_bake.is_none() && el > 0.5 => {
                let ok = self.loaded.contains_key(&st.bake);
                let frames = self.loaded.get(&st.bake).map(|l| l.seq.frames.len()).unwrap_or(0);
                println!(
                    "STRESS: bake the stack: {frames} frames in {el:.1}s = {:.1} frames/s, {} files decoded{}",
                    frames as f32 / el,
                    real_decodes().0 - st.real,
                    if ok { "" } else { " — NOT LOADED" },
                );
                if !ok {
                    println!("STRESS TEST FAIL: the bake did not load ({:?})", self.load_errors.get(&st.bake));
                    std::process::exit(1);
                }
                self.seek_to(PROBES[0]);
                st.last = None;
                st.stage = 15;
                st.clock = Instant::now();
            }
            // The same ticks off the bake: it must be the live picture.
            15 if el > 0.3 => {
                if let Some(v) = self.stress_stable(&mut st) {
                    st.baked.push(v);
                    match PROBES.get(st.baked.len()) {
                        Some(t) => {
                            self.seek_to(*t);
                            st.last = None;
                            st.clock = Instant::now();
                        }
                        None => {
                            let worst = st
                                .live
                                .iter()
                                .zip(&st.baked)
                                .map(|((_, a), b)| (a - b).abs())
                                .fold(0.0f32, f32::max);
                            println!(
                                "STRESS: bake against live at ticks {:?}: live {:?}, baked {:?}, worst difference {worst:.5}",
                                PROBES,
                                st.live.iter().map(|(_, v)| *v).collect::<Vec<_>>(),
                                st.baked,
                            );
                            if worst > 2e-3 {
                                println!("STRESS TEST FAIL: the bake is not the live picture");
                                std::process::exit(1);
                            }
                            self.seek_to(0);
                            self.playing = true;
                            self.stress_mark(&mut st);
                            st.stage = 16;
                        }
                    }
                }
            }
            16 if self.stress_looped(&st) || el > 40.0 => {
                self.stress_report(&st, "play the bake, one pass");
                self.stress_mark(&mut st);
                st.stage = 17;
            }
            17 if self.stress_looped(&st) || el > 40.0 => {
                self.stress_report(&st, "play the bake, second pass");
                self.playing = false;
                if std::env::var_os("WOBBLE_STRESS_STALE").is_none() {
                    println!("STRESS TEST PASS: ran to the end in {:.0}s", st.t0.elapsed().as_secs_f32());
                    std::process::exit(0);
                }
                // STALE FILES: back on the live stack, paused.
                self.egraph.view = Some(st.layer);
                self.seek_to(PROBES[1]);
                st.last = None;
                st.stage = 18;
                st.clock = Instant::now();
            }
            // Write another frame's bytes over the file under the picture
            // (`WOBBLE_STRESS_STALE`: the plates must be a scratch copy).
            18 if el > 0.5 => {
                if let Some(v) = self.stress_stable(&mut st) {
                    let Some(pb) = &self.playback else { return };
                    let seqs = pb.player.sequences();
                    let r = (0..seqs.len()).find(|r| pb.player.is_active(*r)).unwrap_or(0);
                    let i = pb.player.source_index(r, PROBES[1]).unwrap_or(0) as usize;
                    let (dst, src) = (seqs[r].frames[i].clone(), seqs[r].frames[(i + 11) % seqs[r].frames.len()].clone());
                    if let Err(e) = std::fs::copy(&src, &dst) {
                        println!("STRESS TEST FAIL: could not write over {}: {e}", dst.display());
                        std::process::exit(1);
                    }
                    st.live = vec![(PROBES[1], v)];
                    st.stale_at = self.stale_reloads;
                    st.last = None;
                    st.stage = 19;
                    st.clock = Instant::now();
                }
            }
            19 => {
                let noticed = self.stale_reloads > st.stale_at;
                let now = if noticed && el > 0.3 { self.stress_stable(&mut st) } else { None };
                if let Some(v) = now {
                    let was = st.live[0].1;
                    println!(
                        "STRESS: a plate written over on disk: noticed and re-shown in {:.0} ms ({was:.4} -> {v:.4})",
                        el * 1000.0
                    );
                    if (v - was).abs() < 1e-3 {
                        println!("STRESS TEST FAIL: the picture did not change with the file");
                        std::process::exit(1);
                    }
                    println!("STRESS TEST PASS: ran to the end in {:.0}s", st.t0.elapsed().as_secs_f32());
                    std::process::exit(0);
                } else if el > 15.0 {
                    println!(
                        "STRESS TEST FAIL: a plate written over on disk was {} after 15 s",
                        if noticed { "noticed but the picture never settled" } else { "never noticed" }
                    );
                    std::process::exit(1);
                }
            }
            _ => {}
        }
        self.test_stress = Some(st);
    }
}
