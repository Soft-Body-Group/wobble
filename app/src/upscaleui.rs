//! The Upscale node's app half: rendering its input to disk, upscaling on a
//! GPU worker, and adopting what it wrote — the Flow node's runner
//! (`crate::flowui`) with its own job, sharing its types. The node itself
//! is `engine::nodes::upscale`; the network `engine::upscale` / `engine::nn`;
//! the panel and glyph `crate::nodes::upscale`.
//!
//! A run is two phases, both explicit and cancellable:
//!
//! 1. **render** — a `SubtreeBake` of input 0 over the input's frame range
//!    (handles included), at the INPUT's format, written as linear Rec.709
//!    EXRs to `<dir>/Input`;
//! 2. **upscale** — `engine::upscale::run` on a thread, writing
//!    `<dir>/Upscaled` (same names) and its done-marker; the node adopts
//!    that directory as its reader when it lands.
//!
//! `<dir>` is keyed on everything that shapes the pixels — the input's
//! subtree key, the recipe (model, scale), the range — so an edit upstream
//! shows as STALE in the panel and a re-run lands in a new directory; the
//! old result keeps playing until then (the GraphBake contract). Nothing
//! runs on an edit: at seconds to a minute a frame, the network only ever
//! runs on the button.

use crate::corridorkey::frame_name;
use crate::flowui::{Action, Jobs, Msg, Phase, Run, View, ViewState};
use crate::Gfx;
use engine::graph::{NodeKind, UpscaleParams};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::Arc;

pub type UpscaleJobs = Jobs;

pub fn params(kind: &NodeKind) -> Option<&UpscaleParams> {
    match kind {
        NodeKind::Upscale(p) => Some(p),
        _ => None,
    }
}

impl Gfx {
    /// Where a run of `node` at the current recipe lands. None while the
    /// input cannot be rendered.
    pub(crate) fn upscale_dir(&self, node: u64) -> Option<PathBuf> {
        let p = self.egraph.node(node).and_then(|n| params(&n.kind))?;
        let input = self.egraph.input_of(node, 0)?;
        let k0 = self.subtree_key_of(input)?;
        let (from, len) = self.upscale_range(input);
        let size = self.node_resolution(input, &mut std::collections::HashSet::new());
        let recipe = serde_json::to_vec(&(p.recipe(), from, len, size)).ok()?;
        let key = engine::diskcache::content_hash(&[&k0.to_le_bytes(), &recipe]);
        let root = match &self.disk_cache {
            Some(c) => c.root().join("upscale"),
            None => engine::paths::temp_dir().join("wobble-upscale"),
        };
        Some(root.join(format!("{key:016x}")))
    }

    /// The frames a run covers: the input's range, handles included.
    fn upscale_range(&self, input: u64) -> (u64, u64) {
        let ((a, b), _) = self.range_of(input);
        (a.saturating_sub(self.handles.0), b.saturating_add(self.handles.1) + 1)
    }

    pub(crate) fn upscale_view(&self, node: u64) -> View {
        let mine = self.jobs.upscale.run.as_ref().filter(|r| r.node == node);
        let state = match mine.map(|r| &r.phase) {
            None => ViewState::Idle,
            Some(Phase::Render) => match &self.graph_bake {
                Some(b) if b.node == node => {
                    ViewState::Rendering(b.next.saturating_sub(b.from), b.len.saturating_sub(b.from))
                }
                _ => ViewState::Rendering(0, 0),
            },
            Some(Phase::Solving { done, total, .. }) => ViewState::Solving(*done, *total),
            Some(Phase::Failed(e)) => ViewState::Failed(e.clone()),
            Some(Phase::Done(s)) => ViewState::Done(s.clone()),
        };
        let want = self.upscale_dir(node).map(|d| d.join("Upscaled"));
        let loaded = self.loaded.get(&node).map(|l| {
            let current = want.as_ref().is_some_and(|d| l.path == d.to_string_lossy());
            (l.seq.frames.len(), current)
        });
        View {
            state,
            has_input: self.egraph.input_of(node, 0).is_some(),
            weights: self
                .egraph
                .node(node)
                .and_then(|n| params(&n.kind))
                .is_some_and(|p| engine::upscale::weights_ready(p.model)),
            loaded,
            on_disk: want.as_ref().is_some_and(|d| engine::bakes::is_baked(d)),
            busy_elsewhere: self.jobs.upscale.run.as_ref().is_some_and(|r| r.node != node && r.active()),
        }
    }

    pub(crate) fn upscale_action(&mut self, node: u64, a: Action) {
        match a {
            Action::Analyse => self.upscale_start(node),
            Action::Cancel => {
                if let Some(r) = self.jobs.upscale.run.take() {
                    match r.phase {
                        Phase::Solving { cancel, .. } => cancel.store(true, Relaxed),
                        Phase::Render => {
                            if self.graph_bake.as_ref().is_some_and(|b| b.node == r.node) {
                                self.graph_bake = None;
                            }
                        }
                        _ => {}
                    }
                }
            }
            Action::Dismiss => {
                if self.jobs.upscale.run.as_ref().is_some_and(|r| r.node == node && !r.active()) {
                    self.jobs.upscale.run = None;
                }
            }
        }
    }

    fn upscale_fail(&mut self, node: u64, e: &str) {
        match self.jobs.upscale.run.as_mut().filter(|r| r.node == node) {
            Some(r) => r.phase = Phase::Failed(e.to_string()),
            None => self.jobs.upscale.run = Some(Run { node, dir: PathBuf::new(), phase: Phase::Failed(e.into()) }),
        }
    }

    /// The `render` click: adopt a result already on disk, else render the
    /// input.
    fn upscale_start(&mut self, node: u64) {
        if self.jobs.upscale.run.as_ref().is_some_and(|r| r.active()) {
            return;
        }
        #[cfg(target_arch = "wasm32")]
        return self.upscale_fail(node, "the upscale needs a worker thread: not in the web build");
        #[cfg_attr(target_arch = "wasm32", allow(unreachable_code))]
        let Some(input) = self.egraph.input_of(node, 0) else {
            return self.upscale_fail(node, "wire a picture into the input");
        };
        let Some(dir) = self.upscale_dir(node) else {
            return self.upscale_fail(node, "the input is not renderable yet");
        };
        let out = dir.join("Upscaled");
        if engine::bakes::is_baked(&out) {
            self.load_read(node, &out.to_string_lossy());
            self.playback_dirty = true;
            self.jobs.upscale.run = Some(Run { node, dir, phase: Phase::Done("loaded the result on disk".into()) });
            return;
        }
        let model = self.egraph.node(node).and_then(|n| params(&n.kind)).map(|p| p.model).unwrap_or_default();
        if let Err(e) = engine::models::ready_path(model.model_id()) {
            return self.upscale_fail(node, &format!("{e}"));
        }
        self.jobs.upscale.run = Some(Run { node, dir: dir.clone(), phase: Phase::Render });
        let size = self.node_resolution(input, &mut std::collections::HashSet::new());
        if let Err(e) = self.render_input_frames(node, input, dir, size, crate::BakeSink::Upscale) {
            self.upscale_fail(node, &e);
        }
    }

    /// A `SubtreeBake` of `input` into `<dir>/Input`, at `size` (the
    /// input's own format: the canvas is the node's, which a loaded result
    /// has made larger) or the canvas. Shared by the Upscale and SeedVR2
    /// runners.
    pub(crate) fn render_input_frames(
        &mut self,
        node: u64,
        input: u64,
        dir: PathBuf,
        size: Option<(u32, u32)>,
        sink: crate::BakeSink,
    ) -> Result<(), String> {
        let sub = dir.join("Input");
        std::fs::create_dir_all(&sub).map_err(|e| format!("{e}"))?;
        let key = self.subtree_key_of(input).ok_or("the input is not renderable yet")?;
        let (w, h) = size.or_else(|| self.comp.output_size()).ok_or("nothing is rendering yet")?;
        let (from, len) = self.upscale_range(input);
        self.playing = false;
        // Resume: skip the frames a previous run left.
        let mut next = from;
        while next < len && sub.join(frame_name(next)).exists() {
            next += 1;
        }
        let pb = self.playback.as_mut().ok_or("load a sequence first")?;
        pb.player.seek(next.min(len.saturating_sub(1)));
        let info = engine::decode::ImageInfo { width: w, height: h };
        let readback = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("input-render"),
            size: info.staging_bytes(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        self.load_errors.remove(&node);
        let bake = crate::SubtreeBake { node, input, key, dir, from, len, next, readback, info, sink };
        if bake.next >= bake.len {
            self.finish_graph_bake(bake);
        } else {
            self.graph_bake = Some(bake);
        }
        Ok(())
    }

    /// The bake loop finished rendering the input: upscale on a worker.
    pub(crate) fn upscale_render_finished(&mut self, node: u64) {
        let Some(r) = self.jobs.upscale.run.as_ref().filter(|r| r.node == node) else { return };
        let dir = r.dir.clone();
        let Some(p) = self.egraph.node(node).and_then(|n| params(&n.kind)).cloned() else {
            return self.upscale_fail(node, "the node is gone");
        };
        let mut frames: Vec<PathBuf> = match std::fs::read_dir(dir.join("Input")) {
            Ok(rd) => rd
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|e| e == "exr"))
                .collect(),
            Err(e) => return self.upscale_fail(node, &format!("reading the rendered frames: {e}")),
        };
        // Named by comp frame, zero-padded: name order is time order.
        frames.sort();
        let job = engine::upscale::Job { model: p.model, frames, out: dir.join("Upscaled"), scale: p.scale.factor() };
        let (tx, rx) = std::sync::mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let (device, queue, stop) = (self.ctx.device.clone(), self.ctx.queue.clone(), cancel.clone());
        let spawned = std::thread::Builder::new().name("upscale".into()).spawn(move || {
            let mut progress = |a, b| {
                let _ = tx.send(Msg::Frame(a, b));
            };
            let r = engine::upscale::run(device, queue, &job, &mut progress, &stop);
            let _ = tx.send(match r {
                Ok(()) => Msg::Done,
                Err(e) => Msg::Failed(format!("{e:#}")),
            });
        });
        if let Err(e) = spawned {
            return self.upscale_fail(node, &format!("could not start the worker: {e}"));
        }
        if let Some(r) = self.jobs.upscale.run.as_mut() {
            r.phase = Phase::Solving { rx, cancel, done: 0, total: 0 };
        }
    }

    /// Every redraw: drain the worker, catch a render that died, adopt a
    /// finished result.
    pub(crate) fn upscale_poll(&mut self) {
        let Some(r) = self.jobs.upscale.run.as_mut() else { return };
        let node = r.node;
        if matches!(r.phase, Phase::Render) && self.graph_bake.is_none() {
            let why = self.load_errors.get(&node).cloned().unwrap_or_else(|| "the input stopped rendering".into());
            r.phase = Phase::Failed(why);
            return;
        }
        let mut end = None;
        if let Phase::Solving { rx, done, total, .. } = &mut r.phase {
            while let Ok(m) = rx.try_recv() {
                match m {
                    Msg::Frame(a, b) => {
                        *done = a;
                        *total = b;
                    }
                    Msg::Done => end = Some(Ok(())),
                    Msg::Failed(e) => end = Some(Err(e)),
                }
            }
        }
        match end {
            Some(Ok(())) => {
                let out = r.dir.join("Upscaled");
                r.phase = Phase::Done("upscaled".into());
                self.load_read(node, &out.to_string_lossy());
                self.playback_dirty = true;
            }
            Some(Err(e)) => r.phase = Phase::Failed(e),
            None => {}
        }
    }

    /// After a load, an undo, a paste: every Upscale whose result for the
    /// current recipe is on disk plays it.
    pub(crate) fn upscale_adopt(&mut self) {
        let ids: Vec<u64> = self
            .egraph
            .nodes
            .iter()
            .filter(|(id, n)| {
                matches!(n.kind, NodeKind::Upscale(_)) && !self.loaded.contains_key(id) && !self.load_errors.contains_key(id)
            })
            .map(|(id, _)| *id)
            .collect();
        for id in ids {
            let Some(dir) = self.upscale_dir(id) else { continue };
            let out = dir.join("Upscaled");
            if engine::bakes::is_baked(&out) {
                self.load_read(id, &out.to_string_lossy());
                self.playback_dirty = true;
            }
        }
    }
}

// ------------------------------------------------------------- the suite

pub struct TestUpscale {
    stage: u32,
    t0: web_time::Instant,
    since: web_time::Instant,
    node: u64,
    small: u64,
    models: Option<PathBuf>,
    before: Option<f32>,
    last: Option<f32>,
    /// The tick both probes read: a render walks the range and leaves the
    /// playhead at its end.
    at: u64,
}

impl Default for TestUpscale {
    fn default() -> Self {
        let now = web_time::Instant::now();
        Self { stage: 0, t0: now, since: now, node: 0, small: 0, models: None, before: None, last: None, at: 0 }
    }
}

/// The plate the suite upscales: the default Read, reformatted this small
/// so a whole range runs through the network in seconds.
const SMALL: (u32, u32) = (128, 68);

impl Gfx {
    /// `--test-upscale`: an Upscale node end to end in the live app, with
    /// the real network (Real-ESRGAN compact, from `data/upscale/`, byte for
    /// byte, served through a private models dir so nothing downloads). A
    /// Reformat makes the plate 128x68; `render` renders it at that format,
    /// upscales every frame and plays the result: the node's format is now
    /// twice the input's (2x), and the picture through it is the plate's
    /// within what a network may move a smooth picture. Switching to 4x is a
    /// new recipe — STALE until re-run, and a re-run renders the input at
    /// ITS format although the canvas is the larger loaded result. Waits are
    /// wall-clock.
    pub(crate) fn drive_test_upscale(&mut self) {
        let Some(mut t) = self.test.upscale.take() else { return };
        let fail = |m: String| -> ! {
            println!("UPSCALE TEST FAIL: {m}");
            std::process::exit(1);
        };
        if t.t0.elapsed().as_secs() > 180 {
            fail(format!("timeout at stage {}", t.stage));
        }
        let next = |t: &mut TestUpscale, s: u32| {
            t.stage = s;
            t.since = web_time::Instant::now();
            t.last = None;
        };
        let stable = |me: &mut Self, t: &mut TestUpscale, x: u32, y: u32, s: u32| -> Option<f32> {
            if me.force_display || me.eval_tick() != me.timeline_pos || t.since.elapsed().as_millis() < 150 {
                return None;
            }
            let v = me.probe_output_box(x, y, s, s, 0)?;
            t.since = web_time::Instant::now();
            match t.last.replace(v) {
                Some(p) if (p - v).abs() < 1e-5 => {
                    t.last = None;
                    Some(v)
                }
                _ => None,
            }
        };
        let fmt = |me: &Self, n: u64| me.node_resolution(n, &mut std::collections::HashSet::new());
        let phase = |me: &Self| match me.jobs.upscale.run.as_ref().map(|r| &r.phase) {
            Some(Phase::Failed(e)) => Err(e.clone()),
            Some(Phase::Done(_)) => Ok(true),
            _ => Ok(false),
        };
        match t.stage {
            0 if self.playback.is_some() && self.displayed_total >= 2 => {
                let id = engine::upscale::Model::Compact.model_id();
                let file = "data/upscale/realesr-general-x4v3.pth";
                if !std::path::Path::new(file).is_file() {
                    fail(format!("{file} missing: the catalog's {id}, byte for byte (`models get {id}` and copy it)"));
                }
                let models = engine::paths::temp_dir().join(format!("wobble-upscale-models-{}", std::process::id()));
                let _ = std::fs::create_dir_all(&models);
                let spec = engine::models::spec(id).unwrap_or_else(|| fail(format!("{id} is not in the catalog")));
                if let Err(e) = std::fs::copy(file, spec.path_in(&models)) {
                    fail(format!("staging the weights: {e}"));
                }
                std::env::set_var("WOBBLE_MODELS_DIR", &models);
                if let Err(e) = engine::models::ready_path(id) {
                    fail(format!("the staged weights are not the catalog's: {e}"));
                }
                t.models = Some(models);
                let read = self
                    .egraph
                    .nodes
                    .iter()
                    .find(|(_, n)| matches!(n.kind, NodeKind::Read { .. }))
                    .map(|(id, _)| *id)
                    .unwrap_or_else(|| fail("no Read in the default graph".into()));
                let small = self.egraph.add(
                    NodeKind::Reformat(engine::graph::ReformatParams {
                        size: SMALL,
                        mode: engine::graph::ResizeMode::Distort,
                        scale: 1.0,
                        center: true,
                    }),
                    egui::pos2(300.0, 100.0),
                );
                self.egraph.connect(small, 0, read);
                let slot = self.next_read_slot;
                self.next_read_slot += 1;
                let u = self.egraph.add(
                    NodeKind::Upscale(UpscaleParams { scale: engine::graph::UpscaleScale::X2, ..UpscaleParams::new(slot) }),
                    egui::pos2(300.0, 220.0),
                );
                self.egraph.connect(u, 0, small);
                self.egraph.view = Some(u);
                self.playing = false;
                t.node = u;
                t.small = small;
                next(&mut t, 1);
            }
            1 => {
                if fmt(self, t.node) != Some(SMALL) {
                    fail(format!("unloaded, the node should pass its input through at {SMALL:?}, not {:?}", fmt(self, t.node)));
                }
                if let Some(v) = stable(self, &mut t, 40, 24, 16) {
                    t.before = Some(v);
                    t.at = self.timeline_pos;
                    self.upscale_action(t.node, Action::Analyse);
                    next(&mut t, 2);
                }
            }
            2 => match phase(self) {
                Err(e) => fail(format!("the upscale failed: {e}")),
                Ok(true) => {
                    if !self.loaded.contains_key(&t.node) {
                        fail("the result did not load into the node's reader".into());
                    }
                    self.seek_to(t.at);
                    next(&mut t, 3);
                }
                Ok(false) => {}
            },
            3 => {
                let want = (SMALL.0 * 2, SMALL.1 * 2);
                if fmt(self, t.node) != Some(want) {
                    fail(format!("loaded at 2x, the node's format is {:?}, not {want:?}", fmt(self, t.node)));
                }
                if let Some(v) = stable(self, &mut t, 80, 48, 32) {
                    let b = t.before.unwrap();
                    if (v - b).abs() > 0.03 + 0.05 * b.abs() {
                        fail(format!("the result reads {v:.4} where the plate read {b:.4}"));
                    }
                    if let Some(NodeKind::Upscale(p)) = self.egraph.node_mut(t.node).map(|n| &mut n.kind) {
                        p.scale = engine::graph::UpscaleScale::X4;
                    }
                    if !matches!(self.upscale_view(t.node).loaded, Some((_, false))) {
                        fail("a new scale should show the result STALE".into());
                    }
                    self.upscale_action(t.node, Action::Dismiss);
                    self.upscale_action(t.node, Action::Analyse);
                    next(&mut t, 4);
                }
            }
            4 => match phase(self) {
                Err(e) => fail(format!("the 4x re-run failed: {e}")),
                Ok(true) => {
                    self.seek_to(t.at);
                    next(&mut t, 5);
                }
                Ok(false) => {}
            },
            5 => {
                let want = (SMALL.0 * 4, SMALL.1 * 4);
                if fmt(self, t.node) != Some(want) {
                    fail(format!("loaded at 4x, the node's format is {:?}, not {want:?}", fmt(self, t.node)));
                }
                if let Some(v) = stable(self, &mut t, 160, 96, 64) {
                    let b = t.before.unwrap();
                    if (v - b).abs() > 0.03 + 0.05 * b.abs() {
                        fail(format!("the 4x result reads {v:.4} where the plate read {b:.4}: the input was not rendered at its own format"));
                    }
                    if let Some(m) = &t.models {
                        let _ = std::fs::remove_dir_all(m);
                    }
                    println!(
                        "UPSCALE TEST PASS: an Upscale rendered its 128x68 input, ran Real-ESRGAN compact over \
                         the range and played the result at 2x the format with the picture intact; 4x was a new \
                         recipe, STALE until re-run, and the re-run rendered the input at its own format under a \
                         larger canvas"
                    );
                    std::process::exit(0);
                }
            }
            _ => {}
        }
        self.test.upscale = Some(t);
    }
}
