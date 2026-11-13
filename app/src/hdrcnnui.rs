//! The HDRCNN node's app half: rendering its input to disk, reconstructing
//! on a GPU worker, adopting what it wrote — the Upscale runner's shape
//! (`crate::upscaleui`) with the Flow runner's types. The node is
//! `engine::nodes::hdrcnn`; the network `engine::hdrcnn` / `engine::nn`; the
//! panel and glyph `crate::nodes::hdrcnn`.
//!
//! 1. **render** — a `SubtreeBake` of input 0 over its range (handles
//!    included), at the input's format, linear Rec.709 EXRs in `<dir>/Input`;
//! 2. **reconstruct** — `engine::hdrcnn::run` on a thread, writing
//!    `<dir>/Reconstructed` and its done-marker; the node adopts it.
//!
//! `<dir>` is keyed on the input's subtree key, the recipe (model, clip) and
//! the range: an edit upstream shows STALE, a re-run lands in a new
//! directory. Nothing runs on an edit.

use crate::flowui::{Action, Jobs, Msg, Phase, Run, View, ViewState};
use crate::Gfx;
use engine::graph::{HdrCnnParams, NodeKind};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::Arc;

pub type HdrCnnJobs = Jobs;

/// Where the result lands inside a run's directory.
const OUT: &str = "Reconstructed";

pub fn params(kind: &NodeKind) -> Option<&HdrCnnParams> {
    match kind {
        NodeKind::HdrCnn(p) => Some(p),
        _ => None,
    }
}

impl Gfx {
    /// Where a run of `node` at the current recipe lands.
    pub(crate) fn hdrcnn_dir(&self, node: u64) -> Option<PathBuf> {
        let p = self.egraph.node(node).and_then(|n| params(&n.kind))?;
        let input = self.egraph.input_of(node, 0)?;
        let k0 = self.subtree_key_of(input)?;
        let ((a, b), _) = self.range_of(input);
        let size = self.node_resolution(input, &mut std::collections::HashSet::new());
        let recipe = serde_json::to_vec(&(p.recipe(), a, b, self.handles, size)).ok()?;
        let key = engine::diskcache::content_hash(&[&k0.to_le_bytes(), &recipe]);
        let root = match &self.disk_cache {
            Some(c) => c.root().join("hdrcnn"),
            None => engine::paths::temp_dir().join("wobble-hdrcnn"),
        };
        Some(root.join(format!("{key:016x}")))
    }

    pub(crate) fn hdrcnn_view(&self, node: u64) -> View {
        let mine = self.jobs.hdrcnn.run.as_ref().filter(|r| r.node == node);
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
        let want = self.hdrcnn_dir(node).map(|d| d.join(OUT));
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
                .is_some_and(|p| engine::models::ready_path(p.model.model_id()).is_ok()),
            loaded,
            on_disk: want.as_ref().is_some_and(|d| engine::bakes::is_baked(d)),
            busy_elsewhere: self.jobs.hdrcnn.run.as_ref().is_some_and(|r| r.node != node && r.active()),
        }
    }

    pub(crate) fn hdrcnn_action(&mut self, node: u64, a: Action) {
        match a {
            Action::Analyse => self.hdrcnn_start(node),
            Action::Cancel => {
                if let Some(r) = self.jobs.hdrcnn.run.take() {
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
                if self.jobs.hdrcnn.run.as_ref().is_some_and(|r| r.node == node && !r.active()) {
                    self.jobs.hdrcnn.run = None;
                }
            }
        }
    }

    fn hdrcnn_fail(&mut self, node: u64, e: &str) {
        match self.jobs.hdrcnn.run.as_mut().filter(|r| r.node == node) {
            Some(r) => r.phase = Phase::Failed(e.to_string()),
            None => self.jobs.hdrcnn.run = Some(Run { node, dir: PathBuf::new(), phase: Phase::Failed(e.into()) }),
        }
    }

    /// The `render` click: adopt a result already on disk, else render the
    /// input.
    fn hdrcnn_start(&mut self, node: u64) {
        if self.jobs.hdrcnn.run.as_ref().is_some_and(|r| r.active()) {
            return;
        }
        #[cfg(target_arch = "wasm32")]
        return self.hdrcnn_fail(node, "the reconstruction needs a worker thread: not in the web build");
        #[cfg_attr(target_arch = "wasm32", allow(unreachable_code))]
        let Some(input) = self.egraph.input_of(node, 0) else {
            return self.hdrcnn_fail(node, "wire a picture into the input");
        };
        let Some(dir) = self.hdrcnn_dir(node) else {
            return self.hdrcnn_fail(node, "the input is not renderable yet");
        };
        let out = dir.join(OUT);
        if engine::bakes::is_baked(&out) {
            self.load_read(node, &out.to_string_lossy());
            self.playback_dirty = true;
            self.jobs.hdrcnn.run = Some(Run { node, dir, phase: Phase::Done("loaded the result on disk".into()) });
            return;
        }
        let model = self.egraph.node(node).and_then(|n| params(&n.kind)).map(|p| p.model).unwrap_or_default();
        if let Err(e) = engine::models::ready_path(model.model_id()) {
            return self.hdrcnn_fail(node, &format!("{e}"));
        }
        self.jobs.hdrcnn.run = Some(Run { node, dir: dir.clone(), phase: Phase::Render });
        let size = self.node_resolution(input, &mut std::collections::HashSet::new());
        if let Err(e) = self.render_input_frames(node, input, dir, size, crate::BakeSink::HdrCnn) {
            self.hdrcnn_fail(node, &e);
        }
    }

    /// The bake loop finished rendering the input: reconstruct on a worker.
    pub(crate) fn hdrcnn_render_finished(&mut self, node: u64) {
        let Some(r) = self.jobs.hdrcnn.run.as_ref().filter(|r| r.node == node) else { return };
        let dir = r.dir.clone();
        let Some(p) = self.egraph.node(node).and_then(|n| params(&n.kind)).cloned() else {
            return self.hdrcnn_fail(node, "the node is gone");
        };
        let mut frames: Vec<PathBuf> = match std::fs::read_dir(dir.join("Input")) {
            Ok(rd) => rd
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|e| e == "exr"))
                .collect(),
            Err(e) => return self.hdrcnn_fail(node, &format!("reading the rendered frames: {e}")),
        };
        frames.sort();
        let job = engine::hdrcnn::Job { model: p.model, frames, out: dir.join(OUT), clip: p.clip_arg() };
        let (tx, rx) = std::sync::mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let (device, queue, stop) = (self.ctx.device.clone(), self.ctx.queue.clone(), cancel.clone());
        let spawned = std::thread::Builder::new().name("hdrcnn".into()).spawn(move || {
            let mut progress = |a, b| {
                let _ = tx.send(Msg::Frame(a, b));
            };
            let r = engine::hdrcnn::run(device, queue, &job, &mut progress, &stop);
            let _ = tx.send(match r {
                Ok(()) => Msg::Done,
                Err(e) => Msg::Failed(format!("{e:#}")),
            });
        });
        if let Err(e) = spawned {
            return self.hdrcnn_fail(node, &format!("could not start the worker: {e}"));
        }
        if let Some(r) = self.jobs.hdrcnn.run.as_mut() {
            r.phase = Phase::Solving { rx, cancel, done: 0, total: 0 };
        }
    }

    /// Every redraw: drain the worker, catch a render that died, adopt a
    /// finished result.
    pub(crate) fn hdrcnn_poll(&mut self) {
        let Some(r) = self.jobs.hdrcnn.run.as_mut() else { return };
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
                let out = r.dir.join(OUT);
                r.phase = Phase::Done("reconstructed".into());
                self.load_read(node, &out.to_string_lossy());
                self.playback_dirty = true;
            }
            Some(Err(e)) => r.phase = Phase::Failed(e),
            None => {}
        }
    }

    /// After a load, an undo, a paste: every HDRCNN whose result for the
    /// current recipe is on disk plays it.
    pub(crate) fn hdrcnn_adopt(&mut self) {
        let ids: Vec<u64> = self
            .egraph
            .nodes
            .iter()
            .filter(|(id, n)| {
                matches!(n.kind, NodeKind::HdrCnn(_)) && !self.loaded.contains_key(id) && !self.load_errors.contains_key(id)
            })
            .map(|(id, _)| *id)
            .collect();
        for id in ids {
            let Some(dir) = self.hdrcnn_dir(id) else { continue };
            let out = dir.join(OUT);
            if engine::bakes::is_baked(&out) {
                self.load_read(id, &out.to_string_lossy());
                self.playback_dirty = true;
            }
        }
    }
}

// ------------------------------------------------------------- the suite

pub struct TestHdrCnn {
    stage: u32,
    t0: web_time::Instant,
    since: web_time::Instant,
    node: u64,
    models: Option<PathBuf>,
    plate: Option<f32>,
    last: Option<f32>,
    at: u64,
}

impl Default for TestHdrCnn {
    fn default() -> Self {
        let now = web_time::Instant::now();
        Self { stage: 0, t0: now, since: now, node: 0, models: None, plate: None, last: None, at: 0 }
    }
}

/// The plate the suite reconstructs: the default Read, reformatted small
/// so a whole range runs through the network in seconds.
const SMALL: (u32, u32) = (128, 68);

impl Gfx {
    /// `--test-hdrcnn`: an HDRCNN node end to end with the REAL network
    /// (the regularized weights from `data/hdrcnn/`, byte for byte, served
    /// through a private models dir so nothing downloads). With the clip
    /// far above the plate nothing is near it, and the result IS the plate
    /// (the method's blend leaves it alone); with the clip at half the
    /// probe's level the network's highlights take over — never darker than
    /// the clipped picture, and not the plate; the new clip was a new
    /// recipe, STALE until re-run. Waits are wall-clock.
    pub(crate) fn drive_test_hdrcnn(&mut self) {
        let Some(mut t) = self.test.hdrcnn.take() else { return };
        let fail = |m: String| -> ! {
            println!("HDRCNN TEST FAIL: {m}");
            std::process::exit(1);
        };
        if t.t0.elapsed().as_secs() > 180 {
            fail(format!("timeout at stage {}", t.stage));
        }
        let next = |t: &mut TestHdrCnn, s: u32| {
            t.stage = s;
            t.since = web_time::Instant::now();
            t.last = None;
        };
        let stable = |me: &mut Self, t: &mut TestHdrCnn| -> Option<f32> {
            if me.force_display || me.eval_tick() != me.timeline_pos || t.since.elapsed().as_millis() < 150 {
                return None;
            }
            let v = me.probe_output_box(24, 14, 64, 40, 0)?;
            t.since = web_time::Instant::now();
            match t.last.replace(v) {
                Some(p) if (p - v).abs() < 1e-5 => {
                    t.last = None;
                    Some(v)
                }
                _ => None,
            }
        };
        let phase = |me: &Self| match me.jobs.hdrcnn.run.as_ref().map(|r| &r.phase) {
            Some(Phase::Failed(e)) => Err(e.clone()),
            Some(Phase::Done(_)) => Ok(true),
            _ => Ok(false),
        };
        let set_clip = |me: &mut Self, node: u64, c: f32| {
            if let Some(NodeKind::HdrCnn(p)) = me.egraph.node_mut(node).map(|n| &mut n.kind) {
                p.clip = c;
            }
        };
        match t.stage {
            0 if self.playback.is_some() && self.displayed_total >= 2 => {
                let id = engine::hdrcnn::Model::Regularized.model_id();
                let spec = engine::models::spec(id).unwrap_or_else(|| fail(format!("{id} is not in the catalog")));
                let file = std::path::Path::new("data/hdrcnn").join(spec.file);
                if !file.is_file() {
                    fail(format!("{} missing: the catalog's {id}, byte for byte (`models get {id}` and copy it)", file.display()));
                }
                let models = engine::paths::temp_dir().join(format!("wobble-hdrcnn-models-{}", std::process::id()));
                let _ = std::fs::create_dir_all(&models);
                if let Err(e) = std::fs::copy(&file, spec.path_in(&models)) {
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
                let n = self.egraph.add(
                    NodeKind::HdrCnn(HdrCnnParams { clip: 1000.0, ..HdrCnnParams::new(slot) }),
                    egui::pos2(300.0, 220.0),
                );
                self.egraph.connect(n, 0, small);
                self.egraph.view = Some(n);
                self.playing = false;
                t.node = n;
                next(&mut t, 1);
            }
            1 => {
                if let Some(v) = stable(self, &mut t) {
                    if v.abs() < 1e-3 {
                        fail(format!("the plate reads {v} there: too dark to tell anything"));
                    }
                    t.plate = Some(v);
                    t.at = self.timeline_pos;
                    self.hdrcnn_action(t.node, Action::Analyse);
                    next(&mut t, 2);
                }
            }
            2 | 4 => match phase(self) {
                Err(e) => fail(format!("the run failed: {e}")),
                Ok(true) => {
                    if !self.loaded.contains_key(&t.node) {
                        fail("the result did not load into the node's reader".into());
                    }
                    self.seek_to(t.at);
                    let s = t.stage + 1;
                    next(&mut t, s);
                }
                Ok(false) => {}
            },
            3 => {
                if let Some(v) = stable(self, &mut t) {
                    let p = t.plate.unwrap();
                    // Rendered to half-float Rec.709 and back: that much.
                    if (v - p).abs() > 2e-3 + 2e-3 * p.abs() {
                        fail(format!("nothing near a clip of 1000, the result reads {v:.5}, not the plate's {p:.5}"));
                    }
                    let old = self.hdrcnn_dir(t.node);
                    // Half the probe's level: much of it clipped, not all.
                    let half = 0.5 * p;
                    set_clip(self, t.node, half);
                    if self.hdrcnn_dir(t.node) == old {
                        fail("a new clip did not change the result's directory".into());
                    }
                    if !matches!(self.hdrcnn_view(t.node).loaded, Some((_, false))) {
                        fail("a new clip should show the result STALE".into());
                    }
                    self.hdrcnn_action(t.node, Action::Dismiss);
                    self.hdrcnn_action(t.node, Action::Analyse);
                    next(&mut t, 4);
                }
            }
            5 => {
                if let Some(v) = stable(self, &mut t) {
                    let p = t.plate.unwrap();
                    // The picture clipped at half the probe's level averages
                    // at most that; the network starts from the clip and
                    // builds up, and it must not be the plate itself.
                    if !v.is_finite() || v < 0.45 * p {
                        fail(format!("clipped at {:.4}, the reconstruction reads {v:.4}: darker than the clipped picture", 0.5 * p));
                    }
                    if (v - p).abs() < 1e-2 * p {
                        fail(format!("clipped at {:.4}, the reconstruction reads the plate's own {p:.4}: nothing was reconstructed", 0.5 * p));
                    }
                    if let Some(m) = &t.models {
                        let _ = std::fs::remove_dir_all(m);
                    }
                    println!(
                        "HDRCNN TEST PASS: with the clip far above the plate the reconstruction is the plate \
                         (the method leaving it alone); a new clip was a new recipe, STALE until re-run; \
                         clipped at half the probe's level the real network's highlights took over ({p:.4} -> {v:.4} at the probe)"
                    );
                    std::process::exit(0);
                }
            }
            _ => {}
        }
        self.test.hdrcnn = Some(t);
    }
}
