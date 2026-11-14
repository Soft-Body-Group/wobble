//! The Flow node's app half: rendering its input to disk, solving the
//! motion vectors on a worker, and adopting what it wrote. The node itself
//! (params, codegen) is `engine::nodes::flow`; the network
//! `engine::rife`; the panel and glyph `crate::nodes::flow`.
//!
//! A run is two phases, both explicit and cancellable:
//!
//! 1. **render** — a `SubtreeBake` of input 0 over the input's frame range
//!    (handles included), written as linear Rec.709 EXRs to `<dir>/Input`;
//! 2. **solve** — `engine::flowsolve::run` on a thread, over those
//!    frames, writing `<dir>/Vectors` (same names) and its done-marker; the
//!    node adopts that directory as its reader when it lands.
//!
//! `<dir>` is keyed on everything that shapes the vectors — the input's
//! subtree key, the recipe (model, detail, backward), the range — so an
//! edit upstream shows as STALE in the panel and a re-run lands in a new
//! directory; the old vectors keep playing until then (the GraphBake
//! contract). A finished run is adopted on load (`flow_adopt`).
//!
//! Nothing downloads without a click: the weights come through the model
//! catalog's gate at the top of the panel, and `analyse` is disabled until
//! they are there.

use crate::corridorkey::frame_name;
use crate::Gfx;
use engine::graph::{FlowParams, NodeKind};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::mpsc::Receiver;
use std::sync::Arc;

pub enum Msg {
    Frame(usize, usize),
    Done,
    Failed(String),
}

pub enum Phase {
    /// Rendering input 0 (a `SubtreeBake` in `Gfx::graph_bake`).
    Render,
    Solving { rx: Receiver<Msg>, cancel: Arc<AtomicBool>, done: usize, total: usize },
    Failed(String),
    Done(String),
}

pub struct Run {
    pub node: u64,
    pub dir: PathBuf,
    pub phase: Phase,
}

impl Run {
    pub fn active(&self) -> bool {
        matches!(self.phase, Phase::Render | Phase::Solving { .. })
    }
}

#[derive(Default)]
pub struct Jobs {
    pub run: Option<Run>,
}

/// What the panel is told, built by `Gfx::flow_view`.
pub struct View {
    pub state: ViewState,
    pub has_input: bool,
    pub weights: bool,
    /// Frames loaded in the reader, and whether they are THIS recipe's.
    pub loaded: Option<(usize, bool)>,
    pub on_disk: bool,
    pub busy_elsewhere: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ViewState {
    Idle,
    Rendering(u64, u64),
    Solving(usize, usize),
    Failed(String),
    Done(String),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Action {
    Analyse,
    Cancel,
    Dismiss,
}

/// The nodes this runner serves — a Flow's vectors, a Rifetime's frames and
/// a Frame Fix's mended frames run the same render → worker → playback
/// pipeline — and what differs between them: the result key, the folder,
/// the weights.
pub fn is_runner(kind: &NodeKind) -> bool {
    matches!(kind, NodeKind::Flow(_) | NodeKind::Rifetime(_) | NodeKind::FrameFix(_))
}

/// `times`: a curve-mode Rifetime's sampled source positions, which are
/// part of what it makes.
fn recipe_of(kind: &NodeKind, times: Option<&[f64]>) -> Option<Vec<u8>> {
    match kind {
        NodeKind::Flow(p) => serde_json::to_vec(&p.recipe()).ok(),
        NodeKind::Rifetime(p) => serde_json::to_vec(&("retime", p.recipe(), times)).ok(),
        NodeKind::FrameFix(p) => serde_json::to_vec(&("framefix", p.recipe())).ok(),
        _ => None,
    }
}

/// Where the result lands inside a run's directory.
pub fn out_dir_name(kind: &NodeKind) -> &'static str {
    match kind {
        NodeKind::Rifetime(_) => "Frames",
        NodeKind::FrameFix(_) => "Fixed",
        _ => "Vectors",
    }
}

fn model_of(kind: &NodeKind) -> &'static str {
    match kind {
        NodeKind::Flow(p) => p.engine.model(),
        NodeKind::Rifetime(p) => p.models()[0],
        NodeKind::FrameFix(_) => engine::nodes::framefix::MODEL,
        _ => "",
    }
}

impl Gfx {
    /// Where a run of `node` at the current recipe lands. None while the
    /// input cannot be rendered.
    pub(crate) fn flow_dir(&self, node: u64) -> Option<PathBuf> {
        let kind = &self.egraph.node(node)?.kind;
        let input = self.egraph.input_of(node, 0)?;
        let k0 = self.subtree_key_of(input)?;
        let (from, len) = self.flow_range(input);
        let times = self.rifetime_times(node);
        let recipe = serde_json::to_vec(&(recipe_of(kind, times.as_deref())?, from, len)).ok()?;
        let key = engine::diskcache::content_hash(&[&k0.to_le_bytes(), &recipe]);
        let root = match &self.disk_cache {
            Some(c) => c.root().join("flow"),
            None => engine::paths::temp_dir().join("wobble-flow"),
        };
        Some(root.join(format!("{key:016x}")))
    }

    /// A curve-mode Rifetime's source position for every output frame, in
    /// input frames from the first: its source-frame channel sampled at
    /// each tick of the input's range (unkeyed, the authored frame — a
    /// freeze). None for anything else.
    pub(crate) fn rifetime_times(&self, node: u64) -> Option<Vec<f64>> {
        let n = self.egraph.node(node)?;
        let NodeKind::Rifetime(p) = &n.kind else { return None };
        if p.mode != engine::graph::RifetimeMode::Curve {
            return None;
        }
        let (from, len) = self.flow_range(self.egraph.input_of(node, 0)?);
        let curve = n.anim.get(&crate::anim::Channel::RifetimeFrame).filter(|c| !c.keys.is_empty());
        Some(
            (from..len)
                .map(|t| {
                    let f = curve.and_then(|c| c.eval(t)).unwrap_or(p.frame);
                    f as f64 - from as f64
                })
                .collect(),
        )
    }

    /// The frames a run covers: the input's range, handles included.
    fn flow_range(&self, input: u64) -> (u64, u64) {
        let ((a, b), _) = self.range_of(input);
        (a.saturating_sub(self.handles.0), b.saturating_add(self.handles.1) + 1)
    }

    pub(crate) fn flow_view(&self, node: u64) -> View {
        let mine = self.jobs.flow.run.as_ref().filter(|r| r.node == node);
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
        let sub = self.egraph.node(node).map(|n| out_dir_name(&n.kind)).unwrap_or("Vectors");
        let want = self.flow_dir(node).map(|d| d.join(sub));
        let loaded = self.loaded.get(&node).map(|l| {
            let current = want.as_ref().is_some_and(|d| l.path == d.to_string_lossy());
            (l.seq.frames.len(), current)
        });
        View {
            state,
            has_input: self.egraph.input_of(node, 0).is_some(),
            weights: self.egraph.node(node).is_some_and(|n| engine::models::ready_path(model_of(&n.kind)).is_ok()),
            loaded,
            on_disk: want.as_ref().is_some_and(|d| engine::bakes::is_baked(d)),
            busy_elsewhere: self.jobs.flow.run.as_ref().is_some_and(|r| r.node != node && r.active()),
        }
    }

    pub(crate) fn flow_action(&mut self, node: u64, a: Action) {
        match a {
            Action::Analyse => self.flow_start(node),
            Action::Cancel => {
                if let Some(r) = self.jobs.flow.run.take() {
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
                if self.jobs.flow.run.as_ref().is_some_and(|r| r.node == node && !r.active()) {
                    self.jobs.flow.run = None;
                }
            }
        }
    }

    fn flow_fail(&mut self, node: u64, e: &str) {
        match self.jobs.flow.run.as_mut().filter(|r| r.node == node) {
            Some(r) => r.phase = Phase::Failed(e.to_string()),
            None => self.jobs.flow.run = Some(Run { node, dir: PathBuf::new(), phase: Phase::Failed(e.into()) }),
        }
    }

    /// The `analyse` click: adopt vectors already on disk, else render the
    /// input.
    fn flow_start(&mut self, node: u64) {
        if self.jobs.flow.run.as_ref().is_some_and(|r| r.active()) {
            return;
        }
        #[cfg(target_arch = "wasm32")]
        return self.flow_fail(node, "the analysis needs a worker thread: not in the web build");
        #[cfg_attr(target_arch = "wasm32", allow(unreachable_code))]
        let Some(input) = self.egraph.input_of(node, 0) else {
            return self.flow_fail(node, "wire a picture into the input");
        };
        let Some(dir) = self.flow_dir(node) else {
            return self.flow_fail(node, "the input is not renderable yet");
        };
        let Some(kind) = self.egraph.node(node).map(|n| n.kind.clone()) else { return };
        let out = dir.join(out_dir_name(&kind));
        if engine::bakes::is_baked(&out) {
            self.load_read(node, &out.to_string_lossy());
            self.playback_dirty = true;
            self.jobs.flow.run = Some(Run { node, dir, phase: Phase::Done("loaded what is on disk".into()) });
            return;
        }
        let model = model_of(&kind);
        if let Err(e) = engine::models::ready_path(model) {
            return self.flow_fail(node, &format!("{e}"));
        }
        self.jobs.flow.run = Some(Run { node, dir: dir.clone(), phase: Phase::Render });
        if let Err(e) = self.flow_render(node, input, dir) {
            self.flow_fail(node, &e);
        }
    }

    /// A `SubtreeBake` of `input` into `<dir>/Input`.
    fn flow_render(&mut self, node: u64, input: u64, dir: PathBuf) -> Result<(), String> {
        let sub = dir.join("Input");
        std::fs::create_dir_all(&sub).map_err(|e| format!("{e}"))?;
        let key = self.subtree_key_of(input).ok_or("the input is not renderable yet")?;
        let (w, h) = self.comp.output_size().ok_or("nothing is rendering yet")?;
        let (from, len) = self.flow_range(input);
        if len.saturating_sub(from) < 2 {
            return Err("this needs at least two frames; the input's range has one".into());
        }
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
            label: Some("flow-render"),
            size: info.staging_bytes(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        self.load_errors.remove(&node);
        let bake = crate::SubtreeBake {
            node,
            input,
            key,
            dir,
            from,
            len,
            next,
            readback,
            info,
            sink: crate::BakeSink::Flow,
        };
        if bake.next >= bake.len {
            self.finish_graph_bake(bake);
        } else {
            self.graph_bake = Some(bake);
        }
        Ok(())
    }

    /// The bake loop finished rendering the input: solve on a worker.
    pub(crate) fn flow_render_finished(&mut self, node: u64) {
        let Some(r) = self.jobs.flow.run.as_ref().filter(|r| r.node == node) else { return };
        let dir = r.dir.clone();
        let Some(kind) = self.egraph.node(node).map(|n| n.kind.clone()) else {
            return self.flow_fail(node, "the node is gone");
        };
        let weights = match engine::models::ready_path(model_of(&kind)) {
            Ok(w) => w,
            Err(e) => return self.flow_fail(node, &format!("{e}")),
        };
        let mut frames: Vec<PathBuf> = match std::fs::read_dir(dir.join("Input")) {
            Ok(rd) => rd
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|e| e == "exr"))
                .collect(),
            Err(e) => return self.flow_fail(node, &format!("reading the rendered frames: {e}")),
        };
        // Named by comp frame, zero-padded: name order is time order.
        frames.sort();
        // The input's first comp frame: where a retime's output starts.
        let first = self.egraph.input_of(node, 0).map(|i| self.flow_range(i).0).unwrap_or(0);
        let out = dir.join(out_dir_name(&kind));
        enum Work {
            Flow(engine::flowsolve::Job),
            Rifetime(engine::rifetime::Job),
        }
        let work = match &kind {
            NodeKind::Flow(p) => Work::Flow(engine::flowsolve::Job {
                engine: p.engine,
                weights,
                frames,
                out,
                divisor: p.detail.divisor(),
                backward: p.backward,
            }),
            NodeKind::Rifetime(p) => {
                let plan = match self.rifetime_times(node) {
                    Some(times) => engine::nodes::rifetime::plan_times(&times, frames.len()),
                    None => p.plan(frames.len()),
                };
                let plan = engine::rifetime::adjacent(plan);
                Work::Rifetime(engine::rifetime::Job { weights, plan, divisor: p.detail.divisor(), frames, first, out })
            }
            // The same job: good frames linked across, bad ones made from
            // the good frames round them.
            NodeKind::FrameFix(p) => {
                let plan = p.plan(first, frames.len());
                Work::Rifetime(engine::rifetime::Job { weights, plan, divisor: p.detail.divisor(), frames, first, out })
            }
            _ => return self.flow_fail(node, "not a Flow, Rifetime or Frame Fix node"),
        };
        let (tx, rx) = std::sync::mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let (device, queue, stop) = (self.ctx.device.clone(), self.ctx.queue.clone(), cancel.clone());
        let spawned = std::thread::Builder::new().name("flow-solve".into()).spawn(move || {
            let mut progress = |a, b| {
                let _ = tx.send(Msg::Frame(a, b));
            };
            let r = match &work {
                Work::Flow(job) => engine::flowsolve::run(device, queue, job, &mut progress, &stop),
                Work::Rifetime(job) => engine::rifetime::run(device, queue, job, &mut progress, &stop),
            };
            let _ = tx.send(match r {
                Ok(()) => Msg::Done,
                Err(e) => Msg::Failed(format!("{e:#}")),
            });
        });
        if let Err(e) = spawned {
            return self.flow_fail(node, &format!("could not start the worker: {e}"));
        }
        if let Some(r) = self.jobs.flow.run.as_mut() {
            r.phase = Phase::Solving { rx, cancel, done: 0, total: 0 };
        }
    }

    /// Every redraw: drain the worker, catch a render that died, adopt a
    /// finished solve.
    pub(crate) fn flow_poll(&mut self) {
        let Some(r) = self.jobs.flow.run.as_mut() else { return };
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
                let sub = self.egraph.node(node).map(|n| out_dir_name(&n.kind)).unwrap_or("Vectors");
                let Some(r) = self.jobs.flow.run.as_mut() else { return };
                let out = r.dir.join(sub);
                r.phase = Phase::Done("done".into());
                self.load_read(node, &out.to_string_lossy());
                self.playback_dirty = true;
            }
            Some(Err(e)) => r.phase = Phase::Failed(e),
            None => {}
        }
    }

    /// After a load, an undo, a paste: every Flow or Rifetime node whose
    /// result for the current recipe is on disk plays it.
    pub(crate) fn flow_adopt(&mut self) {
        let ids: Vec<u64> = self
            .egraph
            .nodes
            .iter()
            .filter(|(id, n)| {
                is_runner(&n.kind) && !self.loaded.contains_key(id) && !self.load_errors.contains_key(id)
            })
            .map(|(id, _)| *id)
            .collect();
        for id in ids {
            let Some(dir) = self.flow_dir(id) else { continue };
            let Some(sub) = self.egraph.node(id).map(|n| out_dir_name(&n.kind)) else { continue };
            let out = dir.join(sub);
            if engine::bakes::is_baked(&out) {
                self.load_read(id, &out.to_string_lossy());
                self.playback_dirty = true;
            }
        }
    }
}

// ------------------------------------------------------------- the suite

pub struct TestFlow {
    stage: u32,
    t0: web_time::Instant,
    since: web_time::Instant,
    node: u64,
    blur: u64,
    last: Option<f32>,
    fwd: [f32; 2],
    /// Which engine is being judged: SEA-RAFT first, then RIFE.
    engine: engine::graph::FlowEngine,
    sharp: Vec<f32>,
    /// The tick probes wait for.
    at: u64,
    retime: u64,
    /// The retimed shot at 40, 41, 42 and the plate at 20.
    shots: Vec<Vec<f32>>,
    models: Option<PathBuf>,
}

impl Default for TestFlow {
    fn default() -> Self {
        let now = web_time::Instant::now();
        Self { stage: 0, t0: now, since: now, node: 0, blur: 0, last: None, fwd: [0.0; 2], engine: engine::graph::FlowEngine::SeaRaft, sharp: Vec::new(), at: 20, retime: 0, shots: Vec::new(), models: None }
    }
}

impl Gfx {
    /// `--test-flow`: a Flow node end to end in the live app, on the planar
    /// suite's synthetic plate (a textured plane under a KNOWN motion,
    /// written to `data/planar_synth` if missing): `analyse` renders the
    /// plate, solves it with the real network and plays the vectors back;
    /// the forward vector at the frame centre is the true frame-to-frame
    /// motion there — within 0.25 px for SEA-RAFT, then, switched to RIFE
    /// (a new recipe, so a new analysis), within a pixel; then a VectorBlur
    /// over the plate with those vectors smears the texture (a two-frame
    /// shutter against none). Needs both catalog files in `data/`
    /// (`rife/flownet.pkl`, `searaft/C-T-M.safetensors`), byte for byte,
    /// served through a private models dir, so nothing is downloaded.
    /// Waits are wall-clock.
    pub(crate) fn drive_test_flow(&mut self) {
        use engine::planar::{h_apply, h_inv, h_mul, synth};
        let Some(mut t) = self.test.flow.take() else { return };
        let fail = |m: String| -> ! {
            println!("FLOW TEST FAIL: {m}");
            std::process::exit(1);
        };
        if t.t0.elapsed().as_secs() > 420 {
            fail(format!("timeout at stage {} ({})", t.stage, self.flow_state_words()));
        }
        const W: usize = 960;
        const HT: usize = 540;
        const N: usize = 48;
        const AT: u64 = 20;
        let dir = std::path::Path::new("data/planar_synth");
        let next = |t: &mut TestFlow, s: u32| {
            t.stage = s;
            t.since = web_time::Instant::now();
            t.last = None;
        };
        // A reading two probes agree on, 150 ms apart, at the playhead.
        // `px` is the probe's side: 8 for a vector (a mean over a patch),
        // 1 for the picture (a blur moves a pixel, not a patch's mean).
        let stable = |me: &mut Self, t: &mut TestFlow, ch: usize, px: u32| -> Option<f32> {
            if me.force_display || me.eval_tick() != t.at || t.since.elapsed().as_millis() < 150 {
                return None;
            }
            let v = me.probe_output_box(W as u32 / 2 - px / 2, HT as u32 / 2 - px / 2, px, px, ch)?;
            t.since = web_time::Instant::now();
            match t.last.replace(v) {
                Some(p) if (p - v).abs() < 1e-5 => {
                    t.last = None;
                    Some(v)
                }
                _ => None,
            }
        };
        // Nine single pixels around the centre, 6 px apart: a blur moves
        // pixels, and a smooth stretch of texture moves one only a little.
        let pixels = |me: &Self| -> Option<Vec<f32>> {
            let mut out = Vec::new();
            for dy in [-6i32, 0, 6] {
                for dx in [-6i32, 0, 6] {
                    out.push(me.probe_output_box((W as i32 / 2 + dx) as u32, (HT as i32 / 2 + dy) as u32, 1, 1, 0)?);
                }
            }
            Some(out)
        };
        match t.stage {
            0 if self.playback.is_some() && self.displayed_total >= 2 => {
                // Hermetic weights: the catalog's file from data/, in a
                // models dir of our own.
                let models = engine::paths::temp_dir().join(format!("wobble-flow-models-{}", std::process::id()));
                let _ = std::fs::create_dir_all(&models);
                for (e, file) in [
                    (engine::graph::FlowEngine::Rife, "data/rife/flownet.pkl"),
                    (engine::graph::FlowEngine::SeaRaft, "data/searaft/C-T-M.safetensors"),
                ] {
                    let id = e.model();
                    if !std::path::Path::new(file).is_file() {
                        fail(format!("{file} missing: the catalog's {id}, byte for byte (`models get {id}` and copy it)"));
                    }
                    let spec = engine::models::spec(id).unwrap_or_else(|| fail(format!("{id} is not in the catalog")));
                    if let Err(e) = std::fs::copy(file, spec.path_in(&models)) {
                        fail(format!("staging the weights: {e}"));
                    }
                }
                std::env::set_var("WOBBLE_MODELS_DIR", &models);
                for e in engine::graph::FlowEngine::ALL {
                    if let Err(err) = engine::models::ready_path(e.model()) {
                        fail(format!("the staged weights are not the catalog's: {err}"));
                    }
                }
                t.models = Some(models);
                if !dir.join(format!("frame.{:04}.png", N - 1)).exists() {
                    if let Err(e) = synth::write_sequence(dir, W, HT, N, 0.0, false, 0.01) {
                        fail(format!("writing {}: {e:#}", dir.display()));
                    }
                }
                self.load_read(0, &dir.to_string_lossy());
                let slot = self.next_read_slot;
                self.next_read_slot += 1;
                let f = self.egraph.add(
                    NodeKind::Flow(FlowParams::new(slot)),
                    egui::pos2(300.0, 0.0),
                );
                self.egraph.connect(f, 0, 0);
                self.egraph.view = Some(f);
                self.playing = false;
                t.node = f;
                next(&mut t, 1);
            }
            1 if t.since.elapsed().as_millis() > 500 => {
                self.flow_action(t.node, Action::Analyse);
                next(&mut t, 2);
            }
            2 => match self.jobs.flow.run.as_ref().map(|r| &r.phase) {
                Some(Phase::Failed(e)) => fail(format!("the analysis failed: {e}")),
                Some(Phase::Done(_)) => {
                    if !self.loaded.contains_key(&t.node) {
                        fail("the vectors did not load into the node's reader".into());
                    }
                    self.seek_to(AT);
                    next(&mut t, 3);
                }
                _ => {}
            },
            3 => {
                if let Some(x) = stable(self, &mut t, 0, 8) {
                    t.fwd[0] = x;
                    next(&mut t, 4);
                }
            }
            4 => {
                if let Some(y) = stable(self, &mut t, 1, 8) {
                    t.fwd[1] = y;
                    // The truth: frame AT -> AT + 1 at the frame centre.
                    let step = h_mul(&synth::path(W, HT, AT as usize + 1), &h_inv(&synth::path(W, HT, AT as usize)).unwrap());
                    let (cx, cy) = (W as f64 / 2.0, HT as f64 / 2.0);
                    let (nx, ny) = h_apply(&step, cx, cy);
                    let want = [(nx - cx) as f32, (ny - cy) as f32];
                    let err = ((t.fwd[0] - want[0]).powi(2) + (t.fwd[1] - want[1]).powi(2)).sqrt();
                    println!("{:?}: flow at the centre {:?}, truth {want:?}, error {err:.2} px", t.engine, t.fwd);
                    let tol = match t.engine {
                        engine::graph::FlowEngine::SeaRaft => 0.25,
                        engine::graph::FlowEngine::Rife => 1.0,
                    };
                    if err > tol {
                        fail(format!("{:?}: forward vector {:?} is not the true motion {want:?} ({err:.2} px off)", t.engine, t.fwd));
                    }
                    if t.engine == engine::graph::FlowEngine::SeaRaft {
                        // The other engine: a new recipe, a new analysis.
                        t.engine = engine::graph::FlowEngine::Rife;
                        if let Some(NodeKind::Flow(p)) = self.egraph.node_mut(t.node).map(|n| &mut n.kind) {
                            p.engine = t.engine;
                        }
                        self.flow_action(t.node, Action::Dismiss);
                        next(&mut t, 1);
                        self.test.flow = Some(t);
                        return;
                    }
                    // A VectorBlur over the plate, along those vectors.
                    let b = self.egraph.add(
                        NodeKind::VectorBlur(engine::graph::VectorBlurParams { amount: 0.0, ..Default::default() }),
                        egui::pos2(500.0, 0.0),
                    );
                    self.egraph.connect(b, 0, 0);
                    self.egraph.connect(b, 1, t.node);
                    self.egraph.view = Some(b);
                    t.blur = b;
                    next(&mut t, 5);
                }
            }
            5 => {
                if let Some(v) = stable(self, &mut t, 0, 1).and_then(|_| pixels(self)) {
                    t.sharp = v;
                    if let Some(NodeKind::VectorBlur(p)) = self.egraph.node_mut(t.blur).map(|n| &mut n.kind) {
                        // Two frames' worth: a long smear, unmistakable on a
                        // texture whose pixels are mostly gentle.
                        p.amount = 2.0;
                    }
                    next(&mut t, 6);
                }
            }
            6 => {
                if let Some(v) = stable(self, &mut t, 0, 1).and_then(|_| pixels(self)) {
                    let moved: f32 = v.iter().zip(&t.sharp).map(|(a, b)| (a - b).abs()).sum();
                    println!("vector blur moved nine pixels by {moved:.4} in all");
                    if moved < 0.02 {
                        fail(format!("a two-frame shutter along {:?} px left the texture as it was ({moved:.4})", t.fwd));
                    }
                    // A Rifetime of the plate at half speed.
                    let slot = self.next_read_slot;
                    self.next_read_slot += 1;
                    let rt = self.egraph.add(NodeKind::Rifetime(engine::graph::RifetimeParams::new(slot)), egui::pos2(500.0, 120.0));
                    self.egraph.connect(rt, 0, 0);
                    self.egraph.view = Some(rt);
                    t.retime = rt;
                    self.flow_action(rt, Action::Analyse);
                    next(&mut t, 7);
                }
            }
            7 => match self.jobs.flow.run.as_ref().filter(|r| r.node == t.retime).map(|r| &r.phase) {
                Some(Phase::Failed(e)) => fail(format!("the retime failed: {e}")),
                Some(Phase::Done(_)) => {
                    if !self.loaded.contains_key(&t.retime) {
                        fail("the retimed frames did not load into the node's reader".into());
                    }
                    let (r, _) = self.range_of(t.retime);
                    if r != (0, 2 * (N as u64 - 1)) {
                        fail(format!("a half-speed Rifetime of 0..{} runs {r:?}, not 0..{}", N - 1, 2 * (N - 1)));
                    }
                    t.at = 40;
                    self.seek_to(40);
                    next(&mut t, 8);
                }
                _ => {}
            },
            8..=10 => {
                if let Some(v) = stable(self, &mut t, 0, 1).and_then(|_| pixels(self)) {
                    t.shots.push(v);
                    if t.stage < 10 {
                        t.at += 1;
                        self.seek_to(t.at);
                    } else {
                        // The plate itself at source frame 20.
                        t.at = 20;
                        self.egraph.view = Some(0);
                        self.seek_to(20);
                    }
                    let s = t.stage + 1;
                    next(&mut t, s);
                }
            }
            11 => {
                if let Some(src) = stable(self, &mut t, 0, 1).and_then(|_| pixels(self)) {
                    let diff = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f32>();
                    let (d_src, d_prev, d_next) = (diff(&t.shots[0], &src), diff(&t.shots[1], &t.shots[0]), diff(&t.shots[1], &t.shots[2]));
                    println!("retime: tick 40 vs source 20 {d_src:.4}, tick 41 vs 40 {d_prev:.4}, vs 42 {d_next:.4}");
                    if d_src > 0.01 {
                        fail(format!("retimed tick 40 is not source frame 20 ({d_src:.4})"));
                    }
                    if d_prev < 0.005 || d_next < 0.005 {
                        fail(format!("retimed tick 41 repeats a neighbour ({d_prev:.4}, {d_next:.4}): no in-between was made"));
                    }
                    // The same retime as a CURVE: source frame 0 at tick 0,
                    // 23 at tick 46 — half speed, linear — on the input's range.
                    if let Some(NodeKind::Rifetime(p)) = self.egraph.node_mut(t.retime).map(|n| &mut n.kind) {
                        p.mode = engine::graph::RifetimeMode::Curve;
                    }
                    if let Some(n) = self.egraph.node_mut(t.retime) {
                        let c = n.anim.entry(crate::anim::Channel::RifetimeFrame).or_default();
                        c.set_key(0, 0.0);
                        c.set_key(46, 23.0);
                    }
                    self.egraph.view = Some(t.retime);
                    self.flow_action(t.retime, Action::Dismiss);
                    self.flow_action(t.retime, Action::Analyse);
                    next(&mut t, 12);
                    self.test.flow = Some(t);
                    return;
                }
            }
            12 => match self.jobs.flow.run.as_ref().filter(|r| r.node == t.retime).map(|r| &r.phase) {
                Some(Phase::Failed(e)) => fail(format!("the curve retime failed: {e}")),
                Some(Phase::Done(_)) => {
                    let (r, _) = self.range_of(t.retime);
                    if r != (0, N as u64 - 1) {
                        fail(format!("a curve Rifetime runs {r:?}, not its input's 0..{}", N - 1));
                    }
                    t.at = 41;
                    self.seek_to(41);
                    next(&mut t, 13);
                }
                _ => {}
            },
            13 => {
                if let Some(v) = stable(self, &mut t, 0, 1).and_then(|_| pixels(self)) {
                    let d: f32 = v.iter().zip(&t.shots[1]).map(|(a, b)| (a - b).abs()).sum();
                    println!("rifetime curve: tick 41 vs the speed mode's tick 41 {d:.4}");
                    if d > 0.01 {
                        fail(format!("the curve's 20.5 is not the speed mode's 20.5 ({d:.4})"));
                    }
                    if let Some(m) = &t.models {
                        let _ = std::fs::remove_dir_all(m);
                    }
                    println!(
                        "FLOW TEST PASS: a Flow rendered the plate, solved it with SEA-RAFT and then \
                         RIFE and played the vectors back; the forward vector at the centre is the true \
                         motion (0.25 px / 1 px), a VectorBlur along it smears the texture, and a \
                         half-speed Rifetime runs twice as long with made in-betweens between source frames, \
                         and the same retime as a keyed curve makes the same frames on the input's range"
                    );
                    std::process::exit(0);
                }
            }
            _ => {}
        }
        self.test.flow = Some(t);
    }

    fn flow_state_words(&self) -> String {
        match self.jobs.flow.run.as_ref().map(|r| &r.phase) {
            None => "no run".into(),
            Some(Phase::Render) => format!("render, bake {}", self.graph_bake.is_some()),
            Some(Phase::Solving { done, total, .. }) => format!("solving {done}/{total}"),
            Some(Phase::Failed(e)) => format!("failed: {e}"),
            Some(Phase::Done(s)) => format!("done: {s}"),
        }
    }
}
