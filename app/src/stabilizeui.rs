//! The Stabilize node's app half: rendering its input (and its matte) to
//! disk, analysing the camera on a worker, and putting the path on the
//! node. The node (params, path maths, codegen arms) is
//! `engine::nodes::stabilize`, the analysis `engine::stabilize`, the panel
//! `crate::nodes::stabilize`. It runs on the Flow runner's types
//! (`flowui::{Run, Phase, View, Action, Msg}`) — one shape for every
//! click-to-analyse node.
//!
//! A run is explicit and cancellable:
//!
//! 1. **render** — a `SubtreeBake` of input 0 over the input's range into
//!    `<dir>/Input`, then, with a matte wired, of input 1 into
//!    `<dir>/Matte` (both `BakeSink::Stabilize`);
//! 2. **solve** — `engine::stabilize::run` on a thread, writing
//!    `<dir>/Motion/motion.txt` and its done-marker;
//! 3. **adopt** — the path goes onto the node (`StabPath`, behind an Arc),
//!    in the graph AND in `undo_last`: the result of a click, not an edit,
//!    so it is not an undo step of its own.
//!
//! `<dir>` is keyed on everything that shapes the path — both inputs'
//! subtree keys, the recipe (fit model, detail), the range — and the key
//! rides on the path (`StabPath::recipe`), so the panel says STALE when
//! anything upstream moves, and the old path keeps working until a
//! re-analysis. Smoothing, reference, axes, reverse and zoom are NOT in it:
//! they are evaluated per frame from the path, so changing them never
//! re-runs anything.

use crate::corridorkey::frame_name;
use crate::flowui::{Action, Msg, Phase, Run, View, ViewState};
use crate::Gfx;
use engine::graph::{NodeKind, StabPath, StabPathRef};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::Arc;

#[derive(Default)]
pub struct Jobs {
    pub run: Option<Run>,
    /// The render under way is the matte's (port 1), not the picture's.
    pub matte: bool,
}

impl Gfx {
    /// The frames a run covers: the input's range, handles included.
    fn stab_range(&self, input: u64) -> (u64, u64) {
        let ((a, b), _) = self.range_of(input);
        (a.saturating_sub(self.handles.0), b.saturating_add(self.handles.1) + 1)
    }

    /// The key of a run of `node` at the current recipe. None while the
    /// input cannot be rendered.
    fn stab_key(&self, node: u64) -> Option<u64> {
        let NodeKind::Stabilize(p) = &self.egraph.node(node)?.kind else { return None };
        let input = self.egraph.input_of(node, 0)?;
        let k0 = self.subtree_key_of(input)?;
        let k1 = match self.egraph.input_of(node, 1) {
            Some(m) => self.subtree_key_of(m)?,
            None => 0,
        };
        let (from, len) = self.stab_range(input);
        let recipe = serde_json::to_vec(&(p.recipe(), from, len)).ok()?;
        Some(engine::diskcache::content_hash(&[&k0.to_le_bytes(), &k1.to_le_bytes(), &recipe]))
    }

    fn stab_dir(&self, key: u64) -> PathBuf {
        let root = match &self.disk_cache {
            Some(c) => c.root().join("stabilize"),
            None => engine::paths::temp_dir().join("wobble-stabilize"),
        };
        root.join(format!("{key:016x}"))
    }

    pub(crate) fn stab_view(&self, node: u64) -> View {
        let mine = self.stab.run.as_ref().filter(|r| r.node == node);
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
        let key = self.stab_key(node);
        let loaded = match self.egraph.node(node).map(|n| &n.kind) {
            Some(NodeKind::Stabilize(p)) if p.analysed() => Some((p.path.motion.len(), Some(p.path.recipe) == key)),
            _ => None,
        };
        View {
            state,
            has_input: self.egraph.input_of(node, 0).is_some(),
            weights: engine::models::ready_path("searaft-m").is_ok(),
            loaded,
            on_disk: key.is_some_and(|k| engine::bakes::is_baked(&self.stab_dir(k).join("Motion"))),
            busy_elsewhere: self.stab.run.as_ref().is_some_and(|r| r.node != node && r.active()),
        }
    }

    pub(crate) fn stab_action(&mut self, node: u64, a: Action) {
        match a {
            Action::Analyse => self.stab_start(node),
            Action::Cancel => {
                if let Some(r) = self.stab.run.take() {
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
                if self.stab.run.as_ref().is_some_and(|r| r.node == node && !r.active()) {
                    self.stab.run = None;
                }
            }
        }
    }

    fn stab_fail(&mut self, node: u64, e: &str) {
        match self.stab.run.as_mut().filter(|r| r.node == node) {
            Some(r) => r.phase = Phase::Failed(e.to_string()),
            None => self.stab.run = Some(Run { node, dir: PathBuf::new(), phase: Phase::Failed(e.into()) }),
        }
    }

    /// The `analyse` click: adopt a path already on disk, else render.
    fn stab_start(&mut self, node: u64) {
        if self.stab.run.as_ref().is_some_and(|r| r.active()) {
            return;
        }
        #[cfg(target_arch = "wasm32")]
        return self.stab_fail(node, "the analysis needs a worker thread: not in the web build");
        #[cfg_attr(target_arch = "wasm32", allow(unreachable_code))]
        if self.egraph.input_of(node, 0).is_none() {
            return self.stab_fail(node, "wire a picture into the input");
        }
        let Some(key) = self.stab_key(node) else {
            return self.stab_fail(node, "the input is not renderable yet");
        };
        let dir = self.stab_dir(key);
        if engine::bakes::is_baked(&dir.join("Motion")) {
            self.stab.run = Some(Run { node, dir: dir.clone(), phase: Phase::Done("loaded the analysis on disk".into()) });
            if let Err(e) = self.stab_adopt(node, &dir, key) {
                self.stab_fail(node, &e);
            }
            return;
        }
        if let Err(e) = engine::models::ready_path("searaft-m") {
            return self.stab_fail(node, &format!("{e}"));
        }
        self.stab.run = Some(Run { node, dir: dir.clone(), phase: Phase::Render });
        self.stab.matte = false;
        if let Err(e) = self.stab_render(node, 0, dir) {
            self.stab_fail(node, &e);
        }
    }

    /// A `SubtreeBake` of input `port` into `<dir>/Input` or `<dir>/Matte`.
    fn stab_render(&mut self, node: u64, port: usize, dir: PathBuf) -> Result<(), String> {
        let input = self.egraph.input_of(node, port).ok_or("nothing wired")?;
        let sub = dir.join(if port == 0 { "Input" } else { "Matte" });
        std::fs::create_dir_all(&sub).map_err(|e| format!("{e}"))?;
        let key = self.subtree_key_of(input).ok_or("the input is not renderable yet")?;
        let (w, h) = self.comp.output_size().ok_or("nothing is rendering yet")?;
        // Both renders cover the PICTURE's range, so frame i is frame i.
        let (from, len) = self.stab_range(self.egraph.input_of(node, 0).ok_or("nothing wired")?);
        if len.saturating_sub(from) < 2 {
            return Err("this needs at least two frames; the input's range has one".into());
        }
        self.playing = false;
        let mut next = from;
        while next < len && sub.join(frame_name(next)).exists() {
            next += 1;
        }
        let pb = self.playback.as_mut().ok_or("load a sequence first")?;
        pb.player.seek(next.min(len.saturating_sub(1)));
        let info = engine::decode::ImageInfo { width: w, height: h };
        let readback = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("stabilize-render"),
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
            sink: crate::BakeSink::Stabilize { port },
        };
        if bake.next >= bake.len {
            self.finish_graph_bake(bake);
        } else {
            self.graph_bake = Some(bake);
        }
        Ok(())
    }

    /// The bake loop finished a render: the matte next, or the solve.
    pub(crate) fn stab_render_finished(&mut self, node: u64, port: usize) {
        let Some(r) = self.stab.run.as_ref().filter(|r| r.node == node) else { return };
        let dir = r.dir.clone();
        if port == 0 && self.egraph.input_of(node, 1).is_some() {
            self.stab.matte = true;
            if let Err(e) = self.stab_render(node, 1, dir) {
                self.stab_fail(node, &e);
            }
            return;
        }
        self.stab.matte = false;
        let Some(NodeKind::Stabilize(p)) = self.egraph.node(node).map(|n| n.kind.clone()) else {
            return self.stab_fail(node, "the node is gone");
        };
        let weights = match engine::models::ready_path("searaft-m") {
            Ok(w) => w,
            Err(e) => return self.stab_fail(node, &format!("{e}")),
        };
        let list = |sub: &str| -> Result<Vec<PathBuf>, String> {
            let mut v: Vec<PathBuf> = std::fs::read_dir(dir.join(sub))
                .map_err(|e| format!("reading the rendered frames: {e}"))?
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|e| e == "exr"))
                .collect();
            // Named by comp frame, zero-padded: name order is time order.
            v.sort();
            Ok(v)
        };
        let frames = match list("Input") {
            Ok(f) => f,
            Err(e) => return self.stab_fail(node, &e),
        };
        let mattes = if self.egraph.input_of(node, 1).is_some() {
            match list("Matte") {
                Ok(m) if m.len() == frames.len() => Some(m),
                Ok(m) => return self.stab_fail(node, &format!("{} matte frames for {} pictures", m.len(), frames.len())),
                Err(e) => return self.stab_fail(node, &e),
            }
        } else {
            None
        };
        let job = engine::stabilize::Job {
            weights,
            frames,
            mattes,
            out: dir.join("Motion"),
            divisor: p.divisor.clamp(1, 4) as usize,
            model: p.model,
        };
        let (tx, rx) = std::sync::mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let (device, queue, stop) = (self.ctx.device.clone(), self.ctx.queue.clone(), cancel.clone());
        let spawned = std::thread::Builder::new().name("stabilize".into()).spawn(move || {
            let mut progress = |a, b| {
                let _ = tx.send(Msg::Frame(a, b));
            };
            let _ = tx.send(match engine::stabilize::run(device, queue, &job, &mut progress, &stop) {
                Ok(_) => Msg::Done,
                Err(e) => Msg::Failed(format!("{e:#}")),
            });
        });
        if let Err(e) = spawned {
            return self.stab_fail(node, &format!("could not start the worker: {e}"));
        }
        if let Some(r) = self.stab.run.as_mut() {
            r.phase = Phase::Solving { rx, cancel, done: 0, total: 0 };
        }
    }

    /// Read `<dir>/Motion` onto the node — in the graph and in the undo
    /// baseline, so the landing is not an undo step of its own.
    fn stab_adopt(&mut self, node: u64, dir: &std::path::Path, key: u64) -> Result<(), String> {
        let (motion, size) = engine::stabilize::read_motion(&dir.join("Motion")).map_err(|e| format!("{e:#}"))?;
        let first = self.egraph.input_of(node, 0).map(|i| self.stab_range(i).0).unwrap_or(0) as i64;
        let path = StabPathRef::new(StabPath::new(first, size, motion, key));
        for g in [&mut self.egraph, &mut self.undo_last] {
            if let Some(NodeKind::Stabilize(p)) = g.node_mut(node).map(|n| &mut n.kind) {
                p.path = path.clone();
            }
        }
        self.release_render_cache();
        Ok(())
    }

    /// Every redraw: drain the worker, catch a render that died, adopt a
    /// finished analysis.
    pub(crate) fn stab_poll(&mut self) {
        let Some(r) = self.stab.run.as_mut() else { return };
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
                let dir = r.dir.clone();
                r.phase = Phase::Done("done".into());
                // The key is the directory's name.
                let key = dir
                    .file_name()
                    .and_then(|n| u64::from_str_radix(&n.to_string_lossy(), 16).ok())
                    .unwrap_or(0);
                if let Err(e) = self.stab_adopt(node, &dir, key) {
                    self.stab_fail(node, &e);
                }
            }
            Some(Err(e)) => r.phase = Phase::Failed(e),
            None => {}
        }
    }
}

// ------------------------------------------------------------- the suite

pub struct TestStab {
    stage: u32,
    t0: web_time::Instant,
    since: web_time::Instant,
    node: u64,
    back: u64,
    at: u64,
    last: Option<Vec<f32>>,
    /// Probes: the plate at the reference and at `AT`, then the stabilised
    /// picture at both, then the reversed one at `AT`.
    shots: Vec<Vec<f32>>,
    truth: Vec<engine::planar::H>,
    models: Option<PathBuf>,
    summary: String,
    old_key: Option<u64>,
}

impl Default for TestStab {
    fn default() -> Self {
        let now = web_time::Instant::now();
        Self { stage: 0, t0: now, since: now, node: 0, back: 0, at: 0, last: None, shots: Vec::new(), truth: Vec::new(), models: None, summary: String::new(), old_key: None }
    }
}

impl Gfx {
    /// `--test-stabilize`: a Stabilize node end to end in the live app, on a
    /// synthetic shaky shot (`engine::stabilize::write_shaky`, written to
    /// `data/stabilize_synth`): `analyse` renders the plate and solves the
    /// camera with the REAL network (the catalog's SEA-RAFT file from
    /// `data/searaft/`, staged into a private models dir; without it the
    /// suite says SKIP and exits clean). Then: the path against the truth;
    /// lock-off through the node's own pin at every frame (< 0.5 px); the
    /// picture — patches of the stabilised output at a shaky frame match the
    /// reference frame's where the raw plate's do not; and a second
    /// Stabilize with `reverse` puts the plate back. Waits are wall-clock.
    pub(crate) fn drive_test_stab(&mut self) {
        use engine::planar::{h_apply, h_mul};
        let Some(mut t) = self.test.stab.take() else { return };
        let fail = |m: String| -> ! {
            println!("STABILIZE TEST FAIL: {m}");
            std::process::exit(1);
        };
        if t.t0.elapsed().as_secs() > 420 {
            fail(format!("timeout at stage {}", t.stage));
        }
        const W: usize = 960;
        const HT: usize = 540;
        const N: usize = 36;
        const AT: u64 = 23;
        let dir = std::path::Path::new("data/stabilize_synth");
        let next = |t: &mut TestStab, s: u32| {
            t.stage = s;
            t.since = web_time::Instant::now();
            t.last = None;
        };
        // Nine 8x8 patch means across the middle of the frame, read twice
        // 150 ms apart at the playhead and agreed.
        let probe = |me: &mut Self, t: &mut TestStab| -> Option<Vec<f32>> {
            if me.force_display || me.eval_tick() != t.at || t.since.elapsed().as_millis() < 150 {
                return None;
            }
            let mut v = Vec::new();
            for fy in [0.3f32, 0.5, 0.7] {
                for fx in [0.3f32, 0.5, 0.7] {
                    v.push(me.probe_output_box((W as f32 * fx) as u32 - 4, (HT as f32 * fy) as u32 - 4, 8, 8, 0)?);
                }
            }
            t.since = web_time::Instant::now();
            match t.last.replace(v.clone()) {
                Some(p) if p.iter().zip(&v).all(|(a, b)| (a - b).abs() < 1e-5) => {
                    t.last = None;
                    Some(v)
                }
                _ => None,
            }
        };
        let diff = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f32>();
        match t.stage {
            0 if self.playback.is_some() && self.displayed_total >= 2 => {
                let weights = std::path::Path::new("data/searaft/C-T-M.safetensors");
                if !weights.is_file() {
                    println!("STABILIZE TEST SKIP: {} missing (the catalog's searaft-m, byte for byte)", weights.display());
                    std::process::exit(0);
                }
                let models = engine::paths::temp_dir().join(format!("wobble-stab-models-{}", std::process::id()));
                let _ = std::fs::create_dir_all(&models);
                let spec = engine::models::spec("searaft-m").unwrap_or_else(|| fail("searaft-m is not in the catalog".into()));
                if let Err(e) = std::fs::copy(weights, spec.path_in(&models)) {
                    fail(format!("staging the weights: {e}"));
                }
                std::env::set_var("WOBBLE_MODELS_DIR", &models);
                if let Err(e) = engine::models::ready_path("searaft-m") {
                    fail(format!("the staged weights are not the catalog's: {e}"));
                }
                t.models = Some(models);
                // Always rewritten: the truth comes back with the frames.
                match engine::stabilize::write_shaky(dir, W, HT, N) {
                    Ok(truth) => t.truth = truth,
                    Err(e) => fail(format!("writing {}: {e:#}", dir.display())),
                }
                self.load_read(0, &dir.to_string_lossy());
                let s = self.egraph.add(NodeKind::Stabilize(Default::default()), egui::pos2(300.0, 0.0));
                self.egraph.connect(s, 0, 0);
                self.egraph.view = Some(s);
                self.playing = false;
                t.node = s;
                next(&mut t, 1);
            }
            1 if t.since.elapsed().as_millis() > 500 => {
                if self.range_of(0).0 != (0, N as u64 - 1) {
                    fail(format!("the shot reads as {:?}, not 0..{}", self.range_of(0).0, N - 1));
                }
                // (A Load is red while its frames are not cached; that
                // taint is the Load's. The node's own colour is the claim.)
                if self.cost_flow.is_slow(t.node) && !self.cost_flow.uncached.contains(&0) {
                    fail("an unanalysed Stabilize is not cyan".into());
                }
                self.stab_action(t.node, Action::Analyse);
                next(&mut t, 2);
            }
            2 => match self.stab.run.as_ref().map(|r| &r.phase) {
                Some(Phase::Failed(e)) => fail(format!("the analysis failed: {e}")),
                Some(Phase::Done(_)) => {
                    let Some(NodeKind::Stabilize(p)) = self.egraph.node(t.node).map(|n| n.kind.clone()) else {
                        fail("the node is gone".into())
                    };
                    if p.path.motion.len() != N || p.path.first != 0 {
                        fail(format!("a path of {} frames from {}, wanted {N} from 0", p.path.motion.len(), p.path.first));
                    }
                    if self.stab_view(t.node).loaded != Some((N, true)) {
                        fail("the panel does not call the path up to date".into());
                    }
                    if self.egraph != self.undo_last {
                        fail("the analysis landing became an undo step".into());
                    }
                    // The path against the truth, inside the frame.
                    let pts = [[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75], [0.5, 0.5]]
                        .map(|q| [q[0] * W as f64, q[1] * HT as f64]);
                    let mut path_err = 0.0f64;
                    let mut lock = 0.0f64;
                    for f in 0..N {
                        for q in pts {
                            let (a, b) = h_apply(&p.path.motion[f], q[0], q[1]);
                            let (c, d) = h_apply(&t.truth[f], q[0], q[1]);
                            path_err = path_err.max((a - c).hypot(b - d));
                        }
                        // The node's own pin at this frame, through the
                        // engine-graph lowering, undoes the true camera.
                        let g = self.egraph.to_engine_graph_at(Some(f as u64));
                        let Some(NodeKind::Stabilize(live)) = g.nodes.get(&t.node).cloned() else {
                            fail("no Stabilize in the engine graph".into())
                        };
                        let Some(shown) = engine::nodes::stabilize::pin_h(&live.pin, W as f64, HT as f64) else {
                            fail(format!("degenerate pin at {f}"))
                        };
                        let still = h_mul(&shown, &t.truth[f]);
                        for q in pts {
                            let (a, b) = h_apply(&still, q[0], q[1]);
                            lock = lock.max((a - q[0]).hypot(b - q[1]));
                        }
                    }
                    println!("stabilize: path within {path_err:.3} px of the truth, lock-off residual {lock:.3} px");
                    if lock > 0.5 {
                        fail(format!("lock-off leaves {lock:.3} px of motion"));
                    }
                    self.test.stab_lock = lock;
                    if self.cost_flow.is_slow(t.node) && !self.cost_flow.uncached.contains(&0) {
                        fail("an analysed Stabilize is not cyan".into());
                    }
                    // Pictures: the raw plate first.
                    self.egraph.view = Some(0);
                    t.at = 0;
                    self.seek_to(0);
                    next(&mut t, 3);
                }
                _ => {}
            },
            3..=6 => {
                if let Some(v) = probe(self, &mut t) {
                    t.shots.push(v);
                    // 3: plate @0, 4: plate @AT, 5: stabilised @0, 6: stabilised @AT.
                    match t.stage {
                        3 => t.at = AT,
                        4 => {
                            self.egraph.view = Some(t.node);
                            t.at = 0;
                        }
                        5 => t.at = AT,
                        _ => {
                            // A second Stabilize, reversed, on the same path:
                            // it must put the plate back.
                            let Some(NodeKind::Stabilize(p)) = self.egraph.node(t.node).map(|n| n.kind.clone()) else {
                                fail("the node is gone".into())
                            };
                            let r = self.egraph.add(
                                NodeKind::Stabilize(engine::graph::StabilizeParams { reverse: true, ..p }),
                                egui::pos2(500.0, 0.0),
                            );
                            self.egraph.connect(r, 0, t.node);
                            self.egraph.view = Some(r);
                            t.back = r;
                        }
                    }
                    self.seek_to(t.at);
                    let s = t.stage + 1;
                    next(&mut t, s);
                }
            }
            7 => {
                if let Some(v) = probe(self, &mut t) {
                    let (raw, stab, back) = (diff(&t.shots[0], &t.shots[1]), diff(&t.shots[2], &t.shots[3]), diff(&v, &t.shots[1]));
                    println!(
                        "stabilize: nine patches, frame {AT} against the reference: plate {raw:.4}, \
                         stabilised {stab:.4}; reversed against the plate {back:.4}"
                    );
                    if raw < 0.05 {
                        fail(format!("the synthetic shot barely moves ({raw:.4}): the test proves nothing"));
                    }
                    if stab > 0.2 * raw {
                        fail(format!("the stabilised picture still moves: {stab:.4} against the plate's {raw:.4}"));
                    }
                    if back > 0.2 * raw {
                        fail(format!("reverse does not restore the plate: {back:.4} against {raw:.4}"));
                    }
                    // The maths of the pair: reverse ∘ stabilise = identity.
                    let mut ident = 0.0f64;
                    for f in 0..N as u64 {
                        let g = self.egraph.to_engine_graph_at(Some(f));
                        let (Some(NodeKind::Stabilize(a)), Some(NodeKind::Stabilize(b))) =
                            (g.nodes.get(&t.node).cloned(), g.nodes.get(&t.back).cloned())
                        else {
                            fail("no live pair".into())
                        };
                        let ha = engine::nodes::stabilize::pin_h(&a.pin, W as f64, HT as f64).unwrap_or_else(|| fail("degenerate".into()));
                        let hb = engine::nodes::stabilize::pin_h(&b.pin, W as f64, HT as f64).unwrap_or_else(|| fail("degenerate".into()));
                        let both = h_mul(&hb, &ha);
                        for q in [[100.0, 100.0], [860.0, 440.0], [480.0, 270.0]] {
                            let (x, y) = h_apply(&both, q[0], q[1]);
                            ident = ident.max((x - q[0]).hypot(y - q[1]));
                        }
                    }
                    if ident > 1e-2 {
                        fail(format!("reverse after stabilise moves a point {ident:.4} px"));
                    }
                    t.summary = format!(
                        "lock-off holds the true camera to {:.3} px at every frame, the stabilised picture \
                         moves {stab:.4} where the plate moves {raw:.4}, and reverse puts the plate back \
                         ({back:.4}; the pins compose to identity within {ident:.1e} px)",
                        self.test.stab_lock
                    );
                    // A matte over the right half: its pixels leave the
                    // vote, the run is a new recipe, and the camera (which
                    // moves the whole frame) comes out the same.
                    t.old_key = self.egraph.node(t.node).and_then(|n| match &n.kind {
                        NodeKind::Stabilize(p) => Some(p.path.recipe),
                        _ => None,
                    });
                    let m = self.egraph.add(
                        NodeKind::Gradient(engine::graph::GradientParams { inherit: true, ..Default::default() }),
                        egui::pos2(300.0, 150.0),
                    );
                    self.egraph.connect(t.node, 1, m);
                    self.stab_action(t.node, Action::Dismiss);
                    self.stab_action(t.node, Action::Analyse);
                    next(&mut t, 8);
                }
            }
            8 => match self.stab.run.as_ref().map(|r| (&r.phase, r.dir.clone())) {
                Some((Phase::Failed(e), _)) => fail(format!("the matted analysis failed: {e}")),
                Some((Phase::Done(_), dir)) => {
                    let mattes = std::fs::read_dir(dir.join("Matte")).map(|r| r.count()).unwrap_or(0);
                    if mattes != N {
                        fail(format!("{mattes} matte frames rendered, wanted {N}"));
                    }
                    let Some(NodeKind::Stabilize(p)) = self.egraph.node(t.node).map(|n| n.kind.clone()) else {
                        fail("the node is gone".into())
                    };
                    if Some(p.path.recipe) == t.old_key {
                        fail("wiring a matte did not make a new analysis".into());
                    }
                    let mut err = 0.0f64;
                    for f in 0..N {
                        for q in [[0.25, 0.5], [0.75, 0.5], [0.5, 0.25]] {
                            let q = [q[0] * W as f64, q[1] * HT as f64];
                            let (a, b) = h_apply(&p.path.motion[f], q[0], q[1]);
                            let (c, d) = h_apply(&t.truth[f], q[0], q[1]);
                            err = err.max((a - c).hypot(b - d));
                        }
                    }
                    println!("stabilize: with half the frame matted out, the path is within {err:.3} px");
                    if err > 0.75 {
                        fail(format!("the matted analysis is {err:.3} px off the truth"));
                    }
                    if let Some(m) = &t.models {
                        let _ = std::fs::remove_dir_all(m);
                    }
                    println!(
                        "STABILIZE TEST PASS: analysed a {N}-frame shaky shot with SEA-RAFT through the app; {}; \
                         with a matte wired the run renders it too and the path holds ({err:.3} px)",
                        t.summary
                    );
                    std::process::exit(0);
                }
                _ => {}
            },
            _ => {}
        }
        self.test.stab = Some(t);
    }
}
