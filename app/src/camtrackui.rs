//! The CameraTrack node's app half: rendering its input (and its matte)
//! to disk, tracking and solving on a worker, and putting the solve on
//! the node. The node (params, the Camera's read of it) is
//! `engine::nodes::camtrack`, the tracker and solver `engine::camtrack`,
//! the panel `crate::nodes::camtrack`. It runs on the Flow runner's types
//! (`flowui::{Run, Phase, View, Action, Msg}`), like Stabilize.
//!
//! A run is explicit and cancellable:
//!
//! 1. **render** — a `SubtreeBake` of input 0 over the input's range into
//!    `<dir>/Input`, then, with a matte wired, of input 1 into
//!    `<dir>/Matte` (both `BakeSink::CamTrack`);
//! 2. **track** — `engine::camtrack::track` on a thread, writing
//!    `<dir>/Tracks/tracks.txt` and its done-marker;
//! 3. **solve** — `engine::camtrack::solve` on the same thread, writing
//!    `<dir>/Solve-<key>/solve.txt`;
//! 4. **adopt** — the solve goes onto the node (`CamSolve`, behind an Arc),
//!    in the graph AND in `undo_last`: the result of a click, not an edit.
//!
//! `<dir>` is keyed on what shapes the TRACKS (both inputs' subtree keys,
//! the detail, the range); the solve directory on that plus the lens
//! settings. A lens change finds the tracks on disk and re-solves alone.
//! The node's `scale` and `show` are read at use and re-run nothing.
//!
//! `camtrack_sync` keeps `engine::scene3d`'s tracker registry in step with
//! the nodes, so a Camera with `Source::Tracker` reads the solve per frame.

use crate::corridorkey::frame_name;
use crate::flowui::{Action, Msg, Phase, Run, View, ViewState};
use crate::Gfx;
use engine::graph::{CamSolve, CamSolveRef, CamTrackParams, NodeKind};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::Arc;

/// `Msg::Frame` with this `done` marks the hand-over from tracking to
/// solving (the runner's messages carry no phase of their own).
const SOLVING_MARK: usize = usize::MAX;

#[derive(Default)]
pub struct Jobs {
    pub run: Option<Run>,
    /// The render under way is the matte's (port 1), not the picture's.
    pub matte: bool,
    /// The worker is in its tracking half.
    pub tracking: bool,
    /// What the registry was last told per node, to tell a change apart
    /// from a redraw: the solve's Arc, scale, focal, aperture.
    synced: std::collections::HashMap<u64, (CamSolveRef, f32, f32, f32)>,
}

impl Gfx {
    /// The frames a run covers: the input's range, handles included.
    fn ct_range(&self, input: u64) -> (u64, u64) {
        let ((a, b), _) = self.range_of(input);
        (a.saturating_sub(self.handles.0), b.saturating_add(self.handles.1) + 1)
    }

    /// The key of the TRACKS of `node` at the current recipe. None while
    /// the input cannot be rendered.
    fn ct_key_tracks(&self, node: u64) -> Option<u64> {
        let NodeKind::CamTrack(p) = &self.egraph.node(node)?.kind else { return None };
        let input = self.egraph.input_of(node, 0)?;
        let k0 = self.subtree_key_of(input)?;
        let k1 = match self.egraph.input_of(node, 1) {
            Some(m) => self.subtree_key_of(m)?,
            None => 0,
        };
        let (from, len) = self.ct_range(input);
        let recipe = serde_json::to_vec(&(p.recipe_tracks(), from, len)).ok()?;
        Some(engine::diskcache::content_hash(&[&k0.to_le_bytes(), &k1.to_le_bytes(), &recipe]))
    }

    /// The key of the SOLVE: the tracks' key plus the lens settings.
    fn ct_key_solve(&self, node: u64) -> Option<(u64, u64)> {
        let tracks = self.ct_key_tracks(node)?;
        let NodeKind::CamTrack(p) = &self.egraph.node(node)?.kind else { return None };
        let recipe = serde_json::to_vec(&p.recipe_solve()).ok()?;
        Some((tracks, engine::diskcache::content_hash(&[&tracks.to_le_bytes(), &recipe])))
    }

    fn ct_dir(&self, key: u64) -> PathBuf {
        let root = match &self.disk_cache {
            Some(c) => c.root().join("camtrack"),
            None => engine::paths::temp_dir().join("wobble-camtrack"),
        };
        root.join(format!("{key:016x}"))
    }

    fn ct_solve_dir(dir: &std::path::Path, solve_key: u64) -> PathBuf {
        dir.join(format!("Solve-{solve_key:016x}"))
    }

    pub(crate) fn ct_view(&self, node: u64) -> View {
        let mine = self.camtrack.run.as_ref().filter(|r| r.node == node);
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
        let keys = self.ct_key_solve(node);
        let loaded = match self.egraph.node(node).map(|n| &n.kind) {
            Some(NodeKind::CamTrack(p)) if p.solved() => {
                Some((p.solve.solve.poses.len(), Some(p.solve.recipe) == keys.map(|k| k.1)))
            }
            _ => None,
        };
        let on_disk = keys.is_some_and(|(t, s)| {
            let dir = self.ct_dir(t);
            engine::bakes::is_baked(&Self::ct_solve_dir(&dir, s)) || engine::bakes::is_baked(&dir.join("Tracks"))
        });
        View {
            state,
            has_input: self.egraph.input_of(node, 0).is_some(),
            weights: engine::models::ready_path("searaft-m").is_ok(),
            loaded,
            on_disk,
            busy_elsewhere: self.camtrack.run.as_ref().is_some_and(|r| r.node != node && r.active()),
        }
    }

    pub(crate) fn ct_extra(&self, node: u64) -> crate::nodes::camtrack::Extra {
        let summary = match self.egraph.node(node).map(|n| &n.kind) {
            Some(NodeKind::CamTrack(p)) if p.solved() => {
                let s = &p.solve.solve;
                let focal = s.focal_px as f32 / s.size[0].max(1) as f32 * p.aperture;
                Some((s.poses.len(), s.points.len(), s.mean_error(), s.unplaced.len(), focal))
            }
            _ => None,
        };
        crate::nodes::camtrack::Extra { tracking: self.camtrack.tracking, summary }
    }

    pub(crate) fn ct_action(&mut self, node: u64, a: Action) {
        match a {
            Action::Analyse => self.ct_start(node),
            Action::Cancel => {
                if let Some(r) = self.camtrack.run.take() {
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
                if self.camtrack.run.as_ref().is_some_and(|r| r.node == node && !r.active()) {
                    self.camtrack.run = None;
                }
            }
        }
    }

    fn ct_fail(&mut self, node: u64, e: &str) {
        match self.camtrack.run.as_mut().filter(|r| r.node == node) {
            Some(r) => r.phase = Phase::Failed(e.to_string()),
            None => self.camtrack.run = Some(Run { node, dir: PathBuf::new(), phase: Phase::Failed(e.into()) }),
        }
    }

    /// The `solve` click: adopt a solve already on disk, else solve tracks
    /// already on disk, else render.
    fn ct_start(&mut self, node: u64) {
        if self.camtrack.run.as_ref().is_some_and(|r| r.active()) {
            return;
        }
        #[cfg(target_arch = "wasm32")]
        return self.ct_fail(node, "the solve needs a worker thread: not in the web build");
        #[cfg_attr(target_arch = "wasm32", allow(unreachable_code))]
        if self.egraph.input_of(node, 0).is_none() {
            return self.ct_fail(node, "wire a picture into the input");
        }
        let Some((tracks_key, solve_key)) = self.ct_key_solve(node) else {
            return self.ct_fail(node, "the input is not renderable yet");
        };
        let dir = self.ct_dir(tracks_key);
        let sdir = Self::ct_solve_dir(&dir, solve_key);
        if engine::bakes::is_baked(&sdir) {
            self.camtrack.run = Some(Run { node, dir: dir.clone(), phase: Phase::Done("loaded the solve on disk".into()) });
            if let Err(e) = self.ct_adopt(node, &sdir, solve_key) {
                self.ct_fail(node, &e);
            }
            return;
        }
        if engine::bakes::is_baked(&dir.join("Tracks")) {
            self.camtrack.run = Some(Run { node, dir: dir.clone(), phase: Phase::Render });
            self.ct_spawn(node, dir);
            return;
        }
        if let Err(e) = engine::models::ready_path("searaft-m") {
            return self.ct_fail(node, &format!("{e}"));
        }
        self.camtrack.run = Some(Run { node, dir: dir.clone(), phase: Phase::Render });
        self.camtrack.matte = false;
        if let Err(e) = self.ct_render(node, 0, dir) {
            self.ct_fail(node, &e);
        }
    }

    /// A `SubtreeBake` of input `port` into `<dir>/Input` or `<dir>/Matte`.
    fn ct_render(&mut self, node: u64, port: usize, dir: PathBuf) -> Result<(), String> {
        let input = self.egraph.input_of(node, port).ok_or("nothing wired")?;
        let sub = dir.join(if port == 0 { "Input" } else { "Matte" });
        std::fs::create_dir_all(&sub).map_err(|e| format!("{e}"))?;
        let key = self.subtree_key_of(input).ok_or("the input is not renderable yet")?;
        let (w, h) = self.comp.output_size().ok_or("nothing is rendering yet")?;
        // Both renders cover the PICTURE's range, so frame i is frame i.
        let (from, len) = self.ct_range(self.egraph.input_of(node, 0).ok_or("nothing wired")?);
        if len.saturating_sub(from) < 3 {
            return Err("this needs at least three frames; the input's range has fewer".into());
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
            label: Some("camtrack-render"),
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
            sink: crate::BakeSink::CamTrack { port },
        };
        if bake.next >= bake.len {
            self.finish_graph_bake(bake);
        } else {
            self.graph_bake = Some(bake);
        }
        Ok(())
    }

    /// The bake loop finished a render: the matte next, or the worker.
    pub(crate) fn ct_render_finished(&mut self, node: u64, port: usize) {
        let Some(r) = self.camtrack.run.as_ref().filter(|r| r.node == node) else { return };
        let dir = r.dir.clone();
        if port == 0 && self.egraph.input_of(node, 1).is_some() {
            self.camtrack.matte = true;
            if let Err(e) = self.ct_render(node, 1, dir) {
                self.ct_fail(node, &e);
            }
            return;
        }
        self.camtrack.matte = false;
        self.ct_spawn(node, dir);
    }

    /// The worker: track (unless the tracks are on disk), then solve.
    fn ct_spawn(&mut self, node: u64, dir: PathBuf) {
        let Some(NodeKind::CamTrack(p)) = self.egraph.node(node).map(|n| n.kind.clone()) else {
            return self.ct_fail(node, "the node is gone");
        };
        let Some((_, solve_key)) = self.ct_key_solve(node) else {
            return self.ct_fail(node, "the input is not renderable yet");
        };
        let tracks_dir = dir.join("Tracks");
        let have_tracks = engine::bakes::is_baked(&tracks_dir);
        let weights = if have_tracks {
            PathBuf::new()
        } else {
            match engine::models::ready_path("searaft-m") {
                Ok(w) => w,
                Err(e) => return self.ct_fail(node, &format!("{e}")),
            }
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
        let (frames, mattes) = if have_tracks {
            (Vec::new(), None)
        } else {
            let frames = match list("Input") {
                Ok(f) => f,
                Err(e) => return self.ct_fail(node, &e),
            };
            let mattes = if self.egraph.input_of(node, 1).is_some() {
                match list("Matte") {
                    Ok(m) if m.len() == frames.len() => Some(m),
                    Ok(m) => return self.ct_fail(node, &format!("{} matte frames for {} pictures", m.len(), frames.len())),
                    Err(e) => return self.ct_fail(node, &e),
                }
            } else {
                None
            };
            (frames, mattes)
        };
        let job = engine::camtrack::Job { weights, frames, mattes, out: tracks_dir.clone(), divisor: p.divisor.clamp(1, 4) as usize };
        let solve_dir = Self::ct_solve_dir(&dir, solve_key);
        let (focal, aperture, refine, guess) = (p.focal, p.aperture, p.refine_focal, p.guess);
        let (tx, rx) = std::sync::mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let (device, queue, stop) = (self.ctx.device.clone(), self.ctx.queue.clone(), cancel.clone());
        let spawned = std::thread::Builder::new().name("camtrack".into()).spawn(move || {
            let run = || -> anyhow::Result<()> {
                let (tracks, size) = if have_tracks {
                    engine::camtrack::read_tracks(&tracks_dir)?
                } else {
                    let mut progress = |a, b| {
                        let _ = tx.send(Msg::Frame(a, b));
                    };
                    engine::camtrack::track(device, queue, &job, &mut progress, &stop)?
                };
                let _ = tx.send(Msg::Frame(SOLVING_MARK, 0));
                let opts = engine::camtrack::SolveOpts {
                    size,
                    focal_px: focal.max(0.1) as f64 / aperture.max(0.1) as f64 * size[0] as f64,
                    refine_focal: refine,
                    max_keyframes: 100,
                };
                let mut progress = |a, b| {
                    let _ = tx.send(Msg::Frame(a, b));
                };
                let solve = if guess {
                    let starts = CamTrackParams::guess_starts_px(size[0] as f64);
                    engine::camtrack::solve_guess(&tracks, &opts, &starts, &mut progress, &stop)?
                } else {
                    engine::camtrack::solve(&tracks, &opts, &mut progress, &stop)?
                };
                std::fs::create_dir_all(&solve_dir)?;
                engine::camtrack::write_solve(&solve_dir, &solve)?;
                engine::bakes::mark_baked(&solve_dir)?;
                Ok(())
            };
            let _ = tx.send(match run() {
                Ok(()) => Msg::Done,
                Err(e) => Msg::Failed(format!("{e:#}")),
            });
        });
        if let Err(e) = spawned {
            return self.ct_fail(node, &format!("could not start the worker: {e}"));
        }
        self.camtrack.tracking = !have_tracks;
        if let Some(r) = self.camtrack.run.as_mut() {
            r.phase = Phase::Solving { rx, cancel, done: 0, total: 0 };
        }
    }

    /// Read `<solve dir>` onto the node — in the graph and in the undo
    /// baseline, so the landing is not an undo step of its own.
    fn ct_adopt(&mut self, node: u64, sdir: &std::path::Path, key: u64) -> Result<(), String> {
        let solve = engine::camtrack::read_solve(sdir).map_err(|e| format!("{e:#}"))?;
        let first = self.egraph.input_of(node, 0).map(|i| self.ct_range(i).0).unwrap_or(0) as i64;
        let s = CamSolveRef::new(CamSolve { first, solve, recipe: key });
        for g in [&mut self.egraph, &mut self.undo_last] {
            if let Some(NodeKind::CamTrack(p)) = g.node_mut(node).map(|n| &mut n.kind) {
                p.solve = s.clone();
            }
        }
        self.camtrack_sync();
        self.release_render_cache();
        // A 3D view on a scene with this camera frames the solve.
        if let Some(id) = self.view3d_target() {
            if self.view3d_cameras(id).iter().any(|(_, c)| matches!(c.source, engine::scene3d::Source::Tracker { node: n } if n == node)) {
                self.view3d_frame(id);
            }
        }
        Ok(())
    }

    /// Every redraw: drain the worker, catch a render that died, adopt a
    /// finished solve.
    pub(crate) fn ct_poll(&mut self) {
        let Some(r) = self.camtrack.run.as_mut() else { return };
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
                    Msg::Frame(SOLVING_MARK, _) => {
                        self.camtrack.tracking = false;
                        *done = 0;
                        *total = 0;
                    }
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
                match self.ct_key_solve(node) {
                    Some((_, key)) => {
                        let sdir = Self::ct_solve_dir(&dir, key);
                        if let Err(e) = self.ct_adopt(node, &sdir, key) {
                            self.ct_fail(node, &e);
                        }
                    }
                    None => self.ct_fail(node, "the input changed while solving"),
                }
            }
            Some(Err(e)) => r.phase = Phase::Failed(e),
            None => {}
        }
    }

    /// Keep `engine::scene3d`'s tracker registry in step with the nodes
    /// (a Camera with `Source::Tracker` reads it per frame). Cheap: a
    /// pointer compare per CameraTrack node.
    pub(crate) fn camtrack_sync(&mut self) {
        let mut seen = Vec::new();
        for (id, n) in &self.egraph.nodes {
            let NodeKind::CamTrack(p) = &n.kind else { continue };
            seen.push(*id);
            let now = (p.solve.clone(), p.scale, p.focal, p.aperture);
            let same = self
                .camtrack
                .synced
                .get(id)
                .is_some_and(|(s, sc, f, a)| s.ptr_eq(&now.0) && *sc == now.1 && *f == now.2 && *a == now.3);
            if same {
                continue;
            }
            engine::scene3d::set_tracker_solve(*id, p.solved().then(|| p.clone()));
            self.camtrack.synced.insert(*id, now);
        }
        let gone: Vec<u64> = self.camtrack.synced.keys().copied().filter(|k| !seen.contains(k)).collect();
        for id in gone {
            engine::scene3d::set_tracker_solve(id, None);
            self.camtrack.synced.remove(&id);
        }
    }

    /// The `camera…` click: a Camera node reading this solve, to the right
    /// of the tracker, selected.
    pub(crate) fn ct_make_camera(&mut self, node: u64) {
        let base = self.egraph.node(node).map(|n| n.pos).unwrap_or_default();
        let cam = self.egraph.add(
            NodeKind::Camera3D(engine::graph::CameraParams {
                source: engine::scene3d::Source::Tracker { node },
                ..Default::default()
            }),
            base + egui::vec2(180.0, 0.0),
        );
        self.selected = vec![cam];
    }

    /// The solved points at comp frame `t`, projected through the solved
    /// camera, normalised to the node's picture, with whether the point is
    /// in front of it. None without a solve, or with `show` off.
    pub(crate) fn camtrack_overlay(&self, node: u64, t: i64) -> Option<Vec<(usize, [f32; 2])>> {
        let NodeKind::CamTrack(p) = &self.egraph.node(node)?.kind else { return None };
        if !p.show || !p.solved() {
            return None;
        }
        let s = &p.solve.solve;
        let i = (t - p.solve.first).clamp(0, s.poses.len() as i64 - 1) as usize;
        let pose = s.poses[i];
        let k = engine::camtrack::Intrinsics::for_size(s.focal_px, s.size);
        let (w, h) = (s.size[0].max(1) as f32, s.size[1].max(1) as f32);
        Some(
            s.points
                .iter()
                .enumerate()
                .filter_map(|(idx, x)| Some((idx, k.project(pose.apply(*x))?)))
                .map(|(idx, q)| (idx, [q[0] as f32 / w, q[1] as f32 / h]))
                .filter(|(_, q)| q[0] >= 0.0 && q[0] <= 1.0 && q[1] >= 0.0 && q[1] <= 1.0)
                .collect(),
        )
    }

    /// The solved points over the viewer, and the picking gesture: S held
    /// + a click on a point toggles its pick (the roto / Sammie S), the
    /// nearest within `PICK_PX`. Picked points draw larger and brighter.
    pub(crate) fn camtrack_gizmo(
        &mut self,
        ui: &egui::Ui,
        ink: &mut crate::ink::InkShapes,
        node: u64,
        t: i64,
        nav: &egui::Response,
        to_scr: &dyn Fn([f32; 2]) -> egui::Pos2,
        pointer: Option<egui::Pos2>,
    ) {
        const PICK_PX: f32 = 12.0;
        let Some(pts) = self.camtrack_overlay(node, t) else { return };
        let picks: Vec<usize> = match self.egraph.node(node).map(|n| &n.kind) {
            Some(NodeKind::CamTrack(p)) => p.picks().to_vec(),
            _ => Vec::new(),
        };
        let s_held = ui.input(|i| i.key_down(egui::Key::S));
        let mut hover: Option<(usize, f32)> = None;
        if let Some(at) = pointer.filter(|_| s_held && nav.hovered()) {
            for (idx, q) in &pts {
                let d = to_scr(*q).distance(at);
                if d < PICK_PX && hover.is_none_or(|(_, hd)| d < hd) {
                    hover = Some((*idx, d));
                }
            }
        }
        ink.dots(pts.iter().filter(|(i, _)| !picks.contains(i)).map(|(_, q)| (to_scr(*q), None)), 2.0, 0.9, 0.0);
        for (idx, q) in &pts {
            if picks.contains(idx) {
                ink.circle_fill(to_scr(*q), 4.5, 1.0, true);
            }
        }
        if let Some((idx, _)) = hover {
            if let Some(q) = pts.iter().find(|(i, _)| *i == idx).map(|(_, q)| *q) {
                ink.circle_stroke(to_scr(q), 7.0, 1.5, 1.0, true);
            }
            if nav.clicked() {
                if let Some(NodeKind::CamTrack(p)) = self.egraph.node_mut(node).map(|n| &mut n.kind) {
                    p.toggle_pick(idx);
                }
            }
        }
    }

    /// The `card…` click: a Card on the picked points at the playhead,
    /// wired into the Blend 3D this solve's Camera feeds (if one), selected.
    pub(crate) fn ct_make_card(&mut self, node: u64) {
        let t = self.eval_tick() as f32;
        let Some(NodeKind::CamTrack(p)) = self.egraph.node(node).map(|n| n.kind.clone()) else { return };
        let Some(xform) = p.card_from_picks(t) else { return };
        let base = self.egraph.node(node).map(|n| n.pos).unwrap_or_default();
        let card = self.egraph.add(
            NodeKind::Card3D(engine::graph::CardParams { xform, ..Default::default() }),
            base + egui::vec2(180.0, 120.0),
        );
        // A Blend 3D fed by a Camera reading this solve takes the card.
        let cams: Vec<u64> = self
            .egraph
            .nodes
            .iter()
            .filter(|(_, n)| matches!(&n.kind, NodeKind::Camera3D(c) if matches!(c.source, engine::scene3d::Source::Tracker { node: s } if s == node)))
            .map(|(id, _)| *id)
            .collect();
        let scenes: Vec<(u64, usize)> = self
            .egraph
            .nodes
            .iter()
            .filter_map(|(id, n)| match &n.kind {
                NodeKind::Scene3D(s) => Some((*id, s.input_count())),
                _ => None,
            })
            .collect();
        for (sid, n) in scenes {
            let feeds_cam = (0..n).filter_map(|i| self.egraph.input_of(sid, i)).any(|s| cams.contains(&s));
            if !feeds_cam {
                continue;
            }
            match (0..n).find(|i| self.egraph.input_of(sid, *i).is_none()) {
                Some(free) => self.egraph.connect(sid, free, card),
                None => {
                    if let Some(NodeKind::Scene3D(s)) = self.egraph.node_mut(sid).map(|n| &mut n.kind) {
                        s.inputs = (s.inputs + 1).min(engine::scene3d::MAX_SCENE_INPUTS);
                    }
                    self.egraph.connect(sid, n, card);
                }
            }
            break;
        }
        self.selected = vec![card];
    }
}

// ------------------------------------------------------------- the suite

pub struct TestCamTrack {
    stage: u32,
    t0: web_time::Instant,
    since: web_time::Instant,
    node: u64,
    cam: u64,
    truth: Vec<engine::camtrack::Pose>,
    focal_px: f64,
    models: Option<PathBuf>,
    old_key: u64,
}

impl Default for TestCamTrack {
    fn default() -> Self {
        let now = web_time::Instant::now();
        Self { stage: 0, t0: now, since: now, node: 0, cam: 0, truth: Vec::new(), focal_px: 0.0, models: None, old_key: 0 }
    }
}

impl Gfx {
    /// `--test-camtrack`: a CameraTrack node end to end in the live app, on
    /// a synthetic three-plane shot (`engine::camtrack::synth::write_shot`,
    /// written to `data/camtrack_synth`): `solve` renders the plate, tracks
    /// it and solves the camera with the REAL network (the catalog's
    /// SEA-RAFT file from `data/searaft/`, staged into a private models
    /// dir; without it the suite says SKIP and exits clean). Then: the
    /// solved path against the truth (centres within 3% of the travel,
    /// rotation within 0.5°), the landing is no undo step, the node is
    /// cyan, `camera…` makes a Camera whose live pose at a frame is the
    /// solve's, a lens change marks the solve STALE and a re-solve finds
    /// the tracks on disk (no render). Waits are wall-clock.
    pub(crate) fn drive_test_camtrack(&mut self) {
        let Some(mut t) = self.test_camtrack.take() else { return };
        let fail = |m: String| -> ! {
            println!("CAMTRACK TEST FAIL: {m}");
            std::process::exit(1);
        };
        if t.t0.elapsed().as_secs() > 600 {
            fail(format!("timeout at stage {}", t.stage));
        }
        const W: usize = 960;
        const HT: usize = 540;
        const N: usize = 36;
        let dir = std::path::Path::new("data/camtrack_synth");
        let next = |t: &mut TestCamTrack, s: u32| {
            t.stage = s;
            t.since = web_time::Instant::now();
        };
        let judge = |me: &Self, t: &TestCamTrack| -> (f64, f64, f64) {
            let Some(NodeKind::CamTrack(p)) = me.egraph.node(t.node).map(|n| n.kind.clone()) else {
                fail("the node is gone".into())
            };
            if p.solve.solve.poses.len() != N || p.solve.first != 0 {
                fail(format!("a solve of {} frames from {}, wanted {N} from 0", p.solve.solve.poses.len(), p.solve.first));
            }
            let (wc, wr) = engine::camtrack::compare(&p.solve.solve.poses, &t.truth);
            let travel = engine::camtrack::norm(engine::camtrack::sub(t.truth[N - 1].centre(), t.truth[0].centre()));
            (wc / travel, wr, p.solve.solve.focal_px)
        };
        match t.stage {
            0 if self.playback.is_some() && self.displayed_total >= 2 => {
                let weights = std::path::Path::new("data/searaft/C-T-M.safetensors");
                if !weights.is_file() {
                    println!("CAMTRACK TEST SKIP: {} missing (the catalog's searaft-m, byte for byte)", weights.display());
                    std::process::exit(0);
                }
                let models = engine::paths::temp_dir().join(format!("wobble-camtrack-models-{}", std::process::id()));
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
                match engine::camtrack::synth::write_shot(dir, W, HT, N) {
                    Ok((truth, f)) => {
                        t.truth = truth;
                        t.focal_px = f;
                    }
                    Err(e) => fail(format!("writing {}: {e:#}", dir.display())),
                }
                self.load_read(0, &dir.to_string_lossy());
                // The lens: the shot's true focal, in millimetres on a 36 mm back.
                let focal = (t.focal_px / W as f64 * 36.0) as f32;
                let s = self.egraph.add(
                    NodeKind::CamTrack(CamTrackParams { focal, refine_focal: false, ..Default::default() }),
                    egui::pos2(300.0, 0.0),
                );
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
                    fail("an unsolved CameraTrack is not cyan".into());
                }
                self.ct_action(t.node, Action::Analyse);
                next(&mut t, 2);
            }
            2 => match self.camtrack.run.as_ref().map(|r| &r.phase) {
                Some(Phase::Failed(e)) => fail(format!("the solve failed: {e}")),
                Some(Phase::Done(_)) => {
                    let (rel, rot, _) = judge(self, &t);
                    println!("camtrack: centres within {:.2}% of the travel, rotation within {rot:.3}°", rel * 100.0);
                    if rel > 0.03 {
                        fail(format!("the solved camera is {:.2}% of the travel off", rel * 100.0));
                    }
                    if rot > 0.5 {
                        fail(format!("the solved camera is {rot:.3}° off"));
                    }
                    if self.ct_view(t.node).loaded != Some((N, true)) {
                        fail("the panel does not call the solve up to date".into());
                    }
                    if self.egraph != self.undo_last {
                        fail("the solve landing became an undo step".into());
                    }
                    if self.cost_flow.is_slow(t.node) && !self.cost_flow.uncached.contains(&0) {
                        fail("a solved CameraTrack is not cyan".into());
                    }
                    let Some(NodeKind::CamTrack(p)) = self.egraph.node(t.node).map(|n| n.kind.clone()) else {
                        fail("the node is gone".into())
                    };
                    t.old_key = p.solve.recipe;
                    // The overlay has points on screen at a middle frame.
                    let pts = self.camtrack_overlay(t.node, 18).unwrap_or_default();
                    if pts.len() < 200 {
                        fail(format!("{} solved points over the viewer, wanted hundreds", pts.len()));
                    }
                    // A Camera reading the solve.
                    self.ct_make_camera(t.node);
                    let cam = *self.selected.first().unwrap_or(&0);
                    t.cam = cam;
                    self.camtrack_sync();
                    for f in [0u64, 17, 35] {
                        let g = self.egraph.to_engine_graph_at(Some(f));
                        let Some(NodeKind::Camera3D(c)) = g.nodes.get(&cam).cloned() else {
                            fail("no Camera in the engine graph".into())
                        };
                        let want = p.camera_at(f as f32).unwrap_or_else(|| fail("no camera_at".into()));
                        let d = (0..3).map(|k| (c.xform.translate[k] - want.0.translate[k]).abs()).fold(0.0f32, f32::max);
                        if d > 1e-5 || (c.focal - want.1.focal).abs() > 1e-4 {
                            fail(format!("frame {f}: the Camera reads {:?} / {} for the solve's {:?} / {}", c.xform.translate, c.focal, want.0.translate, want.1.focal));
                        }
                    }
                    // A lens change: STALE, and the re-solve needs no render.
                    if let Some(NodeKind::CamTrack(p)) = self.egraph.node_mut(t.node).map(|n| &mut n.kind) {
                        p.refine_focal = true;
                        p.focal *= 1.1;
                    }
                    if self.ct_view(t.node).loaded != Some((N, false)) {
                        fail("a lens change did not mark the solve STALE".into());
                    }
                    self.ct_action(t.node, Action::Dismiss);
                    self.ct_action(t.node, Action::Analyse);
                    if matches!(self.camtrack.run.as_ref().map(|r| &r.phase), Some(Phase::Render)) && self.graph_bake.is_some() {
                        fail("a lens change re-rendered the input".into());
                    }
                    if self.camtrack.tracking {
                        fail("a lens change re-tracked".into());
                    }
                    next(&mut t, 3);
                }
                _ => {}
            },
            3 => match self.camtrack.run.as_ref().map(|r| &r.phase) {
                Some(Phase::Failed(e)) => fail(format!("the re-solve failed: {e}")),
                Some(Phase::Done(_)) => {
                    let (rel, rot, f) = judge(self, &t);
                    let Some(NodeKind::CamTrack(p)) = self.egraph.node(t.node).map(|n| &n.kind) else {
                        fail("the node is gone".into())
                    };
                    if p.solve.recipe == t.old_key {
                        fail("the re-solve kept the old key".into());
                    }
                    let ferr = (f - t.focal_px).abs() / t.focal_px;
                    println!(
                        "camtrack: refined from a 10% wrong lens: focal {:.1} px for {:.1} ({:.2}% off), centres within {:.2}%, rotation within {rot:.3}°",
                        f, t.focal_px, ferr * 100.0, rel * 100.0
                    );
                    if ferr > 0.03 {
                        fail(format!("the refined focal is {:.2}% off", ferr * 100.0));
                    }
                    if rel > 0.05 || rot > 1.0 {
                        fail(format!("the refined solve is {:.2}% / {rot:.3}° off", rel * 100.0));
                    }
                    t.old_key = p.solve.recipe;
                    // A guess from nothing: the lens set to guess.
                    if let Some(NodeKind::CamTrack(p)) = self.egraph.node_mut(t.node).map(|n| &mut n.kind) {
                        p.guess = true;
                        p.focal = 300.0;
                    }
                    self.ct_action(t.node, Action::Dismiss);
                    self.ct_action(t.node, Action::Analyse);
                    next(&mut t, 4);
                }
                _ => {}
            },
            4 => match self.camtrack.run.as_ref().map(|r| &r.phase) {
                Some(Phase::Failed(e)) => fail(format!("the guess failed: {e}")),
                Some(Phase::Done(_)) => {
                    let (rel, rot, f) = judge(self, &t);
                    let Some(NodeKind::CamTrack(p)) = self.egraph.node(t.node).map(|n| &n.kind) else {
                        fail("the node is gone".into())
                    };
                    if p.solve.recipe == t.old_key {
                        fail("the guess kept the refined solve's key".into());
                    }
                    let ferr = (f - t.focal_px).abs() / t.focal_px;
                    println!("camtrack: guessed the lens from nothing: focal {f:.1} px for {:.1} ({:.2}% off)", t.focal_px, ferr * 100.0);
                    if ferr > 0.03 {
                        fail(format!("the guessed focal is {:.2}% off", ferr * 100.0));
                    }
                    // Picks and the alignment: three floor points (the
                    // synthetic floor is y = 1.2 in the first camera's CV
                    // frame; y/z > 0.35 is floor and nothing else) make the
                    // ground, every other floor point lands on it and the
                    // camera stands above; a card on them lies on it; two
                    // picks a known distance apart set the scale.
                    let s = p.solve.solve.clone();
                    let ratio = |x: &[f64; 3]| if x[2] > 1e-6 { x[1] / x[2] } else { 0.0 };
                    let floor: Vec<usize> = (0..s.points.len()).filter(|&k| ratio(&s.points[k]) > 0.35).collect();
                    if floor.len() < 10 {
                        fail(format!("{} floor points solved, wanted dozens", floor.len()));
                    }
                    let picks = [floor[0], floor[floor.len() / 2], floor[floor.len() - 1]];
                    let Some(NodeKind::CamTrack(pm)) = self.egraph.node_mut(t.node).map(|n| &mut n.kind) else {
                        fail("the node is gone".into())
                    };
                    for k in picks {
                        pm.toggle_pick(k);
                    }
                    if pm.picks().len() != 3 || !pm.set_ground() {
                        fail("the ground could not be set from three picks".into());
                    }
                    let pm = pm.clone();
                    let mut worst = 0.0f32;
                    for &k in &floor {
                        worst = worst.max(pm.point_world(k)[1].abs());
                    }
                    let (cam, _) = pm.camera_at(17.0).unwrap_or_else(|| fail("no camera".into()));
                    println!("camtrack: ground from three picks: {} floor points within {worst:.4} of y = 0, camera at y {:.3}", floor.len(), cam.translate[1]);
                    if worst > 0.05 {
                        fail(format!("a floor point sits {worst:.4} off the ground"));
                    }
                    if cam.translate[1] < 0.1 {
                        fail(format!("the camera is not above the ground: y {}", cam.translate[1]));
                    }
                    self.ct_make_card(t.node);
                    let card = *self.selected.first().unwrap_or(&0);
                    let Some(NodeKind::Card3D(c)) = self.egraph.node(card).map(|n| n.kind.clone()) else {
                        fail("card… made no Card".into())
                    };
                    let r = engine::scene3d::rotation(c.xform.rotate);
                    if c.xform.translate[1].abs() > 0.05 || r[1][2] < 0.99 {
                        fail(format!("the card is not on the ground: at {:?}, normal y {}", c.xform.translate, r[1][2]));
                    }
                    if !self.egraph.input_of(t.cam, 0).is_none() {
                        fail("a Camera grew an input".into());
                    }
                    if let Some(NodeKind::CamTrack(pm)) = self.egraph.node_mut(t.node).map(|n| &mut n.kind) {
                        pm.picked = vec![floor[0], floor[floor.len() - 1]];
                        pm.ruler = 3.0;
                        if !pm.set_distance() {
                            fail("two picks could not set the scale".into());
                        }
                        let (a, b) = (pm.point_world(floor[0]), pm.point_world(floor[floor.len() - 1]));
                        let d = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
                        if (d - 3.0).abs() > 1e-3 {
                            fail(format!("the two picks are {d} apart, wanted 3"));
                        }
                    }
                    if let Some(m) = &t.models {
                        let _ = std::fs::remove_dir_all(m);
                    }
                    println!(
                        "CAMTRACK TEST PASS: solved a {N}-frame three-plane dolly with SEA-RAFT tracks through the app \
                         (centres within {:.2}% of the travel, rotation within {rot:.3}°); the landing is no undo step, \
                         the node is cyan, a Camera reads the solve per frame, a lens change re-solves off the tracks on disk, \
                         the lens is guessed from nothing to {:.2}%, three picks make the ground and a card on it, two set the scale",
                        rel * 100.0,
                        ferr * 100.0
                    );
                    std::process::exit(0);
                }
                _ => {}
            },
            _ => {}
        }
        self.test_camtrack = Some(t);
    }
}
