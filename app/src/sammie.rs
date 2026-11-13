//! The Sammie node's app half: where the tool lives, installing it, the
//! viewer clicks, rendering the plate to a shot directory, running SAM 2
//! (and MatAnyone) as a process, and adopting the matte it wrote. The node
//! itself (params, codegen) is `engine::nodes::sammie`; the panel and glyph
//! are `crate::nodes::sammie`. The process plumbing is CorridorKey's
//! (`crate::corridorkey::spawn`), the same protocol line for line.
//!
//! A run is two phases, both explicit and cancellable:
//!
//! 1. **plate** — a `SubtreeBake` of input 0 over the plate's frame range,
//!    linear Rec.709 EXRs in `<dir>/Input`, plus `<dir>/points.json`;
//! 2. **track** — `tools/wobble_sammie.py` (embedded, written next to the
//!    checkout every run) segments from the clicks, tracks through the
//!    shot, optionally mattes, and writes `<dir>/Matte`, adopted as the
//!    node's reader when it says done.
//!
//! `<dir>` is keyed on the plate's subtree key, the recipe (model, matting,
//! every click) and the range: a new click shows the loaded matte as STALE
//! and a re-run lands in a new directory (the GraphBake contract). A
//! finished run is adopted on load (`sammie_adopt`).
//!
//! **Nothing installs or downloads without a click.** `install…` clones
//! Sammie-Roto 2 into `~/.wobble/Sammie-Roto-2` (or `$WOBBLE_SAMMIE_DIR`)
//! and runs `uv sync` there with the torch build for this machine; the
//! driver fetches the picked model's weights on the first track, through
//! the tool's own downloader, and the panel says so before the button.

use crate::corridorkey::{push_log, spawn, uv, Msg, UV_MISSING};
use crate::Gfx;
use engine::graph::{NodeKind, SamPoint, SammieParams};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;

/// The driver, written next to the checkout on every run.
pub const DRIVER: &str = include_str!("../../tools/wobble_sammie.py");
const DRIVER_NAME: &str = "wobble_sammie.py";
pub const REPO: &str = "https://github.com/Zarxrax/Sammie-Roto-2.git";
pub const LICENCE_URL: &str = "https://github.com/Zarxrax/Sammie-Roto-2/blob/main/LICENSE";
pub const MATANYONE_LICENCE_URL: &str = "https://github.com/pq-yang/MatAnyone2/blob/main/LICENSE";

/// Where the tool's checkout lives.
pub fn install_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("WOBBLE_SAMMIE_DIR") {
        return PathBuf::from(d);
    }
    engine::paths::home().unwrap_or_else(|| PathBuf::from(".")).join(".wobble").join("Sammie-Roto-2")
}

/// A checkout with its environment built.
pub fn installed(dir: &Path) -> bool {
    dir.join("sam2").join("build_sam.py").is_file() && dir.join(".venv").is_dir()
}

fn stand_in() -> bool {
    std::env::var_os("WOBBLE_SAMMIE_CMD").is_some()
}

/// Which torch the environment gets: Apple's on a Mac, CUDA where an
/// NVIDIA driver answers, else the CPU build. `$WOBBLE_SAMMIE_EXTRA`
/// overrides (the tool's extras: macos, cpu, cu126, cu130, rocm, xpu).
pub fn torch_extra() -> String {
    if let Ok(e) = std::env::var("WOBBLE_SAMMIE_EXTRA") {
        return e;
    }
    if cfg!(target_os = "macos") {
        return "macos".into();
    }
    let nvidia = std::process::Command::new("nvidia-smi")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success());
    if nvidia { "cu126" } else { "cpu" }.into()
}

/// Where a run is.
pub enum Phase {
    /// Rendering the plate (a `SubtreeBake` in `Gfx::graph_bake`).
    Plate,
    Tracking { rx: Receiver<Msg>, cancel: Arc<AtomicBool>, done: usize, total: usize },
    Failed(String),
    Done(String),
}

pub struct Run {
    pub node: u64,
    pub dir: PathBuf,
    pub phase: Phase,
    pub log: Vec<String>,
    /// Made with non-commercial weights: what `nc_take` becomes on adopt.
    nc: bool,
    /// A PREVIEW: one frame segmented from its own clicks, shown over the
    /// viewer instead of adopted. (comp frame, preview key)
    preview: Option<(i64, u64)>,
}

impl Run {
    pub fn active(&self) -> bool {
        matches!(self.phase, Phase::Plate | Phase::Tracking { .. })
    }
}

pub struct Install {
    rx: Receiver<Msg>,
    pub log: Vec<String>,
    pub result: Option<Result<(), String>>,
}

#[derive(Default)]
pub struct Jobs {
    pub run: Option<Run>,
    pub install: Option<Install>,
    /// The last preview each node made.
    pub previews: std::collections::HashMap<u64, Preview>,
    /// The last click edit, for the auto preview's debounce.
    pub edited: Option<(u64, web_time::Instant)>,
    /// The segmenter: one process holding SAM loaded, so a preview is one
    /// image embedding per frame and a mask decode per click, not a model
    /// load per click.
    pub server: Option<Server>,
}

/// The driver in `--serve` mode: one JSON request a line on stdin, one
/// `WOBBLE done` / `WOBBLE error` a request on stdout. Replies go to
/// whichever request is current; the process dies with the app.
pub struct Server {
    model: engine::graph::SamModel,
    child: std::process::Child,
    stdin: std::process::ChildStdin,
    current: Arc<std::sync::Mutex<Option<Sender<Msg>>>>,
    alive: Arc<AtomicBool>,
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Server {
    fn start(model: engine::graph::SamModel) -> Result<Server, String> {
        use std::io::BufRead;
        use std::process::Stdio;
        let mut cmd = command(&install_dir())?;
        cmd.arg("--serve").arg("--model").arg(model.arg());
        cmd.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
        let mut child = cmd.spawn().map_err(|e| format!("the segmenter: {e}"))?;
        let stdin = child.stdin.take().ok_or("no stdin")?;
        let out = child.stdout.take().ok_or("no stdout")?;
        let err = child.stderr.take().ok_or("no stderr")?;
        let current: Arc<std::sync::Mutex<Option<Sender<Msg>>>> = Default::default();
        let alive = Arc::new(AtomicBool::new(true));
        let send = |cur: &Arc<std::sync::Mutex<Option<Sender<Msg>>>>, m: Msg| {
            if let Some(tx) = cur.lock().ok().and_then(|c| c.clone()) {
                let _ = tx.send(m);
            }
        };
        let c2 = current.clone();
        let _ = std::thread::Builder::new().name("sammie-serve-err".into()).spawn(move || {
            for l in std::io::BufReader::new(err).lines().map_while(Result::ok) {
                send(&c2, Msg::Line(l));
            }
        });
        let (c3, a3) = (current.clone(), alive.clone());
        std::thread::Builder::new()
            .name("sammie-serve".into())
            .spawn(move || {
                for l in std::io::BufReader::new(out).lines().map_while(Result::ok) {
                    let rest = l.strip_prefix("WOBBLE ").map(str::trim);
                    send(
                        &c3,
                        match rest {
                            Some("done") => Msg::Done,
                            Some(r) if r.starts_with("error") => Msg::Failed(r.trim_start_matches("error").trim().into()),
                            Some(r) if r == "ready" => Msg::Line("model loaded".into()),
                            _ => Msg::Line(l),
                        },
                    );
                }
                a3.store(false, Relaxed);
                send(&c3, Msg::Failed("the segmenter stopped".into()));
            })
            .map_err(|e| format!("{e}"))?;
        Ok(Server { model, child, stdin, current, alive })
    }

    /// Ask for one mask; its replies arrive on the returned channel.
    fn ask(&mut self, req: &serde_json::Value) -> Result<Receiver<Msg>, String> {
        use std::io::Write;
        let (tx, rx) = std::sync::mpsc::channel();
        if let Ok(mut c) = self.current.lock() {
            *c = Some(tx);
        }
        writeln!(self.stdin, "{req}").and_then(|_| self.stdin.flush()).map_err(|e| format!("the segmenter: {e}"))?;
        Ok(rx)
    }
}

/// One frame's segment, as an overlay texture.
pub struct Preview {
    pub key: u64,
    pub tick: i64,
    pub tex: egui::TextureHandle,
    /// Fraction of the frame covered.
    pub coverage: f32,
}

/// The preview as the panel and the viewer see it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PreviewState {
    None,
    /// Made from the clicks as they are: (frame, coverage).
    Current(i64, f32),
    /// The clicks on its frame changed since.
    Stale(i64),
}

/// What the panel is told (read-only), built by `Gfx::sammie_view`.
pub struct View {
    pub dir: String,
    pub installed: bool,
    pub uv: bool,
    pub installing: Option<String>,
    pub install_result: Option<Result<(), String>>,
    pub state: ViewState,
    pub log: Vec<String>,
    pub has_input: bool,
    /// Frames loaded in the reader, and whether they are THIS recipe's.
    pub loaded: Option<(usize, bool)>,
    pub on_disk: bool,
    pub busy_elsewhere: bool,
    /// The comp frame on screen, for the clicks summary.
    pub tick: i64,
    pub preview: PreviewState,
    /// The run in flight is a preview.
    pub previewing: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ViewState {
    Idle,
    Rendering(u64, u64),
    Tracking(usize, usize),
    Failed(String),
    Done(String),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Action {
    Track,
    /// Segment the frame on screen from its clicks, and show it.
    Preview,
    Cancel,
    Install,
    Dismiss,
    /// Drop the clicks on the frame on screen.
    ClearFrame,
    ClearAll,
    /// Put the playhead on a clicked frame.
    Seek(i64),
}

pub fn params(kind: &NodeKind) -> Option<&SammieParams> {
    match kind {
        NodeKind::Sammie(p) => Some(p),
        _ => None,
    }
}

/// The command that runs the driver: `$WOBBLE_SAMMIE_CMD` (a suite's
/// stand-in, whitespace-split) or `uv run --project <dir> python <driver>`.
fn command(dir: &Path) -> Result<std::process::Command, String> {
    if let Ok(s) = std::env::var("WOBBLE_SAMMIE_CMD") {
        let mut parts = s.split_whitespace();
        let exe = parts.next().ok_or("WOBBLE_SAMMIE_CMD is empty")?;
        let mut c = std::process::Command::new(exe);
        c.args(parts);
        return Ok(c);
    }
    let uv = uv().ok_or(UV_MISSING)?;
    let driver = dir.join(DRIVER_NAME);
    std::fs::write(&driver, DRIVER).map_err(|e| format!("writing the driver: {e}"))?;
    let mut c = std::process::Command::new(uv);
    c.arg("run").arg("--project").arg(dir).arg("python").arg(&driver);
    c.current_dir(dir);
    Ok(c)
}

/// The clicks file the driver reads.
pub fn points_json(points: &[SamPoint]) -> String {
    let pts: Vec<serde_json::Value> = points
        .iter()
        .map(|p| serde_json::json!({ "frame": p.frame, "x": p.x, "y": p.y, "positive": p.positive }))
        .collect();
    serde_json::json!({ "points": pts }).to_string()
}

// ------------------------------------------------------------- the viewer

/// A viewer click's verdict on the node's points.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Edit {
    Add(SamPoint),
    Remove(usize),
    Move(usize, [f32; 2]),
}

/// The points on frame `t`: drawn bright and editable. `to_scr` maps a
/// normalised position to the screen and `from_screen` a screen point back
/// to it. S-click adds an include point, shift+S-click an exclude point,
/// D-click on a point removes it, a drag moves it.
#[allow(clippy::too_many_arguments)]
pub fn gizmo(
    ui: &mut egui::Ui,
    ink: &mut crate::ink::InkShapes,
    node: u64,
    p: &SammieParams,
    t: i64,
    nav: &egui::Response,
    to_scr: &dyn Fn([f32; 2]) -> egui::Pos2,
    from_screen: &dyn Fn(egui::Pos2) -> [f32; 2],
    pointer_pos: Option<egui::Pos2>,
    preview: Option<(egui::TextureId, bool)>,
) -> Option<Edit> {
    let mut edit = None;
    // The preview segment, under the points: green where SAM sees the
    // subject; grey once a click on this frame has changed since.
    if let Some((tex, current)) = preview {
        let r = egui::Rect::from_two_pos(to_scr([0.0, 0.0]), to_scr([1.0, 1.0]));
        let tint = if current { egui::Color32::WHITE } else { egui::Color32::from_gray(90) };
        ui.painter().image(tex, r, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), tint);
    }
    let d_held = ui.input(|i| i.key_down(egui::Key::D));
    for (i, q) in p.points.iter().enumerate().filter(|(_, q)| q.frame == t) {
        let sp = to_scr([q.x, q.y]);
        let id = ui.id().with(("sammie-pt", node, i));
        let hr = ui.interact(egui::Rect::from_center_size(sp, egui::vec2(16.0, 16.0)), id, egui::Sense::click_and_drag());
        // Where the point was at the press: the drag is measured from it.
        if hr.drag_started() {
            ui.data_mut(|d| d.insert_temp::<[f32; 2]>(id, [q.x, q.y]));
        }
        // Include points green, exclude red, both inked so they read on any
        // plate.
        ink.set_tint(Some(if q.positive { [0.35, 1.0, 0.45] } else { [1.0, 0.3, 0.3] }));
        if q.positive {
            ink.circle_fill(sp, 4.5, 1.0, hr.hovered());
        } else {
            ink.seg(sp - egui::vec2(4.5, 4.5), sp + egui::vec2(4.5, 4.5), 1.6, 1.0, hr.hovered());
            ink.seg(sp - egui::vec2(4.5, -4.5), sp + egui::vec2(4.5, -4.5), 1.6, 1.0, hr.hovered());
        }
        ink.circle_stroke(sp, 7.5, 1.0, 0.8, hr.hovered());
        if hr.clicked() && d_held {
            edit = Some(Edit::Remove(i));
        } else if hr.dragged() {
            if let Some(pp) = pointer_pos {
                // Absolute from the press, never summed deltas.
                let origin = ui.input(|inp| inp.pointer.press_origin()).unwrap_or(pp);
                let base = ui.data(|d| d.get_temp::<[f32; 2]>(id)).unwrap_or([q.x, q.y]);
                let a = from_screen(origin);
                let b = from_screen(pp);
                let at = [
                    (base[0] + b[0] - a[0]).clamp(0.0, 1.0),
                    (base[1] + b[1] - a[1]).clamp(0.0, 1.0),
                ];
                edit = Some(Edit::Move(i, at));
            }
        }
    }
    ink.set_tint(None);
    // Points on other frames: a faint ring each, so you can see where the
    // prompts are without being able to edit them here.
    for q in p.points.iter().filter(|q| q.frame != t) {
        ink.circle_stroke(to_scr([q.x, q.y]), 3.0, 0.8, 0.3, false);
    }
    let s_held = ui.input(|i| i.key_down(egui::Key::S));
    if edit.is_none() && s_held && nav.clicked_by(egui::PointerButton::Primary) {
        if let Some(pp) = nav.interact_pointer_pos() {
            let q = from_screen(pp);
            if (0.0..=1.0).contains(&q[0]) && (0.0..=1.0).contains(&q[1]) {
                let positive = !ui.input(|i| i.modifiers.shift);
                edit = Some(Edit::Add(SamPoint { frame: t, x: q[0], y: q[1], positive }));
            }
        }
    }
    edit
}

/// What the edit does to the clicks.
pub fn apply(points: &mut Vec<SamPoint>, e: Edit) {
    match e {
        Edit::Add(q) => points.push(q),
        Edit::Remove(i) if i < points.len() => {
            points.remove(i);
        }
        Edit::Move(i, at) if i < points.len() => {
            points[i].x = at[0];
            points[i].y = at[1];
        }
        _ => {}
    }
}

impl Gfx {
    /// Where a run of `node` at the current recipe lands, and its key. None
    /// while the plate cannot be rendered.
    pub(crate) fn sammie_dir(&self, node: u64) -> Option<(u64, PathBuf)> {
        let p = self.egraph.node(node).and_then(|n| params(&n.kind))?;
        let input = self.egraph.input_of(node, 0)?;
        let k0 = self.subtree_key_of(input)?;
        let (from, len) = self.sammie_range(input);
        let recipe = serde_json::to_vec(&(p.recipe(), from, len)).ok()?;
        let key = engine::diskcache::content_hash(&[&k0.to_le_bytes(), &recipe]);
        let root = match &self.disk_cache {
            Some(c) => c.root().join("sammie"),
            None => engine::paths::temp_dir().join("wobble-sammie"),
        };
        Some((key, root.join(format!("{key:016x}"))))
    }

    /// The frames a track covers: the plate's own range, cut to the
    /// script range when one is set — NO handles. Handles are for a bake
    /// that may be slipped later; a roto runs on the shot you asked for
    /// (a 67-frame trim of a long clip rendered 247 with them).
    pub(crate) fn sammie_range(&self, input: u64) -> (u64, u64) {
        let ((mut a, mut b), _) = self.range_of(input);
        if let Some((ca, cb)) = self.comp_range {
            if ca.max(a) <= cb.min(b) {
                a = a.max(ca);
                b = b.min(cb);
            }
        }
        (a, b + 1)
    }

    pub(crate) fn sammie_view(&self, node: u64) -> View {
        let dir = install_dir();
        let mine = self.sammie.run.as_ref().filter(|r| r.node == node);
        let state = match mine.map(|r| &r.phase) {
            None => ViewState::Idle,
            Some(Phase::Plate) => match &self.graph_bake {
                Some(b) if b.node == node => ViewState::Rendering(b.next.saturating_sub(b.from), b.len.saturating_sub(b.from)),
                _ => ViewState::Rendering(0, 0),
            },
            Some(Phase::Tracking { done, total, .. }) => ViewState::Tracking(*done, *total),
            Some(Phase::Failed(e)) => ViewState::Failed(e.clone()),
            Some(Phase::Done(s)) => ViewState::Done(s.clone()),
        };
        let want = self.sammie_dir(node).map(|(_, d)| d.join("Matte"));
        let loaded = self.loaded.get(&node).map(|l| {
            let current = want.as_ref().is_some_and(|d| l.path == d.to_string_lossy());
            (l.seq.frames.len(), current)
        });
        View {
            dir: dir.to_string_lossy().into_owned(),
            installed: installed(&dir) || stand_in(),
            uv: uv().is_some(),
            installing: self.sammie.install.as_ref().filter(|i| i.result.is_none()).map(|i| i.log.last().cloned().unwrap_or_default()),
            install_result: self.sammie.install.as_ref().and_then(|i| i.result.clone()),
            state,
            log: mine.map(|r| r.log.clone()).unwrap_or_default(),
            has_input: self.egraph.input_of(node, 0).is_some(),
            loaded,
            on_disk: want.as_ref().is_some_and(|d| engine::bakes::is_baked(d)),
            busy_elsewhere: self.sammie.run.as_ref().is_some_and(|r| r.node != node && r.active()),
            tick: self.timeline_pos as i64,
            preview: self.sammie_preview_state(node),
            previewing: mine.is_some_and(|r| r.preview.is_some() && r.active()),
        }
    }

    pub(crate) fn sammie_action(&mut self, node: u64, a: Action) {
        match a {
            Action::Track => self.sammie_start(node),
            Action::Preview => self.sammie_preview(node),
            Action::Cancel => {
                if let Some(r) = self.sammie.run.take() {
                    match r.phase {
                        Phase::Tracking { cancel, .. } => cancel.store(true, Relaxed),
                        Phase::Plate => {
                            if self.graph_bake.as_ref().is_some_and(|b| b.node == r.node) {
                                self.graph_bake = None;
                            }
                        }
                        _ => {}
                    }
                }
            }
            Action::Install => self.sammie_install(),
            Action::Dismiss => {
                if self.sammie.run.as_ref().is_some_and(|r| r.node == node && !r.active()) {
                    self.sammie.run = None;
                }
                if self.sammie.install.as_ref().is_some_and(|i| i.result.is_some()) {
                    self.sammie.install = None;
                }
            }
            Action::ClearFrame | Action::ClearAll => {
                let t = self.timeline_pos as i64;
                if let Some(NodeKind::Sammie(p)) = self.egraph.node_mut(node).map(|n| &mut n.kind) {
                    p.points.retain(|q| a == Action::ClearFrame && q.frame != t);
                }
            }
            Action::Seek(t) => self.seek_to(t.max(0) as u64),
        }
    }

    /// `install…`: clone the tool and build its environment, on a thread,
    /// its output streamed to the panel.
    fn sammie_install(&mut self) {
        if self.sammie.install.as_ref().is_some_and(|i| i.result.is_none()) {
            return;
        }
        let dir = install_dir();
        let (tx, rx) = std::sync::mpsc::channel();
        let failed = |e: &str| Install { rx: std::sync::mpsc::channel().1, log: vec![], result: Some(Err(e.into())) };
        if cfg!(target_arch = "wasm32") {
            self.sammie.install = Some(failed("the tool runs as a process: not in the web build"));
            return;
        }
        let Some(uv) = uv() else {
            self.sammie.install = Some(failed(UV_MISSING));
            return;
        };
        let extra = torch_extra();
        let spawned = std::thread::Builder::new()
            .name("sammie-install".into())
            .spawn(move || {
                let result = (|| -> Result<(), String> {
                    if !dir.join("sam2").is_dir() {
                        if let Some(parent) = dir.parent() {
                            std::fs::create_dir_all(parent).map_err(|e| format!("{e}"))?;
                        }
                        let _ = tx.send(Msg::Line(format!("git clone {REPO} {}", dir.display())));
                        let mut c = std::process::Command::new("git");
                        c.arg("clone").arg("--depth").arg("1").arg(REPO).arg(&dir);
                        run_logged(c, &tx).map_err(|e| format!("git clone: {e}"))?;
                    }
                    let _ = tx.send(Msg::Line(format!("uv sync --extra {extra} (torch: several GB, a while)")));
                    let mut c = std::process::Command::new(&uv);
                    // The tool's own lock and Python, exactly as its
                    // installer does it.
                    c.args(["sync", "--frozen", "--python", "3.12", "--python-preference", "only-managed", "--extra"])
                        .arg(&extra)
                        .current_dir(&dir);
                    run_logged(c, &tx).map_err(|e| format!("uv sync: {e}"))
                })();
                let _ = tx.send(match result {
                    Ok(()) => Msg::Done,
                    Err(e) => Msg::Failed(e),
                });
            })
            .is_ok();
        self.sammie.install = Some(Install {
            rx,
            log: vec![],
            result: (!spawned).then_some(Err("could not start a thread".into())),
        });
    }

    /// The `track` click: adopt a finished run, else render the plate.
    fn sammie_start(&mut self, node: u64) {
        let fail = |me: &mut Self, e: &str| {
            me.sammie.run = Some(Run { node, dir: PathBuf::new(), phase: Phase::Failed(e.into()), log: vec![], nc: false, preview: None });
        };
        if self.sammie.run.as_ref().is_some_and(|r| r.active()) {
            return;
        }
        if cfg!(target_arch = "wasm32") {
            return fail(self, "tracking runs a process: not in the web build");
        }
        let Some(input) = self.egraph.input_of(node, 0) else {
            return fail(self, "wire the plate into the input");
        };
        let Some(p) = self.egraph.node(node).and_then(|n| params(&n.kind)).cloned() else { return };
        if !p.points.iter().any(|q| q.positive) {
            return fail(self, "S-click the subject in the viewer first");
        }
        let Some((_, dir)) = self.sammie_dir(node) else {
            return fail(self, "the plate is not renderable yet");
        };
        let nc = !p.matting.commercial();
        let out = dir.join("Matte");
        if engine::bakes::is_baked(&out) {
            self.sammie_load(node, &out, nc);
            self.sammie.run = Some(Run { node, dir, phase: Phase::Done("loaded the matte on disk".into()), log: vec![], nc, preview: None });
            return;
        }
        if !installed(&install_dir()) && !stand_in() {
            return fail(self, "Sammie is not installed: press install…");
        }
        let sub = dir.join("Input");
        if let Err(e) = std::fs::create_dir_all(&sub)
            .and_then(|_| std::fs::write(dir.join("points.json"), points_json(&p.points)))
        {
            return fail(self, &format!("{e}"));
        }
        self.sammie.run = Some(Run { node, dir: dir.clone(), phase: Phase::Plate, log: vec![], nc, preview: None });
        let range = self.sammie_range(input);
        if let Err(e) = self.sammie_render(node, input, dir, range) {
            fail(self, &e);
        }
    }

    /// A `SubtreeBake` of the plate into `<dir>/Input`.
    fn sammie_render(&mut self, node: u64, input: u64, dir: PathBuf, range: (u64, u64)) -> Result<(), String> {
        let sub = dir.join("Input");
        let key = self.subtree_key_of(input).ok_or("the plate is not renderable yet")?;
        let (w, h) = self.comp.output_size().ok_or("nothing is rendering yet")?;
        let (from, len) = range;
        self.playing = false;
        // Resume: skip the frames a previous run left.
        let mut next = from;
        while next < len && sub.join(crate::corridorkey::frame_name(next)).exists() {
            next += 1;
        }
        let pb = self.playback.as_mut().ok_or("load a sequence first")?;
        pb.player.seek(next.min(len.saturating_sub(1)));
        let info = engine::decode::ImageInfo { width: w, height: h };
        let readback = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sammie-render"),
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
            sink: crate::BakeSink::Sammie,
        };
        if bake.next >= bake.len {
            self.finish_graph_bake(bake);
        } else {
            self.graph_bake = Some(bake);
        }
        Ok(())
    }

    /// The bake loop finished the plate: run the tool (a track), or ask
    /// the segmenter (a preview).
    pub(crate) fn sammie_render_finished(&mut self, node: u64) {
        let Some(r) = self.sammie.run.as_ref().filter(|r| r.node == node) else { return };
        let dir = r.dir.clone();
        let Some(p) = self.egraph.node(node).and_then(|n| params(&n.kind)).cloned() else {
            return self.sammie_fail(node, "the node is gone");
        };
        if let Some((tick, _)) = r.preview {
            return self.sammie_ask(node, &p, tick, &dir);
        }
        let mut cmd = match command(&install_dir()) {
            Ok(c) => c,
            Err(e) => return self.sammie_fail(node, &e),
        };
        cmd.arg("--shot").arg(&dir).arg("--model").arg(p.model.arg()).arg("--matting").arg(p.matting.arg());
        let (tx, rx) = std::sync::mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        if let Err(e) = spawn(cmd, tx, cancel.clone(), "sammie") {
            return self.sammie_fail(node, &e);
        }
        if let Some(r) = self.sammie.run.as_mut() {
            r.phase = Phase::Tracking { rx, cancel, done: 0, total: 0 };
        }
    }

    /// One mask from the segmenter, started (or restarted for another
    /// model) on demand.
    fn sammie_ask(&mut self, node: u64, p: &SammieParams, tick: i64, dir: &Path) {
        let Some(frames) = self.sammie_frames_dir(node) else {
            return self.sammie_fail(node, "the plate is not renderable yet");
        };
        let name = crate::corridorkey::frame_name(tick.max(0) as u64);
        let points: Vec<SamPoint> = p.points.iter().filter(|q| q.frame == tick).copied().collect();
        let req = serde_json::json!({
            "image": frames.join("Input").join(&name),
            "out": dir.join("Matte").join(&name),
            "points": serde_json::from_str::<serde_json::Value>(&points_json(&points)).map(|v| v["points"].clone()).unwrap_or_default(),
        });
        let stale = self.sammie.server.as_ref().is_some_and(|s| s.model != p.model || !s.alive.load(Relaxed));
        if stale {
            self.sammie.server = None;
        }
        if self.sammie.server.is_none() {
            match Server::start(p.model) {
                Ok(s) => self.sammie.server = Some(s),
                Err(e) => return self.sammie_fail(node, &e),
            }
        }
        let asked = self.sammie.server.as_mut().map(|s| s.ask(&req));
        match asked {
            Some(Ok(rx)) => {
                if let Some(r) = self.sammie.run.as_mut() {
                    r.phase = Phase::Tracking { rx, cancel: Arc::new(AtomicBool::new(false)), done: 0, total: 0 };
                }
            }
            Some(Err(e)) => {
                self.sammie.server = None;
                self.sammie_fail(node, &e);
            }
            None => {}
        }
    }

    /// The plate's frames, rendered one at a time for previews: keyed on
    /// the plate alone, so a frame is rendered once whatever is clicked.
    fn sammie_frames_dir(&self, node: u64) -> Option<PathBuf> {
        let k0 = self.subtree_key_of(self.egraph.input_of(node, 0)?)?;
        let root = match &self.disk_cache {
            Some(c) => c.root().join("sammie"),
            None => engine::paths::temp_dir().join("wobble-sammie"),
        };
        Some(root.join(format!("frames-{k0:016x}")))
    }

    fn sammie_fail(&mut self, node: u64, e: &str) {
        if let Some(r) = self.sammie.run.as_mut().filter(|r| r.node == node) {
            r.phase = Phase::Failed(e.to_string());
        }
    }

    /// Play `out` through the node, and record what its pixels were made
    /// with (rule 10: a matte made with MatAnyone stays tagged).
    fn sammie_load(&mut self, node: u64, out: &Path, nc: bool) {
        self.load_read(node, &out.to_string_lossy());
        self.playback_dirty = true;
        if let Some(NodeKind::Sammie(p)) = self.egraph.node_mut(node).map(|n| &mut n.kind) {
            p.nc_take = nc;
        }
    }

    /// Where a preview of `node` on frame `tick` lands, and its key: the
    /// plate's subtree key, the models and THAT frame's clicks — clicks on
    /// other frames do not change what this frame segments to.
    fn sammie_preview_dir(&self, node: u64, tick: i64) -> Option<(u64, PathBuf)> {
        let p = self.egraph.node(node).and_then(|n| params(&n.kind))?;
        let k0 = self.subtree_key_of(self.egraph.input_of(node, 0)?)?;
        let mut here = p.clone();
        here.points.retain(|q| q.frame == tick);
        let recipe = serde_json::to_vec(&(here.recipe(), tick, "preview")).ok()?;
        let key = engine::diskcache::content_hash(&[&k0.to_le_bytes(), &recipe]);
        let root = match &self.disk_cache {
            Some(c) => c.root().join("sammie"),
            None => engine::paths::temp_dir().join("wobble-sammie"),
        };
        Some((key, root.join(format!("preview-{key:016x}"))))
    }

    pub(crate) fn sammie_preview_state(&self, node: u64) -> PreviewState {
        let Some(pv) = self.sammie.previews.get(&node) else { return PreviewState::None };
        match self.sammie_preview_dir(node, pv.tick) {
            Some((k, _)) if k == pv.key => PreviewState::Current(pv.tick, pv.coverage),
            _ => PreviewState::Stale(pv.tick),
        }
    }

    /// What the viewer overlays for `node` on frame `tick`.
    pub(crate) fn sammie_overlay(&self, node: u64, tick: i64) -> Option<(egui::TextureId, bool)> {
        let pv = self.sammie.previews.get(&node).filter(|p| p.tick == tick)?;
        Some((pv.tex.id(), matches!(self.sammie_preview_state(node), PreviewState::Current(..))))
    }

    /// A click on the viewer changed `node`'s points.
    pub(crate) fn sammie_edited(&mut self, node: u64) {
        self.sammie.edited = Some((node, web_time::Instant::now()));
    }

    /// The `preview` click: the frame on screen, rendered alone and
    /// segmented from its own clicks — the same models as a track, so what
    /// it shows is what the track starts from.
    fn sammie_preview(&mut self, node: u64) {
        let fail = |me: &mut Self, e: &str| {
            me.sammie.run = Some(Run { node, dir: PathBuf::new(), phase: Phase::Failed(e.into()), log: vec![], nc: false, preview: None });
        };
        if self.sammie.run.as_ref().is_some_and(|r| r.active()) {
            return;
        }
        if cfg!(target_arch = "wasm32") {
            return fail(self, "a preview runs a process: not in the web build");
        }
        let tick = self.timeline_pos as i64;
        let Some(input) = self.egraph.input_of(node, 0) else {
            return fail(self, "wire the plate into the input");
        };
        let Some(p) = self.egraph.node(node).and_then(|n| params(&n.kind)).cloned() else { return };
        let here: Vec<SamPoint> = p.points.iter().filter(|q| q.frame == tick).copied().collect();
        if !here.iter().any(|q| q.positive) {
            return fail(self, "S-click the subject on this frame first");
        }
        let Some((key, dir)) = self.sammie_preview_dir(node, tick) else {
            return fail(self, "the plate is not renderable yet");
        };
        let run = Run { node, dir: dir.clone(), phase: Phase::Plate, log: vec![], nc: false, preview: Some((tick, key)) };
        // Made before (an undo back to these clicks): show it again.
        if dir.join("Matte").join(crate::corridorkey::frame_name(tick as u64)).is_file() {
            self.sammie.run = Some(Run { phase: Phase::Tracking { rx: done_now(), cancel: Arc::new(AtomicBool::new(false)), done: 0, total: 0 }, ..run });
            return;
        }
        if !installed(&install_dir()) && !stand_in() {
            return fail(self, "Sammie is not installed: press install…");
        }
        let Some(frames) = self.sammie_frames_dir(node) else {
            return fail(self, "the plate is not renderable yet");
        };
        if let Err(e) = std::fs::create_dir_all(dir.join("Matte")).and_then(|_| std::fs::create_dir_all(frames.join("Input"))) {
            return fail(self, &format!("{e}"));
        }
        self.sammie.run = Some(run);
        // The frame renders once (a rendered frame finishes at once).
        let t = tick.max(0) as u64;
        if let Err(e) = self.sammie_render(node, input, frames, (t, t + 1)) {
            fail(self, &e);
        }
    }

    /// Every redraw: drain the tool's messages, catch a render that died,
    /// adopt a finished matte.
    pub(crate) fn sammie_poll(&mut self) {
        if let Some(i) = self.sammie.install.as_mut() {
            while let Ok(m) = i.rx.try_recv() {
                match m {
                    Msg::Line(l) => push_log(&mut i.log, l),
                    Msg::Done => i.result = Some(Ok(())),
                    Msg::Failed(e) => i.result = Some(Err(e)),
                    Msg::Frame(..) => {}
                }
            }
        }
        // Auto preview: a click segments the frame on screen as soon as the
        // clicks have been still a moment (a point drag edits every frame),
        // and again after any click that landed while one was running.
        if let Some((node, at)) = self.sammie.edited {
            if at.elapsed().as_millis() >= 80 && !self.sammie.run.as_ref().is_some_and(|r| r.active()) {
                self.sammie.edited = None;
                let tick = self.timeline_pos as i64;
                let auto = self.egraph.node(node).and_then(|n| params(&n.kind)).is_some_and(|p| {
                    p.auto_preview && p.points.iter().any(|q| q.frame == tick && q.positive)
                });
                let current = matches!(self.sammie_preview_state(node), PreviewState::Current(f, _) if f == tick);
                if auto && !current {
                    self.sammie_preview(node);
                }
            }
        }
        let Some(r) = self.sammie.run.as_mut() else { return };
        let node = r.node;
        if matches!(r.phase, Phase::Plate) && self.graph_bake.is_none() {
            let why = self.load_errors.get(&node).cloned().unwrap_or_else(|| "the plate stopped rendering".into());
            r.phase = Phase::Failed(why);
            return;
        }
        let mut end = None;
        if let Phase::Tracking { rx, done, total, .. } = &mut r.phase {
            while let Ok(m) = rx.try_recv() {
                match m {
                    Msg::Frame(a, b) => {
                        *done = a;
                        *total = b;
                    }
                    Msg::Line(l) => push_log(&mut r.log, l),
                    Msg::Done => end = Some(Ok(())),
                    Msg::Failed(e) => end = Some(Err(e)),
                }
            }
        }
        match end {
            Some(Ok(())) if r.preview.is_some() => {
                let (tick, key) = r.preview.unwrap();
                let file = r.dir.join("Matte").join(crate::corridorkey::frame_name(tick.max(0) as u64));
                match preview_texture(&self.egui_ctx, &file, node) {
                    Ok((tex, coverage)) => {
                        r.phase = Phase::Done(format!("previewed frame {tick}: {:.1}% of the frame", coverage * 100.0));
                        self.sammie.previews.insert(node, Preview { key, tick, tex, coverage });
                    }
                    Err(e) => r.phase = Phase::Failed(format!("reading the preview: {e}")),
                }
            }
            Some(Ok(())) => {
                let out = r.dir.join("Matte");
                let nc = r.nc;
                if let Err(e) = engine::bakes::mark_baked(&out) {
                    r.phase = Phase::Failed(format!("marking the matte done: {e}"));
                    return;
                }
                r.phase = Phase::Done("tracked".into());
                self.sammie_load(node, &out, nc);
            }
            Some(Err(e)) => r.phase = Phase::Failed(e),
            None => {}
        }
    }

    /// After a load, an undo, a paste: every Sammie node whose run for the
    /// current recipe is on disk plays it.
    pub(crate) fn sammie_adopt(&mut self) {
        let ids: Vec<(u64, bool)> = self
            .egraph
            .nodes
            .iter()
            .filter(|(id, _)| !self.loaded.contains_key(id) && !self.load_errors.contains_key(id))
            .filter_map(|(id, n)| params(&n.kind).map(|p| (*id, !p.matting.commercial())))
            .collect();
        for (id, nc) in ids {
            let Some((_, dir)) = self.sammie_dir(id) else { continue };
            let out = dir.join("Matte");
            if engine::bakes::is_baked(&out) {
                self.sammie_load(id, &out, nc);
            }
        }
    }
}

/// A message channel that has already said done: a preview found on disk
/// goes through the same adoption as one just made.
fn done_now() -> Receiver<Msg> {
    let (tx, rx) = std::sync::mpsc::channel();
    let _ = tx.send(Msg::Done);
    rx
}

/// The preview frame as an overlay: the segment green, translucent,
/// premultiplied; nothing elsewhere. Large plates are sampled down — it is
/// a picture of the mask, not the mask.
fn preview_texture(ctx: &egui::Context, file: &Path, node: u64) -> Result<(egui::TextureHandle, f32), String> {
    use engine::decode::ExrDecoder;
    let dec = engine::decode::AnyDecoder::new(false);
    let info = dec.read_header(file).map_err(|e| format!("{e}"))?;
    let stride = info.row_stride_f16();
    let (w, h) = (info.width as usize, info.height as usize);
    let mut px = vec![half::f16::ZERO; stride * h];
    dec.decode_into(file, info, stride, &mut px).map_err(|e| format!("{e}"))?;
    let step = w.div_ceil(2048).max(1);
    let (tw, th) = (w.div_ceil(step), h.div_ceil(step));
    let mask: Vec<f32> = (0..tw * th)
        .map(|i| px[(i / tw * step) * stride + (i % tw * step) * 4].to_f32().clamp(0.0, 1.0))
        .collect();
    let inside = |x: isize, y: isize| {
        x >= 0 && y >= 0 && (x as usize) < tw && (y as usize) < th && mask[y as usize * tw + x as usize] > 0.5
    };
    let mut img: Vec<u8> = Vec::with_capacity(tw * th * 4);
    let mut covered = 0usize;
    for y in 0..th as isize {
        for x in 0..tw as isize {
            let m = mask[y as usize * tw + x as usize];
            let on = m > 0.5;
            covered += on as usize;
            // The outline, white, two pixels in from the edge: it reads on
            // any subject, the green fill does not on a green one.
            let edge = on && (1..=2).any(|r| !inside(x - r, y) || !inside(x + r, y) || !inside(x, y - r) || !inside(x, y + r));
            let px = if edge {
                [235, 255, 240, 235]
            } else {
                let a = m * 0.45;
                [(70.0 * a) as u8, (255.0 * a) as u8, (110.0 * a) as u8, (255.0 * a) as u8]
            };
            img.extend_from_slice(&px);
        }
    }
    let ci = egui::ColorImage::from_rgba_premultiplied([tw, th], &img);
    let tex = ctx.load_texture(format!("sammie-preview-{node}"), ci, egui::TextureOptions::LINEAR);
    Ok((tex, covered as f32 / (tw * th).max(1) as f32))
}

/// Run a command to completion, every line of its output to the log.
fn run_logged(mut c: std::process::Command, tx: &Sender<Msg>) -> Result<(), String> {
    use std::io::BufRead;
    use std::process::Stdio;
    c.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = c.spawn().map_err(|e| format!("{e}"))?;
    let err = child.stderr.take().unwrap();
    let tx2 = tx.clone();
    let _ = std::thread::spawn(move || {
        for l in std::io::BufReader::new(err).lines().map_while(Result::ok) {
            let _ = tx2.send(Msg::Line(l));
        }
    });
    let out = child.stdout.take().unwrap();
    for l in std::io::BufReader::new(out).lines().map_while(Result::ok) {
        let _ = tx.send(Msg::Line(l));
    }
    let s = child.wait().map_err(|e| format!("{e}"))?;
    if s.success() { Ok(()) } else { Err(format!("exited with {s}")) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_driver_speaks_the_protocol_the_reader_parses() {
        for w in ["WOBBLE frame", "WOBBLE done", "WOBBLE error", "--shot", "--model", "--matting", "points.json", "Matte"] {
            assert!(DRIVER.contains(w), "{w}");
        }
        // Every model the node offers is one the driver knows.
        for m in engine::graph::SamModel::ALL {
            assert!(DRIVER.contains(&format!("\"{}\"", m.arg())), "{}", m.arg());
        }
        for m in engine::graph::SammieMatting::ALL {
            assert!(DRIVER.contains(m.arg()), "{}", m.arg());
        }
    }

    #[test]
    fn the_driver_never_imports_sammies_gpl_code() {
        // It runs the downloader as a PROGRAM; it imports only the vendored
        // model packages.
        for l in DRIVER.lines().map(str::trim) {
            if l.starts_with("import ") || l.starts_with("from ") {
                assert!(!l.contains("sammie"), "{l}");
            }
        }
    }

    #[test]
    fn the_clicks_file_is_what_the_driver_reads() {
        let j = points_json(&[SamPoint { frame: 1004, x: 0.25, y: 0.5, positive: false }]);
        let v: serde_json::Value = serde_json::from_str(&j).unwrap();
        let p = &v["points"][0];
        assert_eq!(p["frame"], 1004);
        assert_eq!(p["x"], 0.25);
        assert_eq!(p["positive"], false);
    }

    #[test]
    fn edits_land_on_the_right_point() {
        let a = SamPoint { frame: 1, x: 0.1, y: 0.1, positive: true };
        let b = SamPoint { frame: 2, x: 0.2, y: 0.2, positive: false };
        let mut pts = vec![a, b];
        apply(&mut pts, Edit::Move(1, [0.5, 0.6]));
        assert_eq!((pts[1].x, pts[1].y, pts[1].positive), (0.5, 0.6, false));
        apply(&mut pts, Edit::Remove(0));
        assert_eq!(pts.len(), 1);
        apply(&mut pts, Edit::Add(a));
        assert_eq!(pts[1], a);
        apply(&mut pts, Edit::Remove(9));
        assert_eq!(pts.len(), 2);
    }
}

// ------------------------------------------------------------- the suite

/// `--sammie-fake --shot <dir> ...`: the stand-in tool `--test-sammie` runs
/// in place of the real one, so render, track, adopt and playback are all
/// exercised with no Python, torch or weights. Its "segment" is a disc
/// around each frame's include points (the nearest clicked frame's), and
/// it speaks the driver's protocol line for line.
pub fn fake(args: &[String]) -> anyhow::Result<()> {
    if args.iter().any(|a| a == "--serve") {
        return fake_serve();
    }
    let shot = args
        .windows(2)
        .find(|w| w[0] == "--shot")
        .map(|w| PathBuf::from(&w[1]))
        .ok_or_else(|| anyhow::anyhow!("--sammie-fake needs --shot"))?;
    use engine::decode::ExrDecoder;
    let dec = engine::decode::AnyDecoder::new(false);
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(shot.join("points.json"))?)?;
    let pts: Vec<(i64, f32, f32)> = v["points"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|p| p["positive"].as_bool() == Some(true))
        .map(|p| (p["frame"].as_i64().unwrap_or(0), p["x"].as_f64().unwrap_or(0.0) as f32, p["y"].as_f64().unwrap_or(0.0) as f32))
        .collect();
    let mut inputs: Vec<PathBuf> = std::fs::read_dir(shot.join("Input"))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "exr"))
        .collect();
    inputs.sort();
    let out = shot.join("Matte");
    std::fs::create_dir_all(&out)?;
    let total = inputs.len();
    println!("WOBBLE frame 0 {total}");
    for (i, path) in inputs.iter().enumerate() {
        let name = path.file_name().unwrap();
        let tick: i64 = name.to_string_lossy().trim_start_matches("f_").trim_end_matches(".exr").parse().unwrap_or(0);
        let info = dec.read_header(path)?;
        let (w, h) = (info.width as usize, info.height as usize);
        let stride = info.row_stride_f16();
        let near = pts.iter().map(|p| p.0).min_by_key(|f| (f - tick).abs());
        let here: Vec<[f32; 2]> = pts.iter().filter(|p| Some(p.0) == near).map(|p| [p.1, p.2]).collect();
        let px = fake_discs(w, h, stride, &here);
        engine::bakes::write_rgba_f16_exr_tagged(&out.join(name), w, h, stride, &px, None)?;
        println!("WOBBLE frame {} {total}", i + 1);
    }
    println!("WOBBLE done");
    Ok(())
}

/// The stand-in's segment: a disc of 0.1 × the short side round each
/// include point.
fn fake_discs(w: usize, h: usize, stride: usize, pts: &[[f32; 2]]) -> Vec<half::f16> {
    let mut px = vec![half::f16::ZERO; stride * h];
    let r = 0.1 * w.min(h) as f32;
    for [x, y] in pts {
        let (cx, cy) = (x * w as f32, y * h as f32);
        for yy in 0..h {
            for xx in 0..w {
                let d = ((xx as f32 + 0.5 - cx).powi(2) + (yy as f32 + 0.5 - cy).powi(2)).sqrt();
                if d < r {
                    for c in 0..4 {
                        px[yy * stride + xx * 4 + c] = half::f16::ONE;
                    }
                }
            }
        }
    }
    px
}

/// `--sammie-fake --serve`: the stand-in segmenter, line for line the
/// driver's `--serve` protocol.
fn fake_serve() -> anyhow::Result<()> {
    use engine::decode::ExrDecoder;
    use std::io::BufRead;
    let dec = engine::decode::AnyDecoder::new(false);
    println!("WOBBLE ready");
    for line in std::io::stdin().lock().lines() {
        let line = line?;
        let reply = (|| -> anyhow::Result<()> {
            let v: serde_json::Value = serde_json::from_str(&line)?;
            let image = PathBuf::from(v["image"].as_str().unwrap_or_default());
            let out = PathBuf::from(v["out"].as_str().unwrap_or_default());
            let info = dec.read_header(&image)?;
            let (w, h, stride) = (info.width as usize, info.height as usize, info.row_stride_f16());
            let pts: Vec<[f32; 2]> = v["points"]
                .as_array()
                .into_iter()
                .flatten()
                .filter(|p| p["positive"].as_bool() == Some(true))
                .map(|p| [p["x"].as_f64().unwrap_or(0.0) as f32, p["y"].as_f64().unwrap_or(0.0) as f32])
                .collect();
            if let Some(d) = out.parent() {
                std::fs::create_dir_all(d)?;
            }
            engine::bakes::write_rgba_f16_exr_tagged(&out, w, h, stride, &fake_discs(w, h, stride, &pts), None)?;
            Ok(())
        })();
        match reply {
            Ok(()) => println!("WOBBLE done"),
            Err(e) => println!("WOBBLE error {e}"),
        }
    }
    Ok(())
}

pub struct TestSammie {
    stage: u32,
    frames: u32,
    t0: web_time::Instant,
    since: web_time::Instant,
    node: u64,
    /// The plate through the node before anything is tracked, at the click.
    before: Option<f32>,
    /// ...and the comp frame it was read on: the plate's grain differs per
    /// frame, so the comparison must be on the same one.
    tick: u64,
    inside: Option<f32>,
    last: Option<f32>,
}

impl Default for TestSammie {
    fn default() -> Self {
        let now = web_time::Instant::now();
        Self { stage: 0, frames: 0, t0: now, since: now, node: 0, before: None, tick: 0, inside: None, last: None }
    }
}

/// Where the suite clicks, normalised: the include point and the exclude.
const INCLUDE: [f32; 2] = [0.35, 0.5];
const EXCLUDE: [f32; 2] = [0.7, 0.3];
/// A second include point, for the auto preview.
const SECOND: [f32; 2] = [0.75, 0.6];

impl Gfx {
    /// `--test-sammie`: a Sammie node end to end in the live app, against
    /// the stand-in above (`--sammie-fake`), through the real viewer input.
    ///
    /// S-click puts an include point where it was clicked, on the frame on
    /// screen; shift+S-click an exclude point; D-click on the exclude point
    /// removes it. `track` renders the plate, runs the stand-in and plays
    /// the matte back: the cutout is the plate inside the segment and black
    /// outside it, the matte output is 1 and 0 on the mask pipe, and a new
    /// click marks the loaded matte STALE without re-running anything.
    /// Waits are wall-clock.
    pub(crate) fn drive_test_sammie(&mut self, raw: &mut egui::RawInput) {
        let Some(mut t) = self.test_sammie.take() else { return };
        t.frames += 1;
        let fail = |m: String| -> ! {
            println!("SAMMIE TEST FAIL: {m}");
            std::process::exit(1);
        };
        if t.t0.elapsed().as_secs() > 120 {
            fail(format!("timeout at stage {}", t.stage));
        }
        let at = |s: &Self, q: [f32; 2]| {
            let r = s.test_rect;
            egui::pos2(r.min.x + q[0] * r.width(), r.min.y + q[1] * r.height())
        };
        let press = |raw: &mut egui::RawInput, pos: egui::Pos2, down: bool, mods: egui::Modifiers| {
            raw.events.insert(0, egui::Event::ModifiersChanged(mods));
            raw.events.push(egui::Event::PointerMoved(pos));
            raw.events.push(egui::Event::PointerButton { pos, button: egui::PointerButton::Primary, pressed: down, modifiers: mods });
        };
        let key = |raw: &mut egui::RawInput, k: egui::Key, down: bool| {
            raw.events.push(egui::Event::Key { key: k, physical_key: None, pressed: down, repeat: false, modifiers: Default::default() });
        };
        let pts = |s: &Self| s.egraph.node(t.node).and_then(|n| params(&n.kind)).map(|p| p.points.clone()).unwrap_or_default();
        let probe = |s: &mut Self, t: &mut TestSammie, q: [f32; 2]| -> Option<f32> {
            if s.force_display || s.eval_tick() != s.timeline_pos || t.since.elapsed().as_millis() < 150 {
                return None;
            }
            let (w, h) = s.comp.output_size()?;
            let v = s.probe_output_box((q[0] * w as f32) as u32 - 8, (q[1] * h as f32) as u32 - 8, 16, 16, 0)?;
            t.since = web_time::Instant::now();
            match t.last.replace(v) {
                Some(p) if (p - v).abs() < 1e-5 => {
                    t.last = None;
                    Some(v)
                }
                _ => None,
            }
        };
        let next = |t: &mut TestSammie, s: u32| {
            t.stage = s;
            t.since = web_time::Instant::now();
            t.last = None;
        };
        let held = egui::Modifiers::default();
        // Each step of a gesture gets UI passes of its own: a press and its
        // release, or a key and the click it modifies, folded into one pass
        // read as a different gesture.
        if ((2..=7).contains(&t.stage) || t.stage == 71 || t.stage == 72) && t.since.elapsed().as_millis() < 120 {
            self.test_sammie = Some(t);
            return;
        }
        match t.stage {
            0 if t.frames >= 6 && self.playback.is_some() && self.displayed_total >= 2 => {
                if !self.layout_settled() {
                    self.test_sammie = Some(t);
                    return;
                }
                let exe = std::env::current_exe().unwrap_or_else(|e| fail(format!("current_exe: {e}")));
                std::env::set_var("WOBBLE_SAMMIE_CMD", format!("{} --sammie-fake", exe.display()));
                let read = self
                    .egraph
                    .nodes
                    .iter()
                    .find(|(_, n)| matches!(n.kind, NodeKind::Read { .. }))
                    .map(|(id, _)| *id)
                    .unwrap_or_else(|| fail("no Read in the default graph".into()));
                let slot = self.next_read_slot;
                self.next_read_slot += 1;
                let k = self.egraph.add(NodeKind::Sammie(SammieParams::new(slot)), egui::pos2(360.0, 200.0));
                self.egraph.connect(k, 0, read);
                self.egraph.view = Some(k);
                self.selected = vec![k];
                self.playing = false;
                self.test_rect = self.image_rect;
                t.node = k;
                next(&mut t, 1);
            }
            1 => {
                // The plate through the untracked node (a cutout passes it).
                if let Some(v) = probe(self, &mut t, INCLUDE) {
                    t.before = Some(v);
                    t.tick = self.timeline_pos;
                    key(raw, egui::Key::S, true);
                    press(raw, at(self, INCLUDE), true, held);
                    next(&mut t, 2);
                }
            }
            2 => {
                press(raw, at(self, INCLUDE), false, held);
                next(&mut t, 3);
            }
            3 => {
                press(raw, at(self, EXCLUDE), true, egui::Modifiers::SHIFT);
                next(&mut t, 4);
            }
            4 => {
                press(raw, at(self, EXCLUDE), false, egui::Modifiers::SHIFT);
                next(&mut t, 5);
            }
            5 => {
                key(raw, egui::Key::S, false);
                raw.events.insert(0, egui::Event::ModifiersChanged(held));
                let p = pts(self);
                if p.len() < 2 {
                    if t.since.elapsed().as_millis() > 1000 {
                        fail(format!("two S-clicks made {} points", p.len()));
                    }
                    self.test_sammie = Some(t);
                    return;
                }
                let tick = self.timeline_pos as i64;
                let near = |q: &SamPoint, w: [f32; 2]| (q.x - w[0]).abs() < 0.01 && (q.y - w[1]).abs() < 0.01;
                if !(p[0].positive && near(&p[0], INCLUDE) && p[0].frame == tick) {
                    fail(format!("the S-click landed as {:?}, wanted include at {INCLUDE:?} on {tick}", p[0]));
                }
                if p[1].positive || !near(&p[1], EXCLUDE) {
                    fail(format!("the shift+S-click landed as {:?}, wanted exclude at {EXCLUDE:?}", p[1]));
                }
                key(raw, egui::Key::D, true);
                press(raw, at(self, EXCLUDE), true, held);
                next(&mut t, 6);
            }
            6 => {
                press(raw, at(self, EXCLUDE), false, held);
                next(&mut t, 7);
            }
            7 => {
                key(raw, egui::Key::D, false);
                let p = pts(self);
                if p.len() != 1 {
                    if t.since.elapsed().as_millis() > 1000 {
                        fail(format!("a D-click on the exclude point left {} points", p.len()));
                    }
                    self.test_sammie = Some(t);
                    return;
                }
                if !p[0].positive {
                    fail("the D-click removed the include point".into());
                }
                // Auto preview is on by default: the clicks alone segment
                // the frame, with no button.
                next(&mut t, 70);
            }
            // The preview: the frame on screen, segmented from its clicks,
            // shown over the viewer — then a new click re-runs it on its
            // own, bigger.
            70 | 73 => match self.sammie.run.as_ref().map(|r| &r.phase) {
                Some(Phase::Failed(e)) => fail(format!("the preview failed: {e}")),
                // Settled: nothing running and the preview made from the
                // clicks as they are now (an auto preview of an earlier
                // click may land first).
                Some(Phase::Done(_))
                    if self.sammie.edited.is_none()
                        && matches!(self.sammie_preview_state(t.node), PreviewState::Current(..)) =>
                {
                    let tick = self.timeline_pos as i64;
                    let PreviewState::Current(f, cov) = self.sammie_preview_state(t.node) else {
                        fail(format!("after a preview the state is {:?}", self.sammie_preview_state(t.node)));
                    };
                    if f != tick || self.sammie_overlay(t.node, tick).map(|o| o.1) != Some(true) {
                        fail("the preview is not on the viewer for the frame on screen".into());
                    }
                    if self.loaded.contains_key(&t.node) {
                        fail("a preview was adopted as the node's matte".into());
                    }
                    // One disc of 0.1 × height radius, then two.
                    let want = if t.stage == 70 { 0.0166 } else { 0.0332 };
                    if (cov - want).abs() > 0.004 {
                        fail(format!("the preview covers {cov:.4} of the frame, wanted ~{want}"));
                    }
                    if t.stage == 73 {
                        // Handles must not widen a track (a 67-frame trim
                        // of a long clip once rendered 247).
                        self.handles = (5, 5);
                        self.sammie_action(t.node, Action::Track);
                        next(&mut t, 8);
                    } else {
                        key(raw, egui::Key::S, true);
                        press(raw, at(self, SECOND), true, held);
                        next(&mut t, 71);
                    }
                }
                _ => {}
            },
            71 => {
                press(raw, at(self, SECOND), false, held);
                next(&mut t, 72);
            }
            72 => {
                key(raw, egui::Key::S, false);
                if pts(self).len() != 2 {
                    if t.since.elapsed().as_millis() > 1000 {
                        fail("the second S-click made no point".into());
                    }
                    self.test_sammie = Some(t);
                    return;
                }
                // Stale until the auto preview re-runs by itself.
                // (The stand-in can finish before this stage looks.)
                let rerun = matches!(self.sammie_preview_state(t.node), PreviewState::Current(_, c) if c > 0.025);
                if rerun || self.sammie.run.as_ref().is_some_and(|r| r.active()) {
                    next(&mut t, 73);
                } else if t.since.elapsed().as_secs() > 5 {
                    fail(format!("a click after a preview did not re-run it ({:?})", self.sammie_preview_state(t.node)));
                }
            }
            8 => match self.sammie.run.as_ref().map(|r| &r.phase) {
                Some(Phase::Failed(e)) => fail(format!("the track failed: {e}")),
                Some(Phase::Done(_)) => {
                    if !self.loaded.contains_key(&t.node) {
                        fail("the matte did not load into the node's reader".into());
                    }
                    let input = self.egraph.input_of(t.node, 0).unwrap_or_default();
                    let ((a, b), _) = self.range_of(input);
                    let got = self.loaded.get(&t.node).map(|l| l.seq.frames.len()).unwrap_or(0) as u64;
                    if got != b - a + 1 {
                        fail(format!("the track covered {got} frames, the plate's range is {}", b - a + 1));
                    }
                    self.seek_to(t.tick);
                    next(&mut t, 9);
                }
                _ => {}
            },
            9 => {
                if let Some(v) = probe(self, &mut t, INCLUDE) {
                    let b = t.before.unwrap();
                    if (v - b).abs() > 0.01 + 0.02 * b.abs() {
                        fail(format!("inside the segment the cutout reads {v:.4}, the plate {b:.4} at frame {}", self.timeline_pos));
                    }
                    t.inside = Some(v);
                    next(&mut t, 10);
                }
            }
            10 => {
                if let Some(v) = probe(self, &mut t, [0.85, 0.85]) {
                    if v.abs() > 1e-3 {
                        fail(format!("outside the segment the cutout reads {v:.4}, not black"));
                    }
                    if let Some(NodeKind::Sammie(p)) = self.egraph.node_mut(t.node).map(|n| &mut n.kind) {
                        p.output = engine::graph::SammieOut::Matte;
                    }
                    next(&mut t, 11);
                }
            }
            11 => {
                if let Some(v) = probe(self, &mut t, INCLUDE) {
                    if (v - 1.0).abs() > 1e-3 {
                        fail(format!("the matte reads {v:.4} inside the segment"));
                    }
                    let kind = self.egraph.node(t.node).map(|n| n.kind.output_kind());
                    if kind != Some(engine::graph::PortKind::Mask) {
                        fail(format!("the matte output is on the {kind:?} pipe"));
                    }
                    if let Some(NodeKind::Sammie(p)) = self.egraph.node_mut(t.node).map(|n| &mut n.kind) {
                        p.points.push(SamPoint { frame: p.points[0].frame, x: 0.2, y: 0.2, positive: true });
                    }
                    next(&mut t, 12);
                }
            }
            12 => {
                let v = self.sammie_view(t.node);
                if v.loaded.map(|l| l.1) != Some(false) {
                    fail(format!("a new click left the loaded matte {:?}, not stale", v.loaded));
                }
                if self.sammie.run.as_ref().is_some_and(|r| r.active()) {
                    fail("a new click started a run".into());
                }
                println!(
                    "SAMMIE TEST PASS: S-click / shift+S-click / D-click edit the clicks on the frame on \
                     screen; track rendered the plate, ran the stand-in and plays the matte back — the \
                     cutout is the plate inside and black outside, the matte is 1 on the mask pipe — and \
                     a new click marks it stale without re-running"
                );
                std::process::exit(0);
            }
            _ => {}
        }
        self.test_sammie = Some(t);
    }
}
