//! The CorridorKey node's app half: where the tool lives, installing it,
//! rendering the node's inputs to a shot directory, running the tool as a
//! process, and adopting what it wrote. The node itself (params, codegen)
//! is `engine::nodes::corridorkey`; the panel and glyph are
//! `crate::nodes::corridorkey`.
//!
//! A run is three phases, each explicit and cancellable:
//!
//! 1. **plate** — a `SubtreeBake` of input 0 over the input's frame range,
//!    written as linear Rec.709 EXRs to `<dir>/Input` (the tool's convention;
//!    a Write does the same conversion);
//! 2. **hint** — the same of input 1 to `<dir>/AlphaHint`, or, with nothing
//!    wired there, a coarse colour-difference key of the plate (a transient
//!    ScreenKey spliced into the eval graph, its matte as the picture): the
//!    network wants a ROUGH mask, and a roto or ScreenKey matte wired in is
//!    the better one;
//! 3. **key** — the tool, one process, driven by `tools/wobble_corridorkey.py`
//!    (embedded, written next to the checkout on every run), reporting a
//!    line per frame; `<dir>/Processed` is adopted as the node's reader
//!    when it says done.
//!
//! `<dir>` is keyed on everything that shapes the pixels — both inputs'
//! subtree keys, the recipe, the range — so an edit upstream shows as
//! STALE in the panel and a re-run lands in a new directory; the old one
//! keeps playing until then (the GraphBake contract). A completed run is
//! adopted on load (`corridor_adopt`), like a bake.
//!
//! **Nothing installs or downloads without a click.** `install…` clones the
//! tool into `~/.wobble/CorridorKey` (or `$WOBBLE_CORRIDORKEY_DIR`) and runs
//! `uv sync` there; the tool fetches its own weights from Hugging Face on
//! its first run, and the panel says so before the button.

use crate::Gfx;
use engine::graph::{CorridorKeyParams, CorridorScreen, NodeKind, ScreenKeyParams, ScreenOut};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;

/// The driver, written next to the checkout on every run.
pub const DRIVER: &str = include_str!("../../tools/wobble_corridorkey.py");
const DRIVER_NAME: &str = "wobble_corridorkey.py";
pub const REPO: &str = "https://github.com/nikopueringer/CorridorKey.git";
pub const LICENCE_URL: &str = "https://github.com/nikopueringer/CorridorKey/blob/main/LICENSE";

/// The transient ScreenKey the hint phase splices in when no hint is
/// wired. Far above any authored id.
pub const HINT_NODE: u64 = u64::MAX - 0x51;

/// Where the tool's checkout lives.
pub fn install_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("WOBBLE_CORRIDORKEY_DIR") {
        return PathBuf::from(d);
    }
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    home.join(".wobble").join("CorridorKey")
}

/// A checkout with its environment built.
pub fn installed(dir: &Path) -> bool {
    dir.join("clip_manager.py").is_file() && dir.join(".venv").is_dir()
}

/// How to get uv, on this platform.
pub(crate) const UV_MISSING: &str = if cfg!(windows) {
    "uv is not installed: winget install astral-sh.uv"
} else {
    "uv is not installed: brew install uv, or curl -LsSf https://astral.sh/uv/install.sh | sh"
};

/// `uv`, wherever a Finder-launched app cannot see it on PATH.
pub fn uv() -> Option<PathBuf> {
    // `uv.exe` on Windows, where the installer puts it in ~.localin.
    let exe = if cfg!(windows) { "uv.exe" } else { "uv" };
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(p) = std::env::var_os("PATH") {
        for d in std::env::split_paths(&p) {
            candidates.push(d.join(exe));
        }
    }
    for d in ["/opt/homebrew/bin", "/usr/local/bin"] {
        candidates.push(PathBuf::from(d).join(exe));
    }
    if let Some(h) = engine::paths::home() {
        candidates.push(h.join(".local").join("bin").join(exe));
        candidates.push(h.join(".cargo").join("bin").join(exe));
    }
    candidates.into_iter().find(|p| p.is_file())
}

/// What a process reports back.
pub enum Msg {
    Frame(usize, usize),
    Line(String),
    Done,
    Failed(String),
}

/// Where a run is.
pub enum Phase {
    /// Rendering input 0 (a `SubtreeBake` in `Gfx::graph_bake`).
    Plate,
    /// Rendering input 1, or the coarse stand-in.
    Hint,
    Keying { rx: Receiver<Msg>, cancel: Arc<AtomicBool>, done: usize, total: usize },
    Failed(String),
    Done(String),
}

pub struct Run {
    pub node: u64,
    pub dir: PathBuf,
    pub phase: Phase,
    /// The last lines the tool printed, for the panel.
    pub log: Vec<String>,
}

impl Run {
    pub fn active(&self) -> bool {
        matches!(self.phase, Phase::Plate | Phase::Hint | Phase::Keying { .. })
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
}

/// What the panel is told (read-only), built by `Gfx::corridor_view`.
pub struct View {
    pub dir: String,
    pub installed: bool,
    pub uv: bool,
    pub installing: Option<String>,
    pub install_result: Option<Result<(), String>>,
    pub state: ViewState,
    pub log: Vec<String>,
    pub has_input: bool,
    pub has_hint: bool,
    /// Frames loaded in the reader, and whether they are THIS recipe's.
    pub loaded: Option<(usize, bool)>,
    /// A finished run for the current recipe sits on disk, unloaded.
    pub on_disk: bool,
    pub busy_elsewhere: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ViewState {
    Idle,
    /// Rendering an input: (which, done, total).
    Rendering(&'static str, u64, u64),
    Keying(usize, usize),
    Failed(String),
    Done(String),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Action {
    Key,
    Cancel,
    Install,
    Dismiss,
}

pub fn params(kind: &NodeKind) -> Option<&CorridorKeyParams> {
    match kind {
        NodeKind::CorridorKey(p) => Some(p),
        _ => None,
    }
}

/// The coarse hint used when nothing is wired: a decisive colour-difference
/// matte of the plate against a stock screen of the picked colour.
pub fn stand_in_hint(screen: CorridorScreen) -> ScreenKeyParams {
    let sample = match screen {
        CorridorScreen::Blue => [0.10, 0.15, 0.60],
        _ => [0.16, 0.55, 0.14],
    };
    ScreenKeyParams {
        screen: sample,
        clip_black: 0.2,
        clip_white: 0.6,
        output: ScreenOut::Matte,
        ..Default::default()
    }
}

/// The command that runs the driver: `$WOBBLE_CORRIDORKEY_CMD` (a suite's
/// stand-in, whitespace-split) or `uv run --project <dir> python <driver>`.
fn command(dir: &Path) -> Result<std::process::Command, String> {
    if let Ok(s) = std::env::var("WOBBLE_CORRIDORKEY_CMD") {
        let mut parts = s.split_whitespace();
        let exe = parts.next().ok_or("WOBBLE_CORRIDORKEY_CMD is empty")?;
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

/// Run a command on a thread, its stdout parsed for WOBBLE lines and every
/// other line (stderr too) forwarded as a log line.
pub(crate) fn spawn(mut cmd: std::process::Command, tx: Sender<Msg>, cancel: Arc<AtomicBool>, name: &str) -> Result<(), String> {
    use std::io::BufRead;
    use std::process::Stdio;
    cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = cmd.spawn().map_err(|e| format!("{name}: {e}"))?;
    let out = child.stdout.take().unwrap();
    let err = child.stderr.take().unwrap();
    let tx2 = tx.clone();
    let _ = std::thread::Builder::new().name(format!("{name}-err")).spawn(move || {
        for line in std::io::BufReader::new(err).lines().map_while(Result::ok) {
            let _ = tx2.send(Msg::Line(line));
        }
    });
    let name = name.to_string();
    std::thread::Builder::new()
        .name(name.clone())
        .spawn(move || {
            let mut said_done = false;
            let mut said_err = None;
            let reader = std::io::BufReader::new(out);
            for line in reader.lines().map_while(Result::ok) {
                if cancel.load(Relaxed) {
                    let _ = child.kill();
                    break;
                }
                let mut w = line.split_whitespace();
                if w.next() == Some("WOBBLE") {
                    match (w.next(), w.next(), w.next()) {
                        (Some("frame"), Some(a), Some(b)) => {
                            if let (Ok(a), Ok(b)) = (a.parse(), b.parse()) {
                                let _ = tx.send(Msg::Frame(a, b));
                            }
                        }
                        (Some("done"), _, _) => said_done = true,
                        (Some("error"), _, _) => {
                            said_err = Some(line.trim_start_matches("WOBBLE error").trim().to_string());
                        }
                        _ => {
                            let _ = tx.send(Msg::Line(line.clone()));
                        }
                    }
                } else {
                    let _ = tx.send(Msg::Line(line));
                }
            }
            let status = child.wait();
            if cancel.load(Relaxed) {
                let _ = tx.send(Msg::Failed("cancelled".into()));
            } else if let Some(e) = said_err {
                let _ = tx.send(Msg::Failed(e));
            } else if said_done {
                let _ = tx.send(Msg::Done);
            } else {
                let _ = tx.send(Msg::Failed(match status {
                    Ok(s) => format!("the tool stopped without finishing ({s})"),
                    Err(e) => format!("{e}"),
                }));
            }
        })
        .map_err(|e| format!("{e}"))?;
    Ok(())
}

/// One frame file, both for the rendered inputs and for the tool's output.
pub fn frame_name(tick: u64) -> String {
    format!("f_{tick:05}.exr")
}

const LOG_KEEP: usize = 12;

pub(crate) fn push_log(log: &mut Vec<String>, line: String) {
    if line.trim().is_empty() {
        return;
    }
    log.push(line);
    if log.len() > LOG_KEEP {
        let drop = log.len() - LOG_KEEP;
        log.drain(..drop);
    }
}

impl Gfx {
    /// Where a run of `node` at the current recipe lands, and the key it
    /// is named by. None while the inputs cannot be rendered.
    pub(crate) fn corridor_dir(&self, node: u64) -> Option<(u64, PathBuf)> {
        let p = self.egraph.node(node).and_then(|n| params(&n.kind))?;
        let input = self.egraph.input_of(node, 0)?;
        let k0 = self.subtree_key_of(input)?;
        let k1 = match self.egraph.input_of(node, 1) {
            Some(h) => self.subtree_key_of(h)?,
            None => 0,
        };
        let (from, len) = self.corridor_range(input);
        let recipe = serde_json::to_vec(&(p.recipe(), from, len)).ok()?;
        let key = engine::diskcache::content_hash(&[&k0.to_le_bytes(), &k1.to_le_bytes(), &recipe]);
        let root = match &self.disk_cache {
            Some(c) => c.root().join("corridorkey"),
            None => engine::paths::temp_dir().join("wobble-corridorkey"),
        };
        Some((key, root.join(format!("{key:016x}"))))
    }

    /// The frames a run covers: the plate's range, handles included, like a
    /// GraphBake.
    fn corridor_range(&self, input: u64) -> (u64, u64) {
        let ((a, b), _) = self.range_of(input);
        (a.saturating_sub(self.handles.0), b.saturating_add(self.handles.1) + 1)
    }

    pub(crate) fn corridor_view(&self, node: u64) -> View {
        let dir = install_dir();
        let mine = self.corridor.run.as_ref().filter(|r| r.node == node);
        let state = match mine.map(|r| &r.phase) {
            None => ViewState::Idle,
            Some(Phase::Plate) | Some(Phase::Hint) => {
                let which = if matches!(mine.map(|r| &r.phase), Some(Phase::Plate)) { "plate" } else { "hint" };
                match &self.graph_bake {
                    Some(b) if b.node == node => ViewState::Rendering(which, b.next.saturating_sub(b.from), b.len.saturating_sub(b.from)),
                    _ => ViewState::Rendering(which, 0, 0),
                }
            }
            Some(Phase::Keying { done, total, .. }) => ViewState::Keying(*done, *total),
            Some(Phase::Failed(e)) => ViewState::Failed(e.clone()),
            Some(Phase::Done(s)) => ViewState::Done(s.clone()),
        };
        let want = self.corridor_dir(node).map(|(_, d)| d.join("Processed"));
        let loaded = self.loaded.get(&node).map(|l| {
            let current = want.as_ref().is_some_and(|d| l.path == d.to_string_lossy());
            (l.seq.frames.len(), current)
        });
        let on_disk = want.as_ref().is_some_and(|d| engine::bakes::is_baked(d));
        View {
            dir: dir.to_string_lossy().into_owned(),
            installed: installed(&dir) || std::env::var_os("WOBBLE_CORRIDORKEY_CMD").is_some(),
            uv: uv().is_some(),
            installing: self.corridor.install.as_ref().filter(|i| i.result.is_none()).map(|i| i.log.last().cloned().unwrap_or_default()),
            install_result: self.corridor.install.as_ref().and_then(|i| i.result.clone()),
            state,
            log: mine.map(|r| r.log.clone()).unwrap_or_default(),
            has_input: self.egraph.input_of(node, 0).is_some(),
            has_hint: self.egraph.input_of(node, 1).is_some(),
            loaded,
            on_disk,
            busy_elsewhere: self.corridor.run.as_ref().is_some_and(|r| r.node != node && r.active()),
        }
    }

    pub(crate) fn corridor_action(&mut self, node: u64, a: Action) {
        match a {
            Action::Key => self.corridor_start(node),
            Action::Cancel => {
                if let Some(r) = self.corridor.run.take() {
                    match r.phase {
                        Phase::Keying { cancel, .. } => cancel.store(true, Relaxed),
                        Phase::Plate | Phase::Hint => {
                            if self.graph_bake.as_ref().is_some_and(|b| b.node == r.node) {
                                self.graph_bake = None;
                            }
                        }
                        _ => {}
                    }
                }
            }
            Action::Install => self.corridor_install(),
            Action::Dismiss => {
                if self.corridor.run.as_ref().is_some_and(|r| r.node == node && !r.active()) {
                    self.corridor.run = None;
                }
                if self.corridor.install.as_ref().is_some_and(|i| i.result.is_some()) {
                    self.corridor.install = None;
                }
            }
        }
    }

    /// `install…`: clone the tool and build its environment, on a thread,
    /// its output streamed to the panel.
    fn corridor_install(&mut self) {
        if self.corridor.install.as_ref().is_some_and(|i| i.result.is_none()) {
            return;
        }
        let dir = install_dir();
        let (tx, rx) = std::sync::mpsc::channel();
        // A page has no processes and no threads (web/).
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (&dir, &tx);
            self.corridor.install = Some(Install {
                rx,
                log: vec![],
                result: Some(Err("the tool runs as a process: not in the web build".into())),
            });
            return;
        }
        #[cfg_attr(target_arch = "wasm32", allow(unreachable_code))]
        let Some(uv) = uv() else {
            self.corridor.install = Some(Install {
                rx,
                log: vec![],
                result: Some(Err(UV_MISSING.into())),
            });
            return;
        };
        let spawned = std::thread::Builder::new()
            .name("corridorkey-install".into())
            .spawn(move || {
                let run = |mut c: std::process::Command, tx: &Sender<Msg>| -> Result<(), String> {
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
                };
                let result = (|| -> Result<(), String> {
                    if !dir.join("clip_manager.py").is_file() {
                        if let Some(parent) = dir.parent() {
                            std::fs::create_dir_all(parent).map_err(|e| format!("{e}"))?;
                        }
                        let _ = tx.send(Msg::Line(format!("git clone {REPO} {}", dir.display())));
                        let mut c = std::process::Command::new("git");
                        c.arg("clone").arg("--depth").arg("1").arg(REPO).arg(&dir);
                        run(c, &tx).map_err(|e| format!("git clone: {e}"))?;
                    }
                    let _ = tx.send(Msg::Line("uv sync (torch and friends: several GB, a while)".into()));
                    let mut c = std::process::Command::new(&uv);
                    c.arg("sync");
                    if cfg!(target_os = "macos") && cfg!(target_arch = "aarch64") {
                        c.arg("--extra").arg("mlx");
                    }
                    c.current_dir(&dir);
                    run(c, &tx).map_err(|e| format!("uv sync: {e}"))?;
                    Ok(())
                })();
                let _ = tx.send(match result {
                    Ok(()) => Msg::Done,
                    Err(e) => Msg::Failed(e),
                });
            })
            .is_ok();
        self.corridor.install = Some(Install {
            rx,
            log: vec![],
            result: (!spawned).then_some(Err("could not start a thread".into())),
        });
    }

    /// The `key` click: adopt a finished run, else render the plate.
    fn corridor_start(&mut self, node: u64) {
        let fail = |me: &mut Self, e: &str| {
            me.corridor.run = Some(Run { node, dir: PathBuf::new(), phase: Phase::Failed(e.into()), log: vec![] });
        };
        if self.corridor.run.as_ref().is_some_and(|r| r.active()) {
            return;
        }
        #[cfg(target_arch = "wasm32")]
        return fail(self, "a key runs a process: not in the web build");
        #[cfg_attr(target_arch = "wasm32", allow(unreachable_code))]
        let Some(input) = self.egraph.input_of(node, 0) else {
            return fail(self, "wire the plate into the first input");
        };
        let Some((_, dir)) = self.corridor_dir(node) else {
            return fail(self, "the inputs are not renderable yet");
        };
        let out = dir.join("Processed");
        if engine::bakes::is_baked(&out) {
            self.load_read(node, &out.to_string_lossy());
            self.playback_dirty = true;
            self.corridor.run = Some(Run { node, dir, phase: Phase::Done("loaded the key on disk".into()), log: vec![] });
            return;
        }
        if !installed(&install_dir()) && std::env::var_os("WOBBLE_CORRIDORKEY_CMD").is_none() {
            return fail(self, "CorridorKey is not installed: press install…");
        }
        self.corridor.run = Some(Run { node, dir: dir.clone(), phase: Phase::Plate, log: vec![] });
        if let Err(e) = self.corridor_render(node, input, 0, None) {
            fail(self, &e);
        }
    }

    /// A `SubtreeBake` of `input` into `<dir>/Input` (port 0) or
    /// `<dir>/AlphaHint` (port 1); `hint` splices the stand-in key in.
    fn corridor_render(&mut self, node: u64, input: u64, port: usize, hint: Option<ScreenKeyParams>) -> Result<(), String> {
        let Some(r) = self.corridor.run.as_ref().filter(|r| r.node == node) else { return Ok(()) };
        let dir = r.dir.clone();
        let sub = dir.join(if port == 0 { "Input" } else { "AlphaHint" });
        std::fs::create_dir_all(&sub).map_err(|e| format!("{e}"))?;
        let key = self.subtree_key_of(input).ok_or("the input is not renderable yet")?;
        let (w, h) = self.comp.output_size().ok_or("nothing is rendering yet")?;
        let (from, len) = self.corridor_range(self.egraph.input_of(node, 0).unwrap_or(input));
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
            label: Some("corridorkey-render"),
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
            sink: crate::BakeSink::Corridor { port, hint },
        };
        if bake.next >= bake.len {
            self.finish_graph_bake(bake);
        } else {
            self.graph_bake = Some(bake);
        }
        Ok(())
    }

    /// The bake loop finished rendering `port`: on to the hint, then the tool.
    pub(crate) fn corridor_render_finished(&mut self, node: u64, port: usize) {
        let Some(r) = self.corridor.run.as_mut().filter(|r| r.node == node) else { return };
        if port == 0 {
            r.phase = Phase::Hint;
            let screen = self.egraph.node(node).and_then(|n| params(&n.kind)).map(|p| p.screen).unwrap_or_default();
            let (input, hint) = match self.egraph.input_of(node, 1) {
                Some(h) => (h, None),
                None => match self.egraph.input_of(node, 0) {
                    Some(i) => (i, Some(stand_in_hint(screen))),
                    None => return self.corridor_fail(node, "the plate was unwired mid-run"),
                },
            };
            if let Err(e) = self.corridor_render(node, input, 1, hint) {
                self.corridor_fail(node, &e);
            }
            return;
        }
        // Both inputs on disk: run the tool.
        let dir = r.dir.clone();
        let Some(p) = self.egraph.node(node).and_then(|n| params(&n.kind)).cloned() else {
            return self.corridor_fail(node, "the node is gone");
        };
        let mut cmd = match command(&install_dir()) {
            Ok(c) => c,
            Err(e) => return self.corridor_fail(node, &e),
        };
        cmd.arg("--shot")
            .arg(&dir)
            .arg("--screen")
            .arg(p.screen.arg())
            .arg("--despill")
            .arg(format!("{}", p.despill.clamp(0.0, 1.0)))
            .arg("--refiner")
            .arg(format!("{}", p.refiner))
            .arg("--despeckle")
            .arg(if p.despeckle { "1" } else { "0" })
            .arg("--despeckle-px")
            .arg(format!("{}", p.despeckle_px));
        let (tx, rx) = std::sync::mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        if let Err(e) = spawn(cmd, tx, cancel.clone(), "corridorkey") {
            return self.corridor_fail(node, &e);
        }
        if let Some(r) = self.corridor.run.as_mut() {
            r.phase = Phase::Keying { rx, cancel, done: 0, total: 0 };
        }
    }

    fn corridor_fail(&mut self, node: u64, e: &str) {
        if let Some(r) = self.corridor.run.as_mut().filter(|r| r.node == node) {
            r.phase = Phase::Failed(e.to_string());
        }
    }

    /// Every redraw: drain the tool's messages, catch a render that died,
    /// adopt a finished key.
    pub(crate) fn corridor_poll(&mut self) {
        if let Some(i) = self.corridor.install.as_mut() {
            while let Ok(m) = i.rx.try_recv() {
                match m {
                    Msg::Line(l) => push_log(&mut i.log, l),
                    Msg::Done => i.result = Some(Ok(())),
                    Msg::Failed(e) => i.result = Some(Err(e)),
                    Msg::Frame(..) => {}
                }
            }
        }
        let Some(r) = self.corridor.run.as_mut() else { return };
        let node = r.node;
        if matches!(r.phase, Phase::Plate | Phase::Hint) && self.graph_bake.is_none() {
            let why = self.load_errors.get(&node).cloned().unwrap_or_else(|| "the input stopped rendering".into());
            r.phase = Phase::Failed(why);
            return;
        }
        let mut end = None;
        if let Phase::Keying { rx, done, total, .. } = &mut r.phase {
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
            Some(Ok(())) => {
                let out = r.dir.join("Processed");
                if let Err(e) = engine::bakes::mark_baked(&out) {
                    r.phase = Phase::Failed(format!("marking the key done: {e}"));
                    return;
                }
                r.phase = Phase::Done("keyed".into());
                self.load_read(node, &out.to_string_lossy());
                self.playback_dirty = true;
            }
            Some(Err(e)) => r.phase = Phase::Failed(e),
            None => {}
        }
    }

    /// After a load, an undo, a paste: every CorridorKey node whose run for
    /// the current recipe is on disk plays it.
    pub(crate) fn corridor_adopt(&mut self) {
        let ids: Vec<u64> = self
            .egraph
            .nodes
            .iter()
            .filter(|(id, n)| {
                matches!(n.kind, NodeKind::CorridorKey(_))
                    && !self.loaded.contains_key(id)
                    && !self.load_errors.contains_key(id)
            })
            .map(|(id, _)| *id)
            .collect();
        for id in ids {
            let Some((_, dir)) = self.corridor_dir(id) else { continue };
            let out = dir.join("Processed");
            if engine::bakes::is_baked(&out) {
                self.load_read(id, &out.to_string_lossy());
                self.playback_dirty = true;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_stand_in_hint_is_a_matte_of_the_picked_screen() {
        let g = stand_in_hint(CorridorScreen::Green);
        assert_eq!(g.output, ScreenOut::Matte);
        assert_eq!(g.dominant(), 1);
        assert_eq!(stand_in_hint(CorridorScreen::Auto).dominant(), 1);
        assert_eq!(stand_in_hint(CorridorScreen::Blue).dominant(), 2);
        // Decisive: the screen is 0, a grey is 1.
        assert_eq!(g.matte(g.screen), 0.0);
        assert_eq!(g.matte([0.2; 3]), 1.0);
    }

    #[test]
    fn frame_names_match_the_bake_convention() {
        assert_eq!(frame_name(7), "f_00007.exr");
        assert_eq!(frame_name(1001), "f_01001.exr");
    }

    #[test]
    fn the_driver_speaks_the_protocol_the_reader_parses() {
        for w in ["WOBBLE frame", "WOBBLE done", "WOBBLE error", "--shot", "--despeckle-px", "input_is_linear=True"] {
            assert!(DRIVER.contains(w), "{w}");
        }
    }
}

// ------------------------------------------------------------- the suite

/// `--corridor-fake <shot>`: the stand-in tool `--test-corridor` runs in
/// place of CorridorKey, so the whole pipeline — render, process, adopt,
/// playback — is exercised with no Python, no torch and no weights. It
/// writes `Processed/` = plate × hint (premultiplied), alpha = hint, and
/// speaks the driver's protocol line for line.
pub fn fake(shot: &Path) -> anyhow::Result<()> {
    use engine::decode::ExrDecoder;
    let dec = engine::decode::AnyDecoder::new(false);
    let mut inputs: Vec<PathBuf> = std::fs::read_dir(shot.join("Input"))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "exr"))
        .collect();
    inputs.sort();
    let out_dir = shot.join("Processed");
    std::fs::create_dir_all(&out_dir)?;
    let total = inputs.len();
    println!("WOBBLE frame 0 {total}");
    for (i, path) in inputs.iter().enumerate() {
        let name = path.file_name().unwrap();
        let hint = shot.join("AlphaHint").join(name);
        let info = dec.read_header(path)?;
        let stride = info.row_stride_f16();
        let n = stride * info.height as usize;
        let mut plate = vec![half::f16::ZERO; n];
        let mut mask = vec![half::f16::ZERO; n];
        dec.decode_into(path, info, stride, &mut plate)?;
        dec.decode_into(&hint, info, stride, &mut mask)?;
        for (p, m) in plate.chunks_exact_mut(4).zip(mask.chunks_exact(4)) {
            let a = m[0].to_f32().clamp(0.0, 1.0);
            for c in p.iter_mut().take(3) {
                *c = half::f16::from_f32(c.to_f32() * a);
            }
            p[3] = half::f16::from_f32(a);
        }
        engine::bakes::write_rgba_f16_exr_tagged(
            &out_dir.join(name),
            info.width as usize,
            info.height as usize,
            stride,
            &plate,
            None,
        )?;
        println!("WOBBLE frame {} {total}", i + 1);
    }
    println!("WOBBLE done");
    Ok(())
}

pub struct TestCorridor {
    stage: u32,
    t0: web_time::Instant,
    since: web_time::Instant,
    node: u64,
    /// The plate through the node before anything is keyed, at the probe.
    before: Option<f32>,
    /// The same, at the right-hand probe.
    before_right: Option<f32>,
    /// The keyed premultiplied picture at the probe.
    premult: Option<f32>,
    /// A probe waiting for a second one that agrees.
    last: Option<f32>,
    left: Option<f32>,
}

impl Default for TestCorridor {
    fn default() -> Self {
        let now = web_time::Instant::now();
        Self { stage: 0, t0: now, since: now, node: 0, before: None, before_right: None, premult: None, last: None, left: None }
    }
}

impl Gfx {
    /// `--test-corridor`: a CorridorKey node end to end in the live app,
    /// against the stand-in above.
    ///
    /// With no hint wired: `key` renders the plate (linear Rec.709) and a
    /// coarse ScreenKey matte of it, runs the stand-in, and the keyed
    /// frames play through the node — premultiplied output = plate × matte
    /// at the probe (the Rec.709 round trip through the working space is
    /// what that checks), and the matte output is the matte. Then a Gradient
    /// wired into the hint port: a re-key lands in a NEW directory, and the
    /// matte on screen is the gradient — dark on the left, bright on the
    /// right — with the premultiplied picture following it. Waits are
    /// wall-clock.
    pub(crate) fn drive_test_corridor(&mut self) {
        let Some(mut t) = self.test.corridor.take() else { return };
        let fail = |m: String| -> ! {
            println!("CORRIDOR TEST FAIL: {m}");
            std::process::exit(1);
        };
        if t.t0.elapsed().as_secs() > 180 {
            fail(format!("timeout at stage {} ({})", t.stage, self.corridor_state_words()));
        }
        let wait = |t: &TestCorridor, ms: u64| t.since.elapsed().as_millis() as u64 >= ms;
        // A reading two probes agree on, 150 ms apart, with the playhead
        // landed: the rig rebuild and the upload have both happened.
        let stable = |me: &mut Self, t: &mut TestCorridor, x: u32, ch: usize| -> Option<f32> {
            if me.force_display || me.eval_tick() != me.timeline_pos || !wait(t, 150) {
                return None;
            }
            let v = me.probe_output_box(x, 500, 32, 32, ch)?;
            t.since = web_time::Instant::now();
            match t.last.replace(v) {
                Some(p) if (p - v).abs() < 1e-5 => {
                    t.last = None;
                    Some(v)
                }
                _ => None,
            }
        };
        let next = |t: &mut TestCorridor, s: u32| {
            t.stage = s;
            t.since = web_time::Instant::now();
            t.last = None;
        };
        let set_out = |me: &mut Self, node: u64, o: engine::graph::CorridorOut| {
            if let Some(NodeKind::CorridorKey(p)) = me.egraph.node_mut(node).map(|n| &mut n.kind) {
                p.output = o;
            }
        };
        let phase = |me: &Self| me.corridor.run.as_ref().map(|r| match &r.phase {
            Phase::Done(_) => Ok(()),
            Phase::Failed(e) => Err(e.clone()),
            _ => Ok(()),
        });
        let done = |me: &Self| matches!(me.corridor.run.as_ref().map(|r| &r.phase), Some(Phase::Done(_)));
        match t.stage {
            0 if self.playback.is_some() && self.displayed_total >= 2 => {
                // Hermetic: the stand-in in place of the tool, so nothing
                // is installed, fetched or run from the user's machine.
                let exe = std::env::current_exe().unwrap_or_else(|e| fail(format!("current_exe: {e}")));
                std::env::set_var("WOBBLE_CORRIDORKEY_CMD", format!("{} --corridor-fake", exe.display()));
                let read = self
                    .egraph
                    .nodes
                    .iter()
                    .find(|(_, n)| matches!(n.kind, NodeKind::Read { .. }))
                    .map(|(id, _)| *id)
                    .unwrap_or_else(|| fail("no Read in the default graph".into()));
                let slot = self.next_read_slot;
                self.next_read_slot += 1;
                let k = self.egraph.add(NodeKind::CorridorKey(CorridorKeyParams::new(slot)), egui::pos2(360.0, 200.0));
                self.egraph.connect(k, 0, read);
                self.egraph.view = Some(k);
                self.playing = false;
                t.node = k;
                next(&mut t, 1);
            }
            1 => {
                if let Some(v) = stable(self, &mut t, 900, 0) {
                    t.before = Some(v);
                    next(&mut t, 10);
                }
            }
            10 => {
                if let Some(v) = stable(self, &mut t, 1800, 0) {
                    t.before_right = Some(v);
                    self.corridor_action(t.node, Action::Key);
                    next(&mut t, 2);
                }
            }
            2 => match phase(self) {
                Some(Err(e)) => fail(format!("the first key failed: {e}")),
                Some(Ok(())) if done(self) => {
                    if !self.loaded.contains_key(&t.node) {
                        fail("the key did not load into the node's reader".into());
                    }
                    next(&mut t, 3);
                }
                _ => {}
            },
            3 => {
                if let Some(v) = stable(self, &mut t, 900, 0) {
                    t.premult = Some(v);
                    set_out(self, t.node, engine::graph::CorridorOut::Matte);
                    next(&mut t, 4);
                }
            }
            4 => {
                if let Some(m) = stable(self, &mut t, 900, 0) {
                    let (before, premult) = (t.before.unwrap(), t.premult.unwrap());
                    if !(0.0..=1.0).contains(&m) {
                        fail(format!("the matte reads {m}, not a coverage"));
                    }
                    if m < 0.05 {
                        fail(format!("the stand-in hint keyed the whole probe out ({m}); the plate is not a screen"));
                    }
                    let want = before * m;
                    if (premult - want).abs() > 0.01 + 0.02 * want.abs() {
                        fail(format!("premultiplied {premult:.4} is not plate {before:.4} × matte {m:.4} = {want:.4}: the Rec.709 round trip moved the picture"));
                    }
                    // A hint of our own: a left-to-right ramp.
                    let g = self.egraph.add(
                        NodeKind::Gradient(engine::graph::GradientParams { inherit: true, ..Default::default() }),
                        egui::pos2(200.0, 320.0),
                    );
                    self.egraph.connect(t.node, 1, g);
                    let old = self.loaded.get(&t.node).map(|l| l.path.clone());
                    self.corridor_action(t.node, Action::Dismiss);
                    self.corridor_action(t.node, Action::Key);
                    if let Some(r) = &self.corridor.run {
                        if Some(r.dir.join("Processed").to_string_lossy().into_owned()) == old {
                            fail("wiring a hint did not change the key's directory".into());
                        }
                    }
                    next(&mut t, 5);
                }
            }
            5 => match phase(self) {
                Some(Err(e)) => fail(format!("the second key failed: {e}")),
                Some(Ok(())) if done(self) => next(&mut t, 6),
                _ => {}
            },
            6 => {
                if let Some(l) = stable(self, &mut t, 200, 0) {
                    t.left = Some(l);
                    next(&mut t, 7);
                }
            }
            7 => {
                if let Some(r) = stable(self, &mut t, 1800, 0) {
                    let l = t.left.unwrap();
                    if r < l + 0.3 {
                        fail(format!("the wired hint is not the matte: left {l:.3}, right {r:.3}"));
                    }
                    set_out(self, t.node, engine::graph::CorridorOut::Premult);
                    t.premult = Some(r);
                    next(&mut t, 8);
                }
            }
            8 => {
                if let Some(v) = stable(self, &mut t, 1800, 0) {
                    let want = t.before_right.unwrap() * t.premult.unwrap();
                    if (v - want).abs() > 0.01 + 0.03 * want.abs() {
                        fail(format!("premultiplied {v:.4} is not plate × the wired matte {want:.4}"));
                    }
                    println!(
                        "CORRIDOR TEST PASS: a CorridorKey rendered the plate and a stand-in hint, \
                         ran the tool, played the key back as plate × matte through the Rec.709 \
                         round trip; a wired Gradient hint re-keyed into a new directory and is \
                         the matte on screen"
                    );
                    std::process::exit(0);
                }
            }
            _ => {}
        }
        self.test.corridor = Some(t);
    }

    fn corridor_state_words(&self) -> String {
        match self.corridor.run.as_ref().map(|r| &r.phase) {
            None => "no run".into(),
            Some(Phase::Plate) => format!("plate, bake {}", self.graph_bake.is_some()),
            Some(Phase::Hint) => format!("hint, bake {}", self.graph_bake.is_some()),
            Some(Phase::Keying { done, total, .. }) => format!("keying {done}/{total}"),
            Some(Phase::Failed(e)) => format!("failed: {e}"),
            Some(Phase::Done(s)) => format!("done: {s}"),
        }
    }
}
