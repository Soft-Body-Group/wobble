//! The Depth node's app half: where Video Depth Anything lives, installing
//! it, rendering the node's input to a shot directory, running the tool as
//! a process, and adopting what it wrote. The node itself (params, codegen)
//! is `engine::nodes::depth`; the panel and glyph `crate::nodes::depth`.
//! The process plumbing is CorridorKey's (`crate::corridorkey::spawn`); the
//! input render is the Upscale node's (`Gfx::render_input_frames`).
//!
//! A run is two phases, each explicit and cancellable:
//!
//! 1. **render** — a `SubtreeBake` of input 0 over the input's frame range,
//!    at the input's own format, written as linear Rec.709 EXRs to
//!    `<dir>/Input`;
//! 2. **run** — the tool, one process, driven by `tools/wobble_depth.py`
//!    (embedded, written next to the checkout on every run); `<dir>/Depth`
//!    is adopted as the node's reader when it says done.
//!
//! `<dir>` is keyed on everything that shapes the pixels — the input's
//! subtree key and format, the recipe (model, input size), the range — so
//! an edit upstream shows as STALE and a re-run lands in a new directory;
//! the old result keeps playing until then (the GraphBake contract). The
//! output pick is NOT in it: the driver writes all three values.
//!
//! **Nothing installs or downloads without a click.** `install…` fetches
//! the pinned checkout into `~/.wobble/VideoDepthAnything` (or
//! `$WOBBLE_DEPTH_DIR`), writes our `pyproject.toml` beside it (upstream
//! has none) and runs `uv sync`. The WEIGHTS are not the tool's business:
//! they are catalog rows (`vda-*`), fetched by the download gate the
//! properties panel draws for any node that declares models, and handed to
//! the driver by path (`models::ready_path`).
//!
//! **Two tools, one runner.** Depth Pro 2 (Apple) is a second checkout
//! (`~/.wobble/DepthPro2`, `$WOBBLE_DEPTHPRO2_DIR`) with its own driver
//! (`tools/wobble_depthpro2.py`, which imports `wobble_depth.py`'s EXR
//! reader and writer, so both are written beside it). `Tool::of` picks by
//! the node's model; everything from the result on is shared.

use crate::corridorkey::{push_log, spawn, uv, Msg, UV_MISSING};
use crate::{editor, Gfx};
use engine::graph::{DepthParams, NodeKind};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;

pub const DRIVER: &str = include_str!("../../tools/wobble_depth.py");
const DRIVER_NAME: &str = "wobble_depth.py";
pub const PRO2_DRIVER: &str = include_str!("../../tools/wobble_depthpro2.py");
const PRO2_DRIVER_NAME: &str = "wobble_depthpro2.py";
pub const PRO2_REPO: &str = "https://github.com/apple-aiml-research/ml-depth-pro-2.git";
/// The commit the Depth Pro 2 driver is written against (it builds
/// `DepthPro2` from the tool's `PRESETS` and calls `infer_video`). Bump it
/// and re-test, never float it.
pub const PRO2_COMMIT: &str = "aa389cbb0b74341664ed663a0966c859905cd6c0";
pub const PRO2_LICENCE_URL: &str = "https://github.com/apple-aiml-research/ml-depth-pro-2/blob/main/LICENSE";
/// Upstream's own project, plus what the driver reads and writes EXRs
/// with; written over the checkout's `pyproject.toml` by `install…` (the
/// package still builds from `src/`). Windows gets PyTorch's CUDA wheels.
pub const PRO2_PYPROJECT: &str = r#"[project]
name = "depth_pro_2"
version = "0.1"
requires-python = ">=3.11,<3.13"
dependencies = [
    "av>=14.1",
    "click",
    "huggingface-hub>=1.0",
    "matplotlib",
    "numpy",
    "pillow>=11.3",
    "pillow-heif",
    "safetensors",
    "timm>=1.0.27,<1.1",
    "torch>=2.7",
    # wobble's driver: EXR in and out (OpenCV is only the reader's fallback;
    # 5.x's Windows wheel has no EXR codec).
    "OpenEXR>=3.3",
    "opencv-python<5",
]

[build-system]
requires = ["setuptools"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.uv.sources]
torch = [{ index = "pytorch-cu126", marker = "sys_platform == 'win32'" }]

[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true
"#;

/// Which tool runs a node: picked by its model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tool {
    /// Video Depth Anything (ByteDance).
    Vda,
    /// Depth Pro 2 (Apple).
    Pro2,
}

impl Tool {
    pub fn of(p: &DepthParams) -> Tool {
        if p.model.pro2() {
            Tool::Pro2
        } else {
            Tool::Vda
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Tool::Vda => "Video Depth Anything",
            Tool::Pro2 => "Depth Pro 2",
        }
    }

    pub fn repo(self) -> &'static str {
        match self {
            Tool::Vda => REPO,
            Tool::Pro2 => PRO2_REPO,
        }
    }

    fn commit(self) -> &'static str {
        match self {
            Tool::Vda => COMMIT,
            Tool::Pro2 => PRO2_COMMIT,
        }
    }

    fn pyproject(self) -> &'static str {
        match self {
            Tool::Vda => PYPROJECT,
            Tool::Pro2 => PRO2_PYPROJECT,
        }
    }

    /// Who made it, and its CODE licence (the weights' is the catalog's).
    pub fn credit(self) -> (&'static str, &'static str, &'static str) {
        match self {
            Tool::Vda => ("Video Depth Anything by ByteDance", "Apache-2.0", LICENCE_URL),
            Tool::Pro2 => ("Depth Pro 2 by Apple", "Apple sample code licence", PRO2_LICENCE_URL),
        }
    }

    /// Where it runs and how it walks a clip, for the panel.
    pub fn about(self) -> &'static str {
        match self {
            Tool::Vda => "runs on Apple Silicon (MPS), an NVIDIA GPU (CUDA) or, slowly, the CPU. The clip is solved in overlapping 32-frame windows, each aligned to the last, so the depth holds steady over time; memory does not grow with the clip",
            Tool::Pro2 => "runs on an NVIDIA GPU (CUDA), Apple Silicon (MPS) or, slowly, the CPU. Metres at the network's input size, the clip in overlapping 16-frame windows cross-faded, one focal length estimated for the whole clip; memory does not grow with the clip",
        }
    }
}
pub const REPO: &str = "https://github.com/DepthAnything/Video-Depth-Anything.git";
/// The commit the driver is written against: it calls the tool's
/// `VideoDepthAnything.forward` and re-does `infer_video_depth`'s windowing
/// with its constants and helpers, which a later commit is free to change. Bump it and re-test, never float it.
pub const COMMIT: &str = "4f5ae23172ba60fd7bc11ef671cca678842c7072";
pub const LICENCE_URL: &str = "https://github.com/DepthAnything/Video-Depth-Anything/blob/main/LICENSE";

/// Upstream ships a requirements.txt pinned to torch 2.1 and xformers (no
/// MPS autocast, no macOS wheels for xformers); this is the environment
/// the driver needs, written into the checkout by `install…`. Windows gets
/// PyTorch's CUDA wheels (PyPI's are CPU-only there); Linux's PyPI torch
/// already carries CUDA, macOS's MPS.
pub const PYPROJECT: &str = r#"[project]
name = "wobble-video-depth-anything"
version = "0.0.0"
requires-python = ">=3.10,<3.13"
dependencies = [
    "torch>=2.5",
    "torchvision>=0.20",
    "numpy",
    # 5.x's Windows wheel has no EXR codec (measured: imread returns None).
    "opencv-python<5",
    "einops",
    # The EXR reader and writer (wheels on every OS); OpenCV's EXR codec is
    # only the fallback.
    "OpenEXR>=3.3",
    "easydict",
    "tqdm",
]

[tool.uv]
package = false

[tool.uv.sources]
torch = [{ index = "pytorch-cu126", marker = "sys_platform == 'win32'" }]
torchvision = [{ index = "pytorch-cu126", marker = "sys_platform == 'win32'" }]

[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true
"#;

pub fn install_dir(tool: Tool) -> PathBuf {
    let (var, name) = match tool {
        Tool::Vda => ("WOBBLE_DEPTH_DIR", "VideoDepthAnything"),
        Tool::Pro2 => ("WOBBLE_DEPTHPRO2_DIR", "DepthPro2"),
    };
    if let Some(d) = std::env::var_os(var) {
        return PathBuf::from(d);
    }
    let home = engine::paths::home().unwrap_or_else(|| PathBuf::from("."));
    home.join(".wobble").join(name)
}

/// Installed, AND synced against the environment this build needs: a
/// checkout whose `pyproject.toml` is an older `PYPROJECT` shows as not
/// installed, so `install…` comes back and re-syncs it.
pub fn installed(tool: Tool, dir: &Path) -> bool {
    let code = match tool {
        Tool::Vda => dir.join("video_depth_anything").join("video_depth.py"),
        Tool::Pro2 => dir.join("src").join("depth_pro_2").join("model.py"),
    };
    code.is_file()
        && dir.join(".venv").is_dir()
        && std::fs::read_to_string(dir.join("pyproject.toml")).is_ok_and(|t| t == tool.pyproject())
}

/// A suite's stand-in command replaces the tool (and its weights).
fn stand_in() -> bool {
    std::env::var_os("WOBBLE_DEPTH_CMD").is_some()
}

pub enum Phase {
    /// Rendering input 0 (a `SubtreeBake` in `Gfx::graph_bake`).
    Render,
    Running { rx: Receiver<Msg>, cancel: Arc<AtomicBool>, done: usize, total: usize },
    Failed(String),
    Done(String),
}

pub struct Run {
    pub node: u64,
    pub dir: PathBuf,
    pub phase: Phase,
    pub log: Vec<String>,
}

impl Run {
    pub fn active(&self) -> bool {
        matches!(self.phase, Phase::Render | Phase::Running { .. })
    }
}

pub struct Install {
    pub tool: Tool,
    rx: Receiver<Msg>,
    pub log: Vec<String>,
    pub result: Option<Result<(), String>>,
}

#[derive(Default)]
pub struct Jobs {
    pub run: Option<Run>,
    pub install: Option<Install>,
    /// Each loaded result's `meta.json` (disparity clip maximum, metric),
    /// by the path it was loaded from; read once at load, handed to the
    /// node's uniform every frame.
    pub stats: std::collections::HashMap<String, (f32, bool)>,
}

/// What a result directory's layout is. Bumped when it changes, which is
/// part of the directory key: a result of another layout is never adopted
/// (it is STALE, and a run makes a new one). 2: one half-float `Z` channel
/// plus `meta.json` (1 was three float32 channels, 74 MB a 4K frame).
pub const LAYOUT: u32 = 2;

/// A result's `meta.json`: (disparity clip maximum, metres). Absent or
/// unreadable answers the identity (1, relative), which shows the stored
/// value itself rather than failing.
pub fn read_meta(out: &Path) -> (f32, bool) {
    let v: Option<serde_json::Value> =
        std::fs::read(out.join("meta.json")).ok().and_then(|b| serde_json::from_slice(&b).ok());
    let top = v.as_ref().and_then(|v| v["top"].as_f64()).filter(|t| *t > 1e-8).unwrap_or(1.0) as f32;
    let metric = v.as_ref().and_then(|v| v["metric"].as_bool()).unwrap_or(false);
    (top, metric)
}

/// What the panel is told, built by `Gfx::depth_view`.
pub struct View {
    /// The tool the node's model runs on.
    pub tool: Tool,
    pub dir: String,
    pub installed: bool,
    pub uv: bool,
    pub installing: Option<String>,
    pub install_result: Option<Result<(), String>>,
    /// The picked model's weights: ready, or why not (the gate above the
    /// panel has the button).
    pub weights: Result<(), String>,
    pub state: ViewState,
    pub log: Vec<String>,
    pub has_input: bool,
    pub loaded: Option<(usize, bool)>,
    pub on_disk: bool,
    pub busy_elsewhere: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ViewState {
    Idle,
    Rendering(u64, u64),
    Running(usize, usize),
    Failed(String),
    Done(String),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Action {
    Run,
    Cancel,
    Install,
    Dismiss,
}

pub fn params(kind: &NodeKind) -> Option<&DepthParams> {
    match kind {
        NodeKind::Depth(p) => Some(p),
        _ => None,
    }
}

/// Every weights file the picked model needs (`DepthParams::models`, the
/// network first) on disk, or why not. Under the stand-in the weights are
/// optional (it reads none): an empty list.
fn weights(p: &DepthParams) -> Result<Vec<PathBuf>, String> {
    let mut out = vec![];
    for id in p.models() {
        match engine::models::ready_path(id) {
            Ok(path) => out.push(path),
            Err(_) if stand_in() => return Ok(vec![]),
            Err(e) => return Err(e.to_string()),
        }
    }
    Ok(out)
}

/// The command that runs the driver: `$WOBBLE_DEPTH_CMD` (a suite's
/// stand-in, whitespace-split) or `uv run --project <dir> python <driver>`.
fn command(tool: Tool, dir: &Path) -> Result<std::process::Command, String> {
    if let Ok(s) = std::env::var("WOBBLE_DEPTH_CMD") {
        let mut parts = s.split_whitespace();
        let exe = parts.next().ok_or("WOBBLE_DEPTH_CMD is empty")?;
        let mut c = std::process::Command::new(exe);
        c.args(parts);
        return Ok(c);
    }
    let uv = uv().ok_or(UV_MISSING)?;
    // Depth Pro 2's driver imports Video Depth Anything's EXR reader and
    // writer: both go beside it.
    let mut driver = dir.join(DRIVER_NAME);
    std::fs::write(&driver, DRIVER).map_err(|e| format!("writing the driver: {e}"))?;
    if tool == Tool::Pro2 {
        driver = dir.join(PRO2_DRIVER_NAME);
        std::fs::write(&driver, PRO2_DRIVER).map_err(|e| format!("writing the driver: {e}"))?;
    }
    let mut c = std::process::Command::new(uv);
    c.arg("run").arg("--project").arg(dir).arg("python").arg(&driver);
    c.current_dir(dir);
    Ok(c)
}

impl Gfx {
    /// Where a run of `node` at the current recipe lands. None while the
    /// input cannot be rendered.
    pub(crate) fn depth_dir(&self, node: u64) -> Option<PathBuf> {
        let p = self.egraph.node(node).and_then(|n| params(&n.kind))?;
        let input = self.egraph.input_of(node, 0)?;
        let k0 = self.subtree_key_of(input)?;
        let ((a, b), _) = self.range_of(input);
        let size = self.node_resolution(input, &mut std::collections::HashSet::new());
        let recipe = serde_json::to_vec(&(p.recipe(), a, b, self.handles, size, LAYOUT)).ok()?;
        let key = engine::diskcache::content_hash(&[&k0.to_le_bytes(), &recipe]);
        let root = match &self.disk_cache {
            Some(c) => c.root().join("depth"),
            None => engine::paths::temp_dir().join("wobble-depth"),
        };
        Some(root.join(format!("{key:016x}")))
    }

    /// Play a finished result: its statistics first, then the frames.
    fn depth_load(&mut self, node: u64, out: &Path) {
        let path = out.to_string_lossy().into_owned();
        self.jobs.depth.stats.insert(path.clone(), read_meta(out));
        self.load_read(node, &path);
    }

    pub(crate) fn depth_view(&self, node: u64) -> View {
        let p = self.egraph.node(node).and_then(|n| params(&n.kind));
        let tool = p.map(Tool::of).unwrap_or(Tool::Vda);
        let dir = install_dir(tool);
        let install = self.jobs.depth.install.as_ref().filter(|i| i.tool == tool);
        let mine = self.jobs.depth.run.as_ref().filter(|r| r.node == node);
        let state = match mine.map(|r| &r.phase) {
            None => ViewState::Idle,
            Some(Phase::Render) => match &self.graph_bake {
                Some(b) if b.node == node => ViewState::Rendering(b.next.saturating_sub(b.from), b.len.saturating_sub(b.from)),
                _ => ViewState::Rendering(0, 0),
            },
            Some(Phase::Running { done, total, .. }) => ViewState::Running(*done, *total),
            Some(Phase::Failed(e)) => ViewState::Failed(e.clone()),
            Some(Phase::Done(s)) => ViewState::Done(s.clone()),
        };
        let want = self.depth_dir(node).map(|d| d.join("Depth"));
        let loaded = self.loaded.get(&node).map(|l| {
            let current = want.as_ref().is_some_and(|d| l.path == d.to_string_lossy());
            (l.seq.frames.len(), current)
        });
        View {
            tool,
            dir: dir.to_string_lossy().into_owned(),
            installed: installed(tool, &dir) || stand_in(),
            uv: uv().is_some(),
            installing: install.filter(|i| i.result.is_none()).map(|i| i.log.last().cloned().unwrap_or_default()),
            install_result: install.and_then(|i| i.result.clone()),
            weights: p.map(|p| weights(p).map(|_| ())).unwrap_or(Ok(())),
            state,
            log: mine.map(|r| r.log.clone()).unwrap_or_default(),
            has_input: self.egraph.input_of(node, 0).is_some(),
            loaded,
            on_disk: want.as_ref().is_some_and(|d| engine::bakes::is_baked(d)),
            busy_elsewhere: self.jobs.depth.run.as_ref().is_some_and(|r| r.node != node && r.active()),
        }
    }

    pub(crate) fn depth_action(&mut self, node: u64, a: Action) {
        match a {
            Action::Run => self.depth_start(node),
            Action::Cancel => {
                if let Some(r) = self.jobs.depth.run.take() {
                    match r.phase {
                        Phase::Running { cancel, .. } => cancel.store(true, Relaxed),
                        Phase::Render => {
                            if self.graph_bake.as_ref().is_some_and(|b| b.node == r.node) {
                                self.graph_bake = None;
                            }
                        }
                        _ => {}
                    }
                }
            }
            Action::Install => {
                let tool = self.egraph.node(node).and_then(|n| params(&n.kind)).map(Tool::of).unwrap_or(Tool::Vda);
                self.depth_install(tool)
            }
            Action::Dismiss => {
                if self.jobs.depth.run.as_ref().is_some_and(|r| r.node == node && !r.active()) {
                    self.jobs.depth.run = None;
                }
                if self.jobs.depth.install.as_ref().is_some_and(|i| i.result.is_some()) {
                    self.jobs.depth.install = None;
                }
            }
        }
    }

    fn depth_fail(&mut self, node: u64, e: &str) {
        match self.jobs.depth.run.as_mut().filter(|r| r.node == node) {
            Some(r) => r.phase = Phase::Failed(e.to_string()),
            None => self.jobs.depth.run = Some(Run { node, dir: PathBuf::new(), phase: Phase::Failed(e.into()), log: vec![] }),
        }
    }

    /// `install…`: fetch the pinned checkout and build its environment, on
    /// a thread, its output streamed to the panel.
    fn depth_install(&mut self, tool: Tool) {
        if self.jobs.depth.install.as_ref().is_some_and(|i| i.result.is_none()) {
            return;
        }
        let dir = install_dir(tool);
        let (tx, rx) = std::sync::mpsc::channel();
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (&dir, &tx);
            self.jobs.depth.install = Some(Install { tool, rx, log: vec![], result: Some(Err("the tool runs as a process: not in the web build".into())) });
            return;
        }
        #[cfg_attr(target_arch = "wasm32", allow(unreachable_code))]
        let Some(uv) = uv() else {
            self.jobs.depth.install = Some(Install { tool, rx, log: vec![], result: Some(Err(UV_MISSING.into())) });
            return;
        };
        let spawned = std::thread::Builder::new()
            .name("depth-install".into())
            .spawn(move || {
                let result = install_into(tool, &dir, &uv, &tx);
                let _ = tx.send(match result {
                    Ok(()) => Msg::Done,
                    Err(e) => Msg::Failed(e),
                });
            })
            .is_ok();
        self.jobs.depth.install = Some(Install { tool, rx, log: vec![], result: (!spawned).then_some(Err("could not start a thread".into())) });
    }

    /// The `run` click: adopt a result on disk, else render the input.
    fn depth_start(&mut self, node: u64) {
        if self.jobs.depth.run.as_ref().is_some_and(|r| r.active()) {
            return;
        }
        #[cfg(target_arch = "wasm32")]
        return self.depth_fail(node, "Depth runs as a process: not in the web build");
        #[cfg_attr(target_arch = "wasm32", allow(unreachable_code))]
        let Some(input) = self.egraph.input_of(node, 0) else {
            return self.depth_fail(node, "wire a picture into the input");
        };
        let Some(dir) = self.depth_dir(node) else {
            return self.depth_fail(node, "the input is not renderable yet");
        };
        let out = dir.join("Depth");
        if engine::bakes::is_baked(&out) {
            self.depth_load(node, &out);
            self.playback_dirty = true;
            self.jobs.depth.run = Some(Run { node, dir, phase: Phase::Done("loaded the result on disk".into()), log: vec![] });
            return;
        }
        let tool = self.egraph.node(node).and_then(|n| params(&n.kind)).map(Tool::of).unwrap_or(Tool::Vda);
        if !installed(tool, &install_dir(tool)) && !stand_in() {
            return self.depth_fail(node, &format!("{} is not installed: press install…", tool.name()));
        }
        if let Some(Err(e)) = self.egraph.node(node).and_then(|n| params(&n.kind)).map(weights) {
            return self.depth_fail(node, &e);
        }
        self.jobs.depth.run = Some(Run { node, dir: dir.clone(), phase: Phase::Render, log: vec![] });
        let size = self.node_resolution(input, &mut std::collections::HashSet::new());
        if let Err(e) = self.render_input_frames(node, input, dir, size, crate::BakeSink::Depth) {
            self.depth_fail(node, &e);
        }
    }

    /// The bake loop finished rendering the input: run the tool.
    pub(crate) fn depth_render_finished(&mut self, node: u64) {
        let Some(r) = self.jobs.depth.run.as_ref().filter(|r| r.node == node) else { return };
        let dir = r.dir.clone();
        let Some(p) = self.egraph.node(node).and_then(|n| params(&n.kind)).cloned() else {
            return self.depth_fail(node, "the node is gone");
        };
        let w = match weights(&p) {
            Ok(w) => w,
            Err(e) => return self.depth_fail(node, &e),
        };
        let tool = Tool::of(&p);
        let mut cmd = match command(tool, &install_dir(tool)) {
            Ok(c) => c,
            Err(e) => return self.depth_fail(node, &e),
        };
        let mut w = w.into_iter();
        cmd.arg("--shot").arg(&dir).arg("--size").arg(p.size_arg().to_string());
        cmd.arg("--weights").arg(w.next().unwrap_or_default());
        match tool {
            Tool::Vda => {
                cmd.arg("--encoder").arg(p.model.encoder());
                if p.model.metric() {
                    cmd.arg("--metric");
                }
            }
            Tool::Pro2 => {
                cmd.arg("--preset").arg(p.model.encoder()).arg("--focal").arg(w.next().unwrap_or_default());
            }
        }
        let (tx, rx) = std::sync::mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        if let Err(e) = spawn(cmd, tx, cancel.clone(), "depth") {
            return self.depth_fail(node, &e);
        }
        if let Some(r) = self.jobs.depth.run.as_mut() {
            r.phase = Phase::Running { rx, cancel, done: 0, total: 0 };
        }
    }

    /// Every redraw: drain the tool's messages, catch a render that died,
    /// adopt a finished result.
    pub(crate) fn depth_poll(&mut self) {
        if let Some(i) = self.jobs.depth.install.as_mut() {
            while let Ok(m) = i.rx.try_recv() {
                match m {
                    Msg::Line(l) => push_log(&mut i.log, l),
                    Msg::Done => i.result = Some(Ok(())),
                    Msg::Failed(e) => i.result = Some(Err(e)),
                    Msg::Frame(..) => {}
                }
            }
        }
        let Some(r) = self.jobs.depth.run.as_mut() else { return };
        let node = r.node;
        if matches!(r.phase, Phase::Render) && self.graph_bake.is_none() {
            let why = self.load_errors.get(&node).cloned().unwrap_or_else(|| "the input stopped rendering".into());
            r.phase = Phase::Failed(why);
            return;
        }
        let mut end = None;
        if let Phase::Running { rx, done, total, .. } = &mut r.phase {
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
                let out = r.dir.join("Depth");
                if let Err(e) = engine::bakes::mark_baked(&out) {
                    r.phase = Phase::Failed(format!("marking the result done: {e}"));
                    return;
                }
                r.phase = Phase::Done("depth done".into());
                self.depth_load(node, &out);
                self.playback_dirty = true;
            }
            Some(Err(e)) => r.phase = Phase::Failed(e),
            None => {}
        }
    }

    /// After a load, an undo, a paste: every Depth node whose result for
    /// the current recipe is on disk plays it.
    pub(crate) fn depth_adopt(&mut self) {
        let ids: Vec<u64> = self
            .egraph
            .nodes
            .iter()
            .filter(|(id, n)| {
                matches!(n.kind, NodeKind::Depth(_)) && !self.loaded.contains_key(id) && !self.load_errors.contains_key(id)
            })
            .map(|(id, _)| *id)
            .collect();
        for id in ids {
            let Some(dir) = self.depth_dir(id) else { continue };
            let out = dir.join("Depth");
            if engine::bakes::is_baked(&out) {
                self.depth_load(id, &out);
                self.playback_dirty = true;
            }
        }
    }
}

/// Fetch the pinned commit into `dir` (fresh, or an earlier install at
/// another commit), write our project file and `uv sync` it.
fn install_into(tool: Tool, dir: &Path, uv: &Path, tx: &Sender<Msg>) -> Result<(), String> {
    let (repo, commit) = (tool.repo(), tool.commit());
    let run = |mut c: std::process::Command| -> Result<(), String> {
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
        if s.success() {
            Ok(())
        } else {
            Err(format!("exited with {s}"))
        }
    };
    let git = |args: &[&str]| {
        let mut c = std::process::Command::new("git");
        c.args(args).current_dir(dir);
        c
    };
    std::fs::create_dir_all(dir).map_err(|e| format!("{e}"))?;
    if !dir.join(".git").is_dir() {
        let _ = tx.send(Msg::Line(format!("git init {}", dir.display())));
        run(git(&["init", "-q"])).map_err(|e| format!("git init: {e}"))?;
    }
    let _ = tx.send(Msg::Line(format!("git fetch {repo} {}", &commit[..12])));
    run(git(&["fetch", "-q", "--depth", "1", repo, commit])).map_err(|e| format!("git fetch: {e}"))?;
    run(git(&["checkout", "-q", "--force", "FETCH_HEAD"])).map_err(|e| format!("git checkout: {e}"))?;
    std::fs::write(dir.join("pyproject.toml"), tool.pyproject()).map_err(|e| format!("writing pyproject.toml: {e}"))?;
    let _ = tx.send(Msg::Line("uv sync (torch and friends: a few GB, a while)".into()));
    let mut c = std::process::Command::new(uv);
    c.arg("sync").arg("--python").arg("3.12").current_dir(dir);
    run(c).map_err(|e| format!("uv sync: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_driver_speaks_the_protocol_the_reader_parses() {
        for w in ["WOBBLE frame", "WOBBLE done", "WOBBLE error", "--shot", "--weights", "--encoder", "--metric", "--size", "compute_scale_and_shift", "get_interpolate_frames", "\"Depth\""] {
            assert!(DRIVER.contains(w), "{w}");
        }
    }

    #[test]
    fn the_driver_never_downloads() {
        // The weights come through the catalog, by path. (Depth Pro 2's
        // `from_pretrained` / `create_model_and_transforms` would fetch:
        // the driver builds the network itself and loads the files given.)
        for (name, driver) in [("wobble_depth.py", DRIVER), ("wobble_depthpro2.py", PRO2_DRIVER)] {
            let code: String = driver.lines().filter(|l| !l.trim_start().starts_with('#')).collect::<Vec<_>>().join("\n");
            let code = code.split("\"\"\"").enumerate().filter(|(i, _)| i % 2 == 0).map(|(_, s)| s).collect::<String>();
            for w in ["urllib", "requests", "hf_hub_download", "huggingface_hub", "wget", "http", "from_pretrained", "create_model_and_transforms", "snapshot_download"] {
                assert!(!code.contains(w), "{name} mentions {w}");
            }
        }
    }

    #[test]
    fn the_pro2_driver_speaks_the_protocol_and_takes_both_weights() {
        for w in ["say(\"frame\"", "say(\"done\")", "\"error\"", "--shot", "--weights", "--focal", "--preset", "--size", "infer_video", "\"Depth\"", "from wobble_depth import", "\"metric\": True"] {
            assert!(PRO2_DRIVER.contains(w), "{w}");
        }
        // Every name it imports from the shared driver is defined there.
        let import = PRO2_DRIVER.split("from wobble_depth import (").nth(1).unwrap().split(')').next().unwrap();
        for name in import.split(',').map(|n| n.trim()).filter(|n| !n.is_empty() && !n.starts_with('#')) {
            assert!(DRIVER.contains(&format!("def {name}(")) || DRIVER.contains(&format!("{name} = ")) || DRIVER.contains(&format!("class {name}(")), "{name}");
        }
        for m in engine::graph::DepthModel::ALL.into_iter().filter(|m| m.pro2()) {
            assert_eq!(Tool::of(&DepthParams { model: m, ..Default::default() }), Tool::Pro2);
        }
        assert_eq!(Tool::of(&DepthParams::default()), Tool::Vda);
    }

    /// The driver's EXR reader on a frame from OUR writer (the one every
    /// Input render lands through), a write/read round trip of its output,
    /// and a truncated frame named with its size and both readers' reasons.
    /// Needs the tool installed (`install…`); skips, saying so, without it.
    #[test]
    fn the_driver_reads_our_exrs_and_names_a_bad_one() {
        let dir = install_dir(Tool::Vda);
        let Some(uv) = uv().filter(|_| installed(Tool::Vda, &dir)) else {
            eprintln!("skipping: Video Depth Anything is not installed at {}", dir.display());
            return;
        };
        let tmp = std::env::temp_dir().join(format!("wobble-depth-io-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        let (w, h) = (6usize, 4usize);
        let px: Vec<half::f16> = (0..w * h).flat_map(|i| [0.25, 0.5, i as f32, 1.0].map(half::f16::from_f32)).collect();
        let good = tmp.join("good.exr");
        engine::bakes::write_rgba_f16_exr_tagged(&good, w, h, w * 4, &px, None).unwrap();
        let bytes = std::fs::read(&good).unwrap();
        std::fs::write(tmp.join("bad.exr"), &bytes[..bytes.len() / 3]).unwrap();
        std::fs::write(tmp.join(DRIVER_NAME), DRIVER).unwrap();
        let script = r#"
import importlib.util, os, sys
d = sys.argv[1]
spec = importlib.util.spec_from_file_location("drv", os.path.join(d, "wobble_depth.py"))
drv = importlib.util.module_from_spec(spec); spec.loader.exec_module(drv)
import numpy as np
m = drv.read_rgb(os.path.join(d, "good.exr"))
assert m.shape == (4, 6, 3), m.shape
assert abs(m[0, 0, 0] - 0.25) < 1e-3 and abs(m[0, 0, 1] - 0.5) < 1e-3 and m[3, 5, 2] == 23.0, m[0, 0]
r = np.full((4, 6), 2.5, np.float32)
drv.write_z(os.path.join(d, "out.exr"), r)
import OpenEXR
with OpenEXR.File(os.path.join(d, "out.exr"), separate_channels=True) as f:
    ch = f.channels()
    assert list(ch) == ["Z"], list(ch)
    assert ch["Z"].pixels.dtype == np.float16 and np.all(ch["Z"].pixels == 2.5), ch["Z"].pixels
o = drv.read_rgb(os.path.join(d, "out.exr"))
assert np.allclose(o, 2.5), o[0, 0]
assert not os.path.exists(os.path.join(d, "out.exr.part"))
try:
    drv.read_rgb(os.path.join(d, "bad.exr"))
    raise SystemExit("a truncated frame read")
except drv.Unreadable as e:
    s = str(e)
    assert "bytes" in s and "OpenEXR" in s and "OpenCV" in s, s
print("ok")
"#;
        let out = std::process::Command::new(uv)
            .args(["run", "--no-sync", "--project"])
            .arg(&dir)
            .args(["python", "-c", script])
            .arg(&tmp)
            .output()
            .unwrap();
        let _ = std::fs::remove_dir_all(&tmp);
        let text = format!("{}{}", String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
        assert!(out.status.success() && text.contains("ok"), "{text}");
    }

    #[test]
    fn the_install_is_pinned() {
        for c in [COMMIT, PRO2_COMMIT] {
            assert_eq!(c.len(), 40);
            assert!(c.chars().all(|c| c.is_ascii_hexdigit()));
        }
        assert!(PRO2_PYPROJECT.contains("\"opencv-python<5\"") && PRO2_PYPROJECT.contains("OpenEXR") && PRO2_PYPROJECT.contains("safetensors"));
        assert!(PRO2_PYPROJECT.contains("where = [\"src\"]"), "the package still builds from the checkout");
        assert!(PYPROJECT.contains("torch>=2.5"), "MPS autocast needs a torch that has it");
        // 5.x's Windows wheel cannot read or write EXR; the driver needs
        // OpenEXR as its reader. Keep the pin identical to the Windows
        // checkout's ("opencv-python<5").
        assert!(PYPROJECT.contains("\"opencv-python<5\"") && PYPROJECT.contains("OpenEXR"));
    }

    #[test]
    fn the_far_cap_matches_the_engine() {
        assert!(DRIVER.contains(&format!("FAR = {:.1}", engine::nodes::depth::FAR)));
    }
}

// ------------------------------------------------------------- the suite

/// What the stand-in writes: one stored disparity and its clip maximum.
/// Through the node they read 4 (depth), 0.5 (disparity) and 0.25 (near):
/// distinct numbers, so a pick that reads the wrong output cannot pass.
pub const FAKE_RAW: f32 = 0.5;
pub const FAKE_TOP: f32 = 2.0;

/// `--depth-fake <args>`: the stand-in tool `--test-depth` runs in place
/// of Video Depth Anything, so the whole pipeline — render, process, adopt,
/// playback — is exercised with no Python, no torch and no weights. Each
/// `Input` frame becomes a `Depth` frame of the same size, the driver's
/// layout (one `Z` channel holding `FAKE_RAW`, `meta.json` with
/// `FAKE_TOP`), and it speaks the driver's protocol line for line.
pub fn fake(args: &[String]) -> anyhow::Result<()> {
    use engine::decode::ExrDecoder;
    let arg = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let shot = PathBuf::from(arg("--shot").ok_or_else(|| anyhow::anyhow!("--shot"))?);
    let dec = engine::decode::AnyDecoder::new(false);
    let mut inputs: Vec<PathBuf> = std::fs::read_dir(shot.join("Input"))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "exr"))
        .collect();
    inputs.sort();
    let out_dir = shot.join("Depth");
    std::fs::create_dir_all(&out_dir)?;
    let total = inputs.len();
    println!("WOBBLE frame 0 {total}");
    for (i, path) in inputs.iter().enumerate() {
        let info = dec.read_header(path)?;
        let (w, h) = (info.width as usize, info.height as usize);
        let px = [FAKE_RAW, 0.0, 0.0, 1.0].map(half::f16::from_f32);
        let out: Vec<half::f16> = (0..w * h).flat_map(|_| px).collect();
        engine::bakes::write_z_f32_exr(&out_dir.join(path.file_name().unwrap()), w, h, w * 4, &out)?;
        println!("WOBBLE frame {} {total}", i + 1);
    }
    std::fs::write(out_dir.join("meta.json"), serde_json::json!({ "layout": LAYOUT, "top": FAKE_TOP, "metric": false }).to_string())?;
    println!("WOBBLE done");
    Ok(())
}

pub struct TestDepth {
    stage: u32,
    t0: web_time::Instant,
    since: web_time::Instant,
    node: u64,
    last: Option<f32>,
    at: u64,
    render: Option<(u64, PathBuf)>,
}

impl Default for TestDepth {
    fn default() -> Self {
        let now = web_time::Instant::now();
        Self { stage: 0, t0: now, since: now, node: 0, last: None, at: 0, render: None }
    }
}

impl Gfx {
    /// `--test-depth`: a Depth node end to end in the live app, against the
    /// stand-in above. Unrun it outputs zero (no depth, no work); `run`
    /// renders the plate, runs the process, adopts `Depth`; the output is
    /// the depth lane on the DATA pipe, and the pick moves it to disparity
    /// and to near WITHOUT a new directory (a uniform). A Defocus's Z port
    /// takes the wire. Then a model change is a new
    /// recipe: STALE until re-run. Waits are wall-clock.
    pub(crate) fn drive_test_depth(&mut self) {
        let Some(mut t) = self.test.depth.take() else { return };
        let fail = |m: String| -> ! {
            println!("DEPTH TEST FAIL: {m}");
            std::process::exit(1);
        };
        if t.t0.elapsed().as_secs() > 120 {
            fail(format!("timeout at stage {}", t.stage));
        }
        let next = |t: &mut TestDepth, s: u32| {
            t.stage = s;
            t.since = web_time::Instant::now();
            t.last = None;
        };
        // A reading two probes agree on, 150 ms apart, at the playhead.
        let stable = |me: &mut Self, t: &mut TestDepth| -> Option<f32> {
            if me.force_display || me.eval_tick() != me.timeline_pos || t.since.elapsed().as_millis() < 150 {
                return None;
            }
            let v = me.probe_output_box(100, 100, 8, 8, 0)?;
            t.since = web_time::Instant::now();
            match t.last.replace(v) {
                Some(p) if (p - v).abs() < 1e-5 => {
                    t.last = None;
                    Some(v)
                }
                _ => None,
            }
        };
        let phase = |me: &Self| match me.jobs.depth.run.as_ref().map(|r| &r.phase) {
            Some(Phase::Failed(e)) => Err(e.clone()),
            Some(Phase::Done(_)) => Ok(true),
            _ => Ok(false),
        };
        let set_out = |me: &mut Self, node: u64, o: engine::graph::DepthOut| {
            if let Some(NodeKind::Depth(p)) = me.egraph.node_mut(node).map(|n| &mut n.kind) {
                p.output = o;
            }
        };
        match t.stage {
            0 if self.playback.is_some() && self.displayed_total >= 2 => {
                let exe = std::env::current_exe().unwrap_or_else(|e| fail(format!("current_exe: {e}")));
                std::env::set_var("WOBBLE_DEPTH_CMD", format!("{} --depth-fake", exe.display()));
                let read = self
                    .egraph
                    .nodes
                    .iter()
                    .find(|(_, n)| matches!(n.kind, NodeKind::Read { .. }))
                    .map(|(id, _)| *id)
                    .unwrap_or_else(|| fail("no Read in the default graph".into()));
                let slot = self.next_read_slot;
                self.next_read_slot += 1;
                let k = self.egraph.add(NodeKind::Depth(DepthParams::new(slot)), egui::pos2(360.0, 200.0));
                self.egraph.connect(k, 0, read);
                if self.egraph.nodes[&k].kind.output_kind() != engine::graph::PortKind::Data {
                    fail("the Depth node's output is not the data pipe".into());
                }
                self.egraph.view = Some(k);
                self.playing = false;
                t.node = k;
                next(&mut t, 1);
            }
            1 => {
                if let Some(v) = stable(self, &mut t) {
                    if v.abs() > 1e-6 {
                        fail(format!("an unrun Depth node reads {v}, not 0"));
                    }
                    t.at = self.timeline_pos;
                    self.depth_action(t.node, Action::Run);
                    next(&mut t, 2);
                }
            }
            2 => match phase(self) {
                Err(e) => fail(format!("the run failed: {e}")),
                Ok(true) => {
                    if !self.loaded.contains_key(&t.node) {
                        fail("the result did not load into the node's reader".into());
                    }
                    self.seek_to(t.at);
                    next(&mut t, 3);
                }
                Ok(false) => {}
            },
            // Depth, then disparity, then near: each output of the fake,
            // derived in the shader from ONE stored value and meta.json.
            3..=5 => {
                if let Some(v) = stable(self, &mut t) {
                    let lane = (t.stage - 3) as usize;
                    let outs = [engine::graph::DepthOut::Depth, engine::graph::DepthOut::Disparity, engine::graph::DepthOut::Near];
                    let want = DepthParams { output: outs[lane], top: FAKE_TOP, ..Default::default() }.derive(FAKE_RAW);
                    if (v - want).abs() > 2e-3 {
                        fail(format!("output lane {lane} reads {v:.4}, the result holds {want}"));
                    }
                    let dir = self.depth_dir(t.node);
                    let o = [engine::graph::DepthOut::Disparity, engine::graph::DepthOut::Near, engine::graph::DepthOut::Depth];
                    set_out(self, t.node, o[lane]);
                    if self.depth_dir(t.node) != dir {
                        fail("the output pick changed the result's directory".into());
                    }
                    let s = t.stage + 1;
                    next(&mut t, s);
                }
            }
            // A Render fed the data pipe writes it as data: one float32
            // channel `Z`, the node's value exactly, no colour transform.
            6 => {
                let dir = std::env::temp_dir().join(format!("wobble-test-depth-render-{}", std::process::id()));
                let _ = std::fs::remove_dir_all(&dir);
                let wp = engine::graph::WriteParams {
                    path: dir.join("z.####.exr").to_string_lossy().into_owned(),
                    range: Some((t.at, t.at)),
                    range_mode: Some(engine::graph::WriteRange::Range),
                    ..Default::default()
                };
                let w = self.egraph.add(NodeKind::Write(wp.clone()), egui::pos2(360.0, 320.0));
                if editor::connect_or_bridge(&mut self.egraph, t.node, 0, w, 0, 1.0).is_some()
                    || self.egraph.input_of(w, 0) != Some(t.node)
                {
                    fail("a Render did not take the Depth's data wire directly".into());
                }
                if !self.write_takes_data(w) {
                    fail("the Render does not see its input as data".into());
                }
                self.start_write(w, &wp);
                if let Some(e) = self.load_errors.get(&w) {
                    fail(format!("the render did not start: {e}"));
                }
                t.render = Some((w, dir));
                next(&mut t, 7);
            }
            7 => {
                let (w, dir) = t.render.clone().unwrap();
                if let Some(e) = self.load_errors.get(&w) {
                    fail(format!("the render failed: {e}"));
                }
                if self.graph_bake.is_none() && self.written.contains_key(&w) {
                    let file = self.written[&w].first().cloned().unwrap_or_else(|| fail("no file written".into()));
                    let img = exr::prelude::read_all_flat_layers_from_file(&file)
                        .unwrap_or_else(|e| fail(format!("reading {}: {e}", file.display())));
                    let names: Vec<String> =
                        img.layer_data[0].channel_data.list.iter().map(|c| c.name.to_string()).collect();
                    if names != ["Z"] || img.attributes.chromaticities.is_some() {
                        fail(format!("a data render wrote channels {names:?} (chromaticities {:?})", img.attributes.chromaticities));
                    }
                    let want = DepthParams { top: FAKE_TOP, ..Default::default() }.derive(FAKE_RAW);
                    match &img.layer_data[0].channel_data.list[0].sample_data {
                        exr::prelude::FlatSamples::F32(v) if v.iter().all(|x| (x - want).abs() < 2e-3) => {}
                        exr::prelude::FlatSamples::F32(v) => {
                            fail(format!("the rendered Z reads {:?}…, the node outputs {want}", &v[..v.len().min(4)]))
                        }
                        _ => fail("the rendered Z is not float32".into()),
                    }
                    let _ = std::fs::remove_dir_all(&dir);
                    self.egraph.remove(w);
                    next(&mut t, 8);
                }
            }
            8 => {
                // A Defocus's Z port takes it, by the one wiring rule.
                let d = self.egraph.add(NodeKind::Defocus(Default::default()), egui::pos2(560.0, 200.0));
                let zk = self.egraph.nodes[&d].kind.input_kind(2);
                if !self.egraph.nodes[&t.node].kind.output_kind().can_feed(zk) {
                    fail(format!("a Depth output cannot feed a Defocus's Z port ({zk:?})"));
                }
                let v = self.depth_view(t.node);
                if !matches!(v.loaded, Some((_, true))) {
                    fail(format!("the result should be current, the view says {:?}", v.loaded));
                }
                let old = self.depth_dir(t.node);
                if let Some(NodeKind::Depth(p)) = self.egraph.node_mut(t.node).map(|n| &mut n.kind) {
                    p.model = engine::graph::DepthModel::SmallMetric;
                }
                if self.depth_dir(t.node) == old {
                    fail("a new model did not change the result's directory".into());
                }
                let v = self.depth_view(t.node);
                if !matches!(v.loaded, Some((_, false))) {
                    fail(format!("a new model should show the result STALE, the view says {:?}", v.loaded));
                }
                println!(
                    "DEPTH TEST PASS: unrun the node outputs 0; run rendered the plate, ran the tool and adopted \
                     its one-channel result; the data output derives depth, disparity and near by a uniform pick \
                     (no new directory); a Render fed it wrote one exact float32 Z channel with no colour \
                     transform; it feeds a Defocus's Z port; a new model is a new recipe, STALE until re-run"
                );
                std::process::exit(0);
            }
            _ => {}
        }
        self.test.depth = Some(t);
    }
}
