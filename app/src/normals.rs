//! The Normals node's app half: where NormalCrafter lives, installing it,
//! rendering the node's input to a shot directory, running the tool as a
//! process, and adopting what it wrote. The node itself (params, codegen)
//! is `engine::nodes::normals`; the panel and glyph `crate::nodes::normals`.
//! The shape is the Depth node's (`crate::depth`): CorridorKey's process
//! plumbing (`crate::corridorkey::spawn`), the Upscale node's input render
//! (`Gfx::render_input_frames`).
//!
//! A run is two phases, each explicit and cancellable:
//!
//! 1. **render** — a `SubtreeBake` of input 0 over the input's frame range,
//!    at the input's own format, written as linear Rec.709 EXRs to
//!    `<dir>/Input`;
//! 2. **run** — the tool, one process, driven by `tools/wobble_normals.py`
//!    (embedded, written next to the checkout on every run); `<dir>/Normals`
//!    is adopted as the node's reader when it says done.
//!
//! `<dir>` is keyed on everything that shapes the pixels — the input's
//! subtree key and format, the recipe (resolution, window, stride), the
//! range — so an edit upstream shows as STALE and a re-run lands in a new
//! directory. The flips and the 0..1 encoding are NOT in it: uniforms.
//!
//! **Nothing installs or downloads without a click.** `install…` fetches
//! the pinned checkout into `~/.wobble/NormalCrafter` (or
//! `$WOBBLE_NORMALS_DIR`), writes our `pyproject.toml` beside it (upstream
//! pins torch 2.0 and xformers, too old for a current GPU) and runs `uv
//! sync`. The weights are the TOOL's (the CorridorKey precedent): the first
//! `run` fetches them from Hugging Face at a pinned revision into the
//! user's Hugging Face cache, which the panel says before the click.

use crate::corridorkey::{push_log, spawn, uv, Msg, UV_MISSING};
use crate::Gfx;
use engine::graph::{NodeKind, NormalsParams};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;

pub const DRIVER: &str = include_str!("../../tools/wobble_normals.py");
const DRIVER_NAME: &str = "wobble_normals.py";
pub const REPO: &str = "https://github.com/Binyr/NormalCrafter.git";
/// The commit the driver is written against: it builds the pipeline from
/// `normalcrafter.normal_crafter_ppl` and calls its `_encode_image`,
/// `ecnode_video_vae` and `generate`, which a later commit may rename.
/// Bump it and re-test, never float it.
pub const COMMIT: &str = "75af9887a2cb14cd1ce3883c5773bc296565777c";
pub const LICENCE_URL: &str = "https://github.com/Binyr/NormalCrafter/blob/main/LICENSE";
/// The weights' page, and the licence they inherit (an SVD fine-tune).
pub const MODEL_PAGE: &str = "https://huggingface.co/Yanrui95/NormalCrafter";
pub const WEIGHTS_LICENCE_URL: &str = "https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/blob/main/LICENSE.md";

/// Upstream's requirements.txt pins torch 2.0.1 + xformers 0.0.20 + CUDA
/// 11 wheels, which do not run on a current GPU (an RTX 50-series needs
/// sm_120, torch 2.7+) and have no MPS autocast; this is the environment
/// the driver needs, written into the checkout by `install…`. Windows gets
/// PyTorch's CUDA wheels (PyPI's are CPU-only there).
pub const PYPROJECT: &str = r#"[project]
name = "wobble-normalcrafter"
version = "0.0.0"
requires-python = ">=3.10,<3.13"
dependencies = [
    "torch>=2.8",
    "diffusers==0.33.1",
    "transformers>=4.49,<5",
    "accelerate>=1.0",
    "safetensors",
    "huggingface_hub<1",
    "numpy",
    # The EXR reader and writer (wheels on every OS); OpenCV's EXR codec is
    # only the fallback, and 5.x's Windows wheel has none.
    "OpenEXR>=3.3",
    "opencv-python<5",
    "Pillow",
]

[tool.uv]
package = false

[tool.uv.sources]
torch = [{ index = "pytorch-cu128", marker = "sys_platform == 'win32'" }]

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true
"#;

/// Where it runs and what it costs, for the panel.
pub const WHERE: &str = "runs on an NVIDIA GPU (CUDA), Apple Silicon (MPS) or, very slowly, the CPU. ~20 GB of GPU memory at 1024; it steps the resolution down on its own when it runs out, and offloads to system memory on a smaller card. The clip is solved in overlapping windows, each blended into the last, so the normals hold steady over time";

pub fn install_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("WOBBLE_NORMALS_DIR") {
        return PathBuf::from(d);
    }
    let home = engine::paths::home().unwrap_or_else(|| PathBuf::from("."));
    home.join(".wobble").join("NormalCrafter")
}

/// Installed, AND synced against the environment this build needs: a
/// checkout whose `pyproject.toml` is an older `PYPROJECT` shows as not
/// installed, so `install…` comes back and re-syncs it.
pub fn installed(dir: &Path) -> bool {
    dir.join("normalcrafter").join("normal_crafter_ppl.py").is_file()
        && dir.join(".venv").is_dir()
        && std::fs::read_to_string(dir.join("pyproject.toml")).is_ok_and(|t| t == PYPROJECT)
}

/// A stand-in command replaces the tool (and its weights).
fn stand_in() -> bool {
    std::env::var_os("WOBBLE_NORMALS_CMD").is_some()
}

/// What a result directory's layout is; part of the directory key.
/// 1: `R G B` = the camera-space normal, half float, plus `meta.json`.
pub const LAYOUT: u32 = 1;

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
    rx: Receiver<Msg>,
    pub log: Vec<String>,
    pub result: Option<Result<(), String>>,
}

#[derive(Default)]
pub struct Jobs {
    pub run: Option<Run>,
    pub install: Option<Install>,
}

/// What the panel is told, built by `Gfx::normals_view`.
pub struct View {
    pub dir: String,
    pub installed: bool,
    pub uv: bool,
    pub installing: Option<String>,
    pub install_result: Option<Result<(), String>>,
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

pub fn params(kind: &NodeKind) -> Option<&NormalsParams> {
    match kind {
        NodeKind::Normals(p) => Some(p),
        _ => None,
    }
}

/// The command that runs the driver: `$WOBBLE_NORMALS_CMD` (a stand-in,
/// whitespace-split) or `uv run --project <dir> python <driver>`.
fn command(dir: &Path) -> Result<std::process::Command, String> {
    if let Ok(s) = std::env::var("WOBBLE_NORMALS_CMD") {
        let mut parts = s.split_whitespace();
        let exe = parts.next().ok_or("WOBBLE_NORMALS_CMD is empty")?;
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

impl Gfx {
    /// Where a run of `node` at the current recipe lands. None while the
    /// input cannot be rendered.
    pub(crate) fn normals_dir(&self, node: u64) -> Option<PathBuf> {
        let p = self.egraph.node(node).and_then(|n| params(&n.kind))?;
        let input = self.egraph.input_of(node, 0)?;
        let k0 = self.subtree_key_of(input)?;
        let ((a, b), _) = self.range_of(input);
        let size = self.node_resolution(input, &mut std::collections::HashSet::new());
        let recipe = serde_json::to_vec(&(p.recipe(), a, b, self.handles, size, LAYOUT)).ok()?;
        let key = engine::diskcache::content_hash(&[&k0.to_le_bytes(), &recipe]);
        let root = match &self.disk_cache {
            Some(c) => c.root().join("normals"),
            None => engine::paths::temp_dir().join("wobble-normals"),
        };
        Some(root.join(format!("{key:016x}")))
    }

    pub(crate) fn normals_view(&self, node: u64) -> View {
        let dir = install_dir();
        let mine = self.jobs.normals.run.as_ref().filter(|r| r.node == node);
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
        let want = self.normals_dir(node).map(|d| d.join("Normals"));
        let loaded = self.loaded.get(&node).map(|l| {
            let current = want.as_ref().is_some_and(|d| l.path == d.to_string_lossy());
            (l.seq.frames.len(), current)
        });
        View {
            dir: dir.to_string_lossy().into_owned(),
            installed: installed(&dir) || stand_in(),
            uv: uv().is_some(),
            installing: self.jobs.normals.install.as_ref().filter(|i| i.result.is_none()).map(|i| i.log.last().cloned().unwrap_or_default()),
            install_result: self.jobs.normals.install.as_ref().and_then(|i| i.result.clone()),
            state,
            log: mine.map(|r| r.log.clone()).unwrap_or_default(),
            has_input: self.egraph.input_of(node, 0).is_some(),
            loaded,
            on_disk: want.as_ref().is_some_and(|d| engine::bakes::is_baked(d)),
            busy_elsewhere: self.jobs.normals.run.as_ref().is_some_and(|r| r.node != node && r.active()),
        }
    }

    pub(crate) fn normals_action(&mut self, node: u64, a: Action) {
        match a {
            Action::Run => self.normals_start(node),
            Action::Cancel => {
                if let Some(r) = self.jobs.normals.run.take() {
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
            Action::Install => self.normals_install(),
            Action::Dismiss => {
                if self.jobs.normals.run.as_ref().is_some_and(|r| r.node == node && !r.active()) {
                    self.jobs.normals.run = None;
                }
                if self.jobs.normals.install.as_ref().is_some_and(|i| i.result.is_some()) {
                    self.jobs.normals.install = None;
                }
            }
        }
    }

    fn normals_fail(&mut self, node: u64, e: &str) {
        match self.jobs.normals.run.as_mut().filter(|r| r.node == node) {
            Some(r) => r.phase = Phase::Failed(e.to_string()),
            None => self.jobs.normals.run = Some(Run { node, dir: PathBuf::new(), phase: Phase::Failed(e.into()), log: vec![] }),
        }
    }

    /// `install…`: fetch the pinned checkout and build its environment, on
    /// a thread, its output streamed to the panel.
    fn normals_install(&mut self) {
        if self.jobs.normals.install.as_ref().is_some_and(|i| i.result.is_none()) {
            return;
        }
        let dir = install_dir();
        let (tx, rx) = std::sync::mpsc::channel();
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (&dir, &tx);
            self.jobs.normals.install = Some(Install { rx, log: vec![], result: Some(Err("the tool runs as a process: not in the web build".into())) });
            return;
        }
        #[cfg_attr(target_arch = "wasm32", allow(unreachable_code))]
        let Some(uv) = uv() else {
            self.jobs.normals.install = Some(Install { rx, log: vec![], result: Some(Err(UV_MISSING.into())) });
            return;
        };
        let spawned = std::thread::Builder::new()
            .name("normals-install".into())
            .spawn(move || {
                let result = install_into(&dir, &uv, &tx);
                let _ = tx.send(match result {
                    Ok(()) => Msg::Done,
                    Err(e) => Msg::Failed(e),
                });
            })
            .is_ok();
        self.jobs.normals.install = Some(Install { rx, log: vec![], result: (!spawned).then_some(Err("could not start a thread".into())) });
    }

    /// The `run` click: adopt a result on disk, else render the input.
    fn normals_start(&mut self, node: u64) {
        if self.jobs.normals.run.as_ref().is_some_and(|r| r.active()) {
            return;
        }
        #[cfg(target_arch = "wasm32")]
        return self.normals_fail(node, "Normals runs as a process: not in the web build");
        #[cfg_attr(target_arch = "wasm32", allow(unreachable_code))]
        let Some(input) = self.egraph.input_of(node, 0) else {
            return self.normals_fail(node, "wire a picture into the input");
        };
        let Some(dir) = self.normals_dir(node) else {
            return self.normals_fail(node, "the input is not renderable yet");
        };
        let out = dir.join("Normals");
        if engine::bakes::is_baked(&out) {
            self.load_read(node, &out.to_string_lossy());
            self.playback_dirty = true;
            self.jobs.normals.run = Some(Run { node, dir, phase: Phase::Done("loaded the result on disk".into()), log: vec![] });
            return;
        }
        if !installed(&install_dir()) && !stand_in() {
            return self.normals_fail(node, "NormalCrafter is not installed: press install…");
        }
        self.jobs.normals.run = Some(Run { node, dir: dir.clone(), phase: Phase::Render, log: vec![] });
        let size = self.node_resolution(input, &mut std::collections::HashSet::new());
        if let Err(e) = self.render_input_frames(node, input, dir, size, crate::BakeSink::Normals) {
            self.normals_fail(node, &e);
        }
    }

    /// The bake loop finished rendering the input: run the tool.
    pub(crate) fn normals_render_finished(&mut self, node: u64) {
        let Some(r) = self.jobs.normals.run.as_ref().filter(|r| r.node == node) else { return };
        let dir = r.dir.clone();
        let Some(p) = self.egraph.node(node).and_then(|n| params(&n.kind)).cloned() else {
            return self.normals_fail(node, "the node is gone");
        };
        let mut cmd = match command(&install_dir()) {
            Ok(c) => c,
            Err(e) => return self.normals_fail(node, &e),
        };
        let (window, stride) = p.window_arg();
        cmd.arg("--shot")
            .arg(&dir)
            .arg("--max-res")
            .arg(p.res_arg().to_string())
            .arg("--window")
            .arg(window.to_string())
            .arg("--stride")
            .arg(stride.to_string());
        let (tx, rx) = std::sync::mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        if let Err(e) = spawn(cmd, tx, cancel.clone(), "normals") {
            return self.normals_fail(node, &e);
        }
        if let Some(r) = self.jobs.normals.run.as_mut() {
            r.phase = Phase::Running { rx, cancel, done: 0, total: 0 };
        }
    }

    /// Every redraw: drain the tool's messages, catch a render that died,
    /// adopt a finished result.
    pub(crate) fn normals_poll(&mut self) {
        if let Some(i) = self.jobs.normals.install.as_mut() {
            while let Ok(m) = i.rx.try_recv() {
                match m {
                    Msg::Line(l) => push_log(&mut i.log, l),
                    Msg::Done => i.result = Some(Ok(())),
                    Msg::Failed(e) => i.result = Some(Err(e)),
                    Msg::Frame(..) => {}
                }
            }
        }
        let Some(r) = self.jobs.normals.run.as_mut() else { return };
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
                let out = r.dir.join("Normals");
                if let Err(e) = engine::bakes::mark_baked(&out) {
                    r.phase = Phase::Failed(format!("marking the result done: {e}"));
                    return;
                }
                r.phase = Phase::Done("normals done".into());
                self.load_read(node, &out.to_string_lossy());
                self.playback_dirty = true;
            }
            Some(Err(e)) => r.phase = Phase::Failed(e),
            None => {}
        }
    }

    /// After a load, an undo, a paste: every Normals node whose result for
    /// the current recipe is on disk plays it.
    pub(crate) fn normals_adopt(&mut self) {
        let ids: Vec<u64> = self
            .egraph
            .nodes
            .iter()
            .filter(|(id, n)| {
                matches!(n.kind, NodeKind::Normals(_)) && !self.loaded.contains_key(id) && !self.load_errors.contains_key(id)
            })
            .map(|(id, _)| *id)
            .collect();
        for id in ids {
            let Some(dir) = self.normals_dir(id) else { continue };
            let out = dir.join("Normals");
            if engine::bakes::is_baked(&out) {
                self.load_read(id, &out.to_string_lossy());
                self.playback_dirty = true;
            }
        }
    }
}

/// Fetch the pinned commit into `dir` (fresh, or an earlier install at
/// another commit), write our project file and `uv sync` it.
fn install_into(dir: &Path, uv: &Path, tx: &Sender<Msg>) -> Result<(), String> {
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
    let _ = tx.send(Msg::Line(format!("git fetch {REPO} {}", &COMMIT[..12])));
    run(git(&["fetch", "-q", "--depth", "1", REPO, COMMIT])).map_err(|e| format!("git fetch: {e}"))?;
    run(git(&["checkout", "-q", "--force", "FETCH_HEAD"])).map_err(|e| format!("git checkout: {e}"))?;
    std::fs::write(dir.join("pyproject.toml"), PYPROJECT).map_err(|e| format!("writing pyproject.toml: {e}"))?;
    let _ = tx.send(Msg::Line("uv sync (torch and diffusers: several GB, a while)".into()));
    let mut c = std::process::Command::new(uv);
    c.arg("sync").arg("--python").arg("3.12").current_dir(dir);
    run(c).map_err(|e| format!("uv sync: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_driver_speaks_the_protocol_the_reader_parses() {
        for w in ["WOBBLE frame", "WOBBLE done", "WOBBLE error", "--shot", "--max-res", "--window", "--stride", "\"Normals\""] {
            assert!(DRIVER.contains(w), "{w}");
        }
        assert!(DRIVER.contains(&format!("LAYOUT = {LAYOUT}")));
    }

    /// The weights are the tool's, fetched on the first run: at a PINNED
    /// revision, so a push to the model repo never changes an old comp.
    #[test]
    fn the_weights_are_pinned_to_a_revision() {
        let rev = DRIVER.lines().find_map(|l| l.strip_prefix("REVISION = \"")).expect("a REVISION line");
        let rev = rev.trim_end_matches('"');
        assert_eq!(rev.len(), 40, "{rev}");
        assert!(rev.chars().all(|c| c.is_ascii_hexdigit()));
        assert!(DRIVER.contains("\"revision\": REVISION"));
    }

    #[test]
    fn the_install_is_pinned() {
        assert_eq!(COMMIT.len(), 40);
        assert!(COMMIT.chars().all(|c| c.is_ascii_hexdigit()));
        assert!(PYPROJECT.contains("torch>=2.8"), "a 50-series card needs sm_120 wheels");
        assert!(PYPROJECT.contains("\"opencv-python<5\"") && PYPROJECT.contains("OpenEXR"));
        assert!(!PYPROJECT.contains("xformers"), "no Mac wheels, and SDPA does the job");
    }

    /// The driver's EXR reader on a frame from OUR writer, a write/read
    /// round trip of its output, and the window plan. Needs the tool
    /// installed (`install…`); skips, saying so, without it.
    #[test]
    fn the_driver_reads_our_exrs_and_plans_windows_like_the_tool() {
        let dir = install_dir();
        let Some(uv) = uv().filter(|_| installed(&dir)) else {
            eprintln!("skipping: NormalCrafter is not installed at {}", dir.display());
            return;
        };
        let tmp = std::env::temp_dir().join(format!("wobble-normals-io-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        let (w, h) = (6usize, 4usize);
        let px: Vec<half::f16> = (0..w * h).flat_map(|i| [0.25, 0.5, i as f32, 1.0].map(half::f16::from_f32)).collect();
        engine::bakes::write_rgba_f16_exr_tagged(&tmp.join("good.exr"), w, h, w * 4, &px, None).unwrap();
        std::fs::write(tmp.join(DRIVER_NAME), DRIVER).unwrap();
        let script = r#"
import importlib.util, os, sys
d = sys.argv[1]
spec = importlib.util.spec_from_file_location("drv", os.path.join(d, "wobble_normals.py"))
drv = importlib.util.module_from_spec(spec); spec.loader.exec_module(drv)
import numpy as np
m = drv.read_rgb(os.path.join(d, "good.exr"))
assert m.shape == (4, 6, 3), m.shape
assert abs(m[0, 0, 0] - 0.25) < 1e-3 and m[3, 5, 2] == 23.0, m[0, 0]
n = np.zeros((4, 6, 3), np.float32); n[..., 2] = 1.0; n[0, 0] = (0.6, -0.8, 0.0)
drv.write_rgb(os.path.join(d, "out.exr"), n)
o = drv.read_rgb(os.path.join(d, "out.exr"))
assert np.allclose(o, n, atol=1e-3), o[0, 0]
assert not os.path.exists(os.path.join(d, "out.exr.part"))
assert drv.windows(30, 14, 10) == [[0, 14], [10, 24], [16, 30]], drv.windows(30, 14, 10)
assert drv.windows(14, 14, 10) == [[0, 14]]
assert drv.windows(24, 14, 10) == [[0, 14], [10, 24]]
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
}
