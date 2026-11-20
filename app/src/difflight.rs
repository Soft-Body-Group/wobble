//! The DiffusionLight node's app half: where DiffusionLight-Turbo lives,
//! installing it, fetching its weights, rendering the node's input to a
//! shot directory, running the estimate as a process, and adopting what it
//! wrote. The node itself (params, codegen) is `engine::nodes::difflight`;
//! the panel and glyph `crate::nodes::difflight`. The process plumbing is
//! CorridorKey's (`crate::corridorkey::spawn`), the input render the
//! Upscale node's (`Gfx::render_input_frames`), the shape of all of it the
//! LTX HDR node's.
//!
//! A run is two phases, each explicit and cancellable:
//!
//! 1. **render** — a `SubtreeBake` of input 0 over the input's frame range,
//!    at the input's own format, written as linear Rec.709 EXRs to
//!    `<dir>/Input`;
//! 2. **run** — the tool, one process, driven by `tools/wobble_difflight.py`
//!    (embedded, written next to the checkout on every run): an estimate
//!    per frame into `<dir>/Raw`, then the smoothing pass into
//!    `<dir>/<out_name>`, which is adopted as the node's reader.
//!
//! `<dir>` is keyed on the input's subtree key and format, the ESTIMATE's
//! recipe (exposure, seed, steps, size) and the range — an edit upstream
//! shows as STALE and a re-run lands in a new directory. The smoothing
//! (stride, sigma, follow) is only the NAME of the directory under it: a
//! change there is STALE too, but its re-run finds the estimates in `Raw`
//! and costs seconds. A cancelled run resumes the same way.
//!
//! **Nothing installs or downloads without a click.** `install…` fetches
//! the pinned checkout into `~/.wobble/DiffusionLight-Turbo` (or
//! `$WOBBLE_DIFFLIGHT_DIR`), writes `PYPROJECT` into it and runs `uv sync`.
//! `fetch weights…` runs the driver's `--fetch`: five public repositories
//! at pinned revisions into `<checkout>/models`. A run is offline.
//!
//! **Two models, one runner.** Everything that differs between
//! DiffusionLight-Turbo and LuxDiT (repository, commit, environment,
//! driver, weights, wording) is a [`Tool`]; `tool(model)` picks it, and the
//! install and fetch jobs are kept per model. LuxDiT runs only on CUDA:
//! `nvidia()` (the adapter's PCI vendor) decides whether the app offers it
//! at all — the menu entry and the model picker leave it out elsewhere,
//! and a scene that already has one still opens and shows it.

use crate::corridorkey::{push_log, spawn, uv, Msg, UV_MISSING};
use crate::Gfx;
use engine::graph::{DiffLightModel, DiffLightParams, NodeKind};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;

pub const DRIVER: &str = include_str!("../../tools/wobble_difflight.py");
const DRIVER_NAME: &str = "wobble_difflight.py";
pub const LUX_DRIVER: &str = include_str!("../../tools/wobble_luxdit.py");
pub const REPO: &str = "https://github.com/DiffusionLight/DiffusionLight-Turbo.git";
/// The commit the driver is written against: it imports the projection
/// from the checkout's `ball2envmap.py`. Bump it and re-test, never float it.
pub const COMMIT: &str = "6e8b0f1ad664f5e11281d80482ae19a490927cc4";
pub const LICENCE_URL: &str = "https://github.com/DiffusionLight/DiffusionLight-Turbo/blob/main/LICENSE";
/// SDXL's and its ControlNet's licence (CreativeML Open RAIL++-M).
pub const SDXL_LICENCE_URL: &str = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md";
/// What `--fetch` downloads, each at a pinned revision: the driver's
/// `PINS`, in its order. The driver writes them into `models/READY` when
/// every file is there, and that file is how the app knows.
pub const PINS: [(&str, &str); 5] = [
    ("stabilityai/stable-diffusion-xl-base-1.0", "462165984030d82259a11f4367a4eed129e94a7b"),
    ("diffusers/controlnet-depth-sdxl-1.0", "17bb97973f29801224cd66f192c5ffacf82648b4"),
    ("DiffusionLight/Flickr2K", "ed635477e5a742e363e54a6ed2193a9a43153113"),
    ("DiffusionLight/DiffusionLight", "bda34823e932601b3eb9aff0276f00925cdeea46"),
    ("Intel/dpt-large", "bc15f29aa3a80d532f2ed650b5e16ac48d8958f9"),
];
/// The download's size, for the button (fp16 safetensors only).
pub const WEIGHTS_BYTES: u64 = 10_950_000_000;

/// The environment the driver runs in. Upstream pins torch 2.0 + CUDA 11.8
/// and diffusers 0.23, which no current GPU driver runs (an RTX 50 needs
/// CUDA 12.8); the inpainting is the authors' own port to diffusers 0.33,
/// so that is what is pinned. Written into the checkout by `install…`;
/// `installed()` is false while the checkout's differs, so an update
/// brings `install…` back.
pub const PYPROJECT: &str = r#"[project]
name = "wobble-diffusionlight-turbo"
version = "0.0.0"
requires-python = ">=3.10,<3.13"
dependencies = [
    "torch>=2.8",
    "diffusers==0.33.1",
    # The releases of the port's day: a later peft no longer loads these LoRAs
    # through diffusers 0.33.
    "transformers==4.51.3",
    "peft==0.15.2",
    "accelerate>=1.0",
    "safetensors",
    "huggingface_hub<1",
    "numpy",
    "Pillow",
    "OpenEXR>=3.3",
    # `ball2envmap` imports them at the top.
    "scikit-image",
    "tqdm",
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
pub const WHERE: &str = "CUDA (about 11 GB of VRAM, about 35 s a frame on an RTX 5080: three SDXL inpaints of 30 steps each), or Apple Silicon (untested). Every frame is its own estimate; the smoothing pass is what steadies them";

/// LuxDiT (NVIDIA): the clip in windows through a video diffusion
/// transformer. The commit `tools/wobble_luxdit.py` imports the pipeline,
/// the VAE, the transformer and the HDR merger from.
pub const LUX_REPO: &str = "https://github.com/nv-tlabs/LuxDiT.git";
pub const LUX_COMMIT: &str = "a191884afbd757bab546bb01f2eaa967fb80b375";
pub const LUX_LICENCE_URL: &str = "https://github.com/nv-tlabs/LuxDiT/blob/main/LICENSE.md";
/// The driver's `PINS`, in its order: only CogVideoX's VAE and scheduler
/// are fetched from the base model, never its text encoder.
pub const LUX_PINS: [(&str, &str); 2] = [
    ("zai-org/CogVideoX-5b-I2V", "a6f0f4858a8395e7429d82493864ce92bf73af11"),
    ("nvidia/LuxDiT", "6c10bcbbe11439660d4bad514250e6d30a32999d"),
];
pub const LUX_WEIGHTS_BYTES: u64 = 23_500_000_000;
/// Upstream pins diffusers 0.32 / transformers 4.47 / peft 0.7 and leaves
/// torch to the user ("tested with 2.4"); a current torch is what an RTX 50
/// needs, and the rest is upstream's own pins (measured on a 5080).
pub const LUX_PYPROJECT: &str = r#"[project]
name = "wobble-luxdit"
version = "0.0.0"
requires-python = ">=3.10,<3.13"
dependencies = [
    "torch>=2.8",
    "torchvision",
    "diffusers==0.32.0",
    "transformers==4.47.1",
    "accelerate==1.3.0",
    "peft==0.7.0",
    "omegaconf",
    "safetensors",
    "huggingface_hub<1",
    "numpy<2.3",
    "Pillow",
    "OpenEXR>=3.3",
    "opencv-python-headless",
    "imageio",
    "imageio-ffmpeg",
    "einops",
    "braceexpand",
    "matplotlib",
    "tqdm",
    "sentencepiece",
]

[tool.uv]
package = false
"#;
pub const LUX_WHERE: &str = "an NVIDIA GPU only (CUDA, Linux tested). On an RTX 5080: 12 GB of VRAM and about 40 s for a 5-frame window at 40 steps. The clip is seen in windows of 'window' frames, each a video to the model, cross-faded into the last";

/// Everything that differs between the two models' tools.
pub struct Tool {
    pub name: &'static str,
    /// `~/.wobble/<dir_name>`, or `$<env_dir>`.
    dir_name: &'static str,
    env_dir: &'static str,
    pub repo: &'static str,
    pub commit: &'static str,
    pub pins: &'static [(&'static str, &'static str)],
    pub weights_bytes: u64,
    pub pyproject: &'static str,
    /// The driver the app writes into the checkout, and its file name.
    driver: &'static str,
    driver_name: &'static str,
    /// A file only an install of this checkout has.
    marker: &'static str,
    pub where_: &'static str,
}

pub const TURBO: Tool = Tool {
    name: "DiffusionLight-Turbo",
    dir_name: "DiffusionLight-Turbo",
    env_dir: "WOBBLE_DIFFLIGHT_DIR",
    repo: REPO,
    commit: COMMIT,
    pins: &PINS,
    weights_bytes: WEIGHTS_BYTES,
    pyproject: PYPROJECT,
    driver: DRIVER,
    driver_name: DRIVER_NAME,
    marker: "ball2envmap.py",
    where_: WHERE,
};

pub const LUXDIT: Tool = Tool {
    name: "LuxDiT",
    dir_name: "LuxDiT",
    env_dir: "WOBBLE_LUXDIT_DIR",
    repo: LUX_REPO,
    commit: LUX_COMMIT,
    pins: &LUX_PINS,
    weights_bytes: LUX_WEIGHTS_BYTES,
    pyproject: LUX_PYPROJECT,
    driver: LUX_DRIVER,
    driver_name: "wobble_luxdit.py",
    marker: "inference_luxdit.py",
    where_: LUX_WHERE,
};

pub fn tool(m: DiffLightModel) -> &'static Tool {
    match m {
        DiffLightModel::Turbo => &TURBO,
        DiffLightModel::LuxDit => &LUXDIT,
    }
}

static NVIDIA: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// Called once with the adapter the app renders on.
pub fn set_adapter(info: &wgpu::AdapterInfo) {
    let _ = NVIDIA.set(info.vendor == 0x10de);
}

/// Is LuxDiT offered here? Only on an NVIDIA GPU (`$WOBBLE_LUXDIT=1`
/// forces it, for a suite or a remote CUDA box).
pub fn nvidia() -> bool {
    std::env::var_os("WOBBLE_LUXDIT").is_some_and(|v| v == "1") || NVIDIA.get().copied().unwrap_or(false)
}

impl Tool {
    pub fn install_dir(&self) -> PathBuf {
        if let Some(d) = std::env::var_os(self.env_dir) {
            return PathBuf::from(d);
        }
        let home = engine::paths::home().unwrap_or_else(|| PathBuf::from("."));
        home.join(".wobble").join(self.dir_name)
    }

    /// Where the weights land: beside the checkout, never ~/.cache and
    /// never the catalog's.
    pub fn models_dir(&self) -> PathBuf {
        self.install_dir().join("models")
    }

    /// Installed, AND synced against the environment this build needs.
    pub fn installed(&self, dir: &Path) -> bool {
        dir.join(self.marker).is_file()
            && dir.join(".venv").is_dir()
            && std::fs::read_to_string(dir.join("pyproject.toml")).is_ok_and(|t| t == self.pyproject)
    }

    /// What the driver's `--fetch` writes once everything is there.
    pub fn ready_text(&self) -> String {
        self.pins.iter().map(|(repo, rev)| format!("{repo} {rev}\n")).collect()
    }

    /// Fetched, at the revisions this build pins.
    pub fn weights_ready(&self, models: &Path) -> bool {
        std::fs::read_to_string(models.join("READY")).is_ok_and(|t| t == self.ready_text())
    }
}

/// A suite's stand-in command replaces the tool (and its weights).
fn stand_in() -> bool {
    std::env::var_os("WOBBLE_DIFFLIGHT_CMD").is_some()
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
    /// The smoothing pass's directory under `dir`, as of the click.
    pub out: PathBuf,
    pub phase: Phase,
    pub log: Vec<String>,
}

impl Run {
    pub fn active(&self) -> bool {
        matches!(self.phase, Phase::Render | Phase::Running { .. })
    }
}

/// An install or a weights fetch: a background job with a log.
pub struct Job {
    rx: Receiver<Msg>,
    pub log: Vec<String>,
    pub progress: (usize, usize),
    pub result: Option<Result<(), String>>,
}

/// One run at a time (one GPU); an install and a fetch per model.
#[derive(Default)]
pub struct Jobs {
    pub run: Option<Run>,
    pub install: [Option<Job>; 2],
    pub fetch: [Option<Job>; 2],
}

fn slot(m: DiffLightModel) -> usize {
    match m {
        DiffLightModel::Turbo => 0,
        DiffLightModel::LuxDit => 1,
    }
}

impl Jobs {
    pub fn any(&self) -> bool {
        self.run.is_some() || self.install.iter().chain(&self.fetch).any(Option::is_some)
    }
}

/// What the panel is told, built by `Gfx::difflight_view`.
pub struct View {
    /// LuxDiT may be picked here (an NVIDIA GPU).
    pub nvidia: bool,
    pub dir: String,
    pub installed: bool,
    pub uv: bool,
    pub installing: Option<String>,
    pub install_result: Option<Result<(), String>>,
    pub weights: bool,
    pub fetching: Option<(String, usize, usize)>,
    pub fetch_result: Option<Result<(), String>>,
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
    Fetch,
    Dismiss,
}

pub fn params(kind: &NodeKind) -> Option<&DiffLightParams> {
    match kind {
        NodeKind::DiffLight(p) => Some(p),
        _ => None,
    }
}

/// The command that runs the driver: `$WOBBLE_DIFFLIGHT_CMD` (a suite's
/// stand-in, whitespace-split) or `uv run --project <dir> python <driver>`.
fn command(t: &Tool, dir: &Path) -> Result<std::process::Command, String> {
    if let Ok(s) = std::env::var("WOBBLE_DIFFLIGHT_CMD") {
        let mut parts = s.split_whitespace();
        let exe = parts.next().ok_or("WOBBLE_DIFFLIGHT_CMD is empty")?;
        let mut c = std::process::Command::new(exe);
        c.args(parts);
        return Ok(c);
    }
    let uv = uv().ok_or(UV_MISSING)?;
    let driver = dir.join(t.driver_name);
    std::fs::write(&driver, t.driver).map_err(|e| format!("writing the driver: {e}"))?;
    // LuxDiT's driver imports the smoothing pass and the EXR helpers from
    // DiffusionLight's: one spec for both.
    if t.driver_name != DRIVER_NAME {
        std::fs::write(dir.join(DRIVER_NAME), DRIVER).map_err(|e| format!("writing the driver: {e}"))?;
    }
    let mut c = std::process::Command::new(uv);
    c.arg("run").arg("--project").arg(dir).arg("python").arg(&driver);
    c.current_dir(dir);
    Ok(c)
}

fn drain(job: &mut Job) {
    while let Ok(m) = job.rx.try_recv() {
        match m {
            Msg::Line(l) => push_log(&mut job.log, l),
            Msg::Done => job.result = Some(Ok(())),
            Msg::Failed(e) => job.result = Some(Err(e)),
            Msg::Frame(a, b) => job.progress = (a, b),
        }
    }
}

impl Gfx {
    /// Where a run of `node` at the current recipe lands. None while the
    /// input cannot be rendered.
    pub(crate) fn difflight_dir(&self, node: u64) -> Option<PathBuf> {
        let p = self.egraph.node(node).and_then(|n| params(&n.kind))?;
        let input = self.egraph.input_of(node, 0)?;
        let ((a, b), _) = self.range_of(input);
        let size = self.node_resolution(input, &mut std::collections::HashSet::new());
        // Keyed at the INPUT's format, not the canvas: viewing the loaded
        // node makes the canvas the latlong's, and a key that followed it
        // would call its own result stale the moment it was shown.
        let k0 = self.subtree_key_at(input, size.or_else(|| self.comp.output_size())?)?;
        let recipe = serde_json::to_vec(&(p.recipe(), a, b, self.handles, size)).ok()?;
        let key = engine::diskcache::content_hash(&[&k0.to_le_bytes(), &recipe]);
        let root = match &self.disk_cache {
            Some(c) => c.root().join("difflight"),
            None => engine::paths::temp_dir().join("wobble-difflight"),
        };
        Some(root.join(format!("{key:016x}")))
    }

    /// Where the latlongs for the current smoothing settings land: a
    /// directory of their own under the estimates'.
    pub(crate) fn difflight_out(&self, node: u64) -> Option<PathBuf> {
        let p = self.egraph.node(node).and_then(|n| params(&n.kind))?;
        Some(self.difflight_dir(node)?.join(p.out_name()))
    }

    pub(crate) fn difflight_view(&self, node: u64) -> View {
        let model = self.egraph.node(node).and_then(|n| params(&n.kind)).map(|p| p.model).unwrap_or_default();
        let t = tool(model);
        let k = slot(model);
        let dir = t.install_dir();
        let mine = self.jobs.difflight.run.as_ref().filter(|r| r.node == node);
        let state = match mine.map(|r| &r.phase) {
            None => ViewState::Idle,
            Some(Phase::Render) => match &self.graph_bake {
                Some(b) if b.node == node => {
                    ViewState::Rendering(b.next.saturating_sub(b.from), b.len.saturating_sub(b.from))
                }
                _ => ViewState::Rendering(0, 0),
            },
            Some(Phase::Running { done, total, .. }) => ViewState::Running(*done, *total),
            Some(Phase::Failed(e)) => ViewState::Failed(e.clone()),
            Some(Phase::Done(s)) => ViewState::Done(s.clone()),
        };
        let want = self.difflight_out(node);
        let loaded = self.loaded.get(&node).map(|l| {
            let current = want.as_ref().is_some_and(|d| l.path == d.to_string_lossy());
            (l.seq.frames.len(), current)
        });
        let fetch = self.jobs.difflight.fetch[k].as_ref();
        View {
            nvidia: nvidia(),
            dir: dir.to_string_lossy().into_owned(),
            installed: t.installed(&dir) || stand_in(),
            uv: uv().is_some(),
            installing: self.jobs.difflight.install[k]
                .as_ref()
                .filter(|i| i.result.is_none())
                .map(|i| i.log.last().cloned().unwrap_or_default()),
            install_result: self.jobs.difflight.install[k].as_ref().and_then(|i| i.result.clone()),
            weights: t.weights_ready(&t.models_dir()) || stand_in(),
            fetching: fetch
                .filter(|f| f.result.is_none())
                .map(|f| (f.log.last().cloned().unwrap_or_default(), f.progress.0, f.progress.1)),
            fetch_result: fetch.and_then(|f| f.result.clone()),
            state,
            log: mine.map(|r| r.log.clone()).unwrap_or_default(),
            has_input: self.egraph.input_of(node, 0).is_some(),
            loaded,
            on_disk: want.as_ref().is_some_and(|d| engine::bakes::is_baked(d)),
            busy_elsewhere: self.jobs.difflight.run.as_ref().is_some_and(|r| r.node != node && r.active()),
        }
    }

    pub(crate) fn difflight_action(&mut self, node: u64, a: Action) {
        match a {
            Action::Run => self.difflight_start(node),
            Action::Cancel => {
                if let Some(r) = self.jobs.difflight.run.take() {
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
            Action::Install => self.difflight_install(self.difflight_model(node)),
            Action::Fetch => self.difflight_fetch(self.difflight_model(node)),
            Action::Dismiss => {
                if self.jobs.difflight.run.as_ref().is_some_and(|r| r.node == node && !r.active()) {
                    self.jobs.difflight.run = None;
                }
                for j in self.jobs.difflight.install.iter_mut().chain(self.jobs.difflight.fetch.iter_mut()) {
                    if j.as_ref().is_some_and(|i| i.result.is_some()) {
                        *j = None;
                    }
                }
            }
        }
    }

    fn difflight_model(&self, node: u64) -> DiffLightModel {
        self.egraph.node(node).and_then(|n| params(&n.kind)).map(|p| p.model).unwrap_or_default()
    }

    fn difflight_fail(&mut self, node: u64, e: &str) {
        match self.jobs.difflight.run.as_mut().filter(|r| r.node == node) {
            Some(r) => r.phase = Phase::Failed(e.to_string()),
            None => {
                self.jobs.difflight.run = Some(Run { node, dir: PathBuf::new(), out: PathBuf::new(), phase: Phase::Failed(e.into()), log: vec![] })
            }
        }
    }

    /// `install…`: fetch the pinned checkout and build its environment, on
    /// a thread, its output streamed to the panel.
    fn difflight_install(&mut self, model: DiffLightModel) {
        let (t, k) = (tool(model), slot(model));
        if self.jobs.difflight.install[k].as_ref().is_some_and(|i| i.result.is_none()) {
            return;
        }
        let dir = t.install_dir();
        let (tx, rx) = std::sync::mpsc::channel();
        let job = |rx, result| Job { rx, log: vec![], progress: (0, 0), result };
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (&dir, &tx);
            self.jobs.difflight.install[k] = Some(job(rx, Some(Err("the tool runs as a process: not in the web build".into()))));
            return;
        }
        #[cfg_attr(target_arch = "wasm32", allow(unreachable_code))]
        let Some(uv) = uv() else {
            self.jobs.difflight.install[k] = Some(job(rx, Some(Err(UV_MISSING.into()))));
            return;
        };
        let spawned = std::thread::Builder::new()
            .name("difflight-install".into())
            .spawn(move || {
                let result = install_into(t, &dir, &uv, &tx);
                let _ = tx.send(match result {
                    Ok(()) => Msg::Done,
                    Err(e) => Msg::Failed(e),
                });
            })
            .is_ok();
        self.jobs.difflight.install[k] = Some(job(rx, (!spawned).then_some(Err("could not start a thread".into()))));
    }

    /// `fetch weights…`: the driver's `--fetch`, as a process.
    fn difflight_fetch(&mut self, model: DiffLightModel) {
        let (t, k) = (tool(model), slot(model));
        if self.jobs.difflight.fetch[k].as_ref().is_some_and(|f| f.result.is_none()) {
            return;
        }
        let (tx, rx) = std::sync::mpsc::channel();
        let mut job = Job { rx, log: vec![], progress: (0, 0), result: None };
        if !t.installed(&t.install_dir()) {
            job.result = Some(Err(format!("install {} first: the fetch runs in its environment", t.name)));
            self.jobs.difflight.fetch[k] = Some(job);
            return;
        }
        let mut cmd = match command(t, &t.install_dir()) {
            Ok(c) => c,
            Err(e) => {
                job.result = Some(Err(e));
                self.jobs.difflight.fetch[k] = Some(job);
                return;
            }
        };
        cmd.arg("--fetch").arg(t.models_dir());
        if let Err(e) = spawn(cmd, tx, Arc::new(AtomicBool::new(false)), "difflight-fetch") {
            job.result = Some(Err(e));
        }
        self.jobs.difflight.fetch[k] = Some(job);
    }

    /// The `run` click: adopt a result on disk, else render the input.
    fn difflight_start(&mut self, node: u64) {
        if self.jobs.difflight.run.as_ref().is_some_and(|r| r.active()) {
            return;
        }
        #[cfg(target_arch = "wasm32")]
        return self.difflight_fail(node, "DiffusionLight runs as a process: not in the web build");
        #[cfg_attr(target_arch = "wasm32", allow(unreachable_code))]
        let Some(input) = self.egraph.input_of(node, 0) else {
            return self.difflight_fail(node, "wire a picture into the input");
        };
        let Some(dir) = self.difflight_dir(node) else {
            return self.difflight_fail(node, "the input is not renderable yet");
        };
        let Some(out) = self.difflight_out(node) else {
            return self.difflight_fail(node, "the node is gone");
        };
        if engine::bakes::is_baked(&out) {
            self.load_read(node, &out.to_string_lossy());
            self.playback_dirty = true;
            self.jobs.difflight.run = Some(Run { node, dir, out, phase: Phase::Done("loaded the result on disk".into()), log: vec![] });
            return;
        }
        if !stand_in() {
            let t = tool(self.difflight_model(node));
            if !t.installed(&t.install_dir()) {
                return self.difflight_fail(node, &format!("{} is not installed: press install…", t.name));
            }
            if !t.weights_ready(&t.models_dir()) {
                return self.difflight_fail(node, "the weights are not fetched: press fetch weights…");
            }
        }
        self.jobs.difflight.run = Some(Run { node, dir: dir.clone(), out, phase: Phase::Render, log: vec![] });
        let size = self.node_resolution(input, &mut std::collections::HashSet::new());
        if let Err(e) = self.render_input_frames(node, input, dir, size, crate::BakeSink::DiffLight) {
            self.difflight_fail(node, &e);
        }
    }

    /// The bake loop finished rendering the input: run the tool.
    pub(crate) fn difflight_render_finished(&mut self, node: u64) {
        let Some(r) = self.jobs.difflight.run.as_ref().filter(|r| r.node == node) else { return };
        let dir = r.dir.clone();
        let Some(p) = self.egraph.node(node).and_then(|n| params(&n.kind)).cloned() else {
            return self.difflight_fail(node, "the node is gone");
        };
        let t = tool(p.model);
        let mut cmd = match command(t, &t.install_dir()) {
            Ok(c) => c,
            Err(e) => return self.difflight_fail(node, &e),
        };
        // The smoothing settings as they are NOW (the render may have
        // taken a while): the name and the arguments from one read.
        let out = dir.join(p.out_name());
        cmd.arg("--shot")
            .arg(&dir)
            .arg("--models")
            .arg(t.models_dir())
            .arg("--out")
            .arg(p.out_name())
            .arg("--seed")
            .arg(p.seed.to_string())
            .arg("--steps")
            .arg(p.steps_arg().to_string())
            .arg("--exposure")
            .arg(format!("{}", p.exposure))
            .arg("--height")
            .arg(p.size.rows().to_string())
            .arg("--stride")
            .arg(p.stride_arg().to_string())
            .arg("--smooth")
            .arg(format!("{}", p.smooth_arg()));
        if !p.follow {
            cmd.arg("--no-follow");
        }
        if p.model == DiffLightModel::LuxDit {
            cmd.arg("--window").arg(p.window_arg().to_string());
        }
        let (tx, rx) = std::sync::mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        if let Err(e) = spawn(cmd, tx, cancel.clone(), "difflight") {
            return self.difflight_fail(node, &e);
        }
        if let Some(r) = self.jobs.difflight.run.as_mut() {
            r.out = out;
            r.phase = Phase::Running { rx, cancel, done: 0, total: 0 };
        }
    }

    /// Every redraw: drain the jobs' messages, catch a render that died,
    /// adopt a finished result.
    pub(crate) fn difflight_poll(&mut self) {
        for j in self.jobs.difflight.install.iter_mut().chain(self.jobs.difflight.fetch.iter_mut()).flatten() {
            drain(j);
        }
        let Some(r) = self.jobs.difflight.run.as_mut() else { return };
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
                let out = r.out.clone();
                if let Err(e) = engine::bakes::mark_baked(&out) {
                    r.phase = Phase::Failed(format!("marking the result done: {e}"));
                    return;
                }
                r.phase = Phase::Done("latlongs done".into());
                self.load_read(node, &out.to_string_lossy());
                self.playback_dirty = true;
            }
            Some(Err(e)) => r.phase = Phase::Failed(e),
            None => {}
        }
    }

    /// After a load, an undo, a paste: every DiffusionLight node whose result for
    /// the current recipe is on disk plays it.
    pub(crate) fn difflight_adopt(&mut self) {
        let ids: Vec<u64> = self
            .egraph
            .nodes
            .iter()
            .filter(|(id, n)| {
                matches!(n.kind, NodeKind::DiffLight(_)) && !self.loaded.contains_key(id) && !self.load_errors.contains_key(id)
            })
            .map(|(id, _)| *id)
            .collect();
        for id in ids {
            let Some(out) = self.difflight_out(id) else { continue };
            if engine::bakes::is_baked(&out) {
                self.load_read(id, &out.to_string_lossy());
                self.playback_dirty = true;
            }
        }
    }
}

/// Fetch the pinned commit into `dir` (fresh, or an earlier install at
/// another commit), write our pyproject into it and `uv sync`.
fn install_into(t: &Tool, dir: &Path, uv: &Path, tx: &Sender<Msg>) -> Result<(), String> {
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
    let _ = tx.send(Msg::Line(format!("git fetch {} {}", t.repo, &t.commit[..12])));
    run(git(&["fetch", "-q", "--depth", "1", t.repo, t.commit])).map_err(|e| format!("git fetch: {e}"))?;
    run(git(&["checkout", "-q", "--force", "FETCH_HEAD"])).map_err(|e| format!("git checkout: {e}"))?;
    std::fs::write(dir.join("pyproject.toml"), t.pyproject).map_err(|e| format!("writing pyproject.toml: {e}"))?;
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
        for w in [
            "WOBBLE\"",
            "\"frame\"",
            "\"done\"",
            "\"error\"",
            "--shot",
            "--models",
            "--out",
            "--seed",
            "--steps",
            "--exposure",
            "--height",
            "--stride",
            "--smooth",
            "--no-follow",
            "--fetch",
            "\"Raw\"",
            "\"Input\"",
            "from ball2envmap import",
            "READY = \"READY\"",
        ] {
            assert!(DRIVER.contains(w), "{w}");
        }
    }

    #[test]
    fn the_luxdit_driver_speaks_the_same_protocol() {
        for w in [
            "WOBBLE\"", "\"frame\"", "\"done\"", "\"error\"", "--shot", "--models", "--out", "--seed", "--steps",
            "--exposure", "--height", "--window", "--stride", "--smooth", "--no-follow", "--fetch", "\"Raw\"",
            "\"Input\"", "from wobble_difflight import", "READY = \"READY\"",
        ] {
            assert!(LUX_DRIVER.contains(w), "{w}");
        }
        // What it imports from DiffusionLight's driver is there.
        for f in ["def read_rgb(", "def write_rgb(", "def srgb_encode(", "def sampled(", "def smooth("] {
            assert!(DRIVER.contains(f), "{f}");
        }
    }

    #[test]
    fn the_drivers_and_the_app_agree_on_the_pins() {
        for (t, driver) in [(&TURBO, DRIVER), (&LUXDIT, LUX_DRIVER)] {
            assert_eq!(t.commit.len(), 40);
            assert!(t.commit.chars().all(|c| c.is_ascii_hexdigit()));
            // The same repositories at the same revisions, in the same
            // order: `ready_text()` here must be the driver's, byte for byte.
            let mut last = 0;
            for (repo, rev) in t.pins {
                assert_eq!(rev.len(), 40);
                assert!(driver.contains(&format!("\"{repo}\"")), "{repo}");
                let at = driver.find(&format!("\"{rev}\"")).unwrap_or_else(|| panic!("{repo} @ {rev}"));
                assert!(at > last, "{repo} is out of order");
                last = at;
            }
            assert!(driver.contains("f\"{repo} {rev}\\n\""));
            assert_eq!(t.ready_text().lines().count(), t.pins.len());
        }
        assert_eq!(tool(DiffLightModel::LuxDit).name, "LuxDiT");
        assert_ne!(TURBO.install_dir(), LUXDIT.install_dir());
    }

    #[test]
    fn only_the_fetch_mode_touches_the_network() {
        // Everything that downloads sits inside `def fetch`, and a run
        // forces the hub offline.
        for code in [DRIVER, LUX_DRIVER] {
            let fetch_at = code.find("def fetch(").unwrap();
            let next_def = code[fetch_at + 1..].find("\ndef ").map(|i| i + fetch_at + 1).unwrap();
            let mut at = 0;
            for (i, l) in code.lines().enumerate() {
                let inside = (fetch_at..next_def).contains(&at);
                for w in ["hf_hub_download", "snapshot_download", "urllib", "requests"] {
                    if l.contains(w) && !l.trim_start().starts_with('#') {
                        assert!(inside, "line {}: {w} outside the fetch mode", i + 1);
                    }
                }
                at += l.len() + 1;
            }
            assert!(code.contains("os.environ[\"HF_HUB_OFFLINE\"] = \"1\""));
        }
    }

    #[test]
    fn an_install_is_ours_only_with_our_environment() {
        for t in [&TURBO, &LUXDIT] {
            let d = std::env::temp_dir().join(format!("wobble-difflight-installed-{}-{}", t.dir_name, std::process::id()));
            let _ = std::fs::remove_dir_all(&d);
            std::fs::create_dir_all(d.join(".venv")).unwrap();
            std::fs::write(d.join(t.marker), "").unwrap();
            assert!(!t.installed(&d));
            std::fs::write(d.join("pyproject.toml"), "[project]").unwrap();
            assert!(!t.installed(&d), "upstream's or an older environment is not installed");
            std::fs::write(d.join("pyproject.toml"), t.pyproject).unwrap();
            assert!(t.installed(&d));
            assert!(!t.weights_ready(&d));
            std::fs::write(d.join("READY"), t.ready_text()).unwrap();
            assert!(t.weights_ready(&d));
            std::fs::write(d.join("READY"), format!("{} 0000\n", t.pins[0].0)).unwrap();
            assert!(!t.weights_ready(&d), "weights at other revisions are not these weights");
            let _ = std::fs::remove_dir_all(&d);
        }
        // The other model's checkout is not this one's install.
        let d = std::env::temp_dir().join(format!("wobble-difflight-cross-{}", std::process::id()));
        std::fs::create_dir_all(d.join(".venv")).unwrap();
        std::fs::write(d.join(TURBO.marker), "").unwrap();
        std::fs::write(d.join("pyproject.toml"), LUXDIT.pyproject).unwrap();
        assert!(!LUXDIT.installed(&d) && !TURBO.installed(&d));
        let _ = std::fs::remove_dir_all(&d);
    }
}

// ------------------------------------------------------------- the suite

/// What the stand-in writes into every pixel: neutral, so the input
/// transform leaves it alone. With `--no-follow`, half of it — a second
/// smoothing pass is a different picture.
pub const FAKE_VALUE: f32 = 0.5;
/// The stand-in's latlong.
pub const FAKE_SIZE: (u32, u32) = (512, 256);

/// `--difflight-fake <args>`: the stand-in tool `--test-difflight` runs in
/// place of DiffusionLight-Turbo, so the whole pipeline — render, process,
/// adopt, playback — is exercised with no Python, no torch and no weights.
/// Each `Input` frame gets a marker in `Raw/` (kept across runs, like the
/// estimates) and a flat latlong in `<out>/`, and it speaks the driver's
/// protocol line for line. It refuses to run if `Raw/` was not kept.
pub fn fake(args: &[String]) -> anyhow::Result<()> {
    let arg = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let shot = PathBuf::from(arg("--shot").ok_or_else(|| anyhow::anyhow!("--shot"))?);
    let out_dir = shot.join(arg("--out").ok_or_else(|| anyhow::anyhow!("--out"))?);
    let follow = !args.iter().any(|a| a == "--no-follow");
    let mut inputs: Vec<PathBuf> = std::fs::read_dir(shot.join("Input"))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "exr"))
        .collect();
    inputs.sort();
    let raw = shot.join("Raw");
    std::fs::create_dir_all(&raw)?;
    std::fs::create_dir_all(&out_dir)?;
    let todo: Vec<&PathBuf> = inputs.iter().filter(|p| !raw.join(p.file_name().unwrap()).exists()).collect();
    println!("{} to estimate", todo.len());
    let total = todo.len() + inputs.len();
    println!("WOBBLE frame 0 {total}");
    for (i, p) in todo.iter().enumerate() {
        std::fs::write(raw.join(p.file_name().unwrap()), b"")?;
        println!("WOBBLE frame {} {total}", i + 1);
    }
    let (w, h) = (FAKE_SIZE.0 as usize, FAKE_SIZE.1 as usize);
    let v = half::f16::from_f32(if follow { FAKE_VALUE } else { FAKE_VALUE * 0.5 });
    let px: Vec<half::f16> = (0..w * h).flat_map(|_| [v, v, v, half::f16::ONE]).collect();
    for (i, path) in inputs.iter().enumerate() {
        engine::bakes::write_rgba_f16_exr_tagged(&out_dir.join(path.file_name().unwrap()), w, h, w * 4, &px, None)?;
        println!("WOBBLE frame {} {total}", todo.len() + i + 1);
    }
    println!("WOBBLE done");
    Ok(())
}

pub struct TestDiffLight {
    stage: u32,
    t0: web_time::Instant,
    since: web_time::Instant,
    node: u64,
    last: Option<f32>,
    at: u64,
}

impl Default for TestDiffLight {
    fn default() -> Self {
        let now = web_time::Instant::now();
        Self { stage: 0, t0: now, since: now, node: 0, last: None, at: 0 }
    }
}

impl Gfx {
    /// `--test-difflight`: a DiffusionLight node end to end in the live
    /// app, against the stand-in above. Unrun it is black (never the
    /// plate); `run` renders the plate, runs the process and adopts the
    /// latlong, at the LATLONG's size; it is not licence-tagged. Turning
    /// `follow` off is the same estimate directory and a new result name:
    /// STALE, and the re-run estimates nothing again. A new seed is a new
    /// estimate directory. Waits are wall-clock.
    pub(crate) fn drive_test_difflight(&mut self) {
        let Some(mut t) = self.test.difflight.take() else { return };
        let fail = |m: String| -> ! {
            println!("DIFFLIGHT TEST FAIL: {m}");
            std::process::exit(1);
        };
        if t.t0.elapsed().as_secs() > 120 {
            fail(format!("timeout at stage {}", t.stage));
        }
        let next = |t: &mut TestDiffLight, s: u32| {
            t.stage = s;
            t.since = web_time::Instant::now();
            t.last = None;
        };
        // A reading two probes agree on, 150 ms apart, at the playhead.
        let stable = |me: &mut Self, t: &mut TestDiffLight| -> Option<f32> {
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
        let phase = |me: &Self| match me.jobs.difflight.run.as_ref().map(|r| &r.phase) {
            Some(Phase::Failed(e)) => Err(e.clone()),
            Some(Phase::Done(_)) => Ok(true),
            _ => Ok(false),
        };
        let edit = |me: &mut Self, node: u64, f: &dyn Fn(&mut DiffLightParams)| {
            if let Some(NodeKind::DiffLight(p)) = me.egraph.node_mut(node).map(|n| &mut n.kind) {
                f(p);
            }
        };
        let near = |a: f32, b: f32| (a - b).abs() <= 2e-3 + 1e-2 * b.abs();
        match t.stage {
            0 if self.playback.is_some() && self.displayed_total >= 2 => {
                let exe = std::env::current_exe().unwrap_or_else(|e| fail(format!("current_exe: {e}")));
                std::env::set_var("WOBBLE_DIFFLIGHT_CMD", format!("{} --difflight-fake", exe.display()));
                let read = self
                    .egraph
                    .nodes
                    .iter()
                    .find(|(_, n)| matches!(n.kind, NodeKind::Read { .. }))
                    .map(|(id, _)| *id)
                    .unwrap_or_else(|| fail("no Read in the default graph".into()));
                let slot = self.next_read_slot;
                self.next_read_slot += 1;
                let k = self.egraph.add(NodeKind::DiffLight(DiffLightParams::new(slot)), egui::pos2(360.0, 200.0));
                self.egraph.connect(k, 0, read);
                if engine::cost::licence_of(&self.egraph.nodes[&k].kind) != engine::cost::Licence::Permissive {
                    fail("the DiffusionLight node is licence-tagged; its licences are permissive".into());
                }
                self.egraph.view = Some(k);
                self.playing = false;
                t.node = k;
                next(&mut t, 1);
            }
            // Unrun: black, not the plate.
            1 => {
                if let Some(v) = stable(self, &mut t) {
                    if v.abs() > 1e-4 {
                        fail(format!("unrun, the node reads {v}: it must be black, not the plate"));
                    }
                    t.at = self.timeline_pos;
                    self.difflight_action(t.node, Action::Run);
                    next(&mut t, 2);
                }
            }
            2 => match phase(self) {
                Err(e) => fail(format!("the run failed: {e}")),
                Ok(true) => {
                    if !self.loaded.contains_key(&t.node) {
                        fail("the result did not load into the node's reader".into());
                    }
                    let res = self.node_resolution(t.node, &mut std::collections::HashSet::new());
                    if res != Some(FAKE_SIZE) {
                        fail(format!("the node's format is {res:?}, the latlong is {FAKE_SIZE:?}"));
                    }
                    if self.cost_flow.is_nc(t.node) {
                        fail("the loaded node carries a licence taint".into());
                    }
                    self.seek_to(t.at);
                    next(&mut t, 3);
                }
                Ok(false) => {}
            },
            3 => {
                if let Some(v) = stable(self, &mut t) {
                    if !near(v, FAKE_VALUE) {
                        fail(format!("loaded, the node reads {v:.4}; the latlong holds {FAKE_VALUE}"));
                    }
                    if !matches!(self.difflight_view(t.node).loaded, Some((_, true))) {
                        fail(format!(
                            "the result should be current: playing {:?}, the node wants {:?}",
                            self.loaded.get(&t.node).map(|l| l.path.clone()),
                            self.difflight_out(t.node)
                        ));
                    }
                    let (dir, out) = (self.difflight_dir(t.node), self.difflight_out(t.node));
                    edit(self, t.node, &|p| p.follow = false);
                    if self.difflight_dir(t.node) != dir {
                        fail("a smoothing setting changed the ESTIMATE's directory".into());
                    }
                    if self.difflight_out(t.node) == out {
                        fail("a smoothing setting did not change the result's name".into());
                    }
                    if !matches!(self.difflight_view(t.node).loaded, Some((_, false))) {
                        fail("a new smoothing setting should show the result STALE".into());
                    }
                    self.difflight_action(t.node, Action::Run);
                    next(&mut t, 4);
                }
            }
            4 => match phase(self) {
                Err(e) => fail(format!("the smoothing re-run failed: {e}")),
                Ok(true) => {
                    let log = self.jobs.difflight.run.as_ref().map(|r| r.log.join(" | ")).unwrap_or_default();
                    if !log.contains("0 to estimate") {
                        fail(format!("the smoothing re-run estimated again: {log}"));
                    }
                    self.seek_to(t.at);
                    next(&mut t, 5);
                }
                Ok(false) => {}
            },
            5 => {
                if let Some(v) = stable(self, &mut t) {
                    if !near(v, FAKE_VALUE * 0.5) {
                        fail(format!("after the re-run the node reads {v:.4}, not the new pass's {}", FAKE_VALUE * 0.5));
                    }
                    let old = self.difflight_dir(t.node);
                    edit(self, t.node, &|p| p.seed += 1);
                    if self.difflight_dir(t.node) == old {
                        fail("a new seed did not change the estimate's directory".into());
                    }
                    // LuxDiT is the same node with the other model: a new
                    // estimate, and research-only, so tagged.
                    let old = self.difflight_dir(t.node);
                    edit(self, t.node, &|p| p.model = engine::graph::DiffLightModel::LuxDit);
                    if self.difflight_dir(t.node) == old {
                        fail("picking LuxDiT did not change the estimate's directory".into());
                    }
                    if engine::cost::licence_of(&self.egraph.nodes[&t.node].kind) != engine::cost::Licence::NonCommercial {
                        fail("a LuxDiT node is not tagged non-commercial".into());
                    }
                    println!(
                        "DIFFLIGHT TEST PASS: unrun the node is black; run rendered the plate, ran the tool and \
                         adopted the latlong at its own size, untagged; a smoothing change is STALE under the \
                         same estimates and its re-run estimates nothing again; a new seed is a new estimate; \
                         LuxDiT is a new estimate and tagged non-commercial"
                    );
                    std::process::exit(0);
                }
            }
            _ => {}
        }
        self.test.difflight = Some(t);
    }
}
