//! The LTX HDR node's app half: where LTX-2 lives, installing it, fetching
//! its (gated) weights, rendering the node's input to a shot directory,
//! running the HDR IC-LoRA pipeline as a process, and adopting what it
//! wrote. The node itself (params, codegen, the composite) is
//! `engine::nodes::ltxhdr`; the panel and glyph `crate::nodes::ltxhdr`.
//! The process plumbing is CorridorKey's (`crate::corridorkey::spawn`), the
//! input render the Upscale node's (`Gfx::render_input_frames`), the shape
//! of all of it the Depth node's.
//!
//! A run is two phases, each explicit and cancellable:
//!
//! 1. **render** — a `SubtreeBake` of input 0 over the input's frame range,
//!    at the input's own format, written as scene-linear ACEScg EXRs to
//!    `<dir>/Input` (the tool's `--input-colorspace acescg`);
//! 2. **run** — the tool, one process, driven by `tools/wobble_ltxhdr.py`
//!    (embedded, written next to the checkout on every run); `<dir>/HDR` is
//!    adopted as the node's reader when it says done.
//!
//! `<dir>` is keyed on the input's subtree key and format, the recipe
//! (scale, quality, keyframes, seed), the range and the fps — an edit
//! upstream shows as STALE and a re-run lands in a new directory. The
//! composite (clip, softness, raw) is NOT in it: it is uniforms over the
//! same files.
//!
//! **Nothing installs or downloads without a click.** `install…` fetches
//! the pinned checkout into `~/.wobble/LTX-2` (or `$WOBBLE_LTXHDR_DIR`) and
//! runs `uv sync`. `fetch weights…` runs the driver's `--fetch`, which
//! downloads four files at pinned revisions with the user's OWN Hugging
//! Face login — both repositories are gated behind the LTX-2 Community
//! Licence, which the user accepts on the model pages, not here.

use crate::corridorkey::{push_log, spawn, uv, Msg, UV_MISSING};
use crate::Gfx;
use engine::graph::{LtxHdrParams, NodeKind};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;

pub const DRIVER: &str = include_str!("../../tools/wobble_ltxhdr.py");
const DRIVER_NAME: &str = "wobble_ltxhdr.py";
pub const REPO: &str = "https://github.com/Lightricks/LTX-2.git";
/// The commit the driver is written against: it builds the pipeline from
/// `ltx_pipelines.hdr_ic_lora` the way that module's `main()` does, and
/// calls its EXR helpers, which a later commit is free to rename. Bump it
/// and re-test, never float it.
pub const COMMIT: &str = "e98f6d927d83c92ef2be6aec76c8440b19306e39";
pub const LICENCE_URL: &str = "https://github.com/Lightricks/LTX-2/blob/main/LICENSE";
/// The two gated model pages the user accepts the licence on.
pub const MODEL_PAGES: [&str; 2] = [
    "https://huggingface.co/Lightricks/LTX-2.5",
    "https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-SDR-To-HDR",
];
/// The four files `--fetch` writes, under the models directory: the same
/// list as the driver's `FILES`, with their sizes for the panel.
pub const WEIGHTS: [(&str, u64); 4] = [
    ("diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors", 42_018_190_584),
    ("vae/ltx-2.5-video-vae-bf16.safetensors", 1_472_223_346),
    ("ltx-2.5-22b-ic-lora-sdr-to-hdr-1.0.safetensors", 1_308_787_472),
    ("ltx-2.5-22b-ic-lora-sdr-to-hdr-scene-emb.safetensors", 12_614_165),
];

/// Where it runs and what it costs, for the panel.
pub const WHERE: &str = "CUDA (the transformer is cast to fp8: ~24 GB of VRAM at HD; pick offload cpu or disk under that), or Apple Silicon in bf16 — the transformer alone is 42 GB, so on a 48 GB Mac it does not fit at HD: use a proxy and offload disk, and expect it to be very slow or not to fit at all. The whole clip is one generation, which is what keeps the highlights steady over time";

pub fn install_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("WOBBLE_LTXHDR_DIR") {
        return PathBuf::from(d);
    }
    let home = engine::paths::home().unwrap_or_else(|| PathBuf::from("."));
    home.join(".wobble").join("LTX-2")
}

/// Where the weights land: beside the checkout, never in the catalog's
/// cache (they are the tool's, fetched under the user's own login).
pub fn models_dir() -> PathBuf {
    install_dir().join("models")
}

pub fn installed(dir: &Path) -> bool {
    dir.join("packages").join("ltx-pipelines").join("src").join("ltx_pipelines").join("hdr_ic_lora.py").is_file()
        && dir.join(".venv").is_dir()
}

/// All four weight files, at their full size.
pub fn weights_ready(models: &Path) -> bool {
    WEIGHTS.iter().all(|(f, n)| std::fs::metadata(models.join(f)).is_ok_and(|m| m.len() == *n))
}

pub fn weights_bytes() -> u64 {
    WEIGHTS.iter().map(|(_, n)| n).sum()
}

/// A suite's stand-in command replaces the tool (and its weights).
fn stand_in() -> bool {
    std::env::var_os("WOBBLE_LTXHDR_CMD").is_some()
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

/// An install or a weights fetch: a background job with a log.
pub struct Job {
    rx: Receiver<Msg>,
    pub log: Vec<String>,
    pub progress: (usize, usize),
    pub result: Option<Result<(), String>>,
}

#[derive(Default)]
pub struct Jobs {
    pub run: Option<Run>,
    pub install: Option<Job>,
    pub fetch: Option<Job>,
}

/// What the panel is told, built by `Gfx::ltxhdr_view`.
pub struct View {
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

pub fn params(kind: &NodeKind) -> Option<&LtxHdrParams> {
    match kind {
        NodeKind::LtxHdr(p) => Some(p),
        _ => None,
    }
}

/// The command that runs the driver: `$WOBBLE_LTXHDR_CMD` (a suite's
/// stand-in, whitespace-split) or `uv run --project <dir> python <driver>`.
fn command(dir: &Path) -> Result<std::process::Command, String> {
    if let Ok(s) = std::env::var("WOBBLE_LTXHDR_CMD") {
        let mut parts = s.split_whitespace();
        let exe = parts.next().ok_or("WOBBLE_LTXHDR_CMD is empty")?;
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
    pub(crate) fn ltxhdr_dir(&self, node: u64) -> Option<PathBuf> {
        let p = self.egraph.node(node).and_then(|n| params(&n.kind))?;
        let input = self.egraph.input_of(node, 0)?;
        let k0 = self.subtree_key_of(input)?;
        let ((a, b), _) = self.range_of(input);
        let size = self.node_resolution(input, &mut std::collections::HashSet::new());
        let recipe = serde_json::to_vec(&(p.recipe(), a, b, self.handles, size, self.fps.to_bits())).ok()?;
        let key = engine::diskcache::content_hash(&[&k0.to_le_bytes(), &recipe]);
        let root = match &self.disk_cache {
            Some(c) => c.root().join("ltxhdr"),
            None => engine::paths::temp_dir().join("wobble-ltxhdr"),
        };
        Some(root.join(format!("{key:016x}")))
    }

    pub(crate) fn ltxhdr_view(&self, node: u64) -> View {
        let dir = install_dir();
        let mine = self.ltxhdr.run.as_ref().filter(|r| r.node == node);
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
        let want = self.ltxhdr_dir(node).map(|d| d.join("HDR"));
        let loaded = self.loaded.get(&node).map(|l| {
            let current = want.as_ref().is_some_and(|d| l.path == d.to_string_lossy());
            (l.seq.frames.len(), current)
        });
        let fetch = self.ltxhdr.fetch.as_ref();
        View {
            dir: dir.to_string_lossy().into_owned(),
            installed: installed(&dir) || stand_in(),
            uv: uv().is_some(),
            installing: self
                .ltxhdr
                .install
                .as_ref()
                .filter(|i| i.result.is_none())
                .map(|i| i.log.last().cloned().unwrap_or_default()),
            install_result: self.ltxhdr.install.as_ref().and_then(|i| i.result.clone()),
            weights: weights_ready(&models_dir()) || stand_in(),
            fetching: fetch
                .filter(|f| f.result.is_none())
                .map(|f| (f.log.last().cloned().unwrap_or_default(), f.progress.0, f.progress.1)),
            fetch_result: fetch.and_then(|f| f.result.clone()),
            state,
            log: mine.map(|r| r.log.clone()).unwrap_or_default(),
            has_input: self.egraph.input_of(node, 0).is_some(),
            loaded,
            on_disk: want.as_ref().is_some_and(|d| engine::bakes::is_baked(d)),
            busy_elsewhere: self.ltxhdr.run.as_ref().is_some_and(|r| r.node != node && r.active()),
        }
    }

    pub(crate) fn ltxhdr_action(&mut self, node: u64, a: Action) {
        match a {
            Action::Run => self.ltxhdr_start(node),
            Action::Cancel => {
                if let Some(r) = self.ltxhdr.run.take() {
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
            Action::Install => self.ltxhdr_install(),
            Action::Fetch => self.ltxhdr_fetch(),
            Action::Dismiss => {
                if self.ltxhdr.run.as_ref().is_some_and(|r| r.node == node && !r.active()) {
                    self.ltxhdr.run = None;
                }
                if self.ltxhdr.install.as_ref().is_some_and(|i| i.result.is_some()) {
                    self.ltxhdr.install = None;
                }
                if self.ltxhdr.fetch.as_ref().is_some_and(|i| i.result.is_some()) {
                    self.ltxhdr.fetch = None;
                }
            }
        }
    }

    fn ltxhdr_fail(&mut self, node: u64, e: &str) {
        match self.ltxhdr.run.as_mut().filter(|r| r.node == node) {
            Some(r) => r.phase = Phase::Failed(e.to_string()),
            None => {
                self.ltxhdr.run = Some(Run { node, dir: PathBuf::new(), phase: Phase::Failed(e.into()), log: vec![] })
            }
        }
    }

    /// `install…`: fetch the pinned checkout and build its environment, on
    /// a thread, its output streamed to the panel.
    fn ltxhdr_install(&mut self) {
        if self.ltxhdr.install.as_ref().is_some_and(|i| i.result.is_none()) {
            return;
        }
        let dir = install_dir();
        let (tx, rx) = std::sync::mpsc::channel();
        let job = |rx, result| Job { rx, log: vec![], progress: (0, 0), result };
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (&dir, &tx);
            self.ltxhdr.install = Some(job(rx, Some(Err("the tool runs as a process: not in the web build".into()))));
            return;
        }
        #[cfg_attr(target_arch = "wasm32", allow(unreachable_code))]
        let Some(uv) = uv() else {
            self.ltxhdr.install = Some(job(rx, Some(Err(UV_MISSING.into()))));
            return;
        };
        let spawned = std::thread::Builder::new()
            .name("ltxhdr-install".into())
            .spawn(move || {
                let result = install_into(&dir, &uv, &tx);
                let _ = tx.send(match result {
                    Ok(()) => Msg::Done,
                    Err(e) => Msg::Failed(e),
                });
            })
            .is_ok();
        self.ltxhdr.install = Some(job(rx, (!spawned).then_some(Err("could not start a thread".into()))));
    }

    /// `fetch weights…`: the driver's `--fetch`, as a process, with the
    /// user's own Hugging Face login.
    fn ltxhdr_fetch(&mut self) {
        if self.ltxhdr.fetch.as_ref().is_some_and(|f| f.result.is_none()) {
            return;
        }
        let (tx, rx) = std::sync::mpsc::channel();
        let mut job = Job { rx, log: vec![], progress: (0, 0), result: None };
        if !installed(&install_dir()) {
            job.result = Some(Err("install LTX-2 first: the fetch runs in its environment".into()));
            self.ltxhdr.fetch = Some(job);
            return;
        }
        let mut cmd = match command(&install_dir()) {
            Ok(c) => c,
            Err(e) => {
                job.result = Some(Err(e));
                self.ltxhdr.fetch = Some(job);
                return;
            }
        };
        cmd.arg("--fetch").arg(models_dir());
        if let Err(e) = spawn(cmd, tx, Arc::new(AtomicBool::new(false)), "ltxhdr-fetch") {
            job.result = Some(Err(e));
        }
        self.ltxhdr.fetch = Some(job);
    }

    /// The `run` click: adopt a result on disk, else render the input.
    fn ltxhdr_start(&mut self, node: u64) {
        if self.ltxhdr.run.as_ref().is_some_and(|r| r.active()) {
            return;
        }
        #[cfg(target_arch = "wasm32")]
        return self.ltxhdr_fail(node, "LTX HDR runs as a process: not in the web build");
        #[cfg_attr(target_arch = "wasm32", allow(unreachable_code))]
        let Some(input) = self.egraph.input_of(node, 0) else {
            return self.ltxhdr_fail(node, "wire a picture into the input");
        };
        let Some(dir) = self.ltxhdr_dir(node) else {
            return self.ltxhdr_fail(node, "the input is not renderable yet");
        };
        let out = dir.join("HDR");
        if engine::bakes::is_baked(&out) {
            self.load_read(node, &out.to_string_lossy());
            self.playback_dirty = true;
            self.ltxhdr.run = Some(Run { node, dir, phase: Phase::Done("loaded the result on disk".into()), log: vec![] });
            return;
        }
        if !stand_in() {
            if !installed(&install_dir()) {
                return self.ltxhdr_fail(node, "LTX-2 is not installed: press install…");
            }
            if !weights_ready(&models_dir()) {
                return self.ltxhdr_fail(node, "the weights are not fetched: press fetch weights…");
            }
        }
        self.ltxhdr.run = Some(Run { node, dir: dir.clone(), phase: Phase::Render, log: vec![] });
        let size = self.node_resolution(input, &mut std::collections::HashSet::new());
        if let Err(e) = self.render_input_frames(node, input, dir, size, crate::BakeSink::LtxHdr) {
            self.ltxhdr_fail(node, &e);
        }
    }

    /// The bake loop finished rendering the input: run the tool.
    pub(crate) fn ltxhdr_render_finished(&mut self, node: u64) {
        let Some(r) = self.ltxhdr.run.as_ref().filter(|r| r.node == node) else { return };
        let dir = r.dir.clone();
        let Some(p) = self.egraph.node(node).and_then(|n| params(&n.kind)).cloned() else {
            return self.ltxhdr_fail(node, "the node is gone");
        };
        let mut cmd = match command(&install_dir()) {
            Ok(c) => c,
            Err(e) => return self.ltxhdr_fail(node, &e),
        };
        cmd.arg("--shot")
            .arg(&dir)
            .arg("--models")
            .arg(models_dir())
            .arg("--fps")
            .arg(format!("{}", self.fps))
            .arg("--seed")
            .arg(p.seed.to_string())
            .arg("--scale")
            .arg(format!("{}", p.scale.factor()))
            .arg("--offload")
            .arg(p.offload.arg());
        if p.high_quality {
            cmd.arg("--high-quality");
        }
        if !p.keyframes {
            cmd.arg("--no-keyframes");
        }
        let (tx, rx) = std::sync::mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        if let Err(e) = spawn(cmd, tx, cancel.clone(), "ltxhdr") {
            return self.ltxhdr_fail(node, &e);
        }
        if let Some(r) = self.ltxhdr.run.as_mut() {
            r.phase = Phase::Running { rx, cancel, done: 0, total: 0 };
        }
    }

    /// Every redraw: drain the jobs' messages, catch a render that died,
    /// adopt a finished result.
    pub(crate) fn ltxhdr_poll(&mut self) {
        if let Some(i) = self.ltxhdr.install.as_mut() {
            drain(i);
        }
        if let Some(f) = self.ltxhdr.fetch.as_mut() {
            drain(f);
        }
        let Some(r) = self.ltxhdr.run.as_mut() else { return };
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
                let out = r.dir.join("HDR");
                if let Err(e) = engine::bakes::mark_baked(&out) {
                    r.phase = Phase::Failed(format!("marking the result done: {e}"));
                    return;
                }
                r.phase = Phase::Done("HDR done".into());
                self.load_read(node, &out.to_string_lossy());
                self.playback_dirty = true;
            }
            Some(Err(e)) => r.phase = Phase::Failed(e),
            None => {}
        }
    }

    /// After a load, an undo, a paste: every LTX HDR node whose result for
    /// the current recipe is on disk plays it.
    pub(crate) fn ltxhdr_adopt(&mut self) {
        let ids: Vec<u64> = self
            .egraph
            .nodes
            .iter()
            .filter(|(id, n)| {
                matches!(n.kind, NodeKind::LtxHdr(_)) && !self.loaded.contains_key(id) && !self.load_errors.contains_key(id)
            })
            .map(|(id, _)| *id)
            .collect();
        for id in ids {
            let Some(dir) = self.ltxhdr_dir(id) else { continue };
            let out = dir.join("HDR");
            if engine::bakes::is_baked(&out) {
                self.load_read(id, &out.to_string_lossy());
                self.playback_dirty = true;
            }
        }
    }
}

/// Fetch the pinned commit into `dir` (fresh, or an earlier install at
/// another commit) and `uv sync` the pipelines package.
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
    let _ = tx.send(Msg::Line("uv sync --package ltx-pipelines (torch and friends: a few GB, a while)".into()));
    let mut c = std::process::Command::new(uv);
    c.arg("sync").arg("--package").arg("ltx-pipelines").arg("--python").arg("3.12").current_dir(dir);
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
            "--fps",
            "--seed",
            "--scale",
            "--high-quality",
            "--no-keyframes",
            "--offload",
            "--fetch",
            "\"HDR\"",
            "\"Input\"",
            "hdr_ic_lora_arg_parser",
            "HDRICLoraPipeline",
            "\"acescg\"",
        ] {
            assert!(DRIVER.contains(w), "{w}");
        }
    }

    #[test]
    fn the_driver_and_the_app_agree_on_the_weights_and_the_pins() {
        for (f, _) in WEIGHTS {
            assert!(DRIVER.contains(f), "{f}");
        }
        assert_eq!(COMMIT.len(), 40);
        assert!(COMMIT.chars().all(|c| c.is_ascii_hexdigit()));
        // Both Hugging Face revisions are pinned commits, never a branch.
        for rev in ["BASE_REV = \"", "HDR_REV = \""] {
            let at = DRIVER.find(rev).unwrap() + rev.len();
            let sha = &DRIVER[at..at + 40];
            assert!(sha.chars().all(|c| c.is_ascii_hexdigit()), "{rev}{sha}");
        }
        for page in MODEL_PAGES {
            let repo = page.trim_start_matches("https://huggingface.co/");
            assert!(DRIVER.contains(&format!("\"{repo}\"")), "{repo}");
        }
    }

    #[test]
    fn the_padding_matches_the_engine() {
        // The driver's copy of `padded_frames`, checked against the engine.
        assert!(DRIVER.contains("-(-(n - 1) // 8) * 8 + 1"));
        for n in [1usize, 2, 9, 10, 48] {
            let py = if n <= 1 { 1 } else { (n - 1).div_ceil(8) * 8 + 1 };
            assert_eq!(py, LtxHdrParams::padded_frames(n));
        }
    }

    #[test]
    fn only_the_fetch_mode_touches_the_network() {
        // Everything that downloads sits inside `def fetch`.
        let code = DRIVER;
        let fetch_at = code.find("def fetch(").unwrap();
        let next_def = code[fetch_at + 1..].find("\ndef ").map(|i| i + fetch_at + 1).unwrap();
        for (i, l) in code.lines().enumerate() {
            let at = code.lines().take(i).map(|l| l.len() + 1).sum::<usize>();
            let inside = (fetch_at..next_def).contains(&at);
            for w in ["hf_hub_download", "snapshot_download", "urllib", "requests"] {
                if l.contains(w) && !l.trim_start().starts_with('#') {
                    assert!(inside, "line {}: {w} outside the fetch mode", i + 1);
                }
            }
        }
    }
}

// ------------------------------------------------------------- the suite

/// What the stand-in does to every pixel: multiplies it, so a result that
/// is shown reads differently from the plate under it.
pub const FAKE_GAIN: f32 = 4.0;

/// `--ltxhdr-fake <args>`: the stand-in tool `--test-ltxhdr` runs in place
/// of LTX-2, so the whole pipeline — render, process, adopt, composite,
/// playback — is exercised with no Python, no torch and no weights. Each
/// `Input` frame becomes an `HDR` frame of the same size (scaled by
/// `--scale`, nearest) holding the input times `FAKE_GAIN`, and it speaks
/// the driver's protocol line for line.
pub fn fake(args: &[String]) -> anyhow::Result<()> {
    use engine::decode::ExrDecoder;
    let arg = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let shot = PathBuf::from(arg("--shot").ok_or_else(|| anyhow::anyhow!("--shot"))?);
    let scale: f32 = arg("--scale").and_then(|s| s.parse().ok()).unwrap_or(1.0);
    let dec = engine::decode::AnyDecoder::new(false);
    let mut inputs: Vec<PathBuf> = std::fs::read_dir(shot.join("Input"))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "exr"))
        .collect();
    inputs.sort();
    let out_dir = shot.join("HDR");
    std::fs::create_dir_all(&out_dir)?;
    let total = inputs.len();
    println!("WOBBLE frame 0 {total}");
    for (i, path) in inputs.iter().enumerate() {
        let info = dec.read_header(path)?;
        let stride = info.row_stride_f16();
        let mut px = vec![half::f16::ZERO; stride * info.height as usize];
        dec.decode_into(path, info, stride, &mut px)?;
        let w = ((info.width as f32 * scale).round() as usize).max(1);
        let h = ((info.height as f32 * scale).round() as usize).max(1);
        let mut out = vec![half::f16::ZERO; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let sx = (x * info.width as usize / w).min(info.width as usize - 1);
                let sy = (y * info.height as usize / h).min(info.height as usize - 1);
                let s = sy * stride + sx * 4;
                for c in 0..3 {
                    out[(y * w + x) * 4 + c] = half::f16::from_f32(px[s + c].to_f32() * FAKE_GAIN);
                }
                out[(y * w + x) * 4 + 3] = half::f16::ONE;
            }
        }
        engine::bakes::write_rgba_f16_exr_tagged(&out_dir.join(path.file_name().unwrap()), w, h, w * 4, &out, None)?;
        println!("WOBBLE frame {} {total}", i + 1);
    }
    println!("WOBBLE done");
    Ok(())
}

pub struct TestLtxHdr {
    stage: u32,
    t0: web_time::Instant,
    since: web_time::Instant,
    node: u64,
    last: Option<f32>,
    at: u64,
    plate: f32,
}

impl Default for TestLtxHdr {
    fn default() -> Self {
        let now = web_time::Instant::now();
        Self { stage: 0, t0: now, since: now, node: 0, last: None, at: 0, plate: 0.0 }
    }
}

impl Gfx {
    /// `--test-ltxhdr`: an LTX HDR node end to end in the live app, against
    /// the stand-in above. Unrun it passes the plate through; `run` renders
    /// the plate (ACEScg), runs the process, adopts `HDR`; with the clip set
    /// under the plate every pixel is "clipped" and the composite shows the
    /// result (the plate times `FAKE_GAIN`); with the clip set over it the
    /// composite shows the plate again, and raw shows the result — none of
    /// the three a new directory. The node is tagged Restricted. A new seed
    /// is a new recipe: STALE until re-run. Waits are wall-clock.
    pub(crate) fn drive_test_ltxhdr(&mut self) {
        let Some(mut t) = self.test_ltxhdr.take() else { return };
        let fail = |m: String| -> ! {
            println!("LTXHDR TEST FAIL: {m}");
            std::process::exit(1);
        };
        if t.t0.elapsed().as_secs() > 120 {
            fail(format!("timeout at stage {}", t.stage));
        }
        let next = |t: &mut TestLtxHdr, s: u32| {
            t.stage = s;
            t.since = web_time::Instant::now();
            t.last = None;
        };
        // A reading two probes agree on, 150 ms apart, at the playhead.
        let stable = |me: &mut Self, t: &mut TestLtxHdr| -> Option<f32> {
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
        let phase = |me: &Self| match me.ltxhdr.run.as_ref().map(|r| &r.phase) {
            Some(Phase::Failed(e)) => Err(e.clone()),
            Some(Phase::Done(_)) => Ok(true),
            _ => Ok(false),
        };
        let edit = |me: &mut Self, node: u64, f: &dyn Fn(&mut LtxHdrParams)| {
            if let Some(NodeKind::LtxHdr(p)) = me.egraph.node_mut(node).map(|n| &mut n.kind) {
                f(p);
            }
        };
        let near = |a: f32, b: f32| (a - b).abs() <= 2e-3 + 1e-2 * b.abs();
        match t.stage {
            0 if self.playback.is_some() && self.displayed_total >= 2 => {
                let exe = std::env::current_exe().unwrap_or_else(|e| fail(format!("current_exe: {e}")));
                std::env::set_var("WOBBLE_LTXHDR_CMD", format!("{} --ltxhdr-fake", exe.display()));
                let read = self
                    .egraph
                    .nodes
                    .iter()
                    .find(|(_, n)| matches!(n.kind, NodeKind::Read { .. }))
                    .map(|(id, _)| *id)
                    .unwrap_or_else(|| fail("no Read in the default graph".into()));
                let slot = self.next_read_slot;
                self.next_read_slot += 1;
                let k = self.egraph.add(NodeKind::LtxHdr(LtxHdrParams::new(slot)), egui::pos2(360.0, 200.0));
                self.egraph.connect(k, 0, read);
                if engine::cost::licence_of(&self.egraph.nodes[&k].kind) != engine::cost::Licence::Restricted {
                    fail("the LTX HDR node is not tagged Restricted".into());
                }
                self.egraph.view = Some(k);
                self.playing = false;
                t.node = k;
                next(&mut t, 1);
            }
            // Unrun: the plate itself.
            1 => {
                if let Some(v) = stable(self, &mut t) {
                    if v.abs() < 1e-3 {
                        fail(format!("the probe reads {v}: the plate is black there, the test cannot tell"));
                    }
                    t.plate = v;
                    t.at = self.timeline_pos;
                    self.ltxhdr_action(t.node, Action::Run);
                    next(&mut t, 2);
                }
            }
            2 => match phase(self) {
                Err(e) => fail(format!("the run failed: {e}")),
                Ok(true) => {
                    if !self.loaded.contains_key(&t.node) {
                        fail("the result did not load into the node's reader".into());
                    }
                    if !self.cost_flow.is_nc(t.node) {
                        fail("the loaded node does not carry the licence taint".into());
                    }
                    // Everything over a tiny clip: composited, all result.
                    edit(self, t.node, &|p| {
                        p.clip = 1e-3;
                        p.softness = 0.0;
                    });
                    self.seek_to(t.at);
                    next(&mut t, 3);
                }
                Ok(false) => {}
            },
            3 => {
                if let Some(v) = stable(self, &mut t) {
                    let want = t.plate * FAKE_GAIN;
                    if !near(v, want) {
                        fail(format!("clipped everywhere, the composite reads {v:.4}; the result holds {want:.4}"));
                    }
                    let dir = self.ltxhdr_dir(t.node);
                    edit(self, t.node, &|p| p.clip = 1000.0);
                    if self.ltxhdr_dir(t.node) != dir {
                        fail("the clip level changed the result's directory".into());
                    }
                    next(&mut t, 4);
                }
            }
            4 => {
                if let Some(v) = stable(self, &mut t) {
                    if !near(v, t.plate) {
                        fail(format!("clipped nowhere, the composite reads {v:.4}, not the plate's {:.4}", t.plate));
                    }
                    edit(self, t.node, &|p| p.raw = true);
                    next(&mut t, 5);
                }
            }
            5 => {
                if let Some(v) = stable(self, &mut t) {
                    let want = t.plate * FAKE_GAIN;
                    if !near(v, want) {
                        fail(format!("raw reads {v:.4}, the result holds {want:.4}"));
                    }
                    let v = self.ltxhdr_view(t.node);
                    if !matches!(v.loaded, Some((_, true))) {
                        fail(format!("the result should be current, the view says {:?}", v.loaded));
                    }
                    let old = self.ltxhdr_dir(t.node);
                    edit(self, t.node, &|p| p.seed += 1);
                    if self.ltxhdr_dir(t.node) == old {
                        fail("a new seed did not change the result's directory".into());
                    }
                    let v = self.ltxhdr_view(t.node);
                    if !matches!(v.loaded, Some((_, false))) {
                        fail(format!("a new seed should show the result STALE, the view says {:?}", v.loaded));
                    }
                    println!(
                        "LTXHDR TEST PASS: unrun the node passes the plate; run rendered it, ran the tool and \
                         adopted the result; the composite shows the result where the plate is over the clip \
                         and the plate where it is under, raw shows the result, none of them a new directory; \
                         tagged Restricted; a new seed is a new recipe, STALE until re-run"
                    );
                    std::process::exit(0);
                }
            }
            _ => {}
        }
        self.test_ltxhdr = Some(t);
    }
}
