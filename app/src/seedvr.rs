//! The SeedVR2 node's app half: where the tool lives, installing it,
//! rendering the node's input to a shot directory, running the tool as a
//! process, and adopting what it wrote. The node itself (params, codegen)
//! is `engine::nodes::seedvr`; the panel and glyph `crate::nodes::seedvr`.
//! The process plumbing is CorridorKey's (`crate::corridorkey::spawn`), the
//! same protocol line for line; the input render is the Upscale node's
//! (`Gfx::render_input_frames`).
//!
//! A run is two phases, each explicit and cancellable:
//!
//! 1. **render** — a `SubtreeBake` of input 0 over the input's frame range,
//!    at the input's own format, written as linear Rec.709 EXRs to
//!    `<dir>/Input`;
//! 2. **run** — the tool, one process, driven by `tools/wobble_seedvr.py`
//!    (embedded, written next to the checkout on every run), reporting a
//!    line per chunk; `<dir>/Output` is adopted as the node's reader when it
//!    says done.
//!
//! `<dir>` is keyed on everything that shapes the pixels — the input's
//! subtree key and format, the recipe, the range — so an edit upstream
//! shows as STALE and a re-run lands in a new directory; the old result
//! keeps playing until then (the GraphBake contract).
//!
//! **Nothing installs or downloads without a click.** `install…` fetches
//! the pinned CLI into `~/.wobble/SeedVR2` (or `$WOBBLE_SEEDVR_DIR`) and
//! runs `uv sync` there; the tool fetches its weights from Hugging Face on
//! its first run, and the panel says how much before the button.
//!
//! **Where it runs**: CUDA or Apple Silicon (the CLI has an MPS path).
//! Measured on an M4 Pro (48 GB): the 3B model, 960x540 to 1920x1080, ~32 s
//! a frame once loaded, 28 GB of unified memory at the peak (VAE tiled).
//! The 7B wants a big card; a 16 GB NVIDIA card runs the fp8 builds.

use crate::corridorkey::{push_log, spawn, uv, Msg, UV_MISSING};
use crate::Gfx;
use engine::graph::{NodeKind, SeedVrParams};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;

pub const DRIVER: &str = include_str!("../../tools/wobble_seedvr.py");
const DRIVER_NAME: &str = "wobble_seedvr.py";
pub const REPO: &str = "https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git";
/// The CLI commit the driver is written against: it calls the tool's own
/// pipeline, which a later commit is free to rename.
pub const COMMIT: &str = "4490bd1f482e026674543386bb2a4d176da245b9";
pub const LICENCE_URL: &str = "https://github.com/ByteDance-Seed/SeedVR/blob/main/LICENSE";

/// What the first run downloads, by model (the fp16 builds; a CUDA card
/// under 20 GB takes the fp8 ones, about half).
pub fn weights_words(p: &SeedVrParams) -> &'static str {
    match p.model {
        engine::graph::SeedModel::B3 => "~7 GB of weights (3B + VAE)",
        _ => "~17 GB of weights (7B + VAE)",
    }
}

pub fn install_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("WOBBLE_SEEDVR_DIR") {
        return PathBuf::from(d);
    }
    let home = engine::paths::home().unwrap_or_else(|| PathBuf::from("."));
    home.join(".wobble").join("SeedVR2")
}

pub fn installed(dir: &Path) -> bool {
    dir.join("inference_cli.py").is_file()
        && dir.join(".venv").is_dir()
        && std::fs::read_to_string(dir.join("pyproject.toml")).is_ok_and(|t| t.contains(CONSTRAINTS))
}

/// Appended to the CLI's own pyproject by `install…`: its `opencv-python`
/// is unpinned, and 5.x's Windows wheel has no EXR codec (imread returns
/// None on every frame the driver reads). An install without it shows as
/// not installed, so the button comes back and re-syncs.
pub const CONSTRAINTS: &str = "\n[tool.uv]\nconstraint-dependencies = [\"opencv-python<5\"]\n";

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

/// What the panel is told, built by `Gfx::seedvr_view`.
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

pub fn params(kind: &NodeKind) -> Option<&SeedVrParams> {
    match kind {
        NodeKind::SeedVr(p) => Some(p),
        _ => None,
    }
}

/// The command that runs the driver: `$WOBBLE_SEEDVR_CMD` (a suite's
/// stand-in, whitespace-split) or `uv run --project <dir> python <driver>`.
fn command(dir: &Path) -> Result<std::process::Command, String> {
    if let Ok(s) = std::env::var("WOBBLE_SEEDVR_CMD") {
        let mut parts = s.split_whitespace();
        let exe = parts.next().ok_or("WOBBLE_SEEDVR_CMD is empty")?;
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
    pub(crate) fn seedvr_dir(&self, node: u64) -> Option<PathBuf> {
        let p = self.egraph.node(node).and_then(|n| params(&n.kind))?;
        let input = self.egraph.input_of(node, 0)?;
        let k0 = self.subtree_key_of(input)?;
        let ((a, b), _) = self.range_of(input);
        let size = self.node_resolution(input, &mut std::collections::HashSet::new());
        let recipe = serde_json::to_vec(&(p.recipe(), a, b, self.handles, size)).ok()?;
        let key = engine::diskcache::content_hash(&[&k0.to_le_bytes(), &recipe]);
        let root = match &self.disk_cache {
            Some(c) => c.root().join("seedvr"),
            None => engine::paths::temp_dir().join("wobble-seedvr"),
        };
        Some(root.join(format!("{key:016x}")))
    }

    pub(crate) fn seedvr_view(&self, node: u64) -> View {
        let dir = install_dir();
        let mine = self.seedvr.run.as_ref().filter(|r| r.node == node);
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
        let want = self.seedvr_dir(node).map(|d| d.join("Output"));
        let loaded = self.loaded.get(&node).map(|l| {
            let current = want.as_ref().is_some_and(|d| l.path == d.to_string_lossy());
            (l.seq.frames.len(), current)
        });
        View {
            dir: dir.to_string_lossy().into_owned(),
            installed: installed(&dir) || std::env::var_os("WOBBLE_SEEDVR_CMD").is_some(),
            uv: uv().is_some(),
            installing: self.seedvr.install.as_ref().filter(|i| i.result.is_none()).map(|i| i.log.last().cloned().unwrap_or_default()),
            install_result: self.seedvr.install.as_ref().and_then(|i| i.result.clone()),
            state,
            log: mine.map(|r| r.log.clone()).unwrap_or_default(),
            has_input: self.egraph.input_of(node, 0).is_some(),
            loaded,
            on_disk: want.as_ref().is_some_and(|d| engine::bakes::is_baked(d)),
            busy_elsewhere: self.seedvr.run.as_ref().is_some_and(|r| r.node != node && r.active()),
        }
    }

    pub(crate) fn seedvr_action(&mut self, node: u64, a: Action) {
        match a {
            Action::Run => self.seedvr_start(node),
            Action::Cancel => {
                if let Some(r) = self.seedvr.run.take() {
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
            Action::Install => self.seedvr_install(),
            Action::Dismiss => {
                if self.seedvr.run.as_ref().is_some_and(|r| r.node == node && !r.active()) {
                    self.seedvr.run = None;
                }
                if self.seedvr.install.as_ref().is_some_and(|i| i.result.is_some()) {
                    self.seedvr.install = None;
                }
            }
        }
    }

    fn seedvr_fail(&mut self, node: u64, e: &str) {
        match self.seedvr.run.as_mut().filter(|r| r.node == node) {
            Some(r) => r.phase = Phase::Failed(e.to_string()),
            None => self.seedvr.run = Some(Run { node, dir: PathBuf::new(), phase: Phase::Failed(e.into()), log: vec![] }),
        }
    }

    /// `install…`: fetch the pinned CLI and build its environment, on a
    /// thread, its output streamed to the panel.
    fn seedvr_install(&mut self) {
        if self.seedvr.install.as_ref().is_some_and(|i| i.result.is_none()) {
            return;
        }
        let dir = install_dir();
        let (tx, rx) = std::sync::mpsc::channel();
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (&dir, &tx);
            self.seedvr.install = Some(Install { rx, log: vec![], result: Some(Err("the tool runs as a process: not in the web build".into())) });
            return;
        }
        #[cfg_attr(target_arch = "wasm32", allow(unreachable_code))]
        let Some(uv) = uv() else {
            self.seedvr.install = Some(Install { rx, log: vec![], result: Some(Err(UV_MISSING.into())) });
            return;
        };
        let spawned = std::thread::Builder::new()
            .name("seedvr-install".into())
            .spawn(move || {
                let result = install_into(&dir, &uv, &tx);
                let _ = tx.send(match result {
                    Ok(()) => Msg::Done,
                    Err(e) => Msg::Failed(e),
                });
            })
            .is_ok();
        self.seedvr.install = Some(Install { rx, log: vec![], result: (!spawned).then_some(Err("could not start a thread".into())) });
    }

    /// The `run` click: adopt a result on disk, else render the input.
    fn seedvr_start(&mut self, node: u64) {
        if self.seedvr.run.as_ref().is_some_and(|r| r.active()) {
            return;
        }
        #[cfg(target_arch = "wasm32")]
        return self.seedvr_fail(node, "SeedVR2 runs as a process: not in the web build");
        #[cfg_attr(target_arch = "wasm32", allow(unreachable_code))]
        let Some(input) = self.egraph.input_of(node, 0) else {
            return self.seedvr_fail(node, "wire a picture into the input");
        };
        let Some(dir) = self.seedvr_dir(node) else {
            return self.seedvr_fail(node, "the input is not renderable yet");
        };
        let out = dir.join("Output");
        if engine::bakes::is_baked(&out) {
            self.load_read(node, &out.to_string_lossy());
            self.playback_dirty = true;
            self.seedvr.run = Some(Run { node, dir, phase: Phase::Done("loaded the result on disk".into()), log: vec![] });
            return;
        }
        if !installed(&install_dir()) && std::env::var_os("WOBBLE_SEEDVR_CMD").is_none() {
            return self.seedvr_fail(node, "SeedVR2 is not installed: press install…");
        }
        self.seedvr.run = Some(Run { node, dir: dir.clone(), phase: Phase::Render, log: vec![] });
        let size = self.node_resolution(input, &mut std::collections::HashSet::new());
        if let Err(e) = self.render_input_frames(node, input, dir, size, crate::BakeSink::SeedVr) {
            self.seedvr_fail(node, &e);
        }
    }

    /// The bake loop finished rendering the input: run the tool.
    pub(crate) fn seedvr_render_finished(&mut self, node: u64) {
        let Some(r) = self.seedvr.run.as_ref().filter(|r| r.node == node) else { return };
        let dir = r.dir.clone();
        let Some(p) = self.egraph.node(node).and_then(|n| params(&n.kind)).cloned() else {
            return self.seedvr_fail(node, "the node is gone");
        };
        let Some(size) = self.egraph.input_of(node, 0).and_then(|i| self.node_resolution(i, &mut std::collections::HashSet::new())) else {
            return self.seedvr_fail(node, "the input has no format");
        };
        let mut cmd = match command(&install_dir()) {
            Ok(c) => c,
            Err(e) => return self.seedvr_fail(node, &e),
        };
        cmd.arg("--shot")
            .arg(&dir)
            .arg("--model")
            .arg(p.model.arg())
            .arg("--resolution")
            .arg(p.resolution_for(size).to_string())
            .arg("--seed")
            .arg(p.seed.to_string())
            .arg("--batch")
            .arg(p.batch_arg().to_string())
            .arg("--colour")
            .arg(p.colour.arg());
        let (tx, rx) = std::sync::mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        if let Err(e) = spawn(cmd, tx, cancel.clone(), "seedvr") {
            return self.seedvr_fail(node, &e);
        }
        if let Some(r) = self.seedvr.run.as_mut() {
            r.phase = Phase::Running { rx, cancel, done: 0, total: 0 };
        }
    }

    /// Every redraw: drain the tool's messages, catch a render that died,
    /// adopt a finished result.
    pub(crate) fn seedvr_poll(&mut self) {
        if let Some(i) = self.seedvr.install.as_mut() {
            while let Ok(m) = i.rx.try_recv() {
                match m {
                    Msg::Line(l) => push_log(&mut i.log, l),
                    Msg::Done => i.result = Some(Ok(())),
                    Msg::Failed(e) => i.result = Some(Err(e)),
                    Msg::Frame(..) => {}
                }
            }
        }
        let Some(r) = self.seedvr.run.as_mut() else { return };
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
                let out = r.dir.join("Output");
                if let Err(e) = engine::bakes::mark_baked(&out) {
                    r.phase = Phase::Failed(format!("marking the result done: {e}"));
                    return;
                }
                r.phase = Phase::Done("restored".into());
                self.load_read(node, &out.to_string_lossy());
                self.playback_dirty = true;
            }
            Some(Err(e)) => r.phase = Phase::Failed(e),
            None => {}
        }
    }

    /// After a load, an undo, a paste: every SeedVR2 node whose result for
    /// the current recipe is on disk plays it.
    pub(crate) fn seedvr_adopt(&mut self) {
        let ids: Vec<u64> = self
            .egraph
            .nodes
            .iter()
            .filter(|(id, n)| {
                matches!(n.kind, NodeKind::SeedVr(_)) && !self.loaded.contains_key(id) && !self.load_errors.contains_key(id)
            })
            .map(|(id, _)| *id)
            .collect();
        for id in ids {
            let Some(dir) = self.seedvr_dir(id) else { continue };
            let out = dir.join("Output");
            if engine::bakes::is_baked(&out) {
                self.load_read(id, &out.to_string_lossy());
                self.playback_dirty = true;
            }
        }
    }
}

/// Fetch the pinned commit into `dir` (fresh, or an earlier install at
/// another commit) and `uv sync` it.
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
    let proj = dir.join("pyproject.toml");
    let mut t = std::fs::read_to_string(&proj).map_err(|e| format!("reading pyproject.toml: {e}"))?;
    t.push_str(CONSTRAINTS);
    std::fs::write(&proj, t).map_err(|e| format!("writing pyproject.toml: {e}"))?;
    let _ = tx.send(Msg::Line("uv sync (torch and friends: several GB, a while)".into()));
    let mut c = std::process::Command::new(uv);
    c.arg("sync").arg("--python").arg("3.12").current_dir(dir);
    run(c).map_err(|e| format!("uv sync: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_driver_speaks_the_protocol_the_reader_parses() {
        for w in ["WOBBLE frame", "WOBBLE done", "WOBBLE error", "--shot", "--resolution", "_process_frames_core", "Output"] {
            assert!(DRIVER.contains(w), "{w}");
        }
    }

    #[test]
    fn the_install_is_pinned() {
        assert_eq!(COMMIT.len(), 40);
        assert!(COMMIT.chars().all(|c| c.is_ascii_hexdigit()));
    }
}

// ------------------------------------------------------------- the suite

/// `--seedvr-fake <args>`: the stand-in tool `--test-seedvr` runs in place
/// of SeedVR2, so the whole pipeline — render, process, adopt, playback —
/// is exercised with no Python, no torch and no weights. It upscales each
/// `Input` frame by pixel replication to the `--resolution` asked for and
/// speaks the driver's protocol line for line.
pub fn fake(args: &[String]) -> anyhow::Result<()> {
    use engine::decode::ExrDecoder;
    let arg = |k: &str| args.iter().position(|a| a == k).and_then(|i| args.get(i + 1)).cloned();
    let shot = PathBuf::from(arg("--shot").ok_or_else(|| anyhow::anyhow!("--shot"))?);
    let res: u32 = arg("--resolution").and_then(|r| r.parse().ok()).ok_or_else(|| anyhow::anyhow!("--resolution"))?;
    let dec = engine::decode::AnyDecoder::new(false);
    let mut inputs: Vec<PathBuf> = std::fs::read_dir(shot.join("Input"))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "exr"))
        .collect();
    inputs.sort();
    let out_dir = shot.join("Output");
    std::fs::create_dir_all(&out_dir)?;
    let total = inputs.len();
    println!("WOBBLE frame 0 {total}");
    for (i, path) in inputs.iter().enumerate() {
        let info = dec.read_header(path)?;
        let stride = info.row_stride_f16();
        let mut px = vec![half::f16::ZERO; stride * info.height as usize];
        dec.decode_into(path, info, stride, &mut px)?;
        let k = (res / info.width.min(info.height)).max(1) as usize;
        let (w, h) = (info.width as usize * k, info.height as usize * k);
        let mut out = vec![half::f16::ZERO; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let s = (y / k) * stride + (x / k) * 4;
                out[(y * w + x) * 4..][..4].copy_from_slice(&px[s..s + 4]);
            }
        }
        engine::bakes::write_rgba_f16_exr_tagged(&out_dir.join(path.file_name().unwrap()), w, h, w * 4, &out, None)?;
        println!("WOBBLE frame {} {total}", i + 1);
    }
    println!("WOBBLE done");
    Ok(())
}

pub struct TestSeedVr {
    stage: u32,
    t0: web_time::Instant,
    since: web_time::Instant,
    node: u64,
    plate: Option<(u32, u32)>,
    before: Option<f32>,
    last: Option<f32>,
    /// The tick both probes read: a render walks the range and leaves the
    /// playhead at its end.
    at: u64,
}

impl Default for TestSeedVr {
    fn default() -> Self {
        let now = web_time::Instant::now();
        Self { stage: 0, t0: now, since: now, node: 0, plate: None, before: None, last: None, at: 0 }
    }
}

impl Gfx {
    /// `--test-seedvr`: a SeedVR2 node end to end in the live app, against
    /// the stand-in above. `run` renders the plate at its own format, runs
    /// the process with the asked-for resolution (the plate's short side
    /// times the node's 2x), adopts `Output`; the node's format is now the
    /// result's — twice the plate — and the picture through it is the plate
    /// (the stand-in replicates pixels; the Rec.709 round trip through the
    /// working space is what that checks). Then a seed change is a new
    /// recipe: STALE until re-run, into a new directory. Waits are
    /// wall-clock.
    pub(crate) fn drive_test_seedvr(&mut self) {
        let Some(mut t) = self.test_seedvr.take() else { return };
        let fail = |m: String| -> ! {
            println!("SEEDVR TEST FAIL: {m}");
            std::process::exit(1);
        };
        if t.t0.elapsed().as_secs() > 180 {
            fail(format!("timeout at stage {}", t.stage));
        }
        let next = |t: &mut TestSeedVr, s: u32| {
            t.stage = s;
            t.since = web_time::Instant::now();
            t.last = None;
        };
        // A reading two probes agree on, 150 ms apart, at the playhead.
        // `s` is the box side: the same patch of plate is twice as big in
        // the result.
        let stable = |me: &mut Self, t: &mut TestSeedVr, x: u32, y: u32, s: u32| -> Option<f32> {
            if me.force_display || me.eval_tick() != me.timeline_pos || t.since.elapsed().as_millis() < 150 {
                return None;
            }
            let v = me.probe_output_box(x, y, s, s, 0)?;
            t.since = web_time::Instant::now();
            match t.last.replace(v) {
                Some(p) if (p - v).abs() < 1e-5 => {
                    t.last = None;
                    Some(v)
                }
                _ => None,
            }
        };
        let phase = |me: &Self| match me.seedvr.run.as_ref().map(|r| &r.phase) {
            Some(Phase::Failed(e)) => Err(e.clone()),
            Some(Phase::Done(_)) => Ok(true),
            _ => Ok(false),
        };
        match t.stage {
            0 if self.playback.is_some() && self.displayed_total >= 2 => {
                let exe = std::env::current_exe().unwrap_or_else(|e| fail(format!("current_exe: {e}")));
                std::env::set_var("WOBBLE_SEEDVR_CMD", format!("{} --seedvr-fake", exe.display()));
                let read = self
                    .egraph
                    .nodes
                    .iter()
                    .find(|(_, n)| matches!(n.kind, NodeKind::Read { .. }))
                    .map(|(id, _)| *id)
                    .unwrap_or_else(|| fail("no Read in the default graph".into()));
                t.plate = self.node_resolution(read, &mut std::collections::HashSet::new());
                let slot = self.next_read_slot;
                self.next_read_slot += 1;
                let k = self.egraph.add(NodeKind::SeedVr(SeedVrParams::new(slot)), egui::pos2(360.0, 200.0));
                self.egraph.connect(k, 0, read);
                self.egraph.view = Some(k);
                self.playing = false;
                t.node = k;
                next(&mut t, 1);
            }
            1 => {
                if let Some(v) = stable(self, &mut t, 400, 300, 16) {
                    t.before = Some(v);
                    t.at = self.timeline_pos;
                    self.seedvr_action(t.node, Action::Run);
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
            3 => {
                let (pw, ph) = t.plate.unwrap_or_else(|| fail("no plate format".into()));
                let fmt = self.node_resolution(t.node, &mut std::collections::HashSet::new());
                if fmt != Some((pw * 2, ph * 2)) {
                    fail(format!("the node's format is {fmt:?}, not twice the plate's {pw}x{ph}"));
                }
                // The plate's (400, 300) is the result's (800, 600).
                if let Some(v) = stable(self, &mut t, 800, 600, 32) {
                    let b = t.before.unwrap();
                    if (v - b).abs() > 0.01 + 0.02 * b.abs() {
                        fail(format!("the result reads {v:.4} where the plate read {b:.4}: the Rec.709 round trip moved it"));
                    }
                    let old = self.seedvr_dir(t.node);
                    if let Some(NodeKind::SeedVr(p)) = self.egraph.node_mut(t.node).map(|n| &mut n.kind) {
                        p.seed = 7;
                    }
                    if self.seedvr_dir(t.node) == old {
                        fail("a new seed did not change the result's directory".into());
                    }
                    let v = self.seedvr_view(t.node);
                    if !matches!(v.loaded, Some((_, false))) {
                        fail(format!("a new seed should show the result STALE, the view says {:?}", v.loaded));
                    }
                    println!(
                        "SEEDVR TEST PASS: a SeedVR2 node rendered its plate, ran the tool at twice its short \
                         side, adopted the result at twice the plate's format with the picture intact through the \
                         Rec.709 round trip, and a new seed is a new recipe, STALE until re-run"
                    );
                    std::process::exit(0);
                }
            }
            _ => {}
        }
        self.test_seedvr = Some(t);
    }
}
