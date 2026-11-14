//! Wobble clips (EXPERIMENTAL, the `wobble` feature + sequence >
//! experimental > Wobble clips). Design: docs/wobble-clips.md.
//!
//! A clip opened in Wobble (cmd+shift+W) gets a folder of its own beside
//! the project (`<project>.wobble/<uid>/`): `vNNN/setup.wbl` +
//! `vNNN/render/out.NNNNNN.exr` per version, and `work.wbl`, the copy the
//! Wobble app edits. Nothing of the source is copied: the comp's locked
//! Load (SCRUB SOURCE) reads the ORIGINAL file — a movie through Wobble's
//! FFmpeg reader (so Wobble must be built with `--features app/movie`), an
//! image sequence by its pattern — placed so comp frame k is source frame k
//! (Scrub's numbering), read as the clip's input colourspace in Scrub, as
//! Scrub shows them. Its locked Render (SCRUB OUT) renders the used source
//! range plus handles to EXR in the project's working space, named by
//! source frame.
//!
//! Wobble is driven over its scripting API, never linked: a hidden
//! `wobble --worker PORT [setup]` builds v001, re-asserts both nodes on
//! every later version (source path, placement and colour; the output's
//! path, range and start), renders, saves and exits. Editing is the real
//! Wobble app on `work.wbl` with `--api`; each save there becomes a new
//! version that is rendered and then plays. A trim that needs frames past
//! the range widens it (a new version of the same setup, rendered wider).
//! The clip's `media` is the active version's render (or the source until
//! one exists), so playback, speed, reframe, dissolves and renders treat it
//! as footage.

use crate::editor::Editor;
use crate::timelineui::Selection;
use api::client::Client;
use model::{Media, WobbleRef};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{channel, Receiver};
use std::time::{Duration, Instant, SystemTime};

/// Handles rendered each side of a clip's used source, in seconds.
pub const HANDLE_SECS: f64 = 1.0;
const SOURCE_NODE: &str = "SCRUB SOURCE";
const OUT_NODE: &str = "SCRUB OUT";

/// A background job's news.
enum Msg {
    Progress(String),
    /// A version rendered: (uid, version, its first frame's path, the first
    /// source frame, frames).
    Rendered(String, u32, PathBuf, i64, i64),
    Failed(String, String),
}

/// How to build Wobble so it reads movies (said wherever one cannot).
pub const MOVIE_BUILD: &str = "cd ~/src/devl/wobble && cargo build --release -p app --features app/movie";

/// What the comp's source node reads.
#[derive(Clone)]
struct SourceSpec {
    /// A movie, or an image sequence's pattern (`dir/name.####.exr`).
    path: String,
    movie: bool,
    /// The clip's input colourspace in Scrub (the Load reads it as that),
    /// and the project's working space (the render comes back in it, which
    /// is how Scrub reads an EXR): the round trip is the identity on an
    /// empty comp whatever the two are.
    input: colour::color::Colorspace,
    working: colour::color::Colorspace,
}

fn source_spec(media: &Path, c: &model::ColourSettings) -> SourceSpec {
    let image = media::sequence::is_image(media);
    let path = match (image, media::sequence::Pattern::of(media)) {
        (true, Some((p, _))) => p.dir.join(format!("{}{}{}", p.prefix, "#".repeat(p.pad.max(1)), p.suffix)).to_string_lossy().into_owned(),
        _ => media.to_string_lossy().into_owned(),
    };
    SourceSpec { path, movie: !image, input: c.input_of(media), working: c.working }
}

/// The Wobble side of the editor.
#[derive(Default)]
pub struct Wob {
    jobs: Vec<Receiver<Msg>>,
    /// What runs now, per uid (for the HUD and the timeline's badge).
    pub busy: Vec<(String, String)>,
    /// Open Wobble app sessions: uid, its working copy, the copy's last
    /// modification seen, the process.
    sessions: Vec<(String, PathBuf, Option<SystemTime>, Option<Child>)>,
    checked: Option<Instant>,
    /// Do not start the Wobble app (headless tests).
    pub no_launch: bool,
    /// Where comps go instead of beside the project (headless tests).
    pub root: Option<PathBuf>,
}

/// The Wobble binary, first found of: $EDIFY_WOBBLE (or $EDIFY_WOBBLE_BIN),
/// a checkout beside Scrub's (`<scrub>/../wobble/target/release/wobble`,
/// from where this binary runs), ~/src/devl/wobble's release build, an app
/// bundle in /Applications or ~/Applications, `wobble` on PATH.
pub fn wobble_bin() -> Option<PathBuf> {
    for var in ["WOBBLE", "WOBBLE_BIN"] {
        if let Some(p) = media::sys::var_os(var).map(PathBuf::from).filter(|p| p.is_file()) {
            return Some(p);
        }
    }
    let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_default();
    // <scrub>/target/release/scrub -> <scrub>/../wobble/target/release/wobble
    let sibling = std::env::current_exe().ok().and_then(|e| e.ancestors().nth(3).map(|root| root.join("../wobble/target/release/wobble")));
    let cands = [
        sibling.unwrap_or_default(),
        home.join("src/devl/wobble/target/release/wobble"),
        PathBuf::from("/Applications/Wobble.app/Contents/MacOS/wobble"),
        home.join("Applications/Wobble.app/Contents/MacOS/wobble"),
    ];
    if let Some(p) = cands.into_iter().find(|p| p.is_file()) {
        return Some(p);
    }
    std::env::var("PATH").ok()?.split(':').map(|d| Path::new(d).join("wobble")).find(|p| p.is_file())
}

fn free_port() -> u16 {
    std::net::TcpListener::bind(("127.0.0.1", 0)).ok().and_then(|l| l.local_addr().ok()).map_or(7990, |a| a.port())
}

fn version_dir(root: &Path, v: u32) -> PathBuf {
    root.join(format!("v{v:03}"))
}

fn render_pattern(root: &Path, v: u32) -> PathBuf {
    version_dir(root, v).join("render/out.######.exr")
}

/// A hidden Wobble worker on a free port, and a client once it answers.
struct Worker {
    child: Child,
    c: Client,
}

impl Worker {
    fn start(bin: &Path, setup: Option<&Path>) -> Result<Worker, String> {
        let port = free_port();
        let mut cmd = Command::new(bin);
        cmd.env("WOBBLE_WORKER", "1").arg("--worker").arg(port.to_string());
        if let Some(s) = setup {
            cmd.arg(s);
        }
        let child = cmd.stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null()).spawn().map_err(|e| format!("cannot start Wobble ({}): {e}", bin.display()))?;
        let t0 = Instant::now();
        let mut w = loop {
            match Client::connect(port) {
                Ok(c) => break Worker { child, c },
                Err(_) if t0.elapsed() < Duration::from_secs(60) => std::thread::sleep(Duration::from_millis(200)),
                Err(e) => {
                    let mut child = child;
                    let _ = child.kill();
                    return Err(format!("Wobble did not come up: {e}"));
                }
            }
        };
        // The setup lands over the first passes.
        let t0 = Instant::now();
        while t0.elapsed() < Duration::from_secs(20) {
            if w.call("graph_get", json!({})).is_ok_and(|g| g["nodes"].as_array().is_some_and(|n| !n.is_empty() || setup.is_none())) {
                break;
            }
            std::thread::sleep(Duration::from_millis(200));
        }
        Ok(w)
    }

    fn call(&mut self, m: &str, p: Value) -> Result<Value, String> {
        self.c.call(m, p).map_err(|e| format!("Wobble {m}: {}", e.message))
    }

    /// The node named `name` (of `kind`), if the graph has one.
    fn node(&mut self, name: &str, kind: &str) -> Result<Option<u64>, String> {
        let g = self.call("graph_get", json!({}))?;
        Ok(g["nodes"].as_array().into_iter().flatten().find(|n| n["name"] == name && n["type"] == kind).and_then(|n| n["id"].as_u64()))
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Build (or re-assert) the locked nodes, render `v`, save its setup.
/// `setup`: the version's setup to start from (None: a fresh v001).
/// Build (or re-assert) the locked nodes, render `v`, save its setup.
/// `setup`: the version's setup to start from (None: a fresh v001).
#[allow(clippy::too_many_arguments)]
fn render_version(bin: &Path, root: &Path, v: u32, setup: Option<&Path>, src: &SourceSpec, first: i64, len: i64, tx: &std::sync::mpsc::Sender<Msg>, uid: &str) -> Result<PathBuf, String> {
    let _ = tx.send(Msg::Progress(format!("{uid}: starting Wobble for v{v:03}")));
    let mut w = Worker::start(bin, setup)?;
    let (load, out) = match setup {
        None => {
            // A fresh comp: just our two nodes.
            let g = w.call("graph_get", json!({}))?;
            for id in g["nodes"].as_array().into_iter().flatten().filter_map(|n| n["id"].as_u64()) {
                w.call("node_delete", json!({ "id": id }))?;
            }
            let l = w.call("node_create", json!({ "type": "Load", "path": src.path, "name": SOURCE_NODE, "x": 0.0, "y": 0.0 }))?["id"].as_u64().ok_or("no Load id")?;
            let o = w.call("node_create", json!({ "type": "Render", "inputs": [l], "name": OUT_NODE, "x": 0.0, "y": 240.0 }))?["id"].as_u64().ok_or("no Render id")?;
            let _ = w.call("view_set", json!({ "id": o }));
            (l, o)
        }
        Some(_) => {
            let l = w.node(SOURCE_NODE, "Load")?.ok_or_else(|| format!("the setup has no Load named {SOURCE_NODE}: keep Scrub's source node"))?;
            let o = w.node(OUT_NODE, "Render")?.ok_or_else(|| format!("the setup has no Render named {OUT_NODE}: keep Scrub's output node"))?;
            (l, o)
        }
    };
    // The source is Scrub's: the original file, placed so comp frame k is
    // source frame k, in the colour Scrub shows it in.
    // (Scrub's colour types are wobble's, copied: their JSON is wobble's.)
    let input = serde_json::to_value(colour::color::InputSpace::Explicit(src.input)).map_err(|e| e.to_string())?;
    w.call("node_patch", json!({ "id": load, "kind": { "Read": { "path": src.path, "start": 0, "input": input } } }))?;
    // Does it read? (A movie needs a Wobble built with FFmpeg.)
    let t0 = Instant::now();
    loop {
        let n = w.call("node_get", json!({ "id": load }))?;
        if let Some(e) = n["error"].as_str() {
            return Err(if src.movie {
                format!("this Wobble cannot read movies ({e}); build it with: {MOVIE_BUILD}")
            } else {
                format!("Wobble cannot read the source: {e}")
            });
        }
        let k = &n["kind"]["Read"];
        if k["last"].as_i64().is_some_and(|l| l >= first + len - 1) || t0.elapsed() > Duration::from_secs(10) {
            if k["last"].as_i64().is_some_and(|l| l < first + len - 1) {
                return Err(if src.movie && k["last"].as_i64().unwrap_or(0) <= 0 {
                    format!("this Wobble did not read the movie (no frames); build it with: {MOVIE_BUILD}")
                } else {
                    format!("Wobble sees frames up to {} of the source, Scrub needs {}", k["last"], first + len - 1)
                });
            }
            break;
        }
        std::thread::sleep(Duration::from_millis(200));
    }
    let pattern = render_pattern(root, v);
    std::fs::create_dir_all(pattern.parent().expect("a folder")).map_err(|e| e.to_string())?;
    let last = first + len - 1;
    // Out in Scrub's working space, whatever the comp works in: that is
    // how Scrub reads an EXR.
    let output = serde_json::to_value(src.working).map_err(|e| e.to_string())?;
    w.call(
        "node_patch",
        json!({ "id": out, "kind": { "Write": {
            "path": pattern.to_string_lossy(), "start": first, "range": [first, last], "range_mode": "Range",
            "output": output, "output_default": false,
        } } }),
    )?;
    w.call("setup_save", json!({ "path": version_dir(root, v).join("setup.wbl") }))?;
    let _ = tx.send(Msg::Progress(format!("{uid}: rendering v{v:03} ({len} frames)")));
    w.call("node_run", json!({ "id": out, "action": "render" }))?;
    let t0 = Instant::now();
    loop {
        std::thread::sleep(Duration::from_millis(300));
        let j = w.call("jobs_get", json!({}))?;
        if j["busy"] == false {
            if let Some(e) = j["errors"].as_object().and_then(|o| o.values().next()) {
                return Err(format!("Wobble: {e}"));
            }
            break;
        }
        if let Some(f) = j["jobs"].as_array().and_then(|a| a.first()).and_then(|x| x["frame"].as_i64()) {
            let _ = tx.send(Msg::Progress(format!("{uid}: rendering v{v:03}: frame {} of {len}", f - first + 1)));
        }
        if t0.elapsed() > Duration::from_secs(3600) {
            return Err("the render took over an hour: stopped".into());
        }
    }
    let first_file = version_dir(root, v).join(format!("render/out.{first:06}.exr"));
    if !first_file.is_file() {
        let e = w.call("node_get", json!({ "id": out })).ok().and_then(|n| n["error"].as_str().map(String::from)).unwrap_or_else(|| "no frames came out".into());
        return Err(format!("Wobble rendered nothing: {e}"));
    }
    Ok(first_file)
}

impl Editor {
    /// The folder Wobble clips keep their comps in.
    pub(crate) fn wobble_root(&self) -> PathBuf {
        if let Some(r) = &self.wob.root {
            return r.clone();
        }
        match &self.project.path {
            // (`x.scrub.wobble`; a project saved as .scrub from an old
            // `x.edify` keeps using `x.edify.wobble` when that is there.)
            Some(p) => {
                let new = PathBuf::from(format!("{}.wobble", p.display()));
                let old = PathBuf::from(format!("{}.wobble", p.with_extension("edify").display()));
                if !new.exists() && old.is_dir() { old } else { new }
            }
            None => media::sys::support_dir().join("wobble"),
        }
    }

    pub(crate) fn wobble_enabled(&self) -> bool {
        self.panes.experimental_wobble
    }

    /// cmd+shift+W: the selected clip in Wobble. First time: make and
    /// render v001 from the original file; then (and every other time) open the
    /// Wobble app on its working copy, watching it for saves.
    pub(crate) fn wobble_open(&mut self) {
        if !self.wobble_enabled() {
            self.say("Wobble clips are experimental: sequence > experimental > Wobble clips");
            return;
        }
        let Some(bin) = wobble_bin() else {
            self.say(format!("no Wobble found: build it ({MOVIE_BUILD}) or set EDIFY_WOBBLE to the wobble binary"));
            return;
        };
        let Some((t, i)) = self.clip_target() else {
            self.say("select a clip to open in Wobble");
            return;
        };
        if !model::VIDEO_TRACKS.contains(&t) || self.rec.seq.tracks[t].clips[i].text.is_some() || self.rec.seq.media[self.rec.seq.tracks[t].clips[i].media].generated {
            self.say("Wobble takes a picture clip (V1-V3, not text)");
            return;
        }
        self.rec.sel = Selection::Clip { track: t, start: self.rec.seq.tracks[t].clips[i].start };
        if let Some(w) = self.rec.seq.tracks[t].clips[i].wobble.clone() {
            if w.versions.is_empty() {
                self.say("this clip's first Wobble version is still being made");
                return;
            }
            // A trim past the rendered range: render the active setup wider
            // as a new version first.
            let (f, l) = self.wobble_need(&w.uid);
            if f < w.first || f + l > w.first + w.len {
                let (nf, nl) = (f.min(w.first), (f + l).max(w.first + w.len) - f.min(w.first));
                self.wobble_set_range(&w.uid, nf, nl);
                let root = self.wobble_root().join(&w.uid);
                let from = version_dir(&root, w.active.max(1)).join("setup.wbl");
                self.wobble_spawn(&w.uid, Some(from), nf, nl, "the clip needs more frames: rendering wider");
            }
            self.wobble_launch(&bin, &w);
            return;
        }
        // The used source range plus handles, clamped to the media.
        let c = self.rec.seq.tracks[t].clips[i].clone();
        let uid = format!("{:08x}{:04x}", SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).map_or(0, |d| d.as_secs()) as u32, std::process::id() as u16 ^ (c.link as u16));
        let wref = WobbleRef { uid: uid.clone(), versions: Vec::new(), active: 1, source: c.media, first: 0, len: 0, renders: Vec::new() };
        for (gt, gi) in self.rec.seq.group(t, i) {
            if model::VIDEO_TRACKS.contains(&gt) {
                self.rec.seq.tracks[gt].clips[gi].wobble = Some(wref.clone());
            }
        }
        let (first, len) = self.wobble_need(&uid);
        self.wobble_set_range(&uid, first, len);
        self.changed(false);
        self.wobble_spawn(&uid, None, first, len, "making v001");
    }

    /// The source frames every clip of a comp uses, plus handles, clamped to
    /// the media: (first, len).
    fn wobble_need(&self, uid: &str) -> (i64, i64) {
        let seq = &self.rec.seq;
        let (mut lo, mut hi, mut src) = (i64::MAX, i64::MIN, None);
        for c in seq.tracks.iter().flat_map(|t| &t.clips) {
            let Some(w) = c.wobble.as_ref().filter(|w| w.uid == uid) else { continue };
            src = Some(w.source);
            // The clip in source frames, whatever plays now.
            let off = w.renders.iter().find(|r| r.media == c.media && c.media != w.source).map_or(0, |r| r.first);
            let probe = model::Clip { media: w.source, src_in: c.src_in + off, ..c.clone() };
            let (a, b) = (seq.source_frame(&probe, c.start), seq.source_frame(&probe, c.end() - 1));
            lo = lo.min(a.min(b));
            hi = hi.max(a.max(b));
        }
        let Some(m) = src.map(|s| &seq.media[s]) else { return (0, 1) };
        let h = (HANDLE_SECS * m.fps.as_f64()).round() as i64;
        let (first, last) = ((lo - h).max(0), (hi + h).min(m.frames - 1));
        (first, (last - first + 1).max(1))
    }

    fn wobble_set_range(&mut self, uid: &str, first: i64, len: i64) {
        for t in &mut self.rec.seq.tracks {
            for c in &mut t.clips {
                if let Some(w) = c.wobble.as_mut().filter(|w| w.uid == uid) {
                    (w.first, w.len) = (first, len);
                }
            }
        }
    }

    /// Make and render the next version (v001 when `from` is None) off the
    /// UI thread.
    fn wobble_spawn(&mut self, uid: &str, from: Option<PathBuf>, first: i64, len: i64, why: &str) {
        let Some(bin) = wobble_bin() else { return };
        let Some(w) = self.rec.seq.tracks.iter().flat_map(|t| &t.clips).filter_map(|c| c.wobble.as_ref()).find(|w| w.uid == uid).cloned() else { return };
        let src = source_spec(&self.rec.seq.media[w.source].path, &self.rec.seq.colour);
        let root = self.wobble_root().join(uid);
        let v = if from.is_none() { 1 } else { w.versions.iter().max().copied().unwrap_or(0).max(self.wob_pending_max(uid)) + 1 };
        let setup = match &from {
            Some(f) => {
                let dir = version_dir(&root, v);
                if let Err(e) = std::fs::create_dir_all(&dir).and_then(|_| std::fs::copy(f, dir.join("setup.wbl")).map(|_| ())) {
                    self.say(format!("Wobble: keeping v{v:03}: {e}"));
                    return;
                }
                Some(dir.join("setup.wbl"))
            }
            None => None,
        };
        let (tx, rx) = channel();
        self.wob.jobs.push(rx);
        self.wob.busy.push((uid.to_string(), format!("v{v:03}: {why}")));
        let uid2 = uid.to_string();
        std::thread::Builder::new()
            .name("wobble-render".into())
            .spawn(move || match render_version(&bin, &root, v, setup.as_deref(), &src, first, len, &tx, &uid2) {
                Ok(p) => {
                    let _ = tx.send(Msg::Rendered(uid2, v, p, first, len));
                }
                Err(e) => {
                    let _ = tx.send(Msg::Failed(uid2, e));
                }
            })
            .expect("spawn");
        self.say(format!("Wobble v{v:03}: {why}"));
    }

    /// Open the Wobble app on a clip's working copy (from its active
    /// version) and watch it for saves.
    fn wobble_launch(&mut self, bin: &Path, w: &WobbleRef) {
        let root = self.wobble_root().join(&w.uid);
        let work = root.join("work.wbl");
        if !work.is_file() {
            if let Err(e) = std::fs::copy(version_dir(&root, w.active.max(1)).join("setup.wbl"), &work) {
                self.say(format!("Wobble: no setup to open: {e}"));
                return;
            }
        }
        let seen = std::fs::metadata(&work).and_then(|m| m.modified()).ok();
        let child = if self.wob.no_launch {
            None
        } else {
            match Command::new(bin).arg("--api").arg(free_port().to_string()).arg(&work).stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null()).spawn() {
                Ok(c) => Some(c),
                Err(e) => {
                    self.say(format!("cannot start Wobble: {e}"));
                    return;
                }
            }
        };
        self.wob.sessions.retain(|s| s.0 != w.uid);
        self.wob.sessions.push((w.uid.clone(), work, seen, child));
        self.say(format!("Wobble open on v{:03}: each save there becomes a new version", w.active));
    }

    /// Every UI pass: news from the jobs, and saves in open Wobble sessions
    /// (checked twice a second).
    pub(crate) fn wobble_tick(&mut self) {
        let mut news = Vec::new();
        self.wob.jobs.retain(|rx| loop {
            match rx.try_recv() {
                Ok(m) => news.push(m),
                Err(std::sync::mpsc::TryRecvError::Empty) => break true,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => break false,
            }
        });
        for m in news {
            match m {
                Msg::Progress(s) => {
                    let (uid, what) = s.split_once(": ").map(|(a, b)| (a.to_string(), b.to_string())).unwrap_or_default();
                    self.wob.busy.retain(|b| b.0 != uid);
                    self.wob.busy.push((uid, what));
                }
                Msg::Failed(uid, e) => {
                    self.wob.busy.retain(|b| b.0 != uid);
                    self.say(format!("Wobble: {e}"));
                }
                Msg::Rendered(uid, v, file, first, len) => {
                    self.wob.busy.retain(|b| b.0 != uid);
                    self.wobble_landed(&uid, v, &file, first, len);
                }
            }
        }
        if self.wob.checked.is_some_and(|t| t.elapsed() < Duration::from_millis(500)) || self.wob.sessions.is_empty() {
            return;
        }
        self.wob.checked = Some(Instant::now());
        let mut saved = Vec::new();
        for s in &mut self.wob.sessions {
            let now = std::fs::metadata(&s.1).and_then(|m| m.modified()).ok();
            if now.is_some() && now != s.2 {
                s.2 = now;
                saved.push(s.0.clone());
            }
            if let Some(c) = &mut s.3 {
                if c.try_wait().ok().flatten().is_some() {
                    s.3 = None;
                }
            }
        }
        for uid in saved {
            self.wobble_new_version(&uid);
        }
    }

    /// A save in Wobble: the working copy becomes the next version, rendered.
    pub(crate) fn wobble_new_version(&mut self, uid: &str) {
        let Some(w) = self.rec.seq.tracks.iter().flat_map(|t| &t.clips).filter_map(|c| c.wobble.as_ref()).find(|w| w.uid == uid).cloned() else { return };
        let work = self.wobble_root().join(uid).join("work.wbl");
        self.wobble_spawn(uid, Some(work), w.first, w.len, "saved in Wobble, rendering");
    }

    fn wob_pending_max(&self, uid: &str) -> u32 {
        let root = self.wobble_root().join(uid);
        (1..1000).take_while(|v| version_dir(&root, *v).is_dir()).last().unwrap_or(0)
    }

    /// A version's render is on disk: its media joins the sequence and the
    /// clips of that comp play it.
    fn wobble_landed(&mut self, uid: &str, v: u32, file: &Path, first: i64, len: i64) {
        let Some(w) = self.rec.seq.tracks.iter().flat_map(|t| &t.clips).filter_map(|c| c.wobble.as_ref()).find(|w| w.uid == uid).cloned() else { return };
        let src = self.rec.seq.media[w.source].clone();
        let mi = self.rec.seq.add_media(Media { path: file.to_path_buf(), name: format!("{} wobble v{v:03}", src.name), frames: len, fps: src.fps, has_audio: false, width: src.width, height: src.height, generated: false });
        let places: Vec<(usize, usize)> = self.rec.seq.tracks.iter().enumerate().flat_map(|(t, tr)| tr.clips.iter().enumerate().filter(|(_, c)| c.wobble.as_ref().is_some_and(|x| x.uid == uid)).map(move |(i, _)| (t, i))).collect();
        for (t, i) in places {
            if let Some(x) = self.rec.seq.tracks[t].clips[i].wobble.as_mut() {
                if !x.versions.contains(&v) {
                    x.versions.push(v);
                }
                x.renders.retain(|r| r.version != v);
                x.renders.push(model::WobbleRender { version: v, media: mi, first });
            }
            self.rec.seq.wobble_activate(t, i, v);
        }
        self.changed(false);
        self.say(format!("Wobble v{v:03} rendered: playing it"));
    }

    /// The versions window: the selected Wobble clip's versions, which
    /// plays, and opening it again.
    pub(crate) fn wobble_window(&mut self, ctx: &egui::Context) {
        let Some((t, i)) = self.clip_target().filter(|&(t, i)| self.rec.seq.tracks[t].clips[i].wobble.is_some()) else { return };
        if !self.wobble_enabled() {
            return;
        }
        let w = self.rec.seq.tracks[t].clips[i].wobble.clone().expect("checked");
        let busy = self.wob.busy.iter().find(|b| b.0 == w.uid).map(|b| b.1.clone());
        let mut pick = None;
        let mut open = false;
        egui::Window::new("wobble clip").collapsible(true).resizable(false).default_pos(egui::pos2(ctx.content_rect().max.x - 260.0, 60.0)).show(ctx, |ui| {
            ui.label(format!("source frames {}-{} (handles included)", w.first, w.first + w.len - 1));
            let mut vs = w.versions.clone();
            vs.sort_unstable();
            for v in vs {
                let rendered = w.render_of(v).is_some();
                let label = format!("v{v:03}{}", if rendered { "" } else { "  (not rendered)" });
                if ui.radio(w.active == v, label).clicked() {
                    pick = Some(v);
                }
            }
            if let Some(b) = &busy {
                ui.colored_label(imfram::theme::warn(), b);
            }
            open = ui.button("open in Wobble  (cmd shift W)").clicked();
            ui.weak("each save in Wobble becomes a new version");
        });
        if let Some(v) = pick {
            for (gt, gi) in self.rec.seq.group(t, i) {
                self.rec.seq.wobble_activate(gt, gi, v);
            }
            self.changed(false);
        }
        if open {
            self.wobble_open();
        }
    }

    /// What is running for a uid (the timeline badge, tests).
    pub fn wobble_busy(&self) -> bool {
        !self.wob.busy.is_empty()
    }
}

/// Test hooks.
impl Editor {
    pub fn wobble_test_setup(&mut self) {
        self.panes.experimental_wobble = true;
        self.wob.no_launch = true;
        self.wob.root = Some(std::env::temp_dir().join(format!("scrub-wobble-test-{}", std::process::id())));
    }

    pub fn select_clip_at(&mut self, track: usize, start: i64) {
        self.rec.sel = Selection::Clip { track, start };
    }

    /// (uid, rendered versions, busy, active) of the clip at (track, start).
    pub fn wobble_state(&self, track: usize, start: i64) -> Option<(String, Vec<u32>, bool, u32)> {
        let c = self.rec.seq.tracks[track].clips.iter().find(|c| c.start == start)?;
        let w = c.wobble.as_ref()?;
        let mut vs: Vec<u32> = w.renders.iter().map(|r| r.version).collect();
        vs.sort_unstable();
        Some((w.uid.clone(), vs, self.wob.busy.iter().any(|b| b.0 == w.uid), w.active))
    }

    pub fn wobble_pick(&mut self, track: usize, start: i64, v: u32) {
        if let Some(i) = self.rec.seq.tracks[track].clips.iter().position(|c| c.start == start) {
            for (gt, gi) in self.rec.seq.group(track, i) {
                self.rec.seq.wobble_activate(gt, gi, v);
            }
            self.changed(false);
        }
    }

    /// The record picture at `t` as the renderer composes it (PNG).
    pub fn capture_record(&self, t: i64) -> Vec<u8> {
        let read = self.read_paths(&self.rec.seq);
        crate::render::still(&self.rec.seq, &read, t, 256).map(|r| r.0).unwrap_or_default()
    }

    /// Render the sequence now (YouTube 1080), on this thread: frames made.
    pub fn render_now(&self, out: &Path) -> Result<u64, String> {
        let preset = crate::render::preset("youtube-1080").ok_or("no preset")?;
        let job = self.render_job(preset, crate::render::Fit::Fit, false, out.to_path_buf(), None);
        crate::render::run(&job, &crate::render::Progress::default()).map(|d| d.frames).map_err(|e| format!("{e:#}"))
    }
}
