//! Help render: a Render node's frames split between this wobble and paired
//! ones on the network (`peers`).
//!
//! What a helper will do is narrow on purpose (a deliberate choice, option 2):
//!
//! - **Only a setup it can already see.** The requester sends a PATH, not a
//!   setup: the saved `.wbl` must sit in a folder both machines share, and
//!   the requester refuses to start while its comp has unsaved changes (the
//!   helpers would render the file, not the screen).
//! - **Never a node that runs something else** — Blender, the process-backed
//!   readers (CorridorKey, Sammie, SeedVR2, Depth, LTX HDR) and GenAI: a
//!   setup naming one is refused by the helper, whoever sent it.
//! - **Only with its own switch on**: "let paired wobbles render here" in
//!   the network window, off by default, on top of sharing. A call from
//!   another machine is refused without it; this machine's own share runs
//!   regardless.
//!
//! A share is rendered by a HIDDEN worker — this binary as
//! `wobble --worker <port> <setup>` — so whatever the helper has open is
//! left alone. The helper points the worker's Render node at the share's
//! folder for just its frames (`node_patch`), drives it through its API
//! (`node_run`, `jobs_get`) and serves the finished files (`render_fetch`).
//! Footage comes from the paths the setup names: a Load that does not load
//! on the helper fails the share before a frame, saying which.
//!
//! The requester (`start`) cuts the range into contiguous shares, drives
//! each on a thread through `peers::Conn` (its own share through its
//! loopback API), writes the fetched frames where the Render node would,
//! under the same numbers, and takes back what a failed helper left.

use std::collections::HashMap;
use std::net::{Ipv4Addr, TcpListener};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use super::peers::{self, Conn};

/// Node kinds a helper refuses: each starts or talks to something outside
/// wobble (a process, Blender, a paid service).
pub const REFUSED: &[&str] = &["Blender", "CorridorKey", "Sammie", "SeedVr", "Depth", "LtxHdr", "GenAi"];

/// The kinds in a setup's graph that a helper refuses (nested graphs too).
pub fn refused_kinds(setup: &Value) -> Vec<String> {
    let mut out = Vec::new();
    fn walk(v: &Value, out: &mut Vec<String>) {
        match v {
            Value::Object(m) => {
                for (k, v) in m {
                    if k == "kind" {
                        if let Some(kind) = v.as_object().and_then(|o| o.keys().next()) {
                            if REFUSED.contains(&kind.as_str()) && !out.contains(kind) {
                                out.push(kind.clone());
                            }
                        }
                    }
                    walk(v, out);
                }
            }
            Value::Array(a) => a.iter().for_each(|v| walk(v, out)),
            _ => {}
        }
    }
    walk(&setup["graph"], &mut out);
    out
}

// ---------------------------------------------------------------- helper side

#[derive(Clone, PartialEq, Debug)]
enum State {
    Starting,
    Running,
    Done,
    Failed(String),
}

struct Job {
    dir: PathBuf,
    state: State,
    /// Frame files finished so far (names inside `dir/out`), in order.
    done: Vec<String>,
    total: usize,
    child: Option<Child>,
    stop: Arc<AtomicBool>,
}

fn jobs() -> std::sync::MutexGuard<'static, HashMap<String, Job>> {
    static JOBS: OnceLock<Mutex<HashMap<String, Job>>> = OnceLock::new();
    JOBS.get_or_init(Default::default).lock().unwrap_or_else(|e| e.into_inner())
}

fn farm_dir() -> PathBuf {
    crate::home_dir().unwrap_or_else(std::env::temp_dir).join(".wobble").join("farm")
}

fn free_port() -> u16 {
    TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).and_then(|l| l.local_addr()).map(|a| a.port()).unwrap_or(0)
}

fn random_id() -> String {
    use std::hash::{BuildHasher, Hasher};
    format!("{:016x}", std::collections::hash_map::RandomState::new().build_hasher().finish())
}

/// The file a Write writes for its `i`-th frame (file name only).
pub fn frame_name(pattern: &str, start: u32, i: u64) -> String {
    let p = engine::graph::WriteParams { path: pattern.to_string(), start, ..Default::default() };
    p.resolve_frame(i).file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default()
}

/// Is this setup one a helper renders? `Ok` with the parsed setup.
pub fn vet_setup(path: &Path) -> Result<Value, String> {
    if path.extension().and_then(|e| e.to_str()) != Some("wbl") || !path.is_absolute() {
        return Err(format!("{}: not a saved setup", path.display()));
    }
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("setup not found here ({}): {e} — it must be in a folder both machines share", path.display()))?;
    let v: Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    let bad = refused_kinds(&v);
    if !bad.is_empty() {
        return Err(format!("helpers do not render setups with {} nodes", bad.join(", ")));
    }
    Ok(v)
}

/// `render_take`: start a share of a saved setup's Render node. Returns
/// the job id at once; a worker process and a thread do the work.
pub fn take(p: &Value) -> Result<Value, String> {
    let setup = PathBuf::from(p["setup"].as_str().ok_or("'setup' must be the saved setup's path")?);
    let node = p["node"].as_u64().ok_or("'node' must be the Render node's id")?;
    let first = p["first"].as_u64().ok_or("'first' must be a frame")?;
    let last = p["last"].as_u64().ok_or("'last' must be a frame")?;
    let start = p["start"].as_u64().ok_or("'start' must be the first file number")? as u32;
    let pattern = p["pattern"].as_str().ok_or("'pattern' must be the Render node's path")?.to_string();
    if last < first {
        return Err("empty frame range".into());
    }
    vet_setup(&setup)?;
    let id = random_id();
    let dir = farm_dir().join(&id);
    let out = dir.join("out");
    std::fs::create_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;
    let base = Path::new(&pattern).file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or(pattern.clone());
    let out_pattern = out.join(&base).to_string_lossy().into_owned();
    let port = free_port();
    let child = Command::new(std::env::current_exe().map_err(|e| e.to_string())?)
        .env("WOBBLE_WORKER", "1")
        .arg("--worker")
        .arg(port.to_string())
        .arg(&setup)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .map_err(|e| format!("cannot start a worker: {e}"))?;
    let total = (last - first + 1) as usize;
    let stop = Arc::new(AtomicBool::new(false));
    jobs().insert(
        id.clone(),
        Job { dir, state: State::Starting, done: Vec::new(), total, child: Some(child), stop: stop.clone() },
    );
    let jid = id.clone();
    std::thread::Builder::new()
        .name("farm-share".into())
        .spawn(move || {
            let r = drive(&jid, &stop, port, node, first, last, start, &out_pattern, &base);
            let mut j = jobs();
            if let Some(job) = j.get_mut(&jid) {
                job.state = match r {
                    Ok(()) => State::Done,
                    Err(e) => State::Failed(e),
                };
                if let Some(mut c) = job.child.take() {
                    let _ = c.kill();
                    let _ = c.wait();
                }
            }
        })
        .map_err(|e| e.to_string())?;
    Ok(json!({ "job": id, "total": total }))
}

/// Drive the worker: wait for its API and graph, refuse missing footage,
/// point the Render node at the share, render, note each frame as it lands.
#[allow(clippy::too_many_arguments)]
fn drive(
    id: &str,
    stop: &AtomicBool,
    port: u16,
    node: u64,
    first: u64,
    last: u64,
    start: u32,
    out_pattern: &str,
    base: &str,
) -> Result<(), String> {
    let deadline = Instant::now() + Duration::from_secs(90);
    let mut c = loop {
        if stop.load(Ordering::Acquire) {
            return Err("cancelled".into());
        }
        match Conn::open(&[Ipv4Addr::LOCALHOST], port, None) {
            Ok(c) => break c,
            Err(_) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(300)),
            Err(e) => return Err(format!("the worker did not come up: {e}")),
        }
    };
    loop {
        let g = c.call("graph_get", json!({}))?;
        if g["nodes"].as_array().is_some_and(|n| n.iter().any(|n| n["id"].as_u64() == Some(node))) {
            break;
        }
        if Instant::now() > deadline {
            return Err(format!("the setup has no node {node} on this machine"));
        }
        std::thread::sleep(Duration::from_millis(300));
    }
    // Loads land over the first frames; then a Load that did not is a
    // missing file here, and a black frame would be worse than an error.
    std::thread::sleep(Duration::from_millis(1500));
    let g = c.call("graph_get", json!({}))?;
    let missing: Vec<String> = g["nodes"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|n| n["type"] == "Load" && n["error"].is_string())
        .map(|n| format!("{}: {}", n["name"].as_str().unwrap_or("?"), n["error"].as_str().unwrap_or("")))
        .collect();
    if !missing.is_empty() {
        return Err(format!("footage missing here — {}", missing.join("; ")));
    }
    c.call(
        "node_patch",
        json!({ "id": node, "kind": { "Write": {
            "path": out_pattern, "start": start, "range": [first, last], "range_mode": "Range",
        } } }),
    )?;
    if let Some(j) = jobs().get_mut(id) {
        j.state = State::Running;
    }
    c.call("node_run", json!({ "id": node, "action": "render" }))?;
    let mut seen = false;
    loop {
        if stop.load(Ordering::Acquire) {
            let _ = c.call("node_run", json!({ "id": node, "action": "cancel" }));
            return Err("cancelled".into());
        }
        let now = c.call("jobs_get", json!({}))?;
        let render = now["jobs"]
            .as_array()
            .into_iter()
            .flatten()
            .find(|j| j["node"].as_u64() == Some(node) && j["kind"] == "render")
            .cloned();
        // A frame is finished once the render has moved past it.
        let done = match &render {
            Some(r) => {
                seen = true;
                r["frame"].as_u64().unwrap_or(first).saturating_sub(first)
            }
            None if seen => last - first + 1,
            None => 0,
        };
        if let Some(j) = jobs().get_mut(id) {
            j.done = (0..done).map(|i| frame_name(base, start, i)).collect();
        }
        if render.is_none() {
            let n = c.call("node_get", json!({ "id": node }))?;
            if let Some(e) = n["error"].as_str() {
                return Err(e.to_string());
            }
            if seen {
                return Ok(());
            }
            if Instant::now() > deadline {
                return Err("the render did not start".into());
            }
        }
        std::thread::sleep(Duration::from_millis(400));
    }
}

/// `render_status`.
pub fn status(p: &Value) -> Result<Value, String> {
    let id = p["job"].as_str().ok_or("'job' must be a job id")?;
    let j = jobs();
    let job = j.get(id).ok_or("no such job here")?;
    let (state, error) = match &job.state {
        State::Starting => ("starting", None),
        State::Running => ("running", None),
        State::Done => ("done", None),
        State::Failed(e) => ("failed", Some(e.clone())),
    };
    Ok(json!({ "state": state, "error": error, "done": job.done, "total": job.total }))
}

/// `render_fetch`: one finished frame's bytes, base64.
pub fn fetch(p: &Value) -> Result<Value, String> {
    let id = p["job"].as_str().ok_or("'job' must be a job id")?;
    let file = p["file"].as_str().ok_or("'file' must be a frame file name")?;
    let path = {
        let j = jobs();
        let job = j.get(id).ok_or("no such job here")?;
        // Only a name this job finished: never a path someone typed.
        if !job.done.iter().any(|d| d == file) {
            return Err(format!("{file} is not a finished frame of this job"));
        }
        job.dir.join("out").join(file)
    };
    let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(json!({ "file": file, "data": api::b64(&bytes) }))
}

/// `render_release`: stop a share if it runs, and delete its folder.
pub fn release(p: &Value) -> Result<Value, String> {
    let id = p["job"].as_str().ok_or("'job' must be a job id")?;
    let Some(mut job) = jobs().remove(id) else { return Err("no such job here".into()) };
    job.stop.store(true, Ordering::Release);
    if let Some(mut c) = job.child.take() {
        let _ = c.kill();
        let _ = c.wait();
    }
    let _ = std::fs::remove_dir_all(&job.dir);
    Ok(json!({ "released": id }))
}

// ------------------------------------------------------------- requester side

#[derive(Clone)]
struct Share {
    who: String,
    first: u64,
    last: u64,
    got: usize,
    state: String,
}

#[derive(Default)]
struct Run {
    active: bool,
    shares: Vec<Share>,
    note: Option<String>,
    cancel: Option<Arc<AtomicBool>>,
}

fn run() -> std::sync::MutexGuard<'static, Run> {
    static RUN: OnceLock<Mutex<Run>> = OnceLock::new();
    RUN.get_or_init(Default::default).lock().unwrap_or_else(|e| e.into_inner())
}

pub fn busy() -> bool {
    run().active
}

pub fn set_note(s: String) {
    run().note = Some(s);
}

/// Where a share is rendered: this wobble (its loopback API, no code) or
/// a peer (its network port and code).
#[derive(Clone)]
pub struct Helper {
    pub name: String,
    pub addrs: Vec<Ipv4Addr>,
    pub port: u16,
    pub code: Option<String>,
}

/// Cut `first..=last` into `n` contiguous shares, as even as they come.
pub fn split(first: u64, last: u64, n: usize) -> Vec<(u64, u64)> {
    let total = last - first + 1;
    let n = (n as u64).clamp(1, total);
    let mut out = Vec::new();
    let mut at = first;
    for k in 0..n {
        let len = total / n + u64::from(k < total % n);
        out.push((at, at + len - 1));
        at += len;
    }
    out
}

/// What a shared render is: the saved setup, the Render node, its pattern,
/// the number its first frame is written as, and its range.
#[derive(Clone)]
pub struct Order {
    pub setup: PathBuf,
    pub node: u64,
    pub pattern: String,
    pub start: u32,
    pub first: u64,
    pub last: u64,
}

/// Start a shared render over `local` (this wobble) and `helpers`.
pub fn start(order: Order, local: Helper, helpers: Vec<Helper>) {
    let mut everyone = vec![local.clone()];
    everyone.extend(helpers);
    let shares = split(order.first, order.last, everyone.len());
    let cancel = Arc::new(AtomicBool::new(false));
    {
        let mut r = run();
        r.active = true;
        r.note = None;
        r.cancel = Some(cancel.clone());
        r.shares = shares
            .iter()
            .zip(&everyone)
            .map(|((a, b), h)| Share { who: h.name.clone(), first: *a, last: *b, got: 0, state: "starting".into() })
            .collect();
    }
    let started = Instant::now();
    std::thread::spawn(move || {
        let handles: Vec<_> = shares
            .iter()
            .copied()
            .zip(everyone)
            .enumerate()
            .map(|(k, ((a, b), h))| {
                let (order, cancel, local) = (order.clone(), cancel.clone(), local.clone());
                std::thread::spawn(move || {
                    let mut r = share(k, &h, &order, a, b, &cancel);
                    // A helper that failed hands what is left to this wobble.
                    if let Err((e, left)) = &r {
                        if !cancel.load(Ordering::Acquire) && h.code.is_some() && !left.is_empty() {
                            let (la, lb) = (left[0], *left.last().unwrap_or(&left[0]));
                            set_share(k, |s| s.state = format!("{e}; this wobble takes {la}–{lb}"));
                            r = share(k, &local, &order, la, lb, &cancel);
                        }
                    }
                    r.map_err(|(e, _)| e)
                })
            })
            .collect();
        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap_or(Err("crashed".into()))).collect();
        let mut r = run();
        r.active = false;
        let frames: usize = r.shares.iter().map(|s| s.got).sum();
        r.note = Some(if cancel.load(Ordering::Acquire) {
            format!("cancelled · {frames} frames written")
        } else if let Some(e) = results.iter().find_map(|r| r.as_ref().err()) {
            format!("finished with errors ({e}) · {frames} frames written")
        } else {
            format!("done · {frames} frames in {:.0} s", started.elapsed().as_secs_f32())
        });
    });
}

fn set_share(k: usize, f: impl FnOnce(&mut Share)) {
    if let Some(s) = run().shares.get_mut(k) {
        f(s);
    }
}

/// One share, start to finish. On failure: why, and the frames it did not
/// deliver (still in order).
fn share(k: usize, h: &Helper, o: &Order, first: u64, last: u64, cancel: &AtomicBool) -> Result<(), (String, Vec<u64>)> {
    let all: Vec<u64> = (first..=last).collect();
    let left = |got: usize| all[got.min(all.len())..].to_vec();
    let start = o.start + (first - o.first) as u32;
    let mut c = Conn::open(&h.addrs, h.port, h.code.as_deref()).map_err(|e| (e, left(0)))?;
    let r = c
        .call(
            "render_take",
            json!({
                "setup": o.setup.to_string_lossy(), "node": o.node, "first": first, "last": last,
                "start": start, "pattern": o.pattern,
            }),
        )
        .map_err(|e| (e, left(0)))?;
    let job = r["job"].as_str().unwrap_or_default().to_string();
    let dir = engine::paths::expand_home(&o.pattern).parent().map(Path::to_path_buf).unwrap_or_default();
    let _ = std::fs::create_dir_all(&dir);
    let mut got: Vec<String> = Vec::new();
    let outcome = loop {
        if cancel.load(Ordering::Acquire) {
            break Err(("cancelled".to_string(), left(got.len())));
        }
        let st = match c.call("render_status", json!({ "job": job })) {
            Ok(s) => s,
            Err(e) => break Err((e, left(got.len()))),
        };
        let state = st["state"].as_str().unwrap_or("?").to_string();
        let mut broke = None;
        for f in st["done"].as_array().into_iter().flatten().filter_map(|f| f.as_str()) {
            if got.iter().any(|g| g == f) {
                continue;
            }
            let bytes = c
                .call("render_fetch", json!({ "job": job, "file": f }))
                .and_then(|v| api::unb64(v["data"].as_str().unwrap_or("")).map_err(|e| e.to_string()));
            match bytes.and_then(|b| std::fs::write(dir.join(f), b).map_err(|e| e.to_string())) {
                Ok(()) => got.push(f.to_string()),
                Err(e) => {
                    broke = Some(e);
                    break;
                }
            }
        }
        if let Some(e) = broke {
            break Err((e, left(got.len())));
        }
        let n = got.len();
        set_share(k, |s| {
            s.got = n;
            s.state = state.clone();
        });
        match state.as_str() {
            "done" if got.len() >= all.len() => break Ok(()),
            "failed" => break Err((st["error"].as_str().unwrap_or("failed").to_string(), left(got.len()))),
            _ => std::thread::sleep(Duration::from_millis(500)),
        }
    };
    let _ = c.call("render_release", json!({ "job": job }));
    match &outcome {
        Ok(()) => set_share(k, |s| s.state = "done".into()),
        Err((e, _)) => set_share(k, |s| s.state = format!("failed: {e}")),
    }
    outcome
}

/// `network_render_status`.
pub fn snapshot() -> Value {
    let r = run();
    json!({
        "running": r.active,
        "note": r.note,
        "shares": r.shares.iter().map(|s| json!({
            "who": s.who, "first": s.first, "last": s.last, "got": s.got, "state": s.state,
        })).collect::<Vec<_>>(),
    })
}

/// Progress, in the network window.
pub fn progress_ui(ui: &mut egui::Ui) {
    let (active, shares, note) = {
        let r = run();
        (r.active, r.shares.clone(), r.note.clone())
    };
    for s in &shares {
        let n = (s.last - s.first + 1) as usize;
        ui.horizontal(|ui| {
            ui.label(format!("{} · {}–{}", s.who, s.first, s.last));
            ui.add(egui::ProgressBar::new(s.got as f32 / n.max(1) as f32).desired_width(110.0).text(format!("{}/{n}", s.got)));
            ui.weak(&s.state);
        });
    }
    if active && ui.button("cancel render").clicked() {
        if let Some(c) = &run().cancel {
            c.store(true, Ordering::Release);
        }
    }
    if let Some(n) = note {
        ui.weak(n);
    }
}

/// This wobble's own name in the progress list.
pub fn local_name() -> String {
    format!("{} (this one)", peers::name())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shares_cover_the_range_once_in_order() {
        assert_eq!(split(0, 9, 3), vec![(0, 3), (4, 6), (7, 9)]);
        assert_eq!(split(1001, 1001, 4), vec![(1001, 1001)], "never more shares than frames");
        let s = split(10, 109, 7);
        assert_eq!(s.first().unwrap().0, 10);
        assert_eq!(s.last().unwrap().1, 109);
        assert!(s.windows(2).all(|w| w[0].1 + 1 == w[1].0));
    }

    #[test]
    fn frame_names_follow_the_write_numbering() {
        assert_eq!(frame_name("comp.####.exr", 1001, 0), "comp.1001.exr");
        assert_eq!(frame_name("/x/y/comp.####.exr", 1001, 5), "comp.1006.exr");
    }

    #[test]
    fn a_helper_refuses_nodes_that_run_something_else() {
        let ok = json!({ "graph": { "nodes": { "0": { "kind": { "Grade": {} } } } } });
        assert!(refused_kinds(&ok).is_empty());
        let bad = json!({ "graph": { "nodes": {
            "0": { "kind": { "Blender": {} } },
            "1": { "kind": { "Noise": { "graph": { "x": { "kind": { "GenAi": {} } } } } } },
        } } });
        assert_eq!(refused_kinds(&bad), vec!["Blender".to_string(), "GenAi".to_string()]);
    }

    #[test]
    fn only_a_saved_setup_is_vetted() {
        assert!(vet_setup(Path::new("relative.wbl")).is_err());
        // Absolute on every OS ("/nowhere" is not, on Windows).
        let gone = std::env::temp_dir().join("wobble-nowhere").join("all.wbl");
        assert!(vet_setup(&gone).unwrap_err().contains("not found here"));
        assert!(vet_setup(Path::new("/etc/hosts")).unwrap_err().contains("not a saved setup"));
    }
}
