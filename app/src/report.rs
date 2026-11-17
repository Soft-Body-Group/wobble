#![cfg_attr(target_arch = "wasm32", allow(dead_code))] // no uploads in the web build
//! Alpha plumbing: crash reports, unclean-exit detection, session logs,
//! autosave + crash recovery, bug / feature reports, session stats and the
//! new-build check — all talking to the wobble feedback Worker
//! (tools/feedback-worker, Cloudflare D1 + R2 behind it).
//!
//! Everything lives under `~/.wobble` (or `$WOBBLE_REPORT_DIR`):
//!
//! - `crash/` is the OUTBOX. Nothing uploads from a dying process: the
//!   panic hook only WRITES a report there, and the next launch sends it on
//!   a background thread (a launch with no network leaves it for the one
//!   after). Session records ride the same outbox.
//! - `sessions/<sid>.lock` is held with an exclusive file lock for the life
//!   of the process; `<sid>.json` beside it is a heartbeat. A launch that
//!   can take the lock of a session it does not own has found a session
//!   that died without `session_end` — a native crash (driver, Metal abort,
//!   stack overflow), a hang the tester force-quit, an OOM kill. Those never
//!   reach a panic hook; this is how they get reported at all. The OS's own
//!   record of that pid is summarised in: macOS's `.ips` crash report, or on
//!   Windows the event log's Application Error / Application Hang entry.
//! - On Windows an unhandled-exception filter (report_win.rs) also catches
//!   native crashes IN the process — code, faulting module (usually a GPU
//!   driver), the symbolised stack from the exception context — and writes a
//!   minidump beside the report, uploaded after it.
//! - `logs/<sid>.log` is the session log (info+ for our crates, warn+ for
//!   everyone else), so a dead session still has its last words.
//! - `autosave/<sid>.wbl` is the comp, written every few seconds while it
//!   differs from what is on disk. After an unclean exit the next launch
//!   offers it back (`Recovery`), and crash reports carry it as the repro.
//!
//! Everything here is inert under a self-test or headless run (`main` gates
//! the calls), for the same reason suites never save prefs: a suite-induced
//! panic must not write to — let alone upload from — the user's real
//! directories.

use crate::theme;
use std::collections::{BTreeSet, VecDeque};
use std::fs;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock, TryLockError};
use web_time::{Duration, Instant, SystemTime};

/// Windows native crashes: the unhandled-exception filter, the stack walk
/// and the minidump. Its report-writing half is `on_native` below.
#[cfg(windows)]
#[path = "report_win.rs"]
mod win;

/// Where reports land. Baked in so a tester's build needs no setup; the env
/// vars override for development against a local `wrangler dev`.
const ENDPOINT: &str = "https://feedback.wobble.ing";
/// Shared ingest key. Spam protection only — it ships inside the binary, so
/// it is extractable by anyone we gave the binary to; the Worker can rotate
/// it server-side the day that matters (see tools/feedback-worker/README.md).
/// A release build takes it from `WOBBLE_INGEST_KEY` at compile time (the
/// release scripts read it from `tools/feedback-worker/.secrets`), so
/// source never holds a live key. Without it a build keeps its reports in
/// the local outbox.
const INGEST_KEY: &str = match option_env!("WOBBLE_INGEST_KEY") {
    Some(k) => k,
    None => "",
};

const RING_CAP: usize = 400;
/// Keep at most this many unsent crash files; a crash loop must not fill
/// the disk before anyone reads the reports.
const CRASH_KEEP: usize = 20;
const LOG_KEEP: usize = 12;
const LOG_MAX_BYTES: u64 = 16 << 20;
/// What a report may carry. The Worker caps again; these keep the request
/// under its body limit with both attached.
const LOGS_CAP: usize = 128 * 1024;
const SETUP_CAP: usize = 256 * 1024;
const AUTOSAVE_EVERY: Duration = Duration::from_secs(15);
const HEARTBEAT_EVERY: Duration = Duration::from_secs(30);
const AUTOSAVE_MAX_AGE: Duration = Duration::from_secs(14 * 24 * 3600);

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const GIT: &str = env!("WOBBLE_GIT");
/// Build stamp (unix seconds; build.rs): what "is there a newer build" compares.
const STAMP: &str = env!("WOBBLE_STAMP");

fn stamp() -> u64 {
    STAMP.parse().unwrap_or(0)
}

// ------------------------------------------------------------------ paths

fn root() -> Option<PathBuf> {
    if let Some(d) = std::env::var_os("WOBBLE_REPORT_DIR").filter(|d| !d.is_empty()) {
        return Some(PathBuf::from(d));
    }
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(|h| PathBuf::from(h).join(".wobble"))
}

fn sub(name: &str) -> Option<PathBuf> {
    root().map(|r| r.join(name))
}

/// The outbox. Named `crash` for the files R's first version left there.
fn outbox() -> Option<PathBuf> {
    sub("crash")
}

fn unix_secs() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn mtime_secs(p: &Path) -> u64 {
    fs::metadata(p)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_secs())
}

/// Write via a temp file + rename, so a crash mid-write never leaves half a
/// report (or half a comp) behind.
fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, bytes)?;
    fs::rename(&tmp, path)
}

fn list(dir: &Path, ext: &str) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.extension().is_some_and(|e| e == ext))
                .collect()
        })
        .unwrap_or_default();
    v.sort();
    v
}

/// The last `cap` bytes of a file, starting at a line boundary.
fn file_tail(p: &Path, cap: usize) -> String {
    let Ok(mut f) = fs::File::open(p) else { return String::new() };
    let len = f.metadata().map(|m| m.len()).unwrap_or(0);
    let from = len.saturating_sub(cap as u64);
    if f.seek(SeekFrom::Start(from)).is_err() {
        return String::new();
    }
    let mut buf = Vec::new();
    let _ = f.read_to_end(&mut buf);
    let s = String::from_utf8_lossy(&buf).into_owned();
    match (from > 0, s.find('\n')) {
        (true, Some(i)) => s[i + 1..].to_string(),
        _ => s,
    }
}

fn cap_str(mut s: String, cap: usize) -> String {
    if s.len() > cap {
        let mut i = cap;
        while !s.is_char_boundary(i) {
            i -= 1;
        }
        s.truncate(i);
    }
    s
}

/// FNV-1a: a hash that is the same in every build, which is the point —
/// crash signatures must group across versions.
fn fnv(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn random_hex() -> String {
    use std::hash::{BuildHasher, Hasher};
    let t = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64);
    let mut out = String::new();
    for i in 0..2u64 {
        // RandomState is seeded from the OS per instance.
        let mut h = std::collections::hash_map::RandomState::new().build_hasher();
        h.write_u64(t ^ i);
        h.write_u32(std::process::id());
        out.push_str(&format!("{:016x}", h.finish()));
    }
    out
}

/// One random id per machine (well, per `~/.wobble`), so reports from a
/// tester who never typed a name still group by machine.
fn install_id() -> &'static str {
    static ID: OnceLock<String> = OnceLock::new();
    ID.get_or_init(|| {
        let Some(dir) = root() else { return String::new() };
        let p = dir.join("install_id");
        if let Ok(s) = fs::read_to_string(&p) {
            let s = s.trim();
            if s.len() >= 8 {
                return s.to_string();
            }
        }
        let id = random_hex();
        let _ = fs::create_dir_all(&dir);
        let _ = fs::write(&p, &id);
        id
    })
}

// ---------------------------------------------------------------- logging

/// Recent log lines, whatever filter stderr runs at. Feeds the `logs` field
/// of crash and (opted-in) feedback reports.
static RING: Mutex<VecDeque<String>> = Mutex::new(VecDeque::new());
static START: OnceLock<Instant> = OnceLock::new();
/// The session log file + bytes written, once `session_begin` opened it.
static LOGFILE: Mutex<Option<(std::io::LineWriter<fs::File>, u64)>> = Mutex::new(None);

/// Our own crates log at info into the ring + file whatever stderr's filter
/// says: a crash report wants "loaded setup x, bake started" context, and
/// those are info lines. Everyone else only at warn.
fn ours(target: &str) -> bool {
    target.starts_with("wobble") || target.starts_with("engine") || target.starts_with("app")
}

fn keep(level: log::Level, target: &str) -> bool {
    level <= log::Level::Warn || (level <= log::Level::Info && ours(target))
}

struct Tee {
    inner: env_logger::Logger,
}

impl log::Log for Tee {
    fn enabled(&self, m: &log::Metadata) -> bool {
        self.inner.enabled(m) || keep(m.level(), m.target())
    }
    fn log(&self, r: &log::Record) {
        if keep(r.level(), r.target()) {
            let t = START.get().map(|s| s.elapsed().as_secs_f32()).unwrap_or(0.0);
            let line = cap_str(
                format!("+{t:8.1}s {:5} {}: {}", r.level(), r.target(), r.args()),
                2000,
            );
            to_file(&line);
            if let Ok(mut ring) = RING.lock() {
                if ring.len() >= RING_CAP {
                    ring.pop_front();
                }
                ring.push_back(line);
            }
        }
        self.inner.log(r);
    }
    fn flush(&self) {
        self.inner.flush()
    }
}

fn to_file(line: &str) {
    // try_lock: the panic hook logs too, and it may fire while this very
    // thread holds the lock.
    let Ok(mut g) = LOGFILE.try_lock() else { return };
    let Some((w, n)) = g.as_mut() else { return };
    if *n >= LOG_MAX_BYTES {
        return;
    }
    let _ = writeln!(w, "{line}");
    *n += line.len() as u64 + 1;
    if *n >= LOG_MAX_BYTES {
        let _ = writeln!(w, "… log capped at {} MB", LOG_MAX_BYTES >> 20);
    }
}

/// Drop-in for the old `env_logger::…::init()` line: same env / default-warn
/// filter, same stderr output, plus the ring and (after `session_begin`) the
/// session log. Timestamps are relative to this call, which is what you want
/// next to a crash ("2s in" vs wall time).
pub fn init_logging() {
    START.get_or_init(Instant::now);
    let inner =
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).build();
    log::set_max_level(inner.filter().max(log::LevelFilter::Info));
    let _ = log::set_boxed_logger(Box::new(Tee { inner }));
}

fn ring_tail() -> String {
    let joined = match RING.try_lock() {
        Ok(r) => r.iter().cloned().collect::<Vec<_>>().join("\n"),
        Err(TryLockError::Poisoned(p)) => p.into_inner().iter().cloned().collect::<Vec<_>>().join("\n"),
        Err(TryLockError::WouldBlock) => String::new(),
    };
    // Keep the END: that is where the crash is.
    let n = joined.len();
    if n > LOGS_CAP {
        let mut i = n - LOGS_CAP;
        while !joined.is_char_boundary(i) {
            i += 1;
        }
        joined[i..].to_string()
    } else {
        joined
    }
}

// ------------------------------------------------------------ environment

static GPU: OnceLock<String> = OnceLock::new();
static OS: OnceLock<String> = OnceLock::new();
/// (who, invite code). `who` is what tags a report: the name, the email, or
/// "name <email>" — whatever the user gave. The code is a legacy alpha
/// invite (the Worker maps it to the name it was minted for); nobody new
/// gets one.
static TESTER: Mutex<(String, String)> = Mutex::new((String::new(), String::new()));

/// GPU, backend and DRIVER version — most native crashes on Windows are in
/// the driver, and the fix is often "update it".
pub fn set_gpu(info: &wgpu::AdapterInfo) {
    let driver = [info.driver.trim(), info.driver_info.trim()]
        .into_iter()
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    let _ = GPU.set(if driver.is_empty() {
        format!("{} ({:?})", info.name, info.backend)
    } else {
        format!("{} ({:?}, {driver})", info.name, info.backend)
    });
}

pub fn set_tester(name: &str, email: &str, code: &str) {
    let (name, email) = (name.trim(), email.trim());
    let who = match (name.is_empty(), email.is_empty()) {
        (false, false) => format!("{name} <{email}>"),
        (false, true) => name.to_string(),
        (true, false) => email.to_string(),
        (true, true) => String::new(),
    };
    if let Ok(mut t) = TESTER.lock() {
        *t = (who, code.trim().to_string());
    }
}

fn tester() -> (String, String) {
    TESTER.try_lock().map(|t| t.clone()).unwrap_or_default()
}

fn run(cmd: &str, args: &[&str]) -> Option<String> {
    let mut c = std::process::Command::new(cmd);
    c.args(args);
    // A console window must not flash up from a windowed app.
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        c.creation_flags(0x0800_0000); // CREATE_NO_WINDOW
    }
    let out = c.output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
}

fn os_string() -> &'static str {
    OS.get_or_init(|| {
        let base = format!("{} {}", std::env::consts::OS, std::env::consts::ARCH);
        let ver = if cfg!(target_os = "macos") {
            run("sw_vers", &["-productVersion"])
        } else if cfg!(target_os = "linux") {
            run("uname", &["-r"])
        } else if cfg!(target_os = "windows") {
            run("cmd", &["/C", "ver"])
        } else {
            None
        };
        match ver {
            Some(v) => format!("{base} {v}"),
            None => base,
        }
    })
}

// ---------------------------------------------------------------- reports

#[derive(serde::Serialize, serde::Deserialize, Default, Clone)]
pub struct Report {
    /// Payload shape version, so the Worker can grow without guessing.
    pub schema: u32,
    /// "crash" | "unclean" | "bug" | "feature" | "test".
    pub kind: String,
    /// Client clock, unix seconds (the server records its own too).
    pub ts: u64,
    pub version: String,
    pub git: String,
    pub os: String,
    pub gpu: String,
    pub tester: String,
    pub title: String,
    pub body: String,
    pub logs: String,
    /// The session the report is ABOUT (a dead one, for unclean exits).
    #[serde(default)]
    pub session: String,
    #[serde(default)]
    pub install: String,
    /// Invite code; the Worker resolves it to the tester it was minted for.
    #[serde(default)]
    pub code: String,
    /// Crash signature: same bug, same sig, across builds and machines.
    #[serde(default)]
    pub sig: String,
    /// The comp (setup JSON) at the time, when there was one.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub setup: String,
    /// An unclean exit whose macOS crash report may not be written yet:
    /// the dead pid + its start, for the upload pass to look again.
    #[serde(default, skip_serializing_if = "is_zero")]
    native_pid: u32,
    #[serde(default, skip_serializing_if = "is_zero")]
    native_after: u64,
    /// A minidump beside this report in the outbox (its file name),
    /// uploaded after the report (Windows native crashes).
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub dump: String,
}

fn is_zero<T: Default + PartialEq>(v: &T) -> bool {
    *v == T::default()
}

impl Report {
    fn new(kind: &str, title: String, body: String, logs: String) -> Self {
        let (name, code) = tester();
        Report {
            schema: 2,
            kind: kind.into(),
            ts: unix_secs(),
            version: VERSION.into(),
            git: GIT.into(),
            os: os_string().into(),
            gpu: GPU.get().cloned().unwrap_or_default(),
            tester: name,
            title,
            body,
            logs,
            session: SESSION.get().map(|s| s.sid.clone()).unwrap_or_default(),
            install: install_id().into(),
            code,
            sig: String::new(),
            setup: String::new(),
            native_pid: 0,
            native_after: 0,
            dump: String::new(),
        }
    }
}

/// "same bug" = same first + last message line (digits and addresses
/// blanked, so frame numbers and pointers don't split a group) at the same
/// first frame of our own code.
fn sig_of(kind: &str, msg: &str, frame: &str) -> String {
    let lines: Vec<&str> = msg.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    let mut m = lines.first().copied().unwrap_or("").to_string();
    if lines.len() > 1 {
        m.push_str(" … ");
        m.push_str(lines[lines.len() - 1]);
    }
    let mut norm = String::with_capacity(m.len());
    let mut prev_digit = false;
    let mut chars = m.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '0' && chars.peek() == Some(&'x') {
            chars.next();
            while chars.peek().is_some_and(|c| c.is_ascii_hexdigit()) {
                chars.next();
            }
            norm.push('#');
            prev_digit = true;
        } else if c.is_ascii_digit() {
            if !prev_digit {
                norm.push('#');
            }
            prev_digit = true;
        } else {
            norm.push(c);
            prev_digit = false;
        }
    }
    format!("{:012x}", fnv(&format!("{kind}|{norm}|{frame}")) & 0xffff_ffff_ffff)
}

/// The panic hook's own machinery is the first dozen frames of every
/// backtrace; drop it so the report opens on the code that panicked, and
/// drop the runtime's frames below `main`. Home-dir and toolchain prefixes
/// are shortened (also keeps the tester's user name out of the report).
/// Returns (trimmed trace, first frame in our own crates).
fn trim_backtrace(bt: &str) -> (String, String) {
    const MACHINERY: [&str; 9] = [
        "std::backtrace",
        "backtrace_rs",
        "std::panicking",
        "core::panicking",
        "rust_begin_unwind",
        "__rust_end_short_backtrace",
        "install_panic_hook",
        "core::ops::function::Fn",
        "std::sys::backtrace",
    ];
    let is_head = |l: &str| {
        let t = l.trim_start();
        t.split_once(": ").is_some_and(|(n, _)| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
    };
    let mut frames: Vec<Vec<&str>> = Vec::new();
    for l in bt.lines() {
        if is_head(l) {
            frames.push(vec![l]);
        } else if let Some(f) = frames.last_mut() {
            f.push(l);
        }
    }
    let sym = |f: &Vec<&str>| f[0].trim_start().split_once(": ").map_or("", |(_, s)| s).to_string();
    let start = frames
        .iter()
        .position(|f| !MACHINERY.iter().any(|m| sym(f).contains(m)))
        .unwrap_or(0);
    let end = frames
        .iter()
        .position(|f| sym(f).contains("std::rt::lang_start"))
        .unwrap_or(frames.len())
        .max(start);
    let first_ours = frames[start..end]
        .iter()
        .map(|f| sym(f))
        .find(|s| {
            let s = s.trim_start_matches('<');
            s.starts_with("wobble::") || s.starts_with("engine::") || s.starts_with("app::")
        })
        .unwrap_or_default();
    let home = std::env::var("HOME").or_else(|_| std::env::var("USERPROFILE")).unwrap_or_default();
    let mut out = String::new();
    for f in &frames[start..end] {
        for l in f {
            let mut l = l.to_string();
            if let Some(i) = l.find("/library/") {
                if l[..i].contains("/rustc/") {
                    l = format!("{}<std>{}", &l[..l.find("/rustc/").unwrap()], &l[i..]);
                }
            }
            if let Some(i) = l.find("/registry/src/") {
                if let Some(j) = l[i + 14..].find('/') {
                    l = format!("{}<crates>{}", &l[..l[..i].rfind(' ').map_or(0, |k| k + 1)], &l[i + 14 + j..]);
                }
            }
            if !home.is_empty() {
                l = l.replace(&home, "~");
            }
            out.push_str(&l);
            out.push('\n');
        }
    }
    if end < frames.len() {
        out.push_str("  … (runtime frames below main trimmed)\n");
    }
    (out, first_ours)
}

fn endpoint() -> Option<String> {
    std::env::var("WOBBLE_REPORT_URL")
        .ok()
        .filter(|s| !s.is_empty())
        .or_else(|| (!ENDPOINT.is_empty()).then(|| ENDPOINT.to_string()))
}

fn ingest_key() -> String {
    std::env::var("WOBBLE_REPORT_KEY").unwrap_or_else(|_| INGEST_KEY.to_string())
}

/// The web build has no HTTP client (no ureq on wasm32).
#[cfg(target_arch = "wasm32")]
fn http(_method: &str, _path: &str, _body: Option<&str>) -> Result<(u16, String), String> {
    Err("reports cannot be sent from the web build".into())
}

#[cfg(not(target_arch = "wasm32"))]
fn http(method: &str, path: &str, body: Option<&str>) -> Result<(u16, String), String> {
    let url = endpoint().ok_or("no report endpoint configured in this build")?;
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(20)))
        // A 4xx is an answer, not a transport failure: read its body.
        .http_status_as_error(false)
        .build()
        .into();
    let url = format!("{}{path}", url.trim_end_matches('/'));
    let code = tester().1;
    let mut resp = match (method, body) {
        ("POST", Some(b)) => agent
            .post(&url)
            .header("content-type", "application/json")
            .header("x-wobble-key", &ingest_key())
            .header("x-wobble-code", &code)
            .send(b),
        _ => agent
            .get(&url)
            .header("x-wobble-key", &ingest_key())
            .header("x-wobble-code", &code)
            .call(),
    }
    .map_err(|e| format!("{e}"))?;
    let status = resp.status().as_u16();
    let text = resp.body_mut().read_to_string().unwrap_or_default();
    Ok((status, text))
}

/// POST one already-serialized payload. Ok(id) on a 2xx; the error string
/// carries the status + body prefix, which is what the status line shows.
fn post(path: &str, body: &str) -> Result<String, String> {
    let (status, text) = http("POST", path, Some(body))?;
    if (200..300).contains(&status) {
        Ok(serde_json::from_str::<serde_json::Value>(&text)
            .ok()
            .and_then(|v| v.get("id").and_then(|i| i.as_str()).map(String::from))
            .unwrap_or_else(|| "?".into()))
    } else {
        Err(format!("HTTP {status}: {}", cap_str(text, 200)))
    }
}

#[cfg(target_arch = "wasm32")]
fn put_bytes(_path: &str, _bytes: &[u8]) -> Result<(), String> {
    Err("no uploads from the web build".into())
}

/// PUT raw bytes (a minidump).
#[cfg(not(target_arch = "wasm32"))]
fn put_bytes(path: &str, bytes: &[u8]) -> Result<(), String> {
    let url = endpoint().ok_or("no report endpoint configured in this build")?;
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(180)))
        .http_status_as_error(false)
        .build()
        .into();
    let mut resp = agent
        .put(format!("{}{path}", url.trim_end_matches('/')))
        .header("content-type", "application/octet-stream")
        .header("x-wobble-key", &ingest_key())
        .send(bytes)
        .map_err(|e| format!("{e}"))?;
    let status = resp.status().as_u16();
    if (200..300).contains(&status) {
        Ok(())
    } else {
        Err(format!("HTTP {status}: {}", cap_str(resp.body_mut().read_to_string().unwrap_or_default(), 200)))
    }
}

fn send(r: &Report) -> Result<String, String> {
    post("/v1/report", &serde_json::to_string(r).map_err(|e| e.to_string())?)
}

// ------------------------------------------------------------- crash path

/// `crash-<session>-<n>` in the outbox (no extension): how the panic hook
/// and the native filter name their reports, so the next launch can tell a
/// session that reported its own death from one that just vanished.
fn crash_stem() -> Option<PathBuf> {
    static N: AtomicU32 = AtomicU32::new(0);
    let dir = outbox()?;
    fs::create_dir_all(&dir).ok()?;
    let sid = SESSION
        .get()
        .map(|s| s.sid.clone())
        .unwrap_or_else(|| format!("{}-{}", unix_secs(), std::process::id()));
    Some(dir.join(format!("crash-{sid}-{}", N.fetch_add(1, Ordering::SeqCst))))
}

fn flush_log() {
    if let Ok(mut g) = LOGFILE.try_lock() {
        if let Some((w, _)) = g.as_mut() {
            let _ = w.flush();
        }
    }
}

/// Capture panics on every thread. Chains to the previous hook, so the
/// normal stderr message still prints; ours runs first and only writes a
/// file — never the network, never a dialog (a panicking render thread has
/// no business blocking on either).
pub fn install_panic_hook() {
    os_string(); // shell out for the OS version now, not inside the hook
    install_id();
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        // A panic inside this hook (or a second thread panicking while it
        // runs) gets the plain message only.
        static IN_HOOK: AtomicBool = AtomicBool::new(false);
        if IN_HOOK.swap(true, Ordering::SeqCst) {
            prev(info);
            return;
        }
        let msg = info
            .payload()
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| info.payload().downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "non-string panic payload".into());
        let thread = std::thread::current();
        let bt = std::backtrace::Backtrace::force_capture().to_string();
        let (trace, first_ours) = trim_backtrace(&bt);
        let loc = info
            .location()
            .map(|l| format!("{}:{}", l.file(), l.line()))
            .unwrap_or_default();
        let body = format!(
            "at {}\nthread: {}\n\nbacktrace:\n{trace}",
            if loc.is_empty() { "unknown location" } else { &loc },
            thread.name().unwrap_or("unnamed"),
        );
        to_file(&format!("PANIC on {}: {msg}", thread.name().unwrap_or("unnamed")));
        let mut report = Report::new("crash", cap_str(format!("panic: {msg}"), 300), body, ring_tail());
        // Location inside a dependency (wgpu's error handler) says nothing
        // about which of OUR calls did it; our first frame does.
        report.sig = sig_of("crash", &msg, if first_ours.is_empty() { &loc } else { &first_ours });
        if let (Some(path), Ok(json)) =
            (crash_stem().map(|s| s.with_extension("json")), serde_json::to_string_pretty(&report))
        {
            if write_atomic(&path, json.as_bytes()).is_ok() {
                eprintln!(
                    "wobble: crash report written to {} — it uploads on the next launch",
                    path.display()
                );
            }
        }
        flush_log();
        IN_HOOK.store(false, Ordering::SeqCst);
        prev(info);
    }));
    #[cfg(windows)]
    win::install(on_native);
}

/// The Windows exception filter's handler (report_win.rs), on the crashing
/// thread. Writes what it knows at once, then reaches for more — a minidump,
/// then the symbolised stack — rewriting the report after each, so a step
/// that dies in the broken process still leaves the last good report.
#[cfg(windows)]
fn on_native(f: &win::Fatal) {
    let what = code_name(f.code);
    let code = format!("0x{:08x}", f.code);
    let thread = std::thread::current().name().unwrap_or("unnamed").to_string();
    to_file(&format!("NATIVE CRASH on {thread}: {code} {what} in {}+0x{:x}", f.module, f.offset));
    flush_log();
    let Some(stem) = crash_stem() else { return };
    let json = stem.with_extension("json");
    let head = format!("exception {code} ({what}) at {}+0x{:x}\nthread: {thread}\n", f.module, f.offset);
    let mut r = Report::new(
        "crash",
        cap_str(format!("native crash: {code} {what} in {}", f.module), 300),
        head.clone(),
        ring_tail(),
    );
    r.sig = sig_of("native", &code, &f.module);
    let save = |r: &Report| {
        if let Ok(j) = serde_json::to_vec_pretty(r) {
            let _ = write_atomic(&json, &j);
        }
    };
    save(&r);
    let dmp = stem.with_extension("dmp");
    if f.write_minidump(&dmp) {
        r.dump = dmp.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        save(&r);
    }
    let frames = f.stack();
    if !frames.is_empty() {
        // The first frame in OUR code says which of our calls led into it —
        // in the title when the fault is in our own module, and always in
        // the signature, so one driver crash reached from two places is two
        // groups.
        let ours = frames.iter().find_map(|l| {
            let sym = l.lines().next()?.split_once("  ")?.1.trim();
            let bare = sym.trim_start_matches('<');
            (bare.starts_with("wobble::") || bare.starts_with("engine::") || bare.starts_with("app::"))
                .then(|| sym.to_string())
        });
        if let Some(o) = &ours {
            r.sig = sig_of("native", &code, &format!("{}|{o}", f.module));
            if f.module.to_ascii_lowercase().ends_with(".exe") {
                r.title = cap_str(format!("native crash: {code} {what} in {o}"), 300);
            } else {
                r.title = cap_str(format!("native crash: {code} {what} in {} (from {o})", f.module), 300);
            }
        }
        r.body = format!("{head}\nstack (faulting thread, from the exception context):\n{}", frames.join("\n"));
        save(&r);
    }
    eprintln!("wobble: native crash report written to {} — it uploads on the next launch", json.display());
}

// ---------------------------------------------------------------- sessions

struct Session {
    sid: String,
    start: u64,
    /// Held for the process lifetime: the lock IS the liveness signal.
    lock: Mutex<Option<fs::File>>,
}

static SESSION: OnceLock<Session> = OnceLock::new();
static ENDED: AtomicBool = AtomicBool::new(false);
/// Node kinds seen in this session's comps + the biggest graph, for stats.
static STATS: Mutex<(BTreeSet<String>, usize)> = Mutex::new((BTreeSet::new(), 0));

/// A session's heartbeat file, and the stats record the Worker receives.
#[derive(serde::Serialize, serde::Deserialize, Default, Clone)]
struct SessionInfo {
    schema: u32,
    id: String,
    install: String,
    tester: String,
    code: String,
    version: String,
    git: String,
    os: String,
    gpu: String,
    pid: u32,
    /// unix seconds
    start: u64,
    /// last heartbeat, unix seconds
    last: u64,
    /// seconds; for an unclean session, up to its last heartbeat
    duration: u64,
    /// None = still running (a start record)
    clean: Option<bool>,
    kinds: Vec<String>,
    peak_nodes: usize,
}

fn session_info(s: &Session, clean: Option<bool>) -> SessionInfo {
    let now = unix_secs();
    let (kinds, peak) = STATS
        .try_lock()
        .map(|g| (g.0.iter().cloned().collect(), g.1))
        .unwrap_or_default();
    let (name, code) = tester();
    SessionInfo {
        schema: 1,
        id: s.sid.clone(),
        install: install_id().into(),
        tester: name,
        code,
        version: VERSION.into(),
        git: GIT.into(),
        os: os_string().into(),
        gpu: GPU.get().cloned().unwrap_or_default(),
        pid: std::process::id(),
        start: s.start,
        last: now,
        duration: now.saturating_sub(s.start),
        clean,
        kinds,
        peak_nodes: peak,
    }
}

fn heartbeat() {
    let (Some(s), Some(dir)) = (SESSION.get(), sub("sessions")) else { return };
    if let Ok(j) = serde_json::to_vec(&session_info(s, None)) {
        let _ = write_atomic(&dir.join(format!("{}.json", s.sid)), &j);
    }
}

fn queue_session(info: &SessionInfo) {
    let Some(dir) = outbox() else { return };
    let _ = fs::create_dir_all(&dir);
    if let Ok(j) = serde_json::to_vec_pretty(info) {
        let _ = write_atomic(&dir.join(format!("session-{}.json", info.id)), &j);
    }
}

/// An earlier session's comp, offered back after it died.
pub struct Recovery {
    sid: String,
    autosave: PathBuf,
    /// The file the comp was opened from / last saved to, if any.
    pub original: Option<PathBuf>,
    /// When the autosave was written, unix seconds.
    saved: u64,
    /// How it died, one line.
    how: String,
}

static RECOVERY: Mutex<Option<Recovery>> = Mutex::new(None);

pub fn take_recovery() -> Option<Recovery> {
    RECOVERY.lock().ok().and_then(|mut r| r.take())
}

#[derive(serde::Serialize, serde::Deserialize, Default)]
struct AutosaveMeta {
    original: Option<PathBuf>,
    saved: u64,
}

/// Start this process's session: adopt any dead ones first (queue their
/// reports, find a comp to recover), then take our own lock, open the log,
/// queue a start record. Local files only — uploads wait for `start_uploads`
/// (which waits for the tester to have seen the alpha notice).
pub fn session_begin() {
    let (Some(sdir), Some(ldir)) = (sub("sessions"), sub("logs")) else { return };
    let _ = fs::create_dir_all(&sdir);
    let _ = fs::create_dir_all(&ldir);
    let how = adopt_dead_sessions(&sdir);
    let rec = orphaned_autosave(&sdir, &how);

    let start = unix_secs();
    let sid = format!("{start}-{}", std::process::id());
    let lock = fs::OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(sdir.join(format!("{sid}.lock")))
        .ok()
        .filter(|f| f.try_lock().is_ok());
    if lock.is_none() {
        log::warn!("session lock unavailable — unclean-exit detection is off this run");
    }
    let _ = SESSION.set(Session { sid: sid.clone(), start, lock: Mutex::new(lock) });

    // The session log, seeded with whatever was logged before it existed.
    if let Ok(f) = fs::File::create(ldir.join(format!("{sid}.log"))) {
        let mut w = std::io::LineWriter::new(f);
        let mut n = 0u64;
        let head = format!("wobble {VERSION} ({GIT}) session {sid}, {}", os_string());
        let _ = writeln!(w, "{head}");
        n += head.len() as u64 + 1;
        if let Ok(ring) = RING.lock() {
            for l in ring.iter() {
                let _ = writeln!(w, "{l}");
                n += l.len() as u64 + 1;
            }
        }
        if let Ok(mut g) = LOGFILE.lock() {
            *g = Some((w, n));
        }
    }
    heartbeat();
    if let Some(s) = SESSION.get() {
        queue_session(&session_info(s, None));
    }

    // Housekeeping: old logs, stale autosaves.
    let logs = list(&ldir, "log");
    for p in logs.iter().take(logs.len().saturating_sub(LOG_KEEP)) {
        let _ = fs::remove_file(p);
    }
    if let Some(adir) = sub("autosave") {
        let keep = rec.as_ref().map(|r| r.sid.clone()).unwrap_or_default();
        for p in fs::read_dir(&adir).into_iter().flatten().filter_map(|e| e.ok().map(|e| e.path())) {
            let stem = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            let old = unix_secs().saturating_sub(mtime_secs(&p)) > AUTOSAVE_MAX_AGE.as_secs();
            if old && (keep.is_empty() || !stem.starts_with(&keep)) {
                let _ = fs::remove_file(&p);
            }
        }
    }
    if let Ok(mut r) = RECOVERY.lock() {
        *r = rec;
    }
}

/// Every `.lock` we can take belongs to a process that is gone without
/// `session_end`. Queue what it deserves (its session record, and a report
/// unless its panic hook already wrote one); returns how each one died.
fn adopt_dead_sessions(sdir: &Path) -> std::collections::HashMap<String, String> {
    let mut how = std::collections::HashMap::new();
    for lock_path in list(sdir, "lock") {
        let Ok(f) = fs::OpenOptions::new().read(true).write(true).open(&lock_path) else { continue };
        if f.try_lock().is_err() {
            continue; // alive: another wobble is running
        }
        drop(f);
        let sid = lock_path.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_string();
        let info_path = sdir.join(format!("{sid}.json"));
        let mut info: SessionInfo = fs::read(&info_path)
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_else(|| SessionInfo { id: sid.clone(), ..Default::default() });
        info.clean = Some(false);
        let setup = sub("autosave")
            .and_then(|d| fs::read_to_string(d.join(format!("{sid}.wbl"))).ok())
            .map(|s| cap_str(s, SETUP_CAP))
            .unwrap_or_default();

        let ob = outbox();
        let prefix = format!("crash-{sid}-");
        let panics: Vec<PathBuf> = ob
            .as_ref()
            .map(|d| {
                list(d, "json")
                    .into_iter()
                    .filter(|p| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with(&prefix)))
                    .collect()
            })
            .unwrap_or_default();
        if !panics.is_empty() {
            // The panic hook already said why; hand its reports the comp.
            how.insert(sid.clone(), "it crashed (the crash was reported)".to_string());
            for p in panics.iter().filter(|_| !setup.is_empty()) {
                if let Some(mut r) = fs::read(p).ok().and_then(|b| serde_json::from_slice::<Report>(&b).ok()) {
                    if r.setup.is_empty() {
                        r.setup = setup.clone();
                        if let Ok(j) = serde_json::to_vec_pretty(&r) {
                            let _ = write_atomic(p, &j);
                        }
                    }
                }
            }
        } else if let Some(dir) = &ob {
            let log_tail = sub("logs")
                .map(|d| file_tail(&d.join(format!("{sid}.log")), LOGS_CAP))
                .unwrap_or_default();
            let mins = info.duration / 60;
            let body = format!(
                "The previous session ended without shutting down cleanly.\n\n\
                 session {sid} · ran {mins} min (to its last heartbeat) · build {} ({})\n\n{}",
                if info.version.is_empty() { "?" } else { &info.version },
                if info.git.is_empty() { "?" } else { &info.git },
                if cfg!(target_os = "macos") {
                    "no macOS crash report exists for that process, so this was most likely a \
                     force-quit of a hang, a kill (OOM, logout) or power loss."
                } else if cfg!(windows) {
                    "Windows logged no crash or hang for that process, so this was most likely \
                     a kill (Task Manager, logout, out of memory) or power loss."
                } else {
                    "native crashes are not captured on this platform yet; this could be a \
                     crash, a force-quit of a hang, a kill or power loss."
                },
            );
            let mut r = Report::new("unclean", format!("unclean exit, no crash report ({mins} min in)"), body, log_tail);
            r.session = sid.clone();
            if !info.version.is_empty() {
                r.version = info.version.clone();
                r.git = info.git.clone();
            }
            if !info.gpu.is_empty() {
                r.gpu = info.gpu.clone();
            }
            r.sig = sig_of("unclean", "unclean exit", "");
            r.setup = setup;
            // The OS writes its record seconds AFTER the crash (macOS's .ips,
            // Windows' WER event) — often after a quick relaunch got here.
            // Look now; the upload pass looks again (`settle_native`) while
            // native_pid is set.
            if cfg!(any(target_os = "macos", windows)) && info.pid != 0 {
                r.native_pid = info.pid;
                r.native_after = info.start;
                if let Some(n) = native_crash(info.pid, info.start) {
                    apply_native(&mut r, &n);
                }
            }
            how.insert(
                sid.clone(),
                match r.kind.as_str() {
                    "crash" => "it crashed",
                    "hang" => "it stopped responding and was closed",
                    _ => "it didn't close cleanly",
                }
                .to_string(),
            );
            let _ = fs::create_dir_all(dir);
            if let Ok(j) = serde_json::to_vec_pretty(&r) {
                let _ = write_atomic(&dir.join(format!("unclean-{sid}.json")), &j);
            }
        }
        if !info.version.is_empty() {
            queue_session(&info);
        }
        let _ = fs::remove_file(&lock_path);
        let _ = fs::remove_file(&info_path);
    }
    how
}

/// Clean exits delete their autosave, so one whose session holds no lock is
/// from a session that died — this launch's adoptions, or an earlier one
/// whose recovery prompt never got shown (it died again before the UI).
/// The newest is offered back.
fn orphaned_autosave(sdir: &Path, how: &std::collections::HashMap<String, String>) -> Option<Recovery> {
    let adir = sub("autosave")?;
    let mut best: Option<Recovery> = None;
    for a in list(&adir, "wbl") {
        let sid = a.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_string();
        if sdir.join(format!("{sid}.lock")).exists() {
            continue; // a live session's
        }
        let meta: AutosaveMeta = fs::read(adir.join(format!("{sid}.meta.json")))
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_default();
        let saved = if meta.saved > 0 { meta.saved } else { mtime_secs(&a) };
        if best.as_ref().is_none_or(|b| saved > b.saved) {
            let how = how.get(&sid).cloned().unwrap_or_else(|| "it didn't close cleanly".to_string());
            best = Some(Recovery { sid, autosave: a, original: meta.original, saved, how });
        }
    }
    best
}

fn apply_native(r: &mut Report, n: &Native) {
    r.kind = n.kind.into();
    r.title = cap_str(n.title.clone(), 300);
    r.sig = n.sig.clone();
    // body = intro ¶ session line ¶ detail: swap the detail for the summary.
    let keep: Vec<&str> = r.body.splitn(3, "\n\n").take(2).collect();
    r.body = format!("{}\n\n{}", keep.join("\n\n"), n.summary);
    r.native_pid = 0;
}

/// Clean shutdown. Idempotent (called from winit's `exiting` AND after the
/// event loop returns).
pub fn session_end() {
    if ENDED.swap(true, Ordering::SeqCst) {
        return;
    }
    let Some(s) = SESSION.get() else { return };
    log::info!("session end after {} s", unix_secs().saturating_sub(s.start));
    queue_session(&session_info(s, Some(true)));
    if let Some(a) = sub("autosave") {
        let _ = fs::remove_file(a.join(format!("{}.wbl", s.sid)));
        let _ = fs::remove_file(a.join(format!("{}.meta.json", s.sid)));
    }
    if let Some(d) = sub("sessions") {
        let _ = fs::remove_file(d.join(format!("{}.json", s.sid)));
        if let Ok(mut l) = s.lock.lock() {
            drop(l.take());
        }
        let _ = fs::remove_file(d.join(format!("{}.lock", s.sid)));
    }
    if let Ok(mut g) = LOGFILE.lock() {
        if let Some((w, _)) = g.as_mut() {
            let _ = w.flush();
        }
    }
}

// ------------------------------------------- native: the OS's own record

/// What the OS recorded about a process that died without telling us.
/// Built on macOS (.ips) and Windows (event log) only.
#[cfg_attr(not(any(target_os = "macos", windows)), allow(dead_code))]
struct Native {
    /// "crash", or "hang" for a window Windows closed as not responding.
    kind: &'static str,
    title: String,
    sig: String,
    summary: String,
}

/// macOS writes `~/Library/Logs/DiagnosticReports/<proc>-<date>.ips` for a
/// native crash: first line a JSON header, then a JSON body with the
/// exception, the faulting thread's frames (symbolicated from our symbol
/// table) and "application specific info" (abort messages). Match by pid.
#[cfg(target_os = "macos")]
fn native_crash(pid: u32, after: u64) -> Option<Native> {
    if pid == 0 {
        return None;
    }
    let home = std::env::var_os("HOME")?;
    let base = PathBuf::from(home).join("Library/Logs/DiagnosticReports");
    for dir in [base.clone(), base.join("Retired")] {
        for p in list(&dir, "ips") {
            if mtime_secs(&p) + 5 < after || fs::metadata(&p).map_or(0, |m| m.len()) > 8 << 20 {
                continue;
            }
            let Ok(text) = fs::read_to_string(&p) else { continue };
            let Some((_, body)) = text.split_once('\n') else { continue };
            let Ok(v) = serde_json::from_str::<serde_json::Value>(body) else { continue };
            if v["pid"].as_u64() != Some(pid as u64) {
                continue;
            }
            return Some(summarise_ips(&v));
        }
    }
    None
}

/// Windows Error Reporting logs every crash (Application Error, 1000) and
/// every not-responding window the user or Windows closed (Application Hang,
/// 1002) to the Application event log — including the deaths no exception
/// filter sees: `__fastfail`, which is how Rust aborts and how its
/// stack-overflow handler exits. Read with `wevtutil` (no admin needed),
/// matched by exe name + pid + time.
#[cfg(windows)]
fn native_crash(pid: u32, after: u64) -> Option<Native> {
    if pid == 0 {
        return None;
    }
    let exe = std::env::current_exe().ok()?.file_name()?.to_string_lossy().into_owned();
    let query = "/q:*[System[(Provider[@Name='Application Error'] or Provider[@Name='Application Hang']) \
                 and TimeCreated[timediff(@SystemTime) <= 1209600000]]]";
    let xml = run("wevtutil", &["qe", "Application", query, "/f:xml", "/c:50", "/rd:true"])?;
    wer_match(&parse_wer_events(&xml), &exe, pid, after)
}

#[cfg(not(any(target_os = "macos", windows)))]
fn native_crash(_pid: u32, _after: u64) -> Option<Native> {
    None
}

#[cfg(any(target_os = "macos", windows, test))]
fn demangle(s: &str) -> String {
    format!("{:#}", rustc_demangle::demangle(s))
}

#[cfg(any(target_os = "macos", test))]
fn summarise_ips(v: &serde_json::Value) -> Native {
    let s = |x: &serde_json::Value| x.as_str().unwrap_or("").to_string();
    let ex = &v["exception"];
    let exception = [s(&ex["type"]), s(&ex["signal"])]
        .into_iter()
        .filter(|x| !x.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    let mut summary = format!("exception: {exception} {}\n", s(&ex["subtype"]));
    summary.push_str(&format!("termination: {}\n", s(&v["termination"]["indicator"])));
    if let Some(asi) = v["asi"].as_object() {
        for (k, msgs) in asi {
            for m in msgs.as_array().into_iter().flatten() {
                summary.push_str(&format!("{k}: {}\n", s(m)));
            }
        }
    }
    let images: Vec<String> = v["usedImages"]
        .as_array()
        .map(|a| a.iter().map(|i| s(&i["name"])).collect())
        .unwrap_or_default();
    let proc_name = s(&v["procName"]);
    let ti = v["faultingThread"].as_u64().unwrap_or(0) as usize;
    let th = &v["threads"][ti];
    summary.push_str(&format!(
        "\nfaulting thread {ti} {} {}:\n",
        s(&th["name"]),
        s(&th["queue"])
    ));
    let mut top = String::new();
    let mut first_ours = String::new();
    const RUNTIME: [&str; 6] = ["std::", "core::", "alloc::", "__rust", "rust_begin_unwind", "_rust"];
    for (i, f) in th["frames"].as_array().into_iter().flatten().take(48).enumerate() {
        let img = images.get(f["imageIndex"].as_u64().unwrap_or(0) as usize).cloned().unwrap_or_default();
        let sym = demangle(f["symbol"].as_str().unwrap_or("???"));
        summary.push_str(&format!("{i:3} {img:<24} {sym} + {}\n", f["symbolLocation"].as_u64().unwrap_or(0)));
        let system = ["libsystem_", "libc++", "libdyld", "dyld"].iter().any(|p| img.starts_with(p));
        if top.is_empty() && !system {
            top = format!("{sym} ({img})");
        }
        if first_ours.is_empty() && img == proc_name && !RUNTIME.iter().any(|r| sym.starts_with(r)) {
            first_ours = sym.clone();
        }
    }
    if top.is_empty() {
        top = "?".into();
    }
    // Our first frame says which of OUR calls died; the top frame (often a
    // driver or libc) is in the summary.
    let at = if first_ours.is_empty() { &top } else { &first_ours };
    Native {
        kind: "crash",
        title: format!("native crash: {exception} in {at}"),
        sig: sig_of("native", &exception, &first_ours),
        summary,
    }
}

/// Names for the exception codes an alpha will actually meet.
#[cfg(any(windows, test))]
fn code_name(code: u32) -> &'static str {
    match code {
        0xC000_0005 => "access violation",
        0xC000_0409 => "fail-fast (an abort, a stack overflow, or a security check)",
        0xC000_00FD => "stack overflow",
        0xC000_001D => "illegal instruction",
        0xC000_0094 => "integer divide by zero",
        0xC000_0374 => "heap corruption",
        0xC000_0006 => "in-page error (the disk or a network file went away)",
        0xC000_0602 => "unknown fail-fast",
        0x8000_0003 => "breakpoint",
        0xE06D_7363 => "C++ exception",
        0x4000_0015 => "fatal app exit",
        _ => "exception",
    }
}

/// One `<Event>` from `wevtutil qe … /f:xml`: provider, id, time (unix s)
/// and the positional `<Data>` values (positions are locale-independent;
/// the rendered message text is not).
#[cfg(any(windows, test))]
struct WerEvent {
    provider: String,
    id: u32,
    time: u64,
    data: Vec<String>,
}

#[cfg(any(windows, test))]
fn xml_unescape(s: &str) -> String {
    s.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&amp;", "&")
}

/// `2026-10-01T12:34:56.1234567Z` → unix seconds.
#[cfg(any(windows, test))]
fn parse_iso_utc(t: &str) -> Option<u64> {
    let n = |a: usize, b: usize| t.get(a..b)?.parse::<i64>().ok();
    let (y, m, d) = (n(0, 4)?, n(5, 7)?, n(8, 10)?);
    let (hh, mm, ss) = (n(11, 13)?, n(14, 16)?, n(17, 19)?);
    // Days from the civil date (Howard Hinnant's algorithm).
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * ((m + 9) % 12) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    u64::try_from(days * 86_400 + hh * 3_600 + mm * 60 + ss).ok()
}

#[cfg(any(windows, test))]
fn parse_wer_events(xml: &str) -> Vec<WerEvent> {
    // The value of `name='…'` (or "…") on the first `<tag …>` element.
    fn attr(chunk: &str, tag: &str, name: &str) -> Option<String> {
        let el = &chunk[chunk.find(&format!("<{tag} "))?..];
        let el = &el[..el.find('>')?];
        let at = &el[el.find(&format!("{name}="))? + name.len() + 1..];
        let q = at.chars().next()?;
        let v = &at[1..];
        Some(xml_unescape(&v[..v.find(q)?]))
    }
    let mut out = Vec::new();
    for chunk in xml.split("<Event ").skip(1) {
        let provider = attr(chunk, "Provider", "Name").unwrap_or_default();
        let id = chunk
            .find("<EventID")
            .and_then(|i| {
                let rest = &chunk[i..];
                let body = &rest[rest.find('>')? + 1..];
                body[..body.find('<')?].trim().parse().ok()
            })
            .unwrap_or(0);
        let time = attr(chunk, "TimeCreated", "SystemTime").and_then(|t| parse_iso_utc(&t)).unwrap_or(0);
        let mut data = Vec::new();
        let mut rest = chunk;
        while let Some(i) = rest.find("<Data") {
            rest = &rest[i + 5..];
            // `<Data>`, `<Data Name='x'>` or an empty `<Data/>` — and not
            // some longer tag that merely starts with "Data".
            if !rest.starts_with(['>', ' ', '/']) {
                continue;
            }
            let Some(gt) = rest.find('>') else { break };
            if rest[..gt].ends_with('/') {
                data.push(String::new());
                rest = &rest[gt + 1..];
                continue;
            }
            rest = &rest[gt + 1..];
            let Some(end) = rest.find("</Data>") else { break };
            data.push(xml_unescape(&rest[..end]));
            rest = &rest[end + 7..];
        }
        out.push(WerEvent { provider, id, time, data });
    }
    out
}

#[cfg(any(windows, test))]
fn wer_match(events: &[WerEvent], exe: &str, pid: u32, after: u64) -> Option<Native> {
    let is_pid = |s: &str| {
        let h = s.trim();
        let h = h.strip_prefix("0x").or_else(|| h.strip_prefix("0X")).unwrap_or(h);
        u32::from_str_radix(h, 16).ok() == Some(pid)
    };
    for e in events {
        // pids get reused: only an event from this session's lifetime.
        if e.time != 0 && e.time + 5 < after {
            continue;
        }
        if !e.data.first().is_some_and(|a| a.eq_ignore_ascii_case(exe)) {
            continue;
        }
        let g = |i: usize| e.data.get(i).cloned().unwrap_or_default();
        match e.provider.as_str() {
            // 0 app, 1 version, 2 app stamp, 3 module, 4 module version,
            // 5 module stamp, 6 exception code, 7 offset, 8 pid (hex),
            // 9 start time, 10 app path, 11 module path, 12 report id
            "Application Error" if is_pid(&g(8)) => {
                let h = g(6);
                let h = h.trim().trim_start_matches("0x");
                let code = u32::from_str_radix(h, 16).unwrap_or(0);
                let exception = format!("0x{code:08x} {}", code_name(code));
                let module = g(3);
                return Some(Native {
                    kind: "crash",
                    title: format!("native crash: {exception} in {module}"),
                    sig: sig_of("native", &format!("0x{code:08x}"), &module),
                    summary: format!(
                        "exception: {exception}\nfaulting module: {module} {} at offset {}\nmodule path: {}\n\
                         source: Windows event log (Application Error, event {})\n",
                        g(4),
                        g(7),
                        g(11),
                        e.id
                    ),
                });
            }
            // 0 app, 1 version, 2 pid (hex), 3 start time, 4 end time, 5 path …
            "Application Hang" if is_pid(&g(2)) => {
                return Some(Native {
                    kind: "hang",
                    title: "hang: stopped responding and was closed".into(),
                    sig: sig_of("hang", "not responding", ""),
                    summary: format!(
                        "Windows closed it after it stopped responding \
                         (event log: Application Hang, event {}).\n",
                        e.id
                    ),
                });
            }
            _ => {}
        }
    }
    None
}

// ---------------------------------------------------------------- uploads

// Both off until the user says otherwise (the first-run window, or
// preferences). Joining alpha testing is what sends crash reports, logs
// and session stats; the new-build check is a separate, anonymous GET.
static CONSENT_CRASH: AtomicBool = AtomicBool::new(false);
static CONSENT_USAGE: AtomicBool = AtomicBool::new(false);
static CONSENT_UPDATES: AtomicBool = AtomicBool::new(false);
static FLUSHED: AtomicBool = AtomicBool::new(false);
static PRUNED: AtomicBool = AtomicBool::new(false);
static CHECKED: AtomicBool = AtomicBool::new(false);
static STATUS: Mutex<Option<String>> = Mutex::new(None);

/// `testing`: alpha testing joined (crash reports, logs, usage stats are
/// uploaded). `updates`: ask download.wobble.ing whether a newer build is out.
pub fn set_consent(testing: bool, updates: bool) {
    CONSENT_CRASH.store(testing, Ordering::SeqCst);
    CONSENT_USAGE.store(testing, Ordering::SeqCst);
    CONSENT_UPDATES.store(updates, Ordering::SeqCst);
}

fn set_status(s: String) {
    if let Ok(mut st) = STATUS.lock() {
        *st = Some(s);
    }
}

/// What the last upload pass did, for the feedback window's footer.
pub fn status() -> Option<String> {
    STATUS.lock().ok().and_then(|s| s.clone())
}

/// Send the outbox, oldest first. Crash-type files wait while crash
/// reporting is off (the tester may turn it on, or send them by hand);
/// session records are dropped while usage stats are off. A network failure
/// stops the pass (the rest retry next launch). A 4xx deletes the file —
/// the server rejected it and retrying forever would wedge the queue —
/// except 401, which is a rotated key: keep it for a build that has the new one.
fn flush_pending() -> (usize, Option<String>) {
    let Some(dir) = outbox() else { return (0, None) };
    let mut files = list(&dir, "json");
    files.sort_by_key(|p| mtime_secs(p));
    let is_session = |p: &PathBuf| p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("session-"));
    // Crash-loop cap: drop the oldest crash files beyond CRASH_KEEP.
    let crashes: Vec<&PathBuf> = files.iter().filter(|p| !is_session(p)).collect();
    let drop_n = crashes.len().saturating_sub(CRASH_KEEP);
    let doomed: Vec<PathBuf> = crashes.into_iter().take(drop_n).cloned().collect();
    for p in &doomed {
        let _ = fs::remove_file(p);
        let _ = fs::remove_file(p.with_extension("dmp"));
    }
    files.retain(|p| !doomed.contains(p));
    // A minidump whose report is gone (sent, or capped) is no use to anyone.
    for d in list(&dir, "dmp") {
        if !d.with_extension("json").exists() {
            let _ = fs::remove_file(d);
        }
    }
    let mut sent = 0;
    for f in files {
        let session = is_session(&f);
        if session && !CONSENT_USAGE.load(Ordering::SeqCst) {
            let _ = fs::remove_file(&f);
            continue;
        }
        if !session && !CONSENT_CRASH.load(Ordering::SeqCst) {
            continue;
        }
        let Ok(mut body) = fs::read_to_string(&f) else { continue };
        if !session {
            if let Some(b) = settle_native(&f, &body) {
                body = b;
            }
        }
        match post(if session { "/v1/session" } else { "/v1/report" }, &body) {
            Ok(id) => {
                if !session {
                    sent += 1;
                    send_dump(&f, &body, &id);
                }
                let _ = fs::remove_file(&f);
            }
            Err(e) => {
                if e.starts_with("HTTP 4") && !e.starts_with("HTTP 401") {
                    let _ = fs::remove_file(&f);
                    continue;
                }
                return (sent, Some(e));
            }
        }
    }
    (sent, None)
}

/// The report's minidump (if it has one) to `/v1/dump/<id>`. Best effort:
/// the report itself already landed, and the dump goes either way.
fn send_dump(report: &Path, body: &str, id: &str) {
    let Some(name) = serde_json::from_str::<Report>(body).ok().map(|r| r.dump).filter(|d| !d.is_empty()) else {
        return;
    };
    let Some(path) = report.parent().map(|d| d.join(&name)) else { return };
    match fs::read(&path) {
        Ok(bytes) if bytes.len() <= 48 << 20 => {
            if let Err(e) = put_bytes(&format!("/v1/dump/{id}"), &bytes) {
                log::warn!("minidump upload failed: {e}");
            }
        }
        _ => {}
    }
    let _ = fs::remove_file(path);
}

/// An unclean exit still waiting on its OS crash record: look again,
/// giving the OS until ~30 s after we adopted the session (this runs on
/// the upload thread, so waiting costs nobody anything). Either way the
/// report is final afterwards. Returns the rewritten body when it changed.
fn settle_native(f: &Path, body: &str) -> Option<String> {
    let mut r: Report = serde_json::from_str(body).ok()?;
    if r.native_pid == 0 {
        return None;
    }
    let ready_at = mtime_secs(f) + 30;
    let mut n = native_crash(r.native_pid, r.native_after);
    if n.is_none() && unix_secs() < ready_at {
        std::thread::sleep(Duration::from_secs(ready_at - unix_secs()));
        n = native_crash(r.native_pid, r.native_after);
    }
    match &n {
        Some(n) => apply_native(&mut r, n),
        None => r.native_pid = 0,
    }
    let out = serde_json::to_string_pretty(&r).ok()?;
    let _ = write_atomic(f, out.as_bytes());
    Some(out)
}

/// Kick off the background pass, as far as consent allows: with alpha
/// testing joined, send the outbox; without it, only tidy it locally (the
/// same pass skips every upload, so it touches no network); with update
/// checks on, ask whether a newer build exists. Safe to call again after a
/// consent change — each part runs once per process.
pub fn start_uploads() {
    let testing = CONSENT_CRASH.load(Ordering::SeqCst) && endpoint().is_some();
    let flush = if testing { !FLUSHED.swap(true, Ordering::SeqCst) } else { !PRUNED.swap(true, Ordering::SeqCst) };
    let check = CONSENT_UPDATES.load(Ordering::SeqCst) && !CHECKED.swap(true, Ordering::SeqCst);
    if !flush && !check {
        return;
    }
    std::thread::Builder::new()
        .name("report-upload".into())
        .spawn(move || {
            if flush {
                match flush_pending() {
                    (0, None) => {}
                    (n, None) => {
                        let s = format!("{n} crash report{} from earlier sessions sent", if n == 1 { "" } else { "s" });
                        log::info!("{s}");
                        set_status(s);
                    }
                    (n, Some(e)) => {
                        let s = format!("report upload stopped after {n}: {e} — retrying next launch");
                        log::warn!("{s}");
                        set_status(s);
                    }
                }
            }
            if check {
                check_latest();
            }
        })
        .ok();
}

// ------------------------------------------------------------ new builds

/// The newest PUBLIC build, from the download worker. No key, no identity,
/// nothing in the request but the platform.
const LATEST_URL: &str = "https://download.wobble.ing/latest.json";

#[derive(serde::Deserialize, Clone, Default)]
pub struct Latest {
    pub git: String,
    /// When it was published (unix seconds).
    pub stamp: u64,
    pub version: String,
    /// Where to get it: the download page.
    pub url: String,
    #[serde(default)]
    pub notes: String,
}

static UPDATE: Mutex<Option<Latest>> = Mutex::new(None);
static INSTALL_ID: Mutex<Option<String>> = Mutex::new(None);

/// The random install id the new-build check sends (`None` = send none).
pub fn set_install_id(id: Option<String>) {
    if let Ok(mut i) = INSTALL_ID.lock() {
        *i = id;
    }
}

/// Where "get it" goes: the download page, told which version this is so
/// an upgrade can be counted apart from a new user.
pub fn download_url(l: &Latest) -> String {
    let sep = if l.url.contains('?') { '&' } else { '?' };
    format!("{}{sep}from={VERSION}", l.url)
}

/// A published build newer than this one. Dev builds (a newer commit than
/// anything published) never see one.
pub fn update_available() -> Option<Latest> {
    UPDATE.try_lock().ok().and_then(|u| u.clone())
}

/// Builds are per platform, named like Rust's `os-arch` (macos-aarch64).
fn latest_url() -> String {
    let base = std::env::var("WOBBLE_UPDATE_URL").ok().filter(|s| !s.is_empty());
    let id = INSTALL_ID.lock().ok().and_then(|i| i.clone());
    format!(
        "{}?platform={}-{}&v={VERSION}{}",
        base.as_deref().unwrap_or(LATEST_URL),
        std::env::consts::OS,
        std::env::consts::ARCH,
        id.map(|i| format!("&id={i}")).unwrap_or_default()
    )
}

#[cfg(target_arch = "wasm32")]
fn fetch_latest() -> Result<(u16, String), String> {
    Err("no update check from the web build".into())
}

#[cfg(not(target_arch = "wasm32"))]
fn fetch_latest() -> Result<(u16, String), String> {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(20)))
        .http_status_as_error(false)
        .build()
        .into();
    let mut resp = agent.get(&latest_url()).call().map_err(|e| format!("{e}"))?;
    let status = resp.status().as_u16();
    Ok((status, resp.body_mut().read_to_string().unwrap_or_default()))
}

/// SemVer precedence, pre-releases included (0.1.0-alpha.2 < 0.1.0-beta.1
/// < 0.1.0). `None` when either is not a version.
pub(crate) fn semver_cmp(a: &str, b: &str) -> Option<std::cmp::Ordering> {
    fn parse(v: &str) -> Option<(Vec<u64>, Vec<&str>)> {
        let v = v.trim().trim_start_matches('v');
        let v = v.split('+').next()?;
        let (core, pre) = match v.split_once('-') {
            Some((c, p)) => (c, p.split('.').collect()),
            None => (v, Vec::new()),
        };
        let core: Vec<u64> = core.split('.').map(|n| n.parse().ok()).collect::<Option<_>>()?;
        (core.len() == 3).then_some((core, pre))
    }
    let ((ca, pa), (cb, pb)) = (parse(a)?, parse(b)?);
    use std::cmp::Ordering::*;
    Some(ca.cmp(&cb).then_with(|| match (pa.is_empty(), pb.is_empty()) {
        (true, true) => Equal,
        (true, false) => Greater,
        (false, true) => Less,
        (false, false) => {
            for (x, y) in pa.iter().zip(&pb) {
                let o = match (x.parse::<u64>(), y.parse::<u64>()) {
                    (Ok(x), Ok(y)) => x.cmp(&y),
                    (Ok(_), Err(_)) => Less,
                    (Err(_), Ok(_)) => Greater,
                    _ => x.cmp(y),
                };
                if o != Equal {
                    return o;
                }
            }
            pa.len().cmp(&pb.len())
        }
    }))
}

/// Whether `l` is a build this one should be told about: a higher version.
/// A manifest without one falls back to "published later, different commit".
fn is_newer(l: &Latest, mine_version: &str, mine_git: &str, mine_stamp: u64) -> bool {
    if let Some(o) = semver_cmp(&l.version, mine_version) {
        return o == std::cmp::Ordering::Greater;
    }
    let mine = mine_git.trim_end_matches("-dirty");
    l.stamp > mine_stamp && !l.git.is_empty() && !mine.starts_with(&l.git) && !l.git.starts_with(mine)
}

fn check_latest() {
    let Ok((200, text)) = fetch_latest() else { return };
    let Ok(l) = serde_json::from_str::<Latest>(&text) else { return };
    if is_newer(&l, VERSION, GIT, stamp()) {
        log::info!("newer build available: {} ({})", l.version, l.git);
        if let Ok(mut u) = UPDATE.lock() {
            *u = Some(l);
        }
    }
}

// -------------------------------------------------------------- autosave

/// Periodic comp snapshots for crash recovery. The caller owns the comp:
/// each frame it asks `due()`, and when told, serialises and calls `store`.
/// Nothing is written while the comp matches the last save/open (the
/// baseline), so recovery never offers back what is already on disk.
pub struct Autosave {
    enabled: bool,
    next: Instant,
    next_beat: Instant,
    baseline: Option<u64>,
    written: Option<u64>,
}

impl Autosave {
    pub fn new(enabled: bool) -> Self {
        let now = Instant::now();
        Autosave { enabled, next: now + AUTOSAVE_EVERY, next_beat: now, baseline: None, written: None }
    }

    fn files(&self) -> Option<(PathBuf, PathBuf)> {
        let s = SESSION.get()?;
        let d = sub("autosave")?;
        Some((d.join(format!("{}.wbl", s.sid)), d.join(format!("{}.meta.json", s.sid))))
    }

    pub fn due(&mut self) -> bool {
        if !self.enabled || SESSION.get().is_none() {
            return false;
        }
        let now = Instant::now();
        if now >= self.next_beat {
            self.next_beat = now + HEARTBEAT_EVERY;
            heartbeat();
        }
        if now >= self.next {
            self.next = now + AUTOSAVE_EVERY;
            return true;
        }
        false
    }

    fn forget(&mut self) {
        if self.written.take().is_some() {
            if let Some((a, m)) = self.files() {
                let _ = fs::remove_file(a);
                let _ = fs::remove_file(m);
            }
        }
    }

    /// The comp as it is on disk right now (just saved or opened).
    pub fn baseline(&mut self, json: &str) {
        self.baseline = Some(fnv(json));
        self.forget();
    }

    /// Whether `json` is the comp as it was last saved or opened. No
    /// baseline (a recovered comp) is never "as saved".
    pub fn matches(&self, json: &str) -> bool {
        self.baseline == Some(fnv(json))
    }

    /// The comp in memory is NOT what is on disk (a recovered comp).
    pub fn unbaseline(&mut self) {
        self.baseline = None;
    }

    pub fn store<'a>(&mut self, json: String, original: Option<&Path>, kinds: impl Iterator<Item = &'a str>) {
        if let Ok(mut st) = STATS.lock() {
            let mut n = 0;
            for k in kinds {
                n += 1;
                if !st.0.contains(k) {
                    st.0.insert(k.to_string());
                }
            }
            st.1 = st.1.max(n);
        }
        let h = fnv(&json);
        if self.baseline == Some(h) {
            self.forget();
            return;
        }
        if self.written == Some(h) {
            return;
        }
        let Some((a, m)) = self.files() else { return };
        if let Some(d) = a.parent() {
            let _ = fs::create_dir_all(d);
        }
        let meta = AutosaveMeta { original: original.map(Path::to_path_buf), saved: unix_secs() };
        if write_atomic(&a, json.as_bytes()).is_ok() {
            if let Ok(j) = serde_json::to_vec(&meta) {
                let _ = write_atomic(&m, &j);
            }
            self.written = Some(h);
        }
    }
}

// ------------------------------------------------------------ recovery UI

pub enum RecoveryAction {
    Wait,
    /// Load this autosave; afterwards `current_setup` = `Recovery::original`.
    Restore(PathBuf),
    Discard,
}

fn ago(secs: u64) -> String {
    let d = unix_secs().saturating_sub(secs);
    match d {
        0..=89 => "just now".into(),
        90..=5399 => format!("{} min ago", (d + 30) / 60),
        5400..=129_599 => format!("{} h ago", (d + 1800) / 3600),
        _ => format!("{} days ago", (d + 43_200) / 86_400),
    }
}

pub fn recovery_window(ctx: &egui::Context, r: &Recovery) -> RecoveryAction {
    let mut act = RecoveryAction::Wait;
    egui::Window::new("recover your comp?")
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .show(ctx, |ui| {
            ui.label(format!("Wobble's last session ended badly: {}.", r.how));
            let name = r
                .original
                .as_ref()
                .and_then(|p| p.file_name())
                .map_or("an unsaved comp".to_string(), |n| n.to_string_lossy().into_owned());
            ui.label(format!("{name} was autosaved {}.", ago(r.saved)));
            if let Some(p) = &r.original {
                ui.weak(format!("restoring keeps it unsaved — save to write it over {}", p.display()));
            }
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                if ui.button("restore").clicked() {
                    act = RecoveryAction::Restore(r.autosave.clone());
                }
                if ui.button("discard").clicked() {
                    act = RecoveryAction::Discard;
                }
            });
        });
    act
}

/// What the new-build popup was answered with. Either answer retires it
/// for that version; the menubar's "new build" button stays.
pub enum UpdateAction {
    Wait,
    Get,
    Later,
}

/// The launch popup for a newer published build (`update_available`).
pub fn update_window(ctx: &egui::Context, l: &Latest) -> UpdateAction {
    let mut act = UpdateAction::Wait;
    egui::Window::new("a new build of wobble is out")
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .show(ctx, |ui| {
            ui.label(format!("{} is out — this is {VERSION}.", l.version));
            if !l.notes.is_empty() {
                ui.weak(&l.notes);
            }
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                if ui.button("get it ↗").on_hover_text(&l.url).clicked() {
                    act = UpdateAction::Get;
                }
                if ui.button("later").on_hover_text("the menubar's \"new build\" button stays").clicked() {
                    act = UpdateAction::Later;
                }
            });
        });
    act
}

/// The decision is made either way: the autosave has served its purpose.
pub fn recovery_finish(r: &Recovery) {
    let _ = fs::remove_file(&r.autosave);
    let _ = fs::remove_file(r.autosave.with_extension("meta.json"));
}

// -------------------------------------------------------------- dev hooks

/// `WOBBLE_CRASH_TEST=1 wobble` panics before a window exists — exercises
/// the hook + file path end to end without stealing focus.
/// `WOBBLE_CRASH_TEST=abort` dies natively instead (no panic hook), which
/// exercises unclean-exit detection and the OS-record lookup on the next
/// launch; `=segv` is a real access violation (the Windows exception filter).
pub fn maybe_crash_for_test() {
    match std::env::var("WOBBLE_CRASH_TEST").as_deref() {
        Ok("abort") => {
            log::warn!("deliberate native abort (WOBBLE_CRASH_TEST=abort)");
            std::process::abort();
        }
        // A real access violation: the Windows exception filter's path (and
        // macOS's .ips path for an EXC_BAD_ACCESS).
        Ok("segv") => {
            log::warn!("deliberate access violation (WOBBLE_CRASH_TEST=segv)");
            // SAFETY: deliberately not — this write IS the crash under test.
            unsafe { std::ptr::without_provenance_mut::<u8>(16).write_volatile(1) };
        }
        Ok(_) => panic!("deliberate test crash (WOBBLE_CRASH_TEST)"),
        Err(_) => {}
    }
}

pub fn selftest_requested() -> bool {
    std::env::var("WOBBLE_REPORT_SELFTEST").is_ok()
}

/// `WOBBLE_REPORT_SELFTEST=1 wobble` validates the whole pipeline from a
/// tester's machine: one direct send, a synchronous flush of the outbox and
/// a latest-build query. Prints PASS/FAIL and exits — no window.
pub fn run_selftest() -> ! {
    log::warn!("report selftest marker");
    let r = Report::new("test", "report self-test".into(), "sent by WOBBLE_REPORT_SELFTEST".into(), ring_tail());
    match send(&r) {
        Ok(id) => println!("REPORT SELFTEST send PASS ({id})"),
        Err(e) => {
            println!("REPORT SELFTEST send FAIL: {e}");
            std::process::exit(1);
        }
    }
    let (sent, err) = flush_pending();
    if let Some(e) = err {
        println!("REPORT SELFTEST flush FAIL after {sent}: {e}");
        std::process::exit(1);
    }
    println!("REPORT SELFTEST flush PASS ({sent} pending sent)");
    match fetch_latest() {
        Ok((s, t)) if s == 200 || s == 204 => {
            println!("REPORT SELFTEST latest PASS ({s} {})", cap_str(t, 160));
            std::process::exit(0);
        }
        Ok((s, t)) => println!("REPORT SELFTEST latest FAIL: HTTP {s} {}", cap_str(t, 160)),
        Err(e) => println!("REPORT SELFTEST latest FAIL: {e}"),
    }
    std::process::exit(1);
}

// ------------------------------------------------------------ feedback UI

#[derive(Clone, Copy, PartialEq)]
pub enum Kind {
    Bug,
    Feature,
    Node,
    /// Anything at all: one message, optional email, "complain".
    General,
}

impl Kind {
    fn label(self) -> &'static str {
        match self {
            Kind::Bug => "report a bug",
            Kind::Feature => "request a feature",
            Kind::Node => "request a node",
            Kind::General => "feedback",
        }
    }
    /// The Worker's kind. A node request is a feature request with a title
    /// prefix, so it needs no Worker change and lands in the same list.
    fn wire(self) -> &'static str {
        match self {
            Kind::Bug => "bug",
            Kind::Feature | Kind::Node | Kind::General => "feature",
        }
    }
    fn title_prefix(self) -> &'static str {
        match self {
            Kind::Node => "[node] ",
            Kind::General => "[feedback] ",
            _ => "",
        }
    }
}

pub struct Feedback {
    pub open: bool,
    /// The first-run window is up.
    pub notice: bool,
    /// First launch: the welcome page comes before the consent page.
    pub welcome: bool,
    /// First launch: the quick start opens once these windows are done.
    pub quickstart_after: bool,
    /// The "what is wobble" window is up.
    pub about: bool,
    /// "request a feature" was picked: the gate in front of the form is up.
    pub gate: bool,
    pub tester: String,
    pub email: String,
    /// Legacy alpha invite code: kept and sent, no longer asked for.
    pub code: String,
    /// Alpha testing joined: crash reports, logs and usage stats upload.
    pub testing: bool,
    /// Ask whether a newer build is out.
    pub updates: bool,
    kind: Kind,
    title: String,
    body: String,
    include_logs: bool,
    include_setup: bool,
    inflight: Option<std::sync::mpsc::Receiver<Result<String, String>>>,
    last: Option<Result<String, String>>,
    /// The first-run window tried to continue without the name it needs.
    notice_tried: bool,
}

impl Feedback {
    pub fn new(tester: String, email: String, code: String, testing: bool, updates: bool, notice: bool) -> Self {
        Feedback {
            open: false,
            notice,
            welcome: false,
            quickstart_after: false,
            about: false,
            gate: false,
            tester,
            email,
            code,
            testing,
            updates,
            kind: Kind::Bug,
            title: String::new(),
            body: String::new(),
            include_logs: false,
            include_setup: false,
            inflight: None,
            last: None,
            notice_tried: false,
        }
    }

    /// First launch: show the welcome page first, and say whether the quick
    /// start waits for these windows.
    pub fn first_launch(mut self, welcome: bool, quickstart: bool) -> Self {
        self.welcome = welcome && self.notice;
        self.quickstart_after = quickstart && self.notice;
        // A look at the about window under a suite (WOBBLE_SHOT).
        self.about = std::env::var("WOBBLE_SHOW_CONSENT").as_deref() == Ok("about");
        self
    }

    /// A help-menu entry: open the form as that kind. A feature request
    /// goes through the gate first.
    pub fn open_as(&mut self, kind: Kind) {
        if self.kind != kind {
            self.last = None;
        }
        self.kind = kind;
        if kind == Kind::Feature {
            self.open = false;
            self.gate = true;
        } else {
            self.gate = false;
            self.open = true;
        }
    }

    /// Whether the pending send wants the comp attached (the caller has it).
    pub fn wants_setup(&self) -> bool {
        self.kind == Kind::Bug && self.include_setup
    }

    fn push(&self) {
        set_tester(&self.tester, &self.email, &self.code);
        set_consent(self.testing, self.updates);
    }
}

#[derive(Default)]
pub struct FeedbackEvent {
    /// Name / email / consent changed: persist them.
    pub prefs_changed: bool,
    /// Send pressed: call `feedback_send` with the comp if `wants_setup`.
    pub send: bool,
}

fn looks_like_email(s: &str) -> bool {
    let s = s.trim();
    match s.split_once('@') {
        Some((a, b)) => !a.is_empty() && b.contains('.') && !b.starts_with('.') && !b.ends_with('.') && !s.contains(' '),
        None => false,
    }
}

/// What alpha testing needs before it can start: an email, or failing
/// that a name, so the reports can be told apart. `None` = good to go.
fn identity_problem(fb: &Feedback) -> Option<&'static str> {
    if !fb.testing {
        return None;
    }
    let (email, name) = (fb.email.trim(), fb.tester.trim());
    if !email.is_empty() && !looks_like_email(email) {
        return Some("that doesn't look like an email address");
    }
    if email.is_empty() && name.is_empty() {
        return Some("no email? then pick a name, so your reports can be told apart");
    }
    None
}

fn email_name_rows(ui: &mut egui::Ui, fb: &mut Feedback, email_hint: &str) -> bool {
    let mut changed = false;
    egui::Grid::new(ui.next_auto_id()).num_columns(2).show(ui, |ui| {
        ui.label("email");
        changed |= ui
            .add(egui::TextEdit::singleline(&mut fb.email).hint_text(email_hint).desired_width(220.0))
            .changed();
        ui.end_row();
        ui.label("name");
        changed |= ui
            .add(egui::TextEdit::singleline(&mut fb.tester).hint_text("or a handle").desired_width(220.0))
            .changed();
        ui.end_row();
    });
    changed
}

const TESTING_HOVER: &str = "crash reports (what crashed, where in the code, the app's recent log, your OS + GPU, \
     and the comp you had open — file paths, never footage), and per session: how long, whether it closed \
     cleanly, which node types were used";

/// Alpha testing + new-build checks: the same switches the first-run
/// window offers. Preferences shows this. Returns whether anything changed.
pub fn sharing_ui(ui: &mut egui::Ui, fb: &mut Feedback) -> bool {
    let mut changed = false;
    changed |= ui
        .checkbox(&mut fb.testing, "alpha testing — share crash reports, logs and usage")
        .on_hover_text(TESTING_HOVER)
        .changed();
    if fb.testing {
        ui.indent("alpha_identity", |ui| {
            changed |= email_name_rows(ui, fb, "optional");
            if let Some(p) = identity_problem(fb) {
                ui.colored_label(ui.visuals().warn_fg_color, p);
            }
        });
    }
    changed |= ui
        .checkbox(&mut fb.updates, "tell me when a new build is out")
        .on_hover_text("asks download.wobble.ing at launch — anonymous, nothing about you is sent")
        .changed();
    ui.weak("off means off: nothing is uploaded, no request is made. no footage or images ever leave your machine.");
    if changed {
        fb.push();
    }
    changed
}

/// The first-run window: two choices, both changeable later in preferences.
/// Returns true when the user is done with it (the caller persists the
/// fields and calls `start_uploads`).
pub fn alpha_notice(ctx: &egui::Context, fb: &mut Feedback) -> bool {
    if fb.welcome {
        welcome(ctx, fb);
        return false;
    }
    let mut done = false;
    egui::Window::new("welcome to wobble")
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .default_width(440.0)
        .show(ctx, |ui| {
            ui.label("Wobble is in alpha, so things will break. Two optional things:");
            ui.add_space(8.0);
            ui.checkbox(&mut fb.testing, egui::RichText::new("join alpha testing").strong())
                .on_hover_text(TESTING_HOVER);
            ui.indent("notice_testing", |ui| {
                ui.weak(
                    "crashes are caught and sent the next time you open wobble, with the app's \
                     recent log and the comp you had open; plus how long sessions last and which \
                     nodes get used.",
                );
                if fb.testing {
                    ui.add_space(4.0);
                    email_name_rows(ui, fb, "optional — so we can follow up");
                    if fb.notice_tried {
                        if let Some(p) = identity_problem(fb) {
                            ui.colored_label(ui.visuals().warn_fg_color, p);
                        }
                    }
                }
            });
            ui.add_space(6.0);
            ui.checkbox(&mut fb.updates, egui::RichText::new("tell me when a new build is out").strong());
            ui.indent("notice_updates", |ui| {
                ui.weak("a quick, anonymous check at launch.");
            });
            ui.add_space(8.0);
            ui.weak("no footage or images ever leave your machine. change either in preferences, any time.");
            ui.weak("bugs and wishes: the help menu — report a bug, request a feature, request a node.");
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                if ui.button("i guess").clicked() {
                    fb.notice_tried = true;
                    if identity_problem(fb).is_none() {
                        done = true;
                    }
                }
                if ui.button("no").on_hover_text("no testing, no update checks").clicked() {
                    fb.testing = false;
                    fb.updates = false;
                    done = true;
                }
            });
        });
    if done {
        fb.push();
        fb.notice = false;
    }
    done
}

/// First launch, before the consent page: what this is.
fn welcome(ctx: &egui::Context, fb: &mut Feedback) {
    egui::Window::new("welcome")
        .collapsible(false)
        .resizable(false)
        .title_bar(false)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .default_width(420.0)
        .show(ctx, |ui| {
            ui.set_max_width(420.0);
            ui.add_space(6.0);
            ui.label(
                egui::RichText::new(format!(
                    "This is wobble version {VERSION}. This is alpha software! Expect issues, and please \
                     report them. Nodes that show as red in the menu are alpha, green is beta."
                ))
                .size(15.0),
            );
            ui.add_space(10.0);
            ui.horizontal(|ui| {
                if ui.button("what is wobble").clicked() {
                    fb.about = true;
                }
                if ui.button(egui::RichText::new("close").strong()).clicked() {
                    fb.welcome = false;
                }
            });
        });
}

const WHAT_IS_WOBBLE: &str = "https://wobble.ing";

/// The "what is wobble" window, opened from the welcome page.
pub fn what_is_wobble(ctx: &egui::Context, fb: &mut Feedback) {
    if !fb.about {
        return;
    }
    let mut open = true;
    let mut close = false;
    egui::Window::new("what is wobble")
        .open(&mut open)
        .collapsible(false)
        .resizable(false)
        .default_width(520.0)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .show(ctx, |ui| {
            ui.set_max_width(520.0);
            for para in [
                "Wobble is a GPU first, free and open source compositing app.",
                "In most cases the node tree is compiled into a single WGPU compute shader and executed \
                 on the GPU. If you have a decent GPU, this means that many comps can be run at or near \
                 realtime, and the bottleneck is loading in the images.",
                "Some nodes will break it, and you can bake at any point in the graph.",
                "It caches your media, optimized for streaming from a fast SSD. If you want the true \
                 wobbly experience, give it a decent amount of dedicated NVME to play with, the faster \
                 the better. It can also use ram, if you have a lot of that.",
                "On adequate hardware and within reason, once inputs are cached, your comp will be fully \
                 real time in most cases. Especially at 4K and lower.",
                "The ethos is GPU speed and caching as a priority. This means accommodating GPU texture \
                 conventions which has informed some design decisions which may seem counter intuitive \
                 if you are used to other compositing apps:",
            ] {
                ui.label(para);
                ui.add_space(6.0);
            }
            for item in [
                "16 bit float tree by default",
                "exr layers are supported but not transmitted throughout the comp by default",
                "matte and data (normal, depth) pipes have their own data types, need converters",
                "no deep",
            ] {
                ui.horizontal_wrapped(|ui| {
                    ui.label("•");
                    ui.label(item);
                });
            }
            ui.add_space(6.0);
            ui.label(
                "If you need things like deep that conflict with that, then for now at least, wobble \
                 isn\u{2019}t the app for you!",
            );
            ui.add_space(10.0);
            ui.horizontal(|ui| {
                if ui.button(egui::RichText::new("close").strong()).clicked() {
                    close = true;
                }
                ui.hyperlink_to("wobble.ing", WHAT_IS_WOBBLE);
            });
        });
    if !open || close {
        fb.about = false;
    }
}

/// The line in front of the feature-request form. "oh" closes it;
/// the other button carries on to the form.
pub fn feature_gate(ctx: &egui::Context, fb: &mut Feedback) {
    if !fb.gate {
        return;
    }
    egui::Window::new("request a feature")
        .collapsible(false)
        .resizable(false)
        .title_bar(false)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .show(ctx, |ui| {
            ui.add_space(6.0);
            ui.label(egui::RichText::new("I am not going to make it more like that other compositing app").size(16.0));
            ui.add_space(10.0);
            ui.horizontal(|ui| {
                if ui.button("oh").clicked() {
                    fb.gate = false;
                }
                if ui.button("that isn't what i was going to say").clicked() {
                    fb.gate = false;
                    fb.open = true;
                }
            });
        });
}

/// Build and send the report the window asked for (`FeedbackEvent::send`).
pub fn feedback_send(fb: &mut Feedback, setup: Option<String>) {
    set_tester(&fb.tester, &fb.email, &fb.code);
    let bug = fb.kind == Kind::Bug;
    let mut report = Report::new(
        fb.kind.wire(),
        format!("{}{}", fb.kind.title_prefix(), fb.title.trim()),
        fb.body.trim().into(),
        if fb.include_logs && bug { ring_tail() } else { String::new() },
    );
    if bug && fb.include_setup {
        report.setup = cap_str(setup.unwrap_or_default(), SETUP_CAP);
    }
    let (tx, rx) = std::sync::mpsc::channel();
    fb.inflight = Some(rx);
    fb.last = None;
    std::thread::Builder::new()
        .name("feedback-send".into())
        .spawn(move || {
            let _ = tx.send(send(&report));
        })
        .ok();
}

/// The feedback form: a bug, a feature or a node. Sending one is its own
/// consent — it works whether or not alpha testing is on.
pub fn feedback_window(ctx: &egui::Context, fb: &mut Feedback) -> FeedbackEvent {
    let mut ev = FeedbackEvent::default();
    if let Some(rx) = &fb.inflight {
        match rx.try_recv() {
            Ok(res) => {
                if res.is_ok() {
                    fb.title.clear();
                    fb.body.clear();
                }
                fb.last = Some(res);
                fb.inflight = None;
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                ctx.request_repaint_after(Duration::from_millis(150));
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                fb.last = Some(Err("send thread died".into()));
                fb.inflight = None;
            }
        }
    }
    let mut open = fb.open;
    egui::Window::new(fb.kind.label())
        .id(egui::Id::new("feedback_window"))
        .open(&mut open)
        .default_width(440.0)
        .show(ctx, |ui| {
            if fb.kind == Kind::General {
                general_form(ui, fb, &mut ev);
                return;
            }
            ui.add(
                egui::TextEdit::singleline(&mut fb.title)
                    .hint_text(match fb.kind {
                        Kind::Node => "the node, in a few words",
                        _ => "one-line summary",
                    })
                    .desired_width(f32::INFINITY),
            );
            ui.add(
                egui::TextEdit::multiline(&mut fb.body)
                    .hint_text(match fb.kind {
                        Kind::Bug => "what you did, what you expected, what happened instead",
                        Kind::Feature => "what you want to do, and what gets in the way today",
                        Kind::Node => "what it should do, its inputs and outputs, and a reference if there is one",
                        Kind::General => "anything at all",
                    })
                    .desired_rows(8)
                    .desired_width(f32::INFINITY),
            );
            ui.horizontal(|ui| {
                if fb.kind == Kind::Bug {
                    ui.checkbox(&mut fb.include_logs, "attach recent log")
                        .on_hover_text("the app's recent activity and warnings — helps a lot with bugs");
                    ui.checkbox(&mut fb.include_setup, "attach comp")
                        .on_hover_text("the comp as it is now (node graph + settings + file paths, never footage) — usually enough to reproduce");
                }
                let can = fb.inflight.is_none() && !fb.title.trim().is_empty();
                if ui
                    .add_enabled(can, egui::Button::new("send"))
                    .on_hover_text("goes straight to the developer")
                    .clicked()
                {
                    ev.send = true;
                }
                if fb.inflight.is_some() {
                    ui.weak("sending…");
                }
                match &fb.last {
                    Some(Ok(_)) => {
                        ui.weak("sent ✔ thank you");
                    }
                    Some(Err(e)) => {
                        ui.colored_label(theme::error(), e);
                    }
                    None => {}
                }
            });
            ui.separator();
            ui.weak("who it's from — optional, leave both blank to stay anonymous");
            if email_name_rows(ui, fb, "so we can reply") {
                set_tester(&fb.tester, &fb.email, &fb.code);
                ev.prefs_changed = true;
            }
            ui.weak(format!("build {VERSION} ({GIT})"));
            if let Some(s) = status() {
                ui.weak(s);
            }
        });
    fb.open = open && fb.open;
    ev
}

/// The general form: a message, an optional email, cancel or complain.
fn general_form(ui: &mut egui::Ui, fb: &mut Feedback, ev: &mut FeedbackEvent) {
    ui.add(
        egui::TextEdit::multiline(&mut fb.body)
            .hint_text("anything at all")
            .desired_rows(8)
            .desired_width(f32::INFINITY),
    );
    ui.horizontal(|ui| {
        ui.label("email");
        if ui
            .add(egui::TextEdit::singleline(&mut fb.email).hint_text("optional — so we can reply").desired_width(240.0))
            .changed()
        {
            set_tester(&fb.tester, &fb.email, &fb.code);
            ev.prefs_changed = true;
        }
    });
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        if ui.button("cancel").clicked() {
            fb.open = false;
            fb.last = None;
        }
        let can = fb.inflight.is_none() && !fb.body.trim().is_empty();
        if ui.add_enabled(can, egui::Button::new("complain")).clicked() {
            // The report wants a title: the message's first line.
            let first = fb.body.trim().lines().next().unwrap_or("").to_string();
            fb.title = cap_str(first, 80);
            ev.send = true;
        }
        if fb.inflight.is_some() {
            ui.weak("sending…");
        }
        match &fb.last {
            Some(Ok(_)) => {
                ui.weak("sent ✔ thank you");
            }
            Some(Err(e)) => {
                ui.colored_label(theme::error(), e);
            }
            None => {}
        }
    });
}

// ------------------------------------------------------------------ tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_report_is_tagged_with_whatever_the_user_gave() {
        set_tester(" sam ", "", "");
        assert_eq!(tester().0, "sam");
        set_tester("", "a@b.co", "");
        assert_eq!(tester().0, "a@b.co");
        set_tester("sam", "a@b.co", "wob-1");
        assert_eq!(tester(), ("sam <a@b.co>".into(), "wob-1".into()));
        set_tester("", "", "");
        assert_eq!(tester().0, "");
    }

    /// Alpha testing needs an email or, failing that, a name; neither
    /// matters until testing is on.
    #[test]
    fn testing_needs_an_email_or_a_name() {
        let mut fb = Feedback::new(String::new(), String::new(), String::new(), false, true, true);
        assert!(identity_problem(&fb).is_none());
        fb.testing = true;
        assert!(identity_problem(&fb).is_some());
        fb.tester = "sam".into();
        assert!(identity_problem(&fb).is_none());
        fb.email = "not an email".into();
        assert!(identity_problem(&fb).is_some());
        fb.email = "sam@example.com".into();
        fb.tester.clear();
        assert!(identity_problem(&fb).is_none());
    }

    #[test]
    fn the_update_check_and_download_link_say_which_version_this_is() {
        let l = Latest { url: "https://wobble.ing/download/".into(), ..Default::default() };
        assert_eq!(download_url(&l), format!("https://wobble.ing/download/?from={VERSION}"));
        set_install_id(Some("abc".into()));
        let u = latest_url();
        assert!(u.contains(&format!("&v={VERSION}")) && u.ends_with("&id=abc"), "{u}");
        set_install_id(None);
        assert!(!latest_url().contains("id="), "no id when there is none");
    }

    #[test]
    fn only_a_newer_published_build_is_offered() {
        let mut l = Latest { git: "abc123".into(), stamp: 200, version: "".into(), url: String::new(), notes: String::new() };
        // No version in the manifest: by publish time and commit.
        assert!(is_newer(&l, "0.1.0-alpha.1", "def456", 100));
        assert!(!is_newer(&l, "0.1.0-alpha.1", "abc123456789", 100), "this IS that build");
        assert!(!is_newer(&l, "0.1.0-alpha.1", "abc123-dirty", 100));
        assert!(!is_newer(&l, "0.1.0-alpha.1", "def456", 300), "a dev build newer than anything published");
        // With one: by version alone.
        l.version = "0.1.0-alpha.2".into();
        assert!(is_newer(&l, "0.1.0-alpha.1", "x", 999));
        assert!(!is_newer(&l, "0.1.0-alpha.2", "x", 0));
        assert!(!is_newer(&l, "0.1.0-beta.1", "x", 0));
    }

    #[test]
    fn versions_order_like_semver() {
        use std::cmp::Ordering::*;
        let order = ["0.1.0-alpha.1", "0.1.0-alpha.2", "0.1.0-alpha.10", "0.1.0-beta.1", "0.1.0", "0.1.1", "1.0.0"];
        for w in order.windows(2) {
            assert_eq!(semver_cmp(w[0], w[1]), Some(Less), "{} < {}", w[0], w[1]);
        }
        assert_eq!(semver_cmp("v0.1.0-alpha.1", "0.1.0-alpha.1"), Some(Equal));
        assert_eq!(semver_cmp("0.1", "0.1.0"), None);
    }

    /// The request feature gate sits in front of the form, and only there.
    #[test]
    fn feature_requests_pass_the_gate_first() {
        let mut fb = Feedback::new(String::new(), String::new(), String::new(), false, false, false);
        fb.open_as(Kind::Feature);
        assert!(fb.gate && !fb.open);
        fb.open_as(Kind::Node);
        assert!(!fb.gate && fb.open);
        assert_eq!((Kind::Node.wire(), Kind::Node.title_prefix()), ("feature", "[node] "));
    }

    /// The crash file IS the wire payload: what the hook writes must parse
    /// and carry the fields the Worker stores, or a crash uploads as junk.
    #[test]
    fn crash_file_is_the_wire_payload() {
        let r = Report::new("crash", "panic: boom".into(), "at x.rs:1:1".into(), "+0.1s WARN t: m".into());
        let json = serde_json::to_string_pretty(&r).unwrap();
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        for key in [
            "schema", "kind", "ts", "version", "git", "os", "gpu", "tester", "title", "body", "logs", "session",
            "install", "code", "sig",
        ] {
            assert!(v.get(key).is_some(), "missing {key}");
        }
        assert_eq!(v["kind"], "crash");
        assert_eq!(v["schema"], 2);
        // build.rs always bakes something in.
        assert!(!v["git"].as_str().unwrap().is_empty());
        // A schema-1 file (R's first version) still loads, so a queued old
        // crash can have a comp attached.
        let old = r#"{"schema":1,"kind":"crash","ts":1,"version":"0.1.0","git":"x","os":"o","gpu":"g","tester":"","title":"t","body":"b","logs":""}"#;
        assert!(serde_json::from_str::<Report>(old).is_ok());
    }

    /// Same message modulo numbers + same frame = same group; a different
    /// frame or cause = a different one.
    #[test]
    fn signatures_group_by_cause_not_by_numbers() {
        let a = sig_of("crash", "index out of bounds: the len is 3 but the index is 7", "wobble::Gfx::redraw");
        let b = sig_of("crash", "index out of bounds: the len is 12 but the index is 40", "wobble::Gfx::redraw");
        let c = sig_of("crash", "index out of bounds: the len is 3 but the index is 7", "engine::codegen::emit");
        assert_eq!(a, b);
        assert_ne!(a, c);
        let w1 = sig_of("crash", "wgpu error: Validation Error\n\nCaused by:\n  label = 'paint-bg' at 0x1f00", "f");
        let w2 = sig_of("crash", "wgpu error: Validation Error\n\nCaused by:\n  label = 'fused-bg' at 0x1f00", "f");
        assert_ne!(w1, w2, "wgpu's cause line must split the generic 'Validation Error'");
        assert_eq!(a.len(), 12);
    }

    #[test]
    fn backtrace_trim_skips_hook_machinery() {
        let bt = "   0: std::backtrace_rs::backtrace::libunwind::trace\n             at /rustc/abc/library/std/src/x.rs:1:1\n   \
                  1: wobble::report::install_panic_hook::{closure#0}\n   \
                  2: std::panicking::panic_with_hook\n   \
                  3: core::panicking::panic_fmt\n   \
                  4: <engine::graph::Graph>::emit\n             at /Users/someone/src/wobble/engine/src/graph.rs:10:5\n   \
                  5: <wobble::Gfx>::redraw\n   \
                  6: std::rt::lang_start_internal\n";
        let (t, first) = trim_backtrace(bt);
        assert!(t.starts_with("   4: <engine::graph::Graph>::emit"), "{t}");
        assert!(!t.contains("lang_start"));
        assert_eq!(first, "<engine::graph::Graph>::emit");
    }

    #[test]
    fn ips_summary_names_the_exception_and_our_frame() {
        let v: serde_json::Value = serde_json::from_str(
            r#"{"pid":42,"procName":"wobble","exception":{"type":"EXC_BAD_ACCESS","signal":"SIGSEGV","subtype":"KERN_INVALID_ADDRESS at 0x8"},
                "termination":{"indicator":"Segmentation fault: 11"},"asi":{"libsystem_c.dylib":["abort() called"]},
                "faultingThread":0,"threads":[{"name":"main","frames":[
                  {"imageIndex":1,"symbol":"__pthread_kill","symbolLocation":8},
                  {"imageIndex":0,"symbol":"_RNvNtCsfiKLhsWjRsE_3std7process5abort","symbolLocation":12},
                  {"imageIndex":2,"symbol":"AGXMetal_encode","symbolLocation":4},
                  {"imageIndex":0,"symbol":"_RNvNtCs1234_6engine10compositor6encode","symbolLocation":20}]}],
                "usedImages":[{"name":"wobble"},{"name":"libsystem_kernel.dylib"},{"name":"AGXMetalG16X"}]}"#,
        )
        .unwrap();
        let n = summarise_ips(&v);
        assert_eq!(n.kind, "crash");
        assert_eq!(n.title, "native crash: EXC_BAD_ACCESS SIGSEGV in engine::compositor::encode");
        assert!(n.summary.contains("abort() called"));
        assert!(n.summary.contains("std::process::abort"), "frames demangled: {}", n.summary);
    }

    /// What `wevtutil qe Application /f:xml` prints: one Application Error
    /// for our pid, one for someone else's, one hang, one older crash of a
    /// reused pid. Positions, not message text — that is localised.
    const WER_XML: &str = "<Event xmlns='http://schemas.microsoft.com/win/2004/08/events/event'><System><Provider Name='Application Error' Guid='{a0e9b465-b939-57d7-b27d-95d8e925ff57}'/><EventID Qualifiers='0'>1000</EventID><TimeCreated SystemTime='2026-10-01T12:00:05.1234567Z'/></System><EventData><Data>wobble.exe</Data><Data>0.1.0.0</Data><Data>6720a1b2</Data><Data>nvwgf2umx.dll</Data><Data>32.0.15.6094</Data><Data>66f1c2d3</Data><Data>c0000005</Data><Data>00000000001a2b3c</Data><Data>0x1f40</Data><Data>0x1db1234</Data><Data>C:\\Users\\t\\wobble\\wobble.exe</Data><Data>C:\\Windows\\System32\\DriverStore\\nvwgf2umx.dll</Data><Data>abc</Data><Data/><Data/></EventData></Event>\
<Event xmlns='x'><System><Provider Name='Application Error'/><EventID>1000</EventID><TimeCreated SystemTime='2026-10-01T12:00:09.0Z'/></System><EventData><Data>other.exe</Data><Data>1</Data><Data>2</Data><Data>x.dll</Data><Data>1</Data><Data>2</Data><Data>c0000409</Data><Data>0</Data><Data>0x1f40</Data></EventData></Event>\
<Event xmlns='x'><System><Provider Name='Application Hang'/><EventID Qualifiers='0'>1002</EventID><TimeCreated SystemTime='2026-10-01T13:00:00.0Z'/></System><EventData><Data>wobble.exe</Data><Data>0.1.0.0</Data><Data>2a3c</Data><Data>01db1234</Data><Data>4294967295</Data><Data>C:\\wobble.exe</Data><Data>id</Data><Data></Data><Data></Data><Data>Quiesce</Data></EventData></Event>\
<Event xmlns='x'><System><Provider Name='Application Error'/><EventID>1000</EventID><TimeCreated SystemTime='2026-09-20T08:00:00.0Z'/></System><EventData><Data>wobble.exe</Data><Data>1</Data><Data>2</Data><Data>old.dll</Data><Data>1</Data><Data>2</Data><Data>c00000fd</Data><Data>0</Data><Data>0x1f41</Data></EventData></Event>";

    #[test]
    fn wer_events_parse_and_match_by_exe_pid_and_time() {
        let ev = parse_wer_events(WER_XML);
        assert_eq!(ev.len(), 4);
        assert_eq!(ev[0].provider, "Application Error");
        assert_eq!(ev[0].id, 1000);
        assert_eq!(ev[0].data.len(), 15, "empty <Data/> still holds its position");
        assert_eq!(ev[0].data[10], "C:\\Users\\t\\wobble\\wobble.exe");
        assert_eq!(ev[2].id, 1002);
        let start = parse_iso_utc("2026-10-01T11:00:00Z").unwrap();

        let n = wer_match(&ev, "WOBBLE.EXE", 0x1f40, start).expect("our crash");
        assert_eq!(n.kind, "crash");
        assert_eq!(n.title, "native crash: 0xc0000005 access violation in nvwgf2umx.dll");
        assert!(n.summary.contains("32.0.15.6094") && n.summary.contains("00000000001a2b3c"), "{}", n.summary);

        let h = wer_match(&ev, "wobble.exe", 0x2a3c, start).expect("our hang");
        assert_eq!(h.kind, "hang");

        // pid 0x1f41 crashed — but before this session started: a reused pid.
        assert!(wer_match(&ev, "wobble.exe", 0x1f41, start).is_none());
        // other.exe crashed with our pid number; it is not us.
        assert!(wer_match(&ev[1..2], "wobble.exe", 0x1f40, start).is_none());
        assert!(wer_match(&ev, "wobble.exe", 0x9999, start).is_none());
    }

    #[test]
    fn iso_times_and_exception_names() {
        assert_eq!(parse_iso_utc("1970-01-01T00:00:00Z"), Some(0));
        assert_eq!(parse_iso_utc("2026-10-01T00:00:00.5Z"), Some(1_790_812_800));
        assert_eq!(parse_iso_utc("2000-02-29T12:00:00Z"), Some(951_825_600));
        assert_eq!(parse_iso_utc("garbage"), None);
        assert_eq!(code_name(0xC000_0005), "access violation");
        assert!(code_name(0xC000_0409).starts_with("fail-fast"));
    }

    /// Without an endpoint the send path must FAIL, not hang or panic —
    /// a build with no baked URL degrades to local crash files only.
    #[test]
    fn no_endpoint_is_a_clean_error() {
        if std::env::var("WOBBLE_REPORT_URL").is_ok() || !ENDPOINT.is_empty() {
            return; // deployed build: this test is about the unconfigured state
        }
        let r = Report::new("test", "t".into(), String::new(), String::new());
        assert!(send(&r).is_err());
    }
}
