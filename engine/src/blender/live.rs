//! The live half of the Blender node: an interactive Blender, launched with
//! the bridge script (`wobble_blender.py`), talking to wobble over a
//! localhost TCP socket and handing pixels back through the shared-memory
//! ring in [`super::shm`].
//!
//! Blender is the server (the script binds the port wobble chose), wobble
//! the client. One flat JSON object per line each way; the wire format is
//! documented at the top of the script. Nothing here blocks the UI: the
//! connect, read and write loops live on threads, a `frame` announcement is
//! turned into pixels ON the read thread (one memcpy out of the ring), and
//! the app drains [`Session::poll`] once a frame.
//!
//! Inbound JSON is parsed by a deliberately small flat-object parser rather
//! than a serde dependency: the engine core stays serde-free, and every
//! message Blender sends is one level deep.

use super::shm;
use anyhow::{Context, Result};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use web_time::{Duration, Instant};

/// Something Blender told us.
#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    /// The bridge is up. `w`/`h` are the scene's render size, `frame` its
    /// current frame, `start..=end` its range.
    Hello {
        version: String,
        w: u32,
        h: u32,
        fps: f32,
        blend: String,
        frame: i64,
        start: i64,
        end: i64,
    },
    /// Blender made (or remade) the ring. The app may import it for the
    /// GPU (`gpuimport`) and then `Session::set_gpu(true)`, after which
    /// frames arrive as `FrameRef` instead of `Frame`.
    Ring(shm::Source),
    /// A frame is in the ring, already copied out (the CPU path).
    Frame(shm::Frame),
    /// A frame is in slot `slot` of the imported ring (the GPU path): copy
    /// it on the GPU, then `Session::ack` the slot so Blender may reuse it.
    FrameRef { slot: usize, desc: shm::Desc },
    /// Blender's native module made a ring of IOSurfaces (`w` x `h`,
    /// rgba32f) it blits frames into; the app imports them
    /// (`gpuimport::import_surfaces`) and answers `Session::surfaces_ok`.
    Surfaces { ids: Vec<u32>, w: u32, h: u32 },
    /// A frame is in surface `slot` (its top-left `stamp.w` x `stamp.h`):
    /// copy it on the GPU, then `Session::sack` the slot.
    SurfaceFrame { slot: usize, seq: u64, stamp: shm::Stamp },
    /// A frame was announced but lapped before it could be copied, or the
    /// ring was not open. Counted, never shown.
    Dropped,
    /// The user moved Blender's own playhead.
    FrameChanged(i64),
    /// Where the eye is: the 3D view as the user left it (free mode), or
    /// the camera (camera mode) — sent on hello, on every scene change
    /// and whenever the user navigates in Blender.
    View(super::ViewState),
    /// The scene changed (an edit, not a frame change): `scene` is the new
    /// version, so frames stamped with an older one are stale.
    Scene(u64),
    /// The user saved the .blend (to `path`).
    Saved { path: String },
    /// Blender wrote its cameras, planes and empties, over the frame range,
    /// to `path` (`scene3d::import`'s JSON) for the 3D cards nodes.
    Scene3d { path: String },
    Error(String),
    Pong,
    /// A line of Blender's stdout/stderr worth showing (errors, our own
    /// `wobble |` chatter).
    Log(String),
    /// Blender is a wobble-fork build that can render Cycles straight into
    /// a buffer wobble shares (`blender::cycles`); `pid` is its process,
    /// which the buffer's handle is duplicated into.
    CyclesFork { pid: u32 },
    /// Stock Blender on Cycles in viewport mode: there is no live Cycles
    /// without the fork, so the picture is the material preview. Sent once
    /// a session, for the app to explain.
    CyclesStock,
    /// Blender's answer to `Session::cycles`: the target is bound (or why
    /// not).
    CyclesBound { ok: bool, msg: String },
    /// The socket closed or the process ended; the session is over.
    Closed(String),
}

/// A message before the ring is involved: what `parse_event` yields.
#[derive(Clone, Debug, PartialEq)]
enum Wire {
    Ev(Event),
    /// Map this ring (Blender made a new one, e.g. for a bigger frame).
    Shm(shm::Source),
    /// Slot `slot` holds the frame with sequence `seq`.
    Frame { slot: usize, seq: u64 },
}

/// A minimal JSON value: what a flat message needs and nothing more.
#[derive(Clone, Debug, PartialEq)]
pub enum Val {
    Str(String),
    Num(f64),
    Bool(bool),
    Null,
}

impl Val {
    fn as_str(&self) -> Option<&str> {
        match self {
            Val::Str(s) => Some(s),
            _ => None,
        }
    }
    fn as_f64(&self) -> Option<f64> {
        match self {
            Val::Num(n) => Some(*n),
            _ => None,
        }
    }
}

/// Parse one flat JSON object (`{"k": "v", "n": 1, "b": true}`). Nested
/// objects and arrays are not messages this protocol sends and return
/// `None`, as does anything malformed.
pub fn parse_flat(line: &str) -> Option<HashMap<String, Val>> {
    let s = line.trim();
    let s = s.strip_prefix('{')?.strip_suffix('}')?;
    let mut out = HashMap::new();
    let mut chars = s.chars().peekable();
    loop {
        skip_ws(&mut chars);
        match chars.peek() {
            None => break,
            Some(',') => {
                chars.next();
                continue;
            }
            Some('"') => {}
            Some(_) => return None,
        }
        let key = parse_string(&mut chars)?;
        skip_ws(&mut chars);
        if chars.next()? != ':' {
            return None;
        }
        skip_ws(&mut chars);
        let val = match chars.peek()? {
            '"' => Val::Str(parse_string(&mut chars)?),
            't' | 'f' | 'n' => {
                let word: String = std::iter::from_fn(|| {
                    chars.next_if(|c| c.is_ascii_alphabetic())
                })
                .collect();
                match word.as_str() {
                    "true" => Val::Bool(true),
                    "false" => Val::Bool(false),
                    "null" => Val::Null,
                    _ => return None,
                }
            }
            c if c.is_ascii_digit() || *c == '-' => {
                let num: String = std::iter::from_fn(|| {
                    chars.next_if(|c| {
                        c.is_ascii_digit() || matches!(c, '-' | '+' | '.' | 'e' | 'E')
                    })
                })
                .collect();
                Val::Num(num.parse().ok()?)
            }
            _ => return None,
        };
        out.insert(key, val);
    }
    Some(out)
}

fn skip_ws(chars: &mut std::iter::Peekable<std::str::Chars>) {
    while chars.next_if(|c| c.is_whitespace()).is_some() {}
}

fn parse_string(chars: &mut std::iter::Peekable<std::str::Chars>) -> Option<String> {
    if chars.next()? != '"' {
        return None;
    }
    let mut s = String::new();
    loop {
        match chars.next()? {
            '"' => return Some(s),
            '\\' => match chars.next()? {
                '"' => s.push('"'),
                '\\' => s.push('\\'),
                '/' => s.push('/'),
                'n' => s.push('\n'),
                'r' => s.push('\r'),
                't' => s.push('\t'),
                'b' => s.push('\u{8}'),
                'f' => s.push('\u{c}'),
                'u' => {
                    let hex: String = (0..4).filter_map(|_| chars.next()).collect();
                    let code = u32::from_str_radix(&hex, 16).ok()?;
                    // Surrogate pairs: paths are the only strings here and
                    // Python escapes non-ASCII as \uXXXX by default, so a
                    // pair is possible (an emoji in a folder name).
                    if (0xD800..0xDC00).contains(&code) {
                        if chars.next()? != '\\' || chars.next()? != 'u' {
                            return None;
                        }
                        let hex2: String = (0..4).filter_map(|_| chars.next()).collect();
                        let low = u32::from_str_radix(&hex2, 16).ok()?;
                        let full = 0x10000 + ((code - 0xD800) << 10) + (low.checked_sub(0xDC00)?);
                        s.push(char::from_u32(full)?);
                    } else {
                        s.push(char::from_u32(code)?);
                    }
                }
                _ => return None,
            },
            c => s.push(c),
        }
    }
}

/// JSON-escape a string for the outbound side.
pub fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Turn one line from Blender into a message. Lines that are not a message
/// this protocol knows are ignored (`None`), never an error — the script
/// may grow events an older wobble has no use for.
fn parse_wire(line: &str) -> Option<Wire> {
    let m = parse_flat(line)?;
    let ev = m.get("ev")?.as_str()?;
    let num = |k: &str| m.get(k).and_then(Val::as_f64);
    let text = |k: &str| m.get(k).and_then(Val::as_str).unwrap_or("").to_string();
    Some(Wire::Ev(match ev {
        "hello" => Event::Hello {
            version: text("version"),
            w: num("w")? as u32,
            h: num("h")? as u32,
            fps: num("fps").unwrap_or(24.0) as f32,
            blend: text("blend"),
            frame: num("frame").unwrap_or(1.0) as i64,
            start: num("start").unwrap_or(1.0) as i64,
            end: num("end").unwrap_or(250.0) as i64,
        },
        "shm" => {
            let name = text("name");
            return Some(Wire::Shm(if !name.is_empty() {
                shm::Source::Posix { name, bytes: num("bytes")? as usize }
            } else {
                shm::Source::File(PathBuf::from(text("path")))
            }));
        }
        "frame" => {
            let slot = num("slot")? as usize;
            let seq = num("seq")? as u64;
            // With a stamp on the wire the frame is in a surface, not the
            // ring (the ring's stamps live in its slot descriptors).
            if let Some(w) = num("w") {
                return Some(Wire::Ev(Event::SurfaceFrame {
                    slot,
                    seq,
                    stamp: shm::Stamp {
                        frame: num("frame").unwrap_or(0.0) as i64,
                        w: w as u32,
                        h: num("h")? as u32,
                        fmt: num("fmt").unwrap_or(shm::FMT_RGBA32F as f64) as u32,
                        flags: num("flags").unwrap_or(0.0) as u32,
                        scene: num("scene").unwrap_or(0.0) as u64,
                        ms: num("ms").unwrap_or(0.0) as f32,
                        samples: num("samples").unwrap_or(0.0) as u32,
                        pass: num("pass").unwrap_or(0.0) as u8,
                    },
                }));
            }
            return Some(Wire::Frame { slot, seq });
        }
        "surfaces" => {
            let ids: Vec<u32> = text("ids").split(',').filter_map(|s| s.trim().parse().ok()).collect();
            return Some(Wire::Ev(Event::Surfaces { ids, w: num("w")? as u32, h: num("h")? as u32 }));
        }
        "frame_changed" => Event::FrameChanged(num("frame")? as i64),
        "view" => Event::View(super::ViewState {
            loc: [num("lx")? as f32, num("ly")? as f32, num("lz")? as f32],
            rot: [num("rw")? as f32, num("rx")? as f32, num("ry")? as f32, num("rz")? as f32],
            dist: num("dist")? as f32,
            lens: num("lens").unwrap_or(50.0) as f32,
            ortho: matches!(m.get("ortho"), Some(Val::Bool(true))),
        }),
        "scene" => Event::Scene(num("scene")? as u64),
        "saved" => Event::Saved { path: text("path") },
        "scene3d" => Event::Scene3d { path: text("path") },
        "error" => Event::Error(text("msg")),
        "pong" => Event::Pong,
        "cycles_fork" => Event::CyclesFork { pid: num("pid")? as u32 },
        "cycles_stock" => Event::CyclesStock,
        "cycles_bound" => Event::CyclesBound { ok: matches!(m.get("ok"), Some(Val::Bool(true))), msg: text("msg") },
        _ => return None,
    }))
}

/// An unused localhost port, found by binding port 0 and letting go. The
/// gap before Blender binds it is real but tiny, and a collision fails
/// loudly in the script's own log.
pub fn free_port() -> Result<u16> {
    let l = TcpListener::bind("127.0.0.1:0").context("finding a free port")?;
    Ok(l.local_addr()?.port())
}

/// How long to keep trying to reach the bridge after launch. Blender on a
/// cold disk with a large .blend takes a while to reach its first timer.
const CONNECT_BUDGET: Duration = Duration::from_secs(120);

/// A running live session. Drop it to DETACH: Blender keeps running with
/// the user's unsaved work, and the node simply has no session any more.
/// `quit()` is the polite request to close.
pub struct Session {
    pub port: u16,
    tx: Sender<String>,
    rx: Receiver<Event>,
    connected: Arc<AtomicBool>,
    closed: Arc<AtomicBool>,
    /// The app has the ring on the GPU: announce frames by slot instead
    /// of copying them out.
    gpu: Arc<AtomicBool>,
    child: Arc<Mutex<Option<Child>>>,
    started: Instant,
}

impl Session {
    /// Launch Blender (with its UI — the live view draws Blender's own
    /// viewport) and the bridge script, and start connecting to it. `blend`
    /// is None for a fresh scene; the node learns the path when the user
    /// saves. The ring lives in `dir`.
    pub fn launch(
        exe: &Path,
        blend: Option<&Path>,
        script: &Path,
        port: u16,
        dir: &Path,
    ) -> Result<Session> {
        let mut cmd = Command::new(exe);
        if let Some(b) = blend {
            cmd.arg(b);
        }
        cmd.arg("--python")
            .arg(script)
            .env("WOBBLE_PORT", port.to_string())
            .env("WOBBLE_SHM_DIR", dir)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = cmd.spawn().with_context(|| {
            format!(
                "launching {} (set WOBBLE_BLENDER to the blender executable)",
                exe.display()
            )
        })?;
        let (etx, erx) = channel();
        // Blender's chatter: keep only what a user would want to see in the
        // panel — errors, and the bridge's own lines.
        for pipe in [
            child.stdout.take().map(|p| Box::new(p) as Box<dyn std::io::Read + Send>),
            child.stderr.take().map(|p| Box::new(p) as Box<dyn std::io::Read + Send>),
        ]
        .into_iter()
        .flatten()
        {
            let etx = etx.clone();
            std::thread::spawn(move || {
                for line in BufReader::new(pipe).lines().map_while(Result::ok) {
                    let l = line.trim();
                    if l.starts_with("wobble |")
                        || l.contains("Error")
                        || l.contains("error:")
                        || l.contains("Traceback")
                    {
                        let _ = etx.send(Event::Log(l.to_string()));
                    }
                }
            });
        }
        Ok(Self::connect_with(port, Some(child), etx, erx))
    }

    /// Attach to a bridge that is already listening (a test's fake Blender,
    /// or one launched by hand with `WOBBLE_PORT` set).
    pub fn connect(port: u16) -> Session {
        let (etx, erx) = channel();
        Self::connect_with(port, None, etx, erx)
    }

    fn connect_with(
        port: u16,
        child: Option<Child>,
        etx: Sender<Event>,
        erx: Receiver<Event>,
    ) -> Session {
        let (tx, crx) = channel::<String>();
        let connected = Arc::new(AtomicBool::new(false));
        let closed = Arc::new(AtomicBool::new(false));
        let gpu = Arc::new(AtomicBool::new(false));
        let child = Arc::new(Mutex::new(child));
        let session = Session {
            port,
            tx,
            rx: erx,
            connected: connected.clone(),
            closed: closed.clone(),
            gpu: gpu.clone(),
            child: child.clone(),
            started: Instant::now(),
        };
        std::thread::spawn(move || {
            let close = |why: String| {
                closed.store(true, Ordering::Relaxed);
                let _ = etx.send(Event::Closed(why));
            };
            let t0 = Instant::now();
            let stream = loop {
                if let Ok(s) = TcpStream::connect(("127.0.0.1", port)) {
                    break s;
                }
                // The process ending before it ever listened is the common
                // failure (a bad .blend path, a script error): say so
                // rather than waiting out the whole budget.
                if let Some(c) = child.lock().unwrap().as_mut() {
                    if let Ok(Some(status)) = c.try_wait() {
                        return close(format!("blender exited ({status}) before connecting"));
                    }
                }
                if t0.elapsed() > CONNECT_BUDGET {
                    return close("no connection from blender".into());
                }
                std::thread::sleep(Duration::from_millis(250));
            };
            let _ = stream.set_nodelay(true);
            connected.store(true, Ordering::Relaxed);
            // Writer: drains the command channel into the socket.
            let mut w = match stream.try_clone() {
                Ok(w) => w,
                Err(e) => return close(format!("socket clone: {e}")),
            };
            std::thread::spawn(move || {
                for line in crx.iter() {
                    if w.write_all(line.as_bytes()).is_err() {
                        break;
                    }
                }
            });
            let mut ring: Option<shm::Reader> = None;
            for line in BufReader::new(stream).lines().map_while(Result::ok) {
                let ev = match parse_wire(&line) {
                    Some(Wire::Ev(ev)) => ev,
                    Some(Wire::Shm(src)) => match shm::Reader::open(&src) {
                        Ok(r) => {
                            ring = Some(r);
                            // A new ring: the app's GPU import of the old
                            // one is stale until it imports this one.
                            gpu.store(false, Ordering::Relaxed);
                            Event::Ring(src)
                        }
                        Err(e) => Event::Error(format!("{e:#}")),
                    },
                    // On the GPU path only the descriptor travels; on the
                    // CPU path the pixels are copied HERE, on the socket
                    // thread, the moment they are announced — the writer
                    // has laps to go before it comes back to this slot,
                    // and the UI thread only uploads.
                    Some(Wire::Frame { slot, seq }) => {
                        if gpu.load(Ordering::Relaxed) {
                            match ring.as_ref().and_then(|r| r.desc(slot)) {
                                Some(d) if d.seq == seq => Event::FrameRef { slot, desc: d },
                                _ => Event::Dropped,
                            }
                        } else {
                            match ring.as_ref().and_then(|r| r.read(slot)) {
                                Some(f) if f.seq == seq => Event::Frame(f),
                                _ => Event::Dropped,
                            }
                        }
                    }
                    None => continue,
                };
                if etx.send(ev).is_err() {
                    return;
                }
            }
            close("blender closed the connection".into());
        });
        session
    }

    /// The bridge answered; commands go somewhere.
    pub fn connected(&self) -> bool {
        self.connected.load(Ordering::Relaxed)
    }

    /// Frames come by reference (the ring is on the GPU) or by copy.
    pub fn set_gpu(&self, on: bool) {
        self.gpu.store(on, Ordering::Relaxed);
    }

    pub fn gpu(&self) -> bool {
        self.gpu.load(Ordering::Relaxed)
    }

    /// Done with a slot announced as `FrameRef`: Blender may write it again.
    pub fn ack(&self, slot: usize, seq: u64) {
        self.send(format!("{{\"cmd\":\"ack\",\"slot\":{slot},\"seq\":{seq}}}\n"));
    }

    /// An `ack` that can be sent from another thread (a GPU completion
    /// callback, the cache copy). `surface` picks the surface ring's ack.
    pub fn acker(&self, surface: bool) -> impl Fn(usize, u64) + Send + Sync + 'static {
        let tx = self.tx.clone();
        let cmd = if surface { "sack" } else { "ack" };
        move |slot, seq| {
            let _ = tx.send(format!("{{\"cmd\":\"{cmd}\",\"slot\":{slot},\"seq\":{seq}}}\n"));
        }
    }

    /// Done with a surface announced as `SurfaceFrame`.
    pub fn sack(&self, slot: usize, seq: u64) {
        self.send(format!("{{\"cmd\":\"sack\",\"slot\":{slot},\"seq\":{seq}}}\n"));
    }

    /// Whether the surfaces Blender announced could be imported; on `false`
    /// Blender goes back to the shared-memory ring.
    pub fn surfaces_ok(&self, ok: bool) {
        self.send(format!("{{\"cmd\":\"surfaces\",\"ok\":{ok}}}\n"));
    }

    /// The session is over (socket closed or process gone).
    pub fn closed(&self) -> bool {
        self.closed.load(Ordering::Relaxed)
    }

    /// Seconds since launch, for the "starting…" state in the panel.
    pub fn age(&self) -> f32 {
        self.started.elapsed().as_secs_f32()
    }

    fn send(&self, line: String) {
        let _ = self.tx.send(line);
    }

    /// Move Blender's playhead. Blender coalesces a burst to the last one.
    pub fn frame(&self, n: i64) {
        self.frame_ex(n, false);
    }

    /// `frame`, with `cached` saying wobble already has this frame settled
    /// and on the canvas: Blender moves its playhead and skips the draw.
    pub fn frame_ex(&self, n: i64, cached: bool) {
        self.send(format!("{{\"cmd\":\"frame\",\"frame\":{n},\"cached\":{cached}}}\n"));
    }

    /// Move the eye (the 3D view or the camera, per the spec's `view`).
    /// `moving` = a drag is in progress: Blender draws at one sample until
    /// the hand stops.
    pub fn view(&self, v: &super::ViewState, moving: bool) {
        self.send(format!(
            "{{\"cmd\":\"view\",\"lx\":{},\"ly\":{},\"lz\":{},\"rw\":{},\"rx\":{},\"ry\":{},\"rz\":{},\"dist\":{},\"lens\":{},\"ortho\":{},\"moving\":{}}}\n",
            v.loc[0], v.loc[1], v.loc[2], v.rot[0], v.rot[1], v.rot[2], v.rot[3], v.dist, v.lens, v.ortho, moving
        ));
    }

    /// Apply the node's spec (`BlenderParams::spec_line`): format, engine,
    /// camera, quality — the whole scene setup, idempotent.
    pub fn spec(&self, line: &str) {
        debug_assert!(line.ends_with('\n'));
        self.send(line.to_string());
    }

    /// Map the ring wobble's textures travel in.
    pub fn shm(&self, src: &shm::Source) {
        self.send(format!("{{\"cmd\":\"shm\",{}}}\n", src.json_fields()));
    }

    /// Input port `port`'s picture is in slot `slot` of that ring.
    pub fn tex(&self, port: usize, slot: usize, seq: u64) {
        self.send(format!("{{\"cmd\":\"tex\",\"port\":{port},\"slot\":{slot},\"seq\":{seq}}}\n"));
    }

    /// Point the fork's Cycles at wobble's shared buffer (`handle` already
    /// duplicated into Blender's process); `handle` 0 detaches it.
    pub fn cycles(&self, handle: u64, size: u64, uuid: &str, stamp: &str) {
        self.send(format!(
            "{{\"cmd\":\"cycles\",\"handle\":{handle},\"size\":{size},\"uuid\":{},\"stamp\":{}}}\n",
            json_str(uuid),
            json_str(stamp)
        ));
    }

    /// Ask for the scene's cameras, planes and empties as a JSON file at
    /// `path` (answered by `Event::Scene3d`).
    pub fn scene3d(&self, path: &str) {
        self.send(format!("{{\"cmd\":\"scene3d\",\"path\":{}}}\n", json_str(path)));
    }

    pub fn ping(&self) {
        self.send("{\"cmd\":\"ping\"}\n".into());
    }

    /// Ask Blender to close (it prompts for unsaved changes itself).
    pub fn quit(&self) {
        self.send("{\"cmd\":\"quit\"}\n".into());
    }

    /// Everything that arrived since the last poll.
    pub fn poll(&self) -> Vec<Event> {
        self.rx.try_iter().collect()
    }

    /// Kill the process outright. Only for a session that never connected
    /// (nothing to lose) — a live one is asked with `quit()`.
    pub fn kill(&self) {
        if let Some(c) = self.child.lock().unwrap().as_mut() {
            let _ = c.kill();
            let _ = c.wait();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_objects_parse() {
        let m = parse_flat(r#"{"ev": "frame", "slot": 1, "w": 1920, "final": false, "x": null}"#)
            .unwrap();
        assert_eq!(m["ev"], Val::Str("frame".into()));
        assert_eq!(m["slot"], Val::Num(1.0));
        assert_eq!(m["final"], Val::Bool(false));
        assert_eq!(m["x"], Val::Null);
        assert!(parse_flat("not json").is_none());
        assert!(parse_flat(r#"{"a": [1]}"#).is_none());
    }

    #[test]
    fn strings_unescape_what_python_emits() {
        // json.dumps escapes non-ASCII as \uXXXX and backslashes doubled;
        // a Windows path and an accented folder must come back verbatim.
        let m = parse_flat(r#"{"file": "C:\\cache\\ring.shm", "p": "caf\u00e9 \ud83c\udfac"}"#)
            .unwrap();
        assert_eq!(m["file"].as_str(), Some("C:\\cache\\ring.shm"));
        assert_eq!(m["p"].as_str(), Some("café 🎬"));
    }

    #[test]
    fn messages_decode() {
        assert_eq!(
            parse_wire(r#"{"ev": "frame", "slot": 2, "seq": 14}"#),
            Some(Wire::Frame { slot: 2, seq: 14 })
        );
        assert_eq!(
            parse_wire(r#"{"ev": "shm", "path": "/c/r.shm"}"#),
            Some(Wire::Shm(shm::Source::File(PathBuf::from("/c/r.shm"))))
        );
        assert_eq!(
            parse_wire(r#"{"ev": "shm", "name": "wob_1_2", "bytes": 65536}"#),
            Some(Wire::Shm(shm::Source::Posix { name: "wob_1_2".into(), bytes: 65536 }))
        );
        assert_eq!(
            parse_wire(r#"{"ev": "surfaces", "ids": "7,8, 9", "w": 640, "h": 360}"#),
            Some(Wire::Ev(Event::Surfaces { ids: vec![7, 8, 9], w: 640, h: 360 }))
        );
        let sf = parse_wire(r#"{"ev": "frame", "slot": 1, "seq": 6, "frame": 12, "w": 320, "h": 180, "flags": 2, "scene": 4, "ms": 3.5, "samples": 16, "pass": 1}"#);
        match sf {
            Some(Wire::Ev(Event::SurfaceFrame { slot: 1, seq: 6, stamp })) => {
                assert_eq!((stamp.frame, stamp.w, stamp.h, stamp.scene, stamp.pass), (12, 320, 180, 4, 1));
                assert_eq!(stamp.fmt, shm::FMT_RGBA32F);
            }
            other => panic!("{other:?}"),
        }
        let hello = parse_wire(
            r#"{"ev": "hello", "version": "5.2.1", "w": 2048, "h": 1080, "fps": 24.0, "blend": "", "frame": 1, "start": 1, "end": 48}"#,
        );
        assert!(matches!(hello, Some(Wire::Ev(Event::Hello { w: 2048, end: 48, .. }))));
        assert_eq!(
            parse_wire(r#"{"ev": "frame_changed", "frame": 12}"#),
            Some(Wire::Ev(Event::FrameChanged(12)))
        );
        assert_eq!(parse_wire(r#"{"ev": "cycles_stock"}"#), Some(Wire::Ev(Event::CyclesStock)));
        let v = parse_wire(
            r#"{"ev": "view", "lx": 1, "ly": 2, "lz": 3, "rw": 1, "rx": 0, "ry": 0, "rz": 0, "dist": 7.5, "lens": 35, "ortho": true}"#,
        );
        assert_eq!(
            v,
            Some(Wire::Ev(Event::View(super::super::ViewState {
                loc: [1.0, 2.0, 3.0],
                rot: [1.0, 0.0, 0.0, 0.0],
                dist: 7.5,
                lens: 35.0,
                ortho: true
            })))
        );
        assert_eq!(parse_wire(r#"{"ev": "pong"}"#), Some(Wire::Ev(Event::Pong)));
        // Unknown events are skipped, not errors.
        assert_eq!(parse_wire(r#"{"ev": "future"}"#), None);
        assert_eq!(parse_wire("garbage"), None);
    }

    #[test]
    fn outbound_strings_escape() {
        assert_eq!(json_str("a\"b\\c\n"), r#""a\"b\\c\n""#);
        assert_eq!(json_str("plain"), "\"plain\"");
    }

    /// A fake bridge on a real socket and a real ring: the session
    /// connects, hears hello, maps the ring, moves the playhead, and gets
    /// the frame's PIXELS back as an event, then sees the close.
    #[test]
    fn session_round_trip_against_a_fake_bridge() {
        let dir = std::env::temp_dir().join(format!("wobble-live-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let ring = dir.join("ring.shm");
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let ring2 = ring.clone();
        std::thread::spawn(move || {
            let (mut conn, _) = listener.accept().unwrap();
            let mut w = shm::Writer::create(&shm::Source::File(ring2.clone()), 3, shm::Writer::bytes_for(2, 2, shm::FMT_RGBA16F).unwrap()).unwrap();
            conn.write_all(
                b"{\"ev\": \"hello\", \"version\": \"fake\", \"w\": 2, \"h\": 2, \"fps\": 24, \"blend\": \"\"}\n",
            )
            .unwrap();
            conn.write_all(format!("{{\"ev\": \"shm\", {}}}\n", w.source.json_fields()).as_bytes())
                .unwrap();
            let mut r = BufReader::new(conn.try_clone().unwrap());
            let mut line = String::new();
            r.read_line(&mut line).unwrap();
            assert_eq!(line.trim(), r#"{"cmd":"frame","frame":1005,"cached":false}"#);
            let stamp = shm::Stamp { frame: 1005, w: 2, h: 2, fmt: shm::FMT_RGBA16F, scene: 3, ..Default::default() };
            let (slot, seq) = w.write(stamp, &[9u8; 32]).unwrap();
            conn.write_all(format!("{{\"ev\": \"frame\", \"slot\": {slot}, \"seq\": {seq}}}\n").as_bytes())
                .unwrap();
            line.clear();
            r.read_line(&mut line).unwrap();
            assert_eq!(line.trim(), r#"{"cmd":"quit"}"#);
        });
        let s = Session::connect(port);
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut got = Vec::new();
        while Instant::now() < deadline && !got.iter().any(|e| matches!(e, Event::Ring(_))) {
            got.extend(s.poll());
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(s.connected(), "never connected: {got:?}");
        s.frame(1005);
        while Instant::now() < deadline && !got.iter().any(|e| matches!(e, Event::Frame(_))) {
            got.extend(s.poll());
            std::thread::sleep(Duration::from_millis(10));
        }
        let f = got.iter().find_map(|e| match e {
            Event::Frame(f) => Some(f.clone()),
            _ => None,
        });
        let f = f.unwrap_or_else(|| panic!("no frame event: {got:?}"));
        assert_eq!((f.frame, f.w, f.h, f.scene), (1005, 2, 2, 3));
        assert_eq!(f.pixels, vec![9u8; 32]);
        s.quit();
        while Instant::now() < deadline && !s.closed() {
            got.extend(s.poll());
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(s.closed(), "session never closed: {got:?}");
    }
}
