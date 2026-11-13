//! Send a finished render to another wobble on the network (`peers`).
//!
//! The Render node's "send to…" copies the frame files it wrote to a
//! paired wobble; that wobble saves them under its inbox folder, in
//! `<inbox>/<sender>/<render name>/`. What crosses is FILE BYTES only —
//! no setup, no path, nothing that runs — and the receiver decides:
//!
//! - **Off until the receiver says so**: "accept renders from paired
//!   wobbles" in its network window, on top of sharing. Without it every
//!   `render_receive` is refused in words, which the sender shows.
//! - **Names, never paths.** Sender, render and file name must each be one
//!   plain component (no separator, no `..`, no leading dot), and the file
//!   must be a picture or movie by its extension; the receiver joins them
//!   onto its own inbox.
//! - **Chunked, in order.** A file goes in `CHUNK` pieces as base64 (the
//!   `render_fetch` precedent); it is written to a dotted `.part` beside
//!   its name and renamed into place when the last byte lands, so a
//!   half-sent frame never looks like a frame.

use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use super::peers::{self, Conn};

/// Bytes per call.
pub const CHUNK: usize = 8 << 20;
/// The largest single file a receiver takes.
const MAX_FILE: u64 = 64 << 30;
/// What a receiver takes, by extension (lower case).
const EXTS: &[&str] = &["exr", "png", "jpg", "jpeg", "dpx", "tif", "tiff", "mov", "mp4", "gif"];

/// Is `s` one plain path component a receiver may use as a name?
pub fn plain(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 200
        && !s.starts_with('.')
        && !s.chars().any(|c| matches!(c, '/' | '\\' | ':' | '\0') || c.is_control())
        && s.trim() == s
}

fn wanted_ext(file: &str) -> bool {
    Path::new(file)
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| EXTS.contains(&e.to_ascii_lowercase().as_str()))
}

// ---------------------------------------------------------------- receiving

#[derive(Clone)]
struct Got {
    dir: PathBuf,
    files: usize,
    bytes: u64,
    at: Instant,
}

fn got() -> std::sync::MutexGuard<'static, BTreeMap<(String, String), Got>> {
    static GOT: OnceLock<Mutex<BTreeMap<(String, String), Got>>> = OnceLock::new();
    GOT.get_or_init(Default::default).lock().unwrap_or_else(|e| e.into_inner())
}

/// Write one chunk under `inbox`. `Ok(Some(path))` once the file is whole.
pub fn write_chunk(
    inbox: &Path,
    from: &str,
    name: &str,
    file: &str,
    offset: u64,
    size: u64,
    data: &[u8],
) -> Result<Option<PathBuf>, String> {
    for (what, s) in [("sender", from), ("render name", name), ("file name", file)] {
        if !plain(s) {
            return Err(format!("{what} {s:?} is not a plain name"));
        }
    }
    if !wanted_ext(file) {
        return Err(format!("{file}: only pictures and movies are received"));
    }
    if size > MAX_FILE || offset + data.len() as u64 > size {
        return Err(format!("{file}: bad size"));
    }
    let dir = inbox.join(from).join(name);
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let part = dir.join(format!(".{file}.part"));
    let mut f = if offset == 0 {
        std::fs::File::create(&part)
    } else {
        std::fs::OpenOptions::new().read(true).write(true).open(&part)
    }
    .map_err(|e| format!("{}: {e}", part.display()))?;
    let len = f.seek(SeekFrom::End(0)).map_err(|e| e.to_string())?;
    if len != offset {
        return Err(format!("{file}: expected byte {len}, got {offset} (start the file again)"));
    }
    f.write_all(data).map_err(|e| format!("{}: {e}", part.display()))?;
    drop(f);
    if offset + data.len() as u64 == size {
        let fin = dir.join(file);
        std::fs::rename(&part, &fin).map_err(|e| format!("{}: {e}", fin.display()))?;
        return Ok(Some(fin));
    }
    Ok(None)
}

/// `render_receive`: one chunk of one file of a render another wobble sends.
pub fn receive(p: &Value) -> Result<Value, String> {
    let inbox = peers::inbox().ok_or(
        "this wobble does not accept renders (network… > accept renders from paired wobbles)",
    )?;
    let s = |k: &str| p[k].as_str().ok_or(format!("'{k}' must be a string"));
    let n = |k: &str| p[k].as_u64().ok_or(format!("'{k}' must be a number"));
    let (from, name, file) = (s("from")?, s("name")?, s("file")?);
    let data = api::unb64(s("data")?)?;
    let whole = write_chunk(&inbox, from, name, file, n("offset")?, n("size")?, &data)?;
    if whole.is_some() {
        let size = n("size")?;
        let mut g = got();
        if !g.contains_key(&(from.to_string(), name.to_string())) {
            peers::open_window();
        }
        let e = g.entry((from.to_string(), name.to_string())).or_insert(Got {
            dir: inbox.join(from).join(name),
            files: 0,
            bytes: 0,
            at: Instant::now(),
        });
        e.files += 1;
        e.bytes += size;
        e.at = Instant::now();
        log::info!("received {file} of {name} from {from}");
    }
    Ok(json!({ "file": file, "complete": whole.is_some() }))
}

/// What arrived, newest first, for the network window.
pub fn received_ui(ui: &mut egui::Ui) {
    let mut list: Vec<_> = got().iter().map(|((f, n), g)| (f.clone(), n.clone(), g.clone())).collect();
    if list.is_empty() {
        return;
    }
    list.sort_by_key(|(_, _, g)| std::cmp::Reverse(g.at));
    ui.label(egui::RichText::new("received").weak());
    for (from, name, g) in list {
        ui.horizontal(|ui| {
            ui.label(format!("{name} from {from}"));
            ui.weak(format!("{} file{} · {}", g.files, if g.files == 1 { "" } else { "s" }, mb(g.bytes)));
            if ui.small_button("reveal").on_hover_text(g.dir.display().to_string()).clicked() {
                reveal(&g.dir);
            }
        });
    }
}

fn mb(b: u64) -> String {
    format!("{:.1} MB", b as f64 / 1e6)
}

pub fn reveal(dir: &Path) {
    let _ = std::fs::create_dir_all(dir);
    #[cfg(target_os = "macos")]
    let _ = std::process::Command::new("open").arg(dir).spawn();
    #[cfg(target_os = "windows")]
    let _ = std::process::Command::new("explorer").arg(dir).spawn();
    #[cfg(all(unix, not(target_os = "macos")))]
    let _ = std::process::Command::new("xdg-open").arg(dir).spawn();
}

// ------------------------------------------------------------------ sending

#[derive(Clone, Default)]
struct Sending {
    active: bool,
    to: String,
    name: String,
    done: usize,
    total: usize,
    note: Option<String>,
}

fn sending() -> std::sync::MutexGuard<'static, Sending> {
    static S: OnceLock<Mutex<Sending>> = OnceLock::new();
    S.get_or_init(Default::default).lock().unwrap_or_else(|e| e.into_inner())
}

/// The files a Render node wrote that are still there, in frame order.
pub fn render_files(files: &[PathBuf]) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    for f in files {
        if f.is_file() && !out.contains(f) {
            out.push(f.clone());
        }
    }
    out
}

/// A render's name on the other side: the folder-safe Render name.
pub fn send_name(raw: &str) -> String {
    let s: String =
        raw.chars().map(|c| if matches!(c, '/' | '\\' | ':' | '\0') || c.is_control() { '_' } else { c }).collect();
    let s = s.trim_matches(|c: char| c == '_' || c == '.' || c.is_whitespace()).to_string();
    if s.is_empty() { "render".into() } else { s }
}

/// Send `files` to the paired wobble `peer` as render `name`, on a thread.
pub fn start(peer: String, name: String, files: Vec<PathBuf>) -> Result<(), String> {
    if files.is_empty() {
        return Err("nothing rendered to send".into());
    }
    let (p, code) = peers::peer(&peer).ok_or("that wobble is not paired (check it in network…)")?;
    {
        let mut s = sending();
        if s.active {
            return Err("already sending a render".into());
        }
        *s = Sending { active: true, to: p.name.clone(), name: name.clone(), done: 0, total: files.len(), note: None };
    }
    let from = send_name(&peers::name());
    std::thread::spawn(move || {
        let started = Instant::now();
        let r = (|| {
            let mut c = Conn::open(&p.addrs, p.port, Some(&code))?;
            for (k, f) in files.iter().enumerate() {
                send_file(&mut c, &from, &name, f)?;
                sending().done = k + 1;
            }
            Ok::<_, String>(())
        })();
        let mut s = sending();
        s.active = false;
        s.note = Some(match r {
            Ok(()) => format!("sent {} file{} to {} in {:.0} s", s.total, if s.total == 1 { "" } else { "s" }, s.to, started.elapsed().as_secs_f32()),
            Err(e) => format!("send to {} failed after {} files: {e}", s.to, s.done),
        });
    });
    Ok(())
}

fn send_file(c: &mut Conn, from: &str, name: &str, path: &Path) -> Result<(), String> {
    let file = path.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default();
    let mut f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let size = f.metadata().map_err(|e| e.to_string())?.len();
    let mut buf = vec![0u8; CHUNK];
    let mut offset = 0u64;
    loop {
        let mut got = 0;
        while got < buf.len() {
            let k = f.read(&mut buf[got..]).map_err(|e| e.to_string())?;
            if k == 0 {
                break;
            }
            got += k;
        }
        c.call(
            "render_receive",
            json!({
                "from": from, "name": name, "file": file, "offset": offset, "size": size,
                "data": api::b64(&buf[..got]),
            }),
        )?;
        offset += got as u64;
        if offset >= size {
            return Ok(());
        }
    }
}

/// `render_send_status`.
pub fn snapshot() -> Value {
    let s = sending();
    json!({ "sending": s.active, "to": s.to, "name": s.name, "done": s.done, "total": s.total, "note": s.note })
}

/// The Render panel's row: "send to…" a paired wobble, and how it is going.
/// `files` is what the node wrote (empty before a render).
pub fn send_row(ui: &mut egui::Ui, name: &str, files: &[PathBuf]) {
    let s = sending().clone();
    let peers = peers::paired();
    ui.horizontal(|ui| {
        let ready = !files.is_empty() && !s.active && !peers.is_empty();
        ui.add_enabled_ui(ready, |ui| {
            ui.menu_button("send to…", |ui| {
                for p in &peers {
                    if ui.button(&p.name).on_hover_text("copy this render's frames to that wobble").clicked() {
                        let r = start(p.id.clone(), send_name(name), render_files(files));
                        if let Err(e) = r {
                            sending().note = Some(e);
                        }
                        ui.close();
                    }
                }
            })
            .response
            .on_disabled_hover_text(if peers.is_empty() {
                "pair another wobble first (preferences > network…)"
            } else if s.active {
                "a render is being sent"
            } else {
                "render first, then send the frames to another wobble"
            });
        });
        if s.active {
            ui.add(
                egui::ProgressBar::new(s.done as f32 / s.total.max(1) as f32)
                    .desired_width(110.0)
                    .text(format!("{}/{} to {}", s.done, s.total, s.to)),
            );
            ui.ctx().request_repaint_after(Duration::from_millis(250));
        } else if let Some(n) = &s.note {
            ui.weak(n);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_plain_or_refused() {
        for ok in ["comp.1001.exr", "render-node01", "my render"] {
            assert!(plain(ok), "{ok}");
        }
        for bad in ["", "..", ".hidden", "a/b", "a\\b", "c:x", " x", "a\nb"] {
            assert!(!plain(bad), "{bad:?}");
        }
        assert_eq!(send_name("comp v3/final"), "comp v3_final");
        assert_eq!(send_name("..."), "render");
    }

    /// Chunks land in a hidden part file, in order, and the frame appears
    /// only when whole; anything that is not a plain picture is refused.
    #[test]
    fn a_file_arrives_in_chunks_and_appears_whole() {
        let inbox = std::env::temp_dir().join(format!("wobble-deliver-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&inbox);
        let bytes: Vec<u8> = (0..1000u32).map(|i| i as u8).collect();
        let fin = inbox.join("hostA").join("comp").join("comp.1001.exr");
        assert_eq!(write_chunk(&inbox, "hostA", "comp", "comp.1001.exr", 0, 1000, &bytes[..600]).unwrap(), None);
        assert!(!fin.exists(), "a half-sent frame never looks like a frame");
        assert!(write_chunk(&inbox, "hostA", "comp", "comp.1001.exr", 100, 1000, &bytes[600..]).is_err(), "out of order");
        assert_eq!(write_chunk(&inbox, "hostA", "comp", "comp.1001.exr", 600, 1000, &bytes[600..]).unwrap(), Some(fin.clone()));
        assert_eq!(std::fs::read(&fin).unwrap(), bytes);
        assert!(write_chunk(&inbox, "..", "comp", "a.exr", 0, 1, b"x").is_err());
        assert!(write_chunk(&inbox, "hostA", "comp", "../a.exr", 0, 1, b"x").is_err());
        assert!(write_chunk(&inbox, "hostA", "comp", "run.sh", 0, 1, b"x").is_err());
        assert!(write_chunk(&inbox, "hostA", "comp", "a.exr", 0, 1, b"xy").is_err(), "more than its size");
        let _ = std::fs::remove_dir_all(&inbox);
    }

    /// The real listener and the real sender, on loopback: every file
    /// arrives byte for byte through several chunks.
    #[test]
    fn send_file_reaches_a_listener() {
        let inbox = std::env::temp_dir().join(format!("wobble-deliver-net-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&inbox);
        let (host, handle) = api::server::serve_network(0, "ABCD-2345".into()).unwrap();
        let rx = host.rx;
        let ib = inbox.clone();
        std::thread::spawn(move || {
            for c in rx {
                let p = &c.params;
                let data = api::unb64(p["data"].as_str().unwrap()).unwrap();
                let r = write_chunk(
                    &ib,
                    p["from"].as_str().unwrap(),
                    p["name"].as_str().unwrap(),
                    p["file"].as_str().unwrap(),
                    p["offset"].as_u64().unwrap(),
                    p["size"].as_u64().unwrap(),
                    &data,
                );
                c.answer(r.map(|w| json!({ "complete": w.is_some() })).map_err(api::ApiError::app));
            }
        });
        let src = inbox.join("src.exr");
        std::fs::create_dir_all(&inbox).unwrap();
        let bytes: Vec<u8> = (0..(CHUNK * 2 + 77)).map(|i| (i * 7) as u8).collect();
        std::fs::write(&src, &bytes).unwrap();
        let mut c = Conn::open(&[std::net::Ipv4Addr::LOCALHOST], host.port, Some("ABCD-2345")).unwrap();
        send_file(&mut c, "hostA", "comp", &src).unwrap();
        assert_eq!(std::fs::read(inbox.join("hostA").join("comp").join("src.exr")).unwrap(), bytes);
        handle.close();
        let _ = std::fs::remove_dir_all(&inbox);
    }
}
