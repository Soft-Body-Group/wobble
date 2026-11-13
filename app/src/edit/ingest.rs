//! Unix-socket ingest: sequence manipulator (or anything else) hands us
//! sequences to add as timeline clips.
//!
//! Protocol: connect to the socket, write one line per sequence, close.
//! A line is JSON `{"sample": "/abs/path/to/any.frame.exr"}` (extensible)
//! or, tolerated, a bare path.
//!
//! The listener is a Unix domain socket, which `std` offers on Unix only.
//! Elsewhere `start` hands back a receiver nothing will ever feed — edit
//! mode works, clips just arrive by the other routes — and says so once in
//! the log. A Windows transport is a second `listen()`; the line protocol
//! and everything downstream of the channel stay as they are.

#[cfg(unix)]
use std::io::BufRead as _;
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};

#[cfg(unix)]
pub fn socket_path() -> PathBuf {
    let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_default();
    home.join("Library/Application Support/WobbleEdit/ingest.sock")
}

/// Bind the ingest socket and stream received sample paths. The egui context
/// is woken on arrival so an idle window picks clips up immediately.
pub fn start(ctx: egui::Context) -> Receiver<PathBuf> {
    let (tx, rx) = channel();
    // Tests construct Apps while a real instance may be running; they must
    // not steal its socket.
    if std::env::var_os("WOBBLE_EDIT_NO_INGEST").is_some() {
        return rx;
    }
    listen(tx, ctx);
    rx
}

/// Accept connections for the life of the process, one thread per
/// connection, each line a sample path onto `tx`.
#[cfg(unix)]
fn listen(tx: Sender<PathBuf>, ctx: egui::Context) {
    std::thread::Builder::new()
        .name("ingest".into())
        .spawn(move || {
            let sock = socket_path();
            if let Some(dir) = sock.parent() {
                std::fs::create_dir_all(dir).ok();
            }
            // A dead instance leaves the file behind and blocks bind; the
            // newest instance takes the socket over.
            std::fs::remove_file(&sock).ok();
            let listener = match std::os::unix::net::UnixListener::bind(&sock) {
                Ok(l) => l,
                Err(e) => {
                    log::error!("ingest bind {}: {e}", sock.display());
                    return;
                }
            };
            log::info!("ingest listening on {}", sock.display());
            for conn in listener.incoming() {
                let Ok(conn) = conn else { continue };
                let tx = tx.clone();
                let ctx = ctx.clone();
                std::thread::spawn(move || {
                    let reader = std::io::BufReader::new(conn);
                    for line in reader.lines().map_while(Result::ok) {
                        let line = line.trim();
                        if line.is_empty() {
                            continue;
                        }
                        let sample = serde_json::from_str::<serde_json::Value>(line)
                            .ok()
                            .and_then(|v| {
                                v.get("sample")
                                    .and_then(|s| s.as_str().map(String::from))
                            })
                            .unwrap_or_else(|| line.to_string());
                        log::info!("ingest: {sample}");
                        tx.send(PathBuf::from(sample)).ok();
                        ctx.request_repaint();
                    }
                });
            }
        })
        .unwrap();
}

/// No Unix domain sockets in `std` here: nothing listens, and dropping `tx`
/// makes the receiver report disconnected rather than wait forever.
#[cfg(not(unix))]
fn listen(tx: Sender<PathBuf>, ctx: egui::Context) {
    let _ = (tx, ctx);
    log::info!("ingest: no transport on this platform (Unix sockets only); not listening");
}
