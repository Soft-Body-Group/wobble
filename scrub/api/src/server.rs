//! The socket side of the app: accepts local connections, reads one
//! JSON-RPC request per line, and hands each to the app as a [`Call`] over a
//! channel. The app drains that channel between frames and answers through
//! the call's reply sender; the connection thread waits for the answer and
//! writes it back. So nothing the protocol does ever runs on the render
//! path, and a slow or dead client costs the app nothing.
//!
//! An idle app sleeps in its event loop until something wakes it, so every
//! call also rings the app's [`Host::set_waker`] bell once it is queued;
//! without it a call waits for the next mouse move or key.

use crate::{method, validate, ApiError};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::net::{Ipv4Addr, TcpListener, TcpStream};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// How long a connection waits for the app to answer before giving up.
/// Long enough for `viewer_capture` on a cold 4K frame, or a long batch.
pub const ANSWER_TIMEOUT: Duration = Duration::from_secs(60);

/// One validated request, waiting for the app.
pub struct Call {
    pub method: String,
    pub params: Value,
    pub reply: Sender<Result<Value, ApiError>>,
}

impl Call {
    pub fn answer(self, r: Result<Value, ApiError>) {
        // The client may have gone; nobody is left to tell.
        let _ = self.reply.send(r);
    }
}

/// Wakes the app's UI loop; called from connection threads.
pub type Waker = Arc<dyn Fn() + Send + Sync>;

/// The listening end the app holds.
pub struct Host {
    pub rx: Receiver<Call>,
    pub port: u16,
    waker: Arc<Mutex<Option<Waker>>>,
}

impl Host {
    /// What wakes the app when a call arrives (set once the UI loop
    /// exists; calls queued before that wait for the first frame).
    pub fn set_waker(&self, wake: impl Fn() + Send + Sync + 'static) {
        if let Ok(mut w) = self.waker.lock() {
            *w = Some(Arc::new(wake));
        }
    }
}

/// The sending end each connection holds: queue a call, then wake the app.
#[derive(Clone)]
pub struct Inbox {
    tx: Sender<Call>,
    waker: Arc<Mutex<Option<Waker>>>,
}

impl Inbox {
    /// Queue `call` and ring the app. Err when the app has gone.
    pub fn send(&self, call: Call) -> Result<(), ()> {
        self.tx.send(call).map_err(|_| ())?;
        let wake = self.waker.lock().ok().and_then(|w| w.clone());
        if let Some(wake) = wake {
            wake();
        }
        Ok(())
    }
}

/// Listen on 127.0.0.1:`port` (0 picks a free one; [`Host::port`] says
/// which). Loopback only: the protocol can open and save files, so it is
/// never offered to the network.
pub fn serve(port: u16) -> std::io::Result<Host> {
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, port))?;
    let port = listener.local_addr()?.port();
    let (tx, rx) = mpsc::channel();
    let waker = Arc::new(Mutex::new(None));
    let inbox = Inbox { tx, waker: waker.clone() };
    std::thread::Builder::new().name("api-accept".into()).spawn(move || {
        for stream in listener.incoming().flatten() {
            let inbox = inbox.clone();
            let _ = std::thread::Builder::new()
                .name("api-conn".into())
                .spawn(move || connection(stream, inbox));
        }
    })?;
    Ok(Host { rx, port, waker })
}

fn connection(stream: TcpStream, inbox: Inbox) {
    let Ok(read) = stream.try_clone() else { return };
    let mut out = stream;
    for line in BufReader::new(read).lines() {
        let Ok(line) = line else { return };
        if line.trim().is_empty() {
            continue;
        }
        // A browser can POST to a loopback port from any web page, and a
        // body of JSON lines would otherwise run after the header lines
        // failed to parse. The protocol never sends HTTP: hang up.
        if looks_like_http(&line) {
            return;
        }
        let resp = handle_line(&line, &inbox);
        if let Some(resp) = resp {
            let mut s = resp.to_string();
            s.push('\n');
            if out.write_all(s.as_bytes()).is_err() {
                return;
            }
        }
    }
}

/// An HTTP request line (`POST / HTTP/1.1`) or header (`Host: …`): never
/// a JSON-RPC line, which starts with `{` or `[`.
pub fn looks_like_http(line: &str) -> bool {
    let t = line.trim_start();
    if t.starts_with('{') || t.starts_with('[') {
        return false;
    }
    t.contains(" HTTP/") || t.split_once(':').is_some_and(|(k, _)| !k.is_empty() && k.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-'))
}

/// One request line to its response (None for a notification).
pub fn handle_line(line: &str, inbox: &Inbox) -> Option<Value> {
    let req: Value = match serde_json::from_str(line) {
        Ok(v) => v,
        Err(e) => return Some(error(Value::Null, -32700, &format!("parse error: {e}"))),
    };
    let id = req.get("id").cloned();
    let reply = |r: Result<Value, ApiError>| -> Option<Value> {
        let id = id.clone()?;
        Some(match r {
            Ok(v) => json!({ "jsonrpc": "2.0", "id": id, "result": v }),
            Err(e) => error(id, e.code, &e.message),
        })
    };
    let Some(name) = req.get("method").and_then(|m| m.as_str()) else {
        return reply(Err(ApiError { code: -32600, message: "no method".into() }));
    };
    let Some(m) = method(name) else {
        return reply(Err(ApiError { code: -32601, message: format!("unknown method {name}") }));
    };
    let params = req.get("params").cloned().unwrap_or_else(|| json!({}));
    if let Err(e) = validate(&m, &params) {
        return reply(Err(e));
    }
    if name == "api_schema" {
        return reply(Ok(crate::schema()));
    }
    let (rtx, rrx) = mpsc::channel();
    if inbox.send(Call { method: name.into(), params, reply: rtx }).is_err() {
        return reply(Err(ApiError::app("the app is shutting down")));
    }
    reply(match rrx.recv_timeout(ANSWER_TIMEOUT) {
        Ok(r) => r,
        Err(_) => Err(ApiError::app("the app did not answer in time")),
    })
}

fn error(id: Value, code: i64, msg: &str) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": msg } })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// A web page POSTing to the loopback port: the JSON line in its body
    /// must never reach the app, and a plain client still works.
    #[test]
    fn an_http_request_is_hung_up_on() {
        let host = serve(0).unwrap();
        let mut s = TcpStream::connect((Ipv4Addr::LOCALHOST, host.port)).unwrap();
        let body = "\n{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"app_info\",\"params\":{}}\n";
        let req = format!(
            "POST / HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: text/plain\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        s.write_all(req.as_bytes()).unwrap();
        let mut rest = String::new();
        let _ = std::io::Read::read_to_string(&mut s, &mut rest);
        assert!(host.rx.recv_timeout(Duration::from_millis(300)).is_err(), "the body ran: {rest}");
        assert!(looks_like_http("Content-Length: 5") && !looks_like_http("{\"a\":\"b: c\"}"));
    }

    /// An idle app only sees a call if the call wakes it: the waker rings
    /// once per call, after the call is queued (so the woken loop finds it).
    #[test]
    fn a_call_wakes_the_app() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let host = serve(0).unwrap();
        let (wtx, wrx) = mpsc::channel::<bool>();
        let wtx = Mutex::new(wtx);
        let rings = Arc::new(AtomicUsize::new(0));
        let r2 = rings.clone();
        // The waker checks the call is already waiting when it rings.
        let host_rx = Arc::new(Mutex::new(host.rx));
        let rx2 = host_rx.clone();
        host.waker.lock().unwrap().replace(Arc::new(move || {
            r2.fetch_add(1, Ordering::SeqCst);
            let call = rx2.lock().unwrap().try_recv();
            let found = call.is_ok();
            if let Ok(c) = call {
                c.answer(Ok(json!({ "ok": true })));
            }
            let _ = wtx.lock().unwrap().send(found);
        }));
        let mut s = TcpStream::connect((Ipv4Addr::LOCALHOST, host.port)).unwrap();
        for id in 1..=2 {
            let line = format!("{{\"jsonrpc\":\"2.0\",\"id\":{id},\"method\":\"app_info\",\"params\":{{}}}}\n");
            s.write_all(line.as_bytes()).unwrap();
            assert_eq!(wrx.recv_timeout(Duration::from_secs(5)), Ok(true), "call {id}: no wake, or woke before the call was queued");
        }
        let mut lines = BufReader::new(s).lines();
        for _ in 0..2 {
            let resp: Value = serde_json::from_str(&lines.next().unwrap().unwrap()).unwrap();
            assert_eq!(resp["result"]["ok"], json!(true), "{resp}");
        }
        assert_eq!(rings.load(Ordering::SeqCst), 2);
    }
}
