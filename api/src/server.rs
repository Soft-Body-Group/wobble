//! The socket side of the app: accepts local connections, reads one
//! JSON-RPC request per line, and hands each to the app as a [`Call`] over a
//! channel. The app drains that channel between frames and answers through
//! the call's reply sender; the connection thread waits for the answer and
//! writes it back. So nothing the protocol does ever runs on the render
//! path, and a slow or dead client costs the app nothing.

use crate::{method, validate, ApiError};
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::net::{Ipv4Addr, Shutdown, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// How long a connection waits for the app to answer before giving up.
/// Long enough for `viewer_capture` on a cold 4K frame.
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

/// The listening end the app holds.
pub struct Host {
    pub rx: Receiver<Call>,
    pub port: u16,
}

/// Listen on 127.0.0.1:`port` (0 picks a free one; [`Host::port`] says
/// which). Loopback only: the protocol can open and save files, so it is
/// never offered to the network.
pub fn serve(port: u16) -> std::io::Result<Host> {
    let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, port))?;
    let port = listener.local_addr()?.port();
    let (tx, rx) = mpsc::channel();
    std::thread::Builder::new().name("api-accept".into()).spawn(move || {
        for stream in listener.incoming().flatten() {
            let tx = tx.clone();
            let _ = std::thread::Builder::new()
                .name("api-conn".into())
                .spawn(move || connection(stream, tx));
        }
    })?;
    Ok(Host { rx, port })
}

/// The network listener's switch: `close` stops accepting and drops every
/// open connection, so turning sharing off really closes the door.
#[derive(Clone, Default)]
pub struct NetHandle {
    stop: Arc<AtomicBool>,
    open: Arc<Mutex<Vec<TcpStream>>>,
}

impl NetHandle {
    pub fn close(&self) {
        self.stop.store(true, Ordering::Release);
        for s in self.open.lock().unwrap_or_else(|e| e.into_inner()).drain(..) {
            let _ = s.shutdown(Shutdown::Both);
        }
    }
}

/// Listen on EVERY interface (`port` 0 picks one) for other machines. A
/// network connection must `auth` with `key` before anything else: each
/// call before that is refused (-32001), and a wrong key is refused too.
/// The API can open and save files, so this is only ever started by the
/// user's own switch, and only with a key.
pub fn serve_network(port: u16, key: String) -> std::io::Result<(Host, NetHandle)> {
    let listener = TcpListener::bind((Ipv4Addr::UNSPECIFIED, port))?;
    listener.set_nonblocking(true)?;
    let port = listener.local_addr()?.port();
    let (tx, rx) = mpsc::channel();
    let handle = NetHandle::default();
    let h = handle.clone();
    std::thread::Builder::new().name("api-net-accept".into()).spawn(move || {
        while !h.stop.load(Ordering::Acquire) {
            match listener.accept() {
                Ok((stream, _)) => {
                    let _ = stream.set_nonblocking(false);
                    if let Ok(c) = stream.try_clone() {
                        h.open.lock().unwrap_or_else(|e| e.into_inner()).push(c);
                    }
                    let (tx, key) = (tx.clone(), key.clone());
                    let _ = std::thread::Builder::new()
                        .name("api-net-conn".into())
                        .spawn(move || connection_with(stream, tx, Some(key)));
                }
                Err(_) => std::thread::sleep(Duration::from_millis(100)),
            }
        }
    })?;
    Ok((Host { rx, port }, handle))
}

/// Two strings equal, in time that does not depend on where they differ.
fn same(a: &str, b: &str) -> bool {
    a.len() == b.len() && a.bytes().zip(b.bytes()).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// One line on a network connection: `auth` unlocks it, anything before
/// that is refused; after it, a call is handled as a local one.
pub fn handle_net_line(line: &str, tx: &Sender<Call>, key: &str, authed: &mut bool) -> Option<Value> {
    let req: Value = match serde_json::from_str(line) {
        Ok(v) => v,
        Err(e) => return Some(error(Value::Null, -32700, &format!("parse error: {e}"))),
    };
    let id = req.get("id").cloned().unwrap_or(Value::Null);
    if req.get("method").and_then(|m| m.as_str()) == Some("auth") {
        let given = req.pointer("/params/key").and_then(|k| k.as_str()).unwrap_or("");
        *authed = same(given, key);
        return Some(if *authed {
            json!({ "jsonrpc": "2.0", "id": id, "result": { "ok": true } })
        } else {
            error(id, -32001, "wrong pairing code")
        });
    }
    if !*authed {
        return Some(error(id, -32001, "send auth with this wobble's pairing code first"));
    }
    handle_line(line, tx)
}

fn connection(stream: TcpStream, tx: Sender<Call>) {
    connection_with(stream, tx, None)
}

fn connection_with(stream: TcpStream, tx: Sender<Call>, key: Option<String>) {
    let Ok(read) = stream.try_clone() else { return };
    let mut out = stream;
    let mut authed = false;
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
        let resp = match &key {
            Some(k) => handle_net_line(&line, &tx, k, &mut authed),
            None => handle_line(&line, &tx),
        };
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
pub fn handle_line(line: &str, tx: &Sender<Call>) -> Option<Value> {
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
    if tx.send(Call { method: name.into(), params, reply: rtx }).is_err() {
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
    use std::io::{BufRead, BufReader, Write};

    fn call(out: &mut TcpStream, read: &mut BufReader<TcpStream>, method: &str, params: Value) -> Value {
        let mut l = json!({ "jsonrpc": "2.0", "id": 1, "method": method, "params": params }).to_string();
        l.push('\n');
        out.write_all(l.as_bytes()).unwrap();
        let mut resp = String::new();
        read.read_line(&mut resp).unwrap();
        serde_json::from_str(&resp).unwrap()
    }

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

    /// From another machine nothing gets through without the pairing code:
    /// a call before `auth` and a wrong code are refused, the right code
    /// lets calls reach the app, and closing the switch drops the connection.
    #[test]
    fn the_network_listener_wants_the_pairing_code() {
        let (host, handle) = serve_network(0, "ABCD-2345".into()).unwrap();
        // The app's side: answer every call that arrives.
        let rx = host.rx;
        std::thread::spawn(move || {
            for c in rx {
                let m = c.method.clone();
                c.answer(Ok(json!({ "answered": m })));
            }
        });
        let s = TcpStream::connect((Ipv4Addr::LOCALHOST, host.port)).unwrap();
        let mut read = BufReader::new(s.try_clone().unwrap());
        let mut out = s;
        let r = call(&mut out, &mut read, "app_info", json!({}));
        assert_eq!(r["error"]["code"], -32001, "{r}");
        let r = call(&mut out, &mut read, "auth", json!({ "key": "WRONG-0000" }));
        assert_eq!(r["error"]["code"], -32001);
        let r = call(&mut out, &mut read, "app_info", json!({}));
        assert_eq!(r["error"]["code"], -32001, "a wrong code unlocks nothing");
        let r = call(&mut out, &mut read, "auth", json!({ "key": "ABCD-2345" }));
        assert_eq!(r["result"]["ok"], true);
        let r = call(&mut out, &mut read, "app_info", json!({}));
        assert_eq!(r["result"]["answered"], "app_info");
        handle.close();
        let mut rest = String::new();
        assert_eq!(read.read_line(&mut rest).unwrap_or(0), 0, "closed by the switch");
    }

    #[test]
    fn same_compares_whole_strings() {
        assert!(same("abc", "abc"));
        assert!(!same("abc", "abd"));
        assert!(!same("abc", "abcd"));
    }
}
