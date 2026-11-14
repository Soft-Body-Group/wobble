//! The MCP bridge: Model Context Protocol over stdio on one side, the
//! app's socket on the other. Every protocol method is an MCP tool of the
//! same name with the same params schema, so the bridge knows nothing about
//! editing and cannot drift from the app — a method added to the table is a
//! tool the next time an agent connects. (wobble's bridge, the same shape.)

use crate::client::Client;
use crate::methods;
use serde_json::{json, Value};
use std::io::{BufRead, Write};

/// What an agent is told on connect: the few things that are not obvious
/// from the tool list.
pub const INSTRUCTIONS: &str = "Drives a running Scrub (a video editor; it listens unless started with --no-api). \
Start with app_info and timeline_get. Tracks are V1 V2 V3 (V3 on top) and A1 A2 A3 (V1-V3's sound) and M (music). \
A clip is named by {track, start}: the record frame it starts on, as timeline_get lists it; cuts by 'at'. \
Three-point editing: media_open a file into the source, marks_set in/out on the source and/or record side, \
targets_set, then edit (insert / overwrite; four marks fit to fill). \
Use viewer_capture to SEE the record (or source) picture before moving on. \
Group several edits in one `batch` (one undo step, all or nothing). Every mutating call is one undo step; \
undo / redo walk them. render_start runs in the background: poll render_status.";

pub struct Bridge {
    port: u16,
    client: Option<Client>,
}

impl Bridge {
    pub fn new(port: u16) -> Self {
        Self { port, client: None }
    }

    /// Serve MCP on a line-oriented reader/writer pair (stdin/stdout).
    pub fn run(&mut self, input: impl BufRead, mut output: impl Write) -> std::io::Result<()> {
        for line in input.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let resp = match serde_json::from_str::<Value>(&line) {
                Ok(msg) => self.handle(&msg),
                Err(e) => Some(json!({ "jsonrpc": "2.0", "id": null,
                    "error": { "code": -32700, "message": format!("parse error: {e}") } })),
            };
            if let Some(r) = resp {
                writeln!(output, "{r}")?;
                output.flush()?;
            }
        }
        Ok(())
    }

    /// One MCP message to its response (None for a notification).
    pub fn handle(&mut self, msg: &Value) -> Option<Value> {
        let id = msg.get("id").cloned()?;
        let method = msg["method"].as_str().unwrap_or_default();
        let result = match method {
            "initialize" => json!({
                "protocolVersion": msg["params"]["protocolVersion"].as_str().unwrap_or("2025-06-18"),
                "capabilities": { "tools": {} },
                "serverInfo": { "name": "scrub", "version": env!("CARGO_PKG_VERSION") },
                "instructions": INSTRUCTIONS,
            }),
            "ping" => json!({}),
            "tools/list" => json!({ "tools": methods().iter().map(|m| json!({
                "name": m.name,
                "description": m.summary,
                "inputSchema": m.params,
                "annotations": {
                    "readOnlyHint": m.read_only(),
                    "destructiveHint": m.destructive(),
                    // Loopback to one app: nothing here reaches the network.
                    "openWorldHint": false,
                },
            })).collect::<Vec<_>>() }),
            "tools/call" => {
                let name = msg["params"]["name"].as_str().unwrap_or_default();
                let args = msg["params"].get("arguments").cloned().unwrap_or_else(|| json!({}));
                self.call_tool(name, args)
            }
            _ => {
                return Some(json!({ "jsonrpc": "2.0", "id": id,
                    "error": { "code": -32601, "message": format!("unknown method {method}") } }))
            }
        };
        Some(json!({ "jsonrpc": "2.0", "id": id, "result": result }))
    }

    fn call_tool(&mut self, name: &str, args: Value) -> Value {
        let r = self.call_app(name, args);
        match r {
            Ok(v) => json!({ "content": content_of(v) }),
            Err(e) => json!({ "content": [{ "type": "text", "text": e }], "isError": true }),
        }
    }

    /// Dial lazily and redial once: the app may have been started after the
    /// agent, or restarted under it.
    fn call_app(&mut self, name: &str, args: Value) -> Result<Value, String> {
        for attempt in 0..2 {
            if self.client.is_none() {
                match Client::connect(self.port) {
                    Ok(c) => self.client = Some(c),
                    Err(e) => {
                        return Err(format!(
                            "no Scrub listening on port {} ({e}). Start Scrub (it listens by default; \
                             `--api PORT` with SCRUB_API_PORT set to match picks another port).",
                            self.port
                        ))
                    }
                }
            }
            match self.client.as_mut().unwrap().call(name, args.clone()) {
                Ok(v) => return Ok(v),
                Err(e) if e.code == -32099 && attempt == 0 => self.client = None,
                Err(e) => return Err(e.message),
            }
        }
        Err("lost the connection to Scrub".into())
    }
}

/// A result as MCP content: a PNG becomes an image block (the rest of the
/// object, if any, a text block beside it), everything else pretty JSON.
fn content_of(mut v: Value) -> Vec<Value> {
    let mut out = Vec::new();
    if let Some(png) = v.as_object_mut().and_then(|o| o.remove("png_base64")) {
        out.push(json!({ "type": "image", "data": png, "mimeType": "image/png" }));
    }
    let empty = v.as_object().is_some_and(|o| o.is_empty());
    if !(empty && !out.is_empty()) {
        out.push(json!({ "type": "text", "text": serde_json::to_string_pretty(&v).unwrap_or_default() }));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::serve;

    fn call(b: &mut Bridge, id: u64, method: &str, params: Value) -> Value {
        b.handle(&json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params })).unwrap()
    }

    #[test]
    fn the_handshake_lists_every_method_as_a_tool() {
        let mut b = Bridge::new(1);
        let init = call(&mut b, 1, "initialize", json!({ "protocolVersion": "2025-06-18" }));
        assert_eq!(init["result"]["protocolVersion"], "2025-06-18");
        assert!(b.handle(&json!({ "jsonrpc": "2.0", "method": "notifications/initialized" })).is_none());
        let tools = call(&mut b, 2, "tools/list", json!({}));
        let names: Vec<&str> =
            tools["result"]["tools"].as_array().unwrap().iter().map(|t| t["name"].as_str().unwrap()).collect();
        assert_eq!(names.len(), methods().len());
        assert!(names.contains(&"viewer_capture"));
    }

    #[test]
    fn a_missing_app_is_a_tool_error_that_says_how_to_start_it() {
        // Port 1 is privileged and never ours: nothing answers.
        let mut b = Bridge::new(1);
        let r = call(&mut b, 3, "tools/call", json!({ "name": "timeline_get", "arguments": {} }));
        assert_eq!(r["result"]["isError"], true);
        assert!(r["result"]["content"][0]["text"].as_str().unwrap().contains("no Scrub listening"));
    }

    #[test]
    fn a_png_result_becomes_an_image_block() {
        let host = serve(0).unwrap();
        let port = host.port;
        std::thread::spawn(move || {
            for c in host.rx {
                c.answer(Ok(json!({ "png_base64": "iVBORw0KGgo=", "width": 4 })));
            }
        });
        let mut b = Bridge::new(port);
        let r = call(&mut b, 4, "tools/call", json!({ "name": "viewer_capture", "arguments": {} }));
        let content = r["result"]["content"].as_array().unwrap();
        assert_eq!(content[0]["type"], "image");
        assert_eq!(content[0]["mimeType"], "image/png");
        assert!(content[1]["text"].as_str().unwrap().contains("width"));
    }
}
