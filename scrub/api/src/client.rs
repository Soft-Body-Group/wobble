//! A blocking client: what the MCP bridge uses, what the app's `--test-api`
//! drives the app through, and what a Rust tool would.

use crate::ApiError;
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::net::{Ipv4Addr, TcpStream};

pub struct Client {
    reader: BufReader<TcpStream>,
    writer: TcpStream,
    next_id: u64,
}

impl Client {
    pub fn connect(port: u16) -> std::io::Result<Self> {
        let s = TcpStream::connect((Ipv4Addr::LOCALHOST, port))?;
        s.set_nodelay(true)?;
        Ok(Self { reader: BufReader::new(s.try_clone()?), writer: s, next_id: 1 })
    }

    /// Call a method. A transport failure comes back as code -32099 so the
    /// caller can tell "the app said no" from "there is no app".
    pub fn call(&mut self, method: &str, params: Value) -> Result<Value, ApiError> {
        let id = self.next_id;
        self.next_id += 1;
        let mut line = json!({ "jsonrpc": "2.0", "id": id, "method": method, "params": params }).to_string();
        line.push('\n');
        let io = |e: std::io::Error| ApiError { code: -32099, message: format!("connection: {e}") };
        self.writer.write_all(line.as_bytes()).map_err(io)?;
        let mut resp = String::new();
        if self.reader.read_line(&mut resp).map_err(io)? == 0 {
            return Err(ApiError { code: -32099, message: "connection closed".into() });
        }
        let v: Value = serde_json::from_str(&resp)
            .map_err(|e| ApiError { code: -32099, message: format!("bad response: {e}") })?;
        if let Some(e) = v.get("error") {
            return Err(ApiError {
                code: e["code"].as_i64().unwrap_or(-32000),
                message: e["message"].as_str().unwrap_or("error").to_string(),
            });
        }
        Ok(v.get("result").cloned().unwrap_or(Value::Null))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::server::serve;

    /// The whole loop with a stand-in app: validation happens before the
    /// app sees anything, `api_schema` needs no app at all, and a call the
    /// app answers comes back to the caller.
    #[test]
    fn a_call_round_trips_through_the_socket() {
        let host = serve(0).unwrap();
        let port = host.port;
        std::thread::spawn(move || {
            for call in host.rx {
                let r = match call.method.as_str() {
                    "playhead_set" => Ok(json!({ "frame": call.params["frame"] })),
                    _ => Err(ApiError::app("nope")),
                };
                call.answer(r);
            }
        });
        let mut c = Client::connect(port).unwrap();
        assert_eq!(c.call("playhead_set", json!({ "frame": 12 })).unwrap()["frame"], 12);
        assert_eq!(c.call("playhead_set", json!({})).unwrap_err().code, -32602);
        assert_eq!(c.call("warp_drive", json!({})).unwrap_err().code, -32601);
        assert_eq!(c.call("undo", json!({})).unwrap_err().message, "nope");
        let schema = c.call("api_schema", json!({})).unwrap();
        assert_eq!(schema["version"], crate::VERSION);
        assert!(schema["methods"].as_array().unwrap().len() > 10);
    }
}
