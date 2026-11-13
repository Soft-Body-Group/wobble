//! The little HTTP the providers need, on one ureq agent.
//!
//! Native only: on wasm32 `Http` is a stub whose every call fails with
//! "pipelines run in the desktop app" (ureq -> ring has no wasm toolchain,
//! and a browser demo sends nothing anywhere). Everything else in the crate
//! compiles on both, so the app carries, draws and saves a pipeline on the
//! web without a single cfg of its own.
//!
//! Two deliberate choices: a 4xx/5xx is NOT a transport error here (ureq's
//! default), because the provider's reason is in the body and "HTTP 422"
//! alone tells nobody which field it wanted; and every body read is capped,
//! because the URLs we download from come out of a response we did not
//! write.

use anyhow::{anyhow, bail, Context, Result};
use serde_json::Value;
#[cfg(not(target_arch = "wasm32"))]
use std::time::Duration;

/// Largest result we will download. A 4K PNG is ~50 MB; the longest clip
/// the catalog can ask for (30 s of 1080p H.264) is ~60-150 MB. 1 GiB
/// leaves room for a high-bitrate mode without letting a hostile URL fill
/// the disk.
pub const MAX_DOWNLOAD: u64 = 1 << 30;
/// Largest JSON we will read.
#[cfg(not(target_arch = "wasm32"))]
const MAX_JSON: u64 = 32 << 20;

pub struct Http {
    #[cfg(not(target_arch = "wasm32"))]
    agent: ureq::Agent,
}

#[cfg(target_arch = "wasm32")]
const NO_NET: &str = "running a pipeline needs the desktop app";

#[cfg(target_arch = "wasm32")]
impl Http {
    pub fn new() -> Self {
        Self {}
    }
    pub fn get_json(&self, _url: &str, _headers: &[(&str, &str)]) -> Result<Value> {
        bail!(NO_NET)
    }
    pub fn post_json(&self, _url: &str, _headers: &[(&str, &str)], _body: &Value) -> Result<Value> {
        bail!(NO_NET)
    }
    pub fn put_empty(&self, _url: &str, _headers: &[(&str, &str)]) -> Result<u16> {
        bail!(NO_NET)
    }
    pub fn get_text(&self, _url: &str, _headers: &[(&str, &str)]) -> Result<(u16, String)> {
        bail!(NO_NET)
    }
    pub fn post_multipart(
        &self,
        _url: &str,
        _fields: &[(&str, &str)],
        _file: (&str, &str, &str, &[u8]),
    ) -> Result<Value> {
        bail!(NO_NET)
    }
    pub fn download(&self, url: &str) -> Result<(Vec<u8>, Option<String>)> {
        match url.strip_prefix("data:") {
            Some(rest) => decode_data_uri(rest),
            None => bail!(NO_NET),
        }
    }
}

impl Default for Http {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Http {
    pub fn new() -> Self {
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .http_status_as_error(false)
            .timeout_connect(Some(Duration::from_secs(20)))
            // Per call, not per job: a job is many polls.
            .timeout_global(Some(Duration::from_secs(300)))
            .user_agent(concat!("wobble-genai/", env!("CARGO_PKG_VERSION")))
            .build()
            .into();
        Self { agent }
    }

    pub fn get_json(&self, url: &str, headers: &[(&str, &str)]) -> Result<Value> {
        let mut req = self.agent.get(url);
        for (k, v) in headers {
            req = req.header(*k, *v);
        }
        let resp = req.call().map_err(|e| anyhow!("GET {}: {e}", redact(url)))?;
        json_of(resp, "GET", url)
    }

    pub fn post_json(&self, url: &str, headers: &[(&str, &str)], body: &Value) -> Result<Value> {
        let mut req = self.agent.post(url).header("content-type", "application/json");
        for (k, v) in headers {
            req = req.header(*k, *v);
        }
        let text = serde_json::to_string(body)?;
        let resp = req.send(text).map_err(|e| anyhow!("POST {}: {e}", redact(url)))?;
        json_of(resp, "POST", url)
    }

    /// GET whatever comes back, error status and all: for a reply whose
    /// failure IS the answer (fal's result of a failed request).
    pub fn get_text(&self, url: &str, headers: &[(&str, &str)]) -> Result<(u16, String)> {
        let mut req = self.agent.get(url);
        for (k, v) in headers {
            req = req.header(*k, *v);
        }
        let mut resp = req.call().map_err(|e| anyhow!("GET {}: {e}", redact(url)))?;
        let status = resp.status().as_u16();
        let text = resp
            .body_mut()
            .with_config()
            .limit(MAX_JSON)
            .read_to_string()
            .with_context(|| format!("GET {}: reading the reply", redact(url)))?;
        Ok((status, text))
    }

    pub fn put_empty(&self, url: &str, headers: &[(&str, &str)]) -> Result<u16> {
        let mut req = self.agent.put(url);
        for (k, v) in headers {
            req = req.header(*k, *v);
        }
        let resp = req.send_empty().map_err(|e| anyhow!("PUT {}: {e}", redact(url)))?;
        Ok(resp.status().as_u16())
    }

    /// POST a multipart/form-data body: text fields plus one file.
    pub fn post_multipart(
        &self,
        url: &str,
        fields: &[(&str, &str)],
        file: (&str, &str, &str, &[u8]), // (field, filename, content type, bytes)
    ) -> Result<Value> {
        let boundary = "wobble-genai-7f3c1d2e9b8a";
        let mut body = Vec::with_capacity(file.3.len() + 1024);
        for (k, v) in fields {
            body.extend_from_slice(
                format!(
                    "--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n"
                )
                .as_bytes(),
            );
        }
        body.extend_from_slice(
            format!(
                "--{boundary}\r\nContent-Disposition: form-data; name=\"{}\"; filename=\"{}\"\r\nContent-Type: {}\r\n\r\n",
                file.0, file.1, file.2
            )
            .as_bytes(),
        );
        body.extend_from_slice(file.3);
        body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
        let resp = self
            .agent
            .post(url)
            .header("content-type", &format!("multipart/form-data; boundary={boundary}"))
            .send(&body[..])
            .map_err(|e| anyhow!("POST {}: {e}", redact(url)))?;
        json_of(resp, "POST", url)
    }

    /// Fetch a result: an http(s) URL or a `data:` URI (fal's `sync_mode`,
    /// and what a relay may inline). Returns the bytes and the content type
    /// if anyone said.
    pub fn download(&self, url: &str) -> Result<(Vec<u8>, Option<String>)> {
        if let Some(rest) = url.strip_prefix("data:") {
            return decode_data_uri(rest);
        }
        if !(url.starts_with("https://") || url.starts_with("http://")) {
            bail!("refusing to fetch a result from {}", redact(url));
        }
        let mut resp = self.agent.get(url).call().map_err(|e| anyhow!("GET {}: {e}", redact(url)))?;
        let status = resp.status().as_u16();
        if !(200..300).contains(&status) {
            bail!("GET {} returned HTTP {status}", redact(url));
        }
        let ct = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .map(|s| s.split(';').next().unwrap_or(s).trim().to_string());
        let bytes = resp
            .body_mut()
            .with_config()
            .limit(MAX_DOWNLOAD)
            .read_to_vec()
            .with_context(|| format!("reading {}", redact(url)))?;
        Ok((bytes, ct))
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn json_of(mut resp: ureq::http::Response<ureq::Body>, verb: &str, url: &str) -> Result<Value> {
    let status = resp.status().as_u16();
    let text = resp
        .body_mut()
        .with_config()
        .limit(MAX_JSON)
        .read_to_string()
        .with_context(|| format!("{verb} {}: reading the reply", redact(url)))?;
    if !(200..300).contains(&status) {
        bail!("{verb} {} returned HTTP {status}: {}", redact(url), reason(&text));
    }
    serde_json::from_str(&text)
        .with_context(|| format!("{verb} {}: the reply is not JSON", redact(url)))
}

/// The human part of an error body, else its first 300 characters. The
/// shapes, as fal documents them:
///
/// - a model error: `{"detail": [{loc, msg, type, url, …}]}` (422 for bad
///   input, 5xx for the model failing);
/// - a request error: `{"detail": "Request timed out", "error_type":
///   "request_timeout"}` (timeouts, runner failures);
/// - the platform API (pricing): `{"error": {"type", "message"}}`;
/// - the relay: `{"error": "…"}`.
pub fn reason(text: &str) -> String {
    if let Ok(v) = serde_json::from_str::<Value>(text) {
        match v.get("error") {
            Some(Value::String(s)) => return s.clone(),
            Some(Value::Object(e)) => {
                if let Some(m) = e.get("message").and_then(|m| m.as_str()) {
                    return m.to_string();
                }
            }
            _ => {}
        }
        match v.get("detail") {
            Some(Value::String(s)) => {
                return match v.get("error_type").and_then(|t| t.as_str()) {
                    Some(t) => format!("{s} ({t})"),
                    None => s.clone(),
                }
            }
            Some(Value::Array(items)) => {
                let parts: Vec<String> = items
                    .iter()
                    .map(|it| {
                        let loc = it
                            .get("loc")
                            .and_then(|l| l.as_array())
                            .map(|l| {
                                l.iter()
                                    .map(|x| x.as_str().map(String::from).unwrap_or(x.to_string()))
                                    .collect::<Vec<_>>()
                                    .join(".")
                            })
                            .unwrap_or_default();
                        let msg = it.get("msg").and_then(|m| m.as_str()).unwrap_or("?");
                        if loc.is_empty() {
                            msg.to_string()
                        } else {
                            format!("{loc}: {msg}")
                        }
                    })
                    .collect();
                if !parts.is_empty() {
                    return parts.join("; ");
                }
            }
            _ => {}
        }
    }
    text.chars().take(300).collect()
}

/// A URL fit for an error message: no query string (signed CDN links carry
/// their credentials there).
pub fn redact(url: &str) -> String {
    if url.starts_with("data:") {
        return "a data: URI".into();
    }
    url.split('?').next().unwrap_or(url).to_string()
}

fn decode_data_uri(rest: &str) -> Result<(Vec<u8>, Option<String>)> {
    use base64::Engine;
    let (meta, data) = rest.split_once(',').ok_or_else(|| anyhow!("malformed data: URI"))?;
    if !meta.ends_with(";base64") {
        bail!("only base64 data: URIs are supported");
    }
    let ct = meta.trim_end_matches(";base64");
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(data.trim())
        .context("decoding a data: URI")?;
    if bytes.len() as u64 > MAX_DOWNLOAD {
        bail!("result too large");
    }
    Ok((bytes, (!ct.is_empty()).then(|| ct.to_string())))
}

/// `data:<ct>;base64,<…>` — how an input image travels to a provider that
/// takes one inline (fal does, for any file input).
pub fn data_uri(content_type: &str, bytes: &[u8]) -> String {
    use base64::Engine;
    format!(
        "data:{content_type};base64,{}",
        base64::engine::general_purpose::STANDARD.encode(bytes)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fal_validation_errors_read_as_field_and_message() {
        let body = r#"{"detail":[{"loc":["body","image_url"],"msg":"field required","type":"missing"}]}"#;
        assert_eq!(reason(body), "body.image_url: field required");
        assert_eq!(reason(r#"{"detail":"nope"}"#), "nope");
        assert_eq!(reason(r#"{"error":"out of credits"}"#), "out of credits");
    }

    #[test]
    fn fal_request_and_platform_errors_read_too() {
        let timeout = r#"{"detail":"Request timed out","error_type":"request_timeout"}"#;
        assert_eq!(reason(timeout), "Request timed out (request_timeout)");
        let pricing = r#"{"error":{"type":"authorization_error","message":"Invalid API key"}}"#;
        assert_eq!(reason(pricing), "Invalid API key");
    }

    #[test]
    fn data_uris_round_trip() {
        let u = data_uri("image/png", b"\x89PNGabc");
        let (b, ct) = decode_data_uri(u.strip_prefix("data:").unwrap()).unwrap();
        assert_eq!(b, b"\x89PNGabc");
        assert_eq!(ct.as_deref(), Some("image/png"));
    }

    #[test]
    fn redact_drops_the_query() {
        assert_eq!(redact("https://cdn.x/a.png?sig=secret"), "https://cdn.x/a.png");
    }

    #[test]
    fn only_http_and_data_are_fetched() {
        assert!(Http::new().download("file:///etc/passwd").is_err());
    }
}
