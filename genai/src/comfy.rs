//! A local ComfyUI: free, on your own GPU, any checkpoint you have.
//!
//! The step carries a workflow exported with "Save (API format)", and marks
//! where wobble's values go with placeholders — a string value that is
//! EXACTLY one of:
//!
//! | placeholder    | becomes                                      |
//! |----------------|----------------------------------------------|
//! | `{{prompt}}`   | the text on the prompt port                  |
//! | `{{seed}}`     | the step's seed, as a number                 |
//! | `{{width}}`    | the comp's width (a number)                  |
//! | `{{height}}`   | the comp's height (a number)                 |
//! | `{{image}}`    | the image port, uploaded; a LoadImage's name |
//! | `{{mask}}`     | the mask port, uploaded                      |
//!
//! `{{prompt}}` also substitutes INSIDE a longer string, so a prompt can be
//! wrapped in house style text. The first image a SaveImage (or any output
//! node) writes is the result.
//!
//! Endpoints: `POST /upload/image` (multipart), `POST /prompt`,
//! `GET /history/<id>`, `GET /view`, `POST /interrupt` to cancel.

use anyhow::{anyhow, bail, Context, Result};
use serde_json::{json, Value};

use crate::http::Http;
use crate::service::{self, Cx, Phase};

pub struct Comfy {
    pub base: String,
}

/// The values a workflow's placeholders take.
#[derive(Debug, Default, Clone)]
pub struct Subs {
    pub prompt: Option<String>,
    pub seed: u32,
    pub width: u32,
    pub height: u32,
    /// PNG bytes for `{{image}}` / `{{mask}}`.
    pub image: Option<Vec<u8>>,
    pub mask: Option<Vec<u8>>,
}

/// Which placeholders a workflow uses.
#[derive(Debug, Default, PartialEq)]
pub struct Wants {
    pub prompt: bool,
    pub image: bool,
    pub mask: bool,
}

pub fn wants(workflow: &Value) -> Wants {
    let mut w = Wants::default();
    walk(workflow, &mut |s| {
        w.prompt |= s.contains("{{prompt}}");
        w.image |= s == "{{image}}";
        w.mask |= s == "{{mask}}";
    });
    w
}

fn walk(v: &Value, f: &mut impl FnMut(&str)) {
    match v {
        Value::String(s) => f(s),
        Value::Array(a) => a.iter().for_each(|x| walk(x, f)),
        Value::Object(o) => o.values().for_each(|x| walk(x, f)),
        _ => {}
    }
}

/// Replace the placeholders. `image`/`mask` are the names ComfyUI gave the
/// uploads.
pub fn substitute(v: &mut Value, subs: &Subs, image: Option<&str>, mask: Option<&str>) {
    match v {
        Value::String(s) => {
            let new = match s.as_str() {
                "{{seed}}" => Some(json!(subs.seed)),
                "{{width}}" => Some(json!(subs.width)),
                "{{height}}" => Some(json!(subs.height)),
                "{{image}}" => image.map(|n| json!(n)),
                "{{mask}}" => mask.map(|n| json!(n)),
                _ if s.contains("{{prompt}}") => {
                    Some(json!(s.replace("{{prompt}}", subs.prompt.as_deref().unwrap_or(""))))
                }
                _ => None,
            };
            if let Some(n) = new {
                *v = n;
            }
        }
        Value::Array(a) => a.iter_mut().for_each(|x| substitute(x, subs, image, mask)),
        Value::Object(o) => o.values_mut().for_each(|x| substitute(x, subs, image, mask)),
        _ => {}
    }
}

/// Parse a workflow file and refuse the UI format early — it is the one
/// mistake everybody makes, and ComfyUI's own error for it is opaque.
pub fn parse_workflow(text: &str) -> Result<Value> {
    let v: Value = serde_json::from_str(text).context("the workflow is not JSON")?;
    if v.get("nodes").is_some_and(|n| n.is_array()) && v.get("links").is_some() {
        bail!("this is a UI workflow; export it with \"Save (API format)\"");
    }
    let ok = v.as_object().is_some_and(|o| {
        !o.is_empty() && o.values().all(|n| n.get("class_type").is_some())
    });
    if !ok {
        bail!("not an API-format workflow (every entry needs a class_type)");
    }
    Ok(v)
}

impl Comfy {
    pub fn new(base: &str) -> Self {
        Self { base: base.trim_end_matches('/').into() }
    }

    fn upload(&self, http: &Http, name: &str, png: &[u8]) -> Result<String> {
        let v = http.post_multipart(
            &format!("{}/upload/image", self.base),
            &[("type", "input"), ("overwrite", "true")],
            ("image", name, "image/png", png),
        )?;
        let n = v.get("name").and_then(|n| n.as_str()).ok_or_else(|| anyhow!("upload: no name"))?;
        let sub = v.get("subfolder").and_then(|s| s.as_str()).unwrap_or("");
        Ok(if sub.is_empty() { n.to_string() } else { format!("{sub}/{n}") })
    }

    /// Run a workflow to its first output image. Returns the bytes.
    pub fn run(&self, workflow: &Value, subs: &Subs, cx: &Cx) -> Result<(Vec<u8>, Option<String>)> {
        let w = wants(workflow);
        if w.image && subs.image.is_none() {
            bail!("the workflow has an {{{{image}}}} placeholder: wire the step's image port");
        }
        if w.mask && subs.mask.is_none() {
            bail!("the workflow has a {{{{mask}}}} placeholder: wire the step's mask port");
        }
        if w.prompt && subs.prompt.is_none() {
            bail!("the workflow has a {{{{prompt}}}} placeholder: wire the step's prompt port");
        }
        let image = match (&subs.image, w.image) {
            (Some(b), true) => Some(self.upload(cx.http, &format!("wobble_{}_image.png", subs.seed), b)?),
            _ => None,
        };
        let mask = match (&subs.mask, w.mask) {
            (Some(b), true) => Some(self.upload(cx.http, &format!("wobble_{}_mask.png", subs.seed), b)?),
            _ => None,
        };
        let mut prompt = workflow.clone();
        substitute(&mut prompt, subs, image.as_deref(), mask.as_deref());

        let v = cx.http.post_json(
            &format!("{}/prompt", self.base),
            &[],
            &json!({ "prompt": prompt, "client_id": "wobble" }),
        )?;
        let id = v
            .get("prompt_id")
            .and_then(|p| p.as_str())
            .ok_or_else(|| anyhow!("ComfyUI accepted nothing: {v}"))?
            .to_string();
        if !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') {
            bail!("ComfyUI returned a malformed prompt id");
        }
        (cx.phase)(Phase::Queued(None));

        let found = service::poll(cx, || {
            let h = cx.http.get_json(&format!("{}/history/{id}", self.base), &[])?;
            let Some(entry) = h.get(&id) else {
                return Ok(None);
            };
            (cx.phase)(Phase::Running);
            let status = entry.get("status");
            if status.and_then(|s| s.get("status_str")).and_then(|s| s.as_str()) == Some("error") {
                let msg = status
                    .and_then(|s| s.get("messages"))
                    .and_then(|m| m.as_array())
                    .and_then(|m| {
                        m.iter().find(|x| x.get(0).and_then(|t| t.as_str()) == Some("execution_error"))
                    })
                    .and_then(|x| x.get(1))
                    .and_then(|d| d.get("exception_message"))
                    .and_then(|e| e.as_str())
                    .unwrap_or("the workflow failed");
                bail!("ComfyUI: {}", msg.trim());
            }
            let done = status
                .and_then(|s| s.get("completed"))
                .and_then(|c| c.as_bool())
                .unwrap_or(true);
            if !done {
                return Ok(None);
            }
            Ok(first_image(entry.get("outputs").unwrap_or(&Value::Null)))
        });
        let (file, sub, kind) = match found {
            Ok(Some(f)) => f,
            Ok(None) => bail!("the workflow finished without saving an image"),
            Err(e) => {
                let _ = cx.http.post_json(&format!("{}/interrupt", self.base), &[], &json!({}));
                return Err(e);
            }
        };
        let url = format!(
            "{}/view?filename={}&subfolder={}&type={}",
            self.base,
            pct(&file),
            pct(&sub),
            pct(&kind)
        );
        cx.http.download(&url)
    }
}

/// `Ok(Some(None))` in the poll above means "finished, no image"; keep the
/// types honest with this helper.
fn first_image(outputs: &Value) -> Option<Option<(String, String, String)>> {
    let mut best: Option<(String, String, String)> = None;
    if let Some(o) = outputs.as_object() {
        // Node ids are numbers-as-strings; take the lowest for a stable pick.
        let mut ids: Vec<&String> = o.keys().collect();
        ids.sort_by_key(|k| k.parse::<u64>().unwrap_or(u64::MAX));
        for id in ids {
            let Some(imgs) = o[id].get("images").and_then(|i| i.as_array()) else { continue };
            for im in imgs {
                let f = im.get("filename").and_then(|x| x.as_str()).unwrap_or("").to_string();
                let s = im.get("subfolder").and_then(|x| x.as_str()).unwrap_or("").to_string();
                let t = im.get("type").and_then(|x| x.as_str()).unwrap_or("output").to_string();
                if f.is_empty() {
                    continue;
                }
                // A SaveImage ("output") beats a preview ("temp").
                if t == "output" {
                    return Some(Some((f, s, t)));
                }
                best.get_or_insert((f, s, t));
            }
        }
    }
    Some(best)
}

/// Percent-encode a query value.
fn pct(s: &str) -> String {
    let mut o = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => o.push(b as char),
            _ => o.push_str(&format!("%{b:02X}")),
        }
    }
    o
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wf() -> Value {
        json!({
            "3": {"class_type": "KSampler", "inputs": {"seed": "{{seed}}", "steps": 20}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "cinematic, {{prompt}}"}},
            "10": {"class_type": "LoadImage", "inputs": {"image": "{{image}}"}},
            "5": {"class_type": "EmptyLatentImage", "inputs": {"width": "{{width}}", "height": "{{height}}"}}
        })
    }

    #[test]
    fn placeholders_are_found_and_substituted_with_the_right_types() {
        let mut v = wf();
        assert_eq!(wants(&v), Wants { prompt: true, image: true, mask: false });
        let subs = Subs { prompt: Some("a moth".into()), seed: 42, width: 1024, height: 576, ..Default::default() };
        substitute(&mut v, &subs, Some("wobble_42_image.png"), None);
        assert_eq!(v["3"]["inputs"]["seed"], json!(42));
        assert_eq!(v["6"]["inputs"]["text"], json!("cinematic, a moth"));
        assert_eq!(v["10"]["inputs"]["image"], json!("wobble_42_image.png"));
        assert_eq!(v["5"]["inputs"]["width"], json!(1024));
    }

    #[test]
    fn the_ui_format_is_refused_with_a_useful_message() {
        let e = parse_workflow(r#"{"nodes":[],"links":[]}"#).unwrap_err();
        assert!(e.to_string().contains("API format"));
        assert!(parse_workflow(&wf().to_string()).is_ok());
    }

    #[test]
    fn a_save_beats_a_preview() {
        let out = json!({
            "9": {"images": [{"filename": "a.png", "subfolder": "", "type": "temp"}]},
            "12": {"images": [{"filename": "b.png", "subfolder": "x", "type": "output"}]}
        });
        assert_eq!(first_image(&out), Some(Some(("b.png".into(), "x".into(), "output".into()))));
    }

    #[test]
    fn query_values_are_encoded() {
        assert_eq!(pct("a b/c.png"), "a%20b%2Fc.png");
    }
}
