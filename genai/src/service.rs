//! The adapter every fal-shaped service implements: submit a request to an
//! endpoint, wait for it, hand back the output JSON — and say what it will
//! cost before any of that.
//!
//! fal directly (your key) and the wobble relay (credits) both implement
//! it, and both return fal's own output JSON, so nothing past this trait
//! knows which one paid. A new service is one more implementation; a new
//! MODEL on an existing one is one catalog row.

use anyhow::Result;
use serde_json::Value;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::http::Http;

/// What a job is doing, for the panel.
#[derive(Debug, Clone, PartialEq)]
pub enum Phase {
    /// Waiting for a worker; `Some(n)` = n ahead of us.
    Queued(Option<u64>),
    Running,
}

/// Everything a call needs besides its request.
pub struct Cx<'a> {
    pub http: &'a Http,
    pub cancel: &'a AtomicBool,
    pub phase: &'a dyn Fn(Phase),
    /// Longest a single job may take before we stop waiting (and cancel).
    pub deadline: Duration,
}

impl Cx<'_> {
    pub fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }
}

/// A price for one call, as the service answered it.
#[derive(Debug, Clone, PartialEq)]
pub struct Price {
    pub usd: f64,
    /// False when the service prices in a unit we could only estimate
    /// (seconds of GPU, say). The panel prefixes it with "≈".
    pub exact: bool,
    /// "per image", "per megapixel ×1.0"…
    pub note: String,
}

/// One call to price: an endpoint and the request we are about to send
/// (a service may price by output size or count).
pub struct QuoteItem<'a> {
    pub endpoint: &'a str,
    pub input: &'a Value,
    /// Output megapixels, when we know them ahead of the call.
    pub megapixels: Option<f64>,
    /// A video call's length and size, when the step makes one.
    pub video: Option<VideoCall>,
}

/// What a video call will make, for the units fal bills video in.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VideoCall {
    pub seconds: f64,
    /// Output pixels per frame, at the requested resolution and the
    /// comp's aspect (an estimate: the model picks the exact size).
    pub w: u32,
    pub h: u32,
    /// Is the resolution the endpoint's default (720p)? A per-second
    /// price is quoted for that; another resolution is billed at another
    /// rate the pricing API does not list, so the quote turns "≈".
    pub default_res: bool,
    /// What a per-second price is multiplied by for this call's settings
    /// (Wan 2.7 at 1080p bills half as much again as its listed 720p).
    pub rate: f64,
}

impl VideoCall {
    /// ByteDance's token count for a clip: h × w × seconds × 24 / 1024.
    pub fn tokens(&self) -> f64 {
        self.w as f64 * self.h as f64 * self.seconds * 24.0 / 1024.0
    }
}

/// A whole run's quote, and what the account holds (relay only).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Quote {
    pub items: Vec<Option<Price>>,
    pub balance_usd: Option<f64>,
}

impl Quote {
    pub fn total(&self) -> Option<f64> {
        self.items.iter().map(|p| p.as_ref().map(|p| p.usd)).sum()
    }
    pub fn exact(&self) -> bool {
        self.items.iter().all(|p| p.as_ref().is_some_and(|p| p.exact))
    }
}

/// A finished call: the service's output JSON (fal's shape, whoever
/// answered) and what was actually charged, when the service says.
#[derive(Debug, Clone, PartialEq)]
pub struct Ran {
    pub output: Value,
    pub charged_usd: Option<f64>,
}

pub trait Service: Send + Sync {
    /// "fal", "wobble relay".
    fn name(&self) -> &'static str;
    /// Run one request to completion. `max_usd` is the price the user
    /// approved; a service that can enforce it (the relay) refuses a job
    /// priced above it. `megapixels` is the output size the quote assumed,
    /// so a service pricing by area prices the job exactly as it quoted it.
    /// Must honour `cx.cancel` between polls, and cancel the remote job
    /// when it stops waiting.
    fn run(&self, endpoint: &str, input: &Value, max_usd: Option<f64>, megapixels: Option<f64>, cx: &Cx) -> Result<Ran>;
    fn quote(&self, http: &Http, items: &[QuoteItem]) -> Result<Quote>;
}

/// Poll until `step` says done, sleeping 0.4 s growing to 2 s, honouring
/// cancel and the deadline. `step` returns `Ok(Some(v))` to finish.
pub fn poll<T>(cx: &Cx, mut step: impl FnMut() -> Result<Option<T>>) -> Result<T> {
    let start = Instant::now();
    let mut wait = Duration::from_millis(400);
    loop {
        if let Some(v) = step()? {
            return Ok(v);
        }
        if cx.cancelled() {
            anyhow::bail!("cancelled");
        }
        if start.elapsed() > cx.deadline {
            anyhow::bail!("gave up waiting after {} s", cx.deadline.as_secs());
        }
        // Sleep in slices so a cancel lands within 100 ms.
        let until = Instant::now() + wait;
        while Instant::now() < until {
            if cx.cancelled() {
                anyhow::bail!("cancelled");
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        wait = (wait * 3 / 2).min(Duration::from_secs(2));
    }
}

/// The picture in a fal-shaped output: `images[0]`, else `image`. Returns
/// the URL, and whether the provider's safety filter flagged it (fal hands
/// back a BLACK image in that case, still billed — worth saying).
pub fn output_image(out: &Value) -> Option<(String, bool)> {
    output_image_at(out, None)
}

/// [`output_image`], or the picture under `key` when a model names its own
/// field (MoGe's `normal_map`, MiDaS's `depth_map`).
pub fn output_image_at(out: &Value, key: Option<&str>) -> Option<(String, bool)> {
    if let Some(k) = key {
        let url = out.get(k).and_then(|i| i.get("url")).and_then(|u| u.as_str())?.to_string();
        return Some((url, false));
    }
    let url = out
        .get("images")
        .and_then(|a| a.get(0))
        .and_then(|i| i.get("url"))
        .or_else(|| out.get("image").and_then(|i| i.get("url")))
        .and_then(|u| u.as_str())?
        .to_string();
    let nsfw = out
        .get("has_nsfw_concepts")
        .and_then(|a| a.get(0))
        .and_then(|b| b.as_bool())
        .unwrap_or(false);
    Some((url, nsfw))
}

/// The words in a language model's output (`output`), or why there are
/// none (`error`).
pub fn output_text(out: &Value) -> std::result::Result<String, String> {
    if let Some(e) = out.get("error").and_then(|e| e.as_str()).filter(|e| !e.is_empty()) {
        return Err(e.to_string());
    }
    match out.get("output").and_then(|o| o.as_str()).map(str::trim) {
        Some(t) if !t.is_empty() => Ok(t.to_string()),
        _ => Err("no text came back".into()),
    }
}

/// The video in a fal-shaped output: `video.url`, else `videos[0].url`.
pub fn output_video(out: &Value) -> Option<String> {
    out.get("video")
        .and_then(|v| v.get("url"))
        .or_else(|| out.get("videos").and_then(|a| a.get(0)).and_then(|v| v.get("url")))
        .and_then(|u| u.as_str())
        .map(String::from)
}

/// Turn a fal pricing row into a price for one call.
pub fn price_of(unit_price: f64, unit: &str, megapixels: Option<f64>, video: Option<VideoCall>) -> Price {
    let u = unit.trim().to_ascii_lowercase();
    let u = u.trim_end_matches('s');
    // "1k tokens", "1000 tokens", "million tokens", "1m tokens": the count
    // per price.
    let per_tokens = || -> Option<f64> {
        let t = u.strip_suffix("token")?.trim().trim_end_matches(['_', ' ']);
        Some(match t {
            "" => 1.0,
            "1k" | "k" | "1000" | "1,000" | "thousand" => 1e3,
            "1m" | "m" | "million" | "1000000" | "1,000,000" => 1e6,
            other => other.parse().ok()?,
        })
    };
    if let (Some(v), Some(per)) = (video, per_tokens()) {
        let usd = unit_price * v.tokens() / per;
        // The token count is ours, from a size the model only roughly
        // keeps: always an estimate.
        return Price { usd, exact: false, note: format!("≈{:.0}k tokens", v.tokens() / 1e3) };
    }
    match (u, megapixels, video) {
        ("second" | "sec" | "video_second", _, Some(v)) => Price {
            usd: unit_price * v.seconds * v.rate,
            exact: v.default_res,
            note: format!("{} s", v.seconds),
        },
        ("image" | "request" | "call" | "video" | "generation", _, _) => {
            Price { usd: unit_price, exact: true, note: format!("per {u}") }
        }
        ("megapixel", Some(mp), _) => {
            // fal rounds up to whole megapixels.
            let billed = mp.ceil().max(1.0);
            Price { usd: unit_price * billed, exact: true, note: format!("{billed} MP") }
        }
        _ => Price { usd: unit_price, exact: false, note: format!("per {unit}") },
    }
}

/// "$0.004", "$1.20".
pub fn usd(v: f64) -> String {
    if v < 0.1 {
        format!("${v:.3}")
    } else {
        format!("${v:.2}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn output_image_reads_both_fal_shapes_and_the_safety_flag() {
        let a = json!({"images":[{"url":"https://x/a.png"}],"has_nsfw_concepts":[true]});
        assert_eq!(output_image(&a), Some(("https://x/a.png".into(), true)));
        let b = json!({"image":{"url":"https://x/b.png"}});
        assert_eq!(output_image(&b), Some(("https://x/b.png".into(), false)));
        assert_eq!(output_image(&json!({"video":{}})), None);
    }

    #[test]
    fn output_video_reads_fal_s_file_shape() {
        let v = json!({"video":{"url":"https://x/v.mp4","content_type":"video/mp4"},"seed":3});
        assert_eq!(output_video(&v).as_deref(), Some("https://x/v.mp4"));
        assert_eq!(output_video(&json!({"images":[{"url":"https://x/a.png"}]})), None);
    }

    #[test]
    fn video_is_priced_per_second_or_by_tokens_and_never_silently() {
        let v = VideoCall { seconds: 5.0, w: 1280, h: 720, default_res: true, rate: 1.0 };
        let p = price_of(0.2419, "seconds", None, Some(v));
        assert!((p.usd - 1.2095).abs() < 1e-9 && p.exact, "{p:?}");
        // Another resolution bills at a rate the API does not list.
        let p = price_of(0.2419, "second", None, Some(VideoCall { default_res: false, ..v }));
        assert!(!p.exact);
        // Tokens: 1280 × 720 × 5 × 24 / 1024 = 108 000.
        let p = price_of(0.0214, "1k tokens", None, Some(v));
        assert!((p.usd - 0.0214 * 108.0).abs() < 1e-9 && !p.exact, "{p:?}");
        let p = price_of(21.4, "million tokens", None, Some(v));
        assert!((p.usd - 21.4 * 0.108).abs() < 1e-9);
        // A unit nobody taught us is shown as an estimate, never dropped.
        let p = price_of(0.5, "compute_unit", None, Some(v));
        assert!(!p.exact && p.usd > 0.0);
    }

    #[test]
    fn bria_s_generations_are_a_price_per_call() {
        let p = price_of(0.04, "generations", Some(8.0), None);
        assert!(p.exact && (p.usd - 0.04).abs() < 1e-9, "{p:?}");
    }

    #[test]
    fn megapixel_pricing_rounds_up_and_unknown_units_are_estimates() {
        let p = price_of(0.003, "megapixels", Some(1.1), None);
        assert!((p.usd - 0.006).abs() < 1e-9 && p.exact);
        let p = price_of(0.025, "image", None, None);
        assert!(p.exact && p.usd == 0.025);
        assert!(!price_of(0.001, "gpu_second", None, None).exact);
    }

    #[test]
    fn a_quote_is_only_exact_when_every_line_is() {
        let q = Quote {
            items: vec![Some(price_of(0.01, "image", None, None)), None],
            balance_usd: None,
        };
        assert_eq!(q.total(), None);
        assert!(!q.exact());
    }
}
