//! fal.ai, with your own key.
//!
//! The queue API: POST the request to `queue.fal.run/<endpoint>`, then poll
//! the `status_url` the reply names until COMPLETED, then GET its
//! `response_url`. We follow the URLs fal hands back rather than building
//! them: for an endpoint with a sub-path (`fal-ai/flux-pro/v1/fill`) the
//! status lives under the APP id, and fal knows which part that is.
//! Prices come from `api.fal.ai/v1/models/pricing`, with the same key.
//!
//! A failed request still ends COMPLETED: the status carries `error` and
//! `error_type`, and the result URL answers 4xx/5xx with fal's `detail`
//! (`[{loc, msg, type}]` for a model error, a string for a request error).
//! The result's body is the one that says what to change, so it is read.
//! Checked against fal's queue, pricing and error docs (2026-10).

use anyhow::{anyhow, bail, Result};
use serde_json::Value;

use crate::http::Http;
use crate::service::{self, Cx, Phase, Quote, QuoteItem, Ran, Service};

pub struct Fal {
    pub queue: String,
    pub api: String,
    key: String,
}

impl Fal {
    pub fn new(key: impl Into<String>) -> Self {
        Self::with_bases(key, "https://queue.fal.run", "https://api.fal.ai")
    }

    /// Against other hosts — a test server, a proxy.
    pub fn with_bases(key: impl Into<String>, queue: &str, api: &str) -> Self {
        Self {
            queue: queue.trim_end_matches('/').into(),
            api: api.trim_end_matches('/').into(),
            key: key.into(),
        }
    }

    fn auth(&self) -> String {
        format!("Key {}", self.key)
    }
}

impl Service for Fal {
    fn name(&self) -> &'static str {
        "fal"
    }

    fn run(
        &self,
        endpoint: &str,
        input: &Value,
        _max_usd: Option<f64>,
        _megapixels: Option<f64>,
        cx: &Cx,
    ) -> Result<Ran> {
        let auth = self.auth();
        let h = [("authorization", auth.as_str())];
        let sub = cx.http.post_json(&format!("{}/{endpoint}", self.queue), &h, input)?;
        let field = |k: &str| {
            sub.get(k)
                .and_then(|v| v.as_str())
                .map(String::from)
                .ok_or_else(|| anyhow!("fal's reply has no {k}"))
        };
        let status_url = field("status_url")?;
        let response_url = field("response_url")?;
        let cancel_url = field("cancel_url").ok();

        let waited = service::poll(cx, || {
            let s = cx.http.get_json(&status_url, &h)?;
            match s.get("status").and_then(|v| v.as_str()).unwrap_or("") {
                "IN_QUEUE" => {
                    (cx.phase)(Phase::Queued(s.get("queue_position").and_then(|v| v.as_u64())));
                    Ok(None)
                }
                "IN_PROGRESS" => {
                    (cx.phase)(Phase::Running);
                    Ok(None)
                }
                // A failed request is COMPLETED too, with `error` and
                // `error_type` set. The status's `error` is a summary; the
                // reason (which field, which policy) is the result's body.
                "COMPLETED" => match s.get("error").filter(|e| !e.is_null()) {
                    Some(e) => Err(failure(cx, &response_url, &h, e, s.get("error_type"))),
                    None => Ok(Some(())),
                },
                other => bail!("fal returned an unexpected status \"{other}\""),
            }
        });
        if let Err(e) = waited {
            // Stop paying for something nobody is waiting for.
            if let Some(c) = cancel_url {
                let _ = cx.http.put_empty(&c, &h);
            }
            return Err(e);
        }
        // fal does not report a charge per request; the quote stands.
        Ok(Ran { output: cx.http.get_json(&response_url, &h)?, charged_usd: None })
    }

    fn quote(&self, http: &Http, items: &[QuoteItem]) -> Result<Quote> {
        let mut ids: Vec<&str> = items.iter().map(|i| i.endpoint).collect();
        ids.sort();
        ids.dedup();
        if ids.is_empty() {
            return Ok(Quote::default());
        }
        let auth = self.auth();
        let url = format!("{}/v1/models/pricing?endpoint_id={}", self.api, ids.join(","));
        let v = http.get_json(&url, &[("authorization", auth.as_str())])?;
        let rows = v.get("prices").and_then(|p| p.as_array()).cloned().unwrap_or_default();
        let find = |id: &str| {
            rows.iter().find(|r| r.get("endpoint_id").and_then(|e| e.as_str()) == Some(id))
        };
        let items = items
            .iter()
            .map(|it| {
                let r = find(it.endpoint)?;
                if r.get("currency").and_then(|c| c.as_str()).is_some_and(|c| c != "USD") {
                    return None;
                }
                let up = r.get("unit_price")?.as_f64()?;
                let unit = r.get("unit").and_then(|u| u.as_str()).unwrap_or("call");
                Some(service::price_of(up, unit, it.megapixels, it.video))
            })
            .collect();
        Ok(Quote { items, balance_usd: None })
    }
}

/// Why a COMPLETED-with-error request failed, from the result's body when
/// it says, else from the status.
fn failure(cx: &Cx, response_url: &str, h: &[(&str, &str)], e: &Value, kind: Option<&Value>) -> anyhow::Error {
    let summary = e.as_str().map(String::from).unwrap_or_else(|| e.to_string());
    let kind = kind.and_then(|k| k.as_str()).filter(|k| !k.is_empty());
    let detail = match cx.http.get_text(response_url, h) {
        Ok((status, text)) if !(200..300).contains(&status) => Some((status, crate::http::reason(&text))),
        _ => None,
    };
    match (detail, kind) {
        (Some((status, why)), _) if !why.trim().is_empty() => anyhow!("fal: HTTP {status}: {why}"),
        (_, Some(k)) => anyhow!("fal: {summary} ({k})"),
        _ => anyhow!("fal: {summary}"),
    }
}
