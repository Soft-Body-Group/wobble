//! The wobble relay: fal's endpoints, paid for with wobble credits.
//!
//! The relay holds the provider keys; the app holds a wobble token. The
//! protocol is deliberately a thin mirror of fal's queue (and is written
//! down in full in `.agents/genai.md` and the relay's own README), so the
//! output JSON that comes back is fal's, untouched, and nothing past
//! `Service` knows which route paid.
//!
//! ```text
//! POST /v1/quote            {items:[{endpoint, input, megapixels?, seconds?}]}
//!                        -> {items:[{usd, exact, note}|null], balance_usd}
//! POST /v1/jobs             {endpoint, input, max_usd, megapixels?}
//!                        -> 202 {job, usd}          402 {error} when short
//! GET  /v1/jobs/<job>    -> {status: queued|running|done|failed|cancelled,
//!                            queue_position?, output?, error?, charged_usd?}
//! POST /v1/jobs/<job>/cancel
//! GET  /v1/me            -> {account, balance_usd}
//! ```
//!
//! `max_usd` is the quote the user saw. A relay whose price moved since
//! refuses rather than charging more than was shown.

use anyhow::{anyhow, bail, Result};
use serde_json::{json, Value};

use crate::http::Http;
use crate::service::{self, Cx, Phase, Price, Quote, QuoteItem, Ran, Service};

pub struct Relay {
    pub base: String,
    token: String,
}

/// What the account holds.
#[derive(Debug, Clone, PartialEq)]
pub struct Account {
    pub account: String,
    pub balance_usd: f64,
}

impl Relay {
    pub fn new(base: &str, token: impl Into<String>) -> Self {
        Self { base: base.trim_end_matches('/').into(), token: token.into() }
    }

    fn auth(&self) -> String {
        format!("Bearer {}", self.token)
    }

    pub fn me(&self, http: &Http) -> Result<Account> {
        let auth = self.auth();
        let v = http.get_json(&format!("{}/v1/me", self.base), &[("authorization", auth.as_str())])?;
        Ok(Account {
            account: v.get("account").and_then(|a| a.as_str()).unwrap_or("?").into(),
            balance_usd: v.get("balance_usd").and_then(|b| b.as_f64()).unwrap_or(0.0),
        })
    }
}

impl Service for Relay {
    fn name(&self) -> &'static str {
        "wobble relay"
    }

    fn run(
        &self,
        endpoint: &str,
        input: &Value,
        max_usd: Option<f64>,
        megapixels: Option<f64>,
        cx: &Cx,
    ) -> Result<Ran> {
        let auth = self.auth();
        let h = [("authorization", auth.as_str())];
        let mut body = json!({ "endpoint": endpoint, "input": input });
        if let Some(m) = max_usd {
            body["max_usd"] = json!(m);
        }
        if let Some(mp) = megapixels {
            body["megapixels"] = json!(mp);
        }
        let sub = cx.http.post_json(&format!("{}/v1/jobs", self.base), &h, &body)?;
        let job = sub
            .get("job")
            .and_then(|j| j.as_str())
            .ok_or_else(|| anyhow!("the relay's reply has no job id"))?
            .to_string();
        // A job id is ours to put in a path; refuse anything that is not
        // plainly an id rather than let a reply steer the next request.
        if job.is_empty() || !job.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
            bail!("the relay returned a malformed job id");
        }
        let status_url = format!("{}/v1/jobs/{job}", self.base);
        let res = service::poll(cx, || {
            let s = cx.http.get_json(&status_url, &h)?;
            match s.get("status").and_then(|v| v.as_str()).unwrap_or("") {
                "queued" => {
                    (cx.phase)(Phase::Queued(s.get("queue_position").and_then(|v| v.as_u64())));
                    Ok(None)
                }
                "running" => {
                    (cx.phase)(Phase::Running);
                    Ok(None)
                }
                "done" => Ok(Some(Ran {
                    output: s
                        .get("output")
                        .cloned()
                        .ok_or_else(|| anyhow!("the relay finished with no output"))?,
                    charged_usd: s.get("charged_usd").and_then(|c| c.as_f64()),
                })),
                "failed" => bail!(
                    "{}",
                    s.get("error").and_then(|e| e.as_str()).unwrap_or("the job failed")
                ),
                "cancelled" => bail!("cancelled"),
                other => bail!("the relay returned an unexpected status \"{other}\""),
            }
        });
        if res.is_err() {
            let _ = cx.http.post_json(&format!("{status_url}/cancel"), &h, &json!({}));
        }
        res
    }

    fn quote(&self, http: &Http, items: &[QuoteItem]) -> Result<Quote> {
        let auth = self.auth();
        let body = json!({
            "items": items.iter().map(|i| json!({
                "endpoint": i.endpoint,
                "input": i.input,
                "megapixels": i.megapixels,
                "seconds": i.video.map(|v| v.seconds),
            })).collect::<Vec<_>>()
        });
        let v = http.post_json(
            &format!("{}/v1/quote", self.base),
            &[("authorization", auth.as_str())],
            &body,
        )?;
        let rows = v.get("items").and_then(|r| r.as_array()).cloned().unwrap_or_default();
        let items = (0..items.len())
            .map(|k| {
                let r = rows.get(k)?;
                Some(Price {
                    usd: r.get("usd")?.as_f64()?,
                    exact: r.get("exact").and_then(|e| e.as_bool()).unwrap_or(false),
                    note: r.get("note").and_then(|n| n.as_str()).unwrap_or("").into(),
                })
            })
            .collect();
        Ok(Quote { items, balance_usd: v.get("balance_usd").and_then(|b| b.as_f64()) })
    }
}
