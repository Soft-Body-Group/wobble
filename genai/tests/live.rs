//! Checks against the REAL fal, for whoever has a key. All `#[ignore]`:
//! nothing here runs in `cargo test`, and nothing here is billed — the
//! pricing API is free.
//!
//! `cargo test -p genai --test live -- --ignored --nocapture`

use genai::catalog::CATALOG;
use genai::config::Config;
use genai::http::Http;

/// Every catalog endpoint is one fal's pricing API knows. A typo in an
/// endpoint, or a model fal retired, shows here instead of as a failed
/// (and, for the steps above it, already billed) run.
#[test]
#[ignore]
fn every_catalog_endpoint_is_known_to_fal() {
    let Some((key, _)) = Config::load().fal_key else {
        panic!("no fal key on this machine: set FAL_KEY");
    };
    let mut ids: Vec<&str> = CATALOG.iter().map(|m| m.endpoint).collect();
    ids.sort();
    ids.dedup();
    let http = Http::new();
    let auth = format!("Key {key}");
    let mut unknown = vec![];
    // One at a time: a single unknown id fails a whole batched request.
    for id in ids {
        let url = format!("https://api.fal.ai/v1/models/pricing?endpoint_id={id}");
        // The pricing API allows a handful of calls a second.
        let mut got = http.get_json(&url, &[("authorization", auth.as_str())]);
        for wait in [2, 5, 10] {
            if !got.as_ref().is_err_and(|e| format!("{e:#}").contains("429")) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_secs(wait));
            got = http.get_json(&url, &[("authorization", auth.as_str())]);
        }
        std::thread::sleep(std::time::Duration::from_millis(400));
        match got {
            Ok(v) => {
                let row = v.get("prices").and_then(|p| p.as_array()).and_then(|a| a.first()).cloned();
                match row {
                    Some(r) => println!("{id}: {} per {}", r["unit_price"], r["unit"]),
                    None => unknown.push(format!("{id}: no price row")),
                }
            }
            Err(e) => unknown.push(format!("{id}: {e:#}")),
        }
    }
    assert!(unknown.is_empty(), "fal does not know:\n{}", unknown.join("\n"));
}
