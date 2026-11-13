//! The wobble account, from the app's side: a stored token, the sign-in that
//! gets one, and authenticated requests to the site.
//!
//! Sign-in is a device code (`gh auth login` style): the app asks the site
//! for a code, opens `/device?code=…` in the browser, and polls while the
//! person signs in there (email link, GitHub, Google — whatever the site
//! offers) and approves. The app never sees a password or an OAuth secret,
//! and its token is listed, and revocable, on the account page.
//!
//! The token lives in `~/.wobble/account.json`, readable by the user only.
//! (The OS keychain is the better home; this is the first step.) The same
//! account and token will carry the cloud storage service.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, TryRecvError};
use std::sync::Arc;

/// The site's origin, from the registry URL (so `WOBBLE_PLUGIN_REGISTRY`
/// pointing at a local `wrangler dev` points the account there too).
pub fn site_base() -> String {
    crate::pluginstore::site_base()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Saved {
    pub token: String,
    pub email: Option<String>,
    /// The site it belongs to: a token for a local dev server is not one
    /// for the real site.
    pub site: String,
}

fn path() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(|h| PathBuf::from(h).join(".wobble/account.json"))
}

/// The saved sign-in for THIS site, if any. Suites never see one: a test run
/// must not act as the user.
pub fn load() -> Option<Saved> {
    if crate::TEST_RUN.load(Ordering::Relaxed) {
        return None;
    }
    let s: Saved = serde_json::from_str(&std::fs::read_to_string(path()?).ok()?).ok()?;
    (s.site == site_base()).then_some(s)
}

fn save(s: &Saved) -> anyhow::Result<()> {
    let p = path().ok_or_else(|| anyhow::anyhow!("no home directory"))?;
    if let Some(d) = p.parent() {
        std::fs::create_dir_all(d)?;
    }
    let text = serde_json::to_string_pretty(s)?;
    #[cfg(unix)]
    {
        use std::io::Write;
        use std::os::unix::fs::OpenOptionsExt;
        let mut f = std::fs::OpenOptions::new().write(true).create(true).truncate(true).mode(0o600).open(&p)?;
        f.write_all(text.as_bytes())?;
    }
    #[cfg(not(unix))]
    std::fs::write(&p, text)?;
    Ok(())
}

/// Forget the sign-in here, and tell the site to revoke the token (best
/// effort: offline, the account page can still sign it out).
pub fn sign_out(s: &Saved) {
    if let Some(p) = path() {
        let _ = std::fs::remove_file(p);
    }
    let (url, token) = (format!("{}/api/auth/signout", s.site), s.token.clone());
    crate::bg::spawn(move || {
        let _ = post_json(&url, &serde_json::json!({}), Some(&token));
    });
}

// --- HTTP ------------------------------------------------------------------

#[cfg(target_arch = "wasm32")]
pub fn get(url: &str, _token: Option<&str>, _cap: u64) -> anyhow::Result<(u16, Vec<u8>)> {
    anyhow::bail!("GET {url}: no account in the web build")
}

#[cfg(target_arch = "wasm32")]
pub fn post_json(url: &str, _body: &serde_json::Value, _token: Option<&str>) -> anyhow::Result<(u16, serde_json::Value)> {
    anyhow::bail!("POST {url}: no account in the web build")
}

/// An agent that hands back 4xx bodies instead of erroring: the site's
/// errors are JSON worth reading (`authorization_pending`, `not_owned`).
#[cfg(not(target_arch = "wasm32"))]
fn agent() -> ureq::Agent {
    ureq::Agent::new_with_config(ureq::Agent::config_builder().http_status_as_error(false).build())
}

#[cfg(not(target_arch = "wasm32"))]
pub fn get(url: &str, token: Option<&str>, cap: u64) -> anyhow::Result<(u16, Vec<u8>)> {
    use std::io::Read;
    let mut req = agent().get(url);
    if let Some(t) = token {
        req = req.header("authorization", &format!("Bearer {t}"));
    }
    let resp = req.call().map_err(|e| anyhow::anyhow!("GET {url}: {e}"))?;
    let status = resp.status().as_u16();
    let mut buf = Vec::new();
    resp.into_body().into_reader().take(cap + 1).read_to_end(&mut buf)?;
    anyhow::ensure!((buf.len() as u64) <= cap, "{url} is larger than {cap} bytes");
    Ok((status, buf))
}

#[cfg(not(target_arch = "wasm32"))]
pub fn post_json(url: &str, body: &serde_json::Value, token: Option<&str>) -> anyhow::Result<(u16, serde_json::Value)> {
    let mut req = agent().post(url).header("content-type", "application/json");
    if let Some(t) = token {
        req = req.header("authorization", &format!("Bearer {t}"));
    }
    let mut resp = req.send(body.to_string()).map_err(|e| anyhow::anyhow!("POST {url}: {e}"))?;
    let status = resp.status().as_u16();
    let text = resp.body_mut().with_config().limit(64 * 1024).read_to_string()?;
    Ok((status, serde_json::from_str(&text).unwrap_or(serde_json::Value::Null)))
}

/// The error code in a site error body, `{"error": "..."}`.
pub fn error_code(v: &serde_json::Value) -> &str {
    v.get("error").and_then(|e| e.as_str()).unwrap_or("")
}

// --- sign-in -----------------------------------------------------------------

pub enum SignInEvent {
    /// Show this code; the browser is open at `url`.
    Code { code: String, url: String },
    Done(anyhow::Result<Saved>),
}

/// A sign-in in progress. Dropping it (or `cancel`) stops the polling.
pub struct SignIn {
    pub code: Option<String>,
    pub url: Option<String>,
    rx: Receiver<SignInEvent>,
    stop: Arc<AtomicBool>,
}

impl SignIn {
    pub fn start() -> SignIn {
        let (tx, rx) = channel();
        let stop = Arc::new(AtomicBool::new(false));
        let quit = stop.clone();
        let base = site_base();
        // A thread, not `bg::spawn`: this job WAITS (polls for minutes), and
        // the web build's one-thread pool must only get jobs that end. The
        // web build has no account (`post_json` fails at once), so this ends
        // there immediately anyway.
        #[cfg(not(target_arch = "wasm32"))]
        std::thread::spawn(move || {
            let r = run(&base, &tx, &quit);
            let _ = tx.send(SignInEvent::Done(r));
        });
        #[cfg(target_arch = "wasm32")]
        let _ = tx.send(SignInEvent::Done(Err(anyhow::anyhow!("no account in the web build"))));
        SignIn { code: None, url: None, rx, stop }
    }

    pub fn cancel(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }

    /// The result, once there is one.
    pub fn poll(&mut self) -> Option<anyhow::Result<Saved>> {
        loop {
            match self.rx.try_recv() {
                Ok(SignInEvent::Code { code, url }) => {
                    self.code = Some(code);
                    self.url = Some(url);
                }
                Ok(SignInEvent::Done(r)) => return Some(r),
                Err(TryRecvError::Empty) => return None,
                Err(TryRecvError::Disconnected) => return Some(Err(anyhow::anyhow!("sign-in stopped"))),
            }
        }
    }
}

impl Drop for SignIn {
    fn drop(&mut self) {
        self.cancel();
    }
}

fn label() -> String {
    let who = std::env::var("USER").or_else(|_| std::env::var("USERNAME")).unwrap_or_default();
    let os = match std::env::consts::OS {
        "macos" => "Mac",
        "windows" => "Windows",
        "linux" => "Linux",
        o => o,
    };
    if who.is_empty() {
        format!("wobble on {os}")
    } else {
        format!("wobble on {who}'s {os}")
    }
}

fn run(base: &str, tx: &std::sync::mpsc::Sender<SignInEvent>, stop: &AtomicBool) -> anyhow::Result<Saved> {
    let (status, start) = post_json(&format!("{base}/api/device/start"), &serde_json::json!({ "label": label() }), None)?;
    anyhow::ensure!(status == 200, "the site refused to start a sign-in ({})", error_code(&start));
    let s = |k: &str| start.get(k).and_then(|v| v.as_str()).unwrap_or_default().to_string();
    let (device, code, url) = (s("device_code"), s("user_code"), s("verification_uri_complete"));
    anyhow::ensure!(!device.is_empty() && url.starts_with(base), "the site sent a sign-in it should not have");
    let every = start.get("interval").and_then(|v| v.as_u64()).unwrap_or(3).clamp(1, 30);
    let expires = start.get("expires_in").and_then(|v| v.as_u64()).unwrap_or(600).min(1800);
    // WOBBLE_NO_BROWSER: a scripted run approves the code itself.
    if std::env::var_os("WOBBLE_NO_BROWSER").is_none() {
        crate::open_url(&url);
    }
    let _ = tx.send(SignInEvent::Code { code, url });

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(expires);
    while std::time::Instant::now() < deadline {
        // Sleep in small steps so a cancel is felt at once.
        for _ in 0..every * 4 {
            if stop.load(Ordering::Relaxed) {
                anyhow::bail!("sign-in cancelled");
            }
            std::thread::sleep(std::time::Duration::from_millis(250));
        }
        let (status, r) = post_json(&format!("{base}/api/device/poll"), &serde_json::json!({ "device_code": device }), None)?;
        if status == 200 {
            let token = r.get("access_token").and_then(|v| v.as_str()).unwrap_or_default().to_string();
            anyhow::ensure!(token.starts_with("wob_"), "the site sent no token");
            let saved = Saved {
                token,
                email: r.get("email").and_then(|v| v.as_str()).map(String::from),
                site: base.to_string(),
            };
            save(&saved)?;
            return Ok(saved);
        }
        match error_code(&r) {
            "authorization_pending" | "slow_down" => continue,
            "access_denied" => anyhow::bail!("sign-in was cancelled in the browser"),
            "expired_token" => break,
            other => anyhow::bail!("sign-in failed ({other})"),
        }
    }
    anyhow::bail!("the sign-in code expired — try again")
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_site_is_the_registry_origin() {
        std::env::set_var("WOBBLE_PLUGIN_REGISTRY", "http://localhost:8787/wobbins/index.json");
        assert_eq!(super::site_base(), "http://localhost:8787");
        std::env::remove_var("WOBBLE_PLUGIN_REGISTRY");
        assert_eq!(super::site_base(), "https://wobble.ing");
    }
}
