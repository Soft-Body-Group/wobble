//! Where a run is billed and where its results land — MACHINE state, never
//! setup state.
//!
//! A comp says WHAT to run (catalog ids, prompts, seeds). It never says
//! where to send it or whose account pays: open a comp someone sent you and
//! it runs on your key or your credits, against your ComfyUI, or not at all.
//! So none of this is serialised into a setup, and none of it lives in the
//! app's prefs either — the secrets sit in their own file, readable only by
//! you, that a support bundle or a shared prefs file can never sweep up.
//!
//! Environment variables win over the file (`FAL_KEY`, `WOBBLE_RELAY_TOKEN`,
//! `WOBBLE_RELAY_URL`, `WOBBLE_COMFY_URL`, and `WOBBLE_FAL_URL` for a proxy
//! or a stand-in fal), which is what a render farm and the tests use.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// The relay's public address. Overridable for a local `wrangler dev`.
#[cfg(feature = "relay")]
pub const RELAY_URL: &str = "https://relay.wobble.ing";
/// ComfyUI's own default.
pub const COMFY_URL: &str = "http://127.0.0.1:8188";

/// Who pays for a remote call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Route {
    /// Wobble credits, through the relay: one account, no provider keys.
    #[default]
    Relay,
    /// Your own provider key, straight to the provider. Free of us
    /// entirely — the relay is a convenience, never a requirement.
    Own,
}

/// The file's shape. Every field optional so a hand-edited file with one
/// line in it still loads.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct File {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fal_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub relay_token: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub comfy_url: Option<String>,
    #[serde(default)]
    pub route: Route,
}

/// Where a value came from, so the panel can say "from FAL_KEY".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Origin {
    Env(&'static str),
    File,
    Default,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub fal_key: Option<(String, Origin)>,
    pub relay_token: Option<(String, Origin)>,
    pub relay_url: String,
    pub comfy_url: String,
    /// fal's queue AND api base, when something stands in for fal (a proxy,
    /// `--test-genai`'s fake). `None` = fal itself.
    pub fal_url: Option<String>,
    pub route: Route,
}

impl Config {
    /// The file plus the environment.
    pub fn load() -> Config {
        let f = read_file().unwrap_or_default();
        let env = |k: &'static str| std::env::var(k).ok().filter(|v| !v.trim().is_empty());
        let fal_key = env("FAL_KEY")
            .map(|v| (v, Origin::Env("FAL_KEY")))
            .or(f.fal_key.clone().filter(|v| !v.is_empty()).map(|v| (v, Origin::File)));
        let relay_token = env("WOBBLE_RELAY_TOKEN")
            .map(|v| (v, Origin::Env("WOBBLE_RELAY_TOKEN")))
            .or(f.relay_token.clone().filter(|v| !v.is_empty()).map(|v| (v, Origin::File)));
        #[allow(unused_mut)]
        let mut c = Config {
            fal_key,
            relay_token,
            #[cfg(feature = "relay")]
            relay_url: env("WOBBLE_RELAY_URL").unwrap_or_else(|| RELAY_URL.into()),
            #[cfg(not(feature = "relay"))]
            relay_url: String::new(),
            comfy_url: env("WOBBLE_COMFY_URL")
                .or(f.comfy_url.clone())
                .unwrap_or_else(|| COMFY_URL.into()),
            fal_url: env("WOBBLE_FAL_URL"),
            route: env("WOBBLE_GENAI_ROUTE")
                .and_then(|r| serde_json::from_value(serde_json::Value::String(r)).ok())
                .unwrap_or(f.route),
        };
        // Without the relay there are no credits to route through.
        #[cfg(not(feature = "relay"))]
        {
            c.route = Route::Own;
            c.relay_token = None;
        }
        c
    }

    /// Can a remote step be billed on the chosen route? `Err` says what is
    /// missing, in words for the panel.
    pub fn can_bill(&self) -> Result<(), String> {
        match self.route {
            #[cfg(feature = "relay")]
            Route::Relay if self.relay_token.is_none() => Err(
                "no wobble account on this machine: add a relay token, or switch to your own fal key"
                    .into(),
            ),
            #[cfg(not(feature = "relay"))]
            Route::Relay => Err("not available in this build".into()),
            Route::Own if self.fal_key.is_none() => {
                Err("no fal key on this machine: set FAL_KEY or add one in the panel".into())
            }
            _ => Ok(()),
        }
    }
}

/// `$WOBBLE_GENAI_CONFIG`, else `~/.wobble/genai.json`.
pub fn config_path() -> PathBuf {
    if let Some(p) = std::env::var_os("WOBBLE_GENAI_CONFIG") {
        return PathBuf::from(p);
    }
    home().join(".wobble/genai.json")
}

/// Where results land: `$WOBBLE_GENAI_DIR`, else `~/.wobble/genai`. One
/// directory per step result, named by its cache key.
pub fn results_dir() -> PathBuf {
    if let Some(p) = std::env::var_os("WOBBLE_GENAI_DIR") {
        return PathBuf::from(p);
    }
    home().join(".wobble/genai")
}

fn home() -> PathBuf {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
}

pub fn read_file() -> Option<File> {
    let text = std::fs::read_to_string(config_path()).ok()?;
    serde_json::from_str(&text).ok()
}

/// Change the file with `edit` and write it back, readable only by you.
pub fn update(edit: impl FnOnce(&mut File)) -> Result<()> {
    let mut f = read_file().unwrap_or_default();
    edit(&mut f);
    let path = config_path();
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    }
    let tmp = path.with_extension("json.tmp");
    write_private(&tmp, serde_json::to_string_pretty(&f)?.as_bytes())?;
    std::fs::rename(&tmp, &path).with_context(|| format!("replace {}", path.display()))?;
    Ok(())
}

#[cfg(unix)]
fn write_private(path: &std::path::Path, bytes: &[u8]) -> Result<()> {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;
    let mut f = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .mode(0o600)
        .open(path)
        .with_context(|| format!("write {}", path.display()))?;
    f.write_all(bytes)?;
    Ok(())
}

#[cfg(not(unix))]
fn write_private(path: &std::path::Path, bytes: &[u8]) -> Result<()> {
    std::fs::write(path, bytes).with_context(|| format!("write {}", path.display()))
}

/// "fal_…3f9a": enough to recognise a key, not enough to use one.
pub fn masked(secret: &str) -> String {
    let n = secret.chars().count();
    if n <= 8 {
        return "•".repeat(n);
    }
    let tail: String = secret.chars().skip(n - 4).collect();
    format!("••••{tail}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_short_secret_is_fully_masked_and_a_long_one_shows_its_tail() {
        assert_eq!(masked("abc"), "•••");
        assert_eq!(masked("0123456789abcdef"), "••••cdef");
    }

    #[test]
    fn the_file_loads_with_any_subset_of_fields() {
        let f: File = serde_json::from_str(r#"{"route":"own"}"#).unwrap();
        assert_eq!(f.route, Route::Own);
        assert!(f.fal_key.is_none());
        let f: File = serde_json::from_str("{}").unwrap();
        assert_eq!(f.route, Route::Relay);
    }
}
