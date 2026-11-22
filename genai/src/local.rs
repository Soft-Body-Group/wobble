//! Models that run on this machine (`Service::Local`): free, no key, and
//! nothing leaves the machine. Today that is LTX-2.5's distilled
//! image/text-to-video, driven as a process by `tools/wobble_ltx.py`.
//!
//! The checkout is the LTX HDR node's (`~/.wobble/LTX-2`, or
//! `$WOBBLE_LTXHDR_DIR`), at the same pinned commit ([`LTX_COMMIT`]), so the
//! two share one Python environment. The weights are not the HDR node's:
//! this runs the NVFP4 distilled transformer (an RTX 50-series runs it
//! natively, streamed from system RAM, so 16 GB of VRAM is enough), fetched
//! by the driver's `--fetch` under the user's OWN Hugging Face login — the
//! repository is gated behind the LTX-2.x Community Licence, which the user
//! accepts on the model page, not here.
//!
//! **Nothing installs or downloads without a click**, and the run itself is
//! a press of RUN like any other step: the plan says what is missing
//! ([`ready`]) before anything starts.

use anyhow::{anyhow, bail, Context, Result};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crate::service::{Cx, Phase};

pub const DRIVER: &str = include_str!("../../tools/wobble_ltx.py");
const DRIVER_NAME: &str = "wobble_ltx.py";
pub const LTX_REPO: &str = "https://github.com/Lightricks/LTX-2.git";
/// The LTX-2 commit the drivers are written against (the HDR node's too).
/// Bump it and re-test both, never float it.
pub const LTX_COMMIT: &str = "e98f6d927d83c92ef2be6aec76c8440b19306e39";
pub const MODEL_PAGE: &str = "https://huggingface.co/Lightricks/LTX-2.5";

/// The five files `--fetch` writes under the models directory, with their
/// sizes: the same list as the driver's `FILES`.
pub const WEIGHTS: [(&str, u64); 5] = [
    ("diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors", 18_721_548_408),
    ("text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors", 26_263_858_182),
    ("vae/ltx-2.5-video-vae-conv-bf16.safetensors", 1_452_269_922),
    ("vae/ltx-2.5-audio-vae-bf16.safetensors", 364_866_540),
    ("latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors", 995_778_752),
];

pub fn weights_bytes() -> u64 {
    WEIGHTS.iter().map(|(_, n)| n).sum()
}

fn home() -> Option<PathBuf> {
    std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")).map(PathBuf::from).filter(|p| !p.as_os_str().is_empty())
}

/// Where the LTX-2 checkout lives (shared with the LTX HDR node).
pub fn install_dir() -> PathBuf {
    if let Some(d) = std::env::var_os("WOBBLE_LTXHDR_DIR") {
        return PathBuf::from(d);
    }
    home().unwrap_or_else(|| PathBuf::from(".")).join(".wobble").join("LTX-2")
}

/// Where the weights land: beside the checkout.
pub fn models_dir() -> PathBuf {
    install_dir().join("models")
}

/// The checkout's own interpreter. Not `uv run`: that re-syncs the
/// environment to its defaults and would drop the NVFP4 kernels, which are
/// an optional group.
pub fn python(dir: &Path) -> PathBuf {
    if cfg!(windows) {
        dir.join(".venv").join("Scripts").join("python.exe")
    } else {
        dir.join(".venv").join("bin").join("python")
    }
}

pub fn installed(dir: &Path) -> bool {
    dir.join("packages").join("ltx-pipelines").join("src").join("ltx_pipelines").join("distilled.py").is_file()
        && python(dir).is_file()
}

/// All five weight files, at their full size.
pub fn weights_ready(models: &Path) -> bool {
    WEIGHTS.iter().all(|(f, n)| std::fs::metadata(models.join(f)).is_ok_and(|m| m.len() == *n))
}

/// A suite's stand-in: a command that replaces the tool and its weights.
/// It is run with the same arguments and must write `--out`.
fn stand_in() -> Option<String> {
    std::env::var("WOBBLE_LTX_CMD").ok().filter(|s| !s.trim().is_empty())
}

/// Can a local step run now? Said in the plan, before anything starts.
pub fn ready() -> Result<(), String> {
    if stand_in().is_some() {
        return Ok(());
    }
    let dir = install_dir();
    if !installed(&dir) {
        return Err(format!(
            "LTX-2 is not installed in {}: install it (the LTX HDR node's panel, or the GenAI panel's local row)",
            dir.display()
        ));
    }
    if !weights_ready(&models_dir()) {
        return Err(format!(
            "the LTX-2.5 weights are not all in {} ({:.0} GB): accept the licence at {MODEL_PAGE}, log in to Hugging Face, then fetch them",
            models_dir().display(),
            weights_bytes() as f64 / 1e9
        ));
    }
    Ok(())
}

/// Run one local generation: `req` is the step's request
/// (`run::request_clip` with paths for images), the clip lands at `out`.
/// Blocking; cancelled through `cx`.
pub fn run(req: &Value, work: &Path, out: &Path, cx: &Cx) -> Result<()> {
    use std::io::{BufRead, BufReader};
    use std::process::{Command, Stdio};

    std::fs::create_dir_all(work).with_context(|| format!("create {}", work.display()))?;
    let job = work.join("job.json");
    std::fs::write(&job, serde_json::to_vec_pretty(req)?)?;
    let dir = install_dir();
    let mut cmd = match stand_in() {
        Some(s) => {
            let mut w = s.split_whitespace();
            let mut c = Command::new(w.next().unwrap());
            c.args(w);
            c
        }
        None => {
            ready().map_err(|e| anyhow!(e))?;
            // Written on every run, so the app and the driver never drift.
            let driver = dir.join(DRIVER_NAME);
            std::fs::write(&driver, DRIVER).with_context(|| format!("write {}", driver.display()))?;
            let mut c = Command::new(python(&dir));
            c.arg(&driver).current_dir(&dir);
            c
        }
    };
    cmd.arg("--job").arg(&job).arg("--models").arg(models_dir()).arg("--out").arg(out);
    cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = cmd.spawn().context("start the LTX driver")?;
    (cx.phase)(Phase::Running);
    let tail = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let said = std::sync::Arc::new(std::sync::Mutex::new((false, None::<String>)));
    let keep = |tail: &std::sync::Mutex<Vec<String>>, l: String| {
        log::info!("ltx: {l}");
        let mut t = tail.lock().unwrap();
        t.push(l);
        let n = t.len();
        if n > 40 {
            t.drain(..n - 40);
        }
    };
    let err = child.stderr.take().unwrap();
    let t2 = tail.clone();
    let e = std::thread::spawn(move || {
        for l in BufReader::new(err).lines().map_while(Result::ok) {
            keep(&t2, l);
        }
    });
    let outp = child.stdout.take().unwrap();
    let (t3, s3) = (tail.clone(), said.clone());
    let o = std::thread::spawn(move || {
        for l in BufReader::new(outp).lines().map_while(Result::ok) {
            if let Some(rest) = l.strip_prefix("WOBBLE ") {
                if rest == "done" {
                    s3.lock().unwrap().0 = true;
                } else if let Some(m) = rest.strip_prefix("error ") {
                    s3.lock().unwrap().1 = Some(m.to_string());
                }
            }
            keep(&t3, l);
        }
    });
    let start = Instant::now();
    let status = loop {
        if cx.cancelled() || start.elapsed() > cx.deadline {
            let _ = child.kill();
            let _ = child.wait();
            bail!(if cx.cancelled() { "cancelled" } else { "the local LTX run took too long and was stopped" });
        }
        match child.try_wait()? {
            Some(s) => break s,
            None => std::thread::sleep(Duration::from_millis(200)),
        }
    };
    let _ = (o.join(), e.join());
    let (done, error) = said.lock().unwrap().clone();
    if let Some(m) = error {
        bail!("LTX: {m}");
    }
    if !status.success() || !done || !out.is_file() {
        let t = tail.lock().unwrap();
        let last = t.iter().rev().find(|l| !l.trim().is_empty()).cloned().unwrap_or_default();
        bail!("the LTX driver stopped ({status}) without a clip: {last}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_driver_and_the_runner_agree() {
        for (f, _) in WEIGHTS {
            assert!(DRIVER.contains(f), "{f}");
        }
        for w in ["--job", "--models", "--out", "--fetch", "\"done\"", "\"error\"", "nvfp4-prequant", "--offload"] {
            assert!(DRIVER.contains(w), "{w}");
        }
        for k in ["\"prompt\"", "\"image\"", "\"seconds\"", "\"width\"", "\"height\"", "\"seed\""] {
            assert!(DRIVER.contains(k), "{k}");
        }
        assert_eq!(LTX_COMMIT.len(), 40);
    }

    #[test]
    fn only_the_fetch_mode_touches_the_network() {
        let run = DRIVER.split("def run(").nth(1).unwrap();
        assert!(!run.contains("hf_hub_download") && !run.contains("huggingface_hub"));
    }
}
