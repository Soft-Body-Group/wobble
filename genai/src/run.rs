//! Turning a pipeline into a file on disk.
//!
//! Three stages, and only the last one touches the network or a wallet:
//!
//! 1. **`plan`** — pure. Orders the steps upstream of Output, gives each a
//!    CACHE KEY (a hash of what it is and of its inputs' keys), and says
//!    which remote steps are already on disk. An unwired port, an unknown
//!    model or a missing credential is reported here, before anything is
//!    sent.
//! 2. **`quote`** — asks the service what the uncached remote steps cost.
//! 3. **`start`** — runs the plan on a thread, step by step, reporting
//!    progress over a channel, and returns the new [`Take`].
//!
//! The cache is the whole reason re-opening a comp never re-bills: a step
//! whose key already has a result directory is read, not run. Keys cover
//! everything that changes the pixels — model id AND endpoint, seed, size,
//! options, the reference's bytes — and nothing that does not (a step's
//! position in the inner graph, the licence flag).

use anyhow::{anyhow, bail, Context, Result};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::Arc;
use std::time::Duration;

use crate::catalog::{self, Dialect, RemoteModel, Task};
use crate::comfy::{self, Comfy, Subs};
use crate::config::{Config, Route};
use crate::fal::Fal;
use crate::http::{self, Http};
use crate::image;
use crate::pipeline::{Op, Pipeline, Take};
#[cfg(feature = "relay")]
use crate::relay::Relay;
use crate::service::{self, Cx, Phase, Quote, QuoteItem, Service, VideoCall};

/// Bump when the key recipe changes, so old results are not mistaken for
/// what a new recipe would have made.
const KEY_VERSION: &str = "wobble-genai/1";
/// The metadata file that marks a result directory complete.
pub const META: &str = "meta.json";

/// An encoded picture from the comp.
#[derive(Debug, Clone)]
pub struct Png {
    pub bytes: Vec<u8>,
    pub w: u32,
    pub h: u32,
}

/// An encoded clip from the comp: an mp4 of the host's input over a frame
/// range, display-referred sRGB like a `Png`.
#[derive(Debug, Clone)]
pub struct Clip {
    pub bytes: Vec<u8>,
    pub w: u32,
    pub h: u32,
    pub frames: u32,
    pub fps: f32,
    /// What the clip IS, for the cache key: a hash of the frames it was
    /// encoded from (`frames_key`). The mp4's own bytes are not it — an
    /// encoder does not write the same file twice — and keying on them
    /// made every re-capture a new request: billed again. Empty = key on
    /// the bytes (a clip from somewhere else).
    pub key: String,
}

/// A clip's identity: its frames' pixels, its size and its rate.
pub fn frames_key(frames: &[Vec<u8>], w: u32, h: u32, fps: f32) -> String {
    let mut d = Sha256::new();
    d.update(w.to_le_bytes());
    d.update(h.to_le_bytes());
    d.update(fps.to_le_bytes());
    for f in frames {
        d.update(f);
    }
    d.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

impl Clip {
    /// The identity a plan keys this clip under.
    fn ident(&self) -> String {
        if self.key.is_empty() {
            sha_hex(&self.bytes)
        } else {
            self.key.clone()
        }
    }

    pub fn seconds(&self) -> f64 {
        self.frames as f64 / self.fps.max(1.0) as f64
    }
    fn shape(&self) -> ClipShape {
        ClipShape { frames: self.frames, fps: self.fps }
    }
}

/// What a clip model's request needs to know about the clip it is sent.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClipShape {
    pub frames: u32,
    pub fps: f32,
}

/// What the host hands a run: its inputs, rendered, and the comp size.
#[derive(Debug, Clone, Default)]
pub struct Inputs {
    /// Per node input, rendered at the playhead: what an `Input` step
    /// that is not a clip reads.
    pub pics: std::collections::BTreeMap<usize, Png>,
    /// Per node input, rendered over the run's range from the playhead:
    /// what a clip `Input` reads.
    pub clips: std::collections::BTreeMap<usize, Clip>,
    /// Per STEP index, a `Shape` step's matte, rendered by the host at
    /// the playhead, and over the run's range.
    pub shapes: std::collections::BTreeMap<usize, Png>,
    pub shape_clips: std::collections::BTreeMap<usize, Clip>,
    pub comp: (u32, u32),
    /// The playhead the inputs were rendered at.
    pub tick: i64,
}

/// The run, decided before anything is sent.
#[derive(Debug, Clone)]
pub struct Plan {
    pub target: usize,
    /// Steps to evaluate, dependencies first.
    pub order: Vec<usize>,
    /// Per step index: its cache key (empty for steps not in `order`).
    pub keys: Vec<String>,
    /// Remote or local-model steps that will actually execute (not cached),
    /// in order.
    pub todo: Vec<usize>,
    /// Are all models upstream of Output free for commercial use?
    pub commercial: bool,
    /// Per step: estimated output megapixels (for quotes).
    pub mp: Vec<f64>,
}

impl Plan {
    /// Remote steps that will be BILLED (Comfy runs locally for free).
    pub fn billed(&self, p: &Pipeline) -> Vec<usize> {
        self.todo.iter().copied().filter(|&i| p.steps[i].op.is_remote()).collect()
    }
}

fn h(parts: &[&str]) -> String {
    let mut d = Sha256::new();
    d.update(KEY_VERSION.as_bytes());
    for p in parts {
        // Length-prefixed, so ("ab","c") and ("a","bc") differ.
        d.update((p.len() as u64).to_le_bytes());
        d.update(p.as_bytes());
    }
    let out = d.finalize();
    out.iter().take(16).map(|b| format!("{b:02x}")).collect()
}

fn sha_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

/// The key tag of an Input step: the names the old fixed inputs keyed
/// under ("reference", "second", "matte", "clip", "matte-clip"), so a
/// result cached before the input bar is still found, and a count after.
fn input_tag(matte: bool, nth: usize, clip: bool) -> String {
    match (matte, clip, nth) {
        (false, false, 0) => "reference".into(),
        (false, false, 1) => "second".into(),
        (false, false, n) => format!("image{n}"),
        (true, false, 0) => "matte".into(),
        (true, false, n) => format!("matte{n}"),
        (false, true, 0) => "clip".into(),
        (false, true, n) => format!("clip{n}"),
        (true, true, 0) => "matte-clip".into(),
        (true, true, n) => format!("matte-clip{n}"),
    }
}

/// The clip a step's video port reads straight off a clip Input (the
/// picture before a matte), for a model that needs to know its shape.
fn step_clip<'a>(p: &Pipeline, i: usize, inputs: &'a Inputs) -> Option<&'a Clip> {
    // Straight off a clip Input, or through comp nodes on the way (a Crop,
    // a Transform, a Spline: rendered by the host as a clip of their own).
    let src = |want_matte: bool| {
        p.steps[i].inputs.iter().flatten().find_map(|&j| match p.steps[j].op {
            Op::Input { input, clip: true, matte, .. } if matte == want_matte => inputs.clips.get(&input),
            Op::Comp { clip: true, .. } if !want_matte => inputs.shape_clips.get(&j),
            Op::Shape { clip: true, .. } if want_matte => inputs.shape_clips.get(&j),
            _ => None,
        })
    };
    src(false).or_else(|| src(true))
}

/// What an old-shape step says when it reaches a run (`normalize` should
/// have turned it into an Input).
const OLD_STEP: &str = "this pipeline has a step from an older build: open and close it to update it";

/// Is there a finished result for this key?
pub fn cached(dir: &Path, key: &str) -> Option<Meta> {
    let text = std::fs::read_to_string(dir.join(key).join(META)).ok()?;
    let m: Meta = serde_json::from_str(&text).ok()?;
    dir.join(key).join(&m.file).exists().then_some(m)
}

/// What a result directory records about how it was made.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq)]
pub struct Meta {
    pub file: String,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub endpoint: String,
    #[serde(default)]
    pub service: String,
    #[serde(default)]
    pub usd: Option<f64>,
    #[serde(default)]
    pub when: u64,
    #[serde(default)]
    pub nsfw_flagged: bool,
    /// The prompt exactly as the model was sent it (after templates and
    /// any enhance step), so a picture never loses the words that made it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u32>,
    /// The request body as sent, every picture or clip in it replaced by a
    /// note of its size ([`scrub_request`]): the model's settings, whole.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request: Option<serde_json::Value>,
}

/// A request body with its inlined media (data URIs) replaced by a short
/// note, so it can be kept as a record of the settings without keeping a
/// second copy of every input.
pub fn scrub_request(v: &serde_json::Value) -> serde_json::Value {
    use serde_json::Value;
    match v {
        Value::String(s) if s.starts_with("data:") => {
            let mime = s[5..].split([';', ',']).next().unwrap_or("");
            Value::String(format!("<{mime}, {} bytes inline>", s.len()))
        }
        Value::Array(a) => Value::Array(a.iter().map(scrub_request).collect()),
        Value::Object(o) => Value::Object(o.iter().map(|(k, v)| (k.clone(), scrub_request(v))).collect()),
        v => v.clone(),
    }
}

/// The text chunks a generated PNG carries: `Description` (the prompt, a
/// key image tools show) and `wobble-genai` (the whole record as JSON).
fn provenance_chunks(meta: &Meta) -> Vec<(&'static str, String)> {
    let mut v = vec![];
    if let Some(p) = meta.prompt.as_deref().filter(|p| !p.is_empty()) {
        v.push(("Description", p.to_string()));
    }
    let rec = serde_json::json!({
        "model": meta.model,
        "endpoint": meta.endpoint,
        "service": meta.service,
        "prompt": meta.prompt,
        "seed": meta.seed,
        "request": meta.request,
        "when": meta.when,
    });
    v.push(("wobble-genai", rec.to_string()));
    v.push(("Software", "wobble".to_string()));
    v
}

/// Order the steps, key them, and find what is cached. Nothing leaves the
/// machine.
pub fn plan(p: &Pipeline, inputs: &Inputs, dir: &Path) -> Result<Plan, String> {
    let target = p.output().ok_or("the pipeline has no Output step")?;
    let order = p.plan(target)?;
    let mut keys = vec![String::new(); p.steps.len()];
    let mut mp = vec![0.0f64; p.steps.len()];
    let comp_mp = inputs.comp.0 as f64 * inputs.comp.1 as f64 / 1e6;
    for &i in &order {
        let step = &p.steps[i];
        let ins: Vec<&str> = step
            .inputs
            .iter()
            .map(|s| s.map(|j| keys[j].as_str()).unwrap_or("-"))
            .collect();
        let in_mp = |port: usize| step.inputs.get(port).copied().flatten().map(|j| mp[j]);
        let (key, m) = match &step.op {
            Op::Prompt { text } => (h(&["prompt", text]), 0.0),
            Op::Input { input, matte, nth, clip, .. } => {
                let what = if *matte { "matte" } else { "image" };
                let tag = input_tag(*matte, *nth, *clip);
                if *clip {
                    if !p.input_is_read(i) {
                        keys[i] = h(&["input-through", &tag]);
                        continue;
                    }
                    let c = inputs.clips.get(input).ok_or(format!(
                        "the pipeline reads the node's {what} input {} as a clip, and none was captured",
                        input + 1
                    ))?;
                    (h(&[&tag, &c.ident()]), c.w as f64 * c.h as f64 * c.frames as f64 / 1e6)
                } else {
                    if !p.input_is_read(i) {
                        // Read only through Comp nodes, whose own render
                        // carries its pixels: nothing of its own to key.
                        keys[i] = h(&["input-through", &tag]);
                        continue;
                    }
                    let r = inputs.pics.get(input).ok_or(format!(
                        "the pipeline reads the node's {what} input {}, and nothing is wired into it",
                        input + 1
                    ))?;
                    (h(&[&tag, &sha_hex(&r.bytes)]), r.w as f64 * r.h as f64 / 1e6)
                }
            }
            // Keyed on the pixels the host rendered, like an Input.
            Op::Shape { clip, .. } | Op::Comp { clip, .. } => {
                if *clip {
                    let c = inputs.shape_clips.get(&i).ok_or("a Spline inside the pipeline was not rendered as a clip")?;
                    (h(&["shape-clip", &c.ident()]), c.w as f64 * c.h as f64 * c.frames as f64 / 1e6)
                } else {
                    let r = inputs.shapes.get(&i).ok_or("a Spline inside the pipeline was not rendered")?;
                    (h(&["shape", &sha_hex(&r.bytes)]), r.w as f64 * r.h as f64 / 1e6)
                }
            }
            // A group makes nothing: nothing reads it, so it is never in
            // a plan's order.
            Op::Group { .. } => (h(&["group"]), 0.0),
            // A popped-out result is its file: the key it was made under.
            Op::Result { key, .. } => (key.clone(), comp_mp),
            Op::PromptIn | Op::Reference | Op::Matte | Op::Second | Op::ClipIn { .. } | Op::MatteClip => {
                return Err(OLD_STEP.into())
            }
            Op::Template { text } => {
                // `{frame}` makes the frame part of the words.
                let tick = if text.contains("{frame}") { inputs.tick.to_string() } else { String::new() };
                let mut parts = vec!["template", text.as_str(), tick.as_str()];
                parts.extend(ins.iter().copied());
                (h(&parts), 0.0)
            }
            Op::Output => (keys[step.inputs[0].unwrap()].clone(), in_mp(0).unwrap_or(0.0)),
            Op::Generate { model, seed, opts } => {
                let m = catalog::get(model).ok_or_else(|| format!("unknown model {model}"))?;
                if m.task.takes_clip() {
                    let c = step_clip(p, i, inputs)
                        .ok_or(format!("{} reads a clip: feed its video port from an Input set to clip", m.label))?;
                    m.dialect.check_clip(m.label, c.frames, c.fps, c.w, c.h)?;
                }
                let size = t2i_size(m, opts, inputs.comp);
                let seed = seed.to_string();
                let size_s = format!("{}x{}", size.0, size.1);
                let scale = format!("{:.4}", opts.scale);
                let mut parts = vec!["generate", m.id, m.endpoint, &seed, &size_s, &scale, &opts.extra];
                // A video's length and resolution are in its key; a still's
                // recipe is untouched, so every key made before video
                // existed still names its result (no KEY_VERSION bump).
                let clip = video_opts(m, opts).map(|(s, r)| format!("{s}s@{r}p"));
                if let Some(c) = &clip {
                    parts.push(c);
                }
                parts.extend(ins.iter().copied());
                let out_mp = match m.task {
                    Task::TextToImage => size.0 as f64 * size.1 as f64 / 1e6,
                    Task::Upscale => in_mp(0).unwrap_or(comp_mp) * (opts.scale as f64).powi(2),
                    Task::Cutout | Task::Depth | Task::Normals | Task::Albedo => in_mp(0).unwrap_or(comp_mp),
                    Task::Relight => in_mp(1).unwrap_or(comp_mp),
                    Task::Edit | Task::Inpaint | Task::EditRef => in_mp(1).unwrap_or(comp_mp),
                    Task::Outpaint => in_mp(1).unwrap_or(comp_mp) * (opts.scale as f64).powi(2),
                    Task::ImageToVideo | Task::TextToVideo => {
                        let (w, h) = video_size(m, opts, inputs.comp);
                        w as f64 * h as f64 / 1e6
                    }
                    // The whole clip's megapixels: what a per-megapixel
                    // video price is billed on.
                    Task::VideoEdit | Task::VideoInpaint => in_mp(1).unwrap_or(0.0),
                    Task::VideoUpscale => in_mp(0).unwrap_or(0.0) * (opts.scale as f64).powi(2),
                    Task::VideoReframe | Task::VideoMatte => in_mp(0).unwrap_or(0.0),
                    // A splat has no megapixels; nor have words.
                    Task::ImageTo3D | Task::Enhance | Task::Describe => 0.0,
                };
                (h(&parts), out_mp)
            }
            Op::Comfy { workflow, seed, .. } => {
                let bytes = std::fs::read(crate_path(workflow))
                    .map_err(|e| format!("cannot read the workflow {workflow}: {e}"))?;
                let comp = format!("{}x{}", inputs.comp.0, inputs.comp.1);
                let seed = seed.to_string();
                let wf = sha_hex(&bytes);
                let mut parts = vec!["comfy", wf.as_str(), &seed, &comp];
                parts.extend(ins.iter().copied());
                (h(&parts), comp_mp)
            }
        };
        if let Op::Generate { model, .. } = &step.op {
            let local = catalog::get(model).filter(|m| m.service == catalog::Service::Local);
            if let Some(m) = local.filter(|_| cached(dir, &key).is_none()) {
                // Not installed, or no weights: said now, not after the
                // steps above it have run.
                crate::local::ready().map_err(|e| format!("{}: {e}", m.label))?;
            }
        }
        keys[i] = key;
        mp[i] = m;
    }
    let todo: Vec<usize> = order
        .iter()
        .copied()
        .filter(|&i| {
            matches!(p.steps[i].op, Op::Generate { .. } | Op::Comfy { .. })
                && cached(dir, &keys[i]).is_none()
        })
        .collect();
    let commercial = order.iter().all(|&i| match &p.steps[i].op {
        Op::Generate { model, .. } => catalog::get(model).is_some_and(|m| m.commercial),
        Op::Comfy { commercial, .. } => *commercial,
        _ => true,
    });
    Ok(Plan { target, order, keys, todo, commercial, mp })
}

/// A typed path as the user wrote it; `~` expands.
fn crate_path(p: &str) -> PathBuf {
    if let Some(rest) = p.strip_prefix("~/").or_else(|| p.strip_prefix("~\\")) {
        // USERPROFILE: Windows has no HOME outside a Unix-style shell.
        if let Some(h) = std::env::var_os("HOME").or_else(|| std::env::var_os("USERPROFILE")) {
            return PathBuf::from(h).join(rest);
        }
    }
    PathBuf::from(p)
}

/// The output size a text-to-image step asks for.
pub fn t2i_size(m: &RemoteModel, opts: &crate::pipeline::Opts, comp: (u32, u32)) -> (u32, u32) {
    if m.task != Task::TextToImage {
        return (0, 0);
    }
    let (lo, hi) = m.dialect.mp_range();
    opts.size
        .unwrap_or_else(|| image::fit_size(comp.0.max(1), comp.1.max(1), opts.megapixels.clamp(lo, hi)))
}

/// A video step's (seconds, resolution), clamped to what its model takes;
/// `None` for a model that does not make video.
pub fn video_opts(m: &RemoteModel, opts: &crate::pipeline::Opts) -> Option<(u32, u32)> {
    if let Dialect::Ltx23 { .. } = m.dialect {
        // Length and resolution are the model's own settings
        // (`Dialect::params`, in extra and so in the key): these are
        // their defaults, which the quote reads past (`quote_settings`).
        return Some((6, 1080));
    }
    let (min_seconds, max_seconds, top_res) = m.dialect.video_range()?;
    let res = [480, 720, 1080]
        .into_iter()
        .filter(|&r| r <= top_res)
        .min_by_key(|&r| r.abs_diff(opts.res))
        .unwrap_or(720);
    Some((opts.seconds.clamp(min_seconds, max_seconds), res))
}

/// The frame a video step should come back at: its resolution on the
/// SHORT side, at the comp's aspect. An estimate (the model picks), used
/// for the quote's token count and the plan's area.
pub fn video_size(m: &RemoteModel, opts: &crate::pipeline::Opts, comp: (u32, u32)) -> (u32, u32) {
    let Some((_, res)) = video_opts(m, opts) else { return (0, 0) };
    let (cw, ch) = (comp.0.max(1) as f64, comp.1.max(1) as f64);
    let even = |v: f64| ((v / 2.0).round() as u32).max(1) * 2;
    if cw >= ch {
        (even(res as f64 * cw / ch), res)
    } else {
        (res, even(res as f64 * ch / cw))
    }
}

/// The request a Generate step sends. Images are data URIs, or whatever
/// placeholder the caller passes (quotes send no pixels).
pub fn request(
    m: &RemoteModel,
    opts: &crate::pipeline::Opts,
    seed: u32,
    comp: (u32, u32),
    prompt: Option<&str>,
    images: &[Option<String>],
) -> Result<Value> {
    request_clip(m, opts, seed, comp, prompt, images, None)
}

/// [`request`], for a model that reads a clip: `clip` is the shape of the
/// one it is sent (frame count and rate ride some requests).
pub fn request_clip(
    m: &RemoteModel,
    opts: &crate::pipeline::Opts,
    seed: u32,
    comp: (u32, u32),
    prompt: Option<&str>,
    images: &[Option<String>],
    clip: Option<ClipShape>,
) -> Result<Value> {
    let img = |k: usize| -> Result<Value> {
        images
            .get(k)
            .cloned()
            .flatten()
            .map(Value::String)
            .ok_or_else(|| anyhow!("{} needs an image", m.label))
    };
    // An edit's picture, then every reference wired after it, in port
    // order, as far as the model takes them.
    let with_refs = || -> Result<Value> {
        let refs = images.iter().skip(2).flatten().take(m.max_refs()).cloned().map(Value::String);
        Ok(Value::Array(std::iter::once(img(1)).chain(refs.map(Ok)).collect::<Result<_>>()?))
    };
    let prompt_text = prompt.unwrap_or("");
    let style = Dialect::lighting_style(prompt.unwrap_or(""));
    let prompt = || json!(m.dialect.instruct(prompt.unwrap_or("")));
    let mut v = match (m.task, m.dialect) {
        // Geometry and relighting: their own field names, no size.
        (_, Dialect::Moge) => json!({
            "image_url": img(0)?,
            // The mesh and point cloud cost time and nobody reads them.
            "export_glb": false, "export_ply": false,
        }),
        (_, Dialect::Marigold { prediction }) => json!({
            "image_url": img(0)?, "prediction_type": prediction,
        }),
        (_, Dialect::Preprocessor) => json!({ "image_url": img(0)? }),
        (_, Dialect::IcLight) => json!({
            "prompt": prompt(), "image_url": img(1)?, "seed": seed, "num_images": 1,
        }),
        (_, Dialect::LightingStyle) => json!({
            "image_url": img(1)?,
            "lighting_style": style,
        }),
        (_, Dialect::KlingEdit) => json!({
            "prompt": prompt(), "video_url": img(1)?,
            // A take is pictures; the comp's sound is not ours to send back.
            "keep_audio": false,
        }),
        (_, Dialect::LucyEdit) => json!({
            "prompt": prompt(), "video_url": img(1)?, "seed": seed,
        }),
        (_, Dialect::LtxInpaint) => {
            let c = clip.ok_or_else(|| anyhow!("{} needs the clip's length", m.label))?;
            json!({
                "prompt": prompt(),
                "video_url": img(1)?,
                "mask_video_url": img(2)?,
                "num_frames": c.frames,
                "frames_per_second": c.fps,
                "seed": seed,
                "generate_audio": false,
                // The prompt says what fills the hole; a rewrite of it
                // would change the request behind the key's back.
                "enable_prompt_expansion": false,
            })
        }
        (_, Dialect::Ltx23 { image, .. }) => {
            let mut v = json!({
                "prompt": prompt(),
                "duration": 6,
                "resolution": "1080p",
                "fps": 25,
                // A take is pictures: no soundtrack to pay for.
                "generate_audio": false,
            });
            if image {
                v["image_url"] = img(1)?;
                // The start frame decides the shape.
                v["aspect_ratio"] = json!("auto");
                if let Some(Some(e)) = images.get(2) {
                    v["end_image_url"] = json!(e);
                }
            } else {
                v["aspect_ratio"] = json!(if comp.1 > comp.0 { "9:16" } else { "16:9" });
            }
            v
        }
        (_, Dialect::LtxExtend) => json!({
            "video_url": img(1)?, "prompt": prompt(), "duration": 5, "mode": "end",
        }),
        (_, Dialect::LtxRetake) => json!({
            "video_url": img(1)?, "prompt": prompt(), "start_time": 0, "duration": 5,
            // Our clips carry no sound to keep or replace.
            "retake_mode": "replace_video",
        }),
        (_, Dialect::LtxCleanPlate | Dialect::LtxRenderToReal) => {
            let c = clip.ok_or_else(|| anyhow!("{} needs the clip's length", m.label))?;
            let mut v = json!({
                "prompt": prompt(),
                "video_url": img(1)?,
                "resolution": "720p",
                "num_frames": c.frames,
                "frames_per_second": c.fps,
                "seed": seed,
                "generate_audio": false,
            });
            if m.dialect == Dialect::LtxRenderToReal {
                v["intensity"] = json!("light");
                v["enable_detail_refine"] = json!(false);
                if let Some(Some(r)) = images.get(2) {
                    v["image_url"] = json!(r);
                }
            }
            v
        }
        (_, Dialect::Ltx25Upscale) => json!({ "video_url": img(0)?, "resolution": "1080p" }),
        (_, Dialect::Ltx25Reframe) => json!({
            "video_url": img(0)?, "aspect_ratio": "9:16", "resolution": "1080p",
        }),
        (_, Dialect::Ltx25Alpha) => json!({ "video_url": img(0)? }),
        (_, Dialect::SeedanceEdit) => json!({
            "prompt": prompt(),
            // Editing: length and shape follow the clip.
            "task": "editing",
            "video_urls": [img(1)?],
            "resolution": "720p",
            "generate_audio": false,
            "seed": seed,
        }),
        (_, Dialect::Wan27Edit) => {
            let mut v = json!({
                "prompt": prompt(),
                "video_url": img(1)?,
                "resolution": "720p",
                // The input's own length, so the take lines up with the plate.
                "duration": 0,
                "audio_setting": "origin",
                "seed": seed,
            });
            // The optional reference picture (port 2).
            if let Some(Some(r)) = images.get(2) {
                v["reference_image_url"] = json!(r);
            }
            v
        }
        (_, Dialect::WanVace) => json!({
            "prompt": prompt(),
            "video_url": img(1)?,
            "mask_video_url": img(2)?,
            // The take must line up with the plate, frame for frame.
            "match_input_num_frames": true,
            "match_input_frames_per_second": true,
            "seed": seed,
            "enable_prompt_expansion": false,
        }),
        (_, Dialect::BriaVideoEraser) => json!({
            "video_url": img(1)?,
            "mask_video_url": img(2)?,
            "output_container_and_codec": "mp4_h264",
            "auto_trim": false,
            "preserve_audio": false,
        }),
        (Task::Describe, Dialect::Llm { system }) => {
            let ask = prompt_text.trim();
            json!({
                "prompt": if ask.is_empty() { "Describe this image." } else { ask },
                "system_prompt": system,
                "image_urls": [img(1)?],
            })
        }
        (_, Dialect::Llm { system }) => json!({ "prompt": prompt(), "system_prompt": system }),
        (_, Dialect::Sam3d) => json!({
            "image_url": img(1)?, "prompt": prompt(), "seed": seed,
        }),
        (_, Dialect::BriaExpand) => {
            // The comp's size stands in for the picture's until the run
            // knows it (`expand_canvas`): a quote sends no pixels.
            let mut v = json!({ "image_url": img(1)?, "prompt": prompt(), "seed": seed });
            expand_canvas(&mut v, comp, opts.scale);
            v
        }
        (_, Dialect::BriaBackground) => json!({
            "image_url": img(1)?, "prompt": prompt(), "seed": seed,
            "refine_prompt": false, "num_images": 1,
        }),
        (_, Dialect::GeminiRef { .. }) => json!({
            "prompt": prompt(), "image_urls": [img(1)?, img(2)?], "seed": seed,
            // The first picture decides the shape.
            "aspect_ratio": "auto",
            "resolution": Dialect::gemini_resolution(opts.megapixels),
            "num_images": 1, "output_format": "png",
        }),
        (_, Dialect::Topaz { model }) => json!({
            "image_url": img(0)?, "model": model, "upscale_factor": opts.scale,
            "output_format": "png",
            // On by default at Topaz. A face redrawn behind your back is
            // not an upscale; extra turns it back on.
            "face_enhancement": false,
        }),
        (_, Dialect::Clarity) => json!({
            "image_url": img(0)?, "upscale_factor": opts.scale, "seed": seed,
        }),
        (_, Dialect::SeedVr) => json!({
            "image_url": img(0)?, "upscale_factor": opts.scale, "seed": seed,
            "output_format": "png",
        }),
        (_, Dialect::Recraft) => json!({ "image_url": img(0)? }),
        (_, Dialect::TopazVideo { model }) => json!({
            "video_url": img(0)?, "model": model, "upscale_factor": opts.scale,
            // H.265 by default, which not every movie build decodes.
            "H264_output": true,
        }),
        (_, Dialect::SeedVrVideo) => json!({
            "video_url": img(0)?, "upscale_factor": opts.scale, "seed": seed,
        }),
        (Task::VideoUpscale, _) => json!({ "video_url": img(0)?, "upscale_factor": opts.scale }),
        (Task::VideoReframe | Task::VideoMatte, _) => json!({ "video_url": img(0)? }),
        (Task::TextToVideo, Dialect::Seedance { .. }) => {
            let (secs, res) = video_opts(m, opts).ok_or_else(|| anyhow!("{} makes no video", m.label))?;
            json!({
                "prompt": prompt(),
                "duration": secs.to_string(),
                "resolution": format!("{res}p"),
                // No picture to take a shape from: the comp's.
                "aspect_ratio": if comp.1 > comp.0 { "9:16" } else { "16:9" },
                // A take is pictures; no soundtrack to pay for.
                "generate_audio": false,
            })
        }
        (Task::TextToVideo, _) => json!({ "prompt": prompt(), "seed": seed }),
        (Task::VideoEdit | Task::VideoInpaint, _) => json!({
            "prompt": prompt(), "video_url": img(1)?, "seed": seed,
        }),
        (Task::ImageToVideo, Dialect::Seedance { reference: true, .. }) => {
            let (secs, res) = video_opts(m, opts).ok_or_else(|| anyhow!("{} makes no video", m.label))?;
            json!({
                "prompt": prompt(),
                "task": "reference",
                // @Image1, then every further reference wired, in port
                // order: @Image2, @Image3…
                "image_urls": std::iter::once(img(1)?)
                    .chain(images.iter().skip(2).flatten().cloned().map(Value::String))
                    .collect::<Vec<_>>(),
                "duration": secs.to_string(),
                "resolution": format!("{res}p"),
                "aspect_ratio": "auto",
                "generate_audio": false,
                "seed": seed,
            })
        }
        (Task::ImageToVideo, Dialect::LtxLocal) => {
            let (secs, res) = video_opts(m, opts).ok_or_else(|| anyhow!("{} makes no video", m.label))?;
            let (w, h) = video_size(m, opts, comp);
            json!({
                "prompt": prompt(),
                // A path on this machine, or nothing: text-to-video.
                "image": images.get(1).cloned().flatten(),
                "seconds": secs,
                "res": res,
                "width": w,
                "height": h,
                "seed": seed,
            })
        }
        (Task::ImageToVideo, _) => {
            let (secs, res) = video_opts(m, opts).ok_or_else(|| anyhow!("{} makes no video", m.label))?;
            json!({
                "prompt": prompt(),
                "image_url": img(1)?,
                "duration": secs.to_string(),
                "resolution": format!("{res}p"),
                // The start frame decides the shape.
                "aspect_ratio": "auto",
                // A take is pictures; nobody should pay for a soundtrack
                // we throw away.
                "generate_audio": false,
            })
        }
        (Task::TextToImage, Dialect::Gemini) => {
            let (w, h) = t2i_size(m, opts, comp);
            json!({
                "prompt": prompt(),
                "aspect_ratio": Dialect::gemini_aspect(w, h),
                "resolution": Dialect::gemini_resolution(w as f32 * h as f32 / 1e6),
                "seed": seed,
                "num_images": 1,
                "output_format": "png",
            })
        }
        (Task::TextToImage, _) => {
            let (w, h) = t2i_size(m, opts, comp);
            json!({
                "prompt": prompt(),
                "image_size": {"width": w, "height": h},
                "seed": seed,
                "num_images": 1,
                "output_format": "png",
            })
        }
        (Task::Edit, Dialect::Gemini) => json!({
            "prompt": prompt(), "image_urls": with_refs()?, "seed": seed,
            // The input decides the shape.
            "aspect_ratio": "auto",
            "resolution": Dialect::gemini_resolution(opts.megapixels),
            "num_images": 1, "output_format": "png",
        }),
        (Task::Edit, Dialect::Seedream | Dialect::GptImage) => json!({
            "prompt": prompt(), "image_urls": with_refs()?,
            "num_images": 1, "output_format": "png",
        }),
        (_, Dialect::KontextMulti) => json!({
            "prompt": prompt(), "image_urls": with_refs()?, "seed": seed,
            "num_images": 1, "output_format": "png",
        }),
        (_, Dialect::QwenEditPlus) => {
            // The comp's aspect: fal's default size is a square.
            let (lo, hi) = m.dialect.mp_range();
            let (w, h) = image::fit_size(comp.0.max(1), comp.1.max(1), opts.megapixels.clamp(lo, hi));
            json!({
                "prompt": prompt(), "image_urls": with_refs()?, "seed": seed,
                "image_size": {"width": w, "height": h},
                "num_images": 1, "output_format": "png",
            })
        }
        (Task::Inpaint, Dialect::GptImage) => json!({
            "prompt": prompt(), "image_urls": [img(1)?], "mask_url": img(2)?,
            "num_images": 1, "output_format": "png",
        }),
        (Task::Edit, _) => json!({
            "prompt": prompt(), "image_url": img(1)?, "seed": seed,
            "num_images": 1, "output_format": "png",
        }),
        (Task::Inpaint, _) => json!({
            "prompt": prompt(), "image_url": img(1)?, "mask_url": img(2)?, "seed": seed,
            "num_images": 1, "output_format": "png",
        }),
        (Task::Upscale, _) => json!({
            "image_url": img(0)?, "scale": opts.scale, "output_format": "png",
        }),
        (Task::Cutout, _) => json!({
            "image_url": img(0)?, "output_format": "png", "refine_foreground": true,
        }),
        // A future depth / normals / albedo / relight model in FLUX's field
        // names.
        (Task::Depth | Task::Normals | Task::Albedo, _) => json!({ "image_url": img(0)?, "seed": seed }),
        // A future outpaint or two-picture edit in FLUX's field names.
        (Task::Outpaint | Task::ImageTo3D, _) => {
            json!({ "prompt": prompt(), "image_url": img(1)?, "seed": seed })
        }
        // A future text model in the plainest shape.
        (Task::Enhance, _) => json!({ "prompt": prompt() }),
        (Task::Describe, _) => json!({ "prompt": prompt(), "image_urls": [img(1)?] }),
        (Task::EditRef, _) => json!({
            "prompt": prompt(), "image_urls": [img(1)?, img(2)?], "seed": seed,
        }),
        (Task::Relight, _) => json!({ "prompt": prompt(), "image_url": img(1)?, "seed": seed }),
    };
    if !m.dialect.sends_seed() {
        // No seed in the schema: the authored seed still keys the cache.
        v.as_object_mut().unwrap().remove("seed");
    }
    if !opts.extra.trim().is_empty() {
        let extra: Map<String, Value> = serde_json::from_str(&opts.extra)
            .map_err(|e| anyhow!("the step's extra JSON is not an object: {e}"))?;
        let o = v.as_object_mut().unwrap();
        for (k, x) in extra {
            o.insert(k, x);
        }
    }
    check_settings(m, &v, clip)?;
    Ok(v)
}

/// What a model refuses that only the request's own settings show (the
/// panel's choices, merged from extra): said here, in the quote, before
/// anything is uploaded or billed.
fn check_settings(m: &RemoteModel, v: &Value, clip: Option<ClipShape>) -> Result<()> {
    let num = |k: &str| v.get(k).and_then(|x| x.as_f64());
    let res = v.get("resolution").and_then(|x| x.as_str()).unwrap_or("");
    match m.dialect {
        Dialect::Ltx23 { fast: true, .. } if num("duration").unwrap_or(0.0) > 10.0 => {
            if num("fps") != Some(25.0) || res != "1080p" {
                bail!("{} runs past 10 s only at 1080p and 25 fps", m.label);
            }
        }
        Dialect::LtxCleanPlate => {
            if let Some(c) = clip {
                let secs = c.frames as f32 / c.fps.max(1.0);
                let top = if res == "480p" { 15.0 } else { 6.0 };
                if secs > top + 0.05 {
                    bail!("{} takes about {top} s at {res}; this clip is {secs:.1} s", m.label);
                }
            }
        }
        Dialect::LtxRetake => {
            if let Some(c) = clip {
                let secs = c.frames as f32 / c.fps.max(1.0);
                let end = num("start_time").unwrap_or(0.0) + num("duration").unwrap_or(5.0);
                if end as f32 > secs + 0.05 {
                    bail!("{} retakes up to {end} s; this clip is {secs:.1} s", m.label);
                }
            }
        }
        _ => {}
    }
    Ok(())
}

/// Bria Expand's canvas for a `w`x`h` picture grown `scale` times: the
/// picture keeps its size and sits in the middle. Bria's canvas tops out
/// under 5000 px a side, so a big picture grows less than asked rather
/// than being refused after the upload. Leaves a request alone whose extra
/// already places the picture (`canvas_size`, `aspect_ratio`).
pub fn expand_canvas(req: &mut Value, (w, h): (u32, u32), scale: f32) {
    let Some(o) = req.as_object_mut() else { return };
    if o.contains_key("aspect_ratio") || (o.contains_key("canvas_size") && o.contains_key("original_image_size")) {
        return;
    }
    let (w, h) = (w.max(1), h.max(1));
    let s = (scale.max(1.0) as f64).min(4990.0 / w as f64).min(4990.0 / h as f64).max(1.0);
    let (cw, ch) = ((w as f64 * s).round() as u32, (h as f64 * s).round() as u32);
    o.insert("canvas_size".into(), json!([cw, ch]));
    o.insert("original_image_size".into(), json!([w, h]));
    o.insert("original_image_location".into(), json!([(cw - w.min(cw)) / 2, (ch - h.min(ch)) / 2]));
}

/// A Template step's text with its variables filled: `{a}`, `{b}`,
/// `{frame}`. Anything else in braces is left as typed.
pub fn fill_template(text: &str, a: &str, b: &str, tick: i64) -> String {
    text.replace("{a}", a).replace("{b}", b).replace("{frame}", &tick.to_string()).trim().to_string()
}

/// A PNG's size, from its header.
fn png_size(path: &Path) -> Option<(u32, u32)> {
    let f = std::io::BufReader::new(std::fs::File::open(path).ok()?);
    let r = png::Decoder::new(f).read_info().ok()?;
    Some((r.info().width, r.info().height))
}

/// The service a remote step is billed through.
pub fn service_for(cfg: &Config) -> Result<Box<dyn Service>, String> {
    cfg.can_bill()?;
    Ok(match cfg.route {
        #[cfg(feature = "relay")]
        Route::Relay => Box::new(Relay::new(&cfg.relay_url, cfg.relay_token.clone().unwrap().0)),
        #[cfg(not(feature = "relay"))]
        Route::Relay => return Err("not available in this build: use your own fal key".into()),
        Route::Own => {
            let key = cfg.fal_key.clone().unwrap().0;
            match &cfg.fal_url {
                Some(u) => Box::new(Fal::with_bases(key, u, u)),
                None => Box::new(Fal::new(key)),
            }
        }
    })
}

/// Price the billed steps of a plan. Blocking (network); run it off the UI
/// thread.
pub fn quote(p: &Pipeline, plan: &Plan, inputs: &Inputs, svc: &dyn Service, http: &Http) -> Result<Quote> {
    let billed = plan.billed(p);
    let reqs: Vec<(String, Value, f64, Option<VideoCall>, bool)> = billed
        .iter()
        .map(|&i| {
            let Op::Generate { model, seed, opts } = &p.steps[i].op else { unreachable!() };
            let m = catalog::get(model).unwrap();
            let ph: Vec<Option<String>> =
                p.steps[i].inputs.iter().map(|s| s.map(|_| "<image>".to_string())).collect();
            // A video upscaler is billed on the clip it is sent too, when
            // that is the captured one.
            let clip = step_clip(p, i, inputs).filter(|_| m.task.reads_video());
            let r = request_clip(m, opts, *seed, inputs.comp, Some("<prompt>"), &ph, clip.map(Clip::shape))?;
            let video = match clip {
                // Billed on the clip sent: its length, its size.
                Some(c) => Some(VideoCall { seconds: c.seconds(), w: c.w, h: c.h, default_res: true, rate: 1.0 }),
                None => video_opts(m, opts).map(|(secs, res)| {
                    let (w, h) = video_size(m, opts, inputs.comp);
                    VideoCall { seconds: secs as f64, w, h, default_res: res == 720, rate: 1.0 }
                }),
            };
            let video = video.map(|v| quote_settings(m, &r, v));
            // Detail refine makes the clip twice the size each way: four
            // times the megapixels it is billed on.
            let refine = r.get("enable_detail_refine").and_then(|x| x.as_bool()) == Some(true);
            let mp = if refine { plan.mp[i] * 4.0 } else { plan.mp[i] };
            Ok((m.endpoint.to_string(), r, mp, video, clip.is_some()))
        })
        .collect::<Result<_>>()?;
    let items: Vec<QuoteItem> = reqs
        .iter()
        .map(|(e, r, mp, video, reads_clip)| QuoteItem {
            endpoint: e,
            input: r,
            // A clip model's megapixels are the whole clip's (a per-MP
            // video price); an image-to-video's are unknown.
            megapixels: (video.is_none() || *reads_clip).then_some(*mp),
            video: *video,
        })
        .collect();
    svc.quote(http, &items)
}

/// A video call as the request's own settings make it (the model panel's
/// resolution, draft, duration: `Dialect::params`, merged into `req`): the
/// quote must be for what is asked, not for the defaults.
fn quote_settings(m: &RemoteModel, req: &Value, mut v: VideoCall) -> VideoCall {
    let draft = req.get("draft").and_then(|d| d.as_bool()).unwrap_or(false);
    let res = match req.get("resolution").and_then(|r| r.as_str()) {
        _ if draft => Some(480.0),
        Some("480p") => Some(480.0),
        Some("720p") => Some(720.0),
        Some("1080p") => Some(1080.0),
        Some("1440p") => Some(1440.0),
        Some("2160p" | "4k") => Some(2160.0),
        _ => None,
    };
    if let Some(res) = res {
        // The frame at that height class, at the call's own aspect.
        let k = res / (v.w.min(v.h).max(1) as f64);
        v.w = (v.w as f64 * k).round() as u32;
        v.h = (v.h as f64 * k).round() as u32;
    }
    match m.dialect {
        Dialect::Wan27Edit => {
            // Listed at 720p; 1080p is $0.15 against $0.10.
            v.rate = if res == Some(1080.0) { 1.5 } else { 1.0 };
            v.default_res = true;
            if let Some(d) = req.get("duration").and_then(|d| d.as_f64()).filter(|d| *d > 0.0) {
                v.seconds = v.seconds.min(d);
            }
        }
        Dialect::Ltx23 { .. } => {
            // Listed at 1080p; 1440p bills twice that, 2160p four times
            // (the same for the fast endpoint).
            v.rate = match res {
                Some(r) if r >= 2160.0 => 4.0,
                Some(r) if r >= 1440.0 => 2.0,
                _ => 1.0,
            };
            v.default_res = true;
            if let Some(d) = req.get("duration").and_then(|d| d.as_f64()) {
                v.seconds = d;
            }
        }
        Dialect::LtxExtend | Dialect::LtxRetake => {
            // Billed on the seconds made, not the clip sent.
            if let Some(d) = req.get("duration").and_then(|d| d.as_f64()) {
                v.seconds = d;
            }
        }
        Dialect::SeedanceEdit => {
            // Tokens count the clip sent AND the clip made, at 0.6 of the
            // price with a video input.
            v.seconds *= 2.0 * 0.6;
        }
        _ => {}
    }
    v
}

/// [`quote`] for every variant of a batch: one quote, the items of each
/// variant's billed steps in variant order (so `Plan::billed` lengths cut
/// it back apart).
pub fn quote_many(
    variants: &[(Pipeline, Plan)],
    inputs: &Inputs,
    svc: &dyn Service,
    http: &Http,
) -> Result<Quote> {
    let mut all = Quote::default();
    for (p, plan) in variants {
        if plan.billed(p).is_empty() {
            continue;
        }
        let q = quote(p, plan, inputs, svc, http)?;
        all.items.extend(q.items);
        all.balance_usd = q.balance_usd.or(all.balance_usd);
    }
    Ok(all)
}

/// Progress, per step.
#[derive(Debug, Clone, PartialEq)]
pub enum StepState {
    Cached,
    Sending,
    Queued(Option<u64>),
    Running,
    Fetching,
    Done,
}

#[derive(Debug, Clone)]
pub enum Event {
    Step(usize, StepState),
    /// The words a text step made (an enhanced prompt, a description),
    /// for the panel: they are what the next step was actually sent.
    Text(usize, String),
    /// One variant of a batch landed, and more follow (`start_many`).
    Take(Take),
    Done(Take),
    Failed(String),
}

/// A running pipeline. Dropping it cancels.
pub struct Job {
    pub rx: Receiver<Event>,
    cancel: Arc<AtomicBool>,
}

impl Job {
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }
}

impl Drop for Job {
    fn drop(&mut self) {
        self.cancel();
    }
}

/// Where the services are.
pub struct Services {
    pub remote: Option<Box<dyn Service>>,
    pub comfy: Comfy,
}

impl Services {
    pub fn from_config(cfg: &Config) -> Self {
        Self { remote: service_for(cfg).ok(), comfy: Comfy::new(&cfg.comfy_url) }
    }
}

/// What the user approved: per billed step, the most it may cost (the
/// quote they saw). The relay refuses a job priced above it.
pub type Approved = Vec<(usize, f64)>;

/// Run a plan on a thread.
pub fn start(p: Pipeline, plan: Plan, inputs: Inputs, svcs: Services, dir: PathBuf, approved: Approved) -> Job {
    start_many(vec![Variant { pipeline: p, plan, approved, inputs: None }], inputs, svcs, dir)
}

/// One pipeline of a batch, planned and approved like a single run.
pub struct Variant {
    pub pipeline: Pipeline,
    pub plan: Plan,
    pub approved: Approved,
    /// This variant's own inputs (a batch over frames captures each
    /// frame's); `None` = the batch's shared ones.
    pub inputs: Option<Inputs>,
}

/// Run several planned pipelines over the same inputs, one after another,
/// on a thread. Each lands a take: `Event::Take` for all but the last,
/// `Event::Done` for the last. A failure stops the batch (what landed
/// before it stays landed, and stays cached). Every take's recipe is
/// written beside its result (`history`).
pub fn start_many(variants: Vec<Variant>, inputs: Inputs, svcs: Services, dir: PathBuf) -> Job {
    let (tx, rx) = channel();
    let cancel = Arc::new(AtomicBool::new(false));
    let stop = cancel.clone();
    let fail = tx.clone();
    let spawned = std::thread::Builder::new().name("genai-run".into()).spawn(move || {
        let n = variants.len();
        for (k, v) in variants.into_iter().enumerate() {
            let ins = v.inputs.as_ref().unwrap_or(&inputs);
            let r = execute(&v.pipeline, &v.plan, ins, &svcs, &dir, &stop, &tx, &v.approved);
            match r {
                Ok(t) => {
                    let _ = crate::history::save_recipe(&dir, &t.key, &v.pipeline.steps);
                    let _ = crate::history::save_stages(&dir, &t.key, &v.pipeline, &v.plan.keys);
                    let _ = tx.send(if k + 1 == n { Event::Done(t) } else { Event::Take(t) });
                }
                Err(e) => {
                    let which = if n > 1 { format!("run {} of {n}: ", k + 1) } else { String::new() };
                    let _ = tx.send(Event::Failed(format!("{which}{e:#}")));
                    return;
                }
            }
        }
        if n == 0 {
            let _ = tx.send(Event::Failed("nothing to run".into()));
        }
    });
    if spawned.is_err() {
        // No threads (the browser build): say so rather than panic.
        let _ = fail.send(Event::Failed("running a pipeline needs the desktop app".into()));
    }
    Job { rx, cancel }
}

#[derive(Clone)]
enum Val {
    Text(String),
    Image(PathBuf),
}

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Write a result directory atomically: into a temp sibling, then rename.
fn commit(dir: &Path, key: &str, file: &str, bytes: &[u8], meta: &Meta) -> Result<PathBuf> {
    std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    let tmp = dir.join(format!(
        ".tmp-{key}-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0)
    ));
    std::fs::create_dir_all(&tmp)?;
    // A model's picture carries its own record; an input does not need one.
    let tagged = (!meta.model.is_empty() && file.ends_with(".png"))
        .then(|| {
            let chunks = provenance_chunks(meta);
            let refs: Vec<(&str, &str)> = chunks.iter().map(|(k, v)| (*k, v.as_str())).collect();
            image::png_with_text(bytes, &refs)
        })
        .flatten();
    std::fs::write(tmp.join(file), tagged.as_deref().unwrap_or(bytes))?;
    std::fs::write(tmp.join(META), serde_json::to_vec_pretty(meta)?)?;
    let fin = dir.join(key);
    if std::fs::rename(&tmp, &fin).is_err() {
        // Someone (another run of the same step) got there first; theirs is
        // the same pixels by construction.
        let _ = std::fs::remove_dir_all(&tmp);
        if cached(dir, key).is_none() {
            bail!("could not write {}", fin.display());
        }
    }
    Ok(fin.join(file))
}

fn data_uri_of(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let ct = match path.extension().and_then(|e| e.to_str()).unwrap_or("") {
        "mp4" => "video/mp4",
        "mov" => "video/quicktime",
        "webm" => "video/webm",
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        _ => "image/png",
    };
    Ok(http::data_uri(ct, &bytes))
}

#[allow(clippy::too_many_arguments)]
fn execute(
    p: &Pipeline,
    plan: &Plan,
    inputs: &Inputs,
    svcs: &Services,
    dir: &Path,
    stop: &AtomicBool,
    tx: &Sender<Event>,
    approved: &Approved,
) -> Result<Take> {
    let web = Http::new();
    let mut vals: Vec<Option<Val>> = vec![None; p.steps.len()];
    let mut charged: Option<f64> = None;
    let mut labels: Vec<String> = vec![];
    // Per step, the prompt it was sent (or, cached, had been sent).
    let mut sent: Vec<Option<String>> = vec![None; p.steps.len()];
    let get = |vals: &[Option<Val>], i: usize, port: usize| -> Option<Val> {
        p.steps[i].inputs.get(port).copied().flatten().and_then(|j| vals[j].clone())
    };
    for &i in &plan.order {
        if stop.load(Ordering::Relaxed) {
            bail!("cancelled");
        }
        let key = &plan.keys[i];
        let v = match &p.steps[i].op {
            Op::Prompt { text } => Val::Text(text.clone()),
            // Read only through Comp nodes: their render carries it.
            Op::Input { .. } if !p.input_is_read(i) => Val::Text(String::new()),
            Op::Input { input, clip, .. } => {
                let (bytes, file) = if *clip {
                    let c = inputs.clips.get(input).ok_or_else(|| anyhow!("missing input clip"))?;
                    (&c.bytes, "input.mp4")
                } else {
                    let png = inputs.pics.get(input).ok_or_else(|| anyhow!("missing input picture"))?;
                    (&png.bytes, "input.png")
                };
                let path = match cached(dir, key) {
                    Some(m) => dir.join(key).join(m.file),
                    None => commit(dir, key, file, bytes, &Meta { file: file.into(), when: now(), ..meta0() })?,
                };
                Val::Image(path)
            }
            Op::Shape { clip, .. } | Op::Comp { clip, .. } => {
                let (bytes, file) = if *clip {
                    let c = inputs.shape_clips.get(&i).ok_or_else(|| anyhow!("missing shape clip"))?;
                    (&c.bytes, "input.mp4")
                } else {
                    let png = inputs.shapes.get(&i).ok_or_else(|| anyhow!("missing shape matte"))?;
                    (&png.bytes, "input.png")
                };
                let path = match cached(dir, key) {
                    Some(m) => dir.join(key).join(m.file),
                    None => commit(dir, key, file, bytes, &Meta { file: file.into(), when: now(), ..meta0() })?,
                };
                Val::Image(path)
            }
            Op::Group { .. } => Val::Text(String::new()),
            Op::Result { key, file, .. } => {
                let path = dir.join(key).join(file);
                if !path.exists() {
                    bail!("a Result node's file is not on this machine");
                }
                Val::Image(path)
            }
            Op::PromptIn | Op::Reference | Op::Matte | Op::Second | Op::ClipIn { .. } | Op::MatteClip => {
                bail!(OLD_STEP)
            }
            Op::Template { text } => {
                let word = |port: usize| match get(&vals, i, port) {
                    Some(Val::Text(t)) => t,
                    _ => String::new(),
                };
                Val::Text(fill_template(text, &word(0), &word(1), inputs.tick))
            }
            Op::Output => get(&vals, i, 0).ok_or_else(|| anyhow!("Output has no input"))?,
            Op::Generate { model, seed, opts } => {
                let m = catalog::get(model).ok_or_else(|| anyhow!("unknown model {model}"))?;
                if !labels.iter().any(|l| l == m.short) {
                    labels.push(m.short.to_string());
                }
                if let Some(meta) = cached(dir, key) {
                    let _ = tx.send(Event::Step(i, StepState::Cached));
                    sent[i] = meta.prompt;
                    let path = dir.join(key).join(meta.file);
                    if m.task.is_text() {
                        let text = std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
                        let _ = tx.send(Event::Text(i, text.clone()));
                        Val::Text(text)
                    } else {
                        Val::Image(path)
                    }
                } else if m.service == catalog::Service::Local {
                    // On this machine: images go as paths, nothing is
                    // uploaded, quoted or billed.
                    let prompt = match get(&vals, i, 0) {
                        Some(Val::Text(t)) => Some(t),
                        _ => None,
                    };
                    let images: Vec<Option<String>> = (0..p.steps[i].inputs.len())
                        .map(|port| match get(&vals, i, port) {
                            Some(Val::Image(path)) => Some(path.to_string_lossy().into_owned()),
                            _ => None,
                        })
                        .collect();
                    let req = request_clip(m, opts, *seed, inputs.comp, prompt.as_deref(), &images, None)?;
                    let _ = tx.send(Event::Step(i, StepState::Sending));
                    let txp = tx.clone();
                    let phase = move |ph: Phase| {
                        let _ = txp.send(Event::Step(
                            i,
                            match ph {
                                Phase::Queued(n) => StepState::Queued(n),
                                Phase::Running => StepState::Running,
                            },
                        ));
                    };
                    let cx = Cx { http: &web, cancel: stop, phase: &phase, deadline: Duration::from_secs(60 * 60) };
                    let work = dir.join(format!(".local-{key}"));
                    let out = work.join("result.mp4");
                    let ran = crate::local::run(&req, &work, &out, &cx);
                    let bytes = ran.and_then(|()| Ok(std::fs::read(&out)?));
                    let _ = std::fs::remove_dir_all(&work);
                    let bytes = bytes.map_err(|e| anyhow!("{}: {e}", m.label))?;
                    let path = commit(
                        dir,
                        key,
                        "result.mp4",
                        &bytes,
                        &Meta {
                            file: "result.mp4".into(),
                            model: m.id.into(),
                            endpoint: m.endpoint.into(),
                            service: "local".into(),
                            usd: Some(0.0),
                            when: now(),
                            nsfw_flagged: false,
                            prompt: prompt.as_deref().map(str::to_string),
                            seed: Some(*seed),
                            request: Some(scrub_request(&req)),
                        },
                    )?;
                    Val::Image(path)
                } else {
                    let svc = svcs.remote.as_deref().ok_or_else(|| {
                        anyhow!(if cfg!(feature = "relay") {
                            "no way to run this step: add a relay token or a fal key"
                        } else {
                            "no way to run this step: add a fal key"
                        })
                    })?;
                    let prompt = match get(&vals, i, 0) {
                        Some(Val::Text(t)) => Some(t),
                        _ => None,
                    };
                    let mut images = vec![];
                    for port in 0..p.steps[i].inputs.len() {
                        images.push(match get(&vals, i, port) {
                            Some(Val::Image(path)) => Some(data_uri_of(&path)?),
                            _ => None,
                        });
                    }
                    let clip = step_clip(p, i, inputs).filter(|_| m.task.takes_clip()).map(Clip::shape);
                    sent[i] = prompt.clone();
                    let mut req = request_clip(m, opts, *seed, inputs.comp, prompt.as_deref(), &images, clip)?;
                    if m.dialect == Dialect::BriaExpand && opts.extra.trim().is_empty() {
                        // Now the picture is in hand, the canvas is ITS
                        // size grown, not the comp's.
                        let size = match get(&vals, i, 1) {
                            Some(Val::Image(path)) => png_size(&path),
                            _ => None,
                        };
                        if let (Some(size), Some(o)) = (size, req.as_object_mut()) {
                            o.remove("canvas_size");
                            expand_canvas(&mut req, size, opts.scale);
                        }
                    }
                    let _ = tx.send(Event::Step(i, StepState::Sending));
                    let txp = tx.clone();
                    let phase = move |ph: Phase| {
                        let _ = txp.send(Event::Step(
                            i,
                            match ph {
                                Phase::Queued(n) => StepState::Queued(n),
                                Phase::Running => StepState::Running,
                            },
                        ));
                    };
                    let cx = Cx { http: &web, cancel: stop, phase: &phase, deadline: Duration::from_secs(30 * 60) };
                    let max = approved.iter().find(|(s, _)| *s == i).map(|(_, u)| *u);
                    let ran = svc.run(m.endpoint, &req, max, Some(plan.mp[i]), &cx)?;
                    if m.task.is_text() {
                        // Words, not a file to fetch: kept as `result.txt`
                        // so the step caches like any other.
                        let text = service::output_text(&ran.output).map_err(|e| anyhow!("{}: {e}", m.label))?;
                        let usd = ran.charged_usd.or(max);
                        if let Some(u) = usd {
                            *charged.get_or_insert(0.0) += u;
                        }
                        commit(
                            dir,
                            key,
                            "result.txt",
                            text.as_bytes(),
                            &Meta {
                                file: "result.txt".into(),
                                model: m.id.into(),
                                endpoint: m.endpoint.into(),
                                service: svc.name().into(),
                                usd,
                                when: now(),
                                nsfw_flagged: false,
                                prompt: prompt.clone(),
                                seed: Some(*seed),
                                request: Some(scrub_request(&req)),
                            },
                        )?;
                        let _ = tx.send(Event::Text(i, text.clone()));
                        let _ = tx.send(Event::Step(i, StepState::Done));
                        vals[i] = Some(Val::Text(text));
                        continue;
                    }
                    let (url, nsfw) = if m.task.is_video() {
                        let url = service::output_video_at(&ran.output, m.dialect.output_key(m.task))
                            .ok_or_else(|| anyhow!("{} returned no video", m.label))?;
                        (url, false)
                    } else {
                        service::output_image_at(&ran.output, m.dialect.output_key(m.task))
                            .ok_or_else(|| anyhow!("{} returned no image", m.label))?
                    };
                    let _ = tx.send(Event::Step(i, StepState::Fetching));
                    let (bytes, ct) = web.download(&url)?;
                    // A splat comes back as an octet stream, often from a
                    // URL with no extension: the task names the file.
                    let ext = if m.task.is_3d() { "ply" } else { image::ext_for(ct.as_deref(), &url) };
                    let file = format!("result.{ext}");
                    let usd = ran.charged_usd.or(max);
                    if let Some(u) = usd {
                        *charged.get_or_insert(0.0) += u;
                    }
                    let path = commit(
                        dir,
                        key,
                        &file,
                        &bytes,
                        &Meta {
                            file: file.clone(),
                            model: m.id.into(),
                            endpoint: m.endpoint.into(),
                            service: svc.name().into(),
                            usd,
                            when: now(),
                            nsfw_flagged: nsfw,
                            prompt: prompt.clone(),
                            seed: Some(*seed),
                            request: Some(scrub_request(&req)),
                        },
                    )?;
                    if nsfw {
                        bail!(
                            "{}'s safety filter blanked the picture (the call was still billed); \
                             change the prompt or the seed",
                            m.label
                        );
                    }
                    Val::Image(path)
                }
            }
            Op::Comfy { workflow, seed, .. } => {
                if !labels.iter().any(|l| l == "comfy") {
                    labels.push("comfy".into());
                }
                if let Some(meta) = cached(dir, key) {
                    let _ = tx.send(Event::Step(i, StepState::Cached));
                    sent[i] = meta.prompt;
                    Val::Image(dir.join(key).join(meta.file))
                } else {
                    let text = std::fs::read_to_string(crate_path(workflow))
                        .with_context(|| format!("read {workflow}"))?;
                    let wf = comfy::parse_workflow(&text)?;
                    let read = |port: usize| -> Result<Option<Vec<u8>>> {
                        match get(&vals, i, port) {
                            Some(Val::Image(path)) => Ok(Some(std::fs::read(path)?)),
                            _ => Ok(None),
                        }
                    };
                    let subs = Subs {
                        prompt: match get(&vals, i, 0) {
                            Some(Val::Text(t)) => Some(t),
                            _ => None,
                        },
                        seed: *seed,
                        width: inputs.comp.0,
                        height: inputs.comp.1,
                        image: read(1)?,
                        mask: read(2)?,
                    };
                    sent[i] = subs.prompt.clone();
                    let _ = tx.send(Event::Step(i, StepState::Sending));
                    let txp = tx.clone();
                    let phase = move |ph: Phase| {
                        let _ = txp.send(Event::Step(
                            i,
                            match ph {
                                Phase::Queued(n) => StepState::Queued(n),
                                Phase::Running => StepState::Running,
                            },
                        ));
                    };
                    let cx = Cx { http: &web, cancel: stop, phase: &phase, deadline: Duration::from_secs(60 * 60) };
                    let (bytes, ct) = svcs.comfy.run(&wf, &subs, &cx)?;
                    let file = format!("result.{}", image::ext_for(ct.as_deref(), ""));
                    let path = commit(
                        dir,
                        key,
                        &file,
                        &bytes,
                        &Meta {
                            file: file.clone(),
                            model: "comfy".into(),
                            endpoint: workflow.clone(),
                            service: "comfyui".into(),
                            when: now(),
                            prompt: subs.prompt.clone(),
                            seed: Some(*seed),
                            ..meta0()
                        },
                    )?;
                    Val::Image(path)
                }
            }
        };
        let _ = tx.send(Event::Step(i, StepState::Done));
        vals[i] = Some(v);
    }
    // A video step's result travels as an Image value too: a path on
    // disk. Only Output may take one (`Op::accepts`).
    let Some(Val::Image(path)) = vals[plan.target].clone() else {
        bail!("the pipeline did not end in a picture");
    };
    let key = plan.keys[plan.target].clone();
    let file = path.file_name().and_then(|f| f.to_str()).unwrap_or("result.png").to_string();
    let prompt = plan.order.iter().find_map(|&i| match &p.steps[i].op {
        Op::Prompt { text } if !text.is_empty() => Some(text.chars().take(40).collect::<String>()),
        _ => None,
    });
    let mut label = labels.join(" → ");
    if let Some(t) = prompt {
        label = format!("{label}: {t}");
    }
    // A depth or normals model feeding Output makes a data take.
    let data = p.steps[plan.target].inputs.first().copied().flatten().is_some_and(|i| {
        matches!(&p.steps[i].op, Op::Generate { model, .. }
            if catalog::get(model).is_some_and(|m| m.task.is_data()))
    });
    // The words behind the picture: what the step feeding Output was sent,
    // else (an upscale after a generate) the nearest model step's.
    let fed = p.steps[plan.target].inputs.first().copied().flatten();
    let prompt = fed
        .and_then(|i| sent[i].clone())
        .or_else(|| plan.order.iter().rev().find_map(|&i| sent[i].clone()))
        .filter(|t| !t.is_empty())
        .unwrap_or_default();
    Ok(Take { key, file, tick: inputs.tick, commercial: plan.commercial, usd: charged, label, when: now(), data, prompt })
}

fn meta0() -> Meta {
    Meta {
        file: String::new(),
        model: String::new(),
        endpoint: String::new(),
        service: String::new(),
        usd: None,
        when: 0,
        nsfw_flagged: false,
        prompt: None,
        seed: None,
        request: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::Opts;

    fn m(id: &str) -> &'static RemoteModel {
        catalog::get(id).unwrap()
    }

    #[test]
    fn multi_picture_edits_grow_a_port_per_reference_and_send_them_all() {
        use crate::pipeline::{Op, Pipe};
        let o = Opts::default();
        for id in [
            "fal/seedream-5-pro-edit",
            "fal/nano-banana-pro-edit",
            "fal/nano-banana-2-edit",
            "fal/gpt-image-2-edit",
            "fal/flux-kontext-pro-multi",
            "fal/qwen-image-edit-plus",
        ] {
            let model = m(id);
            assert!(model.grows_refs(), "{id}");
            // A step with nothing extra wired (an old setup) has its two
            // ports and one free reference; each one wired adds another.
            let step = |refs| Op::Generate { model: id.into(), seed: 1, opts: Opts { refs, ..Opts::default() } };
            let names: Vec<_> = step(0).ports().iter().map(|p| p.1).collect();
            assert_eq!(names, ["prompt", "image", "reference 2"], "{id}");
            assert_eq!(step(2).ports().len(), 5, "{id}");
            assert!(step(2).ports()[2..].iter().all(|p| p.0 == Pipe::Image) && step(2).optional(4));
            assert_eq!(step(99).ports().len(), 2 + model.max_refs(), "{id}: capped");
            // The picture first, then every wired reference in port order;
            // an unwired port in between is skipped.
            let pics = [None, Some("data:a".into()), Some("data:b".into()), None, Some("data:c".into())];
            let r = request(model, &o, 7, (2048, 1080), Some("p"), &pics).unwrap();
            assert_eq!(r["image_urls"], json!(["data:a", "data:b", "data:c"]), "{id}");
            assert!(r.get("image_url").is_none(), "{id}");
            // Only the edited picture: still a list of one.
            let one = request(model, &o, 7, (2048, 1080), Some("p"), &pics[..2]).unwrap();
            assert_eq!(one["image_urls"], json!(["data:a"]), "{id}");
        }
        // No more than the provider takes.
        let gpt = m("fal/gpt-image-2-edit");
        let many: Vec<Option<String>> =
            std::iter::once(None).chain((0..30).map(|k| Some(format!("data:{k}")))).collect();
        let r = request(gpt, &o, 7, (2048, 1080), Some("p"), &many).unwrap();
        assert_eq!(r["image_urls"].as_array().unwrap().len(), 16);
        // Qwen gets the comp's shape, not fal's square default.
        let q = request(m("fal/qwen-image-edit-plus"), &o, 7, (2048, 1080), Some("p"), &many[..2]).unwrap();
        assert!(q["image_size"]["width"].as_u64() > q["image_size"]["height"].as_u64(), "{q}");
        // The fixed two-picture edits and single-picture models are unchanged.
        assert!(!m("fal/nano-banana-pro-edit-ref").grows_refs() && !m("fal/flux-kontext-pro").grows_refs());
        assert!(!m("fal/gpt-image-2-inpaint").grows_refs() && !m("fal/nano-banana-pro").grows_refs());
    }

    #[test]
    fn seedream_asks_for_its_own_shape() {
        let o = Opts::default();
        let t = request(m("fal/seedream-5-pro"), &o, 7, (2048, 1080), Some("p"), &[None]).unwrap();
        assert!(t.get("seed").is_none(), "Seedream has no seed field: {t}");
        let (w, h) = (t["image_size"]["width"].as_u64().unwrap(), t["image_size"]["height"].as_u64().unwrap());
        assert!(w * h >= 1024 * 1024 && w * h <= 1536 * 1536, "{w}x{h}: inside the lower price tier");
        let img = Some("data:x".to_string());
        let e = request(m("fal/seedream-5-pro-edit"), &o, 7, (2048, 1080), Some("p"), &[None, img]).unwrap();
        assert_eq!(e["image_urls"], json!(["data:x"]));
        assert!(e.get("image_url").is_none() && e.get("seed").is_none());
    }

    #[test]
    fn nano_banana_asks_by_aspect_and_resolution() {
        let o = Opts::default();
        let t = request(m("fal/nano-banana-pro"), &o, 7, (2048, 1080), Some("p"), &[None]).unwrap();
        assert!(t.get("image_size").is_none(), "{t}");
        assert_eq!(t["aspect_ratio"], "16:9");
        assert_eq!(t["resolution"], "1K");
        assert_eq!(t["seed"], 7);
        let big = Opts { megapixels: 3.0, ..Opts::default() };
        let t = request(m("fal/nano-banana-2"), &big, 7, (1080, 1920), Some("p"), &[None]).unwrap();
        assert_eq!((t["aspect_ratio"].as_str(), t["resolution"].as_str()), (Some("9:16"), Some("2K")));
        let img = Some("data:x".to_string());
        let e = request(m("fal/nano-banana-pro-edit"), &o, 7, (2048, 1080), Some("p"), &[None, img]).unwrap();
        assert_eq!(e["image_urls"], json!(["data:x"]));
        assert_eq!(e["aspect_ratio"], "auto");
        assert!(e.get("image_url").is_none());
    }

    #[test]
    fn geometry_and_relight_models_ask_in_their_own_words() {
        let o = Opts::default();
        let img = Some("data:x".to_string());
        let g = request(m("fal/moge-3-normals"), &o, 7, (2048, 1080), None, &[img.clone()]).unwrap();
        assert_eq!(g["image_url"], "data:x");
        assert_eq!((g["export_glb"].as_bool(), g["export_ply"].as_bool()), (Some(false), Some(false)));
        assert!(g.get("seed").is_none() && g.get("image_size").is_none(), "{g}");
        let mg = request(m("fal/marigold-albedo"), &o, 7, (2048, 1080), None, &[img.clone()]).unwrap();
        assert_eq!(mg["prediction_type"], "albedo");
        let d = request(m("fal/midas-depth"), &o, 7, (2048, 1080), None, &[img.clone()]).unwrap();
        assert_eq!(d, json!({"image_url": "data:x"}));
        let ic = request(m("fal/iclight-v2"), &o, 7, (2048, 1080), Some("rim light"), &[None, img.clone()]).unwrap();
        assert_eq!((ic["prompt"].as_str(), ic["image_url"].as_str(), ic["seed"].as_u64()), (Some("rim light"), Some("data:x"), Some(7)));
        let r = request(m("fal/relighting"), &o, 7, (2048, 1080), Some("Golden-hour, warm"), &[None, img]).unwrap();
        assert_eq!(r, json!({"image_url": "data:x", "lighting_style": "golden_hour"}));
        assert_eq!(Dialect::lighting_style("backlight please"), "backlight");
        assert_eq!(Dialect::lighting_style("whatever"), "natural");
        // Which field holds the picture.
        let out = json!({"depth_map": {"url": "d"}, "normal_map": {"url": "n"}});
        let key = |id: &str| m(id).dialect.output_key(m(id).task);
        assert_eq!(service::output_image_at(&out, key("fal/midas-normals")).unwrap().0, "n");
        assert_eq!(service::output_image_at(&out, key("fal/midas-depth")).unwrap().0, "d");
        assert_eq!(service::output_image_at(&out, key("fal/moge-2-normals")).unwrap().0, "n");
        assert!(m("fal/marigold-normals").task.is_data() && !m("fal/marigold-albedo").task.is_data());
    }

    #[test]
    fn gpt_image_takes_lists_a_mask_and_no_seed() {
        let o = Opts { megapixels: 0.25, ..Opts::default() };
        let t = request(m("fal/gpt-image-2"), &o, 7, (2048, 1080), Some("p"), &[None]).unwrap();
        assert!(t.get("seed").is_none(), "{t}");
        let (w, h) = (t["image_size"]["width"].as_u64().unwrap(), t["image_size"]["height"].as_u64().unwrap());
        assert!(w * h >= 655_360 && w % 16 == 0 && h % 16 == 0, "{w}x{h}: above GPT Image's floor");
        let img = Some("data:x".to_string());
        let mask = Some("data:m".to_string());
        let e = request(m("fal/gpt-image-2-edit"), &o, 7, (2048, 1080), Some("p"), &[None, img.clone()]).unwrap();
        assert_eq!(e["image_urls"], json!(["data:x"]));
        assert!(e.get("seed").is_none() && e.get("image_url").is_none());
        let i = request(m("fal/gpt-image-2-inpaint"), &o, 7, (2048, 1080), Some("p"), &[None, img, mask]).unwrap();
        assert_eq!(i["image_urls"], json!(["data:x"]));
        assert_eq!(i["mask_url"], "data:m");
    }

    #[test]
    fn text_models_ask_with_a_system_prompt_and_answer_in_words() {
        let one = [None, Some("data:a".to_string())];
        let r = request(m("fal/prompt-enhance"), &Opts::default(), 7, (1000, 500), Some("a moth"), &[None]).unwrap();
        assert_eq!(r["prompt"], "a moth");
        assert!(r["system_prompt"].as_str().unwrap().contains("rewrite prompts"));
        assert!(r.get("seed").is_none());
        let r = request(m("fal/describe"), &Opts::default(), 7, (1000, 500), Some(""), &one).unwrap();
        assert_eq!((r["prompt"].as_str(), &r["image_urls"]), (Some("Describe this image."), &json!(["data:a"])));
        let r = request(m("fal/describe"), &Opts::default(), 7, (1000, 500), Some("the light"), &one).unwrap();
        assert_eq!(r["prompt"], "the light");
        assert_eq!(service::output_text(&json!({"output": "  a moth at night \n"})), Ok("a moth at night".into()));
        assert!(service::output_text(&json!({"output": "", "error": null})).is_err());
        assert_eq!(service::output_text(&json!({"output": "x", "error": "rate limited"})), Err("rate limited".into()));
    }

    #[test]
    fn a_template_fills_its_variables_and_keys_on_what_it_reads() {
        assert_eq!(fill_template("{a}, {b} at frame {frame} {c}", "a moth", "", 1001), "a moth,  at frame 1001 {c}");
        let dir = std::env::temp_dir().join("wobble-genai-template-none");
        let mut p = Pipeline::bare(vec![]);
        let a = p.push(Op::Prompt { text: "a moth".into() }, [0.0, 0.0]);
        let t = p.push(Op::Template { text: "{a}, golden hour".into() }, [0.0, 0.0]);
        let g = p.push(Op::Generate { model: "fal/flux-schnell".into(), seed: 1, opts: Opts::default() }, [0.0, 0.0]);
        let o = p.push(Op::Output, [0.0, 0.0]);
        assert!(p.connect(a, t, 0) && p.connect(t, g, 0) && p.connect(g, o, 0));
        let mut ins = Inputs { comp: (1920, 1080), tick: 5, ..Default::default() };
        let k1 = plan(&p, &ins, &dir).unwrap().keys[g].clone();
        // The frame is not in the words, so it is not in the key.
        ins.tick = 6;
        assert_eq!(plan(&p, &ins, &dir).unwrap().keys[g], k1);
        // The variable is.
        p.steps[a].op = Op::Prompt { text: "a lamp".into() };
        assert_ne!(plan(&p, &ins, &dir).unwrap().keys[g], k1);
        // And with {frame} in the text, so is the frame.
        p.steps[t].op = Op::Template { text: "{a}, frame {frame}".into() };
        let k5 = plan(&p, &ins, &dir).unwrap().keys[g].clone();
        ins.tick = 7;
        assert_ne!(plan(&p, &ins, &dir).unwrap().keys[g], k5);
    }

    #[test]
    fn sam_3d_names_its_object_and_hands_back_a_splat() {
        let two = [None, Some("data:a".to_string())];
        let sam = m("fal/sam-3d-objects");
        let r = request(sam, &Opts::default(), 7, (1000, 500), Some("chair"), &two).unwrap();
        assert_eq!((r["prompt"].as_str(), r["image_url"].as_str(), r["seed"].as_u64()), (Some("chair"), Some("data:a"), Some(7)));
        let out = json!({"gaussian_splat": {"url": "https://x/scene"}, "model_glb": {"url": "https://x/m.glb"}});
        let got = service::output_image_at(&out, sam.dialect.output_key(sam.task));
        assert_eq!(got, Some(("https://x/scene".to_string(), false)));
    }

    #[test]
    fn ready_made_edits_ask_in_their_own_words() {
        let opts = Opts { scale: 2.0, ..Opts::default() };
        let two = [None, Some("data:a".to_string()), Some("data:b".to_string())];
        // Expand: the picture keeps its size, in the middle of the canvas.
        let r = request(m("fal/bria-expand"), &opts, 7, (1000, 500), Some("a beach"), &two).unwrap();
        assert_eq!(r["canvas_size"], json!([2000, 1000]));
        assert_eq!(r["original_image_size"], json!([1000, 500]));
        assert_eq!(r["original_image_location"], json!([500, 250]));
        assert_eq!((r["prompt"].as_str(), r["seed"].as_u64()), (Some("a beach"), Some(7)));
        // A big picture grows as far as Bria's canvas goes, never past it.
        let mut v = json!({});
        expand_canvas(&mut v, (4000, 2000), 3.0);
        assert_eq!(v["canvas_size"], json!([4990, 2495]));
        // An extra that picks an aspect places the picture itself.
        let mut v = json!({"aspect_ratio": "16:9"});
        expand_canvas(&mut v, (1000, 500), 2.0);
        assert!(v.get("canvas_size").is_none());
        let r = request(m("fal/bria-background-replace"), &opts, 7, (1000, 500), Some("a beach"), &two).unwrap();
        assert_eq!((r["image_url"].as_str(), r["refine_prompt"].as_bool()), (Some("data:a"), Some(false)));
        // Colour transfer: both pictures, the image first, and the fixed
        // instruction ahead of whatever was typed.
        let r = request(m("fal/colour-transfer"), &opts, 7, (1000, 500), Some("warmer"), &two).unwrap();
        assert_eq!(r["image_urls"], json!(["data:a", "data:b"]));
        let p = r["prompt"].as_str().unwrap();
        assert!(p.starts_with("Recolour the first image") && p.ends_with(" warmer"), "{p}");
        let r = request(m("fal/colour-transfer"), &opts, 7, (1000, 500), Some(""), &two).unwrap();
        assert!(r["prompt"].as_str().unwrap().ends_with("second image."));
        let r = request(m("fal/nano-banana-pro-edit-ref"), &opts, 7, (1000, 500), Some("in this style"), &two).unwrap();
        assert_eq!(r["prompt"], "in this style");
        // And it needs the second picture.
        let one = [None, Some("data:a".to_string()), None];
        assert!(request(m("fal/colour-transfer"), &opts, 7, (1000, 500), Some(""), &one).is_err());
    }

    #[test]
    fn a_second_picture_keys_the_step_it_feeds() {
        let dir = std::env::temp_dir().join("wobble-genai-second-none");
        let p = Pipeline::for_model("fal/colour-transfer").unwrap();
        let png = |b: u8| Png { bytes: vec![b; 8], w: 4, h: 4 };
        assert_eq!(p.ins.len(), 2, "the image and the second picture");
        let mut ins = Inputs { comp: (1920, 1080), ..Default::default() };
        ins.pics.insert(0, png(1));
        assert!(plan(&p, &ins, &dir).unwrap_err().contains("image input 2"));
        ins.pics.insert(1, png(2));
        let a = plan(&p, &ins, &dir).unwrap();
        ins.pics.insert(1, png(3));
        let b = plan(&p, &ins, &dir).unwrap();
        assert_ne!(a.keys[a.target], b.keys[b.target], "another reference is another request");
    }

    #[test]
    fn upscalers_ask_in_their_own_words() {
        let opts = Opts { scale: 3.0, ..Opts::default() };
        let img = [Some("data:image/png;base64,AA".to_string())];
        let r = request(m("fal/topaz-upscale-hifi"), &opts, 7, (1920, 1080), None, &img).unwrap();
        assert_eq!(r["model"], "High Fidelity V2");
        assert_eq!(r["upscale_factor"], 3.0);
        assert_eq!(r["face_enhancement"], false, "a face is never redrawn unasked");
        assert!(r.get("seed").is_none() && r.get("scale").is_none());
        let r = request(m("fal/clarity-upscaler"), &opts, 7, (1920, 1080), None, &img).unwrap();
        assert_eq!((r["upscale_factor"].as_f64(), r["seed"].as_u64()), (Some(3.0), Some(7)));
        let r = request(m("fal/seedvr2-upscale"), &opts, 7, (1920, 1080), None, &img).unwrap();
        assert_eq!((r["upscale_factor"].as_f64(), r["seed"].as_u64()), (Some(3.0), Some(7)));
        let r = request(m("fal/recraft-crisp-upscale"), &opts, 7, (1920, 1080), None, &img).unwrap();
        assert_eq!(r.as_object().unwrap().len(), 1, "image_url and nothing else: {r}");
        // Real-ESRGAN keeps the request it always had (and so its keys).
        let r = request(m("fal/esrgan"), &opts, 7, (1920, 1080), None, &img).unwrap();
        assert_eq!(r["scale"], 3.0);
        let clip = [Some("data:video/mp4;base64,AA".to_string())];
        let r = request(m("fal/topaz-video-upscale"), &opts, 7, (1920, 1080), None, &clip).unwrap();
        assert_eq!((r["model"].as_str(), r["H264_output"].as_bool()), (Some("Proteus"), Some(true)));
        assert!(r["video_url"].is_string() && r.get("image_url").is_none());
        let r = request(m("fal/seedvr2-video-upscale"), &opts, 7, (1920, 1080), None, &clip).unwrap();
        assert_eq!((r["upscale_factor"].as_f64(), r["seed"].as_u64()), (Some(3.0), Some(7)));
        // An extra field still wins, which is how face enhancement comes back.
        let on = Opts { extra: r#"{"face_enhancement": true}"#.into(), ..opts };
        let r = request(m("fal/topaz-upscale"), &on, 7, (1920, 1080), None, &img).unwrap();
        assert_eq!(r["face_enhancement"], true);
    }

    #[test]
    fn seedance_sends_strings_no_audio_and_clamps_to_the_model() {
        let o = Opts { seconds: 40, res: 1080, ..Opts::default() };
        let img = Some("data:x".to_string());
        let r = request(m("fal/seedance-2-fast-i2v"), &o, 1, (2048, 1080), Some("p"), &[None, img.clone()]).unwrap();
        assert_eq!(r["duration"], "15");
        assert_eq!(r["resolution"], "720p");
        assert_eq!(r["generate_audio"], false);
        assert_eq!(r["image_url"], "data:x");
        assert!(r.get("seed").is_none());
        let r = request(m("fal/seedance-2.5-i2v"), &o, 1, (2048, 1080), Some("p"), &[None, img]).unwrap();
        assert_eq!((r["duration"].as_str(), r["resolution"].as_str()), (Some("30"), Some("1080p")));
        assert_eq!(video_size(m("fal/seedance-2-fast-i2v"), &Opts::default(), (2048, 1080)), (1366, 720));
    }

    #[test]
    fn a_clip_s_length_keys_a_video_step_and_nothing_else() {
        let dir = std::env::temp_dir().join("wobble-genai-keys-none");
        let mut ins = Inputs { comp: (2048, 1080), tick: 0, ..Default::default() };
        ins.pics.insert(0, Png { bytes: vec![1, 2, 3], w: 4, h: 4 });
        let key = |p: &Pipeline| plan(p, &ins, &dir).unwrap().keys[1].clone();
        let set = |p: &mut Pipeline, secs: u32| {
            if let Op::Generate { opts, .. } = &mut p.steps[1].op {
                opts.seconds = secs;
            }
        };
        let mut p = Pipeline::default();
        let k = key(&p);
        set(&mut p, 9);
        assert_eq!(key(&p), k, "a still's key does not see the video knobs");
        p.set_model(1, "fal/seedance-2-fast-i2v");
        let r = p.input_step(0).unwrap();
        assert!(p.connect(r, 1, 1));
        let k = key(&p);
        set(&mut p, 10);
        assert_ne!(key(&p), k);
    }

    #[test]
    fn an_input_keys_under_the_name_the_old_fixed_input_did() {
        // A result cached before the input bar is still found.
        assert_eq!(input_tag(false, 0, false), "reference");
        assert_eq!(input_tag(false, 1, false), "second");
        assert_eq!(input_tag(true, 0, false), "matte");
        assert_eq!(input_tag(false, 0, true), "clip");
        assert_eq!(input_tag(true, 0, true), "matte-clip");
        assert_ne!(input_tag(false, 2, false), input_tag(true, 2, false));
    }

    fn clip(frames: u32, w: u32, h: u32) -> Clip {
        Clip { bytes: vec![frames as u8, w as u8, 7], w, h, frames, fps: 24.0, key: String::new() }
    }

    #[test]
    fn clip_models_ask_in_their_own_words() {
        let o = Opts::default();
        let v = Some("data:video/mp4;base64,AAAA".to_string());
        let shape = ClipShape { frames: 72, fps: 24.0 };
        let k = request_clip(m("fal/kling-o3-pro-video-edit"), &o, 7, (1920, 1080), Some("p"), &[None, v.clone()], Some(shape)).unwrap();
        assert_eq!(k["video_url"], v.clone().unwrap());
        assert_eq!(k["keep_audio"], false);
        assert!(k.get("seed").is_none(), "{k}");
        let l = request_clip(m("fal/ltx-2.3-inpaint"), &o, 7, (1920, 1080), Some("p"), &[None, v.clone(), Some("mask".into())], Some(shape)).unwrap();
        assert_eq!(l["mask_video_url"], "mask");
        assert_eq!(l["num_frames"], 72);
        assert_eq!(l["frames_per_second"], 24.0);
        assert_eq!(l["seed"], 7);
        assert_eq!(l["generate_audio"], false);
        // LTX cannot be asked without knowing the clip.
        assert!(request(m("fal/ltx-2.3-inpaint"), &o, 7, (1920, 1080), Some("p"), &[None, v.clone(), v]).is_err());
    }

    #[test]
    fn a_clip_pipeline_keys_on_the_clips_and_checks_them_before_sending() {
        let dir = std::env::temp_dir().join(format!("wobble-genai-clip-{}", std::process::id()));
        let p = Pipeline::for_model("fal/ltx-2.3-inpaint").unwrap();
        assert_eq!(p.reads(), vec![(0, true), (1, true)]);
        let mut ins = Inputs { comp: (1920, 1080), ..Default::default() };
        assert!(plan(&p, &ins, &dir).unwrap_err().contains("clip"));
        ins.clips.insert(0, clip(72, 1280, 720));
        ins.clips.insert(1, clip(72, 1280, 720));
        let a = plan(&p, &ins, &dir).unwrap();
        ins.clips.insert(1, clip(73, 1280, 720));
        let b = plan(&p, &ins, &dir).unwrap();
        assert_ne!(a.keys[a.target], b.keys[b.target], "the matte clip's bytes are in the key");
        // The whole clip's megapixels, for a per-MP video price.
        let g = p.model_step().unwrap();
        assert!((a.mp[g] - 1280.0 * 720.0 * 72.0 / 1e6).abs() < 1e-6);
        ins.clips.insert(0, clip(4, 1280, 720));
        assert!(plan(&p, &ins, &dir).unwrap_err().contains("9-481"));

        let k = Pipeline::for_model("fal/kling-o3-pro-video-edit").unwrap();
        ins.clips.insert(0, clip(24, 1280, 720));
        assert!(plan(&k, &ins, &dir).unwrap_err().contains("3-15 s"));
        ins.clips.insert(0, clip(72, 640, 360));
        assert!(plan(&k, &ins, &dir).unwrap_err().contains("720 px"));
        ins.clips.insert(0, clip(72, 1280, 720));
        assert!(plan(&k, &ins, &dir).is_ok());
    }

    #[test]
    fn a_quote_follows_the_model_s_own_settings() {
        let wan = m("fal/wan-2.7-video-edit");
        let v = VideoCall { seconds: 5.0, w: 1280, h: 720, default_res: true, rate: 1.0 };
        let hd = quote_settings(wan, &json!({"resolution": "1080p"}), v);
        assert_eq!((hd.w, hd.h, hd.rate), (1920, 1080, 1.5));
        assert!((service::price_of(0.1, "second", None, Some(hd)).usd - 0.75).abs() < 1e-9);
        let cut = quote_settings(wan, &json!({"resolution": "720p", "duration": 3}), v);
        assert_eq!((cut.seconds, cut.rate), (3.0, 1.0));
        // Seedance: 4k is nine times 720p's pixels, a draft is 480p, and an
        // edit bills the clip in and the clip out at 0.6.
        let sd = m("fal/seedance-2.5-reference");
        assert_eq!(quote_settings(sd, &json!({"resolution": "4k"}), v).tokens(), v.tokens() * 9.0);
        let d = quote_settings(sd, &json!({"resolution": "4k", "draft": true}), v);
        assert_eq!((d.w, d.h), (853, 480));
        let e = quote_settings(m("fal/seedance-2.5-video-edit"), &json!({}), v);
        assert!((e.seconds - 6.0).abs() < 1e-9);
    }

    #[test]
    fn seedance_text_to_video_needs_no_picture() {
        let sd = m("fal/seedance-2.5-t2v");
        let opts = crate::pipeline::Opts { seconds: 40, res: 1080, ..Default::default() };
        let v = request(sd, &opts, 7, (1080, 1920), Some("a jellyfish"), &[]).unwrap();
        assert_eq!(v["prompt"], "a jellyfish");
        assert_eq!((v["duration"].as_str(), v["resolution"].as_str()), (Some("30"), Some("1080p")));
        assert_eq!(v["aspect_ratio"], "9:16");
        assert!(v.get("image_url").is_none() && v.get("seed").is_none());
    }

    #[test]
    fn a_clip_keys_on_its_frames_not_on_its_file() {
        // Two encodes of the same frames are two different files and ONE
        // request: re-capturing a clip must find the result it paid for.
        let dir = std::env::temp_dir().join("wobble-genai-clipkey-none");
        let p = Pipeline::for_model("fal/kling-o3-pro-video-edit").unwrap();
        let key = frames_key(&[vec![1, 2, 3], vec![4, 5, 6]], 1280, 720, 24.0);
        assert_ne!(key, frames_key(&[vec![1, 2, 3], vec![4, 5, 7]], 1280, 720, 24.0));
        let enc = |bytes: Vec<u8>| Clip { bytes, w: 1280, h: 720, frames: 72, fps: 24.0, key: key.clone() };
        let mut a = Inputs { comp: (1920, 1080), ..Default::default() };
        a.clips.insert(0, enc(vec![1; 9]));
        let mut b = a.clone();
        b.clips.insert(0, enc(vec![2; 9]));
        let (ka, kb) = (plan(&p, &a, &dir).unwrap(), plan(&p, &b, &dir).unwrap());
        assert_eq!(ka.keys[ka.target], kb.keys[kb.target]);
    }

    #[test]
    fn a_clip_through_a_comp_node_is_still_the_model_s_clip() {
        // Input (clip) -> Crop -> Wan's video port: the model reads the
        // clip the host rendered for the Crop.
        let dir = std::env::temp_dir().join("wobble-genai-compclip-none");
        let mut p = Pipeline::bare(vec![crate::InKind::Image]);
        let img = p.input_step(0).unwrap();
        let ask = p.push(Op::Prompt { text: "bluer".into() }, [0.0, 0.0]);
        let crop = p.push(Op::Comp { kind: "Crop".into(), node: json!({}), clip: true, inputs: 1 }, [0.0, 0.0]);
        let wan = p.push(Op::Generate { model: "fal/wan-2.7-video-edit".into(), seed: 1, opts: Opts::default() }, [0.0, 0.0]);
        let out = p.push(Op::Output, [0.0, 0.0]);
        assert!(p.connect(img, crop, 0) && p.connect(ask, wan, 0) && p.connect(crop, wan, 1) && p.connect(wan, out, 0));
        let mut ins = Inputs { comp: (1920, 1080), ..Default::default() };
        assert!(plan(&p, &ins, &dir).is_err(), "nothing rendered yet");
        ins.shape_clips.insert(crop, clip(72, 1280, 720));
        let pl = plan(&p, &ins, &dir).unwrap();
        assert!(!pl.keys[wan].is_empty());
        // And its own limits are checked on THAT clip: one second is short.
        ins.shape_clips.insert(crop, clip(24, 1280, 720));
        assert!(plan(&p, &ins, &dir).unwrap_err().contains("2-10 s"));
    }

    #[test]
    fn ltx_models_ask_in_their_own_words_and_refuse_before_billing() {
        use crate::pipeline::Op;
        let o = Opts::default();
        let pic = Some("data:a".to_string());
        let vid = Some("data:video/mp4;base64,AAAA".to_string());
        let shape = ClipShape { frames: 72, fps: 24.0 };
        let with = |extra: &str| Opts { extra: extra.into(), ..Opts::default() };

        // Text to video: the comp's aspect, integer fields, no audio, no seed.
        let t = request(m("fal/ltx-2.3-t2v"), &o, 7, (1080, 1920), Some("p"), &[None]).unwrap();
        assert_eq!(t["aspect_ratio"], "9:16");
        assert_eq!((t["duration"].clone(), t["fps"].clone()), (json!(6), json!(25)));
        assert_eq!(t["generate_audio"], false);
        assert!(t.get("seed").is_none() && t.get("image_url").is_none(), "{t}");
        // Past 10 s only at 1080p and 25 fps.
        let fast = m("fal/ltx-2.3-t2v-fast");
        assert!(request(fast, &with(r#"{"duration":20}"#), 7, (1920, 1080), Some("p"), &[None]).is_ok());
        assert!(request(fast, &with(r#"{"duration":20,"fps":24}"#), 7, (1920, 1080), Some("p"), &[None]).is_err());
        assert!(request(fast, &with(r#"{"duration":12,"resolution":"2160p"}"#), 7, (1920, 1080), Some("p"), &[None]).is_err());

        // Image to video: start frame, an optional end frame on port 2.
        let i2v = m("fal/ltx-2.3-i2v");
        let a = request(i2v, &o, 7, (1920, 1080), Some("p"), &[None, pic.clone()]).unwrap();
        assert_eq!(a["image_url"], "data:a");
        assert_eq!(a["aspect_ratio"], "auto");
        assert!(a.get("end_image_url").is_none());
        let b = request(i2v, &o, 7, (1920, 1080), Some("p"), &[None, pic.clone(), Some("data:e".into())]).unwrap();
        assert_eq!(b["end_image_url"], "data:e");
        let names: Vec<_> = Op::Generate { model: i2v.id.into(), seed: 1, opts: Opts::default() }
            .ports()
            .iter()
            .map(|p| p.1)
            .collect();
        assert_eq!(names, ["prompt", "image", "end frame"]);

        // Clean plate: the clip's length and rate, and its time budget.
        let cp = m("fal/ltx-2.3-clean-plate");
        let c = request_clip(cp, &o, 7, (1920, 1080), Some(""), &[None, vid.clone()], Some(shape)).unwrap();
        assert_eq!((c["num_frames"].clone(), c["seed"].clone()), (json!(72), json!(7)));
        assert_eq!((c["resolution"].clone(), c["generate_audio"].clone()), (json!("720p"), json!(false)));
        let long = ClipShape { frames: 240, fps: 24.0 };
        assert!(request_clip(cp, &o, 7, (1920, 1080), Some(""), &[None, vid.clone()], Some(long)).is_err());
        assert!(request_clip(cp, &with(r#"{"resolution":"480p"}"#), 7, (1920, 1080), Some(""), &[None, vid.clone()], Some(long)).is_ok());

        // Render to real: the trigger word leads, once.
        let rr = m("fal/ltx-2.3-render-to-real");
        let r = request_clip(rr, &o, 7, (1920, 1080), Some("rainy street"), &[None, vid.clone(), pic.clone()], Some(shape)).unwrap();
        assert_eq!(r["prompt"], "3DREAL. rainy street");
        assert_eq!(r["image_url"], "data:a");
        assert_eq!(rr.dialect.instruct("3DREAL. x"), "3DREAL. x");

        // Retake: pictures only, and the window must lie inside the clip.
        let rt = m("fal/ltx-2.3-retake");
        let k = request_clip(rt, &with(r#"{"duration":2}"#), 7, (1920, 1080), Some("p"), &[None, vid.clone()], Some(shape)).unwrap();
        assert_eq!(k["retake_mode"], "replace_video");
        assert!(request_clip(rt, &o, 7, (1920, 1080), Some("p"), &[None, vid.clone()], Some(shape)).is_err(), "5 s of a 3 s clip");

        // The 2.5 tools read port 0 like an upscaler; the matte is its own field.
        let up = request(m("fal/ltx-2.5-upscale"), &o, 7, (1920, 1080), None, &[vid.clone()]).unwrap();
        assert_eq!(up, json!({"video_url": vid.clone().unwrap(), "resolution": "1080p"}));
        let rf = request(m("fal/ltx-2.5-reframe"), &with(r#"{"placement":{"anchor":"left"}}"#), 7, (1920, 1080), None, &[vid.clone()]).unwrap();
        assert_eq!(rf["placement"]["anchor"], "left");
        let al = m("fal/ltx-2.5-alpha");
        assert!(al.task.is_data() && al.task.reads_video());
        let out = json!({"matte_video": {"url": "https://x/m.mp4"}, "cutout_video": {"url": "https://x/c.mov"}});
        assert_eq!(service::output_video_at(&out, al.dialect.output_key(al.task)).as_deref(), Some("https://x/m.mp4"));
    }

    #[test]
    fn an_ltx_quote_bills_the_seconds_and_size_asked_for() {
        let v = VideoCall { seconds: 6.0, w: 1920, h: 1080, default_res: false, rate: 1.0 };
        let q = quote_settings(m("fal/ltx-2.3-t2v"), &json!({"duration": 10, "resolution": "2160p"}), v);
        assert_eq!((q.seconds, q.rate, q.default_res), (10.0, 4.0, true));
        let q = quote_settings(m("fal/ltx-2.3-i2v-fast"), &json!({"duration": 6, "resolution": "1080p"}), v);
        assert_eq!((q.seconds, q.rate), (6.0, 1.0));
        let q = quote_settings(m("fal/ltx-2.3-extend"), &json!({"duration": 3}), VideoCall { seconds: 9.0, ..v });
        assert_eq!(q.seconds, 3.0);
    }

    #[test]
    fn a_result_keeps_its_prompt_and_settings_beside_it_and_inside_its_png() {
        let dir = std::env::temp_dir().join(format!("wobble-genai-provenance-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let req = serde_json::json!({
            "prompt": "a moth on a lamp",
            "image_url": "data:image/png;base64,AAAA",
            "image_urls": ["data:image/jpeg;base64,BBBBBB"],
            "guidance_scale": 3.5,
        });
        let scrubbed = scrub_request(&req);
        assert_eq!(scrubbed["guidance_scale"], 3.5);
        assert_eq!(scrubbed["prompt"], "a moth on a lamp");
        assert!(scrubbed["image_url"].as_str().unwrap().starts_with("<image/png,"), "{scrubbed}");
        assert!(scrubbed["image_urls"][0].as_str().unwrap().starts_with("<image/jpeg,"));
        let png = image::png_srgb_rgb(&[0.25; 16], 2, 2).unwrap();
        let meta = Meta {
            file: "result.png".into(),
            model: "fal/flux-dev".into(),
            endpoint: "fal-ai/flux/dev".into(),
            service: "fal".into(),
            when: 7,
            prompt: Some("a moth on a lamp".into()),
            seed: Some(42),
            request: Some(scrubbed.clone()),
            ..meta0()
        };
        let path = commit(&dir, "k", "result.png", &png, &meta).unwrap();
        // Beside it: meta.json, whole.
        let back = cached(&dir, "k").unwrap();
        assert_eq!(back, meta);
        // Inside it: the prompt under a key image tools show, and the record.
        let text = image::png_text(&std::fs::read(&path).unwrap());
        assert!(text.contains(&("Description".into(), "a moth on a lamp".into())), "{text:?}");
        let rec = &text.iter().find(|(k, _)| k == "wobble-genai").unwrap().1;
        let rec: serde_json::Value = serde_json::from_str(rec).unwrap();
        assert_eq!(rec["seed"], 42);
        assert_eq!(rec["request"]["guidance_scale"], 3.5);
        // An input is the comp's own picture: left as it was.
        let input = commit(&dir, "in", "input.png", &png, &Meta { file: "input.png".into(), ..meta0() }).unwrap();
        assert_eq!(std::fs::read(input).unwrap(), png);
        // A meta.json from before the record existed still reads.
        let old: Meta = serde_json::from_str(r#"{"file":"result.png","model":"m"}"#).unwrap();
        assert_eq!(old.prompt, None);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_local_ltx_step_sends_paths_and_is_neither_quoted_nor_billed() {
        let m = catalog::get("local/ltx-2.5-distilled").unwrap();
        let mut o = crate::pipeline::Opts::default();
        o.seconds = 30;
        o.res = 720;
        // Text-to-video: no start frame.
        let r = request(m, &o, 7, (1920, 1080), Some("a cat"), &[None, None]).unwrap();
        assert_eq!(r["image"], Value::Null);
        assert_eq!(r["seconds"], 10, "clamped to the model's longest");
        assert_eq!((r["width"].as_u64(), r["height"].as_u64()), (Some(1280), Some(720)));
        assert_eq!(r["seed"], 7);
        let r = request(m, &o, 7, (1920, 1080), Some("a cat"), &[None, Some("/tmp/f.png".into())]).unwrap();
        assert_eq!(r["image"], "/tmp/f.png");
        let gen = Op::Generate { model: m.id.into(), seed: 1, opts: o };
        assert!(!gen.is_remote());
        assert!(gen.optional(1), "no start frame is text-to-video");
        assert!(!gen.optional(0));
    }

}
