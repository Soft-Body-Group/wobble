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

/// What the host hands a run: its inputs, rendered, and the comp size.
#[derive(Debug, Clone, Default)]
pub struct Inputs {
    pub reference: Option<Png>,
    pub matte: Option<Png>,
    /// The text wired into the host's prompt port, for a `prompt in` step.
    pub prompt: Option<String>,
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
            // The same recipe as a typed prompt: the key is the TEXT that
            // reaches the request, never where it came from.
            Op::PromptIn => {
                let t = inputs.prompt.as_ref().ok_or(
                    "the pipeline reads the node's prompt, and no String node is wired into it",
                )?;
                (h(&["prompt", t]), 0.0)
            }
            Op::Reference => {
                let r = inputs.reference.as_ref().ok_or(
                    "the pipeline reads the node's input, and nothing is wired into it",
                )?;
                (h(&["reference", &sha_hex(&r.bytes)]), r.w as f64 * r.h as f64 / 1e6)
            }
            Op::Matte => {
                let r = inputs.matte.as_ref().ok_or(
                    "the pipeline reads the node's matte, and nothing is wired into it",
                )?;
                (h(&["matte", &sha_hex(&r.bytes)]), r.w as f64 * r.h as f64 / 1e6)
            }
            Op::Output => (keys[step.inputs[0].unwrap()].clone(), in_mp(0).unwrap_or(0.0)),
            Op::Generate { model, seed, opts } => {
                let m = catalog::get(model).ok_or_else(|| format!("unknown model {model}"))?;
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
                    Task::Cutout => in_mp(0).unwrap_or(comp_mp),
                    Task::Edit | Task::Inpaint => in_mp(1).unwrap_or(comp_mp),
                    Task::ImageToVideo => {
                        let (w, h) = video_size(m, opts, inputs.comp);
                        w as f64 * h as f64 / 1e6
                    }
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
    let Dialect::Seedance { max_seconds, top_res } = m.dialect else { return None };
    let res = [480, 720, 1080]
        .into_iter()
        .filter(|&r| r <= top_res)
        .min_by_key(|&r| r.abs_diff(opts.res))
        .unwrap_or(720);
    Some((opts.seconds.clamp(4, max_seconds), res))
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
    let img = |k: usize| -> Result<Value> {
        images
            .get(k)
            .cloned()
            .flatten()
            .map(Value::String)
            .ok_or_else(|| anyhow!("{} needs an image", m.label))
    };
    let prompt = || json!(prompt.unwrap_or(""));
    let mut v = match (m.task, m.dialect) {
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
        (Task::Edit, Dialect::Seedream) => json!({
            "prompt": prompt(), "image_urls": [img(1)?],
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
    Ok(v)
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
    let reqs: Vec<(String, Value, f64, Option<VideoCall>)> = billed
        .iter()
        .map(|&i| {
            let Op::Generate { model, seed, opts } = &p.steps[i].op else { unreachable!() };
            let m = catalog::get(model).unwrap();
            let ph: Vec<Option<String>> =
                p.steps[i].inputs.iter().map(|s| s.map(|_| "<image>".to_string())).collect();
            let r = request(m, opts, *seed, inputs.comp, Some("<prompt>"), &ph)?;
            let video = video_opts(m, opts).map(|(secs, res)| {
                let (w, h) = video_size(m, opts, inputs.comp);
                VideoCall { seconds: secs as f64, w, h, default_res: res == 720 }
            });
            Ok((m.endpoint.to_string(), r, plan.mp[i], video))
        })
        .collect::<Result<_>>()?;
    let items: Vec<QuoteItem> = reqs
        .iter()
        .map(|(e, r, mp, video)| QuoteItem {
            endpoint: e,
            input: r,
            megapixels: (video.is_none()).then_some(*mp),
            video: *video,
        })
        .collect();
    svc.quote(http, &items)
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
    let (tx, rx) = channel();
    let cancel = Arc::new(AtomicBool::new(false));
    let stop = cancel.clone();
    let fail = tx.clone();
    let spawned = std::thread::Builder::new().name("genai-run".into()).spawn(move || {
        let r = execute(&p, &plan, &inputs, &svcs, &dir, &stop, &tx, &approved);
        let _ = tx.send(match r {
            Ok(t) => Event::Done(t),
            Err(e) => Event::Failed(format!("{e:#}")),
        });
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
    std::fs::write(tmp.join(file), bytes)?;
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
            Op::PromptIn => Val::Text(
                inputs.prompt.clone().ok_or_else(|| anyhow!("no String node is wired into the node"))?,
            ),
            Op::Reference | Op::Matte => {
                let png = if matches!(p.steps[i].op, Op::Reference) {
                    inputs.reference.as_ref()
                } else {
                    inputs.matte.as_ref()
                };
                let png = png.ok_or_else(|| anyhow!("missing input picture"))?;
                let path = match cached(dir, key) {
                    Some(m) => dir.join(key).join(m.file),
                    None => commit(
                        dir,
                        key,
                        "input.png",
                        &png.bytes,
                        &Meta { file: "input.png".into(), when: now(), ..meta0() },
                    )?,
                };
                Val::Image(path)
            }
            Op::Output => get(&vals, i, 0).ok_or_else(|| anyhow!("Output has no input"))?,
            Op::Generate { model, seed, opts } => {
                let m = catalog::get(model).ok_or_else(|| anyhow!("unknown model {model}"))?;
                if !labels.iter().any(|l| l == m.short) {
                    labels.push(m.short.to_string());
                }
                if let Some(meta) = cached(dir, key) {
                    let _ = tx.send(Event::Step(i, StepState::Cached));
                    Val::Image(dir.join(key).join(meta.file))
                } else {
                    let svc = svcs.remote.as_deref().ok_or_else(|| {
                        anyhow!("no way to bill this step: add a relay token or a fal key")
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
                    let req = request(m, opts, *seed, inputs.comp, prompt.as_deref(), &images)?;
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
                    let (url, nsfw) = if m.task.is_video() {
                        let url = service::output_video(&ran.output)
                            .ok_or_else(|| anyhow!("{} returned no video", m.label))?;
                        (url, false)
                    } else {
                        service::output_image(&ran.output)
                            .ok_or_else(|| anyhow!("{} returned no image", m.label))?
                    };
                    let _ = tx.send(Event::Step(i, StepState::Fetching));
                    let (bytes, ct) = web.download(&url)?;
                    let file = format!("result.{}", image::ext_for(ct.as_deref(), &url));
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
        Op::PromptIn => inputs.prompt.as_ref().filter(|t| !t.is_empty()).map(|t| t.chars().take(40).collect()),
        _ => None,
    });
    let mut label = labels.join(" → ");
    if let Some(t) = prompt {
        label = format!("{label}: {t}");
    }
    Ok(Take { key, file, tick: inputs.tick, commercial: plan.commercial, usd: charged, label, when: now() })
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
        let ins = Inputs {
            reference: Some(Png { bytes: vec![1, 2, 3], w: 4, h: 4 }),
            matte: None,
            prompt: None,
            comp: (2048, 1080),
            tick: 0,
        };
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
        let r = p.push(Op::Reference, [0.0, 0.0]);
        assert!(p.connect(r, 1, 1));
        let k = key(&p);
        set(&mut p, 10);
        assert_ne!(key(&p), k);
    }

    #[test]
    fn a_wired_prompt_keys_like_the_same_typed_prompt() {
        let dir = std::env::temp_dir().join("wobble-genai-keys-none");
        let mut ins = Inputs { prompt: Some("a moth".into()), comp: (2048, 1080), ..Default::default() };
        let mut typed = Pipeline::default();
        if let Op::Prompt { text } = &mut typed.steps[0].op {
            *text = "a moth".into();
        }
        let mut wired = Pipeline::default();
        wired.steps[0].op = Op::PromptIn;
        let k = |p: &Pipeline, ins: &Inputs| plan(p, ins, &dir).unwrap().keys[1].clone();
        assert_eq!(k(&wired, &ins), k(&typed, &ins), "moving a prompt into the graph re-bills nothing");
        let before = k(&wired, &ins);
        ins.prompt = Some("a lamp".into());
        assert_ne!(k(&wired, &ins), before, "the wired text is in the key");
        ins.prompt = None;
        assert!(plan(&wired, &ins, &dir).unwrap_err().contains("no String node"));
    }
}
