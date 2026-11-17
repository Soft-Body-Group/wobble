//! The step graph a GenAI node carries.
//!
//! A pipeline is DATA on the node, never graph nodes: a handful of steps
//! (write a prompt, take the node's input as a reference, generate, upscale,
//! cut a matte…) wired into one `Output`. The app draws it in place inside
//! the host node (`innards`), the runner walks it, and the engine never sees
//! it at all — it plays back the picked TAKE through a reader slot, exactly
//! like a bake.
//!
//! Two rules the rest of the crate leans on:
//!
//! - **Ports are typed, and `connect` is the only wiring rule.** A step
//!   declares its input and output [`Pipe`]s; a wire is only ever made from
//!   an output to an input of the same pipe. The inner graph is small, but a
//!   prompt landing in an image port is a request the provider rejects after
//!   you have already been billed for the steps above it.
//! - **Indices are ids.** A step is addressed by its position in `steps`,
//!   and every `inputs` entry names one. Removing a step slides every later
//!   index down (`remove`), which is the whole cost of not carrying ids.

use serde::{Deserialize, Serialize};

use crate::catalog::{self, Task};

/// What travels along an inner wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pipe {
    Text,
    Image,
    /// A clip: a video step's result, or the host's input over a frame
    /// range (`ClipIn`, `MatteClip`). Only Output and a video model's
    /// video ports take one: an image port fed a movie would be a request
    /// the provider rejects after the steps above it were billed.
    Video,
    /// A 3D capture (a Gaussian splat file). Only Output takes one.
    Scene,
}

/// One step of a pipeline.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Step {
    pub op: Op,
    /// Where it sits in the host node's inner view, in the host's [-1, 1]
    /// square (the `innards` convention).
    pub at: [f32; 2],
    /// Per input port, the step feeding it. Always as long as the op has
    /// ports (`ports()`); `sync_ports` keeps it so after a model change.
    #[serde(default)]
    pub inputs: Vec<Option<usize>>,
}

/// What one of the host node's inputs carries. The node has as many as
/// its pipeline lists (`Pipeline::ins`), and each one appears INSIDE as
/// exactly one `Op::Input` step that cannot be deleted or added by hand.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InKind {
    /// A picture, sent as sRGB.
    Image,
    /// Coverage: white = where to work.
    Matte,
}

/// What a step does.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Op {
    /// One of the host node's inputs (`input` indexes `Pipeline::ins`),
    /// rendered at the playhead when the run starts, or over a frame range
    /// from it when `clip` (sent as an mp4: what a video model reads).
    /// One per input, made and removed by `Pipeline::sync_inputs` only.
    /// A picture leaves the comp as display-referred sRGB (see `image`);
    /// a matte as coverage.
    Input {
        input: usize,
        /// Mirrors `Pipeline::ins[input]` (kept so by `sync_inputs`): the
        /// step's name and its capture need it without the pipeline.
        #[serde(default)]
        matte: bool,
        /// Which image (or matte) this is, counting from 0 — "image 2".
        #[serde(default)]
        nth: usize,
        #[serde(default)]
        clip: bool,
        /// A clip's length, in seconds of comp time.
        #[serde(default = "default_clip_seconds")]
        seconds: f32,
    },
    /// Text typed into the panel.
    Prompt { text: String },
    /// A matte DRAWN inside the pipeline: a comp node (a Spline) the host
    /// app owns, carried here whole as its own JSON (`node`; this crate
    /// never reads it). Rendered by the host when the run starts, like an
    /// Input — at the playhead, or over the run's range when `clip` (set
    /// by the host from what it is wired into: a video port wants a clip).
    Shape {
        node: serde_json::Value,
        #[serde(default)]
        clip: bool,
    },
    /// A comp node that WORKS ON a picture inside the pipeline (a
    /// Transform, Reformat, Crop, Freeze), carried whole like a `Shape`.
    /// Its one port takes what the host can render: an Input, a Shape or
    /// another Comp (never a model's result, which only exists mid-run).
    /// The host renders it when the run starts, through stand-ins of the
    /// whole chain above it. `kind` is the node's type name, for the
    /// label; `clip` is set by the host from what it feeds.
    Comp {
        kind: String,
        node: serde_json::Value,
        #[serde(default)]
        clip: bool,
    },
    /// LEGACY (read from old files, converted by `Pipeline::normalize`
    /// into an `Input`, never run): the host node's image input.
    Reference,
    /// LEGACY: the host node's old text input. The node has no prompt
    /// input any more (prompts live inside); `normalize` makes this a
    /// Prompt, and the app's setup migration copies in the text that was
    /// wired.
    PromptIn,
    /// LEGACY: the host node's matte input (the first matte).
    Matte,
    /// LEGACY: the host node's second picture input (the second image).
    Second,
    /// LEGACY: the first image input as a clip.
    ClipIn {
        /// Length of the clip, in seconds of comp time.
        #[serde(default = "default_clip_seconds")]
        seconds: f32,
    },
    /// LEGACY: the first matte input as a clip.
    MatteClip,
    /// One call to a remote model from the catalog, by stable id.
    Generate {
        model: String,
        /// Authored, not rolled per run: the same pipeline with the same
        /// seed is the same request, which is what makes a re-open (or an
        /// accidental second click) a cache hit instead of a second bill.
        seed: u32,
        #[serde(default)]
        opts: Opts,
    },
    /// A ComfyUI workflow exported in API format. The server comes from
    /// the app's config, never from the setup (a comp names what to run,
    /// the machine decides where).
    Comfy {
        /// Path to the workflow JSON ("Save (API format)" in ComfyUI).
        workflow: String,
        seed: u32,
        /// The user's own word that the checkpoints this workflow loads are
        /// free for commercial use. We cannot read a licence out of a
        /// ComfyUI graph, so the default is the cautious one (rule 10:
        /// tagged, never refused).
        #[serde(default)]
        commercial: bool,
    },
    /// Text with VARIABLES, filled when the run starts: `{a}` and `{b}`
    /// are the texts wired into its two ports (either may be left
    /// unwired, and reads as nothing), `{frame}` the frame the run is
    /// made at. Local and free: how a wired String, a typed style and a
    /// model's description become one prompt.
    Template { text: String },
    /// Where the node's picture comes from. One per pipeline.
    Output,
}

/// Per-call options a Generate step carries beside its model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Opts {
    /// Output size for a text-to-image model. `None` = fit the comp's
    /// aspect into `megapixels`.
    #[serde(default)]
    pub size: Option<(u32, u32)>,
    /// Pixel budget when the size is inherited from the comp.
    #[serde(default = "default_mp")]
    pub megapixels: f32,
    /// Upscale factor (upscalers only).
    #[serde(default = "default_scale")]
    pub scale: f32,
    /// A JSON object merged over the request the step builds, for fields
    /// this build does not model. Text, so an invalid edit is kept and
    /// reported rather than silently dropped.
    #[serde(default)]
    pub extra: String,
    /// Video length in seconds (image-to-video only).
    #[serde(default = "default_seconds")]
    pub seconds: u32,
    /// Video height class: 480, 720, 1080 (image-to-video only).
    #[serde(default = "default_res")]
    pub res: u32,
}

fn default_seconds() -> u32 {
    5
}
fn default_clip_seconds() -> f32 {
    5.0
}
fn default_res() -> u32 {
    720
}

fn default_mp() -> f32 {
    1.0
}
fn default_scale() -> f32 {
    2.0
}

impl Default for Opts {
    fn default() -> Self {
        Self {
            size: None,
            megapixels: default_mp(),
            scale: default_scale(),
            extra: String::new(),
            seconds: default_seconds(),
            res: default_res(),
        }
    }
}

impl Op {
    /// Input ports, in order, with a short name for each.
    pub fn ports(&self) -> Vec<(Pipe, &'static str)> {
        match self {
            Op::Prompt { .. }
            | Op::Input { .. }
            | Op::Shape { .. }
            | Op::PromptIn
            | Op::Reference
            | Op::Matte
            | Op::Second
            | Op::ClipIn { .. }
            | Op::MatteClip => vec![],
            Op::Output | Op::Comp { .. } => vec![(Pipe::Image, "image")],
            Op::Template { .. } => vec![(Pipe::Text, "a"), (Pipe::Text, "b")],
            Op::Comfy { .. } => {
                vec![(Pipe::Text, "prompt"), (Pipe::Image, "image"), (Pipe::Image, "mask")]
            }
            Op::Generate { model, .. } => match catalog::get(model) {
                Some(m) => task_ports(m.task),
                // An id this build does not know keeps no ports, so its
                // wires are dropped on load rather than guessed at; the
                // panel says what it was.
                None => vec![],
            },
        }
    }

    /// The output pipe, or `None` for the sink.
    pub fn output(&self) -> Option<Pipe> {
        match self {
            Op::Prompt { .. } | Op::PromptIn | Op::Template { .. } => Some(Pipe::Text),
            Op::Generate { model, .. } if catalog::get(model).is_some_and(|m| m.task.is_text()) => {
                Some(Pipe::Text)
            }
            Op::Output => None,
            Op::ClipIn { .. } | Op::MatteClip => Some(Pipe::Video),
            Op::Input { clip: true, .. } | Op::Shape { clip: true, .. } | Op::Comp { clip: true, .. } => {
                Some(Pipe::Video)
            }
            Op::Generate { model, .. } if catalog::get(model).is_some_and(|m| m.task.is_3d()) => {
                Some(Pipe::Scene)
            }
            Op::Generate { model, .. }
                if catalog::get(model).is_some_and(|m| m.task.is_video()) =>
            {
                Some(Pipe::Video)
            }
            _ => Some(Pipe::Image),
        }
    }

    /// May a step whose output is `pipe` feed input `port`? The same pipe,
    /// or a video into Output (the node's picture may be a movie).
    pub fn accepts(&self, port: usize, pipe: Pipe) -> bool {
        // A Comp reads the host's render of what is above it, still or
        // clip alike (whether IT is a clip is its own flag).
        if matches!(self, Op::Comp { .. }) {
            return port == 0 && matches!(pipe, Pipe::Image | Pipe::Video);
        }
        match self.ports().get(port) {
            Some(&(want, _)) => {
                want == pipe || (matches!(self, Op::Output) && matches!(pipe, Pipe::Video | Pipe::Scene))
            }
            None => false,
        }
    }

    /// Does running this step send something off the machine?
    pub fn is_remote(&self) -> bool {
        matches!(self, Op::Generate { .. })
    }

    /// What a step is called as a node in the pipeline's dive (and in its
    /// add menu): its kind, or the model's own name.
    pub fn name(&self) -> &'static str {
        match self {
            Op::Prompt { .. } => "Prompt",
            Op::Input { matte: false, clip: false, .. } => "Image In",
            Op::Input { matte: true, clip: false, .. } => "Matte In",
            Op::Input { matte: false, clip: true, .. } => "Image Clip In",
            Op::Input { matte: true, clip: true, .. } => "Matte Clip In",
            Op::Shape { .. } => "Spline",
            Op::Comp { kind, .. } => match kind.as_str() {
                "Transform" => "Transform",
                "Reformat" => "Reformat",
                "Crop" => "Crop",
                "Freeze" => "Freeze",
                _ => "Comp Node",
            },
            Op::PromptIn => "Prompt In",
            Op::Reference => "Input",
            Op::Matte => "Matte",
            Op::Second => "Second Input",
            Op::ClipIn { .. } => "Input Clip",
            Op::MatteClip => "Matte Clip",
            Op::Generate { model, .. } => catalog::get(model).map(|m| m.label).unwrap_or("Unknown Model"),
            Op::Comfy { .. } => "ComfyUI",
            Op::Template { .. } => "Text Template",
            Op::Output => "Output",
        }
    }

    /// Short label for the inner node.
    pub fn label(&self) -> String {
        match self {
            Op::Prompt { text } => {
                let t: String = text.chars().take(14).collect();
                if t.is_empty() {
                    "prompt".into()
                } else if text.chars().count() > 14 {
                    format!("{t}…")
                } else {
                    t
                }
            }
            Op::Input { matte, nth, clip, seconds, .. } => {
                let what = if *matte { "matte" } else { "image" };
                match (*nth, *clip) {
                    (0, false) => what.into(),
                    (n, false) => format!("{what} {}", n + 1),
                    (0, true) => format!("{what} clip {seconds:.1}s"),
                    (n, true) => format!("{what} {} clip {seconds:.1}s", n + 1),
                }
            }
            Op::Shape { clip: false, .. } => "spline".into(),
            Op::Shape { clip: true, .. } => "spline clip".into(),
            Op::Comp { kind, .. } => kind.to_lowercase(),
            Op::PromptIn => "prompt in".into(),
            Op::Reference => "input".into(),
            Op::Matte => "matte".into(),
            Op::Second => "input 2".into(),
            Op::ClipIn { seconds } => format!("clip {seconds:.1}s"),
            Op::MatteClip => "matte clip".into(),
            Op::Generate { model, .. } => match catalog::get(model) {
                Some(m) => m.short.into(),
                None => "unknown".into(),
            },
            Op::Comfy { .. } => "comfy".into(),
            Op::Template { .. } => "template".into(),
            Op::Output => "out".into(),
        }
    }
}

/// The ports a remote model's task takes.
pub fn task_ports(task: Task) -> Vec<(Pipe, &'static str)> {
    match task {
        Task::TextToImage => vec![(Pipe::Text, "prompt")],
        Task::Edit => vec![(Pipe::Text, "prompt"), (Pipe::Image, "image")],
        Task::Inpaint => {
            vec![(Pipe::Text, "prompt"), (Pipe::Image, "image"), (Pipe::Image, "mask")]
        }
        Task::Upscale | Task::Cutout | Task::Depth | Task::Normals | Task::Albedo => {
            vec![(Pipe::Image, "image")]
        }
        Task::ImageToVideo | Task::Relight => vec![(Pipe::Text, "prompt"), (Pipe::Image, "image")],
        Task::VideoEdit => vec![(Pipe::Text, "prompt"), (Pipe::Video, "video")],
        Task::VideoInpaint => {
            vec![(Pipe::Text, "prompt"), (Pipe::Video, "video"), (Pipe::Video, "mask")]
        }
        Task::VideoUpscale => vec![(Pipe::Video, "video")],
        Task::Outpaint | Task::ImageTo3D | Task::Describe => {
            vec![(Pipe::Text, "prompt"), (Pipe::Image, "image")]
        }
        Task::Enhance => vec![(Pipe::Text, "prompt")],
        Task::EditRef => {
            vec![(Pipe::Text, "prompt"), (Pipe::Image, "image"), (Pipe::Image, "reference")]
        }
    }
}

/// One run's result: a file in the results directory, and what made it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Take {
    /// Cache key of the step that fed Output — also its directory name.
    pub key: String,
    /// File name inside that directory.
    pub file: String,
    /// The playhead the reference was captured at (the tick the take was
    /// made FOR — a still holds across the whole range).
    pub tick: i64,
    /// Were all the models that made these pixels free for commercial use,
    /// when they were made? Recorded, not recomputed: the pixels do not
    /// change licence because the pipeline was edited afterwards.
    pub commercial: bool,
    /// What the run was charged or quoted, in USD, if anyone said.
    #[serde(default)]
    pub usd: Option<f64>,
    /// Human summary: model + prompt.
    #[serde(default)]
    pub label: String,
    /// Unix seconds.
    #[serde(default)]
    pub when: u64,
    /// MEASUREMENTS, not a picture (a depth or normals model made it): the
    /// node plays the file back as stored values, with no sRGB transform.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub data: bool,
}

impl Take {
    /// The picture on disk, under a results directory.
    pub fn path(&self, dir: &std::path::Path) -> std::path::PathBuf {
        dir.join(&self.key).join(&self.file)
    }

    /// A movie rather than a still: it plays from `tick` on, instead of
    /// holding across the whole range. Read off the file, so takes saved
    /// before video existed need nothing.
    pub fn is_video(&self) -> bool {
        is_video_file(&self.file)
    }

    /// A 3D capture rather than a picture: a Splat node shows it, the
    /// GenAI node that made it passes its input through.
    pub fn is_splat(&self) -> bool {
        is_splat_file(&self.file)
    }
}

/// The reference cap for a pipeline with an image upscaler in it: enough
/// for a 2K DCI or UHD-half plate to go whole (2048x1080 = 2.2 MP,
/// 2560x1440 = 3.7 MP).
pub const UPSCALE_REF_MP: f32 = 4.2;

/// The extensions a video step's result can come back as.
pub const VIDEO_EXTS: &[&str] = &["mp4", "mov", "webm"];

/// A Gaussian splat capture, by its name.
pub fn is_splat_file(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    n.ends_with(".ply") || n.ends_with(".spz")
}

pub fn is_video_file(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    VIDEO_EXTS.iter().any(|e| n.ends_with(&format!(".{e}")))
}

/// The whole of a GenAI node's authored state.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Pipeline {
    pub steps: Vec<Step>,
    /// Every run that landed, oldest first. Kept so a variation never
    /// overwrites the one you liked.
    #[serde(default)]
    pub takes: Vec<Take>,
    /// Which take the node shows. Out of range = none.
    #[serde(default)]
    pub take: usize,
    /// The host node's inputs, in node-input order: what each carries.
    /// The node's bar draws the images on the left and the mattes on the
    /// right, but an input's INDEX is the order it was added in, so adding
    /// one never renumbers the wires already there. Empty only in a file
    /// written before the node had a bar (`normalize` fills it: the old
    /// image, matte, prompt and second-picture inputs).
    #[serde(default)]
    pub ins: Vec<InKind>,
}

/// What `normalize` found in a pipeline from before the input bar: where
/// the old fixed inputs went, so the app can move their wires.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Legacy {
    /// The node input the old SECOND picture (old input 3) now is, if the
    /// pipeline read it.
    pub second: Option<usize>,
}

impl Default for Pipeline {
    /// Prompt -> text-to-image -> Output: the shortest thing that makes a
    /// picture.
    fn default() -> Self {
        // The Inputs go after (`sync_inputs` below): step 0 is the prompt,
        // 1 the model, 2 Output, which a lot of code and tests lean on.
        let mut p = Pipeline { steps: vec![], takes: vec![], take: 0, ins: vec![InKind::Image, InKind::Matte] };
        let prompt = p.push(Op::Prompt { text: String::new() }, [-0.55, -0.45]);
        let gen = p.push(
            Op::Generate { model: catalog::DEFAULT_MODEL.into(), seed: 1, opts: Opts::default() },
            [0.0, 0.0],
        );
        let out = p.push(Op::Output, [0.0, 0.55]);
        p.connect(prompt, gen, 0);
        p.connect(gen, out, 0);
        p.sync_inputs();
        p
    }
}

impl Pipeline {
    /// The fixed pipeline of a one-model node: the model's inputs, each fed
    /// by the source its port names (a typed Prompt, the node's image, the
    /// node's matte), into the model, into Output. `None` for an id this
    /// build does not know.
    pub fn for_model(id: &str) -> Option<Self> {
        // A text model makes no picture: it is a step, not a node.
        let m = catalog::get(id).filter(|m| !m.task.is_text())?;
        // The first image is always there (an unloaded node passes it
        // through); the rest are what the model's ports name.
        let mut p = Pipeline::bare(vec![InKind::Image]);
        let gen = p.push(Op::Generate { model: m.id.into(), seed: 1, opts: Opts::default() }, [0.0, 0.0]);
        let ports = task_ports(m.task);
        let n = ports.len().max(1) as f32;
        let mut images = 0;
        for (port, (pipe, name)) in ports.into_iter().enumerate() {
            let x = -0.6 + 1.2 * (port as f32 + 0.5) / n;
            let clip = pipe == Pipe::Video;
            let kind = if name == "mask" { InKind::Matte } else { InKind::Image };
            let src = match (pipe, kind) {
                (Pipe::Text, _) => p.push(Op::Prompt { text: String::new() }, [x, -0.55]),
                (_, InKind::Matte) => {
                    let k = p.add_input(InKind::Matte);
                    p.input_step(k).unwrap()
                }
                (_, InKind::Image) => {
                    let k = if images == 0 { 0 } else { p.add_input(InKind::Image) };
                    images += 1;
                    p.input_step(k).unwrap()
                }
            };
            if let Op::Input { clip: c, .. } = &mut p.steps[src].op {
                *c = clip;
            }
            if pipe != Pipe::Text {
                p.steps[src].at = [x, -0.55];
            }
            p.connect(src, gen, port);
        }
        let out = p.push(Op::Output, [0.0, 0.55]);
        p.connect(gen, out, 0);
        Some(p)
    }

    /// No steps but one Input per input in `ins`.
    pub fn bare(ins: Vec<InKind>) -> Self {
        let mut p = Pipeline { steps: vec![], takes: vec![], take: 0, ins };
        p.sync_inputs();
        p
    }

    /// The Input step of node input `k`.
    pub fn input_step(&self, k: usize) -> Option<usize> {
        self.steps.iter().position(|s| matches!(s.op, Op::Input { input, .. } if input == k))
    }

    /// Add a node input of `kind` at the END (no wire already there is
    /// renumbered), with its Input step. Returns its index.
    pub fn add_input(&mut self, kind: InKind) -> usize {
        self.ins.push(kind);
        self.sync_inputs();
        self.ins.len() - 1
    }

    /// Remove node input `k` and its Input step (wires out of it drop).
    /// Later inputs slide down by one — the caller moves the host's wires
    /// to match. Refused for the last image: the node passes its first
    /// image through until something is generated.
    pub fn remove_input(&mut self, k: usize) -> bool {
        if k >= self.ins.len() {
            return false;
        }
        if self.ins[k] == InKind::Image && self.ins.iter().filter(|i| **i == InKind::Image).count() <= 1 {
            return false;
        }
        if let Some(i) = self.input_step(k) {
            self.remove_step(i);
        }
        self.ins.remove(k);
        for s in &mut self.steps {
            if let Op::Input { input, .. } = &mut s.op {
                if *input > k {
                    *input -= 1;
                }
            }
        }
        self.sync_inputs();
        true
    }

    /// Exactly one Input step per node input, each naming its own kind and
    /// count; any for an input that no longer exists removed, a duplicate
    /// (a paste) folded into the first, a missing one added along the top.
    pub fn sync_inputs(&mut self) {
        let mut i = 0;
        while i < self.steps.len() {
            if let Op::Input { input, .. } = self.steps[i].op {
                let first = self.input_step(input);
                if input >= self.ins.len() {
                    self.remove_step(i);
                    continue;
                }
                if let Some(f) = first.filter(|&f| f != i) {
                    // Its consumers read the first one instead.
                    for s in &mut self.steps {
                        for inp in &mut s.inputs {
                            if *inp == Some(i) {
                                *inp = Some(f);
                            }
                        }
                    }
                    self.remove_step(i);
                    continue;
                }
            }
            i += 1;
        }
        // Heal: an Input an earlier build dropped at the face's spot in a
        // pipeline already in graph units (hidden behind the middle node)
        // is placed again below.
        let graph_units = self.steps.iter().any(|s| s.at[0].abs() > 1.0 || s.at[1].abs() > 1.0);
        let mut lost = vec![];
        if graph_units {
            let mut i = 0;
            while i < self.steps.len() {
                let s = &self.steps[i];
                if matches!(s.op, Op::Input { .. }) && s.at[0].abs() <= 1.0 && s.at[1].abs() <= 1.0 {
                    let consumers: Vec<(usize, usize)> = self
                        .steps
                        .iter()
                        .enumerate()
                        .flat_map(|(j, t)| t.inputs.iter().enumerate().filter(move |(_, x)| **x == Some(i)).map(move |(p, _)| (j, p)))
                        .collect();
                    if consumers.is_empty() {
                        lost.push(self.steps[i].op.clone());
                        self.remove_step(i);
                        continue;
                    }
                }
                i += 1;
            }
        }
        let n = self.ins.len();
        for k in 0..n {
            let kind = self.ins[k];
            let nth = self.ins[..k].iter().filter(|x| **x == kind).count();
            match self.input_step(k) {
                Some(i) => {
                    if let Op::Input { matte, nth: m, .. } = &mut self.steps[i].op {
                        *matte = kind == InKind::Matte;
                        *m = nth;
                    }
                }
                None => {
                    // In the face's [-1, 1] square: along the top. Once
                    // the pipeline has been opened its steps sit in graph
                    // units, where that spot is ON the middle of the graph
                    // (a new Input hid behind the model there): then it
                    // goes beside the other Inputs, right of the last one.
                    let face = self.steps.iter().all(|s| s.at[0].abs() <= 1.0 && s.at[1].abs() <= 1.0);
                    let at = if face {
                        [-0.8 + 1.6 * (k as f32 + 0.5) / n as f32, -0.85]
                    } else {
                        let ins = self.steps.iter().filter(|s| matches!(s.op, Op::Input { .. }));
                        let right = ins.clone().map(|s| s.at[0]).fold(f32::MIN, f32::max);
                        let top = self.steps.iter().map(|s| s.at[1]).fold(f32::MAX, f32::min);
                        match ins.clone().next() {
                            Some(_) => [right + 200.0, ins.map(|s| s.at[1]).fold(f32::MAX, f32::min)],
                            None => [0.0, top - 200.0],
                        }
                    };
                    let _ = n;
                    // A healed Input keeps its clip setting.
                    if let Some(op) = lost.iter().find(|o| matches!(o, Op::Input { input, .. } if *input == k)) {
                        self.push(op.clone(), at);
                        continue;
                    }
                    self.push(
                        Op::Input {
                            input: k,
                            matte: kind == InKind::Matte,
                            nth,
                            clip: false,
                            seconds: default_clip_seconds(),
                        },
                        at,
                    );
                }
            }
        }
    }

    /// Bring a pipeline from any older shape to this one: fill `ins` for a
    /// file from before the input bar (the old image and matte inputs),
    /// turn the old source steps (Reference, Matte, Second, ClipIn,
    /// MatteClip — and a built-in template's) into the Input of the input
    /// they meant, adding inputs they need, and a PromptIn into an empty
    /// Prompt. An Input that names the wrong input (a template made on
    /// another node) is re-pointed at the input of its kind and count.
    /// Idempotent. `Some` when the pipeline was from before the bar.
    pub fn normalize(&mut self) -> Option<Legacy> {
        let legacy = self.ins.is_empty();
        if legacy {
            self.ins = vec![InKind::Image, InKind::Matte];
        }
        let mut found = Legacy::default();
        // (step, kind, nth, clip, seconds) for every source to re-point.
        let mut todo = vec![];
        for (i, s) in self.steps.iter().enumerate() {
            let t = match &s.op {
                Op::Reference => (InKind::Image, 0, false, default_clip_seconds()),
                Op::Second => (InKind::Image, 1, false, default_clip_seconds()),
                Op::Matte => (InKind::Matte, 0, false, default_clip_seconds()),
                Op::ClipIn { seconds } => (InKind::Image, 0, true, *seconds),
                Op::MatteClip => (InKind::Matte, 0, true, default_clip_seconds()),
                Op::Input { input, matte, nth, clip, seconds } => {
                    let kind = if *matte { InKind::Matte } else { InKind::Image };
                    if self.nth_input(kind, *nth) == Some(*input) {
                        continue;
                    }
                    (kind, *nth, *clip, *seconds)
                }
                _ => continue,
            };
            todo.push((i, t));
        }
        // Inputs already spoken for, and as what (still or clip): a still
        // and a clip of the same old input cannot share one Input step.
        let mut taken: Vec<(usize, bool)> = self
            .steps
            .iter()
            .enumerate()
            .filter(|(i, _)| !todo.iter().any(|(j, _)| j == i))
            .filter_map(|(_, s)| match s.op {
                Op::Input { input, clip, .. } => Some((input, clip)),
                _ => None,
            })
            .collect();
        for (i, (kind, nth, clip, seconds)) in todo {
            let mut k = loop {
                match self.nth_input(kind, nth) {
                    Some(k) => break k,
                    None => {
                        self.ins.push(kind);
                    }
                }
            };
            if taken.iter().any(|&(t, c)| t == k && c != clip) {
                self.ins.push(kind);
                k = self.ins.len() - 1;
            }
            taken.push((k, clip));
            if legacy && matches!(self.steps[i].op, Op::Second) {
                found.second = Some(k);
            }
            self.steps[i].op = Op::Input { input: k, matte: kind == InKind::Matte, nth, clip, seconds };
        }
        for s in &mut self.steps {
            if matches!(s.op, Op::PromptIn) {
                s.op = Op::Prompt { text: String::new() };
            }
        }
        self.sync_inputs();
        for i in 0..self.steps.len() {
            self.sync_ports(i);
        }
        legacy.then_some(found)
    }

    /// The node input that is the `nth` of its kind.
    pub fn nth_input(&self, kind: InKind, nth: usize) -> Option<usize> {
        self.ins.iter().enumerate().filter(|(_, k)| **k == kind).nth(nth).map(|(i, _)| i)
    }

    /// The one model step of a one-model pipeline (see `for_model`).
    pub fn model_step(&self) -> Option<usize> {
        self.steps.iter().position(|s| matches!(s.op, Op::Generate { .. }))
    }

    /// Append a step, ports empty. Returns its index.
    pub fn push(&mut self, op: Op, at: [f32; 2]) -> usize {
        let n = op.ports().len();
        self.steps.push(Step { op, at, inputs: vec![None; n] });
        self.steps.len() - 1
    }

    /// The Output step, if there is one.
    pub fn output(&self) -> Option<usize> {
        self.steps.iter().position(|s| matches!(s.op, Op::Output))
    }

    /// The take on screen.
    pub fn picked(&self) -> Option<&Take> {
        self.takes.get(self.take)
    }

    /// Wire `from`'s output into `to`'s input `port`. Refuses a pipe
    /// mismatch, a missing port and anything that would close a cycle.
    /// Replaces whatever fed that port.
    pub fn connect(&mut self, from: usize, to: usize, port: usize) -> bool {
        if from == to || from >= self.steps.len() || to >= self.steps.len() {
            return false;
        }
        let Some(out) = self.steps[from].op.output() else { return false };
        if !self.steps[to].op.accepts(port, out) || self.upstream(from).contains(&to) {
            return false;
        }
        self.sync_ports(to);
        self.steps[to].inputs[port] = Some(from);
        true
    }

    pub fn disconnect(&mut self, to: usize, port: usize) {
        if let Some(slot) = self.steps.get_mut(to).and_then(|s| s.inputs.get_mut(port)) {
            *slot = None;
        }
    }

    /// Remove a step, dropping wires into and out of it and sliding every
    /// later index down. Output cannot be removed (the node would have no
    /// picture to point at, and a run nothing to finish on).
    /// An Input cannot either: it IS one of the node's inputs (remove the
    /// input instead, `remove_input`).
    pub fn remove(&mut self, i: usize) -> bool {
        if i >= self.steps.len() || matches!(self.steps[i].op, Op::Output | Op::Input { .. }) {
            return false;
        }
        self.remove_step(i);
        true
    }

    fn remove_step(&mut self, i: usize) {
        self.steps.remove(i);
        for s in &mut self.steps {
            for inp in &mut s.inputs {
                *inp = match *inp {
                    Some(j) if j == i => None,
                    Some(j) if j > i => Some(j - 1),
                    other => other,
                };
            }
        }
    }

    /// Make step `i`'s input list as long as its op's ports, dropping wires
    /// whose pipe no longer matches — after a model change, say, from an
    /// editor that takes a prompt and an image to an upscaler that takes
    /// only the image.
    pub fn sync_ports(&mut self, i: usize) {
        let ports = self.steps[i].op.ports();
        let old = std::mem::take(&mut self.steps[i].inputs);
        let mut new = vec![None; ports.len()];
        for k in 0..ports.len() {
            if let Some(Some(src)) = old.get(k) {
                let out = self.steps.get(*src).and_then(|s| s.op.output());
                if out.is_some_and(|o| self.steps[i].op.accepts(k, o)) {
                    new[k] = Some(*src);
                }
            }
        }
        self.steps[i].inputs = new;
        // The step's OUTPUT pipe may have changed too (a still model to a
        // video one): a consumer that cannot take the new pipe lets go.
        for j in 0..self.steps.len() {
            if self.steps[j].inputs.contains(&Some(i)) {
                self.sync_ports(j);
            }
        }
    }

    /// Re-point step `i` at a different catalog model, keeping any wire
    /// whose port NAME survives the change.
    pub fn set_model(&mut self, i: usize, id: &str) {
        let before: Vec<(&'static str, Option<usize>)> = self.steps[i]
            .op
            .ports()
            .iter()
            .map(|p| p.1)
            .zip(self.steps[i].inputs.iter().copied())
            .collect();
        if let Op::Generate { model, .. } = &mut self.steps[i].op {
            *model = id.to_string();
        }
        let ports = self.steps[i].op.ports();
        let mut new = vec![None; ports.len()];
        for (k, (_, name)) in ports.iter().enumerate() {
            if let Some((_, Some(src))) = before.iter().find(|(n, _)| n == name) {
                let out = self.steps[*src].op.output();
                if out.is_some_and(|o| self.steps[i].op.accepts(k, o)) {
                    new[k] = Some(*src);
                }
            }
        }
        self.steps[i].inputs = new;
        // The step's OUTPUT pipe may have changed too (a still model to a
        // video one): a consumer that cannot take the new pipe lets go.
        for j in 0..self.steps.len() {
            if self.steps[j].inputs.contains(&Some(i)) {
                self.sync_ports(j);
            }
        }
    }

    /// Every step upstream of `i`, including `i`.
    pub fn upstream(&self, i: usize) -> Vec<usize> {
        let mut seen = vec![false; self.steps.len()];
        let mut stack = vec![i];
        let mut out = vec![];
        while let Some(j) = stack.pop() {
            if j >= self.steps.len() || seen[j] {
                continue;
            }
            seen[j] = true;
            out.push(j);
            stack.extend(self.steps[j].inputs.iter().flatten().copied());
        }
        out
    }

    /// The steps a run of `target` executes, dependencies first. `Err`
    /// names the first thing that stops it: an unwired port or an unknown
    /// model — said before anything is sent anywhere.
    pub fn plan(&self, target: usize) -> Result<Vec<usize>, String> {
        let mut order = vec![];
        let mut state = vec![0u8; self.steps.len()]; // 0 new, 1 open, 2 done
        self.visit(target, &mut state, &mut order)?;
        Ok(order)
    }

    fn visit(&self, i: usize, state: &mut [u8], order: &mut Vec<usize>) -> Result<(), String> {
        match state.get(i) {
            None => return Err(format!("step {i} does not exist")),
            Some(2) => return Ok(()),
            Some(1) => return Err("the pipeline has a cycle".into()),
            _ => {}
        }
        state[i] = 1;
        let step = &self.steps[i];
        if let Op::Generate { model, .. } = &step.op {
            if catalog::get(model).is_none() {
                return Err(format!("this build does not know the model \"{model}\""));
            }
        }
        if let Op::Comp { .. } = &step.op {
            let src = step.inputs.first().copied().flatten().and_then(|j| self.steps.get(j));
            if src.is_some_and(|s| !matches!(s.op, Op::Input { .. } | Op::Shape { .. } | Op::Comp { .. })) {
                return Err(format!(
                    "{} works on the node's inputs (or a Spline, or another such node): a model's result cannot feed it yet",
                    step.op.name()
                ));
            }
        }
        let ports = step.op.ports();
        for (k, (_, name)) in ports.iter().enumerate() {
            match step.inputs.get(k).copied().flatten() {
                Some(src) => self.visit(src, state, order)?,
                // Comfy's ports are all optional: a workflow uses the ones
                // it has placeholders for.
                None if matches!(step.op, Op::Comfy { .. } | Op::Template { .. }) => {}
                None => {
                    return Err(format!("{} needs its {name} input wired", step.op.label()))
                }
            }
        }
        state[i] = 2;
        order.push(i);
        Ok(())
    }

    /// Are the models this pipeline would run all free for commercial use?
    /// Unknown models count as not (rule 10: when in doubt, tag).
    pub fn all_commercial(&self) -> bool {
        self.steps.iter().all(|s| match &s.op {
            Op::Generate { model, .. } => catalog::get(model).is_some_and(|m| m.commercial),
            Op::Comfy { commercial, .. } => *commercial,
            _ => true,
        })
    }

    /// The node inputs a run reads — the Input steps on the way to Output,
    /// as (input, clip) — in plan order. An Input nothing uses is not
    /// captured, and its input need not be wired.
    pub fn reads(&self) -> Vec<(usize, bool)> {
        let Some(Ok(order)) = self.output().map(|o| self.plan(o)) else { return vec![] };
        // An Input read only by Comp nodes is rendered as part of THEIR
        // capture (the host renders the chain), never on its own.
        let direct = |i: usize| {
            order.iter().any(|&j| !matches!(self.steps[j].op, Op::Comp { .. }) && self.steps[j].inputs.contains(&Some(i)))
        };
        order
            .iter()
            .filter_map(|&i| match self.steps[i].op {
                Op::Input { input, clip, .. } if direct(i) => Some((input, clip)),
                _ => None,
            })
            .collect()
    }
    /// Is Input step `i` captured on its own by a run (see `reads`)?
    pub fn input_is_read(&self, i: usize) -> bool {
        matches!(self.steps.get(i).map(|s| &s.op), Some(Op::Input { input, .. })
            if self.reads().iter().any(|r| r.0 == *input))
    }
    /// The mattes drawn inside that a run reads — the Shape steps on the
    /// way to Output, as (step, clip) — in plan order.
    pub fn shapes(&self) -> Vec<(usize, bool)> {
        let Some(Ok(order)) = self.output().map(|o| self.plan(o)) else { return vec![] };
        order
            .into_iter()
            .filter_map(|i| match self.steps[i].op {
                Op::Shape { clip, .. } | Op::Comp { clip, .. } => Some((i, clip)),
                _ => None,
            })
            .collect()
    }
    /// Does a run read anything as a clip?
    pub fn reads_clip(&self) -> bool {
        self.reads().iter().any(|r| r.1) || self.shapes().iter().any(|r| r.1)
    }
    /// What node input `k` carries (an image when out of range).
    pub fn in_kind(&self, k: usize) -> InKind {
        self.ins.get(k).copied().unwrap_or(InKind::Image)
    }
    /// Does the node's picture come straight out of an UPSCALER? Then the
    /// result's own size is the node's format: stretched back over the
    /// input's format, an upscale would be a blur with a bill attached.
    pub fn resizes(&self) -> bool {
        self.output()
            .and_then(|o| self.steps[o].inputs.first().copied().flatten())
            .is_some_and(|i| {
                matches!(&self.steps[i].op, Op::Generate { model, .. }
                    if catalog::get(model).is_some_and(|m| m.task.resizes()))
            })
    }
    /// The most megapixels of the host's image a run should send: `usual`,
    /// or [`UPSCALE_REF_MP`] when an image upscaler is among the steps. An
    /// upscaler fed a shrunken plate hands back the plate it started with.
    pub fn reference_mp(&self, usual: f32) -> f32 {
        let up = self.steps.iter().any(|s| {
            matches!(&s.op, Op::Generate { model, .. }
                if catalog::get(model).is_some_and(|m| m.task == Task::Upscale))
        });
        if up {
            usual.max(UPSCALE_REF_MP)
        } else {
            usual
        }
    }
    /// Set every clip Input's length (they always match). Returns whether
    /// anything changed.
    pub fn set_clip_seconds(&mut self, v: f32) -> bool {
        let mut changed = false;
        for s in &mut self.steps {
            if let Op::Input { clip: true, seconds, .. } = &mut s.op {
                changed |= *seconds != v;
                *seconds = v;
            }
        }
        changed
    }

    /// How long a run's clips are: the first clip Input's length (every
    /// clip a run sends covers the same frames, so they line up frame for
    /// frame), else the default.
    pub fn clip_seconds(&self) -> f32 {
        self.steps
            .iter()
            .find_map(|s| match s.op {
                Op::Input { clip: true, seconds, .. } => Some(seconds),
                _ => None,
            })
            .unwrap_or(default_clip_seconds())
            .clamp(0.1, 60.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_pipeline_plans_prompt_then_generate_then_output() {
        let p = Pipeline::default();
        let out = p.output().unwrap();
        let plan = p.plan(out).unwrap();
        assert_eq!(plan.len(), 3);
        assert!(matches!(p.steps[plan[0]].op, Op::Prompt { .. }));
        assert!(matches!(p.steps[plan[2]].op, Op::Output));
    }

    #[test]
    fn every_model_has_a_one_model_pipeline_with_every_port_fed() {
        for m in catalog::CATALOG.iter().filter(|m| m.task.is_text()) {
            assert!(Pipeline::for_model(m.id).is_none(), "{}: text out is a step, not a node", m.id);
        }
        for m in catalog::CATALOG.iter().filter(|m| !m.task.is_text()) {
            let p = Pipeline::for_model(m.id).unwrap();
            let out = p.output().unwrap();
            let plan = p.plan(out).unwrap_or_else(|e| panic!("{}: {e}", m.id));
            let g = p.model_step().unwrap();
            assert!(p.steps[g].inputs.iter().all(|i| i.is_some()), "{}", m.id);
            let ports = task_ports(m.task);
            // Every step is on the way out, but the first image when the
            // model reads no picture (it is still the node's input).
            let idle = usize::from(!ports.iter().any(|&(k, _)| k != Pipe::Text));
            assert_eq!(plan.len() + idle, p.steps.len(), "{}: every step is on the way out", m.id);
            let reads = p.reads();
            let has = |matte: bool, clip: bool| reads.iter().any(|&(k, c)| (p.ins[k] == InKind::Matte) == matte && c == clip);
            assert_eq!(has(true, false), ports.iter().any(|&(k, n)| k == Pipe::Image && n == "mask"), "{}", m.id);
            assert_eq!(has(true, true), ports.iter().any(|&(k, n)| k == Pipe::Video && n == "mask"), "{}", m.id);
            assert_eq!(has(false, true), ports.iter().any(|&(k, n)| k == Pipe::Video && n != "mask"), "{}", m.id);
            assert_eq!(has(false, false), ports.iter().any(|&(k, n)| k == Pipe::Image && n != "mask"), "{}", m.id);
            assert_eq!(p.ins.iter().filter(|k| **k == InKind::Image).count() >= 2, ports.iter().any(|&(_, n)| n == "reference"), "{}", m.id);
            assert_eq!(p.ins[0], InKind::Image, "{}: the first input is the image passed through", m.id);
        }
        assert!(Pipeline::for_model("no/such-model").is_none());
    }

    #[test]
    fn connect_refuses_a_pipe_mismatch_and_a_cycle() {
        let mut p = Pipeline::default();
        let prompt = 0;
        let gen = 1;
        let out = p.output().unwrap();
        // Text into Output's image port.
        assert!(!p.connect(prompt, out, 0));
        // An editor downstream of gen, then gen fed from it: a cycle.
        let ed = p.push(
            Op::Generate { model: "fal/flux-kontext-pro".into(), seed: 1, opts: Opts::default() },
            [0.5, 0.0],
        );
        assert!(p.connect(gen, ed, 1));
        assert!(p.connect(prompt, ed, 0));
        let up = p.push(
            Op::Generate { model: "fal/esrgan".into(), seed: 1, opts: Opts::default() },
            [0.5, 0.5],
        );
        assert!(p.connect(ed, up, 0));
        // gen has only a text port, so try the image editor's image port
        // fed from its own downstream upscaler.
        assert!(!p.connect(up, ed, 1));
    }

    #[test]
    fn an_unwired_port_is_named_before_anything_runs() {
        let mut p = Pipeline::default();
        p.disconnect(1, 0);
        let err = p.plan(p.output().unwrap()).unwrap_err();
        assert!(err.contains("prompt"), "{err}");
    }

    #[test]
    fn remove_slides_later_indices_and_drops_wires_into_the_removed_step() {
        let mut p = Pipeline::default();
        let r = p.push(Op::Reference, [0.0, -0.8]);
        let ed = p.push(
            Op::Generate { model: "fal/flux-kontext-pro".into(), seed: 1, opts: Opts::default() },
            [0.5, 0.0],
        );
        assert!(p.connect(0, ed, 0));
        assert!(p.connect(r, ed, 1));
        assert!(p.remove(1)); // the text-to-image step
        // Output lost its feed; the editor's refs slid down by one.
        let out = p.output().unwrap();
        assert_eq!(p.steps[out].inputs, vec![None]);
        let ed = ed - 1;
        assert_eq!(p.steps[ed].inputs, vec![Some(0), Some(r - 1)]);
        assert!(!p.remove(out), "Output is not removable");
    }

    #[test]
    fn a_model_change_keeps_wires_whose_port_names_survive() {
        let mut p = Pipeline::default();
        let r = p.push(Op::Reference, [0.0, -0.8]);
        let ed = p.push(
            Op::Generate { model: "fal/flux-kontext-pro".into(), seed: 1, opts: Opts::default() },
            [0.5, 0.0],
        );
        assert!(p.connect(0, ed, 0));
        assert!(p.connect(r, ed, 1));
        p.set_model(ed, "fal/esrgan");
        assert_eq!(p.steps[ed].inputs, vec![Some(r)], "image stays on image, prompt goes");
    }

    #[test]
    fn a_video_feeds_output_and_nothing_else() {
        let mut p = Pipeline::default();
        let r = p.push(Op::Reference, [0.0, -0.8]);
        let up = p.push(
            Op::Generate { model: "fal/esrgan".into(), seed: 1, opts: Opts::default() },
            [0.5, 0.5],
        );
        assert!(p.connect(1, up, 0));
        // The text-to-image step becomes a video one: it keeps its prompt,
        // the upscaler drops it, Output keeps it.
        p.set_model(1, "fal/seedance-2-fast-i2v");
        assert_eq!(p.steps[1].op.output(), Some(Pipe::Video));
        assert_eq!(p.steps[up].inputs, vec![None]);
        let out = p.output().unwrap();
        assert_eq!(p.steps[out].inputs, vec![Some(1)]);
        assert!(!p.connect(1, up, 0));
        assert!(p.connect(r, 1, 1));
        assert!(p.plan(out).is_ok());
    }

    #[test]
    fn a_take_is_a_video_by_its_file() {
        let t = |f: &str| Take {
            key: "k".into(),
            file: f.into(),
            tick: 0,
            commercial: true,
            usd: None,
            label: String::new(),
            when: 0,
            data: false,
        };
        assert!(t("result.mp4").is_video());
        assert!(!t("result.png").is_video());
    }

    #[test]
    fn licence_is_the_catalog_s_and_comfy_is_cautious_by_default() {
        let mut p = Pipeline::default();
        assert!(p.all_commercial());
        p.set_model(1, "fal/flux-dev");
        assert!(!p.all_commercial());
        p.set_model(1, "fal/flux-schnell");
        p.push(Op::Comfy { workflow: String::new(), seed: 1, commercial: false }, [0.0, 0.0]);
        assert!(!p.all_commercial());
    }

    #[test]
    fn serde_round_trips_and_old_steps_default_their_opts() {
        let p = Pipeline::default();
        let j = serde_json::to_string(&p).unwrap();
        let back: Pipeline = serde_json::from_str(&j).unwrap();
        assert_eq!(p, back);
        let s: Step = serde_json::from_str(
            r#"{"op":{"op":"generate","model":"fal/esrgan","seed":3},"at":[0,0]}"#,
        )
        .unwrap();
        match s.op {
            Op::Generate { opts, .. } => assert_eq!(opts, Opts::default()),
            _ => panic!(),
        }
    }

    #[test]
    fn text_steps_feed_prompts_and_a_template_s_ports_are_optional() {
        // image -> describe -> template -> schnell -> out
        let mut p = Pipeline::bare(vec![InKind::Image]);
        let img = p.push(Op::Reference, [0.0, 0.0]);
        let ask = p.push(Op::Prompt { text: String::new() }, [0.0, 0.0]);
        let d = p.push(Op::Generate { model: "fal/describe".into(), seed: 1, opts: Opts::default() }, [0.0, 0.0]);
        let t = p.push(Op::Template { text: "{a}, as a watercolour".into() }, [0.0, 0.0]);
        let g = p.push(Op::Generate { model: "fal/flux-schnell".into(), seed: 1, opts: Opts::default() }, [0.0, 0.0]);
        let out = p.push(Op::Output, [0.0, 0.0]);
        assert_eq!(p.steps[d].op.output(), Some(Pipe::Text));
        assert!(p.connect(ask, d, 0) && p.connect(img, d, 1));
        assert!(p.connect(d, t, 0), "a description is text");
        assert!(p.connect(t, g, 0) && p.connect(g, out, 0));
        // Port b is unwired, and that is fine.
        assert_eq!(p.plan(out).unwrap().len(), 6);
        assert!(!p.connect(d, out, 0), "words are not a picture");
        assert!(p.all_commercial());
    }

    #[test]
    fn a_3d_step_feeds_output_and_nothing_else() {
        let mut p = Pipeline::for_model("fal/sam-3d-objects").unwrap();
        let g = p.model_step().unwrap();
        assert_eq!(p.steps[g].op.output(), Some(Pipe::Scene));
        assert!(p.plan(p.output().unwrap()).is_ok(), "wired into Output by for_model");
        let up = p.push(Op::Generate { model: "fal/esrgan".into(), seed: 1, opts: Opts::default() }, [0.0, 0.0]);
        assert!(!p.connect(g, up, 0), "a splat is not a picture");
        let t = Take {
            key: "k".into(), file: "result.ply".into(), tick: 0, commercial: true,
            usd: None, label: String::new(), when: 0, data: false,
        };
        assert!(t.is_splat() && !t.is_video());
    }

    #[test]
    fn only_a_pipeline_ending_in_an_upscaler_resizes_the_node() {
        assert!(!Pipeline::default().resizes());
        assert!(!Pipeline::for_model("fal/flux-kontext-pro").unwrap().resizes());
        for id in ["fal/esrgan", "fal/topaz-upscale", "fal/topaz-video-upscale"] {
            assert!(Pipeline::for_model(id).unwrap().resizes(), "{id}");
        }
        // An upscaler in the MIDDLE feeds a model that picks its own size.
        let mut p = Pipeline::for_model("fal/esrgan").unwrap();
        let up = p.model_step().unwrap();
        let out = p.output().unwrap();
        let cut = p.push(Op::Generate { model: "fal/birefnet-v2".into(), seed: 1, opts: Opts::default() }, [0.0, 0.3]);
        assert!(p.connect(up, cut, 0) && p.connect(cut, out, 0));
        assert!(!p.resizes());
        // But its reference still goes up whole.
        assert_eq!(p.reference_mp(2.0), UPSCALE_REF_MP);
        assert_eq!(Pipeline::default().reference_mp(2.0), 2.0);
        // A clip is sized by the clip capture, not by this.
        assert_eq!(Pipeline::for_model("fal/topaz-video-upscale").unwrap().reference_mp(2.0), 2.0);
    }

    #[test]
    fn clip_steps_feed_video_ports_and_old_files_still_load() {
        let mut p = Pipeline::bare(vec![InKind::Image, InKind::Image]);
        let c = p.input_step(0).unwrap();
        if let Op::Input { clip, seconds, .. } = &mut p.steps[c].op {
            (*clip, *seconds) = (true, 3.0);
        }
        let r = p.input_step(1).unwrap();
        let g = p.push(
            Op::Generate { model: "fal/kling-o3-pro-video-edit".into(), seed: 1, opts: Opts::default() },
            [0.0, 0.0],
        );
        assert!(!p.connect(r, g, 1), "a still is not a clip");
        assert!(p.connect(c, g, 1));
        assert_eq!(p.clip_seconds(), 3.0);
        // A step saved before clips existed reads as before; a clip's
        // length defaults when it is missing.
        let old: Op = serde_json::from_str(r#"{"op":"reference"}"#).unwrap();
        assert_eq!(old, Op::Reference);
        let c: Op = serde_json::from_str(r#"{"op":"clip_in"}"#).unwrap();
        assert_eq!(c, Op::ClipIn { seconds: 5.0 });
        let m: Op = serde_json::from_str(r#"{"op":"matte_clip"}"#).unwrap();
        assert_eq!(m.output(), Some(Pipe::Video));
    }

    #[test]
    fn every_input_is_one_input_step_and_adding_one_renumbers_nothing() {
        let mut p = Pipeline::default();
        assert_eq!(p.ins, vec![InKind::Image, InKind::Matte], "one image, one matte");
        let img = p.input_step(0).unwrap();
        let ed = p.push(Op::Generate { model: "fal/flux-kontext-pro".into(), seed: 1, opts: Opts::default() }, [0.0, 0.0]);
        assert!(p.connect(img, ed, 1));
        let k = p.add_input(InKind::Image);
        assert_eq!(k, 2);
        assert_eq!(p.steps[p.input_step(2).unwrap()].op.label(), "image 2");
        assert_eq!(p.steps[ed].inputs[1], Some(img), "nothing renumbered");
        // In an opened pipeline (graph units) a new Input lands beside
        // the others, not on the middle of the graph.
        let mut q = Pipeline::default();
        for s in &mut q.steps {
            s.at = [s.at[0] * 260.0, s.at[1] * 260.0];
        }
        let k = q.add_input(InKind::Image);
        let new = q.steps[q.input_step(k).unwrap()].at;
        assert!(q.steps.iter().filter(|s| s.at != new).all(|s| (s.at[0] - new[0]).abs() > 100.0 || (s.at[1] - new[1]).abs() > 100.0), "{new:?}");
        // An Input is the node's: not removable as a step.
        assert!(!p.remove(img));
        // A pasted duplicate folds into the first.
        let dup = p.steps[img].clone();
        p.steps.push(dup);
        let d = p.steps.len() - 1;
        let out = p.output().unwrap();
        p.steps[out].inputs[0] = Some(d);
        p.sync_inputs();
        assert_eq!(p.steps.iter().filter(|s| matches!(s.op, Op::Input { input: 0, .. })).count(), 1);
        // The last image cannot go; the matte can, and the input after it
        // slides down.
        assert!(p.remove_input(1));
        assert_eq!(p.ins, vec![InKind::Image, InKind::Image]);
        assert!(matches!(p.steps[p.input_step(1).unwrap()].op, Op::Input { nth: 1, matte: false, .. }));
        assert!(p.remove_input(1));
        assert!(!p.remove_input(0), "the image passed through stays");
    }

    #[test]
    fn a_pipeline_from_before_the_input_bar_normalizes() {
        // The old fixed inputs: image 0, matte 1, prompt 2, second 3.
        let old = r#"{"steps":[
            {"op":{"op":"prompt_in"},"at":[0,0]},
            {"op":{"op":"reference"},"at":[0,0]},
            {"op":{"op":"second"},"at":[0,0]},
            {"op":{"op":"generate","model":"fal/nano-banana-pro-edit-ref","seed":1},"at":[0,0],"inputs":[0,1,2]},
            {"op":{"op":"output"},"at":[0,0],"inputs":[3]}]}"#;
        let mut p: Pipeline = serde_json::from_str(old).unwrap();
        assert!(p.ins.is_empty());
        let l = p.normalize().expect("legacy");
        assert_eq!(p.ins, vec![InKind::Image, InKind::Matte, InKind::Image]);
        assert_eq!(l.second, Some(2));
        assert!(matches!(p.steps[0].op, Op::Prompt { .. }), "no prompt input outside");
        assert!(matches!(p.steps[1].op, Op::Input { input: 0, .. }));
        assert!(matches!(p.steps[2].op, Op::Input { input: 2, nth: 1, .. }));
        assert!(p.input_step(1).is_some(), "the matte has its Input too");
        assert_eq!(p.reads(), vec![(0, false), (2, false)]);
        assert_eq!(p.normalize(), None, "idempotent");
        // A still and a clip of the same old input get an input each.
        let mut q = Pipeline::bare(vec![]);
        q.push(Op::Reference, [0.0, 0.0]);
        q.push(Op::ClipIn { seconds: 2.0 }, [0.0, 0.0]);
        q.push(Op::MatteClip, [0.0, 0.0]);
        q.ins.clear();
        q.normalize();
        assert_eq!(q.ins, vec![InKind::Image, InKind::Matte, InKind::Image]);
        assert_eq!(q.clip_seconds(), 2.0);
    }
}
